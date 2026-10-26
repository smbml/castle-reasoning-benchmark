"""Runtime execution segments."""

import dataclasses

import numpy as np

import castle_reasoning.runtime.control.batch_tracking as batch_tracking
import castle_reasoning.runtime.diagnostics.timing as timing
import castle_reasoning.runtime.execution.preparation as preparation


def inactive_prepared(nq, ngeom):
    """Finite placeholders; valid only with the world's execution mask disabled."""
    return batch_tracking.Prepared(
        references=np.zeros((1, 1, 7), np.float32),
        bias=np.zeros((1, 1, 6), np.float32),
        lengths=np.ones(1, np.int32),
        contacts=np.full((1, ngeom, ngeom), -1e6, np.float32),
        scene_references=np.zeros((1, nq), np.float32),
        parts=np.full(1, -1, np.int32),
        guards=np.zeros(1, np.int32), monitors=np.zeros(1, np.int32),
        offsets=np.zeros((1, 3), np.float32),
        rotations=np.eye(3, dtype=np.float32)[None],
        attachments=np.zeros(1, np.int32),
        efforts=np.ones((1, 1), np.float32),
        literal_mask=np.zeros(1, np.int32),
        literal_actions=np.zeros((1, 1, 7), np.float32),
    )


@dataclasses.dataclass
class Pending:
    """A complete reference and its contiguous, actually simulated prefix."""

    request: batch_tracking.Request
    prepared: batch_tracking.Prepared
    offset: int = 0
    traces: list = dataclasses.field(default_factory=list)
    events: list = dataclasses.field(default_factory=list)

    @property
    def waiting(self):
        return self.request.segment.name == "batch_wait"

    @property
    def remaining(self):
        return len(self.prepared.references) - self.offset

    def record(self, trace, event):
        event = dict(event)
        event["device_contact_violations"] = [
            {**hit, "control_index": hit["control_index"] + self.offset}
            for hit in event["device_contact_violations"]
        ]
        self.traces.append(trace)
        self.events.append(event)
        self.offset += event["executed_control_steps"]

    def result(self):
        trace = {
            key: np.concatenate([chunk[key] for chunk in self.traces])
            for key in self.traces[0]
        }
        event = dict(self.events[-1])
        for key in (
            "simulator_seconds",
            "executed_control_steps",
            "control_steps",
            "padding_control_steps",
            "discarded_suffix_steps",
        ):
            event[key] = sum(chunk[key] for chunk in self.events)
        for key in ("batch_lane_control_steps", "paused_control_steps"):
            event[key] = sum(chunk.get(key, 0) for chunk in self.events)
        event["requested_steps"] = len(self.prepared.references)
        event["device_contact_violations"] = [
            hit for chunk in self.events for hit in chunk["device_contact_violations"]
        ]
        return trace, event


@timing.measured('prepare.chunk')
def prepare_chunk(pending, count):
    """Slice full-segment feedforward; only finished-stage holds may be extended."""
    values = {}
    for field in dataclasses.fields(batch_tracking.Prepared):
        if field.name == "lengths":
            values[field.name] = np.full(len(pending), count, np.int32)
        elif field.name in ("references", "bias", "literal_actions", "efforts"):
            pieces = []
            for state in pending:
                source = getattr(state.prepared, field.name)
                indices = np.arange(state.offset, state.offset + count)
                if state.waiting:
                    indices = np.minimum(indices, len(source) - 1)
                pieces.append(source[indices])
            values[field.name] = np.concatenate(pieces, axis=1)
        else:
            values[field.name] = np.concatenate(
                [getattr(state.prepared, field.name) for state in pending]
            )
    return batch_tracking.Prepared(**values)


class SegmentExecutor:
    """Keep unfinished paths while the coordinator requests newly available work.

    Physics pauses globally by default, or per world in opt-in mode.
    No state or servo reset occurs at a cut;
    complete references and safety expectations persist across chunks.
    """

    def __init__(self, controller, deadline=float("inf"), *, pause_planning_worlds=False,
                 skip_paused_preparation=False, preparation_sources=None):
        self.controller, self.deadline = controller, deadline
        self.pause_planning_worlds = pause_planning_worlds
        if pause_planning_worlds and not hasattr(controller, "active_worlds"):
            raise ValueError("Paused planning requires an active-world controller")
        if skip_paused_preparation and not pause_planning_worlds:
            raise ValueError("Skipping preparation requires paused worlds")
        self.skip_paused_preparation = skip_paused_preparation
        if preparation_sources is not None and not skip_paused_preparation:
            raise ValueError("Producer preparation requires skipped paused preparation")
        self.preparation_sources = preparation_sources
        self.inactive = {}
        self.pending = None
        self.last_results = None

    @timing.measured('execution.chunk')
    def advance(self, requests):
        if self.pending is None:
            self.pending = [None] * len(requests)
        if len(requests) != len(self.pending):
            raise ValueError("World count changed during a stage")
        initial = self.controller.engine.snapshot()["qpos"]
        for world, request in enumerate(requests):
            state = self.pending[world]
            if state is None:
                if self.skip_paused_preparation and request.segment.name == "batch_wait":
                    model = request.scene.model
                    shape = (model.nq, model.ngeom)
                    if shape not in self.inactive:
                        self.inactive[shape] = inactive_prepared(*shape)
                    self.pending[world] = Pending(request, self.inactive[shape])
                    continue
                if self.preparation_sources is not None:
                    arrays = preparation.validate(
                        request, initial[world:world + 1], self.preparation_sources[world]
                    )
                    self.pending[world] = Pending(request, arrays)
                    continue
                with timing.span(
                    "prepare.slot", slot=world, location="owner",
                    segment_kind=request.segment.name,
                    paused=request.segment.name == "batch_wait",
                    literal=request.actions is not None,
                ):
                    prepared = batch_tracking.prepare([request], initial[world : world + 1])
                    timing.mark("prepare.complete", reference_count=len(prepared.references),
                                bytes=sum(getattr(prepared, f.name).nbytes
                                          for f in dataclasses.fields(prepared)))
                self.pending[world] = Pending(request, prepared)
            elif state.request is not request:
                raise ValueError("Cannot replace an unfinished segment")
        active = [state.remaining for state in self.pending if not state.waiting]
        if not active:
            raise ValueError("At least one active segment is required")
        count = min(min(active), 64) if self.pause_planning_worlds else min(active)
        prepared = prepare_chunk(self.pending, count)
        options = {}
        if self.pause_planning_worlds:
            options["active_worlds"] = np.array([not s.waiting for s in self.pending])
        self.last_results = batch_tracking.execute_prepared(
            self.controller, requests, prepared, self.deadline, stop_on_failure=True, **options
        )
        completed = []
        for world, (trace, event) in enumerate(self.last_results):
            state = self.pending[world]
            if state.waiting and self.pause_planning_worlds:
                completed.append((trace, event))
                self.pending[world] = None
                continue
            if state.waiting:
                event["padding_control_steps"] = max(
                    0, event["control_steps"] - state.remaining
                )
            state.record(trace, event)
            if state.waiting or state.remaining <= 0 or event["failure"]:
                completed.append(state.result())
                self.pending[world] = None
                if self.preparation_sources is not None and not state.waiting:
                    self.preparation_sources[world] = self.preparation_sources[world].completed()
            else:
                completed.append(None)
        return completed
