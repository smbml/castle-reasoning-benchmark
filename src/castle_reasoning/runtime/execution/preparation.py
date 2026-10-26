"""Runtime execution preparation."""

import copy
import dataclasses
from typing import TYPE_CHECKING

import numpy as np

import castle_reasoning.runtime.diagnostics.timing as timing

if TYPE_CHECKING:
    import castle_reasoning.runtime.control.batch_tracking as batch_tracking


@dataclasses.dataclass(frozen=True)
class Source:
    slot: int
    slot_epoch: int
    source_revision: int
    request_sequence: int
    model_id: str

    def completed(self):
        return dataclasses.replace(
            self,
            source_revision=self.source_revision + 1,
            request_sequence=self.request_sequence + 1,
        )


@dataclasses.dataclass(frozen=True)
class PreparedRequest:
    source: Source
    initial_qpos: np.ndarray
    arrays: "batch_tracking.Prepared"


def _freeze(value):
    if isinstance(value, np.ndarray):
        value.flags.writeable = False
    elif dataclasses.is_dataclass(value):
        for field in dataclasses.fields(value):
            _freeze(getattr(value, field.name))
    elif isinstance(value, (tuple, list)):
        for item in value:
            _freeze(item)


def prepare(
    request, initial, source, *, permits=None, cancelled=None, location="producer"
):
    """Copy request inputs, compute unchanged native arrays, then release permit."""
    from castle_reasoning.runtime.control import batch_tracking

    acquired = False
    try:
        with timing.span("prepare.permit", slot=source.slot, location=location):
            while True:
                if cancelled is not None and cancelled.is_set():
                    raise RuntimeError("Motion preparation cancelled")
                if permits is None or permits.acquire(timeout=0.05):
                    acquired = permits is not None
                    break
        with timing.span(
            "prepare.slot",
            slot=source.slot,
            location=location,
            slot_epoch=source.slot_epoch,
            request=source.request_sequence,
            segment_kind=request.segment.name,
            paused=False,
            literal=request.actions is not None,
        ):
            initial = np.array(initial, copy=True)
            request = dataclasses.replace(
                request,
                segment=copy.deepcopy(request.segment),
                scene_reference=copy.deepcopy(request.scene_reference),
                actions=copy.deepcopy(request.actions),
                efforts=copy.deepcopy(request.efforts),
                preparation=None,
            )
            arrays = batch_tracking.prepare([request], initial)
            request.preparation = PreparedRequest(source, initial, arrays)
            for value in (
                request.segment,
                request.scene_reference,
                request.actions,
                request.efforts,
                request.preparation,
            ):
                _freeze(value)
            timing.mark(
                "prepare.complete",
                reference_count=len(arrays.references),
                bytes=sum(
                    getattr(arrays, f.name).nbytes for f in dataclasses.fields(arrays)
                ),
            )
            return request
    finally:
        if acquired:
            permits.release()


@timing.measured("prepare.validate")
def validate(request, initial, source):
    """Reject stale transport or malformed arrays before a lane becomes active."""
    value = request.preparation
    if value is None or value.source != source:
        raise ValueError(
            f"Preparation source mismatch: expected {source}, "
            f"received {None if value is None else value.source}"
        )
    if value.initial_qpos.shape != initial.shape or not np.array_equal(
        value.initial_qpos, initial
    ):
        difference = (
            float(np.max(np.abs(value.initial_qpos - initial)))
            if value.initial_qpos.shape == initial.shape
            else None
        )
        raise ValueError(
            f"Preparation initial state mismatch: max_difference={difference}"
        )
    arrays = value.arrays
    model = request.scene.model
    count = len(arrays.references)
    if not count:
        raise ValueError("Empty prepared reference")
    shapes = {
        "references": (count, 1, 7),
        "bias": (count, 1, 6),
        "lengths": (1,),
        "contacts": (1, model.ngeom, model.ngeom),
        "scene_references": (1, model.nq),
        "parts": (1,),
        "guards": (1,),
        "monitors": (1,),
        "offsets": (1, 3),
        "rotations": (1, 3, 3),
        "attachments": (1,),
        "efforts": (count, 1),
        "literal_mask": (1,),
        "literal_actions": (count, 1, 7),
    }
    integers = {"lengths", "parts", "guards", "monitors", "attachments", "literal_mask"}
    for name, shape in shapes.items():
        array = getattr(arrays, name)
        dtype = np.int32 if name in integers else np.float32
        if array.shape != shape or array.dtype != dtype or not np.isfinite(array).all():
            raise ValueError(f"Invalid prepared {name}: {array.shape}, {array.dtype}")
    if arrays.lengths[0] != count or not np.all(arrays.efforts > 0):
        raise ValueError("Invalid prepared length or effort")
    # IPC serialization does not preserve NumPy write protection.
    for item in (value, request.segment, request.scene_reference,
                 request.actions, request.efforts):
        _freeze(item)
    return arrays
