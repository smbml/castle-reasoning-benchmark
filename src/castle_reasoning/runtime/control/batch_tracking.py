"""Runtime control batch_tracking."""

import dataclasses
import time
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

import castle_reasoning.runtime.control.guards as control_guards
import castle_reasoning.runtime.control.tracking as tracking
import castle_reasoning.runtime.diagnostics.timing as timing
import castle_reasoning.runtime.execution.traces as traces
import castle_reasoning.runtime.planning.collision as collision
import castle_reasoning.runtime.planning.manipulation as manipulation
import castle_reasoning.runtime.simulation.engine as simulation_engine
import castle_reasoning.runtime.simulation.world_slots as world_slots

if TYPE_CHECKING:
    pass


@wp.kernel
def follow_reference(
    active: wp.array(dtype=bool),
    reference: wp.array3d(dtype=float),
    step: wp.array(dtype=int),
    qpos: wp.array2d(dtype=float),
    qvel: wp.array2d(dtype=float),
    arm_target: wp.array2d(dtype=float),
    scene: wp.array2d(dtype=float),
    bias: wp.array3d(dtype=float),
    selected: wp.array(dtype=int),
    guard_enabled: wp.array(dtype=int),
    failure: wp.array(dtype=int),
    action: wp.array2d(dtype=float),
    literal_mask: wp.array(dtype=int),
    literal_actions: wp.array3d(dtype=float),
):
    world = wp.tid()
    if not active[world]:
        return
    if literal_mask[world]:
        for joint in range(7):
            action[world, joint] = literal_actions[step[0], world, joint]
        return
    if failure[world] == 0:
        action[world, 6] = reference[step[0], world, 6]
        if guard_enabled[world]:
            for part in range((qpos.shape[1] - 14) // 7):
                p = 14 + part * 7
                v = 14 + part * 6
                omega = wp.length(
                    wp.vec3(qvel[world, v + 3], qvel[world, v + 4], qvel[world, v + 5])
                )
                if omega > 20.0:
                    failure[world] = 2
                if part != selected[world]:
                    distance = wp.length(
                        wp.vec3(
                            qpos[world, p] - scene[world, p],
                            qpos[world, p + 1] - scene[world, p + 1],
                            qpos[world, p + 2] - scene[world, p + 2],
                        )
                    )
                    dot = float(0.0)  # noqa: UP018 - Warp dynamic loop accumulator.
                    for j in range(4):
                        dot += qpos[world, p + 3 + j] * scene[world, p + 3 + j]
                    if distance > 0.003 or wp.abs(dot) < 0.999390827:
                        failure[world] = 2
        for joint in range(6):
            if wp.abs(qpos[world, joint] - reference[step[0], world, joint]) > 0.08:
                failure[world] = 1
    for joint in range(6):
        if failure[world] != 0:
            action[world, joint] = 0.0
        else:
            action[world, joint] = wp.clamp(
                (
                    reference[step[0], world, joint]
                    + bias[step[0], world, joint]
                    - arm_target[world, joint]
                )
                / (0.02 * wp.pi),
                -0.15,
                0.15,
            )


@wp.kernel
def contact_guard(
    count: wp.array(dtype=int),
    pairs: wp.array(dtype=wp.vec2i),
    worldid: wp.array(dtype=int),
    distance: wp.array(dtype=float),
    allowed_distance: wp.array3d(dtype=float),
    failure: wp.array(dtype=int),
    step: wp.array(dtype=int),
    first_step: wp.array(dtype=int),
    first_distances: wp.array3d(dtype=float),
):
    i = wp.tid()
    if i < count[0]:
        pair = pairs[i]
        if pair[0] >= 0 and pair[1] >= 0:  # noqa: SIM102 - guard array indexing.
            if distance[i] < allowed_distance[worldid[i], pair[0], pair[1]]:
                wp.atomic_max(failure, worldid[i], 4)
                previous_step = wp.atomic_min(first_step, worldid[i], step[0])
                if previous_step >= step[0]:
                    wp.atomic_min(
                        first_distances, worldid[i], pair[0], pair[1], distance[i]
                    )


@wp.kernel
def attachment_guard(
    qpos: wp.array2d(dtype=float),
    positions: wp.array2d(dtype=wp.vec3),
    rotations: wp.array2d(dtype=wp.mat33),
    parts: wp.array(dtype=int),
    site: int,
    offsets: wp.array(dtype=wp.vec3),
    rotations_expected: wp.array(dtype=wp.mat33),
    enabled: wp.array(dtype=int),
    failure: wp.array(dtype=int),
):
    world = wp.tid()
    if not enabled[world]:
        return
    part = parts[world]
    offset = offsets[world]
    rotation = rotations_expected[world]
    p = 14 + 7 * part
    center = wp.vec3(qpos[world, p], qpos[world, p + 1], qpos[world, p + 2])
    relative = wp.transpose(rotations[world, site]) * (center - positions[world, site])
    quat = wp.quat(
        qpos[world, p + 4], qpos[world, p + 5], qpos[world, p + 6], qpos[world, p + 3]
    )
    actual = wp.transpose(rotations[world, site]) * wp.quat_to_matrix(quat)
    error = float(0.0)  # noqa: UP018 - Warp dynamic loop accumulator.
    for i in range(3):
        for j in range(3):
            error += (actual[i, j] - rotation[i, j]) ** 2.0
    if wp.length(relative - offset) > 0.006 or error > 0.05:
        wp.atomic_max(failure, world, 3)


@wp.kernel
def apply_grip_effort(
    efforts: wp.array(dtype=float), limits: wp.array2d(dtype=wp.vec2)
):
    world = wp.tid()
    limits[world, 6] = wp.vec2(-efforts[world], efforts[world])


@dataclasses.dataclass
class Request:
    """One world's independently planned segment and safety expectations."""

    segment: manipulation.Segment
    scene: collision.Scene
    part: int
    scene_reference: np.ndarray | None = None
    guard: bool = True
    monitor_scene: bool = True
    grip_effort: float = 1.0
    actions: np.ndarray | None = None
    efforts: np.ndarray | None = None
    preparation: "preparation.PreparedRequest | None" = None


@dataclasses.dataclass
class Prepared:
    references: np.ndarray
    bias: np.ndarray
    lengths: np.ndarray
    contacts: np.ndarray
    scene_references: np.ndarray
    parts: np.ndarray
    guards: np.ndarray
    monitors: np.ndarray
    offsets: np.ndarray
    rotations: np.ndarray
    attachments: np.ndarray
    efforts: np.ndarray
    literal_mask: np.ndarray
    literal_actions: np.ndarray


@timing.measured('prepare.reference')
def prepare(requests, initial):
    """Retain serial reference math, adding only world axes and terminal holds."""
    if not requests or len(requests) != len(initial):
        raise ValueError("One nonempty request per world is required.")
    shapes = {
        (r.scene.model.nq, r.scene.model.nv, r.scene.model.ngeom) for r in requests
    }
    if len(shapes) != 1:
        raise ValueError("All worlds must share the same model dimensions.")
    efforts = np.array([r.grip_effort for r in requests], np.float32)
    if not np.all(np.isfinite(efforts) & (efforts > 0)):
        raise ValueError("Each world needs a positive finite grip effort.")
    references, bias, commands, forces = [], [], [], []
    for request, qpos in zip(requests, initial, strict=True):
        if request.actions is None:
            refs = tracking.references(request.segment)
            bias.append(tracking.servo_offsets(request.scene.model, refs, qpos))
            commands.append(np.zeros_like(refs))
            forces.append(np.full(len(refs), request.grip_effort, np.float32))
        else:
            actions = np.asarray(request.actions, np.float32)
            effort = np.asarray(request.efforts, np.float32)
            if (
                request.guard
                or request.monitor_scene
                or request.segment.payload is not None
            ):
                raise ValueError(
                    "Literal replay requires an unguarded request; audit the saved trace"
                )
            if (
                actions.ndim != 2
                or actions.shape[1] != 7
                or not len(actions)
                or not np.isfinite(actions).all()
            ):
                raise ValueError(
                    "Literal replay needs finite nonempty [time, 7] actions"
                )
            if effort.shape != (len(actions),) or not np.all(
                np.isfinite(effort) & (effort > 0)
            ):
                raise ValueError(
                    "Literal replay needs positive effort for every action"
                )
            refs = np.tile(np.r_[qpos[:6], actions[0, 6]], (len(actions), 1))
            bias.append(np.zeros((len(actions), 6), np.float32))
            commands.append(actions)
            forces.append(effort)
        references.append(refs)
    lengths = np.array([len(refs) for refs in references], np.int32)
    count = int(lengths.max())

    def padded(values):
        return np.stack(
            [
                np.concatenate([v, np.repeat(v[-1:], count - len(v), axis=0)])
                for v in values
            ],
            axis=1,
        ).astype(np.float32)

    attachments = [r.segment.payload for r in requests]
    for request, attachment in zip(requests, attachments, strict=True):
        if attachment is not None and attachment[0] != request.part:
            raise ValueError("Payload and selected part must agree.")
    literal_actions = padded(commands)
    for world, length in enumerate(lengths):
        literal_actions[length:, world, :6] = 0
    return Prepared(
        references=padded(references),
        bias=padded(bias),
        lengths=lengths,
        contacts=np.array(
            [
                control_guards.contact_limits(r.scene, r.segment, r.part)
                for r in requests
            ],
            np.float32,
        ),
        scene_references=np.array(
            [
                q if r.scene_reference is None else r.scene_reference
                for q, r in zip(initial, requests, strict=True)
            ],
            np.float32,
        ),
        parts=np.array([r.part for r in requests], np.int32),
        guards=np.array([r.guard for r in requests], np.int32),
        monitors=np.array([r.guard and r.monitor_scene for r in requests], np.int32),
        offsets=np.array(
            [p[1] if p is not None else np.zeros(3) for p in attachments], np.float32
        ),
        rotations=np.array(
            [p[2] if p is not None else np.eye(3) for p in attachments], np.float32
        ),
        attachments=np.array(
            [
                r.guard and p is not None
                for r, p in zip(requests, attachments, strict=True)
            ],
            np.int32,
        ),
        efforts=padded(forces),
        literal_mask=np.array([r.actions is not None for r in requests], np.int32),
        literal_actions=literal_actions,
    )


def execute(controller, requests, deadline=float("inf")):
    """Execute independent checked paths; return one aligned trace/event per world.

    Short paths hold their final references until the group completes. Those
    physical holds remain in their traces; they are not discarded padding.
    A nonfinite world fails alone; shared-buffer overflow still aborts the run.
    """
    initial = controller.engine.snapshot()
    prepared = prepare(requests, initial["qpos"])
    return execute_prepared(controller, requests, prepared, deadline)


def _array_bindings(value):
    """Fingerprint device bindings, including nested MJ-Warp model/data arrays."""
    if isinstance(value, wp.array):
        return (value.ptr, value.shape, value.strides, value.dtype)
    if dataclasses.is_dataclass(value):
        return tuple(
            _array_bindings(getattr(value, field.name))
            for field in dataclasses.fields(value)
        )
    return None


@wp.kernel
def record_active(
    active: wp.array[bool],
    step: wp.array[int],
    qpos: wp.array2d[float],
    qvel: wp.array2d[float],
    action: wp.array2d[float],
    positions: wp.array3d[float],
    velocities: wp.array3d[float],
    actions: wp.array3d[float],
):
    world = wp.tid()
    if not active[world]:
        return
    for j in range(qpos.shape[1]):
        positions[step[0], world, j] = qpos[world, j]
    for j in range(qvel.shape[1]):
        velocities[step[0], world, j] = qvel[world, j]
    for j in range(7):
        actions[step[0], world, j] = action[world, j]


class _PreparedExecution:
    """One captured graph with reusable capacity, owned by one controller."""

    def __init__(self, controller, prepared, site, bindings):
        engine = controller.engine
        self.bindings = bindings
        self.active = getattr(controller, "active_worlds", None)
        if self.active is None:
            self.active = wp.ones(engine.nworld, dtype=bool)
        count = max(64, 1 << (len(prepared.references) - 1).bit_length())
        self.capacity = count
        self.arrays = arrays = {}
        for name, dtype in {
            "references": float,
            "bias": float,
            "scene_references": float,
            "parts": int,
            "monitors": int,
            "offsets": wp.vec3,
            "rotations": wp.mat33,
            "attachments": int,
            "efforts": float,
            "literal_mask": int,
            "literal_actions": float,
            "contacts": float,
        }.items():
            shape = getattr(prepared, name).shape
            if name in ("references", "bias", "efforts", "literal_actions"):
                shape = (count, *shape[1:])
            if dtype == wp.vec3:
                shape = shape[:-1]
            elif dtype == wp.mat33:
                shape = shape[:-2]
            arrays[name] = wp.empty(shape, dtype=dtype)
        self.step = step = wp.zeros(1, dtype=int)
        self.failure = failure = wp.zeros(engine.nworld, dtype=int)
        self.first_contact_step = first_contact_step = wp.full(
            engine.nworld, 2147483647, dtype=int
        )
        self.first_contact_distances = first_contact_distances = wp.full(
            (engine.nworld, engine.native_model.ngeom, engine.native_model.ngeom),
            float("inf"),
            dtype=float,
        )
        self.positions = positions = wp.empty(
            (count, engine.nworld, engine.native_model.nq), dtype=float
        )
        self.velocities = velocities = wp.empty(
            (count, engine.nworld, engine.native_model.nv), dtype=float
        )
        self.actions = actions = wp.empty((count, engine.nworld, 7), dtype=float)
        contact = engine.data.contact
        with wp.ScopedCapture() as capture:
            wp.launch(
                replay_grip_effort,
                engine.nworld,
                inputs=[self.active, arrays["efforts"], step,
                        engine.model.actuator_forcerange],
            )
            wp.launch(
                contact_guard,
                contact.dist.shape[0],
                inputs=[
                    engine.data.nacon,
                    contact.geom,
                    contact.worldid,
                    contact.dist,
                    arrays["contacts"],
                    failure,
                    step,
                    first_contact_step,
                    first_contact_distances,
                ],
            )
            wp.launch(
                attachment_guard,
                engine.nworld,
                inputs=[
                    engine.data.qpos,
                    engine.data.site_xpos,
                    engine.data.site_xmat,
                    arrays["parts"],
                    site,
                    arrays["offsets"],
                    arrays["rotations"],
                    arrays["attachments"],
                    failure,
                ],
            )
            wp.launch(
                follow_reference,
                engine.nworld,
                inputs=[
                    self.active,
                    arrays["references"],
                    step,
                    engine.data.qpos,
                    engine.data.qvel,
                    controller.arm_target,
                    arrays["scene_references"],
                    arrays["bias"],
                    arrays["parts"],
                    arrays["monitors"],
                    failure,
                    controller.action,
                    arrays["literal_mask"],
                    arrays["literal_actions"],
                ],
            )
            controller.launch_device_control_step()
            wp.launch(
                record_active,
                engine.nworld,
                inputs=[
                    self.active,
                    step,
                    engine.data.qpos,
                    engine.data.qvel,
                    controller.action,
                    positions,
                    velocities,
                    actions,
                ],
            )
            wp.launch(traces.advance, 1, inputs=[step])
        self.graph = capture.graph

    @timing.measured('execution.upload')
    def upload(self, prepared, allowed):
        """Replace every request parameter and clear per-call failure state."""
        count = len(prepared.references)
        for name, array in self.arrays.items():
            values = allowed if name == "contacts" else getattr(prepared, name)
            target = (
                array[:count]
                if name in ("references", "bias", "efforts", "literal_actions")
                else array
            )
            target.assign(values)
        self.step.zero_()
        self.failure.zero_()
        self.first_contact_step.fill_(2147483647)
        self.first_contact_distances.fill_(float("inf"))


@timing.measured('execution')
def execute_prepared(
    controller, requests, prepared, deadline=float("inf"), *,
    stop_on_failure=False, active_worlds=None,
):
    """Execute a prepared batch without recomputing feedforward at chunk cuts."""
    engine = controller.engine
    if len(requests) != engine.nworld:
        raise ValueError("One request per simulator world is required.")
    models = getattr(engine, "native_models", [engine.native_model] * engine.nworld)
    if any(r.scene.model is not m for r, m in zip(requests, models, strict=True)):
        raise ValueError("Request model must be the simulator model.")
    active = (np.ones(engine.nworld, dtype=bool) if active_worlds is None
              else np.asarray(active_worlds, dtype=bool))
    if active.shape != (engine.nworld,) or not active.any():
        raise ValueError("At least one world must be active, with one mask entry per world")
    if not active.all() and not hasattr(controller, "active_worlds"):
        raise ValueError("Partial execution requires an active-world controller")
    count = len(prepared.references)
    with wp.ScopedDevice(engine.device):
        with timing.span("execution.setup"):
            # Guard-disabled worlds have no forbidden pair, matching serial settling.
            allowed = prepared.contacts.copy()
            allowed[(prepared.guards == 0) | ~active] = -1e6
            prepared = dataclasses.replace(
                prepared, attachments=prepared.attachments * active
            )
            if engine.model.actuator_forcerange.shape[0] == 1 and engine.nworld > 1:
                limits = np.repeat(
                    engine.model.actuator_forcerange.numpy(), engine.nworld, axis=0
                )
                engine.model.actuator_forcerange = wp.array(limits, dtype=wp.vec2)
            bindings = (
                id(engine),
                id(engine.model),
                id(engine.data),
                _array_bindings(engine.model),
                _array_bindings(engine.data),
                tuple(
                    _array_bindings(value)
                    for value in vars(controller).values()
                    if isinstance(value, wp.array)
                ),
                controller.spec,
                tuple(engine.native_model.actuator_ctrlrange[6]),
                requests[0].scene.site,
            )
            execution = getattr(controller, "_prepared_execution", None)
            if (
                execution is None
                or execution.capacity < count
                or execution.bindings != bindings
            ):
                execution = _PreparedExecution(
                    controller, prepared, requests[0].scene.site, bindings
                )
                controller._prepared_execution = execution
            execution.active.assign(active)
            execution.upload(prepared, allowed)
            failure = execution.failure
        with timing.span("simulator"):
            started = time.perf_counter()
            collected = 0
            while collected < count:
                for _ in range(min(64, count - collected)):
                    wp.capture_launch(execution.graph)
                    collected += 1
                failures = failure.numpy()
                if stop_on_failure and np.any(failures):
                    break
                if np.all((failures != 0) | (collected >= prepared.lengths)):
                    break
                if time.perf_counter() >= deadline:
                    failures[(failures == 0) & (collected < prepared.lengths)] = 5
                    wp.copy(failure, wp.array(failures, dtype=int))
                    break
            wp.synchronize_device(engine.device)
            seconds = time.perf_counter() - started
        with timing.span("execution.health"):
            health_errors = _world_health(engine)
        with timing.span("download.arrays"):
            raw = {
                "qpos": execution.positions[:collected].numpy(),
                "qvel": execution.velocities[:collected].numpy(),
                "action": execution.actions[:collected].numpy(),
            }
            failures = failure.numpy()
            contact_steps = execution.first_contact_step.numpy()
            contact_distances = execution.first_contact_distances.numpy()
        with timing.span("execution.evidence"):
            results = []
            for world in range(engine.nworld):
                trace = tracking.finite_prefix(
                    {key: value[:collected if active[world] else 0, world : world + 1]
                     for key, value in raw.items()}
                )
                retained = len(trace["action"])
                executed = collected if active[world] else 0
                health_error = health_errors[world]
                invalid = health_error is not None or retained < executed
                event = {
                    "failure": 6 if invalid else int(failures[world]),
                    "health_error": health_error,
                    "device_contact_violations": [
                        {**hit, "world": world}
                        for hit in control_guards.contact_evidence(
                            models[world],
                            contact_steps[world : world + 1],
                            contact_distances[world : world + 1],
                            allowed[world],
                        )
                    ],
                    "simulator_seconds": seconds,
                    "batch_worlds": engine.nworld,
                    "world": world,
                    "requested_steps": int(prepared.lengths[world]),
                    "executed_control_steps": executed,
                    "batch_lane_control_steps": collected,
                    "paused_control_steps": collected - executed,
                    "control_steps": retained,
                    "padding_control_steps": max(
                        0, retained - int(prepared.lengths[world])
                    ),
                    "discarded_suffix_steps": executed - retained,
                    "tracking_error_rad": float(
                        np.max(
                            np.abs(
                                trace["qpos"][-1, 0, :6]
                                - prepared.references[retained - 1, world, :6]
                            )
                        )
                    )
                    if retained
                    else None,
                }
                results.append((trace, event))
        unhealthy = [w for w, error in enumerate(health_errors) if error is not None]
        if unhealthy:
            _quarantine(controller, unhealthy, results, prepared)
        return results


def _world_health(engine):
    """Per-world nonfinite state; None for a healthy world.

    Worlds are numerically independent, so one world's nonfinite state fails
    only that world. Overflow is different: MJ-Warp flags every world when the
    shared contact buffer overflows, so it remains a run-level error.
    """
    simulation_engine.raise_for_overflow(
        engine.data.overflow.numpy(), global_world_start=engine.global_world_start
    )
    state = {
        name: getattr(engine.data, name).numpy().reshape(engine.nworld, -1)
        for name in ("qpos", "qvel", "ctrl")
    }
    for name in ("observation", "task_state"):
        value = getattr(engine, name, None)
        if value is not None:
            state[name] = value.numpy().reshape(engine.nworld, -1)
    return [
        "; ".join(
            f"nonfinite {name}"
            for name, value in state.items()
            if not np.isfinite(value[world]).all()
        )
        or None
        for world in range(engine.nworld)
    ]


def _quarantine(controller, worlds, results, prepared):
    """Return failed worlds to a finite state so they cannot poison the batch.

    The failed segment is already recorded as failure 6 and ends its attempt;
    the slot is reset again before any later episode. Until then its holds are
    built from this finite state: the last finite recorded state, else the
    chunk's starting scene reference.
    """
    engine = controller.engine
    mask = np.zeros(engine.nworld, dtype=bool)
    mask[worlds] = True
    qpos = np.nan_to_num(engine.data.qpos.numpy())
    qvel = np.nan_to_num(engine.data.qvel.numpy())
    for world in worlds:
        trace = results[world][0]
        qpos[world] = (
            trace["qpos"][-1, 0]
            if len(trace["qpos"])
            else prepared.scene_references[world]
        )
        qvel[world] = 0.0
    world_slots.reset(controller, mask, qpos=qpos, qvel=qvel)


@wp.kernel
def replay_grip_effort(
    active: wp.array(dtype=bool),
    efforts: wp.array2d(dtype=float),
    step: wp.array(dtype=int),
    limits: wp.array2d(dtype=wp.vec2),
):
    world = wp.tid()
    if not active[world]:
        return
    effort = efforts[step[0], world]
    limits[world, 6] = wp.vec2(-effort, effort)


def replay(controller, actions, efforts):
    """Replay each world's recorded actions and forces from its reset state."""
    actions, efforts = np.asarray(actions), np.asarray(efforts)
    if (
        actions.ndim != 3
        or actions.shape[2] != 7
        or not len(actions)
        or efforts.shape != actions.shape[:2]
        or not np.isfinite(actions).all()
        or not np.all(np.isfinite(efforts) & (efforts > 0))
    ):
        raise ValueError(
            "Finite [time, world, 7] actions and positive [time, world] forces required"
        )
    engine = controller.engine
    count, worlds = actions.shape[:2]
    if worlds != engine.nworld:
        raise ValueError("Replay world count differs from simulator")
    with wp.ScopedDevice(engine.device):
        if hasattr(controller, "active_worlds"):
            controller.active_worlds.fill_(True)
        if engine.model.actuator_forcerange.shape[0] == 1 and worlds > 1:
            engine.model.actuator_forcerange = wp.array(
                np.repeat(engine.model.actuator_forcerange.numpy(), worlds, axis=0),
                dtype=wp.vec2,
            )
        commands = wp.array(actions.astype(np.float32), dtype=float)
        forces = wp.array(efforts.astype(np.float32), dtype=float)
        step = wp.zeros(1, dtype=int)
        q = wp.zeros((count, worlds, engine.native_model.nq), dtype=float)
        v = wp.zeros((count, worlds, engine.native_model.nv), dtype=float)
        output = wp.zeros((count, worlds, 7), dtype=float)
        with wp.ScopedCapture() as capture:
            wp.launch(
                replay_grip_effort,
                worlds,
                inputs=[wp.ones(worlds, dtype=bool), forces, step,
                        engine.model.actuator_forcerange],
            )
            wp.launch(
                traces.replay_action,
                (worlds, 7),
                inputs=[commands, step, controller.action],
            )
            controller.launch_device_control_step()
            wp.launch(
                traces.record_state,
                worlds,
                inputs=[
                    step,
                    engine.data.qpos,
                    engine.data.qvel,
                    controller.action,
                    q,
                    v,
                    output,
                ],
            )
            wp.launch(traces.advance, 1, inputs=[step])
        started = time.perf_counter()
        for _ in range(count):
            wp.capture_launch(capture.graph)
        wp.synchronize_device(engine.device)
        seconds = time.perf_counter() - started
        engine.assert_healthy()
        return {"qpos": q.numpy(), "qvel": v.numpy(), "action": output.numpy()}, seconds
