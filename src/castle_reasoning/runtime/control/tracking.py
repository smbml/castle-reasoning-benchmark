"""Runtime control tracking."""

import time

import mujoco
import numpy as np
import warp as wp

import castle_reasoning.runtime.control.guards as guards
import castle_reasoning.runtime.control.joint_velocity as joint_velocity
import castle_reasoning.runtime.control.parameters as parameters
import castle_reasoning.runtime.diagnostics.timing as timing
import castle_reasoning.runtime.execution.traces as traces
import castle_reasoning.runtime.planning.grasps as grasps


def finite_prefix(trace):
    """Retain aligned samples before the first corrupt state or action."""
    valid = np.logical_and.reduce(
        [
            np.isfinite(values).all(axis=tuple(range(1, values.ndim)))
            for values in trace.values()
        ]
    )
    invalid = np.flatnonzero(~valid)
    stop = int(invalid[0]) if len(invalid) else len(valid)
    return {name: values[:stop] for name, values in trace.items()}


@timing.measured('prepare.interpolation')
def references(segment, speed=None):
    """Linear checked edges, with a halt at each free-space planning vertex."""
    speed = segment.reference_speed if speed is None else speed
    values = []
    for i, (a, b) in enumerate(zip(segment.path, segment.path[1:])):
        duration = max(
            np.max(np.abs(b[:6] - a[:6])) / speed,
            abs(segment.closure[i + 1] - segment.closure[i]) * 1.5,
            0.02,
        )
        count = max(1, int(np.ceil(duration / 0.02)))
        for fraction in np.linspace(0, 1, count + 1)[1:]:
            closure = segment.closure[i] + fraction * (
                segment.closure[i + 1] - segment.closure[i]
            )
            values.append(np.r_[a[:6] + fraction * (b[:6] - a[:6]), 2 * closure - 1])
        if segment.name in ("approach", "transit", "home"):
            values.extend([values[-1].copy() for _ in range(10)])
    if not values:
        values = [np.r_[segment.path[-1, :6], 2 * segment.closure[-1] - 1]]
    values.extend(
        [
            values[-1].copy()
            for _ in range(25 if segment.name in ("close", "open") else 10)
        ]
    )
    return np.asarray(values, dtype=np.float32)


@timing.measured('prepare.feedforward')
def servo_offsets(model, refs, initial):
    """Gravity/Coriolis and damping feedforward through the stock position servos.

    Native RNE is a dynamics query, not a physics rollout. Unit-gear servos have
    force kp*(target-q)-kd*qvel; compensate damping at the reference velocity
    and the model's gravitational load without changing any actuator gains.
    """
    if not np.all(model.actuator_gear[:6, 0] == 1):
        raise ValueError("Feedforward requires unit-gear joint servos.")
    state = mujoco.MjData(model)
    state.qpos[:] = initial
    velocity = np.gradient(refs[:, :6].astype(float), 0.02, axis=0)
    force = np.zeros(model.nv)
    offsets = []
    for ref, speed in zip(refs, velocity):
        state.qpos[:6] = ref[:6]
        state.qpos[6:14] = grasps.Aperture.joints((ref[6] + 1) / 2)
        state.qvel[:6] = speed
        mujoco.mj_kinematics(model, state)
        mujoco.mj_comPos(model, state)
        mujoco.mj_comVel(model, state)
        mujoco.mj_rne(model, state, 0, force)
        offsets.append(
            (force[:6] - model.actuator_biasprm[:6, 2] * speed)
            / model.actuator_gainprm[:6, 0]
        )
    return np.array(offsets, dtype=np.float32)


@wp.kernel
def follow_reference(
    reference: wp.array2d(dtype=float),
    step: wp.array(dtype=int),
    qpos: wp.array2d(dtype=float),
    qvel: wp.array2d(dtype=float),
    arm_target: wp.array2d(dtype=float),
    scene: wp.array(dtype=float),
    bias: wp.array2d(dtype=float),
    selected: int,
    guard_enabled: int,
    failure: wp.array(dtype=int),
    action: wp.array2d(dtype=float),
):
    world = wp.tid()
    if failure[world] == 0:
        action[world, 6] = reference[step[0], 6]
        if guard_enabled:
            for part in range((qpos.shape[1] - 14) // 7):
                p = 14 + part * 7
                v = 14 + part * 6
                omega = wp.length(
                    wp.vec3(qvel[world, v + 3], qvel[world, v + 4], qvel[world, v + 5])
                )
                if omega > 20.0:
                    failure[world] = 2
                if part != selected:
                    distance = wp.length(
                        wp.vec3(
                            qpos[world, p] - scene[p],
                            qpos[world, p + 1] - scene[p + 1],
                            qpos[world, p + 2] - scene[p + 2],
                        )
                    )
                    dot = float(0.0)  # noqa: UP018 - Warp dynamic loop accumulator.
                    for j in range(4):
                        dot += qpos[world, p + 3 + j] * scene[p + 3 + j]
                    if distance > 0.003 or wp.abs(dot) < 0.999390827:
                        failure[world] = 2
        for joint in range(6):
            if wp.abs(qpos[world, joint] - reference[step[0], joint]) > 0.08:
                failure[world] = 1
    for joint in range(6):
        if failure[world] != 0:
            action[world, joint] = 0.0
        else:
            action[world, joint] = wp.clamp(
                (
                    reference[step[0], joint]
                    + bias[step[0], joint]
                    - arm_target[world, joint]
                )
                / (0.02 * wp.pi),
                -0.15,
                0.15,
            )


@wp.kernel
def clamp_targets(qpos: wp.array2d(dtype=float), target: wp.array2d(dtype=float)):
    world, joint = wp.tid()
    target[world, joint] = wp.clamp(
        target[world, joint], qpos[world, joint] - 0.12, qpos[world, joint] + 0.12
    )


class PathControl(joint_velocity.MJWarpJointVelocityController):
    """Same bounded servo mapping in planned collection and action replay."""

    def __init__(self, collector):
        super().__init__(
            collector.engine,
            parameters.ControllerSpec(
                physics_timestep_seconds=float(
                    collector.engine.native_model.opt.timestep
                )
            ),
            graph=False,
        )

    def launch_device_control_step(self):
        # Zero action holds the existing target, including the load-supporting
        # offset. It must not silently cancel gravity compensation on replay.
        wp.launch(
            clamp_targets,
            (self.engine.nworld, 6),
            inputs=[self.engine.data.qpos, self.arm_target],
        )
        super().launch_device_control_step()


def execute_segment(
    controller,
    segment,
    scene,
    part,
    guard=True,
    scene_reference=None,
    deadline=float("inf"),
    grip_effort=1.0,
    monitor_scene=True,
):
    """Execute a segment on device; download a trace only at bounded intervals."""
    engine = controller.engine
    initial = engine.snapshot()
    refs = references(segment)
    count = len(refs)
    allowed = guards.contact_limits(scene, segment, part)
    if not np.isfinite(grip_effort) or grip_effort <= 0:
        raise ValueError("Grip effort must be finite and positive.")
    with wp.ScopedDevice(engine.device):
        ref = wp.array(refs, dtype=float)
        original = wp.array(
            initial["qpos"][0] if scene_reference is None else scene_reference,
            dtype=float,
        )
        limits = wp.array(allowed, dtype=float)
        step = wp.zeros(1, dtype=int)
        efforts = wp.full(count, grip_effort, dtype=float)
        bias = wp.array(
            servo_offsets(scene.model, refs, initial["qpos"][0]), dtype=float
        )
        failure = wp.zeros(engine.nworld, dtype=int)
        first_contact_step = wp.full(engine.nworld, 2147483647, dtype=int)
        first_contact_distances = wp.full(
            (engine.nworld, scene.model.ngeom, scene.model.ngeom),
            float("inf"),
            dtype=float,
        )
        q = wp.zeros((count, engine.nworld, scene.model.nq), dtype=float)
        v = wp.zeros((count, engine.nworld, scene.model.nv), dtype=float)
        actions = wp.zeros((count, engine.nworld, 7), dtype=float)
        contact = engine.data.contact
        with wp.ScopedCapture() as capture:
            wp.launch(
                traces.apply_grip_effort,
                engine.model.actuator_forcerange.shape[0],
                inputs=[efforts, step, engine.model.actuator_forcerange],
            )
            if guard:
                wp.launch(
                    guards.contact_guard,
                    contact.dist.shape[0],
                    inputs=[
                        engine.data.nacon,
                        contact.geom,
                        contact.worldid,
                        contact.dist,
                        limits,
                        failure,
                        step,
                        first_contact_step,
                        first_contact_distances,
                    ],
                )
                if segment.payload is not None:
                    _, offset, rotation = segment.payload
                    wp.launch(
                        guards.attachment_guard,
                        engine.nworld,
                        inputs=[
                            engine.data.qpos,
                            engine.data.site_xpos,
                            engine.data.site_xmat,
                            part,
                            scene.site,
                            wp.vec3(*offset),
                            wp.mat33(*rotation.ravel()),
                            failure,
                        ],
                    )
            wp.launch(
                follow_reference,
                engine.nworld,
                inputs=[
                    ref,
                    step,
                    engine.data.qpos,
                    engine.data.qvel,
                    controller.arm_target,
                    original,
                    bias,
                    part,
                    int(guard and monitor_scene),
                    failure,
                    controller.action,
                ],
            )
            controller.launch_device_control_step()
            wp.launch(
                traces.record_state,
                engine.nworld,
                inputs=[
                    step,
                    engine.data.qpos,
                    engine.data.qvel,
                    controller.action,
                    q,
                    v,
                    actions,
                ],
            )
            wp.launch(traces.advance, 1, inputs=[step])
        started = time.perf_counter()
        collected = 0
        while collected < count:
            for _ in range(min(64, count - collected)):
                wp.capture_launch(capture.graph)
                collected += 1
            if failure.numpy().any():
                break
            if time.perf_counter() >= deadline and collected < count:
                failure.fill_(5)
                break
        wp.synchronize_device(engine.device)
        elapsed = time.perf_counter() - started
        health_error = None
        try:
            engine.assert_healthy()
        except FloatingPointError as error:
            # The caller rejects the attempt and exports the earlier valid moves.
            health_error = str(error)
        trace = finite_prefix(
            {
                "qpos": q.numpy()[:collected],
                "qvel": v.numpy()[:collected],
                "action": actions.numpy()[:collected],
            }
        )
        retained = len(trace["action"])
        corrupt = health_error is not None or retained < collected
        return trace, {
            "failure": 6 if corrupt else int(failure.numpy().max()),
            "health_error": health_error,
            "device_contact_violations": guards.contact_evidence(
                scene.model,
                first_contact_step.numpy(),
                first_contact_distances.numpy(),
                allowed,
            ),
            "simulator_seconds": elapsed,
            "requested_steps": count,
            "executed_control_steps": collected,
            "control_steps": retained,
            "discarded_suffix_steps": collected - retained,
            "tracking_error_rad": float(
                np.max(np.abs(trace["qpos"][-1, 0, :6] - refs[retained - 1, :6]))
            )
            if retained
            else None,
        }
