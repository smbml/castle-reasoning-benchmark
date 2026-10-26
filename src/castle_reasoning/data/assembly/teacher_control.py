"""Data assembly teacher_control."""

import numpy as np
import warp as wp

import castle_reasoning.runtime.control.gripper as gripper
import castle_reasoning.runtime.control.joint_velocity as joint_velocity
import castle_reasoning.runtime.control.parameters as parameters


@wp.struct
class Guard:
    phase: wp.array(dtype=int)
    elapsed: wp.array(dtype=float)
    move: wp.array(dtype=int)
    order: wp.array2d(dtype=int)
    retries: wp.array(dtype=int)
    failed: wp.array(dtype=bool)
    completed: wp.array(dtype=bool)
    failure: wp.array(dtype=int)
    orientation: wp.array(dtype=wp.quat)
    target_rpy: wp.array2d(dtype=wp.vec3)
    latched: wp.array(dtype=int)
    offset: wp.array(dtype=wp.vec3)
    waiting: wp.array(dtype=int)
    wait_seconds: wp.array(dtype=float)
    quiet_steps: wp.array(dtype=int)
    event: wp.array(dtype=int)
    interruptions: wp.array(dtype=int)


def make_guard(worlds, parts, proposals, device):
    """Allocate guard state; collection binds oracle fields to existing buffers."""
    guard = Guard()
    for key in [
        "phase",
        "move",
        "retries",
        "failure",
        "latched",
        "waiting",
        "quiet_steps",
        "event",
        "interruptions",
    ]:
        setattr(guard, key, wp.zeros(worlds, dtype=int, device=device))
    for key in ["elapsed", "wait_seconds"]:
        setattr(guard, key, wp.zeros(worlds, dtype=float, device=device))
    for key in ["failed", "completed"]:
        setattr(guard, key, wp.zeros(worlds, dtype=bool, device=device))
    guard.order = wp.zeros((worlds, parts), dtype=int, device=device)
    guard.offset = wp.zeros(worlds, dtype=wp.vec3, device=device)
    guard.orientation = wp.zeros(worlds, dtype=wp.quat, device=device)
    guard.target_rpy = wp.zeros(
        (worlds, parts * proposals), dtype=wp.vec3, device=device
    )
    return guard


@wp.kernel
def guard_action(
    guard: Guard,
    qpos: wp.array2d(dtype=float),
    qvel: wp.array2d(dtype=float),
    site_pos: wp.array2d(dtype=wp.vec3),
    site_rot: wp.array2d(dtype=wp.mat33),
    pinch_id: int,
    action: wp.array2d(dtype=float),
    enabled: wp.array(dtype=int),
    attachment_error: float,
    angular_limit: float,
    retry_limit: int,
    proposals: int,
):
    world = wp.tid()
    if enabled[0] == 0:
        return
    guard.event[world] = 0
    phase = guard.phase[world]
    part = guard.order[world, guard.move[world]]
    p = 14 + 7 * part
    v = 14 + 6 * part
    position = wp.vec3(qpos[world, p], qpos[world, p + 1], qpos[world, p + 2])
    linear = wp.vec3(qvel[world, v], qvel[world, v + 1], qvel[world, v + 2])
    angular = wp.vec3(qvel[world, v + 3], qvel[world, v + 4], qvel[world, v + 5])
    relative = wp.transpose(site_rot[world, pinch_id]) * (
        position - site_pos[world, pinch_id]
    )
    if not guard.failed[world] and not guard.completed[world]:
        if phase == 3 and guard.latched[world] == 0:
            guard.offset[world] = relative
            guard.latched[world] = 1
        if (phase < 3 or phase > 5) and guard.waiting[world] == 0:
            guard.latched[world] = 0
        reason = 0
        if phase >= 3 and phase <= 5 and guard.latched[world] != 0:
            if wp.length(relative - guard.offset[world]) > attachment_error:
                reason = 1
            if wp.length(angular) > angular_limit:
                reason = 2
        if reason != 0:
            guard.retries[world] += 1
        elif phase == 8 and guard.waiting[world] == 0:
            # The timed expert already incremented retries for this failure.
            reason = 3
        if reason != 0:
            guard.event[world] = reason
            guard.interruptions[world] += 1
            guard.failure[world] = 10 + reason
            guard.waiting[world] = 1
            guard.wait_seconds[world] = 0.0
            guard.quiet_steps[world] = 0
            guard.latched[world] = 0
            if guard.retries[world] > retry_limit:
                guard.failed[world] = True
        if guard.waiting[world] != 0 and not guard.failed[world]:
            guard.phase[world] = 8
            guard.elapsed[world] = 0.0
            guard.wait_seconds[world] += 0.02
            if wp.length(linear) < 0.02 and wp.length(angular) < 0.5:
                guard.quiet_steps[world] += 1
            else:
                guard.quiet_steps[world] = 0
            if guard.wait_seconds[world] >= 0.4 and guard.quiet_steps[world] >= 5:
                quat = wp.quat(
                    qpos[world, p + 4],
                    qpos[world, p + 5],
                    qpos[world, p + 6],
                    qpos[world, p + 3],
                )
                rotation = wp.quat_to_matrix(quat)
                reachable = (
                    position[0] >= 0.20
                    and position[0] <= 0.65
                    and wp.abs(position[1]) <= 0.33
                    and position[2] >= 0.0
                    and position[2] <= 0.4
                )
                if not reachable or rotation[2, 2] < 0.985:
                    guard.failed[world] = True
                    guard.failure[world] = 14
                else:
                    proposal = guard.move[world] * proposals + guard.retries[world]
                    yaw = wp.atan2(rotation[1, 0], rotation[0, 0])
                    angles = guard.target_rpy[world, proposal]
                    guard.target_rpy[world, proposal] = wp.vec3(
                        wp.pi, 0.0, angles[2] + yaw
                    )
                    guard.orientation[world] = wp.quat_from_matrix(
                        site_rot[world, pinch_id]
                    )
                    guard.phase[world] = 0
                    guard.elapsed[world] = 0.0
                    guard.waiting[world] = 0
                    guard.event[world] = 4
            elif guard.wait_seconds[world] > 2.0:
                guard.failed[world] = True
                guard.failure[world] = 15
    if (
        guard.waiting[world] != 0
        or guard.failed[world]
        or guard.completed[world]
        or guard.event[world] == 4
    ):
        for joint in range(6):
            action[world, joint] = 0.0
        action[world, 6] = -1.0


@wp.kernel
def record_guard(
    guard: Guard,
    enabled: wp.array(dtype=int),
    step: wp.array(dtype=int),
    output: wp.array3d(dtype=int),
):
    world = wp.tid()
    index = step[0]
    if enabled[0] != 0 and index < output.shape[0]:
        output[index, world, 0] = guard.event[world]
        output[index, world, 1] = guard.interruptions[world]
        output[index, world, 2] = guard.waiting[world]
        output[index, world, 3] = guard.retries[world]


def grasp_parameters(sizes, order, proposals):
    """Top pinches across the shorter side, indexed by move then proposal."""
    selected = np.repeat(np.asarray(sizes)[order], proposals, axis=0)
    offsets = np.zeros_like(selected)
    # Closing the linkage lowers the fingertip to 18 mm below the pinch site.
    # Keep a partial-depth grip so thin beams do not pull on the course below.
    grip_depth = np.minimum(0.020, 0.6 * selected[:, 2])
    offsets[:, 2] = selected[:, 2] / 2 + 0.018 - grip_depth
    angles = np.zeros_like(selected)
    angles[:, 0] = np.pi
    # The stock gripper closes along its local Y axis at zero world yaw.
    angles[:, 2] = np.where(selected[:, 0] < selected[:, 1], np.pi / 2, 0.0)
    return offsets, angles


def configure_grasps(collector, sizes, order, clearance=0.20):
    """Update data buffers after reset; the captured graph reads them in place."""
    import warp as wp

    offsets, angles = grasp_parameters(sizes, order, collector.grasp_proposal_count)
    with wp.ScopedDevice(collector.engine.device):
        for field, values in (
            (collector.grasp_offset, offsets),
            (collector.target_rpy, angles),
        ):
            wp.copy(
                field,
                wp.array(
                    np.broadcast_to(values, (collector.nworld, *values.shape))
                    .copy()
                    .astype(np.float32),
                    dtype=wp.vec3,
                ),
            )
        collector.clearance.fill_(float(clearance))


class CuboidController(joint_velocity.MJWarpJointVelocityController):
    """Shapes approach aperture; a device flag bypasses shaping for exact replay."""

    def __init__(self, collector):
        self.move = collector.block_index
        with wp.ScopedDevice(collector.engine.device):
            self.shaping_enabled = wp.zeros(1, dtype=int)
            self.opening = wp.zeros(
                (collector.nworld, collector.task.block_count), dtype=float
            )
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
        wp.launch(
            gripper.shape_aperture,
            self.engine.nworld,
            inputs=[self.shaping_enabled, self.move, self.opening, self.action],
        )
        super().launch_device_control_step()

    def set_widths(self, sizes, order):
        widths = np.min(np.asarray(sizes)[order, :2], axis=1)
        values = gripper.opening_closure(self.engine.native_model, widths)
        with wp.ScopedDevice(self.engine.device):
            wp.copy(
                self.opening,
                wp.array(
                    np.broadcast_to(values, self.opening.shape)
                    .copy()
                    .astype(np.float32),
                    dtype=float,
                ),
            )


class CuboidControl(CuboidController):
    """Applies the same servo rule to collection, home and recorded replay."""

    def __init__(self, collector, config):
        self.config = config
        self.collector = collector
        super().__init__(collector)
        with wp.ScopedDevice(self.engine.device):
            self.holding = wp.zeros(self.opening.shape, dtype=float)
            self.was_stopped = wp.zeros(self.engine.nworld, dtype=int)
            self.guard = make_guard(
                collector.nworld,
                collector.task.block_count,
                collector.grasp_proposal_count,
                self.engine.device,
            )
            for key, source in {
                "phase": "phase",
                "elapsed": "phase_time",
                "move": "block_index",
                "order": "block_order",
                "retries": "recovery_count",
                "failed": "failed",
                "completed": "completed",
                "failure": "failure_reason",
                "orientation": "orientation_reference",
                "target_rpy": "target_rpy",
            }.items():
                setattr(self.guard, key, getattr(collector, source))
            self.recovery_trace = wp.zeros(
                (collector.record_capacity, collector.nworld, 4), dtype=int
            )

    def reset_recovery(self):
        self.was_stopped.zero_()
        for name in [
            "latched",
            "waiting",
            "wait_seconds",
            "quiet_steps",
            "event",
            "interruptions",
        ]:
            getattr(self.guard, name).zero_()
        self.recovery_trace.zero_()

    def set_widths(self, sizes, order):
        super().set_widths(sizes, order)
        widths = np.min(np.asarray(sizes)[order, :2], axis=1)
        values = gripper.opening_closure(
            self.engine.native_model, widths, margin=-self.config.grip_undersize
        )
        with wp.ScopedDevice(self.engine.device):
            wp.copy(
                self.holding,
                wp.array(
                    np.broadcast_to(values, self.holding.shape)
                    .copy()
                    .astype(np.float32),
                    dtype=float,
                ),
            )
            directions = np.zeros(
                (*self.collector.approach_direction.shape, 3), np.float32
            )
            directions[..., 2] = -1
            wp.copy(
                self.collector.approach_direction, wp.array(directions, dtype=wp.vec3)
            )

    def launch_device_control_step(self):
        wp.launch(
            gripper.shape_hold,
            self.engine.nworld,
            inputs=[
                self.shaping_enabled,
                self.move,
                self.opening,
                self.holding,
                self.action,
            ],
        )
        c = self.collector
        wp.launch(
            guard_action,
            self.engine.nworld,
            inputs=[
                self.guard,
                self.engine.data.qpos,
                self.engine.data.qvel,
                self.engine.data.site_xpos,
                self.engine.data.site_xmat,
                c._pinch_site_id,
                self.action,
                self.shaping_enabled,
                self.config.attachment_error,
                self.config.angular_speed,
                c.parameters.max_recovery_attempts,
                c.grasp_proposal_count,
            ],
        )
        wp.launch(
            record_guard,
            self.engine.nworld,
            inputs=[
                self.guard,
                self.shaping_enabled,
                c.record_step,
                self.recovery_trace,
            ],
        )
        wp.launch(
            gripper.bound_targets,
            self.engine.nworld,
            inputs=[
                self.engine.data.qpos,
                self.action,
                self.arm_target,
                self.config.target_error,
                self.was_stopped,
            ],
        )
        joint_velocity.MJWarpJointVelocityController.launch_device_control_step(
            self
        )
