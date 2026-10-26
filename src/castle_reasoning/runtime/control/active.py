"""Runtime control active."""

import mujoco_warp as mjw
import warp as wp

import castle_reasoning.runtime.control.tracking as tracking
import castle_reasoning.runtime.diagnostics.provenance as provenance


@wp.kernel
def apply_action(
    active: wp.array[bool],
    qpos: wp.array2d[float],
    action: wp.array2d[float],
    target: wp.array2d[float],
    control: wp.array2d[float],
    velocity_limit: wp.array[float],
    lower: wp.array[float],
    upper: wp.array[float],
    gripper_lower: float,
    gripper_upper: float,
    timestep: float,
):
    world, joint = wp.tid()
    if not active[world]:
        return
    # Same clamp and velocity mapping as PathControl + controller_features.
    normalized = wp.clamp(action[world, joint], -1.0, 1.0)
    if joint < 6:
        value = wp.clamp(
            target[world, joint], qpos[world, joint] - 0.12, qpos[world, joint] + 0.12
        )
        value = value + normalized * velocity_limit[joint] * timestep
        value = wp.clamp(value, lower[joint], upper[joint])
        target[world, joint] = value
        control[world, joint] = value
    else:
        closure = 0.5 * (normalized + 1.0)
        control[world, joint] = gripper_lower + closure * (
            gripper_upper - gripper_lower
        )


class ActivePathControl(tracking.PathControl):
    """Preserve paused controller rows and use the explicit private-runtime API."""

    def __init__(self, collector):
        if not hasattr(mjw, "step_active"):
            raise RuntimeError(
                "Build and select the repo-owned MJ-Warp active-world runtime"
            )
        provenance.runtime_provenance()
        super().__init__(collector)
        with wp.ScopedDevice(self.engine.device):
            self.active_worlds = wp.ones(self.engine.nworld, dtype=bool)

    def launch_device_control_step(self):
        engine = self.engine
        limits = engine.native_model.actuator_ctrlrange[6]
        wp.launch(
            apply_action,
            (engine.nworld, 7),
            inputs=[
                self.active_worlds,
                engine.data.qpos,
                self.action,
                self.arm_target,
                engine.data.ctrl,
                self.velocity_limit,
                self.arm_control_lower,
                self.arm_control_upper,
                float(limits[0]),
                float(limits[1]),
                self.spec.control_timestep_seconds,
            ],
        )
        for _ in range(self.spec.physics_steps_per_control):
            mjw.step_active(engine.model, engine.data, self.active_worlds)
