"""Runtime control joint_velocity."""

import math
import time

import mujoco
import mujoco_warp
import numpy as np
import warp as wp

import castle_reasoning.runtime.control.parameters as parameters
import castle_reasoning.runtime.simulation.engine as engine

_ARM_ACTUATORS = (
    "ur5e/shoulder_pan",
    "ur5e/shoulder_lift",
    "ur5e/elbow",
    "ur5e/wrist_1",
    "ur5e/wrist_2",
    "ur5e/wrist_3",
)


_GRIPPER_ACTUATOR = "ur5e/robotiq/fingers_actuator"


def _actuator_name(model: mujoco.MjModel, actuator_id: int) -> str:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
    return "" if name is None else name


def _validate_ogbench_model(
    model: mujoco.MjModel, spec: parameters.ControllerSpec
) -> None:
    if model.nu != 7:
        raise ValueError(f"Expected seven OGBench actuators, found {model.nu}.")
    names = tuple(_actuator_name(model, index) for index in range(model.nu))
    if names[:6] != _ARM_ACTUATORS or names[6] != _GRIPPER_ACTUATOR:
        raise ValueError(f"Unexpected OGBench actuator order: {names}.")
    if not np.array_equal(model.actuator_trnid[:6, 0], np.arange(6)):
        raise ValueError("Arm actuators must address the first six joints.")
    if not math.isclose(
        model.opt.timestep, spec.physics_timestep_seconds, abs_tol=1e-12
    ):
        raise ValueError(
            f"Model timestep {model.opt.timestep} does not match "
            f"{spec.physics_timestep_seconds}."
        )


@wp.kernel
def _sync_arm_targets(
    mask: wp.array(dtype=bool),
    qpos: wp.array2d(dtype=float),
    arm_target: wp.array2d(dtype=float),
):
    world, joint = wp.tid()
    if mask[world]:
        arm_target[world, joint] = qpos[world, joint]


@wp.kernel
def _apply_velocity_action(
    action: wp.array2d(dtype=float),
    arm_target: wp.array2d(dtype=float),
    control: wp.array2d(dtype=float),
    velocity_limit: wp.array(dtype=float),
    arm_control_lower: wp.array(dtype=float),
    arm_control_upper: wp.array(dtype=float),
    gripper_control_lower: float,
    gripper_control_upper: float,
    control_timestep: float,
):
    world, action_index = wp.tid()
    normalized = wp.clamp(action[world, action_index], -1.0, 1.0)
    if action_index < 6:
        target = arm_target[world, action_index]
        target = target + (
            normalized * velocity_limit[action_index] * control_timestep
        )
        target = wp.clamp(
            target,
            arm_control_lower[action_index],
            arm_control_upper[action_index],
        )
        arm_target[world, action_index] = target
        control[world, action_index] = target
    else:
        closure = 0.5 * (normalized + 1.0)
        control[world, action_index] = gripper_control_lower + closure * (
            gripper_control_upper - gripper_control_lower
        )


class MJWarpJointVelocityController:
    """GPU-resident controller with one controller-plus-physics CUDA graph."""

    def __init__(
        self,
        engine: engine.MJWarpEngine,
        spec: parameters.ControllerSpec | None = None,
        *,
        graph: bool = True,
    ):
        self.engine = engine
        self.spec = spec or parameters.ControllerSpec()
        _validate_ogbench_model(engine.native_model, self.spec)
        if graph and not engine.device.is_cuda:
            raise ValueError("Controller graph capture requires a CUDA device.")
        self.graph_enabled = graph
        model = engine.native_model
        with wp.ScopedDevice(engine.device):
            self.action = wp.zeros(
                (engine.nworld, self.spec.action_width), dtype=float
            )
            self.arm_target = wp.zeros((engine.nworld, 6), dtype=float)
            self.sync_mask = wp.ones(engine.nworld, dtype=bool)
            self.velocity_limit = wp.array(
                np.asarray(
                    self.spec.joint_velocity_limits_rad_s, dtype=np.float32
                ),
                dtype=float,
            )
            self.arm_control_lower = wp.array(
                model.actuator_ctrlrange[:6, 0].astype(np.float32),
                dtype=float,
            )
            self.arm_control_upper = wp.array(
                model.actuator_ctrlrange[:6, 1].astype(np.float32),
                dtype=float,
            )
            self.sync_targets()
            self.graph = None
            self.cold_capture_seconds = 0.0
            if graph:
                started = time.perf_counter()
                with wp.ScopedCapture() as capture:
                    self.launch_device_control_step()
                self.graph = capture.graph
                self.cold_capture_seconds = time.perf_counter() - started

    def launch_device_control_step(self) -> None:
        """Enqueues controller and physics kernels for outer graph capture."""
        model = self.engine.native_model
        gripper_lower, gripper_upper = model.actuator_ctrlrange[6]
        wp.launch(
            _apply_velocity_action,
            dim=(self.engine.nworld, self.spec.action_width),
            inputs=[
                self.action,
                self.arm_target,
                self.engine.data.ctrl,
                self.velocity_limit,
                self.arm_control_lower,
                self.arm_control_upper,
                float(gripper_lower),
                float(gripper_upper),
                self.spec.control_timestep_seconds,
            ],
        )
        for _ in range(self.spec.physics_steps_per_control):
            mujoco_warp.step(self.engine.model, self.engine.data)

    def launch_sync_targets(self, mask: wp.array) -> None:
        """Enqueues a device-mask target sync for outer graph capture."""
        wp.launch(
            _sync_arm_targets,
            dim=(self.engine.nworld, 6),
            inputs=[mask, self.engine.data.qpos, self.arm_target],
        )

    def sync_targets(self, mask: np.ndarray | None = None) -> None:
        """Synchronizes selected commanded targets from device qpos."""
        if mask is None:
            mask = np.ones(self.engine.nworld, dtype=bool)
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (self.engine.nworld,):
            raise ValueError(
                f"Sync mask must have shape {(self.engine.nworld,)}, "
                f"got {mask.shape}."
            )
        with wp.ScopedDevice(self.engine.device):
            wp.copy(
                self.sync_mask,
                wp.array(mask, dtype=bool),
            )
            self.launch_sync_targets(self.sync_mask)
            wp.synchronize_device(self.engine.device)

    def reset(self, mask: np.ndarray) -> None:
        """Resets engine worlds and synchronizes their controller targets."""
        self.engine.reset(mask)
        self.sync_targets(mask)

    def step(
        self, normalized_action: np.ndarray, *, graph: bool | None = None
    ) -> engine.StepMetrics:
        """Uploads host actions and executes one 20 ms control interval."""
        action = np.asarray(normalized_action, dtype=np.float32)
        expected = (self.engine.nworld, self.spec.action_width)
        if action.shape != expected:
            raise ValueError(
                f"Action must have shape {expected}, got {action.shape}."
            )
        if not np.isfinite(action).all():
            raise ValueError("Actions must be finite.")
        with wp.ScopedDevice(self.engine.device):
            wp.copy(
                self.action,
                wp.array(action, dtype=float),
            )
        return self.step_device(graph=graph)

    def step_device(
        self, *, graph: bool | None = None
    ) -> engine.StepMetrics:
        """Executes one interval from the GPU-resident `action` buffer."""
        use_graph = self.graph_enabled if graph is None else graph
        started = time.perf_counter()
        with wp.ScopedDevice(self.engine.device):
            if use_graph:
                if self.graph is None:
                    raise ValueError(
                        "Graph execution requested without a captured graph."
                    )
                wp.capture_launch(self.graph)
            else:
                self.launch_device_control_step()
            wp.synchronize_device(self.engine.device)
        seconds = time.perf_counter() - started
        self.engine.assert_healthy()
        steps = self.spec.physics_steps_per_control
        return engine.StepMetrics(
            seconds=seconds,
            physics_steps=steps,
            world_steps=steps * self.engine.nworld,
            used_graph=use_graph,
        )
