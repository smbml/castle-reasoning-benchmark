"""Runtime execution traces."""

import time

import numpy as np
import warp as wp

import castle_reasoning.runtime.simulation.robot as robot


@wp.kernel
def home_action(
    target: wp.array2d(dtype=float),
    home: wp.array(dtype=float),
    action: wp.array2d(dtype=float),
):
    world = wp.tid()
    for j in range(6):
        action[world, j] = wp.clamp(
            2.0 * (home[j] - target[world, j]) / 3.14159265, -0.25, 0.25
        )
    action[world, 6] = -1.0


@wp.kernel
def replay_action(
    actions: wp.array3d(dtype=float),
    step: wp.array(dtype=int),
    output: wp.array2d(dtype=float),
):
    world, j = wp.tid()
    output[world, j] = actions[step[0], world, j]


@wp.kernel
def record_state(
    step: wp.array(dtype=int),
    qpos: wp.array2d(dtype=float),
    qvel: wp.array2d(dtype=float),
    action: wp.array2d(dtype=float),
    positions: wp.array3d(dtype=float),
    velocities: wp.array3d(dtype=float),
    actions: wp.array3d(dtype=float),
):
    world = wp.tid()
    for j in range(qpos.shape[1]):
        positions[step[0], world, j] = qpos[world, j]
    for j in range(qvel.shape[1]):
        velocities[step[0], world, j] = qvel[world, j]
    for j in range(7):
        actions[step[0], world, j] = action[world, j]


@wp.kernel
def advance(step: wp.array(dtype=int)):
    step[0] = step[0] + 1


@wp.kernel
def apply_grip_effort(
    profile: wp.array(dtype=float),
    step: wp.array(dtype=int),
    limits: wp.array2d(dtype=wp.vec2),
):
    world = wp.tid()
    effort = profile[step[0]]
    limits[world, 6] = wp.vec2(-effort, effort)


def run_controls(controller, count, mode="home", actions=None, grip_efforts=None):
    """Executes a captured device loop and downloads one aligned trace."""
    engine = controller.engine
    with wp.ScopedDevice(engine.device):
        step = wp.zeros(1, dtype=int)
        home = wp.array(robot.HOME.astype(np.float32), dtype=float)
        positions = wp.zeros(
            (count, engine.nworld, engine.native_model.nq), dtype=float
        )
        velocities = wp.zeros(
            (count, engine.nworld, engine.native_model.nv), dtype=float
        )
        output = wp.zeros((count, engine.nworld, 7), dtype=float)
        if actions is not None:
            recorded_actions = wp.array(actions.astype(np.float32), dtype=float)
        if grip_efforts is not None:
            if len(grip_efforts) != count or not np.all(
                np.isfinite(grip_efforts) & (np.asarray(grip_efforts) > 0)
            ):
                raise ValueError("One positive finite grip effort per action required.")
            efforts = wp.array(grip_efforts, dtype=float)
        with wp.ScopedCapture() as capture:
            if grip_efforts is not None:
                wp.launch(
                    apply_grip_effort,
                    engine.model.actuator_forcerange.shape[0],
                    inputs=[efforts, step, engine.model.actuator_forcerange],
                )
            if mode == "home":
                wp.launch(
                    home_action,
                    dim=engine.nworld,
                    inputs=[controller.arm_target, home, controller.action],
                )
            elif mode == "replay":
                wp.launch(
                    replay_action,
                    dim=(engine.nworld, 7),
                    inputs=[recorded_actions, step, controller.action],
                )
            else:
                raise ValueError(mode)
            controller.launch_device_control_step()
            wp.launch(
                record_state,
                dim=engine.nworld,
                inputs=[
                    step,
                    engine.data.qpos,
                    engine.data.qvel,
                    controller.action,
                    positions,
                    velocities,
                    output,
                ],
            )
            wp.launch(advance, dim=1, inputs=[step])
        start = time.perf_counter()
        for _ in range(count):
            wp.capture_launch(capture.graph)
        wp.synchronize_device(engine.device)
        elapsed = time.perf_counter() - start
        engine.assert_healthy()
        return dict(
            qpos=positions.numpy(), qvel=velocities.numpy(), action=output.numpy()
        ), elapsed


def episode(initial, trace, world):
    """Converts post-step traces into T actions and T+1 simulator states."""
    count = len(trace["action"])
    if len(trace["qpos"]) != count or len(trace["qvel"]) != count:
        raise ValueError("One post-action state is required for every action.")
    result = {
        "action": trace["action"][:, world].copy(),
        "time": np.arange(count + 1) * 0.02,
    }
    for key in ("qpos", "qvel"):
        result[key] = np.concatenate(
            [initial[key][world : world + 1], trace[key][:, world]]
        )
    if not all(np.isfinite(value).all() for value in result.values()):
        raise ValueError("Nonfinite episode.")
    return result
