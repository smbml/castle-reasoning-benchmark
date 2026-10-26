"""Runtime control gripper."""

import mujoco
import numpy as np
import warp as wp


@wp.kernel
def shape_aperture(
    enabled: wp.array(dtype=int),
    move: wp.array(dtype=int),
    opening: wp.array2d(dtype=float),
    action: wp.array2d(dtype=float),
):
    world = wp.tid()
    if enabled[0] != 0:
        fraction = (action[world, 6] + 1.0) * 0.5
        minimum = opening[world, move[world]]
        action[world, 6] = 2.0 * (minimum + (1.0 - minimum) * fraction) - 1.0


def opening_closure(model, widths, margin=0.012):
    """Invert the ideal parallel-linkage aperture by forward kinematics only."""
    state = mujoco.MjData(model)
    site = model.site("ur5e/robotiq/pinch").id
    values = np.linspace(0, 1, 101)
    gaps = []
    for closure in values:
        theta = closure * 0.8
        state.qpos[6:14] = [theta, 0, theta, -theta] * 2
        mujoco.mj_forward(model, state)
        rotation = state.site_xmat[site].reshape(3, 3)
        inner = []
        for side in ("right", "left"):
            geom = model.geom(f"ur5e/robotiq/{side}_pad1").id
            corners = (
                np.asarray(
                    [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
                )
                * model.geom_size[geom]
            )
            world = (
                corners @ state.geom_xmat[geom].reshape(3, 3).T + state.geom_xpos[geom]
            )
            local = (world - state.site_xpos[site]) @ rotation
            inner.append(local[:, 1].min() if side == "right" else local[:, 1].max())
        gaps.append(inner[0] - inner[1])
    # Positive margin provides approach clearance; negative margin requests a
    # nominal holding preload. Neither is a grasp-success certificate.
    return np.interp(np.asarray(widths) + margin, gaps[::-1], values[::-1])


@wp.kernel
def shape_hold(
    enabled: wp.array(dtype=int),
    move: wp.array(dtype=int),
    opening: wp.array2d(dtype=float),
    holding: wp.array2d(dtype=float),
    action: wp.array2d(dtype=float),
):
    world = wp.tid()
    if enabled[0] != 0:
        fraction = (action[world, 6] + 1.0) * 0.5
        low = opening[world, move[world]]
        high = holding[world, move[world]]
        action[world, 6] = 2.0 * (low + (high - low) * fraction) - 1.0


@wp.kernel
def bound_targets(
    qpos: wp.array2d(dtype=float),
    action: wp.array2d(dtype=float),
    target: wp.array2d(dtype=float),
    limit: float,
    was_stopped: wp.array(dtype=int),
):
    world = wp.tid()
    stopped = True
    for joint in range(6):
        if action[world, joint] != 0.0:
            stopped = False
    for joint in range(6):
        actual = qpos[world, joint]
        if stopped and was_stopped[world] == 0:
            target[world, joint] = actual
        elif not stopped:
            target[world, joint] = wp.clamp(
                target[world, joint], actual - limit, actual + limit
            )
    was_stopped[world] = int(stopped)
