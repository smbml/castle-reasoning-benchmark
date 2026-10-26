"""Data assembly observations."""

import time

import numpy as np

import castle_reasoning.runtime.rendering.cameras as rendering_cameras


def render_learning_observations(
    model, qpos, qvel, directory, size=128, *, goal_qpos, goal_qvel
):
    """Writes lossless T+1 two-view RGB observations aligned with the action data."""
    started = time.perf_counter()
    temporary = directory / "rgb.tmp.npy"
    observations = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.uint8, shape=(len(qpos), 2, size, size, 3)
    )
    cameras = rendering_cameras.Cameras(model, size, size)
    try:
        for index in range(len(qpos)):
            observations[index] = cameras.frames(qpos[index], qvel[index])
        observations.flush()
        np.savez_compressed(
            directory / "observations.npz",
            rgb=observations,
            goal_rgb=cameras.frames(goal_qpos, goal_qvel),
            proprio=np.concatenate([qpos[:, :14], qvel[:, :14]], axis=1),
        )
    finally:
        cameras.close()
        del observations
        temporary.unlink(missing_ok=True)
    return time.perf_counter() - started
