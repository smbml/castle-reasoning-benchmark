"""Tests helpers qualification_columns."""

import dataclasses
import hashlib

import mujoco
import numpy as np

import castle_reasoning.runtime.simulation.robot as robot


@dataclasses.dataclass(frozen=True)
class Target:
    """A candidate target, before physical feasibility certification."""

    positions: np.ndarray
    removal_order: list[int]
    supports: list[tuple[int, int]]
    heights: list[int]
    fingerprint: str
    seed: int

    @property
    def assembly_order(self) -> list[int]:
        return self.removal_order[::-1]


def sample_target(seed: int, candidate: int, block_count: int = 8) -> Target:
    """Samples separated cube columns with explicit support dependencies."""
    if block_count not in (2, 4, 8):
        raise ValueError("Qualification supports two, four, or eight cubes.")
    rng = np.random.default_rng(np.random.SeedSequence([seed, candidate]))
    columns = (
        1 if block_count == 2 else int(rng.integers(2, min(4, block_count - 1) + 1))
    )
    heights = np.ones(columns, dtype=int)
    for _ in range(block_count - columns):
        valid = np.flatnonzero(heights < 4)
        heights[rng.choice(valid)] += 1
    if columns <= 3:
        xy = np.column_stack((np.full(columns, 0.425), np.linspace(-0.1, 0.1, columns)))
        if columns == 1:
            xy[0] = [0.425, 0]
    else:
        xy = np.asarray([[0.37, -0.06], [0.48, -0.06], [0.37, 0.06], [0.48, 0.06]])
    xy += rng.uniform(-0.01, 0.01, size=(1, 2))
    xy += rng.uniform(-0.003, 0.003, size=xy.shape)
    identity = rng.permutation(block_count)
    positions = np.empty((block_count, 3))
    supports = []
    cursor = 0
    for col, height in enumerate(heights):
        for level in range(height):
            block = int(identity[cursor])
            positions[block] = [*xy[col], 0.02 + 0.04 * level]
            if level:
                supports.append((block, int(identity[cursor - 1])))
            cursor += 1
    # Independent columns may be removed in different orders at equal height.
    ties = rng.uniform(size=block_count)
    order = np.lexsort((ties, -positions[:, 2])).tolist()
    fingerprint = hashlib.sha256(np.round(positions, 5).tobytes()).hexdigest()
    return Target(positions, order, supports, heights.tolist(), fingerprint, seed)


def parking_positions(block_count: int) -> np.ndarray:
    """Separated staging slots outside the central assembly footprint."""
    slots = np.asarray(
        [[x, y, 0.02] for y in (-0.27, -0.17, 0.17, 0.27) for x in (0.315, 0.535)]
    )
    return slots[:block_count].copy()


def video_indices(count: int) -> list[int]:
    """Returns the user-facing, one-based odd instance numbers."""
    return list(range(1, count + 1, 2))


def verify_hold(
    qpos: np.ndarray,
    qvel: np.ndarray,
    goals: np.ndarray,
    position_tolerance: float = 0.005,
) -> dict:
    """Checks every frame of a one-second terminal hold; contacts checked separately."""
    count = len(goals)
    if qpos.ndim != 2 or qpos.shape[0] < 51 or qpos.shape[1] != 14 + count * 7:
        raise ValueError(
            "Need at least 51 consecutive states and matching object coordinates."
        )
    if qvel.shape != (len(qpos), 14 + count * 6):
        raise ValueError("Velocity/state length or object count mismatch.")
    poses = qpos[:, 14:].reshape(len(qpos), count, 7)
    positions = poses[:, :, :3]
    finite = bool(np.isfinite(qpos).all() and np.isfinite(qvel).all())
    error = float(np.max(np.linalg.norm(positions - goals, axis=-1)))
    home_error = float(np.max(np.abs(qpos[:, :6] - robot.HOME)))
    velocities = qvel[:, 14:].reshape(len(qvel), count, 6)
    linear_speed = float(np.max(np.linalg.norm(velocities[:, :, :3], axis=-1)))
    angular_speed = float(np.max(np.linalg.norm(velocities[:, :, 3:], axis=-1)))
    # Stable contact can have brief angular solver chatter. Check physical
    # surface speed and actual pose drift, with separate angular units.
    surface_speed = float(
        np.max(
            np.linalg.norm(velocities[:, :, :3], axis=-1)
            + 0.035 * np.linalg.norm(velocities[:, :, 3:], axis=-1)
        )
    )
    quaternion_drift = float(
        np.max(
            np.minimum(
                np.linalg.norm(poses[:, :, 3:] - poses[0, :, 3:], axis=-1),
                np.linalg.norm(poses[:, :, 3:] + poses[0, :, 3:], axis=-1),
            )
        )
    )
    gripper_error = float(np.max(np.abs(qpos[:, 6:14])))
    drift = float(np.max(np.linalg.norm(positions - positions[0], axis=-1)))
    # Compare full orientation to the four upright quarter-turn symmetries.
    reference = np.asarray(
        [[np.cos(k * np.pi / 4), 0.0, 0.0, np.sin(k * np.pi / 4)] for k in range(4)]
    )
    dot = np.abs(poses[:, :, 3:] @ reference.T)
    orientation_error = float(
        np.max(2 * np.arccos(np.clip(dot.max(axis=-1), 0.0, 1.0)))
    )
    tilt = float(np.max(2 * np.sum(poses[:, :, 4:6] ** 2, axis=-1)))
    passed = finite and error <= position_tolerance and home_error <= 0.04
    passed = (
        passed
        and surface_speed <= 0.05
        and quaternion_drift <= 0.005
        and gripper_error <= 0.08
    )
    passed = passed and drift <= 0.002 and orientation_error <= np.deg2rad(5.0)
    return dict(
        passed=bool(passed),
        finite=finite,
        max_position_error_m=error,
        max_home_error_rad=home_error,
        max_surface_speed_m_s=surface_speed,
        max_linear_speed_m_s=linear_speed,
        max_angular_speed_rad_s=angular_speed,
        max_quaternion_drift=quaternion_drift,
        max_gripper_joint_rad=gripper_error,
        max_hold_drift_m=drift,
        max_upright_error=tilt,
        max_orientation_error_degrees=float(np.rad2deg(orientation_error)),
        hold_seconds=(len(qpos) - 1) * 0.02,
    )


def contact_report(model, qpos, goals, supports):
    """Requires released, supported final objects throughout the terminal hold."""
    data = mujoco.MjData(model)
    floor = model.geom("floor").id
    body_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or ""
        for i in range(model.nbody)
    ]
    object_by_body = {model.body(f"object_{i}").id: i for i in range(len(goals))}
    failures = set()
    support_map = dict(supports)
    for state in qpos[-51:]:
        data.qpos[:] = state
        mujoco.mj_forward(model, data)
        observed = set()
        for contact in data.contact:
            if contact.dist > 0.0005:
                continue
            g1, g2 = int(contact.geom1), int(contact.geom2)
            b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
            o1, o2 = object_by_body.get(b1), object_by_body.get(b2)
            if (
                o1 is not None
                and body_names[b2].startswith("ur5e/")
                or o2 is not None
                and body_names[b1].startswith("ur5e/")
            ):
                failures.add("robot_object_contact")
            if o1 is not None and g2 == floor:
                observed.add((o1, -1))
            if o2 is not None and g1 == floor:
                observed.add((o2, -1))
            if o1 is not None and o2 is not None:
                observed.add((o1, o2))
                observed.add((o2, o1))
        for obj in range(len(goals)):
            expected = support_map.get(obj, -1)
            if (obj, expected) not in observed:
                failures.add(f"missing_support_{obj}_{expected}")
    return dict(passed=not failures, failures=sorted(failures), checked_frames=51)
