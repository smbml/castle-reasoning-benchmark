"""Runtime execution success."""

import mujoco
import numpy as np

import castle_reasoning.runtime.diagnostics.timing as timing
import castle_reasoning.runtime.simulation.robot as robot


def orientation_error(quaternions, sizes):
    """Radians to valid upright yaw symmetries for each actual XY footprint."""
    references = np.asarray(
        [[np.cos(k * np.pi / 4), 0, 0, np.sin(k * np.pi / 4)] for k in range(4)]
    )
    similarity = np.abs(np.asarray(quaternions) @ references.T)
    rectangle = ~np.isclose(np.asarray(sizes)[:, 0], np.asarray(sizes)[:, 1])
    similarity[..., rectangle, 1::2] = 0
    return 2 * np.arccos(np.clip(similarity.max(axis=-1), 0, 1))


def waypoint_check(poses, goals, sizes, part):
    """Check the commanded part; retain whole-scene deviations as diagnostics."""
    distances = np.linalg.norm(poses[:, :3] - goals, axis=-1)
    angles = np.rad2deg(orientation_error(poses[:, 3:], sizes))
    return {
        "position_error_m": float(distances[part]),
        "orientation_error_deg": float(angles[part]),
        "scene_position_error_m": float(distances.max()),
        "scene_orientation_error_deg": float(angles.max()),
        "passed": bool(distances[part] <= 0.005 and angles[part] <= 5),
    }


@timing.measured('audit.verify_hold')
def verify_hold(qpos, qvel, goals, sizes, position_tolerance=0.005):
    """Check a complete final second against fixed requested cuboid poses."""
    qpos, qvel = np.asarray(qpos), np.asarray(qvel)
    count = len(sizes)
    if (
        len(qpos) < 51
        or qpos.shape[1] != 14 + count * 7
        or qvel.shape != (len(qpos), 14 + count * 6)
    ):
        raise ValueError("Need 51 aligned states with matching object coordinates.")
    poses = qpos[-51:, 14:].reshape(51, count, 7)
    velocities = qvel[-51:, 14:].reshape(51, count, 6)
    radius = np.linalg.norm(sizes, axis=1) / 2
    metrics = {
        "position_error_m": float(
            np.max(np.linalg.norm(poses[..., :3] - goals, axis=-1))
        ),
        "orientation_error_deg": float(
            np.rad2deg(orientation_error(poses[..., 3:], sizes)).max()
        ),
        "drift_m": float(
            np.max(np.linalg.norm(poses[..., :3] - poses[0, :, :3], axis=-1))
        ),
        "surface_speed_m_s": float(
            np.max(
                np.linalg.norm(velocities[..., :3], axis=-1)
                + radius * np.linalg.norm(velocities[..., 3:], axis=-1)
            )
        ),
        "home_error_rad": float(np.max(np.abs(qpos[-51:, :6] - robot.HOME))),
        "gripper_error_rad": float(np.max(np.abs(qpos[-51:, 6:14]))),
    }
    metrics["pose_valid"] = bool(
        np.isfinite(qpos).all()
        and np.isfinite(qvel).all()
        and metrics["position_error_m"] <= position_tolerance
        and metrics["orientation_error_deg"] <= 5
        and metrics["home_error_rad"] <= 0.04
        and metrics["gripper_error_rad"] <= 0.08
    )
    metrics["settled"] = bool(
        metrics["drift_m"] <= 0.002 and metrics["surface_speed_m_s"] <= 0.05
    )
    metrics["passed"] = metrics["pose_valid"] and metrics["settled"]
    return metrics


@timing.measured('audit.supports')
def contact_report(model, qpos, supports):
    """Require every declared support, including both piers under a lintel."""
    state = mujoco.MjData(model)
    count = (model.nq - 14) // 7
    object_ids = {model.body(f"object_{i}").id: i for i in range(count)}
    floor = model.geom("floor").id
    failures = set()
    max_gap = 0.0
    gap_tolerance = 1e-5
    for frame in qpos[-51:]:
        state.qpos[:] = frame
        mujoco.mj_forward(model, state)
        observed = set()
        for c in state.contact:
            if c.dist > gap_tolerance:
                continue
            g1, g2 = int(c.geom1), int(c.geom2)
            b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
            o1, o2 = object_ids.get(b1), object_ids.get(b2)
            if o1 is not None and o2 is not None:
                observed.update(((o1, o2), (o2, o1)))
            for obj, geom, body in ((o1, g2, b2), (o2, g1, b1)):
                if obj is None:
                    continue
                if geom == floor:
                    observed.add((obj, -1))
                elif (
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or ""
                ).startswith("ur5e/"):
                    failures.add("robot_object_contact")
        for support in supports:
            if tuple(support) not in observed:
                # Native contact lists omit even micron-sized positive gaps in
                # otherwise supported float32 Warp states. Query signed distance
                # explicitly instead of changing simulation collision margins.
                upper, lower = support
                g1 = model.geom(f"object_{upper}").id
                g2 = floor if lower == -1 else model.geom(f"object_{lower}").id
                gap = mujoco.mj_geomDistance(model, state, g1, g2, 0.01, None)
                max_gap = max(max_gap, float(gap))
                if gap > gap_tolerance:
                    failures.add(f"missing_support_{upper}_{lower}")
    return {
        "passed": not failures,
        "failures": sorted(failures),
        "max_missing_contact_gap_m": max_gap,
        "gap_tolerance_m": gap_tolerance,
    }
