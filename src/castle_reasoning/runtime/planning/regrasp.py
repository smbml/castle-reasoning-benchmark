"""Runtime planning regrasp."""

import dataclasses

import mujoco
import numpy as np
from scipy import spatial

import castle_reasoning.runtime.execution.success as success
import castle_reasoning.runtime.planning.arm as arm
import castle_reasoning.runtime.planning.manipulation as manipulation


@dataclasses.dataclass
class Regrasp:
    """Two checked manipulations; the second is replanned after real placement."""

    waypoint: np.ndarray
    first: manipulation.Manipulation
    second: manipulation.Manipulation
    supports: tuple[int, ...]
    support_margin_m: float
    anchor: str


def support_margin(planner, qpos, part):
    """Screen near-upright COM support in conservative projected contact patches.

    This is a local gravity screen, not a guarantee of dynamic or whole-structure
    stability. Physical guards and the complete episode replay remain required.
    """
    if not np.isfinite(qpos).all():
        return -float("inf"), ()
    poses = np.asarray(qpos)[14:].reshape(-1, 7)
    angles = success.orientation_error(poses[:, 3:], planner.sizes)
    supports = planner._source_supports(qpos, part)
    if not supports or angles[part] > np.deg2rad(1):
        return -float("inf"), supports
    scene = planner.scene
    floor = scene.model.geom("floor").id
    part_by_geom = {geom: index for index, geom in scene.objects.items()}
    radii = planner.sizes[:, :2] / 2
    rotation_bounds = np.linalg.norm(planner.sizes, axis=1) * angles / 2
    low = poses[part, :2] - radii[part] + rotation_bounds[part]
    high = poses[part, :2] + radii[part] - rotation_bounds[part]
    corners = []
    for geom in supports:
        if geom == floor:
            if scene.model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_PLANE:
                base_low, base_high = low, high
            else:
                center = scene.data.geom_xpos[geom, :2]
                half = scene.model.geom_size[geom, :2]
                base_low, base_high = center - half, center + half
        else:
            other = part_by_geom[geom]
            if angles[other] > np.deg2rad(1):
                return -float("inf"), supports
            base_low = poses[other, :2] - radii[other] + rotation_bounds[other]
            base_high = poses[other, :2] + radii[other] - rotation_bounds[other]
        patch_low, patch_high = np.maximum(low, base_low), np.minimum(high, base_high)
        if np.all(patch_high > patch_low):
            corners.extend(
                [x, y]
                for x in (patch_low[0], patch_high[0])
                for y in (patch_low[1], patch_high[1])
            )
    if len(corners) < 3:
        return -float("inf"), supports
    try:
        hull = spatial.ConvexHull(np.unique(corners, axis=0))
    except spatial.QhullError:
        return -float("inf"), supports
    distances = -(hull.equations[:, :2] @ poses[part, :2] + hull.equations[:, 2])
    return float(distances.min()), supports


def _clearance(planner, part, supports):
    """Measure unrelated-object clearance at the scene's current state."""
    scene = planner.scene
    return min(
        (
            mujoco.mj_geomDistance(
                scene.model, scene.data, scene.objects[part], geom, 0.04, None
            )
            for other, geom in scene.objects.items()
            if other != part and geom not in supports
        ),
        default=0.04,
    )


def candidates(planner, qpos, part):
    """Prefer supported shifts that open space around the selected object."""
    start = 14 + 7 * part
    original_margin, original_supports = support_margin(planner, qpos, part)
    if original_margin < 0.001:
        return []
    original_clearance = _clearance(planner, part, original_supports)
    options = []
    # Shifts long enough to clear a pad beside a flush neighbor (about 2 cm)
    # are needed when only a side grasp can lift the part; every shift must
    # still keep the same supports with at least 1 mm of support margin.
    for distance in (0.003, 0.006, 0.008, 0.012, 0.016, 0.020, 0.025):
        for dx, dy in ((distance, 0), (-distance, 0), (0, distance), (0, -distance)):
            planner.check_budget()
            state = np.array(qpos).copy()
            state[start : start + 2] += [dx, dy]
            margin, supports = support_margin(planner, state, part)
            if supports != original_supports or margin < 0.001:
                continue
            clearance = _clearance(planner, part, supports)
            options.append(
                (
                    original_clearance - clearance,
                    distance,
                    state[start : start + 3].copy(),
                    supports,
                )
            )
    return sorted(options, key=lambda value: value[:2])


def plan(planner, qpos, part, destination, deadline=float("inf")):
    """Certify both transfers before executing anything; never change target data."""
    with planner.budget.scope(2 * planner.planning_seconds, deadline):
        return _plan(planner, qpos, part, destination, deadline)


def _plan(planner, qpos, part, destination, deadline):
    planner.check_budget()
    if support_margin(planner, qpos, part)[0] < 0.001:
        raise arm.PlanningError("no_supported_regrasp")
    options = [(option, "source") for option in candidates(planner, qpos, part)]
    target_state = np.array(qpos).copy()
    start = 14 + 7 * part
    target_state[start : start + 7] = np.r_[destination, 1, 0, 0, 0]
    options.extend(
        (option, "destination") for option in candidates(planner, target_state, part)
    )
    # Rank improvement relative to each anchor, so open staging does not crowd
    # out small shifts that create access at an obstructed destination.
    options.sort(key=lambda item: item[0][:2])
    budget_error = None
    for (_, _, waypoint, supports), anchor in options:
        planner.check_budget()
        try:
            first = planner.plan(qpos, part, waypoint, deadline=deadline)
            middle = first.segments[-1].path[-1]
            margin, observed_supports = support_margin(planner, middle, part)
            if observed_supports != supports or margin < 0.001:
                continue
            second = planner.plan(middle, part, destination, deadline=deadline)
            return Regrasp(waypoint, first, second, supports, margin, anchor)
        except arm.PlanningBudgetError as error:
            # One transfer may exhaust its allowance before the enclosing
            # two-transfer search does. Keep trying only within the parent limit.
            planner.check_budget()
            budget_error = error
        except arm.PlanningError:
            continue
    if budget_error is not None:
        raise budget_error
    raise arm.PlanningError("no_supported_regrasp")
