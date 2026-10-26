"""Data generation waypoints."""

import numpy as np

import castle_reasoning.data.generation.edge_staging as edge_staging
import castle_reasoning.data.generation.geometry as geometry
import castle_reasoning.data.generation.nested_staging as nested_staging


class PlanningFailure(ValueError):
    """The declared teacher could not plan this unchanged target."""


def vertical_clear(parts, moving, remaining):
    """Check the entire upward extrusion, allowing boundary-only contact."""
    part = parts[moving]
    return not any(
        other != moving
        and min(geometry.overlap(part, parts[other])[:2]) > 0
        and parts[other].end[2] > part.origin[2]
        for other in remaining
    )


def removal_order(parts, max_nodes=4096, diagnostics=None, fixed=frozenset()):
    """Search stable subsets; optionally record one deepest terminal dead end.

    `fixed` parts (a foundation no primitive can move) stay in place: the
    search ends when only they remain, and they are never removed.
    """
    fixed = frozenset(fixed)
    visited = set()
    counts = {"expanded": 0, "static_checks": 0, "blocked": 0}
    if diagnostics is not None:
        diagnostics.clear()

    def search(remaining, prefix):
        if remaining <= fixed:
            return []
        if remaining in visited:
            return None
        if counts["expanded"] >= max_nodes:
            raise PlanningFailure("removal_search_budget")
        visited.add(remaining)
        counts["expanded"] += 1
        rejections = []
        eligible = False
        for moving in sorted(remaining - fixed, key=lambda i: (-parts[i].end[2], i)):
            if not vertical_clear(parts, moving, remaining):
                counts["blocked"] += 1
                rejections.append({"part": moving, "reason": "vertical_collision"})
                continue
            after = remaining - {moving}
            counts["static_checks"] += 1
            if after and not geometry.equilibrium([parts[i] for i in sorted(after)]):
                rejections.append({"part": moving, "reason": "static_equilibrium"})
                continue
            eligible = True
            continuation = search(after, prefix + [moving])
            if continuation is not None:
                return [moving, *continuation]
        if (
            diagnostics is not None
            and not eligible
            and len(prefix) >= len(diagnostics.get("prefix", []))
        ):
            diagnostics.update(
                prefix=prefix, remaining=sorted(remaining), rejections=rejections
            )
        return None

    result = search(frozenset(range(len(parts))), [])
    if diagnostics is not None:
        diagnostics["counts"] = counts.copy()
    if result is None:
        raise PlanningFailure("no_stable_vertical_disassembly")
    return result, counts


def overlaps(positions, sizes, gap=0.0):
    """Report interior XY overlap including the requested separation margin."""
    return any(
        np.all(
            np.abs(positions[i, :2] - positions[j, :2])
            < (sizes[i, :2] + sizes[j, :2]) / 2 + gap - 1e-8
        )
        for i in range(len(positions))
        for j in range(i)
    )


def construction_removal_order(parts, construction_order):
    """Validate a stored placement witness before using its reverse for removal."""
    if any(
        not isinstance(i, (int, np.integer)) or isinstance(i, bool)
        for i in construction_order
    ) or sorted(construction_order) != list(range(len(parts))):
        raise PlanningFailure("invalid_construction_witness_permutation")
    prefix = []
    for moving in construction_order:
        prefix.append(moving)
        if not vertical_clear(parts, moving, prefix):
            raise PlanningFailure(
                f"construction_witness_vertical_collision_part_{moving}"
            )
        if not geometry.equilibrium([parts[i] for i in prefix]):
            raise PlanningFailure(
                f"construction_witness_static_infeasible_part_{moving}"
            )
    return list(reversed(construction_order)), {
        "expanded": 0,
        "static_checks": len(parts),
        "blocked": 0,
    }


def parking_slots(goals, sizes, bounds, gap=0.025):
    """Pack identity-oriented parts outside the assembly footprint on the table."""
    x0, x1, y0, y1 = bounds
    target_low = np.min(goals[:, :2] - sizes[:, :2] / 2, axis=0)
    target_high = np.max(goals[:, :2] + sizes[:, :2] / 2, axis=0)
    target_center = (target_low + target_high) / 2
    target_size = target_high - target_low
    parked = np.zeros_like(goals)
    placed = []
    for i in sorted(range(len(sizes)), key=lambda k: -np.prod(sizes[k, :2])):
        for y in np.arange(y0 + sizes[i, 1] / 2, y1 - sizes[i, 1] / 2 + 1e-8, 0.01):
            for x in np.arange(x0 + sizes[i, 0] / 2, x1 - sizes[i, 0] / 2 + 1e-8, 0.01):
                center = np.array([x, y, sizes[i, 2] / 2])
                if np.all(
                    np.abs(center[:2] - target_center)
                    < (sizes[i, :2] + target_size) / 2 + 0.045
                ):
                    continue
                if any(
                    np.all(
                        np.abs(center[:2] - parked[j, :2])
                        < (sizes[i, :2] + sizes[j, :2]) / 2 + gap
                    )
                    for j in placed
                ):
                    continue
                parked[i] = center
                placed.append(i)
                break
            if i in placed:
                break
        if i not in placed:
            raise PlanningFailure(f"parking_capacity_part_{i}")
    return parked


def stacked_parking_slots(
    goals, sizes, bounds, order, max_height=0.25, gap=0.025, reserved_regions=()
):
    """Place in removal order onto concentric stacks, then unused floor slots.

    Each stack has a shared XY center, so its uniform-density loads lie inside
    every horizontal support patch, including centered overhangs. This is a
    static witness; robot access and dynamic stability require execution checks.
    """
    x0, x1, y0, y1 = bounds
    low = np.min(goals[:, :2] - sizes[:, :2] / 2, axis=0) - 0.045
    high = np.max(goals[:, :2] + sizes[:, :2] / 2, axis=0) + 0.045
    parked = np.zeros_like(goals)
    stacks, supports = [], []

    def valid(center, part, stack):
        radius = sizes[part] / 2
        if any(
            np.all(center[:2] + radius[:2] + gap > np.array(rectangle[:2]))
            and np.all(center[:2] - radius[:2] - gap < np.array(rectangle[2:]))
            for rectangle in reserved_regions
        ):
            return False
        if (
            np.any(center[:2] - radius[:2] < [x0 - 1e-8, y0 - 1e-8])
            or np.any(center[:2] + radius[:2] > [x1 + 1e-8, y1 + 1e-8])
            or center[2] + radius[2] > max_height + 1e-8
        ):
            return False
        if np.all(center[:2] + radius[:2] > low) and np.all(
            center[:2] - radius[:2] < high
        ):
            return False
        return not any(
            np.all(
                np.abs(center[:2] - parked[other, :2])
                < radius[:2] + sizes[other, :2] / 2 + gap - 1e-8
            )
            for other_stack in stacks
            if other_stack is not stack
            for other in other_stack
        )

    for part in order:
        candidates = []
        for stack in sorted(
            stacks, key=lambda s: parked[s[-1], 2] + sizes[s[-1], 2] / 2
        ):
            parent = stack[-1]
            center = parked[parent].copy()
            center[2] += (sizes[parent, 2] + sizes[part, 2]) / 2
            candidates.append((center, stack, parent))
        candidates.extend(
            (np.array([x, y, sizes[part, 2] / 2]), None, -1)
            for y in np.arange(
                y0 + sizes[part, 1] / 2, y1 - sizes[part, 1] / 2 + 1e-8, 0.01
            )
            for x in np.arange(
                x0 + sizes[part, 0] / 2, x1 - sizes[part, 0] / 2 + 1e-8, 0.01
            )
        )
        for center, stack, parent in candidates:
            if not valid(center, part, stack):
                continue
            if stack is None:
                stack = []
                stacks.append(stack)
            parked[part] = center
            stack.append(part)
            supports.append([int(part), int(parent)])
            break
        else:
            raise PlanningFailure(f"stack_parking_capacity_part_{part}")
    return parked, supports


def plan(
    parts,
    meters_per_grid=0.01,
    max_grasp_width=0.07,
    parking_bounds=(0.30, 0.60, -0.29, 0.29),
    max_search_nodes=512,
    staging="stacks",
    max_stack_height=0.25,
    construction_order=None,
    parking_gap=0.025,
    stack_layout="concentric",
    fixed=(),
):
    """Return complete object-goal sequences; robot controls must be executed.

    `fixed` parts remain at their goal as a foundation: they are never removed,
    staged or placed, and their parking pose is their goal.
    """
    fixed = sorted(set(fixed))
    movable = [i for i in range(len(parts)) if i not in fixed]
    if stack_layout not in ("concentric", "nested"):
        raise ValueError(f"Unknown stack layout: {stack_layout}")
    if not np.isfinite(parking_gap) or parking_gap < 0:
        raise ValueError("Parking gap must be finite and nonnegative.")
    if not parts or not geometry.equilibrium(parts):
        raise PlanningFailure("target_static_infeasible")
    sizes = np.asarray([p.size for p in parts], dtype=float) * meters_per_grid
    if staging != "edge" and np.any(
        np.min(sizes[movable, :2], axis=1) > max_grasp_width + 1e-9
    ):
        raise PlanningFailure("grasp_width_exceeds_top_down_aperture")
    low = np.min([p.origin for p in parts], axis=0)
    high = np.max([p.end for p in parts], axis=0)
    goals = np.asarray([p.center for p in parts]) * meters_per_grid
    center = [0.40 if staging == "edge" else 0.425, 0.0]
    goals[:, :2] += np.array(center) - (low[:2] + high[:2]) * meters_per_grid / 2
    if (
        high[2] * meters_per_grid > 0.4
        or np.any(goals[:, 0] - sizes[:, 0] / 2 < 0.20)
        or np.any(goals[:, 0] + sizes[:, 0] / 2 > 0.65)
    ):
        raise PlanningFailure("target_workspace_envelope")
    if construction_order is None:
        order, counts = removal_order(parts, max_nodes=max_search_nodes, fixed=fixed)
    else:
        order, counts = construction_removal_order(parts, construction_order)
    edge_access = None
    if staging in ("stacks", "edge") and (
        not np.isfinite(max_stack_height) or max_stack_height <= 0
    ):
        raise ValueError("Stack height must be finite and positive.")
    if staging == "edge":
        parked, parking_supports, edge_access = edge_staging.arrange(
            goals,
            sizes,
            order,
            max_grasp_width,
            max_stack_height,
            gap=parking_gap,
            stack_layout=stack_layout,
            fixed=fixed,
        )
    elif staging == "stacks":
        allocator = (
            nested_staging.slots if stack_layout == "nested" else stacked_parking_slots
        )
        parked, parking_supports = allocator(
            goals, sizes, parking_bounds, order, max_stack_height, gap=parking_gap
        )
    elif staging == "flat":
        parked = parking_slots(goals, sizes, parking_bounds, gap=parking_gap)
        parking_supports = [[i, -1] for i in order]
    else:
        raise ValueError(f"Unknown staging mode: {staging}")
    for part in fixed:
        parked[part] = goals[part]
        parking_supports.append([part, -1])
    states = [goals.copy()]
    transports = []
    hover = (
        max(
            float(np.max(goals[:, 2] + sizes[:, 2] / 2)),
            float(np.max(parked[:, 2] + sizes[:, 2] / 2)),
        )
        + 0.08
    )
    for i in order:
        source, destination = states[-1][i].copy(), parked[i].copy()
        above_source, above_destination = source.copy(), destination.copy()
        above_source[2] = above_destination[2] = hover + sizes[i, 2] / 2
        transports.append(
            {
                "part": i,
                "centers": np.stack(
                    [source, above_source, above_destination, destination]
                ).tolist(),
            }
        )
        state = states[-1].copy()
        state[i] = parked[i]
        states.append(state)
    contacts = geometry.supports(parts)
    supports = [[c.upper, c.lower] for c in contacts]
    supports += [[i, -1] for i, p in enumerate(parts) if p.origin[2] == 0]
    return {
        "meters_per_grid": meters_per_grid,
        "sizes": sizes.tolist(),
        "target": goals.tolist(),
        "parking": parked.tolist(),
        "parking_supports": parking_supports,
        "staging": staging,
        "stack_layout": stack_layout,
        "max_stack_height": max_stack_height,
        "parking_gap": parking_gap,
        "removal_order": order,
        "assembly_order": order[::-1],
        **({"fixed_parts": fixed} if fixed else {}),
        "disassembly_goals": np.asarray(states).tolist(),
        "assembly_goals": np.asarray(states[::-1]).tolist(),
        "object_transport_waypoints": transports,
        "supports": supports,
        "search": counts,
        "order_source": "subset_search"
        if construction_order is None
        else "construction_witness",
        "robot_execution": "unverified",
        "max_grasp_width": max_grasp_width,
        "parking_bounds": edge_access["parking_bounds"]
        if edge_access
        else list(parking_bounds),
        "max_search_nodes": max_search_nodes,
        **({"edge_access": edge_access} if edge_access is not None else {}),
    }
