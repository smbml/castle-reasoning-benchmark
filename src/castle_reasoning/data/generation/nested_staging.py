"""Data generation nested_staging."""

import numpy as np

import castle_reasoning.data.generation.waypoints as waypoints


def slots(goals, sizes, bounds, order, max_height=0.25, gap=0.025, reserved_regions=()):
    """Form footprint-nested stacks, then reserve the largest floor areas first.

    Parts are appended in removal order. Only multi-part stacks are subject to
    the 3:1 height/base-width bound; a naturally slender single part is allowed.
    This is geometric staging, not a robot-access or dynamic-stability proof.
    """
    x0, x1, y0, y1 = bounds
    low = np.min(goals[:, :2] - sizes[:, :2] / 2, axis=0) - 0.045
    high = np.max(goals[:, :2] + sizes[:, :2] / 2, axis=0) + 0.045
    parked = np.zeros_like(goals)
    stacks = []
    supports = []
    for part in order:
        if sizes[part, 2] > max_height + 1e-8:
            raise waypoints.PlanningFailure(f"stack_height_part_{part}")
        choices = []
        for stack in stacks:
            parent = stack[-1]
            height = parked[parent, 2] + sizes[parent, 2] / 2 + sizes[part, 2]
            if (
                np.all(sizes[part, :2] <= sizes[parent, :2] + 1e-8)
                and height <= max_height + 1e-8
                and height <= 3 * min(sizes[stack[0], :2]) + 1e-8
            ):
                choices.append(
                    (
                        float(np.prod(sizes[parent, :2]) - np.prod(sizes[part, :2])),
                        height,
                        stack,
                    )
                )
        if choices:
            _, _, stack = min(choices, key=lambda x: x[:2])
            parent = stack[-1]
            parked[part, 2] = (
                parked[parent, 2] + (sizes[parent, 2] + sizes[part, 2]) / 2
            )
        else:
            stack = []
            stacks.append(stack)
            parent = -1
            parked[part, 2] = sizes[part, 2] / 2
        stack.append(part)
        supports.append([int(part), int(parent)])
    occupied = []
    for stack in sorted(
        stacks, key=lambda s: float(np.prod(sizes[s[0], :2])), reverse=True
    ):
        base = stack[0]
        radius = sizes[base, :2] / 2
        found = False
        for y in np.arange(y0 + radius[1], y1 - radius[1] + 1e-8, 0.01):
            for x in np.arange(x0 + radius[0], x1 - radius[0] + 1e-8, 0.01):
                center = np.array([x, y])
                if np.all(center + radius > low) and np.all(center - radius < high):
                    continue
                if any(
                    np.all(center + radius + gap > np.array(r[:2]))
                    and np.all(center - radius - gap < np.array(r[2:]))
                    for r in reserved_regions
                ):
                    continue
                if any(
                    np.all(abs(center - other) < radius + half + gap - 1e-8)
                    for other, half in occupied
                ):
                    continue
                found = True
                break
            if found:
                break
        if not found:
            raise waypoints.PlanningFailure(f"nested_parking_capacity_part_{base}")
        parked[stack, :2] = center
        occupied.append((center, radius))
    return parked, supports
