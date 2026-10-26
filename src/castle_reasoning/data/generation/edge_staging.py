"""Data generation edge_staging."""

import numpy as np

import castle_reasoning.data.generation.nested_staging as nested_staging
import castle_reasoning.data.generation.waypoints as waypoints

TABLE_BOUNDS = (-0.15, 0.85, -0.25, 0.40)


PARKING_BOUNDS = (0.20, 0.76, -0.22, 0.37)


def arrange(
    goals,
    sizes,
    order,
    aperture,
    max_height,
    gap=0.025,
    *,
    stack_layout="concentric",
    fixed=(),
):
    """Reserve a ground plate's access before packing ordinary part stacks."""
    if stack_layout not in ("concentric", "nested"):
        raise ValueError(f"Unknown stack layout: {stack_layout}")
    allocator = (
        nested_staging.slots
        if stack_layout == "nested"
        else waypoints.stacked_parking_slots
    )
    plates = np.flatnonzero(np.min(sizes[:, :2], axis=1) > aperture + 1e-9)
    # A fixed foundation stays at its goal and needs no edge access.
    plates = np.setdiff1d(plates, np.asarray(fixed, dtype=int))
    if len(plates) > 1 or any(
        abs(goals[i, 2] - sizes[i, 2] / 2) > 1e-8 for i in plates
    ):
        raise waypoints.PlanningFailure("edge_access_requires_single_ground_plate")
    if any(not 0.008 <= sizes[i, 2] <= 0.020001 for i in plates):
        raise waypoints.PlanningFailure("edge_plate_thickness_outside_qualified_range")
    front = TABLE_BOUNDS[2]
    lanes, rest, exposed = [], {}, {}
    for part in plates.tolist():
        radius = sizes[part, :2] / 2
        if radius[1] - 0.055 < 0.025 - 1e-8:
            raise waypoints.PlanningFailure("edge_plate_insufficient_support_margin")
        if order[-1] != part:
            raise waypoints.PlanningFailure("edge_plate_must_be_removed_last")
        # Reuse the cleared assembly lane only after all supported parts leave.
        # Keeping the same X also keeps the side-facing wrist within arm reach.
        x = float(goals[part, 0])
        rest[part] = np.array([x, front + radius[1] + 0.025, sizes[part, 2] / 2])
        exposed[part] = [
            float(goals[part, 0]),
            front + radius[1] - 0.055,
            float(goals[part, 2]),
        ]
        for center in (goals[part], rest[part]):
            lanes.append(
                [
                    float(center[0] - radius[0] - 0.035),
                    front - 0.30,
                    float(center[0] + radius[0] + 0.035),
                    float(center[1] + radius[1] + 0.07),
                ]
            )
    ordinary = [i for i in order if i not in plates]
    parked, supports = allocator(
        goals,
        sizes,
        PARKING_BOUNDS,
        ordinary,
        max_height,
        gap=gap,
        reserved_regions=lanes,
    )
    for part, center in rest.items():
        parked[part] = center
        supports.append([int(part), -1])
    return (
        parked,
        supports,
        {
            "table_bounds": list(TABLE_BOUNDS),
            "plates": plates.tolist(),
            "reserved_lanes": lanes,
            "edge_positions": exposed,
            "overhang_m": 0.055,
            "rest_edge_gap_m": 0.025,
            "table_contact_seconds": 0.004,
            "grip_effort": 4.0,
            "parking_bounds": list(PARKING_BOUNDS),
            # Plate pushes were qualified with fine integration. At 2 ms the
            # 4 ms table contact is at MuJoCo's 2x-timestep limit and one-frame
            # bounces exceed the 0.3 mm support guard; 1 ms removes them.
            **({"max_physics_timestep_s": 0.001} if len(plates) else {}),
        },
    )
