"""Data plans."""

import numpy as np

import castle_reasoning.data.generation.geometry as geometry
import castle_reasoning.data.generation.waypoints as waypoints
import castle_reasoning.runtime.control.parameters as parameters


def grid_parts(plan):
    """The plan's target cuboids on its integer grid."""
    size = np.asarray(plan["sizes"])
    low = np.asarray(plan["target"]) - size / 2
    low[:, :2] -= low[:, :2].min(axis=0)
    scale = plan["meters_per_grid"]
    return [
        geometry.Part(
            tuple(np.rint(p / scale).astype(int)), tuple(np.rint(s / scale).astype(int))
        )
        for p, s in zip(low, size)
    ]


def removable(plan, remaining):
    """Nominal static eligibility, independent of robot feasibility."""
    parts = grid_parts(plan)
    for part in plan["removal_order"]:
        if any(
            upper == part and lower in remaining
            for upper, lower in plan["parking_supports"]
        ):
            continue
        if part not in remaining or not waypoints.vertical_clear(
            parts, part, remaining
        ):
            continue
        # A fixed foundation never moves but still carries the parts above it.
        rest = [
            parts[i]
            for i in sorted((remaining - {part}) | set(plan.get("fixed_parts", ())))
        ]
        if not rest or geometry.equilibrium(rest):
            yield part


def control_config(physics_timestep=0.0005):
    """Select an integration resolution without changing physical parameters."""
    if physics_timestep not in (0.0005, 0.001, 0.002):
        raise ValueError("Supported physics timesteps are 0.0005, 0.001, and 0.002.")
    return parameters.ControlConfig(physics_timestep=physics_timestep)
