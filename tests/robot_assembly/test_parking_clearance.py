"""Staging must leave space for the robot's checked release and retreat."""

import json
from pathlib import Path

import numpy as np
import pytest

import castle_reasoning.data.generation.geometry as geometry
import castle_reasoning.data.generation.waypoints as waypoints
import castle_reasoning.runtime.planning.manipulation as manipulation
import castle_reasoning.runtime.simulation.models as models


@pytest.mark.parametrize(
    "filename", ["blocked_parking_release.json", "blocked_arm_placement.json"]
)
def test_wider_staging_clears_the_recorded_blocked_release(filename):
    fixture = json.loads((Path(__file__).parent / "fixtures" / filename).read_text())
    sizes = np.array(fixture["sizes"])
    target = np.array(fixture["target"])
    origins = target - sizes / 2
    origins[:, :2] -= origins[:, :2].min(axis=0)
    scale = fixture["meters_per_grid"]
    parts = [
        geometry.Part(
            tuple(np.rint(p / scale).astype(int)), tuple(np.rint(s / scale).astype(int))
        )
        for p, s in zip(origins, sizes, strict=True)
    ]
    plan = waypoints.plan(parts, staging="edge", parking_gap=0.055)
    assert plan["parking_gap"] == 0.055
    np.testing.assert_allclose(plan["target"], target, atol=1e-12)
    np.testing.assert_array_equal(plan["sizes"], sizes)
    assert plan["removal_order"] == fixture["removal_order"]
    parked = set(fixture["completed_order"])
    assert all(
        b == -1 or b in parked for a, b in plan["parking_supports"] if a in parked
    )
    qpos = np.array(fixture["qpos"])
    for part in parked:
        qpos[14 + 7 * part : 17 + 7 * part] += (
            np.array(plan["parking"][part]) - fixture["parking"][part]
        )
    model, _ = models.make_model(
        sizes, target, table_bounds=fixture["edge_access"]["table_bounds"]
    )
    planner = manipulation.Planner(model, sizes, planning_seconds=8, grasp_family="all")
    result = planner.plan(
        qpos, fixture["part"], np.array(plan["parking"][fixture["part"]])
    )
    assert result.segments[-1].name == "retreat"
    for segment in result.segments:
        for position in segment.path:
            assert planner.scene.valid(
                position,
                result.part,
                segment.pads_allowed,
                segment.supports,
                segment.payload,
                segment.departures,
                segment.arrivals,
            )


@pytest.mark.parametrize("gap", [-1, float("nan"), float("inf")])
def test_staging_rejects_invalid_gap(gap):
    with pytest.raises(ValueError, match="Parking gap"):
        waypoints.plan([geometry.Part((0, 0, 0), (2, 2, 2))], parking_gap=gap)


def test_zero_gap_is_valid_and_recorded():
    plan = waypoints.plan([geometry.Part((0, 0, 0), (2, 2, 2))], parking_gap=0)
    assert plan["parking_gap"] == 0
