"""A plate's staging slot reserves its complete future pushing corridor."""

import numpy as np
import pytest

import castle_reasoning.data.generation.geometry as geometry
import castle_reasoning.data.generation.waypoints as waypoints


def plate_scene():
    return [
        geometry.Part((0, 0, 0), (18, 20, 1)),
        geometry.Part((4, 7, 1), (2, 2, 4)),
        geometry.Part((12, 7, 1), (2, 2, 4)),
    ]


def test_edge_staging_preserves_full_support_and_reserves_access_lanes():
    plan = waypoints.plan(plate_scene(), staging="edge", construction_order=[0, 1, 2])
    edge = plan["edge_access"]
    assert plan["parking_bounds"] == edge["parking_bounds"]
    positions, sizes = np.array(plan["parking"]), np.array(plan["sizes"])
    table_front = edge["table_bounds"][2]
    assert edge["plates"] == [0]
    assert 0.015 <= positions[0, 1] - sizes[0, 1] / 2 - table_front <= 0.04
    assert edge["edge_positions"][0][1] - table_front >= 0.025
    assert [0, -1] in plan["parking_supports"]
    assert not any(lower == 0 for _, lower in plan["parking_supports"])
    for rectangle in edge["reserved_lanes"]:
        low, high = np.array(rectangle[:2]), np.array(rectangle[2:])
        for part in (1, 2):
            assert not (
                np.all(positions[part, :2] + sizes[part, :2] / 2 > low)
                and np.all(positions[part, :2] - sizes[part, :2] / 2 < high)
            )


def test_top_down_staging_still_rejects_a_wide_plate():
    with pytest.raises(waypoints.PlanningFailure, match="top_down_aperture"):
        waypoints.plan(plate_scene())


@pytest.mark.parametrize("height", [0, -1, float("nan"), float("inf")])
def test_edge_staging_requires_a_finite_positive_stack_height(height):
    with pytest.raises(ValueError, match="Stack height"):
        waypoints.plan(plate_scene(), staging="edge", max_stack_height=height)


def test_edge_staging_does_not_assume_stacked_wide_plates_are_supported():
    parts = [
        geometry.Part((0, 0, 0), (18, 20, 1)),
        geometry.Part((0, 0, 1), (18, 20, 1)),
    ]
    with pytest.raises(waypoints.PlanningFailure, match="single_ground_plate"):
        waypoints.plan(parts, staging="edge")
