"""Nominal pickup access must precede storage-stack precedence decisions."""

import json
from pathlib import Path

import numpy as np
import pytest

import castle_reasoning.data.generation.edge_staging as edge_staging
import castle_reasoning.data.generation.geometry as geometry
import castle_reasoning.data.generation.removal_access as removal_access
import castle_reasoning.data.generation.waypoints as waypoints
import castle_reasoning.runtime.planning.manipulation as manipulation
import castle_reasoning.runtime.simulation.models as models


def test_unreachable_pickup_reports_the_complete_remaining_set():

    plan = {
        "target_id": "outside_reach",
        "sizes": [[0.02, 0.02, 0.02]],
        "target": [[2.0, 0, 0.01]],
        "removal_order": [0],
        "meters_per_grid": 0.01,
        "edge_access": {"table_bounds": list(edge_staging.TABLE_BOUNDS), "plates": []},
    }
    with pytest.raises(removal_access.OrderFailure) as failure:
        removal_access.order(plan, planning_seconds=1)
    assert failure.value.evidence["remaining"] == [0]
    assert failure.value.evidence["order"] == []
    assert failure.value.evidence["events"]
    assert failure.value.evidence["physics_verified"] is False


def test_pickup_only_has_no_destination_or_release_segments():
    sizes = [[0.02, 0.02, 0.04]]
    model, initial = models.make_model(sizes, [[0.425, 0, 0.02]])
    planner = manipulation.Planner(model, sizes, planning_seconds=8)
    result = planner.plan(initial, 0, None)
    assert [s.name for s in result.segments] == ["approach", "descend", "close", "lift"]
    for segment in result.segments:
        for state in segment.path:
            assert planner.scene.valid(
                state,
                0,
                segment.pads_allowed,
                segment.supports,
                segment.payload,
                segment.departures,
                segment.arrivals,
            )


@pytest.mark.parametrize("target,first,second", [("T0072", 6, 4), ("T0099", 38, 17)])
def test_recorded_targets_order_grasps_before_forming_storage_stacks(
    target, first, second
):

    cases = json.loads(
        (
            Path(__file__).parents[1]
            / "construction_targets/fixtures/staging_cohort.json"
        ).read_text()
    )
    case = next(c for c in cases if c["target_id"] == target)
    plan = case | {
        "meters_per_grid": 0.01,
        "edge_access": {
            "table_bounds": list(edge_staging.TABLE_BOUNDS),
            "plates": np.flatnonzero(
                np.min(np.array(case["sizes"])[:, :2], axis=1) > 0.07
            ).tolist(),
        },
    }
    before = json.dumps(plan, sort_keys=True)
    order, evidence = removal_access.order(plan, planning_seconds=2)
    assert sorted(order) == list(range(len(case["sizes"])))
    assert order.index(first) < order.index(second)
    assert evidence["physics_verified"] is False
    assert evidence["elapsed_seconds"] > 0
    assert json.dumps(plan, sort_keys=True) == before
    sizes = np.array(case["sizes"])
    goals = np.array(case["target"])
    low = goals - sizes / 2
    low[:, :2] -= low[:, :2].min(axis=0)
    parts = [
        geometry.Part(
            tuple(np.rint(p / 0.01).astype(int)), tuple(np.rint(s / 0.01).astype(int))
        )
        for p, s in zip(low, sizes)
    ]
    assert waypoints.construction_removal_order(parts, order[::-1])[0] == order
    _, supports, _ = edge_staging.arrange(
        goals,
        sizes,
        order,
        case["max_grasp_width"],
        case["max_stack_height"],
        gap=case["parking_gap"],
        stack_layout="nested",
    )
    placed = set()
    for part in order:
        parent = dict(supports)[part]
        assert parent == -1 or parent in placed
        placed.add(part)


def test_slotted_needs_standing_neighbours_on_two_opposite_faces():

    row = [geometry.Part((x, 0, 0), (2, 2, 2)) for x in (0, 2, 4)]
    corner = geometry.Part((6, 2, 0), (2, 2, 2))  # touches the row only at an edge
    parts = row + [corner]
    assert removal_access.slotted(parts, 1, {0, 1, 2, 3})
    assert not removal_access.slotted(parts, 1, {0, 1, 3})
    assert not removal_access.slotted(parts, 0, {0, 1, 2, 3})
    assert not removal_access.slotted(parts, 2, {0, 1, 2, 3})


def test_order_takes_a_side_neighbour_before_the_part_between_them():

    plan = {
        "target_id": "row",
        "sizes": [[0.02, 0.02, 0.02]] * 3,
        "target": [[0.40, 0, 0.01], [0.42, 0, 0.01], [0.44, 0, 0.01]],
        "removal_order": [1, 0, 2],
        "meters_per_grid": 0.01,
        "edge_access": {"table_bounds": list(edge_staging.TABLE_BOUNDS), "plates": []},
    }
    order, evidence = removal_access.order(plan, planning_seconds=2)
    assert order[0] != 1 and sorted(order) == [0, 1, 2]
    assert evidence["slot_insertions"] == []
