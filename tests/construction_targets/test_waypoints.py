"""Removal planning preserves supports, exact identity and parking geometry."""

import numpy as np
import pytest

import castle_reasoning.data.generation.geometry as geometry
import castle_reasoning.data.generation.waypoints as waypoints


def bridge():
    return [
        geometry.Part((0, 0, 0), (2, 2, 4)),
        geometry.Part((6, 0, 0), (2, 2, 4)),
        geometry.Part((0, 0, 4), (8, 2, 1)),
    ]


def test_bridge_order_and_exact_reverse():
    plan = waypoints.plan(bridge(), staging="flat")
    assert plan["removal_order"][0] == 2
    assert plan["assembly_order"] == plan["removal_order"][::-1]
    backward = np.asarray(plan["disassembly_goals"])
    forward = np.asarray(plan["assembly_goals"])
    np.testing.assert_array_equal(backward[::-1], forward)
    assert backward.shape == (4, 3, 3)
    assert np.all(
        np.count_nonzero(np.any(np.diff(backward, axis=0), axis=2), axis=1) == 1
    )
    assert not waypoints.overlaps(
        np.asarray(plan["parking"]), np.asarray(plan["sizes"]), gap=0.02
    )
    for count in range(4):
        remaining = [
            p for i, p in enumerate(bridge()) if i not in plan["removal_order"][:count]
        ]
        assert not remaining or geometry.equilibrium(remaining)


def test_swept_collision_checks_the_whole_lift():
    parts = bridge()
    assert not waypoints.vertical_clear(parts, 0, {0, 1, 2})
    assert waypoints.vertical_clear(parts, 2, {0, 1, 2})


def test_wide_grasp_and_parking_failure_are_explicit():
    with pytest.raises(waypoints.PlanningFailure, match="grasp_width"):
        waypoints.plan([geometry.Part((0, 0, 0), (8, 8, 2))])
    with pytest.raises(waypoints.PlanningFailure, match="parking"):
        waypoints.plan(bridge(), parking_bounds=(0.42, 0.43, -0.01, 0.01))


def test_unstable_target_is_rejected():
    with pytest.raises(waypoints.PlanningFailure, match="static"):
        waypoints.plan(
            [geometry.Part((0, 0, 0), (2, 2, 4)), geometry.Part((1, 0, 4), (6, 2, 1))]
        )


def test_balanced_counterweights_cannot_be_removed_one_at_a_time():
    parts = [
        geometry.Part((0, 0, 0), (2, 2, 4)),
        geometry.Part((-2, 0, 4), (6, 2, 1)),
        geometry.Part((-2, 0, 5), (1, 2, 8)),
        geometry.Part((3, 0, 5), (1, 2, 8)),
    ]
    assert geometry.equilibrium(parts)
    assert not geometry.equilibrium(parts[:3])
    with pytest.raises(waypoints.PlanningFailure, match="no_stable"):
        waypoints.removal_order(parts)


def test_stacks_fit_one_floor_slot_and_reverse_without_buried_parts():
    parts = [geometry.Part((0, 0, z), (2, 2, 2)) for z in (0, 2, 4)]
    plan = waypoints.plan(parts, parking_bounds=(0.52, 0.54, -0.01, 0.01))
    assert plan["staging"] == "stacks"
    assert plan["parking_supports"] == [[2, -1], [1, 2], [0, 1]]
    slots = np.asarray(plan["parking"])
    np.testing.assert_allclose(slots[:, :2], [[0.53, 0]] * 3, atol=1e-10)
    np.testing.assert_allclose(slots[:, 2], [0.05, 0.03, 0.01])
    np.testing.assert_array_equal(
        plan["disassembly_goals"][::-1], plan["assembly_goals"]
    )
    placed = set()
    for part in plan["removal_order"]:
        parent = dict(plan["parking_supports"])[part]
        assert parent == -1 or parent in placed
        placed.add(part)
    for part in plan["assembly_order"]:
        assert not any(
            base == part and child in placed for child, base in plan["parking_supports"]
        )
        placed.remove(part)
    staged_parts = [
        geometry.Part(tuple(np.rint((p - 0.01) * 100).astype(int)), (2, 2, 2))
        for p in slots
    ]
    assert geometry.equilibrium(staged_parts)
    with pytest.raises(waypoints.PlanningFailure, match="parking"):
        waypoints.plan(parts, parking_bounds=(0.52, 0.54, -0.01, 0.01), staging="flat")


def test_stack_height_budget_and_overhang_bounds_are_enforced():
    parts = [geometry.Part((0, 0, z), (2, 2, 2)) for z in (0, 2, 4)]
    with pytest.raises(waypoints.PlanningFailure, match="parking"):
        waypoints.plan(
            parts, parking_bounds=(0.52, 0.54, -0.01, 0.01), max_stack_height=0.04
        )
    goals = np.array([[0.425, 0, 0.01], [0.425, 0, 0.03]])
    with pytest.raises(waypoints.PlanningFailure, match="parking"):
        waypoints.stacked_parking_slots(
            goals,
            np.array([[0.02] * 3, [0.04, 0.02, 0.02]]),
            (0.52, 0.54, -0.01, 0.01),
            [0, 1],
        )


def test_centered_overhang_has_a_static_witness_and_clear_transport():
    plan = waypoints.plan(bridge())
    slots, sizes = np.array(plan["parking"]), np.array(plan["sizes"])
    parts = [
        geometry.Part(
            tuple(np.rint((p - s / 2) * 1000).astype(int)),
            tuple(np.rint(s * 1000).astype(int)),
        )
        for p, s in zip(slots, sizes)
    ]
    assert geometry.equilibrium(parts)
    top = np.max(slots[:, 2] + sizes[:, 2] / 2)
    for move in plan["object_transport_waypoints"]:
        assert move["centers"][1][2] - sizes[move["part"], 2] / 2 > top


def test_removal_diagnostics_report_a_valid_prefix_and_each_dead_end_rejection():
    parts = [
        geometry.Part((0, 0, 0), (2, 2, 4)),
        geometry.Part((-2, 0, 4), (6, 2, 1)),
        geometry.Part((-2, 0, 5), (1, 2, 8)),
        geometry.Part((3, 0, 5), (1, 2, 8)),
        geometry.Part((10, 0, 0), (2, 2, 2)),
    ]
    trace = {}
    with pytest.raises(waypoints.PlanningFailure, match="no_stable"):
        waypoints.removal_order(parts, diagnostics=trace)
    assert trace["prefix"] == [4]
    assert trace["remaining"] == [0, 1, 2, 3]
    reasons = {r["part"]: r["reason"] for r in trace["rejections"]}
    assert reasons == {
        0: "vertical_collision",
        1: "vertical_collision",
        2: "static_equilibrium",
        3: "static_equilibrium",
    }
    assert geometry.equilibrium([parts[i] for i in trace["remaining"]])


def test_diagnostics_preserve_successful_search_order_and_counts():
    trace = {"stale": True}
    expected = waypoints.removal_order(bridge())
    assert waypoints.removal_order(bridge(), diagnostics=trace) == expected
    assert "stale" not in trace
