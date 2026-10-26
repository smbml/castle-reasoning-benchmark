"""Placement offsets preserve supports and avoid the measured corner collision."""

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

import castle_reasoning.runtime.planning.manipulation as manipulation
import castle_reasoning.runtime.simulation.models as models


@pytest.fixture
def placement():
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/t0131_placement_neighbor.json").read_text()
    )
    model, _ = models.make_model(
        fixture["sizes"],
        fixture["target"],
        table_bounds=fixture["edge_access"]["table_bounds"],
    )
    planner = manipulation.Planner(model, fixture["sizes"])
    return fixture, planner


def measured_gap(planner, qpos, part, destination, neighbor):
    probe = qpos.copy()
    probe[14 + 7 * part : 21 + 7 * part] = np.r_[destination, 1, 0, 0, 0]
    planner.scene.set_state(probe)
    return mujoco.mj_geomDistance(
        planner.scene.model,
        planner.scene.data,
        planner.scene.objects[part],
        planner.scene.objects[neighbor],
        0.01,
        None,
    )


def test_measured_placement_avoids_neighbor_without_relocating_support(placement):
    fixture, planner = placement
    qpos = np.array(fixture["qpos"])
    goal = np.array(fixture["target"][fixture["part"]])
    q_before, goal_before = qpos.copy(), goal.copy()
    nominal, supports = planner.seating_destination(qpos, fixture["part"], goal)
    assert (
        measured_gap(planner, qpos, fixture["part"], nominal, fixture["neighbor"])
        < 0.0003
    )
    selected = planner.placement_destination(qpos, fixture["part"], goal)
    seated, chosen_supports = planner.seating_destination(
        qpos, fixture["part"], selected
    )
    assert (
        measured_gap(planner, qpos, fixture["part"], seated, fixture["neighbor"])
        >= 0.0003
    )
    assert chosen_supports == supports
    assert np.max(np.abs(selected[:2] - goal[:2])) <= 0.00050001
    assert selected[2] == goal[2]
    np.testing.assert_array_equal(qpos, q_before)
    np.testing.assert_array_equal(goal, goal_before)


def test_clear_destination_is_unchanged():
    model, qpos = models.make_model([[0.02] * 3], [[0.425, 0, 0.01]])
    planner = manipulation.Planner(model, [[0.02] * 3])
    goal = np.array([0.3, -0.15, 0.01])
    np.testing.assert_array_equal(planner.placement_destination(qpos, 0, goal), goal)


def test_final_t0065_part_clears_displaced_neighbor():
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/t0065_placement_neighbor.json").read_text()
    )
    model, _ = models.make_model(
        fixture["sizes"],
        fixture["target"],
        table_bounds=fixture["edge_access"]["table_bounds"],
    )
    planner = manipulation.Planner(model, fixture["sizes"])
    qpos = np.array(fixture["qpos"])
    goal = np.array(fixture["target"][fixture["part"]])
    q_before, goal_before = qpos.copy(), goal.copy()
    nominal, supports = planner.seating_destination(qpos, fixture["part"], goal)
    assert measured_gap(planner, qpos, fixture["part"], nominal, 17) < -0.001
    selected = planner.placement_destination(qpos, fixture["part"], goal)
    seated, chosen_supports = planner.seating_destination(
        qpos, fixture["part"], selected
    )
    assert measured_gap(planner, qpos, fixture["part"], seated, 17) >= 0.0003
    assert chosen_supports == supports
    assert np.linalg.norm(seated - goal) < 0.003
    assert selected[2] == goal[2]
    np.testing.assert_array_equal(qpos, q_before)
    np.testing.assert_array_equal(goal, goal_before)

    # The corrected pose must survive complete arm/payload path planning.
    planner.planning_seconds = 8
    planner.grasp_family = "all"
    result = planner.plan(qpos, fixture["part"], goal)
    setdown = next(s for s in result.segments if s.name == "setdown")
    planner.scene.set_state(setdown.path[-1], setdown.payload)
    start = 14 + 7 * fixture["part"]
    actual = planner.scene.data.qpos[start : start + 3]
    np.testing.assert_allclose(actual, seated, atol=2e-5)


def test_unsupported_destination_is_not_repaired_by_xy_search():
    model, qpos = models.make_model([[0.02] * 3], [[0.425, 0, 0.01]])
    planner = manipulation.Planner(model, [[0.02] * 3])
    goal = np.array([0.3, -0.15, 0.04])
    np.testing.assert_array_equal(planner.placement_destination(qpos, 0, goal), goal)


def test_no_clearance_candidate_keeps_nominal_destination():
    sizes = [[0.02] * 3] * 3
    model, qpos = models.make_model(
        sizes, [[0.4, 0, 0.01], [0.38, 0, 0.01], [0.42, 0, 0.01]]
    )
    planner = manipulation.Planner(model, sizes)
    goal = np.array([0.4, 0, 0.01])
    # The bounded XY search cannot clear both adjacent walls.
    np.testing.assert_array_equal(planner.placement_destination(qpos, 0, goal), goal)


def test_clearance_search_preserves_each_existing_support_overlap():
    sizes = [[0.02] * 3] * 4
    positions = [
        [0.3, -0.15, 0.01],
        [0.381, 0, 0.01],
        [0.419, 0, 0.01],
        [0.38, 0, 0.03],
    ]
    model, qpos = models.make_model(sizes, positions)
    planner = manipulation.Planner(model, sizes)
    goal = np.array([0.4, 0, 0.03])
    # The bridging cube overlaps each lower block by 1 mm. Shifting right would
    # clear the neighbor but discard half its left support, so keep the goal.
    np.testing.assert_array_equal(planner.placement_destination(qpos, 0, goal), goal)
