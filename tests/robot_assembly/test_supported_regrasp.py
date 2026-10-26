"""Two checked moves can connect source and destination grasp families."""

import json
import time
from pathlib import Path

import numpy as np
import pytest

import castle_reasoning.runtime.planning.arm as arm
import castle_reasoning.runtime.planning.manipulation as manipulation
import castle_reasoning.runtime.planning.regrasp as regrasp
import castle_reasoning.runtime.simulation.models as models


@pytest.fixture
def recorded():
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/t0099_regrasp.json").read_text()
    )
    plan = fixture["plan"]
    model, _ = models.make_model(
        plan["sizes"], plan["target"], table_bounds=plan["edge_access"]["table_bounds"]
    )
    planner = manipulation.Planner(
        model, plan["sizes"], planning_seconds=8, grasp_family="all"
    )
    return (
        planner,
        np.array(fixture["qpos"]),
        fixture["part"],
        np.array(plan["parking"][fixture["part"]]),
    )


def test_recorded_transfer_needs_a_supported_regrasp(recorded):
    planner, qpos, part, destination = recorded
    original = qpos.copy()
    with pytest.raises(arm.PlanningError, match="no_complete_manipulation"):
        planner.plan(qpos, part, destination)
    result = regrasp.plan(planner, qpos, part, destination)
    assert abs(np.cos(result.first.grasp.roll)) < 0.1
    assert np.isclose(result.second.grasp.roll, np.pi)
    assert (
        np.linalg.norm(result.waypoint[:2] - qpos[14 + 7 * part : 16 + 7 * part])
        <= 0.00801
    )
    assert result.support_margin_m >= 0.001
    np.testing.assert_array_equal(qpos, original)
    for move in (result.first, result.second):
        assert [segment.name for segment in move.segments] == [
            "approach",
            "descend",
            "close",
            "lift",
            "transit",
            "place",
            "setdown",
            "open",
            "retreat",
        ]
        for segment in move.segments:
            for state in segment.path:
                assert planner.scene.valid(
                    state,
                    part,
                    segment.pads_allowed,
                    segment.supports,
                    segment.payload,
                    segment.departures,
                    segment.arrivals,
                )
    final = result.second.segments[-1].path[-1][14 + 7 * part : 17 + 7 * part]
    assert np.linalg.norm(final - destination) < 0.005


@pytest.mark.parametrize("shift, safe", [(0.0, True), (0.008, True), (0.012, False)])
def test_support_polygon_keeps_com_inside_two_piers(recorded, shift, safe):
    planner, qpos, part, _ = recorded
    qpos[14 + 7 * part] += shift
    margin, supports = regrasp.support_margin(planner, qpos, part)
    assert (margin >= 0.001) == safe
    if safe:
        assert {planner.scene.model.geom(i).name for i in supports} == {
            "object_36",
            "object_37",
        }


def test_tilted_intermediate_is_rejected(recorded):
    planner, qpos, part, _ = recorded
    qpos[17 + 7 * part : 21 + 7 * part] = [np.cos(0.1), np.sin(0.1), 0, 0]
    margin, _ = regrasp.support_margin(planner, qpos, part)
    assert margin < 0.001


def test_unsupported_or_nonfinite_state_has_no_candidate(recorded):
    planner, qpos, part, destination = recorded
    qpos[16 + 7 * part] += 0.02
    with pytest.raises(arm.PlanningError, match="no_supported_regrasp"):
        regrasp.plan(planner, qpos, part, destination)
    qpos[14 + 7 * part] = np.nan
    assert regrasp.support_margin(planner, qpos, part)[0] < 0.001


def test_expired_budget_does_not_produce_a_move(recorded):
    planner, qpos, part, destination = recorded
    original = qpos.copy()
    with pytest.raises(arm.PlanningError, match="budget"):
        regrasp.plan(
            planner, qpos, part, destination, time.perf_counter() - 1
        )
    np.testing.assert_array_equal(qpos, original)
