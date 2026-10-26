"""A blocked destination can require changing grasp after the long transfer."""

import json
from pathlib import Path

import numpy as np

import castle_reasoning.runtime.planning.manipulation as manipulation
import castle_reasoning.runtime.planning.regrasp as regrasp
import castle_reasoning.runtime.simulation.models as models


def test_recorded_assembly_uses_supported_destination_waypoint():
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/t0099_destination_regrasp.json").read_text()
    )
    plan = fixture["plan"]
    model, _ = models.make_model(
        plan["sizes"], plan["target"], table_bounds=plan["edge_access"]["table_bounds"]
    )
    planner = manipulation.Planner(
        model, plan["sizes"], planning_seconds=8, grasp_family="all"
    )
    qpos = np.array(fixture["qpos"])
    part = fixture["part"]
    original = qpos.copy()
    destination = np.array(plan["target"][part])
    result = regrasp.plan(planner, qpos, part, destination)
    assert result.anchor == "destination"
    assert np.linalg.norm(result.waypoint[:2] - destination[:2]) <= 0.00801
    assert result.support_margin_m >= 0.001
    assert np.isclose(result.first.grasp.roll, np.pi)
    assert abs(np.cos(result.second.grasp.roll)) < 0.1
    np.testing.assert_array_equal(qpos, original)
    for move in (result.first, result.second):
        assert move.segments[0].name == "approach"
        assert move.segments[-1].name == "retreat"
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
