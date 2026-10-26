"""Actual cohort closures must not consume the entire route-planning budget."""

import json
from pathlib import Path

import numpy as np

import castle_reasoning.runtime.planning.grasps as grasps
import castle_reasoning.runtime.planning.manipulation as manipulation
import castle_reasoning.runtime.simulation.models as models


def test_recorded_source_closure_failure_has_a_complete_path_within_normal_budget():
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/obstructed_source_closure.json").read_text()
    )
    model, _ = models.make_model(
        fixture["sizes"],
        fixture["target"],
        table_bounds=fixture["edge_access"]["table_bounds"],
    )
    planner = manipulation.Planner(
        model, fixture["sizes"], planning_seconds=8, grasp_family="all"
    )
    part = fixture["part"]
    result = planner.plan(
        np.array(fixture["qpos"]), part, np.array(fixture["parking"][part])
    )
    assert [s.name for s in result.segments] == [
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
    for segment in result.segments:
        for qpos in segment.path:
            assert planner.scene.valid(
                qpos,
                part,
                segment.pads_allowed,
                segment.supports,
                segment.payload,
                segment.departures,
                segment.arrivals,
            )


def test_intended_pad_contact_does_not_mark_an_isolated_pick_obstructed():
    sizes = [[0.02, 0.02, 0.04]]
    model, qpos = models.make_model(sizes, [[0.425, 0, 0.02]])
    planner = manipulation.Planner(model, sizes)
    grasp = planner.aperture.calibrate(grasps.propose(sizes[0])[0])
    original = qpos.copy()
    assert not planner._source_obstructed(qpos, 0, grasp)
    np.testing.assert_array_equal(qpos, original)
