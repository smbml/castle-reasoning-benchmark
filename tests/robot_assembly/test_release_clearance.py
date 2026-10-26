"""Closed-gripper placement geometry must influence grasp ranking."""

import json
from pathlib import Path

import numpy as np

import castle_reasoning.runtime.planning.grasps as grasps
import castle_reasoning.runtime.planning.manipulation as manipulation
import castle_reasoning.runtime.simulation.models as models


def test_closed_placement_collision_does_not_starve_clear_grasps():
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/closed_placement_collision.json").read_text()
    )
    model, _ = models.make_model(
        fixture["sizes"], fixture["target"], table_bounds=fixture["table_bounds"]
    )
    planner = manipulation.Planner(
        model, fixture["sizes"], planning_seconds=8, grasp_family="all"
    )
    part = fixture["part"]
    qpos, destination = np.array(fixture["qpos"]), np.array(fixture["target"][part])
    first = grasps.propose(fixture["sizes"][part])[0]
    assert planner._destination_obstructed(qpos, part, destination, first)
    result = planner.plan(qpos, part, destination)
    assert result.grasp != first
    assert not result.attempts
    assert result.segments[-1].name == "retreat"
