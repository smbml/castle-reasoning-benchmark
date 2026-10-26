"""Recovery uses the same finite workcell as collection and motion planning."""

import json
from pathlib import Path

import numpy as np
import pytest

import castle_reasoning.runtime.execution.recovery as recovery
import castle_reasoning.runtime.planning.manipulation as manipulation
import castle_reasoning.runtime.simulation.models as models


@pytest.mark.parametrize("world", [1, 2, 3])
def test_recorded_held_object_can_recover_in_expanded_staging(world):
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/t0024_recovery_table.json").read_text()
    )
    model, _ = models.make_model(
        fixture["sizes"], fixture["target"], table_bounds=fixture["table_bounds"]
    )
    planner = manipulation.Planner(model, fixture["sizes"])
    state = next(state for state in fixture["states"] if state["world"] == world)
    result = recovery.assess(
        planner,
        np.array(state["qpos"]),
        np.array(state["qvel"]),
        fixture["part"],
        np.array(state["reference"]),
    )
    assert result["held"] and not result["supported"]
    assert result["neighbor_displacement_m"] < 0.000002
    assert result["ready"]


@pytest.mark.parametrize(
    ("position", "ready"),
    [
        ([0.75, 0.35, 0.01], True),
        ([0.86, 0.35, 0.01], False),
        ([0.75, 0.41, 0.01], False),
        ([-0.16, 0.35, 0.01], False),
        ([0.75, -0.26, 0.01], False),
    ],
)
def test_supported_object_must_remain_within_the_finite_table(position, ready):
    sizes = [[0.02] * 3]
    model, qpos = models.make_model(
        sizes, [position], table_bounds=[-0.15, 0.85, -0.25, 0.40]
    )
    planner = manipulation.Planner(model, sizes)
    result = recovery.assess(planner, qpos, np.zeros(model.nv), 0, qpos)
    assert result["supported"]
    assert result["ready"] == ready


def test_infinite_floor_retains_the_original_workcell_bounds():
    sizes = [[0.02] * 3]
    model, qpos = models.make_model(sizes, [[0.4, 0.335, 0.01]])
    planner = manipulation.Planner(model, sizes)
    result = recovery.assess(planner, qpos, np.zeros(model.nv), 0, qpos)
    assert not result["ready"]
