"""Place against observed support surfaces without weakening collision checks."""

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import castle_reasoning.runtime.planning.arm as arm
import castle_reasoning.runtime.planning.manipulation as manipulation
import castle_reasoning.runtime.simulation.models as models


def test_reassembly_reaches_support_without_commanding_penetration():
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/t0052_setdown.json").read_text()
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
    destination = np.array(fixture["target"][part])
    result = planner.plan(np.array(fixture["qpos"]), part, destination)
    setdown = next(s for s in result.segments if s.name == "setdown")
    planner.scene.set_state(setdown.path[-1], setdown.payload)
    actual = planner.scene.data.qpos[14 + 7 * part : 17 + 7 * part]
    assert np.linalg.norm(actual - destination) < 0.001
    assert setdown.supports
    for support in setdown.supports:
        gap = mujoco.mj_geomDistance(
            model,
            planner.scene.data,
            model.geom(f"object_{part}").id,
            support,
            0.01,
            None,
        )
        assert gap > 0


def test_seating_height_uses_tilted_support_face():
    sizes = [[0.06, 0.06, 0.02], [0.02, 0.02, 0.02]]
    model, qpos = models.make_model(sizes, [[0.425, 0, 0.01], [0.3, -0.15, 0.01]])
    qpos[17:21] = np.roll(Rotation.from_euler("y", 1, degrees=True).as_quat(), 1)
    planner = manipulation.Planner(model, sizes)
    target = np.array([0.445, 0, 0.03])
    seated, supports = planner.seating_destination(qpos, 1, target)
    assert supports == (model.geom("object_0").id,)
    assert np.array_equal(seated[:2], target[:2])
    check = qpos.copy()
    check[21:28] = np.r_[seated, 1, 0, 0, 0]
    planner.scene.set_state(check)
    gap = mujoco.mj_geomDistance(
        model,
        planner.scene.data,
        model.geom("object_0").id,
        model.geom("object_1").id,
        0.01,
        None,
    )
    assert gap == pytest.approx(0.0001, abs=2e-6)


def test_seating_cannot_relocate_an_unsupported_destination():
    model, qpos = models.make_model([[0.02] * 3], [[0.425, 0, 0.01]])
    planner = manipulation.Planner(model, [[0.02] * 3])
    with pytest.raises(arm.PlanningError, match="support"):
        planner.seating_destination(qpos, 0, np.array([0.3, -0.15, 0.04]))


def test_seating_rejects_a_surface_outside_the_goal_tolerance():
    sizes = [[0.08, 0.08, 0.02], [0.02, 0.02, 0.02]]
    model, qpos = models.make_model(sizes, [[0.425, 0, 0.01], [0.3, -0.15, 0.01]])
    qpos[17:21] = np.roll(Rotation.from_euler("y", 30, degrees=True).as_quat(), 1)
    planner = manipulation.Planner(model, sizes)
    with pytest.raises(arm.PlanningError, match="support_height"):
        planner.seating_destination(qpos, 1, np.array([0.405, 0, 0.03]))
