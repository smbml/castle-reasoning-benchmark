"""All-active masked servo execution must match the ordinary controller."""

import os

import mujoco
import numpy as np
import pytest

import castle_reasoning.data.assembly.teacher as teacher
import castle_reasoning.runtime.control.active as active
import castle_reasoning.runtime.control.batch_tracking as batch_tracking
import castle_reasoning.runtime.control.tracking as tracking
import castle_reasoning.runtime.planning.collision as collision
import castle_reasoning.runtime.planning.manipulation as manipulation
import castle_reasoning.runtime.simulation.models as models

pytestmark = pytest.mark.skipif(
    os.environ.get("CONSTRUCTION_GPU_TESTS") != "1", reason="Explicit GPU opt-in"
)


def test_all_active_controller_matches_stock_path():

    goals = np.array([[0.425, 0, 0.01]])
    model, home = models.make_model([[0.02] * 3], goals)
    model.opt.timestep = 0.002
    model.opt.solver = mujoco.mjtSolver.mjSOL_CG
    model.opt.iterations = 1000
    model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    collector = teacher.make_collector(model, goals, 2, 1, 0)
    collector.engine.model.block_dim.linesearch_iterative = 128
    scene = collision.Scene(model, home)
    segment = manipulation.hold_segment(home, ticks=12)
    request = batch_tracking.Request(segment, scene, -1, guard=False)
    results = []
    for kind in (tracking.PathControl, active.ActivePathControl):
        controller = collector.controller = kind(collector)
        collector.reset(qpos=np.tile(home, (2, 1)), qvel=np.zeros((2, model.nv)))
        trace = batch_tracking.execute(controller, [request, request])
        results.append(trace)
    for world in range(2):
        for key in ("qpos", "qvel", "action"):
            np.testing.assert_allclose(
                results[0][world][0][key], results[1][world][0][key], atol=2e-5, rtol=0
            )
    # Full-batch replay must not inherit a mask from prior partial execution.
    controller.active_worlds.fill_(False)
    before = collector.engine.data.time.numpy()
    actions = np.concatenate([row[0]["action"] for row in results[1]], axis=1)
    replay, _ = batch_tracking.replay(controller, actions, np.ones(actions.shape[:2]))
    assert np.all(collector.engine.data.time.numpy() > before)
    np.testing.assert_array_equal(replay["action"], actions)
