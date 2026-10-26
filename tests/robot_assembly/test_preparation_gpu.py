"""Fixed-work real GPU equivalence for producer-prepared, frozen-world motion."""

import os

import mujoco
import numpy as np
import pytest

import castle_reasoning.data.assembly.teacher as teacher
import castle_reasoning.runtime.control.active as active
import castle_reasoning.runtime.control.batch_tracking as batch_tracking
import castle_reasoning.runtime.execution.preparation as preparation
import castle_reasoning.runtime.execution.segments as segments
import castle_reasoning.runtime.planning.collision as collision
import castle_reasoning.runtime.planning.manipulation as manipulation
import castle_reasoning.runtime.simulation.models as models
import castle_reasoning.runtime.simulation.world_slots as world_slots

pytestmark = pytest.mark.skipif(
    os.environ.get("CONSTRUCTION_GPU_TESTS") != "1", reason="Explicit GPU opt-in"
)


def test_producer_and_owner_preserve_complete_motion_with_frozen_sibling():
    goals = np.array([[0.425, 0, 0.01]])
    model, home = models.make_model([[0.02] * 3], goals)
    model.opt.timestep = 0.002
    model.opt.solver = mujoco.mjtSolver.mjSOL_CG
    model.opt.iterations = 1000
    model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    collector = teacher.make_collector(model, goals, 2, 1, 0)
    collector.engine.model.block_dim.linesearch_iterative = 128
    controller = collector.controller = active.ActivePathControl(collector)
    scene = collision.Scene(model, home)
    results = []
    for producer in (False, True):
        world_slots.reset(
            controller,
            np.ones(2, bool),
            qpos=np.tile(home, (2, 1)),
            qvel=np.zeros((2, model.nv)),
        )
        sources = [preparation.Source(w, 1, 0, 1, "model") for w in range(2)]
        executor = segments.SegmentExecutor(
            controller,
            pause_planning_worlds=True,
            skip_paused_preparation=producer,
            preparation_sources=sources if producer else None,
        )
        requests = []
        for w, angle in enumerate((0.0, 0.8)):
            path = np.tile(home, (2, 1))
            path[-1, 5] += angle
            request = batch_tracking.Request(
                manipulation.Segment("transit", path, np.zeros(2)),
                scene,
                -1,
                guard=False,
            )
            if producer:
                request = preparation.prepare(
                    request, collector.engine.snapshot()["qpos"][w : w + 1], sources[w]
                )
            requests.append(request)
        short, long = requests
        first = executor.advance(requests)
        assert first[0] is not None and first[1] is None
        assert first[0][1]["failure"] == 0
        frozen = collector.engine.data.qpos.numpy()[0].copy()
        wait = batch_tracking.Request(
            manipulation.Segment("batch_wait", home[None], np.zeros(1)),
            scene,
            -1,
            guard=False,
        )
        while executor.pending[1] is not None:
            completed = executor.advance([wait, long])
            assert completed[0][1]["executed_control_steps"] == 0
            np.testing.assert_array_equal(collector.engine.data.qpos.numpy()[0], frozen)
        assert completed[1][1]["failure"] == 0
        results.append((first[0], completed[1]))
    for (actual, event), (expected, reference) in zip(
        results[1], results[0], strict=True
    ):
        for key in actual:
            np.testing.assert_allclose(actual[key], expected[key], atol=1e-7, rtol=1e-6)
        for key in (
            "control_steps",
            "requested_steps",
            "failure",
            "device_contact_violations",
        ):
            assert event[key] == reference[key]
