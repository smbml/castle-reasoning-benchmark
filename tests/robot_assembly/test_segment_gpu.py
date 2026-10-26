"""Opt-in real-device regression for independent segment failure handling."""

import os

import numpy as np
import pytest

import castle_reasoning.data.assembly.teacher as teacher
import castle_reasoning.runtime.control.batch_tracking as batch_tracking
import castle_reasoning.runtime.control.tracking as tracking
import castle_reasoning.runtime.execution.segments as segments
import castle_reasoning.runtime.planning.collision as collision
import castle_reasoning.runtime.planning.manipulation as manipulation
import castle_reasoning.runtime.simulation.models as models


@pytest.mark.skipif(
    os.environ.get("CONSTRUCTION_GPU_TESTS") != "1", reason="Explicit GPU opt-in"
)
def test_guard_returns_at_check_interval_without_losing_other_worlds_path():
    model, home = models.make_model([[0.02] * 3], [[0.425, 0, 0.01]])
    model.opt.timestep = 0.002
    collector = teacher.make_collector(model, np.array([[0.425, 0, 0.01]]), 2, 1, 0)
    collector.controller = tracking.PathControl(collector)
    collector.reset(qpos=np.tile(home, (2, 1)), qvel=np.zeros((2, model.nv)))
    scene = collision.Scene(model, home)
    requests = []
    for world, angle in enumerate((0.6, 0.36)):
        path = np.array([home, home])
        path[1, 5] += angle
        expected = home.copy()
        if world == 1:
            expected[14] += 0.02  # Deliberately violate only this scene guard.
        requests.append(
            batch_tracking.Request(
                manipulation.Segment("transit", path, np.zeros(2)), scene, -1, expected
            )
        )
    executor = segments.SegmentExecutor(collector.controller)
    result = executor.advance(requests)
    assert executor.last_results[0][1]["executed_control_steps"] == 64
    assert result[0] is None
    assert result[1][1]["failure"] == 2
    actual = collector.engine.snapshot()["qpos"][1]
    wait = batch_tracking.Request(
        manipulation.hold_segment(actual, ticks=2, name="batch_wait"), scene, -1
    )
    result = executor.advance([requests[0], wait])
    assert result[0][1]["failure"] == 0
    assert result[0][1]["control_steps"] == result[0][1]["requested_steps"] == 120
    assert result[0][1]["padding_control_steps"] == 0
