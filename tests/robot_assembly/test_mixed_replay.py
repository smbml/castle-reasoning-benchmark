"""Recorded actions and planned references share physics without changing replay."""

import os
from types import SimpleNamespace

import numpy as np
import pytest

import castle_reasoning.data.assembly.batched_collect as batched_collect
import castle_reasoning.data.assembly.teacher as teacher
import castle_reasoning.runtime.control.batch_tracking as batch_tracking
import castle_reasoning.runtime.control.tracking as tracking
import castle_reasoning.runtime.execution.segments as segments
import castle_reasoning.runtime.planning.collision as collision
import castle_reasoning.runtime.planning.manipulation as manipulation
import castle_reasoning.runtime.simulation.models as models


def requests(model, home):
    scene = collision.Scene(model, home)
    path = np.array([home, home])
    path[1, 5] += 0.6
    planned = batch_tracking.Request(
        manipulation.Segment("transit", path, np.zeros(2)), scene, -1
    )
    actions = np.zeros((100, 7), np.float32)
    actions[:50, 5], actions[50:, 5], actions[:, 6] = 0.05, -0.05, -1
    replay = batch_tracking.Request(
        manipulation.Segment("replay", path, np.zeros(2)),
        scene,
        -1,
        guard=False,
        monitor_scene=False,
        actions=actions,
        efforts=np.linspace(0.8, 1.2, len(actions), dtype=np.float32),
    )
    return planned, replay


def test_preparation_preserves_literal_actions_and_stepwise_gripper_forces():
    model, home = models.make_model([[0.02] * 3], [[0.425, 0, 0.01]])
    planned, replay = requests(model, home)
    prepared = batch_tracking.prepare([planned, replay], np.tile(home, (2, 1)))
    np.testing.assert_array_equal(prepared.literal_mask, [0, 1])
    np.testing.assert_array_equal(prepared.literal_actions[:100, 1], replay.actions)
    np.testing.assert_array_equal(prepared.efforts[:100, 1], replay.efforts)
    np.testing.assert_array_equal(prepared.efforts[:, 0], 1)
    np.testing.assert_array_equal(prepared.literal_actions[100:, 1, :6], 0)
    assert prepared.lengths[1] == 100
    replay.guard = True
    with pytest.raises(ValueError, match="unguarded"):
        batch_tracking.prepare([replay], home[None])


def test_idle_slot_after_replay_holds_without_repeating_old_actions():
    model, home = models.make_model([[0.02] * 3], [[0.425, 0, 0.01]])
    _, replay = requests(model, home)
    group = SimpleNamespace(snapshot={"qpos": home[None]}, last=[replay])
    waiting = batched_collect.ConstructionGroup.waiting(group, 0)
    assert waiting.actions is None
    assert waiting.efforts is None
    assert waiting.segment.name == "batch_wait"


@pytest.mark.skipif(
    os.environ.get("CONSTRUCTION_GPU_TESTS") != "1", reason="GPU opt-in"
)
def test_mixed_replay_matches_independent_literal_physics():
    goal = np.array([[0.425, 0, 0.01]])
    model, home = models.make_model([[0.02] * 3], goal)
    model.opt.timestep = 0.002
    collector = teacher.make_collector(model, goal, 2, 1, 0)
    collector.controller = tracking.PathControl(collector)
    initial = {"qpos": np.tile(home, (2, 1)), "qvel": np.zeros((2, model.nv))}
    collector.reset(**initial)
    planned, replay = requests(model, home)
    baseline, _ = batch_tracking.replay(
        collector.controller,
        np.repeat(replay.actions[:, None], 2, axis=1),
        np.repeat(replay.efforts[:, None], 2, axis=1),
    )
    collector.reset(**initial)
    executor = segments.SegmentExecutor(collector.controller)
    result = executor.advance([planned, replay])
    assert result[0] is None
    trace, event = result[1]
    assert event["failure"] == 0, event
    np.testing.assert_array_equal(trace["action"][:, 0], replay.actions)
    np.testing.assert_allclose(trace["qpos"][:, 0], baseline["qpos"][:, 1], atol=1e-6)
    np.testing.assert_array_equal(
        executor.last_results[1][0]["action"][:, 0], replay.actions
    )
