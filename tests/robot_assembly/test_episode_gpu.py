"""Opt-in physics evidence for independent episode endings and replay."""

import functools
import os
from types import SimpleNamespace

import numpy as np
import pytest

import castle_reasoning.data.assembly.batched_collect as batched_collect
import castle_reasoning.data.assembly.teacher as teacher
import castle_reasoning.runtime.control.batch_tracking as batch_tracking
import castle_reasoning.runtime.control.tracking as tracking
import castle_reasoning.runtime.execution.batching as batching
import castle_reasoning.runtime.execution.segments as segments
import castle_reasoning.runtime.execution.success as success
import castle_reasoning.runtime.planning.collision as collision
import castle_reasoning.runtime.planning.manipulation as manipulation
import castle_reasoning.runtime.simulation.models as models
from castle_reasoning.runtime.execution import traces


@pytest.mark.skipif(
    os.environ.get("CONSTRUCTION_GPU_TESTS") != "1", reason="Explicit GPU opt-in"
)
def test_different_episode_lengths_replay_from_exact_initial_states(tmp_path):
    goal = np.array([[0.425, 0, 0.01]])
    model, home = models.make_model([[0.02] * 3], goal)
    model.opt.timestep = 0.002
    collector = teacher.make_collector(model, goal, 2, 1, 0)
    collector.controller = tracking.PathControl(collector)
    collector.reset(qpos=np.tile(home, (2, 1)), qvel=np.zeros((2, model.nv)))
    initial = collector.engine.snapshot()
    group = batched_collect.ConstructionGroup(
        collector, float("inf"), [tmp_path / "a", tmp_path / "b"]
    )
    scenes = [collision.Scene(model, home) for _ in range(2)]
    group.runners = [SimpleNamespace(planner=SimpleNamespace(scene=s)) for s in scenes]
    group.segments = segments.SegmentExecutor(collector.controller)
    group.stage = batching.StageGroup(2, group.advance, group.waiting)

    def worker(world):
        path = np.array([home, home])
        path[1, 5] += (0.12, 0.6)[world]
        request = batch_tracking.Request(
            manipulation.Segment("transit", path, np.zeros(2)), scenes[world], -1
        )
        trace, event = group.stage.segment(world, request)
        assert event["failure"] == 0
        final = trace["qpos"][-1, 0]
        returning = batch_tracking.Request(
            manipulation.Segment("home", np.array([final, home]), np.zeros(2)),
            scenes[world],
            -1,
        )
        back, event = group.stage.segment(world, returning)
        assert event["failure"] == 0
        final = back["qpos"][-1, 0]
        hold = batch_tracking.Request(
            manipulation.hold_segment(final), scenes[world], -1
        )
        tail, event = group.stage.segment(world, hold)
        assert event["failure"] == 0
        assert group.stage.finish(world) == []
        sample = traces.episode(
            {key: value[world : world + 1] for key, value in initial.items()},
            {key: np.concatenate([trace[key], back[key], tail[key]]) for key in trace},
            0,
        )
        sample["grip_effort"] = np.ones(len(sample["action"]))
        return sample

    samples = group.stage.run([functools.partial(worker, w) for w in range(2)])
    actions, efforts, lengths = batching.replay_commands(samples)
    assert lengths[0] < lengths[1]
    assert group.wait_steps > 0
    assert group.control_steps == 2 * lengths.max()

    collector.reset(qpos=initial["qpos"], qvel=initial["qvel"])
    replay, _ = batch_tracking.replay(collector.controller, actions, efforts)
    for world, count in enumerate(lengths):
        prefix = traces.episode(
            initial, {key: value[:count] for key, value in replay.items()}, world
        )
        np.testing.assert_array_equal(prefix["action"], samples[world]["action"])
        np.testing.assert_allclose(prefix["qpos"], samples[world]["qpos"], atol=1e-6)
        check = success.verify_hold(prefix["qpos"], prefix["qvel"], goal, [[0.02] * 3])
        assert check["passed"], check
    terminal = batching.handoff(collector.engine.snapshot(), samples)
    collector.reset(qpos=terminal["qpos"], qvel=terminal["qvel"])
    actual = collector.engine.snapshot()
    for key in ("qpos", "qvel"):
        np.testing.assert_array_equal(actual[key], terminal[key])
