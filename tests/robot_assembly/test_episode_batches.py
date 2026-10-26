"""Independent episode lengths, exact stage handoffs and replay commands."""

import numpy as np
import pytest

import castle_reasoning.runtime.execution.batching as batching
import castle_reasoning.runtime.simulation.models as models
from castle_reasoning.runtime.execution import traces


def episode(home, nv, count, marker):
    qpos = np.tile(home, (count + 1, 1))
    qpos[:, 0] += np.arange(count + 1) * 0.001 * marker
    return {
        "qpos": qpos,
        "qvel": np.full((count + 1, nv), marker * 0.002),
        "action": np.full((count, 7), marker * 0.01, np.float32),
        "grip_effort": np.arange(count, dtype=np.float32) + 1,
    }


def test_next_stage_uses_exact_endpoint_including_velocity_and_disabled_world():
    model, home = models.make_model([[0.02] * 3], [[0.425, 0, 0.01]])
    samples = [episode(home, model.nv, 1, 1), None, episode(home, model.nv, 4, 2)]
    current = {"qpos": np.tile(home, (3, 1)), "qvel": np.zeros((3, model.nv))}
    current["qpos"][:, 0] += 0.5  # Passive evolution after each episode ended.
    original = {key: value.copy() for key, value in current.items()}
    result = batching.handoff(current, samples)
    for world in (0, 2):
        for key in ("qpos", "qvel"):
            np.testing.assert_array_equal(result[key][world], samples[world][key][-1])
    for key in ("qpos", "qvel"):
        np.testing.assert_array_equal(result[key][1], current[key][1])
        np.testing.assert_array_equal(current[key], original[key])


def test_ragged_replay_preserves_commands_effort_and_terminal_state():
    model, home = models.make_model([[0.02] * 3], [[0.425, 0, 0.01]])
    samples = [episode(home, model.nv, 1, 1), None, episode(home, model.nv, 4, 2)]
    actions, efforts, lengths = batching.replay_commands(samples)
    np.testing.assert_array_equal(lengths, [1, 0, 4])
    assert actions.shape == (4, 3, 7)
    for world in (0, 2):
        count = lengths[world]
        np.testing.assert_array_equal(actions[:count, world], samples[world]["action"])
        np.testing.assert_array_equal(
            efforts[:count, world], samples[world]["grip_effort"]
        )
    np.testing.assert_array_equal(actions[1:, 0, :6], 0)
    np.testing.assert_array_equal(actions[1:, 0, 6], samples[0]["action"][-1, 6])
    np.testing.assert_array_equal(efforts[1:, 0], samples[0]["grip_effort"][-1])
    np.testing.assert_array_equal(actions[:, 1, :6], 0)
    np.testing.assert_array_equal(actions[:, 1, 6], -1)

    # Only post-terminal padding is excluded; the original final state remains.
    initial = {"qpos": np.tile(home, (3, 1)), "qvel": np.zeros((3, model.nv))}
    trace = {key: np.tile(value, (4, 1, 1)) for key, value in initial.items()}
    trace["qpos"][0, 0] = samples[0]["qpos"][-1]
    trace["action"] = actions
    prefix = traces.episode(
        initial, {key: value[: lengths[0]] for key, value in trace.items()}, 0
    )
    assert len(prefix["qpos"]) == len(prefix["action"]) + 1 == 2
    np.testing.assert_array_equal(prefix["qpos"][-1], samples[0]["qpos"][-1])


def test_empty_replay_is_rejected():
    with pytest.raises(ValueError, match="nonempty"):
        batching.replay_commands([None, None])
