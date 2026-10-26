
import numpy as np
import pytest

import castle_reasoning.data.assembly.records as assembly_records
from castle_reasoning.runtime.execution import traces


def test_episode_alignment_and_terminal_boundary():
    initial = {"qpos": np.ones((2, 5)), "qvel": np.ones((2, 4))}
    trace = {
        "qpos": np.ones((3, 2, 5)) * 2,
        "qvel": np.ones((3, 2, 4)) * 3,
        "action": np.zeros((3, 2, 7)),
    }
    episode = traces.episode(initial, trace, 1)
    assert len(episode["qpos"]) == len(episode["action"]) + 1
    np.testing.assert_equal(episode["qpos"][0], initial["qpos"][1])
    assert episode["time"][-1] == 0.06
    with pytest.raises(ValueError):
        traces.episode(initial, dict(trace, action=np.zeros((4, 2, 7))), 0)


def test_yield_counts_failures_and_accounting_includes_replay():
    records = [
        {"accepted": True, "rejection": None},
        {"accepted": False, "rejection": "assembly"},
        {"accepted": False, "rejection": "target_unstable"},
        {"accepted": True, "rejection": None},
    ]
    result = assembly_records.yield_summary(records, saved=2, data_seconds=10, transitions=300)
    assert result["target_yield"] == 0.5
    assert result["verified_trajectories_per_second"] == 0.2
    assert result["saved_action_targets_per_second"] == 30
    assert result["rejections"] == {"assembly": 1, "target_unstable": 1}
