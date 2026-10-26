"""Requested stages use runtime execution and preserve data selection policy."""

import json
import pickle

import numpy as np
import pytest

from castle_reasoning.data.assembly import batched_collect, collect
from castle_reasoning.runtime.execution import runner, session
from tests.robot_assembly.test_stage_rollback import HOLD, STEPS, ScriptedRunner


def requested_runner(tmp_path, behaviour=lambda *_: None, rollbacks=0):
    # Reuse the analytic simulator boundary; stage/rollback logic is the real
    # runtime implementation, with no automatic teacher-order selection.
    class RequestedRunner(ScriptedRunner):
        stage = runner.Runner.stage
        removable = runner.Runner.removable

    state = RequestedRunner(tmp_path, rollbacks, behaviour)
    state.requested_goals = np.array(state.plan["parking"])
    # The harness snapshots current poses into parking before an arbitrary move.
    state.plan["parking"] = np.array(state.plan["target"]).tolist()
    return state


def test_runtime_exposes_the_entire_stage_operation():
    for name in ("stage", "_checkpoint", "_rollback_target", "_roll_back"):
        assert callable(getattr(session.Runner, name))
    for name in ("run_stage", "run_world"):
        assert callable(getattr(session.ConstructionGroup, name))
    operation = session.ConstructionGroup.stage_worker
    assert operation.__module__ == session.__name__
    assert pickle.loads(pickle.dumps(operation)) is operation
    assert batched_collect.ConstructionGroup.stage_worker is batched_collect.stage_worker


def test_runtime_follows_the_supplied_order_and_persists_aligned_traces(tmp_path):
    state = requested_runner(tmp_path)
    check, episode = state.stage("requested", state.requested_goals, [1, 0])
    assert check["passed"] and check["completed_order"] == [1, 0]
    assert len(episode["action"]) == 2 * STEPS + HOLD
    assert len(episode["qpos"]) == len(episode["action"]) + 1
    assert len(episode["grip_effort"]) == len(episode["action"])
    assert [b["part"] for b in check["boundaries"]] == [1, 0]
    assert json.loads((tmp_path / "requested_checks.json").read_text())["completed_order"] == [1, 0]
    with np.load(tmp_path / "requested.npz") as saved:
        for name, values in episode.items():
            np.testing.assert_array_equal(saved[name], values)


def test_runtime_empty_order_performs_the_same_terminal_hold(tmp_path):
    state = requested_runner(tmp_path)
    state.plan["fixed_parts"] = [0, 1]
    check, episode = state.stage("terminal", np.array(state.plan["target"]), [])
    assert check["passed"] and check["completed_order"] == []
    assert len(episode["action"]) == HOLD
    assert check["rollbacks"] == []


def test_runtime_fixed_order_rolls_back_an_interrupted_move(tmp_path):
    failures = []

    def behaviour(phase, part, removed):
        if phase == "move" and part == 1 and not removed and not failures:
            failures.append(part)
            return "guard"

    state = requested_runner(tmp_path, behaviour, rollbacks=1)
    check, episode = state.stage("requested", state.requested_goals, [1, 0])
    assert check["passed"] and check["completed_order"] == [1, 0]
    assert len(check["rollbacks"]) == 1
    assert check["rollbacks"][0]["reason"] == "forbidden_contact"
    assert len(episode["action"]) == 2 * STEPS + HOLD


def test_runtime_does_not_choose_a_demonstration_order(tmp_path):
    state = requested_runner(tmp_path)
    with pytest.raises(ValueError, match="order"):
        state.stage("requested", np.array(state.plan["parking"]))


def test_data_runner_retains_automatic_removal_selection(tmp_path):
    state = ScriptedRunner(tmp_path, 0, lambda *_: None)
    assert collect.Runner.stage is runner.Runner.stage
    check, _ = state.stage("disassembly", np.array(state.plan["parking"]))
    assert check["passed"] and check["completed_order"] == [0, 1]
