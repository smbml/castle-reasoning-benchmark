"""Residual-motion verification uses real replay/hold samples and strict gates."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import castle_reasoning.data.assembly.continuous_collect as continuous_collect
import castle_reasoning.runtime.control.parameters as parameters
import castle_reasoning.runtime.execution.success as success
import castle_reasoning.runtime.planning.collision as collision
import castle_reasoning.runtime.simulation.models as models

FIXTURE = Path(__file__).parent / "fixtures/t0134_replay_settle"


@pytest.fixture
def sample():
    with np.load(FIXTURE.with_suffix(".npz")) as stored:
        return {key: stored[key] for key in stored.files}


def replay_check(sample):
    check = success.verify_hold(
        sample["qpos"], sample["qvel"], sample["goals"], sample["sizes"]
    )
    check.update(
        contacts={"passed": True},
        dynamics={"passed": True},
        forbidden_contact_frames=0,
        execution={"failure": 0},
        complete=True,
    )
    return check


def test_real_replay_fails_only_settling(sample):
    check = replay_check(sample)
    assert not check["passed"]
    assert check["pose_valid"] and not check["settled"]
    assert continuous_collect.can_settle_replay(check)
    held = success.verify_hold(
        sample["hold_qpos"][:51],
        sample["hold_qvel"][:51],
        sample["goals"],
        sample["sizes"],
    )
    assert held["passed"]


@pytest.mark.parametrize(
    "change",
    [
        {"pose_valid": False},
        {"contacts": {"passed": False}},
        {"dynamics": {"passed": False}},
        {"forbidden_contact_frames": 1},
        {"execution": {"failure": 5}},
        {"complete": False},
        {"passed": True},
    ],
)
def test_wait_cannot_hide_other_replay_failures(sample, change):
    check = replay_check(sample)
    check.update(change)
    assert not continuous_collect.can_settle_replay(check)


@pytest.mark.parametrize("fault", ["position", "nonfinite", "arm"])
def test_invalid_pose_is_not_residual_motion(sample, fault):
    if fault == "position":
        sample["qpos"][-1, 14] += 0.02
    elif fault == "nonfinite":
        sample["qvel"][-1, 0] = np.nan
    else:
        sample["qpos"][-1, 0] += 0.2
    check = replay_check(sample)
    assert not check["pose_valid"]
    assert not continuous_collect.can_settle_replay(check)


@pytest.mark.parametrize(
    "case", ["settles", "disabled", "guard", "timeout", "empty_timeout", "unsettled"]
)
def test_verification_tail_does_not_change_replay(sample, tmp_path, case):
    plan = json.loads(FIXTURE.with_suffix(".json").read_text())["plan"]
    model, _ = models.compile_model(
        plan,
        parameters.ControlConfig(physics_timestep=0.002),
        contact_impedance=(0.95, 0.99),
    )
    scene = collision.Scene(model, model.qpos0, margin=-0.0002)
    requests = []

    def recorded_segment(world, request):
        """Replay one captured simulator response at the device boundary."""
        assert world == 0
        requests.append(request)
        qpos = sample["qpos"] if case == "unsettled" else sample["hold_qpos"]
        qvel = sample["qvel"] if case == "unsettled" else sample["hold_qvel"]
        count = 0 if case == "empty_timeout" else 50
        return {
            "qpos": qpos[1 : count + 1, None],
            "qvel": qvel[1 : count + 1, None],
            "action": sample["hold_action"][:count, None],
        }, {
            "failure": {"guard": 4, "timeout": 5, "empty_timeout": 5}.get(case, 0),
            "control_steps": count,
            "simulator_seconds": 0.5,
        }

    collection = object.__new__(continuous_collect.Collection)
    # Owner-thread preparation (the default) has no per-slot preparation sources.
    collection.preparation_sources = None
    collection.settings = {"replay_settle_seconds": 0 if case == "disabled" else 2}
    collection.stage = SimpleNamespace(segment=recorded_segment)
    collection.plans = [plan]
    collection.pool = SimpleNamespace(models=[model])
    collection.runners = [
        SimpleNamespace(
            planner=SimpleNamespace(scene=scene),
            audit_scene=scene,
        )
    ]
    replay = {key: sample[key].copy() for key in ("qpos", "qvel")}
    replay["action"] = np.zeros((50, 7))
    original = copy.deepcopy(replay)
    check = replay_check(sample)
    result = collection.settle_replay(0, 0, replay, check, tmp_path)
    assert all(np.array_equal(replay[k], original[k]) for k in replay)
    if case == "disabled":
        assert result is check and not requests
        assert not (tmp_path / "replay_settle.npz").exists()
        return
    expected_requests = 2 if case == "unsettled" else 1
    expected_steps = 0 if case == "empty_timeout" else 50 * expected_requests
    assert result["passed"] == (case == "settles")
    assert result["settling"]["control_steps"] == expected_steps
    assert not result["settling"]["initial_hold"]["passed"]
    assert len(requests) == expected_requests
    assert result["settling"]["execution_failure"] == {
        "guard": 4,
        "timeout": 5,
        "empty_timeout": 5,
    }.get(case, 0)
    assert requests[0].guard and not requests[0].monitor_scene
    with np.load(tmp_path / "replay_settle.npz") as tail:
        assert len(tail["qpos"]) == expected_steps + 1
        assert len(tail["action"]) == expected_steps
        np.testing.assert_array_equal(tail["qpos"][0], replay["qpos"][-1])
