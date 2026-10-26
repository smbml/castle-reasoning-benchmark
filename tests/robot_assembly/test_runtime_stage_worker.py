"""Real stage coordination, native planning, IPC and persistence without a GPU.

Only the final device execution is replaced with a quiet native-state trace.
The group, process worker, runner, preparation and accounting are real.
"""

import pickle
import time
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from castle_reasoning.data.assembly import batched_collect
from castle_reasoning.runtime.control import batch_tracking
from castle_reasoning.runtime.execution import session, workers
from castle_reasoning.runtime.simulation import models


@pytest.mark.parametrize("method", workers.START_METHODS)
@pytest.mark.parametrize("mode", ["runtime", "data"])
def test_default_stage_worker_and_coordinator(tmp_path, monkeypatch, method, mode):
    model, home = models.make_model([[0.02] * 3], [[0.425, 0, 0.01]])
    model_path = tmp_path / "model.mjb"
    mujoco.mj_saveModel(model, str(model_path), None)
    plan = {
        "sizes": [[0.02] * 3], "target": [[0.425, 0, 0.01]],
        "parking": [[0.425, 0, 0.01]], "fixed_parts": [0],
        "supports": [[0, -1]], "parking_supports": [[0, -1]],
        "removal_order": [0], "meters_per_grid": 0.01,
    }
    physical = {"qpos": home[None].astype(np.float32),
                "qvel": np.zeros((1, model.nv), np.float32)}

    def snapshot():
        return {key: value.copy() for key, value in physical.items()}

    def reset(qpos, qvel):
        physical.update(qpos=np.array(qpos, copy=True), qvel=np.array(qvel, copy=True))

    engine = SimpleNamespace(native_model=model, native_models=[model], nworld=1,
                             snapshot=snapshot)
    controller = SimpleNamespace(engine=engine)
    collector = SimpleNamespace(engine=engine, controller=controller, reset=reset)

    def device_execution(controller, requests, prepared, deadline, *, stop_on_failure):
        count = len(prepared.references)
        positions = np.repeat(physical["qpos"][None], count, axis=0)
        positions[:, :, :6] = prepared.references[:, :, :6]
        physical["qpos"] = positions[-1].copy()
        action = np.zeros((count, 1, 7), np.float32)
        action[..., 6] = -1
        trace = {"qpos": positions, "qvel": np.zeros((count, 1, model.nv), np.float32),
                 "action": action}
        event = {
            "failure": 0, "health_error": None, "device_contact_violations": [],
            "simulator_seconds": 0.0, "batch_worlds": 1, "world": 0,
            "requested_steps": count, "executed_control_steps": count,
            "control_steps": count, "padding_control_steps": 0,
            "discarded_suffix_steps": 0, "tracking_error_rad": 0.0,
        }
        return [(trace, event)]

    monkeypatch.setattr(batch_tracking, "execute_prepared", device_execution)
    group_type, runner_type = (
        (session.ConstructionGroup, session.Runner) if mode == "runtime"
        else (batched_collect.ConstructionGroup, batched_collect.Runner)
    )
    group = group_type(collector, time.perf_counter() + 30, [model_path])
    group.planner_start_method = method
    group.runners = [runner_type(group, 0, plan, tmp_path, 2, group.deadline, seed=1)]
    assert pickle.loads(pickle.dumps(group.stage_worker)) is group.stage_worker
    check, episode = group.run_stage(
        "terminal_check", [np.array(plan["target"])], [True], orders=[[]],
    )[0]
    assert check["passed"] and check["completed_order"] == []
    assert check["oracle_completed"] and check["reason"] is None
    assert len(episode["qpos"]) == len(episode["action"]) + 1
    assert len(episode["grip_effort"]) == len(episode["action"])
    assert group.control_steps == len(episode["action"]) >= 51
    assert (tmp_path / "terminal_check_checks.json").is_file()
    assert (tmp_path / "terminal_check_segments.json").is_file()
