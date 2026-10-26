"""The real stage loop with scripted planning and motion: rollback bookkeeping only."""

import time
from types import SimpleNamespace

import numpy as np
import pytest

import castle_reasoning.data.assembly.collect as collect
import castle_reasoning.runtime.planning.arm as arm
import castle_reasoning.runtime.planning.grasps as grasps
import castle_reasoning.runtime.simulation.robot as robot

STEPS = 30  # per move
HOLD = 51  # the final quiet second the stage's hold check reads


def two_cube_plan():
    return {
        "target_id": "R",
        "sizes": [[0.02] * 3] * 2,
        "target": [[0.40, 0.0, 0.01], [0.45, 0.0, 0.01]],
        "parking": [[0.30, 0.20, 0.01], [0.30, 0.15, 0.01]],
        "supports": [[0, -1], [1, -1]],
        "parking_supports": [[0, -1], [1, -1]],
        "removal_order": [0, 1],
        "meters_per_grid": 0.01,
    }


class ScriptedRunner(collect.Runner):
    """Moves parts by teleporting along short traces; failures are injected."""

    def __init__(self, tmp_path, max_rollbacks, behaviour):
        self.plan = two_cube_plan()
        self.output = tmp_path
        self.deadline = time.perf_counter() + 60
        self.max_rollbacks = max_rollbacks
        self.behaviour = behaviour  # (phase, part, removed parts) -> None | failure
        self.qpos = np.zeros(28)
        self.qpos[:6] = robot.HOME
        for i, position in enumerate(self.plan["target"]):
            self.qpos[14 + 7 * i : 17 + 7 * i] = position
            self.qpos[17 + 7 * i] = 1
        self.qvel = np.zeros(26)
        self.restored = []
        self.trace, self.events, self.phases = [], [], []
        self.boundaries, self.recoveries, self.placements = [], [], []
        self.rollbacks = []
        self.planner = SimpleNamespace(plan=self.plan_part, attempts=[])
        self.collector = SimpleNamespace(
            reset=lambda qpos, qvel: None,
            engine=SimpleNamespace(
                snapshot=lambda: {
                    "qpos": self.qpos[None].copy(),
                    "qvel": self.qvel[None].copy(),
                }
            ),
        )

    def state(self):
        return self.qpos.copy(), self.qvel.copy()

    def restore(self, qpos, qvel):
        self.qpos, self.qvel = qpos.copy(), qvel.copy()
        self.restored.append(qpos.copy())

    def removed(self):
        return frozenset(
            i
            for i in range(2)
            if not np.allclose(
                self.qpos[14 + 7 * i : 17 + 7 * i], self.plan["target"][i]
            )
        )

    def plan_part(self, qpos, part, goal, deadline):
        if self.behaviour("plan", part, self.removed()) == "unplannable":
            raise arm.PlanningError("no_complete_manipulation")
        return SimpleNamespace(
            part=part,
            grasp=grasps.Grasp((0, 0, 0), 0, 0.02),
            query_count=1,
            minimum_valid_clearance_m=0.01,
            attempts=[],
        )

    def regrasp(self, choices, goals):
        raise arm.PlanningError("no_feasible_removal_or_grasp")

    def move(self, result, destination):
        kind = self.behaviour("move", result.part, self.removed())
        start = self.qpos.copy()
        end = start.copy()
        end[14 + 7 * result.part : 17 + 7 * result.part] = destination
        if kind == "disturb":
            other = 1 - result.part
            end[14 + 7 * other] += 0.01
        frames = np.linspace(start, end, STEPS + 1)[1:]
        steps = STEPS if kind != "guard" else STEPS // 2
        self.trace.append(
            {
                "qpos": frames[:steps, None],
                "qvel": np.zeros((steps, 1, 26)),
                "action": np.zeros((steps, 1, 7)),
            }
        )
        self.events.append({"control_steps": steps, "grip_effort": 1.0})
        self.phases.extend([("move", result.part)] * steps)
        self.qpos = frames[steps - 1].copy()
        if kind == "guard":
            raise arm.PlanningError("forbidden_contact")

    def home(self):
        self.trace.append(
            {
                "qpos": np.repeat(self.qpos[None, None], HOLD, axis=0),
                "qvel": np.zeros((HOLD, 1, 26)),
                "action": np.zeros((HOLD, 1, 7)),
            }
        )
        self.events.append({"control_steps": HOLD, "grip_effort": 1.0})
        self.phases.extend([("hold", -1)] * HOLD)

    def _finish_stage(self):
        pass


def run(tmp_path, max_rollbacks, behaviour):
    runner = ScriptedRunner(tmp_path, max_rollbacks, behaviour)
    check, episode = runner.stage("disassembly", np.array(runner.plan["parking"]))
    return runner, check, episode


def once(kind, part=0):
    """Fail the first move of `part` from the assembled target, then behave."""
    fired = []

    def behaviour(phase, moving, removed):
        if phase == "move" and moving == part and not removed and not fired:
            fired.append(moving)
            return kind
        return None

    return behaviour


def test_disturbed_scene_resumes_from_the_last_boundary_with_another_part(tmp_path):
    runner, check, episode = run(tmp_path, 2, once("disturb"))
    assert check["reason"] is None and check["passed"]
    assert check["completed_order"] == [1, 0]
    assert check["rollbacks"] == [
        {
            "resumed_at_move": 0,
            "reason": "whole_scene_boundary",
            "avoided_part": 0,
            "discarded_control_steps": STEPS,
        }
    ]
    # Only the kept branch is recorded, and it starts from the restored state.
    assert len(episode["action"]) == 2 * STEPS + HOLD
    np.testing.assert_allclose(runner.restored[0][14:17], runner.plan["target"][0])
    np.testing.assert_allclose(episode["qpos"][0], runner.restored[0])
    assert [b["part"] for b in check["boundaries"]] == [1, 0]
    assert len(check["segments"]) == 3


def test_without_an_allowance_the_first_failure_is_final(tmp_path):
    runner, check, _ = run(tmp_path, 0, once("disturb"))
    assert check["reason"] == "whole_scene_boundary" and not check["passed"]
    assert check["rollbacks"] == [] and runner.restored == []


def test_interrupted_motion_restores_before_the_partial_move(tmp_path):
    _, check, episode = run(tmp_path, 1, once("guard"))
    assert check["passed"] and check["completed_order"] == [1, 0]
    assert check["rollbacks"][0]["reason"] == "forbidden_contact"
    assert check["rollbacks"][0]["discarded_control_steps"] == STEPS // 2
    assert len(episode["action"]) == 2 * STEPS + HOLD


def test_dead_end_undoes_the_previous_move(tmp_path):
    def behaviour(phase, moving, removed):
        # Part 1 cannot be removed once part 0 is gone, but can before it.
        return "unplannable" if moving == 1 and 0 in removed else None

    _, check, episode = run(tmp_path, 1, behaviour)
    assert check["reason"] is None, check["reason"]
    assert check["passed"] and check["completed_order"] == [1, 0]
    assert check["rollbacks"] == [
        {
            "resumed_at_move": 0,
            "reason": "no_feasible_removal_or_grasp",
            "avoided_part": 0,
            "discarded_control_steps": STEPS,
        }
    ]
    assert len(episode["action"]) == 2 * STEPS + HOLD


def test_rollbacks_stop_at_the_allowance(tmp_path):
    _, check, _ = run(
        tmp_path,
        1,
        lambda phase, moving, removed: (
            "disturb" if phase == "move" and not removed else None
        ),
    )
    assert check["reason"] == "whole_scene_boundary"
    assert len(check["rollbacks"]) == 1


@pytest.mark.parametrize(
    "error",
    [
        arm.PlanningBudgetError("planning_cpu_budget"),
        arm.PlanningError("experiment_wall_budget"),
    ],
)
def test_budget_exhaustion_is_never_rolled_back(tmp_path, error):
    runner = ScriptedRunner(tmp_path, 3, lambda *_: None)
    checkpoints = [runner._checkpoint([], {0, 1}, np.zeros((2, 3)))]
    assert runner._rollback_target(error, checkpoints, moving=0) is None
