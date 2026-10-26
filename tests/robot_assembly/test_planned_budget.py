"""Global-budget censoring must not erase physical failures or local limits."""

import pytest

from castle_reasoning.runtime.execution import runner as execution_runner


@pytest.mark.parametrize(
    "reason",
    [
        "manipulation_planning_budget",
        "motion_search_budget",
        "no_feasible_removal_or_grasp",
        "no_complete_manipulation:manipulation_planning_budget",
    ],
)
def test_planner_budget_is_censored_only_when_the_wall_budget_is_exhausted(reason):
    assert execution_runner.stage_error_reason(reason, True) == "experiment_wall_budget"
    assert execution_runner.stage_error_reason(reason, False) == reason


@pytest.mark.parametrize(
    "reason",
    ["execution_guard:place:4:contacts=1", "whole_scene_boundary", "ik_residual"],
)
def test_wall_deadline_does_not_relabel_a_physical_or_geometric_failure(reason):
    assert execution_runner.stage_error_reason(reason, True) == reason
