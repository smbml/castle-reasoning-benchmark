"""Planner compute, admission, and experiment time have separate budgets."""

import multiprocessing
import threading
import time

import pytest

import castle_reasoning.runtime.planning.arm as arm
import castle_reasoning.runtime.planning.budget as planning_budget


def test_waiting_does_not_spend_cpu_budget():
    budget = planning_budget.Budget(watchdog_seconds=1)
    with budget.scope(0.02):
        time.sleep(0.05)
        budget.check()


def test_admission_wait_precedes_local_clocks():
    permit = multiprocessing.get_context("spawn").BoundedSemaphore(1)
    permit.acquire()
    release = threading.Timer(0.1, permit.release)
    budget = planning_budget.Budget(permits=permit, watchdog_seconds=0.05)
    release.start()
    try:
        with budget.scope(0.02, time.perf_counter() + 2):
            budget.check()
    finally:
        release.join()
    assert budget.last_query["admission_seconds"] >= 0.05
    assert permit.acquire(timeout=0)
    permit.release()


def test_child_cpu_limit_leaves_remaining_parent_allowance():
    budget = planning_budget.Budget()
    with budget.scope(1):
        with pytest.raises(arm.PlanningBudgetError, match="planning_cpu_budget"):
            with budget.scope(1e-9):
                budget.check()
        budget.check()


def test_cpu_work_exhausts_local_budget_and_releases_permit():
    permit = multiprocessing.get_context("spawn").BoundedSemaphore(1)
    budget = planning_budget.Budget(permits=permit)
    with pytest.raises(arm.PlanningBudgetError, match="planning_cpu_budget"):
        with budget.scope(0.01):
            while True:
                budget.check()
    assert permit.acquire(timeout=0)
    permit.release()


def test_nested_scope_reuses_permit_and_cannot_extend_parent_budget():
    permit = multiprocessing.get_context("spawn").BoundedSemaphore(1)
    budget = planning_budget.Budget(permits=permit)
    with pytest.raises(arm.PlanningBudgetError, match="planning_cpu_budget"):
        with budget.scope(0.01):
            with budget.scope(100):
                while True:
                    budget.check()
    assert permit.acquire(timeout=0)
    permit.release()


def test_global_deadline_bounds_admission_without_releasing_someone_elses_permit():
    permit = multiprocessing.get_context("spawn").BoundedSemaphore(1)
    permit.acquire()
    budget = planning_budget.Budget(permits=permit)
    with pytest.raises(arm.PlanningBudgetError, match="experiment_wall_budget"):
        with budget.scope(10, deadline=time.perf_counter() + 0.02):
            pytest.fail("Expired query admitted")
    assert not permit.acquire(timeout=0)
    permit.release()


@pytest.mark.parametrize("global_limit", [False, True])
def test_wall_limits_have_distinct_reasons(global_limit):
    budget = planning_budget.Budget(watchdog_seconds=0.02)
    deadline = time.perf_counter() + 0.01 if global_limit else float("inf")
    reason = "experiment_wall_budget" if global_limit else "planning_watchdog"
    with pytest.raises(arm.PlanningBudgetError, match=reason):
        with budget.scope(10, deadline):
            time.sleep(0.03)


def test_geometry_error_restores_budget_and_releases_permit():
    permit = multiprocessing.get_context("spawn").BoundedSemaphore(1)
    budget = planning_budget.Budget(permits=permit)
    with pytest.raises(arm.PlanningError, match="collision"):
        with budget.scope(1):
            raise arm.PlanningError("collision")
    assert permit.acquire(timeout=0)
    permit.release()
    with budget.scope(1):
        budget.check()


def test_connection_checks_budget_even_for_straight_paths():
    budget = planning_budget.Budget()
    with pytest.raises(arm.PlanningBudgetError, match="planning_cpu_budget"):
        with budget.scope(1e-9):
            arm.connect([0], [1], lambda _: True, [[0, 1]],
                                 check_budget=budget.check)
