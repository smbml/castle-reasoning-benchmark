"""Actual threaded stage rendezvous; the analytic executor tests scheduling only."""

import concurrent.futures
import threading

import pytest

import castle_reasoning.runtime.execution.batching as batching


def test_independent_segments_exclude_finished_world_holds_from_episodes():
    executions = []

    def execute(requests):
        steps = max(r["steps"] for r in requests)
        executions.append(requests)
        return [(r["name"], steps) for r in requests]

    group = batching.StageGroup(
        2, execute, lambda w: {"name": f"wait{w}", "steps": 1}
    )

    def short():
        result = group.segment(0, {"name": "short", "steps": 1})
        return result, group.finish(0)

    def long():
        first = group.segment(1, {"name": "long1", "steps": 2})
        second = group.segment(1, {"name": "long2", "steps": 3})
        return first, second, group.finish(1)

    result = group.run([short, long])
    assert result == [
        (("short", 2), []),
        (("long1", 2), ("long2", 3), []),
    ]
    assert len(executions) == 2
    assert all(len(requests) == 2 for requests in executions)


def test_short_worker_finishes_before_long_worker_submits_next_segment():
    ended = threading.Event()
    group = batching.StageGroup(2, lambda requests: requests, lambda w: "wait")

    def short():
        group.segment(0, "short")
        result = group.finish(0)
        ended.set()
        return result

    def long():
        group.segment(1, "first")
        assert ended.wait(timeout=1), "Finished worker is blocked by its sibling"
        group.segment(1, "second")
        return group.finish(1)

    assert group.run([short, long]) == [[], []]


@pytest.mark.parametrize("failure_site", ["worker", "executor", "missing_finish"])
def test_worker_or_executor_failure_releases_pending_threads(failure_site):
    def execute(requests):
        if failure_site == "executor":
            raise ValueError("executor fault")
        return requests

    group = batching.StageGroup(2, execute, lambda w: w)

    def worker_zero():
        if failure_site == "worker":
            raise ValueError("worker fault")
        if failure_site == "missing_finish":
            return None
        group.segment(0, 0)
        return group.finish(0)

    def worker_one():
        group.segment(1, 1)
        return group.finish(1)

    # A deadlock must fail this test without hanging the test process forever.
    with concurrent.futures.ThreadPoolExecutor(1) as pool:
        future = pool.submit(group.run, [worker_zero, worker_one])
        with pytest.raises((ValueError, RuntimeError)):
            future.result(timeout=3)


def test_all_finished_stage_needs_no_spurious_simulation():
    def unexpected(_):
        raise AssertionError("No segment should execute")

    group = batching.StageGroup(2, unexpected, unexpected)
    assert group.run([lambda: group.finish(0), lambda: group.finish(1)]) == [[], []]


def test_pending_long_segment_survives_another_world_advancing():
    calls = []

    def execute(requests):
        calls.append(requests)
        return ["short done", None] if len(calls) == 1 else requests

    group = batching.StageGroup(2, execute, lambda w: "wait")

    def short():
        assert group.segment(0, "short") == "short done"
        assert group.segment(0, "next") == "next"
        return group.finish(0)

    def long():
        assert group.segment(1, "long") == "long"
        return group.finish(1)

    assert group.run([short, long]) == [[], []]
    assert calls == [["short", "long"], ["next", "long"]]
