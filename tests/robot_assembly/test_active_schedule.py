"""Planner progress must not gate a sibling's ready segment."""

import threading

import castle_reasoning.runtime.execution.batching as batching


def test_ready_world_advances_while_sibling_is_planning():
    advanced = threading.Event()
    requests_seen = []

    def execute(requests):
        requests_seen.append(requests)
        advanced.set()
        return requests

    scheduler = batching.SlotGroup(
        2,
        execute,
        lambda w: "paused",
        lambda requests: {w: None for w in requests},
        pause_planning_worlds=True,
    )

    def planning():
        assert advanced.wait(2), "Ready sibling was blocked by this planner"
        assert scheduler.segment(0, "resumed") == "resumed"
        scheduler.close(0)

    def ready():
        assert scheduler.segment(1, "ready") == "ready"
        scheduler.close(1)

    scheduler.run([planning, ready])
    assert ["paused", "ready"] in requests_seen
