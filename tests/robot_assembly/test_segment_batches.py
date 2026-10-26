"""Real native reference preparation with an analytic simulator boundary."""

from types import SimpleNamespace

import numpy as np
import pytest

import castle_reasoning.runtime.control.batch_tracking as batch_tracking
import castle_reasoning.runtime.execution.preparation as preparation
import castle_reasoning.runtime.execution.segments as segments
import castle_reasoning.runtime.planning.collision as collision
import castle_reasoning.runtime.planning.manipulation as manipulation
import castle_reasoning.runtime.simulation.models as models


def setup_executor(monkeypatch):

    model, home = models.make_model([[0.02] * 3], [[0.425, 0, 0.01]])
    scene = collision.Scene(model, home)
    snapshot = {"qpos": np.tile(home, (2, 1))}
    controller = SimpleNamespace(engine=SimpleNamespace(snapshot=lambda: snapshot))
    chunks = []

    def simulate(controller, requests, prepared, deadline, *, stop_on_failure):
        chunks.append(prepared)
        results = []
        for world in range(2):
            count = len(prepared.references)
            qpos = np.tile(snapshot["qpos"][world], (count, 1, 1))
            qpos[:, 0, :6] = prepared.references[:, world, :6]
            snapshot["qpos"][world] = qpos[-1, 0]
            trace = {
                "qpos": qpos,
                "qvel": np.zeros((count, 1, model.nv)),
                "action": prepared.references[:, world : world + 1].copy(),
            }
            results.append(
                (
                    trace,
                    {
                        "failure": 0,
                        "health_error": None,
                        "device_contact_violations": [],
                        "simulator_seconds": 0.1,
                        "batch_worlds": 2,
                        "world": world,
                        "requested_steps": count,
                        "executed_control_steps": count,
                        "control_steps": count,
                        "padding_control_steps": 0,
                        "discarded_suffix_steps": 0,
                        "tracking_error_rad": 0.0,
                    },
                )
            )
        return results

    monkeypatch.setattr(batch_tracking, "execute_prepared", simulate)

    def request(distance, name="transit"):
        path = np.tile(home, (2, 1))
        path[1, 0] += distance
        return batch_tracking.Request(
            manipulation.Segment(name, path, np.zeros(2)), scene, -1
        )

    return segments.SegmentExecutor(controller), request, chunks, home


def test_splits_preserve_full_reference_and_feedforward(monkeypatch):
    executor, request, chunks, home = setup_executor(monkeypatch)
    short, long = request(0), request(0.6)
    expected = batch_tracking.prepare([long], home[None])
    first = executor.advance([short, long])
    assert first[0] is not None and first[1] is None
    result = executor.advance([request(0, "batch_wait"), long])
    assert all(value is not None for value in result)
    actual_refs = np.concatenate([c.references[:, 1] for c in chunks])
    actual_bias = np.concatenate([c.bias[:, 1] for c in chunks])
    np.testing.assert_array_equal(actual_refs, expected.references[:, 0])
    np.testing.assert_array_equal(actual_bias, expected.bias[:, 0])
    trace, event = result[1]
    np.testing.assert_array_equal(trace["action"][:, 0], actual_refs)
    assert event["requested_steps"] == event["control_steps"] == len(actual_refs)
    assert event["padding_control_steps"] == 0
    assert event["simulator_seconds"] == pytest.approx(0.2)
    assert len(chunks[1].references) > len(chunks[0].references)


def test_cannot_replace_an_unfinished_segment(monkeypatch):
    executor, request, _, _ = setup_executor(monkeypatch)
    executor.advance([request(0), request(0.6)])
    with pytest.raises(ValueError, match="unfinished"):
        executor.advance([request(0), request(0.6)])


def test_failure_completes_pending_segment_and_offsets_contact_index(monkeypatch):
    executor, request, chunks, _ = setup_executor(monkeypatch)
    short, long = request(0), request(0.6)
    executor.advance([short, long])
    simulate = batch_tracking.execute_prepared

    def fail(*args, **kwargs):
        results = simulate(*args, **kwargs)
        results[1][1].update(
            failure=4, device_contact_violations=[{"control_index": 3}]
        )
        return results

    monkeypatch.setattr(batch_tracking, "execute_prepared", fail)
    results = executor.advance([request(0), long])
    event = results[1][1]
    assert event["failure"] == 4
    assert event["control_steps"] < event["requested_steps"]
    assert event["device_contact_violations"] == [
        {"control_index": len(chunks[0].references) + 3}
    ]


def test_frozen_slots_skip_native_preparation_and_reactivate(monkeypatch):

    original, request, chunks, home = setup_executor(monkeypatch)
    original.controller.active_worlds = np.ones(2, bool)
    executor = segments.SegmentExecutor(
        original.controller, pause_planning_worlds=True,
        skip_paused_preparation=True,
    )
    native_prepare = batch_tracking.prepare
    prepared_names = []

    def prepare(requests, initial):
        prepared_names.extend(r.segment.name for r in requests)
        return native_prepare(requests, initial)

    simulate = batch_tracking.execute_prepared

    def masked_simulate(*args, active_worlds, **kwargs):
        prepared = args[2]
        for w in np.flatnonzero(~active_worlds):
            assert prepared.guards[w] == prepared.attachments[w] == 0
            assert np.all(prepared.contacts[w] == -1e6)
        return simulate(*args, **kwargs)

    monkeypatch.setattr(batch_tracking, "prepare", prepare)
    monkeypatch.setattr(batch_tracking, "execute_prepared", masked_simulate)
    long = request(0.6)
    expected = native_prepare([long], home[None])
    while True:
        result = executor.advance([request(0, "batch_wait"), long])
        if result[1] is not None:
            break
    assert prepared_names == ["transit"]
    np.testing.assert_array_equal(
        np.concatenate([c.bias[:, 1] for c in chunks]), expected.bias[:, 0]
    )
    executor.advance([request(0), request(0, "batch_wait")])
    assert prepared_names == ["transit", "transit"]


def test_skipping_holds_requires_frozen_physics(monkeypatch):

    original, _, _, _ = setup_executor(monkeypatch)
    with pytest.raises(ValueError, match="paused"):
        segments.SegmentExecutor(original.controller, skip_paused_preparation=True)


def test_executor_uses_producer_arrays_and_rejects_stale_source(monkeypatch):
    pass

    original, request, _, home = setup_executor(monkeypatch)
    original.controller.active_worlds = np.ones(2, bool)
    sources = [preparation.Source(w, 1, 0, 1, "model") for w in range(2)]
    executor = segments.SegmentExecutor(
        original.controller, pause_planning_worlds=True, skip_paused_preparation=True,
        preparation_sources=sources,
    )
    prepared = [preparation.prepare(request(0), home[None], s) for s in sources]
    simulate = batch_tracking.execute_prepared
    monkeypatch.setattr(batch_tracking, "execute_prepared",
                        lambda *a, active_worlds, **kw: simulate(*a, **kw))

    def forbidden(*args):
        raise AssertionError("Owner attempted native preparation")

    monkeypatch.setattr(batch_tracking, "prepare", forbidden)
    executor.advance(prepared)
    assert [s.source_revision for s in sources] == [1, 1]
    with pytest.raises(ValueError, match="source"):
        executor.advance(prepared)
