"""Producer preparation uses actual source state and preserves native math."""

import dataclasses
import pickle
import threading

import numpy as np
import pytest

import castle_reasoning.runtime.control.batch_tracking as batch_tracking
import castle_reasoning.runtime.execution.batching as batching
import castle_reasoning.runtime.execution.preparation as preparation
import castle_reasoning.runtime.planning.collision as collision
import castle_reasoning.runtime.planning.manipulation as manipulation
import castle_reasoning.runtime.simulation.models as models


def fixture_request(literal=False):
    model, home = models.make_model([[0.02] * 3], [[0.425, 0, 0.01]])
    path = np.tile(home, (2, 1))
    path[-1, 0] += 0.12
    request = batch_tracking.Request(
        manipulation.Segment("transit", path, np.array([0.0, 0.3])),
        collision.Scene(model, home),
        -1,
    )
    if literal:
        request = dataclasses.replace(
            request,
            guard=False,
            monitor_scene=False,
            actions=np.zeros((13, 7), np.float32),
            efforts=np.linspace(1, 2, 13, dtype=np.float32),
        )
    return request, home[None]


@pytest.mark.parametrize("literal", [False, True])
def test_producer_arrays_match_reference_and_do_not_alias_inputs(literal):
    request, initial = fixture_request(literal)
    expected = batch_tracking.prepare([request], initial)
    source = preparation.Source(0, 1, 0, 1, "model-a")
    prepared = preparation.prepare(request, initial, source)
    for field in dataclasses.fields(expected):
        value = getattr(prepared.preparation.arrays, field.name)
        np.testing.assert_array_equal(value, getattr(expected, field.name))
        assert not value.flags.writeable
    request.segment.path[:] = 0
    assert np.any(prepared.segment.path)
    assert not prepared.segment.path.flags.writeable
    preparation.validate(prepared, initial, source)


@pytest.mark.parametrize(
    "field,value",
    [
        ("slot", 1),
        ("slot_epoch", 2),
        ("source_revision", 1),
        ("request_sequence", 2),
        ("model_id", "other-model"),
    ],
)
def test_mismatched_source_is_rejected(field, value):
    request, initial = fixture_request()
    source = preparation.Source(0, 1, 0, 1, "model-a")
    request = preparation.prepare(request, initial, source)
    with pytest.raises(ValueError, match="source"):
        preparation.validate(
            request, initial, dataclasses.replace(source, **{field: value})
        )


def test_changed_initial_state_and_invalid_array_shape_are_rejected():
    request, initial = fixture_request()
    source = preparation.Source(0, 1, 0, 1, "model-a")
    request = preparation.prepare(request, initial, source)
    altered = initial.copy()
    altered[0, 0] += 0.001
    with pytest.raises(ValueError, match="state"):
        preparation.validate(request, altered, source)
    request.preparation.arrays.bias = np.zeros((1, 1, 6), np.float32)
    with pytest.raises(ValueError, match="bias"):
        preparation.validate(request, initial, source)


def test_cancelled_preparation_does_not_wait_for_a_permit():
    request, initial = fixture_request()
    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(RuntimeError, match="cancel"):
        preparation.prepare(
            request,
            initial,
            preparation.Source(0, 1, 0, 1, "model-a"),
            permits=threading.Semaphore(0),
            cancelled=cancelled,
        )


def test_failed_preparation_returns_permit():
    request, initial = fixture_request(True)
    request.efforts[:] = -1
    permits = threading.BoundedSemaphore(1)
    with pytest.raises(ValueError, match="effort"):
        preparation.prepare(
            request,
            initial,
            preparation.Source(0, 1, 0, 1, "model-a"),
            permits=permits,
        )
    assert permits.acquire(blocking=False)


def test_transport_validation_restores_write_protection_on_all_inputs():
    request, initial = fixture_request(True)
    source = preparation.Source(0, 1, 0, 1, "model")
    ready = preparation.prepare(request, initial, source)
    received = pickle.loads(pickle.dumps(dataclasses.replace(ready, scene=None)))
    received.scene = request.scene
    preparation.validate(received, initial, source)
    for array in (received.segment.path, received.segment.closure,
                  received.actions, received.efforts,
                  received.preparation.initial_qpos):
        assert not array.flags.writeable


def test_blocked_preparation_does_not_block_ready_sibling():

    request, initial = fixture_request()
    ready_request = preparation.prepare(
        request, initial, preparation.Source(1, 1, 0, 1, "model")
    )
    permits = threading.BoundedSemaphore(1)
    assert permits.acquire()
    entered, advanced = threading.Event(), threading.Event()
    seen = []

    def execute(requests):
        seen.append(requests)
        return requests

    scheduler = batching.SlotGroup(
        2, execute, lambda w: None, lambda r: {}, pause_planning_worlds=True
    )

    def preparing():
        entered.set()
        value = preparation.prepare(
            request, initial, preparation.Source(0, 1, 0, 1, "model"), permits=permits
        )
        assert advanced.is_set()
        scheduler.segment(0, value)
        scheduler.close(0)

    def ready():
        assert entered.wait(5)
        scheduler.segment(1, ready_request)
        advanced.set()
        permits.release()
        scheduler.close(1)

    scheduler.run([preparing, ready])
    assert seen[0] == [None, ready_request]
