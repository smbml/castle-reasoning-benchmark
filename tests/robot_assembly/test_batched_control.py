"""Different worlds must retain independent commands and safety constraints."""

import numpy as np
import warp as wp

import castle_reasoning.runtime.control.batch_tracking as batch_tracking
import castle_reasoning.runtime.control.tracking as tracking
import castle_reasoning.runtime.planning.collision as collision
import castle_reasoning.runtime.planning.manipulation as manipulation
import castle_reasoning.runtime.simulation.models as simulation_models


def array(value, dtype=float):
    return wp.array(np.asarray(value), dtype=dtype, device="cpu")


def test_contact_allowances_cannot_leak_between_worlds():
    pairs = array([[0, 1], [0, 1]], dtype=wp.vec2i)
    count = array([2], dtype=int)
    worlds = array([0, 1], dtype=int)
    distances = array([-0.0007, -0.0007])
    limits = np.full((2, 2, 2), -0.0002)
    limits[0, 0, 1] = -0.001
    failure = wp.zeros(2, dtype=int, device="cpu")
    step = array([7], dtype=int)
    first = wp.full(2, 2147483647, dtype=int, device="cpu")
    gaps = wp.full((2, 2, 2), float("inf"), dtype=float, device="cpu")
    wp.launch(
        batch_tracking.contact_guard,
        2,
        inputs=[
            count,
            pairs,
            worlds,
            distances,
            array(limits),
            failure,
            step,
            first,
            gaps,
        ],
        device="cpu",
    )
    np.testing.assert_array_equal(failure.numpy(), [0, 4])
    np.testing.assert_array_equal(first.numpy(), [2147483647, 7])
    assert not np.isfinite(gaps.numpy()[0]).any()
    np.testing.assert_allclose(gaps.numpy()[1, 0, 1], -0.0007)


def test_each_world_uses_its_own_reference_and_retains_grip_on_failure():
    refs = np.zeros((1, 2, 7))
    refs[0, :, 0] = [0.001, -0.002]
    refs[0, :, 6] = [0.3, 0.7]
    q = np.zeros((2, 28))
    q[:, [17, 24]] = 1
    scenes = q.copy()
    q[1, 21] = 0.01
    failure = wp.zeros(2, dtype=int, device="cpu")
    action = wp.zeros((2, 7), dtype=float, device="cpu")
    wp.launch(
        batch_tracking.follow_reference,
        2,
        inputs=[
            array([True, True], dtype=bool),
            array(refs),
            array([0], dtype=int),
            array(q),
            array(np.zeros((2, 26))),
            array(np.zeros((2, 6))),
            array(scenes),
            array(np.zeros((1, 2, 6))),
            array([0, 0], dtype=int),
            array([1, 1], dtype=int),
            failure,
            action,
            array([0, 0], dtype=int),
            array(np.zeros((1, 2, 7))),
        ],
        device="cpu",
    )
    np.testing.assert_array_equal(failure.numpy(), [0, 2])
    np.testing.assert_allclose(action.numpy()[:, 6], [0.3, 0.7])
    np.testing.assert_allclose(action.numpy()[0, 0], 0.001 / (0.02 * np.pi))
    np.testing.assert_array_equal(action.numpy()[1, :6], 0)


def test_grip_force_has_a_world_axis_and_preserves_arm_limits():
    initial = np.tile([-150.0, 150.0], (2, 7, 1))
    limits = array(initial, dtype=wp.vec2)
    wp.launch(
        batch_tracking.apply_grip_effort,
        2,
        inputs=[array([1.0, 4.0]), limits],
        device="cpu",
    )
    np.testing.assert_array_equal(limits.numpy()[:, 6], [[-1, 1], [-4, 4]])
    np.testing.assert_array_equal(limits.numpy()[:, :6], initial[:, :6])


def test_attachment_failures_are_scoped_to_the_selected_world():
    q = np.zeros((2, 21))
    q[:, 17] = 1
    q[1, 14] = 0.01
    failure = wp.zeros(2, dtype=int, device="cpu")
    rotations = np.tile(np.eye(3), (2, 1, 1, 1))
    wp.launch(
        batch_tracking.attachment_guard,
        2,
        inputs=[
            array(q),
            array(np.zeros((2, 1, 3)), dtype=wp.vec3),
            array(rotations, dtype=wp.mat33),
            array([0, 0], dtype=int),
            0,
            array(np.zeros((2, 3)), dtype=wp.vec3),
            array(np.tile(np.eye(3), (2, 1, 1)), dtype=wp.mat33),
            array([1, 1], dtype=int),
            failure,
        ],
        device="cpu",
    )
    np.testing.assert_array_equal(failure.numpy(), [0, 3])


def test_shorter_paths_pad_with_stationary_commands_and_gravity_bias():

    model, qpos = simulation_models.make_model([[0.02] * 3], [[0.425, 0, 0.01]])
    scene = collision.Scene(model, qpos)
    short_path = np.array([qpos, qpos])
    long_path = short_path.copy()
    long_path[1, 0] += 0.12
    requests = [
        batch_tracking.Request(
            manipulation.Segment("hold", short_path, np.zeros(2)), scene, -1
        ),
        batch_tracking.Request(
            manipulation.Segment("transit", long_path, np.zeros(2)), scene, -1
        ),
    ]
    prepared = batch_tracking.prepare(requests, np.array([qpos, qpos]))
    short_count, long_count = prepared.lengths
    assert short_count < long_count
    np.testing.assert_array_equal(
        prepared.references[short_count:, 0],
        np.tile(prepared.references[short_count - 1, 0], (long_count - short_count, 1)),
    )
    original = tracking.references(requests[0].segment)
    expected_bias = tracking.servo_offsets(model, original, qpos)[-1]
    np.testing.assert_allclose(
        prepared.bias[short_count:, 0],
        np.tile(expected_bias, (long_count - short_count, 1)),
    )
    assert prepared.references.shape == (long_count, 2, 7)


def test_invalid_force_and_mismatched_world_count_are_rejected_before_execution():
    import pytest


    model, qpos = simulation_models.make_model([[0.02] * 3], [[0.425, 0, 0.01]])
    scene = collision.Scene(model, qpos)
    segment = manipulation.Segment("hold", np.array([qpos, qpos]), np.zeros(2))
    request = batch_tracking.Request(segment, scene, -1, grip_effort=0)
    with pytest.raises(ValueError, match="effort"):
        batch_tracking.prepare([request], qpos[None])
    with pytest.raises(ValueError, match="world"):
        batch_tracking.prepare([request], np.array([qpos, qpos]))


def test_replay_force_uses_both_time_and_world_without_changing_arm_limits():
    initial = np.tile([-150.0, 150.0], (2, 7, 1))
    limits = array(initial, dtype=wp.vec2)
    steps = array([1], dtype=int)
    efforts = array([[1.0, 4.0], [2.0, 3.0]])
    wp.launch(
        batch_tracking.replay_grip_effort,
        2,
        inputs=[array([True, True], dtype=bool), efforts, steps, limits],
        device="cpu",
    )
    np.testing.assert_array_equal(limits.numpy()[:, 6], [[-2, 2], [-3, 3]])
    np.testing.assert_array_equal(limits.numpy()[:, :6], initial[:, :6])


def test_mixed_world_references_use_each_worlds_compiled_servos():

    models, homes, requests = [], [], []
    for gain_scale in (1.0, 2.0):
        model, home = simulation_models.make_model([[0.02] * 3], [[0.425, 0, 0.01]])
        model.actuator_gainprm[:6, 0] *= gain_scale
        scene = collision.Scene(model, home)
        segment = manipulation.Segment("hold", np.array([home, home]), np.zeros(2))
        models.append(model)
        homes.append(home)
        requests.append(batch_tracking.Request(segment, scene, -1))
    prepared = batch_tracking.prepare(requests, np.array(homes))
    for world, request in enumerate(requests):
        expected = tracking.servo_offsets(
            models[world], tracking.references(request.segment), homes[world]
        )
        np.testing.assert_allclose(prepared.bias[:, world], expected)
    np.testing.assert_allclose(prepared.bias[:, 1], prepared.bias[:, 0] / 2)
