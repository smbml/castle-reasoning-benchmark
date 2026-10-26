"""Device path tracking, scene guards, and retained grip on interruption."""

import mujoco
import numpy as np
import warp as wp

import castle_reasoning.runtime.control.guards as guards
import castle_reasoning.runtime.control.tracking as tracking
import castle_reasoning.runtime.planning.collision as collision
import castle_reasoning.runtime.planning.manipulation as manipulation
import castle_reasoning.runtime.simulation.models as models


def test_reference_actions_and_neighbor_guard_preserve_holding_aperture():
    ref = np.zeros((2, 7), np.float32)
    ref[:, :6] = 0.001
    ref[:, 6] = 0.4
    q = np.zeros((1, 28), np.float32)
    q[0, [17, 24]] = 1
    source = q[0].copy()
    q[0, 21] = 0.01  # Nonselected block moved by one centimeter.
    device = "cpu"
    args = [
        wp.array(ref, dtype=float, device=device),
        wp.zeros(1, dtype=int, device=device),
        wp.array(q, dtype=float, device=device),
        wp.zeros((1, 26), dtype=float, device=device),
        wp.zeros((1, 6), dtype=float, device=device),
        wp.array(source, dtype=float, device=device),
        wp.zeros((1, 6), dtype=float, device=device),
        0,
        1,
        wp.zeros(1, dtype=int, device=device),
        wp.zeros((1, 7), dtype=float, device=device),
    ]
    wp.launch(tracking.follow_reference, 1, inputs=args, device=device)
    assert args[-2].numpy()[0] == 2
    np.testing.assert_array_equal(args[-1].numpy()[0, :6], 0)
    np.testing.assert_allclose(args[-1].numpy()[0, 6], 0.4)


def test_references_reach_each_joint_waypoint_and_bound_velocity():
    path = np.zeros((3, 21))
    path[1, 0], path[2, 0], path[2, 1] = 0.1, 0.1, 0.2

    segment = manipulation.Segment("transit", path, np.array([0, 0.4, 0.4]))
    refs = tracking.references(segment)
    np.testing.assert_allclose(refs[-1, :6], path[-1, :6])
    assert np.max(np.abs(np.diff(refs[:, :6], axis=0))) / 0.02 <= 0.301
    assert np.any(np.linalg.norm(refs[:, :6] - path[1, :6], axis=1) < 1e-7)


def test_servo_bias_preserves_support_at_a_replanned_phase_boundary():
    # A constant load causes measured joints to sit 0.02 below the servo target.
    # A path beginning at that measured pose must preserve the holding target.
    ref = wp.zeros((1, 7), dtype=float, device="cpu")
    step = wp.zeros(1, dtype=int, device="cpu")
    q = wp.zeros((1, 14), dtype=float, device="cpu")
    v = wp.zeros((1, 14), dtype=float, device="cpu")
    target = wp.array(np.full((1, 6), 0.02, np.float32), dtype=float, device="cpu")
    scene = wp.zeros(14, dtype=float, device="cpu")
    bias = wp.array(np.full((1, 6), 0.02, np.float32), dtype=float, device="cpu")
    failure = wp.zeros(1, dtype=int, device="cpu")
    action = wp.zeros((1, 7), dtype=float, device="cpu")
    wp.launch(
        tracking.follow_reference,
        1,
        inputs=[ref, step, q, v, target, scene, bias, -1, 0, failure, action],
        device="cpu",
    )
    np.testing.assert_allclose(action.numpy()[0, :6], 0, atol=1e-7)
    wp.launch(tracking.clamp_targets, (1, 6), inputs=[q, target], device="cpu")
    np.testing.assert_allclose(target.numpy(), 0.02, atol=1e-7)


def test_contact_guard_distinguishes_forbidden_neighbor_and_allowed_pad():
    count = wp.array([2], dtype=int, device="cpu")
    pairs = wp.array(np.array([[0, 1], [0, 2]], np.int32), dtype=wp.vec2i, device="cpu")
    worlds = wp.zeros(2, dtype=int, device="cpu")
    distances = wp.array([-0.0001, -0.0007], dtype=float, device="cpu")
    limits = np.full((3, 3), -0.0002, np.float32)
    limits[0, 2] = -0.001
    threshold = wp.array(limits, dtype=float, device="cpu")
    fail = wp.zeros(1, dtype=int, device="cpu")
    step = wp.array([5], dtype=int, device="cpu")
    first = wp.full(1, 2147483647, dtype=int, device="cpu")
    gaps = wp.full((1, 3, 3), float("inf"), dtype=float, device="cpu")
    inputs = [count, pairs, worlds, distances, threshold, fail, step, first, gaps]
    wp.launch(guards.contact_guard, 2, inputs=inputs, device="cpu")
    assert fail.numpy()[0] == 0
    assert not np.isfinite(gaps.numpy()).any()
    wp.copy(distances, wp.array([-0.0005, -0.0007], dtype=float, device="cpu"))
    wp.launch(guards.contact_guard, 2, inputs=inputs, device="cpu")
    assert fail.numpy()[0] == 4
    assert first.numpy()[0] == 5
    np.testing.assert_allclose(gaps.numpy()[0, 0, 1], -0.0005)
    assert np.isinf(gaps.numpy()[0, 0, 2])


def test_contact_evidence_keeps_each_worlds_first_violating_control_index():
    device = "cpu"
    pairs = wp.array(
        [[0, 1], [0, 1], [1, 2], [0, 1], [-1, -1]], dtype=wp.vec2i, device=device
    )
    worlds = wp.array([0, 0, 0, 1, 1], dtype=int, device=device)
    distances = wp.array([-0.0005, -0.0007, -0.0006, 0, -1], dtype=float, device=device)
    limits = wp.full((3, 3), -0.0002, dtype=float, device=device)
    failure = wp.zeros(2, dtype=int, device=device)
    step = wp.array([3], dtype=int, device=device)
    first = wp.full(2, 2147483647, dtype=int, device=device)
    gaps = wp.full((2, 3, 3), float("inf"), dtype=float, device=device)
    inputs = [
        wp.array([5], dtype=int, device=device),
        pairs,
        worlds,
        distances,
        limits,
        failure,
        step,
        first,
        gaps,
    ]
    wp.launch(guards.contact_guard, 5, inputs=inputs, device=device)
    np.testing.assert_array_equal(first.numpy(), [3, 2147483647])
    np.testing.assert_array_equal(failure.numpy(), [4, 0])
    np.testing.assert_allclose(gaps.numpy()[0, 0, 1], -0.0007)
    np.testing.assert_allclose(gaps.numpy()[0, 1, 2], -0.0006)
    assert not np.isfinite(gaps.numpy()[1]).any()

    step.fill_(4)
    distances.fill_(-0.001)
    wp.launch(guards.contact_guard, 5, inputs=inputs, device=device)
    np.testing.assert_array_equal(first.numpy(), [3, 4])
    np.testing.assert_allclose(gaps.numpy()[0, 0, 1], -0.0007)
    np.testing.assert_allclose(gaps.numpy()[0, 1, 2], -0.0006)
    np.testing.assert_allclose(gaps.numpy()[1, 0, 1], -0.001)


def test_contact_evidence_identifies_geometry_and_exact_admission_threshold():
    import json


    model, _ = models.make_model([[0.02] * 3], [[0.425, 0, 0.01]])
    floor, obj = model.geom("floor").id, model.geom("object_0").id
    allowed = np.full((model.ngeom, model.ngeom), -0.0002)
    allowed[obj, floor] = -0.0003
    gaps = np.full((2, model.ngeom, model.ngeom), np.inf)
    assert (
        guards.contact_evidence(model, [2147483647] * 2, gaps, allowed) == []
    )
    gaps[1, obj, floor] = -0.000325
    evidence = guards.contact_evidence(model, [2147483647, 12], gaps, allowed)
    assert json.loads(json.dumps(evidence)) == [
        {
            "world": 1,
            "control_index": 12,
            "geom1": "object_0",
            "geom2": "floor",
            "distance_m": -0.000325,
            "allowed_distance_m": -0.0003,
        }
    ]


def test_feedforward_holds_native_robot_against_gravity():

    model, qpos = models.make_model([[0.02] * 3], [[0.425, 0, 0.01]])
    refs = np.tile(np.r_[qpos[:6], -1.0], (51, 1))
    offsets = tracking.servo_offsets(model, refs, qpos)
    state = mujoco.MjData(model)
    state.qpos[:] = qpos
    state.ctrl[:6] = qpos[:6] + offsets[0]
    state.ctrl[6] = 0
    for _ in range(250):
        mujoco.mj_step(model, state)
    assert np.max(np.abs(state.qpos[:6] - qpos[:6])) < 0.0002


def test_push_contacts_audit_the_moving_object_without_a_rigid_attachment():
    import pytest


    model, qpos = models.make_model(
        [[0.18, 0.2, 0.01], [0.02] * 3], [[0.4, -0.1, 0.005], [0.65, 0.25, 0.01]]
    )
    scene = collision.Scene(model, qpos)
    floor = model.geom("floor").id
    obj, other = model.geom("object_0").id, model.geom("object_1").id
    segment = manipulation.Segment(
        "push", np.array([qpos, qpos]), np.zeros(2), True, (floor,), moving_object=True
    )
    limits = guards.contact_limits(scene, segment, 0)
    assert segment.payload is None
    assert limits[obj, floor] == pytest.approx(-0.0003)
    assert limits[obj, other] == pytest.approx(-0.0002)


def test_failure_trace_keeps_only_the_contiguous_finite_prefix():
    trace = {
        name: np.arange(8, dtype=float).reshape(4, 1, 2)
        for name in ("qpos", "qvel", "action")
    }
    for name, values in tracking.finite_prefix(trace).items():
        np.testing.assert_array_equal(values, trace[name])
    for field in trace:
        for index in (0, 2, 3):
            for invalid in (np.nan, np.inf):
                broken = {name: value.copy() for name, value in trace.items()}
                broken[field][index, 0, 0] = invalid
                result = tracking.finite_prefix(broken)
                for name in trace:
                    np.testing.assert_array_equal(result[name], trace[name][:index])
