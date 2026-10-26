"""Reactive controls preserve replay and stop stale servo targets."""

import numpy as np
import warp as wp

import castle_reasoning.runtime.control.gripper as gripper
import castle_reasoning.runtime.control.guards as guards


def test_width_dependent_hold_and_replay_bypass():
    action = np.zeros((3, 7), np.float32)
    action[:, 6] = [-1, 0, 1]
    act = wp.array(action, dtype=float, device="cpu")
    enabled = wp.array([1], dtype=int, device="cpu")
    move = wp.zeros(3, dtype=int, device="cpu")
    opening = wp.array(np.full((3, 1), 0.5, np.float32), dtype=float, device="cpu")
    holding = wp.array(np.full((3, 1), 0.8, np.float32), dtype=float, device="cpu")
    wp.launch(
        gripper.shape_hold,
        3,
        inputs=[enabled, move, opening, holding, act],
        device="cpu",
    )
    np.testing.assert_allclose(act.numpy()[:, 6], [0, 0.3, 0.6], atol=1e-6)
    saved = act.numpy().copy()
    enabled.zero_()
    wp.launch(
        gripper.shape_hold,
        3,
        inputs=[enabled, move, opening, holding, act],
        device="cpu",
    )
    np.testing.assert_array_equal(act.numpy(), saved)


def test_zero_action_cancels_target_error_and_motion_is_bounded():
    qpos = wp.zeros((2, 14), dtype=float, device="cpu")
    target = wp.array(np.full((2, 6), 2, np.float32), dtype=float, device="cpu")
    action = np.zeros((2, 7), np.float32)
    action[1, :6] = 0.3
    act = wp.array(action, dtype=float, device="cpu")
    stopped = wp.zeros(2, dtype=int, device="cpu")
    wp.launch(
        gripper.bound_targets,
        2,
        inputs=[qpos, act, target, 0.12, stopped],
        device="cpu",
    )
    np.testing.assert_allclose(target.numpy()[0], 0)
    np.testing.assert_allclose(target.numpy()[1], 0.12)


def test_motion_check_rejects_explosive_object_rotation():
    trajectory = {"qvel": np.zeros((3, 20)), "qpos": np.zeros((3, 21))}
    assert guards.motion_check(trajectory, 1)["passed"]
    trajectory["qvel"][1, 19] = 80
    assert not guards.motion_check(trajectory, 1)["passed"]


def test_stopped_target_holds_against_gravity_after_one_sync():
    qpos = wp.zeros((1, 14), dtype=float, device="cpu")
    target = wp.ones((1, 6), dtype=float, device="cpu")
    action = wp.zeros((1, 7), dtype=float, device="cpu")
    stopped = wp.zeros(1, dtype=int, device="cpu")
    wp.launch(
        gripper.bound_targets,
        1,
        inputs=[qpos, action, target, 0.12, stopped],
        device="cpu",
    )
    qpos.fill_(-0.005)
    wp.launch(
        gripper.bound_targets,
        1,
        inputs=[qpos, action, target, 0.12, stopped],
        device="cpu",
    )
    np.testing.assert_allclose(target.numpy(), 0)
