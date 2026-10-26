"""Recovery interrupts lost-object feedback and retries from settled state."""

import numpy as np
import warp as wp

import castle_reasoning.data.assembly.teacher_control as teacher_control


def fixture():
    guard = teacher_control.make_guard(1, 1, 2, device="cpu")
    guard.phase.fill_(3)
    qpos = np.zeros((1, 21), np.float32)
    qpos[0, 14:] = [0.4, 0, 0.02, 1, 0, 0, 0]
    q = wp.array(qpos, dtype=float, device="cpu")
    v = wp.zeros((1, 20), dtype=float, device="cpu")
    pos = wp.array(
        np.array([[[0.4, 0, 0.05]]], np.float32), dtype=wp.vec3, device="cpu"
    )
    rot = wp.array(
        np.eye(3, dtype=np.float32).reshape(1, 1, 3, 3), dtype=wp.mat33, device="cpu"
    )
    action = wp.array(np.ones((1, 7), np.float32) * 0.5, dtype=float, device="cpu")
    enabled = wp.array([1], dtype=int, device="cpu")

    def tick():
        wp.launch(
            teacher_control.guard_action,
            1,
            inputs=[guard, q, v, pos, rot, 0, action, enabled, 0.01, 20.0, 1, 2],
            device="cpu",
        )

    return guard, q, v, action, tick


def test_normal_grasp_is_unchanged_then_drop_interrupts_immediately():
    guard, q, _v, action, tick = fixture()
    tick()
    np.testing.assert_allclose(action.numpy(), 0.5)
    values = q.numpy()
    values[0, 14] += 0.08
    wp.copy(q, wp.array(values, dtype=float, device="cpu"))
    guard.phase.fill_(4)
    tick()
    np.testing.assert_array_equal(action.numpy()[0], [0, 0, 0, 0, 0, 0, -1])
    assert guard.phase.numpy()[0] == 8
    assert guard.retries.numpy()[0] == 1
    assert guard.interruptions.numpy()[0] == 1
    assert not guard.failed.numpy()[0]
    for _ in range(30):
        tick()
    assert guard.phase.numpy()[0] == 0
    assert guard.waiting.numpy()[0] == 0
    assert guard.elapsed.numpy()[0] == 0


def test_instability_interrupts_even_before_position_loss():
    guard, _q, v, _action, tick = fixture()
    tick()
    values = v.numpy()
    values[0, 19] = 50
    wp.copy(v, wp.array(values, dtype=float, device="cpu"))
    tick()
    assert guard.event.numpy()[0] == 2
    assert guard.waiting.numpy()[0] == 1


def test_exhausted_retry_and_tilted_part_fail_closed():
    guard, q, _v, action, tick = fixture()
    tick()
    guard.retries.fill_(1)
    values = q.numpy()
    values[0, 14] += 0.08
    wp.copy(q, wp.array(values, dtype=float, device="cpu"))
    tick()
    assert guard.failed.numpy()[0]
    np.testing.assert_array_equal(action.numpy()[0], [0, 0, 0, 0, 0, 0, -1])
    guard, q, _v, action, tick = fixture()
    tick()
    values = q.numpy()
    values[0, 14] += 0.08
    values[0, 17:] = [0.7071068, 0.7071068, 0, 0]
    wp.copy(q, wp.array(values, dtype=float, device="cpu"))
    tick()
    for _ in range(30):
        tick()
    assert guard.failed.numpy()[0]
