import numpy as np
import pytest

import castle_reasoning.runtime.simulation.robot as robot
import tests.helpers.qualification_columns as qualification_columns


def test_sampling_reproducible_distinct_supported_and_reversible():
    candidates = [qualification_columns.sample_target(19, i, block_count=8) for i in range(30)]
    assert len({c.fingerprint for c in candidates}) == 30
    for i, c in enumerate(candidates):
        np.testing.assert_array_equal(
            c.positions, qualification_columns.sample_target(19, i, 8).positions
        )
        removed = set()
        for block in c.removal_order:
            assert all(
                upper in removed for upper, lower in c.supports if lower == block
            )
            removed.add(block)
        assert sorted(c.removal_order) == list(range(8))
        assert sorted(c.assembly_order) == list(range(8))
        assert c.assembly_order == c.removal_order[::-1]
        assert np.max(c.positions[:, 2]) >= 0.06


def test_verifier_requires_whole_hold_home_and_stable_objects():
    target = qualification_columns.sample_target(19, 0, 4)
    qpos = np.zeros((51, 14 + 7 * 4))
    qpos[:, :6] = robot.HOME
    for i, xyz in enumerate(target.positions):
        qpos[:, 14 + 7 * i : 17 + 7 * i] = xyz
        qpos[:, 17 + 7 * i] = 1
    qvel = np.zeros((51, 14 + 6 * 4))
    assert qualification_columns.verify_hold(qpos, qvel, target.positions)["passed"]
    qpos[0, 14] += 0.008
    assert not qualification_columns.verify_hold(qpos, qvel, target.positions)["passed"]
    qpos[0, 14] -= 0.008
    qpos[-1, 0] += 0.2
    assert not qualification_columns.verify_hold(qpos, qvel, target.positions)["passed"]
    qpos[-1, 0] -= 0.2
    qvel[-1, 14] = 0.15
    assert not qualification_columns.verify_hold(qpos, qvel, target.positions)["passed"]


def test_nonfinite_short_and_wrong_shape_fail_closed():
    target = qualification_columns.sample_target(19, 0, 4)
    with pytest.raises(ValueError):
        qualification_columns.verify_hold(np.zeros((3, 42)), np.zeros((3, 38)), target.positions)
    assert qualification_columns.video_indices(20) == list(range(1, 20, 2))


def test_cube_yaw_must_match_target_modulo_quarter_turns():
    target = qualification_columns.sample_target(19, 0, 2)
    qpos = np.zeros((51, 28))
    qpos[:, :6] = robot.HOME
    qpos[:, 14:].reshape(51, 2, 7)[:, :, :3] = target.positions
    qpos[:, 17::7] = 1
    qvel = np.zeros((51, 26))
    angle = np.deg2rad(30.0)
    qpos[:, 17] = np.cos(angle / 2)
    qpos[:, 20] = np.sin(angle / 2)
    assert not qualification_columns.verify_hold(qpos, qvel, target.positions)["passed"]
    qpos[:, 17] = np.cos(np.pi / 4)
    qpos[:, 20] = np.sin(np.pi / 4)
    assert qualification_columns.verify_hold(qpos, qvel, target.positions)["passed"]
