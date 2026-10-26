"""Distinct physical scenes share layout, not dimensions or inertial values."""

import copy

import mujoco
import mujoco_warp as mjw
import numpy as np
import pytest
import warp as wp

import castle_reasoning.runtime.simulation.model_batch as model_batch
import castle_reasoning.runtime.simulation.models as simulation_models


def box_model(size, density=1000, gravcomp=0, count=1, sleep=False):
    bodies = "".join(
        f'<body pos="{i} 0 1" gravcomp="{gravcomp}"><freejoint/>'
        f'<geom type="box" size="{size}" density="{density}"/></body>'
        for i in range(count)
    )
    flag = '<flag sleep="enable"/>' if sleep else ""
    return mujoco.MjModel.from_xml_string(
        f"<mujoco><option>{flag}</option><worldbody>{bodies}</worldbody></mujoco>"
    )


def test_compiled_fields_include_geometry_inertia_and_derived_weights():
    models = [box_model(".01 .02 .03"), box_model(".02 .03 .04", 1700, 1)]
    with wp.ScopedDevice("cpu"):
        expected = [mjw.put_model(m) for m in models]
        mixed = model_batch.stack_models(models, device="cpu")
    for name in (
        "geom_size",
        "geom_aabb",
        "geom_rbound",
        "body_mass",
        "body_inertia",
        "body_subtreemass",
        "body_invweight0",
        "dof_invweight0",
        "body_gravcomp",
    ):
        actual = getattr(mixed, name).numpy()
        assert actual.shape[0] == 2
        np.testing.assert_array_equal(
            actual, np.concatenate([getattr(m, name).numpy() for m in expected])
        )
    np.testing.assert_array_equal(
        mixed.stat.meaninertia.numpy(),
        np.concatenate([m.stat.meaninertia.numpy() for m in expected]),
    )


def test_different_body_counts_do_not_silently_share_a_model():
    with pytest.raises(ValueError, match="nq|nbody|nv"):
        model_batch.stack_models(
            [box_model(".01 .02 .03"), box_model(".01 .02 .03", count=2)],
            device="cpu",
        )


def test_equal_counts_with_different_collision_types_are_rejected():
    models = [box_model(".01 .02 .03"), box_model(".01 .02 .03")]
    models[1].geom_type[0] = mujoco.mjtGeom.mjGEOM_ELLIPSOID
    with pytest.raises(ValueError, match="geom_type"):
        model_batch.stack_models(models, device="cpu")


def test_unbatched_sleep_lengths_are_not_silently_reused():
    with pytest.raises(ValueError, match="dof_length"):
        model_batch.stack_models(
            [
                box_model(".01 .02 .03", sleep=True),
                box_model(".02 .03 .04", sleep=True),
            ],
            device="cpu",
        )


def test_padding_preserves_the_actual_task_and_uses_separate_inactive_slots():
    plan = {
        "sizes": [[0.02, 0.03, 0.04]],
        "target": [[0.4, 0, 0.02]],
        "parking": [[0.4, -0.2, 0.02]],
        "removal_order": [0],
        "supports": [],
    }
    original = copy.deepcopy(plan)
    padded = model_batch.pad_plan(plan, 3)
    assert plan == original
    assert padded["active_part_count"] == 1
    assert padded["removal_order"] == [0]
    for key in ("sizes", "target", "parking"):
        assert padded[key][0] == plan[key][0]
        assert len(padded[key]) == 3
    assert padded["target"][1:] == padded["parking"][1:]
    assert len({tuple(x) for x in padded["target"][1:]}) == 2
    assert all(x[0] > 1 and x[2] > 0 for x in padded["target"][1:])
    with pytest.raises(ValueError, match="capacity"):
        model_batch.pad_plan(plan, 0)


def test_inactive_bodies_stay_still_without_changing_active_free_fall():
    sizes = [[0.02] * 3, [0.01] * 3]
    positions = [[0.4, 0, 0.5], [2, 0, 0.5]]
    padded, state = simulation_models.make_model(sizes, positions, active_part_count=1)
    baseline, base_state = simulation_models.make_model(sizes[:1], positions[:1])
    actual, expected = mujoco.MjData(padded), mujoco.MjData(baseline)
    actual.qpos[:], expected.qpos[:] = state, base_state
    actual.ctrl[:6], expected.ctrl[:6] = state[:6], base_state[:6]
    mujoco.mj_step(padded, actual, nstep=50)
    mujoco.mj_step(baseline, expected, nstep=50)
    np.testing.assert_allclose(actual.qpos[14:21], expected.qpos[14:21], atol=1e-10)
    np.testing.assert_allclose(actual.qpos[21:28], state[21:28], atol=1e-10)
    np.testing.assert_allclose(actual.qvel[20:26], 0, atol=1e-10)


def test_gpu_worlds_obey_their_own_mass_inertia_and_gravity_compensation():
    if not wp.is_cuda_available():
        pytest.skip("Requires a CUDA device for the batched dynamics check.")
    models = [box_model(".01 .02 .03"), box_model(".02 .03 .04", 1700, 1)]
    with wp.ScopedDevice("cuda:0"):
        mixed = model_batch.stack_models(models)
        state = mjw.make_data(models[0], nworld=2)
        state.qpos.assign(np.array([m.qpos0 for m in models], np.float32))
        force = np.zeros((2, 6), np.float32)
        force[:, 0], force[:, 5] = 0.002, 0.000001
        state.qfrc_applied.assign(force)
        with wp.ScopedCapture() as capture:
            mjw.step(mixed, state)
        for _ in range(100):
            wp.capture_launch(capture.graph)
        wp.synchronize()
        assert not state.overflow.numpy().any()
        velocity = state.qvel.numpy()
    elapsed = models[0].opt.timestep * 100
    for world, model in enumerate(models):
        expected = [
            0.002 / model.body_mass[1] * elapsed,
            -9.81 * (1 - model.body_gravcomp[1]) * elapsed,
            0.000001 / model.body_inertia[1, 2] * elapsed,
        ]
        np.testing.assert_allclose(
            velocity[world, [0, 2, 5]], expected, rtol=1e-5, atol=1e-6
        )
