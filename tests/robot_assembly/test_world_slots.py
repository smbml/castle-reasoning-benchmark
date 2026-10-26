"""Reusable slots preserve compiled physical parameters and episode isolation."""

import os

import mujoco
import numpy as np
import pytest
import warp as wp

import castle_reasoning.data.assembly.teacher as teacher
import castle_reasoning.runtime.control.tracking as tracking
import castle_reasoning.runtime.simulation.models as simulation_models
import castle_reasoning.runtime.simulation.world_slots as world_slots


def box_model(size, density):
    return mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body pos="0 0 1"><freejoint/>'
        f'<geom type="box" size="{size}" density="{density}"/>'
        "</body></worldbody></mujoco>"
    )


def test_model_replacement_updates_derived_fields_without_changing_other_slots():
    models = [box_model(".01 .02 .03", 1000), box_model(".02 .03 .04", 1700)]
    pool = world_slots.ModelPool(models, [0, 0], device="cpu")
    expected = world_slots.ModelPool(models, [0, 1], device="cpu")
    names = [
        "geom_size",
        "geom_aabb",
        "body_mass",
        "body_inertia",
        "body_invweight0",
        "dof_invweight0",
        "body_subtreemass",
    ]
    original = {name: getattr(pool.model, name).numpy() for name in names}
    addresses = {name: getattr(pool.model, name).ptr for name in names}
    pool.select([1], [1])
    for name in names:
        actual = getattr(pool.model, name)
        assert actual.ptr == addresses[name]
        np.testing.assert_array_equal(actual.numpy()[0], original[name][0])
        np.testing.assert_array_equal(
            actual.numpy(), getattr(expected.model, name).numpy()
        )
    assert pool.native_models == models
    pool.select([1], [0])
    for name in names:
        np.testing.assert_array_equal(getattr(pool.model, name).numpy(), original[name])


def test_incompatible_model_bank_and_invalid_slot_are_rejected():
    models = [box_model(".01 .02 .03", 1000), box_model(".02 .03 .04", 1700)]
    models[1].geom_type[0] = mujoco.mjtGeom.mjGEOM_ELLIPSOID
    with pytest.raises(ValueError, match="geom_type"):
        world_slots.ModelPool(models, [0, 1], device="cpu")
    pool = world_slots.ModelPool(models[:1], [0], device="cpu")
    original = pool.model.body_mass.numpy()
    with pytest.raises(ValueError):
        pool.select([2], [0])
    with pytest.raises(ValueError):
        pool.select([0], [-1])
    np.testing.assert_array_equal(pool.model.body_mass.numpy(), original)


def test_batched_model_selection_restores_rows_and_preserves_addresses():
    models = [box_model(".01 .02 .03", 1000), box_model(".02 .03 .04", 1700)]
    pool = world_slots.ModelPool(models, [0, 0, 1], device="cpu")
    expected = world_slots.ModelPool(models, [1, 0, 1], device="cpu")
    addresses = [destination.ptr for destination, _ in pool.rows]
    # Selecting the same model must restore mutable parameters, not skip it.
    masses = pool.model.body_mass.numpy()
    masses[2] *= 3
    pool.model.body_mass.assign(masses)
    pool.select([2, 0], [1, 1])
    for (actual, _), (reference, _), address in zip(
        pool.rows, expected.rows, addresses, strict=True
    ):
        assert actual.ptr == address
        np.testing.assert_array_equal(actual.numpy(), reference.numpy())
    assert pool.assignments == [1, 0, 1]
    assert pool.native_models == [models[1], models[0], models[1]]
    pool.select([], [])
    assert pool.assignments == [1, 0, 1]


@pytest.mark.parametrize("worlds, indices", [
    ([0, 0], [0, 1]), ([0, 1], [0]), ([0, 2], [0, 0]),
    ([0, 1], [0, -1]), ([0, 1], [0, 2]), ([0.5], [0]),
    ([0], [0.5]), ([[0]], [[0]]), ([True], [0]),
])
def test_invalid_model_batch_does_not_partially_mutate(worlds, indices):
    models = [box_model(".01 .02 .03", 1000), box_model(".02 .03 .04", 1700)]
    pool = world_slots.ModelPool(models, [0, 1], device="cpu")
    original = [destination.numpy() for destination, _ in pool.rows]
    with pytest.raises(ValueError):
        pool.select(worlds, indices)
    assert pool.assignments == [0, 1]
    for (actual, _), reference in zip(pool.rows, original, strict=True):
        np.testing.assert_array_equal(actual.numpy(), reference)


@pytest.mark.skipif(
    os.environ.get("CONSTRUCTION_GPU_TESTS") != "1", reason="GPU opt-in"
)
def test_masked_reset_preserves_running_world_servo_physics_and_episode():
    goals = np.array([[0.425, 0, 0.01]])
    model, home = simulation_models.make_model([[0.02] * 3], goals)
    collector = teacher.make_collector(model, goals, 2, 1, 0)
    collector.controller = tracking.PathControl(collector)
    collector.reset(qpos=np.tile(home, (2, 1)), qvel=np.zeros((2, model.nv)))
    controller, engine = collector.controller, collector.engine
    targets = controller.arm_target.numpy()
    targets[0, 5] += 0.015  # Integrated servo target must survive sibling reset.
    controller.arm_target.assign(targets)
    actions = np.zeros((2, 7), np.float32)
    actions[:, 6] = -1
    controller.step(actions, graph=False)
    fields = ("qpos", "qvel", "ctrl", "qacc_warmstart")
    before = {name: getattr(engine.data, name).numpy() for name in fields}
    targets = controller.arm_target.numpy()
    episodes, random = engine.episode.numpy(), engine.random_u32.numpy()
    qpos, qvel = before["qpos"].copy(), before["qvel"].copy()
    qpos[1] = home
    qpos[1, 5] += 0.1
    qvel[1] = 0
    qvel[1, 5] = 0.025
    world_slots.reset(controller, np.array([False, True]), qpos=qpos, qvel=qvel)
    for name in fields:
        np.testing.assert_array_equal(
            getattr(engine.data, name).numpy()[0], before[name][0]
        )
    np.testing.assert_array_equal(controller.arm_target.numpy()[0], targets[0])
    np.testing.assert_array_equal(engine.random_u32.numpy()[0], random[0])
    np.testing.assert_array_equal(engine.episode.numpy(), episodes + [0, 1])
    np.testing.assert_array_equal(engine.data.qpos.numpy()[1], qpos[1])
    np.testing.assert_array_equal(engine.data.qvel.numpy()[1], qvel[1])
    np.testing.assert_array_equal(controller.arm_target.numpy()[1], qpos[1, :6])
    np.testing.assert_array_equal(controller.action.numpy()[1], [0, 0, 0, 0, 0, 0, -1])
    with wp.ScopedDevice(engine.device):
        controller.launch_device_control_step()
        wp.synchronize_device(engine.device)
    engine.assert_healthy()
