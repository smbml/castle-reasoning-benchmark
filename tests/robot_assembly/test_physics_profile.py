"""Recorded prefixes retain the original state/action/force alignment."""

import numpy as np
import pytest

import castle_reasoning.runtime.diagnostics.physics_profile as physics_profile


def test_solver_override_preserves_contact_and_integration_options():
    import mujoco

    model = mujoco.MjModel.from_xml_string("<mujoco/>")
    model.opt.timestep = 0.001
    model.opt.iterations = 100
    model.opt.tolerance = 1e-8
    model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    original = model.opt.solver
    physics_profile.configure_solver(model, "native", None)
    assert model.opt.solver == original
    physics_profile.configure_solver(model, "cg", 128)
    assert model.opt.solver == mujoco.mjtSolver.mjSOL_CG
    assert model.opt.timestep == 0.001
    assert model.opt.iterations == 100
    assert model.opt.tolerance == 1e-8
    assert model.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC


@pytest.mark.parametrize("solver,threads", [("unknown", 32), ("cg", 7)])
def test_invalid_solver_override_fails_before_gpu_allocation(solver, threads):
    import mujoco

    model = mujoco.MjModel.from_xml_string("<mujoco/>")
    with pytest.raises(ValueError):
        physics_profile.configure_solver(model, solver, threads)


def test_prefix_has_pre_action_state_and_original_force_schedule(tmp_path):
    path = tmp_path / "episode.npz"
    np.savez(
        path,
        qpos=np.arange(30).reshape(6, 5),
        qvel=np.arange(24).reshape(6, 4),
        action=np.arange(35).reshape(5, 7),
        grip_effort=[1, 1, 2, 2, 1],
    )
    sample = physics_profile.load_prefix(path, start=1, count=3)
    np.testing.assert_array_equal(sample["qpos"], np.arange(5, 25).reshape(4, 5))
    np.testing.assert_array_equal(sample["qvel"], np.arange(4, 20).reshape(4, 4))
    np.testing.assert_array_equal(sample["action"], np.arange(7, 28).reshape(3, 7))
    np.testing.assert_array_equal(sample["grip_effort"], [1, 2, 2])


@pytest.mark.parametrize("start,count", [(-1, 2), (0, 0), (4, 2)])
def test_out_of_range_prefix_fails_before_allocating_simulator(tmp_path, start, count):
    path = tmp_path / "episode.npz"
    np.savez(
        path,
        qpos=np.zeros((6, 5)),
        qvel=np.zeros((6, 4)),
        action=np.zeros((5, 7)),
        grip_effort=np.ones(5),
    )
    with pytest.raises(ValueError, match="prefix"):
        physics_profile.load_prefix(path, start=start, count=count)
