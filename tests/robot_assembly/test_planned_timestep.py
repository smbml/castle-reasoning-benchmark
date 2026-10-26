"""The resolved integration step changes independently of physical settings."""

import dataclasses

import pytest

import castle_reasoning.data.assembly.collect as collect
import castle_reasoning.data.plans as plans
import castle_reasoning.runtime.control.parameters as parameters


@pytest.mark.parametrize("timestep", [0.0005, 0.001, 0.002])
def test_control_period_and_physical_settings_survive_step_selection(timestep):
    config = plans.control_config(timestep)
    assert config.physics_timestep == timestep
    assert (
        dataclasses.replace(config, physics_timestep=0.0005)
        == parameters.ControlConfig()
    )
    assert round(0.02 / timestep) * timestep == pytest.approx(0.02)


def test_default_preserves_the_existing_integrator_resolution():
    assert plans.control_config() == parameters.ControlConfig()


@pytest.mark.parametrize(
    "timestep", [0, -0.001, 0.003, 0.02, float("nan"), float("inf")]
)
def test_unsupported_step_fails_before_creating_any_run_artifacts(tmp_path, timestep):
    output = tmp_path / "attempt"
    with pytest.raises(ValueError, match="physics timestep"):
        collect.execute({}, output, "invalid", physics_timestep=timestep)
    assert not output.exists()
