"""Friction selection reaches every saved physical model before GPU allocation."""

import mujoco
import numpy as np
import pytest

import castle_reasoning.data.assembly.continuous_collect as continuous_collect
import castle_reasoning.data.plans as data_plans
import castle_reasoning.runtime.simulation.models as simulation_models


class GpuAllocationBoundaryReached(Exception):
    """Stop after real model compilation and serialization, before GPU work."""


@pytest.mark.parametrize(
    "options, cone, solver, iterations",
    [
        ({}, 1, 2, 100),
        ({"friction_cone": "elliptic"}, 1, 2, 100),
        ({"friction_cone": "pyramidal"}, 0, 2, 100),
        ({"solver": "cg", "solver_iterations": 1000}, 1, 1, 1000),
        ({"solver": "newton", "solver_iterations": 300}, 1, 2, 300),
    ],
)
def test_physics_options_are_saved_for_every_world(
    tmp_path, monkeypatch, options, cone, solver, iterations
):
    plans = [
        {
            "sizes": [size],
            "target": [[0.425, 0, size[2] / 2]],
            "parking": [[0.6, 0, size[2] / 2]],
        }
        for size in ([0.02] * 3, [0.03, 0.04, 0.02])
    ]
    output = tmp_path / "run"

    def stop_before_allocation(models, indices):
        assert indices == [0, 1]
        for index, (model, plan) in enumerate(zip(models, plans, strict=True)):
            saved = mujoco.MjModel.from_binary_path(
                str(output / "models" / f"{index}.mjb")
            )
            assert int(model.opt.cone) == int(saved.opt.cone) == cone
            assert int(model.opt.solver) == int(saved.opt.solver) == solver
            assert int(model.opt.iterations) == int(saved.opt.iterations) == iterations
            original, _ = simulation_models.compile_model(
                plan, data_plans.control_config(0.002)
            )
            for field in ("geom_size", "geom_friction", "body_mass", "body_inertia"):
                np.testing.assert_array_equal(
                    getattr(saved, field), getattr(original, field)
                )
        raise GpuAllocationBoundaryReached

    monkeypatch.setattr(
        continuous_collect.world_slots, "ModelPool", stop_before_allocation
    )
    with pytest.raises(GpuAllocationBoundaryReached):
        continuous_collect.execute(plans, output, worlds=2, **options)


def test_bad_friction_cone_fails_before_output_creation(tmp_path):
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="friction cone"):
        continuous_collect.execute([{}], output, friction_cone="invalid")
    assert not output.exists()


@pytest.mark.parametrize(
    "options",
    [
        {"solver": "unknown"},
        {"solver_iterations": 0},
        {"solver_iterations": -1},
        {"solver_iterations": 2.5},
    ],
)
def test_bad_solver_options_fail_before_output_creation(tmp_path, options):
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="[Ss]olver"):
        continuous_collect.execute([{}], output, **options)
    assert not output.exists()
