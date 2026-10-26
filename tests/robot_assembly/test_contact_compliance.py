"""Contact compliance must support heavy beams without altering the robot."""

import mujoco
import numpy as np
import pytest

import castle_reasoning.data.plans as plans
import castle_reasoning.runtime.simulation.models as models

SIZES = [[0.01, 0.01, 0.01], [0.01, 0.01, 0.01], [0.22, 0.02, 0.03]]
POSITIONS = [[0.425, 0, 0.005], [0.425, 0, 0.015], [0.425, 0, 0.035]]


def test_contact_setting_preserves_robot_mass_and_other_contact_parameters():
    original, home = models.make_model(SIZES, POSITIONS)
    model, changed_home = models.make_model(
        SIZES, POSITIONS, contact_impedance=(0.95, 0.99)
    )
    np.testing.assert_array_equal(home, changed_home)
    for field in (
        "body_mass",
        "body_inertia",
        "geom_solref",
        "geom_friction",
        "actuator_gainprm",
        "actuator_biasprm",
    ):
        np.testing.assert_array_equal(getattr(model, field), getattr(original, field))
    for geom in range(model.ngeom):
        name = model.geom(geom).name
        expected = original.geom_solimp[geom].copy()
        if name == "floor" or name.startswith("object_"):
            expected[:2] = [0.95, 0.99]
        np.testing.assert_array_equal(model.geom_solimp[geom], expected)


def test_heavy_beam_resting_on_small_support_meets_existing_overlap_limit():
    # Match the construction pipeline's finite table and its contact time
    # constant; the original OGBench plane has a different floor compliance.
    plan = {
        "sizes": SIZES,
        "target": POSITIONS,
        "edge_access": {
            "table_bounds": [-0.15, 0.85, -0.25, 0.4],
            "table_contact_seconds": 0.004,
        },
    }
    overlaps = []
    for impedance in (None, (0.95, 0.99)):
        model, home = models.compile_model(
            plan, plans.control_config(0.002), contact_impedance=impedance
        )
        state = mujoco.MjData(model)
        state.qpos[:] = home
        state.ctrl[:6] = home[:6]
        mujoco.mj_step(model, state, nstep=2500)
        mujoco.mj_forward(model, state)
        gap = mujoco.mj_geomDistance(
            model,
            state,
            model.geom("object_1").id,
            model.geom("object_2").id,
            0.01,
            None,
        )
        overlaps.append(-gap)
        np.testing.assert_allclose(
            state.qpos[14:].reshape(-1, 7)[:, :3], POSITIONS, atol=0.002
        )
    assert overlaps[0] > 0.0003
    assert 0 < overlaps[1] < 0.0003


def test_collection_compiler_propagates_contact_setting():
    plan = {"sizes": SIZES, "target": POSITIONS}
    model, _ = models.compile_model(
        plan, plans.control_config(0.002), contact_impedance=(0.95, 0.99)
    )
    np.testing.assert_array_equal(model.geom("object_2").solimp[:2], [0.95, 0.99])


@pytest.mark.parametrize("values", [(0.95,), (0.99, 0.95), (0.9, 1), (np.nan, 0.99)])
def test_invalid_impedance_is_rejected(values):
    with pytest.raises(ValueError, match="impedance"):
        models.make_model(SIZES, POSITIONS, contact_impedance=values)
