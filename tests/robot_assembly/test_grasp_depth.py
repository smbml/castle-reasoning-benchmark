"""Insertion depth follows the actual aperture-dependent fingertip reach."""

import mujoco
import numpy as np
import pytest

import castle_reasoning.runtime.planning.grasps as grasps
import castle_reasoning.runtime.simulation.models as models


@pytest.mark.parametrize("width", [0.02, 0.04, 0.06])
@pytest.mark.parametrize("family", ["top_down", "side"])
def test_measured_pads_reach_the_requested_insertion_depth(width, family):
    size = np.array(
        [width, 0.08, 0.01] if family == "top_down" else [0.08, 0.08, width]
    )
    model, qpos = models.make_model([size], [[0.425, 0, 0.2]])
    aperture = grasps.Aperture(model)
    proposal = grasps.propose(size, family=family)[0]
    calibrated = aperture.calibrate(proposal)
    state = mujoco.MjData(model)
    state.qpos[:] = qpos
    state.qpos[6:14] = aperture.joints(aperture.closure(width))
    mujoco.mj_kinematics(model, state)
    site = model.site("ur5e/robotiq/pinch").id
    wrist = state.site_xmat[site].reshape(3, 3)
    position, rotation = calibrated.pose(np.array([0, 0, 0, 1, 0, 0, 0]))
    direction = rotation[:, 2]
    axis = int(np.argmax(np.abs(direction)))
    expected_depth = (
        min(0.02, 0.6 * size[2])
        if family == "top_down"
        else min(0.012, 0.6 * size[axis])
    )
    for side in ("left", "right"):
        geom = model.geom(f"ur5e/robotiq/{side}_pad1").id
        corners = (
            np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
            * model.geom_size[geom]
        )
        world = corners @ state.geom_xmat[geom].reshape(3, 3).T + state.geom_xpos[geom]
        local = (world - state.site_xpos[site]) @ wrist
        placed = local @ rotation.T + position
        actual_depth = (placed @ direction).max() + size[axis] / 2
        assert actual_depth == pytest.approx(expected_depth, abs=2e-6)
    assert calibrated.width == proposal.width
    assert calibrated.roll == proposal.roll
    assert calibrated.yaw == proposal.yaw
    displacement = np.array(calibrated.offset) - proposal.offset
    np.testing.assert_allclose(
        displacement - direction * (direction @ displacement), 0, atol=1e-12
    )
