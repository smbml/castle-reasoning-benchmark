"""Side pinches must approach and withdraw along an accessible plate edge."""

import mujoco
import numpy as np

import castle_reasoning.runtime.planning.grasps as grasps
import castle_reasoning.runtime.planning.manipulation as manipulation
import castle_reasoning.runtime.simulation.models as models


def test_side_proposals_close_across_thickness_from_each_edge():
    size = np.array([0.18, 0.20, 0.01])
    assert not grasps.propose(size)
    candidates = grasps.propose(size, family="side")
    assert len(candidates) >= 4
    directions = set()
    for candidate in candidates:
        assert candidate.width == 0.01
        position, rotation = candidate.pose(np.array([0, 0, 0, 1, 0, 0, 0]))
        np.testing.assert_allclose(np.abs(rotation[:, 1]), [0, 0, 1], atol=1e-12)
        directions.add(tuple(np.rint(rotation[:, 2]).astype(int)))
        assert position @ rotation[:, 2] < 0
    assert directions == {(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)}


def test_open_side_pads_cannot_fit_under_a_flush_plate():
    model, qpos = models.make_model([[0.18, 0.20, 0.01]], [[0.425, 0, 0.005]])
    aperture = grasps.Aperture(model)
    state = mujoco.MjData(model)
    state.qpos[:] = qpos
    state.qpos[6:14] = aperture.joints(aperture.closure(0.022))
    mujoco.mj_kinematics(model, state)
    site = model.site("ur5e/robotiq/pinch").id
    frame = state.site_xmat[site].reshape(3, 3)
    candidate = grasps.propose(np.array([0.18, 0.20, 0.01]), family="side")[0]
    position, rotation = candidate.pose(qpos[14:21])
    minimum = float("inf")
    for side in ("left", "right"):
        geom = model.geom(f"ur5e/robotiq/{side}_pad1").id
        corners = np.array(
            [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
        )
        world = (corners * model.geom_size[geom]) @ state.geom_xmat[geom].reshape(
            3, 3
        ).T + state.geom_xpos[geom]
        local = (world - state.site_xpos[site]) @ frame
        minimum = min(minimum, float((local @ rotation.T + position)[:, 2].min()))
    assert minimum < -0.01


def test_complete_side_pick_place_has_horizontal_insertion_and_withdrawal():
    # Lift and replace on the same support, with ample exposed plate edge.
    # Another pedestal behind the grasp can obstruct the wrist during placement.
    sizes = np.array([[0.18, 0.20, 0.02], [0.04, 0.04, 0.12]])
    model, qpos = models.make_model(sizes, [[0.425, 0, 0.13], [0.425, 0, 0.06]])
    planner = manipulation.Planner(
        model, sizes, planning_seconds=30, grasp_family="side"
    )
    result = planner.plan(qpos, 0, np.array([0.425, 0, 0.13]))
    for segment in result.segments:
        for frame in segment.path:
            assert planner.scene.valid(
                frame, 0, segment.pads_allowed, segment.supports, segment.payload
            )
    for index in (1, -1):
        segment = result.segments[index]
        planner.scene.set_state(segment.path[0])
        start = planner.scene.data.site_xpos[planner.scene.site].copy()
        planner.scene.set_state(segment.path[-1])
        end = planner.scene.data.site_xpos[planner.scene.site].copy()
        assert abs(end[2] - start[2]) < 0.002
        assert np.linalg.norm(end[:2] - start[:2]) > 0.05
