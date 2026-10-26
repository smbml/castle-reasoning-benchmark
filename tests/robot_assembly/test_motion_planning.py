"""Real geometry regressions for cuboid grasp and arm-path planning."""

import itertools
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

import castle_reasoning.runtime.control.guards as guards
import castle_reasoning.runtime.planning.arm as arm
import castle_reasoning.runtime.planning.collision as collision
import castle_reasoning.runtime.planning.grasps as grasps
import castle_reasoning.runtime.planning.manipulation as manipulation
import castle_reasoning.runtime.simulation.models as models


def test_grasp_proposals_are_distinct_and_use_the_selected_axis_width():
    candidates = grasps.propose(np.array([0.08, 0.02, 0.04]))
    assert 4 <= len(candidates) <= 16
    signatures = {(tuple(g.offset), g.yaw) for g in candidates}
    assert len(signatures) == len(candidates)
    assert any(abs(g.offset[0]) > 0.01 for g in candidates)
    assert all(g.width == 0.02 for g in candidates)
    square = grasps.propose(np.array([0.03, 0.04, 0.02]))
    assert {g.width for g in square} == {0.03, 0.04}


def test_real_collision_fixture_rejects_neighbor_during_descent():
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/cuboid_collisions.json").read_text()
    )
    for case in fixture:
        model, _ = models.make_model(case["sizes"], case["target"])
        scene = collision.Scene(model, np.array(case["qpos"]))
        hits = scene.violations(
            np.array(case["qpos"]), selected=case["selected"], pads_allowed=True
        )
        assert any(case["neighbor"] in hit.names for hit in hits)
        assert not scene.valid(np.array(case["qpos"]), case["selected"], True)


def test_whole_gripper_is_not_exempted_for_a_selected_payload():
    model, qpos = models.make_model([[0.02] * 3], [[0.4, 0, 0.01]])
    scene = collision.Scene(model, qpos)
    pad = model.geom("ur5e/robotiq/left_pad1").id
    housing = next(i for i in scene.robot_geoms if "unnamed" in model.geom(i).name)
    obj = model.geom("object_0").id
    assert scene.allowed_pair(pad, obj, 0, True, ())
    assert not scene.allowed_pair(housing, obj, 0, True, ())


def test_edge_cannot_jump_through_a_thin_obstacle():
    model = mujoco.MjModel.from_xml_string("""
      <mujoco><worldbody>
        <geom type="box" size=".001 .1 .1" pos="0 0 0"/>
        <body><joint type="slide" axis="1 0 0"/>
          <geom type="sphere" size=".005" mass="1"/>
        </body></worldbody></mujoco>""")
    state = mujoco.MjData(model)

    def valid(q):
        state.qpos[:] = q
        mujoco.mj_kinematics(model, state)
        return mujoco.mj_geomDistance(model, state, 0, 1, 1, None) > 0

    assert valid(np.array([-0.1])) and valid(np.array([0.1]))
    assert not arm.edge_valid(np.array([-0.1]), np.array([0.1]), valid)


def test_rrt_routes_around_obstacle_and_checks_every_edge():
    model = mujoco.MjModel.from_xml_string("""
      <mujoco><worldbody><geom type="box" size=".04 .04 .1"/>
        <body><joint type="slide" axis="1 0 0"/>
          <joint type="slide" axis="0 1 0"/>
          <geom type="sphere" size=".01" mass="1"/>
        </body></worldbody></mujoco>""")
    state = mujoco.MjData(model)

    def valid(q):
        state.qpos[:] = q
        mujoco.mj_kinematics(model, state)
        return mujoco.mj_geomDistance(model, state, 0, 1, 1, None) > 0.002

    path = arm.connect(
        np.array([-0.15, 0]),
        np.array([0.15, 0]),
        valid,
        np.array([[-0.2, 0.2], [-0.2, 0.2]]),
        seed=3,
    )
    assert len(path) > 2
    assert all(
        arm.edge_valid(a, b, valid) for a, b in itertools.pairwise(path)
    )
    with pytest.raises(arm.PlanningError, match="budget"):
        arm.connect(
            path[0], path[-1], valid, np.array([[-0.2, 0.2], [-0.2, 0.2]]), max_nodes=0
        )


def test_complete_pick_place_path_is_checked_with_payload():

    sizes = np.array([[0.08, 0.02, 0.04]])
    model, qpos = models.make_model(sizes, [[0.425, 0, 0.02]])
    planner = manipulation.Planner(model, sizes, planning_seconds=15)
    result = planner.plan(qpos, 0, np.array([0.43, -0.18, 0.02]))
    assert [s.name for s in result.segments] == [
        "approach",
        "descend",
        "close",
        "lift",
        "transit",
        "place",
        "setdown",
        "open",
        "retreat",
    ]
    assert result.query_count > 10
    for segment in result.segments:
        for q in segment.path:
            assert planner.scene.valid(
                q, 0, segment.pads_allowed, segment.supports, segment.payload
            )
    assert result.segments[4].payload is not None


def test_obstructed_center_grasp_selects_a_clear_beam_end():

    # The obstacle must reach the shallow grasp too; a 2 cm neighbor leaves the
    # upper part of this 4 cm beam accessible from the center.
    sizes = np.array([[0.12, 0.02, 0.04], [0.02, 0.02, 0.04]])
    model, qpos = models.make_model(sizes, [[0.425, 0, 0.02], [0.425, 0.035, 0.02]])
    planner = manipulation.Planner(model, sizes, planning_seconds=15)
    centered = [
        planner.aperture.calibrate(g)
        for g in grasps.propose(sizes[0])
        if np.linalg.norm(g.offset[:2]) < 1e-6
    ]
    assert centered and all(planner._source_obstructed(qpos, 0, g) for g in centered)
    result = planner.plan(qpos, 0, np.array([0.425, -0.18, 0.02]))
    assert abs(result.grasp.offset[0]) > 0.03
    for segment in result.segments:
        for q in segment.path:
            assert planner.scene.valid(
                q, 0, segment.pads_allowed, segment.supports, segment.payload
            )


def test_replay_audit_rejects_collision_even_without_a_live_guard():

    case = json.loads(
        (Path(__file__).parent / "fixtures/cuboid_collisions.json").read_text()
    )[0]
    model, _ = models.make_model(case["sizes"], case["target"])
    scene = collision.Scene(model, np.array(case["qpos"]), margin=0)
    hits = guards.audit_contacts(
        scene, np.array([case["qpos"]]), case["selected"], True, (), False
    )
    assert hits and any(case["neighbor"] in c["names"] for c in hits[0]["contacts"])


def test_payload_collision_is_checked_even_when_the_arm_is_clear():
    model, q = models.make_model(
        [[0.02] * 3, [0.02] * 3], [[0.425, 0, 0.01], [0.441, 0, 0.01]]
    )
    scene = collision.Scene(model, q)
    assert scene.valid(q)  # This query treats the objects as a static scene.
    payload = scene.attachment(q, 0)
    assert not scene.valid(q, 0, True, (model.geom("floor").id,), payload)


def test_switching_payloads_does_not_reuse_another_objects_collision_pairs():
    model, q = models.make_model(
        [[0.02] * 3] * 3,
        [[0.425, 0, 0.01], [0.441, 0, 0.01], [0.5, 0, 0.01]],
    )
    scene = collision.Scene(model, q)
    floor = model.geom("floor").id
    for selected, clear in ((0, False), (2, True), (1, False), (2, True)):
        assert scene.valid(
            q, selected, True, (floor,), scene.attachment(q, selected)
        ) == clear
        assert scene.valid(q)


def test_declared_support_contact_does_not_allow_deep_penetration():
    model, q = models.make_model([[0.02] * 3], [[0.425, 0, 0.0098]])
    scene = collision.Scene(model, q)
    payload = scene.attachment(q, 0)
    assert not scene.valid(q, 0, True, (), payload)
    assert scene.valid(q, 0, True, (model.geom("floor").id,), payload)
    q[16] = 0.007
    assert not scene.valid(
        q, 0, True, (model.geom("floor").id,), scene.attachment(q, 0)
    )


def test_ik_does_not_return_a_pose_equivalent_out_of_limit_seed():
    model, q = models.make_model([[0.02] * 3], [[0.425, 0, 0.01]])
    scene = collision.Scene(model, q)
    scene.set_state(q)
    position = scene.data.site_xpos[scene.site].copy()
    rotation = scene.data.site_xmat[scene.site].reshape(3, 3).copy()
    q[5] += 8 * np.pi
    solved = arm.inverse_kinematics(scene, position, rotation, q)
    assert np.all(solved[:6] <= model.jnt_range[:6, 1])
    assert np.all(solved[:6] >= model.jnt_range[:6, 0])
