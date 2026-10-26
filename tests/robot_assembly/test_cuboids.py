"""Variable shapes preserve robot settings and use correct pose symmetries."""

import numpy as np

import castle_reasoning.data.assembly.teacher_control as teacher_control
import castle_reasoning.runtime.execution.success as success
import castle_reasoning.runtime.simulation.models as models
import castle_reasoning.runtime.simulation.robot as robot
import tests.helpers.column_collection as column_collection


def test_model_dimensions_mass_coordinates_and_actuators():
    sizes = np.array([[0.03, 0.04, 0.06], [0.10, 0.02, 0.02], [0.06, 0.06, 0.01]])
    sizes = np.tile(sizes, (3, 1))
    goals = np.array([[0.4, 0.0, 0.1]] * len(sizes))
    model, home = models.make_model(sizes, goals)
    baseline, _ = column_collection.make_model(2)
    assert model.nq == 14 + 7 * len(sizes)
    assert model.nv == 14 + 6 * len(sizes)
    np.testing.assert_array_equal(model.actuator_gainprm, baseline.actuator_gainprm)
    np.testing.assert_array_equal(model.actuator_biasprm, baseline.actuator_biasprm)
    for i, size in enumerate(sizes):
        np.testing.assert_allclose(model.geom(f"object_{i}").size, size / 2)
        np.testing.assert_allclose(model.body(f"object_{i}").mass, np.prod(size) * 1240)
        assert model.joint(f"object_joint_{i}").qposadr == 14 + 7 * i
    np.testing.assert_array_equal(home[:6], robot.HOME)


def test_rectangular_yaw_is_not_cube_symmetry():
    sizes = np.array([[0.10, 0.02, 0.02], [0.04, 0.04, 0.02]])
    quat = np.array([[np.sqrt(0.5), 0, 0, np.sqrt(0.5)]] * 2)
    errors = success.orientation_error(quat, sizes)
    np.testing.assert_allclose(errors, [np.pi / 2, 0], atol=1e-6)


def test_grasp_buffers_follow_move_order():
    sizes = np.array([[0.08, 0.02, 0.04], [0.02, 0.06, 0.08]])
    offsets, angles = teacher_control.grasp_parameters(sizes, [1, 0], proposals=2)
    np.testing.assert_allclose(angles[:, 2], [np.pi / 2, np.pi / 2, 0, 0])
    assert offsets[0, 2] > offsets[2, 2]


def test_thin_plate_pinch_clears_the_supporting_course():
    sizes = np.array([[0.06, 0.02, 0.01]])
    offsets, _ = teacher_control.grasp_parameters(sizes, [0], proposals=1)
    # Closed Robotiq fingertips extend approximately 18 mm below the pinch site.
    lowest_pad = sizes[0, 2] / 2 + offsets[0, 2] - 0.018
    assert 0.002 < lowest_pad < sizes[0, 2]


def test_contact_query_allows_micrometer_gap_but_not_missing_support():
    sizes = np.array([[0.04, 0.04, 0.04], [0.12, 0.04, 0.02]])
    positions = np.array([[0.425, 0, 0.02], [0.425, 0, 0.050001]])
    model, qpos = models.make_model(sizes, positions)
    trace = np.repeat(qpos[None], 51, axis=0)
    assert success.contact_report(model, trace, [[0, -1], [1, 0]])["passed"]
    trace[:, 14 + 7 + 2] += 0.001
    assert not success.contact_report(model, trace, [[0, -1], [1, 0]])["passed"]


def test_contact_margin_does_not_relax_support_distance_tolerance():
    sizes = np.array([[0.04, 0.04, 0.04]])
    model, qpos = models.make_model(sizes, np.array([[0.425, 0, 0.0201]]))
    model.geom_margin[:] = 0.001
    trace = np.repeat(qpos[None], 51, axis=0)
    assert not success.contact_report(model, trace, [[0, -1]])["passed"]


def test_waypoint_checks_commanded_part_and_logs_other_pose_drift():
    sizes = np.array([[0.02, 0.02, 0.02], [0.02, 0.02, 0.02]])
    goals = np.array([[0.4, -0.1, 0.01], [0.4, 0.1, 0.01]])
    poses = np.column_stack(
        [goals, [[1, 0, 0, 0], [np.cos(np.pi / 60), 0, 0, np.sin(np.pi / 60)]]]
    )
    result = success.waypoint_check(poses, goals, sizes, part=0)
    assert result["passed"]
    assert result["scene_orientation_error_deg"] > 5
    poses[0, 0] += 0.02
    assert not success.waypoint_check(poses, goals, sizes, part=0)["passed"]
