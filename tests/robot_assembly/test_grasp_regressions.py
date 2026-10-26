"""Adding plate skills must preserve the qualified ordinary-object settings."""

import numpy as np
import warp as wp

import castle_reasoning.runtime.control.parameters as parameters
import castle_reasoning.runtime.execution.traces as traces
import castle_reasoning.runtime.planning.grasps as grasps
import castle_reasoning.runtime.planning.manipulation as manipulation
import castle_reasoning.runtime.simulation.models as models


def test_side_grasps_extend_the_complete_top_down_candidate_set():
    for size in ([0.01, 0.02, 0.01], [0.03, 0.12, 0.01], [0.22, 0.01, 0.04]):
        top = grasps.propose(size)
        combined = grasps.propose(size, family="all")
        assert combined[: len(top)] == top
        assert any(g.roll != np.pi for g in combined[len(top) :])


def test_plate_force_does_not_leak_to_other_parts_or_home():
    plan = {"edge_access": {"plates": [1], "grip_effort": 4}}
    assert parameters.grip_effort(plan, 1) == 4
    assert parameters.grip_effort(plan, 0) == 1
    assert parameters.grip_effort(plan, -1) == 1
    assert parameters.grip_effort({}, 0) == 1


def test_force_profile_is_applied_in_action_order_without_changing_arm_limits():
    profile = wp.array([1.0, 4.0, 1.0], dtype=float, device="cpu")
    step = wp.zeros(1, dtype=int, device="cpu")
    values = np.tile([-150.0, 150.0], (1, 7, 1)).astype(np.float32)
    limits = wp.array(values, dtype=wp.vec2, device="cpu")
    for i, force in enumerate([1, 4, 1]):
        step.fill_(i)
        wp.launch(
            traces.apply_grip_effort, 1, inputs=[profile, step, limits], device="cpu"
        )
        np.testing.assert_array_equal(limits.numpy()[0, 6], [-force, force])
        np.testing.assert_array_equal(limits.numpy()[0, :6], values[0, :6])


def test_shallow_fallback_clears_the_neighbors_of_a_narrow_tower_column():

    # Exact base-column geometry of T0214 after its top beams are removed.
    sizes = np.tile([0.02, 0.02, 0.04], (4, 1))
    positions = [
        [0.405, -0.02, 0.02],
        [0.445, -0.02, 0.02],
        [0.405, 0.02, 0.02],
        [0.445, 0.02, 0.02],
    ]
    model, qpos = models.make_model(sizes, positions)
    planner = manipulation.Planner(
        model, sizes, planning_seconds=10, grasp_family="all"
    )
    result = planner.plan(qpos, 0, np.array([0.4, -0.2, 0.02]))
    assert result.grasp.offset[2] > 0.03
    for segment in result.segments:
        for frame in segment.path:
            assert planner.scene.valid(
                frame, 0, segment.pads_allowed, segment.supports, segment.payload
            )


def test_destination_obstruction_is_ranked_before_spending_on_complete_paths():

    sizes = np.tile([0.02, 0.02, 0.04], (4, 1))
    positions = [
        [0.36, -0.23, 0.02],
        [0.445, -0.02, 0.02],
        [0.405, 0.02, 0.02],
        [0.445, 0.02, 0.02],
    ]
    model, qpos = models.make_model(sizes, positions)
    planner = manipulation.Planner(
        model, sizes, planning_seconds=20, grasp_family="all"
    )
    result = planner.plan(qpos, 0, np.array([0.405, -0.02, 0.02]))
    assert result.grasp.offset[2] > 0.03
    # Deep picks fit the source but cannot be released beside these columns.
    # They must not consume the entire full-path budget before shallow picks.
    assert len(result.attempts) < 3
