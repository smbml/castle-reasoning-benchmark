"""Execution respects the supports of the recorded staging configuration."""

import numpy as np

import castle_reasoning.data.generation.geometry as geometry
import castle_reasoning.data.generation.waypoints as waypoints
import castle_reasoning.data.plans as plans
import castle_reasoning.runtime.execution.success as success
import castle_reasoning.runtime.planning.manipulation as manipulation
import castle_reasoning.runtime.simulation.models as models


def test_alternate_removal_cannot_place_on_an_absent_staging_support():
    # All three target parts are independently removable; staging is one stack.
    plan = waypoints.plan([geometry.Part((x, 0, 0), (2, 2, 2)) for x in (0, 4, 8)])
    remaining = {0, 1, 2}
    assert list(plans.removable(plan, remaining)) == [0]
    remaining.remove(0)
    assert list(plans.removable(plan, remaining)) == [1]
    remaining.remove(1)
    assert list(plans.removable(plan, remaining)) == [2]


def test_collision_checked_placement_and_retrieval_on_an_existing_stack():
    sizes = np.array([[0.02] * 3] * 2)
    model, qpos = models.make_model(sizes, [[0.425, 0, 0.01], [0.425, -0.18, 0.01]])
    planner = manipulation.Planner(model, sizes, planning_seconds=20)
    result = planner.plan(qpos, 0, np.array([0.425, -0.18, 0.03]))
    for segment in result.segments:
        assert all(
            planner.scene.valid(
                q, 0, segment.pads_allowed, segment.supports, segment.payload
            )
            for q in segment.path
        )
    # Fresh settled scene: this test exercises geometry, not physical grasp success.
    qpos[14:17] = [0.425, -0.18, 0.03]
    lifted = planner.plan(qpos, 0, np.array([0.425, 0, 0.01]))
    assert model.geom("object_1").id in lifted.segments[3].supports
    check = success.contact_report(
        model, np.repeat(qpos[None], 51, axis=0), [[0, 1], [1, -1]]
    )
    assert check["passed"]
    qpos[16] += 0.002
    assert not success.contact_report(
        model, np.repeat(qpos[None], 51, axis=0), [[0, 1]]
    )["passed"]


def test_seeded_grasp_variation_is_reproducible_and_collision_checked():
    sizes = np.array([[0.02] * 3])
    model, qpos = models.make_model(sizes, [[0.425, 0, 0.01]])
    results = []
    for seed in (0, 1, 0):
        planner = manipulation.Planner(
            model, sizes, planning_seconds=20, seed=seed, shuffle_grasps=True
        )
        result = planner.plan(qpos, 0, np.array([0.425, -0.18, 0.01]))
        results.append(result.grasp)
        for segment in result.segments:
            assert all(
                planner.scene.valid(
                    q, 0, segment.pads_allowed, segment.supports, segment.payload
                )
                for q in segment.path
            )
    assert results[0] == results[2]
    assert results[0] != results[1]
