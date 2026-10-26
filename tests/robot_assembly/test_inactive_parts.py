"""Padding must not become a removal goal or force excessive robot clearance."""

import numpy as np

import castle_reasoning.runtime.planning.manipulation as manipulation
import castle_reasoning.runtime.simulation.models as models


def test_inactive_slots_do_not_raise_the_obstacle_ceiling_or_become_supports():
    sizes = [[0.02] * 3, [0.01] * 3]
    model, state = models.make_model(
        sizes, [[0.425, 0, 0.01], [2, 0, 0.5]], active_part_count=1
    )
    planner = manipulation.Planner(model, sizes, active_part_count=1)
    np.testing.assert_allclose(planner.object_tops(state), [0.02])
    # Even an accidentally coincident inactive slot cannot be selected as support.
    state[21:24] = [0.425, 0, -0.005]
    assert planner._source_supports(state, 0) == (model.geom("floor").id,)


def test_stage_part_count_is_the_actual_task_count():
    plan = {"sizes": [[0.02] * 3] * 42, "active_part_count": 7}
    assert models.active_parts(plan) == range(7)
    assert models.active_parts({"sizes": [[0.02] * 3]}) == range(1)
