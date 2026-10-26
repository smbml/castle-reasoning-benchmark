"""Large contact-rich scenes need more than the small-block buffer defaults."""

import mujoco
import numpy as np

import castle_reasoning.runtime.simulation.models as models


def test_buffers_cover_contact_rich_target_with_twofold_headroom():
    positions = [
        [0.25 + 0.06 * x, -0.18 + 0.06 * y, 0.02] for x in range(6) for y in range(7)
    ]
    model, _ = models.make_model(np.full((42, 3), 0.04), positions)
    state = mujoco.MjData(model)
    mujoco.mj_forward(model, state)
    # Exactly touching contacts have no active constraint until gravity acts.
    potential_rows = 4 * state.ncon
    assert potential_rows > 512
    capacity = models.scene_capacities(model)
    assert capacity.njmax >= 2 * potential_rows
    assert capacity.nconmax >= 2 * state.ncon
    assert capacity.njmax_nnz == capacity.njmax * model.nv


def test_small_scene_keeps_existing_buffer_sizes():
    model, _ = models.make_model([[0.02] * 3], [[0.4, 0, 0.01]])
    capacity = models.scene_capacities(model)
    assert capacity.nconmax == 256
    assert capacity.njmax == 512
