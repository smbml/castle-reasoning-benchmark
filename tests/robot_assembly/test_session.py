"""Policy-free state matches the existing physical collection setup."""

import os

import numpy as np
import pytest


@pytest.mark.gpu
@pytest.mark.skipif(os.environ.get("CONSTRUCTION_GPU_TESTS") != "1", reason="explicit local GPU opt-in required")
def test_session_reset_and_replay_match_teacher_physics():
    from castle_reasoning.data.assembly import teacher
    from castle_reasoning.runtime.control import tracking
    from castle_reasoning.runtime.execution import session, traces
    from castle_reasoning.runtime.simulation import models

    positions = np.array([[0.425, -0.1, 0.02], [0.425, 0.1, 0.02]])
    model, home = models.make_model([[0.04] * 3] * 2, positions)
    reference = teacher.make_collector(model, positions, 1, 1, 7)
    requested = session.make_session(model, seed=7)
    actions = np.zeros((5, 1, 7), np.float32)
    actions[..., 6] = -1
    outputs = []
    for state in (reference, requested):
        state.controller = tracking.PathControl(state)
        state.reset(qpos=home, qvel=np.zeros(model.nv))
        initial = state.engine.snapshot()
        trace, _ = traces.run_controls(state.controller, 5, mode="replay", actions=actions)
        outputs.append((initial, trace))
    for key in ("qpos", "qvel"):
        np.testing.assert_array_equal(outputs[0][0][key], outputs[1][0][key])
        np.testing.assert_array_equal(outputs[0][1][key], outputs[1][1][key])
