
"""Constraint-solver masks must track device mask changes inside captured graphs."""

import os

import mujoco
import mujoco_warp as mjw
import numpy as np
import pytest
import warp as wp

pytestmark = pytest.mark.skipif(
    os.environ.get("CONSTRUCTION_GPU_TESTS") != "1", reason="Explicit GPU opt-in"
)


@pytest.mark.parametrize("solver", ["CG", "Newton"])
@pytest.mark.parametrize("cone", ["elliptic", "pyramidal"])
@pytest.mark.parametrize("jacobian", ["dense", "sparse"])
@pytest.mark.parametrize("conditional", [True, False])
def test_paused_worlds_do_not_iterate(solver, cone, jacobian, conditional):
    """Pausing and reactivating rows must refresh both done and loop counters."""
    wp.init()
    with wp.ScopedDevice("cuda:0"):
        native = mujoco.MjModel.from_xml_string(f'''<mujoco>
          <option timestep=".002" integrator="implicitfast" solver="{solver}"
                  cone="{cone}" jacobian="{jacobian}" iterations="30"/>
          <worldbody><geom type="plane" size="1 1 .1"/>
            <body pos="0 0 .098"><freejoint/>
              <geom type="box" size=".1 .1 .1" mass="1"/></body>
          </worldbody></mujoco>''')
        model = mjw.put_model(native)
        model.opt.graph_conditional = conditional
        data = mjw.put_data(native, mujoco.MjData(native), nworld=3)
        reference = mjw.put_data(native, mujoco.MjData(native), nworld=3)
        active = wp.ones(3, dtype=bool)
        mjw.step_active(model, data, active)
        mjw.step(model, reference)
        for field in ("qpos", "qvel", "qacc_warmstart"):
            np.testing.assert_allclose(
                getattr(data, field).numpy(), getattr(reference, field).numpy(),
                atol=2e-6, rtol=0,
            )
        with wp.ScopedCapture() as capture:
            mjw.step_active(model, data, active)
        fields = ("qpos", "qvel", "act", "time", "qacc_warmstart")
        for mask in ([True, False, True], [False] * 3,
                     [False, True, False], [True] * 3):
            mask = np.array(mask)
            before = {name: getattr(data, name).numpy() for name in fields}
            active.assign(mask)
            wp.capture_launch(capture.graph)
            iterations = data.solver_niter.numpy()
            np.testing.assert_array_equal(iterations[~mask], 0)
            assert np.all(iterations[mask] > 0)
            for name in fields:
                np.testing.assert_array_equal(
                    getattr(data, name).numpy()[~mask], before[name][~mask]
                )
            assert np.all(data.time.numpy()[mask] > before["time"][mask])
