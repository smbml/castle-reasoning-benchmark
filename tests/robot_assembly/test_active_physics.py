
"""Real contact dynamics qualification for the repo-owned MJ-Warp patch."""

import os

import mujoco
import mujoco_warp as mjw
import numpy as np
import pytest
import warp as wp

pytestmark = pytest.mark.skipif(
    os.environ.get("CONSTRUCTION_GPU_TESTS") != "1", reason="Explicit GPU opt-in"
)


@pytest.mark.parametrize("integrator", ["Euler", "implicitfast"])
def test_pause_preserves_integration_state_and_resumes(integrator):
    assert hasattr(mjw, "step_active"), "Repo-owned active-world runtime required"
    wp.init()
    with wp.ScopedDevice("cuda:0"):
        native = mujoco.MjModel.from_xml_string(f'''<mujoco>
          <option timestep="0.002" integrator="{integrator}" solver="CG"/>
          <worldbody><geom type="plane" size="1 1 .1"/>
            <body pos="0 0 .1"><freejoint/><geom type="box" size=".1 .1 .1" mass="1"/></body>
            <body pos=".4 0 .2"><joint name="hinge"/><geom size=".05"/></body>
          </worldbody><actuator><general joint="hinge" dyntype="filter" dynprm=".1"/></actuator>
        </mujoco>''')
        model = mjw.put_model(native)
        data = mjw.put_data(native, mujoco.MjData(native), nworld=2)
        control = np.ones((2, native.nu), np.float32) * 0.1
        data.ctrl.assign(control)
        vel = np.zeros((2, native.nv), np.float32)
        vel[:, 0] = 0.2
        data.qvel.assign(vel)
        mask = wp.array([True, True], dtype=bool)
        for _ in range(10):
            mjw.step_active(model, data, mask)
        fields = ("qpos", "qvel", "act", "time", "qacc_warmstart", "ctrl")
        frozen = {key: getattr(data, key).numpy().copy() for key in fields}
        mask.assign(np.array([False, True]))
        with wp.ScopedCapture() as capture:
            mjw.step_active(model, data, mask)
        for _ in range(25):
            wp.capture_launch(capture.graph)
        for key in fields:
            np.testing.assert_array_equal(getattr(data, key).numpy()[0], frozen[key][0])
        assert data.time.numpy()[1] > frozen["time"][1]
        assert not np.array_equal(data.qpos.numpy()[1], frozen["qpos"][1])
        mask.assign(np.array([True, False]))
        wp.capture_launch(capture.graph)
        assert data.time.numpy()[0] > frozen["time"][0]
        mask.fill_(False)
        frozen = {key: getattr(data, key).numpy().copy() for key in fields}
        wp.capture_launch(capture.graph)
        for key in fields:
            np.testing.assert_array_equal(getattr(data, key).numpy(), frozen[key])


def test_rejects_unsupported_integrator_before_mutation():
    assert hasattr(mjw, "step_active"), "Repo-owned active-world runtime required"
    with wp.ScopedDevice("cuda:0"):
        native = mujoco.MjModel.from_xml_string("""<mujoco><option integrator="RK4"/>
          <worldbody><body><joint/><geom size=".1"/></body></worldbody></mujoco>""")
        model = mjw.put_model(native)
        data = mjw.put_data(native, mujoco.MjData(native), nworld=2)
        with pytest.raises(ValueError, match="Euler or implicitfast"):
            mjw.step_active(model, data, wp.ones(2, dtype=bool))
        np.testing.assert_array_equal(data.time.numpy(), [0, 0])


def test_contact_grasp_survives_pause_and_resumes_like_reference():
    """A force-held payload stays suspended while the other world advances."""
    with wp.ScopedDevice("cuda:0"):
        native = mujoco.MjModel.from_xml_string("""<mujoco>
          <option timestep=".002" integrator="implicitfast" solver="CG" iterations="100"/>
          <default><geom friction="2 .01 .001" solref=".01 1"/></default>
          <worldbody><geom type="plane" size="1 1 .1"/>
            <body pos="0 0 .2"><freejoint/><geom type="box" size=".025 .025 .025" mass=".03"/></body>
            <body pos="-.034 0 .2"><joint name="left" type="slide" axis="1 0 0"/>
              <geom type="box" size=".01 .035 .04" mass=".1"/></body>
            <body pos=".034 0 .2"><joint name="right" type="slide" axis="-1 0 0"/>
              <geom type="box" size=".01 .035 .04" mass=".1"/></body>
          </worldbody><actuator>
            <position joint="left" kp="1000" kv="10"/>
            <position joint="right" kp="1000" kv="10"/>
          </actuator></mujoco>""")
        model = mjw.put_model(native)
        data = mjw.put_data(native, mujoco.MjData(native), nworld=2)
        data.ctrl.fill_(0.003)
        active = wp.ones(2, dtype=bool)
        with wp.ScopedCapture() as capture:
            mjw.step_active(model, data, active)
        for _ in range(100):
            wp.capture_launch(capture.graph)
        assert data.qpos.numpy()[0, 2] > 0.15, "Fixture never established a grasp"
        assert data.nacon.numpy()[0] >= 4
        before = {
            key: getattr(data, key).numpy()
            for key in ("qpos", "qvel", "act", "time", "qacc_warmstart")
        }
        reference = mjw.put_data(native, mujoco.MjData(native), nworld=2)
        reference.ctrl.fill_(0.003)
        for key, values in before.items():
            getattr(reference, key).assign(values)
        active.assign(np.array([False, True]))
        for _ in range(100):
            wp.capture_launch(capture.graph)
        for key, values in before.items():
            np.testing.assert_array_equal(getattr(data, key).numpy()[0], values[0])
        active.assign(np.array([True, False]))
        for _ in range(20):
            wp.capture_launch(capture.graph)
            mjw.step_active(model, reference, active)
        assert data.qpos.numpy()[0, 2] > 0.15
        np.testing.assert_allclose(
            data.qpos.numpy()[0], reference.qpos.numpy()[0], atol=2e-5, rtol=0
        )
        np.testing.assert_allclose(
            data.qvel.numpy()[0], reference.qvel.numpy()[0], atol=2e-4, rtol=0
        )
