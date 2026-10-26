"""Runtime diagnostics checks."""

import importlib.metadata
from typing import Any

import mujoco
import numpy as np
import warp as wp

import castle_reasoning.runtime.diagnostics.provenance as provenance
import castle_reasoning.runtime.planning.arm as arm
import castle_reasoning.runtime.simulation.engine as simulation_engine
import castle_reasoning.runtime.simulation.models as models


def check_runtime() -> dict:
    """Verify source/build identity and a finite native workcell step."""
    import mujoco
    import numpy as np
    verified_sources = provenance.bundled_sources()
    active_runtime = provenance.runtime_provenance()
    model, qpos = models.make_model(
        [[0.04, 0.04, 0.04], [0.12, 0.04, 0.02]],
        [[0.425, -0.1, 0.02], [0.425, 0.1, 0.01]],
    )
    state = mujoco.MjData(model)
    state.qpos[:] = qpos
    mujoco.mj_forward(model, state)
    mujoco.mj_step(model, state)
    if not np.isfinite(state.qpos).all() or not np.isfinite(state.qvel).all():
        raise RuntimeError("Native workcell step produced nonfinite state")
    return {
        "sources": verified_sources,
        "active_runtime": active_runtime,
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "castle-reasoning-benchmark",
                "ogbench",
                "mujoco",
                "mujoco-warp",
                "warp-lang",
                "numpy",
                "scipy",
            )
        },
        "native_step_finite": True,
        "gpu_tested": False,
    }


def compare_pinch_jacobian(
    *,
    model: mujoco.MjModel,
    qpos: np.ndarray,
    qvel: np.ndarray,
    device: str,
) -> dict[str, Any]:
    """Compares the MJWarp pinch-site Jacobian with native MuJoCo."""
    qpos = np.asarray(qpos, dtype=np.float64)
    qvel = np.asarray(qvel, dtype=np.float64)
    if qpos.shape != (model.nq,) or qvel.shape != (model.nv,):
        raise ValueError("Generalized state does not match the model.")
    if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
        raise ValueError("Generalized state must be finite.")
    site_id = models._named_id(model, mujoco.mjtObj.mjOBJ_SITE, "ur5e/robotiq/pinch")
    body_id = int(model.site_bodyid[site_id])
    root_body_id = int(model.body_rootid[body_id])

    native_data = mujoco.MjData(model)
    native_data.qpos[:] = qpos
    native_data.qvel[:] = qvel
    mujoco.mj_forward(model, native_data)
    native_position = np.empty((3, model.nv), dtype=np.float64)
    native_rotation = np.empty((3, model.nv), dtype=np.float64)
    mujoco.mj_jacSite(
        model, native_data, native_position, native_rotation, site_id
    )

    engine = simulation_engine.MJWarpEngine(
        model, nworld=1, device=device, graph=False
    )
    engine.set_state(qpos=qpos[None], qvel=qvel[None])
    with wp.ScopedDevice(engine.device):
        jacobian = wp.zeros((1, 6, 6), dtype=float)
        wp.launch(
            arm._pinch_jacobian,
            dim=(1, 6),
            inputs=[
                engine.data.site_xpos,
                engine.data.subtree_com,
                engine.data.cdof,
                site_id,
                root_body_id,
                jacobian,
            ],
        )
        wp.synchronize_device(engine.device)
        actual = jacobian.numpy()[0]
    position_error = float(np.max(np.abs(actual[:3] - native_position[:, :6])))
    rotation_error = float(np.max(np.abs(actual[3:] - native_rotation[:, :6])))
    tolerance = 2e-6
    return {
        "passed": position_error <= tolerance and rotation_error <= tolerance,
        "site_id": site_id,
        "body_id": body_id,
        "root_body_id": root_body_id,
        "position_max_error": position_error,
        "rotation_max_error": rotation_error,
        "tolerance": tolerance,
    }
