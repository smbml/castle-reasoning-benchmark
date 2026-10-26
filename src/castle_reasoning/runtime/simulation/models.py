"""Runtime simulation models."""

import hashlib
from pathlib import Path

import mujoco
import numpy as np
from ogbench.manipspace import mjcf_utils
from ogbench.manipspace.envs import cube_env

import castle_reasoning.runtime.diagnostics.timing as timing
import castle_reasoning.runtime.simulation.engine as engine
import castle_reasoning.runtime.simulation.robot as robot


def validate_contact_impedance(value):
    """Normalize two ordered impedance coefficients without MuJoCo clamping."""
    if value is None:
        return None
    coefficients = np.asarray(value, dtype=float)
    if (
        coefficients.shape != (2,)
        or not np.isfinite(coefficients).all()
        or not 0.0001 <= coefficients[0] <= coefficients[1] <= 0.9999
    ):
        raise ValueError("Contact impedance needs 0.0001 <= D0 <= Dwidth <= 0.9999.")
    return tuple(map(float, coefficients))


def make_model(
    sizes,
    positions,
    *,
    table_bounds=None,
    active_part_count=None,
    contact_impedance=None,
):
    """Compile real cuboid inertia while preserving the robot's runtime gains."""
    contact_impedance = validate_contact_impedance(contact_impedance)
    sizes, positions = np.asarray(sizes), np.asarray(positions)
    if sizes.shape != positions.shape or sizes.ndim != 2 or sizes.shape[1] != 3:
        raise ValueError("Sizes and positions must have matching N by 3 shape.")
    if active_part_count is None:
        active_part_count = len(sizes)
    if not 0 <= active_part_count <= len(sizes):
        raise ValueError("Active part count must fit the compiled model.")
    if table_bounds is not None:
        bounds = np.asarray(table_bounds, dtype=float)
        if (
            bounds.shape != (4,)
            or not np.isfinite(bounds).all()
            or bounds[0] >= bounds[1]
            or bounds[2] >= bounds[3]
        ):
            raise ValueError("Table bounds must be finite xmin, xmax, ymin, ymax.")
    env = cube_env.CubeEnv(
        env_type="double",
        permute_blocks=False,
        physics_timestep=0.002,
        control_timestep=0.02,
        ob_type="states",
        visualize_info=False,
    )
    try:
        env.reset(seed=0, options={"task_id": 2})
        arena = env.build_mjcf_model()
        if table_bounds is not None:
            xmin, xmax, ymin, ymax = bounds
            floor = arena.find("geom", "floor")
            floor.type = "box"
            floor.pos = [(xmin + xmax) / 2, (ymin + ymax) / 2, -0.04]
            floor.size = [(xmax - xmin) / 2, (ymax - ymin) / 2, 0.04]
            # CubeEnv's invisible containment walls cross the expanded staging
            # area. The finite-table variant has an open edge for plate pickup.
            for i in range(4):
                arena.find("geom", f"wall_{i}").remove()
        for i in range(2):
            arena.find("body", f"object_{i}").remove()
            arena.find("body", f"object_target_{i}").remove()
        for i, (size, position) in enumerate(zip(sizes, positions, strict=True)):
            body = arena.worldbody.add(
                "body",
                name=f"object_{i}",
                pos=position,
                gravcomp=float(i >= active_part_count),
            )
            body.add("freejoint", name=f"object_joint_{i}")
            color = [0.80, 0.57, 0.36, 1]
            if size[2] < max(size[:2]):
                color = [0.33, 0.57, 0.52, 1]
            if min(size[:2]) >= 0.04 and size[2] <= 0.02:
                color = [0.37, 0.55, 0.65, 1]
            body.add(
                "geom",
                name=f"object_{i}",
                type="box",
                size=size / 2,
                rgba=color,
                density=1240,
                solref=[0.004, 1],
                contype=3,
                group=1,
            )
        arena.option.timestep = 0.002
        model = mujoco.MjModel.from_xml_string(
            mjcf_utils.to_string(arena), mjcf_utils.get_assets(arena)
        )
        if contact_impedance is not None:
            for geom in range(model.ngeom):
                name = model.geom(geom).name
                if name == "floor" or name.startswith("object_"):
                    model.geom_solimp[geom, :2] = contact_impedance
        # OGBench edits these after compilation; XML alone does not preserve them.
        for name in ("actuator_gainprm", "actuator_biasprm", "actuator_dynprm"):
            np.copyto(getattr(model, name), getattr(env.model, name))
        home = model.qpos0.copy()
        home[:6], home[6:14] = robot.HOME, 0
        home[14:].reshape(-1, 7)[:, :3] = positions
        home[14:].reshape(-1, 7)[:, 3:] = [1, 0, 0, 0]
        return model, home
    finally:
        env.close()


def scene_capacities(model):
    """Size dynamic buffers with headroom for the target's contact manifold."""
    state = mujoco.MjData(model)
    mujoco.mj_forward(model, state)
    # At exact touching, contacts exist before their constraints activate under
    # gravity. Include their potential friction rows, not just current nefc.
    contact_rows = sum(max(1, 2 * (int(c.dim) - 1)) for c in state.contact)
    nconmax = max(256, 1 << max(0, 2 * state.ncon - 1).bit_length())
    rows = max(state.nefc, contact_rows) * 2
    njmax = max(512, 1 << max(0, rows - 1).bit_length())
    return engine.BufferCapacities(
        nconmax=nconmax, njmax=njmax, njmax_nnz=njmax * model.nv
    )


def _named_id(model: mujoco.MjModel, object_type: int, name: str) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise ValueError(f"OGBench model has no {name!r} object.")
    return object_id


def active_parts(plan):
    """Indices belonging to the construction task rather than batch padding."""
    return range(plan.get("active_part_count", len(plan["sizes"])))


@timing.measured('startup.compile_model')
def compile_model(plan, config, *, contact_impedance=None):
    """Compile the same physical model for fixed and continuously reused slots."""
    edge = plan.get("edge_access")
    model, home = make_model(
        plan["sizes"],
        plan["target"],
        table_bounds=edge["table_bounds"] if edge else None,
        active_part_count=len(active_parts(plan)),
        contact_impedance=contact_impedance,
    )
    if edge:
        model.geom("floor").solref[:] = [edge["table_contact_seconds"], 1]
    model.opt.timestep = config.physics_timestep
    model.actuator_forcerange[6] = [-config.grip_effort, config.grip_effort]
    return model, home


def save_model(model, path):
    """Write and hash the same native MJB buffer without rereading remote storage."""
    buffer = np.empty(mujoco.mj_sizeModel(model), np.uint8)
    mujoco.mj_saveModel(model, buffer=buffer)
    with timing.span("startup.model_write"):
        Path(path).write_bytes(buffer)
    with timing.span("startup.model_hash"):
        return hashlib.sha256(buffer).hexdigest()
