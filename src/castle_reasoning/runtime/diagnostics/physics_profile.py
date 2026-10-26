"""Runtime diagnostics physics_profile."""

import argparse
import ctypes
import dataclasses
import hashlib
import json
import time
from pathlib import Path

import numpy as np

import castle_reasoning.runtime.control.tracking as tracking
import castle_reasoning.runtime.execution.session as session
import castle_reasoning.runtime.execution.traces as traces
import castle_reasoning.runtime.simulation.models as models


def load_prefix(path, *, start, count):
    """Load aligned pre/post states, actions, and their original grip forces."""
    with np.load(path) as episode:
        length = len(episode["action"])
        if start < 0 or count <= 0 or start + count > length:
            raise ValueError("Requested prefix is outside the recorded actions.")
        if any(len(episode[k]) != length + 1 for k in ("qpos", "qvel")):
            raise ValueError("Recorded prefix requires one more state than action.")
        if len(episode["grip_effort"]) != length:
            raise ValueError("Recorded prefix requires one grip force per action.")
        result = {
            key: episode[key][start : start + count + 1].copy()
            for key in ("qpos", "qvel")
        }
        result.update(
            {
                key: episode[key][start : start + count].copy()
                for key in ("action", "grip_effort")
            }
        )
    if not all(np.isfinite(value).all() for value in result.values()):
        raise ValueError("Recorded prefix contains nonfinite values.")
    if not np.all(result["grip_effort"] > 0):
        raise ValueError("Recorded prefix contains nonpositive grip force.")
    return result


def configure_solver(model, solver, cholesky_threads):
    """Validate diagnostic overrides without changing unrelated model options."""
    import mujoco

    choices = {"cg": mujoco.mjtSolver.mjSOL_CG, "newton": mujoco.mjtSolver.mjSOL_NEWTON}
    if solver != "native" and solver not in choices:
        raise ValueError(f"Unsupported solver: {solver}")
    if cholesky_threads not in (None, 32, 64, 128, 256):
        raise ValueError("Cholesky block size must be 32, 64, 128 or 256.")
    if solver != "native":
        model.opt.solver = choices[solver]


def run(
    source,
    output,
    *,
    stage,
    count,
    start=0,
    worlds=1,
    capture=False,
    solver="native",
    cholesky_threads=None,
):
    """Measure production replay without labeling the diagnostic as valid BC."""
    import mujoco
    import warp as wp

    sample = load_prefix(source / f"{stage}.npz", start=start, count=count)
    model_path = source / "model.mjb"
    model = mujoco.MjModel.from_binary_path(str(model_path))
    original_solver = int(model.opt.solver)
    configure_solver(model, solver, cholesky_threads)
    output.mkdir(parents=True, exist_ok=False)
    began = time.perf_counter()
    actions = np.repeat(sample["action"][:, None], worlds, axis=1)
    initial = {
        key: np.repeat(sample[key][:1], worlds, axis=0) for key in ("qpos", "qvel")
    }
    with wp.ScopedDevice("cuda:0"):
        collector = session.make_session(model, worlds=worlds, seed=0)
        if cholesky_threads is not None:
            collector.engine.model.block_dim.update_gradient_cholesky_blocked = (
                cholesky_threads
            )
        controller = tracking.PathControl(collector)
        collector.controller = controller
        collector.reset(**initial)
        traces.run_controls(
            controller,
            min(count, 5),
            mode="replay",
            actions=actions[:5],
            grip_efforts=sample["grip_effort"][:5],
        )
        collector.reset(**initial)
        warmup_seconds = time.perf_counter() - began
        driver = ctypes.CDLL("libcuda.so.1") if capture else None
        with wp.ScopedDevice(collector.engine.device):
            if driver is not None:
                code = driver.cuProfilerStart()
                if code != 0:
                    raise RuntimeError(f"CUDA profiler could not start: {code}.")
            try:
                trace, seconds = traces.run_controls(
                    controller,
                    count,
                    mode="replay",
                    actions=actions,
                    grip_efforts=sample["grip_effort"],
                )
            finally:
                if driver is not None:
                    driver.cuProfilerStop()
        wp.synchronize_device(collector.engine.device)
        iterations = collector.engine.data.solver_niter.numpy().tolist()
    actual = trace["qpos"][:, 0]
    expected = sample["qpos"][1:]
    error = (actual[:, 14:] - expected[:, 14:]).reshape(count, -1, 7)
    result = {
        "source": str(source.resolve()),
        "stage": stage,
        "start": start,
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "episode_sha256": hashlib.sha256((source / f"{stage}.npz").read_bytes()).hexdigest(),
        "worlds": worlds,
        "controls_per_world": count,
        "warmup_seconds": warmup_seconds,
        "physics_seconds": seconds,
        "aggregate_controls_per_second": worlds * count / seconds,
        "aggregate_physics_steps_per_second": worlds
        * count
        * round(0.02 / model.opt.timestep)
        / seconds,
        "solver_niter_final": iterations,
        "nsys_capture": capture,
        "original_solver": original_solver,
        "requested_solver": solver,
        "cholesky_threads": collector.engine.model.block_dim.update_gradient_cholesky_blocked,
        "max_object_center_deviation_m": float(
            np.linalg.norm(error[..., :3], axis=-1).max()
        ),
        "finite": all(np.isfinite(value).all() for value in trace.values()),
        "accepted_bc": False,
        "physics_runtime_overrides": [],
        "reset": "recorded pre-action qpos/qvel; fresh servo/solver state",
        "capacities": dataclasses.asdict(models.scene_capacities(model)),
        "runtime_tolerance": collector.engine.model.opt.tolerance.numpy().tolist(),
        "native_options": {
            key: float(getattr(model.opt, key))
            for key in (
                "solver",
                "iterations",
                "ls_iterations",
                "tolerance",
                "ls_tolerance",
                "timestep",
                "cone",
                "jacobian",
            )
        },
    }
    np.savez_compressed(output / "trace.npz", **trace)
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--stage", default="disassembly")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--worlds", type=int, default=1)
    parser.add_argument("--capture", action="store_true")
    parser.add_argument(
        "--solver", choices=("native", "cg", "newton"), default="native"
    )
    parser.add_argument("--cholesky-threads", type=int)
    run(**vars(parser.parse_args()))
