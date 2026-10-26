"""Data verification screen."""

import argparse
import collections
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import warp as wp

import castle_reasoning.data.assembly.records as records
import castle_reasoning.data.assembly.teacher as teacher
import castle_reasoning.data.plans as data_plans
import castle_reasoning.data.verification.stability as stability
import castle_reasoning.runtime.control.tracking as tracking
import castle_reasoning.runtime.execution.traces as traces
import castle_reasoning.runtime.simulation.model_batch as model_batch
import castle_reasoning.runtime.simulation.models as simulation_models
import castle_reasoning.runtime.simulation.world_slots as world_slots
from castle_reasoning.runtime.diagnostics import io_timing

SETTLE_STEPS = 250


def timestep(plan):
    limit = plan.get("edge_access", {}).get("max_physics_timestep_s")
    return 0.002 if limit is None else min(0.002, limit)


def batches(plans, max_worlds, max_plans, replicates=1):
    """Consecutive plans of one timestep whose states fit one batch."""
    groups = collections.defaultdict(list)
    for plan in plans:
        groups[timestep(plan)].append(plan)
    for step, members in sorted(groups.items()):
        batch, worlds = [], 0
        for plan in members:
            states = replicates * len(stability.configurations(plan))
            if batch and (worlds + states > max_worlds or len(batch) == max_plans):
                yield step, batch
                batch, worlds = [], 0
            batch.append(plan)
            worlds += states
        if batch:
            yield step, batch


def screen(plans, step, seconds, contact_impedance, rng, jitter, replicates):
    capacity = max(len(p["sizes"]) for p in plans)
    padded = [model_batch.pad_plan(p, capacity) for p in plans]
    config = data_plans.control_config(step)
    models, homes = [], []
    for plan in padded:
        model, home = simulation_models.compile_model(
            plan, config, contact_impedance=contact_impedance
        )
        model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        model.opt.solver = mujoco.mjtSolver.mjSOL_CG
        model.opt.iterations = 1000
        models.append(model)
        homes.append(home)
    worlds = [
        (index, number, stability.placed(state, rng, jitter))
        for index, plan in enumerate(padded)
        for _ in range(replicates)
        for number, state in enumerate(stability.configurations(plan))
    ]
    assignments = [index for index, _, _ in worlds]
    qpos = np.stack(
        [stability.initial_qpos(homes[i], s["goals"]) for i, _, s in worlds]
    )
    steps = round(seconds / 0.02)
    pool = world_slots.ModelPool(models, assignments)
    with wp.ScopedDevice("cuda:0"):
        collector = teacher.make_collector(
            models[0],
            np.array(padded[0]["target"]),
            len(worlds),
            1,
            0,
            native_models=pool.native_models,
            capacity_models=models,
        )
        collector.engine.model = pool.model
        collector.engine.native_models = pool.native_models
        pool.model.block_dim.linesearch_iterative = 128
        controller = tracking.PathControl(collector)
        collector.controller = controller
        collector.reset(qpos=qpos, qvel=np.zeros((len(worlds), models[0].nv)))
        began = time.perf_counter()
        trace, _ = traces.run_controls(controller, steps, mode="home")
        simulated = time.perf_counter() - began
    results = collections.defaultdict(list)
    for world, (index, number, state) in enumerate(worlds):
        positions, velocities = trace["qpos"][:, world], trace["qvel"][:, world]
        window = {
            "settle_window": (positions[:SETTLE_STEPS], velocities[:SETTLE_STEPS]),
            "full_hold": (positions, velocities),
        }
        result = {
            name: stability.evaluate(
                models[index], padded[index], state, *frames
            )
            for name, frames in window.items()
        }
        result["verdict"] = stability.verdict(
            state["kind"], result["settle_window"], result["full_hold"]
        )
        results[index, number].append(result)
    rows = collections.defaultdict(list)
    for (index, number), replicas in sorted(results.items()):
        state = stability.configurations(padded[index])[number]
        verdict = stability.reproduced([r["verdict"] for r in replicas])
        rows[index].append(
            {k: state[k] for k in ("removed", "last_removed", "kind")}
            | {"passed": verdict["passed"], "verdict": verdict, "replicates": replicas}
        )
    return padded, rows, {"worlds": len(worlds), "simulator_seconds": simulated}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plans", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--max-worlds", type=int, default=512)
    parser.add_argument("--max-plans", type=int, default=48)
    parser.add_argument(
        "--contact-impedance", type=float, nargs=2, default=[0.95, 0.99]
    )
    parser.add_argument(
        "--placement-jitter",
        type=float,
        default=stability.PLACEMENT_JITTER_M,
        help="max horizontal offset of parked parts (m)",
    )
    parser.add_argument(
        "--replicates",
        type=int,
        default=2,
        help="independently placed runs of each state; a failure must recur",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.replicates < 1:
        parser.error("At least one replicate is required")
    if args.seconds < SETTLE_STEPS * 0.02:
        parser.error("Hold at least the 5 s settle window")
    args.output.mkdir(parents=True, exist_ok=False)
    contact_impedance = simulation_models.validate_contact_impedance(args.contact_impedance)
    plans = [json.loads(p.read_text()) for p in args.plans]
    # Campaigns match screen results to plans by content hash.
    hashes = {id(plan): records.sha256(path) for plan, path in zip(plans, args.plans)}
    rng = np.random.default_rng(args.seed)
    summaries, details, batches_run = [], {}, []
    started = time.perf_counter()
    for step, batch in batches(plans, args.max_worlds, args.max_plans, args.replicates):
        padded, rows, stats = screen(
            batch,
            step,
            args.seconds,
            contact_impedance,
            rng,
            args.placement_jitter,
            args.replicates,
        )
        stats.update(physics_timestep=step, plans=len(batch))
        batches_run.append(stats)
        for index, plan in enumerate(padded):
            states = stability.configurations(plan)
            verdicts = [r["verdict"] for r in rows[index]]
            summary = stability.summarize(plan, states, verdicts)
            summary["physics_timestep"] = step
            summary["plan_sha256"] = hashes[id(batch[index])]
            summaries.append(summary)
            details[plan["target_id"]] = rows[index]
            print(json.dumps(summary), flush=True)
    io_timing.write_json(args.output / "states.json", details)
    failures = collections.Counter(
        (s["first_failure_kind"], s["failure_window"], tuple(s["failure"]))
        for s in summaries
        if not s["passed"]
    )
    io_timing.write_json(
        args.output / "summary.json",
        {
            "plans": len(summaries),
            "passed": sum(s["passed"] for s in summaries),
            "failure_classes": {
                f"{k}:{w}:{'+'.join(v)}": n for (k, w, v), n in failures.items()
            },
            "seconds": args.seconds,
            "contact_impedance": args.contact_impedance,
            "placement_jitter_m": args.placement_jitter,
            "replicates": args.replicates,
            "seed": args.seed,
            "elapsed_seconds": time.perf_counter() - started,
            "batches": batches_run,
            "results": summaries,
        },
    )


if __name__ == "__main__":
    main()
