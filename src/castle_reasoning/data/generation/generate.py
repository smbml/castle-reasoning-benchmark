"""Data generation generate."""


import argparse
import collections
import json
import multiprocessing as mp
import os
import time
import traceback
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


os.environ.setdefault("OMP_NUM_THREADS", "1")


def _prepare(job):
    from castle_reasoning.data.generation import geometry, prepare, waypoints
    from castle_reasoning.runtime.diagnostics import io_timing

    target, seed, out = job
    parts = [
        geometry.Part(tuple(p["origin"]), tuple(p["size"])) for p in target["parts"]
    ]
    began = time.perf_counter()
    record = {
        "target_id": target["target_id"],
        "candidate": target["candidate"],
        "parts": len(parts),
        "root": target["operations"][0]["operation"],
        "fingerprint": target["fingerprint"],
        "descriptors": target["descriptors"],
    }
    try:
        plan = prepare.prepare(parts, target["target_id"], seed, target["fingerprint"])
    except waypoints.PlanningFailure as error:
        record.update(status="preparation_failed", reason=str(error)[:400])
    except Exception as error:  # noqa: BLE001 - a crash is itself a finding
        record.update(
            status="crashed",
            reason=repr(error)[:400],
            traceback=traceback.format_exc()[-2000:],
        )
    else:
        path = Path(out) / "plans" / f"{target['target_id']}.json"
        io_timing.write_json(path, plan)
        edge = plan.get("edge_access") or {}
        record.update(
            status="planned",
            plan=str(path),
            quarter_turns=plan.get("workcell_orientation", {}).get(
                "quarter_turns_ccw", 0
            ),
            fixed_parts=plan.get("foundation_fallback", {}).get("fixed_parts", []),
            max_physics_timestep_s=edge.get("max_physics_timestep_s"),
            active_parts=plan.get("active_part_count", len(parts)),
        )
    record["seconds"] = time.perf_counter() - began
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--proposals", type=int, required=True)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--id-prefix", default="S")
    parser.add_argument("--exclude", type=Path, nargs="*", default=[])
    args = parser.parse_args()
    from castle_reasoning.data.generation import generator
    from castle_reasoning.runtime.diagnostics import io_timing

    args.output.mkdir(parents=True)
    (args.output / "plans").mkdir()
    started = time.perf_counter()
    pool = generator.generate_pool(args.seed, args.proposals)
    generation_seconds = time.perf_counter() - started
    excluded = set()
    for path in args.exclude:
        excluded |= {t["fingerprint"] for t in json.loads(path.read_text())["targets"]}
    targets = [t for t in pool["targets"] if t["fingerprint"] not in excluded]
    for index, target in enumerate(targets, 1):
        target["target_id"] = f"{args.id_prefix}{index:04d}"
    io_timing.write_json(args.output / "pool.json", pool)
    summary = {
        "seed": args.seed,
        "proposals": args.proposals,
        "geometry_accepted": len(pool["targets"]),
        "excluded_overlap": len(pool["targets"]) - len(targets),
        "id_prefix": args.id_prefix,
        "exclude": [str(p) for p in args.exclude],
        "rejections": dict(
            collections.Counter(
                a["rejection"] for a in pool["attempts"] if not a["accepted"]
            )
        ),
        "generation_seconds": generation_seconds,
        "records": [],
    }
    jobs = [(t, args.seed, str(args.output)) for t in targets]
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, maxtasksperchild=8) as workers:
        for record in workers.imap_unordered(_prepare, jobs):
            summary["records"].append(record)
            summary["elapsed_seconds"] = time.perf_counter() - started
            print(
                json.dumps(
                    {
                        k: record.get(k)
                        for k in (
                            "target_id",
                            "parts",
                            "status",
                            "seconds",
                            "quarter_turns",
                            "fixed_parts",
                            "reason",
                        )
                    }
                ),
                flush=True,
            )
            if len(summary["records"]) % 25 == 0:
                io_timing.write_json(args.output / "preparation.json", summary)
    summary["records"].sort(key=lambda r: r["target_id"])
    summary["state"] = "finished"
    io_timing.write_json(args.output / "preparation.json", summary)


if __name__ == "__main__":
    main()
