"""Data assembly run_campaign."""

import argparse
import concurrent.futures
import json
import math
import subprocess
import sys
import threading
import time
from pathlib import Path

import castle_reasoning.data.assembly.campaign as campaign
import castle_reasoning.data.assembly.records as records
from castle_reasoning.runtime.diagnostics import io_timing

QUALIFIED_FLAGS = [
    "--friction-cone",
    "elliptic",
    "--solver",
    "cg",
    "--solver-iterations",
    "1000",
    "--line-search-threads",
    "128",
    "--contact-impedance",
    "0.95",
    "0.99",
    "--replay-settle-seconds",
    "10",
    "--pause-planning-worlds",
    "--skip-paused-preparation",
    "--preparation-mode",
    "producer",
    "--preparation-workers",
    "4",
    "--queue-order",
    "longest_first",
    "--planner-start-method",
    "forkserver",
    "--max-rollbacks",
    "3",
]


def parse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plans", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--repeats",
        type=int,
        default=2,
        help="accepted demonstrations requested per target",
    )
    parser.add_argument(
        "--max-plans-per-shard",
        type=int,
        default=192,
        help="each plan holds a ~63 MB compiled model in host memory",
    )
    parser.add_argument(
        "--min-plans-per-shard",
        type=int,
        help="smaller part buckets join the next smaller bucket "
        "(default: enough plans to fill --worlds)",
    )
    parser.add_argument(
        "--part-buckets",
        type=int,
        nargs="*",
        default=[12, 18, 26],
        help="upper part-count edges separating shards",
    )
    parser.add_argument(
        "--worlds", type=int, default=256, help="most worlds per collector run"
    )
    parser.add_argument("--concurrent-shards", type=int, default=1)
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=3,
        help="first round plus top-up rounds",
    )
    parser.add_argument(
        "--wall-seconds",
        type=float,
        default=6 * 3600,
        help="wall cap of each collector run",
    )
    parser.add_argument(
        "--screen",
        type=Path,
        nargs="+",
        help="stability screen summary.json files; plans that failed are excluded",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    argv = sys.argv[1:]
    split = argv.index("--") if "--" in argv else len(argv)
    args = parser.parse_args(argv[:split])
    args.collector_flags = argv[split + 1 :]
    if args.repeats * args.max_rounds >= campaign.RUNS_PER_SHARD:
        parser.error("repeats x max rounds must stay below the per-shard run limit")
    return args


def manifest(args):
    plans = [campaign.describe_plan(path.resolve()) for path in args.plans]
    screen = None
    if args.screen:
        results = [
            result
            for path in args.screen
            for result in json.loads(path.read_text())["results"]
        ]
        plans, excluded = campaign.apply_screen(plans, results)
        screen = {
            "summaries": [
                {"path": str(path.resolve()), "sha256": records.sha256(path)}
                for path in args.screen
            ],
            "excluded": excluded,
        }
    min_plans = args.min_plans_per_shard or min(
        args.max_plans_per_shard, math.ceil(args.worlds / args.repeats)
    )
    shards = campaign.make_shards(
        plans,
        max_plans=args.max_plans_per_shard,
        min_plans=min_plans,
        part_buckets=args.part_buckets,
    )
    return {
        "root": str(args.output.resolve()),
        "repeats": args.repeats,
        "max_plans_per_shard": args.max_plans_per_shard,
        "min_plans_per_shard": min_plans,
        "part_buckets": args.part_buckets,
        "worlds": args.worlds,
        "max_rounds": args.max_rounds,
        "wall_seconds": args.wall_seconds,
        "seed": args.seed,
        "collector_flags": args.collector_flags or QUALIFIED_FLAGS,
        "screen": screen,
        "shards": shards,
    }


# Statuses of runs whose collector ended by itself; an interrupted round runs again.
FINISHED = ("completed", "failed")


class Campaign:
    def __init__(self, spec):
        self.spec = spec
        self.root = Path(spec["root"])
        self.lock = threading.Lock()
        self.runs = {}
        self.topups = campaign.topup_groups(spec["shards"])

    def publish(self):
        """Called with the lock held after any run changes state."""
        index = campaign.build_index(self.spec, self.runs)
        io_timing.write_json(self.root / "index.json", index)
        return index

    def place(self, group):
        """Id and directory of a shard or a top-up group."""
        if "shard_id" in group:
            return group["shard_id"], self.root / "shards" / group["shard_id"]
        return group["group_id"], self.root / "topups" / group["group_id"]

    def record(self, group, runs):
        key, directory = self.place(group)
        with self.lock:
            self.runs[key] = runs
            io_timing.write_json(directory / "runs.json", runs)
            return self.publish()

    def load(self, group):
        directory = self.place(group)[1]
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "runs.json"
        runs = json.loads(path.read_text()) if path.exists() else []
        for run in runs:
            if run["status"] == "running":
                # The collector may have finished after the driver stopped.
                summary = campaign.read_summary(run["output"]) or {}
                run["status"] = (
                    "completed" if summary.get("completed") else "interrupted"
                )
        self.record(group, runs)
        return runs

    def shard(self, shard):
        """The first round: every target's requested repeats in one run.

        A first round the driver was stopped in is run again for the targets
        it left short.
        """
        runs = self.load(shard)
        if not any(r["status"] in FINISHED for r in runs):
            short = campaign.deficits(shard, runs, self.spec["repeats"])
            if short:
                plans = [p for members in short.values() for p in members]
                self.collect(shard, runs, 0, max(short), plans)
        return shard["shard_id"]

    def topup(self, group):
        """Later rounds: one run per round for the group's targets still short.

        Every target in the run gets the largest shortfall's attempts; the run
        lasts as long as its slowest attempt, so extra attempts cost little
        wall time, and they can add demonstrations beyond the requested repeats.
        """
        runs = self.load(group)
        with self.lock:
            first = [r for s in group["shards"] for r in self.runs.get(s, [])]
        done = [r["round"] for r in runs if r["status"] in FINISHED]
        start = max(done, default=0) + 1
        for round_number in range(start, self.spec["max_rounds"]):
            short = campaign.deficits(group, first + runs, self.spec["repeats"])
            if not short:
                break
            plans = [p for members in short.values() for p in members]
            self.collect(group, runs, round_number, max(short), plans)
        return group["group_id"]

    def collect(self, group, runs, round_number, attempts, plans):
        runs.append(self.launch(group, len(runs), round_number, attempts, plans))
        self.record(group, runs)
        self.finish(runs[-1])
        index = self.record(group, runs)
        print(
            json.dumps(
                {
                    "group": self.place(group)[0],
                    "run": runs[-1]["run"],
                    "status": runs[-1]["status"],
                    "accepted_demonstrations": index["accepted_demonstrations"],
                    "targets_complete": index["targets_complete"],
                }
            ),
            flush=True,
        )

    def launch(self, group, number, round_number, attempts, plans):
        directory = self.place(group)[1]
        seed = campaign.run_seed(self.spec["seed"], group["index"], number)
        worlds = min(self.spec["worlds"], attempts * len(plans))
        output = directory / f"run{number:02d}"
        command = [
            sys.executable,
            "-m",
            'castle_reasoning.data.assembly.continuous',
            "--plans",
            *(p["path"] for p in plans),
            "--output",
            str(output),
            "--worlds",
            str(worlds),
            "--attempts-per-target",
            str(attempts),
            "--physics-timestep",
            str(group["physics_timestep"]),
            "--wall-seconds",
            str(self.spec["wall_seconds"]),
            "--seed",
            str(seed),
            *self.spec["collector_flags"],
        ]
        log = directory / f"run{number:02d}.log"
        run = {
            "run": number,
            "round": round_number,
            "seed": seed,
            "attempts_per_target": attempts,
            "worlds": worlds,
            "targets": [p["target_id"] for p in plans],
            "output": str(output),
            "log": str(log),
            "command": command,
            "status": "running",
            "started": time.time(),
        }
        return run

    def finish(self, run):
        with Path(run["log"]).open("w") as stream:
            code = subprocess.run(
                run["command"], stdout=stream, stderr=subprocess.STDOUT, check=False
            ).returncode
        summary = campaign.read_summary(run["output"]) or {}
        run.update(
            returncode=code,
            finished=time.time(),
            status="completed" if code == 0 and summary.get("completed") else "failed",
        )


def main():
    args = parse()
    spec = manifest(args)
    groups = len(spec["shards"]) + len(campaign.topup_groups(spec["shards"]))
    if not campaign.seeds_fit(spec["seed"], groups):
        raise SystemExit(
            f"--seed {spec['seed']} leaves attempt seeds of {groups} shards and "
            f"top-up groups above {campaign.MAX_SEED}"
        )
    for shard in spec["shards"]:
        print(
            f"{shard['shard_id']}: {len(shard['plans'])} plans, "
            f"max {shard['max_parts']} parts, {shard['physics_timestep']} s"
        )
    if args.dry_run:
        return
    path = args.output / "campaign.json"
    if path.exists():
        if json.loads(path.read_text()) != json.loads(json.dumps(spec)):
            raise SystemExit(
                f"{path} differs from this command; refusing to mix campaigns"
            )
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        io_timing.write_json(path, spec)
    runner = Campaign(spec)
    # Every shard's first round runs before any top-up, so the GPU stays full
    # until only the few short targets are left.
    with concurrent.futures.ThreadPoolExecutor(args.concurrent_shards) as pool:
        for work, groups in (
            (runner.shard, spec["shards"]),
            (runner.topup, runner.topups),
        ):
            for future in concurrent.futures.as_completed(
                [pool.submit(work, group) for group in groups]
            ):
                future.result()
    with runner.lock:
        index = runner.publish()
    print(
        json.dumps({k: v for k, v in index.items() if k != "demonstrations"}),
        flush=True,
    )


if __name__ == "__main__":
    main()
