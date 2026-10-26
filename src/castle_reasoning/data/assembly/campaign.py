"""Data assembly campaign."""

import bisect
import collections
import json
import math
from pathlib import Path

import castle_reasoning.data.assembly.records as records

DEFAULT_TIMESTEP_S = 0.002


# Collector seeds are `seed + attempt`; runs are spaced so they never overlap.
RUN_SEED_STRIDE = 1_000_000


RUNS_PER_SHARD = 100


# The MJ-Warp engine takes nonnegative signed 32-bit seeds.
MAX_SEED = 2**31 - 1


def describe_plan(path):
    """The facts sharding and indexing need, with the plan's content hash."""
    path = Path(path)
    plan = json.loads(path.read_text())
    limit = plan.get("edge_access", {}).get("max_physics_timestep_s")
    return {
        "path": str(path),
        "sha256": records.sha256(path),
        "target_id": plan["target_id"],
        "fingerprint": plan["target_fingerprint"],
        "parts": len(plan["sizes"]),
        "fixed_parts": list(plan.get("fixed_parts", [])),
        "physics_timestep": DEFAULT_TIMESTEP_S
        if limit is None
        else min(DEFAULT_TIMESTEP_S, limit),
    }


def apply_screen(plans, results):
    """Split described plans by a stability screen's per-plan results.

    Results are matched by plan content hash, so a plan changed after screening
    is never taken as screened. Results may come from several screens; screens
    that disagree on the same plan contents are an error. Returns the passing
    plans and a record of each excluded one.
    """
    by_hash = {}
    for result in results:
        earlier = by_hash.setdefault(result["plan_sha256"], result)
        if earlier["passed"] != result["passed"]:
            raise ValueError(f"Stability screens disagree on {result['target_id']}")
    missing = [p["path"] for p in plans if p["sha256"] not in by_hash]
    if missing:
        raise ValueError(f"Plans absent from the stability screen: {missing[:5]}")
    kept, excluded = [], []
    for plan in plans:
        result = by_hash[plan["sha256"]]
        if result["passed"]:
            kept.append(plan)
            continue
        excluded.append(
            {key: plan[key] for key in ("target_id", "path", "sha256")}
            | {
                key: result.get(key)
                for key in (
                    "first_failure_state",
                    "first_failure_kind",
                    "failure_window",
                    "failure",
                    "missing_supports",
                )
            }
        )
    return kept, excluded


def make_shards(plans, *, max_plans, part_buckets, min_plans=8):
    """Group plans by timestep and part bucket, then split each group.

    Every world is padded to its run's largest part count, so plans of similar
    size share a shard. Within a group, plans are sorted by size and split into
    contiguous, near-equal shards, which keeps padding within each shard small.
    A group smaller than `min_plans` joins the next smaller bucket of its
    timestep, since a run of a few plans leaves most worlds idle.
    """
    if not 1 <= min_plans <= max_plans:
        raise ValueError("Shard size limits must satisfy 1 <= min_plans <= max_plans")
    if list(part_buckets) != sorted(set(part_buckets)):
        raise ValueError("Part bucket edges must be strictly increasing")
    ids = collections.Counter(p["target_id"] for p in plans)
    prints = collections.Counter(p["fingerprint"] for p in plans)
    if duplicate := [k for k, n in ids.items() if n > 1]:
        raise ValueError(
            f"Target ids must be unique within a campaign: {duplicate[:5]}"
        )
    if duplicate := [k for k, n in prints.items() if n > 1]:
        raise ValueError(f"Target geometry appears more than once: {duplicate[:5]}")
    groups = collections.defaultdict(list)
    for plan in plans:
        bucket = bisect.bisect_left(part_buckets, plan["parts"])
        groups[plan["physics_timestep"], bucket].append(plan)
    shards, carried = [], collections.defaultdict(list)
    for (timestep, bucket), members in sorted(groups.items(), reverse=True):
        members = carried.pop(timestep, []) + members
        if len(members) < min_plans and any(
            t == timestep and b < bucket for t, b in groups
        ):
            carried[timestep] = members
            continue
        members = sorted(members, key=lambda p: (-p["parts"], p["target_id"]))
        count = math.ceil(len(members) / max_plans)
        bounds = [round(i * len(members) / count) for i in range(count + 1)]
        for index in range(count):
            chunk = members[bounds[index] : bounds[index + 1]]
            shards.append(
                {
                    "shard_id": f"{timestep * 1e3:g}ms-b{bucket}-{index:03d}",
                    "physics_timestep": timestep,
                    "max_parts": max(p["parts"] for p in chunk),
                    "plans": chunk,
                }
            )
    # Start the costliest shards first so the campaign does not end on them.
    shards.sort(
        key=lambda s: (
            -sum(p["parts"] for p in s["plans"])
            * DEFAULT_TIMESTEP_S
            / s["physics_timestep"],
            s["shard_id"],
        )
    )
    for index, shard in enumerate(shards):
        shard["index"] = index
    return shards


def run_seed(base, shard_index, run_number):
    """Disjoint seed ranges for every collector run in the campaign."""
    if not 0 <= run_number < RUNS_PER_SHARD:
        raise ValueError("Too many runs for one shard")
    return base + (shard_index * RUNS_PER_SHARD + run_number) * RUN_SEED_STRIDE


def topup_groups(shards):
    """One top-up group per physics timestep, holding the plans of its shards.

    A top-up run lasts as long as its slowest attempt whatever its size, so the
    few targets left short in each shard are pooled rather than run shard by
    shard. Group indices follow the shard indices, keeping seeds disjoint.
    """
    groups = {}
    for shard in shards:
        timestep = shard["physics_timestep"]
        group = groups.setdefault(
            timestep,
            {
                "group_id": f"topup-{timestep * 1e3:g}ms",
                "physics_timestep": timestep,
                "shards": [],
                "plans": [],
            },
        )
        group["shards"].append(shard["shard_id"])
        group["plans"].extend(shard["plans"])
    for offset, group in enumerate(groups.values()):
        group["index"] = len(shards) + offset
    return list(groups.values())


def seeds_fit(base, groups):
    """Whether every attempt seed of `groups` shards and top-up groups fits."""
    last = run_seed(base, groups - 1, RUNS_PER_SHARD - 1) + RUN_SEED_STRIDE - 1
    return base >= 0 and last <= MAX_SEED


def read_summary(run_directory):
    path = Path(run_directory) / "summary.json"
    return json.loads(path.read_text()) if path.exists() else None


def accepted_counts(shard, runs):
    """Accepted attempts per target of a shard or top-up group over `runs`,
    finished or not."""
    counts = collections.Counter({p["target_id"]: 0 for p in shard["plans"]})
    for run in runs:
        summary = read_summary(run["output"])
        for record in (summary or {}).get("attempts", []):
            if record.get("accepted") and record["target_id"] in counts:
                counts[record["target_id"]] += 1
    return counts


def deficits(shard, runs, repeats):
    """Targets still short of `repeats` accepted attempts, grouped by shortfall."""
    counts = accepted_counts(shard, runs)
    groups = collections.defaultdict(list)
    for plan in shard["plans"]:
        missing = repeats - counts[plan["target_id"]]
        if missing > 0:
            groups[missing].append(plan)
    return dict(sorted(groups.items(), reverse=True))


def build_index(campaign, runs_by_group):
    """One list of accepted demonstrations with the provenance to load each one.

    `runs_by_group` holds the runs of each shard and top-up group by id.
    """
    root = Path(campaign["root"])
    demonstrations, outcomes = [], collections.Counter()
    per_target = collections.Counter()
    plans = {p["target_id"]: p for s in campaign["shards"] for p in s["plans"]}
    shards = {p["target_id"]: s for s in campaign["shards"] for p in s["plans"]}
    for group_id, runs in runs_by_group.items():
        for run in runs:
            summary = read_summary(run["output"]) or {}
            for record in summary.get("attempts", []):
                if "slot" not in record:
                    continue
                outcomes["accepted" if record["accepted"] else record["status"]] += 1
                if not record["accepted"]:
                    continue
                plan = plans[record["target_id"]]
                shard = shards[plan["target_id"]]
                per_target[plan["target_id"]] += 1
                attempt = Path(run["output"]) / f"attempt{record['attempt']:05d}"
                demonstrations.append(
                    {
                        "target_id": plan["target_id"],
                        "fingerprint": plan["fingerprint"],
                        "parts": plan["parts"],
                        "fixed_parts": plan["fixed_parts"],
                        "physics_timestep": shard["physics_timestep"],
                        "shard_id": shard["shard_id"],
                        "run_group": group_id,
                        "run": run["run"],
                        "seed": run["seed"] + record["attempt"],
                        "attempt_directory": str(attempt.relative_to(root)),
                        "action_targets": record.get("action_targets"),
                    }
                )
    targets = len(plans)
    return {
        "targets": targets,
        "requested_per_target": campaign["repeats"],
        "accepted_demonstrations": len(demonstrations),
        "targets_complete": sum(per_target[t] >= campaign["repeats"] for t in plans),
        "targets_with_any": len(per_target),
        "attempt_outcomes": dict(outcomes),
        "demonstrations": demonstrations,
    }
