"""Data generation prepare."""

import argparse
import collections
import json
import random
import time
from pathlib import Path

import numpy as np

import castle_reasoning.data.assembly.records as records
import castle_reasoning.data.generation.generator as generator
import castle_reasoning.data.generation.geometry as geometry
import castle_reasoning.data.generation.removal_access as removal_access
import castle_reasoning.data.generation.render as render
import castle_reasoning.data.generation.restage as restage
import castle_reasoning.data.generation.waypoints as waypoints
from castle_reasoning.runtime.diagnostics import io_timing


def prepare(parts, target_id, seed, fingerprint):
    """Prepare the full task; as a last resort keep unmovable plates as a foundation.

    Ground parts wider than the gripper can only move by edge staging. When no
    workcell yaw allows that, they stay at their goal (`fixed_parts`) and the
    robot disassembles and rebuilds every other part on them. The plan records
    the fallback and the failed full-task attempts.
    """
    try:
        return _prepare_any_yaw(parts, target_id, seed, fingerprint)
    except waypoints.PlanningFailure as error:
        full_task = error
    plates = [
        i
        for i, p in enumerate(parts)
        if p.origin[2] == 0 and min(p.size[:2]) * 0.01 > 0.07 + 1e-9
    ]
    if not plates or len(plates) == len(parts):
        raise full_task
    try:
        plan = _prepare_any_yaw(parts, target_id, seed, fingerprint, fixed=plates)
    except waypoints.PlanningFailure as error:
        error.evidence = {"full_task": getattr(full_task, "evidence", None),
                          "fixed_foundation": getattr(error, "evidence", None)}
        raise
    plan["foundation_fallback"] = {
        "fixed_parts": plates,
        "full_task_failure": str(full_task),
    }
    return plan


def _prepare_any_yaw(parts, target_id, seed, fingerprint, fixed=()):
    """Prepare the target in the first workcell yaw (0, 90, 180, 270 deg) that plans.

    The generator emits axis-aligned geometry whose yaw in the workcell is
    arbitrary. Staging, pickup access and the reachable envelope are not
    rotation invariant (e.g. an edge plate must be deep along the table's Y
    axis), so a target that fails in its sampled yaw may plan after a rigid
    quarter turn. The sampled yaw is always tried first and, when it passes the
    complete-transfer check, its plan is unchanged. Every failed yaw is recorded.
    """
    failed = []
    for turns in range(4):
        oriented = parts if turns == 0 else geometry.quarter_turn(parts, turns)
        try:
            plan = _prepare_oriented(oriented, target_id, seed, fingerprint, fixed)
            # Every plan must show each planned removal is a complete
            # pickup-to-parking manipulation. Pickup access alone admitted plans
            # whose removals only fit within the planning clearance on paper and
            # dead-ended on the GPU after hours of collection.
            transfer = removal_access.transfer_check(plan)
        except waypoints.PlanningFailure as error:
            failed.append(
                {
                    "quarter_turns": turns,
                    "reason": str(error),
                    "evidence": getattr(error, "evidence", None),
                }
            )
            continue
        plan["transfer_check"] = transfer
        if turns or fixed:
            plan["workcell_orientation"] = {
                "quarter_turns_ccw": turns,
                "failed_yaws": failed,
            }
        return plan
    failure = waypoints.PlanningFailure(
        "orientation_search_exhausted:"
        + "; ".join(f"{r['quarter_turns'] * 90}deg={r['reason']}" for r in failed)
    )
    failure.evidence = failed
    raise failure


def _prepare_oriented(parts, target_id, seed, fingerprint, fixed=()):
    """Apply the reference height search, pickup order, and nested staging."""
    rejected = []
    first_height = max(p.size[2] for p in parts)
    for centimeters in range(first_height, 26):
        try:
            plan = waypoints.plan(
                parts,
                meters_per_grid=0.01,
                max_grasp_width=0.07,
                parking_bounds=(0.20, 0.76, -0.22, 0.37),
                max_search_nodes=512,
                staging="edge",
                max_stack_height=centimeters / 100,
                parking_gap=0.055,
                fixed=fixed,
            )
            break
        except waypoints.PlanningFailure as error:
            rejected.append({"height_m": centimeters / 100, "reason": str(error)})
    else:
        raise waypoints.PlanningFailure(f"height_search_exhausted:{rejected}")
    plan.update(target_id=target_id, seed=seed, target_fingerprint=fingerprint)
    plan["staging_height_selection"] = {
        "start_height_m": first_height / 100,
        "selected_height_m": plan["max_stack_height"],
        "rejected_lower_heights": rejected,
    }
    plan = restage.restage(
        plan, source_access_order=True, pickup_planning_seconds=2
    )
    np.testing.assert_array_equal(
        plan["assembly_goals"], plan["disassembly_goals"][::-1]
    )
    staged = set()
    parents = dict(plan["parking_supports"])
    for moved in plan["removal_order"]:
        assert parents[moved] == -1 or parents[moved] in staged
        staged.add(moved)
    return plan


def check_reference(reference):
    """Reproduce an original metric plan before preparing the fresh sample."""
    saved = json.loads(reference.read_text())
    sizes = np.array(saved["sizes"])
    origins = np.array(saved["target"]) - sizes / 2
    origins[:, :2] -= origins[:, :2].min(axis=0)
    parts = [
        geometry.Part(tuple(np.rint(p / 0.01).astype(int)),
                      tuple(np.rint(s / 0.01).astype(int)))
        for p, s in zip(origins, sizes, strict=True)
    ]
    plan = prepare(parts, saved["target_id"], saved["seed"],
                   saved["target_fingerprint"])
    fields = ("sizes", "target", "parking", "parking_supports", "removal_order",
              "assembly_order", "assembly_goals", "disassembly_goals")
    for field in fields:
        np.testing.assert_allclose(plan[field], saved[field], atol=1e-12)
    return {"reference": str(reference), "sha256": records.sha256(reference),
            "fields_matched": list(fields)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-launch", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260924)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    selection_path = args.output / "selection.json"
    if selection_path.exists():
        raise FileExistsError("Existing sample must not be silently replaced")
    started = time.perf_counter()
    launch = json.loads(args.reference_launch.read_text())
    excluded = set()
    for path, digest in launch["plans"].items():
        assert records.sha256(Path(path)) == digest, path
        excluded.add(json.loads(Path(path).read_text())["target_fingerprint"])
    pool = generator.generate_pool(args.seed, 240)
    io_timing.write_json(args.output / "proposal_pool.json", pool)
    eligible = [t for t in pool["targets"] if t["fingerprint"] not in excluded]
    selected = random.Random(args.seed).sample(eligible, 20)
    for index, target in enumerate(selected, 1):
        target["target_id"] = f"N{index:04d}"
    fingerprints = {t["fingerprint"] for t in selected}
    assert len(fingerprints) == 20 and not fingerprints & excluded
    selection = {
        "seed": args.seed,
        "sampling": "uniform without replacement from accepted unique proposals",
        "fixed_before_robot_planning": True,
        "proposal_count": len(pool["attempts"]),
        "geometry_accepted": len(pool["targets"]),
        "geometry_rejections": dict(collections.Counter(
            t["rejection"] for t in pool["attempts"] if not t["accepted"]
        )),
        "excluded_fingerprints": sorted(excluded),
        "geometry_generation_seconds": time.perf_counter() - started,
        "targets": selected,
    }
    io_timing.write_json(selection_path, selection)
    (args.output / "plans").mkdir()
    (args.output / "images").mkdir()
    summary = {"selected_count": 20, "records": [], "state": "preparing"}
    io_timing.write_json(args.output / "preparation.json", summary)
    reference = next(Path(p) for p in launch["plans"] if Path(p).stem == "T0001")
    summary["reference_parity"] = check_reference(reference)
    for target in selected:
        began = time.perf_counter()
        parts = [geometry.Part(tuple(p["origin"]), tuple(p["size"]))
                 for p in target["parts"]]
        target_id = target["target_id"]
        record = {"target_id": target_id, "parts": len(parts),
                  "fingerprint": target["fingerprint"],
                  "root": target["operations"][0]["operation"],
                  "candidate": target["candidate"]}
        try:
            plan = prepare(parts, target_id, args.seed, target["fingerprint"])
        except waypoints.PlanningFailure as error:
            record.update(status="preparation_failed", reason=str(error),
                          evidence=getattr(error, "evidence", None))
        else:
            path = args.output / "plans" / f"{target_id}.json"
            io_timing.write_json(path, plan)
            record.update(status="planned", plan=str(path),
                          plan_sha256=records.sha256(path))
            turns = plan.get("workcell_orientation", {}).get("quarter_turns_ccw", 0)
            parts = geometry.quarter_turn(parts, turns) if turns else parts
        # Render the target in the workcell yaw its saved plan uses.
        render.render(parts, args.output / "images" / f"{target_id}.png",
                      width=600, height=440)
        record["preparation_seconds"] = time.perf_counter() - began
        summary["records"].append(record)
        summary["elapsed_seconds"] = time.perf_counter() - started
        io_timing.write_json(args.output / "preparation.json", summary)
        print(json.dumps(record), flush=True)
    summary.update(state="finished", planned=sum(
        r["status"] == "planned" for r in summary["records"]
    ))
    io_timing.write_json(args.output / "preparation.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}), flush=True)


if __name__ == "__main__":
    main()
