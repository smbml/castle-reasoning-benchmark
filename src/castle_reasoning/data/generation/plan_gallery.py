"""Data generation plan_gallery."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import castle_reasoning.data.generation.geometry as geometry
import castle_reasoning.data.generation.waypoints as waypoints


def assess(gallery, output, staging="stacks"):
    """Export all requested targets' plans or explicit teacher failure reasons."""
    output.mkdir(parents=True, exist_ok=False)
    ids = json.loads((gallery / "selection.json").read_text())["coverage"]
    targets = {
        t["candidate"]: t for t in json.loads((gallery / "targets.json").read_text())
    }
    records = []
    for candidate in ids:
        target = targets[candidate]
        started = time.perf_counter()
        ident = f"T{candidate + 1:04d}"
        parts = [
            geometry.Part(tuple(p["origin"]), tuple(p["size"])) for p in target["parts"]
        ]
        record = {
            "target_id": ident,
            "candidate": candidate,
            "fingerprint": target["fingerprint"],
            "part_count": len(parts),
        }
        try:
            plan = waypoints.plan(
                parts,
                construction_order=target.get("construction_order"),
                staging=staging,
            )
            plan.update(
                target_id=ident,
                seed=target["seed"],
                target_fingerprint=target["fingerprint"],
            )
            (output / f"{ident}.json").write_text(json.dumps(plan, indent=2))
            record.update(planned=True, status="planned", search=plan["search"])
        except waypoints.PlanningFailure as exc:
            record.update(planned=False, status=str(exc))
        record["seconds"] = time.perf_counter() - started
        records.append(record)
        print(json.dumps(record), flush=True)
    summary = {
        "targets": len(records),
        "planned": sum(r["planned"] for r in records),
        "seconds": sum(r["seconds"] for r in records),
        "records": records,
        "source_sha256": hashlib.sha256(
            Path(waypoints.__file__).read_bytes()
        ).hexdigest(),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("gallery", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--staging", choices=("stacks", "flat", "edge"), default="stacks"
    )
    args = parser.parse_args()
    assess(args.gallery, args.output, args.staging)
