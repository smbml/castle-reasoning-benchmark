"""Data generation restage."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

import castle_reasoning.data.generation.edge_staging as edge_staging
import castle_reasoning.data.generation.removal_access as removal_access


def restage(plan, *, source_access_order=False, pickup_planning_seconds=2):
    """Rebuild storage, optionally checking nominal pickup access before packing."""
    if plan["staging"] != "edge":
        raise ValueError("Restaging requires an edge-access construction plan")
    if source_access_order:
        order, evidence = removal_access.order(plan, pickup_planning_seconds)
        plan = plan | {
            "removal_order": order,
            "order_source": "nominal_pickup_access",
            "source_access": evidence,
            "search": {
                "method": "nominal_pickup_access",
                "elapsed_seconds": evidence["elapsed_seconds"],
            },
        }
    goals, sizes = np.array(plan["target"]), np.array(plan["sizes"])
    order = plan["removal_order"]
    parked, supports, edge = edge_staging.arrange(
        goals,
        sizes,
        order,
        plan["max_grasp_width"],
        plan["max_stack_height"],
        gap=plan["parking_gap"],
        stack_layout="nested",
        fixed=plan.get("fixed_parts", ()),
    )
    for part in plan.get("fixed_parts", ()):
        parked[part] = goals[part]
        supports.append([part, -1])
    if edge["table_bounds"] != plan["edge_access"]["table_bounds"]:
        raise ValueError("Restaging must preserve the existing table bounds")
    states, transports = [goals.copy()], []
    hover = (
        max(
            np.max(goals[:, 2] + sizes[:, 2] / 2),
            np.max(parked[:, 2] + sizes[:, 2] / 2),
        )
        + 0.08
    )
    for part in order:
        source, destination = states[-1][part].copy(), parked[part].copy()
        high_source, high_destination = source.copy(), destination.copy()
        high_source[2] = high_destination[2] = hover + sizes[part, 2] / 2
        transports.append(
            {
                "part": part,
                "centers": np.stack(
                    [source, high_source, high_destination, destination]
                ).tolist(),
            }
        )
        state = states[-1].copy()
        state[part] = destination
        states.append(state)
    return plan | {
        "stack_layout": "nested",
        "parking": parked.tolist(),
        "parking_supports": supports,
        "edge_access": edge,
        "parking_bounds": edge["parking_bounds"],
        "assembly_order": order[::-1],
        "disassembly_goals": np.asarray(states).tolist(),
        "assembly_goals": np.asarray(states[::-1]).tolist(),
        "object_transport_waypoints": transports,
        "robot_execution": "unverified",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plans", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-access-order", action="store_true")
    parser.add_argument("--pickup-planning-seconds", type=float, default=2)
    args = parser.parse_args()
    if len({p.name for p in args.plans}) != len(args.plans):
        raise ValueError("Source plan names must be distinct")
    results = []
    for source in args.plans:
        raw = source.read_bytes()
        try:
            plan = restage(
                json.loads(raw),
                source_access_order=args.source_access_order,
                pickup_planning_seconds=args.pickup_planning_seconds,
            )
        except ValueError as error:
            if hasattr(error, "evidence"):
                print(json.dumps(error.evidence), file=sys.stderr)
            raise
        plan["staging_source"] = {
            "path": str(source.resolve()),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        results.append((source.name, plan))
    # Complete every plan before creating the cohort; never silently drop targets.
    args.output.mkdir(parents=True, exist_ok=False)
    for name, plan in results:
        (args.output / name).write_text(json.dumps(plan, indent=2) + "\n")
    print(f"Wrote {len(results)} plans; geometry only, robot execution unverified")


if __name__ == "__main__":
    main()
