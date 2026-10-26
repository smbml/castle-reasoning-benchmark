"""Data generation removal_access."""

import dataclasses
import time

import numpy as np

import castle_reasoning.data.generation.waypoints as waypoints
import castle_reasoning.data.plans as plans
import castle_reasoning.runtime.planning.arm as arm
import castle_reasoning.runtime.planning.manipulation as manipulation
import castle_reasoning.runtime.planning.regrasp as planning_regrasp
import castle_reasoning.runtime.simulation.models as models


class OrderFailure(waypoints.PlanningFailure):
    """No complete greedy order; evidence retains rejected pickup attempts."""

    def __init__(self, evidence):
        self.evidence = evidence
        super().__init__(
            f"source_access_blocked:{evidence['target_id']}:"
            f"remaining={evidence['remaining']}"
        )


def slotted(parts, part, standing):
    """Whether standing parts touch two opposite side faces of `part`.

    Assembly reverses the removal order, so removing such a part first means
    later inserting it between two placed neighbours with no clearance. A
    fraction of a millimetre of placement error closes that slot.
    """
    this = parts[part]
    for axis in (0, 1):
        across = [a for a in range(3) if a != axis]
        sides = set()
        for index in standing - {part}:
            other = parts[index]
            if all(
                min(this.end[a], other.end[a]) > max(this.origin[a], other.origin[a])
                for a in across
            ):
                if other.origin[axis] == this.end[axis]:
                    sides.add(1)
                if other.end[axis] == this.origin[axis]:
                    sides.add(-1)
        if len(sides) == 2:
            return True
    return False


def order(plan, planning_seconds=2):
    """Check pickup paths in nominal subsets; do not create simulator episodes.

    The arm starts at home for each geometric query. Removed objects are excluded
    from later queries. Transfers, staging and dynamics require separate checks.
    Removable parts are tried in the plan's order, except that parts held
    between two standing neighbours go last (`slotted`).
    """
    if not np.isfinite(planning_seconds) or planning_seconds <= 0:
        raise ValueError("Pickup planning budget must be finite and positive")
    started = time.perf_counter()
    model, qpos = models.make_model(
        plan["sizes"],
        plan["target"],
        table_bounds=plan["edge_access"]["table_bounds"],
    )
    planner = manipulation.Planner(
        model,
        plan["sizes"],
        planning_seconds=planning_seconds,
        grasp_family="all",
    )
    # A fixed foundation is never removed.
    fixed = set(plan.get("fixed_parts", ()))
    remaining = set(range(len(plan["sizes"]))) - fixed
    parts = plans.grid_parts(plan)
    deadline = started + 2 * planning_seconds * len(remaining)
    geometry_only = plan | {"parking_supports": []}
    sequence, events = [], []
    plates = set(plan.get("edge_access", {}).get("plates", []))
    slot_insertions = []
    while remaining and time.perf_counter() < deadline:
        chosen = None
        candidates = list(plans.removable(geometry_only, remaining))
        candidates.sort(key=lambda part: slotted(parts, part, remaining | fixed))
        for part in candidates:
            if part in plates:
                if len(remaining) != 1:
                    continue
                events.append({"part": part, "result": "edge_plate_static_eligibility"})
                chosen = part
                break
            attempt = time.perf_counter()
            try:
                pickup = planner.plan(qpos, part, None, deadline=deadline)
            except arm.PlanningError as error:
                events.append(
                    {
                        "part": part,
                        "result": str(error),
                        "seconds": time.perf_counter() - attempt,
                    }
                )
                continue
            events.append(
                {
                    "part": part,
                    "result": "source_pickup_path",
                    "seconds": time.perf_counter() - attempt,
                    "grasp": dataclasses.asdict(pickup.grasp),
                    "queries": pickup.query_count,
                }
            )
            chosen = part
            break
        if chosen is None:
            break
        if slotted(parts, chosen, remaining | fixed):
            slot_insertions.append(chosen)
        sequence.append(chosen)
        remaining.remove(chosen)
        # Exclusion applies only to native geometric queries, never physics/data.
        qpos[14 + 7 * chosen : 17 + 7 * chosen] = [-2, 0, -10]
    evidence = {
        "target_id": plan["target_id"],
        "order": sequence,
        "remaining": sorted(remaining),
        "slot_insertions": slot_insertions,
        "events": events,
        "elapsed_seconds": time.perf_counter() - started,
        "planning_seconds_per_candidate": planning_seconds,
        "physics_verified": False,
    }
    if remaining:
        raise OrderFailure(evidence)
    return sequence, evidence


def transfer_check(plan, planning_seconds=4):
    """Plan every complete removal, pickup through seating at its parking slot.

    `order` checks pickups before storage exists; this runs after staging and
    follows the planned removal order on nominal geometry, moving each removed
    part to its assigned parking pose. Edge plates use the separate push/side
    pickup primitive and are only relocated. It certifies geometry and
    kinematics, not physics.
    """
    if not np.isfinite(planning_seconds) or planning_seconds <= 0:
        raise ValueError("Transfer planning budget must be finite and positive")
    started = time.perf_counter()
    model, qpos = models.make_model(
        plan["sizes"],
        plan["target"],
        table_bounds=plan["edge_access"]["table_bounds"],
    )
    planner = manipulation.Planner(
        model,
        plan["sizes"],
        planning_seconds=planning_seconds,
        grasp_family="all",
    )
    plates = set(plan.get("edge_access", {}).get("plates", []))
    events = []
    for part in plan["removal_order"]:
        destination = np.asarray(plan["parking"][part], dtype=float)
        if part not in plates:
            attempt = time.perf_counter()
            try:
                result = planner.plan(
                    qpos, part, destination, deadline=attempt + planning_seconds
                )
                event = {"result": "complete_transfer", "grasp": result.grasp}
            except arm.PlanningError as error:
                # The executor falls back to a supported regrasp; so does the check.
                try:
                    regrasp = planning_regrasp.plan(
                        planner,
                        qpos,
                        part,
                        destination,
                        deadline=time.perf_counter() + 2 * planning_seconds,
                    )
                except arm.PlanningError as fallback:
                    reason = f"{error}; regrasp: {fallback}"
                    events.append({"part": part, "result": reason})
                    raise TransferFailure(
                        plan["target_id"], part, reason, events, started
                    ) from fallback
                event = {
                    "result": "supported_regrasp_transfer",
                    "grasp": regrasp.second.grasp,
                    "shift_m": float(
                        np.linalg.norm(
                            regrasp.waypoint[:2] - qpos[14 + 7 * part : 16 + 7 * part]
                        )
                    ),
                }
            events.append(
                {
                    "part": part,
                    **event,
                    "grasp": dataclasses.asdict(event["grasp"]),
                    "seconds": time.perf_counter() - attempt,
                }
            )
        qpos[14 + 7 * part : 17 + 7 * part] = destination
    return {
        "events": events,
        "elapsed_seconds": time.perf_counter() - started,
        "planning_seconds_per_part": planning_seconds,
        "physics_verified": False,
    }


class TransferFailure(waypoints.PlanningFailure):
    """A planned removal has no complete pickup-to-parking manipulation."""

    def __init__(self, target_id, part, reason, events, started):
        self.evidence = {
            "target_id": target_id,
            "part": part,
            "events": events,
            "elapsed_seconds": time.perf_counter() - started,
        }
        super().__init__(f"transfer_blocked:{target_id}:part={part}:{reason}")
