"""Data verification stability."""

import numpy as np

import castle_reasoning.runtime.execution.success as success
import castle_reasoning.runtime.simulation.models as models

SCENE_POSITION_TOLERANCE_M = 0.005


SCENE_ORIENTATION_TOLERANCE_DEG = 5


# Horizontal placement error given to parked parts; see `placed`.
PLACEMENT_JITTER_M = 0.0005


def configurations(plan):
    """The target and every state along the planned removal order.

    State k has the first k removed parts at their parking poses and every other
    part at its goal. State 0 is what the settle stage holds; the last state,
    with every movable part parked, is what assembly starts from.
    """
    fixed = set(plan.get("fixed_parts", ()))
    order = [p for p in plan["removal_order"] if p not in fixed]
    target = np.asarray(plan["target"], dtype=float)
    parking = np.asarray(plan["parking"], dtype=float)
    states = []
    for count in range(len(order) + 1):
        removed = set(order[:count])
        goals = target.copy()
        goals[sorted(removed)] = parking[sorted(removed)]
        supports = [
            s
            for s in plan["supports"]
            if s[0] not in removed and (s[1] == -1 or s[1] not in removed)
        ] + [
            s
            for s in plan["parking_supports"]
            if s[0] in removed and (s[1] == -1 or s[1] in removed)
        ]
        kind = (
            "target" if count == 0 else "parked" if count == len(order) else "partial"
        )
        states.append(
            {
                "removed": count,
                "last_removed": order[count - 1] if count else None,
                "kind": kind,
                "parked": sorted(removed),
                "goals": goals,
                "supports": supports,
            }
        )
    return states


def placed(state, rng, jitter=PLACEMENT_JITTER_M):
    """The state with each parked part offset horizontally by up to `jitter`.

    A stacked part whose edges coincide exactly with its support's is a
    degenerate case for box-box contact: identical copies of such a parked
    layout occasionally eject a part (about 5% of copies within 5 s, on both
    the stock and the active-world runtime), while a 0.5 mm offset removes it.
    The robot never parks a part exactly, so parked parts get placement error.
    Standing parts stay exact: they touch side by side, and the collector also
    settles the target from exact poses.
    """
    goals = np.array(state["goals"], dtype=float)
    parked = state["parked"]
    goals[parked, :2] += rng.uniform(-jitter, jitter, (len(parked), 2))
    return dict(state, goals=goals)


def initial_qpos(home, goals):
    """The compiled home state with every part placed upright at its goal."""
    qpos = np.array(home, dtype=float)
    poses = qpos[14:].reshape(-1, 7)
    poses[: len(goals), :3] = goals
    poses[: len(goals), 3:] = [1, 0, 0, 0]
    return qpos


def evaluate(model, plan, state, qpos, qvel):
    """Apply the collector's checks to one simulated hold window (>= 51 frames)."""
    active = len(models.active_parts(plan))
    goals = np.asarray(plan["target"], dtype=float).copy()
    goals[:active] = state["goals"][:active]
    hold = success.verify_hold(qpos, qvel, goals, np.asarray(plan["sizes"]))
    result = {
        k: hold[k]
        for k in (
            "position_error_m",
            "orientation_error_deg",
            "drift_m",
            "surface_speed_m_s",
            "settled",
        )
    }
    if state["kind"] == "partial":
        # During disassembly the collector checks only scene poses after each
        # move; supports and motion are audited at the ends of stages.
        result["passed"] = bool(
            hold["position_error_m"] <= SCENE_POSITION_TOLERANCE_M
            and hold["orientation_error_deg"] <= SCENE_ORIENTATION_TOLERANCE_DEG
        )
        return result
    contacts = success.contact_report(model, qpos, state["supports"])
    result["contacts"] = contacts
    result["passed"] = bool(hold["passed"] and contacts["passed"])
    return result


def verdict(kind, settle, hold):
    """Combine the settle-window and whole-hold evaluations of one state.

    The settle window decides, as in the collector; the target may instead pass
    on the whole hold when only residual motion failed, since the settle stage
    may wait that out (never missing contacts). The rest of the hold looks for
    creep, which the collector meets later in the stage: every declared
    support must still touch (target S0086 passed its window by 0.5 um and
    failed collection), and a partial structure must still be within the
    scene limits and at rest. Momentary motion inside the settle window of a
    partial structure is ignored; the collector never checks it there.
    Returns the verdict and the window that decided a failure.
    """
    passed = settle["passed"] or (
        kind == "target"
        and hold["passed"]
        and settle.get("contacts", {}).get("passed", True)
    )
    if not passed:
        return {**settle, "passed": False, "window": "settle"}
    if kind == "partial":
        held = hold["passed"] and hold["settled"]
    else:
        held = hold.get("contacts", {}).get("passed", True)
    if not held:
        return {**hold, "passed": False, "window": "hold"}
    return {**settle, "passed": True}


def reproduced(verdicts):
    """One state's verdict over replicates: it fails only if every one failed.

    Runs of the same state differ: GPU contact order is not deterministic, and
    support gaps near the 10 um tolerance, brief chatter and rare degenerate
    contacts come and go between runs. A failure every replicate shows belongs
    to the plan; one that comes and goes is left to the collector, whose
    settle check is cheap and whose campaign retries failed targets.
    """
    failed = sum(not v["passed"] for v in verdicts)
    chosen = next((v for v in verdicts if v["passed"]), verdicts[0])
    return {**chosen, "failed_replicates": failed}


def summarize(plan, states, results):
    """Per-plan verdict: the first failing state and its class, if any."""
    first = next(
        (i for i, r in enumerate(results) if not r["passed"]),
        None,
    )
    summary = {
        "target_id": plan["target_id"],
        "parts": len(models.active_parts(plan)),
        "states": len(states),
        "passed": first is None,
    }
    if first is not None:
        state, result = states[first], results[first]
        summary.update(
            first_failure_state=first,
            first_failure_kind=state["kind"],
            failure_window=result.get("window", "settle"),
            last_removed=state["last_removed"],
            failure=sorted(
                k
                for k, ok in (
                    (
                        "position",
                        result["position_error_m"] <= SCENE_POSITION_TOLERANCE_M,
                    ),
                    (
                        "orientation",
                        result["orientation_error_deg"]
                        <= SCENE_ORIENTATION_TOLERANCE_DEG,
                    ),
                    ("settling", result["settled"]),
                    ("contacts", result.get("contacts", {}).get("passed", True)),
                )
                if not ok
            ),
            missing_supports=result.get("contacts", {}).get("failures", []),
            max_missing_contact_gap_m=result.get("contacts", {}).get(
                "max_missing_contact_gap_m"
            ),
        )
    return summary
