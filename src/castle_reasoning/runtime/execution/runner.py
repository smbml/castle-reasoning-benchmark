"""Runtime execution runner."""

import collections
import dataclasses
import itertools
import time

import numpy as np

import castle_reasoning.runtime.control.guards as guards
import castle_reasoning.runtime.control.parameters as parameters
import castle_reasoning.runtime.control.tracking as tracking
import castle_reasoning.runtime.diagnostics.timing as timing
import castle_reasoning.runtime.execution.recovery as recovery
import castle_reasoning.runtime.planning.arm as planning_arm
import castle_reasoning.runtime.planning.collision as collision
import castle_reasoning.runtime.planning.manipulation as manipulation
import castle_reasoning.runtime.planning.regrasp as planning_regrasp
import castle_reasoning.runtime.simulation.models as models
import castle_reasoning.runtime.simulation.robot as robot
from castle_reasoning.runtime.diagnostics import io_timing
from castle_reasoning.runtime.execution import success, traces
from castle_reasoning.runtime.planning import plates


class Runner:
    """One world with bounded planning, persistent action trace, and diagnostics."""
    max_rollbacks = 0
    def __init__(
        self,
        collector,
        plan,
        output,
        planning_seconds,
        deadline,
        seed=0,
        shuffle_grasps=True,
    ):
        self.collector, self.plan, self.output = collector, plan, output
        self.controller = collector.controller
        self.planner = manipulation.Planner(
            collector.engine.native_model,
            plan["sizes"],
            planning_seconds,
            seed=seed,
            shuffle_grasps=shuffle_grasps,
            grasp_family="all",
            active_part_count=len(models.active_parts(plan)),
        )
        self.audit_scene = collision.Scene(
            collector.engine.native_model,
            collector.engine.native_model.qpos0,
            margin=-0.0002,
        )
        self.deadline = deadline
        self.order = []
        self.trace = []
        self.events = []
        self.phases = []
        self.boundaries = []
        self.attempts = []
        self.recoveries = []
        self.placements = []
        self.rollbacks = []
        self.simulator_seconds = 0.0
        self.planning_seconds = 0.0
    def state(self):
        state = self.collector.engine.snapshot()
        return state["qpos"][0], state["qvel"][0]
    def budget(self):
        if time.perf_counter() >= self.deadline:
            raise planning_arm.PlanningError("experiment_wall_budget")
    def segment(self, segment, part, reference, guard=True, monitor_scene=True):
        self.budget()
        trace, event = self._execute_segment(
            segment, part, reference, guard, monitor_scene
        )
        self._record_segment(segment, part, trace, event, guard)
    def _execute_segment(self, segment, part, reference, guard, monitor_scene):
        return tracking.execute_segment(
            self.controller,
            segment,
            self.planner.scene,
            part,
            guard,
            reference,
            self.deadline,
            grip_effort=parameters.grip_effort(self.plan, part),
            monitor_scene=monitor_scene,
        )
    @timing.measured('audit.segment')
    def _record_segment(self, segment, part, trace, event, guard=True):
        self.trace.append(trace)
        self.phases.extend([(segment.name, part)] * len(trace["action"]))
        self.simulator_seconds += event["simulator_seconds"]
        event.update(
            name=segment.name,
            part=part,
            start_action=sum(len(t["action"]) for t in self.trace[:-1]),
            pads_allowed=segment.pads_allowed,
            supports=list(segment.supports),
            carrying=segment.payload is not None,
            moving_object=segment.moving_object,
            departures=segment.departures,
            execution_departures=collision.execution_departures(segment.departures),
            arrivals=segment.arrivals,
            grip_effort=parameters.grip_effort(self.plan, part),
        )
        # Audit actual saved states independently of the planning approximation.
        violations = []
        if guard:
            violations = guards.audit_contacts(
                self.audit_scene,
                trace["qpos"][:, 0],
                part,
                segment.pads_allowed,
                segment.supports,
                segment.payload is not None,
                segment.moving_object,
                segment.departures,
                segment.arrivals,
            )
        event["forbidden_contact_frames"] = len(violations)
        event["first_forbidden_contacts"] = violations[:3]
        self.events.append(event)
        print(segment.name, part, event, flush=True)
        if event["failure"] == 5:
            raise planning_arm.PlanningError("experiment_wall_budget")
        if event["failure"] or violations:
            raise recovery.ExecutionGuard(segment, event)
    def _finish_stage(self):
        """Serial execution needs no waiting for sibling worlds."""
    def move(self, result, destination):
        reference, _ = self.state()
        try:
            self._move(result, destination, reference)
        except recovery.ExecutionGuard as error:
            recovery.recover(self, result, destination, reference, error)
    def _move(self, result, destination, reference):
        part = result.part
        for segment in result.segments[:3]:
            self.segment(segment, part, reference)
        # Use the physical post-closure transform, not an ideal attachment.
        qpos, _ = self.state()
        started = time.perf_counter()
        with self.planner.budget.scope(self.planner.planning_seconds, self.deadline):
            payload = self.planner.scene.attachment(qpos, part)
            old_lift = result.segments[3]
            self.planner.scene.set_state(old_lift.path[-1], old_lift.payload)
            goal = self.planner.scene.data.site_xpos[self.planner.scene.site].copy()
            rotation = (
                self.planner.scene.data.site_xmat[self.planner.scene.site]
                .reshape(3, 3)
                .copy()
            )
            lift = self.planner._cartesian(
                "lift",
                qpos,
                goal,
                rotation,
                old_lift.closure[-1],
                part,
                True,
                old_lift.supports,
                payload,
            )
        self.planning_seconds += time.perf_counter() - started
        self.segment(lift, part, reference)
        self.transport(
            part,
            destination,
            old_lift.closure[-1],
            result.segments[0].closure[-1],
            reference,
        )
    def transport(self, part, destination, holding, opening, reference):
        """Transport, then refresh the attachment before contact-sensitive seating."""
        qpos, _ = self.state()
        started = time.perf_counter()
        with self.planner.budget.scope(self.planner.planning_seconds, self.deadline):
            nominal = np.array(destination, dtype=float).copy()
            destination = self.planner.placement_destination(qpos, part, nominal)
            self.placements.append(
                {
                    "part": part,
                    "nominal": nominal.tolist(),
                    "selected": destination.tolist(),
                    "offset_selected": bool(np.any(destination[:2] != nominal[:2])),
                }
            )
            payload = self.planner.scene.attachment(qpos, part)
            rest = self.planner.continuation(
                qpos,
                part,
                destination,
                payload,
                holding,
                opening,
            )
        self.planning_seconds += time.perf_counter() - started
        self.planner.scene.set_state(rest[0].path[-1], payload)
        high = self.planner.scene.data.site_xpos[self.planner.scene.site, 2]
        for segment in rest[:2]:
            self.segment(segment, part, reference)
        qpos, _ = self.state()
        started = time.perf_counter()
        with self.planner.budget.scope(self.planner.planning_seconds, self.deadline):
            payload = self.planner.scene.attachment(qpos, part)
            tail = self.planner.setdown(
                qpos, part, destination, payload, holding, opening, high
            )
        self.planning_seconds += time.perf_counter() - started
        for segment in tail:
            self.segment(segment, part, reference)
    def _find_regrasp(self, choices, goals):
        proposal = None
        budget_error = None
        for part in choices:
            self.budget()
            self.planner.check_budget()
            qpos, _ = self.state()
            started = time.perf_counter()
            try:
                proposal = planning_regrasp.plan(
                    self.planner, qpos, part, goals[part], self.deadline
                )
            except planning_arm.PlanningError as error:
                if isinstance(error, planning_arm.PlanningBudgetError):
                    self.planner.check_budget()
                    budget_error = error
                self.attempts.append(
                    {
                        "part": part,
                        "primitive": "supported_regrasp",
                        "reason": str(error),
                    }
                )
            finally:
                self.planning_seconds += time.perf_counter() - started
            if proposal is not None:
                break
        if proposal is None:
            if budget_error is not None:
                raise budget_error
            raise planning_arm.PlanningError("no_feasible_removal_or_grasp")
        return part, proposal
    def regrasp(self, choices, goals):
        """Try one supported shift only after all direct transfer choices fail."""
        with self.planner.budget.scope(2 * self.planner.planning_seconds, self.deadline):
            part, proposal = self._find_regrasp(choices, goals)
        record = {
            "part": part,
            "primitive": "supported_regrasp",
            "waypoint": proposal.waypoint.tolist(),
            "waypoint_anchor": proposal.anchor,
            "supports": list(proposal.supports),
            "planned_support_margin_m": proposal.support_margin_m,
            "first_grasp": dataclasses.asdict(proposal.first.grasp),
            "predicted_second_grasp": dataclasses.asdict(proposal.second.grasp),
            "outcome": "started",
        }
        self.attempts.append(record)
        try:
            self.move(proposal.first, proposal.waypoint)
            qpos, _ = self.state()
            margin, supports = planning_regrasp.support_margin(
                self.planner, qpos, part
            )
            record["measured_support_margin_m"] = (
                margin if np.isfinite(margin) else None
            )
            if supports != proposal.supports or margin < 0.001:
                raise planning_arm.PlanningError("regrasp_support_margin")
            started = time.perf_counter()
            try:
                second = self.planner.plan(
                    qpos, part, goals[part], deadline=self.deadline
                )
            finally:
                self.planning_seconds += time.perf_counter() - started
            record["actual_second_grasp"] = dataclasses.asdict(second.grasp)
            self.move(second, goals[part])
        except planning_arm.PlanningError as error:
            record["outcome"] = str(error)
            raise
        record["outcome"] = "completed"
        return part
    def home(self):
        qpos, _ = self.state()
        started = time.perf_counter()
        with self.planner.budget.scope(self.planner.planning_seconds, self.deadline):
            # Plan the actual joint home, including opening and the last retreat.
            goal = qpos.copy()
            goal[:6] = robot.HOME
            goal[6:14] = 0
            path = planning_arm.connect(
                qpos[:6],
                goal[:6],
                lambda arm: self.planner.scene.valid(np.r_[arm, qpos[6:]]),
                self.planner.scene.model.jnt_range[:6],
                check_budget=self.planner.check_budget,
            )
            frames = []
            for arm in path:
                frame = qpos.copy()
                frame[:6] = arm
                frames.append(frame)
            frames.append(goal)
            for a, b in itertools.pairwise(frames):
                self.planner._edge(a, b, None, False, (), None)
        self.planning_seconds += time.perf_counter() - started
        closure = np.array([q[6] / 0.8 for q in frames])
        self.segment(manipulation.Segment("home", np.array(frames), closure), -1, qpos)
        qpos, _ = self.state()
        self.segment(manipulation.hold_segment(qpos), -1, qpos)
    def restore(self, qpos, qvel):
        """Return the simulator to a checkpoint state of the current stage."""
        self.collector.reset(qpos=qpos, qvel=qvel)

    def removable(self, remaining):
        """Require the caller's order; data workflows provide automatic choices."""
        raise ValueError("Runtime stages require an explicit part order")

    def _checkpoint(self, completed, remaining, expected):
        """A passed boundary: the state and the lengths of every stage record."""
        qpos, qvel = self.state()
        return {
            "state": (qpos.copy(), qvel.copy()),
            "completed": list(completed),
            "remaining": set(remaining),
            "expected": expected.copy(),
            **{
                key: len(getattr(self, key))
                for key in ("trace", "events", "phases", "boundaries",
                            "recoveries", "placements")
            },
        }
    def _rollback_target(self, error, checkpoints, moving):
        """Checkpoint to resume from after `error`, or None to fail the stage.

        A failure after any execution since the last boundary resumes from that
        boundary. If nothing could be executed, the state itself is a dead end,
        so the previous move is undone instead. Budget exhaustion is final.
        """
        if len(self.rollbacks) >= self.max_rollbacks or isinstance(
            error, planning_arm.PlanningBudgetError
        ):
            return None
        if str(error) in ("experiment_wall_budget", "planning_cpu_budget",
                          "planning_watchdog"):
            return None
        if moving is not None or len(self.trace) > checkpoints[-1]["trace"]:
            return len(checkpoints) - 1
        return len(checkpoints) - 2 if len(checkpoints) > 1 else None
    def _roll_back(self, checkpoints, index, error, moving, avoid):
        """Restore checkpoint `index`, discard everything after it, and record why.

        The discarded actions never reach the saved trajectory: the stage
        continues from the restored state exactly as if the failed branch had
        not been executed, and the independent replay verifies the result.
        """
        checkpoint = checkpoints[index]
        latest = index == len(checkpoints) - 1
        avoided = moving if latest else checkpoints[index + 1]["completed"][-1]
        if avoided is not None:
            avoid[len(checkpoint["completed"])].add(avoided)
        discarded = sum(len(t["action"]) for t in self.trace[checkpoint["trace"]:])
        self.restore(*checkpoint["state"])
        for key in ("trace", "events", "phases", "boundaries", "recoveries",
                    "placements"):
            del getattr(self, key)[checkpoint[key]:]
        del checkpoints[index + 1:]
        self.rollbacks.append(
            {
                "resumed_at_move": len(checkpoint["completed"]),
                "reason": str(error),
                "avoided_part": avoided,
                "discarded_control_steps": discarded,
            }
        )
        print("rollback", self.rollbacks[-1], flush=True)
        return (
            list(checkpoint["completed"]),
            set(checkpoint["remaining"]),
            checkpoint["expected"].copy(),
        )
    @timing.measured('worker.stage_body', 'name')
    def stage(self, name, goals, order=None):
        started = time.perf_counter()
        # Each exported trajectory and its independent replay must begin with
        # identical servo/solver initialization, preserving the handoff qpos/qvel.
        qpos, qvel = self.state()
        self.collector.reset(qpos=qpos, qvel=qvel)
        initial = self.collector.engine.snapshot()
        self.trace, self.events, self.phases, self.boundaries, self.attempts = (
            [],
            [],
            [],
            [],
            [],
        )
        self.simulator_seconds = self.planning_seconds = 0.0
        self.recoveries = []
        self.placements = []
        expected = np.array(
            self.plan["target"] if name == "disassembly" else self.plan["parking"]
        )
        # A fixed foundation stays at its goal in both stages.
        remaining = set(models.active_parts(self.plan)) - set(self.plan.get("fixed_parts", ()))
        completed = []
        reason = None
        self.rollbacks = []
        checkpoints = [self._checkpoint(completed, remaining, expected)]
        avoid = collections.defaultdict(set)
        try:
            while remaining:
                moving = None
                try:
                    self.budget()
                    choices = (
                        list(self.removable(remaining))
                        if order is None
                        else [order[len(completed)]]
                    )
                    # After a rollback, parts that failed from this state go last.
                    tried = avoid[len(completed)]
                    choices = [c for c in choices if c not in tried] + [
                        c for c in choices if c in tried
                    ]
                    attempt_start = len(self.attempts)
                    result = None
                    edge_part = None
                    for part in choices:
                        self.budget()
                        if part in self.plan.get("edge_access", {}).get("plates", []):
                            edge_part = part
                            break
                        qpos, _ = self.state()
                        began = time.perf_counter()
                        try:
                            result = self.planner.plan(
                                qpos, part, goals[part], deadline=self.deadline
                            )
                        except planning_arm.PlanningError as error:
                            self.attempts.append(
                                {
                                    "part": part,
                                    "reason": str(error),
                                    "grasps": self.planner.attempts.copy(),
                                }
                            )
                        finally:
                            self.planning_seconds += time.perf_counter() - began
                        if result is not None:
                            self.attempts.append(
                                {
                                    "part": part,
                                    "grasp": dataclasses.asdict(result.grasp),
                                    "queries": result.query_count,
                                    "minimum_valid_clearance_lower_bound_m": result.minimum_valid_clearance_m,
                                    "rejected_grasps": result.attempts,
                                }
                            )
                            break
                    if result is None and edge_part is None:
                        try:
                            moved_part = self.regrasp(choices, goals)
                        except planning_arm.PlanningError as error:
                            if str(error) == "no_feasible_removal_or_grasp":
                                for attempt in self.attempts[attempt_start:]:
                                    # Not `reason`: the stage may continue after a rollback.
                                    cause = attempt.get("reason", "")
                                    if cause in ("planning_cpu_budget", "planning_watchdog",
                                                 "experiment_wall_budget"):
                                        raise planning_arm.PlanningBudgetError(cause) from error
                            raise
                    elif edge_part is not None:
                        moving = edge_part
                        plates.PlateMover(self).move(edge_part, goals[edge_part])
                        moved_part = edge_part
                    else:
                        moving = result.part
                        self.move(result, goals[result.part])
                        moved_part = result.part
                    moving = moved_part
                    qpos, _ = self.state()
                    expected[moved_part] = goals[moved_part]
                    poses = qpos[14:].reshape(-1, 7)
                    check = success.waypoint_check(
                        poses, expected, self.plan["sizes"], moved_part
                    )
                    check["passed"] &= (
                        check["scene_position_error_m"] <= 0.005
                        and check["scene_orientation_error_deg"] <= 5
                    )
                    self.boundaries.append(
                        dict(
                            move=len(completed),
                            part=moved_part,
                            state_index=sum(len(t["action"]) for t in self.trace),
                            qpos=qpos.tolist(),
                            **check,
                        )
                    )
                    if not check["passed"]:
                        raise planning_arm.PlanningError("whole_scene_boundary")
                    completed.append(moved_part)
                    remaining.remove(moved_part)
                    checkpoints.append(self._checkpoint(completed, remaining, expected))
                except planning_arm.PlanningError as error:
                    index = self._rollback_target(error, checkpoints, moving)
                    if index is None:
                        raise
                    completed, remaining, expected = self._roll_back(
                        checkpoints, index, error, moving, avoid
                    )
            self.home()
        except planning_arm.PlanningError as error:
            reason = stage_error_reason(
                str(error), time.perf_counter() >= self.deadline
            )
        try:
            self._finish_stage()
        except planning_arm.PlanningError as error:
            if reason is None:
                reason = stage_error_reason(
                    str(error), time.perf_counter() >= self.deadline
                )
        if self.trace:
            trace = {
                k: np.concatenate([t[k] for t in self.trace])
                for k in ("qpos", "qvel", "action")
            }
        else:
            trace = {
                "qpos": np.empty((0, 1, self.collector.engine.native_model.nq)),
                "qvel": np.empty((0, 1, self.collector.engine.native_model.nv)),
                "action": np.empty((0, 1, 7)),
            }
        episode = traces.episode(initial, trace, 0)
        episode["grip_effort"] = (
            np.concatenate(
                [np.full(e["control_steps"], e["grip_effort"]) for e in self.events]
            )
            if self.events
            else np.empty(0)
        )
        io_timing.savez_compressed(self.output / f"{name}.npz", **episode)
        check = (
            success.verify_hold(
                episode["qpos"], episode["qvel"], goals, np.array(self.plan["sizes"])
            )
            if len(episode["qpos"]) >= 51
            else {"passed": False}
        )
        check.update(
            reason=reason,
            completed_order=completed,
            oracle_completed=not remaining,
            boundaries=self.boundaries,
            segments=self.events,
            planning_attempts=self.attempts,
            recoveries=self.recoveries,
            rollbacks=self.rollbacks,
            placement_decisions=self.placements,
            simulator_seconds=self.simulator_seconds,
            planning_seconds=self.planning_seconds,
            elapsed_seconds=time.perf_counter() - started,
            recorded_control_steps=len(episode["action"]),
        )
        check["dynamics"] = guards.motion_check(episode, len(goals))
        check["passed"] &= reason is None and check["dynamics"]["passed"]
        io_timing.write_json(self.output / f"{name}_checks.json", check)
        io_timing.write_json(
            self.output / f"{name}_segments.json",
            [{"name": n, "part": p} for n, p in self.phases],
        )
        return check, episode


def stage_error_reason(reason, wall_budget_exhausted):
    """Distinguish a global deadline inside planning from a local rejection."""
    budget_limited = reason in (
        "manipulation_planning_budget",
        "motion_search_budget",
        "no_feasible_removal_or_grasp",
    ) or reason.endswith((":manipulation_planning_budget", ":motion_search_budget"))
    return (
        "experiment_wall_budget" if wall_budget_exhausted and budget_limited else reason
    )
