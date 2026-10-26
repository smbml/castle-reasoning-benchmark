"""Data assembly continuous_collect."""

import functools
import os
import queue
import threading
import time
from pathlib import Path

import mujoco
import numpy as np

import castle_reasoning.data.assembly.batched_collect as batched_collect
import castle_reasoning.data.assembly.publication as publication
import castle_reasoning.data.assembly.records as records
import castle_reasoning.data.assembly.teacher as teacher
import castle_reasoning.data.plans as data_plans
import castle_reasoning.runtime.control.active as active
import castle_reasoning.runtime.control.batch_tracking as batch_tracking
import castle_reasoning.runtime.control.guards as guards
import castle_reasoning.runtime.control.tracking as tracking
import castle_reasoning.runtime.diagnostics.io_timing as io_timing
import castle_reasoning.runtime.diagnostics.provenance as provenance
import castle_reasoning.runtime.diagnostics.timing as timing
import castle_reasoning.runtime.execution.batching as batching
import castle_reasoning.runtime.execution.preparation as preparation
import castle_reasoning.runtime.execution.segments as segments
import castle_reasoning.runtime.execution.success as success
import castle_reasoning.runtime.execution.workers as workers
import castle_reasoning.runtime.planning.collision as collision
import castle_reasoning.runtime.planning.manipulation as manipulation
import castle_reasoning.runtime.simulation.model_batch as model_batch
import castle_reasoning.runtime.simulation.models as simulation_models
import castle_reasoning.runtime.simulation.world_slots as world_slots
from castle_reasoning.runtime.execution import traces


def can_extend_settle(check):
    """Only residual motion of an otherwise valid target may be waited out."""
    return bool(
        not check["passed"]
        and check.get("pose_valid", False)
        and not check["settled"]
        and check["contacts"]["passed"]
        and not check["execution"]["failure"]
    )


def can_settle_replay(check):
    """Only residual motion may be repaired by a post-replay quiet hold."""
    return bool(
        not check["passed"]
        and check.get("pose_valid", False)
        and not check["settled"]
        and check["contacts"]["passed"]
        and check["dynamics"]["passed"]
        and not check["forbidden_contact_frames"]
        and not check["execution"]["failure"]
        and check["complete"]
    )


class Collection(batched_collect.ConstructionGroup):
    """A slot worker owns an attempt; only the coordinator touches GPU state."""

    @timing.measured('startup.collection')
    def __init__(self, collector, pool, plans, homes, root, settings, started, publisher,
                 *, model_ids=None):
        self.root, self.pool, self.plans, self.homes = root, pool, plans, homes
        self.settings, self.started = settings, started
        self.publisher = publisher
        self.records, self.jobs = [], queue.Queue()
        self.lock = threading.Lock()
        self.reset_seconds, self.reset_count = 0.0, 0
        self.reset_batch_count = 0
        self.slots = [None] * collector.engine.nworld
        self.idle_scenes = {}
        self.replay_steps = 0
        self.batch_lane_steps = 0
        self.paused_steps = 0
        self.provenance = records._provenance()
        super().__init__(
            collector,
            started + settings["wall_seconds"],
            [root / "models" / f"{i}.mjb" for i in pool.assignments],
        )
        self.runners = [None] * collector.engine.nworld
        self.preparation_sources = None
        self.planner_start_method = settings.get("planner_start_method", "spawn")
        self.max_rollbacks = settings.get("max_rollbacks", 0)
        self.rollback_count = 0
        context = workers.planner_context(self.planner_start_method)
        self.planning_permits = context.BoundedSemaphore(settings["planning_workers"])
        self.planning_watchdog_seconds = settings["planning_watchdog_seconds"]
        if settings.get("preparation_mode", "owner") == "producer":
            self.preparation_permits = context.BoundedSemaphore(settings["preparation_workers"])
            self.preparation_cancelled = context.Event()
            if model_ids is None or len(model_ids) != len(plans):
                raise ValueError("Producer preparation needs one saved model ID per plan")
            self.model_ids = model_ids
            self.preparation_sources = [preparation.Source(w, 0, 0, 1, "unassigned")
                                        for w in range(collector.engine.nworld)]
        self.segments = segments.SegmentExecutor(
            collector.controller, self.deadline,
            pause_planning_worlds=settings.get("pause_planning_worlds", False),
            skip_paused_preparation=settings.get("skip_paused_preparation", False),
            preparation_sources=self.preparation_sources,
        )
        self.stage = batching.SlotGroup(
            len(self.slots), self.advance, self.waiting, self.reset_slots,
            pause_planning_worlds=settings.get("pause_planning_worlds", False),
        )
        if self.preparation_sources is not None:
            self.stage.on_error = self.preparation_cancelled.set
        jobs = []
        for repeat in range(settings["attempts_per_target"]):
            for model_index, plan in enumerate(plans):
                attempt = len(self.records)
                jobs.append((attempt, model_index))
                self.records.append(
                    {
                        "attempt": attempt,
                        "target_id": plan["target_id"],
                        "repeat": repeat,
                        "status": "queued",
                        "accepted": False,
                    }
                )
        if settings.get("queue_order") == "longest_first":
            # A cohort cannot finish before its longest attempt; within each
            # repeat round, start the largest targets first so they stay off the
            # tail while every target's first attempt still precedes any repeat.
            # Attempt ids, and therefore seeds, are unchanged.
            jobs.sort(
                key=lambda job: (
                    job[0] // len(plans),
                    -len(simulation_models.active_parts(plans[job[1]])),
                )
            )
        for job in jobs:
            self.jobs.put(job)

    @timing.measured('publish')
    def publish(self, *, completed=False):
        """Called with the record lock held; keep every target and attempt visible."""
        accepted = [r for r in self.records if r["accepted"]]
        begun = [r for r in self.records if "slot" in r]
        elapsed = time.perf_counter() - self.started
        verified_targets = len({r["target_id"] for r in accepted})
        summary = {
            "elapsed_seconds": elapsed,
            "scheduled_attempts": len(self.records),
            "attempted": len(begun),
            "accepted_attempts": len(accepted),
            "target_count": len(self.plans),
            "verified_targets": verified_targets,
            "target_yield": verified_targets / len(self.plans),
            "attempt_yield": len(accepted) / len(begun) if begun else 0,
            "accepted_demonstrations_per_hour": 3600 * len(accepted) / elapsed,
            "worlds": len(self.slots),
            "allocated_part_slots": len(self.plans[0]["sizes"]),
            "attempts": self.records,
            "slots": self.slots,
            "simulator_seconds": self.simulator_seconds,
            "simulated_control_steps": self.control_steps,
            "total_physics_steps": self.control_steps
            * round(0.02 / self.settings["physics_timestep"]),
            "batch_lane_control_steps": self.batch_lane_steps,
            "paused_control_steps": self.paused_steps,
            "forward_dynamics_world_steps": self.batch_lane_steps
            * round(0.02 / self.settings["physics_timestep"]),
            "post_terminal_control_steps": self.wait_steps,
            "post_terminal_guard_failure_chunks": self.wait_failures,
            "replay_control_steps": self.replay_steps,
            "reset_count": self.reset_count,
            "rollback_count": self.rollback_count,
            "reset_batch_count": self.reset_batch_count,
            "reset_seconds": self.reset_seconds,
            "accepted_action_targets": sum(
                r.get("action_targets", 0) for r in accepted
            ),
        }
        if completed:
            summary["completed"] = True
        self.publisher.submit(summary)
        return summary

    def update(self, attempt, **values):
        with self.lock:
            self.records[attempt].update(values)
            self.publish()

    @timing.measured('reset.body')
    def reset_slots(self, requests):
        """Install queued states together; no untouched GPU state is downloaded."""
        if not requests:
            return {}
        worlds = list(requests)
        if any(not 0 <= world < len(self.slots) for world in worlds):
            raise ValueError("Invalid reset world")
        if self.segments.pending is not None and any(
            self.segments.pending[world] is not None for world in worlds
        ):
            raise RuntimeError("Cannot recycle a slot with an unfinished segment")
        started = time.perf_counter()
        engine = self.collector.engine
        qpos = np.zeros((engine.nworld, engine.native_model.nq), np.float32)
        qvel = np.zeros((engine.nworld, engine.native_model.nv), np.float32)
        mask = np.zeros(engine.nworld, bool)
        for world, (_, _, _, position, velocity) in requests.items():
            qpos[world], qvel[world], mask[world] = position, velocity, True
        if not np.isfinite(qpos[mask]).all() or not np.isfinite(qvel[mask]).all():
            raise ValueError("Selected initial states must be finite")
        self.pool.select(worlds, [requests[world][1] for world in worlds])
        world_slots.reset(
            self.collector.controller, mask, qpos=qpos, qvel=qvel
        )
        self.snapshot = engine.snapshot()
        for world, (attempt, model_index, _, _, _) in requests.items():
            if self.preparation_sources is not None:
                self.preparation_sources[world] = preparation.Source(
                    world, self.preparation_sources[world].slot_epoch + 1,
                    0, 1, self.model_ids[model_index],
                )
            self.last[world] = None
            path = self.root / f"attempt{attempt:05d}"
            self.model_paths[world] = self.root / "models" / f"{model_index}.mjb"
            if self.slots[world] is None or self.slots[world]["attempt"] != attempt:
                self.runners[world] = batched_collect.Runner(
                    self, world, self.plans[model_index], path,
                    self.settings["planning_seconds"], self.deadline,
                    seed=self.settings["seed"] + attempt,
                    # Every attempt uses the same grasp policy; its seed alone
                    # differentiates repeats. `attempt` is a global queue index,
                    # so `attempt > 0` singled out whichever target was first.
                    shuffle_grasps=True,
                )
        self.reset_seconds += time.perf_counter() - started
        self.reset_count += len(worlds)
        self.reset_batch_count += 1
        with self.lock:
            for world, (attempt, model_index, name, _, _) in requests.items():
                self.slots[world] = {
                    "attempt": attempt,
                    "target_id": self.plans[model_index]["target_id"],
                    "stage": name,
                }
                self.records[attempt].update(slot=world, status=name)
            self.publish()
        return {
            world: {key: value[world : world + 1].copy()
                    for key, value in self.snapshot.items()}
            for world in worlds
        }

    def waiting(self, world):
        if self.runners[world] is None:
            qpos = self.snapshot["qpos"][world]
            if world not in self.idle_scenes:
                self.idle_scenes[world] = collision.Scene(
                    self.native_models[world], qpos
                )
            return batch_tracking.Request(
                manipulation.hold_segment(qpos, ticks=2, name="batch_wait"),
                self.idle_scenes[world],
                -1,
                guard=False,
            )
        return super().waiting(world)

    def submit_segment(self, world, request):
        """Prepare local settling/replay requests before their GPU admission."""
        if self.preparation_sources is not None:
            request = preparation.prepare(
                request, self.snapshot["qpos"][world:world + 1],
                self.preparation_sources[world], permits=self.preparation_permits,
                cancelled=self.preparation_cancelled, location="slot_thread",
            )
        return self.stage.segment(world, request)

    def record_execution(self, requests, results):
        super().record_execution(requests, results)
        self.batch_lane_steps += sum(
            e.get("batch_lane_control_steps", e["executed_control_steps"]) for _, e in results
        )
        self.paused_steps += sum(e.get("paused_control_steps", 0) for _, e in results)
        self.replay_steps += sum(
            e["executed_control_steps"]
            for r, (_, e) in zip(requests, results, strict=True)
            if r.actions is not None
        )
        self.publisher.check()

    @timing.measured('worker', 'world')
    def worker(self, world):
        while time.perf_counter() < self.deadline:
            try:
                attempt, model_index = self.jobs.get_nowait()
            except queue.Empty:
                break
            self.attempt(world, attempt, model_index)
            with self.lock:
                self.slots[world]["stage"] = "idle"
                self.publish()
        self.stage.close(world)

    @timing.measured('attempt', 'world', 'attempt')
    def attempt(self, world, attempt, model_index):
        plan, model = self.plans[model_index], self.pool.models[model_index]
        path = self.root / f"attempt{attempt:05d}"
        path.mkdir()
        (path / "model.mjb").hardlink_to(self.root / "models" / f"{model_index}.mjb")
        io_timing.write_json(path / "plan.json", plan)
        io_timing.write_json(path / "provenance.json", self.provenance)
        io_timing.write_json(
            path / "execution_config.json",
            self.settings
            | {
                "slot": world,
                "attempt": attempt,
                "seed": self.settings["seed"] + attempt,
            },
        )
        initial = self.stage.reset(
            world,
            (
                attempt,
                model_index,
                "settle",
                self.homes[model_index],
                np.zeros(model.nv),
            ),
        )
        runner = self.runners[world]
        segment = manipulation.hold_segment(initial["qpos"][0], 250)
        trace, event = self.submit_segment(
            world,
            batch_tracking.Request(segment, runner.planner.scene, -1, guard=False),
        )
        sample = traces.episode(initial, trace, 0)
        check = self.check_hold(sample, model, plan, "supports")
        check["execution"] = event
        check["passed"] &= not event["failure"]
        if can_extend_settle(check):
            sample, check = self.extend_settle(world, runner, sample, check, plan, model)
        self.stage.finish(world)
        io_timing.savez_compressed(path / "settle.npz", **sample)
        io_timing.write_json(path / "settle_checks.json", check)
        if not check["passed"]:
            self.update(
                attempt,
                status="budget_exhausted_settle"
                if check["execution"]["failure"] == 5
                else "failed_settle",
                checks={"settle": check},
            )
            return
        checks = {"settle": check}
        order = None
        for name, goal_key, support_key in [
            ("disassembly", "parking", "parking_supports"),
            ("assembly", "target", "supports"),
        ]:
            self.stage.reset(
                world,
                (attempt, model_index, name, sample["qpos"][-1], sample["qvel"][-1]),
            )
            check, sample = self.run_world(world, name, np.array(plan[goal_key]), order)
            check["contacts"] = success.contact_report(
                model, sample["qpos"], plan[support_key]
            )
            check["passed"] &= check["contacts"]["passed"]
            io_timing.write_json(path / f"{name}_checks.json", check)
            checks[name] = {
                k: v
                for k, v in check.items()
                if k not in ("segments", "boundaries", "planning_attempts")
            }
            if not check["passed"]:
                status = (
                    "budget_exhausted_" + name
                    if check.get("reason") == "experiment_wall_budget"
                    else "failed_" + name
                )
                self.update(attempt, status=status, checks=checks)
                return
            if name == "disassembly":
                order = check["completed_order"][::-1]
                io_timing.write_json(
                    path / "execution_order.json",
                    {"disassembly": check["completed_order"], "assembly": order},
                )
        self.replay(world, attempt, model_index, sample, check, checks)

    def restore(self, world, qpos, qvel):
        """Return one world to a passed boundary of its current stage.

        Uses the ordinary slot reset: the same attempt and model, the saved
        position and velocity, and fresh servo and solver state, as at every
        stage handoff. Boundaries follow completed moves with the arm at rest.
        """
        slot = self.slots[world]
        self.stage.reset(
            world,
            (slot["attempt"], self.pool.assignments[world], slot["stage"], qpos, qvel),
        )
        self.rollback_count += 1

    @staticmethod
    @timing.measured('audit.hold')
    def check_hold(sample, model, plan, support_key):
        check = (
            success.verify_hold(
                sample["qpos"],
                sample["qvel"],
                np.array(plan["target"]),
                np.array(plan["sizes"]),
            )
            if len(sample["qpos"]) >= 51
            else {"passed": False}
        )
        check["contacts"] = success.contact_report(
            model, sample["qpos"], plan[support_key]
        )
        check["passed"] &= check["contacts"]["passed"]
        return check

    @timing.measured('replay', 'world', 'attempt')
    def extend_settle(self, world, runner, sample, check, plan, model):
        """Keep holding the untouched target in 1 s windows until it is at rest.

        A slender target can carry a slowly decaying startup transient past the
        fixed 5 s window. As for replay verification, wait at most
        `replay_settle_seconds` more and apply the unchanged final-second check
        to each window. Settle frames are never BC data; disassembly starts
        from the final settled state.
        """
        initial_check, windows = dict(check), []
        home = sample["qpos"][0]
        for _ in range(self.settings["replay_settle_seconds"]):
            request = batch_tracking.Request(
                manipulation.hold_segment(home, 50),
                runner.planner.scene,
                -1,
                guard=False,
            )
            trace, event = self.submit_segment(world, request)
            last = {key: sample[key][-1:] for key in ("qpos", "qvel")}
            window = traces.episode(
                last, {key: trace[key] for key in ("qpos", "qvel", "action")}, 0
            )
            sample = {
                "action": np.concatenate([sample["action"], window["action"]]),
                "time": np.arange(len(sample["qpos"]) + len(window["action"])) * 0.02,
                "qpos": np.concatenate([sample["qpos"], window["qpos"][1:]]),
                "qvel": np.concatenate([sample["qvel"], window["qvel"][1:]]),
            }
            check = self.check_hold(sample, model, plan, "supports")
            check["execution"] = event
            check["passed"] &= not event["failure"]
            windows.append({k: check[k] for k in ("passed", "settled", "pose_valid")})
            if check["passed"] or not can_extend_settle(check):
                break
        check["settling"] = {
            "initial_hold": initial_check,
            "windows": windows,
            "extra_seconds": len(windows),
            "included_in_bc": False,
        }
        return sample, check

    def replay(self, world, attempt, model_index, sample, assembly, checks):
        plan, model = self.plans[model_index], self.pool.models[model_index]
        path = self.root / f"attempt{attempt:05d}"
        initial = self.stage.reset(
            world,
            (attempt, model_index, "replay", sample["qpos"][0], sample["qvel"][0]),
        )
        runner = self.runners[world]
        segment = manipulation.Segment(
            "replay", np.array([sample["qpos"][0]]), np.zeros(1)
        )
        request = batch_tracking.Request(
            segment,
            runner.planner.scene,
            -1,
            guard=False,
            monitor_scene=False,
            actions=sample["action"],
            efforts=sample["grip_effort"],
        )
        trace, event = self.submit_segment(world, request)
        replay = traces.episode(initial, trace, 0)
        replay["grip_effort"] = sample["grip_effort"][: len(replay["action"])]
        io_timing.savez_compressed(path / "replay.npz", **replay)
        check = self.check_hold(replay, model, plan, "supports")
        check["dynamics"] = guards.motion_check(replay, len(plan["sizes"]))
        hits = []
        for e in assembly["segments"]:
            start = e["start_action"] + 1
            found = guards.audit_contacts(
                runner.audit_scene,
                replay["qpos"][start : start + e["control_steps"]],
                e["part"],
                e["pads_allowed"],
                e["supports"],
                e["carrying"],
                e["moving_object"],
                e["departures"],
                e["arrivals"],
            )
            hits.extend({**h, "frame": start + h["frame"]} for h in found)
        check.update(
            forbidden_contact_frames=len(hits),
            first_forbidden_contacts=hits[:3],
            execution=event,
            complete=len(replay["action"]) == len(sample["action"]),
        )
        check["passed"] &= (
            check["dynamics"]["passed"]
            and not hits
            and not event["failure"]
            and check["complete"]
        )
        check = self.settle_replay(world, model_index, replay, check, path)
        self.stage.finish(world)
        checks["replay"] = check
        io_timing.write_json(path / "replay_checks.json", check)
        self.update(
            attempt,
            accepted=bool(check["passed"]),
            status="accepted"
            if check["passed"]
            else "budget_exhausted_replay"
            if event["failure"] == 5
            or check.get("settling", {}).get("execution_failure") == 5
            else "failed_replay",
            checks=checks,
            action_targets=len(sample["action"]),
        )

    def settle_replay(self, world, model_index, replay, check, path):
        """Continue physical verification without appending BC action targets."""
        limit = self.settings["replay_settle_seconds"]
        if not limit or not can_settle_replay(check):
            return check
        initial_check = dict(check)
        check = dict(check)
        model, plan = self.pool.models[model_index], self.plans[model_index]
        runner = self.runners[world]
        positions, velocities = [replay["qpos"][-1:]], [replay["qvel"][-1:]]
        actions, windows = [], []
        simulator_seconds = 0.0
        for _ in range(limit):
            request = batch_tracking.Request(
                manipulation.hold_segment(replay["qpos"][-1], ticks=41),
                runner.planner.scene,
                -1,
                guard=True,
                monitor_scene=False,
            )
            trace, event = self.submit_segment(world, request)
            window = {
                "qpos": np.concatenate([positions[-1][-1:], trace["qpos"][:, 0]]),
                "qvel": np.concatenate([velocities[-1][-1:], trace["qvel"][:, 0]]),
                "action": trace["action"][:, 0],
            }
            positions.append(window["qpos"][1:])
            velocities.append(window["qvel"][1:])
            actions.append(window["action"])
            simulator_seconds += event["simulator_seconds"]
            held = self.check_hold(window, model, plan, "supports")
            held["dynamics"] = guards.motion_check(window, len(plan["sizes"]))
            hits = guards.audit_contacts(
                runner.audit_scene, window["qpos"][1:], -1, False, (), False
            )
            held.update(
                execution=event,
                forbidden_contact_frames=len(hits),
                first_forbidden_contacts=hits[:3],
            )
            safe = not (event["failure"] or hits) and held["dynamics"]["passed"]
            held["passed"] &= safe and len(window["action"]) == 50
            windows.append(held)
            # Keep the original replay's execution and audit records intact.
            check.update(
                {
                    k: v
                    for k, v in held.items()
                    if k
                    not in (
                        "execution",
                        "dynamics",
                        "forbidden_contact_frames",
                        "first_forbidden_contacts",
                    )
                }
            )
            if held["passed"] or not safe or not held.get("pose_valid", False):
                break
        check["settling"] = {
            "initial_hold": initial_check,
            "windows": windows,
            "control_steps": sum(len(a) for a in actions),
            "simulator_seconds": simulator_seconds,
            "execution_failure": windows[-1]["execution"]["failure"],
            "included_in_bc": False,
        }
        io_timing.savez_compressed(
            path / "replay_settle.npz",
            qpos=np.concatenate(positions),
            qvel=np.concatenate(velocities),
            action=np.concatenate(actions),
        )
        return check


@timing.measured('collector')
def execute(
    plans,
    output,
    worlds=20,
    attempts_per_target=1,
    planning_seconds=8,
    wall_seconds=3600,
    seed=0,
    physics_timestep=0.002,
    *,
    contact_impedance=None,
    replay_settle_seconds=10,
    friction_cone=None,
    solver=None,
    solver_iterations=None,
    line_search_threads=None,
    pause_planning_worlds=False,
    skip_paused_preparation=False,
    preparation_mode="owner",
    preparation_workers=None,
    queue_order="submitted",
    planning_workers=None,
    planning_watchdog_seconds=120,
    planner_start_method="spawn",
    max_rollbacks=0,
):
    """Run a balanced finite attempt queue; every accepted assembly is replayed."""
    workers.planner_context(planner_start_method)
    if not isinstance(max_rollbacks, int) or max_rollbacks < 0:
        raise ValueError("Rollback allowance must be a nonnegative integer")
    if queue_order not in ("submitted", "longest_first"):
        raise ValueError("Queue order must be 'submitted' or 'longest_first'")
    if skip_paused_preparation and not pause_planning_worlds:
        raise ValueError("Skipping preparation requires paused worlds")
    if preparation_mode not in ("owner", "producer"):
        raise ValueError("Unknown preparation mode")
    if preparation_mode == "producer" and not skip_paused_preparation:
        raise ValueError("Producer preparation requires skipped paused preparation")
    allocated_cpus = min(len(os.sched_getaffinity(0)),
                         int(os.environ.get("SLURM_CPUS_PER_TASK", len(os.sched_getaffinity(0)))))
    if planning_workers is None:
        planning_workers = max(1, allocated_cpus - 2)
    if not isinstance(planning_workers, int) or planning_workers < 1:
        raise ValueError("Positive planner worker count required")
    if not np.isfinite(planning_watchdog_seconds) or planning_watchdog_seconds <= 0:
        raise ValueError("Positive finite planner watchdog required")
    if preparation_workers is None:
        preparation_workers = min(4, max(1, allocated_cpus - 2))
    if not isinstance(preparation_workers, int) or preparation_workers < 1:
        raise ValueError("Positive preparation worker count required")
    if pause_planning_worlds:
        pass
        import mujoco_warp
        if not hasattr(mujoco_warp, "step_active"):
            raise RuntimeError("Select the repo-owned MJ-Warp active-world runtime")
    if not plans or any(
        not isinstance(v, int) or v < 1 for v in (worlds, attempts_per_target)
    ):
        raise ValueError(
            "Nonempty plans and positive integer world/attempt counts required"
        )
    if not all(np.isfinite(v) and v > 0 for v in (planning_seconds, wall_seconds)):
        raise ValueError("Finite positive planning and wall budgets required")
    if not isinstance(replay_settle_seconds, int) or replay_settle_seconds < 0:
        raise ValueError("Replay settling limit must be a nonnegative integer")
    if friction_cone not in (None, "elliptic", "pyramidal"):
        raise ValueError("Unsupported friction cone")
    if solver not in (None, "cg", "newton"):
        raise ValueError("Unsupported solver")
    if solver_iterations is not None and (
        not isinstance(solver_iterations, int) or solver_iterations <= 0
    ):
        raise ValueError("Solver iterations must be a positive integer")
    if line_search_threads is not None and (
        not isinstance(line_search_threads, int) or line_search_threads not in (32, 128)
    ):
        raise ValueError("Line-search thread count must be 32 or 128")
    # Plate pushes were qualified with fine integration; at 2 ms single-frame
    # contact bounces cross the 0.3 mm support guard, with or without the stiffer
    # contact impedance. Plans that push edge plates record their limit.
    too_coarse = [
        plan.get("target_id")
        for plan in plans
        if physics_timestep
        > plan.get("edge_access", {}).get("max_physics_timestep_s", float("inf")) + 1e-12
    ]
    if too_coarse:
        raise ValueError(
            f"Plans {too_coarse} push edge plates and require a physics timestep "
            "of at most 0.001 s; collect them in a separate run."
        )
    with timing.span("startup"):
        config = data_plans.control_config(physics_timestep)
        contact_impedance = simulation_models.validate_contact_impedance(contact_impedance)
        started = time.perf_counter()
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        (output / "models").mkdir()
        capacity = max(len(p["sizes"]) for p in plans)
        plans = [model_batch.pad_plan(p, capacity) for p in plans]
        models, homes = zip(
            *(
                simulation_models.compile_model(
                    p, config, contact_impedance=contact_impedance
                )
                for p in plans
            ),
            strict=True,
        )
        model_ids = []
        for i, model in enumerate(models):
            if friction_cone is not None:
                model.opt.cone = getattr(mujoco.mjtCone, f"mjCONE_{friction_cone.upper()}")
            if solver is not None:
                model.opt.solver = getattr(mujoco.mjtSolver, f"mjSOL_{solver.upper()}")
            if solver_iterations is not None:
                model.opt.iterations = solver_iterations
            with timing.span("startup.model_save"):
                path = output / "models" / f"{i}.mjb"
                if preparation_mode == "producer":
                    model_ids.append(simulation_models.save_model(model, path))
                else:
                    mujoco.mj_saveModel(model, str(path), None)
        worlds = min(worlds, len(plans) * attempts_per_target)
        indices = [i % len(plans) for i in range(worlds)]
        pool = world_slots.ModelPool(models, indices)
        collector = teacher.make_collector(
            models[0],
            np.array(plans[0]["target"]),
            worlds,
            1,
            seed,
            native_models=pool.native_models,
            capacity_models=models,
        )
        collector.engine.model = pool.model
        if line_search_threads is not None:
            pool.model.block_dim.linesearch_iterative = line_search_threads
        collector.engine.native_models = pool.native_models
        controller_type = active.ActivePathControl if pause_planning_worlds else tracking.PathControl
        collector.controller = controller_type(collector)
        collector.reset(
            qpos=np.array([homes[i] for i in indices]),
            qvel=np.zeros((worlds, models[0].nv)),
        )
        settings = {
            "worlds": worlds,
            "attempts_per_target": attempts_per_target,
            "planning_seconds": planning_seconds,
            "planning_budget_clock": "thread_cpu",
            "planning_workers": planning_workers,
            "planning_watchdog_seconds": planning_watchdog_seconds,
            "planner_start_method": planner_start_method,
            "max_rollbacks": max_rollbacks,
            "wall_seconds": wall_seconds,
            "physics_timestep": physics_timestep,
            "physics_runtime_overrides": (["active-world-integration"] if pause_planning_worlds else []),
            "active_world_runtime": provenance.runtime_provenance() if pause_planning_worlds else None,
            "friction_cone": mujoco.mjtCone(models[0].opt.cone).name,
            "solver": mujoco.mjtSolver(models[0].opt.solver).name,
            "solver_iterations": int(models[0].opt.iterations),
            "line_search_threads": int(pool.model.block_dim.linesearch_iterative),
            "contact_impedance": contact_impedance,
            "replay_settle_seconds": replay_settle_seconds,
            "seed": seed,
            "controller": "continuous_checked_paths",
            "pause_planning_worlds": pause_planning_worlds,
            "skip_paused_preparation": skip_paused_preparation,
            "preparation_mode": preparation_mode,
            "preparation_workers": preparation_workers,
            "queue_order": queue_order,
            "allocated_cpus": allocated_cpus,
            "episode_end": "own hold or recorded failure",
        }
        publisher = publication.SummaryPublisher(output / "summary.json")
        collection = Collection(
            collector, pool, plans, homes, output, settings, started, publisher,
            model_ids=model_ids,
        )
    with publisher:
        collection.publish()
        collection.stage.run(
            [functools.partial(collection.worker, w) for w in range(worlds)]
        )
        for record in collection.records:
            if record["status"] == "queued":
                record["status"] = "not_started_budget"
        summary = collection.publish(completed=True)
    return summary
