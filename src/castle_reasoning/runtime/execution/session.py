"""Runtime execution session."""

import dataclasses
import functools
from pathlib import Path

import numpy as np

import castle_reasoning.runtime.control.batch_tracking as batch_tracking
import castle_reasoning.runtime.control.parameters as parameters
import castle_reasoning.runtime.diagnostics.timing as timing
import castle_reasoning.runtime.execution.runner as core_runner
import castle_reasoning.runtime.planning.manipulation as manipulation
from castle_reasoning.runtime.diagnostics import io_timing
from castle_reasoning.runtime.execution import batching, segments, workers
from castle_reasoning.runtime.planning import budget


class WorldState:
    """Read-only host snapshot of one world between batched simulator calls."""

    def __init__(self, group, world):
        self.group, self.world = group, world
        self.native_model = group.native_models[world]

    def snapshot(self):
        return {
            key: value[self.world : self.world + 1].copy()
            for key, value in self.group.snapshot.items()
        }


class WorldCollector:
    """Runner view; the real shared collector is reset once at each stage boundary."""

    def __init__(self, group, world):
        self.engine = WorldState(group, world)
        self.controller = None

    def reset(self, *, qpos, qvel):
        actual = self.engine.snapshot()
        if not (
            np.array_equal(qpos, actual["qpos"][0])
            and np.array_equal(qvel, actual["qvel"][0])
        ):
            raise ValueError("A worker attempted to reset inside a batched stage")


class Runner(core_runner.Runner):
    """Reuse serial planning, recovery and auditing at batched segment boundaries."""

    def __init__(self, group, world, *args, **kwargs):
        self.group, self.world = group, world
        super().__init__(WorldCollector(group, world), *args, **kwargs)

    def _execute_segment(self, segment, part, reference, guard, monitor_scene):
        request = batch_tracking.Request(
            segment,
            self.planner.scene,
            part,
            reference,
            guard,
            monitor_scene,
            parameters.grip_effort(self.plan, part),
        )
        return self.group.stage.segment(self.world, request)

    def _finish_stage(self):
        self.group.stage.finish(self.world)

    def restore(self, qpos, qvel):
        # Only planner processes whose owner can reset one world mid-stage get
        # a nonzero rollback allowance (see the stage packet).
        self.group.stage.restore(self.world, qpos, qvel)


def stage_worker(connection, packet, *, runner_type=Runner):
    remote = workers.RemoteGroup(
        connection, packet["model_path"], packet["snapshot"],
        source=packet.get("preparation_source"),
        permits=packet.get("preparation_permits"),
        cancelled=packet.get("preparation_cancelled"),
    )
    runner = runner_type(
        remote,
        0,
        packet["plan"],
        Path(packet["output"]),
        packet["planning_seconds"],
        packet["deadline"],
        shuffle_grasps=packet["shuffle_grasps"],
    )
    runner.planner.rng.bit_generator.state = packet["rng_state"]
    runner.max_rollbacks = packet.get("max_rollbacks", 0)
    runner.planner.budget = budget.Budget(
        permits=packet.get("planning_permits"),
        watchdog_seconds=packet.get("planning_watchdog_seconds", 120),
    )
    remote.planning_budget = runner.planner.budget
    check, _ = runner.stage(packet["name"], packet["goals"], packet["order"])
    connection.send(("done", (check, runner.planner.rng.bit_generator.state)))



class ConstructionGroup:
    """Own GPU state on the caller thread; native planning runs in CPU processes."""
    stage_worker = staticmethod(stage_worker)

    def __init__(self, collector, deadline, model_paths):
        self.collector, self.deadline = collector, deadline
        self.native_models = collector.engine.native_models
        self.model_paths = [Path(p).resolve() for p in model_paths]
        if len(self.model_paths) != collector.engine.nworld:
            raise ValueError("One saved model path per world is required.")
        self.snapshot = collector.engine.snapshot()
        self.last = [None] * collector.engine.nworld
        self.simulator_seconds = 0.0
        self.control_steps = 0
        self.wait_steps = 0
        self.wait_failures = 0
        self.padding_steps = 0
        self.stage = None
        self.runners = []
    def execute(self, requests):
        results = batch_tracking.execute(
            self.collector.controller, requests, self.deadline
        )
        self.record_execution(requests, results)
        return results
    def advance(self, requests):
        results = self.segments.advance(requests)
        self.record_execution(requests, self.segments.last_results)
        return results
    @timing.measured('download.snapshot_accounting')
    def record_execution(self, requests, results):
        """Account for each shared chunk once, including all physical waits."""
        self.snapshot = self.collector.engine.snapshot()
        self.last = [dataclasses.replace(r, preparation=None)
                     if r.preparation is not None else r for r in requests]
        self.simulator_seconds += results[0][1]["simulator_seconds"]
        self.control_steps += sum(e["executed_control_steps"] for _, e in results)
        self.padding_steps += sum(e["padding_control_steps"] for _, e in results)
        self.wait_steps += sum(
            e["control_steps"]
            for r, (_, e) in zip(requests, results, strict=True)
            if r.segment.name == "batch_wait"
        )
        self.wait_failures += sum(
            bool(e["failure"])
            for r, (_, e) in zip(requests, results, strict=True)
            if r.segment.name == "batch_wait"
        )
    def waiting(self, world):
        qpos = self.snapshot["qpos"][world]
        segment = manipulation.hold_segment(qpos, ticks=2, name="batch_wait")
        previous = self.last[world]
        if previous is None:
            return batch_tracking.Request(
                segment, self.runners[world].planner.scene, -1, qpos
            )
        # The episode has ended. Keep its physical hold, but never append these
        # later actions to BC data or carry their states into the next episode.
        segment.closure[:] = previous.segment.closure[-1]
        segment.pads_allowed = previous.segment.pads_allowed
        segment.supports = previous.segment.supports
        segment.moving_object = previous.segment.moving_object
        segment.departures, segment.arrivals = (
            previous.segment.departures,
            previous.segment.arrivals,
        )
        if previous.segment.payload is not None:
            # Worker no longer uses its planner after submitting finish.
            segment.payload = previous.scene.attachment(qpos, previous.part)
        return dataclasses.replace(
            previous, segment=segment, scene_reference=qpos, actions=None, efforts=None
        )

    def run_stage(self, name, goals, enabled, orders=None, endpoints=None):
        actual = self.collector.engine.snapshot()
        if endpoints is not None:
            actual = batching.handoff(actual, endpoints)
        self.collector.reset(qpos=actual["qpos"], qvel=actual["qvel"])
        self.snapshot = self.collector.engine.snapshot()
        self.last = [None] * len(self.runners)
        self.segments = segments.SegmentExecutor(
            self.collector.controller, self.deadline
        )
        self.stage = batching.StageGroup(
            len(self.runners), self.advance, self.waiting
        )
        workers = []
        for world, runner in enumerate(self.runners):
            if enabled[world]:
                workers.append(
                    functools.partial(
                        self.run_world,
                        world,
                        name,
                        goals[world],
                        None if orders is None else orders[world],
                    )
                )
            else:
                workers.append(functools.partial(self.stage.finish, world))
        results = self.stage.run(workers)
        return [
            result if enabled[world] else None for world, result in enumerate(results)
        ]
    @timing.measured('worker.stage', 'world', 'name')
    def run_world(self, world, name, goals, order):
        runner = self.runners[world]
        packet = {
            "model_path": str(self.model_paths[world]),
            "plan": runner.plan,
            "output": str(runner.output.resolve()),
            "name": name,
            "goals": goals,
            "order": order,
            "planning_seconds": runner.planner.planning_seconds,
            "planning_permits": getattr(self, "planning_permits", None),
            "planning_watchdog_seconds": getattr(self, "planning_watchdog_seconds", 120),
            "deadline": runner.deadline,
            "shuffle_grasps": runner.planner.shuffle_grasps,
            "max_rollbacks": getattr(self, "max_rollbacks", 0)
            if hasattr(self, "restore")
            else 0,
            "rng_state": runner.planner.rng.bit_generator.state,
            "snapshot": {
                key: value[world : world + 1].copy()
                for key, value in self.snapshot.items()
            },
        }
        if getattr(self, "preparation_sources", None) is not None:
            packet.update(
                preparation_source=self.preparation_sources[world],
                preparation_permits=self.preparation_permits,
                preparation_cancelled=self.preparation_cancelled,
            )
        check, rng_state = workers.relay(
            world, self, runner.planner.scene, packet,
            worker=self.stage_worker, start_method=getattr(self, "planner_start_method", "spawn"),
        )
        runner.planner.rng.bit_generator.state = rng_state
        with io_timing.load(runner.output / f"{name}.npz") as saved:
            sample = {key: saved[key] for key in saved.files}
        return check, sample




class Session:
    """Simulator and controller state for externally requested operations."""

    def __init__(self, engine, controller):
        self.engine = engine
        self.controller = controller
        self.nworld = engine.nworld

    def reset(self, *, qpos, qvel):
        qpos = np.asarray(qpos, dtype=np.float32)
        qvel = np.asarray(qvel, dtype=np.float32)
        if qpos.shape == (self.engine.native_model.nq,):
            qpos = np.broadcast_to(qpos, (self.nworld, qpos.size)).copy()
        if qvel.shape == (self.engine.native_model.nv,):
            qvel = np.broadcast_to(qvel, (self.nworld, qvel.size)).copy()
        self.engine.reset(np.ones(self.nworld, dtype=bool))
        self.engine.set_state(qpos=qpos, qvel=qvel)
        self.controller.sync_targets()
        self.engine.assert_healthy()


def make_session(model, *, worlds=1, seed=0, device="cuda:0",
                 native_models=None, capacity_models=None):
    """Allocate the existing physical runtime without a scripted teacher."""
    from castle_reasoning.runtime.control import joint_velocity, parameters
    from castle_reasoning.runtime.simulation import engine, model_batch, models

    variants = [model] * worlds if native_models is None else list(native_models)
    if len(variants) != worlds or variants[0] is not model:
        raise ValueError("One native model per world, with the base model first.")
    sized = variants + list(capacity_models or ())
    capacities = [models.scene_capacities(m).make_data_kwargs() for m in sized]
    capacity = engine.BufferCapacities(
        **{key: max(c[key] for c in capacities) for key in capacities[0]}
    )
    # Keep five task draws and eight draws for each of two grasp proposals per
    # object, so resetting a session preserves the fixed collection RNG layout.
    physical = engine.MJWarpEngine(
        model, nworld=worlds, device=device, seed=seed,
        random_streams=((model.nq - 14) // 7) * 21, graph=False, capacities=capacity,
    )
    physical.native_models = tuple(variants)
    if any(m is not model for m in variants):
        physical.model = model_batch.stack_models(variants, device)
    controller = joint_velocity.MJWarpJointVelocityController(
        physical,
        parameters.ControllerSpec(physics_timestep_seconds=float(model.opt.timestep)),
        graph=False,
    )
    return Session(physical, controller)
