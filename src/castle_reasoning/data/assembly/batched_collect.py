"""Data assembly batched_collect."""

import time
from pathlib import Path

import mujoco
import numpy as np

import castle_reasoning.data.assembly.collect as collection
import castle_reasoning.data.assembly.records as records
import castle_reasoning.data.assembly.teacher as teacher
import castle_reasoning.data.plans as data_plans
import castle_reasoning.runtime.control.batch_tracking as batch_tracking
import castle_reasoning.runtime.control.guards as guards
import castle_reasoning.runtime.control.tracking as tracking
import castle_reasoning.runtime.diagnostics.io_timing as io_timing
import castle_reasoning.runtime.execution.batching as batching
import castle_reasoning.runtime.execution.session as session
import castle_reasoning.runtime.execution.success as success
import castle_reasoning.runtime.planning.manipulation as manipulation
import castle_reasoning.runtime.simulation.model_batch as model_batch
import castle_reasoning.runtime.simulation.models as simulation_models
from castle_reasoning.runtime.execution import traces


class Runner(session.Runner, collection.Runner):
    """Batched requested-move execution with demonstration stage sequencing."""


def execute(
    plan,
    output,
    target_id,
    worlds=4,
    planning_seconds=8.0,
    wall_seconds=900.0,
    seed=0,
    physics_timestep=0.0005,
):
    """Collect repeated attempts on one target with the existing entrypoint."""
    if not isinstance(worlds, int) or worlds < 1:
        raise ValueError("World count must be a positive integer")
    return _execute(
        [plan] * worlds,
        output,
        [target_id] * worlds,
        planning_seconds,
        wall_seconds,
        seed,
        physics_timestep,
    )


def execute_many(
    plans,
    output,
    target_ids,
    planning_seconds=8.0,
    wall_seconds=900.0,
    seed=0,
    physics_timestep=0.0005,
):
    """Collect one complete attempt per distinct scene in a shared physics batch."""
    return _execute(
        plans,
        output,
        target_ids,
        planning_seconds,
        wall_seconds,
        seed,
        physics_timestep,
        mixed=True,
    )


def _execute(
    plans,
    output,
    target_ids,
    planning_seconds=8.0,
    wall_seconds=900.0,
    seed=0,
    physics_timestep=0.0005,
    *,
    mixed=False,
):
    """Collect complete independent attempts; preserve all outcomes and replay gates."""
    worlds = len(plans)
    if not worlds or len(target_ids) != worlds:
        raise ValueError("One nonempty plan and target ID per world is required.")
    if not all(np.isfinite(v) and v > 0 for v in (planning_seconds, wall_seconds)):
        raise ValueError("Planning and wall budgets must be finite and positive")
    config = data_plans.control_config(physics_timestep)
    started = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if mixed:
        capacity = max(len(plan["sizes"]) for plan in plans)
        plans = [model_batch.pad_plan(plan, capacity) for plan in plans]
    sizes = np.array([plan["sizes"] for plan in plans])
    goals = np.array([plan["target"] for plan in plans])
    models, homes = [], []
    for world, plan in enumerate(plans):
        if world and not mixed:
            models.append(models[0])
            homes.append(homes[0])
            continue
        model, home = simulation_models.compile_model(plan, config)
        models.append(model)
        homes.append(home)
    homes = np.array(homes)
    provenance = records._provenance()
    io_timing.write_json(output / "provenance.json", provenance)
    io_timing.write_json(output / "plans.json", plans)
    if not mixed:
        mujoco.mj_saveModel(models[0], str(output / "model.mjb"), None)
        io_timing.write_json(output / "plan.json", plans[0])
    reports = []
    for world in range(worlds):
        path = output / f"world{world}"
        path.mkdir()
        if mixed:
            mujoco.mj_saveModel(models[world], str(path / "model.mjb"), None)
        else:
            (path / "model.mjb").hardlink_to(output / "model.mjb")
        io_timing.write_json(path / "plan.json", plans[world])
        io_timing.write_json(path / "provenance.json", provenance)
        io_timing.write_json(
            path / "execution_config.json",
            {
                "controller": "batched_checked_paths",
                "cpu_planning": "spawned_process_per_world_stage",
                "seed": seed + world,
                "grasp_order": "shuffled" if mixed or world > 0 else "ordered",
                "worlds": worlds,
                "world": world,
                "physics_timestep": physics_timestep,
                "physics_runtime_overrides": [],
                "control_timestep": 0.02,
                "planning_seconds": planning_seconds,
                "wall_seconds": wall_seconds,
                "physics_cone_curvature": "installed_mujoco_warp",
                "physics_contact_geometry": "installed_mujoco_warp",
                "contact_limits": "same as serial planned_control.contact_limits",
                "episode_end": "own home and final hold, or recorded failure",
                "padding": "post-terminal physics counted separately, excluded from episodes",
                "stage_reset": "saved endpoint qpos/qvel and reset servo/solver at boundaries only",
            },
        )
        io_timing.savez_compressed(
            path / "requested_target.npz",
            qpos=homes[world],
            sizes=sizes[world],
            goals=goals[world],
        )
        reports.append(
            {
                "target_id": target_ids[world],
                "world": world,
                "part_count": len(simulation_models.active_parts(plans[world])),
                "accepted": False,
                "checks": {},
                "status": "prepared",
            }
        )

    def publish():
        for world, report in enumerate(reports):
            report["elapsed_seconds"] = time.perf_counter() - started
            io_timing.write_json(output / f"world{world}" / "summary.json", report)
        summary = {
            "target_id": None if mixed else target_ids[0],
            "target_ids": target_ids,
            "worlds": worlds,
            "part_count": None if mixed else len(sizes[0]),
            "allocated_part_slots": sizes.shape[1],
            "elapsed_seconds": time.perf_counter() - started,
            "accepted_attempts": sum(r["accepted"] for r in reports),
            "accepted_target": any(r["accepted"] for r in reports),
            "attempts": reports,
        }
        io_timing.write_json(output / "summary.json", summary)
        return summary

    publish()
    collector = teacher.make_collector(
        models[0], goals[0], worlds, 1, seed, native_models=models
    )
    collector.controller = tracking.PathControl(collector)
    collector.reset(qpos=homes, qvel=np.zeros((worlds, models[0].nv)))
    group = ConstructionGroup(
        collector,
        started + wall_seconds,
        [output / f"world{w}" / "model.mjb" for w in range(worlds)],
    )
    group.runners = [
        Runner(
            group,
            world,
            plans[world],
            output / f"world{world}",
            planning_seconds,
            started + wall_seconds,
            seed=seed + world,
            # Replicas of one plan keep world 0 ordered as a reference; with mixed
            # plans that would single out one target, so every world shuffles.
            shuffle_grasps=mixed or world > 0,
        )
        for world in range(worlds)
    ]
    initial = collector.engine.snapshot()
    settle = group.execute(
        [
            batch_tracking.Request(
                manipulation.hold_segment(homes[world], 250),
                r.planner.scene,
                -1,
                guard=False,
            )
            for world, r in enumerate(group.runners)
        ]
    )
    enabled = []
    for world, (trace, metrics) in enumerate(settle):
        sample = traces.episode(
            {k: v[world : world + 1] for k, v in initial.items()}, trace, 0
        )
        io_timing.savez_compressed(output / f"world{world}" / "settle.npz", **sample)
        check = (
            success.verify_hold(
                sample["qpos"], sample["qvel"], goals[world], sizes[world]
            )
            if len(sample["qpos"]) >= 51
            else {"passed": False}
        )
        check.update(
            execution=metrics,
            contacts=success.contact_report(
                models[world], sample["qpos"], plans[world]["supports"]
            ),
        )
        check["passed"] &= check["contacts"]["passed"] and not metrics["failure"]
        reports[world]["checks"]["settle"] = check
        reports[world]["status"] = "settled" if check["passed"] else "failed_settle"
        enabled.append(bool(check["passed"]))
    publish()
    orders, forwards, endpoints = [None] * worlds, [None] * worlds, None
    for name, destination_key, support_key in [
        ("disassembly", "parking", "parking_supports"),
        ("assembly", "target", "supports"),
    ]:
        destinations = np.array([plan[destination_key] for plan in plans])
        if not any(enabled):
            break
        results = group.run_stage(
            name,
            destinations,
            enabled,
            orders if name == "assembly" else None,
            endpoints=endpoints,
        )
        endpoints = [None] * worlds
        for world, result in enumerate(results):
            if result is None:
                continue
            check, sample = result
            check["contacts"] = success.contact_report(
                models[world], sample["qpos"], plans[world][support_key]
            )
            check["passed"] &= check["contacts"]["passed"]
            io_timing.write_json(output / f"world{world}" / f"{name}_checks.json", check)
            reports[world]["checks"][name] = {
                k: v
                for k, v in check.items()
                if k not in ("segments", "boundaries", "planning_attempts")
            }
            enabled[world] = bool(check["passed"])
            reports[world]["status"] = (
                {"disassembly": "disassembled", "assembly": "assembled"}[name]
                if enabled[world]
                else "budget_exhausted_" + name
                if check.get("reason") == "experiment_wall_budget"
                else "failed_" + name
            )
            if name == "disassembly" and enabled[world]:
                endpoints[world] = sample
                orders[world] = check["completed_order"][::-1]
                io_timing.write_json(
                    output / f"world{world}" / "execution_order.json",
                    {
                        "disassembly": check["completed_order"],
                        "assembly": orders[world],
                    },
                )
            if name == "assembly":
                forwards[world] = (check, sample)
        publish()
    replay_seconds, replay_steps = 0.0, 0
    replay_padding = 0
    if any(enabled):
        samples = [forwards[w][1] if enabled[w] else None for w in range(worlds)]
        actions, efforts, lengths = batching.replay_commands(samples)
        reserve = (
            max(
                forwards[w][0]["simulator_seconds"] for w in range(worlds) if enabled[w]
            )
            * 1.15
        )
        if time.perf_counter() + reserve > group.deadline:
            for w in range(worlds):
                if enabled[w]:
                    reports[w]["status"] = "budget_exhausted_before_replay"
        else:
            count = len(actions)
            qpos, qvel = homes.copy(), np.zeros((worlds, models[0].nv))
            for world in range(worlds):
                if enabled[world]:
                    sample = forwards[world][1]
                    qpos[world], qvel[world] = sample["qpos"][0], sample["qvel"][0]
            collector.reset(qpos=qpos, qvel=qvel)
            initial = collector.engine.snapshot()
            trace, seconds = batch_tracking.replay(
                collector.controller, actions, efforts
            )
            replay_seconds, replay_steps = seconds, count * worlds
            replay_padding = int(replay_steps - lengths.sum())
            for world in range(worlds):
                if not enabled[world]:
                    continue
                length = lengths[world]
                sample = traces.episode(
                    initial,
                    {key: value[:length] for key, value in trace.items()},
                    world,
                )
                sample["grip_effort"] = efforts[:length, world].copy()
                io_timing.savez_compressed(output / f"world{world}" / "replay.npz", **sample)
                check = success.verify_hold(
                    sample["qpos"], sample["qvel"], goals[world], sizes[world]
                )
                check.update(
                    contacts=success.contact_report(
                        models[world], sample["qpos"], plans[world]["supports"]
                    ),
                    dynamics=guards.motion_check(sample, len(goals[world])),
                )
                hits = []
                for event in forwards[world][0]["segments"]:
                    first = event["start_action"] + 1
                    found = guards.audit_contacts(
                        group.runners[world].audit_scene,
                        sample["qpos"][first : first + event["control_steps"]],
                        event["part"],
                        event["pads_allowed"],
                        event["supports"],
                        event["carrying"],
                        event["moving_object"],
                        event["departures"],
                        event["arrivals"],
                    )
                    hits.extend({**h, "frame": first + h["frame"]} for h in found)
                check.update(
                    forbidden_contact_frames=len(hits),
                    first_forbidden_contacts=hits[:3],
                    simulator_seconds=seconds,
                    boundaries=[],
                )
                check["passed"] &= (
                    check["contacts"]["passed"]
                    and check["dynamics"]["passed"]
                    and not hits
                )
                io_timing.write_json(output / f"world{world}" / "replay_checks.json", check)
                reports[world]["checks"]["replay"] = check
                reports[world].update(
                    accepted=bool(check["passed"]),
                    status="accepted" if check["passed"] else "failed_replay",
                )
    summary = publish()
    summary.update(
        collection_simulator_seconds=group.simulator_seconds,
        collected_control_steps=group.control_steps,
        batch_wait_control_steps=group.wait_steps,
        post_terminal_guard_failure_chunks=group.wait_failures,
        terminal_padding_control_steps=group.padding_steps,
        replay_simulator_seconds=replay_seconds,
        replay_control_steps=replay_steps,
        replay_padding_control_steps=replay_padding,
        total_physics_steps=(group.control_steps + replay_steps)
        * round(0.02 / physics_timestep),
        accepted_action_targets=sum(
            len(forwards[w][1]["action"])
            for w in range(worlds)
            if reports[w]["accepted"]
        ),
        accepted_attempts_per_second=summary["accepted_attempts"]
        / summary["elapsed_seconds"],
    )
    io_timing.write_json(output / "summary.json", summary)
    return summary






def stage_worker(connection, packet):
    """Use the shared worker bootstrap with collection's selection policy."""
    session.stage_worker(connection, packet, runner_type=Runner)


class ConstructionGroup(session.ConstructionGroup):
    """Shared execution with the existing data-collection planner policy."""

    stage_worker = staticmethod(stage_worker)
