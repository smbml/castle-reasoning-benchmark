"""Data assembly collect."""

import dataclasses
import time
from pathlib import Path

import mujoco
import numpy as np

import castle_reasoning.data.assembly.records as records
import castle_reasoning.data.assembly.teacher as teacher
import castle_reasoning.data.generation.waypoints as waypoints
import castle_reasoning.data.plans as plans
import castle_reasoning.runtime.control.guards as guards
import castle_reasoning.runtime.control.tracking as tracking
import castle_reasoning.runtime.diagnostics.io_timing as io_timing
import castle_reasoning.runtime.execution.runner as execution_runner
import castle_reasoning.runtime.execution.success as success
import castle_reasoning.runtime.execution.traces as traces
import castle_reasoning.runtime.planning.manipulation as manipulation
import castle_reasoning.runtime.simulation.models as models


def execute(
    plan,
    output,
    target_id,
    planning_seconds=5.0,
    wall_seconds=400.0,
    seed=0,
    shuffle_grasps=True,
    physics_timestep=0.0005,
):
    """Fresh disassembly, reconstruction and independent recorded-action replay."""
    if not all(np.isfinite(v) and v > 0 for v in (planning_seconds, wall_seconds)):
        raise ValueError("Planning and wall-time budgets must be finite and positive.")
    config = plans.control_config(physics_timestep)
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    io_timing.write_json(output / "plan.json", plan)
    io_timing.write_json(output / "provenance.json", records._provenance())
    edge = plan.get("edge_access")
    sizes, goals = np.array(plan["sizes"]), np.array(plan["target"])
    model, home = models.make_model(
        sizes, goals, table_bounds=edge["table_bounds"] if edge else None
    )
    if edge is not None:
        model.geom("floor").solref[:] = [edge["table_contact_seconds"], 1]
    model.opt.timestep = config.physics_timestep
    model.actuator_forcerange[6] = [-config.grip_effort, config.grip_effort]
    mujoco.mj_saveModel(model, str(output / "model.mjb"), None)
    io_timing.savez_compressed(
        output / "requested_target.npz", qpos=home, sizes=sizes, goals=goals
    )
    io_timing.write_json(
        output / "execution_config.json",
        {
            "controller": "planned",
            "seed": seed,
            "grasp_order": "shuffled" if shuffle_grasps else "ordered",
            "grasp_family": "all",
            "edge_access": edge,
            "buffer_capacities": dataclasses.asdict(models.scene_capacities(model)),
            "waypoint_source_sha256": records.sha256(Path(waypoints.__file__)),
            "control_timestep": 0.02,
            "physics_timestep": model.opt.timestep,
            "physics_runtime_overrides": [],
            "physics_disableflags": int(model.opt.disableflags),
            "physics_cone_curvature": "installed_mujoco_warp",
            "physics_contact_geometry": "installed_mujoco_warp",
            "planning_seconds": planning_seconds,
            "wall_seconds": wall_seconds,
            "collision_margin": 0.002,
            "reference_speed_rad_s": 0.3,
            "grip_effort": config.grip_effort,
            "grip_effort_policy": "per part; plate force only for designated edge plates",
            "servo_feedforward": "native RNE gravity/Coriolis plus reference-velocity damping",
            "guard_scene_displacement_m": 0.003,
            "guard_scene_orientation_deg": 4,
            "forbidden_contact_penetration_m": 0.0002,
            "allowed_pad_penetration_m": 0.001,
            "allowed_support_penetration_m": 0.0003,
            "placement_support_clearance_m": 0.0001,
            "placement_contact_eligibility_m": -0.0002,
            "placement_contact_limit_m": -0.0002,
            "placement_attachment": "remeasured after place, before setdown",
        },
    )
    collector = teacher.make_collector(model, goals, 1, 1, seed)
    collector.controller = tracking.PathControl(collector)
    collector.reset(qpos=home, qvel=np.zeros(model.nv))
    runner = Runner(
        collector,
        plan,
        output,
        planning_seconds,
        started + wall_seconds,
        seed=seed,
        shuffle_grasps=shuffle_grasps,
    )
    report = {
        "target_id": target_id,
        "part_count": len(goals),
        "accepted": False,
        "checks": {},
    }

    def save(status):
        report.update(status=status, elapsed_seconds=time.perf_counter() - started)
        io_timing.write_json(output / "summary.json", report)

    initial = collector.engine.snapshot()
    trace, metrics = tracking.execute_segment(
        collector.controller,
        manipulation.hold_segment(home, 250),
        runner.planner.scene,
        -1,
        guard=False,
    )
    settled = traces.episode(initial, trace, 0)
    io_timing.savez_compressed(output / "settle.npz", **settled)
    check = (
        success.verify_hold(settled["qpos"], settled["qvel"], goals, sizes)
        if len(settled["qpos"]) >= 51
        else {"passed": False, "reason": "incomplete_settling_trace"}
    )
    check["execution"] = metrics
    check["contacts"] = success.contact_report(model, settled["qpos"], plan["supports"])
    check["passed"] &= check["contacts"]["passed"] and not metrics["failure"]
    report["checks"]["settle"] = check
    save("settled" if check["passed"] else "failed_settle")
    if not check["passed"]:
        return report
    backward, episode = runner.stage("disassembly", np.array(plan["parking"]))
    backward["contacts"] = success.contact_report(
        model, episode["qpos"], plan["parking_supports"]
    )
    backward["passed"] &= backward["contacts"]["passed"]
    io_timing.write_json(output / "disassembly_checks.json", backward)
    report["checks"]["disassembly"] = {
        k: v
        for k, v in backward.items()
        if k not in ("boundaries", "segments", "planning_attempts")
    }
    save(
        "disassembled"
        if backward["passed"]
        else "budget_exhausted_disassembly"
        if backward.get("reason") == "experiment_wall_budget"
        else "failed_disassembly"
    )
    if not backward["passed"]:
        return report
    io_timing.write_json(
        output / "execution_order.json",
        {
            "disassembly": backward["completed_order"],
            "assembly": backward["completed_order"][::-1],
        },
    )
    forward, episode = runner.stage(
        "assembly", goals, backward["completed_order"][::-1]
    )
    forward["contacts"] = success.contact_report(
        model, episode["qpos"], plan["supports"]
    )
    forward["passed"] &= forward["contacts"]["passed"]
    io_timing.write_json(output / "assembly_checks.json", forward)
    report["checks"]["assembly"] = {
        k: v
        for k, v in forward.items()
        if k not in ("boundaries", "segments", "planning_attempts")
    }
    save(
        "assembled"
        if forward["passed"]
        else "budget_exhausted_assembly"
        if forward.get("reason") == "experiment_wall_budget"
        else "failed_assembly"
    )
    if not forward["passed"]:
        return report
    if runner.deadline - time.perf_counter() < forward["simulator_seconds"] * 1.15:
        save("budget_exhausted_before_replay")
        return report
    collector.reset(qpos=episode["qpos"][0], qvel=episode["qvel"][0])
    initial = collector.engine.snapshot()
    trace, seconds = traces.run_controls(
        collector.controller,
        len(episode["action"]),
        mode="replay",
        actions=episode["action"][:, None],
        grip_efforts=episode["grip_effort"],
    )
    replay = traces.episode(initial, trace, 0)
    replay["grip_effort"] = episode["grip_effort"].copy()
    io_timing.savez_compressed(output / "replay.npz", **replay)
    check = success.verify_hold(replay["qpos"], replay["qvel"], goals, sizes)
    check["contacts"] = success.contact_report(model, replay["qpos"], plan["supports"])
    check["dynamics"] = guards.motion_check(replay, len(goals))
    replay_violations = []
    for event in forward["segments"]:
        first = event["start_action"] + 1
        last = first + event["control_steps"]
        hits = guards.audit_contacts(
            runner.audit_scene,
            replay["qpos"][first:last],
            event["part"],
            event["pads_allowed"],
            event["supports"],
            event["carrying"],
            event["moving_object"],
            event["departures"],
            event["arrivals"],
        )
        replay_violations.extend({**hit, "frame": first + hit["frame"]} for hit in hits)
    check["forbidden_contact_frames"] = len(replay_violations)
    check["first_forbidden_contacts"] = replay_violations[:3]
    check["passed"] &= (
        check["contacts"]["passed"]
        and check["dynamics"]["passed"]
        and not replay_violations
    )
    check.update(simulator_seconds=seconds, boundaries=[])
    report["checks"]["replay"] = check
    io_timing.write_json(output / "replay_checks.json", check)
    report["accepted"] = check["passed"]
    save("accepted" if check["passed"] else "failed_replay")
    return report




class Runner(execution_runner.Runner):
    """Requested-stage execution with the benchmark's automatic part selection."""

    def removable(self, remaining):
        """Yield the same statically eligible candidates used by collection."""
        return plans.removable(self.plan, remaining)
