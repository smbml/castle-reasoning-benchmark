"""Tests helpers column_collection."""

import json
import time
from pathlib import Path

import hydra
import mujoco
import numpy as np
from ogbench.manipspace.envs import cube_env
from omegaconf import OmegaConf

import castle_reasoning.data.assembly.observations as observations
import castle_reasoning.data.assembly.records as assembly_records
import castle_reasoning.data.assembly.review as review
import castle_reasoning.data.assembly.teacher as teacher
import castle_reasoning.runtime.execution.traces as traces
import castle_reasoning.runtime.rendering.replay as rendering_replay
import castle_reasoning.runtime.simulation.robot as robot
import tests.helpers.qualification_columns as qualification_columns
from castle_reasoning.runtime.diagnostics import io_timing


def make_model(block_count: int):
    """Keeps OGBench's post-compilation actuator edits and native home pose."""
    env = cube_env.CubeEnv(
        env_type={2: "double", 4: "quadruple", 8: "octuple"}[block_count],
        permute_blocks=False,
        physics_timestep=0.002,
        control_timestep=0.02,
        ob_type="states",
        visualize_info=False,
    )
    env.reset(seed=0, options={"task_id": 2})
    model = env.model
    # Hide target ghosts for raw goal/current images. They have no collisions.
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
        if "target" in name:
            model.geom_rgba[i, 3] = 0
    qpos = env.data.qpos.copy()
    qpos[:6] = robot.HOME
    qpos[6:14] = 0
    qpos[14:].reshape(block_count, 7)[:, 3:] = [1, 0, 0, 0]
    env.close()
    return model, qpos


def _hold(trace, world, goal):
    return qualification_columns.verify_hold(
        trace["qpos"][-51:, world], trace["qvel"][-51:, world], goal
    )


def _join(first, second):
    return {
        key: np.concatenate([first[key], second[key]])
        for key in ("qpos", "qvel", "action")
    }


def _stage(collector, goals, orders, initial, steps, home_steps, times, name):
    started = time.perf_counter()
    teacher.set_task(collector, goals, orders)
    collector.reset(qpos=initial["qpos"], qvel=initial["qvel"])
    reset = collector.engine.snapshot()
    metrics = collector.step(steps)
    trace = collector.trace()
    flags = collector.snapshot()
    returned, home_seconds = traces.run_controls(collector.controller, home_steps)
    times[name + "_simulator_seconds"] += metrics.seconds + home_seconds
    times[name + "_total_seconds"] += time.perf_counter() - started
    return _join(trace, returned), flags, reset


def collect(config):
    started = time.perf_counter()
    output = Path(config.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists():
        raise FileExistsError(
            "Use a new output directory; existing data is never overwritten."
        )
    io_timing.write_json(
        output / "config.json", OmegaConf.to_container(config, resolve=True)
    )
    io_timing.write_json(output / "provenance.json", assembly_records._provenance())
    model, home = make_model(config.block_count)
    mujoco.mj_saveModel(model, str(output / "model.mjb"), None)
    first = qualification_columns.sample_target(config.seed, 0, config.block_count)
    collector = teacher.make_collector(
        model,
        first.positions,
        config.worlds,
        config.max_control_steps,
        config.seed,
        config.device,
    )
    zero_velocity = np.zeros((config.worlds, model.nv))
    # Capture before timing warm collection. Setup is still included end-to-end.
    collector.reset(qpos=home, qvel=np.zeros(model.nv))
    setup_seconds = time.perf_counter() - started
    times = {
        key: 0.0
        for key in (
            "settle_simulator_seconds",
            "disassembly_simulator_seconds",
            "disassembly_total_seconds",
            "assembly_simulator_seconds",
            "assembly_total_seconds",
            "replay_simulator_seconds",
            "replay_total_seconds",
            "host_verification_seconds",
            "write_seconds",
        )
    }
    records, saved = [], []
    transitions = 0
    for batch in range(config.max_batches):
        candidates = [
            qualification_columns.sample_target(
                config.seed, batch * config.worlds + i, config.block_count
            )
            for i in range(config.worlds)
        ]
        goals = np.stack([c.positions for c in candidates])
        parking = np.broadcast_to(
            qualification_columns.parking_positions(config.block_count), goals.shape
        ).copy()
        qpos = np.broadcast_to(home, (config.worlds, model.nq)).copy()
        qpos[:, 14:].reshape(config.worlds, config.block_count, 7)[:, :, :3] = goals
        collector.reset(qpos=qpos, qvel=zero_velocity)
        settled, seconds = traces.run_controls(
            collector.controller, config.settle_control_steps
        )
        times["settle_simulator_seconds"] += seconds
        settled_checks = [_hold(settled, i, goals[i]) for i in range(config.worlds)]
        disassembly, dflags, disassembly_initial = _stage(
            collector,
            parking,
            [c.removal_order for c in candidates],
            {key: settled[key][-1] for key in ("qpos", "qvel")},
            config.max_control_steps,
            config.home_control_steps,
            times,
            "disassembly",
        )
        parked_checks = [
            _hold(disassembly, i, parking[i]) for i in range(config.worlds)
        ]
        assembled, aflags, initial = _stage(
            collector,
            goals,
            [c.assembly_order for c in candidates],
            {key: disassembly[key][-1] for key in ("qpos", "qvel")},
            config.max_control_steps,
            config.home_control_steps,
            times,
            "assembly",
        )
        assembled_checks = [_hold(assembled, i, goals[i]) for i in range(config.worlds)]
        replay_started = time.perf_counter()
        collector.reset(qpos=initial["qpos"], qvel=initial["qvel"])
        replay, seconds = traces.run_controls(
            collector.controller,
            len(assembled["action"]),
            mode="replay",
            actions=assembled["action"],
        )
        times["replay_simulator_seconds"] += seconds
        times["replay_total_seconds"] += time.perf_counter() - replay_started
        verify_started = time.perf_counter()
        batch_records = []
        for world, target in enumerate(candidates):
            replay_check = _hold(replay, world, goals[world])
            checks = dict(
                target_settle=settled_checks[world],
                disassembly=parked_checks[world],
                assembly=assembled_checks[world],
                replay=replay_check,
            )
            checks["disassembly"]["oracle_completed"] = bool(dflags["completed"][world])
            checks["assembly"]["oracle_completed"] = bool(aflags["completed"][world])
            rejection = next(
                (
                    key
                    for key, result in checks.items()
                    if not result["passed"] or not result.get("oracle_completed", True)
                ),
                None,
            )
            if rejection is None:
                checks["contacts"] = qualification_columns.contact_report(
                    model, replay["qpos"][:, world], goals[world], target.supports
                )
                if not checks["contacts"]["passed"]:
                    rejection = "contacts"
            record = dict(
                candidate=batch * config.worlds + world,
                seed=config.seed,
                accepted=rejection is None,
                rejection=rejection,
                checks=checks,
                heights=target.heights,
                fingerprint=target.fingerprint,
                assembly_order=target.assembly_order,
                goals=goals[world].tolist(),
            )
            batch_records.append(record)
        times["host_verification_seconds"] += time.perf_counter() - verify_started
        write_started = time.perf_counter()
        for world, record in enumerate(batch_records):
            records.append(record)
            if record["accepted"] and len(saved) < config.required:
                number = len(saved) + 1
                directory = output / f"instance{number:03d}"
                directory.mkdir()
                ep = traces.episode(initial, replay, world)
                np.savez_compressed(
                    directory / "trajectory.npz",
                    **ep,
                    initial_arm_target=initial["qpos"][world, :6],
                    requested_goal_xyz=goals[world],
                    goal_qpos=settled["qpos"][-1, world],
                    goal_qvel=settled["qvel"][-1, world],
                )
                dep = traces.episode(disassembly_initial, disassembly, world)
                np.savez_compressed(directory / "disassembly.npz", **dep)
                metadata = dict(
                    record,
                    instance=number,
                    supports=candidates[world].supports,
                    action_count=len(ep["action"]),
                    control_timestep_seconds=0.02,
                    action_semantics="six normalized joint velocities times pi rad/s, absolute gripper -1 open +1 closed",
                    source="MJ-Warp action replay",
                    goal_source="settled sampled target before disassembly",
                    initial_restore="reset_data; set qpos/qvel; forward; sync arm_target to qpos[:6]",
                    data_sha256=assembly_records.sha256(directory / "trajectory.npz"),
                )
                io_timing.write_json(directory / "metadata.json", metadata)
                saved.append(metadata)
                transitions += len(ep["action"])
        times["write_seconds"] += time.perf_counter() - write_started
        io_timing.write_json(output / "attempts.json", records)
        summary = assembly_records.yield_summary(
            records, len(saved), time.perf_counter() - started, transitions
        )
        summary.update(
            status="collecting",
            setup_seconds=setup_seconds,
            timers=times,
            completed_batches=batch + 1,
            fixed_worlds=config.worlds,
            requested_targets=config.required,
        )
        io_timing.write_json(output / "summary.json", summary)
        review.write_index(output)
        print(json.dumps(summary), flush=True)
        if len(saved) >= config.required:
            break
    generation_seconds = time.perf_counter() - started
    summary = assembly_records.yield_summary(records, len(saved), generation_seconds, transitions)
    summary.update(
        status="rendering" if len(saved) >= config.required else "insufficient_yield",
        setup_seconds=setup_seconds,
        timers=times,
        completed_batches=batch + 1,
        fixed_worlds=config.worlds,
        requested_targets=config.required,
        selected_video_indices=qualification_columns.video_indices(len(saved)),
    )
    io_timing.write_json(output / "summary.json", summary)
    review.write_index(output)
    if len(saved) < config.required:
        raise RuntimeError(
            f"Only {len(saved)}/{config.required} verified instances after bounded collection."
        )
    if config.render:
        render_started = time.perf_counter()
        for metadata in saved:
            number = metadata["instance"]
            directory = output / f"instance{number:03d}"
            ep = np.load(directory / "trajectory.npz", allow_pickle=False)
            seconds = rendering_replay.render_review(
                model,
                ep["qpos"],
                ep["qvel"],
                directory,
                video=number in qualification_columns.video_indices(config.required),
                goal_qpos=ep["goal_qpos"],
                goal_qvel=ep["goal_qvel"],
            )
            image_seconds = 0.0
            if config.learning_images:
                image_seconds = observations.render_learning_observations(
                    model,
                    ep["qpos"],
                    ep["qvel"],
                    directory,
                    config.learning_image_size,
                    goal_qpos=ep["goal_qpos"],
                    goal_qvel=ep["goal_qvel"],
                )
            metadata["review_render_seconds"] = seconds
            metadata["learning_image_seconds"] = image_seconds
            metadata["rgb_shape"] = [
                len(ep["qpos"]),
                2,
                config.learning_image_size,
                config.learning_image_size,
                3,
            ]
            io_timing.write_json(directory / "metadata.json", metadata)
            review.write_index(output)
            print(
                json.dumps(
                    dict(
                        rendered=number,
                        review_seconds=seconds,
                        image_seconds=image_seconds,
                    )
                ),
                flush=True,
            )
        summary["review_render_seconds"] = time.perf_counter() - render_started
    summary["end_to_end_seconds"] = time.perf_counter() - started
    summary["simulator_control_steps"] = (
        (batch + 1)
        * config.worlds
        * (
            config.settle_control_steps
            + 3 * (config.max_control_steps + config.home_control_steps)
        )
    )
    summary["simulator_physics_steps"] = 10 * summary["simulator_control_steps"]
    summary["simulator_seconds"] = sum(
        v for k, v in times.items() if k.endswith("_simulator_seconds")
    )
    summary["simulator_control_steps_per_second"] = (
        summary["simulator_control_steps"] / summary["simulator_seconds"]
    )
    summary["saved_action_targets_per_end_to_end_second"] = (
        transitions / summary["end_to_end_seconds"]
    )
    summary["structural_height_partitions"] = sorted(
        {tuple(sorted(m["heights"])) for m in saved}
    )
    summary["status"] = "complete"
    io_timing.write_json(output / "summary.json", summary)
    review.write_index(output)
    print(json.dumps(summary), flush=True)


@hydra.main(version_base=None, config_path=".", config_name="column_collection")
def main(config):
    collect(config)


if __name__ == "__main__":
    main()
