"""Runtime execution recovery."""

import time

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

import castle_reasoning.runtime.planning.arm as arm
import castle_reasoning.runtime.planning.manipulation as manipulation


class ExecutionGuard(arm.PlanningError):
    """Retain the interrupted segment and guard evidence for a bounded retry."""

    def __init__(self, segment, event):
        self.segment, self.event = segment, event
        super().__init__(
            f"execution_guard:{segment.name}:{event['failure']}:"
            f"contacts={event['forbidden_contact_frames']}"
        )


def retryable(failure, forbidden_frames):
    return failure in (1, 2, 3) and forbidden_frames == 0


def assess(planner, qpos, qvel, part, reference):
    """Check measured support, pad proximity, stability and original-scene drift."""
    poses = qpos[14:].reshape(-1, 7)
    old = reference[14:].reshape(-1, 7)
    velocities = qvel[14:].reshape(-1, 6)
    others = np.arange(len(poses)) != part
    displacement = np.linalg.norm(poses[:, :3] - old[:, :3], axis=1)
    angle = 2 * np.arccos(np.clip(np.abs((poses[:, 3:] * old[:, 3:]).sum(1)), 0, 1))
    rotation = Rotation.from_quat(np.roll(poses[:, 3:], -1, axis=1)).as_matrix()
    linear = float(np.linalg.norm(velocities[:, :3], axis=1).max())
    angular = float(np.linalg.norm(velocities[:, 3:], axis=1).max())
    neighbor_motion = float(displacement[others].max(initial=0))
    neighbor_angle = float(np.rad2deg(angle[others]).max(initial=0))
    supported = bool(planner._source_supports(qpos, part))
    scene = planner.scene
    scene.set_state(qpos)
    obj = scene.objects[part]
    touching = []
    for side in ("left", "right"):
        touching.append(
            any(
                mujoco.mj_geomDistance(scene.model, scene.data, obj, pad, 0.003, None)
                <= 0.001
                for pad in scene.pads
                if f"/{side}_" in scene.names[pad]
            )
        )
    held = all(touching)
    p = poses[part, :3]
    low, high = np.array([0.20, -0.33]), np.array([0.65, 0.33])
    floor = scene.model.geom("floor")
    if floor.type[0] == mujoco.mjtGeom.mjGEOM_BOX:
        # Finite workcells use the axis-aligned table compiled by cuboids.py.
        center = scene.data.geom_xpos[floor.id, :2]
        low, high = center - floor.size[:2], center + floor.size[:2]
    ready = (
        np.isfinite(qpos).all()
        and np.isfinite(qvel).all()
        and linear < 0.02
        and angular < 0.5
        and neighbor_motion <= 0.003
        and neighbor_angle <= 4
        and np.all(rotation[:, 2, 2] >= 0.985)
        and np.all((low <= p[:2]) & (p[:2] <= high))
        and 0 <= p[2] <= 0.4
        and (held or supported)
    )
    return {
        "ready": bool(ready),
        "supported": supported,
        "held": held,
        "linear_speed_m_s": linear,
        "angular_speed_rad_s": angular,
        "neighbor_displacement_m": neighbor_motion,
        "neighbor_angle_deg": neighbor_angle,
    }


def release_path(planner, qpos, part, width):
    """Open and retreat from a supported drop, checking the complete swept path."""
    scene = planner.scene
    scene.set_state(qpos)
    position = scene.data.site_xpos[scene.site].copy()
    rotation = scene.data.site_xmat[scene.site].reshape(3, 3).copy()
    opening = planner.aperture.closure(width + 0.012)
    supports = planner._source_supports(qpos, part)
    release = planner._cartesian(
        "recovery_open", qpos, position, rotation, opening, part, True, supports
    )
    high = position - 0.07 * rotation[:, 2]
    retreat = planner._cartesian(
        "recovery_retreat",
        release.path[-1],
        high,
        rotation,
        opening,
        part,
        True,
        supports,
    )
    release.moving_object = retreat.moving_object = True
    return [release, retreat]


def remaining_lift(planner, qpos, part, interrupted):
    """Replan extraction to its original endpoint using the actual held pose."""
    payload = planner.scene.attachment(qpos, part)
    planner.scene.set_state(interrupted.path[-1], interrupted.payload)
    position = planner.scene.data.site_xpos[planner.scene.site].copy()
    rotation = planner.scene.data.site_xmat[planner.scene.site].reshape(3, 3).copy()
    lift = planner._cartesian(
        "lift",
        qpos,
        position,
        rotation,
        interrupted.closure[-1],
        part,
        True,
        interrupted.supports,
        payload,
    )
    lift.reference_speed = interrupted.reference_speed
    return lift


def recover(runner, result, destination, reference, error):
    """One retry; any error here propagates without recursively retrying."""
    event = error.event
    if not retryable(event["failure"], event["forbidden_contact_frames"]):
        raise error
    part, planner = result.part, runner.planner
    record = {
        "part": part,
        "trigger": str(error),
        "outcome": "started",
        "assessments": [],
    }
    runner.recoveries.append(record)
    try:
        # Allow up to 10 simulated seconds, stopping at the first quiet state.
        # Each chunk is 0.4 seconds. Contact auditing stays enabled; only
        # motion/attachment expectations are suspended during this stop.
        for _ in range(25):
            qpos, _ = runner.state()
            supports = tuple(
                set(error.segment.supports) | set(planner._source_supports(qpos, part))
            )
            hold = manipulation.Segment(
                "recovery_settle",
                np.repeat(qpos[None], 11, axis=0),
                np.full(11, error.segment.closure[-1]),
                True,
                supports,
                moving_object=True,
            )
            runner.segment(hold, part, reference, monitor_scene=False)
            qpos, qvel = runner.state()
            check = assess(planner, qpos, qvel, part, reference)
            record["assessments"].append(check)
            if check["ready"]:
                break
        else:
            raise arm.PlanningError("recovery_state_not_safe")
        started = time.perf_counter()
        if check["held"] and not check["supported"]:
            # Preserve the measured grasp rather than dropping a held object.
            record["mode"] = "held_continuation"
            if error.segment.name == "lift":
                # Transit requires clearance; a stopped extraction may lack it.
                try:
                    with planner.budget.scope(planner.planning_seconds, runner.deadline):
                        lift = remaining_lift(planner, qpos, part, error.segment)
                finally:
                    runner.planning_seconds += time.perf_counter() - started
                runner.segment(lift, part, reference)
                record["resumed_lift"] = True
            runner.transport(
                part,
                destination,
                error.segment.closure[-1],
                result.segments[0].closure[-1],
                reference,
            )
        else:
            record["mode"] = "supported_regrasp"
            with planner.budget.scope(planner.planning_seconds, runner.deadline):
                segments = release_path(planner, qpos, part, result.grasp.width)
            runner.planning_seconds += time.perf_counter() - started
            for segment in segments:
                runner.segment(segment, part, reference)
            qpos, _ = runner.state()
            started = time.perf_counter()
            retry = planner.plan(
                qpos, part, destination, skip=(result.grasp,), deadline=runner.deadline
            )
            runner.planning_seconds += time.perf_counter() - started
            runner._move(retry, destination, reference)
        record["outcome"] = "completed"
    except arm.PlanningError as failure:
        record.update(outcome="failed", reason=str(failure))
        raise
