"""Runtime control guards."""

import dataclasses

import numpy as np
import warp as wp

import castle_reasoning.runtime.diagnostics.timing as timing
import castle_reasoning.runtime.planning.collision as collision


def motion_check(episode, parts):
    """Reject high-energy contact failures even if a final pose is recovered."""
    velocities = np.asarray(episode["qvel"])
    omega = float(
        np.linalg.norm(velocities[:, 14:].reshape(-1, parts, 6)[..., 3:], axis=-1).max()
    )
    joint_speed = float(np.abs(velocities[:, :6]).max())
    return {
        "peak_object_angular_speed": omega,
        "peak_joint_speed": joint_speed,
        "passed": bool(
            np.isfinite(velocities).all() and omega <= 20 and joint_speed <= 4
        ),
    }


@wp.kernel
def contact_guard(
    count: wp.array(dtype=int),
    pairs: wp.array(dtype=wp.vec2i),
    worldid: wp.array(dtype=int),
    distance: wp.array(dtype=float),
    allowed_distance: wp.array2d(dtype=float),
    failure: wp.array(dtype=int),
    step: wp.array(dtype=int),
    first_step: wp.array(dtype=int),
    first_distances: wp.array3d(dtype=float),
):
    i = wp.tid()
    if i < count[0]:
        pair = pairs[i]
        if pair[0] >= 0 and pair[1] >= 0:  # noqa: SIM102 - guard array indexing.
            if distance[i] < allowed_distance[pair[0], pair[1]]:
                wp.atomic_max(failure, worldid[i], 4)
                previous_step = wp.atomic_min(first_step, worldid[i], step[0])
                if previous_step >= step[0]:
                    wp.atomic_min(
                        first_distances, worldid[i], pair[0], pair[1], distance[i]
                    )


def contact_evidence(model, first_steps, distances, allowed):
    """Decode only contacts from the first violating control index per world."""
    return [
        {
            "world": int(world),
            "control_index": int(first_steps[world]),
            "geom1": model.geom(int(a)).name,
            "geom2": model.geom(int(b)).name,
            "distance_m": float(distances[world, a, b]),
            "allowed_distance_m": float(allowed[a, b]),
        }
        for world, a, b in np.argwhere(np.isfinite(distances))
    ]


@wp.kernel
def attachment_guard(
    qpos: wp.array2d(dtype=float),
    positions: wp.array2d(dtype=wp.vec3),
    rotations: wp.array2d(dtype=wp.mat33),
    part: int,
    site: int,
    offset: wp.vec3,
    rotation: wp.mat33,
    failure: wp.array(dtype=int),
):
    world = wp.tid()
    p = 14 + 7 * part
    center = wp.vec3(qpos[world, p], qpos[world, p + 1], qpos[world, p + 2])
    relative = wp.transpose(rotations[world, site]) * (center - positions[world, site])
    quat = wp.quat(
        qpos[world, p + 4], qpos[world, p + 5], qpos[world, p + 6], qpos[world, p + 3]
    )
    actual = wp.transpose(rotations[world, site]) * wp.quat_to_matrix(quat)
    error = float(0.0)  # noqa: UP018 - Warp dynamic loop accumulator.
    for i in range(3):
        for j in range(3):
            error += (actual[i, j] - rotation[i, j]) ** 2.0
    if wp.length(relative - offset) > 0.006 or error > 0.05:
        wp.atomic_max(failure, world, 3)


@timing.measured('prepare.contacts')
def contact_limits(scene, segment, part):
    """Contact allowances distinguish pushing from a rigidly held payload."""
    allowed = np.full((scene.model.ngeom, scene.model.ngeom), -1e6, np.float32)
    obj = scene.objects.get(part)
    pair_limits = dict(
        (*collision.execution_departures(segment.departures), *segment.arrivals)
    )
    for a, b in scene.pairs:
        if (
            a not in scene.robot_geoms
            and b not in scene.robot_geoms
            and not (
                (segment.payload is not None or segment.moving_object) and obj in (a, b)
            )
        ):
            continue
        value = -0.0002
        if scene.allowed_pair(a, b, part, segment.pads_allowed, segment.supports):
            value = -0.001 if (b if a == obj else a) in scene.pads else -0.0003
        elif (segment.payload is not None or segment.moving_object) and obj in (a, b):
            value = pair_limits.get(b if a == obj else a, value)
        allowed[a, b] = allowed[b, a] = value
    return allowed


@timing.measured('audit.contacts')
def audit_contacts(
    scene,
    positions,
    part,
    pads_allowed,
    supports,
    carrying,
    moving_object=False,
    departures=(),
    arrivals=(),
):
    """Audit recorded poses independently of live guard status or final success."""
    departures = collision.execution_departures(departures)
    violations = []
    for i, qpos in enumerate(positions):
        payload = scene.attachment(qpos, part) if carrying or moving_object else None
        hits = scene.violations(
            qpos, part, pads_allowed, supports, payload, departures, arrivals
        )
        if hits:
            violations.append(
                {"frame": i, "contacts": [dataclasses.asdict(h) for h in hits]}
            )
    return violations
