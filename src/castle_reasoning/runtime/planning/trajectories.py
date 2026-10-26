"""Runtime planning trajectories."""

import dataclasses

import numpy as np
import warp as wp


@dataclasses.dataclass(frozen=True)
class HermiteSample:
    """Position and physical-time derivative on a Hermite segment."""

    position: np.ndarray
    velocity: np.ndarray


def waypoint_velocities(
    points: np.ndarray, durations: np.ndarray
) -> np.ndarray:
    """Computes shared physical derivatives for a C1 waypoint path."""
    points = np.asarray(points, dtype=np.float64)
    durations = np.asarray(durations, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 2:
        raise ValueError(
            "Waypoints must have shape [waypoint >= 2, dimension]."
        )
    if durations.shape != (points.shape[0] - 1,):
        raise ValueError("One duration is required per waypoint segment.")
    if (
        not np.isfinite(points).all()
        or not np.isfinite(durations).all()
        or np.any(durations <= 0.0)
    ):
        raise ValueError("Waypoints must be finite and durations positive.")
    velocities = np.zeros_like(points)
    if points.shape[0] > 2:
        elapsed = durations[:-1] + durations[1:]
        velocities[1:-1] = (points[2:] - points[:-2]) / elapsed[:, None]
    return velocities


def hermite_segment(
    start: np.ndarray,
    end: np.ndarray,
    start_velocity: np.ndarray,
    end_velocity: np.ndarray,
    duration: float,
    time_seconds: float,
) -> HermiteSample:
    """Evaluates a cubic Hermite segment and its physical-time derivative."""
    arrays = tuple(
        np.asarray(value, dtype=np.float64)
        for value in (start, end, start_velocity, end_velocity)
    )
    if any(value.shape != arrays[0].shape for value in arrays[1:]):
        raise ValueError(
            "Hermite values and derivatives must have equal shape."
        )
    if not all(np.isfinite(value).all() for value in arrays):
        raise ValueError("Hermite inputs must be finite.")
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError("Hermite duration must be positive and finite.")
    if not np.isfinite(time_seconds):
        raise ValueError("Hermite time must be finite.")
    start, end, start_velocity, end_velocity = arrays
    normalized_time = np.clip(time_seconds / duration, 0.0, 1.0)
    t2 = normalized_time * normalized_time
    t3 = t2 * normalized_time
    position = (
        (2.0 * t3 - 3.0 * t2 + 1.0) * start
        + (t3 - 2.0 * t2 + normalized_time) * duration * start_velocity
        + (-2.0 * t3 + 3.0 * t2) * end
        + (t3 - t2) * duration * end_velocity
    )
    velocity = (
        (6.0 * t2 - 6.0 * normalized_time) * start / duration
        + (3.0 * t2 - 4.0 * normalized_time + 1.0) * start_velocity
        + (-6.0 * t2 + 6.0 * normalized_time) * end / duration
        + (3.0 * t2 - 2.0 * normalized_time) * end_velocity
    )
    return HermiteSample(position=position, velocity=velocity)


def quintic_smootherstep(value: np.ndarray | float) -> np.ndarray:
    """Returns a clipped C2 interpolation fraction with stationary endpoints."""
    value = np.asarray(value, dtype=np.float64)
    if not np.isfinite(value).all():
        raise ValueError("Smootherstep input must be finite.")
    clipped = np.clip(value, 0.0, 1.0)
    return clipped**3 * (clipped * (clipped * 6.0 - 15.0) + 10.0)


def smooth_orientation_reference(
    start_xyzw: np.ndarray,
    target_xyzw: np.ndarray,
    progress: float,
) -> np.ndarray:
    """Interpolates unit quaternions along the shortest smootherstep path."""
    start = np.asarray(start_xyzw, dtype=np.float64)
    target = np.asarray(target_xyzw, dtype=np.float64)
    if start.shape != (4,) or target.shape != (4,):
        raise ValueError("Orientation quaternions must have shape (4,).")
    if not np.isfinite(start).all() or not np.isfinite(target).all():
        raise ValueError("Orientation quaternions must be finite.")
    if not np.isfinite(progress):
        raise ValueError("Orientation progress must be finite.")
    start_norm = np.linalg.norm(start)
    target_norm = np.linalg.norm(target)
    if start_norm <= 0.0 or target_norm <= 0.0:
        raise ValueError("Orientation quaternions must be nonzero.")
    start = start / start_norm
    target = target / target_norm
    cosine = float(np.dot(start, target))
    if cosine < 0.0:
        target = -target
        cosine = -cosine
    fraction = float(quintic_smootherstep(progress))
    if cosine > 1.0 - 1e-8:
        result = (1.0 - fraction) * start + fraction * target
        return result / np.linalg.norm(result)
    angle = np.arccos(np.clip(cosine, -1.0, 1.0))
    denominator = np.sin(angle)
    result = (
        np.sin((1.0 - fraction) * angle) * start
        + np.sin(fraction * angle) * target
    ) / denominator
    return result / np.linalg.norm(result)


def approach_direction_from_rpy(target_rpy: np.ndarray) -> np.ndarray:
    """Returns the target gripper local-Z axis in world coordinates."""
    target_rpy = np.asarray(target_rpy, dtype=np.float64)
    if target_rpy.shape != (3,) or not np.isfinite(target_rpy).all():
        raise ValueError("Target RPY must be a finite three-vector.")
    roll, pitch, yaw = target_rpy
    sine_roll, cosine_roll = np.sin(roll), np.cos(roll)
    sine_pitch, cosine_pitch = np.sin(pitch), np.cos(pitch)
    sine_yaw, cosine_yaw = np.sin(yaw), np.cos(yaw)
    direction = np.asarray(
        [
            cosine_yaw * sine_pitch * cosine_roll + sine_yaw * sine_roll,
            sine_yaw * sine_pitch * cosine_roll - cosine_yaw * sine_roll,
            cosine_pitch * cosine_roll,
        ]
    )
    return direction / np.linalg.norm(direction)


@wp.func
def _waypoint_position_device(
    waypoint: int,
    effector_start: wp.vec3,
    block_start: wp.vec3,
    goal: wp.vec3,
    grasp_offset: wp.vec3,
    approach_direction: wp.vec3,
    approach_distance: float,
    clearance: float,
    goal_error: wp.vec3,
) -> wp.vec3:
    grasp = block_start + grasp_offset
    approach = grasp - approach_direction * approach_distance
    lift = grasp + wp.vec3(0.0, 0.0, clearance)
    placement = goal + goal_error + grasp_offset
    transit = placement + wp.vec3(0.0, 0.0, clearance)
    if waypoint == 0:
        return effector_start
    if waypoint == 1:
        return approach
    if waypoint == 4:
        return lift
    if waypoint == 2 or waypoint == 3:
        return grasp
    if waypoint == 5 or waypoint == 8:
        return transit
    return placement


@wp.func
def _waypoint_velocity_device(
    waypoint: int,
    effector_start: wp.vec3,
    block_start: wp.vec3,
    goal: wp.vec3,
    grasp_offset: wp.vec3,
    approach_direction: wp.vec3,
    approach_distance: float,
    clearance: float,
    goal_error: wp.vec3,
    duration_scale: float,
    phase_durations: wp.array(dtype=float),
) -> wp.vec3:
    if (
        waypoint == 0
        or waypoint == 2
        or waypoint == 3
        or waypoint == 6
        or waypoint == 7
        or waypoint == 8
    ):
        return wp.vec3(0.0)
    previous = _waypoint_position_device(
        waypoint - 1,
        effector_start,
        block_start,
        goal,
        grasp_offset,
        approach_direction,
        approach_distance,
        clearance,
        goal_error,
    )
    following = _waypoint_position_device(
        waypoint + 1,
        effector_start,
        block_start,
        goal,
        grasp_offset,
        approach_direction,
        approach_distance,
        clearance,
        goal_error,
    )
    elapsed = duration_scale * (
        phase_durations[waypoint - 1] + phase_durations[waypoint]
    )
    return (following - previous) / elapsed


@wp.func
def _hermite_device(
    start: wp.vec3,
    end: wp.vec3,
    start_velocity: wp.vec3,
    end_velocity: wp.vec3,
    duration: float,
    time_seconds: float,
) -> wp.spatial_vector:
    normalized_time = wp.clamp(time_seconds / duration, 0.0, 1.0)
    squared = normalized_time * normalized_time
    cubed = squared * normalized_time
    position = (
        (2.0 * cubed - 3.0 * squared + 1.0) * start
        + (cubed - 2.0 * squared + normalized_time) * duration * start_velocity
        + (-2.0 * cubed + 3.0 * squared) * end
        + (cubed - squared) * duration * end_velocity
    )
    velocity = (
        (6.0 * squared - 6.0 * normalized_time) * start / duration
        + (3.0 * squared - 4.0 * normalized_time + 1.0) * start_velocity
        + (-6.0 * squared + 6.0 * normalized_time) * end / duration
        + (3.0 * squared - 2.0 * normalized_time) * end_velocity
    )
    return wp.spatial_vector(position, velocity)


@wp.func
def _quintic_smootherstep_device(value: float) -> float:
    clipped = wp.clamp(value, 0.0, 1.0)
    return (
        clipped * clipped * clipped * (clipped * (clipped * 6.0 - 15.0) + 10.0)
    )
