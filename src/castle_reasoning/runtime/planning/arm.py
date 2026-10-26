"""Runtime planning arm."""

import time

import mujoco
import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation

import castle_reasoning.runtime.diagnostics.timing as timing


class PlanningError(RuntimeError):
    """No qualified path was found under the specified limits."""


class PlanningBudgetError(PlanningError):
    """Search was interrupted; this is not a geometric rejection."""


def edge_valid(start, goal, valid, resolution=0.005):
    """Check endpoints and uniformly subdivide every joint-space edge."""
    steps = max(1, int(np.ceil(np.max(np.abs(goal - start)) / resolution)))
    # Midpoint first rejects crossing edges cheaply without losing endpoint checks.
    if not valid((start + goal) / 2):
        return False
    return all(valid(start + (goal - start) * i / steps) for i in range(steps + 1))


@timing.measured('planning.connect')
def connect(start, goal, valid, bounds, seed=0, max_nodes=4096, deadline=None,
            check_budget=None):
    """RRT-Connect; every extension, connection, and returned edge is checked."""
    original_valid = valid

    def valid(point):
        if check_budget is not None:
            check_budget()
        return original_valid(point)

    start, goal = np.asarray(start), np.asarray(goal)
    if not valid(start) or not valid(goal):
        raise PlanningError("invalid_endpoint")
    if edge_valid(start, goal, valid):
        return np.asarray([start, goal])
    rng = np.random.default_rng(seed)
    trees, parents = [[start], [goal]], [[-1], [-1]]

    def extend(side, target):
        nodes = trees[side]
        nearest = int(np.argmin(np.linalg.norm(np.asarray(nodes) - target, axis=1)))
        delta = target - nodes[nearest]
        point = nodes[nearest] + delta * min(
            1.0, 0.15 / max(np.linalg.norm(delta), 1e-12)
        )
        if not edge_valid(nodes[nearest], point, valid):
            return None
        nodes.append(point)
        parents[side].append(nearest)
        return len(nodes) - 1

    def branch(side, index):
        path = []
        while index >= 0:
            path.append(trees[side][index])
            index = parents[side][index]
        return path[::-1]

    for iteration in range(max_nodes):
        if deadline is not None and time.perf_counter() > deadline:
            break
        side = iteration % 2
        sample = (
            trees[1 - side][-1]
            if rng.random() < 0.2
            else rng.uniform(bounds[:, 0], bounds[:, 1])
        )
        index = extend(side, sample)
        if index is None:
            continue
        point = trees[side][index]
        for _ in range(100):
            if sum(map(len, trees)) >= max_nodes or (
                deadline is not None and time.perf_counter() > deadline
            ):
                break
            other = extend(1 - side, point)
            if other is None:
                break
            if np.linalg.norm(trees[1 - side][other] - point) < 1e-9:
                a, b = branch(side, index), branch(1 - side, other)
                return np.asarray(a + b[-2::-1] if side == 0 else b + a[-2::-1])
    raise PlanningError("motion_search_budget")


def inverse_kinematics(scene, position, rotation, seed, max_iters=100):
    """Damped least squares using six arm columns of the full-model Jacobian."""
    qpos = np.array(seed, dtype=float).copy()
    jacp, jacr = np.zeros((3, scene.model.nv)), np.zeros((3, scene.model.nv))
    limits = scene.model.jnt_range[:6]
    qpos[:6] = np.clip(qpos[:6], limits[:, 0], limits[:, 1])
    for _ in range(max_iters):
        scene.set_state(qpos)
        mujoco.mj_comPos(scene.model, scene.data)
        actual = scene.data.site_xmat[scene.site].reshape(3, 3)
        error = np.r_[
            position - scene.data.site_xpos[scene.site],
            Rotation.from_matrix(rotation @ actual.T).as_rotvec(),
        ]
        if np.linalg.norm(error[:3]) < 0.0002 and np.linalg.norm(error[3:]) < 0.002:
            return qpos
        mujoco.mj_jacSite(scene.model, scene.data, jacp, jacr, scene.site)
        jac = np.vstack([jacp[:, :6], jacr[:, :6]])
        delta = jac.T @ np.linalg.solve(jac @ jac.T + np.eye(6) * 1e-5, error)
        delta *= min(1.0, 0.15 / max(np.max(np.abs(delta)), 1e-12))
        qpos[:6] = np.clip(qpos[:6] + delta, limits[:, 0], limits[:, 1])
    raise PlanningError("ik_residual")


def damped_least_squares(
    jacobian: np.ndarray, error: np.ndarray, *, damping: float
) -> np.ndarray:
    """Solves `J.T (J J.T + damping I)^-1 error`."""
    jacobian = np.asarray(jacobian, dtype=np.float64)
    error = np.asarray(error, dtype=np.float64)
    if jacobian.ndim != 2 or error.shape != (jacobian.shape[0],):
        raise ValueError("Jacobian and task error shapes are incompatible.")
    if not np.isfinite(jacobian).all() or not np.isfinite(error).all():
        raise ValueError("Damped least-squares inputs must be finite.")
    if not np.isfinite(damping) or damping <= 0.0:
        raise ValueError("Damping must be positive and finite.")
    regularized = jacobian @ jacobian.T + damping * np.eye(jacobian.shape[0])
    return jacobian.T @ np.linalg.solve(regularized, error)


@wp.func
def _solve_spd_six(
    matrix: wp.spatial_matrix, right_hand_side: wp.spatial_vector
) -> wp.spatial_vector:
    """Solves a 6x6 SPD system with an in-register Cholesky factor."""
    lower = wp.spatial_matrix(0.0)
    for row in range(6):
        for column in range(6):
            if column <= row:
                residual = matrix[row, column]
                for inner in range(6):
                    if inner < column:
                        residual = residual - (
                            lower[row, inner] * lower[column, inner]
                        )
                if row == column:
                    lower[row, column] = wp.sqrt(wp.max(residual, 1.0e-12))
                else:
                    lower[row, column] = residual / lower[column, column]
    intermediate = wp.spatial_vector()
    for row in range(6):
        residual = right_hand_side[row]
        for column in range(6):
            if column < row:
                residual = residual - lower[row, column] * intermediate[column]
        intermediate[row] = residual / lower[row, row]
    result = wp.spatial_vector()
    for reverse_row in range(6):
        row = 5 - reverse_row
        residual = intermediate[row]
        for column in range(6):
            if column > row:
                residual = residual - lower[column, row] * result[column]
        result[row] = residual / lower[row, row]
    return result


@wp.func
def _damped_least_squares_six(
    jacobian: wp.spatial_matrix,
    error: wp.spatial_vector,
    damping: float,
) -> wp.spatial_vector:
    regularized = jacobian * wp.transpose(jacobian)
    for diagonal in range(6):
        regularized[diagonal, diagonal] = (
            regularized[diagonal, diagonal] + damping
        )
    task_update = _solve_spd_six(regularized, error)
    return wp.transpose(jacobian) * task_update


@wp.kernel
def _solve_damped_least_squares(
    jacobian: wp.array3d(dtype=float),
    error: wp.array2d(dtype=float),
    damping: float,
    output: wp.array2d(dtype=float),
):
    world = wp.tid()
    local_jacobian = wp.spatial_matrix(0.0)
    local_error = wp.spatial_vector()
    for row in range(6):
        local_error[row] = error[world, row]
        for column in range(6):
            local_jacobian[row, column] = jacobian[world, row, column]
    result = _damped_least_squares_six(local_jacobian, local_error, damping)
    for column in range(6):
        output[world, column] = result[column]


def device_damped_least_squares(
    jacobian: np.ndarray,
    error: np.ndarray,
    *,
    damping: float,
    device: str,
) -> np.ndarray:
    """Executes the oracle's fixed-size DLS solver on a selected device."""
    jacobian = np.asarray(jacobian, dtype=np.float32)
    error = np.asarray(error, dtype=np.float32)
    if jacobian.ndim != 3 or jacobian.shape[1:] != (6, 6):
        raise ValueError("Device Jacobians must have shape [world, 6, 6].")
    if error.shape != (jacobian.shape[0], 6):
        raise ValueError("Device errors must have shape [world, 6].")
    if not np.isfinite(jacobian).all() or not np.isfinite(error).all():
        raise ValueError("Device DLS inputs must be finite.")
    if not np.isfinite(damping) or damping <= 0.0:
        raise ValueError("Damping must be positive and finite.")
    selected_device = wp.get_device(device)
    with wp.ScopedDevice(selected_device):
        device_jacobian = wp.array(jacobian, dtype=float)
        device_error = wp.array(error, dtype=float)
        output = wp.zeros((jacobian.shape[0], 6), dtype=float)
        wp.launch(
            _solve_damped_least_squares,
            dim=jacobian.shape[0],
            inputs=[device_jacobian, device_error, damping, output],
        )
        wp.synchronize_device(selected_device)
        return output.numpy()


@wp.kernel
def _pinch_jacobian(
    site_xpos: wp.array2d(dtype=wp.vec3),
    subtree_com: wp.array2d(dtype=wp.vec3),
    cdof: wp.array2d(dtype=wp.spatial_vector),
    site_id: int,
    root_body_id: int,
    jacobian: wp.array3d(dtype=float),
):
    world, dof = wp.tid()
    motion = cdof[world, dof]
    angular = wp.spatial_top(motion)
    linear = wp.spatial_bottom(motion)
    offset = site_xpos[world, site_id] - subtree_com[world, root_body_id]
    position = linear + wp.cross(angular, offset)
    jacobian[world, 0, dof] = position[0]
    jacobian[world, 1, dof] = position[1]
    jacobian[world, 2, dof] = position[2]
    jacobian[world, 3, dof] = angular[0]
    jacobian[world, 4, dof] = angular[1]
    jacobian[world, 5, dof] = angular[2]


@wp.func
def _orientation_error(current: wp.mat33, desired_quat: wp.quat) -> wp.vec3:
    desired = wp.quat_to_matrix(desired_quat)
    current_x = wp.vec3(current[0, 0], current[1, 0], current[2, 0])
    current_y = wp.vec3(current[0, 1], current[1, 1], current[2, 1])
    current_z = wp.vec3(current[0, 2], current[1, 2], current[2, 2])
    desired_x = wp.vec3(desired[0, 0], desired[1, 0], desired[2, 0])
    desired_y = wp.vec3(desired[0, 1], desired[1, 1], desired[2, 1])
    desired_z = wp.vec3(desired[0, 2], desired[1, 2], desired[2, 2])
    return 0.5 * (
        wp.cross(current_x, desired_x)
        + wp.cross(current_y, desired_y)
        + wp.cross(current_z, desired_z)
    )
