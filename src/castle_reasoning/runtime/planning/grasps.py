"""Runtime planning grasps."""

import dataclasses

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

NOMINAL_TIP_REACH = 0.018


@dataclasses.dataclass(frozen=True)
class Grasp:
    """Object-local pinch offset/orientation and required jaw width in meters."""

    offset: tuple[float, float, float]
    yaw: float
    width: float
    roll: float = np.pi

    def pose(self, object_pose):
        """World pinch pose from a MuJoCo XYZ/WXYZ object pose."""
        rotation = Rotation.from_quat(np.roll(object_pose[3:], -1))
        wrist = rotation * Rotation.from_euler("xyz", [self.roll, 0, self.yaw])
        return object_pose[:3] + rotation.apply(self.offset), wrist.as_matrix()


def propose(
    size,
    limit=16,
    aperture=0.070,
    pad_length=0.022,
    family="top_down",
    depth_fractions=(0.6, 0.35),
):
    """Enumerate both axes and offsets; limit applies separately to each family."""
    if family == "side":
        return side_proposals(size, limit, aperture, pad_length)
    if family == "all":
        top = propose(size, limit, aperture, pad_length)
        shallow = propose(size, limit, aperture, pad_length, depth_fractions=(0.15,))
        return (
            top
            + [g for g in shallow if g not in top]
            + exposed_edge_proposals(size, aperture, pad_length)
            + side_proposals(size, limit, aperture, pad_length)
        )
    if family != "top_down":
        raise ValueError(f"Unknown grasp family: {family}")
    size = np.asarray(size, dtype=float)
    result = []
    for depth_fraction in depth_fractions:
        for fraction in (0.0, -0.65, 0.65):
            for closing_axis in np.argsort(size[:2]):
                width = float(size[closing_axis])
                if width > aperture:
                    continue
                along = 1 - closing_axis
                # Keep at least half the pad overlapping the face at both ends.
                extent = max(0.0, size[along] / 2 - min(pad_length, size[along]) / 4)
                if fraction and extent < 0.002:
                    continue
                offset = np.zeros(3)
                offset[along] = fraction * extent
                offset[2] = (
                    size[2] / 2
                    + NOMINAL_TIP_REACH
                    - min(0.020, depth_fraction * size[2])
                )
                for flip in (0, np.pi):
                    yaw = (np.pi / 2 if closing_axis == 0 else 0) + flip
                    grasp = Grasp(tuple(offset), float(yaw), width)
                    if grasp not in result:
                        result.append(grasp)
    return result[:limit]


def exposed_edge_proposals(size, aperture=0.070, pad_length=0.022):
    """Offset a top pinch while retaining half the smaller contact face."""
    size = np.asarray(size, dtype=float)
    result = []
    for closing_axis in np.argsort(size[:2]):
        width = float(size[closing_axis])
        if width > aperture:
            continue
        along = 1 - closing_axis
        overlap = min(pad_length, size[along]) / 2
        extent = (pad_length + size[along]) / 2 - overlap
        for sign in (-1, 1):
            offset = np.zeros(3)
            offset[along] = sign * extent
            offset[2] = size[2] / 2 + NOMINAL_TIP_REACH - min(0.020, 0.15 * size[2])
            for flip in (0, np.pi):
                yaw = (np.pi / 2 if closing_axis == 0 else 0) + flip
                result.append(Grasp(tuple(offset), float(yaw), width))
    return result


def side_proposals(
    size, limit=16, aperture=0.070, pad_length=0.022, insertion_depth=0.012
):
    """Pinch across Z from horizontal edges; scene checks determine clearance."""
    size = np.asarray(size, dtype=float)
    if not np.isfinite(insertion_depth) or insertion_depth <= 0:
        raise ValueError("Side insertion depth must be finite and positive.")
    if size[2] > aperture:
        return []
    result = []
    for fraction in (0.0, -0.55, 0.55):
        for yaw in (0.0, np.pi / 2, np.pi, 3 * np.pi / 2):
            for roll in (np.pi / 2, -np.pi / 2):
                rotation = Rotation.from_euler("xyz", [roll, 0, yaw]).as_matrix()
                direction = rotation[:, 2]
                axis = int(np.argmax(np.abs(direction)))
                along = 1 - axis
                depth = min(insertion_depth, 0.6 * size[axis])
                offset = -direction * (size[axis] / 2 + NOMINAL_TIP_REACH - depth)
                extent = max(0.0, size[along] / 2 - pad_length / 2)
                if fraction and extent < 0.002:
                    continue
                offset[along] += fraction * extent
                result.append(Grasp(tuple(offset), yaw, float(size[2]), roll))
    return result[:limit]


class Aperture:
    """Invert the stock linkage geometry without advancing native physics."""

    def __init__(self, model):
        state = mujoco.MjData(model)
        site = model.site("ur5e/robotiq/pinch").id
        self.closures = np.linspace(0, 1, 101)
        gaps = []
        tips = []
        for closure in self.closures:
            state.qpos[6:14] = self.joints(closure)
            mujoco.mj_kinematics(model, state)
            wrist = state.site_xmat[site].reshape(3, 3)
            surfaces = []
            reaches = []
            for side in ("right", "left"):
                geom = model.geom(f"ur5e/robotiq/{side}_pad1").id
                corners = (
                    np.array(
                        [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
                    )
                    * model.geom_size[geom]
                )
                world = (
                    corners @ state.geom_xmat[geom].reshape(3, 3).T
                    + state.geom_xpos[geom]
                )
                local = (world - state.site_xpos[site]) @ wrist
                reaches.append(local[:, 2].max())
                surfaces.append(
                    local[:, 1].min() if side == "right" else local[:, 1].max()
                )
            gaps.append(surfaces[0] - surfaces[1])
            tips.append(min(reaches))
        self.gaps = np.array(gaps)
        self.tips = np.array(tips)

    def calibrate(self, grasp):
        """Preserve insertion depth as the fingertips retract at wide openings."""
        closure = self.closure(grasp.width)
        reach = np.interp(closure, self.closures, self.tips)
        direction = Rotation.from_euler("xyz", [grasp.roll, 0, grasp.yaw]).as_matrix()[
            :, 2
        ]
        offset = np.array(grasp.offset) + direction * (NOMINAL_TIP_REACH - reach)
        return dataclasses.replace(grasp, offset=tuple(offset))

    @staticmethod
    def joints(closure):
        theta = closure * 0.8
        return np.array([theta, 0, theta, -theta] * 2)

    def closure(self, width):
        if not self.gaps.min() <= width <= self.gaps.max():
            raise ValueError(f"Aperture {width} outside measured linkage range")
        return float(np.interp(width, self.gaps[::-1], self.closures[::-1]))
