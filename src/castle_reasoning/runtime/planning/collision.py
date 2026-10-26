"""Runtime planning collision."""

import dataclasses

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


@dataclasses.dataclass(frozen=True)
class ContactViolation:
    names: tuple[str, str]
    distance: float
    required: float


def execution_departures(departures):
    """Use the standard 0.2 mm physical allowance at departing neighbors.

    Kinematic planning retains the measured initial-gap constraint. Physical
    guards and replay auditing share this tolerance so tiny compliant contacts
    do not halt an otherwise separating lift. Other contact classes are unchanged.
    """
    return tuple((other, min(gap, -0.0002)) for other, gap in departures)


class Scene:
    """Kinematic scene with explicit, bounded manipulation contact allowances."""

    def __init__(self, model, qpos, margin=0.002):
        self.model = model
        self.data = mujoco.MjData(model)
        self.reference = np.array(qpos, dtype=float).copy()
        self.margin = margin
        self.names = [model.geom(i).name for i in range(model.ngeom)]
        active = np.flatnonzero(model.geom_contype | model.geom_conaffinity)
        self.robot_geoms = {i for i in active if self.names[i].startswith("ur5e/")}
        self.pads = {i for i in self.robot_geoms if "_pad" in self.names[i]}
        self.objects = {
            int(n[7:]): i for i, n in enumerate(self.names) if n.startswith("object_")
        }
        exclusions = set(map(int, model.exclude_signature))
        pairs = []
        for a in active:
            for b in active:
                if b <= a:
                    continue
                ba, bb = int(model.geom_bodyid[a]), int(model.geom_bodyid[b])
                wa, wb = int(model.body_weldid[ba]), int(model.body_weldid[bb])
                if wa == wb or ((min(ba, bb) << 16) + max(ba, bb)) in exclusions:
                    continue
                # MuJoCo filters directly adjacent welded-body trees, except world.
                pa = int(model.body_weldid[model.body_parentid[wa]])
                pb = int(model.body_weldid[model.body_parentid[wb]])
                if wa != 0 and wb != 0 and (pa == wb or pb == wa):
                    continue
                if not (
                    (model.geom_contype[a] & model.geom_conaffinity[b])
                    or (model.geom_contype[b] & model.geom_conaffinity[a])
                ):
                    continue
                pairs.append((int(a), int(b)))
        self.pairs = np.asarray(pairs, dtype=int)
        self._robot_pairs = np.isin(self.pairs, list(self.robot_geoms)).any(axis=1)
        self.site = model.site("ur5e/robotiq/pinch").id
        self.queries = 0
        self.valid_clearance_lower_bound = 0.003

    def set_state(self, qpos, payload=None):
        """Set poses; payload is (part, pinch-relative position, rotation)."""
        self.data.qpos[:] = qpos
        mujoco.mj_kinematics(self.model, self.data)
        if payload is not None:
            part, offset, rotation = payload
            wrist = self.data.site_xmat[self.site].reshape(3, 3)
            start = 14 + 7 * part
            self.data.qpos[start : start + 3] = (
                self.data.site_xpos[self.site] + wrist @ offset
            )
            quat = Rotation.from_matrix(wrist @ rotation).as_quat()
            self.data.qpos[start + 3 : start + 7] = np.roll(quat, 1)
            mujoco.mj_kinematics(self.model, self.data)

    def allowed_pair(self, a, b, selected, pads_allowed, supports):
        """Only intended pads or explicitly declared payload supports may touch."""
        obj = self.objects.get(selected)
        if obj not in (a, b):
            return False
        other = b if a == obj else a
        return (pads_allowed and other in self.pads) or other in supports

    def violations(
        self,
        qpos,
        selected=None,
        pads_allowed=False,
        supports=(),
        payload=None,
        departures=(),
        arrivals=(),
    ):
        """Return forbidden clearances; intentional contacts have bounded overlap."""
        self.queries += 1
        self.set_state(qpos, payload)
        obj = self.objects.get(selected)
        pair_limits = dict((*departures, *arrivals)) if payload is not None else {}
        a, b = self.pairs.T
        relevant = self._robot_pairs
        if payload is not None and obj is not None:
            relevant = relevant | (a == obj) | (b == obj)
        distance = np.linalg.norm(
            self.data.geom_xpos[a] - self.data.geom_xpos[b], axis=1
        )
        plane = (self.model.geom_type[a] == 0) | (self.model.geom_type[b] == 0)
        near = plane | (
            distance
            < self.model.geom_rbound[a] + self.model.geom_rbound[b] + self.margin
        )
        violations = []
        # Sphere bounds cover pairs skipped by the narrow phase. Query results
        # are capped distances, hence this diagnostic is explicitly a lower bound.
        skipped = relevant & ~near
        clearance = 0.003
        if np.any(skipped):
            clearance = min(
                clearance,
                float(
                    np.min(
                        (
                            distance
                            - self.model.geom_rbound[a]
                            - self.model.geom_rbound[b]
                        )[skipped]
                    )
                ),
            )
        for a, b in self.pairs[relevant & near]:
            allowed = self.allowed_pair(a, b, selected, pads_allowed, supports)
            required = self.margin
            if allowed:
                other = b if a == obj else a
                required = -0.001 if other in self.pads else -0.0003
            elif obj in (a, b):
                other = b if a == obj else a
                required = pair_limits.get(other, required)
            gap = mujoco.mj_geomDistance(
                self.model, self.data, int(a), int(b), max(self.margin, 0.003), None
            )
            if not allowed:
                clearance = min(clearance, float(gap))
            if gap < required - 1e-8:
                violations.append(
                    ContactViolation(
                        (self.names[a], self.names[b]), float(gap), required
                    )
                )
        if not violations:
            self.valid_clearance_lower_bound = min(
                self.valid_clearance_lower_bound, clearance
            )
        return violations

    def valid(
        self,
        qpos,
        selected=None,
        pads_allowed=False,
        supports=(),
        payload=None,
        departures=(),
        arrivals=(),
    ):
        return not self.violations(
            qpos, selected, pads_allowed, supports, payload, departures, arrivals
        )

    def destination_contacts(self, qpos, selected, destination):
        """Bound intended object contacts at an upright commanded destination."""
        state = np.array(qpos, dtype=float).copy()
        start = 14 + 7 * selected
        state[start : start + 7] = np.r_[destination, 1, 0, 0, 0]
        self.set_state(state)
        obj = self.objects[selected]
        contacts = []
        for other in self.objects.values():
            if other == obj:
                continue
            gap = mujoco.mj_geomDistance(self.model, self.data, obj, other, 0.002, None)
            # Settled soft contacts must satisfy the same bound when qualifying
            # the destination as they do during the checked placement motion.
            if -0.0002 <= gap < 0.002:
                contacts.append((other, -0.0002))
        return tuple(contacts)

    def departure_contacts(self, qpos, selected):
        """Bound departure from initial object contacts without allowing intrusion."""
        self.set_state(qpos)
        obj = self.objects[selected]
        contacts = []
        for other in self.objects.values():
            if other == obj:
                continue
            gap = mujoco.mj_geomDistance(self.model, self.data, obj, other, 0.002, None)
            if -0.0003 <= gap < 0.002:
                # Positive clearance can shrink without creating penetration.
                contacts.append((other, min(0.0, float(gap)) - 0.00002))
        return tuple(contacts)

    def attachment(self, qpos, part):
        """Measure the complete rigid transform after a physical grasp."""
        self.set_state(qpos)
        wrist = self.data.site_xmat[self.site].reshape(3, 3)
        pose = np.asarray(qpos)[14 + 7 * part : 21 + 7 * part]
        rotation = Rotation.from_quat(np.roll(pose[3:], -1)).as_matrix()
        return (
            part,
            wrist.T @ (pose[:3] - self.data.site_xpos[self.site]),
            wrist.T @ rotation,
        )
