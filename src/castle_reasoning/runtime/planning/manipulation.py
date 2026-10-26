"""Runtime planning manipulation."""

import dataclasses
import itertools
import time

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

import castle_reasoning.runtime.diagnostics.timing as timing
import castle_reasoning.runtime.planning.arm as planning_arm
import castle_reasoning.runtime.planning.budget as budget
import castle_reasoning.runtime.planning.collision as collision
import castle_reasoning.runtime.planning.grasps as grasps


@dataclasses.dataclass
class Segment:
    name: str
    path: np.ndarray
    closure: np.ndarray
    pads_allowed: bool = False
    supports: tuple[int, ...] = ()
    payload: tuple | None = None
    moving_object: bool = False
    reference_speed: float = 0.3
    departures: tuple[tuple[int, float], ...] = ()
    arrivals: tuple[tuple[int, float], ...] = ()


@dataclasses.dataclass
class Manipulation:
    part: int
    grasp: grasps.Grasp
    segments: list[Segment]
    planning_seconds: float
    query_count: int
    attempts: list[dict]
    minimum_valid_clearance_m: float


class Planner:
    """Bounded planning against a concrete scene, with no simulator stepping."""

    def __init__(
        self,
        model,
        sizes,
        planning_seconds=5.0,
        margin=0.002,
        seed=0,
        shuffle_grasps=False,
        grasp_family="top_down",
        active_part_count=None,
    ):
        self.scene = collision.Scene(model, model.qpos0, margin)
        self.sizes = np.asarray(sizes)
        self.active_part_count = (
            len(sizes) if active_part_count is None else active_part_count
        )
        self.aperture = grasps.Aperture(model)
        self.planning_seconds = planning_seconds
        self.rng = np.random.default_rng(seed)
        self.shuffle_grasps = shuffle_grasps
        self.grasp_family = grasp_family
        self.budget = budget.Budget()
        self.attempts = []

    def check_budget(self):
        self.budget.check()

    def _check(self, qpos, part, pads, supports, payload, departures=(), arrivals=()):
        self.check_budget()
        hits = self.scene.violations(
            qpos, part, pads, supports, payload, departures, arrivals
        )
        if hits:
            hit = hits[0]
            raise planning_arm.PlanningError(
                f"collision:{hit.names}:{hit.distance:.6f}<{hit.required:.6f}"
            )
        return self.scene.data.qpos.copy()

    def _edge(self, a, b, part, pads, supports, payload, departures=(), arrivals=()):
        def geometry_at(qpos):
            self._check(qpos, part, pads, supports, payload, departures, arrivals)
            state = self.scene.data
            return state.geom_xpos.copy(), state.geom_xmat.copy()

        stack = [(a, b, geometry_at(a), geometry_at(b))]
        while stack:
            left, right, first, last = stack.pop()
            translation = np.linalg.norm(last[0] - first[0], axis=1)
            # Frobenius rotation difference bounds displacement of any point
            # within the geometry's bounding radius, including the payload.
            rotation = np.linalg.norm(last[1] - first[1], axis=1)
            displacement = translation + self.scene.model.geom_rbound * rotation
            if (
                np.max(np.abs(right[:14] - left[:14])) <= 0.005
                and np.max(displacement) <= 0.001
            ):
                continue
            middle = (left + right) / 2
            midpoint = geometry_at(middle)
            stack.extend(
                [(left, middle, first, midpoint), (middle, right, midpoint, last)]
            )

    @timing.measured("planning.cartesian", "name", "part")
    def _cartesian(
        self,
        name,
        start,
        position,
        rotation,
        closure,
        part,
        pads=False,
        supports=(),
        payload=None,
        arrivals=(),
    ):
        departures = (
            self.scene.departure_contacts(start, part)
            if name == "lift" and payload is not None
            else ()
        )
        self.scene.set_state(start)
        origin = self.scene.data.site_xpos[self.scene.site].copy()
        initial = Rotation.from_matrix(
            self.scene.data.site_xmat[self.scene.site].reshape(3, 3)
        )
        delta = Rotation.from_matrix(rotation) * initial.inv()
        # Contact deflects the linkage. Reconstructing the command from measured
        # joint position would release its preload at the start of a loaded move.
        opening = closure if payload is not None else float(start[6] / 0.8)
        count = max(
            1,
            int(np.ceil(np.linalg.norm(position - origin) / 0.003)),
            int(np.ceil(abs(closure - opening) / 0.02)),
        )
        path, commands = [start.copy()], [opening]
        qpos = start.copy()
        for fraction in np.linspace(0, 1, count + 1)[1:]:
            self.check_budget()
            value = opening + (closure - opening) * fraction
            seed = qpos.copy()
            if payload is None:
                seed[6:14] = self.aperture.joints(value)
            wrist = (
                Rotation.from_rotvec(delta.as_rotvec() * fraction) * initial
            ).as_matrix()
            qpos = planning_arm.inverse_kinematics(
                self.scene, origin + (position - origin) * fraction, wrist, seed
            )
            self._edge(
                path[-1], qpos, part, pads, supports, payload, departures, arrivals
            )
            qpos = self._check(
                qpos, part, pads, supports, payload, departures, arrivals
            )
            path.append(qpos.copy())
            commands.append(value)
        if departures:
            # Existing contact is allowed only until the lift clears the scene.
            self._check(qpos, part, pads, (), payload)
        return Segment(
            name,
            np.array(path),
            np.array(commands),
            pads,
            supports,
            payload,
            departures=departures,
            arrivals=arrivals,
        )

    @timing.measured("planning.transit", "name", "part")
    def _transit(self, name, start, position, rotation, closure, part, payload=None):
        seed = start.copy()
        if payload is None:
            seed[6:14] = self.aperture.joints(closure)
        goal = None
        # Damped least squares converges from roughly half of the random seeds
        # near reach limits; 4 tries abandoned reachable endpoints ~6% of the
        # time (e.g. the push approach after a long disassembly), 12 ~0.02%.
        for attempt in range(12):
            self.check_budget()
            if attempt:
                seed[:6] = np.clip(
                    start[:6] + self.rng.normal(0, 0.5, 6),
                    self.scene.model.jnt_range[:6, 0],
                    self.scene.model.jnt_range[:6, 1],
                )
            try:
                trial = planning_arm.inverse_kinematics(
                    self.scene, position, rotation, seed
                )
                self._check(trial, part, payload is not None, (), payload)
                goal = trial
                break
            except planning_arm.PlanningBudgetError:
                raise
            except planning_arm.PlanningError:
                continue
        if goal is None:
            raise planning_arm.PlanningError("no_valid_ik_endpoint")

        def valid(arm):
            qpos = seed.copy()
            qpos[:6] = arm
            self.check_budget()
            return self.scene.valid(qpos, part, payload is not None, (), payload)

        path = planning_arm.connect(
            start[:6],
            goal[:6],
            valid,
            self.scene.model.jnt_range[:6],
            seed=int(self.rng.integers(2**31)),
            check_budget=self.check_budget,
        )
        complete = []
        for arm in path:
            qpos = seed.copy()
            qpos[:6] = arm
            complete.append(self._check(qpos, part, payload is not None, (), payload))
        for a, b in itertools.pairwise(complete):
            self._edge(a, b, part, payload is not None, (), payload)
        return Segment(
            name,
            np.array(complete),
            np.full(len(complete), closure),
            payload is not None,
            (),
            payload,
        )

    def object_tops(self, qpos):
        """Obstacle heights for real task objects, excluding isolated batch slots."""
        count = self.active_part_count
        return qpos[14:].reshape(-1, 7)[:count, 2] + self.sizes[:count, 2] / 2

    def _source_supports(self, qpos, part):
        self.scene.set_state(qpos)
        center = qpos[14 + 7 * part : 17 + 7 * part]
        bottom = center[2] - self.sizes[part, 2] / 2
        result = []
        if abs(bottom) < 0.002:
            result.append(self.scene.model.geom("floor").id)
        for other, geom in self.scene.objects.items():
            if other == part or other >= self.active_part_count:
                continue
            pose = qpos[14 + 7 * other : 17 + 7 * other]
            top = pose[2] + self.sizes[other, 2] / 2
            if abs(top - bottom) < 0.002 and np.all(
                np.abs(pose[:2] - center[:2])
                < (self.sizes[other, :2] + self.sizes[part, :2]) / 2
            ):
                result.append(geom)
        return tuple(result)

    def _pickup(self, qpos, part, grasp, grip_preload=0.0005):
        pose = qpos[14 + 7 * part : 21 + 7 * part]
        position, rotation = grasp.pose(pose)
        opening = self.aperture.closure(grasp.width + 0.012)
        holding = self.aperture.closure(grasp.width - grip_preload)
        # Test low geometry first; avoid spending the budget routing to a bad grasp.
        seed = qpos.copy()
        seed[6:14] = self.aperture.joints(opening)
        low = planning_arm.inverse_kinematics(self.scene, position, rotation, seed)
        self._check(low, part, False, (), None)
        top = float(self.object_tops(qpos).max())
        high = max(
            top + 0.08 + self.sizes[part, 2] / 2 + abs(grasp.offset[2]),
            position[2] + 0.06,
        )
        top_down = abs(rotation[2, 2]) > 0.5
        # Pre-shaping the aperture is safe only if the complete closing edge is clear.
        self._edge(qpos, seed, part, False, (), None)
        # As in `continuation`, the global obstacle ceiling can exceed reach above
        # distant parking. Lower approaches keep the full collision checks, and the
        # lift returns to the same height.
        lowest = position[2] + 0.06
        count = int(np.ceil((high - lowest) / 0.05)) if top_down else 0
        for height in np.linspace(high, lowest, count + 1):
            above = position.copy()
            above[2] = height
            pregrasp = above if top_down else position - 0.07 * rotation[:, 2]
            try:
                approach = self._transit(
                    "approach", seed, pregrasp, rotation, opening, part
                )
                break
            except planning_arm.PlanningBudgetError:
                raise
            except planning_arm.PlanningError as error:
                if str(error) != "no_valid_ik_endpoint":
                    raise
        else:
            raise planning_arm.PlanningError("no_valid_ik_endpoint")
        approach.path = np.concatenate([[qpos], approach.path])
        approach.closure = np.r_[qpos[6] / 0.8, approach.closure]
        descend = self._cartesian(
            "descend", approach.path[-1], position, rotation, opening, part
        )
        close = self._cartesian(
            "close", descend.path[-1], position, rotation, holding, part, True
        )
        payload = self.scene.attachment(close.path[-1], part)
        supports = self._source_supports(qpos, part)
        lift = self._cartesian(
            "lift",
            close.path[-1],
            above,
            rotation,
            holding,
            part,
            True,
            supports,
            payload,
        )
        return [approach, descend, close, lift], high

    def _candidate(self, qpos, part, destination, grasp, grip_preload=0.0005):
        pickup, high = self._pickup(qpos, part, grasp, grip_preload)
        lift = pickup[-1]
        destination = self.placement_destination(qpos, part, destination)
        rest = self.continuation(
            lift.path[-1],
            part,
            destination,
            lift.payload,
            lift.closure[-1],
            pickup[0].closure[-1],
            high,
        )
        return [*pickup, *rest]

    @timing.measured("planning.placement", "part")
    def placement_destination(self, qpos, part, destination):
        """Prefer nearby seating poses with clearance from unrelated objects."""
        goal = np.array(destination, dtype=float).copy()
        state = np.array(qpos, dtype=float).copy()
        start = 14 + 7 * part
        state[start : start + 7] = np.r_[goal, 1, 0, 0, 0]
        supports = self._source_supports(state, part)
        if not supports:
            return goal
        support_parts = [
            other for other, geom in self.scene.objects.items() if geom in supports
        ]

        def overlap(center, other):
            pose = state[14 + 7 * other : 16 + 7 * other]
            low = np.maximum(
                center[:2] - self.sizes[part, :2] / 2,
                pose - self.sizes[other, :2] / 2,
            )
            high = np.minimum(
                center[:2] + self.sizes[part, :2] / 2,
                pose + self.sizes[other, :2] / 2,
            )
            return np.maximum(high - low, 0).prod()

        areas = {other: overlap(goal, other) for other in support_parts}
        offsets = sorted(
            itertools.product(
                (0, -0.0005, 0.0005, -0.001, 0.001, -0.0015, 0.0015, -0.002, 0.002),
                repeat=2,
            ),
            key=lambda value: value[0] ** 2 + value[1] ** 2,
        )
        for offset in offsets:
            self.check_budget()
            candidate = goal + np.r_[offset, 0]
            if any(
                overlap(candidate, other) < 0.9 * area for other, area in areas.items()
            ):
                continue
            try:
                seated, candidate_supports = self.seating_destination(
                    qpos, part, candidate
                )
            except planning_arm.PlanningBudgetError:
                raise
            except planning_arm.PlanningError:
                continue
            if candidate_supports != supports:
                continue
            state[start : start + 7] = np.r_[seated, 1, 0, 0, 0]
            self.scene.set_state(state)
            gaps = [
                mujoco.mj_geomDistance(
                    self.scene.model,
                    self.scene.data,
                    self.scene.objects[part],
                    geom,
                    0.001,
                    None,
                )
                for other, geom in self.scene.objects.items()
                if other != part and geom not in supports
            ]
            if min(gaps, default=0.001) >= 0.0003:
                return candidate
        return goal

    @timing.measured("planning.seating", "part")
    def seating_destination(self, qpos, part, destination):
        """Find the upright release height above observed, intended supports."""
        target = np.array(destination, dtype=float).copy()
        state = np.array(qpos, dtype=float).copy()
        start = 14 + 7 * part
        state[start : start + 7] = np.r_[target, 1, 0, 0, 0]
        supports = self._source_supports(state, part)
        if not supports:
            raise planning_arm.PlanningError("no_destination_support")

        def gap(height):
            state[start + 2] = height
            self.scene.set_state(state)
            return min(
                mujoco.mj_geomDistance(
                    self.scene.model,
                    self.scene.data,
                    self.scene.objects[part],
                    support,
                    0.01,
                    None,
                )
                for support in supports
            )

        # Preserve the task and its 5 mm tolerance. Height correction accounts
        # for tilted or displaced supports; it cannot relocate an invalid goal.
        low, high = target[2] - 0.005, target[2] + 0.005
        clearance = 0.0001
        if gap(high) < clearance:
            raise planning_arm.PlanningError("destination_support_height")
        if gap(low) >= clearance:
            # mj_geomDistance can return the wrong sign at exactly symmetric
            # overlaps (identical 1 cm slabs overlapping by half their
            # thickness report +5 mm). Penetration is monotone in height, so
            # any probed height that penetrates is a valid lower bracket.
            for height in np.linspace(low, target[2], 6)[1:]:
                if gap(height) < clearance:
                    low = height
                    break
            else:
                raise planning_arm.PlanningError("destination_support_height")
        for _ in range(16):
            middle = (low + high) / 2
            if gap(middle) < clearance:
                low = middle
            else:
                high = middle
        target[2] = high
        return target, supports

    @timing.measured('planning.continuation', 'part')
    def continuation(
        self, qpos, part, destination, payload, holding, opening, high=None
    ):
        """Recompute transport from the measured post-lift attachment transform."""
        rotation = payload[2].T
        destination, supports = self.seating_destination(qpos, part, destination)
        arrivals = self.scene.destination_contacts(qpos, part, destination)
        # Set down before opening. A 5 mm free fall exceeds the soft support's
        # penetration budget even when the grasp and transport were collision-free.
        place = np.array(destination, dtype=float) - rotation @ payload[1]
        above = place.copy()
        if high is None:
            # The payload moves with the arm, so it is not an obstacle ceiling.
            tops = self.object_tops(qpos)
            obstacle_top = float(np.delete(tops, part).max(initial=0.0))
            high = max(
                obstacle_top + 0.10,
                self.scene.data.site_xpos[self.scene.site, 2],
            )
        above[2] = max(high, place[2] + 0.06)
        minimum_height = place[2] + 0.06
        count = max(1, int(np.ceil((above[2] - minimum_height) / 0.05)))
        # A global obstacle ceiling can exceed reach at distant destinations.
        # Lower endpoints still require the full arm/payload collision checks.
        for height in np.linspace(above[2], minimum_height, count + 1):
            above[2] = height
            try:
                transit = self._transit(
                    "transit", qpos, above, rotation, holding, part, payload
                )
                break
            except planning_arm.PlanningBudgetError:
                raise
            except planning_arm.PlanningError as error:
                if str(error) != "no_valid_ik_endpoint":
                    raise
        else:
            raise planning_arm.PlanningError("no_valid_ik_endpoint")
        lower = self._cartesian(
            "place",
            transit.path[-1],
            place + [0, 0, 0.01],
            rotation,
            holding,
            part,
            True,
            supports,
            payload,
            arrivals=arrivals,
        )
        return [
            transit,
            lower,
            *self.setdown(
                lower.path[-1], part, destination, payload, holding, opening, above[2]
            ),
        ]

    @timing.measured('planning.setdown', 'part')
    def setdown(self, qpos, part, destination, payload, holding, opening, high):
        """Seat and release using the most recent measured attachment."""
        destination, supports = self.seating_destination(qpos, part, destination)
        arrivals = self.scene.destination_contacts(qpos, part, destination)
        rotation = payload[2].T
        place = destination - rotation @ payload[1]
        above = np.r_[place[:2], high]
        setdown = self._cartesian(
            "setdown",
            qpos,
            place,
            rotation,
            holding,
            part,
            True,
            supports,
            payload,
            arrivals=arrivals,
        )
        setdown.reference_speed = 0.06
        release = self._cartesian(
            "open",
            setdown.path[-1],
            place,
            rotation,
            opening,
            part,
            True,
            supports,
            arrivals=arrivals,
        )
        retreat = self._cartesian(
            "retreat",
            release.path[-1],
            above if abs(rotation[2, 2]) > 0.5 else place - 0.07 * rotation[:, 2],
            rotation,
            opening,
            part,
            True,
            supports,
            arrivals=arrivals,
        )
        release.moving_object = retreat.moving_object = True
        return [setdown, release, retreat]

    def _destination_obstructed(self, qpos, part, destination, grasp):
        """Rank whole-arm release collisions before attempting complete paths."""
        goal = np.array(qpos).copy()
        goal[14 + 7 * part : 21 + 7 * part] = np.r_[destination, 1, 0, 0, 0]
        position, rotation = grasp.pose(goal[14 + 7 * part : 21 + 7 * part])
        supports = self._source_supports(goal, part)
        for width in (grasp.width - 0.0005, grasp.width + 0.012):
            goal[6:14] = self.aperture.joints(self.aperture.closure(width))
            try:
                goal = planning_arm.inverse_kinematics(
                    self.scene, position, rotation, goal
                )
            except planning_arm.PlanningBudgetError:
                raise
            except planning_arm.PlanningError:
                # One IK seed failing is not evidence that release is impossible.
                continue
            hits = self.scene.violations(goal, part, True, supports)
            if any(
                any(name.startswith("ur5e/") for name in hit.names)
                and any(not name.startswith("ur5e/") for name in hit.names)
                for hit in hits
            ):
                return True
        return False

    def _source_obstructed(self, qpos, part, grasp):
        """Rank obvious closing-linkage collisions before planning an approach."""
        position, rotation = grasp.pose(qpos[14 + 7 * part : 21 + 7 * part])
        pose = np.array(qpos).copy()
        opening = self.aperture.closure(grasp.width + 0.012)
        holding = self.aperture.closure(grasp.width - 0.0005)
        for closure in np.linspace(opening, holding, 3):
            self.check_budget()
            pose[6:14] = self.aperture.joints(closure)
            try:
                pose = planning_arm.inverse_kinematics(
                    self.scene, position, rotation, pose
                )
            except planning_arm.PlanningBudgetError:
                raise
            except planning_arm.PlanningError:
                # Another IK seed may work; this cue only ranks known collisions.
                continue
            hits = self.scene.violations(pose, part, True)
            if any(
                any(name.startswith("ur5e/robotiq/") for name in hit.names)
                and any(not name.startswith("ur5e/") for name in hit.names)
                for hit in hits
            ):
                return True
        return False

    @timing.measured('planning.pick_place', 'part')
    def plan(self, qpos, part, destination, skip=(), deadline=float("inf")):
        """Plan a checked manipulation, or only pickup when destination is None."""
        with self.budget.scope(self.planning_seconds, deadline):
            return self._plan(qpos, part, destination, skip)

    def _plan(self, qpos, part, destination, skip):
        started = time.perf_counter()
        initial_queries = self.scene.queries
        self.scene.valid_clearance_lower_bound = 0.003
        self.attempts = []
        candidates = [
            self.aperture.calibrate(grasp)
            for grasp in grasps.propose(self.sizes[part], family=self.grasp_family)
        ]
        if self.shuffle_grasps:
            # Extra grasp families must not consume the old top-down budget first.
            original = {
                self.aperture.calibrate(grasp)
                for grasp in grasps.propose(self.sizes[part])
            }
            groups = [
                [g for g in candidates if g in original],
                [g for g in candidates if g not in original and g.roll == np.pi],
                [g for g in candidates if g.roll != np.pi],
            ]
            for group in groups:
                self.rng.shuffle(group)
            candidates = list(itertools.chain.from_iterable(groups))
        # Check both ends before routing: an obstructed closing linkage or release
        # otherwise consumes the budget before alternative grasp depths are tried.
        seating = (
            None
            if destination is None
            else self.seating_destination(qpos, part, destination)[0]
        )
        self.scene.set_state(np.asarray(qpos))
        wrist_now = Rotation.from_matrix(
            self.scene.data.site_xmat[self.scene.site].reshape(3, 3)
        )
        pose = np.asarray(qpos)[14 + 7 * part : 21 + 7 * part]
        ranked = []
        for grasp in candidates:
            self.check_budget()
            ranked.append(
                (
                    self._source_obstructed(qpos, part, grasp)
                    or (
                        seating is not None
                        and self._destination_obstructed(qpos, part, seating, grasp)
                    ),
                    # Prefer a shorter gravity lever arm; this ranks candidates,
                    # while the physical rollout still establishes grasp quality.
                    round(float(np.linalg.norm(grasp.offset[:2])), 6),
                    # Break exact ties (e.g. the same pinch with a 180-degree
                    # wrist flip) by the reorientation needed from the current
                    # gripper pose; shuffled order otherwise spins the wrist.
                    round(
                        float(
                            (
                                Rotation.from_matrix(grasp.pose(pose)[1])
                                * wrist_now.inv()
                            ).magnitude()
                        ),
                        3,
                    ),
                    grasp,
                )
            )
        candidates = [g for *_, g in sorted(ranked, key=lambda item: item[:3])]
        for grasp in candidates:
            if grasp in skip:
                continue
            try:
                self.check_budget()
                if destination is None:
                    segments, _ = self._pickup(np.asarray(qpos), part, grasp)
                else:
                    segments = self._candidate(
                        np.asarray(qpos), part, destination, grasp
                    )
                return Manipulation(
                    part,
                    grasp,
                    segments,
                    time.perf_counter() - started,
                    self.scene.queries - initial_queries,
                    self.attempts.copy(),
                    self.scene.valid_clearance_lower_bound,
                )
            except planning_arm.PlanningBudgetError:
                raise
            except planning_arm.PlanningError as error:
                self.attempts.append(
                    {"grasp": dataclasses.asdict(grasp), "reason": str(error)}
                )
        reason = self.attempts[-1]["reason"] if self.attempts else "no_grasp"
        raise planning_arm.PlanningError(f"no_complete_manipulation:{reason}")


def hold_segment(qpos, ticks=60, name="hold"):
    return Segment(
        name, np.repeat(qpos[None], ticks, axis=0), np.full(ticks, qpos[6] / 0.8)
    )
