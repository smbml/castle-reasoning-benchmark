"""Runtime planning plates."""

import dataclasses
import time

import numpy as np
from scipy.spatial.transform import Rotation

import castle_reasoning.runtime.execution.success as success
import castle_reasoning.runtime.planning.arm as arm
import castle_reasoning.runtime.planning.grasps as grasps
import castle_reasoning.runtime.planning.manipulation as manipulation

# Half of the 5 mm final position tolerance (cuboids.verify_hold).
SLIDE_LATERAL_TOLERANCE_M = 0.0025


class PlateMover:
    """Use the collector's real state, planner, executor, and trace recording."""

    def __init__(self, runner):
        self.runner = runner
        self.planner = runner.planner
        self.access = runner.plan["edge_access"]
        self.floor = self.planner.scene.model.geom("floor").id

    def _budget(self):
        self.runner.budget()
        self._planning_started = time.perf_counter()

    def _execute(self, segment, part):
        self.runner.planning_seconds += time.perf_counter() - self._planning_started
        reference, _ = self.runner.state()
        segment.supports = (self.floor,)
        segment.moving_object = True
        self.runner.segment(segment, part, reference)

    def _cartesian(self, name, part, pos, rotation, closure, pads=False):
        self._budget()
        qpos, _ = self.runner.state()
        with self.planner.budget.scope(self.planner.planning_seconds, self.runner.deadline):
            segment = self.planner._cartesian(
                name, qpos, pos, rotation, closure, part, pads, (self.floor,)
            )
        self._execute(segment, part)

    def _route(self, part, pos, rotation, closure):
        self._budget()
        qpos, _ = self.runner.state()
        with self.planner.budget.scope(self.planner.planning_seconds, self.runner.deadline):
            seed = qpos.copy()
            seed[6:14] = grasps.Aperture.joints(closure)
            self.planner._edge(qpos, seed, part, False, (), None)
            segment = self.planner._transit(
                "push_approach", seed, pos, rotation, closure, part
            )
            segment.path = np.concatenate([[qpos], segment.path])
            segment.closure = np.r_[qpos[6] / 0.8, segment.closure]
        self._execute(segment, part)

    def push(self, part, destination_y):
        """Translate along table Y, using virtual translation only for planning."""
        self._budget()
        qpos, _ = self.runner.state()
        pose = qpos[14 + 7 * part : 21 + 7 * part]
        direction = np.sign(destination_y - pose[1])
        if abs(destination_y - pose[1]) < 0.002:
            return
        size = self.planner.sizes[part]
        closure = self.planner.aperture.closure(0.012)
        rotation = Rotation.from_euler("xyz", [np.pi, 0, np.pi / 2]).as_matrix()
        seed = qpos.copy()
        seed[6:14] = grasps.Aperture.joints(closure)
        scene = self.planner.scene
        scene.set_state(seed)
        frame = scene.data.site_xmat[scene.site].reshape(3, 3)
        points = []
        for geom in scene.pads:
            corners = np.array(
                [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
            )
            world = (corners * scene.model.geom_size[geom]) @ scene.data.geom_xmat[
                geom
            ].reshape(3, 3).T + scene.data.geom_xpos[geom]
            points.extend(
                (world - scene.data.site_xpos[scene.site]) @ frame @ rotation.T
            )
        points = np.array(points)
        leading = points[:, 1].max() if direction > 0 else points[:, 1].min()
        pos = np.array(
            [
                pose[0],
                pose[1] - direction * (size[1] / 2 + 0.004) - leading,
                0.003 - points[:, 2].min(),
            ]
        )
        high = max(
            0.16,
            float(self.planner.object_tops(qpos).max()) + 0.08,
        )
        self._route(part, np.r_[pos[:2], high], rotation, closure)
        self._cartesian("push_lower", part, pos, rotation, closure)
        pos[1] += direction * 0.004
        self._cartesian("push_contact", part, pos, rotation, closure, True)
        self._budget()
        qpos, _ = self.runner.state()
        scene.set_state(qpos)
        endpoint = scene.data.site_xpos[scene.site].copy()
        endpoint[1] += destination_y - qpos[15 + 7 * part]
        # The virtual transform predicts a conservative nominal swept path;
        # physics uses ordinary contacts and never receives an attachment.
        with self.planner.budget.scope(self.planner.planning_seconds, self.runner.deadline):
            predicted = scene.attachment(qpos, part)
            segment = self.planner._cartesian(
                "push",
                qpos,
                endpoint,
                rotation,
                closure,
                part,
                True,
                (self.floor,),
                predicted,
            )
            segment.payload = None
            segment.reference_speed = 0.04
        self._execute(segment, part)
        qpos, _ = self.runner.state()
        if abs(qpos[15 + 7 * part] - destination_y) > 0.005:
            raise arm.PlanningError("push_endpoint_error")
        scene.set_state(qpos)
        retreat = scene.data.site_xpos[scene.site].copy()
        retreat[1] -= direction * 0.02
        self._cartesian("push_withdraw", part, retreat, rotation, closure, True)
        self._cartesian("push_raise", part, np.r_[retreat[:2], high], rotation, closure)

    def _can_slide(self, qpos, part, destination):
        """A Y push cannot correct lateral, height, or orientation errors.

        It preserves them, so a slide is allowed while the lateral error stays
        within half of the 5 mm acceptance tolerance. A 1 mm limit sent plates
        that drifted ~1 mm during the disassembly push through the full edge
        transfer (push out, side pick-and-place, push back) to correct it.
        """
        pose = qpos[14 + 7 * part : 21 + 7 * part]
        size = self.planner.sizes[part]
        angle = success.orientation_error(pose[None, 3:], size[None])[0]
        error = np.abs(pose[[0, 2]] - np.asarray(destination)[[0, 2]])
        return bool(
            error[0] <= SLIDE_LATERAL_TOLERANCE_M
            and error[1] <= 0.001
            and angle <= np.deg2rad(5)
            and self.floor in self.planner._source_supports(qpos, part)
        )

    def move(self, part, destination):
        qpos, _ = self.runner.state()
        if self._can_slide(qpos, part, destination):
            self.runner.attempts.append({"part": part, "primitive": "table_slide"})
            self.push(part, destination[1])
            return
        size = self.planner.sizes[part]
        front = self.access["table_bounds"][2]
        edge_y = front + size[1] / 2 - self.access["overhang_m"]
        self.push(part, edge_y)
        qpos, _ = self.runner.state()
        edge_destination = np.array(destination).copy()
        edge_destination[1] = edge_y
        self._budget()
        with self.planner.budget.scope(self.planner.planning_seconds, self.runner.deadline):
            initial_queries = self.planner.scene.queries
            self.planner.scene.valid_clearance_lower_bound = 0.003
            errors = []
            for grasp in grasps.side_proposals(size, insertion_depth=0.025):
                grasp = self.planner.aperture.calibrate(grasp)
                _, rotation = grasp.pose(qpos[14 + part * 7 : 21 + part * 7])
                if rotation[1, 2] < 0.9:
                    continue
                try:
                    segments = self.planner._candidate(
                        qpos, part, edge_destination, grasp, grip_preload=0.0018
                    )
                    result = manipulation.Manipulation(
                        part,
                        grasp,
                        segments,
                        time.perf_counter() - self._planning_started,
                        self.planner.scene.queries - initial_queries,
                        errors.copy(),
                        self.planner.scene.valid_clearance_lower_bound,
                    )
                    break
                except arm.PlanningBudgetError:
                    raise
                except arm.PlanningError as error:
                    errors.append(str(error))
            else:
                raise arm.PlanningError(f"plate_transfer_planning:{errors}")
        self.runner.attempts.append(
            {
                "part": part,
                "primitive": "edge_plate_transfer",
                "grasp": dataclasses.asdict(grasp),
                "rejected": errors,
                "queries": result.query_count,
                "minimum_valid_clearance_lower_bound_m": result.minimum_valid_clearance_m,
            }
        )
        self.runner.planning_seconds += time.perf_counter() - self._planning_started
        self.runner.move(result, edge_destination)
        self.push(part, destination[1])
