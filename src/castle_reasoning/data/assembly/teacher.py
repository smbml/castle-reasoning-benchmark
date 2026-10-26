"""Data assembly teacher."""

import dataclasses
import enum
import time
from collections.abc import Sequence

import mujoco
import mujoco_warp
import numpy as np
import warp as wp

import castle_reasoning.runtime.control.joint_velocity as control_joint_velocity
import castle_reasoning.runtime.control.parameters as control_parameters
import castle_reasoning.runtime.diagnostics.timing as timing
import castle_reasoning.runtime.planning.arm as arm
import castle_reasoning.runtime.planning.trajectories as trajectories
import castle_reasoning.runtime.simulation.engine as engine
import castle_reasoning.runtime.simulation.model_batch as model_batch
import castle_reasoning.runtime.simulation.models as models

_TASK_TWO_GEOMETRY = {
    "single": (
        ((0.35, 0.0, 0.02),),
        ((0.50, 0.0, 0.02),),
    ),
    "double": (
        ((0.35, -0.1, 0.02), (0.50, -0.1, 0.02)),
        ((0.35, 0.1, 0.02), (0.50, 0.1, 0.02)),
    ),
    "quadruple": (
        (
            (0.325, -0.2, 0.02),
            (0.325, 0.2, 0.02),
            (0.525, -0.2, 0.02),
            (0.525, 0.2, 0.02),
        ),
        (
            (0.375, 0.1, 0.02),
            (0.475, 0.1, 0.02),
            (0.375, -0.1, 0.02),
            (0.475, -0.1, 0.02),
        ),
    ),
}


@dataclasses.dataclass(frozen=True)
class BlockTaskSpec:
    """Source-derived OGBench block task geometry."""

    variant: str
    initial_positions: np.ndarray
    goal_positions: np.ndarray
    success_radius: float = 0.04

    def __post_init__(self) -> None:
        initial = np.asarray(self.initial_positions, dtype=np.float64)
        goal = np.asarray(self.goal_positions, dtype=np.float64)
        if initial.ndim != 2 or initial.shape[1:] != (3,):
            raise ValueError(
                "Initial block positions must have shape [block, 3]."
            )
        if goal.shape != initial.shape:
            raise ValueError("Initial and goal block positions must match.")
        if not np.isfinite(initial).all() or not np.isfinite(goal).all():
            raise ValueError("Block task geometry must be finite.")
        if not np.isfinite(self.success_radius) or self.success_radius <= 0.0:
            raise ValueError("Success radius must be positive and finite.")
        object.__setattr__(self, "initial_positions", initial.copy())
        object.__setattr__(self, "goal_positions", goal.copy())

    @property
    def block_count(self) -> int:
        return self.initial_positions.shape[0]

    @classmethod
    def task_two(cls, variant: str) -> "BlockTaskSpec":
        """Returns the pinned OGBench task-2 positions for one variant."""
        try:
            initial, goal = _TASK_TWO_GEOMETRY[variant]
        except KeyError as error:
            raise ValueError(
                f"Unsupported block variant: {variant!r}."
            ) from error
        return cls(
            variant=variant,
            initial_positions=np.asarray(initial),
            goal_positions=np.asarray(goal),
        )


class BlockPhase(enum.IntEnum):
    """Nominal and recovery segments in one pick-place attempt."""

    APPROACH = 0
    DESCEND = 1
    CLOSE = 2
    LIFT = 3
    TRANSIT = 4
    PLACE = 5
    OPEN = 6
    RETREAT = 7
    REGRASP_RELEASE = 8
    REGRASP_DESCEND = 9


NOMINAL_BLOCK_PHASES = (
    BlockPhase.APPROACH,
    BlockPhase.DESCEND,
    BlockPhase.CLOSE,
    BlockPhase.LIFT,
    BlockPhase.TRANSIT,
    BlockPhase.PLACE,
    BlockPhase.OPEN,
    BlockPhase.RETREAT,
)


class PhaseFailure(enum.IntEnum):
    """Postcondition that most recently sent a world into recovery."""

    NONE = 0
    APPROACH_TRACKING = 1
    DESCEND_TRACKING = 2
    CLOSE_ATTACHMENT = 3
    LIFT_POSSESSION = 4
    TRANSIT_POSSESSION = 5
    PLACE_TRACKING = 6
    OPEN_PLACEMENT = 7
    RETREAT_GOAL = 8


class GraspFamily(enum.IntEnum):
    """Difficulty-stratified block grasp geometry."""

    TOP_CENTER = 0
    TOP_EDGE = 1
    DIAGONAL = 2
    HIGH_TILT = 3


@dataclasses.dataclass(frozen=True)
class GraspFamilySpec:
    """Sampling bounds for one grasp family."""

    family: GraspFamily
    weight: float
    contact_radius_range: tuple[float, float]
    contact_z_range: tuple[float, float]
    tilt_range: tuple[float, float]
    approach_distance_range: tuple[float, float]

    def __post_init__(self) -> None:
        if not isinstance(self.family, GraspFamily):
            raise ValueError("Grasp family must be a GraspFamily value.")
        if not np.isfinite(self.weight) or self.weight < 0.0:
            raise ValueError(
                "Grasp family weight must be finite and nonnegative."
            )
        for name in (
            "contact_radius_range",
            "contact_z_range",
            "tilt_range",
            "approach_distance_range",
        ):
            bounds = np.asarray(getattr(self, name), dtype=np.float64)
            if (
                bounds.shape != (2,)
                or not np.isfinite(bounds).all()
                or bounds[0] > bounds[1]
                or bounds[0] < 0.0
            ):
                label = name.replace("_", " ")
                raise ValueError(f"Invalid grasp {label}: {tuple(bounds)}.")
        if self.tilt_range[1] >= np.pi / 2.0:
            raise ValueError("Grasp tilt must remain below pi/2 radians.")
        if self.approach_distance_range[0] < 0.04:
            raise ValueError("Grasp approach distance must be at least 4 cm.")


_DEFAULT_GRASP_FAMILIES = (
    GraspFamilySpec(
        family=GraspFamily.TOP_CENTER,
        weight=0.25,
        contact_radius_range=(0.0, 0.008),
        contact_z_range=(0.012, 0.020),
        tilt_range=(0.0, 0.18),
        approach_distance_range=(0.12, 0.18),
    ),
    GraspFamilySpec(
        family=GraspFamily.TOP_EDGE,
        weight=0.25,
        contact_radius_range=(0.008, 0.016),
        contact_z_range=(0.012, 0.024),
        tilt_range=(0.10, 0.35),
        approach_distance_range=(0.11, 0.17),
    ),
    GraspFamilySpec(
        family=GraspFamily.DIAGONAL,
        weight=0.25,
        contact_radius_range=(0.004, 0.017),
        contact_z_range=(0.018, 0.032),
        tilt_range=(0.30, 0.65),
        approach_distance_range=(0.10, 0.16),
    ),
    GraspFamilySpec(
        family=GraspFamily.HIGH_TILT,
        weight=0.25,
        contact_radius_range=(0.006, 0.018),
        contact_z_range=(0.024, 0.038),
        tilt_range=(0.60, 1.00),
        approach_distance_range=(0.09, 0.15),
    ),
)


def recovery_starts_closed(failure_reason: PhaseFailure) -> bool:
    """Returns whether recovery begins from a closed-gripper nominal phase."""
    failure_reason = PhaseFailure(failure_reason)
    return failure_reason in (
        PhaseFailure.CLOSE_ATTACHMENT,
        PhaseFailure.LIFT_POSSESSION,
        PhaseFailure.TRANSIT_POSSESSION,
        PhaseFailure.PLACE_TRACKING,
    )


_FAILURE_BY_PHASE = {
    BlockPhase.APPROACH: PhaseFailure.APPROACH_TRACKING,
    BlockPhase.DESCEND: PhaseFailure.DESCEND_TRACKING,
    BlockPhase.CLOSE: PhaseFailure.CLOSE_ATTACHMENT,
    BlockPhase.LIFT: PhaseFailure.LIFT_POSSESSION,
    BlockPhase.TRANSIT: PhaseFailure.TRANSIT_POSSESSION,
    BlockPhase.PLACE: PhaseFailure.PLACE_TRACKING,
    BlockPhase.OPEN: PhaseFailure.OPEN_PLACEMENT,
    BlockPhase.RETREAT: PhaseFailure.RETREAT_GOAL,
}


@dataclasses.dataclass(frozen=True)
class PhaseThresholds:
    """Geometric tolerances for nominal phase postconditions."""

    effector_tolerance: float = 0.04
    attachment_tolerance: float = 0.06
    minimum_lift_height: float = 0.04
    placement_tolerance: float = 0.04

    def __post_init__(self) -> None:
        values = dataclasses.astuple(self)
        if not all(np.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("Phase thresholds must be positive and finite.")


@dataclasses.dataclass(frozen=True)
class PhaseTransition:
    """State-machine result after one phase boundary."""

    phase: BlockPhase
    block_index: int
    recovery_count: int
    completed: bool = False
    failed: bool = False
    failure_reason: PhaseFailure = PhaseFailure.NONE


def transition_after_phase(
    *,
    phase: BlockPhase,
    boundary_satisfied: bool,
    block_index: int,
    block_count: int,
    recovery_count: int,
    max_recovery_attempts: int,
    failure_reason: PhaseFailure = PhaseFailure.NONE,
) -> PhaseTransition:
    """Returns the explicit state-machine edge at a phase boundary."""
    phase = BlockPhase(phase)
    failure_reason = PhaseFailure(failure_reason)
    if block_count < 1 or not 0 <= block_index < block_count:
        raise ValueError("The block index must address a nonempty task.")
    if recovery_count < 0 or max_recovery_attempts < 0:
        raise ValueError("Recovery counts must be nonnegative.")
    if phase == BlockPhase.REGRASP_RELEASE:
        return PhaseTransition(
            BlockPhase.REGRASP_DESCEND,
            block_index,
            recovery_count,
            failure_reason=failure_reason,
        )
    if phase == BlockPhase.REGRASP_DESCEND:
        return PhaseTransition(
            BlockPhase.CLOSE,
            block_index,
            recovery_count,
            failure_reason=failure_reason,
        )
    if boundary_satisfied:
        if phase != BlockPhase.RETREAT:
            return PhaseTransition(
                BlockPhase(int(phase) + 1), block_index, recovery_count
            )
        next_block = block_index + 1
        return PhaseTransition(
            BlockPhase.APPROACH,
            next_block if next_block < block_count else block_index,
            0,
            completed=next_block == block_count,
        )

    next_recovery = recovery_count + 1
    reason = _FAILURE_BY_PHASE[phase]
    return PhaseTransition(
        BlockPhase.REGRASP_RELEASE
        if next_recovery <= max_recovery_attempts
        else phase,
        block_index,
        next_recovery,
        failed=next_recovery > max_recovery_attempts,
        failure_reason=reason,
    )


def phase_boundary_satisfied(
    *,
    phase: BlockPhase,
    effector_position: np.ndarray,
    block_position: np.ndarray,
    block_start_position: np.ndarray,
    planned_end_position: np.ndarray,
    planned_placement_position: np.ndarray,
    actual_goal_position: np.ndarray,
    success_radius: float,
    thresholds: PhaseThresholds,
) -> bool:
    """Evaluates the geometric postcondition for one nominal phase."""
    phase = BlockPhase(phase)
    vectors = tuple(
        np.asarray(value, dtype=np.float64)
        for value in (
            effector_position,
            block_position,
            block_start_position,
            planned_end_position,
            planned_placement_position,
            actual_goal_position,
        )
    )
    if any(value.shape != (3,) for value in vectors):
        raise ValueError("Phase-boundary vectors must have shape (3,).")
    if not all(np.isfinite(value).all() for value in vectors):
        raise ValueError("Phase-boundary vectors must be finite.")
    if not np.isfinite(success_radius) or success_radius <= 0.0:
        raise ValueError("The success radius must be positive and finite.")
    (
        effector_position,
        block_position,
        block_start_position,
        planned_end_position,
        planned_placement_position,
        actual_goal_position,
    ) = vectors
    if phase in (BlockPhase.REGRASP_RELEASE, BlockPhase.REGRASP_DESCEND):
        return True

    effector_aligned = (
        np.linalg.norm(effector_position - planned_end_position)
        <= thresholds.effector_tolerance
    )
    attached = (
        np.linalg.norm(effector_position - block_position)
        <= thresholds.attachment_tolerance
    )
    lifted = (
        block_position[2] - block_start_position[2]
        >= thresholds.minimum_lift_height
    )
    placed = (
        np.linalg.norm(block_position - planned_placement_position)
        <= thresholds.placement_tolerance
    )
    if phase in (BlockPhase.APPROACH, BlockPhase.DESCEND):
        return bool(effector_aligned)
    if phase == BlockPhase.CLOSE:
        return bool(effector_aligned and attached)
    if phase in (BlockPhase.LIFT, BlockPhase.TRANSIT):
        return bool(effector_aligned and attached and lifted)
    if phase == BlockPhase.PLACE:
        return bool(effector_aligned and placed)
    if phase == BlockPhase.OPEN:
        return bool(placed)
    return bool(
        effector_aligned
        and np.linalg.norm(block_position - actual_goal_position)
        <= success_radius
    )


@dataclasses.dataclass(frozen=True)
class OracleParameters:
    """Validated diversity and recovery bounds for block collection."""

    phase_durations_seconds: tuple[float, ...] = (
        1.40,
        1.00,
        0.50,
        1.10,
        1.70,
        1.00,
        0.50,
        1.10,
    )
    duration_scale_range: tuple[float, float] = (0.85, 1.20)
    max_normalized_arm_speed: float = 0.50
    clearance_range: tuple[float, float] = (0.12, 0.18)
    grasp_families: tuple[GraspFamilySpec, ...] = _DEFAULT_GRASP_FAMILIES
    target_yaw_range: tuple[float, float] = (-np.pi, np.pi)
    intentional_error_probability: float = 0.20
    intentional_error_xy_range: tuple[float, float] = (0.045, 0.065)
    max_recovery_attempts: int = 2
    effector_tolerance: float = 0.04
    attachment_tolerance: float = 0.06
    minimum_lift_height: float = 0.04
    placement_tolerance: float = 0.04
    regrasp_pregrasp_distance: float = 0.05
    regrasp_release_seconds: float = 0.90
    regrasp_descend_seconds: float = 0.80

    def __post_init__(self) -> None:
        durations = np.asarray(self.phase_durations_seconds, dtype=np.float64)
        if durations.shape != (len(NOMINAL_BLOCK_PHASES),) or not np.all(
            durations > 0.0
        ):
            raise ValueError("Eight positive phase durations are required.")
        if not np.isfinite(durations).all():
            raise ValueError("Phase durations must be finite.")
        for name in (
            "duration_scale_range",
            "clearance_range",
            "target_yaw_range",
            "intentional_error_xy_range",
        ):
            bounds = np.asarray(getattr(self, name), dtype=np.float64)
            if (
                bounds.shape != (2,)
                or not np.isfinite(bounds).all()
                or bounds[0] > bounds[1]
            ):
                label = name.replace("_", " ")
                raise ValueError(f"Invalid {label}: {tuple(bounds)}.")
        if self.duration_scale_range[0] <= 0.0:
            raise ValueError("The duration scale must remain positive.")
        if (
            not np.isfinite(self.max_normalized_arm_speed)
            or not 0.0 < self.max_normalized_arm_speed <= 1.0
        ):
            raise ValueError(
                "Normalized arm speed must be finite and in (0, 1]."
            )
        if self.clearance_range[0] < 0.04:
            raise ValueError("The clearance range must clear a 4 cm block.")
        families = tuple(spec.family for spec in self.grasp_families)
        if families != tuple(GraspFamily):
            raise ValueError(
                "Every grasp family must appear exactly once in order."
            )
        weight_sum = sum(spec.weight for spec in self.grasp_families)
        if not np.isclose(weight_sum, 1.0, atol=1e-8):
            raise ValueError("Grasp family weights must sum to one.")
        if not 0.0 <= self.intentional_error_probability <= 1.0:
            raise ValueError("Intentional-error probability must be in [0, 1].")
        if self.intentional_error_xy_range[0] < 0.04:
            raise ValueError(
                "Intentional errors must exceed the success radius."
            )
        if self.max_recovery_attempts < 0:
            raise ValueError("The maximum recovery count must be nonnegative.")
        PhaseThresholds(
            effector_tolerance=self.effector_tolerance,
            attachment_tolerance=self.attachment_tolerance,
            minimum_lift_height=self.minimum_lift_height,
            placement_tolerance=self.placement_tolerance,
        )
        if (
            not np.isfinite(self.regrasp_pregrasp_distance)
            or self.regrasp_pregrasp_distance <= 0.0
        ):
            raise ValueError(
                "The regrasp pregrasp distance must be positive and finite."
            )
        if not all(
            np.isfinite(duration) and duration > 0.0
            for duration in (
                self.regrasp_release_seconds,
                self.regrasp_descend_seconds,
            )
        ):
            raise ValueError("Regrasp durations must be positive and finite.")


@dataclasses.dataclass(frozen=True)
class WaypointProgram:
    """One smooth eight-phase pick-place attempt."""

    phases: tuple[BlockPhase, ...]
    positions: np.ndarray
    rpy: np.ndarray
    opening: np.ndarray
    durations: np.ndarray
    position_velocities: np.ndarray
    rpy_velocities: np.ndarray


def build_waypoint_program(
    *,
    effector_position: np.ndarray,
    effector_rpy: np.ndarray,
    block_position: np.ndarray,
    goal_position: np.ndarray,
    grasp_offset: np.ndarray,
    clearance: float,
    approach_direction: np.ndarray,
    approach_distance: float,
    target_rpy: np.ndarray,
    duration_scale: float,
    parameters: OracleParameters,
) -> WaypointProgram:
    """Builds the canonical approach-through-retreat waypoint sequence."""
    vectors: Sequence[np.ndarray] = (
        effector_position,
        effector_rpy,
        block_position,
        goal_position,
        grasp_offset,
        approach_direction,
        target_rpy,
    )
    vectors = tuple(np.asarray(value, dtype=np.float64) for value in vectors)
    if any(value.shape != (3,) for value in vectors):
        raise ValueError("All Cartesian and RPY values must have shape (3,).")
    if not all(np.isfinite(value).all() for value in vectors):
        raise ValueError("Waypoint inputs must be finite.")
    if not np.isfinite(clearance) or clearance < 0.04:
        raise ValueError("Waypoint clearance must be at least 4 cm.")
    if not np.isfinite(approach_distance) or approach_distance < 0.04:
        raise ValueError("Waypoint approach distance must be at least 4 cm.")
    if not np.isclose(np.linalg.norm(approach_direction), 1.0, atol=1e-6):
        raise ValueError("Waypoint approach direction must be unit length.")
    if not np.isfinite(duration_scale) or duration_scale <= 0.0:
        raise ValueError("Waypoint duration scale must be positive.")
    (
        effector_position,
        effector_rpy,
        block_position,
        goal_position,
        grasp_offset,
        approach_direction,
        target_rpy,
    ) = vectors
    grasp = block_position + grasp_offset
    approach = grasp - approach_direction * approach_distance
    lift = grasp.copy()
    lift[2] += clearance
    placement = goal_position.copy()
    placement += grasp_offset
    transit = placement.copy()
    transit[2] += clearance
    positions = np.stack(
        (
            effector_position,
            approach,
            grasp,
            grasp,
            lift,
            transit,
            placement,
            placement,
            transit,
        )
    )
    rpy = np.concatenate(
        (effector_rpy[None], np.repeat(target_rpy[None], 8, axis=0)),
        axis=0,
    )
    opening = np.asarray([1, 1, 1, 0, 0, 0, 0, 1, 1], dtype=np.float64)
    durations = (
        np.asarray(parameters.phase_durations_seconds, dtype=np.float64)
        * duration_scale
    )
    position_velocities = trajectories.waypoint_velocities(positions, durations)
    position_velocities[[2, 3, 6, 7]] = 0.0
    return WaypointProgram(
        phases=NOMINAL_BLOCK_PHASES,
        positions=positions,
        rpy=rpy,
        opening=opening,
        durations=durations,
        position_velocities=position_velocities,
        rpy_velocities=trajectories.waypoint_velocities(rpy, durations),
    )


@wp.func
def _phase_failure_reason_device(phase: int) -> int:
    if phase == 0:
        return 1
    if phase == 1:
        return 2
    if phase == 2:
        return 3
    if phase == 3:
        return 4
    if phase == 4:
        return 5
    if phase == 5:
        return 6
    if phase == 6:
        return 7
    return 8


@wp.func
def _phase_boundary_satisfied_device(
    phase: int,
    effector_position: wp.vec3,
    block_position: wp.vec3,
    block_start_position: wp.vec3,
    planned_end_position: wp.vec3,
    planned_placement_position: wp.vec3,
    actual_goal_position: wp.vec3,
    success_radius: float,
    effector_tolerance: float,
    attachment_tolerance: float,
    minimum_lift_height: float,
    placement_tolerance: float,
) -> bool:
    if phase >= 8:
        return True
    effector_aligned = (
        wp.length(effector_position - planned_end_position)
        <= effector_tolerance
    )
    attached = (
        wp.length(effector_position - block_position) <= attachment_tolerance
    )
    lifted = block_position[2] - block_start_position[2] >= minimum_lift_height
    placed = (
        wp.length(block_position - planned_placement_position)
        <= placement_tolerance
    )
    if phase == 0 or phase == 1:
        return effector_aligned
    if phase == 2:
        return effector_aligned and attached
    if phase == 3 or phase == 4:
        return effector_aligned and attached and lifted
    if phase == 5:
        return effector_aligned and placed
    if phase == 6:
        return placed
    return (
        effector_aligned
        and wp.length(block_position - actual_goal_position) <= success_radius
    )


@wp.kernel
def _advance_block_oracle_phase(
    qpos: wp.array2d(dtype=float),
    site_xpos: wp.array2d(dtype=wp.vec3),
    pinch_site_id: int,
    block_count: int,
    grasp_proposal_count: int,
    object_qpos_start: int,
    goal_positions: wp.array2d(dtype=wp.vec3),
    block_order: wp.array2d(dtype=int),
    phase_durations: wp.array(dtype=float),
    grasp_offset: wp.array2d(dtype=wp.vec3),
    approach_direction: wp.array2d(dtype=wp.vec3),
    approach_distance: wp.array2d(dtype=float),
    clearance: wp.array2d(dtype=float),
    duration_scale: wp.array2d(dtype=float),
    intentional_error: wp.array2d(dtype=bool),
    goal_error: wp.array2d(dtype=wp.vec3),
    phase: wp.array(dtype=int),
    phase_time: wp.array(dtype=float),
    block_index: wp.array(dtype=int),
    recovery_count: wp.array(dtype=int),
    failure_reason: wp.array(dtype=int),
    completed: wp.array(dtype=bool),
    failed: wp.array(dtype=bool),
    effector_start: wp.array(dtype=wp.vec3),
    block_start: wp.array(dtype=wp.vec3),
    success_radius: float,
    max_recovery_attempts: int,
    effector_tolerance: float,
    attachment_tolerance: float,
    minimum_lift_height: float,
    placement_tolerance: float,
    regrasp_release_seconds: float,
    regrasp_descend_seconds: float,
):
    world = wp.tid()
    if completed[world] or failed[world]:
        return

    current_phase = phase[world]
    current_block = block_index[world]
    scale = duration_scale[world, current_block]
    duration = regrasp_release_seconds
    if current_phase < 8:
        duration = scale * phase_durations[current_phase]
    elif current_phase == 9:
        duration = regrasp_descend_seconds
    if phase_time[world] + 1.0e-6 < duration:
        return

    object_qpos = object_qpos_start + block_order[world, current_block] * 7
    block_position = wp.vec3(
        qpos[world, object_qpos + 0],
        qpos[world, object_qpos + 1],
        qpos[world, object_qpos + 2],
    )
    goal = goal_positions[world, block_order[world, current_block]]
    proposal_attempt = recovery_count[world]
    if proposal_attempt >= grasp_proposal_count:
        proposal_attempt = grasp_proposal_count - 1
    proposal = current_block * grasp_proposal_count + proposal_attempt
    offset = grasp_offset[world, proposal]
    placement_error = wp.vec3(0.0)
    if recovery_count[world] == 0 and intentional_error[world, current_block]:
        placement_error = goal_error[world, current_block]
    planned_placement = goal + placement_error
    planned_end = effector_start[world]
    if current_phase < 8:
        planned_end = trajectories._waypoint_position_device(
            current_phase + 1,
            effector_start[world],
            block_start[world],
            goal,
            offset,
            approach_direction[world, proposal],
            approach_distance[world, proposal],
            clearance[world, current_block],
            placement_error,
        )
    boundary_satisfied = _phase_boundary_satisfied_device(
        current_phase,
        site_xpos[world, pinch_site_id],
        block_position,
        block_start[world],
        planned_end,
        planned_placement,
        goal,
        success_radius,
        effector_tolerance,
        attachment_tolerance,
        minimum_lift_height,
        placement_tolerance,
    )

    if current_phase == 8:
        phase[world] = 9
    elif current_phase == 9:
        phase[world] = 2
    elif boundary_satisfied:
        failure_reason[world] = 0
        if current_phase == 7:
            next_block = current_block + 1
            recovery_count[world] = 0
            if next_block >= block_count:
                completed[world] = True
            else:
                block_index[world] = next_block
                phase[world] = 0
        else:
            phase[world] = current_phase + 1
    else:
        recovery = recovery_count[world] + 1
        recovery_count[world] = recovery
        failure_reason[world] = _phase_failure_reason_device(current_phase)
        if recovery > max_recovery_attempts:
            failed[world] = True
        else:
            phase[world] = 8
    phase_time[world] = 0.0


@wp.func
def _initial_grasp_family_device(
    quantile: float, cumulative_weights: wp.array(dtype=float)
) -> int:
    family = 0
    if quantile >= cumulative_weights[0]:
        family = 1
    if quantile >= cumulative_weights[1]:
        family = 2
    if quantile >= cumulative_weights[2]:
        family = 3
    return family


@wp.func
def _proposal_grasp_family_device(initial_family: int, attempt: int) -> int:
    if attempt == 0:
        return initial_family
    if attempt == 1:
        if initial_family == 0:
            return 1
        return initial_family - 1
    return 0


@wp.kernel
def _initialize_block_oracle(
    reset_mask: wp.array(dtype=bool),
    random_u32: wp.array2d(dtype=wp.uint32),
    episode: wp.array(dtype=int),
    nworld: int,
    block_count: int,
    grasp_proposal_count: int,
    family_cumulative_weights: wp.array(dtype=float),
    family_parameter_bounds: wp.array2d(dtype=float),
    clearance_lower: float,
    clearance_upper: float,
    duration_lower: float,
    duration_upper: float,
    yaw_lower: float,
    yaw_upper: float,
    error_probability: float,
    error_lower: float,
    error_upper: float,
    phase: wp.array(dtype=int),
    phase_time: wp.array(dtype=float),
    block_index: wp.array(dtype=int),
    recovery_count: wp.array(dtype=int),
    failure_reason: wp.array(dtype=int),
    completed: wp.array(dtype=bool),
    failed: wp.array(dtype=bool),
    effector_start: wp.array(dtype=wp.vec3),
    effector_start_velocity: wp.array(dtype=wp.vec3),
    effector_start_quat: wp.array(dtype=wp.quat),
    block_start: wp.array(dtype=wp.vec3),
    orientation_reference: wp.array(dtype=wp.quat),
    grasp_family_id: wp.array2d(dtype=int),
    grasp_offset: wp.array2d(dtype=wp.vec3),
    approach_direction: wp.array2d(dtype=wp.vec3),
    approach_distance: wp.array2d(dtype=float),
    clearance: wp.array2d(dtype=float),
    duration_scale: wp.array2d(dtype=float),
    target_rpy: wp.array2d(dtype=wp.vec3),
    intentional_error: wp.array2d(dtype=bool),
    goal_error: wp.array2d(dtype=wp.vec3),
):
    world = wp.tid()
    if not reset_mask[world]:
        return
    phase[world] = 0
    phase_time[world] = 0.0
    block_index[world] = 0
    recovery_count[world] = 0
    failure_reason[world] = 0
    completed[world] = False
    failed[world] = False
    effector_start[world] = wp.vec3(0.0)
    effector_start_velocity[world] = wp.vec3(0.0)
    effector_start_quat[world] = wp.quat_identity()
    block_start[world] = wp.vec3(0.0)
    orientation_reference[world] = wp.quat(0.0, 0.0, 0.0, 0.0)
    for block in range(block_count):
        streams_per_block = 5 + 8 * grasp_proposal_count
        stream = block * streams_per_block
        clearance_random = wp.randf(random_u32[world, stream + 0])
        clearance[world, block] = clearance_lower + clearance_random * (
            clearance_upper - clearance_lower
        )
        duration_random = wp.randf(random_u32[world, stream + 1])
        duration_scale[world, block] = duration_lower + duration_random * (
            duration_upper - duration_lower
        )
        error_random = wp.randf(random_u32[world, stream + 2])
        use_error = error_random < error_probability
        intentional_error[world, block] = use_error
        angle_random = wp.randf(random_u32[world, stream + 3])
        magnitude_random = wp.randf(random_u32[world, stream + 4])
        angle = 2.0 * wp.pi * angle_random
        magnitude = error_lower + magnitude_random * (error_upper - error_lower)
        if use_error:
            goal_error[world, block] = wp.vec3(
                magnitude * wp.cos(angle), magnitude * wp.sin(angle), 0.0
            )
        else:
            goal_error[world, block] = wp.vec3(0.0)
        initial_family = int(0)
        for attempt in range(grasp_proposal_count):
            proposal = block * grasp_proposal_count + attempt
            proposal_stream = stream + 5 + attempt * 8
            if attempt == 0:
                family_jitter = wp.randf(random_u32[world, proposal_stream + 0])
                stratum = (world + block + episode[world]) % nworld
                quantile = (float(stratum) + family_jitter) / float(nworld)
                initial_family = _initial_grasp_family_device(
                    quantile, family_cumulative_weights
                )
            family = _proposal_grasp_family_device(initial_family, attempt)
            grasp_family_id[world, proposal] = family
            radius_random = wp.randf(random_u32[world, proposal_stream + 1])
            contact_angle_random = wp.randf(
                random_u32[world, proposal_stream + 2]
            )
            z_random = wp.randf(random_u32[world, proposal_stream + 3])
            radius = family_parameter_bounds[family, 0] + radius_random * (
                family_parameter_bounds[family, 1]
                - family_parameter_bounds[family, 0]
            )
            contact_angle = 2.0 * wp.pi * contact_angle_random
            contact_z = family_parameter_bounds[family, 2] + z_random * (
                family_parameter_bounds[family, 3]
                - family_parameter_bounds[family, 2]
            )
            grasp_offset[world, proposal] = wp.vec3(
                radius * wp.cos(contact_angle),
                radius * wp.sin(contact_angle),
                contact_z,
            )
            tilt_random = wp.randf(random_u32[world, proposal_stream + 4])
            tilt_angle_random = wp.randf(random_u32[world, proposal_stream + 5])
            yaw_random = wp.randf(random_u32[world, proposal_stream + 6])
            distance_random = wp.randf(random_u32[world, proposal_stream + 7])
            tilt = family_parameter_bounds[family, 4] + tilt_random * (
                family_parameter_bounds[family, 5]
                - family_parameter_bounds[family, 4]
            )
            tilt_angle = 2.0 * wp.pi * tilt_angle_random
            target_angles = wp.vec3(
                wp.pi + tilt * wp.cos(tilt_angle),
                tilt * wp.sin(tilt_angle),
                yaw_lower + yaw_random * (yaw_upper - yaw_lower),
            )
            target_rpy[world, proposal] = target_angles
            target_orientation = wp.quat_rpy(
                target_angles[0], target_angles[1], target_angles[2]
            )
            approach_direction[world, proposal] = wp.quat_rotate(
                target_orientation, wp.vec3(0.0, 0.0, 1.0)
            )
            approach_distance[world, proposal] = family_parameter_bounds[
                family, 6
            ] + distance_random * (
                family_parameter_bounds[family, 7]
                - family_parameter_bounds[family, 6]
            )


@wp.kernel
def _block_oracle_action(
    qpos: wp.array2d(dtype=float),
    qvel: wp.array2d(dtype=float),
    site_xpos: wp.array2d(dtype=wp.vec3),
    site_xmat: wp.array2d(dtype=wp.mat33),
    subtree_com: wp.array2d(dtype=wp.vec3),
    cdof: wp.array2d(dtype=wp.spatial_vector),
    pinch_site_id: int,
    root_body_id: int,
    block_count: int,
    grasp_proposal_count: int,
    object_qpos_start: int,
    goal_positions: wp.array2d(dtype=wp.vec3),
    block_order: wp.array2d(dtype=int),
    phase_durations: wp.array(dtype=float),
    grasp_offset: wp.array2d(dtype=wp.vec3),
    approach_direction: wp.array2d(dtype=wp.vec3),
    approach_distance: wp.array2d(dtype=float),
    clearance: wp.array2d(dtype=float),
    duration_scale: wp.array2d(dtype=float),
    target_rpy: wp.array2d(dtype=wp.vec3),
    intentional_error: wp.array2d(dtype=bool),
    goal_error: wp.array2d(dtype=wp.vec3),
    phase: wp.array(dtype=int),
    phase_time: wp.array(dtype=float),
    block_index: wp.array(dtype=int),
    recovery_count: wp.array(dtype=int),
    failure_reason: wp.array(dtype=int),
    completed: wp.array(dtype=bool),
    failed: wp.array(dtype=bool),
    effector_start: wp.array(dtype=wp.vec3),
    effector_start_velocity: wp.array(dtype=wp.vec3),
    effector_start_quat: wp.array(dtype=wp.quat),
    block_start: wp.array(dtype=wp.vec3),
    orientation_reference: wp.array(dtype=wp.quat),
    action: wp.array2d(dtype=float),
    control_timestep: float,
    regrasp_pregrasp_distance: float,
    regrasp_release_seconds: float,
    regrasp_descend_seconds: float,
    damping: float,
    position_gain: float,
    orientation_gain: float,
    max_normalized_arm_speed: float,
    velocity_limit: float,
):
    world = wp.tid()
    if completed[world] or failed[world]:
        for action_index in range(6):
            action[world, action_index] = 0.0
        action[world, 6] = -1.0
        orientation_reference[world] = wp.quat_from_matrix(
            site_xmat[world, pinch_site_id]
        )
        return

    current_block = block_index[world]
    current_phase = phase[world]
    object_qpos = object_qpos_start + block_order[world, current_block] * 7
    goal = goal_positions[world, block_order[world, current_block]] + wp.vec3(0.0, 0.0, 0.005)
    block_position = wp.vec3(
        qpos[world, object_qpos + 0],
        qpos[world, object_qpos + 1],
        qpos[world, object_qpos + 2],
    )
    jacobian = wp.spatial_matrix(0.0)
    point_offset = (
        site_xpos[world, pinch_site_id] - subtree_com[world, root_body_id]
    )
    effector_velocity = wp.vec3(0.0)
    for dof in range(6):
        motion = cdof[world, dof]
        angular = wp.spatial_top(motion)
        linear = wp.spatial_bottom(motion) + wp.cross(angular, point_offset)
        jacobian[0, dof] = linear[0]
        jacobian[1, dof] = linear[1]
        jacobian[2, dof] = linear[2]
        jacobian[3, dof] = angular[0]
        jacobian[4, dof] = angular[1]
        jacobian[5, dof] = angular[2]
        effector_velocity = effector_velocity + linear * qvel[world, dof]
    if phase_time[world] == 0.0 and (current_phase == 0 or current_phase == 8):
        effector_start[world] = site_xpos[world, pinch_site_id]
        effector_start_velocity[world] = effector_velocity
        previous_orientation = orientation_reference[world]
        if wp.dot(previous_orientation, previous_orientation) > 0.5:
            effector_start_quat[world] = previous_orientation
        else:
            effector_start_quat[world] = wp.quat_from_matrix(
                site_xmat[world, pinch_site_id]
            )
        block_start[world] = block_position

    start_effector = effector_start[world]
    start_block = block_start[world]
    proposal_attempt = recovery_count[world]
    if proposal_attempt >= grasp_proposal_count:
        proposal_attempt = grasp_proposal_count - 1
    proposal = current_block * grasp_proposal_count + proposal_attempt
    offset = grasp_offset[world, proposal]
    approach = approach_direction[world, proposal]
    proposal_approach_distance = approach_distance[world, proposal]
    block_clearance = clearance[world, current_block]
    scale = duration_scale[world, current_block]
    placement_error = wp.vec3(0.0)
    if recovery_count[world] == 0 and intentional_error[world, current_block]:
        placement_error = goal_error[world, current_block]
    start_position = start_effector
    end_position = start_effector
    start_velocity = wp.vec3(0.0)
    end_velocity = wp.vec3(0.0)
    duration = regrasp_release_seconds
    if current_phase < 8:
        start_position = trajectories._waypoint_position_device(
            current_phase,
            start_effector,
            start_block,
            goal,
            offset,
            approach,
            proposal_approach_distance,
            block_clearance,
            placement_error,
        )
        end_position = trajectories._waypoint_position_device(
            current_phase + 1,
            start_effector,
            start_block,
            goal,
            offset,
            approach,
            proposal_approach_distance,
            block_clearance,
            placement_error,
        )
        start_velocity = trajectories._waypoint_velocity_device(
            current_phase,
            start_effector,
            start_block,
            goal,
            offset,
            approach,
            proposal_approach_distance,
            block_clearance,
            placement_error,
            scale,
            phase_durations,
        )
        end_velocity = trajectories._waypoint_velocity_device(
            current_phase + 1,
            start_effector,
            start_block,
            goal,
            offset,
            approach,
            proposal_approach_distance,
            block_clearance,
            placement_error,
            scale,
            phase_durations,
        )
        if current_phase == 0:
            start_velocity = effector_start_velocity[world]
        if current_phase == 7 and current_block + 1 < block_count:
            next_block = current_block + 1
            next_object_qpos = object_qpos_start + block_order[world, next_block] * 7
            next_block_position = wp.vec3(
                qpos[world, next_object_qpos + 0],
                qpos[world, next_object_qpos + 1],
                qpos[world, next_object_qpos + 2],
            )
            next_proposal = next_block * grasp_proposal_count
            next_offset = grasp_offset[world, next_proposal]
            next_direction = approach_direction[world, next_proposal]
            next_approach = (
                next_block_position
                + next_offset
                - next_direction * approach_distance[world, next_proposal]
            )
            previous_position = trajectories._waypoint_position_device(
                7,
                start_effector,
                start_block,
                goal,
                offset,
                approach,
                proposal_approach_distance,
                block_clearance,
                placement_error,
            )
            bridge_duration = scale * phase_durations[7] + (
                duration_scale[world, next_block] * phase_durations[0]
            )
            end_velocity = (next_approach - previous_position) / bridge_duration
        duration = scale * phase_durations[current_phase]
    elif current_phase == 8:
        contact_position = start_block + offset
        end_position = contact_position - approach * regrasp_pregrasp_distance
        start_velocity = effector_start_velocity[world]
        end_velocity = (contact_position - start_effector) / (
            regrasp_release_seconds + regrasp_descend_seconds
        )
    elif current_phase == 9:
        contact_position = start_block + offset
        start_position = contact_position - approach * regrasp_pregrasp_distance
        end_position = contact_position
        start_velocity = (contact_position - start_effector) / (
            regrasp_release_seconds + regrasp_descend_seconds
        )
        duration = regrasp_descend_seconds

    trajectory = trajectories._hermite_device(
        start_position,
        end_position,
        start_velocity,
        end_velocity,
        duration,
        phase_time[world],
    )
    # Release without loading the support stack. Preserve the actual pinch
    # pose during opening, then retreat vertically from that same pose.
    if current_phase == 6 and phase_time[world] == 0.0:
        effector_start[world] = site_xpos[world, pinch_site_id]
    if current_phase == 6:
        trajectory = wp.spatial_vector(effector_start[world], wp.vec3(0.0))
    elif current_phase == 7:
        trajectory = trajectories._hermite_device(
            effector_start[world], effector_start[world] + wp.vec3(0.0, 0.0, block_clearance),
            wp.vec3(0.0), wp.vec3(0.0), duration, phase_time[world],
        )
    desired_position = wp.spatial_top(trajectory)
    feedforward = wp.spatial_bottom(trajectory)
    position_error = desired_position - site_xpos[world, pinch_site_id]
    # During loaded transport, close the position loop on the held object.
    # A nominal pinch offset alone carries grasp error into every placement.
    if current_phase == 4 or current_phase == 5:
        position_error = desired_position - (block_position + offset)
    segment_progress = wp.clamp(phase_time[world] / duration, 0.0, 1.0)
    target_angles = target_rpy[world, proposal]
    target_orientation = wp.quat_rpy(
        target_angles[0], target_angles[1], target_angles[2]
    )
    desired_orientation = target_orientation
    if current_phase == 0 or current_phase == 8:
        orientation_progress = trajectories._quintic_smootherstep_device(segment_progress)
        desired_orientation = wp.quat_slerp(
            effector_start_quat[world],
            target_orientation,
            orientation_progress,
        )
    elif current_phase == 9:
        desired_orientation = target_orientation
    # Preserve the sampled cube pose, not only its center. While holding the
    # cube, correct its measured orientation; keep the resulting gripper pose
    # fixed while releasing so opening does not twist the placed object.
    if current_phase == 4 or current_phase == 5:
        cube_orientation = wp.quat(
            qpos[world, object_qpos + 4], qpos[world, object_qpos + 5],
            qpos[world, object_qpos + 6], qpos[world, object_qpos + 3],
        )
        desired_orientation = wp.quat_inverse(cube_orientation) * wp.quat_from_matrix(site_xmat[world, pinch_site_id])
    elif current_phase == 6 or current_phase == 7:
        desired_orientation = orientation_reference[world]
    orientation_reference[world] = desired_orientation
    rotation_error = arm._orientation_error(
        site_xmat[world, pinch_site_id], desired_orientation
    )
    task_velocity = wp.spatial_vector(
        feedforward + position_gain * position_error,
        orientation_gain * rotation_error,
    )
    joint_velocity = arm._damped_least_squares_six(jacobian, task_velocity, damping)
    for dof in range(6):
        action[world, dof] = wp.clamp(
            joint_velocity[dof] / velocity_limit,
            -max_normalized_arm_speed,
            max_normalized_arm_speed,
        )

    closure = 0.0
    if current_phase == 2:
        closure = segment_progress
    elif current_phase >= 3 and current_phase <= 5:
        closure = 1.0
    elif current_phase == 6:
        closure = 1.0 - segment_progress
    elif current_phase == 8:
        was_closed = failure_reason[world] >= 3 and failure_reason[world] <= 6
        if was_closed:
            closure = 1.0 - segment_progress
    elif current_phase == 9:
        closure = 0.0
    action[world, 6] = 2.0 * closure - 1.0
    next_time = phase_time[world] + control_timestep
    phase_time[world] = wp.min(next_time, duration)


@wp.kernel
def _record_block_trace(
    record_step: wp.array(dtype=int),
    capacity: int,
    qpos: wp.array2d(dtype=float),
    qvel: wp.array2d(dtype=float),
    action: wp.array2d(dtype=float),
    phase: wp.array(dtype=int),
    block_index: wp.array(dtype=int),
    recovery_count: wp.array(dtype=int),
    failure_reason: wp.array(dtype=int),
    orientation_reference: wp.array(dtype=wp.quat),
    completed: wp.array(dtype=bool),
    failed: wp.array(dtype=bool),
    trace_qpos: wp.array3d(dtype=float),
    trace_qvel: wp.array3d(dtype=float),
    trace_action: wp.array3d(dtype=float),
    trace_phase: wp.array2d(dtype=int),
    trace_block_index: wp.array2d(dtype=int),
    trace_recovery_count: wp.array2d(dtype=int),
    trace_failure_reason: wp.array2d(dtype=int),
    trace_orientation_reference: wp.array2d(dtype=wp.quat),
    trace_completed: wp.array2d(dtype=bool),
    trace_failed: wp.array2d(dtype=bool),
):
    world = wp.tid()
    step = record_step[0]
    if step >= capacity:
        return
    for index in range(qpos.shape[1]):
        trace_qpos[step, world, index] = qpos[world, index]
    for index in range(qvel.shape[1]):
        trace_qvel[step, world, index] = qvel[world, index]
    for index in range(7):
        trace_action[step, world, index] = action[world, index]
    trace_phase[step, world] = phase[world]
    trace_block_index[step, world] = block_index[world]
    trace_recovery_count[step, world] = recovery_count[world]
    trace_failure_reason[step, world] = failure_reason[world]
    trace_orientation_reference[step, world] = orientation_reference[world]
    trace_completed[step, world] = completed[world]
    trace_failed[step, world] = failed[world]


@wp.kernel
def _increment_record_step(record_step: wp.array(dtype=int), capacity: int):
    if record_step[0] < capacity:
        record_step[0] = record_step[0] + 1


@wp.kernel
def _mark_terminal_resets(
    completed: wp.array(dtype=bool),
    failed: wp.array(dtype=bool),
    reset_mask: wp.array(dtype=bool),
    reset_condition: wp.array(dtype=int),
    reset_world_count: wp.array(dtype=int),
    completed_episode_count: wp.array(dtype=int),
    failed_episode_count: wp.array(dtype=int),
):
    world = wp.tid()
    reset = completed[world] or failed[world]
    reset_mask[world] = reset
    if reset:
        wp.atomic_max(reset_condition, 0, 1)
        wp.atomic_add(reset_world_count, 0, 1)
        if completed[world]:
            wp.atomic_add(completed_episode_count, 0, 1)
        else:
            wp.atomic_add(failed_episode_count, 0, 1)


@wp.kernel
def _masked_copy_initial_state(
    reset_mask: wp.array(dtype=bool),
    initial_qpos: wp.array2d(dtype=float),
    initial_qvel: wp.array2d(dtype=float),
    qpos: wp.array2d(dtype=float),
    qvel: wp.array2d(dtype=float),
):
    world, index = wp.tid()
    if not reset_mask[world]:
        return
    if index < qpos.shape[1]:
        qpos[world, index] = initial_qpos[world, index]
    if index < qvel.shape[1]:
        qvel[world, index] = initial_qvel[world, index]


@wp.kernel
def _masked_copy_vec3_field(
    reset_mask: wp.array(dtype=bool),
    source: wp.array2d(dtype=wp.vec3),
    target: wp.array2d(dtype=wp.vec3),
):
    world, index = wp.tid()
    if reset_mask[world]:
        target[world, index] = source[world, index]


@wp.kernel
def _masked_copy_mat33_field(
    reset_mask: wp.array(dtype=bool),
    source: wp.array2d(dtype=wp.mat33),
    target: wp.array2d(dtype=wp.mat33),
):
    world, index = wp.tid()
    if reset_mask[world]:
        target[world, index] = source[world, index]


@wp.kernel
def _masked_copy_spatial_field(
    reset_mask: wp.array(dtype=bool),
    source: wp.array2d(dtype=wp.spatial_vector),
    target: wp.array2d(dtype=wp.spatial_vector),
):
    world, index = wp.tid()
    if reset_mask[world]:
        target[world, index] = source[world, index]


@wp.kernel
def _masked_zero_episode_step(
    reset_mask: wp.array(dtype=bool), episode_step: wp.array(dtype=int)
):
    world = wp.tid()
    if reset_mask[world]:
        episode_step[world] = 0


@wp.kernel
def _increment_episode_step(episode_step: wp.array(dtype=int)):
    world = wp.tid()
    episode_step[world] = episode_step[world] + 1


@wp.kernel
def _record_transition_ring(
    write_step: wp.array(dtype=int),
    consumed_step: wp.array(dtype=int),
    capacity: int,
    qpos: wp.array2d(dtype=float),
    qvel: wp.array2d(dtype=float),
    action: wp.array2d(dtype=float),
    episode: wp.array(dtype=int),
    global_world_ids: wp.array(dtype=int),
    episode_step: wp.array(dtype=int),
    completed: wp.array(dtype=bool),
    failed: wp.array(dtype=bool),
    failure_reason: wp.array(dtype=int),
    ring_qpos: wp.array3d(dtype=float),
    ring_qvel: wp.array3d(dtype=float),
    ring_action: wp.array3d(dtype=float),
    ring_episode: wp.array2d(dtype=int),
    ring_world_id: wp.array2d(dtype=int),
    ring_episode_step: wp.array2d(dtype=int),
    ring_terminal: wp.array2d(dtype=bool),
    ring_success: wp.array2d(dtype=bool),
    ring_failure_reason: wp.array2d(dtype=int),
):
    world = wp.tid()
    step = write_step[0]
    if step - consumed_step[0] >= capacity:
        return
    slot = step % capacity
    for index in range(qpos.shape[1]):
        ring_qpos[slot, world, index] = qpos[world, index]
    for index in range(qvel.shape[1]):
        ring_qvel[slot, world, index] = qvel[world, index]
    for index in range(7):
        ring_action[slot, world, index] = action[world, index]
    ring_episode[slot, world] = episode[world]
    ring_world_id[slot, world] = global_world_ids[world]
    ring_episode_step[slot, world] = episode_step[world]
    ring_terminal[slot, world] = completed[world] or failed[world]
    ring_success[slot, world] = completed[world]
    ring_failure_reason[slot, world] = failure_reason[world]


@wp.kernel
def _publish_transition_ring_step(
    write_step: wp.array(dtype=int),
    consumed_step: wp.array(dtype=int),
    capacity: int,
    overflow: wp.array(dtype=int),
):
    if write_step[0] - consumed_step[0] >= capacity:
        overflow[0] = 1
    else:
        write_step[0] = write_step[0] + 1


def transition_ring_indices(
    *, start_step: int, end_step: int, capacity: int
) -> np.ndarray:
    """Returns physical slots for one ordered bounded ring window."""
    if capacity < 1:
        raise ValueError("Transition ring capacity must be positive.")
    if start_step < 0 or end_step < 0:
        raise ValueError("Transition ring steps must be nonnegative.")
    if end_step < start_step:
        raise ValueError("Transition ring steps must be ordered.")
    if end_step - start_step > capacity:
        raise ValueError("Transition ring window exceeds capacity.")
    return np.arange(start_step, end_step, dtype=np.int64) % capacity


@dataclasses.dataclass(frozen=True)
class TransitionRingStatus:
    """Host snapshot of bounded transition-ring publication state."""

    capacity: int
    write_step: int
    consumed_step: int
    overflow: bool

    def __post_init__(self) -> None:
        if self.capacity < 1:
            raise ValueError("Transition ring capacity must be positive.")
        if self.consumed_step < 0 or self.write_step < self.consumed_step:
            raise ValueError("Transition ring counters are not ordered.")
        if self.available_steps > self.capacity:
            raise ValueError("Transition ring counters exceed capacity.")

    @property
    def available_steps(self) -> int:
        return self.write_step - self.consumed_step

    def checked_acknowledgement(self, through_step: int) -> int:
        """Validates one monotonic consumer acknowledgement."""
        if self.overflow:
            raise RuntimeError(
                "Transition producer overwrote an unconsumed slot."
            )
        if not self.consumed_step <= through_step <= self.write_step:
            raise ValueError(
                "Acknowledgement must lie within the available window."
            )
        return through_step


@dataclasses.dataclass(frozen=True)
class BlockCollectionMetrics:
    """Synchronized execution metrics for fused collection steps."""

    seconds: float
    control_steps: int
    physics_steps: int
    world_steps: int
    used_graph: bool

    @property
    def world_control_steps_per_second(self) -> float:
        return self.world_steps / self.seconds


class MJWarpBlockCollector:
    """GPU-resident block oracle, controller, and MJWarp physics graph."""

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        task: BlockTaskSpec,
        nworld: int,
        device: str,
        seed: int = 0,
        parameters: OracleParameters | None = None,
        graph: bool = True,
        record_capacity: int = 0,
        auto_reset: bool = False,
        transition_ring_capacity: int = 0,
        capacities: engine.BufferCapacities | None = None,
        damping: float = 0.01,
        position_gain: float = 6.0,
        orientation_gain: float = 4.0,
    ):
        if task.block_count * 7 + 14 != model.nq:
            raise ValueError(
                "Block task and OGBench model generalized state differ."
            )
        if damping <= 0.0 or position_gain <= 0.0 or orientation_gain <= 0.0:
            raise ValueError("Oracle gains and damping must be positive.")
        if record_capacity < 0:
            raise ValueError("Trace record capacity must be nonnegative.")
        if transition_ring_capacity < 0:
            raise ValueError("Transition ring capacity must be nonnegative.")
        if graph and auto_reset and not wp.is_conditional_graph_supported():
            raise RuntimeError(
                "Graph-captured terminal reset requires CUDA conditional "
                "graph support."
            )
        self.task = task
        self.parameters = parameters or OracleParameters()
        self.grasp_proposal_count = self.parameters.max_recovery_attempts + 1
        self.nworld = nworld
        self.graph_enabled = graph
        self.damping = damping
        self.position_gain = position_gain
        self.orientation_gain = orientation_gain
        self.record_capacity = record_capacity
        self.auto_reset = auto_reset
        self.transition_ring_capacity = transition_ring_capacity
        family_weights = np.asarray(
            [spec.weight for spec in self.parameters.grasp_families],
            dtype=np.float32,
        )
        family_cumulative_weights = np.cumsum(family_weights)
        family_cumulative_weights[-1] = 1.0
        family_parameter_bounds = np.asarray(
            [
                (
                    *spec.contact_radius_range,
                    *spec.contact_z_range,
                    *spec.tilt_range,
                    *spec.approach_distance_range,
                )
                for spec in self.parameters.grasp_families
            ],
            dtype=np.float32,
        )
        self.capacities = capacities or engine.BufferCapacities(
            nconmax=128,
            njmax=256,
            njmax_nnz=256 * model.nv,
        )
        self.engine = engine.MJWarpEngine(
            model,
            nworld=nworld,
            device=device,
            seed=seed,
            random_streams=task.block_count
            * (5 + 8 * self.grasp_proposal_count),
            graph=False,
            capacities=self.capacities,
        )
        self.controller = control_joint_velocity.MJWarpJointVelocityController(
            self.engine,
            control_parameters.ControllerSpec(physics_timestep_seconds=float(model.opt.timestep)),
            graph=False,
        )
        self._pinch_site_id = models._named_id(
            model, mujoco.mjtObj.mjOBJ_SITE, "ur5e/robotiq/pinch"
        )
        body_id = int(model.site_bodyid[self._pinch_site_id])
        self._root_body_id = int(model.body_rootid[body_id])
        with wp.ScopedDevice(self.engine.device):
            self.goal_positions = wp.array(
                np.broadcast_to(task.goal_positions, (nworld, task.block_count, 3)).astype(np.float32).copy(), dtype=wp.vec3
            )
            self.block_order = wp.array(
                np.broadcast_to(np.arange(task.block_count), (nworld, task.block_count)).astype(np.int32).copy(), dtype=int
            )
            self.phase_durations = wp.array(
                np.asarray(
                    self.parameters.phase_durations_seconds,
                    dtype=np.float32,
                ),
                dtype=float,
            )
            self.family_cumulative_weights = wp.array(
                family_cumulative_weights, dtype=float
            )
            self.family_parameter_bounds = wp.array(
                family_parameter_bounds, dtype=float
            )
            self.phase = wp.zeros(nworld, dtype=int)
            self.phase_time = wp.zeros(nworld, dtype=float)
            self.block_index = wp.zeros(nworld, dtype=int)
            self.recovery_count = wp.zeros(nworld, dtype=int)
            self.failure_reason = wp.zeros(nworld, dtype=int)
            self.completed = wp.zeros(nworld, dtype=bool)
            self.failed = wp.zeros(nworld, dtype=bool)
            self.effector_start = wp.zeros(nworld, dtype=wp.vec3)
            self.effector_start_velocity = wp.zeros(nworld, dtype=wp.vec3)
            self.effector_start_quat = wp.zeros(nworld, dtype=wp.quat)
            self.block_start = wp.zeros(nworld, dtype=wp.vec3)
            self.orientation_reference = wp.zeros(nworld, dtype=wp.quat)
            shape = (nworld, task.block_count)
            proposal_shape = (
                nworld,
                task.block_count * self.grasp_proposal_count,
            )
            self.grasp_family_id = wp.zeros(proposal_shape, dtype=int)
            self.grasp_offset = wp.zeros(proposal_shape, dtype=wp.vec3)
            self.approach_direction = wp.zeros(proposal_shape, dtype=wp.vec3)
            self.approach_distance = wp.zeros(proposal_shape, dtype=float)
            self.clearance = wp.zeros(shape, dtype=float)
            self.duration_scale = wp.zeros(shape, dtype=float)
            self.target_rpy = wp.zeros(proposal_shape, dtype=wp.vec3)
            self.intentional_error = wp.zeros(shape, dtype=bool)
            self.goal_error = wp.zeros(shape, dtype=wp.vec3)
            self.initial_qpos = wp.zeros((nworld, model.nq), dtype=float)
            self.initial_qvel = wp.zeros((nworld, model.nv), dtype=float)
            self.initial_site_xpos = wp.zeros(
                self.engine.data.site_xpos.shape, dtype=wp.vec3
            )
            self.initial_site_xmat = wp.zeros(
                self.engine.data.site_xmat.shape, dtype=wp.mat33
            )
            self.initial_subtree_com = wp.zeros(
                self.engine.data.subtree_com.shape, dtype=wp.vec3
            )
            self.initial_cdof = wp.zeros(
                self.engine.data.cdof.shape, dtype=wp.spatial_vector
            )
            self.reset_condition = wp.zeros(1, dtype=int)
            self.reset_world_count = wp.zeros(1, dtype=int)
            self.completed_episode_count = wp.zeros(1, dtype=int)
            self.failed_episode_count = wp.zeros(1, dtype=int)
            self.episode_step = wp.zeros(nworld, dtype=int)
            self.record_step = wp.zeros(1, dtype=int)
            if record_capacity:
                self.trace_qpos = wp.zeros(
                    (record_capacity, nworld, model.nq), dtype=float
                )
                self.trace_qvel = wp.zeros(
                    (record_capacity, nworld, model.nv), dtype=float
                )
                self.trace_action = wp.zeros(
                    (record_capacity, nworld, 7), dtype=float
                )
                trace_shape = (record_capacity, nworld)
                self.trace_phase = wp.zeros(trace_shape, dtype=int)
                self.trace_block_index = wp.zeros(trace_shape, dtype=int)
                self.trace_recovery_count = wp.zeros(trace_shape, dtype=int)
                self.trace_failure_reason = wp.zeros(trace_shape, dtype=int)
                self.trace_orientation_reference = wp.zeros(
                    trace_shape, dtype=wp.quat
                )
                self.trace_completed = wp.zeros(trace_shape, dtype=bool)
                self.trace_failed = wp.zeros(trace_shape, dtype=bool)
            if transition_ring_capacity:
                ring_vector_shape = (transition_ring_capacity, nworld)
                self.transition_ring_write_step = wp.zeros(1, dtype=int)
                self.transition_ring_consumed_step = wp.zeros(1, dtype=int)
                self.transition_ring_overflow = wp.zeros(1, dtype=int)
                self.ring_qpos = wp.zeros(
                    (transition_ring_capacity, nworld, model.nq), dtype=float
                )
                self.ring_qvel = wp.zeros(
                    (transition_ring_capacity, nworld, model.nv), dtype=float
                )
                self.ring_action = wp.zeros(
                    (transition_ring_capacity, nworld, 7), dtype=float
                )
                self.ring_episode = wp.zeros(ring_vector_shape, dtype=int)
                self.ring_world_id = wp.zeros(ring_vector_shape, dtype=int)
                self.ring_episode_step = wp.zeros(ring_vector_shape, dtype=int)
                self.ring_terminal = wp.zeros(ring_vector_shape, dtype=bool)
                self.ring_success = wp.zeros(ring_vector_shape, dtype=bool)
                self.ring_failure_reason = wp.zeros(
                    ring_vector_shape, dtype=int
                )
        self.graph = None
        self._reset_graph = None
        self.cold_capture_seconds = 0.0

    def _launch_initialize(self, reset_mask: wp.array) -> None:
        parameters = self.parameters
        wp.launch(
            _initialize_block_oracle,
            dim=self.nworld,
            inputs=[
                reset_mask,
                self.engine.random_u32,
                self.engine.episode,
                self.nworld,
                self.task.block_count,
                self.grasp_proposal_count,
                self.family_cumulative_weights,
                self.family_parameter_bounds,
                *parameters.clearance_range,
                *parameters.duration_scale_range,
                *parameters.target_yaw_range,
                parameters.intentional_error_probability,
                *parameters.intentional_error_xy_range,
                self.phase,
                self.phase_time,
                self.block_index,
                self.recovery_count,
                self.failure_reason,
                self.completed,
                self.failed,
                self.effector_start,
                self.effector_start_velocity,
                self.effector_start_quat,
                self.block_start,
                self.orientation_reference,
                self.grasp_family_id,
                self.grasp_offset,
                self.approach_direction,
                self.approach_distance,
                self.clearance,
                self.duration_scale,
                self.target_rpy,
                self.intentional_error,
                self.goal_error,
            ],
        )

    def _launch_device_terminal_reset(self) -> None:
        """Enqueues the masked reset branch used by the collector graph."""
        mujoco_warp.reset_data(
            self.engine.model,
            self.engine.data,
            reset=self.engine.reset_mask,
        )
        wp.launch(
            _masked_copy_initial_state,
            dim=(
                self.nworld,
                max(
                    self.engine.native_model.nq,
                    self.engine.native_model.nv,
                ),
            ),
            inputs=[
                self.engine.reset_mask,
                self.initial_qpos,
                self.initial_qvel,
                self.engine.data.qpos,
                self.engine.data.qvel,
            ],
        )
        wp.launch(
            _masked_copy_vec3_field,
            dim=self.engine.data.site_xpos.shape,
            inputs=[
                self.engine.reset_mask,
                self.initial_site_xpos,
                self.engine.data.site_xpos,
            ],
        )
        wp.launch(
            _masked_copy_mat33_field,
            dim=self.engine.data.site_xmat.shape,
            inputs=[
                self.engine.reset_mask,
                self.initial_site_xmat,
                self.engine.data.site_xmat,
            ],
        )
        wp.launch(
            _masked_copy_vec3_field,
            dim=self.engine.data.subtree_com.shape,
            inputs=[
                self.engine.reset_mask,
                self.initial_subtree_com,
                self.engine.data.subtree_com,
            ],
        )
        wp.launch(
            _masked_copy_spatial_field,
            dim=self.engine.data.cdof.shape,
            inputs=[
                self.engine.reset_mask,
                self.initial_cdof,
                self.engine.data.cdof,
            ],
        )
        self.engine.launch_reset_metadata(self.engine.reset_mask)
        self.controller.launch_sync_targets(self.engine.reset_mask)
        wp.launch(
            _masked_zero_episode_step,
            dim=self.nworld,
            inputs=[self.engine.reset_mask, self.episode_step],
        )
        self._launch_initialize(self.engine.reset_mask)

    def _launch_auto_reset(self) -> None:
        """Marks terminal worlds and conditionally launches the reset graph."""
        self.reset_condition.zero_()
        wp.launch(
            _mark_terminal_resets,
            dim=self.nworld,
            inputs=[
                self.completed,
                self.failed,
                self.engine.reset_mask,
                self.reset_condition,
                self.reset_world_count,
                self.completed_episode_count,
                self.failed_episode_count,
            ],
        )
        if self.graph_enabled:
            if self._reset_graph is None:
                raise RuntimeError(
                    "Terminal reset graph has not been captured."
                )
            wp.capture_if(self.reset_condition, on_true=self._reset_graph)
        else:
            wp.capture_if(
                self.reset_condition,
                on_true=self._launch_device_terminal_reset,
            )

    def _launch_transition_ring_record(self) -> None:
        """Writes and publishes one all-world pre-action transition tick."""
        wp.launch(
            _record_transition_ring,
            dim=self.nworld,
            inputs=[
                self.transition_ring_write_step,
                self.transition_ring_consumed_step,
                self.transition_ring_capacity,
                self.engine.data.qpos,
                self.engine.data.qvel,
                self.controller.action,
                self.engine.episode,
                self.engine.global_world_ids,
                self.episode_step,
                self.completed,
                self.failed,
                self.failure_reason,
                self.ring_qpos,
                self.ring_qvel,
                self.ring_action,
                self.ring_episode,
                self.ring_world_id,
                self.ring_episode_step,
                self.ring_terminal,
                self.ring_success,
                self.ring_failure_reason,
            ],
        )
        wp.launch(
            _publish_transition_ring_step,
            dim=1,
            inputs=[
                self.transition_ring_write_step,
                self.transition_ring_consumed_step,
                self.transition_ring_capacity,
                self.transition_ring_overflow,
            ],
        )

    def _launch_collection_step(self) -> None:
        spec = self.controller.spec
        if self.auto_reset:
            self._launch_auto_reset()
        wp.launch(
            _advance_block_oracle_phase,
            dim=self.nworld,
            inputs=[
                self.engine.data.qpos,
                self.engine.data.site_xpos,
                self._pinch_site_id,
                self.task.block_count,
                self.grasp_proposal_count,
                14,
                self.goal_positions,
                self.block_order,
                self.phase_durations,
                self.grasp_offset,
                self.approach_direction,
                self.approach_distance,
                self.clearance,
                self.duration_scale,
                self.intentional_error,
                self.goal_error,
                self.phase,
                self.phase_time,
                self.block_index,
                self.recovery_count,
                self.failure_reason,
                self.completed,
                self.failed,
                self.effector_start,
                self.block_start,
                self.task.success_radius,
                self.parameters.max_recovery_attempts,
                self.parameters.effector_tolerance,
                self.parameters.attachment_tolerance,
                self.parameters.minimum_lift_height,
                self.parameters.placement_tolerance,
                self.parameters.regrasp_release_seconds,
                self.parameters.regrasp_descend_seconds,
            ],
        )
        wp.launch(
            _block_oracle_action,
            dim=self.nworld,
            inputs=[
                self.engine.data.qpos,
                self.engine.data.qvel,
                self.engine.data.site_xpos,
                self.engine.data.site_xmat,
                self.engine.data.subtree_com,
                self.engine.data.cdof,
                self._pinch_site_id,
                self._root_body_id,
                self.task.block_count,
                self.grasp_proposal_count,
                14,
                self.goal_positions,
                self.block_order,
                self.phase_durations,
                self.grasp_offset,
                self.approach_direction,
                self.approach_distance,
                self.clearance,
                self.duration_scale,
                self.target_rpy,
                self.intentional_error,
                self.goal_error,
                self.phase,
                self.phase_time,
                self.block_index,
                self.recovery_count,
                self.failure_reason,
                self.completed,
                self.failed,
                self.effector_start,
                self.effector_start_velocity,
                self.effector_start_quat,
                self.block_start,
                self.orientation_reference,
                self.controller.action,
                spec.control_timestep_seconds,
                self.parameters.regrasp_pregrasp_distance,
                self.parameters.regrasp_release_seconds,
                self.parameters.regrasp_descend_seconds,
                self.damping,
                self.position_gain,
                self.orientation_gain,
                self.parameters.max_normalized_arm_speed,
                spec.joint_velocity_limits_rad_s[0],
            ],
        )
        if self.transition_ring_capacity:
            self._launch_transition_ring_record()
        wp.launch(
            _increment_episode_step,
            dim=self.nworld,
            inputs=[self.episode_step],
        )
        self.controller.launch_device_control_step()
        if self.record_capacity:
            wp.launch(
                _record_block_trace,
                dim=self.nworld,
                inputs=[
                    self.record_step,
                    self.record_capacity,
                    self.engine.data.qpos,
                    self.engine.data.qvel,
                    self.controller.action,
                    self.phase,
                    self.block_index,
                    self.recovery_count,
                    self.failure_reason,
                    self.orientation_reference,
                    self.completed,
                    self.failed,
                    self.trace_qpos,
                    self.trace_qvel,
                    self.trace_action,
                    self.trace_phase,
                    self.trace_block_index,
                    self.trace_recovery_count,
                    self.trace_failure_reason,
                    self.trace_orientation_reference,
                    self.trace_completed,
                    self.trace_failed,
                ],
            )
            wp.launch(
                _increment_record_step,
                dim=1,
                inputs=[self.record_step, self.record_capacity],
            )

    def reset(self, *, qpos: np.ndarray, qvel: np.ndarray) -> None:
        """Resets every world from one or one-per-world generalized state."""
        qpos = np.asarray(qpos, dtype=np.float32)
        qvel = np.asarray(qvel, dtype=np.float32)
        if qpos.shape == (self.engine.native_model.nq,):
            qpos = np.broadcast_to(qpos, (self.nworld, qpos.size)).copy()
        if qvel.shape == (self.engine.native_model.nv,):
            qvel = np.broadcast_to(qvel, (self.nworld, qvel.size)).copy()
        self.engine.reset(np.ones(self.nworld, dtype=bool))
        self.engine.set_state(qpos=qpos, qvel=qvel)
        self.controller.sync_targets()
        with wp.ScopedDevice(self.engine.device):
            wp.copy(self.initial_qpos, wp.array(qpos, dtype=float))
            wp.copy(self.initial_qvel, wp.array(qvel, dtype=float))
            wp.copy(self.initial_site_xpos, self.engine.data.site_xpos)
            wp.copy(self.initial_site_xmat, self.engine.data.site_xmat)
            wp.copy(self.initial_subtree_com, self.engine.data.subtree_com)
            wp.copy(self.initial_cdof, self.engine.data.cdof)
            self.reset_condition.zero_()
            self.reset_world_count.zero_()
            self.completed_episode_count.zero_()
            self.failed_episode_count.zero_()
            self.episode_step.zero_()
            self.record_step.zero_()
            if self.transition_ring_capacity:
                self.transition_ring_write_step.zero_()
                self.transition_ring_consumed_step.zero_()
                self.transition_ring_overflow.zero_()
            self._launch_initialize(self.engine.reset_mask)
            wp.synchronize_device(self.engine.device)
        self.engine.assert_healthy()
        if self.graph_enabled and self.graph is None:
            started = time.perf_counter()
            with wp.ScopedDevice(self.engine.device):
                if self.auto_reset:
                    with wp.ScopedCapture() as reset_capture:
                        self._launch_device_terminal_reset()
                    self._reset_graph = reset_capture.graph
                with wp.ScopedCapture() as capture:
                    self._launch_collection_step()
                self.graph = capture.graph
            self.cold_capture_seconds = time.perf_counter() - started

    def step(self, control_steps: int = 1) -> BlockCollectionMetrics:
        """Runs fused oracle/controller/physics steps with one final sync."""
        if control_steps < 1:
            raise ValueError("control_steps must be positive.")
        started = time.perf_counter()
        with wp.ScopedDevice(self.engine.device):
            for _ in range(control_steps):
                if self.graph_enabled:
                    if self.graph is None:
                        raise RuntimeError(
                            "Collector must be reset before stepping."
                        )
                    wp.capture_launch(self.graph)
                else:
                    self._launch_collection_step()
            wp.synchronize_device(self.engine.device)
        seconds = time.perf_counter() - started
        self.engine.assert_healthy()
        if self.transition_ring_capacity:
            overflow = int(self.transition_ring_overflow.numpy()[0])
            if overflow:
                raise RuntimeError(
                    "Transition producer overwrote an unconsumed slot."
                )
        physics_steps = (
            control_steps * self.controller.spec.physics_steps_per_control
        )
        return BlockCollectionMetrics(
            seconds=seconds,
            control_steps=control_steps,
            physics_steps=physics_steps,
            world_steps=control_steps * self.nworld,
            used_graph=self.graph_enabled,
        )

    def snapshot(self) -> dict[str, np.ndarray]:
        """Copies bounded acceptance state without exposing mutable buffers."""
        snapshot = self.engine.snapshot()
        snapshot.update(
            {
                "action": self.controller.action.numpy(),
                "phase": self.phase.numpy(),
                "phase_time": self.phase_time.numpy(),
                "block_index": self.block_index.numpy(),
                "recovery_count": self.recovery_count.numpy(),
                "failure_reason": self.failure_reason.numpy(),
                "completed": self.completed.numpy(),
                "failed": self.failed.numpy(),
                "episode_step": self.episode_step.numpy(),
                "reset_world_count": self.reset_world_count.numpy(),
                "completed_episode_count": self.completed_episode_count.numpy(),
                "failed_episode_count": self.failed_episode_count.numpy(),
                "effector_start_velocity": self.effector_start_velocity.numpy(),
                "effector_start_quat": self.effector_start_quat.numpy(),
                "orientation_reference": self.orientation_reference.numpy(),
                "grasp_family_id": self.grasp_family_id.numpy().reshape(
                    self.nworld,
                    self.task.block_count,
                    self.grasp_proposal_count,
                ),
                "grasp_offset": self.grasp_offset.numpy().reshape(
                    self.nworld,
                    self.task.block_count,
                    self.grasp_proposal_count,
                    3,
                ),
                "approach_direction": self.approach_direction.numpy().reshape(
                    self.nworld,
                    self.task.block_count,
                    self.grasp_proposal_count,
                    3,
                ),
                "approach_distance": self.approach_distance.numpy().reshape(
                    self.nworld,
                    self.task.block_count,
                    self.grasp_proposal_count,
                ),
                "clearance": self.clearance.numpy(),
                "duration_scale": self.duration_scale.numpy(),
                "target_rpy": self.target_rpy.numpy().reshape(
                    self.nworld,
                    self.task.block_count,
                    self.grasp_proposal_count,
                    3,
                ),
                "intentional_error": self.intentional_error.numpy(),
                "goal_error": self.goal_error.numpy(),
            }
        )
        return snapshot

    def transition_ring_status(self) -> TransitionRingStatus:
        """Downloads the three scalar counters for the transition ring."""
        if not self.transition_ring_capacity:
            raise RuntimeError(
                "Collector was created without a transition ring."
            )
        wp.synchronize_device(self.engine.device)
        return TransitionRingStatus(
            capacity=self.transition_ring_capacity,
            write_step=int(self.transition_ring_write_step.numpy()[0]),
            consumed_step=int(self.transition_ring_consumed_step.numpy()[0]),
            overflow=bool(self.transition_ring_overflow.numpy()[0]),
        )

    def acknowledge_transition_ring(self, through_step: int) -> None:
        """Acknowledges consumed ticks without changing producer ordering."""
        status = self.transition_ring_status()
        through_step = status.checked_acknowledgement(through_step)
        with wp.ScopedDevice(self.engine.device):
            wp.copy(
                self.transition_ring_consumed_step,
                wp.array([through_step], dtype=int),
            )

    def transition_ring(
        self, *, start_step: int, end_step: int
    ) -> dict[str, np.ndarray]:
        """Downloads one ordered, currently available transition window."""
        status = self.transition_ring_status()
        if start_step < status.consumed_step or end_step > status.write_step:
            raise ValueError(
                "Requested transition window is not currently available."
            )
        indices = transition_ring_indices(
            start_step=start_step,
            end_step=end_step,
            capacity=self.transition_ring_capacity,
        )
        return {
            "qpos": self.ring_qpos.numpy()[indices],
            "qvel": self.ring_qvel.numpy()[indices],
            "action": self.ring_action.numpy()[indices],
            "episode": self.ring_episode.numpy()[indices],
            "world_id": self.ring_world_id.numpy()[indices],
            "episode_step": self.ring_episode_step.numpy()[indices],
            "terminal": self.ring_terminal.numpy()[indices],
            "success": self.ring_success.numpy()[indices],
            "failure_reason": self.ring_failure_reason.numpy()[indices],
        }

    def trace(self) -> dict[str, np.ndarray]:
        """Downloads the bounded device trace recorded by the fused graph."""
        if not self.record_capacity:
            raise RuntimeError("Collector was created without trace storage.")
        wp.synchronize_device(self.engine.device)
        count = min(int(self.record_step.numpy()[0]), self.record_capacity)
        return {
            "qpos": self.trace_qpos.numpy()[:count],
            "qvel": self.trace_qvel.numpy()[:count],
            "action": self.trace_action.numpy()[:count],
            "phase": self.trace_phase.numpy()[:count],
            "block_index": self.trace_block_index.numpy()[:count],
            "recovery_count": self.trace_recovery_count.numpy()[:count],
            "failure_reason": self.trace_failure_reason.numpy()[:count],
            "orientation_reference": self.trace_orientation_reference.numpy()[
                :count
            ],
            "completed": self.trace_completed.numpy()[:count],
            "failed": self.trace_failed.numpy()[:count],
        }


@timing.measured('startup.gpu')
def make_collector(
    model, positions, worlds, steps, seed, device="cuda:0", *, native_models=None,
    capacity_models=None,
):
    """Creates the existing expert with precise top-down grasps and no injected errors."""
    params = OracleParameters(
        phase_durations_seconds=(1.4, 1.2, 0.6, 1.4, 1.8, 2.0, 1.0, 1.2),
        duration_scale_range=(1.0, 1.0),
        clearance_range=(0.16, 0.16),
        grasp_families=(
            GraspFamilySpec(
                GraspFamily.TOP_CENTER,
                1.0,
                (0.0, 0.0),
                (0.012, 0.012),
                (0.0, 0.0),
                (0.14, 0.14),
            ),
            *(
                dataclasses.replace(f, weight=0.0)
                for f in _DEFAULT_GRASP_FAMILIES[1:]
            ),
        ),
        target_yaw_range=(0.0, 0.0),
        intentional_error_probability=0.0,
        max_recovery_attempts=1,
        placement_tolerance=0.010,
        effector_tolerance=0.035,
        attachment_tolerance=0.055,
        minimum_lift_height=0.025,
    )
    task = BlockTaskSpec(
        "assembly", positions, positions, success_radius=0.006
    )
    variants = [model] * worlds if native_models is None else list(native_models)
    if len(variants) != worlds or variants[0] is not model:
        raise ValueError("One native model per world, with the base model first.")
    # Buffers are allocated once; slots may later load any model in the bank
    # (world_slots.ModelPool.select), so size them for every model it holds.
    sized = variants + list(capacity_models or ())
    capacities = [models.scene_capacities(m).make_data_kwargs() for m in sized]
    capacity = engine.BufferCapacities(
        **{key: max(c[key] for c in capacities) for key in capacities[0]}
    )
    collector = MJWarpBlockCollector(
        model,
        task=task,
        nworld=worlds,
        device=device,
        seed=seed,
        parameters=params,
        record_capacity=steps,
        capacities=capacity,
    )
    collector.engine.native_models = tuple(variants)
    if any(m is not model for m in variants):
        # No collector graph exists until reset; every subsequent capture reads
        # these arrays. Each CPU planner retains its corresponding native model.
        collector.engine.model = model_batch.stack_models(variants, device)
    return collector


def set_task(collector, goals, orders):
    """Changes only goal/order buffers between collection episodes."""
    with wp.ScopedDevice(collector.engine.device):
        wp.copy(
            collector.goal_positions,
            wp.array(np.asarray(goals, dtype=np.float32), dtype=wp.vec3),
        )
        wp.copy(
            collector.block_order,
            wp.array(np.asarray(orders, dtype=np.int32), dtype=int),
        )
