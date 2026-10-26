"""Runtime control parameters."""

import dataclasses
import math

import numpy as np


@dataclasses.dataclass(frozen=True)
class ControlConfig:
    """Explicit small-object controller parameters, recorded with every run."""

    grip_undersize: float = 0.0005
    grip_effort: float = 1.0
    target_error: float = 0.12
    attachment_error: float = 0.010
    angular_speed: float = 20.0
    physics_timestep: float = 0.0005


_ACTION_ORDER = (
    "shoulder_pan_velocity",
    "shoulder_lift_velocity",
    "elbow_velocity",
    "wrist_1_velocity",
    "wrist_2_velocity",
    "wrist_3_velocity",
    "gripper_closure",
)


@dataclasses.dataclass(frozen=True)
class ControllerSpec:
    """Physical interpretation of the inferred seven-dimensional action."""

    joint_velocity_limits_rad_s: tuple[float, ...] = (math.pi,) * 6
    physics_timestep_seconds: float = 0.002
    control_timestep_seconds: float = 0.02

    def __post_init__(self) -> None:
        limits = np.asarray(self.joint_velocity_limits_rad_s, dtype=np.float64)
        if limits.shape != (6,) or not np.isfinite(limits).all():
            raise ValueError("Six finite joint velocity limits are required.")
        if np.any(limits <= 0.0):
            raise ValueError("Joint velocity limits must be positive.")
        if (
            not np.isfinite(self.physics_timestep_seconds)
            or self.physics_timestep_seconds <= 0.0
            or not np.isfinite(self.control_timestep_seconds)
            or self.control_timestep_seconds <= 0.0
        ):
            raise ValueError("Control and physics timesteps must be positive.")
        ratio = self.control_timestep_seconds / self.physics_timestep_seconds
        if not math.isclose(ratio, round(ratio), abs_tol=1e-12):
            raise ValueError(
                "A control step must contain an integer number of "
                "physics steps."
            )

    @property
    def action_width(self) -> int:
        return len(_ACTION_ORDER)

    @property
    def physics_steps_per_control(self) -> int:
        return round(
            self.control_timestep_seconds / self.physics_timestep_seconds
        )

    @property
    def control_frequency_hz(self) -> float:
        return 1.0 / self.control_timestep_seconds


def grip_effort(plan, part):
    """Plate handling cannot change the force used for ordinary objects."""
    edge = plan.get("edge_access", {})
    if part in edge.get("plates", ()):
        return edge["grip_effort"]
    return ControlConfig().grip_effort
