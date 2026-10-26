"""Regression tests for demonstration-generation robustness fixes."""

from types import SimpleNamespace

import numpy as np
import pytest

import castle_reasoning.data.assembly.continuous_collect as continuous_collect
import castle_reasoning.data.generation.geometry as geometry
import castle_reasoning.data.generation.waypoints as waypoints
import castle_reasoning.data.plans as plans
import castle_reasoning.runtime.control.batch_tracking as batch_tracking
import castle_reasoning.runtime.planning.manipulation as manipulation
import castle_reasoning.runtime.planning.plates as plates
import castle_reasoning.runtime.simulation.engine as engine
import castle_reasoning.runtime.simulation.model_batch as model_batch


def tower():
    # A plate on two piers plus an offset block: asymmetric in X and Y.
    return [
        geometry.Part((0, 0, 0), (2, 2, 4)),
        geometry.Part((8, 0, 0), (2, 2, 4)),
        geometry.Part((0, 0, 4), (10, 6, 1)),
        geometry.Part((6, 3, 5), (2, 2, 2)),
    ]


@pytest.mark.parametrize("turns", [1, 2, 3])
def test_quarter_turn_is_a_rigid_rotation(turns):
    parts = tower()
    rotated = geometry.quarter_turn(parts, turns)
    assert [p.size[2] for p in rotated] == [p.size[2] for p in parts]
    assert [sorted(p.size[:2]) for p in rotated] == [sorted(p.size[:2]) for p in parts]
    assert min(p.origin[0] for p in rotated) == 0
    assert min(p.origin[1] for p in rotated) == 0

    # Same contact graph and the same equilibrium verdict.
    def contacts(ps):
        return sorted((c.lower, c.upper) for c in geometry.supports(ps))

    assert contacts(rotated) == contacts(parts)
    assert geometry.equilibrium(rotated) == geometry.equilibrium(parts)
    assert geometry.quarter_turn(rotated, 4 - turns) == geometry.quarter_turn(parts, 0)


def test_quarter_turn_swaps_footprint_axes():
    rotated = geometry.quarter_turn([geometry.Part((0, 0, 0), (22, 8, 2))], 1)
    assert rotated[0].size == (8, 22, 2)


def test_pad_plan_keeps_existing_active_count():
    plan = {
        "sizes": [[0.02] * 3] * 3,
        "target": [[0.4, 0, 0.01]] * 3,
        "parking": [[0.5, 0, 0.01]] * 3,
    }
    padded = model_batch.pad_plan(plan, 5)
    assert padded["active_part_count"] == 3
    again = model_batch.pad_plan(padded, 5)
    assert again["active_part_count"] == 3
    assert again["sizes"] == padded["sizes"]


def _fake_engine(qpos, overflow):
    return SimpleNamespace(
        nworld=len(qpos),
        global_world_start=0,
        data=SimpleNamespace(
            overflow=SimpleNamespace(numpy=lambda: overflow),
            qpos=SimpleNamespace(numpy=lambda: qpos),
            qvel=SimpleNamespace(numpy=lambda: np.zeros((len(qpos), 4))),
            ctrl=SimpleNamespace(numpy=lambda: np.zeros((len(qpos), 2))),
        ),
    )


def test_world_health_isolates_a_nonfinite_world():
    qpos = np.zeros((3, 4))
    qpos[1, 2] = np.nan
    errors = batch_tracking._world_health(_fake_engine(qpos, np.zeros(3, int)))
    assert errors[0] is None and errors[2] is None
    assert "nonfinite qpos" in errors[1]


def test_shared_buffer_overflow_still_aborts():

    with pytest.raises(engine.SimulationOverflowError):
        batch_tracking._world_health(
            _fake_engine(np.zeros((3, 4)), np.array([0, 0, 2]))
        )


def _request(name, guard=True):
    segment = manipulation.Segment(name, np.zeros((2, 14)), np.zeros(2))
    return batch_tracking.Request(segment, None, -1, guard=guard)


def test_fixed_foundation_stays_at_goal_and_supports_removals():
    pass

    parts = [
        geometry.Part((0, 0, 0), (12, 8, 1)),  # too wide to grasp: foundation
        geometry.Part((0, 3, 1), (2, 2, 3)),
        geometry.Part((10, 3, 1), (2, 2, 3)),
        geometry.Part((0, 3, 4), (12, 2, 1)),
    ]
    plan = waypoints.plan(parts, staging="stacks", fixed=[0])
    assert plan["fixed_parts"] == [0]
    assert 0 not in plan["removal_order"]
    assert sorted(plan["removal_order"]) == [1, 2, 3]
    assert plan["parking"][0] == plan["target"][0]
    # Parts resting on the foundation are removable once nothing is above them.
    assert list(plans.removable(plan, {1, 2, 3})) == [3]
    grounded = [
        geometry.Part((0, 0, 0), (2, 2, 3)),
        geometry.Part((6, 0, 0), (2, 2, 3)),
    ]
    assert "fixed_parts" not in waypoints.plan(grounded, staging="stacks")


def test_initial_settle_extends_only_for_residual_motion():

    moving = {
        "passed": False,
        "pose_valid": True,
        "settled": False,
        "contacts": {"passed": True},
        "execution": {"failure": 0},
    }
    assert continuous_collect.can_extend_settle(moving)
    assert not continuous_collect.can_extend_settle(moving | {"pose_valid": False})
    assert not continuous_collect.can_extend_settle(
        moving | {"contacts": {"passed": False}}
    )
    assert not continuous_collect.can_extend_settle(
        moving | {"execution": {"failure": 5}}
    )


def test_edge_plate_plans_refuse_coarse_timestep(tmp_path):

    plan = {"target_id": "P", "edge_access": {"max_physics_timestep_s": 0.001}}
    with pytest.raises(ValueError, match="at most 0.001"):
        continuous_collect.execute([plan], tmp_path / "out", physics_timestep=0.002)
    assert not (tmp_path / "out").exists()


def test_plate_slides_while_lateral_error_is_within_half_the_tolerance():

    mover = plates.PlateMover.__new__(plates.PlateMover)
    mover.floor = 0
    mover.planner = SimpleNamespace(
        sizes=np.array([[0.08, 0.16, 0.01]]), _source_supports=lambda qpos, part: (0,)
    )
    qpos = np.zeros(21)
    qpos[14:21] = [0.4, -0.145, 0.005, 1, 0, 0, 0]
    destination = np.array([0.4, 0.0, 0.005])
    assert mover._can_slide(qpos, 0, destination)
    qpos[14] = 0.4011  # the ~1.1 mm drift that used to force an edge transfer
    assert mover._can_slide(qpos, 0, destination)
    qpos[14] = 0.4030  # beyond half of the 5 mm acceptance tolerance
    assert not mover._can_slide(qpos, 0, destination)
    qpos[14], qpos[16] = 0.4, 0.0075  # a height error still cannot be pushed away
    assert not mover._can_slide(qpos, 0, destination)
