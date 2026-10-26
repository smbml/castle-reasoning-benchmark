"""A held payload must finish extracting before entering free-space transit."""

import json
from pathlib import Path

import numpy as np
import pytest

import castle_reasoning.runtime.control.tracking as tracking
import castle_reasoning.runtime.execution.recovery as recovery
import castle_reasoning.runtime.planning.arm as arm
import castle_reasoning.runtime.planning.manipulation as manipulation
import castle_reasoning.runtime.simulation.models as models


def recorded_recovery():
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/t0072_interrupted_lift.json").read_text()
    )
    model, _ = models.make_model(
        fixture["sizes"],
        fixture["target"],
        active_part_count=fixture["active_part_count"],
        table_bounds=fixture["table_bounds"],
        contact_impedance=fixture["contact_impedance"],
    )
    planner = manipulation.Planner(
        model,
        fixture["sizes"],
        active_part_count=fixture["active_part_count"],
    )
    start = np.array(fixture["lift_start"])
    interrupted = manipulation.Segment(
        "lift",
        np.array([start, fixture["lift_endpoint"]]),
        np.full(2, fixture["holding"]),
        True,
        tuple(fixture["supports"]),
        planner.scene.attachment(start, fixture["part"]),
    )
    return planner, fixture, interrupted


def test_recorded_held_recovery_can_extract_then_plan_transport():
    planner, fixture, interrupted = recorded_recovery()
    state, part = np.array(fixture["qpos"]), fixture["part"]
    destination = np.array(fixture["destination"])
    opening = fixture["opening"]
    assessment = recovery.assess(
        planner,
        state,
        np.array(fixture["qvel"]),
        part,
        np.array(fixture["reference"]),
    )
    assert assessment["ready"] and assessment["held"] and not assessment["supported"]
    with pytest.raises(arm.PlanningError, match="invalid_endpoint"):
        planner.continuation(
            state,
            part,
            destination,
            planner.scene.attachment(state, part),
            fixture["holding"],
            opening,
        )

    lift = recovery.remaining_lift(planner, state, part, interrupted)
    np.testing.assert_array_equal(lift.path[0], state)
    assert lift.departures
    assert np.all(lift.closure == fixture["holding"])
    np.testing.assert_allclose(
        tracking.references(lift)[:, 6], fixture["holding_action"], atol=1e-7
    )
    planner.scene.set_state(interrupted.path[-1])
    expected_position = planner.scene.data.site_xpos[planner.scene.site].copy()
    planner.scene.set_state(lift.path[-1], lift.payload)
    np.testing.assert_allclose(
        planner.scene.data.site_xpos[planner.scene.site], expected_position, atol=0.0002
    )
    continuation = planner.continuation(
        lift.path[-1],
        part,
        destination,
        lift.payload,
        fixture["holding"],
        opening,
    )
    for segment in [lift, *continuation]:
        for a, b in zip(segment.path[:-1], segment.path[1:], strict=True):
            planner._edge(
                a,
                b,
                part,
                segment.pads_allowed,
                segment.supports,
                segment.payload,
                segment.departures,
                segment.arrivals,
            )
    assert [s.name for s in continuation] == [
        "transit",
        "place",
        "setdown",
        "open",
        "retreat",
    ]


def test_remaining_lift_rejects_an_endpoint_still_inside_departure_clearance():
    planner, fixture, interrupted = recorded_recovery()
    state = np.array(fixture["qpos"])
    interrupted.path[-1] = state
    with pytest.raises(arm.PlanningError, match="collision"):
        recovery.remaining_lift(planner, state, fixture["part"], interrupted)
