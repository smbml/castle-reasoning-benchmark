"""A held object must not raise the obstacle ceiling above its own lift."""

import json
from pathlib import Path

import numpy as np

import castle_reasoning.runtime.planning.manipulation as manipulation
import castle_reasoning.runtime.simulation.models as models


def test_far_destination_uses_a_lower_collision_checked_transport():
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/t0234_transport_height.json").read_text()
    )
    model, _ = models.make_model(
        fixture["sizes"], fixture["target"], table_bounds=fixture["table_bounds"]
    )
    planner = manipulation.Planner(
        model, fixture["sizes"], planning_seconds=8, grasp_family="all", seed=19
    )
    qpos = np.array(fixture["qpos"])
    part = fixture["part"]
    result = planner.plan(qpos, part, np.array(fixture["destination"]))
    lift = next(s for s in result.segments if s.name == "lift")
    transit = next(s for s in result.segments if s.name == "transit")
    planner.scene.set_state(lift.path[-1], lift.payload)
    lift_height = planner.scene.data.site_xpos[planner.scene.site, 2]
    planner.scene.set_state(transit.path[-1], transit.payload)
    transit_height = planner.scene.data.site_xpos[planner.scene.site, 2]
    assert transit_height < lift_height - 0.02
    retreat = result.segments[-1]
    planner.scene.set_state(retreat.path[-1])
    assert (
        abs(planner.scene.data.site_xpos[planner.scene.site, 2] - transit_height)
        < 0.0003
    )
    for segment in result.segments:
        for state in segment.path:
            assert planner.scene.valid(
                state,
                part,
                segment.pads_allowed,
                segment.supports,
                segment.payload,
                segment.departures,
                segment.arrivals,
            )


def test_recorded_post_lift_state_has_a_complete_checked_continuation():
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/t0024_transport_height.json").read_text()
    )
    model, _ = models.make_model(
        fixture["sizes"], fixture["target"], table_bounds=fixture["table_bounds"]
    )
    planner = manipulation.Planner(model, fixture["sizes"], planning_seconds=8)
    qpos = np.array(fixture["qpos"])
    part = fixture["part"]
    payload = planner.scene.attachment(qpos, part)
    initial_height = planner.scene.data.site_xpos[planner.scene.site, 2]

    with planner.budget.scope(8):
        segments = planner.continuation(
            qpos,
            part,
            np.array(fixture["destination"]),
            payload,
            fixture["holding"],
            planner.aperture.closure(fixture["opening_width"]),
        )

    assert [segment.name for segment in segments] == [
        "transit",
        "place",
        "setdown",
        "open",
        "retreat",
    ]
    planner.scene.set_state(segments[0].path[-1], payload)
    assert abs(
        planner.scene.data.site_xpos[planner.scene.site, 2] - initial_height
    ) < 0.0002
    for segment in segments:
        for state in segment.path:
            assert planner.scene.valid(
                state,
                part,
                segment.pads_allowed,
                segment.supports,
                segment.payload,
                segment.departures,
                segment.arrivals,
            )


def test_pickup_lowers_an_unreachable_approach_above_distant_parking():
    fixture = json.loads(
        (
            Path(__file__).parent / "fixtures/s0323_distant_parking_approach.json"
        ).read_text()
    )
    count = fixture["active_part_count"]
    model, _ = models.make_model(
        fixture["sizes"],
        fixture["target"],
        table_bounds=fixture["table_bounds"],
        active_part_count=count,
    )
    planner = manipulation.Planner(
        model,
        fixture["sizes"],
        planning_seconds=8,
        grasp_family="all",
        active_part_count=count,
    )
    qpos = np.array(fixture["qpos"])
    part = fixture["part"]
    result = planner.plan(qpos, part, np.array(fixture["destination"]))
    approach, lift = result.segments[0], result.segments[3]
    assert (approach.name, lift.name) == ("approach", "lift")
    planner.scene.set_state(approach.path[-1])
    approach_height = planner.scene.data.site_xpos[planner.scene.site, 2]
    ceiling = planner.object_tops(qpos).max() + 0.08
    assert approach_height < ceiling
    planner.scene.set_state(lift.path[-1], lift.payload)
    assert (
        abs(planner.scene.data.site_xpos[planner.scene.site, 2] - approach_height)
        < 0.0003
    )
    for segment in result.segments:
        for state in segment.path:
            assert planner.scene.valid(
                state,
                part,
                segment.pads_allowed,
                segment.supports,
                segment.payload,
                segment.departures,
                segment.arrivals,
            )
