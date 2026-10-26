"""Intentional placement contacts retain bounded penetration and replay checks."""

import json
from pathlib import Path

import numpy as np

import castle_reasoning.runtime.control.guards as guards
import castle_reasoning.runtime.planning.collision as collision
import castle_reasoning.runtime.planning.manipulation as manipulation
import castle_reasoning.runtime.simulation.models as models


def test_actual_touching_destination_has_a_complete_manipulation():
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/touching_destination.json").read_text()
    )
    model, _ = models.make_model(
        fixture["sizes"],
        fixture["target"],
        table_bounds=fixture["edge_access"]["table_bounds"],
    )
    planner = manipulation.Planner(
        model, fixture["sizes"], planning_seconds=8, grasp_family="all"
    )
    result = planner.plan(np.array(fixture["qpos"]), 5, np.array(fixture["target"][5]))
    assert result.part == 5
    for segment in result.segments:
        assert bool(segment.arrivals) == (
            segment.name in {"place", "setdown", "open", "retreat"}
        )
        for qpos in segment.path:
            assert planner.scene.valid(
                qpos,
                5,
                segment.pads_allowed,
                segment.supports,
                segment.payload,
                segment.departures,
                segment.arrivals,
            )


def test_arrival_bounds_allow_touch_but_not_intrusion_or_other_obstacles():

    model, qpos = models.make_model(
        [[0.02] * 3] * 3, [[0.4, 0, 0.03], [0.42, 0, 0.03], [0.4, 0, 0.07]]
    )
    scene = collision.Scene(model, qpos)
    arrivals = scene.destination_contacts(qpos, 0, qpos[14:17])
    assert arrivals == ((scene.objects[1], -0.0002),)
    payload = scene.attachment(qpos, 0)
    assert scene.valid(qpos, 0, True, (), payload, arrivals=arrivals)
    segment = manipulation.Segment(
        "open", qpos[None], np.array([0.5]), moving_object=True, arrivals=arrivals
    )
    limits = guards.contact_limits(scene, segment, 0)
    assert np.isclose(limits[scene.objects[0], scene.objects[1]], -0.0002)
    assert not guards.audit_contacts(
        scene, qpos[None], 0, True, (), False, moving_object=True, arrivals=arrivals
    )
    compressed = qpos.copy()
    compressed[14] += 0.00005  # Soft contact inside the existing execution bound.
    assert scene.valid(
        compressed, 0, True, (), scene.attachment(compressed, 0), arrivals=arrivals
    )
    # A settled soft contact that satisfies the motion bound is also eligible.
    assert scene.destination_contacts(qpos, 0, compressed[14:17]) == arrivals
    inward = qpos.copy()
    inward[14] += 0.00025
    assert not scene.valid(
        inward, 0, True, (), scene.attachment(inward, 0), arrivals=arrivals
    )
    assert guards.audit_contacts(
        scene, inward[None], 0, True, (), False, moving_object=True, arrivals=arrivals
    )
    assert not scene.destination_contacts(qpos, 0, inward[14:17])
    overhead = qpos.copy()
    overhead[16] += 0.025
    assert not scene.valid(
        overhead, 0, True, (), scene.attachment(overhead, 0), arrivals=arrivals
    )


def test_settled_neighbor_inside_existing_contact_bound_remains_placeable():
    fixture = json.loads(
        (
            Path(__file__).parent / "fixtures/settled_contact_destination.json"
        ).read_text()
    )
    model, _ = models.make_model(
        fixture["sizes"],
        fixture["target"],
        table_bounds=fixture["edge_access"]["table_bounds"],
    )
    planner = manipulation.Planner(
        model, fixture["sizes"], planning_seconds=8, grasp_family="all"
    )
    qpos, part = np.array(fixture["qpos"]), fixture["part"]
    destination, _ = planner.seating_destination(qpos, part, fixture["target"][part])
    contacts = dict(planner.scene.destination_contacts(qpos, part, destination))
    neighbor = model.geom(f"object_{fixture['neighbor']}").id
    assert contacts.get(neighbor) == -0.0002
    result = planner.plan(qpos, part, np.array(fixture["target"][part]))
    for segment in result.segments:
        for frame in segment.path:
            assert planner.scene.valid(
                frame,
                part,
                segment.pads_allowed,
                segment.supports,
                segment.payload,
                segment.departures,
                segment.arrivals,
            )
