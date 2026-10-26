"""Existing payload contacts may separate but cannot authorize new collision."""

import json
from pathlib import Path

import numpy as np

import castle_reasoning.runtime.control.guards as guards
import castle_reasoning.runtime.planning.collision as collision
import castle_reasoning.runtime.planning.manipulation as manipulation
import castle_reasoning.runtime.simulation.models as models


def test_lift_departure_preserves_initial_gap_and_checks_new_obstacles():
    sizes = [[0.02] * 3] * 3
    model, qpos = models.make_model(
        sizes, [[0.4, 0, 0.03], [0.42, 0, 0.03], [0.4, 0, 0.07]]
    )
    scene = collision.Scene(model, qpos)
    payload = scene.attachment(qpos, 0)
    departures = scene.departure_contacts(qpos, 0)
    assert len(departures) == 1
    assert departures[0][0] == model.geom("object_1").id
    assert not scene.valid(qpos, 0, True, (), payload)
    assert scene.valid(qpos, 0, True, (), payload, departures)

    # Direct geometry queries retain the original measured departure bound.
    above = qpos.copy()
    above[16] += 0.005
    assert scene.valid(above, 0, True, (), scene.attachment(above, 0), departures)
    inward = qpos.copy()
    inward[14] += 0.001
    assert not scene.valid(inward, 0, True, (), scene.attachment(inward, 0), departures)
    overhead = qpos.copy()
    overhead[16] += 0.025
    hits = scene.violations(
        overhead, 0, True, (), scene.attachment(overhead, 0), departures
    )
    assert any("object_2" in hit.names for hit in hits)
    # No carried payload means no exception, including for the robot.
    assert all(geom not in scene.robot_geoms for geom, _ in departures)


def test_deep_existing_intersection_is_not_a_departure_contact():
    model, qpos = models.make_model(
        [[0.02] * 3] * 2, [[0.4, 0, 0.03], [0.419, 0, 0.03]]
    )
    scene = collision.Scene(model, qpos)
    assert not scene.departure_contacts(qpos, 0)


def test_nearby_departure_allows_positive_clearance_to_shrink():
    model, qpos = models.make_model(
        [[0.02] * 3] * 2, [[0.4, 0, 0.03], [0.4202, 0, 0.03]]
    )
    scene = collision.Scene(model, qpos)
    departures = scene.departure_contacts(qpos, 0)
    assert len(departures) == 1
    qpos[14] += 0.00015  # Still separated by 50 micrometers.
    assert scene.valid(qpos, 0, True, (), scene.attachment(qpos, 0), departures)
    qpos[14] += 0.0001  # Now penetrating by 50 micrometers.
    assert not scene.valid(qpos, 0, True, (), scene.attachment(qpos, 0), departures)


def test_departure_gap_is_enforced_by_device_limits_and_replay_audit():

    model, qpos = models.make_model(
        [[0.02] * 3] * 2, [[0.4, 0, 0.03], [0.42, 0, 0.03]]
    )
    scene = collision.Scene(model, qpos)
    departures = scene.departure_contacts(qpos, 0)
    segment = manipulation.Segment(
        "lift",
        qpos[None],
        np.array([0.5]),
        payload=scene.attachment(qpos, 0),
        departures=departures,
    )
    limits = guards.contact_limits(scene, segment, 0)
    obj, neighbor = scene.objects[0], scene.objects[1]
    assert np.isclose(limits[obj, neighbor], -0.0002)
    assert np.isclose(departures[0][1], -0.00002)
    # The planned geometry remains strict; compliant physical contact is bounded.
    for overlap, passes in [
        (0.000035, True),
        (0.000075, True),
        (0.00019, True),
        (0.00021, False),
    ]:
        shifted = qpos.copy()
        shifted[14] += overlap
        assert not scene.valid(
            shifted, 0, False, (), scene.attachment(shifted, 0), departures
        )
        hits = guards.audit_contacts(
            scene, shifted[None], 0, False, (), True, departures=departures
        )
        assert bool(hits) is not passes
        if hits:
            assert np.isclose(hits[0]["contacts"][0]["required"], -0.0002)
    assert not guards.audit_contacts(
        scene, qpos[None], 0, False, (), True, departures=departures
    )
    qpos[14] += 0.001
    assert guards.audit_contacts(
        scene, qpos[None], 0, False, (), True, departures=departures
    )


def test_touching_top_piece_has_a_complete_exposed_edge_pick():

    fixture = json.loads(
        (Path(__file__).parent / "fixtures/touching_top_piece.json").read_text()
    )
    model, _ = models.make_model(
        fixture["sizes"],
        fixture["target"],
        table_bounds=fixture["edge_access"]["table_bounds"],
    )
    planner = manipulation.Planner(
        model, fixture["sizes"], planning_seconds=20, grasp_family="all"
    )
    result = planner.plan(np.array(fixture["qpos"]), 6, np.array(fixture["parking"][6]))
    assert abs(result.grasp.offset[0]) > 0.005
    for segment in result.segments:
        assert bool(segment.departures) == (segment.name == "lift")
        for qpos in segment.path:
            assert planner.scene.valid(
                qpos,
                6,
                segment.pads_allowed,
                segment.supports,
                segment.payload,
                segment.departures,
            )
    lift = result.segments[3]
    assert planner.scene.valid(lift.path[-1], 6, True, (), lift.payload)
