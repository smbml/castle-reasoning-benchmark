"""Independent loads certify stable prefixes without balancing counterweights."""

import itertools
import json
from pathlib import Path

import pytest

import castle_reasoning.data.generation.generator as generator
import castle_reasoning.data.generation.geometry as geometry
import castle_reasoning.data.generation.plan_gallery as plan_gallery
import castle_reasoning.data.generation.waypoints as waypoints


def box(x, y, z, width, length, height):
    return geometry.Part((x, y, z), (width, length, height))


def bridge():
    return [box(0, 0, 0, 2, 2, 4), box(6, 0, 0, 2, 2, 4), box(0, 0, 4, 8, 2, 1)]


def test_bridge_and_centered_overhang_transmit_independent_loads():
    parts = bridge() + [box(3, 0, 5, 2, 2, 3)]
    assert geometry.construction_rejection([], parts) is None
    overhang = [box(0, 0, 0, 2, 2, 4), box(-2, 0, 4, 6, 2, 1)]
    assert geometry.construction_rejection([], overhang) is None
    assert geometry.equilibrium(parts, loads=[0, 0, 0, 1])


def test_tiny_load_cannot_borrow_the_receivers_weight_as_a_counterbalance():
    base = [box(0, 0, 0, 2, 2, 4), box(-2, 0, 4, 6, 2, 1)]
    load = box(3, 0, 5, 1, 2, 1)
    # Whole-prefix balance alone would accept this small overhanging load.
    assert geometry.equilibrium(base + [load])
    assert not geometry.equilibrium(base + [load], loads=[0, 0, 1])
    assert (
        geometry.construction_rejection(base, [load]) == "independent_load_infeasible"
    )


def test_later_counterweight_cannot_rescue_an_invalid_atomic_operation():
    base = [box(0, 0, 0, 2, 2, 4), box(-2, 0, 4, 6, 2, 1)]
    weights = [box(-2, 0, 5, 1, 2, 8), box(3, 0, 5, 1, 2, 8)]
    assert geometry.equilibrium(base + weights)
    assert (
        geometry.construction_rejection(base, weights) == "independent_load_infeasible"
    )
    assert len(base) == 2


def test_patch_margin_and_vertical_access_are_separate_checks():
    boundary = [box(0, 0, 0, 2, 2, 2), box(1, 0, 2, 2, 2, 1)]
    assert geometry.construction_rejection([], boundary, inset=0) is None
    assert (
        geometry.construction_rejection([], boundary) == "independent_load_infeasible"
    )
    assert geometry.construction_rejection(bridge(), [box(3, 0, 0, 2, 2, 2)]) == (
        "construction_vertical_collision"
    )


@pytest.mark.parametrize("loads", [[1], [1, -1, 1], [1, float("nan"), 1], [[1, 1, 1]]])
def test_invalid_loads_are_rejected(loads):
    with pytest.raises(ValueError, match="loads"):
        geometry.equilibrium(bridge(), loads=loads)


def test_independent_loads_support_every_support_closed_subset():
    parts = bridge() + [box(0, 0, 5, 2, 2, 3), box(6, 0, 5, 2, 2, 2)]
    assert geometry.construction_rejection([], parts) is None
    contacts = geometry.supports(parts)
    for keep in itertools.product([False, True], repeat=len(parts)):
        if not any(keep) or any(keep[c.upper] and not keep[c.lower] for c in contacts):
            continue
        assert geometry.equilibrium([p for i, p in enumerate(parts) if keep[i]])


def test_saved_order_bypasses_search_and_is_revalidated():
    plan = waypoints.plan(bridge(), construction_order=[0, 1, 2], max_search_nodes=0)
    assert plan["removal_order"] == [2, 1, 0]
    assert plan["assembly_order"] == [0, 1, 2]
    assert plan["order_source"] == "construction_witness"
    assert plan["search"]["expanded"] == 0
    assert plan["assembly_goals"] == plan["disassembly_goals"][::-1]
    for order in ([0, 0, 2], [0, 2], [2, 1, 0]):
        with pytest.raises(waypoints.PlanningFailure, match="construction_witness"):
            waypoints.plan(bridge(), construction_order=order)


def test_saved_order_rejects_collisions_and_unstable_intermediates():
    # Both completed targets are balanced; neither supplied order is valid.
    blocked = bridge() + [box(3, 0, 0, 2, 2, 2)]
    with pytest.raises(waypoints.PlanningFailure, match="construction_witness"):
        waypoints.plan(blocked, construction_order=list(range(4)))
    counterweights = [
        box(0, 0, 0, 2, 2, 4),
        box(-2, 0, 4, 6, 2, 1),
        box(-2, 0, 5, 1, 2, 8),
        box(3, 0, 5, 1, 2, 8),
    ]
    with pytest.raises(waypoints.PlanningFailure, match="construction_witness"):
        waypoints.plan(counterweights, construction_order=list(range(4)))


def test_generated_roots_and_growth_have_stable_reversible_prefixes():
    for target in generator.generate_pool(20260915, 12)["targets"]:
        parts = [
            geometry.Part(tuple(p["origin"]), tuple(p["size"])) for p in target["parts"]
        ]
        order = target["construction_order"]
        assert order == list(range(len(parts)))
        assert target["disassembly_order"] == order[::-1]
        assert target["validation"]["independent_loads"] is True
        assert target["validation"]["vertical_construction"] is True
        assert geometry.construction_rejection([], parts) is None
        for end in range(1, len(parts) + 1):
            assert geometry.equilibrium(parts[:end])
            assert waypoints.vertical_clear(parts, end - 1, range(end))


def test_all_five_historical_removal_failures_are_excluded():
    records = json.loads(
        (Path(__file__).parent / "fixtures/removal_failures.json").read_text()
    )
    assert len(records) == 5
    for record in records:
        parts = [geometry.Part(tuple(p[:3]), tuple(p[3:])) for p in record["parts"]]
        assert geometry.fingerprint(parts) == record["fingerprint"]
        assert geometry.equilibrium(parts)
        assert (
            geometry.construction_rejection([], parts) == "independent_load_infeasible"
        )


def test_gallery_assessment_consumes_the_recorded_order(tmp_path):
    target = {
        "candidate": 0,
        "seed": 0,
        "fingerprint": geometry.fingerprint(bridge()),
        "parts": [{"origin": p.origin, "size": p.size} for p in bridge()],
        "construction_order": [0, 1, 2],
    }
    gallery = tmp_path / "gallery"
    gallery.mkdir()
    (gallery / "targets.json").write_text(json.dumps([target]))
    (gallery / "selection.json").write_text(json.dumps({"coverage": [0]}))
    output = tmp_path / "plans"
    summary = plan_gallery.assess(gallery, output)
    assert summary["planned"] == 1
    plan = json.loads((output / "T0001.json").read_text())
    assert plan["removal_order"] == [2, 1, 0]
    assert plan["order_source"] == "construction_witness"
    assert plan["search"]["expanded"] == 0
