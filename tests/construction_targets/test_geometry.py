"""Geometric and static-load contracts for procedural target descriptions."""

import dataclasses

import pytest

import castle_reasoning.data.generation.geometry as geometry


def box(x, y, z, width, length, height):
    return geometry.Part((x, y, z), (width, length, height))


def test_bridge_contacts_and_loads():
    parts = [box(0, 0, 0, 2, 2, 4), box(6, 0, 0, 2, 2, 4), box(0, 0, 4, 8, 2, 1)]
    contacts = geometry.supports(parts)
    assert {(c.lower, c.upper) for c in contacts} == {(0, 2), (1, 2)}
    assert geometry.validate(parts) == []
    assert geometry.equilibrium(parts)


def test_face_contact_is_allowed_but_overlap_and_floating_are_rejected():
    assert geometry.validate([box(0, 0, 0, 2, 2, 2), box(2, 0, 0, 2, 2, 2)]) == []
    assert "interpenetration" in geometry.validate(
        [box(0, 0, 0, 2, 2, 2), box(1, 0, 0, 2, 2, 2)]
    )
    assert "unsupported" in geometry.validate([box(0, 0, 1, 2, 2, 2)])
    # Edge-only and corner-only contact cannot carry a supported face load.
    assert not geometry.supports([box(0, 0, 0, 2, 2, 2), box(2, 2, 2, 2, 2, 2)])
    with pytest.raises(ValueError):
        box(0, 0, 0, 0, 2, 2)


def test_transmitted_upper_load_can_tip_an_individually_balanced_slab():
    balanced = [box(0, 0, 0, 2, 2, 4), box(-2, 0, 4, 6, 2, 1)]
    assert geometry.equilibrium(balanced)
    loaded = balanced + [box(3, 0, 5, 1, 2, 8)]
    assert geometry.validate(loaded) == []
    assert not geometry.equilibrium(loaded)


def test_fingerprint_ignores_order_translation_and_quarter_turn():
    parts = [box(0, 0, 0, 2, 4, 3), box(0, 0, 3, 6, 2, 1)]
    turned = [
        box(
            -p.origin[1] - p.size[1] + 10,
            p.origin[0] - 8,
            p.origin[2],
            p.size[1],
            p.size[0],
            p.size[2],
        )
        for p in parts
    ]
    assert geometry.fingerprint(parts) == geometry.fingerprint(turned[::-1])
    changed = [dataclasses.replace(parts[0], size=(2, 4, 4)), parts[1]]
    assert geometry.fingerprint(parts) != geometry.fingerprint(changed)


def test_void_intrusion_is_detected():
    parts = [box(0, 0, 0, 2, 2, 2)]
    assert geometry.validate(parts, voids=[box(1, 0, 0, 2, 2, 2)]) == [
        "opening_intrusion"
    ]
