"""Seeded generation should produce connected structures with mixed parts."""

import castle_reasoning.data.generation.generator as generator
import castle_reasoning.data.generation.geometry as geometry


def test_root_cannot_exceed_sampled_part_budget():
    target = generator.generate(20260915, 48)
    assert target["requested_part_count"] == 7
    assert target["descriptors"]["part_count"] == 8
    assert not target["accepted"]
    assert target["rejection"] == "root_exceeds_budget"


def test_reproducible_pool_has_shapes_spans_and_branches():
    first = generator.generate_pool(20260915, 40)
    second = generator.generate_pool(20260915, 40)
    assert first == second
    accepted = first["targets"]
    assert len(accepted) >= 20
    assert len({t["fingerprint"] for t in accepted}) == len(accepted)
    assert all(t["descriptors"]["components"] == 1 for t in accepted)
    assert sum(t["descriptors"]["multi_support_parts"] > 1 for t in accepted) >= 10
    assert max(t["descriptors"]["part_count"] for t in accepted) >= 25
    assert min(t["descriptors"]["part_count"] for t in accepted) <= 12
    assert len({t["topology_fingerprint"] for t in accepted}) >= 15
    for target in accepted:
        assert len(target["parts"]) <= target["requested_part_count"]
        parts = [
            geometry.Part(tuple(p["origin"]), tuple(p["size"])) for p in target["parts"]
        ]
        assert geometry.validate(parts) == []
        assert geometry.equilibrium(parts)
        assert target["validation"]["dynamics"] == "unverified"
    selected = generator.select_coverage(accepted, 20)
    assert len(selected) == len(set(selected)) == 20
