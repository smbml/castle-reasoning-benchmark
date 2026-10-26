"""Data generation generator."""

import collections
import dataclasses
import random

import numpy as np

import castle_reasoning.data.generation.geometry as geometry


def _part(x, y, z, width, length, height):
    return geometry.Part(
        (int(x), int(y), int(z)), (int(width), int(length), int(height))
    )


def _frame(x, y, z, width, length, rise, thick, enclosed):
    """A parameterized open frame; shared by roots and recursive enclosures."""
    parts = [
        _part(x, y, z, 2, 2 if enclosed else length, rise),
        _part(x + width - 2, y, z, 2, 2 if enclosed else length, rise),
    ]
    voids = []
    if enclosed:
        parts.extend(
            [
                _part(x, y + length - 2, z, 2, 2, rise),
                _part(x + width - 2, y + length - 2, z, 2, 2, rise),
            ]
        )
        parts.extend(
            [
                _part(x, y, z + rise, width, 2, thick),
                _part(x, y + length - 2, z + rise, width, 2, thick),
                # Raise the side rails one course to rest on the end beams.
                _part(x, y, z + rise + thick, 2, length, thick),
                _part(x + width - 2, y, z + rise + thick, 2, length, thick),
            ]
        )
        voids.append(_part(x + 2, y + 2, z, width - 4, length - 4, rise))
    else:
        parts.append(_part(x, y, z + rise, width, length, thick))
        voids.append(_part(x + 2, y, z, width - 4, length, rise))
    return parts, voids


def _extend(base, rng):
    sizes = [
        rng.choice([1, 2, 2, 3, 4]),
        rng.choice([1, 2, 3, 4]),
        rng.choice([1, 2, 3, 4, 5]),
    ]
    sizes[:2] = [min(sizes[i], base.size[i]) for i in range(2)]
    if rng.random() < 0.3:
        # Continuous wall sections complement isolated posts and open frames.
        axis = int(base.size[1] > base.size[0])
        sizes[axis] = base.size[axis]
    xy = [base.origin[i] + rng.randrange(base.size[i] - sizes[i] + 1) for i in range(2)]
    return [_part(*xy, base.end[2], *sizes)], []


def _branch(base, rng):
    axis = int(base.size[1] > base.size[0])
    if base.size[axis] < 6:
        return [], []
    other = 1 - axis
    height = rng.choice([2, 3, 4, 5, 6])
    foot = min(2, base.size[other])
    origin = list(base.origin)
    origin[2] = base.end[2]
    origin[other] += (base.size[other] - foot) // 2
    size = [2, 2, height]
    size[other] = foot
    left = geometry.Part(tuple(origin), tuple(size))
    origin[axis] += base.size[axis] - 2
    right = geometry.Part(tuple(origin), tuple(size))
    beam_origin = list(left.origin)
    beam_origin[2] += height
    beam_size = [*left.size[:2], rng.choice([1, 2])]
    beam_size[axis] = base.size[axis]
    beam = geometry.Part(tuple(beam_origin), tuple(beam_size))
    gap_origin = list(left.origin)
    gap_origin[axis] += 2
    gap_size = list(left.size)
    gap_size[axis] = base.size[axis] - 4
    return [left, right, beam], [geometry.Part(tuple(gap_origin), tuple(gap_size))]


def _cross(base, rng):
    axis = int(base.size[1] < base.size[0])
    size = [rng.choice([2, 3, 4, 6]), rng.choice([2, 3, 4, 6]), rng.choice([1, 2])]
    size[axis] = rng.choice([6, 8, 10, 12])
    origin = [round(base.center[i] - size[i] / 2) for i in range(2)]
    return [_part(*origin, base.end[2], *size)], []


def _span(parts, rng):
    pairs = [
        (a, b) for i, a in enumerate(parts) for b in parts[:i] if a.end[2] == b.end[2]
    ]
    rng.shuffle(pairs)
    for a, b in pairs:
        for axis in (0, 1):
            other = 1 - axis
            low = max(a.origin[other], b.origin[other])
            high = min(a.end[other], b.end[other])
            if high <= low:
                continue
            first, last = sorted((a, b), key=lambda p: p.origin[axis])
            gap = last.origin[axis] - first.end[axis]
            length = last.end[axis] - first.origin[axis]
            if gap < 1 or length > 20:
                continue
            width = min(high - low, rng.choice([2, 3, 4]))
            origin = [0, 0, first.end[2]]
            origin[axis], origin[other] = first.origin[axis], low
            size = [0, 0, rng.choice([1, 2])]
            size[axis], size[other] = length, width
            return [geometry.Part(tuple(origin), tuple(size))], []
    return [], []


def _enclose(base, rng):
    if min(base.size[:2]) < 6:
        return [], []
    return _frame(
        *base.origin[:2], base.end[2], *base.size[:2], rng.choice([2, 3, 4, 6]), 1, True
    )


def generate(seed: int, candidate: int) -> dict:
    """One bounded candidate; failures are returned rather than resampled away."""
    rng = random.Random(seed * 1_000_003 + candidate)
    ranges = [(7, 12), (14, 24), (28, 44)]
    minimum, maximum = ranges[candidate % 3]
    budget = rng.randint(minimum, maximum)
    width = rng.choice([6, 8, 10, 12, 14, 18, 22])
    length = rng.choice([6, 8, 10, 12, 16, 20])
    height_limit = [12, 20, 36][candidate // 3 % 3]
    root = rng.choices(["portal", "court", "slab"], [3, 4, 2])[0]
    if root == "slab":
        parts, voids = [_part(0, 0, 0, width, length, rng.choice([1, 2]))], []
    else:
        if root == "portal":
            length = rng.choice([2, 3, 4])
        parts, voids = _frame(
            0, 0, 0, width, length, rng.choice([2, 3, 4, 5]), 1, root == "court"
        )
    operations = [{"operation": "seed_" + root, "first": 0, "count": len(parts)}]
    root_rejection = geometry.construction_rejection([], parts)
    rejects = collections.Counter()
    # Profile probabilities vary across candidates; whole structures are not templates.
    weights = [
        rng.uniform(0.5, 2),
        rng.uniform(1, 3),
        rng.uniform(0.5, 2),
        rng.uniform(0.5, 2),
        rng.uniform(0.3, 1.5),
    ]
    attempts = 0
    while root_rejection is None and len(parts) < budget and attempts < 180:
        attempts += 1
        name = rng.choices(["extend", "branch", "span", "cross", "enclose"], weights)[0]
        base = rng.choices(parts, [max(p.size[:2]) for p in parts])[0]
        if name == "span":
            additions, gaps = _span(parts, rng)
        else:
            function = {
                "extend": _extend,
                "branch": _branch,
                "cross": _cross,
                "enclose": _enclose,
            }[name]
            additions, gaps = function(base, rng)
        if not additions:
            rejects["operation_unavailable"] += 1
            continue
        if len(parts) + len(additions) > budget:
            rejects["part_budget"] += 1
            continue
        if any(
            max(abs(v) for v in p.origin[:2]) > 24
            or max(p.end[:2]) > 30
            or p.end[2] > height_limit
            for p in additions
        ):
            rejects["envelope"] += 1
            continue
        reasons = geometry.validate(parts + additions, voids + gaps)
        if reasons:
            rejects.update(reasons)
            continue
        rejection = geometry.construction_rejection(parts, additions)
        if rejection:
            rejects[rejection] += 1
            continue
        operations.append(
            {"operation": name, "first": len(parts), "count": len(additions)}
        )
        parts.extend(additions)
        voids.extend(gaps)
    descriptors = geometry.describe(parts)
    reason = root_rejection
    if len(parts) > budget:
        reason = "root_exceeds_budget"
    elif len(parts) < minimum:
        reason = "growth_exhausted"
    elif descriptors["components"] != 1:
        reason = "disconnected"
    elif not geometry.equilibrium(parts):
        reason = "static_load_infeasible"
    return {
        "generator_version": "independent-load-construction-v1",
        "candidate": candidate,
        "seed": seed,
        "accepted": reason is None,
        "rejection": reason,
        "requested_part_count": budget,
        "height_limit": height_limit,
        "operation_weights": weights,
        "parts": [dict(id=i, **dataclasses.asdict(p)) for i, p in enumerate(parts)],
        "voids": [dataclasses.asdict(p) for p in voids],
        "supports": [dataclasses.asdict(c) for c in geometry.supports(parts)],
        "operations": operations,
        "construction_order": list(range(len(parts))) if reason is None else None,
        "disassembly_order": list(reversed(range(len(parts))))
        if reason is None
        else None,
        "operation_rejections": dict(rejects),
        "descriptors": descriptors,
        "fingerprint": geometry.fingerprint(parts),
        "topology_fingerprint": descriptors["topology_fingerprint"],
        "validation": {
            "independent_loads": root_rejection is None,
            "vertical_construction": root_rejection is None,
            "static_contact_inset": 0.05,
            "geometry": not geometry.validate(parts, voids),
            "static_gravity": reason != "static_load_infeasible"
            if reason in (None, "static_load_infeasible")
            else None,
            "dynamics": "unverified",
            "robot": "unverified",
        },
    }


def generate_pool(seed: int, count: int) -> dict:
    """Preserve every attempt's outcome and every accepted target description."""
    targets, attempts, seen = [], [], set()
    for candidate in range(count):
        target = generate(seed, candidate)
        if target["accepted"] and target["fingerprint"] in seen:
            target.update(accepted=False, rejection="duplicate_geometry")
        if target["accepted"]:
            targets.append(target)
            seen.add(target["fingerprint"])
        attempts.append(
            {
                k: v
                for k, v in target.items()
                if k not in ("parts", "voids", "supports", "operations")
            }
        )
    return {"seed": seed, "targets": targets, "attempts": attempts}


def select_coverage(targets: list[dict], count: int) -> list[int]:
    """Deterministic farthest-point coverage in normalized morphology features."""
    if len(targets) < count:
        raise ValueError("Not enough accepted candidates for the requested gallery.")
    keys = [
        "part_count",
        "height",
        "footprint",
        "fill_fraction",
        "depth",
        "branches",
        "multi_support_parts",
        "shape_types",
        "max_horizontal_aspect",
    ]
    features = np.asarray([[t["descriptors"][key] for key in keys] for t in targets])
    features = (features - features.min(0)) / np.maximum(np.ptp(features, axis=0), 1e-9)
    selected = [int(np.argmax(features[:, 6] + features[:, 5]))]
    distance = np.full(len(targets), np.inf)
    while len(selected) < count:
        distance = np.minimum(
            distance, np.sum((features - features[selected[-1]]) ** 2, axis=1)
        )
        distance[selected] = -1
        selected.append(int(np.argmax(distance)))
    return [targets[i]["candidate"] for i in selected]
