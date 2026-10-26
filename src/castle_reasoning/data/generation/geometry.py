"""Data generation geometry."""

import dataclasses
import hashlib
import json

import numpy as np
from scipy import optimize, sparse


@dataclasses.dataclass(frozen=True)
class Part:
    """Cuboid specified by its minimum corner and XYZ dimensions in grid units."""

    origin: tuple[int, int, int]
    size: tuple[int, int, int]

    def __post_init__(self):
        if len(self.origin) != 3 or len(self.size) != 3 or min(self.size) <= 0:
            raise ValueError("Cuboids require three positive dimensions.")

    @property
    def end(self):
        return tuple(a + b for a, b in zip(self.origin, self.size, strict=True))

    @property
    def center(self):
        return tuple(a + b / 2 for a, b in zip(self.origin, self.size, strict=True))

    @property
    def volume(self):
        return self.size[0] * self.size[1] * self.size[2]


@dataclasses.dataclass(frozen=True)
class Contact:
    lower: int
    upper: int
    patch: tuple[int, int, int, int]


def quarter_turn(parts: list[Part], turns: int) -> list[Part]:
    """Rotate a whole target counterclockwise by `turns` x 90 degrees about Z.

    The structure is unchanged up to a rigid motion: part order, sizes, contacts
    and equilibrium are preserved; only its yaw in the workcell differs. Minimum
    corners are re-normalized so the rotated footprint starts at the origin.
    """
    turns %= 4
    rotated = list(parts)
    for _ in range(turns):
        # (x, y) -> (-y, x): a box [ox, ox+sx] x [oy, oy+sy] maps to
        # [-(oy+sy), -oy] x [ox, ox+sx] and swaps its XY dimensions.
        rotated = [
            Part(
                (-(p.origin[1] + p.size[1]), p.origin[0], p.origin[2]),
                (p.size[1], p.size[0], p.size[2]),
            )
            for p in rotated
        ]
    low = np.min([p.origin for p in rotated], axis=0)
    return [
        Part(
            (int(p.origin[0] - low[0]), int(p.origin[1] - low[1]), p.origin[2]), p.size
        )
        for p in rotated
    ]


def overlap(a: Part, b: Part) -> tuple[int, int, int]:
    return tuple(
        min(a.end[i], b.end[i]) - max(a.origin[i], b.origin[i]) for i in range(3)
    )


def supports(parts: list[Part]) -> list[Contact]:
    """Positive-area horizontal contacts; ground contacts are implicit."""
    result = []
    for upper, part in enumerate(parts):
        for lower, base in enumerate(parts):
            if base.end[2] != part.origin[2]:
                continue
            if min(overlap(base, part)[:2]) <= 0:
                continue
            result.append(
                Contact(
                    lower,
                    upper,
                    (
                        max(base.origin[0], part.origin[0]),
                        max(base.origin[1], part.origin[1]),
                        min(base.end[0], part.end[0]),
                        min(base.end[1], part.end[1]),
                    ),
                )
            )
    return result


def validate(parts: list[Part], voids: list[Part] = ()) -> list[str]:
    """Return geometric rejection reasons; do not infer physical solvability."""
    reasons = set()
    if any(p.origin[2] < 0 for p in parts):
        reasons.add("below_ground")
    for i, part in enumerate(parts):
        if any(min(overlap(part, other)) > 0 for other in parts[:i]):
            reasons.add("interpenetration")
        if any(min(overlap(part, void)) > 0 for void in voids):
            reasons.add("opening_intrusion")
    supported = {c.upper for c in supports(parts)}
    if any(p.origin[2] > 0 and i not in supported for i, p in enumerate(parts)):
        reasons.add("unsupported")
    return sorted(reasons)


def equilibrium(
    parts: list[Part], inset: float = 0.05, *, loads: list[float] | None = None
) -> bool:
    """Check vertical force and XY moment balance for all transmitted loads.

    Uniform density, gravity only, rigid parts, nonadhesive horizontal contacts.
    Contact forces act at four inset patch corners. No dynamics or grasp checks.
    Explicit loads replace volume-proportional weights and act at each part's COM;
    zero-load parts still transmit forces but supply no balancing weight.
    """
    weights = np.asarray([p.volume for p in parts] if loads is None else loads)
    if (
        weights.shape != (len(parts),)
        or not np.all(np.isfinite(weights))
        or np.any(weights < 0)
    ):
        raise ValueError("loads must contain one finite nonnegative weight per part.")
    if not 0 <= inset < 0.5:
        raise ValueError("Contact inset must lie in [0, 0.5).")
    if validate(parts):
        return False
    contacts = supports(parts)
    for i, part in enumerate(parts):
        if part.origin[2] == 0:
            contacts.append(Contact(-1, i, (*part.origin[:2], *part.end[:2])))
    rows, columns, values = [], [], []
    for c_index, contact in enumerate(contacts):
        x0, y0, x1, y1 = contact.patch
        dx, dy = (x1 - x0) * inset, (y1 - y0) * inset
        corners = [
            (x0 + dx, y0 + dy),
            (x0 + dx, y1 - dy),
            (x1 - dx, y0 + dy),
            (x1 - dx, y1 - dy),
        ]
        for corner, (x, y) in enumerate(corners):
            for index, sign in [(contact.upper, 1), (contact.lower, -1)]:
                if index < 0:
                    continue
                cx, cy, _ = parts[index].center
                for eq, value in enumerate((sign, sign * (x - cx), sign * (y - cy))):
                    rows.append(3 * index + eq)
                    columns.append(4 * c_index + corner)
                    values.append(value)
    nforces = 4 * len(contacts)
    if not nforces:
        return False
    matrix = sparse.coo_matrix(
        (values, (rows, columns)), shape=(3 * len(parts), nforces)
    ).tocsr()
    load = np.zeros(3 * len(parts))
    load[::3] = weights
    result = optimize.linprog(
        np.zeros(nforces), A_eq=matrix, b_eq=load, bounds=(0, None), method="highs"
    )
    return bool(result.success)


def construction_rejection(
    parts: list[Part], additions: list[Part], inset: float = 0.05
) -> str | None:
    """Check an extension of an already certified construction, without mutation.

    Each added weight must independently reach the ground through nonnegative
    contact forces. Summing these witnesses supports any construction prefix,
    without counterweights. Vertical insertion makes the reverse order extractable
    for the cuboids themselves; gripper and arm access are separate checks.
    """
    prefix = list(parts)
    for part in additions:
        reasons = validate(prefix + [part])
        if reasons:
            return reasons[0]
        if any(
            min(overlap(part, other)[:2]) > 0 and other.end[2] > part.origin[2]
            for other in prefix
        ):
            return "construction_vertical_collision"
        prefix.append(part)
        if not equilibrium(prefix, inset, loads=[0.0] * (len(prefix) - 1) + [1.0]):
            return "independent_load_infeasible"
    return None


def fingerprint(parts: list[Part]) -> str:
    """Remove XY translation, ID ordering, and global quarter-turn yaw."""
    variants = []
    rotated = parts
    for _ in range(4):
        offset = [min(p.origin[i] for p in rotated) for i in range(2)]
        rows = sorted(
            (p.origin[0] - offset[0], p.origin[1] - offset[1], p.origin[2], *p.size)
            for p in rotated
        )
        variants.append(json.dumps(rows, separators=(",", ":")))
        rotated = [
            Part(
                (-p.end[1], p.origin[0], p.origin[2]), (p.size[1], p.size[0], p.size[2])
            )
            for p in rotated
        ]
    return hashlib.sha256(min(variants).encode()).hexdigest()[:20]


def describe(parts: list[Part]) -> dict:
    """Compute morphology statistics and a topology color-refinement signature."""
    contacts = supports(parts)
    incoming = [[] for _ in parts]
    outgoing = [[] for _ in parts]
    adjacency = [set() for _ in parts]
    for contact in contacts:
        incoming[contact.upper].append(contact.lower)
        outgoing[contact.lower].append(contact.upper)
    for i, part in enumerate(parts):
        for j, other in enumerate(parts[:i]):
            lengths = overlap(part, other)
            if min(lengths) == 0 and sum(v > 0 for v in lengths) == 2:
                adjacency[i].add(j)
                adjacency[j].add(i)
    remaining, components = set(range(len(parts))), 0
    while remaining:
        stack = [remaining.pop()]
        components += 1
        while stack:
            neighbors = adjacency[stack.pop()] & remaining
            remaining -= neighbors
            stack.extend(neighbors)
    depths = [0] * len(parts)
    for i in sorted(range(len(parts)), key=lambda k: parts[k].origin[2]):
        depths[i] = 1 + max((depths[j] for j in incoming[i]), default=0)
    colors = [str((len(a), len(b))) for a, b in zip(incoming, outgoing, strict=True)]
    for _ in range(len(parts)):
        colors = [
            hashlib.sha256(
                json.dumps(
                    (
                        colors[i],
                        sorted(colors[j] for j in incoming[i]),
                        sorted(colors[j] for j in outgoing[i]),
                    )
                ).encode()
            ).hexdigest()[:16]
            for i in range(len(parts))
        ]
    topology = hashlib.sha256("".join(sorted(colors)).encode()).hexdigest()[:16]
    bounds = [
        max(p.end[i] for p in parts) - min(p.origin[i] for p in parts) for i in range(3)
    ]
    return {
        "part_count": len(parts),
        "dimensions": bounds,
        "height": bounds[2] / 2,
        "footprint": bounds[0] * bounds[1] / 4,
        "fill_fraction": round(sum(p.volume for p in parts) / np.prod(bounds), 4),
        "depth": max(depths),
        "branches": sum(len(a) > 1 for a in outgoing),
        "multi_support_parts": sum(len(a) > 1 for a in incoming),
        "shape_types": len({(*sorted(p.size[:2]), p.size[2]) for p in parts}),
        "max_horizontal_aspect": max(max(p.size[:2]) / min(p.size[:2]) for p in parts),
        "components": components,
        "topology_fingerprint": topology,
    }
