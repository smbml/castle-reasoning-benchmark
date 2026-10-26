"""Nested staging preserves the full target cohort and reversible support order."""

import json
from pathlib import Path

import numpy as np
import pytest

import castle_reasoning.data.generation.edge_staging as edge_staging
import castle_reasoning.data.generation.geometry as geometry
import castle_reasoning.data.generation.nested_staging as nested_staging
import castle_reasoning.data.generation.waypoints as waypoints


@pytest.mark.parametrize(
    "case",
    json.loads((Path(__file__).parent / "fixtures/staging_cohort.json").read_text()),
    ids=lambda case: case["target_id"],
)
def test_full_cohort_has_nested_accessible_staging_without_losing_parts(case):
    sizes, goals = np.array(case["sizes"]), np.array(case["target"])
    original = goals.copy()
    parked, supports, edge = edge_staging.arrange(
        goals,
        sizes,
        case["removal_order"],
        case["max_grasp_width"],
        case["max_stack_height"],
        gap=case["parking_gap"],
        stack_layout="nested",
    )
    np.testing.assert_array_equal(goals, original)
    assert parked.shape == goals.shape
    parents = dict(supports)
    assert set(parents) == set(range(len(goals)))
    placed = set()
    for part in case["removal_order"]:
        parent = parents[part]
        assert parent == -1 or parent in placed
        if parent != -1:
            assert np.all(sizes[part, :2] <= sizes[parent, :2] + 1e-8)
            np.testing.assert_array_equal(parked[part, :2], parked[parent, :2])
            assert np.isclose(
                parked[part, 2] - sizes[part, 2] / 2,
                parked[parent, 2] + sizes[parent, 2] / 2,
            )
        placed.add(part)
    for part in reversed(case["removal_order"]):
        assert not any(p == part for child, p in supports if child in placed)
        placed.remove(part)
    ordinary = [i for i in parents if i not in edge["plates"]]
    assert np.all(
        parked[ordinary, 2] + sizes[ordinary, 2] / 2 <= case["max_stack_height"] + 1e-8
    )
    x0, x1, y0, y1 = edge["parking_bounds"]
    low = np.min(goals[:, :2] - sizes[:, :2] / 2, axis=0) - 0.045
    high = np.max(goals[:, :2] + sizes[:, :2] / 2, axis=0) + 0.045
    for i in ordinary:
        base = i
        while parents[base] != -1:
            base = parents[base]
        if base != i:
            assert parked[i, 2] + sizes[i, 2] / 2 <= 3 * min(sizes[base, :2]) + 1e-8
        half = sizes[i, :2] / 2
        assert not (
            np.all(parked[i, :2] + half > low) and np.all(parked[i, :2] - half < high)
        )
        assert np.all(parked[i, :2] - half >= np.array([x0, y0]) - 1e-8)
        assert np.all(parked[i, :2] + half <= np.array([x1, y1]) + 1e-8)
        for rectangle in edge["reserved_lanes"]:
            assert not (
                np.all(parked[i, :2] + half + case["parking_gap"] > rectangle[:2])
                and np.all(parked[i, :2] - half - case["parking_gap"] < rectangle[2:])
            )
        for j in ordinary:
            if j <= i or np.array_equal(parked[i, :2], parked[j, :2]):
                continue
            assert np.any(
                abs(parked[i, :2] - parked[j, :2])
                >= (sizes[i, :2] + sizes[j, :2]) / 2 + case["parking_gap"] - 1e-8
            )
    if case["target_id"] == "T0113":
        assert parents[5] != 11  # The large beam must not rest on the tiny cube.


def test_nested_waypoints_preserve_exact_reverse_and_reject_unknown_layout():

    parts = [geometry.Part((0, 0, z), (2, 2, 2)) for z in (0, 2, 4)]
    plan = waypoints.plan(parts, stack_layout="nested")
    assert plan["stack_layout"] == "nested"
    np.testing.assert_array_equal(
        plan["disassembly_goals"][::-1], plan["assembly_goals"]
    )
    with pytest.raises(ValueError, match="layout"):
        waypoints.plan(parts, stack_layout="missing")


@pytest.mark.parametrize("source_access", [False, True])
def test_restage_cli_preserves_target_and_rebuilds_complete_waypoints(
    tmp_path, source_access
):
    import subprocess
    import sys


    source = tmp_path / "T0000.json"
    plan = waypoints.plan(
        [geometry.Part((0, 0, z), (2, 2, 2)) for z in (0, 2, 4)], staging="edge"
    )
    plan.update(target_id="T0000", target_fingerprint="test-target")
    source.write_text(json.dumps(plan))
    output = tmp_path / "plans"
    command = [
        sys.executable,
        "-m", "castle_reasoning.data.generation.restage",
        "--plans",
        str(source),
        "--output",
        str(output),
    ]
    if source_access:
        command.append("--source-access-order")
    subprocess.run(command, check=True, capture_output=True, text=True)
    result = json.loads((output / source.name).read_text())
    for key in ("target", "sizes", "supports", "removal_order", "target_fingerprint"):
        assert result[key] == plan[key]
    assert result["stack_layout"] == "nested"
    if source_access:
        assert result["order_source"] == "nominal_pickup_access"
        assert result["source_access"]["physics_verified"] is False
    assert result["robot_execution"] == "unverified"
    assert result["disassembly_goals"][-1] == result["parking"]
    assert result["assembly_goals"] == result["disassembly_goals"][::-1]
    assert len(result["object_transport_waypoints"]) == len(plan["sizes"])
    assert source.read_text() == json.dumps(plan)
    # Existing plan sets are immutable; rerunning must not overwrite them.
    assert subprocess.run(command, capture_output=True, check=False).returncode != 0


def test_intrinsically_slender_part_is_not_made_taller_and_height_limit_remains():

    sizes = np.array([[0.01, 0.01, 0.06], [0.01, 0.01, 0.01]])
    goals = np.array([[0.4, 0, 0.03], [0.4, 0.03, 0.005]])
    bounds = (0.55, 0.75, -0.1, 0.1)
    _, supports = nested_staging.slots(goals, sizes, bounds, [0, 1], max_height=0.08)
    assert supports == [[0, -1], [1, -1]]
    with pytest.raises(waypoints.PlanningFailure, match="stack_height"):
        nested_staging.slots(goals, sizes, bounds, [0, 1], max_height=0.05)
    with pytest.raises(waypoints.PlanningFailure, match="parking_capacity"):
        nested_staging.slots(goals, sizes, (0.55, 0.555, -0.1, 0.1), [0, 1])
