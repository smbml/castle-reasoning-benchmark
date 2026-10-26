"""Long beams should not use a large lever arm when the center is accessible."""

import json
from pathlib import Path

import numpy as np
import pytest

import castle_reasoning.runtime.planning.grasps as grasps
import castle_reasoning.runtime.planning.manipulation as manipulation
import castle_reasoning.runtime.simulation.models as models


@pytest.fixture(scope="module")
def beam_scene():
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/t0024_pregrasp.json").read_text()
    )
    model, _ = models.make_model(
        fixture["sizes"], fixture["target"], table_bounds=fixture["table_bounds"]
    )
    return fixture, model


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_accessible_beam_center_precedes_random_offset_grasps(beam_scene, seed):
    fixture, model = beam_scene
    planner = manipulation.Planner(
        model,
        fixture["sizes"],
        planning_seconds=15,
        seed=seed,
        shuffle_grasps=True,
        grasp_family="all",
    )
    result = planner.plan(
        np.array(fixture["qpos"]), fixture["part"], np.array(fixture["destination"])
    )
    assert np.linalg.norm(result.grasp.offset[:2]) < 0.000001
    for segment in result.segments:
        for pose in segment.path:
            assert planner.scene.valid(
                pose,
                result.part,
                segment.pads_allowed,
                segment.supports,
                segment.payload,
                segment.departures,
                segment.arrivals,
            )


def test_offset_grasps_remain_available_when_center_is_excluded(beam_scene):
    fixture, model = beam_scene
    planner = manipulation.Planner(
        model, fixture["sizes"], planning_seconds=15, grasp_family="all"
    )
    candidates = [
        planner.aperture.calibrate(g)
        for g in grasps.propose(fixture["sizes"][fixture["part"]], family="all")
    ]
    centered = [g for g in candidates if np.linalg.norm(g.offset[:2]) < 0.000001]
    assert centered and len(centered) < len(candidates)
    result = planner.plan(
        np.array(fixture["qpos"]),
        fixture["part"],
        np.array(fixture["destination"]),
        skip=centered,
    )
    assert np.linalg.norm(result.grasp.offset[:2]) > 0.001
