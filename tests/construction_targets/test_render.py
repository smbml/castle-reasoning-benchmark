"""Occlusion must depend on geometry rather than input part order."""

import numpy as np
from PIL import Image

import castle_reasoning.data.generation.geometry as geometry
import castle_reasoning.data.generation.render as render


def test_render_is_independent_of_part_order(tmp_path):
    parts = [
        geometry.Part((0, 0, 0), (2, 2, 4)),
        geometry.Part((6, 0, 0), (2, 2, 4)),
        geometry.Part((0, 0, 4), (8, 2, 1)),
    ]
    render.render(parts, tmp_path / "a.png", width=320, height=240)
    render.render(parts[::-1], tmp_path / "b.png", width=320, height=240)
    a, b = [np.asarray(Image.open(tmp_path / name)) for name in ("a.png", "b.png")]
    assert np.mean(a != b) < 0.001
    assert np.std(a) > 10
