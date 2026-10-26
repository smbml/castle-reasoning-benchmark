"""Data paths."""

import json
import os
from importlib.resources import files
from pathlib import Path


def dataset_root(root: str | Path | None = None) -> Path:
    """Return the data directory without creating it or accessing the network."""
    value = root if root is not None else os.environ.get("CASTLE_ROOT")
    return (
        Path(value).expanduser().resolve() if value else Path.cwd() / "dataset"
    )


def dataset_indexes(root: str | Path | None = None) -> list[Path]:
    """Return the packaged campaign index paths without creating task data."""
    specification = json.loads(
        files("castle_reasoning").joinpath("resources/dataset.json").read_text()
    )
    directory = dataset_root(root)
    return [
        directory / campaign / "index.json"
        for campaign in specification["campaigns"]
    ]
