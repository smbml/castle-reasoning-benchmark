"""Data assembly records."""

import collections
import hashlib
import importlib.metadata
import os
from pathlib import Path

import castle_reasoning.runtime.control.joint_velocity as joint_velocity
import castle_reasoning.runtime.diagnostics.provenance as provenance
import castle_reasoning.runtime.simulation.engine as engine


def yield_summary(records, saved, data_seconds, transitions):
    """Includes every candidate, including unstable targets and failed replays."""
    attempted = len(records)
    verified = sum(r["accepted"] for r in records)
    return dict(
        attempted_targets=attempted,
        verified_targets=verified,
        saved_targets=saved,
        target_yield=verified / attempted if attempted else 0.0,
        rejections=dict(
            collections.Counter(r["rejection"] for r in records if not r["accepted"])
        ),
        data_generation_seconds=data_seconds,
        verified_trajectories_per_second=verified / data_seconds,
        saved_trajectories_per_second=saved / data_seconds,
        seconds_per_saved_trajectory=data_seconds / saved if saved else None,
        saved_action_targets=transitions,
        saved_action_targets_per_second=transitions / data_seconds,
        saved_simulated_seconds=transitions * 0.02,
        simulated_seconds_per_generation_second=transitions * 0.02 / data_seconds,
    )




def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _provenance():
    return dict(
        sources=provenance.bundled_sources(),
        runtime_source_sha256={
            module.__name__: sha256(Path(module.__file__))
            for module in (joint_velocity, engine, provenance)
        },
        packages={
            name: importlib.metadata.version(name)
            for name in ("mujoco", "mujoco-warp", "warp-lang", "numpy")
        },
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        code_sha256={
            str(Path("castle_reasoning") / p.relative_to(Path(__file__).parents[2])): sha256(p)
            for p in Path(__file__).parents[2].rglob("*.py")
        },
    )
