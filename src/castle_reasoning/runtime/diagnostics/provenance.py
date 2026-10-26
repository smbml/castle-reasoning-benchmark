"""Runtime diagnostics provenance."""

import hashlib
import importlib
import importlib.metadata
import json
from importlib.resources import files
from pathlib import Path

import mujoco_warp as mjw


def verify_source(root, specification, imported_path):
    """Validate one source snapshot and return its installed package identity."""
    root = root.resolve()
    expected = root / specification["module"]
    if imported_path.resolve() != expected.resolve():
        raise ValueError(f"Import outside the bundled source: {imported_path}")
    for relative, digest in specification["files"].items():
        path = root / relative
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"File outside the bundled source: {path}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Bundled source hash mismatch: {path}")
    identity = {"path": str(root), "version": specification["version"]}
    if "commit" in specification:
        identity["commit"] = specification["commit"]
    return identity


def bundled_sources():
    """Verify installed source snapshots and their distribution ownership."""
    manifest = json.loads(
        files("castle_reasoning").joinpath("resources/runtime-sources.json").read_text()
    )
    verified = {}
    for name, specification in manifest.items():
        imported = Path(importlib.import_module(name).__file__).resolve()
        candidates = importlib.metadata.distributions(name=specification["distribution"])
        distribution = next(
            (candidate for candidate in candidates
             if Path(candidate.locate_file(
                 specification["package_path"] + "/__init__.py"
             )).resolve() == imported),
            None,
        )
        if distribution is None:
            raise ValueError(f"Import outside the installed distribution: {imported}")
        if distribution.version != specification["version"]:
            raise ValueError(f"Unexpected runtime distribution version: {name}")
        root = Path(distribution.locate_file(specification["package_path"]))
        verified[name] = verify_source(root, specification, imported)
    return verified


def runtime_provenance():
    """Require the installed patch to match this checkout and its recorded bytes."""
    package = Path(mjw.__file__).resolve().parent
    manifest = package / "castle-build.json"
    if not manifest.is_file():
        raise RuntimeError("Missing Castle MJ-Warp build provenance")
    provenance = json.loads(manifest.read_text())
    patch = files("castle_reasoning").joinpath("resources/active-worlds.patch")
    if hashlib.sha256(patch.read_bytes()).hexdigest() != provenance["patch_sha256"]:
        raise RuntimeError("Patched MJ-Warp build uses a different repository patch")
    for name, expected in provenance["files"].items():
        if hashlib.sha256((package / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Patched MJ-Warp runtime was modified: {name}")
    return {key: value for key, value in provenance.items() if key != "files"}
