"""Build a hash-checked MJ-Warp package tree without editing an install."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


PATCH_ROOT = Path(__file__).resolve().parents[1] / "third_party/mujoco-warp"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(source, destination):
    """Verify pinned upstream bytes, copy them, and apply the repository patch."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.is_relative_to(source):
        raise ValueError("Build output must be outside the upstream package")
    manifest = json.loads((PATCH_ROOT / "upstream.json").read_text())
    files = manifest["files"]
    for name, expected in files.items():
        path = source / name
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"Upstream source mismatch: {path}")
    metadata = manifest["metadata"]
    for name, expected in metadata.items():
        path = source.parent / name
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"Upstream metadata mismatch: {path}")
    destination.mkdir(parents=True, exist_ok=False)
    for name in metadata:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source.parent / name, target)
    package = destination / "mujoco_warp"
    # Copy only verified inputs, excluding untracked modules and bytecode.
    for name in files:
        target = package / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)
    patch = PATCH_ROOT / "active-worlds.patch"
    subprocess.run(
        ["patch", "--batch", "--fuzz=0", "-p1", "-i", str(patch)],
        cwd=destination,
        check=True,
    )
    provenance = {
        "upstream_version": manifest["version"],
        "upstream_manifest_sha256": digest(PATCH_ROOT / "upstream.json"),
        "patch_sha256": digest(patch),
        "files": {name: digest(package / name) for name in files},
    }
    (destination / "active-worlds-build.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Pinned mujoco_warp package directory",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="New build directory"
    )
    args = parser.parse_args()
    print(build(args.source, args.output))


if __name__ == "__main__":
    main()
