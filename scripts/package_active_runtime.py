"""Rebuild the installed active-runtime source package from a verified wheel."""

import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

WHEEL_SHA256 = (
    "cb1f74c36684ef5d2b610926d8979db8aa39fd78613a7130cf6a7ef802ee5e3c"
)


def package(wheel: Path, output: Path) -> None:
    """Apply the unchanged Castle patch and preserve build/license metadata."""
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if hashlib.sha256(wheel.read_bytes()).hexdigest() != WHEEL_SHA256:
        raise ValueError("Upstream wheel hash mismatch")
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(
        prefix="castle-active-package-"
    ) as temporary:
        temporary = Path(temporary)
        upstream = temporary / "upstream"
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(upstream)
        active = temporary / "active"
        subprocess.run(
            [
                sys.executable,
                str(root / "scripts/build_active_mujoco_warp.py"),
                "--source",
                str(upstream / "mujoco_warp"),
                "--output",
                str(active),
            ],
            check=True,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        assembled = temporary / "distribution"
        assembled.mkdir()
        shutil.copytree(active / "mujoco_warp", assembled / "mujoco_warp")
        shutil.copyfile(
            active / "active-worlds-build.json",
            assembled / "mujoco_warp/castle-build.json",
        )
        notice_root = active / "mujoco_warp-3.13.0.dist-info/licenses"
        for name in ("LICENSE", "AUTHORS"):
            shutil.copyfile(notice_root / name, assembled / name)
        shutil.copyfile(
            root / "third_party/mujoco-warp/pyproject.toml",
            assembled / "pyproject.toml",
        )
        shutil.copytree(assembled, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        package(args.wheel, args.output)
    except (ValueError, FileExistsError, FileNotFoundError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    main()
