
"""The active-runtime build helper authenticates inputs and preserves outputs."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_active_builder_rejects_unverified_wheel(tmp_path):
    wheel = tmp_path / "upstream.whl"
    wheel.write_bytes(b"not the pinned upstream wheel")
    output = tmp_path / "runtime"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/package_active_runtime.py"),
            "--wheel",
            str(wheel),
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "wheel hash mismatch" in result.stderr.lower()
    assert not output.exists()


def test_native_package_retains_all_pinned_upstream_notices():
    import hashlib
    import json
    import tomllib

    root = ROOT / "third_party/mujoco-warp"
    upstream = json.loads((root / "upstream.json").read_text())
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    notices = {
        Path(name).name: digest
        for name, digest in upstream["metadata"].items()
        if "/licenses/" in name
    }
    assert notices
    for name, expected in notices.items():
        assert name in project["license-files"], name
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected
