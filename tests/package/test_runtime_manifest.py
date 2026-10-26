"""The installed source manifest covers the current package files."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_package_sources_match_their_manifest():
    package = ROOT / "src/castle_reasoning"
    manifest = json.loads(
        (package / "resources/runtime-sources.json").read_text()
    )["castle_reasoning"]
    assert "commit" not in manifest
    assert "origins" not in manifest
    actual = {
        str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in package.rglob("*.py")
    }
    assert manifest["files"] == actual
