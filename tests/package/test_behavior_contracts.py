"""Retained physics and execution definitions match the numerical regression contracts."""

import json
from pathlib import Path

from tests.helpers.source_contracts import definition_digest


def test_retained_numerical_and_execution_definitions():
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads((root / "tests/fixtures/runtime-contracts.json").read_text())
    assert len(manifest["definitions"]) >= 250
    for entry in manifest["definitions"]:
        module = entry["path"][4:-3].replace("/", ".")
        actual = definition_digest((root / entry["path"]).read_text(), entry["definition"], module, removal_delegate=entry.get("removal_delegate", False))
        assert actual == entry["sha256"], (entry["path"], entry["definition"])
