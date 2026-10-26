"""Every published accepted target is addressable by its original campaign."""

import collections
import hashlib
import json
from pathlib import Path

from castle_reasoning.data import download

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/hf"


def test_all_published_targets_select_first_indexed_demo_in_their_campaign():
    indices = {
        "campaign-1k-20261010": json.loads(
            (FIXTURES / "campaign-1k-index.json").read_text()
        ),
        "campaign-tasks-20261011": json.loads(
            (FIXTURES / "campaign-tasks-index.json").read_text()
        ),
    }
    proof = json.loads((FIXTURES / "index-authentication.json").read_text())
    assert proof["revision"] == "0bf9ccdd51861eac5112280c6bbb601fdc3ca98c"
    assert len(proof["files"]) == len(indices)
    for record in proof["files"]:
        filename = (
            "campaign-1k-index.json"
            if record["path"].startswith("campaign-1k-")
            else "campaign-tasks-index.json"
        )
        content = (FIXTURES / filename).read_bytes()
        assert hashlib.sha256(content).hexdigest() == record["sha256"]
        git_hash = hashlib.sha1(f"blob {len(content)}\0".encode() + content)
        assert git_hash.hexdigest() == record["git_blob_sha1"]
    catalog = download._catalog(indices)
    assert len(catalog) == 1072
    assert collections.Counter(campaign for campaign, _ in catalog.values()) == {
        "campaign-1k-20261010": 570, "campaign-tasks-20261011": 502
    }
    for campaign, index in indices.items():
        first = {}
        for row in index["demonstrations"]:
            first.setdefault(row["target_id"], row)
        for target_id, row in first.items():
            assert catalog[target_id] == (campaign, row)
    counts = collections.Counter(
        row["target_id"] for index in indices.values()
        for row in index["demonstrations"]
    )
    assert collections.Counter(counts.values()) == {1: 503, 2: 562, 3: 7}
