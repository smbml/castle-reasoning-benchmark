"""Pinned HF download behavior with real hash/path logic and a fake service."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import castle_reasoning.data.download as download

REVISION = "f" * 40
CAMPAIGN = "campaign-tasks-20261011"
ATTEMPT = "shards/2ms-b0-000/run00/attempt00216"
OLDER_CAMPAIGN = "campaign-1k-20261010"


class DownloadTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.root = self.directory / "dataset"
        self.cache = self.directory / "cache"
        self.cache.mkdir()
        self.index = {
            "demonstrations": [
                {
                    "target_id": "U0008",
                    "parts": 7,
                    "attempt_directory": ATTEMPT,
                }
            ]
        }
        self.contents = {
            f"{CAMPAIGN}/index.json": json.dumps(self.index).encode(),
            f"{CAMPAIGN}/campaign.json": b"{}",
            f"{CAMPAIGN}/{ATTEMPT}/plan.json": b'{"target_id":"U0008"}',
            f"{CAMPAIGN}/{ATTEMPT}/model.mjb": b"binary-model",
            f"{CAMPAIGN}/{ATTEMPT}/replay.npz": b"recorded-states",
            f"{CAMPAIGN}/{ATTEMPT}/execution_config.json": b"{}",
            f"{CAMPAIGN}/{ATTEMPT}/execution_order.json": b"{}",
        }
        self.older_index = {
            "demonstrations": [
                {"target_id": "S0074", "parts": 35,
                 "attempt_directory": ATTEMPT},
                {"target_id": "S0074", "parts": 35,
                 "attempt_directory": "topups/later/run00/attempt00000"},
            ]
        }
        self.contents.update({
            f"{OLDER_CAMPAIGN}/index.json": json.dumps(self.older_index).encode(),
            f"{OLDER_CAMPAIGN}/campaign.json": b"{}",
            f"{OLDER_CAMPAIGN}/{ATTEMPT}/plan.json": b'{"target_id":"S0074"}',
            f"{OLDER_CAMPAIGN}/{ATTEMPT}/model.mjb": b"older-model",
            f"{OLDER_CAMPAIGN}/{ATTEMPT}/replay.npz": b"older-recorded-states",
        })
        self.calls = []
        self.api = SimpleNamespace(
            get_paths_info=self.paths_info,
            list_repo_tree=self.list_tree,
            dataset_info=lambda **kwargs: SimpleNamespace(sha=REVISION),
        )
        self.addCleanup(mock.patch.stopall)
        self.api_factory = mock.patch(
            "huggingface_hub.HfApi", return_value=self.api
        ).start()
        mock.patch(
            "huggingface_hub.hf_hub_download", side_effect=self.fetch
        ).start()

    def metadata(self, name):
        content = self.contents[name]
        lfs = (
            {"sha256": hashlib.sha256(content).hexdigest()}
            if name.endswith(".mjb")
            else None
        )
        return SimpleNamespace(
            path=name,
            size=len(content),
            lfs=lfs,
            blob_id=hashlib.sha1(
                f"blob {len(content)}\0".encode() + content
            ).hexdigest(),
        )

    def paths_info(self, *, paths, **kwargs):
        return [self.metadata(name) for name in paths if name in self.contents]

    def list_tree(self, *, path_in_repo, **kwargs):
        return iter(
            self.metadata(name)
            for name in self.contents
            if name.startswith(path_in_repo + "/")
        )

    def fetch(self, *, filename, revision, **kwargs):
        self.assertEqual(revision, REVISION)
        self.calls.append(filename)
        path = self.cache / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(self.contents[filename])
        return str(path)

    def run_download(self, **kwargs):
        return download.download_tasks(["U0008"], self.root, revision=REVISION, **kwargs)

    def test_download_verifies_and_materializes_original_files(self):
        receipt = self.run_download()
        self.assertEqual(receipt["revision"], REVISION)
        expected = {
            name for name in self.contents
            if name.startswith(CAMPAIGN + "/")
            or name in {f"{OLDER_CAMPAIGN}/index.json",
                        f"{OLDER_CAMPAIGN}/campaign.json"}
        }
        self.assertEqual(set(receipt["files"]), expected)
        for name in expected:
            content = self.contents[name]
            self.assertEqual((self.root / name).read_bytes(), content)
            self.assertEqual(
                receipt["files"][name]["sha256"],
                hashlib.sha256(content).hexdigest(),
            )
        self.assertTrue((self.root / "DATASET_MANIFEST.json").is_file())

    def test_verified_existing_files_are_reused(self):
        self.run_download()
        count = len(self.calls)
        self.run_download()
        self.assertEqual(len(self.calls), count)

    def test_existing_mismatch_is_rejected_without_overwrite(self):
        self.run_download()
        target = self.root / CAMPAIGN / ATTEMPT / "model.mjb"
        target.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "mismatch"):
            self.run_download()
        self.assertEqual(target.read_bytes(), b"changed")

    def test_hf_cache_mismatch_does_not_become_an_original(self):
        cached = self.cache / CAMPAIGN / ATTEMPT / "model.mjb"
        cached.parent.mkdir(parents=True)
        cached.write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "mismatch"):
            self.run_download()
        self.assertFalse(
            (self.root / CAMPAIGN / ATTEMPT / "model.mjb").exists()
        )

    def test_unknown_task_is_reported_before_attempt_transfer(self):
        with self.assertRaisesRegex(ValueError, "U9999"):
            download.download_tasks(["U9999"], self.root, revision=REVISION)
        self.assertFalse(any("model.mjb" in name for name in self.calls))

    def test_transfer_budget_is_checked_before_bulk_download(self):
        with self.assertRaisesRegex(ValueError, "budget"):
            self.run_download(max_bytes=1)
        self.assertEqual(self.calls, [])
        self.assertFalse(self.root.exists())

    def test_required_attempt_file_is_not_silently_omitted(self):
        del self.contents[f"{CAMPAIGN}/{ATTEMPT}/replay.npz"]
        with self.assertRaisesRegex(ValueError, "replay.npz"):
            self.run_download()
        self.assertFalse(any("model.mjb" in name for name in self.calls))

    def test_traversal_in_index_is_rejected(self):
        self.index["demonstrations"][0]["attempt_directory"] = "../escape"
        self.contents[f"{CAMPAIGN}/index.json"] = json.dumps(
            self.index
        ).encode()
        with self.assertRaisesRegex(ValueError, "relative|unsafe"):
            self.run_download()

    def test_different_revision_cannot_mix_into_existing_dataset(self):
        self.run_download()
        with self.assertRaisesRegex(ValueError, "revision"):
            download.download_tasks(["U0008"], self.root, revision="e" * 40)

    def test_public_download_does_not_use_account_credentials(self):
        self.run_download()
        self.api_factory.assert_called_once_with(token=False)

    def test_data_symlink_cannot_escape_the_selected_root(self):
        outside = self.directory / "outside"
        outside.mkdir()
        self.root.mkdir()
        (self.root / CAMPAIGN).symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            self.run_download()
        self.assertEqual(list(outside.iterdir()), [])

    def test_missing_authoritative_hash_is_rejected(self):
        original = self.metadata

        def missing(name):
            entry = original(name)
            entry.blob_id = None
            entry.lfs = None
            return entry

        self.metadata = missing
        with self.assertRaisesRegex(ValueError, "authoritative"):
            self.run_download()
        self.assertEqual(self.calls, [])

    def test_download_extra_has_an_actionable_error(self):
        import sys

        with mock.patch.dict(sys.modules, {"huggingface_hub": None}):
            with self.assertRaisesRegex(RuntimeError, "download"):
                self.run_download()

    def test_older_campaign_task_uses_first_indexed_demonstration(self):
        receipt = download.download_tasks(["S0074"], self.root, revision=REVISION)
        self.assertEqual(receipt["tasks"], {"S0074": ATTEMPT})
        self.assertEqual(receipt["task_campaigns"], {"S0074": OLDER_CAMPAIGN})
        self.assertEqual(
            (self.root / OLDER_CAMPAIGN / ATTEMPT / "model.mjb").read_bytes(),
            b"older-model",
        )
        self.assertFalse(any("topups/later" in name for name in self.calls))
        self.assertFalse(any(name.startswith(f"{CAMPAIGN}/{ATTEMPT}/")
                             for name in self.calls))

    def test_mixed_campaign_paths_do_not_collide(self):
        receipt = download.download_tasks(
            ["S0074", "U0008", "S0074"], self.root, revision=REVISION
        )
        self.assertEqual(list(receipt["tasks"]), ["S0074", "U0008"])
        self.assertEqual(receipt["task_campaigns"],
                         {"S0074": OLDER_CAMPAIGN, "U0008": CAMPAIGN})
        for name, content in self.contents.items():
            self.assertEqual((self.root / name).read_bytes(), content)
        self.assertEqual(len(self.calls), len(set(self.calls)))

    def test_existing_u_receipt_extends_without_changing_verified_files(self):
        initial = self.run_download()
        initial.pop("task_campaigns", None)
        receipt_path = self.root / "DATASET_MANIFEST.json"
        receipt_path.write_text(json.dumps(initial))
        contents = {name: (self.root / name).read_bytes()
                    for name in initial["files"]}
        self.calls.clear()
        receipt = download.download_tasks(["S0074"], self.root, revision=REVISION)
        self.assertEqual(receipt["tasks"]["U0008"], initial["tasks"]["U0008"])
        self.assertEqual(receipt["task_campaigns"],
                         {"U0008": CAMPAIGN, "S0074": OLDER_CAMPAIGN})
        for name, content in contents.items():
            self.assertEqual((self.root / name).read_bytes(), content)
            self.assertEqual(receipt["files"][name], initial["files"][name])
        self.assertFalse(any(name.startswith(CAMPAIGN + "/")
                             for name in self.calls))

    def test_combined_transfer_budget_is_checked_before_attempt_download(self):
        metadata_bytes = sum(len(content) for name, content in self.contents.items()
                             if name.endswith(("/index.json", "/campaign.json")))
        with self.assertRaisesRegex(ValueError, "budget"):
            download.download_tasks(
                ["S0074", "U0008"], self.root, revision=REVISION,
                max_bytes=metadata_bytes + 1,
            )
        self.assertFalse(any("/attempt" in name for name in self.calls))
        self.assertFalse(self.root.exists())

    def test_missing_campaign_metadata_is_rejected_before_download(self):
        del self.contents[f"{OLDER_CAMPAIGN}/campaign.json"]
        with self.assertRaisesRegex(ValueError, "metadata"):
            self.run_download()
        self.assertEqual(self.calls, [])

    def test_conflicting_target_ids_across_campaigns_are_rejected(self):
        self.older_index["demonstrations"][0]["target_id"] = "U0008"
        self.contents[f"{OLDER_CAMPAIGN}/index.json"] = json.dumps(
            self.older_index
        ).encode()
        with self.assertRaisesRegex(ValueError, "ambiguous|campaigns"):
            self.run_download()
        self.assertFalse(any("model.mjb" in name for name in self.calls))

    def test_older_campaign_index_corruption_is_rejected(self):
        cached = self.cache / OLDER_CAMPAIGN / "index.json"
        cached.parent.mkdir(parents=True)
        cached.write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "mismatch"):
            self.run_download()
        self.assertFalse(self.root.exists())


if __name__ == "__main__":
    unittest.main()
