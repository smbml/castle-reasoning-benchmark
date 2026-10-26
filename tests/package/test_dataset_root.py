"""Data roots are independent of the installed runtime and have no side effects."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import castle_reasoning.data.paths as paths


class DatasetRootTest(unittest.TestCase):
    def test_default_is_dataset_in_working_directory_without_creating_it(self):
        with tempfile.TemporaryDirectory() as directory:
            previous = Path.cwd()
            try:
                os.chdir(directory)
                with mock.patch.dict(os.environ, {}, clear=True):
                    root = paths.dataset_root()
                self.assertEqual(root, Path(directory) / "dataset")
                self.assertFalse(root.exists())
            finally:
                os.chdir(previous)

    def test_explicit_root_takes_precedence_over_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(
                os.environ, {"CASTLE_ROOT": directory + "/env"}
            ):
                root = paths.dataset_root(Path(directory) / "explicit")
            self.assertEqual(root, Path(directory) / "explicit")
            self.assertFalse(root.exists())

    def test_environment_root_is_expanded_and_resolved(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(
                os.environ, {"CASTLE_ROOT": directory + "/data/../tasks"}
            ):
                self.assertEqual(paths.dataset_root(), Path(directory) / "tasks")

    def test_empty_environment_value_uses_default(self):
        with mock.patch.dict(os.environ, {"CASTLE_ROOT": ""}):
            self.assertEqual(paths.dataset_root(), Path.cwd() / "dataset")

    def test_packaged_indexes_cover_both_campaigns_without_creating_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "absent"
            indexes = paths.dataset_indexes(root)
            self.assertEqual(indexes, [
                root / "campaign-1k-20261010/index.json",
                root / "campaign-tasks-20261011/index.json",
            ])
            self.assertFalse(root.exists())


if __name__ == "__main__":
    unittest.main()
