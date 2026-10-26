"""Installed wheels retain runtime identity, configs, and active physics."""

import importlib.metadata
import subprocess
import sys
import tempfile
import unittest
from importlib.resources import files

import castle_reasoning.runtime.diagnostics.provenance as provenance


class InstalledResourcesTest(unittest.TestCase):
    def test_installed_sources_verify_without_source_checkouts(self):
        result = provenance.bundled_sources()
        self.assertEqual(
            result["ogbench"]["commit"],
            "1d4140997f60c52c6fb0702ec100dc988b18c548",
        )
        self.assertEqual(
            result["castle_reasoning"]["version"],
            importlib.metadata.version("castle-reasoning-benchmark"),
        )

    def test_first_party_identity_has_no_private_source_metadata(self):
        result = provenance.bundled_sources()["castle_reasoning"]
        self.assertEqual(set(result), {"path", "version"})

    def test_castle_resources_are_installed(self):
        self.assertTrue(files("castle_reasoning").joinpath("resources/dataset.json").is_file())
        self.assertTrue(
            files('castle_reasoning.data.generation').joinpath("style.css").is_file()
        )
        self.assertTrue(
            files('castle_reasoning.data.generation')
            .joinpath("resources/generation.md")
            .is_file()
        )

    def test_installed_active_runtime_matches_its_build(self):
        self.assertEqual(
            importlib.metadata.version("mujoco-warp"), "3.13.0+castle.1"
        )
        result = provenance.runtime_provenance()
        self.assertEqual(result["upstream_version"], "3.13.0")

    def test_normal_imports_from_unrelated_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-c",
                    "import importlib, pathlib, sys; "
                    "names = ('castle_reasoning', 'castle_reasoning.data.generation', "
                    "'castle_reasoning.runtime', 'mujoco_warp'); "
                    "modules = [importlib.import_module(n) for n in names]; "
                    "assert all(pathlib.Path(m.__file__).resolve()."
                    "is_relative_to(pathlib.Path(sys.prefix).resolve()) "
                    "for m in modules); "
                    "from castle_reasoning.runtime.diagnostics import provenance; "
                    "provenance.bundled_sources()",
                ],
                cwd=directory,
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
