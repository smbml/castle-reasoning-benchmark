"""The installed runtime is independent of dataset production policies."""

import ast
import subprocess
import sys
from pathlib import Path


def test_runtime_imports_without_data_production(tmp_path):
    result = subprocess.run(
        [sys.executable, "-I", "-c", """
import importlib
import importlib.util
import pathlib
import pkgutil
import sys
import castle_reasoning.runtime as runtime
for info in pkgutil.walk_packages(runtime.__path__, runtime.__name__ + '.'):
    importlib.import_module(info.name)
assert not any(n.startswith('castle_reasoning.data.assembly') for n in sys.modules)
assert pathlib.Path(runtime.__file__).resolve().is_relative_to(pathlib.Path(sys.prefix).resolve())
for name in ('castle_benchmark', 'robot_assembly', 'construction_targets', 'action_chunking_wm', 'scripts'):
    assert importlib.util.find_spec(name) is None, name
"""],
        cwd=tmp_path, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr


def test_runtime_has_no_data_workflow_imports():
    root = Path(__file__).resolve().parents[2] / "src/castle_reasoning/runtime"
    assert root.is_dir()
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("castle_reasoning.data"), path
            elif isinstance(node, ast.Import):
                assert not any(n.name.startswith("castle_reasoning.data") for n in node.names), path
