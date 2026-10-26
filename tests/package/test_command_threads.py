"""Command thread limits precede numerical-library initialization."""

import json
import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("module,overwrite", [
    ("castle_reasoning.data.generation.generate", False),
    ("castle_reasoning.data.assembly.continuous", True),
])
@pytest.mark.parametrize("existing", [None, "2"])
def test_command_sets_thread_limits_before_numpy(module, overwrite, existing, tmp_path):
    environment = dict(os.environ)
    caps = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
    for name in caps:
        environment.pop(name, None)
    if existing is not None:
        for name in caps:
            environment[name] = existing
    code = """
import ctypes, importlib, json, os, sys
importlib.import_module(sys.argv[1])
import numpy
pools = {}
for line in open('/proc/self/maps'):
    library = line.split()[-1]
    if 'openblas' not in library or '.so' not in library or library in pools:
        continue
    loaded = ctypes.CDLL(library)
    for name in ('scipy_openblas_get_num_threads64_', 'scipy_openblas_get_num_threads', 'openblas_get_num_threads64_', 'openblas_get_num_threads'):
        function = getattr(loaded, name, None)
        if function is not None:
            function.restype = ctypes.c_int
            pools[library] = function()
            break
assert pools, 'Expected the pinned NumPy/SciPy OpenBLAS libraries'
assert set(pools.values()) == {int(sys.argv[2])}, pools
print(json.dumps({'openblas': os.environ['OPENBLAS_NUM_THREADS'], 'omp': os.environ['OMP_NUM_THREADS']}))
"""
    expected = "1" if overwrite or existing is None else existing
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code, module, expected],
        env=environment, cwd=tmp_path, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"openblas": expected, "omp": expected}
