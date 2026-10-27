"""Runtime preflight requires usable CUDA and exercises the installed engine."""

import json
import os
import subprocess
import sys

import pytest

from castle_reasoning.runtime.diagnostics import checks


def test_runtime_check_rejects_unavailable_cuda(monkeypatch):
    monkeypatch.setattr(checks.wp, "is_cuda_available", lambda: False)
    with pytest.raises(RuntimeError, match="NVIDIA GPU"):
        checks.check_runtime()


def test_runtime_cli_fails_without_a_visible_gpu():
    result = subprocess.run(
        [sys.executable, "-I", "-m", "castle_reasoning.cli", "check-runtime"],
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="-1"),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert "NVIDIA GPU" in result.stderr
    assert result.stdout == ""


@pytest.mark.gpu
@pytest.mark.skipif(
    os.environ.get("CONSTRUCTION_GPU_TESTS") != "1",
    reason="explicit local GPU opt-in required",
)
def test_runtime_cli_takes_a_real_gpu_step():
    result = subprocess.run(
        [sys.executable, "-I", "-m", "castle_reasoning.cli", "check-runtime"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["native_step_finite"] is True
    assert report["gpu_tested"] is True
    assert report["gpu_step_finite"] is True
    assert report["gpu_device"] == "cuda:0"
