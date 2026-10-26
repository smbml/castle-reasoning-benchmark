"""Collectors must not substitute upstream collision or solver functions."""

import inspect
import subprocess
import sys

import pytest
from mujoco_warp._src import collision_convex, solver

import castle_reasoning.data.assembly.batched_collect as batched_collect
import castle_reasoning.data.assembly.collect as collect
import castle_reasoning.data.assembly.continuous_collect as continuous_collect


def upstream_functions():
    return (
        collision_convex.ccd_kernel_builder,
        solver._JTDACJ_sparse,
        solver._update_gradient_JTCJ_dense,
        solver._update_gradient_cholesky_blocked,
    )


def test_fresh_import_does_not_load_runtime_overrides():
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys\n"
                "from castle_reasoning.data.assembly import batched_collect, continuous_collect, collect\n"
                "from castle_reasoning.runtime.diagnostics import physics_profile\n"
                "assert 'castle_reasoning.runtime.warp_contacts' not in sys.modules\n"
                "assert 'castle_reasoning.runtime.solver_kernels' not in sys.modules\n"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("mode", ["serial", "batched", "continuous"])
def test_upstream_functions_unchanged_inside_collection(mode, tmp_path):
    output = tmp_path / "collection"
    if mode == "serial":
        entry, body, args = (
            collect.execute,
            collect.execute,
            ({}, output, "invalid"),
        )
    elif mode == "batched":
        entry, body, args = (
            batched_collect.execute_many,
            batched_collect._execute,
            ([], output, []),
        )
    else:
        entry, body, args = (
            continuous_collect.execute,
            continuous_collect.execute,
            ([], output),
        )
    code = inspect.unwrap(body).__code__
    expected, observed = upstream_functions(), []

    def observe(frame, event, _arg):
        if event == "call" and frame.f_code is code:
            observed.append(upstream_functions())

    previous = sys.getprofile()
    try:
        sys.setprofile(observe)
        with pytest.raises(ValueError):
            entry(*args, wall_seconds=0)
    finally:
        sys.setprofile(previous)
    assert observed == [expected]
    assert upstream_functions() == expected
    assert not output.exists()
