# MuJoCo-Warp with active-world stepping

This directory packages [MuJoCo-Warp 3.13.0](https://pypi.org/project/mujoco-warp/3.13.0/)
with Castle Reasoning's active-world patch as `mujoco-warp==3.13.0+castle.1`.
The upstream [Apache-2.0 license](LICENSE) and notices are retained.

`active-worlds.patch` changes `_src/forward.py` and `_src/solver.py` and
exports `step_active` from `__init__.py`. Paused rows skip integration and
iterative constraint solving; full-batch physics and solver initialization
remain. `upstream.json` records SHA-256 for every copied upstream
package/metadata file. The installed `mujoco_warp/castle-build.json` records
the patched files and patch identity.

[`build_active_mujoco_warp.py`](../../scripts/build_active_mujoco_warp.py)
verifies the inputs and applies the patch with zero fuzz into a new directory.
[`package_active_runtime.py`](../../scripts/package_active_runtime.py)
authenticates the upstream wheel and assembles this installable distribution.
See [reproduction instructions](../../docs/runtime-build.md#reproduce-the-active-runtime).

The input wheel `mujoco_warp-3.13.0-py3-none-any.whl` has SHA-256
`cb1f74c36684ef5d2b610926d8979db8aa39fd78613a7130cf6a7ef802ee5e3c`.
