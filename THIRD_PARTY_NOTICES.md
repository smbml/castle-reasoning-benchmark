# Third-party notices

Castle Reasoning's first-party code and documentation are licensed under the
[MIT license](LICENSE). Third-party components retain their own terms.

## MuJoCo-Warp

`third_party/mujoco-warp/` and the packaged `resources/active-worlds.patch`
contain code from MuJoCo-Warp 3.13.0, Copyright 2025 The Newton Developers,
licensed under [Apache-2.0](LICENSES/MuJoCo-Warp-LICENSE.txt). The upstream
[AUTHORS](LICENSES/MuJoCo-Warp-AUTHORS.txt) list is retained unchanged.
Castle modifications add active-world stepping in `__init__.py`,
`_src/forward.py`, and `_src/solver.py`; each file carries a modification
notice. These files and the patch remain under Apache-2.0. The upstream
`test_data/collision_sdf/asset/README.md` retains its CC0 asset attribution.

The benchmark distribution includes the Apache-derived patch, so its SPDX
metadata is `MIT AND Apache-2.0`; the first-party code is MIT.

## OGBench

OGBench 1.2.1 is installed as a dependency under its
[MIT license](LICENSES/OGBench-LICENSE.txt). Its distribution retains the
UR5e BSD-3-Clause and Robotiq 2F-85 BSD-2-Clause asset notices. Those assets
are loaded from the installed dependency.

## Dataset

The separately downloaded Hugging Face dataset has its own licensing terms.
The code license does not assign a license to that dataset or its copied
index fixtures in `tests/fixtures/hf/`.
