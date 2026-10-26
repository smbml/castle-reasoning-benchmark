# Runtime components and builds

## Components

Castle Reasoning provides its simulation engine adapter, joint-velocity
controller, motion planner and execution session under `castle_reasoning.runtime`.
The controller accepts six joint velocities and a gripper command. It does not
use OGBench's five-dimensional Cartesian controller. OGBench 1.2.1 supplies
workcell/model helpers and robot assets as an installed dependency.

The scripted teacher and collection workflows live in
`castle_reasoning.data.assembly`; evaluation uses the runtime session directly.

`check-runtime` verifies installed package versions and source hashes, the
active-world patch and its build manifest, then compiles a workcell and takes
a finite native MuJoCo step. It does not run a GPU episode.

## Reproduce the active runtime

`third_party/mujoco-warp/` packages MuJoCo-Warp 3.13.0 with active-world stepping
as `mujoco-warp==3.13.0+castle.1`. Paused worlds skip integration and iterative
constraint solving; full-batch physics and solver initialization remain.
The upstream LICENSE and AUTHORS and file modification notices are retained.

Download `mujoco_warp-3.13.0-py3-none-any.whl` from its
[release files](https://pypi.org/project/mujoco-warp/3.13.0/#files).
Its expected SHA-256 is
`cb1f74c36684ef5d2b610926d8979db8aa39fd78613a7130cf6a7ef802ee5e3c`.
Build into a fresh output directory:

```sh
pixi run --locked python scripts/package_active_runtime.py \
  --wheel /path/to/mujoco_warp-3.13.0-py3-none-any.whl \
  --output /tmp/castle-active-rebuild
```

The helper authenticates the upstream wheel, verifies every copied input,
applies `active-worlds.patch` with zero fuzz, and retains LICENSE, AUTHORS and
the build manifest. It refuses to overwrite existing output. The resulting
package installs normally; no import-path changes are needed.

`upstream.json` records original package and metadata hashes.
`mujoco_warp/castle-build.json` records patched file hashes and patch identity.
The benchmark's `resources/runtime-sources.json` records installed first-party
and OGBench source hashes. See [third-party notices](../THIRD_PARTY_NOTICES.md)
for component licenses and [tasks](tasks.md) for pinned dataset inputs.
