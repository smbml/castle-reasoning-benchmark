# Installation and task data

## Install the runtime

Use Linux x86_64 and [Pixi](https://pixi.sh/). Robot execution requires an
NVIDIA GPU and a compatible driver. Rendering uses EGL/OpenGL. The Pixi
environment includes Python 3.12, the runtime dependencies, and FFmpeg.

From a checkout of this repository:

```sh
pixi install --locked
pixi run --locked check-runtime
```

`check-runtime` requires an NVIDIA GPU with a working CUDA driver. It verifies
installed source/build hashes, compiles a workcell, and takes finite native
MuJoCo and CUDA MuJoCo-Warp steps. It reports `gpu_tested: true` and fails if
CUDA is unavailable or the GPU step fails. This is a small installation check;
full robot episodes are evaluated separately.

Pixi installs two normal distributions: this package and
`mujoco-warp==3.13.0+castle.1` from `third_party/mujoco-warp`. The latter
includes the active-world API used by the controller. Dependencies come from
the lockfile; no import-path configuration is needed.

## Download existing tasks

```sh
pixi run --locked download-tasks --tasks S0074,U0008
pixi run --locked download-tasks --tasks S0074,U0008 --root /path/to/task-data
```

Use a comma-separated list with `--tasks` to select multiple task IDs. The
downloader uses the [existing HF dataset](https://huggingface.co/datasets/alexmzhang/castle-reasoning)
at a pinned revision. All 1,072 accepted targets are addressable: 570 `S`
targets from `campaign-1k-20261010` and 502 `U` targets from
`campaign-tasks-20261011`. Mixed lists such as `S0074,U0008` select both
campaigns. For each requested target the downloader selects the first accepted
demonstration in its campaign's published index order, including targets with
multiple demonstrations.

Each task includes a plan, compiled model, recorded states, optional execution
settings/order, and the authenticated metadata from both campaigns. Original
files remain under their campaign directory; the downloader does not merge or
rewrite the published indices. Evaluation uses downloaded data and performs
no automatic downloads.

The data directory is chosen in this order: `--root`, `CASTLE_ROOT`, then
`dataset/` under the invoking directory. Downloading creates it; imports and
runtime checks do not. `CASTLE_ROOT` refers to task data, not installed code.

Inputs are checked against HF Git-blob IDs or LFS SHA-256 hashes, then recorded
with local SHA-256 hashes in `DATASET_MANIFEST.json`. Verified files are
reused. The receipt records each selected attempt in `tasks` and its campaign
in `task_campaigns`. An existing U-task data directory can be extended with S
tasks while preserving its verified files and selected attempts. A size/hash
mismatch raises an error rather than replacing the file.
The default transfer budget is 256 MiB; `--max-download-mb` sets another
budget. Metadata may be read to determine the selected input sizes before
the budget is checked. Public downloads do not use an account token.

## Run an evaluation harness

The [Castle LLM harness](https://github.com/lianegalanti/castle-harness)
supplies agent tools, scripted/LLM runners, and trajectory rendering. This
package supplies its simulator, planner, and controller. The harness repository
includes a locked Pixi environment with both runtime distributions pinned to
the same benchmark commit. See the [model evaluation quick start](../README.md#evaluate-a-model)
for clone, installation, endpoint configuration, and evaluation commands.

A consuming Pixi workspace must select both distributions. For adjacent local
checkouts, add these entries to its `pixi.toml`:

```toml
[pypi-dependencies]
castle-reasoning-benchmark = { path = "../castle-reasoning-benchmark", extras = ["download"] }
mujoco-warp = { path = "../castle-reasoning-benchmark/third_party/mujoco-warp" }
```

Git dependencies can select the same two packages from one reviewed commit;
the runtime's subdirectory is `third_party/mujoco-warp`. Keep their revisions
aligned. A consuming workspace needs its own committed lockfile: installing a
Git dependency does not inherit this repository's Pixi lock. See the harness documentation for its commands and GPU requirements.

## Python APIs

Import `castle_reasoning.runtime` for simulator, planning, control, execution,
rendering, and diagnostic modules. `runtime.execution.session.make_session`
creates simulator/controller state without a scripted teacher. Requested
manipulations use `runtime.execution.runner` and the batching adapter in
`runtime.execution.session`. `Runner.stage(name, goals, order)` executes the
caller-supplied order, including an empty order for terminal hold checks.
The data assembly runner supplies automatic removal candidates when its order
is omitted. Both use the same execution, rollback, trace and success logic.

Problem creation lives in `castle_reasoning.data.generation`, scripted
demonstrations in `.data.assembly`, and qualification/replay in
`.data.verification`. These workflows call the runtime. No former package
names or import-path configuration are provided. Historical generator labels
inside saved plans remain unchanged.

Hydra/OmegaConf are used only by the column qualification fixture under
`tests/helpers/`; they are development dependencies in the `test` extra.

## Source distributions

The source distribution includes `pixi.toml`, `pixi.lock`, the tests, build
helpers, and the patched runtime subdirectory. From an extracted source
distribution, use the same `pixi install --locked` workflow as a Git checkout.
A wheel installation still needs both installable distributions selected as
described above; bundling the runtime source does not install it automatically.

## Test and build

```sh
pixi run --locked test
pixi run --locked build
pixi run --locked python -m build third_party/mujoco-warp
```

The default suite skips tests that explicitly require GPU opt-in. Those tests
use `CONSTRUCTION_GPU_TESTS=1` and an available GPU. Test results and opt-in
skips should be reported separately.
