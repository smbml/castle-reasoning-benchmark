# Castle Reasoning

Castle Reasoning evaluates spatial reasoning through robot block assembly. An
agent chooses which parts to move and where to place them to reproduce a
target structure. The runtime plans and executes the robot motions, then checks the
assembled structure's poses, support contacts, and stability.

This package provides the task generator, pick-and-place planner, simulation
runtime, and tools for downloading existing benchmark tasks. The
[LLM evaluation harness](https://github.com/lianegalanti/castle-harness)
provides agent tools and episode evaluation. That repository is currently
private; this release provides generation, task data access, and the execution
runtime independently.

![Recorded robot assembly demonstrations, expanding from one scene to a large grid.](docs/media/benchmark-scale.gif)

Recorded robot assembly demonstrations at increasing scale.

![Support graph and corresponding 18-part tower, colored by generation operation.](docs/media/support-graph-and-tower.png)

An illustration composed with the generation operators. Arrows run from each
supporting part to the part above it; colors identify the operations.

## Dataset

The [published dataset](https://huggingface.co/datasets/alexmzhang/castle-reasoning/tree/0bf9ccdd51861eac5112280c6bbb601fdc3ca98c)
contains **1,648 accepted robot demonstrations of 1,072 distinct target
geometries**. Both campaigns are supported: 570 `S` targets and 502 `U`
targets. Task IDs select structures; repeated demonstrations share a task ID.

| Campaign | Targets with demonstrations | Accepted demonstrations / attempts |
| --- | ---: | ---: |
| [October 10](https://huggingface.co/datasets/alexmzhang/castle-reasoning/blob/0bf9ccdd51861eac5112280c6bbb601fdc3ca98c/campaign-1k-20261010/index.json) | 570 | 1,146 / 1,186 (96.6%) |
| [October 11](https://huggingface.co/datasets/alexmzhang/castle-reasoning/blob/0bf9ccdd51861eac5112280c6bbb601fdc3ca98c/campaign-tasks-20261011/index.json) | 502 | 502 / 527 (95.3%) |
| **Total** | **1,072** | **1,648 / 1,713 (96.2%)** |

Part counts range from **7 to 37**, with median **12**, mean **14.37**, and
middle 50% from **9 to 18**. Counting each target once:

| Parts | Targets | Share |
| --- | ---: | ---: |
| 7–12 | 567 | 52.9% |
| 14–24 | 421 | 39.3% |
| 28–37 | 84 | 7.8% |

- 57 targets (5.3%) keep one foundation part fixed; 1,015 have none fixed.
- 503 targets have one demonstration, 562 have two, and seven have three.
- The demonstrations contain 19,613,756 control actions at 50 Hz, about
  109 hours of simulated execution. Median duration is 3 minutes 22 seconds;
  the range is 1 minute 28 seconds to 10 minutes 49 seconds.

These statistics are computed from the linked indices at the pinned dataset
revision. Collection began with 1,100 screened plans; 28 produced no accepted
demonstration. Acceptance measures the scripted collector's checks. Overall
raw-proposal generation yield cannot be recovered from the uploaded metadata.

## Quick start

Requirements: Linux x86_64 and [Pixi](https://pixi.sh/). Robot execution
requires an NVIDIA GPU; rendering requires EGL/OpenGL. Pixi installs Python
3.12 and the pinned dependencies.

```sh
git clone https://github.com/smbml/castle-reasoning-benchmark.git
cd castle-reasoning-benchmark
pixi install --locked
pixi run --locked check-runtime
pixi run --locked download-tasks --tasks S0074,U0008
```

The runtime check verifies source/build hashes, compiles a workcell, and takes
a native MuJoCo step. GPU episode execution is a separate check. The download
command retrieves one task from each campaign, including its plan, compiled
scene, and recorded states. It selects the first accepted demonstration in
each target's published index order; it downloads only the requested targets.

Task data is stored in `dataset/` by default. Set `CASTLE_ROOT` or pass
`--root` to `download-tasks` to use another directory. The downloader creates
the directory, verifies the pinned inputs, and reuses verified files.

## Using the benchmark

See [setup](docs/setup.md) for installation, task downloads, and harness
integration. [Tasks](docs/tasks.md) describes generation, artifact formats,
and success checks. Use the evaluation harness for scripted or language-model
episodes.

## Development

```sh
pixi run --locked test
pixi run --locked build
```

The default tests cover seeded generation, static stability, plan preparation,
numerical regression contracts, packaging, and CPU execution/worker wiring.
GPU execution checks require separate opt-in and a working NVIDIA driver. See
[tasks](docs/tasks.md#generate-geometry-and-plans) for a local generation smoke
command and the distinction between geometry, prepared plans, and robot
execution.

Python APIs live under `src/castle_reasoning/`: `runtime/` contains simulation,
planning, control, execution, rendering, and diagnostics; `data/` contains
generation, assembly, and verification workflows. `third_party/` contains the
patched MuJoCo-Warp distribution. `scripts/` contains its two build helpers.
Tests live in `tests/` and usage documentation in `docs/`.

## License

Castle Reasoning's first-party code and documentation are [MIT licensed](LICENSE).
MuJoCo-Warp and other dependencies retain their own licenses; see
[third-party notices](THIRD_PARTY_NOTICES.md) and [runtime builds](docs/runtime-build.md).
The MIT code license does not cover the separately hosted dataset; see its
Hugging Face card for dataset licensing.
