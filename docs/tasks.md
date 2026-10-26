# Tasks

## Assembly and success

A task specifies cuboid parts and a target structure. The agent selects the
parts, their destinations, and placement order. The runtime plans grasps and
motions, simulates execution, and checks the resulting assembly.

Success checks include target poses within 5 mm and 5 degrees, declared
support contacts, settling, and arm/gripper conditions. Counting parts near
their targets is not sufficient to establish full success. The evaluator
applies the complete runtime predicate.

## Existing benchmark inputs

The [published task data](https://huggingface.co/datasets/alexmzhang/castle-reasoning)
is separate from the code. Its two campaign indices cover 570 `S` targets
and 502 `U` targets, for 1,072 distinct structures and 1,648 accepted
demonstrations. Downloading selects the first accepted indexed attempt for
each requested target and preserves its campaign directory. Repeated
demonstrations are alternative recordings of the same target, not additional
task IDs. See the [dataset statistics](../README.md#dataset).

Each accepted attempt contains:

| File | Purpose |
| --- | --- |
| `plan.json` | Parts, target geometry, support relations, and preparation metadata |
| `model.mjb` | Compiled MuJoCo scene |
| `replay.npz` | Recorded states, including the scene reset and image-target states |
| `execution_config.json` | Execution settings, when provided |
| `execution_order.json` | Recorded order, when provided |

The recorded initial state resets a task; the final state supplies target
views. Agent actions are planned and simulated as fresh executions.
[Download instructions](setup.md#download-existing-tasks) use these existing
inputs directly.

## Generate geometry and plans

The generator identifies its construction rules as
`independent-load-construction-v1`. It records seeds, candidate IDs,
operation traces, geometry fingerprints, rejection reasons, and support
graphs. The Python API returns accepted targets and the outcome of every
proposal:

```python
from castle_reasoning.data.generation import generator

pool = generator.generate_pool(seed=0, count=1)
print(len(pool["targets"]), len(pool["attempts"]))
```

To generate a pool and prepare robot plans, use an output directory that does
not already exist:

```sh
pixi run --locked python -m castle_reasoning.data.generation.generate --help
pixi run --locked python -m castle_reasoning.data.generation.generate \
  --output /tmp/castle-example --seed 0 --proposals 10 --workers 2
```

This writes `pool.json`, prepared files under `plans/`, and
`preparation.json` with per-target outcomes. The command tries ten proposals;
rejections can leave fewer than ten accepted targets. Check each record's
`status` for plan-preparation failures. Acceptance of geometry does not imply
that a robot plan or demonstration succeeds. Preparation, physical screening,
robot demonstrations, and agent evaluation are separate stages.

New proposals do not reproduce the historical dataset by themselves. That
requires its original seeds, exclusions, screening, and collection settings.
Evaluate against the downloaded benchmark inputs when comparing benchmark
results.

## Source references

Geometry and generation live in
[`data/generation/`](../src/castle_reasoning/data/generation/). Scene building,
planning, control, and success checks live in
[`runtime/`](../src/castle_reasoning/runtime/). See
[runtime builds](runtime-build.md) for components and reproducible builds.
