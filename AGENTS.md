# Repository guidance

## Commands

- Install: `pixi install --locked`.
- Test: `pixi run --locked test`.
- Focused test: `pixi run --locked python -m pytest -q tests/package`.
- Build: `pixi run --locked build`.
- Runtime build: `pixi run --locked python -m build third_party/mujoco-warp`.
- Native/build check: `pixi run --locked check-runtime`.
- Lint: `pixi run --locked ruff check src/castle_reasoning tests/package scripts/package_active_runtime.py`.
- Check patch formatting: `git diff --check`.

## Project conventions

- Use isolated branches for changes; keep commits scoped to the task.
- The installed namespace is `castle_reasoning` under `src/`. Runtime modules
  must not import data-production workflows. Data workflows compose runtime.
  Only the patched MuJoCo-Warp distribution belongs in `third_party/`.
- Use normal installable distributions and package resources. Do not modify
  PYTHONPATH, sys.path, or install custom import-path hooks.
- CASTLE_ROOT means downloaded task data; its default is the invoking
  directory's ignored `dataset/`, created on download.
- Keep dataset and dependency revisions explicit. Use small local samples for
  download checks; do not rerun collection campaigns as packaging checks.
- Preserve numerical/control behavior and the regression contracts in
  `tests/fixtures/runtime-contracts.json`. Update installed source hashes when
  their files change; retain upstream licenses and modification notices.

## Documentation

- Write README and usage docs for readers without prior project context.
- Keep package behavior, dataset semantics and build instructions synchronized
  with the code. Distinguish CPU checks, GPU checks and benchmark evaluation.

## References

- [Setup](docs/setup.md)
- [Tasks](docs/tasks.md)
- [Runtime build](docs/runtime-build.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)
