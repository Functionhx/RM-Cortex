# Repository Guidelines

## Project Structure & Module Organization

`README.md` is the public project entry point; `HANDOFF.md` records current implementation context. Rule provenance, the PDF audit, architecture, and execution plan live in `docs/superpowers/specs/`. New code belongs under `rm-sim/src/`: keep the Isaac-free rule engine in `rm_referee/`, lightweight geometry in `rm_world/`, the Isaac Lab adapter in `rm_isaac/`, and training code in `rm_train/`. Put tests under `rm-sim/tests/{rules,contracts,replay}/`. `RMUC-OfflineRL/` is a local, reference-only checkout—do not import, edit, or commit it.

## Build, Test, and Development Commands

Run these commands from `rm-sim/` after its scaffold exists:

```bash
python -m pip install -e ".[dev]"  # install the package and test tools
python -m compileall src           # catch syntax and package-layout errors
pytest -q                          # run all rule, contract, and replay tests
pytest -q tests/contracts          # verify Torch/Isaac rule parity
python scripts/benchmark.py --num-envs 1024 4096
```

The referee package must import and run on CPU without Isaac or Omniverse.

## Coding Style & Naming Conventions

Target Python 3.10+, use four-space indentation, PEP 8 layout, and type hints for public APIs. Name modules and functions `snake_case`, classes `PascalCase`, and constants `UPPER_SNAKE_CASE`. Keep rule values in `constants.py` with manual page references; mark unresolved choices `[SIM]` or `[AMB]`. Prefer batched Torch operations over per-environment or per-unit Python loops. No formatter or linter is configured yet, so preserve ordered imports and nearby style.

## Testing Guidelines

Use pytest. Name files `test_<area>.py` and cases `test_<behavior>()`. Add exact `N_env=1` boundary tests for every rule branch, property tests for tensor invariants, and contract tests for backend parity. Event-injection replay must reproduce observable state exactly; stochastic replay uses documented aggregate tolerances because the 1 Hz data lacks target attribution. Every bug fix needs a regression test.

## Commit & Pull Request Guidelines

The reference history uses imperative, sentence-case subjects such as `Fix action-space wording`. Keep commits single-purpose. Pull requests should describe behavior changes, cite the governing manual/digest section, link issues, and list commands run. Include replay metrics for referee changes and screenshots only for visual or Isaac-facing work.

## Security & Local Configuration

Never commit credentials, official PDFs, raw competition data, extracted databases, model checkpoints, generated runs, or machine-specific Isaac links. Configure local paths through environment variables without embedding secrets.
