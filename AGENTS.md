# Repository Guidelines

## Project Structure & Module Organization

`README.md` is the public project entry point; `HANDOFF.md` records current implementation context. Maintainer-only audits, architecture notes, and plans under `docs/superpowers/` are local and ignored. New code belongs under `rm-sim/src/`: keep the Isaac-free rule engine in `rm_referee/`, lightweight geometry in `rm_world/`, the Isaac Lab adapter in `rm_isaac/`, and training code in `rm_train/`. Put tests under `rm-sim/tests/{rules,contracts,replay}/`. `RMUC-OfflineRL/` is a local, reference-only checkout—do not import, edit, or commit it.

## Build, Test, and Development Commands

Run these commands from `rm-sim/` after its scaffold exists:

```bash
python -m pip install -e ".[dev]"  # install the package and test tools
python -m compileall src           # catch syntax and package-layout errors
ruff check . && ruff format --check .
pytest -q                          # run all rule, contract, and replay tests
pytest -q tests/contracts          # verify Torch/Isaac rule parity
python scripts/benchmark.py --num-envs 1024 4096
```

The referee package must import and run on CPU without Isaac or Omniverse.

## Coding Style & Naming Conventions

Target Python 3.10+, use four-space indentation, Ruff formatting, and type hints for public APIs. Name modules and functions `snake_case`, classes `PascalCase`, and constants `UPPER_SNAKE_CASE`. Keep rule values in `constants.py` with manual page references; mark unresolved choices `[SIM]` or `[AMB]`. Prefer batched Torch operations over per-environment or per-unit Python loops.

## Testing Guidelines

Use pytest. Name files `test_<area>.py` and cases `test_<behavior>()`. Add exact `N_env=1` boundary tests for every rule branch, property tests for tensor invariants, and contract tests for backend parity. Event-injection replay must reproduce observable state exactly; stochastic replay uses documented aggregate tolerances because the 1 Hz data lacks target attribution. Every bug fix needs a regression test.

## Commit & Pull Request Guidelines

Use imperative, sentence-case subjects such as `Fix action-space wording`, and keep commits single-purpose. Core maintainers may push scoped, validated work directly to `main` during bootstrap. When a branch is useful, name it `feature/<topic>`, `fix/<topic>`, or `docs/<topic>`—never `agent/...`. Reserve pull requests for external contributions or changes that need review; include rule citations, checks run, replay metrics for referee changes, and screenshots only for visual work.

## Security & Local Configuration

Never commit credentials, official PDFs, raw competition data, extracted databases, model checkpoints, generated runs, or machine-specific Isaac links. Configure local paths through environment variables without embedding secrets.
