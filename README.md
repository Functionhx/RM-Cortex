<p align="center">
  <img src="./rmcortex-logo.png" alt="RM-Cortex logo" width="360">
</p>

<h1 align="center">RM-Cortex</h1>

<p align="center">
  A rule-grounded multi-agent research stack for RoboMaster.
</p>

<p align="center">
  <a href="./README.md">English</a> · <a href="./README.zh-CN.md">简体中文</a>
</p>

<p align="center">
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-MIT-1683ff.svg" alt="MIT License"></a>
</p>

> [!IMPORTANT]
> RM-Cortex is in **pre-alpha**. The first end-to-end Phase 1 implementation is available; native Isaac Lab validation and full-scale baseline training are still pending.

RM-Cortex is an independent research project for building verifiable RoboMaster decision systems. It connects an official-rule-derived referee, a batched kinematic world, Isaac Lab integration, heterogeneous multi-agent policies, and replay-based calibration in one reproducible stack.

## Architecture

```text
Policies / Scripted Opponents
              │
     heterogeneous actions
              ▼
┌──────────────────────────────┐
│ Torch World │ Isaac Lab      │  geometry, LOS, contacts
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ rm_referee — pure Torch      │  single rule source of truth
└──────────────┬───────────────┘
               ▼
   observations · events · rewards
```

The Torch and Isaac Lab backends never maintain separate game rules. They produce normalized geometry events and call the same vectorized `rm_referee` state machine.

## Quick start

```bash
cd rm-sim
python -m pip install -e ".[dev]"
ruff check . && ruff format --check .
mypy src
pytest -q
python scripts/train.py --updates 1 --num-envs 8
```

The implementation now includes the vectorized referee, a 2.5D Torch arena with LOS and armor geometry, deterministic scripted play, an import-safe Isaac Lab `DirectMARLEnv`, and a parameter-shared MAPPO baseline with centralized critic, action masks, checkpoints, and evaluation.

Run `python scripts/benchmark_world.py --num-envs 1024 4096` for Torch-world throughput. Inside an Isaac Lab launcher environment, run `python scripts/isaac_smoke.py --headless` to validate the optional scene backend.

## Design principles

- **Traceable rules.** Constants and transitions cite the governing manual page; simulator assumptions are marked `[SIM]`.
- **Backend parity.** Fixed actions, geometry events, and random samples must produce identical referee state on both backends.
- **Vectorized first.** Rules and lightweight kinematics use batched Torch tensors for large-scale training.
- **Honest validation.** Exact rule tests are separated from statistical calibration where the 1 Hz dataset lacks shot-to-target attribution.
- **A path to reality.** Phase 1 starts with geometry and kinematics while preserving an Isaac Lab and sim2real upgrade boundary.

## Status and roadmap

- [x] Audit the RoboMaster 2026 V1.5.0 rules text, tables, and key engineering figures
- [x] Correct the referee design and define the pure Torch / Isaac Lab contract
- [x] Implement the vectorized pure-Torch referee
- [x] Build the lightweight Torch world and scripted baseline
- [x] Add the Isaac Lab `DirectMARLEnv` adapter and rule-parity tests
- [x] Implement reproducible PPO/MAPPO training and evaluation
- [ ] Validate the scene in a native Isaac Lab runtime and publish full training metrics
- [ ] Calibrate hit and observation models from competition replays

## Repository map

| Path | Purpose |
|---|---|
| [`HANDOFF.md`](HANDOFF.md) | Current state and next implementation entry point |
| `rm-sim/src/` | Referee, Torch world, Isaac adapter, and MAPPO packages |
| `rm-sim/configs/` | Reproducible experiment configurations |
| `rm-sim/scripts/` | Training, evaluation, Isaac smoke, and benchmark entry points |
| [`AGENTS.md`](AGENTS.md) | Contributor workflow and repository conventions |

Official manuals, competition datasets, Isaac runtimes, training outputs, and checkpoints are intentionally not distributed here. Contributors must obtain authorized source material separately and configure local paths without committing it.

## Contributing

Read [`AGENTS.md`](AGENTS.md) before contributing. Referee changes must cite the governing rule source and include regression tests. Architecture discussions must clearly distinguish official behavior from modeling choices.

## License and notice

RM-Cortex is released under the [MIT License](LICENSE). It is an independent project and is not affiliated with or endorsed by DJI or RoboMaster.
