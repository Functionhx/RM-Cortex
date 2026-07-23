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
> RM-Cortex is in **pre-alpha**. The RoboMaster 2026 V1.5.0 rules audit and Phase 1 architecture are complete. Implementation is now underway.

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
pytest -q
python scripts/benchmark.py --num-envs 1024 4096
```

The current vertical slice includes vectorized combat, armor refractory windows, heat, power, ammunition purchases, read-bar respawn, radar recurrence, building outcomes, lightweight kinematics, and the Isaac geometry normalization boundary.

## Design principles

- **Traceable rules.** Constants and transitions cite the governing manual page; simulator assumptions are marked `[SIM]`.
- **Backend parity.** Fixed actions, geometry events, and random samples must produce identical referee state on both backends.
- **Vectorized first.** Rules and lightweight kinematics use batched Torch tensors for large-scale training.
- **Honest validation.** Exact rule tests are separated from statistical calibration where the 1 Hz dataset lacks shot-to-target attribution.
- **A path to reality.** Phase 1 starts with geometry and kinematics while preserving an Isaac Lab and sim2real upgrade boundary.

## Status and roadmap

- [x] Audit the RoboMaster 2026 V1.5.0 rules text, tables, and key engineering figures
- [x] Correct the referee design and define the pure Torch / Isaac Lab contract
- [ ] Implement the vectorized pure-Torch referee
- [ ] Build the lightweight Torch world and scripted baseline
- [ ] Add the Isaac Lab `DirectMARLEnv` adapter and parity tests
- [ ] Train and evaluate the first PPO/MAPPO baseline
- [ ] Calibrate hit and observation models from competition replays

## Repository map

| Path | Purpose |
|---|---|
| [`HANDOFF.md`](HANDOFF.md) | Current state and next implementation entry point |
| [`docs/superpowers/specs/`](docs/superpowers/specs/) | Rules audit, digest, architecture, and Phase 1 plan |
| `rm-sim/` | Referee, simulation, Isaac, and training packages |
| [`AGENTS.md`](AGENTS.md) | Contributor workflow and repository conventions |

Start with the [PDF audit](docs/superpowers/specs/2026-07-23-rmuc-v1.5.0-pdf-audit.md), then read the [verified rules digest](docs/superpowers/specs/2026-07-23-rmuc-rules-digest.md) and [implementation plan](docs/superpowers/specs/2026-07-23-rmuc-implementation-plan.md).

Official manuals, competition datasets, Isaac runtimes, training outputs, and checkpoints are intentionally not distributed here. Contributors must obtain authorized source material separately and configure local paths without committing it.

## Contributing

Read [`AGENTS.md`](AGENTS.md) before contributing. Referee changes must cite the governing rule source and include regression tests. Architecture discussions must clearly distinguish official behavior from modeling choices.

## License and notice

RM-Cortex is released under the [MIT License](LICENSE). It is an independent project and is not affiliated with or endorsed by DJI or RoboMaster.
