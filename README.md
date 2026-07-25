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
> RM-Cortex is in **pre-alpha**. The first end-to-end Phase 1 implementation is available and native Isaac Lab headless scene validation is passing; full-scale baseline training is still pending.

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

The implementation now includes the vectorized referee, a 2.5D Torch arena with LOS and armor geometry, deterministic scripted controllers, a hierarchical behavior-tree baseline, an import-safe Isaac Lab `DirectMARLEnv`, and a parameter-shared MAPPO baseline with belief-state entity attention, centralized critic, action masks, checkpoints, and evaluation.

Run `python scripts/benchmark_world.py --num-envs 1024 4096` for Torch-world throughput. Inside an Isaac Lab launcher environment, run `PYTHONPATH=src /path/to/IsaacLab/isaaclab.sh -p scripts/isaac_smoke.py --headless` to validate the optional scene backend.

## Observation model and radar

Phase 1 training defaults to a team-shared **belief observation** over all 16 unit slots. The actor receives each track's estimated position and velocity, last observed decision state, separate position/state ages, bounded positive-semidefinite covariance, and source label. Local or shared sightings refresh the complete track; radar reports update position and velocity without pretending that HP or ammunition is fresh, and an unseen enemy otherwise follows the constant-velocity prediction instead of exposing current truth. Set `"observation_mode": "oracle"` in an experiment config to retain the full-information upper-bound baseline; evaluation automatically uses the mode stored in its checkpoint.

The radar system is distinct from the battlefield outpost. Following manual section 5.6.6, accurate or partially accurate reports build marking progress; confirmed enemy positions appear at `P >= 100`, with 15%/20% ground vulnerability at `P >= 100/120`. Reports are absolute field coordinates sent once per 5 Hz policy step, and the policy may explicitly withhold a report. The implementation also covers double vulnerability, interference suppression, and aerial laser countermeasures. The MAPPO action layout currently uses the outpost policy slot only as the carrier for team-level radar commands; the referee actions themselves remain team-scoped.

## Hierarchical behavior-tree baseline

Low-level ground navigation is fixed rather than learned: the pure-Torch world executes cached, deterministic A* paths on a `0.05 m` grid derived from static arena geometry. Above it, one global tree publishes team intent and eight independent role trees—hero, engineer, two infantry, aerial, sentry, base, and outpost—select high-level missions and auditable leaves. Only the five mobile ground roles request A* paths; aerial and structure roles retain their dedicated execution. The deterministic executor handles combat, local separation, economy, radar, and role-specific rule actions.

Run the paired-side evaluator:

```bash
cd rm-sim
python scripts/evaluate_behavior_tree.py \
  --opponent tactical \
  --seeds 1007 \
  --device cpu
```

Each seed produces two independent legs, with the behavior-tree controller assigned RED and then BLUE. A leg ends at match termination or the official 420-second limit; `--max-policy-steps` is only an explicit smoke-test truncation, and unfinished legs are reported separately rather than counted as draws. Both controllers read complete `GameState`/`KinematicState` and can emit the full `WorldActions` schema, including actions absent from MAPPO. This is therefore an **oracle/full-action engineering baseline**, not a fair belief-policy comparison; the implementation pilot below is not a final win-rate claim.

An implementation pilot on seeds `1007`, `2007`, and `3007` completed all six 420-second paired legs with `227/227` successful A* plans. Against the tactical controller it scored `3–3` (`balanced_score=0.5`), with mean accumulated team return `6.539` versus `10.851`. This verifies execution, not tactical superiority; the sample is too small and seed-sensitive for a final win-rate claim.

## Experimental RMUC data path

The current CLI retains the first authorized 1 Hz RMUC-data experiment as a reproducible diagnostic:

```bash
cd rm-sim
python scripts/train_imitation.py \
  --database /path/to/rmuc_region_dataset.sqlite \
  --output-dir runs/rmuc_bc \
  --fire-coefficient 0.5
python scripts/train.py \
  --pretrained-actor runs/rmuc_bc/best.pt \
  --output-dir runs/phase1_bc_mappo
```

The adapter opens SQLite in read-only mode, rejects invalid tracks instead of clipping or forward-filling them, and assigns complete team/time components before creating windows. In the current three-region export this naturally uses the southern region for training, eastern for validation, and northern for the locked test split. Offline inputs reproduce the online 34D actor and 16×51D belief-entity schemas: enemy truth passes through synthetic range/LOS sensing and a causal last-seen constant-velocity belief.

The diagnostic labels are a capped five-second displacement and future-window firing occurrence, not a 0.2-second action, target choice, or behavior-tree decision. Transfer copies the shared encoder/body and mobile-role rows; MAPPO action-head and critic-MLP parameters keep their initialization, although transferred role/team embeddings also change the critic's input representation. The ten-second offline history also differs from the online tracker maintained from match start.

The first 8,192/2,048-window pilot used firing weight `0.5`. Validation displacement RMSE beat the zero-motion baseline (`2.219 m` vs `2.427 m`), but MAE regressed (`1.458 m` vs `1.368 m`); firing F1 was `0.578`. With the same seed-7, 64-environment, 50-update MAPPO budget, deterministic RED-only evaluation at seed 1007 gave BC initialization `0–64`, return `-3.672`, versus scratch `62–2`, `+23.546`, against scripted BLUE. This single-seed, single-side negative transfer does not show that human data is harmful; it shows that this representation-transfer objective is not an accepted initializer.

Future RMUC-data work targets the high-level layer: team tasks, role missions, reliably derivable targets, and behavior-tree leaf/option selection. Fixed A* remains the low-level navigator. The displacement/fire experiment stays available only as a negative ablation; navigation-only PPO/BC is not the roadmap. The default imitation configuration disables the firing proxy.

## Visualization

Export a dependency-light 2D match animation:

```bash
cd rm-sim
python -m pip install -e ".[visualization]"
python scripts/visualize_torch.py
```

The exporter requires `ffmpeg` with H.264 (`libx264`) support on `PATH`. The ignored output `outputs/torch_demo.mp4` simulates the complete official 420-second match and samples one frame per simulated second into an approximately 21-second H.264 replay. It uses a deterministic tactical baseline—not a trained checkpoint—with separate hero deployment, engineer support, infantry lanes, sentry defense, objective pressure, weak-state recovery, official ammunition exchange, and banked aerial-support sorties. The aerial launches only while support is requested and free time remains; it cannot fire on the pad, while support is paused, or while radar-locked. Its motion is constrained to the manual section 4.5 pad/road airspace and safety-tether envelope. Sortie timing is a tactical policy, not an official fixed window. The replay also shows headings, trails, health, fire lines, resources, and elevation-layered polygonal terrain. A neutral sequential palette encodes height above the locally crowned field; contour bands, slope arrows, edge shadows, and a height legend distinguish flat decks from inclines, while red/blue is reserved for team boundaries. Base, outpost, and fortress centers follow figure 4-5; the 150mm fortress and its six 20° faces, the central 200–350mm/10.5° connectors, the trapezoid 200–400mm/23°/43° surfaces, the 17° fly ramp, and the 70mm/240mm bumps follow figures 4-25–4-37. The exporter rejects any ground-footprint overlap. Use `--steps 300` only for a short smoke export.

```bash
PYTHONPATH=src /path/to/IsaacLab/isaaclab.sh \
  -p scripts/visualize_isaac.py --num-envs 1
```

Add `--device cpu` when CUDA simulation is unavailable or GPU memory is occupied. The Isaac scene runs the same tactical controller, polygonal terrain footprints, per-vertex slope elevations, and authoritative Torch game state; non-rectangular surfaces are deterministically triangulated into meshes rather than replaced by bounding boxes. The rough road is instantiated as 70mm ridges at 240mm pitch. It does not introduce a second gameplay state. Terrain placement beyond explicitly dimensioned modules remains a Phase 1 `[SIM]` approximation, not official CAD.

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
- [x] Replace out-of-range actor truth with age- and covariance-aware entity beliefs
- [x] Validate the scene in a native Isaac Lab headless runtime
- [x] Add the global-plus-eight-role behavior-tree baseline and 0.05 m A* navigation
- [x] Add a read-only RMUC adapter and reproduce the first negative transfer diagnostic
- [ ] Learn high-level task, target, and behavior-tree leaf choices from authorized match data
- [ ] Publish full-scale baseline training metrics
- [ ] Calibrate hit and observation models from competition replays

## Repository map

| Path | Purpose |
|---|---|
| [`HANDOFF.md`](HANDOFF.md) | Current state and next implementation entry point |
| `rm-sim/src/` | Referee, Torch world, Isaac adapter, and MAPPO packages |
| `rm-sim/configs/` | Reproducible experiment configurations |
| `rm-sim/scripts/` | Training, evaluation, visualization, Isaac smoke, and benchmark entry points |
| [`AGENTS.md`](AGENTS.md) | Contributor workflow and repository conventions |

Official manuals, competition datasets, Isaac runtimes, training outputs, and checkpoints are intentionally not distributed here. Contributors must obtain authorized source material separately and configure local paths without committing it.

## Contributing

Read [`AGENTS.md`](AGENTS.md) before contributing. Referee changes must cite the governing rule source and include regression tests. Architecture discussions must clearly distinguish official behavior from modeling choices.

## License and notice

RM-Cortex is released under the [MIT License](LICENSE). It is an independent project and is not affiliated with or endorsed by DJI or RoboMaster.
