<p align="center">
  <img src="./rmcortex-logo.png" alt="RM-Cortex logo" width="360">
</p>

<h1 align="center">RM-Cortex</h1>

<p align="center">
  A rule-grounded multi-agent research stack for RoboMaster.<br>
  面向 RoboMaster 的规则驱动多智能体研究栈。
</p>

<p align="center">
  <a href="#english">English</a> · <a href="#简体中文">简体中文</a>
</p>

---

<a id="english"></a>

## English

> [!IMPORTANT]
> RM-Cortex is in **pre-alpha**. The RoboMaster 2026 V1.5.0 rules audit and Phase 1 architecture are complete; the simulator and training packages have not been released yet.

RM-Cortex is an independent research project for building verifiable RoboMaster decision systems. It connects an official-rule-derived referee, a batched kinematic world, Isaac Lab integration, heterogeneous multi-agent policies, and replay-based calibration in one reproducible stack.

### Architecture

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

The Torch and Isaac Lab backends never maintain separate game rules. They produce the same normalized geometry events and call the same vectorized `rm_referee` state machine.

### Design principles

- **Traceable rules.** Constants and state transitions cite the governing manual page; simulator assumptions are explicitly marked `[SIM]`.
- **Backend parity.** Fixed actions, geometry events, and random samples must produce identical referee state on both backends.
- **Vectorized first.** Rules and lightweight kinematics use batched Torch tensors for large-scale training.
- **Honest validation.** Exact rule tests are separated from statistical calibration where the 1 Hz dataset lacks shot-to-target attribution.
- **A path to reality.** Phase 1 starts with geometry and kinematics, while preserving an Isaac Lab and sim2real upgrade boundary.

### Status and roadmap

- [x] Audit the RoboMaster 2026 V1.5.0 rules text, tables, and key engineering figures
- [x] Correct the referee design and define the pure Torch / Isaac Lab contract
- [ ] Implement the vectorized pure-Torch referee
- [ ] Build the lightweight Torch world and scripted baseline
- [ ] Add the Isaac Lab `DirectMARLEnv` adapter and parity tests
- [ ] Train and evaluate the first PPO/MAPPO baseline
- [ ] Calibrate hit and observation models from competition replays

### Repository map

| Path | Purpose |
|---|---|
| [`HANDOFF.md`](HANDOFF.md) | Current state and next implementation entry point |
| [`docs/superpowers/specs/`](docs/superpowers/specs/) | Rules audit, digest, architecture, and Phase 1 plan |
| `rm-sim/` | Planned referee, simulation, Isaac, and training packages |
| [`AGENTS.md`](AGENTS.md) | Contributor workflow and repository conventions |

Start by reading the [PDF audit](docs/superpowers/specs/2026-07-23-rmuc-v1.5.0-pdf-audit.md), then the [verified rules digest](docs/superpowers/specs/2026-07-23-rmuc-rules-digest.md) and [implementation plan](docs/superpowers/specs/2026-07-23-rmuc-implementation-plan.md).

Official manuals, competition datasets, Isaac runtimes, training outputs, and checkpoints are intentionally not distributed in this repository. Contributors must obtain authorized source material separately and configure local paths without committing it.

### Contributing

Contributions are welcome once scoped against the current Phase 1 plan. Read [`AGENTS.md`](AGENTS.md), cite the relevant rule source in referee changes, and include tests for every changed rule branch. Architecture discussions should clearly distinguish official behavior from modeling choices.

### License and notice

A project license has not been selected yet; do not assume reuse permission until a `LICENSE` file is added. RM-Cortex is independent and is not affiliated with or endorsed by DJI or RoboMaster.

---

<a id="简体中文"></a>

## 简体中文

> [!IMPORTANT]
> RM-Cortex 目前处于 **pre-alpha** 阶段。RoboMaster 2026 V1.5.0 规则审计和 Phase 1 架构已经完成，仿真与训练代码尚未发布。

RM-Cortex 是一个面向 RoboMaster 自主决策的独立研究项目。它把基于官方规则的裁判系统、批量运动学世界、Isaac Lab 集成、异构多智能体策略和回放标定连接成一套可复现的完整研究栈。

### 核心架构

```text
策略 / 脚本对手
       │ 异构动作
       ▼
Torch World / Isaac Lab ── 几何、视线、接触
       │
       ▼
rm_referee（纯 Torch，唯一规则真值）
       │
       ▼
观测 · 事件 · 奖励
```

Torch 与 Isaac Lab 后端不各自维护规则。两者只负责产生统一的几何事件，并调用同一个向量化 `rm_referee` 状态机。

### 设计原则

- **规则可追溯：** 常量和状态迁移引用手册页码，仿真假设统一标记为 `[SIM]`。
- **后端一致：** 固定动作、几何事件和随机样本必须在两个后端产生相同裁判状态。
- **向量化优先：** 裁判与轻量运动学使用批量 Torch 张量，服务大规模训练。
- **诚实验证：** 官方规则边界做精确测试；1Hz 数据缺少逐发目标归因的部分只做统计标定。
- **保留上车路径：** Phase 1 从几何与运动学开始，同时保留 Isaac Lab 和 sim2real 的升级边界。

### 当前进度

- [x] 审计 RoboMaster 2026 V1.5.0 正文、表格和关键工程图
- [x] 纠正裁判设计，冻结纯 Torch / Isaac Lab 契约
- [ ] 实现向量化纯 Torch 裁判
- [ ] 实现轻量 Torch World 与脚本基线
- [ ] 接入 Isaac Lab `DirectMARLEnv` 并完成一致性测试
- [ ] 训练和评估首个 PPO/MAPPO 基线
- [ ] 使用比赛回放标定命中与观测模型

### 文档入口

| 文档 | 内容 |
|---|---|
| [`HANDOFF.md`](HANDOFF.md) | 当前状态与下一步入口 |
| [PDF 审计](docs/superpowers/specs/2026-07-23-rmuc-v1.5.0-pdf-audit.md) | 已纠正错误、新信息、图纸结果和歧义 |
| [规则 Digest](docs/superpowers/specs/2026-07-23-rmuc-rules-digest.md) | 带来源标签的实现规则索引 |
| [系统设计](docs/superpowers/specs/2026-07-23-rmuc-referee-system-design.md) | 纯 Torch + Isaac Lab 架构 |
| [实现计划](docs/superpowers/specs/2026-07-23-rmuc-implementation-plan.md) | Phase 1 里程碑、任务与验收标准 |
| [`AGENTS.md`](AGENTS.md) | 贡献流程与仓库约定 |

官方手册、比赛数据、Isaac 运行时、训练产物和 checkpoint 不随仓库分发。贡献者需要自行取得授权材料，并通过本地配置接入，不能将其提交到仓库。

### 参与贡献

提交工作前请先阅读 [`AGENTS.md`](AGENTS.md)，并以当前 Phase 1 计划为边界。任何裁判规则修改都应引用对应手册位置并补充回归测试；架构讨论需要明确区分官方规则与仿真假设。

### 许可与声明

项目尚未选定开源许可证；在加入 `LICENSE` 前，请勿默认取得复用授权。RM-Cortex 为独立项目，与 DJI 或 RoboMaster 不存在隶属或官方背书关系。
