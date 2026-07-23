<p align="center">
  <img src="./rmcortex-logo.png" alt="RM-Cortex Logo" width="360">
</p>

<h1 align="center">RM-Cortex</h1>

<p align="center">
  面向 RoboMaster 的规则驱动多智能体研究栈。
</p>

<p align="center">
  <a href="./README.md">English</a> · <a href="./README.zh-CN.md">简体中文</a>
</p>

<p align="center">
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-MIT-1683ff.svg" alt="MIT License"></a>
</p>

> [!IMPORTANT]
> RM-Cortex 目前处于 **pre-alpha** 阶段。RoboMaster 2026 V1.5.0 规则审计和 Phase 1 架构已经完成，代码实现正在进行。

RM-Cortex 是一个面向 RoboMaster 自主决策的独立研究项目。它把基于官方规则的裁判系统、批量运动学世界、Isaac Lab 集成、异构多智能体策略和回放标定连接成一套可复现的完整研究栈。

## 核心架构

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

## 快速开始

```bash
cd rm-sim
python -m pip install -e ".[dev]"
pytest -q
python scripts/benchmark.py --num-envs 1024 4096
```

当前纵切已经实现向量化战斗、装甲检测窗口、热量、功率、弹量购买、读条复活、雷达递推、建筑胜负、轻量运动学，以及 Isaac 几何事实归一化边界。

## 设计原则

- **规则可追溯：** 常量和状态迁移引用手册页码，仿真假设统一标记为 `[SIM]`。
- **后端一致：** 固定动作、几何事件和随机样本必须在两个后端产生相同裁判状态。
- **向量化优先：** 裁判与轻量运动学使用批量 Torch 张量，服务大规模训练。
- **诚实验证：** 官方规则边界做精确测试；1Hz 数据缺少逐发目标归因的部分只做统计标定。
- **保留上车路径：** Phase 1 从几何与运动学开始，同时保留 Isaac Lab 和 sim2real 的升级边界。

## 当前进度

- [x] 审计 RoboMaster 2026 V1.5.0 正文、表格和关键工程图
- [x] 纠正裁判设计，冻结纯 Torch / Isaac Lab 契约
- [ ] 实现向量化纯 Torch 裁判
- [ ] 实现轻量 Torch World 与脚本基线
- [ ] 接入 Isaac Lab `DirectMARLEnv` 并完成一致性测试
- [ ] 训练和评估首个 PPO/MAPPO 基线
- [ ] 使用比赛回放标定命中与观测模型

## 仓库地图

| 路径 | 内容 |
|---|---|
| [`HANDOFF.md`](HANDOFF.md) | 当前状态与下一步入口 |
| `rm-sim/` | 裁判、仿真、Isaac 与训练包 |
| [`AGENTS.md`](AGENTS.md) | 贡献流程与仓库约定 |

官方手册、比赛数据、Isaac 运行时、训练产物和 checkpoint 不随仓库分发。贡献者需要自行取得授权材料，并通过本地配置接入，不能将其提交到仓库。

## 参与贡献

提交工作前请阅读 [`AGENTS.md`](AGENTS.md)。任何裁判规则修改都应引用对应手册位置并补充回归测试；架构讨论需要明确区分官方规则与仿真假设。

## 许可与声明

RM-Cortex 采用 [MIT License](LICENSE) 开源。项目为独立研究项目，与 DJI 或 RoboMaster 不存在隶属或官方背书关系。
