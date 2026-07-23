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
> RM-Cortex 目前处于 **pre-alpha** 阶段。首个端到端 Phase 1 实现已经落地，原生 Isaac Lab 无头场景验证已通过；完整规模基线训练仍待完成。

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
ruff check . && ruff format --check .
mypy src
pytest -q
python scripts/train.py --updates 1 --num-envs 8
```

当前实现已经包含向量化裁判、带 LOS 与装甲几何的 2.5D Torch 场地、目标优先与分角色战术两类脚本基线、可安全选装的 Isaac Lab `DirectMARLEnv`，以及带集中式 critic、动作掩码、checkpoint 和评估流程的参数共享 MAPPO 基线。

使用 `python scripts/benchmark_world.py --num-envs 1024 4096` 测量 Torch World 吞吐；在 Isaac Lab 启动环境中使用 `PYTHONPATH=src /path/to/IsaacLab/isaaclab.sh -p scripts/isaac_smoke.py --headless` 验证可选场景后端。

## 可视化

导出无需 Isaac Lab 的轻量 2D 对局动画：

```bash
cd rm-sim
python -m pip install -e ".[visualization]"
python scripts/visualize_torch.py
```

生成但不纳入版本控制的 `outputs/torch_demo.gif` 会完整仿真官方 420 秒对局，并按每个仿真秒采样一帧，压缩为约 21 秒回放。演示使用的是确定性战术脚本而非训练好的 checkpoint，包含英雄部署、工程支援、步兵分路、哨兵防守、目标集火、虚弱撤回、官方购弹流程和按库存分波次出动的空中支援；同时显示朝向、轨迹、血条、射击线、资源与图纸地形。无人机仅在请求支援且免费库存可用时起飞，停机坪上、暂停支援或被雷达锁定时不能开火；具体出动波次属于战术策略，并非官方固定窗口。前哨站采用手册图 4-5 标注的对角线中心，导出器发现任何地面外廓重叠都会直接失败。仅需短时冒烟时可追加 `--steps 300`。

```bash
PYTHONPATH=src /path/to/IsaacLab/isaaclab.sh \
  -p scripts/visualize_isaac.py --num-envs 1
```

CUDA 仿真不可用或显存被占用时可追加 `--device cpu`。Isaac 场景运行同一战术控制器、同一组地形 primitive 与权威 Torch 比赛状态，不会引入第二套规则。除明确标注尺寸的模块外，其余地形位置仍是 Phase 1 `[SIM]` 近似，不是官方 CAD。

## 设计原则

- **规则可追溯：** 常量和状态迁移引用手册页码，仿真假设统一标记为 `[SIM]`。
- **后端一致：** 固定动作、几何事件和随机样本必须在两个后端产生相同裁判状态。
- **向量化优先：** 裁判与轻量运动学使用批量 Torch 张量，服务大规模训练。
- **诚实验证：** 官方规则边界做精确测试；1Hz 数据缺少逐发目标归因的部分只做统计标定。
- **保留上车路径：** Phase 1 从几何与运动学开始，同时保留 Isaac Lab 和 sim2real 的升级边界。

## 当前进度

- [x] 审计 RoboMaster 2026 V1.5.0 正文、表格和关键工程图
- [x] 纠正裁判设计，冻结纯 Torch / Isaac Lab 契约
- [x] 实现向量化纯 Torch 裁判
- [x] 实现轻量 Torch World 与脚本基线
- [x] 接入 Isaac Lab `DirectMARLEnv` 并完成规则一致性测试
- [x] 实现可复现的 PPO/MAPPO 训练与评估链路
- [x] 在原生 Isaac Lab 无头运行时中完成场景验证
- [ ] 发布完整规模基线训练指标
- [ ] 使用比赛回放标定命中与观测模型

## 仓库地图

| 路径 | 内容 |
|---|---|
| [`HANDOFF.md`](HANDOFF.md) | 当前状态与下一步入口 |
| `rm-sim/src/` | 裁判、Torch World、Isaac 适配与 MAPPO 包 |
| `rm-sim/configs/` | 可复现实验配置 |
| `rm-sim/scripts/` | 训练、评估、可视化、Isaac 冒烟与性能基准入口 |
| [`AGENTS.md`](AGENTS.md) | 贡献流程与仓库约定 |

官方手册、比赛数据、Isaac 运行时、训练产物和 checkpoint 不随仓库分发。贡献者需要自行取得授权材料，并通过本地配置接入，不能将其提交到仓库。

## 参与贡献

提交工作前请阅读 [`AGENTS.md`](AGENTS.md)。任何裁判规则修改都应引用对应手册位置并补充回归测试；架构讨论需要明确区分官方规则与仿真假设。

## 许可与声明

RM-Cortex 采用 [MIT License](LICENSE) 开源。项目为独立研究项目，与 DJI 或 RoboMaster 不存在隶属或官方背书关系。
