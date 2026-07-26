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

当前实现已经包含向量化裁判、带 LOS 与装甲几何的 2.5D Torch 场地、确定性脚本控制器、分层行为树基线、可安全选装的 Isaac Lab `DirectMARLEnv`，以及带信念状态实体注意力、集中式 critic、动作掩码、checkpoint 和评估流程的参数共享 MAPPO 基线。

使用 `python scripts/benchmark_world.py --num-envs 1024 4096` 测量 Torch World 吞吐；在 Isaac Lab 启动环境中使用 `PYTHONPATH=src /path/to/IsaacLab/isaaclab.sh -p scripts/isaac_smoke.py --headless` 验证可选场景后端。

## 观测模型与雷达

Phase 1 训练默认对全部 16 个单位槽使用队伍共享的**信念观测**。Actor 接收每条轨迹的位置与速度估计、最近一次观测到的决策状态、分开记录的位置/状态时延、有物理上界的半正定协方差和信息来源。己方共享状态或本地视野会刷新完整轨迹；雷达上报只更新位置与速度，不会把陈旧的血量或弹药伪装成新信息；敌方离开观测范围后按恒速模型预测，不再暴露当前真值。在实验配置中设为 `"observation_mode": "oracle"` 可保留全信息性能上界，评估会自动沿用 checkpoint 保存的观测模式。

雷达系统与场内前哨站不是同一个设施。按手册 5.6.6 节，准确/半准确上报会累积标记进度；敌方目标在 `P >= 100` 时显示确认位置，地面机器人在 `P >= 100/120` 时分别获得 15%/20% 易伤。雷达以绝对场地坐标按 5 Hz 策略步单次上报，策略也可明确选择不上报。实现还覆盖双倍易伤、干扰压制和激光反制空中机器人。MAPPO 当前只借用前哨站策略槽承载队伍级雷达指令，裁判层的雷达动作仍然是队伍级状态。

## 分层行为树基线

低层地面导航固定而不参与学习：纯 Torch 世界根据静态场地几何，在 `0.05 m` 网格上执行带缓存、确定性的 A*。其上由一棵全局树发布团队意图，再由八棵独立角色树——英雄、工程、两台步兵、空中、哨兵、基地和前哨站——选择高层任务与可审计叶节点。只有五个地面移动角色请求 A* 路径，空中与建筑角色保留各自执行逻辑；确定性执行器负责战斗、局部分离、经济、雷达和兵种规则动作。

运行换边配对评估：

```bash
cd rm-sim
python scripts/evaluate_behavior_tree.py \
  --opponent tactical \
  --seeds 1007 \
  --device cpu
```

每个 seed 产生两条独立 leg，行为树控制器先执红方、再执蓝方。每条 leg 在比赛终局或官方 420 秒上限结束；`--max-policy-steps` 只用于显式截断冒烟，未完成对局会单独报告而不计作平局。双方控制器均读取完整 `GameState`/`KinematicState`，并可输出 MAPPO 尚未覆盖的完整 `WorldActions`。因此它是 **oracle/full-action 工程基线**，不是与信念策略公平对比的算法基准；下面的实现级 pilot 也不是正式胜率结论。

使用 seeds `1007`、`2007`、`3007` 的实现级 pilot 已完成六条 420 秒换边 leg，A* 规划 `227/227` 成功。对 tactical controller 为 `3–3`（`balanced_score=0.5`），队伍平均累计回报为 `6.539`，对手为 `10.851`。这验证的是完整执行链路，不代表战术优于对手；样本量小且 seed 敏感，不能作为正式胜率。

## RMUC 数据实验

当前 CLI 保留首个取得授权的 1 Hz RMUC 数据实验，用于复现和诊断：

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

适配器强制以只读方式打开 SQLite；异常轨迹只做掩码，不裁剪、不前向填充，并在生成窗口前按完整队伍连通分量和比赛时间切分。当前三赛区数据自然形成南部训练、东部验证、北部锁定测试。离线输入与线上 34 维 actor、16×51 维信念实体契约一致：敌方真值必须先经过合成距离/LOS 感知和因果的“最后可见＋恒速预测”信念。

这个诊断实验的标签是截断的未来 5 秒位移与区间射击事件，不是 0.2 秒动作、目标选择或行为树决策。迁移会复制共享编码器/actor body 与移动角色行；MAPPO 动作头和 critic MLP 参数保留初始化，但迁入的角色/队伍 embedding 也会改变 critic 的输入表征。离线数据仅保留 10 秒历史，也不同于线上从开局持续维护的 tracker。

首轮使用 8,192/2,048 个窗口，射击权重为 `0.5`。验证集位移 RMSE 优于“原地不动”基线（`2.219 m` 对 `2.427 m`），但 MAE 更差（`1.458 m` 对 `1.368 m`）；射击 F1 为 `0.578`。在相同 seed 7、64 环境、50 次 MAPPO 更新下，seed 1007 的确定性红方单侧评估中，BC 初始化对脚本蓝方为 `0–64`、回报 `-3.672`，从头训练则为 `62–2`、`+23.546`。这项单 seed、单侧负迁移不能证明人类数据有害，只说明当前表征迁移目标未达到采用门槛。

RMUC 人类数据的后续用途放在高层：学习团队任务、角色任务、可可靠推导的攻击目标，以及行为树叶节点/option 选择；固定 A* 继续承担低层导航。位移/射击实验只保留为负向消融，不再把仅导航的 PPO/BC 作为路线。默认模仿配置已关闭射击代理。

## 可视化

导出无需 Isaac Lab 的轻量 2D 对局动画：

```bash
cd rm-sim
python -m pip install -e ".[visualization]"
python scripts/visualize_torch.py
```

导出需要系统 `PATH` 中存在支持 H.264（`libx264`）的 `ffmpeg`。生成但不纳入版本控制的 `outputs/torch_demo.mp4` 会完整仿真官方 420 秒对局，并按每个仿真秒采样一帧，压缩为约 21 秒 H.264 回放。演示使用的是确定性战术脚本而非训练好的 checkpoint，包含英雄部署、工程支援、步兵分路、哨兵防守、目标集火、虚弱撤回、官方购弹流程和按库存分波次出动的空中支援。无人机仅在请求支援且免费库存可用时起飞，停机坪上、暂停支援或被雷达锁定时不能开火；运动范围还受手册 4.5 节飞行区与安全绳包络约束，具体出动波次属于战术策略，并非官方固定窗口。

回放统一使用等比例 `36 px/m` 变换：`28×15 m` 场地严格占据 `1008×540 px`，外层画布为 `1120×706 px`，高度栅格直接在最终像素中心采样，不再横向拉伸。低饱和顺序色、50mm 等高色带、坡向箭头和高度图例用于区分高程。物理设施尺寸与裁判区域分开绘制：基地平面边界沿场地 x/y 轴分别为 `1.609×1.881 m`，前哨站主体为 `Ø0.550 m`（手册的 `0.650 m` 仅是底座单向宽度）。停机坪外层结构包络为 `2.200×2.858 m`，中央停机八边形为 `2.149×2.200 m`、水平和竖直直边均为 `1.334 m`；梯形高地（`10.805×4.380 m`）、公路区（`8.901×3.651 m`）、堡垒（`2.240×1.939 m`）、飞坡（`1.145×0.860 m`）和崎岖道路采用图 4-25 至图 4-37 的尺寸。PDF 没有唯一标注的全局位置或内部坡面继续标记 `[SIM]` / `[AMB]`，当前场景不是官方 CAD。导出器发现任何地面外廓重叠都会直接失败；仅需短时冒烟时可追加 `--steps 300`。

```bash
PYTHONPATH=src /path/to/IsaacLab/isaaclab.sh \
  -p scripts/visualize_isaac.py --num-envs 1
```

CUDA 仿真不可用或显存被占用时可追加 `--device cpu`。Isaac 场景运行同一战术控制器、同一组多边形 footprint、逐顶点坡面高度与权威 Torch 比赛状态；非矩形表面会确定性三角剖分为网格而不是退化为外接矩形，每个网格顶点还会叠加其世界坐标处的场地横坡高度，与 Torch 地形表面保持一致。起伏路按 240mm 间距生成 70mm 条带，也不会引入第二套规则。除明确标注尺寸的模块外，其余地形位置仍是 Phase 1 `[SIM]` 近似，不是官方 CAD。

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
- [x] 用带时延和协方差的实体信念替换观测范围外的 actor 真值
- [x] 在原生 Isaac Lab 无头运行时中完成场景验证
- [x] 加入“全局＋八角色”行为树基线与 0.05 m A* 导航
- [x] 加入只读 RMUC 适配器并复现首轮负迁移诊断
- [ ] 使用授权比赛数据学习高层任务、目标与行为树叶节点选择
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
