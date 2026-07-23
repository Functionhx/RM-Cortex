# HANDOFF — RM-Cortex

- 更新日期：2026-07-23
- 当前阶段：官方 V1.5.0 规则审计与 Phase 1 设计已完成，代码尚未脚手架化
- 项目目标：建立可验证的 RMUC 裁判、批量仿真、多智能体训练与后续 sim2real 链路

## 必读顺序

1. [`docs/superpowers/specs/2026-07-23-rmuc-v1.5.0-pdf-audit.md`](docs/superpowers/specs/2026-07-23-rmuc-v1.5.0-pdf-audit.md)：本轮纠错、工程图读取结果与残余歧义。
2. [`docs/superpowers/specs/2026-07-23-rmuc-rules-digest.md`](docs/superpowers/specs/2026-07-23-rmuc-rules-digest.md)：带 `[TXT]`、`[FIG]`、`[SIM]`、`[AMB]` 标签的规则索引。
3. [`docs/superpowers/specs/2026-07-23-rmuc-referee-system-design.md`](docs/superpowers/specs/2026-07-23-rmuc-referee-system-design.md)：纯 Torch + Isaac Lab 架构。
4. [`docs/superpowers/specs/2026-07-23-rmuc-implementation-plan.md`](docs/superpowers/specs/2026-07-23-rmuc-implementation-plan.md)：M1～M5 的执行和验收顺序。

规则真值层级为：**最新官方规则/答疑 > V1.5.0 PDF > digest > 设计/实现**。官方手册和数据集只在本地保存，不进入公开仓库。

## 已冻结的设计

- `rm_referee` 是唯一裁判真值层，只依赖 Torch，不导入 Isaac。
- Phase 1 包含纯 Torch 运动学后端和 Isaac Lab `DirectMARLEnv` adapter；两端共享状态、规则、事件和随机样本。
- 裁判固定 10Hz，运动学默认 60Hz，策略默认 5Hz。
- 高层普通射击目标为 9 类：5 个敌方地面机器人、敌方基地、敌方前哨站、己方能量机关、无目标。
- 常规命中率、小陀螺收益、雷达漂移和工程取件抽象属于 `[SIM]`，必须可配置、可标定。
- 真实数据用于回放与命中模型校准，不再把 1Hz、缺目标归因的日志当作逐发精确真值。

## 不得重新引入的错误

| 错误说法 | 正确语义 |
|---|---|
| 热量/功率超限扣血 | 热量锁定；功率缓冲耗尽后底盘断电 5 秒，均不扣血 |
| 50/200ms 是武器射速上限 | 是单块装甲接受 17/42mm 伤害的检测间隔 |
| 弹量是队伍共享 | 弹量按机器人；只有金币购买累计按队 |
| 42mm 购买上限 1000 | 42mm 为 100 发/队，17mm 才是 1000 发/队 |
| 读条复活为满血 | 10% HP、虚弱、最长 30 秒无敌 |
| 基地护甲展开自带减伤 | 只改变可攻击几何 |
| 空中是普通受击单位 | 不受普通伤害/撞击，不回血、不复活 |
| 雷达直接固定增加 `P` | 先更新连续量 `x`，再执行 `P += x` |

## 仍需显式保留的歧义

- 热量正文使用 `Q1>Q2`，流程图使用 `Q1≥Q2`；默认按流程图实现并保留开关。
- 官方图未定义场地坐标原点/轴向，也未给出起伏路凸起高度。
- 手册没有给出常规命中、小陀螺收益、雷达漂移或不完整飞镖识别的概率函数。

## 下一步

从实现计划 M1 开始，在 `rm-sim/` 创建 `pyproject.toml`、`src/rm_referee/` 和 `tests/rules/`。先冻结张量 schema 和带页码的常量，再按 `fsm/objective → combat → heat/power → economy/upgrade → survive/roles → rune/dart/radar` 实现。M1 规则测试通过后才进入 Torch World，随后接入 Isaac Lab 并建立逐 tick 契约测试。

`RMUC-OfflineRL/` 是本地参考仓库，不导入、不修改、不提交。PDF、数据集、Isaac 运行时、训练日志与 checkpoint 均已由顶层 `.gitignore` 排除。
