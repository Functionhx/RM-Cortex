# RM-Cortex Phase 1 实现计划

- 日期：2026-07-23
- 状态：V1.5.0 规则审计后修订
- 目标：用纯 Torch 规则核心 + 纯 Torch 运动学后端 + Isaac Lab 适配层跑通可复现的多智能体闭环
- 必读顺序：[`规则审计`](2026-07-23-rmuc-v1.5.0-pdf-audit.md) → [`规则 Digest`](2026-07-23-rmuc-rules-digest.md) → [`系统设计`](2026-07-23-rmuc-referee-system-design.md)

## 1. 交付里程碑

| 里程碑 | 交付物 | 完成条件 |
|---|---|---|
| M0：规则基线 | PDF 审计、digest、歧义/仿真假设清单 | 已完成；规则与 `[SIM]` 分层 |
| M1：裁判核心 | `rm_referee` 纯 Torch 包 | 规则边界单测、性质测试全部通过，零 Isaac import |
| M2：Torch World | 批量运动学、场地、LOS、装甲几何、脚本对手 | CPU 单环境可调试；GPU 1024+ 环境稳定 rollout |
| M3：Isaac Lab | `DirectMARLEnv` adapter、资产/传感器同步 | 与 Torch 后端逐 tick 规则一致，最小场景可视化运行 |
| M4：训练闭环 | 异构 action mask、obs/reward、PPO/MAPPO baseline | 可复现实验配置、评估脚本、脚本对手胜率报告 |
| M5：校准 | 数据回放、命中模型参数、域随机化 | 报告可观测误差；不对缺失目标归因做虚假精确承诺 |

M1～M4 共同构成 Phase 1。Isaac Lab 不是第二个裁判实现，而是同一 Torch 规则核心的场景后端。

## 2. 目录与包

```text
rm-sim/
├── pyproject.toml
├── src/
│   ├── rm_referee/
│   │   ├── constants.py
│   │   ├── schema.py
│   │   ├── state.py
│   │   ├── events.py
│   │   ├── random_tape.py
│   │   ├── referee.py
│   │   └── modules/
│   │       ├── combat.py
│   │       ├── heat.py
│   │       ├── power.py
│   │       ├── economy.py
│   │       ├── upgrade.py
│   │       ├── survive.py
│   │       ├── objective.py
│   │       ├── roles.py
│   │       ├── rune.py
│   │       ├── dart.py
│   │       ├── radar.py
│   │       └── fsm.py
│   ├── rm_world/
│   ├── rm_isaac/
│   └── rm_train/
├── tests/
│   ├── rules/
│   ├── contracts/
│   └── replay/
├── configs/
└── scripts/
```

`rm-sim/` 留在顶层仓库中，不创建嵌套 Git 仓库。官方 PDF、数据集、Isaac 运行时和训练产物只保留在本机。

## 3. M1：纯 Torch 裁判核心

### 3.1 先冻结张量契约

- `unit_state[E,16,C]`：红蓝各包含英雄、工程、步兵 3/4、空中、哨兵、基地、前哨站。
- `armor_state[E,16,A,Ca]`：装甲朝向、可见性、最近 17/42mm 受击时间。
- `team_state[E,2,Ct]`：金币、按队购买累计、科技核心、飞镖、雷达和增益。
- `match_state[E,Cm]`：阶段、时间、胜者、终止原因。
- 允许发弹量和立即复活次数放在单位状态，不放在队伍状态。
- 空中的 HP/受击/回血/复活通道始终掩码。

状态更新可采用 dataclass 包装，但热路径只操作固定形状张量。允许遍历固定的规则模块，不允许逐环境或逐单位 Python 循环。

### 3.2 转录常量

`constants.py` 逐项标注手册页码，并至少包含：

```python
HEAT_PER_SHOT = {17: 10, 42: 100}
HEAT_Q2_OFFSET = {17: 100, 42: 200}
HEAT_SETTLE_HZ = 10
POWER_BUFFER_J = 60
POWER_SETTLE_HZ = 10
ARMOR_REFRACTORY_S = {17: 0.050, 42: 0.200}
PURCHASE_CAP_PER_TEAM = {17: 1000, 42: 100}
XP_THRESHOLDS = [0, 550, 1100, 1650, 2200, 2750, 3300, 3850, 4400, 5000]
BASE_HP = 5000
BASE_SHIELD = 150
OUTPOST_HP = 1500
BASE_ARMOR_DEPLOY_HP = 2000
```

`ARMOR_REFRACTORY_S` 只过滤同一装甲模块接受的伤害，不能限制射击、热量或弹量。`[AMB]` 热量等号边界由配置项控制，默认使用流程图的 `Q1 >= Q2`。

### 3.3 规则实现顺序

1. `fsm/objective`：比赛阶段、基地/前哨站保护和胜负阶梯。
2. `combat`：实际射击事件、装甲相交、检测间隔、伤害/暴击/屏蔽。
3. `heat/power`：10Hz 热量状态机、60J 缓冲与底盘断电。
4. `economy/upgrade`：定时金币、兑换、每机器人弹量、经验等级。
5. `survive`：脱战、回血、读条/立即复活、虚弱/无敌。
6. `roles`：英雄部署、工程防御、空中支援、哨兵姿态、底盘能量。
7. `rune/dart/radar`：各自状态机和事件，不把效果散落到环境层。

所有模块返回张量化 event flags；reward 和 UI 订阅事件，但不反向修改规则。

### 3.4 M1 必测纠错

- 热量和功率超限不扣血。
- 50/200ms 检测间隔不减少实际发射、弹量消耗或热量。
- 17/42mm 购买上限分别为每队 1000/100，弹量按机器人持有。
- 读条复活为 10% HP；立即复活次数按机器人累计。
- 空中不能成为普通攻击目标，也不走回血/复活。
- 基地护甲展开不提供隐式减伤。
- 雷达先更新 `x` 再累计 `P`。
- 经验阈值最后一档为 5000。
- 终止时完整执行胜负阶梯，包括双方基地同时为 0 的边界。

## 4. M2：Torch World

1. 声明仿真坐标：场地中心为原点，`+x` 指向蓝方、`+y` 按右手系；红蓝通过 `x` 镜像。该定义标记为 `[SIM]`。
2. 用 Torch primitive 表示 28×15m 场地、建筑、坡面和占领区；图纸公差进入域随机化。
3. 实现批量 2.5D 运动学、速度/角速度约束、解析边界与简化碰撞。
4. 用射线—装甲平面相交得到目标、装甲编号、入射点和 10×10mm 暴击区。
5. 将距离、角误差、LOS、投影面积交给可插拔 `HitModel`；小陀螺收益等参数保留 `[SIM]`。
6. 实现脚本对手，先完成“移动—瞄准—射击—建筑终局”的最小闭环，再接入支线任务。

默认运动学步长为 `1/60s`，裁判步长 `0.1s`，策略步长 `0.2s`。射击事件携带子步时间戳，避免在低频策略步内丢失装甲检测窗口。

## 5. M3：Isaac Lab 适配

1. 用 `DirectMARLEnv` 建立场景生命周期、并行环境和 reset。
2. Phase 1 通过 kinematic controller 同步 root pose；Isaac 提供资产、遮挡/接触和可视化。
3. 将 Isaac 几何输出规范化为与 Torch World 相同的 `RuleInputs`。
4. 调用同一个 `Referee.step`，再把 HP、护甲展开、灯效等状态回写场景。
5. 建立 contract fixture：固定初态、动作、几何事件和 `RandomTape`，逐 tick 比较规则状态；运动状态仅允许声明过的浮点容差。

Isaac adapter 中禁止出现伤害值、金币表、复活公式或胜负判断。

## 6. M4：策略接口与训练

高层普通射击目标固定为 9 类：

```text
5 个敌方地面机器人 + 敌方基地 + 敌方前哨站
+ 己方能量机关 + 无目标
```

- 英雄/步兵/哨兵：导航目标、目标类别、开火门、姿态/小陀螺。
- 工程：导航、占领、取件/装配/重建。
- 空中：3D 导航、开火、支援/返航；不放入敌方普通射击 target mask。
- 飞镖和雷达：独立低频离散动作头。

先用参数共享 actor + 角色 embedding；actor 只接收可观测信息，centralized critic 可读取训练真值。action mask 在采样前应用。奖励从击伤、击毁、资源、任务进度和终局事件构造，并通过消融报告防止 reward hacking。

## 7. 回放与校准

数据集是 1Hz 观测，且部分射击/受击记录缺少目标归因，因此分两种验证：

1. **事件注入回放**：把已记录伤害事件直接作为命中事实，精确验证热量、资源、状态机和终局。
2. **自由命中回放**：使用几何命中模型，只比较队伍级伤害分布、建筑轨迹、资源时序和胜方等统计量；逐机器人 HP 不设伪精确硬门槛。

规则正确性由官方边界测试保证；数据用于校准规则未定义的命中/噪声参数。训练、校准和最终评估必须按比赛/日期拆分，避免同局泄漏。

## 8. 验收命令

脚手架建立后，从 `rm-sim/` 运行：

```bash
python -m pip install -e ".[dev]"
python -m compileall src
pytest -q tests/rules
pytest -q tests/contracts
pytest -q tests/replay
python scripts/benchmark.py --num-envs 1024 4096
```

Isaac 冒烟测试通过项目配置的 Isaac Lab launcher 执行，具体运行时路径由本机环境变量提供，不写死到源码或提交符号链接。

## 9. Phase 1 完成定义

- 所有 V1.5.0 已转录分支均有页码和测试；`[AMB]`/`[SIM]` 可枚举。
- Torch 与 Isaac adapter 共享同一规则包，规则状态逐 tick 一致。
- Torch World 可在 GPU 上批量训练，Isaac 场景可完成等价 rollout 和可视化。
- baseline 可从固定 seed 复现，并报告胜率、吞吐、显存、规则异常计数。
- 仓库不包含官方 PDF、原始数据、运行时、密钥、checkpoint 或机器专属路径。
