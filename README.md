# PWARM
Physical World AI Reasoning Model

### v0.1 Research Preview — 一个自主科学发现的研究原型 / A research prototype for autonomous scientific discovery.

> **诚实定位 / Honest scope:** v0.1 证明的是一个可运行的 *scientific-agent
> prototype*——AI 能在模拟世界中自主发现定律、自选实验、在预算下做科学并申请
> 仪器。它**不是**通用科学智能；通用性是 v0.3+ 的目标。
> v0.1 proves a runnable *scientific-agent prototype* — an AI that discovers
> laws, designs its own experiments, does science under a budget, and requests
> instruments. It is **not** general scientific intelligence; that is v0.3+.

给 AI 科学家一个未知的模拟世界。/ Give an AI scientist an unknown simulated world.

它观察。/ It observes.
它设计实验。/ It designs experiments.
它形成假设。/ It forms hypotheses.
它验证它们。/ It tests them.
它积累知识。/ It accumulates knowledge.

**核心原则：AI 可以观察世界，但不能读取答案。**
**Core principle: the AI may observe the world, but it can never read the answers.**
---

## Mission 001 — AI 能否在不知道重力值的情况下发现重力？/ Can an AI discover gravity without being told its value?

```
估计重力: 9.81000 m/s²     Estimated gravity: 9.81000 m/s²
真值:     9.81000 m/s²     Ground truth:      9.81000 m/s²
实验次数: 3 (5m, 10m, 20m)  Experiments:       3 (5m, 10m, 20m drops)
置信度:   100%              Confidence:        100%
状态:     已验证            Status:            VERIFIED
```

AI 科学家从三个不同高度落下小球，将每条轨迹拟合为二次函数，交叉验证相同的重力加速度独立浮现——这是一条定律，不是巧合。

知识持久化：第二次运行从文明知识中得出结论，无需重新运行任何实验。

完整档案（12 节：隐藏变量、观察空间、原始数据、假设、验证、真值、误差、种子）：
[docs/missions/001-discover-gravity.md](docs/missions/001-discover-gravity.md) ·
可复现档案：[reproducibility/mission_001/](reproducibility/mission_001/)

---



## 快速开始 / Quick Start

```bash
git clone https://github.com/metis-thal/PWARM.git
cd PWARM
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -e .            # 终端体验（核心）/ terminal experience (core)
pip install -e ".[viz]"     # 加 GL 仪表盘 / add the GL dashboards
pip install -e ".[dev]"     # 开发 / development
```

```bash
# 一条命令（无 GPU）/ one command, no GPU:
pwarm demo                  # Mission 001 — discover gravity
pwarm demo 002              # Mission 002 — autonomous experiments + honest unknowns
pwarm demo 003              # Mission 003 — budget + instrument request arc
make test                   # 277 tests
make repro                  # regenerate every reproducibility envelope

# GL 仪表盘（需要窗口）/ GL dashboards (need a window):
python scripts/demo_mission_001.py
python scripts/demo_mission_002.py
python scripts/demo_mission_003.py
```

```python
from pwarm.scientist import ScientistAgent, ScientistState, Mission
from pwarm.scientist import (ScientificModel, model_prediction, disagreement,
                            ConditionBinding, OutputBinding, ObservationReduction)
from pwarm.universes import load_universe

universe = load_universe("universe_001")
# AI 自主发现隐藏在其中的东西。

# 候选模型：两个假设同时预测
h1 = ScientificModel(model_id="linear", params={"k": 1.0})
h2 = ScientificModel(model_id="quadratic", params={"k": 1.0})
pred_a = model_prediction(h1, {"x": 5.0})      # 5.0
pred_b = model_prediction(h2, {"x": 5.0})      # 25.0
D = disagreement(pred_a, pred_b)               # 只产出 prediction 和 disagreement

# Genesis 闭环：承诺先于观测，一次实验，逐模型独立裁决
knowledge = KnowledgeBase("knowledge/universe_001.json", universe="universe_001")
agent = ScientistAgent(Laboratory(universe), knowledge)
proposal = agent.propose_discriminating_experiment(
    (h1, h2), [{"x": 2.0}, {"x": 5.0}])              # x=5 区分度最大
commitments = agent.commit_discriminating_predictions(  # 预测 BEFORE 实验，hash 锚定
    (h1, h2), proposal, kind="drop",
    binding=ConditionBinding({"x": "drop_height"}))
observation = agent.execute_proposal(                    # 实验只执行一次
    proposal, kind="drop", binding=ConditionBinding({"x": "drop_height"}))
verdicts = agent.verify_competing_predictions(           # 每个模型独立 CONFIRMED/REFUTED
    commitments, observation, OutputBinding({"y": "z"}), "y",
    ObservationReduction(channel="z", rule="first"))
```

---

## 任务 / Missions

每个 Mission 回答一个关于人工科学智能的问题。/ Each Mission answers one question about artificial scientific intelligence.

### Mission 001 — 发现重力 / Discover Gravity

**AI 能否在不知道要找什么的情况下发现定律？/ Can an AI discover a law without being told what to look for?**

宇宙隐藏了 `g = 9.81 m/s²`。AI 只接收来自三个落体实验的 `(t, z)` 样本。它独立拟合每条轨迹，然后交叉验证相同的重力加速度浮现——将三次测量转化为一条定律。

```
规划 → 执行 → 观察 → 假设 → 交叉验证 → 发表
plan → execute → observe → hypothesize → cross-verify → publish
```

| 指标 / Metric | 结果 / Result |
|---------------|---------------|
| 实验次数 / Experiments | 3 |
| 测量值 g / Measured g | 9.81000 m/s² |
| 真值 / Ground truth | 9.81000 m/s² |
| 误差 / Error | 0.0007% |

### Mission 002 — 自主实验 / Autonomous Experimentation

**AI 能否决定测量什么？/ Can an AI decide what to measure?**

宇宙隐藏了两种具有未知 `恢复系数`、`摩擦系数` 和 `密度` 的材料。AI 将每个未知量建模为区间，按信息增益评分候选实验，并收获副产品——一次落体同时推导出重力和恢复系数。它还诚实报告它*无法*知道的：密度在重力+接触世界中不可识别（等效原理）。

```
未知量 → 不确定性 → 实验设计 → 观察 → 知识
Unknowns → Uncertainty → Experiment Design → Observation → Knowledge
```

| 指标 / Metric | 结果 / Result |
|---------------|---------------|
| 实验次数 / Experiments | 4 (自主选择) |
| 已识别参数 / Parameters identified | 5 of 6 |
| 不可识别 / Unidentifiable | 密度 (诚实报告) |
| 最大误差 / Worst error | 0.14% |

### Mission 003 — 约束下的科学 / Science Under Constraints

**当实验有成本时，AI 能否做科学？/ Can an AI do science when experiments cost something?**

宇宙授予有限预算：4 次实验、1400 个模拟步、12 个成本单位。AI 按 **价值 = 信息增益 / 成本** 排序候选方案——购买能产生知识的最便宜设计。当密度被落体/滑动装置证明不可识别时，AI 分析缺口，提交仪器申请，接收流体箱，并通过浮力识别密度。

```
预算 → 价值排序 → 实验 → 缺口分析 → 仪器申请 → 授权 → 新实验
Budget → Value Ranking → Experiment → Gap Analysis → Instrument Request → Grant → New Experiment
```

| 指标 / Metric | 结果 / Result |
|---------------|---------------|
| 实验次数 / Experiments | 6 (4 基础 + 2 授权) |
| 已识别参数 / Parameters identified | 7 of 7 |
| 预算纪律 / Budget discipline | 4 + 2 ≤ 6 |
| 仪器弧 / Instrument arc | fluid_tank 授权一次 |

### Mission 004+ — 未来之路 / The Road Ahead

- **Mission 004**: 流体与热力学 — 温度、黏度、热导率
- **Mission 004**: Fluid & Thermodynamics — temperature, viscosity, thermal conductivity
- **Mission 005+**: 化学、材料、光学、地质学
- **Mission 005+**: Chemistry, Materials, Optics, Geology

### Genesis — 预测-承诺-验证的科学闭环 / The Prediction–Commitment–Verification Loop

Genesis 层让 AI 科学家拥有完整的假设检验循环：多个候选模型在实验执行前各自
**承诺**（sha256 锚定、追加式、不可篡改）自己的预测；实验只执行一次；观测经
显式绑定与约简后，对每个承诺**独立裁决**为 CONFIRMED / REFUTED；证据按模型
累积为可追溯的事实账本。全程不读取任何隐藏真值。

The Genesis layer gives the AI scientist a full hypothesis-testing loop:
competing models each **commit** (sha256-anchored, append-only, tamper-evident)
their predictions BEFORE the experiment runs; the experiment runs ONCE; the
observation is bound and reduced through explicit contracts, then every
commitment is **independently adjudicated** to CONFIRMED / REFUTED; evidence
accumulates into a traceable per-model ledger. No hidden truth is ever read.

```
模型 A/B → 预测 → 区分度排序 → 提案 → 承诺(hash) → 实验一次
        → 观测 → 输出绑定 → 观测约简 → 逐模型裁决 → 证据账本
models A/B → predictions → rank discriminating conditions → proposal
        → commit(hash) → experiment ONCE → observation → output binding
        → reduction → per-model verdict → evidence ledger
```

核心数据结构 / Core data structures:

| 结构 / Structure | 职责 / Role |
|------------------|-------------|
| `ScientificModel` | 候选模型：model_id + 参数 / candidate model (id + params) |
| `Prediction` → `PredictionRecord` | 预测值 → 承诺记录：sha256 锚定、追加式、状态 open → confirmed/refuted / prediction → committed record: hash-anchored, append-only |
| `ConditionBinding` | 显式合同：模型条件变量 → ExperimentSpec 参数（未声明即失败）/ model condition variable → spec parameter |
| `OutputBinding` | 显式合同：模型输出变量 → 观测通道（t/z/vx）/ model output → observation field |
| `ComparisonInput` + `ObservationReduction` | 观测通道采样 → 裁决用标量（显式规则，第一版 `first`）/ channel samples → verdict-input scalar |
| `VerificationRecord` | 一次裁决的持久化事实：residual + status + experiment_id / one persisted verdict fact |
| `EvidenceSummary` → `CompetitionState` | 按模型聚合的实验账本；统计单位 = (model, experiment)，冲突显式报告 / per-model evidence; unit = model × experiment, conflicts reported |

事实而非评分：本层产出 residual 与预测级 CONFIRMED/REFUTED，但**不产出**
winner / weight / probability / model elimination——模型存活语义是下一阶段的
决策。全部合同（Condition/Output binding、Reduction）都是声明的数据，缺绑定
或越界目标一律大声失败，绝不静默猜测。

Facts, not scores: this layer yields residuals and per-prediction
CONFIRMED/REFUTED verdicts, but **no** winner / weight / probability / model
elimination — model-survival semantics are the next decision. Every contract
(condition/output binding, reduction) is declared data: a missing binding or
an out-of-vocabulary target fails loudly instead of being guessed.

---

## 架构 / Architecture

```
src/pwarm/
├── geology/    # 地质系统：地层学、热传导、侵蚀、构造运动
│               # Stratigraphy, thermal conduction, erosion, tectonics
├── rules/      # 多学科规则：力学、热力学、流体、材料、化学
│               # Mechanics, thermodynamics, fluids, materials, chemistry
├── ai/         # AI 推理/演化：观察器、假设、验证、实验
│               # Observer, hypothesis, verification, experimentation
├── physics/    # 统一多物理引擎 (类经典架构)
│               # Unified multi-physics engine (Genesis-inspired)
│   ├── core/   # 场景、状态、实体、组件 (唯一真值源)
│   │           # Scene, State, Entity, Component (single source of truth)
│   ├── solvers/ # 刚体、SPH、FEM、PBD、热力学、化学、地质
│   │           # Rigid, SPH, FEM, PBD, Thermal, Chemistry, Geology
│   ├── coupling/ # 显式多物理耦合器
│   │           # Explicit multi-physics coupler
│   ├── collision/ # SAP + GJK/EPA + CCD
│   └── integrator/ # 时间步进器 / TimeStepper
├── interface/  # URDF/MJCF/GLTF 解析器、GUI、传感器、并行环境
│               # URDF/MJCF/GLTF parsers, GUI, Sensors, Parallel envs
├── scientist/  # AI 科学家层
│               # The AI scientist layer
│   ├── state.py # 自我模型：不确定度区间 + 状态 / Self-model: uncertainty intervals + statuses
│   ├── knowledge.py # 文明知识库：定律 + 承诺/验证账本 + 证据聚合 / Civilization knowledge: laws + commitment/verification ledger + evidence aggregation
│   ├── prediction.py # Genesis 核心：承诺、裁决、双向绑定、观测约简、竞争状态 / Genesis core: commitments, verdicts, both bindings, observation reduction, competition state
│   ├── agent.py # 科学方法循环：预测 → 承诺 → 执行 → 验证 → 学习 / The scientific-method loop: predict → commit → execute → verify → learn
│   ├── information.py # 测量分辨率模型 / Measurement resolution models
│   ├── designer.py # 按价值 = 增益 / 成本选择实验 / Choose experiments by value = gain / cost
│   ├── budget.py + experiment_value.py # 预算账本 + 价值排序 / Budget ledger + value ranking
│   ├── instrument.py # 缺口分析 → 申请 → 目录授权 / Gap analysis → request → catalog grant
│   ├── hypothesis.py # 拟合 → 假设 → 交叉验证 / Fit → hypothesis → cross-verification
│   └── experiments/ # 落体测试、滑动测试、浮力测试、浸没测试 / drop_test, slide_test, buoyancy_test, immersion_test
└── viz/        # OpenGL GPU 实例化、PBR、光线追踪
                # OpenGL GPU instancing, PBR, ray-tracing
```

### 核心原则 / Core Principles

1. **零硬编码现象** — 燃烧、断裂、流动从方程中涌现。
   **Zero hardcoded phenomena** — combustion, fracture, flow emerge from equations.
2. **真值/AI 分离** — 物理引擎是绝对真值；AI 是学习到的近似。
   **Ground truth / AI separation** — physics engine is absolute truth; AI is a learned approximation.
3. **每阶段可视化验收** — 禁止无头开发；每阶段必须可演示。
   **Visual acceptance every phase** — no headless development; every phase demoable.
4. **承诺先于观测** — 预测在实验前被 hash 锚定并持久化；裁决只消费
   已承诺内容与新观测，历史不可篡改。
   **Commitment before observation** — predictions are hash-anchored and
   persisted before the experiment runs; verdicts consume only committed
   content plus the new observation, and history cannot be rewritten.

---

## 可复现性 / Reproducibility

```bash
pytest -q                    # 所有测试 / all tests
pytest tests/scientist/      # 仅科学家任务 / scientist missions only
make repro                   # 重新生成全部可复现性档案 / regenerate all envelopes
```

| 测试套件 / Test Suite | 测试数 / Tests | 状态 / Status |
|-----------------------|---------------|---------------|
| Scientist (Mission 001–004 + Genesis Steps 1–10) | 184 | ✅ 全部通过 |
| Engine / physics / geology / viz / rules / ai | 89 | ✅ 全部通过 (3 skip) |
| Reproducibility envelopes | 4 | ✅ 全部通过 |

每个 Mission 附带可复现档案
[`reproducibility/mission_XXX/`](reproducibility/)（config · seed · run.sh ·
expected_output · results），回答八个可信度问题：隐藏了什么 / AI 能观察什么 /
能做哪些实验 / 它选了什么 / 发现了什么 / 真值是什么 / 误差多大 / 种子是什么。
人侧评估（含真值对比与诚实失败记录）在
[`scientific_evaluation/`](scientific_evaluation/)。CI 在每次推送时于
Ubuntu + Windows 重跑全部档案并断言误差阈值。

知识跨会话持久化。第二次任务运行加载 `knowledge/universe_*.json` 并得出结论，无需重新运行实验。

---

## 研究 / Research

PWARM 旨在研究人工智能体如何通过实验获取科学知识。关键研究方向：

- **课程学习** 用于科学推理 / **Curriculum learning** for scientific reasoning
- **元认知**：何时停止测量并承认不确定性 / **Metacognition**: when to stop measuring and admit uncertainty
- **仪器获取**：AI 卡住时请求新能力 / **Instrument acquisition**: the AI requests new capabilities when stuck
- **信息价值**：预算感知的实验选择 / **Value-of-information**: budget-aware experiment selection

---

## 贡献 / Contributing

欢迎贡献。项目遵循以下约束：

- `physics/`、`render/`、`core/`、`solver/` 已冻结——不可修改
  `physics/`, `render/`, `core/`, `solver/` are frozen — no modifications
- 所有更改必须通过 `ruff check` 和 `pytest`
  All changes must pass `ruff check` and `pytest`
- 新 Mission 需要：宇宙配置、实验模块、测试、演示脚本、README 章节
  New Missions require: universe config, experiment modules, tests, demo script, README section

---

## 路线图 / Roadmap

| Mission | 领域 / Domain | 状态 / Status |
|---------|---------------|---------------|
| 001 | 重力 (自由落体) / Gravity (free-fall) | ✅ 完成 |
| 002 | 材料属性 (恢复系数、摩擦) / Material properties (restitution, friction) | ✅ 完成 |
| 003 | 预算约束 + 仪器获取 / Budget constraints + instrument acquisition | ✅ 完成 |
| 004 Day 1 | 未知液体测量基础设施 / Unknown-liquid measurement infrastructure | ✅ 完成 |
| 004 Day 2+ | 密度推断 + 流体定律 / Density inference + fluid laws | 计划中 |
| 005+ | 化学、材料、光学、地质学 / Chemistry, Materials, Optics, Geology | 未来 |
| Genesis | 预测-承诺-验证闭环（Steps 1–10）/ Prediction–commitment–verification loop | ✅ Steps 1–10 完成 |

---

## 许可证 / License

MIT
