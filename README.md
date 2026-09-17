# PWARM
Physical World AI Reasoning Model

### 一个自主科学发现的开放研究环境 / An open research environment for autonomous scientific discovery.

给 AI 科学家一个未知的模拟世界。/ Give an AI scientist an unknown simulated world.

它观察。/ It observes.
它设计实验。/ It designs experiments.
它形成假设。/ It forms hypotheses.
它验证它们。/ It tests them.
它积累知识。/ It accumulates knowledge.

---

## Mission 001 — AI 能否在不知道重力值的情况下发现重力？/ Can an AI discover gravity without being told its value?

```
python scripts/demo_mission_001.py
```

```
估计重力: 9.81000 m/s²     Estimated gravity: 9.81000 m/s²
真值:     9.81000 m/s²     Ground truth:      9.81000 m/s²
实验次数: 3 (5m, 10m, 20m)  Experiments:       3 (5m, 10m, 20m drops)
置信度:   100%              Confidence:        100%
状态:     已验证            Status:            VERIFIED
```

AI 科学家从三个不同高度落下小球，将每条轨迹拟合为二次函数，交叉验证相同的重力加速度独立浮现——这是一条定律，不是巧合。

知识持久化：第二次运行从文明知识中得出结论，无需重新运行任何实验。

---

## 为什么选择 PWARM？/ Why PWARM?

**物理是真实的。** 所有现象从底层微分方程中涌现——没有硬编码的动画，没有预设事件。AI 只能看到测量通道提供的信息。

**科学是诚实的。** AI 无法作弊。它通过科学方法获取知识：规划→执行→观察→假设→验证→发表。它不能知道的，就诚实承认。

**进展是渐进的。** 每个 Mission 增加一个新能力。AI 的世界随着方法的成熟而变得丰富。

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

pip install -e ".[core,ai,viz,parallel,dev]"
```

```bash
# 运行最简单的任务——发现重力
python scripts/demo_mission_001.py

# 自主实验设计
python scripts/demo_mission_002.py

# 预算约束下的科学与仪器申请
python scripts/demo_mission_003.py
```

```python
from pymo.scientist import ScientistAgent, ScientistState, Mission
from pymo.universes import load_universe

universe = load_universe("universe_001")
# AI 自主发现隐藏在其中的东西。
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

---

## 架构 / Architecture

```
src/pymo/
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
│   ├── state.py # 自我模型：不确定度区间 + 状态
│   │            # Self-model: uncertainty intervals + statuses
│   ├── information.py # 测量分辨率模型
│   │            # Measurement resolution models
│   ├── designer.py # 按价值 = 增益 / 成本选择实验
│   │            # Choose experiments by value = gain / cost
│   ├── budget.py + experiment_value.py # 预算账本 + 价值排序
│   │            # Budget ledger + value ranking
│   ├── instrument.py # 缺口分析 → 申请 → 目录授权
│   │            # Gap analysis → request → catalog grant
│   └── experiments/ # 落体测试、滑动测试、浮力测试
│                   # drop_test, slide_test, buoyancy_test
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

---

## 可复现性 / Reproducibility

```bash
pytest -q                    # 所有测试 / all tests
pytest tests/scientist/      # 仅科学家任务 / scientist missions only
```

| 测试套件 / Test Suite | 测试数 / Tests | 状态 / Status |
|-----------------------|---------------|---------------|
| Mission 001 | 5 | ✅ 全部通过 |
| Mission 002 | 8 | ✅ 全部通过 |
| Mission 003 | 8 | ✅ 全部通过 |
| **总计 / Total** | **21** | **✅** |

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
| 004 | 流体与热力学 / Fluid & Thermodynamics | 计划中 |
| 005+ | 化学、材料、光学、地质学 / Chemistry, Materials, Optics, Geology | 未来 |

---

## 许可证 / License

MIT
