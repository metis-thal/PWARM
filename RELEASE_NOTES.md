# PWARM v0.1.0-alpha Release Notes / 发布说明

**Physical World AI Reasoning Model — 首个开源版本 / First open-source release**

> v0.1.0-alpha 是 PWARM "AI 科学家"主线前三章的快照：AI 能发现定律、能自选实验、
> 能在预算下做科学并向系统申请仪器。
> v0.1.0-alpha snapshots the first three chapters of the "AI scientist" arc:
> an AI that discovers laws, designs its own experiments, and does science
> under a budget — including requesting new instruments when stuck.

## Unreleased — Genesis: the prediction–commitment–verification loop

Steps 1–10 of the Genesis arc turn the scientist from a *parameter
identifier* into a *hypothesis-testing agent* (all AI-side, all
measurement-only; no schema breaks, Missions 001–003 pinned by regression):

- **Commitment before observation** (Steps 1/6) — predictions are
  hash-anchored (sha256) and persisted BEFORE the experiment runs;
  tampering is detected; competing models commit independently against
  the same spec.
- **Belief-driven predictions** (Step 2) — predictions are translated from
  the scientist's own uncertainty intervals (no hard-coded priors); a
  confirmed verdict tightens the belief, a refuted one widens it —
  deterministic, no Bayesian machinery.
- **Discriminating proposals** (Steps 2–4.5) — rank candidate conditions by
  model disagreement; explicit contracts bind model vocabulary to
  experiment parameters (ConditionBinding) and model outputs to
  observation channels (OutputBinding); execution goes through the
  standard Laboratory channel, once.
- **Per-model verdicts** (Steps 5–7) — observation reduction to a scalar
  (declared rule), and one CONFIRMED/REFUTED verdict per committed
  prediction from a single observation; verdicts never re-run physics.
- **Evidence ledger** (Steps 8–10) — per-model evidence summaries and
  competition states with full provenance; the statistical unit is
  (model, experiment); disagreements are reported as conflicts, never
  tie-broken. Facts, not scores.
- **Mission 004 Day 1 restored** — unknown-liquid measurement
  infrastructure (universe_004, immersion apparatus, reproducibility
  envelope).
- **Tests**: 277 total (scientist 184), Ruff clean, reproducibility
  envelopes byte-stable.

## Highlights / 亮点

### AI Scientist（三层任务课程 / three-mission curriculum）

| Mission | 能力 / Capability | 结果 / Result |
|---------|-------------------|---------------|
| 001 — Discover Gravity | 从自由落体观测中发现定律，交叉验证并持久化知识 | g 误差 0.0007% |
| 002 — Autonomous Experimentation | 不确定性建模 + 按信息增益自选实验；诚实报告不可识别参数 | 5/6 参数，误差 ≤0.14% |
| 003 — Science Under Constraints | 预算（4 实验/1400 步/12 成本单位）下的**价值排序**决策；不可识别 → 缺口分析 → **仪器申请 → 授权 → 浮力实验** | 7/7 参数（含密度），仪器弧完整 |

Mission 003 的仪器弧是本版本的核心叙事：AI 用落体/滑动装置穷尽后诚实地承认
"密度不可识别"（等效原理），分析缺失的测量能力（需要浸没），提交流体箱申请，
系统授权并扩展预算，浮力实验随即测出 7799.95 / 2700.0 kg/m³（真值 7800/2700）。

### 统一多物理引擎 / Unified multi-physics engine

- 刚体 / SPH / FEM / MPM / PBD / 热力学 / 化学 / 地质求解器，显式耦合器
  Rigid / SPH / FEM / MPM / PBD / Thermal / Chemistry / Geology solvers with an explicit coupler
- 统一碰撞：SAP 宽相 + GJK/EPA 窄相 + CCD
  Unified collision: SAP broad-phase + GJK/EPA narrow-phase + CCD
- 核心原则：零硬编码现象，物理引擎是绝对真值，AI 只能通过测量通道观察
  Zero hardcoded phenomena; the engine is absolute truth, the AI observes
  through a measurement-only channel

## 知识持久化 / Knowledge persistence

`knowledge/universe_*.json` 保存已确立的定律与材料属性。第二次任务运行直接从
"文明知识"得出结论，不重跑任何实验（删除对应 JSON 即可让科学家从零开始）。

## 测试 / Tests

`pytest tests/` — **70 passed, 3 skipped**（其中 scientist 21 项全覆盖三个 Mission）。

## 破坏性变更 / Breaking changes

- 旧版 `pwarm.kernel`（2D/3D 遗留内核）已删除；请迁移到 `pwarm.physics`
  Legacy `pwarm.kernel` removed — migrate to `pwarm.physics`
- 项目发行名由 `pymo` 改为 `PWARM`（v0.1.0-alpha 时导入包名仍为 `pymo`，
  现已一并更名为 `pwarm`——见上方 Unreleased 段）
  Distribution renamed `pymo` → `PWARM` (at v0.1.0-alpha the import package
  was still `pymo`; it has since been renamed to `pwarm` — see Unreleased)
- 实验设计器准则从"信息增益最大化"升级为"价值 = 增益/成本"：
  Mission 002 的首选落体高度由 50 m 变为 10 m（同样达到知识阈值，成本 1/4）
  Designer doctrine upgraded from max-information to value = gain/cost

## 已知限制 / Known limitations

- Python 3.10–3.11；numba 为核心依赖（JIT 编译）
- 可视化依赖 OpenGL 3.3+；无 GPU/无头环境请用 `--frames N` 自动退出或仅跑测试
- 符号回归的 gplearn 后端为可选依赖（`ai.law_discovery` 懒加载），未安装时自动降级
- 尚无 CI 工作流；Windows 为第一开发平台，Linux 未系统验证

## 下载与验证 / Install & verify

```bash
git clone https://github.com/metis-thal/PWARM.git
cd PWARM && python -m venv .venv && .venv/Scripts/activate   # Windows
pip install -e ".[viz,dev]"
pytest tests/scientist/ -q
python scripts/demo_mission_001.py    # 001 → 002 → 003
```

## 路线图 / Roadmap

Mission 004 — Fluid & Thermodynamics（温度、黏度、热导率，热学仪器申请弧）已规划，
详见 README 路线图表。
