# Open Genesis Pilot Specification (v2 — post adversarial review)

Status: SPECIFICATION v2 — incorporates adversarial-review fixes F1–F10.
No code, schema, tests or harness implemented. Implementation of G-1 is
gated (see Implementation Gate).

---

## 0. Pilot 命题（可证伪）

> Scientist 能否在一个答案没有预先注入知识层的微型未知世界中，从观察开始，
> 经过至少一轮自主研究，形成一个可验证的新关系，并对 held-out 条件做出
> 正确预测。

不是证明"可以创建世界"；是证明"可以从观察冷启动走完一圈自主科学"。

## 0.1 修订状态

v2 依据 adversarial review 修订：F1/F2（measurement-only boundary）、
F3（tolerance generation rule）、F4（harness execution log）、F5/F6
（scope 非回溯 + E-1 判据替换）、F7/F10（G-1 纯度边界）、F8（belief path
零调用升级为验收判据）、Generic Capability vs Domain Knowledge 小节、
更新后的成功/失败判据、Implementation Gate。

## A. 初始世界（Universe-005 `genesis_pilot`）

- **存在的物理**：与 universe_001 同族的 rigid body + 均匀重力 + 地面接触
  （restitution/friction 经接触对生效）——即 drop 实验族可运行。
- **隐藏常量**：gravity 与材料常量取**非教科书值**（如 g = 7.2），配置仅存
  在于 physics/world 侧。
- **scientist-visible side 与 world side 的边界（F1/F2，可审计条件）**：
  1. scientist 在 Pilot 中获得的唯一执行入口是 measurement-only facade：
     `run_experiment(spec) -> ObservationRecord`；
  2. 审计断言 A-1：`truth_summary` 不在 scientist 可达对象面上
     （hasattr 断言为假或门面无此成员）；
  3. 审计断言 A-2：`start_experiment` / `ExperimentSession` / `.universe`
     属性 / Universe 实例不在 scientist 可达面；
  4. 审计断言 A-3：scientist 可达对象图中不存在 world configuration、
     physics constants、material constants（对象图扫描）；
  5. 审计断言 A-4：world 配置文件路径不在 scientist 可达输入中。
- **边界归属声明**：truth / Universe / configuration 只属于 harness/world
  side；scientist-visible side 只获得实验测量结果（ObservationRecord 的
  测量通道）。
- **scientist 初始 ontology / vocabulary**：空。不实例化 `ScientistState`
  （规避 `DEFAULT_PRIOR_BOUNDS` 先验）；不使用信念路径（见 F8——这是验收
  判据，不是建议）。
- **初始 KnowledgeBase**：laws / predictions / verifications /
  model_records / definitions / relations / questions 全部为空。
- **初始 RelationRecord = 空**；**初始 ModelRecord = 空**。
- **授予的能力（声明性、ontology 中性）**：`drop` 实验种类的运行许可 +
  一个**发布在先的**探索条件池（池由 harness 固定并在运行前公开，其构造
  与隐藏常量无关）。

## B. 研究入口

唯一入口是 observation：scientist 从授权池选条件 → 门面执行 →
`ObservationRecord`（t/z 测量序列）。随后用**中性数学拟合**（LawDiscovery
PolynomialBackend——通用多项式回归，不注入物理常量；**不使用**
`fit_free_fall`/`_fit_free_fall`，见 G-1 purity boundary）从观测形成模型
形式与候选关系。

禁止：直接把答案写入 RelationRecord；直接给 scientist 正确 formula；直接
注册目标 model；用 hard-coded target relation 指导 planner；用 hidden
ground truth 作为 question selector。

## C. 自主闭环与冷启动缺口

目标链条：observation → question → hypothesis/model → prediction →
commitment → experiment → verification → evidence → discovery →
held-out prediction。

AS v0 的映射：

| 链节 | 现状 |
|---|---|
| observation → neutral fit → register ModelRecord + declare RelationRecord（scope = 已观测条件） | **缺口 G-1**：冷启动组合器尚未存在（各件齐备：门面、LawDiscovery、register_model、define_concept、declare_relation——缺"从原始观测到声明"的接线；其纯度边界见 G-1 purity boundary 节） |
| question | 已有：模型注册后 `unverified_identity` 扫描器自动开火 → `research_cycle` 声明并执行 |
| prediction/commitment/experiment/verification | 已有：`run_held_out_trial` / 竞争链（承诺先于执行——配合 F4 执行日志可证） |
| evidence → discovery | 已有：RelationEvidence 自动更新；`discoveries()` 派生 |
| held-out prediction | 已有：HP 纪律（scope 补集选择） |

**明确缺口**：仅 G-1 一个接线缺口（其纯度边界已限定，见下）；其余全部为
既有能力。

## D. 独立验证（scientist-visible facts vs world ground truth）

- **scientist-visible**：KnowledgeBase + ledger 的全部声明与派生。
- **world ground truth**：universe 配置 + 独立 verifier。verifier 是
  harness 侧独立组件，不 import scientist、不被 scientist import。
- **verifier 输入边界（F 系列要求）**：verifier 只接收**原始标量/引用**——
  held-out spec、scientist 已承诺的 predicted 值与 tolerance、执行日志条
  目——绝不接收 scientist 的对象/API（否则可经对象图间接读答案）。
- 验证方法（physics-as-oracle）：verifier 用同一 spec 在**全新 Laboratory**
  中重跑 held-out 实验，得到独立观测；断言 `|独立观测 − scientist 已承诺
  的 predicted| ≤ scientist 已承诺的 tolerance`（tolerance 本身经 F3 规则
  约束，见下）。附加一致性检查（harness 断言，永不回流）。

## E. 成功标准（离散、可审计的事实条件——全部同时成立）

1. hidden truth 不可达（F1/F2 审计断言 A-1..A-4 全过）；
2. measurement-only boundary 成立（scientist 可达面 = 门面 + 观测）；
3. 目标 relation 非预注册（初始 KB 中 `RelationRecord` 为空，见 F5/F6）；
4. relation scope 非回溯构造（scope 条件全部在声明前被观测；声明后未修改）；
5. scientist 自主形成可检验预测（存在引用其自注册 model 的
   PredictionRecord）；
6. prediction 在 commitment 后才执行（F4 执行日志：commitment_time <
   execution_time）；
7. tolerance rule 在 held-out 前固定（F3：rule fixed < held-out execute）；
8. 至少一个 training evidence confirmed；
9. 至少一个 held-out evidence confirmed，且其承诺晚于 relation 声明、
   tolerance 经 F3 规则产生；
10. `discoveries()` 出现该 relation（纯派生，非标记）；
11. held-out 不泄漏（trial 条件 ∉ scope；未用未观测条件）；
12. discovery 完全由 ledger-derived evidence 产生（无任何 discovery 写入）；
13. belief path 零调用（F8 运行审计）；
14. independent verifier 独立运行（进程/接口隔离，primitives-only 输入）。

## F. 失败标准（任一即失败）

- scientist 读取 truth（`truth_summary` 或等价物）；
- scientist 获取 Universe 对象 / Session / world configuration / physics
  or material constants；
- target relation 或其参数被预注入（声明早于其 scope 条件的观测，或
  初始 KB 非空）；
- scope 使用未来 observation（包含声明时未观测的条件）或声明后被回溯
  修改；
- held-out commitment 早于 relation declaration；
- prediction commitment 晚于 experiment execution（执行日志审计失败）；
- tolerance 根据 held-out outcome 调整（规则未先固定，或新 trial 的
  tolerance 偏离 pre-fixed 规则）；
- agenda / question selector 读取 ground truth；
- discovery 被直接写入（任何非派生的 discovery 状态出现）；
- 人工指定关键 hypothesis（注册/声明由 harness 代做）；
- belief path 被调用（ScientistState / form_prediction /
  prediction_from_belief / DEFAULT_PRIOR_BOUNDS 任一）。

## G. 边界

本 spec 只设计。禁止：修改 production code、修改 schema、新增 Open
Genesis entity、新增 planner、新增 LLM、创建 world mutation API、
commit / push。实施需另行授权（唯一新增 production 接线 = G-1 冷启动
组合器；harness 侧组件 = 门面 + 执行日志 + verifier + 审计断言，均须先
过 design review，见 Implementation Gate）。

## H. Scientist measurement-only boundary（F1/F2）

Pilot 中 scientist 与世界的全部交互仅限：

```
run_experiment(spec) -> ObservationRecord（t/z 等测量通道）
```

truth / Universe / configuration 属于 harness/world side；scientist-visible
side 只获得实验测量结果。boundary 以审计断言 A-1..A-4 表达（可执行检查），
不是代码约定。

## I. Tolerance generation rule（F3）

- 每个模型必须在**进入 held-out 阶段之前**确定其 tolerance generation
  rule；规则由 scientist 声明并由 harness 日志记录（rule fixed 时刻）。
- 规则只能使用 held-out observation 出现之前已经存在的 scientist-visible
  information（training 观测、已声明模型/关系、apparatus 词汇）；
  **held-out observation 不得参与 tolerance 的确定**。
- 时序不变量：`rule fixed < held-out execute`（F4 执行日志审计）。
- `tolerance = deterministic application of pre-held-out information`——
  同一规则对同一 pre-held-out 信息产生同一 tolerance。
- 不得允许 scientist 看到 held-out outcome 后扩大 tolerance：任何后续
  trial 的 tolerance 仍须由同一 pre-fixed 规则产生；偏离即失败判据。
- facts-not-scores：规则与 tolerance 均不引入 score / confidence /
  probability / ranking / winner。

## J. Harness Execution Log（F4）

- harness 维护 scientist **无法写入**的 execution log，至少记录：
  experiment/spec identifier、execution event、monotonic
  timestamp/sequence。
- Pilot audit 必须能证明：对每一个 held-out confirmed pair，
  `commitment_time < execution_time`。
- **不修改 production ledger schema**——execution log 是 harness 侧工件，
  与 ledger 并存。

## K. Relation scope 非回溯构造（F5/F6）

原 E-1 判据（"relation 声明晚于首条 training verification"）**废除**——
它与实际架构冲突（training verification 天然发生在声明之后）。

新的"目标 relation 未预注册"判据：

1. 初始 KnowledgeBase 中 `RelationRecord` 为空；
2. target RelationRecord 必须由 scientist 自己声明；
3. relation 声明时，其 `scope` 中的所有条件已经在声明之前被 scientist
   观察（harness 观测日志为准）；
4. `scope` 不得包含尚未观察的条件；
5. relation 声明之后，才允许针对该 relation 形成 held-out commitment；
6. 任何 held-out commitment 必须晚于 relation declaration；
7. relation scope 不得根据后续 observation 回溯修改。

**特别注意**：scope 条件"已经被观察"≠"relation 已经被验证"。Pilot 允许
且要求：

```
observation → candidate relation → prediction → verification
```

而不是：

```
observation → relation verification → declare relation
```

后者会把实验目标泄漏进 relation construction。

## L. G-1 purity boundary（F7/F10）

G-1 限定为：`observation → neutral fit → register model → declare
relation`。

**Allowed**：

- 通用 polynomial regression（任意次数）；
- generic mathematical fitting（对任意 (回归元, 响应) 观测对，含
  (x, z_first)）；
- `register_model`（formula/parameter 声明来自拟合）；
- generic closed mathematical parameter families（如 k·x、k·x²——人类设计
  的通用参数族，属声明能力）；
- `define_concept` / `declare_relation`；
- scope = already observed conditions。

**Forbidden**：

- `fit_free_fall` / `_fit_free_fall` / `prediction_from_belief` /
  `ScientistState` / `DEFAULT_PRIOR_BOUNDS`；
- `gravity` / `free fall` / `a -> -g/2` 及任何等价解释出现在 G-1 接线中；
- target formula / target parameter / target relation / target concept；
- 从 Universe / truth_summary / world config 读取答案。

**特别明确**：G-1 **不得把 polynomial coefficient 解释成 gravity**（包括
`a → -g/2` 与任何等价隐式物理解释）。Polynomial fitting 可以发现数学关系；
物理语义必须来自 scientist 后续可审计的科学推理链，而不是 G-1 内置。

## M. Belief path 零调用（F8）

Pilot run 中以下符号**零调用**（运行审计断言）：

- `ScientistState`
- `agent.form_prediction`
- `prediction_from_belief`
- `DEFAULT_PRIOR_BOUNDS`

这不是要求删除这些 production API——只是本 Pilot 禁止走这条路径；违例即
Pilot 失败。

## N. Generic Capability vs Domain Knowledge

**允许（generic scientific/mathematical capability）**：

- 通用回归与多项式表达式；
- 参数拟合与基本数学运算；
- 通用数据结构；
- 已声明的 apparatus measurement vocabulary（drop/height/z/t/dt-可测量）；
- 封闭的通用参数族注册表（k·x、k·x²）。

以上不属于 world-specific prior knowledge。

**Forbidden（domain-specific knowledge）**：

- `gravity` / `free fall` / `g` / `a = -g/2`；
- 任何等价的目标解释（包括对多项式系数的物理解释）。

审计形式：G-1 接线的 import/代码文本扫描 + 初始 KB 空快照 + 运行审计。

## O. Implementation Gate

**本 spec 修订完成后，仍然不能实现 G-1。**

下一阶段必须先**单独设计**并过 design review：

1. measurement-only facade；
2. execution log；
3. independent verifier；
4. audit assertions（A-1..A-4 + F3/F4/F5/F8 时序与零调用断言）。

只有这四项完成 design review 之后，才允许 G-1 implementation。

## P. 七项对照（修订后）

1. **AS v0 已被实际证明**：四扫描器议程、问题注册纪律、anomaly 优先、
   单步 cycle 驱动 held-out trial、事实分离、Discovery 派生（均在
   人工铺底的 store 上）。
2. **Pilot 要证明的新能力**：空知识冷启动——从纯观察出发、无种子数据、
   无人供目标，走完同一闭环并产生可独立验证的发现。
3. **已具备**：中性拟合后端、全部门口与闸口、四扫描器、agenda/cycle、
   HP、Discovery、结构边界（AST 级）。
4. **必须新增的最小能力**：G-1 冷启动组合器（纯度边界见 L 节）；harness
   四组件（门面/执行日志/verifier/审计断言——先 design review）。无新
   physics、无新 schema。
5. **泄漏路径**：truth_summary（F1 屏蔽）、Universe/Session（F2 排除）、
   DEFAULT_PRIOR_BOUNDS/信念路径（F8 零调用）、fit_free_fall 域推断
   （F7 排除）、条件池（运行前发布）、错误消息宇宙名（记录在案）。
6. **独立验证方法**：physics-as-oracle 重跑 + primitives-only verifier
   + scientist 自承 tolerance（经 I 节规则约束）。
7. **成功/失败判据**：E / F（修订后）。
