# Knowledge Layer v0 Architecture Proposal

Status: PROPOSAL ONLY — no code changed. Written 2026-09-28 after the
Genesis Core completion (P0-1 → P2-9, HEAD `3cb05a2`). Based on a full read
of the five structures below in their current (post-P2-9) form.

---

## 0. 目标与不做什么

**目标**：把 AI 的知识从「字符串键控的参数记忆」升级为「可寻址、有 lineage、
可派生的实体系统」，为 Autonomous Scientist 的 "What should I investigate
next?" 提供知识基座。

**v0 明确不做**：

- 不引入 score / ranking / winner / probability / elimination（任何形式）
- 不做模型存活判断（establishment 判定是后续阶段的显式 op，v0 只积累事实）
- 不自动迁移 legacy LawRecord（会伪造 lineage）
- 不动 physics / universe 层（全部 AI-side，过 truth-free 检查）
- 不复制 PredictionRecord 的任何功能（见 §3 边界表）

## 1. 现状审计（五个结构，post-P2-9 实测）

| 结构 | 现状 | 与 Knowledge Layer 的关系 |
|---|---|---|
| `PredictionRecord`（16 字段） | 承诺事件：哈希锚定的科学内容 + 验证契约 + `spec_ref` 实验身份 + `supersedes` | **保持原样**。它是"预注册"层，Knowledge Layer 只引用不复制 |
| `VerificationRecord`（8 字段） | 裁决事件：`(prediction_id, experiment_id)` + residual/status + `created_at` | **保持原样**。一切验证产物只存在于此 |
| `EvidenceSummary` | 纯派生：按 `model_ref` 字符串聚合 independent evidence | 被复用；缺口见下 |
| `CompetitionState` | 纯派生：per-model 事实 + `last_verification_at` | 被复用；它就是 ModelRecord 的 standing 视图 |
| `KnowledgeBase` | laws（`LawRecord`：含 **confidence 分数**）+ predictions + verifications，schema v3 | 新增三类知识实体 + 持久化；laws 冻结为 legacy |

**实测发现的三个真实缺口**：

1. **模型身份不可解析**：`PredictionRecord.model_ref` 是自由字符串——竞争路径
   写 `"model linear"`（`models.py:94` 的 `source=f"model {model_id}"`），信念路径写
   `"state belief (unknown)"`。没有任何持久化的模型实体可被解析；换个会话后
   "这个模型是谁、参数是什么、从哪来"全部丢失。
2. **local confirmation 与 generalizable relation 不可区分**：ledger 只知道
   (prediction, experiment) 对；没有任何结构声明"这个关系是在哪些条件下训练的"，
   因此无法派生"哪些验证是 held-out 的"——Phase 2 的 held-out discipline 没有落点。
3. **知识层有分数**：`LawRecord.confidence` 是分数（mission 层的 `knows()` 门槛），
   与 facts-not-scores 教义冲突；且 LawRecord 与验证记录之间**零 lineage**
   （只有 experiments id 列表，没有 prediction/verification 指针）。

## 2. 设计原则（facts-not-scores 在知识层的具体化）

1. **知识记录不存储任何裁决结果**。验证产物（observed/residual/status）只存在
   ledger；知识层对验证内容的唯一参与方式是**引用 id + 读取时派生**。
2. **声明（declaration）与派生（derivation）严格分层**。存储字段 = 声明事实 +
   bookkeeping；一切"这个模型/关系表现如何"都是读取时的纯派生函数（与
   `EvidenceSummary`/`CompetitionState` 同一模式——已有先例，不新发明）。
3. **历史事实不可变**。知识实体一经创建，声明字段冻结（内容哈希锚定，复用
   commitment_hash 模式）；修正是**新版本 + `supersedes` 回指**（复用 P2-6 模式），
   绝不原地改写。
4. **状态转移是显式 op，不是计算结果**。知识实体的 status 只能通过带 provenance
   的持久化操作改变（如 supersede/retire），不存在"分数低于阈值→淘汰"类路径。
5. **lineage 一律按引用（id pointer），零拷贝**。

## 3. 与 PredictionRecord 的功能边界（要求 1）

| 职责 | 归属 | 知识层是否复制 |
|---|---|---|
| 预注册 / 预测值 / tolerance / 验证契约 / 哈希锚定承诺 | `PredictionRecord` | ❌ 不复制（知识记录哈希的是**声明内容**，不是预测值） |
| 裁决（residual / confirmed / refuted） | `verify_prediction` + `VerificationRecord` | ❌ 知识层无裁决字段 |
| 身份完整性（spec_ref battery 检查） | `record_verification` | ❌ 知识层不写 VerificationRecord |
| 实验证据聚合 | `evidence_for_model` / `competition_state` | ♻️ 直接复用（ModelRecord 的 standing 视图 = 现有函数，仅键改为已注册 model_id） |
| 撤回/版本化（supersedes 链） | P2-6 模式 | ♻️ 模式复用（op 语义，非新机制） |

## 4. 三个新实体

所有实体放入新模块 `pwarm/scientist/knowledge_records.py`（ledger 事件留在
`records.py`；知识实体是另一类对象），持久化在 `knowledge.py`（schema v4）。

### 4.1 ModelRecord — 可解析的模型身份

```python
@dataclass(frozen=True)
class ModelRecord:
    model_id: str                    # 可解析身份（替代裸字符串 model_ref）
    formula: str                     # 人类可读形式（声明；v0 对应 _FORMULAS 词汇）
    params: dict[str, float]         # 声明参数，创建即冻结
    derived_from: str                # 声明 provenance：产线（如实验 id / hypothesis source）
    created_at: str = ""
    status: str = "candidate"        # candidate | active | retired | superseded（仅显式 op）
    supersedes: str = ""
    content_hash: str = ""           # 锚定 model_id/formula/params/derived_from
```

- **来自事实（声明）**：model_id / formula / params / derived_from。
- **必须由验证产生**：**无**——任何存储字段都不来自验证；模型的 standing 完全
  是派生（见下）。这是要求 5 的直接回答：验证产生的数字一个都不进知识记录。
- **standing 视图（派生，不存储）**：`competition_state(model_id)` 原样复用。
  status 的 `candidate → active` 与 `retire()` 是显式 op；v0 无自动转移。
- **hash 策略**：复用 `commitment_hash`（sorted-keys JSON）；status/supersedes/
  created_at 在哈希外（与 PredictionRecord 的 bookkeeping 策略一致）。

### 4.2 DefinitionRecord — 可操作的概念定义

```python
@dataclass(frozen=True)
class DefinitionRecord:
    concept_id: str                  # claim 词汇："gravity"（claim 字符串的正式化）
    kind: str = "measurand"          # v0 仅 measurand；后续 derived-quantity / object-class
    unit: str = ""
    channel: str = ""                # 操作定义：测量通道
    reduction_rule: str = ""         # 操作定义：归约规则
    description: str = ""
    created_at: str = ""
    status: str = "active"           # active | superseded（定义被修正，不是被驳倒）
    supersedes: str = ""
    content_hash: str = ""           # 锚定 concept_id/kind/unit/channel/reduction_rule/description
```

- **关键约束**：`(channel, reduction_rule)` 必须通过现有已声明词汇的校验
  （`_OBSERVATION_FIELDS` / `_REDUCTION_RULES`）——定义是**操作性**的（如何测量），
  不是名义的。这把 `PredictionRecord` 的验证契约词汇和知识词汇首次接通。
- **来自事实（声明）**：全部存储字段。**必须由验证产生**：无。
- 定义不"被驳倒"，只被**修正**（v2 定义 supersede v1）；旧定义保留可查。

### 4.3 RelationRecord — 候选关系（局部确认 ≠ 一般规律）

```python
@dataclass(frozen=True)
class RelationRecord:
    relation_id: str
    subject: str                     # concept_id（被解释量，如 "z_first"）
    formula: str                     # 声明的一般化形式（如 "z_first = drop_height"）
    parameters_ref: str              # ModelRecord.model_id（参数实例化来源）
    scope: tuple[str, ...]           # 训练 battery：spec ids / 条件声明（决定 held-out 边界）
    declared_by: str = ""
    created_at: str = ""
    status: str = "candidate"        # candidate | retired | superseded（v0 无 established）
    supersedes: str = ""
    content_hash: str = ""           # 锚定 subject/formula/parameters_ref/scope/declared_by
```

- **来自事实（声明）**：subject / formula / parameters_ref / scope。
- **必须由验证产生**：**无存储字段**——但这是要求 5 的核心落点：关系的全部
  科学内容来自一个**派生视图**：

```python
@dataclass(frozen=True)
class RelationEvidence:                  # 纯派生，永不持久化
    training_pairs: tuple[...]           # scope 内 (prediction_id, experiment_id) + verdict
    heldout_pairs: tuple[...]            # scope 外 = held-out 试验 + verdict
    heldout_confirmed_count: int         # 事实计数（非分数：无加权、无阈值）
    heldout_refuted_count: int
    distinct_heldout_conditions: int
```

  `relation_evidence(relation, kb)` 按 `parameters_ref → model 的 predictions →
  各自 verifications`（复用 `evidence_for_model`）派生，按 `scope` 把证据切成
  **局部确认（in-scope）** 与 **泛化证据（held-out, out-of-scope）**。一个模型在
  训练条件下全确认 ≠ 关系成立——只有 held-out 区有事实，泛化才算有依据。
- **v0 无 established 状态**：standing 完全由 RelationEvidence 事实呈现；
  "何时可以称定律" 是后续阶段的显式判定 op（需人类可审的判据，不是隐藏阈值）。

## 5. Lineage：prediction → experiment → verification → knowledge（要求 6）

```
DefinitionRecord("gravity", unit=m/s², op=(z, free_fall_g))      ← 词汇层（声明）
        ▲ claim 引用
ScientificModel(params) ──register──▶ ModelRecord(model_id, params, derived_from)
        │                                   ▲ model_ref 引用（KL-1 规范化后）
        │ commit                                    │
        ▼                                           │
PredictionRecord ──spec_ref──▶ ExperimentSpec.id == ObservationRecord.experiment_id
        │                                            ▲ experiment_id 引用
        │ adjudicate（契约内嵌）                      │
        ▼                                            │
VerificationRecord ─────────────────────────────────┘
        │
        ▼  派生（只读）
EvidenceSummary / CompetitionState          ← ModelRecord 的 standing（事实）
        │
        ▼  派生（只读，按 scope 切分）
RelationEvidence：training vs held-out      ← RelationRecord 的证据（事实）
```

- 每条箭头都是 **id 引用**，无一处数据拷贝；任一 ledger 事实被否证，下游派生
  视图自动反映（不存在需要同步的缓存）。
- 历史不可变 + supersedes 版本链贯穿三类实体（与 PredictionRecord 的 P2-6
  语义一致）。

## 6. Schema evolution 策略（要求 7）

沿用已验证两次的加性模式（v1→v2 加记录种类；v2→v3 加契约字段）：

1. `SCHEMA_VERSION = 4`；`KnowledgeBase` 新增三个 dict：`model_records` /
   `relations` / `definitions`；save 用 `asdict` 数组，load 全部 `rec.get(default)`
   —— **旧文件（v1/v2/v3）加载不变，保存时升级**。
2. 不迁移 laws：`LawRecord` 原样保留、继续服务 mission 层（`knows()` 门槛），
   文档标注 legacy；新知识只进新实体。未来当 AI 重新导出同一定律时，自然会以
   DefinitionRecord + RelationRecord + lineage 的形式出现——不伪造历史。
3. 内容哈希复用 `commitment_hash`（同实现、不同 payload），无新哈希机制。
4. truth-free 检查同步：`knowledge_records.py` 加入 genesis 测试的
   `GENESIS_MODULES` 与 `test_api_contract.AI_MODULES`。

## 7. 最小实现路线（要求 8；每片 = 读现状 → 最小改 → 测试 → 全量回归）

| 切片 | 内容 | 测试锚点 |
|---|---|---|
| **KL-1** | `knowledge_records.py` + `ModelRecord` + KnowledgeBase 注册/supersede/retire ops + schema v4 持久化；`model_ref` 规范化（竞争路径 commit 用注册的 `model_id`，替代 `"model {id}"` 拼串——模型记录与 ledger 的接缝） | 注册/冻结/哈希 round-trip；ops 拒绝矩阵；standing 派生 == 既有 `competition_state(model_id)`；孤儿 model_ref 容忍（与 orphan 纪律一致）；no-score pin（结构中无判定数值字段） |
| **KL-2** | `DefinitionRecord` + `define_concept()`；`(channel, reduction_rule)` 对照现有注册表校验 | 未知 channel/rule 拒绝；supersede 链；哈希 round-trip；v3 文件加载不变 |
| **KL-3** | `RelationRecord` + `declare_relation()`（校验 parameters_ref 可解析、scope 非空）+ `relation_evidence()` 纯派生 | in-scope / held-out 切分正确且确定性；held-out 计数是事实不是分数；拒绝存储任何 verdict 字段（结构 pin） |
| **KL-4** | lineage 查询（model→predictions→verifications→relation）+ 文档同步 + `GENESIS_MODULES`/`AI_MODULES` 扩展 | v4 round-trip；全量回归；四问复核 |

**显式推迟**（记录触发条件）：established 判定 op（自主循环需要"下一步研究什么"
的判断时）、held-out 条件的自主选择（Autonomy 阶段）、Anomaly 挂钩（Anomaly
Engine）、概念发现（Concept Discovery）、LawRecord 退役。

## 8. 四问复核

1. **新能力**：知识从字符串记忆变为可寻址实体 + 可区分局部确认与泛化证据的
   派生视图——此前结构上不可能区分。
2. **无需人工答案**：全部声明/派生在 AI 侧，零 universe 访问（truth-free 检查覆盖）。
3. **最小可复现实验**：每切片的针对性测试 + 全量回归（315 测试为底）。
4. **是否做到以前做不到的事**：是——KL-3 的测试场景将证明：同一模型在 scope 内
   全 confirmed 而在 scope 外 refuted 时，RelationEvidence 如实呈现"局部成立、
   泛化被否"，而这一区分在现有结构中无法表达。若该测试不能证明此能力，切片不
   通过（不堆功能）。


## Realization status (as of KL-4, 2026-09-29)

| Slice | Delivered | Schema |
|---|---|---|
| KL-1 | `ModelRecord` + `register_model` / `supersede_model`; model_ref normalized to registered ids; exact-match compat | v4 |
| KL-2 | `DefinitionRecord` + `define_concept` (atomic revision, single active per concept) + `active_definition` / `definitions_for_procedure`; concept_id snake_case discipline; loader refuses newer files | v5 |
| KL-3 | `RelationRecord` (subject/formula/parameters_ref/scope) + `RelationEvidence` pure derivation (local vs held-out split, fact counts); rivals coexist; no established | v6 |
| KL-4 | `ModelLineage` + `model_lineage` / `predictions_for_model` / `predictions_for_concept` — pure reference resolution (model→predictions→verifications, concept→predictions); schema unchanged | v6 |
| HP | `propose_held_out_condition` + `run_held_out_trial` (agent) — the generalization-trial discipline: deterministic caller-order selection over the scope/tested complement, commit-before-execute, verdicts via the existing chain; held-out is a design boundary, not an evaluation; schema unchanged | v6 |
Lineage chains are id references end to end; every derivation is pure
(no writes, deterministic); no score/ranking/winner/confidence exists in
the layer. Next (separately authorized): held-out prediction, autonomy.
