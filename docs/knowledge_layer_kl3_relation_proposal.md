# KL-3 Relation Layer Proposal

Status: PROPOSAL (auto-approved for immediate implementation per long-task
authorization). Baseline: `068738c` (KL-2, schema v5).

---

## 0. 立场

**RelationRecord 是声明实体，不是评价实体**：它声明"一个候选的一般化关系"——
主题概念、公式形式、实例化参数的来源模型、以及**训练 scope**。它的全部科学
内容来自 `RelationEvidence`——一个**纯派生视图**，把该模型在 ledger 中的验证
事实按 scope 切成"局部确认（training）"与"泛化证据（held-out）"。永不持久化、
永不写入 schema、永不复制 VerificationRecord。

## 1. 代码核查结论

- battery 词汇 = 实验 id（`ExperimentSpec.id`，经 `format_spec_refs`/`parse_spec_refs`）。
- **scope 概念尚不存在**——需最小新增：`RelationRecord.scope: tuple[str, ...]`
  （实验 id 元组，与 spec_ref 同词汇）。
- 派生基底可完全复用：`evidence_for_model(parameters_ref).independent_evidence`
  已给出"每 (prediction_id, experiment_id) 对一个代表 VerificationRecord"的
  确定性序列（P2-9 语义）——scope 切分只需按 `experiment_id ∈ scope` 二分。
- confirmed/refuted 唯一来源 = VerificationRecord.status（投影，不复制 residual/
  evidence 文本）。

## 2. RelationRecord（声明字段均标注来源）

| 字段 | 来源 | 说明 |
|---|---|---|
| `relation_id` | bookkeeping | `"rel-NNNN"`（`_next_ordinal`，rel- 前缀） |
| `subject` | 声明 | 被解释量的 concept_id（validate_concept_id 格式校验；不要求已注册定义——定义关联属后续） |
| `formula` | 声明 | 候选一般化形式（人类可读） |
| `parameters_ref` | 声明 | 实例化模型的 ModelRecord.model_id（**声明时必须存在且 status=registered**；模型日后被 supersede 不回溯影响关系——evidence 从 ledger 继续） |
| `scope` | 声明 | 训练 battery（非空、无重复、条目非空的实验 id 元组）——**training / held-out 边界的唯一来源** |
| `declared_by` | 声明 | provenance |
| `created_at` / `status` / `supersedes` / `content_hash` | bookkeeping | status ∈ {candidate, superseded}（**无 established**）；hash 锚定五个声明字段 |

状态转换：`candidate → superseded`（仅经原子 `declare_relation(..., supersedes=)`）；
无其他转换。与 DefinitionRecord 的有意差异：**关系允许多个候选并存**
（对抗性候选是科学竞争的本体），因此无"每概念至多一个 active"不变量；
supersede 仅用于"同一关系的修订"——校验 old 为 candidate 且
`subject`/`parameters_ref` 不变（formula/scope 可变）。

## 3. RelationEvidence（纯派生，永不持久化）

```python
@dataclass(frozen=True)
class RelationPair:                    # 投影：id 引用 + 唯一投影事实 status
    prediction_id: str
    experiment_id: str
    verification_id: str
    status: str                        # 来自 VerificationRecord.status

@dataclass(frozen=True)
class RelationEvidence:
    relation_id: str
    parameters_ref: str
    scope: tuple[str, ...]
    training_pairs: tuple[RelationPair, ...]     # experiment_id ∈ scope
    heldout_pairs: tuple[RelationPair, ...]      # experiment_id ∉ scope
    training_confirmed_count: int
    training_refuted_count: int
    heldout_confirmed_count: int
    heldout_refuted_count: int
    distinct_training_conditions: tuple[str, ...]  # sorted
    distinct_heldout_conditions: tuple[str, ...]   # sorted
    # 均为事实计数/集合；无 score/ranking/winner/confidence/established/quality
```

派生逻辑：`relation_evidence(relation, evidence: EvidenceSummary)`——对
`evidence.independent_evidence`（确定性顺序的代表验证序列）按
`experiment_id ∈ relation.scope` 二分为 RelationPair，聚合计数与条件集合。
输入是声明 + ledger 派生摘要；无写入、无随机、重复查询逐字节相等。
KnowledgeBase 仅加薄包装 `relation_evidence(relation)` =
`relation_evidence(relation, self.evidence_for_model(relation.parameters_ref))`。

重复验证纪律：与 EvidenceSummary 一致——每 (prediction, experiment) 对取
verification-id 序的第一个为代表；完整历史仍在 ledger。

## 4. Schema v6

加性：`relations` 数组 + load `.get` 默认 + save asdict；v5 文件加载为空注册表；
loader 拒绝更新版本（KL-2 已实现，天然覆盖）。PredictionRecord schema 不动。

## 5. 测试（不变量优先）

1. 字段/状态 pin（RelationRecord + RelationEvidence 禁止评价字段）
2. 声明冻结 + 逐字段篡改检测 + round-trip（v6）
3. scope/subject/parameters_ref 校验拒绝矩阵
4. 原子 supersede 链 + 拒绝矩阵（跨 subject / 跨模型 / 非 candidate）
5. **无验证则无证据**：未验证的预测对 RelationEvidence 零贡献
6. **纯派生 pin**：派生不写库（state/save 不变）、两次派生相等
7. **核心能力**：quadratic 在 scope 内（x=1）confirmed、scope 外（x=5）refuted
   → RelationEvidence 如实呈现"局部确认、泛化被否"
8. held-out 分类：in/out 切分、计数、distinct 条件排序
9. orphan（未注册 parameters_ref / 无匹配预测）→ 空证据
10. 重复验证代表纪律（与 EvidenceSummary 一致）
11. v5→v6 兼容 + loader 拒绝新版（复用 KL-2 机制）

## 6. 不做

RelationRecord 的 established 判定、held-out 条件的自主选择、subject→
DefinitionRecord 强制关联、LawRecord 迁移、KL-4、Autonomy。
