# KL-4 Lineage Queries Proposal

Status: PROPOSAL (auto-approved for immediate implementation per long-task
authorization). Baseline: `068738c` (+ uncommitted KL-3 in the worktree).

---

## 0. 目标

完成 Knowledge Layer v0 收尾：把 model→predictions→verifications 与
concept→predictions 两条引用链升级为一等查询。**Schema 不动（v6 保持）**——
lineage 是纯派生，零新记录、零新存储、零缓存。

## 1. 代码核查结论

- 链的存储已齐备（model_records / predictions / verifications），但无一等查询：
  `evidence_for_model` 只给验证聚合，predictions 本体与按预测分组不可直达。
- claim 侧（DefinitionRecord.concept_id → predictions）同样无查询。
- 无任何可复用的 lineage 结构 → 新增最小派生视图。

## 2. 交付物

| 交付 | 位置 | 性质 |
|---|---|---|
| `ModelLineage` | knowledge_records.py | 派生视图 dataclass（引用现有记录，零拷贝内容——predictions/verifications 是**同一对象的引用**，不复制字段） |
| `KnowledgeBase.model_lineage(model_id)` | knowledge.py | 纯派生包装 |
| `KnowledgeBase.predictions_for_model(model_id)` | knowledge.py | 精确匹配查询（KL-1 exact-match 纪律），seq 承诺序 |
| `KnowledgeBase.predictions_for_concept(concept_id)` | knowledge.py | claim 侧解析（DefinitionRecord.concept_id → predictions），seq 序 |
| Minor-a 收尾 | knowledge.py | KB.relation_evidence docstring 写明纯委托 |
| 文档收尾 | v0 提案文档 + ARCHITECTURE | "实现状态（KL-1–KL-4）"节 + 模块行更新 |

## 3. 字段来源

```python
@dataclass(frozen=True)
class ModelLineage:
    model_record: ModelRecord | None   # 引用（None = 从未注册的 id，legacy 字符串容忍）
    predictions: tuple[PredictionRecord, ...]   # 引用，seq 承诺序
    verifications: dict[str, tuple[VerificationRecord, ...]]
    # 引用，按 prediction_id 分组、verification_id 序
```

无任何新事实字段；standing 继续由 `competition_state(model_id)` 提供
（lineage = 结构链，standing = 裁决聚合，二者不重复）。

## 4. 不变量（测试优先覆盖）

1. 纯派生：调用后 store 的 JSON 逐字节不变；重复查询相等。
2. 精确匹配：legacy 字符串的 model_ref 永不并入注册 id 的 lineage（模型记录
   为 None 但 predictions 仍按存储键列出——KL-1 政策延续）。
3. seq 承诺序 / verification_id 序确定性。
4. 空情形：未注册 id、无预测、无验证 → 结构完整且为空。
5. 无新增评价字段（结构 pin 延续到 ModelLineage）。

## 5. 不做

新 schema/记录种类、缓存、established、KL-4 之后的一切（Held-out /
Autonomy 另行授权）。
