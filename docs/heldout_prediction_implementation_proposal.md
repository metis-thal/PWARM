# Held-out Prediction Implementation Proposal

Status: PROPOSAL (auto-approved for immediate implementation per HP long-task
authorization). Baseline: `068738c` + uncommitted KL-3/KL-4. Companion to
`knowledge_layer_heldout_prediction_proposal.md` (architecture rationale).

---

## 0. 事实核查报告（第一阶段）

1. **held-out condition 可表达**：`proposal_to_spec(proposal, kind, binding) →
   ExperimentSpec.id` 即条件身份（spec 派生确定性字符串），与
   `RelationRecord.scope` 同词汇。
2. **agent 调用链**：`propose_discriminating_experiment`（排序选择）→
   `commit_discriminating_predictions`（proposal_to_spec + model_prediction +
   commit_predictions）→ `execute_proposal`（proposal_to_spec + lab.run）→
   `verify_competing_predictions`（comparison_input + reduction +
   verify_prediction + record_verification）。held-out 闭环完整复用此链，
   仅替换"选择"环节。
3. **RelationRecord/RelationEvidence 消费**：测试中 53 处引用，派生纪律成熟；
   选择阶段消费 `relation.scope`（声明边界）+
   `evidence_for_model(parameters_ref).experiment_ids`（已测条件）。
4. **旧测试假设**：单模型 propose 返回 None（零分歧）为已知行为，测试用
   手工 ConditionComparison 绕过（既有模式）；truth-free AST 扫描覆盖
   agent_module，新代码必须保持 universe-free。无阻塞。
5. **schema**：**不需要**。held-out 是关系对实验的补集分类（纯派生），
   PredictionRecord/VerificationRecord/schema v6 全部不动。

## 1. 八个设计问题

1. **held-out condition 的身份来源**：`ExperimentSpec.id`——与 relation.scope
   条目同一词汇；候选条件经 `proposal_to_spec` 翻译后取 id。
2. **边界在哪里**：`relation.scope` 是唯一声明来源；held-out =
   `spec.id ∉ scope` **且** `spec.id ∉ 已测集合`
   （`evidence_for_model(parameters_ref).experiment_ids`）——每次提出的是
   新的 held-out 试验。
3. **谁生成候选条件**：调用方构造条件池（复用既有 designer 词汇）；条件
   生成本身留给 Autonomy。方法只过滤与选择。
4. **AI 如何选择**：调用方顺序取第一个合格条件——确定性、无偏差；**不使用
   分歧排序**（授权约束 4）。
5. **如何证明选择不使用实验结果**：三重证据——(a) 方法签名只接受
   (models, relation, pool, kind, binding)，无任何 record/observation 参数；
   (b) 提案阶段零写入（无 commit、无 lab 调用、JSON 逐字节不变——测试钉死）；
   (c) 选择在物理运行之前（时间性测试钉死），未来结果在结构上不可见。
6. **失败实验如何入账**：承诺不可变（→ refuted）+ VerificationRecord 追加 +
   RelationEvidence 自动 `heldout_refuted_count +1`；无状态变化、无淘汰。
7. **旧路径兼容**：零修改既有方法（propose/commit/execute/verify 原样）；
   新增两个 agent 方法 + 一个瞬态结果 dataclass；无 schema 变更；旧测试零改动。
8. **最小切片**：HP-1 `propose_held_out_condition`（过滤器+确定性选择）→
   HP-2 `run_held_out_trial` 闭环 + `HeldOutTrial` 瞬态结果 → HP-3 文档。

## 2. 实现要点

- `propose_held_out_condition(models, relation, candidate_conditions, kind,
  binding=None) -> ConditionComparison | None`：按调用方序逐个翻译
  proposal→spec，排除 scope 成员与已测 id，返回首个合格条件的
  ConditionComparison（predictions 按模型预填充——选择发生在计算之前）。
- `run_held_out_trial(models, relation, candidate_conditions, kind, binding,
  tolerances, output_binding, output, reduction) -> HeldOutTrial | None`：
  校验 `relation.parameters_ref ∈ models`（否则 verdict 永不进入该关系的
  evidence——拒绝）→ propose → commit（先于执行）→ execute → verify →
  返回 `HeldOutTrial(relation_id, experiment_id, committed, verifications)`
  （瞬态句柄，事实全在 ledger）。
- 容忍参数：tolerances/output_binding/output/reduction 透传给既有 commit/
  verify（缺失由既有 loud 校验拒绝，不新增默认值）。
