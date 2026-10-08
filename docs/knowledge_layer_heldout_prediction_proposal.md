# Held-out Prediction Architecture Proposal

Status: PROPOSAL ONLY — no code changed. Baseline: `068738c` + uncommitted
KL-3/KL-4 in the worktree (Knowledge Layer v0 complete: KL-1..KL-4).

---

## 0. 立场

**held-out 是实验设计边界，不是模型评分**。Held-out Prediction 不给任何实体
打分、不淘汰任何模型、不改变任何状态——它把"在训练边界之外做一次诚实的
预测-承诺-验证"变成 AI 可执行的纪律闭环。全部产出是 ledger 事实；
一切评价性解读留给 RelationEvidence 的消费者。

## 1. 代码事实核查结论（五点）

| # | 核查点 | 结论 |
|---|---|---|
| 1 | prediction/experiment/spec_ref 能否表达 held-out 条件 | **能**。`proposal_to_spec(proposal, kind, binding) → ExperimentSpec → spec.id` 即条件身份；任意数值条件可表达，无需新字段 |
| 2 | battery/spec_ref 能否区分 training/held-out | **验证侧已能**（KL-3 的 `experiment_id ∈ relation.scope` 二分）；**提案侧没有**——新增一个设计时派生（条件 → spec.id → scope 成员 + 已测检查），零 schema 变更 |
| 3 | RelationEvidence 能否作为消费基础 | **能，且自动更新**：任何 held-out 验证落账后 `relation_evidence(relation)` 立即包含该 pair（纯派生）。选择基础另需 `evidence_for_model().experiment_ids`（已测条件，已存在） |
| 4 | 下一实验选择入口 | 三个已有：mission planner / `propose_discriminating_experiment`（**调用方供池**，按分歧排序）/ `designer.choose`（value 排序）。held-out 采用"调用方供池"模式组合，但**不复用分歧排序**（见 Q5） |
| 5 | 旁路 | **无**。ledger 写入口唯一（`commit_prediction` / `record_verification`，均过 hash 闸口）。`execute_proposal` 可无承诺运行（合法探索），verdict 仍被闸口保护 |

## 2. 八个设计问题

### Q1 解决哪个缺口？

KL-3 提供了 local/held-out 的**事实视图**，但没有任何机制**驱动** AI 主动产生
held-out 试验——泛化证据目前只能"碰巧"积累（AI 恰好试了新条件时）。缺口 =
**选择（排除 scope 内与已测条件）→ 承诺（先于实验）→ 执行 → 验证** 的闭环
作为一等纪律。完成后，RelationEvidence 中的 held-out 事实从"偶然"变为
"纪律性产生"。

### Q2 training scope 与 held-out scope 如何表达？

- training scope：`RelationRecord.scope`（已声明，冻结，进哈希）。
- held-out scope：**不声明**——它是补集。判定是纯派生：
  `candidate 条件 → proposal_to_spec → spec.id`，`spec.id ∉ relation.scope`
  即 held-out；再排除该模型**已测**条件
  （`evidence_for_model(parameters_ref).experiment_ids`），保证每次提出的是
  **新**的 held-out 条件。零 schema 变更。

### Q3 谁负责提出新的 held-out 条件？

**分层**：条件池（candidate pool）由调用方构造——复用既有 designer/planner
词汇（drop heights 等），不发明新的条件生成器（那是 Autonomy 阶段的能力）；
**选择**由新的 agent 方法 `propose_held_out_condition(models, relation,
candidate_conditions, kind, binding)` 执行——按调用方顺序取第一个满足
"∉ scope 且未测"的条件（确定性、无偏差），池耗尽返回诚实的 `None`。
纪律在过滤器里，自由在池构造里——且池的每次使用都留痕（见 Q5）。

### Q4 PredictionRecord 是否需要修改？

**不需要**。`commit_discriminating_predictions` 已能为任意条件提交带完整
验证契约与注册身份的预测；held-out 不是预测的属性，而是**关系对实验的
分类**（补集派生）。无新字段、无哈希变更、无迁移。

### Q5 如何避免"挑容易成功的测试条件"？

四层机制（诚实声明残余风险）：

1. **结构上无法以成败选条件**：选择只看 scope 成员与已测集合，不接触任何
   观测/真值——提案阶段物理未运行，无"成功"可言；
2. **确定性选择规则**：调用方顺序，不按"最可能正确"排序（分歧排序也弃用，
   避免任何设计偏差面）；
3. **失败不可隐藏**：每次尝试都是不可变承诺，REFUTED 永久入账；
   RelationEvidence 计入全部 held-out pair——AI 无法选择性呈现；
4. **残余风险 = 池构造偏差**（只挑靠近训练条件的简单点）：v0 通过
   lineage（承诺 + scope 全部可查）使其**可审计**而非可阻止；条件池构造
   政策是 Autonomy 阶段的实验设计问题，显式推迟。

### Q6 held-out 失败后产生什么事实？

- 承诺保留（不可变，status → refuted）+ 新 `VerificationRecord`（residual/status）；
- `RelationEvidence` 自动：`heldout_refuted_count +1`、pair 进入
  `heldout_pairs`、条件进入 `distinct_heldout_conditions`；
- **无状态变化**：关系不自动退役（无 established/refuted 关系状态）、模型
  不被淘汰、信念路径的认知更新（Phase-2 Step 2）仅限信念路径。失败是事实
  输入，不是触发器。

### Q7 如何与 RelationEvidence / ModelLineage 连接？

零接线成本：trial 的承诺与验证落账后，
`knowledge.relation_evidence(relation)` 立即包含新的 held-out pair
（KL-3 纯派生）；
`knowledge.model_lineage(parameters_ref)` 按承诺序列出该 held-out 预测及其
验证。选择阶段消费 `evidence_for_model().experiment_ids`（已测）与
`relation.scope`（边界）——**全程只消费既有派生，无新状态**。

### Q8 最小实施切片？

| 切片 | 内容 | 测试锚点 |
|---|---|---|
| **HP-1** | `agent.propose_held_out_condition(models, relation, candidate_conditions, kind, binding)`：按调用方序返回第一个"∉ scope 且未测"的条件（ConditionComparison，复用现有载体）；池耗尽 → None；提案阶段零写入 | in-scope 排除 / 已测排除 / 确定性顺序 / 池耗尽诚实 None / 提案零写入（无 commit、无 lab 调用） |
| **HP-2** | `agent.run_held_out_trial(models, relation, candidate_conditions, kind, binding, tolerances, output_binding, output, reduction)`：propose → commit（先于执行，时间性 pin）→ execute → verify；返回 `HeldOutTrial(experiment_id, committed, verifications)`（瞬态结果，不持久化）；**校验 relation.parameters_ref 必须在 models 中**（否则证据永不链接——拒绝）；全部 in-scope 池 → None 且零承诺 | 时间性 pin / 关系-模型绑定校验 / 端到端：quadratic scope x=1 → trial x=5 → RelationEvidence 呈现"局部 confirmed、held-out refuted" / 池耗尽零承诺 / HeldOutTrial 结构 pin |
| **HP-3** | 文档收尾（v0 实现状态表 + ARCHITECTURE 行） | — |

schema 不变（v6）；PredictionRecord 不变；新代码全部 agent.py + 瞬态结果
dataclass；全部 AI-side（truth-free 检查覆盖不变）。

## 3. 不做

条件池的自主生成（Autonomy）、近 scope 条件的距离政策、关系/模型的自动
退役、belief 路径的 held-out（v0 竞争路径）、KL-4 之后的新查询。
