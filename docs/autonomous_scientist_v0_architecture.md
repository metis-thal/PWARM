# Autonomous Scientist v0 Architecture

Status: DESIGN ONLY — no code changed. Baseline: `068738c` + uncommitted
KL-3/KL-4/HP worktree (Knowledge Layer v0 + Held-out Prediction complete).

---

## 0. 立场

**自主性来自"事实驱动的议程 + 确定性纪律"，不是来自一个会发散的头脑。**
v0 的 AI 自主性 = 它根据**自己知识库中的事实**生成研究问题（声明）、按
**声明的确定性优先级**选择、用**既有纪律闭环**执行——全程无可评分字段、
无真值访问、每一步留痕。人类不再提供 Mission，但提供了脚手架：扫描器、
优先级声明、执行闭环。

## 1. 事实核查（五点）

| # | 核查点 | 结论 |
|---|---|---|
| 1 | agent 能产生哪些动作 | 18 个公开方法，覆盖完整科学方法环：预测/承诺/验证（信念+竞争双路径）、held-out trial、mission 三种循环、知识发布。**但全部由人供 Mission 或人调方法驱动** |
| 2 | Knowledge Layer 只读事实 | 16+ 派生查询齐备：`competition_state`（模型 standing）/ `relation_evidence`（local vs held-out）/ `model_lineage` / `predictions_for_concept` / `active_definition` / `evidence_for_model`——**议程的全部输入已存在** |
| 3 | planner/proposal/execution 边界 | planner 被 mission 门控且仅 gravity（`mission.target != "gravity" → []`）；designer 按 expected gain 排序（Mission 002/03 既有机制）；AI↔physics 边界 = proposal→Laboratory→ObservationRecord，完好 |
| 4 | 缺失能力 | ① 无 question 概念（0 命中）；② 无 anomaly 概念（0 命中）——refuted 事实躺在 ledger 里无人扫描；③ 本体硬编码（`DEFAULT_PRIOR_BOUNDS` 预设 gravity/friction/...）；④ 无自驱动循环（CLI 硬编码 Mission 001–003）；⑤ 无"发现"的判定 |
| 5 | 绝不能复制 | PredictionRecord/VerificationRecord（承诺/裁决事件）、三类知识实体、四个派生视图、reduction/bindings 词汇表、六个 ledger 写闸口——**v0 只新增 QuestionRecord + 事实扫描器 + 循环组合** |

## 2. 八个设计问题

### Q1 "What should I investigate next?" 的输入是什么？

全部是**既有派生 + 新问题注册表**的事实形状输入（无评分）：

| 输入 | 来源 | 驱动的问题类 |
|---|---|---|
| 关系的 local/held-out 事实 | `relation_evidence`（KL-3） | 未测泛化（training 有、held-out 空） |
| 模型 standing 事实 | `competition_state`（P2-9） | 异常（refuted/conflicts 的模型） |
| 预测的 claim 词汇 | `predictions_for_concept`（KL-4） | 未定义概念（有 claim 无 active 定义） |
| 模型身份与验证 | `model_lineage`（KL-4） | 未验证身份（注册后零验证） |
| 能力边界 | designer.enabled_kinds / Laboratory.supported_experiments | 行动可行性过滤 |
| 预算 | ExperimentBudget（既有） | 可执行性过滤 |
| 问题注册表 | `QuestionRecord`（新增） | 已提出待研究的问题 |

### Q2 candidate question 如何产生？

**事实驱动的确定性扫描器**（纯函数：KB → 候选问题），不是生成式头脑：

| 扫描器（v0 固定集合） | 触发事实（全部可查） | 产生的行动（复用既有闭环） |
|---|---|---|
| `untested_generality` | 关系 training 非空 **且 held-out 为空**（RelationEvidence 事实） | `run_held_out_trial`（HP 闭环） |
| `anomaly` | 模型存在 refuted/conflicts 的验证事实 | 竞争性判别试验（commit→execute→verify） |
| `undefined_concept` | 某 claim 出现于预测但无 active DefinitionRecord（KL-4 查询） | `define_concept`（KL-2 原子修订） |
| `unverified_identity` | 注册模型零验证（model_lineage 事实） | 判别试验 |

扫描不发明、不臆测——它只把**已有事实的空缺**翻译成问题候选。AI 的自主
创造性边界 = 扫描器集合（v0 固定、可审计、可扩展）。

### Q3 hypothesis 如何表示？

**不新增实体**——hypothesis 在本架构中已是三个既有声明的总称：
- 参数假设 = `ScientificModel` + `ModelRecord`（身份+参数，KL-1）；
- 一般化假设 = `RelationRecord`（subject/formula/scope，KL-3）；
- 预测假设 = `PredictionRecord`（承诺+契约，P1-4）。
问题的"回答"动作就是产出这些既有声明（register/declare/commit）。
复制一个 hypothesis 实体将违反非复制纪律。

### Q4 experiment priority 是否允许？

**允许，但必须是声明性纪律，不是计算分**。两层分离：

- **问题选择**：确定性声明优先级（v0 固定序：anomaly → untested_generality →
  undefined_concept → unverified_identity，同层按 id 序）——结构性地把
  **反驳性事实排在最前**（见 Q6），无任何 utility/urgency 数值字段；
- **实验选择**（问题确定后选哪个条件/设计）：可复用既有 designer 的
  expected-information-gain 机制——那是 Mission 002/003 已接受的**设计内**
  排序（排的是实验设计的信息量，不是模型的优劣），与 held-out 选择弃用
  排序的理由不冲突（held-out 的排序偏差风险在于"挑易成功条件"，信息增益
  排序不构成该偏差面）。

边界一句话：**排序允许用于"哪个实验更有信息量"，禁止用于"哪个模型/问题
更重要"**。

### Q5 如何保持 facts-not-scores？

1. `QuestionRecord` 字段 pin：question_id / kind（扫描器名）/ target ids
   （指向触发事实的 id 引用）/ declared_by / created_at / status（open |
   withdrawn）/ supersedes / content_hash——**无 priority/urgency/importance**；
2. 议程输入全部是既有事实派生；
3. 每个循环决策留下 ledger 事实（问题声明、承诺、验证）——**审计轨迹即
   决策依据**，不存在需要额外记录的"决策分数"；
4. 结构 pin 测试禁止评价字段（延续 KL 全系纪律）。

### Q6 如何避免 confirmation bias？

1. **声明优先级把反驳放最前**：anomaly 扫描器居首——循环在结构性上必须
   先面对反驳事实，才能轮到寻找确认；
2. **held-out 从补集选**（HP 纪律）：循环不能重测训练条件冒充泛化；
3. **承诺先于实验**（P1-4 纪律）：无法事后改预测；
4. **失败永久且全计**（P0-3 + KL-3）：refuted 不可翻转、全量计入；
5. **确定性选择**：无"挑喜欢的"自由度；
6. **残余偏差 = 扫描器覆盖**（v0 四类）：可审计、可扩展，诚实声明。

### Q7 discovery 的最小定义是什么？

**一个纯粹可派生的事实时刻：某已声明关系的 RelationEvidence 从"零
held-out 确认"变为"存在 ≥1 个 held-out confirmed pair"。**

即"第一次在训练边界之外存活"。它是 ledger 事实（该 pair 存在），不是授予
的标签——无阈值、无评分、无状态翻转；`discoveries(knowledge)` 纯派生扫描
即可枚举全部发现。更丰富的发现（概念形成、异常消解、新关系声明）属于
后续阶段；v0 只承认这一最小科学时刻，并诚实声明其单薄。

### Q8 最小实现切片？

| 切片 | 内容 | schema |
|---|---|---|
| **AS-1** | `QuestionRecord` + `declare_question` / `withdraw_question`（问题注册表，状态 open/withdrawn，supersedes 链） | v7（+questions 数组） |
| **AS-2** | 四个事实扫描器（纯函数）+ 声明优先级 + `research_cycle(knowledge, agent, capabilities, budget)`：扫描 → 确定性选择 → 声明问题 → 委派既有闭环执行 → 返回本轮事实；单步循环，多步 = 调用方循环 | 无（扫描是派生；声明走 AS-1） |
| **AS-3** | `discoveries(knowledge)` 派生 + 文档收尾 | 无 |

**不做**：LLM/生成式提问、新实验种类、anomaly 残差模式挖掘、概念发明、
多步规划、belief 路径改造、Open Genesis。

## 3. 执行纪律的复用映射（无新闭环）

| 问题类 | 执行委托给 |
|---|---|
| untested_generality | `run_held_out_trial`（HP：确定性选择 + 承诺先于执行） |
| anomaly | `commit_discriminating_predictions` → `execute_proposal` → `verify_competing_predictions`（竞争判别链） |
| undefined_concept | `define_concept`（KL-2，校验复用词汇表） |
| unverified_identity | 同 anomaly（判别试验产生首个验证） |

## 4. 四问预答

1. **新能力**：AI 第一次能从**自己知识库的事实空缺**生成研究议程并自主执行
   ——此前一切研究目标由人供 Mission 定义。
2. **无需人工答案**：扫描器/优先级/执行闭环全部消费 AI 侧事实，零真值访问。
3. **最小可复现实验**：构造"关系已声明、training 确认、held-out 为空"的
   store → `research_cycle` 一步自动产生 untested_generality 问题 → 运行
   held-out trial → `discoveries` 报告首个发现（无需任何人供 Mission）。
4. **是否做到以前做不到的事**：是——从空知识库起跑，循环自主产生并执行
   研究（判别→泛化→定义），全程无人指定目标。


---
