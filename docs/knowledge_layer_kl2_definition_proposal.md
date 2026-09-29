# KL-2 Definition Layer Proposal

Status: PROPOSAL ONLY — no code changed. Baseline: `432f55c` (KL-1 committed,
schema v4). Answers the seven questions in order; implementation plan follows
on approval.

---

## 0. 设计立场（一句话）

**DefinitionRecord 是概念的操作性定义（"如何测量"），不是概念的真相或评价**
——它给 `claim` 词汇一个可解析的测量程序，本身永远不携带任何验证结论。

---

## 1. DefinitionRecord 与 ModelRecord 的边界

| | ModelRecord（KL-1） | DefinitionRecord（KL-2） |
|---|---|---|
| 回答的问题 | "**谁**在预测？"（候选模型的身份） | "**claim 的词汇**是什么意思？"（测量程序） |
| 被谁引用 | `PredictionRecord.model_ref` → 注册 id | `PredictionRecord.claim` → `concept_id`；验证契约 `(channel, rule)` → 操作程序 |
| 有无 standing | 有（派生：`competition_state(model_id)`） | **没有**。定义不参加竞争、不被确认/驳倒；只有可派生的**使用事实**（哪些预测/契约引用了它） |
| 版本化单位 | 模型身份本身版本化（`linear` → `linear-v2` 是不同模型） | **概念身份稳定**（`gravity` 永远是 `gravity`），被版本化的是**定义**（测量程序的修订） |
| 生命周期 | registered → superseded | active → superseded（定义被**修正**，不是被驳倒） |
| 存储 key | `model_id`（id 即身份） | `definition_id`（`def-NNNN`）；`concept_id` 是字段——保证 claim 解析跨版本稳定 |

一句话边界：**删掉一个 ModelRecord，验证记录仍指向它的 predictions；删掉一个
DefinitionRecord，`claim="gravity"` 就失去注册的测量含义**——前者是认识主体，
后者是词汇基础设施。

## 2. DefinitionRecord 是否可以被预测引用？

**可以，且不改 PredictionRecord 的任何字段**。两条引用路径都是读取时派生：

1. **claim 解析（信念路径）**：`PredictionRecord.claim == concept_id` →
   `active_definition("gravity")`。claim 此前和 model_ref 一样是不可解析的
   自由字符串，KL-2 使其可解析——机制与 KL-1 同构，但零 schema 变更
   （claim 字段已经存在）。
2. **契约解析（竞争路径）**：竞争预测的 claim 是 model_id 而非概念名，其概念
   落地来自**验证契约**——`PredictionRecord.(reduction_channel, reduction_rule)`
   与定义的 `(channel, reduction_rule)` 匹配（`definitions_for_procedure`）。
   例如 `("z", "first")` 的预测由"释放高度读数"定义来解释。

**KL-2 不给 commit 加概念注册门槛**：竞争路径的 claim 是模型 id，强制注册会
破坏现有流；定义层是加性词汇，门槛留给自主循环阶段（显式推迟，测试钉死
"未注册 claim 仍可提交"）。

## 3. 如何连接 channel / reduction rule / unit / concept identity

```python
@dataclass(frozen=True)
class DefinitionRecord:
    definition_id: str = ""      # "def-NNNN"（_next_ordinal，bookkeeping）
    concept_id: str = ""         # 概念身份（claim 词汇；跨版本稳定）
    kind: str = "measurand"      # v0 仅此值（refuse 其他——诚实可扩展）
    unit: str = ""               # 声明单位，如 "m/s^2"
    channel: str = ""            # 观测通道：必须 ∈ _OBSERVATION_FIELDS
    reduction_rule: str = ""     # 归约规则：必须 ∈ _REDUCTION_RULES
    description: str = ""        # 操作性陈述（人类可读）
    created_at: str = ""         # bookkeeping
    status: str = "active"       # active | superseded（bookkeeping）
    supersedes: str = ""         # 被修订的 definition_id（bookkeeping）
    content_hash: str = ""       # 锚定 concept_id/kind/unit/channel/reduction_rule/description
```

连接机制：`(channel, reduction_rule)` 是**操作性定义**——"gravity 就是 z 通道
经 `free_fall_g` 归约所得的标量，单位 m/s²"。校验直接复用 contracts.py 的两个
既有已声明词汇（不发明新词汇表）；为此给 contracts.py 加两个**只读访问器**
`observation_fields()` / `reduction_rules()`（当前注册表是模块私有）。

概念身份（`concept_id`）与测量程序（`(channel, rule)`）在一条记录里绑定，
正是"操作性定义"的含义：没有测量程序的概念不是定义，是无词汇的噪声。

## 4. definition 被修正时的 lineage

与 P2-6/KL-1 同一 supersedes 模式，但有一个**有意的不对称**：

- **原子修订**：`define_concept(concept_id, …, supersedes=old_definition_id)`
  一步完成——校验 old 存在、old 为 active、**old.concept_id == 新 concept_id**、
  new 未链接他者；创建新 active 记录并把 old 翻为 superseded。
  为何原子而 ModelRecord 是两步：概念 id 跨版本**相同**，若允许
  "先建新 active 再关旧"，同一概念会瞬时存在两个 active 定义——claim 解析
  会歧义。原子修订使"每概念至多一个 active"成为结构不变量。
- **历史完整**：superseded 定义永久保留；`definition_history(concept_id)` 沿
  supersedes 链走（新→旧），`active_definition(concept_id)` 恰好返回零或一个。
- **零迁移语义**：修订定义不触碰任何既有记录——旧预测的验证契约里**已经**
  冻结了当时的程序（P1-4 的成果），历史裁决自动与旧定义自洽；新程序只影响
  未来的解释与提交。

## 5. 如何避免 concept score / truth value / confidence / winner

1. **字段 pin**：字段清单测试禁止 score/confidence/accuracy/winner/truth/
   verified/rank/weight/probability（与 KL-1 同款）。
2. **status 词汇 = {active, superseded}**：只表达"是否为当前定义"，永远不表达
   "是否为真"。定义不是命题，没有真值可言。
3. **无 standing**：定义层没有任何计数/统计字段。可派生的只有**使用事实**
   （引用它的活跃预测数、匹配它的契约数）——是引用计数式的事实，不是评价。
4. **定义不可被"确认"**：不存在 confirm/reject 类 op；对概念性质的判断属于
   KL-3 的 RelationRecord 与 ledger，永不属于定义层。
5. **知识记录零验证数据原则延续**：DefinitionRecord 不存储任何 residual/
   observed/status 类字段（KL-1 已确立的分层）。

## 6. 如何兼容 claim="gravity" 与 reduction="free_fall_g"

- `("z", "free_fall_g")` 正是 `LEGACY_REDUCTION`（adjudication.py）。首个注册的
  定义天然是：`concept_id="gravity", channel="z", rule="free_fall_g",
  unit="m/s^2"` ——**它是对既有程序的事后命名，不改一行裁决代码**。
- 兼容保证（测试钉死）：
  - `active_definition("gravity")` 解析信念路径的 claim；
  - `definitions_for_procedure("z", "free_fall_g")` 命中同一记录——契约侧与
    词汇侧在此汇合；
  - 未注册定义的 claim 仍可提交、旧验证仍可读（v3/v4 时代预测零迁移）；
  - 定义修订后，旧验证记录的 evidence 文本与 residual 原样不变。

## 7. schema：继续 v4 还是独立版本？

**全局升 v5，不设独立版本流。**理由：

1. `SCHEMA_VERSION` 描述的是**文件布局**，不是功能——独立版本流会造成
   "v4 + KL2?" 之类无法在加载时判定的组合矩阵。
2. 与既有先例一致：v2 加记录种类、v3 加契约字段、v4 加 model_records——
   每次用户可见的布局扩展单调 +1。
3. 当前 loader 不校验版本号（键容错双向兼容），升版零破坏。
4. **顺带的诚实加固（建议纳入 KL-2，3 行）**：loader 目前对**更新**版本的
   文件也不拒绝——旧代码读新文件再 save 会静默丢弃不认识的键。加
   `if stored_version > SCHEMA_VERSION: raise ValueError(...)` 让版本号真正
   有意义（旧文件加载不受影响）。此项是新行为，需你批准。

## 8. 实现清单（批准后执行，单切片）

| 文件 | 改动 |
|---|---|
| `knowledge_records.py` | +DefinitionRecord / definition_payload / definition_content_hash / verify_definition_record（~90 行） |
| `contracts.py` | +两个只读访问器 `observation_fields()` / `reduction_rules()` |
| `knowledge.py` | schema v5；`definitions` dict + load/save；`define_concept`（原子修订）/ `active_definition` / `definition` / `definitions_for_procedure` / `definition_history` |
| 测试 | 8 组：freeze+round-trip、逐字段篡改、校验拒绝矩阵、原子修订链、claim 解析、procedure 解析（LEGACY_REDUCTION 汇合）、facts-not-scores pin、v4→v5 兼容 + 未注册 claim 仍可提交 |
| 文档 | DATA_STRUCTURES §13（新表）+ ARCHITECTURE 模块行更新 |

**不做**：概念注册门槛、非 measurand kind、定义退役 op（无生产者）、
LawRecord 迁移、KL-3 RelationRecord。

## 9. 四问预答

1. **新能力**：claim 与验证契约从"裸字符串/裸程序"变为可解析到"带单位的
   操作性概念定义"——词汇层第一次可寻址。
2. **无需人工答案**：定义是 AI 侧声明，校验复用既有 AI 侧词汇注册表。
3. **最小可复现实验**：`definitions_for_procedure("z","free_fall_g")` 与
   `active_definition("gravity")` 在同一条记录汇合——既有实践被定义层完整命名。
4. **是否做到以前做不到的事**：是——同一 ledger 上，"gravity 这个 claim 意味着
   什么测量程序、什么单位、被修订过几次"第一次可回答；且定义修订不影响任何
   历史裁决（由契约内嵌保证）。
