# KL-1 Implementation Plan — ModelRecord (AI-side addressable model identity)

Status: PLAN ONLY — no code changed. Baseline: `3cb05a2` (Genesis Core v1).
Scope: **ModelRecord only** — no DefinitionRecord, no RelationRecord.

---

## 1. 目标

给 AI 一个可寻址、持久化、内容不可变的模型身份：`PredictionRecord.model_ref`
从自由字符串变成可解析的注册 id。模型 standing 继续由既有派生
（`competition_state`）给出——ModelRecord 本身是**声明事实，不是评价对象**。

## 2. 现状与接缝（实测）

- `ScientificModel(model_id, params)` 是内存对象，无持久身份。
- `model_prediction` 生成 `source=f"model {model_id}"`（`models.py:94`）→
  `commit_predictions` 用 `guess.source` 作为 `model_ref` → ledger 里是
  `"model linear"` 这类不可解析字符串。
- `KnowledgeBase` 只有 laws/predictions/verifications 三个 dict（schema v3）。
- 派生（`evidence_for_model`/`competition_state`）按 model_ref 字符串**精确匹配**。
- 竞争路径无 model_ref 值断言；信念路径断言 `startswith("state belief")`（不动）。
- 仓库无任何已持久化的 predictions（v3 schema 窗口开着）。

## 3. 设计

### 3.1 ModelRecord（新模块 `pwarm/scientist/knowledge_records.py`）

```python
@dataclass(frozen=True)
class ModelRecord:
    model_id: str                    # 可解析身份（必填非空）
    formula: str                     # 声明的人类可读形式
    params: dict[str, float]         # 声明参数，创建即冻结
    derived_from: str = ""           # 声明 provenance（实验 id / hypothesis 来源）
    created_at: str = ""
    status: str = "registered"       # registered | superseded —— 仅生命周期事实
    supersedes: str = ""             # 被替换的 model_id
    content_hash: str = ""           # 锚定 model_id/formula/params/derived_from
```

- **不存**：score / confidence / accuracy / winner / rank / weight / probability
  ——字段清单 pin 测试禁止出现。
- **status 词汇 = {registered, superseded}**：只表达"是否仍是当前身份"，
  不表达优劣。唯一的写入者是 `register_model` / `supersede_model` 两个显式 op。
- **params 内容不可变**：`content_hash` 锚定声明四字段；改参数 = 注册新
  ModelRecord + supersedes 链（复用 commitment_hash 实现，不新造哈希机制；
  status/supersedes/created_at 在哈希外——与 PredictionRecord bookkeeping 策略一致）。

### 3.2 KnowledgeBase 集成（schema v3 → v4，加性）

- 新 dict `self.model_records: dict[str, ModelRecord]`；load 全 `rec.get(default)`，
  save asdict 数组；`SCHEMA_VERSION = 4`。旧 v1/v2/v3 文件加载不变。
- Ops（持久化前全部校验，异常为现有 ValueError 风格）：
  - `register_model(model_id, formula, params, derived_from="") -> ModelRecord`
    ——拒绝空 model_id、**重复 model_id**（身份唯一；参数变化走新记录）。
  - `supersede_model(old_id, new_id) -> (old, new)`
    ——拒绝：未知 id、自替换、old 已 superseded、new 已指向其他记录。
    效果：old.status="superseded"，new.supersedes=old_id（两条记录 replace 而非
    原地改，均仍 hash-verify——与 P2-6 模式一致）。
  - `model_record(model_id) -> ModelRecord | None`（查询）。
- **standing 零新增**：`competition_state(model_id)` 原样复用，不加任何包装字段。

### 3.3 model_id 替换自由字符串 model_ref（集成点）

1. `models.py`：`model_prediction` 的 `source=f"model {model.model_id}"` →
   `source=model.model_id`（一行）。信念路径 `prediction_from_belief` 的
   `source` **不动**（`"state belief (…)"` 是自模型状态描述，非 ScientificModel
   身份；容忍为可解析性之外的引用——与 orphan 纪律一致，文档写明）。
2. `agent.commit_discriminating_predictions` 增加注册校验（在 tolerance 校验前）：
   每个 `model.model_id` 必须 `knowledge.model_record(id)` 存在且
   `status == "registered"`——未注册按名拒绝；已 superseded 的身份拒绝再接新预测
   （与"superseded 承诺不可验证"同构）。model_ref 经由 `guess.source` 自然成为
   注册 id。
3. 信念路径（`run_mission`）不注册模型，model_ref 维持原样。

### 3.4 旧 PredictionRecord 兼容读取策略（不迁移）

- **零迁移、零改写、零猜测**：`evidence_for_model` / `competition_state` 继续
  按**存储的 model_ref 字符串精确匹配**。已注册 id 的 standing 恰好聚合
  model_ref == model_id 的预测；历史字符串（如 "model linear"）的预测原样可读、
  原样聚合在自己的键下，永不并入注册 id 的视图（并集 = 猜测，禁止）。
- v3 时代 store（无 model_records 键）加载为空注册表；新提交要求注册——
  store 加性演化。

## 4. 文件级最小改动清单

| 文件 | 改动 |
|---|---|
| `src/pwarm/scientist/knowledge_records.py` | **新增**：ModelRecord + model payload/hash 助手（复用 `records.commitment_hash`） |
| `src/pwarm/scientist/knowledge.py` | SCHEMA_VERSION=4；model_records dict + load/save；register_model / supersede_model / model_record |
| `src/pwarm/scientist/models.py` | source 一行（`f"model {id}"` → `model_id`） |
| `src/pwarm/scientist/agent.py` | commit_discriminating_predictions 的注册/状态校验（~8 行） |
| `tests/scientist/test_genesis_prediction.py` | 新 KL-1 测试段（~10 个）+ fixture 适配：`_committed_rivals`/`_competed_and_executed` 与 3 个直接调用点先注册模型 |
| `tests/scientist/test_genesis_prediction.py` + `test_api_contract.py` | schema pin 3→4；`knowledge_records` 进 GENESIS_MODULES 与 AI_MODULES |
| `DATA_STRUCTURES.md` / `ARCHITECTURE.md` | ModelRecord 行 / 模块表一行 |

**不改动**：`PredictionRecord`（零新字段）、`VerificationRecord`、
`record_verification`、`records.py`（只 import commitment_hash）、
信念路径 `form_prediction`/`run_mission` 的提交流、physics 层。

## 5. 测试清单（新增，全部 facts-only 断言）

1. 注册：字段冻结、content_hash 验证、params 篡改 → hash mismatch、
   save/load round-trip（含 status/supersedes/created_at）。
2. 重复 model_id 拒绝；空 model_id 拒绝。
3. supersede_model：old→superseded、new 回指、两条记录仍 hash-verify；
   拒绝矩阵（未知/自替换/已 superseded 的 old/已指向他者的 new）。
4. **状态词汇 pin**：ModelRecord 字段清单 + status ∈ {registered, superseded}
   + 无 score/confidence/accuracy/winner 类字段。
5. model_ref 替换：竞争路径提交后 `PredictionRecord.model_ref == 注册 id`
   （不再是 "model linear"）；未注册模型按名拒绝；superseded 模型拒绝新预测。
6. standing 派生链证明：经完整竞争流程（注册→提交→实验→验证）后，
   `competition_state(model_id)` 的 provenance 可解析到
   prediction→experiment→verification——身份可寻址的端到端证明。
7. 兼容读取：v3 时代文件（含 model_ref="model linear" 的预测、无 model_records）
   加载不变，保存升级 v4；老字符串预测保持原键聚合，**不并入**注册 id 视图
   （exact-match 策略 pin）。
8. truth-free：knowledge_records 加入扫描列表后全部 AST 检查通过。

## 6. 执行顺序（单切片，四步）

代码（新模块 → knowledge.py → models.py 一行 → agent.py 校验）→
fixture 适配 + 新测试 → scientist 套件 → 全量 pytest + ruff + 报告。

## 7. 显式不做（本轮）

DefinitionRecord / RelationRecord / established 判定 / retire 状态 / 信念路径
模型注册 / 老 model_ref 迁移或归一化 / standing 缓存或新字段。

## 8. 四问预答

1. **新能力**：模型身份跨会话可解析、参数版本有 lineage 链——此前不存在。
2. **无需人工答案**：注册/替换是 AI 侧声明操作，零 universe 访问。
3. **最小可复现实验**：测试 6 的端到端链（注册→提交→实验→验证→standing 解析）。
4. **是否做到以前做不到的事**：是——同一 ledger 上，standing 查询第一次可以
   以持久身份为键解析完整 provenance；且 status 词汇被 pin 死为非评价性。
