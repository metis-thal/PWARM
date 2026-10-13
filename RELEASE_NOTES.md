# PWARM Release Notes / 发布说明

## v0.1.0a1 — Pilot 2 / 试点二号（2026-10-12）

**Physical World AI Reasoning Model — 试点能力集的第二版发布候选 / Release
Candidate 2 of the pilot capability set**

> v0.1.0a1 将试点能力集（首发于 `v0.1.0-pilot`，自称 Release Candidate 1）加固到
> 可分发状态：生产导入完整性修复、双平台 CI 通过、wheel 经洁净安装验证。
> v0.1.0a1 hardens the pilot capability set (first released as `v0.1.0-pilot`,
> self-described as Release Candidate 1) for distribution: production import
> integrity restored, CI green on both platforms, and the wheel verified by a
> clean-room install.

### 能力概览 / What the scientist can do（承 v0.1.0-pilot，未变更）

Genesis 弧 Steps 1–10 使科学家从*参数辨识器*变成*假设检验智能体*（全部 AI 侧、
纯测量通道；无 schema 破坏，Mission 001–003 由回归测试钉住）。以下能力随
`v0.1.0-pilot` 发布并在本版中原样保留 / The following shipped in
`v0.1.0-pilot` and carries unchanged into this release:

- **承诺先于观测**（Steps 1/6）— 预测以 sha256 锚定并在实验运行前持久化；
  篡改可被检出；竞争模型独立承诺 / Commitment before observation:
  hash-anchored predictions persisted before the experiment runs; tampering
  detected; competing models commit independently.
- **信念驱动预测**（Step 2）— 预测来自科学家自身的不确定区间（无硬编码先验）；
  确认收窄信念、反驳放宽信念 / Belief-driven predictions from the scientist's
  own uncertainty intervals.
- **判别性提案**（Steps 2–4.5）— 按模型分歧排序条件；ConditionBinding/
  OutputBinding 显式契约绑定模型词汇与实验通道 / Discriminating proposals with
  explicit model↔experiment contracts.
- **逐模型裁决**（Steps 5–7）— 单次观测产生单一 CONFIRMED/REFUTED 裁决；
  裁决绝不重跑物理 / One verdict per committed prediction; verdicts never
  re-run physics.
- **证据台账**（Steps 8–10）— 逐模型证据汇总与竞争状态，完整溯源；统计单位为
  (model, experiment)；分歧报告为冲突而非调和 / Evidence ledger with full
  provenance; facts, not scores.
- **Mission 004 Day 1 基础设施** — 未知液体测量装置（universe_004、浸没装置、
  可复现性包络）/ Mission 004 Day-1 measurement infrastructure.

### 本版核心修复 / Core fixes in this release

- **生产导入完整性**（提交 `875a4cf`）— 在 `[viz,dev]` 依赖环境下，除 4 个登记
  豁免外，导入契约检查覆盖的模块均可在无 stub 的生产解释器中导入（裸安装下另有
  3 个可选依赖模块不可导入，见下方洁净安装验证的分账）；由子进程导入契约测试
  守护（新坏导入或过期豁免都会使测试失败）。修复：4 个注解级 `rules` 模块
  （kernel 导入移入 `TYPE_CHECKING`）、`import pwarm.parallel` 包级惰性门控
  （缺 ray 时给出 `pip install 'PWARM[parallel]'` 可操作提示）、`rules/fracture`
  重指向真实 2D 值类型（`rules/bodies2d`，语义忠实恢复：代码与历史实现逐 AST
  节点等价，仅 docstring 存在适配性差异）。
  Production import integrity: under the `[viz,dev]` dependency set, every
  module covered by the import contract imports in a stub-free interpreter
  except 4 registered exemptions (bare installs additionally lack 3
  optional-dependency modules — see the per-wheel evidence below), gated by a
  subprocess import contract; fracture re-pointed onto the restored 2-D value
  types.
- **CI Windows 防护**（提交 `b4efa2d`）— 无 GPU 的 CI Windows runner 上 VTK
  原生渲染访问违例（进程内不可捕获），pyvista 查看器测试改为静态跳过；本地开发机
  仍运行全部 14 项 / Viewer tests skip on GPU-less CI Windows runners (VTK
  native access violation, uncatchable in-process); developer machines still
  run all 14.

### 验证成果 / Verification（全部对应提交 `b4efa2d`）

- 测试 **896 = 889 passed + 7 skipped**（3 项慢速地质测试 + 4 项导入豁免）；
  分支覆盖率 **87.89%**（门槛 ≥ 80%）；Ruff 通过；可复现性测试 6/0/0。
  896 tests (889 passed, 7 skipped), branch coverage 87.89% (gate ≥ 80%),
  Ruff clean, reproducibility suite green.
- **CI 状态**：双平台 CI 通过——发布提交上 run `38044388419`（ubuntu + windows ×
  py3.11）**双 job 全绿**。CI 历史：早期双平台成功记录已存在（run `37746416770`，
  提交 `52e5341`，2026-10-08，双 job 成功）；**当前测试套件**在真实 CI 上的首次
  执行为 run `38030465129`（`875a4cf`），ubuntu 绿、windows 因 VTK 原生崩溃失败，
  随后由 `b4efa2d` 的防护修复使 run `38044388419` 双 job 全绿。
  CI status: both platforms green — run `38044388419` on the release commit is
  green on both jobs. History: an earlier double-green record exists (run
  `37746416770`, commit `52e5341`, 2026-10-08); the CURRENT test suite's first
  real-CI execution was run `38030465129` (`875a4cf`), where windows crashed on
  VTK, fixed by `b4efa2d` so run `38044388419` is green on both jobs.
- **洁净安装验证（两个 wheel 的证据分账）**：
  - `0.1.0a0` wheel（构建自 `b4efa2d` 提交树，`git archive` 隔离导出）：全新
    venv 中 **75/82 模块可导入**（7 个失败恰为 4 个登记豁免 + 3 个可选依赖
    模块），`pwarm demo 001` → DISCOVERY CONFIRMED（误差 0.0001%）。
  - 候选 `0.1.0a1` wheel（构建自候选工作树——变更尚未提交，**不存在"从发布
    提交构建"**）：实测通过版本四重一致（模块 `__version__` / wheel METADATA /
    `pip show` / `pwarm --version`）、`--help` 与 `pwarm demo 001` 冒烟
    （DISCOVERY CONFIRMED）。
  - 两个 wheel 的代码树除版本元数据外等价；**75/82 导入结果来自 `0.1.0a0`
    wheel 的实测**，不重复声明为 `0.1.0a1` wheel 已执行的测试。
  Clean-room install evidence, per wheel: the `0.1.0a0` wheel (built from the
  `b4efa2d` tree via an isolated `git archive` export) passed the 82-module
  import contract in a fresh venv (75/82 importable) and the stranger test;
  the candidate `0.1.0a1` wheel (built from the uncommitted candidate tree —
  no release commit exists) is verified for four-way version consistency,
  CLI and the `demo 001` smoke test. The two trees differ only in version
  metadata; the 75/82 result belongs to the `0.1.0a0` wheel and is not
  re-claimed as an executed test of the `0.1.0a1` wheel.
- **Open Genesis Pilot 端到端 ACCEPTED**（全部审计断言，含 E-1
  hidden-truth-unreachable 与 E-13 belief-path-zero-calls）。
  Open Genesis Pilot accepted end-to-end with all audit assertions.

### 支持范围 / Support scope

**稳定支持 / Stable:**
CLI（`pwarm demo`/`mission`/`dashboard`）、Missions 001–003、Open Genesis
Pilot、`pwarm.scientist`（预测-承诺-验证-竞争-知识层）、`pwarm.physics`、
`pwarm.geology`、`pwarm.rules`（fluid/chemistry/thermal/ecology/fracture/
materials/bodies2d）、`pwarm.viz` 渲染管线（SceneSnapshot/GLRenderer）、
可复现性封装、`pwarm` wheel 的干净安装。

**实验性或暂不支持 / Experimental or not supported:**
- `parallel` extra（Ray）— **未验证**：无真实 Ray 环境执行证据；即使安装 ray，
  `parallel/ray_parallel` 仍因遗留内核导入不可用（登记为 P2 迁移项）。
  The Ray path is UNVERIFIED and `ray_parallel` is unusable even with ray
  installed (P2 migration pending).
- `pwarm.viz.viewer` / `pwarm.viz.viewer3d`、`pwarm.ai.experiment` — 遗留模块，
  待退役或迁移（P2）。
- `pwarm.ai.observer.collect_free_fall` 与
  `pwarm.viz.snapshot.build_snapshot_from_world` — 可导入但调用即失败
  （P1-2/P1-3 待修复）。
- GPU 文本叠加（`pwarm.viz.text_overlay`）— 依赖 Pillow，**裸安装不含**
  （Pillow 未声明于任何 extra；`[viz]` 及以上安装经 matplotlib 传递获得）。
  GPU text overlay needs Pillow, which bare installs do not include.

### 已知限制 / Known limitations

- Python 3.10–3.11；**CI 仅覆盖 py3.11**（py3.10 下限仅有本地验证证据）。
- **真实 Ray 执行路径未验证**（见上）。
- **可复现性**：`tests/test_reproducibility.py`（6 项）通过；`reproducibility/
  mission_*/run.sh` 的 shell 双跑对比在本周期**未执行**。
- **PyInstaller 可执行文件本版未重建、未验证**（见下节）。
- numba 为核心依赖（JIT 编译）；符号回归 gplearn 后端为可选且懒加载，未安装时
  自动降级；DEFERRED_WORK.md 登记的 6 个 Minor 待后续处理。

### 与旧 PyInstaller 产物的区别 / Not the old .exe artifacts

本发布**不包含**仓库 `dist/` 目录中 2026-09-04 构建的 `PWARM_*.exe`——它们
构建于 `pymo → pwarm` 包名变更之前，与当前代码无关且未经验证。本发布的交付物
是从发布提交构建的 wheel（安装法见下）；旧 exe 不构成发布内容。
This release does NOT include the `PWARM_*.exe` files in the repository's
`dist/` directory (built 2026-09-04, before the package rename, unverified
against current code). The deliverable is the wheel built from the release
commit.

### 安装与验证 / Install & verify

**wheel 安装（本地构建流程 — 目前没有下载地址，请自行从源码构建）/
Local wheel build + install (no download location exists yet — build from
source):**

```bash
git clone https://github.com/metis-thal/PWARM.git
cd PWARM
git checkout v0.1.0a1                                  # 发布标签（打标签后可用）
python -m pip wheel . --no-deps -w dist                # 本地构建 wheel
python -m venv .venv-clean
source .venv-clean/Scripts/activate                    # Windows + Git Bash
# cmd: .venv-clean\Scripts\activate.bat ；POSIX: source .venv-clean/bin/activate
pip install dist/pwarm-0.1.0a1-py3-none-any.whl
pwarm --version                                        # → pwarm 0.1.0a1
pwarm demo 001                                         # → DISCOVERY CONFIRMED
```

**源码/开发安装（修正后的命令）/ Source & dev install (corrected commands):**

```bash
git clone https://github.com/metis-thal/PWARM.git
cd PWARM
python -m venv .venv
source .venv/Scripts/activate                          # Windows + Git Bash
# cmd: .venv\Scripts\activate.bat ；POSIX: source .venv/bin/activate
pip install -e ".[viz,dev]"
pytest tests/ -q                                       # 完整套件（本版 896 项）
python scripts/demo_mission_001.py                     # 001 → 002 → 003
```

---

## v0.1.0-alpha（2026-09-18，历史存档 / historical archive）

> ⚠️ **历史存档 / Historical archive** — 以下按 v0.1.0-alpha 发布时的事实原样
> 保留：其中的测试数量（70 passed）与"尚无 CI 工作流"等表述在当时属实，现已过时
> （当前为 896 项测试、双平台 CI 全绿，见上方 v0.1.0a1 段）。历史安装命令中
> `.venv/Scripts/activate` 缺少调用前缀，正确写法见上方 v0.1.0a1 安装段。
> The text below is preserved verbatim as it was written for v0.1.0-alpha;
> several statements (test counts, "no CI workflow") were true then and are
> outdated now — see the v0.1.0a1 section above.

**Physical World AI Reasoning Model — 首个开源版本 / First open-source release**

> v0.1.0-alpha 是 PWARM "AI 科学家"主线前三章的快照：AI 能发现定律、能自选实验、
> 能在预算下做科学并向系统申请仪器。
> v0.1.0-alpha snapshots the first three chapters of the "AI scientist" arc:
> an AI that discovers laws, designs its own experiments, and does science
> under a budget — including requesting new instruments when stuck.

### Unreleased — Genesis: the prediction–commitment–verification loop
（原 "Unreleased" 段，其内容已整理至上方 v0.1.0a1 段 / the content of this
section has been reorganized into the v0.1.0a1 entry above。）

### Highlights / 亮点

#### AI Scientist（三层任务课程 / three-mission curriculum）

| Mission | 能力 / Capability | 结果 / Result |
|---------|-------------------|---------------|
| 001 — Discover Gravity | 从自由落体观测中发现定律，交叉验证并持久化知识 | g 误差 0.0007% |
| 002 — Autonomous Experimentation | 不确定性建模 + 按信息增益自选实验；诚实报告不可识别参数 | 5/6 参数，误差 ≤0.14% |
| 003 — Science Under Constraints | 预算（4 实验/1400 步/12 成本单位）下的**价值排序**决策；不可识别 → 缺口分析 → **仪器申请 → 授权 → 浮力实验** | 7/7 参数（含密度），仪器弧完整 |

Mission 003 的仪器弧是本版本的核心叙事：AI 用落体/滑动装置穷尽后诚实地承认
"密度不可识别"（等效原理），分析缺失的测量能力（需要浸没），提交流体箱申请，
系统授权并扩展预算，浮力实验随即测出 7799.95 / 2700.0 kg/m³（真值 7800/2700）。

#### 统一多物理引擎 / Unified multi-physics engine

- 刚体 / SPH / FEM / MPM / PBD / 热力学 / 化学 / 地质求解器，显式耦合器
  Rigid / SPH / FEM / MPM / PBD / Thermal / Chemistry / Geology solvers with an explicit coupler
- 统一碰撞：SAP 宽相 + GJK/EPA 窄相 + CCD
  Unified collision: SAP broad-phase + GJK/EPA narrow-phase + CCD
- 核心原则：零硬编码现象，物理引擎是绝对真值，AI 只能通过测量通道观察
  Zero hardcoded phenomena; the engine is absolute truth, the AI observes
  through a measurement-only channel

### 知识持久化 / Knowledge persistence

`knowledge/universe_*.json` 保存已确立的定律与材料属性。第二次任务运行直接从
"文明知识"得出结论，不重跑任何实验（删除对应 JSON 即可让科学家从零开始）。

### 测试 / Tests

`pytest tests/` — **70 passed, 3 skipped**（其中 scientist 21 项全覆盖三个 Mission）。
（发布当时数字 / as measured at the v0.1.0-alpha release; current numbers in the
v0.1.0a1 section above.）

### 破坏性变更 / Breaking changes

- 旧版 `pwarm.kernel`（2D/3D 遗留内核）已删除；请迁移到 `pwarm.physics`
  Legacy `pwarm.kernel` removed — migrate to `pwarm.physics`
- 项目发行名由 `pymo` 改为 `PWARM`（v0.1.0-alpha 时导入包名仍为 `pymo`，
  现已一并更名为 `pwarm`——见上方 v0.1.0a1 段）
  Distribution renamed `pymo` → `PWARM` (at v0.1.0-alpha the import package
  was still `pymo`; it has since been renamed to `pwarm` — see v0.1.0a1 above)
- 实验设计器准则从"信息增益最大化"升级为"价值 = 增益/成本"：
  Mission 002 的首选落体高度由 50 m 变为 10 m（同样达到知识阈值，成本 1/4）
  Designer doctrine upgraded from max-information to value = gain/cost

### 已知限制 / Known limitations（历史 / as of v0.1.0-alpha）

- Python 3.10–3.11；numba 为核心依赖（JIT 编译）
- 可视化依赖 OpenGL 3.3+；无 GPU/无头环境请用 `--frames N` 自动退出或仅跑测试
- 符号回归的 gplearn 后端为可选依赖（`ai.law_discovery` 懒加载），未安装时自动降级
- 尚无 CI 工作流；Windows 为第一开发平台，Linux 未系统验证
  （现已过时：CI 工作流已建立且双平台全绿 — see v0.1.0a1 above）

### 下载与验证 / Install & verify（历史 / as of v0.1.0-alpha）

```bash
git clone https://github.com/metis-thal/PWARM.git
cd PWARM && python -m venv .venv && .venv/Scripts/activate   # Windows
pip install -e ".[viz,dev]"
pytest tests/scientist/ -q
python scripts/demo_mission_001.py    # 001 → 002 → 003
```

### 路线图 / Roadmap

Mission 004 — Fluid & Thermodynamics（温度、黏度、热导率，热学仪器申请弧）已规划，
详见 README 路线图表。
