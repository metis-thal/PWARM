# Genesis Core Completion Plan

Status: PLAN ONLY — no code changed yet. Written 2026-09-28 after a full read
of `src/pwarm/scientist/` (all modules), `tests/scientist/` (Step 1–10 test
contracts), `ARCHITECTURE.md` §Scientist Layer, `DATA_STRUCTURES.md` §12,
`DEFERRED_WORK.md` §Genesis Arc, and the persisted `knowledge/*.json`.

Goal: turn the existing Step 1–10 loop (predict → commit → experiment →
observe → verify → evidence → competition state) into a reliable, traceable,
extensible knowledge system — by closing semantic gaps with minimal changes —
before any new capability (Knowledge Layer, Autonomy, Open Genesis) is added.

---

## 0. Baseline snapshot

- HEAD = `5f8728e` (Steps 1–10 committed), working tree clean.
- Test suite: all green after repairing the venv — it was missing `pyyaml`
  (declared in `pyproject.toml` but absent from `.venv`; collection failed for
  every scientist test). Fixed via Aliyun mirror during this audit. ~280
  passed, 3 skipped.
- Genesis ledger APIs (`PredictionRecord`, `VerificationRecord`,
  `commit_prediction`, `record_verification`, `evidence_for_model`,
  `competition_state`) are consumed ONLY inside `pwarm/scientist/`
  (agent / knowledge / prediction / `__init__`). No scripts or CLI depend on
  the schema.
- All three persisted knowledge files (`knowledge/universe_00{1,2,3}.json`)
  are pre-Phase-1 schema (laws only, no `predictions`/`verifications` keys).
  **The schema-evolution window is open**: changing `PredictionRecord`'s
  hashed payload or adding fields breaks nothing in the repo today.

## 1. Gap map (audit checklist → findings)

| # | Checklist item | Current code | Finding |
|---|---|---|---|
| 1 | Commitment binding | `prediction.py:315-348`, `knowledge.py:182-209`, `agent.py:163-175,246-269` | Commitment hashes 5 fields (model_ref, claim, spec_ref, predicted, tolerance). **NOT bound**: condition binding, output binding, observation reduction, and the commitment→experiment identity is never checked at verification time. |
| 2 | Tolerance source | `prediction.py:82-105` (default `1.0`), `agent.py:265-268` (`per_model.get(id, 1.0)`) | Belief path is principled (belief span/2 + floor, `state.py`). Competition path silently defaults to 1.0 with no provenance — a hidden heuristic. |
| 3 | Verification lifecycle | `knowledge.py:237-240` | Implicit rule: open→confirmed→refuted; **refuted is absorbing**; confirmed CAN flip to refuted via a later verification. Undocumented, unpinned by a direct test. No SUPERSEDED (docstrings promise "supersede it with a new prediction" but no mechanism exists). |
| 4 | Model survival | `prediction.py:414-482` | Correctly facts-only (`CompetitionState`: confirmed/refuted/conflicts, full provenance). No winner/score — matches doctrine. Assessment layer correctly absent; keep it that way until autonomy needs it. |
| 5 | Binding responsibilities | `agent.py:71-91,230-304` | Agent declares bindings per call; reduction executed agent-side; verdict pure (`verify_prediction`); persistence in `KnowledgeBase.record_verification`. Gap: the declaration is ephemeral — never persisted into the commitment (see #1). |
| 6 | Time & identity of Exp/Obs/Verif | `experiment.py:71-82`, `prediction.py:351-361` | `ExperimentSpec.id` is spec-derived (same spec ⇒ same experiment_id). Deterministic engine ⇒ same spec re-run yields an identical record, so merging is informationally sound TODAY, but breaks at World Epochs. `VerificationRecord` has **no timestamp**. |
| 7 | CompetitionState temporality | `prediction.py:414-482` | No time ordering anywhere; ordering is inferred from string-sorted ids (`verif-NNNN`). |
| 8 | prediction.py cohesion | 597 lines | Five responsibilities (models / commitments / verdicts / bindings+reduction / competition) + `state↔knowledge↔prediction` TYPE_CHECKING cycle. Split is mechanical, do after semantic fixes. |

Two additional findings beyond the checklist:

- **Two divergent adjudication paths.** `adjudicate` (belief path,
  `prediction.py:564-597`) hardcodes `claim == "gravity"` and fits the whole
  record (`fit_free_fall`); `verify_competing_predictions` (competition path,
  `agent.py:273-304`) uses OutputBinding + ObservationReduction. Same
  commitment, two different comparison semantics, chosen by which function the
  caller invokes.
- **Spec translation duplication.** `commit_discriminating_predictions` and
  `execute_proposal` each translate proposal→spec independently; equality of
  committed `spec_ref` and executed `experiment_id` holds only by caller
  discipline. (P0-1's identity check closes the verification-time hole.)

## 2. The plan (priority order)

Each item = one minimal slice: read → minimal change → tests → full
regression → report. All new code stays AI-side (the AST truth-free tests
must keep passing, extended to any new module).

---

### P0-1 · Enforce commitment→experiment identity at verification time  【must】

- **Where**: `knowledge.py:211-242` (`record_verification`).
- **Now**: any committed prediction can be adjudicated against ANY record and
  the mismatched verdict is persisted as immutable history.
- **Problem**: violates "commitment binds experiment identity"; a wrong-pair
  verification becomes permanent scientific fact.
- **Minimal change**: `record_verification` refuses when
  `record.experiment_id` is not covered by the commitment's `spec_ref`
  (comma-joined list on the belief path). Both verification paths flow
  through this one function, so one check covers both.
- **Tests**: matching id passes; mismatched refused loudly; `run_mission`
  regression (one prediction, joined 3-spec `spec_ref`, 3 records);
  competition-path regression (Step 7 suite unchanged).
- **Size**: tiny.

### P0-2 · VerificationRecord timestamp + collision-safe id generation  【must】

- **Where**: `prediction.py:351-361`; `knowledge.py:192-193, 226`.
- **Now**: `VerificationRecord` has no `created_at`; ids are `len(dict)+1`.
- **Problem**: no temporal dimension at all (checklist 6/7); `len`-derived
  ids can collide if a file is hand-merged or hit by the known parallel-session
  hazard.
- **Minimal change**: add `created_at: str = ""` (additive; loader defaults
  missing keys to `""`; outside any hash — VerificationRecord has none);
  derive next prediction/verification seq from `max(existing)+1`, not `len`.
- **Tests**: old-schema file loads unchanged; new field round-trips; ids stay
  monotonic after delete-free reload.
- **Size**: tiny.

### P0-3 · Pin the status lifecycle in docs + tests  【must】

- **Where**: `knowledge.py:237-240`; `DATA_STRUCTURES.md` §12; `ARCHITECTURE.md`.
- **Now**: transition rule is implicit. Behavior: open→confirmed;
  confirmed→refuted allowed (a later verification may overturn a confirmation);
  refuted→anything refused (absorbing).
- **Problem**: the lifecycle the project actually runs is nowhere stated, so
  every later decision (survival, reopen, anomaly) builds on undocumented
  semantics.
- **Minimal change**: no behavior change. Document the state machine; add
  direct tests pinning each allowed/refused transition.
- **Deliberate scope decision**: add **no** INCONCLUSIVE / REOPENED now —
  nothing in the code produces them (reduction failures refuse loudly and
  persist nothing, which is honest). SUPERSEDED is separate (P2-6).
- **Size**: tiny.

### P1-4 · Bind the verification contract INTO the commitment  【must, core】

- **Where**: `prediction.py:315-348, 517-533` (record + payload);
  `agent.py:246-269` (commit), `273-304` (verify); `knowledge.py:93-119` (load).
- **Now**: OutputBinding / ObservationReduction / ConditionBinding are
  caller-supplied at verification time. The same committed prediction can be
  adjudicated through different channels/rules producing different persisted
  verdicts (DEFERRED_WORK #1).
- **Problem**: post-hoc choice of comparison channel is a researcher degree of
  freedom; the commitment must declare HOW it will be checked, before the
  experiment.
- **Minimal change**:
  1. Add to `PredictionRecord`: `condition_binding: dict`, `output_binding:
     dict`, `reduction_channel: str`, `reduction_rule: str` — included in
     `commitment_payload` (they are declared pre-experiment scientific
     content). Bump `SCHEMA_VERSION` 2→3; loader defaults missing keys to
     empty (old files load unchanged; the repo has no persisted predictions).
  2. `verify_competing_predictions` rebuilds the contract FROM the commitment
     and refuses a caller contract that differs.
  3. Transition rule: empty contract ⇒ legacy belief-path adjudication still
     allowed (keeps this slice independently shippable until P2-7).
- **Tests**: schema-pin tests updated to the new canonical field list — this
  is a planned, reviewed schema change, not test-gaming: the hash-immutability
  tests (B/C) keep their meaning untouched; contract-mismatch refused;
  contract-match passes; round-trip through JSON.
- **Size**: medium.

### P1-5 · Make tolerance provenance explicit  【must】

- **Where**: `prediction.py:82-105` (`model_prediction(..., tolerance=1.0)`),
  `agent.py:265-268` (`per_model.get(model.model_id, 1.0)`).
- **Now**: competition-path tolerances silently default to 1.0; nothing
  records where a tolerance came from. Belief path is principled already
  (span/2, floored — `prediction.py:496-514`).
- **Problem**: a hidden heuristic sitting inside a hashed commitment
  (discipline: no heuristic posing as science).
- **Minimal change**: `commit_discriminating_predictions` requires an explicit
  tolerance per model and raises when one is missing (no guess);
  `model_prediction`'s default removed or made explicit-only on the commit
  path; optionally add unhashed `tolerance_source: str = ""` provenance to
  `PredictionRecord` (e.g. "belief span/2", "declared by model H1").
  The g·dt² recording offset stays absorbed by tolerance — never corrected
  (existing test F keeps enforcing this).
- **Tests**: missing tolerance → refused; explicit per-model tolerances flow
  unchanged (`test_rivals_can_have_different_tolerances` survives).
- **Size**: small.

### P2-6 · SUPERSEDED status + changed-mind linkage  【should, before Phase 2】

- **Where**: `knowledge.py:182-209`; the status machine from P0-3.
- **Now**: docstrings say "supersede it with a new prediction" but there is no
  mechanism; a replaced prediction remains an unlinked, open ledger entry
  forever.
- **Minimal change**: `KnowledgeBase.supersede(old_id, new_id)` — allowed only
  from `open`; sets status `"superseded"`; optional unhashed
  `supersedes: str = ""` field on the new record pointing back. Do NOT add
  REOPENED/INCONCLUSIVE (no producers exist yet — revisit when the Anomaly
  engine creates a real trigger).
- **Tests**: only-open rule; refused from confirmed/refuted; evidence
  aggregation treats superseded predictions as history (they carry no
  verifications, so `CompetitionState` is unaffected — pin that).
- **Size**: small.

### P2-7 · Unify the two adjudication paths  【should】

- **Where**: `prediction.py:564-597` (`evaluate` hardcodes `claim ==
  "gravity"` and fits); `agent.py:177-191` vs `273-304`.
- **Now**: belief path = fit whole record; competition path = declared
  reduction. Two semantics for "compare a commitment to an observation".
- **Minimal change**: register `fit_free_fall` as a reduction rule (e.g.
  `"free_fall_g"`: channels (t, z) → g scalar); `evaluate` becomes that
  rule's application; after P1-4 the belief path declares the same contract at
  commit time. The claim hardcode disappears; `adjudicate` and
  `verify_competing_predictions` share one shape: commitment → contract →
  reduce → `verify_prediction`.
- **Tests**: existing A–F verdict semantics byte-identical (adjudicate results
  unchanged on the same records); new rule numerically equal to `evaluate`.
- **Size**: medium (can ride in the same slice as P1-4 or immediately after).

### P2-8 · CompetitionState minimal temporality  【may defer】

- **Where**: `prediction.py:414-482`; depends on P0-2.
- **Minimal change**: derive `last_verification_at` (and/or time-ordered
  provenance) in `competition_state` — pure derivation, no new writes.
  Deeper temporal semantics (evidence expiry, epoch scoping) wait for
  Creation/Epoch.
- **Size**: tiny once P0-2 lands.

### P2-9 · Split prediction.py  【may defer; mechanical】

- **Where**: `prediction.py` (597 lines, 5 responsibilities);
  `ARCHITECTURE.md` module table.
- **Minimal change**: mechanical split (models / commitment / contracts /
  adjudication / competition) with `prediction.py` kept as a re-export shim so
  every existing import and the AST truth-free checks keep working; extend the
  AST check to the new module files.
- **Order**: after P1-4/P2-7 so the code moves once, in its final shape.
- **Size**: medium, zero behavior change, full regression gate.

## 3. Explicitly deferred (documented triggers, nothing built now)

| Item | Trigger to build |
|---|---|
| Experiment run-identity (`ObservationRecord` run id distinct from spec id) | World Epoch / Creation makes same-spec-different-observation possible. Today the engine is deterministic: same spec ⇒ identical record ⇒ merging duplicates is informationally correct. Document, don't build. (`experiment.py:71-82`) |
| Model survival / assessment layer (no winner/score/probability) | The autonomy loop needs "what should I investigate next" judgments. Must be a separate layer ON TOP of `CompetitionState` facts, never inside `EvidenceSummary`. Cross-model discriminating-pair view (A confirmed E1 while B refuted E1) is derivable facts-only when first needed. |
| INCONCLUSIVE / REOPENED statuses | Real producers exist: partial observations (INCONCLUSIVE) and the Anomaly engine (REOPENED). |
| Knowledge Layer: ModelRecord → Evidence lineage → Held-out prediction → RelationRecord → Anomaly → Definition/Concept | After Genesis Core is complete; first slice = ModelRecord (model identity + lineage), keeping `ScientificModel` as the in-memory candidate. |
| Held-out prediction / migration discipline (fit → freeze → predict under unseen conditions → commit → run → verify) | After ModelRecord + P1-4 (the commitment contract machinery is exactly what held-out verification needs). |
| `ambient_fluid.density` designer-routing landmine (DEFERRED_WORK #6) | Mission-layer issue, not Genesis core. |
| Anomaly engine, Concept discovery, Creation/Epoch, Open Genesis Pilot, AI-decides-what-to-study | Strictly after core completion, in that order. |

## 4. Suggested slice order

P0-1 → P0-2 → P0-3 → P1-5 → P1-4 (+P2-7) → P2-6 → P2-8 → P2-9.

Rationale: P0 items are tiny and make the existing ledger trustworthy before
anything else touches it; P1 items are the two real semantic completions;
P2 items round out lifecycle/temporal/structure. Every slice ends with the
full regression suite plus a report (what changed, why, which tests, what
remains).

## 5. Four-questions check (per completed stage)

1. **New capability**: P0 = verdicts you can trust (identity-checked,
   timestamped, pinned lifecycle); P1 = commitments that carry their own
   verification contract and honest tolerances; P2 = explicit changed-mind
   provenance and a single adjudication semantics.
2. **Runs without human answers**: all items are AI-side bookkeeping/contract
   changes — no universe truth involved (AST truth-free tests keep enforcing).
3. **Minimal reproducible experiment**: each slice's new tests + the existing
   Step 1–10 suites + full regression.
4. **Does it enable what was impossible before?** Yes, in the narrow but
   load-bearing sense: after P1-4 + P1-5, a committed prediction is a
   COMPLETE, self-describing contract — the precondition for held-out
   prediction and for an autonomy loop that commits without human-supplied
   comparison rules. If a slice cannot answer (4), it does not ship.

## 6. Discipline notes

- Historical facts stay immutable: every item above appends or refuses; the
  only sanctioned status transitions are the pinned ones.
- The P1-4 schema change is made openly (schema-pin tests updated to the new
  canonical list) while the window is open — no persisted predictions exist in
  the repo, and hash-immutability tests keep their semantics.
- No new abstraction beyond the fields/functions listed; no parallel system
  next to the existing mechanisms.
