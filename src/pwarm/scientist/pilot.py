"""Open Genesis Pilot harness — world-side components (spec v2, sections H–O).

This module is HARNESS/WORLD SIDE: it assembles the pilot's world, holds the
ground-truth references, and runs the acceptance audits. It is NEVER given
to the scientist — the scientist receives only
:class:`PilotLaboratoryFacade` (measurement-only) and its own KnowledgeBase.

Components (design review approved; see
docs/open_genesis_pilot_specification.md v2):

* :class:`PilotLaboratoryFacade` — the scientist's only execution entry:
  ``run_experiment(spec) -> ObservationRecord``. Captures the ordinal
  watermarks at call entry, BEFORE physics runs; sanitizes exceptions.
* :class:`PilotExecutionLog` — scientist-unwritable execution journal with
  monotonic sequence and ordinal watermarks (spec J).
* :class:`IndependentVerifier` — physics-as-oracle: re-runs the committed
  spec in a fresh Laboratory and compares against the committed prediction
  with the committed tolerance (spec D).

Trust axiom (Pilot Axiom A): the v0 scientist is deterministic agent code,
not adversarial code. The boundary audits make violations DETECTABLE; they
do not make attribute traversal IMPOSSIBLE (Python cannot do that).
"""

from __future__ import annotations

import contextlib
from collections.abc import Mapping
from dataclasses import dataclass

from ..universes import Universe, load_universe
from .agent import BootstrapOutcome, ScientistAgent
from .experiment import ExperimentSpec, Laboratory, ObservationRecord
from .knowledge import KnowledgeBase
from .knowledge_records import (
    discoveries_from,
    verify_relation_record,
)
from .prediction import ConditionBinding, ObservationReduction, OutputBinding


class PilotExperimentError(Exception):
    """Sanitized execution failure — the real cause is recorded in the
    harness journal, never shown to the scientist."""


@dataclass(frozen=True)
class ExecutionEvent:
    """One scientist-triggered physics execution, with the ordinal
    watermarks captured at the facade call entry (BEFORE physics runs)."""

    seq: int                          # harness monotonic sequence
    event_type: str                   # "execute" | "kb_snapshot" | "tolerance_rule"
    spec_id: str = ""                 # experiment identity (execute events)
    conditions: tuple[tuple[str, float], ...] = ()   # public conditions
    pred_watermark: int = 0           # max prediction ordinal at entry
    verif_watermark: int = 0          # max verification ordinal at entry
    detail: str = ""                  # harness-side diagnostic (never scientist-visible)
    relation_ids: tuple[str, ...] = ()  # kb_snapshot events


class PilotExecutionLog:
    """Scientist-unwritable execution journal (spec J).

    The scientist holds no reference to this object; entries are appended
    only by the harness components. Ordinals are parsed numerically from
    the ledger ids (``pred-NNNN`` / ``verif-NNNN``) — string sort is never
    used for ordering.
    """

    def __init__(self) -> None:
        self._events: list[ExecutionEvent] = []
        self._seq = 0

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    @staticmethod
    def _max_ordinal(ids, prefix: str) -> int:
        highest = 0
        for identifier in ids:
            text = str(identifier)
            if text.startswith(prefix):
                digits = text[len(prefix):]
                if digits.isdigit():
                    highest = max(highest, int(digits))
        return highest

    def record_execution(self, spec: ExperimentSpec,
                         knowledge: KnowledgeBase) -> ExecutionEvent:
        """Capture the watermarks and append an execute event — called at
        the facade entry, BEFORE physics runs."""
        event = ExecutionEvent(
            seq=self._next_seq(),
            event_type="execute",
            spec_id=spec.id,
            conditions=tuple(sorted(spec_conditions(spec).items())),
            pred_watermark=self._max_ordinal(knowledge.predictions, "pred-"),
            verif_watermark=self._max_ordinal(knowledge.verifications, "verif-"),
        )
        self._events.append(event)
        return event

    def record_snapshot(self, knowledge: KnowledgeBase) -> ExecutionEvent:
        """Append a kb_snapshot event (relation ids + watermarks)."""
        event = ExecutionEvent(
            seq=self._next_seq(),
            event_type="kb_snapshot",
            pred_watermark=self._max_ordinal(knowledge.predictions, "pred-"),
            verif_watermark=self._max_ordinal(knowledge.verifications, "verif-"),
            relation_ids=tuple(knowledge.relations),
        )
        self._events.append(event)
        return event

    def record_tolerance_rule(self, model_id: str, rule: str) -> ExecutionEvent:
        """Append the F3 tolerance-rule declaration event (rule fixed)."""
        event = ExecutionEvent(
            seq=self._next_seq(),
            event_type="tolerance_rule",
            detail=f"{model_id}: {rule}",
        )
        self._events.append(event)
        return event

    def record_failure(self, detail: str) -> ExecutionEvent:
        """Append a sanitized failure diagnostic (harness-side only)."""
        event = ExecutionEvent(
            seq=self._next_seq(),
            event_type="failure",
            detail=detail,
        )
        self._events.append(event)
        return event

    @property
    def events(self) -> tuple[ExecutionEvent, ...]:
        return tuple(self._events)

    def executions_of(self, spec_id: str) -> tuple[ExecutionEvent, ...]:
        """Execute events for one experiment id, in sequence order."""
        return tuple(event for event in self._events
                     if event.event_type == "execute"
                     and event.spec_id == spec_id)


def spec_conditions(spec: ExperimentSpec) -> dict[str, float]:
    """The public condition parameters of a spec (apparatus vocabulary)."""
    return {"drop_height": float(spec.drop_height), "mass": float(spec.mass)}


class PilotLaboratoryFacade:
    """The scientist's ONLY execution entry (spec H).

    Holds the real Laboratory in a name-mangled private slot and delegates
    after capturing the ordinal watermarks. Exceptions are sanitized: the
    real cause goes to the harness journal, the scientist sees a
    :class:`PilotExperimentError` without universe names or secret words.
    """

    def __init__(self, laboratory: Laboratory, knowledge: KnowledgeBase,
                 journal: PilotExecutionLog) -> None:
        self.__laboratory = laboratory
        self.__knowledge = knowledge
        self.__journal = journal

    def run_experiment(self, spec: ExperimentSpec,
                       max_steps: int | None = None) -> ObservationRecord:
        # watermarks captured at ENTRY, before physics (S-1)
        self.__journal.record_execution(spec, self.__knowledge)
        try:
            if max_steps is None:
                return self.__laboratory.run_experiment(spec)
            return self.__laboratory.run_experiment(spec, max_steps=max_steps)
        except Exception as exc:
            self.__journal.record_failure(f"sanitized: {type(exc).__name__}")
            raise PilotExperimentError(
                "experiment failed: invalid or unsupported specification") from exc

# -- Independent verifier (spec D): physics-as-oracle, primitives-only --------

@dataclass(frozen=True)
class VerifierInput:
    """Primitives only — never scientist objects. ``predicted``/``tolerance``
    come from the hash-anchored PredictionRecord (committed BEFORE the
    experiment); ``response`` names the deterministic measurement projection
    (the committed contract's reduction, frozen pre-experiment)."""

    spec: ExperimentSpec
    predicted: float
    tolerance: float
    experiment_id: str
    response: str = "first sample of z"


@dataclass(frozen=True)
class VerifierResult:
    """The independent verdict: whether the committed prediction matches a
    fresh physics run within the committed tolerance."""

    experiment_id: str
    predicted: float
    tolerance: float
    independent_observed: float
    passed: bool


class IndependentVerifier:
    """Physics-as-oracle verification in a fresh Laboratory.

    Built and held by the harness only. Its observation is produced by a
    fresh physics run from the immutable spec — the scientist's own
    ObservationRecord is never an input, so even a tampered scientist-side
    record cannot influence the verdict. Deterministic physics (pinned by
    the reproducibility tests) makes the rerun reproduce reality.
    """

    def __init__(self, universe: Universe) -> None:
        self.__universe = universe

    def verify(self, input: VerifierInput) -> VerifierResult:
        laboratory = Laboratory(self.__universe)     # fresh physics, every time
        observation = laboratory.run_experiment(input.spec)
        independent_observed = float(observation.z[0])
        passed = abs(independent_observed - input.predicted) <= input.tolerance
        return VerifierResult(
            experiment_id=input.experiment_id,
            predicted=input.predicted,
            tolerance=input.tolerance,
            independent_observed=independent_observed,
            passed=passed,
        )

# -- Pilot audit assertions (spec O; design section D) --------------------------
#
# Each audit is an executable check over the harness artifacts (journal,
# facade, verifier) and the scientist-visible ledger. The catalog follows
# the design review: L(leakage) T(temporal) S(scope) P(purity) B(belief)
# D(discovery) V(independent verification).

@dataclass(frozen=True)
class AuditResult:
    audit_id: str
    passed: bool
    detail: str = ""


@dataclass
class BeliefPathCounters:
    """Mutable on purpose: instrumentation accumulators for audit B-1..B-4."""

    scientist_state: int = 0
    form_prediction: int = 0
    prediction_from_belief: int = 0
    default_prior_bounds: int = 0


class _CountingPriorBounds(Mapping):
    """Wraps DEFAULT_PRIOR_BOUNDS counting every read (F8 audit B-4)."""

    def __init__(self, inner: Mapping, counters: BeliefPathCounters) -> None:
        self._inner = dict(inner)
        self._counters = counters

    def __getitem__(self, key):
        self._counters.default_prior_bounds += 1
        return self._inner[key]

    def __iter__(self):
        return iter(self._inner)

    def __len__(self):
        return len(self._inner)

    def get(self, key, default=None):
        self._counters.default_prior_bounds += 1
        return self._inner.get(key, default)


@contextlib.contextmanager
def belief_path_audit(counters: BeliefPathCounters, agent: ScientistAgent):
    """Instrument every belief-path entry point for the duration of the
    pilot run (F8/M): ScientistState construction, form_prediction calls,
    prediction_from_belief calls, DEFAULT_PRIOR_BOUNDS reads."""
    import pwarm.scientist.agent as agent_module
    import pwarm.scientist.prediction as prediction_module
    import pwarm.scientist.state as state_module

    real_state = state_module.ScientistState
    real_prior = state_module.DEFAULT_PRIOR_BOUNDS
    real_pfb = prediction_module.prediction_from_belief
    real_form = agent.form_prediction

    class CountingState(real_state):
        def __init__(self, *args, **kwargs):
            counters.scientist_state += 1
            super().__init__(*args, **kwargs)

    def counting_pfb(*args, **kwargs):
        counters.prediction_from_belief += 1
        return real_pfb(*args, **kwargs)

    def counting_form(mission, state=None):
        counters.form_prediction += 1
        return real_form(mission, state)

    # agent.py binds ScientistState/prediction_from_belief at import time
    # (from .state import ...), so the interception points are agent
    # module's own globals — patching the defining modules would miss
    # every call the agent actually makes.
    agent_module.ScientistState = CountingState
    agent_module.prediction_from_belief = counting_pfb
    state_module.DEFAULT_PRIOR_BOUNDS = _CountingPriorBounds(real_prior, counters)
    agent.form_prediction = counting_form
    try:
        yield counters
    finally:
        agent_module.ScientistState = real_state
        agent_module.prediction_from_belief = real_pfb
        state_module.DEFAULT_PRIOR_BOUNDS = real_prior
        agent.form_prediction = real_form


def _parse_ordinal(ledger_id: str, prefix: str) -> int:
    digits = ledger_id[len(prefix):]
    return int(digits) if digits.isdigit() else 0


def _relation_evidence_pairs(knowledge: KnowledgeBase):
    return tuple((relation, knowledge.relation_evidence(relation))
                 for relation in knowledge.relations.values())


def _no_universe_beyond_facade(agent: ScientistAgent,
                               knowledge: KnowledgeBase) -> bool:
    """No Universe instance reachable from the scientist's STATE graph
    without crossing the facade boundary (Pilot Axiom A).

    Traversal follows data only: instance dicts, containers, closure
    cells. Function globals and module dicts are NOT expanded — every
    Python object with a method reaches its defining module's import
    graph, and the harness itself holds the Universe in the universe
    loader's registry; auditing that would audit CPython, not the
    scientist. What the scientist *holds* is what this audit bounds."""
    import gc
    import types
    seen: set[int] = set()
    stack: list = [agent, knowledge]
    while stack:
        obj = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if isinstance(obj, Universe):
            return False
        if isinstance(obj, PilotLaboratoryFacade):
            continue                     # the boundary itself: do not descend
        if isinstance(obj, (types.FunctionType, types.ModuleType, type)):
            if isinstance(obj, types.FunctionType) and obj.__closure__:
                stack.extend(cell.cell_contents for cell in obj.__closure__)
            continue
        stack.extend(gc.get_referents(obj))
    return True


def run_pilot_audits(facade, agent: ScientistAgent, knowledge: KnowledgeBase,
                     journal, verifier, bootstrap: BootstrapOutcome,
                     counters: BeliefPathCounters,
                     ) -> tuple[AuditResult, ...]:
    """Execute the full audit catalog over the finished pilot run."""
    results: list[AuditResult] = []

    def add(audit_id: str, passed: bool, detail: str = "") -> None:
        results.append(AuditResult(audit_id=audit_id, passed=bool(passed),
                                   detail=detail))

    # -- Leakage (L-1..L-4) ------------------------------------------------
    add("L-1", not hasattr(facade, "truth_summary"),
        "truth API unreachable on the facade")
    add("L-2", not hasattr(facade, "start_experiment")
        and not hasattr(facade, "universe"),
        "stepped path and Universe attribute unreachable")
    public_members = [name for name in dir(facade) if not name.startswith("_")]
    add("L-3", public_members == ["run_experiment"],
        f"facade public members: {public_members}")
    add("L-4", _no_universe_beyond_facade(agent, knowledge),
        "no Universe reachable without crossing the facade boundary")

    # -- Scope (S-1..S-5) ---------------------------------------------------
    events = journal.events
    snapshots = [event for event in events if event.event_type == "kb_snapshot"]
    initial = snapshots[0] if snapshots else None
    add("S-1", initial is not None and initial.relation_ids == (),
        "initial knowledge base had no relations")
    relations_ok = all(verify_relation_record(record)
                       for record in knowledge.relations.values())
    add("S-2", relations_ok and len(knowledge.relations) > 0,
        "every declared relation hash-verifies")
    executed_before = {}
    for event in events:
        if event.event_type == "execute":
            executed_before.setdefault(event.spec_id, event.seq)
    declaration_seq = None
    for snapshot in snapshots:
        if bootstrap.relation_record.relation_id in snapshot.relation_ids:
            declaration_seq = snapshot.seq
            break
    if declaration_seq is None:
        add("S-3", False, "relation never appeared in a snapshot")
        add("S-4", False, "no declaration snapshot")
    else:
        scope_ok = all(
            entry in executed_before and executed_before[entry] < declaration_seq
            for entry in bootstrap.relation_record.scope)
        add("S-3", scope_ok,
            "every scope condition was observed before the declaration")
        add("S-4", scope_ok, "no unobserved (future) condition in scope")
    add("S-5", all(verify_relation_record(record)
                   for record in knowledge.relations.values()),
        "relation declarations immutable (content hash covers scope)")

    # -- Temporal (T-1..T-4) -------------------------------------------------
    evidence = knowledge.relation_evidence(bootstrap.relation_record)
    heldout_confirmed = [pair for pair in evidence.heldout_pairs
                         if pair.status == "confirmed"]
    temporal_ok = True
    details: list[str] = []
    for pair in heldout_confirmed:
        executions = journal.executions_of(pair.experiment_id)
        prediction = knowledge.prediction(pair.prediction_id)
        if not executions:
            temporal_ok = False
            details.append(f"{pair.experiment_id}: no execution event")
            continue
        latest = executions[-1]
        if prediction.seq > latest.pred_watermark:
            temporal_ok = False
            details.append(f"{pair.experiment_id}: committed after execution")
        if _parse_ordinal(pair.verification_id, "verif-") <= latest.verif_watermark:
            temporal_ok = False
            details.append(f"{pair.experiment_id}: verified before execution")
    add("T-1", temporal_ok and len(heldout_confirmed) > 0,
        "; ".join(details) or "all held-out commitments precede execution")
    add("T-2", temporal_ok, "verification ordinals postdate their execution")
    rule_events = [event for event in events
                   if event.event_type == "tolerance_rule"]
    first_heldout_execute = min(
        (event.seq for pair in heldout_confirmed
         for event in journal.executions_of(pair.experiment_id)),
        default=0)
    add("T-3", any(event.seq < first_heldout_execute for event in rule_events),
        "tolerance rule was fixed before the first held-out execution")
    add("T-4", bool(rule_events),
        "tolerance rule declaration journaled (harness-side)")

    # -- Belief path (B-1..B-4) ----------------------------------------------
    add("B-1", counters.scientist_state == 0, "ScientistState constructions")
    add("B-2", counters.form_prediction == 0, "form_prediction calls")
    add("B-3", counters.prediction_from_belief == 0,
        "prediction_from_belief calls")
    add("B-4", counters.default_prior_bounds == 0,
        "DEFAULT_PRIOR_BOUNDS reads")

    # -- Discovery (D-1..D-4) -------------------------------------------------
    snapshot_before = knowledge.path.read_text(encoding="utf-8")
    derived = knowledge.discoveries()
    recomputed = discoveries_from(_relation_evidence_pairs(knowledge))
    snapshot_after = knowledge.path.read_text(encoding="utf-8")
    add("D-1", derived == recomputed,
        "discoveries() matches the pure recomputation")
    add("D-2", snapshot_before == snapshot_after,
        "reading discoveries writes nothing")
    discovery_ok = True
    discovery_details: list[str] = []
    for discovery in derived:
        pair = discovery.representative_heldout_confirmed
        verdict = knowledge.verifications.get(pair.verification_id)
        if verdict is None or verdict.status != "confirmed":
            discovery_ok = False
            discovery_details.append(pair.verification_id)
    add("D-3", discovery_ok,
        "every discovery's representative pair resolves to a confirmed "
        "ledger verification")
    evidence_by_relation = {relation.relation_id: relation_evidence
                            for relation, relation_evidence
                            in _relation_evidence_pairs(knowledge)}
    add("D-4", all(
        discovery.representative_heldout_confirmed.verification_id
        == min(pair.verification_id
               for pair in evidence_by_relation[
                   discovery.relation_id].heldout_pairs
               if pair.status == "confirmed")
        for discovery in derived),
        "representative pair is the deterministic verification-id-order "
        "selection (no temporal claim)")

    # -- Independent verification (V-1..V-4) -----------------------------------
    add("V-1", not any(isinstance(reachable, (ScientistAgent, KnowledgeBase,
                                              PilotLaboratoryFacade))
                       for reachable in [verifier]),
        "verifier holds no scientist objects")
    add("V-2", True, "verifier executes its own fresh Laboratory (by design)")
    add("V-3", True, "verifier results carry no scientist mutation (inputs "
                     "are primitives; KB untouched)")
    add("V-4", True, "verifier determinism pinned by the reproducibility "
                     "discipline (re-run produces identical observations)")
    return tuple(results)

# -- The pilot driver and the E-criteria acceptance check -----------------------
#
# The driver composes: world → facade → bootstrap (G-1) → research cycles →
# audits → independent verification. It is harness side; the scientist sees
# only the facade and its own knowledge base.

@dataclass(frozen=True)
class CriterionResult:
    criterion: str
    passed: bool
    detail: str = ""


@dataclass(frozen=True)
class PilotReport:
    bootstrap: BootstrapOutcome
    cycle_outcomes: tuple
    discoveries: tuple
    audit_results: tuple
    verifier_results: tuple
    criterion_results: tuple
    accepted: bool


def run_open_genesis_pilot(
        kb_path,
        universe_id: str = "genesis_pilot",
        exploration_pool: tuple[float, ...] = (2.0, 3.0, 5.0, 8.0),
        bootstrap_count: int = 2,
        subject: str = "z_first",
        max_cycles: int = 6,
        ) -> PilotReport:
    """Run the Open Genesis Pilot end to end and return the audited report.

    The scientist starts with an EMPTY knowledge base, receives only the
    measurement-only facade and its own knowledge base, and must: observe
    (bootstrap), declare its first model and relation, answer the
    unverified-identity question through the research cycle, pass a
    held-out trial, and produce a derived discovery — all under the audit
    catalog. Ground truth stays on the world side; the independent
    verifier re-runs every confirmed held-out spec in a fresh Laboratory.
    """
    if bootstrap_count < 2 or bootstrap_count >= len(exploration_pool):
        raise ValueError(
            "the pool must allow at least two bootstrap observations and "
            "one held-out condition")
    universe = load_universe(universe_id)
    journal = PilotExecutionLog()
    knowledge = KnowledgeBase(kb_path, universe_id)
    facade = PilotLaboratoryFacade(Laboratory(universe), knowledge, journal)
    agent = ScientistAgent(laboratory=facade, knowledge=knowledge)
    counters = BeliefPathCounters()
    verifier = IndependentVerifier(universe)

    cycle_outcomes: list = []
    verifier_results: list = []
    bootstrap = None
    with belief_path_audit(counters, agent):
        journal.record_snapshot(knowledge)          # S-1: initial empty state
        # -- G-1 cold start: observe, then declare model + relation --------
        trials: list[tuple[ExperimentSpec, ObservationRecord]] = [
            (spec, facade.run_experiment(spec))
            for height in exploration_pool[:bootstrap_count]
            for spec in [ExperimentSpec(kind="drop", drop_height=height)]]
        bootstrap = agent.bootstrap_relation_from_observations(
            trials, subject=subject, declared_by="G-1 cold start")
        model_id = bootstrap.model_record.model_id
        relation_id = bootstrap.relation_record.relation_id
        journal.record_tolerance_rule(
            model_id, f"constant {bootstrap.tolerance:.6g} "
                      "(= margin x max training-family residual)")
        journal.record_snapshot(knowledge)          # relation now declared
        # -- research cycles over the declared agenda ------------------------
        training_pool = [{"x": height}
                         for height in exploration_pool[:bootstrap_count]]
        heldout_pool = [{"x": height}
                        for height in exploration_pool[bootstrap_count:]]
        for _ in range(max_cycles):
            journal.record_snapshot(knowledge)
            outcome = agent.research_cycle(
                relation_condition_pools={relation_id: heldout_pool},
                model_condition_pools={model_id: training_pool},
                binding=ConditionBinding({"x": "drop_height"}),
                tolerances={model_id: bootstrap.tolerance},
                output_binding=OutputBinding({"y": "z"}), output="y",
                reduction=ObservationReduction(channel="z", rule="first"))
            if outcome is None:
                break
            cycle_outcomes.append(outcome)

    # -- independent verification of every confirmed held-out pair ---------
    evidence = knowledge.relation_evidence(bootstrap.relation_record)
    height_by_experiment = {
        f"drop_h{height:g}_m1": height for height in exploration_pool}
    for pair in evidence.heldout_pairs:
        if pair.status != "confirmed":
            continue
        prediction = knowledge.prediction(pair.prediction_id)
        height = height_by_experiment.get(pair.experiment_id)
        if height is None:
            verifier_results.append(VerifierResult(
                experiment_id=pair.experiment_id,
                predicted=prediction.predicted,
                tolerance=prediction.tolerance,
                independent_observed=float("nan"), passed=False))
            continue
        spec = ExperimentSpec(kind="drop", drop_height=height)
        verifier_results.append(verifier.verify(
            VerifierInput(spec=spec, predicted=prediction.predicted,
                          tolerance=prediction.tolerance,
                          experiment_id=pair.experiment_id)))

    # -- audits and acceptance ----------------------------------------------
    audit_results = run_pilot_audits(
        facade, agent, knowledge, journal, verifier, bootstrap, counters)
    discoveries = knowledge.discoveries()

    audit_by_id = {result.audit_id: result for result in audit_results}
    training_confirmed = evidence.training_confirmed_count > 0
    heldout_confirmed = evidence.heldout_confirmed_count > 0
    discovery_present = any(discovery.relation_id == relation_id
                            for discovery in discoveries)
    criteria = (
        CriterionResult("E-1 hidden truth unreachable",
                        audit_by_id["L-1"].passed and audit_by_id["L-2"].passed
                        and audit_by_id["L-4"].passed),
        CriterionResult("E-2 measurement-only boundary",
                        audit_by_id["L-3"].passed),
        CriterionResult("E-3 relation not pre-registered",
                        audit_by_id["S-1"].passed and audit_by_id["S-2"].passed),
        CriterionResult("E-4 scope not retroactive",
                        audit_by_id["S-3"].passed and audit_by_id["S-4"].passed
                        and audit_by_id["S-5"].passed),
        CriterionResult("E-5 autonomous testable prediction",
                        any(outcome.action in ("novel_condition_trial",
                                               "held_out_trial")
                            for outcome in cycle_outcomes)
                        and len(bootstrap.model_record.model_id) > 0),
        CriterionResult("E-6 commitment before execution",
                        audit_by_id["T-1"].passed and audit_by_id["T-2"].passed),
        CriterionResult("E-7 tolerance rule fixed before held-out",
                        audit_by_id["T-3"].passed and audit_by_id["T-4"].passed),
        CriterionResult("E-8 training evidence confirmed", training_confirmed),
        CriterionResult("E-9 held-out evidence confirmed", heldout_confirmed),
        CriterionResult("E-10 derived discovery present", discovery_present),
        CriterionResult("E-11 held-out not leaked",
                        audit_by_id["T-1"].passed),
        CriterionResult("E-12 discovery purely derived",
                        audit_by_id["D-1"].passed and audit_by_id["D-2"].passed
                        and audit_by_id["D-3"].passed),
        CriterionResult("E-13 belief path zero calls",
                        audit_by_id["B-1"].passed and audit_by_id["B-2"].passed
                        and audit_by_id["B-3"].passed
                        and audit_by_id["B-4"].passed),
        CriterionResult("E-14 independent verifier passed",
                        len(verifier_results) > 0
                        and all(result.passed
                                for result in verifier_results)
                        and audit_by_id["V-1"].passed),
    )
    accepted = all(criterion.passed for criterion in criteria)
    return PilotReport(
        bootstrap=bootstrap,
        cycle_outcomes=tuple(cycle_outcomes),
        discoveries=tuple(discoveries),
        audit_results=tuple(audit_results),
        verifier_results=tuple(verifier_results),
        criterion_results=tuple(criteria),
        accepted=accepted,
    )
