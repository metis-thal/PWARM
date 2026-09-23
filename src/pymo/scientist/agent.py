"""ScientistAgent — the AI scientist's body.

The agent holds a ``Laboratory`` (execute + observe), a ``KnowledgeBase``
(memory), a planner (experimental design) and a discovery backend (fitting).
It NEVER touches the universe's hidden parameters: its loop is the scientific
method itself.

    plan -> experiment -> observe -> hypothesis -> cross-verify -> publish

Core rule (structurally enforced): the AI proposes experiments, physics
executes them, the AI observes results. The agent's only data channel is
:class:`ObservationRecord` — measurements, nothing else.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .budget import ExperimentBudget
from .designer import ExperimentDesigner
from .experiment import ExperimentSpec, Laboratory, ObservationRecord
from .experiments import REGISTRY
from .hypothesis import Hypothesis, Verification, fit_free_fall, verify
from .instrument import InstrumentCatalog, analyze_gap
from .knowledge import KnowledgeBase
from .mission import Mission, MissionReport
from .planner import ExperimentPlanner
from .prediction import (
    ConditionBinding,
    ConditionComparison,
    Prediction,
    PredictionRecord,
    ScientificModel,
    VerificationRecord,
    adjudicate,
    model_prediction,
    prediction_from_belief,
    rank_discriminating_conditions,
)
from .state import ScientistState


def _evidence(records: list[ObservationRecord],
              hypotheses: list[Hypothesis]) -> tuple[list[dict], list[dict]]:
    """AI-side evidence for a MissionReport: measurement records + fits."""
    observations = [
        {"experiment_id": r.experiment_id,
             "t": [round(float(v), 6) for v in r.t],
             "z": [round(float(v), 6) for v in r.z],
             "vx": ([round(float(v), 6) for v in r.vx] if r.vx is not None else None)}
        for r in records
    ]
    hyps = [{"claim": h.claim, "value": h.value, "r2": h.r2,
                 "formula": h.formula, "experiment_id": h.experiment_id}
            for h in hypotheses]
    return observations, hyps

# Proposal -> ExperimentSpec translation (Model Competition Step 4): the
# AI-side half of crossing the boundary. ExperimentSpec is the existing
# AI->physics contract carrier, so no new channel is invented and the
# physics side needs no knowledge of AI modules.

_SPEC_CONDITIONS = frozenset(
    {"drop_height", "mass", "radius", "v0"})   # numeric spec parameters


def proposal_to_spec(proposal: ConditionComparison,
                     kind: str,
                     binding: ConditionBinding | None = None) -> ExperimentSpec:
    """Translate an AI-side proposal into the standard experiment contract.

    Conditions must end up naming numeric ExperimentSpec parameters, via
    one of two honest paths: the proposal already uses spec vocabulary,
    or an explicit :class:`ConditionBinding` renames the model's condition
    variables into it. Unknown keys are refused instead of silently
    ignored — a condition either drives the experiment or the proposal
    is rejected.
    """
    conditions = dict(proposal.conditions)
    if binding is not None:
        conditions = binding.translate(conditions)
    unknown = set(conditions) - _SPEC_CONDITIONS
    if unknown:
        raise ValueError(
            f"proposal conditions {sorted(unknown)} are not ExperimentSpec "
            f"parameters; valid: {sorted(_SPEC_CONDITIONS)}")
    return ExperimentSpec(kind=kind, **conditions)


class ScientistAgent:
    """An autonomous scientist working in an unknown universe."""

    def __init__(self, laboratory: Laboratory, knowledge: KnowledgeBase,
                 planner: ExperimentPlanner | None = None):
        self.laboratory = laboratory
        self.knowledge = knowledge
        self.planner = planner or ExperimentPlanner()
        # Genesis Phase 1: the last guesses, their committed (hashed,
        # persisted) records, and the verification verdicts.
        self.last_predictions: list[Prediction] = []
        self.last_committed_predictions: list[PredictionRecord] = []
        self.last_verifications: list[VerificationRecord] = []
        self.last_state: ScientistState | None = None

    # -- faculties -----------------------------------------------------------

    def create_experiment(self, mission: Mission) -> list[ExperimentSpec]:
        """Decide what to test next (consults planner + knowledge)."""
        return self.planner.propose(mission, self.knowledge)

    def observe(self, record: ObservationRecord) -> dict:
        """Perceive one observation record (the AI's only sensory input)."""
        if len(record.t) == 0:
            return {"experiment": record.experiment_id, "samples": 0}
        return {
            "experiment": record.experiment_id,
            "samples": len(record.t),
            "t_span": (float(record.t[0]), float(record.t[-1])),
            "z_span": (float(record.z.min()), float(record.z.max())),
        }

    def form_hypothesis(self, records: list[ObservationRecord],
                        claim: str = "gravity",
                        unit: str = "m/s^2") -> list[Hypothesis]:
        """Fit each observation record and derive a claim per experiment."""
        hypotheses: list[Hypothesis] = []
        for record in records:
            if len(record.t) < _MIN_FIT_SAMPLES:
                continue
            _, hypothesis = fit_free_fall(record)
            hypotheses.append(hypothesis)
        return hypotheses

    def verify(self, hypotheses: list[Hypothesis]) -> Verification | None:
        """Cross-experiment verification (None when there is nothing to verify)."""
        return verify(hypotheses) if hypotheses else None

    # -- Genesis Phase 1: predict + commit BEFORE observing -------------------

    def form_prediction(self, mission: Mission,
                        state: ScientistState | None = None) -> list[Prediction]:
        """The AI's commitment about what the experiment will show, formed
        BEFORE any experiment executes. The guess is translated from the
        scientist's OWN gravity belief — its self-model (Genesis Phase 2);
        no observation data participates, and the ordering guarantee stays
        structural: run_mission calls this before run_experiment.
        """
        if mission.target != "gravity":
            return []
        if state is None:
            # Minimal self-model: one belief for the mission's target claim,
            # built from AI-visible vocabulary only (claim name + knowledge).
            state = ScientistState((mission.target,), self.knowledge)
        belief = state.belief("gravity")
        if belief is None or belief.status == "unidentifiable":
            return []     # honest metacognition: no belief to commit
        return [prediction_from_belief(belief)]

    def commit_predictions(self, guesses: list[Prediction],
                           specs: list[ExperimentSpec],
                           ) -> list[PredictionRecord]:
        """Hash and persist each commitment — BEFORE physics runs. From
        here on the predictions are immutable scientific history: a changed
        mind commits a NEW prediction, never an edit.
        """
        spec_ref = ", ".join(spec.id for spec in specs)
        return [self.knowledge.commit_prediction(
                    model_ref=guess.source, claim=guess.claim,
                    spec_ref=spec_ref, value=guess.value,
                    tolerance=guess.tolerance)
                for guess in guesses]

    def verify_predictions(self, committed: list[PredictionRecord],
                           records: list[ObservationRecord],
                           ) -> list[VerificationRecord]:
        """Adjudicate the committed predictions against the NEW observations:
        hash-check each commitment, fit the record AI-side, and persist one
        VerificationRecord per (prediction, record). Confirmed and refuted
        are both first-class outcomes — history is never overwritten.
        """
        verifications: list[VerificationRecord] = []
        for prediction in committed:
            for record in records:
                outcome = adjudicate(prediction, record)
                verifications.append(self.knowledge.record_verification(
                    prediction.prediction_id, record.experiment_id, outcome))
        return verifications

    def learn_from_verifications(self, state: ScientistState,
                                 committed: list[PredictionRecord],
                                 verifications: list[VerificationRecord],
                                 ) -> None:
        """Genesis Phase 2 Step 2: fold each verification verdict back into
        the self-model (confirmed tightens the belief, refuted widens it).
        Deterministic and AI-side only — the knowledge base is untouched."""
        by_id = {p.prediction_id: p for p in committed}
        for verification in verifications:
            prediction = by_id.get(verification.prediction_id)
            if prediction is not None:
                state.learn_from_verification(
                    claim=prediction.claim, status=verification.status,
                    observed=verification.observed,
                    predicted=prediction.predicted,
                    tolerance=prediction.tolerance)

    # -- Model Competition Step 3: the AI proposes, physics will execute ----

    def propose_discriminating_experiment(
            self, models: Sequence[ScientificModel],
            candidate_conditions: Sequence[dict[str, float]],
    ) -> ConditionComparison | None:
        """The AI's experiment SUGGESTION from model competition: the
        candidate condition under which the rival models diverge most.

        This stays on the AI side of the boundary — nothing runs here, no
        commitment is made, no belief changes. The proposal becomes an
        observation only when a later step takes it through the
        Laboratory. Returns None honestly when no candidate can separate
        the models (all disagreements zero, or no candidates given).
        """
        ranked = rank_discriminating_conditions(models, candidate_conditions)
        if not ranked or ranked[0].disagreement == 0.0:
            return None
        return ranked[0]

    def execute_proposal(self, proposal: ConditionComparison,
                         kind: str,
                         binding: ConditionBinding | None = None,
                         ) -> ObservationRecord:
        """Cross the boundary ONCE for this proposal: translate it into
        the standard ExperimentSpec contract (optionally through an
        explicit condition binding) and let the Laboratory run it.
        Physics executes exactly one experiment and returns its
        ObservationRecord — nothing more happens in this step: no model
        verdict, no belief update, no knowledge write.
        """
        spec = proposal_to_spec(proposal, kind, binding)
        return self.laboratory.run_experiment(spec)

    # -- Genesis Step 6: competition prediction commitment -------------------

    def commit_discriminating_predictions(
            self, models: Sequence[ScientificModel],
            proposal: ConditionComparison,
            kind: str,
            binding: ConditionBinding | None = None,
            tolerances: Mapping[str, float] | None = None,
            ) -> list[PredictionRecord]:
        """Commit EVERY rival model's prediction under the winning
        condition, BEFORE the experiment runs.

        Reuses the existing commitment machinery (commit_predictions ->
        KnowledgeBase.commit_prediction): one PredictionRecord per model,
        each with its own id, sequence index and hash, all pointing at
        the SAME experiment spec. Nothing runs here — no Laboratory call,
        no verification, no belief change; execution and adjudication
        are later steps.
        """
        spec = proposal_to_spec(proposal, kind, binding)
        conditions = dict(proposal.conditions)
        per_model = tolerances or {}
        guesses = [model_prediction(model, conditions,
                                    per_model.get(model.model_id, 1.0))
                   for model in models]
        return self.commit_predictions(guesses, [spec])

    # -- the mission loop ------------------------------------------------------

    def run_mission(self, mission: Mission,
                    state: ScientistState | None = None) -> MissionReport:
        """Execute the full scientific method for one mission.

        ``state`` optionally supplies the AI's self-model; the prediction
        step reads its gravity belief (a minimal one-belief self-model is
        built from the mission target when omitted).
        """
        specs = self.create_experiment(mission)
        if not specs:
            law = self.knowledge.get(mission.target)
            if law is not None:
                return MissionReport(
                    mission_id=mission.id,
                    status="CONCLUDED_FROM_KNOWLEDGE",
                    value=law.value,
                    unit=law.unit,
                    confidence=law.confidence,
                    formula=law.formula,
                    summary=(f"'{mission.target}' already established — "
                             "civilization knowledge, no re-discovery needed"),
                )
            return MissionReport(
                mission_id=mission.id,
                status="INCOMPLETE",
                summary="planner produced no experiments for this mission",
            )

        # Genesis Phase 2: the mission runs against the AI's self-model —
        # predictions are drawn from its beliefs, verifications feed back.
        if state is None:
            state = ScientistState((mission.target,), self.knowledge)
        self.last_state = state

        # Genesis Phase 1: predict, then COMMIT (hash + persist) BEFORE any
        # experiment executes — the result must not be available when the
        # prediction is formed.
        guesses = self.form_prediction(mission, state)
        self.last_predictions = list(guesses)
        committed = self.commit_predictions(guesses, specs)
        self.last_committed_predictions = list(committed)

        # 1-2. propose + execute (physics runs the world; AI only observes)
        records: list[ObservationRecord] = []
        for spec in specs:
            records.append(self.laboratory.run_experiment(spec))

        # Genesis Phase 1: verify the committed predictions against the new
        # observations (committed hash + record only — never engine truth).
        self.last_verifications = self.verify_predictions(committed, records)
        # Verification resolves statuses in the store (records are replaced,
        # never mutated in place) — refresh the agent's view of its history.
        self.last_committed_predictions = [
            self.knowledge.prediction(p.prediction_id) for p in committed]
        # Genesis Phase 2 Step 2: verification feeds back into the beliefs —
        # confirmed tightens, refuted widens. Cognition only, never law.
        self.learn_from_verifications(state, committed, self.last_verifications)

        # 3-4. observe + hypothesize
        for record in records:
            self.observe(record)
        hypotheses = self.form_hypothesis(records, claim=mission.target,
                                          unit=mission.unit)
        if not hypotheses:
            return MissionReport(
                mission_id=mission.id,
                status="INCOMPLETE",
                experiments_run=len(records),
                summary="observations insufficient to form a hypothesis",
            )

        # 5. cross-verify across independent experiments
        verification = self.verify(hypotheses)
        concluded = bool(
            verification is not None
            and verification.stable
            and verification.confidence >= mission.min_confidence
        )
        observations, hyp_dicts = _evidence(records, hypotheses)
        report = MissionReport(
            mission_id=mission.id,
            status="DISCOVERED" if concluded else "INCOMPLETE",
            experiments_run=len(records),
            observations=observations,
            hypotheses=hyp_dicts,
            estimates=[h.value for h in hypotheses],
            value=verification.mean if verification is not None else None,
            unit=mission.unit,
            confidence=verification.confidence if verification is not None else 0.0,
            formula=hypotheses[0].formula,
            summary=(
                f"{len(records)} independent experiments agree: "
                f"{verification.rel_spread * 100:.4f}% spread"
                if verification is not None else ""
            ),
        )

        # 6. publish to the knowledge base (civilization accumulates)
        if concluded and verification is not None:
            self.knowledge.record_law(
                name=mission.target,
                formula=hypotheses[0].formula,
                value=verification.mean,
                unit=mission.unit,
                confidence=verification.confidence,
                r2=verification.mean_r2,
                experiments=[r.experiment_id for r in records],
                derived_by=hypotheses[0].source,
            )
            self.knowledge.save()
            report.knowledge_saved = True
        return report

    # -- Mission 002: adaptive experimentation --------------------------------

    def run_adaptive_mission(self, mission: Mission, state: ScientistState,
                             designer: ExperimentDesigner | None = None,
                             max_experiments: int = 12) -> MissionReport:
        """The Mission 002 loop: choose -> execute -> observe -> update.

        Unlike run_mission's fixed plan, the designer picks each experiment
        by expected information gain against the live ScientistState; the
        loop ends when no design clears the minimum gain. Unreachable
        unknowns are then marked unidentifiable (honest metacognition: the
        scientist reports WHAT it cannot know and WHY), and established
        material properties are published to the knowledge base.
        """
        designer = designer or ExperimentDesigner()
        hypotheses: list[Hypothesis] = []
        records: list[ObservationRecord] = []
        proposal_log: list[str] = []
        experiments_run = 0

        for _ in range(max_experiments):
            proposal = designer.choose(state)
            if proposal is None:
                break
            proposal_log.append(proposal.reason)

            spec = ExperimentSpec(
                kind=proposal.kind,
                drop_height=float(proposal.design.get("height", 10.0)),
                v0=proposal.design.get("v0"),
                material=proposal.design.get("material"),
            )
            record = self.laboratory.run_experiment(spec)
            records.append(record)
            experiments_run += 1
            self.observe(record)

            g_known = None
            gravity_belief = state.belief("gravity")
            if gravity_belief is not None and gravity_belief.status == "known":
                g_known = gravity_belief.midpoint
            module = REGISTRY[proposal.kind]
            hypothesis, fitted_g = module.derive(
                record, g_known, proposal.design.get("material", "material"))
            hypotheses.append(hypothesis)
            # Update by the hypothesis's OWN claim (a gravity-targeted drop
            # still yields a restitution hypothesis), then gravity from the
            # free-fall byproduct — one experiment, two claims advanced.
            state.update(hypothesis.claim, hypothesis.value, proposal.rel_resolution)
            if (fitted_g is not None and "gravity" in state.beliefs
                    and state.beliefs["gravity"].status != "known"):
                state.update("gravity", fitted_g, proposal.rel_resolution)

        # Honest metacognition: unknowns with no informing experiment.
        for claim in designer.unreachable_claims(state):
            state.mark_unidentifiable(
                claim, "no available experiment informs this parameter "
                       "(identifiability limit of the current apparatus)")

        # Publish everything established to the knowledge base.
        published, confidence_best = self.publish_established(state, hypotheses)

        summary = state.report()
        if proposal_log:
            summary += "\n\nexperiment proposals:\n" + "\n".join(
                f"  {i + 1}. {reason}" for i, reason in enumerate(proposal_log))

        observations, hyp_dicts = _evidence(records, hypotheses)
        return MissionReport(
            mission_id=mission.id,
            status="DISCOVERED" if published else "INCOMPLETE",
            experiments_run=experiments_run,
            estimates=[h.value for h in hypotheses],
            confidence=confidence_best,
            formula="material properties" if published else "",
            knowledge_saved=published > 0,
            summary=summary,
            observations=observations,
            hypotheses=hyp_dicts,
        )

    # -- Mission 003: science under constraints --------------------------------

    def run_constrained_mission(self, mission: Mission, state: ScientistState,
                                budget: ExperimentBudget,
                                designer: ExperimentDesigner | None = None,
                                catalog: InstrumentCatalog | None = None,
                                max_experiments: int = 16) -> MissionReport:
        """The Mission 003 loop: value-ranked choices under a live budget.

        Two differences from :meth:`run_adaptive_mission`:

        * Every candidate design has a COST; the designer ranks by value
          (expected utility / cost) and only affordable designs are chosen.
          Each executed experiment DEBITS the budget — experiments, world
          ticks and cost units all count.
        * When the apparatus cannot inform a claim at all, the claim is
          marked unidentifiable, the gap analysis names the missing
          capability, and an INSTRUMENT REQUEST is filed. If the universe's
          instrument catalog grants it, the budget is EXTENDED (only the
          grant path may do this — never the scientist), the new experiment
          kind is unlocked, and the loop CONTINUES: the claim gets a second
          chance with the new instrument.

        ``budget_exhausted`` is the honest outcome when candidates exist but
        cannot be afforded — reported as out-of-resources, never confused
        with unidentifiability.
        """
        designer = designer or ExperimentDesigner()
        hypotheses: list[Hypothesis] = []
        records: list[ObservationRecord] = []
        proposal_log: list[str] = []
        instrument_log: list[str] = []
        experiments_run = 0
        budget_exhausted = False

        for _ in range(max_experiments):
            proposal = designer.choose(state, budget)
            if proposal is None:
                if designer.available_experiments(state):
                    budget_exhausted = True
                    break
                # Honest metacognition: unknowns no enabled design informs.
                for claim in designer.unreachable_claims(state):
                    state.mark_unidentifiable(
                        claim, "no available experiment informs this parameter "
                               "(identifiability limit of the current apparatus)")
                request = analyze_gap(state) if catalog is not None else None
                if request is None:
                    break
                grant = catalog.grant(request)
                if grant is None:
                    break   # no such instrument (or already granted): stay honest
                budget.extend(**{key: grant.budget.get(key, 0)
                                 for key in ("experiments", "simulation_steps",
                                             "compute_cost")})
                designer.enable_kind(grant.enables)
                for claim in request.target_claims:
                    state.revive(claim)
                instrument_log.append(
                    f"requested {request.instrument} ({request.capability}) "
                    f"-> GRANTED; unlocks {grant.enables} for "
                    f"{', '.join(request.target_claims)}")
                continue

            proposal_log.append(proposal.reason)
            spec = ExperimentSpec(
                kind=proposal.kind,
                drop_height=float(proposal.design.get(
                    "height", proposal.design.get("depth", 10.0))),
                v0=proposal.design.get("v0"),
                material=proposal.design.get("material"),
            )
            record = self.laboratory.run_experiment(spec)
            records.append(record)
            budget.spend(proposal.cost, record.steps)
            experiments_run += 1
            self.observe(record)

            g_known = None
            gravity_belief = state.belief("gravity")
            if gravity_belief is not None and gravity_belief.status == "known":
                g_known = gravity_belief.midpoint
            module = REGISTRY[proposal.kind]
            hypothesis, fitted_g = module.derive(
                record, g_known, proposal.design.get("material", "material"))
            hypotheses.append(hypothesis)
            # One experiment can advance two claims: its own target plus the
            # free-fall gravity byproduct (drop and buoyancy both fall).
            state.update(hypothesis.claim, hypothesis.value,
                         proposal.rel_resolution)
            if (fitted_g is not None and "gravity" in state.beliefs
                    and state.beliefs["gravity"].status != "known"):
                state.update("gravity", fitted_g, proposal.rel_resolution)

        if not budget_exhausted:
            # Max-experiment cutoff (or a finished arc): anything still
            # uninformed by the enabled apparatus stays honestly marked.
            for claim in designer.unreachable_claims(state):
                state.mark_unidentifiable(
                    claim, "no available experiment informs this parameter "
                           "(identifiability limit of the current apparatus)")

        published, confidence_best = self.publish_established(state, hypotheses)

        summary = state.report()
        summary += "\n\nbudget:\n" + "\n".join(budget.report_lines())
        if instrument_log:
            summary += "\n\ninstrument arc:\n" + "\n".join(
                f"  {line}" for line in instrument_log)
        if budget_exhausted:
            summary += ("\n\nBUDGET EXHAUSTED: informative candidates remain "
                        "but cannot be afforded.")
        if proposal_log:
            summary += "\n\nexperiment proposals:\n" + "\n".join(
                f"  {i + 1}. {reason}" for i, reason in enumerate(proposal_log))

        observations, hyp_dicts = _evidence(records, hypotheses)
        return MissionReport(
            mission_id=mission.id,
            status="DISCOVERED" if published else "INCOMPLETE",
            experiments_run=experiments_run,
            estimates=[h.value for h in hypotheses],
            confidence=confidence_best,
            formula="material properties" if published else "",
            knowledge_saved=published > 0,
            summary=summary,
            observations=observations,
            hypotheses=hyp_dicts,
        )

    # -- publication -----------------------------------------------------------

    def publish_established(self, state: ScientistState,
                             hypotheses: list[Hypothesis]) -> tuple[int, float]:
        """Publish established material properties + gravity to the
        knowledge base. Returns (entries published, best confidence)."""
        published = 0
        confidence_best = 0.0
        materials = sorted({n.split(".", 1)[0] for n in state.beliefs if "." in n})
        for material in materials:
            props: dict[str, float] = {}
            widths: list[float] = []
            for name, belief in state.beliefs.items():
                if not name.startswith(material + "."):
                    continue
                if belief.status == "known":
                    props[name.split(".", 1)[1]] = round(belief.midpoint, 6)
                    widths.append((belief.hi - belief.lo)
                                  / max(abs(belief.midpoint), 1e-9))
            if props:
                confidence = max(0.0, 1.0 - sum(widths) / len(widths))
                confidence_best = max(confidence_best, confidence)
                self.knowledge.record_material(
                    material, props, confidence, 1.0,
                    [h.experiment_id for h in hypotheses],
                    derived_by="adaptive-designer",
                )
                published += 1
        # Publish gravity too if THIS mission established it (a fresh
        # universe's knowledge would otherwise force a re-measurement).
        gravity_belief = state.belief("gravity")
        if (gravity_belief is not None and gravity_belief.status == "known"
                and self.knowledge.get("gravity") is None):
            rel_width = ((gravity_belief.hi - gravity_belief.lo)
                         / max(abs(gravity_belief.midpoint), 1e-9))
            self.knowledge.record_law(
                name="gravity",
                formula=f"g = {gravity_belief.midpoint:.4f} m/s^2 (free-fall fit)",
                value=gravity_belief.midpoint,
                unit="m/s^2",
                confidence=max(0.0, 1.0 - rel_width),
                r2=1.0,
                experiments=[h.experiment_id for h in hypotheses[:1]],
                derived_by="adaptive-designer",
            )
            published += 1
        if published:
            self.knowledge.save()
        return published, confidence_best


_MIN_FIT_SAMPLES = 5
