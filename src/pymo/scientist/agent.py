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

from .designer import ExperimentDesigner
from .experiment import ExperimentSpec, Laboratory, ObservationRecord
from .experiments import REGISTRY
from .hypothesis import Hypothesis, Verification, fit_free_fall, verify
from .knowledge import KnowledgeBase
from .mission import Mission, MissionReport
from .planner import ExperimentPlanner
from .state import ScientistState


class ScientistAgent:
    """An autonomous scientist working in an unknown universe."""

    def __init__(self, laboratory: Laboratory, knowledge: KnowledgeBase,
                 planner: ExperimentPlanner | None = None):
        self.laboratory = laboratory
        self.knowledge = knowledge
        self.planner = planner or ExperimentPlanner()

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

    # -- the mission loop ------------------------------------------------------

    def run_mission(self, mission: Mission) -> MissionReport:
        """Execute the full scientific method for one mission."""
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

        # 1-2. propose + execute (physics runs the world; AI only observes)
        records: list[ObservationRecord] = []
        for spec in specs:
            records.append(self.laboratory.run_experiment(spec))

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
        report = MissionReport(
            mission_id=mission.id,
            status="DISCOVERED" if concluded else "INCOMPLETE",
            experiments_run=len(records),
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

        # Publish established material properties to the knowledge base.
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

        summary = state.report()
        if proposal_log:
            summary += "\n\nexperiment proposals:\n" + "\n".join(
                f"  {i + 1}. {reason}" for i, reason in enumerate(proposal_log))

        return MissionReport(
            mission_id=mission.id,
            status="DISCOVERED" if published else "INCOMPLETE",
            experiments_run=experiments_run,
            estimates=[h.value for h in hypotheses],
            confidence=confidence_best,
            formula="material properties" if published else "",
            knowledge_saved=published > 0,
            summary=summary,
        )


_MIN_FIT_SAMPLES = 5
