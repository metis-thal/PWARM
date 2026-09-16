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

from .experiment import ExperimentSpec, Laboratory, ObservationRecord
from .hypothesis import Hypothesis, Verification, fit_free_fall, verify
from .knowledge import KnowledgeBase
from .mission import Mission, MissionReport
from .planner import ExperimentPlanner


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


_MIN_FIT_SAMPLES = 5
