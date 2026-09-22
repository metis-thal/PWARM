"""AI Scientist Layer — missions, experiments, hypotheses, knowledge, planning.

The scientist layer turns pymo from a physics simulator into an artificial
scientist: it proposes experiments, observes outcomes through a strictly
measurement-only channel, derives claims, cross-verifies them across
independent experiments, and accumulates civilization knowledge on disk.

Core rule (structurally enforced): the physics engine is absolute truth; the
AI scientist is a learned approximation that never reads universe secrets —
it can only propose experiments and observe their results.
"""

from .agent import ScientistAgent
from .budget import ExperimentBudget
from .designer import ExperimentDesigner, ExperimentProposal
from .experiment import ExperimentSession, ExperimentSpec, Laboratory, ObservationRecord
from .experiments import REGISTRY
from .hypothesis import Hypothesis, Verification, fit_free_fall, verify
from .instrument import InstrumentCatalog, InstrumentGrant, InstrumentRequest, analyze_gap
from .knowledge import KnowledgeBase, LawRecord
from .mission import Mission, MissionReport
from .planner import ExperimentPlanner
from .prediction import (
    ComparisonInput,
    ConditionBinding,
    ConditionComparison,
    OutputBinding,
    ScientificModel,
    comparison_input,
    disagreement,
    model_prediction,
    rank_discriminating_conditions,
)
from .state import ScientistState
from .uncertainty import certainty_class, relative_width, total_uncertainty

__all__ = [
    "REGISTRY",
    "ComparisonInput",
    "ConditionBinding",
    "ConditionComparison",
    "ExperimentBudget",
    "ExperimentDesigner",
    "ExperimentPlanner",
    "ExperimentProposal",
    "ExperimentSession",
    "ExperimentSpec",
    "Hypothesis",
    "InstrumentCatalog",
    "InstrumentGrant",
    "InstrumentRequest",
    "KnowledgeBase",
    "Laboratory",
    "LawRecord",
    "Mission",
    "MissionReport",
    "ObservationRecord",
    "OutputBinding",
    "ScientificModel",
    "ScientistAgent",
    "ScientistState",
    "Verification",
    "analyze_gap",
    "certainty_class",
    "comparison_input",
    "disagreement",
    "fit_free_fall",
    "model_prediction",
    "rank_discriminating_conditions",
    "relative_width",
    "total_uncertainty",
    "verify",
]
