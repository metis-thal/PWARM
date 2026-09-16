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
from .experiment import ExperimentSession, ExperimentSpec, Laboratory, ObservationRecord
from .hypothesis import Hypothesis, Verification, fit_free_fall, verify
from .knowledge import KnowledgeBase, LawRecord
from .mission import Mission, MissionReport
from .planner import ExperimentPlanner

__all__ = [
    "ExperimentPlanner",
    "ExperimentSession",
    "ExperimentSpec",
    "Hypothesis",
    "KnowledgeBase",
    "Laboratory",
    "LawRecord",
    "Mission",
    "MissionReport",
    "ObservationRecord",
    "ScientistAgent",
    "Verification",
    "fit_free_fall",
    "verify",
]
