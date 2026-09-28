"""Experiment planning — the AI decides WHAT to test next.

The planner encodes experimental design (strategy), not physics: it chooses
conditions (drop heights, masses) that make a law testable and checks the
knowledge base so established laws are not re-discovered. This is the
"AI proposes, physics executes" boundary — planning is about information
gain, never about touching the simulation directly.
"""

from __future__ import annotations

from .experiment import ExperimentSpec
from .knowledge import KnowledgeBase
from .mission import Mission


class ExperimentPlanner:
    """Designs drop experiments for gravity-type missions."""

    def __init__(self, drop_heights: tuple[float, ...] = (10.0, 20.0, 5.0),
                 mass: float = 1.0):
        self.drop_heights = tuple(drop_heights)
        self.mass = mass

    def propose(self, mission: Mission,
                knowledge: KnowledgeBase) -> list[ExperimentSpec]:
        """Return the experiment plan for a mission (empty = nothing to do).

        * Knowledge check: if the claim is already established with
          sufficient confidence, propose nothing — civilization accumulates,
          it does not re-discover.
        * Design: several DIFFERENT drop heights. A law that survives
          independent conditions is a law; a single fit is an anecdote.
        """
        if knowledge.knows(mission.target, mission.min_confidence):
            return []
        if mission.target != "gravity":
            return []      # planner v1 only designs drop experiments
        return [
            ExperimentSpec(kind="drop", drop_height=height, mass=self.mass)
            for height in self.drop_heights
        ]
