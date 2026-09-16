"""Mission definitions — research goals for the AI scientist.

A :class:`Mission` states which hidden parameter of the universe to establish
and what evidence counts as conclusive. A :class:`MissionReport` is what the
mission returns: status, estimates, and whether new knowledge was published.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Mission:
    """A research goal for the AI scientist."""

    id: str
    title: str
    objective: str
    target: str                        # claim to establish, e.g. "gravity"
    unit: str = ""
    min_confidence: float = 0.999      # mission acceptance threshold
    max_rel_spread: float = 0.001      # cross-experiment agreement bound


@dataclass
class MissionReport:
    """Outcome of running a mission."""

    mission_id: str
    status: str            # CONCLUDED_FROM_KNOWLEDGE | DISCOVERED | INCOMPLETE
    experiments_run: int = 0
    estimates: list[float] = field(default_factory=list)
    value: float | None = None
    unit: str = ""
    confidence: float = 0.0
    formula: str = ""
    knowledge_saved: bool = False
    summary: str = ""

    def __str__(self) -> str:
        lines = [f"Mission {self.mission_id}: {self.status}"]
        if self.value is not None:
            lines.append(f"  {self.formula}")
            unit = f" {self.unit}" if self.unit else ""
            lines.append(f"  value = {self.value:.6f}{unit}"
                         f"  confidence = {self.confidence * 100:.2f}%")
        lines.append(f"  experiments run: {self.experiments_run}"
                     f"  knowledge saved: {self.knowledge_saved}")
        if self.summary:
            lines.append(f"  {self.summary}")
        return "\n".join(lines)
