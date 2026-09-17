"""Experiment budget — science under constraints.

Real research never has infinite resources. :class:`ExperimentBudget` gives
the AI scientist three limited currencies:

    experiments        how many experiments may be RUN at all
    simulation_steps   total world ticks the laboratory may consume
    compute_cost       total cost units (setup + instrument time per design)

The designer must respect all three: a candidate that cannot be afforded is
not a candidate. Budgets can be ``extend``-ed — but only by an external
grant (Mission 003's instrument director), never by the scientist itself.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ExperimentBudget:
    """The mission's resource envelope and its consumption ledger."""

    experiments: int            # max number of experiments
    simulation_steps: int       # max total world ticks
    compute_cost: float         # max total cost units

    used_experiments: int = 0
    used_steps: int = 0
    used_cost: float = 0.0

    def can_afford(self, cost: float) -> bool:
        """Whether one more experiment costing ``cost`` units still fits."""
        return (self.used_experiments < self.experiments
                and self.used_steps < self.simulation_steps
                and self.used_cost + cost <= self.compute_cost + 1e-9)

    def spend(self, cost: float, steps: int = 0) -> None:
        """Record one executed experiment's consumption."""
        self.used_experiments += 1
        self.used_steps += int(steps)
        self.used_cost += float(cost)

    def extend(self, experiments: int = 0, simulation_steps: int = 0,
               compute_cost: float = 0.0) -> None:
        """Accept an external grant (instrument award). The scientist cannot
        call this itself — only the grant path in :mod:`instrument` does."""
        self.experiments += int(experiments)
        self.simulation_steps += int(simulation_steps)
        self.compute_cost += float(compute_cost)

    @property
    def exhausted(self) -> bool:
        return not (self.used_experiments < self.experiments
                    and self.used_steps < self.simulation_steps
                    and self.used_cost < self.compute_cost)

    def report_lines(self) -> list[str]:
        """Ledger lines for the dashboard's EXPERIMENT BUDGET panel."""
        exp_bar = _bar(self.used_experiments, self.experiments)
        step_bar = _bar(self.used_steps, self.simulation_steps)
        cost_bar = _bar(self.used_cost, self.compute_cost)
        return [
            f"  experiments {exp_bar} {self.used_experiments}/{self.experiments}",
            f"  steps       {step_bar} {self.used_steps}/{self.simulation_steps}",
            f"  cost        {cost_bar} {self.used_cost:.1f}/{self.compute_cost:.1f}",
        ]


def _bar(used: float, total: float, width: int = 10) -> str:
    if total <= 0:
        return "[" + "?" * width + "]"
    filled = round(width * min(1.0, used / total))
    return "[" + "#" * filled + "." * (width - filled) + "]"
