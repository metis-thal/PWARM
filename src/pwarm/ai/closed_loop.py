"""Closed-loop AI reasoning: observe -> discover law -> predict -> compare.

This implements the Phase 1 core loop:
    world simulation -> observer collects data -> symbolic regression discovers a
    law -> the AI predicts future states -> error measured against the physics
    kernel's ground truth.

The physics kernel is the ABSOLUTE truth; the AI's discovered law is a LEARNED
model. Prediction error measures how well the AI has learned the world.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .law_discovery import DiscoveredLaw, LawDiscovery
from .observer import TimeSeriesDataset


@dataclass
class PredictionResult:
    """Result of the closed-loop prediction for one observed quantity."""

    quantity: str
    law: DiscoveredLaw
    t_train: np.ndarray
    y_train: np.ndarray
    t_test: np.ndarray
    y_true: np.ndarray          # ground truth from physics kernel
    y_pred: np.ndarray          # AI prediction from discovered law
    relative_error: float       # RMSE / (range of y) on the test window

    def passed(self, threshold: float = 0.05) -> bool:
        return self.relative_error < threshold


class ClosedLoopAI:
    """Runs the observe -> discover -> predict -> compare loop."""

    def __init__(self, discovery: LawDiscovery | None = None, train_fraction: float = 0.6):
        self.discovery = discovery or LawDiscovery()
        self.train_fraction = train_fraction
        self.results: list[PredictionResult] = []

    def run(
        self, dataset: TimeSeriesDataset, quantities: list[str] | None = None
    ) -> list[PredictionResult]:
        """For each quantity, discover its law on a training window, then predict
        the held-out test window and measure error vs ground truth."""
        self.results.clear()
        t = dataset.time()
        if quantities is None:
            quantities = [o.name for o in dataset.observations if o.name != "time"]

        n_train = int(len(t) * self.train_fraction)
        t_train, t_test = t[:n_train], t[n_train:]

        for q in quantities:
            y = dataset.get(q)
            if y is None or len(y) < 10:
                continue
            y = np.asarray(y, dtype=float).ravel()
            law = self.discovery.discover(t_train, y[:n_train])
            y_pred = law.predict(t_test)
            y_true = y[n_train:]
            rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
            scale = float(np.ptp(y_true)) if np.ptp(y_true) > 0 else 1.0
            rel_err = rmse / max(scale, 1e-12)
            self.results.append(
                PredictionResult(
                    quantity=q,
                    law=law,
                    t_train=t_train,
                    y_train=y[:n_train],
                    t_test=t_test,
                    y_true=y_true,
                    y_pred=y_pred,
                    relative_error=rel_err,
                )
            )
        return self.results

    def summary(self) -> str:
        lines = ["Closed-loop AI discovery results:"]
        for r in self.results:
            mark = "PASS" if r.passed() else "FAIL"
            lines.append(
                f"  {r.quantity:<16} R2={r.law.r2:7.4f} "
                f"test-rel-err={r.relative_error:8.4f} [{mark}]  {r.law.expression}"
            )
        return "\n".join(lines)
