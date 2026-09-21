"""Knowledge base — the AI scientist's accumulated civilization knowledge.

Laws established in past missions persist as JSON on disk so future missions
do not re-discover what is already known. This file is the AI's possession:
it is written by the scientist layer and read by the planner; it never
contains universe secrets, only what the AI derived from observations.

Genesis Phase 1 adds two APPEND-ONLY record kinds alongside laws: committed
predictions (hashed and persisted BEFORE the experiment runs) and their
verifications. The schema evolves additively — ``schema_version`` marks the
layout, and files written by earlier missions (laws only) load unchanged.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

from .prediction import (
    PredictionOutcome,
    PredictionRecord,
    VerificationRecord,
    commitment_hash,
    commitment_payload,
    verify_commitment,
)

SCHEMA_VERSION = 2


@dataclass
class LawRecord:
    """One established law in the knowledge base."""

    name: str
    formula: str
    value: float
    unit: str
    confidence: float
    r2: float
    experiments: list[str] = field(default_factory=list)
    derived_by: str = ""
    # Material laws carry named properties instead of a single value
    # (e.g. {"restitution": 0.72, "friction": 0.4}).
    properties: dict[str, float] = field(default_factory=dict)


class KnowledgeBase:
    """Persistent law store — one JSON file per universe."""

    def __init__(self, path: Path | str, universe: str):
        self.path = Path(path)
        self.universe = universe
        self.laws: dict[str, LawRecord] = {}
        self.predictions: dict[str, PredictionRecord] = {}
        self.verifications: dict[str, VerificationRecord] = {}
        self.load()

    @classmethod
    def load_or_create(cls, path: Path | str, universe: str) -> KnowledgeBase:
        """Open the store, creating an empty one if the file does not exist."""
        return cls(path, universe)

    # -- persistence ---------------------------------------------------------

    def load(self) -> None:
        if not self.path.exists():
            self.laws = {}
            return
        with self.path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        self.laws = {
            rec["name"]: LawRecord(
                name=rec["name"],
                formula=rec.get("formula", ""),
                value=float(rec.get("value", 0.0)),
                unit=rec.get("unit", ""),
                confidence=float(rec.get("confidence", 0.0)),
                r2=float(rec.get("r2", 0.0)),
                experiments=list(rec.get("experiments", [])),
                derived_by=rec.get("derived_by", ""),
                properties=dict(rec.get("properties", {})),
            )
            for rec in data.get("laws", [])
        }
        self.predictions = {
            rec["prediction_id"]: PredictionRecord(
                prediction_id=rec["prediction_id"],
                model_ref=rec.get("model_ref", ""),
                claim=rec.get("claim", ""),
                spec_ref=rec.get("spec_ref", ""),
                predicted=float(rec.get("predicted", 0.0)),
                tolerance=float(rec.get("tolerance", 0.0)),
                committed_hash=rec.get("committed_hash", ""),
                seq=int(rec.get("seq", 0)),
                created_at=rec.get("created_at", ""),
                status=rec.get("status", "open"),
            )
            for rec in data.get("predictions", [])
        }
        self.verifications = {
            rec["verification_id"]: VerificationRecord(
                verification_id=rec["verification_id"],
                prediction_id=rec.get("prediction_id", ""),
                experiment_id=rec.get("experiment_id", ""),
                observed=float(rec.get("observed", 0.0)),
                residual=float(rec.get("residual", 0.0)),
                status=rec.get("status", ""),
                evidence=rec.get("evidence", ""),
            )
            for rec in data.get("verifications", [])
        }

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "universe": self.universe,
            "schema_version": SCHEMA_VERSION,
            "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "laws": [asdict(rec) for rec in self.laws.values()],
            "predictions": [asdict(rec) for rec in self.predictions.values()],
            "verifications": [asdict(rec)
                              for rec in self.verifications.values()],
        }
        with self.path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
            fh.write("\n")

    # -- queries -------------------------------------------------------------

    def knows(self, claim: str, min_confidence: float = 0.999) -> bool:
        """True when the claim is already established well enough that a new
        mission should not re-run experiments for it."""
        law = self.laws.get(claim)
        return law is not None and law.confidence >= min_confidence

    def get(self, claim: str) -> LawRecord | None:
        return self.laws.get(claim)

    # -- mutation ------------------------------------------------------------

    def record_law(self, name: str, formula: str, value: float, unit: str,
                   confidence: float, r2: float, experiments: list[str],
                   derived_by: str) -> LawRecord:
        rec = LawRecord(
            name=name, formula=formula, value=value, unit=unit,
            confidence=confidence, r2=r2,
            experiments=list(experiments), derived_by=derived_by,
        )
        self.laws[name] = rec
        return rec

    def record_material(self, material: str, properties: dict[str, float],
                        confidence: float, r2: float, experiments: list[str],
                        derived_by: str) -> LawRecord:
        """Publish a material's measured property set as one knowledge entry."""
        rec = LawRecord(
            name=material, formula="material properties", value=0.0, unit="",
            confidence=confidence, r2=r2,
            experiments=list(experiments), derived_by=derived_by,
            properties=dict(properties),
        )
        self.laws[material] = rec
        return rec

    def material(self, name: str) -> dict[str, float] | None:
        """A material's measured properties, or None if unknown."""
        rec = self.laws.get(name)
        if rec is None or not rec.properties:
            return None
        return dict(rec.properties)

    # -- Genesis Phase 1: prediction commitments (append-only kinds) ---------

    def commit_prediction(self, model_ref: str, claim: str, spec_ref: str,
                          value: float, tolerance: float) -> PredictionRecord:
        """Hash and persist a commitment BEFORE the experiment runs.

        The record is append-only scientific history: the committed content
        (model_ref, claim, spec_ref, predicted, tolerance) is covered by a
        sha256 hash, nothing may edit it afterwards (check with
        :func:`verify_commitment`), and a changed mind commits a NEW
        prediction instead.
        """
        seq = len(self.predictions) + 1
        prediction_id = f"pred-{seq:04d}"
        record = PredictionRecord(
            prediction_id=prediction_id,
            model_ref=model_ref,
            claim=claim,
            spec_ref=spec_ref,
            predicted=float(value),
            tolerance=float(tolerance),
            committed_hash="",
            seq=seq,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        record = replace(record,
                         committed_hash=commitment_hash(commitment_payload(record)))
        self.predictions[prediction_id] = record
        self.save()
        return record

    def record_verification(self, prediction_id: str, experiment_id: str,
                            outcome: PredictionOutcome) -> VerificationRecord:
        """Append a VerificationRecord and resolve the prediction's status.

        This is the ONLY sanctioned status transition (open -> confirmed |
        refuted; a refutation is final for the phase). The committed content
        stays immutable and hash-checkable; a tampered record is refused.
        """
        prediction = self.predictions.get(prediction_id)
        if prediction is None:
            raise ValueError(f"unknown prediction {prediction_id!r}")
        if not verify_commitment(prediction):
            raise ValueError(
                f"commitment hash mismatch for {prediction_id}: refusing to "
                "verify a tampered prediction")
        verification_id = f"verif-{len(self.verifications) + 1:04d}"
        record = VerificationRecord(
            verification_id=verification_id,
            prediction_id=prediction_id,
            experiment_id=experiment_id,
            observed=outcome.observed,
            residual=outcome.residual,
            status=outcome.status,
            evidence=(f"|observed - predicted| = {abs(outcome.residual):.6g} "
                      f"vs tolerance {prediction.tolerance:g}"),
        )
        self.verifications[verification_id] = record
        if prediction.status != "refuted":
            prediction = replace(prediction, status=outcome.status)
            self.predictions[prediction_id] = prediction
        self.save()
        return record

    def prediction(self, prediction_id: str) -> PredictionRecord | None:
        return self.predictions.get(prediction_id)

    def verifications_for(self, prediction_id: str) -> list[VerificationRecord]:
        return [v for v in self.verifications.values()
                if v.prediction_id == prediction_id]

    def summary(self) -> str:
        if not self.laws:
            return f"knowledge: empty ({self.universe})"
        lines = [f"knowledge of '{self.universe}': {len(self.laws)} law(s)"]
        for rec in self.laws.values():
            unit = f" {rec.unit}" if rec.unit else ""
            lines.append(f"  {rec.name} = {rec.value:.6f}{unit}"
                         f"  confidence {rec.confidence * 100:.2f}%"
                         f"  [{rec.derived_by}]")
        return "\n".join(lines)
