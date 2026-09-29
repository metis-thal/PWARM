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
import re
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

from .knowledge_records import ModelRecord, model_content_hash
from .prediction import (
    CompetitionState,
    EvidenceSummary,
    PredictionOutcome,
    PredictionRecord,
    VerificationRecord,
    commitment_hash,
    commitment_payload,
    parse_spec_refs,
    verify_commitment,
)
from .prediction import (
    competition_state as build_competition_state,
)

# Schema 4: the Knowledge Layer's first entity — ModelRecord, an addressable
# model identity whose declared content is frozen and hash-anchored. v3-era
# files load unchanged (empty registry); predictions are untouched and their
# stored model_ref strings are NEVER migrated or rewritten — derivations
# keep matching them exactly as stored, so a registered model_id aggregates
# precisely the predictions that reference it.
SCHEMA_VERSION = 4


def _next_ordinal(ids, prefix: str) -> int:
    """One past the highest numeric suffix among ``prefixNNNN`` ids.

    Monotonic even if records were ever removed or merged from an older
    store — unlike a ``len()``-based counter, which would reuse ids and
    silently collide two distinct ledger entries.
    """
    pattern = re.compile(rf"^{prefix}(\d+)$")
    highest = max((int(m.group(1)) for i in ids
                   if (m := pattern.match(i))), default=0)
    return highest + 1



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
        self.model_records: dict[str, ModelRecord] = {}
        self.load()

    @classmethod
    def load_or_create(cls, path: Path | str, universe: str) -> KnowledgeBase:
        """Open the store, creating an empty one if the file does not exist."""
        return cls(path, universe)

    # -- persistence ---------------------------------------------------------

    def load(self) -> None:
        if not self.path.exists():
            self.laws = {}
            self.model_records = {}
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
                condition_binding=dict(rec.get("condition_binding", {})),
                output_binding=dict(rec.get("output_binding", {})),
                output=rec.get("output", ""),
                reduction_channel=rec.get("reduction_channel", ""),
                reduction_rule=rec.get("reduction_rule", ""),
                committed_hash=rec.get("committed_hash", ""),
                seq=int(rec.get("seq", 0)),
                created_at=rec.get("created_at", ""),
                status=rec.get("status", "open"),
                supersedes=rec.get("supersedes", ""),
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
                created_at=rec.get("created_at", ""),
            )
            for rec in data.get("verifications", [])
        }
        self.model_records = {
            rec["model_id"]: ModelRecord(
                model_id=rec["model_id"],
                formula=rec.get("formula", ""),
                params=dict(rec.get("params", {})),
                derived_from=rec.get("derived_from", ""),
                created_at=rec.get("created_at", ""),
                status=rec.get("status", "registered"),
                supersedes=rec.get("supersedes", ""),
                content_hash=rec.get("content_hash", ""),
            )
            for rec in data.get("model_records", [])
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
            "model_records": [asdict(rec)
                              for rec in self.model_records.values()],
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
                          value: float, tolerance: float,
                          condition_binding: dict[str, str] | None = None,
                          output_binding: dict[str, str] | None = None,
                          output: str = "",
                          reduction_channel: str = "",
                          reduction_rule: str = "",
                          supersedes: str = "") -> PredictionRecord:
        """Hash and persist a commitment BEFORE the experiment runs.

        The record is append-only scientific history: the committed content
        (model_ref, claim, spec_ref, predicted, tolerance, plus the declared
        verification contract — condition/output binding and observation
        reduction) is covered by a sha256 hash, nothing may edit it
        afterwards (check with :func:`verify_commitment`), and a changed
        mind commits a NEW prediction instead.

        The verification contract kwargs are store-level plumbing: the
        competition path fills them (its commitments are contract-complete);
        the legacy belief path commits without one and adjudicates through
        the free-fall fit.
        """
        seq = _next_ordinal(self.predictions, "pred-")
        prediction_id = f"pred-{seq:04d}"
        record = PredictionRecord(
            prediction_id=prediction_id,
            model_ref=model_ref,
            claim=claim,
            spec_ref=spec_ref,
            predicted=float(value),
            tolerance=float(tolerance),
            condition_binding=dict(condition_binding or {}),
            output_binding=dict(output_binding or {}),
            output=output,
            reduction_channel=reduction_channel,
            reduction_rule=reduction_rule,
            supersedes=supersedes,
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

        This is the ONLY sanctioned status transition. The prediction's
        status machine (explicit, tested):

            open ----confirmed verdict----> confirmed
            open ----refuted verdict-----> refuted
            confirmed --refuted verdict--> refuted     (a later verification
                                                        may overturn a
                                                        confirmation)
            refuted ----any verdict------> refuted     (ABSORBING: later
                                                        verifications still
                                                        append records, but
                                                        the status never
                                                        leaves refuted)
            open --supersede()---------> superseded    (withdrawn: this gate
                                                        refuses to verify it,
                                                        ever)

        No other statuses or transitions exist at this layer. The committed
        content stays immutable and hash-checkable; a tampered record is
        refused.

        Identity integrity: ``experiment_id`` must be covered by the
        commitment's ``spec_ref`` (the declared experiment battery, parsed
        by :func:`parse_spec_refs`). A verification against any other
        experiment is refused loudly — nothing is appended, no status flips,
        nothing is saved.
        """
        prediction = self.predictions.get(prediction_id)
        if prediction is None:
            raise ValueError(f"unknown prediction {prediction_id!r}")
        if not verify_commitment(prediction):
            raise ValueError(
                f"commitment hash mismatch for {prediction_id}: refusing to "
                "verify a tampered prediction")
        if prediction.status == "superseded":
            raise ValueError(
                f"prediction {prediction_id} was superseded — a withdrawn "
                "commitment can never be verified")
        if experiment_id not in parse_spec_refs(prediction.spec_ref):
            raise ValueError(
                f"experiment {experiment_id!r} is not covered by prediction "
                f"{prediction_id}'s committed spec_ref "
                f"{prediction.spec_ref!r} — a prediction may only be "
                "verified against experiments it was committed for")
        verification_id = f"verif-{_next_ordinal(self.verifications, 'verif-'):04d}"
        record = VerificationRecord(
            verification_id=verification_id,
            prediction_id=prediction_id,
            experiment_id=experiment_id,
            observed=outcome.observed,
            residual=outcome.residual,
            status=outcome.status,
            evidence=(f"|observed - predicted| = {abs(outcome.residual):.6g} "
                      f"vs tolerance {prediction.tolerance:g}"),
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        self.verifications[verification_id] = record
        if prediction.status != "refuted":
            prediction = replace(prediction, status=outcome.status)
            self.predictions[prediction_id] = prediction
        self.save()
        return record

    def supersede(self, old_id: str, new_id: str) -> tuple[PredictionRecord,
                                                           PredictionRecord]:
        """Withdraw an OPEN commitment in favor of a NEW prediction.

        The changed-mind linkage, explicit: the old commitment's status
        becomes ``superseded`` and the new record's ``supersedes`` points
        back at it — provenance without editing either commitment's
        hashed content (both records are replaced, never mutated in
        place, and both still hash-verify).

        Refused loudly when: either id is unknown, old and new are the
        same prediction, the old commitment already carries a verdict
        (confirmed/refuted are immutable history — a changed mind simply
        commits a new prediction alongside), or the new prediction
        already supersedes something else.
        """
        old = self.predictions.get(old_id)
        if old is None:
            raise ValueError(f"unknown prediction {old_id!r}")
        new = self.predictions.get(new_id)
        if new is None:
            raise ValueError(f"unknown prediction {new_id!r}")
        if old_id == new_id:
            raise ValueError(f"prediction {old_id!r} cannot supersede itself")
        if old.status != "open":
            raise ValueError(
                f"only an open commitment can be superseded; {old_id} is "
                f"{old.status!r} — verdicts are immutable history")
        if new.supersedes:
            raise ValueError(
                f"{new_id} already supersedes {new.supersedes!r} — commit "
                "a fresh prediction instead")
        old = replace(old, status="superseded")
        new = replace(new, supersedes=old_id)
        self.predictions[old_id] = old
        self.predictions[new_id] = new
        self.save()
        return old, new

    # -- Knowledge Layer KL-1: addressable model identities ------------------
    #
    # ModelRecords are DECLARED FACTS, not evaluations: the declared content
    # (model_id, formula, params, derived_from) is frozen at registration and
    # hash-anchored; a changed model is a NEW record plus a supersedes link,
    # never an edit. Status is lifecycle-only (registered | superseded) and
    # can never express quality — no score, confidence, accuracy or winner
    # exists in this layer. Model standing is derived from the ledger
    # (competition_state), never stored. Standing matches the STORED
    # model_ref strings exactly: old-format predictions are never migrated,
    # rewritten or merged into a registered id's view.

    def register_model(self, model_id: str, formula: str,
                       params: dict[str, float] | None = None,
                       derived_from: str = "") -> ModelRecord:
        """Register a model identity — a DECLARED FACT, frozen at creation.

        The identity is unique: a duplicate model_id is refused (parameter
        changes are a NEW ModelRecord plus a supersede link, never an
        edit). The declared content is hash-anchored; status is
        lifecycle-only (registered | superseded).
        """
        if not model_id:
            raise ValueError("model_id must be a non-empty identifier")
        if model_id in self.model_records:
            raise ValueError(
                f"model {model_id!r} is already registered — parameter "
                "changes are a NEW ModelRecord plus a supersede link, "
                "never an edit")
        record = ModelRecord(
            model_id=model_id,
            formula=formula,
            params=dict(params or {}),
            derived_from=derived_from,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        record = replace(record, content_hash=model_content_hash(record))
        self.model_records[model_id] = record
        self.save()
        return record

    def supersede_model(self, old_id: str,
                        new_id: str) -> tuple[ModelRecord, ModelRecord]:
        """Replace one registered model identity with another.

        The changed-model linkage, explicit: the old identity's status
        becomes ``superseded`` and the new record's ``supersedes`` points
        back — provenance without editing either record's declared
        content (both are replaced, never mutated, both still
        hash-verify). Refused loudly when: either id is unknown, old and
        new are the same identity, the old identity is already
        superseded, or the new record already supersedes something else.
        """
        old = self.model_records.get(old_id)
        if old is None:
            raise ValueError(f"unknown model {old_id!r}")
        new = self.model_records.get(new_id)
        if new is None:
            raise ValueError(f"unknown model {new_id!r}")
        if old_id == new_id:
            raise ValueError(f"model {old_id!r} cannot supersede itself")
        if old.status != "registered":
            raise ValueError(
                f"only a registered model identity can be superseded; "
                f"{old_id} is {old.status!r}")
        if new.supersedes:
            raise ValueError(
                f"{new_id} already supersedes {new.supersedes!r} — "
                "register a fresh identity instead")
        old = replace(old, status="superseded")
        new = replace(new, supersedes=old_id)
        self.model_records[old_id] = old
        self.model_records[new_id] = new
        self.save()
        return old, new

    def model_record(self, model_id: str) -> ModelRecord | None:
        """The registered identity for model_id, or None when unknown."""
        return self.model_records.get(model_id)

    def prediction(self, prediction_id: str) -> PredictionRecord | None:
        return self.predictions.get(prediction_id)

    def verifications_for(self, prediction_id: str) -> list[VerificationRecord]:
        return [v for v in self.verifications.values()
                if v.prediction_id == prediction_id]

    # -- Genesis Step 8: per-model evidence summary (pure read) ---------------

    def evidence_for_model(self, model_ref: str) -> EvidenceSummary:
        """Aggregate a model's experimental evidence from persisted records.

        Each persisted :class:`VerificationRecord` is traced back through its
        ``prediction_id`` to the owning :class:`PredictionRecord`'s
        ``model_ref`` — the model is NEVER guessed. A verification whose
        prediction_id has no matching prediction (an orphaned record) has no
        knowable model and is skipped explicitly; it never lands in any
        model's summary.

        The summary separates the RAW verification count from the three
        INDEPENDENT counts. Independent evidence is defined as a distinct
        ``(prediction_id, experiment_id)`` pair: re-verifying the same
        prediction against the same experiment adds no independent evidence,
        while a distinct prediction OR a distinct experiment does.
        ``independent_evidence`` holds one representative record per distinct
        pair (the first, in verification-id order).

        Pure read: no record is created, edited or deleted, the knowledge
        base is not saved, and the Laboratory is never consulted. The result
        is fully deterministic — repeated queries return identical summaries.
        """
        model_of = {pid: record.model_ref
                    for pid, record in self.predictions.items()}
        matched: list[VerificationRecord] = []
        for verification_id in sorted(self.verifications):  # deterministic order
            verification = self.verifications[verification_id]
            owner = model_of.get(verification.prediction_id)
            if owner is None:
                continue                       # orphaned: skip, never guess
            if owner == model_ref:
                matched.append(verification)
        # Independent evidence: distinct (prediction, experiment) pairs. The
        # first occurrence (lowest verification id) of each pair is kept.
        seen_pairs: set[tuple[str, str]] = set()
        independent: list[VerificationRecord] = []
        for verification in matched:
            pair = (verification.prediction_id, verification.experiment_id)
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                independent.append(verification)
        statuses = [v.status for v in matched]
        return EvidenceSummary(
            model_ref=model_ref,
            prediction_ids=tuple(sorted({v.prediction_id for v in matched})),
            experiment_ids=tuple(sorted({v.experiment_id for v in matched})),
            independent_prediction_count=len({v.prediction_id for v in matched}),
            independent_experiment_count=len({v.experiment_id for v in matched}),
            independent_evidence_count=len(independent),
            verification_count=len(matched),
            confirmed_count=statuses.count("confirmed"),
            refuted_count=statuses.count("refuted"),
            residuals=tuple(v.residual for v in matched),
            statuses=tuple(statuses),
            evidence=tuple(matched),
            independent_evidence=tuple(independent),
        )

    def competition_state(self, model_ref: str) -> CompetitionState:
        """One competition model's standing, from persisted records only.

        Pure read over :meth:`evidence_for_model`: the statistical unit is
        (model_ref, experiment_id), disagreements between the model's own
        predictions about the same experiment are reported as conflicts —
        never silently resolved. Facts, not scores.
        """
        return build_competition_state(model_ref,
                                       self.evidence_for_model(model_ref))

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
