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

from .contracts import observation_fields, reduction_rules
from .knowledge_records import (
    QUESTION_KINDS,
    DefinitionRecord,
    ModelLineage,
    ModelRecord,
    QuestionRecord,
    RelationEvidence,
    RelationRecord,
    definition_content_hash,
    model_content_hash,
    question_content_hash,
    relation_content_hash,
    validate_concept_id,
)
from .knowledge_records import (
    relation_evidence as derive_relation_evidence,
)
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

# Schema 7: the Knowledge Layer's fourth entity — QuestionRecord, a declared
# open research question traceable to its triggering facts (AS fact-driven
# agenda). v6-era files load unchanged (empty question registry);
# predictions are untouched and their stored model_ref strings are NEVER
# migrated or rewritten — derivations keep matching them exactly as stored.
# Loading a file written by a NEWER schema is refused (an older program
# re-saving it would silently drop newer fields).
SCHEMA_VERSION = 7


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
        self.definitions: dict[str, DefinitionRecord] = {}
        self.relations: dict[str, RelationRecord] = {}
        self.questions: dict[str, QuestionRecord] = {}
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
            self.definitions = {}
            self.relations = {}
            self.questions = {}
            return
        with self.path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        stored_version = int(data.get("schema_version", 0))
        if stored_version > SCHEMA_VERSION:
            raise ValueError(
                f"knowledge file {self.path} was written by schema "
                f"v{stored_version}, newer than this program's "
                f"v{SCHEMA_VERSION} — loading and re-saving would silently "
                "drop newer fields; refusing")
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
        self.definitions = {
            rec["definition_id"]: DefinitionRecord(
                definition_id=rec["definition_id"],
                concept_id=rec.get("concept_id", ""),
                kind=rec.get("kind", "measurand"),
                unit=rec.get("unit", ""),
                channel=rec.get("channel", ""),
                reduction_rule=rec.get("reduction_rule", ""),
                description=rec.get("description", ""),
                created_at=rec.get("created_at", ""),
                status=rec.get("status", "active"),
                supersedes=rec.get("supersedes", ""),
                content_hash=rec.get("content_hash", ""),
            )
            for rec in data.get("definitions", [])
        }
        self.relations = {
            rec["relation_id"]: RelationRecord(
                relation_id=rec["relation_id"],
                subject=rec.get("subject", ""),
                formula=rec.get("formula", ""),
                parameters_ref=rec.get("parameters_ref", ""),
                scope=tuple(rec.get("scope", [])),
                declared_by=rec.get("declared_by", ""),
                created_at=rec.get("created_at", ""),
                status=rec.get("status", "candidate"),
                supersedes=rec.get("supersedes", ""),
                content_hash=rec.get("content_hash", ""),
            )
            for rec in data.get("relations", [])
        }
        self.questions = {
            rec["question_id"]: QuestionRecord(
                question_id=rec["question_id"],
                kind=rec.get("kind", ""),
                source_fact_type=rec.get("source_fact_type", ""),
                source_ids=tuple(rec.get("source_ids", [])),
                question=rec.get("question", ""),
                declared_by=rec.get("declared_by", ""),
                created_at=rec.get("created_at", ""),
                status=rec.get("status", "open"),
                supersedes=rec.get("supersedes", ""),
                content_hash=rec.get("content_hash", ""),
            )
            for rec in data.get("questions", [])
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
            "definitions": [asdict(rec)
                            for rec in self.definitions.values()],
            "relations": [asdict(rec) for rec in self.relations.values()],
            "questions": [asdict(rec) for rec in self.questions.values()],
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

    # -- Knowledge Layer KL-2: operational concept definitions ----------------
    #
    # DefinitionRecords are vocabulary, not truth: the declared procedure
    # (concept_id + channel + reduction_rule + unit) is frozen and
    # hash-anchored; a revised procedure is a NEW record plus an atomic
    # supersede (a concept has AT MOST ONE active definition — claim
    # resolution must never be ambiguous). Definitions are referenced for
    # LOOKUP only and can never become a source for VerificationRecords;
    # they carry no standing, no counts, no verdict-shaped data.

    def define_concept(self, concept_id: str, unit: str, channel: str,
                       reduction_rule: str, description: str = "",
                       kind: str = "measurand",
                       supersedes: str = "") -> DefinitionRecord:
        """Define a concept operationally, or revise it atomically.

        First registration: ``supersedes`` empty — the concept must not
        already have an active definition. Revision: ``supersedes`` names
        the current definition of the SAME concept — it is closed
        (status ``superseded``) and the new record becomes the single
        active definition in one step, so claim resolution is never
        ambiguous. Both the old and the new record still hash-verify.

        Validated loudly against the declared vocabularies: concept_id
        (lowercase snake_case), kind (v0: measurand), channel and
        reduction_rule (contracts.py's registries), and a non-empty unit.
        """
        validate_concept_id(concept_id)
        if kind != "measurand":
            raise ValueError(
                f"unsupported definition kind {kind!r}; v0 defines "
                "measurands only")
        if channel not in observation_fields():
            raise ValueError(
                f"channel {channel!r} is not an observation field; "
                f"valid: {sorted(observation_fields())}")
        if reduction_rule not in reduction_rules():
            raise ValueError(
                f"reduction rule {reduction_rule!r} is not declared; "
                f"valid: {list(reduction_rules())}")
        if not unit:
            raise ValueError(
                f"a measurand definition needs a unit (concept "
                f"{concept_id!r})")
        active = self.active_definition(concept_id)
        if supersedes:
            old = self.definitions.get(supersedes)
            if old is None:
                raise ValueError(f"unknown definition {supersedes!r}")
            if old.status != "active":
                raise ValueError(
                    f"only an active definition can be superseded; "
                    f"{supersedes} is {old.status!r}")
            if old.concept_id != concept_id:
                raise ValueError(
                    f"cannot supersede {supersedes} (concept "
                    f"{old.concept_id!r}) with a definition of "
                    f"{concept_id!r}")
        elif active is not None:
            raise ValueError(
                f"concept {concept_id!r} already has an active definition "
                f"({active.definition_id}) — revise it with "
                "supersedes=...")
        record = DefinitionRecord(
            definition_id=f"def-{_next_ordinal(self.definitions, 'def-'):04d}",
            concept_id=concept_id,
            kind=kind,
            unit=unit,
            channel=channel,
            reduction_rule=reduction_rule,
            description=description,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            supersedes=supersedes,
        )
        record = replace(record, content_hash=definition_content_hash(record))
        if supersedes:
            self.definitions[supersedes] = replace(
                self.definitions[supersedes], status="superseded")
        self.definitions[record.definition_id] = record
        self.save()
        return record

    def definition(self, definition_id: str) -> DefinitionRecord | None:
        """The definition record for definition_id, or None when unknown."""
        return self.definitions.get(definition_id)

    def active_definition(self, concept_id: str) -> DefinitionRecord | None:
        """The current operational definition of a concept, or None.

        At most one exists — the atomic revision invariant keeps claim
        resolution unambiguous.
        """
        for record in self.definitions.values():
            if record.concept_id == concept_id and record.status == "active":
                return record
        return None

    def definition_history(self, concept_id: str) -> tuple[DefinitionRecord, ...]:
        """The concept's definition chain, newest first.

        Definition ids are monotonic ordinals, so reverse id order IS the
        revision order; the active definition (if any) leads the chain.
        """
        return tuple(sorted(
            (record for record in self.definitions.values()
             if record.concept_id == concept_id),
            key=lambda record: record.definition_id, reverse=True))

    def definitions_for_procedure(self, channel: str,
                                  reduction_rule: str) -> tuple[DefinitionRecord, ...]:
        """Active definitions whose operational procedure is (channel, rule)
        — the contract-side resolution of a prediction's declared reduction.
        """
        return tuple(
            record for record in self.definitions.values()
            if record.status == "active"
            and record.channel == channel
            and record.reduction_rule == reduction_rule)
    # -- Knowledge Layer KL-4: lineage queries --------------------------------
    #
    # Pure reference resolution over the store — no copies, no cache, no
    # writes. Exact-match discipline (KL-1): a legacy model_ref string is
    # never merged into a registered id's lineage.

    def predictions_for_model(self, model_ref: str) -> tuple[PredictionRecord, ...]:
        """Predictions whose stored model_ref equals model_id, in commitment
        (seq) order — the identity side of the lineage chain."""
        return tuple(sorted(
            (record for record in self.predictions.values()
             if record.model_ref == model_ref),
            key=lambda record: record.seq))

    def predictions_for_concept(self, concept_id: str) -> tuple[PredictionRecord, ...]:
        """Predictions whose claim names the concept, in commitment (seq)
        order — the claim side of the lineage chain (a DefinitionRecord's
        concept_id resolves here)."""
        return tuple(sorted(
            (record for record in self.predictions.values()
             if record.claim == concept_id),
            key=lambda record: record.seq))

    def model_lineage(self, model_id: str) -> ModelLineage:
        """The structural chain of one model identity: its registered
        record (None when unregistered) plus the predictions referencing
        the id, each with its verifications grouped in ledger order.

        Pure derivation — nothing is written, nothing is cached; repeated
        calls on an unchanged store return equal values. Standing is not
        part of the lineage: derive it with ``competition_state``.
        """
        predictions = self.predictions_for_model(model_id)
        verifications: dict[str, tuple[VerificationRecord, ...]] = {
            record.prediction_id: () for record in predictions}
        for record in self.verifications.values():   # ledger insertion order
            if record.prediction_id in verifications:
                verifications[record.prediction_id] += (record,)
        return ModelLineage(
            model_record=self.model_records.get(model_id),
            predictions=predictions,
            verifications=verifications,
        )

    # -- Knowledge Layer KL-3: candidate relations ----------------------------
    #
    # RelationRecords are DECLARATIONS, not evaluations: subject, formula,
    # instantiating model and the training scope are frozen and hash-
    # anchored; the scope is the single source of the training / held-out
    # boundary. Rival candidates coexist by design (no unique-active
    # invariant); supersedes chains revisions of the SAME relation. The
    # relation's standing is DERIVED (relation_evidence) from the ledger —
    # counts and condition sets only, never a stored judgment, and there
    # is no "established" status.

    def declare_relation(self, subject: str, formula: str,
                         parameters_ref: str, scope: tuple[str, ...],
                         declared_by: str = "",
                         supersedes: str = "") -> RelationRecord:
        """Declare a candidate relation, or revise one atomically.

        First declaration: ``supersedes`` empty. Revision: ``supersedes``
        names a CANDIDATE relation of the SAME subject AND the SAME
        instantiating model (formula and scope may change) — the old
        declaration is closed (status ``superseded``) and the new one is
        created in a single step; both records still hash-verify.

        Validated loudly: subject is a well-formed concept id, formula is
        non-empty, the parameters_ref is a REGISTERED model identity, and
        the scope is a non-empty duplicate-free tuple of non-empty
        experiment ids.
        """
        validate_concept_id(subject)
        if not formula:
            raise ValueError("a relation declaration needs a formula")
        model = self.model_records.get(parameters_ref)
        if model is None:
            raise ValueError(
                f"parameters_ref {parameters_ref!r} is not a registered "
                "model identity — declare relations only against "
                "registered models")
        if model.status != "registered":
            raise ValueError(
                f"model {parameters_ref!r} is superseded — a withdrawn "
                "identity cannot ground new relations")
        if not scope:
            raise ValueError(
                "a relation declaration needs a non-empty training scope")
        scope = tuple(scope)
        if len(set(scope)) != len(scope):
            raise ValueError("scope entries must be unique (no duplicates)")
        if any(not entry for entry in scope):
            raise ValueError("scope entries must be non-empty experiment ids")
        old = self.relations.get(supersedes) if supersedes else None
        if supersedes:
            if old is None:
                raise ValueError(f"unknown relation {supersedes!r}")
            if old.status != "candidate":
                raise ValueError(
                    f"only a candidate relation can be superseded; "
                    f"{supersedes} is {old.status!r}")
            if old.subject != subject or old.parameters_ref != parameters_ref:
                raise ValueError(
                    f"cannot supersede {supersedes} (subject "
                    f"{old.subject!r}, model {old.parameters_ref!r}) with a "
                    f"relation of subject {subject!r}, model "
                    f"{parameters_ref!r} — declare a rival candidate instead")
        record = RelationRecord(
            relation_id=f"rel-{_next_ordinal(self.relations, 'rel-'):04d}",
            subject=subject,
            formula=formula,
            parameters_ref=parameters_ref,
            scope=scope,
            declared_by=declared_by,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            supersedes=supersedes,
        )
        record = replace(record, content_hash=relation_content_hash(record))
        if supersedes:
            self.relations[supersedes] = replace(
                self.relations[supersedes], status="superseded")
        self.relations[record.relation_id] = record
        self.save()
        return record

    def relation(self, relation_id: str) -> RelationRecord | None:
        """The relation record for relation_id, or None when unknown."""
        return self.relations.get(relation_id)

    def relation_evidence(self, relation: RelationRecord) -> RelationEvidence:
        """The DERIVED evidence view of a relation: local (in-scope) vs
        generalization (held-out) facts from the ledger.

        Thin convenience wrapper: the derivation itself lives in
        ``knowledge_records.relation_evidence`` (imported aliased as
        ``derive_relation_evidence``) — this method only feeds it the
        model's ledger summary. Nothing is written, nothing is persisted;
        repeated calls on an unchanged store return equal values.
        """
        return derive_relation_evidence(
            relation, self.evidence_for_model(relation.parameters_ref))


    # -- Autonomous Scientist AS-1: declared research questions ---------------
    #
    # QuestionRecords are TO-INVESTIGATE markers, not evaluations: no
    # priority, importance, urgency or confidence exists on a question —
    # the research agenda's order comes from a declared deterministic
    # precedence (AS-2), never from a computed score. Questions are
    # traceable to their triggering facts (source_fact_type + source_ids,
    # id references only) and their kind vocabulary is CLOSED (the
    # fact-driven scanners); free-form questions are refused by design.

    def declare_question(self, kind: str, source_fact_type: str,
                         source_ids: tuple[str, ...], question: str,
                         declared_by: str = "") -> QuestionRecord:
        """Declare an open research question.

        The kind must come from the closed scanner vocabulary
        (``QUESTION_KINDS``); the source reference must be non-empty;
        and an identical OPEN question (same kind and source ids) is
        refused — the fact-driven agenda never accumulates duplicate
        markers for one gap. Withdrawing a question releases its slot
        (the scanners may re-propose it while the fact exists).
        """
        if kind not in QUESTION_KINDS:
            raise ValueError(
                f"unknown question kind {kind!r}; valid: {list(QUESTION_KINDS)}")
        if not source_fact_type:
            raise ValueError("a question needs a source_fact_type (which "
                             "fact derivation triggered it)")
        source_ids = tuple(source_ids)
        if not source_ids or any(not sid for sid in source_ids):
            raise ValueError(
                "source_ids must be a non-empty tuple of non-empty id "
                "references")
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("source_ids must be unique")
        if not question:
            raise ValueError("a question needs a stated research question")
        if self.open_question(kind, source_ids) is not None:
            raise ValueError(
                f"an open {kind!r} question for {sorted(source_ids)} already "
                "exists — the agenda never duplicates a marker for one gap")
        record = QuestionRecord(
            question_id=f"ques-{_next_ordinal(self.questions, 'ques-'):04d}",
            kind=kind,
            source_fact_type=source_fact_type,
            source_ids=source_ids,
            question=question,
            declared_by=declared_by,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        record = replace(record, content_hash=question_content_hash(record))
        self.questions[record.question_id] = record
        self.save()
        return record

    def withdraw_question(self, question_id: str) -> QuestionRecord:
        """Withdraw an open question — "not now", never "answered".

        Whether the underlying fact still exists is re-derived from the
        store on every scan; withdrawal only removes the marker from the
        open agenda.
        """
        record = self.questions.get(question_id)
        if record is None:
            raise ValueError(f"unknown question {question_id!r}")
        if record.status != "open":
            raise ValueError(
                f"only an open question can be withdrawn; {question_id} is "
                f"{record.status!r}")
        record = replace(record, status="withdrawn")
        self.questions[question_id] = record
        self.save()
        return record

    def question(self, question_id: str) -> QuestionRecord | None:
        """The question record for question_id, or None when unknown."""
        return self.questions.get(question_id)

    def open_question(self, kind: str,
                      source_ids: tuple[str, ...]) -> QuestionRecord | None:
        """The open question for an exact (kind, source_ids) gap, or None."""
        ids = tuple(source_ids)
        for record in self.questions.values():
            if (record.status == "open" and record.kind == kind
                    and record.source_ids == ids):
                return record
        return None

    def open_questions(self) -> tuple[QuestionRecord, ...]:
        """All open questions, in declaration (ordinal) order — the raw
        agenda the declared precedence of AS-2 orders."""
        return tuple(sorted(
            (record for record in self.questions.values()
             if record.status == "open"),
            key=lambda record: record.question_id))

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
