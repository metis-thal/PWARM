"""Knowledge records — persistent, addressable knowledge entities.

Two kinds of declared facts live here, both strictly NON-evaluative:

* :class:`ModelRecord` (KL-1) — an addressable model identity whose
  declared content is frozen and hash-anchored; standing is derived from
  the ledger, never stored.
* :class:`DefinitionRecord` (KL-2) — the operational definition of a
  concept (which channel, which reduction rule, which unit): vocabulary,
  not truth. It is referenced for LOOKUP only — it can never become a
  source for VerificationRecords (P1-4's frozen contracts remain the sole
  adjudication basis) and never carries verdict-shaped data.
* :class:`RelationRecord` (KL-3) — a DECLARED candidate generalization
  (subject concept, formula, instantiating model, training scope). Its
  scientific content is never stored: :func:`relation_evidence` derives
  it from the ledger, splitting verifications into local (in-scope) and
  generalization (held-out) facts. Counts and condition sets only —
  never a score, ranking, winner or establishment judgment.

No score, confidence, accuracy, winner or truth value exists in this
layer; status vocabularies are lifecycle-only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .records import (
    EvidenceSummary,
    PredictionRecord,
    VerificationRecord,
    commitment_hash,
)


@dataclass(frozen=True)
class ModelRecord:
    """A registered model identity — declared fact, never an evaluation.

    ``content_hash`` anchors the DECLARED fields (model_id, formula,
    params, derived_from); created_at / status / supersedes are
    bookkeeping and stay outside the hash. Parameters are immutable: a
    changed model is a NEW ModelRecord plus a supersedes link, never an
    edit. Status is lifecycle-only (registered | superseded) — it can
    never express quality.

    Standing (what the experiments say about this model) is DERIVED from
    the ledger via ``competition_state(model_id)`` — nothing is stored
    here.
    """

    model_id: str
    formula: str
    params: dict[str, float] = field(default_factory=dict)
    derived_from: str = ""            # declared provenance (experiment/hypothesis)
    created_at: str = ""              # ISO timestamp (informational)
    status: str = "registered"        # registered | superseded
    supersedes: str = ""              # the model_id this record replaces
    content_hash: str = ""


def model_payload(record: ModelRecord) -> dict:
    """The tamper-evident content of a model registration: the declared
    fields only (bookkeeping stays outside the hash)."""
    return {
        "model_id": record.model_id,
        "formula": record.formula,
        "params": dict(record.params),
        "derived_from": record.derived_from,
    }


def model_content_hash(record: ModelRecord) -> str:
    """sha256 over the declared fields (deterministic serialization)."""
    return commitment_hash(model_payload(record))


def verify_model_record(record: ModelRecord) -> bool:
    """Recompute the content hash — False means the declared content was
    edited after registration (tampering or corruption)."""
    return record.content_hash == model_content_hash(record)


# -- KL-2: operational concept definitions ------------------------------------
#
# Concept ids are the claim vocabulary: lowercase snake_case, so claim
# resolution can never degenerate back into free-form strings (KL-1's
# identity discipline, applied to concepts).

_CONCEPT_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


def validate_concept_id(concept_id: str) -> str:
    """Return the concept_id when it is a well-formed concept identity.

    Lowercase snake_case (``gravity``, ``air_density``); decorated names
    (``Gravity``, ``gravity force!!!``) are refused loudly.
    """
    if not _CONCEPT_ID_PATTERN.match(concept_id or ""):
        raise ValueError(
            f"concept_id {concept_id!r} is not a valid concept identity "
            "(lowercase snake_case, e.g. 'gravity' or 'air_density')")
    return concept_id


@dataclass(frozen=True)
class DefinitionRecord:
    """The operational definition of a concept — vocabulary, not truth.

    ``(channel, reduction_rule)`` IS the definition: which observation
    channel, reduced by which declared rule, yields the concept's scalar
    in ``unit``. Both names must exist in contracts.py's declared
    vocabularies at definition time; the definition is frozen and
    hash-anchored over its declared fields.

    ``concept_id`` is STABLE across revisions (predictions' claims keep
    resolving); what changes between revisions is the procedure, chained
    through ``supersedes``. A definition is never confirmed or refuted —
    it is USED and REVISED (status is lifecycle-only: active |
    superseded), and it can never become a source for VerificationRecords.
    """

    definition_id: str = ""           # "def-NNNN" ordinal (bookkeeping)
    concept_id: str = ""              # stable claim vocabulary
    kind: str = "measurand"           # v0 defines measurands only
    unit: str = ""                    # declared unit of the concept's scalar
    channel: str = ""                 # observation channel (declared vocabulary)
    reduction_rule: str = ""          # reduction rule (declared vocabulary)
    description: str = ""
    created_at: str = ""              # ISO timestamp (informational)
    status: str = "active"            # active | superseded
    supersedes: str = ""              # definition_id this revision replaces
    content_hash: str = ""


def definition_payload(record: DefinitionRecord) -> dict:
    """The tamper-evident content of a definition: the declared fields
    only (definition_id and other bookkeeping stay outside the hash)."""
    return {
        "concept_id": record.concept_id,
        "kind": record.kind,
        "unit": record.unit,
        "channel": record.channel,
        "reduction_rule": record.reduction_rule,
        "description": record.description,
    }


def definition_content_hash(record: DefinitionRecord) -> str:
    """sha256 over the declared fields (deterministic serialization)."""
    return commitment_hash(definition_payload(record))


def verify_definition_record(record: DefinitionRecord) -> bool:
    """Recompute the content hash — False means the declared content was
    edited after definition (tampering or corruption)."""
    return record.content_hash == definition_content_hash(record)



# -- KL-3: candidate relations and their derived evidence ----------------------
#
# A RelationRecord is a DECLARATION, not an evaluation: subject concept,
# candidate formula, instantiating model and the training scope that defines
# the held-out boundary. Everything science says about it is DERIVED from
# the ledger by relation_evidence — local (in-scope) vs generalization
# (held-out) facts, counts and condition sets only.

@dataclass(frozen=True)
class RelationRecord:
    """A declared candidate generalization — declaration, not evaluation.

    ``content_hash`` anchors the DECLARED fields (subject, formula,
    parameters_ref, scope, declared_by); relation_id / created_at /
    status / supersedes are bookkeeping and stay outside the hash. The
    scope is the single source of the training / held-out boundary: a
    verification whose experiment_id is in the scope is LOCAL evidence,
    one outside it is GENERALIZATION evidence.

    Status is lifecycle-only (candidate | superseded) — there is no
    "established": whether a relation generalizes is a fact question
    answered by :class:`RelationEvidence`, never a stored judgment.
    Rival candidates coexist by design; ``supersedes`` chains revisions
    of the SAME relation (same subject and parameters_ref).
    """

    relation_id: str = ""             # "rel-NNNN" ordinal (bookkeeping)
    subject: str = ""                 # declared: concept_id of the explained quantity
    formula: str = ""                 # declared: candidate generalization form
    parameters_ref: str = ""          # declared: ModelRecord.model_id instantiating it
    scope: tuple[str, ...] = field(default_factory=tuple)
    # declared: the training battery's experiment ids — the held-out boundary
    declared_by: str = ""             # declared provenance
    created_at: str = ""              # ISO timestamp (informational)
    status: str = "candidate"         # candidate | superseded
    supersedes: str = ""              # relation_id this declaration revises
    content_hash: str = ""


def relation_payload(record: RelationRecord) -> dict:
    """The tamper-evident content of a relation declaration: the declared
    fields only (bookkeeping stays outside the hash)."""
    return {
        "subject": record.subject,
        "formula": record.formula,
        "parameters_ref": record.parameters_ref,
        "scope": list(record.scope),
        "declared_by": record.declared_by,
    }


def relation_content_hash(record: RelationRecord) -> str:
    """sha256 over the declared fields (deterministic serialization)."""
    return commitment_hash(relation_payload(record))


def verify_relation_record(record: RelationRecord) -> bool:
    """Recompute the content hash — False means the declared content was
    edited after declaration (tampering or corruption)."""
    return record.content_hash == relation_content_hash(record)


@dataclass(frozen=True)
class RelationPair:
    """One (prediction, experiment) verification projected to ids plus the
    single fact used for counting: the status, taken from the referenced
    VerificationRecord — no residual or evidence text is copied."""

    prediction_id: str
    experiment_id: str
    verification_id: str
    status: str                       # confirmed | refuted (from the ledger)


@dataclass(frozen=True)
class RelationEvidence:
    """The DERIVED standing of a relation — pure projection of ledger
    facts, never persisted, never written into any schema.

    Every count is a fact (how many verifications say what); every
    condition set is a fact (which experiments were tried). There is no
    score, ranking, winner, confidence, establishment or quality here —
    whether the relation generalizes is read directly from the held-out
    facts by whoever consumes this view.
    """

    relation_id: str
    parameters_ref: str
    scope: tuple[str, ...]                        # the declared boundary, echoed
    training_pairs: tuple[RelationPair, ...]      # experiment_id ∈ scope
    heldout_pairs: tuple[RelationPair, ...]       # experiment_id ∉ scope
    training_confirmed_count: int
    training_refuted_count: int
    heldout_confirmed_count: int
    heldout_refuted_count: int
    distinct_training_conditions: tuple[str, ...]  # sorted experiment ids
    distinct_heldout_conditions: tuple[str, ...]   # sorted experiment ids


def relation_evidence(relation: RelationRecord,
                      evidence: EvidenceSummary) -> RelationEvidence:
    """Derive a relation's evidence view from its model's ledger summary.

    Pure function of (declaration, ledger-derived summary): each
    independent (prediction, experiment) pair — the representative
    verification, in verification-id order, exactly as
    ``EvidenceSummary.independent_evidence`` defines it — is classified
    by ``experiment_id ∈ relation.scope`` into LOCAL (training) or
    GENERALIZATION (held-out) evidence. Confirmed/refuted come only from
    the referenced VerificationRecords; pairs without a verification
    contribute nothing. No writes, no randomness: repeated calls on an
    unchanged store return equal values.
    """
    training: list[RelationPair] = []
    heldout: list[RelationPair] = []
    scope = tuple(relation.scope)
    for verification in evidence.independent_evidence:
        pair = RelationPair(
            prediction_id=verification.prediction_id,
            experiment_id=verification.experiment_id,
            verification_id=verification.verification_id,
            status=verification.status,
        )
        if verification.experiment_id in scope:
            training.append(pair)
        else:
            heldout.append(pair)
    return RelationEvidence(
        relation_id=relation.relation_id,
        parameters_ref=relation.parameters_ref,
        scope=scope,
        training_pairs=tuple(training),
        heldout_pairs=tuple(heldout),
        training_confirmed_count=_count_status(training, "confirmed"),
        training_refuted_count=_count_status(training, "refuted"),
        heldout_confirmed_count=_count_status(heldout, "confirmed"),
        heldout_refuted_count=_count_status(heldout, "refuted"),
        distinct_training_conditions=_distinct_conditions(training),
        distinct_heldout_conditions=_distinct_conditions(heldout),
    )


def _count_status(pairs: list[RelationPair], status: str) -> int:
    return sum(1 for pair in pairs if pair.status == status)


def _distinct_conditions(pairs: list[RelationPair]) -> tuple[str, ...]:
    return tuple(sorted({pair.experiment_id for pair in pairs}))
# -- KL-4: lineage views -------------------------------------------------------
#
# A ModelLineage is a pure reference-resolution over the store: the model's
# identity record (when registered) plus the predictions that reference the
# id, each with its verifications grouped in ledger order. Nothing is
# copied, cached or persisted; standing continues to come from
# competition_state — the lineage is the structural chain, the competition
# state is the verdict aggregate.

@dataclass(frozen=True)
class ModelLineage:
    """The structural chain of one model identity — pure reference resolution.

    ``model_record`` is None when the id was never registered (a legacy
    free-form string): the predictions are still listed under their stored
    key, exactly as stored — never migrated, never merged (KL-1 policy).
    Standing (what the experiments say) is NOT part of the lineage; derive
    it with ``competition_state(model_id)``.
    """

    model_record: ModelRecord | None   # reference (None = unregistered id)
    predictions: tuple[PredictionRecord, ...]   # commitment (seq) order
    verifications: dict[str, tuple[VerificationRecord, ...]]
    # keyed by prediction_id, verification-id order within each group


