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
* :class:`QuestionRecord` (AS-1) — a DECLARED open research question,
  traceable to the facts that triggered it (source_fact_type +
  source_ids). Questions are TO-INVESTIGATE markers, not evaluations:
  no priority, importance, urgency or confidence exists; the research
  agenda's order comes from a declared deterministic precedence, never
  from a computed score.

No score, confidence, accuracy, winner or truth value exists in this
layer; status vocabularies are lifecycle-only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .records import (
    CompetitionState,
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
# -- AS-1: declared research questions -----------------------------------------
#
# A QuestionRecord is a TO-INVESTIGATE marker, not an evaluation: it records
# that the AI has committed to investigating a fact-shaped gap, with full
# traceability to the triggering facts (source_fact_type + source_ids — id
# references, never ledger copies). The question kinds are a CLOSED
# vocabulary (the fact-driven scanners of AS-2); free-form questions are
# refused by design. There is no priority, importance, urgency or
# confidence on a question — the research agenda's order comes from a
# declared deterministic precedence, never from a computed score.

QUESTION_KINDS = (
    "untested_generality",     # a relation has training facts but no held-out ones
    "anomaly",                 # a model carries refuted/conflicted verdicts
    "undefined_concept",       # a claim appears in predictions without a definition
    "unverified_identity",     # a registered model has zero verifications
)


@dataclass(frozen=True)
class QuestionRecord:
    """A declared open research question — to-investigate, not evaluation.

    ``content_hash`` anchors the DECLARED fields (kind, source_fact_type,
    source_ids, question, declared_by); question_id / created_at / status
    / supersedes are bookkeeping and stay outside the hash.

    ``source_ids`` are id REFERENCES into the store (relation ids, model
    ids, concept ids — per ``source_fact_type``), never copies of ledger
    content. Status is lifecycle-only (open | withdrawn): withdrawing
    says "not now", never "answered" — whether a question's underlying
    fact still exists is re-derived from the store on every scan.
    """

    question_id: str = ""             # "ques-NNNN" ordinal (bookkeeping)
    kind: str = ""                    # declared: one of QUESTION_KINDS
    source_fact_type: str = ""        # declared: which fact derivation triggered it
    source_ids: tuple[str, ...] = field(default_factory=tuple)
    # declared: id references to the triggering facts
    question: str = ""                # declared: the research question, stated
    declared_by: str = ""             # declared provenance
    created_at: str = ""              # ISO timestamp (informational)
    status: str = "open"              # open | withdrawn
    supersedes: str = ""              # bookkeeping lineage slot (consistent form)
    content_hash: str = ""


def question_payload(record: QuestionRecord) -> dict:
    """The tamper-evident content of a question declaration: the declared
    fields only (bookkeeping stays outside the hash)."""
    return {
        "kind": record.kind,
        "source_fact_type": record.source_fact_type,
        "source_ids": list(record.source_ids),
        "question": record.question,
        "declared_by": record.declared_by,
    }


def question_content_hash(record: QuestionRecord) -> str:
    """sha256 over the declared fields (deterministic serialization)."""
    return commitment_hash(question_payload(record))


def verify_question_record(record: QuestionRecord) -> bool:
    """Recompute the content hash — False means the declared content was
    edited after declaration (tampering or corruption)."""
    return record.content_hash == question_content_hash(record)
# -- AS-2: fact-driven research scanners and the declared precedence -----------
#
# The research agenda is FACT-DRIVEN: four deterministic scanners translate
# existing fact gaps into question candidates — they invent nothing, they
# read no outcomes beyond the FAILURE facts (anomaly), and they never rank
# models or questions by quality. The agenda's order is a DECLARED
# deterministic precedence (anomaly first — the loop must face refuting
# evidence before it may look for confirmations), never a computed score.

@dataclass(frozen=True)
class QuestionCandidate:
    """A fact-gap translated into a declarable question — the exact
    arguments of ``KnowledgeBase.declare_question``, produced by a
    scanner. Pure data: no score, no priority, no urgency."""

    kind: str
    source_fact_type: str
    source_ids: tuple[str, ...]
    question: str


# The declared deterministic precedence of the research agenda (anomaly
# first, by design: refuting facts outrank confirmation-seeking). This is
# a DECLARED constant, auditable and fixed — never a computed ranking.
RESEARCH_PRECEDENCE = (
    "anomaly",
    "untested_generality",
    "undefined_concept",
    "unverified_identity",
)


def order_research_candidates(
        candidates: tuple[QuestionCandidate, ...]) -> tuple[QuestionCandidate, ...]:
    """Order candidates by the declared precedence, then by source ids.

    Pure and deterministic: the same candidates always order identically,
    regardless of discovery order or any verdict content.
    """
    rank = {kind: index for index, kind in enumerate(RESEARCH_PRECEDENCE)}
    return tuple(sorted(
        candidates,
        key=lambda candidate: (rank.get(candidate.kind, len(rank)),
                               candidate.source_ids,
                               candidate.question)))


def scan_untested_generality(
        relation_evidences: tuple[tuple[RelationRecord, RelationEvidence], ...],
        ) -> tuple[QuestionCandidate, ...]:
    """Relations with training facts but NO held-out facts yet.

    A relation that never left its training scope is an untested
    generalization — the gap declares itself. A relation with no training
    pairs at all is not scanned (there is nothing yet to generalize).
    """
    candidates: list[QuestionCandidate] = []
    for relation, evidence in relation_evidences:
        if evidence.training_pairs and not evidence.heldout_pairs:
            candidates.append(QuestionCandidate(
                kind="untested_generality",
                source_fact_type="relation_evidence",
                source_ids=(relation.relation_id,),
                question=(f"does relation {relation.relation_id} "
                          f"({relation.formula}) generalize beyond its "
                          "declared training scope?"),
            ))
    return tuple(candidates)


def scan_anomaly(
        model_standings: tuple[tuple[ModelRecord, CompetitionState], ...],
        ) -> tuple[QuestionCandidate, ...]:
    """Models carrying REFUTED or CONFLICTED verdicts — the adversarial
    scanner: failures demand investigation before confirmation-seeking."""
    candidates: list[QuestionCandidate] = []
    for model, state in model_standings:
        if state.refuted_count > 0 or state.conflicts:
            candidates.append(QuestionCandidate(
                kind="anomaly",
                source_fact_type="competition_state",
                source_ids=(model.model_id,),
                question=(f"why does model {model.model_id} carry refuted "
                          "or conflicted verdicts?"),
            ))
    return tuple(candidates)


def scan_undefined_concept(
        claim_status: tuple[tuple[str, bool], ...],
        ) -> tuple[QuestionCandidate, ...]:
    """Claims that appear in predictions but have no active operational
    definition — vocabulary gaps declared by the AI's own usage.

    ``claim_status`` carries (claim, has_active_definition) facts.
    """
    candidates: list[QuestionCandidate] = []
    for claim, defined in claim_status:
        if not defined:
            candidates.append(QuestionCandidate(
                kind="undefined_concept",
                source_fact_type="predictions_for_concept",
                source_ids=(claim,),
                question=(f"what does the claim {claim!r} mean "
                          "operationally?"),
            ))
    return tuple(candidates)


def scan_unverified_identity(
        models_with_verifications: tuple[tuple[ModelRecord, int], ...],
        ) -> tuple[QuestionCandidate, ...]:
    """Registered models with ZERO verifications — an identity that has
    never faced an experiment. ``models_with_verifications`` carries
    (model, verification_count) facts."""
    candidates: list[QuestionCandidate] = []
    for model, verification_count in models_with_verifications:
        if verification_count == 0:
            candidates.append(QuestionCandidate(
                kind="unverified_identity",
                source_fact_type="model_lineage",
                source_ids=(model.model_id,),
                question=(f"does model {model.model_id} predict at all? "
                          "(no verification exists)"),
            ))
    return tuple(candidates)


