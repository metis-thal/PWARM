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

No score, confidence, accuracy, winner or truth value exists in this
layer; status vocabularies are lifecycle-only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .records import commitment_hash


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
