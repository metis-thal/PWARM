"""Knowledge records — persistent, addressable knowledge entities.

A :class:`ModelRecord` is a DECLARED FACT, not an evaluation object: it
gives the AI a persistent, addressable model identity whose declared
content (identity, form, parameters, provenance) is frozen at
registration and anchored by a content hash. Model standing is NEVER
stored here — it is derived from the ledger
(``evidence_for_model`` / ``competition_state``). The status vocabulary
is lifecycle-only (registered | superseded) and can never express
quality: no score, no confidence, no accuracy, no winner exists in this
layer.
"""

from __future__ import annotations

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
