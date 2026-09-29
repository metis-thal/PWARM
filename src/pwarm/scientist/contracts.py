"""Verification contracts — the declared vocabulary between model, spec
and observation.

Both vocabulary bridges are DECLARED data, never guessed: an unbound
variable or an out-of-vocabulary target is refused loudly.
:class:`ConditionBinding` renames model condition variables into
ExperimentSpec parameters; :class:`OutputBinding` maps model output
variables onto ObservationRecord measurement channels;
:func:`format_spec_refs`/:func:`parse_spec_refs` define the single
spec_ref identity format; and :class:`ObservationReduction` reduces a
record's channels to the scalar a verdict would compare, through a
registry of data-justified rules.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .hypothesis import free_fall_g_value
from .records import Prediction

if TYPE_CHECKING:  # pragma: no cover - typing only; attribute access only
    from .experiment import ObservationRecord


# -- Genesis Step 4.5: condition binding (model variable -> spec parameter) -

@dataclass(frozen=True)
class ConditionBinding:
    """Explicit AI-side contract between the two condition vocabularies:
    which ExperimentSpec parameter each model condition variable means.

    Binding is DECLARED data, never guessed — an unmapped model variable
    is refused loudly, and nothing here knows about hidden truths or
    experimental results: it only renames condition keys.
    """

    mapping: dict[str, str]     # model variable -> ExperimentSpec parameter

    def translate(self, conditions: dict[str, float]) -> dict[str, float]:
        """Rename model-vocabulary conditions into spec parameters.

        Refuses, loudly, any variable the binding does not cover —
        silence would be a guessed conversion.
        """
        unmapped = [key for key in conditions if key not in self.mapping]
        if unmapped:
            raise ValueError(
                f"condition variable(s) {sorted(unmapped)} are not bound "
                f"to experiment parameters; binding covers "
                f"{sorted(self.mapping)}")
        return {self.mapping[key]: value for key, value in conditions.items()}


# -- Genesis Step 5A: output binding (model output -> observation field) ----

# ObservationRecord's measurement channels (its ``field_names``): the
# identifier (experiment_id) and the bookkeeping counter (steps) are not
# observable channels a model output could map to.
_OBSERVATION_FIELDS = frozenset({"t", "z", "vx"})


@dataclass(frozen=True)
class OutputBinding:
    """Explicit AI-side contract between the model's output vocabulary and
    the observation channel: which ObservationRecord measurement field
    each model output variable means.

    Like ConditionBinding, this is DECLARED data, never guessed — an
    unbound output or a non-observation target is refused loudly.
    """

    mapping: dict[str, str]     # model output variable -> record field

    def field_for(self, output: str) -> str:
        """The observation field this model output variable maps to."""
        if output not in self.mapping:
            raise ValueError(
                f"model output {output!r} is not bound to an observation "
                f"field; binding covers {sorted(self.mapping)}")
        field = self.mapping[output]
        if field not in _OBSERVATION_FIELDS:
            raise ValueError(
                f"{field!r} is not an ObservationRecord measurement field; "
                f"valid: {sorted(_OBSERVATION_FIELDS)}")
        return field


@dataclass(frozen=True)
class ComparisonInput:
    """The aligned pair an adjudicator would consume, and nothing more:
    the model's claimed value plus the observed channel it maps to. No
    verdict exists at this step.

    ``channels`` carries every measurement channel of the originating
    record (all AI-side data), so multi-channel reductions (e.g. the
    free-fall fit over t and z) can be declared contract rules too.
    """

    prediction: Prediction
    field: str                    # ObservationRecord measurement channel
    observed: tuple[float, ...]   # the channel's measured samples
    channels: dict[str, tuple[float, ...]] = field(default_factory=dict)


def comparison_input(prediction: Prediction,
                     record: ObservationRecord,
                     binding: OutputBinding,
                     output: str) -> ComparisonInput:
    """Align a model prediction with its bound observation channel.

    Pure and read-only: extracts the bound field's measured samples from
    the record (plus every channel, for multi-channel reductions);
    computes no confirmed/refuted verdict and updates nothing —
    adjudication is a later step.
    """
    field = binding.field_for(output)
    channels: dict[str, tuple[float, ...]] = {
        "t": tuple(float(v) for v in record.t),
        "z": tuple(float(v) for v in record.z),
    }
    if record.vx is not None:
        channels["vx"] = tuple(float(v) for v in record.vx)
    series = channels.get(field)
    if series is None:
        raise ValueError(
            f"observation field {field!r} is absent in this record "
            f"(experiment {record.experiment_id!r})")
    return ComparisonInput(prediction=prediction, field=field,
                           observed=series, channels=channels)


# -- Genesis Step 5B-1: observation reduction (sample sequence -> scalar) ----

# Reduction rules over a record's measurement channels. Each rule receives
# the ComparisonInput (the bound channel's samples plus every channel of
# the originating record), so single-channel rules ("first") and
# multi-channel rules ("free_fall_g") share one registry. Every rule is
# DECLARED, data-justified contract — never implicit code:
#
#   "first"      — the channel's first recorded sample. Matches the drop
#                  experiment's recorded semantics: the record starts one
#                  integration step after release, so the first z sample is
#                  release_height - g*dt^2 (semi-implicit Euler).
#   "free_fall_g" — the gravity scalar implied by the free-fall fit of the
#                  t/z channels. The legacy belief path's declared
#                  reduction; it derives a gravity claim and refuses any
#                  other claim (declared coherence, like OutputBinding).
_REDUCTION_RULES = {
    "first": lambda comparison: comparison.observed[0],
    "free_fall_g": lambda comparison: _reduce_free_fall_g(comparison),
}


def _reduce_free_fall_g(comparison: ComparisonInput) -> float:
    """The free-fall reduction: refuse incoherent claims, then derive g."""
    if comparison.prediction.claim != "gravity":
        raise ValueError(
            f"the free_fall_g reduction derives a gravity scalar; "
            f"prediction claims {comparison.prediction.claim!r}")
    missing = [name for name in ("t", "z") if not comparison.channels.get(name)]
    if missing:
        raise ValueError(
            f"the free_fall_g reduction needs the record's t and z "
            f"channels; missing: {sorted(missing)}")
    if len(comparison.channels["t"]) < 5:
        raise ValueError(
            "record too short for the free_fall_g reduction "
            "(needs >= 5 samples)")
    return free_fall_g_value(comparison.channels["t"],
                             comparison.channels["z"])


@dataclass(frozen=True)
class ObservationReduction:
    """Explicit, reproducible contract for reducing a record's measurement
    channels to the scalar a verdict would compare.

    Two declared rules exist, both data-justified (see _REDUCTION_RULES):
    ``("z", "first")`` — the channel's first sample, the release-height
    reading; and ``("z", "free_fall_g")`` — the free-fall fit's gravity
    scalar, the legacy belief path's declared reduction. Declared, never
    guessed: an unsupported rule, an empty rule, a channel mismatch, or an
    empty sample sequence is refused loudly.
    """

    channel: str      # ObservationRecord measurement field (e.g. "z")
    rule: str         # reduction rule name; see _REDUCTION_RULES

    def reduce(self, comparison: ComparisonInput) -> float:
        """Reduce the comparison's observed samples to one scalar."""
        if comparison.field != self.channel:
            raise ValueError(
                f"reduction targets channel {self.channel!r} but the "
                f"comparison carries {comparison.field!r}")
        rule = _REDUCTION_RULES.get(self.rule)
        if rule is None:
            raise ValueError(
                f"unsupported reduction rule {self.rule!r}; "
                f"supported: {sorted(_REDUCTION_RULES)}")
        if not comparison.observed:
            raise ValueError(
                f"no samples to reduce on channel {self.channel!r}")
        return float(rule(comparison))


# -- Experiment identity: the spec_ref contract ------------------------------
#
# A committed prediction declares, in ``spec_ref``, WHICH experiments it may
# be verified against: the planned battery's experiment ids joined with
# ", ". A verification persisted into the ledger must reference an
# experiment from this declared set — a prediction committed for E1 is never
# verifiable against E2. The two helpers below are the single definition of
# that format; the commit side serializes through ``format_spec_refs`` and
# the persistence side parses through ``parse_spec_refs``, so the identity
# relationship has exactly one source.

def format_spec_refs(spec_ids: Sequence[str]) -> str:
    """Serialize the planned experiment battery into a PredictionRecord's
    ``spec_ref`` (", "-joined experiment ids)."""
    return ", ".join(spec_ids)


def parse_spec_refs(spec_ref: str) -> tuple[str, ...]:
    """The experiment ids a committed prediction's ``spec_ref`` covers."""
    return tuple(s.strip() for s in spec_ref.split(",") if s.strip())
