#!/usr/bin/env python3
"""Deterministic capture for the sorted crossing-boundary owner
release/realization unit
(``2026-08-03-sorted-crossing-boundary-owner-release-realization``).

Frozen unit::

    research/investigations/qwen3-vl-dense-enumeration/experiments/
    2026-08-03-sorted-crossing-boundary-owner-release-realization/unit.md

What this module is
-------------------
It executes -- unchanged -- the sealed CPU plan built by
``prepare_sorted_crossing_boundary_owner_release_realization.py`` and emits raw,
per-request observable score rows plus per-owner observable records for the two
frozen measurement ladders at ``P`` and ``P+E``:

1. **natural description release** -- the target's exact ``D_C`` path and the
   exact native next action, both teacher-forced from the same literal context,
   with the first observable description divergence, the target-minus-native
   margin at that divergence, the complete description sum/token mean, and
   whether deterministic argmax follows the target through every observable
   description token; and
2. **target-conditioned coordinate realization** -- the sealed target-local and
   same-category physical-owner candidate families scored after forcing
   ``D_C``, plus a coordinate-only greedy decode of exactly four coordinate
   tokens through ``<|box_end|>``, owner-matched with the predecessor's one-box
   category-local strict matcher at the frozen IoU threshold.

What this module is deliberately **not**
----------------------------------------
* It is not an analyzer.  Primary branch assignment, the interpretability gate
  and the two-thirds routing rule live here only as *pure, unit-tested helpers*
  for the later analysis pass; the capture path never writes a branch onto an
  owner record.  The single exception is the mandatory cached-versus-uncached
  smoke, where ``unit.md`` explicitly requires the compared primary branch to
  be preserved -- there the branch is computed on both sides of one comparison
  and reported only inside the parity receipt.
* It never calls ``model.generate()``, never samples, and implements none of
  the optional temperature-0.2/K=16 in-row diagnostic.  Only the primary
  deterministic pass exists here.
* It never retokenizes.  Every scored token is a literal token id taken from
  the sealed plan or the sealed census registries; decoded prose is refused
  outright (:data:`FORBIDDEN_TEXT_KEYS`).
* It executes only ``readout_tier == "primary"`` requests.  ``unit.md`` seals
  every primary branch *before* any secondary compatibility field is read, and
  branch sealing is not this pass; the planned secondary requests are counted
  and deferred in the receipt rather than silently executed out of order.

Reused machinery (never re-implemented here)
--------------------------------------------
``score_sorted_owner_accessibility_census_shard`` supplies the production HF
session seam (``build_hf_session_spec``/``open_hf_backend``/``HFCensusBackend``,
its deterministic ``FakeCensusBackend`` twin, the full-vocabulary fp32
``readout``, and the atomic staging-directory publish), which in turn wraps
``score_sorted_owner_basin_landscape``'s ``prefill_context``/``HFCacheBackend``/
``BranchCursor`` explicit-Qwen-mrope machinery.  Owner matching of a freshly
decoded greedy box reuses
``build_sorted_all_person_greedy_boundary_census.independent_owner_assignment``
at ``build_sorted_owner_accessibility_census_plan.IOU_THRESHOLD``.

Outputs (one explicit shard directory, published atomically)::

    crossing-boundary-scores.jsonl        raw per-request observable rows
    crossing-boundary-owner-records.jsonl per-owner observable records
    crossing-boundary-parity.json         smoke matrix + cache/batch parity
    crossing-boundary-receipt.json        identities, digests, counters

A shard stopped by the quarantine rule publishes ``crossing-boundary-
quarantine.json`` alone and no primary evidence.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, NoReturn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research import (  # noqa: E402
    build_sorted_all_person_greedy_boundary_census as greedy_census,
)
from scripts.research import build_sorted_owner_accessibility_census_plan as planner  # noqa: E402
from scripts.research import (  # noqa: E402
    prepare_sorted_crossing_boundary_owner_release_realization as plan_builder,
)

# ---------------------------------------------------------------------------
# 0. Frozen schema / unit / cohort constants
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "sorted_crossing_boundary_owner_release_scores.v1"
RECEIPT_SCHEMA_VERSION = "sorted_crossing_boundary_owner_release_scores_receipt.v1"
OWNER_RECORD_SCHEMA_VERSION = "sorted_crossing_boundary_owner_release_owner_record.v1"
PARITY_SCHEMA_VERSION = "sorted_crossing_boundary_owner_release_parity.v1"
QUARANTINE_SCHEMA_VERSION = "sorted_crossing_boundary_owner_release_quarantine.v1"
ADMISSION_SCHEMA_VERSION = "sorted_crossing_boundary_owner_release_admission.v1"

UNIT_ID = "2026-08-03-sorted-crossing-boundary-owner-release-realization"

SCORES_NAME = "crossing-boundary-scores.jsonl"
OWNER_RECORDS_NAME = "crossing-boundary-owner-records.jsonl"
PARITY_NAME = "crossing-boundary-parity.json"
RECEIPT_NAME = "crossing-boundary-receipt.json"
QUARANTINE_NAME = "crossing-boundary-quarantine.json"
ADMISSION_NAME = "crossing-boundary-admission.json"

#: The two run modes.  ``smoke`` proves cached execution and batching on one
#: image that carries the whole required role matrix and seals an admission
#: receipt; ``capture`` consumes that receipt and scores per-image shards.
MODE_SMOKE = "smoke"
MODE_CAPTURE = "capture"

#: Files that are *primary evidence*.  A quarantined shard leaves none of them.
PRIMARY_OUTPUT_NAMES: tuple[str, ...] = (SCORES_NAME, OWNER_RECORDS_NAME, PARITY_NAME)

#: Wrapper / coordinate token identity.  Taken from the sealed census planner so
#: this module cannot drift from the registry it scores against; the literal
#: values are additionally pinned by the contract tests.
OBJECT_REF_START = planner.WRAPPER_TOKEN_IDS["object_ref_start"]
OBJECT_REF_END = planner.WRAPPER_TOKEN_IDS["object_ref_end"]
BOX_START = planner.WRAPPER_TOKEN_IDS["box_start"]
BOX_END = planner.WRAPPER_TOKEN_IDS["box_end"]
IM_END = planner.WRAPPER_TOKEN_IDS["im_end"]
COORDINATE_TOKEN_ID_START = planner.COORD_TOKEN_START
COORDINATE_TOKEN_ID_END_EXCLUSIVE = planner.COORD_TOKEN_END + 1
COORDINATE_TOKEN_COUNT = 4

#: Exactly four coordinate tokens then ``<|box_end|>``; nothing else is a box.
COORDINATE_ROW_TOKEN_COUNT = COORDINATE_TOKEN_COUNT + 1

#: unit.md "Exact state pair".  U owns the primary denominator; L and the
#: same-context U&L cohort are sensitivities that never replace it.
PRIMARY_OWNER_COUNT_U = plan_builder.EXPECTED_U_CROSSING_COUNT
PRIMARY_OWNER_COUNT_L = plan_builder.EXPECTED_L_CROSSING_COUNT
PRIMARY_OWNER_COUNT_SAME_CONTEXT_UL = plan_builder.EXPECTED_EXACT_UL_CROSSING_COUNT
MATCHED_E_OWNER_COUNT = plan_builder.EXPECTED_MATCHED_E_COUNT
UNMATCHED_E_OWNER_COUNT = plan_builder.EXPECTED_UNMATCHED_E_COUNT

#: unit.md "Timing controls": the audit reconstruction expects 14 owners.  A
#: prior scout note said 21; that value is wrong and must never reappear.
TIMING_CONTROL_OWNER_COUNT = plan_builder.EXPECTED_TIMING_CONTROL_COUNT
STALE_TIMING_CONTROL_OWNER_COUNT = 21
#: unit.md "Timing controls": one due-boundary native true positive per image.
TP_CALIBRATION_OWNER_COUNT = plan_builder.EXPECTED_TP_REPLAY_CONTROL_COUNT

#: unit.md "Native replay alignment".
MAX_PRIMARY_QUARANTINES = 2
CACHE_PARITY_MAX_SELECTED_LOGIT_ABS_DIFF = 1e-3

#: unit.md "Stop rule".
MIN_INTERPRETABLE_OWNERS = 20
MIN_INTERPRETABLE_MATCHED_E = 6
MIN_INTERPRETABLE_UNMATCHED_E = 7
ROUTING_MAJORITY_NUMERATOR = 2
ROUTING_MAJORITY_DENOMINATOR = 3

#: unit.md "Primary classification": the exhaustive order is 1 -> 4.
BRANCH_ORDER: tuple[str, ...] = tuple(plan_builder.BRANCH_ORDER)
PRIMARY_BRANCHES = frozenset(BRANCH_ORDER)
DISPLACED_SUBTAGS = frozenset(plan_builder.BRANCH_SUB_TAGS["displaced"])

#: unit.md "Native replay alignment": cached execution is admitted only if a
#: real matched-E, unmatched-E and same-description smoke each pass.
REQUIRED_SMOKE_MATRIX_ROLES = frozenset(
    {"matched_e_diff_desc", "unmatched_e_diff_desc", "same_desc"}
)
OPTIONAL_SMOKE_MATRIX_ROLES = frozenset({"f_compatibility", "tp_calibration"})
SMOKE_MATRIX_ROLES = REQUIRED_SMOKE_MATRIX_ROLES | OPTIONAL_SMOKE_MATRIX_ROLES

#: Decoded text is never re-tokenized to reconstruct a state.  Mirrors
#: ``score_sorted_owner_basin_landscape.FORBIDDEN_TEXT_KEYS``.
FORBIDDEN_TEXT_KEYS = frozenset(
    {"prefix_text", "generated_text", "prefix_chat_text", "chat_text", "decoded_text"}
)

BOUNDARY_LABELS = frozenset({"P", "P_plus_E"})

#: The sealed sidecar match statuses.  Only ``matched`` is a strict physical
#: owner match; ``ambiguous_neutral`` names an owner without being one.
SIDECAR_MATCH_STATUSES = frozenset(
    {
        plan_builder.SIDECAR_MATCHED,
        plan_builder.SIDECAR_UNMATCHED,
        plan_builder.SIDECAR_AMBIGUOUS_NEUTRAL,
    }
)

#: Deterministic greedy-box owner-match dispositions.
GREEDY_TARGET_MATCH = "target_match"
GREEDY_OTHER_OWNER_MATCH = "other_owner_match"
GREEDY_UNMATCHED = "unmatched"
GREEDY_AMBIGUOUS = "ambiguous_neutral"
GREEDY_MALFORMED = "malformed"
GREEDY_STATUSES = frozenset(
    {
        GREEDY_TARGET_MATCH,
        GREEDY_OTHER_OWNER_MATCH,
        GREEDY_UNMATCHED,
        GREEDY_AMBIGUOUS,
        GREEDY_MALFORMED,
    }
)
#: Greedy dispositions that are "target-missed, unmatched, or malformed".
GREEDY_TARGET_MISSED_STATUSES = frozenset(
    {GREEDY_UNMATCHED, GREEDY_AMBIGUOUS, GREEDY_MALFORMED}
)

#: Calibrated **localization support**: a property of the shape of an owner's
#: local coordinate landscape.  The predecessor census freezes this as an
#: invariant -- "rank and margin are a routing/competition surface and are never
#: a support criterion" -- so support is derived only from ``peak_lift`` and
#: ``local_concentration`` against the census's own thresholds, never from where
#: the target sits in an owner ranking.
SUPPORT_SUPPORTED = "supported"
SUPPORT_UNSUPPORTED = "unsupported"
SUPPORT_AMBIGUOUS_TIE = "ambiguous_tie"
#: No calibration is bound, so no support claim is made.  Routes to branch 4
#: rather than being silently read as either support or its absence.
SUPPORT_CALIBRATION_UNAVAILABLE = "calibration_unavailable"
SUPPORT_DISPOSITIONS = frozenset(
    {
        SUPPORT_SUPPORTED,
        SUPPORT_UNSUPPORTED,
        SUPPORT_AMBIGUOUS_TIE,
        SUPPORT_CALIBRATION_UNAVAILABLE,
    }
)

#: The *rank* surface, kept lexically distinct from support so the two can never
#: be read for one another again.
RANK_TARGET_FIRST = "target_ranks_first"
RANK_TARGET_OUTRANKED = "target_outranked"
RANK_TIE = "rank_tie"
RANK_TARGET_UNCONTESTED = "target_uncontested"
RANK_DISPOSITIONS = frozenset(
    {RANK_TARGET_FIRST, RANK_TARGET_OUTRANKED, RANK_TIE, RANK_TARGET_UNCONTESTED}
)

#: The census's frozen support calibration, reused rather than re-derived.
SUPPORT_STATISTICS: tuple[str, ...] = planner.SUPPORT_STATISTICS
SUPPORT_EPSILON = planner.SUPPORT_EPSILON
#: Where the sealed plan declares the census thresholds this unit must clear.
#: Owned by the CPU plan builder; until it seals them, support is reported as
#: :data:`SUPPORT_CALIBRATION_UNAVAILABLE` and no branch claims support.
SUPPORT_CALIBRATION_KEY = "support_calibration"
SUPPORT_CALIBRATION_REQUIRED_FIELDS: tuple[str, ...] = (
    "theta_peak_lift",
    "theta_local_concentration",
)
#: One physical candidate scored twice from the same root must agree to within
#: float noise; a larger gap is a request/batching alignment failure, not a
#: duplicate to be silently collapsed.
POPULATION_DUPLICATE_MAX_ABS_DIFF = 1e-9
#: The plan builder's published support-calibration contract.  Bound by name so
#: a future re-cut of that contract cannot be consumed as if it were this one.
SUPPORT_CALIBRATION_CONTRACT_ID = plan_builder.SUPPORT_CONTRACT_ID

#: The frozen one-box matcher threshold this unit inherits from the census.
OWNER_MATCH_IOU_THRESHOLD = planner.IOU_THRESHOLD

#: The only repetition-penalty stratum this unit may execute under.  Raw fp32
#: logits are the evidence channel; a penalised policy view is never one.
NATIVE_REPETITION_PENALTY_STRATUM = planner.NATIVE_REPETITION_PENALTY_STRATUM
LIKELIHOOD_CHANNEL = (
    "raw_fp32_lm_head_log_softmax_over_the_full_vocabulary_no_repetition_penalty"
)

KV_CACHE_BACKEND = "kv_cache"
UNCACHED_BACKEND = "full_reforward_uncached"
CACHE_ADMITTED = "cache_admitted"
UNCACHED_FALLBACK = "uncached_fallback"

#: Ladder surfaces that fall back to uncached scoring independently.
SURFACE_RELEASE = "natural_release"
SURFACE_COORDINATE = "coordinate_realization"
SURFACES: tuple[str, ...] = (SURFACE_RELEASE, SURFACE_COORDINATE)

#: The plan's per-context variant names, mapped onto the two decision labels.
BOUNDARY_LABEL_BY_VARIANT: Mapping[str, str] = {
    "at_p": "P",
    "at_p_plus_e": "P_plus_E",
    "at_control_boundary": "P",
    "at_control_boundary_plus_row": "P_plus_E",
    "at_due_boundary": "P",
}

#: Batch sizes are configurable; batching is a throughput device that may only
#: be enabled after the semantic smoke has already passed.
DEFAULT_BATCH_SIZE = 1
BATCH_PARITY_MAX_ABS_DIFF = 1e-4

#: Every request family the sealed plan can emit.
REQUEST_NATURAL_RELEASE = plan_builder.REQUEST_NATURAL_RELEASE
REQUEST_NATIVE_NEXT_ACTION = plan_builder.REQUEST_NATIVE_NEXT_ACTION
REQUEST_COORDINATE_TARGET_LOCAL = plan_builder.REQUEST_COORDINATE_TARGET_LOCAL
REQUEST_COORDINATE_COMPETITOR = plan_builder.REQUEST_COORDINATE_COMPETITOR
REQUEST_COORDINATE_GREEDY = plan_builder.REQUEST_COORDINATE_GREEDY
REQUEST_DOWNSTREAM_COMPATIBILITY = plan_builder.REQUEST_DOWNSTREAM_COMPATIBILITY

RELEASE_FAMILIES = frozenset({REQUEST_NATURAL_RELEASE, REQUEST_NATIVE_NEXT_ACTION})
COORDINATE_FAMILIES = frozenset(
    {
        REQUEST_COORDINATE_TARGET_LOCAL,
        REQUEST_COORDINATE_COMPETITOR,
        REQUEST_COORDINATE_GREEDY,
    }
)
PRIMARY_READOUT_TIER = "primary"


class CrossingBoundaryContractError(RuntimeError):
    """A precondition of this unit's deterministic capture was not proven."""


def _fail(message: str) -> NoReturn:
    raise CrossingBoundaryContractError(message)


# ---------------------------------------------------------------------------
# 1. Digests (self-contained; identical canonical form to every sibling pair)
# ---------------------------------------------------------------------------


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _token_ids(value: Any, *, label: str) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        _fail(f"{label} is not a token-id sequence")
    tokens: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            _fail(f"{label} carries a non-integer token id {item!r}")
        tokens.append(int(item))
    return tokens


def assert_no_retokenizable_text(payload: Any, *, label: str) -> None:
    """Refuse any decoded-prose key, at any depth, as a token source."""

    if isinstance(payload, Mapping):
        present = sorted(FORBIDDEN_TEXT_KEYS & set(map(str, payload)))
        if present:
            _fail(
                f"{label} carries re-tokenizable text key(s) {present!r}; only literal "
                "token ids may reconstruct a state"
            )
        for value in payload.values():
            assert_no_retokenizable_text(value, label=label)
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        for value in payload:
            assert_no_retokenizable_text(value, label=label)


# ---------------------------------------------------------------------------
# 2. Registry token capture and full-row E suffix identity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegistryTokenSpan:
    """One literal, digest-verified token span read from a sealed registry."""

    row_id: str
    context_id: str
    token_ids: tuple[int, ...]
    token_ids_sha256: str
    label: str
    registry_path: str
    registry_sha256: str


def load_registry_token_span(
    row: Mapping[str, Any],
    *,
    registry_path: Path,
    registry_sha256: str,
    label: str,
) -> RegistryTokenSpan:
    """Bind one registry row's literal tokens to the registry file it came from.

    The source file digest is re-derived rather than trusted, decoded prose is
    refused, and the row's own token digest must reconstruct.
    """

    path = Path(registry_path)
    observed = sha256_file(path)
    if observed != str(registry_sha256):
        _fail(
            f"{label}: registry {path} hashes to {observed}, not the declared "
            f"{registry_sha256}; the token source is not the sealed one"
        )
    assert_no_retokenizable_text(row, label=f"{label}: registry row")
    tokens = _token_ids(row.get("token_ids"), label=f"{label}: token_ids")
    if not tokens:
        _fail(f"{label}: registry row carries an empty token span")
    declared = row.get("token_ids_sha256")
    if sha256_json(tokens) != declared:
        _fail(f"{label}: registry row token digest does not reconstruct")
    return RegistryTokenSpan(
        row_id=str(row.get("row_id")),
        context_id=str(row.get("context_id")),
        token_ids=tuple(tokens),
        token_ids_sha256=str(declared),
        label=str(label),
        registry_path=str(path),
        registry_sha256=str(registry_sha256),
    )


def derive_e_row_suffix(
    *, p_tokens: Sequence[int], p_plus_e_tokens: Sequence[int]
) -> tuple[int, ...]:
    """``tokens(P+E) - tokens(P)``: the only owner of full-row ``E`` identity."""

    p_literal = [int(v) for v in p_tokens]
    pe_literal = [int(v) for v in p_plus_e_tokens]
    if len(pe_literal) <= len(p_literal):
        _fail(
            "P+E must be strictly longer than P; an empty or negative row suffix cannot "
            "own row E's token identity"
        )
    if pe_literal[: len(p_literal)] != p_literal:
        _fail(
            "P is not a literal token prefix of P+E; the adjacent-context row attribution "
            "is unsafe"
        )
    return tuple(pe_literal[len(p_literal) :])


@dataclass(frozen=True)
class ERowBinding:
    """The sidecar validated against the literal ``P+E`` minus ``P`` suffix."""

    row_id: str
    row_index: int
    coord_token_ids: tuple[int, int, int, int]
    coord_token_ids_sha256: str
    coord_offset_in_suffix: int
    strict_match_status: str
    raw_span_digest: str
    full_row_suffix: tuple[int, ...]
    full_row_suffix_sha256: str


def validate_e_row_binding(
    sidecar: Mapping[str, Any],
    *,
    p_tokens: Sequence[int],
    p_plus_e_tokens: Sequence[int],
    coordinate_domain: range,
) -> ERowBinding:
    """Validate the ``E`` sidecar against the literal row suffix it describes.

    The sidecar is a *validator*, never a token source: it owns only ``E``'s
    coordinate subsequence/digest, row id/index, strict-match fields and
    raw-span digest.
    """

    assert_no_retokenizable_text(sidecar, label="E sidecar row")
    suffix = derive_e_row_suffix(p_tokens=p_tokens, p_plus_e_tokens=p_plus_e_tokens)
    coord = tuple(_token_ids(sidecar.get("coord_token_ids"), label="E sidecar coord_token_ids"))
    if len(coord) != COORDINATE_TOKEN_COUNT:
        _fail(
            f"E sidecar declares {len(coord)} coordinate tokens; exactly "
            f"{COORDINATE_TOKEN_COUNT} are required"
        )
    if sha256_json(list(coord)) != sidecar.get("coord_token_ids_sha256"):
        _fail("E sidecar coordinate digest does not reconstruct")
    outside = [token for token in coord if token not in coordinate_domain]
    if outside:
        _fail(
            f"E sidecar coordinate token(s) {outside!r} are outside the frozen coordinate "
            "domain"
        )
    occurrences = [
        index
        for index in range(len(suffix) - COORDINATE_TOKEN_COUNT + 1)
        if tuple(suffix[index : index + COORDINATE_TOKEN_COUNT]) == coord
    ]
    if len(occurrences) != 1:
        _fail(
            f"E sidecar coordinate tokens occur {len(occurrences)} times in the literal row "
            "suffix; exactly one occurrence is required"
        )
    status = str(sidecar.get("strict_match_status"))
    if status not in SIDECAR_MATCH_STATUSES:
        _fail(f"E sidecar carries unknown strict_match_status {status!r}")
    return ERowBinding(
        row_id=str(sidecar.get("row_id")),
        row_index=int(sidecar["row_index"]),
        coord_token_ids=(coord[0], coord[1], coord[2], coord[3]),
        coord_token_ids_sha256=str(sidecar.get("coord_token_ids_sha256")),
        coord_offset_in_suffix=occurrences[0],
        strict_match_status=status,
        raw_span_digest=str(sidecar.get("raw_span_digest")),
        full_row_suffix=suffix,
        full_row_suffix_sha256=sha256_json(list(suffix)),
    )


# ---------------------------------------------------------------------------
# 3. Ladder 1 -- natural description release
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NaturalReleaseObservation:
    """Ladder-1 observables at one boundary.  No branch is decided here."""

    boundary_label: str
    same_description: bool
    first_divergence_index: int | None
    target_minus_native_margin: float | None
    description_sum: float
    description_token_mean: float
    description_token_count: int
    argmax_follows_target: bool
    target_first_divergent_token_id: int | None
    native_first_divergent_token_id: int | None


def score_natural_release(
    *,
    boundary_label: str,
    target_token_ids: Sequence[int],
    native_token_ids: Sequence[int],
    target_description_logprobs: Sequence[float],
    native_description_logprobs: Sequence[float],
    target_argmax_ids: Sequence[int],
) -> NaturalReleaseObservation:
    """Compare the exact target description path against the exact native action.

    Identical descriptions are *not* a separate factorial level: they simply
    have no observable description divergence, so the margin fields are absent
    and the observation is coordinate-only.
    """

    if str(boundary_label) not in BOUNDARY_LABELS:
        _fail(
            f"unknown boundary label {boundary_label!r}; only {sorted(BOUNDARY_LABELS)!r} "
            "are decision contexts of this unit"
        )
    target = [int(v) for v in target_token_ids]
    native = [int(v) for v in native_token_ids]
    target_logprobs = [float(v) for v in target_description_logprobs]
    native_logprobs = [float(v) for v in native_description_logprobs]
    argmax_ids = [int(v) for v in target_argmax_ids]
    if not target:
        _fail("the target description path is empty")
    if len(target) != len(target_logprobs):
        _fail("target token ids and target description log-probabilities differ in length")
    if len(native) != len(native_logprobs):
        _fail("native token ids and native description log-probabilities differ in length")
    if len(argmax_ids) != len(target):
        _fail("one deterministic argmax token id is required per target description token")

    same_description = target == native
    divergence: int | None = None
    margin: float | None = None
    target_divergent: int | None = None
    native_divergent: int | None = None
    if not same_description:
        limit = min(len(target), len(native))
        divergence = next(
            (index for index in range(limit) if target[index] != native[index]), limit
        )
        if divergence < limit:
            margin = target_logprobs[divergence] - native_logprobs[divergence]
            target_divergent = target[divergence]
            native_divergent = native[divergence]
        elif divergence < len(target):
            # One path is a strict prefix of the other: the divergence is the
            # first token only one of them scores, so no paired margin exists.
            target_divergent = target[divergence]
    return NaturalReleaseObservation(
        boundary_label=str(boundary_label),
        same_description=same_description,
        first_divergence_index=divergence,
        target_minus_native_margin=margin,
        description_sum=math.fsum(target_logprobs),
        description_token_mean=math.fsum(target_logprobs) / len(target_logprobs),
        description_token_count=len(target),
        argmax_follows_target=argmax_ids == target,
        target_first_divergent_token_id=target_divergent,
        native_first_divergent_token_id=native_divergent,
    )


# ---------------------------------------------------------------------------
# 4. Ladder 2 -- the exact four-coordinate + box_end grammar
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoordinateGrammarResult:
    """A coordinate row, validated without ever being repaired."""

    status: str
    token_ids: tuple[int, ...]
    coord_token_ids: tuple[int, ...]
    malformed_reason: str | None


def validate_coordinate_grammar(
    token_ids: Sequence[int], *, coordinate_domain: range, box_end_token_id: int
) -> CoordinateGrammarResult:
    """Exactly four coordinate-domain tokens, then exactly ``<|box_end|>``.

    Any other token, arity, premature termination, extra coordinate, or invalid
    box is ``malformed``.  A malformed result keeps the *literal* offending
    tokens; it is never truncated or padded into a plausible reconstruction.
    """

    tokens = tuple(int(v) for v in token_ids)
    head = tokens[:COORDINATE_TOKEN_COUNT]
    terminal = int(box_end_token_id)

    if not tokens or tokens[-1] != terminal:
        return CoordinateGrammarResult(
            status="malformed",
            token_ids=tokens,
            coord_token_ids=head,
            malformed_reason=(
                "premature termination: the coordinate row does not close with the exact "
                "box_end token"
            ),
        )
    if len(tokens) - 1 != COORDINATE_TOKEN_COUNT:
        return CoordinateGrammarResult(
            status="malformed",
            token_ids=tokens,
            coord_token_ids=head,
            malformed_reason=(
                f"arity: {len(tokens) - 1} tokens precede box_end, not the frozen "
                f"{COORDINATE_TOKEN_COUNT}"
            ),
        )
    outside = [token for token in head if token not in coordinate_domain]
    if outside:
        return CoordinateGrammarResult(
            status="malformed",
            token_ids=tokens,
            coord_token_ids=head,
            malformed_reason=f"domain: token(s) {outside!r} are not coordinate tokens",
        )
    return CoordinateGrammarResult(
        status="valid", token_ids=tokens, coord_token_ids=head, malformed_reason=None
    )


def greedy_decode_coordinate_row(
    argmax_token_fn: Callable[[int, tuple[int, ...]], int],
    *,
    coordinate_domain: range,
    box_end_token_id: int,
) -> CoordinateGrammarResult:
    """Deterministic coordinate-only decode: five explicit argmax steps, no more.

    ``argmax_token_fn(step_index, tokens_so_far)`` is the only source of a
    token.  There is no temperature, top-p/k, sampler or ``generate()``
    parameter to pass, and the loop fails closed at the first grammar violation
    instead of probing further steps.
    """

    terminal = int(box_end_token_id)
    tokens: list[int] = []
    for step_index in range(COORDINATE_ROW_TOKEN_COUNT):
        token = int(argmax_token_fn(step_index, tuple(tokens)))
        tokens.append(token)
        if step_index < COORDINATE_TOKEN_COUNT:
            if token not in coordinate_domain:
                return CoordinateGrammarResult(
                    status="malformed",
                    token_ids=tuple(tokens),
                    coord_token_ids=tuple(tokens[:COORDINATE_TOKEN_COUNT]),
                    malformed_reason=(
                        f"domain: step {step_index} decoded {token}, which is not a "
                        "coordinate token"
                    ),
                )
        elif token != terminal:
            return CoordinateGrammarResult(
                status="malformed",
                token_ids=tuple(tokens),
                coord_token_ids=tuple(tokens[:COORDINATE_TOKEN_COUNT]),
                malformed_reason=(
                    f"premature termination: the terminal step decoded {token} rather than "
                    "the exact box_end token"
                ),
            )
    return validate_coordinate_grammar(
        tokens, coordinate_domain=coordinate_domain, box_end_token_id=terminal
    )


# ---------------------------------------------------------------------------
# 5. Ladder 2 -- target-conditioned candidate ranking (within-context only)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateScore:
    """One owner-identifiable candidate's complete-box likelihood."""

    candidate_id: str
    owner_id: str
    context_id: str
    complete_box_logprob_sum: float


@dataclass(frozen=True)
class OwnerRankResult:
    """Owner-level rank/competitor/margin inside exactly one context.

    Deliberately carries **no** support field.  Rank and margin describe
    competition between owners; localization support describes the shape of one
    owner's own landscape.  Aliasing the two makes ``realization_fail``
    unreachable (every rank-unsupported case is already ``likelihood_displaced``)
    and inflates ``release_lost`` with every target that merely ranks first.
    """

    target_owner_id: str
    target_rank: int
    best_competitor_owner_id: str | None
    target_minus_competitor_margin: float | None
    family_rank_disposition: str


def rank_owner_candidates(
    candidates: Sequence[CandidateScore], *, target_owner_id: str
) -> OwnerRankResult:
    """Rank *owners* by their best candidate, never pooling across contexts."""

    if not candidates:
        _fail("owner ranking requires at least one candidate score")
    context_ids = {candidate.context_id for candidate in candidates}
    if len(context_ids) != 1:
        _fail(
            f"owner ranking mixes contexts {sorted(context_ids)!r}; raw log probabilities "
            "are only comparable within one context"
        )
    best_by_owner: dict[str, float] = {}
    for candidate in candidates:
        score = float(candidate.complete_box_logprob_sum)
        current = best_by_owner.get(candidate.owner_id)
        if current is None or score > current:
            best_by_owner[candidate.owner_id] = score
    if str(target_owner_id) not in best_by_owner:
        _fail(
            f"target owner {target_owner_id!r} has no candidate in this context; the "
            "target-local family cannot be ranked"
        )

    target_score = best_by_owner[str(target_owner_id)]
    ordered = sorted(best_by_owner.items(), key=lambda item: (-item[1], item[0]))
    target_rank = 1 + sum(1 for _, score in ordered if score > target_score)

    competitors = [
        (owner_id, score)
        for owner_id, score in ordered
        if owner_id != str(target_owner_id)
    ]
    if not competitors:
        return OwnerRankResult(
            target_owner_id=str(target_owner_id),
            target_rank=target_rank,
            best_competitor_owner_id=None,
            target_minus_competitor_margin=None,
            family_rank_disposition=RANK_TARGET_UNCONTESTED,
        )
    competitor_id, competitor_score = competitors[0]
    margin = target_score - competitor_score
    if margin > 0.0:
        disposition = RANK_TARGET_FIRST
    elif margin < 0.0:
        disposition = RANK_TARGET_OUTRANKED
    else:
        disposition = RANK_TIE
    return OwnerRankResult(
        target_owner_id=str(target_owner_id),
        target_rank=target_rank,
        best_competitor_owner_id=competitor_id,
        target_minus_competitor_margin=margin,
        family_rank_disposition=disposition,
    )


# ---------------------------------------------------------------------------
# 5b. Calibrated target-local support (the census's own frozen statistics)
# ---------------------------------------------------------------------------

#: unit.md: "U owns primary support, rank, competitor, and branch fields.  L is
#: a sensitivity reported beside U and never changes the U branch."
SUPPORT_BOUND_U = "u"
SUPPORT_BOUND_L = "l"
SUPPORT_BOUNDS: tuple[str, ...] = (SUPPORT_BOUND_U, SUPPORT_BOUND_L)

#: The census's sealed owner-support partition, reproduced exactly:
#: ``strict_assigned_self`` counts under both bounds; ``ambiguous_neutral`` and
#: ``unmatched`` count under U only; a probe that strict-matches *another* owner
#: is excluded from both and belongs to the collision diagnostic.  Leaving
#: unmatched probes in L would let a perturbation that landed on nothing
#: manufacture a conservative support claim.
SUPPORT_BOUND_MEMBERSHIP: Mapping[str, frozenset[str]] = {
    SUPPORT_BOUND_L: frozenset({"strict_assigned_self"}),
    SUPPORT_BOUND_U: frozenset(
        {"strict_assigned_self", "ambiguous_upper", "unmatched_generator_local"}
    ),
}
#: The census's own partition vocabulary, so an added partition cannot silently
#: fall outside both bounds.
SUPPORT_PARTITIONS: tuple[str, ...] = tuple(planner.OWNER_CANDIDATE_PARTITIONS)


@dataclass(frozen=True)
class BankProbe:
    """One scored probe of the target owner's generator-local bank."""

    candidate_id: str
    complete_box_logprob_sum: float
    bank_class: str


@dataclass(frozen=True)
class SupportBoundFeatures:
    """The census's two support statistics for one owner under one bound."""

    bound: str
    owner_best: float | None
    owner_best_candidate_id: str | None
    peak_lift: float | None
    local_concentration: float | None
    bank_median: float | None
    bank_size: int
    unique_population_size: int


@dataclass(frozen=True)
class TargetLocalSupportResult:
    """Calibrated localization support -- never a rank or margin statement."""

    bound: str
    support_disposition: str
    features: SupportBoundFeatures
    peak_lift_threshold: float | None
    local_concentration_threshold: float | None
    epsilon: float
    calibration_source: str


def classify_bank_probe(
    *, strict_assignment_status: str, strict_assignment_gt_owner_id: Any, target_owner_id: str
) -> str:
    """Which bound partition one generator-local probe belongs to.

    Analyzer-facing only.  The capture path does **not** derive membership this
    way: it reads the plan's sealed per-bound membership, which is the frozen
    census classification of each probe against *this* owner.  A candidate's
    global strict assignment answers a different question, so re-deriving from
    it here would not reproduce the sealed partition.
    """

    status = str(strict_assignment_status)
    if status == plan_builder.SIDECAR_MATCHED:
        return (
            "strict_assigned_self"
            if str(strict_assignment_gt_owner_id) == str(target_owner_id)
            else "other_owner_strict"
        )
    if status == plan_builder.SIDECAR_AMBIGUOUS_NEUTRAL:
        return "ambiguous_upper"
    if status == plan_builder.SIDECAR_UNMATCHED:
        return "unmatched_generator_local"
    _fail(f"unknown strict assignment status {strict_assignment_status!r} on a bank probe")


def unique_population_log_posteriors(
    entries: Sequence[tuple[str, float]],
) -> tuple[dict[str, float], int]:
    """Within-(context, category) log posteriors over the *unique* population.

    The population is the union of the target-local and same-category competitor
    families at one ``(context, normalized_description)``, deduplicated by
    physical candidate id exactly as the census merge does: the same candidate
    reached through both families is one member, and counting it twice would
    flatten every owner's ``peak_lift``.
    """

    best_by_key: dict[str, float] = {}
    for key, score in entries:
        current = best_by_key.get(key)
        if current is None:
            best_by_key[key] = float(score)
            continue
        # The same physical candidate reached through both families was scored
        # from the same root under the same forced ``D_C``, so its two readings
        # must agree.  Keeping the maximum instead would let a broken request or
        # batching alignment pass as a deduplication.
        if abs(current - float(score)) > POPULATION_DUPLICATE_MAX_ABS_DIFF:
            _fail(
                f"candidate {key!r} was scored twice in one query group with differing "
                f"complete-box sums ({current!r} vs {float(score)!r}, tolerance "
                f"{POPULATION_DUPLICATE_MAX_ABS_DIFF}); the two families did not score the "
                "same candidate from the same root"
            )
    if not best_by_key:
        _fail("peak_lift requires a non-empty unique candidate population")
    values = list(best_by_key.values())
    peak = max(values)
    log_sum_exp = peak + math.log(math.fsum(math.exp(value - peak) for value in values))
    return (
        {key: value - log_sum_exp for key, value in best_by_key.items()},
        len(best_by_key),
    )


def _median(values: Sequence[float]) -> float:
    ordered = sorted(float(value) for value in values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def compute_support_bound_features(
    probes: Sequence[BankProbe],
    *,
    bound: str,
    log_posterior_by_key: Mapping[str, float],
    unique_population_size: int,
) -> SupportBoundFeatures:
    """``peak_lift``/``local_concentration`` exactly as the census merge defines them.

    ``peak_lift = log_posterior(owner best) + log(N_unique)``;
    ``local_concentration = owner best - median(owner's own bank under this bound)``.
    Neither consults an owner ranking.
    """

    if str(bound) not in SUPPORT_BOUND_MEMBERSHIP:
        _fail(f"unknown support bound {bound!r}")
    members = [
        probe
        for probe in probes
        if probe.bank_class in SUPPORT_BOUND_MEMBERSHIP[str(bound)]
    ]
    if not members:
        return SupportBoundFeatures(
            bound=str(bound),
            owner_best=None,
            owner_best_candidate_id=None,
            peak_lift=None,
            local_concentration=None,
            bank_median=None,
            bank_size=0,
            unique_population_size=int(unique_population_size),
        )
    top = max(members, key=lambda probe: (probe.complete_box_logprob_sum, probe.candidate_id))
    log_posterior = log_posterior_by_key.get(top.candidate_id)
    if log_posterior is None:
        _fail(
            f"the owner's best {bound!r}-bound probe {top.candidate_id!r} is absent from the "
            "unique query-group population; peak_lift would be computed against a population "
            "that does not contain it"
        )
    median = _median([probe.complete_box_logprob_sum for probe in members])
    return SupportBoundFeatures(
        bound=str(bound),
        owner_best=top.complete_box_logprob_sum,
        owner_best_candidate_id=top.candidate_id,
        peak_lift=float(log_posterior) + math.log(float(unique_population_size)),
        local_concentration=top.complete_box_logprob_sum - median,
        bank_median=median,
        bank_size=len(members),
        unique_population_size=int(unique_population_size),
    )


def evaluate_target_local_support(
    features: SupportBoundFeatures,
    *,
    peak_lift_threshold: float | None,
    local_concentration_threshold: float | None,
    epsilon: float = SUPPORT_EPSILON,
    calibration_source: str,
) -> TargetLocalSupportResult:
    """Calibrated support: both statistics must clear their census threshold.

    Delegates the decision to the census planner's own :func:`clears_support`,
    so this unit cannot drift from the calibration it claims to reuse.  Without
    a bound threshold no support claim is made at all -- the disposition is
    ``calibration_unavailable``, which routes to branch 4 rather than being read
    as either support or its absence.
    """

    unavailable = (
        peak_lift_threshold is None
        or local_concentration_threshold is None
        or features.peak_lift is None
        or features.local_concentration is None
    )
    if unavailable:
        disposition = SUPPORT_CALIBRATION_UNAVAILABLE
    elif planner.clears_support(
        peak_lift=float(features.peak_lift),
        local_concentration=float(features.local_concentration),
        peak_lift_threshold=float(peak_lift_threshold),
        local_concentration_threshold=float(local_concentration_threshold),
        epsilon=float(epsilon),
    ):
        disposition = SUPPORT_SUPPORTED
    else:
        disposition = SUPPORT_UNSUPPORTED
    return TargetLocalSupportResult(
        bound=features.bound,
        support_disposition=disposition,
        features=features,
        peak_lift_threshold=(
            None if peak_lift_threshold is None else float(peak_lift_threshold)
        ),
        local_concentration_threshold=(
            None
            if local_concentration_threshold is None
            else float(local_concentration_threshold)
        ),
        epsilon=float(epsilon),
        calibration_source=str(calibration_source),
    )


def read_request_support_calibration(request: Mapping[str, Any]) -> dict[str, Any] | None:
    """The thresholds the sealed plan binds to *this* target-local request.

    The plan builder copies these from the sealed census calibration receipt
    after proving its digest chain, and republishes them per request; reading
    them here keeps the threshold bound to the exact family being scored rather
    than to a manifest-level value that a request need not have used.  Nothing
    is re-derived: an unknown contract, a missing threshold, or a block that
    declares rank or margin to be a support input fails closed.
    """

    family = request.get("candidate_family")
    if not isinstance(family, Mapping):
        return None
    block = family.get("support_calibration")
    if block is None:
        return None
    if not isinstance(block, Mapping):
        _fail("a request's candidate_family.support_calibration is not a mapping")
    contract_id = str(block.get("support_contract_id"))
    if contract_id != SUPPORT_CALIBRATION_CONTRACT_ID:
        _fail(
            f"the sealed request publishes support calibration contract {contract_id!r}, not "
            f"{SUPPORT_CALIBRATION_CONTRACT_ID!r}; the threshold semantics are not proven"
        )
    if bool(block.get("rank_is_a_support_input")) or bool(
        block.get("margin_is_a_support_input")
    ):
        _fail(
            "the sealed calibration declares rank or margin to be a support input; this unit "
            "refuses to score under a calibration that contradicts the frozen census invariant"
        )
    missing = sorted(
        field
        for field in ("theta_peak_lift", "theta_local_concentration")
        if block.get(field) is None
    )
    if missing:
        _fail(
            f"the sealed request's support calibration is missing {missing!r}; a partial "
            "calibration is never completed with a default threshold"
        )
    return {
        "peak_lift_threshold": float(block["theta_peak_lift"]),
        "local_concentration_threshold": float(block["theta_local_concentration"]),
        "epsilon": float(block.get("epsilon", SUPPORT_EPSILON)),
        "source": (
            f"sealed_census_calibration_sha256:{block.get('calibration_sha256')}"
        ),
        "contract_id": contract_id,
        "primary_bound": str(block.get("primary_bound", SUPPORT_BOUND_U)),
    }


def sealed_bound_candidate_ids(request: Mapping[str, Any], *, bound: str) -> list[str]:
    """The plan's own membership for one ambiguity bound, digest-verified.

    The partition is the frozen census classification of each probe against
    *this* owner, so it is read from the plan rather than re-derived from a
    candidate's global strict assignment, which answers a different question.
    """

    family = request.get("candidate_family")
    if not isinstance(family, Mapping):
        _fail("the target-local request carries no candidate_family")
    bounds = family.get("bounds")
    if not isinstance(bounds, Mapping):
        _fail("the target-local request seals no per-bound support membership")
    block = bounds.get(str(bound))
    if not isinstance(block, Mapping):
        _fail(f"the target-local request seals no membership for bound {bound!r}")
    candidate_ids = [str(value) for value in (block.get("candidate_ids") or ())]
    if sha256_json(candidate_ids) != block.get("candidate_ids_sha256"):
        _fail(f"the sealed {bound!r}-bound membership digest does not reconstruct")
    return candidate_ids


def read_plan_support_calibration(plan: SealedPlan) -> dict[str, Any] | None:
    """The manifest-level support-calibration *lineage*, for the receipt.

    The thresholds a capture actually scores under are read per request by
    :func:`read_request_support_calibration`; this reader exists so the shard
    receipt can name the calibration those thresholds came from -- contract,
    census digest, quantile and observation count -- instead of leaving that
    provenance implicit inside the plan manifest digest.

    Nothing is re-derived.  An absent block reports unavailable; an unknown
    contract, a missing threshold, or a block that declares rank or margin to be
    a support input fails closed.
    """

    block = (plan.manifest or {}).get(SUPPORT_CALIBRATION_KEY)
    if block is None:
        return None
    if not isinstance(block, Mapping):
        _fail(f"the sealed plan's {SUPPORT_CALIBRATION_KEY!r} block is not a mapping")
    contract_id = str(block.get("support_contract_id"))
    if contract_id != SUPPORT_CALIBRATION_CONTRACT_ID:
        _fail(
            f"the sealed plan publishes support calibration contract {contract_id!r}, not "
            f"{SUPPORT_CALIBRATION_CONTRACT_ID!r}; the threshold semantics are not proven"
        )
    rank_and_margin = block.get("rank_and_margin")
    if isinstance(rank_and_margin, Mapping) and (
        bool(rank_and_margin.get("rank_is_a_support_input"))
        or bool(rank_and_margin.get("margin_is_a_support_input"))
    ):
        _fail(
            "the sealed plan's support calibration declares rank or margin to be a support "
            "input; this unit refuses to score under a calibration that contradicts the "
            "frozen census invariant"
        )
    thresholds = block.get("thresholds")
    if not isinstance(thresholds, Mapping):
        _fail("the sealed plan's support calibration carries no thresholds block")
    missing = sorted(
        field
        for field in SUPPORT_CALIBRATION_REQUIRED_FIELDS
        if thresholds.get(field) is None
    )
    if missing:
        _fail(
            f"the sealed plan's {SUPPORT_CALIBRATION_KEY!r} thresholds are missing "
            f"{missing!r}; a partial calibration is never completed with a default threshold"
        )
    source = block.get("source") if isinstance(block.get("source"), Mapping) else {}
    return {
        "support_contract_id": contract_id,
        "criterion_id": block.get("criterion_id"),
        "derived_here": bool(block.get("derived_here", False)),
        "primary_bound": (block.get("bounds") or {}).get("primary_bound"),
        "sensitivity_bound": (block.get("bounds") or {}).get("sensitivity_bound"),
        "theta_peak_lift": float(thresholds["theta_peak_lift"]),
        "theta_local_concentration": float(thresholds["theta_local_concentration"]),
        "epsilon": float(thresholds.get("epsilon", SUPPORT_EPSILON)),
        "quantile": thresholds.get("quantile"),
        "observation_count": thresholds.get("observation_count"),
        "calibration_stratum": thresholds.get("calibration_stratum"),
        "calibration_sha256": source.get("calibration_sha256"),
        "calibration_source_path": source.get("path"),
        "calibration_run_root": source.get("run_root"),
        "merge_source_sha256": source.get("merge_source_sha256"),
    }


# ---------------------------------------------------------------------------
# 6. Pure branch helpers (analyzer-facing; never written onto an owner record)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DisplacementResult:
    """Both displacement sub-tags, retained independently, plus disagreement."""

    likelihood_displaced: bool
    likelihood_displaced_owner_id: str | None
    greedy_displaced: bool
    greedy_displaced_owner_id: str | None
    decoding_contradicted: bool


def classify_displacement(
    *,
    target_owner_id: str,
    owner_rank: OwnerRankResult,
    greedy_status: str,
    greedy_owner_match: str | None,
) -> DisplacementResult:
    """unit.md branch 1: ``likelihood_displaced`` and ``greedy_displaced``.

    The two sub-tags are never collapsed.  When the candidate landscape says
    another owner wins while the deterministic greedy box still strict-matches
    the target, the case is additionally tagged ``decoding_contradicted`` --
    the sensitivity cell the stop rule reports separately.
    """

    status = str(greedy_status)
    if status not in GREEDY_STATUSES:
        _fail(f"unknown greedy owner-match status {greedy_status!r}")
    match = None if greedy_owner_match is None else str(greedy_owner_match)
    if status == GREEDY_OTHER_OWNER_MATCH and (match is None or match == str(target_owner_id)):
        _fail("a greedy other-owner match must name a physical owner other than the target")
    if status == GREEDY_TARGET_MATCH and match != str(target_owner_id):
        _fail("a greedy target match must name the target owner")

    # unit.md branch 1 is a *competition* statement: the U best identifiable
    # candidate belongs to another named physical owner at a strictly negative
    # target-minus-competitor margin.  It is deliberately independent of
    # calibrated support -- conditioning it on support would collapse the two
    # surfaces again and make ``realization_fail`` unreachable.
    competitor = owner_rank.best_competitor_owner_id
    margin = owner_rank.target_minus_competitor_margin
    likelihood_displaced = bool(
        competitor is not None
        and competitor != str(target_owner_id)
        and margin is not None
        and margin < 0.0
    )
    greedy_displaced = status == GREEDY_OTHER_OWNER_MATCH
    return DisplacementResult(
        likelihood_displaced=likelihood_displaced,
        likelihood_displaced_owner_id=competitor if likelihood_displaced else None,
        greedy_displaced=greedy_displaced,
        greedy_displaced_owner_id=match if greedy_displaced else None,
        decoding_contradicted=bool(likelihood_displaced and status == GREEDY_TARGET_MATCH),
    )


@dataclass(frozen=True)
class PrimaryBranchResult:
    """At most one branch, plus the predicates that decided it."""

    branch: str
    reasons: tuple[str, ...]


def classify_primary_branch(
    *,
    displacement: DisplacementResult,
    release_observable: bool,
    release_margin: float | None,
    forced_dc_support_disposition: str,
    greedy_status: str,
    tie_or_nonunique: bool,
    missing_fields: bool,
    boundary_label: str = "P_plus_E",
) -> PrimaryBranchResult:
    """unit.md "Primary classification": the exhaustive order 1 -> 4.

    Only ``P+E`` is decision-bearing.  At ``P`` a same-description coordinate
    readout is construction-determined -- ``D_C`` is already the exact prefix of
    native row ``E`` -- so it is recorded for replay and never classified.
    """

    if str(boundary_label) != "P_plus_E":
        _fail(
            f"primary branch assignment is only decision-bearing at P+E, not at "
            f"{boundary_label!r}; the P readout is recorded for replay only"
        )
    status = str(greedy_status)
    if status not in GREEDY_STATUSES:
        _fail(f"unknown greedy owner-match status {greedy_status!r}")
    support = str(forced_dc_support_disposition)
    if support not in SUPPORT_DISPOSITIONS:
        _fail(f"unknown forced-D_C support disposition {forced_dc_support_disposition!r}")

    # Branch 1 -- displaced.  Either sub-tag being uniquely true is enough; the
    # sub-tags already encode uniqueness (a tie is never ``unsupported`` and an
    # ambiguous box is never ``other_owner_match``).
    if displacement.likelihood_displaced or displacement.greedy_displaced:
        reasons = []
        if displacement.likelihood_displaced:
            reasons.append(
                "likelihood_displaced: the U best candidate belongs to "
                f"{displacement.likelihood_displaced_owner_id!r} at a strictly negative "
                "target-minus-competitor margin"
            )
        if displacement.greedy_displaced:
            reasons.append(
                "greedy_displaced: the coordinate-only greedy box strict-matches "
                f"{displacement.greedy_displaced_owner_id!r}"
            )
        if displacement.decoding_contradicted:
            reasons.append(
                "decoding_contradicted: the greedy box still strict-matches the target"
            )
        return PrimaryBranchResult(branch="displaced", reasons=tuple(reasons))

    # Branches 2 and 3 need determinable predicates.  A tie, a nonunique owner
    # match or a missing score field makes them undeterminable, which unit.md
    # routes to branch 4 rather than to a guessed branch.
    determinable = not (bool(tie_or_nonunique) or bool(missing_fields))

    # Branch 2 -- release_lost.
    if (
        determinable
        and bool(release_observable)
        and release_margin is not None
        and float(release_margin) < 0.0
        and support == SUPPORT_SUPPORTED
    ):
        return PrimaryBranchResult(
            branch="release_lost",
            reasons=(
                "release_lost: the natural target description loses at its first observable "
                "divergence with a strictly negative margin while forced D_C retains "
                "U-calibrated target-local coordinate support",
            ),
        )

    # Branch 3 -- realization_fail.
    if (
        determinable
        and support == SUPPORT_UNSUPPORTED
        and status in GREEDY_TARGET_MISSED_STATUSES
    ):
        return PrimaryBranchResult(
            branch="realization_fail",
            reasons=(
                "realization_fail: forced D_C lacks U-calibrated target-local support and the "
                f"greedy box is {status!r} without a unique other-owner displacement",
            ),
        )

    # Branch 4 -- ambiguous.
    reasons = []
    if bool(missing_fields):
        reasons.append("ambiguous: a required primary score field is missing")
    if bool(tie_or_nonunique):
        reasons.append("ambiguous: an exact tie or nonunique owner match is present")
    if not bool(release_observable):
        reasons.append("ambiguous: description release is not observable at this boundary")
    if not reasons:
        reasons.append(
            "ambiguous: no branch predicate is satisfied without optional sampling"
        )
    return PrimaryBranchResult(branch="ambiguous", reasons=tuple(reasons))


# ---------------------------------------------------------------------------
# 7. Cache/context isolation, parity, native replay, quarantine
# ---------------------------------------------------------------------------


def assert_fresh_context_per_owner(context_group_ids: Sequence[str]) -> None:
    """Every logical context group gets its own fresh cache; none is reused."""

    ids = [str(value) for value in context_group_ids]
    if not ids:
        _fail("no logical context group was executed; there is nothing to attest")
    seen: set[str] = set()
    for group_id in ids:
        if group_id in seen:
            _fail(
                f"logical context group {group_id!r} was reused; a cache carried over "
                "between owners would contaminate the next reading"
            )
        seen.add(group_id)


@dataclass(frozen=True)
class SurfaceParityStreams:
    """Every token one surface scored, in a cached/uncached comparable order.

    unit.md bounds the *maximum* selected-logit difference over the compared
    scoring, not one summary scalar, and requires *every* compared argmax to be
    preserved.  Both therefore have to travel as full aligned sequences: a
    single release margin and the first greedy coordinate token would miss
    drift in any later token of either ladder.
    """

    request_ids: tuple[str, ...]
    selected_logprobs: tuple[float, ...]
    argmax_token_ids: tuple[int, ...]


@dataclass(frozen=True)
class SurfaceParityResult:
    """One surface's numeric and argmax agreement, reported per surface."""

    surface: str
    max_selected_logit_abs_diff: float
    argmax_parity: bool
    compared_token_count: int
    aligned: bool


def _surface_of_family(request_family: str) -> str | None:
    if request_family in RELEASE_FAMILIES:
        return SURFACE_RELEASE
    if request_family in COORDINATE_FAMILIES:
        return SURFACE_COORDINATE
    return None


def parity_streams_by_surface(
    score_rows: Sequence[Mapping[str, Any]],
) -> dict[str, SurfaceParityStreams]:
    """Concatenate one execution's scored tokens per ladder, request-id ordered.

    Ordering by request id (not by execution order) is what makes the cached
    and uncached streams comparable position by position; the request id is
    itself derived from the sealed request identity, so equal ids mean equal
    literal token spans.
    """

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in score_rows:
        surface = _surface_of_family(str(row.get("request_family")))
        if surface is None:
            continue
        grouped.setdefault(surface, []).append(row)
    streams: dict[str, SurfaceParityStreams] = {}
    for surface, rows in grouped.items():
        ordered = sorted(rows, key=lambda row: str(row["request_id"]))
        streams[surface] = SurfaceParityStreams(
            request_ids=tuple(str(row["request_id"]) for row in ordered),
            selected_logprobs=tuple(
                float(value) for row in ordered for value in row["selected_logprobs"]
            ),
            argmax_token_ids=tuple(
                int(value) for row in ordered for value in row["argmax_token_ids"]
            ),
        )
    return streams


@dataclass(frozen=True)
class ParityCheckInputs:
    """One cached reading paired with its independent uncached ground truth.

    The ``*_streams`` mappings carry the full per-surface token evidence.  The
    remaining scalars are the derived decision fields unit.md names explicitly
    (rank, support, owner match, branch) plus both declared margin signs.
    """

    cached_selected_logit: float
    uncached_selected_logit: float
    cached_argmax_token_id: int
    uncached_argmax_token_id: int
    cached_margin_sign: int
    uncached_margin_sign: int
    cached_owner_rank: int
    uncached_owner_rank: int
    cached_support_disposition: str
    uncached_support_disposition: str
    cached_owner_match: str | None
    uncached_owner_match: str | None
    cached_primary_branch: str
    uncached_primary_branch: str
    cached_streams: Mapping[str, SurfaceParityStreams] = field(default_factory=dict)
    uncached_streams: Mapping[str, SurfaceParityStreams] = field(default_factory=dict)
    cached_release_margin_sign: int = 0
    uncached_release_margin_sign: int = 0


@dataclass(frozen=True)
class ParityCheckResult:
    """Whether cached execution is admissible evidence for this surface."""

    status: str
    max_selected_logit_abs_diff: float
    mismatched_fields: tuple[str, ...]
    per_surface: tuple[SurfaceParityResult, ...] = ()


#: unit.md: cached execution preserves *every* one of these, exactly.  The two
#: margin signs are compared separately: a release sign can flip while the
#: coordinate sign, rank, support, match and branch all stay put.
PARITY_COMPARED_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("argmax", "cached_argmax_token_id", "uncached_argmax_token_id"),
    ("margin_sign", "cached_margin_sign", "uncached_margin_sign"),
    ("release_margin_sign", "cached_release_margin_sign", "uncached_release_margin_sign"),
    ("owner_rank", "cached_owner_rank", "uncached_owner_rank"),
    ("support_disposition", "cached_support_disposition", "uncached_support_disposition"),
    ("owner_match", "cached_owner_match", "uncached_owner_match"),
    ("primary_branch", "cached_primary_branch", "uncached_primary_branch"),
)


def _compare_surface_streams(
    cached: Mapping[str, SurfaceParityStreams],
    uncached: Mapping[str, SurfaceParityStreams],
) -> tuple[list[SurfaceParityResult], list[str]]:
    """Per-surface max selected-logit drift and exact full argmax agreement."""

    results: list[SurfaceParityResult] = []
    mismatched: list[str] = []
    # An absent stream is not agreement.  If a call site ever stops passing the
    # score rows, the comparison must fail closed rather than admit a cache on
    # the strength of a few summary scalars.
    for surface in SURFACES:
        if surface not in cached or surface not in uncached:
            mismatched.append(f"stream_missing:{surface}")
    for surface in sorted(set(cached) | set(uncached)):
        left = cached.get(surface)
        right = uncached.get(surface)
        aligned = (
            left is not None
            and right is not None
            and left.request_ids == right.request_ids
            and len(left.selected_logprobs) == len(right.selected_logprobs)
            and len(left.argmax_token_ids) == len(right.argmax_token_ids)
        )
        if not aligned:
            # Unaligned streams are not evidence of agreement; they are evidence
            # that the two executions did not score the same thing.
            mismatched.append(f"stream_alignment:{surface}")
            results.append(
                SurfaceParityResult(
                    surface=surface,
                    max_selected_logit_abs_diff=math.inf,
                    argmax_parity=False,
                    compared_token_count=0,
                    aligned=False,
                )
            )
            continue
        diffs = [
            abs(float(a) - float(b))
            for a, b in zip(left.selected_logprobs, right.selected_logprobs, strict=True)
        ]
        if not diffs:
            # Aligned but empty: nothing was compared, so nothing is proven.
            mismatched.append(f"stream_empty:{surface}")
        surface_max = max(diffs) if diffs else 0.0
        argmax_parity = left.argmax_token_ids == right.argmax_token_ids
        if not math.isfinite(surface_max) or surface_max > (
            CACHE_PARITY_MAX_SELECTED_LOGIT_ABS_DIFF
        ):
            mismatched.append(f"selected_logit:{surface}")
        if not argmax_parity:
            mismatched.append(f"argmax:{surface}")
        results.append(
            SurfaceParityResult(
                surface=surface,
                max_selected_logit_abs_diff=surface_max,
                argmax_parity=argmax_parity,
                compared_token_count=len(diffs),
                aligned=True,
            )
        )
    return results, mismatched


def evaluate_cache_parity(inputs: ParityCheckInputs) -> ParityCheckResult:
    """Numeric tolerance *and* exact discrete agreement, or fall back uncached.

    The reported maximum is global: the largest selected-logit difference over
    every compared token of every surface, never the drift of one summary
    scalar.
    """

    mismatched: list[str] = []
    per_surface, surface_mismatches = _compare_surface_streams(
        inputs.cached_streams, inputs.uncached_streams
    )
    scalar_diff = abs(
        float(inputs.cached_selected_logit) - float(inputs.uncached_selected_logit)
    )
    diff = max(
        [scalar_diff, *(result.max_selected_logit_abs_diff for result in per_surface)]
    )
    if not math.isfinite(diff) or diff > CACHE_PARITY_MAX_SELECTED_LOGIT_ABS_DIFF:
        mismatched.append("selected_logit")
    if any(not result.argmax_parity for result in per_surface):
        mismatched.append("argmax")
    for name, cached_attr, uncached_attr in PARITY_COMPARED_FIELDS:
        if getattr(inputs, cached_attr) != getattr(inputs, uncached_attr):
            mismatched.append(name)
    mismatched.extend(
        name for name in surface_mismatches if name not in mismatched
    )
    return ParityCheckResult(
        status=CACHE_ADMITTED if not mismatched else UNCACHED_FALLBACK,
        max_selected_logit_abs_diff=diff,
        mismatched_fields=tuple(dict.fromkeys(mismatched)),
        per_surface=tuple(per_surface),
    )


def replay_argmax_through_prefix(
    *,
    expected_token_ids: Sequence[int],
    argmax_token_ids: Sequence[int],
    up_to_index: int,
) -> bool:
    """Deterministic argmax must reproduce every token *before* the divergence."""

    expected = [int(v) for v in expected_token_ids]
    observed = [int(v) for v in argmax_token_ids]
    limit = int(up_to_index)
    if limit < 0:
        _fail("replay depth cannot be negative")
    if len(expected) < limit or len(observed) < limit:
        _fail(
            f"replay depth {limit} exceeds the scored token count "
            f"(expected={len(expected)}, argmax={len(observed)})"
        )
    return expected[:limit] == observed[:limit]


@dataclass(frozen=True)
class QuarantineEntry:
    owner_id: str
    reason: str
    detail: str


@dataclass(frozen=True)
class QuarantineLedger:
    """Append-only: an entry is never edited away once a case has failed."""

    entries: tuple[QuarantineEntry, ...]

    @property
    def count(self) -> int:
        return len(self.entries)


def apply_native_replay_quarantine(
    *,
    argmax_replay_matches: bool,
    owner_id: str,
    reason: str,
    detail: str,
    ledger: QuarantineLedger,
) -> QuarantineLedger:
    """Quarantine on replay mismatch; otherwise return the ledger unchanged."""

    if argmax_replay_matches:
        return ledger
    return QuarantineLedger(
        entries=(
            *ledger.entries,
            QuarantineEntry(owner_id=str(owner_id), reason=str(reason), detail=str(detail)),
        )
    )


def check_quarantine_stop(ledger: QuarantineLedger) -> bool:
    """More than two quarantined primary owners stops the unit."""

    return ledger.count > MAX_PRIMARY_QUARANTINES


# ---------------------------------------------------------------------------
# 8. Exact request / output identity
# ---------------------------------------------------------------------------


def build_request_identity(
    *,
    context_id: str,
    prefix_token_ids: Sequence[int],
    appended_token_ids: Sequence[int],
) -> dict[str, Any]:
    """The literal, deterministic identity of one scored request."""

    payload = {
        "context_id": str(context_id),
        "prefix_token_ids": [int(v) for v in prefix_token_ids],
        "appended_token_ids": [int(v) for v in appended_token_ids],
    }
    return {**payload, "request_identity_sha256": sha256_json(payload)}


def build_output_identity(
    *,
    request_identity_sha256: str,
    selected_logits: Sequence[float],
    token_ids: Sequence[int],
) -> dict[str, Any]:
    """The literal, deterministic identity of one request's readout."""

    logits = [float(v) for v in selected_logits]
    tokens = [int(v) for v in token_ids]
    if len(logits) != len(tokens):
        _fail(
            f"an output identity needs one selected logit per token "
            f"(logits={len(logits)}, tokens={len(tokens)})"
        )
    payload = {
        "request_identity_sha256": str(request_identity_sha256),
        "selected_logits": logits,
        "token_ids": tokens,
    }
    return {**payload, "output_identity_sha256": sha256_json(payload)}


# ---------------------------------------------------------------------------
# 9. Smoke matrix, cohort denominators, timing-control count
# ---------------------------------------------------------------------------


def validate_smoke_matrix_roles(roles: Sequence[str]) -> dict[str, Any]:
    """The three required smoke roles must all be present; unknowns fail closed."""

    observed = [str(role) for role in roles]
    unknown = sorted(set(observed) - SMOKE_MATRIX_ROLES)
    if unknown:
        _fail(f"unknown smoke-matrix role(s) {unknown!r}")
    missing = sorted(REQUIRED_SMOKE_MATRIX_ROLES - set(observed))
    if missing:
        _fail(
            f"the cached-execution smoke matrix is missing required role(s) {missing!r}; "
            "cached evidence is not admissible without all three"
        )
    return {
        "roles": sorted(set(observed)),
        "required": sorted(REQUIRED_SMOKE_MATRIX_ROLES),
        "optional_present": sorted(set(observed) & OPTIONAL_SMOKE_MATRIX_ROLES),
    }


def validate_primary_cohort_counts(
    *,
    u_count: int,
    l_count: int,
    same_context_ul_count: int,
    matched_e_count: int,
    unmatched_e_count: int,
) -> dict[str, int]:
    """Every frozen denominator, re-proven before a single score is read."""

    for label, actual, expected in (
        ("U-bound crossing cohort", u_count, PRIMARY_OWNER_COUNT_U),
        ("L-bound sensitivity cohort", l_count, PRIMARY_OWNER_COUNT_L),
        (
            "same-context U&L sensitivity cohort",
            same_context_ul_count,
            PRIMARY_OWNER_COUNT_SAME_CONTEXT_UL,
        ),
        ("matched-E stratum", matched_e_count, MATCHED_E_OWNER_COUNT),
        ("unmatched-E stratum", unmatched_e_count, UNMATCHED_E_OWNER_COUNT),
    ):
        if int(actual) != int(expected):
            _fail(
                f"{label} is {int(actual)}, not the frozen {int(expected)}; the sealed plan "
                "no longer reproduces this unit's denominator"
            )
    if int(matched_e_count) + int(unmatched_e_count) != PRIMARY_OWNER_COUNT_U:
        _fail("the two preregistered native-row strata do not partition the U cohort")
    return {
        "u_count": int(u_count),
        "l_count": int(l_count),
        "same_context_ul_count": int(same_context_ul_count),
        "matched_e_count": int(matched_e_count),
        "unmatched_e_count": int(unmatched_e_count),
    }


def validate_timing_control_count(count: int) -> int:
    """The disjoint timing-control registry has 14 owners -- not the stale 21."""

    actual = int(count)
    if actual == STALE_TIMING_CONTROL_OWNER_COUNT:
        _fail(
            f"the timing-control registry declares {actual} owners; that is the retired "
            "scout figure, never this unit's count"
        )
    if actual != TIMING_CONTROL_OWNER_COUNT:
        _fail(
            f"the timing-control registry declares {actual} owners, not the sealed "
            f"{TIMING_CONTROL_OWNER_COUNT}; a mismatch is reported, never patched by "
            "allowing primary/control overlap"
        )
    return actual


# ---------------------------------------------------------------------------
# 10. Interpretability gate and two-thirds routing (analyzer-facing helpers)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OwnerRecord:
    """One owner's deterministic disposition, as the analyzer reads it back."""

    owner_id: str
    stratum: str
    interpretable: bool
    branch: str | None


def evaluate_interpretability_gate(records: Sequence[OwnerRecord]) -> dict[str, Any]:
    """unit.md stop rule: >= 20/26 interpretable, >= 6 matched-E, >= 7 unmatched-E."""

    interpretable = [record for record in records if record.interpretable]
    matched = sum(1 for record in interpretable if record.stratum == "matched_e")
    unmatched = sum(1 for record in interpretable if record.stratum == "unmatched_e")
    return {
        "interpretable_count": len(interpretable),
        "matched_e_interpretable_count": matched,
        "unmatched_e_interpretable_count": unmatched,
        "minimum_interpretable": MIN_INTERPRETABLE_OWNERS,
        "minimum_matched_e": MIN_INTERPRETABLE_MATCHED_E,
        "minimum_unmatched_e": MIN_INTERPRETABLE_UNMATCHED_E,
        "passed": bool(
            len(interpretable) >= MIN_INTERPRETABLE_OWNERS
            and matched >= MIN_INTERPRETABLE_MATCHED_E
            and unmatched >= MIN_INTERPRETABLE_UNMATCHED_E
        ),
    }


def _branch_share(records: Sequence[OwnerRecord]) -> tuple[dict[str, int], str | None, int]:
    counts: dict[str, int] = {}
    for record in records:
        if record.branch is None:
            continue
        counts[record.branch] = counts.get(record.branch, 0) + 1
    if not counts:
        return counts, None, 0
    leading = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
    return counts, leading[0], leading[1]


def _reaches_two_thirds(count: int, total: int) -> bool:
    # Integer arithmetic: an exactly-two-thirds share must clear the threshold,
    # and a float ratio would decide that case on rounding.
    return total > 0 and count * ROUTING_MAJORITY_DENOMINATOR >= (
        total * ROUTING_MAJORITY_NUMERATOR
    )


def evaluate_branch_routing(
    records: Sequence[OwnerRecord],
    decoding_contradicted_owner_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Route one successor only at a two-thirds majority that survives exclusion.

    If dropping the decoding-contradicted likelihood cells would change whether
    a branch reaches two thirds, the result is split and no successor is routed.
    """

    gate = evaluate_interpretability_gate(records)
    if not gate["passed"]:
        _fail(
            "branch routing was requested below the interpretability gate "
            f"({gate['interpretable_count']} interpretable, "
            f"{gate['matched_e_interpretable_count']} matched-E, "
            f"{gate['unmatched_e_interpretable_count']} unmatched-E); no successor is routed"
        )
    interpretable = [record for record in records if record.interpretable]
    counts, leading_branch, leading_count = _branch_share(interpretable)
    routed = leading_branch is not None and _reaches_two_thirds(
        leading_count, len(interpretable)
    )

    excluded_ids = {str(value) for value in (decoding_contradicted_owner_ids or ())}
    sensitivity: dict[str, Any] | None = None
    if excluded_ids:
        retained = [record for record in interpretable if record.owner_id not in excluded_ids]
        retained_counts, retained_branch, retained_count = _branch_share(retained)
        retained_routed = retained_branch is not None and _reaches_two_thirds(
            retained_count, len(retained)
        )
        sensitivity = {
            "excluded_owner_ids": sorted(excluded_ids),
            "retained_interpretable_count": len(retained),
            "retained_branch_counts": dict(sorted(retained_counts.items())),
            "retained_leading_branch": retained_branch,
            "retained_reaches_two_thirds": retained_routed,
        }
        if routed and not (retained_routed and retained_branch == leading_branch):
            routed = False

    return {
        "status": "routed" if routed else "split",
        "routed_branch": leading_branch if routed else None,
        "interpretable_count": len(interpretable),
        "branch_counts": dict(sorted(counts.items())),
        "leading_branch": leading_branch,
        "leading_branch_count": leading_count,
        "majority_rule": (
            f"{ROUTING_MAJORITY_NUMERATOR}/{ROUTING_MAJORITY_DENOMINATOR} of the "
            "deterministic interpretable owners"
        ),
        "interpretability_gate": gate,
        "decoding_contradicted_sensitivity": sensitivity,
    }


# ---------------------------------------------------------------------------
# 11. Sealed plan loading
# ---------------------------------------------------------------------------


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not Path(path).is_file():
        _fail(f"{label} is missing at {path}")
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"{label} is unreadable at {path}: {exc}")
    if not isinstance(value, Mapping):
        _fail(f"{label} at {path} is not a JSON object")
    return dict(value)


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    if not Path(path).is_file():
        _fail(f"{label} is missing at {path}")
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            _fail(f"{label} line {number} is not valid JSON: {exc}")
        if not isinstance(value, Mapping):
            _fail(f"{label} line {number} is not a JSON object")
        rows.append(dict(value))
    return rows


def declared_sha256(value: Any, *, label: str) -> str:
    """Read one declared digest from either a bare hex string or a file record.

    The plan manifest is migrating its lineage entries from bare ``sha256``
    scalars to explicit ``{path, byte_size, sha256}`` records.  Both shapes mean
    the same thing, so this reads either rather than pinning one; anything else
    fails closed instead of silently comparing ``None`` to ``None``.
    """

    if isinstance(value, str):
        return value
    if isinstance(value, Mapping) and isinstance(value.get("sha256"), str):
        return str(value["sha256"])
    _fail(f"{label} is neither a sha256 string nor a file record carrying one")


def declared_digest_map(value: Any, *, label: str) -> dict[str, str]:
    """Normalise a declared digest collection to ``{relative_path: sha256}``.

    Accepts the historical ``{name: "hex"}`` map, the patched
    ``{name: {path, byte_size, sha256}}`` map, and a list of such records keyed
    by their own ``path``/``relative_path``/``name`` field.
    """

    if isinstance(value, Mapping):
        return {
            str(key): declared_sha256(entry, label=f"{label}[{key!r}]")
            for key, entry in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        normalized: dict[str, str] = {}
        for index, entry in enumerate(value):
            if not isinstance(entry, Mapping):
                _fail(f"{label}[{index}] is not a file record")
            key = entry.get("path") or entry.get("relative_path") or entry.get("name")
            if not isinstance(key, str):
                _fail(f"{label}[{index}] names no path")
            if key in normalized:
                _fail(f"{label} declares {key!r} more than once")
            normalized[key] = declared_sha256(entry, label=f"{label}[{key!r}]")
        return normalized
    _fail(f"{label} is neither a digest map nor a list of file records")


#: Manifest keys that have carried the same fact under two schema versions.
#: ``.v2`` replaced the flat ``*_input_file_sha256`` maps and the scalar
#: ``builder_source_sha256`` with explicit ``{path, byte_size, sha256}``
#: records; both spellings are read so a shard is not pinned to one revision.
LINEAGE_DIGEST_KEYS: Mapping[str, tuple[str, ...]] = {
    "prevalence": ("prevalence_input_files", "prevalence_input_file_sha256"),
    "census": ("census_input_files", "census_input_file_sha256"),
}
BUILDER_SOURCE_KEYS: tuple[str, ...] = ("builder_source", "builder_source_sha256")


def _first_present(mapping: Mapping[str, Any], keys: Sequence[str], *, label: str) -> Any:
    for key in keys:
        if mapping.get(key) is not None:
            return mapping[key]
    _fail(f"{label}: none of {list(keys)!r} is present")


def lineage_digest_map(lineage: Mapping[str, Any], *, surface: str) -> dict[str, str]:
    keys = LINEAGE_DIGEST_KEYS[surface]
    return declared_digest_map(
        _first_present(lineage, keys, label=f"manifest lineage for {surface}"),
        label=f"lineage {surface} input digests",
    )


def builder_source_sha256(manifest: Mapping[str, Any]) -> str:
    return declared_sha256(
        _first_present(manifest, BUILDER_SOURCE_KEYS, label="manifest builder source"),
        label="manifest builder source",
    )


def declared_path(value: Any, *, label: str) -> str:
    """Read a declared run root from either a bare path string or a record."""

    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        for key in ("path", "run_root", "root"):
            candidate = value.get(key)
            if isinstance(candidate, str):
                return candidate
    _fail(f"{label} is neither a path string nor a record carrying one")


@dataclass(frozen=True)
class SealedPlan:
    """The sealed CPU plan plus the sealed census surfaces it points at."""

    plan_dir: Path
    manifest: dict[str, Any]
    cohort_rows: list[dict[str, Any]]
    control_rows: list[dict[str, Any]]
    request_rows: list[dict[str, Any]]
    plan_file_sha256: dict[str, str]
    inputs: Any
    candidates_by_id: dict[str, dict[str, Any]]
    owners_by_image: dict[str, list[dict[str, Any]]]

    @property
    def contexts_by_id(self) -> Mapping[str, Mapping[str, Any]]:
        return self.inputs.census.contexts_by_id

    def context_prefix_token_ids(self, context_id: str) -> list[int]:
        """The *generated* self-prefix the plan's ``base_prefix`` digest covers.

        This is the plan-bound identity of a boundary, not the sequence a model
        forwards: the executed prefix additionally carries the image's
        processor-expanded prompt (see :meth:`executed_prefix_token_ids`).
        """

        context = self.contexts_by_id.get(str(context_id))
        if context is None:
            _fail(f"context {context_id!r} is absent from the sealed context registry")
        return _token_ids(
            context.get("generated_prefix_token_ids"), label=f"context {context_id!r}"
        )

    def image_of_context(self, context_id: str) -> str:
        context = self.contexts_by_id.get(str(context_id))
        if context is None:
            _fail(f"context {context_id!r} is absent from the sealed context registry")
        return str(context["image_id"])

    def assert_context_belongs_to_session_image(
        self, context_id: str, *, session_image_id: str, label: str
    ) -> None:
        """A session holds exactly one image's vision state; nothing else may use it.

        One model session is opened per image and its ``pixel_values`` /
        ``image_grid_thw`` are folded into the prefill cache.  Forwarding another
        image's literal prefix through it would pair that prefix with the wrong
        picture and produce numbers that look ordinary while being meaningless.

        The deciding check is the context's sealed ``image_id``.  The literal
        prompt-token check that follows is a redundant cross-check on the same
        registries -- :meth:`executed_prefix_token_ids` builds the prompt from
        the *context's own* image, so it cannot disagree once the ids agree --
        kept because it also fails closed when either registry lookup is
        missing or malformed.
        """

        observed_image = self.image_of_context(context_id)
        if observed_image != str(session_image_id):
            _fail(
                f"{label}: context {context_id!r} belongs to image {observed_image!r} but the "
                f"open session holds image {session_image_id!r}; refusing to score a prefix "
                "against another image's visual state"
            )
        session_image = self.inputs.images_by_id.get(str(session_image_id))
        if session_image is None:
            _fail(f"{label}: image {session_image_id!r} is absent from the image registry")
        prompt = _token_ids(
            session_image.get("prompt_token_ids"),
            label=f"image {session_image_id!r} prompt",
        )
        executed = self.executed_prefix_token_ids(context_id)
        if executed[: len(prompt)] != prompt:
            _fail(
                f"{label}: the executed prefix of context {context_id!r} does not begin with "
                f"image {session_image_id!r}'s sealed prompt tokens"
            )

    def executed_prefix_token_ids(self, context_id: str) -> list[int]:
        """``prompt_token_ids + generated_prefix_token_ids``: what is forwarded.

        Mirrors the census scorer's own observed-prefix construction.  Scoring
        the generated prefix alone would silently drop the multimodal prompt and
        forward a sequence that is not the native one.
        """

        context = self.contexts_by_id.get(str(context_id))
        if context is None:
            _fail(f"context {context_id!r} is absent from the sealed context registry")
        image = self.inputs.images_by_id.get(str(context["image_id"]))
        if image is None:
            _fail(f"context {context_id!r} names an image absent from the image registry")
        prompt = _token_ids(
            image.get("prompt_token_ids"), label=f"image {context['image_id']!r} prompt"
        )
        return prompt + self.context_prefix_token_ids(context_id)


def load_sealed_plan(
    plan_dir: Path,
    *,
    prevalence_run_root: Path | None = None,
    census_run_root: Path | None = None,
) -> SealedPlan:
    """Re-prove every digest of the sealed plan and both predecessor runs."""

    plan_dir = Path(plan_dir)
    manifest = _read_json(plan_dir / plan_builder.MANIFEST_NAME, "plan/manifest.json")
    if manifest.get("schema_version") != plan_builder.MANIFEST_SCHEMA_VERSION:
        _fail(
            f"plan manifest schema {manifest.get('schema_version')!r} is not "
            f"{plan_builder.MANIFEST_SCHEMA_VERSION!r}"
        )
    if str(manifest.get("unit_id")) != UNIT_ID:
        _fail(f"plan manifest belongs to unit {manifest.get('unit_id')!r}, not {UNIT_ID!r}")

    declared_seal = manifest.get("manifest_content_sha256")
    unsealed = {key: value for key, value in manifest.items() if key != "manifest_content_sha256"}
    if sha256_json(unsealed) != declared_seal:
        _fail("plan manifest does not self-seal; it has been edited after the build")

    digests = manifest.get("output_file_digests")
    if not isinstance(digests, Mapping):
        _fail("plan manifest carries no output_file_digests")
    plan_file_sha256: dict[str, str] = {}
    declared_outputs = declared_digest_map(digests, label="manifest output_file_digests")
    for name in (
        plan_builder.COHORT_REGISTRY_NAME,
        plan_builder.CONTROL_REGISTRY_NAME,
        plan_builder.REQUEST_PLAN_NAME,
    ):
        if name not in declared_outputs:
            _fail(f"plan manifest declares no digest for {name}")
        observed = sha256_file(plan_dir / name)
        if observed != declared_outputs[name]:
            _fail(
                f"sealed plan file {name} hashes to {observed}, not the manifest's "
                f"{declared_outputs[name]}"
            )
        plan_file_sha256[name] = observed

    cohort_rows = _read_jsonl(plan_dir / plan_builder.COHORT_REGISTRY_NAME, "cohort registry")
    control_rows = _read_jsonl(plan_dir / plan_builder.CONTROL_REGISTRY_NAME, "control registry")
    request_rows = _read_jsonl(plan_dir / plan_builder.REQUEST_PLAN_NAME, "request plan")
    for label, rows, expected_schema in (
        ("cohort registry", cohort_rows, plan_builder.COHORT_SCHEMA_VERSION),
        ("control registry", control_rows, plan_builder.CONTROL_SCHEMA_VERSION),
        ("request plan", request_rows, plan_builder.REQUEST_SCHEMA_VERSION),
    ):
        if not rows:
            _fail(f"{label} is empty")
        for row in rows:
            if row.get("schema_version") != expected_schema:
                _fail(f"{label} carries a row with schema {row.get('schema_version')!r}")
            if str(row.get("unit_id")) != UNIT_ID:
                _fail(f"{label} carries a row from another unit")
        assert_no_retokenizable_text(rows, label=label)

    counts = manifest.get("cohort_counts") or {}
    validate_primary_cohort_counts(
        u_count=int(counts.get("u_bound_crossing_count", -1)),
        l_count=int(counts.get("l_bound_crossing_count", -1)),
        same_context_ul_count=int(counts.get("exact_same_context_u_and_l_count", -1)),
        matched_e_count=int(counts.get("matched_e_count", -1)),
        unmatched_e_count=int(counts.get("unmatched_e_count", -1)),
    )
    control_counts = manifest.get("control_counts") or {}
    validate_timing_control_count(int(control_counts.get("timing_control_count", -1)))
    if int(control_counts.get("tp_replay_control_count", -1)) != TP_CALIBRATION_OWNER_COUNT:
        _fail(
            f"the TP replay-control registry declares "
            f"{control_counts.get('tp_replay_control_count')!r} owners, not the sealed "
            f"{TP_CALIBRATION_OWNER_COUNT}"
        )
    if len(cohort_rows) != PRIMARY_OWNER_COUNT_U:
        _fail(
            f"the cohort registry holds {len(cohort_rows)} rows, not the frozen "
            f"{PRIMARY_OWNER_COUNT_U}"
        )

    lineage = manifest.get("lineage") or {}
    resolved_prevalence = Path(
        prevalence_run_root
        if prevalence_run_root is not None
        else declared_path(
            lineage.get("prevalence_run_root"), label="lineage prevalence_run_root"
        )
    )
    resolved_census = Path(
        census_run_root
        if census_run_root is not None
        else declared_path(lineage.get("census_run_root"), label="lineage census_run_root")
    )
    inputs = plan_builder.load_plan_inputs(resolved_prevalence, resolved_census)
    for label, observed in (
        ("prevalence", inputs.prevalence_file_sha256),
        ("census", inputs.census_file_sha256),
    ):
        declared = lineage_digest_map(lineage, surface=label)
        # Subset, not equality: the lineage may enumerate more predecessor files
        # than this capture re-reads, but every file it does re-read must hash
        # to exactly the sealed value.
        drifted = sorted(
            name
            for name, digest in observed.items()
            if name in declared and declared[name] != digest
        )
        missing = sorted(name for name in observed if name not in declared)
        if drifted or missing:
            _fail(
                f"the {label} run's file digests do not reconcile with the plan manifest "
                f"(drifted={drifted!r}, unsealed={missing!r}); the capture would score "
                "against a drifted predecessor"
            )

    candidates_by_id: dict[str, dict[str, Any]] = {}
    for rows in inputs.candidates_by_image_description.values():
        for row in rows:
            candidates_by_id[str(row["candidate_id"])] = row
    owners_by_image: dict[str, list[dict[str, Any]]] = {}
    for owner in inputs.owners_by_id.values():
        owners_by_image.setdefault(str(owner["image_id"]), []).append(owner)
    for rows in owners_by_image.values():
        rows.sort(key=lambda row: str(row["gt_owner_id"]))

    return SealedPlan(
        plan_dir=plan_dir,
        manifest=manifest,
        cohort_rows=cohort_rows,
        control_rows=control_rows,
        request_rows=request_rows,
        plan_file_sha256=plan_file_sha256,
        inputs=inputs,
        candidates_by_id=candidates_by_id,
        owners_by_image=owners_by_image,
    )


def validate_request_row(plan: SealedPlan, row: Mapping[str, Any]) -> dict[str, Any]:
    """Re-derive a request's identity and bind it to the literal sealed context."""

    request_id = str(row.get("request_id"))
    prefix = row.get("prefix")
    if not isinstance(prefix, Mapping):
        _fail(f"request {request_id!r} carries no prefix binding")
    if bool(prefix.get("retokenized")):
        _fail(f"request {request_id!r} declares a retokenized prefix")

    appended = _token_ids(
        prefix.get("appended_token_ids"), label=f"request {request_id!r} appended tokens"
    )
    if sha256_json(appended) != prefix.get("appended_token_ids_sha256"):
        _fail(f"request {request_id!r} appended-token digest does not reconstruct")
    if len(appended) != int(prefix.get("appended_token_count", -1)):
        _fail(f"request {request_id!r} appended-token count does not match its tokens")

    context_id = str(row.get("context_id"))
    context_tokens = plan.context_prefix_token_ids(context_id)
    if sha256_json(context_tokens) != prefix.get("base_prefix_token_ids_sha256"):
        _fail(
            f"request {request_id!r} base prefix digest does not match the sealed context "
            f"{context_id!r}"
        )
    if len(context_tokens) != int(prefix.get("base_prefix_token_count", -1)):
        _fail(f"request {request_id!r} base prefix token count does not match the context")

    identity = {
        "unit_id": UNIT_ID,
        "request_family": row.get("request_family"),
        "cohort": row.get("cohort"),
        "gt_owner_id": row.get("gt_owner_id"),
        "context_id": context_id,
        "variant": row.get("variant"),
        "appended_token_ids": appended,
        "base_prefix_token_ids_sha256": prefix.get("base_prefix_token_ids_sha256"),
        "scored_target": dict(row.get("scored_target") or {}),
        "decode": None if row.get("decode") is None else dict(row["decode"]),
        "candidate_family": (
            None if row.get("candidate_family") is None else dict(row["candidate_family"])
        ),
    }
    identity_digest = sha256_json(identity)
    if identity_digest != row.get("identity_digest"):
        _fail(f"request {request_id!r} identity digest does not reconstruct from its own fields")
    if request_id != f"req:{identity_digest[:32]}":
        _fail(f"request {request_id!r} id is not derived from its identity digest")
    if str(row.get("branch_schema_id")) != plan_builder.BRANCH_SCHEMA_ID:
        _fail(f"request {request_id!r} binds an unknown branch schema")

    return {
        "context_token_ids": context_tokens,
        "executed_prefix_token_ids": plan.executed_prefix_token_ids(context_id),
        "appended_token_ids": appended,
        "identity_digest": identity_digest,
    }


# ---------------------------------------------------------------------------
# 12. Runtime capture
# ---------------------------------------------------------------------------


def _census_shard() -> Any:
    """Lazy import of the predecessor's production backend seam.

    Deferred so the pure contract layer above stays importable without torch,
    a model, or a GPU.
    """

    from scripts.research import score_sorted_owner_accessibility_census_shard as shard

    return shard


def _basin() -> Any:
    from scripts.research import score_sorted_owner_basin_landscape as basin

    return basin


@dataclass(frozen=True)
class ScoredToken:
    """One teacher-forced position: what was scored, and what argmax preferred."""

    token_id: int
    selected_logprob: float
    argmax_token_id: int
    argmax_logprob: float


def _teacher_forced(
    backend: Any,
    prefill: Any,
    *,
    root_token_ids: Sequence[int],
    token_ids: Sequence[int],
    uncached: bool,
) -> list[ScoredToken]:
    """Teacher-force a literal token span from one admitted root.

    ``uncached`` swaps the KV-cache branch for an independent ``use_cache=False``
    reforward of the growing literal sequence.  Both paths read raw fp32 logits
    through the same full-vocabulary ``readout``.
    """

    shard = _census_shard()
    tokens = [int(v) for v in token_ids]
    if not tokens:
        _fail("teacher forcing requires at least one token")
    scored: list[ScoredToken] = []
    if uncached:
        running = [int(v) for v in root_token_ids]
        for token in tokens:
            view = shard.readout(backend.full_reforward(running), selected_token_id=token, top_k=1)
            scored.append(
                ScoredToken(
                    token_id=token,
                    selected_logprob=view.selected_logprob,
                    argmax_token_id=view.argmax_token_id,
                    argmax_logprob=view.argmax_logprob,
                )
            )
            running.append(token)
        return scored
    with prefill.branch() as branch:
        logits = prefill.root_logits
        for index, token in enumerate(tokens):
            view = shard.readout(logits, selected_token_id=token, top_k=1)
            scored.append(
                ScoredToken(
                    token_id=token,
                    selected_logprob=view.selected_logprob,
                    argmax_token_id=view.argmax_token_id,
                    argmax_logprob=view.argmax_logprob,
                )
            )
            if index + 1 < len(tokens):
                logits = branch.step([token])[-1]
    return scored


def _greedy_coordinate_row(
    backend: Any, prefill: Any, *, root_token_ids: Sequence[int], uncached: bool
) -> tuple[CoordinateGrammarResult, list[ScoredToken]]:
    """Greedy-decode exactly the coordinate row, one explicit argmax per step."""

    shard = _census_shard()
    domain = range(COORDINATE_TOKEN_ID_START, COORDINATE_TOKEN_ID_END_EXCLUSIVE)
    scored: list[ScoredToken] = []

    if uncached:
        running = [int(v) for v in root_token_ids]

        def _argmax(step_index: int, tokens_so_far: tuple[int, ...]) -> int:
            del step_index, tokens_so_far
            view = shard.readout(backend.full_reforward(running), top_k=1)
            scored.append(
                ScoredToken(
                    token_id=view.selected_token_id,
                    selected_logprob=view.selected_logprob,
                    argmax_token_id=view.argmax_token_id,
                    argmax_logprob=view.argmax_logprob,
                )
            )
            running.append(view.selected_token_id)
            return view.selected_token_id

        return (
            greedy_decode_coordinate_row(
                _argmax, coordinate_domain=domain, box_end_token_id=BOX_END
            ),
            scored,
        )

    with prefill.branch() as branch:
        state: dict[str, Any] = {"logits": prefill.root_logits}

        def _argmax_cached(step_index: int, tokens_so_far: tuple[int, ...]) -> int:
            del step_index, tokens_so_far
            view = shard.readout(state["logits"], top_k=1)
            scored.append(
                ScoredToken(
                    token_id=view.selected_token_id,
                    selected_logprob=view.selected_logprob,
                    argmax_token_id=view.argmax_token_id,
                    argmax_logprob=view.argmax_logprob,
                )
            )
            state["logits"] = branch.step([view.selected_token_id])[-1]
            return view.selected_token_id

        result = greedy_decode_coordinate_row(
            _argmax_cached, coordinate_domain=domain, box_end_token_id=BOX_END
        )
    return result, scored


def _score_candidate_boxes_scalar(
    backend: Any,
    prefill: Any,
    *,
    root_token_ids: Sequence[int],
    coord_rows: Sequence[Sequence[int]],
    uncached: bool,
) -> list[list[ScoredToken]]:
    return [
        _teacher_forced(
            backend,
            prefill,
            root_token_ids=root_token_ids,
            token_ids=list(coords) + [BOX_END],
            uncached=uncached,
        )
        for coords in coord_rows
    ]


def _score_candidate_boxes_batched(
    prefill: Any, *, coord_rows: Sequence[Sequence[int]], batch_size: int
) -> list[list[ScoredToken]]:
    """Equal-shape lanes over one copied root; a throughput device only.

    Every lane shares this group's image, prompt, literal prefix and forced
    ``D_C``, and advances exactly one token per step, so lane shape is equal by
    construction.
    """

    shard = _census_shard()
    rows: list[list[ScoredToken]] = []
    width = max(1, int(batch_size))
    for start in range(0, len(coord_rows), width):
        chunk = [list(row) + [BOX_END] for row in coord_rows[start : start + width]]
        lengths = {len(row) for row in chunk}
        if len(lengths) != 1:
            _fail("batched candidate lanes must advance an equal number of tokens")
        per_lane: list[list[ScoredToken]] = [[] for _ in chunk]
        with prefill.batched_branch(len(chunk)) as lanes:
            logits_per_lane = [prefill.root_logits] * len(chunk)
            for position in range(len(chunk[0])):
                for lane_index, row in enumerate(chunk):
                    view = shard.readout(
                        logits_per_lane[lane_index], selected_token_id=row[position], top_k=1
                    )
                    per_lane[lane_index].append(
                        ScoredToken(
                            token_id=row[position],
                            selected_logprob=view.selected_logprob,
                            argmax_token_id=view.argmax_token_id,
                            argmax_logprob=view.argmax_logprob,
                        )
                    )
                if position + 1 < len(chunk[0]):
                    stepped = lanes.step([[row[position]] for row in chunk])
                    logits_per_lane = [stepped[index][-1] for index in range(len(chunk))]
        rows.extend(per_lane)
    return rows


def _complete_box_logprob_sum(scored: Sequence[ScoredToken]) -> float:
    return math.fsum(token.selected_logprob for token in scored[:COORDINATE_TOKEN_COUNT])


def _description_path(token_ids: Sequence[int]) -> list[int]:
    """The observable description path of a native action: through ``<|box_start|>``.

    A terminal STOP action has no description; its single ``<|im_end|>`` token
    is its whole observable action and therefore its own path.
    """

    tokens = [int(v) for v in token_ids]
    if BOX_START in tokens:
        return tokens[: tokens.index(BOX_START) + 1]
    return tokens


def _owner_match_greedy_box(
    plan: SealedPlan,
    *,
    image_id: str,
    normalized_description: str,
    target_owner_id: str,
    grammar: CoordinateGrammarResult,
) -> dict[str, Any]:
    """Match one freshly decoded box with the predecessor's one-box matcher.

    Category-local by construction: the box was produced under a forced
    ``D_C`` query, so its competition is same-description.  The threshold is the
    census's frozen ``IOU_THRESHOLD``; nothing here re-tunes it.
    """

    if grammar.status != "valid":
        return {
            "greedy_status": GREEDY_MALFORMED,
            "greedy_owner_match": None,
            "greedy_nonunique_match": False,
            "malformed_reason": grammar.malformed_reason,
            "decoded_bbox_pixel_xyxy": None,
            "assignment": None,
        }
    image = plan.inputs.images_by_id.get(str(image_id))
    if image is None:
        _fail(f"image {image_id!r} is absent from the sealed image registry")
    canvas = planner.Canvas(int(image["image_width"]), int(image["image_height"]))
    bins = [token - COORDINATE_TOKEN_ID_START for token in grammar.coord_token_ids]
    pixel_box = list(canvas.bins_to_pixel(bins))
    owners = [
        owner
        for owner in plan.owners_by_image.get(str(image_id), ())
        if str(owner["normalized_description"]) == str(normalized_description)
    ]
    assignment = greedy_census.independent_owner_assignment(
        pixel_box, owners, iou_threshold=OWNER_MATCH_IOU_THRESHOLD
    )
    status = str(assignment["strict_assignment_status"])
    owner_id = assignment.get("strict_assignment_gt_owner_id")
    if status == plan_builder.SIDECAR_MATCHED:
        greedy_status = (
            GREEDY_TARGET_MATCH if str(owner_id) == str(target_owner_id) else GREEDY_OTHER_OWNER_MATCH
        )
    elif status == plan_builder.SIDECAR_AMBIGUOUS_NEUTRAL:
        greedy_status = GREEDY_AMBIGUOUS
    else:
        greedy_status = GREEDY_UNMATCHED
    ambiguity_owner_ids = [str(value) for value in (assignment.get("ambiguity_owner_ids") or ())]
    return {
        "greedy_status": greedy_status,
        "greedy_owner_match": None if greedy_status == GREEDY_UNMATCHED else owner_id,
        # unit.md routes a nonunique owner match to branch 4; the matcher's own
        # ambiguity set is the signal, not just the coarse status.
        "greedy_nonunique_match": bool(
            greedy_status == GREEDY_AMBIGUOUS or len(ambiguity_owner_ids) > 1
        ),
        "malformed_reason": None,
        "decoded_bbox_pixel_xyxy": pixel_box,
        "assignment": {
            "matcher": "independent_owner_assignment_one_box_category_local",
            "iou_threshold": OWNER_MATCH_IOU_THRESHOLD,
            "population_normalized_description": str(normalized_description),
            "population_size": len(owners),
            "strict_assignment_status": status,
            "strict_assignment_gt_owner_id": owner_id,
            "ambiguity_owner_ids": list(assignment.get("ambiguity_owner_ids") or ()),
        },
    }


def _candidate_scores(
    plan: SealedPlan,
    *,
    candidate_ids: Sequence[str],
    context_id: str,
    box_sums: Sequence[float],
) -> tuple[list[CandidateScore], list[str]]:
    """Owner-identifiable candidate scores, using the bank's own sealed match.

    A candidate whose sealed category-local strict assignment is not ``matched``
    is not owner-identifiable, so it cannot enter an owner ranking; it is
    reported instead of being silently attributed to its generator.
    """

    scores: list[CandidateScore] = []
    unattributable: list[str] = []
    for candidate_id, box_sum in zip(candidate_ids, box_sums, strict=True):
        candidate = plan.candidates_by_id.get(str(candidate_id))
        if candidate is None:
            _fail(f"candidate {candidate_id!r} is absent from the sealed candidate bank")
        if str(candidate.get("strict_assignment_status")) != plan_builder.SIDECAR_MATCHED:
            unattributable.append(str(candidate_id))
            continue
        scores.append(
            CandidateScore(
                candidate_id=str(candidate_id),
                owner_id=str(candidate["strict_assignment_gt_owner_id"]),
                context_id=str(context_id),
                complete_box_logprob_sum=float(box_sum),
            )
        )
    return scores, unattributable


def _candidate_coord_tokens(plan: SealedPlan, candidate_id: str) -> list[int]:
    candidate = plan.candidates_by_id.get(str(candidate_id))
    if candidate is None:
        _fail(f"candidate {candidate_id!r} is absent from the sealed candidate bank")
    tokens = _token_ids(
        candidate.get("coord_token_ids"), label=f"candidate {candidate_id!r} coord tokens"
    )
    if len(tokens) != COORDINATE_TOKEN_COUNT:
        _fail(f"candidate {candidate_id!r} does not carry exactly four coordinate tokens")
    if sha256_json(tokens) != candidate.get("coord_token_ids_sha256"):
        _fail(f"candidate {candidate_id!r} coordinate digest does not reconstruct")
    outside = [
        token
        for token in tokens
        if not COORDINATE_TOKEN_ID_START <= token < COORDINATE_TOKEN_ID_END_EXCLUSIVE
    ]
    if outside:
        _fail(f"candidate {candidate_id!r} carries out-of-domain coordinate token(s) {outside!r}")
    return tokens


# ---------------------------------------------------------------------------
# 13. Per-owner capture
# ---------------------------------------------------------------------------


@dataclass
class CaptureState:
    """Mutable bookkeeping shared by every owner of one shard."""

    context_group_ids: list[str] = field(default_factory=list)
    score_rows: list[dict[str, Any]] = field(default_factory=list)
    owner_records: list[dict[str, Any]] = field(default_factory=list)
    quarantine: QuarantineLedger = field(
        default_factory=lambda: QuarantineLedger(entries=())
    )
    surface_backend: dict[str, str] = field(
        default_factory=lambda: {surface: KV_CACHE_BACKEND for surface in SURFACES}
    )
    unattributable_candidate_ids: list[str] = field(default_factory=list)
    batched_candidate_rows: int = 0
    scalar_candidate_rows: int = 0


def _context_group_id(
    *,
    gt_owner_id: str,
    context_id: str,
    variant: str,
    appended_digest: str,
    scoring_backend: str,
) -> str:
    """The identity of one logical context group: one prefill, one fresh cache.

    ``scoring_backend`` participates because an uncached fallback re-read of the
    same boundary is a *separate* execution on its own fresh state, not a reused
    cache; conflating the two would make the freshness attestation fire on the
    fallback path this unit explicitly requires.
    """

    return sha256_json(
        {
            "unit_id": UNIT_ID,
            "gt_owner_id": str(gt_owner_id),
            "context_id": str(context_id),
            "variant": str(variant),
            "appended_token_ids_sha256": str(appended_digest),
            "scoring_backend": str(scoring_backend),
        }
    )


def _score_row(
    *,
    shard_id: str,
    request: Mapping[str, Any],
    binding: Mapping[str, Any],
    context_group_id: str,
    scoring_backend: str,
    scored_tokens: Sequence[ScoredToken],
    observables: Mapping[str, Any],
) -> dict[str, Any]:
    # The identity binds the *executed* literal prefix (processor-expanded
    # prompt plus native self-prefix); the plan's generated-prefix digest is
    # carried alongside it so the two are never confused.
    identity = build_request_identity(
        context_id=str(request["context_id"]),
        prefix_token_ids=binding["executed_prefix_token_ids"],
        appended_token_ids=binding["appended_token_ids"],
    )
    output = build_output_identity(
        request_identity_sha256=identity["request_identity_sha256"],
        selected_logits=[token.selected_logprob for token in scored_tokens],
        token_ids=[token.token_id for token in scored_tokens],
    )
    prefix = request["prefix"]
    return {
        "schema_version": SCHEMA_VERSION,
        "row_kind": "crossing_boundary_score",
        "unit_id": UNIT_ID,
        "shard_id": str(shard_id),
        "request_id": str(request["request_id"]),
        "request_key": str(request["request_key"]),
        "request_family": str(request["request_family"]),
        "cohort": str(request["cohort"]),
        "variant": str(request["variant"]),
        "readout_tier": str(request["readout_tier"]),
        "gt_owner_id": str(request["gt_owner_id"]),
        "image_id": str(request["image_id"]),
        "context_id": str(request["context_id"]),
        "boundary_index": int(request["boundary_index"]),
        "boundary_label": BOUNDARY_LABEL_BY_VARIANT.get(str(request["variant"])),
        "context_group_id": str(context_group_id),
        "plan_identity_digest": str(binding["identity_digest"]),
        "request_identity_sha256": identity["request_identity_sha256"],
        "output_identity_sha256": output["output_identity_sha256"],
        "base_prefix_token_count": int(prefix["base_prefix_token_count"]),
        "base_prefix_token_ids_sha256": str(prefix["base_prefix_token_ids_sha256"]),
        "executed_prefix_token_count": len(binding["executed_prefix_token_ids"]),
        "executed_prefix_token_ids_sha256": sha256_json(binding["executed_prefix_token_ids"]),
        "appended_token_count": int(prefix["appended_token_count"]),
        "appended_token_ids_sha256": str(prefix["appended_token_ids_sha256"]),
        "scored_token_ids": [token.token_id for token in scored_tokens],
        "scored_token_ids_sha256": sha256_json([token.token_id for token in scored_tokens]),
        "selected_logprobs": [token.selected_logprob for token in scored_tokens],
        "argmax_token_ids": [token.argmax_token_id for token in scored_tokens],
        "argmax_logprobs": [token.argmax_logprob for token in scored_tokens],
        "scoring_backend": str(scoring_backend),
        "likelihood_channel": LIKELIHOOD_CHANNEL,
        "repetition_penalty_stratum": float(NATIVE_REPETITION_PENALTY_STRATUM),
        "uses_model_generate": False,
        "sampling": "disabled_primary_deterministic_pass_only",
        "retokenized": False,
        "observables": dict(observables),
    }


@dataclass(frozen=True)
class LadderReadout:
    """Everything one (owner, boundary) pair observed, before any classification."""

    boundary_label: str
    context_id: str
    context_group_id: str
    release: NaturalReleaseObservation | None
    native_action_kind: str
    native_replay_admitted: bool
    native_replay_depth: int
    owner_rank: OwnerRankResult | None
    greedy: dict[str, Any]
    candidate_count: int
    unattributable_candidate_ids: tuple[str, ...]
    release_scoring_backend: str
    coordinate_scoring_backend: str
    #: Calibrated support under both ambiguity bounds.  U owns the primary
    #: branch; L travels beside it as a sensitivity and never changes it.
    support_by_bound: Mapping[str, TargetLocalSupportResult] = field(
        default_factory=dict
    )

    @property
    def primary_support(self) -> TargetLocalSupportResult | None:
        return self.support_by_bound.get(SUPPORT_BOUND_U)

    @property
    def primary_support_disposition(self) -> str:
        support = self.primary_support
        return (
            SUPPORT_CALIBRATION_UNAVAILABLE
            if support is None
            else support.support_disposition
        )


def _capture_boundary(
    plan: SealedPlan,
    backend: Any,
    state: CaptureState,
    *,
    shard_id: str,
    gt_owner_id: str,
    image_id: str,
    session_image_id: str,
    normalized_description: str,
    variant: str,
    requests: Mapping[str, Mapping[str, Any]],
    bindings: Mapping[str, Mapping[str, Any]],
    batch_size: int,
    surfaces: frozenset[str] = frozenset(SURFACES),
) -> LadderReadout:
    """Both ladders at exactly one context, each on its own fresh context group.

    ``surfaces`` narrows the work to one ladder, which is how an uncached
    fallback re-reads *only* the affected surface instead of redundantly
    re-executing -- and re-publishing -- the surface that already passed.
    """

    boundary_label = BOUNDARY_LABEL_BY_VARIANT.get(str(variant))
    if boundary_label is None:
        _fail(f"unknown plan variant {variant!r}")
    release_uncached = state.surface_backend[SURFACE_RELEASE] == UNCACHED_BACKEND
    coordinate_uncached = state.surface_backend[SURFACE_COORDINATE] == UNCACHED_BACKEND

    context_id = str(next(iter(requests.values()))["context_id"])
    if str(image_id) != str(session_image_id):
        _fail(
            f"owner {gt_owner_id!r} belongs to image {image_id!r} but the open session holds "
            f"image {session_image_id!r}"
        )
    plan.assert_context_belongs_to_session_image(
        context_id,
        session_image_id=session_image_id,
        label=f"owner {gt_owner_id!r} at variant {variant!r}",
    )
    # Vision-bearing prompt plus the literal native self-prefix: exactly the
    # sequence the native rollout forwarded at this boundary.
    context_tokens = plan.executed_prefix_token_ids(context_id)

    # --- ladder 1: the bare context group -------------------------------
    release: NaturalReleaseObservation | None = None
    native_kind = "absent"
    replay_admitted = True
    replay_depth = 0
    bare_group_id = _context_group_id(
        gt_owner_id=gt_owner_id,
        context_id=context_id,
        variant=variant,
        appended_digest=sha256_json([]),
        scoring_backend=UNCACHED_BACKEND if release_uncached else KV_CACHE_BACKEND,
    )
    release_request = requests.get(REQUEST_NATURAL_RELEASE)
    native_request = requests.get(REQUEST_NATIVE_NEXT_ACTION)
    if (
        SURFACE_RELEASE in surfaces
        and release_request is not None
        and native_request is not None
    ):
        state.context_group_ids.append(bare_group_id)
        target_tokens = _token_ids(
            release_request["scored_target"]["token_ids"], label="target description path"
        )
        native_tokens = _token_ids(
            native_request["scored_target"]["token_ids"], label="native next action"
        )
        native_kind = str(native_request["scored_target"]["kind"])
        native_path = _description_path(native_tokens)

        prefill = backend.prefill(context_tokens)
        try:
            prefill.assert_rooted(label=f"release ladder {gt_owner_id}@{context_id}")
            target_scored = _teacher_forced(
                backend,
                prefill,
                root_token_ids=context_tokens,
                token_ids=target_tokens,
                uncached=release_uncached,
            )
            native_scored = _teacher_forced(
                backend,
                prefill,
                root_token_ids=context_tokens,
                token_ids=native_path,
                uncached=release_uncached,
            )
            prefill.assert_rooted(label=f"release ladder {gt_owner_id}@{context_id} (post)")
        finally:
            prefill.close()

        release = score_natural_release(
            boundary_label=boundary_label,
            target_token_ids=target_tokens,
            native_token_ids=native_path,
            target_description_logprobs=[t.selected_logprob for t in target_scored],
            native_description_logprobs=[t.selected_logprob for t in native_scored],
            target_argmax_ids=[t.argmax_token_id for t in target_scored],
        )
        replay_depth = (
            len(native_path)
            if release.first_divergence_index is None
            else min(int(release.first_divergence_index), len(native_path))
        )
        replay_admitted = replay_argmax_through_prefix(
            expected_token_ids=native_path,
            argmax_token_ids=[t.argmax_token_id for t in native_scored],
            up_to_index=replay_depth,
        )
        backend_name = UNCACHED_BACKEND if release_uncached else KV_CACHE_BACKEND
        state.score_rows.append(
            _score_row(
                shard_id=shard_id,
                request=release_request,
                binding=bindings[str(release_request["request_id"])],
                context_group_id=bare_group_id,
                scoring_backend=backend_name,
                scored_tokens=target_scored,
                observables={
                    "same_description": release.same_description,
                    "first_divergence_index": release.first_divergence_index,
                    "target_minus_native_margin": release.target_minus_native_margin,
                    "description_sum": release.description_sum,
                    "description_token_mean": release.description_token_mean,
                    "description_token_count": release.description_token_count,
                    "argmax_follows_target": release.argmax_follows_target,
                },
            )
        )
        state.score_rows.append(
            _score_row(
                shard_id=shard_id,
                request=native_request,
                binding=bindings[str(native_request["request_id"])],
                context_group_id=bare_group_id,
                scoring_backend=backend_name,
                scored_tokens=native_scored,
                observables={
                    "native_action_kind": native_kind,
                    "native_description_path_token_count": len(native_path),
                    "argmax_replay_depth": replay_depth,
                    "argmax_replay_admitted": replay_admitted,
                },
            )
        )

    # --- ladder 2: the forced D_C context group -------------------------
    coordinate_requests = (
        {family: requests[family] for family in COORDINATE_FAMILIES if family in requests}
        if SURFACE_COORDINATE in surfaces
        else {}
    )
    owner_rank: OwnerRankResult | None = None
    greedy_observables: dict[str, Any] = {"greedy_status": None, "greedy_owner_match": None}
    candidate_count = 0
    unattributable: list[str] = []
    # The (context, category) unique population for ``peak_lift`` is the
    # target-local and same-category competitor families together; the bank for
    # ``local_concentration`` is the target's generator-local family alone.
    population_entries: list[tuple[str, float]] = []
    target_local_request: Mapping[str, Any] | None = None
    target_local_scores: dict[str, float] = {}
    support_by_bound: dict[str, TargetLocalSupportResult] = {}
    if coordinate_requests:
        appended = bindings[str(next(iter(coordinate_requests.values()))["request_id"])][
            "appended_token_ids"
        ]
        for request in coordinate_requests.values():
            if bindings[str(request["request_id"])]["appended_token_ids"] != appended:
                _fail(
                    f"owner {gt_owner_id!r} coordinate requests at {context_id!r} do not share "
                    "one forced D_C prefix; they cannot share a context group"
                )
        dc_group_id = _context_group_id(
            gt_owner_id=gt_owner_id,
            context_id=context_id,
            variant=variant,
            appended_digest=sha256_json(appended),
            scoring_backend=UNCACHED_BACKEND if coordinate_uncached else KV_CACHE_BACKEND,
        )
        state.context_group_ids.append(dc_group_id)
        root_tokens = list(context_tokens) + list(appended)

        prefill = backend.prefill(root_tokens)
        try:
            prefill.assert_rooted(label=f"coordinate ladder {gt_owner_id}@{context_id}")
            all_scores: list[CandidateScore] = []
            for family in (REQUEST_COORDINATE_TARGET_LOCAL, REQUEST_COORDINATE_COMPETITOR):
                request = coordinate_requests.get(family)
                if request is None:
                    continue
                candidate_ids = [
                    str(value) for value in request["candidate_family"]["candidate_ids"]
                ]
                if sha256_json(candidate_ids) != request["candidate_family"][
                    "candidate_ids_sha256"
                ]:
                    _fail(f"request {request['request_id']!r} candidate id digest does not hold")
                coord_rows = [
                    _candidate_coord_tokens(plan, candidate_id) for candidate_id in candidate_ids
                ]
                if not coord_rows:
                    continue
                use_batching = batch_size > 1 and not coordinate_uncached
                if use_batching:
                    scored_rows = _score_candidate_boxes_batched(
                        prefill, coord_rows=coord_rows, batch_size=batch_size
                    )
                    state.batched_candidate_rows += len(scored_rows)
                else:
                    scored_rows = _score_candidate_boxes_scalar(
                        backend,
                        prefill,
                        root_token_ids=root_tokens,
                        coord_rows=coord_rows,
                        uncached=coordinate_uncached,
                    )
                    state.scalar_candidate_rows += len(scored_rows)
                box_sums = [_complete_box_logprob_sum(row) for row in scored_rows]
                family_scores, family_unattributable = _candidate_scores(
                    plan,
                    candidate_ids=candidate_ids,
                    context_id=context_id,
                    box_sums=box_sums,
                )
                all_scores.extend(family_scores)
                unattributable.extend(family_unattributable)
                candidate_count += len(candidate_ids)
                population_entries.extend(
                    (candidate_id, float(box_sum))
                    for candidate_id, box_sum in zip(candidate_ids, box_sums, strict=True)
                )
                if family == REQUEST_COORDINATE_TARGET_LOCAL:
                    target_local_request = request
                    target_local_scores = dict(
                        zip(candidate_ids, (float(value) for value in box_sums), strict=True)
                    )
                state.score_rows.append(
                    _score_row(
                        shard_id=shard_id,
                        request=request,
                        binding=bindings[str(request["request_id"])],
                        context_group_id=dc_group_id,
                        scoring_backend=(
                            UNCACHED_BACKEND
                            if coordinate_uncached
                            else KV_CACHE_BACKEND
                        ),
                        scored_tokens=[token for row in scored_rows for token in row],
                        observables={
                            "candidate_family": str(request["scored_target"]["family"]),
                            "candidate_ids": candidate_ids,
                            "complete_box_logprob_sums": box_sums,
                            "owner_identifiable_candidate_ids": [
                                score.candidate_id for score in family_scores
                            ],
                            "candidate_owner_ids": [score.owner_id for score in family_scores],
                            "not_owner_identifiable_candidate_ids": family_unattributable,
                            "batched": bool(use_batching),
                            "batch_size": int(batch_size) if use_batching else 1,
                        },
                    )
                )

            greedy_request = coordinate_requests.get(REQUEST_COORDINATE_GREEDY)
            if greedy_request is not None:
                grammar, greedy_scored = _greedy_coordinate_row(
                    backend,
                    prefill,
                    root_token_ids=root_tokens,
                    uncached=coordinate_uncached,
                )
                greedy_observables = _owner_match_greedy_box(
                    plan,
                    image_id=image_id,
                    normalized_description=normalized_description,
                    target_owner_id=gt_owner_id,
                    grammar=grammar,
                )
                greedy_observables.update(
                    {
                        "grammar_status": grammar.status,
                        "token_ids": list(grammar.token_ids),
                        "coord_token_ids": list(grammar.coord_token_ids),
                        "complete_box_logprob_sum": (
                            _complete_box_logprob_sum(greedy_scored)
                            if grammar.status == "valid"
                            else None
                        ),
                    }
                )
                state.score_rows.append(
                    _score_row(
                        shard_id=shard_id,
                        request=greedy_request,
                        binding=bindings[str(greedy_request["request_id"])],
                        context_group_id=dc_group_id,
                        scoring_backend=(
                            UNCACHED_BACKEND if coordinate_uncached else KV_CACHE_BACKEND
                        ),
                        scored_tokens=greedy_scored,
                        observables=greedy_observables,
                    )
                )
            prefill.assert_rooted(label=f"coordinate ladder {gt_owner_id}@{context_id} (post)")
        finally:
            prefill.close()

        if all_scores:
            owner_rank = rank_owner_candidates(all_scores, target_owner_id=gt_owner_id)

        if target_local_request is not None and population_entries:
            log_posterior_by_key, unique_population_size = unique_population_log_posteriors(
                population_entries
            )
            calibration = read_request_support_calibration(target_local_request)
            for bound in SUPPORT_BOUNDS:
                members = sealed_bound_candidate_ids(target_local_request, bound=bound)
                unknown = sorted(set(members) - set(target_local_scores))
                if unknown:
                    _fail(
                        f"the sealed {bound!r}-bound membership names candidate(s) {unknown!r} "
                        "that this request did not score; support would be computed over a "
                        "bank the capture never read"
                    )
                probes = [
                    BankProbe(
                        candidate_id=candidate_id,
                        complete_box_logprob_sum=target_local_scores[candidate_id],
                        # Membership is the plan's own frozen census
                        # classification of this probe against this owner.
                        bank_class=next(iter(SUPPORT_BOUND_MEMBERSHIP[bound])),
                    )
                    for candidate_id in members
                ]
                support_by_bound[bound] = evaluate_target_local_support(
                    compute_support_bound_features(
                        probes,
                        bound=bound,
                        log_posterior_by_key=log_posterior_by_key,
                        unique_population_size=unique_population_size,
                    ),
                    peak_lift_threshold=(
                        None if calibration is None else calibration["peak_lift_threshold"]
                    ),
                    local_concentration_threshold=(
                        None
                        if calibration is None
                        else calibration["local_concentration_threshold"]
                    ),
                    epsilon=(
                        SUPPORT_EPSILON
                        if calibration is None
                        else float(calibration["epsilon"])
                    ),
                    calibration_source=(
                        "absent_from_the_sealed_plan"
                        if calibration is None
                        else str(calibration["source"])
                    ),
                )

    return LadderReadout(
        boundary_label=boundary_label,
        context_id=context_id,
        context_group_id=bare_group_id,
        release=release,
        native_action_kind=native_kind,
        native_replay_admitted=replay_admitted,
        native_replay_depth=replay_depth,
        owner_rank=owner_rank,
        greedy=greedy_observables,
        candidate_count=candidate_count,
        unattributable_candidate_ids=tuple(unattributable),
        release_scoring_backend=UNCACHED_BACKEND if release_uncached else KV_CACHE_BACKEND,
        coordinate_scoring_backend=(
            UNCACHED_BACKEND if coordinate_uncached else KV_CACHE_BACKEND
        ),
        support_by_bound=support_by_bound,
    )


def _support_payload(support: TargetLocalSupportResult) -> dict[str, Any]:
    features = support.features
    return {
        "bound": support.bound,
        "support_disposition": support.support_disposition,
        "peak_lift": features.peak_lift,
        "local_concentration": features.local_concentration,
        "owner_best": features.owner_best,
        "owner_best_candidate_id": features.owner_best_candidate_id,
        "bank_median": features.bank_median,
        "bank_size": features.bank_size,
        "unique_population_size": features.unique_population_size,
        "peak_lift_threshold": support.peak_lift_threshold,
        "local_concentration_threshold": support.local_concentration_threshold,
        "epsilon": support.epsilon,
        "calibration_source": support.calibration_source,
        "derived_from_rank_or_margin": False,
    }


def _ladder_payload(readout: LadderReadout) -> dict[str, Any]:
    release = readout.release
    rank = readout.owner_rank
    return {
        "boundary_label": readout.boundary_label,
        "context_id": readout.context_id,
        "context_group_id": readout.context_group_id,
        "native_action_kind": readout.native_action_kind,
        "native_argmax_replay_admitted": readout.native_replay_admitted,
        "native_argmax_replay_depth": readout.native_replay_depth,
        "release_scoring_backend": readout.release_scoring_backend,
        "coordinate_scoring_backend": readout.coordinate_scoring_backend,
        "release": (
            None
            if release is None
            else {
                "observable": not release.same_description,
                "same_description_coordinate_only": release.same_description,
                "first_divergence_index": release.first_divergence_index,
                "target_minus_native_margin": release.target_minus_native_margin,
                "description_sum": release.description_sum,
                "description_token_mean": release.description_token_mean,
                "description_token_count": release.description_token_count,
                "argmax_follows_target": release.argmax_follows_target,
                "gate_margin_is_versus_stop": readout.native_action_kind
                == plan_builder.NATIVE_ACTION_STOP,
            }
        ),
        "coordinate": {
            "scored_candidate_count": readout.candidate_count,
            "not_owner_identifiable_candidate_ids": list(readout.unattributable_candidate_ids),
            "target_rank": None if rank is None else rank.target_rank,
            "best_competitor_owner_id": None if rank is None else rank.best_competitor_owner_id,
            "target_minus_competitor_margin": (
                None if rank is None else rank.target_minus_competitor_margin
            ),
            # Rank and calibrated support are reported side by side and never
            # merged: the first is competition between owners, the second is the
            # shape of the target's own landscape.
            "family_rank_disposition": (
                None if rank is None else rank.family_rank_disposition
            ),
            "support_by_bound": {
                bound: _support_payload(support)
                for bound, support in sorted(readout.support_by_bound.items())
            },
            "primary_support_bound": SUPPORT_BOUND_U,
            "support_disposition": readout.primary_support_disposition,
            "support_sensitivity_bound": SUPPORT_BOUND_L,
            "greedy": dict(readout.greedy),
        },
    }


# ---------------------------------------------------------------------------
# 14. Smoke matrix, cache parity, batch parity
# ---------------------------------------------------------------------------


def _smoke_branch(readout: LadderReadout) -> str:
    """The branch a single smoke execution would assign, for parity only.

    unit.md requires cached and uncached scoring to preserve the *compared*
    primary branch.  This is that comparison's input and never leaves the parity
    receipt: no owner record carries a branch.
    """

    if readout.boundary_label != "P_plus_E":
        # Only P+E is decision-bearing; the P readout is compared through every
        # other parity field instead of a branch it must never be assigned.
        return "not_classified_outside_p_plus_e"
    rank = readout.owner_rank
    greedy_status = readout.greedy.get("greedy_status") or GREEDY_MALFORMED
    missing = rank is None or readout.release is None
    if missing:
        return "ambiguous"
    displacement = classify_displacement(
        target_owner_id=str(rank.target_owner_id),
        owner_rank=rank,
        greedy_status=str(greedy_status),
        greedy_owner_match=readout.greedy.get("greedy_owner_match"),
    )
    return classify_primary_branch(
        displacement=displacement,
        release_observable=not readout.release.same_description,
        release_margin=readout.release.target_minus_native_margin,
        # Calibrated support, never the rank disposition.
        forced_dc_support_disposition=readout.primary_support_disposition,
        greedy_status=str(greedy_status),
        tie_or_nonunique=(
            rank.family_rank_disposition == RANK_TIE
            or greedy_status == GREEDY_AMBIGUOUS
            or bool(readout.greedy.get("greedy_nonunique_match"))
        ),
        missing_fields=False,
        boundary_label="P_plus_E",
    ).branch


def _margin_sign(margin: float | None) -> int:
    if margin is None or float(margin) == 0.0:
        return 0
    return 1 if float(margin) > 0.0 else -1


def _parity_inputs_from(
    cached: LadderReadout,
    uncached: LadderReadout,
    *,
    cached_score_rows: Sequence[Mapping[str, Any]] = (),
    uncached_score_rows: Sequence[Mapping[str, Any]] = (),
) -> ParityCheckInputs:
    def _selected(readout: LadderReadout) -> float:
        release = readout.release
        if release is None or release.target_minus_native_margin is None:
            return float(release.description_sum) if release is not None else 0.0
        return float(release.target_minus_native_margin)

    def _sign(readout: LadderReadout) -> int:
        rank = readout.owner_rank
        return _margin_sign(None if rank is None else rank.target_minus_competitor_margin)

    def _release_sign(readout: LadderReadout) -> int:
        release = readout.release
        return _margin_sign(None if release is None else release.target_minus_native_margin)

    def _argmax(readout: LadderReadout) -> int:
        coords = readout.greedy.get("coord_token_ids") or []
        return int(coords[0]) if coords else -1

    return ParityCheckInputs(
        cached_selected_logit=_selected(cached),
        uncached_selected_logit=_selected(uncached),
        cached_argmax_token_id=_argmax(cached),
        uncached_argmax_token_id=_argmax(uncached),
        cached_margin_sign=_sign(cached),
        uncached_margin_sign=_sign(uncached),
        cached_release_margin_sign=_release_sign(cached),
        uncached_release_margin_sign=_release_sign(uncached),
        cached_streams=parity_streams_by_surface(cached_score_rows),
        uncached_streams=parity_streams_by_surface(uncached_score_rows),
        cached_owner_rank=-1 if cached.owner_rank is None else cached.owner_rank.target_rank,
        uncached_owner_rank=(
            -1 if uncached.owner_rank is None else uncached.owner_rank.target_rank
        ),
        # Calibrated support under the primary U bound, not the rank surface.
        cached_support_disposition=cached.primary_support_disposition,
        uncached_support_disposition=uncached.primary_support_disposition,
        cached_owner_match=cached.greedy.get("greedy_owner_match"),
        uncached_owner_match=uncached.greedy.get("greedy_owner_match"),
        cached_primary_branch=_smoke_branch(cached),
        uncached_primary_branch=_smoke_branch(uncached),
    )


def smoke_role_of(row: Mapping[str, Any]) -> str:
    """The smoke-matrix role one sealed cohort row belongs to."""

    if bool(row.get("same_description_as_e")):
        return "same_desc"
    if str(row["e_row"]["stratum"]) == plan_builder.STRATUM_MATCHED_E:
        return "matched_e_diff_desc"
    return "unmatched_e_diff_desc"


def smoke_roles_by_image(
    cohort_rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, list[str]]]:
    """``{image_id: {role: [owner_id, ...]}}`` over the sealed primary cohort."""

    by_image: dict[str, dict[str, list[str]]] = {}
    for row in cohort_rows:
        image = by_image.setdefault(str(row["image_id"]), {})
        image.setdefault(smoke_role_of(row), []).append(str(row["gt_owner_id"]))
    for roles in by_image.values():
        for owner_ids in roles.values():
            owner_ids.sort()
    return by_image


def images_with_all_required_smoke_roles(
    cohort_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Images that can carry the whole smoke matrix inside one model session.

    Most images cannot: the cohort's 26 owners are spread thin, so the required
    matched-E, unmatched-E and same-description roles co-occur on only a couple
    of images.  That is exactly why the smoke is a dedicated single-image run
    whose receipt admits the later per-image shards, rather than a per-shard
    step that would otherwise have to score another image's context.
    """

    by_image = smoke_roles_by_image(cohort_rows)
    return sorted(
        (
            image_id
            for image_id, roles in by_image.items()
            if REQUIRED_SMOKE_MATRIX_ROLES <= set(roles)
        ),
        key=lambda value: (len(value), value),
    )


def select_smoke_owners(
    cohort_rows: Sequence[Mapping[str, Any]], *, image_id: str
) -> dict[str, str]:
    """One real owner per required smoke role, all from *one* image.

    Deterministic and score-blind: the lexicographically smallest owner id in
    each role, restricted to ``image_id``.  Restricting to one image is a
    correctness requirement, not a convenience: a session is opened for exactly
    one image's ``pixel_values``/``image_grid_thw``, so scoring another image's
    context through it would silently pair a prefix with the wrong picture.
    """

    roles = smoke_roles_by_image(cohort_rows).get(str(image_id), {})
    missing = sorted(REQUIRED_SMOKE_MATRIX_ROLES - set(roles))
    if missing:
        eligible = images_with_all_required_smoke_roles(cohort_rows)
        _fail(
            f"image {image_id!r} has no owner for required smoke role(s) {missing!r}; the "
            "smoke matrix must run inside one image session, and the sealed cohort supports "
            f"it only on image(s) {eligible!r}"
        )
    selected = {role: roles[role][0] for role in sorted(REQUIRED_SMOKE_MATRIX_ROLES)}
    validate_smoke_matrix_roles(sorted(selected))
    return selected


def run_batch_parity(
    plan: SealedPlan,
    backend: Any,
    *,
    context_token_ids: Sequence[int],
    appended_token_ids: Sequence[int],
    candidate_ids: Sequence[str],
    batch_size: int,
) -> dict[str, Any]:
    """Prove batched lanes equal the single branch before any batching is used."""

    coord_rows = [_candidate_coord_tokens(plan, cid) for cid in candidate_ids[:batch_size]]
    if len(coord_rows) < 2:
        return {
            "status": "not_applicable",
            "reason": "fewer than two candidates share this admitted prefix",
            "requested_batch_size": int(batch_size),
            "effective_batch_size": 1,
        }
    root_tokens = list(context_token_ids) + list(appended_token_ids)
    prefill = backend.prefill(root_tokens)
    try:
        scalar = _score_candidate_boxes_scalar(
            backend, prefill, root_token_ids=root_tokens, coord_rows=coord_rows, uncached=False
        )
        batched = _score_candidate_boxes_batched(
            prefill, coord_rows=coord_rows, batch_size=len(coord_rows)
        )
    finally:
        prefill.close()

    diffs = [
        abs(_complete_box_logprob_sum(left) - _complete_box_logprob_sum(right))
        for left, right in zip(scalar, batched, strict=True)
    ]
    argmax_parity = all(
        [token.argmax_token_id for token in left] == [token.argmax_token_id for token in right]
        for left, right in zip(scalar, batched, strict=True)
    )
    max_diff = max(diffs) if diffs else 0.0
    passed = bool(argmax_parity and max_diff <= BATCH_PARITY_MAX_ABS_DIFF)
    return {
        "status": "passed" if passed else "failed",
        "requested_batch_size": int(batch_size),
        "effective_batch_size": len(coord_rows) if passed else 1,
        "compared_candidate_count": len(coord_rows),
        "max_complete_box_logprob_abs_diff": max_diff,
        "max_abs_diff_threshold": BATCH_PARITY_MAX_ABS_DIFF,
        "argmax_parity": argmax_parity,
        "equal_image_prefix_and_context_shape": True,
    }


# ---------------------------------------------------------------------------
# 15. Shard assembly and publication
# ---------------------------------------------------------------------------


def _requests_for_owner(
    plan: SealedPlan, gt_owner_id: str
) -> dict[str, dict[str, Mapping[str, Any]]]:
    """Primary requests of one owner, grouped by plan variant then family."""

    grouped: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in plan.request_rows:
        if str(row.get("gt_owner_id")) != str(gt_owner_id):
            continue
        if str(row.get("readout_tier")) != PRIMARY_READOUT_TIER:
            continue
        variant = str(row["variant"])
        family = str(row["request_family"])
        if family in grouped.setdefault(variant, {}):
            _fail(
                f"owner {gt_owner_id!r} has two {family!r} requests at variant {variant!r}"
            )
        grouped[variant][family] = row
    return grouped


def _publish(output_dir: Path, files: Mapping[str, bytes]) -> dict[str, Any]:
    """Atomic create-or-identical publish, on the predecessor's own primitives."""

    shard = _census_shard()
    output_dir = Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        state = shard.inspect_published_shard(output_dir, files)
        if state["identical"]:
            return {
                "output_dir": str(output_dir),
                "published": False,
                "publish_mode": "no_op_identical_rerun",
                "file_names": sorted(files),
            }
        _fail(
            f"refusing to publish into existing output directory {output_dir}: it is not a "
            f"byte-identical capture of this shard (missing={state['missing']!r}, "
            f"differing={state['differing']!r}, unexpected={state['unexpected']!r}); the "
            "existing directory is left untouched"
        )
    staging = output_dir.parent / f"{output_dir.name}.staging-{os.getpid()}-{int(time.time())}"
    if staging.exists():
        _fail(f"staging directory {staging} already exists")
    staging.mkdir(parents=True)
    published = False
    try:
        for name in sorted(files):
            shard._write_durable(staging / name, files[name])  # noqa: SLF001
        shard._fsync_dir(staging)  # noqa: SLF001
        os.rename(staging, output_dir)
        published = True
    finally:
        if not published:
            for child in sorted(staging.iterdir()):
                child.unlink()
            staging.rmdir()
    shard._fsync_dir(output_dir.parent)  # noqa: SLF001
    return {
        "output_dir": str(output_dir),
        "published": True,
        "publish_mode": "atomic_staging_directory_rename",
        "file_names": sorted(files),
    }


@dataclass
class ShardResult:
    receipt: dict[str, Any]
    score_rows: list[dict[str, Any]] = field(default_factory=list)
    owner_records: list[dict[str, Any]] = field(default_factory=list)
    parity: dict[str, Any] = field(default_factory=dict)
    quarantine: dict[str, Any] | None = None
    admission: dict[str, Any] | None = None


def shard_output_files(result: ShardResult) -> dict[str, bytes]:
    """The indivisible published byte content of one shard."""

    if result.quarantine is not None:
        return {
            QUARANTINE_NAME: canonical_json_bytes(result.quarantine) + b"\n",
            RECEIPT_NAME: canonical_json_bytes(result.receipt) + b"\n",
        }
    if result.admission is not None:
        return {
            ADMISSION_NAME: canonical_json_bytes(result.admission) + b"\n",
            RECEIPT_NAME: canonical_json_bytes(result.receipt) + b"\n",
        }
    return {
        SCORES_NAME: b"".join(canonical_json_bytes(row) + b"\n" for row in result.score_rows),
        OWNER_RECORDS_NAME: b"".join(
            canonical_json_bytes(row) + b"\n" for row in result.owner_records
        ),
        PARITY_NAME: canonical_json_bytes(result.parity) + b"\n",
        RECEIPT_NAME: canonical_json_bytes(result.receipt) + b"\n",
    }


#: Runtime fields the admission digest covers.  Deliberately *excludes* every
#: per-image field (``executed_media_sha256``, ``executed_prompt_token_count``,
#: ``image_grid_thw``, ``session_image_id``): those legitimately differ between
#: shards and are fail-closed separately, per shard, by
#: :meth:`SealedPlan.assert_context_belongs_to_session_image`.
ADMISSION_IDENTITY_FIELDS: tuple[str, ...] = (
    "backend",
    "model_identity",
    "tokenizer_identity",
    "adapter_identity",
    "numerics",
    "source_identity",
)

#: Modules whose source actually determines parity, runtime and matcher
#: semantics.  An admission proved under one revision must not authorize a
#: capture running a different one.
SOURCE_IDENTITY_MODULES: tuple[str, ...] = (
    "scripts.research.score_sorted_crossing_boundary_owner_release",
    "scripts.research.score_sorted_owner_accessibility_census_shard",
    "scripts.research.score_sorted_owner_basin_landscape",
    "scripts.research.prepare_sorted_crossing_boundary_owner_release_realization",
    "scripts.research.build_sorted_all_person_greedy_boundary_census",
    "scripts.research.build_sorted_owner_accessibility_census_plan",
    # Not imported, but this unit reproduces its ``_support_features``
    # peak_lift/local_concentration reconstruction and its U/L bound partition;
    # binding the digest makes a drift between the two visible instead of silent.
    "scripts.research.merge_sorted_owner_accessibility_census_shards",
)


def source_identity() -> dict[str, str]:
    """Content digests of every module this capture's semantics depend on."""

    import importlib

    digests: dict[str, str] = {}
    for name in SOURCE_IDENTITY_MODULES:
        module = importlib.import_module(name)
        source = getattr(module, "__file__", None)
        if source is None:
            _fail(f"module {name!r} exposes no source file to hash")
        digests[name] = sha256_file(Path(source))
    return dict(sorted(digests.items()))


def _observed_numerics(backend: Any) -> dict[str, Any]:
    """Precision and attention state read off the *live* model, not a receipt."""

    model = getattr(backend, "_model", None)
    if model is None:
        return {"model_introspected": False}
    config = getattr(model, "config", None)
    dtype_counts: dict[str, int] = {}
    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        for parameter in parameters():
            key = str(getattr(parameter, "dtype", None))
            dtype_counts[key] = dtype_counts.get(key, 0) + 1
    return {
        "model_introspected": True,
        "config_dtype": str(
            getattr(config, "dtype", None) or getattr(config, "torch_dtype", None)
        ),
        "attn_implementation": str(
            getattr(config, "_attn_implementation", None)
            or getattr(config, "attn_implementation", None)
        ),
        "parameter_dtype_tensor_counts": dict(sorted(dtype_counts.items())),
    }


def build_runtime_numerics(backend: Any, *, infer_config: Path | None) -> dict[str, Any]:
    """Every runtime field that can move a cached-versus-uncached comparison.

    unit.md requires precision, attention, position-ID and repetition-penalty
    identity to be bound.  Model and tokenizer identity alone are not enough:
    the same checkpoint under a different infer config, dtype, attention kernel
    or TF32 setting is a different numerical runtime, and would otherwise
    inherit a cache admission it never earned.
    """

    numerics: dict[str, Any] = {
        "explicit_position_ids": True,
        "uses_model_generate": False,
        "vision_kwargs_at_prefill_only": True,
        "repetition_penalty_stratum": float(NATIVE_REPETITION_PENALTY_STRATUM),
        "likelihood_channel": LIKELIHOOD_CHANNEL,
        "matmul_precision": _census_shard().matmul_precision_state(),
        **_observed_numerics(backend),
    }
    declared = infer_config or (backend.identity or {}).get("infer_config")
    if declared is None:
        numerics["infer_config_path"] = None
        numerics["infer_config_sha256"] = None
        return numerics
    path = Path(str(declared)).expanduser()
    if not path.is_file():
        _fail(
            f"the executed infer config {path} is not readable; its content hash must be "
            "bound into the admission identity"
        )
    numerics["infer_config_path"] = str(path)
    numerics["infer_config_sha256"] = sha256_file(path)
    return numerics


def runtime_identity_digest(runtime_identity: Mapping[str, Any]) -> str:
    """The identity a capture shard must reproduce to inherit an admission."""

    return sha256_json(admission_identity_payload(runtime_identity))


def admission_identity_payload(runtime_identity: Mapping[str, Any]) -> dict[str, Any]:
    """Every field the admission digest covers, carried literally.

    Generated from :data:`ADMISSION_IDENTITY_FIELDS` rather than enumerated at
    the call site: a capture rebuilds the digest from the admission's *own*
    declarations, so a field the receipt failed to carry would read back as
    ``None``, reconstruct a different digest, and refuse every honest capture.
    """

    return {field: runtime_identity.get(field) for field in ADMISSION_IDENTITY_FIELDS}


def smoke_executed_counters(state: CaptureState) -> dict[str, Any]:
    """What a smoke executed, kept distinct from what a smoke publishes.

    A smoke publishes no primary evidence, so ``score_row_count`` is zero by
    contract.  Sealing only that would read as "nothing was scored", so the
    executed row count and the logical-context-group count are sealed beside
    it, taken from the state the per-role executions fold back into.
    """

    return {
        "score_row_count": 0,
        "executed_unpublished_score_row_count": len(state.score_rows),
        "logical_context_group_count": len(state.context_group_ids),
        "logical_context_groups_sha256": sha256_json(sorted(state.context_group_ids)),
    }


def validate_admission_receipt(
    admission: Mapping[str, Any],
    *,
    plan: SealedPlan,
    runtime_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """An admission is inherited only under the exact plan and runtime it proved.

    The smoke ran inside one image session; its conclusions transfer to other
    images only because the model, tokenizer and sealed plan are identical, so
    every one of those is re-proven here rather than assumed.
    """

    if str(admission.get("schema_version")) != ADMISSION_SCHEMA_VERSION:
        _fail(
            f"admission receipt schema {admission.get('schema_version')!r} is not "
            f"{ADMISSION_SCHEMA_VERSION!r}"
        )
    if str(admission.get("unit_id")) != UNIT_ID:
        _fail("admission receipt belongs to another unit")
    declared_seal = admission.get("admission_content_sha256")
    unsealed = {
        key: value
        for key, value in admission.items()
        if key != "admission_content_sha256"
    }
    if sha256_json(unsealed) != declared_seal:
        _fail("admission receipt does not self-seal; it has been edited after the smoke")
    if admission.get("plan_manifest_content_sha256") != plan.manifest.get(
        "manifest_content_sha256"
    ):
        _fail(
            "admission receipt was sealed against a different plan manifest; a smoke cannot "
            "admit a capture of another plan"
        )
    # Recomputed from the admission's *own* declared identity fields, never
    # read as a trusted scalar: an edited receipt that re-seals its content hash
    # would otherwise keep a stale ``runtime_identity_sha256`` and authorize a
    # runtime it never proved.
    declared_digest = runtime_identity_digest(admission)
    if declared_digest != str(admission.get("runtime_identity_sha256")):
        _fail(
            "admission receipt's runtime_identity_sha256 does not reconstruct from its own "
            "declared identity fields; the receipt is internally inconsistent"
        )
    observed = runtime_identity_digest(runtime_identity)
    if declared_digest != observed:
        differing = sorted(
            field
            for field in ADMISSION_IDENTITY_FIELDS
            if sha256_json(admission.get(field)) != sha256_json(runtime_identity.get(field))
        )
        _fail(
            "admission receipt was sealed under a different runtime identity "
            f"(differing field(s): {differing or ['<unreported>']!r}); cached execution and "
            "batching are never inherited across a changed model, tokenizer, numerical "
            "runtime or scorer revision"
        )
    if not bool(admission.get("cache_admitted")) and any(
        str(value) != UNCACHED_BACKEND
        for value in (admission.get("surface_backend") or {}).values()
    ):
        _fail("admission receipt is internally inconsistent about its surface backends")
    for surface in SURFACES:
        if surface not in (admission.get("surface_backend") or {}):
            _fail(f"admission receipt declares no backend for surface {surface!r}")
    validate_smoke_matrix_roles(
        [str(row["role"]) for row in (admission.get("smoke_rows") or ())]
    )
    return dict(admission)


def _base_receipt(
    plan: SealedPlan,
    *,
    shard_id: str,
    runtime_identity: Mapping[str, Any],
    mode: str,
    executed: Mapping[str, Any],
    quarantine: QuarantineLedger,
    stopped: bool,
) -> dict[str, Any]:
    """The identity/denominator/policy spine both run modes seal."""

    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "mode": str(mode),
        "shard_id": str(shard_id),
        "scorer_source_sha256": sha256_bytes(Path(__file__).read_bytes()),
        "plan": {
            "plan_dir": str(plan.plan_dir),
            "manifest_schema_version": plan.manifest.get("schema_version"),
            "manifest_content_sha256": plan.manifest.get("manifest_content_sha256"),
            "builder_source_sha256": builder_source_sha256(plan.manifest),
            "plan_file_sha256": dict(sorted(plan.plan_file_sha256.items())),
            "lineage": plan.manifest.get("lineage"),
            "cohort_counts": plan.manifest.get("cohort_counts"),
            "control_counts": plan.manifest.get("control_counts"),
            # Named explicitly rather than left implicit inside the manifest
            # digest: an audit has to see which census calibration the support
            # dispositions in this shard were decided against.
            "support_calibration": read_plan_support_calibration(plan),
        },
        "runtime_identity": dict(runtime_identity),
        "runtime_identity_sha256": runtime_identity_digest(runtime_identity),
        "cohort_denominators": validate_primary_cohort_counts(
            u_count=int(plan.manifest["cohort_counts"]["u_bound_crossing_count"]),
            l_count=int(plan.manifest["cohort_counts"]["l_bound_crossing_count"]),
            same_context_ul_count=int(
                plan.manifest["cohort_counts"]["exact_same_context_u_and_l_count"]
            ),
            matched_e_count=int(plan.manifest["cohort_counts"]["matched_e_count"]),
            unmatched_e_count=int(plan.manifest["cohort_counts"]["unmatched_e_count"]),
        ),
        "timing_control_owner_count": validate_timing_control_count(
            int(plan.manifest["control_counts"]["timing_control_count"])
        ),
        "tp_calibration_owner_count": TP_CALIBRATION_OWNER_COUNT,
        "executed": dict(executed),
        "quarantine": {
            "schema_version": QUARANTINE_SCHEMA_VERSION,
            "count": quarantine.count,
            "maximum_primary_quarantines": MAX_PRIMARY_QUARANTINES,
            "stopped": bool(stopped),
            "entries": [
                {"owner_id": entry.owner_id, "reason": entry.reason, "detail": entry.detail}
                for entry in quarantine.entries
            ],
        },
        "policy": {
            "uses_model_generate": False,
            "sampling": "not_implemented_primary_deterministic_pass_only",
            "retokenizes": False,
            "likelihood_channel": LIKELIHOOD_CHANNEL,
            "repetition_penalty_stratum": float(NATIVE_REPETITION_PENALTY_STRATUM),
            "owner_match_iou_threshold": OWNER_MATCH_IOU_THRESHOLD,
            "vision_kwargs_passed_at_prefill_only": True,
            "fresh_cache_per_logical_context_group": True,
            "one_image_session_per_shard": True,
            "branch_assignment_performed": False,
            "secondary_compatibility_executed": False,
        },
        # No wall-clock field is sealed: the published artifact set must be
        # byte-identical across re-runs so an idempotent re-capture publishes as
        # a no-op instead of colliding with itself.
        "artifact_determinism": "no_wall_clock_or_host_specific_field_is_sealed",
    }


# ---------------------------------------------------------------------------
# 16. Shard driver
# ---------------------------------------------------------------------------


def _validate_request_rows(plan: SealedPlan) -> dict[str, dict[str, Any]]:
    return {
        str(row["request_id"]): validate_request_row(plan, row) for row in plan.request_rows
    }


def run_smoke_shard(
    *,
    plan: SealedPlan,
    backend: Any,
    shard_id: str,
    image_id: str,
    batch_size: int,
    runtime_identity: Mapping[str, Any],
) -> ShardResult:
    """The dedicated single-image smoke that admits every later capture shard.

    All three required roles are scored inside *one* image session, so no
    context is ever forwarded against another image's visual state.  The result
    is a sealed admission receipt: cached-versus-uncached parity, the compared
    primary branch, and the proven batch width.
    """

    cohort_by_owner = {str(row["gt_owner_id"]): row for row in plan.cohort_rows}
    validated = _validate_request_rows(plan)
    smoke_owners = select_smoke_owners(plan.cohort_rows, image_id=image_id)

    smoke_rows: list[dict[str, Any]] = []
    smoke_matrix_admitted = True
    state = CaptureState()
    variant = "at_p_plus_e"

    # The three required roles at the decision-bearing P+E boundary, plus one
    # P readout of the same owner: cheap insurance that the cached mechanism is
    # proven at both boundaries this unit reads, still inside one image session.
    smoke_units: list[tuple[str, str, str]] = [
        (role, owner_id, variant) for role, owner_id in sorted(smoke_owners.items())
    ]
    smoke_units.append(("matched_e_diff_desc", smoke_owners["matched_e_diff_desc"], "at_p"))

    for role, owner_id, unit_variant in smoke_units:
        cohort_row = cohort_by_owner[owner_id]
        grouped = _requests_for_owner(plan, owner_id)
        if unit_variant not in grouped:
            _fail(
                f"smoke owner {owner_id!r} has no {unit_variant!r} requests in the sealed plan"
            )
        readouts: dict[str, LadderReadout] = {}
        side_rows: dict[str, list[dict[str, Any]]] = {}
        for label, backends in (
            ("uncached", {surface: UNCACHED_BACKEND for surface in SURFACES}),
            ("cached", {surface: KV_CACHE_BACKEND for surface in SURFACES}),
        ):
            # Each side needs its own ``surface_backend``, so the two executions
            # cannot share one state object; their bookkeeping is folded back
            # into the shard's state instead of being discarded, which is what
            # makes the sealed group counter the number actually executed.
            side_state = CaptureState(surface_backend=dict(backends))
            readouts[label] = _capture_boundary(
                plan,
                backend,
                side_state,
                shard_id=shard_id,
                gt_owner_id=owner_id,
                image_id=str(cohort_row["image_id"]),
                session_image_id=str(image_id),
                normalized_description=str(cohort_row["normalized_description"]),
                variant=unit_variant,
                requests=grouped[unit_variant],
                bindings=validated,
                batch_size=1,
            )
            state.context_group_ids.extend(side_state.context_group_ids)
            state.score_rows.extend(side_state.score_rows)
            side_rows[label] = list(side_state.score_rows)
        parity = evaluate_cache_parity(
            _parity_inputs_from(
                readouts["cached"],
                readouts["uncached"],
                cached_score_rows=side_rows["cached"],
                uncached_score_rows=side_rows["uncached"],
            )
        )
        if parity.status != CACHE_ADMITTED:
            smoke_matrix_admitted = False
        smoke_rows.append(
            {
                "role": role,
                "gt_owner_id": owner_id,
                "image_id": str(cohort_row["image_id"]),
                "variant": unit_variant,
                "boundary_label": BOUNDARY_LABEL_BY_VARIANT.get(unit_variant),
                "status": parity.status,
                "max_selected_logit_abs_diff": parity.max_selected_logit_abs_diff,
                "max_selected_logit_abs_diff_threshold": (
                    CACHE_PARITY_MAX_SELECTED_LOGIT_ABS_DIFF
                ),
                "mismatched_fields": list(parity.mismatched_fields),
                "compared_fields": [name for name, _, _ in PARITY_COMPARED_FIELDS],
                # The conclusion-bearing detail: how many tokens each ladder
                # actually compared, and its own worst drift.
                "per_surface": [
                    {
                        "surface": result.surface,
                        "aligned": result.aligned,
                        "compared_token_count": result.compared_token_count,
                        "max_selected_logit_abs_diff": result.max_selected_logit_abs_diff,
                        "argmax_parity": result.argmax_parity,
                    }
                    for result in parity.per_surface
                ],
                "cached_primary_branch_compared_only_here": _smoke_branch(readouts["cached"]),
                "uncached_primary_branch_compared_only_here": _smoke_branch(
                    readouts["uncached"]
                ),
            }
        )
    smoke_summary = validate_smoke_matrix_roles([row["role"] for row in smoke_rows])
    # The cached and uncached sides of one role are two separate executions on
    # their own fresh state, never one cache read twice; this proves it.
    assert_fresh_context_per_owner(state.context_group_ids)

    surface_backend = {
        surface: (KV_CACHE_BACKEND if smoke_matrix_admitted else UNCACHED_BACKEND)
        for surface in SURFACES
    }

    # Batching engages only after the semantic smoke has already passed.
    batch_parity: dict[str, Any] = {
        "status": "not_requested",
        "requested_batch_size": int(batch_size),
        "effective_batch_size": 1,
    }
    if int(batch_size) > 1:
        if not smoke_matrix_admitted:
            batch_parity = {
                "status": "skipped_semantic_smoke_not_admitted",
                "requested_batch_size": int(batch_size),
                "effective_batch_size": 1,
            }
        else:
            probe = _requests_for_owner(plan, smoke_owners["matched_e_diff_desc"])[variant][
                REQUEST_COORDINATE_TARGET_LOCAL
            ]
            plan.assert_context_belongs_to_session_image(
                str(probe["context_id"]),
                session_image_id=str(image_id),
                label="batch-parity probe",
            )
            batch_parity = run_batch_parity(
                plan,
                backend,
                context_token_ids=plan.executed_prefix_token_ids(str(probe["context_id"])),
                appended_token_ids=validated[str(probe["request_id"])]["appended_token_ids"],
                candidate_ids=[
                    str(value) for value in probe["candidate_family"]["candidate_ids"]
                ],
                batch_size=int(batch_size),
            )

    admission = {
        "schema_version": ADMISSION_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "smoke_shard_id": str(shard_id),
        "smoke_image_id": str(image_id),
        "plan_manifest_content_sha256": plan.manifest.get("manifest_content_sha256"),
        "runtime_identity_sha256": runtime_identity_digest(runtime_identity),
        "admission_identity_fields": list(ADMISSION_IDENTITY_FIELDS),
        **admission_identity_payload(runtime_identity),
        "smoke_matrix": smoke_summary,
        "smoke_rows": smoke_rows,
        "cache_admitted": smoke_matrix_admitted,
        "surface_backend": dict(sorted(surface_backend.items())),
        "batch_parity": batch_parity,
        "admitted_batch_size": (
            int(batch_parity["effective_batch_size"])
            if batch_parity.get("status") == "passed"
            else 1
        ),
        "scope": (
            "one image session carried the whole required role matrix; no context of another "
            "image was forwarded through it"
        ),
    }
    admission["admission_content_sha256"] = sha256_json(admission)

    receipt = _base_receipt(
        plan,
        shard_id=shard_id,
        runtime_identity=runtime_identity,
        mode=MODE_SMOKE,
        executed={
            "smoke_image_id": str(image_id),
            "smoke_owner_ids": dict(sorted(smoke_owners.items())),
            **smoke_executed_counters(state),
        },
        quarantine=state.quarantine,
        stopped=False,
    )
    receipt["admission"] = admission
    receipt["receipt_content_sha256"] = sha256_json(receipt)
    return ShardResult(receipt=receipt, admission=admission)


def run_capture_shard(
    *,
    plan: SealedPlan,
    backend: Any,
    shard_id: str,
    session_image_id: str,
    owner_ids: Sequence[str],
    admission: Mapping[str, Any],
    runtime_identity: Mapping[str, Any],
) -> ShardResult:
    """Execute the primary deterministic pass for one image's owners."""

    state = CaptureState(
        surface_backend={
            surface: str(admission["surface_backend"][surface]) for surface in SURFACES
        }
    )
    effective_batch_size = int(admission["admitted_batch_size"])
    cohort_by_owner = {str(row["gt_owner_id"]): row for row in plan.cohort_rows}
    control_by_owner = {str(row["gt_owner_id"]): row for row in plan.control_rows}
    validated = _validate_request_rows(plan)

    deferred_secondary = sum(
        1
        for row in plan.request_rows
        if str(row.get("readout_tier")) != PRIMARY_READOUT_TIER
    )
    stopped_by_quarantine = False
    for owner_id in owner_ids:
        registry_row = cohort_by_owner.get(owner_id) or control_by_owner.get(owner_id)
        if registry_row is None:
            _fail(f"owner {owner_id!r} is in neither the cohort nor the control registry")
        is_primary = owner_id in cohort_by_owner
        grouped = _requests_for_owner(plan, owner_id)
        if not grouped:
            _fail(f"owner {owner_id!r} has no primary request in the sealed plan")
        description = str(
            registry_row.get("normalized_description")
            or registry_row["target_description_path"]["normalized_description"]
        )
        readouts: dict[str, LadderReadout] = {}
        for variant in sorted(grouped):
            readouts[variant] = _capture_boundary(
                plan,
                backend,
                state,
                shard_id=shard_id,
                gt_owner_id=owner_id,
                image_id=str(registry_row["image_id"]),
                session_image_id=str(session_image_id),
                normalized_description=description,
                variant=variant,
                requests=grouped[variant],
                bindings=validated,
                batch_size=effective_batch_size,
            )

        replay_failures = sorted(
            variant for variant, readout in readouts.items() if not readout.native_replay_admitted
        )
        if replay_failures and state.surface_backend[SURFACE_RELEASE] == KV_CACHE_BACKEND:
            # Uncached fallback for exactly the affected surface.  The coordinate
            # ladder already passed on the admitted backend and is not re-run.
            fallback_state = CaptureState(
                surface_backend={
                    SURFACE_RELEASE: UNCACHED_BACKEND,
                    SURFACE_COORDINATE: state.surface_backend[SURFACE_COORDINATE],
                }
            )
            for variant in replay_failures:
                retry = _capture_boundary(
                    plan,
                    backend,
                    fallback_state,
                    shard_id=shard_id,
                    gt_owner_id=owner_id,
                    image_id=str(registry_row["image_id"]),
                    session_image_id=str(session_image_id),
                    normalized_description=description,
                    variant=variant,
                    requests=grouped[variant],
                    bindings=validated,
                    batch_size=effective_batch_size,
                    surfaces=frozenset({SURFACE_RELEASE}),
                )
                readouts[variant] = replace(
                    readouts[variant],
                    release=retry.release,
                    native_action_kind=retry.native_action_kind,
                    native_replay_admitted=retry.native_replay_admitted,
                    native_replay_depth=retry.native_replay_depth,
                    release_scoring_backend=UNCACHED_BACKEND,
                )
            state.score_rows.extend(fallback_state.score_rows)
            state.context_group_ids.extend(fallback_state.context_group_ids)
            replay_failures = sorted(
                variant
                for variant, readout in readouts.items()
                if not readout.native_replay_admitted
            )

        replay_admitted = not replay_failures
        if is_primary:
            state.quarantine = apply_native_replay_quarantine(
                argmax_replay_matches=replay_admitted,
                owner_id=owner_id,
                reason="pre_divergence_argmax_mismatch",
                detail=(
                    "native argmax replay diverged before the tested boundary at variant(s) "
                    f"{replay_failures!r}, cached and uncached"
                ),
                ledger=state.quarantine,
            )
        state.unattributable_candidate_ids.extend(
            candidate_id
            for readout in readouts.values()
            for candidate_id in readout.unattributable_candidate_ids
        )

        state.owner_records.append(
            {
                "schema_version": OWNER_RECORD_SCHEMA_VERSION,
                "row_kind": "crossing_boundary_owner_record",
                "unit_id": UNIT_ID,
                "shard_id": str(shard_id),
                "gt_owner_id": owner_id,
                "image_id": str(registry_row["image_id"]),
                "cohort": str(registry_row["cohort"]),
                "normalized_description": description,
                "stratum": (
                    str(registry_row["e_row"]["stratum"]) if is_primary else None
                ),
                "description_observability": registry_row.get("description_observability"),
                "same_description_as_e": registry_row.get("same_description_as_e"),
                "replay_admitted": replay_admitted,
                "quarantined": bool(is_primary and not replay_admitted),
                "ladders": {
                    variant: _ladder_payload(readout)
                    for variant, readout in sorted(readouts.items())
                },
                "branch_assignment": (
                    "not_performed_in_capture_pure_helpers_own_it_in_the_later_analysis"
                ),
                "secondary_compatibility": (
                    "deferred_until_primary_branches_are_sealed"
                ),
            }
        )

        if is_primary and check_quarantine_stop(state.quarantine):
            stopped_by_quarantine = True
            break

    assert_fresh_context_per_owner(state.context_group_ids)

    parity_payload = {
        "schema_version": PARITY_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "shard_id": str(shard_id),
        "session_image_id": str(session_image_id),
        "inherited_admission": {
            "smoke_shard_id": admission.get("smoke_shard_id"),
            "smoke_image_id": admission.get("smoke_image_id"),
            "admission_content_sha256": admission.get("admission_content_sha256"),
            "cache_admitted": admission.get("cache_admitted"),
            "admitted_batch_size": admission.get("admitted_batch_size"),
        },
        "surface_backend": dict(sorted(state.surface_backend.items())),
        "effective_batch_size": effective_batch_size,
        "batched_candidate_rows": state.batched_candidate_rows,
        "scalar_candidate_rows": state.scalar_candidate_rows,
        "branch_parity_scope": (
            "the compared primary branch exists only inside the admission receipt; owner "
            "records carry no branch"
        ),
    }

    receipt = _base_receipt(
        plan,
        shard_id=shard_id,
        runtime_identity=runtime_identity,
        mode=MODE_CAPTURE,
        executed={
            "session_image_id": str(session_image_id),
            "owner_ids": list(owner_ids),
            "owner_count": len(state.owner_records),
            "score_row_count": len(state.score_rows),
            "logical_context_group_count": len(state.context_group_ids),
            "logical_context_groups_sha256": sha256_json(sorted(state.context_group_ids)),
            "request_ids_sha256": sha256_json(
                sorted({str(row["request_id"]) for row in state.score_rows})
            ),
            "deferred_secondary_request_count": deferred_secondary,
            "not_owner_identifiable_candidate_ids": sorted(
                set(state.unattributable_candidate_ids)
            ),
        },
        quarantine=state.quarantine,
        stopped=stopped_by_quarantine,
    )
    receipt["admission"] = {
        "smoke_shard_id": admission.get("smoke_shard_id"),
        "smoke_image_id": admission.get("smoke_image_id"),
        "admission_content_sha256": admission.get("admission_content_sha256"),
        "surface_backend": dict(sorted(state.surface_backend.items())),
        "admitted_batch_size": effective_batch_size,
    }
    receipt["receipt_content_sha256"] = sha256_json(receipt)

    if stopped_by_quarantine:
        return ShardResult(
            receipt=receipt,
            quarantine={
                "schema_version": QUARANTINE_SCHEMA_VERSION,
                "unit_id": UNIT_ID,
                "shard_id": str(shard_id),
                "session_image_id": str(session_image_id),
                "reason": "more_than_two_quarantined_primary_owners",
                "entries": [
                    {"owner_id": entry.owner_id, "reason": entry.reason, "detail": entry.detail}
                    for entry in state.quarantine.entries
                ],
                "primary_evidence_withheld": list(PRIMARY_OUTPUT_NAMES),
                "next_step": "repair the runtime alignment rather than changing thresholds",
            },
        )
    return ShardResult(
        receipt=receipt,
        score_rows=state.score_rows,
        owner_records=state.owner_records,
        parity=parity_payload,
    )


# ---------------------------------------------------------------------------
# 17. CLI
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CensusImageRegistryView:
    """The one attribute ``build_hf_session_spec`` reads off a census plan.

    That function resolves a production session from ``plan.images`` alone: it
    checks membership and then reads the image's ``prompt_token_ids`` and
    ``executed_media_sha256``.  This unit's :class:`SealedPlan` carries the same
    sealed rows under ``inputs.images_by_id``, so the seam needs a name
    adapter, not a copy -- the mapping handed over is the *same* registry the
    scorer binds contexts against in
    :meth:`SealedPlan.assert_context_belongs_to_session_image`, which is what
    keeps one image identity across the session and the scored prefixes.
    """

    images: Mapping[str, Mapping[str, Any]]


def census_session_plan_view(plan: SealedPlan) -> CensusImageRegistryView:
    """Adapt this unit's sealed plan to the census session seam, by reference."""

    registry = getattr(plan.inputs, "images_by_id", None)
    if not isinstance(registry, Mapping) or not registry:
        _fail(
            "the sealed plan exposes no image registry under inputs.images_by_id; the "
            "production session seam cannot resolve an image without it"
        )
    return CensusImageRegistryView(images=registry)


def _open_backend(args: argparse.Namespace, plan: SealedPlan, image_id: str) -> Any:
    """Open exactly one immutable session, on the predecessor's own seam."""

    shard = _census_shard()
    if args.backend == "fake":
        import contextlib

        @contextlib.contextmanager
        def _fake():
            yield shard.FakeCensusBackend(seed=f"crossing:{image_id}")

        return _fake()
    if args.infer_config is None:
        _fail("live scoring requires --infer-config")
    spec = shard.build_hf_session_spec(
        census_session_plan_view(plan),
        image_id,
        infer_config=Path(args.infer_config).expanduser().resolve(strict=True),
    )
    return shard.open_hf_backend(spec)


#: Where a sealed runtime identity may live inside a predecessor artifact.  The
#: census shard receipt nests it under ``backend_identity``; a hand-rolled
#: identity file may carry the same keys at the top level.  Reading the sealed
#: receipt directly is preferred, because it keeps the identity's own lineage
#: (path and file digest) instead of an ad-hoc copy with none.
RUNTIME_IDENTITY_NEST_KEYS: tuple[str, ...] = ("backend_identity", "runtime_identity")
RUNTIME_IDENTITY_COMPARED_KEYS: tuple[str, ...] = ("model_identity", "tokenizer_identity")

#: Matmul-precision fields that genuinely change fp32 results and must match the
#: predecessor capture exactly.
MATMUL_PRECISION_BINDING_FIELDS: tuple[str, ...] = (
    "float32_matmul_precision",
    "cuda_matmul_allow_tf32",
    "torch_version",
)
#: Compared and *reported*, never silently equated.  ``pin_fp32_parity_flags``
#: disables the cuDNN TF32 path that the predecessor census left enabled, so
#: this field is expected to differ in the strictly-more-precise direction; it
#: is named as a bounded lineage caveat instead of being ignored or used to
#: block a launch that is numerically tighter than its predecessor.
MATMUL_PRECISION_REPORTED_FIELDS: tuple[str, ...] = ("cudnn_allow_tf32",)

#: The predecessor census receipt seals its matmul state one level *beside* the
#: identity (``numerics.matmul_precision``), not inside ``backend_identity``.
FROZEN_NUMERICS_KEY = "numerics"
FROZEN_MATMUL_PRECISION_KEY = "matmul_precision"

#: The predecessor census sealed only the infer config's *path*, never its
#: content hash.  Re-hashing that path today therefore proves what the file
#: holds now, not what the predecessor executed.  The comparison is still worth
#: making -- a drifted config is a real reason to stop -- but it is recorded as
#: this bounded caveat and never reported as a historical content comparison.
INFER_CONFIG_LINEAGE_CAVEAT = (
    "the predecessor receipt seals the infer config path only, with no content hash; the "
    "compared digest was recomputed from that path now, so this proves the file's current "
    "content, not the content the predecessor executed"
)


def extract_frozen_matmul_precision(payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """The predecessor's sealed matmul state, wherever the artifact put it."""

    for container in (
        payload.get(FROZEN_NUMERICS_KEY),
        *(payload.get(key) for key in RUNTIME_IDENTITY_NEST_KEYS),
        payload,
    ):
        if not isinstance(container, Mapping):
            continue
        nested = container.get(FROZEN_MATMUL_PRECISION_KEY)
        if isinstance(nested, Mapping):
            return nested
    return None


def compare_matmul_precision(
    frozen: Mapping[str, Any] | None, observed: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Bind the fields that move fp32 results; report the one that may differ.

    ``float32_matmul_precision``, the CUDA TF32 matmul switch and the torch
    revision all change float32 matmul output, so a capture that does not
    reproduce them is not joinable to the predecessor and fails closed.
    ``cudnn_allow_tf32`` is expected to differ -- ``pin_fp32_parity_flags``
    disables a path the predecessor left on -- so it is reported as a named
    difference in the strictly-more-precise direction rather than ignored.
    """

    if not isinstance(frozen, Mapping):
        return {
            "compared": False,
            "reason": "the frozen runtime identity seals no numerics.matmul_precision block",
        }
    if not isinstance(observed, Mapping):
        _fail(
            "the frozen runtime identity seals a numerics.matmul_precision block but this "
            "session observed none; the executed float32 matmul state is unproven"
        )
    drifted = sorted(
        field
        for field in MATMUL_PRECISION_BINDING_FIELDS
        if field in frozen and frozen[field] != observed.get(field)
    )
    if drifted:
        _fail(
            "the executed float32 matmul state differs from the predecessor census in "
            f"binding field(s) {drifted!r} (frozen="
            f"{ {field: frozen.get(field) for field in drifted}!r}, observed="
            f"{ {field: observed.get(field) for field in drifted}!r}); TF32 and torch "
            "revision change float32 matmul results, so this capture would not be joinable"
        )
    return {
        "compared": True,
        "binding_fields": list(MATMUL_PRECISION_BINDING_FIELDS),
        "binding_fields_agree": True,
        "reported_fields": {
            field: {"frozen": frozen.get(field), "observed": observed.get(field)}
            for field in MATMUL_PRECISION_REPORTED_FIELDS
        },
        "reported_field_differences": sorted(
            field
            for field in MATMUL_PRECISION_REPORTED_FIELDS
            if field in frozen and frozen[field] != observed.get(field)
        ),
        "reported_field_note": (
            "cuDNN TF32 is disabled here by pin_fp32_parity_flags; a difference is the "
            "strictly-more-precise direction and is reported, never equated"
        ),
    }


def extract_frozen_runtime_identity(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Find the sealed identity in a predecessor receipt or a standalone file."""

    for key in RUNTIME_IDENTITY_NEST_KEYS:
        nested = payload.get(key)
        if isinstance(nested, Mapping) and all(
            nested.get(name) is not None for name in RUNTIME_IDENTITY_COMPARED_KEYS
        ):
            return nested
    if all(payload.get(name) is not None for name in RUNTIME_IDENTITY_COMPARED_KEYS):
        return payload
    _fail(
        "the runtime identity file carries no model_identity/tokenizer_identity, at the top "
        f"level or nested under any of {list(RUNTIME_IDENTITY_NEST_KEYS)!r}; point "
        "--runtime-identity at a sealed predecessor shard receipt"
    )


def validate_runtime_identity(
    observed: Mapping[str, Any],
    frozen_path: Path | None,
    *,
    numerics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind the executed model/tokenizer/config identity to the frozen declaration.

    ``frozen_path`` may be a sealed predecessor shard receipt: the identity is
    read from its nested ``backend_identity`` and the receipt's own path and
    content hash are recorded, so the binding keeps its lineage instead of
    depending on a hand-copied file with none.
    """

    if frozen_path is None:
        return {
            **dict(observed),
            "frozen_identity_path": None,
            "frozen_identity_bound": False,
        }
    path = Path(frozen_path)
    frozen_document = _read_json(path, "runtime identity source")
    frozen = extract_frozen_runtime_identity(frozen_document)
    for key in RUNTIME_IDENTITY_COMPARED_KEYS:
        if sha256_json(frozen.get(key)) != sha256_json(observed.get(key)):
            _fail(
                f"the opened session's {key} differs from the frozen runtime identity sealed "
                f"at {path}; this capture would not be joinable to the predecessor census"
            )
    frozen_stratum = frozen.get("repetition_penalty_stratum")
    if frozen_stratum is not None and float(frozen_stratum) != float(
        NATIVE_REPETITION_PENALTY_STRATUM
    ):
        _fail(
            f"the frozen runtime identity was captured at repetition-penalty stratum "
            f"{frozen_stratum!r}, but this unit scores only "
            f"{NATIVE_REPETITION_PENALTY_STRATUM}"
        )
    # TF32 and the torch revision move float32 matmul results, so the sealed
    # numerics are compared before any cached reading is trusted.
    matmul_comparison = compare_matmul_precision(
        extract_frozen_matmul_precision(frozen_document),
        (numerics or {}).get("matmul_precision"),
    )
    # The same checkpoint under a different infer config is a different runtime.
    frozen_config = frozen.get("infer_config")
    executed_config = (numerics or {}).get("infer_config_path")
    frozen_config_sha256: str | None = None
    if frozen_config is not None and executed_config is not None:
        frozen_config_path = Path(str(frozen_config)).expanduser()
        if not frozen_config_path.is_file():
            _fail(
                f"the frozen runtime identity names infer config {frozen_config_path}, which "
                "is not readable here; its content cannot be compared"
            )
        frozen_config_sha256 = sha256_file(frozen_config_path)
        if frozen_config_sha256 != (numerics or {}).get("infer_config_sha256"):
            _fail(
                f"the executed infer config {executed_config} differs in content from the "
                f"file the frozen runtime identity names ({frozen_config_path}) as it stands "
                f"now; {INFER_CONFIG_LINEAGE_CAVEAT}, and a drifted config is not scored "
                "through on an unproven precision, attention or repetition-penalty semantics"
            )
    return {
        **dict(observed),
        "frozen_identity_path": str(path),
        "frozen_identity_bound": True,
        "frozen_identity_source_sha256": sha256_file(path),
        "frozen_identity_infer_config": frozen_config,
        "frozen_identity_infer_config_sha256": frozen_config_sha256,
        "frozen_identity_infer_config_sha256_provenance": (
            None
            if frozen_config_sha256 is None
            else "recomputed_now_from_the_declared_path_not_sealed_by_the_predecessor"
        ),
        "frozen_identity_infer_config_caveat": (
            None if frozen_config is None else INFER_CONFIG_LINEAGE_CAVEAT
        ),
        "frozen_identity_matmul_precision": matmul_comparison,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-dir", required=True, type=Path, help="sealed CPU plan directory")
    parser.add_argument("--prevalence-run-root", type=Path, default=None)
    parser.add_argument("--census-run-root", type=Path, default=None)
    parser.add_argument("--infer-config", type=Path, default=None)
    parser.add_argument(
        "--runtime-identity",
        type=Path,
        default=None,
        help="frozen model/tokenizer identity JSON; required for --backend hf",
    )
    parser.add_argument("--output-dir", required=True, type=Path, help="explicit output shard")
    parser.add_argument("--shard-id", required=True, help="explicit shard identity")
    parser.add_argument(
        "--mode",
        choices=(MODE_SMOKE, MODE_CAPTURE),
        default=MODE_CAPTURE,
        help=(
            "smoke: prove cached execution and batching on one image carrying the whole "
            "required role matrix and seal an admission receipt. "
            "capture: score one image's owners under a sealed admission."
        ),
    )
    parser.add_argument(
        "--image-id",
        required=True,
        help="the one image this run opens a session for; nothing else may be scored on it",
    )
    parser.add_argument(
        "--admission-receipt",
        type=Path,
        default=None,
        help=f"the smoke run's {ADMISSION_NAME}; required for --mode capture",
    )
    parser.add_argument(
        "--gt-owner-id",
        action="append",
        default=None,
        help="restrict this shard to specific owners of --image-id (repeatable)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="candidate lane width to prove in --mode smoke; ignored in --mode capture",
    )
    parser.add_argument("--backend", choices=("hf", "fake"), default="hf")
    parser.add_argument(
        "--validate-plan-only",
        action="store_true",
        help="re-prove every plan/predecessor digest and exit without loading a model",
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    started_at = time.time()
    plan = load_sealed_plan(
        args.plan_dir,
        prevalence_run_root=args.prevalence_run_root,
        census_run_root=args.census_run_root,
    )
    if args.validate_plan_only:
        for row in plan.request_rows:
            validate_request_row(plan, row)
        return {
            "status": "plan_validated",
            "plan_dir": str(plan.plan_dir),
            "manifest_content_sha256": plan.manifest.get("manifest_content_sha256"),
            "request_count": len(plan.request_rows),
            "cohort_count": len(plan.cohort_rows),
            "control_count": len(plan.control_rows),
            "smoke_eligible_image_ids": images_with_all_required_smoke_roles(
                plan.cohort_rows
            ),
            "loads_model": False,
        }

    if args.backend == "hf" and args.runtime_identity is None:
        _fail("--runtime-identity is required for --backend hf")
    if args.batch_size < 1:
        _fail("--batch-size must be at least one")

    image_id = str(args.image_id)
    selected: list[str] = []
    if args.mode == MODE_SMOKE:
        # Fails closed with the eligible images named when this one cannot carry
        # the whole matrix, which is the common case.
        select_smoke_owners(plan.cohort_rows, image_id=image_id)
    else:
        if args.admission_receipt is None:
            _fail(
                f"--mode {MODE_CAPTURE} requires --admission-receipt pointing at the smoke "
                f"run's {ADMISSION_NAME}; cached execution and batching are never assumed"
            )
        selected = [
            str(row["gt_owner_id"])
            for row in (*plan.cohort_rows, *plan.control_rows)
            if str(row["image_id"]) == image_id
        ]
        if args.gt_owner_id:
            requested = {str(value) for value in args.gt_owner_id}
            unknown = sorted(requested - set(selected))
            if unknown:
                _fail(f"owner(s) {unknown!r} are not registered for image {image_id!r}")
            selected = [owner_id for owner_id in selected if owner_id in requested]
        if not selected:
            _fail(f"image {image_id!r} has no registered owner to capture")
        selected = sorted(set(selected))

    if args.backend == "hf":
        _basin().pin_fp32_parity_flags()

    with _open_backend(args, plan, image_id) as backend:
        numerics = build_runtime_numerics(backend, infer_config=args.infer_config)
        runtime_identity = {
            **validate_runtime_identity(
                backend.identity, args.runtime_identity, numerics=numerics
            ),
            "numerics": numerics,
            "source_identity": source_identity(),
            "session_image_id": image_id,
        }
        if args.mode == MODE_SMOKE:
            result = run_smoke_shard(
                plan=plan,
                backend=backend,
                shard_id=str(args.shard_id),
                image_id=image_id,
                batch_size=int(args.batch_size),
                runtime_identity=runtime_identity,
            )
        else:
            admission = validate_admission_receipt(
                _read_json(Path(args.admission_receipt), "admission receipt"),
                plan=plan,
                runtime_identity=runtime_identity,
            )
            result = run_capture_shard(
                plan=plan,
                backend=backend,
                shard_id=str(args.shard_id),
                session_image_id=image_id,
                owner_ids=selected,
                admission=admission,
                runtime_identity=runtime_identity,
            )

    publish = _publish(Path(args.output_dir), shard_output_files(result))
    status = (
        "quarantined"
        if result.quarantine is not None
        else ("admitted" if result.admission is not None else "captured")
    )
    return {
        "status": status,
        "mode": str(args.mode),
        "shard_id": str(args.shard_id),
        "image_id": image_id,
        "owner_count": len(result.owner_records),
        "score_row_count": len(result.score_rows),
        "receipt_content_sha256": result.receipt.get("receipt_content_sha256"),
        "admission_content_sha256": (
            None if result.admission is None else result.admission["admission_content_sha256"]
        ),
        # Reported, never sealed: a wall-clock duration would make otherwise
        # identical re-captures publish differing bytes.
        "elapsed_seconds": round(time.time() - started_at, 3),
        "publish": publish,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run(args)
    except (CrossingBoundaryContractError, plan_builder.PlanContractError) as exc:
        print(f"FAIL-CLOSED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
