#!/usr/bin/env python3
"""CPU-only fixed-budget reanalysis for the sorted false-negative mechanism
decomposition unit.

Maps a proposed fixed-budget candidate lattice onto existing predecessor
score rows by exact one-to-one candidate identity: box tokens
(``coord_token_ids``, or the ``box_tokens`` test alias), candidate ID (a
fixed-budget candidate may bind a ``predecessor_candidate_id`` explicitly to
reuse a predecessor-scored row whose own ID differs from the successor's;
otherwise the candidate's own ``candidate_id`` is the match key), raw-row
identity, and source digest. ``raw_row_identity`` and ``source_digest`` are
both optional: the real predecessor complete-box score row schema never
carries either, and a missing value is recorded as ``None`` -- never
fabricated. 1:1 matching still requires every identity element to agree
exactly, including ``None`` matching only ``None``. Every unmatched
candidate is reported for future rescoring; there is no nearest-match
fallback anywhere in this module. Score values accept either the real
predecessor schema (``raw_model_logprob.complete_box_logprob_sum``) or the
flat ``raw_score`` test alias.

A predecessor artifact root may merge multiple raw request kinds into one
``landscape-scores.jsonl`` (e.g. conditional single-coordinate
``raw_bin_scan``/``dense_scan`` rows alongside complete free-tree-box rows).
``load_score_index`` classifies every row against a *closed* ``request_kind``
vocabulary (see ``_classify_score_row``): eligibility is never inferred from
row shape alone. Only an explicit ``request_kind == "complete_box"`` row --
one that itself carries a complete box identity and a complete-box score --
is ever admitted into the 1:1 reuse index (or the narrow, explicitly
unlabeled ``box_tokens``/``raw_score`` test-alias exception, which never
matches a row that also carries real complete-box fields). Only an explicit
``request_kind == "dense_scan"`` row is filtered, and only when it does
*not* also carry complete-box-shaped fields -- a row whose declared kind
contradicts its shape is rejected, not silently admitted or filtered. Every
other ``request_kind`` value is unrecognized and rejected. Filtered rows are
always counted, never silently dropped (see the returned/report
``*_filter_report``).

The budget ladder is a frozen, non-uniform rung schedule (Fable-reviewed,
recorded 2026-08-02):

  * ``L0`` core: 24-40 target candidates (equal-count decoys);
  * ``L1``: exactly 256 target candidates plus 256 family-mirrored decoys,
    with at least 64 target-strict-region candidates;
  * ``L2``: exactly 1024 target candidates plus 1024 decoys, admitted only
    after a same-``matched_control_group`` ``L1`` dense-reference-validation
    failure or a prospectively declared target near-miss. Escalation is
    stratum-local: an ``L1`` failure in one ``matched_control_group`` never
    authorizes ``L2`` for an unrelated group, and a target-only null result
    never self-authorizes ``L2``.

"Family-mirrored" decoys means the decoy population's ``family_id`` values
reproduce the *exact same multiset of counts* as the target population's
``family_id`` values (not merely overlapping description strings).

``scalar_smoke`` is a distinct, size-unconstrained rung reserved for
owner-contexts whose ``control_kind`` is ``strict_rescue`` (no dense
predecessor reference exists for that control kind on this panel, e.g.
``gt:7511:17``); reuse of old/dense score rows is refused for it.

Dense-reference validation (``--dense-candidates``/``--dense-score-rows``)
replaces the earlier, weaker "control_preservation" same-subset-peak check.
For owner-contexts with ``control_kind`` in ``strict_positive``,
``loose_only``, ``b2_pair_before``, or ``b2_pair_after``, each fixed-budget
rung is validated against the *complete* dense reference population per IoU
threshold: whether a target-strict peak exists within a frozen numeric
tolerance (default 0.5 nats) of the dense peak, whether the localized-rank
*side* (at or below a frozen upper quantile, default 0.90) agrees, and --
for a ``b2_pair_before``/``b2_pair_after`` pair sharing one
``matched_control_group`` -- whether the paired-change (selective-decline)
sign agrees between the dense reference and the fixed-budget subset. If no
dense reference is supplied for a control owner-context that needs one, its
status is the explicit ``no_dense_reference`` state; it is never silently
reported as preserved.

The "usable" and rank-side checks implemented here are a *local* proxy
computed from each owner-context's own dense/subset decoy pool (frozen
default quantile 0.90 as the rank-side boundary, positive background
prominence). They are not the cross-stratum, multi-context quantile-pooled
calibration described in unit.md's full protocol, which needs a historical
corpus of prominence values across many owner-contexts of the same
description/size/crowding stratum; that corpus is not an input this CPU
contract ingests, and no such calibration is claimed here.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "sorted-fn-fixed-budget-reanalysis.v3"
UNIT_ID = "2026-08-02-sorted-false-negative-mechanism-decomposition"

# The candidate-row schema this reanalysis consumer requires. A stale v1 row
# (missing the neighborhood/reference/parent-binding fields introduced in
# v2 -- candidate_neighborhood_member/_id, mechanism_decision_rules_sha256,
# and reference-population support) fails fast rather than being silently
# read as if it had them.
CANDIDATE_ROW_SCHEMA_VERSION = "sorted-fn-successor-fixed-budget-candidates.v2"
_STALE_CANDIDATE_ROW_SCHEMA_VERSIONS = frozenset({"sorted-fn-successor-fixed-budget-candidates.v1"})

DEFAULT_IOU_THRESHOLDS: tuple[float, ...] = (0.4, 0.5, 0.6)
DEFAULT_PEAK_TOLERANCE_NATS = 0.5
DEFAULT_RANK_SIDE_THRESHOLD = 0.90

_VALID_REGIONS = frozenset({"target_strict", "target_halo", "other_owner", "background"})
# "reference" (near_other_micro, region="other_owner") is accepted but
# explicitly excluded from the target/decoy equal-count pairing and from the
# localized_rank target-union pool; it only ever feeds other_owner_margin
# (see _population_statistics: pooled_scored/background_scored are scoped to
# ("target", "decoy") only, while other_owner_scored is scoped by region, not
# population, so a reference row is exactly what populates it).
_VALID_POPULATIONS = frozenset({"target", "decoy", "reference"})
_REFERENCE_COUNT_BY_RUNG = {"L0": 7, "L1": 65, "scalar_smoke": 7}
_VALID_RUNGS = frozenset({"L0", "L1", "L2", "scalar_smoke"})
_NO_DENSE_REFERENCE_CONTROL_KINDS = frozenset({"strict_rescue"})
_NO_DENSE_REFERENCE_ALLOWED_RUNGS = frozenset({"scalar_smoke", "L1"})
_DENSE_REFERENCE_CONTROL_KINDS = frozenset(
    {"strict_positive", "loose_only", "b2_pair_before", "b2_pair_after"}
)
_B2_BEFORE_KIND = "b2_pair_before"
_B2_AFTER_KIND = "b2_pair_after"

# Fields that would let the recorded score leak back into candidate
# generation or region/threshold selection. Any candidate or generator row
# carrying one of these keys is rejected before scoring.
_FORBIDDEN_LEAKAGE_FIELDS = frozenset(
    {
        "raw_score",
        "raw_model_logprob",
        "old_score",
        "reused_score",
        "target_peak",
        "background_prominence",
        "other_owner_margin",
        "localized_rank",
        "target_bank_score",
    }
)

IdentityKey = tuple[str, tuple[int, ...], str | None, str | None]


class ReanalysisError(ValueError):
    """Raised before output when the proposed reanalysis is not admissible."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReanalysisError(f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ReanalysisError(f"{label} must be an array")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ReanalysisError(f"{label} must be a non-empty trimmed string")
    return value


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReanalysisError(f"{label} must be a number")
    return float(value)


def _resolved_file(path: str | Path, label: str) -> Path:
    try:
        resolved = Path(path).expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise ReanalysisError(f"{label} does not exist") from exc
    if not resolved.is_file():
        raise ReanalysisError(f"{label} must be a regular file")
    return resolved


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ReanalysisError(
                    f"{label} line {line_number} is not valid JSON"
                ) from exc
            rows.append(dict(_mapping(record, f"{label} line {line_number}")))
    return rows


def _reject_leakage_fields(row: Mapping[str, Any], *, label: str) -> None:
    leaked = _FORBIDDEN_LEAKAGE_FIELDS & set(row)
    if leaked:
        raise ReanalysisError(
            f"{label} contains score-derived field(s) {sorted(leaked)}; the "
            "candidate generator and region/threshold selection must be "
            "frozen before any target row is read"
        )


# --------------------------------------------------------------------------
# Real-predecessor-schema-compatible box identity and score extraction
# --------------------------------------------------------------------------


def _box_identity_tokens(row: Mapping[str, Any], *, label: str) -> tuple[int, ...]:
    if "coord_token_ids" in row:
        source, field = row["coord_token_ids"], "coord_token_ids"
    elif "box_tokens" in row:
        source, field = row["box_tokens"], "box_tokens"
    else:
        raise ReanalysisError(f"{label} must declare coord_token_ids or box_tokens")
    return tuple(int(token) for token in _sequence(source, f"{label}.{field}"))


def _raw_row_identity(row: Mapping[str, Any], *, label: str) -> str | None:
    value = row.get("raw_row_identity")
    if value is None:
        return None
    return _string(value, f"{label}.raw_row_identity")


def _optional_source_digest(row: Mapping[str, Any], *, label: str) -> str | None:
    """``source_digest`` is optional, exactly like ``raw_row_identity``: the
    real predecessor complete-box score row schema (``coord_token_ids`` +
    ``raw_model_logprob.complete_box_logprob_sum``) never carries one. A
    missing digest is recorded as ``None`` -- never fabricated -- and 1:1
    matching still requires every identity element, including this one, to
    agree exactly (``None`` matches only ``None``, never a real digest)."""

    value = row.get("source_digest")
    if value is None:
        return None
    return _string(value, f"{label}.source_digest")


def _match_identity_key(row: Mapping[str, Any], *, label: str) -> IdentityKey:
    """Identity key used to look a *candidate* up in an old/dense score index."""

    predecessor_id = row.get("predecessor_candidate_id")
    if predecessor_id is not None:
        match_id = _string(predecessor_id, f"{label}.predecessor_candidate_id")
    else:
        match_id = _string(row.get("candidate_id"), f"{label}.candidate_id")
    return (
        match_id,
        _box_identity_tokens(row, label=label),
        _raw_row_identity(row, label=label),
        _optional_source_digest(row, label=label),
    )


def _own_identity_key(row: Mapping[str, Any], *, label: str) -> IdentityKey:
    """Identity key an old/dense score row is indexed under (its own candidate_id)."""

    return (
        _string(row.get("candidate_id"), f"{label}.candidate_id"),
        _box_identity_tokens(row, label=label),
        _raw_row_identity(row, label=label),
        _optional_source_digest(row, label=label),
    )


def _score_value(row: Mapping[str, Any], *, label: str) -> float:
    if "raw_model_logprob" in row:
        block = _mapping(row["raw_model_logprob"], f"{label}.raw_model_logprob")
        return _number(
            block.get("complete_box_logprob_sum"),
            f"{label}.raw_model_logprob.complete_box_logprob_sum",
        )
    if "raw_score" in row:
        return _number(row["raw_score"], f"{label}.raw_score")
    raise ReanalysisError(
        f"{label} must declare raw_model_logprob.complete_box_logprob_sum or raw_score"
    )


# --------------------------------------------------------------------------
# Candidate and score-row loading
# --------------------------------------------------------------------------


def _parse_candidate_row(row: Mapping[str, Any], *, label: str, require_rung: bool) -> dict[str, Any]:
    _reject_leakage_fields(row, label=label)
    if require_rung:
        schema_version = row.get("schema_version")
        if schema_version in _STALE_CANDIDATE_ROW_SCHEMA_VERSIONS:
            raise ReanalysisError(
                f"{label}.schema_version {schema_version!r} is a stale v1 fixed-budget "
                "candidate schema (missing neighborhood/reference/parent-binding "
                f"fields); {CANDIDATE_ROW_SCHEMA_VERSION!r} is required"
            )
        if schema_version != CANDIDATE_ROW_SCHEMA_VERSION:
            raise ReanalysisError(
                f"{label}.schema_version must be {CANDIDATE_ROW_SCHEMA_VERSION!r}, "
                f"got {schema_version!r}"
            )
    candidate_id = _string(row.get("candidate_id"), f"{label}.candidate_id")
    box_identity = _box_identity_tokens(row, label=label)
    raw_row_identity = _raw_row_identity(row, label=label)
    source_digest = _optional_source_digest(row, label=label)
    predecessor_candidate_id = row.get("predecessor_candidate_id")
    if predecessor_candidate_id is not None:
        predecessor_candidate_id = _string(
            predecessor_candidate_id, f"{label}.predecessor_candidate_id"
        )
    owner_context_id = _string(row.get("owner_context_id"), f"{label}.owner_context_id")
    if require_rung:
        rung = _string(row.get("rung"), f"{label}.rung")
        if rung not in _VALID_RUNGS:
            raise ReanalysisError(f"{label}.rung {rung!r} is not one of {sorted(_VALID_RUNGS)}")
    else:
        rung = "dense"
    region = _string(row.get("region"), f"{label}.region")
    if region not in _VALID_REGIONS:
        raise ReanalysisError(f"{label}.region {region!r} is not one of {sorted(_VALID_REGIONS)}")
    population = _string(row.get("population"), f"{label}.population")
    if population not in _VALID_POPULATIONS:
        raise ReanalysisError(
            f"{label}.population {population!r} is not one of {sorted(_VALID_POPULATIONS)}"
        )
    is_control = bool(row.get("is_control", False))
    control_kind = row.get("control_kind")
    if control_kind is not None:
        control_kind = _string(control_kind, f"{label}.control_kind")
    matched_control_group = _string(
        row.get("matched_control_group"), f"{label}.matched_control_group"
    )
    family_id = _string(row.get("family_id"), f"{label}.family_id")
    iou_to_target = _number(row.get("iou_to_target", 0.0), f"{label}.iou_to_target")
    return {
        "candidate_id": candidate_id,
        "predecessor_candidate_id": predecessor_candidate_id,
        "box_identity": box_identity,
        "raw_row_identity": raw_row_identity,
        "source_digest": source_digest,
        "owner_context_id": owner_context_id,
        "rung": rung,
        "region": region,
        "population": population,
        "is_control": is_control,
        "control_kind": control_kind,
        "matched_control_group": matched_control_group,
        "family_id": family_id,
        "iou_to_target": iou_to_target,
    }


def load_candidates(path: Path) -> list[dict[str, Any]]:
    raw_rows = _read_jsonl(path, "fixed-budget candidates")
    candidates: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, row in enumerate(raw_rows):
        label = f"candidate row {index}"
        candidate = _parse_candidate_row(row, label=label, require_rung=True)
        if candidate["candidate_id"] in seen_ids:
            raise ReanalysisError(f"duplicate candidate_id {candidate['candidate_id']!r}")
        seen_ids.add(candidate["candidate_id"])
        candidates.append(candidate)
    return candidates


def load_dense_candidates(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    raw_rows = _read_jsonl(path, "dense candidates")
    candidates: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, row in enumerate(raw_rows):
        label = f"dense candidate row {index}"
        candidate = _parse_candidate_row(row, label=label, require_rung=False)
        if candidate["candidate_id"] in seen_ids:
            raise ReanalysisError(f"duplicate dense candidate_id {candidate['candidate_id']!r}")
        seen_ids.add(candidate["candidate_id"])
        candidates.append(candidate)
    return candidates


# A predecessor artifact root may merge multiple raw request kinds into one
# landscape-scores.jsonl (e.g. conditional single-coordinate raw_bin_scan
# rows alongside complete free-tree-box rows). Eligibility for 1:1 reuse is
# never inferred from row shape: it is driven by a *closed* request_kind
# vocabulary, and a row's declared kind must not contradict its shape.
#
#   * request_kind == "complete_box": the only scored kind. Must carry a
#     complete box identity and score (real predecessor schema, or the
#     box_tokens/raw_score test alias); otherwise rejected.
#   * request_kind == "dense_scan": the only supported filtered alternate
#     (a conditional raw_bin_scan row from the predecessor). Must NOT carry
#     complete-box-shaped fields; a row labeled dense_scan but shaped like a
#     complete box is a contradictory payload and is rejected, never
#     silently admitted.
#   * any other request_kind value: unrecognized, rejected.
#   * request_kind absent entirely: rejected, *except* the narrow, explicit
#     test-alias exception below -- real predecessor rows always declare
#     request_kind, so an unlabeled row is never admitted by shape alone.
_COMPLETE_BOX_REQUEST_KIND = "complete_box"
_DENSE_SCAN_REQUEST_KIND = "dense_scan"
_RECOGNIZED_ALTERNATE_REQUEST_KINDS = frozenset({_DENSE_SCAN_REQUEST_KIND})


def _has_real_complete_box_shape(row: Mapping[str, Any]) -> bool:
    return "coord_token_ids" in row and (
        isinstance(row.get("raw_model_logprob"), Mapping)
        and "complete_box_logprob_sum" in row["raw_model_logprob"]
    )


def _has_test_alias_complete_box_shape(row: Mapping[str, Any]) -> bool:
    return "box_tokens" in row and "raw_score" in row


def _has_any_complete_box_shape(row: Mapping[str, Any]) -> bool:
    return _has_real_complete_box_shape(row) or _has_test_alias_complete_box_shape(row)


def _classify_score_row(row: Mapping[str, Any], *, row_label: str) -> str:
    """Return ``"admit"`` or ``"filter"`` under the closed request_kind
    vocabulary above, or raise ``ReanalysisError``. Never infers eligibility
    from shape alone, and never lets an unrecognized or shape-contradicting
    request_kind through."""

    request_kind = row.get("request_kind")

    if request_kind is None:
        # Narrow, explicit exception: the unlabeled test-alias schema
        # (box_tokens + raw_score, with no real coord_token_ids/
        # raw_model_logprob at all). Anything else with no request_kind --
        # including an unlabeled row with real complete-box shape -- is
        # rejected rather than admitted by inference.
        if _has_test_alias_complete_box_shape(row) and not _has_real_complete_box_shape(row):
            return "admit"
        raise ReanalysisError(
            f"{row_label} has no request_kind and is not the narrow unlabeled "
            "test-alias complete-box shape (box_tokens + raw_score, with no "
            "coord_token_ids/raw_model_logprob); refusing to infer eligibility "
            "from shape alone"
        )

    if request_kind == _COMPLETE_BOX_REQUEST_KIND:
        if _has_any_complete_box_shape(row):
            return "admit"
        raise ReanalysisError(
            f"{row_label} declares request_kind={_COMPLETE_BOX_REQUEST_KIND!r} but "
            "lacks the required complete-box shape (coord_token_ids + "
            "raw_model_logprob.complete_box_logprob_sum, or the box_tokens + "
            "raw_score test alias)"
        )

    if request_kind in _RECOGNIZED_ALTERNATE_REQUEST_KINDS:
        if _has_any_complete_box_shape(row):
            raise ReanalysisError(
                f"{row_label} declares request_kind={request_kind!r} but carries "
                "complete-box-shaped fields; a row contradicting its declared "
                "request_kind is rejected, never silently admitted or filtered"
            )
        return "filter"

    raise ReanalysisError(
        f"{row_label} has unrecognized request_kind {request_kind!r}; only "
        f"{_COMPLETE_BOX_REQUEST_KIND!r} and "
        f"{sorted(_RECOGNIZED_ALTERNATE_REQUEST_KINDS)} are supported"
    )


def load_score_index(
    path: Path | None, *, label: str
) -> tuple[dict[IdentityKey, float], dict[str, Any]]:
    """Return ``(index, filter_report)``. ``filter_report`` records how many
    rows were read, how many were explicit complete-box score rows admitted
    into the index, and how many were filtered as the recognized non-
    complete-box ``dense_scan`` request kind (never silently, and never
    counted as a match)."""

    if path is None:
        return {}, {
            "total_rows": 0,
            "complete_box_rows": 0,
            "filtered_non_complete_box_rows": 0,
            "filtered_request_kinds": {},
        }
    raw_rows = _read_jsonl(path, label)
    index: dict[IdentityKey, float] = {}
    filtered_kinds: dict[str, int] = {}
    admitted = 0
    for row_index, row in enumerate(raw_rows):
        row_label = f"{label} row {row_index}"
        disposition = _classify_score_row(row, row_label=row_label)
        if disposition == "filter":
            request_kind = row["request_kind"]
            filtered_kinds[request_kind] = filtered_kinds.get(request_kind, 0) + 1
            continue
        admitted += 1
        key = _own_identity_key(row, label=row_label)
        score = _score_value(row, label=row_label)
        if key in index:
            raise ReanalysisError(
                f"{label} have an ambiguous duplicate identity key {key!r}; 1:1 reuse "
                "requires a unique candidate identity"
            )
        index[key] = score
    filter_report = {
        "total_rows": len(raw_rows),
        "complete_box_rows": admitted,
        "filtered_non_complete_box_rows": sum(filtered_kinds.values()),
        "filtered_request_kinds": dict(sorted(filtered_kinds.items())),
    }
    return index, filter_report


# --------------------------------------------------------------------------
# Rung-shape validation
# --------------------------------------------------------------------------


def _validate_reference_count(
    *, owner_context_id: str, rung: str, candidates: Sequence[Mapping[str, Any]]
) -> None:
    """``reference`` (near_other_micro) rows are accepted but never part of
    the target/decoy equal-count pairing; when present, their count must
    match the frozen quota for this rung (0 is always allowed -- no
    same-description non-overlapping owner may exist for this owner-context)."""

    reference_count = sum(1 for row in candidates if row["population"] == "reference")
    if reference_count == 0:
        return
    expected = _REFERENCE_COUNT_BY_RUNG.get(rung)
    if expected is None or reference_count != expected:
        raise ReanalysisError(
            f"owner-context {owner_context_id} rung {rung} has {reference_count} "
            f"reference candidates; expected 0 or exactly {expected} for this rung"
        )


def _validate_rung_shape(
    *, owner_context_id: str, rung: str, control_kind: str | None, candidates: Sequence[Mapping[str, Any]]
) -> None:
    target_rows = [row for row in candidates if row["population"] == "target"]
    decoy_rows = [row for row in candidates if row["population"] == "decoy"]
    if len(target_rows) != len(decoy_rows):
        raise ReanalysisError(
            f"owner-context {owner_context_id} rung {rung} has unequal target "
            f"({len(target_rows)}) and decoy ({len(decoy_rows)}) candidate counts"
        )
    target_family_counts = Counter(row["family_id"] for row in target_rows)
    decoy_family_counts = Counter(row["family_id"] for row in decoy_rows)
    if target_family_counts != decoy_family_counts:
        raise ReanalysisError(
            f"owner-context {owner_context_id} rung {rung} decoy population is not "
            f"family-mirrored: target family_id counts {dict(sorted(target_family_counts.items()))} "
            f"do not equal decoy family_id counts {dict(sorted(decoy_family_counts.items()))}"
        )
    _validate_reference_count(owner_context_id=owner_context_id, rung=rung, candidates=candidates)
    target_count = len(target_rows)
    if control_kind in _NO_DENSE_REFERENCE_CONTROL_KINDS:
        # strict_rescue (e.g. gt:7511:17) emits both an unconstrained,
        # positive-direction-only scalar_smoke rung and an unconditionally
        # frozen L1 rung (scored on marginal/negative evidence before any
        # stop-rule-4 judgment); no dense predecessor reference exists for
        # either, but L1 still has to meet the normal L1 target-count/
        # strict-region shape.
        if rung not in _NO_DENSE_REFERENCE_ALLOWED_RUNGS:
            raise ReanalysisError(
                f"owner-context {owner_context_id} has control_kind {control_kind!r}, "
                "which has no dense predecessor reference on this panel; it must use "
                f"rung 'scalar_smoke' or 'L1', not {rung!r}"
            )
        if rung == "scalar_smoke":
            if target_count < 1:
                raise ReanalysisError(
                    f"owner-context {owner_context_id} scalar_smoke rung has no target candidates"
                )
            return
        # rung == "L1": fall through to the normal L1 shape check below.
    elif rung == "scalar_smoke":
        raise ReanalysisError(
            f"owner-context {owner_context_id} uses rung 'scalar_smoke' but "
            f"control_kind {control_kind!r} is not in {sorted(_NO_DENSE_REFERENCE_CONTROL_KINDS)}"
        )
    if rung == "L0":
        if not 24 <= target_count <= 40:
            raise ReanalysisError(
                f"owner-context {owner_context_id} rung L0 has {target_count} target "
                "candidates; the frozen core band is 24-40"
            )
    elif rung == "L1":
        if target_count != 256:
            raise ReanalysisError(
                f"owner-context {owner_context_id} rung L1 has {target_count} target "
                "candidates; exactly 256 is required"
            )
        strict_region_count = sum(1 for row in target_rows if row["region"] == "target_strict")
        if strict_region_count < 64:
            raise ReanalysisError(
                f"owner-context {owner_context_id} rung L1 has only "
                f"{strict_region_count} target-strict-region candidates; at least 64 "
                "are required"
            )
    elif rung == "L2":
        if target_count != 1024:
            raise ReanalysisError(
                f"owner-context {owner_context_id} rung L2 has {target_count} target "
                "candidates; exactly 1024 is required"
            )


def _validate_l2_escalation_trigger(
    *,
    owner_context_id: str,
    matched_control_group: str,
    l1_stratum_failure: Mapping[str, bool],
    near_miss_declarations: frozenset[str],
) -> None:
    if l1_stratum_failure.get(matched_control_group, False):
        return
    if owner_context_id in near_miss_declarations:
        return
    raise ReanalysisError(
        f"owner-context {owner_context_id} (matched_control_group "
        f"{matched_control_group!r}) has L2 candidates without a valid escalation "
        "trigger: no L1 dense-reference-validation failure was observed within its "
        "own matched_control_group, and the owner-context was not prospectively "
        "declared a target near-miss; an L1 failure in an unrelated group never "
        "authorizes L2, and a target-only null result never authorizes L2 by itself"
    )


# --------------------------------------------------------------------------
# Population statistics (shared by fixed-budget subsets and the dense
# reference population)
# --------------------------------------------------------------------------


def _region_at_threshold(row: Mapping[str, Any], threshold: float) -> str:
    if row["region"] in ("other_owner", "background"):
        return row["region"]
    if row["iou_to_target"] >= threshold:
        return "target_strict"
    if row["iou_to_target"] > 0.0:
        return "target_halo"
    return "background"


def _match_candidates(
    candidates: Sequence[Mapping[str, Any]],
    score_index: Mapping[IdentityKey, float],
    *,
    reuse_disabled: bool,
) -> tuple[dict[str, float], list[str]]:
    reused: dict[str, float] = {}
    unmatched: list[str] = []
    for row in candidates:
        key = (
            row["predecessor_candidate_id"] or row["candidate_id"],
            row["box_identity"],
            row["raw_row_identity"],
            row["source_digest"],
        )
        if not reuse_disabled and key in score_index:
            reused[row["candidate_id"]] = score_index[key]
        else:
            unmatched.append(row["candidate_id"])
    return reused, unmatched


def _population_statistics(
    candidates: Sequence[Mapping[str, Any]],
    reused_scores: Mapping[str, float],
    iou_thresholds: Sequence[float],
) -> dict[str, dict[str, Any]]:
    by_candidate_id = {row["candidate_id"]: row for row in candidates}
    scored_pool = [cid for cid in by_candidate_id if cid in reused_scores]
    per_threshold: dict[str, dict[str, Any]] = {}
    for threshold in iou_thresholds:
        regions = {cid: _region_at_threshold(by_candidate_id[cid], threshold) for cid in by_candidate_id}
        target_strict_scored = [
            cid
            for cid in scored_pool
            if by_candidate_id[cid]["population"] == "target" and regions[cid] == "target_strict"
        ]
        background_scored = [
            cid
            for cid in scored_pool
            if by_candidate_id[cid]["population"] == "decoy" and regions[cid] == "background"
        ]
        other_owner_scored = [cid for cid in scored_pool if regions[cid] == "other_owner"]
        target_peak_candidate = (
            max(target_strict_scored, key=lambda cid: reused_scores[cid])
            if target_strict_scored
            else None
        )
        target_peak = reused_scores[target_peak_candidate] if target_peak_candidate else None
        background_peak = (
            max(reused_scores[cid] for cid in background_scored) if background_scored else None
        )
        other_owner_peak = (
            max(reused_scores[cid] for cid in other_owner_scored) if other_owner_scored else None
        )
        pooled_scored = [
            cid for cid in scored_pool if by_candidate_id[cid]["population"] in ("target", "decoy")
        ]
        localized_rank = None
        if target_peak_candidate is not None and pooled_scored:
            ordered = sorted(pooled_scored, key=lambda cid: reused_scores[cid], reverse=True)
            localized_rank = ordered.index(target_peak_candidate) / len(ordered)
        per_threshold[str(threshold)] = {
            "target_peak_candidate_id": target_peak_candidate,
            "target_peak": target_peak,
            "background_prominence": (
                target_peak - background_peak
                if target_peak is not None and background_peak is not None
                else None
            ),
            "other_owner_margin": (
                target_peak - other_owner_peak
                if target_peak is not None and other_owner_peak is not None
                else None
            ),
            "localized_rank": localized_rank,
            "scored_target_strict_count": len(target_strict_scored),
        }
    return per_threshold


def _rank_side(rank: float | None, *, rank_side_threshold: float) -> str | None:
    if rank is None:
        return None
    return "localized" if rank <= rank_side_threshold else "not_localized"


def _usable(stats_at_threshold: Mapping[str, Any], *, rank_side_threshold: float) -> bool | None:
    prominence = stats_at_threshold["background_prominence"]
    rank = stats_at_threshold["localized_rank"]
    if prominence is None or rank is None:
        return None
    return prominence > 0.0 and rank <= rank_side_threshold


def _dense_reference_comparison(
    *,
    dense_stats_at_threshold: Mapping[str, Any] | None,
    subset_stats_at_threshold: Mapping[str, Any],
    peak_tolerance_nats: float,
    rank_side_threshold: float,
) -> dict[str, Any]:
    if dense_stats_at_threshold is None:
        return {
            "dense_available": False,
            "peak_preserved": None,
            "rank_side_preserved": None,
            "usable_status_preserved": None,
            "overall_preserved": None,
        }
    dense_peak = dense_stats_at_threshold["target_peak"]
    subset_peak = subset_stats_at_threshold["target_peak"]
    peak_preserved = (
        abs(dense_peak - subset_peak) <= peak_tolerance_nats
        if dense_peak is not None and subset_peak is not None
        else None
    )
    dense_rank_side = _rank_side(
        dense_stats_at_threshold["localized_rank"], rank_side_threshold=rank_side_threshold
    )
    subset_rank_side = _rank_side(
        subset_stats_at_threshold["localized_rank"], rank_side_threshold=rank_side_threshold
    )
    rank_side_preserved = (
        dense_rank_side == subset_rank_side
        if dense_rank_side is not None and subset_rank_side is not None
        else None
    )
    dense_usable = _usable(dense_stats_at_threshold, rank_side_threshold=rank_side_threshold)
    subset_usable = _usable(subset_stats_at_threshold, rank_side_threshold=rank_side_threshold)
    usable_status_preserved = (
        dense_usable == subset_usable if dense_usable is not None and subset_usable is not None else None
    )
    sub_verdicts = [peak_preserved, rank_side_preserved, usable_status_preserved]
    evaluated = [verdict for verdict in sub_verdicts if verdict is not None]
    overall_preserved = all(evaluated) if evaluated else None
    return {
        "dense_available": True,
        "dense_target_peak": dense_peak,
        "subset_target_peak": subset_peak,
        "peak_preserved": peak_preserved,
        "dense_localized_rank": dense_stats_at_threshold["localized_rank"],
        "subset_localized_rank": subset_stats_at_threshold["localized_rank"],
        "dense_rank_side": dense_rank_side,
        "subset_rank_side": subset_rank_side,
        "rank_side_preserved": rank_side_preserved,
        "dense_usable": dense_usable,
        "subset_usable": subset_usable,
        "usable_status_preserved": usable_status_preserved,
        "overall_preserved": overall_preserved,
    }


def _fold_b2_sign_into_overall(
    threshold_entry: Mapping[str, Any], b2_sign_preserved: bool | None
) -> bool | None:
    """Re-fold a b2_pair_after context's per-threshold verdict once its
    paired-decline sign is known. Unlike the base three-verdict rule (which
    ignores an individual None), a b2-applicable context's four verdicts use
    a strict aggregator: any False makes the threshold False (failed); else
    any None makes it None (insufficient, never silently preserved); only
    all-True makes it True (may preserve)."""

    verdicts = [
        threshold_entry.get("peak_preserved"),
        threshold_entry.get("rank_side_preserved"),
        threshold_entry.get("usable_status_preserved"),
        b2_sign_preserved,
    ]
    if any(verdict is False for verdict in verdicts):
        return False
    if any(verdict is None for verdict in verdicts):
        return None
    return True


def _dense_reference_status(per_threshold: Mapping[str, dict[str, Any]]) -> str:
    # "no_dense_reference" means no dense population was supplied at all for
    # this owner-context; that is distinct from "dense was supplied but the
    # fixed-budget subset had too little matched score coverage to compare",
    # which must not be silently reported as either "no reference" or
    # "preserved".
    if not any(entry["dense_available"] for entry in per_threshold.values()):
        return "no_dense_reference"
    overall = [entry["overall_preserved"] for entry in per_threshold.values() if entry["dense_available"]]
    if any(value is False for value in overall):
        return "failed"
    if overall and all(value is True for value in overall):
        return "preserved"
    return "insufficient_subset_evidence"


# --------------------------------------------------------------------------
# Top-level build
# --------------------------------------------------------------------------


def reanalyze_sorted_fn_fixed_budget_controls(
    *,
    fixed_budget_candidates: str | Path,
    old_score_rows: str | Path,
    dense_candidates: str | Path | None = None,
    dense_score_rows: str | Path | None = None,
    near_miss_declarations: Sequence[str] | None = None,
    iou_thresholds: Sequence[float] = DEFAULT_IOU_THRESHOLDS,
    peak_tolerance_nats: float = DEFAULT_PEAK_TOLERANCE_NATS,
    rank_side_threshold: float = DEFAULT_RANK_SIDE_THRESHOLD,
    output: str | Path,
) -> dict[str, Any]:
    candidates_path = _resolved_file(fixed_budget_candidates, "fixed-budget candidates")
    old_scores_path = _resolved_file(old_score_rows, "old score rows")
    dense_candidates_path = (
        _resolved_file(dense_candidates, "dense candidates") if dense_candidates is not None else None
    )
    dense_scores_path = (
        _resolved_file(dense_score_rows, "dense score rows") if dense_score_rows is not None else None
    )
    near_miss = frozenset(near_miss_declarations or ())
    thresholds = tuple(sorted(float(value) for value in iou_thresholds))
    if not thresholds:
        raise ReanalysisError("at least one IoU threshold is required")

    candidates = load_candidates(candidates_path)
    old_scores, old_scores_filter_report = load_score_index(old_scores_path, label="old score rows")
    dense_candidates_rows = load_dense_candidates(dense_candidates_path)
    dense_scores, dense_scores_filter_report = load_score_index(dense_scores_path, label="dense score rows")

    by_owner_rung: dict[tuple[str, str], list[dict[str, Any]]] = {}
    control_kind_by_owner: dict[str, str | None] = {}
    is_control_by_owner: dict[str, bool] = {}
    matched_control_group_by_owner: dict[str, str] = {}
    for row in candidates:
        owner_context_id = row["owner_context_id"]
        by_owner_rung.setdefault((owner_context_id, row["rung"]), []).append(row)
        if owner_context_id in matched_control_group_by_owner:
            if matched_control_group_by_owner[owner_context_id] != row["matched_control_group"]:
                raise ReanalysisError(
                    f"owner-context {owner_context_id} declares inconsistent "
                    "matched_control_group values across its candidate rows"
                )
        else:
            matched_control_group_by_owner[owner_context_id] = row["matched_control_group"]
        if row["is_control"]:
            is_control_by_owner[owner_context_id] = True
            control_kind_by_owner.setdefault(owner_context_id, row["control_kind"])

    for (owner_context_id, rung), rows in sorted(by_owner_rung.items()):
        control_kind = control_kind_by_owner.get(owner_context_id)
        _validate_rung_shape(
            owner_context_id=owner_context_id, rung=rung, control_kind=control_kind, candidates=rows
        )

    dense_by_owner: dict[str, list[dict[str, Any]]] = {}
    for row in dense_candidates_rows:
        dense_by_owner.setdefault(row["owner_context_id"], []).append(row)

    dense_stats_by_owner: dict[str, dict[str, dict[str, Any]] | None] = {}
    for owner_context_id, dense_rows in dense_by_owner.items():
        reuse_disabled = control_kind_by_owner.get(owner_context_id) in _NO_DENSE_REFERENCE_CONTROL_KINDS
        dense_reused, _dense_unmatched = _match_candidates(
            dense_rows, dense_scores, reuse_disabled=reuse_disabled
        )
        dense_stats_by_owner[owner_context_id] = _population_statistics(
            dense_rows, dense_reused, thresholds
        )

    owner_context_reports: dict[str, dict[str, Any]] = {}
    for (owner_context_id, rung), rows in sorted(by_owner_rung.items()):
        control_kind = control_kind_by_owner.get(owner_context_id)
        is_control = is_control_by_owner.get(owner_context_id, False)
        reuse_disabled = control_kind in _NO_DENSE_REFERENCE_CONTROL_KINDS
        if reuse_disabled:
            stray = [row for row in rows if (
                row["predecessor_candidate_id"] or row["candidate_id"],
                row["box_identity"],
                row["raw_row_identity"],
                row["source_digest"],
            ) in old_scores]
            if stray:
                raise ReanalysisError(
                    f"owner-context {owner_context_id} has control_kind {control_kind!r}, "
                    "for which no dense predecessor score rows are expected to exist, but "
                    f"{len(stray)} candidate(s) matched an old score row; refusing silent reuse"
                )
        reused_scores, unmatched_ids = _match_candidates(rows, old_scores, reuse_disabled=reuse_disabled)
        subset_per_threshold = _population_statistics(rows, reused_scores, thresholds)

        dense_reference_validation: dict[str, Any]
        if not is_control or control_kind not in _DENSE_REFERENCE_CONTROL_KINDS:
            dense_reference_validation = {"status": "not_applicable", "per_threshold": {}}
        else:
            dense_per_threshold = dense_stats_by_owner.get(owner_context_id)
            per_threshold_comparison = {
                str(threshold): _dense_reference_comparison(
                    dense_stats_at_threshold=(
                        dense_per_threshold[str(threshold)] if dense_per_threshold is not None else None
                    ),
                    subset_stats_at_threshold=subset_per_threshold[str(threshold)],
                    peak_tolerance_nats=peak_tolerance_nats,
                    rank_side_threshold=rank_side_threshold,
                )
                for threshold in thresholds
            }
            dense_reference_validation = {
                "status": _dense_reference_status(per_threshold_comparison),
                "per_threshold": per_threshold_comparison,
            }

        stats = {
            "owner_context_id": owner_context_id,
            "rung": rung,
            "candidate_count": len(rows),
            "scored_count": len(reused_scores),
            "per_threshold": subset_per_threshold,
            "dense_reference_validation": dense_reference_validation,
            "unmatched_candidate_ids": sorted(unmatched_ids),
            "needs_rescore": sorted(unmatched_ids),
            "control_kind": control_kind,
            "matched_control_group": matched_control_group_by_owner[owner_context_id],
        }
        owner_context_reports.setdefault(owner_context_id, {})[rung] = stats

    # B2 paired-change (selective-decline) sign check: for every
    # matched_control_group containing both a b2_pair_before and a
    # b2_pair_after control owner-context, compare the decline sign between
    # the dense reference and each rung shared by both members. This is
    # conclusion-bearing: a disagreeing (False) sign must make the
    # after-context's dense_reference_validation status "failed" at that
    # rung/threshold, and an unresolvable (None) sign must downgrade an
    # otherwise-"preserved" status to "insufficient_subset_evidence" --
    # never silently leave a "preserved" status standing beside it.
    b2_pairs: dict[str, dict[str, Any]] = {}
    groups: dict[str, dict[str, str]] = {}
    for owner_context_id, control_kind in control_kind_by_owner.items():
        if control_kind in (_B2_BEFORE_KIND, _B2_AFTER_KIND):
            group = matched_control_group_by_owner[owner_context_id]
            groups.setdefault(group, {})[control_kind] = owner_context_id
    for group, members in groups.items():
        if _B2_BEFORE_KIND not in members or _B2_AFTER_KIND not in members:
            continue
        before_id, after_id = members[_B2_BEFORE_KIND], members[_B2_AFTER_KIND]
        dense_before = dense_stats_by_owner.get(before_id)
        dense_after = dense_stats_by_owner.get(after_id)
        by_threshold: dict[str, Any] = {}
        for threshold in thresholds:
            key = str(threshold)
            dense_sign = None
            if dense_before is not None and dense_after is not None:
                before_peak = dense_before[key]["target_peak"]
                after_peak = dense_after[key]["target_peak"]
                if before_peak is not None and after_peak is not None:
                    decline = after_peak - before_peak
                    dense_sign = "non_positive" if decline <= 0 else "positive"
            rung_signs: dict[str, Any] = {}
            common_rungs = set(owner_context_reports.get(before_id, {})) & set(
                owner_context_reports.get(after_id, {})
            )
            for rung in sorted(common_rungs):
                before_peak = owner_context_reports[before_id][rung]["per_threshold"][key]["target_peak"]
                after_peak = owner_context_reports[after_id][rung]["per_threshold"][key]["target_peak"]
                subset_sign = None
                if before_peak is not None and after_peak is not None:
                    decline = after_peak - before_peak
                    subset_sign = "non_positive" if decline <= 0 else "positive"
                sign_preserved = (
                    dense_sign == subset_sign if dense_sign is not None and subset_sign is not None else None
                )
                rung_signs[rung] = {
                    "subset_decline_sign": subset_sign,
                    "sign_preserved": sign_preserved,
                }
                after_validation = owner_context_reports[after_id][rung]["dense_reference_validation"]
                threshold_entry = after_validation["per_threshold"][key]
                threshold_entry["b2_sign_preserved"] = sign_preserved
                threshold_entry["overall_preserved"] = _fold_b2_sign_into_overall(
                    threshold_entry, sign_preserved
                )
                # Recompute this rung's status from the now-integrated
                # per-threshold data before it can ever be read as final.
                after_validation["status"] = _dense_reference_status(after_validation["per_threshold"])
            by_threshold[key] = {"dense_decline_sign": dense_sign, "by_rung": rung_signs}
        b2_pairs[group] = {
            "before_owner_context_id": before_id,
            "after_owner_context_id": after_id,
            "per_threshold": by_threshold,
        }

    # l1_stratum_failure is computed only after B2 integration above, so a
    # B2 sign disagreement that flips an L1 after-context's status to
    # "failed" is reflected here before any L2 escalation check runs.
    l1_stratum_failure: dict[str, bool] = {}
    for owner_context_id, rungs in owner_context_reports.items():
        l1_report = rungs.get("L1")
        if l1_report is not None and l1_report["dense_reference_validation"]["status"] == "failed":
            l1_stratum_failure[matched_control_group_by_owner[owner_context_id]] = True

    for (owner_context_id, rung), _rows in by_owner_rung.items():
        if rung == "L2":
            _validate_l2_escalation_trigger(
                owner_context_id=owner_context_id,
                matched_control_group=matched_control_group_by_owner[owner_context_id],
                l1_stratum_failure=l1_stratum_failure,
                near_miss_declarations=near_miss,
            )

    content: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "sources": {
            "fixed_budget_candidates_path": str(candidates_path),
            "old_score_rows_path": str(old_scores_path),
            "old_score_rows_filter_report": old_scores_filter_report,
            "dense_candidates_path": str(dense_candidates_path) if dense_candidates_path else None,
            "dense_score_rows_path": str(dense_scores_path) if dense_scores_path else None,
            "dense_score_rows_filter_report": dense_scores_filter_report,
        },
        "iou_thresholds": list(thresholds),
        "peak_tolerance_nats": peak_tolerance_nats,
        "rank_side_threshold": rank_side_threshold,
        "near_miss_declarations": sorted(near_miss),
        "l1_stratum_failure": dict(sorted(l1_stratum_failure.items())),
        "b2_pairs": dict(sorted(b2_pairs.items())),
        "owner_contexts": {
            owner_context_id: dict(sorted(rungs.items()))
            for owner_context_id, rungs in sorted(owner_context_reports.items())
        },
    }
    document = {**content, "report_digest": sha256_json(content)}

    output_path = Path(output).expanduser().resolve(strict=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_json_bytes(document) + b"\n"
    try:
        with output_path.open("xb") as handle:
            handle.write(encoded)
    except FileExistsError:
        if not output_path.is_file() or output_path.read_bytes() != encoded:
            raise ReanalysisError(
                "output already exists with different content; refusing to overwrite"
            ) from None
    return document


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-budget-candidates", required=True)
    parser.add_argument("--old-score-rows", required=True)
    parser.add_argument("--dense-candidates", default=None)
    parser.add_argument("--dense-score-rows", default=None)
    parser.add_argument(
        "--near-miss-declaration",
        action="append",
        default=[],
        dest="near_miss_declarations",
        help="owner_context_id prospectively declared a target near-miss (repeatable)",
    )
    parser.add_argument(
        "--iou-threshold",
        action="append",
        type=float,
        default=None,
        dest="iou_thresholds",
        help="repeatable; defaults to 0.4, 0.5, 0.6",
    )
    parser.add_argument("--peak-tolerance-nats", type=float, default=DEFAULT_PEAK_TOLERANCE_NATS)
    parser.add_argument("--rank-side-threshold", type=float, default=DEFAULT_RANK_SIDE_THRESHOLD)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    document = reanalyze_sorted_fn_fixed_budget_controls(
        fixed_budget_candidates=args.fixed_budget_candidates,
        old_score_rows=args.old_score_rows,
        dense_candidates=args.dense_candidates,
        dense_score_rows=args.dense_score_rows,
        near_miss_declarations=args.near_miss_declarations,
        iou_thresholds=args.iou_thresholds or DEFAULT_IOU_THRESHOLDS,
        peak_tolerance_nats=args.peak_tolerance_nats,
        rank_side_threshold=args.rank_side_threshold,
        output=args.output,
    )
    print(
        json.dumps(
            {
                "output": str(Path(args.output).expanduser().resolve(strict=False)),
                "report_digest": document["report_digest"],
                "owner_context_count": len(document["owner_contexts"]),
                "l1_stratum_failure": document["l1_stratum_failure"],
                "old_score_rows_filter_report": document["sources"]["old_score_rows_filter_report"],
                "dense_score_rows_filter_report": document["sources"]["dense_score_rows_filter_report"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
