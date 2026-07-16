#!/usr/bin/env python3
"""Run the fixed-encoding row-scoring-query-only eligibility crossover.

The experiment reuses one full-image feature encoding and changes only the
causal image-key eligibility of the query rows that score one canonical row.
All image-token queries, unscored prefix queries, non-image keys, and causal
future-key blocking remain unchanged.  This is an experiment-local scorer; it
does not train, generate, crop, resize, or modify shared inference code.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.research.run_fixed_encoding_object_centered_spatial_eligibility_crossover import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_LEDGER,
    DEFAULT_SOURCE_JSONL,
    _load_ledger,
    _object_pixel_box,
    _phase_score_map,
    _row_token_ids,
    _score_model_sequence,
    build_causal_key_eligibility_mask,
    build_translated_competitor_mask,
    compute_crossover_and_release,
    derive_explicit_position_ids,
    feature_bundle_fingerprint,
    first_differing_description_index,
    score_row_log_likelihoods,
)
from src.analysis.visual_support_counterfactual import (  # noqa: E402
    build_merged_support_mask,
    capture_feature_bundle,
    validate_feature_layout,
)


UNIT_ID = "2026-07-15-fixed-encoding-row-scoring-query-only-spatial-key-eligibility-crossover"
PARENT_UNIT_ID = "2026-07-15-fixed-encoding-object-centered-spatial-eligibility-crossover"
DEFAULT_PARENT_RECEIPT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-07-15-fixed-encoding-object-centered-spatial-eligibility-crossover/"
    "cohort-six-float32-20260715b/receipt.json"
)
PARENT_RECEIPT_SHA256 = "6148fa75998385eeab0ac0118d3fe540acaa6c7dfece85ee6116907ed3a703ee"
DEFAULT_CANONICAL_ROWS_RECEIPT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-07-15-fixed-encoding-soft-spatial-key-bias-dose-response/"
    "cohort-four-float32-20260715a/receipt.json"
)
CANONICAL_ROWS_RECEIPT_SHA256 = "468af3e49962b4e78c69ecac0c004e774c38da93e47e4bd5e19c05814c12580a"

DEFAULT_IMAGE_IDS = ("139", "632", "12120", "12639")
EXPECTED_ANCHOR_IDS = frozenset(DEFAULT_IMAGE_IDS)
TOLERANCE = 1e-4
REQUIRED_CROSSOVER_FLOOR = 0.10
OWNER_RELEASE_FLOOR = 0.05
OWNER_DESTRUCTION_TOLERANCE = 0.05


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.expanduser().resolve(strict=True).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _as_float_list(values: Iterable[Any]) -> list[float]:
    return [float(value) for value in values]


def _max_vector_drift(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("token log-probability vectors have different lengths")
    return max((abs(float(a) - float(b)) for a, b in zip(left, right, strict=True)), default=0.0)


def validate_parent_receipt(path: Path) -> tuple[dict[str, Any], str]:
    """Validate the immutable parent digest before parsing its JSON payload."""

    resolved = Path(path).expanduser().resolve(strict=True)
    observed_sha = _sha256_file(resolved)
    if observed_sha != PARENT_RECEIPT_SHA256:
        raise ValueError("parent receipt SHA-256 does not match frozen unit contract")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload.get("unit_id") != PARENT_UNIT_ID:
        raise ValueError("parent receipt has unexpected unit_id")
    return payload, observed_sha


def validate_canonical_rows_receipt(
    path: Path, *, hard_parent_sha256: str
) -> tuple[dict[str, Any], str]:
    """Validate the immutable soft-bias receipt that owns canonical rows."""

    resolved = Path(path).expanduser().resolve(strict=True)
    observed_sha = _sha256_file(resolved)
    if observed_sha != CANONICAL_ROWS_RECEIPT_SHA256:
        raise ValueError("canonical-row receipt SHA-256 does not match frozen unit contract")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload.get("unit_id") != "2026-07-15-fixed-encoding-soft-spatial-key-bias-dose-response":
        raise ValueError("canonical-row receipt has unexpected unit_id")
    if payload.get("parent_receipt_sha256") != str(hard_parent_sha256):
        raise ValueError("canonical-row receipt does not reference the frozen hard parent")
    return payload, observed_sha


def validate_canonical_rows_panel(
    payload: Mapping[str, Any], *, parent: Mapping[str, Any]
) -> dict[str, Mapping[str, Any]]:
    """Validate the immutable four-case row contract against the hard parent.

    The hard receipt owns the frozen image, annotation, support, feature, and
    runtime identities; the soft-bias receipt owns the exact token rows.  This
    cross-check happens before the model is assembled so a stale or mixed
    artifact cannot silently redefine the scored rows.
    """

    parent_common = {
        "source_jsonl_sha256": parent.get("source_jsonl_sha256"),
        "config_sha256": parent.get("config_sha256"),
        "audit_ledger_sha256": parent.get("audit_ledger_sha256"),
    }
    canonical_common = {
        "source_jsonl_sha256": payload.get("source_jsonl_sha256"),
        "config_sha256": payload.get("config_sha256"),
        "audit_ledger_sha256": payload.get("audit_ledger_sha256"),
    }
    if canonical_common != parent_common:
        raise ValueError("canonical-row receipt source/config/ledger contract differs from parent")
    raw_results = payload.get("results")
    if not isinstance(raw_results, Sequence) or isinstance(raw_results, (str, bytes)):
        raise ValueError("canonical-row receipt results are missing or invalid")
    by_id = {str(result.get("image_id")): result for result in raw_results if isinstance(result, Mapping)}
    if set(by_id) != EXPECTED_ANCHOR_IDS or len(by_id) != len(raw_results):
        raise ValueError("canonical-row receipt must contain exactly the four frozen anchors")
    parent_by_id = {str(result.get("image_id")): result for result in parent.get("results", [])}
    if not EXPECTED_ANCHOR_IDS.issubset(parent_by_id):
        raise ValueError("hard parent receipt is missing one or more frozen anchors")
    for image_id in sorted(EXPECTED_ANCHOR_IDS):
        canonical_result = by_id[image_id]
        parent_result = parent_by_id[image_id]
        for field in (
            "target_annotation_id",
            "competitor_annotation_id",
            "target_mask_indices",
            "competitor_mask_indices",
            "image_grid_thw",
            "merge_size",
            "feature_fingerprint",
        ):
            if canonical_result.get(field) != parent_result.get(field):
                raise ValueError(f"canonical-row case {image_id} differs from parent field {field}")
        for field in ("target_row_token_ids", "competitor_row_token_ids", "canonical_rows_sha256"):
            if field not in canonical_result:
                raise ValueError(f"canonical-row case {image_id} is missing {field}")
    return by_id


def compare_frozen_case_contract(
    *, parent_result: Mapping[str, Any], canonical_result: Mapping[str, Any], image_id: str,
    target_row: Sequence[int], competitor_row: Sequence[int],
) -> dict[str, Any]:
    """Check immutable IDs, masks, and row arrays before model scoring."""

    fields = ("target_annotation_id", "competitor_annotation_id", "target_mask_indices", "competitor_mask_indices")
    checks: dict[str, bool] = {"image_id": str(canonical_result.get("image_id")) == str(image_id)}
    for field in fields:
        if field.endswith("indices"):
            checks[field] = list(parent_result.get(field, [])) == list(canonical_result.get(field, []))
        else:
            checks[field] = str(parent_result.get(field)) == str(canonical_result.get(field))
    rows = compare_parent_canonical_rows(
        parent_result=canonical_result, target_row=target_row, competitor_row=competitor_row
    )
    checks["canonical_row_arrays_present"] = bool(rows["passed"])
    checks["canonical_rows_hash_matches"] = str(canonical_result.get("canonical_rows_sha256")) == _canonical_rows_hash(target_row, competitor_row)
    return {
        "image_id": str(image_id),
        "passed": bool(all(checks.values())),
        "checks": checks,
        "parent": {field: parent_result.get(field) for field in fields},
        "canonical": {field: canonical_result.get(field) for field in fields},
    }


def validate_requested_image_ids(values: Sequence[Any]) -> list[str]:
    """Allow a frozen-anchor subset for smoke runs without weakening the panel gate."""

    requested = [str(value) for value in values]
    if not requested:
        raise ValueError("image_ids must contain at least one frozen anchor")
    if len(requested) != len(set(requested)):
        raise ValueError("image_ids must not contain duplicate frozen anchors")
    unexpected = sorted(set(requested) - EXPECTED_ANCHOR_IDS)
    if unexpected:
        raise ValueError(f"image_ids contain unexpected anchors: {unexpected}")
    return requested


def _canonical_fingerprint(value: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for stream in ("primary", "deepstack"):
        entries = value.get(stream, [])
        if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
            raise ValueError(f"feature fingerprint stream {stream!r} is invalid")
        result[stream] = [
            {
                "sha256": str(entry["sha256"]),
                "shape": [int(v) for v in entry["shape"]],
                "dtype": str(entry["dtype"]),
            }
            for entry in entries
        ]
    return result


def compare_feature_fingerprints(expected: Mapping[str, Any], observed: Mapping[str, Any]) -> dict[str, Any]:
    expected_canonical = _canonical_fingerprint(expected)
    observed_canonical = _canonical_fingerprint(observed)
    return {
        "passed": expected_canonical == observed_canonical,
        "expected": expected_canonical,
        "observed": observed_canonical,
        "device_ignored": True,
    }


def query_row_range(*, prefix_length: int, row_length: int) -> tuple[int, int]:
    """Return inclusive causal query bounds for one teacher-forced row.

    The row token at relative index ``i`` is predicted by query index
    ``P+i-1``.  Therefore the exact range is ``[P-1, P+K-2]``.
    """

    prefix = int(prefix_length)
    length = int(row_length)
    if prefix <= 0:
        raise ValueError("prefix_length must be positive")
    if length <= 0:
        raise ValueError("row_length must be positive")
    return prefix - 1, prefix + length - 2


def build_query_scoped_key_eligibility_mask(
    *,
    sequence_length: int,
    image_key_positions: Sequence[int],
    eligible_image_positions: Sequence[int],
    prefix_length: int,
    row_length: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Build a boolean causal mask with regional restriction on one row only."""

    length = int(sequence_length)
    if length <= 0:
        raise ValueError("sequence_length must be positive")
    image = {int(value) for value in image_key_positions}
    eligible = {int(value) for value in eligible_image_positions}
    if any(value < 0 or value >= length for value in image | eligible):
        raise ValueError("image key position is outside the sequence")
    if not eligible.issubset(image):
        raise ValueError("eligible image keys must be a subset of image keys")
    query_start, query_end = query_row_range(prefix_length=prefix_length, row_length=row_length)
    if query_start < 0 or query_end >= length:
        raise ValueError("row-scoring query range is outside the sequence")
    mask = torch.tril(torch.ones((length, length), dtype=torch.bool, device=device))
    blocked = image - eligible
    if blocked:
        blocked_tensor = torch.tensor(sorted(blocked), dtype=torch.long, device=device)
        for query in range(query_start, query_end + 1):
            mask[query, blocked_tensor] = False
    return mask.unsqueeze(0).unsqueeze(0).contiguous()


# Descriptive aliases are useful to small downstream probes and keep the
# semantic name explicit without introducing another implementation.
build_row_scoring_query_only_key_eligibility_mask = build_query_scoped_key_eligibility_mask
build_query_scoped_causal_key_eligibility_mask = build_query_scoped_key_eligibility_mask


def inspect_query_scoped_mask_structure(
    mask: torch.Tensor,
    *,
    sequence_length: int,
    image_key_positions: Sequence[int],
    eligible_image_positions: Sequence[int],
    prefix_length: int,
    row_length: int,
) -> dict[str, Any]:
    """Emit a structural execution receipt for one regional row mask."""

    length = int(sequence_length)
    if tuple(mask.shape) != (1, 1, length, length) or mask.dtype is not torch.bool:
        raise ValueError("query-scoped mask must be boolean [1,1,S,S]")
    baseline = torch.tril(torch.ones((length, length), dtype=torch.bool, device=mask.device))
    observed = mask[0, 0]
    image = {int(value) for value in image_key_positions}
    eligible = {int(value) for value in eligible_image_positions}
    query_start, query_end = query_row_range(prefix_length=prefix_length, row_length=row_length)
    query_rows = list(range(query_start, query_end + 1))
    expected = baseline.clone()
    blocked = image - eligible
    if blocked:
        blocked_tensor = torch.tensor(sorted(blocked), dtype=torch.long, device=mask.device)
        for query in query_rows:
            expected[query, blocked_tensor] = False
    changed = observed != baseline
    non_image_changed = changed.clone()
    if image:
        image_tensor = torch.tensor(sorted(image), dtype=torch.long, device=mask.device)
        non_image_changed[:, image_tensor] = False
    off_scope_changed = changed.clone()
    off_scope_changed[query_start : query_end + 1, :] = False
    blocked_image_cells = 0
    if blocked:
        blocked_tensor = torch.tensor(sorted(blocked), dtype=torch.long, device=mask.device)
        blocked_image_cells = int(changed[:, blocked_tensor].sum().item())
    structural = {
        "passed": False,
        "query_range": {"start_inclusive": query_start, "end_inclusive": query_end, "indices": query_rows},
        "query_range_nonempty": bool(query_rows),
        "blocked_image_key_count": len(blocked),
        "changed_cell_count": int(changed.sum().item()),
        "blocked_image_cell_count": blocked_image_cells,
        "changed_non_image_key_cell_count": int(non_image_changed.sum().item()),
        "changed_off_scope_query_cell_count": int(off_scope_changed.sum().item()),
        "future_key_blocking_unchanged": bool(torch.equal(torch.triu(observed, diagonal=1), torch.zeros_like(torch.triu(observed, diagonal=1)))),
        "baseline_future_key_blocking_unchanged": bool(torch.equal(torch.triu(baseline, diagonal=1), torch.zeros_like(torch.triu(baseline, diagonal=1)))),
        "exact_expected_mask_match": bool(torch.equal(observed, expected)),
        "eligible_image_cells_unchanged": True,
        "mask_shape": list(mask.shape),
        "mask_dtype": str(mask.dtype),
    }
    # Compare future visibility directly to the baseline, rather than relying
    # on both being all-false, so the receipt remains useful if the baseline
    # constructor changes.
    structural["future_key_blocking_unchanged"] = bool(
        torch.equal(torch.triu(observed, diagonal=1), torch.triu(baseline, diagonal=1))
    )
    if eligible:
        eligible_tensor = torch.tensor(sorted(eligible), dtype=torch.long, device=mask.device)
        eligible_visible = baseline[:, eligible_tensor]
        structural["eligible_image_cells_unchanged"] = bool(
            torch.equal(observed[:, eligible_tensor], eligible_visible)
        )
    structural["passed"] = bool(
        structural["query_range_nonempty"]
        and structural["blocked_image_key_count"] > 0
        and structural["blocked_image_cell_count"] > 0
        and structural["changed_non_image_key_cell_count"] == 0
        and structural["changed_off_scope_query_cell_count"] == 0
        and structural["future_key_blocking_unchanged"]
        and structural["exact_expected_mask_match"]
        and structural["eligible_image_cells_unchanged"]
    )
    return structural


def assess_parent_all_allowed_continuity(
    *, parent_all_allowed: Mapping[str, Any], observed_all_allowed: Mapping[str, Any], tolerance: float = TOLERANCE
) -> dict[str, Any]:
    owners: dict[str, Any] = {}
    passed = True
    for owner in ("target", "competitor"):
        expected = parent_all_allowed[owner]
        observed = observed_all_allowed[owner]
        drift = _max_vector_drift(expected["token_log_probabilities"], observed["token_log_probabilities"])
        ranks_equal = list(map(int, expected["selected_token_ranks"])) == list(map(int, observed["selected_token_ranks"]))
        owners[owner] = {"max_abs_logprob_drift": drift, "selected_token_ranks_equal": ranks_equal, "passed": bool(drift <= tolerance and ranks_equal)}
        passed = passed and owners[owner]["passed"]
    return {"passed": bool(passed), "tolerance": float(tolerance), "owners": owners}


def compare_parent_canonical_rows(
    *, parent_result: Mapping[str, Any], target_row: Sequence[int], competitor_row: Sequence[int]
) -> dict[str, Any]:
    """Require exact row-token identity before any model scoring is attempted."""

    expected_target = parent_result.get("target_row_token_ids")
    expected_competitor = parent_result.get("competitor_row_token_ids")
    observed_target = [int(value) for value in target_row]
    observed_competitor = [int(value) for value in competitor_row]
    has_expected = isinstance(expected_target, Sequence) and not isinstance(expected_target, (str, bytes)) and isinstance(expected_competitor, Sequence) and not isinstance(expected_competitor, (str, bytes))
    if not has_expected:
        return {
            "passed": False,
            "reason": "parent receipt is missing target_row_token_ids or competitor_row_token_ids",
            "expected_target": expected_target,
            "expected_competitor": expected_competitor,
            "observed_target": observed_target,
            "observed_competitor": observed_competitor,
        }
    expected_target_ids = [int(value) for value in expected_target]
    expected_competitor_ids = [int(value) for value in expected_competitor]
    target_equal = expected_target_ids == observed_target
    competitor_equal = expected_competitor_ids == observed_competitor
    return {
        "passed": bool(target_equal and competitor_equal),
        "target_equal": target_equal,
        "competitor_equal": competitor_equal,
        "expected_target": expected_target_ids,
        "expected_competitor": expected_competitor_ids,
        "observed_target": observed_target,
        "observed_competitor": observed_competitor,
    }


def _effect_floor(no_op_drift: float) -> float:
    return max(10.0 * float(no_op_drift), 0.01)


def _phase_reversal(values: Mapping[str, Any], *, effect_floor: float) -> bool:
    return bool(
        float(values.get("crossover", 0.0)) >= max(REQUIRED_CROSSOVER_FLOOR, effect_floor)
        and float(values.get("gamma_target", 0.0)) > effect_floor
        and float(values.get("gamma_competitor", 0.0)) < -effect_floor
    )


def _owner_row_not_destructive(
    *,
    observed_arm: Mapping[str, Any],
    parent_arm: Mapping[str, Any],
    owner: str,
) -> dict[str, Any]:
    observed = float(observed_arm[owner]["full_row"]["mean"])
    parent = float(parent_arm[owner]["full_row"]["mean"])
    delta = observed - parent
    return {"owner": owner, "observed_mean": observed, "parent_mean": parent, "delta": delta, "passed": bool(delta >= -OWNER_DESTRUCTION_TOLERANCE)}


def _phase_owner_reversal(crossover: Mapping[str, Any], *, phase: str, no_op_drift: float) -> bool:
    return _phase_reversal(crossover.get(phase, {}), effect_floor=_effect_floor(no_op_drift))


def classify_query_scoped_case(
    *,
    image_id: str,
    crossover: Mapping[str, Mapping[str, Any]],
    parent_crossover: Mapping[str, Mapping[str, Any]],
    no_op_drift: float,
    different_category: bool,
    owner_rows_not_destructive: bool,
) -> dict[str, Any]:
    """Apply the frozen 139/12120/12639 case predicates."""

    floor = _effect_floor(no_op_drift)
    complete = _phase_owner_reversal(crossover, phase="full_row", no_op_drift=no_op_drift)
    geometry = _phase_owner_reversal(crossover, phase="geometry", no_op_drift=no_op_drift)
    description = _phase_owner_reversal(crossover, phase="description", no_op_drift=no_op_drift)
    parent_full = float(parent_crossover.get("full_row", {}).get("crossover", 0.0))
    parent_geometry = float(parent_crossover.get("geometry", {}).get("crossover", 0.0))
    phase_details: dict[str, Any] = {
        "effect_floor": floor,
        "crossover_floor": max(REQUIRED_CROSSOVER_FLOOR, floor),
        "complete_row_reversal": complete,
        "geometry_reversal": geometry,
        "description_reversal": description,
        "parent_full_row_crossover": parent_full,
        "parent_geometry_crossover": parent_geometry,
        "full_row_recovery": float(crossover.get("full_row", {}).get("crossover", 0.0)) / parent_full if parent_full > 0 else None,
        "geometry_recovery": float(crossover.get("geometry", {}).get("crossover", 0.0)) / parent_geometry if parent_geometry > 0 else None,
        "target_release": float(crossover.get("full_row", {}).get("target_release", 0.0)),
        "target_release_floor_passed": float(crossover.get("full_row", {}).get("target_release", 0.0)) >= OWNER_RELEASE_FLOOR,
        "complete_switch": bool(complete and geometry and (not different_category or description)),
        "semantic_switch": bool(different_category and description),
    }
    if image_id == "139":
        passed = bool(complete and geometry and (not different_category or description) and owner_rows_not_destructive)
        return {"classification": "retain_primary_complete_switch" if passed else "inconclusive", "predicate": {**phase_details, "passed": passed, "owner_rows_not_destructive": owner_rows_not_destructive}}
    if image_id == "12120":
        # The interaction map already combines the target-region and
        # competitor-region owner gammas.  Description is intentionally not
        # required: this is the frozen geometry-only phase anchor.
        passed = bool(complete and geometry and owner_rows_not_destructive)
        return {"classification": "retain_secondary_anchor" if passed else "inconclusive", "predicate": {**phase_details, "passed": passed, "owner_rows_not_destructive": owner_rows_not_destructive}}
    if image_id == "12639":
        current_full = float(crossover.get("full_row", {}).get("crossover", 0.0))
        current_geometry = float(crossover.get("geometry", {}).get("crossover", 0.0))
        target_gamma_ok = all(
            float(crossover.get(phase, {}).get("gamma_target", 0.0))
            >= float(parent_crossover.get(phase, {}).get("gamma_target", 0.0)) - 0.05
            for phase in ("full_row", "geometry")
        )
        competitor_gamma_ok = all(
            float(crossover.get(phase, {}).get("gamma_competitor", 0.0)) <= -floor
            for phase in ("full_row", "geometry")
        )
        target_release_positive = float(crossover.get("full_row", {}).get("target_release", 0.0)) > 0.0
        recovery_ok = bool(
            current_full >= 0.5 * parent_full and current_geometry >= 0.5 * parent_geometry
        )
        passed = bool(recovery_ok and target_gamma_ok and competitor_gamma_ok and target_release_positive and owner_rows_not_destructive)
        phase_details.update({"target_gamma_tolerance_passed": target_gamma_ok, "competitor_gamma_floor_passed": competitor_gamma_ok, "target_release_positive": target_release_positive, "half_parent_recovery_passed": recovery_ok})
        return {"classification": "retain_secondary_anchor" if passed else "inconclusive", "predicate": {**phase_details, "passed": passed, "owner_rows_not_destructive": owner_rows_not_destructive}}
    return {"classification": "inconclusive", "predicate": {**phase_details, "passed": False, "owner_rows_not_destructive": owner_rows_not_destructive}}


def classify_panel(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_id = {str(result.get("image_id")): result for result in results}
    observed_ids = set(by_id)
    missing_ids = sorted(EXPECTED_ANCHOR_IDS - observed_ids)
    unexpected_ids = sorted(observed_ids - EXPECTED_ANCHOR_IDS)
    duplicate_ids = sorted(
        image_id
        for image_id in EXPECTED_ANCHOR_IDS
        if sum(1 for result in results if str(result.get("image_id")) == image_id) > 1
    )
    invalid_ids = sorted(
        image_id
        for image_id, result in by_id.items()
        if str(result.get("classification", "")).startswith("invalid_")
    )
    if missing_ids or unexpected_ids or duplicate_ids or invalid_ids:
        reason = "missing_or_unexpected_anchors" if missing_ids or unexpected_ids or duplicate_ids else "invalid_anchor_cases"
        return {
            "classification": f"incomplete_panel_{reason}",
            "missing_anchor_ids": missing_ids,
            "unexpected_anchor_ids": unexpected_ids,
            "duplicate_anchor_ids": duplicate_ids,
            "invalid_case_ids": invalid_ids,
            "requested_case_count": len(results),
        }
    critical = [by_id[image_id] for image_id in ("139", "12120", "12639")]
    gated_critical = [
        result
        for result in critical
        if bool(result.get("structural_mask_gate_passed"))
        and bool(result.get("no_op_trust_gate", {}).get("passed"))
        and bool(result.get("parent_continuity", {}).get("passed"))
        and bool(result.get("feature_continuity", {}).get("passed"))
        and isinstance(result.get("crossover"), Mapping)
    ]
    if len(gated_critical) != len(critical):
        return {
            "classification": "incomplete_panel_decision_critical_cases",
            "decision_critical_ids": [str(result.get("image_id")) for result in critical],
            "gated_decision_critical_ids": [str(result.get("image_id")) for result in gated_critical],
            "invalid_case_ids": invalid_ids,
        }
    primary = by_id["139"]
    secondary = [by_id[image_id] for image_id in ("12120", "12639") if by_id[image_id].get("classification") == "retain_secondary_anchor"]
    retained = [primary] + secondary if primary.get("classification") == "retain_primary_complete_switch" else secondary
    recovery_ok = all(
        float(result.get("predicate", {}).get("full_row_recovery") or 0.0) >= 0.5
        and float(result.get("predicate", {}).get("geometry_recovery") or 0.0) >= 0.5
        for result in retained
    )
    direct_gate = bool(
        primary.get("classification") == "retain_primary_complete_switch"
        and secondary
        and recovery_ok
        and all(bool(result.get("predicate", {}).get("owner_rows_not_destructive")) for result in retained)
    )
    complete_switches = [
        result
        for result in (primary, by_id["12120"], by_id["12639"])
        if bool(result.get("predicate", {}).get("complete_switch"))
    ]
    semantic_complete = any(
        bool(result.get("predicate", {}).get("semantic_switch")) for result in complete_switches
    )
    semantic_complete_with_release = any(
        bool(result.get("predicate", {}).get("semantic_switch"))
        and float(result.get("predicate", {}).get("target_release", 0.0)) >= OWNER_RELEASE_FLOOR
        for result in complete_switches
    )
    if direct_gate and len(complete_switches) >= 2 and semantic_complete_with_release:
        classification = "promote_one_bounded_free_row_replay"
    elif direct_gate:
        classification = "support_direct_row_read_contribution"
    elif primary.get("classification") != "retain_primary_complete_switch" and not secondary:
        classification = "support_dependence_on_earlier_query_computation"
    else:
        classification = "mixed_query_scoped_result"
    return {
        "classification": classification,
        "retained_case_ids": [str(result.get("image_id")) for result in retained],
        "secondary_case_ids": [str(result.get("image_id")) for result in secondary],
        "complete_switch_case_ids": [str(result.get("image_id")) for result in complete_switches],
        "semantic_complete_switch_present": semantic_complete,
        "semantic_complete_switch_with_release_present": semantic_complete_with_release,
        "direct_read_gate_passed": direct_gate,
        "recovery_all_retained_cases_at_least_half": recovery_ok,
        "invalid_case_ids": invalid_ids,
        "requested_case_count": len(results),
    }


def _score_row(
    *, model: Any, model_inputs: Mapping[str, Any], features: Any, grid_thw: Sequence[int], merge_size: int,
    prompt_ids: Sequence[int], row: Sequence[int], image_token_id: int, image_grid_thw: torch.Tensor,
    position_ids: torch.Tensor | None, custom_mask: torch.Tensor | None, terminal_token_id: int | None,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    ids = torch.tensor([list(prompt_ids) + [int(v) for v in row]], dtype=torch.long, device=device)
    logits = _score_model_sequence(
        model=model, model_inputs=model_inputs, input_ids=ids, attention_mask=torch.ones_like(ids, dtype=torch.long),
        position_ids=position_ids, custom_mask=custom_mask, features=features, merge_size=merge_size, grid_thw=grid_thw,
    )
    return score_row_log_likelihoods(logits, prefix_length=len(prompt_ids), row_tokens=row, description_length=len(row) - 8, terminal_token_id=terminal_token_id)


def _score_arm_pair(
    *, model: Any, model_inputs: Mapping[str, Any], features: Any, grid_thw: Sequence[int], merge_size: int,
    prompt_ids: Sequence[int], target_row: Sequence[int], competitor_row: Sequence[int], image_token_id: int,
    target_indices: Sequence[int], competitor_indices: Sequence[int], region: str | None,
    image_grid_thw: torch.Tensor, terminal_token_id: int | None,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    if region is None:
        selected_indices = None
    elif region == "all_allowed":
        selected_indices = None
    elif region == "target":
        selected_indices = target_indices
    elif region == "competitor":
        selected_indices = competitor_indices
    else:
        raise ValueError(f"unknown region {region!r}")
    image_positions_by_row: dict[str, list[int]] = {}
    scores: dict[str, Any] = {}
    structural: dict[str, Any] = {}
    for owner_name, row in (("target", target_row), ("competitor", competitor_row)):
        ids = torch.tensor([list(prompt_ids) + list(row)], dtype=torch.long, device=device)
        image_positions = [int(v) for v in torch.where(ids[0] == int(image_token_id))[0].tolist()]
        if len(image_positions) == 0:
            raise ValueError("prompt contains no Qwen image placeholder keys")
        if selected_indices is not None and len(image_positions) <= max(selected_indices, default=-1):
            raise ValueError("frozen image-key index exceeds image-token count")
        positions = derive_explicit_position_ids(model, input_ids=ids, attention_mask=torch.ones_like(ids), image_grid_thw=image_grid_thw.to(device=device))
        custom = None
        if selected_indices is not None:
            custom = build_query_scoped_key_eligibility_mask(
                sequence_length=ids.shape[1], image_key_positions=image_positions,
                eligible_image_positions=[image_positions[int(i)] for i in selected_indices],
                prefix_length=len(prompt_ids), row_length=len(row), device=device,
            )
            structural[owner_name] = inspect_query_scoped_mask_structure(
                custom, sequence_length=ids.shape[1], image_key_positions=image_positions,
                eligible_image_positions=[image_positions[int(i)] for i in selected_indices],
                prefix_length=len(prompt_ids), row_length=len(row),
            )
        else:
            custom = build_causal_key_eligibility_mask(
                sequence_length=ids.shape[1], image_key_positions=image_positions,
                eligible_image_positions=image_positions, device=device,
            ) if region == "all_allowed" else None
        scores[owner_name] = _score_row(
            model=model, model_inputs=model_inputs, features=features, grid_thw=grid_thw, merge_size=merge_size,
            prompt_ids=prompt_ids, row=row, image_token_id=image_token_id, image_grid_thw=image_grid_thw,
            position_ids=None if region is None else positions, custom_mask=custom, terminal_token_id=terminal_token_id,
        )
        image_positions_by_row[owner_name] = image_positions
    first_diff = first_differing_description_index(target_row, competitor_row, description_length=len(target_row) - 8)
    arm = {"target": scores["target"], "competitor": scores["competitor"], "image_positions_by_row": image_positions_by_row}
    if structural:
        arm["structural_mask_receipt"] = structural
    if first_diff is not None:
        arm["first_differing_description"] = {
            "target": {"sum": scores["target"]["token_log_probabilities"][first_diff], "mean": scores["target"]["token_log_probabilities"][first_diff], "count": 1},
            "competitor": {"sum": scores["competitor"]["token_log_probabilities"][first_diff], "mean": scores["competitor"]["token_log_probabilities"][first_diff], "count": 1},
        }
    return arm


def _interactions(arms: Mapping[str, Any], target_name: str, competitor_name: str) -> dict[str, Any]:
    return compute_crossover_and_release(
        full_scores=_phase_score_map(arms["all_allowed_4d"]),
        target_scores=_phase_score_map(arms[target_name]),
        competitor_scores=_phase_score_map(arms[competitor_name]),
    )


def _canonical_rows_hash(target_row: Sequence[int], competitor_row: Sequence[int]) -> str:
    return _sha256_json({"target": [int(v) for v in target_row], "competitor": [int(v) for v in competitor_row]})


def _run_case(
    *, qwen: Any, model_inputs: Mapping[str, Any], features: Any, grid_thw: Sequence[int], merge_size: int,
    prompt_ids: Sequence[int], target_row: Sequence[int], competitor_row: Sequence[int], target_indices: Sequence[int],
    competitor_indices: Sequence[int], image_grid_thw: torch.Tensor, image_token_id: int,
    parent_result: Mapping[str, Any], different_category: bool, target_id: str, competitor_id: str,
    target_description: str, competitor_description: str,
) -> dict[str, Any]:
    model = qwen.model
    arms = {
        "implicit_standard_2d": _score_arm_pair(
            model=model, model_inputs=model_inputs, features=features, grid_thw=grid_thw, merge_size=merge_size,
            prompt_ids=prompt_ids, target_row=target_row, competitor_row=competitor_row, image_token_id=image_token_id,
            target_indices=target_indices, competitor_indices=competitor_indices, region=None,
            image_grid_thw=image_grid_thw, terminal_token_id=qwen.tokenizer.eos_token_id,
        ),
    }
    # ``region=explicit`` is not an intervention; score the same explicit
    # positions without a custom mask to keep this arm independent of mask
    # construction.  Reuse the helper with a sentinel implementation below.
    arms["explicit_standard_2d"] = _score_arm_pair_no_mask(
        model=model, model_inputs=model_inputs, features=features, grid_thw=grid_thw, merge_size=merge_size,
        prompt_ids=prompt_ids, target_row=target_row, competitor_row=competitor_row, image_token_id=image_token_id,
        image_grid_thw=image_grid_thw, terminal_token_id=qwen.tokenizer.eos_token_id,
    )
    arms["all_allowed_4d"] = _score_arm_pair(
        model=model, model_inputs=model_inputs, features=features, grid_thw=grid_thw, merge_size=merge_size,
        prompt_ids=prompt_ids, target_row=target_row, competitor_row=competitor_row, image_token_id=image_token_id,
        target_indices=target_indices, competitor_indices=competitor_indices, region="all_allowed",
        image_grid_thw=image_grid_thw, terminal_token_id=qwen.tokenizer.eos_token_id,
    )
    arms["target_row_query_only_hard"] = _score_arm_pair(
        model=model, model_inputs=model_inputs, features=features, grid_thw=grid_thw, merge_size=merge_size,
        prompt_ids=prompt_ids, target_row=target_row, competitor_row=competitor_row, image_token_id=image_token_id,
        target_indices=target_indices, competitor_indices=competitor_indices, region="target",
        image_grid_thw=image_grid_thw, terminal_token_id=qwen.tokenizer.eos_token_id,
    )
    arms["competitor_row_query_only_hard"] = _score_arm_pair(
        model=model, model_inputs=model_inputs, features=features, grid_thw=grid_thw, merge_size=merge_size,
        prompt_ids=prompt_ids, target_row=target_row, competitor_row=competitor_row, image_token_id=image_token_id,
        target_indices=target_indices, competitor_indices=competitor_indices, region="competitor",
        image_grid_thw=image_grid_thw, terminal_token_id=qwen.tokenizer.eos_token_id,
    )
    parent_all = parent_result["arms"]["all_allowed_4d"]
    parent_continuity = assess_parent_all_allowed_continuity(parent_all_allowed=parent_all, observed_all_allowed=arms["all_allowed_4d"])
    no_op = _assess_no_op(arms["implicit_standard_2d"], arms["explicit_standard_2d"], arms["all_allowed_4d"])
    structural = {
        "target_row_query_only_hard": arms["target_row_query_only_hard"].get("structural_mask_receipt", {}),
        "competitor_row_query_only_hard": arms["competitor_row_query_only_hard"].get("structural_mask_receipt", {}),
    }
    structure_passed = all(all(bool(item.get("passed")) for item in region.values()) for region in structural.values())
    result: dict[str, Any] = {
        "image_id": str(parent_result["image_id"]), "target_annotation_id": target_id, "competitor_annotation_id": competitor_id,
        "target_description": target_description, "competitor_description": competitor_description, "different_category": different_category,
        "target_row_token_ids": [int(v) for v in target_row], "competitor_row_token_ids": [int(v) for v in competitor_row],
        "canonical_rows_sha256": _canonical_rows_hash(target_row, competitor_row), "target_mask_indices": [int(v) for v in target_indices], "competitor_mask_indices": [int(v) for v in competitor_indices],
        "arms": arms, "parent_continuity": parent_continuity, "no_op_trust_gate": no_op, "structural_mask_receipt": structural,
        "structural_mask_gate_passed": structure_passed, "parent_hard_endpoint": {"target_eligibility": parent_result["arms"].get("target_eligibility"), "competitor_eligibility": parent_result["arms"].get("competitor_eligibility"), "crossover": parent_result.get("crossover")},
    }
    no_op_drift = float(no_op["max_abs_logprob_drift"])
    result["no_op_max_abs_logprob_drift"] = no_op_drift
    if not (no_op["passed"] and parent_continuity["passed"] and structure_passed):
        result["classification"] = "invalid_no_op_trust_gate" if not no_op["passed"] or not parent_continuity["passed"] else "invalid_structural_mask_gate"
        return result
    interactions = _interactions(arms, "target_row_query_only_hard", "competitor_row_query_only_hard")
    result["crossover"] = interactions
    result["owner_row_non_destructive"] = {
        "target_under_target_region": _owner_row_not_destructive(observed_arm=arms["target_row_query_only_hard"], parent_arm=parent_result["arms"]["target_eligibility"], owner="target"),
        "competitor_under_competitor_region": _owner_row_not_destructive(observed_arm=arms["competitor_row_query_only_hard"], parent_arm=parent_result["arms"]["competitor_eligibility"], owner="competitor"),
    }
    owner_ok = all(item["passed"] for item in result["owner_row_non_destructive"].values())
    result.update(classify_query_scoped_case(image_id=str(parent_result["image_id"]), crossover=interactions, parent_crossover=parent_result.get("crossover", {}), no_op_drift=no_op_drift, different_category=different_category, owner_rows_not_destructive=owner_ok))
    return result


def _score_arm_pair_no_mask(
    *, model: Any, model_inputs: Mapping[str, Any], features: Any, grid_thw: Sequence[int], merge_size: int,
    prompt_ids: Sequence[int], target_row: Sequence[int], competitor_row: Sequence[int], image_token_id: int,
    image_grid_thw: torch.Tensor, terminal_token_id: int | None,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    scores: dict[str, Any] = {}
    for owner_name, row in (("target", target_row), ("competitor", competitor_row)):
        ids = torch.tensor([list(prompt_ids) + list(row)], dtype=torch.long, device=device)
        positions = derive_explicit_position_ids(model, input_ids=ids, attention_mask=torch.ones_like(ids), image_grid_thw=image_grid_thw.to(device=device))
        scores[owner_name] = _score_row(model=model, model_inputs=model_inputs, features=features, grid_thw=grid_thw, merge_size=merge_size, prompt_ids=prompt_ids, row=row, image_token_id=image_token_id, image_grid_thw=image_grid_thw, position_ids=positions, custom_mask=None, terminal_token_id=terminal_token_id)
    arm = {"target": scores["target"], "competitor": scores["competitor"]}
    first_diff = first_differing_description_index(target_row, competitor_row, description_length=len(target_row) - 8)
    if first_diff is not None:
        arm["first_differing_description"] = {"target": {"sum": scores["target"]["token_log_probabilities"][first_diff], "mean": scores["target"]["token_log_probabilities"][first_diff], "count": 1}, "competitor": {"sum": scores["competitor"]["token_log_probabilities"][first_diff], "mean": scores["competitor"]["token_log_probabilities"][first_diff], "count": 1}}
    return arm


def _assess_no_op(implicit: Mapping[str, Any], explicit: Mapping[str, Any], custom: Mapping[str, Any]) -> dict[str, Any]:
    drifts: list[float] = []
    ranks_equal = True
    owners: dict[str, Any] = {}
    for owner in ("target", "competitor"):
        left = implicit[owner]
        rows = {}
        for name, arm in (("explicit", explicit), ("all_allowed", custom)):
            drift = _max_vector_drift(left["token_log_probabilities"], arm[owner]["token_log_probabilities"])
            equal = list(map(int, left["selected_token_ranks"])) == list(map(int, arm[owner]["selected_token_ranks"]))
            rows[name] = {"max_abs_logprob_drift": drift, "selected_token_ranks_equal": equal}
            drifts.append(drift)
            ranks_equal = ranks_equal and equal
        owners[owner] = rows
    maximum = max(drifts, default=0.0)
    return {"passed": bool(maximum <= TOLERANCE and ranks_equal), "max_abs_logprob_drift": maximum, "selected_token_ranks_equal": ranks_equal, "tolerance": TOLERANCE, "owners": owners}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--infer-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-jsonl", type=Path, default=DEFAULT_SOURCE_JSONL)
    parser.add_argument("--audit-ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--parent-receipt", type=Path, default=DEFAULT_PARENT_RECEIPT)
    parser.add_argument("--canonical-rows-receipt", type=Path, default=DEFAULT_CANONICAL_ROWS_RECEIPT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-ids", nargs="+", default=list(DEFAULT_IMAGE_IDS))
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    from scripts.research.run_sampled_rescue_transition import _temporary_cwd
    from src.config.fingerprint import sha256_json
    from src.config.inference import load_infer_config
    from src.data import load_raw_examples
    from src.inference.image_plan import materialize_image_plan_batch, verify_processor_model_vision_parity
    from src.inference.pipeline import _processor_config, _template_config
    from src.inference.prompt import build_prompt_record
    from src.inference.runtime import assemble_runtime

    parent_path = Path(args.parent_receipt).expanduser().resolve(strict=True)
    try:
        parent, parent_sha = validate_parent_receipt(parent_path)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    canonical_path = Path(args.canonical_rows_receipt).expanduser().resolve(strict=True)
    try:
        canonical, canonical_sha = validate_canonical_rows_receipt(
            canonical_path, hard_parent_sha256=parent_sha
        )
        canonical_by_image = validate_canonical_rows_panel(canonical, parent=parent)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    try:
        requested_image_ids = validate_requested_image_ids(args.image_ids)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    source_path = Path(args.source_jsonl).resolve(strict=True)
    source_sha = _sha256_file(source_path)
    if source_sha != parent.get("source_jsonl_sha256"):
        raise SystemExit("source JSONL hash does not match parent receipt")
    ledger_path = Path(args.audit_ledger).resolve(strict=True)
    ledger_sha = _sha256_file(ledger_path)
    if ledger_sha != parent.get("audit_ledger_sha256"):
        raise SystemExit("audit ledger hash does not match parent receipt")
    config_path = Path(args.infer_config).resolve(strict=True)
    with _temporary_cwd(config_path.parents[3]):
        resolved = load_infer_config(config_path)
    config = resolved.config.model_copy(update={"model": resolved.config.model.model_copy(update={"dtype": "fp32"})})
    config_sha = sha256_json(config.model_dump(mode="json"))
    if config_sha != parent.get("config_sha256"):
        raise SystemExit("resolved config hash does not match parent receipt")
    raw_rows = load_raw_examples(source_path)
    raw_by_image = {str(r.metadata["source"]["image_id"]): r for r in raw_rows if isinstance(r.metadata.get("source"), Mapping) and r.metadata["source"].get("image_id") is not None}
    ledger = _load_ledger(ledger_path)
    runtime = assemble_runtime(config, source_gate_root=config_path.parents[3])
    qwen = runtime.qwen
    qwen.model.eval()
    verify_processor_model_vision_parity(processor_identity=qwen.processor_identity, model_config=qwen.model.config)
    image_token_id = getattr(qwen.model.config, "image_token_id", None) or qwen.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    template = _template_config(resolved.config)
    parent_by_image = {str(r["image_id"]): r for r in parent.get("results", [])}
    results: list[dict[str, Any]] = []
    for image_id in requested_image_ids:
        if image_id not in parent_by_image:
            raise SystemExit(f"image {image_id} is absent from parent receipt")
        parent_result = parent_by_image[image_id]
        canonical_result = canonical_by_image[image_id]
        if str(parent_result.get("classification", "")).startswith("invalid_"):
            results.append({"image_id": image_id, "classification": "invalid_parent_case", "reason": "parent case invalid"})
            if image_id == "139":
                break
            continue
        raw = raw_by_image.get(image_id)
        if raw is None:
            results.append({"image_id": image_id, "classification": "inconclusive", "reason": "source image absent"})
            continue
        objects_by_id = {str(obj.object_id): obj for obj in raw.objects}
        target_id, competitor_id = str(parent_result["target_annotation_id"]), str(parent_result["competitor_annotation_id"])
        target, competitor = objects_by_id.get(target_id), objects_by_id.get(competitor_id)
        if target is None or competitor is None:
            raise SystemExit(f"frozen parent object absent for image {image_id}")
        ledger_by_id = {str(item.get("object_identifier", "")): item for item in ledger.get(image_id, [])}
        base_prompt = build_prompt_record(raw, template, processor=qwen.processor, row_index=0)
        prompt_ids = [int(v) for v in base_prompt.prompt_token_ids]
        target_row, competitor_row = _row_token_ids(qwen.tokenizer, target), _row_token_ids(qwen.tokenizer, competitor)
        row_contract = compare_frozen_case_contract(
            parent_result=parent_result,
            canonical_result=canonical_result,
            image_id=image_id,
            target_row=target_row,
            competitor_row=competitor_row,
        )
        if not row_contract["passed"]:
            raise SystemExit(f"fresh canonical row contract differs for image {image_id}: {row_contract}")
        image_plan = materialize_image_plan_batch([raw], components=qwen, processor_config=_processor_config(resolved.config), materialize=True, row_indices=[0])
        model_inputs = image_plan.model_inputs_by_row_id[raw.example_id]
        grid_thw = [int(v) for v in model_inputs["image_grid_thw"].reshape(-1, 3)[0].tolist()]
        if grid_thw != [int(v) for v in parent_result.get("image_grid_thw", [])]:
            raise SystemExit(f"image_grid_thw differs from parent for image {image_id}")
        merge_size = int(qwen.processor_identity.merge_size)
        if merge_size != int(parent_result.get("merge_size")):
            raise SystemExit(f"merge_size differs from parent for image {image_id}")
        features = capture_feature_bundle(qwen.model, model_inputs)
        layout = validate_feature_layout(features, features, grid_thw=grid_thw, merge_size=merge_size)
        feature_check = compare_feature_fingerprints(parent_result["feature_fingerprint"], feature_bundle_fingerprint(features))
        target_box = _object_pixel_box(target, ledger_by_id, width=raw.image.width, height=raw.image.height)
        competitor_box = _object_pixel_box(competitor, ledger_by_id, width=raw.image.width, height=raw.image.height)
        target_support = build_merged_support_mask(bbox_xyxy=target_box, image_width=raw.image.width, image_height=raw.image.height, layout=layout, halo=1)
        target_indices = [int(v) for v in parent_result["target_mask_indices"]]
        competitor_indices = [int(v) for v in parent_result["competitor_mask_indices"]]
        translated_competitor = build_translated_competitor_mask(target_support, target_bbox_xyxy=target_box, competitor_bbox_xyxy=competitor_box, image_width=raw.image.width, image_height=raw.image.height, temporal=layout.temporal, merged_height=layout.merged_height, merged_width=layout.merged_width)
        if sorted(int(v) for v in target_support.nonzero().flatten().tolist()) != target_indices or sorted(int(v) for v in translated_competitor.nonzero().flatten().tolist()) != competitor_indices:
            raise SystemExit(f"frozen support differs from parent for image {image_id}")
        if not feature_check["passed"]:
            results.append({"image_id": image_id, "classification": "invalid_feature_continuity", "feature_continuity": feature_check})
            if image_id == "139":
                break
            continue
        case = _run_case(qwen=qwen, model_inputs=model_inputs, features=features, grid_thw=grid_thw, merge_size=merge_size, prompt_ids=prompt_ids, target_row=target_row, competitor_row=competitor_row, target_indices=target_indices, competitor_indices=competitor_indices, image_grid_thw=model_inputs["image_grid_thw"], image_token_id=int(image_token_id), parent_result=parent_result, different_category=str(target.description) != str(competitor.description), target_id=target_id, competitor_id=competitor_id, target_description=str(target.description), competitor_description=str(competitor.description))
        case["feature_continuity"] = feature_check
        case["row_contract"] = row_contract
        case["feature_fingerprint"] = feature_bundle_fingerprint(features)
        case["image_grid_thw"] = grid_thw
        case["merge_size"] = merge_size
        case["target_mask_count"] = len(target_indices)
        case["competitor_mask_count"] = len(competitor_indices)
        if image_id == "139" and str(case.get("classification", "")).startswith("invalid_"):
            results.append(case)
            break
        results.append(case)
    return {"schema_version": "fixed_encoding_row_scoring_query_only_spatial_key_eligibility_crossover.v1", "unit_id": UNIT_ID, "model_dtype": "torch.float32", "parent_receipt": str(parent_path), "parent_receipt_sha256": parent_sha, "canonical_rows_receipt": str(canonical_path), "canonical_rows_receipt_sha256": canonical_sha, "source_jsonl": str(source_path), "source_jsonl_sha256": source_sha, "audit_ledger": str(ledger_path), "audit_ledger_sha256": ledger_sha, "config_sha256": config_sha, "results": results, "panel_decision": classify_panel(results)}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(args)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "receipt.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
