#!/usr/bin/env python3
"""CPU-only scientific analyzer for the sorted false-negative successor.

This module consumes only sealed registries, fixed-budget candidate identities,
merged raw-fp32 score rows, their run attestation, and (optionally) the sealed
behavior output.  It deliberately does not load a model, tokenizer, image, or
GPU runtime.  Every conclusion is an evidence conjunction from ``unit.md``;
missing controls or behavior produce an explicit unresolved disposition.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Reuse the producer's exact description-alias normalization instead of a
# second, independently-drifting implementation. The successor-behavior
# runner's CPU contract layer (this import) is documented import-safe:
# torch/transformers are only imported by its run_live path.
from scripts.research.run_sorted_fn_successor_behavior import (  # noqa: E402
    _normalise_description as _producer_normalise_description,
)


UNIT_ID = "2026-08-02-sorted-false-negative-mechanism-decomposition"
REGISTRY_SCHEMA_VERSION = "sorted-fn-mechanism-registry.v2"
PLANNER_RECEIPT_SCHEMA_VERSION = "sorted-fn-successor-input-plan-receipt.v2"
FIXED_BUDGET_SCHEMA_VERSION = "sorted-fn-successor-fixed-budget-candidates.v2"
MECHANISM_DECISION_RULES_SCHEMA_VERSION = "sorted-fn-mechanism-decision-rules.v1"
MERGE_RECEIPT_SCHEMA_VERSION = "sorted_fn_successor_score_shard_merge.v2"
RUN_ATTESTATION_SCHEMA_VERSION = "sorted_fn_successor_score_run_attestation.v2"
SUCCESSOR_SCORE_ROW_SCHEMA_VERSION = "sorted_fn_fixed_budget_scores.v1"
SUCCESSOR_SCORE_RECEIPT_SCHEMA_VERSION = "sorted_fn_fixed_budget_scores_receipt.v1"
SUCCESSOR_DECISION_CHANNEL = "raw_model_logprob.complete_box_logprob_sum"
BEHAVIOR_SCHEMA_VERSION = "sorted-fn-successor-behavior.v1"
BEHAVIOR_LANDSCAPE_ADMISSION_SCHEMA_VERSION = (
    "sorted-fn-successor-behavior-landscape-admission.v1"
)
DESCRIPTION_EQUIVALENCE_SCHEMA_VERSION = "sorted-fn-description-equivalence.v1"
ANALYSIS_SCHEMA_VERSION = "sorted-fn-mechanism-analysis.v1"
RECEIPT_SCHEMA_VERSION = "sorted-fn-mechanism-analysis-receipt.v1"
LANDSCAPE_EVIDENCE_SCHEMA_VERSION = "sorted-fn-landscape-evidence.v1"
COLLISION_EVIDENCE_SCHEMA_VERSION = "sorted-fn-collision-evidence.v1"
MECHANISM_EVIDENCE_SCHEMA_VERSION = "sorted-fn-owner-mechanism-evidence.v1"
SAMPLING_ADMISSION_SCHEMA_VERSION = "sorted-fn-successor-sampling-admission.v2"
L2_ADMISSION_SCHEMA_VERSION = "sorted-fn-l2-prospective-admission.v1"

ANALYSIS_NAME = "mechanism-analysis.json"
RECEIPT_NAME = "mechanism-analysis-receipt.json"
LANDSCAPE_EVIDENCE_NAME = "landscape-evidence.jsonl"
COLLISION_EVIDENCE_NAME = "collision-evidence.jsonl"
MECHANISM_EVIDENCE_NAME = "mechanism-evidence.jsonl"
SAMPLING_ADMISSION_NAME = "sampling-admission.json"
BEHAVIOR_LANDSCAPE_ADMISSION_NAME = "behavior-landscape-admission.json"

IOU_THRESHOLDS = (0.4, 0.5, 0.6)
PRIMARY_IOU_THRESHOLD = 0.5
LOWER_PROMINENCE_QUANTILE = 0.10
UPPER_RANK_QUANTILE = 0.90
PRIMARY_REPETITION_PENALTY = 1.0
REQUIRED_NULL_PAIR_IDS = frozenset(
    {"wit-16228-46-a", "wit-5001-6-a", "wit-4134-35-a"}
)
COLLISION_VARIANTS = (
    "F1_full_target_strict_max",
    "F2_near_gt_micro_strict_max",
    "F3_exact_gt_singleton",
)
NULL_PAIR_EXPECTATIONS = {
    "wit-16228-46-a": {"image_id": "16228", "is_mechanical": False},
    "wit-5001-6-a": {"image_id": "5001", "is_mechanical": True},
    "wit-4134-35-a": {"image_id": "4134", "is_mechanical": False},
}
POSITIVE_CONTROL_KINDS = frozenset(
    {"strict_positive", "strict_rescue", "loose_only"}
)
CONCLUSION_RUNGS = frozenset({"L1", "L2"})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class MechanismAnalysisError(ValueError):
    """A conclusion-critical analyzer precondition failed."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MechanismAnalysisError(f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise MechanismAnalysisError(f"{label} must be an array")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise MechanismAnalysisError(f"{label} must be a non-empty trimmed string")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MechanismAnalysisError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise MechanismAnalysisError(f"{label} must be finite")
    return result


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise MechanismAnalysisError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _resolved_file(path: str | Path, label: str) -> Path:
    try:
        resolved = Path(path).expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise MechanismAnalysisError(f"{label} does not exist") from exc
    if not resolved.is_file():
        raise MechanismAnalysisError(f"{label} must be a regular file")
    return resolved


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        return dict(_mapping(json.loads(path.read_text(encoding="utf-8")), label))
    except json.JSONDecodeError as exc:
        raise MechanismAnalysisError(f"{label} is not valid JSON") from exc


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise MechanismAnalysisError(f"{label} contains a blank line at {line_number}")
        try:
            rows.append(dict(_mapping(json.loads(line), f"{label} row {line_number}")))
        except json.JSONDecodeError as exc:
            raise MechanismAnalysisError(
                f"{label} row {line_number} is not valid JSON"
            ) from exc
    if not rows:
        raise MechanismAnalysisError(f"{label} is empty")
    return rows


def _validate_self_digest(document: Mapping[str, Any], key: str, label: str) -> str:
    declared = _digest(document.get(key), f"{label}.{key}")
    observed = sha256_json({k: v for k, v in document.items() if k != key})
    if observed != declared:
        raise MechanismAnalysisError(
            f"{label} {key} mismatch: observed {observed}, expected {declared}"
        )
    return declared


def _validate_file_ref(value: Any, path: Path, label: str) -> None:
    ref = _mapping(value, label)
    if Path(str(ref.get("path", ""))).expanduser().resolve() != path:
        raise MechanismAnalysisError(f"{label}.path does not name the supplied artifact")
    expected = _digest(ref.get("sha256"), f"{label}.sha256")
    observed = sha256_file(path)
    if observed != expected:
        raise MechanismAnalysisError(
            f"{label} digest mismatch: observed {observed}, expected {expected}"
        )


def _write_create_or_identical(path: Path, encoded: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(encoded)
        return "created"
    except FileExistsError:
        if not path.is_file() or path.read_bytes() != encoded:
            raise MechanismAnalysisError(
                f"{path} already exists with different content; refusing to overwrite"
            ) from None
        return "identical_existing"


def quantile_linear(values: Sequence[float], q: float) -> float:
    """Hyndman-Fan type-7 (linear) quantile, frozen for this analyzer."""

    if not values:
        raise MechanismAnalysisError("quantile requires at least one value")
    if not 0.0 <= q <= 1.0:
        raise MechanismAnalysisError("quantile q must be in [0,1]")
    ordered = sorted(_finite(value, "quantile value") for value in values)
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _logsumexp_weighted(scores_and_weights: Sequence[tuple[float, float]]) -> float:
    if not scores_and_weights:
        raise MechanismAnalysisError("proposal-weighted log-sum-exp needs candidates")
    total_weight = sum(weight for _score, weight in scores_and_weights)
    if not math.isfinite(total_weight) or total_weight <= 0:
        raise MechanismAnalysisError("proposal weights must have a positive finite sum")
    maximum = max(score for score, _weight in scores_and_weights)
    weighted = sum(weight * math.exp(score - maximum) for score, weight in scores_and_weights)
    return maximum + math.log(weighted)


def _region_at_threshold(candidate: Mapping[str, Any], threshold: float) -> str:
    declared = str(candidate.get("region"))
    if declared in {"other_owner", "background"}:
        return declared
    iou_value = _finite(candidate.get("iou_to_target"), "candidate.iou_to_target")
    if iou_value >= threshold:
        return "target_strict"
    if iou_value > 0:
        return "target_halo"
    return "background"


def _proposal_measure(candidate: Mapping[str, Any], population: str) -> str:
    value = candidate.get("proposal_measure_id", candidate.get("proposal_measure"))
    if value is None:
        return "fixed_budget_family_mirrored_equal_weight"
    return _string(value, f"{population} proposal measure")


def compute_landscape_statistics(
    candidates: Sequence[Mapping[str, Any]],
    scores_by_candidate_id: Mapping[str, float],
    *,
    iou_threshold: float = PRIMARY_IOU_THRESHOLD,
    other_owner_reference_bound: bool | None = None,
    target_gt_box: Sequence[int] | None = None,
    target_gt_owner_id: str | None = None,
) -> dict[str, Any]:
    """Compute the frozen raw-fp32 functional for one context and rung."""

    if not candidates:
        raise MechanismAnalysisError("landscape population is empty")
    candidate_ids = [_string(row.get("candidate_id"), "candidate_id") for row in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise MechanismAnalysisError("landscape population has duplicate candidate ids")
    if set(candidate_ids) != set(scores_by_candidate_id):
        missing = sorted(set(candidate_ids) - set(scores_by_candidate_id))
        extra = sorted(set(scores_by_candidate_id) - set(candidate_ids))
        raise MechanismAnalysisError(
            f"landscape score join is not exact; missing={missing[:5]} extra={extra[:5]}"
        )

    target_rows = [row for row in candidates if row.get("population") == "target"]
    decoy_rows = [row for row in candidates if row.get("population") == "decoy"]
    reference_rows = [
        row for row in candidates if row.get("population") == "reference"
    ]
    invalid_populations = sorted(
        {str(row.get("population")) for row in candidates}
        - {"target", "decoy", "reference"}
    )
    if invalid_populations:
        raise MechanismAnalysisError(f"invalid candidate populations: {invalid_populations}")
    target_families = Counter(str(row.get("family_id")) for row in target_rows)
    decoy_families = Counter(str(row.get("family_id")) for row in decoy_rows)
    equal_count = bool(target_rows) and len(target_rows) == len(decoy_rows)
    family_multisets_equal = target_families == decoy_families

    strict_rows = [
        row for row in target_rows if _region_at_threshold(row, iou_threshold) == "target_strict"
    ]
    background_rows = [
        row for row in decoy_rows if _region_at_threshold(row, iou_threshold) == "background"
    ]
    equal_background_count = bool(target_rows) and len(background_rows) == len(target_rows)
    registered_other_rows = [
        row
        for row in reference_rows
        if _region_at_threshold(row, iou_threshold) == "other_owner"
        and isinstance(row.get("other_owner_gt_owner_id"), str)
        and row.get("other_owner_gt_owner_id")
    ]
    if other_owner_reference_bound is True and not reference_rows:
        raise MechanismAnalysisError(
            "owner-context ledger binds an other-owner reference but candidate rows are missing"
        )
    if other_owner_reference_bound is False and reference_rows:
        raise MechanismAnalysisError(
            "reference candidates exist without an owner-context ledger binding"
        )
    if reference_rows and len(registered_other_rows) != len(reference_rows):
        raise MechanismAnalysisError(
            "reference population contains unbound or non-other-owner rows"
        )

    rungs = {str(row.get("rung")) for row in candidates}
    if len(rungs) != 1:
        raise MechanismAnalysisError("candidate rows mix budget rungs")
    rung = next(iter(rungs))
    expected_margin_counts = {
        "L0": (6, 7),
        "scalar_smoke": (6, 7),
        "L1": (64, 65),
    }.get(rung)
    target_margin_rows = [
        row
        for row in target_rows
        if row.get("family_id") == "near_gt_micro"
        and row.get("candidate_neighborhood_member") is True
    ]
    target_near_gt_strict_rows = [
        row
        for row in target_margin_rows
        if _region_at_threshold(row, iou_threshold) == "target_strict"
    ]
    reference_margin_rows = [
        row
        for row in registered_other_rows
        if row.get("family_id") == "near_other_micro"
    ]
    exact_gt_rows = [
        row
        for row in target_margin_rows
        if row.get("exact_gt_singleton_member") is True
    ]
    requires_exact_singleton = target_gt_box is not None or any(
        row.get("schema_version") == FIXED_BUDGET_SCHEMA_VERSION
        for row in candidates
    )
    if requires_exact_singleton:
        exact_gt_box = (
            None
            if target_gt_box is None
            else [int(value) for value in target_gt_box]
        )
        if exact_gt_box is not None and len(exact_gt_box) != 4:
            raise MechanismAnalysisError("target_gt_box must contain four bins")
        for row in candidates:
            expected_neighborhood_member = (
                row.get("population") == "target"
                and row.get("family_id") == "near_gt_micro"
            )
            neighborhood_member = row.get("candidate_neighborhood_member")
            neighborhood_id = row.get("candidate_neighborhood_id")
            if neighborhood_member is not expected_neighborhood_member:
                raise MechanismAnalysisError(
                    "candidate-neighborhood marker does not match target/near_gt_micro eligibility"
                )
            if expected_neighborhood_member != (
                isinstance(neighborhood_id, str) and bool(neighborhood_id)
            ):
                raise MechanismAnalysisError(
                    "candidate-neighborhood ID presence does not match its member marker"
                )
            member = row.get("exact_gt_singleton_member")
            singleton_id = row.get("exact_gt_singleton_id")
            if not isinstance(member, bool):
                raise MechanismAnalysisError(
                    "every candidate must declare exact_gt_singleton_member as a boolean"
                )
            if member is False and singleton_id is not None:
                raise MechanismAnalysisError(
                    "non-singleton candidate declares an exact_gt_singleton_id"
                )
        neighborhood_ids = [
            str(row["candidate_neighborhood_id"])
            for row in candidates
            if row.get("candidate_neighborhood_member") is True
        ]
        if len(neighborhood_ids) != len(set(neighborhood_ids)):
            raise MechanismAnalysisError(
                "candidate-neighborhood IDs are duplicated within one context/rung"
            )
        if len(exact_gt_rows) != 1:
            raise MechanismAnalysisError(
                "F3 requires exactly one explicitly marked near_gt_micro candidate"
            )
        if exact_gt_box is not None and list(exact_gt_rows[0].get("box", [])) != exact_gt_box:
            raise MechanismAnalysisError(
                "the marked F3 singleton candidate is not the exact GT box"
            )
        if not isinstance(exact_gt_rows[0].get("exact_gt_singleton_id"), str) or not exact_gt_rows[0][
            "exact_gt_singleton_id"
        ]:
            raise MechanismAnalysisError(
                "F3 exact-GT singleton must carry a non-empty exact_gt_singleton_id"
            )
        if target_gt_owner_id is not None and exact_gt_rows[0][
            "exact_gt_singleton_id"
        ] != f"exact-gt:{target_gt_owner_id}:{rung}":
            raise MechanismAnalysisError(
                "F3 exact-GT singleton ID does not follow the bound owner/rung identity"
            )
    if reference_rows:
        if expected_margin_counts is None:
            raise MechanismAnalysisError(
                f"other-owner margin is undefined for rung {rung!r}"
            )
        expected_target_raw_count, expected_logical_count = expected_margin_counts
        if (
            len(target_margin_rows) != expected_target_raw_count
            or len(reference_margin_rows) != expected_logical_count
            or len(exact_gt_rows) != 1
        ):
            raise MechanismAnalysisError(
                "other-owner margin requires the exact logical near-GT/reference counts "
                f"({expected_logical_count}/{expected_logical_count}) from "
                f"{expected_target_raw_count} near_gt_micro rows plus one marked F3 "
                f"singleton versus {expected_logical_count} reference rows; observed "
                f"raw/logical/reference={len(target_margin_rows)}/"
                f"{len(target_margin_rows) + len(exact_gt_rows)}/"
                f"{len(reference_margin_rows)}"
            )
        reference_owner_ids = {
            str(row["other_owner_gt_owner_id"]) for row in reference_margin_rows
        }
        if len(reference_owner_ids) != 1:
            raise MechanismAnalysisError(
                "reference population does not bind exactly one other owner"
            )

    target_peak_row = None
    if strict_rows:
        target_peak_row = min(
            strict_rows,
            key=lambda row: (-float(scores_by_candidate_id[str(row["candidate_id"])]), str(row["candidate_id"])),
        )
    target_peak = (
        None
        if target_peak_row is None
        else float(scores_by_candidate_id[str(target_peak_row["candidate_id"])])
    )
    background_peak = (
        max(float(scores_by_candidate_id[str(row["candidate_id"])]) for row in background_rows)
        if background_rows and equal_count and equal_background_count
        else None
    )
    target_margin_peak = (
        max(float(scores_by_candidate_id[str(row["candidate_id"])]) for row in target_margin_rows)
        if reference_rows
        else None
    )
    near_gt_strict_peak = (
        max(
            float(scores_by_candidate_id[str(row["candidate_id"])])
            for row in target_near_gt_strict_rows
        )
        if target_near_gt_strict_rows
        else None
    )
    other_owner_peak = (
        max(float(scores_by_candidate_id[str(row["candidate_id"])]) for row in reference_margin_rows)
        if reference_rows
        else None
    )
    exact_gt_singleton_score = (
        float(scores_by_candidate_id[str(exact_gt_rows[0]["candidate_id"])])
        if len(exact_gt_rows) == 1
        else None
    )

    localized_rank = None
    if target_peak is not None and equal_count:
        # The reference lattice is a separate registered other-owner margin
        # control. It never changes target/decoy rank, equal-count, or
        # proposal-measure mass.
        union_scores = [
            float(scores_by_candidate_id[str(row["candidate_id"])])
            for row in [*target_rows, *decoy_rows]
        ]
        localized_rank = sum(score > target_peak for score in union_scores) / len(union_scores)

    target_measures = {_proposal_measure(row, "target") for row in target_rows}
    decoy_measures = {_proposal_measure(row, "decoy") for row in decoy_rows}
    measures_equal = len(target_measures) == 1 and target_measures == decoy_measures
    target_weights = {
        str(row["candidate_id"]): _finite(
            row.get("proposal_weight", 1.0 / len(target_rows)),
            "target candidate.proposal_weight",
        )
        for row in target_rows
    }
    decoy_weights = {
        str(row["candidate_id"]): _finite(
            row.get("proposal_weight", 1.0 / len(decoy_rows)),
            "decoy candidate.proposal_weight",
        )
        for row in decoy_rows
    }
    if any(weight <= 0 for weight in [*target_weights.values(), *decoy_weights.values()]):
        raise MechanismAnalysisError("candidate.proposal_weight must be positive")
    weight_sums_equal = math.isclose(
        sum(target_weights.values()),
        sum(decoy_weights.values()),
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    proposal_guard = (
        equal_count
        and family_multisets_equal
        and measures_equal
        and weight_sums_equal
    )
    target_bank_score = None
    if proposal_guard and strict_rows:
        weighted: list[tuple[float, float]] = []
        for row in strict_rows:
            weight = target_weights[str(row["candidate_id"])]
            weighted.append((float(scores_by_candidate_id[str(row["candidate_id"])]), weight))
        target_bank_score = _logsumexp_weighted(weighted)

    raw_candidates = []
    for row in sorted(candidates, key=lambda item: str(item["candidate_id"])):
        raw_candidates.append(
            {
                "candidate_id": row["candidate_id"],
                "family_id": row.get("family_id"),
                "population": row.get("population"),
                "declared_region": row.get("region"),
                "region_at_threshold": _region_at_threshold(row, iou_threshold),
                "iou_to_target": row.get("iou_to_target"),
                "box": row.get("box"),
                "other_owner_gt_owner_id": row.get("other_owner_gt_owner_id"),
                "geometry_cluster_id": row.get("geometry_cluster_id"),
                "peak_cluster_id": row.get("peak_cluster_id"),
                "candidate_neighborhood_id": row.get("candidate_neighborhood_id"),
                "candidate_neighborhood_member": row.get(
                    "candidate_neighborhood_member"
                ),
                "exact_gt_singleton_member": row.get(
                    "exact_gt_singleton_member"
                ),
                "exact_gt_singleton_id": row.get("exact_gt_singleton_id"),
                "control_kind": row.get("control_kind"),
                "matched_control_group": row.get("matched_control_group"),
                "source_digest": row.get("source_digest"),
                "coord_token_ids": row.get("coord_token_ids"),
                "raw_fp32_complete_box_logprob_sum": float(
                    scores_by_candidate_id[str(row["candidate_id"])]
                ),
            }
        )

    return {
        "iou_threshold": float(iou_threshold),
        "target_population_count": len(target_rows),
        "decoy_population_count": len(decoy_rows),
        "reference_population_count": len(reference_rows),
        "equal_count_guard": equal_count,
        "equal_count_background_guard": equal_background_count,
        "target_family_multiset": dict(sorted(target_families.items())),
        "decoy_family_multiset": dict(sorted(decoy_families.items())),
        "family_multiset_guard": family_multisets_equal,
        "proposal_measure_guard": measures_equal,
        "proposal_weight_sum_guard": weight_sums_equal,
        "target_control_proposal_comparable": proposal_guard,
        "target_peak": target_peak,
        "target_peak_candidate_id": None if target_peak_row is None else target_peak_row["candidate_id"],
        "target_peak_box": None if target_peak_row is None else target_peak_row.get("box"),
        "collision_variant_peaks": {
            "F1_full_target_strict_max": target_peak,
            "F2_near_gt_micro_strict_max": near_gt_strict_peak,
            "F3_exact_gt_singleton": exact_gt_singleton_score,
        },
        "background_peak": background_peak,
        "background_prominence": (
            None if target_peak is None or background_peak is None else target_peak - background_peak
        ),
        "other_owner_peak": other_owner_peak,
        "other_owner_margin": (
            None
            if target_margin_peak is None or other_owner_peak is None
            else target_margin_peak - other_owner_peak
        ),
        "other_owner_margin_target_peak": target_margin_peak,
        "other_owner_margin_status": (
            "paired_reference_margin"
            if reference_rows
            else "structurally_inapplicable_no_bound_same_description_neighbor"
        ),
        "other_owner_margin_population_accounting": {
            "target_near_gt_micro_raw_row_count": len(target_margin_rows),
            "target_exact_gt_singleton_repeat_count": len(exact_gt_rows),
            "target_logical_multiset_count": len(target_margin_rows)
            + len(exact_gt_rows),
            "target_unique_geometry_count": len(
                {tuple(row.get("box", [])) for row in target_margin_rows}
            ),
            "reference_near_other_micro_raw_row_count": len(
                reference_margin_rows
            ),
            "reference_unique_geometry_count": len(
                {tuple(row.get("box", [])) for row in reference_margin_rows}
            ),
        },
        "registered_other_owner_control_count": len(registered_other_rows),
        "localized_rank": localized_rank,
        "target_bank_score": target_bank_score,
        "raw_candidates": raw_candidates,
    }


def compute_collision_envelope(
    null_declines: Mapping[str, float], scalar_tolerance: float
) -> dict[str, Any]:
    """Return the global numerical envelope, including the mechanical null."""

    tolerance = _finite(scalar_tolerance, "scalar_tolerance")
    if tolerance < 0:
        raise MechanismAnalysisError("scalar_tolerance must be non-negative")
    values = {str(key): _finite(value, f"null {key}") for key, value in null_declines.items()}
    if len(values) < 3:
        return {
            "status": "insufficient_nulls",
            "null_declines": dict(sorted(values.items())),
            "threshold": None,
            "leave_one_out": {},
        }
    threshold = min(values.values()) - tolerance
    leave_one_out = {}
    for omitted in sorted(values):
        retained = [value for pair_id, value in values.items() if pair_id != omitted]
        leave_one_out[omitted] = {
            "retained_null_count": len(retained),
            "threshold": min(retained) - tolerance,
        }
    return {
        "status": "admitted",
        "null_declines": dict(sorted(values.items())),
        "most_negative_observed_null": min(values.values()),
        "scalar_tolerance": tolerance,
        "threshold": threshold,
        "mechanical_null_included": True,
        "leave_one_out": leave_one_out,
    }


def evaluate_collision_candidate(
    *,
    target_gt_owner_id: str,
    image_id: str,
    selective_declines_by_variant_and_iou: Mapping[str, Mapping[str, float]],
    envelopes_by_variant_and_iou: Mapping[
        str, Mapping[str, Mapping[str, Any]]
    ],
    behavior_co_movement: bool | None,
    token_identical_matched_null: bool,
    candidate_neighborhood_sign_stable: bool | None,
) -> dict[str, Any]:
    """Apply the frozen 16228/2685/tiny-person asymmetric collision gate."""

    reasons: list[str] = []
    primary_key = str(PRIMARY_IOU_THRESHOLD)
    f1_declines = _mapping(
        selective_declines_by_variant_and_iou.get(COLLISION_VARIANTS[0], {}),
        "F1 selective declines",
    )
    primary = f1_declines.get(primary_key)
    primary_envelope = _mapping(
        envelopes_by_variant_and_iou.get(COLLISION_VARIANTS[0], {}),
        "F1 collision envelopes",
    ).get(primary_key, {})
    threshold = primary_envelope.get("threshold")
    likelihood_exceeds = bool(
        primary is not None and threshold is not None and float(primary) < float(threshold)
    )

    def _leave_one_out_exceeded(entry: object) -> bool:
        if primary is None or not isinstance(entry, Mapping):
            return False
        entry_threshold = entry.get("threshold")
        if isinstance(entry_threshold, bool) or not isinstance(
            entry_threshold, (int, float)
        ):
            return False
        return float(primary) < float(entry_threshold)

    leave_one_out_exceedance = {
        pair_id: _leave_one_out_exceeded(entry)
        for pair_id, entry in _mapping(
            primary_envelope.get("leave_one_out", {}), "collision leave-one-out envelope"
        ).items()
    }
    sign_stability_by_variant = {
        variant: bool(selective_declines_by_variant_and_iou.get(variant))
        and set(selective_declines_by_variant_and_iou[variant])
        == {str(value) for value in IOU_THRESHOLDS}
        and all(
            float(value) < 0
            for value in selective_declines_by_variant_and_iou[variant].values()
        )
        for variant in COLLISION_VARIANTS
    }
    sign_stable = all(sign_stability_by_variant.values())
    if not likelihood_exceeds:
        reasons.append("selective_decline_does_not_exceed_global_null_envelope")
    if not sign_stable:
        reasons.append("selective_decline_sign_not_stable_across_iou_band")
    if candidate_neighborhood_sign_stable is None:
        reasons.append("frozen_candidate_neighborhood_perturbation_band_missing")
    elif not candidate_neighborhood_sign_stable:
        reasons.append("selective_decline_sign_not_stable_across_candidate_neighborhood")

    structural_status = "global_envelope_only"
    may_support = False
    if target_gt_owner_id == "gt:16228:30":
        structural_status = "same_image_person_null_required"
        may_support = token_identical_matched_null
        if not token_identical_matched_null:
            reasons.append("missing_token_identical_same_image_P_G_F_null_match")
    elif target_gt_owner_id == "gt:2685:15" or image_id == "2685":
        structural_status = "missing_bottle_stratum_null"
        reasons.append("global_envelope_is_diagnostic_only_without_bottle_null")
    elif image_id == "7511":
        structural_status = "missing_tiny_person_stratum_null"
        reasons.append("global_envelope_cannot_decide_tiny_person_collision")
    else:
        reasons.append("global_envelope_lacks_a_declared_matched_collision_stratum")

    if behavior_co_movement is None:
        reasons.append(
            "behavior_co_movement_unavailable_missing_output_or_autonomous_suffix_partition"
        )
    elif not behavior_co_movement:
        reasons.append("required_behavior_co_movement_absent")

    supported = bool(
        likelihood_exceeds
        and sign_stable
        and may_support
        and behavior_co_movement is True
        and candidate_neighborhood_sign_stable is True
    )
    if supported:
        status = "collision_supported_in_tested_sampled_context"
        reasons = ["all_collision_likelihood_structure_and_behavior_gates_passed"]
    elif likelihood_exceeds:
        status = (
            "likelihood_only_collision_consistent_unresolved"
            if behavior_co_movement is None
            else "collision_consistent_unresolved"
        )
    else:
        status = "collision_unresolved"
    return {
        "supported": supported,
        "status": status,
        "structural_status": structural_status,
        "likelihood_exceeds_envelope": likelihood_exceeds,
        "leave_one_out_exceedance": leave_one_out_exceedance,
        "leave_one_out_all_exceeded": bool(leave_one_out_exceedance)
        and all(leave_one_out_exceedance.values()),
        "sign_stable_across_iou_band": sign_stable,
        "sign_stability_by_variant": sign_stability_by_variant,
        "behavior_co_movement": behavior_co_movement,
        "candidate_neighborhood_sign_stable": candidate_neighborhood_sign_stable,
        "token_identical_matched_null": token_identical_matched_null,
        "reasons": reasons,
    }


def validate_common_candidate_neighborhood_ids(
    by_arm: Mapping[str, Mapping[str, float]],
) -> list[str]:
    """Require one exact non-empty neighborhood ID set across P/G/F."""

    if set(by_arm) != {"P", "G", "F"}:
        raise MechanismAnalysisError(
            "candidate-neighborhood comparison requires exact P/G/F arms"
        )
    ids = {arm: set(scores) for arm, scores in by_arm.items()}
    if not ids["P"] or ids["P"] != ids["G"] or ids["P"] != ids["F"]:
        raise MechanismAnalysisError(
            "candidate-neighborhood IDs are not one common non-empty set across P/G/F"
        )
    return sorted(ids["P"])


def validate_common_exact_gt_singleton_ids(
    by_arm: Mapping[str, Sequence[str]],
) -> str:
    """Require one identical explicit F3 singleton ID across P/G/F."""

    if set(by_arm) != {"P", "G", "F"}:
        raise MechanismAnalysisError(
            "exact-GT-singleton comparison requires exact P/G/F arms"
        )
    singleton_ids = {
        arm: [str(value) for value in values if isinstance(value, str) and value]
        for arm, values in by_arm.items()
    }
    if any(len(values) != 1 for values in singleton_ids.values()):
        raise MechanismAnalysisError(
            "each P/G/F arm must expose exactly one explicit exact-GT-singleton ID"
        )
    selected = {values[0] for values in singleton_ids.values()}
    if len(selected) != 1:
        raise MechanismAnalysisError(
            "exact-GT-singleton IDs are not identical across P/G/F"
        )
    return next(iter(selected))


def _all_registry_roles(registry: Mapping[str, Any]) -> list[dict[str, Any]]:
    smoke = _mapping(registry.get("smoke"), "registry.smoke")
    roles = [
        dict(_mapping(role, "registry.smoke.roles[]"))
        for role in _sequence(smoke.get("roles", []), "registry.smoke.roles")
    ]
    envelope = _mapping(smoke.get("null_pair_envelope", {}), "registry null envelope")
    for pair in _sequence(envelope.get("pairs", []), "registry null pairs"):
        for role in _sequence(_mapping(pair, "null pair").get("roles", []), "null pair roles"):
            roles.append(dict(_mapping(role, "null pair role")))
    role_ids = [str(role.get("role_id")) for role in roles]
    if any(not role_id for role_id in role_ids) or len(role_ids) != len(set(role_ids)):
        raise MechanismAnalysisError("registry role ids are missing or duplicated")
    return roles


def _token_bin_map(rules: Mapping[str, Any]) -> dict[int, int]:
    token_registry = _mapping(rules.get("token_registry"), "decision rules.token_registry")
    coordinate = _mapping(
        token_registry.get("coordinate_bin_to_token_id"),
        "decision rules.token_registry.coordinate_bin_to_token_id",
    )
    token_ids = _sequence(
        coordinate.get("coordinate_bin_token_ids"), "coordinate_bin_token_ids"
    )
    reverse: dict[int, int] = {}
    for bin_value, token_id in enumerate(token_ids):
        token = int(token_id)
        if token in reverse:
            raise MechanismAnalysisError("coordinate token registry is not one-to-one")
        reverse[token] = bin_value
    return reverse


def _attach_candidate_boxes(
    rows: Sequence[Mapping[str, Any]], rules: Mapping[str, Any]
) -> list[dict[str, Any]]:
    reverse = _token_bin_map(rules)
    attached: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        if item.get("schema_version") != FIXED_BUDGET_SCHEMA_VERSION:
            raise MechanismAnalysisError("fixed-budget candidate has a foreign schema_version")
        tokens = [int(value) for value in _sequence(item.get("coord_token_ids"), "coord_token_ids")]
        if len(tokens) != 4 or any(token not in reverse for token in tokens):
            raise MechanismAnalysisError("candidate coordinate tokens do not map to one complete box")
        decoded = [reverse[token] for token in tokens]
        if "box" in item and list(item["box"]) != decoded:
            raise MechanismAnalysisError("candidate box disagrees with its coordinate token identity")
        item["box"] = decoded
        attached.append(item)
    return attached


def _extract_scalar_tolerance(
    attestation: Mapping[str, Any], rules: Mapping[str, Any]
) -> dict[str, float]:
    backend = _mapping(attestation.get("scalar_acceptance_backend"), "scalar acceptance backend")
    candidates = []
    for value in (
        attestation.get("scalar_numeric_tolerance"),
        attestation.get("numeric_tolerance"),
        backend.get("numeric_tolerance"),
        backend.get("atol"),
        backend.get("rtol"),
    ):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            candidates.append(_finite(value, "scalar numerical tolerance"))
    if not candidates or any(value < 0 for value in candidates):
        raise MechanismAnalysisError("run attestation does not expose a usable scalar tolerance")
    rules_tolerance = rules.get("numeric_tolerance")
    rule_value = (
        _finite(rules_tolerance, "decision rules.numeric_tolerance")
        if isinstance(rules_tolerance, (int, float)) and not isinstance(rules_tolerance, bool)
        else 0.0
    )
    return {
        "attested_scalar_tolerance": max(candidates),
        "rules_numeric_tolerance": rule_value,
        "effective_tolerance": max([rule_value, *candidates]),
    }


def _reconstruct_alias_to_category(categories: Mapping[str, Any]) -> dict[str, str] | None:
    """Deterministically rebuild alias_to_category from frozen categories.

    Mirrors validate_description_equivalence's collision rules in
    run_sorted_fn_successor_behavior.py exactly (normalization is imported
    from there, not reimplemented) so a contract's claimed derivation can be
    checked for exact equality rather than merely trusted. Returns None
    (fail-closed) for anything the producer itself would refuse to freeze:
    an empty/blank alias, or one alias claimed by two categories.
    """
    alias_owner: dict[str, str] = {}
    for category, raw_aliases in categories.items():
        category_name = _producer_normalise_description(category)
        if isinstance(raw_aliases, (str, bytes)) or not isinstance(raw_aliases, Sequence):
            return None
        aliases = {_producer_normalise_description(value) for value in raw_aliases}
        aliases.add(category_name)
        if not category_name or "" in aliases:
            return None
        for alias in aliases:
            existing = alias_owner.get(alias)
            if existing is not None and existing != category_name:
                return None
            alias_owner[alias] = category_name
    if not alias_owner:
        return None
    return alias_owner


def _load_and_bind_inputs(
    *,
    registry_path: Path,
    planner_receipt_path: Path,
    fixed_budget_path: Path,
    merged_scores_path: Path,
    run_attestation_path: Path,
    behavior_output_path: Path | None,
) -> dict[str, Any]:
    registry = _read_json(registry_path, "FN mechanism registry")
    if registry.get("schema_version") != REGISTRY_SCHEMA_VERSION or registry.get("unit_id") != UNIT_ID:
        raise MechanismAnalysisError("FN mechanism registry schema/unit mismatch")
    registry_digest = _validate_self_digest(registry, "registry_digest", "FN mechanism registry")

    planner = _read_json(planner_receipt_path, "planner receipt")
    if planner.get("schema_version") != PLANNER_RECEIPT_SCHEMA_VERSION or planner.get("unit_id") != UNIT_ID:
        raise MechanismAnalysisError("planner receipt schema/unit mismatch")
    planner_digest = _validate_self_digest(planner, "receipt_digest", "planner receipt")
    _validate_file_ref(
        _mapping(planner.get("sources"), "planner.sources").get("registry"),
        registry_path,
        "planner.sources.registry",
    )
    outputs = _mapping(planner.get("outputs"), "planner.outputs")
    ledger_ref = _mapping(
        outputs.get("owner_context_ledger"), "planner owner-context ledger output"
    )
    ledger_path = _resolved_file(
        str(ledger_ref.get("path", "")), "planner owner-context ledger"
    )
    _validate_file_ref(
        ledger_ref, ledger_path, "planner owner-context ledger output"
    )
    ledger_rows = _read_jsonl(ledger_path, "owner-context ledger")
    if ledger_ref.get("row_count") != len(ledger_rows):
        raise MechanismAnalysisError(
            "owner-context ledger row count does not match the planner receipt"
        )
    ledger_by_context = {
        _string(row.get("context_id"), "owner-context ledger context_id"): row
        for row in ledger_rows
    }
    if len(ledger_by_context) != len(ledger_rows):
        raise MechanismAnalysisError("owner-context ledger context ids are duplicated")
    _validate_file_ref(outputs.get("fixed_budget_candidates"), fixed_budget_path, "planner fixed-budget output")
    rules_ref = _mapping(outputs.get("landscape_decision_rules"), "planner decision rules output")
    rules_path = _resolved_file(str(rules_ref.get("path", "")), "planner decision rules")
    _validate_file_ref(rules_ref, rules_path, "planner decision rules output")
    rules = _read_json(rules_path, "landscape decision rules")

    mechanism_ref = _mapping(
        outputs.get("mechanism_decision_rules"),
        "planner mechanism decision rules output",
    )
    mechanism_rules_path = _resolved_file(
        str(mechanism_ref.get("path", "")), "mechanism decision rules"
    )
    _validate_file_ref(
        mechanism_ref,
        mechanism_rules_path,
        "planner mechanism decision rules output",
    )
    mechanism_rules = _read_json(mechanism_rules_path, "mechanism decision rules")
    if (
        mechanism_rules.get("schema_version")
        != MECHANISM_DECISION_RULES_SCHEMA_VERSION
        or mechanism_rules.get("unit_id") != UNIT_ID
    ):
        raise MechanismAnalysisError("mechanism decision rules schema/unit mismatch")
    mechanism_rules_digest = _validate_self_digest(
        mechanism_rules, "self_digest", "mechanism decision rules"
    )
    planner_sources = _mapping(planner.get("sources"), "planner.sources")
    rules_template_ref = _mapping(
        planner_sources.get("rules_template"), "planner rules-template source"
    )
    rules_template_path = _resolved_file(
        str(rules_template_ref.get("path", "")), "planner rules-template source"
    )
    _validate_file_ref(
        rules_template_ref, rules_template_path, "planner rules-template source"
    )
    upstream = _mapping(
        mechanism_rules.get("upstream_digests"),
        "mechanism decision rules upstream_digests",
    )
    if (
        upstream.get("execution_landscape_decision_rules_sha256")
        != sha256_file(rules_path)
        or upstream.get("fn_mechanism_registry_sha256") != registry_digest
        or upstream.get("rules_template_sha256") != rules_template_ref.get("sha256")
    ):
        raise MechanismAnalysisError(
            "mechanism decision rules parent execution-rules/registry/template binding mismatch"
        )
    if _mapping(mechanism_rules.get("geometry"), "mechanism rules geometry").get(
        "iou_thresholds"
    ) != list(IOU_THRESHOLDS):
        raise MechanismAnalysisError(
            "mechanism decision rules IoU thresholds differ from the analyzer"
        )
    calibration_rules = _mapping(
        mechanism_rules.get("calibration"), "mechanism rules calibration"
    )
    if (
        calibration_rules.get("quantile_algorithm") != "type7"
        or calibration_rules.get("lower_quantile") != LOWER_PROMINENCE_QUANTILE
        or calibration_rules.get("upper_quantile") != UPPER_RANK_QUANTILE
    ):
        raise MechanismAnalysisError(
            "mechanism decision rules calibration differs from the analyzer"
        )
    populations = _mapping(
        mechanism_rules.get("populations"), "mechanism rules populations"
    )
    if set(populations) != {"target", "decoy", "reference"}:
        raise MechanismAnalysisError(
            "mechanism decision rules do not freeze target/decoy/reference populations"
        )
    rung_quotas = _mapping(
        mechanism_rules.get("rung_quotas"), "mechanism rules rung_quotas"
    )
    scalar_rules = _mapping(
        rung_quotas.get("scalar_smoke"), "mechanism rules scalar_smoke"
    )
    l1_rules = _mapping(rung_quotas.get("L1"), "mechanism rules L1")
    if (
        scalar_rules.get("claim_direction") != "positive_only"
        or scalar_rules.get("target_count") != 30
        or scalar_rules.get("decoy_count") != "equal_to_target"
        or scalar_rules.get("reference_count") != 7
        or l1_rules.get("target_count") != 256
        or l1_rules.get("decoy_count") != "equal_to_target"
        or l1_rules.get("reference_count") != 65
    ):
        raise MechanismAnalysisError(
            "mechanism decision rules scalar/L1 reference or claim semantics drifted"
        )
    collision_statistics = _mapping(
        _mapping(
            mechanism_rules.get("collision"), "mechanism rules collision"
        ).get("statistics"),
        "mechanism rules collision statistics",
    )
    if set(collision_statistics) != {"F1", "F2", "F3"}:
        raise MechanismAnalysisError(
            "mechanism decision rules do not freeze the F1/F2/F3 collision band"
        )
    neighborhood_rules = _mapping(
        mechanism_rules.get("neighborhood"), "mechanism rules neighborhood"
    )
    if (
        neighborhood_rules.get("eligible_family") != "near_gt_micro"
        or neighborhood_rules.get("eligible_population") != "target"
        or neighborhood_rules.get("member_field")
        != "candidate_neighborhood_member"
        or neighborhood_rules.get("id_field") != "candidate_neighborhood_id"
    ):
        raise MechanismAnalysisError(
            "mechanism decision rules candidate-neighborhood semantics drifted"
        )
    singleton_rules = _mapping(
        mechanism_rules.get("exact_gt_singleton"),
        "mechanism rules exact-GT singleton",
    )
    if (
        singleton_rules.get("eligible_family") != "near_gt_micro"
        or singleton_rules.get("eligible_population") != "target"
        or singleton_rules.get("member_field")
        != "exact_gt_singleton_member"
        or singleton_rules.get("id_field") != "exact_gt_singleton_id"
        or singleton_rules.get("id_rule") != "exact-gt:<gt_owner_id>:<rung>"
    ):
        raise MechanismAnalysisError(
            "mechanism decision rules exact-GT-singleton semantics drifted"
        )
    neighborhood_assertion = _mapping(
        planner.get("neighborhood_consistency"),
        "planner neighborhood consistency assertion",
    )
    if neighborhood_assertion.get("status") != "consistent":
        raise MechanismAnalysisError(
            "planner receipt lacks a passed candidate-neighborhood consistency assertion"
        )

    attestation = _read_json(run_attestation_path, "run attestation")
    if attestation.get("schema_version") != RUN_ATTESTATION_SCHEMA_VERSION or attestation.get("unit_id") != UNIT_ID:
        raise MechanismAnalysisError("run attestation schema/unit mismatch")
    if attestation.get("disposition") != "accepted":
        raise MechanismAnalysisError("run attestation is not accepted")
    if attestation.get("run_mode") not in {"scalar_smoke", "scale"}:
        raise MechanismAnalysisError("run attestation has an unknown v2 run_mode")
    channel = _mapping(
        attestation.get("decision_channel_attestation"),
        "decision channel attestation",
    )
    if channel.get("decision_bearing_channel") != SUCCESSOR_DECISION_CHANNEL:
        raise MechanismAnalysisError("run attestation does not bind the raw complete-box channel")
    rp_stratum = _mapping(channel.get("repetition_penalty_stratum"), "rp stratum")
    if float(rp_stratum.get("value", -1)) != 1.0 or rp_stratum.get("status") != "passed":
        raise MechanismAnalysisError("run attestation is not in the raw rp=1.0 mechanism stratum")
    channel_schema = _mapping(
        channel.get("successor_scorer_schema"),
        "decision channel successor scorer schema",
    )
    expected_scorer_schema = {
        "row_schema_version": SUCCESSOR_SCORE_ROW_SCHEMA_VERSION,
        "receipt_schema_version": SUCCESSOR_SCORE_RECEIPT_SCHEMA_VERSION,
    }
    if dict(channel_schema) != expected_scorer_schema:
        raise MechanismAnalysisError(
            "run attestation does not bind the current successor scorer schemas"
        )
    raw_channel_declaration = channel.get("raw_channel_declaration")
    if (
        not isinstance(raw_channel_declaration, str)
        or "fp32" not in raw_channel_declaration
    ):
        raise MechanismAnalysisError(
            "run attestation does not preserve the v2 fp32 raw-channel declaration"
        )
    backend = _mapping(
        attestation.get("scalar_acceptance_backend"),
        "scalar acceptance backend",
    )
    if backend.get("cache_enabled") is not False or backend.get("use_cache") is not False:
        raise MechanismAnalysisError(
            "run attestation does not bind an uncached full-prefix reforward"
        )
    if attestation["run_mode"] == "scalar_smoke":
        if (
            backend.get("required") is not True
            or backend.get("status") != "passed"
            or backend.get("effective_batch_sizes") != [1]
        ):
            raise MechanismAnalysisError(
                "scalar-smoke attestation does not carry the v2 scalar backend admission"
            )
    elif (
        backend.get("required") is not False
        or backend.get("status") != "not_applicable_scale_mode"
    ):
        raise MechanismAnalysisError(
            "scale attestation does not carry the v2 scale backend disposition"
        )

    merge_ref = _mapping(attestation.get("merge_receipt"), "attestation.merge_receipt")
    if merge_ref.get("schema_version") != MERGE_RECEIPT_SCHEMA_VERSION:
        raise MechanismAnalysisError(
            "run attestation does not bind the current merge receipt schema"
        )
    merge_path = _resolved_file(str(merge_ref.get("path", "")), "merged score receipt")
    _validate_file_ref(merge_ref, merge_path, "attestation.merge_receipt")
    merge = _read_json(merge_path, "merge receipt")
    if merge.get("schema_version") != MERGE_RECEIPT_SCHEMA_VERSION or merge.get("unit_id") != UNIT_ID:
        raise MechanismAnalysisError("merge receipt schema/unit mismatch")
    if merge.get("generic_arbitrary_role_merger") is not True:
        raise MechanismAnalysisError(
            "merge receipt is not the current generic arbitrary-role merger"
        )
    merge_channel = _mapping(merge.get("decision_channel"), "merge decision channel")
    if (
        merge_channel.get("name") != SUCCESSOR_DECISION_CHANNEL
        or float(merge_channel.get("primary_repetition_penalty_stratum", -1)) != 1.0
        or merge_channel.get("auxiliary_policy_is_not_a_model_likelihood") is not True
    ):
        raise MechanismAnalysisError(
            "merge receipt does not preserve the v2 raw-rp1 decision channel"
        )
    merge_scorer_provenance = _mapping(
        merge.get("successor_scorer_provenance"),
        "merge successor scorer provenance",
    )
    attested_scorer_provenance = _mapping(
        attestation.get("successor_scorer_provenance"),
        "attestation successor scorer provenance",
    )
    expected_scorer_provenance = {
        **expected_scorer_schema,
        "unit_id": UNIT_ID,
    }
    for key, expected in expected_scorer_provenance.items():
        if (
            merge_scorer_provenance.get(key) != expected
            or attested_scorer_provenance.get(key) != expected
        ):
            raise MechanismAnalysisError(
                "merge/attestation successor scorer provenance mismatch"
            )
    if dict(attested_scorer_provenance) != dict(merge_scorer_provenance):
        raise MechanismAnalysisError(
            "attestation does not preserve the merger's successor scorer provenance"
        )
    selected_rungs = [
        _string(value, "merge selected_rungs[]")
        for value in _sequence(merge.get("selected_rungs"), "merge selected_rungs")
    ]
    if not selected_rungs or len(selected_rungs) != len(set(selected_rungs)):
        raise MechanismAnalysisError(
            "merge selected_rungs are empty or duplicated"
        )
    attested_selected_rungs = [
        _string(value, "attestation selected_rungs[]")
        for value in _sequence(
            attestation.get("selected_rungs"), "attestation selected_rungs"
        )
    ]
    if attested_selected_rungs != selected_rungs:
        raise MechanismAnalysisError(
            "attestation selected_rungs do not exactly match the merge receipt"
        )
    sources = _mapping(merge.get("source_digests"), "merge source digests")
    _validate_file_ref(sources.get("fn_mechanism_registry"), registry_path, "merge registry source")
    _validate_file_ref(sources.get("fixed_budget_candidates"), fixed_budget_path, "merge fixed-budget source")
    merge_mechanism_ref = _mapping(
        sources.get("mechanism_decision_rules"),
        "merge mechanism decision rules source",
    )
    _validate_file_ref(
        merge_mechanism_ref,
        mechanism_rules_path,
        "merge mechanism decision rules source",
    )
    if (
        merge_mechanism_ref.get("self_digest") != mechanism_rules_digest
        or merge_mechanism_ref.get("parent_execution_rules_sha256")
        != sha256_file(rules_path)
    ):
        raise MechanismAnalysisError(
            "merge mechanism-rules self/parent digest binding mismatch"
        )
    merge_planner = _mapping(merge.get("planner_receipt"), "merge planner receipt")
    _validate_file_ref(merge_planner, planner_receipt_path, "merge planner receipt")
    if merge_planner.get("receipt_digest") != planner_digest:
        raise MechanismAnalysisError("merge receipt binds a different planner receipt_digest")
    decision_ref = _mapping(sources.get("decision_rules"), "merge decision rules source")
    _validate_file_ref(decision_ref, rules_path, "merge decision rules source")
    if attestation.get("decision_rules_sha256") != sha256_file(rules_path):
        raise MechanismAnalysisError("attestation decision-rules digest mismatch")

    score_ref = _mapping(
        _mapping(merge.get("output_artifacts"), "merge outputs").get("merged_scores"),
        "merge merged_scores",
    )
    _validate_file_ref(score_ref, merged_scores_path, "merge merged scores")
    attested_scores = _mapping(attestation.get("merged_scores"), "attestation merged_scores")
    _validate_file_ref(attested_scores, merged_scores_path, "attestation merged scores")
    if score_ref.get("row_count") != attested_scores.get("row_count"):
        raise MechanismAnalysisError("score row-count bindings disagree")
    mechanism_attestation = _mapping(
        attestation.get("mechanism_decision_rules_attestation"),
        "attestation mechanism decision rules",
    )
    if (
        Path(str(mechanism_attestation.get("path", ""))).expanduser().resolve()
        != mechanism_rules_path
        or mechanism_attestation.get("sha256") != sha256_file(mechanism_rules_path)
        or mechanism_attestation.get("schema_version")
        != MECHANISM_DECISION_RULES_SCHEMA_VERSION
        or mechanism_attestation.get("self_digest") != mechanism_rules_digest
        or mechanism_attestation.get("parent_execution_rules_sha256")
        != sha256_file(rules_path)
        or mechanism_attestation.get("fn_mechanism_registry_sha256")
        != registry_digest
        or mechanism_attestation.get("exact_gt_singleton_binding")
        != dict(singleton_rules)
    ):
        raise MechanismAnalysisError(
            "run attestation mechanism-rules binding mismatch"
        )

    fixed_rows = _attach_candidate_boxes(
        _read_jsonl(fixed_budget_path, "fixed-budget candidates"), rules
    )
    mechanism_rules_file_sha256 = sha256_file(mechanism_rules_path)
    if any(
        row.get("mechanism_decision_rules_sha256")
        != mechanism_rules_file_sha256
        for row in fixed_rows
    ):
        raise MechanismAnalysisError(
            "fixed-budget candidates do not all bind the loaded mechanism decision rules"
        )
    fixed_ref = _mapping(outputs.get("fixed_budget_candidates"), "planner fixed-budget output")
    if fixed_ref.get("row_count") != len(fixed_rows):
        raise MechanismAnalysisError("fixed-budget row count does not match the planner receipt")
    score_rows = _read_jsonl(merged_scores_path, "merged score rows")
    if len(score_rows) != score_ref.get("row_count"):
        raise MechanismAnalysisError("merged score row count does not match its receipt")
    observed_population_counts = dict(
        sorted(Counter(str(row.get("population")) for row in score_rows).items())
    )
    if mechanism_attestation.get("population_counts") != observed_population_counts:
        raise MechanismAnalysisError(
            "run attestation population counts do not match merged score rows"
        )
    selected_contexts = set(
        str(value)
        for value in _sequence(
            _mapping(merge.get("context_selection"), "merge context selection").get("selected_context_ids"),
            "selected context ids",
        )
    )
    expected_keys = {
        (str(row["owner_context_id"]), str(row["candidate_id"]))
        for row in fixed_rows
        if str(row["owner_context_id"]) in selected_contexts
        and str(row.get("rung")) in set(selected_rungs)
    }
    observed: dict[tuple[str, str], float] = {}
    for row in score_rows:
        if (
            row.get("schema_version") != SUCCESSOR_SCORE_ROW_SCHEMA_VERSION
            or row.get("unit_id") != UNIT_ID
        ):
            raise MechanismAnalysisError(
                "merged score row schema/unit is not the current successor scorer"
            )
        if float(row.get("native_repetition_penalty_stratum", -1)) != PRIMARY_REPETITION_PENALTY:
            raise MechanismAnalysisError("score row is outside the raw rp=1.0 mechanism stratum")
        key = (_string(row.get("context_id"), "score context_id"), _string(row.get("candidate_id"), "score candidate_id"))
        if key in observed:
            raise MechanismAnalysisError(f"duplicate merged score key: {key}")
        raw = _mapping(row.get("raw_model_logprob"), "score raw_model_logprob")
        observed[key] = _finite(raw.get("complete_box_logprob_sum"), "raw complete-box score")
    if set(observed) != expected_keys:
        missing = sorted(expected_keys - set(observed))
        extra = sorted(set(observed) - expected_keys)
        raise MechanismAnalysisError(
            f"analyzer score/candidate join is not exact; missing={missing[:8]} extra={extra[:8]}"
        )

    behavior = None
    behavior_landscape_admission = None
    if behavior_output_path is not None:
        behavior = _read_json(behavior_output_path, "behavior output")
        if behavior.get("schema_version") != BEHAVIOR_SCHEMA_VERSION or behavior.get("unit_id") != UNIT_ID:
            raise MechanismAnalysisError("behavior output schema/unit mismatch")
        _validate_self_digest(behavior, "output_content_sha256", "behavior output")
        contract = _mapping(behavior.get("contract"), "behavior contract")
        behavior_registry = _mapping(contract.get("registry"), "behavior contract registry")
        if behavior_registry.get("file_sha256") != sha256_file(registry_path) or behavior_registry.get("registry_digest") != registry_digest:
            raise MechanismAnalysisError("behavior output binds a different FN registry")
        landscape = _mapping(contract.get("landscape"), "behavior landscape receipt")
        if landscape.get("sha256") != sha256_file(merge_path):
            raise MechanismAnalysisError("behavior output binds a different landscape merge receipt")
        roles_by_id = {str(role["role_id"]): role for role in _all_registry_roles(registry)}
        selected_prefixes = [
            _mapping(value, "behavior selected role prefix")
            for value in _sequence(
                behavior.get("selected_role_prefixes", []), "behavior selected role prefixes"
            )
        ]
        selected_ids = [str(value.get("role_id")) for value in selected_prefixes]
        if len(selected_ids) != len(set(selected_ids)):
            raise MechanismAnalysisError("behavior selected role prefixes are duplicated")
        for value in selected_prefixes:
            role_id = str(value.get("role_id"))
            role = roles_by_id.get(role_id)
            if role is None:
                raise MechanismAnalysisError(f"behavior output names unknown role {role_id!r}")
            if value.get("full_prefix_token_ids_sha256") != _role_prefix_digest(role):
                raise MechanismAnalysisError(
                    f"behavior role {role_id!r} full-prefix digest disagrees with the FN registry"
                )
        admission_ref = _mapping(
            contract.get("landscape_admission"), "behavior landscape admission receipt"
        )
        admission_path = _resolved_file(
            str(admission_ref.get("path", "")), "behavior landscape admission"
        )
        _validate_file_ref(
            admission_ref, admission_path, "behavior landscape admission receipt"
        )
        behavior_landscape_admission = _read_json(
            admission_path, "behavior landscape admission"
        )
        if (
            behavior_landscape_admission.get("schema_version")
            != BEHAVIOR_LANDSCAPE_ADMISSION_SCHEMA_VERSION
            or behavior_landscape_admission.get("status") != "passed"
            or behavior_landscape_admission.get("registry_digest") != registry_digest
            or behavior_landscape_admission.get("landscape_receipt_sha256")
            != sha256_file(merge_path)
        ):
            raise MechanismAnalysisError(
                "behavior landscape admission schema/status/identity binding mismatch"
            )
        expected_mechanism_binding = {
            "sha256": sha256_file(mechanism_rules_path),
            "self_digest": mechanism_rules_digest,
            "parent_execution_rules_sha256": sha256_file(rules_path),
            "planner_receipt_digest": planner_digest,
            "registry_digest": registry_digest,
            "fixed_budget_candidates_sha256": sha256_file(fixed_budget_path),
        }
        if behavior_landscape_admission.get(
            "mechanism_decision_rules"
        ) != expected_mechanism_binding:
            raise MechanismAnalysisError(
                "behavior landscape admission mechanism-rules binding mismatch"
            )
        admitted_roles = [
            _mapping(value, "behavior landscape admitted role")
            for value in _sequence(
                behavior_landscape_admission.get("admitted_roles", []),
                "behavior landscape admitted roles",
            )
        ]
        admission_by_role = {str(value.get("role_id")): value for value in admitted_roles}
        contract_selected_ids = {
            str(value)
            for value in _sequence(
                contract.get("selected_role_ids", []), "behavior selected role ids"
            )
        }
        receipt_admitted_ids = {
            str(value)
            for value in _sequence(
                admission_ref.get("admitted_role_ids", []),
                "behavior landscape admission receipt role ids",
            )
        }
        if (
            not contract_selected_ids <= set(admission_by_role)
            or set(selected_ids) != contract_selected_ids
            or receipt_admitted_ids != contract_selected_ids
        ):
            raise MechanismAnalysisError(
                "behavior roles, prefix bindings, and landscape admissions do not exactly join"
            )
        for role_id in sorted(contract_selected_ids):
            entry = admission_by_role[role_id]
            role = roles_by_id[role_id]
            condition = _mapping(
                entry.get("landscape_condition"),
                f"behavior landscape admission {role_id}.landscape_condition",
            )
            if (
                entry.get("status") != "passed"
                or entry.get("gt_owner_id") != role.get("gt_owner_id")
                or entry.get("context_id") != f"ctx:fn:{role_id}"
                or condition.get("status") != "passed"
                or condition.get("name")
                not in {
                    "usable_target_strict_region_peak",
                    "multiple_separated_owner_localized_peaks",
                }
            ):
                raise MechanismAnalysisError(
                    f"behavior role {role_id!r} lacks an exact passed landscape admission"
                )
        equivalence_receipt = _mapping(
            contract.get("description_equivalence_receipt"),
            "behavior description equivalence receipt",
        )
        eligible = equivalence_receipt.get("semantic_drift_decision_eligible")
        if not isinstance(eligible, bool):
            raise MechanismAnalysisError(
                "behavior description equivalence receipt eligibility must be a boolean"
            )
        if eligible:
            if equivalence_receipt.get("status") != "frozen":
                raise MechanismAnalysisError(
                    "behavior description equivalence receipt status is not frozen"
                )
            equivalence_path = _resolved_file(
                str(equivalence_receipt.get("path", "")),
                "behavior description equivalence",
            )
            _validate_file_ref(
                equivalence_receipt,
                equivalence_path,
                "behavior description equivalence receipt",
            )
            equivalence_document = _read_json(
                equivalence_path, "behavior description equivalence"
            )
            if (
                equivalence_document.get("schema_version")
                != DESCRIPTION_EQUIVALENCE_SCHEMA_VERSION
                or equivalence_document.get("status") != "frozen"
            ):
                raise MechanismAnalysisError(
                    "behavior description equivalence schema/status is not frozen"
                )
            _validate_self_digest(
                equivalence_document,
                "equivalence_digest",
                "behavior description equivalence",
            )
            if equivalence_receipt.get("equivalence_digest") != equivalence_document.get(
                "equivalence_digest"
            ):
                raise MechanismAnalysisError(
                    "behavior description equivalence receipt digest does not match "
                    "the frozen document"
                )
            raw_projection = _mapping(
                contract.get("description_equivalence"),
                "behavior description equivalence raw projection",
            )
            if {
                key: value
                for key, value in raw_projection.items()
                if key != "alias_to_category"
            } != equivalence_document:
                raise MechanismAnalysisError(
                    "behavior contract's raw description-equivalence projection "
                    "diverges from the frozen document"
                )
            reconstructed_alias_to_category = _reconstruct_alias_to_category(
                _mapping(
                    equivalence_document.get("categories"),
                    "behavior description equivalence categories",
                )
            )
            if reconstructed_alias_to_category is None:
                raise MechanismAnalysisError(
                    "behavior description equivalence categories cannot be "
                    "deterministically reconstructed into an alias_to_category map"
                )
            if raw_projection.get("alias_to_category") != reconstructed_alias_to_category:
                raise MechanismAnalysisError(
                    "behavior contract's alias_to_category diverges from the "
                    "deterministic reconstruction of the frozen document's categories"
                )
        elif equivalence_receipt.get("status") != "not_supplied_semantic_relation_neutral":
            raise MechanismAnalysisError(
                "behavior description equivalence receipt status/eligibility mismatch"
            )

    tolerance = _extract_scalar_tolerance(attestation, rules)
    return {
        "registry": registry,
        "registry_digest": registry_digest,
        "planner": planner,
        "planner_digest": planner_digest,
        "rules": rules,
        "rules_path": rules_path,
        "mechanism_rules": mechanism_rules,
        "mechanism_rules_path": mechanism_rules_path,
        "mechanism_rules_digest": mechanism_rules_digest,
        "ledger_by_context": ledger_by_context,
        "merge": merge,
        "merge_path": merge_path,
        "attestation": attestation,
        "fixed_rows": fixed_rows,
        "fixed_budget_path": fixed_budget_path,
        "score_rows": score_rows,
        "scores": observed,
        "selected_contexts": selected_contexts,
        "selected_rungs": frozenset(selected_rungs),
        "behavior": behavior,
        "behavior_output_path": behavior_output_path,
        "behavior_landscape_admission": behavior_landscape_admission,
        "tolerance": tolerance,
    }


def _context_family(role_kind: str) -> str:
    if role_kind in {"root_context"}:
        return "root"
    if role_kind in {"due_turn_context", "first_skip_pre", "first_skip_post"}:
        return "due_turn_or_transition"
    if "collision" in role_kind or "null" in role_kind:
        return "sampled_collision_context"
    return role_kind


def _calibration_stratum(
    candidate: Mapping[str, Any], target: Mapping[str, Any] | None, role: Mapping[str, Any]
) -> str:
    explicit = candidate.get(
        "description_size_crowding_stratum", candidate.get("calibration_stratum")
    )
    if isinstance(explicit, str) and explicit:
        return explicit
    if target is not None and isinstance(target.get("stratum"), str):
        return str(target["stratum"])
    trajectory = _mapping(role.get("trajectory", {}), "role trajectory")
    return f"unmatched:{trajectory.get('image_id', 'unknown')}"


def _validate_l2_admission(
    admission_path: Path | None,
    *,
    registry_digest: str,
    planner_digest: str,
) -> set[str]:
    if admission_path is None:
        return set()
    document = _read_json(admission_path, "L2 prospective admission")
    if document.get("schema_version") != L2_ADMISSION_SCHEMA_VERSION or document.get("status") != "passed":
        raise MechanismAnalysisError("L2 admission schema/status is not passed")
    if document.get("registry_digest") != registry_digest or document.get("planner_receipt_digest") != planner_digest:
        raise MechanismAnalysisError("L2 admission identity binding mismatch")
    contexts = [str(value) for value in _sequence(document.get("admitted_context_ids"), "L2 admitted contexts")]
    if not contexts or len(contexts) != len(set(contexts)):
        raise MechanismAnalysisError("L2 admitted contexts are empty or duplicated")
    return set(contexts)


def _build_landscape_evidence(
    loaded: Mapping[str, Any], *, l2_admitted_contexts: set[str]
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    registry = loaded["registry"]
    roles = _all_registry_roles(registry)
    roles_by_context = {f"ctx:fn:{role['role_id']}": role for role in roles}
    targets = {
        str(row["gt_owner_id"]): row
        for row in _sequence(
            _mapping(registry.get("mechanism_cohort"), "mechanism cohort").get("targets"),
            "mechanism targets",
        )
    }
    fixed_by_group: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in loaded["fixed_rows"]:
        context_id = str(row["owner_context_id"])
        if context_id not in loaded["selected_contexts"]:
            continue
        rung = str(row.get("rung"))
        if rung not in loaded["selected_rungs"]:
            continue
        if rung == "L2" and context_id not in l2_admitted_contexts:
            raise MechanismAnalysisError(
                f"L2 scores for {context_id!r} lack a prospective passed admission"
            )
        fixed_by_group[(context_id, rung)].append(row)

    entries: list[dict[str, Any]] = []
    by_context_rung: dict[tuple[str, str], dict[str, Any]] = {}
    for (context_id, rung), candidates in sorted(fixed_by_group.items()):
        role = roles_by_context.get(context_id)
        if role is None:
            raise MechanismAnalysisError(f"scored context {context_id!r} is absent from the FN registry")
        owner_id = str(role["gt_owner_id"])
        ledger_row = loaded["ledger_by_context"].get(context_id)
        if ledger_row is None:
            raise MechanismAnalysisError(
                f"scored context {context_id!r} is absent from the owner-context ledger"
            )
        reference_status = _string(
            ledger_row.get("other_owner_reference_status"),
            f"owner-context ledger {context_id}.other_owner_reference_status",
        )
        if reference_status.startswith("bound:"):
            reference_bound = True
        elif reference_status == "no_same_description_non_overlapping_owner":
            reference_bound = False
        else:
            raise MechanismAnalysisError(
                f"owner-context ledger {context_id!r} has an unknown other-owner reference status"
            )
        target_gt_box = _sequence(
            _mapping(ledger_row.get("ground_truth"), "ledger ground_truth").get(
                "box"
            ),
            "ledger ground-truth box",
        )
        score_map = {
            str(row["candidate_id"]): loaded["scores"][(context_id, str(row["candidate_id"]))]
            for row in candidates
        }
        per_threshold = {
            str(threshold): compute_landscape_statistics(
                candidates,
                score_map,
                iou_threshold=threshold,
                other_owner_reference_bound=reference_bound,
                target_gt_box=[int(value) for value in target_gt_box],
                target_gt_owner_id=owner_id,
            )
            for threshold in IOU_THRESHOLDS
        }
        first = candidates[0]
        if any(
            row.get("control_kind") != first.get("control_kind")
            or row.get("matched_control_group") != first.get("matched_control_group")
            or row.get("is_control") != first.get("is_control")
            for row in candidates
        ):
            raise MechanismAnalysisError("candidate metadata changes inside one context/rung")
        peak_ids = [per_threshold[str(threshold)]["target_peak_candidate_id"] for threshold in IOU_THRESHOLDS]
        prominence_signs = [
            None
            if per_threshold[str(threshold)]["background_prominence"] is None
            else per_threshold[str(threshold)]["background_prominence"] > loaded["tolerance"]["effective_tolerance"]
            for threshold in IOU_THRESHOLDS
        ]
        entry = {
            "schema_version": LANDSCAPE_EVIDENCE_SCHEMA_VERSION,
            "unit_id": UNIT_ID,
            "context_id": context_id,
            "role_id": role["role_id"],
            "role_kind": role["role_kind"],
            "context_family": _context_family(str(role["role_kind"])),
            "gt_owner_id": owner_id,
            "rung": rung,
            "is_control": bool(first.get("is_control")),
            "control_kind": first.get("control_kind"),
            "matched_control_group": first.get("matched_control_group"),
            "description_size_crowding_stratum": _calibration_stratum(
                first, targets.get(owner_id), role
            ),
            "raw_score_channel": "raw_model_logprob.complete_box_logprob_sum",
            "native_repetition_penalty_stratum": PRIMARY_REPETITION_PENALTY,
            "per_iou_threshold": per_threshold,
            "iou_band_stability": {
                "peak_identity_stable": len(set(peak_ids)) == 1 and peak_ids[0] is not None,
                "positive_prominence_side_stable": (
                    None not in prominence_signs and len(set(prominence_signs)) == 1
                ),
                "thresholds": list(IOU_THRESHOLDS),
            },
        }
        entries.append(entry)
        by_context_rung[(context_id, rung)] = entry
    return entries, by_context_rung


def _apply_control_calibration(entries: list[dict[str, Any]], tolerance: float) -> list[dict[str, Any]]:
    calibration_values: dict[tuple[str, str, str], dict[str, list[Any]]] = defaultdict(
        lambda: {"prominence": [], "rank": [], "control_ids": []}
    )
    primary_key = str(PRIMARY_IOU_THRESHOLD)
    for entry in entries:
        if entry["rung"] == "scalar_smoke":
            continue
        if not entry["is_control"] or entry["control_kind"] not in POSITIVE_CONTROL_KINDS:
            continue
        stats = entry["per_iou_threshold"][primary_key]
        prominence = stats["background_prominence"]
        rank = stats["localized_rank"]
        if prominence is None or rank is None:
            continue
        key = (
            entry["description_size_crowding_stratum"],
            entry["context_family"],
            entry["rung"],
        )
        calibration_values[key]["prominence"].append(float(prominence))
        calibration_values[key]["rank"].append(float(rank))
        calibration_values[key]["control_ids"].append(
            f"{entry['context_id']}:{entry['control_kind']}"
        )

    table = []
    for key in sorted(
        {
            (
                entry["description_size_crowding_stratum"],
                entry["context_family"],
                entry["rung"],
            )
            for entry in entries
        }
    ):
        if key[2] == "scalar_smoke":
            table.append(
                {
                    "description_size_crowding_stratum": key[0],
                    "context_family": key[1],
                    "rung": key[2],
                    "status": "positive_sign_only_scalar_smoke",
                    "prominence_lower_quantile": None,
                    "localized_rank_upper_quantile": None,
                    "control_ids": [],
                }
            )
            continue
        values = calibration_values.get(key)
        if values is None or not values["prominence"]:
            table.append(
                {
                    "description_size_crowding_stratum": key[0],
                    "context_family": key[1],
                    "rung": key[2],
                    "status": "unresolved_missing_matched_controls",
                    "prominence_lower_quantile": None,
                    "localized_rank_upper_quantile": None,
                    "control_ids": [],
                }
            )
            continue
        table.append(
            {
                "description_size_crowding_stratum": key[0],
                "context_family": key[1],
                "rung": key[2],
                "status": "calibrated",
                "quantile_algorithm": "linear_type7",
                "prominence_quantile": LOWER_PROMINENCE_QUANTILE,
                "rank_quantile": UPPER_RANK_QUANTILE,
                "prominence_lower_quantile": quantile_linear(
                    values["prominence"], LOWER_PROMINENCE_QUANTILE
                ),
                "localized_rank_upper_quantile": quantile_linear(
                    values["rank"], UPPER_RANK_QUANTILE
                ),
                "control_ids": sorted(values["control_ids"]),
            }
        )
    calibration_by_key = {
        (row["description_size_crowding_stratum"], row["context_family"], row["rung"]): row
        for row in table
    }
    for entry in entries:
        key = (
            entry["description_size_crowding_stratum"],
            entry["context_family"],
            entry["rung"],
        )
        calibration = calibration_by_key[key]
        stats = entry["per_iou_threshold"][primary_key]
        if entry["rung"] == "scalar_smoke":
            prominence = stats.get("background_prominence")
            positive = bool(
                prominence is not None and float(prominence) > float(tolerance)
            )
            entry["matched_control_calibration"] = deepcopy(calibration)
            entry["usable_target_support"] = False
            entry["usable_target_support_reasons"] = [
                "scalar_smoke_is_positive_sign_only_and_never_quantile_calibrated"
            ]
            entry["scalar_smoke_interpretation"] = {
                "claim_direction": "positive_only",
                "positive_sign_observed": positive,
                "negative_mechanism_eligible": False,
                "quantile_calibration_eligible": False,
                "l1_admission": {
                    "status": (
                        "positive_sign_observed"
                        if positive
                        else "L1_required"
                    ),
                    "reason": (
                        "scalar_positive_sign_is_machinery_or_sign_sensitivity_only"
                        if positive
                        else "scalar_margin_is_marginal_or_negative_and_cannot_decide_stop_rule_4"
                    ),
                },
            }
            continue
        reasons = []
        usable = False
        if calibration["status"] != "calibrated":
            reasons.append("missing_matched_description_size_crowding_control_no_pooling")
        elif not stats["equal_count_guard"]:
            reasons.append("target_decoy_counts_unequal")
        elif stats["background_prominence"] is None or stats["localized_rank"] is None:
            reasons.append("target_or_background_statistic_missing")
        else:
            prominence = float(stats["background_prominence"])
            rank = float(stats["localized_rank"])
            if prominence <= tolerance:
                reasons.append("background_prominence_does_not_exceed_scalar_tolerance")
            if prominence < float(calibration["prominence_lower_quantile"]):
                reasons.append("background_prominence_below_matched_control_quantile")
            if rank > float(calibration["localized_rank_upper_quantile"]):
                reasons.append("localized_rank_above_matched_control_quantile")
            if stats["other_owner_margin"] is not None and float(stats["other_owner_margin"]) <= tolerance:
                reasons.append("other_owner_margin_does_not_exceed_scalar_tolerance")
            if not entry["iou_band_stability"]["peak_identity_stable"]:
                reasons.append("target_peak_identity_not_stable_across_iou_band")
            if not entry["iou_band_stability"]["positive_prominence_side_stable"]:
                reasons.append("prominence_side_not_stable_across_iou_band")
            usable = not reasons
        entry["matched_control_calibration"] = deepcopy(calibration)
        entry["usable_target_support"] = usable
        entry["usable_target_support_reasons"] = (
            ["all_matched_control_rank_prominence_margin_and_iou_band_gates_passed"]
            if usable
            else reasons
        )
    for calibration in table:
        key = (
            calibration["description_size_crowding_stratum"],
            calibration["context_family"],
            calibration["rung"],
        )
        controls = [
            entry
            for entry in entries
            if (
                entry["description_size_crowding_stratum"],
                entry["context_family"],
                entry["rung"],
            )
            == key
            and entry["is_control"]
            and entry["control_kind"] in POSITIVE_CONTROL_KINDS
        ]
        calibration["positive_control_outcomes"] = {
            f"{entry['context_id']}:{entry['control_kind']}": bool(
                entry["usable_target_support"]
            )
            for entry in controls
        }
        calibration["positive_controls_pass"] = bool(controls) and all(
            entry["usable_target_support"] for entry in controls
        )
    calibration_by_key = {
        (row["description_size_crowding_stratum"], row["context_family"], row["rung"]): row
        for row in table
    }
    for entry in entries:
        key = (
            entry["description_size_crowding_stratum"],
            entry["context_family"],
            entry["rung"],
        )
        calibration = calibration_by_key[key]
        entry["matched_control_calibration"] = deepcopy(calibration)
        if (
            not entry["is_control"]
            and entry["usable_target_support"]
            and not calibration["positive_controls_pass"]
        ):
            entry["usable_target_support"] = False
            entry["usable_target_support_reasons"] = [
                "matched_positive_control_failed_or_missing"
            ]
    return table


def _role_results_by_id(
    behavior: Mapping[str, Any] | None,
    landscape_admission: Mapping[str, Any] | None,
    registry: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    if behavior is None:
        return {}
    primary_views = [
        view
        for view in _sequence(behavior.get("policy_views", []), "behavior policy views")
        if float(_mapping(view, "policy view").get("repetition_penalty", -1)) == 1.0
    ]
    if len(primary_views) != 1:
        raise MechanismAnalysisError("behavior output must contain exactly one rp=1.0 policy view")
    roles = _sequence(_mapping(primary_views[0], "primary policy view").get("roles"), "behavior roles")
    admission_by_role = {
        str(entry.get("role_id")): entry
        for entry in (
            []
            if landscape_admission is None
            else [
                _mapping(value, "behavior admitted role")
                for value in _sequence(
                    landscape_admission.get("admitted_roles", []),
                    "behavior admitted roles",
                )
            ]
        )
    }
    equivalence = _mapping(
        _mapping(behavior.get("contract"), "behavior contract").get(
            "description_equivalence_receipt", {}
        ),
        "behavior description equivalence receipt",
    )
    registry_roles = {str(role["role_id"]): role for role in _all_registry_roles(registry)}
    sources = _mapping(registry.get("sources"), "registry sources")
    prediction_ref = _mapping(
        sources.get("prediction_row_ledger"), "registry prediction row ledger"
    )
    prediction_path = _resolved_file(
        str(prediction_ref.get("path", "")), "registry prediction row ledger"
    )
    _validate_file_ref(
        prediction_ref, prediction_path, "registry prediction row ledger"
    )
    prediction_by_id = {
        str(row.get("pred_row_id")): row
        for row in _read_jsonl(prediction_path, "registry prediction row ledger")
    }
    result = {}
    for raw_role in roles:
        role = dict(_mapping(raw_role, "behavior role"))
        role["_behavior_role_content_sha256"] = sha256_json(role)
        role_id = str(role["role_id"])
        admission = admission_by_role.get(role_id)
        if admission is not None:
            role["_validated_landscape_admission"] = deepcopy(dict(admission))
        registry_role = registry_roles.get(role_id)
        if registry_role is None:
            raise MechanismAnalysisError(f"behavior output names foreign role {role_id!r}")
        prefix = _mapping(registry_role.get("prefix"), f"registry role {role_id}.prefix")
        prefix_owner_ids = set()
        for pred_row_id in _sequence(
            prefix.get("prefix_pred_row_ids", []),
            f"registry role {role_id}.prefix_pred_row_ids",
        ):
            prediction = prediction_by_id.get(str(pred_row_id))
            if prediction is None:
                raise MechanismAnalysisError(
                    f"registry role {role_id!r} prefix names unknown prediction row {pred_row_id!r}"
                )
            owner = prediction.get("strict_match_gt_owner_id")
            if owner is not None:
                prefix_owner_ids.add(str(owner))
        role["_validated_prefix_owner_ids"] = sorted(prefix_owner_ids)
        role["_validated_semantic_drift_eligible"] = bool(
            equivalence.get("semantic_drift_decision_eligible")
            and equivalence.get("status") == "frozen"
        )
        result[role_id] = role
    if len(result) != len(roles):
        raise MechanismAnalysisError("behavior output has duplicate role ids")
    return result


def _passed_role_landscape_admission(role: Mapping[str, Any]) -> bool:
    admission = role.get("_validated_landscape_admission")
    return isinstance(admission, Mapping) and admission.get("status") == "passed"


def _autonomous_suffix(arm: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    """Validate and return the runner's exact autonomous suffix partition."""
    if not isinstance(arm, Mapping):
        return None
    rows = arm.get("rows")
    released = arm.get("released_suffix_rows")
    autonomous_ids = arm.get("autonomous_evidence_row_ids")
    if (
        isinstance(rows, (str, bytes))
        or not isinstance(rows, Sequence)
        or isinstance(released, (str, bytes))
        or not isinstance(released, Sequence)
        or isinstance(autonomous_ids, (str, bytes))
        or not isinstance(autonomous_ids, Sequence)
        or "intervention" not in arm
    ):
        return None
    forced = arm.get("arm_kind") == "forced_description"
    expected_released = list(rows[1:] if forced else rows)
    if list(released) != expected_released:
        return None
    expected_ids = [
        str(_mapping(row, "autonomous released row").get("pred_row_id"))
        for row in expected_released
    ]
    if list(autonomous_ids) != expected_ids:
        return None
    declared_strict = arm.get("autonomous_suffix_target_recovered_strict")
    declared_loose = arm.get("autonomous_suffix_target_recovered_loose")
    if not isinstance(declared_strict, bool) or not isinstance(
        declared_loose, bool
    ):
        return None
    intervention = arm.get("intervention")
    if forced:
        if not rows or not isinstance(intervention, Mapping):
            return None
        if intervention.get("row") != rows[0]:
            return None
        if intervention.get("accounting_status") != (
            "excluded_from_autonomous_suffix_but_included_in_final_conditioned_set"
        ):
            return None
    elif intervention is not None:
        return None
    strict = {
        str(owner)
        for row in expected_released
        for owner in _mapping(row, "autonomous released row").get(
            "strict_matched_owner_ids", []
        )
    }
    loose = {
        str(owner)
        for row in expected_released
        for owner in _mapping(row, "autonomous released row").get(
            "loose_matched_owner_ids", []
        )
    }
    target = arm.get("target_gt_owner_id")
    # The current runner does not repeat target_gt_owner_id inside the arm;
    # its top-level booleans are nevertheless recomputed from exactly these
    # released rows.  Require internal boolean consistency by accepting the
    # row-derived owner sets and comparing once the caller supplies the role.
    semantic = [
        evidence
        for row in expected_released
        for evidence in _mapping(row, "autonomous released row").get(
            "semantic_drift_evidence", []
        )
    ]
    if list(arm.get("semantic_drift_evidence", [])) != semantic:
        return None
    return {
        "rows": expected_released,
        "owner_ids_strict": sorted(strict),
        "owner_ids_loose": sorted(loose),
        "declared_target_recovered_strict": declared_strict,
        "declared_target_recovered_loose": declared_loose,
        "semantic_drift_evidence": semantic,
        "target_gt_owner_id": target,
    }


def _arm_recovery_observed(
    arm: Mapping[str, Any] | None, target_gt_owner_id: str | None = None
) -> bool | None:
    suffix = _autonomous_suffix(arm)
    if suffix is None:
        return None
    if target_gt_owner_id is None:
        return bool(
            suffix["declared_target_recovered_strict"]
            or suffix["declared_target_recovered_loose"]
        )
    strict = target_gt_owner_id in suffix["owner_ids_strict"]
    loose = target_gt_owner_id in suffix["owner_ids_loose"]
    if strict != suffix["declared_target_recovered_strict"] or loose != suffix[
        "declared_target_recovered_loose"
    ]:
        return None
    return strict or loose


def _arm_recovered(
    arm: Mapping[str, Any] | None, target_gt_owner_id: str | None = None
) -> bool:
    return _arm_recovery_observed(arm, target_gt_owner_id) is True


def _behavior_collision_co_movement(
    role_results: Mapping[str, Mapping[str, Any]], pair_id: str
) -> bool | None:
    covering = role_results.get(f"collision:{pair_id}:P+G")
    foil = role_results.get(f"collision:{pair_id}:P+F")
    if covering is None or foil is None:
        return None
    if not _passed_role_landscape_admission(covering) or not _passed_role_landscape_admission(foil):
        return None
    covering_forced = _mapping(
        _mapping(covering.get("arms"), "collision covering arms").get("forced_description_greedy"),
        "collision covering forced arm",
    )
    foil_forced = _mapping(
        _mapping(foil.get("arms"), "collision foil arms").get("forced_description_greedy"),
        "collision foil forced arm",
    )
    covering_recovery = _arm_recovery_observed(
        covering_forced, str(covering.get("gt_owner_id"))
    )
    foil_recovery = _arm_recovery_observed(
        foil_forced, str(foil.get("gt_owner_id"))
    )
    if covering_recovery is None or foil_recovery is None:
        return None
    return (not covering_recovery) and foil_recovery


def _role_prefix_digest(role: Mapping[str, Any]) -> str:
    prefix = _mapping(role.get("prefix"), "role prefix")
    declared = prefix.get("token_ids_sha256")
    if isinstance(declared, str):
        return _digest(declared, "role prefix token digest")
    return sha256_json([int(value) for value in _sequence(prefix.get("token_ids"), "role prefix token ids")])


def _token_identical_null_match(
    collision_roles: Sequence[Mapping[str, Any]], null_pairs: Sequence[Mapping[str, Any]]
) -> bool:
    by_suffix = {str(role["role_id"]).rsplit(":", 1)[-1]: role for role in collision_roles}
    if not {"P", "P+G", "P+F"} <= set(by_suffix):
        return False
    candidate_images = {
        str(_mapping(role.get("trajectory"), "collision role trajectory").get("image_id"))
        for role in by_suffix.values()
    }
    if candidate_images != {"16228"}:
        return False
    for pair in null_pairs:
        if str(pair.get("pair_id")) != "wit-16228-46-a":
            continue
        if str(pair.get("image_id")) != "16228":
            continue
        null_by_kind = {}
        for role in _sequence(_mapping(pair, "null pair").get("roles", []), "null roles"):
            role = _mapping(role, "null role")
            kind = str(role.get("role_kind"))
            if kind.endswith("baseline"):
                null_by_kind["P"] = role
            elif kind.endswith("covering"):
                null_by_kind["P+G"] = role
            elif kind.endswith("foil"):
                null_by_kind["P+F"] = role
        if set(null_by_kind) != {"P", "P+G", "P+F"}:
            continue
        if any(_role_prefix_digest(by_suffix[key]) != _role_prefix_digest(null_by_kind[key]) for key in null_by_kind):
            continue
        if by_suffix["P+G"].get("pred_row_id") != null_by_kind["P+G"].get("pred_row_id"):
            continue
        if by_suffix["P+F"].get("pred_row_id") != null_by_kind["P+F"].get("pred_row_id"):
            continue
        return True
    return False


def _build_collision_evidence(
    loaded: Mapping[str, Any],
    landscape_by_context_rung: Mapping[tuple[str, str], Mapping[str, Any]],
    role_results: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    registry = loaded["registry"]
    smoke = _mapping(registry.get("smoke"), "registry.smoke")
    all_roles = _sequence(smoke.get("roles", []), "smoke roles")
    collision_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for role in all_roles:
        role = _mapping(role, "smoke role")
        role_id = str(role.get("role_id"))
        if role_id.startswith("collision:"):
            collision_groups[role_id.split(":", 2)[1]].append(role)
    null_envelope = _mapping(smoke.get("null_pair_envelope", {}), "null envelope")
    null_pairs = [
        _mapping(pair, "null pair")
        for pair in _sequence(null_envelope.get("pairs", []), "null pairs")
    ]

    # Scalar smoke is a sign-sensitivity/plumbing probe only.  It must never
    # become the rung that supplies a collision-mechanism verdict.
    preferred_rungs = ("L1", "L2", "L0")

    def declines_for_roles(
        roles: Sequence[Mapping[str, Any]],
    ) -> tuple[str | None, dict[str, dict[str, float]]]:
        suffix_to_role = {}
        for role in roles:
            role_id = str(role["role_id"])
            kind = str(role.get("role_kind"))
            if kind.endswith("baseline") or role_id.endswith(":P") or role_id.endswith(":P_null"):
                suffix_to_role["P"] = role
            elif kind.endswith("covering") or role_id.endswith(":P+G") or role_id.endswith(":P_null+G_null"):
                suffix_to_role["G"] = role
            elif kind.endswith("foil") or role_id.endswith(":P+F") or role_id.endswith(":P_null+F_null"):
                suffix_to_role["F"] = role
        if set(suffix_to_role) != {"P", "G", "F"}:
            return None, {}
        context_ids = {key: f"ctx:fn:{role['role_id']}" for key, role in suffix_to_role.items()}
        rung = next(
            (
                candidate_rung
                for candidate_rung in preferred_rungs
                if all((context_id, candidate_rung) in landscape_by_context_rung for context_id in context_ids.values())
            ),
            None,
        )
        if rung is None:
            return None, {}
        result: dict[str, dict[str, float]] = {
            variant: {} for variant in COLLISION_VARIANTS
        }
        for threshold in IOU_THRESHOLDS:
            key = str(threshold)
            for variant in COLLISION_VARIANTS:
                peaks = {
                    arm: _mapping(
                        landscape_by_context_rung[(context_id, rung)][
                            "per_iou_threshold"
                        ][key].get("collision_variant_peaks", {}),
                        f"{arm} collision variant peaks",
                    ).get(variant)
                    for arm, context_id in context_ids.items()
                }
                if any(value is None for value in peaks.values()):
                    continue
                result[variant][key] = (
                    float(peaks["G"]) - float(peaks["P"])
                ) - (float(peaks["F"]) - float(peaks["P"]))
        return rung, result

    def candidate_neighborhood_for_roles(
        roles: Sequence[Mapping[str, Any]], rung: str | None
    ) -> tuple[bool | None, dict[str, float]]:
        if rung is None:
            return None, {}
        arm_roles = {}
        for role in roles:
            role_id = str(role["role_id"])
            kind = str(role.get("role_kind"))
            if kind.endswith("baseline") or role_id.endswith(":P"):
                arm_roles["P"] = role
            elif kind.endswith("covering") or role_id.endswith(":P+G"):
                arm_roles["G"] = role
            elif kind.endswith("foil") or role_id.endswith(":P+F"):
                arm_roles["F"] = role
        if set(arm_roles) != {"P", "G", "F"}:
            return None, {}
        by_arm = {}
        singleton_ids_by_arm: dict[str, list[str]] = {}
        for arm, role in arm_roles.items():
            entry = landscape_by_context_rung[(f"ctx:fn:{role['role_id']}", rung)]
            rows = entry["per_iou_threshold"][str(PRIMARY_IOU_THRESHOLD)][
                "raw_candidates"
            ]
            by_arm[arm] = {
                str(row["candidate_neighborhood_id"]): float(
                    row["raw_fp32_complete_box_logprob_sum"]
                )
                for row in rows
                if row.get("candidate_neighborhood_member") is True
                and isinstance(row.get("candidate_neighborhood_id"), str)
                and row.get("candidate_neighborhood_id")
            }
            singleton_ids_by_arm[arm] = [
                str(row["exact_gt_singleton_id"])
                for row in rows
                if row.get("exact_gt_singleton_member") is True
                and isinstance(row.get("exact_gt_singleton_id"), str)
                and row.get("exact_gt_singleton_id")
            ]
        common = validate_common_candidate_neighborhood_ids(by_arm)
        validate_common_exact_gt_singleton_ids(singleton_ids_by_arm)
        declines = {
            member_id: (by_arm["G"][member_id] - by_arm["P"][member_id])
            - (by_arm["F"][member_id] - by_arm["P"][member_id])
            for member_id in sorted(common)
        }
        return all(value < 0 for value in declines.values()), declines

    null_rows = []
    null_declines_by_variant_and_threshold: dict[
        str, dict[str, dict[str, float]]
    ] = {
        variant: {str(threshold): {} for threshold in IOU_THRESHOLDS}
        for variant in COLLISION_VARIANTS
    }
    for pair in null_pairs:
        pair_id = str(pair.get("pair_id"))
        expectation = NULL_PAIR_EXPECTATIONS.get(pair_id)
        if expectation is not None and (
            str(pair.get("image_id")) != expectation["image_id"]
            or bool(pair.get("is_mechanical")) != expectation["is_mechanical"]
        ):
            raise MechanismAnalysisError(
                f"frozen null pair {pair_id!r} image/mechanical identity mismatch"
            )
        rung, declines = declines_for_roles(
            [_mapping(role, "null role") for role in _sequence(pair.get("roles", []), "null roles")]
        )
        null_rows.append(
            {
                "pair_id": pair_id,
                "is_mechanical": bool(pair.get("is_mechanical")),
                "rung": rung,
                "selective_decline_by_variant_and_iou": declines,
            }
        )
        for variant, by_threshold in declines.items():
            for threshold, value in by_threshold.items():
                null_declines_by_variant_and_threshold[variant][threshold][
                    pair_id
                ] = value

    envelopes: dict[str, dict[str, dict[str, Any]]] = {}
    for variant in COLLISION_VARIANTS:
        envelopes[variant] = {}
        for threshold in map(str, IOU_THRESHOLDS):
            values = null_declines_by_variant_and_threshold[variant][threshold]
            envelope = compute_collision_envelope(
                values, loaded["tolerance"]["effective_tolerance"]
            )
            registered_ids = set(values)
            if registered_ids != REQUIRED_NULL_PAIR_IDS:
                envelope["status"] = "insufficient_frozen_null_registry"
                envelope["missing_required_pair_ids"] = sorted(
                    REQUIRED_NULL_PAIR_IDS - registered_ids
                )
                envelope["foreign_pair_ids"] = sorted(
                    registered_ids - REQUIRED_NULL_PAIR_IDS
                )
                envelope["threshold"] = None
            envelopes[variant][threshold] = envelope

    collision_rows = []
    collision_by_owner = {}
    for pair_id, roles in sorted(collision_groups.items()):
        rung, declines = declines_for_roles(roles)
        target_owner = str(roles[0]["gt_owner_id"])
        image_id = str(_mapping(roles[0].get("trajectory"), "collision trajectory").get("image_id"))
        token_match = _token_identical_null_match(roles, null_pairs)
        neighborhood_stable, neighborhood_declines = candidate_neighborhood_for_roles(
            roles, rung
        )
        behavior_co_movement = (
            None
            if loaded["behavior"] is None
            else _behavior_collision_co_movement(role_results, pair_id)
        )
        evaluation = evaluate_collision_candidate(
            target_gt_owner_id=target_owner,
            image_id=image_id,
            selective_declines_by_variant_and_iou=declines,
            envelopes_by_variant_and_iou=envelopes,
            behavior_co_movement=behavior_co_movement,
            token_identical_matched_null=token_match,
            candidate_neighborhood_sign_stable=neighborhood_stable,
        )
        row = {
            "schema_version": COLLISION_EVIDENCE_SCHEMA_VERSION,
            "unit_id": UNIT_ID,
            "pair_id": pair_id,
            "gt_owner_id": target_owner,
            "image_id": image_id,
            "rung": rung,
            "selective_decline_by_variant_and_iou": declines,
            "primary_selective_decline_by_iou": declines.get(
                COLLISION_VARIANTS[0], {}
            ),
            "candidate_neighborhood_selective_declines": neighborhood_declines,
            "evaluation": evaluation,
        }
        collision_rows.append(row)
        collision_by_owner[target_owner] = row
    summary = {
        "required_null_pair_ids": sorted(REQUIRED_NULL_PAIR_IDS),
        "null_pairs": null_rows,
        "envelopes_by_variant_and_iou": envelopes,
        "primary_envelopes_by_iou": envelopes[COLLISION_VARIANTS[0]],
    }
    return collision_rows, collision_by_owner, summary


def _owner_behavior_roles(
    role_results: Mapping[str, Mapping[str, Any]], owner_id: str
) -> list[Mapping[str, Any]]:
    return [role for role in role_results.values() if role.get("gt_owner_id") == owner_id]


def _natural_extent_evidence(role: Mapping[str, Any]) -> bool:
    if not _passed_role_landscape_admission(role):
        return False
    arms = _mapping(role.get("arms"), "behavior arms")
    free = _mapping(arms.get("free_next_row"), "free next row arm")
    suffix = _autonomous_suffix(free)
    target = str(role.get("gt_owner_id"))
    return bool(
        suffix
        and _arm_recovery_observed(free, target) is True
        and target in suffix["owner_ids_loose"]
        and target not in suffix["owner_ids_strict"]
    )


def _autonomous_owner_accounting(role: Mapping[str, Any]) -> Mapping[str, Any] | None:
    if not _passed_role_landscape_admission(role):
        return None
    arms = _mapping(role.get("arms"), "behavior arms")
    free = _mapping(arms.get("free_next_row"), "free arm")
    forced = _mapping(arms.get("forced_description_greedy"), "forced arm")
    target = str(role.get("gt_owner_id"))
    free_suffix = _autonomous_suffix(free)
    forced_suffix = _autonomous_suffix(forced)
    prefix_owners = role.get("_validated_prefix_owner_ids")
    if (
        free_suffix is None
        or forced_suffix is None
        or isinstance(prefix_owners, (str, bytes))
        or not isinstance(prefix_owners, Sequence)
        or _arm_recovery_observed(free, target) is None
        or _arm_recovery_observed(forced, target) is None
    ):
        return None
    intervention = forced.get("intervention")
    intervention_row = (
        intervention.get("row") if isinstance(intervention, Mapping) else None
    )
    intervention_strict_ids = (
        intervention_row.get("strict_matched_owner_ids")
        if isinstance(intervention_row, Mapping)
        else None
    )
    if isinstance(intervention_strict_ids, (str, bytes)) or not isinstance(
        intervention_strict_ids, Sequence
    ):
        return None
    forced_intervention_owners = {str(value) for value in intervention_strict_ids}
    prefix_set = {str(value) for value in prefix_owners}
    free_set = prefix_set | set(free_suffix["owner_ids_strict"])
    # The forced arm's declared final set is intentionally row0-inclusive
    # (accounting_layer="final_intervention_conditioned_row0_plus_autonomous_suffix"):
    # the forced intervention row itself is excluded from suffix diagnostics
    # but still counted toward the final conditioned owner set.
    forced_set = (
        prefix_set | forced_intervention_owners | set(forced_suffix["owner_ids_strict"])
    )
    if free_set != {
        str(value) for value in free.get("final_unique_strict_owner_ids", [])
    } or forced_set != {
        str(value) for value in forced.get("final_unique_strict_owner_ids", [])
    }:
        return None
    expected = {
        "gained_owner_ids": sorted(forced_set - free_set),
        "retained_owner_ids": sorted(forced_set & free_set),
        "lost_owner_ids": sorted(free_set - forced_set),
        "target_recovery_exchange": bool(
            _arm_recovered(forced, target) and ((free_set - forced_set) - {target})
        ),
        "exchange_lost_owner_ids": (
            sorted((free_set - forced_set) - {target})
            if _arm_recovered(forced, target)
            else []
        ),
    }
    comparison = _mapping(role.get("comparison"), "behavior comparison")
    observed = comparison.get("downstream_owner_accounting")
    if not isinstance(observed, Mapping):
        return None
    if any(observed.get(key) != value for key, value in expected.items()):
        return None
    return observed


def _forced_release_without_exchange(role: Mapping[str, Any]) -> bool:
    if not _passed_role_landscape_admission(role):
        return False
    arms = _mapping(role.get("arms"), "behavior arms")
    forced = _mapping(arms.get("forced_description_greedy"), "forced greedy arm")
    accounting = _autonomous_owner_accounting(role)
    if accounting is None:
        return False
    return _arm_recovered(forced, str(role.get("gt_owner_id"))) and not bool(
        accounting.get("target_recovery_exchange")
    )


def _natural_failed(role: Mapping[str, Any]) -> bool:
    if not _passed_role_landscape_admission(role):
        return False
    free = _mapping(_mapping(role.get("arms"), "behavior arms").get("free_next_row"), "free arm")
    recovery = _arm_recovery_observed(free, str(role.get("gt_owner_id")))
    return recovery is False


def _semantic_drift_contrast(role: Mapping[str, Any]) -> bool:
    if not _passed_role_landscape_admission(role) or not role.get(
        "_validated_semantic_drift_eligible"
    ):
        return False
    arms = _mapping(role.get("arms"), "behavior arms")
    free = _mapping(arms.get("free_next_row"), "free arm")
    suffix = _autonomous_suffix(free)
    return bool(
        suffix
        and suffix.get("semantic_drift_evidence")
        and _natural_failed(role)
        and _forced_release_without_exchange(role)
    )


def _no_behavior_contrary_support(roles: Sequence[Mapping[str, Any]]) -> bool:
    if not roles:
        return False
    for role in roles:
        if not _passed_role_landscape_admission(role):
            return False
        arms = _mapping(role.get("arms"), "behavior arms")
        for name in ("free_next_row", "forced_description_greedy"):
            recovery = _arm_recovery_observed(
                _mapping(arms.get(name), f"{name} arm"),
                str(role.get("gt_owner_id")),
            )
            if recovery is None or recovery:
                return False
        for sample in _sequence(
            arms.get("forced_description_low_temperature_samples", []), "sample arms"
        ):
            recovery = _arm_recovery_observed(
                _mapping(sample, "sample arm"), str(role.get("gt_owner_id"))
            )
            if recovery is None or recovery:
                return False
    return True


def _build_owner_mechanisms(
    loaded: Mapping[str, Any],
    landscape_entries: Sequence[Mapping[str, Any]],
    collision_by_owner: Mapping[str, Mapping[str, Any]],
    role_results: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    registry = loaded["registry"]
    targets = [
        _mapping(target, "mechanism target")
        for target in _sequence(
            _mapping(registry.get("mechanism_cohort"), "mechanism cohort").get("targets"),
            "mechanism targets",
        )
    ]
    entries_by_owner: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for entry in landscape_entries:
        entries_by_owner[str(entry["gt_owner_id"])].append(entry)

    rows = []
    tolerance = loaded["tolerance"]["effective_tolerance"]
    for target in targets:
        owner_id = str(target["gt_owner_id"])
        landscapes = entries_by_owner.get(owner_id, [])
        behavior_roles = _owner_behavior_roles(role_results, owner_id)
        usable = [entry for entry in landscapes if entry.get("usable_target_support")]
        reasons: list[str] = []

        geometry_ok = bool(
            usable
            and any(_natural_extent_evidence(role) for role in behavior_roles)
            and any(_forced_release_without_exchange(role) for role in behavior_roles)
        )
        geometry_reasons = (
            ["natural_loose_target_extent_plus_canonical_support_plus_nonexchange_release"]
            if geometry_ok
            else [
                *([] if usable else ["usable_canonical_target_region_support_missing"]),
                *(
                    []
                    if any(_natural_extent_evidence(role) for role in behavior_roles)
                    else ["target_overlapping_natural_strict_failure_extent_evidence_missing"]
                ),
                *(
                    []
                    if any(_forced_release_without_exchange(role) for role in behavior_roles)
                    else ["canonical_conditioned_nonexchange_release_missing"]
                ),
            ]
        )

        pre_entries = [
            entry
            for entry in usable
            if entry["role_kind"] in {"first_skip_pre", "due_turn_context", "collision_baseline"}
        ]
        post_decline = False
        by_role_rung = {(entry["role_kind"], entry["rung"]): entry for entry in landscapes}
        for rung in ("L1", "scalar_smoke", "L2", "L0"):
            before = by_role_rung.get(("first_skip_pre", rung))
            after = by_role_rung.get(("first_skip_post", rung))
            if before and after:
                before_peak = before["per_iou_threshold"][str(PRIMARY_IOU_THRESHOLD)]["target_peak"]
                after_peak = after["per_iou_threshold"][str(PRIMARY_IOU_THRESHOLD)]["target_peak"]
                if before_peak is not None and after_peak is not None and float(after_peak) - float(before_peak) < -tolerance:
                    post_decline = True
        conditional_release = any(
            _natural_failed(role) and _forced_release_without_exchange(role)
            for role in behavior_roles
        )
        collision_supported = bool(
            collision_by_owner.get(owner_id, {}).get("evaluation", {}).get("supported")
        )
        route_ok = bool(pre_entries and (post_decline or conditional_release) and not collision_supported)
        route_reasons = (
            ["usable_prepass_support_plus_decline_or_nonexchange_owner_conditioned_release"]
            if route_ok
            else [
                *([] if pre_entries else ["usable_prepass_target_support_missing"]),
                *([] if post_decline or conditional_release else ["postpass_decline_or_owner_conditioned_release_missing"]),
                *(["selective_collision_evidence_prevents_route_only_support"] if collision_supported else []),
            ]
        )

        semantic_ok = bool(usable and any(_semantic_drift_contrast(role) for role in behavior_roles))
        semantic_reasons = (
            ["spatial_different_description_evidence_plus_canonical_localization_release_contrast"]
            if semantic_ok
            else [
                *([] if usable else ["spatial_target_support_missing"]),
                *(
                    []
                    if any(_semantic_drift_contrast(role) for role in behavior_roles)
                    else ["free_next_row_semantic_drift_and_canonical_contrast_missing"]
                ),
            ]
        )

        declared_context_ids = {
            f"ctx:fn:{role['role_id']}"
            for role in _all_registry_roles(registry)
            if role.get("gt_owner_id") == owner_id
        }
        scored_context_ids = {str(entry["context_id"]) for entry in landscapes}
        conclusion_entries = [entry for entry in landscapes if entry["rung"] in CONCLUSION_RUNGS]
        every_probe_null = bool(conclusion_entries) and not any(
            entry.get("usable_target_support") for entry in conclusion_entries
        )
        controls_available = bool(conclusion_entries) and all(
            entry.get("matched_control_calibration", {}).get("status") == "calibrated"
            and entry.get("matched_control_calibration", {}).get(
                "positive_controls_pass"
            )
            for entry in conclusion_entries
        )
        no_support_ok = bool(
            declared_context_ids
            and declared_context_ids <= scored_context_ids
            and every_probe_null
            and controls_available
            and loaded["behavior"] is not None
            and _no_behavior_contrary_support(behavior_roles)
            and "annotation_neutral" not in str(target.get("declared_role"))
        )
        no_support_reasons = (
            ["all_declared_conclusion_probes_null_controls_pass_and_no_behavior_contrary_support"]
            if no_support_ok
            else [
                *([] if declared_context_ids else ["no_declared_target_contexts"]),
                *([] if declared_context_ids <= scored_context_ids else ["not_every_declared_context_scored"]),
                *([] if every_probe_null else ["not_every_conclusion_bearing_probe_is_null"]),
                *([] if controls_available else ["matched_positive_controls_missing_or_unpassed"]),
                *([] if loaded["behavior"] is not None else ["behavior_output_required_for_no_contrary_support_gate"]),
                *(
                    []
                    if loaded["behavior"] is not None and _no_behavior_contrary_support(behavior_roles)
                    else ["registered_greedy_or_sampled_contrary_support_present_or_unchecked"]
                ),
                *(
                    ["annotation_neutral_owner_cannot_receive_model_mechanism_verdict"]
                    if "annotation_neutral" in str(target.get("declared_role"))
                    else []
                ),
            ]
        )

        collision = collision_by_owner.get(owner_id)
        collision_flag = (
            {
                "supported": False,
                "status": "unresolved_no_registered_collision_pair",
                "reasons": ["no_prospectively_bound_collision_P_G_F_roles"],
            }
            if collision is None
            else deepcopy(collision["evaluation"])
        )
        flags = {
            "geometry_or_extent": {
                "supported": geometry_ok,
                "status": "supported" if geometry_ok else "unresolved",
                "reasons": geometry_reasons,
            },
            "route_or_traversal": {
                "supported": route_ok,
                "status": "supported" if route_ok else "unresolved",
                "reasons": route_reasons,
            },
            "same_description_collision": collision_flag,
            "semantic_drift": {
                "supported": semantic_ok,
                "status": "supported" if semantic_ok else "unresolved",
                "reasons": semantic_reasons,
            },
            "no_usable_localization_support_under_tested_interface": {
                "supported": no_support_ok,
                "status": "supported" if no_support_ok else "unresolved",
                "reasons": no_support_reasons,
            },
        }
        supported = [name for name, flag in flags.items() if flag.get("supported")]
        if not supported:
            supported = ["unresolved"]
            unresolved_reasons = set()
            for flag in flags.values():
                raw_reasons = flag.get("reasons", [])
                if isinstance(raw_reasons, Sequence) and not isinstance(
                    raw_reasons, (str, bytes)
                ):
                    unresolved_reasons.update(str(reason) for reason in raw_reasons)
            reasons.extend(sorted(unresolved_reasons))
        final_set_gain = None
        admissible_accounting = [
            (role, accounting)
            for role in behavior_roles
            if (accounting := _autonomous_owner_accounting(role)) is not None
        ]
        if admissible_accounting:
            final_set_gain = any(
                owner_id
                in accounting.get("gained_owner_ids", [])
                and not accounting.get("target_recovery_exchange")
                for _role, accounting in admissible_accounting
            )
        rows.append(
            {
                "schema_version": MECHANISM_EVIDENCE_SCHEMA_VERSION,
                "unit_id": UNIT_ID,
                "gt_owner_id": owner_id,
                "image_id": target.get("image_id"),
                "description_size_crowding_stratum": target.get("stratum"),
                "declared_role": target.get("declared_role"),
                "source_cohort": target.get("cohort", target.get("expected_cohort")),
                "source_class_preserved": True,
                "behavior_evidence_admission_status": (
                    "admitted_autonomous_suffix_partition"
                    if admissible_accounting
                    else (
                        "behavior_present_but_missing_role_admission_or_autonomous_suffix_partition"
                        if loaded["behavior"] is not None
                        else "behavior_not_supplied"
                    )
                ),
                "mechanism_evidence_flags": flags,
                "supported_dispositions": supported,
                "final_set_gain": final_set_gain,
                "likelihood_only_never_counted_as_final_set_gain": True,
                "exact_disposition_reasons": (
                    ["one_or_more_complete_mechanism_evidence_conjunctions_passed"]
                    if supported != ["unresolved"]
                    else reasons
                ),
            }
        )
    return rows


def _separated_localized_peaks(entry: Mapping[str, Any]) -> bool:
    stats = entry["per_iou_threshold"][str(PRIMARY_IOU_THRESHOLD)]
    candidates = stats.get("raw_candidates", [])
    calibration = entry.get("matched_control_calibration", {})
    prominence_threshold = calibration.get("prominence_lower_quantile")
    background_peak = stats.get("background_peak")
    if (
        calibration.get("status") != "calibrated"
        or prominence_threshold is None
        or background_peak is None
    ):
        return False
    qualifying_clusters = set()
    for row in candidates:
        cluster = row.get("geometry_cluster_id", row.get("peak_cluster_id"))
        if row.get("region_at_threshold") != "target_strict" or cluster is None:
            continue
        score = row.get("raw_fp32_complete_box_logprob_sum")
        if score is None:
            continue
        if float(score) - float(background_peak) >= float(prominence_threshold):
            qualifying_clusters.add(cluster)
    return len(qualifying_clusters) >= 2


def _sampling_parameters(
    *,
    manifest_path: Path | None,
    temperature: float | None,
    top_p: float | None,
    repetition_penalty: float | None,
    k: int | None,
    seeds: Sequence[int],
    horizon_rows: int | None,
) -> dict[str, Any] | None:
    supplied_cli = any(
        value is not None
        for value in (temperature, top_p, repetition_penalty, k, horizon_rows)
    ) or bool(seeds)
    if manifest_path is not None and supplied_cli:
        raise MechanismAnalysisError("sampling parameters must come from CLI or one manifest, not both")
    if manifest_path is not None:
        document = _read_json(manifest_path, "sampling parameter manifest")
        return dict(_mapping(document.get("decode_parameters", document), "sampling decode parameters"))
    if not supplied_cli:
        return None
    if any(value is None for value in (temperature, top_p, repetition_penalty, k, horizon_rows)) or not seeds:
        raise MechanismAnalysisError("explicit sampling requires temperature/top-p/rp/K/seeds/horizon")
    return {
        "temperature": temperature,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "k": k,
        "seeds": [int(value) for value in seeds],
        "horizon_rows": horizon_rows,
    }


def _build_behavior_landscape_admission(
    *, loaded: Mapping[str, Any], landscape_entries: Sequence[Mapping[str, Any]]
) -> dict[str, Any] | None:
    admitted_by_role: dict[str, dict[str, Any]] = {}
    for entry in landscape_entries:
        if entry["rung"] not in {"L1", "L2"}:
            continue
        condition = None
        if entry.get("usable_target_support"):
            condition = "usable_target_strict_region_peak"
        elif _separated_localized_peaks(entry):
            condition = "multiple_separated_owner_localized_peaks"
        if condition is None:
            continue
        role_id = str(entry["role_id"])
        candidate = {
            "role_id": role_id,
            "gt_owner_id": entry["gt_owner_id"],
            "context_id": entry["context_id"],
            "status": "passed",
            "landscape_condition": {
                "name": condition,
                "status": "passed",
                "rung": entry["rung"],
            },
        }
        existing = admitted_by_role.get(role_id)
        if existing is None or (
            existing["landscape_condition"]["name"]
            == "multiple_separated_owner_localized_peaks"
            and condition == "usable_target_strict_region_peak"
        ):
            admitted_by_role[role_id] = candidate
    if not admitted_by_role:
        return None
    return {
        "schema_version": BEHAVIOR_LANDSCAPE_ADMISSION_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "status": "passed",
        "registry_digest": loaded["registry_digest"],
        "landscape_receipt_sha256": sha256_file(loaded["merge_path"]),
        "mechanism_decision_rules": {
            "sha256": sha256_file(loaded["mechanism_rules_path"]),
            "self_digest": loaded["mechanism_rules_digest"],
            "parent_execution_rules_sha256": sha256_file(loaded["rules_path"]),
            "planner_receipt_digest": loaded["planner_digest"],
            "registry_digest": loaded["registry_digest"],
            "fixed_budget_candidates_sha256": sha256_file(
                loaded["fixed_budget_path"]
            ),
        },
        "admitted_roles": [admitted_by_role[key] for key in sorted(admitted_by_role)],
    }


def _build_scalar_l1_required_admissions(
    landscape_entries: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    admissions = []
    for entry in landscape_entries:
        if entry.get("rung") != "scalar_smoke":
            continue
        interpretation = _mapping(
            entry.get("scalar_smoke_interpretation"),
            "scalar smoke interpretation",
        )
        l1_admission = _mapping(
            interpretation.get("l1_admission"), "scalar L1 admission"
        )
        if l1_admission.get("status") != "L1_required":
            continue
        admissions.append(
            {
                "status": "L1_required",
                "context_id": entry.get("context_id"),
                "role_id": entry.get("role_id"),
                "gt_owner_id": entry.get("gt_owner_id"),
                "reason": l1_admission.get("reason"),
                "stop_rule_4_eligible": False,
                "claim_scope": "scalar_smoke_marginal_or_negative_requires_frozen_L1",
            }
        )
    return admissions


def _build_sampling_admission(
    *,
    loaded: Mapping[str, Any],
    landscape_entries: Sequence[Mapping[str, Any]],
    role_results: Mapping[str, Mapping[str, Any]],
    parameters: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if parameters is None:
        return None
    temperature = _finite(parameters.get("temperature"), "sampling temperature")
    top_p = _finite(parameters.get("top_p"), "sampling top_p")
    rp = _finite(parameters.get("repetition_penalty"), "sampling repetition penalty")
    k = parameters.get("k")
    horizon = parameters.get("horizon_rows")
    seeds = [int(value) for value in _sequence(parameters.get("seeds"), "sampling seeds")]
    conditional_sampling = _mapping(
        loaded["mechanism_rules"].get("conditional_sampling"),
        "mechanism rules conditional_sampling",
    )
    frozen_temperature = _finite(
        conditional_sampling.get("temperature"),
        "mechanism rules sampling temperature",
    )
    frozen_top_p = _finite(
        conditional_sampling.get("top_p"), "mechanism rules sampling top_p"
    )
    frozen_rp = _finite(
        conditional_sampling.get("repetition_penalty"),
        "mechanism rules sampling repetition penalty",
    )
    null_semantics = _string(
        conditional_sampling.get("null_semantics"),
        "mechanism rules sampling null semantics",
    )
    if (
        temperature != frozen_temperature
        or top_p != frozen_top_p
        or rp != frozen_rp
        or isinstance(k, bool)
        or not isinstance(k, int)
        or k != len(seeds)
        or len(set(seeds)) != len(seeds)
        or isinstance(horizon, bool)
        or not isinstance(horizon, int)
        or not 1 <= horizon <= 8
    ):
        raise MechanismAnalysisError(
            "sampling tuple disagrees with mechanism rules or K/seeds/horizon constraints"
        )
    admitted_roles = []
    condition_by_role: dict[str, dict[str, Any]] = {}
    for entry in landscape_entries:
        if entry["rung"] not in CONCLUSION_RUNGS:
            continue
        role_id = str(entry["role_id"])
        behavior = role_results.get(role_id)
        if behavior is None or not _passed_role_landscape_admission(behavior):
            continue
        forced = _mapping(
            _mapping(behavior.get("arms"), "behavior arms").get("forced_description_greedy"),
            "forced greedy arm",
        )
        forced_recovery = _arm_recovery_observed(
            forced, str(behavior.get("gt_owner_id"))
        )
        if forced_recovery is None or forced_recovery:
            continue
        condition = None
        if entry.get("usable_target_support"):
            condition = "usable_target_strict_region_peak"
        elif _separated_localized_peaks(entry):
            condition = "multiple_separated_owner_localized_peaks"
        if condition is not None:
            context_id = str(entry.get("context_id"))
            gt_owner_id = str(entry.get("gt_owner_id"))
            if (
                context_id != f"ctx:fn:{role_id}"
                or gt_owner_id != str(behavior.get("gt_owner_id"))
            ):
                raise MechanismAnalysisError(
                    f"sampling role {role_id!r} behavior/landscape context identity mismatch"
                )
            admitted_roles.append(role_id)
            condition_by_role[role_id] = {
                "role_id": role_id,
                "context_id": context_id,
                "gt_owner_id": gt_owner_id,
                "landscape_condition": {
                    "name": condition,
                    "status": "passed",
                    "rung": entry["rung"],
                },
                "greedy_canonical_description_recovery": {
                    "status": "failed",
                    "arm": "forced_description_greedy",
                    "target_recovered": False,
                    "behavior_role_content_sha256": behavior[
                        "_behavior_role_content_sha256"
                    ],
                },
            }
    admitted_roles = sorted(set(admitted_roles))
    if not admitted_roles:
        return None
    behavior_path = loaded.get("behavior_output_path")
    behavior_document = loaded.get("behavior")
    if not isinstance(behavior_path, Path) or not isinstance(
        behavior_document, Mapping
    ):
        raise MechanismAnalysisError(
            "sampling admission requires an exact validated behavior output artifact"
        )
    return {
        "schema_version": SAMPLING_ADMISSION_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "status": "passed",
        "registry_digest": loaded["registry_digest"],
        "landscape_receipt_sha256": sha256_file(loaded["merge_path"]),
        "mechanism_decision_rules": {
            "path": str(loaded["mechanism_rules_path"]),
            "sha256": sha256_file(loaded["mechanism_rules_path"]),
            "self_digest": loaded["mechanism_rules_digest"],
            "conditional_sampling": {
                "temperature": frozen_temperature,
                "top_p": frozen_top_p,
                "repetition_penalty": frozen_rp,
                "null_semantics": null_semantics,
            },
        },
        "behavior_output": {
            "path": str(behavior_path),
            "sha256": sha256_file(behavior_path),
            "output_content_sha256": behavior_document[
                "output_content_sha256"
            ],
        },
        "landscape_condition": {
            "status": "passed",
            "name": "per_role_localized_landscape_with_greedy_failure",
            "condition_by_role": condition_by_role,
            "greedy_canonical_description_recovery_failed": True,
        },
        "admitted_role_ids": admitted_roles,
        "decode_parameters": {
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": rp,
            "k": k,
            "seeds": seeds,
            "horizon_rows": horizon,
        },
        "finite_sampling_failure_semantics": null_semantics,
    }


def analyze_sorted_fn_mechanisms(
    *,
    fn_mechanism_registry: str | Path,
    planner_receipt: str | Path,
    fixed_budget_candidates: str | Path,
    merged_scores: str | Path,
    run_attestation: str | Path,
    output_dir: str | Path,
    behavior_output: str | Path | None = None,
    l2_admission: str | Path | None = None,
    sampling_manifest: str | Path | None = None,
    sampling_temperature: float | None = None,
    sampling_top_p: float | None = None,
    sampling_repetition_penalty: float | None = None,
    sampling_k: int | None = None,
    sampling_seeds: Sequence[int] = (),
    sampling_horizon_rows: int | None = None,
) -> dict[str, Any]:
    registry_path = _resolved_file(fn_mechanism_registry, "FN mechanism registry")
    planner_path = _resolved_file(planner_receipt, "planner receipt")
    fixed_path = _resolved_file(fixed_budget_candidates, "fixed-budget candidates")
    scores_path = _resolved_file(merged_scores, "merged scores")
    attestation_path = _resolved_file(run_attestation, "run attestation")
    behavior_path = (
        None if behavior_output is None else _resolved_file(behavior_output, "behavior output")
    )
    l2_path = None if l2_admission is None else _resolved_file(l2_admission, "L2 admission")
    sampling_manifest_path = (
        None
        if sampling_manifest is None
        else _resolved_file(sampling_manifest, "sampling manifest")
    )
    loaded = _load_and_bind_inputs(
        registry_path=registry_path,
        planner_receipt_path=planner_path,
        fixed_budget_path=fixed_path,
        merged_scores_path=scores_path,
        run_attestation_path=attestation_path,
        behavior_output_path=behavior_path,
    )
    l2_contexts = _validate_l2_admission(
        l2_path,
        registry_digest=loaded["registry_digest"],
        planner_digest=loaded["planner_digest"],
    )
    landscape_entries, landscape_by_context_rung = _build_landscape_evidence(
        loaded, l2_admitted_contexts=l2_contexts
    )
    calibration = _apply_control_calibration(
        landscape_entries, loaded["tolerance"]["effective_tolerance"]
    )
    role_results = _role_results_by_id(
        loaded["behavior"], loaded["behavior_landscape_admission"], loaded["registry"]
    )
    collision_rows, collision_by_owner, collision_summary = _build_collision_evidence(
        loaded, landscape_by_context_rung, role_results
    )
    mechanism_rows = _build_owner_mechanisms(
        loaded, landscape_entries, collision_by_owner, role_results
    )
    parameters = _sampling_parameters(
        manifest_path=sampling_manifest_path,
        temperature=sampling_temperature,
        top_p=sampling_top_p,
        repetition_penalty=sampling_repetition_penalty,
        k=sampling_k,
        seeds=sampling_seeds,
        horizon_rows=sampling_horizon_rows,
    )
    sampling_admission_document = _build_sampling_admission(
        loaded=loaded,
        landscape_entries=landscape_entries,
        role_results=role_results,
        parameters=parameters,
    )
    behavior_landscape_admission_document = _build_behavior_landscape_admission(
        loaded=loaded, landscape_entries=landscape_entries
    )
    scalar_l1_required_admissions = _build_scalar_l1_required_admissions(
        landscape_entries
    )

    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    landscape_path = output_path / LANDSCAPE_EVIDENCE_NAME
    collision_path = output_path / COLLISION_EVIDENCE_NAME
    mechanism_path = output_path / MECHANISM_EVIDENCE_NAME
    analysis_path = output_path / ANALYSIS_NAME
    receipt_path = output_path / RECEIPT_NAME
    sampling_path = output_path / SAMPLING_ADMISSION_NAME
    behavior_landscape_admission_path = (
        output_path / BEHAVIOR_LANDSCAPE_ADMISSION_NAME
    )

    landscape_bytes = b"".join(canonical_json_bytes(row) + b"\n" for row in landscape_entries)
    collision_bytes = b"".join(canonical_json_bytes(row) + b"\n" for row in collision_rows)
    mechanism_bytes = b"".join(canonical_json_bytes(row) + b"\n" for row in mechanism_rows)
    _write_create_or_identical(landscape_path, landscape_bytes)
    _write_create_or_identical(collision_path, collision_bytes)
    _write_create_or_identical(mechanism_path, mechanism_bytes)
    if sampling_admission_document is not None:
        _write_create_or_identical(
            sampling_path, canonical_json_bytes(sampling_admission_document) + b"\n"
        )
    if behavior_landscape_admission_document is not None:
        _write_create_or_identical(
            behavior_landscape_admission_path,
            canonical_json_bytes(behavior_landscape_admission_document) + b"\n",
        )

    input_artifacts = {
        "fn_mechanism_registry": {"path": str(registry_path), "sha256": sha256_file(registry_path)},
        "planner_receipt": {"path": str(planner_path), "sha256": sha256_file(planner_path), "receipt_digest": loaded["planner_digest"]},
        "mechanism_decision_rules": {
            "path": str(loaded["mechanism_rules_path"]),
            "sha256": sha256_file(loaded["mechanism_rules_path"]),
            "self_digest": loaded["mechanism_rules_digest"],
            "parent_execution_rules_sha256": sha256_file(loaded["rules_path"]),
        },
        "fixed_budget_candidates": {"path": str(fixed_path), "sha256": sha256_file(fixed_path)},
        "merged_scores": {"path": str(scores_path), "sha256": sha256_file(scores_path)},
        "run_attestation": {"path": str(attestation_path), "sha256": sha256_file(attestation_path)},
        "behavior_output": (
            None if behavior_path is None else {"path": str(behavior_path), "sha256": sha256_file(behavior_path)}
        ),
        "l2_admission": None if l2_path is None else {"path": str(l2_path), "sha256": sha256_file(l2_path)},
        "sampling_parameter_source": (
            None
            if parameters is None
            else (
                {
                    "kind": "manifest",
                    "path": str(sampling_manifest_path),
                    "sha256": sha256_file(sampling_manifest_path),
                    "decode_parameters": dict(parameters),
                }
                if sampling_manifest_path is not None
                else {"kind": "explicit_cli", "decode_parameters": dict(parameters)}
            )
        ),
    }
    behavior_evidence_admitted = any(
        row["behavior_evidence_admission_status"]
        == "admitted_autonomous_suffix_partition"
        for row in mechanism_rows
    )
    analysis_content = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "analysis_surface": "pure_cpu_raw_fp32_fixed_budget_scientific_functional",
        "input_artifacts": input_artifacts,
        "registry_digest": loaded["registry_digest"],
        "planner_receipt_digest": loaded["planner_digest"],
        "quantitative_functional": {
            "raw_channel": "raw_model_logprob.complete_box_logprob_sum",
            "rp_policy_views_used_for_primary_decisions": False,
            "iou_thresholds": list(IOU_THRESHOLDS),
            "primary_iou_threshold": PRIMARY_IOU_THRESHOLD,
            "prominence_lower_quantile": LOWER_PROMINENCE_QUANTILE,
            "localized_rank_upper_quantile": UPPER_RANK_QUANTILE,
            "quantile_algorithm": "linear_type7",
            "numeric_tolerance": loaded["tolerance"],
        },
        "control_calibration": calibration,
        "collision_null_envelope": collision_summary,
        "owner_dispositions": mechanism_rows,
        "sampling_admission": sampling_admission_document,
        "behavior_landscape_admission": behavior_landscape_admission_document,
        "scalar_l1_required_admissions": scalar_l1_required_admissions,
        "likelihood_without_behavior_semantics": (
            "behavior_bound"
            if behavior_evidence_admitted
            else (
                "behavior_present_but_not_admissible_likelihood_only"
                if loaded["behavior"] is not None
                else "likelihood_only_no_collision_support_and_no_final_set_gain"
            )
        ),
    }
    analysis_document = {
        **analysis_content,
        "analysis_content_sha256": sha256_json(analysis_content),
    }
    _write_create_or_identical(analysis_path, canonical_json_bytes(analysis_document) + b"\n")

    output_artifacts = {
        "analysis": {"path": str(analysis_path), "sha256": sha256_file(analysis_path)},
        "landscape_evidence": {"path": str(landscape_path), "sha256": sha256_file(landscape_path), "row_count": len(landscape_entries)},
        "collision_evidence": {"path": str(collision_path), "sha256": sha256_file(collision_path), "row_count": len(collision_rows)},
        "mechanism_evidence": {"path": str(mechanism_path), "sha256": sha256_file(mechanism_path), "row_count": len(mechanism_rows)},
        "sampling_admission": (
            None if sampling_admission_document is None else {"path": str(sampling_path), "sha256": sha256_file(sampling_path)}
        ),
        "behavior_landscape_admission": (
            None
            if behavior_landscape_admission_document is None
            else {
                "path": str(behavior_landscape_admission_path),
                "sha256": sha256_file(behavior_landscape_admission_path),
            }
        ),
    }
    exact_dispositions = {
        row["gt_owner_id"]: {
            "supported_dispositions": row["supported_dispositions"],
            "exact_disposition_reasons": row["exact_disposition_reasons"],
        }
        for row in mechanism_rows
    }
    receipt_content = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "input_artifacts": input_artifacts,
        "output_artifacts": output_artifacts,
        "exact_dispositions": exact_dispositions,
        "exact_dispositions_sha256": sha256_json(exact_dispositions),
        "create_or_identical": True,
        "model_or_gpu_loaded": False,
    }
    receipt_document = {
        **receipt_content,
        "receipt_digest": sha256_json(receipt_content),
    }
    _write_create_or_identical(receipt_path, canonical_json_bytes(receipt_document) + b"\n")
    return analysis_document


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fn-mechanism-registry", type=Path, required=True)
    parser.add_argument("--planner-receipt", type=Path, required=True)
    parser.add_argument("--fixed-budget-candidates", type=Path, required=True)
    parser.add_argument("--merged-scores", type=Path, required=True)
    parser.add_argument("--run-attestation", type=Path, required=True)
    parser.add_argument("--behavior-output", type=Path)
    parser.add_argument("--l2-admission", type=Path)
    parser.add_argument("--sampling-manifest", type=Path)
    parser.add_argument("--sampling-temperature", type=float)
    parser.add_argument("--sampling-top-p", type=float)
    parser.add_argument("--sampling-repetition-penalty", type=float)
    parser.add_argument("--sampling-k", type=int)
    parser.add_argument("--sampling-seed", action="append", type=int, default=[])
    parser.add_argument("--sampling-horizon-rows", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    document = analyze_sorted_fn_mechanisms(
        fn_mechanism_registry=args.fn_mechanism_registry,
        planner_receipt=args.planner_receipt,
        fixed_budget_candidates=args.fixed_budget_candidates,
        merged_scores=args.merged_scores,
        run_attestation=args.run_attestation,
        output_dir=args.output_dir,
        behavior_output=args.behavior_output,
        l2_admission=args.l2_admission,
        sampling_manifest=args.sampling_manifest,
        sampling_temperature=args.sampling_temperature,
        sampling_top_p=args.sampling_top_p,
        sampling_repetition_penalty=args.sampling_repetition_penalty,
        sampling_k=args.sampling_k,
        sampling_seeds=args.sampling_seed,
        sampling_horizon_rows=args.sampling_horizon_rows,
    )
    print(
        json.dumps(
            {
                "schema_version": document["schema_version"],
                "owner_count": len(document["owner_dispositions"]),
                "behavior_bound": document["likelihood_without_behavior_semantics"] == "behavior_bound",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
