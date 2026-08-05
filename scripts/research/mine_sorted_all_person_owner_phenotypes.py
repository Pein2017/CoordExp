#!/usr/bin/env python3
"""Mine continuous, discovery-only owner phenotypes from sealed route artifacts.

This CPU-only join deliberately emits no cluster assignments, confirmation
cohorts, or visual-absence claims.  It combines the fixed nine-candidate score
bank with the existing analyzer census and the sealed task0/cohort ledgers.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
from typing import Any, NoReturn


UNIT_ID = "2026-08-03-sorted-all-person-owner-relative-route-landscape"
SCHEMA_VERSION = "sorted-all-person-owner-phenotype-mining.v1"
SUMMARY_SCHEMA_VERSION = "sorted-all-person-owner-phenotype-summary.v1"
RECEIPT_SCHEMA_VERSION = "sorted-all-person-owner-phenotype-mining-receipt.v1"
ANALYZER_OWNER_SCHEMA = "sorted-all-person-route-landscape-owner-phenotype.v1"
ANALYZER_RECEIPT_SCHEMA = "sorted-all-person-route-landscape-analysis-receipt.v1"

OWNER_CENSUS_NAME = "owner-phenotype-census.jsonl"
ANALYZER_RECEIPT_NAME = "route-landscape-analysis-receipt.json"
FEATURES_NAME = "owner-phenotype-features.jsonl"
SUMMARY_NAME = "owner-phenotype-descriptive-summary.json"
RECEIPT_NAME = "owner-phenotype-mining-receipt.json"

CONTEXT_IDS = (
    "root",
    "self-due-gt2",
    "self-due-gt17",
    "self-due-gt22",
    "self-due-gt32",
    "skip-post-gt17",
)
CALIBRATION_CONTEXT_BY_OWNER = {
    "gt:7511:2": "self-due-gt2",
    "gt:7511:17": "self-due-gt17",
    "gt:7511:22": "self-due-gt22",
    "gt:7511:32": "self-due-gt32",
}
OWNER_ID_RE = re.compile(r"^primary:(gt:[^:]+:[^:]+):\d{2}:[^:]+$")


class MiningContractError(RuntimeError):
    """Raised when a sealed source or exact join contract is violated."""


def _fail(message: str, **context: Any) -> NoReturn:
    if context:
        message = f"{message} | context: {json.dumps(context, sort_keys=True)}"
    raise MiningContractError(message)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
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
        _fail(f"{label} must be a JSON object")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{label} must be a finite number", observed=value)
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{label} must be finite", observed=value)
    return result


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise MiningContractError(f"{label} is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise MiningContractError(f"{label} is not valid JSON: {path}") from exc
    return dict(_mapping(value, label))


def read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise MiningContractError(f"{label} is missing: {path}") from exc
    if not lines:
        _fail(f"{label} must contain at least one row")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line:
            _fail(f"{label} line {line_number} is blank")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise MiningContractError(
                f"{label} line {line_number} is not valid JSON"
            ) from exc
        rows.append(dict(_mapping(value, f"{label} line {line_number}")))
    return rows


def _index_unique(
    rows: Iterable[Mapping[str, Any]], key: str, label: str
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        value = row.get(key)
        if not isinstance(value, str) or not value or value in result:
            _fail(f"{label} has missing or duplicate {key}", observed=value)
        result[value] = row
    return result


def stable_bank_statistics(scores: Sequence[float]) -> dict[str, Any]:
    """Return stable log-mass and softmax concentration for one fixed bank."""
    if not scores:
        _fail("candidate bank must not be empty")
    values = [_finite(value, "candidate score") for value in scores]
    maximum = max(values)
    shifted = [math.exp(value - maximum) for value in values]
    denominator = math.fsum(shifted)
    logsumexp = maximum + math.log(denominator)
    probabilities = [value / denominator for value in shifted]
    entropy = -math.fsum(
        probability * math.log(probability)
        for probability in probabilities
        if probability > 0.0
    )
    count = len(values)
    ordered_scores = sorted(values, reverse=True)
    return {
        "candidate_count": count,
        "logsumexp": logsumexp,
        "logmeanexp": logsumexp - math.log(count),
        "maximum_score": maximum,
        "top_candidate_probability": max(probabilities),
        "entropy": entropy,
        "normalized_entropy": entropy / math.log(count) if count > 1 else 0.0,
        "effective_candidate_count": math.exp(entropy),
        "top1_minus_top2_score": (
            ordered_scores[0] - ordered_scores[1] if count > 1 else None
        ),
    }


def _load_analyzer_rows(path_or_dir: Path, label: str) -> tuple[Path, list[dict[str, Any]]]:
    resolved = path_or_dir.expanduser().resolve(strict=True)
    census_path = resolved / OWNER_CENSUS_NAME if resolved.is_dir() else resolved
    rows = read_jsonl(census_path, label)
    for row in rows:
        if row.get("schema_version") != ANALYZER_OWNER_SCHEMA:
            _fail(
                f"{label} has an unexpected schema",
                observed=row.get("schema_version"),
            )
        if row.get("unit_id") != UNIT_ID:
            _fail(f"{label} has an unexpected unit_id", observed=row.get("unit_id"))
        contexts = _mapping(row.get("contexts"), f"{label} contexts")
        if set(contexts) != set(CONTEXT_IDS):
            _fail(
                f"{label} must contain the exact frozen context set",
                observed=sorted(contexts),
            )
    return census_path, rows


def _validate_analyzer_receipt(
    analysis_dir: Path, census_path: Path
) -> tuple[Path, dict[str, Any]]:
    receipt_path = analysis_dir / ANALYZER_RECEIPT_NAME
    receipt = read_json(receipt_path, "analyzer receipt")
    if receipt.get("schema_version") != ANALYZER_RECEIPT_SCHEMA:
        _fail("analyzer receipt has an unexpected schema")
    if receipt.get("unit_id") != UNIT_ID:
        _fail("analyzer receipt has an unexpected unit_id")
    declared = (
        _mapping(receipt.get("outputs"), "analyzer receipt outputs")
        .get(OWNER_CENSUS_NAME, {})
    )
    declared = _mapping(declared, "analyzer census output declaration")
    observed_digest = sha256_file(census_path)
    if declared.get("sha256") != observed_digest:
        _fail(
            "analyzer census digest does not match its receipt",
            observed=observed_digest,
            declared=declared.get("sha256"),
        )
    return receipt_path, receipt


def _load_score_banks(
    receipt: Mapping[str, Any], owner_ids: set[str]
) -> tuple[dict[tuple[str, str], list[float]], list[dict[str, Any]]]:
    entries = receipt.get("score_inputs")
    if not isinstance(entries, list) or not entries:
        _fail("analyzer receipt must declare non-empty score_inputs")
    banks: dict[tuple[str, str], dict[str, float]] = {}
    sources: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()
    for index, raw_entry in enumerate(entries):
        entry = _mapping(raw_entry, f"score input {index}")
        score_path = Path(str(entry.get("score_path"))).expanduser().resolve(strict=True)
        receipt_path = Path(str(entry.get("receipt_path"))).expanduser().resolve(strict=True)
        if score_path in seen_paths:
            _fail("analyzer receipt repeats a score input path", path=str(score_path))
        seen_paths.add(score_path)
        score_digest = sha256_file(score_path)
        receipt_digest = sha256_file(receipt_path)
        if score_digest != entry.get("score_sha256"):
            _fail("score input digest mismatch", path=str(score_path))
        if receipt_digest != entry.get("receipt_sha256"):
            _fail("score receipt digest mismatch", path=str(receipt_path))
        sources.append(
            {
                "score_path": str(score_path),
                "score_sha256": score_digest,
                "receipt_path": str(receipt_path),
                "receipt_sha256": receipt_digest,
            }
        )
        for row in read_jsonl(score_path, f"score input {index}"):
            if row.get("request_kind") != "primary" or row.get("primary_role") is not True:
                continue
            context_id = row.get("context_id")
            if context_id not in CONTEXT_IDS:
                _fail("primary score row has an unknown context", observed=context_id)
            candidate_id = row.get("candidate_id")
            match = OWNER_ID_RE.fullmatch(candidate_id) if isinstance(candidate_id, str) else None
            if match is None:
                _fail("primary score row has an invalid candidate_id", observed=candidate_id)
            owner_id = match.group(1)
            if owner_id not in owner_ids:
                _fail("primary score row refers to an unknown analyzer owner", owner_id=owner_id)
            score = _finite(
                _mapping(row.get("raw_model_logprob"), "raw_model_logprob").get(
                    "complete_box_logprob_sum"
                ),
                "complete_box_logprob_sum",
            )
            key = (context_id, owner_id)
            candidate_scores = banks.setdefault(key, {})
            if candidate_id in candidate_scores:
                _fail("duplicate primary candidate score", candidate_id=candidate_id)
            candidate_scores[candidate_id] = score
    expected_keys = {(context_id, owner_id) for context_id in CONTEXT_IDS for owner_id in owner_ids}
    if set(banks) != expected_keys:
        _fail(
            "raw score banks do not cover the exact owner/context product",
            missing_count=len(expected_keys - set(banks)),
            extra_count=len(set(banks) - expected_keys),
        )
    compact: dict[tuple[str, str], list[float]] = {}
    for key, candidate_scores in banks.items():
        if len(candidate_scores) != 9:
            _fail(
                "every stable owner/context bank must contain exactly nine candidates",
                context_id=key[0],
                owner_id=key[1],
                observed=len(candidate_scores),
            )
        compact[key] = [candidate_scores[candidate] for candidate in sorted(candidate_scores)]
    return compact, sources


def _nullable_bool(value: Any, label: str) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        _fail(f"{label} must be boolean or null", observed=value)
    return value


def _positive_evidence_status(value: bool | None, positive: str) -> str:
    if value is None:
        return "missing"
    return positive if value else "no_positive_evidence"


def build_behavior_status(
    row: Mapping[str, Any], behavior_evidence: Mapping[str, Any]
) -> dict[str, Any]:
    strict = _mapping(row.get("primary_strict_support"), "primary_strict_support")
    spatial = _mapping(row.get("natural_spatial_support"), "natural_spatial_support")
    cohort_greedy = _nullable_bool(strict.get("rp1_00_greedy"), "rp1_00_greedy")
    cohort_rescued = _nullable_bool(strict.get("strict_rescued"), "strict_rescued")
    greedy = _nullable_bool(
        behavior_evidence.get("greedy_strict_positive"), "behavior greedy_strict_positive"
    )
    rescued = _nullable_bool(
        behavior_evidence.get("strict_rescued"), "behavior strict_rescued"
    )
    if greedy != cohort_greedy or rescued != cohort_rescued:
        _fail(
            "sealed behavior ledger and cohort assignment disagree",
            owner_id=row.get("gt_owner_id"),
            behavior_greedy=greedy,
            cohort_greedy=cohort_greedy,
            behavior_rescued=rescued,
            cohort_rescued=cohort_rescued,
        )
    loose = _nullable_bool(spatial.get("admits_loose_only_b1"), "admits_loose_only_b1")
    no_free = _nullable_bool(
        spatial.get("no_free_spatial_support"), "no_free_spatial_support"
    )
    return {
        "source_scope": "sealed_canonical_12_image_task0_behavior_and_cohort_ledgers",
        "sealed_cohort": row.get("cohort"),
        "greedy": {
            "strict_positive": greedy,
            "status": _positive_evidence_status(greedy, "strict_positive"),
        },
        "strict_rescued": {
            "strict_positive": rescued,
            "status": _positive_evidence_status(rescued, "strict_positive"),
            "sampled_strict_positive_count": behavior_evidence.get(
                "sampled_strict_positive_count"
            ),
            "sampled_trajectory_count": behavior_evidence.get("sampled_trajectory_count"),
        },
        "loose_only_b1": {
            "positive": loose,
            "status": _positive_evidence_status(loose, "loose_positive"),
        },
        "no_free": {
            "positive": no_free,
            "status": _positive_evidence_status(no_free, "no_free_positive"),
        },
        "source_null_status": row.get("null_status"),
        "interpretation_boundary": (
            "false or null means no positive evidence under the sealed behavioral surface; "
            "it never establishes visual absence"
        ),
    }


def load_behavior_evidence(
    rows: Sequence[Mapping[str, Any]], owner_ids: set[str]
) -> dict[str, dict[str, Any]]:
    """Validate and reduce the sealed 1 greedy + 16 sampled task0 trajectories."""
    grouped: dict[str, list[Mapping[str, Any]]] = {owner_id: [] for owner_id in owner_ids}
    for row in rows:
        owner_id = row.get("gt_owner_id")
        if owner_id in grouped and row.get("policy_stratum") == "primary_rp_1.00":
            grouped[str(owner_id)].append(row)
    result: dict[str, dict[str, Any]] = {}
    expected_sampled_seeds = set(range(21001, 21017))
    for owner_id, owner_rows in grouped.items():
        greedy_rows = [row for row in owner_rows if row.get("decode_mode") == "greedy"]
        sampled_rows = [row for row in owner_rows if row.get("decode_mode") == "sampled"]
        if len(greedy_rows) != 1 or greedy_rows[0].get("seed") != 0:
            _fail(
                "behavior ledger must contain exactly one seed-0 greedy row per owner",
                owner_id=owner_id,
                observed=len(greedy_rows),
            )
        sampled_seeds = {row.get("seed") for row in sampled_rows}
        if len(sampled_rows) != 16 or sampled_seeds != expected_sampled_seeds:
            _fail(
                "behavior ledger must contain the exact 16 sampled seeds per owner",
                owner_id=owner_id,
                observed_seeds=sorted(seed for seed in sampled_seeds if isinstance(seed, int)),
            )
        greedy_positive = _nullable_bool(
            greedy_rows[0].get("strict_match_presence"), "greedy strict_match_presence"
        )
        sampled_values = [
            _nullable_bool(row.get("strict_match_presence"), "sampled strict_match_presence")
            for row in sampled_rows
        ]
        sampled_positive_count = sum(value is True for value in sampled_values)
        sampled_any = None if any(value is None for value in sampled_values) else sampled_positive_count > 0
        strict_rescued = (
            None
            if greedy_positive is None or sampled_any is None
            else (not greedy_positive and sampled_any)
        )
        result[owner_id] = {
            "greedy_strict_positive": greedy_positive,
            "sampled_any_strict_positive": sampled_any,
            "sampled_strict_positive_count": sampled_positive_count,
            "sampled_trajectory_count": len(sampled_rows),
            "strict_rescued": strict_rescued,
        }
    return result


def _analyzer_context_features(
    context: Mapping[str, Any], source: str
) -> dict[str, Any]:
    exact = _mapping(context.get("exact_anchor"), "exact_anchor")
    neighborhood = _mapping(context.get("neighborhood"), "neighborhood")
    peak = _mapping(context.get("peak"), "peak")
    analyzer_concentration = _mapping(peak.get("concentration"), "peak concentration")
    sidecars = _mapping(context.get("realized_sidecars"), "realized_sidecars")
    ambiguity = _mapping(context.get("ambiguity_bounds"), "ambiguity_bounds")
    return {
        "analyzer_value_source": source,
        "exact_anchor": {
            "candidate_id": exact.get("candidate_id"),
            "score": exact.get("score"),
            "competition_rank": exact.get("competition_rank"),
            "tied_owner_count": exact.get("tied_owner_count"),
        },
        "local_max": {
            "score": neighborhood.get("score"),
            "competition_rank": neighborhood.get("competition_rank"),
            "margin": neighborhood.get("margin"),
            "margin_sign": neighborhood.get("margin_sign"),
            "point_outcome": neighborhood.get("point_outcome"),
            "winning_candidate_score": peak.get("winning_candidate_score"),
            "winning_candidate_ids": peak.get("winning_candidate_ids"),
            "winning_transforms": peak.get("winning_transforms"),
        },
        "analyzer_unambiguous_bank_concentration": dict(analyzer_concentration),
        "sidecar_gap": {
            "maximum_gap": sidecars.get("maximum_gap"),
            "finite_bank_undercoverage": sidecars.get("finite_bank_undercoverage"),
            "entry_count": len(sidecars.get("entries", [])),
        },
        "ambiguity": {
            "status": ambiguity.get("status"),
            "outcome_invariant": ambiguity.get("outcome_invariant"),
            "invariant_outcome": ambiguity.get("invariant_outcome"),
            "best_rank": ambiguity.get("best_rank"),
            "worst_rank": ambiguity.get("worst_rank"),
            "lower_margin": ambiguity.get("lower_margin"),
            "upper_margin": ambiguity.get("upper_margin"),
        },
    }


def scorer_behavior_agreement(
    owner_id: str,
    analyzer_context: Mapping[str, Any],
    behavior: Mapping[str, Any],
) -> dict[str, Any] | None:
    context_id = CALIBRATION_CONTEXT_BY_OWNER.get(owner_id)
    if context_id is None:
        return None
    context = _mapping(analyzer_context.get(context_id), f"{context_id} analyzer context")
    neighborhood = _mapping(context.get("neighborhood"), "calibration neighborhood")
    ambiguity = _mapping(context.get("ambiguity_bounds"), "calibration ambiguity")
    if ambiguity.get("status") != "invariant" or ambiguity.get("outcome_invariant") is not True:
        scorer_positive: bool | None = None
        scorer_status = "ambiguity_unresolved"
    else:
        scorer_positive = (
            neighborhood.get("competition_rank") == 1
            and neighborhood.get("margin") is not None
            and _finite(neighborhood.get("margin"), "calibration margin") > 0.0
        )
        scorer_status = (
            "owner_relative_positive"
            if scorer_positive
            else "no_owner_relative_positive_evidence"
        )
    greedy = _mapping(behavior.get("greedy"), "behavior greedy").get("strict_positive")
    rescued = _mapping(
        behavior.get("strict_rescued"), "behavior strict_rescued"
    ).get("strict_positive")
    strict_any = None if greedy is None or rescued is None else bool(greedy or rescued)

    def compare(behavior_positive: bool | None) -> str:
        if scorer_positive is None or behavior_positive is None:
            return "unresolved_missing_or_ambiguous"
        if scorer_positive and behavior_positive:
            return "agree_positive"
        if not scorer_positive and not behavior_positive:
            return "agree_no_positive_evidence"
        if scorer_positive:
            return "scorer_positive_behavior_no_positive_evidence"
        return "behavior_positive_scorer_no_positive_evidence"

    return {
        "calibration_owner": True,
        "scorer_context_id": context_id,
        "scorer_status": scorer_status,
        "scorer_owner_relative_positive": scorer_positive,
        "scorer_neighborhood_rank": neighborhood.get("competition_rank"),
        "scorer_owner_relative_margin": neighborhood.get("margin"),
        "scorer_ambiguity_status": ambiguity.get("status"),
        "task0_greedy_strict_positive": greedy,
        "task0_any_strict_positive": strict_any,
        "greedy_agreement": compare(greedy),
        "any_strict_agreement": compare(strict_any),
        "scope_boundary": (
            "the self-due complete-box scorer cell and sealed task0 natural behavior are "
            "different measurement surfaces; agreement is calibration description, not causality"
        ),
    }


def _delta(current: Any, root: Any) -> float | None:
    if current is None or root is None:
        return None
    return _finite(current, "delta current") - _finite(root, "delta root")


def _context_deltas(contexts: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    root = contexts["root"]
    root_bank = _mapping(root.get("candidate_bank"), "root candidate bank")
    root_exact = _mapping(root.get("exact_anchor"), "root exact anchor")
    root_local = _mapping(root.get("local_max"), "root local max")
    root_sidecar = _mapping(root.get("sidecar_gap"), "root sidecar gap")
    deltas: dict[str, Any] = {}
    for context_id in CONTEXT_IDS[1:]:
        cell = contexts[context_id]
        bank = _mapping(cell.get("candidate_bank"), "candidate bank")
        exact = _mapping(cell.get("exact_anchor"), "exact anchor")
        local = _mapping(cell.get("local_max"), "local max")
        sidecar = _mapping(cell.get("sidecar_gap"), "sidecar gap")
        deltas[context_id] = {
            "candidate_bank_logsumexp": _delta(bank.get("logsumexp"), root_bank.get("logsumexp")),
            "candidate_bank_logmeanexp": _delta(bank.get("logmeanexp"), root_bank.get("logmeanexp")),
            "candidate_bank_maximum_score": _delta(
                bank.get("maximum_score"), root_bank.get("maximum_score")
            ),
            "candidate_bank_top_candidate_probability": _delta(
                bank.get("top_candidate_probability"),
                root_bank.get("top_candidate_probability"),
            ),
            "candidate_bank_effective_candidate_count": _delta(
                bank.get("effective_candidate_count"),
                root_bank.get("effective_candidate_count"),
            ),
            "exact_anchor_score": _delta(exact.get("score"), root_exact.get("score")),
            "local_max_score": _delta(local.get("score"), root_local.get("score")),
            "local_max_margin": _delta(local.get("margin"), root_local.get("margin")),
            "sidecar_maximum_gap": _delta(
                sidecar.get("maximum_gap"), root_sidecar.get("maximum_gap")
            ),
        }
    return deltas


def build_feature_rows(
    analyzer_rows: Sequence[Mapping[str, Any]],
    score_banks: Mapping[tuple[str, str], Sequence[float]],
    owner_rows: Mapping[str, Mapping[str, Any]],
    cohort_rows: Mapping[str, Mapping[str, Any]],
    behavior_evidence: Mapping[str, Mapping[str, Any]],
    overlay_rows: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    analyzer_by_id = _index_unique(analyzer_rows, "owner_id", "analyzer census")
    owner_ids = set(analyzer_by_id)
    if set(owner_rows) != owner_ids:
        _fail(
            "task0 owner ledger join is not exact",
            missing=sorted(owner_ids - set(owner_rows)),
            extra=sorted(set(owner_rows) - owner_ids),
        )
    if set(cohort_rows) != owner_ids:
        _fail(
            "cohort ledger join is not exact",
            missing=sorted(owner_ids - set(cohort_rows)),
            extra=sorted(set(cohort_rows) - owner_ids),
        )
    if set(behavior_evidence) != owner_ids:
        _fail("sealed behavior ledger join is not exact")
    if overlay_rows is not None and set(overlay_rows) != owner_ids:
        _fail("scalar overlay owner set must exactly match the base analyzer census")

    ordered_gt = sorted(
        owner_ids,
        key=lambda owner_id: (
            _finite(owner_rows[owner_id].get("bbox_xyxy")[1], "bbox y1"),
            _finite(owner_rows[owner_id].get("bbox_xyxy")[0], "bbox x1"),
            owner_id,
        ),
    )
    sorted_indices = {owner_id: index for index, owner_id in enumerate(ordered_gt)}
    output: list[dict[str, Any]] = []
    for owner_id in ordered_gt:
        base_row = analyzer_by_id[owner_id]
        source_row = overlay_rows[owner_id] if overlay_rows is not None else base_row
        analyzer_source = (
            "scalar_overlay_analyzer_output"
            if overlay_rows is not None
            else "base_analyzer_output"
        )
        analyzer_contexts = _mapping(source_row.get("contexts"), "analyzer contexts")
        geometry = _mapping(
            base_row.get("geometry_and_neighbor_context"),
            "geometry_and_neighbor_context",
        )
        task0_owner = owner_rows[owner_id]
        bbox = task0_owner.get("bbox_xyxy")
        if not isinstance(bbox, list) or len(bbox) != 4:
            _fail("task0 owner bbox_xyxy must be a four-value array", owner_id=owner_id)
        x1, y1, x2, y2 = [_finite(value, "bbox coordinate") for value in bbox]
        if x2 < x1 or y2 < y1:
            _fail("task0 owner bbox has negative extent", owner_id=owner_id)
        behavior = build_behavior_status(cohort_rows[owner_id], behavior_evidence[owner_id])
        contexts: dict[str, Any] = {}
        for context_id in CONTEXT_IDS:
            context = _mapping(analyzer_contexts.get(context_id), f"{context_id} context")
            contexts[context_id] = {
                "candidate_bank": stable_bank_statistics(score_banks[(context_id, owner_id)]),
                **_analyzer_context_features(context, analyzer_source),
            }
        output.append(
            {
                "schema_version": SCHEMA_VERSION,
                "unit_id": UNIT_ID,
                "owner_id": owner_id,
                "image_id": str(task0_owner.get("image_id")),
                "original_annotation_index": task0_owner.get("original_annotation_index"),
                "gt_geometry": {
                    "bbox_pixel_xyxy": [x1, y1, x2, y2],
                    "area_pixels2": (x2 - x1) * (y2 - y1),
                    "width_pixels": x2 - x1,
                    "height_pixels": y2 - y1,
                    "sorted_index_zero_based": sorted_indices[owner_id],
                    "sorted_index_one_based": sorted_indices[owner_id] + 1,
                    "overlap_other_owner_count": geometry.get("overlap_other_owner_count"),
                    "strict_iou_0p5_other_owner_count": geometry.get(
                        "strict_iou_0p5_other_owner_count"
                    ),
                    "sum_other_owner_iou": geometry.get("sum_other_owner_iou"),
                    "max_other_owner_iou": geometry.get("max_other_owner_iou"),
                    "nearest_other_center_distance_pixels": geometry.get(
                        "nearest_other_center_distance_pixels"
                    ),
                    "nearest_other_center_distance_over_gt_diagonal": geometry.get(
                        "nearest_other_center_distance_over_gt_diagonal"
                    ),
                    "neighbor_center_count_within_1x_gt_diagonal": geometry.get(
                        "neighbor_center_count_within_1x_gt_diagonal"
                    ),
                    "neighbor_center_count_within_2x_gt_diagonal": geometry.get(
                        "neighbor_center_count_within_2x_gt_diagonal"
                    ),
                },
                "behavior_status": behavior,
                "contexts": contexts,
                "root_to_context_deltas": _context_deltas(contexts),
                "scorer_to_behavior_agreement": scorer_behavior_agreement(
                    owner_id, analyzer_contexts, behavior
                ),
                "discovery_boundary": {
                    "hard_phenotype_label": None,
                    "cluster_assignment": None,
                    "claim_scope": "continuous_descriptive_discovery_only",
                    "visual_absence_inferred": False,
                },
            }
        )
    return output


def _flatten_numeric(value: Any, prefix: str = "") -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    if value is None:
        if prefix:
            result[prefix] = None
    elif isinstance(value, bool):
        if prefix:
            result[prefix] = float(value)
    elif isinstance(value, (int, float)):
        number = float(value)
        if prefix and math.isfinite(number):
            result[prefix] = number
    elif isinstance(value, Mapping):
        for key in sorted(value):
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten_numeric(value[key], child_prefix))
    return result


def _average_ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and ordered[end][1] == ordered[cursor][1]:
            end += 1
        average = ((cursor + 1) + end) / 2.0
        for position in range(cursor, end):
            ranks[ordered[position][0]] = average
        cursor = end
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 3 or len(right) != len(left):
        return None
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    denominator = math.sqrt(
        math.fsum(value * value for value in left_centered)
        * math.fsum(value * value for value in right_centered)
    )
    if denominator == 0.0:
        return None
    return math.fsum(
        a * b for a, b in zip(left_centered, right_centered, strict=True)
    ) / denominator


def _correlation(
    feature: str,
    comparator: str,
    flattened: Sequence[Mapping[str, float | None]],
) -> dict[str, Any]:
    pairs = [
        (row.get(feature), row.get(comparator))
        for row in flattened
        if row.get(feature) is not None and row.get(comparator) is not None
    ]
    left = [float(pair[0]) for pair in pairs]
    right = [float(pair[1]) for pair in pairs]
    pearson = _pearson(left, right)
    spearman = _pearson(_average_ranks(left), _average_ranks(right)) if pairs else None
    return {
        "feature": feature,
        "comparator": comparator,
        "pairwise_complete_count": len(pairs),
        "pearson": pearson,
        "spearman": spearman,
        "status": "computed" if pearson is not None or spearman is not None else "undefined",
    }


def build_descriptive_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    flattened = [_flatten_numeric(row) for row in rows]
    excluded_prefixes = (
        "original_annotation_index",
        "scorer_to_behavior_agreement",
        "discovery_boundary",
    )
    feature_names = sorted(
        {
            key
            for row in flattened
            for key in row
            if not key.startswith(excluded_prefixes)
        }
    )
    summaries: dict[str, Any] = {}
    for feature in feature_names:
        values = [float(row[feature]) for row in flattened if row.get(feature) is not None]
        if not values:
            continue
        summaries[feature] = {
            "observed_count": len(values),
            "missing_count": len(rows) - len(values),
            "minimum": min(values),
            "median": statistics.median(values),
            "mean": statistics.fmean(values),
            "maximum": max(values),
            "population_standard_deviation": statistics.pstdev(values),
        }
    score_features = [
        name
        for name in feature_names
        if name.startswith("contexts.") or name.startswith("root_to_context_deltas.")
    ]
    comparators = [
        name
        for name in feature_names
        if name.startswith("gt_geometry.")
        or name
        in {
            "behavior_status.greedy.strict_positive",
            "behavior_status.strict_rescued.strict_positive",
            "behavior_status.loose_only_b1.positive",
            "behavior_status.no_free.positive",
        }
    ]
    correlations = [
        _correlation(feature, comparator, flattened)
        for feature in score_features
        for comparator in comparators
        if feature != comparator
    ]
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "row_count": len(rows),
        "analysis_scope": "descriptive summaries and pairwise-complete correlations only",
        "hard_phenotype_labels_emitted": False,
        "clustering_performed": False,
        "feature_summaries": summaries,
        "correlations": correlations,
        "correlation_missingness": (
            "each coefficient uses pairwise-complete observations; undefined variance or fewer than "
            "three pairs yields null rather than imputation"
        ),
    }


def _write_create_or_identical(path: Path, content: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != content:
            _fail(f"{path} already exists with different content; refusing to overwrite")
        return "identical_existing_output"
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(content)
    os.replace(temporary, path)
    return "created"


def mine(
    analysis_dir: Path,
    owner_ledger_path: Path,
    behavior_ledger_path: Path,
    cohort_ledger_path: Path,
    scalar_overlay_analyzer_output: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    analysis_dir = analysis_dir.expanduser().resolve(strict=True)
    if not analysis_dir.is_dir():
        _fail("--analysis-dir must be a directory")
    census_path, analyzer_rows = _load_analyzer_rows(analysis_dir, "analyzer census")
    if len(analyzer_rows) != 41:
        _fail("production analyzer census must contain exactly 41 owners", observed=len(analyzer_rows))
    analyzer_receipt_path, analyzer_receipt = _validate_analyzer_receipt(
        analysis_dir, census_path
    )
    analyzer_by_id = _index_unique(analyzer_rows, "owner_id", "analyzer census")
    owner_ids = set(analyzer_by_id)
    score_banks, score_sources = _load_score_banks(analyzer_receipt, owner_ids)

    owner_ledger_path = owner_ledger_path.expanduser().resolve(strict=True)
    behavior_ledger_path = behavior_ledger_path.expanduser().resolve(strict=True)
    cohort_ledger_path = cohort_ledger_path.expanduser().resolve(strict=True)
    all_owner_rows = read_jsonl(owner_ledger_path, "task0 owner ledger")
    all_behavior_rows = read_jsonl(behavior_ledger_path, "sealed task0 behavior ledger")
    all_cohort_rows = read_jsonl(cohort_ledger_path, "sealed cohort ledger")
    selected_owner_rows = _index_unique(
        (row for row in all_owner_rows if row.get("gt_owner_id") in owner_ids),
        "gt_owner_id",
        "selected task0 owners",
    )
    selected_cohort_rows = _index_unique(
        (row for row in all_cohort_rows if row.get("gt_owner_id") in owner_ids),
        "gt_owner_id",
        "selected cohort owners",
    )
    behavior_evidence = load_behavior_evidence(all_behavior_rows, owner_ids)
    overlay_path: Path | None = None
    overlay_by_id: dict[str, Mapping[str, Any]] | None = None
    if scalar_overlay_analyzer_output is not None:
        overlay_path, overlay_rows = _load_analyzer_rows(
            scalar_overlay_analyzer_output, "scalar overlay analyzer census"
        )
        overlay_by_id = _index_unique(
            overlay_rows, "owner_id", "scalar overlay analyzer census"
        )
    feature_rows = build_feature_rows(
        analyzer_rows,
        score_banks,
        selected_owner_rows,
        selected_cohort_rows,
        behavior_evidence,
        overlay_by_id,
    )
    summary = build_descriptive_summary(feature_rows)
    sources = [
        {
            "role": "base_analyzer_owner_census",
            "path": str(census_path),
            "sha256": sha256_file(census_path),
        },
        {
            "role": "base_analyzer_receipt",
            "path": str(analyzer_receipt_path),
            "sha256": sha256_file(analyzer_receipt_path),
        },
        {
            "role": "canonical_task0_owner_ledger",
            "path": str(owner_ledger_path),
            "sha256": sha256_file(owner_ledger_path),
        },
        {
            "role": "sealed_task0_behavior_ledger",
            "path": str(behavior_ledger_path),
            "sha256": sha256_file(behavior_ledger_path),
        },
        {
            "role": "sealed_behavior_cohort_ledger",
            "path": str(cohort_ledger_path),
            "sha256": sha256_file(cohort_ledger_path),
        },
        *[
            {"role": "raw_score_and_receipt", **source}
            for source in sorted(score_sources, key=lambda source: source["score_path"])
        ],
    ]
    if overlay_path is not None:
        sources.append(
            {
                "role": "optional_scalar_overlay_analyzer_census",
                "path": str(overlay_path),
                "sha256": sha256_file(overlay_path),
            }
        )
    return feature_rows, summary, sources


def write_outputs(
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    sources: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / FEATURES_NAME
    summary_path = output_dir / SUMMARY_NAME
    receipt_path = output_dir / RECEIPT_NAME
    feature_content = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    summary_content = canonical_json_bytes(summary) + b"\n"
    feature_status = _write_create_or_identical(feature_path, feature_content)
    summary_status = _write_create_or_identical(summary_path, summary_content)
    code_path = Path(__file__).resolve()
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "execution_mode": "cpu_only",
        "claim_scope": "continuous_descriptive_discovery_only",
        "code": {"path": str(code_path), "sha256": sha256_file(code_path)},
        "source_artifacts": list(sources),
        "coverage": {
            "owner_count": len(rows),
            "context_count_per_owner": len(CONTEXT_IDS),
            "candidate_count_per_owner_context": 9,
            "calibration_owner_count": sum(
                row.get("scorer_to_behavior_agreement") is not None for row in rows
            ),
            "hard_phenotype_labels_emitted": False,
            "clustering_performed": False,
        },
        "outputs": {
            FEATURES_NAME: {"rows": len(rows), "sha256": sha256_file(feature_path)},
            SUMMARY_NAME: {"rows": 1, "sha256": sha256_file(summary_path)},
        },
        "interpretation_boundary": (
            "missing or negative positive-evidence flags are preserved and never promoted to visual "
            "absence; scorer/behavior comparisons are descriptive calibration only"
        ),
    }
    receipt["receipt_content_sha256"] = sha256_json(receipt)
    receipt_status = _write_create_or_identical(
        receipt_path, canonical_json_bytes(receipt) + b"\n"
    )
    return {
        "feature_path": str(feature_path),
        "feature_status": feature_status,
        "summary_path": str(summary_path),
        "summary_status": summary_status,
        "receipt_path": str(receipt_path),
        "receipt_status": receipt_status,
        "owner_count": len(rows),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        required=True,
        help="directory containing the 41-owner analyzer census and receipt",
    )
    parser.add_argument("--owner-ledger", type=Path, required=True)
    parser.add_argument("--behavior-ledger", type=Path, required=True)
    parser.add_argument("--cohort-ledger", type=Path, required=True)
    parser.add_argument(
        "--scalar-overlay-analyzer-output",
        type=Path,
        default=None,
        help=(
            "optional analyzer output directory or owner-phenotype-census.jsonl whose exact/local "
            "fields overlay the base analyzer; fixed raw bank mass remains bound to --analysis-dir"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows, summary, sources = mine(
        args.analysis_dir,
        args.owner_ledger,
        args.behavior_ledger,
        args.cohort_ledger,
        args.scalar_overlay_analyzer_output,
    )
    result = write_outputs(args.output_dir, rows, summary, sources)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
