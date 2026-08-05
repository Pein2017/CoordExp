#!/usr/bin/env python3
"""Seal the CPU-only image-2299 S1 calibration-transfer diagnostic.

The primary result is inherited, not re-estimated: the frozen support rule
transfers to 14/19 native-TP controls, below the frozen 0.8 gate.  Therefore
all 27 native-FN dispositions remain withheld and neither S2 nor S3 opens.

This diagnostic adds only two post-hoc descriptions: the TP transfer split by
category and owner-level any-hit reachability from the already captured 50
free greedy query-suffix box sidecars.  It never fits a threshold, pools image
2299 with another image, runs inference, or treats a sidecar miss as evidence
that visual support is absent.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, NoReturn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research import build_sorted_owner_accessibility_census_plan as planner  # noqa: E402


UNIT_ID = "2026-08-04-sorted-image2299-prospective-mechanism-extension"
LEGACY_UNIT_ID = "2026-08-03-sorted-owner-accessibility-phenotype-census"
IMAGE_ID = "2299"

REPORT_SCHEMA_VERSION = "sorted-image2299-calibration-transfer-audit.v1"
TP_ROW_SCHEMA_VERSION = "sorted-image2299-calibration-transfer-tp-control.v1"
FN_ROW_SCHEMA_VERSION = "sorted-image2299-calibration-transfer-fn-descriptive.v1"
RECEIPT_SCHEMA_VERSION = "sorted-image2299-calibration-transfer-audit-receipt.v1"

S0_RECEIPT_SCHEMA_VERSION = "sorted-image2299-native-ledger-receipt.v1"
PLAN_SCHEMA_VERSION = "sorted-owner-accessibility-census-plan.v1"
SHARD_RECEIPT_SCHEMA_VERSION = "sorted-owner-accessibility-census-shard-receipt.v1"
SOURCE_ANALYSIS_SCHEMA_VERSION = "sorted-image2299-owner-accessibility-analysis.v1"
SOURCE_OWNER_SCHEMA_VERSION = "sorted-image2299-owner-accessibility-owner-summary.v1"
SOURCE_CONTEXT_SCHEMA_VERSION = "sorted-image2299-owner-accessibility-owner-context.v1"
SOURCE_RECEIPT_SCHEMA_VERSION = "sorted-image2299-owner-accessibility-receipt.v1"
FREE_BOX_SCHEMA_VERSION = "sorted-owner-accessibility-census-free-decode.v1"

REPORT_JSON_NAME = "report.json"
REPORT_MD_NAME = "report.md"
TP_ROWS_NAME = "tp-control-rows.jsonl"
FN_ROWS_NAME = "fn-descriptive-rows.jsonl"
RECEIPT_NAME = "receipt.json"
OUTPUT_NAMES = frozenset(
    {REPORT_JSON_NAME, REPORT_MD_NAME, TP_ROWS_NAME, FN_ROWS_NAME, RECEIPT_NAME}
)

EXPECTED_PLAN_FILES = frozenset(
    {
        "candidate-bank.jsonl",
        "capture-rules.json",
        "category-registry.jsonl",
        "context-registry.jsonl",
        "image-registry.jsonl",
        "native-sidecar-registry.jsonl",
        "owner-registry.jsonl",
        "query-group-registry.jsonl",
        "shard-manifest.jsonl",
    }
)
EXPECTED_SHARD_PRIMARY_FILES = frozenset(
    {
        "census-scores.jsonl",
        "free-decode-sidecars.jsonl",
        "proposal-surface.jsonl",
        "x1-distributions.jsonl",
    }
)
EXPECTED_ANALYSIS_FILES = frozenset(
    {
        "analysis.json",
        "context-registry.jsonl",
        "owner-context-features.jsonl",
        "owner-summaries.jsonl",
    }
)

EXPECTED_OWNER_COUNT = 46
EXPECTED_TP_COUNT = 19
EXPECTED_FN_COUNT = 27
EXPECTED_CATEGORY_COUNTS = {"person": 38, "tie": 8}
EXPECTED_TP_CATEGORY_COUNTS = {"person": 16, "tie": 3}
EXPECTED_TRANSFER_CATEGORY_COUNTS = {
    "person": {"supported": 13, "total": 16},
    "tie": {"supported": 1, "total": 3},
}
EXPECTED_TRANSFER_SUPPORTED = 14
TRANSFER_FLOOR = 0.8
FROZEN_CALIBRATION_CONTENT_SHA256 = (
    "9dd6d7646fc55db6155124dc4bbfa46642b32b006d39758bd2e24d1ca97058c5"
)

DISPOSITION_WITHHELD = "withheld_calibration_nontransfer"
FN_ROLE = "descriptive_only_nontransferring"
BOX_ROW_KIND = "census_free_greedy_box_sidecar"


class AuditContractError(RuntimeError):
    """A sealed input or conclusion-bearing invariant changed."""


def _fail(message: str) -> NoReturn:
    raise AuditContractError(message)


@dataclass(frozen=True)
class SourceDirs:
    s0_native: Path
    s1_plan: Path
    s1_shard: Path
    s1_analysis: Path


@dataclass(frozen=True)
class LoadedInputs:
    sources: SourceDirs
    input_file_sha256: dict[str, str]
    analysis: dict[str, Any]
    owner_summaries: list[dict[str, Any]]
    owner_contexts: dict[str, dict[str, Any]]
    owners: list[dict[str, Any]]
    image: dict[str, Any]
    free_box_sidecars: list[dict[str, Any]]


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        _fail(f"input is unreadable at {path}: {exc}")
    return digest.hexdigest()


def _receipt_content_sha256(receipt: Mapping[str, Any]) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {key: value for key, value in receipt.items() if key != "receipt_content_sha256"}
        )
    )


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        _fail(f"{label} is missing at {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"{label} is unreadable at {path}: {exc}")
    if not isinstance(value, Mapping):
        _fail(f"{label} is not a JSON object")
    return dict(value)


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        _fail(f"{label} is missing at {path}")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        _fail(f"{label} is unreadable at {path}: {exc}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            _fail(f"{label} line {line_number} is invalid JSON: {exc}")
        if not isinstance(value, Mapping):
            _fail(f"{label} line {line_number} is not a JSON object")
        rows.append(dict(value))
    return rows


def _verify_digest(path: Path, expected: str, label: str) -> str:
    observed = sha256_file(path)
    if observed != expected:
        _fail(f"{label} digest drifted: {observed} != {expected}")
    return observed


def _verify_self_seal(receipt: Mapping[str, Any], label: str) -> str:
    declared = receipt.get("receipt_content_sha256")
    observed = _receipt_content_sha256(receipt)
    if declared != observed:
        _fail(f"{label} content seal drifted: {declared!r} != {observed!r}")
    return observed


def _verify_declared_files(
    directory: Path,
    declared: Mapping[str, Any],
    expected_names: frozenset[str],
    label: str,
) -> dict[str, str]:
    if set(declared) != expected_names:
        _fail(f"{label} declared file set drifted")
    result: dict[str, str] = {}
    for name in sorted(expected_names):
        item = declared[name]
        digest = item.get("sha256") if isinstance(item, Mapping) else item
        if not isinstance(digest, str):
            _fail(f"{label} lacks a digest for {name}")
        path = directory / name
        result[name] = _verify_digest(path, digest, f"{label} {name}")
        if isinstance(item, Mapping) and item.get("bytes") != path.stat().st_size:
            _fail(f"{label} byte count drifted for {name}")
    return result


def _validate_s0(sources: SourceDirs) -> tuple[dict[str, Any], dict[str, str]]:
    receipt_path = sources.s0_native / "receipt.json"
    receipt = _read_json(receipt_path, "S0 receipt")
    if (
        receipt.get("schema_version") != S0_RECEIPT_SCHEMA_VERSION
        or receipt.get("unit_id") != UNIT_ID
        or receipt.get("status") != "admitted"
    ):
        _fail("S0 receipt identity/status drifted")
    if (receipt.get("scope") or {}).get("image_id") != IMAGE_ID:
        _fail("S0 receipt is not the sealed image-2299 slice")
    if (receipt.get("scope") or {}).get("gpu_execution") != "not_performed":
        _fail("S0 ledger no longer declares its CPU-only materialization scope")
    expected_counts = {
        "false_negatives": EXPECTED_FN_COUNT,
        "false_positives": 5,
        "matched": EXPECTED_TP_COUNT,
        "matcher_ambiguity_classes": 0,
        "owners": EXPECTED_OWNER_COUNT,
        "predictions": 24,
    }
    if receipt.get("counts") != expected_counts:
        _fail("S0 owner/prediction denominator drifted")
    _verify_self_seal(receipt, "S0 receipt")
    files = _verify_declared_files(
        sources.s0_native,
        receipt.get("outputs") or {},
        frozenset({"greedy.json", "owner-ledger.jsonl", "prediction-row-ledger.jsonl"}),
        "S0",
    )
    files["receipt.json"] = sha256_file(receipt_path)
    return receipt, files


def _validate_plan(
    sources: SourceDirs, s0_receipt: Mapping[str, Any], s0_files: Mapping[str, str]
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, str]]:
    receipt_path = sources.s1_plan / "receipt.json"
    receipt = _read_json(receipt_path, "S1 plan receipt")
    if (
        receipt.get("schema_version") != PLAN_SCHEMA_VERSION
        or receipt.get("unit_id") != LEGACY_UNIT_ID
        or receipt.get("extension_unit_id") != UNIT_ID
    ):
        _fail("S1 plan receipt identity drifted")
    _verify_self_seal(receipt, "S1 plan receipt")
    shape = receipt.get("census_shape") or {}
    expected_shape = {
        "image_count": 1,
        "image_ids": [IMAGE_ID],
        "owner_count": EXPECTED_OWNER_COUNT,
        "native_true_positive_owner_count": EXPECTED_TP_COUNT,
        "native_false_negative_owner_count": EXPECTED_FN_COUNT,
        "owner_category_counts": EXPECTED_CATEGORY_COUNTS,
        "context_count": 25,
        "query_group_count": 50,
    }
    if any(shape.get(key) != value for key, value in expected_shape.items()):
        _fail("S1 plan census shape drifted")
    score_policy = receipt.get("score_input_policy") or {}
    if (
        score_policy.get("candidate_selection_uses_scores") is not False
        or score_policy.get("reads_any_score_artifact") is not False
        or score_policy.get("frozen_thresholds_loaded_by_downstream_analyzer_only") is not True
    ):
        _fail("S1 plan is no longer score-independent")
    source_digests = receipt.get("source_digests") or {}
    expected_s0_bindings = {
        "greedy_rollout": s0_files["greedy.json"],
        "owner_ledger": s0_files["owner-ledger.jsonl"],
        "prediction_ledger": s0_files["prediction-row-ledger.jsonl"],
        "s0_receipt": s0_files["receipt.json"],
    }
    if any(source_digests.get(key) != value for key, value in expected_s0_bindings.items()):
        _fail("S1 plan is not digest-bound to the supplied S0 files")
    if (receipt.get("source_content_digests") or {}).get(
        "s0_receipt_content_sha256"
    ) != s0_receipt.get("receipt_content_sha256"):
        _fail("S1 plan is not content-bound to the supplied S0 receipt")
    plan_files = _verify_declared_files(
        sources.s1_plan,
        receipt.get("output_file_digests") or {},
        EXPECTED_PLAN_FILES,
        "S1 plan",
    )
    plan_files["receipt.json"] = sha256_file(receipt_path)

    owners = _read_jsonl(sources.s1_plan / "owner-registry.jsonl", "S1 owner registry")
    images = _read_jsonl(sources.s1_plan / "image-registry.jsonl", "S1 image registry")
    if len(owners) != EXPECTED_OWNER_COUNT or len(images) != 1:
        _fail("S1 owner/image registry denominator drifted")
    if len({str(row.get("gt_owner_id")) for row in owners}) != EXPECTED_OWNER_COUNT:
        _fail("S1 owner registry has duplicate owner IDs")
    if Counter(str(row.get("normalized_description")) for row in owners) != Counter(
        EXPECTED_CATEGORY_COUNTS
    ):
        _fail("S1 owner registry category composition drifted")
    if str(images[0].get("image_id")) != IMAGE_ID:
        _fail("S1 image registry is not image 2299")
    return receipt, owners, images[0], plan_files


def _validate_shard(
    sources: SourceDirs,
    plan_receipt: Mapping[str, Any],
    analysis_receipt: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    receipt_path = sources.s1_shard / "shard-receipt.json"
    receipt = _read_json(receipt_path, "S1 shard receipt")
    if (
        receipt.get("schema_version") != SHARD_RECEIPT_SCHEMA_VERSION
        or receipt.get("unit_id") != LEGACY_UNIT_ID
        or str(receipt.get("image_id")) != IMAGE_ID
        or receipt.get("status") != "captured"
        or receipt.get("capture_completeness") != "complete_shard"
    ):
        _fail("S1 shard identity/completeness drifted")
    subset = receipt.get("subset_capture") or {}
    if subset.get("is_subset") is not False or subset.get(
        "usable_as_complete_shard_evidence"
    ) is not True:
        _fail("S1 shard is subset or smoke evidence")
    plan_binding = receipt.get("plan") or {}
    if (
        plan_binding.get("receipt_content_sha256")
        != plan_receipt.get("receipt_content_sha256")
        or plan_binding.get("capture_rules_sha256")
        != plan_receipt.get("capture_rules_sha256")
    ):
        _fail("S1 shard is bound to another plan")
    required_checks = (
        "all_finite",
        "canonical_suffix_verified",
        "coordinate_domain_ok",
        "cross_owner_tuple_collapse_verified",
        "every_row_bound_to_an_admission_receipt",
    )
    if any((receipt.get("checks") or {}).get(key) is not True for key in required_checks):
        _fail("S1 shard did not pass every frozen scorer check")
    phase = receipt.get("phase_order") or {}
    if (
        phase.get("behavior_sidecars_captured") is not True
        or phase.get("generation_phase_after_decision_scoring") is not True
        or phase.get("likelihood_scored_after_generation_phase") is not False
    ):
        _fail("S1 sidecar/decision-scoring phase separation drifted")
    counts = receipt.get("counts") or {}
    if counts.get("query_group_count") != 50 or counts.get("free_decode_sidecar_rows") != 75:
        _fail("S1 shard sidecar denominator drifted")

    sealed = (analysis_receipt.get("scored_shard") or {}).get("primary_file_sha256") or {}
    shard_files = _verify_declared_files(
        sources.s1_shard, sealed, EXPECTED_SHARD_PRIMARY_FILES, "S1 shard"
    )
    receipt_digest = sha256_file(receipt_path)
    if (analysis_receipt.get("scored_shard") or {}).get("receipt_sha256") != receipt_digest:
        _fail("S1 analysis receipt is not bound to the supplied shard receipt")
    shard_files["shard-receipt.json"] = receipt_digest

    free_rows = _read_jsonl(
        sources.s1_shard / "free-decode-sidecars.jsonl", "S1 free-decode sidecars"
    )
    box_rows = [row for row in free_rows if row.get("row_kind") == BOX_ROW_KIND]
    if len(box_rows) != 50:
        _fail("S1 shard does not contain exactly 50 free greedy box sidecars")
    return receipt, box_rows, shard_files


def _load_analysis_receipt(sources: SourceDirs) -> tuple[dict[str, Any], dict[str, str]]:
    receipt_path = sources.s1_analysis / "receipt.json"
    receipt = _read_json(receipt_path, "S1 analysis receipt")
    if (
        receipt.get("schema_version") != SOURCE_RECEIPT_SCHEMA_VERSION
        or receipt.get("unit_id") != UNIT_ID
    ):
        _fail("S1 analysis receipt identity drifted")
    _verify_self_seal(receipt, "S1 analysis receipt")
    files = _verify_declared_files(
        sources.s1_analysis,
        receipt.get("output_file_digests") or {},
        EXPECTED_ANALYSIS_FILES,
        "S1 analysis",
    )
    files["receipt.json"] = sha256_file(receipt_path)
    return receipt, files


def _validate_analysis_binding(
    receipt: Mapping[str, Any], plan_receipt: Mapping[str, Any]
) -> None:
    plan = receipt.get("plan") or {}
    if (
        plan.get("receipt_content_sha256") != plan_receipt.get("receipt_content_sha256")
        or plan.get("capture_rules_sha256") != plan_receipt.get("capture_rules_sha256")
    ):
        _fail("S1 analysis is bound to another plan")
    calibration = receipt.get("calibration") or {}
    if (
        calibration.get("content_sha256") != FROZEN_CALIBRATION_CONTENT_SHA256
        or calibration.get("thresholds_retuned") is not False
        or calibration.get("phenotype_fitted") is not False
    ):
        _fail("S1 analysis did not preserve the frozen calibration")


def load_inputs(sources: SourceDirs) -> LoadedInputs:
    s0_receipt, s0_files = _validate_s0(sources)
    plan_receipt, owners, image, plan_files = _validate_plan(
        sources, s0_receipt, s0_files
    )
    analysis_receipt, analysis_files = _load_analysis_receipt(sources)
    _validate_analysis_binding(analysis_receipt, plan_receipt)
    _, free_box_sidecars, shard_files = _validate_shard(
        sources, plan_receipt, analysis_receipt
    )

    analysis = _read_json(sources.s1_analysis / "analysis.json", "S1 analysis")
    summaries = _read_jsonl(
        sources.s1_analysis / "owner-summaries.jsonl", "S1 owner summaries"
    )
    contexts = _read_jsonl(
        sources.s1_analysis / "owner-context-features.jsonl", "S1 owner contexts"
    )
    context_by_id: dict[str, dict[str, Any]] = {}
    for row in contexts:
        key = str(row.get("owner_context_id"))
        if key in context_by_id:
            _fail(f"duplicate S1 owner-context ID {key!r}")
        context_by_id[key] = row

    input_file_sha256 = {
        **{f"s0-native/{name}": value for name, value in s0_files.items()},
        **{f"s1-plan/{name}": value for name, value in plan_files.items()},
        **{f"s1-shard/{name}": value for name, value in shard_files.items()},
        **{f"s1-analysis/{name}": value for name, value in analysis_files.items()},
    }
    return LoadedInputs(
        sources=sources,
        input_file_sha256=dict(sorted(input_file_sha256.items())),
        analysis=analysis,
        owner_summaries=summaries,
        owner_contexts=context_by_id,
        owners=owners,
        image=image,
        free_box_sidecars=free_box_sidecars,
    )


def _validate_primary_conclusion(inputs: LoadedInputs) -> dict[str, Mapping[str, Any]]:
    analysis = inputs.analysis
    if (
        analysis.get("schema_version") != SOURCE_ANALYSIS_SCHEMA_VERSION
        or analysis.get("unit_id") != UNIT_ID
        or str(analysis.get("image_id")) != IMAGE_ID
    ):
        _fail("S1 analysis identity drifted")
    expected_denominators = {
        "image2299_owner_count": EXPECTED_OWNER_COUNT,
        "image2299_native_tp_count": EXPECTED_TP_COUNT,
        "image2299_native_fn_count": EXPECTED_FN_COUNT,
        "legacy_12_owner_count_unchanged": 346,
        "legacy_12_eligible_native_fn_denominator_unchanged": 202,
        "pooled_13_image_denominator_created": False,
    }
    if analysis.get("denominators") != expected_denominators:
        _fail("S1 analysis denominator contract drifted")
    calibration = analysis.get("calibration") or {}
    if (
        calibration.get("content_sha256") != FROZEN_CALIBRATION_CONTENT_SHA256
        or calibration.get("thresholds_retuned") is not False
        or calibration.get("phenotype_fitted") is not False
    ):
        _fail("S1 analysis calibration identity drifted")
    transfer = analysis.get("calibration_transfer") or {}
    if (
        transfer.get("supported_due_boundary_count") != EXPECTED_TRANSFER_SUPPORTED
        or transfer.get("transfer_denominator_native_tp_count") != EXPECTED_TP_COUNT
        or not math.isclose(
            float(transfer.get("support_rate", math.nan)),
            EXPECTED_TRANSFER_SUPPORTED / EXPECTED_TP_COUNT,
        )
        or transfer.get("floor") != TRANSFER_FLOOR
        or transfer.get("passes") is not False
        or transfer.get("status") != "calibration_nontransferring"
        or transfer.get("on_failure") != "withhold_all_native_fn_dispositions"
    ):
        _fail("primary 14/19 calibration-transfer failure drifted")
    owner_rows = transfer.get("owner_rows") or []
    due_by_owner = {str(row.get("gt_owner_id")): row for row in owner_rows}
    if len(owner_rows) != EXPECTED_TP_COUNT or len(due_by_owner) != EXPECTED_TP_COUNT:
        _fail("S1 transfer control rows are not exactly the 19 native TPs")

    if len(inputs.owner_summaries) != EXPECTED_OWNER_COUNT:
        _fail("S1 owner-summary denominator drifted")
    summaries: dict[str, Mapping[str, Any]] = {}
    for row in inputs.owner_summaries:
        if row.get("schema_version") != SOURCE_OWNER_SCHEMA_VERSION:
            _fail("S1 owner summary schema drifted")
        owner_id = str(row.get("gt_owner_id"))
        if owner_id in summaries:
            _fail(f"duplicate S1 owner summary {owner_id!r}")
        summaries[owner_id] = row
    if set(summaries) != {str(row.get("gt_owner_id")) for row in inputs.owners}:
        _fail("S1 owner summaries do not exactly cover the plan owners")
    native_tp = [row for row in summaries.values() if row.get("native_true_positive") is True]
    native_fn = [row for row in summaries.values() if row.get("native_false_negative") is True]
    if len(native_tp) != EXPECTED_TP_COUNT or len(native_fn) != EXPECTED_FN_COUNT:
        _fail("S1 owner summary TP/FN split drifted")
    if Counter(str(row.get("normalized_description")) for row in native_tp) != Counter(
        EXPECTED_TP_CATEGORY_COUNTS
    ):
        _fail("S1 TP category split drifted")
    for row in native_fn:
        if (
            row.get("disposition") != DISPOSITION_WITHHELD
            or row.get("fn_disposition_interpretable") is not False
            or row.get("disposition_role") != DISPOSITION_WITHHELD
        ):
            _fail("one or more native-FN dispositions are no longer withheld")
    if analysis.get("fn_disposition_counts") != {DISPOSITION_WITHHELD: EXPECTED_FN_COUNT}:
        _fail("S1 FN withheld count drifted")
    return {"due_by_owner": due_by_owner, "summaries": summaries}


def _criterion_block(
    block: Mapping[str, Any], *, theta_peak: float, theta_concentration: float, epsilon: float
) -> dict[str, Any]:
    peak_gate = theta_peak + epsilon
    concentration_gate = theta_concentration + epsilon
    peak = float(block["peak_lift"])
    concentration = float(block["local_concentration"])
    failed = []
    if peak < peak_gate:
        failed.append("peak_lift")
    if concentration < concentration_gate:
        failed.append("local_concentration")
    clears = not failed
    if block.get("clears_frozen_support_rule") is not clears:
        _fail("S1 continuous features disagree with the frozen support gate")
    return {
        "peak_lift": peak,
        "local_concentration": concentration,
        "rank": int(block["rank"]),
        "population": int(block["unique_population_size"]),
        "failed_frozen_criteria": failed,
        "clears_frozen_support_rule": clears,
    }


def _build_tp_rows(
    inputs: LoadedInputs,
    due_by_owner: Mapping[str, Mapping[str, Any]],
    summaries: Mapping[str, Mapping[str, Any]],
    sidecar_hits: Mapping[str, list[str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    calibration = inputs.analysis["calibration"]
    theta_peak = float(calibration["theta_peak_lift"])
    theta_concentration = float(calibration["theta_local_concentration"])
    epsilon = float(calibration["epsilon"])
    rows: list[dict[str, Any]] = []
    for owner_id in sorted(due_by_owner):
        due = due_by_owner[owner_id]
        summary = summaries[owner_id]
        if summary.get("native_true_positive") is not True:
            _fail(f"transfer owner {owner_id!r} is not a native TP")
        context_id = str(due.get("due_context_id"))
        context_key = f"{owner_id}@{context_id}"
        context = inputs.owner_contexts.get(context_key)
        if context is None or context.get("schema_version") != SOURCE_CONTEXT_SCHEMA_VERSION:
            _fail(f"missing sealed exact due-boundary context {context_key!r}")
        bounds = (context.get("localization") or {}).get(
            "generator_local_max_excluding_other_owner_strict"
        ) or {}
        lower = _criterion_block(
            bounds.get("ambiguity_excluded_l") or {},
            theta_peak=theta_peak,
            theta_concentration=theta_concentration,
            epsilon=epsilon,
        )
        upper = _criterion_block(
            bounds.get("ambiguity_included_u") or {},
            theta_peak=theta_peak,
            theta_concentration=theta_concentration,
            epsilon=epsilon,
        )
        supported = lower["clears_frozen_support_rule"] and upper[
            "clears_frozen_support_rule"
        ]
        if due.get("eligible") is not True or due.get("supported_under_both_bounds") is not supported:
            _fail(f"TP due-boundary support row drifted for {owner_id!r}")
        proposal = context.get("proposal_surface") or {}
        category_event = proposal.get("category_routing_event") or {}
        boundary_gate = proposal.get("boundary_gate") or {}
        hit_ids = sorted(sidecar_hits.get(owner_id, []))
        rows.append(
            {
                "schema_version": TP_ROW_SCHEMA_VERSION,
                "row_kind": "native_tp_frozen_transfer_control",
                "gt_owner_id": owner_id,
                "image_id": IMAGE_ID,
                "normalized_description": str(summary["normalized_description"]),
                "due_context_id": context_id,
                "supported_under_both_bounds": supported,
                "frozen_support_gate": {
                    "theta_peak_lift_plus_epsilon": theta_peak + epsilon,
                    "theta_local_concentration_plus_epsilon": theta_concentration + epsilon,
                    "thresholds_retuned": False,
                    "rank_is_support_input": False,
                },
                "ambiguity_excluded_l": lower,
                "ambiguity_included_u": upper,
                "owner_competition_l": dict(context.get("owner_competition_l") or {}),
                "owner_competition_u": dict(context.get("owner_competition_u") or {}),
                "category_routing": {
                    "rank": category_event.get("within_context_rank"),
                    "population": category_event.get("within_context_population"),
                    "raw_sequence_logprob_sum": category_event.get(
                        "raw_sequence_logprob_sum"
                    ),
                    "role": "routing_diagnostic_never_support_gate",
                },
                "boundary_gate": {
                    "continue_logprob": boundary_gate.get("continue_logprob"),
                    "stop_logprob": boundary_gate.get("stop_logprob"),
                    "continue_vs_stop_logprob_margin": boundary_gate.get(
                        "continue_vs_stop_logprob_margin"
                    ),
                    "role": "proposal_gate_only_never_description_accessibility",
                },
                "query_suffix_sidecar_reachability": {
                    "any_hit": bool(hit_ids),
                    "hit_count": len(hit_ids),
                    "hit_sidecar_ids": hit_ids,
                    "role": "free_descriptive_sidecar_never_support_gate",
                },
            }
        )
    category: dict[str, Any] = {}
    for name in sorted(EXPECTED_TP_CATEGORY_COUNTS):
        group = [row for row in rows if row["normalized_description"] == name]
        supported_count = sum(row["supported_under_both_bounds"] for row in group)
        expected = EXPECTED_TRANSFER_CATEGORY_COUNTS[name]
        if len(group) != expected["total"] or supported_count != expected["supported"]:
            _fail(f"post-hoc TP transfer sensitivity drifted for {name!r}")
        category[name] = {
            "supported": supported_count,
            "total": len(group),
            "rate": supported_count / len(group),
            "role": "post_hoc_sensitivity_only_no_new_threshold",
        }
    return rows, category


def _sidecar_reachability(
    inputs: LoadedInputs,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    canvas = planner.Canvas(int(inputs.image["image_width"]), int(inputs.image["image_height"]))
    hits: dict[str, list[str]] = {}
    assignment_counts: Counter[str] = Counter()
    category_assignment_counts: dict[str, Counter[str]] = {}
    seen_ids: set[str] = set()
    decoded_noncollapsed_count = 0
    for row in inputs.free_box_sidecars:
        sidecar_id = str(row.get("sidecar_id"))
        description = str(row.get("normalized_description"))
        if sidecar_id in seen_ids:
            _fail(f"duplicate free box sidecar ID {sidecar_id!r}")
        seen_ids.add(sidecar_id)
        if (
            row.get("schema_version") != FREE_BOX_SCHEMA_VERSION
            or row.get("channel") != "query_suffix"
            or row.get("is_sidecar") is not True
            or row.get("is_behavior_not_probability") is not True
            or row.get("enters_core_ranks") is not False
            or row.get("generation_phase_after_decision_scoring") is not True
            or row.get("well_formed_box") is not True
            or description not in EXPECTED_CATEGORY_COUNTS
        ):
            _fail(f"free box sidecar contract drifted for {sidecar_id!r}")
        bins = row.get("coord_bins")
        if (
            not isinstance(bins, Sequence)
            or isinstance(bins, (str, bytes))
            or len(bins) != 4
            or any(not isinstance(value, int) or not 0 <= value < 1000 for value in bins)
        ):
            _fail(f"free box sidecar has invalid norm1000 coordinates: {sidecar_id!r}")
        pixel_box = canvas.bins_to_pixel(bins)
        if canvas.valid_pixel_box(pixel_box):
            decoded_noncollapsed_count += 1
        assignment = planner.strict_assignment(
            pixel_box, inputs.owners, normalized_description=description
        )
        status = str(assignment["strict_assignment_status"])
        assignment_counts[status] += 1
        category_assignment_counts.setdefault(description, Counter())[status] += 1
        owner_id = assignment.get("strict_assignment_gt_owner_id")
        if status == "matched":
            if owner_id is None:
                _fail(f"matched sidecar lacks a strict owner: {sidecar_id!r}")
            hits.setdefault(str(owner_id), []).append(sidecar_id)
        elif owner_id is not None:
            _fail(f"non-matched sidecar carries a strict owner: {sidecar_id!r}")
    if Counter(str(row["normalized_description"]) for row in inputs.free_box_sidecars) != Counter(
        {"person": 25, "tie": 25}
    ):
        _fail("the 50 box sidecars are not 25 person + 25 tie")
    return hits, {
        "sidecar_count": len(inputs.free_box_sidecars),
        "well_formed_box_count": len(inputs.free_box_sidecars),
        "decoded_noncollapsed_pixel_box_count": decoded_noncollapsed_count,
        "decoded_collapsed_pixel_box_count": (
            len(inputs.free_box_sidecars) - decoded_noncollapsed_count
        ),
        "assignment_status_counts": dict(sorted(assignment_counts.items())),
        "assignment_status_counts_by_category": {
            key: dict(sorted(value.items()))
            for key, value in sorted(category_assignment_counts.items())
        },
        "unique_strict_owner_hit_count": len(hits),
        "coordinate_conversion": "inherited_Canvas_norm1000_bins_to_pixel_round",
        "assignment_semantics": "inherited_category_local_strict_assignment_iou_0p5",
        "aggregation": "owner_level_any_hit_over_50_box_sidecars",
        "diagnostic_role": "free_descriptive_sidecar_reachability",
        "miss_inference_forbidden": "a_miss_does_not_establish_absent_visual_support",
    }


def _reachability_breakdown(
    summaries: Mapping[str, Mapping[str, Any]], hits: Mapping[str, list[str]]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for native_role, predicate in (
        ("native_tp", lambda row: row.get("native_true_positive") is True),
        ("native_fn", lambda row: row.get("native_false_negative") is True),
    ):
        role_rows = [row for row in summaries.values() if predicate(row)]
        category: dict[str, Any] = {}
        for name in sorted(EXPECTED_CATEGORY_COUNTS):
            group = [row for row in role_rows if row.get("normalized_description") == name]
            successes = sum(str(row["gt_owner_id"]) in hits for row in group)
            category[name] = {
                "owners_with_any_hit": successes,
                "owner_count": len(group),
                "rate": successes / len(group) if group else None,
            }
        successes = sum(str(row["gt_owner_id"]) in hits for row in role_rows)
        result[native_role] = {
            "owners_with_any_hit": successes,
            "owner_count": len(role_rows),
            "rate": successes / len(role_rows),
            "by_category": category,
        }
    return result


def _build_fn_rows(
    summaries: Mapping[str, Mapping[str, Any]], hits: Mapping[str, list[str]]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    for owner_id, summary in sorted(summaries.items()):
        if summary.get("native_false_negative") is not True:
            continue
        disposition = str(summary.get("frozen_disposition_descriptive"))
        hit_ids = sorted(hits.get(owner_id, []))
        rows.append(
            {
                "schema_version": FN_ROW_SCHEMA_VERSION,
                "row_kind": "native_fn_frozen_disposition_descriptive",
                "gt_owner_id": owner_id,
                "image_id": IMAGE_ID,
                "normalized_description": str(summary["normalized_description"]),
                "published_disposition": DISPOSITION_WITHHELD,
                "frozen_disposition_descriptive": disposition,
                "disposition_role": FN_ROLE,
                "valid_visual_recall_rate": False,
                "query_suffix_sidecar_reachability": {
                    "any_hit": bool(hit_ids),
                    "hit_count": len(hit_ids),
                    "hit_sidecar_ids": hit_ids,
                    "role": "free_descriptive_sidecar_never_support_gate",
                    "miss_inference_forbidden": (
                        "a_miss_does_not_establish_absent_visual_support"
                    ),
                },
            }
        )
    if len(rows) != EXPECTED_FN_COUNT:
        _fail("FN descriptive row denominator drifted")
    counts = dict(
        sorted(Counter(row["frozen_disposition_descriptive"] for row in rows).items())
    )
    return rows, counts


def build_audit(inputs: LoadedInputs) -> dict[str, Any]:
    validated = _validate_primary_conclusion(inputs)
    due_by_owner = validated["due_by_owner"]
    summaries = validated["summaries"]
    sidecar_hits, sidecar_summary = _sidecar_reachability(inputs)
    tp_rows, category_transfer = _build_tp_rows(
        inputs, due_by_owner, summaries, sidecar_hits
    )
    fn_rows, fn_descriptive_counts = _build_fn_rows(summaries, sidecar_hits)
    failed_tp = [row for row in tp_rows if not row["supported_under_both_bounds"]]
    if len(failed_tp) != EXPECTED_TP_COUNT - EXPECTED_TRANSFER_SUPPORTED:
        _fail("failed TP control count drifted")
    sidecar_summary["owner_level_breakdown"] = _reachability_breakdown(
        summaries, sidecar_hits
    )

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "image_id": IMAGE_ID,
        "execution": {
            "cpu_only": True,
            "gpu_used": False,
            "inference_run": False,
            "thresholds_fit_or_proposed": False,
            "raw_cross_image_pooling": False,
        },
        "primary_conclusion": {
            "calibration_transfer": {
                "supported": EXPECTED_TRANSFER_SUPPORTED,
                "total": EXPECTED_TP_COUNT,
                "rate": EXPECTED_TRANSFER_SUPPORTED / EXPECTED_TP_COUNT,
                "frozen_floor": TRANSFER_FLOOR,
                "passes": False,
                "status": "calibration_nontransferring",
            },
            "native_fn_dispositions_withheld": EXPECTED_FN_COUNT,
            "s2_gate_open": False,
            "s3_gate_open": False,
            "action": "stop_after_s1_transfer_audit",
        },
        "post_hoc_tp_category_sensitivity": category_transfer,
        "failed_tp_exact_due_boundary_details": failed_tp,
        "fn_frozen_disposition_descriptive": {
            "role": FN_ROLE,
            "owner_count": EXPECTED_FN_COUNT,
            "counts": fn_descriptive_counts,
            "valid_visual_recall_rate": False,
        },
        "free_query_suffix_sidecar_reachability": sidecar_summary,
        "claim_boundary": {
            "no_new_threshold": True,
            "no_raw_cross_image_pooling": True,
            "fn_dispositions_validity_bearing": False,
            "sidecar_reachability_valid_visual_recall_rate": False,
            "sidecar_misses_imply_absent_visual_support": False,
            "s2_or_s3_authorized": False,
        },
    }
    return {"report": report, "tp_rows": tp_rows, "fn_rows": fn_rows}


def render_markdown(report: Mapping[str, Any]) -> str:
    transfer = report["primary_conclusion"]["calibration_transfer"]
    category = report["post_hoc_tp_category_sensitivity"]
    sidecar = report["free_query_suffix_sidecar_reachability"]
    lines = [
        "# Image 2299 sealed calibration-transfer audit",
        "",
        "## Primary conclusion",
        "",
        f"Frozen transfer is **{transfer['supported']}/{transfer['total']} = "
        f"{transfer['rate']:.3f}**, below the frozen {transfer['frozen_floor']:.1f} gate.",
        "All **27** native-FN dispositions remain withheld. S2 and S3 do not open.",
        "",
        "## Post-hoc TP category sensitivity",
        "",
        f"- person: {category['person']['supported']}/{category['person']['total']} = "
        f"{category['person']['rate']:.3f}",
        f"- tie: {category['tie']['supported']}/{category['tie']['total']} = "
        f"{category['tie']['rate']:.3f}",
        "",
        "These are sensitivities under the unchanged frozen gate, not fitted category thresholds.",
        "",
        "## Failed native-TP controls at the exact due boundary",
        "",
        "| owner | category | due boundary | L failed | U failed | owner rank L/U | category rank | continue-stop margin |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in report["failed_tp_exact_due_boundary_details"]:
        lower = ", ".join(row["ambiguity_excluded_l"]["failed_frozen_criteria"]) or "none"
        upper = ", ".join(row["ambiguity_included_u"]["failed_frozen_criteria"]) or "none"
        lines.append(
            f"| {row['gt_owner_id']} | {row['normalized_description']} | "
            f"{row['due_context_id']} | {lower} | {upper} | "
            f"{row['owner_competition_l'].get('rank')}/"
            f"{row['owner_competition_u'].get('rank')} | "
            f"{row['category_routing'].get('rank')} | "
            f"{row['boundary_gate'].get('continue_vs_stop_logprob_margin'):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Native-FN descriptive dispositions",
            "",
            f"Role: `{FN_ROLE}`. Counts: "
            f"`{json.dumps(report['fn_frozen_disposition_descriptive']['counts'], sort_keys=True)}`.",
            "These counts are not a valid visual-recall rate.",
            "",
            "## Free query-suffix sidecar reachability",
            "",
            f"The diagnostic uses owner-level any-hit over {sidecar['sidecar_count']} sealed box "
            "sidecars with inherited Canvas norm1000-to-pixel conversion and category-local "
            "strict assignment.",
        ]
    )
    for role in ("native_tp", "native_fn"):
        block = sidecar["owner_level_breakdown"][role]
        lines.append(
            f"- {role}: {block['owners_with_any_hit']}/{block['owner_count']} owners with any hit "
            f"(person {block['by_category']['person']['owners_with_any_hit']}/"
            f"{block['by_category']['person']['owner_count']}; tie "
            f"{block['by_category']['tie']['owners_with_any_hit']}/"
            f"{block['by_category']['tie']['owner_count']})"
        )
    lines.extend(
        [
            "",
            "A sidecar miss does not establish that an owner lacks visual support. This free "
            "diagnostic does not change the transfer gate or authorize S2/S3.",
            "",
        ]
    )
    return "\n".join(lines)


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) + b"\n" for row in rows)


def materialize_files(
    result: Mapping[str, Any], inputs: LoadedInputs, analyzer_path: Path | None = None
) -> dict[str, bytes]:
    report_json = canonical_json_bytes(result["report"]) + b"\n"
    report_md = render_markdown(result["report"]).encode("utf-8")
    tp_rows = _jsonl_bytes(result["tp_rows"])
    fn_rows = _jsonl_bytes(result["fn_rows"])
    primary = {
        REPORT_JSON_NAME: report_json,
        REPORT_MD_NAME: report_md,
        TP_ROWS_NAME: tp_rows,
        FN_ROWS_NAME: fn_rows,
    }
    source_path = Path(__file__).resolve() if analyzer_path is None else analyzer_path
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "image_id": IMAGE_ID,
        "input_directories": {
            "s0_native": str(inputs.sources.s0_native),
            "s1_plan": str(inputs.sources.s1_plan),
            "s1_shard": str(inputs.sources.s1_shard),
            "s1_analysis": str(inputs.sources.s1_analysis),
        },
        "input_file_sha256": inputs.input_file_sha256,
        "analyzer_source_sha256": sha256_file(source_path),
        "output_file_sha256": {
            name: sha256_bytes(content) for name, content in sorted(primary.items())
        },
        "primary_conclusion": {
            "transfer_supported": EXPECTED_TRANSFER_SUPPORTED,
            "transfer_total": EXPECTED_TP_COUNT,
            "transfer_passes": False,
            "native_fn_dispositions_withheld": EXPECTED_FN_COUNT,
            "s2_gate_open": False,
            "s3_gate_open": False,
        },
        "publication_policy": "atomic_create_or_identical",
        "cpu_only": True,
    }
    receipt["receipt_content_sha256"] = _receipt_content_sha256(receipt)
    return {**primary, RECEIPT_NAME: canonical_json_bytes(receipt) + b"\n"}


def publish_create_or_identical(output_dir: Path, files: Mapping[str, bytes]) -> str:
    output_dir = Path(output_dir)

    def validate_existing() -> None:
        if output_dir.is_symlink() or not output_dir.is_dir():
            _fail(f"output exists but is not a regular directory: {output_dir}")
        entries = {path.name: path for path in output_dir.iterdir()}
        if set(entries) != OUTPUT_NAMES or any(
            path.is_symlink() or not path.is_file() for path in entries.values()
        ):
            _fail("existing output directory contains a foreign or partial file set")
        if any(entries[name].read_bytes() != content for name, content in files.items()):
            _fail("existing output directory is not byte-identical")

    if output_dir.exists() or output_dir.is_symlink():
        validate_existing()
        return "identical_existing_output"
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    published = False
    try:
        for name, content in sorted(files.items()):
            (staging / name).write_bytes(content)
        try:
            os.replace(staging, output_dir)
            published = True
        except OSError as exc:
            if not output_dir.exists() and not output_dir.is_symlink():
                _fail(f"atomic output publication failed for {output_dir}: {exc}")
            validate_existing()
        return "created" if published else "identical_existing_output"
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s0-native-dir", type=Path, required=True)
    parser.add_argument("--s1-plan-dir", type=Path, required=True)
    parser.add_argument("--s1-shard-dir", type=Path, required=True)
    parser.add_argument("--s1-analysis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    sources = SourceDirs(
        s0_native=args.s0_native_dir,
        s1_plan=args.s1_plan_dir,
        s1_shard=args.s1_shard_dir,
        s1_analysis=args.s1_analysis_dir,
    )
    try:
        inputs = load_inputs(sources)
        result = build_audit(inputs)
        files = materialize_files(result, inputs)
        status = publish_create_or_identical(args.output_dir, files)
    except AuditContractError as exc:
        raise SystemExit(f"audit contract violated: {exc}") from exc
    report = result["report"]
    sidecar = report["free_query_suffix_sidecar_reachability"]["owner_level_breakdown"]
    receipt = json.loads(files[RECEIPT_NAME])
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "publication_status": status,
                "transfer": "14/19",
                "person_transfer": "13/16",
                "tie_transfer": "1/3",
                "native_fn_dispositions_withheld": 27,
                "sidecar_native_tp_any_hit": (
                    f"{sidecar['native_tp']['owners_with_any_hit']}/"
                    f"{sidecar['native_tp']['owner_count']}"
                ),
                "sidecar_native_fn_any_hit": (
                    f"{sidecar['native_fn']['owners_with_any_hit']}/"
                    f"{sidecar['native_fn']['owner_count']}"
                ),
                "receipt_content_sha256": receipt["receipt_content_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
