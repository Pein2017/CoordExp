#!/usr/bin/env python3
"""Reconstruct the sealed legacy TP-control context for image 2299.

This CPU-only audit applies the already frozen native-TP due-boundary support
rule to the twelve legacy images and describes the resulting instrument
context next to image 2299's asserted 14/19 transfer result.  It does not
retune, create a threshold, retroactively re-gate the legacy study, or support
any false-negative mechanism or prevalence conclusion.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, NoReturn


LEGACY_UNIT_ID = "2026-08-03-sorted-owner-accessibility-phenotype-census"
CURRENT_UNIT_ID = "2026-08-04-sorted-image2299-prospective-mechanism-extension"
IMAGE_ID = "2299"
REFERENCE_FLOOR = 0.8

REPORT_SCHEMA_VERSION = "sorted-image2299-legacy-transfer-context.v1"
ROW_SCHEMA_VERSION = "sorted-image2299-legacy-transfer-context-control.v1"
RECEIPT_SCHEMA_VERSION = "sorted-image2299-legacy-transfer-context-receipt.v1"

REPORT_JSON_NAME = "report.json"
REPORT_MD_NAME = "report.md"
ROWS_NAME = "legacy-tp-control-rows.jsonl"
RECEIPT_NAME = "receipt.json"
OUTPUT_NAMES = frozenset({REPORT_JSON_NAME, REPORT_MD_NAME, ROWS_NAME, RECEIPT_NAME})

EXPECTED_DISCOVERY_IMAGES = frozenset({"10707", "14038", "2685", "5001", "6040", "7511"})
EXPECTED_CONFIRMATION_IMAGES = frozenset({"13348", "13923", "14439", "1584", "16228", "4134"})
EXPECTED_LEGACY_IMAGES = EXPECTED_DISCOVERY_IMAGES | EXPECTED_CONFIRMATION_IMAGES
EXPECTED_LEGACY_OWNER_COUNT = 346
EXPECTED_DISCOVERY_TP_COUNT = 70
EXPECTED_CONFIRMATION_TP_COUNT = 71
EXPECTED_POOLED_TP_COUNT = 141
EXPECTED_2299_SUPPORTED = 14
EXPECTED_2299_TOTAL = 19
FROZEN_CALIBRATION_CONTENT_SHA256 = (
    "9dd6d7646fc55db6155124dc4bbfa46642b32b006d39758bd2e24d1ca97058c5"
)


class AuditContractError(RuntimeError):
    """A named sealed input or conclusion-bearing invariant changed."""


def _fail(message: str) -> NoReturn:
    raise AuditContractError(message)


@dataclass(frozen=True)
class Inputs:
    legacy_root: Path
    s1_analysis_dir: Path
    input_file_sha256: dict[str, str]
    calibration: dict[str, Any]
    owners: list[dict[str, Any]]
    native_sidecars: list[dict[str, Any]]
    contexts_by_split: dict[str, dict[tuple[str, str], dict[str, Any]]]
    summaries_by_split: dict[str, dict[str, dict[str, Any]]]
    image2299_analysis: dict[str, Any]


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


def _content_seal(value: Mapping[str, Any], field: str) -> str:
    return sha256_bytes(canonical_json_bytes({k: v for k, v in value.items() if k != field}))


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
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        _fail(f"{label} is unreadable at {path}: {exc}")
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


def _verify_receipt(receipt: Mapping[str, Any], label: str) -> None:
    declared = receipt.get("receipt_content_sha256")
    observed = _content_seal(receipt, "receipt_content_sha256")
    if declared != observed:
        _fail(f"{label} content seal drifted: {declared!r} != {observed!r}")


def _verify_declared_digest(
    path: Path, declared: Mapping[str, Any], name: str, label: str
) -> str:
    expected = declared.get(name)
    observed = sha256_file(path)
    if not isinstance(expected, str) or observed != expected:
        _fail(f"{label} digest drifted for {name}: {observed} != {expected!r}")
    return observed


def load_inputs(legacy_root: Path, s1_analysis_dir: Path) -> Inputs:
    legacy_root = Path(legacy_root).resolve()
    s1_analysis_dir = Path(s1_analysis_dir).resolve()
    plan_dir = legacy_root / "plan"
    discovery_dir = legacy_root / "phases" / "discovery-sealed"
    confirmation_dir = legacy_root / "phases" / "confirmation"

    paths = {
        "legacy/plan/receipt.json": plan_dir / "receipt.json",
        "legacy/plan/owner-registry.jsonl": plan_dir / "owner-registry.jsonl",
        "legacy/plan/native-sidecar-registry.jsonl": plan_dir / "native-sidecar-registry.jsonl",
        "legacy/discovery/support-calibration.json": discovery_dir / "support-calibration.json",
        "legacy/discovery/owner-context-features.jsonl": discovery_dir / "owner-context-features.jsonl",
        "legacy/discovery/owner-summaries.jsonl": discovery_dir / "owner-summaries.jsonl",
        "legacy/discovery/merge-receipt.json": discovery_dir / "merge-receipt.json",
        "legacy/confirmation/owner-context-features.jsonl": confirmation_dir / "owner-context-features.jsonl",
        "legacy/confirmation/owner-summaries.jsonl": confirmation_dir / "owner-summaries.jsonl",
        "legacy/confirmation/merge-receipt.json": confirmation_dir / "merge-receipt.json",
        "image2299/s1-analysis/analysis.json": s1_analysis_dir / "analysis.json",
        "image2299/s1-analysis/receipt.json": s1_analysis_dir / "receipt.json",
    }

    plan_receipt = _read_json(paths["legacy/plan/receipt.json"], "legacy plan receipt")
    discovery_receipt = _read_json(
        paths["legacy/discovery/merge-receipt.json"], "legacy discovery receipt"
    )
    confirmation_receipt = _read_json(
        paths["legacy/confirmation/merge-receipt.json"], "legacy confirmation receipt"
    )
    s1_receipt = _read_json(paths["image2299/s1-analysis/receipt.json"], "S1 receipt")
    for receipt, label in (
        (plan_receipt, "legacy plan receipt"),
        (discovery_receipt, "legacy discovery receipt"),
        (confirmation_receipt, "legacy confirmation receipt"),
        (s1_receipt, "S1 receipt"),
    ):
        _verify_receipt(receipt, label)

    if plan_receipt.get("unit_id") != LEGACY_UNIT_ID:
        _fail("legacy plan receipt belongs to another unit")
    for receipt, phase in ((discovery_receipt, "discovery"), (confirmation_receipt, "confirmation")):
        if receipt.get("unit_id") != LEGACY_UNIT_ID or receipt.get("phase") != phase:
            _fail(f"legacy {phase} receipt identity drifted")
    if s1_receipt.get("unit_id") != CURRENT_UNIT_ID:
        _fail("S1 receipt belongs to another unit")

    plan_digests = plan_receipt.get("output_file_digests") or {}
    discovery_digests = discovery_receipt.get("output_file_digests") or {}
    confirmation_digests = confirmation_receipt.get("output_file_digests") or {}
    s1_digests = s1_receipt.get("output_file_digests") or {}
    _verify_declared_digest(
        paths["legacy/plan/owner-registry.jsonl"], plan_digests,
        "owner-registry.jsonl", "legacy plan",
    )
    _verify_declared_digest(
        paths["legacy/plan/native-sidecar-registry.jsonl"], plan_digests,
        "native-sidecar-registry.jsonl", "legacy plan",
    )
    for split, digests in (("discovery", discovery_digests), ("confirmation", confirmation_digests)):
        _verify_declared_digest(
            paths[f"legacy/{split}/owner-context-features.jsonl"], digests,
            "owner-context-features.jsonl", f"legacy {split}",
        )
        _verify_declared_digest(
            paths[f"legacy/{split}/owner-summaries.jsonl"], digests,
            "owner-summaries.jsonl", f"legacy {split}",
        )
    _verify_declared_digest(
        paths["image2299/s1-analysis/analysis.json"], s1_digests,
        "analysis.json", "S1 analysis",
    )

    calibration = _read_json(
        paths["legacy/discovery/support-calibration.json"], "frozen support calibration"
    )
    if (
        calibration.get("unit_id") != LEGACY_UNIT_ID
        or calibration.get("calibration_sha256") != FROZEN_CALIBRATION_CONTENT_SHA256
        or _content_seal(calibration, "calibration_sha256") != FROZEN_CALIBRATION_CONTENT_SHA256
        or calibration.get("rank_is_not_a_support_input") is not True
        or calibration.get("confirmation_evidence_consumed") is not False
    ):
        _fail("frozen support calibration identity drifted")

    image2299_analysis = _read_json(
        paths["image2299/s1-analysis/analysis.json"], "image-2299 S1 analysis"
    )
    s1_calibration = image2299_analysis.get("calibration") or {}
    if (
        s1_calibration.get("content_sha256") != FROZEN_CALIBRATION_CONTENT_SHA256
        or s1_calibration.get("file_sha256")
        != sha256_file(paths["legacy/discovery/support-calibration.json"])
        or s1_calibration.get("thresholds_retuned") is not False
        or s1_calibration.get("phenotype_fitted") is not False
    ):
        _fail("image-2299 S1 does not bind the unchanged frozen calibration")

    owners = _read_jsonl(paths["legacy/plan/owner-registry.jsonl"], "legacy owners")
    native_sidecars = _read_jsonl(
        paths["legacy/plan/native-sidecar-registry.jsonl"], "legacy native sidecars"
    )
    contexts_by_split: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    summaries_by_split: dict[str, dict[str, dict[str, Any]]] = {}
    for split in ("discovery", "confirmation"):
        context_rows = _read_jsonl(
            paths[f"legacy/{split}/owner-context-features.jsonl"], f"legacy {split} contexts"
        )
        summary_rows = _read_jsonl(
            paths[f"legacy/{split}/owner-summaries.jsonl"], f"legacy {split} summaries"
        )
        contexts: dict[tuple[str, str], dict[str, Any]] = {}
        for row in context_rows:
            key = (str(row.get("gt_owner_id")), str(row.get("context_id")))
            if key in contexts:
                _fail(f"duplicate legacy {split} owner-context key {key!r}")
            contexts[key] = row
        summaries: dict[str, dict[str, Any]] = {}
        for row in summary_rows:
            owner_id = str(row.get("gt_owner_id"))
            if owner_id in summaries:
                _fail(f"duplicate legacy {split} owner summary {owner_id!r}")
            summaries[owner_id] = row
        contexts_by_split[split] = contexts
        summaries_by_split[split] = summaries

    input_file_sha256 = {name: sha256_file(path) for name, path in sorted(paths.items())}
    return Inputs(
        legacy_root=legacy_root,
        s1_analysis_dir=s1_analysis_dir,
        input_file_sha256=input_file_sha256,
        calibration=calibration,
        owners=owners,
        native_sidecars=native_sidecars,
        contexts_by_split=contexts_by_split,
        summaries_by_split=summaries_by_split,
        image2299_analysis=image2299_analysis,
    )


def _validate_image2299_assertion(analysis: Mapping[str, Any]) -> dict[str, Any]:
    if (
        analysis.get("schema_version") != "sorted-image2299-owner-accessibility-analysis.v1"
        or analysis.get("unit_id") != CURRENT_UNIT_ID
        or str(analysis.get("image_id")) != IMAGE_ID
    ):
        _fail("image-2299 S1 analysis identity drifted")
    transfer = analysis.get("calibration_transfer") or {}
    expected_rate = EXPECTED_2299_SUPPORTED / EXPECTED_2299_TOTAL
    if (
        transfer.get("supported_due_boundary_count") != EXPECTED_2299_SUPPORTED
        or transfer.get("transfer_denominator_native_tp_count") != EXPECTED_2299_TOTAL
        or not math.isclose(float(transfer.get("support_rate", math.nan)), expected_rate)
        or transfer.get("floor") != REFERENCE_FLOOR
        or transfer.get("passes") is not False
        or transfer.get("status") != "calibration_nontransferring"
    ):
        _fail("current image-2299 14/19 assertion drifted")
    return {
        "image_id": IMAGE_ID,
        "supported": EXPECTED_2299_SUPPORTED,
        "total": EXPECTED_2299_TOTAL,
        "rate": expected_rate,
        "frozen_floor": REFERENCE_FLOOR,
        "passes": False,
        "status": "calibration_nontransferring",
        "asserted_from_current_s1": True,
    }


def reconstruct_legacy_controls(inputs: Inputs) -> list[dict[str, Any]]:
    if len(inputs.owners) != EXPECTED_LEGACY_OWNER_COUNT:
        _fail(f"legacy owner denominator drifted: {len(inputs.owners)}")
    owner_ids = [str(row.get("gt_owner_id")) for row in inputs.owners]
    if len(set(owner_ids)) != len(owner_ids):
        _fail("legacy owner registry has duplicate owner IDs")
    image_ids = {str(row.get("image_id")) for row in inputs.owners}
    if image_ids != EXPECTED_LEGACY_IMAGES:
        _fail(f"legacy image panel drifted: {sorted(image_ids)}")

    sidecar_by_pred: dict[str, dict[str, Any]] = {}
    for row in inputs.native_sidecars:
        pred_id = str(row.get("pred_row_id"))
        if pred_id in sidecar_by_pred:
            _fail(f"duplicate legacy native prediction ID {pred_id!r}")
        sidecar_by_pred[pred_id] = row

    calibration = inputs.calibration
    peak_gate = float(calibration["theta_peak_lift"]) + float(calibration["epsilon"])
    concentration_gate = (
        float(calibration["theta_local_concentration"]) + float(calibration["epsilon"])
    )
    rows: list[dict[str, Any]] = []
    plan_ids_by_split: dict[str, set[str]] = defaultdict(set)
    for owner in inputs.owners:
        split = str(owner.get("split"))
        if split not in {"discovery", "confirmation"}:
            _fail(f"unknown legacy split {split!r}")
        owner_id = str(owner.get("gt_owner_id"))
        plan_ids_by_split[split].add(owner_id)
        summary = inputs.summaries_by_split[split].get(owner_id)
        if summary is None:
            _fail(f"legacy owner summary is missing for {owner_id!r}")
        for field in ("image_id", "normalized_description", "native_true_positive"):
            if summary.get(field) != owner.get(field):
                _fail(f"legacy owner summary {field} drifted for {owner_id!r}")
        if owner.get("native_true_positive") is not True:
            continue
        matches = owner.get("native_strict_match_pred_row_ids")
        if not isinstance(matches, list) or len(matches) != 1:
            _fail(f"native TP {owner_id!r} lacks exactly one strict native match")
        sidecar = sidecar_by_pred.get(str(matches[0]))
        if (
            sidecar is None
            or sidecar.get("strict_match_status") != "matched"
            or str(sidecar.get("strict_match_gt_owner_id")) != owner_id
            or str(sidecar.get("image_id")) != str(owner.get("image_id"))
        ):
            _fail(f"native TP sidecar identity drifted for {owner_id!r}")
        row_index = sidecar.get("row_index")
        if not isinstance(row_index, int) or row_index < 0:
            _fail(f"native TP sidecar row index is invalid for {owner_id!r}")
        context_id = f"{owner['image_id']}:boundary-{row_index:03d}"
        context = inputs.contexts_by_split[split].get((owner_id, context_id))
        if context is None:
            _fail(f"exact native-TP due-boundary context is missing for {owner_id!r}")
        if (context.get("loop_marking") or {}).get("loop_tail") is not False:
            _fail(f"native-TP due boundary is not an eligible non-loop context for {owner_id!r}")
        bounds = ((context.get("localization") or {}).get(
            "generator_local_max_excluding_other_owner_strict"
        ) or {})
        bound_results: dict[str, dict[str, Any]] = {}
        for bound_name in ("ambiguity_excluded_l", "ambiguity_included_u"):
            block = bounds.get(bound_name) or {}
            peak_lift = float(block.get("peak_lift", math.nan))
            concentration = float(block.get("local_concentration", math.nan))
            clears = peak_lift >= peak_gate and concentration >= concentration_gate
            declared_clears = block.get("clears_frozen_support_rule")
            if declared_clears is not None and declared_clears is not clears:
                _fail(f"sealed support flag disagrees with frozen rule for {owner_id!r}")
            bound_results[bound_name] = {
                "peak_lift": peak_lift,
                "local_concentration": concentration,
                "clears_frozen_support_rule": clears,
            }
        supported = all(
            block["clears_frozen_support_rule"] for block in bound_results.values()
        )
        rows.append(
            {
                "schema_version": ROW_SCHEMA_VERSION,
                "row_kind": "legacy_native_tp_exact_due_boundary_control",
                "split": split,
                "image_id": str(owner["image_id"]),
                "gt_owner_id": owner_id,
                "normalized_description": str(owner["normalized_description"]),
                "native_pred_row_id": str(matches[0]),
                "due_context_id": context_id,
                "supported_under_both_bounds": supported,
                "ambiguity_excluded_l": bound_results["ambiguity_excluded_l"],
                "ambiguity_included_u": bound_results["ambiguity_included_u"],
            }
        )

    for split in ("discovery", "confirmation"):
        if set(inputs.summaries_by_split[split]) != plan_ids_by_split[split]:
            _fail(f"legacy {split} owner summaries do not exactly cover the plan")
    split_counts = Counter(row["split"] for row in rows)
    if split_counts != Counter(
        {"discovery": EXPECTED_DISCOVERY_TP_COUNT, "confirmation": EXPECTED_CONFIRMATION_TP_COUNT}
    ):
        _fail(f"legacy TP-control split drifted: {dict(split_counts)}")
    return sorted(rows, key=lambda row: (int(row["image_id"]), row["gt_owner_id"]))


def _rate(supported: int, total: int) -> dict[str, Any]:
    if total <= 0:
        _fail("cannot report a zero-denominator TP-control rate")
    return {"supported": supported, "total": total, "rate": supported / total}


def _category_composition(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split in ("discovery", "confirmation", "pooled"):
        group = list(rows) if split == "pooled" else [row for row in rows if row["split"] == split]
        counts = Counter(str(row["normalized_description"]) for row in group)
        supported = Counter(
            str(row["normalized_description"])
            for row in group
            if row["supported_under_both_bounds"] is True
        )
        result[split] = {
            "total": len(group),
            "category_counts": dict(sorted(counts.items())),
            "supported_category_counts": dict(sorted(supported.items())),
            "tie": {
                "count": counts.get("tie", 0),
                "fraction_of_controls": counts.get("tie", 0) / len(group),
                "supported": supported.get("tie", 0),
            },
        }
    if result["discovery"]["tie"]["count"] != 0:
        _fail("discovery calibration unexpectedly contains tie controls")
    if result["pooled"]["tie"] != {
        "count": 1,
        "fraction_of_controls": 1 / EXPECTED_POOLED_TP_COUNT,
        "supported": 1,
    }:
        _fail("whole-panel tie-control support drifted")
    return result


def build_report(
    control_rows: Sequence[Mapping[str, Any]], image2299_analysis: Mapping[str, Any]
) -> dict[str, Any]:
    if len(control_rows) != EXPECTED_POOLED_TP_COUNT:
        _fail(f"legacy pooled TP-control denominator drifted: {len(control_rows)}")
    per_image: list[dict[str, Any]] = []
    for image_id in sorted(EXPECTED_LEGACY_IMAGES, key=int):
        group = [row for row in control_rows if str(row["image_id"]) == image_id]
        if not group:
            _fail(f"legacy image {image_id} has no TP controls")
        split_values = {str(row["split"]) for row in group}
        if len(split_values) != 1:
            _fail(f"legacy image {image_id} crosses phase splits")
        supported = sum(row["supported_under_both_bounds"] is True for row in group)
        item = {
            "image_id": image_id,
            "split": next(iter(split_values)),
            **_rate(supported, len(group)),
        }
        item["below_fixed_0_8_reference_floor"] = item["rate"] < REFERENCE_FLOOR
        per_image.append(item)

    phase_rates: dict[str, Any] = {}
    for split in ("discovery", "confirmation", "pooled"):
        group = list(control_rows) if split == "pooled" else [
            row for row in control_rows if row["split"] == split
        ]
        phase_rates[split] = _rate(
            sum(row["supported_under_both_bounds"] is True for row in group), len(group)
        )

    rates = [float(row["rate"]) for row in per_image]
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "unit_id": CURRENT_UNIT_ID,
        "report_role": "descriptive_instrument_context_only",
        "frozen_support_rule": {
            "due_boundary": (
                "row index of each owner's unique strict native match from the sealed "
                "native-sidecar registry"
            ),
            "criterion": (
                "peak_lift >= theta_peak_lift + epsilon AND local_concentration >= "
                "theta_local_concentration + epsilon under both ambiguity bounds"
            ),
            "calibration_sha256": FROZEN_CALIBRATION_CONTENT_SHA256,
            "rank_is_support_input": False,
            "thresholds_retuned": False,
        },
        "image2299_transfer_assertion": _validate_image2299_assertion(image2299_analysis),
        "legacy_per_image_tp_control_rates": per_image,
        "legacy_per_image_context": {
            "image_count": len(per_image),
            "rate_range": {"minimum": min(rates), "maximum": max(rates)},
            "fixed_reference_floor": REFERENCE_FLOOR,
            "count_below_fixed_reference_floor": sum(rate < REFERENCE_FLOOR for rate in rates),
            "reference_floor_role": "contextual_comparison_only_not_a_legacy_gate",
        },
        "legacy_phase_rates": phase_rates,
        "calibration_control_category_composition": _category_composition(control_rows),
        "claim_boundary": {
            "descriptive_instrument_context_only": True,
            "retroactive_legacy_regate": False,
            "new_threshold_created": False,
            "false_negative_mechanism_conclusion": False,
            "false_negative_prevalence_conclusion": False,
            "statement": (
                "Descriptive instrument-context only: this is not a retroactive re-gate, "
                "creates no new threshold, and supports no FN mechanism or prevalence conclusion."
            ),
        },
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    assertion = report["image2299_transfer_assertion"]
    context = report["legacy_per_image_context"]
    phases = report["legacy_phase_rates"]
    composition = report["calibration_control_category_composition"]
    lines = [
        "# Image 2299 legacy transfer context",
        "",
        (
            f"Current S1 is asserted at **{assertion['supported']}/{assertion['total']} "
            f"({assertion['rate']:.6f})**, below the unchanged 0.8 floor."
        ),
        "",
        "## Legacy native-TP controls at the exact due boundary",
        "",
        "| image | split | supported | total | rate | below 0.8 reference |",
        "|---:|:---|---:|---:|---:|:---:|",
    ]
    for row in report["legacy_per_image_tp_control_rates"]:
        lines.append(
            f"| {row['image_id']} | {row['split']} | {row['supported']} | "
            f"{row['total']} | {row['rate']:.6f} | "
            f"{'yes' if row['below_fixed_0_8_reference_floor'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            (
                f"Per-image range: {context['rate_range']['minimum']:.6f} to "
                f"{context['rate_range']['maximum']:.6f}; "
                f"{context['count_below_fixed_reference_floor']}/12 images are below 0.8."
            ),
            "",
            "## Legacy phase summaries",
            "",
            f"- Discovery: {phases['discovery']['supported']}/{phases['discovery']['total']} "
            f"({phases['discovery']['rate']:.6f})",
            "",
            f"- Confirmation: {phases['confirmation']['supported']}/{phases['confirmation']['total']} "
            f"({phases['confirmation']['rate']:.6f})",
            "",
            f"- Pooled: {phases['pooled']['supported']}/{phases['pooled']['total']} "
            f"({phases['pooled']['rate']:.6f})",
            "",
            "## Calibration-control composition",
            "",
            (
                f"Discovery has {composition['discovery']['tie']['count']} tie controls. "
                f"The whole legacy panel has only {composition['pooled']['tie']['count']}/"
                f"{composition['pooled']['total']} tie control, and it clears the frozen rule "
                f"({composition['pooled']['tie']['supported']}/1)."
            ),
            "",
            "Full category counts are preserved in `report.json`.",
            "",
            "## Claim boundary",
            "",
            report["claim_boundary"]["statement"],
            "",
        ]
    )
    return "\n".join(lines)


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) + b"\n" for row in rows)


def materialize_files(
    report: Mapping[str, Any], control_rows: Sequence[Mapping[str, Any]], inputs: Inputs,
    analyzer_path: Path | None = None,
) -> dict[str, bytes]:
    primary = {
        REPORT_JSON_NAME: canonical_json_bytes(report) + b"\n",
        REPORT_MD_NAME: render_markdown(report).encode("utf-8"),
        ROWS_NAME: _jsonl_bytes(control_rows),
    }
    source_path = Path(__file__).resolve() if analyzer_path is None else Path(analyzer_path)
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "unit_id": CURRENT_UNIT_ID,
        "input_roots": {
            "legacy_run_root": str(inputs.legacy_root),
            "s1_analysis_dir": str(inputs.s1_analysis_dir),
        },
        "input_file_sha256": inputs.input_file_sha256,
        "analyzer_source_sha256": sha256_file(source_path),
        "output_file_sha256": {name: sha256_bytes(content) for name, content in sorted(primary.items())},
        "legacy_control_count": len(control_rows),
        "image2299_transfer_assertion": "14/19",
        "publication_policy": "atomic_create_or_identical",
        "cpu_only": True,
        "claim_role": "descriptive_instrument_context_only",
    }
    receipt["receipt_content_sha256"] = _content_seal(receipt, "receipt_content_sha256")
    return {**primary, RECEIPT_NAME: canonical_json_bytes(receipt) + b"\n"}


def publish_create_or_identical(output_dir: Path, files: Mapping[str, bytes]) -> str:
    output_dir = Path(output_dir)
    if set(files) != OUTPUT_NAMES:
        _fail("materialized output file set is incomplete")

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
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))
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
    parser.add_argument("--legacy-run-root", type=Path, required=True)
    parser.add_argument("--s1-analysis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        inputs = load_inputs(args.legacy_run_root, args.s1_analysis_dir)
        rows = reconstruct_legacy_controls(inputs)
        report = build_report(rows, inputs.image2299_analysis)
        files = materialize_files(report, rows, inputs)
        status = publish_create_or_identical(args.output_dir, files)
    except AuditContractError as exc:
        raise SystemExit(f"audit contract violated: {exc}") from exc
    receipt = json.loads(files[RECEIPT_NAME])
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "publication_status": status,
                "image2299_transfer_assertion": "14/19",
                "legacy_discovery": "55/70",
                "legacy_confirmation": "67/71",
                "legacy_pooled": "122/141",
                "legacy_images_below_0_8": 3,
                "receipt_content_sha256": receipt["receipt_content_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
