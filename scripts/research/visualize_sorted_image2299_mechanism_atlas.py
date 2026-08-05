#!/usr/bin/env python3
"""Render the score-independent prospective image-2299 mechanism atlas.

The atlas consumes the sealed S0 native ledger, S1 plan and S1 analysis.  A
sealed S2 reachability analysis may be attached after it exists.  Every one of
the 46 physical owners is rendered in original annotation order; no score,
rank, margin, or threshold is read for case selection.
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

from scripts.research import analyze_sorted_image2299_owner_accessibility as s1  # noqa: E402
from scripts.research import analyze_sorted_image2299_supported_fn_reachability as s2  # noqa: E402
from scripts.research import build_sorted_image2299_native_ledger as s0  # noqa: E402
from scripts.research import build_sorted_owner_accessibility_census_plan as legacy_plan  # noqa: E402
from scripts.research.visualize_sorted_owner_accessibility_visual_atlas import (  # noqa: E402
    CROP_PADDING_PIXELS,
    fixed_padding_crop,
)

UNIT_ID = s1.UNIT_ID
IMAGE_ID = s1.IMAGE_ID
EXPECTED_OWNER_COUNT = s1.EXPECTED_OWNER_COUNT
VISUAL_SCHEMA_VERSION = "sorted-image2299-mechanism-atlas.v1"
MANIFEST_SCHEMA_VERSION = "sorted-image2299-mechanism-atlas-manifest.v1"
SPEC_NAME = "atlas-spec.json"
MANIFEST_NAME = "visual-manifest.json"
OWNER_MAP_NAMES = {
    "all": "owner-map-all.png",
    "person": "owner-map-person.png",
    "tie": "owner-map-tie.png",
}
OWNERS_PER_CROP_PANEL = 9

CLASS_NATIVE_TP = "native_true_positive"
CLASS_RESOLVED = "resolved_tested_localization_support"
CLASS_PERSISTENT = "persistent_no_tested_localization_support"
CLASS_AMBIGUITY_FLIP = "ambiguity_bound_disposition_flip"
CLASS_UNRESOLVED = "unresolved_insufficient_tested_localization_support"
CLASS_WITHHELD = s1.DISPOSITION_WITHHELD
CLASS_UNDERPOWERED = "descriptive_only_calibration_transfer_underpowered"

CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    CLASS_NATIVE_TP: (62, 146, 204),
    CLASS_RESOLVED: (76, 175, 80),
    CLASS_PERSISTENT: (229, 83, 83),
    CLASS_AMBIGUITY_FLIP: (255, 193, 7),
    CLASS_UNRESOLVED: (171, 71, 188),
    CLASS_WITHHELD: (158, 158, 158),
    CLASS_UNDERPOWERED: (120, 144, 156),
}


class VisualContractError(RuntimeError):
    """Raised when an input seal or owner identity is incoherent."""


def _fail(message: str) -> NoReturn:
    raise VisualContractError(message)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    try:
        return sha256_bytes(path.read_bytes())
    except OSError as exc:
        _fail(f"input is unreadable at {path}: {exc}")


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        _fail(f"{label} is missing at {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"{label} is unreadable at {path}: {exc}")
    if not isinstance(value, Mapping):
        _fail(f"{label} at {path} is not a JSON object")
    return dict(value)


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        _fail(f"{label} is missing at {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            _fail(f"{label} line {line_number} is not valid JSON: {exc}")
        if not isinstance(value, Mapping):
            _fail(f"{label} line {line_number} is not a JSON object")
        rows.append(dict(value))
    return rows


def _verify_self_seal(receipt: Mapping[str, Any], *, label: str) -> str:
    observed = receipt.get("receipt_content_sha256")
    expected = sha256_json(
        {key: value for key, value in receipt.items() if key != "receipt_content_sha256"}
    )
    if observed != expected:
        _fail(f"{label} does not reconstruct its receipt_content_sha256")
    return expected


def _verify_digest(path: Path, expected: Any, *, label: str) -> str:
    observed = sha256_file(path)
    if observed != expected:
        _fail(f"{label} digest mismatch")
    return observed


@dataclass(frozen=True)
class LoadedArtifacts:
    s0_root: Path
    plan_dir: Path
    s1_analysis_dir: Path
    s2_analysis_dir: Path | None
    image_path: Path
    image_width: int
    image_height: int
    owners: list[dict[str, Any]]
    s2_status: str
    input_files: dict[str, dict[str, Any]]
    receipt_content_sha256: dict[str, str]


def _load_s0(s0_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, str]]:
    receipt_path = s0_root / "receipt.json"
    owners_path = s0_root / "owner-ledger.jsonl"
    receipt = _read_json(receipt_path, "S0 receipt")
    if receipt.get("schema_version") != s0.RECEIPT_SCHEMA_VERSION:
        _fail("S0 receipt has an unexpected schema_version")
    if receipt.get("unit_id") != UNIT_ID or receipt.get("status") != "admitted":
        _fail("S0 receipt is not the admitted image-2299 unit")
    _verify_self_seal(receipt, label="S0 receipt")
    outputs = receipt.get("outputs")
    if not isinstance(outputs, Mapping):
        _fail("S0 receipt lacks output seals")
    block = outputs.get(owners_path.name)
    if not isinstance(block, Mapping):
        _fail("S0 receipt does not seal owner-ledger.jsonl")
    owner_digest = _verify_digest(
        owners_path, block.get("sha256"), label="S0 owner-ledger.jsonl"
    )
    owners = _read_jsonl(owners_path, "S0 owner ledger")
    if len(owners) != EXPECTED_OWNER_COUNT:
        _fail(f"S0 owner ledger has {len(owners)} rows, expected {EXPECTED_OWNER_COUNT}")
    return owners, receipt, {
        "s0/owner-ledger.jsonl": owner_digest,
        "s0/receipt.json": sha256_file(receipt_path),
    }


def _load_plan(
    plan_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, str]]:
    receipt_path = plan_dir / "receipt.json"
    image_path = plan_dir / "image-registry.jsonl"
    owner_path = plan_dir / "owner-registry.jsonl"
    receipt = _read_json(receipt_path, "S1 plan receipt")
    if (
        receipt.get("schema_version") != legacy_plan.PLAN_SCHEMA_VERSION
        or receipt.get("extension_unit_id") != UNIT_ID
    ):
        _fail("S1 plan receipt is not the finalized image-2299 extension plan")
    receipt_digest = _verify_self_seal(receipt, label="S1 plan receipt")
    seals = receipt.get("output_file_digests")
    if not isinstance(seals, Mapping):
        _fail("S1 plan receipt lacks output_file_digests")
    digests = {
        "plan/image-registry.jsonl": _verify_digest(
            image_path, seals.get(image_path.name), label="plan image registry"
        ),
        "plan/owner-registry.jsonl": _verify_digest(
            owner_path, seals.get(owner_path.name), label="plan owner registry"
        ),
        "plan/receipt.json": sha256_file(receipt_path),
    }
    images = _read_jsonl(image_path, "plan image registry")
    if len(images) != 1 or str(images[0].get("image_id")) != IMAGE_ID:
        _fail("S1 plan image registry is not exactly image 2299")
    owners = _read_jsonl(owner_path, "plan owner registry")
    return owners, images[0], receipt, {**digests, "plan/receipt_content": receipt_digest}


def _load_s1(
    analysis_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, str]]:
    receipt_path = analysis_dir / "receipt.json"
    analysis_path = analysis_dir / "analysis.json"
    summaries_path = analysis_dir / "owner-summaries.jsonl"
    receipt = _read_json(receipt_path, "S1 analysis receipt")
    if receipt.get("schema_version") != s1.RECEIPT_SCHEMA_VERSION or receipt.get("unit_id") != UNIT_ID:
        _fail("S1 receipt is not the finalized image-2299 analysis receipt")
    receipt_digest = _verify_self_seal(receipt, label="S1 analysis receipt")
    seals = receipt.get("output_file_digests")
    if not isinstance(seals, Mapping):
        _fail("S1 receipt lacks output_file_digests")
    digests: dict[str, str] = {}
    for name in s1.OUTPUT_NAMES:
        path = analysis_dir / name
        digests[f"s1/{name}"] = _verify_digest(path, seals.get(name), label=f"S1 {name}")
    digests["s1/receipt.json"] = sha256_file(receipt_path)
    analysis = _read_json(analysis_path, "S1 analysis")
    if (
        analysis.get("schema_version") != s1.ANALYSIS_SCHEMA_VERSION
        or analysis.get("unit_id") != UNIT_ID
        or str(analysis.get("image_id")) != IMAGE_ID
    ):
        _fail("S1 analysis identity or schema drifted")
    summaries = _read_jsonl(summaries_path, "S1 owner summaries")
    return summaries, analysis, receipt, {**digests, "s1/receipt_content": receipt_digest}


def _load_s2(
    analysis_dir: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, str]]:
    receipt_path = analysis_dir / s2.RECEIPT_NAME
    report_path = analysis_dir / s2.REPORT_JSON_NAME
    records_path = analysis_dir / s2.OWNER_RECORDS_NAME
    receipt = _read_json(receipt_path, "S2 receipt")
    if receipt.get("schema_version") != s2.RECEIPT_SCHEMA_VERSION or receipt.get("unit_id") != UNIT_ID:
        _fail("S2 receipt is not the finalized image-2299 reachability receipt")
    receipt_digest = _verify_self_seal(receipt, label="S2 receipt")
    seals = receipt.get("output_file_sha256")
    if not isinstance(seals, Mapping):
        _fail("S2 receipt lacks output_file_sha256")
    digests: dict[str, str] = {}
    for name in (s2.REPORT_JSON_NAME, s2.REPORT_MD_NAME, s2.OWNER_RECORDS_NAME):
        path = analysis_dir / name
        digests[f"s2/{name}"] = _verify_digest(path, seals.get(name), label=f"S2 {name}")
    digests["s2/receipt.json"] = sha256_file(receipt_path)
    report = _read_json(report_path, "S2 report")
    if (
        report.get("schema_version") != s2.REPORT_SCHEMA_VERSION
        or report.get("unit_id") != UNIT_ID
        or str(report.get("image_id")) != IMAGE_ID
    ):
        _fail("S2 report identity or schema drifted")
    records: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(records_path, "S2 owner records"):
        if row.get("schema_version") != s2.OWNER_RECORD_SCHEMA_VERSION:
            _fail("S2 owner record has an unexpected schema_version")
        owner_id = str(row.get("gt_owner_id"))
        if owner_id in records:
            _fail(f"S2 repeats owner {owner_id!r}")
        records[owner_id] = row
    return records, report, receipt, {**digests, "s2/receipt_content": receipt_digest}


def _resolve_image(
    image_row: Mapping[str, Any], plan_receipt: Mapping[str, Any], explicit: Path | None
) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    source_paths = plan_receipt.get("source_paths")
    if isinstance(source_paths, Mapping) and source_paths.get("panel"):
        panel_path = Path(str(source_paths["panel"]))
        if panel_path.is_file():
            panel_rows = _read_jsonl(panel_path, "plan source panel")
            matches = [row for row in panel_rows if str(row.get("image_id")) == IMAGE_ID]
            if len(matches) == 1:
                refs = matches[0].get("images")
                if isinstance(refs, list) and len(refs) == 1:
                    candidates.append(panel_path.parent / str(refs[0]))
    file_name = str(image_row.get("file_name") or "")
    if file_name:
        candidates.append(Path("/data/CoordExp/public_data/coco/rescale_32_1024_bbox") / file_name)
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            continue
        expected = (plan_receipt.get("source_content_digests") or {}).get(
            "image2299_bytes_sha256"
        )
        if expected is not None and sha256_file(resolved) != expected:
            continue
        return resolved
    _fail("image 2299 could not be resolved from --image-path or the sealed plan sources")


def _bbox(row: Mapping[str, Any], field: str) -> list[int]:
    value = row.get(field)
    if not isinstance(value, list) or len(value) != 4:
        _fail(f"owner {row.get('gt_owner_id')!r} lacks four-value {field}")
    box = [int(round(float(item))) for item in value]
    if box[2] <= box[0] or box[3] <= box[1]:
        _fail(f"owner {row.get('gt_owner_id')!r} has degenerate {field}")
    return box


def _descriptive_sensitivity_class(summary: Mapping[str, Any]) -> str:
    descriptive = str(summary.get("frozen_disposition_descriptive") or summary.get("disposition"))
    mapping = {
        s1.DISPOSITION_RESOLVED: CLASS_RESOLVED,
        s1.DISPOSITION_PERSISTENT: CLASS_PERSISTENT,
        s1.DISPOSITION_FLIP: CLASS_AMBIGUITY_FLIP,
        s1.DISPOSITION_UNRESOLVED: CLASS_UNRESOLVED,
    }
    if descriptive not in mapping:
        _fail(
            f"native-FN owner {summary.get('gt_owner_id')!r} has unsupported frozen-rule "
            f"sensitivity disposition {descriptive!r}"
        )
    return mapping[descriptive]


def _primary_visual_class(summary: Mapping[str, Any]) -> str:
    if summary.get("native_true_positive") is True:
        return CLASS_NATIVE_TP
    interpretable = summary.get("fn_disposition_interpretable")
    if not isinstance(interpretable, bool):
        _fail(
            f"native-FN owner {summary.get('gt_owner_id')!r} lacks boolean "
            "fn_disposition_interpretable"
        )
    disposition = str(summary.get("disposition"))
    if not interpretable:
        if disposition == s1.DISPOSITION_WITHHELD:
            return CLASS_WITHHELD
        if summary.get("disposition_role") == "descriptive_only_calibration_transfer_underpowered":
            return CLASS_UNDERPOWERED
        _fail(
            f"non-interpretable native-FN owner {summary.get('gt_owner_id')!r} has "
            f"unsupported disposition {disposition!r}"
        )
    mapping = {
        s1.DISPOSITION_RESOLVED: CLASS_RESOLVED,
        s1.DISPOSITION_PERSISTENT: CLASS_PERSISTENT,
        s1.DISPOSITION_FLIP: CLASS_AMBIGUITY_FLIP,
        s1.DISPOSITION_UNRESOLVED: CLASS_UNRESOLVED,
    }
    if disposition not in mapping:
        _fail(
            f"native-FN owner {summary.get('gt_owner_id')!r} has unsupported S1 disposition "
            f"{disposition!r}"
        )
    return mapping[disposition]


def _secondary_sensitivity(summary: Mapping[str, Any], primary_class: str) -> dict[str, str] | None:
    if primary_class == CLASS_WITHHELD:
        return {
            "role": "descriptive_only_nontransferring",
            "frozen_rule_class": _descriptive_sensitivity_class(summary),
        }
    if primary_class == CLASS_UNDERPOWERED:
        return {
            "role": "descriptive_only_calibration_transfer_underpowered",
            "frozen_rule_class": _descriptive_sensitivity_class(summary),
        }
    return None


def _reachability_view(row: Mapping[str, Any]) -> dict[str, Any]:
    upper = row.get("upper_bound_u")
    crossing = row.get("exact_crossing")
    if not isinstance(upper, Mapping) or not isinstance(crossing, Mapping):
        _fail(f"S2 owner {row.get('gt_owner_id')!r} lacks categorical reachability fields")
    return {
        "support_context_count": int(upper.get("support_context_count", 0)),
        "any_gate_open": bool(upper.get("any_gate_open")),
        "any_category_rank_top3": bool(upper.get("any_category_rank_top3")),
        "any_owner_rank_one": bool(upper.get("any_owner_rank_one")),
        "any_favorable_top3_before_or_at_frontier": bool(
            upper.get("any_favorable_top3_before_or_at_frontier")
        ),
        "exact_crossing_exists": bool(crossing.get("exact_crossing_exists")),
        "exact_crossing_u_favorable_supported": bool(crossing.get("u_favorable_supported")),
        "source_context_ids": dict(row.get("source_context_ids") or {}),
    }


def load_artifacts(
    s0_root: Path,
    plan_dir: Path,
    s1_analysis_dir: Path,
    *,
    s2_analysis_dir: Path | None = None,
    image_path: Path | None = None,
) -> LoadedArtifacts:
    s0_root = Path(s0_root).resolve()
    plan_dir = Path(plan_dir).resolve()
    s1_analysis_dir = Path(s1_analysis_dir).resolve()
    s2_analysis_dir = None if s2_analysis_dir is None else Path(s2_analysis_dir).resolve()
    s0_rows, s0_receipt, s0_files = _load_s0(s0_root)
    plan_rows, image_row, plan_receipt, plan_files = _load_plan(plan_dir)
    summaries, analysis, s1_receipt, s1_files = _load_s1(s1_analysis_dir)
    s2_rows: dict[str, dict[str, Any]] = {}
    s2_files: dict[str, str] = {}
    s2_receipt: dict[str, Any] | None = None
    s2_report: dict[str, Any] | None = None
    if s2_analysis_dir is not None:
        s2_rows, s2_report, s2_receipt, s2_files = _load_s2(s2_analysis_dir)

    transfer = analysis.get("calibration_transfer")
    if not isinstance(transfer, Mapping):
        _fail("S1 analysis lacks calibration_transfer")
    calibration_transfer_status = str(transfer.get("status"))
    if s2_analysis_dir is None:
        s2_status = "not_attached"
    elif s2_rows:
        s2_status = "available"
    elif calibration_transfer_status == "calibration_nontransferring":
        denominator = (s2_report or {}).get("resolved_false_negative_cohort", {}).get(
            "denominator"
        )
        if denominator != 0:
            _fail("nontransferring S2 report must have a zero resolved-FN denominator")
        s2_status = "not_run_calibration_nontransfer"
    else:
        s2_status = "no_resolved_native_false_negative_cohort"

    if not (len(plan_rows) == len(summaries) == EXPECTED_OWNER_COUNT):
        _fail("S0, plan, and S1 must each preserve exactly 46 owners")
    by_s0: dict[str, dict[str, Any]] = {}
    for row in s0_rows:
        if row.get("schema_version") != s0.OWNER_SCHEMA_VERSION:
            _fail("S0 owner row has an unexpected schema_version")
        owner_id = str(row.get("gt_owner_id"))
        if owner_id in by_s0:
            _fail(f"duplicate S0 owner ID {owner_id!r}")
        by_s0[owner_id] = row
    by_plan = {str(row.get("gt_owner_id")): row for row in plan_rows}
    by_s1 = {str(row.get("gt_owner_id")): row for row in summaries}
    if len(by_plan) != EXPECTED_OWNER_COUNT or len(by_s1) != EXPECTED_OWNER_COUNT:
        _fail("plan or S1 owner IDs are not unique")
    if set(by_s0) != set(by_plan) or set(by_s0) != set(by_s1):
        _fail("S0, plan, and S1 owner-ID sets differ")

    indices: set[int] = set()
    owners: list[dict[str, Any]] = []
    for owner_id, s0_row in by_s0.items():
        plan_row = by_plan[owner_id]
        summary = by_s1[owner_id]
        if (
            plan_row.get("schema_version") != legacy_plan.PLAN_SCHEMA_VERSION
            or summary.get("schema_version") != s1.OWNER_SCHEMA_VERSION
        ):
            _fail(f"owner {owner_id!r} has a foreign plan or S1 schema")
        index_value = s0_row.get("original_annotation_index")
        if not isinstance(index_value, int) or isinstance(index_value, bool):
            _fail(f"owner {owner_id!r} lacks its integer original_annotation_index")
        if index_value in indices:
            _fail(f"duplicate original_annotation_index {index_value}")
        indices.add(index_value)
        s0_box = _bbox(s0_row, "bbox_xyxy")
        plan_box = _bbox(plan_row, "bbox_pixel_xyxy")
        summary_box = _bbox(summary, "bbox_pixel_xyxy")
        if s0_box != plan_box or s0_box != summary_box:
            _fail(f"owner {owner_id!r} geometry differs across S0, plan, and S1")
        native_tp = s0_row.get("native_greedy_match_status") == "matched"
        if bool(plan_row.get("native_true_positive")) != native_tp:
            _fail(f"owner {owner_id!r} native TP/FN identity differs between S0 and plan")
        if summary.get("native_true_positive") is not native_tp:
            _fail(f"owner {owner_id!r} native TP/FN identity differs between S0 and S1")
        normalized_description = str(s0_row.get("normalized_description"))
        if normalized_description != str(summary.get("normalized_description")):
            _fail(f"owner {owner_id!r} category differs between S0 and S1")
        primary_class = _primary_visual_class(summary)
        if native_tp:
            owner_s2_status = "not_applicable_native_true_positive"
        elif owner_id in s2_rows:
            owner_s2_status = "available"
        elif s2_status == "available":
            owner_s2_status = "not_applicable_not_resolved_native_false_negative"
        else:
            owner_s2_status = s2_status
        owners.append(
            {
                "owner_index": index_value,
                "gt_owner_id": owner_id,
                "label": f"#{index_value:02d}",
                "normalized_description": normalized_description,
                "bbox_pixel_xyxy": s0_box,
                "native_outcome": "TP" if native_tp else "FN",
                "primary_visual_class": primary_class,
                "s1_disposition": summary.get("disposition"),
                "fn_disposition_interpretable": summary.get(
                    "fn_disposition_interpretable"
                ),
                "s1_frozen_disposition_descriptive": summary.get(
                    "frozen_disposition_descriptive"
                ),
                "secondary_sensitivity": _secondary_sensitivity(
                    summary, primary_class
                ),
                "s1_ambiguity_bound_disposition_flip": bool(
                    summary.get("ambiguity_bound_disposition_flip")
                ),
                "s2_reachability": (
                    _reachability_view(s2_rows[owner_id]) if owner_id in s2_rows else None
                ),
                "s2_status": owner_s2_status,
            }
        )
    owners.sort(key=lambda row: int(row["owner_index"]))
    if indices != set(range(EXPECTED_OWNER_COUNT)):
        _fail("original_annotation_index must remain the complete unique range 0..45")
    category_counts = Counter(str(row["normalized_description"]) for row in owners)
    if dict(sorted(category_counts.items())) != s1.EXPECTED_CATEGORY_COUNTS:
        _fail("owner categories do not preserve the frozen 38 person + 8 tie composition")

    resolved_ids = {
        str(row["gt_owner_id"])
        for row in owners
        if row["s1_disposition"] == s1.DISPOSITION_RESOLVED
    }
    if s2_analysis_dir is not None and set(s2_rows) != resolved_ids:
        _fail("optional S2 owner records do not exactly equal the S1 resolved native-FN cohort")
    denominators = analysis.get("denominators")
    if not isinstance(denominators, Mapping) or denominators.get(
        "image2299_owner_count"
    ) != EXPECTED_OWNER_COUNT:
        _fail("S1 analysis denominator is not exactly 46 owners")

    resolved_image = _resolve_image(image_row, plan_receipt, image_path)
    expected_width = int(image_row.get("image_width", 0))
    expected_height = int(image_row.get("image_height", 0))
    from PIL import Image

    with Image.open(resolved_image) as image:
        if image.size != (expected_width, expected_height):
            _fail("resolved image dimensions differ from the S1 plan")
    image_digest = sha256_file(resolved_image)
    input_files = {
        **{name: {"sha256": digest} for name, digest in s0_files.items()},
        **{
            name: {"sha256": digest}
            for name, digest in plan_files.items()
            if not name.endswith("/receipt_content")
        },
        **{
            name: {"sha256": digest}
            for name, digest in s1_files.items()
            if not name.endswith("/receipt_content")
        },
        **{
            name: {"sha256": digest}
            for name, digest in s2_files.items()
            if not name.endswith("/receipt_content")
        },
        "image": {"path": str(resolved_image), "sha256": image_digest},
    }
    return LoadedArtifacts(
        s0_root=s0_root,
        plan_dir=plan_dir,
        s1_analysis_dir=s1_analysis_dir,
        s2_analysis_dir=s2_analysis_dir,
        image_path=resolved_image,
        image_width=expected_width,
        image_height=expected_height,
        owners=owners,
        s2_status=s2_status,
        input_files=input_files,
        receipt_content_sha256={
            "s0": str(s0_receipt["receipt_content_sha256"]),
            "plan": str(plan_receipt["receipt_content_sha256"]),
            "s1": str(s1_receipt["receipt_content_sha256"]),
            **(
                {"s2": str(s2_receipt["receipt_content_sha256"])}
                if s2_receipt is not None
                else {}
            ),
        },
    )


def build_atlas_spec(artifacts: LoadedArtifacts) -> dict[str, Any]:
    owners = [dict(row) for row in artifacts.owners]
    for owner in owners:
        crop = fixed_padding_crop(
            [owner["bbox_pixel_xyxy"]],
            width=artifacts.image_width,
            height=artifacts.image_height,
            padding=CROP_PADDING_PIXELS,
        )
        window = crop["window_pixel_xyxy"]
        neighbours = [
            other["gt_owner_id"]
            for other in owners
            if _boxes_intersect(window, other["bbox_pixel_xyxy"])
        ]
        owner["crop"] = {**crop, "neighbour_owner_ids": neighbours}
        owner["crop_panel_index"] = int(owner["owner_index"]) // OWNERS_PER_CROP_PANEL + 1
    return {
        "schema_version": VISUAL_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "image_id": IMAGE_ID,
        "image": {
            "path": str(artifacts.image_path),
            "width": artifacts.image_width,
            "height": artifacts.image_height,
            "sha256": artifacts.input_files["image"]["sha256"],
        },
        "selection_contract": {
            "owner_denominator": EXPECTED_OWNER_COUNT,
            "selection": "all_owners_in_original_annotation_index_order",
            "score_based_case_selection": False,
            "score_fields_read": False,
            "rank_or_margin_used_for_selection": False,
        },
        "class_colors_rgb": {name: list(color) for name, color in CLASS_COLORS.items()},
        "primary_class_counts": dict(
            sorted(Counter(row["primary_visual_class"] for row in owners).items())
        ),
        "secondary_sensitivity_counts": dict(
            sorted(
                Counter(
                    row["secondary_sensitivity"]["frozen_rule_class"]
                    for row in owners
                    if row["secondary_sensitivity"] is not None
                ).items()
            )
        ),
        "native_outcome_counts": dict(
            sorted(Counter(row["native_outcome"] for row in owners).items())
        ),
        "s2_attached": artifacts.s2_analysis_dir is not None,
        "s2_status": artifacts.s2_status,
        "owner_count": len(owners),
        "owners": owners,
    }


def _boxes_intersect(left: Sequence[int], right: Sequence[int]) -> bool:
    return not (
        right[2] <= left[0]
        or right[0] >= left[2]
        or right[3] <= left[1]
        or right[1] >= left[3]
    )


def _font(size: int) -> Any:
    from PIL import ImageFont

    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _draw_label(draw: Any, xy: tuple[int, int], text: str, color: tuple[int, int, int], font: Any) -> None:
    left, top, right, bottom = draw.textbbox(xy, text, font=font)
    draw.rectangle((left - 2, top - 1, right + 2, bottom + 1), fill=(12, 12, 12))
    draw.text(xy, text, fill=color, font=font)


def render_owner_map(
    spec: Mapping[str, Any], image: Any, *, category: str, output_path: Path
) -> None:
    from PIL import Image, ImageDraw

    scale = 1.65
    source_width, source_height = image.size
    map_width = int(round(source_width * scale))
    map_height = int(round(source_height * scale))
    legend_width = 690
    canvas = Image.new("RGB", (map_width + legend_width, map_height), (20, 20, 24))
    canvas.paste(image.resize((map_width, map_height), Image.Resampling.LANCZOS), (0, 0))
    draw = ImageDraw.Draw(canvas)
    label_font = _font(17)
    title_font = _font(24)
    owners = [
        row
        for row in spec["owners"]
        if category == "all" or row["normalized_description"] == category
    ]
    legend_step = min(23, max(14, (map_height - 108) // max(1, len(owners))))
    legend_font = _font(16 if legend_step >= 22 else 13)
    for row in owners:
        box = [int(round(float(value) * scale)) for value in row["bbox_pixel_xyxy"]]
        color = CLASS_COLORS[str(row["primary_visual_class"])]
        draw.rectangle(box, outline=color, width=4)
        label_xy = (box[0] + 2, max(1, box[1] + 2))
        _draw_label(draw, label_xy, str(row["label"]), color, label_font)
    draw.text((map_width + 18, 14), f"image 2299 owners: {category}", fill="white", font=title_font)
    draw.text(
        (map_width + 18, 48),
        "all cases; original annotation indexes",
        fill=(210, 210, 215),
        font=legend_font,
    )
    note_font = _font(13)
    draw.text(
        (map_width + 18, 70),
        "Main color = sealed S1 disposition.",
        fill=(205, 205, 210),
        font=note_font,
    )
    draw.text(
        (map_width + 18, 86),
        "Frozen-rule sensitivity is secondary only: see crop panels.",
        fill=(180, 180, 186),
        font=note_font,
    )
    y = 108
    for row in owners:
        color = CLASS_COLORS[str(row["primary_visual_class"])]
        text = (
            f"{row['label']}  {row['gt_owner_id']}  {row['normalized_description']}  "
            f"{row['native_outcome']}  {row['primary_visual_class']}"
        )
        draw.rectangle((map_width + 18, y + 3, map_width + 30, y + 15), fill=color)
        draw.text((map_width + 38, y), text[:76], fill=(235, 235, 238), font=legend_font)
        y += legend_step
    canvas.save(output_path)


def _fit_crop(image: Any, window: Sequence[int], size: tuple[int, int]) -> tuple[Any, float, int, int]:
    from PIL import Image

    crop = image.crop(tuple(window))
    scale = min(size[0] / crop.width, size[1] / crop.height)
    width = max(1, int(round(crop.width * scale)))
    height = max(1, int(round(crop.height * scale)))
    resized = crop.resize((width, height), Image.Resampling.LANCZOS)
    return resized, scale, (size[0] - width) // 2, (size[1] - height) // 2


def _reachability_text(value: Mapping[str, Any] | None, *, status: str) -> str:
    if value is None:
        return f"S2: {status}"
    return (
        f"S2 U gate={int(value['any_gate_open'])} cat<=3={int(value['any_category_rank_top3'])} "
        f"owner#1={int(value['any_owner_rank_one'])} crossing={int(value['exact_crossing_u_favorable_supported'])}"
    )


def render_crop_panels(
    spec: Mapping[str, Any], image: Any, output_dir: Path
) -> list[str]:
    from PIL import Image, ImageDraw

    owners = list(spec["owners"])
    page_count = math.ceil(len(owners) / OWNERS_PER_CROP_PANEL)
    output_names: list[str] = []
    for page_index in range(page_count):
        page = Image.new("RGB", (1500, 1260), (19, 19, 23))
        draw = ImageDraw.Draw(page)
        title_font = _font(18)
        detail_font = _font(14)
        start = page_index * OWNERS_PER_CROP_PANEL
        for slot, row in enumerate(owners[start : start + OWNERS_PER_CROP_PANEL]):
            column = slot % 3
            line = slot // 3
            tile_x = column * 500
            tile_y = line * 420
            window = row["crop"]["window_pixel_xyxy"]
            fitted, scale, offset_x, offset_y = _fit_crop(image, window, (470, 285))
            paste_x = tile_x + 15 + offset_x
            paste_y = tile_y + 92 + offset_y
            page.paste(fitted, (paste_x, paste_y))
            target_id = str(row["gt_owner_id"])
            neighbours = set(row["crop"]["neighbour_owner_ids"])
            for other in owners:
                if other["gt_owner_id"] not in neighbours:
                    continue
                box = other["bbox_pixel_xyxy"]
                transformed = [
                    int(round(paste_x + (box[0] - window[0]) * scale)),
                    int(round(paste_y + (box[1] - window[1]) * scale)),
                    int(round(paste_x + (box[2] - window[0]) * scale)),
                    int(round(paste_y + (box[3] - window[1]) * scale)),
                ]
                transformed = [
                    max(paste_x, transformed[0]),
                    max(paste_y, transformed[1]),
                    min(paste_x + fitted.width - 1, transformed[2]),
                    min(paste_y + fitted.height - 1, transformed[3]),
                ]
                if transformed[2] <= transformed[0] or transformed[3] <= transformed[1]:
                    continue
                is_target = other["gt_owner_id"] == target_id
                color = (
                    CLASS_COLORS[str(other["primary_visual_class"])]
                    if is_target
                    else (180, 180, 185)
                )
                draw.rectangle(transformed, outline=color, width=4 if is_target else 1)
                if is_target:
                    _draw_label(
                        draw,
                        (transformed[0] + 2, transformed[1] + 2),
                        str(other["label"]),
                        color,
                        detail_font,
                    )
            color = CLASS_COLORS[str(row["primary_visual_class"])]
            draw.text(
                (tile_x + 15, tile_y + 10),
                f"{row['label']} {row['gt_owner_id']} | {row['normalized_description']} | {row['native_outcome']}",
                fill=color,
                font=title_font,
            )
            draw.text(
                (tile_x + 15, tile_y + 36),
                str(row["primary_visual_class"]),
                fill=(235, 235, 238),
                font=detail_font,
            )
            secondary = row.get("secondary_sensitivity")
            if secondary is not None:
                draw.text(
                    (tile_x + 15, tile_y + 53),
                    f"secondary: {secondary['role']}",
                    fill=(185, 185, 190),
                    font=detail_font,
                )
                draw.text(
                    (tile_x + 15, tile_y + 70),
                    f"frozen-rule sensitivity: {secondary['frozen_rule_class']}",
                    fill=(175, 175, 182),
                    font=detail_font,
                )
            draw.text(
                (tile_x + 15, tile_y + 392),
                _reachability_text(
                    row["s2_reachability"], status=str(row["s2_status"])
                ),
                fill=(205, 205, 212),
                font=detail_font,
            )
        name = f"owner-crops-page-{page_index + 1:02d}.png"
        page.save(output_dir / name)
        output_names.append(name)
    return output_names


def _output_descriptor(path: Path) -> dict[str, Any]:
    return {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def build_manifest(
    artifacts: LoadedArtifacts,
    spec: Mapping[str, Any],
    output_dir: Path,
    output_names: Sequence[str],
) -> dict[str, Any]:
    by_owner = {
        str(row["gt_owner_id"]): {
            "owner_index": row["owner_index"],
            "native_outcome": row["native_outcome"],
            "primary_visual_class": row["primary_visual_class"],
            "secondary_sensitivity": row["secondary_sensitivity"],
            "s2_status": row["s2_status"],
            "crop_panel": f"owner-crops-page-{int(row['crop_panel_index']):02d}.png",
        }
        for row in spec["owners"]
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "image_id": IMAGE_ID,
        "authority": "sealed_s0_s1_and_optional_s2_artifacts",
        "gpu_used": False,
        "reads_model": False,
        "reads_scores": False,
        "score_based_case_selection": False,
        "selection": "all_46_owners_in_original_annotation_index_order",
        "owner_ids_and_indexes_preserved": True,
        "owner_count": len(by_owner),
        "owner_index_range": [0, EXPECTED_OWNER_COUNT - 1],
        "s2_attached": artifacts.s2_analysis_dir is not None,
        "s2_status": artifacts.s2_status,
        "primary_class_counts": dict(spec["primary_class_counts"]),
        "secondary_sensitivity_counts": dict(spec["secondary_sensitivity_counts"]),
        "native_outcome_counts": dict(spec["native_outcome_counts"]),
        "receipt_content_sha256": dict(artifacts.receipt_content_sha256),
        "input_files": dict(sorted(artifacts.input_files.items())),
        "output_files": {
            name: _output_descriptor(output_dir / name) for name in sorted(output_names)
        },
        "owners": by_owner,
        "presentation": {
            "whole_image_maps": list(OWNER_MAP_NAMES.values()),
            "crop_padding_pixels": CROP_PADDING_PIXELS,
            "crop_policy": "fixed_padding_presentation_only_model_saw_full_image",
            "owners_per_crop_panel": OWNERS_PER_CROP_PANEL,
            "neighbour_boxes_rendered": True,
            "full_owner_id_on_every_crop": True,
        },
        "visualizer_source_sha256": sha256_file(Path(__file__).resolve()),
        "self_seal_contract": {
            "field": "manifest_content_sha256",
            "excluded_top_level_fields": ["manifest_content_sha256"],
            "canonicalization": "UTF-8 JSON, ensure_ascii=true, sorted keys, compact separators",
        },
    }
    manifest["manifest_content_sha256"] = sha256_json(manifest)
    return manifest


def materialize(artifacts: LoadedArtifacts, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists() or output_dir.is_symlink():
        _fail(f"refusing to overwrite existing output path {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))
    published = False
    try:
        spec = build_atlas_spec(artifacts)
        spec_path = staging / SPEC_NAME
        spec_path.write_bytes(canonical_json_bytes(spec) + b"\n")
        from PIL import Image

        with Image.open(artifacts.image_path) as source:
            image = source.convert("RGB")
        for category, name in OWNER_MAP_NAMES.items():
            render_owner_map(spec, image, category=category, output_path=staging / name)
        crop_names = render_crop_panels(spec, image, staging)
        output_names = [SPEC_NAME, *OWNER_MAP_NAMES.values(), *crop_names]
        manifest = build_manifest(artifacts, spec, staging, output_names)
        (staging / MANIFEST_NAME).write_bytes(canonical_json_bytes(manifest) + b"\n")
        os.replace(staging, output_dir)
        published = True
        return manifest
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s0-root", type=Path, required=True)
    parser.add_argument("--plan-dir", type=Path, required=True)
    parser.add_argument("--s1-analysis-dir", type=Path, required=True)
    parser.add_argument("--s2-analysis-dir", type=Path)
    parser.add_argument("--image-path", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        artifacts = load_artifacts(
            args.s0_root,
            args.plan_dir,
            args.s1_analysis_dir,
            s2_analysis_dir=args.s2_analysis_dir,
            image_path=args.image_path,
        )
        manifest = materialize(artifacts, args.output_dir)
    except VisualContractError as exc:
        print(f"visual contract error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "manifest": str(args.output_dir / MANIFEST_NAME),
                "owner_count": manifest["owner_count"],
                "s2_attached": manifest["s2_attached"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
