#!/usr/bin/env python3
"""Build the strict owner ledger for the July 22 preservation screen.

This analyzer is intentionally limited to the three matched clean greedy
artifacts used by the Source-route preservation screen.  It imports the
existing canonical coordinate conversion and cardinality-first matching logic;
it does not define another detection matcher or attach semantic labels to
unmatched predictions.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

if __package__ in {None, ""}:  # Allow ``python scripts/research/...py``.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.research.compare_clean_rollout_owner_coverage import (  # noqa: E402
    _distribution,
    _global_matches,
    _gt_objects,
    _gt_signature,
    _pred_objects,
    _read_jsonl,
    iou_xyxy,
)


SCHEMA_VERSION = "source_route_preservation_screen_comparator.v2"
ARM_LABELS = ("source", "single_route", "multi_route")
ACCEPTED_PARSE_STATUSES = frozenset({"accepted", "accepted_with_drops"})
IMAGE_ID_RE = re.compile(r"(?:^|/)(\d{12})\.(?:jpg|jpeg|png)$", re.IGNORECASE)
RUN_NAMES = {
    "source": "qwen3-vl-2b-source-route-preservation-screen-source-train-256-hf",
    "single_route": "qwen3-vl-2b-source-route-preservation-screen-single-route-step31-train-256-hf",
    "multi_route": "qwen3-vl-2b-source-route-preservation-screen-multi-route-step31-train-256-hf",
}
CONFIG_BASENAMES = {
    "source": "qwen3_vl_2b_source_route_preservation_screen_source_train_256_hf.yaml",
    "single_route": "qwen3_vl_2b_source_route_preservation_screen_single_route_step31_train_256_hf.yaml",
    "multi_route": "qwen3_vl_2b_source_route_preservation_screen_multi_route_step31_train_256_hf.yaml",
}
STATE_BANK_ARM_KEYS = {
    "single_route": "single_route_plus_source_preservation",
    "multi_route": "multi_route_plus_source_preservation",
}
STATE_BANK_FAMILIES = {
    "single_route": "single_route_plus_source_preservation",
    "multi_route": "multi_route_plus_source_preservation",
}


class ScreenAnalysisError(ValueError):
    """Raised when the named artifacts cannot prove a paired screen result."""


def _resolved(path: str | Path) -> Path:
    try:
        return Path(path).expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise ScreenAnalysisError(f"missing input artifact: {path}") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ScreenAnalysisError(f"invalid JSON: {path}") from exc
    if not isinstance(value, Mapping):
        raise ScreenAnalysisError(f"{path} must contain a JSON object")
    return value


def _read_diagnostics(path: Path) -> dict[str, Mapping[str, Any]]:
    rows: dict[str, Mapping[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ScreenAnalysisError(
                    f"{path}:{line_number} is not valid JSON"
                ) from exc
            if not isinstance(value, Mapping):
                raise ScreenAnalysisError(f"{path}:{line_number} must contain an object")
            row_id = value.get("row_id", value.get("example_id"))
            if row_id is None:
                raise ScreenAnalysisError(f"{path}:{line_number} is missing row_id")
            key = str(row_id)
            if key in rows:
                raise ScreenAnalysisError(f"{path} contains duplicate row_id {key!r}")
            rows[key] = value
    return rows


def _read_jsonl_values(path: Path) -> list[Mapping[str, Any]]:
    values: list[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ScreenAnalysisError(
                    f"{path}:{line_number} is not valid JSON"
                ) from exc
            if not isinstance(value, Mapping):
                raise ScreenAnalysisError(f"{path}:{line_number} must contain an object")
            values.append(value)
    return values


def _image_id(row: Mapping[str, Any]) -> str:
    raw = row.get("image_id")
    if raw is not None and not isinstance(raw, bool):
        try:
            return str(int(raw))
        except (TypeError, ValueError) as exc:
            raise ScreenAnalysisError(f"row has invalid image_id {raw!r}") from exc
    for field in ("image_path", "row_id", "example_id"):
        match = IMAGE_ID_RE.search(str(row.get(field, "")))
        if match:
            return str(int(match.group(1)))
    row_id = row.get("row_id", row.get("example_id", ""))
    raise ScreenAnalysisError(f"row {row_id!r} has no stable image identity")


def _input_image_id(row: Mapping[str, Any]) -> str:
    """Read the physical image identity from the frozen inference input row."""
    if "image_id" in row:
        return _image_id(row)
    for field in ("file_name", "image_path", "path"):
        match = IMAGE_ID_RE.search(str(row.get(field, "")))
        if match:
            return str(int(match.group(1)))
    images = row.get("images")
    if isinstance(images, list) and len(images) == 1:
        match = IMAGE_ID_RE.search(str(images[0]))
        if match:
            return str(int(match.group(1)))
    raise ScreenAnalysisError("frozen evaluation input row has no stable image identity")


def _owner_id(image_id: str, object_value: Any, index: int) -> str:
    if not isinstance(object_value, Mapping):
        raise ScreenAnalysisError(f"image {image_id} GT object {index} must be an object")
    value = object_value.get("object_id", object_value.get("id"))
    if value is None or isinstance(value, bool) or not str(value):
        raise ScreenAnalysisError(f"image {image_id} GT object {index} has no stable object_id")
    return f"{image_id}:{value}"


def _owner_ids(row: Mapping[str, Any]) -> list[str]:
    objects = row.get("gt")
    if not isinstance(objects, list):
        raise ScreenAnalysisError(f"row {row.get('row_id')!r} has malformed GT list")
    image_id = _image_id(row)
    owner_ids = [_owner_id(image_id, object_value, index) for index, object_value in enumerate(objects)]
    if len(owner_ids) != len(set(owner_ids)):
        raise ScreenAnalysisError(f"image {image_id} has duplicate stable GT owner IDs")
    return owner_ids


def _nonnegative_int(value: Any, *, field: str, row_id: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ScreenAnalysisError(f"row {row_id!r} has invalid {field}")
    if value < 0:
        raise ScreenAnalysisError(f"row {row_id!r} has negative {field}")
    return value


def _validate_sidecars(
    *,
    label: str,
    rows: Mapping[str, Mapping[str, Any]],
    summary: Mapping[str, Any],
    diagnostics: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    row_ids = set(rows)
    if set(diagnostics) != row_ids:
        raise ScreenAnalysisError(f"{label} parse diagnostics row set does not match gt_vs_pred")
    if summary.get("terminal_status") != "completed":
        raise ScreenAnalysisError(f"{label} summary is not a completed inference artifact")
    raw_prediction_count = 0
    parser_failure_count = 0
    dropped_prediction_count = 0
    stop_reasons: Counter[str] = Counter()
    malformed_row_count = 0
    invalid_row_count = 0
    for row_id, row in rows.items():
        diagnostic = diagnostics[row_id]
        status = str(row.get("parse_status", ""))
        if diagnostic.get("parse_status") != status:
            raise ScreenAnalysisError(f"{label} parse status differs for row_id {row_id!r}")
        prediction_count = len(row.get("pred", [])) if isinstance(row.get("pred"), list) else 0
        if diagnostic.get("valid_prediction_count") != prediction_count:
            raise ScreenAnalysisError(f"{label} diagnostic prediction count differs for row_id {row_id!r}")
        dropped = _nonnegative_int(
            row.get("dropped_prediction_count", 0),
            field="dropped_prediction_count",
            row_id=row_id,
        )
        raw_dropped = row.get("dropped_predictions")
        if not isinstance(raw_dropped, list) or len(raw_dropped) != dropped:
            raise ScreenAnalysisError(
                f"{label} gt_vs_pred dropped prediction evidence differs for row_id {row_id!r}"
            )
        if diagnostic.get("dropped_prediction_count") != dropped:
            raise ScreenAnalysisError(f"{label} diagnostic dropped count differs for row_id {row_id!r}")
        diagnostic_dropped = diagnostic.get("dropped_predictions")
        if not isinstance(diagnostic_dropped, list) or len(diagnostic_dropped) != dropped:
            raise ScreenAnalysisError(
                f"{label} diagnostic dropped prediction evidence differs for row_id {row_id!r}"
            )
        if diagnostic_dropped != raw_dropped:
            raise ScreenAnalysisError(
                f"{label} dropped prediction evidence differs for row_id {row_id!r}"
            )
        raw_prediction_count += prediction_count
        dropped_prediction_count += dropped
        parser_failure_count += int(status not in ACCEPTED_PARSE_STATUSES)
        malformed_row_count += int("malformed" in status.lower())
        invalid_row_count += int("invalid" in status.lower())
        stop_reasons[str(row.get("decode_stop_reason", ""))] += 1
    expected_summary = {
        "row_count": len(rows),
        "raw_row_count": len(rows),
        "decode_success_count": len(rows),
        "parser_failure_count": parser_failure_count,
        "dropped_prediction_count": dropped_prediction_count,
        "truncated_decode_count": stop_reasons["length"],
        "decode_stop_reasons": dict(sorted(stop_reasons.items())),
    }
    for field, expected in expected_summary.items():
        if summary.get(field) != expected:
            raise ScreenAnalysisError(
                f"{label} summary {field} does not match gt_vs_pred/parse diagnostics"
            )
    return {
        "prediction_count": raw_prediction_count,
        "parser_failure_count": parser_failure_count,
        "malformed_row_count": malformed_row_count,
        "invalid_row_count": invalid_row_count,
        "dropped_prediction_count": dropped_prediction_count,
        "natural_closure_count": stop_reasons["im_end"],
        "truncated_decode_count": stop_reasons["length"],
        "decode_stop_reasons": dict(sorted(stop_reasons.items())),
    }


def _row_matches(
    row: Mapping[str, Any], *, match_iou_threshold: float, duplicate_iou_threshold: float
) -> tuple[set[str], dict[str, tuple[float, float, float, tuple[float, float, float, float]]], int, int]:
    row_id = str(row.get("row_id", row.get("example_id", "")))
    gt = _gt_objects(dict(row), row_id=row_id)
    pred, invalid_prediction_count = _pred_objects(dict(row))
    owner_ids = _owner_ids(row)
    matched: dict[str, tuple[float, float, float, tuple[float, float, float, float]]] = {}
    for gt_index, pred_index, overlap in _global_matches(gt, pred, match_iou_threshold):
        gx1, gy1, gx2, gy2 = gt[gt_index][1]
        px1, py1, px2, py2 = pred[pred_index][1]
        center_error = math.hypot(
            (gx1 + gx2 - px1 - px2) / 2,
            (gy1 + gy2 - py1 - py2) / 2,
        )
        size_error = math.hypot(
            (gx2 - gx1) - (px2 - px1),
            (gy2 - gy1) - (py2 - py1),
        )
        matched[owner_ids[gt_index]] = (
            overlap,
            center_error,
            size_error,
            tuple(abs(left - right) for left, right in zip(gt[gt_index][1], pred[pred_index][1])),
        )
    owner_hits: Counter[int] = Counter()
    ambiguous_duplicate_candidate_count = 0
    for category, pred_box in pred:
        candidates = [
            gt_index
            for gt_index, (gt_category, gt_box) in enumerate(gt)
            if gt_category == category and iou_xyxy(gt_box, pred_box) >= duplicate_iou_threshold
        ]
        if len(candidates) == 1:
            owner_hits[candidates[0]] += 1
        elif len(candidates) > 1:
            ambiguous_duplicate_candidate_count += 1
    return (
        set(matched),
        matched,
        sum(max(0, count - 1) for count in owner_hits.values()),
        ambiguous_duplicate_candidate_count,
    )


def _scope_summary(
    *,
    image_ids: set[str],
    gt_by_image: Mapping[str, set[str]],
    matched_by_image: Mapping[str, set[str]],
) -> dict[str, Any]:
    gt_owner_ids = sorted(
        (owner_id for image_id in image_ids for owner_id in gt_by_image[image_id]),
    )
    matched_owner_ids = sorted(
        (owner_id for image_id in image_ids for owner_id in matched_by_image[image_id]),
    )
    matched_set = set(matched_owner_ids)
    gt_set = set(gt_owner_ids)
    return {
        "image_count": len(image_ids),
        "image_ids": sorted(image_ids, key=int),
        "gt_owner_count": len(gt_owner_ids),
        "gt_owner_ids": gt_owner_ids,
        "matched_owner_count": len(matched_owner_ids),
        "matched_owner_ids": matched_owner_ids,
        "owner_coverage": len(matched_owner_ids) / len(gt_owner_ids) if gt_owner_ids else 0.0,
        "owner_false_negative_count": len(gt_set - matched_set),
        "owner_false_negative_ids": sorted(gt_set - matched_set),
    }


def _arm_receipt(
    *,
    rows: Mapping[str, Mapping[str, Any]],
    sidecar_counts: Mapping[str, Any],
    admitted_image_ids: set[str],
    match_iou_threshold: float,
    duplicate_iou_threshold: float,
) -> tuple[dict[str, Any], dict[str, set[str]], dict[str, dict[str, Any]]]:
    gt_by_image: dict[str, set[str]] = {}
    matched_by_image: dict[str, set[str]] = {}
    geometry_by_image: dict[str, dict[str, Any]] = {}
    duplicate_candidate_count = 0
    ambiguous_duplicate_candidate_count = 0
    invalid_prediction_count = 0
    for row_id in sorted(rows):
        row = rows[row_id]
        image_id = _image_id(row)
        if image_id in gt_by_image:
            raise ScreenAnalysisError(f"arm contains duplicate image identity {image_id}")
        gt_by_image[image_id] = set(_owner_ids(row))
        matched, geometry, duplicate_count, ambiguous_count = _row_matches(
            row,
            match_iou_threshold=match_iou_threshold,
            duplicate_iou_threshold=duplicate_iou_threshold,
        )
        matched_by_image[image_id] = matched
        geometry_by_image[image_id] = {owner_id: value for owner_id, value in geometry.items()}
        duplicate_candidate_count += duplicate_count
        ambiguous_duplicate_candidate_count += ambiguous_count
        _, invalid_predictions = _pred_objects(dict(row))
        invalid_prediction_count += invalid_predictions
    image_ids = set(gt_by_image)
    if not admitted_image_ids <= image_ids:
        missing = sorted(admitted_image_ids - image_ids, key=int)
        raise ScreenAnalysisError(f"arm is missing admitted image IDs: {missing}")
    scopes = {
        "overall": _scope_summary(
            image_ids=image_ids, gt_by_image=gt_by_image, matched_by_image=matched_by_image
        ),
        "admitted": _scope_summary(
            image_ids=admitted_image_ids,
            gt_by_image=gt_by_image,
            matched_by_image=matched_by_image,
        ),
        "non_admitted": _scope_summary(
            image_ids=image_ids - admitted_image_ids,
            gt_by_image=gt_by_image,
            matched_by_image=matched_by_image,
        ),
    }
    return (
        {
            "scopes": scopes,
            "prediction_count": sidecar_counts["prediction_count"],
            "duplicate_candidates": {
                "count": duplicate_candidate_count,
                "ambiguous_count": ambiguous_duplicate_candidate_count,
                "interpretation": "geometry-derived candidates only; not human-confirmed duplicates",
            },
            "parser_and_drop": {
                "parser_failure_count": sidecar_counts["parser_failure_count"],
                "malformed_row_count": sidecar_counts["malformed_row_count"],
                "invalid_row_count": sidecar_counts["invalid_row_count"],
                "invalid_prediction_count": invalid_prediction_count,
                "dropped_prediction_count": sidecar_counts["dropped_prediction_count"],
            },
            "closure": {
                "natural_closure_count": sidecar_counts["natural_closure_count"],
                "truncated_decode_count": sidecar_counts["truncated_decode_count"],
                "decode_stop_reasons": sidecar_counts["decode_stop_reasons"],
            },
        },
        matched_by_image,
        geometry_by_image,
    )


def _treatment_ledger(
    *,
    source_matches: Mapping[str, set[str]],
    treatment_matches: Mapping[str, set[str]],
    admitted_image_ids: set[str],
    image_ids: set[str],
    selected_positive_owner_ids: set[str],
) -> dict[str, Any]:
    ledger: dict[str, Any] = {}
    gained_all: set[str] = set()
    lost_all: set[str] = set()
    retained_all: set[str] = set()
    for image_id in sorted(image_ids, key=int):
        source = source_matches[image_id]
        treatment = treatment_matches[image_id]
        gained = treatment - source
        lost = source - treatment
        retained = source & treatment
        gained_all |= gained
        lost_all |= lost
        retained_all |= retained
        ledger[image_id] = {
            "admission_scope": "admitted" if image_id in admitted_image_ids else "non_admitted",
            "source_matched_owner_ids": sorted(source),
            "treatment_matched_owner_ids": sorted(treatment),
            "gained_owner_ids": sorted(gained),
            "lost_owner_ids": sorted(lost),
            "retained_owner_ids": sorted(retained),
        }
    targeted_gained = gained_all & selected_positive_owner_ids
    untargeted_gained = gained_all - selected_positive_owner_ids
    return {
        "gained_owner_count": len(gained_all),
        "gained_owner_ids": sorted(gained_all),
        "lost_owner_count": len(lost_all),
        "lost_owner_ids": sorted(lost_all),
        "retained_owner_count": len(retained_all),
        "retained_owner_ids": sorted(retained_all),
        "selected_positive_path_owner_count": len(selected_positive_owner_ids),
        "gained_selected_positive_path_owner_count": len(targeted_gained),
        "gained_selected_positive_path_owner_ids": sorted(targeted_gained),
        "gained_untargeted_owner_count": len(untargeted_gained),
        "gained_untargeted_owner_ids": sorted(untargeted_gained),
        "per_image": ledger,
    }


def _geometry_delta(
    *,
    source_geometry: Mapping[str, Mapping[str, Any]],
    treatment_geometry: Mapping[str, Mapping[str, Any]],
    image_ids: set[str],
) -> dict[str, Any]:
    values = {"iou": [], "center_error_px": [], "size_error_px": [], "x1_abs_error_px": [], "y1_abs_error_px": [], "x2_abs_error_px": [], "y2_abs_error_px": []}
    owner_count = 0
    for image_id in sorted(image_ids, key=int):
        common = set(source_geometry[image_id]) & set(treatment_geometry[image_id])
        for owner_id in common:
            source = source_geometry[image_id][owner_id]
            treatment = treatment_geometry[image_id][owner_id]
            owner_count += 1
            values["iou"].append(treatment[0] - source[0])
            values["center_error_px"].append(treatment[1] - source[1])
            values["size_error_px"].append(treatment[2] - source[2])
            for name, treatment_error, source_error in zip(
                ("x1_abs_error_px", "y1_abs_error_px", "x2_abs_error_px", "y2_abs_error_px"),
                treatment[3],
                source[3],
            ):
                values[name].append(treatment_error - source_error)
    return {
        "common_owner_count": owner_count,
        "delta_convention": "treatment_minus_source",
        **{name: _distribution(items) for name, items in values.items()},
    }


def _admitted_image_ids(selection_receipt: Mapping[str, Any]) -> set[str]:
    raw_ids = selection_receipt.get("image_ids")
    if not isinstance(raw_ids, list):
        raise ScreenAnalysisError("selection receipt lacks image_ids")
    try:
        image_ids = {str(int(value)) for value in raw_ids}
    except (TypeError, ValueError) as exc:
        raise ScreenAnalysisError("selection receipt has invalid image_ids") from exc
    if len(image_ids) != len(raw_ids):
        raise ScreenAnalysisError("selection receipt has duplicate image_ids")
    if selection_receipt.get("image_count") != 118 or len(image_ids) != 118:
        raise ScreenAnalysisError("selection receipt must define exactly 118 admitted image_ids")
    return image_ids


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ScreenAnalysisError(f"{field} must be an object")
    return value


def _path_value(value: Any, *, field: str) -> str:
    if not isinstance(value, str | Path) or not str(value):
        raise ScreenAnalysisError(f"{field} must be a non-empty path")
    return str(Path(value).expanduser().resolve())


def _run_artifact_paths(run_dir: str | Path) -> dict[str, Path]:
    root = _resolved(run_dir)
    if not root.is_dir():
        raise ScreenAnalysisError(f"run directory is not a directory: {root}")
    return {
        "run_dir": root,
        "gt_vs_pred": _resolved(root / "gt_vs_pred.jsonl"),
        "summary": _resolved(root / "summary.json"),
        "parse_diagnostics": _resolved(root / "parse_diagnostics.jsonl"),
        "provenance": _resolved(root / "gt_vs_pred_scored.jsonl.provenance.json"),
        "scored": _resolved(root / "gt_vs_pred_scored.jsonl"),
        "run_manifest": _resolved(root / "run_manifest.json"),
        "resolved_config": _resolved(root / "configs" / "resolved.json"),
    }


def _generation_identity(value: Mapping[str, Any], *, field: str) -> dict[str, Any]:
    required = ("do_sample", "max_new_tokens", "temperature", "top_p", "repetition_penalty")
    if any(key not in value for key in required):
        raise ScreenAnalysisError(f"{field} lacks the exact greedy generation policy")
    result = {key: value[key] for key in required}
    if result["do_sample"] is not False:
        raise ScreenAnalysisError(f"{field} is not greedy")
    return result


def _manifest_model_paths(manifest: Mapping[str, Any], *, label: str) -> tuple[str, str]:
    identity = _mapping(manifest.get("model_identity"), field=f"{label} run_manifest.model_identity")
    nested = identity.get("model_identity")
    if isinstance(nested, Mapping):
        identity = nested
    adapter = _mapping(identity.get("adapter"), field=f"{label} model adapter")
    embedding = _mapping(identity.get("embedding_delta"), field=f"{label} model embedding delta")
    embedding_identity = _mapping(
        embedding.get("identity"), field=f"{label} model embedding delta identity"
    )
    return (
        _path_value(adapter.get("adapter_path"), field=f"{label} model adapter path"),
        _path_value(embedding_identity.get("delta_path"), field=f"{label} model embedding delta path"),
    )


def _verify_canonical_provenance(
    *, label: str, paths: Mapping[str, Path], manifest: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Require the scored-artifact provenance binding before reading raw rows."""
    provenance = _read_json(paths["provenance"])
    raw_artifact = _mapping(provenance.get("raw_artifact"), field=f"{label} provenance raw artifact")
    if raw_artifact.get("path") != "gt_vs_pred.jsonl":
        raise ScreenAnalysisError(f"{label} provenance does not name canonical gt_vs_pred.jsonl")
    if raw_artifact.get("sha256") != _sha256_file(paths["gt_vs_pred"]):
        raise ScreenAnalysisError(f"{label} gt_vs_pred.jsonl does not match canonical provenance")
    scored_artifact = _mapping(
        provenance.get("scored_artifact"), field=f"{label} provenance scored artifact"
    )
    if scored_artifact.get("path") != "gt_vs_pred_scored.jsonl":
        raise ScreenAnalysisError(f"{label} provenance does not name canonical scored artifact")
    if scored_artifact.get("sha256") != _sha256_file(paths["scored"]):
        raise ScreenAnalysisError(f"{label} scored artifact does not match canonical provenance")
    raw_rows = _read_jsonl(paths["gt_vs_pred"])
    row_ids = [str(row_id) for row_id in raw_rows]
    row_binding = _mapping(provenance.get("row_binding"), field=f"{label} provenance row binding")
    expected_row_hash = hashlib.sha256(
        json.dumps(row_ids, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if row_binding.get("row_count") != len(row_ids) or row_binding.get("row_ids_sha256") != expected_row_hash:
        raise ScreenAnalysisError(f"{label} provenance row binding does not match gt_vs_pred")
    for field in (
        "model_identity",
        "model_identity_fingerprint",
        "adapter_identity",
        "embedding_delta_identity",
        "generation_policy",
    ):
        if field not in provenance or field not in manifest or provenance[field] != manifest[field]:
            raise ScreenAnalysisError(f"{label} provenance {field} does not match run manifest")
    if not isinstance(provenance["model_identity"], Mapping):
        raise ScreenAnalysisError(f"{label} provenance lacks content-bearing model identity")
    if not isinstance(provenance["model_identity_fingerprint"], str) or not provenance[
        "model_identity_fingerprint"
    ]:
        raise ScreenAnalysisError(f"{label} provenance lacks model identity fingerprint")
    return provenance


def _verify_run_identity(
    *,
    label: str,
    paths: Mapping[str, Path],
    frozen_eval_input: Path,
    expected_model_paths: Mapping[str, tuple[str | Path, str | Path]],
) -> dict[str, Any]:
    manifest = _read_json(paths["run_manifest"])
    provenance = _verify_canonical_provenance(label=label, paths=paths, manifest=manifest)
    resolved = _read_json(paths["resolved_config"])
    config = _mapping(resolved.get("config"), field=f"{label} resolved config")
    resolution = _mapping(resolved.get("resolution"), field=f"{label} resolution")
    run = _mapping(config.get("run"), field=f"{label} resolved config run")
    if run.get("name") != RUN_NAMES[label]:
        raise ScreenAnalysisError(f"{label} run name does not identify the required screen arm")
    entry_config = _path_value(
        resolution.get("entry_config_path"), field=f"{label} resolved config entry path"
    )
    if Path(entry_config).name != CONFIG_BASENAMES[label]:
        raise ScreenAnalysisError(f"{label} resolved config is not the required screen config")
    data = _mapping(config.get("data"), field=f"{label} resolved config data")
    configured_input = _path_value(data.get("input_jsonl"), field=f"{label} input JSONL")
    origins = _mapping(resolution.get("path_origins"), field=f"{label} config path origins")
    input_origin = _mapping(origins.get("data.input_jsonl"), field=f"{label} input JSONL origin")
    origin_input = _path_value(
        input_origin.get("resolved_path"), field=f"{label} input JSONL origin path"
    )
    if configured_input != str(frozen_eval_input) or origin_input != str(frozen_eval_input):
        raise ScreenAnalysisError(f"{label} is not bound to the frozen 256-image evaluation input")
    generation = _mapping(config.get("generation"), field=f"{label} resolved config generation")
    config_generation = {
        "do_sample": False,
        "max_new_tokens": generation.get("max_new_tokens"),
        "temperature": generation.get("temperature"),
        "top_p": generation.get("top_p"),
        "repetition_penalty": generation.get("repetition_penalty"),
    }
    manifest_generation = _generation_identity(
        _mapping(manifest.get("generation_policy"), field=f"{label} run manifest generation policy"),
        field=f"{label} run manifest generation policy",
    )
    if config_generation != manifest_generation:
        raise ScreenAnalysisError(f"{label} run manifest generation policy differs from resolved config")
    config_adapter = _path_value(
        _mapping(config.get("adapter"), field=f"{label} resolved config adapter").get("path"),
        field=f"{label} resolved config adapter path",
    )
    config_embedding = _path_value(
        _mapping(config.get("embedding_delta"), field=f"{label} resolved config embedding delta").get("path"),
        field=f"{label} resolved config embedding delta path",
    )
    expected_adapter, expected_embedding = expected_model_paths[label]
    expected = (
        _path_value(expected_adapter, field=f"expected {label} adapter path"),
        _path_value(expected_embedding, field=f"expected {label} embedding delta path"),
    )
    observed = _manifest_model_paths(manifest, label=label)
    if (config_adapter, config_embedding) != expected or observed != expected:
        raise ScreenAnalysisError(f"{label} adapter or embedding delta path does not match the expected arm")
    dataset_identity = _mapping(manifest.get("dataset_identity"), field=f"{label} dataset identity")
    return {
        "manifest": manifest,
        "provenance": provenance,
        "generation_policy": manifest_generation,
        "dataset_identity": dict(dataset_identity),
        "entry_config_path": entry_config,
        "adapter_path": observed[0],
        "embedding_delta_path": observed[1],
    }


def _frozen_eval_image_ids(path: Path) -> set[str]:
    image_ids = [_input_image_id(row) for row in _read_jsonl_values(path)]
    if len(image_ids) != 256 or len(set(image_ids)) != 256:
        raise ScreenAnalysisError("frozen evaluation input must contain exactly 256 unique images")
    return set(image_ids)


def _positive_path_owner_ids(
    *, selection_receipt: Mapping[str, Any], label: str, admitted_image_ids: set[str]
) -> tuple[set[str], dict[str, Path]]:
    arms = _mapping(selection_receipt.get("arms"), field="state-banks-v2 root receipt arms")
    arm_receipt_path = _resolved(arms.get(STATE_BANK_ARM_KEYS[label]))
    arm_receipt = _read_json(arm_receipt_path)
    if arm_receipt.get("arm") != STATE_BANK_FAMILIES[label]:
        raise ScreenAnalysisError(f"state-banks-v2 {label} receipt has the wrong treatment family")
    counts = _mapping(arm_receipt.get("counts"), field=f"state-banks-v2 {label} counts")
    if counts.get("rollout_rows") != 992:
        raise ScreenAnalysisError(f"state-banks-v2 {label} receipt is not the exact 992-event bank")
    manifest_path = _resolved(arm_receipt.get("state_bank_manifest_path"))
    manifest = _read_json(manifest_path)
    records_path = _resolved(manifest_path.parent / str(manifest.get("records_file", "records.jsonl")))
    records = _read_jsonl_values(records_path)
    if manifest.get("record_count") != 992 or len(records) != 992:
        raise ScreenAnalysisError(f"state-banks-v2 {label} records are not the exact 992-event bank")
    if manifest.get("records_sha256") != _sha256_file(records_path):
        raise ScreenAnalysisError(f"state-banks-v2 {label} records checksum does not match manifest")
    owners: set[str] = set()
    for index, record in enumerate(records):
        if not bool(record.get("positive_path_imitation_eligible")):
            continue
        image = _mapping(record.get("image"), field=f"state-bank {label} record {index}.image")
        image_id = str(int(image.get("image_id")))
        if image_id not in admitted_image_ids:
            raise ScreenAnalysisError(f"state-bank {label} positive record is outside the admitted cohort")
        candidates = record.get("candidates")
        if not isinstance(candidates, list):
            raise ScreenAnalysisError(f"state-bank {label} record {index}.candidates must be a list")
        for candidate in candidates:
            item = _mapping(candidate, field=f"state-bank {label} candidate")
            if item.get("role") == "positive":
                owner = item.get("physical_owner_id")
                if not isinstance(owner, str) or not owner:
                    raise ScreenAnalysisError(f"state-bank {label} positive candidate lacks physical_owner_id")
                owners.add(owner)
    if not owners:
        raise ScreenAnalysisError(f"state-banks-v2 {label} records contain no positive-path owner IDs")
    return owners, {
        "arm_receipt": arm_receipt_path,
        "manifest": manifest_path,
        "records": records_path,
    }


def analyze_source_route_preservation_screen(
    *,
    selection_receipt_path: str | Path,
    frozen_eval_input_path: str | Path,
    source_run_dir: str | Path,
    single_route_run_dir: str | Path,
    multi_route_run_dir: str | Path,
    expected_model_paths: Mapping[str, tuple[str | Path, str | Path]],
    match_iou_threshold: float = 0.50,
    duplicate_iou_threshold: float = 0.30,
) -> dict[str, Any]:
    """Compare the three exact clean arms and return one deterministic receipt."""
    if not 0.0 <= match_iou_threshold <= 1.0:
        raise ScreenAnalysisError("match IoU threshold must be between 0 and 1")
    if not 0.0 <= duplicate_iou_threshold <= 1.0:
        raise ScreenAnalysisError("duplicate IoU threshold must be between 0 and 1")
    if set(expected_model_paths) != set(ARM_LABELS):
        raise ScreenAnalysisError("expected model paths must name Source, single-route, and multi-route arms")
    selection_receipt_path = _resolved(selection_receipt_path)
    selection_receipt = _read_json(selection_receipt_path)
    if selection_receipt.get("schema_version") != "source_preservation_multi_route_state_bank_assembler.v1":
        raise ScreenAnalysisError("selection receipt is not the state-banks-v2 root receipt")
    if selection_receipt.get("status") != "assembled":
        raise ScreenAnalysisError("state-banks-v2 root receipt is not assembled")
    admitted_image_ids = _admitted_image_ids(selection_receipt)
    frozen_eval_input = _resolved(frozen_eval_input_path)
    frozen_image_ids = _frozen_eval_image_ids(frozen_eval_input)
    if admitted_image_ids - frozen_image_ids:
        raise ScreenAnalysisError("state-banks-v2 admitted images are outside the frozen evaluation cohort")
    if len(frozen_image_ids - admitted_image_ids) != 138:
        raise ScreenAnalysisError("frozen evaluation cohort must contain exactly 138 non-admitted images")
    paths = {
        "source": _run_artifact_paths(source_run_dir),
        "single_route": _run_artifact_paths(single_route_run_dir),
        "multi_route": _run_artifact_paths(multi_route_run_dir),
    }
    run_identity = {
        label: _verify_run_identity(
            label=label,
            paths=paths[label],
            frozen_eval_input=frozen_eval_input,
            expected_model_paths=expected_model_paths,
        )
        for label in ARM_LABELS
    }
    source_generation = run_identity["source"]["generation_policy"]
    source_dataset = run_identity["source"]["dataset_identity"]
    for label in ("single_route", "multi_route"):
        if run_identity[label]["generation_policy"] != source_generation:
            raise ScreenAnalysisError(f"{label} generation policy does not match Source")
        if run_identity[label]["dataset_identity"] != source_dataset:
            raise ScreenAnalysisError(f"{label} dataset identity does not match Source")
    rows = {label: _read_jsonl(paths[label]["gt_vs_pred"]) for label in ARM_LABELS}
    # Establish stable physical-owner and image identities before any paired
    # comparison.  A fallback-to-index identity would make a bad artifact look
    # comparable, which is unacceptable for the owner ledger.
    for label in ARM_LABELS:
        for row in rows[label].values():
            _owner_ids(row)
    for label in ("single_route", "multi_route"):
        if set(rows[label]) != set(rows["source"]):
            raise ScreenAnalysisError(f"{label} row set does not match Source")
        for row_id in sorted(rows["source"]):
            if _gt_signature(rows[label][row_id], row_id=row_id) != _gt_signature(
                rows["source"][row_id], row_id=row_id
            ):
                raise ScreenAnalysisError(f"{label} GT mismatch for row_id {row_id!r}")
    for label in ARM_LABELS:
        arm_images = {_image_id(row) for row in rows[label].values()}
        if arm_images != frozen_image_ids:
            raise ScreenAnalysisError(f"{label} images do not exactly match the frozen 256-image cohort")
    sidecar_counts = {
        label: _validate_sidecars(
            label=label,
            rows=rows[label],
            summary=_read_json(paths[label]["summary"]),
            diagnostics=_read_diagnostics(paths[label]["parse_diagnostics"]),
        )
        for label in ARM_LABELS
    }
    arm_results: dict[str, Any] = {}
    arm_matches: dict[str, dict[str, set[str]]] = {}
    arm_geometry: dict[str, dict[str, dict[str, Any]]] = {}
    for label in ARM_LABELS:
        arm_results[label], arm_matches[label], arm_geometry[label] = _arm_receipt(
            rows=rows[label],
            sidecar_counts=sidecar_counts[label],
            admitted_image_ids=admitted_image_ids,
            match_iou_threshold=match_iou_threshold,
            duplicate_iou_threshold=duplicate_iou_threshold,
        )
    positive_owners: dict[str, set[str]] = {}
    state_bank_paths: dict[str, dict[str, Path]] = {}
    for label in ("single_route", "multi_route"):
        positive_owners[label], state_bank_paths[label] = _positive_path_owner_ids(
            selection_receipt=selection_receipt,
            label=label,
            admitted_image_ids=admitted_image_ids,
        )
    scopes = {
        "overall": frozen_image_ids,
        "admitted": admitted_image_ids,
        "non_admitted": frozen_image_ids - admitted_image_ids,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "inputs": {
            "selection_receipt": {
                "path": str(selection_receipt_path),
                "sha256": _sha256_file(selection_receipt_path),
            },
            "frozen_eval_input": {
                "path": str(frozen_eval_input),
                "sha256": _sha256_file(frozen_eval_input),
            },
            "runs": {
                label: {
                    name: {"path": str(path), "sha256": _sha256_file(path)}
                    for name, path in run_paths.items()
                    if name != "run_dir"
                }
                for label, run_paths in paths.items()
            },
            "state_banks_v2": {
                label: {
                    name: {"path": str(path), "sha256": _sha256_file(path)}
                    for name, path in arm_paths.items()
                }
                for label, arm_paths in state_bank_paths.items()
            },
        },
        "policy": {
            "match_iou_threshold": match_iou_threshold,
            "duplicate_iou_threshold": duplicate_iou_threshold,
            "duplicate_candidates_are_human_confirmed": False,
            "unmatched_predictions_are_not_hallucinations": True,
        },
        "admission": {
            "admitted_image_count": len(admitted_image_ids),
            "admitted_image_ids": sorted(admitted_image_ids, key=int),
            "non_admitted_image_count": len(frozen_image_ids - admitted_image_ids),
        },
        "provenance": run_identity,
        "arms": arm_results,
        "source_to_treatment": {
            label: {
                "selected_positive_path_owner_ids": sorted(positive_owners[label]),
                "by_scope": {
                    scope: {
                        "owner_ledger": _treatment_ledger(
                            source_matches=arm_matches["source"],
                            treatment_matches=arm_matches[label],
                            admitted_image_ids=admitted_image_ids,
                            image_ids=image_ids,
                            selected_positive_owner_ids=positive_owners[label],
                        ),
                        "common_owner_geometry": _geometry_delta(
                            source_geometry=arm_geometry["source"],
                            treatment_geometry=arm_geometry[label],
                            image_ids=image_ids,
                        ),
                    }
                    for scope, image_ids in scopes.items()
                },
            }
            for label in ("single_route", "multi_route")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-receipt", type=Path, required=True)
    parser.add_argument("--frozen-eval-input", type=Path, required=True)
    for label in ARM_LABELS:
        prefix = label.replace("_", "-")
        parser.add_argument(f"--{prefix}-run-dir", type=Path, required=True)
        parser.add_argument(f"--expected-{prefix}-adapter", type=Path, required=True)
        parser.add_argument(f"--expected-{prefix}-embedding-delta", type=Path, required=True)
    parser.add_argument("--match-iou-threshold", type=float, default=0.50)
    parser.add_argument("--duplicate-iou-threshold", type=float, default=0.30)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = analyze_source_route_preservation_screen(
        selection_receipt_path=args.selection_receipt,
        frozen_eval_input_path=args.frozen_eval_input,
        source_run_dir=args.source_run_dir,
        single_route_run_dir=args.single_route_run_dir,
        multi_route_run_dir=args.multi_route_run_dir,
        expected_model_paths={
            "source": (args.expected_source_adapter, args.expected_source_embedding_delta),
            "single_route": (
                args.expected_single_route_adapter,
                args.expected_single_route_embedding_delta,
            ),
            "multi_route": (
                args.expected_multi_route_adapter,
                args.expected_multi_route_embedding_delta,
            ),
        },
        match_iou_threshold=args.match_iou_threshold,
        duplicate_iou_threshold=args.duplicate_iou_threshold,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
