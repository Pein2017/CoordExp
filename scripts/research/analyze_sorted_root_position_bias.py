#!/usr/bin/env python3
"""CPU-only root-position diagnostic for the sorted owner census.

The diagnostic deliberately uses only ``context_role == "root"`` rows.  Once
the sorted model has emitted a row, ``(y1, x1)`` position and due-row index are
confounded, so post-root rows cannot answer the input-side question.

Outputs are deterministic and published create-or-identical:

* ``root-position-owner-rows.jsonl`` -- one row per owner;
* ``root-position-summary.json`` -- separate legacy/prospective slices;
* ``root-position-report.md`` -- a compact human-readable rendering;
* ``root-position-visual-spec.json`` -- data-only point specifications; and
* ``root-position-analysis-receipt.json`` -- input/output/source digests.

Raw cross-image log probabilities are never emitted or aggregated.  The only
score-derived fields retained are the already context-local ``peak_lift`` and
``local_concentration`` features plus the frozen calibrated support decisions.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import statistics
import sys
from typing import Any, NoReturn


UNIT_ID = "2026-08-04-sorted-image2299-prospective-mechanism-extension"
CENSUS_UNIT_ID = "2026-08-03-sorted-owner-accessibility-phenotype-census"

LEGACY_SLICE = "legacy_12"
PROSPECTIVE_SLICE = "prospective_2299"
PROSPECTIVE_IMAGE_ID = "2299"

LEGACY_PLAN_SCHEMA_VERSION = "sorted-owner-accessibility-census-plan.v1"
LEGACY_CONTEXT_SCHEMA_VERSION = "sorted-owner-accessibility-census-owner-context.v1"
LEGACY_SUMMARY_SCHEMA_VERSION = "sorted-owner-accessibility-census-owner-summary.v1"
LEGACY_MERGE_RECEIPT_SCHEMA_VERSION = "sorted-owner-accessibility-census-merge.v1"
PROSPECTIVE_ANALYSIS_SCHEMA_VERSION = "sorted-image2299-owner-accessibility-analysis.v1"
PROSPECTIVE_SUMMARY_SCHEMA_VERSION = (
    "sorted-image2299-owner-accessibility-owner-summary.v1"
)
PROSPECTIVE_CONTEXT_SCHEMA_VERSION = (
    "sorted-image2299-owner-accessibility-owner-context.v1"
)
PROSPECTIVE_RECEIPT_SCHEMA_VERSION = "sorted-image2299-owner-accessibility-receipt.v1"
PROSPECTIVE_EXPECTED_OWNER_COUNT = 46
PROSPECTIVE_EXPECTED_CATEGORY_COUNTS = {"person": 38, "tie": 8}

OWNER_ROW_SCHEMA_VERSION = "sorted-root-position-owner-row.v1"
SUMMARY_SCHEMA_VERSION = "sorted-root-position-summary.v1"
VISUAL_SPEC_SCHEMA_VERSION = "sorted-root-position-visual-spec.v1"
RECEIPT_SCHEMA_VERSION = "sorted-root-position-analysis-receipt.v1"

OWNER_ROWS_NAME = "root-position-owner-rows.jsonl"
SUMMARY_NAME = "root-position-summary.json"
REPORT_NAME = "root-position-report.md"
VISUAL_SPEC_NAME = "root-position-visual-spec.json"
RECEIPT_NAME = "root-position-analysis-receipt.json"

IMAGE_PAD_TOKEN_ID = 151655
MERGED_VISUAL_TOKEN_EDGE_PIXELS = 32

DEFAULT_LEGACY_RUN_ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-08-03-sorted-owner-accessibility-phenotype-census/20260803T065743Z"
)

ANALYSIS_FEATURES: tuple[str, ...] = (
    "center_x_normalized",
    "center_y_normalized",
    "area_normalized",
    "same_category_competitor_count",
    "manhattan_distance_to_visual_block_end_normalized",
    "diagonal_distance_to_visual_block_end_normalized",
    "raster_token_distance_to_visual_block_end_normalized",
)

CLAIM_BOUNDARY = (
    "descriptive root-context association only; no causal, attention, RoPE, "
    "routing-mechanism, architecture, loss, or population claim"
)


class RootPositionContractError(RuntimeError):
    """Raised when sealed inputs or the diagnostic contract do not agree."""


def _fail(message: str) -> NoReturn:
    raise RootPositionContractError(message)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def sha256_file(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"cannot read JSON {path}: {exc}")
    if not isinstance(value, dict):
        _fail(f"JSON document {path} is not an object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with Path(path).open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    _fail(f"{path}:{line_number} is not a JSON object")
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"cannot read JSONL {path}: {exc}")
    return rows


def assert_self_seal(document: Mapping[str, Any], *, path: Path) -> None:
    declared = document.get("receipt_content_sha256")
    if not isinstance(declared, str):
        _fail(f"{path} carries no receipt_content_sha256")
    payload = {
        key: value for key, value in document.items() if key != "receipt_content_sha256"
    }
    if declared != sha256_json(payload):
        _fail(f"{path} does not reconstruct its receipt_content_sha256")


def _digest_entry(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping) and isinstance(value.get("sha256"), str):
        return str(value["sha256"])
    return None


def assert_declared_digest(
    path: Path, digest_map: Mapping[str, Any], *, name: str | None = None
) -> None:
    key = name or path.name
    declared = _digest_entry(digest_map.get(key))
    if declared is None:
        _fail(f"sealed receipt does not declare a digest for {key}")
    actual = sha256_file(path)
    if actual != declared:
        _fail(f"{path} digest differs from its sealed receipt ({declared} != {actual})")


def _paths_from_features(features: Path) -> dict[str, Path]:
    features = Path(features).resolve()
    presentation_dir = features.parent
    run_root = presentation_dir.parent.parent
    plan_dir = run_root / "plan"
    return {
        "features": features,
        "summaries": presentation_dir / "owner-summaries.jsonl",
        "merge_receipt": presentation_dir / "merge-receipt.json",
        "owner_registry": plan_dir / "owner-registry.jsonl",
        "image_registry": plan_dir / "image-registry.jsonl",
        "context_registry": plan_dir / "context-registry.jsonl",
        "category_registry": plan_dir / "category-registry.jsonl",
        "plan_receipt": plan_dir / "receipt.json",
    }


def _paths_from_prospective_dirs(
    *, analysis_dir: Path, plan_dir: Path
) -> dict[str, Path]:
    analysis_dir = Path(analysis_dir).resolve()
    plan_dir = Path(plan_dir).resolve()
    return {
        "features": analysis_dir / "owner-context-features.jsonl",
        "summaries": analysis_dir / "owner-summaries.jsonl",
        "analysis_json": analysis_dir / "analysis.json",
        "analysis_context_registry": analysis_dir / "context-registry.jsonl",
        "analysis_receipt": analysis_dir / "receipt.json",
        "owner_registry": plan_dir / "owner-registry.jsonl",
        "image_registry": plan_dir / "image-registry.jsonl",
        "context_registry": plan_dir / "context-registry.jsonl",
        "category_registry": plan_dir / "category-registry.jsonl",
        "plan_receipt": plan_dir / "receipt.json",
    }


def _paths_from_run_root(run_root: Path) -> dict[str, Path]:
    return _paths_from_features(
        Path(run_root).resolve()
        / "phases"
        / "presentation"
        / "owner-context-features.jsonl"
    )


def _one_by_key(
    rows: Iterable[Mapping[str, Any]], *, key: str, label: str
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        value = row.get(key)
        if not isinstance(value, (str, int)):
            _fail(f"{label} row has no scalar {key}")
        text = str(value)
        if text in result:
            _fail(f"{label} repeats {key}={text!r}")
        result[text] = row
    return result


def _root_rows(
    rows: Sequence[Mapping[str, Any]], *, image_allowlist: set[str]
) -> tuple[dict[str, Mapping[str, Any]], int]:
    selected: dict[str, Mapping[str, Any]] = {}
    excluded = 0
    for row in rows:
        image_id = str(row.get("image_id"))
        if image_id not in image_allowlist:
            continue
        is_root = row.get("context_role") == "root" and row.get("boundary_index") == 0
        if not is_root:
            excluded += 1
            continue
        owner_id = str(row.get("gt_owner_id"))
        expected_context = f"{image_id}:boundary-000"
        if row.get("context_id") != expected_context:
            _fail(
                f"root row for {owner_id} has unexpected context_id {row.get('context_id')!r}"
            )
        if owner_id in selected:
            _fail(f"root features repeat owner {owner_id}")
        selected[owner_id] = row
    return selected, excluded


def _visual_grid(image: Mapping[str, Any]) -> dict[str, int]:
    width = image.get("image_width")
    height = image.get("image_height")
    prompt = image.get("prompt_token_ids")
    if (
        not isinstance(width, int)
        or not isinstance(height, int)
        or width <= 0
        or height <= 0
    ):
        _fail(f"image {image.get('image_id')} has invalid dimensions")
    if (
        width % MERGED_VISUAL_TOKEN_EDGE_PIXELS
        or height % MERGED_VISUAL_TOKEN_EDGE_PIXELS
    ):
        _fail(
            f"image {image.get('image_id')} dimensions do not form an exact 32-pixel grid"
        )
    if not isinstance(prompt, list) or not all(
        isinstance(token, int) for token in prompt
    ):
        _fail(f"image {image.get('image_id')} has no integer prompt_token_ids")
    pad_positions = [
        index for index, token in enumerate(prompt) if token == IMAGE_PAD_TOKEN_ID
    ]
    columns = width // MERGED_VISUAL_TOKEN_EDGE_PIXELS
    rows = height // MERGED_VISUAL_TOKEN_EDGE_PIXELS
    expected = rows * columns
    if len(pad_positions) != expected:
        _fail(
            f"image {image.get('image_id')} visual-pad count {len(pad_positions)} "
            f"does not match its {rows}x{columns} grid"
        )
    if not pad_positions or pad_positions[-1] - pad_positions[0] + 1 != len(
        pad_positions
    ):
        _fail(f"image {image.get('image_id')} visual pads are not one contiguous block")
    return {
        "rows": rows,
        "columns": columns,
        "token_count": expected,
        "prompt_token_count": len(prompt),
        "visual_start_prompt_index": pad_positions[0],
        "visual_end_prompt_index": pad_positions[-1],
        "query_anchor_prompt_index": len(prompt),
    }


def _quartiles(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)

    def interpolate(probability: float) -> float:
        position = probability * (len(ordered) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "minimum": ordered[0],
        "q1": interpolate(0.25),
        "median": interpolate(0.5),
        "q3": interpolate(0.75),
        "maximum": ordered[-1],
    }


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        stop = cursor
        while (
            stop + 1 < len(order) and values[order[stop + 1]] == values[order[cursor]]
        ):
            stop += 1
        rank = (cursor + stop) / 2.0 + 1.0
        for index in range(cursor, stop + 1):
            ranks[order[index]] = rank
        cursor = stop + 1
    return ranks


def spearman_rho(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) != len(ys):
        _fail("Spearman inputs have different lengths")
    if len(xs) < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    ranks_x = _average_ranks(xs)
    ranks_y = _average_ranks(ys)
    mean_x = statistics.fmean(ranks_x)
    mean_y = statistics.fmean(ranks_y)
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in zip(ranks_x, ranks_y))
    variance_x = sum((x - mean_x) ** 2 for x in ranks_x)
    variance_y = sum((y - mean_y) ** 2 for y in ranks_y)
    if variance_x <= 0.0 or variance_y <= 0.0:
        return None
    return covariance / math.sqrt(variance_x * variance_y)


def _support_from_summary(
    summary: Mapping[str, Any], *, context_id: str, bound_key: str
) -> bool:
    block = summary.get(bound_key)
    if not isinstance(block, Mapping):
        _fail(f"owner {summary.get('gt_owner_id')} lacks {bound_key}")
    by_context = block.get("non_loop_context_support")
    if not isinstance(by_context, Mapping) or context_id not in by_context:
        _fail(
            f"owner {summary.get('gt_owner_id')} lacks root support under {bound_key}"
        )
    value = by_context[context_id]
    if not isinstance(value, bool):
        _fail(f"owner {summary.get('gt_owner_id')} root support is not boolean")
    return value


def _local_features(root: Mapping[str, Any], *, bound: str) -> dict[str, float | None]:
    localization = root.get("localization")
    if not isinstance(localization, Mapping):
        _fail(f"root row {root.get('owner_context_id')} lacks localization")
    bounds = localization.get("generator_local_max_excluding_other_owner_strict")
    if not isinstance(bounds, Mapping):
        _fail(f"root row {root.get('owner_context_id')} lacks local bound features")
    block = bounds.get(bound)
    if not isinstance(block, Mapping):
        _fail(f"root row {root.get('owner_context_id')} lacks {bound}")
    result: dict[str, float | None] = {}
    for name in ("peak_lift", "local_concentration"):
        value = block.get(name)
        if value is not None and not isinstance(value, (int, float)):
            _fail(f"root row {root.get('owner_context_id')} has nonnumeric {name}")
        result[name] = None if value is None else float(value)
    return result


def _quadrant(x: float, y: float) -> str:
    vertical = "top" if y < 0.5 else "bottom"
    horizontal = "left" if x < 0.5 else "right"
    return f"{vertical}_{horizontal}"


def _native_outcome(owner: Mapping[str, Any]) -> str:
    native_tp = owner.get("native_true_positive")
    eligibility = owner.get("disposition_eligibility")
    native_fn = (
        eligibility.get("native_false_negative")
        if isinstance(eligibility, Mapping)
        else None
    )
    if native_tp is True and native_fn is not True:
        return "native_tp"
    if native_tp is False and native_fn is True:
        return "native_fn"
    if owner.get("excluded_from_census") is True:
        return "outside_native_matching_universe"
    _fail(f"owner {owner.get('gt_owner_id')} has contradictory native TP/FN state")


def _native_matching_membership(owner: Mapping[str, Any]) -> tuple[bool, str | None]:
    greedy_eligible = owner.get("greedy_eligible")
    if not isinstance(greedy_eligible, bool):
        _fail(f"owner {owner.get('gt_owner_id')} lacks boolean greedy_eligible")
    if greedy_eligible:
        return True, None
    status = owner.get("greedy_eligibility_status")
    return False, str(status) if status is not None else "not_greedy_eligible"


def build_owner_rows(
    *,
    slice_id: str,
    image_allowlist: set[str],
    features: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
    owners: Sequence[Mapping[str, Any]],
    images: Sequence[Mapping[str, Any]],
    contexts: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    owner_rows = [row for row in owners if str(row.get("image_id")) in image_allowlist]
    summary_rows = [
        row for row in summaries if str(row.get("image_id")) in image_allowlist
    ]
    image_rows = [row for row in images if str(row.get("image_id")) in image_allowlist]
    context_rows = [
        row for row in contexts if str(row.get("image_id")) in image_allowlist
    ]
    owner_by_id = _one_by_key(
        owner_rows, key="gt_owner_id", label=f"{slice_id} owner registry"
    )
    summary_by_id = _one_by_key(
        summary_rows, key="gt_owner_id", label=f"{slice_id} owner summaries"
    )
    image_by_id = _one_by_key(
        image_rows, key="image_id", label=f"{slice_id} image registry"
    )
    roots, excluded_context_feature_rows = _root_rows(
        features, image_allowlist=image_allowlist
    )
    if set(owner_by_id) != set(summary_by_id) or set(owner_by_id) != set(roots):
        _fail(
            f"{slice_id} owner/root/summary identities differ: owners={len(owner_by_id)}, "
            f"summaries={len(summary_by_id)}, roots={len(roots)}"
        )
    root_context_ids = {
        str(row.get("context_id"))
        for row in context_rows
        if row.get("context_role") == "root" and row.get("boundary_index") == 0
    }
    expected_context_ids = {f"{image_id}:boundary-000" for image_id in image_by_id}
    if root_context_ids != expected_context_ids:
        _fail(
            f"{slice_id} context registry does not contain exactly one root per image"
        )

    due_index: dict[str, int] = {}
    category_counts: dict[tuple[str, str], int] = defaultdict(int)
    for image_id in image_by_id:
        members = [row for row in owner_rows if str(row.get("image_id")) == image_id]
        members.sort(
            key=lambda row: (
                tuple(row.get("owner_sort_key") or ()),
                str(row.get("gt_owner_id")),
            )
        )
        for index, owner in enumerate(members):
            due_index[str(owner["gt_owner_id"])] = index
            category_counts[(image_id, str(owner.get("normalized_description")))] += 1

    grids = {image_id: _visual_grid(image) for image_id, image in image_by_id.items()}
    emitted: list[dict[str, Any]] = []
    for owner_id in sorted(
        owner_by_id, key=lambda value: (int(value.split(":")[1]), value)
    ):
        owner = owner_by_id[owner_id]
        summary = summary_by_id[owner_id]
        root = roots[owner_id]
        image_id = str(owner["image_id"])
        image = image_by_id[image_id]
        grid = grids[image_id]
        bbox = owner.get("bbox_pixel_xyxy")
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or not all(isinstance(value, (int, float)) for value in bbox)
        ):
            _fail(f"owner {owner_id} has invalid bbox_pixel_xyxy")
        x1, y1, x2, y2 = (float(value) for value in bbox)
        width = float(image["image_width"])
        height = float(image["image_height"])
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            _fail(f"owner {owner_id} box falls outside its image")
        center_x = ((x1 + x2) / 2.0) / width
        center_y = ((y1 + y2) / 2.0) / height
        area = ((x2 - x1) * (y2 - y1)) / (width * height)
        token_column = min(grid["columns"] - 1, int(((x1 + x2) / 2.0) // 32))
        token_row = min(grid["rows"] - 1, int(((y1 + y2) / 2.0) // 32))
        raster_index = token_row * grid["columns"] + token_column
        raster_distance = grid["token_count"] - 1 - raster_index
        manhattan = (grid["columns"] - 1 - token_column) + (
            grid["rows"] - 1 - token_row
        )
        diagonal = math.hypot(
            grid["columns"] - 1 - token_column,
            grid["rows"] - 1 - token_row,
        )
        manhattan_denominator = max(1, grid["columns"] + grid["rows"] - 2)
        diagonal_denominator = max(
            1.0, math.hypot(grid["columns"] - 1, grid["rows"] - 1)
        )
        raster_denominator = max(1, grid["token_count"] - 1)
        sequence_distance = grid["query_anchor_prompt_index"] - (
            grid["visual_start_prompt_index"] + raster_index
        )
        root_context_id = f"{image_id}:boundary-000"
        support_l = _support_from_summary(
            summary, context_id=root_context_id, bound_key="lower_bound_l"
        )
        support_u = _support_from_summary(
            summary, context_id=root_context_id, bound_key="upper_bound_u"
        )
        if support_l and not support_u:
            _fail(f"owner {owner_id} has impossible lower-only root support")
        support_state = (
            "supported_both_bounds"
            if support_l
            else "ambiguity_bound_flip"
            if support_u
            else "unsupported_both_bounds"
        )
        category = str(owner.get("normalized_description"))
        category_count = category_counts[(image_id, category)]
        native_outcome = _native_outcome(owner)
        native_false_negative = native_outcome == "native_fn"
        in_native_matching_universe, matching_exclusion_reason = (
            _native_matching_membership(owner)
        )
        emitted.append(
            {
                "schema_version": OWNER_ROW_SCHEMA_VERSION,
                "unit_id": UNIT_ID,
                "slice_id": slice_id,
                "split": summary.get("split"),
                "image_id": image_id,
                "gt_owner_id": owner_id,
                "normalized_description": category,
                "native_outcome": native_outcome,
                "native_true_positive": bool(owner.get("native_true_positive")),
                "native_false_negative": native_false_negative,
                "in_native_matching_universe": in_native_matching_universe,
                "native_false_negative_in_matching_universe": (
                    native_false_negative if in_native_matching_universe else None
                ),
                "native_matching_universe_exclusion_reason": matching_exclusion_reason,
                "in_false_negative_prevalence_denominator": bool(
                    summary.get("in_false_negative_prevalence_denominator")
                ),
                "support_disposition": summary.get("disposition"),
                "ambiguity_bound_disposition_flip": bool(
                    summary.get("ambiguity_bound_disposition_flip")
                ),
                "root_context": {
                    "context_id": root_context_id,
                    "context_role": "root",
                    "boundary_index": 0,
                    "support_lower_bound": support_l,
                    "support_upper_bound": support_u,
                    "support_state": support_state,
                    "local_calibrated_features": {
                        "lower_bound": _local_features(
                            root, bound="ambiguity_excluded_l"
                        ),
                        "upper_bound": _local_features(
                            root, bound="ambiguity_included_u"
                        ),
                        "role": "context_local_only_never_cross_image_raw_score",
                    },
                },
                "position": {
                    "bbox_pixel_xyxy": [x1, y1, x2, y2],
                    "center_x_normalized": center_x,
                    "center_y_normalized": center_y,
                    "quadrant": _quadrant(center_x, center_y),
                    "area_normalized": area,
                    "visual_grid_rows": grid["rows"],
                    "visual_grid_columns": grid["columns"],
                    "center_visual_token_row": token_row,
                    "center_visual_token_column": token_column,
                    "center_visual_raster_index": raster_index,
                    "raster_token_distance_to_visual_block_end": raster_distance,
                    "raster_token_distance_to_visual_block_end_normalized": (
                        raster_distance / raster_denominator
                    ),
                    "manhattan_distance_to_visual_block_end": manhattan,
                    "manhattan_distance_to_visual_block_end_normalized": (
                        manhattan / manhattan_denominator
                    ),
                    "diagonal_distance_to_visual_block_end": diagonal,
                    "diagonal_distance_to_visual_block_end_normalized": (
                        diagonal / diagonal_denominator
                    ),
                    "visual_block_end_prompt_index": grid["visual_end_prompt_index"],
                    "root_query_anchor_prompt_index": grid["query_anchor_prompt_index"],
                    "center_token_to_query_anchor_sequence_distance": sequence_distance,
                    "center_token_to_query_anchor_sequence_distance_normalized": (
                        sequence_distance / max(1, grid["prompt_token_count"])
                    ),
                    "visual_block_end_to_query_anchor_sequence_distance": (
                        grid["query_anchor_prompt_index"]
                        - grid["visual_end_prompt_index"]
                    ),
                    "distance_semantics": (
                        "owner center is mapped to the sealed 32x32 merged-token raster; "
                        "grid distances end at the bottom-right visual token and sequence "
                        "distance ends at the root category-query append anchor"
                    ),
                },
                "same_category_crowding": {
                    "same_category_owner_count": category_count,
                    "same_category_competitor_count": category_count - 1,
                    "definition": "same normalized description among physical owners in this image",
                },
                "sorted_due_index": {
                    "zero_based_index": due_index[owner_id],
                    "sort_key": owner.get("owner_sort_key"),
                    "role": "provenance_only_excluded_from_every_association_and_adjustment",
                    "confounded_with_position_after_root": True,
                },
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    emitted.sort(
        key=lambda row: (row["slice_id"], int(row["image_id"]), row["gt_owner_id"])
    )
    return emitted, {
        "input_owner_count": len(owner_by_id),
        "root_feature_row_count": len(roots),
        "post_root_feature_rows_excluded": excluded_context_feature_rows,
        "image_count": len(image_by_id),
    }


def _feature_value(row: Mapping[str, Any], name: str) -> float:
    if name == "same_category_competitor_count":
        return float(row["same_category_crowding"][name])
    return float(row["position"][name])


def _group_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    lower = [bool(row["root_context"]["support_lower_bound"]) for row in rows]
    upper = [bool(row["root_context"]["support_upper_bound"]) for row in rows]
    matching = [row for row in rows if row["in_native_matching_universe"]]
    outside = [row for row in rows if not row["in_native_matching_universe"]]
    native_fn_count = sum(
        row["native_false_negative_in_matching_universe"] is True for row in matching
    )
    return {
        "owner_count": len(rows),
        "root_support_lower_bound_count": sum(lower),
        "root_support_lower_bound_fraction": sum(lower) / len(rows) if rows else None,
        "root_support_upper_bound_count": sum(upper),
        "root_support_upper_bound_fraction": sum(upper) / len(rows) if rows else None,
        "native_matching_universe_owner_count": len(matching),
        "native_false_negative_count": native_fn_count,
        "native_false_negative_fraction": (
            native_fn_count / len(matching) if matching else None
        ),
        "outside_native_matching_universe_count": len(outside),
        "native_false_negative_denominator_semantics": (
            "native matching universe only; owners with greedy_eligible=false are excluded "
            "and counted separately"
        ),
        "feature_five_number_summaries": {
            name: _quartiles([_feature_value(row, name) for row in rows])
            for name in ANALYSIS_FEATURES
        },
    }


def _stratify(
    rows: Sequence[Mapping[str, Any]], getter: Any
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(getter(row))].append(row)
    return {key: _group_summary(groups[key]) for key in sorted(groups)}


def _within_image_associations(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_image: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_image[str(row["image_id"])].append(row)
    result: dict[str, Any] = {}
    for feature in ANALYSIS_FEATURES:
        per_image: dict[str, Any] = {}
        coefficients: list[float] = []
        for image_id in sorted(by_image, key=int):
            image_rows = by_image[image_id]
            coefficient = spearman_rho(
                [_feature_value(row, feature) for row in image_rows],
                [
                    float(row["root_context"]["support_lower_bound"])
                    for row in image_rows
                ],
            )
            per_image[image_id] = {
                "owner_count": len(image_rows),
                "spearman_rho": coefficient,
                "defined": coefficient is not None,
            }
            if coefficient is not None:
                coefficients.append(coefficient)
        result[feature] = {
            "support_target": "frozen_root_support_under_lower_ambiguity_bound",
            "stratification": "one coefficient per image; no cross-image raw score",
            "per_image": per_image,
            "defined_image_count": len(coefficients),
            "coefficient_five_number_summary": _quartiles(coefficients),
            "interpretation": "descriptive rank association only",
        }
    return result


def _within_image_native_fn_associations(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Feature associations with native FN, restricted to the matching universe."""

    by_image: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_image[str(row["image_id"])].append(row)
    result: dict[str, Any] = {}
    for feature in ANALYSIS_FEATURES:
        per_image: dict[str, Any] = {}
        coefficients: list[float] = []
        for image_id in sorted(by_image, key=int):
            all_image_rows = by_image[image_id]
            matching_rows = [
                row for row in all_image_rows if row["in_native_matching_universe"]
            ]
            coefficient = spearman_rho(
                [_feature_value(row, feature) for row in matching_rows],
                [
                    float(row["native_false_negative_in_matching_universe"])
                    for row in matching_rows
                ],
            )
            per_image[image_id] = {
                "native_matching_universe_owner_count": len(matching_rows),
                "outside_native_matching_universe_count": (
                    len(all_image_rows) - len(matching_rows)
                ),
                "native_false_negative_count": sum(
                    row["native_false_negative_in_matching_universe"] is True
                    for row in matching_rows
                ),
                "spearman_rho": coefficient,
                "defined": coefficient is not None,
            }
            if coefficient is not None:
                coefficients.append(coefficient)
        result[feature] = {
            "target": "native_false_negative_indicator_within_native_matching_universe",
            "independent_of_root_support_association": True,
            "stratification": (
                "one coefficient per image; outside-matching-universe owners excluded; "
                "no post-root context or sorted due index"
            ),
            "per_image": per_image,
            "defined_image_count": len(coefficients),
            "coefficient_five_number_summary": _quartiles(coefficients),
            "interpretation": "descriptive P(miss|input geometry) screen only; never causal",
        }
    return result


def build_slice_summary(
    slice_id: str, rows: Sequence[Mapping[str, Any]], diagnostics: Mapping[str, Any]
) -> dict[str, Any]:
    native_counts: dict[str, int] = defaultdict(int)
    disposition_counts: dict[str, int] = defaultdict(int)
    support_state_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        native_counts[str(row["native_outcome"])] += 1
        disposition_counts[str(row["support_disposition"])] += 1
        support_state_counts[str(row["root_context"]["support_state"])] += 1
    binding = diagnostics.get("binding")
    transfer = (
        binding.get("calibration_transfer") if isinstance(binding, Mapping) else None
    )
    if slice_id == PROSPECTIVE_SLICE:
        gate_met = isinstance(transfer, Mapping) and transfer.get("passes") is True
        root_support_status = (
            "frozen_root_support_transfer_gate_met"
            if gate_met
            else "descriptive_only_unmet_calibration_transfer_gate"
        )
    else:
        gate_met = None
        root_support_status = "frozen_legacy_root_support_descriptive"
    interpretation_scope = {
        "frozen_root_support_numbers_and_associations": {
            "status": root_support_status,
            "calibration_transfer_gate_met": gate_met,
            "calibration_transfer": dict(transfer)
            if isinstance(transfer, Mapping)
            else None,
            "applies_to": [
                "overall and stratified frozen root-support counts/fractions",
                "quadrant frozen root-support columns",
                "within_image_associations",
            ],
        },
        "native_fn_spatial_numbers_and_associations": {
            "status": "calibration_independent_unaffected",
            "depends_on_frozen_support_calibration": False,
            "applies_to": [
                "quadrant native-FN counts/fractions",
                "within_image_native_fn_associations",
            ],
        },
    }
    return {
        "slice_id": slice_id,
        "image_ids": sorted({str(row["image_id"]) for row in rows}, key=int),
        "diagnostics": dict(diagnostics),
        "overall": _group_summary(rows),
        "native_outcome_counts": dict(sorted(native_counts.items())),
        "support_disposition_counts": dict(sorted(disposition_counts.items())),
        "root_support_state_counts": dict(sorted(support_state_counts.items())),
        "interpretation_scope": interpretation_scope,
        "strata": {
            "native_outcome": _stratify(rows, lambda row: row["native_outcome"]),
            "support_disposition": _stratify(
                rows, lambda row: row["support_disposition"]
            ),
            "quadrant": _stratify(rows, lambda row: row["position"]["quadrant"]),
            "same_category_owner_count": _stratify(
                rows,
                lambda row: row["same_category_crowding"]["same_category_owner_count"],
            ),
        },
        "within_image_associations": _within_image_associations(rows),
        "within_image_native_fn_associations": (
            _within_image_native_fn_associations(rows)
        ),
        "adjustment_scope": (
            "separate image-stratified Spearman screens for frozen root support and native "
            "FN, plus exact observed strata; no fitted support threshold, causal coefficient, "
            "post-root t, sorted-due-index adjustment, or pooled raw likelihood"
        ),
    }


def assert_no_pooled_raw_scores(value: Any, *, label: str = "payload") -> None:
    """Reject score fields that would permit raw cross-image comparison."""

    forbidden = {
        "logprob",
        "log_probability",
        "raw_logprob",
        "raw_score",
        "raw_sequence_logprob_sum",
        "row_prefix_block_raw_sequence_logprob_sum",
        "complete_box_logprob",
        "complete_box_logprob_sum",
        "bank_median",
        "strict_assigned_max",
        "exact_anchor_score",
    }

    def walk(item: Any, path: str) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                lowered = str(key).lower()
                if lowered in forbidden:
                    _fail(f"{label} contains forbidden raw-score field {path}.{key}")
                walk(child, f"{path}.{key}")
        elif isinstance(item, list):
            for index, child in enumerate(item):
                walk(child, f"{path}[{index}]")

    walk(value, label)


def _assert_rows_schema(
    rows: Sequence[Mapping[str, Any]], *, expected: str, label: str
) -> None:
    drifted = sorted(
        {
            str(row.get("schema_version"))
            for row in rows
            if row.get("schema_version") != expected
        }
    )
    if drifted:
        _fail(f"{label} schema drifted from {expected!r}: {drifted}")


def _validate_plan_receipt(
    receipt: Mapping[str, Any], *, path: Path, layout: str
) -> Mapping[str, Any]:
    assert_self_seal(receipt, path=path)
    if receipt.get("schema_version") != LEGACY_PLAN_SCHEMA_VERSION:
        _fail(f"{path} is not a scorer-compatible owner-accessibility plan receipt")
    if receipt.get("unit_id") != CENSUS_UNIT_ID:
        _fail(f"{path} belongs to another scorer ABI unit")
    if layout == "prospective_s1":
        if receipt.get("extension_unit_id") != UNIT_ID:
            _fail(f"{path} is not the prospective image-2299 extension plan")
        shape = receipt.get("census_shape")
        denominator = receipt.get("denominator_contract")
        if not isinstance(shape, Mapping) or not isinstance(denominator, Mapping):
            _fail(f"{path} lacks prospective shape/denominator contracts")
        if (
            shape.get("image_ids") != [PROSPECTIVE_IMAGE_ID]
            or shape.get("owner_count") != PROSPECTIVE_EXPECTED_OWNER_COUNT
            or shape.get("owner_category_counts")
            != PROSPECTIVE_EXPECTED_CATEGORY_COUNTS
        ):
            _fail(f"{path} is not exactly the 46-owner 38-person/8-tie image-2299 plan")
        if (
            denominator.get("prospective_image2299_owner_count")
            != PROSPECTIVE_EXPECTED_OWNER_COUNT
            or denominator.get("legacy_12_owner_count_unchanged") != 346
            or denominator.get("legacy_12_eligible_native_fn_denominator_unchanged")
            != 202
            or denominator.get("pooled_13_image_denominator_created") is not False
            or denominator.get("report_slices_separately") is not True
        ):
            _fail(f"{path} prospective denominator contract drifted")
    return receipt


def _normalize_s1_summaries(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    _assert_rows_schema(
        rows,
        expected=PROSPECTIVE_SUMMARY_SCHEMA_VERSION,
        label="prospective S1 owner summaries",
    )
    normalized: list[dict[str, Any]] = []
    for row in rows:
        native_tp = row.get("native_true_positive")
        native_fn = row.get("native_false_negative")
        if not isinstance(native_tp, bool) or not isinstance(native_fn, bool):
            _fail(
                f"prospective summary {row.get('gt_owner_id')} lacks native TP/FN booleans"
            )
        if native_tp == native_fn:
            _fail(
                f"prospective summary {row.get('gt_owner_id')} has contradictory TP/FN state"
            )
        normalized.append(
            {
                **dict(row),
                "split": PROSPECTIVE_SLICE,
                "in_false_negative_prevalence_denominator": native_fn,
            }
        )
    return normalized


def _validate_s1_analysis(
    *,
    analysis: Mapping[str, Any],
    analysis_receipt: Mapping[str, Any],
    plan_receipt: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> None:
    if analysis.get("schema_version") != PROSPECTIVE_ANALYSIS_SCHEMA_VERSION:
        _fail("prospective S1 analysis.json schema drifted")
    if analysis.get("unit_id") != UNIT_ID or str(analysis.get("image_id")) != "2299":
        _fail("prospective S1 analysis.json has foreign unit/image identity")
    denominator = analysis.get("denominators")
    proposal_contract = analysis.get("proposal_localization_contract")
    calibration_transfer = analysis.get("calibration_transfer")
    if (
        not isinstance(denominator, Mapping)
        or not isinstance(proposal_contract, Mapping)
        or not isinstance(calibration_transfer, Mapping)
    ):
        _fail("prospective S1 analysis lacks denominator/proposal/transfer contracts")
    if (
        denominator.get("image2299_owner_count") != PROSPECTIVE_EXPECTED_OWNER_COUNT
        or denominator.get("legacy_12_owner_count_unchanged") != 346
        or denominator.get("legacy_12_eligible_native_fn_denominator_unchanged") != 202
        or denominator.get("pooled_13_image_denominator_created") is not False
    ):
        _fail("prospective S1 analysis denominator contract drifted")
    if (
        proposal_contract.get("proposal_surface_kept_separate") is not True
        or proposal_contract.get("proposal_surface_used_as_support_input") is not False
        or proposal_contract.get("localization_estimand")
        != "category_field_support_at_owner_geometry"
        or proposal_contract.get("is_per_owner_proposal_probability") is not False
    ):
        _fail("prospective S1 analysis weakened the localization estimand contract")
    if (
        calibration_transfer.get("passes") not in {True, False, None}
        or not isinstance(calibration_transfer.get("status"), str)
        or not isinstance(calibration_transfer.get("validity_bearing"), bool)
        or not isinstance(calibration_transfer.get("support_rate"), (int, float))
        or not isinstance(calibration_transfer.get("floor"), (int, float))
        or not isinstance(calibration_transfer.get("supported_due_boundary_count"), int)
        or not isinstance(
            calibration_transfer.get("transfer_denominator_native_tp_count"), int
        )
    ):
        _fail("prospective S1 calibration-transfer gate is malformed")
    assert_self_seal(analysis_receipt, path=paths["analysis_receipt"])
    if analysis_receipt.get("schema_version") != PROSPECTIVE_RECEIPT_SCHEMA_VERSION:
        _fail("prospective S1 analysis receipt schema drifted")
    if analysis_receipt.get("unit_id") != UNIT_ID:
        _fail("prospective S1 analysis receipt belongs to another unit")
    plan_binding = analysis_receipt.get("plan")
    if not isinstance(plan_binding, Mapping) or plan_binding.get(
        "receipt_content_sha256"
    ) != plan_receipt.get("receipt_content_sha256"):
        _fail("prospective S1 analysis receipt is bound to a different plan receipt")
    if analysis_receipt.get("denominator_contract") != denominator:
        _fail("prospective S1 analysis receipt denominator differs from analysis.json")
    output_digests = analysis_receipt.get("output_file_digests")
    if not isinstance(output_digests, Mapping):
        _fail("prospective S1 analysis receipt lacks output_file_digests")
    for key in (
        "features",
        "summaries",
        "analysis_json",
        "analysis_context_registry",
    ):
        assert_declared_digest(paths[key], output_digests)


def _load_slice(
    *,
    slice_id: str,
    paths: Mapping[str, Path],
    image_allowlist: set[str] | None,
    layout: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, str]]:
    for path in paths.values():
        if not path.is_file():
            _fail(f"required {slice_id} input is missing: {path}")
    plan_receipt = read_json(paths["plan_receipt"])
    _validate_plan_receipt(plan_receipt, path=paths["plan_receipt"], layout=layout)
    plan_digests = plan_receipt.get("output_file_digests")
    if not isinstance(plan_digests, Mapping):
        _fail(f"{slice_id} plan receipt does not carry output_file_digests")
    for key in (
        "owner_registry",
        "image_registry",
        "context_registry",
        "category_registry",
    ):
        assert_declared_digest(paths[key], plan_digests)

    features = read_jsonl(paths["features"])
    summaries = read_jsonl(paths["summaries"])
    owners = read_jsonl(paths["owner_registry"])
    images = read_jsonl(paths["image_registry"])
    contexts = read_jsonl(paths["context_registry"])
    categories = read_jsonl(paths["category_registry"])
    binding: dict[str, Any] = {
        "plan_receipt_content_sha256": str(plan_receipt["receipt_content_sha256"])
    }
    if layout == "prospective_s1":
        analysis = read_json(paths["analysis_json"])
        analysis_receipt = read_json(paths["analysis_receipt"])
        _validate_s1_analysis(
            analysis=analysis,
            analysis_receipt=analysis_receipt,
            plan_receipt=plan_receipt,
            paths=paths,
        )
        _assert_rows_schema(
            features,
            expected=PROSPECTIVE_CONTEXT_SCHEMA_VERSION,
            label="prospective S1 owner contexts",
        )
        summaries = _normalize_s1_summaries(summaries)
        analysis_contexts = read_jsonl(paths["analysis_context_registry"])
        _assert_rows_schema(
            analysis_contexts,
            expected=LEGACY_PLAN_SCHEMA_VERSION,
            label="prospective S1 analysis context registry",
        )
        if analysis_contexts != contexts:
            _fail("prospective S1 analysis and scorer plan context registries differ")
        prospective_owners = [
            row for row in owners if str(row.get("image_id")) == PROSPECTIVE_IMAGE_ID
        ]
        category_counts: dict[str, int] = defaultdict(int)
        for owner in prospective_owners:
            category_counts[str(owner.get("normalized_description"))] += 1
        if (
            len(prospective_owners) != PROSPECTIVE_EXPECTED_OWNER_COUNT
            or len(owners) != PROSPECTIVE_EXPECTED_OWNER_COUNT
            or dict(sorted(category_counts.items()))
            != PROSPECTIVE_EXPECTED_CATEGORY_COUNTS
            or {str(row.get("image_id")) for row in images} != {PROSPECTIVE_IMAGE_ID}
        ):
            _fail(
                "prospective S1 plan registries are not exactly 46 = 38 person + 8 tie"
            )
        native_tp_count = sum(
            owner.get("native_true_positive") is True for owner in prospective_owners
        )
        analysis_denominators = analysis["denominators"]
        plan_shape = plan_receipt["census_shape"]
        if (
            plan_shape.get("native_true_positive_owner_count") != native_tp_count
            or plan_shape.get("native_false_negative_owner_count")
            != PROSPECTIVE_EXPECTED_OWNER_COUNT - native_tp_count
        ):
            _fail(
                "prospective S1 plan native TP/FN counts disagree with its owner registry"
            )
        if (
            analysis_denominators.get("image2299_native_tp_count") != native_tp_count
            or analysis_denominators.get("image2299_native_fn_count")
            != PROSPECTIVE_EXPECTED_OWNER_COUNT - native_tp_count
        ):
            _fail("prospective S1 analysis native TP/FN counts disagree with its plan")
        binding["analysis_receipt_content_sha256"] = str(
            analysis_receipt["receipt_content_sha256"]
        )
        binding["analysis_schema_version"] = str(analysis["schema_version"])
        transfer = analysis["calibration_transfer"]
        binding["calibration_transfer"] = {
            "status": str(transfer["status"]),
            "passes": transfer["passes"],
            "validity_bearing": bool(transfer["validity_bearing"]),
            "support_rate": float(transfer["support_rate"]),
            "floor": float(transfer["floor"]),
            "supported_due_boundary_count": int(
                transfer["supported_due_boundary_count"]
            ),
            "transfer_denominator_native_tp_count": int(
                transfer["transfer_denominator_native_tp_count"]
            ),
        }
    else:
        merge_receipt = read_json(paths["merge_receipt"])
        assert_self_seal(merge_receipt, path=paths["merge_receipt"])
        if merge_receipt.get("schema_version") != LEGACY_MERGE_RECEIPT_SCHEMA_VERSION:
            _fail(f"{slice_id} legacy merge receipt schema drifted")
        if merge_receipt.get("unit_id") != CENSUS_UNIT_ID:
            _fail(f"{slice_id} legacy merge receipt belongs to another unit")
        merge_digests = merge_receipt.get("output_file_digests")
        if not isinstance(merge_digests, Mapping):
            _fail(f"{slice_id} merge receipt does not carry output_file_digests")
        assert_declared_digest(paths["features"], merge_digests)
        assert_declared_digest(paths["summaries"], merge_digests)
        _assert_rows_schema(
            features,
            expected=LEGACY_CONTEXT_SCHEMA_VERSION,
            label=f"{slice_id} legacy owner contexts",
        )
        _assert_rows_schema(
            summaries,
            expected=LEGACY_SUMMARY_SCHEMA_VERSION,
            label=f"{slice_id} legacy owner summaries",
        )
        binding["merge_receipt_content_sha256"] = str(
            merge_receipt["receipt_content_sha256"]
        )
    _assert_rows_schema(
        owners,
        expected=LEGACY_PLAN_SCHEMA_VERSION,
        label=f"{slice_id} owner registry",
    )
    _assert_rows_schema(
        images,
        expected=LEGACY_PLAN_SCHEMA_VERSION,
        label=f"{slice_id} image registry",
    )
    _assert_rows_schema(
        contexts,
        expected=LEGACY_PLAN_SCHEMA_VERSION,
        label=f"{slice_id} plan context registry",
    )
    _assert_rows_schema(
        categories,
        expected=LEGACY_PLAN_SCHEMA_VERSION,
        label=f"{slice_id} category registry",
    )
    del categories  # identity-sealed provenance; crowding is reconstructed from owners.
    available_images = {str(row.get("image_id")) for row in images}
    selected_images = (
        available_images
        if image_allowlist is None
        else available_images & image_allowlist
    )
    if not selected_images:
        _fail(f"{slice_id} contains none of the requested images")
    if slice_id == PROSPECTIVE_SLICE and selected_images != {PROSPECTIVE_IMAGE_ID}:
        _fail(f"prospective slice must resolve to exactly image {PROSPECTIVE_IMAGE_ID}")
    rows, diagnostics = build_owner_rows(
        slice_id=slice_id,
        image_allowlist=selected_images,
        features=features,
        summaries=summaries,
        owners=owners,
        images=images,
        contexts=contexts,
    )
    input_digests = {str(path): sha256_file(path) for path in paths.values()}
    binding["selected_image_ids"] = sorted(selected_images, key=int)
    binding["input_layout"] = layout
    diagnostics = {**diagnostics, "binding": binding}
    return rows, diagnostics, input_digests


def run_analysis(
    *,
    legacy_run_root: Path = DEFAULT_LEGACY_RUN_ROOT,
    prospective_features: Path | None = None,
    prospective_analysis_dir: Path | None = None,
    prospective_plan_dir: Path | None = None,
) -> dict[str, Any]:
    if (prospective_analysis_dir is None) != (prospective_plan_dir is None):
        _fail(
            "--prospective-analysis-dir and --prospective-plan-dir must be provided together"
        )
    if prospective_features is not None and prospective_analysis_dir is not None:
        _fail(
            "--prospective-features cannot be combined with the explicit prospective "
            "analysis/plan directory inputs"
        )
    legacy_rows, legacy_diagnostics, legacy_digests = _load_slice(
        slice_id=LEGACY_SLICE,
        paths=_paths_from_run_root(legacy_run_root),
        image_allowlist=None,
        layout="legacy_census",
    )
    rows_by_slice: dict[str, list[dict[str, Any]]] = {LEGACY_SLICE: legacy_rows}
    diagnostics_by_slice: dict[str, dict[str, Any]] = {LEGACY_SLICE: legacy_diagnostics}
    input_digests: dict[str, str] = dict(legacy_digests)
    if prospective_analysis_dir is not None and prospective_plan_dir is not None:
        prospective_paths = _paths_from_prospective_dirs(
            analysis_dir=prospective_analysis_dir, plan_dir=prospective_plan_dir
        )
        prospective_layout = "prospective_s1"
    elif prospective_features is not None:
        prospective_paths = _paths_from_features(prospective_features)
        prospective_layout = "legacy_census"
    else:
        prospective_paths = None
        prospective_layout = None
    if prospective_paths is not None and prospective_layout is not None:
        prospective_rows, prospective_diagnostics, prospective_digests = _load_slice(
            slice_id=PROSPECTIVE_SLICE,
            paths=prospective_paths,
            image_allowlist={PROSPECTIVE_IMAGE_ID},
            layout=prospective_layout,
        )
        rows_by_slice[PROSPECTIVE_SLICE] = prospective_rows
        diagnostics_by_slice[PROSPECTIVE_SLICE] = prospective_diagnostics
        for path, digest in prospective_digests.items():
            existing = input_digests.get(path)
            if existing is not None and existing != digest:
                _fail(f"input path {path} has two different digests")
            input_digests[path] = digest

    slice_order = (
        [PROSPECTIVE_SLICE, LEGACY_SLICE]
        if PROSPECTIVE_SLICE in rows_by_slice
        else [LEGACY_SLICE]
    )
    owner_rows = [row for slice_id in slice_order for row in rows_by_slice[slice_id]]
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "analysis_name": "S4_input_side_root_position_diagnostic",
        "slice_order": slice_order,
        "slices_are_never_pooled": True,
        "slices": {
            slice_id: build_slice_summary(
                slice_id, rows_by_slice[slice_id], diagnostics_by_slice[slice_id]
            )
            for slice_id in slice_order
        },
        "confounding_guard": {
            "eligible_context_role": "root",
            "eligible_boundary_index": 0,
            "post_root_rows_excluded": True,
            "sorted_due_index_reported_as_provenance_only": True,
            "sorted_due_index_enters_any_analysis": False,
            "post_root_t_enters_native_fn_analysis": False,
            "reason": (
                "after root, geometry-sorted (y1,x1) due order and owner position are confounded"
            ),
        },
        "score_policy": {
            "raw_cross_image_log_probabilities_compared": False,
            "raw_scores_emitted": False,
            "retained_score_derived_fields": ["peak_lift", "local_concentration"],
            "retained_fields_scope": "context-local calibrated features and support only",
        },
        "native_fn_screen_policy": {
            "target": "native_false_negative_indicator_within_native_matching_universe",
            "outside_native_matching_universe_reported_separately": True,
            "independent_of_frozen_root_support_association": True,
            "post_root_t_used": False,
            "sorted_due_index_used": False,
            "claim": "descriptive MCA-like P(miss|x,y,input features) screen only",
        },
        "claim_boundary": CLAIM_BOUNDARY,
        "not_claimed": [
            "no causal position effect",
            "no RoPE or attention mechanism",
            "no new support threshold",
            "no pooled thirteen-image result",
            "no independent-population claim for image 2299",
        ],
        "compute_scope": "CPU-only deterministic analysis; no model, tokenizer, or media opened",
    }
    visual_spec = {
        "schema_version": VISUAL_SPEC_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "data_only": True,
        "slice_order": slice_order,
        "encodings": {
            "x": "position.center_x_normalized",
            "y": "position.center_y_normalized (render top at zero)",
            "color": "root_context.support_state",
            "shape": "native_outcome",
            "facet": "slice_id then image_id",
            "size": "sqrt(position.area_normalized)",
        },
        "quadrant_boundaries": {"x": 0.5, "y": 0.5, "descriptive_only": True},
        "points": [
            {
                "slice_id": row["slice_id"],
                "image_id": row["image_id"],
                "gt_owner_id": row["gt_owner_id"],
                "x": row["position"]["center_x_normalized"],
                "y": row["position"]["center_y_normalized"],
                "area": row["position"]["area_normalized"],
                "quadrant": row["position"]["quadrant"],
                "support_state": row["root_context"]["support_state"],
                "native_outcome": row["native_outcome"],
                "in_native_matching_universe": row["in_native_matching_universe"],
                "native_false_negative_in_matching_universe": row[
                    "native_false_negative_in_matching_universe"
                ],
                "same_category_competitor_count": row["same_category_crowding"][
                    "same_category_competitor_count"
                ],
            }
            for row in owner_rows
        ],
        "claim_boundary": CLAIM_BOUNDARY,
    }
    assert_no_pooled_raw_scores(owner_rows, label="owner rows")
    assert_no_pooled_raw_scores(summary, label="summary")
    assert_no_pooled_raw_scores(visual_spec, label="visual spec")
    return {
        "owner_rows": owner_rows,
        "summary": summary,
        "visual_spec": visual_spec,
        "input_file_sha256": dict(sorted(input_digests.items())),
    }


def render_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Sorted root-position diagnostic",
        "",
        "This is a descriptive, CPU-only root-context readout. Post-root rows and sorted due "
        "index are excluded from every association because position and due order are confounded.",
        "",
    ]
    for slice_id in summary["slice_order"]:
        block = summary["slices"][slice_id]
        overall = block["overall"]
        lines.extend(
            [
                f"## {slice_id}",
                "",
                f"- Images: `{', '.join(block['image_ids'])}`",
                f"- Owners: `{overall['owner_count']}`",
                "- Frozen root support (lower bound): "
                f"`{overall['root_support_lower_bound_count']}/{overall['owner_count']}` "
                f"(`{overall['root_support_lower_bound_fraction']:.3f}`)",
                "- Frozen root support (upper bound): "
                f"`{overall['root_support_upper_bound_count']}/{overall['owner_count']}` "
                f"(`{overall['root_support_upper_bound_fraction']:.3f}`)",
                f"- Native outcomes: `{json.dumps(block['native_outcome_counts'], sort_keys=True)}`",
                "- Native matching universe: "
                f"`{overall['native_false_negative_count']}/"
                f"{overall['native_matching_universe_owner_count']}` native FNs "
                f"(`{overall['native_false_negative_fraction']:.3f}`); "
                f"outside universe: `{overall['outside_native_matching_universe_count']}`",
                "- Support dispositions: "
                f"`{json.dumps(block['support_disposition_counts'], sort_keys=True)}`",
            ]
        )
        if slice_id == PROSPECTIVE_SLICE:
            root_scope = block["interpretation_scope"][
                "frozen_root_support_numbers_and_associations"
            ]
            transfer = root_scope["calibration_transfer"]
            if isinstance(transfer, Mapping):
                transfer_detail = (
                    f"status `{transfer['status']}`, "
                    f"support `{transfer['supported_due_boundary_count']}/"
                    f"{transfer['transfer_denominator_native_tp_count']} = "
                    f"{transfer['support_rate']:.3f}` against floor `{transfer['floor']:.3f}`"
                )
            else:
                transfer_detail = "transfer status unavailable"
            lines.extend(
                [
                    "- Frozen root-support interpretation: "
                    f"`{root_scope['status']}` ({transfer_detail}). This qualification applies "
                    "to root-support counts, quadrant columns, and root-support associations.",
                    "- Native-FN spatial interpretation: `calibration_independent_unaffected`; "
                    "its quadrant values and associations do not use the frozen support gate.",
                ]
            )
        lines.extend(
            [
                "",
                "### Quadrants",
                "",
                "| Quadrant | Owners | Lower-bound root support | Native FN / matching universe "
                "| Outside universe |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for quadrant, group in block["strata"]["quadrant"].items():
            fraction = group["root_support_lower_bound_fraction"]
            native_fn_fraction = group["native_false_negative_fraction"]
            native_fn_rendered = (
                "n/a"
                if native_fn_fraction is None
                else (
                    f"{group['native_false_negative_count']}/"
                    f"{group['native_matching_universe_owner_count']} "
                    f"({native_fn_fraction:.3f})"
                )
            )
            lines.append(
                f"| {quadrant} | {group['owner_count']} | "
                f"{group['root_support_lower_bound_count']}/{group['owner_count']} "
                f"({fraction:.3f}) | {native_fn_rendered} | "
                f"{group['outside_native_matching_universe_count']} |"
            )
        lines.extend(
            [
                "",
                "### Image-stratified frozen-root-support associations",
                "",
                "Spearman coefficients are computed separately within each image against "
                "frozen lower-bound root support; the table robustly summarizes those image-level "
                "coefficients. For a prospective slice with an unmet calibration-transfer gate, "
                "this entire table is descriptive only.",
                "",
                "| Feature | Defined images | Q1 rho | Median rho | Q3 rho |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for feature, association in block["within_image_associations"].items():
            five = association["coefficient_five_number_summary"]
            if five is None:
                rendered = ("n/a", "n/a", "n/a")
            else:
                rendered = tuple(f"{five[key]:.3f}" for key in ("q1", "median", "q3"))
            lines.append(
                f"| {feature} | {association['defined_image_count']} | {rendered[0]} | "
                f"{rendered[1]} | {rendered[2]} |"
            )
        lines.extend(
            [
                "",
                "### Image-stratified native-FN associations",
                "",
                "This second screen is independent of frozen root support. Native FN is defined "
                "only inside the native matching universe; outside-universe owners are excluded "
                "and reported in the quadrant table.",
                "",
                "| Feature | Defined images | Q1 rho | Median rho | Q3 rho |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for feature, association in block[
            "within_image_native_fn_associations"
        ].items():
            five = association["coefficient_five_number_summary"]
            if five is None:
                rendered = ("n/a", "n/a", "n/a")
            else:
                rendered = tuple(f"{five[key]:.3f}" for key in ("q1", "median", "q3"))
            lines.append(
                f"| {feature} | {association['defined_image_count']} | {rendered[0]} | "
                f"{rendered[1]} | {rendered[2]} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Interpretation boundary",
            "",
            f"{summary['claim_boundary']}. Raw cross-image log probabilities are not emitted "
            "or compared. Frozen-root-support and native-FN associations are separate, "
            "within-image screens; the native-FN screen excludes owners outside the native "
            "matching universe and uses no post-root t or sorted due index.",
            "",
        ]
    )
    return "\n".join(lines)


def build_output_files(result: Mapping[str, Any]) -> dict[str, bytes]:
    owner_bytes = b"".join(
        canonical_json_bytes(row) + b"\n" for row in result["owner_rows"]
    )
    summary_bytes = (
        json.dumps(result["summary"], indent=2, sort_keys=True, ensure_ascii=False)
        + "\n"
    ).encode("utf-8")
    report_bytes = render_markdown(result["summary"]).encode("utf-8")
    visual_bytes = (
        json.dumps(result["visual_spec"], indent=2, sort_keys=True, ensure_ascii=False)
        + "\n"
    ).encode("utf-8")
    outputs = {
        OWNER_ROWS_NAME: owner_bytes,
        SUMMARY_NAME: summary_bytes,
        REPORT_NAME: report_bytes,
        VISUAL_SPEC_NAME: visual_bytes,
    }
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "census_unit_id": CENSUS_UNIT_ID,
        "analyzer_source_sha256": sha256_file(Path(__file__).resolve()),
        "input_file_sha256": dict(result["input_file_sha256"]),
        "slice_order": list(result["summary"]["slice_order"]),
        "slices_are_never_pooled": True,
        "artifact_determinism": (
            "no wall-clock, random, or output-path field is sealed; exact input paths and "
            "their byte digests are sealed"
        ),
        "policy": {
            "root_only": True,
            "sorted_due_index_enters_any_analysis": False,
            "post_root_t_enters_any_analysis": False,
            "raw_cross_image_log_probabilities_compared": False,
            "new_threshold_fitted": False,
            "native_fn_target": (
                "native_false_negative_indicator_within_native_matching_universe"
            ),
            "outside_native_matching_universe_reported_separately": True,
            "claim_boundary": CLAIM_BOUNDARY,
        },
        "output_file_digests": {
            name: {
                "path": name,
                "byte_size": len(content),
                "sha256": sha256_bytes(content),
                **(
                    {"row_count": len(result["owner_rows"])}
                    if name == OWNER_ROWS_NAME
                    else {}
                ),
            }
            for name, content in outputs.items()
        },
    }
    assert_no_pooled_raw_scores(receipt, label="receipt")
    receipt["receipt_content_sha256"] = sha256_json(receipt)
    return {**outputs, RECEIPT_NAME: canonical_json_bytes(receipt) + b"\n"}


def publish_analysis(output_dir: Path, files: Mapping[str, bytes]) -> dict[str, Any]:
    output_dir = Path(output_dir)
    if output_dir.exists():
        present = sorted(entry.name for entry in output_dir.iterdir())
        missing = sorted(set(files) - set(present))
        unexpected = sorted(set(present) - set(files))
        differing = sorted(
            name
            for name, content in files.items()
            if name in present and (output_dir / name).read_bytes() != content
        )
        if not missing and not unexpected and not differing:
            return {
                "output_dir": str(output_dir),
                "published": False,
                "publish_mode": "no_op_identical_rerun",
                "file_names": sorted(files),
            }
        _fail(
            f"refusing non-identical publication at {output_dir} "
            f"(missing={missing}, differing={differing}, unexpected={unexpected})"
        )
    staging = output_dir.parent / f"{output_dir.name}.staging-{os.getpid()}"
    if staging.exists():
        _fail(f"staging directory already exists: {staging}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    published = False
    try:
        for name in sorted(files):
            (staging / name).write_bytes(files[name])
        os.rename(staging, output_dir)
        published = True
    finally:
        if not published and staging.exists():
            for child in staging.iterdir():
                child.unlink()
            staging.rmdir()
    return {
        "output_dir": str(output_dir),
        "published": True,
        "publish_mode": "atomic_staging_directory_rename",
        "file_names": sorted(files),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--legacy-run-root",
        type=Path,
        default=DEFAULT_LEGACY_RUN_ROOT,
        help="sealed legacy census run root (defaults to the frozen 20260803T065743Z run)",
    )
    parser.add_argument(
        "--prospective-features",
        type=Path,
        default=None,
        help=(
            "optional image-2299 owner-context-features.jsonl; owner summaries and plan "
            "registries are inferred from the same census run tree"
        ),
    )
    parser.add_argument(
        "--prospective-analysis-dir",
        type=Path,
        default=None,
        help=(
            "actual image-2299 S1 analysis directory containing analysis.json, receipt.json, "
            "owner summaries, owner contexts, and its context registry"
        ),
    )
    parser.add_argument(
        "--prospective-plan-dir",
        type=Path,
        default=None,
        help=(
            "scorer-compatible image-2299 S1 plan directory; required with "
            "--prospective-analysis-dir"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_analysis(
            legacy_run_root=args.legacy_run_root,
            prospective_features=args.prospective_features,
            prospective_analysis_dir=args.prospective_analysis_dir,
            prospective_plan_dir=args.prospective_plan_dir,
        )
        published = publish_analysis(args.output_dir, build_output_files(result))
    except RootPositionContractError as exc:
        raise SystemExit(f"root-position diagnostic contract violated: {exc}") from exc
    compact = {
        "analysis": published,
        "slice_order": result["summary"]["slice_order"],
        "slices": {
            slice_id: {
                "owner_count": block["overall"]["owner_count"],
                "root_support_lower_bound_count": block["overall"][
                    "root_support_lower_bound_count"
                ],
                "root_support_lower_bound_fraction": block["overall"][
                    "root_support_lower_bound_fraction"
                ],
            }
            for slice_id, block in result["summary"]["slices"].items()
        },
    }
    print(json.dumps(compact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
