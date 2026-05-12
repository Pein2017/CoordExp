#!/usr/bin/env python
"""Calibrate IoU-Gibbs coordinate targets over CoordExp JSONL bboxes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import shlex
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
repo_root_text = str(REPO_ROOT)
sys.path = [path for path in sys.path if path != repo_root_text]
sys.path.insert(0, repo_root_text)

from src.detection.coord_soft_targets import (
    CoordSoftTargetCandidate,
    CoordSoftTargetDistributionName,
    CoordSoftTargetRuntimeConfig,
    build_iou_gibbs_coord_target,
)

COORD_MIN = 0
COORD_MAX = 999
SLOT_NAMES = ("x1", "y1", "x2", "y2")
DEFAULT_TAU = 0.0090909091
BBOX_FORMAT = "xyxy"
COORDINATE_SURFACE = "coord_token_xyxy"
COORD_TOKEN_RE = re.compile(r"^<\|coord_(\d+)\|>$")
TARGET_METRICS = (
    "entropy",
    "perplexity",
    "peak_prob",
    "std",
    "support_bin_count",
    "effective_support_size",
)


def iou_xyxy(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    inter = inter_w * inter_h
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def one_token_iou_losses(bbox_xyxy: tuple[int, int, int, int]) -> list[float]:
    x1, y1, x2, y2 = bbox_xyxy
    losses: list[float] = []
    for slot_index in range(4):
        for delta in (-1, 1):
            candidate = [x1, y1, x2, y2]
            candidate[slot_index] += delta
            cx1, cy1, cx2, cy2 = candidate
            if not (0 <= cx1 < cx2 <= 999 and 0 <= cy1 < cy2 <= 999):
                continue
            losses.append(1.0 - iou_xyxy((x1, y1, x2, y2), (cx1, cy1, cx2, cy2)))
    return losses


def summarize_losses(losses: Iterable[float]) -> dict[str, float | int | None]:
    return summarize_values(list(losses))


def summarize_values(values: Iterable[float]) -> dict[str, float | int | None]:
    sorted_values = sorted(float(value) for value in values)
    count = len(sorted_values)
    if count == 0:
        return {
            "count": 0,
            "median": None,
            "mean": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "p99": None,
        }
    mean = sum(sorted_values) / count
    return {
        "count": count,
        "median": _percentile(sorted_values, 50.0),
        "mean": mean,
        "p75": _percentile(sorted_values, 75.0),
        "p90": _percentile(sorted_values, 90.0),
        "p95": _percentile(sorted_values, 95.0),
        "p99": _percentile(sorted_values, 99.0),
    }


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        raise ValueError("percentile requires at least one value")
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (percentile / 100.0) * (len(sorted_values) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_values[int(rank)]
    weight = rank - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _parse_coord(value: Any, *, line_no: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"line {line_no}: bbox coordinate must not be bool")
    if isinstance(value, int):
        coord = value
    elif isinstance(value, str):
        match = COORD_TOKEN_RE.match(value)
        if match is None:
            raise ValueError(f"line {line_no}: unsupported coord token {value!r}")
        coord = int(match.group(1))
    else:
        raise TypeError(f"line {line_no}: bbox coordinate must be int or coord token")
    if not (COORD_MIN <= coord <= COORD_MAX):
        raise ValueError(f"line {line_no}: bbox coordinate {coord} outside 0..999")
    return coord


def _extract_bboxes(payload: dict[str, Any], *, line_no: int) -> list[tuple[int, int, int, int]]:
    objects = payload.get("objects")
    if not isinstance(objects, list):
        raise TypeError(f"line {line_no}: expected top-level objects list")
    bboxes: list[tuple[int, int, int, int]] = []
    for object_index, obj in enumerate(objects):
        if not isinstance(obj, dict):
            raise TypeError(f"line {line_no}: object {object_index} must be a mapping")
        raw_bbox = obj.get("bbox_2d")
        if raw_bbox is None:
            continue
        if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
            raise ValueError(f"line {line_no}: object {object_index} bbox_2d must have four slots")
        bbox = tuple(_parse_coord(value, line_no=line_no) for value in raw_bbox)
        if not _is_valid_bbox(bbox):
            raise ValueError(f"line {line_no}: object {object_index} has invalid bbox {bbox}")
        bboxes.append(bbox)
    return bboxes


def _is_valid_bbox(bbox: tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = bbox
    return COORD_MIN <= x1 < x2 <= COORD_MAX and COORD_MIN <= y1 < y2 <= COORD_MAX


def _candidate_values_for_slot(bbox: tuple[int, int, int, int], slot_index: int) -> range:
    x1, y1, x2, y2 = bbox
    if slot_index == 0:
        return range(COORD_MIN, x2)
    if slot_index == 1:
        return range(COORD_MIN, y2)
    if slot_index == 2:
        return range(x1 + 1, COORD_MAX + 1)
    if slot_index == 3:
        return range(y1 + 1, COORD_MAX + 1)
    raise ValueError(f"unsupported slot index {slot_index}")


def target_shape_stats_for_slot(
    bbox: tuple[int, int, int, int],
    slot_index: int,
    *,
    tau: float = DEFAULT_TAU,
    target_distribution: CoordSoftTargetDistributionName = "iou_gibbs_v0",
) -> dict[str, float | int]:
    if tau <= 0.0 or not math.isfinite(tau):
        raise ValueError("tau must be finite and > 0")
    dist = build_iou_gibbs_coord_target(
        (
            CoordSoftTargetCandidate(
                object_instance_id="analysis",
                slot_name=SLOT_NAMES[slot_index],
                bbox_xyxy=bbox,
                probability=1.0,
            ),
        ),
        CoordSoftTargetRuntimeConfig(
            target_distribution=target_distribution,
            tau=float(tau),
            coord_token_start=COORD_MIN,
            coord_token_end=COORD_MAX,
        ),
    )
    return {
        "entropy": float(dist.entropy.item()),
        "perplexity": float(dist.perplexity.item()),
        "peak_prob": float(dist.peak_prob.item()),
        "std": float(dist.std.item()),
        "support_bin_count": int(dist.support_bin_count.item()),
        "effective_support_size": float(dist.effective_support_size.item()),
    }


def _one_slot_iou_loss(bbox: tuple[int, int, int, int], slot_index: int, coord: int) -> float:
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    if slot_index == 0:
        iou = width / (x2 - coord) if coord <= x1 else (x2 - coord) / width
    elif slot_index == 1:
        iou = height / (y2 - coord) if coord <= y1 else (y2 - coord) / height
    elif slot_index == 2:
        iou = width / (coord - x1) if coord >= x2 else (coord - x1) / width
    elif slot_index == 3:
        iou = height / (coord - y1) if coord >= y2 else (coord - y1) / height
    else:
        raise ValueError(f"unsupported slot index {slot_index}")
    return 1.0 - iou


def analyze_jsonl(
    jsonl_path: Path,
    *,
    target_audit_sample: int,
    tau: float = DEFAULT_TAU,
    rng_seed: int = 0,
    bbox_format: str = BBOX_FORMAT,
    coordinate_surface: str = COORDINATE_SURFACE,
    target_distribution: CoordSoftTargetDistributionName = "iou_gibbs_v0",
    command: str | None = None,
) -> dict[str, Any]:
    _validate_surface_contract(
        jsonl_path,
        bbox_format=bbox_format,
        coordinate_surface=coordinate_surface,
    )
    sha256 = hashlib.sha256()
    row_count = 0
    object_count = 0
    included_perturbations = 0
    skipped_invalid_perturbations = 0
    losses: list[float] = []
    sampled_bboxes: list[tuple[int, int, int, int]] = []
    rng = random.Random(rng_seed)

    with jsonl_path.open("rb") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            sha256.update(raw_line)
            if not raw_line.strip():
                continue
            row_count += 1
            payload = json.loads(raw_line)
            _validate_record_bbox_format_metadata(
                payload,
                line_no=line_no,
                bbox_format=bbox_format,
            )
            for bbox in _extract_bboxes(payload, line_no=line_no):
                object_count += 1
                bbox_losses = one_token_iou_losses(bbox)
                losses.extend(bbox_losses)
                included_perturbations += len(bbox_losses)
                skipped_invalid_perturbations += 8 - len(bbox_losses)
                if target_audit_sample > 0:
                    _reservoir_add(sampled_bboxes, bbox, limit=target_audit_sample, seen=object_count, rng=rng)

    loss_summary = summarize_losses(losses)
    input_sha256 = sha256.hexdigest()
    return {
        "bbox_format": bbox_format,
        "coordinate_surface": coordinate_surface,
        "target_distribution": target_distribution,
        "input_path": str(jsonl_path),
        "input_path_resolved": str(jsonl_path.resolve()),
        "input_sha256": input_sha256,
        "train_jsonl": str(jsonl_path),
        "sha256": input_sha256,
        "row_count": row_count,
        "object_count": object_count,
        "included_perturbations": included_perturbations,
        "skipped_invalid_perturbations": skipped_invalid_perturbations,
        **loss_summary,
        "tau_data": loss_summary["median"],
        "target_audit_sample": len(sampled_bboxes),
        "target_audit_sample_requested": target_audit_sample,
        "target_audit_tau": tau,
        "rng_seed": rng_seed,
        "provenance": _build_provenance(
            jsonl_path,
            input_sha256=input_sha256,
            command=command,
            rng_seed=rng_seed,
            tau=tau,
            target_audit_sample_requested=target_audit_sample,
            bbox_format=bbox_format,
            coordinate_surface=coordinate_surface,
            target_distribution=target_distribution,
        ),
        "target_shape_audit": compute_target_shape_audit(
            sampled_bboxes,
            tau=tau,
            target_distribution=target_distribution,
        ),
    }


def _validate_surface_contract(
    jsonl_path: Path,
    *,
    bbox_format: str,
    coordinate_surface: str,
) -> None:
    if bbox_format != BBOX_FORMAT:
        raise ValueError(
            f"bbox_format must be {BBOX_FORMAT!r}; got {bbox_format!r}. "
            "IoU-Gibbs calibration only supports canonical xyxy bbox_2d."
        )
    if coordinate_surface != COORDINATE_SURFACE:
        raise ValueError(
            f"coordinate_surface must be {COORDINATE_SURFACE!r}; "
            f"got {coordinate_surface!r}"
        )
    path_text = str(jsonl_path).lower()
    noncanonical_markers = ("cxcy", "center_size", "logw", "logh")
    if any(marker in path_text for marker in noncanonical_markers):
        raise ValueError(
            "Refusing to calibrate IoU-Gibbs xyxy targets on a path that looks "
            f"non-canonical: {jsonl_path}"
        )


def _validate_record_bbox_format_metadata(
    payload: dict[str, Any],
    *,
    line_no: int,
    bbox_format: str,
) -> None:
    metadata = payload.get("metadata")
    prepared_format = (
        metadata.get("prepared_bbox_format") if isinstance(metadata, dict) else None
    )
    if bbox_format == BBOX_FORMAT and prepared_format not in (None, "", BBOX_FORMAT):
        raise ValueError(
            f"line {line_no}: bbox_format={BBOX_FORMAT} cannot calibrate an "
            f"offline-prepared non-canonical bbox branch ({prepared_format!r})"
        )


def _build_provenance(
    jsonl_path: Path,
    *,
    input_sha256: str,
    command: str | None,
    rng_seed: int,
    tau: float,
    target_audit_sample_requested: int,
    bbox_format: str,
    coordinate_surface: str,
    target_distribution: CoordSoftTargetDistributionName,
) -> dict[str, Any]:
    script_path = Path(__file__).resolve()
    return {
        "command": command,
        "cwd": str(Path.cwd()),
        "input_path": str(jsonl_path),
        "input_path_resolved": str(jsonl_path.resolve()),
        "input_sha256": input_sha256,
        "script_path": str(script_path),
        "script_sha256": _sha256_file(script_path),
        "python": sys.executable,
        "rng_seed": rng_seed,
        "tau": tau,
        "target_audit_sample_requested": target_audit_sample_requested,
        "bbox_format": bbox_format,
        "coordinate_surface": coordinate_surface,
        "target_distribution": target_distribution,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reservoir_add(
    sample: list[tuple[int, int, int, int]],
    bbox: tuple[int, int, int, int],
    *,
    limit: int,
    seen: int,
    rng: random.Random,
) -> None:
    if len(sample) < limit:
        sample.append(bbox)
        return
    replacement_index = rng.randrange(seen)
    if replacement_index < limit:
        sample[replacement_index] = bbox


def compute_target_shape_audit(
    bboxes: list[tuple[int, int, int, int]],
    *,
    tau: float = DEFAULT_TAU,
    target_distribution: CoordSoftTargetDistributionName = "iou_gibbs_v0",
) -> dict[str, Any]:
    min_side_deciles = _rank_decile_labels([min(bbox[2] - bbox[0], bbox[3] - bbox[1]) for bbox in bboxes])
    area_deciles = _rank_decile_labels([(bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) for bbox in bboxes])
    groups: dict[str, Any] = {
        "overall": _new_metric_accumulator(),
        "min_side_decile": defaultdict(_new_metric_accumulator),
        "area_decile": defaultdict(_new_metric_accumulator),
        "slot": defaultdict(_new_metric_accumulator),
        "boundary_flag": defaultdict(_new_metric_accumulator),
        "min_side<=32": defaultdict(_new_metric_accumulator),
        "min_side<=50": defaultdict(_new_metric_accumulator),
    }

    for bbox_index, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox
        min_side = min(x2 - x1, y2 - y1)
        boundary_flag = x1 == COORD_MIN or y1 == COORD_MIN or x2 == COORD_MAX or y2 == COORD_MAX
        base_labels = {
            "min_side_decile": min_side_deciles[bbox_index],
            "area_decile": area_deciles[bbox_index],
            "boundary_flag": str(boundary_flag).lower(),
            "min_side<=32": str(min_side <= 32).lower(),
            "min_side<=50": str(min_side <= 50).lower(),
        }
        for slot_index, slot_name in enumerate(SLOT_NAMES):
            stats = target_shape_stats_for_slot(
                bbox,
                slot_index,
                tau=tau,
                target_distribution=target_distribution,
            )
            _add_stats(groups["overall"], stats)
            _add_stats(groups["slot"][slot_name], stats)
            for group_name, group_label in base_labels.items():
                _add_stats(groups[group_name][group_label], stats)

    return _summarize_group_tree(groups)


def _rank_decile_labels(values: list[int]) -> list[str]:
    if not values:
        return []
    sorted_indices = sorted(range(len(values)), key=lambda index: (values[index], index))
    labels = ["D01"] * len(values)
    for rank, index in enumerate(sorted_indices):
        decile = min(9, int(rank * 10 / len(values))) + 1
        labels[index] = f"D{decile:02d}"
    return labels


def _new_metric_accumulator() -> dict[str, list[float]]:
    return {metric: [] for metric in TARGET_METRICS}


def _add_stats(accumulator: dict[str, list[float]], stats: dict[str, float | int]) -> None:
    for metric in TARGET_METRICS:
        accumulator[metric].append(float(stats[metric]))


def _summarize_group_tree(groups: dict[str, Any]) -> dict[str, Any]:
    summarized: dict[str, Any] = {}
    for group_name, group_value in groups.items():
        if group_name == "overall":
            summarized[group_name] = _summarize_metric_accumulator(group_value)
        else:
            summarized[group_name] = {
                label: _summarize_metric_accumulator(accumulator)
                for label, accumulator in sorted(group_value.items())
            }
    return summarized


def _summarize_metric_accumulator(accumulator: dict[str, list[float]]) -> dict[str, Any]:
    return {metric: summarize_values(values) for metric, values in accumulator.items()}


def write_markdown_report(output_path: Path, result: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    target_distribution = str(result.get("target_distribution", "iou_gibbs_v0"))
    body = [
        f"# Coord SoftCE {target_distribution} Tau v0",
        "",
        "Calibration and no-training target-shape audit for geometry-aware coordinate SoftCE.",
        "",
        "```json",
        json.dumps(result, indent=2, sort_keys=True),
        "```",
        "",
    ]
    output_path.write_text("\n".join(body), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", type=Path, required=True, help="Training JSONL to audit.")
    parser.add_argument("--output", type=Path, required=True, help="Markdown artifact output path.")
    parser.add_argument(
        "--bbox-format",
        choices=[BBOX_FORMAT],
        default=BBOX_FORMAT,
        help="Explicit bbox_2d chart contract. Only canonical xyxy is supported.",
    )
    parser.add_argument(
        "--coordinate-surface",
        choices=[COORDINATE_SURFACE],
        default=COORDINATE_SURFACE,
        help="Explicit coordinate-token surface contract.",
    )
    parser.add_argument("--rng-seed", type=int, default=0, help="Reservoir sampling seed.")
    parser.add_argument(
        "--target-distribution",
        choices=["iou_gibbs_v0", "ciou_gibbs_v0"],
        default="iou_gibbs_v0",
        help="Coordinate soft-target energy used for the target-shape audit.",
    )
    parser.add_argument(
        "--target-audit-sample",
        type=int,
        default=30000,
        help="Reservoir-sampled object count for target-shape audit.",
    )
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU, help="Tau used for target-shape audit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.target_audit_sample < 0:
        raise ValueError("--target-audit-sample must be >= 0")
    result = analyze_jsonl(
        args.jsonl,
        target_audit_sample=args.target_audit_sample,
        tau=args.tau,
        rng_seed=args.rng_seed,
        bbox_format=args.bbox_format,
        coordinate_surface=args.coordinate_surface,
        target_distribution=args.target_distribution,
        command=" ".join(shlex.quote(arg) for arg in sys.argv),
    )
    write_markdown_report(args.output, result)
    print(json.dumps({key: result[key] for key in ("target_distribution", "median", "mean", "object_count", "target_audit_sample")}))


if __name__ == "__main__":
    main()
