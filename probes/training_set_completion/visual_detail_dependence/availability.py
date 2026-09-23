"""CPU-only Lane B availability audit for the current COCO image supply."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

from PIL import Image

from .runtime import (
    ASPECT_TOLERANCE,
    BASELINE_MAX_PIXELS,
    GRID_FACTOR,
    HIGH_MAX_PIXELS,
    _binding,
    _fit_size,
)

DEFAULT_BASELINE_ROOT = Path("/data/CoordExp/public_data/coco/rescale_32_1024_bbox")
DEFAULT_ORIGINAL_ROOT = Path("/data/CoordExp/public_data/coco/raw")
DEFAULT_OUTPUT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-22-address-readout-pilot/selection/lane-b-availability.json"
)


def _directory_binding(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {"path": str(path), "kind": "directory", "exists": path.is_dir()}


def _reason_entry(reason: str, *, row: dict[str, Any], baseline: Path, original: Path) -> dict[str, Any]:
    return {
        "reason": reason,
        "image_id": row.get("image_id"),
        "split": row.get("metadata", {}).get("split"),
        "file_name": row.get("file_name"),
        "baseline_path": str(baseline),
        "original_path": str(original),
    }


def _audit_row(
    row: dict[str, Any],
    *,
    baseline_root: Path,
    original_root: Path,
) -> tuple[str, dict[str, Any]]:
    file_name = str(row.get("file_name", ""))
    baseline = (baseline_root / file_name).resolve()
    original = (original_root / file_name).resolve()
    if not file_name or Path(file_name).is_absolute():
        return "invalid_relative_file_name", _reason_entry("invalid_relative_file_name", row=row, baseline=baseline, original=original)
    split = str(row.get("metadata", {}).get("split", ""))
    expected_folder = {"train": "train2017", "val": "val2017"}.get(split)
    parts = Path(file_name).parts
    if expected_folder is None or len(parts) < 2 or parts[-2] != expected_folder:
        return "source_split_filename_mismatch", _reason_entry("source_split_filename_mismatch", row=row, baseline=baseline, original=original)
    if not baseline.is_file() or not original.is_file():
        return "source_image_missing", _reason_entry("source_image_missing", row=row, baseline=baseline, original=original)
    try:
        with Image.open(baseline) as baseline_image, Image.open(original) as original_image:
            baseline_size = tuple(int(x) for x in baseline_image.size)
            original_size = tuple(int(x) for x in original_image.size)
    except Exception as exc:  # noqa: BLE001
        detail = _reason_entry("image_decode_failure", row=row, baseline=baseline, original=original)
        detail["error"] = repr(exc)
        return "image_decode_failure", detail
    declared = (int(row.get("width", -1)), int(row.get("height", -1)))
    if declared != baseline_size:
        detail = _reason_entry("baseline_declared_dimension_mismatch", row=row, baseline=baseline, original=original)
        detail.update({"declared_baseline_size": list(declared), "decoded_baseline_size": list(baseline_size)})
        return "baseline_declared_dimension_mismatch", detail
    if Path(file_name).stem != f"{int(row.get('image_id', -1)):012d}":
        return "image_id_filename_mismatch", _reason_entry("image_id_filename_mismatch", row=row, baseline=baseline, original=original)
    if any(value % GRID_FACTOR for value in baseline_size):
        detail = _reason_entry("baseline_dimensions_not_grid_aligned", row=row, baseline=baseline, original=original)
        detail.update({"baseline_size": list(baseline_size), "original_size": list(original_size)})
        return "baseline_dimensions_not_grid_aligned", detail
    aspect_delta = abs(baseline_size[0] / baseline_size[1] - original_size[0] / original_size[1])
    high_size = _fit_size(*original_size, max_pixels=HIGH_MAX_PIXELS, factor=GRID_FACTOR)
    detail = _reason_entry("raw_original_not_larger_than_baseline", row=row, baseline=baseline, original=original)
    detail.update({
        "baseline_size": list(baseline_size),
        "original_size": list(original_size),
        "high_size": list(high_size),
        "baseline_pixels": baseline_size[0] * baseline_size[1],
        "original_pixels": original_size[0] * original_size[1],
        "high_pixels": high_size[0] * high_size[1],
        "aspect_delta": aspect_delta,
    })
    if aspect_delta > ASPECT_TOLERANCE:
        return "baseline_original_aspect_mismatch", {**detail, "reason": "baseline_original_aspect_mismatch"}
    if original_size[0] * original_size[1] <= baseline_size[0] * baseline_size[1]:
        return "raw_original_not_larger_than_baseline", detail
    if high_size[0] * high_size[1] <= baseline_size[0] * baseline_size[1]:
        return "high_grid_not_larger_than_baseline", {**detail, "reason": "high_grid_not_larger_than_baseline"}
    if high_size == baseline_size:
        return "high_grid_same_as_baseline", {**detail, "reason": "high_grid_same_as_baseline"}
    detail["reason"] = "eligible"
    return "eligible", detail


def audit_availability(
    *,
    baseline_root: Path = DEFAULT_BASELINE_ROOT,
    original_root: Path = DEFAULT_ORIGINAL_ROOT,
    output: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    baseline_root = baseline_root.resolve()
    original_root = original_root.resolve()
    split_paths = {
        "train": baseline_root / "train.jsonl",
        "val": baseline_root / "val.jsonl",
    }
    counts: Counter[str] = Counter()
    split_counts: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    identities: set[tuple[str, int]] = set()
    duplicates: list[list[Any]] = []
    total_records = 0
    for split, source_path in split_paths.items():
        with source_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                total_records += 1
                identity = (split, int(row.get("image_id", -1)))
                if identity in identities:
                    duplicates.append([split, identity[1], line_number])
                identities.add(identity)
                reason, detail = _audit_row(row, baseline_root=baseline_root, original_root=original_root)
                counts[reason] += 1
                split_counts[split][reason] += 1
                if len(examples[reason]) < 3:
                    examples[reason].append(detail)
    eligible = counts["eligible"]
    result = {
        "schema": "address_readout_pilot.lane_b.availability.v1",
        "status": "eligible_supply_available" if eligible else "HOLD",
        "lane": "visual_detail_dependence",
        "selection": {"requested_max_images": 32, "selection_by_outcome": False},
        "source": {
            "baseline_root": _directory_binding(baseline_root),
            "original_root": _directory_binding(original_root),
            "pipeline_manifest": _binding(baseline_root / "pipeline_manifest.json"),
            "split_lists": {split: _binding(path) for split, path in split_paths.items()},
            "mapping": "baseline_root/file_name to original_root/file_name; metadata split must agree with train2017/val2017; image_id must match 12-digit filename stem",
        },
        "frozen_rule": {
            "baseline": "exact decoded pre-rescaled image",
            "baseline_max_pixels": BASELINE_MAX_PIXELS,
            "high_max_pixels": HIGH_MAX_PIXELS,
            "grid_factor": GRID_FACTOR,
            "aspect_tolerance": ASPECT_TOLERANCE,
            "eligible": "decoded raw original pixels > baseline pixels, fitted high grid pixels > baseline pixels, high grid differs, dimensions grid aligned, aspect delta within tolerance",
            "no_image_bytes_hashed": True,
        },
        "counts": {
            "total_records": total_records,
            "unique_split_image_ids": len(identities),
            "duplicate_id_records": len(duplicates),
            "eligible": eligible,
            "excluded": total_records - eligible,
            "by_reason": dict(sorted(counts.items())),
            "by_split": {split: dict(sorted(values.items())) for split, values in sorted(split_counts.items())},
        },
        "representative_cases": {reason: values for reason, values in sorted(examples.items())},
        "duplicate_id_examples": duplicates[:3],
        "producer": _binding(Path(__file__)),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--original-root", type=Path, default=DEFAULT_ORIGINAL_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = audit_availability(
        baseline_root=args.baseline_root,
        original_root=args.original_root,
        output=args.output,
    )
    print(json.dumps(result["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
