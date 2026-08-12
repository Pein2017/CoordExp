#!/usr/bin/env python3
"""Project Human-13 clean-greedy outputs onto the frozen owner ledger.

The analyzer deliberately repeats outcome-time duplicate exclusion and owner
matching.  It never consults selected native training rows: a newly generated
box earns final metric credit whenever it matches a frozen owner.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:  # pragma: no cover - exercised by CLI test
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.research.build_human13_k_union_manifest import (
    Human13KUnionManifest,
    ImageRecord,
    load_manifest,
)
from scripts.research.compare_clean_rollout_owner_coverage import (
    _global_matches,
    iou_xyxy,
)
from src.eval.detection_categories import normalize_coco_category_name


SCHEMA_VERSION = "human13_k_union_analysis.v1"
DEFAULT_FIXED_ROW_BUDGETS = (1, 2, 4, 8, 16, 32)
_CAP_STOP_REASONS = frozenset(
    {"cap", "cap_stop", "length", "length_truncated", "max_new_tokens"}
)
_NATURAL_STOP_REASONS = frozenset({"eos", "im_end", "natural_stop"})
_JSONL_CONTRACT = """raw output JSONL contract (one object per image/arm/milestone):
  required: image_id, arm_id, milestone,
            decode_mode='original_prompt_clean_greedy', repetition_penalty=1.0,
            predictions (or pred), generated_token_ids, stop_reason
  each prediction: generated_order (or row_index), description/category, bbox
  optional: malformed_row_count (or dropped_prediction_count), runtime
"""


def _integer(value: Any, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _finite_number(value: Any, field: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{field} must be finite and >= {minimum}")
    return result


def _box(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        box = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in box):
        return None
    x1, y1, x2, y2 = box
    if x1 >= x2 or y1 >= y2:
        return None
    return box


def _prediction_category(value: Mapping[str, Any]) -> str:
    return normalize_coco_category_name(
        value.get(
            "description",
            value.get("category", value.get("category_name", value.get("desc", ""))),
        )
    )


def _prediction_box(value: Mapping[str, Any]) -> tuple[float, float, float, float] | None:
    return _box(value.get("bbox", value.get("bbox_xyxy")))


def _ordered_predictions(output: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = output.get("predictions", output.get("pred", ()))
    if not isinstance(raw, list):
        raise ValueError("output predictions must be a list")
    ordered: list[dict[str, Any]] = []
    seen_orders: set[int] = set()
    for fallback_order, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise ValueError("every output prediction must be an object")
        order = _integer(
            item.get("generated_order", item.get("row_index", fallback_order)),
            "prediction.generated_order",
        )
        if order in seen_orders:
            raise ValueError(f"output contains duplicate generated_order {order}")
        seen_orders.add(order)
        ordered.append({**dict(item), "generated_order": order})
    ordered.sort(key=lambda item: int(item["generated_order"]))
    return ordered


def _match_prefix(
    image: ImageRecord,
    predictions: Sequence[Mapping[str, Any]],
    *,
    duplicate_iou_threshold: float,
    owner_iou_threshold: float,
) -> dict[str, Any]:
    retained: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []
    duplicate_rows: list[dict[str, Any]] = []
    for prediction in predictions:
        order = int(prediction["generated_order"])
        category = _prediction_category(prediction)
        bbox = _prediction_box(prediction)
        if not category or bbox is None:
            invalid_rows.append({"generated_order": order})
            continue
        duplicate = next(
            (
                (earlier, iou_xyxy(earlier["bbox"], bbox))
                for earlier in retained
                if iou_xyxy(earlier["bbox"], bbox) > duplicate_iou_threshold
            ),
            None,
        )
        if duplicate is not None:
            earlier, overlap = duplicate
            duplicate_rows.append(
                {
                    "generated_order": order,
                    "retained_generated_order": int(earlier["generated_order"]),
                    "prediction_to_prediction_iou": overlap,
                }
            )
            continue
        retained.append(
            {
                "generated_order": order,
                "category": category,
                "bbox": bbox,
            }
        )

    owners = sorted(
        image.owners, key=lambda item: (item.source_object_index, item.owner_id)
    )
    gt = [
        (normalize_coco_category_name(owner.category), owner.bbox) for owner in owners
    ]
    pred = [(item["category"], item["bbox"]) for item in retained]
    matches = _global_matches(gt, pred, owner_iou_threshold)
    owner_matches = {
        owners[owner_index].owner_id: {
            "owner_id": owners[owner_index].owner_id,
            "owner_iou": overlap,
            "generated_order": int(retained[prediction_index]["generated_order"]),
            "bbox": list(retained[prediction_index]["bbox"]),
        }
        for owner_index, prediction_index, overlap in matches
    }
    matched_prediction_indices = {prediction_index for _, prediction_index, _ in matches}
    unmatched_rows = [
        {"generated_order": int(item["generated_order"])}
        for index, item in enumerate(retained)
        if index not in matched_prediction_indices
    ]
    return {
        "owner_matches": owner_matches,
        "duplicate_rows": duplicate_rows,
        "unmatched_rows": unmatched_rows,
        "invalid_rows": invalid_rows,
    }


def _coverage(
    owner_matches: Mapping[str, Mapping[str, Any]], image: ImageRecord
) -> dict[str, Any]:
    owner_ids = sorted(owner_matches)
    g = set(image.g_owner_ids)
    h = set(image.h_owner_ids)
    m = set(image.m_owner_ids)
    return {
        "owner_ids": owner_ids,
        "owner_count": len(owner_ids),
        "source_owner_count": len(set(owner_ids) & g),
        "k_hit_owner_count": len(set(owner_ids) & h),
        "k_miss_owner_count": len(set(owner_ids) & m),
    }


def _runtime(value: Any) -> dict[str, float]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("output runtime must be an object")
    return {
        str(key): _finite_number(item, f"runtime.{key}")
        for key, item in sorted(value.items())
    }


def _analyze_one(
    *,
    image: ImageRecord,
    output: Mapping[str, Any],
    fixed_row_budgets: tuple[int, ...],
    duplicate_iou_threshold: float,
    owner_iou_threshold: float,
) -> dict[str, Any]:
    image_id = _integer(output.get("image_id"), "output.image_id")
    arm_id = output.get("arm_id")
    if not isinstance(arm_id, str) or not arm_id:
        raise ValueError("output.arm_id must be a non-empty string")
    milestone = _integer(output.get("milestone"), "output.milestone")
    if output.get("decode_mode") != "original_prompt_clean_greedy":
        raise ValueError("output must be an original-prompt clean-greedy decode")
    if _finite_number(
        output.get("repetition_penalty"), "output.repetition_penalty"
    ) != 1.0:
        raise ValueError("clean-greedy output repetition_penalty must equal 1.0")
    predictions = _ordered_predictions(output)
    malformed = _integer(
        output.get("malformed_row_count", output.get("dropped_prediction_count", 0)),
        "output.malformed_row_count",
    )
    token_ids = output.get("generated_token_ids")
    if not isinstance(token_ids, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in token_ids
    ):
        raise ValueError("output.generated_token_ids must be a list of integers")
    stop_reason = output.get("stop_reason")
    if not isinstance(stop_reason, str) or not stop_reason:
        raise ValueError("output.stop_reason must be a non-empty string")

    natural = _match_prefix(
        image,
        predictions,
        duplicate_iou_threshold=duplicate_iou_threshold,
        owner_iou_threshold=owner_iou_threshold,
    )
    final_owner_ids = set(natural["owner_matches"])
    g = set(image.g_owner_ids)
    h = set(image.h_owner_ids)
    m = set(image.m_owner_ids)
    burden = {
        "duplicate_rows": len(natural["duplicate_rows"]),
        "unmatched_rows": len(natural["unmatched_rows"]),
        "malformed_rows": malformed,
        "invalid_rows": len(natural["invalid_rows"]),
        "cap_stops": int(stop_reason in _CAP_STOP_REASONS),
    }
    fixed_coverage: dict[str, Any] = {}
    for budget in fixed_row_budgets:
        matched = _match_prefix(
            image,
            predictions[:budget],
            duplicate_iou_threshold=duplicate_iou_threshold,
            owner_iou_threshold=owner_iou_threshold,
        )
        fixed_coverage[str(budget)] = _coverage(matched["owner_matches"], image)

    return {
        "image_id": image_id,
        "arm_id": arm_id,
        "milestone": milestone,
        "k_hit_gained_owner_ids": sorted(final_owner_ids & h),
        "source_retained_owner_ids": sorted(final_owner_ids & g),
        "source_lost_owner_ids": sorted(g - final_owner_ids),
        "k_miss_incidental_gain_owner_ids": sorted(final_owner_ids & m),
        "final_unique_owner_ids": sorted(final_owner_ids),
        "prediction_row_count": len(predictions) + malformed,
        "generated_token_count": len(token_ids),
        "fixed_budget_coverage": fixed_coverage,
        "natural_stop_coverage": _coverage(natural["owner_matches"], image),
        "natural_stop_reached": stop_reason in _NATURAL_STOP_REASONS,
        "burden": burden,
        "duplicate_rows": natural["duplicate_rows"],
        "unmatched_rows": natural["unmatched_rows"],
        "invalid_rows": natural["invalid_rows"],
        "stop_reason": stop_reason,
        "runtime": _runtime(output.get("runtime")),
        "_owner_matches": natural["owner_matches"],
    }


def _add_source_comparison(
    record: dict[str, Any], source: Mapping[str, Any], image: ImageRecord
) -> None:
    output_matches = record["_owner_matches"]
    source_matches = source["_owner_matches"]
    common = sorted(set(output_matches) & set(source_matches))
    per_owner = [
        {
            "owner_id": owner_id,
            "source_iou": float(source_matches[owner_id]["owner_iou"]),
            "output_iou": float(output_matches[owner_id]["owner_iou"]),
            "iou_delta": float(output_matches[owner_id]["owner_iou"])
            - float(source_matches[owner_id]["owner_iou"]),
        }
        for owner_id in common
    ]
    if per_owner:
        source_mean = sum(item["source_iou"] for item in per_owner) / len(per_owner)
        output_mean = sum(item["output_iou"] for item in per_owner) / len(per_owner)
    else:
        source_mean = None
        output_mean = None
    record["common_owner_iou"] = {
        "owner_ids": common,
        "owner_count": len(common),
        "source_mean_iou": source_mean,
        "output_mean_iou": output_mean,
        "mean_iou_delta": (
            None if source_mean is None else output_mean - source_mean  # type: ignore[operator]
        ),
        "per_owner": per_owner,
    }
    source_burden = source["burden"]
    record["safe_in_panel_consolidation"] = bool(
        record["k_hit_gained_owner_ids"]
        and not record["source_lost_owner_ids"]
        and record["burden"]["malformed_rows"]
        <= source_burden["malformed_rows"]
        and record["burden"]["cap_stops"] <= source_burden["cap_stops"]
    )
    record["full_h_mastery"] = set(image.g_owner_ids) | set(
        image.h_owner_ids
    ) <= set(record["final_unique_owner_ids"])


def _sum_coverage(records: Sequence[Mapping[str, Any]], field: str) -> dict[str, Any]:
    owner_ids = [
        f"{record['image_id']}:{owner_id}"
        for record in records
        for owner_id in record[field]["owner_ids"]
    ]
    return {
        "owner_ids": owner_ids,
        "owner_count": sum(int(record[field]["owner_count"]) for record in records),
        "source_owner_count": sum(
            int(record[field]["source_owner_count"]) for record in records
        ),
        "k_hit_owner_count": sum(
            int(record[field]["k_hit_owner_count"]) for record in records
        ),
        "k_miss_owner_count": sum(
            int(record[field]["k_miss_owner_count"]) for record in records
        ),
    }


def _aggregate(
    records: Sequence[Mapping[str, Any]],
    *,
    panel: str,
    arm_id: str,
    milestone: int,
) -> dict[str, Any]:
    runtime: dict[str, float] = defaultdict(float)
    for record in records:
        for key, value in record["runtime"].items():
            runtime[str(key)] += float(value)
    common_per_owner = [
        {"image_id": record["image_id"], **item}
        for record in records
        for item in record["common_owner_iou"]["per_owner"]
    ]
    if common_per_owner:
        source_mean = sum(item["source_iou"] for item in common_per_owner) / len(
            common_per_owner
        )
        output_mean = sum(item["output_iou"] for item in common_per_owner) / len(
            common_per_owner
        )
    else:
        source_mean = None
        output_mean = None
    return {
        "panel": panel,
        "arm_id": arm_id,
        "milestone": milestone,
        "image_ids": [int(record["image_id"]) for record in records],
        "image_count": len(records),
        "k_hit_gained_count": sum(
            len(record["k_hit_gained_owner_ids"]) for record in records
        ),
        "source_retained_count": sum(
            len(record["source_retained_owner_ids"]) for record in records
        ),
        "source_lost_count": sum(
            len(record["source_lost_owner_ids"]) for record in records
        ),
        "k_miss_incidental_gain_count": sum(
            len(record["k_miss_incidental_gain_owner_ids"]) for record in records
        ),
        "final_unique_owner_count": sum(
            len(record["final_unique_owner_ids"]) for record in records
        ),
        "final_unique_owner_ids": [
            f"{record['image_id']}:{owner_id}"
            for record in records
            for owner_id in record["final_unique_owner_ids"]
        ],
        "prediction_row_count": sum(
            int(record["prediction_row_count"]) for record in records
        ),
        "generated_token_count": sum(
            int(record["generated_token_count"]) for record in records
        ),
        "fixed_budget_coverage": {},
        "natural_stop_coverage": _sum_coverage(records, "natural_stop_coverage"),
        "natural_stop_reached_image_count": sum(
            bool(record["natural_stop_reached"]) for record in records
        ),
        "burden": {
            key: sum(int(record["burden"][key]) for record in records)
            for key in (
                "duplicate_rows",
                "unmatched_rows",
                "malformed_rows",
                "invalid_rows",
                "cap_stops",
            )
        },
        "common_owner_iou": {
            "owner_count": len(common_per_owner),
            "source_mean_iou": source_mean,
            "output_mean_iou": output_mean,
            "mean_iou_delta": (
                None if source_mean is None else output_mean - source_mean  # type: ignore[operator]
            ),
            "per_owner": common_per_owner,
        },
        "runtime": dict(sorted(runtime.items())),
        "safe_in_panel_consolidation_image_count": sum(
            bool(record["safe_in_panel_consolidation"]) for record in records
        ),
        "full_h_mastery_image_count": sum(
            bool(record["full_h_mastery"]) for record in records
        ),
    }


def _aggregate_fixed_budgets(
    aggregate: dict[str, Any],
    records: Sequence[Mapping[str, Any]],
    fixed_row_budgets: tuple[int, ...],
) -> None:
    aggregate["fixed_budget_coverage"] = {
        str(budget): {
            "owner_ids": [
                f"{record['image_id']}:{owner_id}"
                for record in records
                for owner_id in record["fixed_budget_coverage"][str(budget)][
                    "owner_ids"
                ]
            ],
            "owner_count": sum(
                int(record["fixed_budget_coverage"][str(budget)]["owner_count"])
                for record in records
            ),
            "source_owner_count": sum(
                int(
                    record["fixed_budget_coverage"][str(budget)][
                        "source_owner_count"
                    ]
                )
                for record in records
            ),
            "k_hit_owner_count": sum(
                int(
                    record["fixed_budget_coverage"][str(budget)]["k_hit_owner_count"]
                )
                for record in records
            ),
            "k_miss_owner_count": sum(
                int(
                    record["fixed_budget_coverage"][str(budget)][
                        "k_miss_owner_count"
                    ]
                )
                for record in records
            ),
        }
        for budget in fixed_row_budgets
    }


def analyze_outputs(
    manifest: Human13KUnionManifest,
    outputs: Sequence[Mapping[str, Any]],
    *,
    fixed_row_budgets: Sequence[int] = DEFAULT_FIXED_ROW_BUDGETS,
) -> dict[str, Any]:
    """Analyze Source and arm raw outputs without using the native row bank."""

    budgets = tuple(
        sorted({_integer(item, "fixed_row_budget", minimum=1) for item in fixed_row_budgets})
    )
    if not budgets:
        raise ValueError("at least one fixed row budget is required")
    images = {image.image_id: image for image in manifest.images}
    if not images or len(images) != len(manifest.images):
        raise ValueError("manifest must contain unique image records")
    matcher = manifest.binding.matcher
    if (
        matcher.algorithm != "cardinality_first_max_total_iou"
        or not matcher.same_category
        or matcher.duplicate_comparison != "strictly_greater"
    ):
        raise ValueError("manifest matcher identity is not supported by this analyzer")

    records: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    for raw_output in outputs:
        if not isinstance(raw_output, Mapping):
            raise ValueError("every output must be an object")
        image_id = _integer(raw_output.get("image_id"), "output.image_id")
        if image_id not in images:
            raise ValueError(f"output image {image_id} is absent from the manifest")
        record = _analyze_one(
            image=images[image_id],
            output=raw_output,
            fixed_row_budgets=budgets,
            duplicate_iou_threshold=matcher.duplicate_iou_threshold,
            owner_iou_threshold=matcher.owner_iou_threshold,
        )
        identity = (record["arm_id"], record["milestone"], image_id)
        if identity in seen:
            raise ValueError(f"duplicate output identity {identity}")
        seen.add(identity)
        records.append(record)

    source_by_image: dict[int, dict[str, Any]] = {}
    for record in records:
        if record["arm_id"] != "frozen_source":
            continue
        if record["milestone"] != 0:
            raise ValueError("frozen_source outputs must use milestone zero")
        source_by_image[int(record["image_id"])] = record
    if set(source_by_image) != set(images):
        raise ValueError("outputs require exactly one frozen_source row per manifest image")

    group_images: dict[tuple[str, int], set[int]] = defaultdict(set)
    for record in records:
        group_images[(str(record["arm_id"]), int(record["milestone"]))].add(
            int(record["image_id"])
        )
    incomplete = {
        identity: sorted(set(images) - image_ids)
        for identity, image_ids in group_images.items()
        if image_ids != set(images)
    }
    if incomplete:
        raise ValueError(f"arm/milestone outputs do not cover the manifest: {incomplete}")

    for record in records:
        image_id = int(record["image_id"])
        _add_source_comparison(record, source_by_image[image_id], images[image_id])
    panel_order = {image.image_id: index for index, image in enumerate(manifest.images)}
    records.sort(
        key=lambda item: (
            0 if item["arm_id"] == "frozen_source" else 1,
            str(item["arm_id"]),
            int(item["milestone"]),
            panel_order[int(item["image_id"])],
        )
    )

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(str(record["arm_id"]), int(record["milestone"]))].append(record)
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "unit_id": manifest.binding.unit_id,
        "panel_sha256": manifest.binding.panel.panel_sha256,
        "fixed_row_budgets": list(budgets),
        "per_image": [],
        "legacy12": [],
        "image2299": [],
        "pooled": [],
    }
    for record in records:
        public = {key: value for key, value in record.items() if not key.startswith("_")}
        result["per_image"].append(public)
    for (arm_id, milestone), group in sorted(grouped.items()):
        for panel, selected in (
            ("legacy12", [item for item in group if item["image_id"] != 2299]),
            ("image2299", [item for item in group if item["image_id"] == 2299]),
            ("pooled", group),
        ):
            aggregate = _aggregate(
                selected,
                panel=panel,
                arm_id=arm_id,
                milestone=milestone,
            )
            _aggregate_fixed_budgets(aggregate, selected, budgets)
            result[panel].append(aggregate)
    return result


def load_outputs(path: str | Path) -> list[dict[str, Any]]:
    """Load one raw output object per JSONL line."""

    result: list[dict[str, Any]] = []
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{line_number} is not valid JSON") from exc
            if not isinstance(item, dict):
                raise ValueError(f"{source}:{line_number} must contain an object")
            result.append(item)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=_JSONL_CONTRACT,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--fixed-row-budget",
        type=int,
        action="append",
        dest="fixed_row_budgets",
        help="Positive parsed-row prefix budget; may be repeated.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite analysis: {args.output}")
    manifest = load_manifest(args.manifest, require_full_panel=True)
    result = analyze_outputs(
        manifest,
        load_outputs(args.outputs),
        fixed_row_budgets=(
            DEFAULT_FIXED_ROW_BUDGETS
            if args.fixed_row_budgets is None
            else args.fixed_row_budgets
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0


__all__ = ["analyze_outputs", "load_outputs", "main", "parse_args"]


if __name__ == "__main__":
    raise SystemExit(main())
