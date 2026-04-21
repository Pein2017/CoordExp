"""Aggregation helpers for raw-text pre-burst margin probes."""

from __future__ import annotations

from statistics import mean
from typing import Sequence


def build_preburst_variants(
    *,
    prefix_objects: Sequence[dict[str, object]],
    source_object_index: int,
    gt_next: dict[str, object],
) -> list[dict[str, object]]:
    if source_object_index < 0 or source_object_index >= len(prefix_objects):
        raise IndexError("source_object_index out of range")

    def _clone_objects() -> list[dict[str, object]]:
        return [
            {
                "desc": str(obj.get("desc") or ""),
                "bbox_2d": [int(value) for value in list(obj["bbox_2d"])],
            }
            for obj in prefix_objects
        ]

    gt_bbox = [int(value) for value in list(gt_next["bbox_2d"])]
    baseline_objects = _clone_objects()
    baseline_duplicate = dict(baseline_objects[source_object_index])
    variants: list[dict[str, object]] = [
        {
            "variant_label": "baseline",
            "prefix_objects": baseline_objects,
            "duplicate_object": baseline_duplicate,
        },
        {
            "variant_label": "drop_source",
            "prefix_objects": [
                obj
                for idx, obj in enumerate(_clone_objects())
                if idx != int(source_object_index)
            ],
            "duplicate_object": baseline_duplicate,
        },
    ]
    x1y1_objects = _clone_objects()
    source_bbox = [int(value) for value in x1y1_objects[source_object_index]["bbox_2d"]]
    source_bbox[:2] = gt_bbox[:2]
    x1y1_objects[source_object_index]["bbox_2d"] = source_bbox
    variants.append(
        {
            "variant_label": "source_x1y1_from_gt_next",
            "prefix_objects": x1y1_objects,
            "duplicate_object": dict(x1y1_objects[source_object_index]),
        }
    )
    full_bbox_objects = _clone_objects()
    full_bbox_objects[source_object_index]["bbox_2d"] = list(gt_bbox)
    variants.append(
        {
            "variant_label": "source_bbox_from_gt_next",
            "prefix_objects": full_bbox_objects,
            "duplicate_object": dict(full_bbox_objects[source_object_index]),
        }
    )
    return variants


def summarize_preburst_margin_rows(
    rows: Sequence[dict[str, object]],
) -> dict[str, object]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["model_alias"]), str(row["variant_label"]))
        grouped.setdefault(key, []).append(dict(row))

    baseline_lookup: dict[tuple[str, str], dict[str, object]] = {}
    for row in rows:
        if str(row["variant_label"]) != "baseline":
            continue
        baseline_lookup[(str(row["model_alias"]), str(row["case_id"]))] = dict(row)

    variant_metrics: list[dict[str, object]] = []
    for (model_alias, variant_label), group_rows in sorted(grouped.items()):
        delta_sum_values = []
        delta_mean_values = []
        for row in group_rows:
            baseline = baseline_lookup.get((model_alias, str(row["case_id"])))
            if baseline is None:
                continue
            delta_sum_values.append(
                float(row["margin_sum_logprob"]) - float(baseline["margin_sum_logprob"])
            )
            delta_mean_values.append(
                float(row["margin_mean_logprob"])
                - float(baseline["margin_mean_logprob"])
            )
        variant_metrics.append(
            {
                "model_alias": model_alias,
                "variant_label": variant_label,
                "num_cases": len(group_rows),
                "mean_margin_sum_logprob": float(
                    mean(float(row["margin_sum_logprob"]) for row in group_rows)
                ),
                "mean_margin_mean_logprob": float(
                    mean(float(row["margin_mean_logprob"]) for row in group_rows)
                ),
                "positive_margin_sum_rate": float(
                    mean(
                        1.0 if float(row["margin_sum_logprob"]) > 0.0 else 0.0
                        for row in group_rows
                    )
                ),
                "positive_margin_mean_rate": float(
                    mean(
                        1.0 if float(row["margin_mean_logprob"]) > 0.0 else 0.0
                        for row in group_rows
                    )
                ),
                "mean_delta_from_baseline_sum_logprob": (
                    float(mean(delta_sum_values)) if delta_sum_values else None
                ),
                "mean_delta_from_baseline_mean_logprob": (
                    float(mean(delta_mean_values)) if delta_mean_values else None
                ),
            }
        )
    return {
        "num_case_variant_rows": len(rows),
        "variant_metrics": variant_metrics,
    }
