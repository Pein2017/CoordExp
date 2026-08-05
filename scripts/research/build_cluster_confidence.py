#!/usr/bin/env python3
"""Rebuild canonical union cluster membership, labels, and confidence features.

This script owns Lane C of
`research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-29-sampled-span-likelihood-and-consensus-union-filtering/unit.md`.

The stored evaluation artifacts (`f1-metrics.json` / `cluster_receipts`) record
only aggregate cluster counts; they do not say which spans belong to which
cluster or which ground-truth object a cluster matched. This script
reconstructs that membership deterministically from the same read-only source
artifacts the canonical union matcher used, reusing the *exact* clustering and
matching functions owned by `compute_sampled_union_f1_metrics.py` (imported by
path, not reimplemented), and extends the canonical counts to per-cluster and
per-span records.

It is read-only over every input artifact and loads no model. It writes three
files under `<rollout-root>/likelihood-mining-v1/` (or `--output-dir`):

- `cluster-confidence.json`: one record per canonical union cluster, with
  membership, canonical verdict, false-positive taxonomy, spatial dispersion,
  and an optional likelihood/percentile block.
- `span-labels.jsonl`: one row per object span, joinable on `object_span_id`,
  carrying per-span and cluster-inherited labels only (no likelihood values --
  those live in Lane B's `span-likelihood.jsonl` and are never recomputed
  here, per the frozen inter-lane contract).
- `cluster-confidence-receipt.json`: the exact pre-registered acceptance
  numbers, computed from this script's own reconstruction, with a top-level
  `acceptance_passed` boolean.

The acceptance numbers are fixed by pre-registration (see the unit doc above)
and are asserted here without any tuning. A mismatch means this script has a
bug; the correct response is to report the discrepancy, not to adjust
thresholds, ordering, or tie-breaking to force agreement.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = "cluster_confidence.v1"

CLUSTER_IOU_THRESHOLD = 0.50
MATCH_IOU_THRESHOLD = 0.50
CATASTROPHIC_IOU_THRESHOLD = 0.10
DUPLICATE_IOU_THRESHOLD = 0.50

DEFAULT_ROLLOUT_ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration"
    "/2026-07-29-three-checkpoint-human-refined12-max3084"
)
DEFAULT_RUN_ROOT = Path("/data/CoordExp/outputs/coordexp_swift/infer/val200")

RUN_DIR_BY_CHECKPOINT = {
    "sorted": "qwen3-vl-2b-desc-first-geo-sorted-step4887-human-refined12-hf-fp32",
    "random": "qwen3-vl-2b-desc-first-random-step4887-human-refined12-hf-fp32",
    "permutation": "qwen3-vl-2b-desc-first-random-permutation-bundle-step4887-human-refined12-hf-fp32",
}

# Fields carried in Lane B's `span-likelihood.jsonl` that populate the
# `likelihood` block's five aggregations (medoid, member_mean, member_min,
# member_max, member_std). Fixed by the frozen inter-lane contract.
LIKELIHOOD_FIELDS = (
    "coord_mean",
    "coord_min",
    "description_mean",
    "wrapper_mean",
    "row_logprob_mean",
)
# Primary features that additionally carry within-checkpoint / within-image
# percentile ranks.
PRIMARY_PERCENTILE_FIELDS = ("coord_mean", "coord_min", "description_mean")

# Pre-registered acceptance table (unit.md "Pre-Registration Baseline" and the
# task's exact acceptance table). The reconstruction must reproduce these
# numbers unit-for-unit; do not edit these constants to match a buggy run.
EXPECTED = {
    "sorted": {
        "clusters": 667,
        "tp": 191,
        "fp": 476,
        "catastrophic_class_absent": 13,
        "catastrophic_misgrounded": 184,
        "ordinary_duplicate": 65,
        "ordinary_loose": 214,
        "greedy_missed_union_recovered": 88,
        "span_count": 3570,
        "ground_truth_owners": 346,
        "greedy_tp": 107,
    },
    "random": {
        "clusters": 593,
        "tp": 162,
        "fp": 431,
        "catastrophic_class_absent": 4,
        "catastrophic_misgrounded": 127,
        "ordinary_duplicate": 70,
        "ordinary_loose": 230,
        "greedy_missed_union_recovered": 44,
        "span_count": 2852,
        "ground_truth_owners": 346,
        "greedy_tp": 132,
    },
    "permutation": {
        "clusters": 592,
        "tp": 169,
        "fp": 423,
        "catastrophic_class_absent": 2,
        "catastrophic_misgrounded": 136,
        "ordinary_duplicate": 67,
        "ordinary_loose": 218,
        "greedy_missed_union_recovered": 60,
        "span_count": 2918,
        "ground_truth_owners": 346,
        "greedy_tp": 121,
    },
}
TOTAL_EXPECTED_SPANS = 9340


def _load_union_module() -> Any:
    """Import the canonical union matcher by path and reuse its functions.

    `_normalize`, `_iou`, `_complete_link_clusters`, and `_medoid` are used
    exactly as defined there; they are the authority on clustering and
    matching semantics and are not reimplemented here.
    """

    path = Path(__file__).with_name("compute_sampled_union_f1_metrics.py")
    spec = importlib.util.spec_from_file_location("sampled_union_f1_metrics", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load canonical union matcher from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_categories_and_images(
    coco_gt: Mapping[str, Any]
) -> tuple[dict[str, int], dict[str, int], dict[int, str]]:
    categories = {str(item["name"]): int(item["id"]) for item in coco_gt["categories"]}
    image_by_row = {str(item["row_id"]): int(item["id"]) for item in coco_gt["images"]}
    row_by_image_id = {value: key for key, value in image_by_row.items()}
    return categories, image_by_row, row_by_image_id


def _group_annotations(coco_gt: Mapping[str, Any]) -> dict[int, list[dict[str, Any]]]:
    """Group COCO annotations by image, preserving on-disk order.

    Preserving order matters: the greedy manifest's `gt_index` addresses this
    same per-image list positionally (verified 0/346 mismatches per
    checkpoint by `_check_gt_index_alignment`), and the canonical union
    matcher builds its own per-image annotation lists the same way.
    """

    annotations: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in coco_gt["annotations"]:
        x, y, width, height = (float(value) for value in item["bbox"])
        annotations[int(item["image_id"])].append(
            {**item, "bbox_xyxy": [x, y, x + width, y + height]}
        )
    return annotations


def _check_gt_index_alignment(
    manifest: Mapping[str, Any],
    image_by_row: Mapping[str, int],
    annotations: Mapping[int, list[dict[str, Any]]],
) -> int:
    """Assert the greedy manifest's `gt_index` addresses the same object as
    the COCO annotation order. This has been verified upstream (0/346
    mismatches per checkpoint); this function re-derives that count rather
    than trusting the claim blindly.
    """

    mismatches = 0
    for item in manifest["items"]:
        image_id = image_by_row[str(item["row_id"])]
        gt_objects = item["gt_objects"]
        anns = annotations[image_id]
        if len(gt_objects) != len(anns):
            raise ValueError(
                f"greedy manifest and COCO ground truth disagree on object count "
                f"for {item['row_id']}: {len(gt_objects)} vs {len(anns)}"
            )
        for gt_object, annotation in zip(gt_objects, anns, strict=True):
            deltas = (
                abs(left - right)
                for left, right in zip(
                    gt_object["bbox_pixel_xyxy"], annotation["bbox_xyxy"], strict=True
                )
            )
            if max(deltas) > 1.5:
                mismatches += 1
    return mismatches


def _greedy_matched_owners(
    manifest: Mapping[str, Any], image_by_row: Mapping[str, int]
) -> set[tuple[int, int]]:
    """Ground-truth owners matched by the checkpoint's native greedy decode."""

    return {
        (image_by_row[str(item["row_id"])], int(pair["gt_index"]))
        for item in manifest["items"]
        for pair in item["match"]["matched_pairs"]
    }


def _build_candidates(
    sampled_paths: Sequence[Path],
    image_by_row: Mapping[str, int],
    categories: Mapping[str, int],
    module: Any,
) -> dict[int, list[dict[str, Any]]]:
    """Rebuild the exact candidate rows the canonical matcher consumes.

    Iteration order is shard order (paths given in order), then rollout order
    as stored in the shard, then row order (`enumerate` over the parsed
    predictions list). This matches
    `compute_sampled_union_f1_metrics.py::compute` exactly. `row_index` here
    is the enumerate index used for clustering tie-breaking -- it is *not*
    the same as the prediction's own `generated_order` field once any rows
    were dropped by the parser (verified: 36/1684 rows differ within one
    shard alone), so both are carried but only `row_index` feeds clustering.
    """

    candidates: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for path in sampled_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for rollout in payload["rollouts"]:
            seed = int(rollout["seed"])
            example_id = str(rollout["example_id"])
            image_id = image_by_row[example_id]
            predictions = rollout["predictions"].get("predictions", [])
            for row_index, prediction in enumerate(predictions):
                bbox_format = prediction.get("bbox_format")
                if bbox_format not in (None, "xyxy"):
                    raise ValueError(f"unexpected bbox_format {bbox_format!r} in {path}")
                category = module._normalize(prediction["description"])
                candidates[image_id].append(
                    {
                        "category": category,
                        "category_id": categories.get(category),
                        "bbox": [float(value) for value in prediction["bbox"]],
                        "trajectory_id": f"seed-{seed}",
                        "row_index": row_index,
                        "object_span_id": str(prediction["object_span_id"]),
                        "example_id": example_id,
                        "seed": seed,
                        "generated_order": int(prediction["generated_order"]),
                    }
                )
    return candidates


def _match_pairs(
    predictions: Sequence[Mapping[str, Any]],
    annotations: Sequence[Mapping[str, Any]],
    module: Any,
    threshold: float,
) -> dict[int, tuple[int, float]]:
    """Greedy one-to-one class-aware IoU match, same tie-break as the
    canonical matcher's `_match`, but returning per-prediction winner detail
    (`_match` only returns aggregate tp/fp/fn counts).
    """

    scored: list[tuple[float, int, int]] = []
    for pred_index, prediction in enumerate(predictions):
        for gt_index, annotation in enumerate(annotations):
            if prediction["category_id"] != annotation["category_id"]:
                continue
            value = module._iou(prediction["bbox"], annotation["bbox_xyxy"])
            if value >= threshold:
                scored.append((value, gt_index, pred_index))

    used_gt: set[int] = set()
    used_pred: set[int] = set()
    match_of_pred: dict[int, tuple[int, float]] = {}
    for value, gt_index, pred_index in sorted(scored, key=lambda item: (-item[0], item[1], item[2])):
        if gt_index in used_gt or pred_index in used_pred:
            continue
        used_gt.add(gt_index)
        used_pred.add(pred_index)
        match_of_pred[pred_index] = (gt_index, value)
    return match_of_pred


def _spatial_dispersion(
    cluster: Sequence[Mapping[str, Any]], medoid: Mapping[str, Any], module: Any
) -> dict[str, float]:
    """Mean pairwise member IoU, mean member-to-medoid IoU, and the standard
    deviation of member box centers in pixels.

    Design choices, since these three statistics are descriptive rather than
    acceptance-tested:
    - "mean member-to-medoid IoU" includes the medoid's IoU to itself (1.0),
      matching `_medoid`'s own convention of summing IoU over all cluster
      members including the candidate itself.
    - A singleton cluster has no pairs and zero dispersion by construction:
      mean pairwise IoU and mean medoid IoU are defined as 1.0 (perfect
      self-agreement) and center std is 0.0.
    - Center std is reported as one scalar: sqrt(Var(cx) + Var(cy)), the
      population root-mean-square distance of member centers from their
      centroid (equivalent to a 2D "spatial standard deviation").
    """

    member_count = len(cluster)
    if member_count == 1:
        return {"mean_pairwise_iou": 1.0, "mean_medoid_iou": 1.0, "center_std_pixels": 0.0}

    pairwise = [
        module._iou(cluster[i]["bbox"], cluster[j]["bbox"])
        for i in range(member_count)
        for j in range(i + 1, member_count)
    ]
    mean_pairwise_iou = sum(pairwise) / len(pairwise)
    mean_medoid_iou = sum(module._iou(member["bbox"], medoid["bbox"]) for member in cluster) / member_count

    centers = [
        ((member["bbox"][0] + member["bbox"][2]) / 2.0, (member["bbox"][1] + member["bbox"][3]) / 2.0)
        for member in cluster
    ]
    mean_cx = sum(c[0] for c in centers) / member_count
    mean_cy = sum(c[1] for c in centers) / member_count
    variance = sum((cx - mean_cx) ** 2 + (cy - mean_cy) ** 2 for cx, cy in centers) / member_count
    return {
        "mean_pairwise_iou": mean_pairwise_iou,
        "mean_medoid_iou": mean_medoid_iou,
        "center_std_pixels": math.sqrt(variance),
    }


def _load_span_likelihood(path: Path | None) -> dict[tuple[str, str], dict[str, Any]]:
    """Load Lane B's `span-likelihood.jsonl`, keyed by `(checkpoint,
    object_span_id)`.

    `object_span_id` alone is *not* globally unique across checkpoints: its
    format is `{example_id}:seed-{seed}:span-{n}`, and all three checkpoints
    replay the same 12 images with the same 16 seeds, so the same nominal id
    string recurs once per checkpoint with a different bbox/description
    (verified directly against the shards). Lane B's per-span contract
    (unit.md Stage 1) emits `checkpoint` as one of the required per-span
    fields, so joining on the pair avoids silently mixing likelihoods across
    checkpoints. A row missing `checkpoint` is treated as a contract
    violation and raises rather than being silently keyed ambiguously.
    """

    if path is None or not path.exists():
        return {}
    table: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "checkpoint" not in row or "object_span_id" not in row:
                raise ValueError(
                    f"{path}:{line_number}: span-likelihood row missing 'checkpoint' or "
                    "'object_span_id'; cannot join safely (object_span_id repeats across "
                    "checkpoints)"
                )
            key = (str(row["checkpoint"]), str(row["object_span_id"]))
            table[key] = row
    return table


def _likelihood_aggregate(rows: Sequence[Mapping[str, Any]], field: str, kind: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    if not values:
        return None
    if kind == "mean":
        return statistics.fmean(values)
    if kind == "min":
        return min(values)
    if kind == "max":
        return max(values)
    if kind == "std":
        return statistics.pstdev(values) if len(values) > 1 else 0.0
    raise ValueError(kind)


def _likelihood_block(
    cluster: Sequence[Mapping[str, Any]],
    medoid: Mapping[str, Any],
    checkpoint: str,
    span_likelihood: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any] | None:
    if not span_likelihood:
        return None
    medoid_row = span_likelihood.get((checkpoint, medoid["object_span_id"]))
    if medoid_row is None:
        return None
    member_rows = [
        span_likelihood[(checkpoint, member["object_span_id"])]
        for member in cluster
        if (checkpoint, member["object_span_id"]) in span_likelihood
    ]
    block: dict[str, Any] = {
        "medoid": {
            field: (float(medoid_row[field]) if medoid_row.get(field) is not None else None)
            for field in LIKELIHOOD_FIELDS
        }
    }
    for agg_name, kind in (
        ("member_mean", "mean"),
        ("member_min", "min"),
        ("member_max", "max"),
        ("member_std", "std"),
    ):
        block[agg_name] = {
            field: _likelihood_aggregate(member_rows, field, kind) for field in LIKELIHOOD_FIELDS
        }
    return block


def _percentile_ranks(values: Sequence[float]) -> list[float]:
    """Fractional percentile rank in [0, 1] with mid-rank tie handling.

    Rank 0.0 is the minimum value (or the sole value, when n == 1, gets 0.5),
    rank 1.0 is the maximum; ties share the average rank of their group.
    """

    n = len(values)
    if n == 0:
        return []
    if n == 1:
        return [0.5]
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        average_position = (i + j) / 2.0
        rank = average_position / (n - 1)
        for k in range(i, j + 1):
            ranks[order[k]] = rank
        i = j + 1
    return ranks


def _attach_percentiles(
    cluster_records: list[dict[str, Any]],
    checkpoint: str,
    span_likelihood: Mapping[tuple[str, str], Mapping[str, Any]],
) -> None:
    """Attach `percentiles` in place for every cluster record belonging to
    `checkpoint`. Null (whole block) when likelihoods are unavailable
    globally; per-field null when this cluster's own medoid value for that
    field is unavailable.
    """

    indices = [i for i, record in enumerate(cluster_records) if record["checkpoint"] == checkpoint]
    if not span_likelihood or not indices:
        for i in indices:
            cluster_records[i]["percentiles"] = None
        return

    for field in PRIMARY_PERCENTILE_FIELDS:
        # within_checkpoint scope
        scope_indices = [
            i
            for i in indices
            if cluster_records[i]["likelihood"] is not None
            and cluster_records[i]["likelihood"]["medoid"][field] is not None
        ]
        scope_values = [cluster_records[i]["likelihood"]["medoid"][field] for i in scope_indices]
        scope_ranks = _percentile_ranks(scope_values)
        for i, rank in zip(scope_indices, scope_ranks):
            record = cluster_records[i]
            record.setdefault("percentiles", {})
            record["percentiles"].setdefault(field, {})
            record["percentiles"][field]["within_checkpoint"] = rank

        # within_image scope, grouped by image_id within this checkpoint
        by_image: dict[int, list[int]] = defaultdict(list)
        for i in scope_indices:
            by_image[cluster_records[i]["image_id"]].append(i)
        for image_indices in by_image.values():
            image_values = [cluster_records[i]["likelihood"]["medoid"][field] for i in image_indices]
            image_ranks = _percentile_ranks(image_values)
            for i, rank in zip(image_indices, image_ranks):
                cluster_records[i]["percentiles"][field]["within_image"] = rank

    # Fill None for clusters/fields that never got a rank (no medoid likelihood).
    for i in indices:
        record = cluster_records[i]
        if record["likelihood"] is None:
            record["percentiles"] = None
            continue
        record.setdefault("percentiles", {})
        for field in PRIMARY_PERCENTILE_FIELDS:
            record["percentiles"].setdefault(field, {"within_checkpoint": None, "within_image": None})
            record["percentiles"][field].setdefault("within_checkpoint", None)
            record["percentiles"][field].setdefault("within_image", None)


def build_checkpoint(
    *,
    checkpoint: str,
    rollout_root: Path,
    run_root: Path,
    module: Any,
    span_likelihood: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Return (cluster_records, span_records, receipt_row) for one checkpoint."""

    run_dir = run_root / RUN_DIR_BY_CHECKPOINT[checkpoint]
    coco_gt_path = run_dir / "evaluation" / "detection" / "coco_gt.json"
    manifest_path = rollout_root / "visualization" / checkpoint / "manifest.json"
    sampled_paths = [rollout_root / checkpoint / "sampled" / f"shard-{index}.json" for index in (0, 1)]

    coco_gt = json.loads(coco_gt_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    categories, image_by_row, row_by_image_id = _load_categories_and_images(coco_gt)
    annotations = _group_annotations(coco_gt)

    alignment_mismatches = _check_gt_index_alignment(manifest, image_by_row, annotations)
    if alignment_mismatches:
        raise ValueError(
            f"{checkpoint}: greedy gt_index alignment failed with "
            f"{alignment_mismatches} mismatches against COCO annotation order"
        )
    greedy_owners = _greedy_matched_owners(manifest, image_by_row)

    candidates = _build_candidates(sampled_paths, image_by_row, categories, module)
    span_count = sum(len(items) for items in candidates.values())

    cluster_records: list[dict[str, Any]] = []
    span_records: list[dict[str, Any]] = []
    taxonomy = {
        "catastrophic_class_absent": 0,
        "catastrophic_misgrounded": 0,
        "ordinary_duplicate": 0,
        "ordinary_loose": 0,
    }
    union_tp = 0
    union_fp = 0
    recovered_count = 0

    for image_id in sorted(image_by_row.values()):
        clusters = module._complete_link_clusters(candidates[image_id], CLUSTER_IOU_THRESHOLD)
        anns = annotations[image_id]
        medoids = [module._medoid(cluster) for cluster in clusters]
        pred_rows = [{"bbox": medoid["bbox"], "category_id": medoid["category_id"]} for medoid in medoids]
        match_of_pred = _match_pairs(pred_rows, anns, module, MATCH_IOU_THRESHOLD)

        for cluster_index, cluster in enumerate(clusters):
            medoid = medoids[cluster_index]
            support = len({member["trajectory_id"] for member in cluster})
            member_span_ids = [member["object_span_id"] for member in cluster]

            same_class_ious = [
                module._iou(medoid["bbox"], annotation["bbox_xyxy"])
                for annotation in anns
                if annotation["category_id"] == medoid["category_id"]
            ]
            max_same_class_gt_iou = max(same_class_ious) if same_class_ious else 0.0

            if cluster_index in match_of_pred:
                gt_index, matched_iou = match_of_pred[cluster_index]
                canonical_verdict = "tp"
                fp_subclass = None
                recovered = (image_id, gt_index) not in greedy_owners
                union_tp += 1
                if recovered:
                    recovered_count += 1
            else:
                gt_index = None
                matched_iou = None
                canonical_verdict = "fp"
                recovered = False
                union_fp += 1
                if not same_class_ious:
                    fp_subclass = "catastrophic_class_absent"
                elif max_same_class_gt_iou < CATASTROPHIC_IOU_THRESHOLD:
                    fp_subclass = "catastrophic_misgrounded"
                elif max_same_class_gt_iou >= DUPLICATE_IOU_THRESHOLD:
                    fp_subclass = "ordinary_duplicate"
                else:
                    fp_subclass = "ordinary_loose"
                taxonomy[fp_subclass] += 1

            dispersion = _spatial_dispersion(cluster, medoid, module)
            likelihood = _likelihood_block(cluster, medoid, checkpoint, span_likelihood)

            cluster_id = f"{checkpoint}:{image_id}:{cluster_index}"
            record = {
                "cluster_id": cluster_id,
                "checkpoint": checkpoint,
                "example_id": row_by_image_id[image_id],
                "image_id": image_id,
                "category": medoid["category"],
                "category_id": medoid["category_id"],
                "member_count": len(cluster),
                "support": support,
                "member_span_ids": member_span_ids,
                "medoid_span_id": medoid["object_span_id"],
                "medoid_bbox": medoid["bbox"],
                "canonical_verdict": canonical_verdict,
                "matched_gt_index": gt_index,
                "matched_iou": matched_iou,
                "fp_subclass": fp_subclass,
                "max_same_class_gt_iou": max_same_class_gt_iou,
                "greedy_missed_union_recovered": recovered,
                "spatial_dispersion": dispersion,
                "likelihood": likelihood,
            }
            cluster_records.append(record)

            for member in cluster:
                same_class_indices = [
                    idx for idx, annotation in enumerate(anns) if annotation["category_id"] == member["category_id"]
                ]
                if same_class_indices:
                    same_class_ious_member = [
                        module._iou(member["bbox"], anns[idx]["bbox_xyxy"]) for idx in same_class_indices
                    ]
                    best_local = max(range(len(same_class_ious_member)), key=lambda k: same_class_ious_member[k])
                    per_span_best_iou = same_class_ious_member[best_local]
                    per_span_best_gt_index = same_class_indices[best_local]
                else:
                    per_span_best_iou = 0.0
                    per_span_best_gt_index = None
                per_span_label = "tp_independent" if per_span_best_iou >= MATCH_IOU_THRESHOLD else "fp_independent"

                span_records.append(
                    {
                        "checkpoint": checkpoint,
                        "example_id": member["example_id"],
                        "image_id": image_id,
                        "seed": member["seed"],
                        "object_span_id": member["object_span_id"],
                        "generated_order": member["generated_order"],
                        "cluster_id": cluster_id,
                        "is_medoid": member["object_span_id"] == medoid["object_span_id"],
                        "cluster_support": support,
                        "per_span_best_iou": per_span_best_iou,
                        "per_span_best_gt_index": per_span_best_gt_index,
                        "per_span_label": per_span_label,
                        "cluster_inherited_label": canonical_verdict,
                        "cluster_fp_subclass": fp_subclass,
                    }
                )

    _attach_percentiles(cluster_records, checkpoint, span_likelihood)

    receipt_row = {
        "checkpoint": checkpoint,
        "clusters": len(cluster_records),
        "tp": union_tp,
        "fp": union_fp,
        "catastrophic_class_absent": taxonomy["catastrophic_class_absent"],
        "catastrophic_misgrounded": taxonomy["catastrophic_misgrounded"],
        "ordinary_duplicate": taxonomy["ordinary_duplicate"],
        "ordinary_loose": taxonomy["ordinary_loose"],
        "greedy_missed_union_recovered": recovered_count,
        "span_count": span_count,
        "ground_truth_owners": sum(len(value) for value in annotations.values()),
        "greedy_tp": len(greedy_owners),
    }
    return cluster_records, span_records, receipt_row


def _check_acceptance(receipt_row: Mapping[str, Any]) -> list[str]:
    checkpoint = receipt_row["checkpoint"]
    expected = EXPECTED[checkpoint]
    failures = []
    for key, expected_value in expected.items():
        actual_value = receipt_row[key]
        if actual_value != expected_value:
            failures.append(
                f"{checkpoint}.{key}: expected {expected_value}, got {actual_value}"
            )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-root", type=Path, default=DEFAULT_ROLLOUT_ROOT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument(
        "--checkpoint",
        action="append",
        choices=sorted(RUN_DIR_BY_CHECKPOINT),
        help="restrict to one checkpoint; repeatable (default: all three)",
    )
    parser.add_argument(
        "--span-likelihood",
        type=Path,
        default=None,
        help="path to Lane B's span-likelihood.jsonl; optional, may not exist yet",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    module = _load_union_module()
    checkpoints = args.checkpoint or ["sorted", "random", "permutation"]
    span_likelihood = _load_span_likelihood(args.span_likelihood)

    output_dir = args.output_dir or (args.rollout_root / "likelihood-mining-v1")

    all_cluster_records: list[dict[str, Any]] = []
    all_span_records: list[dict[str, Any]] = []
    receipt_rows: list[dict[str, Any]] = []
    failures: list[str] = []

    for checkpoint in checkpoints:
        cluster_records, span_records, receipt_row = build_checkpoint(
            checkpoint=checkpoint,
            rollout_root=args.rollout_root,
            run_root=args.run_root,
            module=module,
            span_likelihood=span_likelihood,
        )
        all_cluster_records.extend(cluster_records)
        all_span_records.extend(span_records)
        receipt_rows.append(receipt_row)
        failures.extend(_check_acceptance(receipt_row))

    if set(checkpoints) == set(RUN_DIR_BY_CHECKPOINT):
        total_spans = sum(row["span_count"] for row in receipt_rows)
        if total_spans != TOTAL_EXPECTED_SPANS:
            failures.append(f"total span_count: expected {TOTAL_EXPECTED_SPANS}, got {total_spans}")

    all_span_records.sort(key=lambda row: (row["checkpoint"], row["example_id"], row["seed"], row["generated_order"]))

    acceptance_passed = not failures

    output_dir.mkdir(parents=True, exist_ok=True)

    cluster_confidence_payload = {
        "schema_version": SCHEMA_VERSION,
        "provenance": {
            "rollout_root": str(args.rollout_root.resolve()),
            "run_root": str(args.run_root.resolve()),
            "span_likelihood_path": str(args.span_likelihood.resolve()) if args.span_likelihood else None,
            "span_likelihood_available": bool(span_likelihood),
            "checkpoints": checkpoints,
            "cluster_iou_threshold": CLUSTER_IOU_THRESHOLD,
            "match_iou_threshold": MATCH_IOU_THRESHOLD,
            "catastrophic_iou_threshold": CATASTROPHIC_IOU_THRESHOLD,
            "duplicate_iou_threshold": DUPLICATE_IOU_THRESHOLD,
            "run_dir_by_checkpoint": {
                checkpoint: str((args.run_root / RUN_DIR_BY_CHECKPOINT[checkpoint]).resolve())
                for checkpoint in checkpoints
            },
        },
        "clusters": {
            checkpoint: [record for record in all_cluster_records if record["checkpoint"] == checkpoint]
            for checkpoint in checkpoints
        },
    }
    (output_dir / "cluster-confidence.json").write_text(
        json.dumps(cluster_confidence_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with (output_dir / "span-labels.jsonl").open("w", encoding="utf-8") as handle:
        for row in all_span_records:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    receipt_payload = {
        "schema_version": SCHEMA_VERSION,
        "acceptance_passed": acceptance_passed,
        "failures": failures,
        "expected": {checkpoint: EXPECTED[checkpoint] for checkpoint in checkpoints},
        "actual": {row["checkpoint"]: row for row in receipt_rows},
    }
    (output_dir / "cluster-confidence-receipt.json").write_text(
        json.dumps(receipt_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(json.dumps(receipt_payload, indent=2, sort_keys=True))
    print(f"cluster-confidence.json: {(output_dir / 'cluster-confidence.json').resolve()}")
    print(f"span-labels.jsonl: {(output_dir / 'span-labels.jsonl').resolve()} ({len(all_span_records)} rows)")
    print(f"cluster-confidence-receipt.json: {(output_dir / 'cluster-confidence-receipt.json').resolve()}")

    return 0 if acceptance_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
