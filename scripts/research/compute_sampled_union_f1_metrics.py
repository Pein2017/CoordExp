#!/usr/bin/env python3
"""Compute scoreless single F1 and a deterministic sampled-union diagnostic."""

from __future__ import annotations

import argparse
import contextlib
import copy
import glob
import io
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCHEMA_VERSION = "sampled_union_f1_metrics.v1"


def _run_coco_bbox_eval(coco_gt_payload: dict[str, Any], predictions: list[dict[str, Any]]) -> dict[str, float]:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO()
    coco_gt.dataset = copy.deepcopy(coco_gt_payload)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(copy.deepcopy(predictions))
        coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
        coco_eval.params.imgIds = [int(image["id"]) for image in coco_gt_payload["images"]]
        coco_eval.params.catIds = [int(category["id"]) for category in coco_gt_payload["categories"]]
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
    stats = [float(value) for value in coco_eval.stats]
    return {"bbox_AP": stats[0], "bbox_AR100": stats[8]}


def _iou(left: Sequence[float], right: Sequence[float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    denominator = left_area + right_area - intersection
    return 0.0 if denominator <= 0.0 else intersection / denominator


def _normalize(value: Any) -> str:
    return " ".join(str(value).strip().lower().replace("_", " ").split())


def _micro(tp: int, fp: int, fn: int) -> dict[str, Any]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def _single_metrics(manifest: Mapping[str, Any]) -> dict[str, Any]:
    tp = sum(int(item["match"]["tp"]) for item in manifest["items"])
    fp = sum(int(item["match"]["fp"]) for item in manifest["items"])
    fn = sum(int(item["match"]["fn"]) for item in manifest["items"])
    return _micro(tp, fp, fn)


def _complete_link_clusters(candidates: list[dict[str, Any]], threshold: float) -> list[list[dict[str, Any]]]:
    clusters: list[list[dict[str, Any]]] = []
    for candidate in sorted(
        candidates,
        key=lambda item: (item["category"], tuple(item["bbox"]), item["trajectory_id"], item["row_index"]),
    ):
        compatible = [
            (min(_iou(candidate["bbox"], member["bbox"]) for member in cluster), index)
            for index, cluster in enumerate(clusters)
            if cluster[0]["category"] == candidate["category"]
            and all(_iou(candidate["bbox"], member["bbox"]) >= threshold for member in cluster)
        ]
        if compatible:
            _, index = max(compatible, key=lambda item: (item[0], -item[1]))
            clusters[index].append(candidate)
        else:
            clusters.append([candidate])
    return clusters


def _medoid(cluster: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        cluster,
        key=lambda candidate: (
            sum(_iou(candidate["bbox"], other["bbox"]) for other in cluster),
            -candidate["row_index"],
            candidate["trajectory_id"],
        ),
    )


def _match(predictions: list[dict[str, Any]], annotations: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    candidates = []
    for pred_index, prediction in enumerate(predictions):
        for gt_index, annotation in enumerate(annotations):
            value = _iou(prediction["bbox"], annotation["bbox_xyxy"])
            if prediction["category_id"] == annotation["category_id"] and value >= threshold:
                candidates.append((value, gt_index, pred_index))
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    for _, gt_index, pred_index in sorted(candidates, key=lambda item: (-item[0], item[1], item[2])):
        if gt_index in used_gt or pred_index in used_pred:
            continue
        used_gt.add(gt_index)
        used_pred.add(pred_index)
    return _micro(len(used_gt), len(predictions) - len(used_pred), len(annotations) - len(used_gt))


def compute(
    *,
    single_manifest: Path,
    single_metrics: Path,
    sampled_paths: Iterable[Path],
    coco_gt_path: Path,
    cluster_iou: float,
    match_iou: float,
) -> dict[str, Any]:
    manifest = json.loads(single_manifest.read_text(encoding="utf-8"))
    official_single = json.loads(single_metrics.read_text(encoding="utf-8"))
    coco_gt = json.loads(coco_gt_path.read_text(encoding="utf-8"))
    categories = {str(item["name"]): int(item["id"]) for item in coco_gt["categories"]}
    image_by_row = {str(item["row_id"]): int(item["id"]) for item in coco_gt["images"]}
    row_by_image_id = {value: key for key, value in image_by_row.items()}
    annotations: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in coco_gt["annotations"]:
        x, y, width, height = (float(value) for value in item["bbox"])
        annotations[int(item["image_id"])].append(
            {**item, "bbox_xyxy": [x, y, x + width, y + height]}
        )

    candidates: dict[int, list[dict[str, Any]]] = defaultdict(list)
    seeds: set[int] = set()
    source_paths = sorted(set(sampled_paths))
    for path in source_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for rollout in payload["rollouts"]:
            seed = int(rollout["seed"])
            seeds.add(seed)
            row_id = str(rollout["example_id"])
            image_id = image_by_row[row_id]
            for row_index, prediction in enumerate(rollout["predictions"].get("predictions", [])):
                category = _normalize(prediction["description"])
                candidates[image_id].append(
                    {
                        "category": category,
                        "category_id": categories.get(category),
                        "bbox": [float(value) for value in prediction["bbox"]],
                        "trajectory_id": f"seed-{seed}",
                        "row_index": row_index,
                    }
                )
    if not seeds:
        raise ValueError("sampled panel has no seeds")

    coco_predictions: list[dict[str, Any]] = []
    union_by_image: dict[int, list[dict[str, Any]]] = {}
    cluster_receipts: dict[str, list[dict[str, Any]]] = {}
    for image_id in sorted(image_by_row.values()):
        clusters = _complete_link_clusters(candidates[image_id], cluster_iou)
        union_predictions: list[dict[str, Any]] = []
        receipts: list[dict[str, Any]] = []
        for cluster_index, cluster in enumerate(clusters):
            representative = _medoid(cluster)
            support = len({item["trajectory_id"] for item in cluster})
            score = support / len(seeds) + 1e-9 / (cluster_index + 1)
            prediction = {
                "category": representative["category"],
                "category_id": representative["category_id"],
                "bbox": representative["bbox"],
                "score": score,
            }
            union_predictions.append(prediction)
            x1, y1, x2, y2 = prediction["bbox"]
            if prediction["category_id"] is not None:
                coco_predictions.append(
                    {
                        "image_id": image_id,
                        "category_id": prediction["category_id"],
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": score,
                    }
                )
            receipts.append(
                {
                    "category": prediction["category"],
                    "representative_bbox_xyxy": prediction["bbox"],
                    "member_count": len(cluster),
                    "trajectory_support": support,
                    "score": score,
                }
            )
        union_by_image[image_id] = union_predictions
        cluster_receipts[row_by_image_id[image_id]] = receipts

    per_image = {
        row_by_image_id[image_id]: _match(union_by_image[image_id], annotations[image_id], match_iou)
        for image_id in sorted(union_by_image)
    }
    union_micro = _micro(
        sum(item["tp"] for item in per_image.values()),
        sum(item["fp"] for item in per_image.values()),
        sum(item["fn"] for item in per_image.values()),
    )
    coco_metrics = _run_coco_bbox_eval(coco_gt, coco_predictions)
    return {
        "schema_version": SCHEMA_VERSION,
        "claim_boundary": (
            "Diagnostic only. Single F1 is the shared renderer's scoreless class-aware F1@0.50. "
            "K-sample union uses deterministic class-aware complete-link IoU clustering; cluster medoids "
            "are detections and distinct-seed support is their COCO score."
        ),
        "cluster_iou_threshold": cluster_iou,
        "match_iou_threshold": match_iou,
        "sampled_seed_count": len(seeds),
        "sampled_seeds": sorted(seeds),
        "single": {
            **_single_metrics(manifest),
            "mRecall": float(official_single["mRecall"]),
            "manifest": str(single_manifest),
            "official_metrics": str(single_metrics),
        },
        "sampled_union": {
            **union_micro,
            "mRecall": float(coco_metrics["bbox_AR100"]),
            "mAP": float(coco_metrics["bbox_AP"]),
            "prediction_cluster_count": sum(len(items) for items in union_by_image.values()),
            "coco_eligible_prediction_cluster_count": len(coco_predictions),
            "unknown_category_prediction_cluster_count": sum(
                prediction["category_id"] is None
                for predictions in union_by_image.values()
                for prediction in predictions
            ),
            "per_image": per_image,
            "cluster_receipts": cluster_receipts,
            "source_artifacts": [str(path) for path in source_paths],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-manifest", required=True, type=Path)
    parser.add_argument("--single-metrics", required=True, type=Path)
    parser.add_argument("--sampled-artifact", required=True, action="append")
    parser.add_argument("--coco-gt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cluster-iou", type=float, default=0.50)
    parser.add_argument("--match-iou", type=float, default=0.50)
    args = parser.parse_args()
    sampled_paths = sorted(
        {Path(item).resolve() for pattern in args.sampled_artifact for item in glob.glob(pattern)}
    )
    result = compute(
        single_manifest=args.single_manifest.resolve(),
        single_metrics=args.single_metrics.resolve(),
        sampled_paths=sampled_paths,
        coco_gt_path=args.coco_gt.resolve(),
        cluster_iou=args.cluster_iou,
        match_iou=args.match_iou,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
