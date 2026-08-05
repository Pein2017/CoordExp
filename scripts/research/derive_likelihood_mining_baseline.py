#!/usr/bin/env python3
"""Re-derive the pre-registration baseline for the sampled span-likelihood mining unit.

This script owns the provenance of the numbers that define the unit's stop rule:
the canonical union aggregates, the false-positive taxonomy, and the
greedy-missed union-recovered owner counts. It is read-only over existing
artifacts and loads no model.

Research unit:
research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-29-sampled-span-likelihood-and-consensus-union-filtering/unit.md
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "likelihood_mining_baseline.v1"

CATASTROPHIC_IOU = 0.10
DUPLICATE_IOU = 0.50

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

# Declared in the unit; the recomputation must reproduce these exactly.
EXPECTED_UNION = {
    "sorted": {"tp": 191, "fn": 155, "fp": 476},
    "random": {"tp": 162, "fn": 184, "fp": 431},
    "permutation": {"tp": 169, "fn": 177, "fp": 423},
}


def _load_union_module() -> Any:
    path = Path(__file__).with_name("compute_sampled_union_f1_metrics.py")
    spec = importlib.util.spec_from_file_location("sampled_union_f1_metrics", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load canonical union matcher from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _greedy_matched_owners(
    manifest: dict[str, Any], image_by_row: dict[str, int]
) -> set[tuple[int, int]]:
    """Ground-truth owners matched by the checkpoint's native greedy decode."""

    return {
        (image_by_row[str(item["row_id"])], int(pair["gt_index"]))
        for item in manifest["items"]
        for pair in item["match"]["matched_pairs"]
    }


def _check_gt_index_alignment(
    manifest: dict[str, Any],
    image_by_row: dict[str, int],
    annotations: dict[int, list[dict[str, Any]]],
) -> int:
    """Greedy `gt_index` must address the same object as the COCO annotation order."""

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


def derive(
    *,
    checkpoint: str,
    rollout_root: Path,
    run_root: Path,
    module: Any,
) -> dict[str, Any]:
    run_dir = run_root / RUN_DIR_BY_CHECKPOINT[checkpoint]
    coco_gt_path = run_dir / "evaluation" / "detection" / "coco_gt.json"
    manifest_path = rollout_root / "visualization" / checkpoint / "manifest.json"
    sampled_paths = [
        rollout_root / checkpoint / "sampled" / f"shard-{index}.json" for index in (0, 1)
    ]

    # 1. Canonical aggregates must reproduce the stored artifact byte-identically.
    stored = json.loads((rollout_root / checkpoint / "f1-metrics.json").read_text(encoding="utf-8"))
    recomputed = module.compute(
        single_manifest=manifest_path.resolve(),
        single_metrics=(run_dir / "evaluation" / "detection" / "metrics.json").resolve(),
        sampled_paths=[path.resolve() for path in sampled_paths],
        coco_gt_path=coco_gt_path.resolve(),
        cluster_iou=0.50,
        match_iou=0.50,
    )
    identical = json.dumps(recomputed, sort_keys=True) == json.dumps(stored, sort_keys=True)
    union = recomputed["sampled_union"]
    expected = EXPECTED_UNION[checkpoint]
    aggregates_match = all(int(union[key]) == value for key, value in expected.items())

    coco_gt = json.loads(coco_gt_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    categories = {str(item["name"]): int(item["id"]) for item in coco_gt["categories"]}
    image_by_row = {str(item["row_id"]): int(item["id"]) for item in coco_gt["images"]}
    annotations: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in coco_gt["annotations"]:
        x, y, width, height = (float(value) for value in item["bbox"])
        annotations[int(item["image_id"])].append(
            {**item, "bbox_xyxy": [x, y, x + width, y + height]}
        )

    alignment_mismatches = _check_gt_index_alignment(manifest, image_by_row, annotations)
    greedy_owners = _greedy_matched_owners(manifest, image_by_row)

    # 2. Rebuild the same candidate rows the canonical matcher consumes.
    candidates: dict[int, list[dict[str, Any]]] = defaultdict(list)
    unknown_category_rows = 0
    for path in sampled_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for rollout in payload["rollouts"]:
            image_id = image_by_row[str(rollout["example_id"])]
            for row_index, prediction in enumerate(
                rollout["predictions"].get("predictions", [])
            ):
                category = module._normalize(prediction["description"])
                if category not in categories:
                    unknown_category_rows += 1
                candidates[image_id].append(
                    {
                        "category": category,
                        "category_id": categories.get(category),
                        "bbox": [float(value) for value in prediction["bbox"]],
                        "trajectory_id": f"seed-{rollout['seed']}",
                        "row_index": row_index,
                    }
                )

    union_owners: set[tuple[int, int]] = set()
    cluster_count = 0
    taxonomy = {
        "catastrophic_class_absent": 0,
        "catastrophic_misgrounded": 0,
        "ordinary_duplicate": 0,
        "ordinary_loose": 0,
    }
    for image_id in sorted(image_by_row.values()):
        clusters = module._complete_link_clusters(candidates[image_id], 0.50)
        predictions = []
        for cluster in clusters:
            medoid = module._medoid(cluster)
            predictions.append(
                {
                    "bbox": medoid["bbox"],
                    "category_id": medoid["category_id"],
                    "support": len({member["trajectory_id"] for member in cluster}),
                }
            )
        cluster_count += len(predictions)
        anns = annotations[image_id]

        # Same greedy one-to-one assignment as the canonical matcher, but retaining
        # which ground-truth owner and which cluster each match consumed.
        pairs = sorted(
            (
                (module._iou(prediction["bbox"], annotation["bbox_xyxy"]), gt_index, pred_index)
                for pred_index, prediction in enumerate(predictions)
                for gt_index, annotation in enumerate(anns)
                if prediction["category_id"] == annotation["category_id"]
                and module._iou(prediction["bbox"], annotation["bbox_xyxy"]) >= 0.50
            ),
            key=lambda item: (-item[0], item[1], item[2]),
        )
        used_gt: set[int] = set()
        used_pred: set[int] = set()
        for _, gt_index, pred_index in pairs:
            if gt_index in used_gt or pred_index in used_pred:
                continue
            used_gt.add(gt_index)
            used_pred.add(pred_index)
            union_owners.add((image_id, gt_index))

        for pred_index, prediction in enumerate(predictions):
            if pred_index in used_pred:
                continue
            same_class = [
                module._iou(prediction["bbox"], annotation["bbox_xyxy"])
                for annotation in anns
                if annotation["category_id"] == prediction["category_id"]
            ]
            if not same_class:
                taxonomy["catastrophic_class_absent"] += 1
            elif max(same_class) < CATASTROPHIC_IOU:
                taxonomy["catastrophic_misgrounded"] += 1
            elif max(same_class) >= DUPLICATE_IOU:
                taxonomy["ordinary_duplicate"] += 1
            else:
                taxonomy["ordinary_loose"] += 1

    catastrophic = (
        taxonomy["catastrophic_class_absent"] + taxonomy["catastrophic_misgrounded"]
    )
    false_positives = cluster_count - len(union_owners)
    recovered = union_owners - greedy_owners

    return {
        "checkpoint": checkpoint,
        "gate_canonical_json_identical": identical,
        "gate_declared_aggregates_match": aggregates_match,
        "gate_gt_index_alignment_mismatches": alignment_mismatches,
        "unknown_category_rows": unknown_category_rows,
        "ground_truth_owners": sum(len(value) for value in annotations.values()),
        "cluster_count": cluster_count,
        "union_tp": len(union_owners),
        "union_fp": false_positives,
        "greedy_tp": len(greedy_owners),
        "greedy_missed_union_recovered_owners": len(recovered),
        "false_positive_taxonomy": taxonomy,
        "catastrophic_total": catastrophic,
        # Stop rule: rejecting at least 20% of the catastrophic tail.
        "catastrophic_rejection_pass_threshold": -(-catastrophic * 20 // 100),
    }


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
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    module = _load_union_module()
    checkpoints = args.checkpoint or ["sorted", "random", "permutation"]
    rows = [
        derive(
            checkpoint=checkpoint,
            rollout_root=args.rollout_root,
            run_root=args.run_root,
            module=module,
        )
        for checkpoint in checkpoints
    ]

    failures = [
        row["checkpoint"]
        for row in rows
        if not row["gate_canonical_json_identical"]
        or not row["gate_declared_aggregates_match"]
        or row["gate_gt_index_alignment_mismatches"]
    ]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "rollout_root": str(args.rollout_root.resolve()),
        "catastrophic_iou_threshold": CATASTROPHIC_IOU,
        "duplicate_iou_threshold": DUPLICATE_IOU,
        "gates_passed": not failures,
        "failed_checkpoints": failures,
        "checkpoints": rows,
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(args.output.resolve())
    else:
        print(text, end="")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
