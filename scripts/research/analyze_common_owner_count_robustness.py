#!/usr/bin/env python3
"""Hold the retention denominator fixed across checkpoints and re-test the filter.

The Stage 2 result gave Sorted 88 greedy-missed union-recovered owners, Random
44, and Permutation 60. Sorted passed the pre-registered bar and Permutation
failed, so the obvious alternative explanation is composition: Sorted may pass
only because it has twice as many owners to spend its 5% retention budget on.

This subsamples every checkpoint's recovered-owner set to a common size, redoes
the leave-one-image-out selection using only the drawn subset as the retention
constraint, and measures rejection over the full catastrophic tail. Per-fold
feature mappers and scores do not depend on the drawn subset, so they are built
once and reused across draws.

It also reports panel annotation provenance, because the meaning of a
"catastrophic false positive" depends on the ground truth being reasonably
complete.

Research unit:
research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-29-likelihood-filter-robustness-and-panel-annotation-validity/unit.md
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import statistics
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "common_owner_count_robustness.v1"

CATASTROPHIC = ("catastrophic_class_absent", "catastrophic_misgrounded")
RETENTION_TARGET = 0.95
DRAW_SEED = 20260729

DEFAULT_PANEL = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration"
    "/2026-07-21-best-sampled-trajectory-positive-row-imitation-screen"
    "/evaluation-inputs/human-refined-12.coord.jsonl"
)


def _load_stage2_module() -> Any:
    path = Path(__file__).with_name("analyze_cluster_confidence_retention.py")
    spec = importlib.util.spec_from_file_location("stage2_retention", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Stage 2 analysis from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def panel_annotation_provenance(panel: Path) -> dict[str, Any]:
    """Split panel ground truth into COCO-original and human-added objects.

    The refinement pass marks human-added objects with a NEGATIVE
    `coco_ann_id`; presence of the key is not the discriminator.
    """

    original = added = 0
    per_image: dict[str, dict[str, int]] = {}
    for line in panel.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        image = str(record["images"][0]).rsplit("/", 1)[-1]
        counts = {"coco_original": 0, "human_added": 0}
        for obj in record["objects"]:
            if int(obj["coco_ann_id"]) < 0:
                counts["human_added"] += 1
                added += 1
            else:
                counts["coco_original"] += 1
                original += 1
        per_image[image] = counts
    total = original + added
    return {
        "panel": str(panel),
        "total_objects": total,
        "coco_original": original,
        "human_added": added,
        "human_added_fraction": added / total if total else None,
        "per_image": per_image,
    }


def robustness_for_family(
    module: Any,
    clusters: list[dict[str, Any]],
    extractors: list[Any],
    common_size: int,
    draws: int,
    targets: list[float],
) -> dict[str, Any]:
    images = sorted({c["image_id"] for c in clusters})
    folds = []
    for held_out in images:
        train = [c for c in clusters if c["image_id"] != held_out]
        test = [c for c in clusters if c["image_id"] == held_out]
        mappers = [
            module._percentile_mapper([v for v in (e(c) for c in train) if v is not None])
            for e in extractors
        ]
        folds.append(
            (train, module._score(train, extractors, mappers), test, module._score(test, extractors, mappers))
        )

    catastrophic_total = sum(1 for c in clusters if c["fp_subclass"] in CATASTROPHIC)
    recovered = [c["cluster_id"] for c in clusters if c["greedy_missed_union_recovered"]]
    pass_threshold = -(-catastrophic_total * 20 // 100)
    degenerate = len(recovered) <= common_size

    rng = random.Random(DRAW_SEED)
    rejections: list[int] = []
    for _ in range(draws):
        keep = set(
            recovered if degenerate else rng.sample(recovered, common_size)
        )
        best: int | None = None
        for target in targets:
            retained = denominator = rejected = 0
            for train, train_scores, test, test_scores in folds:
                pool = [
                    s
                    for s, c in zip(train_scores, train, strict=True)
                    if s is not None and c["cluster_id"] in keep
                ]
                if not pool:
                    continue
                threshold = module._threshold_for_retention(pool, target)
                for score, cluster in zip(test_scores, test, strict=True):
                    if score is None:
                        continue
                    if cluster["cluster_id"] in keep:
                        denominator += 1
                        retained += score >= threshold
                    if cluster["fp_subclass"] in CATASTROPHIC and score < threshold:
                        rejected += 1
            if denominator and retained / denominator >= RETENTION_TARGET:
                if best is None or rejected > best:
                    best = rejected
        rejections.append(best if best is not None else 0)

    fractions = sorted(r / catastrophic_total for r in rejections)
    return {
        "catastrophic_total": catastrophic_total,
        "pass_threshold": pass_threshold,
        "recovered_owner_clusters": len(recovered),
        "subsample_degenerate": degenerate,
        "draws": draws,
        "rejection_median": statistics.median(fractions),
        "rejection_p10": fractions[int(0.10 * len(fractions))],
        "rejection_p90": fractions[int(0.90 * len(fractions))],
        "draws_passing": sum(1 for r in rejections if r >= pass_threshold),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = Path(
        "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration"
        "/2026-07-29-three-checkpoint-human-refined12-max3084/likelihood-mining-v1"
    )
    parser.add_argument("--input-dir", type=Path, default=default_root)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--common-size", type=int, default=44)
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    module = _load_stage2_module()
    clusters_by_checkpoint = json.loads(
        (args.input_dir / "cluster-confidence.json").read_text(encoding="utf-8")
    )["clusters"]

    families = {
        "coordinate_only": [module._feature_coord],
        "support_plus_likelihood": [module._feature_support, module._feature_coord],
    }
    targets = [round(0.90 + 0.01 * step, 2) for step in range(11)]

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "common_owner_count": args.common_size,
        "retention_target": RETENTION_TARGET,
        "panel_annotation_provenance": panel_annotation_provenance(args.panel),
        "families": {},
    }
    for family, extractors in families.items():
        payload["families"][family] = {
            checkpoint: robustness_for_family(
                module, clusters, extractors, args.common_size, args.draws, targets
            )
            for checkpoint, clusters in clusters_by_checkpoint.items()
        }

    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(args.output.resolve())
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
