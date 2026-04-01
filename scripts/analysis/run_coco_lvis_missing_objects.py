from __future__ import annotations

import argparse
from dataclasses import replace
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.coco_lvis_missing_objects import (  # noqa: E402
    DEFAULT_COCO_ANNOTATION_PATHS,
    DEFAULT_LVIS_ANNOTATION_PATHS,
    AnalysisConfig,
    run_coco_lvis_projection_analysis,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where summary tables and unmatched-instance artifacts will be written.",
    )
    parser.add_argument(
        "--coco-annotation",
        type=Path,
        action="append",
        default=None,
        help=(
            "COCO instances JSON path. Repeat to add multiple splits. "
            "Defaults to train2017 + val2017."
        ),
    )
    parser.add_argument(
        "--lvis-annotation",
        type=Path,
        action="append",
        default=None,
        help=(
            "LVIS instances JSON path. Repeat to add multiple splits. "
            "Defaults to lvis_v1_train + lvis_v1_val."
        ),
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="Legacy flag alias for the recovery same-class IoU threshold.",
    )
    parser.add_argument(
        "--mapping-mode",
        choices=("strict", "expanded"),
        default="strict",
        help=(
            "Category-link policy. 'strict' keeps exact/synonym/manual-alias matches only; "
            "'expanded' adds curated many-to-one broad mappings such as sports-ball subclasses."
        ),
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional cap on the number of shared images to process after filtering.",
    )
    parser.add_argument(
        "--include-crowd",
        action="store_true",
        help="Include iscrowd annotations instead of skipping them.",
    )
    parser.add_argument(
        "--coco-image-split",
        action="append",
        choices=("train2017", "val2017"),
        default=None,
        help=(
            "Restrict analysis to shared images whose underlying COCO image folder matches "
            "this split. Repeatable."
        ),
    )
    parser.add_argument(
        "--evidence-pair-iou-threshold",
        type=float,
        default=0.3,
        help="Minimum IoU required for an LVIS/COCO pair to contribute mapping evidence.",
    )
    parser.add_argument(
        "--recovery-max-conflicting-coco-iou",
        type=float,
        default=0.5,
        help=(
            "Maximum IoU allowed against a COCO annotation from a different class when "
            "emitting a recovered candidate."
        ),
    )
    parser.add_argument(
        "--recovery-min-lvis-box-area",
        type=float,
        default=0.0,
        help="Minimum LVIS box area required for recovery/evidence eligibility.",
    )
    parser.add_argument(
        "--strict-min-match-count",
        type=int,
        default=None,
        help="Optional override for strict-tier minimum matched-pair support.",
    )
    parser.add_argument(
        "--strict-min-precision-like",
        type=float,
        default=None,
        help="Optional override for strict-tier minimum precision_like.",
    )
    parser.add_argument(
        "--strict-min-coverage-like",
        type=float,
        default=None,
        help="Optional override for strict-tier minimum coverage_like.",
    )
    parser.add_argument(
        "--usable-min-match-count",
        type=int,
        default=None,
        help="Optional override for usable-tier minimum matched-pair support.",
    )
    parser.add_argument(
        "--usable-min-precision-like",
        type=float,
        default=None,
        help="Optional override for usable-tier minimum precision_like.",
    )
    parser.add_argument(
        "--usable-min-coverage-like",
        type=float,
        default=None,
        help="Optional override for usable-tier minimum coverage_like.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AnalysisConfig(
        iou_threshold=float(args.iou_threshold),
        ignore_crowd=not bool(args.include_crowd),
        mapping_mode=str(args.mapping_mode),
        max_images=args.max_images,
        allowed_coco_image_splits=tuple(args.coco_image_split or ()),
        evidence_pair_iou_threshold=float(args.evidence_pair_iou_threshold),
        recovery_iou_threshold=float(args.iou_threshold),
        recovery_min_lvis_box_area=float(args.recovery_min_lvis_box_area),
        recovery_max_conflicting_coco_iou=float(
            args.recovery_max_conflicting_coco_iou
        ),
    )
    if (
        args.strict_min_match_count is not None
        or args.strict_min_precision_like is not None
        or args.strict_min_coverage_like is not None
    ):
        config = replace(
            config,
            strict_exact_thresholds=replace(
                config.strict_exact_thresholds,
                min_match_count=(
                    args.strict_min_match_count
                    if args.strict_min_match_count is not None
                    else config.strict_exact_thresholds.min_match_count
                ),
                min_precision_like=(
                    args.strict_min_precision_like
                    if args.strict_min_precision_like is not None
                    else config.strict_exact_thresholds.min_precision_like
                ),
                min_coverage_like=(
                    args.strict_min_coverage_like
                    if args.strict_min_coverage_like is not None
                    else config.strict_exact_thresholds.min_coverage_like
                ),
            ),
            strict_semantic_thresholds=replace(
                config.strict_semantic_thresholds,
                min_match_count=(
                    args.strict_min_match_count
                    if args.strict_min_match_count is not None
                    else config.strict_semantic_thresholds.min_match_count
                ),
                min_precision_like=(
                    args.strict_min_precision_like
                    if args.strict_min_precision_like is not None
                    else config.strict_semantic_thresholds.min_precision_like
                ),
                min_coverage_like=(
                    args.strict_min_coverage_like
                    if args.strict_min_coverage_like is not None
                    else config.strict_semantic_thresholds.min_coverage_like
                ),
            ),
        )
    if (
        args.usable_min_match_count is not None
        or args.usable_min_precision_like is not None
        or args.usable_min_coverage_like is not None
    ):
        config = replace(
            config,
            usable_exact_thresholds=replace(
                config.usable_exact_thresholds,
                min_match_count=(
                    args.usable_min_match_count
                    if args.usable_min_match_count is not None
                    else config.usable_exact_thresholds.min_match_count
                ),
                min_precision_like=(
                    args.usable_min_precision_like
                    if args.usable_min_precision_like is not None
                    else config.usable_exact_thresholds.min_precision_like
                ),
                min_coverage_like=(
                    args.usable_min_coverage_like
                    if args.usable_min_coverage_like is not None
                    else config.usable_exact_thresholds.min_coverage_like
                ),
            ),
            usable_semantic_thresholds=replace(
                config.usable_semantic_thresholds,
                min_match_count=(
                    args.usable_min_match_count
                    if args.usable_min_match_count is not None
                    else config.usable_semantic_thresholds.min_match_count
                ),
                min_precision_like=(
                    args.usable_min_precision_like
                    if args.usable_min_precision_like is not None
                    else config.usable_semantic_thresholds.min_precision_like
                ),
                min_coverage_like=(
                    args.usable_min_coverage_like
                    if args.usable_min_coverage_like is not None
                    else config.usable_semantic_thresholds.min_coverage_like
                ),
            ),
        )
    summary = run_coco_lvis_projection_analysis(
        output_dir=args.output_dir,
        coco_annotation_paths=args.coco_annotation or DEFAULT_COCO_ANNOTATION_PATHS,
        lvis_annotation_paths=args.lvis_annotation or DEFAULT_LVIS_ANNOTATION_PATHS,
        config=config,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
