from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.coco_lvis_mapping_visualization import (  # noqa: E402
    SemanticVisConfig,
    render_usable_semantic_mapping_audit,
)
from src.analysis.coco_lvis_missing_objects import (  # noqa: E402
    DEFAULT_COCO_ANNOTATION_PATHS,
    DEFAULT_LVIS_ANNOTATION_PATHS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--projection-dir",
        type=Path,
        required=True,
        help="Directory containing learned_mapping.json and recovered_coco80_instances.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where semantic mapping review scenes will be written.",
    )
    parser.add_argument(
        "--lvis-category",
        action="append",
        default=None,
        help=(
            "Target LVIS usable-semantic category name to visualize. "
            "Repeatable. If omitted, the script auto-selects the top recovered usable-semantic mappings."
        ),
    )
    parser.add_argument(
        "--examples-per-mapping",
        type=int,
        default=2,
        help="Number of representative images to render for each selected LVIS category.",
    )
    parser.add_argument(
        "--auto-top-mappings",
        type=int,
        default=5,
        help="How many usable-semantic mappings to auto-select when --lvis-category is not provided.",
    )
    parser.add_argument(
        "--max-total-gt-objects",
        type=int,
        default=12,
        help="Skip scenes with more than this many LVIS target objects.",
    )
    parser.add_argument(
        "--max-total-pred-objects",
        type=int,
        default=12,
        help="Skip scenes with more than this many COCO mapped-class boxes.",
    )
    parser.add_argument(
        "--max-sibling-lvis-instances",
        type=int,
        default=1,
        help=(
            "Skip scenes with more than this many sibling LVIS categories "
            "that map to the same COCO class."
        ),
    )
    parser.add_argument(
        "--include-sibling-lvis-in-gt",
        action="store_true",
        help=(
            "Also draw accepted sibling LVIS categories in the GT panel for context. "
            "Default keeps GT focused on the target subtype only."
        ),
    )
    parser.add_argument(
        "--root-image-dir",
        type=Path,
        default=Path("public_data/coco/raw/images"),
        help="Root image directory passed to the shared GT-vs-Pred renderer.",
    )
    parser.add_argument(
        "--coco-annotation",
        type=Path,
        action="append",
        default=None,
        help="Optional COCO annotation JSON. Repeat to override defaults.",
    )
    parser.add_argument(
        "--lvis-annotation",
        type=Path,
        action="append",
        default=None,
        help="Optional LVIS annotation JSON. Repeat to override defaults.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = render_usable_semantic_mapping_audit(
        projection_dir=args.projection_dir,
        output_dir=args.output_dir,
        coco_annotation_paths=args.coco_annotation or DEFAULT_COCO_ANNOTATION_PATHS,
        lvis_annotation_paths=args.lvis_annotation or DEFAULT_LVIS_ANNOTATION_PATHS,
        explicit_lvis_category_names=tuple(args.lvis_category or ()),
        root_image_dir=args.root_image_dir,
        config=SemanticVisConfig(
            examples_per_mapping=int(args.examples_per_mapping),
            max_total_gt_objects=int(args.max_total_gt_objects),
            max_total_pred_objects=int(args.max_total_pred_objects),
            max_sibling_lvis_instances=int(args.max_sibling_lvis_instances),
            auto_top_mappings=int(args.auto_top_mappings),
            include_sibling_lvis_in_gt=bool(args.include_sibling_lvis_in_gt),
        ),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
