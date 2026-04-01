from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.analysis.coco_lvis_missing_objects import (
    DEFAULT_COCO_ANNOTATION_PATHS,
    ProxyAugmentConfig,
    export_augmented_coco_with_lvis_proxies,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export augmented COCO coord JSONL with LVIS-derived proxy objects."
    )
    parser.add_argument(
        "--base-jsonl",
        type=Path,
        required=True,
        help="Base COCO coord JSONL to augment.",
    )
    parser.add_argument(
        "--projection-dir",
        type=Path,
        required=True,
        help="Directory containing recovered_coco80_instances.jsonl from projection analysis.",
    )
    parser.add_argument(
        "--determined-mapping-csv",
        type=Path,
        default=Path(
            "openspec/changes/add-lvis-coco-proxy-supervision/artifacts/determined_proxy_mappings_val2017.csv"
        ),
        help="Versioned semantic proxy mapping artifact used to rank/filter semantic mappings.",
    )
    parser.add_argument(
        "--raw-coco-annotation",
        type=Path,
        action="append",
        default=None,
        help="Raw COCO annotation JSON path(s) used to recover original image width/height for norm1000 encoding.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        required=True,
        help="Output augmented coord JSONL path.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional path for exporter summary JSON. Defaults to <output>.summary.json.",
    )
    parser.add_argument(
        "--exclude-plausible",
        action="store_true",
        help="Export strict proxies only.",
    )
    parser.add_argument(
        "--metadata-namespace",
        type=str,
        default="coordexp_proxy_supervision",
        help="Top-level metadata namespace for aligned proxy supervision entries.",
    )
    parser.add_argument("--strict-desc-ce-weight", type=float, default=1.0)
    parser.add_argument("--strict-coord-weight", type=float, default=1.0)
    parser.add_argument("--plausible-desc-ce-weight", type=float, default=0.25)
    parser.add_argument("--plausible-coord-weight", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = export_augmented_coco_with_lvis_proxies(
        base_jsonl_path=args.base_jsonl,
        projection_dir=args.projection_dir,
        determined_mapping_csv_path=args.determined_mapping_csv,
        output_jsonl_path=args.output_jsonl,
        raw_coco_annotation_paths=tuple(
            args.raw_coco_annotation or DEFAULT_COCO_ANNOTATION_PATHS
        ),
        config=ProxyAugmentConfig(
            include_plausible=not bool(args.exclude_plausible),
            metadata_namespace=str(args.metadata_namespace),
            strict_desc_ce_weight=float(args.strict_desc_ce_weight),
            strict_coord_weight=float(args.strict_coord_weight),
            plausible_desc_ce_weight=float(args.plausible_desc_ce_weight),
            plausible_coord_weight=float(args.plausible_coord_weight),
        ),
    )
    summary_path = args.summary_json or args.output_jsonl.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(result.summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result.summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
