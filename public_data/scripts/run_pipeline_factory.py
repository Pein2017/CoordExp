#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from public_data.pipeline import PipelineConfig, PipelinePlanner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified public-data pipeline/factory entrypoint")
    parser.add_argument("--mode", choices=["rescale", "coord", "validate", "full"], required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--preset", type=str, required=True)

    parser.add_argument("--max-objects", type=int, default=None)

    parser.add_argument("--image-factor", type=int, default=32)
    parser.add_argument("--max-pixels", type=int, default=32 * 32 * 768)
    parser.add_argument("--min-pixels", type=int, default=32 * 32 * 4)
    parser.add_argument("--num-workers", type=int, default=1)

    parser.add_argument("--assume-normalized", action="store_true")
    parser.add_argument("--compact", action="store_true")

    parser.add_argument("--run-validation-stage", action="store_true")
    parser.add_argument("--skip-image-check", action="store_true")
    parser.add_argument("--validate-raw", action="store_true")
    parser.add_argument("--validate-preset", action="store_true")

    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[pipeline][warn] Ignoring unsupported passthrough args: {' '.join(unknown)}")
    return args


def main() -> None:
    args = parse_args()

    if args.mode == "validate":
        validate_raw = bool(args.validate_raw)
        validate_preset = bool(args.validate_preset)
        if not validate_raw and not validate_preset:
            validate_raw = True
            validate_preset = True
    else:
        validate_raw = True
        validate_preset = True

    config = PipelineConfig(
        dataset_id=args.dataset_id,
        dataset_dir=args.dataset_dir,
        raw_dir=args.raw_dir,
        preset=args.preset,
        max_objects=args.max_objects,
        image_factor=args.image_factor,
        max_pixels=args.max_pixels,
        min_pixels=args.min_pixels,
        num_workers=args.num_workers,
        relative_images=True,
        assume_normalized=args.assume_normalized,
        compact_json=args.compact,
        skip_image_check=args.skip_image_check,
        run_validation_stage=args.run_validation_stage,
    )

    planner = PipelinePlanner()
    result = planner.run(
        config=config,
        mode=args.mode,
        validate_raw=validate_raw,
        validate_preset=validate_preset,
    )

    print(f"[pipeline] dataset={result.dataset_id}")
    print(f"[pipeline] preset={result.preset}")
    print(f"[pipeline] output_dir={result.preset_dir}")
    for split in sorted(result.split_artifacts.keys()):
        paths = result.split_artifacts[split]
        print(f"[pipeline] {split}.raw={paths.raw}")
        print(f"[pipeline] {split}.norm={paths.norm}")
        print(f"[pipeline] {split}.coord={paths.coord}")


if __name__ == "__main__":
    main()
