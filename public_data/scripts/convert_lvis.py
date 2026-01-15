#!/usr/bin/env python3
"""
Main script to convert LVIS annotations to Qwen3-VL JSONL format.

This is the primary entry point for LVIS conversion.
"""

import argparse
import json
import os
import sys
from dataclasses import asdict
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Dict, MutableMapping, cast

from public_data.converters.lvis_converter import ConversionConfig, LVISConverter
from public_data.converters.sorting import canonicalize_poly, sort_objects_tlbr
from src.datasets.preprocessors.resize import SmartResizeParams, SmartResizePreprocessor


def _relativize_images(row: MutableMapping[str, Any], base_dir: Path) -> Dict[str, Any]:
    """Relativize image paths from absolute to be relative to the output JSONL."""
    images = row.get("images") or []
    rel_images = []
    base_dir_resolved = base_dir.resolve()

    for img in images:
        p = Path(str(img))

        # If already relative, check if it needs normalization
        if not p.is_absolute():
            # Already relative - keep as is if it looks valid
            rel_images.append(str(p))
            continue

        # Try to make it relative to base_dir
        try:
            rel_images.append(str(p.resolve().relative_to(base_dir_resolved)))
        except ValueError:
            # Path is not under base_dir - try to construct a relative path
            # This can happen if paths point to raw images outside the output directory
            # Extract just the filename and put it under images/
            rel_images.append(f"images/{p.name}")

    # Convert to dict to allow modification and ensure return type
    result = dict(row)
    result["images"] = rel_images
    return result


def _presort_objects_in_row(row: MutableMapping[str, Any]) -> Dict[str, Any]:
    """Canonicalize polygon vertices and sort objects (TLBR) in a JSONL row.

    This is a post-smart-resize step: resizing/clamping can change which vertex is
    "top-most then left-most" due to rounding, so we enforce canonical ordering here.
    """
    objects = row.get("objects") or []
    if not isinstance(objects, list) or not objects:
        return dict(row)

    processed = []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        if obj.get("poly") is not None:
            poly = obj.get("poly")
            if isinstance(poly, list):
                try:
                    canon = canonicalize_poly(poly)
                    canon_ints = [
                        int(round(v)) if isinstance(v, (int, float)) else v for v in canon
                    ]
                    updated = dict(obj)
                    updated["poly"] = canon_ints
                    if "poly_points" in updated:
                        updated["poly_points"] = len(canon_ints) // 2
                    processed.append(updated)
                    continue
                except Exception:
                    # If canonicalization fails, keep the original polygon.
                    processed.append(obj)
                    continue
        processed.append(obj)

    updated_row = dict(row)
    updated_row["objects"] = sort_objects_tlbr(processed)
    return updated_row


def _process_row_worker(args_tuple):
    """Worker function for parallel processing of rows."""
    (
        row,
        params_dict,
        jsonl_dir,
        output_dir,
        relative_output_root,
        relative_image_paths,
        output_jsonl_parent,
    ) = args_tuple

    # Recreate preprocessor in worker (each worker has its own instance)
    # params_dict is already a dict from asdict(), so we can pass it directly
    params = SmartResizeParams(**params_dict)
    pre = SmartResizePreprocessor(
        params=params,
        jsonl_dir=Path(jsonl_dir),
        output_dir=Path(output_dir),
        write_images=True,
        relative_output_root=Path(relative_output_root)
        if relative_output_root
        else None,
    )

    updated = pre.preprocess(row) or row
    updated = _presort_objects_in_row(cast(MutableMapping[str, Any], updated))
    if relative_image_paths:
        updated = _relativize_images(
            cast(MutableMapping[str, Any], updated), Path(output_jsonl_parent)
        )
    return json.dumps(updated, ensure_ascii=False)


def _run_smart_resize(
    *,
    input_jsonl: Path,
    output_jsonl: Path,
    images_output_dir: Path,
    params: SmartResizeParams,
    relative_image_paths: bool,
    num_workers: int = 1,
) -> None:
    """Apply shared smart resize to JSONL + images on disk."""
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    images_output_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    # Read all rows
    rows = []
    with input_jsonl.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    # Determine actual number of workers
    if num_workers <= 0:
        num_workers = 1
    max_workers = min(num_workers, cpu_count(), len(rows))

    if max_workers == 1:
        # Sequential processing
        pre = SmartResizePreprocessor(
            params=params,
            jsonl_dir=input_jsonl.parent,
            output_dir=images_output_dir,
            write_images=True,
            relative_output_root=output_jsonl.parent,
        )

        with output_jsonl.open("w", encoding="utf-8") as fout:
            row_iterator = rows
            if tqdm is not None:
                row_iterator = tqdm(rows, desc="Resizing images", unit="sample")

            for row in row_iterator:
                updated = pre.preprocess(row) or row
                updated = _presort_objects_in_row(
                    cast(MutableMapping[str, Any], updated)
                )
                if relative_image_paths:
                    updated = _relativize_images(
                        cast(MutableMapping[str, Any], updated), output_jsonl.parent
                    )
                fout.write(json.dumps(updated, ensure_ascii=False) + "\n")
    else:
        # Parallel processing
        if tqdm is None:
            print(f"🚀 Processing {len(rows)} samples with {max_workers} workers...")

        # Prepare params dict for serialization (convert dataclass to dict)
        params_dict = asdict(params)

        # Prepare arguments for workers
        worker_args = [
            (
                row,
                params_dict,
                str(input_jsonl.parent),
                str(images_output_dir),
                str(output_jsonl.parent),
                relative_image_paths,
                str(output_jsonl.parent),
            )
            for row in rows
        ]

        # Process in parallel
        with Pool(processes=max_workers) as pool:
            if tqdm is not None:
                results = list(
                    tqdm(
                        pool.imap(_process_row_worker, worker_args),
                        total=len(worker_args),
                        desc=f"Resizing images ({max_workers} workers)",
                        unit="sample",
                    )
                )
            else:
                results = pool.map(_process_row_worker, worker_args)

        # Write results in order
        with output_jsonl.open("w", encoding="utf-8") as fout:
            for result in results:
                fout.write(result + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Convert LVIS dataset to Qwen3-VL JSONL format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
LVIS (Large Vocabulary Instance Segmentation):
- 1203 categories with long-tail distribution
- Uses COCO 2017 images (you need to download COCO images)
- 100K train images, 19.8K val images

Expected directory structure after download:
  ./lvis/raw/
    annotations/
      lvis_v1_train.json
      lvis_v1_val.json
    images/
      train2017/
        000000000001.jpg
        ...
      val2017/
        000000000001.jpg
        ...

Output JSONL format (Qwen3-VL):
  {
    "images": ["relative/path/to/image.jpg"],
    "objects": [
      {"bbox_2d": [x1, y1, x2, y2], "desc": "category_name"}
    ],
    "width": 640,
    "height": 480
  }

Examples:

  # Convert train split
  python convert_lvis.py --split train
  
  # Convert val split
  python convert_lvis.py --split val
  
  # Test with first 100 samples
  python convert_lvis.py --split train --max_samples 100
  
  # Custom paths
  python convert_lvis.py \\
    --annotation /path/to/lvis_v1_train.json \\
    --image_root /path/to/train2017 \\
    --output /path/to/output.jsonl
  
  # Skip clipping boxes to image bounds
  python convert_lvis.py --split train --no-clip-boxes
        """,
    )

    # Path arguments
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val"],
        default="train",
        help="Dataset split to convert (default: train)",
    )

    parser.add_argument(
        "--annotation",
        type=str,
        help="Path to LVIS JSON annotation file (overrides --split)",
    )

    parser.add_argument(
        "--image_root",
        type=str,
        help="Root directory containing split folders (train2017/val2017)",
    )

    parser.add_argument(
        "--output", type=str, help="Output JSONL path (overrides --split)"
    )

    parser.add_argument(
        "--base_dir",
        type=str,
        help="Base directory for default paths (defaults to the repo's public_data folder)",
    )

    # Conversion options
    parser.add_argument(
        "--max_samples", type=int, help="Limit number of samples (for testing)"
    )

    parser.add_argument(
        "--min_box_area",
        type=float,
        default=1.0,
        help="Minimum bbox area in pixels (default: 1.0)",
    )

    parser.add_argument(
        "--min_box_dimension",
        type=float,
        default=1.0,
        help="Minimum bbox width/height in pixels (default: 1.0)",
    )

    parser.add_argument(
        "--no-clip-boxes",
        action="store_true",
        help="Don't clip boxes to image boundaries",
    )

    parser.add_argument(
        "--keep-crowd", action="store_true", help="Keep crowd annotations (iscrowd=1)"
    )

    parser.add_argument(
        "--absolute-paths",
        action="store_true",
        help="Use absolute image paths instead of relative",
    )

    parser.add_argument(
        "--use-polygon",
        action="store_true",
        help="Convert segmentation polygons to poly (arbitrary N-point polygons)",
    )
    parser.add_argument(
        "--poly-max-points",
        type=int,
        default=12,
        help="Convert polygons with more than N vertices to bbox_2d during conversion (default: 12)",
    )

    # Smart-resize options
    parser.add_argument(
        "--smart-resize",
        action="store_true",
        help="Apply shared smart resize (pixel cap + grid align) to images and geometry",
    )
    parser.add_argument(
        "--max_pixels",
        type=int,
        default=SmartResizeParams().max_pixels,
        help="Maximum pixel budget after resize",
    )
    parser.add_argument(
        "--image_factor",
        type=int,
        default=SmartResizeParams().image_factor,
        help="Grid factor to snap dimensions to (e.g., 32)",
    )
    parser.add_argument(
        "--min_pixels",
        type=int,
        default=SmartResizeParams().min_pixels,
        help="Minimum pixel budget after resize",
    )
    parser.add_argument(
        "--resize_output_root",
        type=str,
        help="Root directory for resized outputs (images + JSONL). Defaults to public_data/lvis/resized_<factor>_<blocks>.",
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: convert 10 samples and validate output",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of parallel workers for multiprocessing (1=sequential, >1=parallel). Default: 1",
    )

    args = parser.parse_args()

    # Test mode overrides
    if args.test:
        if not args.max_samples:
            args.max_samples = 10
        print("\n⚠ TEST MODE: Converting only 10 samples for validation\n")

    # Resolve paths
    if args.base_dir:
        base_dir = os.path.abspath(args.base_dir)
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.abspath(os.path.join(script_dir, os.pardir))

    if args.annotation:
        annotation_path = os.path.abspath(args.annotation)
    else:
        annotation_path = os.path.join(
            base_dir, "lvis", "raw", "annotations", f"lvis_v1_{args.split}.json"
        )

    split_suffix = f"{args.split}2017"

    if args.image_root:
        image_root = os.path.abspath(args.image_root)
    else:
        image_root = os.path.join(base_dir, "lvis", "raw", "images")

    # If user points directly to train2017/val2017, normalize to the parent folder
    image_root_norm = os.path.normpath(image_root)
    if os.path.basename(image_root_norm) == split_suffix:
        print(
            f"  Detected image_root already includes '{split_suffix}'. Using its parent for consistency."
        )
        image_root = os.path.dirname(image_root_norm)

    split_image_dir = os.path.join(image_root, split_suffix)

    max_blocks = args.max_pixels // max(1, args.image_factor * args.image_factor)

    if args.smart_resize:
        if args.resize_output_root:
            resize_root = Path(args.resize_output_root).resolve()
        else:
            resize_root = (
                Path(base_dir) / "lvis" / f"resized_{args.image_factor}_{max_blocks}"
            )
        resize_root.mkdir(parents=True, exist_ok=True)
        images_output_dir = resize_root / "images"
        if args.output:
            output_path = Path(args.output).resolve()
        else:
            output_path = resize_root / f"{args.split}.jsonl"
        raw_output_path = resize_root / f"{args.split}_unscaled.jsonl"
    else:
        images_output_dir = None
        raw_output_path = None
        if args.output:
            output_path = os.path.abspath(args.output)
        else:
            output_path = os.path.join(
                base_dir, "lvis", "processed", f"{args.split}.jsonl"
            )

    # Validate inputs
    if not os.path.exists(annotation_path):
        print(f"✗ Error: Annotation file not found: {annotation_path}")
        print("\nDid you run download_lvis.py first?")
        print("  python scripts/download_lvis.py")
        sys.exit(1)

    if not os.path.exists(image_root):
        print(f"✗ Error: Image root directory not found: {image_root}")
        print("\nLVIS uses COCO 2017 images. Download them with:")
        print("  python scripts/download_lvis.py")
        sys.exit(1)

    if not os.path.exists(split_image_dir):
        print(f"✗ Error: Split directory not found: {split_image_dir}")
        print("\nExpected structure:")
        print(f"  {image_root}/")
        print("    train2017/")
        print("    val2017/")
        sys.exit(1)

    # Create config
    convert_output = raw_output_path if raw_output_path else Path(output_path)
    config = ConversionConfig(
        input_path=annotation_path,
        output_path=str(convert_output),
        image_root=image_root,
        split=args.split,
        max_samples=args.max_samples,
        min_box_area=args.min_box_area,
        min_box_dimension=args.min_box_dimension,
        clip_boxes=not args.no_clip_boxes,
        skip_crowd=not args.keep_crowd,
        relative_image_paths=not args.absolute_paths,
    )

    use_polygon = args.use_polygon or args.test  # Test mode always tries polygon
    poly_max_points = None
    if args.poly_max_points is not None:
        if args.poly_max_points <= 0:
            raise ValueError("--poly-max-points must be a positive integer")
        poly_max_points = int(args.poly_max_points)
    # Note: poly_max_points defaults to 12 in argparse, so it will be set unless explicitly disabled

    print("=" * 60)
    print("LVIS to Qwen3-VL JSONL Converter")
    print("=" * 60)
    print(f"  Mode: {'TEST (polygon enabled)' if args.test else 'PRODUCTION'}")
    print(f"  Split: {args.split}")
    print(f"  Annotation: {annotation_path}")
    print(f"  Images: {image_root}")
    print(f"  Output: {output_path}")
    if args.smart_resize:
        print(
            f"  Smart-resize: ENABLED (factor={args.image_factor}, max_pixels={args.max_pixels})"
        )
        print(f"  Resized images dir: {images_output_dir}")
        print(f"  Raw JSONL (pre-resize): {convert_output}")
    if args.max_samples:
        print(f"  Max samples: {args.max_samples}")
    if use_polygon:
        print("  Polygon mode: ENABLED (N-point polygons → poly)")
        if poly_max_points is not None:
            print(f"  Polygon cap: >{poly_max_points} points → bbox_2d")
    if args.num_workers > 1:
        print(f"  Multiprocessing: ENABLED ({args.num_workers} workers)")
    print("=" * 60 + "\n")

    # Convert
    try:
        converter = LVISConverter(
            config,
            use_polygon=use_polygon,
            poly_max_points=poly_max_points,
        )
        converter.convert()

        final_output = Path(output_path)
        if args.smart_resize:
            if images_output_dir is None:
                raise ValueError(
                    "images_output_dir must be set when smart_resize is enabled"
                )
            params = SmartResizeParams(
                max_pixels=int(args.max_pixels),
                image_factor=int(args.image_factor),
                min_pixels=int(args.min_pixels),
            )
            _run_smart_resize(
                input_jsonl=convert_output
                if isinstance(convert_output, Path)
                else Path(str(convert_output)),
                output_jsonl=final_output,
                images_output_dir=images_output_dir,
                params=params,
                relative_image_paths=not args.absolute_paths,
                num_workers=args.num_workers,
            )
            # Clean temp JSONL when present
            if raw_output_path and Path(raw_output_path).exists():
                try:
                    Path(raw_output_path).unlink()
                except OSError:
                    pass

        # Test mode: validate output
        if args.test:
            print("\n" + "=" * 60)
            print("Running Test Validation...")
            print("=" * 60)
            _validate_output(str(final_output), use_polygon)

        print("\n" + "=" * 60)
        print("✓ Conversion Complete!")
        print("=" * 60)
        print(f"\nOutput: {final_output}")
        print("\nNext steps:")
        print("  1. Validate output:")
        print(f"     python scripts/validate_jsonl.py {final_output}")
        print("  2. Create sampled subset:")
        print("     python scripts/sample_dataset.py \\")
        print(f"       --input {final_output} \\")
        print(
            f"       --output {base_dir}/lvis/processed/samples/{args.split}_5k_stratified.jsonl \\"
        )
        print("       --num_samples 5000 \\")
        print("       --strategy stratified")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n✗ Error during conversion: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


def _validate_output(jsonl_path: str, has_polygon: bool) -> None:
    """Quick validation of output format."""
    print("\n  Checking JSONL format...")

    errors = []
    samples_checked = 0
    bbox_count = 0
    poly_count = 0

    with open(jsonl_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            try:
                sample = json.load(json.loads(line))  # Double parse for safety
                sample = json.loads(line)
            except Exception:
                errors.append(f"Line {line_num}: Invalid JSON")
                continue

            samples_checked += 1

            # Check required fields
            for field in ["images", "objects", "width", "height"]:
                if field not in sample:
                    errors.append(f"Line {line_num}: Missing '{field}'")

            # Check objects
            if "objects" in sample:
                for obj in sample["objects"]:
                    if "bbox_2d" in obj:
                        bbox_count += 1
                    if "poly" in obj:
                        poly_count += 1
                        # Validate poly_points field if present
                        if "poly_points" in obj and obj["poly_points"] * 2 != len(
                            obj["poly"]
                        ):
                            errors.append(f"Line {line_num}: poly length mismatch")
                    if "desc" not in obj:
                        errors.append(f"Line {line_num}: Object missing 'desc'")

    print(f"    Samples: {samples_checked}")
    print(f"    bbox_2d objects: {bbox_count}")
    print(f"    poly objects: {poly_count}")

    if errors:
        print(f"\n  ✗ Found {len(errors)} errors:")
        for err in errors[:5]:
            print(f"    • {err}")
    else:
        print("\n  ✓ All samples valid!")

    # Show sample
    print("\n  Sample output:")
    with open(jsonl_path, "r") as f:
        sample = json.loads(f.readline())
        print(f"    Images: {sample['images']}")
        print(f"    Size: {sample['width']}x{sample['height']}")
        print(f"    Objects: {len(sample['objects'])}")
        for i, obj in enumerate(sample["objects"][:2]):
            geom = "poly" if "poly" in obj else "bbox_2d"
            print(f"      [{i}] {geom}, desc='{obj['desc']}'")


if __name__ == "__main__":
    main()
