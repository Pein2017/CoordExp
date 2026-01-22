#!/usr/bin/env python3
"""
Dataset-agnostic smart-resize for JSONL detection/grounding datasets.

Takes an existing JSONL that already follows the CoordExp contract
(images, objects, width, height with pixel coordinates) and:
  - Resizes images to fit a pixel budget while snapping dimensions to a grid
  - Rewrites geometry fields (bbox_2d / poly / line) to match the resized images
  - Optionally relativizes image paths to the output JSONL location

Example:
  PYTHONPATH=. /root/miniconda3/envs/ms/bin/python public_data/scripts/rescale_jsonl.py \\
    --input-jsonl public_data/lvis/raw/train.jsonl \\
    --output-jsonl public_data/lvis/rescale_32_768/train.jsonl \\
    --output-images public_data/lvis/rescale_32_768 \\
    --image-factor 32 --max-pixels $((32*32*768)) --min-pixels $((32*32*4)) \\
    --num-workers 8 --relative-images
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Dict, MutableMapping, cast

from src.datasets.preprocessors.resize import SmartResizeParams, SmartResizePreprocessor


def _relativize_images(row: MutableMapping[str, Any], base_dir: Path) -> Dict[str, Any]:
    """Rewrite absolute image paths to be relative to base_dir."""
    images = row.get("images") or []
    rel_images = []
    base_dir_resolved = base_dir.resolve()

    for img in images:
        p = Path(str(img))
        if not p.is_absolute():
            rel_images.append(str(p))
            continue
        try:
            rel_images.append(str(p.resolve().relative_to(base_dir_resolved)))
        except ValueError:
            rel_images.append(f"images/{p.name}")

    result = dict(row)
    result["images"] = rel_images
    return result


def _process_row_worker(args_tuple):
    row, params_dict, jsonl_dir, output_dir, relative_output_root, relative_image_paths, images_root = args_tuple
    params = SmartResizeParams(**params_dict)
    pre = SmartResizePreprocessor(
        params=params,
        jsonl_dir=Path(jsonl_dir),
        output_dir=Path(output_dir),
        write_images=True,
        relative_output_root=Path(relative_output_root),
        images_root_override=Path(images_root) if images_root else None,
    )
    updated = pre.preprocess(row) or row
    if relative_image_paths:
        updated = _relativize_images(cast(MutableMapping[str, Any], updated), Path(relative_output_root))
    return json.dumps(updated, ensure_ascii=False)


def run_smart_resize(
    *,
    input_jsonl: Path,
    output_jsonl: Path,
    images_output_dir: Path,
    params: SmartResizeParams,
    relative_image_paths: bool,
    images_root_override: Path | None,
    num_workers: int = 1,
) -> None:
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    images_output_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with input_jsonl.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    max_workers = max(1, min(num_workers, cpu_count(), len(rows)))
    if max_workers == 1:
        pre = SmartResizePreprocessor(
            params=params,
            jsonl_dir=input_jsonl.parent,
            output_dir=images_output_dir,
            write_images=True,
            relative_output_root=output_jsonl.parent,
            images_root_override=images_root_override,
        )
        iterator = rows if tqdm is None else tqdm(rows, desc="Resizing", unit="sample")
        with output_jsonl.open("w", encoding="utf-8") as fout:
            for row in iterator:
                updated = pre.preprocess(row) or row
                if relative_image_paths:
                    updated = _relativize_images(cast(MutableMapping[str, Any], updated), output_jsonl.parent)
                fout.write(json.dumps(updated, ensure_ascii=False) + "\n")
    else:
        params_dict = asdict(params)
        worker_args = [
            (
                row,
                params_dict,
                str(input_jsonl.parent),
                str(images_output_dir),
                str(output_jsonl.parent),
                relative_image_paths,
                str(images_root_override) if images_root_override else None,
            )
            for row in rows
        ]
        with Pool(processes=max_workers) as pool:
            iterable = pool.imap(_process_row_worker, worker_args)
            results = iterable if tqdm is None else tqdm(iterable, total=len(rows), desc=f"Resizing ({max_workers} workers)")
            with output_jsonl.open("w", encoding="utf-8") as fout:
                for out_line in results:
                    fout.write(out_line + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smart-resize JSONL + images (dataset agnostic)")
    p.add_argument("--input-jsonl", type=Path, required=True, help="Source JSONL with pixel coords")
    p.add_argument("--output-jsonl", type=Path, required=True, help="Destination JSONL after resize")
    p.add_argument("--output-images", type=Path, required=True, help="Directory to write resized images")
    p.add_argument("--image-root", type=Path, help="Override root for resolving image paths (defaults to input JSONL dir)")
    p.add_argument("--image-factor", type=int, default=SmartResizeParams().image_factor, help="Grid factor (e.g., 32)")
    p.add_argument("--max-pixels", type=int, default=SmartResizeParams().max_pixels, help="Max pixel budget after resize")
    p.add_argument("--min-pixels", type=int, default=SmartResizeParams().min_pixels, help="Min pixel budget after resize")
    p.add_argument("--num-workers", type=int, default=1, help="Parallel workers (<= CPU cores)")
    p.add_argument("--relative-images", action="store_true", help="Rewrite image paths to be relative to output JSONL dir")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    params = SmartResizeParams(
        max_pixels=args.max_pixels,
        image_factor=args.image_factor,
        min_pixels=args.min_pixels,
    )
    run_smart_resize(
        input_jsonl=args.input_jsonl,
        output_jsonl=args.output_jsonl,
        images_output_dir=args.output_images,
        params=params,
        relative_image_paths=args.relative_images,
        images_root_override=args.image_root,
        num_workers=args.num_workers,
    )
    print(f"[+] Wrote resized JSONL: {args.output_jsonl}")
    print(f"[+] Resized images under: {args.output_images}")


if __name__ == "__main__":
    main()
