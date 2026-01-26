#!/usr/bin/env python
"""Scan a CoordExp JSONL for image/grid mismatches that can crash Qwen3-VL.

This checks that the number of visual tokens implied by image_grid_thw matches
pixel_values length (fast processor) and that per-record image counts align.
Optionally writes a cleaned JSONL excluding bad records.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable, List, Tuple

from PIL import Image
from transformers import AutoProcessor


def _extract_image_paths(record: dict, jsonl_dir: str) -> List[str]:
    # Prefer raw "images" list if present (CoordExp JSONL format).
    images = record.get("images") or []
    paths: List[str] = []
    if isinstance(images, list) and images:
        for img in images:
            if isinstance(img, str):
                path = img
            elif isinstance(img, dict) and "image" in img:
                path = img["image"]
            else:
                continue
            if not os.path.isabs(path):
                path = os.path.join(jsonl_dir, path)
            paths.append(path)
        return paths

    # Fallback: parse messages for image content
    messages = record.get("messages") or []
    if isinstance(messages, list):
        for turn in messages:
            if not isinstance(turn, dict):
                continue
            contents = turn.get("content") or []
            if not isinstance(contents, list):
                continue
            for item in contents:
                if isinstance(item, dict) and item.get("type") == "image":
                    path = item.get("image") or item.get("url")
                    if isinstance(path, str):
                        if not os.path.isabs(path):
                            path = os.path.join(jsonl_dir, path)
                        paths.append(path)
    return paths


def _open_images(paths: Iterable[str]) -> Tuple[List[Any], List[str]]:
    images = []
    bad = []
    for p in paths:
        try:
            with Image.open(p) as im:
                images.append(im.convert("RGB"))
        except Exception:
            bad.append(p)
    return images, bad


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", required=True, help="Path to JSONL to scan")
    parser.add_argument(
        "--model",
        default="model_cache/Qwen3-VL-8B-Instruct-coordexp",
        help="Processor/model path",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Max records to scan (0=all)"
    )
    parser.add_argument(
        "--clean_jsonl",
        default="",
        help="If set, write a cleaned JSONL excluding bad records",
    )
    parser.add_argument(
        "--bad_index_path",
        default="",
        help="If set, write bad indices to this file (one per line)",
    )
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl)
    jsonl_dir = str(jsonl_path.parent)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    image_processor = processor.image_processor

    bad_indices: List[int] = []
    kept = 0
    scanned = 0

    clean_f = None
    if args.clean_jsonl:
        clean_path = Path(args.clean_jsonl)
        clean_path.parent.mkdir(parents=True, exist_ok=True)
        clean_f = clean_path.open("w", encoding="utf-8")

    with jsonl_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if args.limit and scanned >= args.limit:
                break
            scanned += 1
            try:
                record = json.loads(line)
            except Exception:
                bad_indices.append(idx)
                continue

            image_paths = _extract_image_paths(record, jsonl_dir)
            if not image_paths:
                # No images; skip (text-only). Not expected for CoordExp, but not a vision mismatch.
                if clean_f is not None:
                    clean_f.write(line)
                kept += 1
                continue

            images, bad_paths = _open_images(image_paths)
            if bad_paths:
                bad_indices.append(idx)
                continue

            try:
                inputs = image_processor(images=images, return_tensors="pt", do_resize=False)
            except Exception:
                bad_indices.append(idx)
                continue

            pixel_values = inputs.get("pixel_values")
            grid_thw = inputs.get("image_grid_thw")
            if pixel_values is None or grid_thw is None:
                bad_indices.append(idx)
                continue

            # Basic per-record consistency checks
            num_images = len(images)
            try:
                if grid_thw.shape[0] != num_images:
                    bad_indices.append(idx)
                    continue
            except Exception:
                bad_indices.append(idx)
                continue

            # Fast processor returns patch tokens [num_patches, hidden]; compare against grid
            try:
                if hasattr(grid_thw, "prod"):
                    expected = int(grid_thw.prod(dim=-1).sum().item())
                else:
                    expected = None
            except Exception:
                expected = None

            try:
                if pixel_values is not None and pixel_values.ndim == 2 and expected is not None:
                    actual = int(pixel_values.shape[0])
                    if actual != expected:
                        bad_indices.append(idx)
                        continue
            except Exception:
                bad_indices.append(idx)
                continue

            if clean_f is not None:
                clean_f.write(line)
            kept += 1

    if clean_f is not None:
        clean_f.close()

    if args.bad_index_path:
        with open(args.bad_index_path, "w", encoding="utf-8") as bf:
            for i in bad_indices:
                bf.write(f"{i}\n")

    print(
        f"Scanned {scanned} records. Kept {kept}. Bad {len(bad_indices)}."
    )
    if bad_indices:
        print("First 20 bad indices:", bad_indices[:20])


if __name__ == "__main__":
    main()
