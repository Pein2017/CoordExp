#!/usr/bin/env python3
"""Rescale JSONL records to an exact image size.

This script resizes every image to ``target_size x target_size`` and scales
geometry coordinates to remain aligned in pixel space.
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Dict, Iterator, List, MutableMapping, cast

from PIL import Image

from src.common.paths import resolve_image_path_best_effort
from src.datasets.geometry import clamp_points, scale_points


def _fix_bbox_xyxy(bbox_xyxy: list[float] | tuple[float, ...], width: int, height: int) -> List[int]:
    if len(bbox_xyxy) != 4:
        return [int(v) for v in bbox_xyxy]

    x1, y1, x2, y2 = (
        int(bbox_xyxy[0]),
        int(bbox_xyxy[1]),
        int(bbox_xyxy[2]),
        int(bbox_xyxy[3]),
    )

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    if x2 <= x1:
        if x2 < width - 1:
            x2 = min(width - 1, x1 + 1)
        elif x1 > 0:
            x1 = max(0, x2 - 1)
    if y2 <= y1:
        if y2 < height - 1:
            y2 = min(height - 1, y1 + 1)
        elif y1 > 0:
            y1 = max(0, y2 - 1)

    return [x1, y1, x2, y2]


def _scale_and_clamp_geometry(
    objects: list[dict[str, Any]],
    sx: float,
    sy: float,
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    scaled: list[dict[str, Any]] = []

    for obj in objects:
        updated = dict(obj)
        if obj.get("line") is not None or obj.get("line_points") is not None:
            raise ValueError("line geometry is not supported")

        if obj.get("bbox_2d") is not None:
            pts = scale_points(obj["bbox_2d"], sx, sy)
            updated["bbox_2d"] = _fix_bbox_xyxy(clamp_points(pts, width, height), width, height)
            updated.pop("poly", None)
            updated.pop("line", None)
            updated.pop("line_points", None)
        elif obj.get("poly") is not None:
            pts = scale_points(obj["poly"], sx, sy)
            updated["poly"] = clamp_points(pts, width, height)
            updated.pop("bbox_2d", None)
            updated.pop("line", None)
            updated.pop("line_points", None)

        scaled.append(updated)

    return scaled


def _relativize_images(
    row: MutableMapping[str, Any], base_dir: Path
) -> Dict[str, Any]:
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


def _resolve_image_paths(
    images: list[str], *, jsonl_dir: Path, image_root: Path | None
) -> list[Path]:
    paths: list[Path] = []
    for image in images:
        p = resolve_image_path_best_effort(
            image,
            jsonl_dir=jsonl_dir,
            root_image_dir=image_root,
            env_root_var=None,
        )
        paths.append(p)
    return paths


def _resize_image(image_path: Path, width: int, height: int, output_dir: Path, jsonl_dir: Path | None) -> Path:
    rel = image_path.name
    if jsonl_dir is not None:
        try:
            rel = str(image_path.relative_to(jsonl_dir))
        except ValueError:
            rel = image_path.name

    out_path = (output_dir / rel).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        with Image.open(out_path) as existing:
            if existing.size == (width, height):
                return out_path

    with Image.open(image_path) as img:
        rgb = img.convert("RGB")
        resized = rgb.resize((width, height), Image.Resampling.LANCZOS)
        resized.save(out_path)
    return out_path


def _process_row_worker(args_tuple):
    (
        row,
        target_size,
        jsonl_dir,
        output_dir,
        relative_output_root,
        relative_image_paths,
        images_root,
    ) = args_tuple

    images = row.get("images") or []
    if not images:
        return json.dumps(row, ensure_ascii=False)

    resolved = _resolve_image_paths(
        cast(list[str], images),
        jsonl_dir=Path(jsonl_dir),
        image_root=Path(images_root) if images_root else None,
    )

    width = int(row.get("width"))
    height = int(row.get("height"))
    target_w = int(target_size)
    target_h = int(target_size)

    sx = float(target_w) / float(width)
    sy = float(target_h) / float(height)

    row["width"] = target_w
    row["height"] = target_h
    row["objects"] = _scale_and_clamp_geometry(
        cast(list[dict[str, Any]], row.get("objects") or []),
        sx,
        sy,
        target_w,
        target_h,
    )

    rewritten = []
    for path in resolved:
        rewritten.append(
            str(
                _resize_image(
                    image_path=path,
                    width=target_w,
                    height=target_h,
                    output_dir=Path(output_dir),
                    jsonl_dir=Path(jsonl_dir),
                )
            )
        )

    row["images"] = rewritten

    if relative_image_paths:
        row = _relativize_images(row, Path(relative_output_root))

    return json.dumps(row, ensure_ascii=False)


def _iter_jsonl_rows(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fin:
        for raw in fin:
            line = raw.strip()
            if not line:
                continue
            yield cast(Dict[str, Any], json.loads(line))


def _count_nonempty_rows(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            if line.strip():
                count += 1
    return count


def run_exact_resize(
    *,
    input_jsonl: Path,
    output_jsonl: Path,
    images_output_dir: Path,
    target_size: int,
    relative_image_paths: bool,
    image_root_override: Path | None,
    num_workers: int = 1,
) -> None:
    if images_output_dir.exists() and images_output_dir.is_symlink():
        raise RuntimeError(
            f"Refusing to write resized images into symlinked output dir: {images_output_dir}"
        )

    images_output_dir.mkdir(parents=True, exist_ok=True)

    out_images = images_output_dir / "images"
    if out_images.exists():
        if out_images.is_symlink():
            out_images.unlink()
            out_images.mkdir(parents=True, exist_ok=True)
        elif out_images.is_file():
            raise RuntimeError(f"Expected images dir, found file: {out_images}")

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    total_rows = _count_nonempty_rows(input_jsonl)
    if total_rows <= 0:
        output_jsonl.write_text("", encoding="utf-8")
        return

    max_workers = max(1, min(int(num_workers), cpu_count(), total_rows))
    if max_workers == 1:
        with output_jsonl.open("w", encoding="utf-8") as fout:
            for row in _iter_jsonl_rows(input_jsonl):
                out = _process_row_worker(
                    (
                        row,
                        target_size,
                        str(input_jsonl.parent),
                        str(images_output_dir),
                        str(output_jsonl.parent),
                        relative_image_paths,
                        str(image_root_override) if image_root_override else None,
                    )
                )
                fout.write(out + "\n")
    else:
        args = (
            [
                (
                    row,
                    target_size,
                    str(input_jsonl.parent),
                    str(images_output_dir),
                    str(output_jsonl.parent),
                    relative_image_paths,
                    str(image_root_override) if image_root_override else None,
                )
                for row in _iter_jsonl_rows(input_jsonl)
            ]
        )

        try:
            from tqdm import tqdm
            iterator = tqdm(args, total=total_rows, desc=f"Resizing ({max_workers} workers)", unit="sample")
        except ImportError:
            iterator = args

        with Pool(processes=max_workers) as pool, output_jsonl.open("w", encoding="utf-8") as fout:
            for out_line in pool.imap(_process_row_worker, iterator, chunksize=32):
                fout.write(out_line + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Exact resize JSONL + images to fixed square size and adjust geometry"
    )
    p.add_argument("--input-jsonl", type=Path, required=True)
    p.add_argument("--output-jsonl", type=Path, required=True)
    p.add_argument("--output-images", type=Path, required=True)
    p.add_argument("--image-root", type=Path, help="Override root for resolving image paths")
    p.add_argument("--target-size", type=int, default=1024, help="Target width/height in pixels")
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--relative-images", action="store_true", help="Relativize image paths to output JSONL dir")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    run_exact_resize(
        input_jsonl=args.input_jsonl,
        output_jsonl=args.output_jsonl,
        images_output_dir=args.output_images,
        target_size=args.target_size,
        relative_image_paths=args.relative_images,
        image_root_override=args.image_root,
        num_workers=args.num_workers,
    )

    print(f"[+] Wrote resized JSONL: {args.output_jsonl}")
    print(f"[+] Wrote resized images under: {args.output_images}")


if __name__ == "__main__":
    main()
