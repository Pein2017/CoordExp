"""Deterministic decoded-RGB compositor for the frozen visual-binding masks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image


def binding(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    data = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


def _box(mask: dict[str, Any], size: tuple[int, int]) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = (int(value) for value in mask["pixel_xyxy"])
    width, height = size
    box = (max(0, x1), max(0, y1), min(width, x2), min(height, y2))
    if box[0] >= box[2] or box[1] >= box[3]:
        raise ValueError(f"empty mask: {box}")
    return box


def _fill_rgb(image: Image.Image, box: tuple[int, int, int, int]) -> tuple[int, int, int]:
    x1, y1, x2, y2 = box
    left, top, right, bottom = max(0, x1 - 8), max(0, y1 - 8), min(image.width, x2 + 8), min(image.height, y2 + 8)
    pixels = image.load()
    total = [0, 0, 0]
    count = 0
    for y in range(top, bottom):
        for x in range(left, right):
            if x1 <= x < x2 and y1 <= y < y2:
                continue
            rgb = pixels[x, y]
            total[0] += int(rgb[0])
            total[1] += int(rgb[1])
            total[2] += int(rgb[2])
            count += 1
    if not count:
        raise ValueError("surrounding ring is empty")
    return tuple(int(round(value / count)) for value in total)  # type: ignore[return-value]


def compose(source: Path, mask: dict[str, Any], out_dir: Path, *, name: str) -> dict[str, Any]:
    """Write clean/ablated RGB and exact binary mask; return immutable bindings."""
    out_dir.mkdir(parents=True, exist_ok=False)
    with Image.open(source) as opened:
        source_rgb = opened.convert("RGB")
        decoded = source_rgb.copy()
    box = _box(mask, decoded.size)
    fill = _fill_rgb(decoded, box)
    mask_image = Image.new("L", decoded.size, 0)
    mask_image.paste(255, box)
    altered = decoded.copy()
    altered.paste(fill, box)
    image_path = out_dir / f"{name}.png"
    mask_path = out_dir / f"{name}-mask.png"
    altered.save(image_path, format="PNG", optimize=False, compress_level=6)
    mask_image.save(mask_path, format="PNG", optimize=False, compress_level=6)
    with Image.open(image_path) as check:
        if check.convert("RGB").size != decoded.size or check.convert("RGB").tobytes() == decoded.tobytes():
            raise ValueError("compositor did not produce a changed image")
    outside_changed = 0
    altered_pixels, source_pixels = altered.load(), decoded.load()
    x1, y1, x2, y2 = box
    for y in range(decoded.height):
        for x in range(decoded.width):
            if x1 <= x < x2 and y1 <= y < y2:
                continue
            if altered_pixels[x, y] != source_pixels[x, y]:
                outside_changed += 1
    if outside_changed:
        raise ValueError(f"compositor changed {outside_changed} complement pixels")
    changed_pixel_count = sum(1 for y in range(decoded.height) for x in range(decoded.width) if altered_pixels[x, y] != source_pixels[x, y])
    if changed_pixel_count == 0:
        raise ValueError("compositor changed no pixels")
    receipt = {
        "schema": "visual_instance_binding.compositor.v1",
        "source": binding(source),
        "decoded_size": list(decoded.size),
        "pixel_xyxy": list(box),
        "fill_rgb": list(fill),
        "ring": {"radius_px": 8, "clipped": True, "excludes_mask": True, "rounding": "round-half-even"},
        "changed_pixel_count": changed_pixel_count,
        "unchanged_complement_pixel_count": decoded.width * decoded.height - (x2 - x1) * (y2 - y1),
        "image": binding(image_path),
        "mask": binding(mask_path),
    }
    receipt_path = out_dir / "binding.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    receipt["binding"] = binding(receipt_path)
    return receipt


def clean_copy(source: Path, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=False)
    with Image.open(source) as opened:
        decoded = opened.convert("RGB")
    path = out_dir / "clean.png"
    decoded.save(path, format="PNG", optimize=False, compress_level=6)
    with Image.open(path) as check:
        if check.convert("RGB").size != decoded.size or check.convert("RGB").tobytes() != decoded.tobytes():
            raise ValueError("clean compositor roundtrip changed decoded RGB")
    result = {"schema": "visual_instance_binding.clean_compositor.v1", "source": binding(source), "decoded_size": list(decoded.size), "image": binding(path)}
    receipt = out_dir / "binding.json"
    receipt.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    result["binding"] = binding(receipt)
    return result
