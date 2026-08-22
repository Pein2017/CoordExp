"""Deterministic image/geometry resize used by COCO and LVIS preparation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, MutableMapping

from PIL import Image

from public_data.geometry import clamp_points, round_points, scale_points


@dataclass(frozen=True)
class SmartResizeParams:
    image_factor: int = 32
    min_pixels: int = 32 * 32 * 4
    max_pixels: int = 32 * 32 * 768

    def __post_init__(self) -> None:
        if self.image_factor <= 0:
            raise ValueError("image_factor must be positive")
        if self.min_pixels <= 0 or self.max_pixels < self.min_pixels:
            raise ValueError("pixel bounds must satisfy 0 < min_pixels <= max_pixels")


def smart_resize(height: int, width: int, *, params: SmartResizeParams | None = None) -> tuple[int, int]:
    params = params or SmartResizeParams()
    if height <= 0 or width <= 0:
        raise ValueError("image dimensions must be positive")
    factor = params.image_factor
    area = height * width
    # Public-data presets intentionally fill the configured visual-token budget
    # in both directions; this preserves the established preparation contract.
    scale = math.sqrt(params.max_pixels / area)
    new_height = max(factor, int(math.floor(height * scale / factor)) * factor)
    new_width = max(factor, int(math.floor(width * scale / factor)) * factor)
    while new_height * new_width > params.max_pixels:
        if new_height >= new_width and new_height > factor:
            new_height -= factor
        elif new_width > factor:
            new_width -= factor
        else:
            break
    return new_height, new_width


class SmartResizePreprocessor:
    def __init__(self, *, params: SmartResizeParams, jsonl_dir: Path, output_dir: Path,
                 write_images: bool, relative_output_root: Path,
                 images_root_override: Path | None = None) -> None:
        self.params = params
        self.jsonl_dir = Path(jsonl_dir)
        self.output_dir = Path(output_dir)
        self.write_images = write_images
        self.relative_output_root = Path(relative_output_root)
        self.images_root_override = Path(images_root_override) if images_root_override else None

    def preprocess(self, row: MutableMapping[str, Any]) -> dict[str, Any]:
        images = row.get("images") or []
        if len(images) != 1:
            raise ValueError("public-data resize expects exactly one image per row")
        declared = Path(str(images[0]))
        source_root = self.images_root_override or self.jsonl_dir
        source = declared if declared.is_absolute() else source_root / declared
        if not source.is_file():
            raise FileNotFoundError(f"image does not exist: {source}")
        with Image.open(source) as image:
            image = image.convert("RGB")
            old_width, old_height = image.size
            new_height, new_width = smart_resize(old_height, old_width, params=self.params)
            parts = declared.parts
            relative = Path(*parts[parts.index("images") + 1 :]) if "images" in parts else Path(declared.name)
            target = self.output_dir / "images" / relative
            if self.write_images:
                target.parent.mkdir(parents=True, exist_ok=True)
                image.resize((new_width, new_height), Image.Resampling.BICUBIC).save(target)
        updated = dict(row)
        updated["width"], updated["height"] = new_width, new_height
        updated["images"] = [str(target)]
        updated["objects"] = [
            _resize_object(obj, new_width / old_width, new_height / old_height, new_width, new_height)
            for obj in row.get("objects") or []
        ]
        return updated


def _resize_object(obj: Any, scale_x: float, scale_y: float, width: int, height: int) -> Any:
    if not isinstance(obj, dict):
        return obj
    result = dict(obj)
    for key in ("bbox_2d", "poly", "line"):
        values = result.get(key)
        if isinstance(values, list) and values:
            result[key] = round_points(clamp_points(scale_points(values, scale_x, scale_y), width, height))
    return result
