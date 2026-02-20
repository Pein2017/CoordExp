"""Visualization stage for the unified inference pipeline.

This is intentionally dependency-light (PIL only) so `vis` can be run as a
stage in CI / headless environments.

Input contract: unified artifact `gt_vs_pred.jsonl`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from PIL import Image, ImageDraw

from src.common.paths import resolve_image_path_strict
from src.utils import get_logger

logger = get_logger(__name__)


def _resolve_image_path(
    jsonl_path: Path,
    image_field: Optional[str],
    *,
    root_image_dir: Optional[Path],
) -> Optional[Path]:
    return resolve_image_path_strict(
        image_field,
        jsonl_dir=jsonl_path.parent,
        root_image_dir=root_image_dir,
    )


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(rec, dict):
                yield rec


def _color_rgba(hex_rgb: str, alpha: int) -> Tuple[int, int, int, int]:
    hex_rgb = hex_rgb.lstrip("#")
    r = int(hex_rgb[0:2], 16)
    g = int(hex_rgb[2:4], 16)
    b = int(hex_rgb[4:6], 16)
    return (r, g, b, alpha)


def _draw_objs(draw: ImageDraw.ImageDraw, objs: List[Dict[str, Any]], *, outline: str) -> None:
    for obj in objs:
        gtype = obj.get("type")
        pts = obj.get("points") or []
        if not isinstance(pts, list):
            continue
        if gtype == "bbox_2d" and len(pts) == 4:
            x1, y1, x2, y2 = [int(v) for v in pts]
            draw.rectangle([x1, y1, x2, y2], outline=outline, width=3)
        elif gtype == "poly" and len(pts) >= 6 and len(pts) % 2 == 0:
            xy = [(int(pts[i]), int(pts[i + 1])) for i in range(0, len(pts), 2)]
            draw.line(xy + [xy[0]], fill=outline, width=3)


def render_vis_from_jsonl(
    jsonl_path: Path,
    *,
    out_dir: Path,
    limit: int = 20,
    root_image_dir: Optional[Path] = None,
    root_source: str = "none",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    if root_image_dir is None:
        root_env = str(os.environ.get("ROOT_IMAGE_DIR") or "").strip()
        if root_env:
            root_image_dir = Path(root_env).resolve()
            root_source = "env"

    if root_image_dir is not None:
        logger.info(
            "vis: resolved ROOT_IMAGE_DIR (source=%s): %s", root_source, root_image_dir
        )
    else:
        logger.info(
            "vis: ROOT_IMAGE_DIR not set; falling back to JSONL directory for image resolution"
        )

    for idx, rec in enumerate(_iter_jsonl(jsonl_path)):
        if limit and idx >= limit:
            break

        image_field = rec.get("image")
        img_path = _resolve_image_path(
            jsonl_path,
            image_field,
            root_image_dir=root_image_dir,
        )
        if img_path is None:
            logger.warning("vis: missing image for record %d (%r)", idx, image_field)
            continue

        try:
            img = Image.open(img_path).convert("RGBA")
        except OSError:
            logger.warning("vis: failed to open image %s", img_path)
            continue

        # Overlay: GT (green) + pred (red)
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(overlay)
        gt_raw = rec.get("gt")
        gt = gt_raw if isinstance(gt_raw, list) else []

        pred_raw = rec.get("pred")
        pred = pred_raw if isinstance(pred_raw, list) else []

        _draw_objs(d, gt, outline="#00ff00")
        _draw_objs(d, pred, outline="#ff0000")

        composed = Image.alpha_composite(img, overlay).convert("RGB")
        save_path = out_dir / f"vis_{idx:04d}.png"
        composed.save(save_path)
