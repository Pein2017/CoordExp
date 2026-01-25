"""Visualization utility for CoordExp unified inference outputs.

Features
--------
- Optionally run inference via ``src.infer.InferenceEngine`` (coord/text mode).
- Render pixel-space GT vs prediction geometries from the standardized
  ``pred.jsonl`` schema (bbox_2d, poly).
- Saves per-sample PNG overlays to ``save_dir``.

Usage (inside repo root, ms env):
  python vis_tools/vis_coordexp.py

Configuration is controlled via the `CONFIG` object below.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image
from tqdm import tqdm

from src.common.geometry import pair_points

# ---------------------- Configuration ---------------------- #


@dataclass
class Config:
    """Configuration for visualization script."""

    pred_jsonl: str = "output/infer/coord/pred.jsonl"  # Required
    save_dir: str = "vis_out"
    limit: int = 0  # Max samples to render (<=0 = all)
    root_image_dir: str | None = None  # Optional override for resolving relative image paths


CONFIG = Config()


# ---------------------- IO helpers ---------------------- #


def load_jsonl(path: Path, limit: int | None) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            records.append(rec)
            if limit and limit > 0 and len(records) >= limit:
                break
    return records


def resolve_image_path(base: Path, image_rel: str | None) -> Optional[Path]:
    if image_rel is None:
        return None
    if os.path.isabs(image_rel):
        return Path(image_rel)
    root = os.environ.get("ROOT_IMAGE_DIR")
    base_dir = Path(root) if root else base
    return (base_dir / image_rel).resolve()


# ---------------------- Visualization ---------------------- #


def _generate_colors(labels: List[str]) -> Dict[str, str]:
    base_colors = [
        "#FF6B6B",
        "#4ECDC4",
        "#45B7D1",
        "#96CEB4",
        "#FFEAA7",
        "#DDA0DD",
        "#98D8C8",
        "#F7DC6F",
        "#BB8FCE",
        "#85C1E9",
        "#F8C471",
        "#82E0AA",
        "#F1948A",
        "#85929E",
        "#F4D03F",
        "#AED6F1",
        "#A9DFBF",
        "#F9E79F",
        "#D7BDE2",
        "#A2D9CE",
        "#FADBD8",
        "#D5DBDB",
    ]
    colors: Dict[str, str] = {}
    for i, label in enumerate(sorted(set(labels))):
        colors[label] = base_colors[i % len(base_colors)]
    return colors


@dataclass
class VisItem:
    image_path: Path
    width: int
    height: int
    gt: List[Dict[str, Any]]
    pred_objs: List[Dict[str, Any]]
    errors: List[str]


def draw_sample(item: VisItem, save_path: Path) -> None:
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    img = Image.open(item.image_path).convert("RGB")
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    ax_gt, ax_pred, ax_leg = axes

    # Collect labels and counts
    labels = [obj.get("desc", "") or "object" for obj in item.gt + item.pred_objs]
    color_map = _generate_colors(labels or ["object"])
    counts: Dict[str, List[int]] = {lab: [0, 0] for lab in color_map.keys()}
    for obj in item.gt:
        label = obj.get("desc", "") or "object"
        counts.setdefault(label, [0, 0])[0] += 1
    for obj in item.pred_objs:
        label = obj.get("desc", "") or "object"
        counts.setdefault(label, [0, 0])[1] += 1

    def _draw(ax, objs):
        ax.imshow(img)
        ax.axis("off")
        for obj in objs:
            pts = obj.get("points") or []
            gtype = obj.get("type", "")
            label = obj.get("desc", "") or "object"
            color = color_map.get(label, "#1f77b4")
            linestyle = "-" if gtype == "bbox_2d" else "--"
            if gtype == "bbox_2d" and len(pts) == 4:
                x1, y1, x2, y2 = pts
                rect = patches.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    linewidth=2,
                    edgecolor=color,
                    facecolor="none",
                    linestyle=linestyle,
                )
                ax.add_patch(rect)
            elif gtype == "poly" and len(pts) >= 6:
                poly_pts = pair_points(pts)
                poly = patches.Polygon(
                    poly_pts,
                    closed=True,
                    linewidth=2,
                    edgecolor=color,
                    facecolor="none",
                    linestyle=linestyle,
                )
                ax.add_patch(poly)

    ax_gt.set_title("Ground Truth")
    _draw(ax_gt, item.gt)

    ax_pred.set_title("Prediction")
    _draw(ax_pred, item.pred_objs)

    # Legend (right panel)
    ax_leg.axis("off")
    legend_handles = []
    active = [lab for lab, c in counts.items() if c[0] > 0 or c[1] > 0]
    active.sort(key=lambda lab: sum(counts[lab]), reverse=True)
    for lab in active:
        gt_c, pr_c = counts[lab]
        legend_label = f"{lab} (gt {gt_c} / pred {pr_c})"
        handle = patches.Patch(
            facecolor="none", edgecolor=color_map[lab], label=legend_label
        )
        legend_handles.append(handle)
    if legend_handles:
        ax_leg.legend(
            handles=legend_handles, loc="center", framealpha=0.95, fontsize=10
        )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------------------- Main flow ---------------------- #


def run_inference_if_needed(cfg: Config) -> Path:
    # Visualization no longer runs inference; pred_jsonl must exist and include GT.
    return Path(cfg.pred_jsonl)


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Visualization for CoordExp unified outputs (pred.jsonl with inline gt)"
    )
    parser.add_argument(
        "--pred_jsonl", required=True, help="Existing pred.jsonl containing gt and pred"
    )
    parser.add_argument("--save_dir", default="vis_out")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--root_image_dir",
        default=None,
        help="Optional ROOT_IMAGE_DIR override to resolve relative image paths",
    )
    args = parser.parse_args()

    cfg = Config(
        pred_jsonl=args.pred_jsonl,
        save_dir=args.save_dir,
        limit=args.limit,
        root_image_dir=args.root_image_dir,
    )
    return cfg


def main() -> None:
    cfg = parse_args()
    if cfg.root_image_dir:
        os.environ["ROOT_IMAGE_DIR"] = cfg.root_image_dir
    pred_path = run_inference_if_needed(cfg)
    records = load_jsonl(pred_path, cfg.limit)
    base_dir = pred_path.parent
    vis_dir = Path(cfg.save_dir)
    vis_dir.mkdir(parents=True, exist_ok=True)

    skipped = 0
    saved = 0
    for idx, rec in enumerate(tqdm(records, desc="Render", unit="img")):
        image_rel = rec.get("image") or (rec.get("images") or [None])[0]
        img_path = resolve_image_path(base_dir, image_rel)
        if not img_path or not img_path.exists():
            skipped += 1
            continue
        item = VisItem(
            image_path=img_path,
            width=int(rec.get("width", 0) or 0),
            height=int(rec.get("height", 0) or 0),
            gt=rec.get("gt", []),
            pred_objs=rec.get("pred", []),
            errors=rec.get("errors", []),
        )
        save_path = vis_dir / f"vis_{idx:04d}.png"
        draw_sample(item, save_path)
        saved += 1
    print(f"[vis_coordexp] saved={saved} skipped_missing_image={skipped} out_dir={vis_dir}")


if __name__ == "__main__":
    main()
