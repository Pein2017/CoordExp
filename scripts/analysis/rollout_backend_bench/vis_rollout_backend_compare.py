"""Visualization: compare rollout predictions between HF and vLLM backends.

Input is the `compare_*.jsonl` produced by:
  scripts/analysis/rollout_backend_bench/benchmark_rollout_backends.py

Each record should contain:
  - image, width, height
  - gt: List[{type, points, desc}]            (pixel-space)
  - pred_hf: List[{type, points, desc}]       (pixel-space)
  - pred_vllm: List[{type, points, desc}]     (pixel-space)

Usage:
  /root/miniconda3/envs/ms/bin/python scripts/analysis/rollout_backend_bench/vis_rollout_backend_compare.py \\
    --compare_jsonl output/bench/rollout_backend_bench/<run>/compare_gpu0_seed17.jsonl \\
    --save_dir vis_out/rollout_backend_compare --limit 50
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


@dataclass
class Config:
    compare_jsonl: str
    save_dir: str = "vis_out/rollout_backend_compare"
    limit: int = 0


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


def _draw(ax, img: Image.Image, objs: List[Dict[str, Any]], color_map: Dict[str, str]) -> None:
    import matplotlib.patches as patches

    ax.imshow(img)
    ax.axis("off")
    for obj in objs:
        pts = obj.get("points") or []
        gtype = obj.get("type", "")
        label = obj.get("desc", "") or "object"
        color = color_map.get(label, "#1f77b4")
        linestyle = "-" if gtype == "bbox_2d" else "--" if gtype == "poly" else ":"
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
        elif gtype == "line" and len(pts) >= 4:
            xy = pair_points(pts)
            xs, ys = zip(*xy)
            ax.plot(xs, ys, color=color, linewidth=2, linestyle=linestyle)


def draw_compare(
    *,
    image_path: Path,
    gt: List[Dict[str, Any]],
    pred_hf: List[Dict[str, Any]],
    pred_vllm: List[Dict[str, Any]],
    save_path: Path,
    title_suffix: str = "",
) -> None:
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    img = Image.open(image_path).convert("RGB")
    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    ax_gt, ax_hf, ax_vv, ax_leg = axes

    # Color map shared across panels.
    labels = [obj.get("desc", "") or "object" for obj in (gt + pred_hf + pred_vllm)]
    color_map = _generate_colors(labels or ["object"])

    ax_gt.set_title("Ground Truth" + title_suffix)
    _draw(ax_gt, img, gt, color_map)

    ax_hf.set_title("HF Rollout" + title_suffix)
    _draw(ax_hf, img, pred_hf, color_map)

    ax_vv.set_title("vLLM Rollout" + title_suffix)
    _draw(ax_vv, img, pred_vllm, color_map)

    # Legend (counts per label: gt / hf / vllm)
    counts: Dict[str, List[int]] = {lab: [0, 0, 0] for lab in color_map.keys()}
    for obj in gt:
        lab = obj.get("desc", "") or "object"
        counts.setdefault(lab, [0, 0, 0])[0] += 1
    for obj in pred_hf:
        lab = obj.get("desc", "") or "object"
        counts.setdefault(lab, [0, 0, 0])[1] += 1
    for obj in pred_vllm:
        lab = obj.get("desc", "") or "object"
        counts.setdefault(lab, [0, 0, 0])[2] += 1

    ax_leg.axis("off")
    legend_handles = []
    active = [lab for lab, c in counts.items() if any(x > 0 for x in c)]
    active.sort(key=lambda lab: sum(counts[lab]), reverse=True)
    for lab in active[:30]:  # cap legend length for readability
        gt_c, hf_c, vv_c = counts[lab]
        legend_label = f"{lab} (gt {gt_c} / hf {hf_c} / vllm {vv_c})"
        handle = patches.Patch(facecolor="none", edgecolor=color_map[lab], label=legend_label)
        legend_handles.append(handle)
    if legend_handles:
        ax_leg.legend(handles=legend_handles, loc="center", framealpha=0.95, fontsize=9)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Compare rollout outputs between HF and vLLM")
    p.add_argument("--compare_jsonl", required=True)
    p.add_argument("--save_dir", default="vis_out/rollout_backend_compare")
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    return Config(compare_jsonl=a.compare_jsonl, save_dir=a.save_dir, limit=a.limit)


def main() -> None:
    cfg = parse_args()
    compare_path = Path(cfg.compare_jsonl)
    records = load_jsonl(compare_path, cfg.limit)
    base_dir = compare_path.parent
    out_dir = Path(cfg.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, rec in enumerate(tqdm(records, desc="Render", unit="img")):
        img_rel = rec.get("image")
        img_path = resolve_image_path(base_dir, img_rel)
        if not img_path or not img_path.exists():
            continue
        gt = rec.get("gt") or []
        pred_hf = rec.get("pred_hf") or []
        pred_vv = rec.get("pred_vllm") or []
        title_suffix = f" (line {rec.get('line_idx', '')})"
        save_path = out_dir / f"compare_{i:04d}.png"
        draw_compare(
            image_path=img_path,
            gt=gt,
            pred_hf=pred_hf,
            pred_vllm=pred_vv,
            save_path=save_path,
            title_suffix=title_suffix,
        )


if __name__ == "__main__":
    main()
