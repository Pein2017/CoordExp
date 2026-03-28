"""Visualization: compare rollout predictions between HF and vLLM backends.

Preferred input is two single-view prediction JSONLs, one per backend:
- `pred_hf.jsonl`
- `pred_vllm.jsonl`

Those member files are canonicalized independently and then composed only after
GT equivalence is verified. The legacy `compare_*.jsonl` flag is kept as a thin
compatibility wrapper that resolves those member files next to the compare
artifact produced by the benchmark script.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from PIL import Image

# Allow direct `python scripts/.../vis_rollout_backend_compare.py` invocation.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.paths import resolve_image_path_strict
from src.vis import DEFAULT_BBOX_OUTLINE_WIDTH, compose_comparison_scenes_from_jsonls


@dataclass
class Config:
    hf_jsonl: str | None
    vllm_jsonl: str | None
    compare_jsonl: str | None
    save_dir: str = "vis_out/rollout_backend_compare"
    limit: int = 0


def _derive_member_paths(compare_jsonl: Path) -> tuple[Path, Path]:
    stem = compare_jsonl.stem
    if not stem.startswith("compare_"):
        raise ValueError(
            "Legacy --compare_jsonl must be named like compare_<run_id>.jsonl so the "
            "member prediction JSONLs can be resolved."
        )
    run_id = stem[len("compare_") :]
    hf_jsonl = compare_jsonl.parent / f"{run_id}_hf" / "pred_hf.jsonl"
    vllm_jsonl = compare_jsonl.parent / f"{run_id}_vllm" / "pred_vllm.jsonl"
    if not hf_jsonl.exists() or not vllm_jsonl.exists():
        raise FileNotFoundError(
            "Could not resolve member prediction JSONLs from legacy compare artifact. "
            f"Expected {hf_jsonl} and {vllm_jsonl}."
        )
    return hf_jsonl, vllm_jsonl


def _resolve_member_inputs(cfg: Config) -> tuple[Path, Path]:
    if cfg.hf_jsonl and cfg.vllm_jsonl:
        return Path(cfg.hf_jsonl), Path(cfg.vllm_jsonl)
    if cfg.compare_jsonl:
        return _derive_member_paths(Path(cfg.compare_jsonl))
    raise ValueError("Provide either --hf_jsonl/--vllm_jsonl or legacy --compare_jsonl")


def _resolve_image_path(base_dir: Path, image_rel: str | None) -> Optional[Path]:
    if image_rel is None:
        return None
    root = os.environ.get("ROOT_IMAGE_DIR")
    root_dir = Path(root).resolve() if root else None
    return resolve_image_path_strict(
        image_rel,
        jsonl_dir=base_dir,
        root_image_dir=root_dir,
    )


def _draw_objects(ax, objs: Sequence[Dict[str, Any]], *, color: str, linestyle: str) -> None:
    import matplotlib.patches as patches

    for obj in objs:
        bbox = obj.get("bbox_2d") or []
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox]
        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=float(DEFAULT_BBOX_OUTLINE_WIDTH),
            edgecolor=color,
            facecolor="none",
            linestyle=linestyle,
        )
        ax.add_patch(rect)


def _draw_scene(
    *,
    scene: Dict[str, Any],
    image_path: Path,
    save_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    img = Image.open(image_path).convert("RGB")
    members = list(scene.get("members") or [])
    fig, axes = plt.subplots(1, len(members) + 1, figsize=(6 * (len(members) + 1), 6))
    if not isinstance(axes, (list, tuple)):
        axes = list(axes)
    else:
        axes = list(axes)

    gt_ax = axes[0]
    gt_ax.imshow(img)
    gt_ax.axis("off")
    gt_ax.set_title(f"Ground Truth (record {scene.get('record_idx')})")
    _draw_objects(gt_ax, list(scene.get("gt") or []), color="#2ca02c", linestyle="-")

    member_colors = ["#1f77b4", "#d62728", "#ff7f0e", "#9467bd"]
    for idx, member in enumerate(members, start=1):
        ax = axes[idx]
        ax.imshow(img)
        ax.axis("off")
        label = str(member.get("label") or f"member_{idx}")
        record = member.get("record") or {}
        ax.set_title(label)
        _draw_objects(
            ax,
            list(record.get("pred") or []),
            color=member_colors[(idx - 1) % len(member_colors)],
            linestyle="--",
        )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Compare rollout outputs between HF and vLLM")
    parser.add_argument("--hf_jsonl", default=None)
    parser.add_argument("--vllm_jsonl", default=None)
    parser.add_argument("--compare_jsonl", default=None)
    parser.add_argument("--save_dir", default="vis_out/rollout_backend_compare")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    return Config(
        hf_jsonl=args.hf_jsonl,
        vllm_jsonl=args.vllm_jsonl,
        compare_jsonl=args.compare_jsonl,
        save_dir=args.save_dir,
        limit=args.limit,
    )


def main() -> None:
    cfg = parse_args()
    hf_jsonl, vllm_jsonl = _resolve_member_inputs(cfg)
    scenes = compose_comparison_scenes_from_jsonls(
        {
            "hf": hf_jsonl,
            "vllm": vllm_jsonl,
        }
    )
    out_dir = Path(cfg.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_dir = hf_jsonl.parent
    for index, scene in enumerate(scenes):
        if cfg.limit and index >= int(cfg.limit):
            break
        image_path = _resolve_image_path(base_dir, str(scene.get("image") or ""))
        if image_path is None or not image_path.exists():
            continue
        save_path = out_dir / f"compare_{index:04d}.png"
        _draw_scene(
            scene=scene,
            image_path=image_path,
            save_path=save_path,
        )


if __name__ == "__main__":
    main()
