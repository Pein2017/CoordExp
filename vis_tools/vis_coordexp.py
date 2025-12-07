"""
Lightweight generation + visualization tool for CoordExp checkpoints.

Features:
- Loads a Qwen3-VL checkpoint and runs generation on a JSONL dataset.
- Parses coord-token outputs (<|coord_k|>, k in 0..999) and denormalizes to pixels.
- Supports bbox_2d, poly, and line geometries.
- Saves side-by-side GT vs prediction renders plus a JSONL dump of raw text/predictions.

Run (inside repo root, ms env):
  python vis_tools/vis_coordexp.py

Configuration is defined at the top of this file.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from src.config.prompts import SYSTEM_PROMPT, USER_PROMPT

# Local helpers
from src.coord_tokens.codec import token_to_int, value_in_coord_range

COORD_RE = re.compile(r"<\|coord_(\d{1,4})\|>")
GEOM_KEYS = ("bbox_2d", "poly", "line")
MAX_BIN = 999  # coord tokens are 0..999 inclusive


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


# ---------------------- Configuration ---------------------- #


@dataclass
class Config:
    """Configuration for visualization script."""

    ckpt: str = "output/debug/coord_merged_ck400"
    jsonl: str = "public_data/lvis/rescale_32_768_poly_max_20/val_tiny.coord.jsonl"
    device: str = "cuda:0"
    limit: int = 3  # Max samples to run (<=0 = all)
    temperature: float = 0.01
    top_p: float = 0.95
    max_new_tokens: int = 1024
    save_dir: str = "vis_out/debug"
    user_prompt: str | None = (
        None  # Override user prompt (defaults to training USER_PROMPT)
    )
    system_prompt: str | None = (
        None  # Override system prompt (defaults to training SYSTEM_PROMPT)
    )
    repetition_penalty: float | None = 1.05  # Repetition penalty for generation


# Global config instance
CONFIG = Config()


# ---------------------- IO helpers ---------------------- #


def load_jsonl(path: Path, limit: int | None) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if limit and limit > 0 and len(records) >= limit:
                break
    return records


def resolve_image_path(jsonl_path: Path, image_rel: str) -> Path:
    if os.path.isabs(image_rel):
        return Path(image_rel)
    root = os.environ.get("ROOT_IMAGE_DIR")
    base = Path(root) if root else jsonl_path.parent
    return (base / image_rel).resolve()


# ---------------------- Parsing helpers ---------------------- #


def extract_json_block(text: str) -> str | None:
    """Return the largest balanced {...} block, or None if not found.

    If the JSON is incomplete (truncated), attempts to repair it by:
    1. Finding the last complete object entry
    2. Adding missing closing braces/quotes
    """
    start = None
    depth = 0
    last_good = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0 and start is None:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    last_good = i
    if start is not None and last_good is not None and last_good >= start:
        return text[start : last_good + 1]

    # If no balanced block found, try to repair incomplete JSON
    if start is not None:
        candidate = text[start:]

        # Strategy 1: Find the last complete object entry
        # Look for "},\n", "},\"", or "}, " patterns that indicate a complete object
        # We want the position right after the "}" (before the comma)
        best_pos = -1
        for pattern in ["},\n", '},"', "}, "]:
            idx = candidate.rfind(pattern)
            if idx > best_pos:
                best_pos = idx

        if best_pos > 0:
            # Extract up to and including the "}" of the last complete object
            repaired = candidate[: best_pos + 1] + "}"
            try:
                json.loads(repaired)
                return repaired
            except Exception:
                pass

        # Strategy 2: Find any "}," pattern (more lenient)
        last_comma_idx = candidate.rfind("},")
        if last_comma_idx > 0:
            # Extract up to the "}" (before the comma) and close the JSON
            repaired = candidate[: last_comma_idx + 1] + "}"
            try:
                json.loads(repaired)
                return repaired
            except Exception:
                pass

        # Strategy 3: Count braces and try to close (last resort)
        open_braces = candidate.count("{")
        close_braces = candidate.count("}")
        missing = open_braces - close_braces
        if missing > 0:
            # Try adding closing braces (may work if structure is mostly complete)
            repaired = candidate + "}" * missing
            try:
                json.loads(repaired)
                return repaired
            except Exception:
                pass

    return None


def _flatten_points(pts: Any) -> List[Any] | None:
    if not isinstance(pts, (list, tuple)):
        return None
    if not pts:
        return []
    if isinstance(pts[0], (list, tuple)):
        flat: List[Any] = []
        for pair in pts:
            if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
                return None
            flat.extend(pair)
        return flat
    return list(pts)


def _coerce_coord(value: Any) -> int | None:
    if isinstance(value, str) and COORD_RE.fullmatch(value):
        v = token_to_int(value)
        return v if value_in_coord_range(v) else None
    try:
        v = int(round(float(value)))
    except Exception:
        return None
    return v if 0 <= v <= MAX_BIN else None


def parse_prediction(text: str) -> List[Dict[str, Any]]:
    """Parse model output JSON into a list of objects with integer coords."""
    block = extract_json_block(text)
    if not block:
        return []
    try:
        obj = json.loads(block)
    except Exception:
        return []

    # Handle outer wrappers like {"图片_1": {...}}
    if (
        isinstance(obj, dict)
        and len(obj) == 1
        and isinstance(next(iter(obj.values())), dict)
    ):
        maybe_inner = next(iter(obj.values()))
        if isinstance(maybe_inner, dict):
            obj = maybe_inner

    if not isinstance(obj, dict):
        return []

    parsed: List[Dict[str, Any]] = []
    for key, val in sorted(obj.items(), key=lambda kv: str(kv[0])):
        if not isinstance(val, dict):
            continue
        geom_keys = [g for g in GEOM_KEYS if g in val]
        if len(geom_keys) != 1:
            continue
        gtype = geom_keys[0]
        pts_raw = _flatten_points(val.get(gtype))
        if pts_raw is None or len(pts_raw) % 2 != 0:
            continue
        ints: List[int] = []
        ok = True
        for p in pts_raw:
            c = _coerce_coord(p)
            if c is None:
                ok = False
                break
            ints.append(c)
        if not ok:
            continue
        if gtype == "line":
            lp = val.get("line_points")
            if isinstance(lp, int) and lp > 0 and lp * 2 != len(ints):
                continue
        parsed.append(
            {
                "desc": str(val.get("desc", "")),
                "type": gtype,
                "points": ints,
            }
        )
    return parsed


# ---------------------- Geometry conversion ---------------------- #


def ints_to_pixels(ints: Sequence[int], width: float, height: float) -> List[float]:
    out: List[float] = []
    denom_x = max(1.0, float(width) - 1.0)
    denom_y = max(1.0, float(height) - 1.0)
    for i, v in enumerate(ints):
        frac = float(v) / float(MAX_BIN)
        if i % 2 == 0:
            out.append(frac * denom_x)
        else:
            out.append(frac * denom_y)
    return out


def pair_points(points: Sequence[float]) -> List[Tuple[float, float]]:
    assert len(points) % 2 == 0
    return [(float(points[i]), float(points[i + 1])) for i in range(0, len(points), 2)]


# ---------------------- Visualization ---------------------- #


@dataclass
class VisItem:
    image_path: Path
    width: int
    height: int
    gt: List[Dict[str, Any]]
    pred_raw: str
    pred_objs: List[Dict[str, Any]]


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


# ---------------------- Generation ---------------------- #


def build_messages(
    image: Image.Image, user_prompt: str, system_prompt: str
) -> List[Dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]


def generate_once(
    model,
    processor,
    image: Image.Image,
    user_prompt: str,
    system_prompt: str,
    gen_kwargs: dict,
    device: str,
) -> Tuple[str, str]:
    messages = build_messages(image, user_prompt, system_prompt)
    prompt_text = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    model_inputs = processor(text=prompt_text, images=[image], return_tensors="pt")
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

    with torch.inference_mode():
        gen_ids = model.generate(**model_inputs, **gen_kwargs)

    prompt_len = model_inputs["input_ids"].shape[1]
    gen_only = gen_ids[:, prompt_len:]
    # Keep coord tokens intact: do NOT skip special tokens because coord_* were
    # added as additional_special_tokens and would be stripped otherwise.
    raw_text = processor.tokenizer.batch_decode(
        gen_only, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )[0]
    clean_text = raw_text  # retained for backward compatibility in dumps
    return raw_text, clean_text


# ---------------------- Main ---------------------- #


def main() -> None:
    cfg = CONFIG
    jsonl_path = Path(cfg.jsonl)
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Honor ROOT_IMAGE_DIR if set; else derive from JSONL parent
    if not os.environ.get("ROOT_IMAGE_DIR"):
        os.environ["ROOT_IMAGE_DIR"] = str(jsonl_path.parent.resolve())

    records = load_jsonl(jsonl_path, cfg.limit if cfg.limit > 0 else None)
    if not records:
        raise SystemExit("No records loaded from JSONL.")

    print(f"[INFO] Loading model from {cfg.ckpt}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        cfg.ckpt, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).to(cfg.device)
    model.eval()

    processor = AutoProcessor.from_pretrained(cfg.ckpt, trust_remote_code=True)
    user_prompt = cfg.user_prompt or USER_PROMPT
    system_prompt = cfg.system_prompt or SYSTEM_PROMPT

    gen_kwargs = dict(
        max_new_tokens=cfg.max_new_tokens,
        do_sample=cfg.temperature > 0,
        temperature=max(1e-4, cfg.temperature),
        top_p=cfg.top_p,
        use_cache=True,
    )
    if cfg.repetition_penalty is not None:
        gen_kwargs["repetition_penalty"] = cfg.repetition_penalty

    # JSONL dump of preds
    dump_path = save_dir / "predictions.jsonl"
    with dump_path.open("w", encoding="utf-8") as fout:
        for idx, rec in enumerate(records):
            images = rec.get("images") or []
            if not images:
                continue
            img_path = resolve_image_path(jsonl_path, images[0])
            image = Image.open(img_path).convert("RGB")

            raw_text, clean_text = generate_once(
                model,
                processor,
                image,
                user_prompt,
                system_prompt,
                gen_kwargs,
                cfg.device,
            )
            # Parse using the untouched text so coord tokens are preserved
            pred_objs_norm = parse_prediction(raw_text)

            # Denorm predictions to pixels (rounded to nearest int)
            pred_px: List[Dict[str, Any]] = []
            width = float(rec.get("width") or image.width)
            height = float(rec.get("height") or image.height)
            for obj in pred_objs_norm:
                pts_px = [
                    round(v) for v in ints_to_pixels(obj["points"], width, height)
                ]
                pred_px.append(
                    {"type": obj["type"], "points": pts_px, "desc": obj.get("desc", "")}
                )

            # Prepare GT pixel coords (convert tokenized if present)
            gt_objs: List[Dict[str, Any]] = []
            for obj in rec.get("objects", []):
                for gkey in GEOM_KEYS:
                    if gkey in obj and obj[gkey] is not None:
                        pts_raw = _flatten_points(obj[gkey])
                        if pts_raw is None or len(pts_raw) % 2 != 0:
                            continue
                        ints = []
                        for p in pts_raw:
                            c = _coerce_coord(p)
                            if c is None:
                                # assume pixel already
                                try:
                                    ints = [float(v) for v in pts_raw]
                                except Exception:
                                    ints = []
                                break
                            ints.append(c)
                        if not ints:
                            continue
                        pts_px = (
                            ints_to_pixels(ints, width, height)
                            if all(isinstance(v, int) for v in ints)
                            else ints
                        )
                        pts_px = [round(v) for v in pts_px]
                        gt_objs.append(
                            {
                                "type": gkey,
                                "points": pts_px,
                                "desc": obj.get("desc", ""),
                            }
                        )
                        break

            vis_item = VisItem(
                image_path=img_path,
                width=int(width),
                height=int(height),
                gt=gt_objs,
                pred_raw=raw_text,
                pred_objs=pred_px,
            )
            img_save = save_dir / f"sample_{idx:04d}.png"
            try:
                draw_sample(vis_item, img_save)
            except Exception as exc:
                print(f"[WARN] draw failed for sample {idx}: {exc}")

            fout.write(
                json.dumps(
                    {
                        "index": idx,
                        "image": str(img_path),
                        "width": width,
                        "height": height,
                        "raw_text": raw_text,
                        "clean_text": clean_text,
                        "pred": pred_px,
                        "gt": gt_objs,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            fout.flush()
            print(f"[INFO] sample {idx}: saved {img_save}")


if __name__ == "__main__":
    main()
