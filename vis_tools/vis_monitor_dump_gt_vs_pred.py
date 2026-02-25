"""\
Visualize Stage-2 AB `monitor_dumps/step_*.json` samples as GT vs Pred side-by-side.

This is meant for diagnosing crowded-scene behaviors (near-duplicate preds, unlabeled
objects, FP/FN tradeoffs) using the same sample dumps produced by the trainer.

The output is a 2-column layout:
- left: GT boxes
- right: Pred boxes

The header contains:
- a top-right color legend (GT/FN/matched/FP)
- a per-sample class distribution summary for FN and FP

Usage (repo root, ms env):
  PYTHONPATH=. conda run -n ms python -P vis_tools/vis_monitor_dump_gt_vs_pred.py \
      --monitor_json output/stage2_ab/.../monitor_dumps/step_000900.json \
      --save_dir output/stage2_ab/.../monitor_dumps/vis_step_000900 \
      --limit 10

You can also pass a directory for `--monitor_json` (it will render all `step_*.json`).

Notes:
- This script uses PIL only (dependency-light), similar to `src/infer/vis.py`.
- Monitor dumps typically store an absolute image path in the user message content.
- Coord boxes are stored as `points_norm1000` in `[x1, y1, x2, y2]` order.
- When rendering multiple samples, the script writes an aggregate per-class FN/FP summary
  to `class_summary.json` under `--save_dir`.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Render GT vs Pred side-by-side from Stage-2 monitor dumps"
    )
    ap.add_argument(
        "--monitor_json",
        required=True,
        help="Path to step_*.json monitor dump OR a directory containing them",
    )
    ap.add_argument("--save_dir", required=True, help="Output directory for PNGs")
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max samples per dump to render (0 = all)",
    )
    ap.add_argument(
        "--box_width",
        type=int,
        default=3,
        help="Rectangle outline width in pixels",
    )
    ap.add_argument(
        "--gap",
        type=int,
        default=12,
        help="Gap in pixels between GT and Pred panels",
    )
    ap.add_argument(
        "--header_h",
        type=int,
        default=46,
        help="Minimum header height in pixels (auto-expands for legends)",
    )
    ap.add_argument(
        "--class_topk",
        type=int,
        default=8,
        help="Top-K classes to list for FN/FP summaries",
    )
    return ap.parse_args()


def _iter_monitor_paths(path: Path) -> Iterable[Path]:
    if path.is_dir():
        yield from sorted(path.glob("step_*.json"))
        return
    yield path


def _extract_image_path(sample: Mapping[str, Any]) -> str | None:
    # Prefer explicit fields if present.
    for k in ("image", "image_path"):
        v = sample.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()

    # Fall back to the first multimodal message content image.
    messages = sample.get("messages")
    if not isinstance(messages, list):
        return None

    for msg in messages:
        if not isinstance(msg, Mapping):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, Mapping):
                continue
            if str(part.get("type") or "") != "image":
                continue
            img = part.get("image")
            if isinstance(img, str) and img.strip():
                return img.strip()

    return None


def _norm1000_to_px(
    points_norm1000: list[Any], *, width: int, height: int
) -> tuple[int, int, int, int] | None:
    if len(points_norm1000) != 4:
        return None

    try:
        x1, y1, x2, y2 = (float(v) for v in points_norm1000)
    except (TypeError, ValueError):
        return None

    def _clamp01k(v: float) -> float:
        return max(0.0, min(999.0, v))

    y1 = _clamp01k(y1)
    x1 = _clamp01k(x1)
    y2 = _clamp01k(y2)
    x2 = _clamp01k(x2)

    # Map 0..999 bins to pixel coordinates 0..(size-1).
    def _scale(v: float, size: int) -> int:
        if size <= 1:
            return 0
        return int(round((v / 999.0) * float(size - 1)))

    x1p = _scale(x1, width)
    y1p = _scale(y1, height)
    x2p = _scale(x2, width)
    y2p = _scale(y2, height)

    x_lo, x_hi = (x1p, x2p) if x1p <= x2p else (x2p, x1p)
    y_lo, y_hi = (y1p, y2p) if y1p <= y2p else (y2p, y1p)
    return x_lo, y_lo, x_hi, y_hi


def _safe_desc(obj: Mapping[str, Any]) -> str:
    desc = obj.get("desc")
    if not isinstance(desc, str):
        return "<none>"
    desc = desc.strip()
    return desc or "<none>"


def _desc_counter(objs: list[Mapping[str, Any]], indices: Iterable[int]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for i in indices:
        if not isinstance(i, int):
            continue
        if i < 0 or i >= len(objs):
            continue
        counter[_safe_desc(objs[i])] += 1
    return counter


def _format_topk(counter: Counter[str], *, topk: int) -> str:
    if not counter:
        return "(none)"
    items = counter.most_common(max(1, int(topk)))
    parts = [f"{k}({v})" for k, v in items]
    extra_classes = len(counter) - len(items)
    if extra_classes > 0:
        parts.append(f"(+{extra_classes} cls)")
    return " ".join(parts)


def _draw_boxes(
    *,
    img: Image.Image,
    objs: list[Mapping[str, Any]],
    color_for_obj: callable,
    box_width: int,
) -> Image.Image:
    out = img.convert("RGB")
    draw = ImageDraw.Draw(out)

    width, height = out.size

    for obj in objs:
        gtype = str(obj.get("geom_type") or obj.get("type") or "").strip()
        if gtype != "bbox_2d":
            continue
        pts = obj.get("points_norm1000")
        if not isinstance(pts, list):
            continue
        rect = _norm1000_to_px(pts, width=width, height=height)
        if rect is None:
            continue

        outline = str(color_for_obj(obj))
        x1, y1, x2, y2 = rect
        draw.rectangle([x1, y1, x2, y2], outline=outline, width=int(box_width))

    return out


def _text_wh(draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont, text: str) -> tuple[int, int]:
    l, t, r, b = draw.textbbox((0, 0), str(text), font=font)
    return int(r - l), int(b - t)


def _truncate_to_width(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    text: str,
    max_width: int,
) -> str:
    text = str(text)
    if max_width <= 0:
        return ""
    if _text_wh(draw, font, text)[0] <= max_width:
        return text

    ell = "…"
    lo = 0
    hi = len(text)
    best = ell
    while lo <= hi:
        mid = (lo + hi) // 2
        cand = text[:mid].rstrip() + ell
        if _text_wh(draw, font, cand)[0] <= max_width:
            best = cand
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def _draw_class_summary(
    *,
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    x: int,
    y: int,
    max_width: int,
    fn_counter: Counter[str],
    fp_counter: Counter[str],
    topk: int,
    line_gap: int = 2,
) -> int:
    fn_total = int(sum(fn_counter.values()))
    fp_total = int(sum(fp_counter.values()))

    fn_str = _format_topk(fn_counter, topk=topk)
    fp_str = _format_topk(fp_counter, topk=topk)

    line1 = f"FN (GT) n={fn_total}: {fn_str}"
    line2 = f"FP (Pred) n={fp_total}: {fp_str}"

    line1 = _truncate_to_width(draw, font, line1, max_width)
    line2 = _truncate_to_width(draw, font, line2, max_width)

    draw.text((x, y), line1, fill=(0, 0, 0), font=font)
    _, th = _text_wh(draw, font, line1)
    y2 = y + th + int(line_gap)
    draw.text((x, y2), line2, fill=(0, 0, 0), font=font)
    _, th2 = _text_wh(draw, font, line2)
    return y2 + th2


def _render_pair(
    *,
    img_path: Path,
    gt_objs: list[Mapping[str, Any]],
    pred_objs: list[Mapping[str, Any]],
    match: Mapping[str, Any],
    stats: Mapping[str, Any],
    header_h: int,
    gap: int,
    box_width: int,
    class_topk: int,
) -> Image.Image:
    img = Image.open(img_path).convert("RGB")

    fn_gt = {int(i) for i in (match.get("fn_gt_indices") or []) if isinstance(i, int)}
    fp_pred = {int(i) for i in (match.get("fp_pred_indices") or []) if isinstance(i, int)}

    matched_pairs = match.get("matched_pairs") or []
    matched_pred: set[int] = set()
    matched_gt: set[int] = set()
    if isinstance(matched_pairs, list):
        for pair in matched_pairs:
            if (
                isinstance(pair, list)
                and len(pair) == 2
                and isinstance(pair[0], int)
                and isinstance(pair[1], int)
            ):
                matched_pred.add(int(pair[0]))
                matched_gt.add(int(pair[1]))

    fn_counter = _desc_counter(gt_objs, fn_gt)
    fp_counter = _desc_counter(pred_objs, fp_pred)

    def gt_color(obj: Mapping[str, Any]) -> str:
        idx = obj.get("index")
        if isinstance(idx, int) and idx in fn_gt:
            return "#ff9900"  # FN GT boxes
        return "#00ff00"  # GT boxes

    def pred_color(obj: Mapping[str, Any]) -> str:
        idx = obj.get("index")
        if isinstance(idx, int) and idx in fp_pred:
            return "#ff0000"  # FP
        if isinstance(idx, int) and idx in matched_pred:
            return "#00ff00"  # matched
        return "#00aaff"  # fallback

    left = _draw_boxes(
        img=img,
        objs=gt_objs,
        color_for_obj=gt_color,
        box_width=box_width,
    )
    right = _draw_boxes(
        img=img,
        objs=pred_objs,
        color_for_obj=pred_color,
        box_width=box_width,
    )

    w, h = img.size
    gap = max(0, int(gap))
    canvas_w = w * 2 + gap

    font = ImageFont.load_default()
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))

    title_y = 4
    title_h = _text_wh(measure, font, "GT")[1]
    line_h = _text_wh(measure, font, "Ag")[1]

    # Legend panel (top-right, avoids cluttering the boxes).
    legend_rows = [
        [("#00ff00", "GT (left)"), ("#ff9900", "FN GT (left)")],
        [("#00ff00", "Matched (right)"), ("#ff0000", "FP (right)")],
    ]
    swatch = 10
    pad = 6
    item_gap = 10
    swatch_gap = 4
    row_gap = 4
    section_gap = 6
    margin = 6

    row_ws: list[int] = []
    row_hs: list[int] = []
    for row in legend_rows:
        row_w = 0
        row_h = swatch
        for i, (_color, label) in enumerate(row):
            tw, th = _text_wh(measure, font, str(label))
            row_h = max(row_h, th)
            item_w = swatch + swatch_gap + tw
            if i > 0:
                row_w += item_gap
            row_w += item_w
        row_ws.append(int(row_w))
        row_hs.append(int(row_h))

    color_content_h = sum(row_hs) + row_gap * (len(row_hs) - 1)
    legend_min_w = max(row_ws) + 2 * pad

    # Add class summaries (FN/FP) inside the same top-right legend panel.
    fn_total = int(sum(fn_counter.values()))
    fp_total = int(sum(fp_counter.values()))

    fn_str = _format_topk(fn_counter, topk=int(class_topk))
    fp_str = _format_topk(fp_counter, topk=int(class_topk))

    fn_line = f"FN (GT) n={fn_total}: {fn_str}"
    fp_line = f"FP (Pred) n={fp_total}: {fp_str}"

    # Keep the legend width bounded so it stays on the right and doesn't
    # cover the "Pred" title on smaller images.
    legend_max_w = int(round(canvas_w * 0.45))
    legend_w = min(max(legend_min_w, legend_max_w), max(10, int(canvas_w - 2 * margin)))
    inner_w = max(10, int(legend_w - 2 * pad))

    summary_max_w = max(10, int(inner_w - (swatch + swatch_gap)))
    fn_line = _truncate_to_width(measure, font, fn_line, summary_max_w)
    fp_line = _truncate_to_width(measure, font, fp_line, summary_max_w)

    summary_rows = [("#ff9900", fn_line), ("#ff0000", fp_line)]
    summary_row_hs: list[int] = []
    for _c, text in summary_rows:
        th = _text_wh(measure, font, str(text))[1]
        summary_row_hs.append(int(max(swatch, th)))

    summary_content_h = sum(summary_row_hs) + row_gap * (len(summary_row_hs) - 1)

    legend_h = int(color_content_h + section_gap + summary_content_h + 2 * pad)

    x0 = max(0, int(canvas_w - margin - legend_w))
    y0 = title_y

    stats_h = line_h

    header_needed = max(
        int(header_h),
        int(y0 + legend_h + 4),
        int(title_y + title_h + 4 + stats_h + 4),
    )
    header_h = max(0, int(header_needed))

    canvas = Image.new("RGB", (canvas_w, h + header_h), color=(255, 255, 255))
    canvas.paste(left, (0, header_h))
    canvas.paste(right, (w + gap, header_h))

    draw = ImageDraw.Draw(canvas)

    # Titles.
    draw.text((6, title_y), "GT", fill=(0, 0, 0), font=font)
    draw.text((w + gap + 6, title_y), "Pred", fill=(0, 0, 0), font=font)

    # Legend box.
    draw.rectangle(
        [x0, y0, x0 + legend_w, y0 + legend_h],
        fill=(255, 255, 255),
        outline=(0, 0, 0),
    )

    # Color legend rows.
    y = y0 + pad
    for row, row_h in zip(legend_rows, row_hs):
        x = x0 + pad
        for color, label in row:
            label_s = str(label)
            tw, th = _text_wh(draw, font, label_s)
            sy = int(y + (row_h - swatch) * 0.5)
            ty = int(y + (row_h - th) * 0.5)
            draw.rectangle(
                [x, sy, x + swatch, sy + swatch],
                fill=str(color),
                outline=(0, 0, 0),
            )
            draw.text((x + swatch + swatch_gap, ty), label_s, fill=(0, 0, 0), font=font)
            x += swatch + swatch_gap + tw + item_gap
        y += row_h + row_gap
    if row_hs:
        y -= row_gap

    # Summary rows (object descriptions), in the same legend panel.
    y += section_gap
    for (color, text), row_h in zip(summary_rows, summary_row_hs):
        text_s = str(text)
        tw, th = _text_wh(draw, font, text_s)
        sw_y = int(y + (row_h - swatch) * 0.5)
        tx_y = int(y + (row_h - th) * 0.5)
        draw.rectangle(
            [x0 + pad, sw_y, x0 + pad + swatch, sw_y + swatch],
            fill=str(color),
            outline=(0, 0, 0),
        )
        draw.text(
            (x0 + pad + swatch + swatch_gap, tx_y),
            text_s,
            fill=(0, 0, 0),
            font=font,
        )
        y += row_h + row_gap

    # Stats line (bottom of header).
    prec = stats.get("precision")
    rec = stats.get("recall")
    f1 = stats.get("f1")
    gt_n = stats.get("gt_objects")
    pred_n = stats.get("valid_pred_objects")

    stat_str = (
        f"gt={gt_n} pred={pred_n} matched={stats.get('matched')} "
        f"p={prec:.3f} r={rec:.3f} f1={f1:.3f}"
        if isinstance(prec, (int, float))
        and isinstance(rec, (int, float))
        and isinstance(f1, (int, float))
        else f"gt={gt_n} pred={pred_n} matched={stats.get('matched')}"
    )
    stats_y = max(title_y, int(header_h - stats_h - 4))
    draw.text((6, stats_y), stat_str, fill=(0, 0, 0), font=font)

    return canvas


def main() -> None:
    args = parse_args()

    monitor_path = Path(args.monitor_json)
    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    agg_fn: Counter[str] = Counter()
    agg_fp: Counter[str] = Counter()
    rendered_count = 0
    sample_count = 0
    dump_count = 0

    for dump_path in _iter_monitor_paths(monitor_path):
        dump_count += 1
        try:
            dump = json.loads(dump_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue

        samples = dump.get("samples")
        if not isinstance(samples, list):
            continue

        step = dump.get("global_step")
        step_s = f"step_{int(step):06d}" if isinstance(step, int) else dump_path.stem

        limit = int(args.limit)
        for si, sample in enumerate(samples):
            if limit and si >= limit:
                break
            if not isinstance(sample, Mapping):
                continue
            sample_count += 1

            img_field = _extract_image_path(sample)
            if not img_field:
                continue
            img_path = Path(img_field)
            if not img_path.exists():
                # If monitor dumps ever store relative paths, resolve relative to dump dir.
                cand = dump_path.parent / img_field
                if cand.exists():
                    img_path = cand
                else:
                    continue

            gt_raw = sample.get("gt_objects")
            pred_raw = sample.get("pred_objects")
            gt_objs = [o for o in gt_raw if isinstance(o, Mapping)] if isinstance(gt_raw, list) else []
            pred_objs = (
                [o for o in pred_raw if isinstance(o, Mapping)] if isinstance(pred_raw, list) else []
            )
            match = sample.get("match")
            match_d = match if isinstance(match, Mapping) else {}
            stats = sample.get("stats")
            stats_d = stats if isinstance(stats, Mapping) else {}

            fn_gt_indices = match_d.get("fn_gt_indices") or []
            fp_pred_indices = match_d.get("fp_pred_indices") or []
            if isinstance(fn_gt_indices, list):
                agg_fn.update(_desc_counter(gt_objs, fn_gt_indices))
            if isinstance(fp_pred_indices, list):
                agg_fp.update(_desc_counter(pred_objs, fp_pred_indices))

            base_idx = sample.get("base_idx")
            base_s = f"base{int(base_idx):06d}" if isinstance(base_idx, int) else "baseNA"

            rendered = _render_pair(
                img_path=img_path,
                gt_objs=gt_objs,
                pred_objs=pred_objs,
                match=match_d,
                stats=stats_d,
                header_h=int(args.header_h),
                gap=int(args.gap),
                box_width=int(args.box_width),
                class_topk=int(args.class_topk),
            )

            save_name = f"{step_s}_s{si:02d}_{base_s}.png"
            rendered.save(out_dir / save_name)
            rendered_count += 1

    summary_path = out_dir / "class_summary.json"
    summary = {
        "monitor_json": str(monitor_path),
        "save_dir": str(out_dir),
        "dumps_seen": int(dump_count),
        "samples_seen": int(sample_count),
        "images_rendered": int(rendered_count),
        "fn_total": int(sum(agg_fn.values())),
        "fp_total": int(sum(agg_fp.values())),
        "fn_by_class": agg_fn.most_common(),
        "fp_by_class": agg_fp.most_common(),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    print(f"[vis_monitor_dump_gt_vs_pred] wrote: {summary_path}")
    print(f"[vis_monitor_dump_gt_vs_pred] rendered: {rendered_count} images")
    print(
        f"[vis_monitor_dump_gt_vs_pred] aggregate FN top={min(20, len(agg_fn))}: "
        + _format_topk(agg_fn, topk=20)
    )
    print(
        f"[vis_monitor_dump_gt_vs_pred] aggregate FP top={min(20, len(agg_fp))}: "
        + _format_topk(agg_fp, topk=20)
    )


if __name__ == "__main__":
    main()
