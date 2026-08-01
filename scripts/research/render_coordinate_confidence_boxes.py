#!/usr/bin/env python3
"""Render union-cluster medoid boxes colored only by coordinate confidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

from PIL import Image, ImageDraw, ImageFont


LOW_COLOR = (255, 247, 188)
MID_COLOR = (83, 182, 214)
HIGH_COLOR = (8, 48, 107)
CHECKPOINTS = ("sorted", "random", "permutation")
COLOR_MODES = ("within-checkpoint-percentile", "absolute-logprob-bins")
ABSOLUTE_LOGPROB_EDGES = (-4.0, -3.5, -3.0, -2.5)
HEADER_HEIGHT = 52
FOOTER_HEIGHT = 54


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot compute a quantile of an empty sequence")
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _mix(
    left: tuple[int, int, int], right: tuple[int, int, int], t: float
) -> tuple[int, int, int]:
    return (
        round(left[0] + (right[0] - left[0]) * t),
        round(left[1] + (right[1] - left[1]) * t),
        round(left[2] + (right[2] - left[2]) * t),
    )


ABSOLUTE_COLORS = (
    LOW_COLOR,
    _mix(LOW_COLOR, MID_COLOR, 0.5),
    MID_COLOR,
    _mix(MID_COLOR, HIGH_COLOR, 0.5),
    HIGH_COLOR,
)
Font = ImageFont.ImageFont | ImageFont.FreeTypeFont


def _confidence_color(percentile: float) -> tuple[int, int, int]:
    value = min(1.0, max(0.0, percentile))
    if value <= 0.5:
        return _mix(LOW_COLOR, MID_COLOR, value * 2.0)
    return _mix(MID_COLOR, HIGH_COLOR, (value - 0.5) * 2.0)


def _font(size: int) -> Font:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if path.is_file():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _read_panel(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not raw_line.strip():
            continue
        row = json.loads(raw_line)
        file_name = Path(str(row["file_name"]))
        example_id = f"coco2017_val_{file_name.stem}"
        if example_id in rows:
            raise ValueError(
                f"duplicate panel example {example_id} at line {line_number}"
            )
        rows[example_id] = row
    if len(rows) != 12:
        raise ValueError(f"expected 12 panel rows, found {len(rows)}")
    return rows


def _image_path(panel_path: Path, row: dict[str, Any]) -> Path:
    images = row.get("images")
    if not isinstance(images, list) or len(images) != 1:
        raise ValueError("panel row must contain exactly one image")
    path = (panel_path.parent / str(images[0])).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _absolute_bin_index(logprob: float) -> int:
    return sum(logprob >= edge for edge in ABSOLUTE_LOGPROB_EDGES)


def _absolute_bin_labels() -> tuple[str, ...]:
    probabilities = tuple(math.exp(edge) * 100.0 for edge in ABSOLUTE_LOGPROB_EDGES)
    return (
        f"< -4.0  (<{probabilities[0]:.1f}%)",
        f"-4.0 to -3.5  ({probabilities[0]:.1f}-{probabilities[1]:.1f}%)",
        f"-3.5 to -3.0  ({probabilities[1]:.1f}-{probabilities[2]:.1f}%)",
        f"-3.0 to -2.5  ({probabilities[2]:.1f}-{probabilities[3]:.1f}%)",
        f">= -2.5  (>={probabilities[3]:.1f}%)",
    )


def _draw_legend(
    draw: ImageDraw.ImageDraw,
    width: int,
    top: int,
    font: Font,
    color_mode: str,
) -> None:
    left = 24
    right = width - 24
    bar_top = top + 7
    bar_bottom = top + 20
    if color_mode == "absolute-logprob-bins":
        labels = _absolute_bin_labels()
        bin_width = (right - left) / len(labels)
        for index, (color, label) in enumerate(
            zip(ABSOLUTE_COLORS, labels, strict=True)
        ):
            bin_left = round(left + index * bin_width)
            bin_right = round(left + (index + 1) * bin_width)
            draw.rectangle((bin_left, bar_top, bin_right, bar_bottom), fill=color)
            draw.rectangle(
                (bin_left, bar_top, bin_right, bar_bottom),
                outline=(45, 45, 45),
                width=1,
            )
            label_box = draw.textbbox((0, 0), label, font=font)
            label_width = label_box[2] - label_box[0]
            draw.text(
                (bin_left + (bin_right - bin_left - label_width) / 2, bar_bottom + 4),
                label,
                fill=(30, 30, 30),
                font=font,
            )
        return
    segments = max(1, right - left)
    for offset in range(segments):
        percentile = offset / max(1, segments - 1)
        draw.line(
            [(left + offset, bar_top), (left + offset, bar_bottom)],
            fill=_confidence_color(percentile),
            width=1,
        )
    draw.rectangle((left, bar_top, right, bar_bottom), outline=(45, 45, 45), width=1)
    draw.text(
        (left, bar_bottom + 3),
        "LOW coord confidence (lighter)",
        fill=(30, 30, 30),
        font=font,
    )
    high_text = "HIGH coord confidence (darker)"
    high_box = draw.textbbox((0, 0), high_text, font=font)
    draw.text(
        (right - (high_box[2] - high_box[0]), bar_bottom + 3),
        high_text,
        fill=(30, 30, 30),
        font=font,
    )


def _display_bbox(
    bbox: Any,
    width: int,
    height: int,
) -> tuple[tuple[int, int, int, int], bool]:
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError(f"invalid medoid bbox: {bbox!r}")
    left, top, right, bottom = (int(value) for value in bbox)
    if not (0 <= left <= right <= width and 0 <= top <= bottom <= height):
        raise ValueError(f"bbox outside image {width}x{height}: {bbox!r}")
    adjusted = False
    if left == right:
        adjusted = True
        if right < width:
            right += 1
        else:
            left -= 1
    if top == bottom:
        adjusted = True
        if bottom < height:
            bottom += 1
        else:
            top -= 1
    return (left, top, right, bottom), adjusted


def _distribution(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "min": min(values),
        "p05": _quantile(values, 0.05),
        "p10": _quantile(values, 0.10),
        "p25": _quantile(values, 0.25),
        "median": _quantile(values, 0.50),
        "p75": _quantile(values, 0.75),
        "p90": _quantile(values, 0.90),
        "p95": _quantile(values, 0.95),
        "max": max(values),
    }


def render(args: argparse.Namespace) -> dict[str, Any]:
    payload = json.loads(args.cluster_confidence.read_text(encoding="utf-8"))
    panel_rows = _read_panel(args.panel_jsonl)
    clusters_payload = payload.get("clusters", {})
    clusters_by_checkpoint: dict[str, list[dict[str, Any]]] = {}
    by_checkpoint_example: dict[str, dict[str, list[dict[str, Any]]]] = {}
    global_values: dict[str, list[float]] = {}
    for checkpoint in CHECKPOINTS:
        checkpoint_clusters = clusters_payload.get(checkpoint)
        if not isinstance(checkpoint_clusters, list) or not checkpoint_clusters:
            raise ValueError(f"no clusters for checkpoint {checkpoint!r}")
        clusters_by_checkpoint[checkpoint] = checkpoint_clusters
        by_example: dict[str, list[dict[str, Any]]] = {key: [] for key in panel_rows}
        values: list[float] = []
        for cluster in checkpoint_clusters:
            example_id = str(cluster["example_id"])
            if example_id not in by_example:
                raise ValueError(
                    f"cluster references unknown panel example {example_id}"
                )
            value = float(cluster["likelihood"]["medoid"]["coord_mean"])
            percentile = float(
                cluster["percentiles"]["coord_mean"]["within_checkpoint"]
            )
            if not math.isfinite(value) or not 0.0 <= percentile <= 1.0:
                raise ValueError(
                    f"invalid coordinate confidence for {cluster['cluster_id']}"
                )
            cluster["_coord_value"] = value
            cluster["_coord_percentile"] = percentile
            by_example[example_id].append(cluster)
            values.append(value)
        if any(not clusters for clusters in by_example.values()):
            missing = [
                example_id
                for example_id, clusters in by_example.items()
                if not clusters
            ]
            raise ValueError(f"checkpoint {checkpoint!r} has no clusters for {missing}")
        by_checkpoint_example[checkpoint] = by_example
        global_values[checkpoint] = values

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty output: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    title_font = _font(20)
    legend_font = _font(12)
    rendered: list[dict[str, Any]] = []
    display_adjustments = {checkpoint: 0 for checkpoint in CHECKPOINTS}
    for index, (example_id, row) in enumerate(panel_rows.items(), 1):
        image_path = _image_path(args.panel_jsonl, row)
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        width, height = image.size
        if (width, height) != (int(row["width"]), int(row["height"])):
            raise ValueError(f"panel dimensions disagree for {example_id}")
        canvas_width = width * len(CHECKPOINTS)
        canvas = Image.new(
            "RGB", (canvas_width, height + HEADER_HEIGHT + FOOTER_HEIGHT), "white"
        )
        draw = ImageDraw.Draw(canvas)
        line_width = max(3, round(min(width, height) / 280))
        halo_width = line_width + 2
        checkpoint_summaries: dict[str, Any] = {}
        for checkpoint_index, checkpoint in enumerate(CHECKPOINTS):
            offset = checkpoint_index * width
            canvas.paste(image, (offset, HEADER_HEIGHT))
            sort_field = (
                "_coord_value"
                if args.color_mode == "absolute-logprob-bins"
                else "_coord_percentile"
            )
            clusters = sorted(
                by_checkpoint_example[checkpoint][example_id],
                key=lambda item: item[sort_field],
            )
            for cluster in clusters:
                (left, top, right, bottom), adjusted = _display_bbox(
                    cluster["medoid_bbox"], width, height
                )
                display_adjustments[checkpoint] += int(adjusted)
                shifted = (
                    left + offset,
                    top + HEADER_HEIGHT,
                    right + offset,
                    bottom + HEADER_HEIGHT,
                )
                draw.rectangle(shifted, outline=(35, 35, 35), width=halo_width)
                draw.rectangle(
                    shifted,
                    outline=(
                        ABSOLUTE_COLORS[_absolute_bin_index(cluster["_coord_value"])]
                        if args.color_mode == "absolute-logprob-bins"
                        else _confidence_color(cluster["_coord_percentile"])
                    ),
                    width=line_width,
                )
            values = [float(cluster["_coord_value"]) for cluster in clusters]
            header = (
                f"{checkpoint.title()} | {len(clusters)} boxes | "
                f"median coord logp {median(values):.3f}"
            )
            draw.text((offset + 18, 14), header, fill=(20, 20, 20), font=title_font)
            checkpoint_summaries[checkpoint] = {
                "cluster_count": len(clusters),
                "absolute_bin_counts": [
                    sum(_absolute_bin_index(value) == bin_index for value in values)
                    for bin_index in range(len(ABSOLUTE_COLORS))
                ],
                "coord_logprob": {
                    "min": min(values),
                    "median": median(values),
                    "max": max(values),
                },
            }
            if checkpoint_index:
                draw.line(
                    [(offset, 0), (offset, HEADER_HEIGHT + height)],
                    fill=(30, 30, 30),
                    width=2,
                )
        _draw_legend(
            draw,
            canvas_width,
            HEADER_HEIGHT + height,
            legend_font,
            args.color_mode,
        )
        output_path = args.output_dir / f"{index:02d}-{example_id}.png"
        canvas.save(output_path, format="PNG", optimize=True)
        rendered.append(
            {
                "example_id": example_id,
                "source_image": str(image_path),
                "output_png": str(output_path.resolve()),
                "checkpoints": checkpoint_summaries,
            }
        )

    for checkpoint in CHECKPOINTS:
        rendered_count = sum(
            item["checkpoints"][checkpoint]["cluster_count"] for item in rendered
        )
        if rendered_count != len(clusters_by_checkpoint[checkpoint]):
            raise RuntimeError(
                f"rendered cluster total does not match input for {checkpoint}"
            )
    absolute_bins = []
    labels = _absolute_bin_labels()
    for bin_index, (label, color) in enumerate(
        zip(labels, ABSOLUTE_COLORS, strict=True)
    ):
        absolute_bins.append(
            {
                "index": bin_index,
                "label": label,
                "color_rgb": list(color),
                "counts": {
                    checkpoint: sum(
                        _absolute_bin_index(value) == bin_index
                        for value in global_values[checkpoint]
                    )
                    for checkpoint in CHECKPOINTS
                },
            }
        )
    manifest = {
        "schema_version": "coordinate_confidence_comparison.v1",
        "layout": "one row by three checkpoint panels per source image",
        "checkpoints": list(CHECKPOINTS),
        "confidence_field": "likelihood.medoid.coord_mean",
        "color_mode": args.color_mode,
        "color_field": (
            "likelihood.medoid.coord_mean"
            if args.color_mode == "absolute-logprob-bins"
            else "percentiles.coord_mean.within_checkpoint"
        ),
        "color_semantics": "lighter is lower confidence; darker is higher confidence",
        "absolute_logprob_bins": absolute_bins,
        "cross_checkpoint_color_semantics": (
            "the same raw thresholds are rendered across checkpoints, but raw-model "
            "likelihood is not established as calibrated or transferable across checkpoints"
            if args.color_mode == "absolute-logprob-bins"
            else "colors compare within-checkpoint rank, not absolute raw likelihood calibration"
        ),
        "matching_semantics": "none; every union-cluster medoid bbox is rendered",
        "display_bbox_adjustments": {
            "semantics": "zero-area source boxes are expanded to one pixel for visibility only",
            "counts": display_adjustments,
        },
        "source_cluster_confidence": str(args.cluster_confidence.resolve()),
        "source_panel": str(args.panel_jsonl.resolve()),
        "global_distributions": {
            checkpoint: _distribution(global_values[checkpoint])
            for checkpoint in CHECKPOINTS
        },
        "images": rendered,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cluster-confidence", type=Path, required=True)
    parser.add_argument("--panel-jsonl", type=Path, required=True)
    parser.add_argument(
        "--color-mode", choices=COLOR_MODES, default="within-checkpoint-percentile"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = render(args)
    print(json.dumps(manifest["global_distributions"], sort_keys=True))
    print(f"rendered {len(manifest['images'])} images to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
