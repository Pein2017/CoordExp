#!/usr/bin/env python3
"""Render first-30/60 post-native forced-continuation boxes by coord confidence."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping

from PIL import Image, ImageDraw

from scripts.research.render_coordinate_confidence_boxes import (
    FOOTER_HEIGHT,
    _confidence_color,
    _distribution,
    _draw_legend,
    _font,
    _image_path,
    _read_panel,
)


DEFAULT_SCORE_DIR = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-07-30-iterative-forced-continue-exact-native/"
    "post-native-prefix-coordinate-confidence-six-arms"
)
DEFAULT_PANEL = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-07-21-best-sampled-trajectory-positive-row-imitation-screen/"
    "evaluation-inputs/human-refined-12.coord.jsonl"
)
CHECKPOINTS = ("sorted", "random", "permutation")
RP_LABELS = ("rp1.0", "rp1.1")
PREDICTION_BUDGETS = (30, 60)
HEADER_HEIGHT = 48


def _load_scores(score_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for checkpoint in CHECKPOINTS:
        path = score_dir / f"coord-likelihood-{checkpoint}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row["checkpoint"] != checkpoint:
                raise ValueError(f"{path}:{line_number}: checkpoint mismatch")
            confidence = float(row["coord_confidence_percentile"])
            coord_mean = float(row["coord_mean"])
            if not 0.0 <= confidence <= 1.0 or not math.isfinite(coord_mean):
                raise ValueError(f"{path}:{line_number}: invalid coordinate confidence")
            rows.append(row)
    return rows


def _display_bbox(
    bbox: Any, width: int, height: int
) -> tuple[tuple[int, int, int, int] | None, bool]:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None, False
    left, top, right, bottom = (int(value) for value in bbox)
    if not (0 <= left <= right <= width and 0 <= top <= bottom <= height):
        return None, False
    adjusted = False
    if left == right:
        adjusted = True
        if right < width:
            right += 1
        elif left > 0:
            left -= 1
        else:
            return None, False
    if top == bottom:
        adjusted = True
        if bottom < height:
            bottom += 1
        elif top > 0:
            top -= 1
        else:
            return None, False
    return (left, top, right, bottom), adjusted


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n")


def render(args: argparse.Namespace) -> dict[str, Any]:
    panel_rows = _read_panel(args.panel_jsonl)
    scores = _load_scores(args.score_dir)
    by_arm_image: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in scores:
        key = (str(row["arm"]), str(row["image_id"]))
        by_arm_image[key].append(row)
    for rows in by_arm_image.values():
        rows.sort(key=lambda row: int(row["post_native_prediction_order"]))

    expected_arms = {
        f"{checkpoint}-{rp_label}"
        for checkpoint in CHECKPOINTS
        for rp_label in RP_LABELS
    }
    observed_arms = {arm for arm, _ in by_arm_image}
    if observed_arms != expected_arms:
        raise ValueError(
            f"arm mismatch: missing={sorted(expected_arms-observed_arms)} "
            f"extra={sorted(observed_arms-expected_arms)}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_budgets: dict[str, Any] = {}
    title_font = _font(18)
    legend_font = _font(12)
    for budget in PREDICTION_BUDGETS:
        budget_dir = args.output_dir / f"first-{budget}-post-native"
        if budget_dir.exists() and any(budget_dir.iterdir()):
            raise FileExistsError(f"refusing to overwrite non-empty {budget_dir}")
        budget_dir.mkdir(parents=True, exist_ok=True)
        rendered: list[dict[str, Any]] = []
        selected_score_rows: list[dict[str, Any]] = []
        global_values: dict[str, list[float]] = {arm: [] for arm in expected_arms}
        omitted_nonrenderable: dict[str, int] = {arm: 0 for arm in expected_arms}
        display_adjustments: dict[str, int] = {arm: 0 for arm in expected_arms}

        for image_index, (example_id, panel_row) in enumerate(panel_rows.items(), 1):
            image_id = str(int(Path(str(panel_row["file_name"])).stem))
            image_path = _image_path(args.panel_jsonl, panel_row)
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
            width, height = image.size
            canvas_width = width * len(CHECKPOINTS)
            panel_stride = HEADER_HEIGHT + height
            canvas_height = panel_stride * len(RP_LABELS) + FOOTER_HEIGHT
            canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
            draw = ImageDraw.Draw(canvas)
            line_width = max(3, round(min(width, height) / 280))
            halo_width = line_width + 2
            panel_summaries: dict[str, Any] = {}

            for rp_index, rp_label in enumerate(RP_LABELS):
                row_top = rp_index * panel_stride
                for checkpoint_index, checkpoint in enumerate(CHECKPOINTS):
                    arm = f"{checkpoint}-{rp_label}"
                    x_offset = checkpoint_index * width
                    canvas.paste(image, (x_offset, row_top + HEADER_HEIGHT))
                    available = by_arm_image.get((arm, image_id), [])
                    selected = available[:budget]
                    selected_score_rows.extend(selected)
                    global_values[arm].extend(float(row["coord_mean"]) for row in selected)
                    rendered_count = 0
                    for row in sorted(
                        selected,
                        key=lambda item: float(item["coord_confidence_percentile"]),
                    ):
                        displayed, adjusted = _display_bbox(
                            row["bbox_pixel_xyxy"], width, height
                        )
                        if displayed is None:
                            omitted_nonrenderable[arm] += 1
                            continue
                        display_adjustments[arm] += int(adjusted)
                        left, top, right, bottom = displayed
                        shifted = (
                            left + x_offset,
                            top + row_top + HEADER_HEIGHT,
                            right + x_offset,
                            bottom + row_top + HEADER_HEIGHT,
                        )
                        draw.rectangle(shifted, outline=(35, 35, 35), width=halo_width)
                        draw.rectangle(
                            shifted,
                            outline=_confidence_color(
                                float(row["coord_confidence_percentile"])
                            ),
                            width=line_width,
                        )
                        rendered_count += 1
                    values = [float(row["coord_mean"]) for row in selected]
                    median_text = f"{median(values):.3f}" if values else "n/a"
                    title = (
                        f"{checkpoint.title()} | RP={rp_label[2:]} | "
                        f"{rendered_count}/{len(selected)} boxes | median {median_text}"
                    )
                    draw.text(
                        (x_offset + 14, row_top + 13),
                        title,
                        fill=(20, 20, 20),
                        font=title_font,
                    )
                    panel_summaries[arm] = {
                        "available_scored_predictions": len(available),
                        "selected_predictions": len(selected),
                        "rendered_predictions": rendered_count,
                        "omitted_nonrenderable": len(selected) - rendered_count,
                        "coord_mean": _distribution(values) if values else None,
                    }
                    if checkpoint_index:
                        draw.line(
                            [
                                (x_offset, row_top),
                                (x_offset, row_top + panel_stride),
                            ],
                            fill=(30, 30, 30),
                            width=2,
                        )
                if rp_index:
                    draw.line(
                        [(0, row_top), (canvas_width, row_top)],
                        fill=(30, 30, 30),
                        width=2,
                    )

            _draw_legend(
                draw,
                canvas_width,
                panel_stride * len(RP_LABELS),
                legend_font,
                "within-checkpoint-percentile",
            )
            output_path = budget_dir / f"{image_index:02d}-{example_id}.png"
            canvas.save(output_path, format="PNG", optimize=True)
            rendered.append(
                {
                    "example_id": example_id,
                    "image_id": image_id,
                    "source_image": str(image_path),
                    "output_png": str(output_path.resolve()),
                    "panels": panel_summaries,
                }
            )

        score_slice_path = budget_dir / "bbox-coordinate-confidence.jsonl"
        selected_score_rows.sort(
            key=lambda row: (
                str(row["checkpoint"]),
                float(row["repetition_penalty"]),
                int(row["image_id"]),
                int(row["post_native_prediction_order"]),
            )
        )
        _write_jsonl(score_slice_path, selected_score_rows)
        budget_manifest = {
            "schema_version": "forced_continue_coordinate_confidence_visualization.v1",
            "prediction_scope": (
                f"first {budget} parsed post-native predictions in executed generation order; "
                "native predictions excluded; forced rows and post-force natural rows retained"
            ),
            "layout": "two RP rows by three checkpoint columns per source image",
            "row_order": list(RP_LABELS),
            "column_order": list(CHECKPOINTS),
            "matching_semantics": "none; GT, TP/FP/FN, and duplication are not rendered",
            "confidence_field": "coord_mean",
            "confidence_definition": (
                "FP32 raw-model teacher-forced mean log probability over four coordinate tokens"
            ),
            "color_field": "coord_confidence_percentile",
            "color_reference": (
                "checkpoint-specific K16 union-cluster medoid coord_mean CDF from the "
                "2026-07-29 likelihood probe"
            ),
            "color_semantics": "lighter is lower coordinate confidence; darker is higher",
            "cross_checkpoint_color_semantics": (
                "colors are checkpoint-relative confidence ranks, not absolute calibrated "
                "probabilities across checkpoints"
            ),
            "source_score_dir": str(args.score_dir.resolve()),
            "source_panel": str(args.panel_jsonl.resolve()),
            "score_slice": str(score_slice_path.resolve()),
            "selected_prediction_count": len(selected_score_rows),
            "omitted_nonrenderable": omitted_nonrenderable,
            "display_bbox_adjustments": display_adjustments,
            "global_coord_mean_distributions": {
                arm: _distribution(values) if values else None
                for arm, values in sorted(global_values.items())
            },
            "images": rendered,
        }
        manifest_path = budget_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(budget_manifest, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        manifest_budgets[str(budget)] = {
            "manifest": str(manifest_path.resolve()),
            "image_count": len(rendered),
            "selected_prediction_count": len(selected_score_rows),
        }

    root_manifest = {
        "schema_version": "forced_continue_coordinate_confidence_bundle.v1",
        "budgets": manifest_budgets,
    }
    root_path = args.output_dir / "manifest.json"
    root_path.write_text(
        json.dumps(root_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return root_manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-dir", type=Path, default=DEFAULT_SCORE_DIR)
    parser.add_argument("--panel-jsonl", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    args.score_dir = args.score_dir.resolve(strict=True)
    args.panel_jsonl = args.panel_jsonl.resolve(strict=True)
    args.output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (args.score_dir / "rendered-confidence-only").resolve()
    )
    manifest = render(args)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
