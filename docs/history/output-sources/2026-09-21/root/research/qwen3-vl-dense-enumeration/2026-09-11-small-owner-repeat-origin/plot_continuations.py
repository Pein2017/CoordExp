#!/usr/bin/env python3
"""Render first-five free-row continuation panels from a frozen reduction.

Only ``result.first5_rows`` are drawn as generated detections. Forced prefix
seeds are optional dotted context and are never counted or labeled as model
outputs. No GT or owner-truth adjudication is performed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image


ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-11-small-owner-repeat-origin"
)
DEFAULT_PACKET = ROOT / "packet.json"
DEFAULT_REDUCTION = ROOT / "reduction.json"
DEFAULT_SMOKE = ROOT / "smoke-reduction.json"
DEFAULT_OUT = ROOT / "continuation-visualizations"

CELL_ORDER = [
    ("original", "native"),
    ("original", "translated"),
    ("donor", "native"),
    ("donor", "translated"),
]
BOUNDARIES = ["early", "late"]
ROW_COLORS = ["#00A6D6", "#F28E2B", "#59A14F", "#B07AA1", "#EDC948"]
INVALID_COLOR = "#D62728"
SEED_COLOR = "#4D4D4D"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bins_to_pixels(bins: list[int], width: int, height: int) -> list[int]:
    if len(bins) != 4 or any(isinstance(value, bool) or not isinstance(value, int) for value in bins):
        raise ValueError(f"coordinate bins must be four integers: {bins!r}")
    return [
        round(bins[0] * width / 1000),
        round(bins[1] * height / 1000),
        round(bins[2] * width / 1000),
        round(bins[3] * height / 1000),
    ]


def draw_box_or_line(
    ax: Any,
    pixels: list[int],
    *,
    color: str,
    label: str | None,
    valid: bool,
    linestyle: str = "-",
    linewidth: float = 2.0,
    alpha: float = 1.0,
) -> None:
    x1, y1, x2, y2 = pixels
    plot_color = color if valid else INVALID_COLOR
    if x1 == x2 or y1 == y2:
        ax.plot(
            [x1, x2],
            [y1, y2],
            color=plot_color,
            linewidth=max(3.8, linewidth),
            linestyle=linestyle,
            alpha=alpha,
        )
        anchor_x, anchor_y = min(x1, x2), min(y1, y2)
    else:
        ax.plot(
            [x1, x2, x2, x1, x1],
            [y1, y1, y2, y2, y1],
            color=plot_color,
            linewidth=linewidth,
            linestyle=linestyle,
            alpha=alpha,
        )
        anchor_x, anchor_y = min(x1, x2), min(y1, y2)
    if label is not None:
        suffix = " invalid-line" if not valid else ""
        ax.text(
            anchor_x + 3,
            anchor_y + 15,
            label + suffix,
            fontsize=7.2,
            color="white",
            fontweight="bold",
            bbox={"facecolor": plot_color, "edgecolor": "white", "alpha": 0.86, "pad": 1.3},
        )


def verified_image(case: dict[str, Any], role: str) -> tuple[Image.Image, dict[str, Any]]:
    path = Path(case["image_path"])
    observed_hash = sha256_file(path)
    expected_hash = case["image_plan"]["image_content_sha256"]
    if observed_hash != expected_hash:
        raise ValueError(f"{role} image hash differs from packet: {path}")
    image = Image.open(path).convert("RGB")
    width, height = image.size
    expected_size = (int(case["image_width"]), int(case["image_height"]))
    if image.size != expected_size:
        raise ValueError(f"{role} image dimensions differ from packet: {path}")
    return image, {
        "path": str(path),
        "sha256": observed_hash,
        "width": width,
        "height": height,
    }


def expected_job_id(boundary: str, image: str, history: str) -> str:
    return f"{boundary}_{image}_{history}"


def validate_cell(
    result: dict[str, Any],
    *,
    case_id: str,
    boundary: str,
    image: str,
    history: str,
) -> None:
    expected = expected_job_id(boundary, image, history)
    fields = {
        "case_id": case_id,
        "job_id": expected,
        "boundary": boundary,
        "image_condition": image,
        "history_condition": history,
    }
    for key, value in fields.items():
        if str(result[key]) != str(value):
            raise ValueError(f"cell {expected} has wrong {key}: {result[key]!r}")
    rows = result["first5_rows"]
    if len(rows) > 5 or len(rows) != min(5, int(result["free_complete_rows"])):
        raise ValueError(f"cell {expected} first5 row count is inconsistent")
    for row in rows:
        bins_to_pixels(row["coord_bins"], 1000, 1000)


def stop_label(result: dict[str, Any]) -> str:
    if result["stop"] == "im_end":
        return f"native EOS after {result['free_tokens']} free tok"
    if result["stop"] == "length":
        return f"{result['free_tokens']}-token horizon"
    return f"stop={result['stop']} after {result['free_tokens']} free tok"


def render_cell(
    ax: Any,
    *,
    result: dict[str, Any],
    packet_case: dict[str, Any],
    background: Image.Image,
    boundary: str,
    image_condition: str,
    history_condition: str,
) -> dict[str, Any]:
    width, height = background.size
    ax.imshow(background)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_xticks([])
    ax.set_yticks([])

    seed = packet_case["seeds"][boundary]
    seed_bins = seed["original_bins"] if history_condition == "native" else seed["translated_bins"]
    draw_box_or_line(
        ax,
        bins_to_pixels(seed_bins, width, height),
        color=SEED_COLOR,
        label=None,
        valid=bool(seed["geometry_valid"]),
        linestyle=":",
        linewidth=1.8,
        alpha=0.85,
    )

    invalid_generated = []
    rendered_rows = []
    for free_number, row in enumerate(result["first5_rows"], start=1):
        valid = bool(row["geometry_valid"])
        if not valid:
            invalid_generated.append(int(row["raw_row_index"]))
        pixels = bins_to_pixels(row["coord_bins"], width, height)
        draw_box_or_line(
            ax,
            pixels,
            color=ROW_COLORS[free_number - 1],
            label=f"free#{free_number} r{row['raw_row_index']} {row['description']}",
            valid=valid,
            linewidth=2.15,
        )
        rendered_rows.append(
            {
                "free_complete_ordinal": free_number,
                "raw_row_index": int(row["raw_row_index"]),
                "description": row["description"],
                "coord_bins": row["coord_bins"],
                "geometry_valid": valid,
                "pixel_xyxy_unvalidated": pixels,
            }
        )

    ax.set_title(
        f"{boundary} | {image_condition} image | {history_condition} history\n"
        f"free complete={result['free_complete_rows']} · {stop_label(result)}\n"
        f"valid={result['free_valid_rows']} · strict repeat={result['free_strict_repeats_against_all_earlier']} · "
        f"geometry-invalid={result['free_geometry_invalid_rows']}",
        fontsize=8.5,
    )
    return {
        "job_id": result["job_id"],
        "boundary": boundary,
        "image_condition": image_condition,
        "history_condition": history_condition,
        "free_complete_rows": result["free_complete_rows"],
        "free_tokens": result["free_tokens"],
        "stop": result["stop"],
        "free_valid_rows": result["free_valid_rows"],
        "free_strict_repeats_against_all_earlier": result["free_strict_repeats_against_all_earlier"],
        "free_geometry_invalid_rows": result["free_geometry_invalid_rows"],
        "rendered_first5_free_complete_rows": rendered_rows,
        "rendered_invalid_raw_row_indices": invalid_generated,
        "forced_seed_context": {
            "bins": seed_bins,
            "geometry_valid": seed["geometry_valid"],
            "style": "dotted_context_not_generated_detection",
        },
    }


def legend_handles() -> list[Line2D]:
    handles = [
        Line2D([0], [0], color=color, linewidth=2.2, label=f"freely generated complete row #{index}")
        for index, color in enumerate(ROW_COLORS, start=1)
    ]
    handles.extend(
        [
            Line2D([0], [0], color=INVALID_COLOR, linewidth=4, label="generated geometry-invalid row (line if degenerate)"),
            Line2D([0], [0], color=SEED_COLOR, linewidth=1.8, linestyle=":", label="forced prefix seed context; not generated detection"),
        ]
    )
    return handles


def prepare_case_images(packet_case: dict[str, Any]) -> tuple[dict[str, Image.Image], dict[str, Any]]:
    source, source_receipt = verified_image(packet_case["source_case"], "source")
    donor, donor_receipt = verified_image(packet_case["donor_case"], "donor")
    if source.size != donor.size:
        raise ValueError(f"case {packet_case['case_id']} source/donor dimensions differ")
    return {"original": source, "donor": donor}, {"source": source_receipt, "donor": donor_receipt}


def index_results(reduction: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for result in reduction["results"]:
        key = (str(result["case_id"]), str(result["job_id"]))
        if key in indexed:
            raise ValueError(f"duplicate reduction cell: {key}")
        indexed[key] = result
    return indexed


def render_full_case(
    *,
    packet_case: dict[str, Any],
    indexed: dict[tuple[str, str], dict[str, Any]],
    out_dir: Path,
) -> dict[str, Any]:
    case_id = str(packet_case["case_id"])
    backgrounds, image_receipts = prepare_case_images(packet_case)
    figure = plt.figure(figsize=(25, 13.6), constrained_layout=True)
    grid = figure.add_gridspec(4, 4, height_ratios=[0.22, 1, 1, 0.18])
    header = figure.add_subplot(grid[0, :])
    footer = figure.add_subplot(grid[3, :])
    header.axis("off")
    footer.axis("off")
    header.set_title(
        f"Case {case_id}: first five freely generated complete rows | "
        f"source {packet_case['source_case']['input_record']['image_id']} · donor {packet_case['donor_image_id']} | "
        "GT/owner truth not adjudicated",
        fontsize=15,
        fontweight="bold",
        pad=1,
    )

    cells = []
    for row_index, boundary in enumerate(BOUNDARIES, start=1):
        for column_index, (image, history) in enumerate(CELL_ORDER):
            job_id = expected_job_id(boundary, image, history)
            result = indexed[(case_id, job_id)]
            validate_cell(
                result,
                case_id=case_id,
                boundary=boundary,
                image=image,
                history=history,
            )
            ax = figure.add_subplot(grid[row_index, column_index])
            cells.append(
                render_cell(
                    ax,
                    result=result,
                    packet_case=packet_case,
                    background=backgrounds[image],
                    boundary=boundary,
                    image_condition=image,
                    history_condition=history,
                )
            )
    footer.legend(handles=legend_handles(), loc="center", ncol=4, fontsize=9, frameon=True)
    footer.text(
        0.5,
        -0.18,
        "Solid overlays are only result.first5_rows. Dotted seed geometry belongs to the forced history prefix and is not a detection.",
        ha="center",
        va="top",
        fontsize=9,
    )
    output = out_dir / f"case-{case_id}-continuations.png"
    figure.savefig(output, dpi=145, facecolor="white")
    plt.close(figure)
    return {
        "case_id": case_id,
        "source_image_id": int(packet_case["source_case"]["input_record"]["image_id"]),
        "donor_image_id": int(packet_case["donor_image_id"]),
        "images": image_receipts,
        "cells": cells,
        "figure": {"path": str(output), "sha256": sha256_file(output), "size_bytes": output.stat().st_size},
    }


def render_prototype(
    *,
    packet: dict[str, Any],
    reduction: dict[str, Any],
    packet_path: Path,
    reduction_path: Path,
    out_dir: Path,
) -> None:
    if reduction.get("scope") != "smoke" or int(reduction.get("cells", -1)) != 5:
        raise ValueError("prototype requires the five-cell smoke reduction")
    indexed = index_results(reduction)
    packet_case = next(case for case in packet["cases"] if str(case["case_id"]) == "351017")
    case_id = "351017"
    backgrounds, image_receipts = prepare_case_images(packet_case)
    figure = plt.figure(figsize=(24, 7.2), constrained_layout=True)
    grid = figure.add_gridspec(3, 4, height_ratios=[0.24, 1, 0.2])
    header = figure.add_subplot(grid[0, :])
    footer = figure.add_subplot(grid[2, :])
    header.axis("off")
    footer.axis("off")
    header.set_title(
        "SMOKE PARTIAL — case 351017 early boundary only (4 factorial cells of eventual 74-cell reduction)\n"
        "First five freely generated complete rows; GT/owner truth not adjudicated",
        fontsize=14,
        fontweight="bold",
        pad=0,
    )
    cells = []
    for column_index, (image, history) in enumerate(CELL_ORDER):
        job_id = expected_job_id("early", image, history)
        result = indexed[(case_id, job_id)]
        validate_cell(result, case_id=case_id, boundary="early", image=image, history=history)
        ax = figure.add_subplot(grid[1, column_index])
        cells.append(
            render_cell(
                ax,
                result=result,
                packet_case=packet_case,
                background=backgrounds[image],
                boundary="early",
                image_condition=image,
                history_condition=history,
            )
        )
    footer.legend(handles=legend_handles(), loc="center", ncol=4, fontsize=8.5, frameon=True)
    output = out_dir / "prototype-smoke-partial-case-351017-early.png"
    figure.savefig(output, dpi=145, facecolor="white")
    plt.close(figure)
    script_path = Path(__file__).resolve()
    receipt = {
        "schema": "small_owner_repeat_origin.continuation_visualization_prototype.v1",
        "status": "smoke_partial_not_full_manifest",
        "inputs": {
            "packet": {"path": str(packet_path.resolve()), "sha256": sha256_file(packet_path)},
            "reduction": {"path": str(reduction_path.resolve()), "sha256": sha256_file(reduction_path)},
            "renderer": {"path": str(script_path), "sha256": sha256_file(script_path)},
        },
        "case_id": case_id,
        "boundary": "early",
        "images": image_receipts,
        "cells": cells,
        "figure": {"path": str(output), "sha256": sha256_file(output), "size_bytes": output.stat().st_size},
        "claim_boundary": "renderer prototype only; not the full eight-case visualization package",
    }
    (out_dir / "prototype-receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")


def render_full(
    *,
    packet: dict[str, Any],
    reduction: dict[str, Any],
    packet_path: Path,
    reduction_path: Path,
    out_dir: Path,
) -> None:
    if reduction.get("schema") != "small_owner_repeat_origin.reduction.v1":
        raise ValueError("unexpected reduction schema")
    if reduction.get("scope") != "full_panel" or int(reduction.get("cells", -1)) != 74:
        raise ValueError("full rendering requires the complete 74-cell full_panel reduction")
    if len(packet["cases"]) != 8:
        raise ValueError("full rendering requires eight packet cases")
    indexed = index_results(reduction)
    required = {
        (str(case["case_id"]), expected_job_id(boundary, image, history))
        for case in packet["cases"]
        for boundary in BOUNDARIES
        for image, history in CELL_ORDER
    }
    if not required.issubset(indexed):
        missing = sorted(required - set(indexed))
        raise ValueError(f"reduction lacks factorial cells: {missing}")

    receipts = [render_full_case(packet_case=case, indexed=indexed, out_dir=out_dir) for case in packet["cases"]]
    script_path = Path(__file__).resolve()
    manifest = {
        "schema": "small_owner_repeat_origin.continuation_visualizations.v1",
        "status": "complete_8_case_64_factorial_cell_package",
        "scope": {
            "reduction_cell_count": 74,
            "rendered_case_count": 8,
            "rendered_factorial_cell_count": 64,
            "rows_per_case": ["early", "late"],
            "columns_per_case": [f"{image}_{history}" for image, history in CELL_ORDER],
            "generated_overlay": "literal result.first5_rows only",
            "forced_seed_overlay": "dotted context only, never labeled as generated detection",
            "no_model_calls": True,
            "no_gt_or_owner_truth_adjudication": True,
        },
        "inputs": {
            "packet": {"path": str(packet_path.resolve()), "sha256": sha256_file(packet_path)},
            "reduction": {"path": str(reduction_path.resolve()), "sha256": sha256_file(reduction_path)},
            "renderer": {"path": str(script_path), "sha256": sha256_file(script_path)},
        },
        "cases": receipts,
        "figure_paths": [receipt["figure"]["path"] for receipt in receipts],
        "acceptance": {
            "figure_count": len(receipts),
            "rendered_cell_count": sum(len(receipt["cells"]) for receipt in receipts),
            "all_packet_image_hashes_and_dimensions_match": True,
            "all_first5_rows_rendered_from_literal_bins": True,
            "geometry_invalid_rows_preserved_as_lines": True,
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, default=DEFAULT_PACKET)
    parser.add_argument("--reduction", type=Path)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--prototype", action="store_true")
    args = parser.parse_args()
    reduction_path = args.reduction or (DEFAULT_SMOKE if args.prototype else DEFAULT_REDUCTION)
    packet = json.loads(args.packet.read_text())
    reduction = json.loads(reduction_path.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.prototype:
        render_prototype(
            packet=packet,
            reduction=reduction,
            packet_path=args.packet,
            reduction_path=reduction_path,
            out_dir=args.out_dir,
        )
    else:
        render_full(
            packet=packet,
            reduction=reduction,
            packet_path=args.packet,
            reduction_path=reduction_path,
            out_dir=args.out_dir,
        )


if __name__ == "__main__":
    main()
