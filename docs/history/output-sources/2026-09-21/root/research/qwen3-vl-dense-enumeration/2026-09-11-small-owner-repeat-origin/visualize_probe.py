#!/usr/bin/env python3
"""Render static onset figures for the frozen small-owner repeat probe.

The figures show raw generated geometry only.  They do not use annotations,
infer visual object identity, or recompute repetition from pair overlap.
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
DEFAULT_PACKET = ROOT / "packet-draft.json"
DEFAULT_CENSUS = ROOT / "census/census.json"
DEFAULT_OUT = ROOT / "visualizations"

COLORS = {
    "first": "#00B8D9",
    "early": "#FFB000",
    "late": "#D81B60",
    "invalid": "#D73027",
}
COORD_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
COORD_LABELS = ["x1", "y1", "x2", "y2"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bins_to_pixels(bins: list[int], width: int, height: int) -> list[int]:
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
    label: str,
    valid: bool,
    linewidth: float,
    linestyle: str = "-",
) -> None:
    x1, y1, x2, y2 = pixels
    if x1 == x2 or y1 == y2:
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=max(3.5, linewidth), linestyle=linestyle)
        anchor_x, anchor_y = min(x1, x2), min(y1, y2)
    else:
        xs = [x1, x2, x2, x1, x1]
        ys = [y1, y1, y2, y2, y1]
        ax.plot(xs, ys, color=color, linewidth=linewidth, linestyle=linestyle)
        anchor_x, anchor_y = min(x1, x2), min(y1, y2)
    suffix = " (invalid line)" if not valid else ""
    ax.text(
        anchor_x + 4,
        anchor_y + 18,
        label + suffix,
        color="white",
        fontsize=9,
        fontweight="bold",
        bbox={"facecolor": color, "edgecolor": "white", "alpha": 0.86, "pad": 2},
    )


def verified_image(case: dict[str, Any], role: str) -> tuple[Image.Image, dict[str, Any]]:
    path = Path(case["image_path"])
    observed_hash = sha256_file(path)
    expected_hash = case["image_plan"]["image_content_sha256"]
    if observed_hash != expected_hash:
        raise ValueError(f"{role} image hash differs from packet for {path}")
    image = Image.open(path).convert("RGB")
    width, height = image.size
    if (width, height) != (int(case["image_width"]), int(case["image_height"])):
        raise ValueError(f"{role} image dimensions differ from packet for {path}")
    return image, {
        "path": str(path),
        "sha256": observed_hash,
        "width": width,
        "height": height,
    }


def source_first_row(census_case: dict[str, Any]) -> dict[str, Any] | None:
    pointer = census_case["onset_summary"]["first_valid_instance_of_repeat_description"]
    if pointer is None:
        return None
    row = census_case["raw_rows"][int(pointer["raw_row_index"])]
    if not row["geometry_valid"] or row["pixel_box_xyxy"] is None:
        raise ValueError("census first repeat-description row is not native-valid")
    return row


def render_trace(ax: Any, census_case: dict[str, Any], packet_case: dict[str, Any]) -> dict[str, Any]:
    complete = [
        row
        for row in census_case["raw_rows"]
        if row["complete_canonical_row"] and row["coord_bins"] is not None
    ][:64]
    indices = [int(row["raw_row_index"]) for row in complete]
    for coord_index, (label, color) in enumerate(zip(COORD_LABELS, COORD_COLORS)):
        values = [row["coord_bins"][coord_index] for row in complete]
        ax.plot(indices, values, marker="o", markersize=2.4, linewidth=1.15, color=color, label=label)

    invalid_indices = []
    for row in complete:
        if not row["geometry_valid"]:
            index = int(row["raw_row_index"])
            invalid_indices.append(index)
            ax.axvspan(index - 0.45, index + 0.45, color=COLORS["invalid"], alpha=0.12, zorder=0)
            ax.scatter(
                [index] * 4,
                row["coord_bins"],
                color=COLORS["invalid"],
                marker="x",
                s=28,
                linewidths=1.0,
                zorder=6,
            )

    first_repeat = census_case["onset_summary"]["first_strict_iou_gt_0_95_repeat"]
    if first_repeat is not None and first_repeat["raw_row_index"] in indices:
        ax.axvline(first_repeat["raw_row_index"], color=COLORS["first"], linewidth=1.4, linestyle=":")
    for name in ("early", "late"):
        row_index = int(packet_case["seeds"][name]["raw_row_index"])
        if row_index in indices:
            ax.axvline(
                row_index,
                color=COLORS[name],
                linewidth=1.3,
                linestyle="--" if name == "early" else "-.",
            )

    ax.set_ylim(-25, 1025)
    if indices:
        ax.set_xlim(min(indices) - 0.7, max(indices) + 0.7)
    ax.set_xlabel("complete raw row index (first 64)")
    ax.set_ylabel("coordinate bin")
    ax.set_title("Native first-64 complete-row coordinate trace; invalid rows shaded red")
    ax.grid(True, alpha=0.22, linewidth=0.6)
    ax.legend(ncol=4, loc="upper right", fontsize=8, framealpha=0.9)
    return {"shown_complete_row_indices": indices, "invalid_row_indices": invalid_indices}


def render_case(
    packet_case: dict[str, Any],
    census_case: dict[str, Any],
    out_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    source_image, source_receipt = verified_image(packet_case["source_case"], "source")
    donor_image, donor_receipt = verified_image(packet_case["donor_case"], "donor")
    if source_image.size != donor_image.size:
        raise ValueError("packet source/donor images must have identical native dimensions")
    width, height = source_image.size

    fig = plt.figure(figsize=(18, 10.2), constrained_layout=True)
    grid = fig.add_gridspec(3, 2, height_ratios=[0.32, 3.0, 1.45])
    ax_header = fig.add_subplot(grid[0, :])
    ax_source = fig.add_subplot(grid[1, 0])
    ax_donor = fig.add_subplot(grid[1, 1])
    ax_trace = fig.add_subplot(grid[2, :])
    ax_header.axis("off")
    ax_source.imshow(source_image)
    ax_donor.imshow(donor_image)
    for ax in (ax_source, ax_donor):
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.set_xticks([])
        ax.set_yticks([])

    first = source_first_row(census_case)
    if first is not None:
        draw_box_or_line(
            ax_source,
            first["pixel_box_xyxy"],
            color=COLORS["first"],
            label=f"1 first {first['description']} r{first['raw_row_index']}",
            valid=True,
            linewidth=2.8,
            linestyle=":",
        )

    for number, name in ((2, "early"), (3, "late")):
        seed = packet_case["seeds"][name]
        source_pixels = bins_to_pixels(seed["original_bins"], width, height)
        donor_pixels = bins_to_pixels(seed["translated_bins"], width, height)
        valid = bool(seed["geometry_valid"])
        draw_box_or_line(
            ax_source,
            source_pixels,
            color=COLORS[name] if valid else COLORS["invalid"],
            label=f"{number} {name} r{seed['raw_row_index']}",
            valid=valid,
            linewidth=2.4,
            linestyle="--" if name == "early" else "-.",
        )
        draw_box_or_line(
            ax_donor,
            donor_pixels,
            color=COLORS[name] if valid else COLORS["invalid"],
            label=f"{number} {name} translated",
            valid=valid,
            linewidth=2.4,
            linestyle="--" if name == "early" else "-.",
        )

    source_id = int(packet_case["source_case"]["input_record"]["image_id"])
    donor_id = int(packet_case["donor_case"]["input_record"]["image_id"])
    repeat_description = census_case["onset_summary"]["repeat_description"]
    if repeat_description is None:
        recurrence_text = "no strict recurrence in census"
    else:
        repeat_row = census_case["onset_summary"]["first_strict_iou_gt_0_95_repeat"]["raw_row_index"]
        recurrence_text = f"first strict class-blind recurrence r{repeat_row}; emitted desc={repeat_description}"
    ax_source.set_title(f"Source image {source_id}\n{recurrence_text}")
    ax_donor.set_title(f"Donor image {donor_id}\npacket-translated seed geometry")

    trace_receipt = render_trace(ax_trace, census_case, packet_case)
    legend_handles = [
        Line2D([0], [0], color=COLORS["first"], linewidth=2.8, linestyle=":", label="1 first repeat-description row"),
        Line2D([0], [0], color=COLORS["early"], linewidth=2.4, linestyle="--", label="2 early seed"),
        Line2D([0], [0], color=COLORS["late"], linewidth=2.4, linestyle="-.", label="3 late seed"),
        Line2D([0], [0], color=COLORS["invalid"], linewidth=3.5, label="geometry-invalid seed/row"),
    ]
    ax_header.set_title(
        f"Case {packet_case['case_id']}: frozen natural row onset | source {source_id} → donor {donor_id} | "
        "generated geometry only; GT/visual identity not adjudicated",
        fontsize=13,
        fontweight="bold",
        pad=1,
    )
    ax_header.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=4,
        frameon=True,
        fontsize=9,
    )

    out_path = out_dir / f"case-{packet_case['case_id']}-onset.png"
    fig.savefig(out_path, dpi=165, facecolor="white")
    plt.close(fig)
    return out_path, {
        "case_id": str(packet_case["case_id"]),
        "source_image_id": source_id,
        "donor_image_id": donor_id,
        "source_image": source_receipt,
        "donor_image": donor_receipt,
        "first_repeat_description_row_index": first["raw_row_index"] if first else None,
        "first_strict_repeat_row_index": (
            census_case["onset_summary"]["first_strict_iou_gt_0_95_repeat"]["raw_row_index"]
            if census_case["onset_summary"]["first_strict_iou_gt_0_95_repeat"]
            else None
        ),
        "early_seed": packet_case["seeds"]["early"],
        "late_seed": packet_case["seeds"]["late"],
        "trace": trace_receipt,
        "figure": {
            "path": str(out_path),
            "sha256": sha256_file(out_path),
            "size_bytes": out_path.stat().st_size,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, default=DEFAULT_PACKET)
    parser.add_argument("--census", type=Path, default=DEFAULT_CENSUS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    packet = json.loads(args.packet.read_text())
    census = json.loads(args.census.read_text())
    if len(packet["cases"]) != 8 or len(census["cases"]) != 8:
        raise ValueError("static package requires exactly eight packet and census cases")
    census_by_id = {str(case["image_id"]): case for case in census["cases"]}
    packet_ids = [str(case["case_id"]) for case in packet["cases"]]
    if packet_ids != [str(case["image_id"]) for case in census["cases"]]:
        raise ValueError("packet/census case order or identity differs")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    receipts = []
    for packet_case in packet["cases"]:
        case_id = str(packet_case["case_id"])
        out_path, receipt = render_case(packet_case, census_by_id[case_id], args.out_dir)
        if not out_path.is_file():
            raise ValueError(f"figure was not written: {out_path}")
        receipts.append(receipt)

    script_path = Path(__file__).resolve()
    manifest = {
        "schema": "small_owner_repeat_origin.static_visualizations.v1",
        "status": "complete_static_8_case_package",
        "scope": {
            "case_count": 8,
            "no_model_calls": True,
            "no_gt_adjudication": True,
            "no_visual_truth_claim": True,
            "repeat_binding": "first recurrence metadata is read from the frozen census strict class-blind IoU>0.95 result; the visualization does not infer repetition from plotted pair overlap",
            "trace_scope": "first 64 complete canonical original-output rows with four emitted coordinate bins; geometry-invalid rows retained and colored red",
        },
        "inputs": {
            "packet": {"path": str(args.packet.resolve()), "sha256": sha256_file(args.packet), "status": "draft_geometry_packet"},
            "census": {"path": str(args.census.resolve()), "sha256": sha256_file(args.census)},
            "renderer": {"path": str(script_path), "sha256": sha256_file(script_path)},
        },
        "cases": receipts,
        "figure_paths": [receipt["figure"]["path"] for receipt in receipts],
        "acceptance": {
            "figure_count": len(receipts),
            "all_source_and_donor_hashes_match_packet": True,
            "all_source_and_donor_dimensions_match": True,
            "all_packet_and_census_case_ids_match": True,
            "invalid_rows_retained_in_trace": True,
        },
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
