"""Freeze one same-scene spatial/history transform and its native prefix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image

from probes.training_set_completion.readout_norm_fresh import _binding


ROOT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-untied-highconfidence18-natural")
FEEDBACK_ROOT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-numerical-recurrence-feedback")
OUT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source")
PANEL = ROOT / "panel.json"
IMAGE_ID = 417044
GROUP = "refined-03"
DEFAULT_MODEL = "tied"
POLICY = "original"
SHIFT_PX = 128
FILL_RGB = (0, 0, 0)
COORD_BASE = 151670
COORD_LIMIT = COORD_BASE + 1000
OBJ_START = 151646
OBJ_END = 151647
BOX_START = 151648
BOX_END = 151649


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def round_even(value: float) -> int:
    return int(round(value))


def map_bin(value: int, *, source_width: int, canvas_width: int, tx: int) -> int:
    source_pixel = value * (source_width - 1) / 999.0
    canvas_pixel = source_pixel + tx
    mapped = round_even(canvas_pixel * 999.0 / (canvas_width - 1))
    if not 0 <= mapped <= 999:
        raise ValueError(f"mapped coordinate outside canvas: {value} -> {mapped}")
    return mapped


def inverse_bin(value: int, *, source_width: int, canvas_width: int, tx: int) -> int:
    canvas_pixel = value * (canvas_width - 1) / 999.0
    source_pixel = canvas_pixel - tx
    return round_even(source_pixel * 999.0 / (source_width - 1))


def map_coordinate(
    value: int,
    *,
    coordinate_index: int,
    source_width: int,
    source_height: int,
    canvas_width: int,
    canvas_height: int,
    tx: int,
    ty: int = 0,
) -> int:
    """Map x coordinates with x geometry and y coordinates with y geometry."""

    if coordinate_index in (0, 2):
        return map_bin(value, source_width=source_width, canvas_width=canvas_width, tx=tx)
    return map_bin(value, source_width=source_height, canvas_width=canvas_height, tx=ty)


def inverse_coordinate(
    value: int,
    *,
    coordinate_index: int,
    source_width: int,
    source_height: int,
    canvas_width: int,
    canvas_height: int,
    tx: int,
    ty: int = 0,
) -> int:
    if coordinate_index in (0, 2):
        return inverse_bin(value, source_width=source_width, canvas_width=canvas_width, tx=tx)
    return inverse_bin(value, source_width=source_height, canvas_width=canvas_height, tx=ty)


def row_starts(token_ids: list[int]) -> list[int]:
    starts = [index for index, token in enumerate(token_ids) if token == OBJ_START]
    if not starts:
        raise ValueError("saved output has no object rows")
    return starts


def transform_history(
    token_ids: list[int],
    *,
    source_width: int,
    canvas_width: int,
    tx: int,
    source_height: int | None = None,
    canvas_height: int | None = None,
    ty: int = 0,
) -> tuple[list[int], list[dict[str, Any]]]:
    if source_height is None:
        source_height = source_width
    if canvas_height is None:
        canvas_height = canvas_width
    starts = row_starts(token_ids)
    transformed = list(token_ids)
    boxes: list[dict[str, Any]] = []
    for row_index, start in enumerate(starts):
        stop = starts[row_index + 1] if row_index + 1 < len(starts) else len(token_ids)
        coordinates = [
            index for index in range(start, stop) if COORD_BASE <= token_ids[index] < COORD_LIMIT
        ]
        if len(coordinates) != 4:
            raise ValueError(f"row {row_index} has {len(coordinates)} coordinates")
        source = [token_ids[index] - COORD_BASE for index in coordinates]
        mapped = [
            map_coordinate(
                value,
                coordinate_index=coordinate_index,
                source_width=source_width,
                source_height=source_height,
                canvas_width=canvas_width,
                canvas_height=canvas_height,
                tx=tx,
                ty=ty,
            )
            for coordinate_index, value in enumerate(source)
        ]
        for index, value in zip(coordinates, mapped, strict=True):
            transformed[index] = COORD_BASE + value
        inverse = [
            inverse_coordinate(
                value,
                coordinate_index=coordinate_index,
                source_width=source_width,
                source_height=source_height,
                canvas_width=canvas_width,
                canvas_height=canvas_height,
                tx=tx,
                ty=ty,
            )
            for coordinate_index, value in enumerate(mapped)
        ]
        source_valid = source[0] < source[2] and source[1] < source[3]
        mapped_valid = mapped[0] < mapped[2] and mapped[1] < mapped[3]
        boxes.append(
            {
                "row_index": row_index,
                "source_bins": source,
                "mapped_bins": mapped,
                "inverse_bins": inverse,
                "max_inverse_drift": max(abs(a - b) for a, b in zip(source, inverse, strict=True)),
                "token_count": stop - start,
                "source_valid": source_valid,
                "mapped_valid": mapped_valid,
                "order_preserved": source_valid == mapped_valid,
                "validity_changed_by_rounding": source_valid != mapped_valid,
            }
        )
    return transformed, boxes


def transformed_image(source: Path, *, x_offset: int, canvas_width: int, out: Path) -> Path:
    with Image.open(source) as image:
        image = image.convert("RGB")
        canvas = Image.new("RGB", (canvas_width, image.height), FILL_RGB)
        canvas.paste(image, (x_offset, 0))
        target = out / "images" / f"image-{IMAGE_ID}-x{x_offset}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(target, format="PNG", optimize=False, compress_level=6)
        if canvas.crop((x_offset, 0, x_offset + image.width, image.height)).tobytes() != image.tobytes():
            raise AssertionError("source crop changed during canvas construction")
        return target


def select_case(panel: dict[str, Any]) -> dict[str, Any]:
    group = next(item for item in panel["groups"] if item["key"] == GROUP)
    matches = [case for case in group["cases"] if int(case["input_record"]["image_id"]) == IMAGE_ID]
    if len(matches) != 1:
        raise ValueError(f"expected one source case for {IMAGE_ID}, got {len(matches)}")
    return matches[0]


def _selection_record(model: str) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    if model not in {"tied", "untied"}:
        raise ValueError(f"unsupported model {model!r}")
    selection_path = FEEDBACK_ROOT / "selection.json"
    selection = json.loads(selection_path.read_text())
    selection_id = f"{model}-{IMAGE_ID}-failure"
    records = [record for record in selection["boundaries"] if record.get("id") == selection_id]
    if len(records) != 1:
        raise ValueError(f"expected one frozen selection record {selection_id}, got {len(records)}")
    record = records[0]
    if record.get("model") != model or int(record.get("image_id")) != IMAGE_ID or record.get("group") != GROUP:
        raise ValueError(f"selection record identity drift for {selection_id}")
    return selection_path, selection, record


def prepare(model: str = DEFAULT_MODEL, out: Path = OUT) -> dict[str, Any]:
    selection_path, selection, selected = _selection_record(model)
    mature_raw = ROOT / "runtime" / f"{model}-original" / GROUP / "raw.json"
    panel = json.loads(PANEL.read_text())
    case = select_case(panel)
    source_image = Path(case["image_path"]).resolve(strict=True)
    with Image.open(source_image) as image:
        source_width, source_height = image.size
    canvas_width = source_width + 2 * SHIFT_PX
    if source_width % 32 or source_height % 32 or canvas_width % 32:
        raise ValueError("source and common canvas must be divisible by the 32-pixel grid")
    saved = json.loads(mature_raw.read_text())
    saved_row = next(row for row in saved["rows"] if int(row["image_id"]) == IMAGE_ID)
    token_ids = [int(value) for value in saved_row["token_ids"]]
    starts = row_starts(token_ids)
    source_row_index = int(selected["source_row"]["index"])
    prefix_rows = source_row_index + 1
    if len(starts) <= source_row_index:
        raise ValueError("saved output is shorter than the frozen prefix")
    prefix_start = starts[source_row_index]
    prefix_end = starts[source_row_index + 1] if source_row_index + 1 < len(starts) else len(token_ids)
    prefix = token_ids[:prefix_end]
    if len(prefix) != prefix_end:
        raise AssertionError("prefix slicing drift")
    if prefix_start != int(selected["source_row"]["start"]):
        raise AssertionError(f"selected source row start drift: {prefix_start} != {selected['source_row']['start']}")
    if prefix_end != int(selected["source_row"]["end"]):
        raise AssertionError(f"selected source row end drift: {prefix_end} != {selected['source_row']['end']}")
    prefix_sha256 = digest_json(prefix)
    if prefix_sha256 != selected.get("prefix_hash"):
        raise AssertionError(f"selected prefix hash drift: {prefix_sha256} != {selected.get('prefix_hash')}")
    source_row_coords = [
        token_ids[offset] - COORD_BASE for offset in selected["source_row"]["coordinate_offsets"]
    ]
    if source_row_coords != list(selected["source_row"]["values"]):
        raise AssertionError(f"selected source row coordinate drift: {source_row_coords}")
    source_image_binding = _binding(source_image)
    cells: dict[str, dict[str, Any]] = {}
    for sign, visual_offset, history_relative_tx in (
        ("-", 0, -SHIFT_PX),
        ("+", 2 * SHIFT_PX, SHIFT_PX),
    ):
        key = f"11{sign}"
        image_path = transformed_image(source_image, x_offset=visual_offset, canvas_width=canvas_width, out=out)
        history_offset = SHIFT_PX + history_relative_tx
        history, boxes = transform_history(
            prefix,
            source_width=source_width,
            canvas_width=canvas_width,
            tx=history_offset,
            source_height=source_height,
            canvas_height=source_height,
        )
        cells[key] = {
            "key": key,
            "image_path": str(image_path),
            "image": _binding(image_path),
            "visual_offset_px": visual_offset,
            "history_tx_relative_px": history_relative_tx,
            "history_offset_px": history_offset,
            "history": history,
            "history_sha256": digest_json(history),
            "history_boxes": boxes,
        }
    center_image = transformed_image(source_image, x_offset=SHIFT_PX, canvas_width=canvas_width, out=out)
    for key, history_relative_tx in (("00", 0), ("01-", -SHIFT_PX), ("01+", SHIFT_PX)):
        history_offset = SHIFT_PX + history_relative_tx
        history, boxes = transform_history(
            prefix,
            source_width=source_width,
            canvas_width=canvas_width,
            tx=history_offset,
            source_height=source_height,
            canvas_height=source_height,
        )
        cells[key] = {
            "key": key,
            "image_path": str(center_image),
            "image": _binding(center_image),
            "visual_offset_px": SHIFT_PX,
            "history_tx_relative_px": history_relative_tx,
            "history_offset_px": history_offset,
            "history": history,
            "history_sha256": digest_json(history),
            "history_boxes": boxes,
        }
    # `10` shares centered history, while its image uses the chosen sign.
    for sign, offset in (("-", 0), ("+", 2 * SHIFT_PX)):
        key = f"10{sign}"
        image_path = transformed_image(source_image, x_offset=offset, canvas_width=canvas_width, out=out)
        cells[key] = {
            "key": key,
            "image_path": str(image_path),
            "image": _binding(image_path),
            "visual_offset_px": offset,
            "history_tx_relative_px": 0,
            "history_offset_px": SHIFT_PX,
            "history": list(prefix),
            "history_sha256": digest_json(prefix),
            "history_boxes": transform_history(
                prefix,
                source_width=source_width,
                canvas_width=canvas_width,
                tx=SHIFT_PX,
                source_height=source_height,
                canvas_height=source_height,
            )[1],
        }
    manifest = {
        "schema": "recurrence_spatial_source.v1",
        "unit_id": "2026-09-19-recurrence-spatial-source",
        "source": {
            "mature_panel": _binding(PANEL),
            "mature_raw": _binding(mature_raw),
            "feedback_selection": _binding(selection_path),
            "feedback_selection_id": selected["id"],
            "feedback_selection_record": selected,
            "feedback_panel_sha256": selection.get("panel_sha256"),
            "group": GROUP,
            "image_id": IMAGE_ID,
            "model": model,
            "policy": POLICY,
            "case": case,
            "source_image": source_image_binding,
        },
        "geometry": {
            "operation": "lossless_copy_into_common_canvas",
            "source_width": source_width,
            "source_height": source_height,
            "canvas_width": canvas_width,
            "canvas_height": source_height,
            "shift_px": SHIFT_PX,
            "grid_cell_px": 32,
            "fill_rgb": list(FILL_RGB),
            "rounding": "round-half-even",
            "pixel_coordinate_domain": "[0,width-1]",
            "no_resize": True,
        },
        "prefix": {
            "rows": prefix_rows,
            "source_row_index": source_row_index,
            "source_row_start": prefix_start,
            "source_row_end": prefix_end,
            "source_token_count": len(prefix),
            "source_sha256": prefix_sha256,
            "selection_prefix_sha256": selected["prefix_hash"],
            "source_stop": saved_row["stop"],
            "source_row_starts": starts[: prefix_rows + 1],
            "next_row_index": source_row_index + 1,
            "mapped_token_count_by_cell": {key: len(value["history"]) for key, value in cells.items()},
        },
        "cells": cells,
        "bindings": {"script": _binding(Path(__file__))},
    }
    if set(cells) != {"00", "10-", "10+", "01-", "01+", "11-", "11+"}:
        raise AssertionError("transform cell set drift")
    if len({len(value["history"]) for value in cells.values()}) != 1:
        raise AssertionError("history token count changed across cells")
    target = out / f"transform-manifest-{model}.json"
    target.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--model", choices=("tied", "untied"), default=DEFAULT_MODEL)
    args = parser.parse_args()
    result = prepare(args.model, args.out)
    print(json.dumps({"status": "prepared", "model": args.model, "selection_id": result["source"]["feedback_selection_id"], "cells": list(result["cells"]), "prefix_tokens": result["prefix"]["source_token_count"]}, indent=2))
