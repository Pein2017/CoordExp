"""CPU-only independent reduction for the spatial source probe."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image
from transformers import AutoTokenizer


EOS = 151645
OBJ_START = 151646
OBJ_END = 151647
COORD_BASE = 151670
COORD_LIMIT = 152670
BASE = Path("/data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent")
OUT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source")
DEFAULT_MODEL = "tied"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def binding(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": sha256(path), "size_bytes": path.stat().st_size}


def inverse(value: int, *, source_width: int, canvas_width: int, tx: int) -> int:
    canvas_pixel = value * (canvas_width - 1) / 999.0
    source_pixel = canvas_pixel - tx
    return int(round(source_pixel * 999.0 / (source_width - 1)))


def iou(left: list[int], right: list[int]) -> float:
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    intersection = max(0, min(lx2, rx2) - max(lx1, rx1)) * max(0, min(ly2, ry2) - max(ly1, ry1))
    union = (lx2 - lx1) * (ly2 - ly1) + (rx2 - rx1) * (ry2 - ry1) - intersection
    return 0.0 if union <= 0 else intersection / union


def parse(token_ids: list[int], *, tokenizer: Any, tx: int, source_width: int, canvas_width: int) -> dict[str, Any]:
    starts = [index for index, token in enumerate(token_ids) if token == OBJ_START]
    rows: list[dict[str, Any]] = []
    for row_index, start in enumerate(starts):
        stop = starts[row_index + 1] if row_index + 1 < len(starts) else len(token_ids)
        segment = token_ids[start:stop]
        positions = [index for index, token in enumerate(segment) if COORD_BASE <= token < COORD_LIMIT]
        if len(positions) != 4 or OBJ_END not in segment:
            rows.append({"row_index": row_index, "status": "malformed", "token_count": len(segment)})
            continue
        bins = [segment[index] - COORD_BASE for index in positions]
        source_bins = [inverse(value, source_width=source_width, canvas_width=canvas_width, tx=tx) for value in bins]
        description = tokenizer.decode(segment[1 : segment.index(OBJ_END)], skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
        source_in_bounds = all(0 <= value <= 999 for value in source_bins)
        rows.append(
            {
                "row_index": row_index,
                "status": "valid" if bins[0] < bins[2] and bins[1] < bins[3] else "invalid",
                "description": description,
                "coord_bins_canvas": bins,
                "coord_bins_source": source_bins,
                "source_in_bounds": source_in_bounds,
                "canvas_border": any(value in (0, 999) for value in bins),
                "source_geometry_valid": bins[0] < bins[2] and bins[1] < bins[3] and source_in_bounds and source_bins[0] < source_bins[2] and source_bins[1] < source_bins[3],
                "token_count": len(segment),
            }
        )
    exact = runs(rows, near=False)
    near = runs(rows, near=True)
    return {
        "complete_rows": len(rows),
        "valid_rows": sum(row["status"] == "valid" for row in rows),
        "invalid_rows": sum(row["status"] == "invalid" for row in rows),
        "malformed_rows": sum(row["status"] == "malformed" for row in rows),
        "exact_runs": exact,
        "near_runs": near,
        "failure_predicate": bool(exact or near),
        "rows": rows,
    }


def runs(rows: list[dict[str, Any]], *, near: bool) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    for row in rows + [{"status": "boundary"}]:
        if row.get("status") != "valid" or not row.get("source_geometry_valid", False):
            if len(current) >= 3:
                found.append({"start_row": current[0]["row_index"], "length": len(current), "description": current[0]["description"], "coord_bins_source": current[0]["coord_bins_source"]})
            current = []
            continue
        if not current:
            current = [row]
            continue
        same_desc = row["description"] == current[-1]["description"]
        same_box = row["coord_bins_source"] == current[-1]["coord_bins_source"]
        close_box = all(abs(a - b) <= 8 for a, b in zip(row["coord_bins_source"], current[-1]["coord_bins_source"], strict=True))
        if same_desc and (same_box if not near else close_box):
            current.append(row)
        else:
            if len(current) >= 3:
                found.append({"start_row": current[0]["row_index"], "length": len(current), "description": current[0]["description"], "coord_bins_source": current[0]["coord_bins_source"]})
            current = [row]
    return found


def known(rows: list[dict[str, Any]], bank: list[dict[str, Any]]) -> dict[str, Any]:
    choices = []
    for row in rows:
        if row.get("status") != "valid" or not row.get("source_geometry_valid", False):
            continue
        for target in bank:
            if row["description"].lower() == str(target["normalized_description"]).lower():
                choices.append((iou(row["coord_bins_source"], list(target["reference_coord_bins_1000"])), row, target))
    choices.sort(key=lambda item: item[0], reverse=True)
    used_rows: set[int] = set()
    used_owners: set[str] = set()
    matches = []
    for score, row, target in choices:
        owner = str(target["owner_id"])
        if score < 0.5 or row["row_index"] in used_rows or owner in used_owners:
            continue
        used_rows.add(row["row_index"])
        used_owners.add(owner)
        matches.append({"row_index": row["row_index"], "owner_id": owner, "iou": score})
    return {"bank_count": len(bank), "matched_count": len(matches), "matches": matches}


def bank_for(panel: dict[str, Any], *, image_id: int, split: str | None = None) -> tuple[list[dict[str, Any]], str | None]:
    keys = [str(image_id)]
    if split:
        keys.insert(0, f"{split}:{image_id}")
    for name in ("refined_banks", "new_banks", "sentinel_banks"):
        banks = panel.get(name)
        if not isinstance(banks, dict):
            continue
        for key in keys:
            if key in banks:
                return list(banks[key]), f"{name}[{key}]"
    return [], None


def reduce(
    out: Path = OUT,
    model_name: str = DEFAULT_MODEL,
    runtime_path: Path | None = None,
    reduced_path: Path | None = None,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    if manifest_path is None:
        manifest_path = out / f"transform-manifest-{model_name}.json"
    manifest_path = manifest_path.resolve(strict=True)
    if runtime_path is None:
        runtime_path = out / f"runtime-result-{model_name}.json"
    if reduced_path is None:
        reduced_path = out / f"reduced-{model_name}.json"
    manifest = json.loads(manifest_path.read_text())
    runtime = json.loads(runtime_path.read_text())
    if manifest["source"]["model"] != model_name or runtime.get("model") != model_name:
        raise AssertionError("manifest/runtime model identity drift")
    panel_path = Path(manifest["source"]["mature_panel"]["path"])
    panel = json.loads(panel_path.read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(BASE), use_fast=False)
    geometry = manifest["geometry"]
    source_width = int(geometry["source_width"])
    canvas_width = int(geometry["canvas_width"])
    image_id = str(manifest["source"]["image_id"])
    bank, bank_source = bank_for(
        panel,
        image_id=int(image_id),
        split=manifest["source"].get("split"),
    )
    cells: dict[str, Any] = {}
    for key, cell_result in runtime.get("cells", {}).items():
        cell = manifest["cells"][key]
        token_ids = [int(value) for value in cell_result["free"]["token_ids"]]
        parsed = parse(token_ids, tokenizer=tokenizer, tx=int(cell["visual_offset_px"]), source_width=source_width, canvas_width=canvas_width)
        matched = known(parsed["rows"], bank)
        if parsed["complete_rows"] != int(cell_result["free"]["parse"]["complete_rows"]):
            raise AssertionError(f"row count mismatch in {key}")
        cells[key] = {
            "token_count": len(token_ids),
            "stop_reason": cell_result["free"]["stop_reason"],
            "parse": parsed,
            "known": matched,
            "known_bank_source": bank_source,
            "input_identity": cell_result["input_identity"],
        }
        with Image.open(cell["image_path"]) as image:
            if image.size != (canvas_width, int(geometry["canvas_height"])):
                raise AssertionError(f"canvas size drift in {key}")
        if binding(Path(cell["image_path"])) != cell["image"]:
            raise AssertionError(f"image binding drift in {key}")
    result = {
        "schema": "recurrence_spatial_source_reduction.v1",
        "status": runtime["status"],
        "runtime": {
            "model_forwards": runtime.get("model_forwards"),
            "vision_forwards": runtime.get("vision_forwards"),
            "elapsed_seconds": runtime.get("elapsed_seconds"),
            "gpu_seconds": None,
            "no_parameter_mutation": runtime.get("no_parameter_mutation"),
        },
        "source": {
            "panel": binding(panel_path),
            "transform_manifest": binding(manifest_path),
            "runtime_result": binding(runtime_path),
            "mature_raw": manifest["source"]["mature_raw"],
            "feedback_selection": manifest["source"].get("feedback_selection"),
            "feedback_selection_id": manifest["source"].get("feedback_selection_id"),
        },
        "admission": runtime.get("admission"),
        "cells": cells,
        "interpretation": (
            "centered 00 passed the proxy valid-row gate without a recurrence requirement; transformed cells are control trajectory evidence and require root acceptance"
            if (runtime.get("admission") or {}).get("admitted", False)
            and (runtime.get("admission") or {}).get("mode") == "proxy"
            else "centered 00 retained the frozen recurrence predicate; transformed cells are candidate trajectory evidence and require root acceptance"
            if (runtime.get("admission") or {}).get("admitted", False)
            else (
                "this receipt contains only transform cells; pair it with the separately admitted centered 00 receipt before interpretation"
                if runtime.get("admission") is None
                else "centered 00 failed the frozen recurrence admission predicate; transformed cells are not interpreted"
            )
        ),
        "model": model_name,
        "image_id": int(image_id),
        "group": manifest["source"]["group"],
    }
    reduced_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--model", choices=("tied", "untied"), default=DEFAULT_MODEL)
    parser.add_argument("--runtime-path", type=Path)
    parser.add_argument("--reduced-path", type=Path)
    parser.add_argument("--manifest-path", type=Path)
    args = parser.parse_args()
    print(json.dumps(reduce(args.out, args.model, args.runtime_path, args.reduced_path, args.manifest_path), indent=2))
