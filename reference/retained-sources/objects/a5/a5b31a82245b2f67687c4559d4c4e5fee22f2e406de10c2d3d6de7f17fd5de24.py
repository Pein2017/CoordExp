"""CPU-only independent reduction for the spatial source probe."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image
from transformers import AutoTokenizer

from probes.training_set_completion.recurrence_spatial.recurrence_semantics import (
    inverse_bin as _role_aware_inverse_bin,
    parse_rows as _role_aware_parse_rows,
    runs as _pairwise_runs,
)


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


def inverse(
    value: int,
    *,
    source_width: int,
    canvas_width: int,
    tx: int,
    coordinate_index: int = 0,
    source_height: int | None = None,
    canvas_height: int | None = None,
    ty: int = 0,
) -> int:
    if source_height is None:
        source_height = source_width
    if canvas_height is None:
        canvas_height = canvas_width
    return _role_aware_inverse_bin(
        value,
        coordinate_index=coordinate_index,
        source_width=source_width,
        source_height=source_height,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        tx=tx,
        ty=ty,
    )


def iou(left: list[int], right: list[int]) -> float:
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    intersection = max(0, min(lx2, rx2) - max(lx1, rx1)) * max(0, min(ly2, ry2) - max(ly1, ry1))
    union = (lx2 - lx1) * (ly2 - ly1) + (rx2 - rx1) * (ry2 - ry1) - intersection
    return 0.0 if union <= 0 else intersection / union


def parse(
    token_ids: list[int],
    *,
    tokenizer: Any,
    tx: int,
    source_width: int,
    canvas_width: int,
    source_height: int | None = None,
    canvas_height: int | None = None,
) -> dict[str, Any]:
    if source_height is None:
        source_height = source_width
    if canvas_height is None:
        canvas_height = canvas_width
    geometry = {
        "source_width": source_width,
        "source_height": source_height,
        "canvas_width": canvas_width,
        "canvas_height": canvas_height,
    }
    cell = {"visual_offset_px": tx}
    return _role_aware_parse_rows(token_ids, tokenizer, cell=cell, geometry=geometry)


def runs(rows: list[dict[str, Any]], *, near: bool) -> list[dict[str, Any]]:
    return _pairwise_runs(rows, near=near)


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


def compact_capture(capture: dict[str, Any] | None) -> dict[str, Any] | None:
    """Retain boundary competition without duplicating 1000-bin arrays."""

    if capture is None:
        return None
    fields = {key: capture[key] for key in ("chosen_token_id", "stop_reason", "input_width", "top10", "eos", "coordinate_family") if key in capture}
    return fields


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
        parsed = parse(
            token_ids,
            tokenizer=tokenizer,
            tx=int(cell["visual_offset_px"]),
            source_width=source_width,
            source_height=int(geometry["source_height"]),
            canvas_width=canvas_width,
            canvas_height=int(geometry["canvas_height"]),
        )
        matched = known(parsed["rows"], bank)
        runtime_parse = cell_result["free"].get("parse", {})
        parse_matches_runtime = parsed["complete_rows"] == int(runtime_parse.get("complete_rows", -1))
        cells[key] = {
            "token_count": len(token_ids),
            "stop_reason": cell_result["free"]["stop_reason"],
            "row_limit": cell_result["free"].get("row_limit"),
            "parse": parsed,
            "known": matched,
            "known_bank_source": bank_source,
            "input_identity": cell_result["input_identity"],
            "runtime_parse_compatibility": {
                "complete_rows_match": parse_matches_runtime,
                "runtime_complete_rows": runtime_parse.get("complete_rows"),
                "corrected_complete_rows": parsed["complete_rows"],
            },
            "boundary": {
                "opener": compact_capture(cell_result.get("boundary", {}).get("opener")),
                "forced_description_x1": compact_capture(cell_result.get("boundary", {}).get("forced_description_x1")),
                "windows": cell_result.get("boundary", {}).get("forced_description_x1", {}).get("windows"),
                "windows_by_sign": cell_result.get("boundary", {}).get("forced_description_x1", {}).get("windows_by_sign"),
                "window_selection": cell_result.get("boundary", {}).get("forced_description_x1", {}).get("window_selection"),
                "fixed_sign_status": (
                    "present"
                    if isinstance(cell_result.get("boundary", {}).get("forced_description_x1", {}).get("windows_by_sign"), dict)
                    else "absent_in_pre_correction_runtime"
                ),
            },
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
