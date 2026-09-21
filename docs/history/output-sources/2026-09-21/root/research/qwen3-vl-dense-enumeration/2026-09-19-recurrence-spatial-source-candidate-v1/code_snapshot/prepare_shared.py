"""Prepare the frozen seven-cell manifests for the final shared panel.

This is a bounded CPU preparation step.  It resolves the already frozen
state-entry receipt, rebuilds only the transformed PNGs and literal history
prefixes for states that are not covered by the exact 417044 pilot, and writes
one producer-compatible manifest per state.  It does not load a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image

from probes.training_set_completion.recurrence_spatial.prepare import (
    FILL_RGB,
    SHIFT_PX,
    transform_history,
)
from probes.training_set_completion.readout_norm_fresh import _binding
from probes.training_set_completion.recurrence_spatial.state_entry import digest_json


OUT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source")
PANEL = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-distribution-census/shared-panel.json")
READINESS = OUT / "generic-entry-readiness.json"
SNAPSHOT = OUT / "final" / "source-snapshot.json"
GRID = 32
COORD_BASE = 151670
OBJ_START = 151646
CELL_KEYS = ("00", "10-", "10+", "01-", "01+", "11-", "11+")
PILOT_MANIFESTS = {
    "tied-417044-failure": OUT / "transform-manifest-tied.json",
    "untied-417044-failure": OUT / "transform-manifest-untied.json",
}
PILOT_RESULTS = {
    "tied-417044-failure": {
        "center": OUT / "runtime-result-tied.json",
        "transforms": OUT / "runtime-result-tied-transforms.json",
        "reduced_center": OUT / "reduced-tied-00.json",
        "reduced_transforms": OUT / "reduced-tied-transforms.json",
    },
    "untied-417044-failure": {
        "center": OUT / "runtime-result-untied.json",
        "transforms": OUT / "runtime-result-untied-transforms.json",
        "reduced_center": OUT / "reduced-untied-00.json",
        "reduced_transforms": OUT / "reduced-untied-transforms.json",
    },
}


def bind(path: Path) -> dict[str, Any]:
    return _binding(path.resolve(strict=True))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _raw_tokens(state: dict[str, Any]) -> tuple[list[int], list[int], dict[str, Any]]:
    raw_path = Path(state["source"]["raw"]["path"])
    raw = _load_json(raw_path)
    image_id = int(state["source"]["image_id"])
    rows = [row for row in raw["rows"] if int(row["image_id"]) == image_id]
    if len(rows) != 1:
        raise ValueError(f"expected one raw row for {state['id']}, got {len(rows)}")
    row = rows[0]
    tokens = [int(token) for token in row["token_ids"]]
    starts = [index for index, token in enumerate(tokens) if token == OBJ_START]
    prefix_end = int(state["prefix"]["source_row_end"])
    prefix = tokens[:prefix_end]
    if digest_json(tokens) != state["prefix"]["native_token_sha256"]:
        raise ValueError(f"native token hash drift for {state['id']}")
    if digest_json(prefix) != state["prefix"]["source_sha256"]:
        raise ValueError(f"prefix hash drift for {state['id']}")
    index = int(state["prefix"]["source_row_index"])
    if starts[index] != int(state["prefix"]["source_row_start"]):
        raise ValueError(f"source row start drift for {state['id']}")
    if index + 1 >= len(starts):
        raise ValueError(f"selected boundary has no next row for {state['id']}")
    if starts[index + 1] != prefix_end:
        raise ValueError(f"source row end drift for {state['id']}")
    return tokens, starts, row


def _case(panel: dict[str, Any], state: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    wanted_group = state["source"]["group"]
    groups = [group for group in panel["groups"] if str(group["key"]) == wanted_group]
    if len(groups) != 1:
        raise ValueError(f"expected one group {wanted_group!r} in {state['source']['mature_panel']['path']}")
    group = groups[0]
    image_id = int(state["source"]["image_id"])
    cases = [case for case in group["cases"] if int(case["input_record"]["image_id"]) == image_id]
    if len(cases) != 1:
        raise ValueError(f"expected one case {image_id} in {wanted_group!r}")
    target = dict(cases[0])
    # The new-cohort panel intentionally stores only the frozen input row.
    # Reconstruct the already-qualified image plan from that group's saved
    # receipt; this is source binding, not a new processor qualification.
    if "image_plan" not in target or "image_path" not in target:
        input_record = dict(target["input_record"])
        image_path = Path(str(input_record["images"][0]))
        if not image_path.is_absolute():
            image_path = Path(str(group["input_jsonl"])).resolve().parent / image_path
        image_path = image_path.resolve(strict=True)
        receipt_path = Path(state["source"]["raw"]["path"]).resolve().parent / "receipt.json"
        receipt = _load_json(receipt_path)
        identity = receipt.get("input_identity", {})
        request_ids = list(identity.get("request_ids", []))
        row_id = str(target["row_id"])
        if row_id not in request_ids:
            raise ValueError(f"new-cohort receipt has no target request {row_id}")
        request_index = request_ids.index(row_id)
        grids = identity.get("image_grids", [])
        media = identity.get("media_sha256", [])
        prompts = identity.get("prompt_token_ids", [])
        if request_index >= len(grids) or request_index >= len(media) or request_index >= len(prompts):
            raise ValueError(f"new-cohort receipt identity is incomplete for {row_id}")
        grid = [int(value) for value in grids[request_index]]
        if len(grid) != 3 or grid[0] != 1 or (grid[1] * grid[2]) % 4:
            raise ValueError(f"unsupported saved image grid for {row_id}: {grid}")
        target["image_path"] = str(image_path)
        target["image_width"] = int(input_record["width"])
        target["image_height"] = int(input_record["height"])
        target["image_plan"] = {
            "backend_prompt_token_count": len(prompts[request_index]),
            "image_content_sha256": str(media[request_index]),
            "logical_transform_id": "identity",
            "merged_visual_tokens": (grid[1] * grid[2]) // 4,
            "observed_image_grid_thw": grid,
            "source_receipt": bind(receipt_path),
        }
    local_index = group["cases"].index(cases[0])
    expected = state["source"].get("batch_index")
    if expected is not None and local_index != int(expected):
        raise ValueError(f"case local index drift for {state['id']}: {local_index} != {expected}")
    return group, target


def _transform_image(source: Path, *, offset: int, canvas_width: int, out: Path) -> Path:
    with Image.open(source) as opened:
        image = opened.convert("RGB")
        if image.width + 2 * SHIFT_PX != canvas_width:
            raise ValueError("canvas width drift")
        canvas = Image.new("RGB", (canvas_width, image.height), FILL_RGB)
        canvas.paste(image, (offset, 0))
        out.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(out, format="PNG", optimize=False, compress_level=6)
        if canvas.crop((offset, 0, offset + image.width, image.height)).tobytes() != image.tobytes():
            raise AssertionError("source crop changed during transform")
    return out


def _cell_specs() -> dict[str, tuple[int, int, int]]:
    # (visual offset, history offset, relative history shift)
    return {
        "00": (SHIFT_PX, SHIFT_PX, 0),
        "10-": (0, SHIFT_PX, 0),
        "10+": (2 * SHIFT_PX, SHIFT_PX, 0),
        "01-": (SHIFT_PX, 0, -SHIFT_PX),
        "01+": (SHIFT_PX, 2 * SHIFT_PX, SHIFT_PX),
        "11-": (0, 0, -SHIFT_PX),
        "11+": (2 * SHIFT_PX, 2 * SHIFT_PX, SHIFT_PX),
    }


def _manifest_for_state(
    state: dict[str, Any],
    *,
    shared_panel_binding: dict[str, Any],
    readiness_binding: dict[str, Any],
    snapshot_binding: dict[str, Any],
    manifest_path: Path,
    image_root: Path,
) -> dict[str, Any]:
    panel_path = Path(state["source"]["mature_panel"]["path"])
    panel = _load_json(panel_path)
    group, case = _case(panel, state)
    source_image = Path(state["source"]["image"]["path"])
    with Image.open(source_image) as image:
        source_width, source_height = image.size
    canvas_width = source_width + 2 * SHIFT_PX
    if any(value % GRID for value in (source_width, source_height, canvas_width)):
        raise ValueError(f"source/canvas dimensions are not processor-grid aligned for {state['id']}")
    tokens, starts, saved_row = _raw_tokens(state)
    prefix_end = int(state["prefix"]["source_row_end"])
    prefix = tokens[:prefix_end]
    source_row_index = int(state["prefix"]["source_row_index"])
    cells: dict[str, dict[str, Any]] = {}
    image_by_offset: dict[int, Path] = {}
    for key, (visual_offset, history_offset, relative_shift) in _cell_specs().items():
        image_path = image_by_offset.get(visual_offset)
        if image_path is None:
            image_path = image_root / f"image-x{visual_offset}.png"
            _transform_image(source_image, offset=visual_offset, canvas_width=canvas_width, out=image_path)
            image_by_offset[visual_offset] = image_path
        history, boxes = transform_history(
            prefix,
            source_width=source_width,
            canvas_width=canvas_width,
            tx=history_offset,
        )
        cells[key] = {
            "key": key,
            "image_path": str(image_path),
            "image": bind(image_path),
            "visual_offset_px": visual_offset,
            "history_tx_relative_px": relative_shift,
            "history_offset_px": history_offset,
            "history": history,
            "history_sha256": digest_json(history),
            "history_boxes": boxes,
        }
    if set(cells) != set(CELL_KEYS) or len({len(cell["history"]) for cell in cells.values()}) != 1:
        raise AssertionError(f"cell/history shape drift for {state['id']}")
    source_prefix_rows = source_row_index + 1
    source_start = int(state["prefix"]["source_row_start"])
    next_row_start = starts[source_row_index + 1]
    next_row_end = starts[source_row_index + 2] if source_row_index + 2 < len(starts) else len(tokens)
    raw_binding = bind(Path(state["source"]["raw"]["path"]))
    state_record = {
        "id": state["id"],
        "model": state["model"],
        "condition": state.get("condition"),
        "kind": state.get("kind"),
        "source": state["source"],
        "prefix": state["prefix"],
    }
    feedback_selection = Path(
        "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-numerical-recurrence-feedback/selection.json"
    )
    source = {
        "mature_panel": bind(panel_path),
        "mature_raw": raw_binding,
        "shared_panel": shared_panel_binding,
        "state_entry": readiness_binding,
        "source_snapshot": snapshot_binding,
        "feedback_selection": bind(feedback_selection) if feedback_selection.exists() else None,
        "feedback_selection_id": state.get("id") if feedback_selection.exists() else None,
        "state_record_sha256": digest_json(state_record),
        "group": str(group["key"]),
        "image_id": int(state["source"]["image_id"]),
        "batch_index": state["source"].get("batch_index"),
        "split": state["source"].get("split"),
        "model": state["model"],
        "kind": state.get("kind"),
        "condition": state.get("condition"),
        "case": case,
        "source_image": bind(source_image),
    }
    manifest = {
        "schema": "recurrence_spatial_source.final_state.v1",
        "unit_id": "2026-09-19-recurrence-spatial-source",
        "state_id": state["id"],
        "source": source,
        "native_route": state["native_route"],
        "geometry": {
            "operation": "lossless_copy_into_common_canvas",
            "source_width": source_width,
            "source_height": source_height,
            "canvas_width": canvas_width,
            "canvas_height": source_height,
            "shift_px": SHIFT_PX,
            "grid_cell_px": GRID,
            "fill_rgb": list(FILL_RGB),
            "rounding": "round-half-even",
            "pixel_coordinate_domain": "[0,width-1]",
            "no_resize": True,
        },
        "prefix": {
            "rows": source_prefix_rows,
            "source_row_index": source_row_index,
            "source_row_start": source_start,
            "source_row_end": prefix_end,
            "source_token_count": len(prefix),
            "source_sha256": digest_json(prefix),
            "native_token_count": len(tokens),
            "native_token_sha256": digest_json(tokens),
            "source_stop": saved_row.get("stop"),
            "source_row_starts": starts[: source_prefix_rows + 1],
            "next_row_index": source_row_index + 1,
            "next_row_start": next_row_start,
            "next_row_end": next_row_end,
            "mapped_token_count_by_cell": {key: len(value["history"]) for key, value in cells.items()},
        },
        "cells": cells,
        "bindings": {
            "script": bind(Path(__file__)),
            "shared_panel": shared_panel_binding,
            "state_entry": readiness_binding,
            "source_snapshot": snapshot_binding,
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def prepare_shared(
    *,
    out: Path = OUT,
    panel_path: Path = PANEL,
    readiness_path: Path = READINESS,
    snapshot_path: Path = SNAPSHOT,
) -> dict[str, Any]:
    out = out.resolve()
    panel_path = panel_path.resolve(strict=True)
    readiness_path = readiness_path.resolve(strict=True)
    snapshot_path = snapshot_path.resolve(strict=True)
    panel_binding = bind(panel_path)
    readiness_binding = bind(readiness_path)
    snapshot = _load_json(snapshot_path)
    if snapshot["panel"] != panel_binding:
        raise ValueError("final panel changed after source snapshot")
    readiness = _load_json(readiness_path)
    if readiness.get("status") != "ready" or readiness.get("resolved_state_count") != 45:
        raise ValueError("final CPU readiness receipt is not 45/45 ready")
    states = readiness["states"]
    manifests = out / "final" / "manifests"
    images = out / "final" / "images"
    index: dict[str, Any] = {
        "schema": "recurrence_spatial.final_execution_index.v1",
        "unit_id": "2026-09-19-recurrence-spatial-source",
        "panel": panel_binding,
        "readiness": readiness_binding,
        "source_snapshot": bind(snapshot_path),
        "declared_states": len(states),
        "states": [],
        "new_manifest_count": 0,
        "pilot_reuse_count": 0,
        "new_cell_count": 0,
        "reused_cell_count": 0,
    }
    for state in states:
        state_id = state["id"]
        if state_id in PILOT_MANIFESTS:
            manifest = PILOT_MANIFESTS[state_id]
            if not manifest.exists():
                raise FileNotFoundError(manifest)
            pilot = _load_json(manifest)
            if pilot["source"]["model"] != state["model"] or pilot["source"]["image_id"] != state["source"]["image_id"]:
                raise ValueError(f"pilot source identity drift for {state_id}")
            if pilot["prefix"]["source_sha256"] != state["prefix"]["source_sha256"]:
                raise ValueError(f"pilot prefix drift for {state_id}")
            entry = {
                "id": state_id,
                "model": state["model"],
                "kind": state.get("kind"),
                "mode": "reuse_pilot",
                "manifest": bind(manifest),
                "runtime": {key: bind(path) for key, path in PILOT_RESULTS[state_id].items()},
                "cell_keys": list(CELL_KEYS),
                "cell_count": 7,
                "prefix_sha256": state["prefix"]["source_sha256"],
            }
            index["pilot_reuse_count"] += 1
            index["reused_cell_count"] += 7
        else:
            state_dir = images / state_id
            manifest_path = manifests / f"{state_id}.json"
            _manifest_for_state(
                state,
                shared_panel_binding=panel_binding,
                readiness_binding=readiness_binding,
                snapshot_binding=bind(snapshot_path),
                manifest_path=manifest_path,
                image_root=state_dir,
            )
            entry = {
                "id": state_id,
                "model": state["model"],
                "kind": state.get("kind"),
                "mode": "new_native",
                "manifest": bind(manifest_path),
                "runtime_path": str(out / "final" / "runtime" / f"{state_id}.json"),
                "reduced_path": str(out / "final" / "reduced" / f"{state_id}.json"),
                "cell_keys": list(CELL_KEYS),
                "cell_count": 7,
                "prefix_sha256": state["prefix"]["source_sha256"],
            }
            index["new_manifest_count"] += 1
            index["new_cell_count"] += 7
        index["states"].append(entry)
    index_path = out / "final" / "execution-index.json"
    index_path.write_text(json.dumps(index, indent=2) + "\n")
    index["index"] = bind(index_path)
    index_path.write_text(json.dumps(index, indent=2) + "\n")
    return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--panel", type=Path, default=PANEL)
    parser.add_argument("--readiness", type=Path, default=READINESS)
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    args = parser.parse_args()
    print(json.dumps(prepare_shared(out=args.out, panel_path=args.panel, readiness_path=args.readiness, snapshot_path=args.snapshot), indent=2))
