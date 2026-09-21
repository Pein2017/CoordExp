"""Targeted current-token state replacement for the successful-row follow-up.

The caller keeps the existing native heterogeneous batch-4 generation path and
changes one target row at one action offset.  Baseline arms save the donor
states; later arms replace one selected decoder-layer output or the final
normalized lm-head input.  The runtime never edits model parameters or the
native image/cache preparation path.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from transformers import LogitsProcessor, LogitsProcessorList

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from probes.training_set_completion import repetition_history_runtime as native_runtime


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-17-successful-row-mechanism"
)
STAGE2_ROOT = ARTIFACT_ROOT / "stage2"
PREDECESSOR_ROOT = ARTIFACT_ROOT.parent / "2026-09-17-repetition-history-mechanism"
SELECTION_PATH = ARTIFACT_ROOT / "selection.json"

EOS = 151645
ROW_OPEN = 151646
REF_END = 151647
BOX_START = 151648
BOX_END = 151649
TARGET_IMAGE_ID = 309264
TARGET_POSITION = 1
COMMON_PREFIX_LENGTH = 54
RELEASE_OFFSET = 63
FORK_OFFSET = 67
MAX_NEW_TOKENS = 3084
DEPTHS = (6, 13, 20)
CAPTURE_OFFSETS = (63, 67, 68, 76, 103)
CACHE_OFFSETS = (67, 68)
EXPECTED_S_COORDS = (78, 609, 106, 641)
EXPECTED_F_COORDS = (1, 551, 36, 596)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _binding(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def _json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _tensor_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(
        value.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def _tokens(value: object, label: str) -> list[int]:
    if not isinstance(value, list) or any(
        isinstance(token, bool) or not isinstance(token, int) for token in value
    ):
        raise ValueError(f"{label} must be a list of integer token IDs")
    return list(value)


def _rows(value: object) -> dict[int, Mapping[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("raw rows must be a list")
    result: dict[int, Mapping[str, Any]] = {}
    for row in value:
        if not isinstance(row, Mapping):
            raise ValueError("raw row must be an object")
        image = native_runtime._image_id(row)
        if image in result:
            raise ValueError(f"duplicate raw image_id={image}")
        result[image] = row
    return result


def _row_tokens(row: Mapping[str, Any]) -> list[int]:
    return _tokens(row.get("token_ids", row.get("generated_token_ids")), "raw row")


def _row_stop(row: Mapping[str, Any]) -> str:
    value = row.get("stop", row.get("decode_stop_reason"))
    if not isinstance(value, str):
        raise ValueError("raw row has no stop reason")
    return value


def _check_binding(value: object, label: str) -> Path:
    if not isinstance(value, Mapping) or not isinstance(value.get("path"), str):
        raise ValueError(f"{label} binding is malformed")
    path = Path(value["path"])
    if _binding(path) != dict(value):
        raise AssertionError(f"{label} binding changed: {path}")
    return path


def _source_bindings() -> list[dict[str, Any]]:
    """Bind the actual installed Qwen/PEFT source used by this process."""
    import peft
    import transformers
    from transformers.models.qwen3_vl import modular_qwen3_vl, modeling_qwen3_vl

    paths: list[Path] = [
        Path(__file__).resolve(),
        Path(native_runtime.__file__).resolve(),
        Path(native_runtime.fresh.__file__).resolve(),
        Path(inspect.getsourcefile(transformers) or transformers.__file__).resolve(),
        Path(inspect.getsourcefile(modular_qwen3_vl) or modular_qwen3_vl.__file__).resolve(),
        Path(inspect.getsourcefile(modeling_qwen3_vl) or modeling_qwen3_vl.__file__).resolve(),
        Path(inspect.getsourcefile(peft.PeftModel) or peft.__file__).resolve(),
    ]
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        if path.is_file() and str(path) not in seen:
            result.append(_binding(path))
            seen.add(str(path))
    return result


def _selection_sources() -> list[dict[str, Any]]:
    selection = _read(SELECTION_PATH)
    sources = selection.get("sources")
    if not isinstance(sources, list):
        raise ValueError("selection has no source list")
    result = []
    for value in sources:
        path = _check_binding(value, "selection source")
        result.append(_binding(path))
    return result


def _source_raw_paths() -> tuple[Path, Path]:
    selection = _read(SELECTION_PATH)
    values = selection.get("sources")
    if not isinstance(values, list) or len(values) < 3:
        raise ValueError("selection does not bind S/F raw inputs")
    return _check_binding(values[1], "S raw"), _check_binding(values[2], "F raw")


def _selection_prefixes() -> tuple[list[int], list[int]]:
    selection = _read(SELECTION_PATH)
    common = _tokens(selection.get("common_prefix"), "selection common_prefix")
    if len(common) != COMMON_PREFIX_LENGTH:
        raise AssertionError("selection common prefix is not 54 tokens")
    pulse = selection.get("selected_tokens")
    if not isinstance(pulse, list) or len(pulse) < 2:
        raise ValueError("selection has no S/F selected rows")
    s_pulse = _tokens(pulse[0], "selection S row")
    f_pulse = _tokens(pulse[1], "selection F row")
    if len(s_pulse) != 9 or len(f_pulse) != 9:
        raise AssertionError("selected S/F rows must each contain nine tokens")
    s_prefix, f_prefix = common + s_pulse, common + f_pulse
    if _coordinate_values(s_pulse) != EXPECTED_S_COORDS:
        raise AssertionError("selection S coordinates changed")
    if _coordinate_values(f_pulse) != EXPECTED_F_COORDS:
        raise AssertionError("selection F coordinates changed")
    return s_prefix, f_prefix


def _coordinate_values(tokens: Sequence[int]) -> tuple[int, int, int, int]:
    if len(tokens) != 9 or tokens[0:4] != [ROW_OPEN, 22592, REF_END, BOX_START] or tokens[-1] != BOX_END:
        raise AssertionError("selected row does not retain the frozen bird grammar")
    return tuple(int(token - 151670) for token in tokens[4:8])


def _load_predecessor_panel() -> dict[str, Any]:
    path = PREDECESSOR_ROOT / "stage-a-panel.json"
    value = _read(path)
    if not isinstance(value.get("cases"), list) or len(value["cases"]) != 1:
        raise ValueError("predecessor stage-a panel has no singleton case")
    return value


def _load_saved_bindings() -> tuple[dict[str, Any], dict[str, Any]]:
    case = _load_predecessor_panel()["cases"][0]
    raw = case.get("saved_raw")
    receipt = case.get("saved_receipt")
    _check_binding(raw, "saved native raw")
    _check_binding(receipt, "saved native receipt")
    return dict(raw), dict(receipt)


def _prepare_panel(route: str) -> dict[str, Any]:
    if route not in {"S", "F"}:
        raise ValueError("route must be S or F")
    predecessor = _load_predecessor_panel()
    case0 = predecessor["cases"][0]
    group = copy.deepcopy(case0["group"])
    s_prefix, f_prefix = _selection_prefixes()
    raw_s, raw_f = _source_raw_paths()
    source_raw = {"S": _binding(raw_s), "F": _binding(raw_f)}
    saved_raw, saved_receipt = _load_saved_bindings()
    saved = _rows(_read(Path(saved_raw["path"])).get("rows"))
    target_saved = _row_tokens(saved[TARGET_IMAGE_ID])
    if target_saved[:COMMON_PREFIX_LENGTH] != s_prefix[:COMMON_PREFIX_LENGTH]:
        raise AssertionError("saved native target does not share the frozen 54-token prefix")
    source_rows = {
        "S": _rows(_read(raw_s).get("rows")),
        "F": _rows(_read(raw_f).get("rows")),
    }
    for name, rows, prefix in (("S", source_rows["S"], s_prefix), ("F", source_rows["F"], f_prefix)):
        if TARGET_IMAGE_ID not in rows or _row_tokens(rows[TARGET_IMAGE_ID])[:RELEASE_OFFSET] != prefix:
            raise AssertionError(f"{name} raw route does not retain the frozen 63-token prefix")
    case = copy.deepcopy(case0)
    case.update(
        image_id=TARGET_IMAGE_ID,
        target_position=TARGET_POSITION,
        target_prefix_token_ids=s_prefix if route == "S" else f_prefix,
        prefix_start_offset=0,
        common_native_prefix_length=COMMON_PREFIX_LENGTH,
        capture_offsets=list(CAPTURE_OFFSETS),
        saved_raw=saved_raw,
        saved_receipt=saved_receipt,
        native_identity=False,
        successful_row_route=route,
        source_sampled_raw=source_raw[route],
        release_offset=RELEASE_OFFSET,
        fork_offset=FORK_OFFSET,
    )
    return {
        "schema": "successful_row_mechanism.stage2.panel.v1",
        "route": route,
        "target_image_id": TARGET_IMAGE_ID,
        "target_position": TARGET_POSITION,
        "config": copy.deepcopy(predecessor["config"]),
        "group": group,
        "cases": [case],
        "banks": copy.deepcopy(predecessor["banks"]),
        "coordinate_ids": copy.deepcopy(predecessor["coordinate_ids"]),
        "coefficients": copy.deepcopy(predecessor["coefficients"]),
        "sources": _selection_sources()
        + [
            _binding(PREDECESSOR_ROOT / "stage-a-panel.json"),
            *_source_bindings(),
        ],
        "state_contract": {
            "common_prefix_length": COMMON_PREFIX_LENGTH,
            "release_offset": RELEASE_OFFSET,
            "fork_offset": FORK_OFFSET,
            "capture_offsets": list(CAPTURE_OFFSETS),
            "cache_offsets": list(CACHE_OFFSETS),
            "decoder_layers": list(DEPTHS),
            "decoder_layer_count": 28,
            "max_new_tokens": MAX_NEW_TOKENS,
            "target_only": True,
            "prefixes": {
                "S_sha256": _json_hash(s_prefix),
                "F_sha256": _json_hash(f_prefix),
            },
        },
        "sampled_route_sources": source_raw,
        "native_saved": {"raw": saved_raw, "receipt": saved_receipt},
        "no_training": True,
    }


def prepare_stage2(root: Path, *, overwrite: bool = False) -> None:
    root.mkdir(parents=True, exist_ok=True)
    panel_dir = root / "panels"
    panel_dir.mkdir(exist_ok=True)
    panel_bindings: dict[str, dict[str, Any]] = {}
    for route in ("S", "F"):
        path = panel_dir / f"{route}.json"
        if path.exists() and not overwrite:
            raise FileExistsError(path)
        _write(path, _prepare_panel(route))
        panel_bindings[route] = _binding(path)

    conditions: list[dict[str, Any]] = []
    for route in ("S", "F"):
        conditions.append(
            {
                "id": f"native-{route}",
                "panel": str(panel_dir / f"{route}.json"),
                "route": route,
                "mode": "native",
                "kind": "none",
                "donor_route": None,
                "baseline": f"native-{route}",
                "comparison_group": route,
                "intervention_offset": FORK_OFFSET,
            }
        )
        conditions.append(
            {
                "id": f"self-{route}",
                "panel": str(panel_dir / f"{route}.json"),
                "route": route,
                "mode": "self",
                "kind": "self",
                "donor_route": route,
                "baseline": f"native-{route}",
                "comparison_group": route,
                "intervention_offset": FORK_OFFSET,
            }
        )
    for donor, recipient in (("S", "F"), ("F", "S")):
        conditions.append(
            {
                "id": f"head-{donor}-to-{recipient}",
                "panel": str(panel_dir / f"{recipient}.json"),
                "route": recipient,
                "mode": "head",
                "kind": "head",
                "donor_route": donor,
                "baseline": f"native-{recipient}",
                "comparison_group": recipient,
                "intervention_offset": FORK_OFFSET,
            }
        )
        for depth in DEPTHS:
            conditions.append(
                {
                    "id": f"residual-{donor}-to-{recipient}-layer{depth}",
                    "panel": str(panel_dir / f"{recipient}.json"),
                    "route": recipient,
                    "mode": "residual",
                    "kind": "residual",
                    "depth": depth,
                    "donor_route": donor,
                    "baseline": f"native-{recipient}",
                    "comparison_group": recipient,
                    "intervention_offset": FORK_OFFSET,
                }
            )
        conditions.append(
            {
                "id": f"rebuild-{donor}-to-{recipient}",
                "panel": str(panel_dir / f"{recipient}.json"),
                "route": recipient,
                "mode": "rebuild",
                "kind": "rebuild",
                "donor_route": donor,
                "baseline": f"native-{recipient}",
                "comparison_group": recipient,
                "intervention_offset": FORK_OFFSET,
                "requires_selected_patch": True,
            }
        )
    cells = []
    for condition in conditions:
        out = root / "runtime" / condition["id"]
        cells.append(
            {
                **condition,
                "raw": str(out / "raw.json"),
                "receipt": str(out / "receipt.json"),
                "states": str(out / "states.pt"),
                "logits": str(out / "logits.pt"),
                "cache": str(out / "cache.pt"),
            }
        )
    manifest = {
        "schema": "successful_row_mechanism.stage2.manifest.v1",
        "status": "prepared",
        "artifact_root": str(root),
        "panel_bindings": panel_bindings,
        "constants": {
            "common_prefix_length": COMMON_PREFIX_LENGTH,
            "release_offset": RELEASE_OFFSET,
            "fork_offset": FORK_OFFSET,
            "depths": list(DEPTHS),
            "capture_offsets": list(CAPTURE_OFFSETS),
            "cache_offsets": list(CACHE_OFFSETS),
            "max_new_tokens": MAX_NEW_TOKENS,
            "planned_native_arms": 12,
            "planned_rebuild_controls": 2,
        },
        "cells": cells,
        "commands": {
            "cpu_check": f"python {Path(__file__).resolve()} --cpu-check --output {root / 'cpu-check.json'}",
            "native_S": f"python {Path(__file__).resolve()} --panel {panel_dir / 'S.json'} --condition native-S --output {root / 'runtime/native-S'}",
            "native_F": f"python {Path(__file__).resolve()} --panel {panel_dir / 'F.json'} --condition native-F --output {root / 'runtime/native-F'}",
            "patch_template": f"python {Path(__file__).resolve()} --panel <recipient-panel> --condition <condition-id> --donor-states <baseline-states.pt> --output <condition-output>",
            "rebuild_template": f"python {Path(__file__).resolve()} --panel <recipient-panel> --condition <rebuild-id> --rebuild-from <selected-patch-raw.json> --output <rebuild-output>",
        },
        "stop_rule": "Run only the frozen native/self/head/three-depth arms; run rebuild controls only for a selected effective intermediate patch.",
    }
    manifest_path = root / "runtime-manifest.json"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(manifest_path)
    _write(manifest_path, manifest)


def _condition_spec(name: str) -> dict[str, Any]:
    if name in {"native-S", "native-F"}:
        return {"mode": "native", "route": name[-1], "donor": None, "depth": None}
    if name in {"self-S", "self-F"}:
        return {"mode": "self", "route": name[-1], "donor": name[-1], "depth": None}
    for kind in ("head", "rebuild"):
        prefix = kind + "-"
        if name.startswith(prefix) and "-to-" in name:
            donor, route = name[len(prefix) :].split("-to-", 1)
            if donor in {"S", "F"} and route in {"S", "F"} and donor != route:
                return {"mode": kind, "route": route, "donor": donor, "depth": None}
    prefix = "residual-"
    if name.startswith(prefix) and "-to-" in name and "-layer" in name:
        direction, depth_text = name[len(prefix) :].rsplit("-layer", 1)
        donor, route = direction.split("-to-")
        depth = int(depth_text)
        if donor in {"S", "F"} and route in {"S", "F"} and donor != route and depth in DEPTHS:
            return {"mode": "residual", "route": route, "donor": donor, "depth": depth}
    raise ValueError(f"unsupported frozen Stage2 condition: {name}")


def _find_language_model(model: Any) -> tuple[Any, str]:
    # R16 loads the native Qwen3-VL class directly; DoRA is attached through
    # Transformers' PeftAdapterMixin and leaves this exact module path intact.
    language = getattr(getattr(model, "model", None), "language_model", None)
    if language is None or not hasattr(language, "layers"):
        raise RuntimeError("R16 Qwen3-VL model.model.language_model.layers is unavailable")
    if language.__class__.__name__ != "Qwen3VLTextModel":
        raise AssertionError(
            f"unexpected Qwen3-VL language model class: {language.__class__.__name__}"
        )
    return language, "model.model.language_model"


def _module_path(root: Any, target: Any) -> str:
    for name, module in root.named_modules():
        if module is target:
            return name or "<root>"
    return "<unresolved>"


def _tensor_meta(value: torch.Tensor, *, site: str, offset: int, batch_index: int | None = None) -> dict[str, Any]:
    return {
        "site": site,
        "action_offset": offset,
        "batch_index": batch_index,
        "sequence_index": -1,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "numel": int(value.numel()),
        "sha256": _tensor_hash(value),
    }


def _past_key_values(output: Any) -> Any:
    value = getattr(output, "past_key_values", None)
    if value is None:
        raise AssertionError("R16 Qwen3-VL output has no past_key_values attribute")
    return value


def _cache_entries(cache: Any) -> list[tuple[int, torch.Tensor, torch.Tensor]]:
    if cache is None or cache.__class__.__name__ != "DynamicCache":
        return []
    layers = getattr(cache, "layers", None)
    if not isinstance(layers, (tuple, list)) or len(layers) != 28:
        return []
    result = []
    for index, layer in enumerate(layers):
        key = getattr(layer, "keys", None)
        value = getattr(layer, "values", None)
        if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
            return []
        result.append((index, key, value))
    return result


def _cache_snapshot(cache: Any, *, offset: int, target_position: int) -> tuple[dict[str, Any], dict[str, Any]]:
    entries = _cache_entries(cache)
    if not entries:
        raise AssertionError("native generation returned no inspectable past_key_values")
    metadata: dict[str, Any] = {"action_offset": offset, "layer_count": len(entries), "layers": {}}
    tensors: dict[str, Any] = {}
    for index, key, value in entries:
        if key.ndim < 3 or value.ndim < 3 or key.shape[0] <= target_position or value.shape[0] <= target_position:
            raise AssertionError(f"cache layer {index} has unexpected shape")
        key_last = key[:, :, -1, :].detach().cpu().clone()
        value_last = value[:, :, -1, :].detach().cpu().clone()
        tensors[str(index)] = {"key_last": key_last, "value_last": value_last}
        metadata["layers"][str(index)] = {
            "key_shape": list(key.shape),
            "value_shape": list(value.shape),
            "key_dtype": str(key.dtype),
            "value_dtype": str(value.dtype),
            "key_last": _tensor_meta(key_last, site=f"past_key_values.layers[{index}].keys[:, :, -1, :]", offset=offset),
            "value_last": _tensor_meta(value_last, site=f"past_key_values.layers[{index}].values[:, :, -1, :]", offset=offset),
            "target_key_sha256": _tensor_hash(key_last[target_position]),
            "target_value_sha256": _tensor_hash(value_last[target_position]),
        }
    return metadata, tensors


def _force_target(scores: torch.Tensor, row: int, token: int) -> torch.Tensor:
    if scores.ndim != 2 or not 0 <= row < scores.shape[0] or not 0 <= token < scores.shape[1]:
        raise ValueError("target force shape/token is invalid")
    transformed = scores.clone()
    transformed[row].fill_(-torch.inf)
    transformed[row, token] = 0
    companions = torch.ones(scores.shape[0], dtype=torch.bool, device=scores.device)
    companions[row] = False
    if not torch.equal(scores[companions], transformed[companions]):
        raise AssertionError("target force changed a native companion")
    return transformed


def _replace_last_target(value: torch.Tensor, row: int, replacement: torch.Tensor) -> torch.Tensor:
    if value.ndim != 3 or replacement.ndim != 1 or value.shape[-1] != replacement.shape[0]:
        raise ValueError("last-token residual replacement shape is invalid")
    transformed = value.clone()
    transformed[row, -1, :] = replacement.to(device=value.device, dtype=value.dtype)
    if not torch.equal(value[:row], transformed[:row]) or not torch.equal(value[row + 1 :], transformed[row + 1 :]):
        raise AssertionError("residual replacement changed a native companion")
    if value.shape[1] > 1 and not torch.equal(value[:, :-1, :], transformed[:, :-1, :]):
        raise AssertionError("residual replacement changed a non-current token")
    return transformed


def _replace_head_target(value: torch.Tensor, row: int, replacement: torch.Tensor) -> torch.Tensor:
    return _replace_last_target(value, row, replacement)


def _top2(scores: torch.Tensor) -> tuple[int, float]:
    values, indices = torch.topk(scores, 2)
    return int(indices[0]), float((values[0] - values[1]).item())


def _capture_metadata(captures: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for offset, capture in captures.items():
        entry: dict[str, Any] = {}
        layers = capture.get("layers", {})
        if isinstance(layers, Mapping):
            entry["layers"] = {}
            for depth, value in layers.items():
                if not isinstance(value, Mapping):
                    continue
                layer_entry: dict[str, Any] = {}
                action_offset = int(offset)
                for name in ("before", "after"):
                    tensor = value.get(name)
                    if isinstance(tensor, torch.Tensor):
                        layer_entry[name] = _tensor_meta(
                            tensor[TARGET_POSITION],
                            site=f"decoder.layers[{depth}].output.{name}",
                            offset=action_offset,
                            batch_index=TARGET_POSITION,
                        )
                entry["layers"][str(depth)] = layer_entry
        for name in ("head_before", "head_after"):
            tensor = capture.get(name)
            if isinstance(tensor, torch.Tensor):
                entry[name] = _tensor_meta(
                    tensor[TARGET_POSITION],
                    site=f"lm_head.input.{name.removeprefix('head_')}",
                    offset=int(offset),
                    batch_index=TARGET_POSITION,
                )
        result[str(offset)] = entry
    return result


def _state_tensor(states: Mapping[str, Any], offset: int, key: str, depth: int | None, target_position: int) -> torch.Tensor:
    capture = states.get("captures", {}).get(str(offset))
    if not isinstance(capture, Mapping):
        raise ValueError(f"donor states have no capture at offset {offset}")
    if key == "head":
        value = capture.get("head_before")
    else:
        layers = capture.get("layers")
        value = layers.get(str(depth), {}).get("before") if isinstance(layers, Mapping) else None
    if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[0] <= target_position:
        raise ValueError(f"donor state {key}/{depth} at offset {offset} is malformed")
    return value[target_position]


def _load_donor(path: Path, *, route: str, target_position: int, input_identity: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(loaded, Mapping) or loaded.get("schema") != "successful_row_mechanism.state.v1":
        raise ValueError("donor state file has an unsupported schema")
    if loaded.get("route") != route:
        raise AssertionError("donor state route differs from condition")
    if loaded.get("target_position") != target_position:
        raise AssertionError("donor state target position differs from condition")
    if loaded.get("input_identity") != dict(input_identity):
        raise AssertionError("donor state native input identity differs")
    return dict(loaded), _binding(path)


def _load_native_recipient_reference(
    panel_path: Path, *, route: str, target_position: int, input_identity: Mapping[str, Any]
) -> dict[str, Any]:
    """Load the already-produced native recipient fork logits for patch deltas."""
    stage2_root = panel_path.parent.parent
    output = stage2_root / "runtime" / f"native-{route}"
    receipt_path = output / "receipt.json"
    logits_path = output / "logits.pt"
    receipt = _read(receipt_path)
    if receipt.get("status") != "candidate_complete" or receipt.get("condition") != f"native-{route}":
        raise AssertionError(f"native recipient baseline is not complete: {receipt_path}")
    expected_identity = receipt.get("input_identity")
    if not isinstance(expected_identity, Mapping) or dict(expected_identity) != dict(input_identity):
        raise AssertionError("native recipient baseline input identity differs")
    payload = torch.load(logits_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("schema") != "successful_row_mechanism.logits.v1":
        raise ValueError("native recipient baseline logits have an unsupported schema")
    raw_logits = payload.get("raw_logits")
    if not isinstance(raw_logits, Mapping):
        raise ValueError("native recipient baseline has no raw logits")
    full = raw_logits.get(FORK_OFFSET)
    if not isinstance(full, torch.Tensor) or full.ndim != 2 or full.shape[0] <= target_position:
        raise ValueError("native recipient baseline fork logits are not full batch logits")
    return {
        "condition": f"native-{route}",
        "receipt": _binding(receipt_path),
        "logits": _binding(logits_path),
        "raw_fork": full.detach().cpu().clone(),
    }


def _logit_delta(
    baseline: torch.Tensor, patched: torch.Tensor, *, offset: int, target_position: int
) -> dict[str, Any]:
    if baseline.ndim != 2 or patched.ndim != 2 or baseline.shape != patched.shape:
        raise ValueError("baseline and patched fork logits must have equal full-batch shapes")
    base = baseline[target_position].float()
    current = patched[target_position].float()
    delta = current - base
    base_winner, base_margin = _top2(base)
    current_winner, current_margin = _top2(current)
    return {
        "action_offset": offset,
        "target_position": target_position,
        "same_prefix_through_fork": True,
        "baseline_argmax": base_winner,
        "baseline_top2_margin": base_margin,
        "patched_raw_argmax": current_winner,
        "patched_raw_top2_margin": current_margin,
        "exact_equal": bool(torch.equal(base, current)),
        "max_abs": float(delta.abs().max().item()),
        "mean_abs": float(delta.abs().mean().item()),
        "l2": float(torch.linalg.vector_norm(delta).item()),
        "baseline_sha256": _tensor_hash(base),
        "patched_raw_sha256": _tensor_hash(current),
    }


def _validate_panel(panel: Mapping[str, Any], route: str) -> tuple[Mapping[str, Any], Mapping[str, Any], list[int]]:
    if panel.get("schema") != "successful_row_mechanism.stage2.panel.v1" or panel.get("route") != route:
        raise ValueError("panel schema/route does not match frozen Stage2 condition")
    native_runtime.fresh._check_sources(panel)
    cases = panel.get("cases")
    if not isinstance(cases, list) or len(cases) != 1:
        raise ValueError("Stage2 panel must contain one target case")
    case = cases[0]
    if not isinstance(case, Mapping) or case.get("target_position") != TARGET_POSITION:
        raise ValueError("Stage2 panel target position is not frozen")
    group = case.get("group")
    if not isinstance(group, Mapping) or not isinstance(group.get("cases"), list) or len(group["cases"]) != 4:
        raise ValueError("Stage2 panel does not contain native batch-4 cases")
    prefix = _tokens(case.get("target_prefix_token_ids"), "target_prefix_token_ids")
    if len(prefix) != RELEASE_OFFSET or case.get("common_native_prefix_length") != COMMON_PREFIX_LENGTH:
        raise ValueError("Stage2 target prefix length is not frozen")
    expected_s, expected_f = _selection_prefixes()
    expected = expected_s if route == "S" else expected_f
    if prefix != expected:
        raise AssertionError("panel target prefix differs from frozen S/F route")
    if case.get("capture_offsets") != list(CAPTURE_OFFSETS):
        raise AssertionError("panel capture offsets differ from frozen contract")
    source_raw = case.get("source_sampled_raw")
    _check_binding(source_raw, "sampled route raw")
    return case, group, prefix


def _run(
    *,
    panel_path: Path,
    condition: str,
    output: Path,
    donor_states_path: Path | None = None,
    rebuild_from: Path | None = None,
) -> dict[str, Any]:
    spec = _condition_spec(condition)
    panel = _read(panel_path)
    case, group, frozen_prefix = _validate_panel(panel, spec["route"])
    config = panel.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("Stage2 panel has no config")
    infer = native_runtime.fresh.InferConfig.model_validate(config)
    if infer.backend.type != "hf" or infer.model.dtype != "fp32" or infer.backend.hf.attn_implementation != "sdpa":
        raise AssertionError("Stage2 requires frozen HF FP32/SDPA execution")
    cases = group["cases"]
    image_ids = [native_runtime._image_id(item) for item in cases]
    if image_ids[TARGET_POSITION] != TARGET_IMAGE_ID or len(set(image_ids)) != 4:
        raise AssertionError("native batch target/order changed")
    saved_binding = panel.get("native_saved", {}).get("raw") if isinstance(panel.get("native_saved"), Mapping) else case.get("saved_raw")
    receipt_binding = panel.get("native_saved", {}).get("receipt") if isinstance(panel.get("native_saved"), Mapping) else case.get("saved_receipt")
    saved_path = _check_binding(saved_binding, "saved native raw")
    saved_receipt_path = _check_binding(receipt_binding, "saved native receipt")
    saved_receipt = _read(saved_receipt_path)
    if saved_receipt.get("status") != "candidate_complete":
        raise AssertionError("saved native receipt is not candidate_complete")
    saved = _rows(_read(saved_path).get("rows"))
    if set(saved) != set(image_ids):
        raise AssertionError("saved native rows do not match Stage2 batch")
    if spec["mode"] in {"self", "head", "residual"} and donor_states_path is None:
        raise ValueError(f"{condition} requires --donor-states")
    if spec["mode"] == "rebuild" and rebuild_from is None:
        raise ValueError(f"{condition} requires --rebuild-from")
    target_prefix = list(frozen_prefix)
    if spec["mode"] == "rebuild":
        rebuilt = _read(rebuild_from)
        rebuilt_rows = _rows(rebuilt.get("rows"))
        target_prefix = _row_tokens(rebuilt_rows[TARGET_IMAGE_ID])[: FORK_OFFSET + 1]
        if target_prefix[:RELEASE_OFFSET] != frozen_prefix or EOS in target_prefix:
            raise AssertionError("rebuild source does not contain the recipient 68-token prefix")
    output.mkdir(parents=True, exist_ok=False)
    receipt: dict[str, Any] = {
        "schema": "successful_row_mechanism.state_runtime.v1",
        "status": "running",
        "condition": condition,
        "mode": spec["mode"],
        "route": spec["route"],
        "donor_route": spec["donor"],
        "depth": spec["depth"],
        "image_id": TARGET_IMAGE_ID,
        "image_ids": image_ids,
        "target_position": TARGET_POSITION,
        "prefix_start_offset": 0,
        "prefix_length": len(frozen_prefix),
        "common_prefix_length": COMMON_PREFIX_LENGTH,
        "release_offset": RELEASE_OFFSET,
        "fork_offset": FORK_OFFSET,
        "capture_offsets": list(CAPTURE_OFFSETS),
        "cache_offsets": list(CACHE_OFFSETS),
        "decoder_layers": list(DEPTHS),
        "max_new_tokens": MAX_NEW_TOKENS,
        "panel": _binding(panel_path),
        "producer": _binding(Path(__file__).resolve()),
        "pid": os.getpid(),
        "installed_source_bindings": _source_bindings(),
        "saved_native_raw": _binding(saved_path),
        "saved_native_receipt": _binding(saved_receipt_path),
        "raw_path": str(output / "raw.json"),
        "states_path": str(output / "states.pt"),
        "logits_path": str(output / "logits.pt"),
        "cache_path": str(output / "cache.pt"),
        "intervention": {
            "site": "none" if spec["mode"] in {"native", "rebuild"} else spec["mode"],
            "action_offset": None if spec["mode"] in {"native", "rebuild"} else FORK_OFFSET,
            "target_only": True,
            "parameter_mutation": False,
        },
    }
    if rebuild_from is not None:
        receipt["rebuild_from"] = _binding(rebuild_from)
    _write(output / "receipt.json", receipt)

    qwen: Any = None
    handles: list[Any] = []
    began = time.monotonic()
    model_forwards = 0
    prefill_mrope: str | None = None
    mrope_positions: dict[str, dict[str, Any]] = {}
    forward_trace: list[dict[str, Any]] = []
    histories: list[list[int]] = [[] for _ in image_ids]
    done = [False] * len(image_ids)
    captures: dict[str, dict[str, Any]] = {}
    logits: dict[int, torch.Tensor] = {}
    logits_after: dict[int, torch.Tensor] = {}
    logit_records: list[dict[str, Any]] = []
    cache_metadata: dict[str, Any] = {}
    cache_tensors: dict[str, Any] = {}
    patch_records: list[dict[str, Any]] = []
    donor: dict[str, Any] | None = None
    donor_binding: dict[str, Any] | None = None
    recipient_baseline: dict[str, Any] | None = None

    try:
        qwen, loaded_identity = native_runtime.fresh.load_policy(infer, device=torch.device("cuda:0"))
        model = qwen.model.eval()
        language_model, language_path = _find_language_model(model)
        layers = language_model.layers
        if len(layers) != 28:
            raise AssertionError(f"installed Qwen3-VL layer count is {len(layers)}, expected 28")
        lm_head = model.get_output_embeddings()
        if lm_head is None:
            raise AssertionError("loaded model has no output head")
        request_config = copy.deepcopy(config)
        request_config['data']['input_jsonl'] = group['input_jsonl']
        requests, _ = native_runtime.fresh.build_bound_native_requests(qwen, request_config, cases)
        batch = native_runtime.fresh.prepare_native_inputs(
            qwen.processor, requests, device="cuda:0", record_media_identity=True
        )
        input_identity = native_runtime.fresh._input_identity(batch)
        expected_identity = saved_receipt.get("input_identity")
        if not isinstance(expected_identity, Mapping) or input_identity != dict(expected_identity):
            raise AssertionError("Stage2 native input identity differs from saved receipt")
        width = int(batch.inputs["input_ids"].shape[1])
        if width <= 0:
            raise AssertionError("native prompt width is empty")
        if spec["mode"] != "native":
            recipient_baseline = _load_native_recipient_reference(
                panel_path,
                route=spec["route"],
                target_position=TARGET_POSITION,
                input_identity=input_identity,
            )
        if spec["mode"] in {"self", "head", "residual"}:
            donor, donor_binding = _load_donor(
                donor_states_path, route=spec["donor"], target_position=TARGET_POSITION, input_identity=input_identity
            )
        initial_versions = {name: int(parameter._version) for name, parameter in model.named_parameters()}

        def model_pre_hook(module: Any, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> None:
            nonlocal model_forwards, prefill_mrope
            model_forwards += 1
            action_offset = model_forwards - 1
            input_ids = kwargs.get("input_ids")
            if isinstance(input_ids, torch.Tensor):
                input_width = int(input_ids.shape[1])
            else:
                embeds = kwargs.get("inputs_embeds")
                input_width = int(embeds.shape[1]) if isinstance(embeds, torch.Tensor) else None
            cache_position = kwargs.get("cache_position")
            forward_trace.append(
                {
                    "forward_count": model_forwards,
                    "action_offset": action_offset,
                    "input_width": input_width,
                    "past_key_values_present": kwargs.get("past_key_values") is not None,
                    "cache_position": None if not isinstance(cache_position, torch.Tensor) else cache_position.detach().cpu().tolist(),
                    "pixel_values_present": isinstance(kwargs.get("pixel_values"), torch.Tensor),
                }
            )
            if model_forwards > MAX_NEW_TOKENS:
                raise AssertionError("native generation exceeded the 3084 forward cap")

        def mrope_hook(module: Any, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> None:
            nonlocal prefill_mrope
            positions = kwargs.get("position_ids")
            if not isinstance(positions, torch.Tensor):
                raise AssertionError("R16 Qwen3-VL did not expose MRoPE position IDs")
            offset = model_forwards - 1
            if model_forwards == 1:
                prefill_mrope = _tensor_hash(positions)
            if offset in CAPTURE_OFFSETS:
                if positions.ndim != 3 or positions.shape[0] != 3 or positions.shape[1] != len(image_ids):
                    raise AssertionError(
                        f"R16 MRoPE position_ids shape changed at offset {offset}: {tuple(positions.shape)}"
                    )
                captured = positions.detach().cpu().clone()
                mrope_positions[str(offset)] = {
                    "action_offset": offset,
                    "forward_count": model_forwards,
                    "shape": list(captured.shape),
                    "dtype": str(captured.dtype),
                    "device": str(positions.device),
                    "sha256": _tensor_hash(captured),
                    "position_ids": captured.tolist(),
                    "target_position_ids": captured[:, TARGET_POSITION, :].tolist(),
                }

        def model_output_hook(module: Any, args: tuple[Any, ...], output_value: Any) -> None:
            offset = model_forwards - 1
            if offset not in CACHE_OFFSETS:
                return
            metadata, tensors = _cache_snapshot(
                _past_key_values(output_value), offset=offset, target_position=TARGET_POSITION
            )
            cache_metadata[str(offset)] = metadata
            cache_tensors[str(offset)] = tensors

        def layer_hook(depth: int):
            def hook(module: Any, args: tuple[Any, ...], output_value: Any) -> Any:
                offset = model_forwards - 1
                if not isinstance(output_value, torch.Tensor) or output_value.ndim != 3 or output_value.shape[0] != len(image_ids):
                    raise AssertionError(f"decoder layer {depth} output shape is not native batch-3D")
                offset = model_forwards - 1
                if offset not in CAPTURE_OFFSETS:
                    if not (
                        offset == FORK_OFFSET
                        and spec["mode"] == "self"
                    ) and not (
                        offset == FORK_OFFSET
                        and spec["mode"] == "residual"
                        and depth == spec["depth"]
                    ):
                        return output_value
                before = output_value[:, -1, :].detach().clone()
                capture = captures.setdefault(str(offset), {"action_offset": offset, "layers": {}})
                layer_capture = capture["layers"].setdefault(str(depth), {"before": before})
                layer_capture["before"] = before
                if spec["mode"] == "self" or (spec["mode"] == "residual" and depth == spec["depth"]):
                    if offset != FORK_OFFSET:
                        return output_value
                    replacement = _state_tensor(donor or {}, FORK_OFFSET, "layer", depth, TARGET_POSITION)
                    transformed = _replace_last_target(output_value, TARGET_POSITION, replacement)
                    after = transformed[:, -1, :].detach().clone()
                    layer_capture["after"] = after
                    patch_records.append(
                        {
                            "kind": "residual",
                            "depth": depth,
                            "action_offset": offset,
                            "forward_count": model_forwards,
                            "module_path": f"{language_path}.layers[{depth}]",
                            "target_position": TARGET_POSITION,
                            "before": _tensor_meta(before[TARGET_POSITION], site=f"{language_path}.layers[{depth}].output.before", offset=offset, batch_index=TARGET_POSITION),
                            "after": _tensor_meta(after[TARGET_POSITION], site=f"{language_path}.layers[{depth}].output.after", offset=offset, batch_index=TARGET_POSITION),
                            "companion_equal": bool(torch.equal(before[[i for i in range(len(image_ids)) if i != TARGET_POSITION]], after[[i for i in range(len(image_ids)) if i != TARGET_POSITION]])),
                        }
                    )
                    return transformed
                return output_value

            return hook

        def head_hook(module: Any, args: tuple[Any, ...]) -> tuple[Any, ...] | None:
            if not args or not isinstance(args[0], torch.Tensor):
                raise AssertionError("lm_head pre-hook did not receive hidden states")
            hidden = args[0]
            if hidden.ndim != 3 or hidden.shape[0] != len(image_ids):
                raise AssertionError("lm_head input shape differs from native batch")
            offset = model_forwards - 1
            if offset not in CAPTURE_OFFSETS and not (
                offset == FORK_OFFSET and spec["mode"] in {"self", "head"}
            ):
                return None
            capture = captures.setdefault(str(offset), {"action_offset": offset, "layers": {}})
            capture["head_before"] = hidden[:, -1, :].detach().clone()
            if spec["mode"] in {"self", "head"} and offset == FORK_OFFSET:
                replacement = _state_tensor(donor or {}, FORK_OFFSET, "head", None, TARGET_POSITION)
                transformed = _replace_head_target(hidden, TARGET_POSITION, replacement)
                capture["head_after"] = transformed[:, -1, :].detach().clone()
                patch_records.append(
                    {
                        "kind": "head",
                        "action_offset": offset,
                        "forward_count": model_forwards,
                        "module_path": _module_path(model, lm_head),
                        "target_position": TARGET_POSITION,
                        "before": _tensor_meta(hidden[TARGET_POSITION, -1, :], site="lm_head.input.before", offset=offset, batch_index=TARGET_POSITION),
                        "after": _tensor_meta(transformed[TARGET_POSITION, -1, :], site="lm_head.input.after", offset=offset, batch_index=TARGET_POSITION),
                        "companion_equal": bool(torch.equal(hidden[[i for i in range(len(image_ids)) if i != TARGET_POSITION], -1, :], transformed[[i for i in range(len(image_ids)) if i != TARGET_POSITION], -1, :])),
                    }
                )
                return (transformed, *args[1:])
            return None

        handles.extend([
            model.register_forward_pre_hook(model_pre_hook, with_kwargs=True),
            model.register_forward_hook(model_output_hook),
            language_model.register_forward_pre_hook(mrope_hook, with_kwargs=True),
            lm_head.register_forward_pre_hook(head_hook),
        ])
        for depth in DEPTHS:
            handles.append(layers[depth].register_forward_hook(layer_hook(depth)))

        class Processor(LogitsProcessor):
            def __call__(self, ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
                offset = int(ids.shape[1] - width)
                if offset != model_forwards - 1:
                    raise AssertionError(f"processor offset {offset} != native forward offset {model_forwards - 1}")
                if offset < 0 or offset >= MAX_NEW_TOKENS:
                    raise AssertionError("processor action offset is outside frozen budget")
                raw = scores.detach().clone()
                transformed = scores
                target_done = done[TARGET_POSITION]
                if offset < COMMON_PREFIX_LENGTH:
                    if int(raw[TARGET_POSITION].argmax()) != frozen_prefix[offset]:
                        raise AssertionError('natural prehistory argmax changed before supplied row')
                if offset < len(target_prefix):
                    transformed = _force_target(transformed, TARGET_POSITION, target_prefix[offset])
                for index in range(len(image_ids)):
                    if done[index]:
                        continue
                    history = ids[index, width:].tolist()
                    if history != histories[index]:
                        raise AssertionError(f"native history forked for image {image_ids[index]}")
                if offset in CAPTURE_OFFSETS and not target_done:
                    logits[offset] = raw.detach().cpu().clone() if offset == FORK_OFFSET else raw[TARGET_POSITION].detach().cpu().clone()
                    logits_after[offset] = transformed.detach().cpu().clone() if offset == FORK_OFFSET else transformed[TARGET_POSITION].detach().cpu().clone()
                    target_raw = raw[TARGET_POSITION]
                    target_after = transformed[TARGET_POSITION]
                    raw_winner, raw_margin = _top2(target_raw)
                    after_winner, after_margin = _top2(target_after)
                    stored_raw = logits[offset]
                    stored_after = logits_after[offset]
                    logit_records.append(
                        {
                            "action_offset": offset,
                            "forward_count": model_forwards,
                            "history_sha256": _json_hash(histories[TARGET_POSITION]),
                            "target_position": TARGET_POSITION,
                            "raw_argmax": raw_winner,
                            "raw_top2_margin": raw_margin,
                            "transformed_argmax": after_winner,
                            "transformed_top2_margin": after_margin,
                            "raw_eos_logit": float(target_raw[EOS].item()),
                            "transformed_eos_logit": float(target_after[EOS].item()),
                            "full_vocab": offset == FORK_OFFSET,
                            "raw_tensor": _tensor_meta(
                                stored_raw,
                                site="logits_processor.raw_scores",
                                offset=offset,
                            ),
                            "transformed_tensor": _tensor_meta(
                                stored_after,
                                site="logits_processor.transformed_scores",
                                offset=offset,
                            ),
                        }
                    )
                for index in range(len(image_ids)):
                    if done[index]:
                        continue
                    chosen = int(torch.argmax(transformed[index]).item())
                    if index == TARGET_POSITION and offset < len(target_prefix) and chosen != target_prefix[offset]:
                        raise AssertionError("target prefix force was not replayed")
                    histories[index].append(chosen)
                    if chosen == EOS:
                        done[index] = True
                return transformed

        original_generate = model.generate

        def generate_wrapper(**kwargs: Any) -> Any:
            if (
                kwargs.get("max_new_tokens") != MAX_NEW_TOKENS
                or kwargs.get("repetition_penalty") != 1
                or kwargs.get("do_sample")
                or "logits_processor" in kwargs
            ):
                raise AssertionError("Stage2 generation settings changed")
            return original_generate(**kwargs, logits_processor=LogitsProcessorList([Processor()]))

        model.generate = generate_wrapper
        try:
            with torch.no_grad():
                values = native_runtime.fresh.generate_continuations(
                    model,
                    batch,
                    extensions=[[] for _ in cases],
                    budgets=[MAX_NEW_TOKENS for _ in cases],
                    eos_token_id=EOS,
                    pad_token_id=qwen.tokenizer.pad_token_id,
                    policy=native_runtime.fresh.NativeGenerationPolicy(
                        temperature=0,
                        top_p=1,
                        top_k=0,
                        repetition_penalty=1,
                        use_model_defaults=False,
                    ),
                    trace="none",
                    seed=None,
                )
        finally:
            model.generate = original_generate
        if len(values) != len(cases) or prefill_mrope is None:
            raise AssertionError("native generation identity is incomplete")
        if model_forwards != len(forward_trace) or not forward_trace:
            raise AssertionError("model forward trace is incomplete")
        if spec["mode"] in {"self", "head", "residual"}:
            if not patch_records or any(item["action_offset"] != FORK_OFFSET for item in patch_records):
                raise AssertionError("state patch did not occur exactly at the frozen fork")
        if spec["mode"] in {"native", "rebuild"} and patch_records:
            raise AssertionError("unpatched arm unexpectedly recorded a patch")
        current_versions = {name: int(parameter._version) for name, parameter in model.named_parameters()}
        if current_versions != initial_versions:
            raise AssertionError("generation mutated model parameters")
        if native_runtime.fresh._input_identity(batch) != input_identity:
            raise AssertionError("generation mutated native inputs")
        rows = []
        for image, request, value in zip(image_ids, requests, values, strict=True):
            token_ids = list(value.token_ids)
            rows.append(
                {
                    "image_id": image,
                    "request_id": request.request_id,
                    "token_ids": token_ids,
                    "tokens_sha256": _json_hash(token_ids),
                    "text": qwen.tokenizer.decode(token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False),
                    "stop": value.stop_reason,
                }
            )
        for row in rows:
            image = row["image_id"]
            if image != TARGET_IMAGE_ID:
                if row["token_ids"] != _row_tokens(saved[image]) or row["stop"] != _row_stop(saved[image]):
                    raise AssertionError(f"native companion changed for image {image}")
        target_tokens = rows[TARGET_POSITION]["token_ids"]
        if target_tokens[: len(target_prefix)] != target_prefix:
            raise AssertionError("target literal prefix was not retained")
        if spec['mode'] in {'native', 'self'}:
            reference_rows = _rows(_read(Path(case['source_sampled_raw']['path']))['rows'])
            reference = reference_rows[TARGET_IMAGE_ID]
            if target_tokens != _row_tokens(reference) or rows[TARGET_POSITION]['stop'] != _row_stop(reference):
                raise AssertionError('native/self target differs from frozen sampled route')
        if spec["mode"] == "rebuild" and len(target_prefix) != FORK_OFFSET + 1:
            raise AssertionError("rebuild prefix length changed")
        if recipient_baseline is not None:
            baseline_full = recipient_baseline["raw_fork"]
            patched_full = logits.get(FORK_OFFSET)
            if not isinstance(patched_full, torch.Tensor) or patched_full.ndim != 2:
                raise AssertionError("patched arm did not capture full fork logits")
            fork_delta = _logit_delta(
                baseline_full,
                patched_full,
                offset=FORK_OFFSET,
                target_position=TARGET_POSITION,
            )
            for record in logit_records:
                if record["action_offset"] == FORK_OFFSET:
                    record["recipient_native_baseline_delta"] = fork_delta
        else:
            baseline_full = None
            fork_delta = None
        state_payload = {
            "schema": "successful_row_mechanism.state.v1",
            "condition": condition,
            "route": spec["route"],
            "target_position": TARGET_POSITION,
            "image_ids": image_ids,
            "input_identity": input_identity,
            "prefill_mrope_sha256": prefill_mrope,
            "mrope_positions": mrope_positions,
            "captures": captures,
            "capture_metadata": _capture_metadata(captures),
            "patch_records": patch_records,
            "recipient_native_baseline": None
            if recipient_baseline is None
            else {
                "condition": recipient_baseline["condition"],
                "receipt": recipient_baseline["receipt"],
                "logits": recipient_baseline["logits"],
                "raw_fork": _tensor_meta(
                    recipient_baseline["raw_fork"],
                    site="native_recipient.logits.raw_fork",
                    offset=FORK_OFFSET,
                ),
            },
            "recipient_baseline_fork_delta": fork_delta,
            "module_paths": {
                "language_model": language_path,
                "layers": {str(depth): f"{language_path}.layers[{depth}]" for depth in DEPTHS},
                "lm_head": _module_path(model, lm_head),
            },
            "target_prefix": target_prefix,
            "capture_offsets": list(CAPTURE_OFFSETS),
        }
        torch.save(state_payload, output / "states.pt")
        torch.save(
            {
                "schema": "successful_row_mechanism.logits.v1",
                "condition": condition,
                "offsets": sorted(logits),
                "raw_logits": logits,
                "transformed_logits": logits_after,
                "records": logit_records,
                "recipient_native_raw_fork": baseline_full,
                "recipient_native_raw_fork_meta": None
                if baseline_full is None
                else _tensor_meta(
                    baseline_full,
                    site="native_recipient.logits.raw_fork",
                    offset=FORK_OFFSET,
                ),
                "recipient_native_baseline": None
                if recipient_baseline is None
                else {
                    "condition": recipient_baseline["condition"],
                    "receipt": recipient_baseline["receipt"],
                    "logits": recipient_baseline["logits"],
                },
                "recipient_baseline_fork_delta": fork_delta,
            },
            output / "logits.pt",
        )
        torch.save(
            {
                "schema": "successful_row_mechanism.cache.v1",
                "condition": condition,
                "offsets": sorted(cache_tensors),
                "metadata": cache_metadata,
                "last_position": cache_tensors,
            },
            output / "cache.pt",
        )
        raw = {
            "schema": "successful_row_mechanism.raw.v1",
            "condition": condition,
            "route": spec["route"],
            "image_id": TARGET_IMAGE_ID,
            "image_ids": image_ids,
            "target_position": TARGET_POSITION,
            "rows": rows,
            "input_identity": input_identity,
            "prefill_mrope_sha256": prefill_mrope,
            "mrope_positions": mrope_positions,
            "prefix": {"start_offset": 0, "token_ids": target_prefix},
            "fork": {
                "action_offset": FORK_OFFSET,
                "target_token_id": None if len(target_tokens) <= FORK_OFFSET else target_tokens[FORK_OFFSET],
                "logits": _binding(output / "logits.pt"),
            },
            "state_captures": _binding(output / "states.pt"),
            "cache_captures": _binding(output / "cache.pt"),
            "intervention": {
                "mode": spec["mode"],
                "donor_route": spec["donor"],
                "depth": spec["depth"],
                "action_offset": None if spec["mode"] in {"native", "rebuild"} else FORK_OFFSET,
            "patch_records": patch_records,
            "recipient_native_baseline": None
            if recipient_baseline is None
            else {
                "condition": recipient_baseline["condition"],
                "receipt": recipient_baseline["receipt"],
                "logits": recipient_baseline["logits"],
            },
            "recipient_baseline_fork_delta": fork_delta,
                "donor_states": donor_binding,
            },
            "sparse_decisions": {
                "requested_offsets": list(CAPTURE_OFFSETS),
                "observed_offsets": sorted(logits),
                "records": logit_records,
            },
            "native_forward_trace": [item for item in forward_trace if item["action_offset"] in CACHE_OFFSETS or item["action_offset"] == FORK_OFFSET],
            "cache_boundary": {
                "captured_offsets": sorted(cache_metadata),
                "layer_count": next(iter(cache_metadata.values()))["layer_count"] if cache_metadata else 0,
                "target_last_position_only": True,
            },
            "policy": "native greedy with target literal prefix force; one current-token state replacement only for patch arms",
        }
        _write(output / "raw.json", raw)
        receipt.update(
            status="candidate_complete",
            loaded_model_identity=loaded_identity,
            input_identity=input_identity,
            prefill_mrope_sha256=prefill_mrope,
            mrope_positions=mrope_positions,
            model_forwards=model_forwards,
            observed_logit_offsets=sorted(logits),
            observed_cache_offsets=sorted(cache_metadata),
            patch_count=len(patch_records),
            patch_records=patch_records,
            donor_states=donor_binding,
            states=_binding(output / "states.pt"),
            logits=_binding(output / "logits.pt"),
            cache=_binding(output / "cache.pt"),
            raw=_binding(output / "raw.json"),
            elapsed_seconds=time.monotonic() - began,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        )
        _write(output / "receipt.json", receipt)
        return receipt
    except BaseException as exc:
        receipt.update(status="technical_invalid", error=repr(exc), elapsed_seconds=time.monotonic() - began)
        _write(output / "receipt.json", receipt)
        raise
    finally:
        for handle in handles:
            handle.remove()


def _cpu_check(path: Path) -> None:
    if path.exists():
        raise FileExistsError(path)
    scores = torch.tensor([[8.0, 1.0, 7.0], [6.0, 5.0, 4.0]])
    forced = _force_target(scores, TARGET_POSITION if TARGET_POSITION < scores.shape[0] else 1, 2)
    assert torch.equal(forced[0], scores[0])
    assert int(torch.argmax(forced[1]).item()) == 2
    hidden = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    replacement = torch.full((4,), -1.0)
    transformed = _replace_last_target(hidden, 1, replacement)
    assert torch.equal(hidden[0], transformed[0])
    assert torch.equal(hidden[1, :-1], transformed[1, :-1])
    assert torch.equal(transformed[1, -1], replacement)
    assert _coordinate_values([ROW_OPEN, 22592, REF_END, BOX_START, 78 + 151670, 609 + 151670, 106 + 151670, 641 + 151670, BOX_END]) == EXPECTED_S_COORDS
    assert _condition_spec("residual-S-to-F-layer13") == {"mode": "residual", "route": "F", "donor": "S", "depth": 13}
    try:
        _replace_last_target(hidden, 1, torch.zeros(5))
    except ValueError:
        pass
    else:
        raise AssertionError("invalid residual shape was accepted")
    _write(
        path,
        {
            "status": "passed",
            "target_only_logits_force": True,
            "target_only_last_token_residual": True,
            "prefix_length": RELEASE_OFFSET,
            "fork_offset": FORK_OFFSET,
            "decoder_layers": list(DEPTHS),
            "cache_offsets": list(CACHE_OFFSETS),
            "no_parameter_mutation_path": True,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path)
    parser.add_argument("--condition")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--donor-states", type=Path)
    parser.add_argument("--rebuild-from", type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--artifact-root", type=Path, default=STAGE2_ROOT)
    parser.add_argument("--cpu-check", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare_stage2(args.artifact_root, overwrite=args.refresh)
        return
    if args.cpu_check:
        _cpu_check(args.output or args.artifact_root / "cpu-check.json")
        return
    if args.panel is None or args.condition is None or args.output is None:
        parser.error("--panel, --condition, and --output are required")
    _run(
        panel_path=args.panel,
        condition=args.condition,
        output=args.output,
        donor_states_path=args.donor_states,
        rebuild_from=args.rebuild_from,
    )


if __name__ == "__main__":
    main()
