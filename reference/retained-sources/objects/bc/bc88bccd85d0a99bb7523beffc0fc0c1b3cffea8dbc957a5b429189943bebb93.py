"""Native short ancestral draws for conditional numerical-repeat states.

The panel chooses states and supplies their histories.  This entrypoint only
rebuilds the current native image/prompt state, conditions on the frozen
description, and samples the bounded row tail with the existing HF helpers.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from probes.human13.runtime import _build_requests
from src.config.fingerprint import sha256_file, sha256_json
from src.config.inference import InferConfig
from src.data import load_raw_examples
from src.data.examples import raw_example_from_jsonl_row
from src.inference.runtime import assemble_frontend
from src.qwen.generation import NativeGenerationPolicy, generate_continuations
from src.qwen.native import NativeBatch, NativeRequest, exact_history_inputs, model_device, prepare_native_inputs


UNIT_ID = "2026-09-19-recurrence-conditional-mass"
DRAW_SCHEMA = "recurrence_conditional_mass.draws.v1"
MAX_STATES = 48
DRAWS_PER_STATE = 256
DEFAULT_CHUNK = 8
SEED_NAMESPACE = "coordexp-recurrence-conditional-mass-v1"
SPECIAL_TOKENS = (
    "<|object_ref_start|>",
    "<|object_ref_end|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|im_end|>",
)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def derive_stream_seed(*, state_id: str, model: str, image_id: Any, chunk_index: int) -> int:
    if not state_id or not model or isinstance(chunk_index, bool) or chunk_index < 0:
        raise ValueError("invalid stream-seed fields")
    material = f"{SEED_NAMESPACE}\0{UNIT_ID}\0{state_id}\0{model}\0{image_id}\0{chunk_index}"
    return int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], "big") & ((1 << 63) - 1)


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _states(panel: Mapping[str, Any]) -> list[dict[str, Any]]:
    if "states" in panel:
        value = panel.get("states")
    elif "all_boundaries" in panel:
        # The final recurrence census keeps old and prospective boundaries in
        # separate fields and publishes their frozen union explicitly.
        value = panel.get("all_boundaries")
    else:
        value = panel.get("panel", panel.get("existing_boundaries"))
    if isinstance(value, Mapping):
        value = value.get("states", value.get("entries"))
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("shared panel lacks a states sequence")
    result = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(f"panel state {index} is not an object")
        state = dict(item)
        state.setdefault("state_id", state.get("id", f"state-{index:03d}"))
        if not state.get("state_id"):
            raise ValueError(f"panel state {index} has no state_id")
        _bind_source_example_id(state)
        # Lane A's frozen prelaunch panel preserves the original boundary
        # records rather than duplicating derived histories.  Project only
        # those fields here; the source bytes remain bound to the panel.
        if "prefix_token_ids" not in state and isinstance(state.get("native_tokens"), Sequence):
            source_row = state.get("source_row")
            next_row = state.get("next_row")
            if (
                isinstance(source_row, Mapping)
                and isinstance(source_row.get("end"), int)
                and isinstance(next_row, Mapping)
                and isinstance(next_row.get("start"), int)
                and int(next_row["start"]) == int(source_row["end"])
            ):
                native = tuple(int(v) for v in state["native_tokens"])
                boundary = int(source_row["end"])
                if 0 <= boundary <= len(native):
                    description_tokens = tuple(int(v) for v in next_row.get("description_tokens", ()))
                    state["prefix_token_ids"] = list(native[:boundary])
                    state["description_token_ids"] = list(description_tokens)
                    state["event_row"] = copy.deepcopy(dict(next_row))
                    state.setdefault("boundary_source", "shared_panel.existing_boundaries.source_row.end")
                    state["boundary_prefix_source_row_end"] = boundary
                    state.setdefault("source_row_bins", list(source_row.get("values", ())))
                    derived = _derive_repeat_union_from_tokens(
                        native,
                        source_start=boundary,
                        description_tokens=description_tokens,
                    )
                    state.setdefault("repeat_union_bins", derived)
                    state.setdefault("literal_repeat_union_bins", derived)
        source_row = state.get("source_row")
        next_row = state.get("next_row", state.get("event_row"))
        explicit_prefix = state.get("prefix_token_ids")
        if (
            isinstance(explicit_prefix, Sequence)
            and not isinstance(explicit_prefix, (str, bytes))
            and isinstance(source_row, Mapping)
            and isinstance(source_row.get("end"), int)
            and isinstance(next_row, Mapping)
            and isinstance(next_row.get("start"), int)
        ):
            if int(next_row["start"]) != int(source_row["end"]):
                raise ValueError(f"{state['state_id']}: saved next-row boundary does not equal source_row.end")
            if len(explicit_prefix) != int(source_row["end"]):
                raise ValueError(f"{state['state_id']}: explicit prefix does not end at source_row.end")
        result.append(state)
    if len(result) > MAX_STATES:
        raise ValueError(f"shared panel exceeds {MAX_STATES} states")
    return result


def _bind_source_example_id(state: dict[str, Any]) -> None:
    """Bind a state to split-aware source identity before model lookup."""

    if any(state.get(name) for name in ("source_example_id", "example_id", "row_id")):
        return
    raw_path = state.get("raw_path")
    if not isinstance(raw_path, str) or not raw_path:
        return
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        return
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return
    rows = payload.get("rows") if isinstance(payload, Mapping) else None
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return
    image_id = state.get("image_id")
    batch_index = state.get("batch_index")
    candidate = None
    if isinstance(batch_index, int) and 0 <= batch_index < len(rows):
        row = rows[batch_index]
        if isinstance(row, Mapping) and (image_id is None or row.get("image_id") == image_id):
            candidate = row
    if candidate is None:
        matches = [
            row for row in rows
            if isinstance(row, Mapping) and image_id is not None and row.get("image_id") == image_id
        ]
        if len(matches) == 1:
            candidate = matches[0]
    if isinstance(candidate, Mapping) and isinstance(candidate.get("row_id"), str):
        state["source_example_id"] = candidate["row_id"]
        state.setdefault("source_identity_binding", str(path))


def _state_source_example_id(state: Mapping[str, Any]) -> str:
    nested = state.get("source_image_identity")
    if isinstance(nested, Mapping):
        for name in ("example_id", "row_id", "source_example_id"):
            value = nested.get(name)
            if isinstance(value, str) and value and not value.strip().isdigit():
                return value
    for name in ("source_example_id", "example_id", "row_id"):
        value = state.get(name)
        if isinstance(value, str) and value and not value.strip().isdigit():
            return value
    raise ValueError(
        f"{state.get('state_id')}: split-aware source example identity is required; numeric image_id fallback is forbidden"
    )


def _derive_repeat_union_from_tokens(
    tokens: Sequence[int], *, source_start: int, description_tokens: tuple[int, ...]
) -> list[list[int]]:
    """Recover prior same-description <=8-bin boxes from a bound native route."""

    rows: list[tuple[int, tuple[int, ...], tuple[int, int, int, int]]] = []
    index = 0
    while index + 8 < len(tokens):
        if int(tokens[index]) != 151646:
            index += 1
            continue
        try:
            ref_end = next(i for i in range(index + 1, len(tokens)) if int(tokens[i]) == 151647)
        except StopIteration:
            break
        if ref_end + 6 >= len(tokens) or int(tokens[ref_end + 1]) != 151648:
            index += 1
            continue
        coord = tuple(int(v) - 151670 for v in tokens[ref_end + 2: ref_end + 6])
        if len(coord) != 4 or any(v < 0 or v > 999 for v in coord) or int(tokens[ref_end + 6]) != 151649:
            index = ref_end + 1
            continue
        rows.append((index, tuple(int(v) for v in tokens[index + 1:ref_end]), coord))
        index = ref_end + 7
    prior = [box for start, desc, box in rows if start < source_start and desc == description_tokens]
    if not prior:
        return []
    # Keep the historical boxes themselves frozen.  The <=8-bin predicate is
    # applied to each sampled box by the reducer, so a draw near multiple
    # members is still one event rather than several pair edges.
    return [list(box) for box in dict.fromkeys(prior)]


def _first(state: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in state and state[name] is not None:
            return state[name]
    return None


def _prefix(state: Mapping[str, Any]) -> tuple[int, ...]:
    value = _first(state, "prefix_token_ids", "native_prefix_token_ids", "history_token_ids")
    if value is None and isinstance(state.get("prefix"), Mapping):
        value = _first(_require_mapping(state["prefix"], "prefix"), "token_ids", "generated_token_ids")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{state.get('state_id')}: explicit prefix_token_ids are required")
    result = tuple(int(v) for v in value)
    if any(v < 0 for v in result):
        raise ValueError("prefix token IDs must be nonnegative")
    return result


def _repeat_union(state: Mapping[str, Any], *, literal: bool = False) -> list[list[int]]:
    names = (
        ("literal_repeat_union_bins", "literal_invalid_repeat_union_bins")
        if literal
        else ("repeat_union_bins", "same_description_repeat_union_bins", "historical_repeat_union_bins")
    )
    value = _first(state, *names)
    if value is None:
        nested = state.get("repeat_union")
        if isinstance(nested, Mapping):
            value = _first(nested, "literal_bins" if literal else "bins", "boxes")
    if value is None:
        raise ValueError(f"{state.get('state_id')}: frozen repeat union is missing")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("repeat union must be a sequence")
    result = []
    for index, box in enumerate(value):
        if not isinstance(box, Sequence) or isinstance(box, (str, bytes)) or len(box) != 4:
            raise ValueError(f"repeat union member {index} is not a four-bin box")
        bins = [int(v) for v in box]
        if any(v < 0 or v > 999 for v in bins):
            raise ValueError("repeat union bins must be in [0,999]")
        result.append(bins)
    return result


def _config_for_state(
    state: Mapping[str, Any], panel: Mapping[str, Any], sources: Mapping[str, Any]
) -> dict[str, Any]:
    direct = state.get("config")
    if isinstance(direct, Mapping):
        return copy.deepcopy(dict(direct))
    key = str(_first(state, "model_config_key", "model", "config_key"))
    configs = panel.get("configs", sources.get("configs", sources.get("models")))
    if not isinstance(configs, Mapping):
        # The shared mechanism panel binds the mature and feedback source
        # panels by digest.  Reuse their already-bound model configs instead
        # of reconstructing checkpoint paths from prose.
        candidates: list[Mapping[str, Any]] = []
        for binding_name in ("mature_panel", "feedback_panel"):
            binding = sources.get(binding_name)
            if isinstance(binding, Mapping) and isinstance(binding.get("path"), str):
                path = Path(binding["path"]).expanduser().resolve(strict=True)
                payload = json.loads(path.read_text())
                if isinstance(payload, Mapping) and isinstance(payload.get("configs"), Mapping):
                    candidates.append(payload["configs"])
        for candidate in candidates:
            if key in candidate and isinstance(candidate[key], Mapping):
                return copy.deepcopy(dict(candidate[key]))
    if not isinstance(configs, Mapping) or key not in configs or not isinstance(configs[key], Mapping):
        raise ValueError(f"{state.get('state_id')}: no config binding for model key {key}")
    return copy.deepcopy(dict(configs[key]))


def _source_jsonl(config_payload: Mapping[str, Any], sources: Mapping[str, Any]) -> Path:
    data = config_payload.get("data")
    candidate = data.get("input_jsonl") if isinstance(data, Mapping) else None
    if candidate is None:
        candidate = _first(sources, "input_jsonl", "source_jsonl")
    if not isinstance(candidate, str) or not candidate:
        raise ValueError("no bound source JSONL")
    return Path(candidate).expanduser().resolve(strict=True)


def _source_jsonls(config_payload: Mapping[str, Any], sources: Mapping[str, Any]) -> tuple[Path, ...]:
    """Return the primary source plus already-bound supplemental JSONL sources.

    Mature and feedback panels bind their configs through panel JSON files.  A
    final shared source object can additionally bind a qualified production
    source.  Loading those paths here keeps image identity lookup exact while
    retaining the config's own source as the first choice for duplicate IDs.
    """

    primary = _source_jsonl(config_payload, sources)
    candidates: list[Path] = [primary]
    # Only direct source bindings are supplemental inputs.  Recursing through
    # historical panel JSONs would pull in annotation snapshots and foreign
    # scenes whose image roots are intentionally not part of this unit.
    for value in sources.values():
        path = value.get("path") if isinstance(value, Mapping) else value
        if not isinstance(path, str) or not path:
            continue
        resolved = Path(path).expanduser().resolve()
        if resolved.suffix == ".jsonl" and resolved.exists():
            candidates.append(resolved)
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        path = path.resolve()
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return tuple(unique)


def _selected_manifest_bindings(
    states: Sequence[Mapping[str, Any]], sources: Mapping[str, Any], missing_ids: set[str]
) -> list[dict[str, Any]]:
    """Resolve prospective rows through the frozen manifest, by split and ID."""

    manifest = sources.get("processed_manifest")
    if not isinstance(manifest, Mapping):
        return []
    selected = manifest.get("selected_rows")
    source_files = manifest.get("source_files")
    if not isinstance(selected, Sequence) or isinstance(selected, (str, bytes)):
        return []
    if not isinstance(source_files, Sequence) or isinstance(source_files, (str, bytes)):
        return []
    by_split: dict[str, Path] = {}
    for item in source_files:
        if not isinstance(item, Mapping) or not isinstance(item.get("path"), str):
            continue
        path = Path(item["path"]).expanduser().resolve(strict=True)
        name = path.name
        if name.startswith("train."):
            by_split["train"] = path
        elif name.startswith("val."):
            by_split["val"] = path
    selected_by_key: dict[tuple[str, int], Mapping[str, Any]] = {}
    for item in selected:
        if not isinstance(item, Mapping):
            continue
        key = item.get("key")
        if not isinstance(key, Sequence) or isinstance(key, (str, bytes)) or len(key) < 2:
            continue
        try:
            selected_by_key[(str(key[0]), int(key[1]))] = item
        except (TypeError, ValueError):
            continue
    bindings: list[dict[str, Any]] = []
    seen: set[str] = set()
    for state in states:
        source_id = _state_source_example_id(state)
        if source_id not in missing_ids or source_id in seen:
            continue
        split = state.get("split")
        image_id = _first(state, "image_id", "source_image_id")
        if not isinstance(split, str) or image_id is None:
            raise ValueError(f"{state.get('state_id')}: missing split/image identity for processed manifest")
        try:
            key = (split, int(image_id))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{state.get('state_id')}: invalid split/image identity") from exc
        selected_row = selected_by_key.get(key)
        source_path = by_split.get(split)
        if source_path is None:
            raise ValueError(
                f"{state.get('state_id')}: source split is absent from frozen processed source files"
            )
        line_number = None if selected_row is None else selected_row.get("source_line_number")
        expected_hash = None if selected_row is None else selected_row.get("source_line_sha256")
        if line_number is not None and (not isinstance(line_number, int) or line_number <= 0):
            raise ValueError(f"{state.get('state_id')}: processed manifest row binding is incomplete")
        if expected_hash is not None and not isinstance(expected_hash, str):
            raise ValueError(f"{state.get('state_id')}: processed manifest source hash is invalid")
        bindings.append(
            {
                "source_example_id": source_id,
                "split": split,
                "image_id": int(image_id),
                "path": source_path,
                "line_number": line_number,
                "source_line_sha256": expected_hash,
                "manifest_selected": selected_row is not None,
            }
        )
        seen.add(source_id)
    return bindings


def _load_selected_manifest_examples(bindings: Sequence[Mapping[str, Any]]) -> tuple[tuple[Any, ...], list[dict[str, Any]]]:
    """Read only frozen selected rows needed by a model runtime."""

    targets: dict[Path, list[Mapping[str, Any]]] = {}
    for binding in bindings:
        path = Path(str(binding["path"])).expanduser().resolve(strict=True)
        targets.setdefault(path, []).append(binding)
    examples: list[Any] = []
    receipts: list[dict[str, Any]] = []
    for path, line_targets in targets.items():
        line_targets = targets[path]
        by_line = {
            int(binding["line_number"]): binding
            for binding in line_targets
            if binding.get("line_number") is not None
        }
        fallback = [binding for binding in line_targets if binding.get("line_number") is None]
        found_ids: set[str] = set()
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                binding = by_line.get(line_number)
                if binding is None and not fallback:
                    continue
                line = raw_line.rstrip("\n")
                payload = json.loads(line)
                if binding is None:
                    image_id = payload.get("image_id") if isinstance(payload, Mapping) else None
                    split = payload.get("metadata", {}).get("split") if isinstance(payload, Mapping) and isinstance(payload.get("metadata"), Mapping) else None
                    matches = [
                        candidate
                        for candidate in fallback
                        if int(candidate["image_id"]) == int(image_id)
                        and str(candidate["split"]) == str(split)
                        and str(candidate["source_example_id"]) not in found_ids
                    ]
                    if not matches:
                        continue
                    binding = matches[0]
                actual_hash = hashlib.sha256(line.encode("utf-8")).hexdigest()
                expected_hash = binding.get("source_line_sha256")
                if expected_hash is not None and actual_hash != str(expected_hash):
                    raise ValueError(
                        f"{binding['source_example_id']}: processed source line hash changed"
                    )
                example = raw_example_from_jsonl_row(
                    payload,
                    jsonl_path=path,
                    row_number=line_number,
                    raw_line=line,
                )
                if str(example.example_id) != str(binding["source_example_id"]):
                    raise ValueError(
                        f"{binding['source_example_id']}: processed source row identity disagrees"
                    )
                examples.append(example)
                receipts.append(
                    {
                        "source_example_id": str(binding["source_example_id"]),
                        "split": str(binding["split"]),
                        "image_id": int(binding["image_id"]),
                        "path": str(path),
                        "line_number": line_number,
                        "source_line_sha256": actual_hash,
                        "manifest_selected": bool(binding.get("manifest_selected")),
                    }
                )
                found_ids.add(str(binding["source_example_id"]))
    if len(examples) != len(bindings):
        found = {item["source_example_id"] for item in receipts}
        missing = sorted(str(item["source_example_id"]) for item in bindings if item["source_example_id"] not in found)
        raise ValueError(f"processed source rows were not found: {missing}")
    return tuple(examples), receipts


def _prepare_config(payload: Mapping[str, Any], *, artifact_root: Path) -> InferConfig:
    config = copy.deepcopy(dict(payload))
    config.setdefault("schema_version", 1)
    config.setdefault("debug", {})
    config["debug"].update(smoke=True, dry_run=False)
    config.setdefault("generation", {})
    config["generation"].update(
        batch_size=1,
        max_new_tokens=8,
        temperature=0.0,
        top_p=1.0,
        n=1,
        repetition_penalty=1.0,
    )
    config.setdefault("run", {})
    config["run"].update(
        artifact_root=str(artifact_root), output_dir=None, collision_policy="fail"
    )
    config.setdefault("scoring", {"enabled": True})
    config.setdefault("artifacts", {"write_token_trace": True, "write_parse_diagnostics": True, "include_raw_model_logprob": False})
    return InferConfig.model_validate(config)


def _model_runtime(
    config_payload: Mapping[str, Any],
    sources: Mapping[str, Any],
    *,
    output_root: Path,
    model_key: str,
    states: Sequence[Mapping[str, Any]],
) -> tuple[Any, Any, Any, Any, Any, dict[str, Any], dict[str, int]]:
    config = _prepare_config(config_payload, artifact_root=output_root)
    frontend = assemble_frontend(
        config,
        generation_config_fingerprint=sha256_json(config.generation.model_dump(mode="json")),
    )
    # The mature tied/untied bindings use a split input/output delta for the
    # untied checkpoint.  Reuse that accepted loader instead of routing the
    # untied sidecar through the generic shared-delta attachment path.
    from probes.training_set_completion.untied_shared import load_model as load_frozen_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    qwen, loader_identity = load_frozen_model(model_key, device)
    model = qwen.model
    if model is None:
        raise RuntimeError("source loader returned no model")
    model.eval()
    runtime_counters = {"model_forwards": 0, "vision_forwards": 0}

    def count_forward(*_: Any) -> None:
        runtime_counters["model_forwards"] += 1

    def count_vision(*_: Any) -> None:
        runtime_counters["vision_forwards"] += 1

    handles = [model.register_forward_pre_hook(count_forward)]
    visual = getattr(getattr(model, "model", None), "visual", None)
    if visual is not None:
        handles.append(visual.register_forward_pre_hook(count_vision))
    source_jsonls = _source_jsonls(config_payload, sources)
    raw_examples_list: list[Any] = []
    requests_list: list[Any] = []
    by_image: dict[str, tuple[Any, Any, str]] = {}
    for source_jsonl in source_jsonls:
        batch = tuple(load_raw_examples(source_jsonl))
        batch_requests = _build_requests(config, frontend, batch)
        raw_examples_list.extend(batch)
        requests_list.extend(batch_requests)
        for raw, request in zip(batch, batch_requests, strict=True):
            # The config's source is first.  Supplemental sources may overlap
            # it; a duplicate identity must retain the first bound row.
            by_image.setdefault(
                str(raw.example_id),
                (raw, request, str(source_jsonl)),
            )
    missing_ids = {
        _state_source_example_id(state)
        for state in states
        if _state_source_example_id(state) not in by_image
    }
    selected_bindings = _selected_manifest_bindings(states, sources, missing_ids)
    if missing_ids and not selected_bindings:
        raise ValueError(
            f"{model_key}: bound source examples absent from runtime and processed manifest: {sorted(missing_ids)}"
        )
    if selected_bindings:
        selected_examples, selected_receipts = _load_selected_manifest_examples(selected_bindings)
        selected_requests = _build_requests(config, frontend, selected_examples)
        raw_examples_list.extend(selected_examples)
        requests_list.extend(selected_requests)
        selected_path_by_id = {
            str(item["source_example_id"]): str(item["path"])
            for item in selected_receipts
        }
        for raw, request in zip(selected_examples, selected_requests, strict=True):
            by_image.setdefault(
                str(raw.example_id),
                (raw, request, selected_path_by_id[str(raw.example_id)]),
            )
    else:
        selected_receipts = []
    missing_after = {
        _state_source_example_id(state)
        for state in states
        if _state_source_example_id(state) not in by_image
    }
    if missing_after:
        raise ValueError(f"{model_key}: selected source binding still missing: {sorted(missing_after)}")
    raw_examples = tuple(raw_examples_list)
    requests = tuple(requests_list)
    identity = {
        "model_key": model_key,
        "loader_identity": loader_identity,
        "base_model": str(config.model.base_model),
        "base_model_sha256": None,
        "adapter": None if config.adapter is None else config.adapter.model_dump(mode="json"),
        "embedding_delta": None if config.embedding_delta is None else config.embedding_delta.model_dump(mode="json"),
        "device": str(device),
        "dtype": str(next(model.parameters()).dtype),
        "attn_implementation": getattr(getattr(model, "config", None), "_attn_implementation", None),
        "tokenizer_class": type(qwen.tokenizer).__name__,
        "processor_class": type(qwen.processor).__name__,
        "source_jsonl": str(_source_jsonl(config_payload, sources)),
        "source_jsonls": [str(path) for path in source_jsonls],
        "selected_manifest_rows": selected_receipts,
    }
    return model, qwen, raw_examples, requests, by_image, identity, runtime_counters


def _token_id(tokenizer: Any, token: str) -> int:
    value = tokenizer.convert_tokens_to_ids(token)
    if value is None or isinstance(value, bool) or int(value) < 0:
        raise ValueError(f"tokenizer lacks atomic token {token}")
    decoded = tokenizer.decode([int(value)], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if decoded != token:
        raise ValueError(f"tokenizer token is not atomic: {token!r} -> {decoded!r}")
    return int(value)


def _row_condition(
    tokenizer: Any,
    description: str | None,
    *,
    description_token_ids: Sequence[int] | None = None,
) -> tuple[tuple[int, ...], dict[str, Any]]:
    if description_token_ids is not None:
        if isinstance(description_token_ids, (str, bytes)):
            raise ValueError("description_token_ids must be a token sequence")
        frozen_description_ids = tuple(int(v) for v in description_token_ids)
        if not frozen_description_ids or any(v < 0 for v in frozen_description_ids):
            raise ValueError("description_token_ids must be nonempty and nonnegative")
        decoded = tokenizer.decode(
            list(frozen_description_ids),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if not isinstance(decoded, str) or not decoded.strip():
            raise ValueError("frozen description tokens decode to empty text")
        if description is not None:
            observed = tuple(int(v) for v in tokenizer.encode(description, add_special_tokens=False))
            if observed != frozen_description_ids:
                raise ValueError("text description disagrees with frozen description_token_ids")
        description = decoded
        description_ids = frozen_description_ids
    else:
        if not isinstance(description, str) or not description.strip():
            raise ValueError("state description must be nonempty")
        description_ids = tuple(int(v) for v in tokenizer.encode(description, add_special_tokens=False))
        if not description_ids:
            raise ValueError("description tokenization is empty")
    ids = {
        token: _token_id(tokenizer, token)
        for token in SPECIAL_TOKENS
    }
    prefix = (
        ids["<|object_ref_start|>"],
        *description_ids,
        ids["<|object_ref_end|>"],
        ids["<|box_start|>"],
    )
    coord_ids = tuple(_token_id(tokenizer, f"<|coord_{index}|>") for index in range(1000))
    tail = tuple((*[coord_ids[index] for index in (0, 1, 2, 3)], ids["<|box_end|>"]))
    serialized = (
        "<|object_ref_start|>"
        + description
        + "<|object_ref_end|><|box_start|>"
        + "<|coord_0|><|coord_1|><|coord_2|><|coord_3|><|box_end|>"
    )
    encoded = tuple(int(v) for v in tokenizer.encode(serialized, add_special_tokens=False))
    if encoded != (*prefix, *tail):
        raise ValueError("actual serializer is not row-prefix plus five-token tail")
    return prefix, {
        "description": description,
        "description_token_ids": list(description_ids),
        "row_prefix_token_ids": list(prefix),
        "coordinate_token_ids": list(coord_ids),
        "coordinate_token_ids_sha256": sha256_json(coord_ids),
        "box_end_token_id": ids["<|box_end|>"],
        "im_end_token_id": ids["<|im_end|>"],
        "serializer_probe_text": serialized,
        "serializer_probe_token_ids": list(encoded),
        "serializer_probe_token_count": len(encoded),
        "bounded_horizon": len(tail),
    }


def _native_request(qwen: Any, raw: Any, request: Any, *, request_id: str) -> NativeRequest:
    return NativeRequest(
        request_id=request_id,
        chat_text=request.chat_text,
        image=request.image_path,
        expected_token_ids=tuple(request.expected_executed_prompt_token_ids),
        expected_image_grid=tuple(request.expected_image_grid_thw) if request.expected_image_grid_thw else None,
        expected_image_size=(request.decoded_image_width, request.decoded_image_height),
        image_sha256=request.image_sha256,
        logical_transform=request.logical_transform_id,
    )


def _row_boundary_evidence(
    model: Any,
    qwen: Any,
    native: Any,
    *,
    prompt_ids: Sequence[int],
    prefix_ids: Sequence[int],
    row_prefix: Sequence[int],
    description_ids: Sequence[int],
) -> dict[str, Any]:
    from torch.nn import functional as F

    history = (*prompt_ids, *prefix_ids)
    inputs = exact_history_inputs(model, native.inputs, [history], pad_token_id=0, logits_to_keep=1)
    with torch.inference_mode():
        output = model(**inputs)
    logits = output.logits.float()[0, -1]
    logp = F.log_softmax(logits, dim=-1)
    probabilities = logp.exp()
    opener = int(row_prefix[0])
    eos = _token_id(qwen.tokenizer, "<|im_end|>")
    first_desc = int(description_ids[0])
    ranking = lambda token: int((logits > logits[int(token)]).sum().item()) + 1
    return {
        "row_boundary_history_token_count": len(history),
        "opener_token_id": opener,
        "opener_logprob": float(logp[opener].item()),
        "opener_rank": ranking(opener),
        "eos_token_id": eos,
        "eos_logprob": float(logp[eos].item()),
        "eos_rank": ranking(eos),
        "description_first_token_id": first_desc,
        "description_entry_logprob_after_opener": None,
        "description_first_token_rank_after_opener": None,
        "full_vocabulary_softmax_finite": bool(
            torch.isfinite(logits).all().item() and torch.isfinite(probabilities).all().item()
        ),
        "full_vocabulary_probability_mass": float(probabilities.sum().item()),
        "full_vocabulary_size": int(logits.numel()),
    }


def _description_entry_evidence(
    model: Any,
    native: Any,
    *,
    prompt_ids: Sequence[int],
    prefix_ids: Sequence[int],
    row_prefix: Sequence[int],
    description_ids: Sequence[int],
) -> dict[str, Any]:
    from torch.nn import functional as F

    history = (*prompt_ids, *prefix_ids, *row_prefix[:1])
    inputs = exact_history_inputs(model, native.inputs, [history], pad_token_id=0, logits_to_keep=1)
    with torch.inference_mode():
        logits = model(**inputs).logits.float()[0, -1]
    logp = F.log_softmax(logits, dim=-1)
    token = int(description_ids[0])
    return {
        "description_entry_logprob_after_opener": float(logp[token].item()),
        "description_first_token_rank_after_opener": int((logits > logits[token]).sum().item()) + 1,
    }


def _source_score_batch4_qualification(
    model: Any,
    qwen: Any,
    raw: Any,
    request: Any,
    native_one: Any,
    *,
    prompt_ids: Sequence[int],
    prefix_ids: Sequence[int],
    state: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify the duplicated native batch path against the accepted bs4 trace."""

    requests = tuple(
        _native_request(qwen, raw, request, request_id=f"{state['state_id']}:parity:{index}")
        for index in range(4)
    )
    native_four = prepare_native_inputs(
        qwen.processor,
        requests,
        device=model_device(model),
        record_media_identity=True,
    )
    if any(row != tuple(prompt_ids) for row in native_four.prompt_token_ids):
        raise RuntimeError("batch-4 native prompt expansion does not preserve the source prompt")
    requests_eight = tuple(
        _native_request(qwen, raw, request, request_id=f"{state['state_id']}:parity8:{index}")
        for index in range(8)
    )
    native_eight = prepare_native_inputs(
        qwen.processor,
        requests_eight,
        device=model_device(model),
        record_media_identity=True,
    )
    if any(row != tuple(prompt_ids) for row in native_eight.prompt_token_ids):
        raise RuntimeError("batch-8 native prompt expansion does not preserve the source prompt")
    history = (*prompt_ids, *prefix_ids)
    inputs_one = exact_history_inputs(
        model,
        native_one.inputs,
        [history],
        pad_token_id=0,
        logits_to_keep=1,
    )
    inputs_four = exact_history_inputs(
        model,
        native_four.inputs,
        [history] * 4,
        pad_token_id=0,
        logits_to_keep=1,
    )
    inputs_eight = exact_history_inputs(
        model,
        native_eight.inputs,
        [history] * 8,
        pad_token_id=0,
        logits_to_keep=1,
    )
    with torch.inference_mode():
        logits_one = model(**inputs_one).logits.float()[0, -1]
        logits_four = model(**inputs_four).logits.float()[:, -1]
        logits_eight = model(**inputs_eight).logits.float()[:, -1]
    batch_delta = float(torch.max(torch.abs(logits_four - logits_one.unsqueeze(0))).item())
    row_delta = float(
        torch.max(torch.abs(logits_four - logits_four[0].unsqueeze(0))).item()
    )
    batch_eight_delta = float(
        torch.max(torch.abs(logits_eight - logits_one.unsqueeze(0))).item()
    )
    row_eight_delta = float(
        torch.max(torch.abs(logits_eight - logits_eight[0].unsqueeze(0))).item()
    )
    position_one = inputs_one.get("position_ids")
    position_four = inputs_four.get("position_ids")
    position_delta = None
    if isinstance(position_one, torch.Tensor) and isinstance(position_four, torch.Tensor):
        position_delta = int(
            torch.max(
                torch.abs(position_four[0].to(dtype=torch.int64) - position_one[0].to(dtype=torch.int64))
            ).item()
        )
    position_eight = inputs_eight.get("position_ids")
    position_eight_delta = None
    if isinstance(position_one, torch.Tensor) and isinstance(position_eight, torch.Tensor):
        position_eight_delta = int(
            torch.max(
                torch.abs(position_eight[0].to(dtype=torch.int64) - position_one[0].to(dtype=torch.int64))
            ).item()
        )
    media_one = None if native_one.media_sha256 is None else list(native_one.media_sha256)
    media_four = None if native_four.media_sha256 is None else list(native_four.media_sha256)
    tensor_dtypes = {
        key: str(value.dtype)
        for key, value in native_four.inputs.items()
        if isinstance(value, torch.Tensor) and key in {"input_ids", "attention_mask", "pixel_values"}
    }
    tensor_dtypes_eight = {
        key: str(value.dtype)
        for key, value in native_eight.inputs.items()
        if isinstance(value, torch.Tensor) and key in {"input_ids", "attention_mask", "pixel_values"}
    }
    if tensor_dtypes_eight != tensor_dtypes:
        raise RuntimeError("batch-4 and batch-8 native tensor dtypes differ")
    trace_path = state.get("trace_path")
    if not isinstance(trace_path, str) or not trace_path:
        raise RuntimeError("qualified source state lacks a bound native trace path")
    trace_file = Path(trace_path).expanduser().resolve(strict=True)
    trace_payload = json.loads(trace_file.read_text())
    steps = trace_payload.get("steps") if isinstance(trace_payload, Mapping) else None
    offset = len(prefix_ids)
    batch_index = state.get("batch_index")
    if (
        not isinstance(steps, Sequence)
        or isinstance(steps, (str, bytes))
        or not isinstance(batch_index, int)
        or offset < 0
        or offset >= len(steps)
    ):
        raise RuntimeError("accepted source trace lacks the corrected native boundary step")
    step = steps[offset]
    if not isinstance(step, Mapping):
        raise RuntimeError("accepted source trace boundary step is not an object")
    winners = step.get("raw_winners")
    chosen_raw_logits = step.get("chosen_raw_logits")
    if (
        not isinstance(winners, Sequence)
        or not isinstance(chosen_raw_logits, Sequence)
        or batch_index >= len(winners)
        or batch_index >= len(chosen_raw_logits)
    ):
        raise RuntimeError("accepted source trace boundary lacks raw winner evidence")
    trace_token = int(winners[batch_index])
    trace_logit = float(chosen_raw_logits[batch_index])
    observed_logit = float(logits_one[trace_token].item())
    observed_batch_eight_logit = float(logits_eight[0, trace_token].item())
    trace_report = {
        "status": "matched",
        "path": str(trace_file),
        "sha256": sha256_file(trace_file),
        "offset": offset,
        "batch_index": batch_index,
        "trace_token_id": trace_token,
        "observed_winner_token_id": int(torch.argmax(logits_one).item()),
        "batch8_observed_winner_token_id": int(torch.argmax(logits_eight[0]).item()),
        "trace_raw_logit": trace_logit,
        "observed_raw_logit": observed_logit,
        "batch8_raw_logit": observed_batch_eight_logit,
        "absolute_logit_delta": abs(observed_logit - trace_logit),
        "batch8_absolute_logit_delta": abs(observed_batch_eight_logit - trace_logit),
        "winner_agrees": int(torch.argmax(logits_one).item()) == trace_token,
        "batch8_winner_agrees": int(torch.argmax(logits_eight[0]).item()) == trace_token,
    }
    if (
        trace_report["absolute_logit_delta"] > 2e-4
        or trace_report["batch8_absolute_logit_delta"] > 2e-4
        or not trace_report["winner_agrees"]
        or not trace_report["batch8_winner_agrees"]
    ):
        raise RuntimeError("native source score disagrees with the accepted bs4 trace")
    if (
        batch_delta > 2e-4
        or row_delta > 2e-4
        or batch_eight_delta > 2e-4
        or row_eight_delta > 2e-4
        or position_delta not in (0, None)
        or position_eight_delta not in (0, None)
    ):
        raise RuntimeError("duplicated native batch-4/8 logits or positions are not source-parity stable")
    return {
        "status": "passed",
        "batch_size_one_prompt_token_count": len(native_one.prompt_token_ids[0]),
        "batch_size_four_prompt_token_count": [len(row) for row in native_four.prompt_token_ids],
        "batch_size_eight_prompt_token_count": [len(row) for row in native_eight.prompt_token_ids],
        "batch_logit_max_abs_delta": batch_delta,
        "within_batch_logit_max_abs_delta": row_delta,
        "batch8_logit_max_abs_delta": batch_eight_delta,
        "within_batch8_logit_max_abs_delta": row_eight_delta,
        "position_id_max_abs_delta": position_delta,
        "position8_id_max_abs_delta": position_eight_delta,
        "media_sha256_batch1": media_one,
        "media_sha256_batch4": media_four,
        "media_sha256_batch8": None if native_eight.media_sha256 is None else list(native_eight.media_sha256),
        "image_grid_batch1": [list(grid) if grid is not None else None for grid in native_one.image_grids],
        "image_grid_batch4": [list(grid) if grid is not None else None for grid in native_four.image_grids],
        "image_grid_batch8": [list(grid) if grid is not None else None for grid in native_eight.image_grids],
        "native_tensor_dtypes": tensor_dtypes,
        "native_tensor_dtypes_batch8": tensor_dtypes_eight,
        "source_trace": trace_report,
        "tolerance": 2e-4,
    }


def _saved_original_event(
    state: Mapping[str, Any],
    *,
    prefix_ids: Sequence[int],
    description_ids: Sequence[int],
    serializer: Mapping[str, Any],
    repeat_union: Sequence[Sequence[int]],
) -> dict[str, Any]:
    """Audit the saved native row at the exact conditioned prefix.

    A saved boundary is a source audit only.  If its prefix or description
    does not bind exactly, membership is ``None`` rather than an inferred
    greedy event.
    """

    native = state.get("native_tokens")
    source_row = state.get("event_row", state.get("next_row"))
    result: dict[str, Any] = {
        "available": isinstance(native, Sequence) and isinstance(source_row, Mapping),
        "native_entry_status": "unavailable",
        "event_membership": None,
        "invalid_geometry_near_repeat_membership": None,
        "exact_invalid_event_membership": None,
        "exact_prefix_match": False,
        "description_compatible": False,
        "mismatch_reason": None,
    }
    if not isinstance(native, Sequence) or isinstance(native, (str, bytes)):
        result["mismatch_reason"] = "saved_native_tokens_missing"
        return result
    if not isinstance(source_row, Mapping):
        result["mismatch_reason"] = "saved_source_row_missing"
        return result
    tokens = tuple(int(v) for v in native)
    start = source_row.get("start")
    end = source_row.get("end")
    saved_description = source_row.get("description_tokens")
    offsets = source_row.get("coordinate_offsets")
    if not isinstance(start, int) or not isinstance(end, int) or not (0 <= start < end <= len(tokens)):
        result["mismatch_reason"] = "saved_row_bounds_invalid"
        return result
    result["exact_prefix_match"] = tuple(tokens[:start]) == tuple(int(v) for v in prefix_ids)
    result["description_compatible"] = (
        isinstance(saved_description, Sequence)
        and not isinstance(saved_description, (str, bytes))
        and tuple(int(v) for v in saved_description) == tuple(int(v) for v in description_ids)
    )
    if not result["exact_prefix_match"] or not result["description_compatible"]:
        reasons = []
        if not result["exact_prefix_match"]:
            reasons.append("prefix_mismatch")
        if not result["description_compatible"]:
            reasons.append("description_mismatch")
        result["mismatch_reason"] = "+".join(reasons)
        result["native_entry_status"] = "conditioned_prefix_or_description_mismatch"
        return result
    if not isinstance(offsets, Sequence) or isinstance(offsets, (str, bytes)) or len(offsets) != 4:
        result["mismatch_reason"] = "saved_coordinate_offsets_missing"
        result["native_entry_status"] = "saved_row_unparseable"
        return result
    coordinate_offsets = tuple(int(v) for v in offsets)
    coord_ids = tuple(int(v) for v in serializer["coordinate_token_ids"])
    box_end_id = int(serializer["box_end_token_id"])
    if any(not (start <= offset < end) for offset in coordinate_offsets):
        result["mismatch_reason"] = "saved_coordinate_offset_outside_row"
        result["native_entry_status"] = "saved_row_unparseable"
        return result
    coordinate_tokens = tuple(tokens[offset] for offset in coordinate_offsets)
    if any(token not in coord_ids for token in coordinate_tokens):
        result["mismatch_reason"] = "saved_coordinate_token_invalid"
        result["native_entry_status"] = "grammar_escape"
        return result
    bins = tuple(coord_ids.index(token) for token in coordinate_tokens)
    terminator = coordinate_offsets[-1] + 1
    complete = terminator < end and tokens[terminator] == box_end_id
    geometry_valid = bins[0] < bins[2] and bins[1] < bins[3]
    repeat_tolerance = 8
    matches = [
        index
        for index, historical in enumerate(repeat_union)
        if max(abs(a - b) for a, b in zip(bins, historical)) <= repeat_tolerance
    ]
    exact_matches = [
        index
        for index, historical in enumerate(repeat_union)
        if tuple(bins) == tuple(int(value) for value in historical)
    ]
    result.update(
        {
            "saved_row_token_ids": list(tokens[start:end]),
            "saved_coordinate_bins": list(bins),
            "saved_terminator_token_id": tokens[terminator] if terminator < end else None,
            "complete_legal_bbox": bool(complete and geometry_valid),
            "geometry_valid": bool(geometry_valid),
            "terminator_present": bool(complete),
            "repeat_tolerance_bins": repeat_tolerance,
            "repeat_union_matches": matches,
            "exact_repeat_union_matches": exact_matches,
            "event_membership": bool(complete and geometry_valid and matches),
            "invalid_geometry_near_repeat_membership": bool(complete and not geometry_valid and matches),
            "exact_invalid_event_membership": bool(complete and not geometry_valid and exact_matches),
            "mismatch_reason": None,
            "native_entry_status": (
                "legal_repeat"
                if complete and geometry_valid and matches
                else "legal_nonrepeat"
                if complete and geometry_valid
                else "invalid_geometry_near_repeat"
                if complete and matches
                else "invalid_extent"
                if complete
                else "grammar_escape"
            ),
        }
    )
    return result


def _sample_chunk(
    model: Any,
    qwen: Any,
    raw: Any,
    request: Any,
    *,
    extension: Sequence[int],
    horizon: int,
    chunk_size: int,
    seed: int,
    state_id: str,
    chunk_index: int,
) -> list[dict[str, Any]]:
    device = model_device(model)
    requests = tuple(
        _native_request(qwen, raw, request, request_id=f"{state_id}:chunk{chunk_index}:draw{index}")
        for index in range(chunk_size)
    )
    native = prepare_native_inputs(qwen.processor, requests, device=device, record_media_identity=True)
    generated = generate_continuations(
        model,
        native,
        extensions=[tuple(int(v) for v in extension) for _ in requests],
        budgets=[horizon for _ in requests],
        eos_token_id=_token_id(qwen.tokenizer, "<|im_end|>"),
        pad_token_id=int(qwen.tokenizer.pad_token_id),
        policy=NativeGenerationPolicy(
            temperature=1.0,
            top_p=1.0,
            repetition_penalty=1.0,
            top_k=0,
            use_model_defaults=False,
        ),
        trace="policy",
        seed=seed,
    )
    rows: list[dict[str, Any]] = []
    for offset, result in enumerate(generated):
        trace = result.trace
        rows.append(
            {
                "draw_index": offset,
                "stream_seed": seed,
                "stream_chunk_index": chunk_index,
                "stream_offset": offset,
                "token_ids": [int(v) for v in result.token_ids],
                "stop_reason": result.stop_reason,
                "policy_logprobs": None if trace is None else [float(v) for v in result.policy_logprobs or ()],
            }
        )
    return rows


def _write_sampler_entry_receipt(
    output_root: Path,
    *,
    state: Mapping[str, Any],
    source_example_id: str,
    model_key: str,
    serializer: Mapping[str, Any],
    prefix_ids: Sequence[int],
    row_prefix: Sequence[int],
    chunk: Sequence[Mapping[str, Any]],
    chunk_index: int,
    chunk_size: int,
) -> None:
    """Persist and validate the first counted native sampling chunk."""

    if chunk_index != 0 or chunk_size != DEFAULT_CHUNK or len(chunk) != DEFAULT_CHUNK:
        raise ValueError("sampler-entry receipt requires the first complete batch8 chunk")
    receipt_path = output_root / "sampler-entry.json"
    if receipt_path.exists():
        raise FileExistsError(f"sampler-entry receipt already exists: {receipt_path}")
    expected_offsets = list(range(DEFAULT_CHUNK))
    actual_offsets = [int(row.get("stream_offset", -1)) for row in chunk]
    if actual_offsets != expected_offsets:
        raise ValueError("first sampler chunk has noncontiguous RNG offsets")
    seeds = {int(row.get("stream_seed", -1)) for row in chunk}
    if len(seeds) != 1 or next(iter(seeds)) < 0:
        raise ValueError("first sampler chunk does not have one valid stream seed")
    token_rows: list[dict[str, Any]] = []
    coordinate_ids = {int(value) for value in serializer["coordinate_token_ids"]}
    box_end = int(serializer["box_end_token_id"])
    escape_count = 0
    finite_logprob_count = 0
    for row in chunk:
        token_ids = row.get("token_ids")
        logprobs = row.get("policy_logprobs")
        if not isinstance(token_ids, Sequence) or isinstance(token_ids, (str, bytes)):
            raise ValueError("first sampler chunk lacks raw token IDs")
        if not isinstance(logprobs, Sequence) or isinstance(logprobs, (str, bytes)):
            raise ValueError("first sampler chunk lacks policy token logprobs")
        token_ids = [int(value) for value in token_ids]
        logprobs = [float(value) for value in logprobs]
        if len(token_ids) != len(logprobs) or not all(math.isfinite(value) for value in logprobs):
            raise ValueError("first sampler chunk token/logprob readback is inconsistent")
        finite_logprob_count += len(logprobs)
        grammar_escape = (
            len(token_ids) < 5
            or any(token not in coordinate_ids for token in token_ids[:4])
            or token_ids[4] != box_end
        )
        escape_count += int(grammar_escape)
        token_rows.append(
            {
                "draw_index": int(row["draw_index"]),
                "stream_seed": int(row["stream_seed"]),
                "stream_chunk_index": int(row["stream_chunk_index"]),
                "stream_offset": int(row["stream_offset"]),
                "token_ids": token_ids,
                "policy_logprobs": logprobs,
                "stop_reason": str(row.get("stop_reason")),
                "grammar_escape_by_frozen_serializer": grammar_escape,
            }
        )
    receipt = {
        "schema": "recurrence_conditional_mass.sampler_entry.v1",
        "unit_id": UNIT_ID,
        "status": "passed",
        "counts_toward_state_draws": DEFAULT_CHUNK,
        "state_id": str(state["state_id"]),
        "source_example_id": source_example_id,
        "model": model_key,
        "conditioning": {
            "prefix_token_ids_sha256": sha256_json(prefix_ids),
            "row_prefix_token_ids_sha256": sha256_json(row_prefix),
            "description_is_conditioned": True,
        },
        "serializer": {
            "bounded_horizon": int(serializer["bounded_horizon"]),
            "coordinate_token_ids_sha256": str(serializer["coordinate_token_ids_sha256"]),
            "box_end_token_id": box_end,
        },
        "draw_policy": {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
            "repetition_penalty": 1.0,
            "use_model_defaults": False,
            "full_vocabulary_softmax": True,
            "grammar_constraint": None,
            "retry_on_escape": False,
        },
        "rng": {
            "stream_seed": next(iter(seeds)),
            "chunk_index": chunk_index,
            "chunk_size": chunk_size,
            "offsets": actual_offsets,
            "seed_rule": SEED_NAMESPACE,
            "image_identity": "source_example_id",
        },
        "readback": {
            "draw_count": len(token_rows),
            "finite_policy_logprob_count": finite_logprob_count,
            "grammar_escape_count": escape_count,
            "raw_tokens_retained": True,
            "policy_logprobs_retained": True,
            "draws": token_rows,
        },
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


def _state_result(
    state: Mapping[str, Any],
    panel: Mapping[str, Any],
    sources: Mapping[str, Any],
    *,
    runtime_bundle: tuple[Any, Any, Any, Any, Any, dict[str, Any], dict[str, int]],
    output_root: Path,
    draws: bool,
    chunk_size: int,
    limit_draws: int | None,
) -> dict[str, Any]:
    state_id = str(state["state_id"])
    model_key = str(_first(state, "model_config_key", "model", "config_key"))
    config_payload = _config_for_state(state, panel, sources)
    model, qwen, _raw_examples, _requests, by_image, runtime_identity, runtime_counters = runtime_bundle
    state_started = time.perf_counter()
    forward_start = dict(runtime_counters)
    image_id = str(_first(state, "image_id", "source_image_id"))
    source_example_id = _state_source_example_id(state)
    if source_example_id not in by_image:
        raise ValueError(f"{state_id}: source example {source_example_id} absent from bound source")
    raw, request, actual_source_jsonl = by_image[source_example_id]
    prefix_ids = _prefix(state)
    description_value = _first(state, "description", "current_description", "category", "predicted_description")
    frozen_description_ids = _first(state, "description_token_ids")
    if description_value is not None:
        description_value = str(description_value)
    row_prefix, serializer = _row_condition(
        qwen.tokenizer,
        description_value,
        description_token_ids=frozen_description_ids,
    )
    description = str(serializer["description"])
    prompt_ids = tuple(int(v) for v in request.expected_executed_prompt_token_ids)
    device = model_device(model)
    native = prepare_native_inputs(
        qwen.processor,
        [_native_request(qwen, raw, request, request_id=f"{state_id}:qualification")],
        device=device,
        record_media_identity=True,
    )
    observed_prompt = tuple(int(v) for v in native.prompt_token_ids[0])
    if observed_prompt != prompt_ids:
        raise ValueError(f"{state_id}: native prompt differs from frozen request")
    boundary = _row_boundary_evidence(
        model,
        qwen,
        native,
        prompt_ids=prompt_ids,
        prefix_ids=prefix_ids,
        row_prefix=row_prefix,
        description_ids=serializer["description_token_ids"],
    )
    boundary.update(
        _description_entry_evidence(
            model,
            native,
            prompt_ids=prompt_ids,
            prefix_ids=prefix_ids,
            row_prefix=row_prefix,
            description_ids=serializer["description_token_ids"],
        )
    )
    source_score_parity = (
        _source_score_batch4_qualification(
            model,
            qwen,
            raw,
            request,
            native,
            prompt_ids=prompt_ids,
            prefix_ids=prefix_ids,
            state=state,
        )
        if not draws
        else {"status": "not_run_scientific"}
    )
    prefix_history = (*prompt_ids, *prefix_ids, *row_prefix)
    repeat_union = _repeat_union(state)
    saved_event = _saved_original_event(
        state,
        prefix_ids=prefix_ids,
        description_ids=serializer["description_token_ids"],
        serializer=serializer,
        repeat_union=repeat_union,
    )
    # The qualification uses the same current native entry but no scientific
    # draw is counted.  The caller can request a real draw batch afterwards.
    start = time.perf_counter()
    scientific_draws: list[dict[str, Any]] = []
    if draws:
        target = DRAWS_PER_STATE if limit_draws is None else min(DRAWS_PER_STATE, limit_draws)
        if target <= 0:
            raise ValueError("limit_draws must be positive")
        if target != DRAWS_PER_STATE:
            raise ValueError("scientific run requires exactly 256 draws per state")
        for chunk_index, begin in enumerate(range(0, target, chunk_size)):
            current = min(chunk_size, target - begin)
            # The split-aware source identity is the image binding used for
            # replay.  Keep the numeric image ID only as a displayed/source
            # stratum; it can recur across dataset splits.
            seed = derive_stream_seed(
                state_id=state_id,
                model=model_key,
                image_id=source_example_id,
                chunk_index=chunk_index,
            )
            chunk = _sample_chunk(
                model,
                qwen,
                raw,
                request,
                extension=(*prefix_ids, *row_prefix),
                horizon=int(serializer["bounded_horizon"]),
                chunk_size=current,
                seed=seed,
                state_id=state_id,
                chunk_index=chunk_index,
            )
            for offset, item in enumerate(chunk):
                item["draw_index"] = begin + offset
            if chunk_index == 0 and not (output_root / "sampler-entry.json").exists():
                _write_sampler_entry_receipt(
                    output_root,
                    state=state,
                    source_example_id=source_example_id,
                    model_key=model_key,
                    serializer=serializer,
                    prefix_ids=prefix_ids,
                    row_prefix=row_prefix,
                    chunk=chunk,
                    chunk_index=chunk_index,
                    chunk_size=current,
                )
            scientific_draws.extend(chunk)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - state_started
    forward_delta = {
        key: int(runtime_counters[key] - forward_start[key])
        for key in runtime_counters
    }
    return {
        "schema": DRAW_SCHEMA,
        "unit_id": UNIT_ID,
        "state_id": state_id,
        "state_type": _first(state, "state_type", "stratum", "kind"),
        "model": model_key,
        "image_id": _first(state, "image_id", "source_image_id"),
        "source_example_id": source_example_id,
        "source_policy": _first(state, "source_policy", "policy"),
        "description": str(description),
        "conditioning": {
            "prefix_token_ids": list(prefix_ids),
            "prefix_token_ids_sha256": sha256_json(prefix_ids),
            "prompt_token_ids": list(prompt_ids),
            "prompt_token_ids_sha256": sha256_json(prompt_ids),
            "row_prefix_token_ids": list(row_prefix),
            "row_prefix_token_ids_sha256": sha256_json(row_prefix),
            "description_is_conditioned": True,
            "description_probability_is_not_in_primary_q": True,
            "boundary_source": _first(state, "boundary_source"),
            "boundary_prefix_source_row_end": _first(state, "boundary_prefix_source_row_end"),
        },
        "serializer": serializer,
        "horizon": int(serializer["bounded_horizon"]),
        "box_end_token_id": int(serializer["box_end_token_id"]),
        "im_end_token_id": int(serializer["im_end_token_id"]),
        "coordinate_registry": {
            "coordinate_token_ids": serializer["coordinate_token_ids"],
            "coordinate_token_ids_sha256": serializer["coordinate_token_ids_sha256"],
        },
        "repeat_union_bins": repeat_union,
        "repeat_tolerance_bins": 8,
        "literal_repeat_union_bins": _repeat_union(state, literal=True) if _first(state, "literal_repeat_union_bins", "literal_invalid_repeat_union_bins") is not None else _repeat_union(state),
        "source_binding": {
            "source_example_id": source_example_id,
            "source_identity_binding": _first(state, "source_identity_binding"),
            "image_path": str(request.image_path),
            "image_sha256": str(request.image_sha256),
            "decoded_image_width": int(request.decoded_image_width),
            "decoded_image_height": int(request.decoded_image_height),
            "expected_image_grid_thw": list(request.expected_image_grid_thw or ()),
            "logical_transform_id": str(request.logical_transform_id),
            "source_jsonl": actual_source_jsonl,
        },
        "runtime_identity": runtime_identity,
        "row_boundary_evidence": boundary,
        "source_score_batch4_qualification": source_score_parity,
        "saved_original_event": saved_event,
        "saved_native_next_event": saved_event,
        "draw_policy": {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
            "use_model_defaults": False,
            "repetition_penalty": 1.0,
            "full_vocabulary_softmax": True,
            "grammar_constraint": None,
            "retry_on_escape": False,
            "rng_seed_rule": SEED_NAMESPACE,
            "rng_stream_unit": "state-model-image-chunk",
            "rng_image_identity": "source_example_id",
            "rng_chunk_size": chunk_size,
            "rng_draw_record": "stream_seed+stream_chunk_index+stream_offset",
        },
        "draw_count": len(scientific_draws),
        "draws": scientific_draws,
        "qualification": {
            "native_prompt_match": True,
            "native_media_sha256": None if native.media_sha256 is None else list(native.media_sha256),
            "prefix_history_token_count": len(prefix_history),
            "elapsed_seconds": elapsed,
            "model_forwards": forward_delta["model_forwards"],
            "vision_forwards": forward_delta["vision_forwards"],
            "native_generation_calls": (
                0
                if not draws
                else math.ceil(len(scientific_draws) / chunk_size)
            ),
        },
    }


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _load_completed_results(path: Path, expected_state_ids: set[str]) -> list[dict[str, Any]]:
    """Load only fully committed state records for a safe scientific resume."""

    if not path.exists() or path.stat().st_size == 0:
        return []
    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping) or value.get("schema") != DRAW_SCHEMA:
            raise ValueError(f"draws.jsonl line {line_number} is not a scientific state record")
        state_id = str(value.get("state_id"))
        if state_id in seen:
            raise ValueError(f"draws.jsonl repeats completed state {state_id}")
        if state_id not in expected_state_ids:
            raise ValueError(f"draws.jsonl contains state outside the frozen panel: {state_id}")
        if int(value.get("draw_count", -1)) != DRAWS_PER_STATE or len(value.get("draws", ())) != DRAWS_PER_STATE:
            raise ValueError(f"draws.jsonl state {state_id} is not a complete 256-draw record")
        results.append(dict(value))
        seen.add(state_id)
    return results


def _file_binding(path: Path) -> dict[str, str]:
    path = path.expanduser().resolve(strict=True)
    return {"path": str(path), "sha256": sha256_file(path)}


def _validate_scientific_inputs(
    panel_path: Path, sources_path: Path, panel: Mapping[str, Any]
) -> None:
    """Refuse to turn Lane A's rule-frozen prelaunch record into data."""

    if panel_path.name != "shared-panel.json" or sources_path.name != "shared-sources.json":
        raise ValueError(
            "scientific run requires Lane A final shared-panel.json and shared-sources.json"
        )
    if panel.get("schema") == "recurrence_census.prelaunch_mechanism_rule.v1":
        raise ValueError("prelaunch mechanism panel is not a scientific input")
    status = str(panel.get("status", "")).strip().lower()
    if status in {"rule_frozen_before_new_outputs", "prelaunch", "pending"}:
        raise ValueError(f"panel status {status!r} is not final for scientific sampling")


def _qualification_inputs(selection_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Project the accepted untied-417044 boundary into a one-state qualifier."""

    selection_path = selection_path.expanduser().resolve(strict=True)
    selection = _load_json(selection_path)
    boundaries = selection.get("boundaries")
    if not isinstance(boundaries, Sequence) or isinstance(boundaries, (str, bytes)):
        raise ValueError("qualification selection lacks boundaries")
    selected = [
        dict(boundary)
        for boundary in boundaries
        if isinstance(boundary, Mapping) and boundary.get("id") == "untied-417044-failure"
    ]
    if len(selected) != 1:
        raise ValueError("qualification requires exactly one untied-417044-failure boundary")
    boundary = selected[0]
    feedback_panel = selection_path.parent / "panel.json"
    feedback_payload = _load_json(feedback_panel)
    raw_path = boundary.get("raw_path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("qualification boundary has no bound raw_path")
    raw_path = Path(raw_path).expanduser().resolve(strict=True)
    mature_panel: Path | None = None
    for parent in raw_path.parents:
        candidate = parent / "panel.json"
        if candidate.exists():
            try:
                payload = _load_json(candidate)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if isinstance(payload.get("configs"), Mapping):
                mature_panel = candidate.resolve()
                break
    if mature_panel is None:
        raise ValueError("could not recover mature panel binding from qualification raw_path")
    sources = {
        "schema": "recurrence_conditional_mass.qualifier_sources.v1",
        "mature_panel": _file_binding(mature_panel),
        "feedback_panel": _file_binding(feedback_panel),
        "feedback_selection": _file_binding(selection_path),
    }
    panel = {
        "schema": "recurrence_conditional_mass.qualifier_panel.v1",
        "status": "mechanical_qualification_only",
        "unit_id": UNIT_ID,
        "existing_boundaries": [boundary],
        "source_selection": _file_binding(selection_path),
        "source_feedback_panel": _file_binding(feedback_panel),
        "source_mature_panel": _file_binding(mature_panel),
        "configs": feedback_payload.get("configs", {}),
    }
    refs = {
        "panel": str(selection_path),
        "sources": str(feedback_panel),
        "panel_sha256": sha256_file(selection_path),
        "sources_sha256": sha256_file(feedback_panel),
        "derived_panel_sha256": sha256_json(panel),
        "derived_sources_sha256": sha256_json(sources),
        "selection_boundary_id": boundary["id"],
        "mature_panel": str(mature_panel),
    }
    return panel, sources, refs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path)
    parser.add_argument("--sources", type=Path)
    parser.add_argument(
        "--qualify-selection",
        type=Path,
        help="qualify the accepted untied-417044 failure boundary from selection.json",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--limit-states", type=int)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK)
    parser.add_argument("--qualification-only", action="store_true")
    args = parser.parse_args(argv)
    if args.chunk_size != DEFAULT_CHUNK:
        parser.error(f"--chunk-size is frozen at {DEFAULT_CHUNK} for this unit")
    if args.qualify_selection is not None:
        if args.panel is not None or args.sources is not None:
            parser.error("--qualify-selection cannot be combined with --panel/--sources")
        panel, sources, refs = _qualification_inputs(args.qualify_selection)
        qualification_only = True
        panel_path = Path(refs["panel"])
        sources_path = Path(refs["sources"])
    else:
        if args.panel is None or args.sources is None:
            parser.error("--panel and --sources are required for a scientific run")
        panel_path = args.panel.expanduser().resolve(strict=True)
        sources_path = args.sources.expanduser().resolve(strict=True)
        panel = _load_json(panel_path)
        if not args.qualification_only:
            _validate_scientific_inputs(panel_path, sources_path, panel)
        sources = _load_json(sources_path)
        refs = {
            "panel": str(panel_path),
            "sources": str(sources_path),
            "panel_sha256": sha256_file(panel_path),
            "sources_sha256": sha256_file(sources_path),
        }
        qualification_only = bool(args.qualification_only)
    if args.qualify_selection is None and args.qualification_only:
        qualification_only = True
    states = _states(panel)
    if args.limit_states is not None:
        if args.limit_states <= 0:
            parser.error("--limit-states must be positive")
        states = states[: args.limit_states]
    if not states:
        raise SystemExit("shared panel has no states")
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    panel_hash = refs.get("panel_sha256") or sha256_file(panel_path)
    sources_hash = refs.get("sources_sha256") or sha256_file(sources_path)
    results: list[dict[str, Any]] = []
    draws_path = output_root / "draws.jsonl"
    sampler_entry_path = output_root / "sampler-entry.json"
    draws_handle = None
    if not qualification_only:
        expected_state_ids = {str(state["state_id"]) for state in states}
        if draws_path.exists():
            if draws_path.stat().st_size == 0:
                raise FileExistsError(
                    f"refusing to reuse an empty scientific draws file: {draws_path}"
                )
            if not sampler_entry_path.exists():
                raise FileExistsError(
                    "existing scientific draws lack the required sampler-entry receipt"
                )
            sampler_entry = _load_json(sampler_entry_path)
            if sampler_entry.get("status") != "passed" or int(sampler_entry.get("counts_toward_state_draws", -1)) != DEFAULT_CHUNK:
                raise ValueError("existing sampler-entry receipt is incomplete")
            results = _load_completed_results(draws_path, expected_state_ids)
            completed_ids = {str(result["state_id"]) for result in results}
            if str(sampler_entry.get("state_id")) not in completed_ids:
                raise ValueError("sampler-entry receipt does not belong to a completed state")
            draws_handle = draws_path.open("a")
        elif sampler_entry_path.exists():
            raise FileExistsError(
                f"refusing to start after an uncommitted sampler-entry chunk: {sampler_entry_path}"
            )
        else:
            draws_handle = draws_path.open("x")
    completed_ids = {str(result["state_id"]) for result in results}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for state in states:
        key = str(_first(state, "model_config_key", "model", "config_key"))
        grouped.setdefault(key, []).append(state)
    completed = len(results)
    for model_key, model_states in grouped.items():
        model_states = [state for state in model_states if str(state["state_id"]) not in completed_ids]
        if not model_states:
            continue
        config_payload = _config_for_state(model_states[0], panel, sources)
        runtime_bundle = _model_runtime(
            config_payload,
            sources,
            output_root=output_root / f"runtime-{model_key}",
            model_key=model_key,
            states=model_states,
        )
        for state in model_states:
            result = _state_result(
                state,
                panel,
                sources,
                runtime_bundle=runtime_bundle,
                output_root=output_root,
                draws=not qualification_only,
                chunk_size=args.chunk_size,
                limit_draws=None,
            )
            results.append(result)
            completed_ids.add(str(result["state_id"]))
            completed += 1
            if draws_handle is not None:
                draws_handle.write(_canonical(result) + "\n")
                draws_handle.flush()
            (output_root / "run-progress.json").write_text(
                json.dumps(
                    {
                        "schema": "recurrence_conditional_mass.run_progress.v1",
                        "unit_id": UNIT_ID,
                        "state_count_completed": completed,
                        "state_count_target": len(states),
                        "draw_count_completed": sum(int(item["draw_count"]) for item in results),
                        "state_ids": [item["state_id"] for item in results],
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            if qualification_only:
                break
            print(f"completed state {completed}/{len(states)}: {result['state_id']}", flush=True)
        del runtime_bundle
        if qualification_only:
            break
    manifest = {
        "schema": "recurrence_conditional_mass.run_manifest.v1",
        "unit_id": UNIT_ID,
        "panel": {"path": str(panel_path), "sha256": panel_hash},
        "sources": {"path": str(sources_path), "sha256": sources_hash},
        "state_count": len(results),
        "draw_count": sum(int(result["draw_count"]) for result in results),
        "draws_per_state": DRAWS_PER_STATE if not qualification_only else 0,
        "qualification_only": bool(qualification_only),
        "chunk_size": args.chunk_size,
        "state_ids": [result["state_id"] for result in results],
        "cost": {
            "model_forwards": sum(int(result["qualification"]["model_forwards"]) for result in results),
            "vision_forwards": sum(int(result["qualification"]["vision_forwards"]) for result in results),
            "native_generation_calls": sum(int(result["qualification"]["native_generation_calls"]) for result in results),
            "elapsed_seconds": sum(float(result["qualification"]["elapsed_seconds"]) for result in results),
        },
        "host_pid": os.getpid(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    (output_root / "run-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if qualification_only:
        (output_root / "qualification.json").write_text(json.dumps(results[0], indent=2, sort_keys=True) + "\n")
    else:
        assert draws_handle is not None
        draws_handle.close()
    print(json.dumps({"state_count": len(results), "draw_count": manifest["draw_count"], "qualification_only": manifest["qualification_only"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
