"""Diagnostic bridge between the old Mixin loader and the live source loader.

This is a bounded loader comparison on the three qualification images.  It is
diagnostic evidence only: it deliberately has no parity gate and does not load
an expanded model or the coordinate codebook.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
from safetensors import safe_open

from probes.training_set_completion.artifacts import binding
from probes.training_set_completion.coordinate_codebook_alignment.parity import (
    IMAGE_IDS,
    _compare_case,
    _infer_config,
    _load_components,
    _load_source,
    _plan_case,
    _read_rows,
)
from src.adapters.dora import attach_dora_adapter, normalize_dora_state_key
from src.config.loader import load_train_config
from src.qwen import untied_embeddings

def _load_old_mixin(config: Any, device: torch.device) -> tuple[Any, dict[str, Any]]:
    """Load the historical source route used by the old inference caller."""

    components = _load_components(config, device)
    adapter = attach_dora_adapter(
        components.model,
        adapter_path=config.adapter.source_adapter_path,
        base_model_path=components.base_model_path,
        adapter_name="default",
    )
    embedding = untied_embeddings.load_inference_embedding_delta(
        config=SimpleNamespace(
            embedding_delta=SimpleNamespace(
                path=config.adapter.repaired_embedding_payload_path
            )
        ),
        qwen=components,
    )
    components.model.to(device).eval()
    for parameter in components.model.parameters():
        parameter.requires_grad_(False)
    return components, {"adapter": adapter, "embedding": embedding, "codebook": "absent"}


def _sha256_tensor(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def _source_payload(path: Path) -> dict[str, tuple[str, torch.Tensor]]:
    payload: dict[str, tuple[str, torch.Tensor]] = {}
    with safe_open(str(path / "adapter_model.safetensors"), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            normalized = normalize_dora_state_key(key, adapter_name="default")
            if normalized.endswith('.lora_magnitude_vector'):
                normalized += '.weight'
            if normalized in payload:
                raise ValueError(f"duplicate normalized source adapter key: {normalized}")
            payload[normalized] = (key, handle.get_tensor(key).cpu())
    if not payload:
        raise ValueError("source adapter payload is empty")
    return payload


def _mature_parameter_snapshot(model: Any, source_adapter_path: Path) -> dict[str, Any]:
    source = _source_payload(source_adapter_path)
    runtime: dict[str, tuple[str, torch.Tensor]] = {}
    for name, parameter in model.named_parameters():
        normalized = normalize_dora_state_key(name, adapter_name="default")
        if normalized in runtime:
            raise ValueError(f"duplicate normalized runtime adapter key: {normalized}")
        runtime[normalized] = (name, parameter.detach())
    records = []
    missing = []
    for normalized in sorted(source):
        source_name, source_tensor = source[normalized]
        runtime_entry = runtime.get(normalized)
        if runtime_entry is None:
            missing.append(normalized)
            continue
        runtime_name, runtime_tensor = runtime_entry
        if tuple(source_tensor.shape) != tuple(runtime_tensor.shape):
            raise ValueError(f"mature tensor shape mismatch for {normalized}")
        source_as_runtime = source_tensor.to(device=runtime_tensor.device, dtype=runtime_tensor.dtype)
        records.append({
            "normalized_key": normalized,
            "source_key": source_name,
            "runtime_name": runtime_name,
            "source_dtype": str(source_tensor.dtype),
            "runtime_dtype": str(runtime_tensor.dtype),
            "source_sha256": _sha256_tensor(source_tensor),
            "runtime_sha256": _sha256_tensor(runtime_tensor),
            "source_max_abs": float(source_tensor.abs().max().item()),
            "runtime_max_abs": float(runtime_tensor.abs().max().item()),
            "cast_value_max_abs_difference": float(
                (runtime_tensor.float() - source_as_runtime.float()).abs().max().item()
            ),
        })
    if missing:
        raise ValueError(f"runtime is missing mature source tensors: {missing[:8]}")
    return {
        "source_payload": binding(source_adapter_path / "adapter_model.safetensors"),
        "normalized_parameter_count": len(records),
        "parameters": records,
    }


def _selected_row_snapshot(model: Any) -> dict[str, Any]:
    inputs = model.get_input_embeddings()
    outputs = model.get_output_embeddings()
    ids = outputs.selected_token_ids.detach().cpu()
    if tuple(inputs.selection.token_ids) != tuple(outputs.selection.token_ids):
        raise ValueError("input/output selected-token mappings differ")
    device = next(model.parameters()).device
    input_rows = inputs(ids.to(device)).detach()
    output_rows = (
        outputs.base.weight.index_select(0, ids.to(outputs.base.weight.device))
        + outputs.shared_embed_delta.detach()
    ).detach()
    return {
        "selected_count": int(ids.numel()),
        "selected_ids_sha256": _sha256_tensor(ids),
        "input_rows_sha256": _sha256_tensor(input_rows),
        "output_rows_sha256": _sha256_tensor(output_rows),
        "input_rows_max_abs": float(input_rows.float().abs().max().item()),
        "output_rows_max_abs": float(output_rows.float().abs().max().item()),
        "input_delta_sha256": _sha256_tensor(inputs.shared_embed_delta),
        "output_delta_sha256": _sha256_tensor(outputs.shared_embed_delta),
    }


def _install_counters(model: Any) -> tuple[dict[str, int], list[Any]]:
    from peft import PeftModel
    if isinstance(model, PeftModel):
        model = model.get_base_model()
    from src.qwen.coordinate_codebook import _visual_module

    counters = {"model_forwards": 0, "vision_forwards": 0}
    handles = [
        model.register_forward_hook(lambda *_: counters.__setitem__("model_forwards", counters["model_forwards"] + 1)),
        _visual_module(model).register_forward_hook(
            lambda *_: counters.__setitem__("vision_forwards", counters["vision_forwards"] + 1)
        ),
    ]
    return counters, handles


def run_bridge(config_path: Path, dataset: Path, output: Path, *, device: str = "cuda:0") -> dict[str, Any]:
    resolved = load_train_config(config_path)
    config = resolved.config
    rows = _read_rows(dataset)
    selected = [row for row in rows if int(row["image_id"]) in IMAGE_IDS]
    if {int(row["image_id"]) for row in selected} != set(IMAGE_IDS):
        raise ValueError(f"qualification dataset must contain exactly the requested IDs: {IMAGE_IDS}")
    infer_config = _infer_config(config, dataset)
    torch_device = torch.device(device)
    result: dict[str, Any] = {
        "schema": "coordinate_codebook_alignment.runtime_bridge.v1",
        "status": "running",
        "diagnostic_only": True,
        "comparison": "old_mixin_source_vs_new_live_source",
        "config": binding(config_path),
        "dataset": binding(dataset),
        "source_checkpoint": str(config.adapter.source_adapter_path),
        "device": device,
        "runtime": {
            "dtype": config.training.precision,
            "attn_implementation": config.model.attn_implementation,
            "codebook": "absent",
            "parity_gate": "none",
        },
        "cases": [],
        "model_forwards": 0,
        "vision_forwards": 0,
    }
    handles: list[Any] = []
    counters: list[dict[str, int]] = []
    started = time.time()
    try:
        old, old_receipt = _load_old_mixin(config, torch_device)
        new, new_receipt = _load_source(config, torch_device)
        old_counter, old_handles = _install_counters(old.model)
        new_counter, new_handles = _install_counters(new.model)
        counters.extend((old_counter, new_counter))
        handles.extend(old_handles)
        handles.extend(new_handles)
        result["old_mixin"] = {
            "receipt": old_receipt,
            "mature_parameters": _mature_parameter_snapshot(old.model, Path(config.adapter.source_adapter_path)),
            "selected_rows": _selected_row_snapshot(old.model),
        }
        result["new_live"] = {
            "receipt": new_receipt,
            "mature_parameters": _mature_parameter_snapshot(new.model, Path(config.adapter.source_adapter_path)),
            "selected_rows": _selected_row_snapshot(new.model),
        }
        cases = [_plan_case(old, row, infer_config, dataset) for row in selected]
        for case in cases:
            report = _compare_case(old, new, case, infer_config, dataset)
            result["cases"].append({
                **report,
                "old_mixin_logits_sha256": report.pop("source_logits_sha256"),
                "new_live_logits_sha256": report.pop("expanded_logits_sha256"),
                "old_mixin_logits_shape": report.pop("source_logits_shape"),
                "new_live_logits_shape": report.pop("expanded_logits_shape"),
                "old_mixin_short_greedy": report.pop("short_greedy_source"),
                "new_live_short_greedy": report.pop("short_greedy_expanded"),
            })
        result["status"] = "complete"
    except BaseException as exc:
        result["status"] = "failed"
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        for handle in handles:
            handle.remove()
        result["model_forwards"] = sum(item["model_forwards"] for item in counters)
        result["vision_forwards"] = sum(item["vision_forwards"] for item in counters)
        result["counter_by_loader"] = counters
        result["wall_seconds"] = time.time() - started
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = run_bridge(args.config, args.dataset, args.output, device=args.device)
    print(json.dumps({"status": result["status"], "output": str(args.output), "cases": len(result["cases"])}, sort_keys=True))


if __name__ == "__main__":
    main()
