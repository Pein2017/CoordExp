"""Source versus OFF-codebook expansion parity qualification.

The CLI intentionally performs real HF forwards only when invoked by the
parent execution driver.  Importing this module and its preparation helpers
is CPU-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch

from src.artifacts.utf8_json import binding
from src.adapters import attach_dora_adapter, setup_dora_adapter
from src.adapters.source_gates import (
    build_adapter_setup_plan,
    load_default_adapter_source_gate_evidence,
)
from src.config.loader import load_train_config
from src.config.models import TemplateConfig
from src.data.examples import raw_example_from_jsonl_row
from src.inference.bound_requests import build_bound_native_requests
from src.inference.inputs import plan_examples
from src.qwen.coordinate_codebook import install_coordinate_codebook
from src.qwen.generation import NativeGenerationPolicy, generate_continuations
from src.qwen.input_identity import input_identity, tensor_hash
from src.qwen.native import prepare_native_inputs, prepare_replay
from src.qwen.runtime_loading import QwenLoadOptions, load_qwen_components_from_options
from src.templates.renderer import render_example
from src.qwen import untied_embeddings


IMAGE_IDS = (2299, 13004, 417044)
PARITY_TOLERANCE = 2e-4
SHORT_GREEDY_TOKENS = 8


def _read_rows(dataset: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(dataset.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"dataset row {line_number} is not an object")
        row["_parity_line_number"] = line_number
        rows.append(row)
    return rows


def _config_dict(config: Any, dataset: Path) -> dict[str, Any]:
    payload = config.model_dump(mode="json")
    payload["data"] = {"input_jsonl": str(dataset.resolve(strict=True))}
    return payload


def _infer_config(config: Any, dataset: Path) -> Any:
    """Build the maintained inference planning view from the training config."""

    from src.config.inference import InferConfig

    payload = _config_dict(config, dataset)
    attn_implementation = payload["model"]["attn_implementation"]
    patch_embed_linearization = payload["model"]["runtime_patches"]["patch_embed_linearization"]
    payload["model"] = {
        "base_model": payload["model"]["base_model"],
        "dtype": payload["training"]["precision"],
        "processor": {"do_resize": False},
    }
    payload["backend"] = {
        "type": "hf",
        "hf": {
            "attn_implementation": attn_implementation,
            "patch_embed_linearization": patch_embed_linearization,
        },
    }
    payload["run"] = {"name": "coordinate-codebook-parity", "artifact_root": ".", "output_dir": None, "collision_policy": "fail"}
    payload["generation"] = {"batch_size": 1, "max_new_tokens": 3084, "temperature": 0.0, "top_p": 1.0, "n": 1, "repetition_penalty": 1.0}
    payload["scoring"] = {"enabled": True}
    # The maintained canonical inference contract requires these receipts even
    # for a parity slice; the driver may discard them after binding the result.
    payload["artifacts"] = {"write_token_trace": True, "write_parse_diagnostics": True, "include_raw_model_logprob": False}
    payload["debug"] = {"smoke": True, "dry_run": False}
    payload["adapter"] = {"type": "dora", "path": str(config.adapter.source_adapter_path), "name": "default"}
    payload["embedding_delta"] = {"path": str(config.adapter.repaired_embedding_payload_path)}
    allowed = {
        "schema_version", "run", "model", "data", "template", "backend",
        "generation", "scoring", "artifacts", "debug", "adapter", "embedding_delta",
    }
    return InferConfig.model_validate({key: value for key, value in payload.items() if key in allowed})


def _load_components(config: Any, device: torch.device) -> Any:
    return load_qwen_components_from_options(
        QwenLoadOptions(
            base_model=config.model.base_model,
            dtype=config.training.precision,
            attn_implementation=config.model.attn_implementation,
            patch_embed_linearization=config.model.runtime_patches.patch_embed_linearization,
            load_model=True,
        )
    )


def _load_source(config: Any, device: torch.device) -> tuple[Any, dict[str, Any]]:
    components = _load_components(config, device)
    from src.adapters.dora import load_live_dora_adapter
    model, adapter = load_live_dora_adapter(
        components.model,
        adapter_path=config.adapter.source_adapter_path,
        base_model_path=components.base_model_path,
        adapter_name="default",
    )
    components = replace(components, model=model)
    embedding = untied_embeddings.load_inference_embedding_delta(
        config=SimpleNamespace(
            embedding_delta=SimpleNamespace(path=config.adapter.repaired_embedding_payload_path)
        ),
        qwen=components,
    )
    components.model.to(device).eval()
    for parameter in components.model.parameters():
        parameter.requires_grad_(False)
    return components, {"adapter": adapter, "embedding": embedding, "codebook": "absent"}


def _validate_expanded_warm_start(adapter: Any, *, require_new_targets: bool = True) -> dict[str, Any]:
    """Bind source-copy and zero-B/function-identity evidence before parity."""

    receipt = getattr(adapter, "receipt", None)
    warm_start = getattr(receipt, "warm_start", None)
    if not isinstance(warm_start, Mapping):
        raise ValueError("expanded adapter is missing warm-start evidence")
    if warm_start.get("post_copy_equality") != "pass":
        raise ValueError("expanded adapter warm-start copy equality did not pass")
    ignored = tuple(str(item) for item in warm_start.get("ignored_source_tensors", ()))
    if ignored:
        raise ValueError(f"expanded adapter ignored source tensors: {ignored}")
    model = adapter.model
    named = dict(model.named_parameters())
    initialized = tuple(str(item) for item in warm_start.get("initialized_target_tensors", ()))
    missing = tuple(name for name in initialized if name not in named)
    if missing:
        raise ValueError(f"warm-start receipt names missing model tensors: {missing}")
    zero_b = tuple(name for name in initialized if "lora_B" in name)
    nonzero_b = tuple(
        name for name in zero_b
        if not torch.equal(named[name].detach(), torch.zeros_like(named[name].detach()))
    )
    if (require_new_targets and not zero_b) or nonzero_b:
        raise ValueError(f"new DoRA lora_B tensors are not zero initialized: {nonzero_b}")
    return {
        "post_copy_equality": "pass",
        "ignored_source_tensors": [],
        "initialized_target_tensors": list(initialized),
        "initialized_zero_lora_B_tensors": list(zero_b),
        "initialized_zero_lora_B_count": len(zero_b),
        "copied_tensors": list(warm_start.get("copied_tensors", ())),
        "source_adapter_tensor_sha256": warm_start.get("source_adapter_tensor_sha256"),
    }


def _load_expanded(config: Any, device: torch.device, *, require_new_targets: bool = True) -> tuple[Any, dict[str, Any]]:
    components = _load_components(config, device)
    evidence = load_default_adapter_source_gate_evidence(Path.cwd())
    plan = build_adapter_setup_plan(
        config.adapter,
        evidence,
        base_model_path=components.base_model_path,
    )
    adapter = setup_dora_adapter(components.model, plan)
    selection = untied_embeddings.build_default_special_token_selection(
        config.model.special_token_embeddings,
        components.token_identity,
    )
    embedding_result = untied_embeddings.install_special_token_embedding_deltas(
        adapter.model, selection, tie_word_embeddings=False
    )
    embedding = untied_embeddings.load_special_token_embedding_deltas(
        embedding_result,
        Path(config.adapter.repaired_embedding_payload_path),
        expected_base_model_path=components.base_model_path,
        expected_base_config_sha256=components.base_config_sha256,
        expected_tokenizer_sha256=components.tokenizer_sha256,
    )
    codebook_receipt: dict[str, Any] = {"status": "absent"}
    if config.model.coordinate_codebook is not None:
        codebook = install_coordinate_codebook(
            adapter.model,
            components.token_identity.coordinate_token_ids,
            initial_gain=config.model.coordinate_codebook.initial_gain,
            mode=config.model.coordinate_codebook.mode,
            projection_seed=config.model.coordinate_codebook.projection_seed,
        )
        codebook.enabled = False
        codebook_receipt = {
            "status": "installed_disabled",
            "enabled": False,
            "architecture": codebook.architecture_metadata(),
        }
    components = replace(components, model=adapter.model)
    warm_start = _validate_expanded_warm_start(adapter, require_new_targets=require_new_targets)
    components.model.to(device).eval()
    from src.adapters.dora import finalize_dora_initialization
    warm_start['device_initialization'] = finalize_dora_initialization(components.model)
    for parameter in components.model.parameters():
        parameter.requires_grad_(False)
    return components, {
        "adapter": adapter.receipt.to_artifact_dict(),
        "warm_start_surface": warm_start,
        "embedding": embedding.to_artifact_dict(),
        "codebook": codebook_receipt,
    }


def _effective_rows(model: Any) -> dict[str, Any]:
    inputs = model.get_input_embeddings()
    outputs = model.get_output_embeddings()
    ids = outputs.selected_token_ids.detach().cpu()
    if tuple(inputs.selection.token_ids) != tuple(outputs.selection.token_ids):
        raise ValueError("input/output selected-token mappings differ")
    input_rows = inputs(ids.to(next(model.parameters()).device)).detach()
    output_rows = (
        outputs.base.weight.index_select(0, ids.to(outputs.base.weight.device))
        + outputs.shared_embed_delta.detach()
    ).detach()
    return {
        "selected_ids_sha256": tensor_hash(ids),
        "input_rows_sha256": _tensor_stream_hash(input_rows),
        "output_rows_sha256": _tensor_stream_hash(output_rows),
        "input_delta_sha256": _tensor_stream_hash(inputs.shared_embed_delta),
        "output_delta_sha256": _tensor_stream_hash(outputs.shared_embed_delta),
        "selected_count": int(ids.numel()),
        "independent_deltas": inputs.shared_embed_delta is not outputs.shared_embed_delta,
    }


def _tensor_stream_hash(value: torch.Tensor, *, chunk_rows: int = 32) -> str:
    digest = hashlib.sha256()
    value = value.detach()
    for chunk in value.split(chunk_rows, dim=0):
        digest.update(chunk.cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _target_ids(qwen: Any, row: Mapping[str, Any], infer_config: Any, dataset: Path) -> list[int]:
    clean = {key: value for key, value in row.items() if not key.startswith("_")}
    raw = raw_example_from_jsonl_row(
        clean,
        jsonl_path=dataset,
        row_number=int(row["_parity_line_number"]),
        raw_line=json.dumps(clean, sort_keys=True),
    )
    template = TemplateConfig(
        object_field_order=infer_config.template.object_field_order,
        object_ordering=infer_config.template.object_ordering,
        assistant_format=infer_config.template.assistant_format,
        prompt=infer_config.template.prompt.model_dump(),
    )
    rendered = render_example(raw, template)
    return list(qwen.tokenizer.encode(rendered.supervised_response_text, add_special_tokens=False))


def _plan_case(qwen: Any, row: Mapping[str, Any], infer_config: Any, dataset: Path) -> dict[str, Any]:
    clean = {key: value for key, value in row.items() if not key.startswith("_")}
    raw = raw_example_from_jsonl_row(
        clean,
        jsonl_path=dataset,
        row_number=int(row["_parity_line_number"]),
        raw_line=json.dumps(clean, sort_keys=True),
    )
    planned = plan_examples([raw], config=infer_config, components=qwen, row_indices=[int(row["_parity_line_number"]) - 1])[0]
    return {
        "row_id": str(raw.example_id),
        "row_index": int(row["_parity_line_number"]) - 1,
        "input_record": clean,
        "image_path": planned.image.image_path,
        "image_width": planned.image.decoded_width,
        "image_height": planned.image.decoded_height,
        "image_plan": {**planned.image.to_artifact_dict(),
                       "backend_prompt_token_count": len(planned.prompt.expected_executed_prompt_token_ids),
                       "observed_image_grid_thw": planned.image.expected_image_grid_thw},
    }


def _compare_case(source: Any, expanded: Any, case: Mapping[str, Any], infer_config: Any, dataset: Path) -> dict[str, Any]:
    requests, _ = build_bound_native_requests(source, infer_config.model_dump(mode="json"), [case])
    batch = prepare_native_inputs(source.processor, requests, device=next(source.model.parameters()).device, record_media_identity=True)
    targets = _target_ids(source, {**case["input_record"], "_parity_line_number": int(case["row_index"]) + 1}, infer_config, dataset)
    source_replay = prepare_replay(source.model, batch.inputs, prompt_token_ids=batch.prompt_token_ids[0], continuation_token_ids=targets, compact_logits=True)
    expanded_replay = prepare_replay(expanded.model, batch.inputs, prompt_token_ids=batch.prompt_token_ids[0], continuation_token_ids=targets, compact_logits=True)
    with torch.inference_mode():
        source_logits = source.model(**source_replay.inputs).logits
        expanded_logits = expanded.model(**expanded_replay.inputs).logits
    source_aligned = source_replay.aligned_logits(source_logits)
    expanded_aligned = expanded_replay.aligned_logits(expanded_logits)
    if source_aligned.shape != expanded_aligned.shape:
        raise AssertionError(f"parity shape mismatch for {case['row_id']}: {tuple(source_aligned.shape)} vs {tuple(expanded_aligned.shape)}")
    difference = (source_aligned.float() - expanded_aligned.float()).abs()
    short_source = generate_continuations(source.model, batch, extensions=[()], budgets=[SHORT_GREEDY_TOKENS], eos_token_id=source.tokenizer.convert_tokens_to_ids("<|im_end|>"), pad_token_id=source.tokenizer.pad_token_id, policy=NativeGenerationPolicy(temperature=0.0, top_p=1.0, repetition_penalty=1.0, use_model_defaults=False))[0]
    short_expanded = generate_continuations(expanded.model, batch, extensions=[()], budgets=[SHORT_GREEDY_TOKENS], eos_token_id=expanded.tokenizer.convert_tokens_to_ids("<|im_end|>"), pad_token_id=expanded.tokenizer.pad_token_id, policy=NativeGenerationPolicy(temperature=0.0, top_p=1.0, repetition_penalty=1.0, use_model_defaults=False))[0]
    return {
        "row_id": case["row_id"],
        "image_id": int(case["input_record"]["image_id"]),
        "prompt_token_ids": list(batch.prompt_token_ids[0]),
        "target_token_ids_sha256": hashlib.sha256(torch.tensor(targets, dtype=torch.long).numpy().tobytes()).hexdigest(),
        "target_token_count": len(targets),
        "image_grid_thw": list(batch.image_grids[0]),
        "merged_visual_tokens": int(batch.image_grids[0][1] * batch.image_grids[0][2] // 4),
        "input_identity": input_identity(batch),
        "source_logits_shape": list(source_aligned.shape),
        "expanded_logits_shape": list(expanded_aligned.shape),
        "source_logits_sha256": _tensor_stream_hash(source_aligned),
        "expanded_logits_sha256": _tensor_stream_hash(expanded_aligned),
        "max_abs_logit_difference_fp32": float(difference.max().item()),
        "short_greedy_source": list(short_source.token_ids),
        "short_greedy_expanded": list(short_expanded.token_ids),
        "short_greedy_equal": list(short_source.token_ids) == list(short_expanded.token_ids),
    }


def run_parity(config_path: Path, dataset: Path, output: Path, *, device: str = "cuda:0") -> dict[str, Any]:
    resolved = load_train_config(config_path)
    config = resolved.config
    rows = _read_rows(dataset)
    selected = [row for row in rows if int(row["image_id"]) in IMAGE_IDS]
    if {int(row["image_id"]) for row in selected} != set(IMAGE_IDS):
        raise ValueError(f"qualification dataset must contain exactly the requested IDs: {IMAGE_IDS}")
    infer_config = _infer_config(config, dataset)
    torch_device = torch.device(device)
    started = time.time()
    result = {
        "schema": "coordinate_codebook_alignment.source_parity.v1",
        "status": "running",
        "config": binding(config_path),
        "dataset": binding(dataset),
        "source_checkpoint": str(config.adapter.source_adapter_path),
        "device": device,
        "runtime": {"dtype": config.training.precision, "attn_implementation": config.model.attn_implementation, "codebook_enabled": False, "tolerance_fp32": PARITY_TOLERANCE},
        "cases": [], "model_forwards": 0, "vision_forwards": 0,
    }
    handles = []
    def count(field):
        def hook(*_):
            result[field] += 1
        return hook
    try:
        from src.qwen.coordinate_codebook import _visual_module
        source, source_receipt = _load_source(config, torch_device)
        expanded, expanded_receipt = _load_expanded(config, torch_device)
        from src.adapters.dora import _target_trainable_key_to_source_key
        def mature(qwen):
            return {_target_trainable_key_to_source_key(name, adapter_name='default'): parameter
                    for name, parameter in qwen.model.named_parameters() if 'lora_' in name}
        source_parameters, expanded_parameters = mature(source), mature(expanded)
        preservation = []
        for name, parameter in source_parameters.items():
            other = expanded_parameters[name]
            equal = parameter.dtype == other.dtype and torch.equal(parameter, other)
            preservation.append({'name': name, 'dtype': str(parameter.dtype), 'equal': equal})
            if not equal:
                raise AssertionError(f'mature runtime tensor mismatch: {name}')
        result['mature_runtime_preservation'] = preservation
        result['reference_contract'] = 'live_promoted_v1; original target set only; no codebook; no merge'
        result['prior_diagnosis_correction'] = 'parity-v4 used PeftAdapterMixin.load_adapter, not merge_dora_adapter_for_execution; prior merged-source explanation is withdrawn'
        for qwen in (source, expanded):
            from peft import PeftModel
            counter_model = qwen.model.get_base_model() if isinstance(qwen.model, PeftModel) else qwen.model
            handles.append(counter_model.register_forward_pre_hook(count('model_forwards')))
            handles.append(_visual_module(qwen.model).register_forward_pre_hook(count('vision_forwards')))
        source_rows = _effective_rows(source.model)
        expanded_rows = _effective_rows(expanded.model)
        result.update(source=source_receipt, expanded=expanded_receipt,
                      source_effective_rows=source_rows, expanded_effective_rows=expanded_rows)
        if source_rows != expanded_rows or not source_rows['independent_deltas']:
            raise AssertionError("expanded input/output effective rows differ from source or are tied")
        cases = [_plan_case(source, row, infer_config, dataset) for row in selected]
        for case in cases:
            result['cases'].append(_compare_case(source, expanded, case, infer_config, dataset))
        if any(item['max_abs_logit_difference_fp32'] > PARITY_TOLERANCE or not item['short_greedy_equal']
               for item in result['cases']):
            raise AssertionError('OFF-codebook source parity exceeded tolerance')
        result['status'] = 'complete'
    except BaseException as exc:
        result.update(status='failed', error=repr(exc))
        raise
    finally:
        for handle in handles:
            handle.remove()
        result.update(start_time=started, terminal_time=time.time(), wall_seconds=time.time() - started,
                      max_cuda_memory_allocated_bytes=torch.cuda.max_memory_allocated())
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, indent=2, sort_keys=True)
            stream.write('\n')
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = run_parity(args.config, args.dataset, args.output, device=args.device)
    print(json.dumps({"status": result["status"], "output": str(args.output), "cases": len(result["cases"])}, sort_keys=True))


if __name__ == "__main__":
    main()
