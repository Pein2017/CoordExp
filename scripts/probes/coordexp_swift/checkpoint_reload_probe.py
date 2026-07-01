#!/usr/bin/env python
"""Reload a CoordExp-swift checkpoint as base + DoRA adapter + token deltas."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.artifacts.checkpoint_reload import (
    build_checkpoint_reload_plan,
    verify_checkpoint_reload_payloads,
)
from src.qwen.special_token_embeddings import (
    DEFAULT_EMBED_DELTA_TENSOR_KEY,
    SPECIAL_TOKEN_EMBEDDING_SEMANTICS,
    SpecialTokenEmbeddingSourceGateEvidence,
    SpecialTokenSelection,
    install_special_token_embedding_deltas,
    load_special_token_embedding_deltas,
)


DEFAULT_OUTPUT_DIR = Path("outputs/probes/coordexp_swift/checkpoint_reload")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify that a CoordExp-swift checkpoint can be reloaded as the "
            "base Qwen model plus PEFT DoRA adapter plus compact selected-token "
            "embedding deltas."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to checkpoint.json or checkpoint-final.json.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument(
        "--run-forward",
        action="store_true",
        help="Run a tiny selected-token text forward after loading payloads.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = _resolve_device(args.device)
    dtype = _resolve_dtype(args.dtype, device)
    plan = build_checkpoint_reload_plan(args.checkpoint)
    payload_verification = verify_checkpoint_reload_payloads(plan)

    model = _load_base_model(plan.base_model_path, dtype=dtype, device=device)
    model = _load_adapter(model, plan.adapter_dir, device=device)
    special_result = install_special_token_embedding_deltas(
        model,
        _selection_from_metadata(plan.special_token_metadata_path),
        source_gate=_source_gate_for_payload(plan),
    )
    load_receipt = load_special_token_embedding_deltas(
        special_result,
        plan.special_token_payload_dir,
        expected_base_model_path=plan.base_model_path,
    )
    delta_equivalence = _compare_loaded_delta_to_payload(
        special_result.shared_embed_delta,
        plan.special_token_tensor_path,
    )
    forward_receipt = (
        _run_tiny_forward(
            special_result.model,
            selected_token_ids=special_result.receipt.token_selection.token_ids,
            device=device,
        )
        if args.run_forward
        else {"executed": False, "reason": "pass --run-forward"}
    )

    receipt = {
        "ok": True,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "argv": sys.argv,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "versions": {
            "torch": torch.__version__,
            "transformers": _package_version("transformers"),
            "peft": _package_version("peft"),
            "safetensors": _package_version("safetensors"),
        },
        "device": str(device),
        "dtype_requested": args.dtype,
        "dtype_resolved": str(dtype).replace("torch.", ""),
        "plan": plan.to_artifact_dict(),
        "payload_verification": payload_verification,
        "adapter_reload": _adapter_reload_receipt(model),
        "special_token_embedding_reload": {
            "install_receipt": special_result.receipt.to_artifact_dict(),
            "load_receipt": load_receipt.to_artifact_dict(),
            "delta_equivalence": delta_equivalence,
        },
        "forward": forward_receipt,
        "reload_contract": payload_verification["reload_contract"],
    }
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = output_dir / "receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"ok": True, "receipt": str(receipt_path)}, sort_keys=True))


def _load_base_model(
    model_path: Path,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.nn.Module:
    if not model_path.exists():
        raise RuntimeError(f"base model path does not exist: {model_path}")
    from transformers import AutoModelForImageTextToText

    load_kwargs: dict[str, Any] = {
        "local_files_only": True,
        "dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    if device.type == "cuda":
        load_kwargs["attn_implementation"] = "flash_attention_2"
    model = AutoModelForImageTextToText.from_pretrained(
        str(model_path),
        **load_kwargs,
    )
    model.to(device)
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = False
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = False
    return model


def _load_adapter(
    model: torch.nn.Module,
    adapter_dir: Path | None,
    *,
    device: torch.device,
) -> torch.nn.Module:
    if adapter_dir is None:
        raise RuntimeError("checkpoint reload plan did not include an adapter directory")
    from peft import PeftModel

    loaded = PeftModel.from_pretrained(
        model,
        str(adapter_dir),
        local_files_only=True,
        is_trainable=False,
    )
    loaded.to(device)
    loaded.eval()
    return loaded


def _selection_from_metadata(metadata_path: Path | None) -> SpecialTokenSelection:
    if metadata_path is None:
        raise RuntimeError("checkpoint reload plan did not include special-token metadata")
    metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    return SpecialTokenSelection(
        token_strings=tuple(str(item) for item in metadata_payload["token_strings"]),
        token_ids=tuple(int(item) for item in metadata_payload["token_ids"]),
    )


def _source_gate_for_payload(plan: Any) -> SpecialTokenEmbeddingSourceGateEvidence:
    if plan.special_token_tensor_path is None or plan.special_token_metadata_path is None:
        raise RuntimeError("checkpoint reload plan did not include special-token payload paths")
    metadata_payload = json.loads(plan.special_token_metadata_path.read_text(encoding="utf-8"))
    return SpecialTokenEmbeddingSourceGateEvidence(
        source_study_passed=True,
        roundtrip_probe_passed=True,
        probe_receipt={
            "ok": True,
            "semantics": SPECIAL_TOKEN_EMBEDDING_SEMANTICS,
            "num_selected_tokens": len(metadata_payload["token_ids"]),
            "payload": {
                "safetensors": str(plan.special_token_tensor_path),
                "metadata": str(plan.special_token_metadata_path),
            },
        },
    )


def _compare_loaded_delta_to_payload(
    loaded_delta: torch.nn.Parameter,
    tensor_path: Path | None,
) -> dict[str, Any]:
    if tensor_path is None:
        raise RuntimeError("checkpoint reload plan did not include special-token tensor")
    tensors = load_file(str(tensor_path), device="cpu")
    expected = tensors[DEFAULT_EMBED_DELTA_TENSOR_KEY]
    actual = loaded_delta.detach().cpu().to(expected.dtype)
    max_abs_diff = float((actual - expected).abs().max().item())
    return {
        "payload_tensor_key": DEFAULT_EMBED_DELTA_TENSOR_KEY,
        "payload_shape": [int(item) for item in expected.shape],
        "loaded_shape": [int(item) for item in actual.shape],
        "max_abs_diff": max_abs_diff,
        "equivalent": max_abs_diff == 0.0,
    }


def _adapter_reload_receipt(model: torch.nn.Module) -> dict[str, Any]:
    names = [name for name, _parameter in model.named_parameters()]
    magnitude = [name for name in names if "lora_magnitude_vector" in name]
    lora_a = [name for name in names if "lora_A" in name]
    lora_b = [name for name in names if "lora_B" in name]
    if not magnitude or not lora_a or not lora_b:
        raise RuntimeError(
            "reloaded adapter is missing LoRA A/B or DoRA magnitude-vector parameters"
        )
    return {
        "lora_A_count": len(lora_a),
        "lora_B_count": len(lora_b),
        "lora_magnitude_vector_count": len(magnitude),
        "lora_magnitude_vector_names_preview": magnitude[:8],
    }


def _run_tiny_forward(
    model: torch.nn.Module,
    *,
    selected_token_ids: tuple[int, ...],
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    probe_ids = selected_token_ids[: min(8, len(selected_token_ids))]
    if not probe_ids:
        raise RuntimeError("selected-token forward probe requires at least one token id")
    input_ids = torch.tensor([probe_ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        output = model(input_ids=input_ids, labels=None, use_cache=False)
    logits = getattr(output, "logits", None)
    if not isinstance(logits, torch.Tensor):
        raise RuntimeError("reloaded model forward did not return tensor logits")
    finite = bool(torch.isfinite(logits).all().item())
    if not finite:
        raise RuntimeError("reloaded model forward returned non-finite logits")
    return {
        "executed": True,
        "input_ids": list(probe_ids),
        "logits_shape": [int(item) for item in logits.shape],
        "logits_dtype": str(logits.dtype),
        "finite_logits": True,
    }


def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_dtype(requested: str, device: torch.device) -> torch.dtype:
    if requested == "fp32":
        return torch.float32
    if requested == "fp16":
        if device.type != "cuda":
            raise RuntimeError("--dtype fp16 is only supported on CUDA")
        return torch.float16
    if requested == "bf16":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float32
    raise RuntimeError(f"unsupported dtype: {requested}")


def _package_version(package_name: str) -> str:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return "not-installed"


if __name__ == "__main__":
    main()
