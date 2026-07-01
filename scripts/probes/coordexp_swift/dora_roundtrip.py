#!/usr/bin/env python
"""Minimal PEFT DoRA round-trip probe for the CoordExp Swift adapter gate."""

from __future__ import annotations

import argparse
import json
import math
import random
from importlib import metadata
from pathlib import Path
from typing import Any

import torch


DEFAULT_MODEL_PATH = Path(
    "/data/CoordExp/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent"
)
DEFAULT_OUTPUT_DIR = Path("outputs/probes/coordexp_swift/dora_roundtrip")
ADAPTER_NAME = "default"


def fail(message: str) -> None:
    raise RuntimeError(f"DoRA round-trip probe failed: {message}")


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not-installed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Probe PEFT DoRA mechanics for Qwen3-VL: language-only linear target "
            "discovery, finite forward/backward, adapter save/reload, and eval "
            "logit equivalence."
        )
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument(
        "--max-targets",
        type=int,
        default=4,
        help="Limit language linear targets for a tiny probe; use <=0 for all matches.",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            fail("--device cuda requested but CUDA is not available")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_dtype(requested: str, device: torch.device) -> torch.dtype:
    if requested == "fp32":
        return torch.float32
    if requested == "fp16":
        if device.type != "cuda":
            fail("--dtype fp16 is only supported for this probe on CUDA")
        return torch.float16
    if requested == "bf16":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float32
    fail(f"unsupported dtype: {requested}")


def require_local_model(model_path: Path) -> None:
    if not model_path.exists():
        fail(f"model path does not exist: {model_path}")
    for name in ("config.json",):
        if not (model_path / name).exists():
            fail(f"local model path is missing {name}: {model_path}")


def load_base_model(model_path: Path, dtype: torch.dtype, device: torch.device) -> torch.nn.Module:
    from transformers import AutoModelForImageTextToText

    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.config.use_cache = False
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = False
    return model


def is_language_linear_target(name: str, module: torch.nn.Module) -> bool:
    if not isinstance(module, torch.nn.Linear):
        return False
    if name == "lm_head" or name.endswith(".lm_head"):
        return False
    return name.startswith("language_model.") or ".language_model." in name


def discover_language_linear_targets(
    model: torch.nn.Module, max_targets: int
) -> tuple[list[str], list[str]]:
    matches = [
        name
        for name, module in model.named_modules()
        if is_language_linear_target(name, module)
    ]
    if not matches:
        fail("no language-tower torch.nn.Linear targets found")
    if any(name == "lm_head" or name.endswith(".lm_head") for name in matches):
        fail(f"lm_head leaked into target discovery: {matches}")
    selected = matches if max_targets <= 0 else matches[:max_targets]
    if not selected:
        fail("--max-targets selected zero language targets")
    return selected, matches


def build_dora_model(
    model: torch.nn.Module, target_modules: list[str]
) -> tuple[torch.nn.Module, Any]:
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=2,
        lora_alpha=4,
        lora_dropout=0.0,
        bias="none",
        target_modules=target_modules,
        use_dora=True,
    )
    peft_model = get_peft_model(model, config)
    return peft_model, config


def trainable_parameter_names(model: torch.nn.Module) -> list[str]:
    return [name for name, param in model.named_parameters() if param.requires_grad]


def assert_trainable_dora_surface(
    model: torch.nn.Module, target_modules: list[str]
) -> dict[str, Any]:
    trainable = trainable_parameter_names(model)
    lora_a = [name for name in trainable if "lora_A" in name]
    lora_b = [name for name in trainable if "lora_B" in name]
    magnitude = [name for name in trainable if "lora_magnitude_vector" in name]
    if not lora_a:
        fail("no trainable lora_A parameters found after PEFT injection")
    if not lora_b:
        fail("no trainable lora_B parameters found after PEFT injection")
    if not magnitude:
        fail("no trainable lora_magnitude_vector parameters found after PEFT DoRA injection")
    for label, names in (("lora_A", lora_a), ("lora_B", lora_b), ("magnitude", magnitude)):
        if len(names) != len(target_modules):
            fail(
                f"expected one trainable {label} parameter per selected target; "
                f"targets={len(target_modules)} {label}={len(names)}"
            )
    unexpected = [
        name
        for name in trainable
        if "lora_A" not in name
        and "lora_B" not in name
        and "lora_magnitude_vector" not in name
    ]
    if unexpected:
        fail(f"unexpected non-DoRA trainable parameters: {unexpected[:8]}")
    optimizer = torch.optim.SGD([param for param in model.parameters() if param.requires_grad], lr=0.0)
    opt_ids = [id(param) for group in optimizer.param_groups for param in group["params"]]
    duplicate_ids = len(opt_ids) - len(set(opt_ids))
    if duplicate_ids:
        fail(f"optimizer contains duplicate trainable parameters: {duplicate_ids}")
    missing_from_optimizer = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad and id(param) not in set(opt_ids)
    ]
    if missing_from_optimizer:
        fail(f"trainable parameters missing from optimizer group: {missing_from_optimizer[:8]}")
    return {
        "total": len(trainable),
        "names": trainable,
        "lora_A": lora_a,
        "lora_B": lora_b,
        "lora_magnitude_vector": magnitude,
        "optimizer_group_count": len(optimizer.param_groups),
        "optimizer_param_count": len(opt_ids),
        "optimizer_duplicate_param_count": duplicate_ids,
    }


def tiny_batch(model: torch.nn.Module, device: torch.device) -> dict[str, torch.Tensor]:
    vocab_size = int(getattr(model.config, "text_config", model.config).vocab_size)
    bos = int(getattr(model.config, "bos_token_id", 1) or 1)
    eos = int(getattr(model.config, "eos_token_id", 2) or 2)
    mid = min(max(42, 0), vocab_size - 1)
    input_ids = torch.tensor([[bos, mid, eos]], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, device=device)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def logits_from_output(output: Any) -> torch.Tensor:
    logits = getattr(output, "logits", None)
    if logits is None:
        fail("model forward output did not contain logits")
    return logits


def forward_backward_check(
    model: torch.nn.Module, batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, dict[str, Any]]:
    model.train()
    model.zero_grad(set_to_none=True)
    output = model(**batch)
    logits = logits_from_output(output)
    if not torch.isfinite(logits).all():
        fail("forward logits contain non-finite values")
    loss = logits.float().mean()
    if not torch.isfinite(loss):
        fail("tiny loss is non-finite")
    loss.backward()
    magnitude_grad_names: list[str] = []
    nonfinite_magnitude_grad_names: list[str] = []
    base_requires_grad = []
    for name, param in model.named_parameters():
        if "lora_magnitude_vector" in name:
            if param.grad is not None and torch.isfinite(param.grad).all():
                magnitude_grad_names.append(name)
            elif param.grad is not None:
                nonfinite_magnitude_grad_names.append(name)
        elif (
            "lora_A" not in name
            and "lora_B" not in name
            and "lora_magnitude_vector" not in name
            and param.requires_grad
        ):
            base_requires_grad.append(name)
    if nonfinite_magnitude_grad_names:
        fail(f"non-finite magnitude-vector gradients: {nonfinite_magnitude_grad_names[:8]}")
    if not magnitude_grad_names:
        fail("no lora_magnitude_vector parameter received a finite gradient")
    if base_requires_grad:
        fail(f"frozen base parameters unexpectedly require gradients: {base_requires_grad[:8]}")
    return logits.detach().cpu(), {
        "loss": float(loss.detach().cpu()),
        "finite_logits": True,
        "finite_magnitude_vector_gradient": True,
        "finite_magnitude_vector_gradient_names": magnitude_grad_names,
    }


def load_saved_tensor_keys(output_dir: Path) -> list[str]:
    safetensors_path = output_dir / "adapter_model.safetensors"
    bin_path = output_dir / "adapter_model.bin"
    if safetensors_path.exists():
        from safetensors import safe_open

        with safe_open(safetensors_path, framework="pt", device="cpu") as handle:
            return list(handle.keys())
    if bin_path.exists():
        payload = torch.load(bin_path, map_location="cpu")
        if not isinstance(payload, dict):
            fail(f"unexpected adapter_model.bin payload type: {type(payload)!r}")
        return list(payload.keys())
    fail(f"no PEFT adapter payload found under {output_dir}")


def assert_saved_adapter(output_dir: Path) -> dict[str, Any]:
    config_path = output_dir / "adapter_config.json"
    if not config_path.exists():
        fail(f"missing adapter_config.json after save: {config_path}")
    adapter_config = json.loads(config_path.read_text())
    if adapter_config.get("use_dora") is not True:
        fail("adapter_config.json did not preserve use_dora: true")
    keys = load_saved_tensor_keys(output_dir)
    lora_a = [key for key in keys if "lora_A" in key]
    lora_b = [key for key in keys if "lora_B" in key]
    magnitude = [key for key in keys if "lora_magnitude_vector" in key]
    if not lora_a or not lora_b or not magnitude:
        fail(
            "saved adapter payload is missing LoRA A/B or DoRA magnitude-vector tensors: "
            f"lora_A={len(lora_a)} lora_B={len(lora_b)} magnitude={len(magnitude)}"
        )
    return {
        "adapter_config_path": str(config_path),
        "adapter_payload_keys": keys,
        "saved_lora_A_count": len(lora_a),
        "saved_lora_B_count": len(lora_b),
        "saved_lora_magnitude_vector_count": len(magnitude),
        "adapter_config_use_dora": adapter_config.get("use_dora"),
    }


def reload_adapter(
    model_path: Path, adapter_path: Path, dtype: torch.dtype, device: torch.device
) -> torch.nn.Module:
    from peft import PeftModel

    base = load_base_model(model_path, dtype, device)
    model = PeftModel.from_pretrained(base, adapter_path, local_files_only=True, is_trainable=False)
    model.to(device)
    return model


def assert_reloaded_surface(model: torch.nn.Module) -> dict[str, Any]:
    names = [name for name, _ in model.named_parameters()]
    magnitude = [name for name in names if "lora_magnitude_vector" in name]
    if not magnitude:
        fail("reloaded adapter has no lora_magnitude_vector parameters")
    return {
        "reloaded_magnitude_vector_count": len(magnitude),
        "reloaded_magnitude_vector_names": magnitude,
    }


def eval_logits(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        logits = logits_from_output(model(**batch)).detach().float().cpu()
    if not torch.isfinite(logits).all():
        fail("eval logits contain non-finite values")
    return logits


def compare_eval_logits(
    original_logits: torch.Tensor,
    reloaded: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    dtype: torch.dtype,
) -> dict[str, Any]:
    reloaded.eval()
    with torch.no_grad():
        logits_b = logits_from_output(reloaded(**batch)).detach().float().cpu()
    if not torch.isfinite(logits_b).all():
        fail("reloaded eval logits contain non-finite values")
    max_abs_diff = float((original_logits - logits_b).abs().max().item())
    tolerance = 1.0e-5 if dtype == torch.float32 else 1.0e-2
    equivalent = math.isfinite(max_abs_diff) and max_abs_diff <= tolerance
    if not equivalent:
        fail(f"reload logits differ beyond tolerance: max_abs_diff={max_abs_diff} tol={tolerance}")
    return {
        "finite_original_eval_logits": True,
        "finite_reloaded_eval_logits": True,
        "max_abs_diff": max_abs_diff,
        "tolerance": tolerance,
        "equivalent": True,
    }


def write_receipt(output_dir: Path, receipt: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = output_dir / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt_path


def main() -> None:
    args = parse_args()
    random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    require_local_model(args.model_path)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_base_model(args.model_path, dtype, device)
    selected_targets, all_language_targets = discover_language_linear_targets(
        model, args.max_targets
    )
    dora_model, lora_config = build_dora_model(model, selected_targets)
    trainable = assert_trainable_dora_surface(dora_model, selected_targets)
    batch = tiny_batch(dora_model, device)
    _, gradient_result = forward_backward_check(dora_model, batch)

    original_eval_logits = eval_logits(dora_model, batch)
    dora_model.save_pretrained(output_dir, safe_serialization=True)
    saved = assert_saved_adapter(output_dir)

    del dora_model
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    reloaded = reload_adapter(args.model_path, output_dir, dtype, device)
    reloaded_surface = assert_reloaded_surface(reloaded)
    equivalence = compare_eval_logits(original_eval_logits, reloaded, batch, dtype)

    receipt = {
        "versions": {
            "torch": torch.__version__,
            "transformers": package_version("transformers"),
            "peft": package_version("peft"),
            "safetensors": package_version("safetensors"),
        },
        "model_path": str(args.model_path),
        "public_adapter_type": "dora",
        "peft": {
            "config_class": type(lora_config).__name__,
            "use_dora": bool(getattr(lora_config, "use_dora", False)),
            "r": int(getattr(lora_config, "r")),
            "lora_alpha": int(getattr(lora_config, "lora_alpha")),
            "lora_dropout": float(getattr(lora_config, "lora_dropout")),
            "bias": getattr(lora_config, "bias"),
        },
        "device": str(device),
        "dtype_requested": args.dtype,
        "dtype_resolved": str(dtype).replace("torch.", ""),
        "target_towers": ["language"],
        "target_policy": "all_linear",
        "lm_head_excluded": True,
        "max_targets": args.max_targets,
        "matched_language_target_count": len(all_language_targets),
        "matched_language_targets": all_language_targets,
        "selected_target_count": len(selected_targets),
        "selected_target_modules": selected_targets,
        "trainable_counts": {
            "total_trainable_parameter_tensors": trainable["total"],
            "lora_A": len(trainable["lora_A"]),
            "lora_B": len(trainable["lora_B"]),
            "lora_magnitude_vector": len(trainable["lora_magnitude_vector"]),
            "optimizer_group_count": trainable["optimizer_group_count"],
            "optimizer_param_count": trainable["optimizer_param_count"],
            "optimizer_duplicate_param_count": trainable["optimizer_duplicate_param_count"],
        },
        "trainable_names": trainable["names"],
        "magnitude_vectors": {
            "trainable_names": trainable["lora_magnitude_vector"],
            "reloaded_names": reloaded_surface["reloaded_magnitude_vector_names"],
        },
        "gradient_result": gradient_result,
        "save": {
            "output_dir": str(output_dir),
            **saved,
        },
        "reload": {
            "success": True,
            **reloaded_surface,
        },
        "equivalence": equivalence,
    }
    receipt_path = write_receipt(output_dir, receipt)
    print(json.dumps({"ok": True, "receipt": str(receipt_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
