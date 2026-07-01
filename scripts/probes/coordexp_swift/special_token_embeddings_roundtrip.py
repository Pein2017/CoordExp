#!/usr/bin/env python3
"""Round-trip probe for CoordExp Swift selected special-token embedding deltas.

This is intentionally self-contained probe code, not production runtime code.
It validates the local Qwen3-VL CoordExp tokenizer/model and proves the V1
additive selected-token delta semantics described in the Wave 1A source study.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import json
import sys
from datetime import datetime, timezone
from importlib import metadata as package_metadata
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer

DEFAULT_MODEL_PATH = Path(
    "/data/CoordExp/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent"
)
DEFAULT_OUTPUT_DIR = Path(
    "outputs/probes/coordexp_swift/special_token_embeddings_roundtrip"
)
WRAPPER_TOKENS = {
    "<|object_ref_start|>": 151646,
    "<|object_ref_end|>": 151647,
    "<|box_start|>": 151648,
    "<|box_end|>": 151649,
}
COORD_START = 151670
COORD_COUNT = 1000
TENSOR_KEY = "shared_embed_delta"
SEMANTICS = "additive_delta"


class ProbeError(RuntimeError):
    """Fail-fast probe error with a user-facing message."""


def fail(message: str) -> None:
    raise ProbeError(message)


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe additive selected special-token embedding round-trip behavior."
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto"
    )
    return parser.parse_args()


def package_version(name: str) -> str:
    try:
        return package_metadata.version(name)
    except package_metadata.PackageNotFoundError:
        return "not-installed"


def torch_dtype(name: str) -> torch.dtype | str:
    if name == "auto":
        return "auto"
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_token_spec() -> list[tuple[str, int]]:
    tokens = list(WRAPPER_TOKENS.items())
    tokens.extend((f"<|coord_{i}|>", COORD_START + i) for i in range(COORD_COUNT))
    return tokens


def validate_tokenizer(model_path: Path) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), trust_remote_code=True, local_files_only=True
    )
    token_spec = canonical_token_spec()
    token_strings = [tok for tok, _ in token_spec]
    token_ids: list[int] = []

    for token, expected_id in token_spec:
        actual_id = tokenizer.convert_tokens_to_ids(token)
        assert_true(
            actual_id == expected_id,
            f"token id drift for {token}: expected {expected_id}, got {actual_id}",
        )
        encoded = tokenizer.encode(token, add_special_tokens=False)
        assert_true(
            encoded == [expected_id],
            f"token {token} is not single-token: expected [{expected_id}], got {encoded}",
        )
        token_ids.append(actual_id)

    assert_true(len(token_ids) == 1004, f"expected 1004 selected ids, got {len(token_ids)}")
    assert_true(
        len(set(token_ids)) == len(token_ids),
        "selected special-token ids contain duplicates",
    )
    coord_ids = token_ids[len(WRAPPER_TOKENS) :]
    expected_coords = list(range(COORD_START, COORD_START + COORD_COUNT))
    assert_true(coord_ids == expected_coords, "coordinate token ids are not contiguous")

    return {
        "tokenizer": tokenizer,
        "token_strings": token_strings,
        "token_ids": token_ids,
        "wrapper_token_ids": dict(WRAPPER_TOKENS),
        "coord_start": COORD_START,
        "coord_end": COORD_START + COORD_COUNT - 1,
        "tokenizer_sha256": sha256_file(model_path / "tokenizer.json"),
        "assertions": {
            "selected_count_is_1004": True,
            "all_selected_tokens_single_token": True,
            "selected_token_ids_unique": True,
            "coordinate_token_ids_contiguous": True,
        },
    }


def load_model(model_path: Path, device: str, dtype_name: str) -> nn.Module:
    if not model_path.exists():
        fail(f"model path does not exist: {model_path}")
    dtype = torch_dtype(dtype_name)
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "local_files_only": True,
    }
    if dtype == "auto":
        kwargs["torch_dtype"] = "auto"
    else:
        kwargs["torch_dtype"] = dtype
    try:
        model = AutoModelForImageTextToText.from_pretrained(str(model_path), **kwargs)
    except Exception as exc:  # pragma: no cover - probe fail-fast path
        fail(f"failed to load real model from {model_path}: {exc}")
    model.to(device)
    model.eval()
    return model


def validate_tied_model(model: nn.Module) -> tuple[nn.Embedding, nn.Linear]:
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", None)
    tie_config = bool(getattr(config, "tie_word_embeddings", False))
    text_tie_config = bool(getattr(text_config, "tie_word_embeddings", tie_config))
    assert_true(tie_config, "model config tie_word_embeddings is not true")
    assert_true(text_tie_config, "text_config tie_word_embeddings is not true")

    input_embedding = model.get_input_embeddings()
    lm_head = model.get_output_embeddings()
    assert_true(input_embedding is not None, "model.get_input_embeddings() returned None")
    assert_true(lm_head is not None, "model.get_output_embeddings() returned None")
    assert_true(
        input_embedding.weight is lm_head.weight,
        "runtime input embedding and lm_head weights are not the same Parameter object",
    )
    return input_embedding, lm_head


class SelectedDeltaInputEmbedding(nn.Module):
    def __init__(self, base: nn.Embedding, selected_token_ids: list[int], delta: nn.Parameter):
        super().__init__()
        self.base = base
        self.selected_token_ids = torch.tensor(selected_token_ids, dtype=torch.long)
        self.register_buffer("token_to_row", self._build_token_to_row(selected_token_ids), persistent=False)
        self.delta = delta

    @staticmethod
    def _build_token_to_row(selected_token_ids: list[int]) -> torch.Tensor:
        mapping = torch.full((max(selected_token_ids) + 1,), -1, dtype=torch.long)
        mapping[torch.tensor(selected_token_ids, dtype=torch.long)] = torch.arange(
            len(selected_token_ids), dtype=torch.long
        )
        return mapping

    @property
    def weight(self) -> nn.Parameter:
        return self.base.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        base_out = self.base(input_ids)
        safe_ids = input_ids.clamp_min(0).clamp_max(self.token_to_row.numel() - 1)
        rows = self.token_to_row.to(input_ids.device)[safe_ids]
        mask = (input_ids < self.token_to_row.numel()) & (rows >= 0)
        safe_rows = rows.clamp_min(0)
        delta_out = self.delta.to(base_out.dtype)[safe_rows]
        return base_out + delta_out * mask.unsqueeze(-1).to(base_out.dtype)


class SelectedDeltaOutputHead(nn.Module):
    def __init__(self, base: nn.Linear, selected_token_ids: list[int], delta: nn.Parameter):
        super().__init__()
        self.base = base
        self.register_buffer(
            "selected_token_ids", torch.tensor(selected_token_ids, dtype=torch.long), persistent=False
        )
        self.delta = delta

    @property
    def weight(self) -> nn.Parameter:
        return self.base.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base.bias

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.base(hidden_states)
        correction = hidden_states.to(self.delta.dtype) @ self.delta.t()
        correction = correction.to(logits.dtype)
        selected_ids = self.selected_token_ids.to(logits.device)
        scatter_index = selected_ids.view(*([1] * (correction.ndim - 1)), -1).expand_as(
            correction
        )
        logits = logits.clone()
        logits.scatter_add_(-1, scatter_index, correction)
        return logits


def install_probe_wrappers(
    model: nn.Module,
    base_embedding: nn.Embedding,
    base_lm_head: nn.Linear,
    selected_token_ids: list[int],
) -> tuple[SelectedDeltaInputEmbedding, SelectedDeltaOutputHead, nn.Parameter]:
    for param in model.parameters():
        param.requires_grad_(False)

    delta = nn.Parameter(
        torch.zeros(
            len(selected_token_ids),
            base_embedding.embedding_dim,
            device=base_embedding.weight.device,
            dtype=base_embedding.weight.dtype,
        )
    )
    input_wrapper = SelectedDeltaInputEmbedding(base_embedding, selected_token_ids, delta)
    output_wrapper = SelectedDeltaOutputHead(base_lm_head, selected_token_ids, delta)
    model.set_input_embeddings(input_wrapper)
    model.set_output_embeddings(output_wrapper)

    trainable = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    trainable_names = [name for name, _ in trainable]
    assert_true(
        len(trainable) == 1 and trainable[0][1] is delta and trainable[0][0].endswith(".delta"),
        f"unexpected trainable parameters after wrapper install: {trainable_names}",
    )
    optimizer_group = {"name": "token_embeddings", "params": [delta]}
    assert_true(
        sum(param is delta for param in optimizer_group["params"]) == 1,
        "optimizer group token_embeddings does not own selected delta exactly once",
    )
    assert_true(not base_embedding.weight.requires_grad, "base embedding still requires grad")
    assert_true(not base_lm_head.weight.requires_grad, "base lm_head still requires grad")
    return input_wrapper, output_wrapper, delta


def assert_close(actual: torch.Tensor, expected: torch.Tensor, message: str, atol: float = 1e-5) -> None:
    if not torch.allclose(actual, expected, atol=atol, rtol=0):
        max_diff = (actual - expected).abs().max().item()
        fail(f"{message}; max_abs_diff={max_diff}")


def assert_loaded_delta_behavior(
    input_wrapper: SelectedDeltaInputEmbedding,
    output_wrapper: SelectedDeltaOutputHead,
    delta: nn.Parameter,
    selected_token_ids: list[int],
    *,
    label: str,
) -> dict[str, Any]:
    device = delta.device
    dtype = delta.dtype
    selected_atol = 1e-5 if dtype == torch.float32 else 5e-3
    selected_row = 4
    selected_id = selected_token_ids[selected_row]
    non_selected_id = 0
    selected_tensor = torch.tensor(selected_token_ids, device=device, dtype=torch.long)

    input_ids = torch.tensor([[selected_id, non_selected_id]], device=device)
    base_input = input_wrapper.base(input_ids).detach()
    wrapped_input = input_wrapper(input_ids).detach()
    input_diff = wrapped_input - base_input
    assert_close(
        input_diff[0, 0],
        delta.detach()[selected_row].to(input_diff.dtype),
        f"{label}: loaded selected input delta mismatch",
        atol=selected_atol,
    )
    assert_close(
        input_diff[0, 1],
        torch.zeros_like(input_diff[0, 1]),
        f"{label}: loaded non-selected input changed",
    )

    hidden = torch.zeros((2, delta.shape[1]), device=device, dtype=dtype)
    hidden[0, : min(16, delta.shape[1])] = torch.arange(
        1, min(16, delta.shape[1]) + 1, device=device, dtype=dtype
    ) / 100
    hidden[1, : min(16, delta.shape[1])] = 0.5
    base_logits = output_wrapper.base(hidden).detach()
    wrapped_logits = output_wrapper(hidden).detach()
    logits_diff = wrapped_logits - base_logits
    expected_correction = (hidden.to(dtype) @ delta.detach().t()).to(logits_diff.dtype)
    assert_close(
        logits_diff.index_select(-1, selected_tensor),
        expected_correction,
        f"{label}: loaded selected output correction mismatch",
        atol=selected_atol,
    )
    non_selected_columns = torch.ones(logits_diff.shape[-1], dtype=torch.bool, device=device)
    non_selected_columns[selected_tensor] = False
    assert_close(
        logits_diff[..., non_selected_columns],
        torch.zeros_like(logits_diff[..., non_selected_columns]),
        f"{label}: loaded non-selected output columns changed",
    )
    return {
        "selected_input_delta_matches": True,
        "non_selected_input_unchanged": True,
        "selected_output_columns_match_hidden_delta": True,
        "non_selected_output_columns_unchanged": True,
    }


def run_behavior_checks(
    input_wrapper: SelectedDeltaInputEmbedding,
    output_wrapper: SelectedDeltaOutputHead,
    delta: nn.Parameter,
    selected_token_ids: list[int],
) -> dict[str, Any]:
    device = delta.device
    dtype = delta.dtype
    selected_atol = 1e-5 if dtype == torch.float32 else 5e-3
    selected_row = 4  # <|coord_0|>, after four wrappers.
    selected_id = selected_token_ids[selected_row]
    non_selected_id = 0
    assert_true(non_selected_id not in set(selected_token_ids), "probe non-selected id is selected")

    input_ids = torch.tensor([[selected_id, non_selected_id]], device=device)
    before_input = input_wrapper(input_ids).detach()
    hidden_size = delta.shape[1]
    perturb = torch.linspace(0.125, 0.25, hidden_size, device=device, dtype=dtype)
    with torch.no_grad():
        delta[selected_row].copy_(perturb)
    after_input = input_wrapper(input_ids).detach()
    input_diff = after_input - before_input
    assert_close(
        input_diff[0, 0],
        perturb,
        "selected input lookup did not change by delta",
        atol=selected_atol,
    )
    assert_close(
        input_diff[0, 1], torch.zeros_like(input_diff[0, 1]), "non-selected input lookup changed"
    )

    hidden = torch.zeros((2, hidden_size), device=device, dtype=dtype)
    hidden[0, : min(16, hidden_size)] = torch.arange(
        1, min(16, hidden_size) + 1, device=device, dtype=dtype
    ) / 100
    hidden[1, : min(16, hidden_size)] = 0.5
    with torch.no_grad():
        saved = delta.detach().clone()
        delta.zero_()
        logits_before = output_wrapper(hidden).detach()
        delta.copy_(saved)
        logits_after = output_wrapper(hidden).detach()
    logits_diff = logits_after - logits_before
    selected_tensor = torch.tensor(selected_token_ids, device=device, dtype=torch.long)
    expected_correction = (hidden.to(dtype) @ delta.detach().t()).to(logits_diff.dtype)
    assert_close(
        logits_diff.index_select(-1, selected_tensor),
        expected_correction,
        "selected output correction is not hidden @ delta.T",
        atol=selected_atol,
    )
    non_selected_columns = torch.ones(logits_diff.shape[-1], dtype=torch.bool, device=device)
    non_selected_columns[selected_tensor] = False
    assert_close(
        logits_diff[..., non_selected_columns],
        torch.zeros_like(logits_diff[..., non_selected_columns]),
        "non-selected output columns changed",
    )

    if delta.grad is not None:
        delta.grad = None
    if input_wrapper.base.weight.grad is not None:
        input_wrapper.base.weight.grad = None
    if output_wrapper.base.weight.grad is not None:
        output_wrapper.base.weight.grad = None
    tiny_input = torch.tensor(
        [[selected_token_ids[0], selected_token_ids[selected_row], non_selected_id]], device=device
    )
    tiny_hidden = input_wrapper(tiny_input).to(dtype)
    tiny_logits = output_wrapper(tiny_hidden)
    loss = tiny_logits[..., [selected_token_ids[0], selected_id]].sum()
    loss.backward()
    assert_true(delta.grad is not None, "selected delta grad is absent after backward")
    assert_true(delta.grad.abs().sum().item() > 0, "selected delta grad is zero after backward")
    base_embed_grad = input_wrapper.base.weight.grad
    base_head_grad = output_wrapper.base.weight.grad
    assert_true(
        base_embed_grad is None or base_embed_grad.abs().sum().item() == 0,
        "frozen base embedding received grad",
    )
    assert_true(
        base_head_grad is None or base_head_grad.abs().sum().item() == 0,
        "frozen base lm_head received grad",
    )

    return {
        "perturbed_token": "<|coord_0|>",
        "perturbed_token_id": selected_id,
        "perturbed_row": selected_row,
        "delta_abs_sum": float(delta.detach().abs().sum().item()),
        "delta_grad_abs_sum": float(delta.grad.detach().abs().sum().item()),
        "assertions": {
            "selected_input_delta_matches": True,
            "non_selected_input_unchanged": True,
            "selected_output_columns_match_hidden_delta": True,
            "non_selected_output_columns_unchanged": True,
            "selected_delta_grad_nonzero": True,
            "base_embedding_grad_absent_or_zero": True,
            "base_lm_head_grad_absent_or_zero": True,
        },
    }


def metadata_for(
    model_path: Path,
    config: Any,
    token_info: dict[str, Any],
    delta: torch.Tensor,
    dtype_name: str,
    device: str,
) -> dict[str, Any]:
    text_config = getattr(config, "text_config", None)
    return {
        "semantics": SEMANTICS,
        "tensor_key": TENSOR_KEY,
        "tensor_shape": list(delta.shape),
        "tensor_dtype": str(delta.dtype).replace("torch.", ""),
        "token_strings": token_info["token_strings"],
        "token_ids": token_info["token_ids"],
        "wrapper_token_ids": token_info["wrapper_token_ids"],
        "coord_start": token_info["coord_start"],
        "coord_end": token_info["coord_end"],
        "base_model_path": str(model_path),
        "base_config_sha256": sha256_file(model_path / "config.json"),
        "tokenizer_sha256": token_info["tokenizer_sha256"],
        "architectures": list(getattr(config, "architectures", []) or []),
        "model_type": getattr(config, "model_type", None),
        "text_hidden_size": getattr(text_config, "hidden_size", None),
        "text_vocab_size": getattr(text_config, "vocab_size", None),
        "tie_word_embeddings": bool(getattr(config, "tie_word_embeddings", False)),
        "text_tie_word_embeddings": bool(
            getattr(text_config, "tie_word_embeddings", getattr(config, "tie_word_embeddings", False))
        ),
        "cli_dtype": dtype_name,
        "cli_device": device,
    }


def save_and_reload_check(
    output_dir: Path,
    metadata: dict[str, Any],
    delta: nn.Parameter,
    output_wrapper: SelectedDeltaOutputHead,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = output_dir / "special_token_embeddings.safetensors"
    meta_path = output_dir / "special_token_embeddings.json"
    save_file({TENSOR_KEY: delta.detach().cpu()}, str(tensor_path))
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
        f.write("\n")

    loaded = load_file(str(tensor_path), device=str(delta.device))[TENSOR_KEY]
    assert_true(tuple(loaded.shape) == tuple(delta.shape), "reloaded delta shape mismatch")
    assert_true(loaded.dtype == delta.dtype, "reloaded delta dtype mismatch")
    assert_close(loaded, delta.detach(), "reloaded delta values mismatch")

    hidden = torch.zeros((1, delta.shape[1]), device=delta.device, dtype=delta.dtype)
    hidden[0, 0] = 1
    with torch.no_grad():
        saved = delta.detach().clone()
        logits_saved = output_wrapper(hidden).detach()
        delta.copy_(loaded)
        logits_loaded = output_wrapper(hidden).detach()
        delta.copy_(saved)
    assert_close(logits_loaded, logits_saved, "reload changed output behavior")
    return {"safetensors": str(tensor_path), "metadata": str(meta_path)}


def fresh_base_reload_check(
    *,
    model_path: Path,
    output_dir: Path,
    device: str,
    dtype_name: str,
    expected_metadata: dict[str, Any],
) -> dict[str, Any]:
    tensor_path = output_dir / "special_token_embeddings.safetensors"
    meta_path = output_dir / "special_token_embeddings.json"
    loaded_metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    assert_true(loaded_metadata["semantics"] == SEMANTICS, "fresh reload metadata semantics mismatch")
    assert_true(loaded_metadata["tensor_key"] == TENSOR_KEY, "fresh reload tensor key mismatch")
    assert_true(
        loaded_metadata["token_ids"] == expected_metadata["token_ids"],
        "fresh reload token ids mismatch",
    )
    assert_true(
        loaded_metadata["base_config_sha256"] == sha256_file(model_path / "config.json"),
        "fresh reload base config hash mismatch",
    )
    assert_true(
        loaded_metadata["tokenizer_sha256"] == sha256_file(model_path / "tokenizer.json"),
        "fresh reload tokenizer hash mismatch",
    )

    model = load_model(model_path, device, dtype_name)
    base_embedding, base_lm_head = validate_tied_model(model)
    input_wrapper, output_wrapper, delta = install_probe_wrappers(
        model, base_embedding, base_lm_head, loaded_metadata["token_ids"]
    )
    loaded_delta = load_file(str(tensor_path), device=str(delta.device))[TENSOR_KEY]
    assert_true(tuple(loaded_delta.shape) == tuple(delta.shape), "fresh reload delta shape mismatch")
    assert_true(loaded_delta.dtype == delta.dtype, "fresh reload delta dtype mismatch")
    with torch.no_grad():
        delta.copy_(loaded_delta)
    behavior = assert_loaded_delta_behavior(
        input_wrapper,
        output_wrapper,
        delta,
        loaded_metadata["token_ids"],
        label="fresh_base_reload",
    )
    return {
        "fresh_base_reloaded": True,
        "metadata_validated": True,
        "tensor_loaded": True,
        "runtime_tied_input_lm_head_identity": True,
        **behavior,
    }


def write_receipt(output_dir: Path, receipt: dict[str, Any]) -> Path:
    receipt_path = output_dir / "receipt.json"
    with receipt_path.open("w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2, sort_keys=True)
        f.write("\n")
    return receipt_path


def main() -> int:
    args = parse_args()
    try:
        token_info = validate_tokenizer(args.model_path)
        config = AutoConfig.from_pretrained(
            str(args.model_path), trust_remote_code=True, local_files_only=True
        )
        model = load_model(args.model_path, args.device, args.dtype)
        base_embedding, base_lm_head = validate_tied_model(model)
        input_wrapper, output_wrapper, delta = install_probe_wrappers(
            model, base_embedding, base_lm_head, token_info["token_ids"]
        )
        behavior = run_behavior_checks(input_wrapper, output_wrapper, delta, token_info["token_ids"])
        metadata = metadata_for(args.model_path, config, token_info, delta.detach(), args.dtype, args.device)
        payload_paths = save_and_reload_check(args.output_dir, metadata, delta, output_wrapper)
        in_process_loaded_behavior = assert_loaded_delta_behavior(
            input_wrapper,
            output_wrapper,
            delta,
            token_info["token_ids"],
            label="in_process_reload",
        )
        del input_wrapper
        del output_wrapper
        del delta
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        fresh_reload = fresh_base_reload_check(
            model_path=args.model_path,
            output_dir=args.output_dir,
            device=args.device,
            dtype_name=args.dtype,
            expected_metadata=metadata,
        )
        receipt = {
            "ok": True,
            "semantics": SEMANTICS,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "argv": sys.argv,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "versions": {
                "torch": torch.__version__,
                "transformers": package_version("transformers"),
                "safetensors": package_version("safetensors"),
            },
            "model_path": str(args.model_path),
            "output_dir": str(args.output_dir),
            "num_selected_tokens": len(token_info["token_ids"]),
            "runtime_tied_input_lm_head_identity": True,
            "tokenizer_identity": token_info["assertions"],
            "payload": payload_paths,
            "behavior": behavior,
            "in_process_reload": in_process_loaded_behavior,
            "fresh_base_reload": fresh_reload,
        }
        receipt_path = write_receipt(args.output_dir, receipt)
        print(json.dumps({**receipt, "receipt": str(receipt_path)}, indent=2, sort_keys=True))
        return 0
    except ProbeError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
