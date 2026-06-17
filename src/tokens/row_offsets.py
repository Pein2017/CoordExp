"""Trainable token-row offset adapter.

This module keeps the base embedding / lm_head weights frozen and applies
per-ID offsets during forward passes. Offsets live under a dedicated
submodule (`token_embeddings_adapter`) so they can be persisted via PEFT
`modules_to_save` without sidecar files.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn

DEFAULT_COORD_ID_RANGE: Tuple[int, int] = (151_670, 152_669)  # inclusive
DEFAULT_EXCLUDE_IDS = {151_669}  # <|coord_*|>
TOKEN_EMBEDDINGS_ADAPTER_NAME = "token_embeddings_adapter"


def _sanitize_ids(ids: Iterable[int] | None) -> List[int]:
    if ids is None:
        start, end = DEFAULT_COORD_ID_RANGE
        ids = range(start, end + 1)
    unique_sorted = sorted({int(i) for i in ids if int(i) not in DEFAULT_EXCLUDE_IDS})
    return unique_sorted


def _to_dtype(tensor: torch.Tensor, dtype: str | None) -> torch.dtype:
    if dtype is None or dtype.lower() == "auto":
        return tensor.dtype
    dtype_norm = dtype.lower()
    if dtype_norm in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if dtype_norm in {"fp16", "float16", "half"}:
        return torch.float16
    if dtype_norm in {"fp32", "float32", "single"}:
        return torch.float32
    raise ValueError(f"Unsupported token_embeddings_adapter.dtype: {dtype}")


class TokenEmbeddingsAdapter(nn.Module):
    """Holds token-row offsets and applies them via forward hooks."""

    def __init__(
        self,
        *,
        token_ids: Sequence[int] | None = None,
        tie_head: bool = True,
        embed_dim: int,
        head_dim: int,
        base_dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        token_ids = _sanitize_ids(token_ids)
        if not token_ids:
            raise ValueError("token_embeddings_adapter ids must be non-empty when enabled")

        self.tie_head = bool(tie_head)
        if self.tie_head and embed_dim != head_dim:
            raise ValueError(
                f"tie_head requires embed_dim == head_dim (got {embed_dim} vs {head_dim})"
            )

        self.register_buffer("token_ids", torch.tensor(token_ids, dtype=torch.long))
        self.embed_offset = nn.Parameter(
            torch.zeros(len(token_ids), embed_dim, device=device, dtype=base_dtype)
        )
        if self.tie_head:
            self.head_offset = None
        else:
            self.head_offset = nn.Parameter(
                torch.zeros(len(token_ids), head_dim, device=device, dtype=base_dtype)
            )
        self._embed_hook_handle = None
        self._head_hook_handle = None

    @property
    def module_name(self) -> str:
        return TOKEN_EMBEDDINGS_ADAPTER_NAME

    def attach(self, embed_module: nn.Embedding, head_module: nn.Linear) -> None:
        """Register forward hooks on embedding and lm_head."""

        def _embed_hook(module: nn.Module, inputs, output):
            if not isinstance(output, torch.Tensor):
                return output
            if not inputs:
                return output
            input_ids = inputs[0]
            if not torch.is_tensor(input_ids):
                return output

            coord_ids = self.token_ids.to(input_ids.device)
            flat_ids = input_ids.reshape(-1)
            mask = torch.isin(flat_ids, coord_ids)
            if not torch.any(mask):
                return output

            # Map token ids -> offset rows via searchsorted (coord_ids is sorted)
            matched_ids = flat_ids[mask]
            output_device = output.device
            idx = torch.searchsorted(coord_ids, matched_ids).to(output_device)
            output_mask = mask.to(output_device)
            offsets = self.embed_offset.to(output_device).index_select(0, idx)

            delta = torch.zeros_like(output)
            flat_delta = delta.reshape(-1, delta.size(-1))
            flat_delta[output_mask] = offsets.to(flat_delta.dtype)
            return output + delta

        def _head_hook(module: nn.Module, inputs, output):
            if not isinstance(output, torch.Tensor):
                return output
            if not inputs:
                return output
            hidden_states = inputs[0]
            if not torch.is_tensor(hidden_states):
                return output

            head_offset = self.embed_offset if self.tie_head else self.head_offset
            if head_offset is None:
                return output
            head_offset = head_offset.to(hidden_states.device)
            if head_offset.numel() == 0:
                return output

            flat_hidden = hidden_states.reshape(-1, hidden_states.size(-1))
            extra_logits = flat_hidden.to(head_offset.dtype) @ head_offset.T  # (N, num_ids)

            coord_ids = self.token_ids.to(output.device)
            delta = torch.zeros_like(output)
            flat_delta = delta.reshape(-1, delta.size(-1))
            # Clone the expanded index so repeated independent forwards do not
            # share a view-backed LongTensor across autograd graphs.
            scatter_idx = (
                coord_ids.unsqueeze(0).expand(extra_logits.size(0), -1).clone()
            )
            flat_delta.scatter_add_(
                1,
                scatter_idx,
                extra_logits.to(device=output.device, dtype=flat_delta.dtype),
            )
            return output + delta

        if self._embed_hook_handle is None:
            self._embed_hook_handle = embed_module.register_forward_hook(_embed_hook)
        if self._head_hook_handle is None:
            self._head_hook_handle = head_module.register_forward_hook(_head_hook)


def _find_first_named_module(model: nn.Module, target_name: str) -> nn.Module | None:
    for name, module in model.named_modules():
        if name.endswith(target_name):
            return module
    return None


def install_token_embeddings_adapter(
    model: nn.Module,
    *,
    token_ids: Iterable[int] | None = None,
    tie_head: bool = True,
    dtype: str | None = None,
) -> TokenEmbeddingsAdapter:
    """Install token embeddings adapter onto the model.

    Returns the created adapter module for further inspection.
    """
    if hasattr(model, TOKEN_EMBEDDINGS_ADAPTER_NAME):
        return getattr(model, TOKEN_EMBEDDINGS_ADAPTER_NAME)

    embed_module = _find_first_named_module(model, "embed_tokens")
    head_module = _find_first_named_module(model, "lm_head")
    if embed_module is None or head_module is None:
        raise ValueError(
            "Could not locate embed_tokens and lm_head modules needed for token_embeddings_adapter."
        )

    embed_weight = getattr(embed_module, "weight", None)
    head_weight = getattr(head_module, "weight", None)
    if embed_weight is None or head_weight is None:
        raise ValueError("embed_tokens/lm_head missing weight parameters.")

    target_dtype = _to_dtype(embed_weight, dtype)

    # Freeze base weights to ensure only offsets learn
    embed_weight.requires_grad_(False)
    head_weight.requires_grad_(False)

    adapter = TokenEmbeddingsAdapter(
        token_ids=_sanitize_ids(token_ids),
        tie_head=tie_head,
        embed_dim=embed_weight.size(1),
        head_dim=head_weight.size(1),
        base_dtype=target_dtype,
        device=embed_weight.device,
    )
    adapter.attach(embed_module, head_module)
    setattr(model, adapter.module_name, adapter)
    return adapter


def reattach_token_embeddings_adapter_hooks(model: nn.Module) -> TokenEmbeddingsAdapter | None:
    """Re-bind token embeddings adapter hooks after PEFT/Swift wrapping.

    When the adapter is wrapped by ModulesToSaveWrapper, the active module is the
    copied adapter under modules_to_save, not the original instance we attached
    before wrapping. This helper finds the active adapter instance and reattaches
    its hooks to the current embed_tokens / lm_head modules.
    """
    try:
        from peft.utils.other import ModulesToSaveWrapper
    except ImportError:
        ModulesToSaveWrapper = None  # type: ignore

    # Find adapter instance (unwrap ModulesToSaveWrapper when present)
    adapter: TokenEmbeddingsAdapter | None = None
    for _, module in model.named_modules():
        if isinstance(module, TokenEmbeddingsAdapter):
            adapter = module
            break
        if ModulesToSaveWrapper and isinstance(module, ModulesToSaveWrapper):
            active = module.active_adapters[0] if getattr(module, "active_adapters", []) else None
            target = None
            if active and active in module.modules_to_save:
                target = module.modules_to_save[active]
            elif len(module.modules_to_save):
                # fallback to any stored module
                target = next(iter(module.modules_to_save.values()))
            if isinstance(target, TokenEmbeddingsAdapter):
                adapter = target
                break

    if adapter is None:
        return None

    embed_module = _find_first_named_module(model, "embed_tokens")
    head_module = _find_first_named_module(model, "lm_head")
    if embed_module is None or head_module is None:
        raise ValueError(
            "Could not locate embed_tokens and lm_head modules needed to reattach token_embeddings_adapter hooks."
        )

    # Remove stale hooks (they point to pre-wrapped modules)
    if getattr(adapter, "_embed_hook_handle", None):
        adapter._embed_hook_handle.remove()  # type: ignore[attr-defined]
        adapter._embed_hook_handle = None  # type: ignore[attr-defined]
    if getattr(adapter, "_head_hook_handle", None):
        adapter._head_hook_handle.remove()  # type: ignore[attr-defined]
        adapter._head_hook_handle = None  # type: ignore[attr-defined]

    adapter.attach(embed_module, head_module)
    return adapter


__all__ = [
    "DEFAULT_COORD_ID_RANGE",
    "DEFAULT_EXCLUDE_IDS",
    "TokenEmbeddingsAdapter",
    "TOKEN_EMBEDDINGS_ADAPTER_NAME",
    "install_token_embeddings_adapter",
    "reattach_token_embeddings_adapter_hooks",
]
