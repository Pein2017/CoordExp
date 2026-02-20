"""Coord offset adapter that adds trainable offsets for coord token IDs.

This module keeps the base embedding / lm_head weights frozen and applies
per‑ID offsets during forward passes. Offsets live under a dedicated
submodule (`coord_offset_adapter`) so they can be persisted via PEFT
`modules_to_save` without sidecar files.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn

DEFAULT_COORD_ID_RANGE: Tuple[int, int] = (151_670, 152_669)  # inclusive
DEFAULT_EXCLUDE_IDS = {151_669}  # <|coord_*|>


@dataclass
class CoordOffsetConfig:
    enabled: bool = False
    tie_head: bool = True
    ids: Sequence[int] = ()
    embed_lr: float | None = None
    head_lr: float | None = None
    weight_decay: float = 0.0
    dtype: str | None = None  # 'auto' or torch dtype name


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
    raise ValueError(f"Unsupported coord_offset.dtype: {dtype}")


class CoordOffsetAdapter(nn.Module):
    """Holds coord offsets and applies them via forward hooks."""

    def __init__(
        self,
        *,
        coord_ids: Sequence[int],
        tie_head: bool = True,
        embed_dim: int,
        head_dim: int,
        base_dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        coord_ids = _sanitize_ids(coord_ids)
        if not coord_ids:
            raise ValueError("coord_offset ids must be non-empty when enabled")

        self.tie_head = bool(tie_head)
        if self.tie_head and embed_dim != head_dim:
            raise ValueError(
                f"tie_head requires embed_dim == head_dim (got {embed_dim} vs {head_dim})"
            )

        self.register_buffer("coord_ids", torch.tensor(coord_ids, dtype=torch.long))
        self.embed_offset = nn.Parameter(
            torch.zeros(len(coord_ids), embed_dim, device=device, dtype=base_dtype)
        )
        if self.tie_head:
            self.head_offset = None
        else:
            self.head_offset = nn.Parameter(
                torch.zeros(len(coord_ids), head_dim, device=device, dtype=base_dtype)
            )
        self._embed_hook_handle = None
        self._head_hook_handle = None

    @property
    def module_name(self) -> str:
        return "coord_offset_adapter"

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

            coord_ids = self.coord_ids.to(input_ids.device)
            flat_ids = input_ids.reshape(-1)
            mask = torch.isin(flat_ids, coord_ids)
            if not torch.any(mask):
                return output

            # Map token ids -> offset rows via searchsorted (coord_ids is sorted)
            matched_ids = flat_ids[mask]
            idx = torch.searchsorted(coord_ids, matched_ids)
            offsets = self.embed_offset.to(output.device).index_select(0, idx)

            flat_out = output.view(-1, output.size(-1))
            flat_out[mask] = flat_out[mask] + offsets.to(flat_out.dtype)
            return output

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

            coord_ids = self.coord_ids.to(output.device)
            flat_output = output.view(-1, output.size(-1))
            # Expand coord_ids to match extra_logits shape for scatter_add_
            scatter_idx = coord_ids.unsqueeze(0).expand(extra_logits.size(0), -1)
            flat_output.scatter_add_(1, scatter_idx, extra_logits.to(flat_output.dtype))
            return output

        if self._embed_hook_handle is None:
            self._embed_hook_handle = embed_module.register_forward_hook(_embed_hook)
        if self._head_hook_handle is None:
            self._head_hook_handle = head_module.register_forward_hook(_head_hook)


def _find_first_named_module(model: nn.Module, target_name: str) -> nn.Module | None:
    for name, module in model.named_modules():
        if name.endswith(target_name):
            return module
    return None


def install_coord_offset_adapter(
    model: nn.Module,
    *,
    coord_ids: Iterable[int] | None,
    tie_head: bool = True,
    dtype: str | None = None,
) -> CoordOffsetAdapter:
    """Install coord offset adapter onto the model.

    Returns the created adapter module for further inspection.
    """
    if hasattr(model, "coord_offset_adapter"):
        return getattr(model, "coord_offset_adapter")

    embed_module = _find_first_named_module(model, "embed_tokens")
    head_module = _find_first_named_module(model, "lm_head")
    if embed_module is None or head_module is None:
        raise ValueError(
            "Could not locate embed_tokens and lm_head modules needed for coord_offset."
        )

    embed_weight = getattr(embed_module, "weight", None)
    head_weight = getattr(head_module, "weight", None)
    if embed_weight is None or head_weight is None:
        raise ValueError("embed_tokens/lm_head missing weight parameters.")

    target_dtype = _to_dtype(embed_weight, dtype)

    # Freeze base weights to ensure only offsets learn
    embed_weight.requires_grad_(False)
    head_weight.requires_grad_(False)

    adapter = CoordOffsetAdapter(
        coord_ids=_sanitize_ids(coord_ids),
        tie_head=tie_head,
        embed_dim=embed_weight.size(1),
        head_dim=head_weight.size(1),
        base_dtype=target_dtype,
        device=embed_weight.device,
    )
    adapter.attach(embed_module, head_module)
    setattr(model, adapter.module_name, adapter)
    return adapter


def reattach_coord_offset_hooks(model: nn.Module) -> CoordOffsetAdapter | None:
    """Re-bind coord-offset hooks after PEFT/Swift wrapping.

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
    adapter: CoordOffsetAdapter | None = None
    for _, module in model.named_modules():
        if isinstance(module, CoordOffsetAdapter):
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
            if isinstance(target, CoordOffsetAdapter):
                adapter = target
                break

    if adapter is None:
        return None

    embed_module = _find_first_named_module(model, "embed_tokens")
    head_module = _find_first_named_module(model, "lm_head")
    if embed_module is None or head_module is None:
        raise ValueError(
            "Could not locate embed_tokens and lm_head modules needed to reattach coord_offset hooks."
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
