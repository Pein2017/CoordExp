from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from src.tokens.row_offsets import CoordOffsetAdapter


_TOKEN_ROW_MODULE_NAMES = {"coord_offset_adapter", "token_row_offset_adapter"}
_UNRESOLVED_ACTIVE_ADAPTER = object()


def materialize_state_dict_for_vllm_full_sync(
    model: Any,
    state_dict: Mapping[str, Any],
    logger: Any | None = None,
) -> dict[str, Any]:
    """Return a vLLM-loadable full-sync snapshot.

    CoordExp token-row adapters are train-time modules, but native vLLM full
    sync accepts ordinary model weights. This helper patches the active adapter
    deltas into cloned embedding/head rows and strips learner-only adapter keys.
    """

    has_token_row_keys = any(_is_token_row_key(key) for key in state_dict)
    adapter = _find_active_coord_offset_adapter(model)
    if adapter is None:
        if has_token_row_keys:
            raise ValueError(
                "state_dict contains coord_offset_adapter token-row keys, but no "
                "active CoordOffsetAdapter could be discovered on the model"
            )
        return _validate_no_forbidden_keys(_filtered_state_dict(state_dict))

    materialized = _filtered_state_dict(state_dict)

    embed_key = _find_shortest_suffix_key(materialized, "embed_tokens.weight")
    if embed_key is None:
        raise ValueError(
            "active coord_offset_adapter requires an embed_tokens.weight entry "
            "in the vLLM full-sync state_dict"
        )
    embed_weight = materialized[embed_key]
    if not torch.is_tensor(embed_weight):
        raise ValueError(f"{embed_key} must be a tensor")

    coord_ids, embed_offset = _validate_coord_and_offset(
        coord_ids=getattr(adapter, "coord_ids", None),
        offset=getattr(adapter, "embed_offset", None),
        target=embed_weight,
        offset_name="embed_offset",
    )
    materialized[embed_key] = _patched_rows(embed_weight, coord_ids, embed_offset)

    tie_head = bool(getattr(adapter, "tie_head", True))
    head_key = _find_shortest_suffix_key(materialized, "lm_head.weight")
    if tie_head:
        if head_key is None:
            _require_confirmed_tie_for_missing_head(model)
        else:
            head_weight = materialized[head_key]
            if not torch.is_tensor(head_weight):
                raise ValueError(f"{head_key} must be a tensor")
            _, head_offset = _validate_coord_and_offset(
                coord_ids=coord_ids,
                offset=embed_offset,
                target=head_weight,
                offset_name="embed_offset",
            )
            materialized[head_key] = _patched_rows(head_weight, coord_ids, head_offset)
    else:
        if head_key is None:
            raise ValueError(
                "untied coord_offset_adapter full-sync requires lm_head.weight"
            )
        head_weight = materialized[head_key]
        if not torch.is_tensor(head_weight):
            raise ValueError(f"{head_key} must be a tensor")
        head_offset_value = getattr(adapter, "head_offset", None)
        if head_offset_value is None:
            raise ValueError(
                "untied coord_offset_adapter full-sync requires head_offset"
            )
        _, head_offset = _validate_coord_and_offset(
            coord_ids=coord_ids,
            offset=head_offset_value,
            target=head_weight,
            offset_name="head_offset",
        )
        materialized[head_key] = _patched_rows(head_weight, coord_ids, head_offset)

    materialized = _validate_no_forbidden_keys(materialized)
    if logger is not None:
        logger.info(
            "materialized coord_offset_adapter rows for vLLM full-sync: "
            "rows=%s embed_key=%s head_key=%s tie_head=%s",
            int(coord_ids.numel()),
            embed_key,
            head_key,
            tie_head,
        )
    return materialized


def _find_active_coord_offset_adapter(model: Any) -> CoordOffsetAdapter | None:
    named_modules = getattr(model, "named_modules", None)
    if not callable(named_modules):
        return None

    direct: CoordOffsetAdapter | None = None
    for name, module in named_modules():
        wrapped = _active_modules_to_save_adapter(module)
        if wrapped is _UNRESOLVED_ACTIVE_ADAPTER:
            return None
        if wrapped is not None:
            return wrapped
        if (
            direct is None
            and isinstance(module, CoordOffsetAdapter)
            and not _is_wrapper_internal_module_name(name)
        ):
            direct = module
    return direct


def _active_modules_to_save_adapter(
    module: Any,
) -> CoordOffsetAdapter | object | None:
    modules_to_save = getattr(module, "modules_to_save", None)
    if not _looks_like_module_mapping(modules_to_save):
        return None

    active_adapters = _active_adapter_names(module)
    if active_adapters is not None:
        for name in active_adapters:
            if name in modules_to_save and isinstance(
                modules_to_save[name], CoordOffsetAdapter
            ):
                return modules_to_save[name]
        return _UNRESOLVED_ACTIVE_ADAPTER

    for candidate in modules_to_save.values():
        if isinstance(candidate, CoordOffsetAdapter):
            return candidate
    return None


def _active_adapter_names(module: Any) -> list[str] | None:
    if hasattr(module, "active_adapters"):
        active_adapters = getattr(module, "active_adapters")
    elif hasattr(module, "active_adapter"):
        active_adapters = getattr(module, "active_adapter")
    else:
        return None

    if isinstance(active_adapters, str):
        return [active_adapters]
    return list(active_adapters or [])


def _is_wrapper_internal_module_name(name: str) -> bool:
    return any(
        component in {"original_module", "modules_to_save"}
        for component in name.split(".")
    )


def _looks_like_module_mapping(value: Any) -> bool:
    return isinstance(value, Mapping) or (
        hasattr(value, "__contains__")
        and hasattr(value, "__getitem__")
        and hasattr(value, "values")
    )


def _filtered_state_dict(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in state_dict.items() if not _is_forbidden_key(key)
    }


def _is_token_row_key(key: str) -> bool:
    return any(component in _TOKEN_ROW_MODULE_NAMES for component in key.split("."))


def _is_forbidden_key(key: str) -> bool:
    return any(_is_forbidden_component(component) for component in key.split("."))


def _is_forbidden_component(component: str) -> bool:
    return (
        component in _TOKEN_ROW_MODULE_NAMES
        or component == "modules_to_save"
        or component == "original_module"
        or component.startswith("lora_")
    )


def _validate_no_forbidden_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
    forbidden = [key for key in state_dict if _is_forbidden_key(key)]
    if forbidden:
        preview = ", ".join(forbidden[:5])
        raise ValueError(
            "vLLM full-sync state_dict still contains forbidden learner-only "
            f"keys: {preview}"
        )
    return state_dict


def _find_shortest_suffix_key(state_dict: Mapping[str, Any], suffix: str) -> str | None:
    candidates = [key for key in state_dict if key.endswith(suffix)]
    if not candidates:
        return None
    return min(candidates, key=lambda key: (len(key), key))


def _validate_coord_and_offset(
    *,
    coord_ids: Any,
    offset: Any,
    target: torch.Tensor,
    offset_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not torch.is_tensor(coord_ids):
        raise ValueError("coord_ids must be a tensor")
    if coord_ids.ndim != 1:
        raise ValueError("coord_ids must be 1D for vLLM full-sync materialization")
    if coord_ids.numel() == 0:
        raise ValueError("coord_ids must be non-empty for vLLM full-sync materialization")
    if not torch.is_tensor(offset):
        raise ValueError(f"{offset_name} must be a tensor")
    if offset.ndim != 2:
        raise ValueError(f"{offset_name} must be 2D")
    if offset.size(0) != coord_ids.numel():
        raise ValueError(
            f"{offset_name} row count {offset.size(0)} must match coord_ids "
            f"row count {coord_ids.numel()}"
        )
    if target.ndim != 2:
        raise ValueError("target weight must be 2D")
    if offset.size(1) != target.size(1):
        raise ValueError(
            f"{offset_name} hidden size {offset.size(1)} must match target "
            f"hidden size {target.size(1)}"
        )

    coord_ids_long = coord_ids.detach().to(dtype=torch.long, device="cpu")
    if torch.any(coord_ids_long < 0):
        raise ValueError("coord_ids contain negative token ids outside vocab bounds")
    if torch.any(coord_ids_long >= target.size(0)):
        raise ValueError("coord_ids exceed target vocab row bounds")
    return coord_ids.detach(), offset.detach()


def _patched_rows(
    weight: torch.Tensor,
    coord_ids: torch.Tensor,
    offset: torch.Tensor,
) -> torch.Tensor:
    row_ids = coord_ids.to(device=weight.device, dtype=torch.long)
    offset_on_weight = offset.to(device=weight.device, dtype=weight.dtype)
    patched = weight.clone()
    patched[row_ids] = patched[row_ids] + offset_on_weight
    return patched


def _require_confirmed_tie_for_missing_head(model: Any) -> None:
    config_tie = _config_tie_word_embeddings(model)
    storage_tied = _input_output_storage_tied(model)
    if storage_tied is True:
        return
    if config_tie is False:
        raise ValueError(
            "tied coord_offset_adapter state_dict is missing lm_head.weight; "
            "config.tie_word_embeddings=False and storage does not confirm tying"
        )
    if config_tie is True:
        return
    raise ValueError(
        "tied coord_offset_adapter state_dict is missing lm_head.weight; "
        "tie_word_embeddings or shared input/output embedding storage must "
        "confirm tying"
    )


def _config_tie_word_embeddings(model: Any) -> bool | None:
    config = getattr(model, "config", None)
    value = getattr(config, "tie_word_embeddings", None)
    if value is not None:
        return bool(value)
    text_config = getattr(config, "text_config", None)
    value = getattr(text_config, "tie_word_embeddings", None)
    if value is not None:
        return bool(value)
    return None


def _input_output_storage_tied(model: Any) -> bool | None:
    get_input = getattr(model, "get_input_embeddings", None)
    get_output = getattr(model, "get_output_embeddings", None)
    if not callable(get_input) or not callable(get_output):
        return None
    input_module = get_input()
    output_module = get_output()
    input_weight = getattr(input_module, "weight", None)
    output_weight = getattr(output_module, "weight", None)
    if not torch.is_tensor(input_weight) or not torch.is_tensor(output_weight):
        return None
    return input_weight.untyped_storage().data_ptr() == output_weight.untyped_storage().data_ptr()


__all__ = ["materialize_state_dict_for_vllm_full_sync"]
