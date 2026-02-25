from __future__ import annotations

from typing import Any, Mapping, MutableMapping, Sequence, Tuple

import torch


def prepare_forward_inputs(
    *,
    model: Any,
    inputs: Mapping[str, Any],
    ignored_keys: Sequence[str],
    packing_enabled: bool,
    where: str,
) -> tuple[Any, dict[str, Any], str]:
    core_model = getattr(model, "module", model)
    inputs_for_model = {k: v for k, v in inputs.items() if k not in set(ignored_keys)}

    model_type = str(
        getattr(getattr(core_model, "config", None), "model_type", "") or ""
    )

    text_position_ids = inputs.get("text_position_ids")
    position_ids = inputs_for_model.get("position_ids")
    if (
        model_type.startswith("qwen")
        and isinstance(text_position_ids, torch.Tensor)
        and isinstance(position_ids, torch.Tensor)
        and position_ids.ndim == 3
        and int(position_ids.shape[0]) == 3
        and text_position_ids.ndim == 2
        and text_position_ids.shape == position_ids.shape[1:]
    ):
        inputs_for_model["position_ids"] = torch.cat(
            [text_position_ids.unsqueeze(0), position_ids],
            dim=0,
        )

    if packing_enabled and model_type.startswith("qwen"):
        pos4 = inputs_for_model.get("position_ids")
        if not (
            isinstance(pos4, torch.Tensor)
            and pos4.ndim == 3
            and int(pos4.shape[0]) == 4
        ):
            raise ValueError(
                f"{where}: packing enabled but missing Qwen3-VL 4-row position_ids metadata"
            )

    return core_model, inputs_for_model, model_type


def run_no_cache_forward(*, model: Any, inputs_for_model: Mapping[str, Any]) -> Any:
    fwd_inputs: MutableMapping[str, Any] = dict(inputs_for_model)
    fwd_inputs["use_cache"] = False
    fwd_inputs.pop("past_key_values", None)
    return model(**fwd_inputs)


def assert_unsliced_logits(
    *,
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    where: str,
) -> None:
    if logits.shape[:2] != input_ids.shape[:2]:
        raise ValueError(
            f"{where}: model returned sliced logits (logits_to_keep-style); full logits are required"
        )
