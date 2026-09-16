"""Activation-checkpointing mechanics for Qwen language decoder blocks."""

from __future__ import annotations

from typing import Any


def install_language_decoder_checkpointing(
    model: Any,
    *,
    expected_layer_count: int,
) -> dict[str, Any]:
    """Checkpoint exactly the named language decoder blocks while preserving eval mode."""

    from torch.utils.checkpoint import checkpoint

    if model.training:
        raise ValueError("checkpoint installation requires eval-mode model")
    owners = [
        (name, module)
        for name, module in model.named_modules()
        if name.endswith("language_model")
    ]
    if len(owners) != 1:
        raise ValueError("single named language decoder owner")
    owner_name, owner = owners[0]
    layers = owner.layers
    if len(layers) != expected_layer_count or any(layer.training for layer in layers):
        raise ValueError("unexpected or non-eval language decoder blocks")
    state: dict[str, Any] = {
        "enabled": False,
        "phase": "disabled",
        "decoder_layers": len(layers),
        "decoder_module_name": owner_name,
        "use_reentrant": False,
        "layer_calls": 0,
        "checkpoint_invocations": 0,
        "checkpoint_body_calls": 0,
        "bypass_no_grad_calls": 0,
        "bypass_disabled_calls": 0,
    }

    def wrapper(original_forward: Any) -> Any:
        def checkpointed_forward(*args: Any, **kwargs: Any) -> Any:
            import torch

            state["layer_calls"] += 1
            if not torch.is_grad_enabled():
                state["bypass_no_grad_calls"] += 1
                return original_forward(*args, **kwargs)
            if not state["enabled"]:
                state["bypass_disabled_calls"] += 1
                return original_forward(*args, **kwargs)
            state["checkpoint_invocations"] += 1

            def invoke(*inner_args: Any, **inner_kwargs: Any) -> Any:
                state["checkpoint_body_calls"] += 1
                return original_forward(*inner_args, **inner_kwargs)

            return checkpoint(invoke, *args, use_reentrant=False, **kwargs)

        return checkpointed_forward

    for layer in layers:
        layer.forward = wrapper(layer.forward)
    if model.training or any(layer.training for layer in layers):
        raise ValueError("checkpoint installation changed eval mode")
    return state


def language_decoder_checkpointing_receipt(
    model: Any,
    state: dict[str, Any],
) -> dict[str, Any]:
    """Summarize checkpoint use without owning the caller's enable/phase policy."""

    layers = model.get_submodule(state["decoder_module_name"]).layers
    expected_layer_count = int(state["decoder_layers"])
    if len(layers) != expected_layer_count or model.training or any(layer.training for layer in layers):
        raise ValueError("checkpointed model left its bound eval-mode decoder layout")
    body_calls = int(state["checkpoint_body_calls"])
    invocations = int(state["checkpoint_invocations"])
    return {
        "mode": "non_reentrant_per_language_decoder_block_grad_only",
        "decoder_layers": expected_layer_count,
        "use_reentrant": False,
        "decoder_module_name": state["decoder_module_name"],
        "enabled": bool(state["enabled"]),
        "phase": str(state["phase"]),
        "layer_calls": int(state["layer_calls"]),
        "checkpoint_invocations": invocations,
        "checkpoint_body_calls": body_calls,
        "recompute_body_calls": max(0, body_calls - invocations),
        "bypass_no_grad_calls": int(state["bypass_no_grad_calls"]),
        "bypass_disabled_calls": int(state["bypass_disabled_calls"]),
        "model_eval": True,
        "all_decoder_layers_eval": True,
    }
