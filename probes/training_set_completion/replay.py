"""Completion-route replay shared by actual fixed-teacher and Source256 callers.

Owns native materialization, causal alignment and original per-route CE/geometry
terms. Cohorts, route selection, alternative CE denominators, branch weighting,
update schedules and gradient collectives remain with their recipe owners.
"""

from __future__ import annotations
from typing import Any, Mapping, Sequence
import torch
from probes.training_set_completion import training
from src.qwen.native import select_compact_replay_logits


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def microbatch_slices(count: int, microbatch_size: int) -> list[list[int]]:
    require(type(count) is int and count > 0, "microbatch item count")
    require(microbatch_size in (1, 2, 3), "microbatch size must be 1, 2, or 3")
    return [
        list(range(start, min(count, start + microbatch_size)))
        for start in range(0, count, microbatch_size)
    ]


def batched_aligned_logits(
    model: torch.nn.Module,
    native_inputs: Mapping[str, Any],
    routes: Sequence[Mapping[str, Any]],
    *,
    pad_token_id: int,
) -> tuple[list[torch.Tensor], dict[str, int]]:
    from src.qwen.native import exact_history_inputs

    continuations = [list(route["continuation_token_ids"]) for route in routes]
    histories = [
        [*route["prompt_token_ids"], *continuation]
        for route, continuation in zip(routes, continuations, strict=True)
    ]
    width = max(map(len, continuations)) + 1
    inputs = exact_history_inputs(
        model,
        native_inputs,
        histories,
        pad_token_id=pad_token_id,
        logits_to_keep=width,
    )
    output = model(**inputs).logits
    return select_compact_replay_logits(output, [len(row) for row in continuations]), {
        "history_width": max(map(len, histories)),
        "history_padding_tokens": sum(
            max(map(len, histories)) - len(row) for row in histories
        ),
        "compact_logit_width": width,
    }


def route_terms(
    logits: torch.Tensor,
    route: Mapping[str, Any],
    hinge: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, int, dict[str, Any]]:
    targets = torch.tensor(
        route["continuation_token_ids"], dtype=torch.long, device=logits.device
    )
    ce, metrics = training.masked_ce_loss(logits, targets, route["ce_weights"])
    raw_hinge = training.raw_axis_validity_hinge(
        logits,
        route["trusted_boxes"],
        coordinate_token_ids=hinge["coordinate_token_ids"],
        coordinate_bin_values=hinge["coordinate_bin_values"],
        margin=hinge["margin"],
    )
    require(
        bool(torch.isfinite(ce)) and bool(torch.isfinite(raw_hinge)),
        "nonfinite route terms",
    )
    active = int(metrics["active_tokens"])
    card = {
        "route_id": route["route_id"],
        "active_tokens": active,
        "masked_nll_sum": metrics["masked_nll_sum"],
        "active_token_mean_ce": float(ce.detach()),
        "raw_axis_validity_hinge": float(raw_hinge.detach()),
    }
    return ce, raw_hinge, active, card


def prepare_microbatches(
    qwen: Any,
    manifest: Mapping[str, Any],
    routes: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    microbatch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from src.inference.bound_requests import (
        build_bound_native_requests as build_requests,
    )
    from src.qwen.native import prepare_native_inputs

    groups = []
    prompt_padding = 0
    for indices in microbatch_slices(len(routes), microbatch_size):
        selected = [routes[index] for index in indices]
        requests, _ = build_requests(
            qwen, manifest["model_config"], [route["case"] for route in selected]
        )
        batch = prepare_native_inputs(
            qwen.processor, requests, device=device, record_media_identity=True
        )
        require(
            [list(row) for row in batch.prompt_token_ids]
            == [route["prompt_token_ids"] for route in selected],
            "live batched prompt differs",
        )
        require(
            list(batch.media_sha256 or ())
            == [route["image_identity"]["executed_media_sha256"] for route in selected],
            "live batched media differs",
        )
        require(
            [list(row) if row is not None else None for row in batch.image_grids]
            == [
                route["image_identity"]["observed_image_grid_thw"] for route in selected
            ],
            "live batched grids differ",
        )
        attention = batch.inputs.get("attention_mask")
        require(
            isinstance(attention, torch.Tensor) and attention.ndim == 2,
            "batched prompt attention mask",
        )
        prompt_padding += int((attention == 0).sum().item())
        groups.append({"routes": selected, "inputs": dict(batch.inputs)})
    return groups, {
        "microbatch_size": microbatch_size,
        "microbatch_count": len(groups),
        "microbatch_image_counts": [len(group["routes"]) for group in groups],
        "prompt_padding_tokens": prompt_padding,
    }
