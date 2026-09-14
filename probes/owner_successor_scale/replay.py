"""Length-aware exact replay for the owner-successor throughput probe.

This module deliberately accepts the existing per-record materialization from
``parallel_owner_research.training._materialize``.  It only batches model
forwards; records, loss masks, reductions, and optimizer scheduling remain
with that consumer.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from src.qwen.native import exact_history_inputs, padded_histories


_HISTORY_KEYS = frozenset(
    {
        "input_ids",
        "attention_mask",
        "position_ids",
        "token_type_ids",
        "cache_position",
        "rope_deltas",
        "past_key_values",
        "inputs_embeds",
    }
)


def _target_ids(entry: Mapping[str, Any]) -> list[int]:
    record = entry.get("record")
    if not isinstance(record, Mapping):
        raise ValueError("batched replay entry lacks literal record")
    values = record.get("target_token_ids")
    if not isinstance(values, list) or not values or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in values
    ):
        raise ValueError("batched replay entry target IDs are not literal nonempty IDs")
    return list(values)


def _prompt_ids(entry: Mapping[str, Any]) -> list[int]:
    values = entry.get("prompt_ids")
    if not isinstance(values, list) or not values or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in values
    ):
        raise ValueError("batched replay entry prompt IDs are not literal nonempty IDs")
    return list(values)


def _combine_native_inputs(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Combine singleton materializations without guessing non-tensor semantics."""
    input_rows: list[Mapping[str, Any]] = []
    for entry in entries:
        inputs = entry.get("inputs")
        if not isinstance(inputs, Mapping):
            raise ValueError("batched replay entry lacks materialized inputs")
        input_ids = inputs.get("input_ids")
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("batched replay requires singleton materialized input_ids")
        input_rows.append(inputs)
    keys = set(input_rows[0])
    if any(set(row) != keys for row in input_rows[1:]):
        raise ValueError("batched replay materializations have different native fields")
    if "image_grid_thw" not in keys:
        raise ValueError("batched replay requires native image_grid_thw")

    result: dict[str, Any] = {}
    for key in keys - _HISTORY_KEYS:
        values = [row[key] for row in input_rows]
        if not all(isinstance(value, torch.Tensor) for value in values):
            raise ValueError(f"batched replay cannot combine non-tensor native field: {key}")
        tensors = [value for value in values if isinstance(value, torch.Tensor)]
        if not all(tensor.ndim >= 1 for tensor in tensors):
            raise ValueError(f"batched replay cannot combine scalar native field: {key}")
        try:
            result[key] = torch.cat(tensors, dim=0)
        except RuntimeError as exc:
            raise ValueError(f"batched replay cannot concatenate native field: {key}") from exc

    # ``exact_history_inputs`` only needs this source field for its batch-size
    # assertion before it replaces it with the exact replay histories.
    prompts = [_prompt_ids(entry) for entry in entries]
    result["input_ids"], _ = padded_histories(prompts, pad_token_id=0)
    return result


def batched_aligned_logits(
    model: Any, entries: Sequence[Mapping[str, Any]]
) -> list[torch.Tensor]:
    """Return one ordered, causal, exact-target logit tensor per materialized entry.

    One compact trailing width covers every item: it is the longest target plus
    its causal predecessor.  Each shorter item selects its own suffix from
    that common compact output.  Returned tensors have the identical per-item
    shape and causal alignment as serial ``prepare_replay(...).aligned_logits``.
    """
    entries = tuple(entries)
    if not entries:
        raise ValueError("batched replay requires at least one entry")
    targets = [_target_ids(entry) for entry in entries]
    histories = [(*_prompt_ids(entry), *target) for entry, target in zip(entries, targets, strict=True)]
    native_inputs = _combine_native_inputs(entries)
    replay_inputs = exact_history_inputs(
        model,
        native_inputs,
        histories,
        pad_token_id=0,
        logits_to_keep=max(map(len, targets)) + 1,
    )
    logits = model(**replay_inputs).logits
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3 or logits.shape[0] != len(entries):
        raise ValueError("batched replay model logits do not match input batch")
    width = max(map(len, targets)) + 1
    if logits.shape[1] != width:
        raise ValueError("batched replay model did not return requested compact logits")
    result: list[torch.Tensor] = []
    for index, target in enumerate(targets):
        start = width - len(target) - 1
        selected = logits[index, start : width - 1]
        if selected.shape[0] != len(target):
            raise ValueError("batched replay logits do not cover exact target")
        result.append(selected.float())
    return result
