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

from src.qwen.native import combine_singleton_native_inputs, exact_history_inputs, select_compact_replay_logits


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
    prompts = [_prompt_ids(entry) for entry in entries]
    histories = [(*prompt, *target) for prompt, target in zip(prompts, targets, strict=True)]
    native_inputs = combine_singleton_native_inputs(
        [entry.get("inputs") for entry in entries], prompt_token_ids=prompts,
    )
    replay_inputs = exact_history_inputs(
        model,
        native_inputs,
        histories,
        pad_token_id=0,
        logits_to_keep=max(map(len, targets)) + 1,
    )
    return select_compact_replay_logits(
        model(**replay_inputs).logits,
        [len(target) for target in targets],
    )
