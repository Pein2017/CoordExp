"""Raw-row collator for Stage-1 set-continuation training.

The set-continuation trainer re-encodes branch candidates at loss time, so the
collator must preserve the original rendered sample metadata instead of routing
through the ordinary Stage-1 diagnostics wrapper.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any, Callable

import torch

from src.data_collators.enrichers import resolve_dataset_label


_PRESERVED_META_KEYS: tuple[str, ...] = (
    "assistant_payload",
    "messages",
    "metadata",
    "objects",
    "sample_id",
    "base_idx",
    "dataset",
    "dataset_name",
)


def _validate_assistant_payload(row: Mapping[str, Any], *, row_index: int) -> None:
    payload = row.get("assistant_payload")
    if not isinstance(payload, Mapping):
        raise ValueError(
            f"stage1_set_continuation sample {row_index} is missing assistant_payload"
        )
    objects = payload.get("objects")
    if not isinstance(objects, Sequence) or isinstance(
        objects, (str, bytes, bytearray)
    ):
        raise ValueError(
            "stage1_set_continuation requires assistant_payload.objects to be a sequence"
        )


def _validate_messages(row: Mapping[str, Any], *, row_index: int) -> None:
    messages = row.get("messages")
    if not isinstance(messages, Sequence) or isinstance(
        messages, (str, bytes, bytearray)
    ):
        raise ValueError(
            f"stage1_set_continuation sample {row_index} is missing messages"
        )
    has_assistant = any(
        isinstance(message, Mapping) and message.get("role") == "assistant"
        for message in messages
    )
    if not has_assistant:
        raise ValueError(
            f"stage1_set_continuation sample {row_index} messages must include an assistant message"
        )


def _extract_set_continuation_meta(
    row: Mapping[str, Any], *, row_index: int
) -> dict[str, Any]:
    _validate_assistant_payload(row, row_index=row_index)
    _validate_messages(row, row_index=row_index)
    meta = {
        key: copy.deepcopy(row[key])
        for key in _PRESERVED_META_KEYS
        if key in row
    }
    meta["dataset_label"] = resolve_dataset_label(row)
    return meta


def _default_pack_num_samples(size: int) -> torch.Tensor:
    return torch.ones((int(size),), dtype=torch.long)


def build_stage1_set_continuation_collator(
    base_collator: Callable[[list[dict[str, Any]]], dict[str, Any]] | None = None,
) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
    """Build a collator that preserves raw metadata under `set_continuation_meta`.

    If a base collator is supplied, its tensor outputs are retained as a
    convenience for debugging, but raw non-model extras still live exclusively in
    `set_continuation_meta`.
    """

    def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        if not isinstance(batch, list):
            raise TypeError("stage1_set_continuation collator expects a list batch")
        meta = [
            _extract_set_continuation_meta(row, row_index=index)
            for index, row in enumerate(batch)
        ]
        collated = base_collator(batch) if base_collator is not None else {}
        if not isinstance(collated, Mapping):
            raise TypeError(
                "stage1_set_continuation base_collator must return a mapping"
            )
        out = dict(collated)
        out["set_continuation_meta"] = meta
        out["dataset_labels"] = [item["dataset_label"] for item in meta]
        out.setdefault("dataset_segments", [0 for _ in meta])
        out.setdefault("pack_num_samples", _default_pack_num_samples(len(meta)))
        return out

    return _collate


__all__ = ["build_stage1_set_continuation_collator"]
