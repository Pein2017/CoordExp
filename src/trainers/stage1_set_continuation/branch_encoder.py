"""Branch encoding helpers for Stage-1 set-continuation training."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from src.coord_tokens.codec import is_coord_token
from src.trainers.stage1_set_continuation.serialization import (
    build_candidate_entry_text,
    build_close_prefix_text,
    build_global_close_text,
    build_prefix_text,
    render_indexed_object_list,
)


_RAW_ONLY_KEYS = {
    "assistant_payload",
    "messages",
    "metadata",
    "objects",
    "sample_id",
    "base_idx",
    "dataset",
    "dataset_name",
    "length",
}


@dataclass(frozen=True)
class EncodedSetContinuationBranch:
    branch_inputs: dict[str, Any]
    labels: torch.Tensor
    candidate_entry_label_mask: torch.Tensor
    coord_label_mask: torch.Tensor
    non_coord_label_mask: torch.Tensor
    structural_close_start_mask: torch.Tensor
    structural_close_sequence_mask: torch.Tensor
    rendered_text: str
    prefix_text: str
    continuation_text: str
    prefix_indices: tuple[int, ...]
    candidate_index: int | None


def _assistant_objects(meta: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    payload = meta.get("assistant_payload")
    if not isinstance(payload, Mapping):
        raise ValueError("set-continuation metadata is missing assistant_payload")
    objects = payload.get("objects")
    if not isinstance(objects, Sequence) or isinstance(
        objects, (str, bytes, bytearray)
    ):
        raise ValueError(
            "set-continuation metadata is missing assistant_payload.objects"
        )
    resolved: list[Mapping[str, Any]] = []
    for index, obj in enumerate(objects):
        if not isinstance(obj, Mapping):
            raise ValueError(
                "set-continuation assistant_payload.objects entries must be mappings "
                f"(bad index={index})"
            )
        resolved.append(copy.deepcopy(obj))
    return resolved


def _messages_without_system(
    meta: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], str | None]:
    messages_raw = meta.get("messages")
    if not isinstance(messages_raw, Sequence) or isinstance(
        messages_raw, (str, bytes, bytearray)
    ):
        raise ValueError("set-continuation branch encoding requires messages")
    messages = [
        copy.deepcopy(dict(message))
        for message in messages_raw
        if isinstance(message, Mapping)
    ]
    if not messages:
        raise ValueError("set-continuation branch encoding requires non-empty messages")

    system_prompt = None
    if messages and messages[0].get("role") == "system":
        system_message = messages.pop(0)
        system_prompt = _message_text(system_message)
    if not any(message.get("role") == "assistant" for message in messages):
        raise ValueError(
            "set-continuation branch encoding requires an assistant message"
        )
    return messages, system_prompt


def _message_text(message: Mapping[str, Any]) -> str | None:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence) and not isinstance(
        content, (bytes, bytearray, str)
    ):
        texts = [
            item.get("text")
            for item in content
            if isinstance(item, Mapping) and item.get("type") == "text"
        ]
        text = "\n".join(str(item) for item in texts if item is not None)
        return text or None
    return None


def _with_assistant_text(
    messages: Sequence[Mapping[str, Any]], text: str
) -> list[dict[str, Any]]:
    out = [copy.deepcopy(dict(message)) for message in messages]
    assistant_indices = [
        index for index, message in enumerate(out) if message.get("role") == "assistant"
    ]
    if not assistant_indices:
        raise ValueError(
            "set-continuation branch encoding requires an assistant message"
        )
    assistant = out[assistant_indices[-1]]
    content = assistant.get("content")
    if isinstance(content, str):
        assistant["content"] = text
    elif isinstance(content, Sequence) and not isinstance(
        content, (str, bytes, bytearray)
    ):
        replaced = False
        new_content: list[Any] = []
        for item in content:
            if (
                not replaced
                and isinstance(item, Mapping)
                and item.get("type") == "text"
            ):
                new_item = dict(item)
                new_item["text"] = text
                new_content.append(new_item)
                replaced = True
            else:
                new_content.append(copy.deepcopy(item))
        if not replaced:
            new_content.append({"type": "text", "text": text})
        assistant["content"] = new_content
    else:
        assistant["content"] = [{"type": "text", "text": text}]
    return out


def _temporary_template_system(template: Any, system_prompt: str | None):
    class _TemplateSystemContext:
        def __enter__(self_nonlocal):
            self_nonlocal.had_system = hasattr(template, "system")
            self_nonlocal.original_system = (
                getattr(template, "system", None) if self_nonlocal.had_system else None
            )
            if system_prompt is not None:
                setattr(template, "system", system_prompt)

        def __exit__(self_nonlocal, exc_type, exc, tb):
            if system_prompt is not None:
                if self_nonlocal.had_system:
                    setattr(template, "system", self_nonlocal.original_system)
                elif hasattr(template, "system"):
                    delattr(template, "system")
            return False

    return _TemplateSystemContext()


def _encode(
    template: Any, rendered: Mapping[str, Any], system_prompt: str | None
) -> dict[str, Any]:
    if not hasattr(template, "encode"):
        raise TypeError("set-continuation branch encoder requires template.encode")
    with _temporary_template_system(template, system_prompt):
        encoded = template.encode(dict(rendered), return_length=True)
    if not isinstance(encoded, Mapping):
        raise TypeError("template.encode must return a mapping")
    return dict(encoded)


def _to_2d_long(value: Any, *, key: str) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.tensor(value)
    tensor = tensor.long()
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2:
        raise ValueError(f"Encoded branch {key} must be rank-1 or rank-2")
    return tensor


def _tensorize_branch_inputs(encoded: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in encoded.items():
        if key in _RAW_ONLY_KEYS:
            continue
        if key in {"input_ids", "labels", "attention_mask"}:
            out[key] = _to_2d_long(value, key=key)
        else:
            out[key] = value
    return out


def _encoded_len(
    *,
    template: Any,
    base_rendered: Mapping[str, Any],
    base_messages: Sequence[Mapping[str, Any]],
    system_prompt: str | None,
    assistant_text: str,
) -> int:
    rendered = dict(base_rendered)
    rendered["messages"] = _with_assistant_text(base_messages, assistant_text)
    encoded = _encode(template, rendered, system_prompt)
    if "input_ids" not in encoded:
        raise ValueError("template.encode output is missing input_ids")
    input_ids = _to_2d_long(encoded["input_ids"], key="input_ids")
    sequence = [int(value) for value in input_ids.reshape(-1).tolist()]

    tokenizer = getattr(template, "tokenizer", None)
    if tokenizer is None or not hasattr(tokenizer, "encode"):
        return int(input_ids.shape[-1])

    content_ids = [
        int(value)
        for value in tokenizer.encode(assistant_text, add_special_tokens=False)
    ]
    if not content_ids:
        return int(input_ids.shape[-1])

    content_len = len(content_ids)
    for start in range(len(sequence) - content_len, -1, -1):
        if sequence[start : start + content_len] == content_ids:
            return start + content_len

    raise ValueError(
        "Unable to locate assistant content tokens inside encoded branch; "
        "check chat-template/tokenizer compatibility for set-continuation spans"
    )


def _token_strings_for_labels(template: Any, labels: torch.Tensor) -> list[str | None]:
    tokenizer = getattr(template, "tokenizer", None)
    if tokenizer is None or not hasattr(tokenizer, "convert_ids_to_tokens"):
        return [None for _ in range(int(labels.numel()))]
    label_values = [int(value) for value in labels.reshape(-1).tolist()]
    visible_ids = [value if value >= 0 else 0 for value in label_values]
    try:
        tokens = tokenizer.convert_ids_to_tokens(visible_ids)
    except Exception:  # pragma: no cover - defensive tokenizer compatibility
        return [None for _ in label_values]
    return [
        None if label < 0 else str(token)
        for label, token in zip(label_values, list(tokens))
    ]


def _span_mask(length: int, start: int, end: int) -> torch.Tensor:
    start = max(0, min(int(start), int(length)))
    end = max(start, min(int(end), int(length)))
    mask = torch.zeros((1, int(length)), dtype=torch.bool)
    if end > start:
        mask[:, start:end] = True
    return mask


def encode_set_continuation_branch(
    *,
    meta: Mapping[str, Any],
    template: Any,
    prefix_indices: Sequence[int],
    candidate_index: int | None,
    object_field_order: str = "desc_first",
) -> EncodedSetContinuationBranch:
    """Encode one independent set-continuation branch."""

    assistant_objects = _assistant_objects(meta)
    rendered_objects = render_indexed_object_list(
        assistant_objects,
        object_field_order=object_field_order,
    )
    if candidate_index is None:
        prefix = build_close_prefix_text(rendered_objects, prefix_indices)
        continuation = build_global_close_text()
        continuation_text = continuation.text
        candidate_start = candidate_end = len(prefix.text)
        close_sequence_start = candidate_end
        close_start_text = prefix.text + continuation.text[:1]
        full_text = prefix.text + continuation.text
    else:
        prefix = build_prefix_text(rendered_objects, prefix_indices)
        candidate = build_candidate_entry_text(
            rendered_objects,
            prefix_indices=prefix_indices,
            candidate_index=int(candidate_index),
        )
        if "<|coord_" not in candidate.text:
            raise ValueError(
                "set-continuation candidate branch must contain coord-token geometry"
            )
        continuation_text = candidate.text
        candidate_start = len(prefix.text)
        close_sequence_start = candidate_start + candidate.candidate_span.end
        candidate_end = candidate_start + candidate.post_candidate_span.end
        close_start_text = (
            prefix.text + candidate.text[: candidate.global_close_start_span.end]
            if candidate.continuation_mode == "close"
            else ""
        )
        full_text = prefix.text + candidate.text

    base_messages, system_prompt = _messages_without_system(meta)
    branch_messages = _with_assistant_text(base_messages, full_text)
    rendered: dict[str, Any] = {
        "messages": branch_messages,
        "assistant_payload": {"objects": assistant_objects},
    }
    for key in ("objects", "metadata"):
        if key in meta:
            rendered[key] = copy.deepcopy(meta[key])

    encoded = _encode(template, rendered, system_prompt)
    branch_inputs = _tensorize_branch_inputs(encoded)
    if "input_ids" not in branch_inputs or "labels" not in branch_inputs:
        raise ValueError("Encoded branch must contain input_ids and labels")
    input_ids = branch_inputs["input_ids"]
    labels = branch_inputs["labels"]
    seq_len = int(input_ids.shape[-1])

    candidate_start_token = _encoded_len(
        template=template,
        base_rendered=rendered,
        base_messages=base_messages,
        system_prompt=system_prompt,
        assistant_text=full_text[:candidate_start],
    )
    candidate_end_token = _encoded_len(
        template=template,
        base_rendered=rendered,
        base_messages=base_messages,
        system_prompt=system_prompt,
        assistant_text=full_text[:candidate_end],
    )
    close_sequence_start_token = _encoded_len(
        template=template,
        base_rendered=rendered,
        base_messages=base_messages,
        system_prompt=system_prompt,
        assistant_text=full_text[:close_sequence_start],
    )
    close_sequence_end_token = _encoded_len(
        template=template,
        base_rendered=rendered,
        base_messages=base_messages,
        system_prompt=system_prompt,
        assistant_text=full_text,
    )
    if close_start_text:
        close_start_token = _encoded_len(
            template=template,
            base_rendered=rendered,
            base_messages=base_messages,
            system_prompt=system_prompt,
            assistant_text=close_start_text,
        )
    else:
        close_start_token = close_sequence_start_token

    candidate_mask = _span_mask(seq_len, candidate_start_token, candidate_end_token)
    close_sequence_mask = (
        _span_mask(seq_len, close_sequence_start_token, close_sequence_end_token)
        if close_start_text
        else torch.zeros((1, seq_len), dtype=torch.bool)
    )
    close_start_mask = _span_mask(
        seq_len, close_sequence_start_token, close_start_token
    )
    if not bool(close_start_mask.any().item()) and bool(
        close_sequence_mask.any().item()
    ):
        # Some tokenizers merge the first close character with the previous
        # token. Keep close-start supervision well-defined by using the first
        # available structural-close label token.
        first_close = close_sequence_mask.nonzero(as_tuple=False)[0]
        close_start_mask[int(first_close[0]), int(first_close[1])] = True

    coord_mask = torch.zeros_like(candidate_mask)
    token_strings = _token_strings_for_labels(template, labels)
    for index, token in enumerate(token_strings):
        if (
            candidate_mask.reshape(-1)[index]
            and token is not None
            and is_coord_token(token)
        ):
            coord_mask.reshape(-1)[index] = True
    non_coord_mask = candidate_mask & ~coord_mask

    return EncodedSetContinuationBranch(
        branch_inputs=branch_inputs,
        labels=labels,
        candidate_entry_label_mask=candidate_mask,
        coord_label_mask=coord_mask,
        non_coord_label_mask=non_coord_mask,
        structural_close_start_mask=close_start_mask,
        structural_close_sequence_mask=close_sequence_mask,
        rendered_text=full_text,
        prefix_text=prefix.text,
        continuation_text=continuation_text,
        prefix_indices=tuple(int(index) for index in prefix_indices),
        candidate_index=None if candidate_index is None else int(candidate_index),
    )


__all__ = [
    "EncodedSetContinuationBranch",
    "encode_set_continuation_branch",
]
