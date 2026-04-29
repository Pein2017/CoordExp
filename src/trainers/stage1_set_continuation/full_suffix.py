from __future__ import annotations

import copy
import hashlib
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from src.coord_tokens.codec import is_coord_token
from src.trainers.stage1_set_continuation.branch_encoder import (
    _assistant_objects,
    _encode,
    _encoded_len,
    _messages_without_system,
    _span_mask_for_text,
    _tensorize_branch_inputs,
    _with_assistant_text,
)
from src.trainers.stage1_set_continuation.branch_scorer import (
    _batch_model_inputs,
    _pad_2d,
)
from src.trainers.stage1_set_continuation.entry_trie import (
    EntryTrieCandidate,
    EntryTrieTarget,
    build_entry_trie_target_steps,
)
from src.trainers.stage1_set_continuation.serialization import (
    build_close_prefix_text,
    build_global_close_text,
    build_prefix_text,
    render_indexed_object_list,
)


@dataclass(frozen=True)
class FullSuffixTargetStep:
    row_index: int
    label_position: int
    teacher_token_id: int
    token_type: str
    phase: str
    targets: tuple[EntryTrieTarget | tuple[int, float], ...]
    active_object_count: int = 0

    @property
    def normalized_targets(self) -> tuple[EntryTrieTarget, ...]:
        resolved: list[EntryTrieTarget] = []
        for target in self.targets:
            if isinstance(target, EntryTrieTarget):
                resolved.append(target)
            else:
                token_id, probability = target
                resolved.append(
                    EntryTrieTarget(
                        token_id=int(token_id),
                        object_count=0,
                        probability=float(probability),
                    )
                )
        return tuple(resolved)

    @property
    def is_branch(self) -> bool:
        return self.phase == "entry" and len(self.normalized_targets) > 1


@dataclass(frozen=True)
class FullSuffixTensorInput:
    model_inputs: dict[str, torch.Tensor]
    labels: torch.Tensor
    steps: tuple[FullSuffixTargetStep, ...]


@dataclass(frozen=True)
class EncodedFullSuffixBranch:
    branch_inputs: dict[str, Any]
    labels: torch.Tensor
    steps: tuple[FullSuffixTargetStep, ...]
    rendered_text: str
    prefix_indices: tuple[int, ...]
    suffix_indices: tuple[int, ...]


@dataclass(frozen=True)
class FullSuffixLossResult:
    loss: torch.Tensor
    branch_loss: torch.Tensor
    unique_loss: torch.Tensor
    boundary_loss: torch.Tensor
    close_loss: torch.Tensor
    eos_loss: torch.Tensor
    branch_tokens: int
    unique_tokens: int
    boundary_tokens: int
    close_tokens: int
    eos_tokens: int
    metrics: dict[str, float]


def _zero(logits: torch.Tensor) -> torch.Tensor:
    return logits.new_zeros(())


def _normalize_token_type(value: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"text", "desc", "description"}:
        return "text"
    if normalized in {"coord", "coordinate"}:
        return "coord"
    if normalized in {"struct", "structure", "structural", "json"}:
        return "structural"
    return "other"


def _normalize_phase(value: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"entry", "boundary", "close", "eos"}:
        return normalized
    raise ValueError(f"unsupported full-suffix target phase: {value!r}")


def _stable_seed_from_parts(seed_parts: tuple[Any, ...]) -> int:
    digest = hashlib.sha256(repr(tuple(seed_parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def resolve_full_suffix_order(
    *,
    remaining_indices: Sequence[int],
    suffix_order: str,
    seed_parts: tuple[Any, ...],
) -> tuple[int, ...]:
    resolved = tuple(int(index) for index in remaining_indices)
    mode = str(suffix_order)
    if mode == "dataset":
        return resolved
    if mode != "random":
        raise ValueError(
            "full-suffix suffix_order must be one of {'random', 'dataset'}"
        )
    shuffled = list(resolved)
    random.Random(_stable_seed_from_parts(tuple(seed_parts))).shuffle(shuffled)
    return tuple(shuffled)


def _tokenize_text(tokenizer: Any, text: str) -> tuple[int, ...]:
    if hasattr(tokenizer, "encode_text"):
        return tuple(int(token) for token in tokenizer.encode_text(text))
    if hasattr(tokenizer, "encode"):
        return tuple(
            int(token)
            for token in tokenizer.encode(str(text), add_special_tokens=False)
        )
    if callable(tokenizer):
        tokenized = tokenizer(str(text), add_special_tokens=False)
        if isinstance(tokenized, Mapping) and "input_ids" in tokenized:
            return tuple(int(token) for token in tokenized["input_ids"])
    raise TypeError("entry-trie full-suffix encoding requires a text tokenizer")


def _tokens_to_strings(tokenizer: Any, token_ids: Sequence[int]) -> tuple[str, ...]:
    if hasattr(tokenizer, "convert_ids_to_tokens"):
        try:
            tokens = tokenizer.convert_ids_to_tokens(
                [int(token) for token in token_ids]
            )
            return tuple(str(token) for token in tokens)
        except Exception:  # pragma: no cover - defensive tokenizer compatibility
            pass
    return tuple(str(int(token)) for token in token_ids)


_STRUCTURAL_TOKEN_TEXT = {
    "{",
    "}",
    "[",
    "]",
    ":",
    ",",
    '"',
    "desc",
    "bbox",
    "bbox_2d",
    "poly",
}


def _classify_token_string(token: str) -> str:
    normalized = str(token).replace("Ġ", "").replace("▁", "").strip()
    if is_coord_token(normalized):
        return "coord"
    if not normalized:
        return "structural"
    if normalized in _STRUCTURAL_TOKEN_TEXT:
        return "structural"
    if all(char in '{}[]:," ' for char in normalized):
        return "structural"
    return "text"


def _entry_candidate_from_tokens(
    *,
    object_index: int,
    tokenizer: Any,
    tokens: Sequence[int],
) -> EntryTrieCandidate:
    resolved_tokens = tuple(int(token) for token in tokens)
    token_strings = _tokens_to_strings(tokenizer, resolved_tokens)
    return EntryTrieCandidate(
        object_index=int(object_index),
        tokens=resolved_tokens,
        token_types=tuple(_classify_token_string(token) for token in token_strings),
    )


def _target_step_with_position(
    *,
    source: FullSuffixTargetStep,
    row_index: int,
    label_position: int,
) -> FullSuffixTargetStep:
    return FullSuffixTargetStep(
        row_index=row_index,
        label_position=label_position,
        teacher_token_id=int(source.teacher_token_id),
        token_type=_normalize_token_type(source.token_type),
        phase=_normalize_phase(source.phase),
        targets=source.normalized_targets,
        active_object_count=int(source.active_object_count),
    )


def _shift_steps(
    steps: Sequence[FullSuffixTargetStep],
    *,
    row_index: int,
    label_offset: int,
) -> tuple[FullSuffixTargetStep, ...]:
    shifted: list[FullSuffixTargetStep] = []
    for step in steps:
        shifted.append(
            _target_step_with_position(
                source=step,
                row_index=row_index,
                label_position=int(step.label_position) - int(label_offset),
            )
        )
    return tuple(shifted)


def _supervised_suffix_start(
    labels: torch.Tensor, steps: Sequence[FullSuffixTargetStep]
) -> int:
    if labels.ndim != 2:
        raise ValueError("full-suffix labels must be rank-2")
    valid_positions = [
        int(step.label_position)
        for step in steps
        if int(step.label_position) > 0
        and int(step.label_position) < int(labels.shape[-1])
    ]
    if not valid_positions:
        return 0
    return max(0, min(valid_positions) - 1)


def supervised_full_suffix_start(
    labels: torch.Tensor, steps: Sequence[FullSuffixTargetStep]
) -> int:
    return _supervised_suffix_start(labels, steps)


def _label_positions_for_span(
    *,
    template: Any,
    base_rendered: Mapping[str, Any],
    base_messages: Sequence[Mapping[str, Any]],
    system_prompt: str | None,
    full_text: str,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    char_start: int,
    char_end: int,
) -> tuple[int, ...]:
    tokenizer = getattr(template, "tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "encode_text"):
        start_token = len(tokenizer.encode_text(full_text[: int(char_start)]))
        end_token = len(tokenizer.encode_text(full_text[: int(char_end)]))
        mask = torch.zeros_like(labels, dtype=torch.bool)
        if end_token > start_token:
            mask[:, start_token:end_token] = True
    else:
        mask = _span_mask_for_text(
            template=template,
            base_rendered=base_rendered,
            base_messages=base_messages,
            system_prompt=system_prompt,
            full_text=full_text,
            input_ids=input_ids,
            char_start=char_start,
            char_end=char_end,
        )
    positions = [
        int(index)
        for index in mask.reshape(-1).nonzero(as_tuple=False).reshape(-1).tolist()
        if int(index) > 0 and int(labels.reshape(-1)[int(index)].item()) != -100
    ]
    return tuple(positions)


def _add_hard_steps_for_span(
    *,
    steps: list[FullSuffixTargetStep],
    label_positions: Sequence[int],
    labels: torch.Tensor,
    phase: str,
    token_type: str,
    used_positions: set[int] | None = None,
) -> None:
    for label_position in label_positions:
        resolved_position = int(label_position)
        if used_positions is not None and resolved_position in used_positions:
            continue
        token_id = int(labels[0, int(label_position)].item())
        steps.append(
            FullSuffixTargetStep(
                row_index=0,
                label_position=resolved_position,
                teacher_token_id=token_id,
                token_type=token_type,
                phase=phase,
                targets=((token_id, 1.0),),
            )
        )
        if used_positions is not None:
            used_positions.add(resolved_position)


def _entry_candidate_for_context(
    *,
    meta: Mapping[str, Any],
    template: Any,
    base_messages: Sequence[Mapping[str, Any]],
    system_prompt: str | None,
    context_text: str,
    entry_text: str,
    boundary_text: str,
    object_index: int,
    tokenizer: Any,
) -> EntryTrieCandidate:
    candidate_text = context_text + entry_text + boundary_text
    candidate_messages = _with_assistant_text(base_messages, candidate_text)
    rendered: dict[str, Any] = {
        "messages": candidate_messages,
        "assistant_payload": copy.deepcopy(meta.get("assistant_payload", {})),
    }
    for key in ("objects", "metadata"):
        if key in meta:
            rendered[key] = copy.deepcopy(meta[key])
    encoded = _encode(template, rendered, system_prompt)
    branch_inputs = _tensorize_branch_inputs(encoded)
    if "input_ids" not in branch_inputs or "labels" not in branch_inputs:
        raise ValueError(
            "Encoded entry-trie candidate must contain input_ids and labels"
        )
    labels = branch_inputs["labels"]
    positions = _label_positions_for_span(
        template=template,
        base_rendered=rendered,
        base_messages=base_messages,
        system_prompt=system_prompt,
        full_text=candidate_text,
        input_ids=branch_inputs["input_ids"],
        labels=labels,
        char_start=len(context_text),
        char_end=len(context_text) + len(entry_text),
    )
    tokens = tuple(int(labels[0, int(position)].item()) for position in positions)
    if not tokens:
        raise ValueError("entry-trie context candidate produced no entry tokens")
    return _entry_candidate_from_tokens(
        object_index=int(object_index),
        tokenizer=tokenizer,
        tokens=tokens,
    )


def encode_full_suffix_branch(
    *,
    meta: Mapping[str, Any],
    template: Any,
    prefix_indices: Sequence[int],
    suffix_indices: Sequence[int],
    object_field_order: str = "desc_first",
) -> EncodedFullSuffixBranch:
    """Encode one recursive full-suffix row for ET-RMP-CE.

    The logical trie spans only serialized object-entry text. Candidate entries
    are tokenized in the current autoregressive context, including the following
    boundary text solely to match chat-template label boundaries. Boundary comma,
    global close, and chat-template end tokens are added separately as hard CE
    steps so they cannot contaminate MP branch labels.
    """

    assistant_objects = _assistant_objects(meta)
    rendered_objects = render_indexed_object_list(
        assistant_objects,
        object_field_order=object_field_order,
    )
    prefix_indices = tuple(int(index) for index in prefix_indices)
    suffix_indices = tuple(int(index) for index in suffix_indices)
    prefix_index_set = set(prefix_indices)
    suffix_index_set = set(suffix_indices)
    if prefix_index_set.intersection(suffix_index_set):
        raise ValueError("full-suffix prefix and suffix indices must be disjoint")
    if len(suffix_index_set) != len(suffix_indices):
        raise ValueError("full-suffix suffix_indices must not contain duplicates")
    if prefix_index_set.union(suffix_index_set) != set(range(len(assistant_objects))):
        raise ValueError(
            "full-suffix prefix+suffix must cover every object exactly once"
        )

    if suffix_indices:
        prefix = build_prefix_text(rendered_objects, prefix_indices)
    else:
        prefix = build_close_prefix_text(rendered_objects, prefix_indices)
    close_text = build_global_close_text().text
    pieces = [prefix.text]
    cursor = len(prefix.text)
    entry_spans: dict[int, tuple[int, int]] = {}
    boundary_spans: list[tuple[int, int]] = []
    for position, object_index in enumerate(suffix_indices):
        entry_text = rendered_objects.entry_texts_by_index[int(object_index)]
        start = cursor
        pieces.append(entry_text)
        cursor += len(entry_text)
        entry_spans[int(object_index)] = (start, cursor)
        if position < len(suffix_indices) - 1:
            boundary_start = cursor
            pieces.append(", ")
            cursor += 2
            boundary_spans.append((boundary_start, cursor))
    close_start = cursor
    pieces.append(close_text)
    cursor += len(close_text)
    close_span = (close_start, cursor)
    full_text = "".join(pieces)

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
        raise ValueError("Encoded full suffix must contain input_ids and labels")
    input_ids = branch_inputs["input_ids"]
    labels = branch_inputs["labels"]
    tokenizer = getattr(template, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("full-suffix entry trie requires template.tokenizer")

    steps: list[FullSuffixTargetStep] = []
    used_positions: set[int] = set()

    if not prefix_indices:
        opener_positions = _label_positions_for_span(
            template=template,
            base_rendered=rendered,
            base_messages=base_messages,
            system_prompt=system_prompt,
            full_text=full_text,
            input_ids=input_ids,
            labels=labels,
            char_start=0,
            char_end=len(prefix.text),
        )
        _add_hard_steps_for_span(
            steps=steps,
            label_positions=opener_positions,
            labels=labels,
            phase="boundary",
            token_type="structural",
            used_positions=used_positions,
        )

    remaining = set(int(index) for index in suffix_indices)
    for object_index in suffix_indices:
        span_start, span_end = entry_spans[int(object_index)]
        remaining_after_candidate = len(remaining) - 1
        boundary_text = ", " if remaining_after_candidate > 0 else close_text
        entry_candidates = {
            index: _entry_candidate_for_context(
                meta=meta,
                template=template,
                base_messages=base_messages,
                system_prompt=system_prompt,
                context_text=full_text[:span_start],
                entry_text=rendered_objects.entry_texts_by_index[index],
                boundary_text=boundary_text,
                object_index=index,
                tokenizer=tokenizer,
            )
            for index in remaining
        }
        label_positions = _label_positions_for_span(
            template=template,
            base_rendered=rendered,
            base_messages=base_messages,
            system_prompt=system_prompt,
            full_text=full_text,
            input_ids=input_ids,
            labels=labels,
            char_start=span_start,
            char_end=span_end,
        )
        teacher_candidate = entry_candidates[int(object_index)]
        teacher_label_tokens = tuple(
            int(labels[0, int(position)].item()) for position in label_positions
        )
        if teacher_label_tokens != teacher_candidate.tokens:
            raise ValueError(
                "full-suffix entry trie cannot align context-tokenized "
                "object-entry tokens with chat-template labels"
            )
        trie_steps = build_entry_trie_target_steps(
            candidates=tuple(entry_candidates.values()),
            teacher_tokens=teacher_candidate.tokens,
        )
        if len(trie_steps) != len(label_positions):
            raise ValueError("entry-trie target count must match object-entry labels")
        for trie_step, label_position in zip(trie_steps, label_positions, strict=True):
            steps.append(
                FullSuffixTargetStep(
                    row_index=0,
                    label_position=int(label_position),
                    teacher_token_id=int(trie_step.teacher_token_id),
                    token_type=trie_step.token_type,
                    phase="entry",
                    targets=trie_step.targets,
                    active_object_count=int(trie_step.active_object_count),
                )
            )
            used_positions.add(int(label_position))
        remaining.remove(int(object_index))

        suffix_position = suffix_indices.index(int(object_index))
        if suffix_position < len(boundary_spans):
            boundary_start, boundary_end = boundary_spans[suffix_position]
            boundary_positions = _label_positions_for_span(
                template=template,
                base_rendered=rendered,
                base_messages=base_messages,
                system_prompt=system_prompt,
                full_text=full_text,
                input_ids=input_ids,
                labels=labels,
                char_start=boundary_start,
                char_end=boundary_end,
            )
            _add_hard_steps_for_span(
                steps=steps,
                label_positions=boundary_positions,
                labels=labels,
                phase="boundary",
                token_type="structural",
                used_positions=used_positions,
            )

    close_positions = _label_positions_for_span(
        template=template,
        base_rendered=rendered,
        base_messages=base_messages,
        system_prompt=system_prompt,
        full_text=full_text,
        input_ids=input_ids,
        labels=labels,
        char_start=close_span[0],
        char_end=close_span[1],
    )
    _add_hard_steps_for_span(
        steps=steps,
        label_positions=close_positions,
        labels=labels,
        phase="close",
        token_type="structural",
        used_positions=used_positions,
    )

    content_end = _encoded_len(
        template=template,
        base_rendered=rendered,
        base_messages=base_messages,
        system_prompt=system_prompt,
        assistant_text=full_text,
    )
    eos_positions = tuple(
        index
        for index in range(int(content_end), int(labels.shape[-1]))
        if int(labels[0, index].item()) != -100
    )
    _add_hard_steps_for_span(
        steps=steps,
        label_positions=eos_positions,
        labels=labels,
        phase="eos",
        token_type="structural",
        used_positions=used_positions,
    )

    steps.sort(key=lambda step: int(step.label_position))
    return EncodedFullSuffixBranch(
        branch_inputs=branch_inputs,
        labels=labels,
        steps=tuple(steps),
        rendered_text=full_text,
        prefix_indices=prefix_indices,
        suffix_indices=suffix_indices,
    )


def _validate_step(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    step: FullSuffixTargetStep,
) -> None:
    if step.row_index < 0 or step.row_index >= int(logits.shape[0]):
        raise ValueError("full-suffix target row_index is out of range")
    if step.label_position <= 0 or step.label_position >= int(labels.shape[-1]):
        raise ValueError("full-suffix target label_position is out of range")
    teacher = int(labels[int(step.row_index), int(step.label_position)].item())
    if teacher != int(step.teacher_token_id):
        raise ValueError(
            "full-suffix target teacher_token_id does not match labels at "
            f"row={step.row_index} label_position={step.label_position}"
        )


def _step_nll(
    *,
    log_probs: torch.Tensor,
    step: FullSuffixTargetStep,
    entry_trie_mp: bool,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    targets = step.normalized_targets
    if not targets:
        raise ValueError("full-suffix target step requires at least one target")
    if step.phase == "entry" and len(targets) > 1 and entry_trie_mp:
        token_ids = torch.tensor(
            [int(target.token_id) for target in targets],
            dtype=torch.long,
            device=log_probs.device,
        )
        probabilities = torch.tensor(
            [float(target.probability) for target in targets],
            dtype=log_probs.dtype,
            device=log_probs.device,
        )
        nll = -(probabilities * log_probs.index_select(dim=-1, index=token_ids)).sum()
        valid_mass = torch.exp(log_probs.index_select(dim=-1, index=token_ids)).sum()
        return nll, valid_mass, True

    teacher_id = torch.tensor(
        int(step.teacher_token_id),
        dtype=torch.long,
        device=log_probs.device,
    )
    nll = -log_probs.index_select(dim=-1, index=teacher_id.view(1)).squeeze(0)
    valid_mass = torch.exp(
        log_probs.index_select(dim=-1, index=teacher_id.view(1))
    ).sum()
    return nll, valid_mass, False


def _mean(total: torch.Tensor, count: int) -> torch.Tensor:
    return total / max(int(count), 1)


def compute_full_suffix_loss(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    steps: Sequence[FullSuffixTargetStep],
    entry_trie_mp: bool = True,
) -> FullSuffixLossResult:
    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError("full-suffix logits must be rank-3 and labels rank-2")
    if int(logits.shape[0]) != int(labels.shape[0]):
        raise ValueError("full-suffix logits and labels batch dimensions differ")

    zero = _zero(logits)
    branch_sum = zero
    unique_sum = zero
    boundary_sum = zero
    close_sum = zero
    eos_sum = zero
    coord_branch_sum = zero
    text_branch_sum = zero
    branch_count = 0
    unique_count = 0
    boundary_count = 0
    close_count = 0
    eos_count = 0
    coord_branch_count = 0
    text_branch_count = 0
    branch_valid_mass_sum = zero
    branch_entropy_sum = 0.0
    branch_teacher_top1 = 0
    branch_valid_top1 = 0
    valid_child_total = 0
    branch_type_counts = {"text": 0, "coord": 0, "structural": 0, "other": 0}

    for raw_step in steps:
        step = _target_step_with_position(
            source=raw_step,
            row_index=int(raw_step.row_index),
            label_position=int(raw_step.label_position),
        )
        _validate_step(logits=logits, labels=labels, step=step)
        logit_index = int(step.label_position) - 1
        log_probs = torch.log_softmax(
            logits[int(step.row_index), logit_index].float(),
            dim=-1,
        ).to(dtype=logits.dtype)
        nll, valid_mass, is_branch = _step_nll(
            log_probs=log_probs,
            step=step,
            entry_trie_mp=entry_trie_mp,
        )

        if is_branch:
            branch_count += 1
            branch_sum = branch_sum + nll
            branch_valid_mass_sum = branch_valid_mass_sum + valid_mass
            branch_type = _normalize_token_type(step.token_type)
            branch_type_counts[branch_type] += 1
            entropy = -sum(
                target.probability
                * torch.log(
                    logits.new_tensor(float(target.probability)).clamp_min(1e-12)
                ).item()
                for target in step.normalized_targets
            )
            branch_entropy_sum += float(entropy)
            valid_ids = {int(target.token_id) for target in step.normalized_targets}
            top1 = int(torch.argmax(log_probs).detach().item())
            if top1 == int(step.teacher_token_id):
                branch_teacher_top1 += 1
            if top1 in valid_ids:
                branch_valid_top1 += 1
            valid_child_total += len(valid_ids)
            if branch_type == "coord":
                coord_branch_count += 1
                coord_branch_sum = coord_branch_sum + nll
            if branch_type == "text":
                text_branch_count += 1
                text_branch_sum = text_branch_sum + nll
            continue

        phase = _normalize_phase(step.phase)
        if phase == "entry":
            unique_count += 1
            unique_sum = unique_sum + nll
        elif phase == "boundary":
            boundary_count += 1
            boundary_sum = boundary_sum + nll
        elif phase == "close":
            close_count += 1
            close_sum = close_sum + nll
        elif phase == "eos":
            eos_count += 1
            eos_sum = eos_sum + nll

    branch_loss = _mean(branch_sum, branch_count)
    unique_loss = _mean(unique_sum, unique_count)
    boundary_loss = _mean(boundary_sum, boundary_count)
    close_loss = _mean(close_sum, close_count)
    eos_loss = _mean(eos_sum, eos_count)
    total_tokens = (
        branch_count + unique_count + boundary_count + close_count + eos_count
    )
    loss = (branch_sum + unique_sum + boundary_sum + close_sum + eos_sum) / max(
        total_tokens, 1
    )
    metrics = {
        "rmp/branch_nodes": float(branch_count),
        "rmp/branch_nodes_desc_text": float(branch_type_counts["text"]),
        "rmp/branch_nodes_coord": float(branch_type_counts["coord"]),
        "rmp/branch_nodes_structural": float(branch_type_counts["structural"]),
        "rmp/branch_nodes_other": float(branch_type_counts["other"]),
        "rmp/valid_children_mean": float(valid_child_total / max(branch_count, 1)),
        "rmp/target_entropy_mean": float(branch_entropy_sum / max(branch_count, 1)),
        "rmp/valid_child_mass_mean": float(
            (branch_valid_mass_sum / max(branch_count, 1)).detach().item()
        ),
        "rmp/teacher_branch_top1_acc": float(
            branch_teacher_top1 / max(branch_count, 1)
        ),
        "rmp/valid_child_top1_acc": float(branch_valid_top1 / max(branch_count, 1)),
        "loss/rmp": float(loss.detach().item()),
        "loss/rmp_branch_ce": float(branch_loss.detach().item()),
        "loss/rmp_unique_ce": float(unique_loss.detach().item()),
        "loss/rmp_coord_branch_ce": float(
            _mean(coord_branch_sum, coord_branch_count).detach().item()
        ),
        "loss/rmp_desc_text_branch_ce": float(
            _mean(text_branch_sum, text_branch_count).detach().item()
        ),
        "loss/rmp_boundary_ce": float(boundary_loss.detach().item()),
        "loss/rmp_close_ce": float(close_loss.detach().item()),
        "loss/rmp_eos_ce": float(eos_loss.detach().item()),
    }
    return FullSuffixLossResult(
        loss=loss,
        branch_loss=branch_loss,
        unique_loss=unique_loss,
        boundary_loss=boundary_loss,
        close_loss=close_loss,
        eos_loss=eos_loss,
        branch_tokens=branch_count,
        unique_tokens=unique_count,
        boundary_tokens=boundary_count,
        close_tokens=close_count,
        eos_tokens=eos_count,
        metrics=metrics,
    )


def build_recursive_entry_trie_steps(
    *,
    entry_candidates: Mapping[int, EntryTrieCandidate],
    suffix_order: Sequence[int],
    start_label_position: int,
    row_index: int = 0,
) -> tuple[FullSuffixTargetStep, ...]:
    remaining = {int(index): candidate for index, candidate in entry_candidates.items()}
    steps: list[FullSuffixTargetStep] = []
    label_position = int(start_label_position)
    for object_index in suffix_order:
        resolved_index = int(object_index)
        if resolved_index not in remaining:
            raise ValueError(f"suffix object {resolved_index} is not remaining")
        teacher_candidate = remaining[resolved_index]
        trie_steps = build_entry_trie_target_steps(
            candidates=tuple(remaining.values()),
            teacher_tokens=teacher_candidate.tokens,
        )
        for trie_step in trie_steps:
            steps.append(
                FullSuffixTargetStep(
                    row_index=int(row_index),
                    label_position=label_position,
                    teacher_token_id=int(trie_step.teacher_token_id),
                    token_type=trie_step.token_type,
                    phase="entry",
                    targets=trie_step.targets,
                    active_object_count=int(trie_step.active_object_count),
                )
            )
            label_position += 1
        del remaining[resolved_index]
    return tuple(steps)


def _coerce_item(
    raw: FullSuffixTensorInput | Mapping[str, Any],
) -> FullSuffixTensorInput:
    if isinstance(raw, FullSuffixTensorInput):
        return raw
    if not isinstance(raw, Mapping):
        raise TypeError(
            "full-suffix batch items must be mappings or FullSuffixTensorInput"
        )
    return FullSuffixTensorInput(
        model_inputs=dict(raw["model_inputs"]),
        labels=raw["labels"],
        steps=tuple(raw["steps"]),
    )


def _mask_from_steps(
    labels: torch.Tensor, steps: Sequence[FullSuffixTargetStep]
) -> torch.Tensor:
    mask = torch.zeros_like(labels, dtype=torch.bool)
    for step in steps:
        if 0 <= int(step.label_position) < int(labels.shape[-1]):
            mask[:, int(step.label_position)] = True
    return mask


def score_full_suffix_retained(
    *,
    model: Any,
    model_inputs: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
    steps: Sequence[FullSuffixTargetStep],
    logits_mode: str = "full",
    entry_trie_mp: bool = True,
    forward_fn: Any | None = None,
) -> FullSuffixLossResult:
    mode = str(logits_mode)
    suffix_start = (
        _supervised_suffix_start(labels, steps) if mode == "supervised_suffix" else 0
    )
    inputs = dict(model_inputs)
    if mode == "supervised_suffix":
        inputs["logits_to_keep"] = max(1, int(labels.shape[-1]) - int(suffix_start))
    outputs = forward_fn(**inputs) if forward_fn is not None else model(**inputs)
    cropped_labels = labels[:, suffix_start:]
    shifted_steps = _shift_steps(steps, row_index=0, label_offset=suffix_start)
    return compute_full_suffix_loss(
        logits=outputs.logits,
        labels=cropped_labels.to(device=outputs.logits.device),
        steps=shifted_steps,
        entry_trie_mp=entry_trie_mp,
    )


def score_full_suffix_batch_retained(
    *,
    model: Any,
    items: Sequence[FullSuffixTensorInput | Mapping[str, Any]],
    logits_mode: str = "full",
    entry_trie_mp: bool = True,
    forward_fn: Any | None = None,
) -> list[FullSuffixLossResult]:
    resolved = [_coerce_item(item) for item in items]
    if not resolved:
        return []
    max_length = max(int(item.labels.shape[-1]) for item in resolved)
    suffix_starts = [
        _supervised_suffix_start(item.labels, item.steps)
        if str(logits_mode) == "supervised_suffix"
        else 0
        for item in resolved
    ]
    global_suffix_start = (
        min(suffix_starts) if str(logits_mode) == "supervised_suffix" else 0
    )

    class _CompatItem:
        def __init__(self, item: FullSuffixTensorInput) -> None:
            self.model_inputs = item.model_inputs
            self.labels = item.labels

    batched_inputs = _batch_model_inputs(
        [_CompatItem(item) for item in resolved],
        max_length=max_length,
    )
    labels = torch.cat(
        [
            _pad_2d(item.labels, target_length=max_length, value=-100)
            for item in resolved
        ],
        dim=0,
    )
    if str(logits_mode) == "supervised_suffix":
        batched_inputs["logits_to_keep"] = max(1, max_length - int(global_suffix_start))
    outputs = (
        forward_fn(**batched_inputs)
        if forward_fn is not None
        else model(**batched_inputs)
    )
    cropped_labels = labels[:, global_suffix_start:]
    results: list[FullSuffixLossResult] = []
    for row_index, item in enumerate(resolved):
        row_steps = tuple(
            _target_step_with_position(
                source=step,
                row_index=row_index,
                label_position=int(step.label_position) - int(global_suffix_start),
            )
            for step in item.steps
        )
        results.append(
            compute_full_suffix_loss(
                logits=outputs.logits,
                labels=cropped_labels.to(device=outputs.logits.device),
                steps=row_steps,
                entry_trie_mp=entry_trie_mp,
            )
        )
    return results


__all__ = [
    "EncodedFullSuffixBranch",
    "FullSuffixLossResult",
    "FullSuffixTargetStep",
    "FullSuffixTensorInput",
    "build_recursive_entry_trie_steps",
    "compute_full_suffix_loss",
    "encode_full_suffix_branch",
    "resolve_full_suffix_order",
    "score_full_suffix_batch_retained",
    "score_full_suffix_retained",
    "supervised_full_suffix_start",
]
