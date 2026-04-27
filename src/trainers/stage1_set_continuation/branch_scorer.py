from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import torch
from torch.utils.checkpoint import checkpoint as _torch_checkpoint  # pyright: ignore[reportUnknownVariableType]

from src.trainers.teacher_forcing.forwards import (
    assert_unsliced_logits,
    prepare_forward_inputs,
    run_no_cache_forward,
)
from .branch_encoder import EncodedSetContinuationBranch
from .losses import CandidateLogProbResult, compute_candidate_full_entry_logprob

LogitsMode = Literal["full", "supervised_suffix"]


@dataclass(frozen=True)
class BranchScoreBundle:
    score: torch.Tensor
    logprob: CandidateLogProbResult
    outputs: Any
    branch_inputs: dict[str, Any]


@dataclass(frozen=True)
class TensorBranchScoreInput:
    model_inputs: dict[str, torch.Tensor]
    labels: torch.Tensor
    candidate_entry_label_mask: torch.Tensor
    coord_label_mask: torch.Tensor
    schema_open_label_mask: torch.Tensor | None = None
    json_structural_label_mask: torch.Tensor | None = None


def _validate_logits_mode(logits_mode: str) -> LogitsMode:
    if logits_mode not in {"full", "supervised_suffix"}:
        raise ValueError(
            "stage1 set-continuation logits mode must be one of "
            "{'full', 'supervised_suffix'}"
        )
    return cast(LogitsMode, logits_mode)


def supervised_suffix_start(
    *,
    labels: torch.Tensor,
    supervised_label_mask: torch.Tensor,
) -> int:
    """Return the first original logit position needed by shifted LM labels."""

    if labels.ndim != 2 or supervised_label_mask.ndim != 2:
        raise ValueError("labels and supervised_label_mask must be rank-2")
    if labels.shape != supervised_label_mask.shape:
        raise ValueError("labels and supervised_label_mask must have the same shape")
    if labels.shape[1] <= 1:
        return 0
    supervised_label_mask = supervised_label_mask.to(device=labels.device)
    valid = supervised_label_mask.bool() & labels.ne(-100)
    valid[:, 0] = False
    positions = valid.nonzero(as_tuple=False)
    if int(positions.shape[0]) == 0:
        return 0
    first_label_pos = int(positions[:, 1].min().detach().cpu().item())
    return max(0, first_label_pos - 1)


def crop_tensors_for_logits(
    *,
    suffix_start: int,
    labels: torch.Tensor,
    candidate_entry_label_mask: torch.Tensor,
    coord_label_mask: torch.Tensor,
    schema_open_label_mask: torch.Tensor | None = None,
    json_structural_label_mask: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    start = max(0, int(suffix_start))
    if start <= 0:
        return (
            labels,
            candidate_entry_label_mask,
            coord_label_mask,
            schema_open_label_mask,
            json_structural_label_mask,
        )
    return (
        labels[:, start:],
        candidate_entry_label_mask[:, start:],
        coord_label_mask[:, start:],
        schema_open_label_mask[:, start:]
        if schema_open_label_mask is not None
        else None,
        json_structural_label_mask[:, start:]
        if json_structural_label_mask is not None
        else None,
    )


def _suffix_start_for_mode(
    *,
    logits_mode: str,
    labels: torch.Tensor,
    supervised_label_mask: torch.Tensor,
) -> int:
    mode = _validate_logits_mode(logits_mode)
    if mode == "full":
        return 0
    return supervised_suffix_start(
        labels=labels,
        supervised_label_mask=supervised_label_mask,
    )


def _logits_to_keep_for_suffix(
    *,
    labels: torch.Tensor,
    suffix_start: int,
) -> int:
    return max(1, int(labels.shape[-1]) - max(0, int(suffix_start)))


def _with_optional_logits_to_keep(
    *,
    model_inputs: dict[str, Any],
    labels: torch.Tensor,
    suffix_start: int,
    logits_mode: str,
) -> dict[str, Any]:
    result = dict(model_inputs)
    if _validate_logits_mode(logits_mode) == "supervised_suffix":
        result["logits_to_keep"] = _logits_to_keep_for_suffix(
            labels=labels,
            suffix_start=suffix_start,
        )
    return result


def score_branch_retained_graph(
    *,
    trainer: Any,
    model: Any,
    branch: EncodedSetContinuationBranch,
    coord_token_ids: torch.Tensor,
    logits_mode: LogitsMode = "full",
) -> BranchScoreBundle:
    suffix_start = _suffix_start_for_mode(
        logits_mode=logits_mode,
        labels=branch.labels,
        supervised_label_mask=branch.candidate_entry_label_mask,
    )
    logits_to_keep = (
        _logits_to_keep_for_suffix(labels=branch.labels, suffix_start=suffix_start)
        if logits_mode == "supervised_suffix"
        else None
    )
    outputs, branch_inputs = trainer._forward_branch(
        model=model,
        branch=branch,
        logits_to_keep=logits_to_keep,
    )
    logits = outputs.logits
    labels = branch_inputs["labels"].to(device=logits.device)
    candidate_entry_label_mask = branch.candidate_entry_label_mask.to(
        device=logits.device
    )
    coord_label_mask = branch.coord_label_mask.to(device=logits.device)
    schema_open_label_mask = branch.schema_open_label_mask.to(device=logits.device)
    json_structural_label_mask = branch.json_structural_label_mask.to(
        device=logits.device
    )
    (
        labels,
        candidate_entry_label_mask,
        coord_label_mask,
        schema_open_label_mask,
        json_structural_label_mask,
    ) = crop_tensors_for_logits(
        suffix_start=suffix_start,
        labels=labels,
        candidate_entry_label_mask=candidate_entry_label_mask,
        coord_label_mask=coord_label_mask,
        schema_open_label_mask=schema_open_label_mask,
        json_structural_label_mask=json_structural_label_mask,
    )
    logprob = compute_candidate_full_entry_logprob(
        logits=logits,
        labels=labels,
        candidate_entry_label_mask=candidate_entry_label_mask,
        coord_label_mask=coord_label_mask,
        coord_token_ids=coord_token_ids.to(device=logits.device),
        schema_open_label_mask=schema_open_label_mask,
        json_structural_label_mask=json_structural_label_mask,
    )
    return BranchScoreBundle(
        score=logprob.score,
        logprob=logprob,
        outputs=outputs,
        branch_inputs=branch_inputs,
    )


def _result_from_logits(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    candidate_entry_label_mask: torch.Tensor,
    coord_label_mask: torch.Tensor,
    coord_token_ids: torch.Tensor,
    schema_open_label_mask: torch.Tensor | None = None,
    json_structural_label_mask: torch.Tensor | None = None,
    suffix_start: int = 0,
) -> CandidateLogProbResult:
    (
        labels,
        candidate_entry_label_mask,
        coord_label_mask,
        schema_open_label_mask,
        json_structural_label_mask,
    ) = crop_tensors_for_logits(
        suffix_start=suffix_start,
        labels=labels,
        candidate_entry_label_mask=candidate_entry_label_mask,
        coord_label_mask=coord_label_mask,
        schema_open_label_mask=schema_open_label_mask,
        json_structural_label_mask=json_structural_label_mask,
    )
    return compute_candidate_full_entry_logprob(
        logits=logits,
        labels=labels.to(device=logits.device),
        candidate_entry_label_mask=candidate_entry_label_mask.to(device=logits.device),
        coord_label_mask=coord_label_mask.to(device=logits.device),
        coord_token_ids=coord_token_ids.to(device=logits.device),
        schema_open_label_mask=schema_open_label_mask.to(device=logits.device)
        if schema_open_label_mask is not None
        else None,
        json_structural_label_mask=json_structural_label_mask.to(device=logits.device)
        if json_structural_label_mask is not None
        else None,
    )


def _token_counts(
    *,
    candidate_entry_label_mask: torch.Tensor,
    coord_label_mask: torch.Tensor,
    schema_open_label_mask: torch.Tensor | None = None,
    json_structural_label_mask: torch.Tensor | None = None,
) -> tuple[int, int, int, int, int]:
    candidate_mask = candidate_entry_label_mask.bool()
    coord_mask = coord_label_mask.bool()
    schema_mask = (
        schema_open_label_mask.bool() & candidate_mask
        if schema_open_label_mask is not None
        else torch.zeros_like(candidate_mask)
    )
    json_structural_mask = (
        json_structural_label_mask.bool() & candidate_mask
        if json_structural_label_mask is not None
        else torch.zeros_like(candidate_mask)
    )
    tokens = int(candidate_mask.sum().item())
    coord_tokens = int((candidate_mask & coord_mask).sum().item())
    return (
        tokens,
        coord_tokens,
        tokens - coord_tokens,
        int(schema_mask.sum().item()),
        int(json_structural_mask.sum().item()),
    )


def _pad_2d(
    tensor: torch.Tensor,
    *,
    target_length: int,
    value: int | bool,
) -> torch.Tensor:
    if tensor.ndim != 2 or int(tensor.shape[0]) != 1:
        raise ValueError("branch batch sequence tensors must have shape [1, seq_len]")
    pad = int(target_length) - int(tensor.shape[1])
    if pad <= 0:
        return tensor
    fill = tensor.new_full((1, pad), value)
    return torch.cat([tensor, fill], dim=1)


def _pad_3d_sequence(
    tensor: torch.Tensor,
    *,
    target_length: int,
) -> torch.Tensor:
    if tensor.ndim != 3 or int(tensor.shape[0]) != 1:
        raise ValueError(
            "branch batch embedded sequence tensors must have shape [1, seq_len, dim]"
        )
    pad = int(target_length) - int(tensor.shape[1])
    if pad <= 0:
        return tensor
    fill = tensor.new_zeros((1, pad, int(tensor.shape[2])))
    return torch.cat([tensor, fill], dim=1)


def _sequence_pad_value(key: str) -> int:
    if key == "attention_mask":
        return 0
    if key == "labels":
        return -100
    return 0


def _batch_model_inputs(
    items: Sequence[TensorBranchScoreInput],
    *,
    max_length: int,
) -> dict[str, Any]:
    if not items:
        return {}
    keys = set(items[0].model_inputs.keys())
    for item in items[1:]:
        if set(item.model_inputs.keys()) != keys:
            raise ValueError(
                "smart branch batching requires candidate branches to expose the "
                "same model input keys"
            )
    batched: dict[str, Any] = {}
    seq_lengths = [int(item.labels.shape[-1]) for item in items]
    cat_dim0_keys = {
        "image_grid_thw",
        "video_grid_thw",
        "pixel_values",
        "pixel_values_videos",
        "second_per_grid_ts",
    }
    for key in sorted(keys):
        values = [item.model_inputs[key] for item in items]
        if not all(isinstance(value, torch.Tensor) for value in values):
            batched[key] = values[0]
            continue
        tensors = cast(list[torch.Tensor], values)
        first = tensors[0]
        if (
            first.ndim == 2
            and int(first.shape[0]) == 1
            and all(
                tensor.ndim == 2
                and int(tensor.shape[0]) == 1
                and int(tensor.shape[1]) == seq_len
                for tensor, seq_len in zip(tensors, seq_lengths, strict=True)
            )
        ):
            pad_value = _sequence_pad_value(key)
            batched[key] = torch.cat(
                [
                    _pad_2d(
                        tensor,
                        target_length=max_length,
                        value=pad_value,
                    )
                    for tensor in tensors
                ],
                dim=0,
            )
            continue
        if (
            key == "inputs_embeds"
            and first.ndim == 3
            and int(first.shape[0]) == 1
            and all(
                tensor.ndim == 3
                and int(tensor.shape[0]) == 1
                and int(tensor.shape[1]) == seq_len
                for tensor, seq_len in zip(tensors, seq_lengths, strict=True)
            )
        ):
            batched[key] = torch.cat(
                [
                    _pad_3d_sequence(tensor, target_length=max_length)
                    for tensor in tensors
                ],
                dim=0,
            )
            continue
        if key in cat_dim0_keys:
            batched[key] = torch.cat(tensors, dim=0)
            continue
        if all(tuple(tensor.shape) == tuple(first.shape) for tensor in tensors):
            batched[key] = (
                torch.cat(tensors, dim=0) if int(first.shape[0]) == 1 else first
            )
            continue
        raise ValueError(
            "smart branch batching does not know how to batch model input "
            f"{key!r} with shapes {[tuple(tensor.shape) for tensor in tensors]}"
        )
    return batched


def _batch_labels_and_masks(
    items: Sequence[TensorBranchScoreInput],
    *,
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    labels = torch.cat(
        [_pad_2d(item.labels, target_length=max_length, value=-100) for item in items],
        dim=0,
    )
    candidate_masks = torch.cat(
        [
            _pad_2d(
                item.candidate_entry_label_mask.bool(),
                target_length=max_length,
                value=False,
            )
            for item in items
        ],
        dim=0,
    )
    coord_masks = torch.cat(
        [
            _pad_2d(
                item.coord_label_mask.bool(),
                target_length=max_length,
                value=False,
            )
            for item in items
        ],
        dim=0,
    )
    schema_masks = torch.cat(
        [
            _pad_2d(
                (
                    item.schema_open_label_mask
                    if item.schema_open_label_mask is not None
                    else torch.zeros_like(item.candidate_entry_label_mask)
                ).bool(),
                target_length=max_length,
                value=False,
            )
            for item in items
        ],
        dim=0,
    )
    json_structural_masks = torch.cat(
        [
            _pad_2d(
                (
                    item.json_structural_label_mask
                    if item.json_structural_label_mask is not None
                    else torch.zeros_like(item.candidate_entry_label_mask)
                ).bool(),
                target_length=max_length,
                value=False,
            )
            for item in items
        ],
        dim=0,
    )
    return labels, candidate_masks, coord_masks, schema_masks, json_structural_masks


def score_tensor_batch_retained(
    *,
    model: Any,
    items: Sequence[TensorBranchScoreInput],
    coord_token_ids: torch.Tensor,
    logits_mode: LogitsMode = "full",
    forward_fn: Callable[..., Any] | None = None,
) -> list[CandidateLogProbResult]:
    if not items:
        return []
    mode = _validate_logits_mode(logits_mode)
    max_length = max(int(item.labels.shape[-1]) for item in items)
    suffix_starts = [
        _suffix_start_for_mode(
            logits_mode=mode,
            labels=item.labels,
            supervised_label_mask=item.candidate_entry_label_mask,
        )
        for item in items
    ]
    global_suffix_start = min(suffix_starts) if mode == "supervised_suffix" else 0
    batched_inputs = _batch_model_inputs(items, max_length=max_length)
    labels, candidate_masks, coord_masks, schema_masks, json_structural_masks = (
        _batch_labels_and_masks(
            items,
            max_length=max_length,
        )
    )
    if mode == "supervised_suffix":
        batched_inputs["logits_to_keep"] = _logits_to_keep_for_suffix(
            labels=labels,
            suffix_start=global_suffix_start,
        )
    outputs = (
        forward_fn(**batched_inputs)
        if forward_fn is not None
        else model(**batched_inputs)
    )
    logits = outputs.logits
    expected_length = (
        max_length - global_suffix_start if mode == "supervised_suffix" else max_length
    )
    if int(logits.shape[0]) != len(items) or int(logits.shape[1]) != int(
        expected_length
    ):
        raise ValueError(
            "smart branch batching model forward returned logits with shape "
            f"{tuple(logits.shape)}, expected batch={len(items)} "
            f"and seq={int(expected_length)}"
        )
    return [
        _result_from_logits(
            logits=logits[row_index : row_index + 1],
            labels=labels[row_index : row_index + 1],
            candidate_entry_label_mask=candidate_masks[row_index : row_index + 1],
            coord_label_mask=coord_masks[row_index : row_index + 1],
            coord_token_ids=coord_token_ids,
            schema_open_label_mask=schema_masks[row_index : row_index + 1],
            json_structural_label_mask=json_structural_masks[row_index : row_index + 1],
            suffix_start=global_suffix_start,
        )
        for row_index in range(len(items))
    ]


def score_tensor_retained(
    *,
    model: Any,
    model_inputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    candidate_entry_label_mask: torch.Tensor,
    coord_label_mask: torch.Tensor,
    coord_token_ids: torch.Tensor,
    schema_open_label_mask: torch.Tensor | None = None,
    json_structural_label_mask: torch.Tensor | None = None,
    logits_mode: LogitsMode = "full",
) -> CandidateLogProbResult:
    suffix_start = _suffix_start_for_mode(
        logits_mode=logits_mode,
        labels=labels,
        supervised_label_mask=candidate_entry_label_mask,
    )
    prepared_inputs = _with_optional_logits_to_keep(
        model_inputs=model_inputs,
        labels=labels,
        suffix_start=suffix_start,
        logits_mode=logits_mode,
    )
    outputs = model(**prepared_inputs)
    return _result_from_logits(
        logits=outputs.logits,
        labels=labels,
        candidate_entry_label_mask=candidate_entry_label_mask,
        coord_label_mask=coord_label_mask,
        coord_token_ids=coord_token_ids,
        schema_open_label_mask=schema_open_label_mask,
        json_structural_label_mask=json_structural_label_mask,
        suffix_start=suffix_start,
    )


def score_tensor_checkpointed(
    *,
    model: Any,
    model_inputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    candidate_entry_label_mask: torch.Tensor,
    coord_label_mask: torch.Tensor,
    coord_token_ids: torch.Tensor,
    use_reentrant: bool,
    preserve_rng_state: bool,
    forward_fn: Callable[..., Any] | None = None,
    logits_mode: LogitsMode = "full",
    schema_open_label_mask: torch.Tensor | None = None,
    json_structural_label_mask: torch.Tensor | None = None,
) -> CandidateLogProbResult:
    suffix_start = _suffix_start_for_mode(
        logits_mode=logits_mode,
        labels=labels,
        supervised_label_mask=candidate_entry_label_mask,
    )
    prepared_inputs = _with_optional_logits_to_keep(
        model_inputs=dict(model_inputs),
        labels=labels,
        suffix_start=suffix_start,
        logits_mode=logits_mode,
    )
    tensor_input_names = tuple(
        name
        for name, value in prepared_inputs.items()
        if isinstance(value, torch.Tensor)
    )
    input_values = tuple(prepared_inputs[name] for name in tensor_input_names)
    static_inputs = {
        name: value
        for name, value in prepared_inputs.items()
        if not isinstance(value, torch.Tensor)
    }

    def _forward_scores(
        *flat_inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        kwargs = dict(static_inputs)
        kwargs.update(dict(zip(tensor_input_names, flat_inputs, strict=True)))
        outputs = forward_fn(**kwargs) if forward_fn is not None else model(**kwargs)
        logits = outputs.logits
        input_ids = kwargs.get("input_ids")
        if isinstance(input_ids, torch.Tensor) and logits_mode == "full":
            assert_unsliced_logits(
                logits=logits,
                input_ids=input_ids,
                where="stage1-set-continuation checkpointed training",
            )
        result = _result_from_logits(
            logits=logits,
            labels=labels,
            candidate_entry_label_mask=candidate_entry_label_mask,
            coord_label_mask=coord_label_mask,
            coord_token_ids=coord_token_ids,
            schema_open_label_mask=schema_open_label_mask,
            json_structural_label_mask=json_structural_label_mask,
            suffix_start=suffix_start,
        )
        return (
            result.score,
            result.coord_score,
            result.non_coord_score,
            result.schema_score,
            result.json_structural_score,
        )

    score, coord_score, non_coord_score, schema_score, json_structural_score = cast(
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        _torch_checkpoint(
            _forward_scores,
            *input_values,
            use_reentrant=use_reentrant,
            preserve_rng_state=preserve_rng_state,
        ),
    )
    (
        tokens,
        coord_tokens,
        non_coord_tokens,
        schema_tokens,
        json_structural_tokens,
    ) = _token_counts(
        candidate_entry_label_mask=candidate_entry_label_mask,
        coord_label_mask=coord_label_mask,
        schema_open_label_mask=schema_open_label_mask,
        json_structural_label_mask=json_structural_label_mask,
    )
    return CandidateLogProbResult(
        score=score,
        coord_score=coord_score,
        non_coord_score=non_coord_score,
        schema_score=schema_score,
        json_structural_score=json_structural_score,
        tokens=tokens,
        coord_tokens=coord_tokens,
        non_coord_tokens=non_coord_tokens,
        schema_tokens=schema_tokens,
        json_structural_tokens=json_structural_tokens,
    )


def score_branch_checkpointed_exact(
    *,
    trainer: Any,
    model: Any,
    branch: EncodedSetContinuationBranch,
    coord_token_ids: torch.Tensor,
    use_reentrant: bool,
    preserve_rng_state: bool,
    logits_mode: LogitsMode = "full",
) -> BranchScoreBundle:
    branch_inputs = trainer._prepare_branch_inputs(model=model, branch=branch)
    _core_model, inputs_for_model, _model_type = prepare_forward_inputs(
        model=model,
        inputs=branch_inputs,
        ignored_keys=(
            "labels",
            "compute_loss_func",
            "loss_scale",
            "text_position_ids",
            "channel",
            "logits_to_keep",
        ),
        packing_enabled=False,
        where="stage1-set-continuation-checkpointed",
    )
    tensor_inputs = {
        key: value
        for key, value in inputs_for_model.items()
        if isinstance(value, torch.Tensor)
    }

    def _no_cache_forward(**kwargs: torch.Tensor) -> Any:
        return run_no_cache_forward(
            model=model,
            inputs_for_model=dict(kwargs),
        )

    logprob = score_tensor_checkpointed(
        model=model,
        model_inputs=tensor_inputs,
        labels=branch_inputs["labels"],
        candidate_entry_label_mask=branch.candidate_entry_label_mask,
        coord_label_mask=branch.coord_label_mask,
        coord_token_ids=coord_token_ids,
        schema_open_label_mask=branch.schema_open_label_mask,
        json_structural_label_mask=branch.json_structural_label_mask,
        use_reentrant=use_reentrant,
        preserve_rng_state=preserve_rng_state,
        forward_fn=_no_cache_forward,
        logits_mode=logits_mode,
    )
    return BranchScoreBundle(
        score=logprob.score,
        logprob=logprob,
        outputs=None,
        branch_inputs=branch_inputs,
    )


__all__ = [
    "BranchScoreBundle",
    "TensorBranchScoreInput",
    "crop_tensors_for_logits",
    "score_branch_checkpointed_exact",
    "score_branch_retained_graph",
    "score_tensor_batch_retained",
    "score_tensor_checkpointed",
    "score_tensor_retained",
    "supervised_suffix_start",
]
