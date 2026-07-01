"""Qwen forward boundary helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from src.common.errors import QwenForwardContractError
from src.packing.planner import PackedSequence
from src.qwen.fa2 import (
    Fa2VarlenBranchProof,
    Fa2VarlenPlan,
    build_fa2_varlen_plan,
    capture_fa2_varlen_branch,
    validate_fa2_varlen_branch_evidence,
    validate_fa2_varlen_plan_matches_pack,
)
from src.qwen.positions import QwenPositionInputs


_PROTECTED_FORWARD_OVERRIDE_KEYS = frozenset(
    {
        "input_ids",
        "position_ids",
        "pixel_values",
        "image_grid_thw",
        "attention_mask",
        "cu_seq_lens_q",
        "cu_seq_lens_k",
        "max_length_q",
        "max_length_k",
        "past_key_values",
        "cache_position",
    }
)


@dataclass(frozen=True)
class QwenForwardReceipt:
    pack_index: int
    pack_length: int
    segment_count: int
    input_ids_shape: tuple[int, ...]
    position_ids_shape: tuple[int, ...]
    position_row_meaning: tuple[str, ...]
    pixel_values_shape: tuple[int, ...]
    image_grid_thw: tuple[tuple[int, int, int], ...]
    placeholder_token_count: int
    expected_visual_token_count: int
    labels_passed: bool
    use_cache: bool
    logits_to_keep: int | tuple[int, ...]
    inputs_embeds_used: bool
    fa2_varlen_plan: Fa2VarlenPlan
    fa2_branch_proof: Fa2VarlenBranchProof | None = None
    output_logits_shape: tuple[int, ...] | None = None
    model_loss_present: bool = False
    model_loss_ignored: bool = False
    past_key_values_present: bool = False
    rope_deltas_present: bool = False

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "pack_index": self.pack_index,
            "pack_length": self.pack_length,
            "segment_count": self.segment_count,
            "input_ids_shape": list(self.input_ids_shape),
            "position_ids_shape": list(self.position_ids_shape),
            "position_row_meaning": list(self.position_row_meaning),
            "pixel_values_shape": list(self.pixel_values_shape),
            "image_grid_thw": [list(grid) for grid in self.image_grid_thw],
            "placeholder_token_count": self.placeholder_token_count,
            "expected_visual_token_count": self.expected_visual_token_count,
            "labels_passed": self.labels_passed,
            "use_cache": self.use_cache,
            "logits_to_keep": _logits_to_keep_artifact(self.logits_to_keep),
            "inputs_embeds_used": self.inputs_embeds_used,
            "fa2_varlen": {
                **self.fa2_varlen_plan.to_artifact_dict(),
                "proof": (
                    None
                    if self.fa2_branch_proof is None
                    else self.fa2_branch_proof.to_artifact_dict()
                ),
            },
            "output_logits_shape": (
                None if self.output_logits_shape is None else list(self.output_logits_shape)
            ),
            "model_loss_present": self.model_loss_present,
            "model_loss_ignored": self.model_loss_ignored,
            "past_key_values_present": self.past_key_values_present,
            "rope_deltas_present": self.rope_deltas_present,
        }


@dataclass(frozen=True)
class QwenForwardInputs:
    pack_index: int
    input_ids: torch.Tensor
    position_ids: torch.Tensor
    pixel_values: torch.Tensor
    image_grid_thw: torch.Tensor
    fa2_varlen_plan: Fa2VarlenPlan
    receipt: QwenForwardReceipt
    logits_to_keep: int | torch.Tensor = 0
    logits_position_ids: tuple[int, ...] | None = None

    @property
    def pack_length(self) -> int:
        return int(self.input_ids.shape[1])

    @property
    def expected_logits_length(self) -> int:
        if self.logits_position_ids is None:
            return self.pack_length
        return len(self.logits_position_ids)

    def to_model_kwargs(self) -> dict[str, Any]:
        return {
            "input_ids": self.input_ids,
            "position_ids": self.position_ids,
            "pixel_values": self.pixel_values,
            "image_grid_thw": self.image_grid_thw,
            "labels": None,
            "use_cache": False,
            "logits_to_keep": self.logits_to_keep,
            **self.fa2_varlen_plan.to_model_kwargs(),
        }

    def to_artifact_dict(self) -> dict[str, Any]:
        return self.receipt.to_artifact_dict()


@dataclass(frozen=True)
class QwenForwardResult:
    logits: torch.Tensor
    model_output: Any
    receipt: QwenForwardReceipt
    logits_position_ids: tuple[int, ...] | None = None


def build_qwen_forward_inputs(
    pack: PackedSequence,
    encoded_examples: Sequence[Any],
    position_inputs: QwenPositionInputs,
    *,
    fa2_varlen_plan: Fa2VarlenPlan | None = None,
    logits_to_keep_positions: Sequence[int] | None = None,
    device: torch.device | str | None = None,
) -> QwenForwardInputs:
    if not isinstance(pack, PackedSequence):
        raise QwenForwardContractError(
            "Qwen forward input construction requires a PackedSequence",
            code="qwen.forward_pack_type",
            context={"value_type": type(pack).__name__},
        )
    if not isinstance(position_inputs, QwenPositionInputs):
        raise QwenForwardContractError(
            "Qwen forward input construction requires QwenPositionInputs",
            code="qwen.forward_position_type",
            context={"value_type": type(position_inputs).__name__},
        )
    examples_by_id = _examples_by_id(encoded_examples)
    if position_inputs.pack_index != pack.pack_index or position_inputs.position_ids.shape != (
        4,
        1,
        pack.length,
    ):
        raise QwenForwardContractError(
            "Qwen position inputs must match the packed sequence",
            code="qwen.forward_position_shape",
            context={
                "pack_index": pack.pack_index,
                "position_pack_index": position_inputs.pack_index,
                "position_shape": [int(item) for item in position_inputs.position_ids.shape],
                "pack_length": pack.length,
            },
        )

    torch_device = torch.device("cpu") if device is None else torch.device(device)
    if fa2_varlen_plan is None:
        fa2_varlen_plan = build_fa2_varlen_plan(pack, device=torch_device)
    else:
        validate_fa2_varlen_plan_matches_pack(fa2_varlen_plan, pack)
    input_ids = torch.tensor([list(pack.input_ids)], dtype=torch.long, device=torch_device)
    position_ids = position_inputs.position_ids.to(device=torch_device, dtype=torch.long)

    pixel_values_parts: list[torch.Tensor] = []
    image_grids: list[tuple[int, int, int]] = []
    placeholder_token_count = 0
    expected_visual_token_count = 0
    for segment_summary in position_inputs.segments:
        example = examples_by_id.get(segment_summary.example_id)
        if example is None:
            raise QwenForwardContractError(
                "position segment references an encoded example that was not provided",
                code="qwen.forward_example_missing",
                context={
                    "pack_index": pack.pack_index,
                    "segment_index": segment_summary.segment_index,
                    "example_id": segment_summary.example_id,
                },
            )
        image_encoding = getattr(example, "image_encoding", None)
        image_grid = _image_grid_thw(image_encoding, example_id=segment_summary.example_id)
        if image_grid != segment_summary.image_grid_thw:
            raise QwenForwardContractError(
                "encoded image grid must match the position input grid",
                code="qwen.forward_grid_mismatch",
                context={
                    "example_id": segment_summary.example_id,
                    "position_grid": list(segment_summary.image_grid_thw),
                    "encoded_grid": list(image_grid),
                },
            )
        pixel_values = _pixel_values(image_encoding, example_id=segment_summary.example_id)
        pixel_values_parts.append(pixel_values.to(device=torch_device))
        image_grids.append(image_grid)
        placeholder_count = _count_token_id(
            pack.input_ids[segment_summary.start:segment_summary.end],
            segment_summary.image_token_id,
        )
        expected_visual_tokens = _merged_visual_tokens(
            image_grid,
            merge_size=segment_summary.merge_size,
        )
        if placeholder_count != expected_visual_tokens:
            raise QwenForwardContractError(
                "image placeholder count must match grid-derived visual-token count",
                code="qwen.forward_placeholder_grid_mismatch",
                context={
                    "example_id": segment_summary.example_id,
                    "image_token_id": segment_summary.image_token_id,
                    "placeholder_count": placeholder_count,
                    "expected_visual_tokens": expected_visual_tokens,
                    "image_grid_thw": list(image_grid),
                    "merge_size": segment_summary.merge_size,
                },
            )
        placeholder_token_count += placeholder_count
        expected_visual_token_count += expected_visual_tokens

    pixel_values = torch.cat(pixel_values_parts, dim=0)
    image_grid_thw = torch.tensor(image_grids, dtype=torch.long, device=torch_device)
    logits_to_keep, logits_position_ids = _resolve_logits_to_keep(
        logits_to_keep_positions,
        pack_length=pack.length,
        device=torch_device,
    )
    receipt = QwenForwardReceipt(
        pack_index=pack.pack_index,
        pack_length=pack.length,
        segment_count=len(pack.segments),
        input_ids_shape=tuple(int(item) for item in input_ids.shape),
        position_ids_shape=tuple(int(item) for item in position_ids.shape),
        position_row_meaning=position_inputs.row_meaning,
        pixel_values_shape=tuple(int(item) for item in pixel_values.shape),
        image_grid_thw=tuple(image_grids),
        placeholder_token_count=placeholder_token_count,
        expected_visual_token_count=expected_visual_token_count,
        labels_passed=False,
        use_cache=False,
        logits_to_keep=0 if logits_position_ids is None else logits_position_ids,
        inputs_embeds_used=False,
        fa2_varlen_plan=fa2_varlen_plan,
    )
    return QwenForwardInputs(
        pack_index=pack.pack_index,
        input_ids=input_ids,
        position_ids=position_ids,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        fa2_varlen_plan=fa2_varlen_plan,
        receipt=receipt,
        logits_to_keep=logits_to_keep,
        logits_position_ids=logits_position_ids,
    )


def run_qwen_forward(
    model: Any,
    forward_inputs: QwenForwardInputs,
    *,
    expected_vocab_size: int | None = None,
    extra_model_kwargs: Mapping[str, Any] | None = None,
    fa2_branch_evidence: Mapping[str, Any] | None = None,
    fa2_model_dtype: str | None = None,
    capture_fa2_branch: bool = False,
    require_fa2_branch_proof: bool = False,
) -> QwenForwardResult:
    if not isinstance(forward_inputs, QwenForwardInputs):
        raise QwenForwardContractError(
            "Qwen forward runner requires QwenForwardInputs",
            code="qwen.forward_inputs_type",
            context={"value_type": type(forward_inputs).__name__},
        )
    overrides = dict(extra_model_kwargs or {})
    _reject_unsafe_overrides(overrides)
    model_kwargs = forward_inputs.to_model_kwargs()
    model_kwargs.update(overrides)
    if capture_fa2_branch and fa2_branch_evidence is None:
        with capture_fa2_varlen_branch() as fa2_capture:
            output = model(**model_kwargs)
        fa2_branch_evidence = fa2_capture.evidence_for_plan(
            forward_inputs.fa2_varlen_plan
        )
    else:
        output = model(**model_kwargs)
    logits = getattr(output, "logits", None)
    if not isinstance(logits, torch.Tensor):
        raise QwenForwardContractError(
            "Qwen model output must expose tensor logits",
            code="qwen.forward_logits_missing",
            context={"output_type": type(output).__name__},
        )
    vocab_size = _expected_vocab_size(model, expected_vocab_size)
    _validate_logits_shape(
        logits,
        expected_logits_length=forward_inputs.expected_logits_length,
        expected_vocab_size=vocab_size,
    )
    fa2_branch_proof = None
    if fa2_branch_evidence is not None:
        fa2_branch_proof = validate_fa2_varlen_branch_evidence(
            forward_inputs.fa2_varlen_plan,
            fa2_branch_evidence,
            resolved_attention_implementation="flash_attention_2",
            model_dtype=fa2_model_dtype or _model_dtype(model),
        )
    elif require_fa2_branch_proof and forward_inputs.fa2_varlen_plan.branch_evidence_required:
        raise QwenForwardContractError(
            "required FA2 branch proof was not observed during Qwen forward",
            code="qwen.fa2_branch_evidence_missing",
            context={
                "pack_index": forward_inputs.pack_index,
                "segment_boundaries": list(
                    forward_inputs.fa2_varlen_plan.segment_boundaries
                ),
                "capture_fa2_branch": capture_fa2_branch,
            },
        )
    receipt = _receipt_with_output(
        forward_inputs.receipt,
        output,
        logits,
        fa2_branch_proof=fa2_branch_proof,
    )
    return QwenForwardResult(
        logits=logits,
        model_output=output,
        receipt=receipt,
        logits_position_ids=forward_inputs.logits_position_ids,
    )


def _reject_unsafe_overrides(overrides: Mapping[str, Any]) -> None:
    protected_keys = sorted(key for key in overrides if key in _PROTECTED_FORWARD_OVERRIDE_KEYS)
    if protected_keys:
        if protected_keys == ["attention_mask"] and overrides.get("attention_mask") is not None:
            raise QwenForwardContractError(
                "packed FA2 forward must not rely on an ordinary attention_mask",
                code="qwen.forward_attention_mask",
                context={"attention_mask_type": type(overrides.get("attention_mask")).__name__},
            )
        raise QwenForwardContractError(
            "V1 Qwen forward boundary tensors and cache state cannot be overridden",
            code="qwen.forward_boundary_override",
            context={"keys": protected_keys},
        )
    if overrides.get("inputs_embeds") is not None:
        raise QwenForwardContractError(
            "V1 Qwen forward must not pass inputs_embeds",
            code="qwen.forward_inputs_embeds",
            context={},
        )
    if overrides.get("labels") is not None:
        raise QwenForwardContractError(
            "V1 Qwen forward must not pass labels into the model",
            code="qwen.forward_labels",
            context={},
        )
    if "use_cache" in overrides and overrides["use_cache"] is not False:
        raise QwenForwardContractError(
            "V1 Qwen forward must run with use_cache=False",
            code="qwen.forward_use_cache",
            context={},
        )
    if "logits_to_keep" in overrides and int(overrides["logits_to_keep"]) != 0:
        raise QwenForwardContractError(
            "V1 Qwen forward owns logits_to_keep; pass supervised physical "
            "positions through QwenForwardInputs.logits_to_keep_positions instead",
            code="qwen.forward_logits_to_keep",
            context={"logits_to_keep": overrides["logits_to_keep"]},
        )
    if (
        overrides.get("pixel_values_videos") is not None
        or overrides.get("video_grid_thw") is not None
    ):
        raise QwenForwardContractError(
            "V1 Qwen forward does not accept video payloads",
            code="qwen.forward_video_payload",
            context={},
        )


def _resolve_logits_to_keep(
    logits_to_keep_positions: Sequence[int] | None,
    *,
    pack_length: int,
    device: torch.device,
) -> tuple[int | torch.Tensor, tuple[int, ...] | None]:
    if logits_to_keep_positions is None:
        return 0, None
    positions = tuple(int(position) for position in logits_to_keep_positions)
    if not positions:
        raise QwenForwardContractError(
            "explicit logits_to_keep positions must not be empty",
            code="qwen.forward_logits_to_keep_empty",
        )
    if len(set(positions)) != len(positions):
        raise QwenForwardContractError(
            "explicit logits_to_keep positions must be unique",
            code="qwen.forward_logits_to_keep_duplicate",
            context={"logits_to_keep": list(positions)},
        )
    out_of_bounds = [
        position for position in positions if position < 0 or position >= pack_length
    ]
    if out_of_bounds:
        raise QwenForwardContractError(
            "explicit logits_to_keep positions must stay inside pack length",
            code="qwen.forward_logits_to_keep_bounds",
            context={
                "pack_length": pack_length,
                "out_of_bounds": out_of_bounds,
                "logits_to_keep": list(positions),
            },
        )
    return torch.tensor(positions, dtype=torch.long, device=device), positions


def _logits_to_keep_artifact(value: int | tuple[int, ...]) -> int | list[int]:
    if isinstance(value, tuple):
        return list(value)
    return int(value)


def _examples_by_id(encoded_examples: Sequence[Any]) -> dict[str, Any]:
    examples: dict[str, Any] = {}
    for example in encoded_examples:
        example_id = getattr(example, "example_id", None)
        if not isinstance(example_id, str) or not example_id:
            raise QwenForwardContractError(
                "encoded example must expose a non-empty example_id",
                code="qwen.forward_example_id",
                context={"value_type": type(example_id).__name__},
            )
        if example_id in examples:
            raise QwenForwardContractError(
                "encoded example ids must be unique for Qwen forward input construction",
                code="qwen.forward_duplicate_example_id",
                context={"example_id": example_id},
            )
        examples[example_id] = example
    return examples


def _image_grid_thw(image_encoding: Any, *, example_id: str) -> tuple[int, int, int]:
    value = getattr(image_encoding, "image_grid_thw", None)
    if value is None:
        raise QwenForwardContractError(
            "encoded image payload must expose image_grid_thw",
            code="qwen.forward_image_grid",
            context={"example_id": example_id},
        )
    image_grid = tuple(int(item) for item in value)
    if len(image_grid) != 3:
        raise QwenForwardContractError(
            "encoded image_grid_thw must contain three values",
            code="qwen.forward_image_grid",
            context={"example_id": example_id, "image_grid_thw": list(image_grid)},
        )
    return image_grid


def _pixel_values(image_encoding: Any, *, example_id: str) -> torch.Tensor:
    pixel_values = getattr(image_encoding, "pixel_values", None)
    if not isinstance(pixel_values, torch.Tensor):
        raise QwenForwardContractError(
            "encoded image payload must expose tensor pixel_values",
            code="qwen.forward_pixel_values",
            context={"example_id": example_id, "value_type": type(pixel_values).__name__},
        )
    if pixel_values.ndim != 2:
        raise QwenForwardContractError(
            "encoded image pixel_values must have shape [patch_rows, width]",
            code="qwen.forward_pixel_values_shape",
            context={"example_id": example_id, "shape": [int(item) for item in pixel_values.shape]},
        )
    return pixel_values


def _count_token_id(token_ids: Sequence[int], token_id: int) -> int:
    return sum(1 for value in token_ids if int(value) == token_id)


def _merged_visual_tokens(image_grid: tuple[int, int, int], *, merge_size: int) -> int:
    t, h, w = image_grid
    merge_area = merge_size * merge_size
    raw_patch_rows = t * h * w
    if raw_patch_rows % merge_area != 0:
        raise QwenForwardContractError(
            "image grid must divide evenly by merge_size**2",
            code="qwen.forward_image_grid_merge",
            context={"image_grid_thw": list(image_grid), "merge_size": merge_size},
        )
    return raw_patch_rows // merge_area


def _expected_vocab_size(model: Any, expected_vocab_size: int | None) -> int:
    if expected_vocab_size is not None:
        return int(expected_vocab_size)
    text_config = getattr(getattr(model, "config", None), "text_config", None)
    vocab_size = getattr(text_config, "vocab_size", None)
    if vocab_size is None:
        vocab_size = getattr(getattr(model, "config", None), "vocab_size", None)
    if vocab_size is None:
        raise QwenForwardContractError(
            "expected vocab size must be supplied when model config does not expose it",
            code="qwen.forward_vocab_size",
            context={"model_type": type(model).__name__},
        )
    return int(vocab_size)


def _model_dtype(model: Any) -> str:
    dtype = getattr(model, "dtype", None)
    if dtype is not None:
        return str(dtype)
    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        try:
            first_parameter = next(parameters())
        except StopIteration:
            first_parameter = None
        if first_parameter is not None:
            return str(first_parameter.dtype)
    raise QwenForwardContractError(
        "FA2 branch proof requires model dtype evidence",
        code="qwen.forward_model_dtype",
        context={"model_type": type(model).__name__},
    )


def _validate_logits_shape(
    logits: torch.Tensor,
    *,
    expected_logits_length: int,
    expected_vocab_size: int,
) -> None:
    observed_shape = tuple(int(item) for item in logits.shape)
    expected_shape = (1, expected_logits_length, expected_vocab_size)
    if observed_shape != expected_shape:
        raise QwenForwardContractError(
            "Qwen forward must return logits with shape [1, selected_or_pack_length, vocab_size]",
            code="qwen.forward_logits_shape",
            context={
                "observed_shape": list(observed_shape),
                "expected_shape": list(expected_shape),
            },
        )


def _receipt_with_output(
    receipt: QwenForwardReceipt,
    output: Any,
    logits: torch.Tensor,
    *,
    fa2_branch_proof: Fa2VarlenBranchProof | None,
) -> QwenForwardReceipt:
    model_loss_present = getattr(output, "loss", None) is not None
    return QwenForwardReceipt(
        pack_index=receipt.pack_index,
        pack_length=receipt.pack_length,
        segment_count=receipt.segment_count,
        input_ids_shape=receipt.input_ids_shape,
        position_ids_shape=receipt.position_ids_shape,
        position_row_meaning=receipt.position_row_meaning,
        pixel_values_shape=receipt.pixel_values_shape,
        image_grid_thw=receipt.image_grid_thw,
        placeholder_token_count=receipt.placeholder_token_count,
        expected_visual_token_count=receipt.expected_visual_token_count,
        labels_passed=receipt.labels_passed,
        use_cache=receipt.use_cache,
        logits_to_keep=receipt.logits_to_keep,
        inputs_embeds_used=receipt.inputs_embeds_used,
        fa2_varlen_plan=receipt.fa2_varlen_plan,
        fa2_branch_proof=fa2_branch_proof,
        output_logits_shape=tuple(int(item) for item in logits.shape),
        model_loss_present=model_loss_present,
        model_loss_ignored=model_loss_present,
        past_key_values_present=getattr(output, "past_key_values", None) is not None,
        rope_deltas_present=getattr(output, "rope_deltas", None) is not None,
    )


__all__ = [
    "QwenForwardInputs",
    "QwenForwardReceipt",
    "QwenForwardResult",
    "build_qwen_forward_inputs",
    "run_qwen_forward",
]
