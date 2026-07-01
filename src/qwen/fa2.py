"""FlashAttention 2 varlen segment-isolation contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from src.common.errors import QwenForwardContractError
from src.packing.planner import PackedSequence


PADDING_FREE_VARLEN_BRANCH = "padding_free_varlen"


@dataclass(frozen=True)
class Fa2VarlenPlan:
    segment_boundaries: tuple[int, ...]
    segment_lengths: tuple[int, ...]
    cu_seq_lens_q: torch.Tensor
    cu_seq_lens_k: torch.Tensor
    max_length_q: int
    max_length_k: int
    attention_mask: None
    branch_evidence_required: bool = True

    @property
    def segment_count(self) -> int:
        return len(self.segment_lengths)

    def to_model_kwargs(self) -> dict[str, Any]:
        return {
            "attention_mask": None,
            "cu_seq_lens_q": self.cu_seq_lens_q,
            "cu_seq_lens_k": self.cu_seq_lens_k,
            "max_length_q": self.max_length_q,
            "max_length_k": self.max_length_k,
        }

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "segment_count": self.segment_count,
            "segment_boundaries": list(self.segment_boundaries),
            "segment_lengths": list(self.segment_lengths),
            "cumulative_sequence_lengths": list(self.segment_boundaries),
            "cu_seq_lens_q": _tensor_int_list(self.cu_seq_lens_q),
            "cu_seq_lens_k": _tensor_int_list(self.cu_seq_lens_k),
            "max_length_q": self.max_length_q,
            "max_length_k": self.max_length_k,
            "attention_mask": None,
            "branch_evidence_required": self.branch_evidence_required,
        }


@dataclass(frozen=True)
class Fa2VarlenBranchProof:
    observed_branch: str
    segment_boundaries: tuple[int, ...]
    cu_seq_lens_q: tuple[int, ...]
    cu_seq_lens_k: tuple[int, ...]
    max_length_q: int
    max_length_k: int
    resolved_attention_implementation: str
    model_dtype: str
    branch_evidence_from_explicit_varlen_kwargs: bool
    flash_fn_called: bool
    flash_varlen_fn_called: bool
    pad_fn_called: bool
    unpad_fn_called: bool
    observed_call: Mapping[str, Any]

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "status": "pass",
            "observed_branch": self.observed_branch,
            "segment_boundaries": list(self.segment_boundaries),
            "cu_seq_lens_q": list(self.cu_seq_lens_q),
            "cu_seq_lens_k": list(self.cu_seq_lens_k),
            "max_length_q": self.max_length_q,
            "max_length_k": self.max_length_k,
            "resolved_attention_implementation": self.resolved_attention_implementation,
            "model_dtype": self.model_dtype,
            "branch_evidence_from_explicit_varlen_kwargs": (
                self.branch_evidence_from_explicit_varlen_kwargs
            ),
            "flash_fn_called": self.flash_fn_called,
            "flash_varlen_fn_called": self.flash_varlen_fn_called,
            "pad_fn_called": self.pad_fn_called,
            "unpad_fn_called": self.unpad_fn_called,
            "observed_call": _artifact_value(self.observed_call),
        }


def build_fa2_varlen_plan(
    pack: PackedSequence,
    *,
    attention_mask: Any = None,
    device: torch.device | str | None = None,
) -> Fa2VarlenPlan:
    if attention_mask is not None:
        raise QwenForwardContractError(
            "packed FA2 segment isolation must not rely on an ordinary attention_mask",
            code="qwen.fa2_attention_mask",
            context={"attention_mask_type": type(attention_mask).__name__},
        )
    if not isinstance(pack, PackedSequence):
        raise QwenForwardContractError(
            "FA2 varlen planning requires a PackedSequence",
            code="qwen.fa2_pack_type",
            context={"value_type": type(pack).__name__},
        )
    boundaries = _segment_boundaries(pack)
    lengths = tuple(
        boundaries[index + 1] - boundaries[index]
        for index in range(len(boundaries) - 1)
    )
    if not lengths:
        raise QwenForwardContractError(
            "FA2 varlen planning requires at least one packed segment",
            code="qwen.fa2_empty_pack",
            context={"pack_index": pack.pack_index},
        )
    torch_device = torch.device("cpu") if device is None else torch.device(device)
    cu_seq_lens = torch.tensor(boundaries, dtype=torch.int32, device=torch_device)
    max_length = max(lengths)
    return Fa2VarlenPlan(
        segment_boundaries=boundaries,
        segment_lengths=lengths,
        cu_seq_lens_q=cu_seq_lens,
        cu_seq_lens_k=cu_seq_lens.clone(),
        max_length_q=max_length,
        max_length_k=max_length,
        attention_mask=None,
    )


def validate_fa2_varlen_plan_matches_pack(
    plan: Fa2VarlenPlan,
    pack: PackedSequence,
) -> None:
    if not isinstance(plan, Fa2VarlenPlan):
        raise QwenForwardContractError(
            "FA2 varlen plan validation requires a Fa2VarlenPlan",
            code="qwen.fa2_plan_type",
            context={"value_type": type(plan).__name__},
        )
    if not isinstance(pack, PackedSequence):
        raise QwenForwardContractError(
            "FA2 varlen plan validation requires a PackedSequence",
            code="qwen.fa2_pack_type",
            context={"value_type": type(pack).__name__},
        )
    expected_boundaries = _segment_boundaries(pack)
    expected_lengths = tuple(
        expected_boundaries[index + 1] - expected_boundaries[index]
        for index in range(len(expected_boundaries) - 1)
    )
    if plan.segment_boundaries != expected_boundaries:
        raise QwenForwardContractError(
            "FA2 varlen plan boundaries must match packed segment boundaries",
            code="qwen.fa2_plan_boundaries",
            context={
                "expected_boundaries": list(expected_boundaries),
                "plan_boundaries": list(plan.segment_boundaries),
            },
        )
    if tuple(_tensor_int_list(plan.cu_seq_lens_q)) != expected_boundaries:
        raise QwenForwardContractError(
            "FA2 plan cu_seq_lens_q must match packed segment boundaries",
            code="qwen.fa2_cu_seq_lens",
            context={
                "expected_boundaries": list(expected_boundaries),
                "cu_seq_lens_q": _tensor_int_list(plan.cu_seq_lens_q),
            },
        )
    if tuple(_tensor_int_list(plan.cu_seq_lens_k)) != expected_boundaries:
        raise QwenForwardContractError(
            "FA2 plan cu_seq_lens_k must match packed segment boundaries",
            code="qwen.fa2_cu_seq_lens",
            context={
                "expected_boundaries": list(expected_boundaries),
                "cu_seq_lens_k": _tensor_int_list(plan.cu_seq_lens_k),
            },
        )
    expected_max_length = max(expected_lengths)
    if plan.max_length_q != expected_max_length or plan.max_length_k != expected_max_length:
        raise QwenForwardContractError(
            "FA2 plan max lengths must match packed segment lengths",
            code="qwen.fa2_max_length",
            context={
                "expected_max_length": expected_max_length,
                "max_length_q": plan.max_length_q,
                "max_length_k": plan.max_length_k,
            },
        )
    if plan.attention_mask is not None:
        raise QwenForwardContractError(
            "FA2 plan must not carry an ordinary attention_mask",
            code="qwen.fa2_attention_mask",
            context={"attention_mask": _artifact_value(plan.attention_mask)},
        )


def validate_fa2_varlen_branch_evidence(
    plan: Fa2VarlenPlan,
    evidence: Mapping[str, Any],
    *,
    resolved_attention_implementation: str,
    model_dtype: str,
) -> Fa2VarlenBranchProof:
    if not isinstance(plan, Fa2VarlenPlan):
        raise QwenForwardContractError(
            "FA2 branch evidence validation requires a Fa2VarlenPlan",
            code="qwen.fa2_plan_type",
            context={"value_type": type(plan).__name__},
        )
    if evidence.get("attention_mask") is not None:
        raise QwenForwardContractError(
            "FA2 padding-free varlen proof requires attention_mask=None",
            code="qwen.fa2_attention_mask",
            context={"attention_mask": _artifact_value(evidence.get("attention_mask"))},
        )
    if resolved_attention_implementation != "flash_attention_2":
        raise QwenForwardContractError(
            "FA2 branch proof requires flash_attention_2 implementation",
            code="qwen.fa2_attention_implementation",
            context={"resolved_attention_implementation": resolved_attention_implementation},
        )
    if model_dtype not in {"torch.bfloat16", "torch.float16", "bfloat16", "float16"}:
        raise QwenForwardContractError(
            "FA2 branch proof requires bf16 or fp16 model dtype",
            code="qwen.fa2_dtype",
            context={"model_dtype": model_dtype},
        )
    observed_branch = str(evidence.get("observed_branch", ""))
    flash_fn_called = bool(evidence.get("flash_fn_called", False))
    flash_varlen_fn_called = bool(evidence.get("flash_varlen_fn_called", False))
    pad_fn_called = bool(evidence.get("pad_fn_called", False))
    unpad_fn_called = bool(evidence.get("unpad_fn_called", False))
    if (
        observed_branch != PADDING_FREE_VARLEN_BRANCH
        or flash_fn_called
        or not flash_varlen_fn_called
        or pad_fn_called
        or unpad_fn_called
    ):
        raise QwenForwardContractError(
            "FA2 branch evidence must show the padding-free varlen branch only",
            code="qwen.fa2_branch",
            context={
                "observed_branch": observed_branch,
                "flash_fn_called": flash_fn_called,
                "flash_varlen_fn_called": flash_varlen_fn_called,
                "pad_fn_called": pad_fn_called,
                "unpad_fn_called": unpad_fn_called,
            },
        )
    if not bool(evidence.get("branch_evidence_from_explicit_varlen_kwargs", False)):
        raise QwenForwardContractError(
            "FA2 proof must come from explicit varlen kwargs, not only shapes",
            code="qwen.fa2_branch_evidence",
            context={},
        )

    cu_seq_lens_q = _int_tuple(evidence.get("cu_seq_lens_q"), code="qwen.fa2_cu_seq_lens")
    cu_seq_lens_k = _int_tuple(evidence.get("cu_seq_lens_k"), code="qwen.fa2_cu_seq_lens")
    if cu_seq_lens_q != plan.segment_boundaries or cu_seq_lens_k != plan.segment_boundaries:
        raise QwenForwardContractError(
            "FA2 cu_seq_lens must match packed segment boundaries",
            code="qwen.fa2_cu_seq_lens",
            context={
                "segment_boundaries": list(plan.segment_boundaries),
                "cu_seq_lens_q": list(cu_seq_lens_q),
                "cu_seq_lens_k": list(cu_seq_lens_k),
            },
        )
    max_length_q = _int_value(evidence.get("max_length_q"), code="qwen.fa2_max_length")
    max_length_k = _int_value(evidence.get("max_length_k"), code="qwen.fa2_max_length")
    if max_length_q != plan.max_length_q or max_length_k != plan.max_length_k:
        raise QwenForwardContractError(
            "FA2 max lengths must match packed segment lengths",
            code="qwen.fa2_max_length",
            context={
                "expected_max_length_q": plan.max_length_q,
                "expected_max_length_k": plan.max_length_k,
                "max_length_q": max_length_q,
                "max_length_k": max_length_k,
            },
        )
    observed_call = evidence.get("observed_call")
    if not isinstance(observed_call, Mapping):
        raise QwenForwardContractError(
            "FA2 branch proof requires observed varlen call kwargs",
            code="qwen.fa2_observed_call",
            context={"observed_call": _artifact_value(observed_call)},
        )
    _validate_observed_call_matches_plan(observed_call, plan)
    return Fa2VarlenBranchProof(
        observed_branch=observed_branch,
        segment_boundaries=plan.segment_boundaries,
        cu_seq_lens_q=cu_seq_lens_q,
        cu_seq_lens_k=cu_seq_lens_k,
        max_length_q=max_length_q,
        max_length_k=max_length_k,
        resolved_attention_implementation=resolved_attention_implementation,
        model_dtype=model_dtype,
        branch_evidence_from_explicit_varlen_kwargs=True,
        flash_fn_called=flash_fn_called,
        flash_varlen_fn_called=flash_varlen_fn_called,
        pad_fn_called=pad_fn_called,
        unpad_fn_called=unpad_fn_called,
        observed_call=observed_call,
    )


def _segment_boundaries(pack: PackedSequence) -> tuple[int, ...]:
    boundaries = [0]
    for segment in pack.segments:
        if segment.start != boundaries[-1] or segment.end <= segment.start:
            raise QwenForwardContractError(
                "PackedSegment boundaries must be contiguous for FA2 varlen planning",
                code="qwen.fa2_segment_boundaries",
                context={
                    "pack_index": pack.pack_index,
                    "segment_index": segment.segment_index,
                    "start": segment.start,
                    "end": segment.end,
                    "expected_start": boundaries[-1],
                },
            )
        boundaries.append(segment.end)
    if boundaries[-1] != pack.length:
        raise QwenForwardContractError(
            "FA2 segment boundaries must end at pack length",
            code="qwen.fa2_segment_boundaries",
            context={"pack_length": pack.length, "last_boundary": boundaries[-1]},
        )
    return tuple(boundaries)


def _validate_observed_call_matches_plan(
    observed_call: Mapping[str, Any],
    plan: Fa2VarlenPlan,
) -> None:
    observed_cu_seq_lens_q = _observed_call_int_tuple(
        observed_call,
        ("cu_seqlens_q", "cu_seq_lens_q"),
    )
    observed_cu_seq_lens_k = _observed_call_int_tuple(
        observed_call,
        ("cu_seqlens_k", "cu_seq_lens_k"),
    )
    observed_max_length_q = _observed_call_int_value(
        observed_call,
        ("max_seqlen_q", "max_length_q"),
    )
    observed_max_length_k = _observed_call_int_value(
        observed_call,
        ("max_seqlen_k", "max_length_k"),
    )
    if (
        observed_cu_seq_lens_q != plan.segment_boundaries
        or observed_cu_seq_lens_k != plan.segment_boundaries
        or observed_max_length_q != plan.max_length_q
        or observed_max_length_k != plan.max_length_k
    ):
        raise QwenForwardContractError(
            "observed FA2 varlen call kwargs must match the packed plan",
            code="qwen.fa2_observed_call",
            context={
                "segment_boundaries": list(plan.segment_boundaries),
                "observed_cu_seq_lens_q": list(observed_cu_seq_lens_q),
                "observed_cu_seq_lens_k": list(observed_cu_seq_lens_k),
                "expected_max_length_q": plan.max_length_q,
                "expected_max_length_k": plan.max_length_k,
                "observed_max_length_q": observed_max_length_q,
                "observed_max_length_k": observed_max_length_k,
            },
        )


def _observed_call_int_tuple(
    observed_call: Mapping[str, Any],
    keys: tuple[str, ...],
) -> tuple[int, ...]:
    value = _observed_call_value(observed_call, keys)
    return _int_tuple(value, code="qwen.fa2_observed_call")


def _observed_call_int_value(observed_call: Mapping[str, Any], keys: tuple[str, ...]) -> int:
    value = _observed_call_value(observed_call, keys)
    return _int_value(value, code="qwen.fa2_observed_call")


def _observed_call_value(observed_call: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in observed_call:
            return observed_call[key]
    raise QwenForwardContractError(
        "observed FA2 varlen call is missing required kwargs",
        code="qwen.fa2_observed_call",
        context={"missing_any_of": list(keys), "observed_keys": sorted(observed_call)},
    )


def _tensor_int_list(value: torch.Tensor) -> list[int]:
    return [int(item) for item in value.detach().cpu().tolist()]


def _int_tuple(value: Any, *, code: str) -> tuple[int, ...]:
    try:
        return tuple(int(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise QwenForwardContractError(
            "FA2 evidence value must be an integer sequence",
            code=code,
            context={"value": _artifact_value(value)},
            cause=exc,
        ) from exc


def _int_value(value: Any, *, code: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise QwenForwardContractError(
            "FA2 evidence value must be an integer",
            code=code,
            context={"value": _artifact_value(value)},
            cause=exc,
        ) from exc


def _artifact_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _artifact_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_artifact_value(item) for item in value]
    return value


__all__ = [
    "PADDING_FREE_VARLEN_BRANCH",
    "Fa2VarlenBranchProof",
    "Fa2VarlenPlan",
    "build_fa2_varlen_plan",
    "validate_fa2_varlen_branch_evidence",
    "validate_fa2_varlen_plan_matches_pack",
]
