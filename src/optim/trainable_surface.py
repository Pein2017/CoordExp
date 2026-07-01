"""Trainable-surface receipts emitted before the first backward pass."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch import nn

from src.adapters.dora import DoraAdapterSetupReceipt
from src.common.errors import RuntimeContractError
from src.optim.parameter_groups import OptimizerGroupPlan
from src.qwen.special_token_embeddings import SpecialTokenEmbeddingInstallReceipt


TRAINABLE_SURFACE_PHASE = "before_first_backward"
V1_BASE_TOWERS = ("language", "vision", "aligner")
_PREVIEW_LIMIT = 20


@dataclass(frozen=True)
class FrozenReasonSummary:
    reason: str
    parameter_count: int
    scalar_count: int
    parameter_names_preview: tuple[str, ...]
    context: Mapping[str, Any]

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "parameter_count": self.parameter_count,
            "scalar_count": self.scalar_count,
            "parameter_names_preview": list(self.parameter_names_preview),
            "context": dict(self.context),
        }


@dataclass(frozen=True)
class TrainableSurfaceReceipt:
    phase: str
    frozen_towers: tuple[str, ...]
    trainable_towers: tuple[str, ...]
    adapter_targets: Mapping[str, Any]
    selected_embedding_tokens: Mapping[str, Any]
    parameter_counts: Mapping[str, Any]
    optimizer_groups: tuple[Mapping[str, Any], ...]
    unmatched_trainable_names: tuple[str, ...]
    frozen_reason_summaries: tuple[FrozenReasonSummary, ...]

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "frozen_towers": list(self.frozen_towers),
            "trainable_towers": list(self.trainable_towers),
            "adapter_targets": dict(self.adapter_targets),
            "selected_embedding_tokens": dict(self.selected_embedding_tokens),
            "parameter_counts": dict(self.parameter_counts),
            "optimizer_groups": [dict(group) for group in self.optimizer_groups],
            "unmatched_trainable_names": list(self.unmatched_trainable_names),
            "frozen_reason_summaries": [
                summary.to_artifact_dict()
                for summary in self.frozen_reason_summaries
            ],
        }


def build_trainable_surface_receipt(
    model: nn.Module,
    *,
    adapter_receipt: DoraAdapterSetupReceipt | None,
    special_token_receipt: SpecialTokenEmbeddingInstallReceipt | None,
    optimizer_group_plan: OptimizerGroupPlan,
    phase: str = TRAINABLE_SURFACE_PHASE,
) -> TrainableSurfaceReceipt:
    if phase != TRAINABLE_SURFACE_PHASE:
        raise RuntimeContractError(
            "trainable-surface receipt must be emitted before first backward",
            code="trainable_surface.phase_unsupported",
            context={
                "phase": phase,
                "expected_phase": TRAINABLE_SURFACE_PHASE,
            },
        )

    named_parameters = dict(model.named_parameters())
    trainable_parameters = {
        name: parameter
        for name, parameter in named_parameters.items()
        if parameter.requires_grad
    }
    frozen_parameters = {
        name: parameter
        for name, parameter in named_parameters.items()
        if not parameter.requires_grad
    }
    optimizer_parameter_names = _optimizer_parameter_names(optimizer_group_plan)
    _validate_optimizer_surface(
        actual_trainable_parameters=trainable_parameters,
        optimizer_parameter_names=optimizer_parameter_names,
        optimizer_group_plan=optimizer_group_plan,
    )

    optimizer_groups = tuple(
        group.to_artifact_dict()
        for group in optimizer_group_plan.groups
    )
    return TrainableSurfaceReceipt(
        phase=phase,
        frozen_towers=V1_BASE_TOWERS,
        trainable_towers=tuple(
            group.group_name
            for group in optimizer_group_plan.groups
            if group.parameter_names
        ),
        adapter_targets=_adapter_target_artifact(adapter_receipt),
        selected_embedding_tokens=_selected_embedding_artifact(special_token_receipt),
        parameter_counts=_parameter_counts(
            named_parameters=named_parameters,
            trainable_parameters=trainable_parameters,
            frozen_parameters=frozen_parameters,
            optimizer_group_plan=optimizer_group_plan,
        ),
        optimizer_groups=optimizer_groups,
        unmatched_trainable_names=tuple(optimizer_group_plan.unmatched_trainable_names),
        frozen_reason_summaries=_frozen_reason_summaries(
            frozen_parameters=frozen_parameters,
            special_token_receipt=special_token_receipt,
        ),
    )


def write_trainable_surface_receipt(
    receipt: TrainableSurfaceReceipt,
    path: Path,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            receipt.to_artifact_dict(),
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _optimizer_parameter_names(
    optimizer_group_plan: OptimizerGroupPlan,
) -> tuple[str, ...]:
    return tuple(
        parameter_name
        for group in optimizer_group_plan.groups
        for parameter_name in group.parameter_names
    )


def _validate_optimizer_surface(
    *,
    actual_trainable_parameters: Mapping[str, nn.Parameter],
    optimizer_parameter_names: tuple[str, ...],
    optimizer_group_plan: OptimizerGroupPlan,
) -> None:
    actual_trainable_names = set(actual_trainable_parameters)
    name_counts = Counter(optimizer_parameter_names)
    duplicate_names = sorted(name for name, count in name_counts.items() if count > 1)
    optimizer_name_set = set(optimizer_parameter_names)
    missing_from_optimizer = sorted(actual_trainable_names - optimizer_name_set)
    extra_in_optimizer = sorted(optimizer_name_set - actual_trainable_names)
    unmatched_from_plan = sorted(optimizer_group_plan.unmatched_trainable_names)
    missing_from_parameter_map = sorted(
        name
        for name in optimizer_name_set
        if name not in optimizer_group_plan.parameters_by_name
    )
    parameter_object_mismatch = sorted(
        name
        for name in optimizer_name_set & actual_trainable_names
        if optimizer_group_plan.parameters_by_name.get(name)
        is not actual_trainable_parameters[name]
    )
    if (
        duplicate_names
        or missing_from_optimizer
        or extra_in_optimizer
        or unmatched_from_plan
        or missing_from_parameter_map
        or parameter_object_mismatch
    ):
        raise RuntimeContractError(
            "optimizer group plan does not exactly cover the actual trainable surface",
            code="trainable_surface.optimizer_plan_mismatch",
            context={
                "duplicate_in_optimizer": duplicate_names,
                "missing_from_optimizer": missing_from_optimizer,
                "extra_in_optimizer": extra_in_optimizer,
                "unmatched_from_optimizer_plan": unmatched_from_plan,
                "missing_from_optimizer_parameter_map": missing_from_parameter_map,
                "parameter_object_mismatch": parameter_object_mismatch,
            },
        )


def _adapter_target_artifact(
    adapter_receipt: DoraAdapterSetupReceipt | None,
) -> dict[str, Any]:
    if adapter_receipt is None:
        return {
            "enabled": False,
            "adapter_type": None,
            "adapter_name": None,
            "matched_modules": [],
            "matched_count": 0,
        }
    artifact = adapter_receipt.target_discovery.to_artifact_dict()
    artifact.update(
        {
            "enabled": True,
            "adapter_type": adapter_receipt.adapter_type,
            "adapter_name": adapter_receipt.adapter_name,
            "trainable_counts": dict(adapter_receipt.trainable_counts),
        }
    )
    return artifact


def _selected_embedding_artifact(
    special_token_receipt: SpecialTokenEmbeddingInstallReceipt | None,
) -> dict[str, Any]:
    if special_token_receipt is None:
        return {
            "enabled": False,
            "selected_token_count": 0,
            "token_strings": [],
            "token_ids": [],
        }
    artifact = special_token_receipt.token_selection.to_artifact_dict()
    artifact.update(
        {
            "enabled": True,
            "semantics": special_token_receipt.semantics,
            "tensor_key": special_token_receipt.tensor_key,
            "delta_parameter_names": list(
                special_token_receipt.delta_parameter_names
            ),
            "delta_shape": list(special_token_receipt.delta_shape),
            "delta_dtype": special_token_receipt.delta_dtype,
        }
    )
    return artifact


def _parameter_counts(
    *,
    named_parameters: Mapping[str, nn.Parameter],
    trainable_parameters: Mapping[str, nn.Parameter],
    frozen_parameters: Mapping[str, nn.Parameter],
    optimizer_group_plan: OptimizerGroupPlan,
) -> dict[str, Any]:
    by_optimizer_group: dict[str, dict[str, int]] = {}
    for group in optimizer_group_plan.groups:
        group_parameters = [named_parameters[name] for name in group.parameter_names]
        by_optimizer_group[group.group_name] = {
            "parameter_count": len(group_parameters),
            "scalar_count": _scalar_count(group_parameters),
        }
    return {
        "total_parameter_count": len(named_parameters),
        "trainable_parameter_count": len(trainable_parameters),
        "frozen_parameter_count": len(frozen_parameters),
        "total_scalar_count": _scalar_count(named_parameters.values()),
        "trainable_scalar_count": _scalar_count(trainable_parameters.values()),
        "frozen_scalar_count": _scalar_count(frozen_parameters.values()),
        "by_optimizer_group": by_optimizer_group,
    }


def _frozen_reason_summaries(
    *,
    frozen_parameters: Mapping[str, nn.Parameter],
    special_token_receipt: SpecialTokenEmbeddingInstallReceipt | None,
) -> tuple[FrozenReasonSummary, ...]:
    summaries = [
        FrozenReasonSummary(
            reason="base_towers_frozen_v1",
            parameter_count=len(frozen_parameters),
            scalar_count=_scalar_count(frozen_parameters.values()),
            parameter_names_preview=tuple(sorted(frozen_parameters)[:_PREVIEW_LIMIT]),
            context={
                "base_finetuning_supported": False,
                "frozen_towers": list(V1_BASE_TOWERS),
            },
        )
    ]
    if special_token_receipt is not None:
        base_embedding_names = tuple(
            name
            for name in (
                special_token_receipt.base_embedding_parameter_name,
                special_token_receipt.base_lm_head_parameter_name,
            )
            if name is not None and name in frozen_parameters
        )
        summaries.append(
            FrozenReasonSummary(
                reason="selected_embedding_delta_active",
                parameter_count=len(base_embedding_names),
                scalar_count=_scalar_count(
                    frozen_parameters[name] for name in base_embedding_names
                ),
                parameter_names_preview=base_embedding_names,
                context={
                    "selected_delta_parameter_names": list(
                        special_token_receipt.delta_parameter_names
                    ),
                    "semantics": special_token_receipt.semantics,
                    "base_embedding_parameter_name": (
                        special_token_receipt.base_embedding_parameter_name
                    ),
                    "base_lm_head_parameter_name": (
                        special_token_receipt.base_lm_head_parameter_name
                    ),
                },
            )
        )
    return tuple(summaries)


def _scalar_count(parameters: Any) -> int:
    return sum(int(parameter.numel()) for parameter in parameters)
