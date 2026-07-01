"""Explicit optimizer parameter grouping for CoordExp-swift."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import nn

from src.adapters.dora import DoraAdapterSetupReceipt
from src.common.errors import RuntimeContractError
from src.config.models import OptimizerConfig, OptimizerGroupConfig
from src.qwen.special_token_embeddings import SpecialTokenEmbeddingInstallReceipt


ADAPTER_GROUP_PREFIX = "adapter."
TOKEN_EMBEDDINGS_GROUP = "token_embeddings"


@dataclass(frozen=True)
class OptimizerGroupAssignment:
    group_name: str
    lr: float
    weight_decay: float
    parameter_names: tuple[str, ...]

    @property
    def parameter_count(self) -> int:
        return len(self.parameter_names)

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "group_name": self.group_name,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "parameter_count": self.parameter_count,
            "parameter_names": list(self.parameter_names),
        }


@dataclass(frozen=True)
class OptimizerGroupPlan:
    groups: tuple[OptimizerGroupAssignment, ...]
    parameters_by_name: dict[str, nn.Parameter]
    unmatched_trainable_names: tuple[str, ...] = ()

    def to_torch_param_groups(self) -> list[dict[str, Any]]:
        param_groups: list[dict[str, Any]] = []
        for group in self.groups:
            param_groups.append(
                {
                    "name": group.group_name,
                    "lr": group.lr,
                    "weight_decay": group.weight_decay,
                    "params": [
                        self.parameters_by_name[name]
                        for name in group.parameter_names
                    ],
                }
            )
        return param_groups

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "groups": [group.to_artifact_dict() for group in self.groups],
            "unmatched_trainable_names": list(self.unmatched_trainable_names),
            "trainable_parameter_count": sum(group.parameter_count for group in self.groups),
        }


def build_optimizer_group_plan(
    model: nn.Module,
    optimizer_config: OptimizerConfig,
    *,
    adapter_receipt: DoraAdapterSetupReceipt | None,
    special_token_receipt: SpecialTokenEmbeddingInstallReceipt | None,
) -> OptimizerGroupPlan:
    parameters_by_name = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    group_names_by_parameter: dict[str, list[str]] = {
        name: [] for name in parameters_by_name
    }

    if adapter_receipt is not None:
        _match_adapter_parameters(
            group_names_by_parameter,
            adapter_receipt=adapter_receipt,
        )
    if special_token_receipt is not None:
        _match_token_embedding_parameters(
            group_names_by_parameter,
            special_token_receipt=special_token_receipt,
        )

    for parameter_name, group_names in group_names_by_parameter.items():
        if len(group_names) > 1:
            raise RuntimeContractError(
                "trainable parameter matched multiple optimizer groups",
                code="optimizer.trainable_duplicate_match",
                context={
                    "parameter_name": parameter_name,
                    "matched_groups": group_names,
                },
            )

    unmatched = tuple(
        name
        for name, group_names in group_names_by_parameter.items()
        if not group_names
    )
    if unmatched:
        raise RuntimeContractError(
            "trainable parameter did not match an explicit optimizer group",
            code="optimizer.trainable_unmatched",
            context={"unmatched_trainable_names": list(unmatched)},
        )

    assignments: list[OptimizerGroupAssignment] = []
    for group_name in (
        "adapter.language",
        "adapter.vision",
        "adapter.aligner",
        TOKEN_EMBEDDINGS_GROUP,
    ):
        parameter_names = tuple(
            name
            for name, matched_groups in group_names_by_parameter.items()
            if matched_groups == [group_name]
        )
        if not parameter_names:
            continue
        group_config = _optimizer_group_config(optimizer_config, group_name)
        assignments.append(
            OptimizerGroupAssignment(
                group_name=group_name,
                lr=group_config.lr,
                weight_decay=group_config.weight_decay,
                parameter_names=parameter_names,
            )
        )

    return OptimizerGroupPlan(
        groups=tuple(assignments),
        parameters_by_name=parameters_by_name,
    )


def _match_adapter_parameters(
    group_names_by_parameter: dict[str, list[str]],
    *,
    adapter_receipt: DoraAdapterSetupReceipt,
) -> None:
    receipt_trainables = set(adapter_receipt.trainable_names)
    actual_trainables = set(group_names_by_parameter)
    missing_receipt_trainables = sorted(receipt_trainables - actual_trainables)
    if missing_receipt_trainables:
        raise RuntimeContractError(
            "adapter receipt trainable parameter is missing from the model trainable surface",
            code="optimizer.receipt_parameter_missing",
            context={
                "group_name": "adapter",
                "parameter_name": missing_receipt_trainables[0],
                "missing_parameter_names": missing_receipt_trainables,
            },
        )
    target_to_group = {
        target: f"{ADAPTER_GROUP_PREFIX}{_tower_for_adapter_target(target)}"
        for target in adapter_receipt.target_discovery.matched_modules
    }
    for parameter_name in group_names_by_parameter:
        if parameter_name not in receipt_trainables:
            continue
        matched_target_groups = [
            group_name
            for target, group_name in target_to_group.items()
            if _parameter_belongs_to_adapter_target(parameter_name, target)
        ]
        if len(matched_target_groups) != 1:
            raise RuntimeContractError(
                "adapter trainable parameter could not be mapped to exactly one target",
                code="optimizer.adapter_target_match",
                context={
                    "parameter_name": parameter_name,
                    "matched_target_groups": matched_target_groups,
                },
            )
        group_name = matched_target_groups[0]
        _require_group_config(group_name)
        group_names_by_parameter[parameter_name].append(group_name)


def _match_token_embedding_parameters(
    group_names_by_parameter: dict[str, list[str]],
    *,
    special_token_receipt: SpecialTokenEmbeddingInstallReceipt,
) -> None:
    for parameter_name in special_token_receipt.delta_parameter_names:
        if parameter_name not in group_names_by_parameter:
            raise RuntimeContractError(
                "selected embedding delta parameter from receipt is not trainable",
                code="optimizer.receipt_parameter_missing",
                context={
                    "group_name": TOKEN_EMBEDDINGS_GROUP,
                    "parameter_name": parameter_name,
                },
            )
        group_names_by_parameter[parameter_name].append(TOKEN_EMBEDDINGS_GROUP)


def _optimizer_group_config(
    optimizer_config: OptimizerConfig,
    group_name: str,
) -> OptimizerGroupConfig:
    if group_name == TOKEN_EMBEDDINGS_GROUP:
        return optimizer_config.groups.token_embeddings
    if group_name == "adapter.language":
        group_config = optimizer_config.groups.adapters.language
    elif group_name == "adapter.vision":
        group_config = optimizer_config.groups.adapters.vision
    elif group_name == "adapter.aligner":
        group_config = optimizer_config.groups.adapters.aligner
    else:
        group_config = None
    if group_config is None:
        raise RuntimeContractError(
            "optimizer group requires an explicit learning-rate configuration",
            code="optimizer.group_missing",
            context={"group_name": group_name},
        )
    return group_config


def _require_group_config(group_name: str) -> None:
    if group_name not in {
        "adapter.language",
        "adapter.vision",
        "adapter.aligner",
    }:
        raise RuntimeContractError(
            "unsupported adapter optimizer group",
            code="optimizer.group_unsupported",
            context={"group_name": group_name},
        )


def _tower_for_adapter_target(target: str) -> str:
    if _is_aligner_target(target):
        return "aligner"
    if "visual." in target or target.startswith("visual."):
        return "vision"
    if "language_model." in target or target.startswith("language_model."):
        return "language"
    raise RuntimeContractError(
        "adapter target cannot be mapped to an optimizer tower",
        code="optimizer.adapter_target_tower_unknown",
        context={"target": target},
    )


def _is_aligner_target(target: str) -> bool:
    return (
        ".visual.merger" in target
        or target.startswith("visual.merger")
        or "deepstack_merger" in target
    )


def _parameter_belongs_to_adapter_target(parameter_name: str, target: str) -> bool:
    return f".{target}." in parameter_name or parameter_name.startswith(f"{target}.")
