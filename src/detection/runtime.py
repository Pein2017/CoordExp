"""Latest-schema detection launch/runtime policy helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, Mapping, cast

from src.config.prompts import get_template_prompts
from src.config.schema import (
    CoordOffsetConfig,
    CoordTokensConfig,
    LatestDetectionTrainingConfig,
)
from src.detection.dataset import DetectionTrainingDataset
from src.detection.coord_soft_targets import CoordSoftTargetRuntimeConfig
from src.detection.tokenizer_contract import resolve_compact_training_stop_contract

LatestDetectionRuntimeMode = Literal[
    "sorted_sft",
    "random_order_sft",
    "random_permutation_et_rmp_ce",
    "prefix_rollin_et_rmp_ce",
]


@dataclass(frozen=True)
class DetectionRuntimeSupport:
    """Resolved latest-detection runtime support policy."""

    recursive_sidecars_required: bool


@dataclass(frozen=True)
class RecursiveDetectionCERuntimeConfig:
    enabled: bool
    trie_support_weight: float
    trie_balance_weight: float
    variant: str = "random_permutation_et_rmp_ce"
    separator_continue_weight: float = 0.50
    eos_stop_weight: float = 0.50
    boundary_component_weight: float = 0.30
    coord_soft_ce: CoordSoftTargetRuntimeConfig | None = None


def is_latest_detection_config(training_config: Any) -> bool:
    return isinstance(training_config, LatestDetectionTrainingConfig)


def latest_detection_sequence_format(
    training_config: LatestDetectionTrainingConfig,
) -> str:
    if training_config.detection_template.id == "compact_full":
        return "compact_full"
    if training_config.detection_template.id == "stage1_json_pretty":
        return "coordjson"
    raise ValueError(
        f"Unsupported detection_template.id={training_config.detection_template.id!r}"
    )


def latest_detection_prompt_variant(
    training_config: LatestDetectionTrainingConfig,
) -> str | None:
    return "coco_80" if training_config.prompt.prompt_variant_enabled else None


def resolve_latest_detection_prompts(
    training_config: LatestDetectionTrainingConfig,
) -> tuple[str, str]:
    if training_config.prompt.system_variant != "stage1_detection":
        raise ValueError(
            "latest detection runtime currently supports only "
            "prompt.system_variant=stage1_detection"
        )
    if (
        training_config.detection_template.id == "compact_full"
        and training_config.prompt.user_variant != "compact_detection"
    ):
        raise ValueError(
            "detection_template.id=compact_full requires "
            "prompt.user_variant=compact_detection"
        )
    if (
        training_config.detection_template.id == "stage1_json_pretty"
        and training_config.prompt.user_variant != "stage1_detection"
    ):
        raise ValueError(
            "detection_template.id=stage1_json_pretty requires "
            "prompt.user_variant=stage1_detection"
        )

    object_field_order = training_config.detection_template.object_field_order
    if object_field_order is None:
        object_field_order = "desc_first"
    ordering = (
        "random"
        if training_config.data.object_ordering == "random_permutation"
        else "sorted"
    )
    return get_template_prompts(
        ordering=ordering,
        coord_mode="coord_tokens",
        prompt_variant=latest_detection_prompt_variant(training_config),
        object_field_order=str(object_field_order),
        bbox_format=str(training_config.detection_template.bbox_format),
        detection_sequence_format=latest_detection_sequence_format(training_config),
    )


def build_latest_detection_runtime_custom_shim(
    training_config: LatestDetectionTrainingConfig,
) -> SimpleNamespace:
    _system_prompt, user_prompt = resolve_latest_detection_prompts(training_config)
    object_field_order = training_config.detection_template.object_field_order
    if object_field_order is None:
        object_field_order = "desc_first"
    object_ordering = (
        "random"
        if training_config.data.object_ordering == "random_permutation"
        else "sorted"
    )
    prompt_variant = latest_detection_prompt_variant(training_config)
    return SimpleNamespace(
        extra={"prompt_variant": prompt_variant} if prompt_variant else {},
        train_jsonl=training_config.data.train_jsonl,
        val_jsonl=training_config.data.val_jsonl,
        bypass_prob=0.0,
        augmentation=False,
        augmentation_curriculum=False,
        train_sample_limit=None,
        val_sample_limit=None,
        val_sample_with_replacement=False,
        use_summary=False,
        system_prompt_summary=None,
        user_prompt=user_prompt,
        emit_norm="none",
        json_format="standard",
        coord_tokens=CoordTokensConfig(enabled=True, skip_bbox_norm=True),
        offline_max_pixels=None,
        object_ordering=object_ordering,
        object_field_order=str(object_field_order),
        bbox_format=str(training_config.detection_template.bbox_format),
        detection_sequence_format=latest_detection_sequence_format(training_config),
        eval_detection=None,
        token_type_metrics=None,
        coord_soft_ce_w1=None,
        bbox_geo=None,
        bbox_size_aux=None,
        sft_structural_close=None,
        dump_conversation_text=False,
        dump_conversation_path=None,
        coord_offset=CoordOffsetConfig(enabled=False),
        trainable_token_rows=training_config.token_rows,
    )


def latest_detection_mode(
    training_config: LatestDetectionTrainingConfig,
) -> LatestDetectionRuntimeMode:
    variant = training_config.objective.variant
    supported = {
        "sorted_sft",
        "random_order_sft",
        "random_permutation_et_rmp_ce",
        "prefix_rollin_et_rmp_ce",
    }
    if variant in supported:
        return cast(LatestDetectionRuntimeMode, variant)
    raise ValueError(
        "latest detection runtime does not support "
        f"objective.variant={variant!r}; use sorted_sft, random_order_sft, "
        "random_permutation_et_rmp_ce, or prefix_rollin_et_rmp_ce"
    )


def resolve_detection_runtime_support(
    training_config: LatestDetectionTrainingConfig,
) -> DetectionRuntimeSupport:
    return DetectionRuntimeSupport(
        recursive_sidecars_required=(
            training_config.detection_template.id == "compact_full"
            and training_config.objective.id == "recursive_detection_ce"
        )
    )


def assert_latest_detection_runtime_supported(
    training_config: LatestDetectionTrainingConfig,
    *,
    encoded_sample_cache_cfg: Any,
    tokenizer: object | None = None,
) -> None:
    support = resolve_detection_runtime_support(training_config)
    if not support.recursive_sidecars_required:
        return

    configured_padding_side = training_config.training.get("padding_side")
    if configured_padding_side not in (None, "", "right"):
        raise ValueError(
            "latest recursive detection sidecars require training.padding_side='right' "
            "until sidecar offset rewriting is implemented"
        )
    if tokenizer is not None:
        padding_side = getattr(tokenizer, "padding_side", "right")
        if padding_side not in (None, "right"):
            raise ValueError(
                "latest recursive detection sidecars require tokenizer.padding_side='right' "
                "until sidecar offset rewriting is implemented"
            )

    if training_config.objective.variant == "prefix_rollin_et_rmp_ce":
        if tokenizer is None:
            raise ValueError(
                "prefix_rollin_et_rmp_ce requires tokenizer context for <|im_end|> "
                "stop-contract validation"
            )
        resolve_compact_training_stop_contract(tokenizer)

    if bool(training_config.training.get("packing", False)):
        raise ValueError(
            "latest recursive detection sidecars currently require "
            "training.packing=false; "
            "packed target-position offset rewriting is not implemented yet"
        )
    if bool(training_config.training.get("eval_packing", False)):
        raise ValueError(
            "latest recursive detection sidecars currently require "
            "training.eval_packing=false; "
            "packed target-position offset rewriting is not implemented yet"
        )
    if bool(training_config.training.get("use_logits_to_keep", False)):
        raise ValueError(
            "latest recursive detection sidecars require "
            "training.use_logits_to_keep=false because full logits are required"
        )
    if "loss_scale" in training_config.training and training_config.training.get(
        "loss_scale"
    ) not in (None, ""):
        raise ValueError(
            "latest recursive detection sidecars do not support training.loss_scale; "
            "recursive_detection_ce owns the token loss and metric scale"
        )
    if training_config.packing.static_packing:
        raise ValueError(
            "latest recursive detection sidecars currently require "
            "packing.static_packing=false; "
            "packed target-position offset rewriting is not implemented yet"
        )
    if training_config.packing.padding_free_packed:
        raise ValueError(
            "latest recursive detection sidecars currently require "
            "packing.padding_free_packed=false; "
            "packed target-position offset rewriting is not implemented yet"
        )
    if encoded_sample_cache_cfg.enabled:
        raise ValueError(
            "latest recursive detection sidecars currently reject "
            "training.encoded_sample_cache until sidecar cache fingerprints are implemented"
        )


def resolve_recursive_detection_ce_runtime_cfg(
    training_config: Any,
) -> RecursiveDetectionCERuntimeConfig | None:
    objective = getattr(training_config, "objective", None)
    if objective is None:
        return None
    objective_id = (
        str(objective.get("id"))
        if isinstance(objective, Mapping) and objective.get("id") is not None
        else str(getattr(objective, "id", "") or "")
    )
    if objective_id != "recursive_detection_ce":
        return None
    variant = (
        str(objective.get("variant"))
        if isinstance(objective, Mapping) and objective.get("variant") is not None
        else str(getattr(objective, "variant", "") or "")
    )

    def _field(container: Any, field_name: str) -> Any:
        if isinstance(container, Mapping):
            return container.get(field_name)
        return getattr(container, field_name, None)

    def _objective_float(field_name: str) -> float:
        raw = _field(objective, field_name)
        if raw is None:
            raise ValueError(f"objective.{field_name} is required")
        value = float(raw)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"objective.{field_name} must be finite and >= 0")
        return value

    if variant == "random_permutation_et_rmp_ce":
        trie_support_weight = _objective_float("trie_support_weight")
        trie_balance_weight = _objective_float("trie_balance_weight")
        separator_continue_weight = 0.50
        eos_stop_weight = 0.50
        boundary_component_weight = 0.30
    elif variant == "prefix_rollin_et_rmp_ce":
        target = _field(objective, "target")
        if target is None:
            raise ValueError(
                "objective.target is required for "
                "objective.variant=prefix_rollin_et_rmp_ce"
            )
        trie_support_weight = float(_field(target, "support_weight"))
        trie_balance_weight = float(_field(target, "balance_weight"))
        for field_name, value in (
            ("support_weight", trie_support_weight),
            ("balance_weight", trie_balance_weight),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"objective.target.{field_name} must be finite and > 0 "
                    "for prefix_rollin_et_rmp_ce"
                )
        boundary = _field(objective, "boundary")
        if boundary is None:
            raise ValueError(
                "objective.boundary is required for "
                "objective.variant=prefix_rollin_et_rmp_ce"
            )
        separator_continue_weight = float(_field(boundary, "separator_continue_weight"))
        eos_stop_weight = float(_field(boundary, "eos_stop_weight"))
        boundary_component_weight = float(_field(boundary, "component_weight"))
        for field_name, value in (
            ("separator_continue_weight", separator_continue_weight),
            ("eos_stop_weight", eos_stop_weight),
            ("component_weight", boundary_component_weight),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"objective.boundary.{field_name} must be finite and > 0 "
                    "for prefix_rollin_et_rmp_ce"
                )
    else:
        raise ValueError(
            "recursive_detection_ce runtime currently supports only "
            "objective.variant=random_permutation_et_rmp_ce or "
            "prefix_rollin_et_rmp_ce"
        )

    if trie_support_weight + trie_balance_weight <= 0.0:
        raise ValueError(
            "recursive_detection_ce support and balance weights must sum to > 0"
        )
    coord_soft_ce = _resolve_coord_soft_ce_runtime_config(
        training_config=training_config,
        objective=objective,
        field_getter=_field,
    )
    return RecursiveDetectionCERuntimeConfig(
        enabled=True,
        trie_support_weight=trie_support_weight,
        trie_balance_weight=trie_balance_weight,
        variant=variant,
        separator_continue_weight=separator_continue_weight,
        eos_stop_weight=eos_stop_weight,
        boundary_component_weight=boundary_component_weight,
        coord_soft_ce=coord_soft_ce,
    )


def _resolve_coord_soft_ce_runtime_config(
    *,
    training_config: LatestDetectionTrainingConfig,
    objective: Any,
    field_getter: Any,
) -> CoordSoftTargetRuntimeConfig | None:
    raw_cfg = field_getter(objective, "coord_soft_ce")
    if raw_cfg is None or not bool(field_getter(raw_cfg, "enabled")):
        return None
    if field_getter(raw_cfg, "replace_coord_hard_ce") is not True:
        raise ValueError("objective.coord_soft_ce.replace_coord_hard_ce must be true")

    coord_group = None
    for group in training_config.token_rows.groups.values():
        role = getattr(group.role, "value", group.role)
        if str(role) == "coord_geometry":
            coord_group = group
            break
    if coord_group is None:
        raise ValueError(
            "objective.coord_soft_ce requires a token_rows coord_geometry group"
        )
    if coord_group.expected_start is None or coord_group.expected_end is None:
        raise ValueError(
            "objective.coord_soft_ce requires token_rows coord_geometry expected_start/end"
        )

    return CoordSoftTargetRuntimeConfig(
        target_distribution=str(field_getter(raw_cfg, "target_distribution")),
        tau=float(field_getter(raw_cfg, "tau")),
        coord_token_start=int(coord_group.expected_start),
        coord_token_end=int(coord_group.expected_end),
        weighting=str(field_getter(raw_cfg, "weighting")),
        apply_to_multi_positive=str(field_getter(raw_cfg, "apply_to_multi_positive")),
    )


def build_latest_detection_dataset(
    jsonl_path: str | Path,
    *,
    swift_template: Any,
    training_config: LatestDetectionTrainingConfig,
    custom_config: Any,
    system_prompt: str | None,
    seed: int,
    sample_limit: int | None,
    dataset_name: str,
) -> DetectionTrainingDataset:
    eos_trust_weight_config = None
    type_gate_config = None
    if training_config.objective.eos is not None:
        eos_trust_weight_config = training_config.objective.eos.eos_trust_weight
    if training_config.objective.variant == "prefix_rollin_et_rmp_ce":
        eos_cfg = training_config.objective.eos
        if eos_cfg is None:
            raise ValueError("prefix_rollin_et_rmp_ce requires objective.eos")
        type_gate_config = training_config.objective.type_gate
    return DetectionTrainingDataset.from_jsonl(
        jsonl_path,
        swift_template=swift_template,
        image_root=training_config.data.image_root,
        detection_template_id=training_config.detection_template.id,
        mode=latest_detection_mode(training_config),
        object_ordering=training_config.data.object_ordering,
        user_prompt=custom_config.user_prompt,
        system_prompt=system_prompt,
        max_objects=training_config.data.max_objects,
        seed=seed,
        state_weighting=training_config.objective.state_weighting,
        normalization=training_config.objective.normalization,
        eos_trust_weight_config=eos_trust_weight_config,
        type_gate_config=type_gate_config,
        sample_limit=sample_limit,
        dataset_name=dataset_name,
    )


__all__ = [
    "DetectionRuntimeSupport",
    "LatestDetectionRuntimeMode",
    "RecursiveDetectionCERuntimeConfig",
    "assert_latest_detection_runtime_supported",
    "build_latest_detection_dataset",
    "build_latest_detection_runtime_custom_shim",
    "is_latest_detection_config",
    "latest_detection_mode",
    "latest_detection_prompt_variant",
    "latest_detection_sequence_format",
    "resolve_detection_runtime_support",
    "resolve_latest_detection_prompts",
    "resolve_recursive_detection_ce_runtime_cfg",
]
