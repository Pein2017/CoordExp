"""Current-schema detection launch/runtime policy helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, Mapping, cast

from src.config.prompts import get_template_prompts
from src.config.schema import (
    CoordTokensConfig,
    DetectionTrainingConfig,
)
from src.detection.dataset import DetectionTrainingDataset
from src.detection.coord_soft_targets import CoordSoftTargetRuntimeConfig
from src.detection.prefix_denoising import PrefixDenoisingTrainingDataset
from src.detection.prefix_denoising.builder import (
    prefix_denoising_fast_estimator_fingerprint,
)
from src.detection.template_contracts import resolve_detection_template_contract
from src.detection.tokenizer_contract import resolve_compact_training_stop_contract

DetectionRuntimeMode = Literal[
    "sorted_sft",
    "random_order_sft",
    "prefix_denoising_sft",
    "random_permutation_et_rmp_ce",
    "prefix_rollin_et_rmp_ce",
]


@dataclass(frozen=True)
class DetectionRuntimeSupport:
    """Resolved latest-detection runtime support policy."""

    recursive_sidecars_required: bool
    teacher_forcing_target_ir_required: bool = False


@dataclass(frozen=True)
class RecursiveDetectionCERuntimeConfig:
    enabled: bool
    trie_support_weight: float
    trie_balance_weight: float
    variant: str = "random_permutation_et_rmp_ce"
    coord_soft_ce: CoordSoftTargetRuntimeConfig | None = None
    type_gate: Any | None = None


def is_detection_config(training_config: Any) -> bool:
    return isinstance(training_config, DetectionTrainingConfig)


def detection_sequence_format(
    training_config: DetectionTrainingConfig,
) -> str:
    contract = resolve_detection_template_contract(training_config.detection_template.id)
    if contract.is_compact:
        return "compact"
    if contract.template_id == "stage1_json_pretty":
        return "coordjson"
    raise ValueError(
        f"Unsupported detection_template.id={training_config.detection_template.id!r}"
    )


def detection_prompt_variant(
    training_config: DetectionTrainingConfig,
) -> str | None:
    return "coco_80" if training_config.prompt.prompt_variant_enabled else None


def resolve_detection_prompts(
    training_config: DetectionTrainingConfig,
) -> tuple[str, str]:
    if training_config.prompt.system_variant != "stage1_detection":
        raise ValueError(
            "detection runtime currently supports only "
            "prompt.system_variant=stage1_detection"
        )
    template_contract = resolve_detection_template_contract(
        training_config.detection_template.id
    )
    if (
        template_contract.is_compact
        and training_config.prompt.user_variant != "compact_detection"
    ):
        raise ValueError(
            f"detection_template.id={training_config.detection_template.id} requires "
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
        prompt_variant=detection_prompt_variant(training_config),
        object_field_order=str(object_field_order),
        bbox_format=str(training_config.detection_template.bbox_format),
        detection_sequence_format=detection_sequence_format(training_config),
    )


def build_detection_runtime_custom_shim(
    training_config: DetectionTrainingConfig,
) -> SimpleNamespace:
    _system_prompt, user_prompt = resolve_detection_prompts(training_config)
    object_field_order = training_config.detection_template.object_field_order
    if object_field_order is None:
        object_field_order = "desc_first"
    object_ordering = (
        "random"
        if training_config.data.object_ordering == "random_permutation"
        else "sorted"
    )
    prompt_variant = detection_prompt_variant(training_config)
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
        detection_sequence_format=detection_sequence_format(training_config),
        eval_detection=None,
        token_type_metrics=None,
        coord_soft_ce_w1=None,
        bbox_geo=None,
        bbox_size_aux=None,
        sft_structural_close=None,
        dump_conversation_text=False,
        dump_conversation_path=None,
        token_embeddings_adapter=training_config.token_rows,
    )


def detection_mode(
    training_config: DetectionTrainingConfig,
) -> DetectionRuntimeMode:
    objective_id = getattr(training_config.objective, "id", None)
    if objective_id == "teacher_forcing":
        prefix_denoising = getattr(training_config, "prefix_denoising", None)
        if getattr(prefix_denoising, "enabled", False):
            return "prefix_denoising_sft"
        if training_config.objective.profile not in {
            "hard_sft",
            "pure_valid_set_marginal",
        }:
            raise ValueError(
                "teacher_forcing detection runtime currently supports "
                "objective.profile in {'hard_sft', 'pure_valid_set_marginal'}"
            )
        rollin_policy = training_config.objective.target_ir.rollin_policy
        if rollin_policy.name != "random_permutation":
            raise ValueError(
                "teacher_forcing detection runtime currently supports only "
                "objective.target_ir.rollin_policy.name=random_permutation"
            )
        return "random_order_sft"

    variant = training_config.objective.variant
    supported = {
        "sorted_sft",
        "random_order_sft",
        "random_permutation_et_rmp_ce",
        "prefix_rollin_et_rmp_ce",
    }
    if variant in supported:
        return cast(DetectionRuntimeMode, variant)
    raise ValueError(
        "detection runtime does not support "
        f"objective.variant={variant!r}; use sorted_sft, random_order_sft, "
        "random_permutation_et_rmp_ce, or prefix_rollin_et_rmp_ce"
    )


def resolve_detection_runtime_support(
    training_config: DetectionTrainingConfig,
) -> DetectionRuntimeSupport:
    objective_id = getattr(training_config.objective, "id", None)
    is_compact = resolve_detection_template_contract(
        training_config.detection_template.id
    ).is_compact
    return DetectionRuntimeSupport(
        recursive_sidecars_required=(
            is_compact and objective_id == "recursive_detection_ce"
        ),
        teacher_forcing_target_ir_required=(
            is_compact and objective_id == "teacher_forcing"
        ),
    )


def assert_detection_runtime_supported(
    training_config: DetectionTrainingConfig,
    *,
    encoded_sample_cache_cfg: Any,
    tokenizer: object | None = None,
) -> None:
    support = resolve_detection_runtime_support(training_config)
    if support.teacher_forcing_target_ir_required:
        prefix_denoising = getattr(training_config, "prefix_denoising", None)
        prefix_denoising_enabled = bool(getattr(prefix_denoising, "enabled", False))
        if prefix_denoising_enabled and bool(
            training_config.training.get("use_logits_to_keep", False)
        ):
            raise ValueError(
                "prefix_denoising requires training.use_logits_to_keep=false"
            )
        if (
            bool(training_config.training.get("packing", False))
            and not prefix_denoising_enabled
        ):
            raise ValueError(
                "latest teacher_forcing_target_ir requires "
                "training.packing=false; exact atom-position packing mapping "
                "is not implemented yet"
            )
        if bool(training_config.training.get("eval_packing", False)):
            raise ValueError(
                "latest teacher_forcing_target_ir requires "
                "training.eval_packing=false; exact atom-position packing "
                "mapping is not implemented yet"
            )
        if training_config.packing.static_packing and not prefix_denoising_enabled:
            raise ValueError(
                "latest teacher_forcing_target_ir requires "
                "packing.static_packing=false; exact atom-position packing "
                "mapping is not implemented yet"
            )
        if (
            prefix_denoising_enabled
            and training_config.packing.static_packing
            and not bool(training_config.training.get("packing", False))
        ):
            raise ValueError(
                "prefix_denoising requires packing.static_packing=true only when "
                "training.packing=true"
            )
        if training_config.packing.padding_free_packed:
            raise ValueError(
                "latest teacher_forcing_target_ir requires "
                "packing.padding_free_packed=false; exact atom-position "
                "packing mapping is not implemented yet"
            )
        if getattr(encoded_sample_cache_cfg, "enabled", False):
            raise ValueError(
                "latest teacher_forcing_target_ir requires "
                "training.encoded_sample_cache.enabled=false; exact "
                "atom-position cache replay is not implemented yet"
            )
        return
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

    objective_variant = str(getattr(training_config.objective, "variant", "") or "")
    if objective_variant == "prefix_rollin_et_rmp_ce":
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
        coord_soft_ce=coord_soft_ce,
        type_gate=_field(objective, "type_gate"),
    )


def _resolve_coord_soft_ce_runtime_config(
    *,
    training_config: DetectionTrainingConfig,
    objective: Any,
    field_getter: Any,
) -> CoordSoftTargetRuntimeConfig | None:
    raw_cfg = field_getter(objective, "coord_soft_ce")
    if raw_cfg is None or not bool(field_getter(raw_cfg, "enabled")):
        return None

    coord_group = None
    groups = getattr(training_config.token_rows, "groups", {})
    coord_group = groups.get("coord_geometry") if isinstance(groups, Mapping) else None
    if coord_group is None:
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

    target_distribution = str(field_getter(raw_cfg, "target_distribution"))
    if target_distribution == "instance_trie_gaussian":
        gaussian_mixture_weight = field_getter(raw_cfg, "gaussian_mixture_weight")
        gaussian_r95_axis_fraction = field_getter(
            raw_cfg,
            "gaussian_r95_axis_fraction",
        )
        gaussian_r95_cap_bins = field_getter(raw_cfg, "gaussian_r95_cap_bins")
        return CoordSoftTargetRuntimeConfig(
            target_distribution="instance_trie_gaussian",
            coord_token_start=int(coord_group.expected_start),
            coord_token_end=int(coord_group.expected_end),
            gaussian_mixture_weight=(
                0.1
                if gaussian_mixture_weight is None
                else float(gaussian_mixture_weight)
            ),
            gaussian_r95_axis_fraction=(
                0.04
                if gaussian_r95_axis_fraction is None
                else float(gaussian_r95_axis_fraction)
            ),
            gaussian_r95_cap_bins=(
                8 if gaussian_r95_cap_bins is None else int(gaussian_r95_cap_bins)
            ),
        )

    if field_getter(raw_cfg, "replace_coord_hard_ce") is not True:
        raise ValueError("objective.coord_soft_ce.replace_coord_hard_ce must be true")
    return CoordSoftTargetRuntimeConfig(
        target_distribution=target_distribution,
        tau=float(field_getter(raw_cfg, "tau")),
        coord_token_start=int(coord_group.expected_start),
        coord_token_end=int(coord_group.expected_end),
        weighting=str(field_getter(raw_cfg, "weighting")),
        apply_to_multi_positive=str(field_getter(raw_cfg, "apply_to_multi_positive")),
    )


def build_detection_dataset(
    jsonl_path: str | Path,
    *,
    swift_template: Any,
    training_config: DetectionTrainingConfig,
    custom_config: Any,
    system_prompt: str | None,
    seed: int,
    sample_limit: int | None,
    dataset_name: str,
) -> DetectionTrainingDataset | PrefixDenoisingTrainingDataset:
    prefix_denoising = getattr(training_config, "prefix_denoising", None)
    if getattr(prefix_denoising, "enabled", False):
        max_length = _resolve_detection_max_length(training_config)
        object_ordering = str(
            getattr(training_config.data, "object_ordering", "sorted") or "sorted"
        )
        return PrefixDenoisingTrainingDataset.from_jsonl(
            jsonl_path,
            swift_template=swift_template,
            image_root=training_config.data.image_root,
            user_prompt=custom_config.user_prompt,
            system_prompt=system_prompt,
            prefix_denoising=prefix_denoising,
            detection_template_id=training_config.detection_template.id,
            object_ordering=object_ordering,
            max_length=max_length,
            seed=seed,
            sample_limit=sample_limit,
            dataset_name=dataset_name,
            **_prefix_denoising_eligibility_cache_kwargs(
                jsonl_path=jsonl_path,
                swift_template=swift_template,
                training_config=training_config,
                custom_config=custom_config,
                system_prompt=system_prompt,
                max_length=max_length,
                seed=seed,
                sample_limit=sample_limit,
                dataset_name=dataset_name,
            ),
        )

    type_gate_config = None
    objective = training_config.objective
    objective_id = str(getattr(objective, "id", "") or "")
    objective_variant = str(getattr(objective, "variant", "") or "")
    if objective_variant in {
        "random_permutation_et_rmp_ce",
        "prefix_rollin_et_rmp_ce",
    }:
        type_gate_config = getattr(objective, "type_gate", None)
    if objective_id == "teacher_forcing":
        state_weighting = "uniform_permutation"
        normalization = "semantic_image_bucket_balanced"
        teacher_forcing_profile = str(getattr(objective, "profile"))
        teacher_forcing_rollin_base_seed = int(
            getattr(objective.target_ir.rollin_policy, "base_seed")
        )
    else:
        state_weighting = getattr(objective, "state_weighting")
        normalization = getattr(objective, "normalization")
        teacher_forcing_profile = None
        teacher_forcing_rollin_base_seed = None
    return DetectionTrainingDataset.from_jsonl(
        jsonl_path,
        swift_template=swift_template,
        image_root=training_config.data.image_root,
        detection_template_id=training_config.detection_template.id,
        mode=detection_mode(training_config),
        object_ordering=training_config.data.object_ordering,
        user_prompt=custom_config.user_prompt,
        system_prompt=system_prompt,
        seed=seed,
        state_weighting=state_weighting,
        normalization=normalization,
        object_field_order=str(
            training_config.detection_template.object_field_order or "desc_first"
        ),
        type_gate_config=type_gate_config,
        teacher_forcing_profile=teacher_forcing_profile,
        teacher_forcing_rollin_base_seed=teacher_forcing_rollin_base_seed,
        sample_limit=sample_limit,
        dataset_name=dataset_name,
    )


def _resolve_detection_max_length(training_config: DetectionTrainingConfig) -> int:
    if training_config.global_max_length is not None:
        return int(training_config.global_max_length)
    template_max_length = training_config.template.get("max_length")
    if template_max_length is not None:
        return int(template_max_length)
    raise ValueError(
        "prefix_denoising dataset construction requires global_max_length "
        "or template.max_length"
    )


def _prefix_denoising_eligibility_cache_kwargs(
    *,
    jsonl_path: str | Path,
    swift_template: Any,
    training_config: DetectionTrainingConfig,
    custom_config: Any,
    system_prompt: str | None,
    max_length: int,
    seed: int,
    sample_limit: int | None,
    dataset_name: str,
) -> dict[str, Any]:
    training = getattr(training_config, "training", {}) or {}
    if not isinstance(training, Mapping):
        return {}
    if not bool(training.get("packing", False)):
        return {}
    static_cache = training.get("static_packing_cache") or {}
    if not isinstance(static_cache, Mapping):
        return {}
    root_dir = static_cache.get("root_dir")
    if root_dir in (None, ""):
        return {}

    path = Path(jsonl_path).expanduser().resolve(strict=False)
    try:
        stat_result = path.stat()
        jsonl_size = int(stat_result.st_size)
        jsonl_mtime_ns = int(stat_result.st_mtime_ns)
    except OSError:
        jsonl_size = None
        jsonl_mtime_ns = None

    workers_raw = training.get("packing_length_precompute_workers", 1)
    try:
        workers = int(workers_raw)
    except (TypeError, ValueError):
        workers = 1
    workers = max(workers, 1)

    wait_raw = training.get("packing_wait_timeout_s", 7200.0)
    try:
        wait_timeout_s = float(wait_raw)
    except (TypeError, ValueError):
        wait_timeout_s = 7200.0

    prefix_denoising = getattr(training_config, "prefix_denoising", None)
    detection_template = getattr(training_config, "detection_template", None)
    noise = getattr(prefix_denoising, "noise", None)
    kl = getattr(prefix_denoising, "current_object_kl", None)
    fingerprint = {
        "schema_version": "prefix_denoising_runtime_eligibility_v1",
        "dataset_jsonl": str(path),
        "dataset_jsonl_size": jsonl_size,
        "dataset_jsonl_mtime_ns": jsonl_mtime_ns,
        "dataset_split": str(dataset_name),
        "sample_limit": sample_limit,
        "image_root": str(
            Path(str(training_config.data.image_root))
            .expanduser()
            .resolve(strict=False)
        ),
        "user_prompt": str(custom_config.user_prompt),
        "system_prompt": system_prompt,
        "seed": int(seed),
        "object_ordering": str(
            getattr(training_config.data, "object_ordering", "sorted") or "sorted"
        ),
        "detection_template_id": str(training_config.detection_template.id),
        "max_length": int(max_length),
        "template": {
            "max_length": training_config.template.get("max_length"),
            "chat_template": training_config.template.get("template"),
        },
        "detection_template": {
            "id": getattr(detection_template, "id", None),
            "bbox_format": getattr(detection_template, "bbox_format", None),
            "object_field_order": getattr(
                detection_template, "object_field_order", None
            ),
        },
        "fast_estimator": prefix_denoising_fast_estimator_fingerprint(swift_template),
        "prefix_denoising": {
            "enabled": bool(getattr(prefix_denoising, "enabled", False)),
            "noise": {
                "center_shift_frac": float(
                    getattr(noise, "center_shift_frac", 0.0) or 0.0
                ),
                "uniform_scale_range": [
                    float(value)
                    for value in getattr(noise, "uniform_scale_range", (0.0, 0.0))
                ],
            },
            "current_object_kl": {
                "weight": float(getattr(kl, "weight", 0.0) or 0.0),
                "window_radius": int(getattr(kl, "window_radius", 0) or 0),
                "num_objects_per_image": int(
                    getattr(kl, "num_objects_per_image", 0) or 0
                ),
            },
        },
    }
    return {
        "eligibility_cache_dir": Path(str(root_dir)).expanduser().resolve(strict=False)
        / "prefix_denoising_eligibility",
        "eligibility_cache_fingerprint": fingerprint,
        "eligibility_cache_wait_timeout_s": wait_timeout_s,
        "eligibility_precompute_workers": workers,
    }


__all__ = [
    "DetectionRuntimeSupport",
    "DetectionRuntimeMode",
    "RecursiveDetectionCERuntimeConfig",
    "assert_detection_runtime_supported",
    "build_detection_dataset",
    "build_detection_runtime_custom_shim",
    "is_detection_config",
    "detection_mode",
    "detection_prompt_variant",
    "detection_sequence_format",
    "resolve_detection_runtime_support",
    "resolve_detection_prompts",
    "resolve_recursive_detection_ce_runtime_cfg",
]
