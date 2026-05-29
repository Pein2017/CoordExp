from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

CollatorFamily: TypeAlias = Literal["default", "identity"]
PackingOwner: TypeAlias = Literal["dataset", "trainer"]
PipelineNamespace: TypeAlias = Literal["stage2_rollout_correction.pipeline"]


@dataclass(frozen=True, slots=True)
class TrainingRuntimePlan:
    """Import-safe trainer-variant setup ownership contract."""

    variant: str
    preserve_raw_sample_metadata: bool
    dataset_static_packing_allowed: bool
    dataset_static_packing_owner: PackingOwner | None
    post_rollout_packing_owner: PackingOwner | None
    collator_family: CollatorFamily
    ordinary_stage1_mixins_allowed: bool
    required_pipeline_namespace: PipelineNamespace | None
    requires_top_level_rollout_matching: bool


_REMOVED_VARIANT_REPLACEMENTS: Final[dict[str, str]] = {
    "stage2_" "ab_training": "stage2_rollout_correction",
    "stage2_" "two_channel": "stage2_rollout_correction",
    "rollout_matching_sft": "stage2_rollout_correction",
    "stage2_rollout_aligned": "stage2_rollout_correction",
    "stage2_rollout_runtime": "stage2_rollout_correction",
    "stage1_set_continuation": "prefix_rollin_et_rmp_ce",
}
_GENERIC_STAGE1_EXTENSION_VARIANTS: Final[frozenset[str]] = frozenset(
    {
        "gkd_monitor",
    }
)


def resolve_training_runtime_plan(trainer_variant: str | None) -> TrainingRuntimePlan:
    """Resolve setup policy for a trainer variant without importing trainer code."""

    variant = str(trainer_variant or "").strip()
    replacement = _REMOVED_VARIANT_REPLACEMENTS.get(variant)
    if replacement is not None:
        raise ValueError(
            f"custom.trainer_variant={variant} has been removed; use {replacement}"
        )

    if variant == "stage2_rollout_correction":
        return _stage2_plan(
            variant=variant,
            required_pipeline_namespace="stage2_rollout_correction.pipeline",
        )

    if variant and variant not in _GENERIC_STAGE1_EXTENSION_VARIANTS:
        raise ValueError(
            f"custom.trainer_variant={variant} is not supported; "
            "use stage2_rollout_correction for Stage-2 rollout correction or omit "
            "custom.trainer_variant for the default Stage-1 SFT runtime."
        )

    return TrainingRuntimePlan(
        variant=variant,
        preserve_raw_sample_metadata=False,
        dataset_static_packing_allowed=True,
        dataset_static_packing_owner="dataset",
        post_rollout_packing_owner=None,
        collator_family="default",
        ordinary_stage1_mixins_allowed=True,
        required_pipeline_namespace=None,
        requires_top_level_rollout_matching=False,
    )


def _stage2_plan(
    *,
    variant: str,
    required_pipeline_namespace: PipelineNamespace,
) -> TrainingRuntimePlan:
    return TrainingRuntimePlan(
        variant=variant,
        preserve_raw_sample_metadata=True,
        dataset_static_packing_allowed=False,
        dataset_static_packing_owner=None,
        post_rollout_packing_owner="trainer",
        collator_family="identity",
        ordinary_stage1_mixins_allowed=False,
        required_pipeline_namespace=required_pipeline_namespace,
        requires_top_level_rollout_matching=True,
    )
