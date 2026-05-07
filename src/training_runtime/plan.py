from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

CollatorFamily: TypeAlias = Literal["default", "stage1_set_continuation", "identity"]
PackingOwner: TypeAlias = Literal["dataset", "trainer"]
PipelineNamespace: TypeAlias = Literal[
    "stage2_ab.pipeline", "rollout_matching.pipeline"
]


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
    "stage2_ab_training": "stage2_two_channel",
    "rollout_matching_sft": "stage2_rollout_aligned",
}


def resolve_training_runtime_plan(trainer_variant: str | None) -> TrainingRuntimePlan:
    """Resolve setup policy for a trainer variant without importing trainer code."""

    variant = str(trainer_variant or "")
    replacement = _REMOVED_VARIANT_REPLACEMENTS.get(variant)
    if replacement is not None:
        raise ValueError(
            f"custom.trainer_variant={variant} has been removed; use {replacement}"
        )

    if variant == "stage1_set_continuation":
        return TrainingRuntimePlan(
            variant=variant,
            preserve_raw_sample_metadata=True,
            dataset_static_packing_allowed=False,
            dataset_static_packing_owner=None,
            post_rollout_packing_owner=None,
            collator_family="stage1_set_continuation",
            ordinary_stage1_mixins_allowed=False,
            required_pipeline_namespace=None,
            requires_top_level_rollout_matching=False,
        )

    if variant == "stage2_two_channel":
        return _stage2_plan(
            variant=variant, required_pipeline_namespace="stage2_ab.pipeline"
        )

    if variant == "stage2_rollout_aligned":
        return _stage2_plan(
            variant=variant,
            required_pipeline_namespace="rollout_matching.pipeline",
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
