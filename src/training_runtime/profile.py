from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

from src.training_runtime.plan import (
    CollatorFamily,
    PackingOwner,
    PipelineNamespace,
    TrainingRuntimePlan,
    resolve_training_runtime_plan,
)

EncodedCachePolicy: TypeAlias = Literal[
    "dataset_static_cache_allowed",
    "raw_metadata_preserved_no_static_cache",
    "static_cache_disabled",
    "trainer_post_rollout_dynamic",
]
ManifestFamily: TypeAlias = Literal[
    "stage1",
    "stage1_set_continuation",
    "stage2_ab",
    "rollout_matching",
]
PackingPolicy: TypeAlias = Literal[
    "dataset_static_packing",
    "packing_disabled",
    "trainer_post_rollout_packing",
]
RuntimeStage: TypeAlias = Literal["stage1", "stage1_set_continuation", "stage2"]


@dataclass(frozen=True, slots=True)
class TrainingRuntimeProfile:
    """Trainer/bootstrap-facing view derived from TrainingRuntimePlan."""

    variant: str
    runtime_stage: RuntimeStage
    preserve_raw_sample_metadata: bool
    rollout_runtime_owner: PackingOwner | None
    rollout_runtime_owned: bool
    explicit_pipeline_required: bool
    required_pipeline_namespace: PipelineNamespace | None
    ordinary_stage1_mixins_allowed: bool
    ordinary_stage1_mixins_excluded: bool
    dataset_static_packing_allowed: bool
    dataset_static_packing_owner: PackingOwner | None
    post_rollout_packing_owner: PackingOwner | None
    packing_policy: PackingPolicy
    encoded_cache_policy: EncodedCachePolicy
    collator_family: CollatorFamily
    manifest_family: ManifestFamily


def resolve_training_runtime_profile(
    trainer_variant: str | None,
) -> TrainingRuntimeProfile:
    """Resolve the trainer-facing runtime profile through the setup plan."""

    plan = resolve_training_runtime_plan(trainer_variant)
    return build_training_runtime_profile(plan)


def build_training_runtime_profile(plan: TrainingRuntimePlan) -> TrainingRuntimeProfile:
    """Build a trainer-facing profile without a second variant registry."""

    explicit_pipeline_required = plan.required_pipeline_namespace is not None
    runtime_stage = _derive_runtime_stage(plan)

    return TrainingRuntimeProfile(
        variant=plan.variant,
        runtime_stage=runtime_stage,
        preserve_raw_sample_metadata=plan.preserve_raw_sample_metadata,
        rollout_runtime_owner=plan.post_rollout_packing_owner,
        rollout_runtime_owned=plan.post_rollout_packing_owner is not None,
        explicit_pipeline_required=explicit_pipeline_required,
        required_pipeline_namespace=plan.required_pipeline_namespace,
        ordinary_stage1_mixins_allowed=plan.ordinary_stage1_mixins_allowed,
        ordinary_stage1_mixins_excluded=not plan.ordinary_stage1_mixins_allowed,
        dataset_static_packing_allowed=plan.dataset_static_packing_allowed,
        dataset_static_packing_owner=plan.dataset_static_packing_owner,
        post_rollout_packing_owner=plan.post_rollout_packing_owner,
        packing_policy=_derive_packing_policy(plan),
        encoded_cache_policy=_derive_encoded_cache_policy(plan),
        collator_family=plan.collator_family,
        manifest_family=_derive_manifest_family(plan, runtime_stage),
    )


def _derive_runtime_stage(plan: TrainingRuntimePlan) -> RuntimeStage:
    if (
        plan.post_rollout_packing_owner is not None
        or plan.required_pipeline_namespace is not None
        or plan.requires_top_level_rollout_matching
    ):
        return "stage2"
    if plan.collator_family != "default":
        return plan.collator_family
    return "stage1"


def _derive_packing_policy(plan: TrainingRuntimePlan) -> PackingPolicy:
    if plan.post_rollout_packing_owner is not None:
        return "trainer_post_rollout_packing"
    if plan.dataset_static_packing_allowed:
        return "dataset_static_packing"
    return "packing_disabled"


def _derive_encoded_cache_policy(plan: TrainingRuntimePlan) -> EncodedCachePolicy:
    if plan.post_rollout_packing_owner is not None:
        return "trainer_post_rollout_dynamic"
    if plan.dataset_static_packing_allowed:
        return "dataset_static_cache_allowed"
    if plan.preserve_raw_sample_metadata:
        return "raw_metadata_preserved_no_static_cache"
    return "static_cache_disabled"


def _derive_manifest_family(
    plan: TrainingRuntimePlan,
    runtime_stage: RuntimeStage,
) -> ManifestFamily:
    if plan.required_pipeline_namespace is not None:
        if plan.required_pipeline_namespace == "stage2_ab.pipeline":
            return "stage2_ab"
        return "rollout_matching"
    if runtime_stage == "stage1_set_continuation":
        return "stage1_set_continuation"
    return "stage1"


__all__ = [
    "TrainingRuntimeProfile",
    "build_training_runtime_profile",
    "resolve_training_runtime_profile",
]
