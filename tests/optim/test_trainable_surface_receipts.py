from __future__ import annotations

import json

import pytest
import torch
from torch import nn

from src.adapters.dora import DoraAdapterSetupReceipt, DoraTargetDiscoveryReceipt
from src.common.errors import RuntimeContractError
from src.config.models import (
    AdapterOptimizerGroupsConfig,
    OptimizerConfig,
    OptimizerGroupConfig,
    OptimizerGroupsConfig,
    SchedulerConfig,
)
from src.optim.parameter_groups import OptimizerGroupPlan, build_optimizer_group_plan
from src.optim.trainable_surface import (
    FrozenReasonSummary,
    TrainableSurfaceReceipt,
    build_trainable_surface_receipt,
    write_trainable_surface_receipt,
)
from src.qwen.special_token_embeddings import (
    SpecialTokenEmbeddingInstallReceipt,
    SpecialTokenSelection,
)


def test_trainable_surface_receipt_proves_optimizer_groups_and_sources(
    tmp_path,
) -> None:
    model = FakeTrainableSurface(
        adapter_targets=(
            "model.language_model.q_proj",
            "model.visual.block_proj",
            "model.visual.merger.mlp",
        ),
        token_delta=True,
    )
    adapter_receipt = _adapter_receipt(
        targets=(
            "model.language_model.q_proj",
            "model.visual.block_proj",
            "model.visual.merger.mlp",
        )
    )
    token_receipt = _token_receipt(delta_names=("embed_tokens.shared_embed_delta",))
    optimizer_plan = build_optimizer_group_plan(
        model,
        _optimizer_config(
            language=_group(lr=1.0e-4, weight_decay=0.01),
            vision=_group(lr=2.0e-4, weight_decay=0.02),
            aligner=_group(lr=3.0e-4, weight_decay=0.03),
            token_embeddings=_group(lr=4.0e-4, weight_decay=0.0),
        ),
        adapter_receipt=adapter_receipt,
        special_token_receipt=token_receipt,
    )

    receipt = build_trainable_surface_receipt(
        model,
        adapter_receipt=adapter_receipt,
        special_token_receipt=token_receipt,
        optimizer_group_plan=optimizer_plan,
    )
    artifact = receipt.to_artifact_dict()
    json.dumps(artifact, allow_nan=False)

    assert artifact["phase"] == "before_first_backward"
    assert artifact["trainable_towers"] == [
        "adapter.language",
        "adapter.vision",
        "adapter.aligner",
        "token_embeddings",
    ]
    assert artifact["frozen_towers"] == ["language", "vision", "aligner"]
    assert artifact["adapter_targets"]["matched_modules"] == [
        "model.language_model.q_proj",
        "model.visual.block_proj",
        "model.visual.merger.mlp",
    ]
    assert artifact["selected_embedding_tokens"]["token_ids"] == [2, 3]
    assert artifact["selected_embedding_tokens"]["selected_token_count"] == 2
    assert artifact["parameter_counts"]["trainable_parameter_count"] == 10
    assert artifact["parameter_counts"]["trainable_scalar_count"] == 13
    assert artifact["parameter_counts"]["frozen_parameter_count"] == 4
    assert artifact["parameter_counts"]["by_optimizer_group"]["token_embeddings"] == {
        "parameter_count": 1,
        "scalar_count": 4,
    }
    assert [group["group_name"] for group in artifact["optimizer_groups"]] == [
        "adapter.language",
        "adapter.vision",
        "adapter.aligner",
        "token_embeddings",
    ]
    assert artifact["unmatched_trainable_names"] == []
    assert {
        summary["reason"] for summary in artifact["frozen_reason_summaries"]
    } == {
        "base_towers_frozen_v1",
        "selected_embedding_delta_active",
    }

    output_path = write_trainable_surface_receipt(
        receipt,
        tmp_path / "trainable_surface.json",
    )

    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact


def test_trainable_surface_receipt_rejects_optimizer_plan_missing_trainable() -> None:
    model = FakeTrainableSurface(
        adapter_targets=("model.language_model.q_proj",),
        token_delta=True,
    )
    adapter_receipt = _adapter_receipt(targets=("model.language_model.q_proj",))
    token_receipt = _token_receipt(delta_names=("embed_tokens.shared_embed_delta",))
    optimizer_plan = build_optimizer_group_plan(
        model,
        _optimizer_config(language=_group(), token_embeddings=_group()),
        adapter_receipt=adapter_receipt,
        special_token_receipt=token_receipt,
    )
    forged_plan = OptimizerGroupPlan(
        groups=tuple(
            group
            for group in optimizer_plan.groups
            if group.group_name != "token_embeddings"
        ),
        parameters_by_name=optimizer_plan.parameters_by_name,
    )

    with pytest.raises(RuntimeContractError) as exc_info:
        build_trainable_surface_receipt(
            model,
            adapter_receipt=adapter_receipt,
            special_token_receipt=token_receipt,
            optimizer_group_plan=forged_plan,
        )

    assert exc_info.value.code == "trainable_surface.optimizer_plan_mismatch"
    assert exc_info.value.context["missing_from_optimizer"] == [
        "embed_tokens.shared_embed_delta"
    ]


def test_trainable_surface_receipt_rejects_optimizer_plan_unknown_parameter() -> None:
    model = FakeTrainableSurface(adapter_targets=("model.language_model.q_proj",))
    adapter_receipt = _adapter_receipt(targets=("model.language_model.q_proj",))
    optimizer_plan = build_optimizer_group_plan(
        model,
        _optimizer_config(language=_group()),
        adapter_receipt=adapter_receipt,
        special_token_receipt=None,
    )
    original_group = optimizer_plan.groups[0]
    forged_plan = OptimizerGroupPlan(
        groups=(
            type(original_group)(
                group_name=original_group.group_name,
                lr=original_group.lr,
                weight_decay=original_group.weight_decay,
                parameter_names=(*original_group.parameter_names, "not.a.parameter"),
            ),
        ),
        parameters_by_name=optimizer_plan.parameters_by_name,
    )

    with pytest.raises(RuntimeContractError) as exc_info:
        build_trainable_surface_receipt(
            model,
            adapter_receipt=adapter_receipt,
            special_token_receipt=None,
            optimizer_group_plan=forged_plan,
        )

    assert exc_info.value.code == "trainable_surface.optimizer_plan_mismatch"
    assert exc_info.value.context["extra_in_optimizer"] == ["not.a.parameter"]


def test_trainable_surface_receipt_rejects_stale_optimizer_parameter_object() -> None:
    model = FakeTrainableSurface(adapter_targets=("model.language_model.q_proj",))
    adapter_receipt = _adapter_receipt(targets=("model.language_model.q_proj",))
    optimizer_plan = build_optimizer_group_plan(
        model,
        _optimizer_config(language=_group()),
        adapter_receipt=adapter_receipt,
        special_token_receipt=None,
    )
    stale_name = optimizer_plan.groups[0].parameter_names[0]
    forged_parameters_by_name = dict(optimizer_plan.parameters_by_name)
    forged_parameters_by_name[stale_name] = nn.Parameter(torch.zeros(()))
    forged_plan = OptimizerGroupPlan(
        groups=optimizer_plan.groups,
        parameters_by_name=forged_parameters_by_name,
    )

    with pytest.raises(RuntimeContractError) as exc_info:
        build_trainable_surface_receipt(
            model,
            adapter_receipt=adapter_receipt,
            special_token_receipt=None,
            optimizer_group_plan=forged_plan,
        )

    assert exc_info.value.code == "trainable_surface.optimizer_plan_mismatch"
    assert exc_info.value.context["parameter_object_mismatch"] == [stale_name]


def test_write_trainable_surface_receipt_rejects_non_standard_json(
    tmp_path,
) -> None:
    receipt = TrainableSurfaceReceipt(
        phase="before_first_backward",
        frozen_towers=(),
        trainable_towers=("adapter.language",),
        adapter_targets={},
        selected_embedding_tokens={},
        parameter_counts={},
        optimizer_groups=(
            {
                "group_name": "adapter.language",
                "lr": float("inf"),
                "weight_decay": 0.0,
                "parameter_count": 1,
                "parameter_names": ["x"],
            },
        ),
        unmatched_trainable_names=(),
        frozen_reason_summaries=(
            FrozenReasonSummary(
                reason="base_towers_frozen_v1",
                parameter_count=0,
                scalar_count=0,
                parameter_names_preview=(),
                context={},
            ),
        ),
    )

    with pytest.raises(ValueError):
        write_trainable_surface_receipt(receipt, tmp_path / "receipt.json")


class ParamLeaf(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))


class FakeTrainableSurface(nn.Module):
    def __init__(
        self,
        *,
        adapter_targets: tuple[str, ...] = (),
        token_delta: bool = False,
    ) -> None:
        super().__init__()
        self.base_model = nn.Module()
        self.base_model.model = nn.Module()
        self.language_frozen = nn.Parameter(torch.zeros(2), requires_grad=False)
        self.vision_frozen = nn.Parameter(torch.zeros(3), requires_grad=False)
        self.aligner_frozen = nn.Parameter(torch.zeros(4), requires_grad=False)
        self.embed_base_frozen = nn.Parameter(torch.zeros(5), requires_grad=False)
        for target in adapter_targets:
            self._install_adapter_target(target)
        if token_delta:
            self.embed_tokens = nn.Module()
            self.embed_tokens.shared_embed_delta = nn.Parameter(torch.zeros(2, 2))

    def _install_adapter_target(self, target: str) -> None:
        owner = self.base_model.model
        parts = target.split(".")
        for part in parts[:-1]:
            if not hasattr(owner, part):
                setattr(owner, part, nn.Module())
            owner = getattr(owner, part)
        leaf = nn.Module()
        leaf.lora_A = nn.Module()
        leaf.lora_B = nn.Module()
        leaf.lora_magnitude_vector = nn.Module()
        leaf.lora_A.default = ParamLeaf()
        leaf.lora_B.default = ParamLeaf()
        leaf.lora_magnitude_vector.default = ParamLeaf()
        setattr(owner, parts[-1], leaf)


def _adapter_receipt(*, targets: tuple[str, ...]) -> DoraAdapterSetupReceipt:
    trainable_names = tuple(
        f"base_model.model.{target}.{kind}.default.weight"
        for target in targets
        for kind in ("lora_A", "lora_B", "lora_magnitude_vector")
    )
    return DoraAdapterSetupReceipt(
        mode="initialize_new",
        adapter_type="dora",
        adapter_name="default",
        adapter_path=None,
        base_model_path=None,
        target_discovery=DoraTargetDiscoveryReceipt(
            target_policy="all_linear",
            target_towers=("language", "vision", "aligner"),
            matched_modules=targets,
            counts_by_tower={
                "language": sum("language_model." in target for target in targets),
                "vision": sum(
                    ".visual." in target and ".merger." not in target
                    for target in targets
                ),
                "aligner": sum(".visual.merger" in target for target in targets),
            },
            lm_head_seen=True,
            lm_head_excluded=True,
        ),
        peft_config={"use_dora": True},
        trainable_names=trainable_names,
        trainable_counts={
            "total": len(trainable_names),
            "lora_A": len(targets),
            "lora_B": len(targets),
            "lora_magnitude_vector": len(targets),
        },
        package_versions={},
    )


def _token_receipt(
    *,
    delta_names: tuple[str, ...],
) -> SpecialTokenEmbeddingInstallReceipt:
    return SpecialTokenEmbeddingInstallReceipt(
        semantics="additive_delta",
        tensor_key="shared_embed_delta",
        tie_word_embeddings=True,
        token_selection=SpecialTokenSelection(
            token_strings=("<a>", "<b>"),
            token_ids=(2, 3),
        ),
        delta_shape=(2, 2),
        delta_dtype="float32",
        delta_parameter_names=delta_names,
        base_embedding_parameter_name="embed_tokens.base.weight",
        base_lm_head_parameter_name="lm_head.base.weight",
    )


def _optimizer_config(
    *,
    language: OptimizerGroupConfig | None = None,
    vision: OptimizerGroupConfig | None = None,
    aligner: OptimizerGroupConfig | None = None,
    token_embeddings: OptimizerGroupConfig | None = None,
) -> OptimizerConfig:
    return OptimizerConfig(
        name="adamw_torch",
        betas=(0.9, 0.95),
        epsilon=1.0e-8,
        groups=OptimizerGroupsConfig(
            adapters=AdapterOptimizerGroupsConfig(
                language=language,
                vision=vision,
                aligner=aligner,
            ),
            token_embeddings=token_embeddings or _group(lr=5.0e-4),
        ),
        scheduler=SchedulerConfig(name="cosine_with_warmup", warmup_ratio=0.03),
    )


def _group(
    *,
    lr: float = 1.0e-4,
    weight_decay: float = 0.0,
) -> OptimizerGroupConfig:
    return OptimizerGroupConfig(lr=lr, weight_decay=weight_decay)
