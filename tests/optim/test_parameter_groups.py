from __future__ import annotations

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
from src.optim.parameter_groups import build_optimizer_group_plan
from src.qwen.special_token_embeddings import (
    SpecialTokenEmbeddingInstallReceipt,
    SpecialTokenSelection,
)


def test_optimizer_group_plan_matches_adapter_towers_and_token_embeddings() -> None:
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

    plan = build_optimizer_group_plan(
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

    artifact = plan.to_artifact_dict()
    assert [group["group_name"] for group in artifact["groups"]] == [
        "adapter.language",
        "adapter.vision",
        "adapter.aligner",
        "token_embeddings",
    ]
    assert artifact["groups"][0]["parameter_count"] == 3
    assert artifact["groups"][1]["parameter_count"] == 3
    assert artifact["groups"][2]["parameter_count"] == 3
    assert artifact["groups"][3]["parameter_names"] == [
        "embed_tokens.shared_embed_delta"
    ]
    assert artifact["unmatched_trainable_names"] == []
    torch_groups = plan.to_torch_param_groups()
    assert torch_groups[0]["lr"] == pytest.approx(1.0e-4)
    assert torch_groups[0]["weight_decay"] == pytest.approx(0.01)
    assert torch_groups[3]["name"] == "token_embeddings"
    assert torch_groups[3]["params"][0] is model.embed_tokens.shared_embed_delta


def test_optimizer_group_plan_requires_explicit_lr_for_trainable_tower() -> None:
    model = FakeTrainableSurface(adapter_targets=("model.language_model.q_proj",))

    with pytest.raises(RuntimeContractError) as exc_info:
        build_optimizer_group_plan(
            model,
            _optimizer_config(language=None),
            adapter_receipt=_adapter_receipt(targets=("model.language_model.q_proj",)),
            special_token_receipt=None,
        )

    assert exc_info.value.code == "optimizer.group_missing"
    assert exc_info.value.context["group_name"] == "adapter.language"


def test_optimizer_group_plan_rejects_unmatched_trainable_base_weight() -> None:
    model = FakeTrainableSurface(
        adapter_targets=("model.language_model.q_proj",),
        extra_trainable=True,
    )

    with pytest.raises(RuntimeContractError) as exc_info:
        build_optimizer_group_plan(
            model,
            _optimizer_config(language=_group()),
            adapter_receipt=_adapter_receipt(targets=("model.language_model.q_proj",)),
            special_token_receipt=None,
        )

    assert exc_info.value.code == "optimizer.trainable_unmatched"
    assert exc_info.value.context["unmatched_trainable_names"] == ["extra_weight"]


def test_optimizer_group_plan_rejects_adapter_receipt_parameter_missing() -> None:
    model = FakeTrainableSurface(adapter_targets=("model.language_model.q_proj",))
    model.base_model.model.model.language_model.q_proj.lora_A.default.weight.requires_grad_(False)

    with pytest.raises(RuntimeContractError) as exc_info:
        build_optimizer_group_plan(
            model,
            _optimizer_config(language=_group()),
            adapter_receipt=_adapter_receipt(targets=("model.language_model.q_proj",)),
            special_token_receipt=None,
        )

    assert exc_info.value.code == "optimizer.receipt_parameter_missing"
    assert exc_info.value.context["group_name"] == "adapter"
    assert exc_info.value.context["parameter_name"] == (
        "base_model.model.model.language_model.q_proj.lora_A.default.weight"
    )


def test_optimizer_group_plan_rejects_duplicate_group_match() -> None:
    model = FakeTrainableSurface(
        adapter_targets=("model.language_model.q_proj",),
        token_delta=True,
    )
    adapter_name = (
        "base_model.model.model.language_model.q_proj.lora_A.default.weight"
    )

    with pytest.raises(RuntimeContractError) as exc_info:
        build_optimizer_group_plan(
            model,
            _optimizer_config(language=_group()),
            adapter_receipt=_adapter_receipt(targets=("model.language_model.q_proj",)),
            special_token_receipt=_token_receipt(
                delta_names=(adapter_name, "embed_tokens.shared_embed_delta")
            ),
        )

    assert exc_info.value.code == "optimizer.trainable_duplicate_match"
    assert exc_info.value.context["parameter_name"] == adapter_name
    assert exc_info.value.context["matched_groups"] == [
        "adapter.language",
        "token_embeddings",
    ]


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
        extra_trainable: bool = False,
    ) -> None:
        super().__init__()
        self.base_model = nn.Module()
        self.base_model.model = nn.Module()
        self._install_frozen_base()
        for target in adapter_targets:
            self._install_adapter_target(target)
        if token_delta:
            self.embed_tokens = nn.Module()
            self.embed_tokens.shared_embed_delta = nn.Parameter(torch.zeros(()))
        if extra_trainable:
            self.extra_weight = nn.Parameter(torch.ones(()))

    def _install_frozen_base(self) -> None:
        self.frozen_base = nn.Parameter(torch.zeros(()), requires_grad=False)

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
        token_selection=SpecialTokenSelection(token_strings=("<a>",), token_ids=(2,)),
        delta_shape=(1, 4),
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
