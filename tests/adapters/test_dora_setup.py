from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

from src.adapters.dora import (
    discover_dora_targets,
    inspect_dora_adapter_payload,
    merge_dora_adapter_for_execution,
    setup_dora_adapter,
    _validate_trainable_dora_surface,
)
from src.adapters.source_gates import (
    AdapterSetupPlan,
    AdapterSourceGateEvidence,
    build_adapter_setup_plan,
)
from src.common.errors import RuntimeContractError
from src.config.models import AdapterConfig


LOCAL_QWEN_MODEL = Path(
    "/data/CoordExp/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent"
)


def test_execution_dora_identity_is_path_independent_and_binds_payload(
    tmp_path: Path,
) -> None:
    adapter_dir = _write_source_adapter(tmp_path, target_towers=("language",))
    config_path = adapter_dir / "adapter_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["base_model_name_or_path"] = "/models/qwen-base"
    config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")

    identity = inspect_dora_adapter_payload(
        adapter_dir,
        expected_base_model_path=Path("/models/qwen-base"),
    )
    copied_dir = tmp_path / "copied-adapter"
    shutil.copytree(adapter_dir, copied_dir)
    copied = inspect_dora_adapter_payload(copied_dir)

    assert identity["kind"] == "dora_adapter"
    assert identity["fingerprint"] == copied["fingerprint"]
    assert identity["root"] != copied["root"]
    assert identity["semantic_identity"]["peft_type"] == "LORA"
    assert identity["tensor_manifest"]["lora_magnitude_vector_count"] == 1

    tensors_path = copied_dir / "adapter_model.safetensors"
    with safe_open(str(tensors_path), framework="pt", device="cpu") as handle:
        tensors = {key: handle.get_tensor(key) for key in handle.keys()}
    first_key = sorted(tensors)[0]
    tensors[first_key] = tensors[first_key].clone()
    tensors[first_key].view(-1)[0] += 1
    save_file(tensors, tensors_path)
    changed = inspect_dora_adapter_payload(copied_dir)
    assert changed["fingerprint"] != identity["fingerprint"]


def test_merge_dora_adapter_for_execution_is_frozen_safe_and_residue_free(
    tmp_path: Path,
) -> None:
    adapter_dir = _write_source_adapter(tmp_path, target_towers=("language",))

    merged, receipt = merge_dora_adapter_for_execution(
        TinyQwenLikeModel(),
        adapter_dir,
    )

    assert type(merged) is TinyQwenLikeModel
    assert receipt["status"] == "merged"
    assert receipt["load"]["is_trainable"] is False
    assert receipt["load"]["autocast_adapter_dtype"] is False
    assert receipt["load"]["trainable_parameter_count"] == 0
    assert receipt["load"]["status"]["active_adapters"] == ["default"]
    assert receipt["load"]["status"]["merged_adapters"] == []
    assert receipt["merge"] == {
        "safe_merge": True,
        "adapter_names": ["default"],
    }
    assert not any(receipt["residue"].values())
    assert not hasattr(merged, "peft_config")
    assert all(not parameter.requires_grad for parameter in merged.parameters())
    assert all(
        marker not in name.lower()
        for name, _ in merged.named_parameters()
        for marker in ("lora_", "dora", "magnitude_vector")
    )


def test_merge_dora_adapter_rejects_payload_drift_after_inspection(
    tmp_path: Path,
) -> None:
    adapter_dir = _write_source_adapter(tmp_path, target_towers=("language",))
    identity = inspect_dora_adapter_payload(adapter_dir)
    tensor_path = adapter_dir / "adapter_model.safetensors"
    with safe_open(str(tensor_path), framework="pt", device="cpu") as handle:
        tensors = {key: handle.get_tensor(key) for key in handle.keys()}
    first_key = sorted(tensors)[0]
    tensors[first_key] = tensors[first_key].clone()
    tensors[first_key].view(-1)[0] += 1
    save_file(tensors, tensor_path)

    with pytest.raises(RuntimeContractError) as exc_info:
        merge_dora_adapter_for_execution(
            TinyQwenLikeModel(),
            adapter_dir,
            expected_identity=identity,
        )
    assert exc_info.value.code == "adapter.execution_identity_mismatch"


def test_execution_dora_inspection_rejects_non_dora_config(tmp_path: Path) -> None:
    adapter_dir = _write_source_adapter(tmp_path, target_towers=("language",))
    config_path = adapter_dir / "adapter_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["use_dora"] = False
    config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")

    with pytest.raises(RuntimeContractError) as exc_info:
        inspect_dora_adapter_payload(adapter_dir)

    assert exc_info.value.code == "adapter.execution_not_dora"


def test_dora_target_discovery_is_tower_scoped_and_excludes_lm_head() -> None:
    plan = _setup_plan(target_towers=("language", "vision", "aligner"))
    model = TinyQwenLikeModel()

    receipt = discover_dora_targets(model, plan)

    assert receipt.target_policy == "all_linear"
    assert receipt.target_towers == ("language", "vision", "aligner")
    assert receipt.matched_modules == (
        "model.language_model.q_proj",
        "model.visual.block_proj",
        "model.visual.merger.mlp",
    )
    assert receipt.counts_by_tower == {"language": 1, "vision": 1, "aligner": 1}
    assert receipt.lm_head_seen is True
    assert receipt.lm_head_excluded is True


def test_real_qwen_meta_target_discovery_matches_language_tower_contract() -> None:
    from accelerate import init_empty_weights
    from transformers import AutoConfig, Qwen3VLForConditionalGeneration

    config = AutoConfig.from_pretrained(
        LOCAL_QWEN_MODEL,
        local_files_only=True,
        trust_remote_code=True,
    )
    plan = _setup_plan(target_towers=("language",))

    with init_empty_weights():
        model = Qwen3VLForConditionalGeneration._from_config(config)
    receipt = discover_dora_targets(model, plan)

    assert receipt.counts_by_tower["language"] == 196
    assert (
        receipt.matched_modules[0] == "model.language_model.layers.0.self_attn.q_proj"
    )
    assert "lm_head" not in receipt.matched_modules
    assert receipt.lm_head_seen is True
    assert receipt.lm_head_excluded is True


def test_real_qwen_meta_dora_setup_counts_discovered_modules_not_peft_compact_targets() -> (
    None
):
    from accelerate import init_empty_weights
    from transformers import AutoConfig, Qwen3VLForConditionalGeneration

    config = AutoConfig.from_pretrained(
        LOCAL_QWEN_MODEL,
        local_files_only=True,
        trust_remote_code=True,
    )
    plan = _setup_plan(target_towers=("language",))

    with init_empty_weights():
        model = Qwen3VLForConditionalGeneration._from_config(config)
        result = setup_dora_adapter(model, plan)

    artifact = result.receipt.to_artifact_dict()
    assert artifact["target_discovery"]["counts_by_tower"]["language"] == 196
    assert artifact["trainable_counts"]["lora_A"] == 196
    assert artifact["trainable_counts"]["lora_B"] == 196
    assert artifact["trainable_counts"]["lora_magnitude_vector"] == 196
    assert len(artifact["peft_config"]["target_modules"]) < 196


def test_real_qwen_meta_target_discovery_covers_vision_and_aligner_towers() -> None:
    from accelerate import init_empty_weights
    from transformers import AutoConfig, Qwen3VLForConditionalGeneration

    config = AutoConfig.from_pretrained(
        LOCAL_QWEN_MODEL,
        local_files_only=True,
        trust_remote_code=True,
    )
    plan = _setup_plan(target_towers=("vision", "aligner"))

    with init_empty_weights():
        model = Qwen3VLForConditionalGeneration._from_config(config)
    receipt = discover_dora_targets(model, plan)

    assert receipt.counts_by_tower == {"vision": 96, "aligner": 8}
    assert "model.visual.blocks.0.attn.qkv" in receipt.matched_modules
    assert "model.visual.merger.linear_fc1" in receipt.matched_modules
    assert any("deepstack_merger_list" in name for name in receipt.matched_modules)
    assert receipt.lm_head_seen is True
    assert receipt.lm_head_excluded is True


def test_fresh_dora_setup_uses_peft_dora_and_trainable_magnitude_vectors() -> None:
    plan = _setup_plan(target_towers=("language",))
    model = TinyQwenLikeModel()

    result = setup_dora_adapter(model, plan)

    artifact = result.receipt.to_artifact_dict()
    assert artifact["mode"] == "initialize_new"
    assert artifact["peft_config"]["use_dora"] is True
    assert artifact["peft_config"]["target_modules"] == ["model.language_model.q_proj"]
    assert artifact["trainable_counts"]["lora_A"] == 1
    assert artifact["trainable_counts"]["lora_B"] == 1
    assert artifact["trainable_counts"]["lora_magnitude_vector"] == 1
    assert artifact["target_discovery"]["lm_head_excluded"] is True
    trainable_names = artifact["trainable_names"]
    assert any("lora_A" in name for name in trainable_names)
    assert any("lora_B" in name for name in trainable_names)
    assert any("lora_magnitude_vector" in name for name in trainable_names)

    output = result.model(torch.ones(2, 4))
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("bias", ["lora_only", "all"])
def test_fresh_dora_setup_rejects_trainable_bias_surface(bias: str) -> None:
    plan = _setup_plan(target_towers=("language",), bias=bias)

    with pytest.raises(RuntimeContractError) as exc_info:
        setup_dora_adapter(TinyBiasfulQwenLikeModel(), plan)

    assert exc_info.value.code == "adapter.dora_trainable_surface"
    assert "bias" in str(exc_info.value.context["unexpected_trainable_names"])


def test_existing_dora_adapter_path_loads_as_trainable_dora(tmp_path: Path) -> None:
    fresh_plan = _setup_plan(target_towers=("language",))
    fresh = setup_dora_adapter(TinyQwenLikeModel(), fresh_plan)
    adapter_dir = tmp_path / "adapter"
    fresh.model.save_pretrained(adapter_dir)

    load_plan = _setup_plan(
        target_towers=("language",),
        adapter_path=str(adapter_dir),
        base_model_path=Path("/models/qwen-base"),
    )
    loaded = setup_dora_adapter(TinyQwenLikeModel(), load_plan)

    artifact = loaded.receipt.to_artifact_dict()
    assert artifact["mode"] == "load_existing"
    assert artifact["adapter_identity"]["path"] == str(adapter_dir)
    assert artifact["base_model_identity"]["path"] == "/models/qwen-base"
    assert artifact["peft_config"]["use_dora"] is True
    assert artifact["trainable_counts"]["lora_magnitude_vector"] == 1
    assert torch.isfinite(loaded.model(torch.ones(2, 4))).all()


def test_existing_dora_adapter_load_accepts_peft_sorted_target_modules(
    tmp_path: Path,
) -> None:
    fresh_plan = _setup_plan(target_towers=("language",))
    fresh = setup_dora_adapter(TinyTwoLanguageTargetModel(), fresh_plan)
    adapter_dir = tmp_path / "adapter"
    fresh.model.save_pretrained(adapter_dir)

    load_plan = _setup_plan(
        target_towers=("language",),
        adapter_path=str(adapter_dir),
        base_model_path=Path("/models/qwen-base"),
    )
    loaded = setup_dora_adapter(TinyTwoLanguageTargetModel(), load_plan)

    artifact = loaded.receipt.to_artifact_dict()
    assert artifact["target_discovery"]["matched_modules"] == [
        "model.language_model.z_proj",
        "model.language_model.a_proj",
    ]
    assert artifact["peft_config"]["target_modules"] == [
        "model.language_model.a_proj",
        "model.language_model.z_proj",
    ]
    assert artifact["trainable_counts"]["lora_magnitude_vector"] == 2
    assert torch.isfinite(loaded.model(torch.ones(2, 4))).all()


def test_existing_dora_adapter_load_rejects_base_model_mismatch(
    tmp_path: Path,
) -> None:
    fresh_plan = _setup_plan(target_towers=("language",))
    fresh = setup_dora_adapter(TinyQwenLikeModel(), fresh_plan)
    adapter_dir = tmp_path / "adapter"
    fresh.model.save_pretrained(adapter_dir)
    adapter_config_path = adapter_dir / "adapter_config.json"
    adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    adapter_config["base_model_name_or_path"] = "/models/other-qwen-base"
    adapter_config_path.write_text(
        json.dumps(adapter_config, sort_keys=True),
        encoding="utf-8",
    )

    load_plan = _setup_plan(
        target_towers=("language",),
        adapter_path=str(adapter_dir),
        base_model_path=Path("/models/qwen-base"),
    )

    with pytest.raises(RuntimeContractError) as exc_info:
        setup_dora_adapter(TinyQwenLikeModel(), load_plan)

    assert exc_info.value.code == "adapter.loaded_base_model_mismatch"
    assert exc_info.value.context["loaded_base_model_name_or_path"] == (
        "/models/other-qwen-base"
    )
    assert exc_info.value.context["requested_base_model_path"] == "/models/qwen-base"


def test_existing_dora_adapter_load_rejects_base_model_class_mismatch(
    tmp_path: Path,
) -> None:
    fresh_plan = _setup_plan(target_towers=("language",))
    fresh = setup_dora_adapter(TinyQwenLikeModel(), fresh_plan)
    adapter_dir = tmp_path / "adapter"
    fresh.model.save_pretrained(adapter_dir)
    adapter_config_path = adapter_dir / "adapter_config.json"
    adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    adapter_config["auto_mapping"]["base_model_class"] = "OtherQwenLikeModel"
    adapter_config_path.write_text(
        json.dumps(adapter_config, sort_keys=True),
        encoding="utf-8",
    )

    load_plan = _setup_plan(
        target_towers=("language",),
        adapter_path=str(adapter_dir),
        base_model_path=Path("/models/qwen-base"),
    )

    with pytest.raises(RuntimeContractError) as exc_info:
        setup_dora_adapter(TinyQwenLikeModel(), load_plan)

    assert exc_info.value.code == "adapter.loaded_base_model_class_mismatch"
    assert exc_info.value.context["loaded_base_model_class"] == "OtherQwenLikeModel"
    assert exc_info.value.context["current_base_model_class"] == "TinyQwenLikeModel"


def test_existing_dora_adapter_load_rejects_target_mismatch(tmp_path: Path) -> None:
    fresh_plan = _setup_plan(target_towers=("language",))
    fresh = setup_dora_adapter(TinyQwenLikeModel(), fresh_plan)
    adapter_dir = tmp_path / "adapter"
    fresh.model.save_pretrained(adapter_dir)

    load_plan = _setup_plan(
        target_towers=("vision",),
        adapter_path=str(adapter_dir),
        base_model_path=Path("/models/qwen-base"),
    )

    with pytest.raises(RuntimeContractError) as exc_info:
        setup_dora_adapter(TinyQwenLikeModel(), load_plan)

    assert exc_info.value.code == "adapter.loaded_target_mismatch"
    assert exc_info.value.context["loaded_target_modules"] == [
        "model.language_model.q_proj"
    ]
    assert exc_info.value.context["discovered_target_modules"] == [
        "model.visual.block_proj"
    ]


def test_existing_dora_adapter_load_rejects_lm_head_target(tmp_path: Path) -> None:
    from peft import LoraConfig, get_peft_model

    model = TinyQwenLikeModel()
    peft_model = get_peft_model(
        model,
        LoraConfig(
            r=2,
            lora_alpha=4,
            lora_dropout=0.0,
            bias="none",
            target_modules=["lm_head"],
            use_dora=True,
        ),
    )
    adapter_dir = tmp_path / "adapter"
    peft_model.save_pretrained(adapter_dir)

    load_plan = _setup_plan(
        target_towers=("language",),
        adapter_path=str(adapter_dir),
        base_model_path=Path("/models/qwen-base"),
    )

    with pytest.raises(RuntimeContractError) as exc_info:
        setup_dora_adapter(TinyQwenLikeModel(), load_plan)

    assert exc_info.value.code == "adapter.loaded_target_mismatch"
    assert exc_info.value.context["loaded_target_modules"] == ["lm_head"]


def test_warm_start_expand_dora_requires_source_adapter_and_embedding_payload() -> None:
    with pytest.raises(ValidationError):
        AdapterConfig(
            type="dora",
            seed_mode="warm_start_expand_dora",
            source_adapter_path="/tmp/source-adapter",
            target_towers=("language", "vision", "aligner"),
            target_modules="all_linear",
            rank=2,
            alpha=4,
            dropout=0.0,
        )

    adapter = AdapterConfig(
        type="dora",
        seed_mode="warm_start_expand_dora",
        source_adapter_path="/tmp/source-adapter",
        repaired_embedding_payload_path="/tmp/repaired-embedding-payload",
        target_towers=("language", "vision", "aligner"),
        target_modules="all_linear",
        rank=2,
        alpha=4,
        dropout=0.0,
    )

    plan = build_adapter_setup_plan(
        adapter,
        AdapterSourceGateEvidence(
            dora_source_study_passed=True,
            dora_probe_passed=True,
            dora_probe_receipt=_probe_receipt_for_towers(("language",)),
        ),
        base_model_path=Path("/models/qwen-base"),
    )

    assert plan.mode == "warm_start_expand_dora"
    assert plan.source_adapter_path == Path("/tmp/source-adapter")
    assert plan.repaired_embedding_payload_path == Path(
        "/tmp/repaired-embedding-payload"
    )
    assert plan.target_towers == ("language", "vision", "aligner")


def test_warm_start_expand_dora_uses_configured_target_subset() -> None:
    adapter = AdapterConfig(
        type="dora",
        seed_mode="warm_start_expand_dora",
        source_adapter_path="/tmp/source-adapter",
        repaired_embedding_payload_path="/tmp/repaired-embedding-payload",
        target_towers=("language", "vision"),
        target_modules="all_linear",
        rank=2,
        alpha=4,
        dropout=0.0,
    )

    plan = build_adapter_setup_plan(
        adapter,
        AdapterSourceGateEvidence(
            dora_source_study_passed=True,
            dora_probe_passed=True,
            dora_probe_receipt=_probe_receipt_for_towers(("language",)),
        ),
        base_model_path=Path("/models/qwen-base"),
    )

    assert plan.mode == "warm_start_expand_dora"
    assert plan.target_towers == ("language", "vision")


def test_warm_start_expand_dora_copies_language_tensors_and_initializes_new_towers(
    tmp_path: Path,
) -> None:
    source_dir = _write_source_adapter(tmp_path, target_towers=("language",))

    result = setup_dora_adapter(
        TinyQwenLikeModel(),
        _warm_start_plan(source_adapter_path=source_dir),
    )

    params = dict(result.model.named_parameters())
    assert torch.equal(
        params["base_model.model.model.language_model.q_proj.lora_A.default.weight"],
        torch.full((2, 4), 1.25),
    )
    assert torch.equal(
        params["base_model.model.model.language_model.q_proj.lora_B.default.weight"],
        torch.full((4, 2), 2.5),
    )
    assert torch.equal(
        params[
            "base_model.model.model.language_model.q_proj.lora_magnitude_vector.default.weight"
        ],
        torch.full((4,), 3.75),
    )

    warm_start = result.receipt.to_artifact_dict()["warm_start"]
    assert warm_start["seed_mode"] == "warm_start_expand_dora"
    assert warm_start["copied"] == {
        "lora_A": 1,
        "lora_B": 1,
        "dora_magnitude": 1,
    }
    assert warm_start["initialized"] == {"language": 0, "vision": 3, "aligner": 3}
    assert warm_start["repaired_embedding_payload_path"] == (
        "/tmp/repaired-embedding-payload"
    )
    initialized_targets = warm_start["initialized_target_tensors"]
    assert any("visual.block_proj.lora_A" in name for name in initialized_targets)
    assert any(
        "visual.merger.mlp.lora_magnitude_vector" in name
        for name in initialized_targets
    )
    assert not any("language_model.q_proj" in name for name in initialized_targets)


def test_warm_start_expand_dora_reuses_any_existing_target_and_initializes_missing(
    tmp_path: Path,
) -> None:
    source_dir = _write_source_adapter(tmp_path, target_towers=("language", "vision"))

    result = setup_dora_adapter(
        TinyQwenLikeModel(),
        _warm_start_plan(source_adapter_path=source_dir),
    )

    params = dict(result.model.named_parameters())
    assert torch.equal(
        params["base_model.model.model.language_model.q_proj.lora_A.default.weight"],
        torch.full((2, 4), 1.25),
    )
    assert torch.equal(
        params["base_model.model.model.visual.block_proj.lora_A.default.weight"],
        torch.full((2, 4), 1.25),
    )

    warm_start = result.receipt.to_artifact_dict()["warm_start"]
    assert warm_start["copied"] == {
        "lora_A": 2,
        "lora_B": 2,
        "dora_magnitude": 2,
    }
    assert warm_start["initialized"] == {"language": 0, "vision": 0, "aligner": 3}
    initialized_targets = warm_start["initialized_target_tensors"]
    assert any("visual.merger.mlp.lora_A" in name for name in initialized_targets)
    assert not any("visual.block_proj" in name for name in initialized_targets)


def test_warm_start_expand_dora_requires_language_magnitude_tensor(
    tmp_path: Path,
) -> None:
    source_dir = _write_source_adapter(tmp_path, target_towers=("language",))
    _rewrite_source_tensors(
        source_dir,
        remove_key_fragments=("language_model.q_proj.lora_magnitude_vector",),
    )

    with pytest.raises(RuntimeContractError) as exc_info:
        setup_dora_adapter(
            TinyQwenLikeModel(),
            _warm_start_plan(source_adapter_path=source_dir),
        )

    assert exc_info.value.code == "adapter.warm_start_partial_target_tensor"
    assert "lora_magnitude_vector" in str(exc_info.value.context["missing_source_keys"])


def test_dora_setup_fails_when_requested_tower_has_no_targets() -> None:
    plan = _setup_plan(target_towers=("vision",))
    model = TinyLanguageOnlyModel()

    with pytest.raises(RuntimeContractError) as exc_info:
        setup_dora_adapter(model, plan)

    assert exc_info.value.code == "adapter.target_discovery_empty"
    assert exc_info.value.context["target_tower"] == "vision"


def test_dora_trainable_surface_rejects_substring_impostor_name() -> None:
    with pytest.raises(RuntimeContractError) as exc_info:
        _validate_trainable_dora_surface(
            trainable_counts={
                "total": 3,
                "lora_A": 1,
                "lora_B": 1,
                "lora_magnitude_vector": 1,
            },
            expected_target_count=1,
            trainable_names=(
                "base_model.model.not_a_real_lora_A_param.weight",
                "base_model.model.target.lora_B.default.weight",
                "base_model.model.target.lora_magnitude_vector.default.weight",
            ),
        )

    assert exc_info.value.code == "adapter.dora_trainable_surface"
    assert exc_info.value.context["unexpected_trainable_names"] == [
        "base_model.model.not_a_real_lora_A_param.weight"
    ]


class TinyQwenLikeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.q_proj = nn.Linear(4, 4, bias=False)
        self.model.visual = nn.Module()
        self.model.visual.block_proj = nn.Linear(4, 4, bias=False)
        self.model.visual.merger = nn.Module()
        self.model.visual.merger.mlp = nn.Linear(4, 4, bias=False)
        self.lm_head = nn.Linear(4, 4, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.model.language_model.q_proj(hidden)


class TinyBiasfulQwenLikeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.q_proj = nn.Linear(4, 4, bias=True)
        self.lm_head = nn.Linear(4, 4, bias=True)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.model.language_model.q_proj(hidden)


class TinyTwoLanguageTargetModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.z_proj = nn.Linear(4, 4, bias=False)
        self.model.language_model.a_proj = nn.Linear(4, 4, bias=False)
        self.lm_head = nn.Linear(4, 4, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        language_model = self.model.language_model
        return language_model.z_proj(hidden) + language_model.a_proj(hidden)


class TinyLanguageOnlyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.q_proj = nn.Linear(4, 4, bias=False)
        self.lm_head = nn.Linear(4, 4, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.model.language_model.q_proj(hidden)


def _setup_plan(
    *,
    target_towers: tuple[str, ...],
    adapter_path: str | None = None,
    base_model_path: Path | None = None,
    bias: str = "none",
) -> AdapterSetupPlan:
    adapter = AdapterConfig(
        type="dora",
        path=adapter_path,
        target_towers=target_towers,
        target_modules="all_linear",
        rank=2,
        alpha=4,
        dropout=0.0,
        bias=bias,
    )
    return build_adapter_setup_plan(
        adapter,
        AdapterSourceGateEvidence(
            dora_source_study_passed=True,
            dora_probe_passed=True,
            dora_probe_receipt=_probe_receipt_for_towers(target_towers),
        ),
        base_model_path=base_model_path,
    )


def _write_source_adapter(
    tmp_path: Path,
    *,
    target_towers: tuple[str, ...],
) -> Path:
    source = setup_dora_adapter(
        TinyQwenLikeModel(),
        _setup_plan(target_towers=target_towers),
    )
    for name, parameter in source.model.named_parameters():
        if "lora_A" in name:
            parameter.data.fill_(1.25)
        elif "lora_B" in name:
            parameter.data.fill_(2.5)
        elif "lora_magnitude_vector" in name:
            parameter.data.fill_(3.75)
    adapter_dir = tmp_path / f"{'_'.join(target_towers)}_adapter"
    source.model.save_pretrained(adapter_dir)
    return adapter_dir


def _rewrite_source_tensors(
    adapter_dir: Path,
    *,
    remove_key_fragments: tuple[str, ...] = (),
) -> None:
    tensor_path = adapter_dir / "adapter_model.safetensors"
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(tensor_path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if not any(fragment in key for fragment in remove_key_fragments):
                tensors[key] = handle.get_tensor(key)
    save_file(tensors, tensor_path)


def _warm_start_plan(*, source_adapter_path: Path) -> AdapterSetupPlan:
    adapter = AdapterConfig(
        type="dora",
        seed_mode="warm_start_expand_dora",
        source_adapter_path=str(source_adapter_path),
        repaired_embedding_payload_path="/tmp/repaired-embedding-payload",
        target_towers=("language", "vision", "aligner"),
        target_modules="all_linear",
        rank=2,
        alpha=4,
        dropout=0.0,
        bias="none",
    )
    return build_adapter_setup_plan(
        adapter,
        AdapterSourceGateEvidence(
            dora_source_study_passed=True,
            dora_probe_passed=True,
            dora_probe_receipt=_probe_receipt_for_towers(("language",)),
        ),
        base_model_path=Path("/models/qwen-base"),
    )


def _probe_receipt_for_towers(target_towers: tuple[str, ...]) -> dict[str, object]:
    selected_targets = {
        "language": "model.language_model.q_proj",
        "vision": "model.visual.block_proj",
        "aligner": "model.visual.merger.mlp",
    }
    target_modules = [selected_targets[tower] for tower in target_towers]
    trainable_names = (
        [
            f"base_model.model.{target}.lora_A.default.weight"
            for target in target_modules
        ]
        + [
            f"base_model.model.{target}.lora_B.default.weight"
            for target in target_modules
        ]
        + [
            f"base_model.model.{target}.lora_magnitude_vector.default.weight"
            for target in target_modules
        ]
    )
    payload_keys = (
        [f"base_model.model.{target}.lora_A.weight" for target in target_modules]
        + [f"base_model.model.{target}.lora_B.weight" for target in target_modules]
        + [
            f"base_model.model.{target}.lora_magnitude_vector.weight"
            for target in target_modules
        ]
    )
    magnitude_names = [
        f"base_model.model.{target}.lora_magnitude_vector.default.weight"
        for target in target_modules
    ]
    target_count = len(target_modules)
    return {
        "public_adapter_type": "dora",
        "peft": {
            "config_class": "LoraConfig",
            "use_dora": True,
            "r": 2,
            "lora_alpha": 4,
            "lora_dropout": 0.0,
            "bias": "none",
        },
        "target_towers": list(target_towers),
        "target_policy": "all_linear",
        "lm_head_excluded": True,
        "selected_target_count": target_count,
        "selected_target_modules": target_modules,
        "trainable_names": trainable_names,
        "magnitude_vectors": {
            "trainable_names": magnitude_names,
            "reloaded_names": magnitude_names,
        },
        "save": {
            "adapter_config_use_dora": True,
            "adapter_payload_keys": payload_keys,
            "saved_lora_A_count": target_count,
            "saved_lora_B_count": target_count,
            "saved_lora_magnitude_vector_count": target_count,
        },
        "reload": {
            "success": True,
            "reloaded_magnitude_vector_count": target_count,
        },
        "equivalence": {
            "equivalent": True,
            "finite_original_eval_logits": True,
            "finite_reloaded_eval_logits": True,
        },
        "gradient_result": {
            "finite_logits": True,
            "finite_magnitude_vector_gradient": True,
        },
    }
