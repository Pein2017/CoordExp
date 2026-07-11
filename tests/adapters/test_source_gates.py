from __future__ import annotations

from pathlib import Path

import pytest

from src.adapters.source_gates import (
    AdapterSourceGateEvidence,
    build_adapter_setup_plan,
    load_default_adapter_source_gate_evidence,
    load_dora_probe_receipt,
)
from src.common.errors import RuntimeContractError
from src.config.models import AdapterConfig


def test_default_dora_source_gate_loads_historical_evidence() -> None:
    evidence = load_default_adapter_source_gate_evidence(
        Path(__file__).resolve().parents[2]
    )

    assert evidence.dora_source_study_passed is True
    assert evidence.dora_probe_passed is True
    assert evidence.dora_source_study_path is not None
    assert "docs/history/architecture/proposals" in str(
        evidence.dora_source_study_path
    )


def test_dora_setup_plan_requires_source_study_before_initialization() -> None:
    adapter = _adapter_config(path=None)
    evidence = AdapterSourceGateEvidence(
        dora_source_study_passed=False,
        dora_probe_passed=True,
        dora_probe_receipt={
            "public_adapter_type": "dora",
            "peft": {"config_class": "LoraConfig", "use_dora": True},
            "lm_head_excluded": True,
            "reload": {"success": True, "reloaded_magnitude_vector_count": 1},
            "save": {
                "adapter_config_use_dora": True,
                "saved_lora_A_count": 1,
                "saved_lora_B_count": 1,
                "saved_lora_magnitude_vector_count": 1,
            },
            "equivalence": {"equivalent": True, "finite_reloaded_eval_logits": True},
            "gradient_result": {"finite_magnitude_vector_gradient": True},
        },
    )

    with pytest.raises(RuntimeContractError) as exc_info:
        build_adapter_setup_plan(adapter, evidence)

    assert exc_info.value.code == "adapter.dora_source_gate_missing"
    assert "source-study" in exc_info.value.message
    assert exc_info.value.context["missing_gates"] == ["dora_source_study"]


def test_dora_setup_plan_records_initialization_receipt_after_gate_passes() -> None:
    adapter = _adapter_config(path=None)
    evidence = AdapterSourceGateEvidence(
        dora_source_study_passed=True,
        dora_probe_passed=True,
        dora_probe_receipt=_passing_dora_probe_receipt(),
        dora_source_study_path=Path("docs/source-studies/dora.md"),
        dora_probe_receipt_path=Path("outputs/probes/dora/receipt.json"),
    )

    plan = build_adapter_setup_plan(adapter, evidence)

    assert plan.mode == "initialize_new"
    assert plan.adapter_type == "dora"
    assert plan.adapter_path is None
    assert plan.target_policy == "all_linear"
    assert plan.target_towers == ("language",)
    artifact = plan.to_artifact_dict()
    assert artifact["source_gate"]["status"] == "passed"
    assert artifact["source_gate"]["peft_mechanism"] == "LoraConfig(use_dora=True)"
    assert artifact["source_gate"]["magnitude_vector"]["saved_count"] == 1
    assert artifact["source_gate"]["magnitude_vector"]["reload_count"] == 1
    assert artifact["source_gate"]["lm_head_excluded"] is True


def test_dora_setup_plan_records_existing_adapter_identity_after_gate_passes() -> None:
    adapter = _adapter_config(path="/tmp/adapter")
    evidence = AdapterSourceGateEvidence(
        dora_source_study_passed=True,
        dora_probe_passed=True,
        dora_probe_receipt=_passing_dora_probe_receipt(),
    )

    plan = build_adapter_setup_plan(
        adapter,
        evidence,
        base_model_path=Path("/models/qwen-base"),
    )

    assert plan.mode == "load_existing"
    assert plan.adapter_path == Path("/tmp/adapter")
    artifact = plan.to_artifact_dict()
    assert artifact["base_model_identity"]["path"] == "/models/qwen-base"
    assert artifact["adapter_identity"]["path"] == "/tmp/adapter"


@pytest.mark.parametrize("target_towers", [("vision",), ("aligner",), ("language", "vision")])
def test_dora_setup_plan_requires_receipt_to_cover_requested_towers(
    target_towers: tuple[str, ...],
) -> None:
    adapter = _adapter_config(path=None, target_towers=target_towers)
    evidence = AdapterSourceGateEvidence(
        dora_source_study_passed=True,
        dora_probe_passed=True,
        dora_probe_receipt=_passing_dora_probe_receipt(),
    )

    with pytest.raises(RuntimeContractError) as exc_info:
        build_adapter_setup_plan(adapter, evidence)

    assert exc_info.value.code == "adapter.dora_target_gate_missing"
    assert "target tower" in exc_info.value.message


def test_dora_setup_plan_requires_receipt_to_match_target_policy() -> None:
    adapter = _adapter_config(path=None)
    receipt = _passing_dora_probe_receipt()
    receipt["target_policy"] = "named_modules"
    evidence = AdapterSourceGateEvidence(
        dora_source_study_passed=True,
        dora_probe_passed=True,
        dora_probe_receipt=receipt,
    )

    with pytest.raises(RuntimeContractError) as exc_info:
        build_adapter_setup_plan(adapter, evidence)

    assert exc_info.value.code == "adapter.dora_target_gate_missing"
    assert "target policy" in exc_info.value.message


def test_dora_setup_plan_rejects_tower_label_without_matching_module_evidence() -> None:
    adapter = _adapter_config(path=None, target_towers=("vision",))
    receipt = _passing_dora_probe_receipt()
    receipt["target_towers"] = ["vision"]
    evidence = AdapterSourceGateEvidence(
        dora_source_study_passed=True,
        dora_probe_passed=True,
        dora_probe_receipt=receipt,
    )

    with pytest.raises(RuntimeContractError) as exc_info:
        build_adapter_setup_plan(adapter, evidence)

    assert exc_info.value.code == "adapter.dora_target_gate_missing"
    assert "module evidence" in exc_info.value.message


@pytest.mark.parametrize(
    ("target_tower", "selected_target"),
    [
        ("vision", "visual.blocks.0.attn.qkv"),
        ("aligner", "visual.merger.mlp.0"),
    ],
)
def test_dora_setup_plan_rejects_cross_tower_magnitude_evidence(
    target_tower: str,
    selected_target: str,
) -> None:
    adapter = _adapter_config(path=None, target_towers=(target_tower,))
    receipt = _passing_dora_probe_receipt()
    receipt["target_towers"] = [target_tower]
    receipt["selected_target_modules"] = [selected_target]
    evidence = AdapterSourceGateEvidence(
        dora_source_study_passed=True,
        dora_probe_passed=True,
        dora_probe_receipt=receipt,
    )

    with pytest.raises(RuntimeContractError) as exc_info:
        build_adapter_setup_plan(adapter, evidence)

    assert exc_info.value.code == "adapter.dora_probe_contract"
    assert "selected DoRA target" in exc_info.value.message


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("public_adapter_type", "dlora", "public adapter type"),
        ("peft.use_dora", False, "PEFT use_dora"),
        ("save.saved_lora_magnitude_vector_count", 0, "magnitude-vector"),
        ("reload.success", False, "reload"),
        ("lm_head_excluded", False, "lm_head"),
    ],
)
def test_dora_probe_receipt_must_prove_required_contracts(
    field: str,
    value: object,
    message: str,
) -> None:
    adapter = _adapter_config(path=None)
    receipt = _passing_dora_probe_receipt()
    _set_nested(receipt, field, value)
    evidence = AdapterSourceGateEvidence(
        dora_source_study_passed=True,
        dora_probe_passed=True,
        dora_probe_receipt=receipt,
    )

    with pytest.raises(RuntimeContractError) as exc_info:
        build_adapter_setup_plan(adapter, evidence)

    assert exc_info.value.code == "adapter.dora_probe_contract"
    assert message in exc_info.value.message


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("selected_target_count", 2),
        ("save.saved_lora_A_count", 2),
        ("save.saved_lora_B_count", 2),
        ("reload.reloaded_magnitude_vector_count", 2),
    ],
)
def test_dora_probe_receipt_counts_must_match_selected_target_count(
    field: str,
    value: object,
) -> None:
    adapter = _adapter_config(path=None)
    receipt = _passing_dora_probe_receipt()
    _set_nested(receipt, field, value)
    evidence = AdapterSourceGateEvidence(
        dora_source_study_passed=True,
        dora_probe_passed=True,
        dora_probe_receipt=receipt,
    )

    with pytest.raises(RuntimeContractError) as exc_info:
        build_adapter_setup_plan(adapter, evidence)

    assert exc_info.value.code == "adapter.dora_probe_contract"
    assert "DoRA target count coverage" in exc_info.value.message


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("save.adapter_payload_keys", ["base_model.model.foo.lora_A.default.weight"]),
        ("trainable_names", ["base_model.model.foo.lora_A.default.weight"]),
        ("magnitude_vectors.reloaded_names", []),
    ],
)
def test_dora_probe_receipt_requires_magnitude_vector_name_evidence(
    field: str,
    value: object,
) -> None:
    adapter = _adapter_config(path=None)
    receipt = _passing_dora_probe_receipt()
    _set_nested(receipt, field, value)
    evidence = AdapterSourceGateEvidence(
        dora_source_study_passed=True,
        dora_probe_passed=True,
        dora_probe_receipt=receipt,
    )

    with pytest.raises(RuntimeContractError) as exc_info:
        build_adapter_setup_plan(adapter, evidence)

    assert exc_info.value.code == "adapter.dora_probe_contract"
    assert "lora_magnitude_vector" in exc_info.value.message


def test_load_dora_probe_receipt_reads_json_with_path_context(tmp_path: Path) -> None:
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text('{"public_adapter_type": "dora"}', encoding="utf-8")

    receipt = load_dora_probe_receipt(receipt_path)

    assert receipt == {"public_adapter_type": "dora"}


def _adapter_config(
    path: str | None,
    *,
    target_towers: tuple[str, ...] = ("language",),
) -> AdapterConfig:
    return AdapterConfig(
        type="dora",
        path=path,
        target_towers=target_towers,
        target_modules="all_linear",
        rank=8,
        alpha=16,
        dropout=0.0,
        bias="none",
    )


def _passing_dora_probe_receipt() -> dict[str, object]:
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
        "target_towers": ["language"],
        "target_policy": "all_linear",
        "lm_head_excluded": True,
        "selected_target_count": 1,
        "matched_language_target_count": 196,
        "selected_target_modules": ["model.language_model.layers.0.self_attn.q_proj"],
        "trainable_names": [
            "base_model.model.model.language_model.layers.0.self_attn.q_proj."
            "lora_A.default.weight",
            "base_model.model.model.language_model.layers.0.self_attn.q_proj."
            "lora_B.default.weight",
            "base_model.model.model.language_model.layers.0.self_attn.q_proj."
            "lora_magnitude_vector.default.weight",
        ],
        "magnitude_vectors": {
            "trainable_names": [
                "base_model.model.model.language_model.layers.0.self_attn.q_proj."
                "lora_magnitude_vector.default.weight"
            ],
            "reloaded_names": [
                "base_model.model.model.language_model.layers.0.self_attn.q_proj."
                "lora_magnitude_vector.default.weight"
            ],
        },
        "save": {
            "adapter_config_use_dora": True,
            "adapter_payload_keys": [
                "base_model.model.model.language_model.layers.0.self_attn.q_proj."
                "lora_A.weight",
                "base_model.model.model.language_model.layers.0.self_attn.q_proj."
                "lora_B.weight",
                "base_model.model.model.language_model.layers.0.self_attn.q_proj."
                "lora_magnitude_vector.weight",
            ],
            "saved_lora_A_count": 1,
            "saved_lora_B_count": 1,
            "saved_lora_magnitude_vector_count": 1,
        },
        "reload": {
            "success": True,
            "reloaded_magnitude_vector_count": 1,
        },
        "equivalence": {
            "equivalent": True,
            "finite_original_eval_logits": True,
            "finite_reloaded_eval_logits": True,
            "max_abs_diff": 0.0,
        },
        "gradient_result": {
            "finite_logits": True,
            "finite_magnitude_vector_gradient": True,
        },
    }


def _set_nested(payload: dict[str, object], dotted: str, value: object) -> None:
    parts = dotted.split(".")
    current: dict[str, object] = payload
    for part in parts[:-1]:
        child = current[part]
        assert isinstance(child, dict)
        current = child
    current[parts[-1]] = value
