from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.config.loader import ConfigLoader
from src.config.schema import LatestDetectionTrainingConfig
from src.data_collators.batch_extras_collator import build_batch_extras_collator
from src.detection.dataset import (
    DETECTION_DROPPED_BEFORE_MODEL_KEYS,
    REGISTERED_DETECTION_SIDECAR_KEYS,
    strip_non_model_detection_sidecars,
)
from src.sft import (
    _assert_latest_detection_runtime_supported,
    _latest_detection_mode,
    _latest_detection_runtime_custom_shim,
    _resolve_recursive_detection_ce_cfg,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SFT_PATH = REPO_ROOT / "src" / "sft.py"
RUNTIME_PATH = REPO_ROOT / "src" / "detection" / "runtime.py"


def _prod_latest_detection_config() -> LatestDetectionTrainingConfig:
    config_path = (
        REPO_ROOT
        / "configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2.yaml"
    )
    cfg = ConfigLoader.load_materialized_training_config(str(config_path))
    assert isinstance(cfg, LatestDetectionTrainingConfig)
    return cfg


def test_sft_resolves_recursive_detection_ce_runtime_cfg_from_latest_objective() -> None:
    cfg = _resolve_recursive_detection_ce_cfg(
        SimpleNamespace(
            objective=SimpleNamespace(
                id="recursive_detection_ce",
                variant="random_permutation_et_rmp_ce",
                trie_support_weight=2.0,
                trie_balance_weight=1.0,
            )
        )
    )

    assert cfg is not None
    assert cfg.enabled is True
    assert cfg.trie_support_weight == pytest.approx(2.0)
    assert cfg.trie_balance_weight == pytest.approx(1.0)


def test_sft_resolves_prefix_rollin_runtime_cfg_from_objectized_target() -> None:
    cfg = _resolve_recursive_detection_ce_cfg(
        SimpleNamespace(
            objective=SimpleNamespace(
                id="recursive_detection_ce",
                variant="prefix_rollin_et_rmp_ce",
                trie_support_weight=None,
                trie_balance_weight=None,
                target=SimpleNamespace(
                    support_weight=1.0,
                    balance_weight=2.0,
                ),
            )
        )
    )

    assert cfg is not None
    assert cfg.enabled is True
    assert cfg.variant == "prefix_rollin_et_rmp_ce"
    assert cfg.trie_support_weight == pytest.approx(1.0)
    assert cfg.trie_balance_weight == pytest.approx(2.0)


def test_latest_detection_mode_accepts_prefix_rollin_variant() -> None:
    assert (
        _latest_detection_mode(
            SimpleNamespace(
                objective=SimpleNamespace(variant="prefix_rollin_et_rmp_ce")
            )
        )
        == "prefix_rollin_et_rmp_ce"
    )


def test_sft_rejects_unsupported_recursive_detection_ce_variant() -> None:
    with pytest.raises(ValueError, match="random_permutation_et_rmp_ce"):
        _resolve_recursive_detection_ce_cfg(
            SimpleNamespace(
                objective=SimpleNamespace(
                    id="recursive_detection_ce",
                    variant="trie_disabled_full_suffix_ce",
                    trie_support_weight=0.0,
                    trie_balance_weight=0.0,
                )
            )
        )


def test_sft_live_bootstrap_passes_recursive_ce_cfg_to_composition() -> None:
    tree = ast.parse(SFT_PATH.read_text(encoding="utf-8"))
    compose_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "compose_trainer_class"
    ]

    assert compose_calls
    assert any(
        kw.arg == "recursive_detection_ce_cfg"
        for call in compose_calls
        for kw in call.keywords
    )


def test_sft_live_bootstrap_attaches_recursive_ce_cfg_to_trainer() -> None:
    tree = ast.parse(SFT_PATH.read_text(encoding="utf-8"))
    setattr_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "setattr"
    ]

    assert any(
        len(call.args) >= 2
        and isinstance(call.args[1], ast.Constant)
        and call.args[1].value == "recursive_detection_ce_cfg"
        for call in setattr_calls
    )


def test_sft_fails_fast_if_coord_offset_hooks_are_missing_after_peft_wrap() -> None:
    tree = ast.parse(SFT_PATH.read_text(encoding="utf-8"))
    raise_messages: list[str] = []
    for ast_node in ast.walk(tree):
        if not isinstance(ast_node, ast.Raise):
            continue
        exc = ast_node.exc
        if not isinstance(exc, ast.Call):
            continue
        if not isinstance(exc.func, ast.Name) or exc.func.id != "RuntimeError":
            continue
        if not exc.args:
            continue
        message = exc.args[0]
        if isinstance(message, ast.Constant) and isinstance(message.value, str):
            raise_messages.append(message.value)

    assert any(
        "coord_offset_adapter not found after prepare_model" in message
        for message in raise_messages
    )


def test_latest_detection_runtime_constructs_dataset_and_sft_delegates() -> None:
    runtime_tree = ast.parse(RUNTIME_PATH.read_text(encoding="utf-8"))
    from_jsonl_calls = [
        node
        for node in ast.walk(runtime_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "from_jsonl"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "DetectionTrainingDataset"
    ]

    assert from_jsonl_calls
    assert any(
        kw.arg == "swift_template"
        for call in from_jsonl_calls
        for kw in call.keywords
    )

    sft_tree = ast.parse(SFT_PATH.read_text(encoding="utf-8"))
    build_dataset_calls = [
        node
        for node in ast.walk(sft_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "build_latest_detection_dataset"
    ]

    assert build_dataset_calls


def test_latest_detection_runtime_shim_preserves_trainable_token_rows() -> None:
    cfg = _prod_latest_detection_config()
    custom_config = _latest_detection_runtime_custom_shim(cfg)

    assert custom_config.trainable_token_rows is cfg.token_rows
    assert custom_config.trainable_token_rows.enabled is True
    assert "coord_geometry" in custom_config.trainable_token_rows.groups
    assert getattr(custom_config.coord_offset, "enabled", None) is False


def test_sft_does_not_apply_recursive_sidecar_guard_to_other_latest_objectives() -> None:
    cfg = _prod_latest_detection_config()
    cfg = replace(
        cfg,
        objective=replace(
            cfg.objective,
            id="sft",
            variant="sorted_sft",
            trie_support_weight=0.0,
            trie_balance_weight=0.0,
            state_weighting="none",
            normalization="token_mean",
        ),
        training={**dict(cfg.training), "packing": True},
    )

    _assert_latest_detection_runtime_supported(
        cfg,
        encoded_sample_cache_cfg=SimpleNamespace(enabled=True),
    )


def test_sft_rejects_latest_recursive_detection_packing_preflight_config() -> None:
    config_path = (
        REPO_ROOT
        / "configs/stage1/recursive_detection_ce_latest/negative/compact_full_static_packing_should_fail.yaml"
    )

    with pytest.raises(ValueError, match=r"recursive_detection_ce.*static packing"):
        ConfigLoader.load_materialized_training_config(str(config_path))


@pytest.mark.parametrize(
    ("training_packing", "packing_update", "match"),
    [
        (True, {}, "training\\.packing=false"),
        (False, {"static_packing": True}, "packing\\.static_packing=false"),
        (False, {"padding_free_packed": True}, "packing\\.padding_free_packed=false"),
    ],
)
def test_sft_rejects_latest_recursive_detection_packing_surfaces(
    training_packing: bool,
    packing_update: dict[str, bool],
    match: str,
) -> None:
    cfg = _prod_latest_detection_config()
    cfg = replace(
        cfg,
        training={**dict(cfg.training), "packing": training_packing},
        packing=replace(cfg.packing, **packing_update),
    )

    with pytest.raises(ValueError, match=match):
        _assert_latest_detection_runtime_supported(
            cfg,
            encoded_sample_cache_cfg=SimpleNamespace(enabled=False),
        )


def test_sft_rejects_latest_recursive_detection_encoded_sample_cache() -> None:
    cfg = _prod_latest_detection_config()

    with pytest.raises(ValueError, match="training\\.encoded_sample_cache"):
        _assert_latest_detection_runtime_supported(
            cfg,
            encoded_sample_cache_cfg=SimpleNamespace(enabled=True),
        )


def test_sft_runtime_preflight_rejects_latest_recursive_detection_eval_packing() -> None:
    cfg = _prod_latest_detection_config()
    cfg = replace(cfg, training={**dict(cfg.training), "eval_packing": True})

    with pytest.raises(ValueError, match="training\\.eval_packing=false"):
        _assert_latest_detection_runtime_supported(
            cfg,
            encoded_sample_cache_cfg=SimpleNamespace(enabled=False),
        )


def test_sft_runtime_preflight_rejects_latest_recursive_detection_use_logits_to_keep() -> None:
    cfg = _prod_latest_detection_config()
    cfg = replace(cfg, training={**dict(cfg.training), "use_logits_to_keep": True})

    with pytest.raises(ValueError, match="use_logits_to_keep=false"):
        _assert_latest_detection_runtime_supported(
            cfg,
            encoded_sample_cache_cfg=SimpleNamespace(enabled=False),
        )


def test_sft_runtime_preflight_rejects_latest_recursive_detection_loss_scale() -> None:
    cfg = _prod_latest_detection_config()
    cfg = replace(cfg, training={**dict(cfg.training), "loss_scale": "default"})

    with pytest.raises(ValueError, match="training\\.loss_scale"):
        _assert_latest_detection_runtime_supported(
            cfg,
            encoded_sample_cache_cfg=SimpleNamespace(enabled=False),
        )


def test_sft_runtime_preflight_rejects_left_padding_for_all_recursive_sidecars() -> None:
    cfg = _prod_latest_detection_config()

    with pytest.raises(ValueError, match="tokenizer\\.padding_side='right'"):
        _assert_latest_detection_runtime_supported(
            cfg,
            encoded_sample_cache_cfg=SimpleNamespace(enabled=False),
            tokenizer=SimpleNamespace(padding_side="left"),
        )


def test_recursive_detection_sidecars_survive_collation_but_not_model_forward() -> None:
    target_sidecar = {"token_targets": (), "loss_atoms": ()}
    sample = {
        "input_ids": [1, 2, 3],
        "attention_mask": [1, 1, 1],
        "labels": [-100, 2, 3],
        "recursive_detection_targets": target_sidecar,
        "detection_metadata": {"template_id": "compact_full"},
        "assistant_payload": {"objects": []},
        "sample_id": 42,
        "dataset": "latest_detection_train",
        "base_idx": 0,
        "messages": [{"role": "assistant", "content": [{"type": "text", "text": ""}]}],
        "metadata": {"image_id": "image-1"},
    }

    def _base_collator(batch):
        assert batch == [sample]
        return {
            "input_ids": [sample["input_ids"]],
            "attention_mask": [sample["attention_mask"]],
            "labels": [sample["labels"]],
        }

    collator = build_batch_extras_collator(
        SimpleNamespace(data_collator=_base_collator)
    )
    batch = collator([sample])

    assert batch["recursive_detection_targets"] == (target_sidecar,)

    for key in REGISTERED_DETECTION_SIDECAR_KEYS:
        if key == "recursive_detection_targets":
            continue
        batch[key] = sample[key]

    for key in DETECTION_DROPPED_BEFORE_MODEL_KEYS:
        batch[key] = sample[key]

    model_inputs = strip_non_model_detection_sidecars(batch)

    assert model_inputs is batch
    for key in REGISTERED_DETECTION_SIDECAR_KEYS:
        assert key not in model_inputs
    for key in DETECTION_DROPPED_BEFORE_MODEL_KEYS:
        assert key not in model_inputs


def test_recursive_detection_model_boundary_preserves_attention_kwargs_but_drops_trainer_loss_func() -> None:
    batch = {
        "input_ids": [[1, 2, 3]],
        "attention_mask": [[1, 1, 1]],
        "labels": [[-100, 2, 3]],
        "recursive_detection_targets": {"token_targets": (), "loss_atoms": ()},
        "compute_loss_func": object(),
        "cu_seq_lens_q": object(),
        "cu_seq_lens_k": object(),
        "max_length_q": 3,
        "max_length_k": 3,
    }

    model_inputs = strip_non_model_detection_sidecars(batch)

    assert "recursive_detection_targets" not in model_inputs
    assert "compute_loss_func" not in model_inputs
    assert "cu_seq_lens_q" in model_inputs
    assert "cu_seq_lens_k" in model_inputs
    assert "max_length_q" in model_inputs
    assert "max_length_k" in model_inputs


def test_recursive_detection_sidecar_stripping_rejects_unknown_extras() -> None:
    batch = {
        "input_ids": [[1, 2, 3]],
        "attention_mask": [[1, 1, 1]],
        "labels": [[-100, 2, 3]],
        "unexpected_sidecar": object(),
    }

    with pytest.raises(ValueError, match="Unregistered detection batch extras"):
        strip_non_model_detection_sidecars(batch)
