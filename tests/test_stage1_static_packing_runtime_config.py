from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import src.sft as sft_module
from src.sft import (
    PackingRuntimeConfig,
    _build_static_packing_fingerprint,
    _parse_packing_config,
    _recompute_gas_for_packing,
    _validate_stage1_static_packing_policy,
    _validate_static_packing_accumulation_windows,
)


class _Template:
    def __init__(self, max_length: int):
        self.max_length = int(max_length)


def test_parse_packing_config_defaults_to_static_mode() -> None:
    cfg = _parse_packing_config(
        training_cfg={"packing": True},
        template=_Template(max_length=256),
        train_args=SimpleNamespace(max_model_len=512),
    )

    assert cfg.enabled is True
    assert cfg.mode == "static"
    assert cfg.packing_length == 256
    assert cfg.eval_packing is True
    assert cfg.wait_timeout_s == pytest.approx(7200.0)
    assert cfg.length_cache_persist_every is None
    assert cfg.length_precompute_workers == 8


def test_parse_packing_config_accepts_static_mode() -> None:
    cfg = _parse_packing_config(
        training_cfg={"packing": True, "packing_mode": "static"},
        template=_Template(max_length=128),
        train_args=SimpleNamespace(max_model_len=0),
    )

    assert cfg.mode == "static"
    assert cfg.packing_length == 128


def test_parse_packing_config_allows_disabling_eval_packing() -> None:
    cfg = _parse_packing_config(
        training_cfg={"packing": True, "packing_mode": "static", "eval_packing": False},
        template=_Template(max_length=128),
        train_args=SimpleNamespace(max_model_len=0),
    )

    assert cfg.eval_packing is False


def test_parse_packing_config_accepts_static_cache_runtime_knobs() -> None:
    cfg = _parse_packing_config(
        training_cfg={
            "packing": True,
            "packing_mode": "static",
            "packing_wait_timeout_s": 1234,
            "packing_length_cache_persist_every": 2048,
            "packing_length_precompute_workers": 6,
        },
        template=_Template(max_length=128),
        train_args=SimpleNamespace(max_model_len=0),
    )

    assert cfg.wait_timeout_s == pytest.approx(1234.0)
    assert cfg.length_cache_persist_every == 2048
    assert cfg.length_precompute_workers == 6


def test_parse_packing_config_rejects_negative_wait_timeout() -> None:
    with pytest.raises(ValueError, match="packing_wait_timeout_s"):
        _parse_packing_config(
            training_cfg={
                "packing": True,
                "packing_mode": "static",
                "packing_wait_timeout_s": -1,
            },
            template=_Template(max_length=128),
            train_args=SimpleNamespace(max_model_len=0),
        )


def test_parse_packing_config_rejects_nonpositive_length_cache_persist_every() -> None:
    with pytest.raises(ValueError, match="packing_length_cache_persist_every"):
        _parse_packing_config(
            training_cfg={
                "packing": True,
                "packing_mode": "static",
                "packing_length_cache_persist_every": 0,
            },
            template=_Template(max_length=128),
            train_args=SimpleNamespace(max_model_len=0),
        )


def test_parse_packing_config_rejects_nonpositive_length_precompute_workers() -> None:
    with pytest.raises(ValueError, match="packing_length_precompute_workers"):
        _parse_packing_config(
            training_cfg={
                "packing": True,
                "packing_mode": "static",
                "packing_length_precompute_workers": 0,
            },
            template=_Template(max_length=128),
            train_args=SimpleNamespace(max_model_len=0),
        )


def test_parse_packing_config_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="packing_mode"):
        _parse_packing_config(
            training_cfg={"packing": True, "packing_mode": "foobar"},
            template=_Template(max_length=128),
            train_args=SimpleNamespace(max_model_len=0),
        )


def test_parse_packing_config_warns_on_deprecated_dynamic_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warning_logs: list[str] = []

    def _capture_warning(message: str, *args: object) -> None:
        if args:
            warning_logs.append(message % args)
        else:
            warning_logs.append(message)

    monkeypatch.setattr(sft_module.logger, "warning", _capture_warning)

    cfg = _parse_packing_config(
        training_cfg={"packing": True, "packing_mode": "dynamic"},
        template=_Template(max_length=128),
        train_args=SimpleNamespace(max_model_len=0),
    )

    assert cfg.mode == "dynamic"
    assert any("packing_mode=dynamic is deprecated" in msg for msg in warning_logs)


def test_validate_stage1_static_packing_policy_rejects_dynamic_mode() -> None:
    with pytest.raises(
        ValueError,
        match="deprecated and unsupported for Stage-1",
    ):
        _validate_stage1_static_packing_policy(
            packing_cfg=PackingRuntimeConfig(enabled=True, mode="dynamic"),
            trainer_variant=None,
        )


def test_validate_stage1_static_packing_policy_allows_eval_packing() -> None:
    _validate_stage1_static_packing_policy(
        packing_cfg=PackingRuntimeConfig(
            enabled=True,
            mode="static",
            eval_packing=True,
        ),
        trainer_variant=None,
    )


def test_validate_stage1_static_packing_policy_skips_rollout_matching_variants() -> None:
    _validate_stage1_static_packing_policy(
        packing_cfg=PackingRuntimeConfig(enabled=True, mode="dynamic"),
        trainer_variant="stage2_ab_training",
    )


def test_static_packing_fingerprint_includes_dataset_source_identity(
    tmp_path: Path,
) -> None:
    train_jsonl = tmp_path / "train.jsonl"
    train_jsonl.write_text('{"id": 1}\n', encoding="utf-8")

    packing_cfg = _parse_packing_config(
        training_cfg={"packing": True, "packing_mode": "static"},
        template=_Template(max_length=128),
        train_args=SimpleNamespace(max_model_len=0),
    )

    fingerprint = _build_static_packing_fingerprint(
        training_config=SimpleNamespace(
            global_max_length=1024,
            template={"system": "sys", "truncation_strategy": "raise"},
            training={"train_dataloader_shuffle": True},
        ),
        custom_config=SimpleNamespace(
            user_prompt="prompt",
            emit_norm="none",
            json_format="standard",
            object_ordering="none",
            object_field_order="geometry_first",
            use_summary=False,
            system_prompt_dense=None,
            system_prompt_summary=None,
        ),
        template=_Template(max_length=128),
        train_args=SimpleNamespace(max_model_len=512),
        dataset_seed=7,
        packing_cfg=packing_cfg,
        train_jsonl=str(train_jsonl),
        fusion_config_path=None,
    )

    assert fingerprint["dataset_split"] == "train"
    assert fingerprint["dataset_jsonl"] == str(train_jsonl)
    assert fingerprint["custom_train_jsonl"] == str(train_jsonl)
    source = fingerprint["dataset_source_jsonl"]
    assert isinstance(source, dict)
    assert source["raw_path"] == str(train_jsonl)
    assert source["resolved_path"] == str(train_jsonl.resolve())
    assert source["exists"] is True
    assert source["size_bytes"] == train_jsonl.stat().st_size


def test_static_packing_fingerprint_tracks_eval_split_inputs(
    tmp_path: Path,
) -> None:
    val_jsonl = tmp_path / "val.jsonl"
    val_jsonl.write_text('{"id": 1}\n', encoding="utf-8")

    packing_cfg = _parse_packing_config(
        training_cfg={"packing": True, "packing_mode": "static"},
        template=_Template(max_length=128),
        train_args=SimpleNamespace(max_model_len=0),
    )

    fingerprint = _build_static_packing_fingerprint(
        training_config=SimpleNamespace(
            global_max_length=1024,
            template={"system": "sys", "truncation_strategy": "raise"},
            training={"train_dataloader_shuffle": True},
        ),
        custom_config=SimpleNamespace(
            user_prompt="prompt",
            emit_norm="none",
            json_format="standard",
            object_ordering="none",
            object_field_order="geometry_first",
            use_summary=False,
            system_prompt_dense=None,
            system_prompt_summary=None,
        ),
        template=_Template(max_length=128),
        train_args=SimpleNamespace(max_model_len=512),
        dataset_seed=7,
        packing_cfg=packing_cfg,
        train_jsonl=str(val_jsonl),
        fusion_config_path=None,
        dataset_split="eval",
        eval_sample_limit=99,
        eval_sample_with_replacement=False,
    )

    assert fingerprint["dataset_split"] == "eval"
    assert fingerprint["eval_sample_limit"] == 99
    assert fingerprint["eval_sample_with_replacement"] is False


def test_recompute_gas_for_packing_uses_effective_batch_size() -> None:
    nested = SimpleNamespace(gradient_accumulation_steps=2)
    args = SimpleNamespace(
        gradient_accumulation_steps=2,
        training_args=nested,
    )

    new_gas = _recompute_gas_for_packing(
        train_args=args,
        training_cfg={"effective_batch_size": 32},
        original_per_device_bs=4,
        original_gas=2,
        world_size=2,
    )

    assert new_gas == 16
    assert args.gradient_accumulation_steps == 16
    assert nested.gradient_accumulation_steps == 16


def test_recompute_gas_for_packing_preserves_implied_global_batch() -> None:
    args = SimpleNamespace(
        gradient_accumulation_steps=3,
        training_args=None,
    )

    new_gas = _recompute_gas_for_packing(
        train_args=args,
        training_cfg={},
        original_per_device_bs=4,
        original_gas=3,
        world_size=2,
    )

    assert new_gas == 12
    assert args.gradient_accumulation_steps == 12


def test_recompute_gas_for_packing_logs_requested_vs_realized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = SimpleNamespace(
        gradient_accumulation_steps=1,
        training_args=None,
    )
    info_logs: list[str] = []

    def _capture_info(message: str, *args: object) -> None:
        if args:
            info_logs.append(message % args)
        else:
            info_logs.append(message)

    monkeypatch.setattr(sft_module.logger, "info", _capture_info)

    new_gas = _recompute_gas_for_packing(
        train_args=args,
        training_cfg={"effective_batch_size": 8},
        original_per_device_bs=1,
        original_gas=1,
        world_size=2,
    )

    assert new_gas == 4
    assert any(
        "requested_global_packs_per_step=8" in msg for msg in info_logs
    )
    assert any(
        "realized_global_packs_per_step=8" in msg for msg in info_logs
    )


def test_recompute_gas_for_packing_raises_when_effective_batch_not_divisible_by_world_size() -> None:
    args = SimpleNamespace(
        gradient_accumulation_steps=1,
        training_args=None,
    )

    with pytest.raises(
        ValueError,
        match="must be divisible by world_size",
    ):
        _recompute_gas_for_packing(
            train_args=args,
            training_cfg={"effective_batch_size": 7},
            original_per_device_bs=1,
            original_gas=1,
            world_size=3,
        )


def test_validate_static_packing_accumulation_windows_warns_on_partial_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warning_logs: list[str] = []

    def _capture_warning(message: str, *args: object) -> None:
        if args:
            warning_logs.append(message % args)
        else:
            warning_logs.append(message)

    monkeypatch.setattr(sft_module.logger, "warning", _capture_warning)

    _validate_static_packing_accumulation_windows(
        packing_cfg=PackingRuntimeConfig(enabled=True, mode="static"),
        trainer_variant=None,
        per_rank_batches_est=15,
        gradient_accumulation_steps=32,
        world_size=1,
        dataloader_drop_last=True,
    )

    assert any(
        "per_rank_batches_est=15 is smaller than gradient_accumulation_steps=32"
        in msg
        for msg in warning_logs
    )


def test_validate_static_packing_accumulation_windows_warns_on_remainder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warning_logs: list[str] = []

    def _capture_warning(message: str, *args: object) -> None:
        if args:
            warning_logs.append(message % args)
        else:
            warning_logs.append(message)

    monkeypatch.setattr(sft_module.logger, "warning", _capture_warning)

    _validate_static_packing_accumulation_windows(
        packing_cfg=PackingRuntimeConfig(
            enabled=True,
            mode="static",
        ),
        trainer_variant=None,
        per_rank_batches_est=15,
        gradient_accumulation_steps=8,
        world_size=1,
        dataloader_drop_last=False,
    )

    assert any(
        "is not divisible by gradient_accumulation_steps=8" in msg
        for msg in warning_logs
    )
