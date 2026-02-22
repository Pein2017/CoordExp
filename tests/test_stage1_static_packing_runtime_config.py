from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.sft import (
    _parse_packing_config,
    _recompute_gas_for_packing,
)


class _Template:
    def __init__(self, max_length: int):
        self.max_length = int(max_length)


def test_parse_packing_config_defaults_to_dynamic_mode() -> None:
    cfg = _parse_packing_config(
        training_cfg={"packing": True},
        template=_Template(max_length=256),
        train_args=SimpleNamespace(max_model_len=512),
    )

    assert cfg.enabled is True
    assert cfg.mode == "dynamic"
    assert cfg.packing_length == 256


def test_parse_packing_config_accepts_static_mode() -> None:
    cfg = _parse_packing_config(
        training_cfg={"packing": True, "packing_mode": "static"},
        template=_Template(max_length=128),
        train_args=SimpleNamespace(max_model_len=0),
    )

    assert cfg.mode == "static"
    assert cfg.packing_length == 128


def test_parse_packing_config_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="packing_mode"):
        _parse_packing_config(
            training_cfg={"packing": True, "packing_mode": "foobar"},
            template=_Template(max_length=128),
            train_args=SimpleNamespace(max_model_len=0),
        )


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
