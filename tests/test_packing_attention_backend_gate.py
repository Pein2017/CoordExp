from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.sft import _validate_attention_backend_for_packing


def test_validate_attention_backend_for_packing_rejects_non_flash() -> None:
    training_config = SimpleNamespace(training={"packing": True}, model={"attn_impl": "sdpa"})

    with pytest.raises(ValueError, match=r"training\.packing=true requires model\.attn_impl"):
        _validate_attention_backend_for_packing(training_config=training_config)


def test_validate_attention_backend_for_packing_allows_flash_attention_2() -> None:
    training_config = SimpleNamespace(
        training={"packing": True},
        model={"attn_impl": "flash_attention_2"},
    )

    _validate_attention_backend_for_packing(training_config=training_config)


def test_validate_attention_backend_for_packing_requires_attn_impl_key() -> None:
    training_config = SimpleNamespace(training={"packing": True}, model={})

    with pytest.raises(ValueError):
        _validate_attention_backend_for_packing(training_config=training_config)
