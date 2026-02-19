from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.sft import _resolve_dataset_seed


def test_resolve_dataset_seed_prefers_training_config_seed() -> None:
    training_config = SimpleNamespace(training={"seed": 123})
    train_args = SimpleNamespace(seed=999)

    assert _resolve_dataset_seed(training_config=training_config, train_args=train_args) == 123


def test_resolve_dataset_seed_falls_back_to_train_args_seed() -> None:
    training_config = SimpleNamespace(training={})
    train_args = SimpleNamespace(seed=777)

    assert _resolve_dataset_seed(training_config=training_config, train_args=train_args) == 777


def test_resolve_dataset_seed_rejects_non_int_like_seed() -> None:
    training_config = SimpleNamespace(training={"seed": "not_an_int"})
    train_args = SimpleNamespace(seed=1)

    with pytest.raises(TypeError, match=r"training\.seed must be an int"):
        _resolve_dataset_seed(training_config=training_config, train_args=train_args)
