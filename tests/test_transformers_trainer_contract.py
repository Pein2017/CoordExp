from __future__ import annotations

import torch.nn as nn

from transformers import Trainer, TrainingArguments


def test_trainer_get_decay_parameter_names_callable_with_none_self() -> None:
    model = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))

    names = Trainer.get_decay_parameter_names(None, model)

    assert isinstance(names, list)
    assert all(isinstance(name, str) for name in names)

    param_names = {name for name, _ in model.named_parameters()}
    assert set(names).issubset(param_names)

    # Contract: biases are excluded from weight decay.
    assert "0.bias" in param_names
    assert "0.bias" not in names


def test_trainer_get_optimizer_cls_and_kwargs_contract() -> None:
    model = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
    args = TrainingArguments(output_dir="/tmp/coordexp_tmp", report_to=[])

    optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(args, model)

    assert callable(optimizer_cls)
    assert isinstance(optimizer_kwargs, dict)
