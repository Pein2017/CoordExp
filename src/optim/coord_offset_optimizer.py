"""Optimizer variant that adds coord-offset parameter buckets."""

from __future__ import annotations

from typing import List, Tuple

import torch
from transformers import Trainer

try:
    from swift.plugin.optimizer import (
        create_multimodal_optimizer,
        get_param_startswith,
    )
except ImportError as exc:  # pragma: no cover - defensive for environments without swift
    raise ImportError(
        "swift.plugin.optimizer is required for coord_offset optimizer."
    ) from exc


def _split_decay(
    parameters: List[Tuple[str, torch.nn.Parameter]],
    decay_parameters: set[str],
    lr: float,
    weight_decay: float,
) -> list[dict]:
    groups: list[dict] = []
    no_decay = [p for n, p in parameters if n not in decay_parameters]
    with_decay = [p for n, p in parameters if n in decay_parameters]
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0, "lr": lr})
    if with_decay:
        groups.append({"params": with_decay, "weight_decay": weight_decay, "lr": lr})
    return groups


def create_multimodal_coord_offset_optimizer(args, model, dataset):
    coord_cfg = getattr(args, "coord_offset_config", None)
    if not coord_cfg or not getattr(coord_cfg, "enabled", False):
        return create_multimodal_optimizer(args, model, dataset)

    decay_parameters = set(Trainer.get_decay_parameter_names(None, model))
    rejected_prefix = ["coord_offset_adapter"]

    seen_params: set[int] = set()

    def _dedup(params):
        uniq = []
        for p in params:
            if id(p) in seen_params:
                continue
            uniq.append(p)
            seen_params.add(id(p))
        return uniq

    coord_params = [
        (n, p)
        for n, p in model.named_parameters()
        if p.requires_grad and "coord_offset_adapter" in n
    ]
    embed_params = [p for n, p in coord_params if "embed_offset" in n]
    head_params = [p for n, p in coord_params if "head_offset" in n]

    embed_lr = coord_cfg.embed_lr if coord_cfg.embed_lr is not None else args.learning_rate
    head_lr = coord_cfg.head_lr if coord_cfg.head_lr is not None else args.learning_rate
    offset_wd = coord_cfg.weight_decay if coord_cfg.weight_decay is not None else 0.0

    optimizer_grouped_parameters: list[dict] = []
    if embed_params:
        optimizer_grouped_parameters.append(
            {"params": _dedup(embed_params), "lr": embed_lr, "weight_decay": offset_wd}
        )
    if head_params:
        optimizer_grouped_parameters.append(
            {"params": _dedup(head_params), "lr": head_lr, "weight_decay": offset_wd}
        )

    model_arch = getattr(getattr(model, "model_meta", None), "model_arch", None)
    def _strip_coord(params: List[Tuple[str, torch.nn.Parameter]]) -> List[Tuple[str, torch.nn.Parameter]]:
        return [(n, p) for n, p in params if "coord_offset_adapter" not in n]

    if model_arch is not None:
        vit_parameters = _strip_coord(
            get_param_startswith(model, model_arch.vision_tower, rejected_prefix)
        )
        aligner_parameters = _strip_coord(
            get_param_startswith(model, model_arch.aligner, rejected_prefix)
        )
        llm_parameters = _strip_coord(
            get_param_startswith(model, model_arch.language_model, rejected_prefix)
        )
        for lr, parameters in zip(
            [args.vit_lr, args.aligner_lr, args.learning_rate],
            [vit_parameters, aligner_parameters, llm_parameters],
        ):
            lr = lr if lr is not None else args.learning_rate
            groups = _split_decay(parameters, decay_parameters, lr, args.weight_decay)
            for g in groups:
                g["params"] = _dedup(g["params"])
            optimizer_grouped_parameters.extend(groups)
    else:
        # Fallback: treat all remaining trainable params as a single group
        remaining = [
            (n, p)
            for n, p in model.named_parameters()
            if p.requires_grad and "coord_offset_adapter" not in n
        ]
        optimizer_grouped_parameters.extend(
            _split_decay(_dedup([p for _, p in remaining]), decay_parameters, args.learning_rate, args.weight_decay)
        )

    optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(args, model)
    return optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs), None


def register_coord_offset_optimizer() -> None:
    from swift.plugin import optimizers_map

    if "multimodal_coord_offset" not in optimizers_map:
        optimizers_map["multimodal_coord_offset"] = (
            create_multimodal_coord_offset_optimizer
        )
