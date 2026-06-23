"""Optimizer variant that adds token-embeddings adapter parameter buckets."""

from __future__ import annotations

from typing import List, Tuple
from types import SimpleNamespace

import torch
from transformers import Trainer

try:
    from swift.plugin.optimizer import (
        create_multimodal_optimizer,
        get_param_startswith,
    )
    _OPTIMIZER_CALLBACK_BASE = None
except ImportError:
    try:
        from swift.optimizers import OptimizerCallback, optimizers_map
        from swift.optimizers.multimodal import (
            MultimodalOptimizerCallback,
            get_param_startswith,
        )
    except ImportError as exc:  # pragma: no cover - defensive for environments without swift
        raise ImportError(
            "swift optimizer plugin APIs are required for token_embeddings_adapter optimizer."
        ) from exc

    _OPTIMIZER_CALLBACK_BASE = OptimizerCallback

    def create_multimodal_optimizer(args, model, dataset):
        _ = dataset
        trainer = SimpleNamespace(model=model)
        optimizer = MultimodalOptimizerCallback(args, trainer).create_optimizer(model)
        return optimizer, None


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


def create_multimodal_token_embeddings_adapter_optimizer(args, model, dataset):
    adapter_cfg = getattr(args, "token_embeddings_adapter_config", None)
    if not adapter_cfg or not getattr(adapter_cfg, "enabled", False):
        return create_multimodal_optimizer(args, model, dataset)

    decay_parameters = set(Trainer.get_decay_parameter_names(None, model))
    rejected_prefix = ["token_embeddings_adapter", "coverage_ledger_head"]

    seen_params: set[int] = set()

    def _dedup(params):
        uniq = []
        for p in params:
            if id(p) in seen_params:
                continue
            uniq.append(p)
            seen_params.add(id(p))
        return uniq

    def _is_token_embeddings_adapter_param(name: str) -> bool:
        return "token_embeddings_adapter" in name

    def _is_coverage_ledger_head_param(name: str) -> bool:
        return (
            name.startswith("coverage_ledger_head.")
            or ".coverage_ledger_head." in name
        )

    adapter_params = [
        (n, p)
        for n, p in model.named_parameters()
        if p.requires_grad and _is_token_embeddings_adapter_param(n)
    ]
    embed_params = [p for n, p in adapter_params if "embed_offset" in n]
    head_params = [p for n, p in adapter_params if "head_offset" in n]
    coverage_ledger_head_params = [
        (n, p)
        for n, p in model.named_parameters()
        if p.requires_grad and _is_coverage_ledger_head_param(n)
    ]

    embed_lr = adapter_cfg.embed_lr if adapter_cfg.embed_lr is not None else args.learning_rate
    head_lr = adapter_cfg.head_lr if adapter_cfg.head_lr is not None else args.learning_rate
    offset_wd = adapter_cfg.weight_decay if adapter_cfg.weight_decay is not None else 0.0

    optimizer_grouped_parameters: list[dict] = []
    if embed_params:
        optimizer_grouped_parameters.append(
            {"params": _dedup(embed_params), "lr": embed_lr, "weight_decay": offset_wd}
        )
    if head_params:
        optimizer_grouped_parameters.append(
            {"params": _dedup(head_params), "lr": head_lr, "weight_decay": offset_wd}
        )
    ledger_head_groups = _split_decay(
        coverage_ledger_head_params,
        decay_parameters,
        args.learning_rate,
        args.weight_decay,
    )
    for group in ledger_head_groups:
        group["params"] = _dedup(group["params"])
        if group["params"]:
            optimizer_grouped_parameters.append(group)

    model_arch = getattr(getattr(model, "model_meta", None), "model_arch", None)
    def _strip_special_params(
        params: List[Tuple[str, torch.nn.Parameter]]
    ) -> List[Tuple[str, torch.nn.Parameter]]:
        return [
            (n, p)
            for n, p in params
            if not _is_token_embeddings_adapter_param(n)
            and not _is_coverage_ledger_head_param(n)
        ]

    if model_arch is not None:
        vit_parameters = _strip_special_params(
            get_param_startswith(model, model_arch.vision_tower, rejected_prefix)
        )
        aligner_parameters = _strip_special_params(
            get_param_startswith(model, model_arch.aligner, rejected_prefix)
        )
        llm_parameters = _strip_special_params(
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
            if p.requires_grad
            and not _is_token_embeddings_adapter_param(n)
            and not _is_coverage_ledger_head_param(n)
        ]
        dedup_remaining = []
        for name, param in remaining:
            if id(param) in seen_params:
                continue
            dedup_remaining.append((name, param))
            seen_params.add(id(param))
        optimizer_grouped_parameters.extend(
            _split_decay(
                dedup_remaining,
                decay_parameters,
                args.learning_rate,
                args.weight_decay,
            )
        )

    optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(args, model)
    return optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs), None


def register_token_embeddings_adapter_optimizer() -> None:
    try:
        from swift.plugin import optimizers_map as plugin_optimizers_map
    except ImportError:
        plugin_optimizers_map = None

    if plugin_optimizers_map is not None:
        if "multimodal_token_embeddings_adapter" not in plugin_optimizers_map:
            plugin_optimizers_map["multimodal_token_embeddings_adapter"] = (
                create_multimodal_token_embeddings_adapter_optimizer
            )
        return

    class TokenEmbeddingsAdapterOptimizerCallback(_OPTIMIZER_CALLBACK_BASE):
        def create_optimizer(self, model=None):
            model = model if model is not None else self.trainer.model
            optimizer, _scheduler = create_multimodal_token_embeddings_adapter_optimizer(
                self.args,
                model,
                None,
            )
            return optimizer

    if "multimodal_token_embeddings_adapter" not in optimizers_map:
        optimizers_map["multimodal_token_embeddings_adapter"] = (
            TokenEmbeddingsAdapterOptimizerCallback
        )
