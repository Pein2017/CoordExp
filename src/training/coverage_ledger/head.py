"""Trainable projection head for coverage-ledger auxiliary supervision."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class CoverageLedgerHeadSpec:
    hidden_size: int
    visual_dim: int
    ledger_projection_dim: int
    normalize_eps: float


class CoverageLedgerHead(nn.Module):
    """Owns the trainable projection weights for coverage-ledger matching."""

    state_projection: nn.Linear
    region_anchor_state_projection: nn.Linear
    object_projection: nn.Linear

    def __init__(
        self,
        *,
        hidden_size: int,
        visual_dim: int,
        ledger_projection_dim: int,
        normalize_eps: float,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("coverage_ledger_head hidden_size must be positive")
        if visual_dim <= 0:
            raise ValueError("coverage_ledger_head visual_dim must be positive")
        if ledger_projection_dim <= 0:
            raise ValueError(
                "coverage_ledger_head ledger_projection_dim must be positive"
            )
        if normalize_eps <= 0:
            raise ValueError("coverage_ledger_head normalize_eps must be positive")

        self.hidden_size = int(hidden_size)
        self.visual_dim = int(visual_dim)
        self.ledger_projection_dim = int(ledger_projection_dim)
        self.normalize_eps = float(normalize_eps)
        self.state_projection = nn.Linear(
            self.hidden_size,
            self.ledger_projection_dim,
            bias=False,
        )
        self.region_anchor_state_projection = nn.Linear(
            self.visual_dim,
            self.ledger_projection_dim,
            bias=False,
        )
        self.object_projection = nn.Linear(
            self.hidden_size,
            self.ledger_projection_dim,
            bias=False,
        )


def install_coverage_ledger_head(
    model: nn.Module,
    coverage_ledger_cfg: Any,
    *,
    visual_dim: int | None = None,
    sample_image_embeddings: torch.Tensor | None = None,
) -> CoverageLedgerHead | None:
    """Install ``coverage_ledger_head`` on the trainable model when enabled."""

    if not _coverage_ledger_enabled(coverage_ledger_cfg):
        return None

    spec = CoverageLedgerHeadSpec(
        hidden_size=_infer_hidden_size(model),
        visual_dim=_infer_visual_dim(
            model,
            explicit_visual_dim=visual_dim,
            sample_image_embeddings=sample_image_embeddings,
        ),
        ledger_projection_dim=_positive_int_cfg(
            coverage_ledger_cfg,
            "ledger_projection_dim",
        ),
        normalize_eps=_positive_float_cfg(coverage_ledger_cfg, "normalize_eps"),
    )

    existing = getattr(model, "coverage_ledger_head", None)
    if existing is not None:
        if not isinstance(existing, CoverageLedgerHead):
            raise ValueError(
                "coverage_ledger_head already exists with incompatible type "
                f"{type(existing).__name__}"
            )
        _require_matching_spec(existing, spec)
        return existing

    device, dtype = _resolve_trainable_parameter_device_dtype(model)
    head = CoverageLedgerHead(
        hidden_size=spec.hidden_size,
        visual_dim=spec.visual_dim,
        ledger_projection_dim=spec.ledger_projection_dim,
        normalize_eps=spec.normalize_eps,
    )
    head.to(device=device, dtype=dtype)
    model.add_module("coverage_ledger_head", head)
    return head


def _coverage_ledger_enabled(cfg: Any) -> bool:
    if cfg is None:
        return False
    if isinstance(cfg, Mapping):
        return bool(cfg.get("enabled", False))
    return bool(getattr(cfg, "enabled", False))


def _positive_int_cfg(cfg: Any, field_name: str) -> int:
    value = _cfg_value(cfg, field_name)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"coverage_ledger.{field_name} must be a positive integer")
    return int(value)


def _positive_float_cfg(cfg: Any, field_name: str) -> float:
    value = _cfg_value(cfg, field_name)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"coverage_ledger.{field_name} must be a positive float")
    parsed = float(value)
    if not torch.isfinite(torch.tensor(parsed)) or parsed <= 0:
        raise ValueError(
            f"coverage_ledger.{field_name} must be a positive finite float"
        )
    return parsed


def _cfg_value(cfg: Any, field_name: str) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(field_name)
    return getattr(cfg, field_name, None)


def _require_matching_spec(
    existing: CoverageLedgerHead,
    expected: CoverageLedgerHeadSpec,
) -> None:
    actual = CoverageLedgerHeadSpec(
        hidden_size=int(existing.state_projection.in_features),
        visual_dim=int(existing.region_anchor_state_projection.in_features),
        ledger_projection_dim=int(existing.state_projection.out_features),
        normalize_eps=float(existing.normalize_eps),
    )
    if actual != expected:
        raise ValueError(
            "coverage_ledger_head already exists with incompatible dimensions: "
            f"actual={actual}, expected={expected}"
        )


def _infer_hidden_size(model: nn.Module) -> int:
    for config in _iter_model_configs(model):
        text_config = getattr(config, "text_config", None)
        hidden = _positive_int_attr(text_config, "hidden_size")
        if hidden is not None:
            return hidden
    for config in _iter_model_configs(model):
        for attr_name in ("hidden_size", "n_embd", "d_model"):
            hidden = _positive_int_attr(config, attr_name)
            if hidden is not None:
                return hidden
    raise ValueError(
        "coverage_ledger_head could not infer language hidden_size from model config"
    )


def _infer_visual_dim(
    model: nn.Module,
    *,
    explicit_visual_dim: int | None,
    sample_image_embeddings: torch.Tensor | None,
) -> int:
    if explicit_visual_dim is not None:
        return _require_positive_int(explicit_visual_dim, "visual_dim")
    if sample_image_embeddings is not None:
        if sample_image_embeddings.ndim == 0:
            raise ValueError(
                "coverage_ledger_head sample_image_embeddings must have a final dimension"
            )
        return _require_positive_int(
            int(sample_image_embeddings.shape[-1]),
            "sample_image_embeddings.shape[-1]",
        )
    for config in _iter_model_configs(model):
        for container_attr in ("vision_config", "visual_config", "vision_model_config"):
            container = getattr(config, container_attr, None)
            for attr_name in (
                "out_hidden_size",
                "projection_dim",
                "hidden_size",
                "embed_dim",
                "width",
            ):
                visual = _positive_int_attr(container, attr_name)
                if visual is not None:
                    return visual
        for attr_name in (
            "visual_dim",
            "vision_hidden_size",
            "image_hidden_size",
            "image_embed_dim",
        ):
            visual = _positive_int_attr(config, attr_name)
            if visual is not None:
                return visual
    raise ValueError(
        "coverage_ledger_head could not infer visual_dim from model config; "
        "pass visual_dim or sample_image_embeddings from captured post-merger "
        "image embeddings"
    )


def _iter_model_configs(model: nn.Module) -> Iterator[Any]:
    seen: set[int] = set()
    stack: list[Any] = [model]
    while stack:
        current = stack.pop(0)
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        config = getattr(current, "config", None)
        if config is not None and id(config) not in seen:
            seen.add(id(config))
            yield config
        for attr_name in ("module", "base_model", "model", "language_model"):
            child = getattr(current, attr_name, None)
            if child is not None and id(child) not in seen:
                stack.append(child)


def _positive_int_attr(container: Any, attr_name: str) -> int | None:
    if container is None:
        return None
    value = getattr(container, attr_name, None)
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return int(value)
    return None


def _require_positive_int(value: int, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"coverage_ledger_head {label} must be a positive integer")
    return int(value)


def _resolve_trainable_parameter_device_dtype(
    model: nn.Module,
) -> tuple[torch.device, torch.dtype]:
    fallback: tuple[torch.device, torch.dtype] | None = None
    for param in model.parameters():
        if param.is_floating_point():
            candidate = (param.device, param.dtype)
            if fallback is None:
                fallback = candidate
            if param.requires_grad:
                return candidate
    if fallback is not None:
        return fallback
    return torch.device("cpu"), torch.float32
