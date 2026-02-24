from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as F

from src.coord_tokens.soft_ce_w1 import coord_soft_ce_w1
from src.trainers.losses.coord_soft_ce_w1 import coord_vocab_gate_loss

from ..contracts import ModuleResult, PipelineModuleSpec, TeacherForcingContext


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def run_coord_reg_module(
    *,
    context: TeacherForcingContext,
    spec: PipelineModuleSpec,
    state: Mapping[str, Any],
) -> ModuleResult:
    cfg = spec.config if isinstance(spec.config, Mapping) else {}

    coord_logits = state.get("coord_logits")
    coord_logits_full = state.get("coord_logits_full")
    target_bins = state.get("coord_target_bins")

    if not isinstance(coord_logits, torch.Tensor) or int(coord_logits.numel()) == 0:
        z = context.logits.new_tensor(0.0)
        metrics = {
            "loss/coord_reg": 0.0,
            "loss/coord_token_ce": 0.0,
            "loss/coord_soft_ce": 0.0,
            "loss/coord_w1": 0.0,
            "loss/coord_el1": 0.0,
            "loss/coord_ehuber": 0.0,
            "loss/coord_entropy": 0.0,
            "loss/coord_gate": 0.0,
            "loss/text_gate": 0.0,
        }
        return ModuleResult(loss=z, metrics=metrics, state={"coord_reg": z})

    if not isinstance(coord_logits_full, torch.Tensor) or not isinstance(target_bins, torch.Tensor):
        raise ValueError("coord_reg module requires bbox_geo state: coord_logits_full + coord_target_bins")

    weights = {
        "coord_ce_weight": max(0.0, _coerce_float(cfg.get("coord_ce_weight", context.extra.get("coord_ce_weight", 0.0)), 0.0)),
        "coord_soft_ce_weight": max(
            0.0,
            _coerce_float(
                cfg.get(
                    "coord_soft_ce_weight",
                    cfg.get(
                        "soft_ce_weight",
                        context.extra.get("coord_soft_ce_weight", 0.0),
                    ),
                ),
                0.0,
            ),
        ),
        "coord_w1_weight": max(
            0.0,
            _coerce_float(
                cfg.get(
                    "coord_w1_weight",
                    cfg.get(
                        "w1_weight",
                        context.extra.get("coord_w1_weight", 0.0),
                    ),
                ),
                0.0,
            ),
        ),
        "coord_el1_weight": max(0.0, _coerce_float(cfg.get("coord_el1_weight", context.extra.get("coord_el1_weight", 0.0)), 0.0)),
        "coord_ehuber_weight": max(0.0, _coerce_float(cfg.get("coord_ehuber_weight", context.extra.get("coord_ehuber_weight", 0.0)), 0.0)),
        "coord_entropy_weight": _coerce_float(cfg.get("coord_entropy_weight", context.extra.get("coord_entropy_weight", 0.0)), 0.0),
        "coord_gate_weight": max(0.0, _coerce_float(cfg.get("coord_gate_weight", context.extra.get("coord_gate_weight", 0.0)), 0.0)),
        "text_gate_weight": max(0.0, _coerce_float(cfg.get("text_gate_weight", context.extra.get("text_gate_weight", 0.0)), 0.0)),
    }

    temperature = max(
        1e-6,
        _coerce_float(
            cfg.get("temperature", float(context.temperature)),
            float(context.temperature),
        ),
    )
    huber_delta = max(
        1e-6,
        _coerce_float(
            cfg.get("coord_huber_delta", context.extra.get("coord_huber_delta", 0.001)),
            0.001,
        ),
    )

    coord_ce = context.logits.new_tensor(0.0)
    coord_soft_ce = context.logits.new_tensor(0.0)
    coord_w1 = context.logits.new_tensor(0.0)
    coord_el1 = context.logits.new_tensor(0.0)
    coord_ehuber = context.logits.new_tensor(0.0)
    coord_entropy = context.logits.new_tensor(0.0)
    coord_gate = context.logits.new_tensor(0.0)
    text_gate = context.logits.new_tensor(0.0)

    if weights["coord_ce_weight"] != 0.0:
        coord_ce = F.cross_entropy(
            coord_logits.float(),
            target_bins.long(),
            reduction="mean",
        ).to(dtype=context.logits.dtype)

    if weights["coord_soft_ce_weight"] != 0.0 or weights["coord_w1_weight"] != 0.0:
        target_sigma = max(
            1e-6,
            _coerce_float(
                cfg.get("target_sigma", context.extra.get("coord_soft_sigma", 2.0)),
                2.0,
            ),
        )
        target_truncate = cfg.get(
            "target_truncate",
            context.extra.get("coord_soft_truncate", None),
        )
        if target_truncate is not None:
            target_truncate = int(target_truncate)
            if target_truncate < 0:
                target_truncate = None
        dist = coord_soft_ce_w1(
            coord_logits,
            target_bins.long(),
            sigma=float(target_sigma),
            truncate=target_truncate,
            temperature=float(temperature),
            soft_ce_weight=1.0,
            w1_weight=1.0,
            normalize_w1=True,
        )
        coord_soft_ce = dist.soft_ce_per_token.mean().to(dtype=context.logits.dtype)
        coord_w1 = dist.w1_per_token.mean().to(dtype=context.logits.dtype)

    if weights["coord_gate_weight"] != 0.0:
        gate_per_token, _mass_mean = coord_vocab_gate_loss(
            logits_full=coord_logits_full,
            logits_coord=coord_logits,
            temperature=float(temperature),
        )
        coord_gate = gate_per_token.mean().to(dtype=context.logits.dtype)

    if (
        weights["coord_el1_weight"] != 0.0
        or weights["coord_ehuber_weight"] != 0.0
        or weights["coord_entropy_weight"] != 0.0
    ):
        probs = torch.softmax(coord_logits.float() / float(temperature), dim=-1)

        if weights["coord_entropy_weight"] != 0.0:
            p = probs.clamp(min=1e-12)
            coord_entropy = (-(p * p.log()).sum(dim=-1)).mean().to(dtype=context.logits.dtype)

        bins_f = torch.arange(0, probs.shape[-1], device=probs.device, dtype=torch.float32) / 999.0
        gt = target_bins.float() / 999.0
        diff = bins_f.unsqueeze(0) - gt.unsqueeze(1)

        if weights["coord_el1_weight"] != 0.0:
            coord_el1 = (probs * diff.abs()).sum(dim=-1).mean().to(dtype=context.logits.dtype)

        if weights["coord_ehuber_weight"] != 0.0:
            absd = diff.abs()
            huber = torch.where(
                absd < float(huber_delta),
                0.5 * (absd**2) / float(huber_delta),
                absd - 0.5 * float(huber_delta),
            )
            coord_ehuber = (probs * huber).sum(dim=-1).mean().to(dtype=context.logits.dtype)

    loss = (
        weights["coord_ce_weight"] * coord_ce
        + weights["coord_soft_ce_weight"] * coord_soft_ce
        + weights["coord_w1_weight"] * coord_w1
        + weights["coord_el1_weight"] * coord_el1
        + weights["coord_ehuber_weight"] * coord_ehuber
        + weights["coord_entropy_weight"] * coord_entropy
        + weights["coord_gate_weight"] * coord_gate
        + weights["text_gate_weight"] * text_gate
    )

    metrics = {
        "loss/coord_reg": float(loss.detach().cpu().item()),
        "loss/coord_token_ce": float(coord_ce.detach().cpu().item()),
        "loss/coord_soft_ce": float(coord_soft_ce.detach().cpu().item()),
        "loss/coord_w1": float(coord_w1.detach().cpu().item()),
        "loss/coord_el1": float(coord_el1.detach().cpu().item()),
        "loss/coord_ehuber": float(coord_ehuber.detach().cpu().item()),
        "loss/coord_entropy": float(coord_entropy.detach().cpu().item()),
        "loss/coord_gate": float(coord_gate.detach().cpu().item()),
        "loss/text_gate": float(text_gate.detach().cpu().item()),
    }

    return ModuleResult(loss=loss, metrics=metrics, state={"coord_reg": loss})
