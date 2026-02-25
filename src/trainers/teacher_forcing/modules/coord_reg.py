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


def _stable_coord_mass_from_logits(
    *,
    logits_full: torch.Tensor,
    logits_coord: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Compute p_coord(t) = sum_{i in coord_vocab} softmax(logits_full / T)[i] for each row.

    This is done via logsumexp differences for numerical stability:
      log p_coord = logsumexp(coord/T) - logsumexp(all/T)
    """

    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    full = torch.nan_to_num(
        logits_full.float(), nan=0.0, posinf=1e4, neginf=-1e4
    ).clamp(min=-1e4, max=1e4) / float(temperature)
    coord = torch.nan_to_num(
        logits_coord.float(), nan=0.0, posinf=1e4, neginf=-1e4
    ).clamp(min=-1e4, max=1e4) / float(temperature)

    lse_all = torch.logsumexp(full, dim=-1)
    lse_coord = torch.logsumexp(coord, dim=-1)
    log_p = (lse_coord - lse_all).clamp(min=-50.0, max=0.0)
    p = torch.exp(log_p).clamp(min=0.0, max=1.0)
    return torch.nan_to_num(p, nan=0.0, posinf=1.0, neginf=0.0)


def run_coord_reg_module(
    *,
    context: TeacherForcingContext,
    spec: PipelineModuleSpec,
    state: Mapping[str, Any],
) -> ModuleResult:
    if str(context.registry_context) == "gt":
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

    registry_context = str(context.registry_context or "").strip().lower()

    soft_ce_weight = _coerce_float(
        cfg.get(
            "soft_ce_weight",
            context.extra.get("coord_soft_ce_weight", 0.0),
        ),
        0.0,
    )
    if registry_context == "self_context":
        soft_ce_weight = _coerce_float(
            cfg.get("self_context_soft_ce_weight", soft_ce_weight),
            soft_ce_weight,
        )

    weights = {
        "coord_ce_weight": max(
            0.0,
            _coerce_float(
                cfg.get("coord_ce_weight", context.extra.get("coord_ce_weight", 0.0)),
                0.0,
            ),
        ),
        "coord_soft_ce_weight": max(0.0, float(soft_ce_weight)),
        "coord_w1_weight": max(
            0.0,
            _coerce_float(
                cfg.get("w1_weight", context.extra.get("coord_w1_weight", 0.0)),
                0.0,
            ),
        ),
        "coord_el1_weight": max(
            0.0,
            _coerce_float(
                cfg.get("coord_el1_weight", context.extra.get("coord_el1_weight", 0.0)),
                0.0,
            ),
        ),
        "coord_ehuber_weight": max(
            0.0,
            _coerce_float(
                cfg.get(
                    "coord_ehuber_weight",
                    context.extra.get("coord_ehuber_weight", 0.0),
                ),
                0.0,
            ),
        ),
        "coord_entropy_weight": _coerce_float(
            cfg.get("coord_entropy_weight", context.extra.get("coord_entropy_weight", 0.0)),
            0.0,
        ),
        "coord_gate_weight": max(
            0.0,
            _coerce_float(
                cfg.get("coord_gate_weight", context.extra.get("coord_gate_weight", 0.0)),
                0.0,
            ),
        ),
        "text_gate_weight": max(
            0.0,
            _coerce_float(
                cfg.get("text_gate_weight", context.extra.get("text_gate_weight", 0.0)),
                0.0,
            ),
        ),
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

    if weights["text_gate_weight"] != 0.0:
        masks = context.token_type_masks if isinstance(context.token_type_masks, Mapping) else {}
        mask_struct = masks.get("struct")
        mask_desc = masks.get("desc")
        mask_coord = masks.get("coord")
        mask_eos = masks.get("eos")

        weights_masked = state.get("weights_masked")

        mask_text: torch.Tensor | None = None
        if (
            isinstance(weights_masked, torch.Tensor)
            and isinstance(context.logits, torch.Tensor)
            and context.logits.ndim == 3
            and weights_masked.shape[:2] == context.logits.shape[:2]
        ):
            # Prefer token_ce-derived supervision mask so gate terms respect the same
            # FP/desc-masking semantics as CE (desc can be disabled by weight=0).
            mask_text = weights_masked.to(dtype=torch.bool)
        elif (
            isinstance(mask_struct, torch.Tensor)
            and isinstance(context.logits, torch.Tensor)
            and context.logits.ndim == 3
            and mask_struct.shape[:2] == context.logits.shape[:2]
        ):
            mask_text = mask_struct.to(dtype=torch.bool)
            if str(context.registry_context) != "self_context":
                if isinstance(mask_desc, torch.Tensor) and mask_desc.shape == mask_text.shape:
                    mask_text = mask_text | mask_desc.to(dtype=torch.bool)

        if isinstance(mask_text, torch.Tensor):
            if isinstance(mask_coord, torch.Tensor) and mask_coord.shape == mask_text.shape:
                mask_text = mask_text & (~mask_coord.to(dtype=torch.bool))
            if isinstance(mask_eos, torch.Tensor) and mask_eos.shape == mask_text.shape:
                mask_text = mask_text & (~mask_eos.to(dtype=torch.bool))

            mask_text_next = mask_text[:, 1:]
            if bool(mask_text_next.any().item()):
                logits_next = context.logits[:, :-1, :]
                flat_full = logits_next[mask_text_next]
                if int(flat_full.numel()) > 0:
                    vocab = int(flat_full.shape[-1])
                    coord_ids = torch.tensor(
                        [int(i) for i in context.coord_token_ids],
                        device=flat_full.device,
                        dtype=torch.long,
                    )
                    valid = (coord_ids >= 0) & (coord_ids < vocab)
                    coord_ids = coord_ids[valid]
                    if int(coord_ids.numel()) > 0:
                        flat_coord = flat_full.index_select(dim=-1, index=coord_ids)
                        p_coord = _stable_coord_mass_from_logits(
                            logits_full=flat_full,
                            logits_coord=flat_coord,
                            temperature=float(temperature),
                        )
                        one_minus = (1.0 - p_coord).clamp(min=1e-6, max=1.0)
                        gate_per_token = (-one_minus.log()).clamp(min=0.0, max=1e4)
                        gate_per_token = torch.nan_to_num(
                            gate_per_token, nan=0.0, posinf=1e4, neginf=0.0
                        )
                        text_gate = gate_per_token.mean().to(dtype=context.logits.dtype)

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

    coord_token_ce_contrib = weights["coord_ce_weight"] * coord_ce
    coord_soft_ce_contrib = weights["coord_soft_ce_weight"] * coord_soft_ce
    coord_w1_contrib = weights["coord_w1_weight"] * coord_w1
    coord_el1_contrib = weights["coord_el1_weight"] * coord_el1
    coord_ehuber_contrib = weights["coord_ehuber_weight"] * coord_ehuber
    coord_entropy_contrib = weights["coord_entropy_weight"] * coord_entropy
    coord_gate_contrib = weights["coord_gate_weight"] * coord_gate
    text_gate_contrib = weights["text_gate_weight"] * text_gate

    loss = (
        coord_token_ce_contrib
        + coord_soft_ce_contrib
        + coord_w1_contrib
        + coord_el1_contrib
        + coord_ehuber_contrib
        + coord_entropy_contrib
        + coord_gate_contrib
        + text_gate_contrib
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

    state = {
        "coord_reg": loss,
        "coord_token_ce_contrib": coord_token_ce_contrib,
        "coord_soft_ce_contrib": coord_soft_ce_contrib,
        "coord_w1_contrib": coord_w1_contrib,
        "coord_el1_contrib": coord_el1_contrib,
        "coord_ehuber_contrib": coord_ehuber_contrib,
        "coord_entropy_contrib": coord_entropy_contrib,
        "coord_gate_contrib": coord_gate_contrib,
        "text_gate_contrib": text_gate_contrib,
    }

    return ModuleResult(loss=loss, metrics=metrics, state=state)
