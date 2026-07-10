from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F

from src.coord_tokens.gaussian_rps import coord_gaussian_rps
from src.data_collators.token_types import TokenType
from src.detection.token_types import build_compact_token_type_groups

TOKEN_TYPE_TO_GATE_GROUP = {
    TokenType.DESC: "desc",
    TokenType.COORD: "coord",
    TokenType.FORMAT: "struct",
}

_TYPE_GATE_CHUNK_TOKENS = 64


def _zero_loss_like_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.numel() == 0:
        return torch.zeros((), device=logits.device, dtype=torch.float32)
    return logits.reshape(-1)[:1].float().sum() * 0.0


@dataclass(frozen=True)
class CoordGaussianRPSResult:
    coord_loss: torch.Tensor
    gaussian_contrib: torch.Tensor
    rps_contrib: torch.Tensor
    ce_contrib: torch.Tensor
    type_gate_contrib: torch.Tensor

    coord_tokens: int
    type_gate_tokens: int

    type_gate_allowed_mass_mean: torch.Tensor | None
    target_entropy: torch.Tensor | None
    target_peak_prob: torch.Tensor | None
    target_r95_radius_mean: torch.Tensor | None
    target_r95_radius_max: torch.Tensor | None
    coord_acc_top5: torch.Tensor | None
    coord_p_gt_mean: torch.Tensor | None
    coord_margin_mean: torch.Tensor | None
    coord_expected_bin_mae: torch.Tensor | None
    coord_expected_bin_abs_err_p90: torch.Tensor | None


def count_supervised_tokens(labels: torch.Tensor) -> int:
    if labels.ndim < 2:
        return 0
    labels_next = labels[:, 1:]
    return int((labels_next != -100).sum().detach().item())


def build_coord_id_map(
    *, vocab_size: int, device: torch.device, coord_token_ids: list[int]
) -> torch.Tensor:
    id_map = torch.full((int(vocab_size),), -1, dtype=torch.long, device=device)
    if not coord_token_ids:
        return id_map
    coord_ids = torch.tensor(coord_token_ids, device=device, dtype=torch.long)
    values = torch.arange(coord_ids.numel(), device=device, dtype=torch.long)
    valid = (coord_ids >= 0) & (coord_ids < int(vocab_size))
    if valid.any().item():
        id_map[coord_ids[valid]] = values[valid]
    return id_map


def _sanitize_nonnegative_int(
    value: int | float,
    *,
    cap: int,
    default: int,
) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError):
        out = int(default)
    return max(0, min(int(cap), out))


def infer_shape_aware_r95_radii(
    *,
    labels_next: torch.Tensor,
    target_bins_all: torch.Tensor,
    coord_positions_mask: torch.Tensor,
    gaussian_r95_axis_fraction: float,
    gaussian_r95_cap_bins: int,
    gaussian_r95_min_bins: int,
    gaussian_r95_fallback_bins: int,
) -> torch.Tensor:
    """Infer coord-token R95 radii from adjacent compact-full `xyxy` quads."""

    cap = _sanitize_nonnegative_int(
        gaussian_r95_cap_bins,
        cap=999,
        default=8,
    )
    min_bins = _sanitize_nonnegative_int(
        gaussian_r95_min_bins,
        cap=cap,
        default=1,
    )
    fallback = _sanitize_nonnegative_int(
        gaussian_r95_fallback_bins,
        cap=cap,
        default=cap,
    )
    fallback = max(min_bins, fallback)
    fraction = float(gaussian_r95_axis_fraction)
    if not math.isfinite(fraction) or fraction <= 0.0:
        raise ValueError("gaussian_r95_axis_fraction must be finite and > 0")

    radius_all = torch.full_like(
        target_bins_all,
        fill_value=int(fallback),
        dtype=torch.long,
    )
    batch, _seq_len = target_bins_all.shape
    for row in range(int(batch)):
        idxs = torch.nonzero(coord_positions_mask[row], as_tuple=False).flatten()
        if int(idxs.numel()) < 4:
            continue
        values = target_bins_all[row]
        labels_row = labels_next[row]
        start = 0
        while start < int(idxs.numel()):
            end = start + 1
            while (
                end < int(idxs.numel())
                and int(idxs[end].item()) == int(idxs[end - 1].item()) + 1
            ):
                end += 1
            run = idxs[start:end]
            usable = (int(run.numel()) // 4) * 4
            if usable > 0:
                quads = run[:usable].view(-1, 4)
                for quad in quads:
                    if not bool((labels_row[quad] != -100).all().item()):
                        continue
                    x1, y1, x2, y2 = [int(v) for v in values[quad].tolist()]
                    if not (0 <= x1 < x2 <= 999 and 0 <= y1 < y2 <= 999):
                        continue

                    def _axis_radius(length: int) -> int:
                        raw = int(math.floor(min(float(cap), fraction * float(length))))
                        return max(int(min_bins), raw)

                    x_radius = _axis_radius(x2 - x1)
                    y_radius = _axis_radius(y2 - y1)
                    radius_all[row, quad[0]] = x_radius
                    radius_all[row, quad[2]] = x_radius
                    radius_all[row, quad[1]] = y_radius
                    radius_all[row, quad[3]] = y_radius
            start = end
    return radius_all[coord_positions_mask]


def resolve_standard_ce_type_gate_groups(
    *,
    labels_next: torch.Tensor,
    token_types_next: torch.Tensor | None,
    tokenizer: Any,
    coord_id_map: torch.Tensor,
) -> tuple[str, ...]:
    groups = build_compact_token_type_groups(tokenizer)
    flat_labels = labels_next.reshape(-1)
    flat_types = (
        token_types_next.reshape(-1)
        if isinstance(token_types_next, torch.Tensor)
        and tuple(token_types_next.shape) == tuple(labels_next.shape)
        else None
    )
    out: list[str] = []
    coord_id_map_cpu = coord_id_map.detach().cpu()
    struct_ids = set(int(token_id) for token_id in groups.struct)
    coord_ids = set(int(token_id) for token_id in groups.coord)
    desc_ids = set(int(token_id) for token_id in groups.desc)
    eos_ids = set(int(token_id) for token_id in groups.eos)
    for index, raw_label in enumerate(flat_labels.detach().cpu().tolist()):
        label = int(raw_label)
        if label == -100:
            out.append("ignore")
            continue
        if 0 <= label < int(coord_id_map_cpu.numel()) and int(coord_id_map_cpu[label]) >= 0:
            out.append("coord")
            continue
        if label in eos_ids:
            out.append("eos")
            continue
        token_type = None
        if flat_types is not None:
            token_type = int(flat_types.detach().cpu()[index].item())
        mapped = TOKEN_TYPE_TO_GATE_GROUP.get(token_type)
        if mapped is not None:
            out.append(mapped)
            continue
        if label in struct_ids:
            out.append("struct")
        elif label in coord_ids:
            out.append("coord")
        elif label in desc_ids:
            out.append("desc")
        elif label in eos_ids:
            out.append("eos")
        else:
            out.append("ignore")
    return tuple(out)


def _cfg_value(config: Any, name: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _weight(config: Any, name: str) -> float:
    weights = _cfg_value(config, "weights")
    value = _cfg_value(weights, name, 0.0)
    weight = float(value)
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError(f"type_gate.weights.{name} must be finite and >= 0")
    return weight


def _group_token_ids(
    *,
    group_name: str,
    tokenizer: Any,
    vocab_size: int,
    device: torch.device,
) -> torch.Tensor:
    groups = build_compact_token_type_groups(tokenizer)
    raw_ids = {
        "struct": groups.struct,
        "coord": groups.coord,
        "desc": groups.desc,
        "eos": groups.eos,
    }[group_name]
    valid_ids = [
        int(token_id)
        for token_id in raw_ids
        if 0 <= int(token_id) < int(vocab_size)
    ]
    return torch.tensor(sorted(set(valid_ids)), device=device, dtype=torch.long)


def _compute_type_gate_contrib(
    *,
    logits_next: torch.Tensor,
    labels_next: torch.Tensor,
    token_types_next: torch.Tensor | None,
    tokenizer: Any,
    coord_id_map: torch.Tensor,
    cfg: Any,
    average_tokens_across_devices: bool,
    model_accepts_loss_kwargs: bool,
    accelerator_num_processes: int | None,
) -> tuple[torch.Tensor, int, torch.Tensor | None]:
    if not bool(_cfg_value(cfg, "enabled", False)):
        zero = _zero_loss_like_logits(logits_next)
        return zero, 0, None

    vocab_size = int(logits_next.shape[-1])
    flat_logits = logits_next.reshape(-1, vocab_size)
    flat_labels = labels_next.reshape(-1)
    group_names = resolve_standard_ce_type_gate_groups(
        labels_next=labels_next,
        token_types_next=token_types_next,
        tokenizer=tokenizer,
        coord_id_map=coord_id_map,
    )
    if len(group_names) != int(flat_labels.numel()):
        raise RuntimeError("type-gate group resolution did not align with labels")

    weights = {
        "struct": _weight(cfg, "struct"),
        "coord": _weight(cfg, "coord"),
        "desc": _weight(cfg, "desc"),
        "eos": _weight(cfg, "eos"),
    }
    chunk_tokens = max(1, int(_TYPE_GATE_CHUNK_TOKENS))
    loss_sum = flat_logits.new_zeros((), dtype=torch.float32)
    mass_sum = flat_logits.new_zeros((), dtype=torch.float32)
    count_local = 0
    token_id_cache: dict[str, torch.Tensor] = {}
    for group_name, weight in weights.items():
        if weight == 0.0:
            continue
        mask_values = [
            index
            for index, name in enumerate(group_names)
            if name == group_name and int(flat_labels[index].item()) != -100
        ]
        if not mask_values:
            continue
        type_ids = token_id_cache.get(group_name)
        if type_ids is None:
            type_ids = _group_token_ids(
                group_name=group_name,
                tokenizer=tokenizer,
                vocab_size=vocab_size,
                device=flat_logits.device,
            )
            token_id_cache[group_name] = type_ids
        if int(type_ids.numel()) == 0:
            continue
        row_ids_all = torch.tensor(mask_values, device=flat_logits.device, dtype=torch.long)
        for start in range(0, int(row_ids_all.numel()), chunk_tokens):
            row_ids = row_ids_all[start : start + chunk_tokens]
            chunk_logits = flat_logits.index_select(0, row_ids)
            safe_logits = torch.nan_to_num(
                chunk_logits.float(),
                nan=0.0,
                posinf=1e4,
                neginf=-1e4,
            ).clamp(min=-1e4, max=1e4)
            log_probs = F.log_softmax(safe_logits, dim=-1)
            selected = log_probs.index_select(-1, type_ids)
            log_mass = torch.logsumexp(selected, dim=-1)
            loss_sum = loss_sum + (-log_mass * float(weight)).sum()
            mass_sum = mass_sum + torch.exp(log_mass.clamp(min=-50.0, max=0.0)).sum()
            count_local += int(row_ids.numel())

    if count_local <= 0:
        zero = _zero_loss_like_logits(logits_next)
        return zero, 0, None

    denom_local = torch.tensor(
        float(count_local),
        device=loss_sum.device,
        dtype=loss_sum.dtype,
    )
    denom = denom_local
    if (
        average_tokens_across_devices
        and model_accepts_loss_kwargs
        and dist.is_available()
        and dist.is_initialized()
    ):
        denom = denom_local.detach().clone()
        dist.all_reduce(denom, op=dist.ReduceOp.SUM)
    denom = torch.where(denom > 0, denom, denom.new_tensor(1.0))
    contrib = loss_sum / denom
    if average_tokens_across_devices and model_accepts_loss_kwargs:
        if dist.is_available() and dist.is_initialized():
            scale = float(dist.get_world_size())
        else:
            scale = float(accelerator_num_processes or 1)
        contrib = contrib * scale
    mass_mean = mass_sum / denom_local.clamp(min=1.0)
    contrib = torch.nan_to_num(contrib, nan=0.0, posinf=1e4, neginf=0.0)
    mass_mean = torch.nan_to_num(mass_mean, nan=0.0, posinf=1.0, neginf=0.0)
    return contrib, count_local, mass_mean


def _distributed_component(
    *,
    loss_sum: torch.Tensor,
    denom: torch.Tensor,
    average_tokens_across_devices: bool,
    model_accepts_loss_kwargs: bool,
    accelerator_num_processes: int | None,
) -> torch.Tensor:
    denom = torch.where(denom > 0, denom, denom.new_tensor(1.0))
    loss = loss_sum / denom
    if average_tokens_across_devices and model_accepts_loss_kwargs:
        if dist.is_available() and dist.is_initialized():
            scale = float(dist.get_world_size())
        else:
            scale = float(accelerator_num_processes or 1)
        loss = loss * scale
    return torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=0.0)


def compute_coord_gaussian_rps_loss(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    masked_labels: torch.Tensor,
    coord_token_weights: torch.Tensor | None,
    coord_token_ids: list[int],
    coord_id_map: torch.Tensor,
    tokenizer: Any | None,
    token_types: torch.Tensor | None = None,
    cfg: Any,
    average_tokens_across_devices: bool,
    model_accepts_loss_kwargs: bool,
    accelerator_num_processes: int | None,
) -> CoordGaussianRPSResult | None:
    if not isinstance(logits, torch.Tensor) or not isinstance(labels, torch.Tensor):
        raise TypeError("logits and labels must be torch.Tensors")
    if not isinstance(masked_labels, torch.Tensor):
        raise TypeError("masked_labels must be a torch.Tensor")
    if tuple(masked_labels.shape) != tuple(labels.shape):
        raise ValueError("masked_labels must match labels shape")
    if tokenizer is None:
        raise ValueError("coord_gaussian_rps requires a tokenizer for compact type gate")

    seq_len = min(int(logits.shape[1]), max(int(labels.shape[1]) - 1, 0))
    if seq_len <= 0:
        return None

    logits_next = logits[:, :seq_len, :]
    labels_next = labels[:, 1 : seq_len + 1]
    token_types_next = None
    if isinstance(token_types, torch.Tensor) and tuple(token_types.shape) == tuple(labels.shape):
        token_types_next = token_types[:, 1 : seq_len + 1]

    vocab_size = int(logits_next.shape[-1])
    if not coord_token_ids:
        raise RuntimeError("coord_gaussian_rps enabled but no coord token ids were provided")
    if max(coord_token_ids) >= vocab_size:
        raise ValueError(
            f"coord_gaussian_rps enabled but coord token ids exceed vocab_size={vocab_size}"
        )

    labels_safe = labels_next.clamp(min=0)
    target_bins_all = coord_id_map[labels_safe].to(dtype=torch.long)
    coord_positions_mask = (target_bins_all >= 0) & (labels_next != -100)

    coord_position_weights = None
    if isinstance(coord_token_weights, torch.Tensor):
        if tuple(coord_token_weights.shape) != tuple(labels.shape):
            raise ValueError("coord_token_weights must match labels shape when provided")
        coord_weights_next = coord_token_weights[:, 1 : seq_len + 1]
        coord_position_weights = coord_weights_next[coord_positions_mask].to(
            dtype=torch.float32
        )

    if coord_position_weights is None:
        denom_local = coord_positions_mask.sum().to(dtype=torch.float32)
    else:
        denom_local = coord_position_weights.sum().to(dtype=torch.float32)
    denom = denom_local
    if (
        average_tokens_across_devices
        and model_accepts_loss_kwargs
        and dist.is_available()
        and dist.is_initialized()
    ):
        denom = denom_local.detach().clone()
        dist.all_reduce(denom, op=dist.ReduceOp.SUM)

    zero = logits_next.float().sum() * 0.0
    ce_contrib = zero
    gaussian_contrib = zero
    rps_contrib = zero
    target_entropy = None
    target_peak_prob = None
    target_r95_radius_mean = None
    target_r95_radius_max = None
    coord_acc_top5 = None
    coord_p_gt_mean = None
    coord_margin_mean = None
    coord_expected_bin_mae = None
    coord_expected_bin_abs_err_p90 = None
    coord_tokens = int(coord_positions_mask.sum().detach().item())

    if float(denom_local.detach().item()) > 0.0:
        flat_logits_full = logits_next[coord_positions_mask]
        flat_target_bins = target_bins_all[coord_positions_mask]
        if coord_position_weights is None:
            coord_position_weights = torch.ones(
                (int(flat_target_bins.numel()),),
                dtype=torch.float32,
                device=flat_target_bins.device,
            )
        else:
            coord_position_weights = coord_position_weights.to(device=flat_target_bins.device)

        coord_ids = torch.tensor(
            coord_token_ids,
            device=flat_logits_full.device,
            dtype=torch.long,
        )
        flat_logits = flat_logits_full.index_select(-1, coord_ids)
        radii = infer_shape_aware_r95_radii(
            labels_next=labels_next,
            target_bins_all=target_bins_all,
            coord_positions_mask=coord_positions_mask,
            gaussian_r95_axis_fraction=float(
                getattr(cfg, "gaussian_r95_axis_fraction", 0.04)
            ),
            gaussian_r95_cap_bins=int(getattr(cfg, "gaussian_r95_cap_bins", 8)),
            gaussian_r95_min_bins=int(getattr(cfg, "gaussian_r95_min_bins", 1)),
            gaussian_r95_fallback_bins=int(
                getattr(cfg, "gaussian_r95_fallback_bins", 8)
            ),
        ).to(device=flat_logits.device)
        temperature = float(getattr(cfg, "temperature", 1.0))
        out = coord_gaussian_rps(
            flat_logits,
            flat_target_bins,
            radii,
            temperature=temperature,
            gaussian_weight=1.0,
            rps_weight=1.0,
            normalize_rps=True,
        )
        weights = coord_position_weights.to(dtype=out.gaussian_ce_per_token.dtype)
        gaussian_sum = (out.gaussian_ce_per_token * weights).sum()
        rps_sum = (out.rps_per_token * weights.to(dtype=out.rps_per_token.dtype)).sum()
        ce_sum = gaussian_sum.new_tensor(0.0)
        if float(getattr(cfg, "ce_weight", 0.0)) != 0.0:
            ce_per_token = F.cross_entropy(
                flat_logits.float(),
                flat_target_bins,
                reduction="none",
            )
            ce_per_token = torch.nan_to_num(
                ce_per_token,
                nan=0.0,
                posinf=1e4,
                neginf=0.0,
            )
            ce_sum = (ce_per_token * weights.to(dtype=ce_per_token.dtype)).sum()

        ce_contrib = float(getattr(cfg, "ce_weight", 0.0)) * _distributed_component(
            loss_sum=ce_sum,
            denom=denom,
            average_tokens_across_devices=average_tokens_across_devices,
            model_accepts_loss_kwargs=model_accepts_loss_kwargs,
            accelerator_num_processes=accelerator_num_processes,
        )
        gaussian_contrib = float(
            getattr(cfg, "gaussian_weight", 1.0)
        ) * _distributed_component(
            loss_sum=gaussian_sum,
            denom=denom,
            average_tokens_across_devices=average_tokens_across_devices,
            model_accepts_loss_kwargs=model_accepts_loss_kwargs,
            accelerator_num_processes=accelerator_num_processes,
        )
        rps_contrib = float(getattr(cfg, "rps_weight", 1.0)) * _distributed_component(
            loss_sum=rps_sum,
            denom=denom,
            average_tokens_across_devices=average_tokens_across_devices,
            model_accepts_loss_kwargs=model_accepts_loss_kwargs,
            accelerator_num_processes=accelerator_num_processes,
        )

        with torch.no_grad():
            probs = out.pred_probs.to(dtype=torch.float32)
            bins = torch.arange(
                int(probs.shape[-1]),
                device=probs.device,
                dtype=torch.float32,
            )
            k5 = min(5, int(flat_logits.shape[-1]))
            top5 = flat_logits.topk(k=k5, dim=-1).indices
            coord_acc_top5 = (
                top5 == flat_target_bins.view(-1, 1)
            ).any(dim=-1).float().mean()
            coord_p_gt_mean = probs.gather(1, flat_target_bins.view(-1, 1)).mean()
            pred_expected = (probs * bins.view(1, -1)).sum(dim=-1)
            abs_err = (pred_expected - flat_target_bins.to(dtype=torch.float32)).abs()
            coord_expected_bin_mae = abs_err.mean()
            if abs_err.numel() > 0:
                coord_expected_bin_abs_err_p90 = torch.quantile(abs_err, 0.9)
            logits_scaled = flat_logits.float() / max(float(temperature), 1e-6)
            gt_logit = logits_scaled.gather(1, flat_target_bins.view(-1, 1)).squeeze(1)
            max_logit = logits_scaled.max(dim=-1).values
            coord_margin_mean = (max_logit - gt_logit).mean()
            positive_probs = out.target_probs > 0
            target_entropy = -(
                out.target_probs[positive_probs] * out.target_probs[positive_probs].log()
            ).sum() / float(max(1, int(out.target_probs.shape[0])))
            target_peak_prob = out.target_probs.max(dim=-1).values.mean()
            target_r95_radius_mean = radii.to(dtype=torch.float32).mean()
            target_r95_radius_max = radii.to(dtype=torch.float32).max()

    type_gate_cfg = getattr(cfg, "type_gate", None)
    type_gate_contrib, type_gate_tokens, type_gate_allowed_mass_mean = (
        _compute_type_gate_contrib(
            logits_next=logits_next,
            labels_next=labels_next,
            token_types_next=token_types_next,
            tokenizer=tokenizer,
            coord_id_map=coord_id_map,
            cfg=type_gate_cfg,
            average_tokens_across_devices=average_tokens_across_devices,
            model_accepts_loss_kwargs=model_accepts_loss_kwargs,
            accelerator_num_processes=accelerator_num_processes,
        )
    )

    if coord_tokens <= 0 and type_gate_tokens <= 0:
        return None

    coord_loss = ce_contrib + gaussian_contrib + rps_contrib + type_gate_contrib
    coord_loss = torch.nan_to_num(coord_loss, nan=0.0, posinf=1e4, neginf=0.0)
    return CoordGaussianRPSResult(
        coord_loss=coord_loss,
        gaussian_contrib=torch.nan_to_num(
            gaussian_contrib,
            nan=0.0,
            posinf=1e4,
            neginf=0.0,
        ),
        rps_contrib=torch.nan_to_num(rps_contrib, nan=0.0, posinf=1e4, neginf=0.0),
        ce_contrib=torch.nan_to_num(ce_contrib, nan=0.0, posinf=1e4, neginf=0.0),
        type_gate_contrib=torch.nan_to_num(
            type_gate_contrib,
            nan=0.0,
            posinf=1e4,
            neginf=0.0,
        ),
        coord_tokens=coord_tokens,
        type_gate_tokens=int(type_gate_tokens),
        type_gate_allowed_mass_mean=type_gate_allowed_mass_mean,
        target_entropy=target_entropy,
        target_peak_prob=target_peak_prob,
        target_r95_radius_mean=target_r95_radius_mean,
        target_r95_radius_max=target_r95_radius_max,
        coord_acc_top5=coord_acc_top5,
        coord_p_gt_mean=coord_p_gt_mean,
        coord_margin_mean=coord_margin_mean,
        coord_expected_bin_mae=coord_expected_bin_mae,
        coord_expected_bin_abs_err_p90=coord_expected_bin_abs_err_p90,
    )


__all__ = [
    "CoordGaussianRPSResult",
    "build_coord_id_map",
    "compute_coord_gaussian_rps_loss",
    "count_supervised_tokens",
    "infer_shape_aware_r95_radii",
    "resolve_standard_ce_type_gate_groups",
]
