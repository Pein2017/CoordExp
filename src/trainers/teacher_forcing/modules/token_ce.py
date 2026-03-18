from __future__ import annotations

from typing import Any, Mapping, Optional

import torch
import torch.nn.functional as F

from ..contracts import ModuleResult, PipelineModuleSpec, TeacherForcingContext
from ..module_registry import normalize_token_ce_stop_signal_damping_config
from ..token_types import build_token_type_masks, iter_segment_views


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def run_token_ce_module(
    *,
    context: TeacherForcingContext,
    spec: PipelineModuleSpec,
) -> ModuleResult:
    input_ids = context.input_ids
    logits_ce = context.logits_ce
    meta = context.meta
    coord_id_set = context.coord_id_set

    cfg = spec.config if isinstance(spec.config, Mapping) else {}

    registry_context = str(context.registry_context or "").strip().lower()

    desc_ce_weight = max(
        0.0,
        _coerce_float(
            cfg.get("desc_ce_weight", context.extra.get("desc_ce_weight", 1.0)),
            1.0,
        ),
    )
    fn_desc_ce_weight = max(
        0.0,
        _coerce_float(
            cfg.get("rollout_fn_desc_weight", desc_ce_weight),
            desc_ce_weight,
        ),
    )
    matched_prefix_struct_ce_weight = max(
        0.0,
        _coerce_float(
            cfg.get("rollout_matched_prefix_struct_weight", 1.0),
            1.0,
        ),
    )
    struct_ce_weight = max(
        0.0,
        _coerce_float(cfg.get("struct_ce_weight", 0.0), 0.0),
    )
    stop_cfg = normalize_token_ce_stop_signal_damping_config(
        cfg.get("stop_signal_damping"),
        path="token_ce.config.stop_signal_damping",
    )
    stop_signal_enabled = bool(stop_cfg["enabled"]) and registry_context == "gt"

    labels_masked = torch.full_like(input_ids, -100)
    base_weights = input_ids.new_zeros(input_ids.shape, dtype=torch.float32)
    struct_weights = input_ids.new_zeros(input_ids.shape, dtype=torch.float32)
    desc_weights = input_ids.new_zeros(input_ids.shape, dtype=torch.float32)

    channel = str(context.channel or "").strip().upper()

    # Stage-2 Channel-A (self_context): optional format/EOS stabilizer.
    #
    # This is a separate forward with self-context logits. It is *struct-only* by
    # default (desc disabled) and is internally scaled by struct_ce_weight,
    # so that the pipeline sees a single module loss contribution.
    if registry_context == "self_context" and channel != "B":
        if float(struct_ce_weight) == 0.0:
            z = logits_ce.new_tensor(0.0)
            metrics = {
                "loss/token_ce_struct": 0.0,
                "loss/token_ce_desc": 0.0,
            }
            return ModuleResult(loss=z, metrics=metrics, state={})
        desc_ce_weight = 0.0
        fn_desc_ce_weight = 0.0
        stop_signal_enabled = False

    eligible_seq_count = 0
    stop_branch_specs: list[tuple[int, int, int, int]] = []

    for b, seg_start, seg_end, seg in iter_segment_views(input_ids=input_ids, meta=meta):
        prompt_len = int(seg.get("prompt_len", 0) or 0)
        prefix_len = int(seg.get("prefix_len", 0) or 0)
        train_len = int(seg.get("train_len", 0) or 0)

        prefix_struct_pos = [int(p) for p in (seg.get("prefix_struct_pos") or [])]
        tail_ignore_pos = [int(p) for p in (seg.get("tail_ignore_pos") or [])]
        tail_desc_pos = [int(p) for p in (seg.get("tail_desc_pos") or [])]
        tail_desc_weights = list(seg.get("tail_desc_weights") or [])
        tail_closure_pos = [int(p) for p in (seg.get("tail_closure_pos") or [])]

        seg_start_i = int(seg_start)
        seg_end_i = int(seg_end)

        prefix_start = max(seg_start_i + 1, min(seg_end_i, seg_start_i + prompt_len))
        prefix_end = max(
            prefix_start,
            min(seg_end_i, seg_start_i + prompt_len + prefix_len),
        )

        tail_start = max(
            prefix_end, min(seg_end_i, seg_start_i + prompt_len + prefix_len)
        )
        tail_end = max(
            tail_start,
            min(seg_end_i, seg_start_i + prompt_len + train_len),
        )

        if channel == "B" and prefix_struct_pos:
            for rel in prefix_struct_pos:
                p = int(prefix_start + rel)
                if p < int(prefix_start) or p >= int(prefix_end):
                    continue
                if int(input_ids[b, p].item()) in coord_id_set:
                    continue
                labels_masked[b, p] = input_ids[b, p]
                base_weights[b, p] = float(matched_prefix_struct_ce_weight)
                struct_weights[b, p] = float(matched_prefix_struct_ce_weight)

        tail_ignore = {int(x) for x in tail_ignore_pos if int(x) >= 0}
        tail_desc = {int(x) for x in tail_desc_pos if int(x) >= 0}
        tail_desc_weight_by_pos: dict[int, float] = {}
        if tail_desc_weights:
            if len(tail_desc_weights) != len(tail_desc_pos):
                raise ValueError(
                    "tail_desc_weights must align 1:1 with tail_desc_pos entries"
                )
            for rel_raw, weight_raw in zip(tail_desc_pos, tail_desc_weights):
                try:
                    rel_i = int(rel_raw)
                    weight_i = float(weight_raw)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "tail_desc_weights entries must be float-compatible"
                    ) from exc
                if rel_i < 0:
                    continue
                tail_desc_weight_by_pos[int(rel_i)] = float(weight_i)
        tail_closure = {int(x) for x in tail_closure_pos if int(x) >= 0}

        stop_rel_pos: Optional[int] = None
        if stop_signal_enabled:
            stop_rel_raw = seg.get("stop_rel_pos")
            stop_token_raw = seg.get("stop_token_id")
            continue_token_raw = seg.get("continue_token_id")
            if stop_rel_raw is None or stop_token_raw is None or continue_token_raw is None:
                raise ValueError(
                    "token_ce stop_signal_damping enabled but segment is missing stop-rel/token metadata"
                )
            try:
                stop_rel_pos = int(stop_rel_raw)
                stop_token_id = int(stop_token_raw)
                continue_token_id = int(continue_token_raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "token_ce stop_signal_damping metadata must be int-compatible"
                ) from exc
            if continue_token_id == stop_token_id:
                raise ValueError(
                    "token_ce stop_signal_damping requires distinct stop and continue token ids"
                )
            p_stop = int(tail_start + stop_rel_pos)
            if p_stop < int(tail_start) or p_stop >= int(tail_end):
                raise ValueError(
                    f"token_ce stop_signal_damping stop_rel_pos out of range: rel={int(stop_rel_pos)} "
                    f"tail=[{int(tail_start)},{int(tail_end)})"
                )
            if int(input_ids[b, p_stop].item()) != int(stop_token_id):
                raise ValueError(
                    "token_ce stop_signal_damping metadata does not match the supervised stop token id"
                )
            eligible_seq_count += 1
            stop_branch_specs.append(
                (int(b), int(p_stop), int(stop_token_id), int(continue_token_id))
            )

        for p in range(int(tail_start), int(tail_end)):
            tok_id = int(input_ids[b, p].item())
            if tok_id in coord_id_set:
                continue
            rel = int(p - tail_start)
            if rel in tail_ignore and rel not in tail_closure:
                continue

            labels_masked[b, p] = input_ids[b, p]

            if rel in tail_desc:
                desc_multiplier = float(tail_desc_weight_by_pos.get(rel, 1.0))
                w_desc = float(fn_desc_ce_weight if channel == "B" else desc_ce_weight)
                w_desc *= float(desc_multiplier)
                base_weights[b, p] = float(w_desc)
                desc_weights[b, p] = float(w_desc)
            elif stop_signal_enabled and stop_rel_pos is not None and rel == int(stop_rel_pos):
                # Semantic stop positions get their own optional bounded CE path and
                # are excluded from the ordinary structure bucket when enabled.
                continue
            else:
                base_weights[b, p] = 1.0
                struct_weights[b, p] = 1.0

        for rel in tail_closure:
            p = int(tail_start + rel)
            if p < int(tail_start) or p >= int(tail_end):
                continue
            if int(input_ids[b, p].item()) in coord_id_set:
                continue
            labels_masked[b, p] = input_ids[b, p]
            base_weights[b, p] = max(float(base_weights[b, p].item()), 1.0)
            struct_weights[b, p] = max(float(struct_weights[b, p].item()), 1.0)

    bsz, _, vocab = logits_ce.shape
    logits_next = logits_ce[:, :-1, :]
    labels_next = labels_masked[:, 1:]
    base_weights_next = base_weights[:, 1:]
    struct_weights_next = struct_weights[:, 1:]
    desc_weights_next = desc_weights[:, 1:]
    stop_weights_next = torch.zeros_like(base_weights_next)

    stop_signal_ce = logits_ce.new_tensor(0.0)
    token_ce_stop_signal = logits_ce.new_tensor(0.0)
    stop_weight_mean = logits_ce.new_tensor(0.0)
    stop_p_mean = logits_ce.new_tensor(0.0)
    stop_p_cont_mean = logits_ce.new_tensor(0.0)
    stop_margin_mean = logits_ce.new_tensor(0.0)

    branch_count = 0
    if stop_signal_enabled and stop_branch_specs:
        batch_idx = torch.tensor(
            [int(item[0]) for item in stop_branch_specs],
            device=logits_next.device,
            dtype=torch.long,
        )
        stop_pos = torch.tensor(
            [int(item[1]) for item in stop_branch_specs],
            device=logits_next.device,
            dtype=torch.long,
        )
        stop_row_idx = stop_pos - 1
        if bool((stop_row_idx < 0).any().item()):
            raise ValueError("token_ce stop_signal_damping produced a pre-shift stop row")

        stop_token_ids = torch.tensor(
            [int(item[2]) for item in stop_branch_specs],
            device=logits_next.device,
            dtype=torch.long,
        )
        continue_token_ids = torch.tensor(
            [int(item[3]) for item in stop_branch_specs],
            device=logits_next.device,
            dtype=torch.long,
        )

        stop_labels = labels_next[batch_idx, stop_row_idx]
        if not bool(torch.equal(stop_labels, stop_token_ids)):
            raise ValueError(
                "token_ce stop_signal_damping metadata/labels mismatch at stop rows"
            )

        branch_logits = logits_next[batch_idx, stop_row_idx, :]
        stop_logits = branch_logits.gather(1, stop_token_ids.view(-1, 1)).squeeze(1)
        continue_logits = branch_logits.gather(
            1, continue_token_ids.view(-1, 1)
        ).squeeze(1)

        branch_temperature = float(stop_cfg["branch_temperature"])
        margin = (stop_logits - continue_logits) / float(branch_temperature)
        p_stop = torch.sigmoid(margin)
        p_cont = 1.0 - p_stop

        gate_input = p_stop.detach() if bool(stop_cfg["detach_gate"]) else p_stop
        min_weight = float(stop_cfg["min_weight"])
        max_weight = float(stop_cfg["max_weight"])
        curve_gamma = float(stop_cfg["curve_gamma"])
        stop_weights_eff = float(min_weight) + (
            float(max_weight) - float(min_weight)
        ) * gate_input.pow(float(curve_gamma))

        stop_weights_next[batch_idx, stop_row_idx] = stop_weights_eff.to(
            dtype=stop_weights_next.dtype
        )
        branch_count = int(stop_weights_eff.numel())
        stop_weight_mean = stop_weights_eff.mean()
        stop_p_mean = p_stop.mean()
        stop_p_cont_mean = p_cont.mean()
        stop_margin_mean = margin.mean()

    total_weights_next = base_weights_next + stop_weights_next

    # Keep CE memory bounded by chunking flattened rows. This avoids materializing
    # a full [bsz*(seq-1)] per-token CE tensor for long packed sequences.
    flat_logits = logits_next.reshape(-1, vocab)
    flat_labels = labels_next.reshape(-1)
    flat_total_weights = total_weights_next.reshape(-1)
    flat_struct_weights = struct_weights_next.reshape(-1)
    flat_desc_weights = desc_weights_next.reshape(-1)
    flat_stop_weights = stop_weights_next.reshape(-1)

    # A small constant chunk keeps peak memory stable while preserving exact math.
    rows_per_chunk = 4096
    n_rows = int(flat_labels.numel())

    ce_num_total: Optional[torch.Tensor] = None
    ce_num_struct: Optional[torch.Tensor] = None
    ce_num_desc: Optional[torch.Tensor] = None
    ce_num_stop: Optional[torch.Tensor] = None
    for start in range(0, n_rows, rows_per_chunk):
        end = min(start + rows_per_chunk, n_rows)
        ce_chunk = F.cross_entropy(
            flat_logits[start:end],
            flat_labels[start:end],
            ignore_index=-100,
            reduction="none",
        )
        if ce_num_total is None:
            z = ce_chunk.new_tensor(0.0)
            ce_num_total = z
            ce_num_struct = z
            ce_num_desc = z
            ce_num_stop = z

        w_total_chunk = flat_total_weights[start:end].to(dtype=ce_chunk.dtype)
        w_struct_chunk = flat_struct_weights[start:end].to(dtype=ce_chunk.dtype)
        w_desc_chunk = flat_desc_weights[start:end].to(dtype=ce_chunk.dtype)
        w_stop_chunk = flat_stop_weights[start:end].to(dtype=ce_chunk.dtype)

        ce_num_total = ce_num_total + (ce_chunk * w_total_chunk).sum()
        ce_num_struct = ce_num_struct + (ce_chunk * w_struct_chunk).sum()
        ce_num_desc = ce_num_desc + (ce_chunk * w_desc_chunk).sum()
        ce_num_stop = ce_num_stop + (ce_chunk * w_stop_chunk).sum()

    if (
        ce_num_total is None
        or ce_num_struct is None
        or ce_num_desc is None
        or ce_num_stop is None
    ):
        z = logits_ce.new_tensor(0.0)
        ce_num_total = z
        ce_num_struct = z
        ce_num_desc = z
        ce_num_stop = z

    def _weighted_mean(num: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        denom = w.sum().to(dtype=num.dtype)
        if float(denom.detach().cpu().item()) <= 0.0:
            return num.new_tensor(0.0)
        return num / denom.clamp(min=1e-6)

    struct_ce = _weighted_mean(ce_num_struct, flat_struct_weights)
    desc_ce = _weighted_mean(ce_num_desc, flat_desc_weights)
    if branch_count > 0:
        stop_signal_ce = _weighted_mean(ce_num_stop, flat_stop_weights)

    # Decompose the token_ce objective into atomic contributions by token type.
    #
    # token_ce is a single weighted mean over (struct + desc + optional stop-signal)
    # supervised positions:
    #   token_ce = sum(per_tok * (w_struct + w_desc + w_stop)) / sum(...)
    #
    # We expose the corresponding per-type contributions using the *same* global
    # denominator, so the parts sum exactly to token_ce.
    denom_total = flat_total_weights.sum()
    if float(denom_total.detach().cpu().item()) > 0.0:
        denom_safe = denom_total.to(dtype=ce_num_total.dtype).clamp(min=1e-6)
        token_ce_struct = ce_num_struct / denom_safe
        token_ce_desc = ce_num_desc / denom_safe
        token_ce_stop_signal = ce_num_stop / denom_safe
    else:
        token_ce_struct = ce_num_total.new_tensor(0.0)
        token_ce_desc = ce_num_total.new_tensor(0.0)
        token_ce_stop_signal = ce_num_total.new_tensor(0.0)

    token_ce = token_ce_struct + token_ce_desc + token_ce_stop_signal

    token_masks = build_token_type_masks(
        input_ids=input_ids,
        meta=meta,
        coord_id_set=coord_id_set,
        channel=channel,
    )

    loss = token_ce

    token_ce_struct_contrib = token_ce_struct
    token_ce_desc_contrib = token_ce_desc
    token_ce_stop_signal_contrib = token_ce_stop_signal
    if registry_context == "self_context" and channel != "B":
        scale = float(struct_ce_weight)
        loss = loss * scale
        token_ce_struct_contrib = token_ce_struct_contrib * scale
        token_ce_desc_contrib = token_ce_desc_contrib * scale
        token_ce_stop_signal_contrib = token_ce_stop_signal_contrib * scale

    metrics = {
        "loss/token_ce": float(token_ce.detach().cpu().item()),
        "loss/struct_ce": float(struct_ce.detach().cpu().item()),
        "loss/desc_ce": float(desc_ce.detach().cpu().item()),
        "loss/token_ce_struct": float(token_ce_struct.detach().cpu().item()),
        "loss/token_ce_desc": float(token_ce_desc.detach().cpu().item()),
    }
    if branch_count > 0:
        metrics.update(
            {
                "loss/stop_signal_ce": float(stop_signal_ce.detach().cpu().item()),
                "loss/token_ce_stop_signal": float(
                    token_ce_stop_signal.detach().cpu().item()
                ),
                "stop_signal/eligible_seq_count": float(eligible_seq_count),
                "stop_signal/branch_count": float(branch_count),
                "stop_signal/weight_mean": float(stop_weight_mean.detach().cpu().item()),
                "stop_signal/p_stop_mean": float(stop_p_mean.detach().cpu().item()),
                "stop_signal/p_cont_mean": float(
                    stop_p_cont_mean.detach().cpu().item()
                ),
                "stop_signal/margin_mean": float(
                    stop_margin_mean.detach().cpu().item()
                ),
            }
        )

    stop_weights_masked = input_ids.new_zeros(input_ids.shape, dtype=torch.float32)
    stop_weights_masked[:, 1:] = stop_weights_next

    state = {
        "token_ce": loss,
        "struct_ce": struct_ce,
        "desc_ce": desc_ce,
        "token_ce_struct_contrib": token_ce_struct_contrib,
        "token_ce_desc_contrib": token_ce_desc_contrib,
        "labels_masked": labels_masked,
        "weights_masked": base_weights + stop_weights_masked,
        "token_type_masks": token_masks,
    }
    if branch_count > 0:
        state.update(
            {
                "stop_signal_ce": stop_signal_ce,
                "token_ce_stop_signal_contrib": token_ce_stop_signal_contrib,
            }
        )

    return ModuleResult(loss=loss, metrics=metrics, state=state)
