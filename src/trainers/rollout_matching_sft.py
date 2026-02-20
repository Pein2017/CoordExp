"""Rollout-matching SFT trainer (stage_2).

Implements the OpenSpec change:
  openspec/changes/2026-01-15-add-rollout-matching-trainer

High-level loop per batch:
  rollout (no grad) -> strict token-aligned parse -> Hungarian match -> build Y_train
  -> one teacher-forced forward -> masked CE + distributional coord losses.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from contextlib import contextmanager, nullcontext
from copy import copy as shallow_copy
from copy import deepcopy
from dataclasses import asdict
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
import torch
import torch.nn.functional as F
from swift.trainers import Seq2SeqTrainer
from swift.trainers.rlhf_trainer.utils import (
    get_gather_if_zero3_context,
    replace_assistant_response_with_ids,
)
from swift.utils import get_logger, unwrap_model_for_generation

from src.common.object_field_order import normalize_object_field_order
from src.common.geometry import bbox_from_points, flatten_points
from src.coord_tokens.codec import (
    get_coord_token_ids,
    token_to_int,
)
from src.coord_tokens.soft_ce_w1 import coord_soft_ce_w1

from .rollout_matching.contracts import (
    GTObject,
    ParsedPredObject,
)
from .rollout_matching.matching import _mask_iou_norm1000, hungarian_match_maskiou
from .rollout_matching.packing import (
    DropRemainderAccumulationWindow as _DropRemainderAccumulationWindow,
)
from .rollout_matching.parsing import (
    coerce_int as _coerce_int,
    find_desc_value_token_positions as _find_desc_value_token_positions,
    parse_rollout_for_matching,
    points_from_coord_tokens as _points_from_coord_tokens,
    serialize_append_fragment as _serialize_append_fragment,
)
from .rollout_matching.telemetry import (
    PendingTrainRolloutLog as _PendingTrainRolloutLog,
)

logger = get_logger()


def _contiguous_chunk_slices(n: int, num_chunks: int) -> List[Tuple[int, int]]:
    """Deterministically slice `range(n)` into `num_chunks` contiguous chunks.

    This is the normative chunking used for multi-server vLLM rollout distribution.
    It preserves order and is stable across runs.

    Returns a list of (start, end) index pairs of length `num_chunks`.
    """
    if num_chunks <= 0:
        raise ValueError("num_chunks must be > 0")
    if n < 0:
        raise ValueError("n must be >= 0")
    if n == 0:
        return [(0, 0) for _ in range(int(num_chunks))]

    chunk_size = int((n + num_chunks - 1) // num_chunks)
    out: List[Tuple[int, int]] = []
    for i in range(int(num_chunks)):
        start = min(int(i * chunk_size), int(n))
        end = min(int((i + 1) * chunk_size), int(n))
        if end < start:
            end = start
        out.append((start, end))
    return out


def _contiguous_weighted_chunk_slices(
    n: int, weights: Sequence[int]
) -> List[Tuple[int, int]]:
    """Deterministically slice `range(n)` into weighted contiguous chunks.

    `weights[i]` expresses the relative capacity of chunk i.

    Contract:
    - preserves order (contiguous slices)
    - stable across runs given the same `n` and `weights`
    - sums to exactly `n`

    Returns a list of (start, end) index pairs of length `len(weights)`.
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    if not isinstance(weights, (list, tuple)) or not weights:
        raise ValueError("weights must be a non-empty list")

    ws: List[int] = []
    for i, w_raw in enumerate(weights):
        try:
            w = int(w_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"weights[{int(i)}] must be an int") from exc
        if w < 0:
            raise ValueError(f"weights[{int(i)}] must be >= 0")
        ws.append(int(w))

    # Degenerate case: all weights are 0. Fall back to uniform chunking.
    total = int(sum(ws))
    if total <= 0:
        return _contiguous_chunk_slices(int(n), int(len(ws)))

    if n == 0:
        return [(0, 0) for _ in range(int(len(ws)))]

    # Base allocation via floor, then distribute the remainder by largest fractional part.
    base_counts: List[int] = [int((int(n) * int(w)) // total) for w in ws]
    remainder = int(n) - int(sum(base_counts))
    if remainder < 0:
        remainder = 0

    # Fractional parts are (n*w % total). Larger means closer to receiving an extra item.
    frac_rank: List[Tuple[int, int]] = [
        (int((int(n) * int(w)) % total), int(i)) for i, w in enumerate(ws)
    ]
    frac_rank.sort(key=lambda x: (-int(x[0]), int(x[1])))
    for k in range(int(remainder)):
        _frac, idx = frac_rank[int(k % len(frac_rank))]
        base_counts[int(idx)] += 1

    # Convert counts to contiguous slices.
    out: List[Tuple[int, int]] = []
    start = 0
    for c in base_counts:
        end = int(start + int(c))
        out.append((int(start), int(end)))
        start = end

    # Strict sanity check (should always hold).
    if out and int(out[-1][1]) != int(n):
        raise RuntimeError(
            "weighted chunking produced invalid slices: "
            f"n={int(n)} weights={ws} slices={out}"
        )

    return out


def _per_server_rank_request_caps(
    *,
    per_rank_chunk_size: int,
    server_world_sizes: Sequence[int],
    learner_world_size: int,
    learner_rank: int,
) -> List[int]:
    """Compute strict per-server request caps for one learner rank.

    We project one optimizer-step "global budget" onto learner ranks and servers via
    contiguous slices so each rank gets exactly `per_rank_chunk_size` requests while
    preserving deterministic server weighting.

    Let:
      - `W = learner_world_size`
      - `r = learner_rank`
      - `C = per_rank_chunk_size`
      - `global_budget = W * C`

    We first split `[0, global_budget)` into server-weighted contiguous slices using
    `server_world_sizes`, then take the overlap with this rank's interval
    `[r*C, (r+1)*C)`.

    Properties:
      - per-rank total equals exactly `C`
      - server totals across ranks equal the weighted global split
      - deterministic given `(C, server_world_sizes, W, r)`
    """
    chunk = int(max(0, int(per_rank_chunk_size)))
    world = int(max(1, int(learner_world_size)))
    rank = int(max(0, int(learner_rank)))

    if world > 0:
        rank = min(rank, world - 1)

    ws = [int(max(1, int(x))) for x in server_world_sizes]
    if not ws:
        return []
    if chunk == 0:
        return [0 for _ in ws]

    global_budget = int(world * chunk)
    server_slices = _contiguous_weighted_chunk_slices(int(global_budget), ws)

    rank_start = int(rank * chunk)
    rank_end = int(rank_start + chunk)

    out: List[int] = []
    for start, end in server_slices:
        overlap = max(0, min(int(rank_end), int(end)) - max(int(rank_start), int(start)))
        out.append(int(overlap))

    if int(sum(out)) != int(chunk):
        raise RuntimeError(
            "invalid per-server rank cap allocation: "
            f"chunk={int(chunk)} world={int(world)} rank={int(rank)} "
            f"server_world_sizes={ws} caps={out}"
        )

    return out


def _allocate_weighted_counts_with_caps(n: int, caps: Sequence[int]) -> List[int]:
    """Allocate `n` contiguous requests across servers with strict per-server caps."""
    n_i = int(n)
    if n_i < 0:
        raise ValueError("n must be >= 0")
    caps_i: List[int] = []
    for idx, raw in enumerate(caps):
        try:
            c = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"caps[{int(idx)}] must be an int") from exc
        if c < 0:
            raise ValueError(f"caps[{int(idx)}] must be >= 0")
        caps_i.append(int(c))

    total_cap = int(sum(caps_i))
    if n_i > total_cap:
        raise ValueError(
            "requested batch exceeds strict per-server cap budget: "
            f"n={n_i} total_cap={total_cap} caps={caps_i}"
        )
    if n_i == 0 or total_cap == 0:
        return [0 for _ in caps_i]

    pos: List[Tuple[int, int]] = [
        (int(i), int(c)) for i, c in enumerate(caps_i) if int(c) > 0
    ]
    if not pos:
        return [0 for _ in caps_i]

    pos_idx = [int(i) for i, _c in pos]
    pos_caps = [int(c) for _i, c in pos]

    chunks = _contiguous_weighted_chunk_slices(int(n_i), pos_caps)
    out = [0 for _ in caps_i]
    for local_i, (start, end) in enumerate(chunks):
        idx = int(pos_idx[local_i])
        cnt = int(end - start)
        if cnt < 0 or cnt > int(caps_i[idx]):
            raise RuntimeError(
                "invalid weighted capped allocation: "
                f"idx={idx} cnt={cnt} cap={caps_i[idx]} n={n_i} caps={caps_i}"
            )
        out[idx] = int(cnt)

    if int(sum(out)) != int(n_i):
        raise RuntimeError(
            "invalid weighted capped allocation total: "
            f"sum={int(sum(out))} n={int(n_i)} caps={caps_i}"
        )
    return out


def _strip_trailing_assistant_turns_for_rollout(messages: Any) -> List[Any]:
    """Build a prompt-only message list for rollout generation.

    Many training datasets include a teacher-forced assistant answer inside
    `sample["messages"]`. For rollouts (on-policy decoding), we must generate from
    the prompt, which should end with a user turn. This helper keeps all messages
    up to the last user turn and drops any trailing assistant turns.

    This is intentionally conservative: it preserves any earlier assistant turns
    that may be part of the conversational context.
    """

    if not isinstance(messages, list):
        return list(messages) if messages is not None else []

    last_user_idx: int | None = None
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if isinstance(m, dict) and m.get("role") == "user":
            last_user_idx = int(i)
            break

    if last_user_idx is None:
        # Best-effort: drop assistant messages entirely (rare in our datasets).
        trimmed: List[Any] = []
        for m in messages:
            if isinstance(m, dict) and m.get("role") == "assistant":
                continue
            trimmed.append(m)
        return trimmed if trimmed else list(messages)

    return list(messages[: last_user_idx + 1])


def _ensure_system_prompt_message(messages: Any, system_prompt: str) -> List[Any]:
    """Prepend a system prompt message when absent.

    HF template.encode() injects the resolved template/system prompt internally.
    In vLLM backends (colocate/server), we send raw OpenAI-style messages to
    ms-swift/vLLM, so we must include the system instruction explicitly to keep
    output formatting stable (e.g., top-level {"objects": [...]} CoordJSON).

    NOTE: ms-swift's rollout server expects the system message `content` to be a
    plain string (OpenAI-style), not the multimodal `[{'type':'text',...}]` list.
    """

    if not system_prompt:
        return list(messages) if isinstance(messages, list) else []

    if not isinstance(messages, list):
        messages_list: List[Any] = []
    else:
        messages_list = list(messages)

    for m in messages_list:
        if isinstance(m, dict) and m.get("role") == "system":
            return messages_list

    sys_msg = {"role": "system", "content": str(system_prompt)}
    return [sys_msg, *messages_list]


_IM_END = "<|im_end|>"


def _sinkhorn_barycentric_targets(
    *,
    pred_points: np.ndarray,  # [N,2] in norm1000
    gt_points: np.ndarray,  # [M,2] in norm1000
    epsilon: float,
    iters: int,
    cost: Literal["l1", "l2"],
) -> np.ndarray:
    """Compute barycentric-projected GT targets for each pred point via Sinkhorn OT."""
    if pred_points.size == 0 or gt_points.size == 0:
        return pred_points.copy()
    eps = float(epsilon)
    if not math.isfinite(eps) or eps <= 0:
        eps = 1.0
    n_iter = max(1, int(iters))

    p = torch.tensor(pred_points, dtype=torch.float32)
    g = torch.tensor(gt_points, dtype=torch.float32)
    if cost == "l1":
        c = torch.cdist(p, g, p=1)
    else:
        c = torch.cdist(p, g, p=2)

    # Uniform marginals.
    n = p.shape[0]
    m = g.shape[0]
    a = torch.full((n,), 1.0 / float(n), dtype=torch.float32)
    b = torch.full((m,), 1.0 / float(m), dtype=torch.float32)

    k = torch.exp((-c / eps).clamp(min=-50.0, max=50.0))
    u = torch.ones_like(a)
    v = torch.ones_like(b)
    for _ in range(n_iter):
        kv = k @ v
        kv = torch.where(kv > 0, kv, torch.ones_like(kv))
        u = a / kv
        ku = k.t() @ u
        ku = torch.where(ku > 0, ku, torch.ones_like(ku))
        v = b / ku

    t = (u[:, None] * k) * v[None, :]
    row_sum = t.sum(dim=1, keepdim=True)
    row_sum = torch.where(row_sum > 0, row_sum, torch.ones_like(row_sum))
    w = t / row_sum
    g_hat = w @ g
    return g_hat.detach().cpu().numpy()


def _extract_gt_objects(sample: Mapping[str, Any]) -> List[GTObject]:
    payload = sample.get("assistant_payload")
    if not isinstance(payload, Mapping):
        raise ValueError("rollout-matching requires assistant_payload in each sample")
    objects_raw = payload.get("objects")
    if not isinstance(objects_raw, list):
        raise ValueError("assistant_payload must contain top-level 'objects' list")

    objs: List[GTObject] = []
    for idx, entry in enumerate(objects_raw):
        if not isinstance(entry, Mapping):
            raise ValueError(f"assistant_payload.objects[{int(idx)}] must be a mapping")
        desc = entry.get("desc")
        if not isinstance(desc, str) or not desc.strip():
            raise ValueError(
                f"assistant_payload.objects[{int(idx)}].desc must be a non-empty string"
            )
        geom_keys = [
            k for k in ("bbox_2d", "poly") if k in entry and entry[k] is not None
        ]
        if len(geom_keys) != 1:
            raise ValueError(
                f"assistant_payload.objects[{int(idx)}] must contain exactly one geometry key (bbox_2d|poly)"
            )
        geom_key = geom_keys[0]
        raw_pts = flatten_points(entry.get(geom_key))
        if raw_pts is None or len(raw_pts) % 2 != 0:
            raise ValueError(
                f"assistant_payload.objects[{int(idx)}].{geom_key} must be a flat even-length sequence"
            )
        pts: List[int] = []
        ok = True
        for v in raw_pts:
            if isinstance(v, str) and v.startswith("<|coord_"):
                try:
                    pts.append(int(token_to_int(v)))
                except (TypeError, ValueError):
                    ok = False
                    break
            else:
                vi = _coerce_int(v)
                if vi is None:
                    ok = False
                    break
                pts.append(int(vi))
        if not ok:
            raise ValueError(
                f"assistant_payload.objects[{int(idx)}].{geom_key} contains invalid coordinate values"
            )
        if geom_key == "bbox_2d" and len(pts) != 4:
            raise ValueError(
                f"assistant_payload.objects[{int(idx)}].bbox_2d must contain exactly 4 coordinates"
            )
        if geom_key == "poly" and (len(pts) < 6 or len(pts) % 2 != 0):
            raise ValueError(
                f"assistant_payload.objects[{int(idx)}].poly must contain >=6 coordinates and even arity"
            )
        objs.append(
            GTObject(
                index=int(idx),
                geom_type=geom_key,
                points_norm1000=pts,
                desc=desc.strip(),
            )
        )
    return objs


def _coord_vocab_gate_loss(
    *, logits_full: torch.Tensor, logits_coord: torch.Tensor, temperature: float
) -> torch.Tensor:
    from src.trainers.losses.coord_soft_ce_w1 import coord_vocab_gate_loss

    gate, _mass_mean = coord_vocab_gate_loss(
        logits_full=logits_full,
        logits_coord=logits_coord,
        temperature=float(temperature),
    )
    return gate


def _build_labels_and_coord_targets_for_sample(
    *,
    input_ids_1d: torch.Tensor,  # [T]
    prompt_len: int,
    prefix_len: int,
    train_len: int,
    coord_id_set: set[int],
    coord_id_to_bin: Mapping[int, int],
    prefix_coord_pos: Sequence[int],
    prefix_coord_target_bins: Sequence[int],
    tail_ignore_pos: Optional[Sequence[int]] = None,
) -> Tuple[torch.Tensor, List[int], List[int], List[bool]]:
    """Create CE labels and coord supervision targets for a single sample.

    Invariants (unit-tested):
    - Prefix non-coord tokens are ignored for CE.
    - Coord tokens never contribute to CE.
    - Tail non-coord tokens contribute to CE (except for explicitly ignored positions like desc values).
    """
    seq_len = int(input_ids_1d.shape[0])
    labels = torch.full((seq_len,), -100, dtype=torch.long, device=input_ids_1d.device)

    coord_pos: List[int] = []
    coord_bins: List[int] = []
    coord_is_prefix: List[bool] = []

    # Assistant span sanity: supervised coord indices must never point into the prompt span.
    assistant_start = int(prompt_len)
    assistant_end = int(prompt_len) + int(train_len)
    if assistant_start < 0:
        raise ValueError(f"invalid prompt_len={prompt_len}")
    if assistant_end < assistant_start:
        raise ValueError(f"invalid train_len={train_len} for prompt_len={prompt_len}")
    assistant_end = min(assistant_end, seq_len)
    if assistant_end <= assistant_start:
        raise ValueError(
            f"invalid assistant span [{assistant_start},{assistant_end}) for seq_len={seq_len}"
        )

    # Tail: [prompt_len + prefix_len, prompt_len + train_len)
    tail_start = prompt_len + prefix_len
    tail_end = prompt_len + train_len
    tail_start = max(
        1, min(int(tail_start), seq_len)
    )  # p-1 must be valid for logits_next gather
    tail_end = max(tail_start, min(int(tail_end), seq_len))

    ignore_set = set(int(i) for i in (tail_ignore_pos or []) if int(i) >= 0)
    for p in range(tail_start, tail_end):
        if p < assistant_start or p >= assistant_end:
            raise ValueError(
                f"tail supervision index out of assistant span: p={p} span=[{assistant_start},{assistant_end})"
            )
        tok_id = int(input_ids_1d[p].item())
        if tok_id in coord_id_set:
            bin_idx = coord_id_to_bin.get(tok_id)
            if bin_idx is not None:
                coord_pos.append(int(p))
                coord_bins.append(int(bin_idx))
                coord_is_prefix.append(False)
            continue
        rel = int(p - tail_start)
        if rel in ignore_set:
            continue
        labels[p] = input_ids_1d[p]

    # Prefix self-context: supervised coord slots only (no CE).
    if len(prefix_coord_pos) != len(prefix_coord_target_bins):
        raise ValueError(
            "prefix_coord_pos and prefix_coord_target_bins must have identical length"
        )
    for local_idx, tbin in zip(prefix_coord_pos, prefix_coord_target_bins):
        li = int(local_idx)
        if li < 0 or li >= prefix_len:
            continue
        p = prompt_len + li
        if p <= 0 or p >= seq_len:
            continue
        if p < assistant_start or p >= assistant_end:
            raise ValueError(
                f"prefix supervision index out of assistant span: p={p} span=[{assistant_start},{assistant_end})"
            )
        coord_pos.append(int(p))
        coord_bins.append(int(tbin))
        coord_is_prefix.append(True)

    return labels, coord_pos, coord_bins, coord_is_prefix


def _build_labels_and_coord_targets_for_batch(
    *,
    input_ids: torch.Tensor,  # [B, T]
    meta: List[Mapping[str, Any]],
    coord_id_set: set[int],
    coord_id_to_bin: Mapping[int, int],
) -> Tuple[torch.Tensor, List[int], List[int], List[int], List[bool]]:
    """Build masked CE labels + coord supervision targets for a batch.

    Supports two meta contracts:
    - Un-packed: len(meta) == bsz and meta[b] describes one row.
    - Packed: bsz == 1 and meta is a list of per-segment dicts (order matches concatenation),
      each with an `encoded_len` key.
    """
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [B, T]")
    bsz, seq_len = input_ids.shape

    labels_masked = torch.full_like(input_ids, -100)

    supervised_batch: List[int] = []
    supervised_pos: List[int] = []
    supervised_bin: List[int] = []
    supervised_is_prefix: List[bool] = []

    if len(meta) == bsz:
        # Un-packed path (existing behavior)
        for b in range(bsz):
            m = meta[b]
            prompt_len = int(m["prompt_len"])
            prefix_len = int(m["prefix_len"])
            train_len = int(m["train_len"])
            prompt_ids = m.get("prompt_ids")

            # Sanity: prompt prefix matches (avoid silent misalignment).
            if prompt_len <= 0 or prompt_len >= seq_len:
                raise ValueError(
                    f"invalid prompt_len={prompt_len} for seq_len={seq_len}"
                )
            if isinstance(prompt_ids, list):
                teacher_prefix = input_ids[b, :prompt_len].detach().cpu().tolist()
                if teacher_prefix != prompt_ids:
                    raise ValueError(
                        "prompt tokenization mismatch between generation and teacher-forced encoding"
                    )

            prefix_pos_local = m.get("prefix_coord_pos") or []
            prefix_bins = m.get("prefix_coord_target_bins") or []
            tail_ignore_pos = m.get("tail_ignore_pos") or []
            labels_1d, cpos, cbins, cis_prefix = (
                _build_labels_and_coord_targets_for_sample(
                    input_ids_1d=input_ids[b],
                    prompt_len=prompt_len,
                    prefix_len=prefix_len,
                    train_len=train_len,
                    coord_id_set=coord_id_set,
                    coord_id_to_bin=coord_id_to_bin,
                    prefix_coord_pos=prefix_pos_local,
                    prefix_coord_target_bins=prefix_bins,
                    tail_ignore_pos=tail_ignore_pos,
                )
            )
            labels_masked[b] = labels_1d
            for p, tbin, is_pref in zip(cpos, cbins, cis_prefix):
                supervised_batch.append(int(b))
                supervised_pos.append(int(p))
                supervised_bin.append(int(tbin))
                supervised_is_prefix.append(bool(is_pref))

        return (
            labels_masked,
            supervised_batch,
            supervised_pos,
            supervised_bin,
            supervised_is_prefix,
        )

    # Packed path (one row containing multiple segments).
    if bsz != 1:
        raise ValueError(
            "packed-mode meta requires bsz==1; got len(meta)=%s bsz=%s"
            % (len(meta), bsz)
        )
    if not meta:
        raise ValueError("packed-mode meta must be a non-empty list")

    offset = 0
    for seg in meta:
        if not isinstance(seg, Mapping):
            raise ValueError("packed-mode meta must be a list of dict-like segments")
        encoded_len = int(seg.get("encoded_len") or 0)
        if encoded_len <= 0:
            raise ValueError("packed-mode segment missing/invalid encoded_len")
        if offset + encoded_len > seq_len:
            raise ValueError("packed-mode segments exceed packed seq_len")

        seg_input_ids = input_ids[0, offset : offset + encoded_len]
        seg_prompt_len = int(seg["prompt_len"])
        seg_prefix_len = int(seg["prefix_len"])
        seg_train_len = int(seg["train_len"])
        prompt_ids = seg.get("prompt_ids")

        if seg_prompt_len <= 0 or seg_prompt_len >= encoded_len:
            raise ValueError(
                f"invalid prompt_len={seg_prompt_len} for encoded_len={encoded_len}"
            )
        if isinstance(prompt_ids, list):
            teacher_prefix = seg_input_ids[:seg_prompt_len].detach().cpu().tolist()
            if teacher_prefix != prompt_ids:
                raise ValueError(
                    "prompt tokenization mismatch between generation and teacher-forced encoding"
                )

        prefix_pos_local = seg.get("prefix_coord_pos") or []
        prefix_bins = seg.get("prefix_coord_target_bins") or []
        tail_ignore_pos = seg.get("tail_ignore_pos") or []
        labels_1d, cpos, cbins, cis_prefix = _build_labels_and_coord_targets_for_sample(
            input_ids_1d=seg_input_ids,
            prompt_len=seg_prompt_len,
            prefix_len=seg_prefix_len,
            train_len=seg_train_len,
            coord_id_set=coord_id_set,
            coord_id_to_bin=coord_id_to_bin,
            prefix_coord_pos=prefix_pos_local,
            prefix_coord_target_bins=prefix_bins,
            tail_ignore_pos=tail_ignore_pos,
        )
        labels_masked[0, offset : offset + encoded_len] = labels_1d
        for p, tbin, is_pref in zip(cpos, cbins, cis_prefix):
            supervised_batch.append(0)
            supervised_pos.append(int(offset + p))
            supervised_bin.append(int(tbin))
            supervised_is_prefix.append(bool(is_pref))

        offset += encoded_len

    return (
        labels_masked,
        supervised_batch,
        supervised_pos,
        supervised_bin,
        supervised_is_prefix,
    )


class _FixedRawMicroBatchStacker:
    """Stack identity-collated raw micro-batches into fixed-size lists.

    Stage_2 trainers often keep `training.per_device_train_batch_size=1` so the learner
    does exactly one packed forward/backward per micro-step (global_max_length capped).

    However, post-rollout packing needs *multiple* raw samples to form a meaningfully
    filled pack. This wrapper allows us to decouple those two concerns in a way that is
    compatible with epoch-based training: `__len__` is adjusted to reflect the reduced
    number of emitted micro-batches.

    Expected input from the underlying dataloader: a list of raw samples.
    Output: a list of raw samples of size `target_raw_batch_size` (last one may be smaller).
    """

    def __init__(
        self,
        dataloader,
        *,
        target_raw_batch_size: int,
        base_raw_batch_size: int,
    ):
        self.dataloader = dataloader
        self.target_raw_batch_size = max(1, int(target_raw_batch_size))
        self.base_raw_batch_size = max(1, int(base_raw_batch_size))

    def __iter__(self):
        buf: List[Any] = []
        for b in self.dataloader:
            if not isinstance(b, list):
                raise ValueError(
                    "fixed raw microbatch stacker expects identity-collated train batches (list of raw samples)"
                )
            buf.extend(b)
            while len(buf) >= self.target_raw_batch_size:
                out = buf[: self.target_raw_batch_size]
                del buf[: self.target_raw_batch_size]
                yield out

        if buf:
            yield buf

    def __len__(self) -> int:
        # Underlying dataloader length is in units of micro-batches. Convert to a raw
        # sample count using the base batch size, then divide by the target size.
        n_micro = int(len(self.dataloader))
        n_raw = int(n_micro) * int(self.base_raw_batch_size)
        return int(
            (n_raw + self.target_raw_batch_size - 1) // self.target_raw_batch_size
        )

    def __getattr__(self, name: str):
        return getattr(self.dataloader, name)


class _AdaptiveRawMicroBatchStacker:
    """Stack identity-collated raw micro-batches into adaptive-size lists.

    This helper is intentionally lightweight: it chooses a target raw microbatch size
    based on the trainer's observed post-rollout packing fill, so repeated underfilled
    packs can automatically increase the raw sample budget.

    The class is currently used for unit tests and diagnostics; production runs
    typically rely on a configured `decode_batch_size`.
    """

    def __init__(self, dataloader, *, trainer: Any):
        self.dataloader = dataloader
        self.trainer = trainer

    def _target_microbatch_size(self) -> int:
        packing_length = int(self.trainer._packing_length())
        min_fill = float(self.trainer._packing_min_fill_ratio())
        buf_cap = int(self.trainer._packing_buffer_cap())

        # Base estimate from the rolling average segment length.
        avg_len = float(getattr(self.trainer, "_rm_avg_segment_len", 0.0) or 0.0)
        if avg_len > 0.0:
            base = int(math.ceil((packing_length * min_fill) / avg_len))
        else:
            base = 1
        base = max(1, base)

        # Bump estimate from last-pack underfill (only when there is no carry buffer).
        bump = 0
        last_fill = float(getattr(self.trainer, "_rm_last_pack_fill", 0.0) or 0.0)
        last_segments = int(getattr(self.trainer, "_rm_last_pack_segments", 0) or 0)
        last_buf_after = int(
            getattr(self.trainer, "_rm_last_pack_buffer_after", 0) or 0
        )
        if (
            last_buf_after == 0
            and last_segments > 0
            and min_fill > 0.0
            and 0.0 < last_fill < min_fill
        ):
            bump = int(math.ceil((last_segments * min_fill) / last_fill))

        target = max(base, bump)
        if buf_cap > 0:
            target = min(target, buf_cap)
        return max(1, target)

    def __iter__(self):
        buf: List[Any] = []
        target = int(self._target_microbatch_size())
        for b in self.dataloader:
            if not isinstance(b, list):
                raise ValueError(
                    "adaptive raw microbatch stacker expects identity-collated train batches (list of raw samples)"
                )
            buf.extend(b)
            while len(buf) >= target:
                out = buf[:target]
                del buf[:target]
                yield out

        if buf:
            yield buf

    def __len__(self) -> int:
        # Best-effort length; if the underlying dataloader does not implement __len__,
        # fall back to 0 (PyTorch IterableDataset semantics).
        try:
            n_micro = int(len(self.dataloader))
        except TypeError:
            return 0

        # Identity collator yields raw samples; base microbatch size defaults to 1.
        base_raw_batch_size = 1
        try:
            base_raw_batch_size = int(
                getattr(
                    getattr(self.trainer, "args", None),
                    "per_device_train_batch_size",
                    1,
                )
                or 1
            )
        except (TypeError, ValueError):
            base_raw_batch_size = 1

        n_raw = int(n_micro) * int(base_raw_batch_size)
        target = int(self._target_microbatch_size())
        return int((n_raw + target - 1) // target)

    def __getattr__(self, name: str):
        return getattr(self.dataloader, name)


class RolloutMatchingSFTTrainer(Seq2SeqTrainer):
    """Rollout-matching (stage_2) trainer variant."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._coord_token_ids: Optional[List[int]] = None
        self._coord_id_to_bin: Optional[Dict[int, int]] = None
        self._debug_dump_count: int = 0
        # Rank-local carry buffer for dynamic post-rollout packing (stage_2 only).
        # Each entry is (encoded, meta, encoded_len).
        self._post_rollout_segments: List[
            Tuple[Dict[str, Any], Dict[str, Any], int]
        ] = []

        # vLLM rollout backend state (lazy init).
        self._vllm_engine: Any = None
        self._vllm_tp_group: Any = None
        self._vllm_tp_size: int = 1
        self._vllm_last_loaded_step: int = -1

        # vLLM server-mode rollout backend state (lazy init).
        self._vllm_server_client: Any = None
        self._vllm_server_client_lock = threading.Lock()
        self._vllm_server_comm_inited: bool = False
        self._vllm_server_last_synced_step: int = -1
        self._vllm_server_debug_dump_count: int = 0
        self._vllm_server_debug_last_step: Optional[int] = None
        self._vllm_server_last_logged_step: int = -1
        self._vllm_server_force_full_sync: bool = False

        # Buffered training logs: accumulate across micro-batches and merge into the step log.
        # Keyed by the *post-optimizer* global_step (HF logs after increment).
        self._rm_pending_train_logs: Dict[int, _PendingTrainRolloutLog] = {}

        # Periodic qualitative dumps (rank0 only): rollout vs GT vs training target.
        self._monitor_dump_last_step: Optional[int] = None
        self._monitor_dump_count: int = 0

        # Optional semantic desc monitoring (lazy init; metrics only).
        self._desc_semantic_encoder: Any = None
        self._desc_semantic_encoder_sig: Any = None

        # Mutable config injected by src/sft.py after construction.
        self.rollout_matching_cfg: Mapping[str, Any] = {}

    def _merge_rollout_matching_batch_metrics(
        self, batch: MutableMapping[str, Any], metrics: Mapping[str, Any]
    ) -> None:
        """Merge rollout-matching batch metrics onto an existing batch.

        Treat `_rollout_matching_batch_metrics` as merge-only so that later pipeline
        stages (packing, async prefetch, post-processing) can add telemetry without
        losing base rollout/decode metrics.
        """
        if not isinstance(batch, MutableMapping):
            raise TypeError("batch must be a MutableMapping")
        if not isinstance(metrics, Mapping):
            raise TypeError("metrics must be a Mapping")

        existing = batch.get("_rollout_matching_batch_metrics")
        out: Dict[str, Any] = dict(existing) if isinstance(existing, Mapping) else {}
        for k, v in metrics.items():
            out[str(k)] = v
        batch["_rollout_matching_batch_metrics"] = out

    # ------------------------ config helpers ------------------------ #
    def _cfg(self, key: str, default: Any) -> Any:
        cfg = getattr(self, "rollout_matching_cfg", None)
        if not isinstance(cfg, Mapping):
            return default
        return cfg.get(str(key), default)

    def _object_field_order(self) -> Literal["desc_first", "geometry_first"]:
        raw = getattr(self, "object_field_order", None)
        if raw is None:
            raw = self._cfg("object_field_order", "desc_first")
        return normalize_object_field_order(raw, path="custom.object_field_order")

    def _validate_rollout_matching_cfg(self) -> None:
        cfg = getattr(self, "rollout_matching_cfg", None)
        if cfg is None:
            return
        if not isinstance(cfg, Mapping):
            raise TypeError(
                "rollout_matching_cfg must be a mapping (injected from rollout_matching)"
            )

        removed = [
            k
            for k in (
                # Older top-level decoding knobs.
                "temperature",
                "top_p",
                "top_k",
                # Removed buffer reuse.
                "rollout_buffer",
                # Legacy batching knobs (replaced by decode_batch_size).
                "rollout_generate_batch_size",
                "rollout_infer_batch_size",
                # Removed packing-scope knob.
                "post_rollout_pack_scope",
            )
            if k in cfg
        ]
        if removed:
            rendered: List[str] = []
            for k in removed:
                if k in {"rollout_generate_batch_size", "rollout_infer_batch_size"}:
                    rendered.append(
                        f"rollout_matching.{k} (use rollout_matching.decode_batch_size)"
                    )
                elif k == "post_rollout_pack_scope":
                    rendered.append(
                        f"rollout_matching.{k} (remove; micro-scope packing is standard)"
                    )
                else:
                    rendered.append(f"rollout_matching.{k}")

            legacy_s = ", ".join(rendered)
            raise ValueError(
                "Legacy rollout-matching keys have been removed: "
                f"{legacy_s}. (No backward compatibility.)"
            )

        # Validate unified decode batch size knob.
        decode_bs_raw = cfg.get("decode_batch_size", 1)
        try:
            decode_bs = int(decode_bs_raw)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "rollout_matching.decode_batch_size must be an int"
            ) from exc
        if decode_bs <= 0:
            raise ValueError(
                "rollout_matching.decode_batch_size must be > 0"
            )

        dec = cfg.get("decoding", None)
        if dec is None:
            dec = {}
        if not isinstance(dec, Mapping):
            raise TypeError(
                "rollout_matching.decoding must be a mapping when provided"
            )

        # Validate decoding ranges (robust defaults).
        try:
            temperature = float(dec.get("temperature", 0.0) or 0.0)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "rollout_matching.decoding.temperature must be a float"
            ) from exc
        if temperature < 0.0:
            raise ValueError(
                "rollout_matching.decoding.temperature must be >= 0"
            )

        try:
            top_p = float(
                dec.get("top_p", 1.0) if dec.get("top_p", None) is not None else 1.0
            )
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "rollout_matching.decoding.top_p must be a float"
            ) from exc
        if not (0.0 < top_p <= 1.0):
            raise ValueError(
                "rollout_matching.decoding.top_p must be in (0, 1]"
            )

        top_k_raw = dec.get("top_k", -1)
        try:
            top_k = int(top_k_raw)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "rollout_matching.decoding.top_k must be an int"
            ) from exc
        if top_k != -1 and top_k < 1:
            raise ValueError(
                "rollout_matching.decoding.top_k must be -1 (disabled) or >= 1"
            )

    def _decoding_cfg(self) -> Mapping[str, Any]:
        # `rollout_matching_cfg` is injected in src/sft.py. Use a nested dict for decoding.
        raw = self._cfg("decoding", {})
        if raw is None:
            return {}
        if not isinstance(raw, Mapping):
            raise TypeError(
                "rollout_matching.decoding must be a mapping when provided"
            )
        return raw

    def _decoding_params(self) -> Tuple[float, float, int]:
        dec = self._decoding_cfg()

        temperature_raw = dec.get("temperature", 0.0)
        try:
            temperature = float(temperature_raw or 0.0)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "rollout_matching.decoding.temperature must be a float"
            ) from exc
        if temperature < 0.0:
            raise ValueError(
                "rollout_matching.decoding.temperature must be >= 0"
            )

        top_p_raw = dec.get("top_p", 1.0)
        try:
            top_p = float(top_p_raw if top_p_raw is not None else 1.0)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "rollout_matching.decoding.top_p must be a float"
            ) from exc
        if not (0.0 < top_p <= 1.0):
            raise ValueError(
                "rollout_matching.decoding.top_p must be in (0, 1]"
            )

        top_k_raw = dec.get("top_k", -1)
        try:
            top_k = int(top_k_raw)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "rollout_matching.decoding.top_k must be an int"
            ) from exc
        if top_k != -1 and top_k < 1:
            raise ValueError(
                "rollout_matching.decoding.top_k must be -1 (disabled) or >= 1"
            )

        return float(temperature), float(top_p), int(top_k)

    @staticmethod
    def _apply_rollout_decoding_to_generation_config(
        *,
        gen_cfg: Any,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
    ) -> None:
        do_sample = bool(float(temperature) > 0.0)
        gen_cfg.do_sample = do_sample
        gen_cfg.temperature = max(1e-4, float(temperature)) if do_sample else 1.0
        gen_cfg.top_p = float(top_p) if do_sample else 1.0
        gen_cfg.top_k = int(top_k) if (do_sample and int(top_k) != -1) else 0
        gen_cfg.repetition_penalty = float(repetition_penalty)

    @staticmethod
    def _rollout_vllm_request_config_kwargs(
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
    ) -> Dict[str, Any]:
        return {
            "n": 1,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "top_k": int(top_k),
            "repetition_penalty": float(repetition_penalty),
            "stop": [_IM_END],
            "return_details": True,
        }

    def _monitor_dump_cfg(self) -> Mapping[str, Any]:
        cfg = self._cfg("monitor_dump", {}) or {}
        return cfg if isinstance(cfg, Mapping) else {}

    def _desc_monitor_cfg(self) -> Mapping[str, Any]:
        cfg = self._cfg("desc_monitor", {}) or {}
        return cfg if isinstance(cfg, Mapping) else {}

    def _get_desc_semantic_encoder(self, cfg: Mapping[str, Any]) -> Any:
        """Return a cached semantic encoder instance, or None if disabled/unavailable."""

        mode = str(cfg.get("mode", "semantic") or "semantic").strip().lower()
        if mode not in {"semantic", "both"}:
            return None

        try:
            from src.metrics.semantic_desc import SemanticDescEncoder
        except ImportError:
            return None

        model_name = str(
            cfg.get("semantic_model", "sentence-transformers/all-MiniLM-L6-v2")
        )
        device = str(cfg.get("semantic_device", "cpu"))
        batch_size = int(cfg.get("semantic_batch_size", 64) or 64)
        max_length = int(cfg.get("semantic_max_length", 64) or 64)

        sig = (model_name, device, batch_size, max_length)
        enc = getattr(self, "_desc_semantic_encoder", None)
        enc_sig = getattr(self, "_desc_semantic_encoder_sig", None)
        if enc is not None and enc_sig == sig:
            return enc

        enc = SemanticDescEncoder(
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
        )
        setattr(self, "_desc_semantic_encoder", enc)
        setattr(self, "_desc_semantic_encoder_sig", sig)
        return enc

    def _is_main_process(self) -> bool:
        acc = getattr(self, "accelerator", None)
        if acc is not None and hasattr(acc, "is_main_process"):
            try:
                return bool(acc.is_main_process)
            except (TypeError, ValueError):
                raise
        return bool(getattr(self, "is_world_process_zero", False))

    def _should_monitor_dump(self, *, global_step: int) -> bool:
        cfg = self._monitor_dump_cfg()
        if not bool(cfg.get("enabled", False)):
            return False
        if (
            bool(cfg.get("only_world_process_zero", True))
            and not self._is_main_process()
        ):
            return False

        max_events = int(cfg.get("max_events", 20) or 0)
        if max_events > 0 and self._monitor_dump_count >= max_events:
            return False

        gs = int(global_step)
        if (
            self._monitor_dump_last_step is not None
            and int(self._monitor_dump_last_step) == gs
        ):
            return False

        every = cfg.get("every_steps", None)
        if every is None:
            every = int(getattr(self.args, "logging_steps", 1) or 1)
        every = max(1, int(every))

        dump_first = bool(
            cfg.get(
                "dump_first_step", bool(getattr(self.args, "logging_first_step", False))
            )
        )
        if gs == 0 and not dump_first:
            return False
        if gs % every != 0:
            return False
        return True

    @staticmethod
    def _clip_text(text: Any, *, max_chars: int) -> str:
        s = ""
        try:
            s = str(text)
        except (TypeError, ValueError):
            s = ""
        if max_chars <= 0:
            return s
        if len(s) <= max_chars:
            return s
        return s[:max_chars] + "...<truncated>"

    @staticmethod
    def _ascii_safe_text(text: str) -> str:
        # Keep dumps ASCII by escaping non-ASCII characters, but preserve newlines.
        out: List[str] = []
        for ch in text:
            if ord(ch) < 128:
                out.append(ch)
            else:
                out.append("\\u%04x" % ord(ch))
        return "".join(out)

    def _write_monitor_dump(
        self, *, global_step: int, payload: Mapping[str, Any]
    ) -> None:
        cfg = self._monitor_dump_cfg()
        out_dir = cfg.get("out_dir")
        if not isinstance(out_dir, str) or not out_dir.strip():
            out_dir = os.path.join(
                str(getattr(self.args, "output_dir", ".")), "monitor_dumps"
            )
        os.makedirs(out_dir, exist_ok=True)

        # One file per optimizer step by default (easy to inspect while training).
        step_path = os.path.join(out_dir, f"step_{int(global_step):06d}.json")
        with open(step_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)

        if bool(cfg.get("write_markdown", True)):
            md_path = os.path.join(out_dir, f"step_{int(global_step):06d}.md")
            try:
                md = self._format_monitor_dump_markdown(payload)
                with open(md_path, "w", encoding="utf-8") as f:
                    f.write(md)
            except (TypeError, ValueError):
                raise

    def _format_monitor_dump_markdown(self, payload: Mapping[str, Any]) -> str:
        # Human-readable dump; keep it ASCII-safe to avoid surprising tooling issues.
        max_chars = int(self._monitor_dump_cfg().get("max_text_chars", 4000) or 4000)

        def _j(obj: Any) -> str:
            try:
                return json.dumps(obj, ensure_ascii=True, indent=2)
            except (TypeError, ValueError):
                return "{}"

        lines: List[str] = []
        gs = payload.get("global_step")
        lines.append(f"# Rollout-Matching Monitor Dump (global_step={gs})\n")
        meta = payload.get("meta") or {}
        lines.append("## Meta\n")
        lines.append("```json\n" + _j(meta) + "\n```\n")

        samples = (
            payload.get("samples") if isinstance(payload.get("samples"), list) else []
        )
        for i, s in enumerate(samples):
            if not isinstance(s, Mapping):
                continue
            lines.append(f"## Sample {i}\n")
            sid = s.get("sample_id")
            bidx = s.get("base_idx")
            img = s.get("image") or s.get("images")
            lines.append(f"- sample_id: `{sid}`\n")
            lines.append(f"- base_idx: `{bidx}`\n")
            lines.append(f"- image(s): `{img}`\n\n")

            lines.append("### Messages\n")
            lines.append("```json\n" + _j(s.get("messages")) + "\n```\n")

            lines.append("### Rollout (raw)\n")
            lines.append(
                "```text\n"
                + self._ascii_safe_text(
                    self._clip_text(s.get("rollout_text"), max_chars=max_chars)
                )
                + "\n```\n"
            )
            lines.append("### Prefix Used (append-ready)\n")
            lines.append(
                "```text\n"
                + self._ascii_safe_text(
                    self._clip_text(s.get("prefix_text"), max_chars=max_chars)
                )
                + "\n```\n"
            )
            lines.append("### Training Target (prefix + FN append)\n")
            lines.append(
                "```text\n"
                + self._ascii_safe_text(
                    self._clip_text(s.get("train_text"), max_chars=max_chars)
                )
                + "\n```\n"
            )

            lines.append("### GT Objects\n")
            lines.append("```json\n" + _j(s.get("gt_objects")) + "\n```\n")
            lines.append("### Pred Objects (valid)\n")
            lines.append("```json\n" + _j(s.get("pred_objects")) + "\n```\n")
            lines.append("### Match\n")
            lines.append("```json\n" + _j(s.get("match")) + "\n```\n")
            lines.append("### Stats\n")
            lines.append("```json\n" + _j(s.get("stats")) + "\n```\n")

        return "".join(lines)

    def _offload_settings(self) -> Tuple[bool, bool, bool]:
        cfg_raw = self._cfg("offload", {}) or {}
        if cfg_raw is None:
            cfg_raw = {}
        if not isinstance(cfg_raw, Mapping):
            raise ValueError("rollout_matching.offload must be a mapping")

        enabled = bool(cfg_raw.get("enabled", False))
        offload_model = bool(cfg_raw.get("offload_model", False))
        offload_optimizer = bool(cfg_raw.get("offload_optimizer", False))
        return enabled, offload_model, offload_optimizer

    @contextmanager
    def _maybe_rollout_offload_context(self):
        """Offload training state during colocate vLLM rollout generation.

        Fail-fast when offload is requested but not safe to apply.
        """

        enabled, offload_model, offload_optimizer = self._offload_settings()
        if not enabled or (not offload_model and not offload_optimizer):
            yield
            return

        # Only applicable for colocate vLLM rollouts.
        if self._rollout_backend() != "vllm" or self._vllm_mode() != "colocate":
            yield
            return

        # Fail-fast on known-incompatible setups.
        if bool(getattr(self, "is_deepspeed_enabled", False)):
            raise RuntimeError(
                "rollout offload is not supported with DeepSpeed/ZeRO in this trainer. "
                "Mitigations: disable rollout_matching.offload, switch rollout_backend=hf, "
                "or disable DeepSpeed."
            )

        train_device = getattr(getattr(self, "accelerator", None), "device", None)
        if train_device is None:
            train_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = getattr(
            getattr(self, "accelerator", None), "unwrap_model", lambda x: x
        )(self.model)
        opt = getattr(self, "optimizer", None)

        @torch.no_grad()
        def _offload_model_to_cpu(m) -> None:
            for p in m.parameters():
                p.data = p.data.to(torch.device("cpu"), non_blocking=True)

        @torch.no_grad()
        def _load_model_to_device(m) -> None:
            for p in m.parameters():
                p.data = p.data.to(train_device, non_blocking=True)

        @torch.no_grad()
        def _offload_opt_to_cpu(o) -> None:
            if o is None or not getattr(o, "state", None):
                return
            for pg in o.param_groups:
                for p in pg.get("params", []):
                    st = o.state.get(p)
                    if not isinstance(st, dict):
                        continue
                    for k, v in list(st.items()):
                        if isinstance(v, torch.Tensor):
                            st[k] = v.to(torch.device("cpu"), non_blocking=True)

        @torch.no_grad()
        def _load_opt_to_device(o) -> None:
            if o is None or not getattr(o, "state", None):
                return
            for pg in o.param_groups:
                for p in pg.get("params", []):
                    st = o.state.get(p)
                    if not isinstance(st, dict):
                        continue
                    for k, v in list(st.items()):
                        if isinstance(v, torch.Tensor):
                            st[k] = v.to(train_device, non_blocking=True)

        try:
            if offload_model:
                _offload_model_to_cpu(model)
            if offload_optimizer:
                _offload_opt_to_cpu(opt)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            yield
        finally:
            if offload_model:
                _load_model_to_device(model)
            if offload_optimizer:
                _load_opt_to_device(opt)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _rollout_backend(self) -> Literal["hf", "vllm"]:
        backend = str(self._cfg("rollout_backend", "vllm")).strip().lower()
        if backend not in {"hf", "vllm"}:
            raise ValueError(
                f"rollout_matching.rollout_backend must be one of {{hf,vllm}}, got {backend!r}"
            )
        return backend  # type: ignore[return-value]

    def _vllm_mode(self) -> Literal["colocate", "server"]:
        """vLLM integration mode.

        - `colocate` (default): instantiate a local vLLM engine.
        - `server`: connect to a pre-launched ms-swift rollout server.
        """
        vcfg_raw = self._cfg("vllm", {}) or {}
        if not isinstance(vcfg_raw, Mapping):
            raise ValueError("rollout_matching.vllm must be a mapping")
        mode = str(vcfg_raw.get("mode", "colocate") or "colocate").strip().lower()
        if mode not in {"colocate", "server"}:
            raise ValueError(
                "rollout_matching.vllm.mode must be 'colocate' or 'server'; "
                f"got {mode!r}"
            )
        return mode  # type: ignore[return-value]

    def _vllm_server_cfg(self) -> Mapping[str, Any]:
        vcfg_raw = self._cfg("vllm", {}) or {}
        if not isinstance(vcfg_raw, Mapping):
            raise ValueError("rollout_matching.vllm must be a mapping")
        scfg_raw = vcfg_raw.get("server", {}) or {}
        if not isinstance(scfg_raw, Mapping):
            raise ValueError(
                "rollout_matching.vllm.server must be a mapping"
            )
        return scfg_raw

    def _vllm_server_specs(self) -> List[Dict[str, Any]]:
        """Normalize server list config to a list of {base_url, group_port} dicts.

        Spec contract (2026-02-15 strict schema):
        - Only `rollout_matching.vllm.server.servers: [...]` is supported.
        - Legacy paired-list form (`server.base_url` + `server.group_port`) is removed.
        """

        scfg = self._vllm_server_cfg()

        if "base_url" in scfg or "group_port" in scfg:
            # Loader-level schema already fails fast for this shape, but keep a defensive
            # runtime error for direct-instantiation / test helpers.
            raise ValueError(
                "Legacy rollout server config has been removed: "
                "rollout_matching.vllm.server.base_url/group_port. "
                "Use rollout_matching.vllm.server.servers[] (list of {base_url, group_port})."
            )

        servers_raw = scfg.get("servers", None)
        if not isinstance(servers_raw, list) or not servers_raw:
            raise ValueError(
                "rollout_matching.vllm.server.servers must be a non-empty list"
            )

        out: List[Dict[str, Any]] = []
        for i, s in enumerate(servers_raw):
            if not isinstance(s, Mapping):
                raise ValueError(
                    "rollout_matching.vllm.server.servers[%d] must be a mapping"
                    % int(i)
                )

            base_url = s.get("base_url")
            if not isinstance(base_url, str) or not base_url.strip():
                raise ValueError(
                    "rollout_matching.vllm.server.servers[%d].base_url must be a non-empty string"
                    % int(i)
                )

            group_port_entry_raw = s.get("group_port")
            try:
                group_port_entry = int(group_port_entry_raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "rollout_matching.vllm.server.servers[%d].group_port must be an int"
                    % int(i)
                ) from exc
            if group_port_entry <= 0:
                raise ValueError(
                    "rollout_matching.vllm.server.servers[%d].group_port must be > 0"
                    % int(i)
                )

            out.append(
                {
                    "base_url": base_url.strip().rstrip("/"),
                    "group_port": int(group_port_entry),
                }
            )

        return out

    def _vllm_server_timeouts(self) -> Tuple[float, Optional[float]]:
        scfg = self._vllm_server_cfg()

        timeout_raw = scfg.get("timeout_s", None)
        if timeout_raw is None:
            timeout_s = 240.0
        else:
            try:
                timeout_s = float(timeout_raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "rollout_matching.vllm.server.timeout_s must be a float/int"
                ) from exc
        if timeout_s <= 0:
            raise ValueError(
                "rollout_matching.vllm.server.timeout_s must be > 0"
            )

        # Infer (read) timeout for /infer/ requests:
        # - null/unset: no timeout (allows long rollouts without client-side aborts)
        # - <= 0: also treated as disabled
        # - > 0: enforced as (connect, read) timeout tuple downstream
        infer_timeout_raw = scfg.get("infer_timeout_s", None)
        if infer_timeout_raw is None:
            infer_timeout_s: Optional[float] = None
        else:
            try:
                infer_timeout_s = float(infer_timeout_raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "rollout_matching.vllm.server.infer_timeout_s must be null or a float/int"
                ) from exc
            if infer_timeout_s <= 0:
                infer_timeout_s = None

        return float(timeout_s), (
            float(infer_timeout_s) if infer_timeout_s is not None else None
        )

    def _vllm_server_world_sizes(self) -> List[int]:
        """Return cached vLLM server data-parallel world sizes (one per server).

        The rollout server exposes `/get_world_size/` which returns JSON with a
        `world_size` field.
        """
        cached = getattr(self, "_vllm_server_cached_world_sizes", None)
        if (
            isinstance(cached, list)
            and cached
            and all(isinstance(x, int) and x > 0 for x in cached)
        ):
            return list(int(x) for x in cached)

        servers = self._vllm_server_specs()
        timeout_s, _infer_timeout_s = self._vllm_server_timeouts()

        import json as _json
        import urllib.request as _urllib

        opener = _urllib.build_opener(_urllib.ProxyHandler({}))
        out: List[int] = []
        for s in servers:
            base_url = str(s["base_url"]).rstrip("/")
            url = f"{base_url}/get_world_size/"
            req = _urllib.Request(url, method="GET")
            with opener.open(req, timeout=float(timeout_s)) as resp:
                code = int(resp.getcode())
                body = resp.read()
            if code != 200:
                raise RuntimeError(
                    f"vLLM rollout server /get_world_size/ returned HTTP {code}: {url}"
                )
            try:
                data = _json.loads(body.decode("utf-8"))
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"vLLM rollout server /get_world_size/ returned non-JSON payload: {url}"
                ) from exc
            try:
                ws = int(data.get("world_size", 1)) if isinstance(data, dict) else 1
            except (TypeError, ValueError):
                ws = 1
            out.append(max(1, int(ws)))

        setattr(self, "_vllm_server_cached_world_sizes", list(out))
        logger.info("vLLM rollout server world_size(s): %s", out)
        return list(out)


    def _rollout_decode_batch_size_per_rank(self) -> int:
        """Derived rollout request chunk size per learner rank.

        `decode_batch_size` is defined as a per-rollout-GPU cap per generation call.

        - HF backend and vLLM colocate mode: each learner rank decodes locally on its own
          device(s), so the per-call batch size is exactly `decode_batch_size`.
        - vLLM server mode: learner ranks concurrently issue rollout RPCs to a pool of
          rollout GPUs (data-parallel replicas). To preserve a per-rollout-GPU cap, we
          derive a per-rank request chunk size based on the rollout server world size
          and learner world size.
        """
        cap = int(self._decode_batch_size())
        backend = str(self._rollout_backend()).strip().lower()
        if backend != "vllm":
            return max(1, int(cap))

        mode = str(self._vllm_mode()).strip().lower()
        if mode != "server":
            return max(1, int(cap))

        # vLLM server mode: derive the maximum number of requests each learner rank may
        # issue per call so that (under DDP) total concurrent requests across ranks is
        # bounded by `cap * rollout_world_size`.
        server_world_sizes = self._vllm_server_world_sizes()
        rollout_world = int(sum(int(x) for x in server_world_sizes))
        if rollout_world <= 0:
            rollout_world = 1

        learner_world = 1
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                learner_world = int(dist.get_world_size())
        except (TypeError, ValueError):
            learner_world = 1

        if learner_world <= 0:
            learner_world = 1

        if int(cap) * int(rollout_world) < int(learner_world):
            raise ValueError(
                "decode_batch_size cap is infeasible for the current topology: "
                f"decode_batch_size={cap} rollout_world_size={rollout_world} learner_world_size={learner_world}. "
                "Increase rollout server DP world size, reduce learner world size, or increase decode_batch_size."
            )

        per_rank = max(1, int(int(cap) * int(rollout_world) // int(learner_world)))

        meta = (
            int(cap),
            int(learner_world),
            tuple(int(x) for x in server_world_sizes),
            int(per_rank),
        )
        if meta != getattr(self, "_last_logged_rollout_decode_chunk_meta", None):
            logger.info(
                "Rollout decode batching (vLLM server): decode_batch_size_cap=%s learner_world_size=%s "
                "rollout_server_world_sizes=%s rollout_world_size=%s per_rank_chunk=%s total_chunk_across_ranks=%s",
                int(cap),
                int(learner_world),
                list(int(x) for x in server_world_sizes),
                int(rollout_world),
                int(per_rank),
                int(per_rank) * int(learner_world),
            )
            setattr(self, "_last_logged_rollout_decode_chunk_meta", meta)

        return int(per_rank)

    def _vllm_server_sync_cfg(self) -> Tuple[str, bool]:
        vcfg_raw = self._cfg("vllm", {}) or {}
        if not isinstance(vcfg_raw, Mapping):
            raise ValueError("rollout_matching.vllm must be a mapping")
        sync_raw = vcfg_raw.get("sync", {}) or {}
        if not isinstance(sync_raw, Mapping):
            raise ValueError(
                "rollout_matching.vllm.sync must be a mapping"
            )

        mode = str(sync_raw.get("mode", "full") or "full").strip().lower()
        if mode not in {"full", "adapter", "auto"}:
            raise ValueError(
                "rollout_matching.vllm.sync.mode must be one of: full|adapter|auto"
            )
        fallback_to_full = bool(sync_raw.get("fallback_to_full", True))
        return mode, fallback_to_full

    def _derive_rollout_seed_base(self, *, global_step: int) -> int:
        """Deterministic seed base for rollouts.

        Contract: per-request seeds are derived deterministically from:
        - `training.seed` (HF TrainingArguments.seed)
        - `global_step` (optimizer-step index)
        - within-batch sample index
        """
        base = int(getattr(getattr(self, "args", None), "seed", 0) or 0)
        gs = int(global_step)
        # Keep in signed int32 range for compatibility with various backends.
        return int((base + gs * 1000003) & 0x7FFFFFFF)

    def _decode_batch_size(self) -> int:
        raw = self._cfg("decode_batch_size", 1)
        try:
            v = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "rollout_matching.decode_batch_size must be an int"
            ) from exc
        if v <= 0:
            raise ValueError(
                "rollout_matching.decode_batch_size must be > 0"
            )
        return int(v)

    def _packing_enabled(self) -> bool:
        return bool(self._cfg("packing_enabled", False))

    def _packing_length(self) -> int:
        try:
            return int(self._cfg("packing_length", 0) or 0)
        except (TypeError, ValueError) as exc:
            raise ValueError("packing_length must be an int") from exc

    def _assert_single_packed_forward(
        self, batch: Mapping[str, Any], *, where: str
    ) -> None:
        input_ids = batch.get("input_ids") if isinstance(batch, Mapping) else None
        if not isinstance(input_ids, torch.Tensor):
            return
        if input_ids.ndim != 2:
            raise ValueError(
                f"{where}: expected input_ids with shape [B, T], got {tuple(input_ids.shape)}"
            )
        bsz, seq_len = input_ids.shape
        if int(bsz) != 1:
            raise ValueError(
                f"{where}: packing must produce exactly one packed sequence per forward pass (batch_size=1), got batch_size={int(bsz)}"
            )
        max_len = 0
        try:
            max_len = int(self._packing_length() or 0)
        except (TypeError, ValueError):
            max_len = 0
        if int(max_len) > 0 and int(seq_len) > int(max_len):
            raise ValueError(
                f"{where}: packed seq_len={int(seq_len)} exceeds packing_length/global_max_length={int(max_len)}"
            )

    def _packing_buffer_cap(self) -> int:
        try:
            return int(self._cfg("packing_buffer", 0) or 0)
        except (TypeError, ValueError) as exc:
            raise ValueError("packing_buffer must be an int") from exc

    def _packing_min_fill_ratio(self) -> float:
        try:
            v = float(self._cfg("packing_min_fill_ratio", 0.65))
        except (TypeError, ValueError) as exc:
            raise ValueError("packing_min_fill_ratio must be a float") from exc
        if not (0 < v <= 1):
            raise ValueError("packing_min_fill_ratio must be in (0, 1]")
        return float(v)

    def _packing_drop_last(self) -> bool:
        return bool(self._cfg("packing_drop_last", True))


    @staticmethod
    def _extract_encoded_len(encoded: Mapping[str, Any]) -> int:
        length = encoded.get("length")
        if isinstance(length, int) and length > 0:
            return int(length)
        input_ids = encoded.get("input_ids")
        if input_ids is not None and hasattr(input_ids, "__len__"):
            try:
                n = int(len(input_ids))
                if n > 0:
                    return n
            except (TypeError, ValueError):
                raise
        raise ValueError("encoded sample is missing a valid length/input_ids")

    @contextmanager
    def _template_state_context(
        self,
        *,
        packing: Optional[bool] = None,
        padding_free: Optional[bool] = None,
        mode: Optional[str] = None,
    ):
        template = self.template

        lock = getattr(self, "_template_toggle_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._template_toggle_lock = lock

        tls = getattr(self, "_template_toggle_tls", None)
        if tls is None:
            tls = threading.local()
            self._template_toggle_tls = tls

        depth = int(getattr(tls, "depth", 0) or 0)
        acquired = False
        if depth == 0:
            lock.acquire()
            acquired = True
            tls.stack = []

        tls.depth = depth + 1
        stack = getattr(tls, "stack", None)
        if stack is None:
            stack = []
            tls.stack = stack

        old_padding_free = getattr(template, "padding_free", False)
        old_packing = getattr(template, "packing", False)
        old_mode = getattr(template, "mode", None)
        stack.append((old_padding_free, old_packing, old_mode))

        try:
            if packing is not None:
                try:
                    template.packing = bool(packing)
                except (TypeError, ValueError):
                    raise
            if padding_free is not None:
                try:
                    template.padding_free = bool(padding_free)
                except (TypeError, ValueError):
                    raise
            if (
                mode is not None
                and old_mode is not None
                and hasattr(template, "set_mode")
            ):
                try:
                    if str(old_mode) != str(mode):
                        template.set_mode(str(mode))
                except (TypeError, ValueError):
                    raise

            yield
        finally:
            try:
                old_padding_free, old_packing, old_mode = stack.pop()
            except (TypeError, ValueError):
                old_padding_free, old_packing, old_mode = False, False, None

            if old_mode is not None and hasattr(template, "set_mode"):
                try:
                    template.set_mode(old_mode)
                except (TypeError, ValueError):
                    raise
            try:
                template.padding_free = old_padding_free
            except (TypeError, ValueError):
                raise
            try:
                template.packing = old_packing
            except (TypeError, ValueError):
                raise

            try:
                tls.depth = int(getattr(tls, "depth", 1) or 1) - 1
            except (TypeError, ValueError):
                tls.depth = 0

            if int(getattr(tls, "depth", 0) or 0) <= 0:
                tls.depth = 0
                try:
                    tls.stack = []
                except (TypeError, ValueError):
                    raise
                if acquired:
                    lock.release()

    @contextmanager
    def _template_packing_disabled(self):
        """Temporarily disable ms-swift template packing/padding-free flags."""
        with self._template_state_context(packing=False, padding_free=False):
            yield

    @contextmanager
    def _template_train_mode(self):
        """Temporarily force template mode to `train` for teacher-forced encoding.

        Some runners may keep the template in a non-training mode (e.g. `pt`), which
        would strip assistant responses from messages. Stage_2 needs the assistant span
        present in `input_ids` for masking/loss construction.
        """
        with self._template_state_context(mode="train"):
            yield

    @contextmanager
    def _template_packing_enabled(self):
        """Temporarily enable ms-swift template packing/padding-free flags."""
        with self._template_state_context(packing=True, padding_free=True):
            yield

    def _maybe_debug_dump_parse_failure(
        self,
        *,
        sample: Mapping[str, Any],
        response_text: str,
        prefix_text: str,
        dropped_invalid: int,
        dropped_ambiguous: int,
        truncated: bool,
        decode_mode: str,
    ) -> None:
        if not bool(self._cfg("debug_dump_parse_failures", False)):
            return
        max_dumps = int(self._cfg("debug_dump_max", 3))
        if max_dumps <= 0 or self._debug_dump_count >= max_dumps:
            return
        if dropped_invalid <= 0 and dropped_ambiguous <= 0 and not truncated:
            return

        self._debug_dump_count += 1
        images = (
            sample.get("images") if isinstance(sample.get("images"), list) else None
        )
        image = sample.get("image") if isinstance(sample.get("image"), str) else None
        tag = f"images={images!r}" if images else f"image={image!r}"

        def _clip(text: str, n: int = 600) -> str:
            t = text.replace("\n", "\\n")
            if len(t) <= n:
                return t
            return t[:n] + "...<truncated>"

        logger.warning(
            "rollout debug dump #%s (mode=%s %s): dropped_invalid=%s dropped_ambiguous=%s truncated=%s raw=%s prefix=%s",
            self._debug_dump_count,
            decode_mode,
            tag,
            dropped_invalid,
            dropped_ambiguous,
            truncated,
            _clip(response_text),
            _clip(prefix_text),
        )

    def _get_coord_token_ids(self) -> List[int]:
        if self._coord_token_ids is not None:
            return self._coord_token_ids
        tok = getattr(getattr(self, "template", None), "tokenizer", None)
        if tok is None:
            return []
        ids = get_coord_token_ids(tok, validate=True)
        self._coord_token_ids = [int(i) for i in ids]
        self._coord_id_to_bin = {int(tok_id): int(i) for i, tok_id in enumerate(ids)}
        return self._coord_token_ids

    def _coord_id_map(self) -> Dict[int, int]:
        if self._coord_id_to_bin is None:
            _ = self._get_coord_token_ids()
        return self._coord_id_to_bin or {}

    # ------------------------ rollout + batch prep ------------------------ #
    # ---- rollout backends -------------------------------------------------
    def _ensure_vllm_engine(self) -> Any:
        """Initialize a colocated vLLM engine (lazy)."""
        if self._vllm_engine is not None:
            return self._vllm_engine

        vcfg_raw = self._cfg("vllm", {}) or {}
        if not isinstance(vcfg_raw, Mapping):
            raise ValueError("rollout_matching.vllm must be a mapping")
        vcfg = dict(vcfg_raw)

        try:
            import torch.distributed as dist
        except (TypeError, ValueError):
            dist = None  # type: ignore[assignment]

        world_size = 1
        if dist is not None and dist.is_available() and dist.is_initialized():
            world_size = int(dist.get_world_size())

        # Defaults: TP=4 on 4-GPU runs; otherwise TP=1 unless explicitly set.
        default_tp = 4 if world_size == 4 else 1
        tp_size = int(vcfg.get("tensor_parallel_size", default_tp))
        if tp_size <= 0:
            raise ValueError(
                "rollout_matching.vllm.tensor_parallel_size must be > 0"
            )
        if world_size % tp_size != 0:
            raise ValueError(
                f"vLLM colocate requires world_size % tp == 0; world_size={world_size} tp={tp_size}"
            )

        max_model_len_raw = vcfg.get("max_model_len", None)
        if max_model_len_raw is None:
            raise ValueError(
                "rollout_matching.vllm.max_model_len is required when rollout_backend=vllm "
                "(it must cover prompt_len + max_new_tokens)."
            )
        max_model_len = int(max_model_len_raw)
        if max_model_len <= 0:
            raise ValueError(
                "rollout_matching.vllm.max_model_len must be > 0"
            )

        # NOTE: vLLM LoRA on multimodal models (Qwen3-VL ViT) can be unstable on
        # some stacks. Allow disabling vLLM LoRA and instead syncing merged
        # weights into the colocated vLLM engine (GRPO-style).
        enable_lora = bool(vcfg.get("enable_lora", False))

        load_format = vcfg.get("load_format", None)
        if load_format is None:
            # When we sync weights from the training model, loading real weights
            # from disk is unnecessary; dummy init reduces overhead.
            load_format = "dummy" if not enable_lora else "auto"
        if not isinstance(load_format, str):
            raise ValueError(
                "rollout_matching.vllm.load_format must be a string"
            )
        load_format = load_format.strip()

        gpu_mem = float(vcfg.get("gpu_memory_utilization", 0.45))
        enable_prefix_caching = bool(vcfg.get("enable_prefix_caching", True))
        sleep_level = int(vcfg.get("sleep_level", 0) or 0)
        enforce_eager = bool(vcfg.get("enforce_eager", False))
        disable_custom_all_reduce = bool(vcfg.get("disable_custom_all_reduce", True))
        max_num_seqs_raw = vcfg.get("max_num_seqs", None)
        max_num_seqs: Optional[int] = None
        if max_num_seqs_raw is not None:
            try:
                max_num_seqs = int(max_num_seqs_raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "rollout_matching.vllm.max_num_seqs must be an int"
                ) from exc
            if max_num_seqs <= 0:
                raise ValueError(
                    "rollout_matching.vllm.max_num_seqs must be > 0"
                )

        # Extra vLLM engine kwargs (passed through to vLLM EngineArgs by ms-swift VllmEngine).
        # This is useful to avoid hard-coded vLLM defaults that can break long-context multimodal rollouts.
        vllm_engine_kwargs: Dict[str, Any] = {}
        limit_mm_per_prompt: Optional[Dict[str, int]] = None
        max_num_batched_tokens_raw = vcfg.get("max_num_batched_tokens", None)
        if max_num_batched_tokens_raw is not None:
            try:
                max_num_batched_tokens = int(max_num_batched_tokens_raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "rollout_matching.vllm.max_num_batched_tokens must be an int"
                ) from exc
            if max_num_batched_tokens <= 0:
                raise ValueError(
                    "rollout_matching.vllm.max_num_batched_tokens must be > 0"
                )
            vllm_engine_kwargs["max_num_batched_tokens"] = max_num_batched_tokens

        # Optional vLLM EngineArgs knobs (allow-list).
        #
        # NOTE: These are passed to vLLM via ms-swift's VllmEngine(engine_kwargs=...),
        # which forwards them into vLLM EngineArgs. Keep this list small and validated
        # to avoid silent typos.
        if "enable_chunked_prefill" in vcfg:
            vllm_engine_kwargs["enable_chunked_prefill"] = bool(
                vcfg.get("enable_chunked_prefill")
            )
        if "disable_chunked_mm_input" in vcfg:
            vllm_engine_kwargs["disable_chunked_mm_input"] = bool(
                vcfg.get("disable_chunked_mm_input")
            )
        if "kv_cache_dtype" in vcfg and vcfg.get("kv_cache_dtype") is not None:
            kv_cache_dtype = vcfg.get("kv_cache_dtype")
            if not isinstance(kv_cache_dtype, str):
                raise ValueError(
                    "rollout_matching.vllm.kv_cache_dtype must be a string"
                )
            vllm_engine_kwargs["kv_cache_dtype"] = kv_cache_dtype
        if "cpu_offload_gb" in vcfg and vcfg.get("cpu_offload_gb") is not None:
            try:
                cpu_offload_gb = float(vcfg.get("cpu_offload_gb"))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "rollout_matching.vllm.cpu_offload_gb must be a float"
                ) from exc
            if cpu_offload_gb < 0:
                raise ValueError(
                    "rollout_matching.vllm.cpu_offload_gb must be >= 0"
                )
            vllm_engine_kwargs["cpu_offload_gb"] = cpu_offload_gb
        if "swap_space" in vcfg and vcfg.get("swap_space") is not None:
            try:
                swap_space = float(vcfg.get("swap_space"))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "rollout_matching.vllm.swap_space must be a float"
                ) from exc
            if swap_space < 0:
                raise ValueError(
                    "rollout_matching.vllm.swap_space must be >= 0"
                )
            vllm_engine_kwargs["swap_space"] = swap_space
        if (
            "limit_mm_per_prompt" in vcfg
            and vcfg.get("limit_mm_per_prompt") is not None
        ):
            limit_raw = vcfg.get("limit_mm_per_prompt")
            if not isinstance(limit_raw, Mapping):
                raise ValueError(
                    "rollout_matching.vllm.limit_mm_per_prompt must be a mapping"
                )
            limit_parsed: Dict[str, int] = {}
            for k, v in limit_raw.items():
                if not isinstance(k, str):
                    raise ValueError(
                        "rollout_matching.vllm.limit_mm_per_prompt keys must be strings"
                    )
                try:
                    limit_parsed[k] = int(v)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "rollout_matching.vllm.limit_mm_per_prompt values must be ints"
                    ) from exc
            # NOTE: ms-swift's `VllmEngine` already exposes `limit_mm_per_prompt` as a
            # top-level kwarg, and forwards it to `_prepare_engine_kwargs`. Passing it
            # again via `engine_kwargs` would cause a `got multiple values` TypeError.
            limit_mm_per_prompt = limit_parsed

        if "mm_encoder_tp_mode" in vcfg and vcfg.get("mm_encoder_tp_mode") is not None:
            mm_encoder_tp_mode = vcfg.get("mm_encoder_tp_mode")
            if not isinstance(mm_encoder_tp_mode, str):
                raise ValueError(
                    "rollout_matching.vllm.mm_encoder_tp_mode must be a string"
                )
            mm_encoder_tp_mode = mm_encoder_tp_mode.strip().lower()
            if mm_encoder_tp_mode not in {"weights", "data"}:
                raise ValueError(
                    "rollout_matching.vllm.mm_encoder_tp_mode must be 'weights' or 'data'"
                )
            vllm_engine_kwargs["mm_encoder_tp_mode"] = mm_encoder_tp_mode

        # Multimodal encoder cache profiling can create extremely large dummy
        # multimodal batches (e.g., video) during vLLM initialization, which is
        # unnecessary for our image-only rollouts and can interact badly with
        # vLLM LoRA kernels. Allow skipping it via EngineArgs.skip_mm_profiling.
        if "skip_mm_profiling" in vcfg:
            vllm_engine_kwargs["skip_mm_profiling"] = bool(
                vcfg.get("skip_mm_profiling")
            )

        # Patch vLLM to allow loading LoRA from in-memory tensors (only needed
        # when vLLM LoRA is enabled).
        if enable_lora:
            try:
                from swift.trainers.rlhf_trainer.utils import patch_vllm_load_adapter

                patch_vllm_load_adapter()
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    "vLLM rollout backend is enabled but vLLM is unavailable or incompatible. "
                    "Install/upgrade vLLM in the ms env, or set rollout_matching.rollout_backend: hf."
                ) from exc

        model_dir = getattr(self.model, "model_dir", None) or getattr(
            getattr(self.model, "model", None), "model_dir", None
        )
        if not model_dir:
            raise RuntimeError(
                "vLLM rollout backend requires a ms-swift model wrapper with `model_dir`. "
                "Set rollout_backend: hf to disable vLLM rollouts."
            )
        model_info = getattr(self.model, "model_info", None)
        torch_dtype = (
            getattr(model_info, "torch_dtype", None) if model_info is not None else None
        )

        # Derive max_lora_rank from peft config when possible (must be >= actual rank).
        max_lora_rank = 16
        if enable_lora:
            peft_cfg = getattr(self.model, "peft_config", None)
            if isinstance(peft_cfg, Mapping) and peft_cfg:
                cfg0 = peft_cfg.get("default") or next(iter(peft_cfg.values()), None)
                try:
                    max_lora_rank = int(getattr(cfg0, "r", max_lora_rank))
                except (TypeError, ValueError):
                    raise

        # Build TP subgroup (colocate only; server mode unsupported here).
        if tp_size > 1:
            if dist is None or not dist.is_initialized():
                raise RuntimeError(
                    "vLLM tensor parallel requires torch.distributed to be initialized"
                )
            self._vllm_tp_group, _ = dist.new_subgroups_by_enumeration(
                [
                    list(range(i * tp_size, (i + 1) * tp_size))
                    for i in range(world_size // tp_size)
                ]
            )
        self._vllm_tp_size = int(tp_size)

        # Use a shallow-copied template; vLLM expects template.mode='vllm'.
        vllm_template = shallow_copy(self.template)
        try:
            vllm_template.packing = False
            vllm_template.padding_free = False
            vllm_template.set_mode("vllm")
        except (TypeError, ValueError):
            raise

        logger.info(
            "Initializing vLLM rollout engine: tp=%s world_size=%s max_model_len=%s gpu_memory_utilization=%.2f "
            "max_num_seqs=%s limit_mm_per_prompt=%s engine_kwargs=%s",
            tp_size,
            world_size,
            max_model_len,
            gpu_mem,
            max_num_seqs if max_num_seqs is not None else 256,
            limit_mm_per_prompt,
            vllm_engine_kwargs or {},
        )

        dist_backend_raw = vcfg.get("distributed_executor_backend")
        if dist_backend_raw is None:
            dist_backend = (
                "mp" if world_size == 1 and tp_size == 1 else "external_launcher"
            )
        else:
            dist_backend = str(dist_backend_raw).strip() or (
                "mp" if world_size == 1 and tp_size == 1 else "external_launcher"
            )

        try:
            from swift.llm import VllmEngine

            engine = VllmEngine(
                model_dir,
                torch_dtype=torch_dtype,
                template=vllm_template,
                tensor_parallel_size=tp_size,
                gpu_memory_utilization=gpu_mem,
                max_model_len=max_model_len,
                max_num_seqs=max_num_seqs if max_num_seqs is not None else 256,
                enforce_eager=enforce_eager,
                disable_custom_all_reduce=disable_custom_all_reduce,
                limit_mm_per_prompt=limit_mm_per_prompt,
                load_format=load_format,
                enable_lora=enable_lora,
                max_loras=1,
                max_lora_rank=max_lora_rank,
                enable_prefix_caching=enable_prefix_caching,
                engine_kwargs=vllm_engine_kwargs or None,
                distributed_executor_backend=dist_backend,
            )
        except (TypeError, ValueError) as exc:
            logger.exception(
                "vLLM engine init failed (backend=%s): %s", dist_backend, exc
            )
            raise RuntimeError(
                "Failed to initialize vLLM engine for rollout generation. "
                "Set rollout_backend: hf to bypass vLLM."
            ) from exc

        if sleep_level > 0:
            try:
                engine.engine.sleep(sleep_level)
            except (TypeError, ValueError):
                raise

        self._vllm_engine = engine
        return engine

    def _sync_vllm_rollout_model_if_needed(self) -> None:
        """Sync the rollout model weights into the colocated vLLM engine.

        We support two modes:
        - vLLM LoRA enabled: push adapter tensors via `add_lora` (fast, but can be
          unstable on some multimodal stacks).
        - vLLM LoRA disabled: merge adapters into the training model weights and
          load the merged weights into vLLM (GRPO-style; more robust for ViT).
        """
        vcfg = self._cfg("vllm", {}) or {}
        enable_lora = (
            bool(vcfg.get("enable_lora", False)) if isinstance(vcfg, Mapping) else False
        )
        if enable_lora:
            self._sync_vllm_lora_if_needed()
        else:
            self._sync_vllm_full_weights_if_needed()

    def _sync_vllm_full_weights_if_needed(self) -> None:
        """Sync merged (LoRA-applied) weights into vLLM when vLLM LoRA is disabled."""
        step = int(getattr(getattr(self, "state", None), "global_step", 0) or 0)
        if step == self._vllm_last_loaded_step:
            return

        engine = self._ensure_vllm_engine()

        from contextlib import nullcontext

        try:
            from accelerate.utils import is_peft_model
        except (TypeError, ValueError):
            is_peft_model = None  # type: ignore[assignment]

        is_peft = (
            bool(is_peft_model(self.model)) if is_peft_model is not None else False
        )

        merge_cm = nullcontext()
        unmerge_cm = nullcontext()
        if is_peft:
            try:
                from swift.trainers.rlhf_trainer.utils import (
                    patch_lora_merge,
                    patch_lora_unmerge,
                )

                merge_cm = patch_lora_merge(self.model)
                unmerge_cm = patch_lora_unmerge(self.model)
            except (TypeError, ValueError):
                merge_cm = nullcontext()
                unmerge_cm = nullcontext()

        params = [p for _, p in self.model.named_parameters()]
        gather_if_zero3 = get_gather_if_zero3_context(self)

        with gather_if_zero3(params), merge_cm, torch.no_grad():
            merged = False
            try:
                if is_peft:
                    try:
                        # Merge adapter weights into base weights for extraction.
                        self.model.merge_adapter()
                        merged = True
                    except (TypeError, ValueError) as exc:
                        raise RuntimeError(
                            "vLLM LoRA is disabled, but we failed to merge the adapter weights from the training "
                            "model. Mitigations: set rollout_matching.vllm.enable_lora=true "
                            "(may be unstable on multimodal), or ensure your PEFT stack supports "
                            "merge_adapter/unmerge_adapter."
                        ) from exc

                state_dict = self.model.state_dict()
                if is_peft:
                    # Follow ms-swift GRPO key mapping conventions to match vLLM model names.
                    prefix_removed = {
                        k.removeprefix("base_model.model."): v
                        for k, v in state_dict.items()
                    }
                    state_dict = {
                        k.replace(".base_layer", ""): v
                        for k, v in prefix_removed.items()
                    }
                    prefix = getattr(self.model, "prefix", None)
                    if isinstance(prefix, str) and prefix:
                        state_dict = {
                            k: v for k, v in state_dict.items() if prefix not in k
                        }
                    state_dict = {
                        k.replace("modules_to_save.default.", ""): v
                        for k, v in state_dict.items()
                        if "original_module" not in k
                    }
                    # vLLM LoRA is disabled: do not pass LoRA tensors (they're already merged).
                    state_dict = {
                        k: v for k, v in state_dict.items() if "lora_" not in k
                    }

                engine.inner_model.load_weights(state_dict.items())
            finally:
                # Never leave the training model in merged state, even if vLLM loading fails.
                if is_peft and merged:
                    with unmerge_cm:
                        self.model.unmerge_adapter()

        try:
            engine.engine.reset_prefix_cache()
        except (TypeError, ValueError):
            raise

        self._vllm_last_loaded_step = step

    def _sync_vllm_lora_if_needed(self) -> None:
        """Sync LoRA adapter weights into vLLM on global_step boundaries."""
        step = int(getattr(getattr(self, "state", None), "global_step", 0) or 0)
        if step == self._vllm_last_loaded_step:
            return

        engine = self._ensure_vllm_engine()

        # Build LoRA tensors (CPU) for vLLM.
        peft_cfg = getattr(self.model, "peft_config", None)
        if not isinstance(peft_cfg, Mapping) or not peft_cfg:
            raise RuntimeError(
                "vLLM rollout backend requires a PEFT LoRA model (peft_config missing). "
                "Switch rollout_backend: hf if training is not LoRA."
            )
        cfg0 = peft_cfg.get("default") or next(iter(peft_cfg.values()), None)

        try:
            peft_cfg_dict = asdict(cfg0)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            if hasattr(cfg0, "to_dict"):
                peft_cfg_dict = cfg0.to_dict()  # type: ignore[assignment]
            else:
                peft_cfg_dict = {}

        try:
            from peft.utils.save_and_load import get_peft_model_state_dict
        except (TypeError, ValueError) as exc:
            raise RuntimeError("peft is required for vLLM LoRA sync") from exc

        named = [(n, p) for n, p in self.model.named_parameters() if "lora_" in n]
        if not named:
            raise RuntimeError(
                "No LoRA parameters found on model, but vLLM LoRA sync is required. "
                "Disable vLLM (rollout_backend: hf) or ensure LoRA is enabled."
            )
        names = [n for n, _ in named]
        params = [p for _, p in named]
        gather_if_zero3 = get_gather_if_zero3_context(self)
        with gather_if_zero3(params):
            subset = {}
            for n, p in zip(names, params):
                t = p.full_tensor() if hasattr(p, "full_tensor") else p
                subset[n] = t.detach()
            lora_params = get_peft_model_state_dict(self.model, subset)
            lora_params = {
                k: (v.full_tensor() if hasattr(v, "full_tensor") else v).detach().cpu()
                for k, v in lora_params.items()
            }

        try:
            from swift.trainers.rlhf_trainer.utils import TensorLoRARequest
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "Unable to import TensorLoRARequest for vLLM LoRA sync"
            ) from exc
        if TensorLoRARequest is None:
            raise RuntimeError("vLLM is not available (TensorLoRARequest is None)")

        lora_int_id = int(time.time_ns() % 0x7FFFFFFF)
        lora_request = TensorLoRARequest(
            lora_name=f"{lora_int_id}",
            lora_int_id=lora_int_id,
            lora_path="dummy_lora_path",
            peft_config=peft_cfg_dict,
            lora_tensors=lora_params,
        )
        try:
            engine.engine.add_lora(lora_request)
            try:
                engine.engine.reset_prefix_cache()
            except (TypeError, ValueError):
                raise
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "Failed to load LoRA adapter into vLLM. "
                "If this is due to multimodal/Vision LoRA incompatibility, freeze ViT or set rollout_backend: hf."
            ) from exc

        self._vllm_last_loaded_step = step

    def _effective_vllm_server_sync_mode(self) -> str:
        vcfg_raw = self._cfg("vllm", {}) or {}
        enable_lora = (
            bool(vcfg_raw.get("enable_lora", False))
            if isinstance(vcfg_raw, Mapping)
            else False
        )

        mode, _fallback_to_full = self._vllm_server_sync_cfg()
        if bool(getattr(self, "_vllm_server_force_full_sync", False)):
            return "full"
        if mode == "auto":
            return "adapter" if enable_lora else "full"
        return mode

    def _ensure_vllm_server_client(self) -> Any:
        """Create an ms-swift vLLM server client (lazy).

        Important:
        - HTTP `/infer/` does NOT require the NCCL communicator.
        - The NCCL communicator is only required for in-memory weight sync.
        - Under a multi-process learner (`torchrun`, `world_size>1`), communicator init
          MUST be rank0-only.

        Thread safety:
        - Stage2-AB async actor-learner may call this from a background prefetch thread.
          Guard creation so we never build multiple clients / init communicator twice.
        """
        if self._vllm_server_client is not None:
            return self._vllm_server_client

        lock = getattr(self, "_vllm_server_client_lock", None)
        if lock is None:
            lock = threading.Lock()
            setattr(self, "_vllm_server_client_lock", lock)

        with lock:
            if self._vllm_server_client is not None:
                return self._vllm_server_client

            # Dist metadata is used only for communicator-init behavior.
            rank = 0
            world_size = 1
            try:
                import torch.distributed as dist
            except (TypeError, ValueError):
                dist = None  # type: ignore[assignment]

            if dist is not None and dist.is_available() and dist.is_initialized():
                try:
                    rank = int(dist.get_rank())
                    world_size = int(dist.get_world_size())
                except (TypeError, ValueError):
                    rank = 0
                    world_size = 1

            servers = self._vllm_server_specs()
            timeout_s, _infer_timeout_s = self._vllm_server_timeouts()

            try:
                from swift.trainers.rlhf_trainer.vllm_client import VLLMClient
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    "vLLM server mode requires ms-swift's VLLMClient (and vLLM + pynccl). "
                    "Install/enable vLLM in the ms env, or switch to vllm.mode=colocate or rollout_backend=hf."
                ) from exc

            base_urls = [str(s["base_url"]) for s in servers]
            group_ports = [int(s["group_port"]) for s in servers]

            try:
                client = VLLMClient(
                    base_urls=base_urls,
                    group_ports=group_ports,
                    connection_timeout=float(timeout_s),
                )
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    "Failed to connect to vLLM rollout server(s). "
                    "Check rollout_matching.vllm.server (base_url/group_port) and ensure /health/ is reachable."
                ) from exc

            # Communicator init is deferred until first weight sync.
            if int(world_size) > 1 and int(rank) == 0:
                logger.info(
                    "vLLM server client created for multi-process learner; communicator init deferred (rank0-only). world_size=%s",
                    int(world_size),
                )

            # Best-effort: log server runtime type (enable_lora, async engine, etc.).
            try:
                info = client.get_engine_type()
                logger.info("vLLM rollout server engine_type: %s", info)
            except (TypeError, ValueError):
                raise

            self._vllm_server_client = client
            return client

    def _ensure_vllm_server_communicator_rank0(self, client: Any) -> None:
        """Initialize vLLM server NCCL communicator (rank0-only under DDP)."""
        if bool(getattr(self, "_vllm_server_comm_inited", False)):
            return

        rank = 0
        try:
            import torch.distributed as dist
        except (TypeError, ValueError):
            dist = None  # type: ignore[assignment]

        if dist is not None and dist.is_available() and dist.is_initialized():
            rank = int(dist.get_rank())

        if int(rank) != 0:
            raise RuntimeError(
                "vLLM server communicator init must be rank0-only under DDP. "
                f"Got rank={int(rank)}."
            )

        try:
            client.init_communicator(device=0)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "Failed to initialize NCCL communicator with vLLM rollout server(s). "
                "Mitigations: verify group_port reachability, set NCCL env, or increase vllm.server.timeout_s."
            ) from exc

        setattr(self, "_vllm_server_comm_inited", True)

    def _shutdown_vllm_server_client(
        self, *, close_communicator: bool = True, close_sessions: bool = True
    ) -> None:
        """Best-effort cleanup for vLLM server client resources.

        This is called during trainer shutdown to reduce teardown races between
        background rollout workers and process-exit cleanup.
        """
        lock = getattr(self, "_vllm_server_client_lock", None)
        if lock is None:
            lock = threading.Lock()
            setattr(self, "_vllm_server_client_lock", lock)

        with lock:
            client = getattr(self, "_vllm_server_client", None)
            if client is None:
                setattr(self, "_vllm_server_comm_inited", False)
                return

            rank = 0
            world_size = 1
            try:
                import torch.distributed as dist
            except (TypeError, ValueError):
                dist = None  # type: ignore[assignment]

            if dist is not None and dist.is_available() and dist.is_initialized():
                try:
                    rank = int(dist.get_rank())
                    world_size = int(dist.get_world_size())
                except (TypeError, ValueError):
                    rank = 0
                    world_size = 1

            if bool(close_communicator):
                should_close_comm = int(world_size) <= 1 or int(rank) == 0
                if should_close_comm:
                    try:
                        close_fn = getattr(client, "close_communicator", None)
                        if callable(close_fn):
                            close_fn()
                    except (TypeError, ValueError) as exc:
                        logger.warning(
                            "Failed to close vLLM server communicator during shutdown: %r",
                            exc,
                        )

            if bool(close_sessions):
                try:
                    sessions = getattr(client, "sessions", None)
                    if isinstance(sessions, list):
                        for sess in sessions:
                            try:
                                close_fn = getattr(sess, "close", None)
                                if callable(close_fn):
                                    close_fn()
                            except (TypeError, ValueError):
                                raise
                except (TypeError, ValueError):
                    raise

            self._vllm_server_client = None
            setattr(self, "_vllm_server_comm_inited", False)

    def _vllm_server_infer_guard(self):
        """Optional hook for staging safe vLLM server inference.

        Stage2-AB async actor-learner may override this to prevent HTTP `/infer/` calls
        from racing with rank0 weight sync.
        """
        return nullcontext()

    def _maybe_debug_dump_vllm_server_rollouts(
        self,
        *,
        global_step: int,
        seed_base: int,
        infer_requests: Sequence[Mapping[str, Any]],
        outputs: Sequence[Tuple[List[int], str, str, List[int]]],
        samples: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> None:
        """Optional raw rollout dump for diagnosing vLLM server-mode formatting.

        Controlled via:
          rollout_matching.vllm.server.debug_dump:
            enabled: true
            every_steps: 10           # defaults to args.logging_steps
            dump_first_step: false    # defaults to args.logging_first_step
            only_world_process_zero: true
            max_events: 3
            max_samples: 1
            max_chars: 4000           # <=0 disables clipping (full raw text)
            out_dir: null             # defaults to <output_dir>/vllm_server_debug

        Notes:
        - In DDP, the default is rank0-only dumping to avoid I/O storms.
        - If only_world_process_zero=false, dumps go to per-rank subdirectories.
        - Payload is intentionally minimal for human review: GT text + rollout text.
        """

        try:
            scfg_raw = self._vllm_server_cfg()
        except (AttributeError, TypeError, ValueError):
            return

        debug_raw = (
            scfg_raw.get("debug_dump", {}) if isinstance(scfg_raw, Mapping) else {}
        )
        if not isinstance(debug_raw, Mapping) or not bool(
            debug_raw.get("enabled", False)
        ):
            return

        only_main = bool(debug_raw.get("only_world_process_zero", True))
        if only_main and not self._is_main_process():
            return

        max_events = int(debug_raw.get("max_events", 3) or 0)
        if max_events > 0 and int(self._vllm_server_debug_dump_count) >= int(
            max_events
        ):
            return

        gs = int(global_step)
        if (
            self._vllm_server_debug_last_step is not None
            and int(self._vllm_server_debug_last_step) == gs
        ):
            return

        every = debug_raw.get("every_steps", None)
        if every is None:
            every = int(getattr(self.args, "logging_steps", 1) or 1)
        every = max(1, int(every))

        dump_first = bool(
            debug_raw.get(
                "dump_first_step", bool(getattr(self.args, "logging_first_step", False))
            )
        )
        if gs == 0 and not dump_first:
            return
        if gs % every != 0:
            return

        out_dir = debug_raw.get("out_dir")
        if not isinstance(out_dir, str) or not out_dir.strip():
            out_dir = os.path.join(
                str(getattr(self.args, "output_dir", ".")), "vllm_server_debug"
            )

        rank = 0
        world_size = 1
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                rank = int(dist.get_rank())
                world_size = int(dist.get_world_size())
        except (TypeError, ValueError):
            rank = 0
            world_size = 1

        # DDP-safe output naming: if dumping from multiple ranks, isolate paths.
        if (not only_main) and int(world_size) > 1:
            out_dir = os.path.join(str(out_dir), f"rank_{int(rank)}")

        os.makedirs(out_dir, exist_ok=True)

        self._vllm_server_debug_last_step = int(gs)
        self._vllm_server_debug_dump_count += 1
        event = int(self._vllm_server_debug_dump_count)

        max_samples = int(debug_raw.get("max_samples", 1) or 1)
        max_chars_raw = debug_raw.get("max_chars", 4000)
        try:
            max_chars = int(max_chars_raw) if max_chars_raw is not None else 0
        except (TypeError, ValueError):
            max_chars = 4000

        def _content_to_text(content: Any) -> str:
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts: List[str] = []
                for item in content:
                    if isinstance(item, Mapping):
                        # OpenAI-style multimodal content item.
                        if item.get("type") == "text" and item.get("text") is not None:
                            parts.append(str(item.get("text")))
                        elif item.get("text") is not None:
                            parts.append(str(item.get("text")))
                    elif item is not None:
                        parts.append(str(item))
                return "\n".join(parts)
            if content is None:
                return ""
            text = ""
            try:
                text = str(content)
            except (TypeError, ValueError):
                text = ""
            return text

        def _extract_gt_text(sample_obj: Any) -> str:
            if not isinstance(sample_obj, Mapping):
                return ""
            messages = sample_obj.get("messages")
            if not isinstance(messages, list):
                return ""
            for m in reversed(messages):
                if not isinstance(m, Mapping):
                    continue
                if str(m.get("role", "")).lower() != "assistant":
                    continue
                return _content_to_text(m.get("content"))
            return ""

        sample_list: List[Mapping[str, Any]] = []
        if isinstance(samples, Sequence):
            for s in samples:
                if isinstance(s, Mapping):
                    sample_list.append(s)
                else:
                    sample_list.append({})

        samples_dump: List[Dict[str, Any]] = []
        for i, (_, out) in enumerate(zip(infer_requests, outputs)):
            if i >= max_samples:
                break

            _, resp_text, _, _ = out
            sample_obj = sample_list[i] if i < len(sample_list) else {}

            gt_text_raw = _extract_gt_text(sample_obj)
            gt_text = self._ascii_safe_text(
                self._clip_text(gt_text_raw, max_chars=max_chars)
            )
            rollout_text = self._ascii_safe_text(
                self._clip_text(resp_text, max_chars=max_chars)
            )

            samples_dump.append(
                {
                    "gt_text": gt_text,
                    "rollout_text": rollout_text,
                }
            )

        payload = {
            "global_step": int(gs),
            "samples": samples_dump,
        }

        path = os.path.join(
            out_dir,
            f"step_{int(gs):06d}_event_{int(event):03d}.json",
        )
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=True, indent=2)
            logger.warning(
                "vLLM server debug dump wrote %s (samples=%s)", path, len(samples_dump)
            )
        except (OSError, TypeError, ValueError):
            return

    def _sync_vllm_server_rollout_model_if_needed(self) -> None:
        """Sync weights/adapters to rollout server for vLLM server mode.

        DDP safety (when torch.distributed is initialized):
        - Rank0-only communicator init + weight push.
        - Strict ordering: barrier -> rank0 sync -> barrier.
        - All ranks must take the same control-flow to avoid deadlocks.

        NOTE: This sync is intended for the synchronous rollout path.
        Async actor-learner should coordinate server sync at safe boundaries and
        avoid invoking DDP collectives from background prefetch threads.
        """
        step = int(getattr(getattr(self, "state", None), "global_step", 0) or 0)

        rank = 0
        world_size = 1
        try:
            import torch.distributed as dist
        except (TypeError, ValueError):
            dist = None  # type: ignore[assignment]

        if dist is not None and dist.is_available() and dist.is_initialized():
            rank = int(dist.get_rank())
            world_size = int(dist.get_world_size())

        last = int(getattr(self, "_vllm_server_last_synced_step", -1))
        need_sync = int(step != last)

        # Under DDP, ensure all ranks take the same early-return decision.
        if (
            dist is not None
            and dist.is_available()
            and dist.is_initialized()
            and int(world_size) > 1
        ):
            try:
                try:
                    backend = str(dist.get_backend()).lower()
                except (TypeError, ValueError):
                    backend = ""

                reduce_device = torch.device("cpu")
                if backend == "nccl" and torch.cuda.is_available():
                    reduce_device = self.model.device

                flag = torch.tensor(
                    [need_sync], device=reduce_device, dtype=torch.int32
                )
                dist.broadcast(flag, src=0)
                need_sync = int(flag.item())
            except (TypeError, ValueError):
                need_sync = int(step != last)

        if need_sync == 0:
            return

        eff_mode = self._effective_vllm_server_sync_mode()
        if int(world_size) > 1 and eff_mode != "full":
            raise ValueError(
                "rollout_matching.vllm.sync.mode must resolve to 'full' under multi-process learners "
                f"(world_size={int(world_size)}). Got effective sync mode={eff_mode!r}."
            )

        # Single-process learner: allow adapter/full sync modes.
        if (
            dist is None
            or (not dist.is_available())
            or (not dist.is_initialized())
            or int(world_size) == 1
        ):
            vcfg_raw = self._cfg("vllm", {}) or {}
            enable_lora = (
                bool(vcfg_raw.get("enable_lora", False))
                if isinstance(vcfg_raw, Mapping)
                else False
            )

            mode, fallback_to_full = self._vllm_server_sync_cfg()
            _ = mode

            if eff_mode == "adapter" and not enable_lora:
                raise ValueError(
                    "rollout_matching.vllm.sync.mode=adapter requires rollout_matching.vllm.enable_lora: true"
                )

            client = self._ensure_vllm_server_client()
            if not bool(getattr(self, "_vllm_server_comm_inited", False)):
                try:
                    client.init_communicator(device=0)
                    setattr(self, "_vllm_server_comm_inited", True)
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(
                        "Failed to initialize NCCL communicator with vLLM rollout server(s). "
                        "Mitigations: verify group_port reachability, set NCCL env, or increase vllm.server.timeout_s."
                    ) from exc

            if eff_mode == "adapter":
                # Optional runtime sanity check (server must have vLLM LoRA enabled).
                try:
                    info = client.get_engine_type()
                    if isinstance(info, dict) and not bool(
                        info.get("enable_lora", False)
                    ):
                        raise RuntimeError(
                            "vLLM server reports enable_lora=false, but adapter-only sync was requested. "
                            "Launch the rollout server with vLLM LoRA enabled (e.g. swift rollout --vllm_enable_lora true), "
                            "or set vllm.sync.mode=full."
                        )
                except (TypeError, ValueError) as exc:
                    # If the check itself fails, continue; sync will fail with a clearer error.
                    logger.warning(
                        "Unable to verify rollout server LoRA capability: %s", exc
                    )

                try:
                    self._sync_vllm_server_adapter(client)
                except (TypeError, ValueError) as exc:
                    if bool(fallback_to_full):
                        logger.warning(
                            "Adapter-only vLLM server sync failed; falling back to full sync for the remainder of the run. "
                            "Error: %s",
                            exc,
                        )
                        self._vllm_server_force_full_sync = True
                        self._sync_vllm_server_full_weights(client)
                    else:
                        raise
            else:
                self._sync_vllm_server_full_weights(client)

            # Keep local state consistent across ranks so the next call is stable.
            self._vllm_server_last_synced_step = step
            return

        # Multi-process learner (DDP): rank0-only full sync with strict barrier ordering.
        assert dist is not None and dist.is_initialized()

        # IMPORTANT: no early returns after this point without symmetric barriers.
        dist.barrier()
        try:
            if int(rank) == 0:
                client = self._ensure_vllm_server_client()
                self._ensure_vllm_server_communicator_rank0(client)
                self._sync_vllm_server_full_weights(client)
        finally:
            dist.barrier()

        # Keep local state consistent on all ranks.
        self._vllm_server_last_synced_step = step

    def _sync_vllm_server_full_weights(self, client: Any) -> None:
        """Full merged-weight sync to vLLM server (robust default)."""
        from contextlib import nullcontext

        try:
            from accelerate.utils import is_peft_model
        except (TypeError, ValueError):
            is_peft_model = None  # type: ignore[assignment]

        is_peft = (
            bool(is_peft_model(self.model)) if is_peft_model is not None else False
        )

        merge_cm = nullcontext()
        unmerge_cm = nullcontext()
        if is_peft:
            try:
                from swift.trainers.rlhf_trainer.utils import (
                    patch_lora_merge,
                    patch_lora_unmerge,
                )

                merge_cm = patch_lora_merge(self.model)
                unmerge_cm = patch_lora_unmerge(self.model)
            except (TypeError, ValueError):
                merge_cm = nullcontext()
                unmerge_cm = nullcontext()

        params = [p for _, p in self.model.named_parameters()]
        gather_if_zero3 = get_gather_if_zero3_context(self)

        with gather_if_zero3(params), merge_cm, torch.no_grad():
            merged = False
            try:
                if is_peft:
                    try:
                        self.model.merge_adapter()
                        merged = True
                    except (TypeError, ValueError) as exc:
                        raise RuntimeError(
                            "vLLM server full sync requires merging adapter weights from the training model. "
                            "Mitigations: ensure PEFT supports merge_adapter/unmerge_adapter, or use sync.mode=adapter."
                        ) from exc

                state_dict = self.model.state_dict()
                if is_peft:
                    prefix_removed = {
                        k.removeprefix("base_model.model."): v
                        for k, v in state_dict.items()
                    }
                    state_dict = {
                        k.replace(".base_layer", ""): v
                        for k, v in prefix_removed.items()
                    }
                    prefix = getattr(self.model, "prefix", None)
                    if isinstance(prefix, str) and prefix:
                        state_dict = {
                            k: v for k, v in state_dict.items() if prefix not in k
                        }
                    state_dict = {
                        k.replace("modules_to_save.default.", ""): v
                        for k, v in state_dict.items()
                        if "original_module" not in k
                    }
                    # LoRA already merged: do not send LoRA tensors.
                    state_dict = {
                        k: v for k, v in state_dict.items() if "lora_" not in k
                    }

                self._vllm_server_update_state_dict(client, state_dict)
            finally:
                if is_peft and merged:
                    with unmerge_cm:
                        self.model.unmerge_adapter()

        # Reset server prefix cache to avoid stale cached states.
        try:
            client.reset_prefix_cache()
        except (TypeError, ValueError) as exc:
            logger.warning(
                "Failed to reset vLLM server prefix cache after full sync: %s", exc
            )

    def _vllm_server_update_state_dict(
        self, client: Any, state_dict: Mapping[str, Any]
    ) -> None:
        """Bucket + broadcast a state_dict into the vLLM server via NCCL."""
        try:
            from swift.trainers.rlhf_trainer.utils import FlattenedTensorBucket
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "FlattenedTensorBucket is required for vLLM server sync"
            ) from exc

        bucket_size_mb = int(os.environ.get("SWIFT_UPDATE_WEIGHTS_BUCKET_SIZE", 512))
        bucket_size_bytes = int(bucket_size_mb) * 1024 * 1024

        bucket: List[Tuple[str, torch.Tensor]] = []
        bucket_bytes = 0

        def _flush_bucket() -> None:
            nonlocal bucket, bucket_bytes
            if not bucket:
                return
            b = FlattenedTensorBucket(named_tensors=bucket)
            client.update_flattened_params(b.get_metadata(), b.get_flattened_tensor())
            bucket = []
            bucket_bytes = 0

        for name, t in state_dict.items():
            if t is None or not isinstance(t, torch.Tensor):
                continue
            if t.numel() == 0:
                continue
            ten = t.detach()
            nbytes = int(ten.numel() * ten.element_size())
            if (
                bucket
                and bucket_size_bytes > 0
                and bucket_bytes + nbytes > bucket_size_bytes
            ):
                _flush_bucket()
            bucket.append((str(name), ten))
            bucket_bytes += nbytes

        _flush_bucket()

    def _sync_vllm_server_adapter(self, client: Any) -> None:
        """Adapter-only sync to vLLM server (requires vLLM LoRA)."""
        peft_cfg = getattr(self.model, "peft_config", None)
        if not isinstance(peft_cfg, Mapping) or not peft_cfg:
            raise RuntimeError(
                "vLLM server adapter sync requires a PEFT LoRA model (peft_config missing). "
                "Use sync.mode=full or disable server mode."
            )
        cfg0 = peft_cfg.get("default") or next(iter(peft_cfg.values()), None)

        try:
            peft_cfg_dict = asdict(cfg0)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            if hasattr(cfg0, "to_dict"):
                peft_cfg_dict = cfg0.to_dict()  # type: ignore[assignment]
            else:
                peft_cfg_dict = {}

        try:
            from peft.utils.save_and_load import get_peft_model_state_dict
        except (TypeError, ValueError) as exc:
            raise RuntimeError("peft is required for vLLM server adapter sync") from exc

        named = [(n, p) for n, p in self.model.named_parameters() if "lora_" in n]
        if not named:
            raise RuntimeError(
                "No LoRA parameters found on model, but adapter sync is enabled. "
                "Mitigations: set sync.mode=full or ensure LoRA/DoRA is enabled."
            )

        names = [n for n, _ in named]
        params = [p for _, p in named]
        gather_if_zero3 = get_gather_if_zero3_context(self)
        with gather_if_zero3(params):
            subset: Dict[str, torch.Tensor] = {}
            for n, p in zip(names, params):
                t = p.full_tensor() if hasattr(p, "full_tensor") else p
                subset[n] = t.detach()
            lora_params = get_peft_model_state_dict(self.model, subset)
            lora_params = {
                k: (v.full_tensor() if hasattr(v, "full_tensor") else v).detach()
                for k, v in lora_params.items()
            }

        try:
            from swift.trainers.rlhf_trainer.utils import FlattenedTensorBucket
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "FlattenedTensorBucket is required for vLLM server adapter sync"
            ) from exc

        bucket = FlattenedTensorBucket(named_tensors=list(lora_params.items()))
        client.update_adapter_flattened_param(
            peft_cfg_dict,
            bucket.get_metadata(),
            bucket.get_flattened_tensor(),
        )

        try:
            client.reset_prefix_cache()
        except (TypeError, ValueError) as exc:
            logger.warning(
                "Failed to reset vLLM server prefix cache after adapter sync: %s", exc
            )

    def _vllm_infer_tp_group(
        self, infer_requests: List[Dict[str, Any]], request_config: Any
    ) -> List[Any]:
        """TP-group gather/slice pattern for colocate vLLM rollouts (matches ms-swift behavior)."""
        engine = self._ensure_vllm_engine()
        tp = int(self._vllm_tp_size)
        # Optional micro-batching to reduce peak vLLM (vision) memory usage in colocate mode.
        vcfg = self._cfg("vllm", None)
        infer_batch_size: Optional[int] = None
        if isinstance(vcfg, Mapping):
            raw = vcfg.get("infer_batch_size", None)
            if raw is not None:
                try:
                    infer_batch_size = int(raw)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "rollout_matching.vllm.infer_batch_size must be an int"
                    ) from exc
                if infer_batch_size <= 0:
                    infer_batch_size = None

        def _infer_batched(reqs: List[Dict[str, Any]]) -> List[Any]:
            if not reqs:
                return []
            if infer_batch_size is None or infer_batch_size >= len(reqs):
                return engine.infer(reqs, request_config=request_config, use_tqdm=False)
            outs: List[Any] = []
            for i in range(0, len(reqs), infer_batch_size):
                outs.extend(
                    engine.infer(
                        reqs[i : i + infer_batch_size],
                        request_config=request_config,
                        use_tqdm=False,
                    )
                )
            return outs

        if tp <= 1:
            return _infer_batched(infer_requests)

        import torch.distributed as dist

        group = self._vllm_tp_group
        local_rank = int(dist.get_rank(group=group))
        local_len = int(len(infer_requests))
        all_lens: List[int] = [0 for _ in range(tp)]
        dist.all_gather_object(all_lens, local_len, group=group)
        start_idx = sum(int(x) for x in all_lens[:local_rank])
        end_idx = start_idx + local_len

        gathered: List[List[Dict[str, Any]]] = [[] for _ in range(tp)]
        dist.all_gather_object(gathered, infer_requests, group=group)
        flat: List[Dict[str, Any]] = [x for sub in gathered for x in sub]

        outs = _infer_batched(flat)
        return outs[start_idx:end_idx]

    @torch.no_grad()
    def _rollout_many_hf(
        self, samples: Sequence[Mapping[str, Any]]
    ) -> List[Tuple[List[int], str, str, List[int]]]:
        """HF (transformers) rollout backend with per-rank microbatching (padded batch)."""
        template = self.template
        tok = template.tokenizer
        decode_mode = str(self._cfg("decode_mode", "greedy")).lower()
        max_new_tokens = int(self._cfg("max_new_tokens", 512))
        num_beams = int(self._cfg("num_beams", 1))
        temperature, top_p, top_k = self._decoding_params()
        repetition_penalty = float(self._cfg("repetition_penalty", 1.0) or 1.0)
        if repetition_penalty <= 0:
            raise ValueError(
                "rollout_matching.repetition_penalty must be > 0"
            )

        # Build GenerationConfig from model defaults.
        gen_cfg = getattr(self.model, "generation_config", None)
        if gen_cfg is None:
            from transformers import GenerationConfig

            gen_cfg = GenerationConfig()
        gen_cfg = deepcopy(gen_cfg)
        gen_cfg.max_new_tokens = max_new_tokens
        self._apply_rollout_decoding_to_generation_config(
            gen_cfg=gen_cfg,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )
        if decode_mode == "beam":
            gen_cfg.num_beams = max(1, num_beams)
            gen_cfg.num_return_sequences = max(
                1, int(self._cfg("num_return_sequences", gen_cfg.num_beams))
            )
        else:
            gen_cfg.num_beams = 1
            gen_cfg.num_return_sequences = 1

        out: List[Tuple[List[int], str, str, List[int]]] = []
        mb = int(self._decode_batch_size())

        from swift.llm import to_device

        idx = 0
        while idx < len(samples):
            chunk = list(samples[idx : idx + mb])
            idx += len(chunk)

            with self._template_packing_disabled():
                with template.generate_context():
                    encoded_list = [
                        template.encode(dict(s), return_length=True) for s in chunk
                    ]
                    # IMPORTANT: keep generate_context active for collation so we left-pad for decoder-only
                    # generation (prevents incorrect generation + avoids HF "right-padding detected" warning).
                    batch = template.data_collator(encoded_list)

            batch = to_device(batch, self.model.device)
            input_ids_t = batch["input_ids"]
            attn = batch.get("attention_mask")
            if attn is None:
                pad_id = int(getattr(tok, "pad_token_id", 0) or 0)
                attn = (input_ids_t != pad_id).to(dtype=torch.long)

            # Prompt token ids for strict sanity checks (strip padding using attention_mask).
            prompt_ids_list: List[List[int]] = []
            for row_ids, row_mask in zip(input_ids_t, attn):
                ids = [
                    int(t)
                    for t, m in zip(
                        row_ids.detach().cpu().tolist(),
                        row_mask.detach().cpu().tolist(),
                    )
                    if int(m) == 1
                ]
                prompt_ids_list.append(ids)

            prompt_pad_len = int(input_ids_t.shape[1])
            model_inputs = {k: v for k, v in batch.items() if k != "labels"}
            model_inputs.pop("position_ids", None)
            model_inputs.pop("text_position_ids", None)

            logits_processor = None

            with unwrap_model_for_generation(
                self.model_wrapped,
                self.accelerator,
                gather_deepspeed3_params=getattr(
                    self.args, "ds3_gather_for_generation", False
                ),
            ) as unwrapped:
                unwrapped.eval()
                with self._template_packing_disabled():
                    with template.generate_context():
                        if (
                            getattr(self.model, "model_meta", None) is not None
                            and self.model.model_meta.is_multimodal
                        ):
                            _, model_inputs = template.pre_forward_hook(
                                unwrapped, None, model_inputs
                            )
                        model_inputs.pop("position_ids", None)
                        model_inputs.pop("text_position_ids", None)
                        gen_out = template.generate(
                            unwrapped,
                            **model_inputs,
                            generation_config=gen_cfg,
                            return_dict_in_generate=True,
                            logits_processor=logits_processor,
                        )
                unwrapped.train()

            sequences = gen_out.sequences
            if sequences.ndim != 2:
                raise ValueError("unexpected generate output shape")

            bsz = int(input_ids_t.shape[0])
            nret = int(getattr(gen_cfg, "num_return_sequences", 1) or 1)
            if nret < 1:
                nret = 1

            # Pick best sequence per sample for beam mode when possible.
            if (
                decode_mode == "beam"
                and nret > 1
                and hasattr(gen_out, "sequences_scores")
                and gen_out.sequences_scores is not None
            ):
                scores = gen_out.sequences_scores
                if scores.ndim != 1 or sequences.shape[0] != bsz * nret:
                    best_idx = torch.zeros(
                        (bsz,), dtype=torch.long, device=sequences.device
                    )
                else:
                    scores = scores.view(bsz, nret)
                    best_idx = torch.argmax(scores, dim=1)
                sequences = sequences.view(bsz, nret, -1)
                best_seqs = sequences[
                    torch.arange(bsz, device=sequences.device), best_idx
                ]
            else:
                # Default: first sequence per sample.
                if sequences.shape[0] == bsz * nret:
                    sequences = sequences.view(bsz, nret, -1)[:, 0, :]
                else:
                    sequences = sequences[:bsz, :]
                best_seqs = sequences

            for i in range(bsz):
                seq = best_seqs[i]
                resp_ids = seq[prompt_pad_len:].tolist()
                resp_ids = template.skip_stop_tokens(resp_ids, is_finished=True)
                text = template.decode(
                    resp_ids,
                    is_finished=True,
                    first_token=True,
                    clean_up_tokenization_spaces=False,
                )
                out.append((resp_ids, text, decode_mode, prompt_ids_list[i]))

        return out

    @torch.no_grad()
    def _rollout_many_vllm(
        self,
        samples: Sequence[Mapping[str, Any]],
        *,
        debug_samples: Optional[Sequence[Mapping[str, Any]]] = None,
        request_index_offset: int = 0,
    ) -> List[Tuple[List[int], str, str, List[int]]]:
        """vLLM rollout backend (colocate default, server optional)."""
        mode = self._vllm_mode()
        if mode == "server":
            return self._rollout_many_vllm_server(
                samples,
                debug_samples=debug_samples,
                request_index_offset=int(request_index_offset),
            )
        return self._rollout_many_vllm_colocate(samples)

    @torch.no_grad()
    def _rollout_many_vllm_colocate(
        self, samples: Sequence[Mapping[str, Any]]
    ) -> List[Tuple[List[int], str, str, List[int]]]:
        """vLLM colocate rollout backend (token ids)."""
        decode_mode = str(self._cfg("decode_mode", "greedy")).lower()
        if decode_mode != "greedy":
            raise ValueError(
                "vLLM rollout backend currently supports decode_mode=greedy only"
            )

        max_new_tokens = int(self._cfg("max_new_tokens", 512))
        temperature, top_p, top_k = self._decoding_params()
        repetition_penalty = float(self._cfg("repetition_penalty", 1.0) or 1.0)
        if repetition_penalty <= 0:
            raise ValueError(
                "rollout_matching.repetition_penalty must be > 0"
            )

        try:
            from swift.llm import RequestConfig
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "swift.llm.RequestConfig is required for vLLM rollouts"
            ) from exc

        request_config = RequestConfig(
            **self._rollout_vllm_request_config_kwargs(
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
            )
        )

        # Build infer requests using swift.llm.InferRequest (ms-swift stable contract).
        # NOTE: Do not pass dataset-level GT "objects" into vLLM infer; it may be interpreted as multimodal payload.
        try:
            from swift.llm import InferRequest
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "swift.llm.InferRequest is required for vLLM rollouts"
            ) from exc

        infer_requests: List[Any] = []
        for s in samples:
            msgs = s.get("messages")
            if not isinstance(msgs, list):
                raise ValueError(
                    "rollout-matching samples must contain messages (list)"
                )
            infer_requests.append(InferRequest(messages=msgs))

        # Ensure vLLM engine init + LoRA sync + infer are all covered by offload.
        with self._maybe_rollout_offload_context():
            self._sync_vllm_rollout_model_if_needed()
            outs: List[Any] = self._vllm_infer_tp_group(infer_requests, request_config)

        if len(outs) != len(infer_requests):
            raise RuntimeError("vLLM returned unexpected number of outputs")

        results: List[Tuple[List[int], str, str, List[int]]] = []
        for out in outs:
            if isinstance(out, Exception):
                results.append(([], "", decode_mode, []))
                continue
            text = ""
            token_ids: List[int] = []
            prompt_ids: List[int] = []
            try:
                text = str(out.choices[0].message.content or "")
                token_ids = [int(t) for t in (out.choices[0].token_ids or [])]
                prompt_ids = [
                    int(t) for t in (getattr(out, "prompt_token_ids", None) or [])
                ]
            except (TypeError, ValueError):
                text = ""
                token_ids = []
                prompt_ids = []
            results.append((token_ids, text, decode_mode, prompt_ids))
        return results

    @torch.no_grad()
    def _rollout_many_vllm_server(
        self,
        samples: Sequence[Mapping[str, Any]],
        *,
        debug_samples: Optional[Sequence[Mapping[str, Any]]] = None,
        request_index_offset: int = 0,
    ) -> List[Tuple[List[int], str, str, List[int]]]:
        """vLLM server rollout backend (token ids via ms-swift `swift rollout`)."""
        decode_mode = str(self._cfg("decode_mode", "greedy")).lower()
        if decode_mode != "greedy":
            raise ValueError(
                "vLLM server rollout backend currently supports decode_mode=greedy only"
            )

        n = int(len(samples))
        if n == 0:
            return []

        # Sync weights to server only when fresh rollouts are requested.
        #
        # IMPORTANT: vLLM server sync uses DDP collectives/barriers when learner world_size>1.
        # If rollouts are issued from a background thread (Stage2-AB pipelined mode),
        # set `_stage2_skip_vllm_server_sync=True` and perform sync on the main thread
        # at a safe boundary.
        if not bool(getattr(self, "_stage2_skip_vllm_server_sync", False)):
            self._sync_vllm_server_rollout_model_if_needed()

        max_new_tokens = int(self._cfg("max_new_tokens", 512))
        temperature, top_p, top_k = self._decoding_params()
        repetition_penalty = float(self._cfg("repetition_penalty", 1.0) or 1.0)
        if repetition_penalty <= 0:
            raise ValueError(
                "rollout_matching.repetition_penalty must be > 0"
            )

        try:
            from swift.llm import RequestConfig
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "swift.llm.RequestConfig is required for vLLM server rollouts"
            ) from exc

        # Base request config (per-server seed is set deterministically below).
        base_request_config = RequestConfig(
            **self._rollout_vllm_request_config_kwargs(
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
            )
        )
        base_request_config_dict = asdict(base_request_config)

        gs = int(getattr(getattr(self, "state", None), "global_step", 0) or 0)
        seed_base = int(self._derive_rollout_seed_base(global_step=gs))
        request_index_offset_i = max(0, int(request_index_offset))
        effective_seed_base = int(seed_base + request_index_offset_i)

        # Build JSON-serializable ms-swift RolloutInferRequest-compatible dicts.
        infer_requests: List[Dict[str, Any]] = []
        for s in samples:
            msgs = s.get("messages")
            if not isinstance(msgs, list):
                raise ValueError(
                    "rollout-matching samples must contain messages (list)"
                )
            try:
                msgs_json = json.loads(json.dumps(msgs))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "vLLM server mode requires JSON-serializable messages. "
                    "Ensure images are passed as strings (path/url/base64), not PIL objects."
                ) from exc

            req: Dict[str, Any] = {"messages": msgs_json}

            # Best-effort: include images list when present (common ms-swift multimodal contract).
            images_raw = s.get("images", None)
            if images_raw is None:
                img = s.get("image", None)
                if isinstance(img, str) and img:
                    images_raw = [img]
            if images_raw is not None:
                if isinstance(images_raw, str):
                    images = [images_raw]
                elif isinstance(images_raw, (list, tuple)):
                    images = list(images_raw)
                else:
                    raise ValueError(
                        "vLLM server mode expects sample['images'] to be a string or list of strings"
                    )
                if not all(isinstance(x, str) for x in images):
                    raise ValueError(
                        "vLLM server mode expects all image entries to be strings (path/url/base64)"
                    )
                req["images"] = images

            infer_requests.append(req)

        servers = self._vllm_server_specs()
        if not servers:
            raise ValueError("vLLM server mode requires a non-empty server list")

        _timeout_s, infer_timeout_s = self._vllm_server_timeouts()

        client = self._ensure_vllm_server_client()

        server_world_sizes = self._vllm_server_world_sizes()
        if len(server_world_sizes) != int(len(servers)):
            raise RuntimeError(
                "vLLM server world_size discovery returned unexpected length: "
                f"servers={int(len(servers))} world_sizes={server_world_sizes}"
            )

        learner_world = 1
        learner_rank = 0
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                learner_world = int(dist.get_world_size())
                learner_rank = int(dist.get_rank())
        except (TypeError, ValueError):
            learner_world = 1
            learner_rank = 0
        learner_world = max(1, int(learner_world))
        learner_rank = max(0, int(learner_rank))

        decode_cap = int(self._decode_batch_size())
        # Use the canonical per-rank chunk contract (with feasibility check) so
        # server mode behaves identically across all callers.
        per_rank_chunk = int(self._rollout_decode_batch_size_per_rank())

        per_server_rank_caps = _per_server_rank_request_caps(
            per_rank_chunk_size=int(per_rank_chunk),
            server_world_sizes=[int(x) for x in server_world_sizes],
            learner_world_size=int(learner_world),
            learner_rank=int(learner_rank),
        )
        round_cap_total = int(sum(int(x) for x in per_server_rank_caps))
        if int(round_cap_total) != int(per_rank_chunk):
            raise RuntimeError(
                "internal per-rank rollout cap mismatch: "
                f"per_rank_chunk={int(per_rank_chunk)} round_cap_total={int(round_cap_total)} "
                f"learner_rank={int(learner_rank)} learner_world_size={int(learner_world)} "
                f"server_world_sizes={list(int(x) for x in server_world_sizes)}"
            )

        # Log reproducibility metadata once per optimizer step (E-steps only).
        if gs != int(getattr(self, "_vllm_server_last_logged_step", -1)):
            seed_plan: List[Dict[str, Any]] = []
            if int(len(infer_requests)) > 0 and round_cap_total > 0:
                cursor = 0
                round_idx = 0
                while cursor < int(len(infer_requests)):
                    remaining = int(len(infer_requests) - cursor)
                    round_budget = int(min(remaining, round_cap_total))
                    counts = _allocate_weighted_counts_with_caps(
                        int(round_budget), per_server_rank_caps
                    )
                    offset = int(cursor)
                    for i, cnt in enumerate(counts):
                        if int(cnt) <= 0:
                            continue
                        start = int(offset)
                        end = int(offset + int(cnt))
                        seed_plan.append(
                            {
                                "round": int(round_idx),
                                "server_idx": int(i),
                                "base_url": str(servers[i].get("base_url", "")),
                                "start": int(start),
                                "end": int(end),
                                "cap_for_rank": int(per_server_rank_caps[i]),
                                # Effective per-server-call seed used for RequestConfig.seed:
                                # seed = rollout_seed_base + chunk_start
                                "seed": int((effective_seed_base + int(start)) & 0x7FFFFFFF),
                            }
                        )
                        offset = int(end)
                    cursor = int(cursor + round_budget)
                    round_idx = int(round_idx + 1)

            logger.info(
                "vLLM server rollout metadata: servers=%s sync_mode=%s request_n=%s rollout_seed_base=%s request_index_offset=%s effective_seed_base=%s decode_batch_size_cap=%s per_rank_chunk=%s learner_world_size=%s learner_rank=%s server_world_sizes=%s per_server_rank_caps=%s round_cap_total=%s seed_plan=%s",
                servers,
                self._effective_vllm_server_sync_mode(),
                int(len(infer_requests)),
                int(seed_base),
                int(request_index_offset_i),
                int(effective_seed_base),
                int(decode_cap),
                int(per_rank_chunk),
                int(learner_world),
                int(learner_rank),
                [int(x) for x in server_world_sizes],
                [int(x) for x in per_server_rank_caps],
                int(round_cap_total),
                seed_plan,
            )
            self._vllm_server_last_logged_step = int(gs)

        results: List[Optional[Tuple[List[int], str, str, List[int]]]] = [None] * len(
            infer_requests
        )

        def _parse_output(raw: Any) -> Tuple[List[int], str, List[int]]:

            if isinstance(raw, dict):
                if isinstance(raw.get("response"), dict):
                    raw = raw.get("response")

            if not isinstance(raw, dict):
                raise RuntimeError("vLLM server returned a non-dict output")
            if raw.get("object") == "error":
                raise RuntimeError(str(raw.get("message") or raw))

            prompt_ids_raw = raw.get("prompt_token_ids")
            if not isinstance(prompt_ids_raw, list) or not prompt_ids_raw:
                raise RuntimeError(
                    "vLLM server response missing prompt_token_ids; ensure request_config.return_details=true"
                )
            prompt_ids = [int(t) for t in prompt_ids_raw]

            choices = raw.get("choices")
            if not isinstance(choices, list) or not choices:
                raise RuntimeError("vLLM server response missing choices")
            ch0 = choices[0]
            if not isinstance(ch0, dict):
                raise RuntimeError("vLLM server response choice is not a dict")

            msg = ch0.get("message")
            if not isinstance(msg, dict):
                msg = {}
            text = str(msg.get("content") or "")

            token_ids_raw = ch0.get("token_ids")
            if not isinstance(token_ids_raw, list):
                raise RuntimeError(
                    "vLLM server response missing token_ids; ensure request_config.return_details=true"
                )
            token_ids = [int(t) for t in token_ids_raw]
            return token_ids, text, prompt_ids

        def _infer_on_server(server_idx: int, start: int, end: int) -> None:
            if start >= end:
                return
            base_url = str(servers[server_idx]["base_url"]).rstrip("/")

            # Derive per-server-call seed so per-request seeds are stable.
            req_cfg = dict(base_request_config_dict)
            req_cfg["seed"] = int((effective_seed_base + int(start)) & 0x7FFFFFFF)

            payload = {
                "infer_requests": infer_requests[start:end],
                "request_config": req_cfg,
                "metrics": None,
                "template": None,
                "use_tqdm": None,
                "adapter_request": None,
            }

            url = f"{base_url}/infer/"
            session = client.sessions[server_idx]
            req_timeout: Optional[Tuple[float, float]]
            if infer_timeout_s is None:
                req_timeout = None
            else:
                req_timeout_s = float(infer_timeout_s)
                # Use a (connect, read) timeout tuple to prevent indefinite hangs on broken keep-alive sockets.
                req_timeout = (min(10.0, req_timeout_s), req_timeout_s)
            try:
                with self._vllm_server_infer_guard():
                    resp = session.post(url, json=payload, timeout=req_timeout)
            except (TypeError, ValueError) as exc:
                # Retry once with a fresh session. This helps when the server was idle for A steps
                # (AAB schedules) and the underlying keep-alive connection was dropped.
                try:
                    import requests

                    client.sessions[server_idx] = requests.Session()
                    session = client.sessions[server_idx]
                    with self._vllm_server_infer_guard():
                        resp = session.post(url, json=payload, timeout=req_timeout)
                except (TypeError, ValueError) as exc2:
                    # If this was a batched request, fall back to smaller batches. This is a
                    # common failure mode when a few samples hit max_new_tokens and the read
                    # timeout is exceeded.
                    if int(end - start) > 1:
                        mid = int((start + end) // 2)
                        logger.warning(
                            "vLLM server infer request failed; splitting batch: url=%s start=%s end=%s mid=%s exc=%r",
                            url,
                            int(start),
                            int(end),
                            int(mid),
                            exc2,
                        )
                        _infer_on_server(int(server_idx), int(start), int(mid))
                        _infer_on_server(int(server_idx), int(mid), int(end))
                        return

                    raise RuntimeError(
                        f"vLLM server infer request failed after retry: url={url} exc={exc!r}"
                    ) from exc2

            if int(getattr(resp, "status_code", 0) or 0) != 200:
                raise RuntimeError(
                    f"vLLM server infer failed: url={url} status={getattr(resp, 'status_code', None)} body={getattr(resp, 'text', '')}"
                )

            data = resp.json()
            if not isinstance(data, list):
                raise RuntimeError("vLLM server returned non-list JSON")
            if len(data) != int(end - start):
                raise RuntimeError(
                    "vLLM server returned unexpected number of outputs: "
                    f"expected={int(end - start)} got={len(data)}"
                )

            for j, raw_out in enumerate(data):
                token_ids, text, prompt_ids = _parse_output(raw_out)
                idx = int(start + j)
                results[idx] = (token_ids, text, decode_mode, prompt_ids)

        # Parallelize by server within each round.
        # Each round enforces strict per-server caps for this learner rank.
        from concurrent.futures import ThreadPoolExecutor

        cursor = 0
        while cursor < int(len(infer_requests)):
            remaining = int(len(infer_requests) - cursor)
            round_budget = int(min(remaining, max(1, int(round_cap_total))))
            counts = _allocate_weighted_counts_with_caps(
                int(round_budget), per_server_rank_caps
            )

            round_slices: List[Tuple[int, int, int]] = []
            offset = int(cursor)
            for i, cnt in enumerate(counts):
                if int(cnt) <= 0:
                    continue
                start = int(offset)
                end = int(offset + int(cnt))
                round_slices.append((int(i), int(start), int(end)))
                offset = int(end)

            if not round_slices:
                raise RuntimeError(
                    "vLLM server rollout produced an empty dispatch round under non-empty workload: "
                    f"cursor={int(cursor)} remaining={int(remaining)} per_server_rank_caps={per_server_rank_caps}"
                )

            with ThreadPoolExecutor(max_workers=int(len(round_slices))) as ex:
                futs = [
                    ex.submit(_infer_on_server, int(i), int(start), int(end))
                    for i, start, end in round_slices
                ]
                for f in futs:
                    f.result()

            cursor = int(cursor + round_budget)

        out: List[Tuple[List[int], str, str, List[int]]] = []
        for r in results:
            if r is None:
                raise RuntimeError(
                    "vLLM server failed to produce outputs for all requests"
                )
            out.append(r)

        self._maybe_debug_dump_vllm_server_rollouts(
            global_step=gs,
            seed_base=effective_seed_base,
            infer_requests=infer_requests,
            outputs=out,
            samples=debug_samples if debug_samples is not None else samples,
        )

        return out

    @torch.no_grad()
    def _rollout_many(
        self, samples: Sequence[Mapping[str, Any]]
    ) -> List[Tuple[List[int], str, str, List[int]]]:
        backend = self._rollout_backend()

        # vLLM backends receive raw OpenAI-style messages; ensure the resolved
        # system prompt is present so output formatting is stable.
        system_prompt: str | None = None
        if backend == "vllm":
            sp = getattr(self.template, "system", None)
            if isinstance(sp, str) and sp.strip():
                system_prompt = sp
            else:
                # Fallback: ms-swift templates may not expose the resolved system prompt
                # as `template.system` in all execution contexts. Use CoordExp's
                # canonical dense system prompt to stabilize server-mode rollouts.
                try:
                    from src.config.prompts import SYSTEM_PROMPT_SORTED_TOKENS

                    system_prompt = str(SYSTEM_PROMPT_SORTED_TOKENS)
                except (TypeError, ValueError):
                    system_prompt = None

        # IMPORTANT: generate from a prompt that ends with a user turn.
        # Many datasets include a teacher-forced assistant answer in `messages` for training.
        # For rollouts, we must drop any trailing assistant turns.
        samples_for_rollout: List[Mapping[str, Any]] = []
        for s in samples:
            msgs = s.get("messages")
            if isinstance(msgs, list):
                modified = False

                trimmed = _strip_trailing_assistant_turns_for_rollout(msgs)
                if len(trimmed) != len(msgs):
                    modified = True
                    msgs_out: List[Any] = trimmed
                else:
                    msgs_out = msgs

                if backend == "vllm" and system_prompt is not None:
                    msgs_sys = _ensure_system_prompt_message(msgs_out, system_prompt)
                    if len(msgs_sys) != len(msgs_out):
                        modified = True
                        msgs_out = msgs_sys

                if modified:
                    s2 = dict(s)
                    s2["messages"] = msgs_out
                    samples_for_rollout.append(s2)
                else:
                    samples_for_rollout.append(s)
            else:
                samples_for_rollout.append(s)

        if backend == "hf":
            return self._rollout_many_hf(samples_for_rollout)

        if backend == "vllm":
            mode = self._vllm_mode()
            if mode == "server":
                # Enforce the per-rank rollout request cap centrally so all callers
                # (Stage2-AB + evaluator) obey decode_batch_size topology constraints.
                chunk_size = max(1, int(self._rollout_decode_batch_size_per_rank()))
                if int(len(samples_for_rollout)) > 0:
                    chunk_size = min(chunk_size, int(len(samples_for_rollout)))

                out: List[Tuple[List[int], str, str, List[int]]] = []

                for off in range(0, int(len(samples_for_rollout)), int(chunk_size)):
                    chunk_samples = samples_for_rollout[
                        int(off) : int(off + chunk_size)
                    ]
                    chunk_debug_samples = samples[int(off) : int(off + chunk_size)]
                    chunk_out = self._rollout_many_vllm(
                        chunk_samples,
                        debug_samples=chunk_debug_samples,
                        request_index_offset=int(off),
                    )
                    out.extend(chunk_out)
            else:
                out = self._rollout_many_vllm(
                    samples_for_rollout,
                    debug_samples=samples,
                )
            return out

        raise AssertionError("unreachable")

    def _append_post_rollout_segments(
        self, segments: Sequence[Tuple[Dict[str, Any], Dict[str, Any], int]]
    ) -> None:
        """Append newly produced post-rollout segments to the rank-local buffer.

        Safety: fail-fast if any single segment exceeds packing_length, at insertion time.
        """
        packing_length = int(self._packing_length())
        if packing_length <= 0:
            raise ValueError("packing is enabled but packing_length is invalid")

        seg_list = segments if isinstance(segments, list) else list(segments)

        for _, _, seg_len in seg_list:
            sl = int(seg_len)
            if sl > packing_length:
                raise ValueError(
                    f"post-rollout packing cannot fit a single segment: encoded_len={sl} > packing_length={packing_length}. "
                    "Mitigations: increase global_max_length/template.max_length, reduce max_new_tokens, or disable packing."
                )

        cap = int(self._packing_buffer_cap())
        if cap > 0:
            new_size = len(self._post_rollout_segments) + len(seg_list)
            if new_size > cap:
                raise ValueError(
                    "post-rollout packing buffer overflow: "
                    f"buffer_size={new_size} > packing_buffer={cap}. "
                    "Mitigations: reduce per_device_train_batch_size, increase training.packing_buffer, "
                    "or enable multi-pack-per-step in a future change."
                )

        self._post_rollout_segments.extend(seg_list)

    @staticmethod
    def _select_post_rollout_segment_indices(
        encoded_lens: Sequence[int],
        packing_length: int,
    ) -> List[int]:
        """Select segment indices for one packed forward pass.

        Input `encoded_lens` is in insertion order (index 0 is oldest). Output indices
        are in insertion order, MUST include the oldest segment, and total length MUST
        be <= packing_length.

        Selection is:
          - FIFO-greedy baseline (current behavior), and
          - ms-swift-like constant-volume binpacking candidate constrained to include oldest,
        with a strict "never worse than FIFO" fallback rule.
        """
        packing_length = int(packing_length)
        if packing_length <= 0:
            raise ValueError("packing_length must be positive")
        if not encoded_lens:
            return []

        try:
            import binpacking
        except ImportError as exc:
            raise ImportError(
                "binpacking is required for stage-2 post-rollout packing selection; "
                "install `binpacking` or disable `training.packing`."
            ) from exc

        lens = [int(x) for x in encoded_lens]
        oldest_len = int(lens[0])
        if oldest_len > packing_length:
            raise ValueError(
                f"post-rollout packing cannot fit a single segment: encoded_len={oldest_len} > packing_length={packing_length}. "
                "Mitigations: increase global_max_length/template.max_length, reduce max_new_tokens, or disable packing."
            )
        if oldest_len <= 0:
            raise ValueError("oldest post-rollout segment has non-positive encoded_len")

        # Defensive invariant check (insertion should already enforce this).
        for sl in lens:
            if int(sl) > packing_length:
                raise ValueError(
                    f"post-rollout packing buffer contains an oversized segment: encoded_len={int(sl)} > packing_length={packing_length}."
                )

        # 1) FIFO-greedy baseline (current behavior): always include oldest, then scan.
        baseline: List[int] = [0]
        used = int(oldest_len)
        for i in range(1, len(lens)):
            sl = int(lens[i])
            if sl <= 0:
                continue
            if used + sl <= packing_length:
                baseline.append(int(i))
                used += sl
        baseline_total = int(used)

        # 2) Binpacking candidate under the residual cap (oldest is pinned).
        cap_rem = int(packing_length - oldest_len)
        if cap_rem <= 0:
            return baseline

        items: List[Tuple[int, int]] = []
        for i in range(1, len(lens)):
            sl = int(lens[i])
            if sl <= 0:
                continue
            if sl <= cap_rem:
                items.append((int(i), int(sl)))

        bins = (
            binpacking.to_constant_volume(items, cap_rem, weight_pos=1) if items else []
        )

        best_rest: List[int] = []
        best_key: Optional[Tuple[int, int, Tuple[int, ...]]] = None
        for b in bins:
            rest = sorted(int(idx) for idx, _ in b)
            total = int(sum(int(lens[i]) for i in rest))
            key = (-total, len(rest), tuple(rest))
            if best_key is None or key < best_key:
                best_key = key
                best_rest = rest

        candidate: List[int] = [0] + best_rest
        candidate.sort()
        candidate_total = int(sum(int(lens[i]) for i in candidate))
        if candidate_total > packing_length:
            raise AssertionError(
                "post-rollout packing selection overflowed packing_length"
            )

        # Baseline-fallback rule: only switch if binpacking strictly improves total length.
        if candidate_total > baseline_total:
            return candidate
        return baseline

    def _pop_post_rollout_pack(
        self,
    ) -> Tuple[List[Tuple[Dict[str, Any], Dict[str, Any], int]], Dict[str, float]]:
        """Select and remove a subset of buffered segments for one packed forward pass (carry-only).

        Returns (selected_segments, packing_metrics). Packing metrics are emitted into the main
        training log line (merged with loss) to avoid per-micro-batch log spam.
        """
        packing_length = int(self._packing_length())
        if packing_length <= 0:
            raise ValueError("packing is enabled but packing_length is invalid")
        if not self._post_rollout_segments:
            raise ValueError(
                "packing is enabled but no post-rollout segments are available"
            )

        encoded_lens = [int(seg_len) for _, _, seg_len in self._post_rollout_segments]
        selected_idx = self._select_post_rollout_segment_indices(
            encoded_lens, packing_length
        )
        if not selected_idx:
            raise AssertionError("post-rollout packing selected an empty segment set")
        total_len = int(sum(encoded_lens[i] for i in selected_idx))

        selected = [self._post_rollout_segments[i] for i in selected_idx]
        for i in reversed(selected_idx):
            self._post_rollout_segments.pop(i)

        fill = float(total_len) / float(packing_length) if packing_length > 0 else 0.0
        target = float(self._packing_min_fill_ratio())

        # Expose last-pack stats for adaptive raw batching.
        try:
            self._rm_last_pack_fill = float(fill)
            self._rm_last_pack_segments = int(len(selected))
            self._rm_last_pack_buffer_after = int(len(self._post_rollout_segments))
        except (TypeError, ValueError):
            raise

        if fill < target:
            logger.warning(
                "post-rollout packing underfilled: fill=%.3f target=%.3f segments=%s buffer=%s",
                fill,
                target,
                len(selected),
                len(self._post_rollout_segments),
            )

        pack_metrics: Dict[str, float] = {
            "packing/post_rollout_fill": float(fill),
            "packing/post_rollout_selected_total_len": float(total_len),
            "packing/post_rollout_segments": float(len(selected)),
            "packing/post_rollout_buffer": float(len(self._post_rollout_segments)),
        }

        # Update a running average segment length estimate for adaptive raw batching.
        try:
            seg_count = int(len(selected))
            if seg_count > 0:
                avg = float(total_len) / float(seg_count)
                prev = float(getattr(self, "_rm_avg_segment_len", 0.0) or 0.0)

                # If we're underfilled *and* the buffer emptied, we were supply-limited.
                # Update aggressively downward so the next raw batch is larger.
                supply_limited = bool(
                    fill < target and len(self._post_rollout_segments) == 0
                )

                alpha = 0.2
                if supply_limited:
                    alpha = 0.5

                ema = (
                    float(avg)
                    if prev <= 0
                    else float((1.0 - alpha) * prev + alpha * avg)
                )
                if supply_limited:
                    ema = min(float(ema), float(avg))

                self._rm_avg_segment_len = float(ema)
                pack_metrics["packing/avg_segment_len_last"] = float(avg)
                pack_metrics["packing/avg_segment_len_ema"] = float(ema)
        except (TypeError, ValueError):
            raise

        return selected, pack_metrics

    def _prepare_batch_inputs(
        self,
        inputs: List[Mapping[str, Any]],
        _segments_only: bool = False,
    ) -> Any:
        template = self.template
        tok = template.tokenizer

        coord_token_ids = self._get_coord_token_ids()
        coord_id_set = set(int(i) for i in coord_token_ids if int(i) >= 0)
        coord_id_to_bin = self._coord_id_map()

        gate_thr = float(self._cfg("maskiou_gate", 0.3))
        top_k = int(self._cfg("candidate_top_k", 10))
        mask_res = int(self._cfg("maskiou_resolution", 256))

        fp_cost = float(self._cfg("fp_cost", 1.0))
        fn_cost = float(self._cfg("fn_cost", 1.0))

        ot_eps = float(self._cfg("ot_epsilon", 10.0))
        ot_iters = int(self._cfg("ot_iters", 30))
        ot_cost = str(self._cfg("ot_cost", "l2")).lower()
        ot_cost_kind: Literal["l1", "l2"] = "l1" if ot_cost == "l1" else "l2"

        packing_enabled = self._packing_enabled()
        if packing_enabled and not self._packing_drop_last():
            raise ValueError(
                "stage_2 post-rollout packing uses carry-only mode and requires training.packing_drop_last: true"
            )
        if packing_enabled and self._packing_buffer_cap() <= 0:
            raise ValueError(
                "training.packing_buffer must be a positive int when packing is enabled"
            )
        if packing_enabled and self._packing_length() <= 0:
            raise ValueError(
                "packing is enabled but no valid packing_length/template.max_length is set (check global_max_length)"
            )

        # Optional qualitative monitoring dumps: rollout vs GT vs training target.
        gs = int(getattr(getattr(self, "state", None), "global_step", 0) or 0)
        do_dump = False
        dump_cfg = self._monitor_dump_cfg()
        dump_max_samples = 0
        dump_max_chars = 0
        dump_samples: List[Dict[str, Any]] = []
        if self._should_monitor_dump(global_step=gs):
            do_dump = True
            dump_max_samples = max(1, int(dump_cfg.get("max_samples", 1) or 1))
            dump_max_chars = max(0, int(dump_cfg.get("max_text_chars", 4000) or 4000))
            # Mark early to avoid duplicate dumps in the same optimizer step.
            self._monitor_dump_last_step = int(gs)

        # Phase A: rollout generation (no grad, un-packed; batched via backend).
        t_gen0 = time.perf_counter()
        rollout_results = self._rollout_many(inputs)
        if len(rollout_results) != len(inputs):
            raise RuntimeError("rollout backend returned unexpected number of results")
        t_gen_s = time.perf_counter() - t_gen0

        # Phase B: strict parse/match/build targets, then teacher-forced encode per sample.
        encoded_batch: List[Dict[str, Any]] = []
        meta_unpacked: List[Dict[str, Any]] = []
        segments: List[Tuple[Dict[str, Any], Dict[str, Any], int]] = []

        t_parse_match_s = 0.0
        t_encode_s = 0.0
        rollout_lens: List[int] = []

        for sample, (resp_ids, resp_text, decode_mode, prompt_ids) in zip(
            inputs, rollout_results
        ):
            if "messages" not in sample:
                raise ValueError(
                    "rollout-matching requires 'messages' in dataset samples"
                )

            # 1) Strict token-aligned parsing + suffix-only prefix trimming
            t_pm0 = time.perf_counter()
            parse = parse_rollout_for_matching(
                tokenizer=tok,
                response_token_ids=resp_ids,
                object_field_order=self._object_field_order(),
            )
            rollout_lens.append(int(len(parse.response_token_ids)))
            self._maybe_debug_dump_parse_failure(
                sample=sample,
                response_text=resp_text,
                prefix_text=parse.prefix_text,
                dropped_invalid=int(parse.dropped_invalid),
                dropped_ambiguous=int(parse.dropped_ambiguous),
                truncated=bool(parse.truncated),
                decode_mode=str(decode_mode),
            )

            # 2) Extract predicted objects (valid only) and map coord tokens -> bins
            preds: List[GTObject] = []
            pred_meta: List[ParsedPredObject] = []
            for pobj in parse.valid_objects:
                pts = _points_from_coord_tokens(
                    response_token_ids=parse.response_token_ids,
                    coord_token_indices=pobj.coord_token_indices,
                    coord_id_to_bin=coord_id_to_bin,
                )
                if pts is None:
                    continue
                # For matching, keep geometry in norm1000.
                preds.append(
                    GTObject(
                        index=int(pobj.index),
                        geom_type=pobj.geom_type,
                        points_norm1000=pts,
                        desc="",
                    )
                )
                pred_meta.append(pobj)

            # 3) Extract GT objects and match
            gts = _extract_gt_objects(sample)
            match = hungarian_match_maskiou(
                preds=preds,
                gts=gts,
                top_k=top_k,
                gate_threshold=gate_thr,
                mask_resolution=mask_res,
                fp_cost=fp_cost,
                fn_cost=fn_cost,
            )

            # 3.0) Optional desc monitor (metrics only; does not affect loss).
            desc_cfg = self._desc_monitor_cfg()
            desc_monitor_ran = False
            desc_pairs_total = 0
            desc_exact_ok = 0
            desc_sem_ok = 0
            desc_sem_sim_sum = 0.0
            desc_sem_sim_count = 0
            desc_sem_enabled = 0
            if isinstance(desc_cfg, Mapping) and bool(desc_cfg.get("enabled", False)):
                every = int(desc_cfg.get("every_steps", 0) or 0)
                if every <= 0:
                    every = int(
                        getattr(getattr(self, "args", None), "logging_steps", 0) or 0
                    )
                if every <= 0:
                    every = 1
                if int(gs) % int(every) == 0:
                    desc_monitor_ran = True
                    max_pairs = int(desc_cfg.get("max_pairs", 64) or 64)
                    thr = float(desc_cfg.get("semantic_threshold", 0.6) or 0.6)
                    mode = (
                        str(desc_cfg.get("mode", "semantic") or "semantic")
                        .strip()
                        .lower()
                    )

                    try:
                        from src.metrics.semantic_desc import normalize_desc
                    except (TypeError, ValueError):
                        normalize_desc = None  # type: ignore[assignment]

                    pairs = list(match.matched_pairs)
                    if max_pairs > 0 and len(pairs) > max_pairs:
                        pairs = pairs[:max_pairs]

                    norm_pairs: List[Tuple[str, str, bool]] = []
                    uniq: set[str] = set()
                    for pred_i, gt_i in pairs:
                        if pred_i < 0 or pred_i >= len(pred_meta):
                            continue
                        if gt_i < 0 or gt_i >= len(gts):
                            continue
                        pred_desc_raw = str(
                            getattr(pred_meta[pred_i], "desc", "") or ""
                        )
                        gt_desc_raw = str(getattr(gts[gt_i], "desc", "") or "")
                        if normalize_desc is None:
                            p = pred_desc_raw.strip().lower()
                            g = gt_desc_raw.strip().lower()
                        else:
                            p = normalize_desc(pred_desc_raw)
                            g = normalize_desc(gt_desc_raw)
                        exact_ok = bool(p) and (p == g)
                        if exact_ok:
                            desc_exact_ok += 1
                        if p and g:
                            norm_pairs.append((p, g, bool(exact_ok)))
                            uniq.add(p)
                            uniq.add(g)

                    desc_pairs_total = int(len(norm_pairs))

                    if mode in {"semantic", "both"} and desc_pairs_total > 0:
                        enc = None
                        try:
                            enc = self._get_desc_semantic_encoder(desc_cfg)
                        except (TypeError, ValueError):
                            enc = None

                        if enc is not None:
                            # Best-effort: if model load fails (missing cache/network), skip semantics.
                            try:
                                emb = enc.encode_norm_texts(sorted(uniq))
                            except (TypeError, ValueError):
                                emb = {}
                                enc = None

                        if enc is not None:
                            desc_sem_enabled = 1
                            for p, g, exact_ok in norm_pairs:
                                pv = emb.get(p)
                                gv = emb.get(g)
                                if pv is None or gv is None:
                                    ok = bool(exact_ok)
                                    sim = None
                                else:
                                    sim = float(np.dot(pv, gv))
                                    ok = bool(exact_ok or sim >= thr)
                                if ok:
                                    desc_sem_ok += 1
                                if sim is not None:
                                    desc_sem_sim_sum += float(sim)
                                    desc_sem_sim_count += 1

            # 3.1) Build self-context supervision targets for matched pairs.
            # If target construction fails, exclude that object from supervision and treat GT as FN.
            prefix_pos: List[int] = []
            prefix_target_bins: List[int] = []
            excluded = 0

            matched_gt_for_supervision: set[int] = set()
            for pred_i, gt_i in match.matched_pairs:
                if pred_i < 0 or pred_i >= len(preds) or pred_i >= len(pred_meta):
                    continue
                if gt_i < 0 or gt_i >= len(gts):
                    continue
                pobj = pred_meta[pred_i]
                pred_obj = preds[pred_i]
                gt_obj = gts[gt_i]
                try:
                    targets = self._build_prefix_targets(
                        pred_obj=pred_obj,
                        gt_obj=gt_obj,
                        pred_coord_indices=pobj.coord_token_indices,
                        ot_epsilon=ot_eps,
                        ot_iters=ot_iters,
                        ot_cost=ot_cost_kind,
                    )
                except (TypeError, ValueError):
                    targets = None
                if targets is None or len(targets) != len(pobj.coord_token_indices):
                    excluded += 1
                    continue
                matched_gt_for_supervision.add(gt_i)
                for local_idx, tbin in zip(pobj.coord_token_indices, targets):
                    if local_idx < 0 or local_idx >= len(parse.prefix_token_ids):
                        continue
                    prefix_pos.append(int(local_idx))
                    prefix_target_bins.append(int(tbin))

            fn_gt_indices_final = [
                i for i in range(len(gts)) if i not in matched_gt_for_supervision
            ]
            fn_objs = [gts[i] for i in fn_gt_indices_final]

            # 4) Serialize append fragment (mandatory FN append) and build Y_train ids
            append_text = _serialize_append_fragment(
                fn_objects=fn_objs,
                prefix_text=parse.prefix_text,
                object_field_order=self._object_field_order(),
            )
            append_ids = tok.encode(append_text, add_special_tokens=False)
            # Ignore desc value tokens in the appended tail for CE (GT desc can be noisy).
            tail_ignore_pos = _find_desc_value_token_positions(
                tokenizer=tok, token_ids=append_ids
            )
            y_train_ids = list(parse.prefix_token_ids) + [int(t) for t in append_ids]
            t_parse_match_s += time.perf_counter() - t_pm0

            # 5) Teacher-forced encoding using the exact token ids (no re-tokenization)
            t_enc0 = time.perf_counter()
            data_for_encode = dict(sample)
            # Deepcopy messages to avoid in-place mutations across dataloader workers.
            messages = json.loads(json.dumps(sample["messages"]))
            has_assistant = False
            try:
                for m in messages:
                    if isinstance(m, dict) and m.get("role") == "assistant":
                        has_assistant = True
                        break
            except (TypeError, ValueError):
                has_assistant = False

            if has_assistant:
                data_for_encode["messages"] = replace_assistant_response_with_ids(
                    messages, y_train_ids
                )
            else:
                # Some datasets keep only the user turn in `messages` and store GT separately.
                # Stage_2 needs an assistant turn to inject token ids for teacher forcing.
                data_for_encode["messages"] = list(messages) + [
                    {"role": "assistant", "content": y_train_ids}
                ]
            with self._template_train_mode():
                encoded = template.encode(data_for_encode, return_length=True)
            t_encode_s += time.perf_counter() - t_enc0

            encoded_len = self._extract_encoded_len(encoded)
            if int(encoded_len) <= int(len(prompt_ids)):
                raise ValueError(
                    "teacher-forced encode produced no assistant span: "
                    f"prompt_len={int(len(prompt_ids))} encoded_len={int(encoded_len)} train_len={int(len(y_train_ids))} "
                    f"sample_id={sample.get('sample_id')} base_idx={sample.get('base_idx')}. "
                    "This indicates the assistant turn was not injected or got truncated; check max_length/truncation settings."
                )

            if do_dump and len(dump_samples) < dump_max_samples:
                try:
                    # Build a compact, human-readable record (strings are clipped).
                    gt_objs_dump = [
                        {
                            "index": int(o.index),
                            "geom_type": str(o.geom_type),
                            "points_norm1000": list(o.points_norm1000),
                            "desc": str(o.desc),
                        }
                        for o in gts
                    ]
                    pred_objs_dump = [
                        {
                            "key": str(pred_meta[i].key) if i < len(pred_meta) else "",
                            "index": int(o.index),
                            "geom_type": str(o.geom_type),
                            "points_norm1000": list(o.points_norm1000),
                            "desc": str(getattr(pred_meta[i], "desc", "") or "")
                            if i < len(pred_meta)
                            else "",
                        }
                        for i, o in enumerate(preds)
                    ]

                    pair_details: List[Dict[str, Any]] = []
                    for pred_i, gt_i in match.matched_pairs:
                        if pred_i < 0 or pred_i >= len(preds):
                            continue
                        if gt_i < 0 or gt_i >= len(gts):
                            continue
                        iou = _mask_iou_norm1000(
                            pred_kind=preds[pred_i].geom_type,
                            pred_points=preds[pred_i].points_norm1000,
                            gt_kind=gts[gt_i].geom_type,
                            gt_points=gts[gt_i].points_norm1000,
                            resolution=mask_res,
                        )
                        pair_details.append(
                            {
                                "pred_i": int(pred_i),
                                "gt_i": int(gt_i),
                                "mask_iou": float(iou),
                                "pred_index": int(preds[pred_i].index),
                                "gt_index": int(gts[gt_i].index),
                                "pred_desc": str(
                                    getattr(pred_meta[pred_i], "desc", "") or ""
                                )
                                if pred_i < len(pred_meta)
                                else "",
                                "gt_desc": str(gts[gt_i].desc),
                            }
                        )

                    # Per-sample derived quality stats.
                    gt_n = float(len(gts))
                    pred_n = float(len(preds))
                    matched_n = float(len(matched_gt_for_supervision))
                    prec = (matched_n / pred_n) if pred_n > 0 else 0.0
                    rec = (matched_n / gt_n) if gt_n > 0 else 0.0
                    f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

                    dump_samples.append(
                        {
                            "sample_id": sample.get("sample_id"),
                            "base_idx": sample.get("base_idx"),
                            "image": sample.get("image"),
                            "images": sample.get("images"),
                            "width": sample.get("width"),
                            "height": sample.get("height"),
                            "messages": sample.get("messages"),
                            "rollout_text": self._clip_text(
                                parse.response_text, max_chars=dump_max_chars
                            ),
                            "prefix_text": self._clip_text(
                                parse.prefix_text, max_chars=dump_max_chars
                            ),
                            "append_text": self._clip_text(
                                append_text, max_chars=dump_max_chars
                            ),
                            "train_text": self._clip_text(
                                tok.decode(
                                    y_train_ids,
                                    skip_special_tokens=False,
                                    clean_up_tokenization_spaces=False,
                                ),
                                max_chars=dump_max_chars,
                            ),
                            "gt_objects": gt_objs_dump,
                            "pred_objects": pred_objs_dump,
                            "match": {
                                "matched_pairs": list(match.matched_pairs),
                                "matched_pair_details": pair_details,
                                "fn_gt_indices": list(match.fn_gt_indices),
                                "fp_pred_indices": list(match.fp_pred_indices),
                                "gating_rejections": int(match.gating_rejections),
                            },
                            "stats": {
                                "decode_mode": str(decode_mode),
                                "parse_dropped_invalid": int(parse.dropped_invalid),
                                "parse_dropped_ambiguous": int(parse.dropped_ambiguous),
                                "parse_truncated": bool(parse.truncated),
                                "valid_pred_objects": int(len(preds)),
                                "gt_objects": int(len(gts)),
                                "matched_for_supervision": int(
                                    len(matched_gt_for_supervision)
                                ),
                                "excluded_from_supervision": int(excluded),
                                "fn_count": int(len(fn_objs)),
                                "precision": float(prec),
                                "recall": float(rec),
                                "f1": float(f1),
                                "matched_maskiou_mean": float(
                                    (
                                        match.matched_maskiou_sum
                                        / match.matched_maskiou_count
                                    )
                                    if match.matched_maskiou_count > 0
                                    else 0.0
                                ),
                            },
                        }
                    )
                except (TypeError, ValueError):
                    raise

            meta_entry = {
                "prompt_len": int(len(prompt_ids)),
                "prompt_ids": prompt_ids,
                "rollout_len": int(len(parse.response_token_ids)),
                "prefix_len": int(len(parse.prefix_token_ids)),
                "train_len": int(len(y_train_ids)),
                "encoded_len": int(encoded_len),
                "decode_mode": decode_mode,
                "parse_dropped_invalid": int(parse.dropped_invalid),
                "parse_dropped_ambiguous": int(parse.dropped_ambiguous),
                "parse_truncated": bool(parse.truncated),
                "valid_pred_objects": int(len(parse.valid_objects)),
                "matched_pairs": match.matched_pairs,
                "matched_for_supervision": int(len(matched_gt_for_supervision)),
                "matched_maskiou_sum": float(match.matched_maskiou_sum),
                "matched_maskiou_count": int(match.matched_maskiou_count),
                "gt_objects": int(len(gts)),
                "fn_count": int(len(fn_objs)),
                "gating_rejections": int(match.gating_rejections),
                "excluded_from_supervision": int(excluded),
                "prefix_coord_pos": prefix_pos,
                "prefix_coord_target_bins": prefix_target_bins,
                "tail_ignore_pos": tail_ignore_pos,
                # Optional desc monitor (metrics-only).
                "desc_monitor_ran": bool(desc_monitor_ran),
                "desc_pairs_total": int(desc_pairs_total),
                "desc_exact_ok": int(desc_exact_ok),
                "desc_sem_ok": int(desc_sem_ok),
                "desc_sem_sim_sum": float(desc_sem_sim_sum),
                "desc_sem_sim_count": int(desc_sem_sim_count),
                "desc_sem_enabled": int(desc_sem_enabled),
            }

            segments.append((encoded, meta_entry, int(encoded_len)))
            if not packing_enabled:
                encoded_batch.append(encoded)
                meta_unpacked.append(meta_entry)

        from swift.llm import to_device

        # Batch-level metrics are accumulated across micro-batches and merged into the
        # main step log line (together with train/loss) to avoid messy TB curves.
        batch_metrics: Dict[str, float] = {
            "time/rollout_generate_s": float(t_gen_s),
            "time/rollout_parse_match_s": float(t_parse_match_s),
            "time/rollout_teacher_encode_s": float(t_encode_s),
        }

        if bool(_segments_only):
            return segments, batch_metrics

        # For monitor dumps only (no TB logging here).
        toks_per_s = (
            float(sum(int(x) for x in rollout_lens)) / float(t_gen_s)
            if t_gen_s > 0
            else 0.0
        )

        if do_dump:
            try:
                payload = {
                    "global_step": int(gs),
                    "epoch": float(
                        getattr(getattr(self, "state", None), "epoch", 0.0) or 0.0
                    ),
                    "time": float(time.time()),
                    "meta": {
                        "rollout_backend": str(self._rollout_backend()),
                        "decode_mode": str(self._cfg("decode_mode", "greedy")),
                        "max_new_tokens": int(self._cfg("max_new_tokens", 0) or 0),
                        "candidate_top_k": int(top_k),
                        "maskiou_gate": float(gate_thr),
                        "maskiou_resolution": int(mask_res),
                        "fp_cost": float(fp_cost),
                        "fn_cost": float(fn_cost),
                        "ot_cost": str(ot_cost_kind),
                        "ot_epsilon": float(ot_eps),
                        "ot_iters": int(ot_iters),
                        "packing_enabled": bool(packing_enabled),
                        "rollout_generate_s": float(t_gen_s),
                        "rollout_tokens_per_s": float(toks_per_s)
                        if "toks_per_s" in locals()
                        else 0.0,
                    },
                    "samples": dump_samples,
                }
                self._write_monitor_dump(global_step=int(gs), payload=payload)
                self._monitor_dump_count += 1
            except (TypeError, ValueError):
                raise

        if packing_enabled:
            self._append_post_rollout_segments(segments)

            t_pack0 = time.perf_counter()
            selected, pack_metrics = self._pop_post_rollout_pack()
            with self._template_packing_enabled():
                packed = template.data_collator([enc for enc, _, _ in selected])
            batch = to_device(packed, self.model.device)
            self._assert_single_packed_forward(
                batch, where="rollout_matching/_prepare_batch_inputs"
            )
            batch["_rollout_matching_meta"] = [m for _, m, _ in selected]

            batch_metrics.update(pack_metrics)
            batch_metrics["time/post_rollout_pack_s"] = float(
                time.perf_counter() - t_pack0
            )
            self._merge_rollout_matching_batch_metrics(batch, batch_metrics)
            return batch

        with self._template_packing_disabled():
            batch = to_device(template.data_collator(encoded_batch), self.model.device)
        batch["_rollout_matching_meta"] = meta_unpacked
        self._merge_rollout_matching_batch_metrics(batch, batch_metrics)
        return batch

    # ------------------------ loss ------------------------ #

    def _reduce_train_rollout_log_payload_global(
        self, payload: Mapping[str, Any]
    ) -> Dict[str, float]:
        reduced: Dict[str, float] = {}
        for k, v in payload.items():
            try:
                reduced[str(k)] = float(v)
            except (TypeError, ValueError):
                continue

        trunc_num_key = "rollout/_parse_truncated_num"
        trunc_den_key = "rollout/_parse_truncated_den"
        sample_total_key = "train/samples_total"

        reduced.pop("rollout/parse_truncated_rate", None)
        reduced.setdefault(
            trunc_num_key,
            float(reduced.get("rollout/parse_truncated", 0.0)),
        )
        reduced.setdefault(
            trunc_den_key,
            float(reduced.get(sample_total_key, 0.0)),
        )

        sample_total_local = float(reduced.get(sample_total_key, 0.0))
        if (
            "rollout/matched_maskiou_mean" in reduced
            and "rollout/matched_maskiou_count" in reduced
        ):
            reduced["rollout/_matched_maskiou_sum"] = float(
                float(reduced.get("rollout/matched_maskiou_mean", 0.0))
                * float(reduced.get("rollout/matched_maskiou_count", 0.0))
            )
        if sample_total_local > 0.0:
            if "rollout/sample_valid_pred_rate" in reduced:
                reduced["rollout/_sample_valid_pred_num"] = float(
                    float(reduced.get("rollout/sample_valid_pred_rate", 0.0))
                    * sample_total_local
                )
            if "rollout/sample_any_match_rate" in reduced:
                reduced["rollout/_sample_any_match_num"] = float(
                    float(reduced.get("rollout/sample_any_match_rate", 0.0))
                    * sample_total_local
                )
        if (
            "rollout/desc_exact_acc_on_matched" in reduced
            and "rollout/desc_pairs_total" in reduced
        ):
            reduced["rollout/_desc_exact_ok"] = float(
                float(reduced.get("rollout/desc_exact_acc_on_matched", 0.0))
                * float(reduced.get("rollout/desc_pairs_total", 0.0))
            )
        if (
            "rollout/desc_sem_acc_on_matched" in reduced
            and "rollout/desc_pairs_total" in reduced
        ):
            reduced["rollout/_desc_sem_ok"] = float(
                float(reduced.get("rollout/desc_sem_acc_on_matched", 0.0))
                * float(reduced.get("rollout/desc_pairs_total", 0.0))
            )

        try:
            import torch.distributed as dist
        except (TypeError, ValueError):
            dist = None  # type: ignore[assignment]

        rank = 0
        world_size = 1
        if dist is not None and dist.is_available() and dist.is_initialized():
            try:
                world_size = int(dist.get_world_size())
            except (TypeError, ValueError):
                world_size = 1
            try:
                rank = int(dist.get_rank())
            except (TypeError, ValueError):
                rank = 0

        metric_keys: List[str] = sorted(str(k) for k in reduced.keys())
        if (
            dist is not None
            and dist.is_available()
            and dist.is_initialized()
            and int(world_size) > 1
        ):
            try:
                gathered_keys: List[Any] = [None] * int(world_size)
                dist.all_gather_object(gathered_keys, metric_keys)

                merged_keys: Dict[str, None] = {}
                for item in gathered_keys:
                    if not isinstance(item, (list, tuple)):
                        continue
                    for key_raw in item:
                        key = str(key_raw)
                        merged_keys[key] = None
                        reduced.setdefault(key, 0.0)
                metric_keys = sorted(merged_keys.keys())
            except (TypeError, ValueError) as exc:
                if int(rank) == 0:
                    logger.warning(
                        "rollout metric key sync failed; proceeding without key union: %r",
                        exc,
                    )

        sum_key_set: set[str] = set()
        max_key_set: set[str] = set()
        mean_key_set: set[str] = set()

        sum_explicit = {
            sample_total_key,
            "rollout/parse_truncated",
            "rollout/parse_dropped_invalid",
            "rollout/parse_dropped_ambiguous",
            "rollout/valid_pred_objects",
            "rollout/gt_objects",
            "rollout/matched_for_supervision",
            "rollout/excluded_from_supervision",
            "rollout/fp",
            "rollout/fn",
            "rollout/gating_rejections",
            "rollout/fn_appended",
            "rollout/prefix_coord_targets_total",
            "rollout/matched_maskiou_count",
            "rollout/desc_pairs_total",
            "rollout/desc_sem_sim_count",
            "rollout/_matched_maskiou_sum",
            "rollout/_sample_valid_pred_num",
            "rollout/_sample_any_match_num",
            "rollout/_desc_exact_ok",
            "rollout/_desc_sem_ok",
            "rollout/decode_non_beam_count",
            "rollout/decode_beam_count",
            trunc_num_key,
            trunc_den_key,
        }
        max_explicit = {
            "rollout/backend_hf",
            "rollout/backend_vllm",
            "rollout/decode_mode_greedy",
            "rollout/decode_mode_beam",
            "rollout/hf_seeded_global",
            "rollout/do_sample",
            "rollout/desc_sem_enabled",
            "rollout/gen_new_tokens_p90",
            "rollout/gen_new_tokens_p99",
        }

        for key in metric_keys:
            if key.startswith("time/"):
                max_key_set.add(key)
                continue
            if key in max_explicit:
                max_key_set.add(key)
                continue
            if key in sum_explicit or key.endswith("_total"):
                sum_key_set.add(key)
                continue
            mean_key_set.add(key)

        sum_keys = sorted(sum_key_set)
        max_keys = sorted(max_key_set)
        mean_keys = sorted(mean_key_set)

        if (
            dist is not None
            and dist.is_available()
            and dist.is_initialized()
            and int(world_size) > 1
        ):
            try:
                device = torch.device("cpu")
                try:
                    model = getattr(self, "model", None)
                    if model is not None and hasattr(model, "device"):
                        device = model.device
                    elif model is not None:
                        device = next(model.parameters()).device
                except (TypeError, ValueError):
                    device = torch.device("cpu")

                def _all_reduce(keys: List[str], op: Any) -> None:
                    if not keys:
                        return
                    values = torch.tensor(
                        [float(reduced.get(k, 0.0)) for k in keys],
                        dtype=torch.float64,
                        device=device,
                    )
                    dist.all_reduce(values, op=op)
                    for i, key_i in enumerate(keys):
                        reduced[key_i] = float(values[i].item())

                _all_reduce(sum_keys + mean_keys, dist.ReduceOp.SUM)
                _all_reduce(max_keys, dist.ReduceOp.MAX)

                scale = float(world_size)
                if scale > 0.0:
                    for key in mean_keys:
                        reduced[key] = float(reduced.get(key, 0.0) / scale)
            except (TypeError, ValueError) as exc:
                if int(rank) == 0:
                    logger.warning(
                        "rollout metric all-reduce failed; falling back to local metrics: %r",
                        exc,
                    )

        sample_total = float(reduced.get(sample_total_key, 0.0))
        trunc_num = float(reduced.get(trunc_num_key, 0.0))
        trunc_den = float(reduced.get(trunc_den_key, sample_total))
        reduced["rollout/parse_truncated_rate"] = (
            float(trunc_num / trunc_den) if trunc_den > 0.0 else 0.0
        )

        new_tok_total = float(reduced.get("rollout/gen_new_tokens_total", 0.0))
        if "rollout/gen_new_tokens_mean" in reduced:
            reduced["rollout/gen_new_tokens_mean"] = (
                float(new_tok_total / sample_total) if sample_total > 0.0 else 0.0
            )

        rollout_gen_s = float(reduced.get("time/rollout_generate_s", 0.0))
        if "rollout/gen_tokens_per_s" in reduced:
            reduced["rollout/gen_tokens_per_s"] = (
                float(new_tok_total / rollout_gen_s) if rollout_gen_s > 0.0 else 0.0
            )

        pred_total = float(reduced.get("rollout/valid_pred_objects", 0.0))
        gt_total = float(reduced.get("rollout/gt_objects", 0.0))
        matched_total = float(reduced.get("rollout/matched_for_supervision", 0.0))
        precision = (matched_total / pred_total) if pred_total > 0.0 else 0.0
        recall = (matched_total / gt_total) if gt_total > 0.0 else 0.0
        f1 = (
            (2.0 * precision * recall / (precision + recall))
            if (precision + recall) > 0.0
            else 0.0
        )
        reduced["rollout/precision"] = float(precision)
        reduced["rollout/recall"] = float(recall)
        reduced["rollout/f1"] = float(f1)

        if "rollout/excluded_rate" in reduced:
            excluded_total = float(
                reduced.get("rollout/excluded_from_supervision", 0.0)
            )
            denom = float(matched_total + excluded_total)
            reduced["rollout/excluded_rate"] = (
                float(excluded_total / denom) if denom > 0.0 else 0.0
            )

        if "rollout/prefix_coord_targets_per_matched" in reduced:
            prefix_total = float(reduced.get("rollout/prefix_coord_targets_total", 0.0))
            reduced["rollout/prefix_coord_targets_per_matched"] = (
                float(prefix_total / matched_total) if matched_total > 0.0 else 0.0
            )

        if "rollout/gating_rejection_rate" in reduced:
            top_k = int(self._cfg("candidate_top_k", 10))
            denom = float(pred_total * float(max(1, top_k)))
            reduced["rollout/gating_rejection_rate"] = (
                float(reduced.get("rollout/gating_rejections", 0.0) / denom)
                if denom > 0.0
                else 0.0
            )

        if sample_total > 0.0:
            for key_total, key_out in (
                ("rollout/gt_objects", "rollout/gt_per_sample"),
                ("rollout/valid_pred_objects", "rollout/pred_per_sample"),
                ("rollout/fp", "rollout/fp_per_sample"),
                ("rollout/fn", "rollout/fn_per_sample"),
            ):
                if key_out in reduced or key_total in reduced:
                    reduced[key_out] = float(reduced.get(key_total, 0.0) / sample_total)

        if (
            "rollout/parse_obj_valid_frac" in reduced
            or "rollout/parse_obj_drop_frac" in reduced
            or "rollout/parse_obj_total" in reduced
        ):
            dropped_invalid_total = float(
                reduced.get("rollout/parse_dropped_invalid", 0.0)
            )
            dropped_amb_total = float(
                reduced.get("rollout/parse_dropped_ambiguous", 0.0)
            )
            obj_total = pred_total + dropped_invalid_total + dropped_amb_total
            reduced["rollout/parse_obj_total"] = float(obj_total)
            reduced["rollout/parse_obj_valid_frac"] = (
                float(pred_total / obj_total) if obj_total > 0.0 else 0.0
            )
            reduced["rollout/parse_obj_drop_frac"] = (
                float((dropped_invalid_total + dropped_amb_total) / obj_total)
                if obj_total > 0.0
                else 0.0
            )

        if "rollout/matched_maskiou_mean" in reduced:
            iou_sum = float(reduced.get("rollout/_matched_maskiou_sum", 0.0))
            iou_cnt = float(reduced.get("rollout/matched_maskiou_count", 0.0))
            reduced["rollout/matched_maskiou_mean"] = (
                float(iou_sum / iou_cnt) if iou_cnt > 0.0 else 0.0
            )

        if "rollout/sample_valid_pred_rate" in reduced:
            valid_pred_num = float(reduced.get("rollout/_sample_valid_pred_num", 0.0))
            reduced["rollout/sample_valid_pred_rate"] = (
                float(valid_pred_num / sample_total) if sample_total > 0.0 else 0.0
            )
        if "rollout/sample_any_match_rate" in reduced:
            any_match_num = float(reduced.get("rollout/_sample_any_match_num", 0.0))
            reduced["rollout/sample_any_match_rate"] = (
                float(any_match_num / sample_total) if sample_total > 0.0 else 0.0
            )

        if "rollout/desc_exact_acc_on_matched" in reduced:
            pairs_total = float(reduced.get("rollout/desc_pairs_total", 0.0))
            exact_ok = float(reduced.get("rollout/_desc_exact_ok", 0.0))
            reduced["rollout/desc_exact_acc_on_matched"] = (
                float(exact_ok / pairs_total) if pairs_total > 0.0 else 1.0
            )
        if "rollout/desc_sem_acc_on_matched" in reduced:
            pairs_total = float(reduced.get("rollout/desc_pairs_total", 0.0))
            sem_ok = float(reduced.get("rollout/_desc_sem_ok", 0.0))
            reduced["rollout/desc_sem_acc_on_matched"] = (
                float(sem_ok / pairs_total) if pairs_total > 0.0 else 1.0
            )
        if "rollout/desc_sem_sim_mean" in reduced:
            sim_sum = float(reduced.get("rollout/desc_sem_sim_sum", 0.0))
            sim_cnt = float(reduced.get("rollout/desc_sem_sim_count", 0.0))
            reduced["rollout/desc_sem_sim_mean"] = (
                float(sim_sum / sim_cnt) if sim_cnt > 0.0 else 0.0
            )

        for key in (
            trunc_num_key,
            trunc_den_key,
            "rollout/_matched_maskiou_sum",
            "rollout/_sample_valid_pred_num",
            "rollout/_sample_any_match_num",
            "rollout/_desc_exact_ok",
            "rollout/_desc_sem_ok",
        ):
            reduced.pop(key, None)
        return reduced

    def log(self, logs: Dict[str, float]) -> None:
        """Merge buffered rollout-matching metrics into the main train log record.

        HF/Swift logs `loss` after the optimizer step (global_step already incremented).
        Our rollout metrics are computed inside `compute_loss` (before the increment),
        so we buffer them keyed by `global_step + 1` and merge here.

        This keeps one scalar per step per tag in TensorBoard (clean plots) and reduces
        `logging.jsonl` fragmentation.
        """

        try:
            if (
                isinstance(logs, dict)
                and "loss" in logs
                and not any(str(k).startswith("eval_") for k in logs.keys())
            ):
                step = int(getattr(getattr(self, "state", None), "global_step", 0) or 0)
                pending = self._rm_pending_train_logs.pop(step, None)
                if pending is not None:
                    payload = self._build_train_rollout_log_payload(pending)
                    payload = self._reduce_train_rollout_log_payload_global(payload)
                    sample_step = int(
                        round(float(payload.get("train/samples_total", 0.0)))
                    )
                    try:
                        self._rm_train_samples_seen = (
                            int(getattr(self, "_rm_train_samples_seen", 0) or 0)
                            + sample_step
                        )
                    except (TypeError, ValueError):
                        self._rm_train_samples_seen = sample_step

                    logs.update(payload)
                    logs["train/samples_seen"] = float(
                        getattr(self, "_rm_train_samples_seen", 0) or 0
                    )
        except (TypeError, ValueError):
            raise

        return super().log(logs)

    def _build_rollout_metrics_from_meta(
        self, meta: List[Mapping[str, Any]]
    ) -> Dict[str, float]:
        """Compute step-level rollout metrics from slim meta dicts."""

        n_samples = float(len(meta))
        gt_total = float(sum(int(m.get("gt_objects", 0)) for m in meta))
        matched_total = float(
            sum(int(m.get("matched_for_supervision", 0)) for m in meta)
        )
        pred_total = float(sum(int(m.get("valid_pred_objects", 0)) for m in meta))
        excluded_total = float(
            sum(int(m.get("excluded_from_supervision", 0)) for m in meta)
        )

        # Sample-level rates (helps detect systematic parse failures).
        n_samples_valid_pred = float(
            sum(1 for m in meta if int(m.get("valid_pred_objects", 0)) > 0)
        )
        n_samples_any_match = float(
            sum(1 for m in meta if int(m.get("matched_for_supervision", 0)) > 0)
        )
        sample_valid_pred_rate = (
            (n_samples_valid_pred / n_samples) if n_samples > 0 else 0.0
        )
        sample_any_match_rate = (
            (n_samples_any_match / n_samples) if n_samples > 0 else 0.0
        )

        fp_total = max(0.0, pred_total - matched_total)
        fn_total = max(0.0, gt_total - matched_total)
        precision = (matched_total / pred_total) if pred_total > 0 else 0.0
        recall = (matched_total / gt_total) if gt_total > 0 else 0.0
        f1 = (
            (2.0 * precision * recall / (precision + recall))
            if (precision + recall) > 0.0
            else 0.0
        )

        dropped_invalid_total = float(
            sum(int(m.get("parse_dropped_invalid", 0)) for m in meta)
        )
        dropped_ambiguous_total = float(
            sum(int(m.get("parse_dropped_ambiguous", 0)) for m in meta)
        )
        obj_total = pred_total + dropped_invalid_total + dropped_ambiguous_total
        obj_valid_frac = (pred_total / obj_total) if obj_total > 0 else 0.0
        obj_drop_frac = (
            ((dropped_invalid_total + dropped_ambiguous_total) / obj_total)
            if obj_total > 0
            else 0.0
        )

        trunc_samples = float(sum(1 for m in meta if m.get("parse_truncated")))
        trunc_rate = (trunc_samples / n_samples) if n_samples > 0 else 0.0

        gate_rejections_total = float(
            sum(int(m.get("gating_rejections", 0)) for m in meta)
        )
        top_k = int(self._cfg("candidate_top_k", 10))
        gate_rejection_rate = (
            (gate_rejections_total / (pred_total * float(max(1, top_k))))
            if pred_total > 0
            else 0.0
        )

        matched_iou_sum = float(
            sum(float(m.get("matched_maskiou_sum", 0.0)) for m in meta)
        )
        matched_iou_count = float(
            sum(int(m.get("matched_maskiou_count", 0)) for m in meta)
        )
        matched_iou_mean = (
            (matched_iou_sum / matched_iou_count) if matched_iou_count > 0 else 0.0
        )

        # Supervision coverage diagnostics.
        excluded_rate = (
            (excluded_total / (matched_total + excluded_total))
            if (matched_total + excluded_total) > 0
            else 0.0
        )
        prefix_targets_total = float(
            sum(len(m.get("prefix_coord_target_bins") or []) for m in meta)
        )
        prefix_targets_per_matched = (
            (prefix_targets_total / matched_total) if matched_total > 0 else 0.0
        )
        tail_ignore_total = float(
            sum(len(m.get("tail_ignore_pos") or []) for m in meta)
        )
        append_len_total = float(
            sum(
                max(0, int(m.get("train_len", 0)) - int(m.get("prefix_len", 0)))
                for m in meta
            )
        )
        tail_ignore_frac = (
            (tail_ignore_total / append_len_total) if append_len_total > 0 else 0.0
        )

        # Length stats (prompt/prefix/train/encoded) help diagnose truncation/packing behavior.
        def _int_list(key: str) -> List[int]:
            xs: List[int] = []
            for m in meta:
                try:
                    xs.append(int(m.get(key, 0)))
                except (TypeError, ValueError):
                    continue
            return xs

        prompt_lens = _int_list("prompt_len")
        prefix_lens = _int_list("prefix_len")
        train_lens = _int_list("train_len")
        encoded_lens = _int_list("encoded_len")
        rollout_lens = _int_list("rollout_len")
        append_lens: List[int] = []
        for m in meta:
            try:
                append_lens.append(
                    int(m.get("train_len", 0)) - int(m.get("prefix_len", 0))
                )
            except (TypeError, ValueError):
                continue

        def _mean(xs: List[int]) -> float:
            return float(sum(xs) / len(xs)) if xs else 0.0

        def _p(xs: List[int], q: float) -> float:
            if not xs:
                return 0.0
            arr = np.asarray(xs, dtype=np.float64)
            return float(np.percentile(arr, float(q)))

        payload: Dict[str, float] = {
            "rollout/parse_dropped_invalid": float(dropped_invalid_total),
            "rollout/parse_dropped_ambiguous": float(dropped_ambiguous_total),
            "rollout/parse_truncated": float(trunc_samples),
            "rollout/parse_truncated_rate": float(trunc_rate),
            "rollout/parse_obj_total": float(obj_total),
            "rollout/parse_obj_valid_frac": float(obj_valid_frac),
            "rollout/parse_obj_drop_frac": float(obj_drop_frac),
            "rollout/sample_valid_pred_rate": float(sample_valid_pred_rate),
            "rollout/sample_any_match_rate": float(sample_any_match_rate),
            "rollout/fn_appended": float(sum(int(m.get("fn_count", 0)) for m in meta)),
            "rollout/gating_rejections": float(gate_rejections_total),
            "rollout/gating_rejection_rate": float(gate_rejection_rate),
            "rollout/valid_pred_objects": float(pred_total),
            "rollout/gt_objects": float(gt_total),
            "rollout/precision": float(precision),
            "rollout/recall": float(recall),
            "rollout/f1": float(f1),
            "rollout/fp": float(fp_total),
            "rollout/fn": float(fn_total),
            "rollout/gt_per_sample": float(gt_total / n_samples)
            if n_samples > 0
            else 0.0,
            "rollout/pred_per_sample": float(pred_total / n_samples)
            if n_samples > 0
            else 0.0,
            "rollout/fp_per_sample": float(fp_total / n_samples)
            if n_samples > 0
            else 0.0,
            "rollout/fn_per_sample": float(fn_total / n_samples)
            if n_samples > 0
            else 0.0,
            "rollout/matched_maskiou_mean": float(matched_iou_mean),
            "rollout/matched_maskiou_count": float(matched_iou_count),
            "rollout/excluded_rate": float(excluded_rate),
            "rollout/prefix_coord_targets_total": float(prefix_targets_total),
            "rollout/prefix_coord_targets_per_matched": float(
                prefix_targets_per_matched
            ),
            "rollout/tail_ignore_frac": float(tail_ignore_frac),
            "rollout/prompt_len_mean": float(_mean(prompt_lens)),
            "rollout/prompt_len_p90": float(_p(prompt_lens, 90)),
            "rollout/prefix_len_mean": float(_mean(prefix_lens)),
            "rollout/rollout_len_mean": float(_mean(rollout_lens)),
            "rollout/rollout_len_p90": float(_p(rollout_lens, 90)),
            "rollout/train_len_mean": float(_mean(train_lens)),
            "rollout/train_len_p90": float(_p(train_lens, 90)),
            "rollout/append_len_mean": float(_mean(append_lens)),
            "rollout/append_len_p90": float(_p(append_lens, 90)),
            "rollout/encoded_len_mean": float(_mean(encoded_lens)),
            "rollout/encoded_len_p90": float(_p(encoded_lens, 90)),
            "rollout/decode_non_beam_count": float(
                sum(1 for m in meta if str(m.get("decode_mode", "")).lower() != "beam")
            ),
            "rollout/decode_beam_count": float(
                sum(1 for m in meta if str(m.get("decode_mode", "")).lower() == "beam")
            ),
            "rollout/matched_for_supervision": float(matched_total),
            "rollout/excluded_from_supervision": float(excluded_total),
        }

        try:
            temperature, top_p, top_k = self._decoding_params()
            do_sample = bool(float(temperature) > 0.0)
            payload["rollout/do_sample"] = float(1.0 if do_sample else 0.0)
            payload["rollout/temperature"] = float(temperature)
            payload["rollout/top_p"] = float(top_p)
            payload["rollout/top_k"] = float(top_k)
        except (TypeError, ValueError):
            raise

        # Desc monitor outputs (matched pairs only).
        try:
            if any(bool(m.get("desc_monitor_ran", False)) for m in meta):
                pairs_total = float(
                    sum(int(m.get("desc_pairs_total", 0)) for m in meta)
                )
                exact_ok_total = float(
                    sum(int(m.get("desc_exact_ok", 0)) for m in meta)
                )
                exact_acc = (exact_ok_total / pairs_total) if pairs_total > 0 else 1.0
                payload["rollout/desc_pairs_total"] = float(pairs_total)
                payload["rollout/desc_exact_acc_on_matched"] = float(exact_acc)

                sem_enabled_total = float(
                    sum(int(m.get("desc_sem_enabled", 0)) for m in meta)
                )
                payload["rollout/desc_sem_enabled"] = float(
                    1.0 if sem_enabled_total > 0 else 0.0
                )
                if sem_enabled_total > 0:
                    sem_ok_total = float(
                        sum(int(m.get("desc_sem_ok", 0)) for m in meta)
                    )
                    sem_acc = (sem_ok_total / pairs_total) if pairs_total > 0 else 1.0
                    payload["rollout/desc_sem_acc_on_matched"] = float(sem_acc)

                    sim_sum_total = float(
                        sum(float(m.get("desc_sem_sim_sum", 0.0)) for m in meta)
                    )
                    sim_count_total = float(
                        sum(int(m.get("desc_sem_sim_count", 0)) for m in meta)
                    )
                    if sim_count_total > 0:
                        payload["rollout/desc_sem_sim_mean"] = float(
                            sim_sum_total / sim_count_total
                        )
                        payload["rollout/desc_sem_sim_count"] = float(sim_count_total)
        except (TypeError, ValueError):
            raise

        return payload

    def _build_train_rollout_log_payload(
        self, pending: _PendingTrainRolloutLog
    ) -> Dict[str, float]:
        payload = self._build_rollout_metrics_from_meta(pending.meta)

        sample_total = float(len(pending.meta))
        payload["train/samples_total"] = float(sample_total)
        payload["train/micro_steps"] = float(pending.n_micro)
        payload["train/samples_per_micro"] = (
            float(sample_total / float(pending.n_micro)) if pending.n_micro > 0 else 0.0
        )

        payload["rollout/_parse_truncated_num"] = float(
            payload.get("rollout/parse_truncated", 0.0)
        )
        payload["rollout/_parse_truncated_den"] = float(sample_total)

        if pending.n_micro > 0:
            payload["loss/ce"] = float(pending.ce_loss_sum / float(pending.n_micro))
            payload["loss/coord"] = float(
                pending.coord_loss_sum / float(pending.n_micro)
            )
            payload["loss/coord_prefix"] = float(
                pending.coord_prefix_sum / float(pending.n_micro)
            )
            payload["loss/coord_tail"] = float(
                pending.coord_tail_sum / float(pending.n_micro)
            )

        payload["time/forward_s"] = float(pending.time_forward_s)
        payload["time/mask_build_s"] = float(pending.time_mask_build_s)

        payload["time/rollout_generate_s"] = float(pending.time_rollout_generate_s)
        payload["time/rollout_parse_match_s"] = float(
            pending.time_rollout_parse_match_s
        )
        payload["time/rollout_teacher_encode_s"] = float(
            pending.time_rollout_teacher_encode_s
        )
        if pending.time_post_rollout_pack_s > 0:
            payload["time/post_rollout_pack_s"] = float(
                pending.time_post_rollout_pack_s
            )

        if pending.packing_count > 0:
            payload["packing/post_rollout_fill"] = float(
                pending.packing_fill_sum / float(pending.packing_count)
            )
            payload["packing/post_rollout_selected_total_len"] = float(
                pending.packing_selected_total_len_sum / float(pending.packing_count)
            )
            payload["packing/post_rollout_segments"] = float(
                pending.packing_segments_sum / float(pending.packing_count)
            )
            payload["packing/post_rollout_buffer"] = float(pending.packing_buffer_last)

        # Generation-length stats are only meaningful when we actually ran a rollout this step.
        if pending.time_rollout_generate_s > 0:
            rollout_lens = [int(m.get("rollout_len", 0)) for m in pending.meta]

            def _p(xs: List[int], q: float) -> float:
                if not xs:
                    return 0.0
                arr = np.asarray(xs, dtype=np.float64)
                return float(np.percentile(arr, float(q)))

            new_tok_total = float(sum(int(x) for x in rollout_lens))
            new_tok_mean = (
                float(new_tok_total / len(rollout_lens)) if rollout_lens else 0.0
            )
            payload["rollout/gen_new_tokens_total"] = float(new_tok_total)
            payload["rollout/gen_new_tokens_mean"] = float(new_tok_mean)
            payload["rollout/gen_new_tokens_p90"] = float(_p(rollout_lens, 90))
            payload["rollout/gen_new_tokens_p99"] = float(_p(rollout_lens, 99))
            payload["rollout/gen_tokens_per_s"] = float(
                (new_tok_total / float(pending.time_rollout_generate_s))
                if pending.time_rollout_generate_s > 0
                else 0.0
            )

        return payload

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        meta = inputs.pop("_rollout_matching_meta", None)
        if not isinstance(meta, list):
            raise ValueError("rollout-matching trainer requires _rollout_matching_meta")

        batch_metrics = inputs.pop("_rollout_matching_batch_metrics", None)

        # Always compute logits; do not rely on model.loss (we need custom masking).
        # NOTE: ms-swift's Seq2SeqTrainer/_prepare_inputs may inject helper keys
        # like compute_loss_func/loss_scale/channel. Strip them before model forward.
        ignored_keys = {
            "labels",
            "compute_loss_func",
            "loss_scale",
            "text_position_ids",
            "channel",
            "logits_to_keep",
        }
        inputs_for_model = {k: v for k, v in inputs.items() if k not in ignored_keys}

        # Qwen-VL (mRoPE) + padding_free packing:
        # - Swift templates emit `position_ids` as the 3 mRoPE rows (t/h/w) and a separate
        #   `text_position_ids` (pure sequential) for packing metadata.
        # - Transformers' SDPA/eager mask creation may interpret non-unit diffs in position_ids as
        #   packed-sequence boundaries when `attention_mask is None` (padding_free mode).
        #   For Qwen3-VL, the mRoPE temporal row is not strictly sequential through vision tokens,
        #   so we pass a 4-row `position_ids` where the first row is `text_position_ids`.
        #   This matches HF Qwen3-VL's forward contract and keeps attention/masking correct.
        try:
            model_type = str(
                getattr(getattr(model, "config", None), "model_type", "") or ""
            )
        except (TypeError, ValueError):
            model_type = ""
        text_position_ids = inputs.get("text_position_ids")
        position_ids = inputs_for_model.get("position_ids")
        if (
            model_type.startswith("qwen")
            and isinstance(text_position_ids, torch.Tensor)
            and isinstance(position_ids, torch.Tensor)
            and position_ids.ndim == 3
            and position_ids.shape[0] == 3
            and text_position_ids.ndim == 2
            and text_position_ids.shape == position_ids.shape[1:]
        ):
            inputs_for_model["position_ids"] = torch.cat(
                [text_position_ids.unsqueeze(0), position_ids], dim=0
            )
        # Disable KV cache during training to reduce memory and avoid accidental PKV returns.
        inputs_for_model["use_cache"] = False
        inputs_for_model.pop("past_key_values", None)

        t_fwd0 = time.perf_counter()
        outputs = model(**inputs_for_model)
        t_fwd_s = time.perf_counter() - t_fwd0
        logits = outputs.logits
        if logits is None:
            raise ValueError("model did not return logits")

        input_ids = inputs.get("input_ids")
        if (
            isinstance(input_ids, torch.Tensor)
            and logits.shape[:2] != input_ids.shape[:2]
        ):
            raise ValueError(
                "model returned sliced logits (logits_to_keep-style). Disable logits slicing for rollout-matching training."
            )

        bsz, seq_len, vocab = logits.shape
        coord_token_ids = self._get_coord_token_ids()
        coord_ids_t = torch.tensor(
            coord_token_ids, device=logits.device, dtype=torch.long
        )
        coord_id_set = set(int(i) for i in coord_token_ids if int(i) >= 0)
        coord_id_to_bin = self._coord_id_map()

        # Build custom labels for CE (tail non-coord tokens only) and collect
        # coord supervision targets (prefix self-context + tail GT).
        input_ids = inputs["input_ids"]
        t_mask0 = time.perf_counter()
        (
            labels_masked,
            supervised_batch,
            supervised_pos,
            supervised_bin,
            supervised_is_prefix,
        ) = _build_labels_and_coord_targets_for_batch(
            input_ids=input_ids,
            meta=meta,
            coord_id_set=coord_id_set,
            coord_id_to_bin=coord_id_to_bin,
        )
        t_mask_s = time.perf_counter() - t_mask0

        # Standard CE on masked labels (mean over supervised tokens).
        logits_next = logits[:, :-1, :]
        labels_next = labels_masked[:, 1:]
        ce_loss = F.cross_entropy(
            logits_next.reshape(-1, vocab),
            labels_next.reshape(-1),
            ignore_index=-100,
            reduction="mean",
        )

        # Coord losses (mean over coord-supervised tokens).
        coord_loss = ce_loss.new_tensor(0.0)
        prefix_coord_mean = ce_loss.new_tensor(0.0)
        tail_coord_mean = ce_loss.new_tensor(0.0)
        if supervised_pos:
            b_t = torch.tensor(supervised_batch, device=logits.device, dtype=torch.long)
            pos_t = torch.tensor(supervised_pos, device=logits.device, dtype=torch.long)
            bin_t = torch.tensor(
                supervised_bin, device=logits.device, dtype=torch.long
            ).clamp(min=0, max=999)
            is_prefix_t = torch.tensor(
                supervised_is_prefix, device=logits.device, dtype=torch.bool
            )

            logit_pos = (pos_t - 1).clamp(min=0, max=seq_len - 2)
            logits_full = logits_next[b_t, logit_pos, :]  # [N, V]
            logits_coord = logits_full.index_select(-1, coord_ids_t)  # [N, 1000]

            # Loss weights come from rollout cfg, falling back to coord_soft_ce_w1_cfg.
            cfg = getattr(self, "coord_soft_ce_w1_cfg", None)
            sigma = float(
                self._cfg("target_sigma", float(getattr(cfg, "target_sigma", 2.0)))
            )
            truncate = self._cfg(
                "target_truncate", getattr(cfg, "target_truncate", None)
            )
            temperature = float(
                self._cfg("temperature_coord", float(getattr(cfg, "temperature", 1.0)))
            )
            soft_w = float(
                self._cfg("soft_ce_weight", float(getattr(cfg, "soft_ce_weight", 1.0)))
            )
            w1_w = float(self._cfg("w1_weight", float(getattr(cfg, "w1_weight", 1.0))))
            gate_w = float(
                self._cfg("gate_weight", float(getattr(cfg, "gate_weight", 0.0)))
            )

            out = coord_soft_ce_w1(
                logits_coord,
                bin_t,
                sigma=sigma,
                truncate=truncate,
                temperature=temperature,
                soft_ce_weight=1.0,
                w1_weight=1.0,
                normalize_w1=True,
            )
            gate_per = (
                _coord_vocab_gate_loss(
                    logits_full=logits_full,
                    logits_coord=logits_coord,
                    temperature=temperature,
                )
                if gate_w != 0.0
                else logits_full.new_zeros((logits_full.shape[0],), dtype=torch.float32)
            )

            per_tok = (
                soft_w * out.soft_ce_per_token
                + w1_w * out.w1_per_token
                + gate_w * gate_per
            )
            denom = per_tok.numel()
            if denom > 0:
                coord_loss = per_tok.mean().to(dtype=ce_loss.dtype)
            if is_prefix_t.any().item():
                prefix_coord_mean = per_tok[is_prefix_t].mean().to(dtype=ce_loss.dtype)
            if (~is_prefix_t).any().item():
                tail_coord_mean = per_tok[~is_prefix_t].mean().to(dtype=ce_loss.dtype)

        total = ce_loss + coord_loss

        # Accumulate rollout-matching metrics across micro-batches and merge them into the
        # *post-optimizer* train log line (same step as train/loss). This avoids duplicated
        # TB scalars at the same step (messy plots) under gradient accumulation.
        try:
            step = int(getattr(getattr(self, "state", None), "global_step", 0) or 0)
            target_step = step + 1
            pending = self._rm_pending_train_logs.get(target_step)
            if pending is None:
                pending = _PendingTrainRolloutLog()
                self._rm_pending_train_logs[target_step] = pending
            pending.add_micro(
                meta=meta,
                ce_loss=float(ce_loss.detach().cpu().item()),
                coord_loss=float(coord_loss.detach().cpu().item()),
                coord_prefix=float(prefix_coord_mean.detach().cpu().item()),
                coord_tail=float(tail_coord_mean.detach().cpu().item()),
                time_forward_s=float(t_fwd_s),
                time_mask_build_s=float(t_mask_s),
                batch_metrics=batch_metrics
                if isinstance(batch_metrics, Mapping)
                else None,
            )
        except (TypeError, ValueError):
            raise

        return (total, outputs) if return_outputs else total

    def get_train_dataloader(self):
        dl = super().get_train_dataloader()

        try:
            per_dev = int(getattr(self.args, "per_device_train_batch_size", 1) or 1)
        except (TypeError, ValueError):
            per_dev = 1

        # Optional fixed raw batching for post-rollout packing.
        #
        # Stage-2 trainers use identity collator, so the dataloader yields lists of raw
        # samples. When per_device_train_batch_size==1, a single raw sample per micro-step
        # can lead to poor packing fill (until the carry buffer grows).
        #
        # We only apply this wrapper for the standalone rollout-matching SFT variant.
        # Stage2-AB budgets raw samples per optimizer step via training.effective_batch_size
        # and must not have its micro-batches implicitly resized here.
        trainer_variant = str(getattr(self.args, "trainer_variant", "") or "")
        if trainer_variant == "rollout_matching_sft":
            try:
                decode_bs = int(self._decode_batch_size())
            except (TypeError, ValueError):
                decode_bs = 1
            decode_bs = max(1, int(decode_bs))

            if self._packing_enabled() and per_dev == 1 and int(decode_bs) > 1:
                dl = _FixedRawMicroBatchStacker(
                    dl,
                    target_raw_batch_size=int(decode_bs),
                    base_raw_batch_size=int(per_dev),
                )

        gas = int(getattr(self.args, "gradient_accumulation_steps", 1) or 1)

        # Drop the final partial accumulation window when requested.
        #
        # This keeps optimizer-step semantics consistent for step-budgeted Stage-2 runs
        # (fixed samples per step) and avoids a trailing underfull/no-op step.
        try:
            drop_last = bool(getattr(self.args, "dataloader_drop_last", False))
        except (TypeError, ValueError):
            drop_last = False
        if self._packing_enabled() and drop_last and int(gas) > 1:
            dl = _DropRemainderAccumulationWindow(dl, gas=gas)

        return dl

    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
    ):
        """Production-style evaluator: rollout -> parse -> Hungarian match.

        This intentionally skips teacher-forced encoding and loss computation to keep eval
        fast and reflective of real rollout performance on unseen data.
        """

        was_training = bool(getattr(self.model, "training", False))
        self.model.eval()

        t0 = time.perf_counter()
        dl = self.get_eval_dataloader(eval_dataset)

        template = self.template
        tok = template.tokenizer

        gate_thr = float(self._cfg("maskiou_gate", 0.3))
        top_k = int(self._cfg("candidate_top_k", 10))
        mask_res = int(self._cfg("maskiou_resolution", 256))
        fp_cost = float(self._cfg("fp_cost", 1.0))
        fn_cost = float(self._cfg("fn_cost", 1.0))

        # Optional semantic desc monitoring (metrics only).
        desc_cfg = self._desc_monitor_cfg()
        desc_enabled = isinstance(desc_cfg, Mapping) and bool(
            desc_cfg.get("enabled", False)
        )
        desc_mode = str(desc_cfg.get("mode", "semantic") or "semantic").strip().lower()
        desc_thr = float(desc_cfg.get("semantic_threshold", 0.6) or 0.6)
        desc_max_pairs = int(desc_cfg.get("max_pairs", 0) or 0)

        try:
            from src.metrics.semantic_desc import normalize_desc
        except (TypeError, ValueError):
            normalize_desc = None  # type: ignore[assignment]

        sem_loaded_local = 0.0
        sem_encoder = None
        if desc_enabled and desc_mode in {"semantic", "both"}:
            try:
                sem_encoder = self._get_desc_semantic_encoder(desc_cfg)
            except (TypeError, ValueError):
                sem_encoder = None
            if sem_encoder is not None:
                # Probe once so failures are detected consistently across ranks.
                try:
                    _ = sem_encoder.encode_norm_texts(["__probe__"])
                    sem_loaded_local = 1.0
                except (TypeError, ValueError):
                    sem_encoder = None
                    sem_loaded_local = 0.0

        n_samples = 0.0
        gt_total = 0.0
        pred_total = 0.0
        matched_total = 0.0
        fp_total = 0.0
        fn_total = 0.0
        gating_rejections_total = 0.0
        dropped_invalid_total = 0.0
        dropped_ambiguous_total = 0.0
        trunc_samples = 0.0
        matched_iou_sum = 0.0
        matched_iou_count = 0.0
        n_samples_valid_pred = 0.0
        n_samples_any_match = 0.0

        # Desc monitor accumulators (matched pairs only).
        desc_pairs_total = 0.0
        desc_exact_ok_total = 0.0
        desc_sem_ok_total = 0.0
        desc_sem_sim_sum_total = 0.0
        desc_sem_sim_count_total = 0.0

        n_steps = 0.0

        with torch.no_grad():
            for batch in dl:
                # For rollout-matching, we expect identity_data_collator to yield a
                # list[dict] of raw samples (with `messages` + GT geometry).
                if not isinstance(batch, list):
                    raise ValueError(
                        "rollout-matching evaluator expects eval batches as list[dict]; "
                        f"got {type(batch).__name__}"
                    )
                if not batch:
                    continue

                n_steps += 1.0
                rollout_results = self._rollout_many(batch)
                if len(rollout_results) != len(batch):
                    raise RuntimeError(
                        "rollout backend returned unexpected number of results"
                    )

                for sample, (resp_ids, _resp_text, _decode_mode, _prompt_ids) in zip(
                    batch, rollout_results
                ):
                    n_samples += 1.0

                    parse = parse_rollout_for_matching(
                        tokenizer=tok,
                        response_token_ids=resp_ids,
                        object_field_order=self._object_field_order(),
                    )
                    dropped_invalid_total += float(parse.dropped_invalid)
                    dropped_ambiguous_total += float(parse.dropped_ambiguous)
                    trunc_samples += 1.0 if bool(parse.truncated) else 0.0

                    # Pred objects (valid only) -> norm1000 geometry.
                    coord_id_to_bin = self._coord_id_map()
                    pred_meta = list(parse.valid_objects)
                    preds: List[GTObject] = []
                    for pobj in pred_meta:
                        pts = _points_from_coord_tokens(
                            response_token_ids=parse.response_token_ids,
                            coord_token_indices=pobj.coord_token_indices,
                            coord_id_to_bin=coord_id_to_bin,
                        )
                        if pts is None:
                            continue
                        preds.append(
                            GTObject(
                                index=int(pobj.index),
                                geom_type=pobj.geom_type,
                                points_norm1000=pts,
                                desc="",
                            )
                        )

                    gts = _extract_gt_objects(sample)
                    gt_total += float(len(gts))
                    pred_total += float(len(preds))
                    if len(preds) > 0:
                        n_samples_valid_pred += 1.0

                    match = hungarian_match_maskiou(
                        preds=preds,
                        gts=gts,
                        top_k=top_k,
                        gate_threshold=gate_thr,
                        mask_resolution=mask_res,
                        fp_cost=fp_cost,
                        fn_cost=fn_cost,
                    )

                    matched = float(len(match.matched_pairs))
                    matched_total += matched
                    fp_total += float(len(match.fp_pred_indices))
                    fn_total += float(len(match.fn_gt_indices))
                    gating_rejections_total += float(match.gating_rejections)
                    matched_iou_sum += float(match.matched_maskiou_sum)
                    matched_iou_count += float(match.matched_maskiou_count)
                    if matched > 0:
                        n_samples_any_match += 1.0

                    # Optional desc semantic monitor on matched pairs.
                    if desc_enabled and match.matched_pairs:
                        pairs = list(match.matched_pairs)
                        if desc_max_pairs > 0 and len(pairs) > desc_max_pairs:
                            pairs = pairs[:desc_max_pairs]

                        uniq: set[str] = set()
                        norm_pairs: List[Tuple[str, str, bool]] = []
                        for pred_i, gt_i in pairs:
                            if pred_i < 0 or pred_i >= len(pred_meta):
                                continue
                            if gt_i < 0 or gt_i >= len(gts):
                                continue
                            pred_desc_raw = str(
                                getattr(pred_meta[pred_i], "desc", "") or ""
                            )
                            gt_desc_raw = str(getattr(gts[gt_i], "desc", "") or "")
                            if normalize_desc is None:
                                p = pred_desc_raw.strip().lower()
                                g = gt_desc_raw.strip().lower()
                            else:
                                p = normalize_desc(pred_desc_raw)
                                g = normalize_desc(gt_desc_raw)
                            exact_ok = bool(p) and (p == g)
                            if exact_ok:
                                desc_exact_ok_total += 1.0
                            if p and g:
                                norm_pairs.append((p, g, bool(exact_ok)))
                                uniq.add(p)
                                uniq.add(g)

                        desc_pairs_total += float(len(norm_pairs))

                        if (
                            sem_loaded_local > 0.0
                            and sem_encoder is not None
                            and norm_pairs
                        ):
                            try:
                                emb = sem_encoder.encode_norm_texts(sorted(uniq))
                            except (TypeError, ValueError):
                                emb = {}
                                sem_encoder = None
                                sem_loaded_local = 0.0

                            if sem_loaded_local > 0.0 and sem_encoder is not None:
                                for p, g, exact_ok in norm_pairs:
                                    pv = emb.get(p)
                                    gv = emb.get(g)
                                    if pv is None or gv is None:
                                        ok = bool(exact_ok)
                                        sim = None
                                    else:
                                        sim = float(np.dot(pv, gv))
                                        ok = bool(exact_ok or sim >= desc_thr)
                                    if ok:
                                        desc_sem_ok_total += 1.0
                                    if sim is not None:
                                        desc_sem_sim_sum_total += float(sim)
                                        desc_sem_sim_count_total += 1.0

        t_local = time.perf_counter() - t0

        try:
            import torch.distributed as dist
        except (TypeError, ValueError):
            dist = None  # type: ignore[assignment]

        world_size = 1
        if dist is not None and dist.is_available() and dist.is_initialized():
            world_size = int(dist.get_world_size())

        # Reduce sums across ranks.
        sums_t = torch.tensor(
            [
                n_samples,
                gt_total,
                pred_total,
                matched_total,
                fp_total,
                fn_total,
                gating_rejections_total,
                dropped_invalid_total,
                dropped_ambiguous_total,
                trunc_samples,
                matched_iou_sum,
                matched_iou_count,
                n_samples_valid_pred,
                n_samples_any_match,
                n_steps,
                # desc monitor
                desc_pairs_total,
                desc_exact_ok_total,
                desc_sem_ok_total,
                desc_sem_sim_sum_total,
                desc_sem_sim_count_total,
                sem_loaded_local,
            ],
            device=self.model.device,
            dtype=torch.float64,
        )
        rt_t = torch.tensor(
            [float(t_local)], device=self.model.device, dtype=torch.float64
        )
        if dist is not None and dist.is_available() and dist.is_initialized():
            dist.all_reduce(sums_t, op=dist.ReduceOp.SUM)
            # Use max runtime as the global wall time.
            dist.all_reduce(rt_t, op=dist.ReduceOp.MAX)

        (
            n_samples,
            gt_total,
            pred_total,
            matched_total,
            fp_total,
            fn_total,
            gating_rejections_total,
            dropped_invalid_total,
            dropped_ambiguous_total,
            trunc_samples,
            matched_iou_sum,
            matched_iou_count,
            n_samples_valid_pred,
            n_samples_any_match,
            n_steps,
            desc_pairs_total,
            desc_exact_ok_total,
            desc_sem_ok_total,
            desc_sem_sim_sum_total,
            desc_sem_sim_count_total,
            sem_loaded_sum,
        ) = [float(x.item()) for x in sums_t]
        runtime = float(rt_t.item())

        precision = (matched_total / pred_total) if pred_total > 0 else 0.0
        recall = (matched_total / gt_total) if gt_total > 0 else 0.0
        f1 = (
            (2.0 * precision * recall / (precision + recall))
            if (precision + recall) > 0.0
            else 0.0
        )

        def _k(suffix: str) -> str:
            return f"{metric_key_prefix}_{suffix}"

        metrics: Dict[str, float] = {}
        metrics[_k("time/runtime_s")] = float(runtime)
        if runtime > 0:
            metrics[_k("time/samples_per_s")] = float(n_samples / runtime)
            metrics[_k("time/steps_per_s")] = float(n_steps / runtime)

        metrics[_k("rollout/precision")] = float(precision)
        metrics[_k("rollout/recall")] = float(recall)
        metrics[_k("rollout/f1")] = float(f1)

        metrics[_k("rollout/pred_objects")] = float(pred_total)
        metrics[_k("rollout/gt_objects")] = float(gt_total)
        metrics[_k("rollout/matched")] = float(matched_total)
        metrics[_k("rollout/fp")] = float(fp_total)
        metrics[_k("rollout/fn")] = float(fn_total)
        metrics[_k("rollout/gating_rejections")] = float(gating_rejections_total)

        metrics[_k("rollout/parse_dropped_invalid")] = float(dropped_invalid_total)
        metrics[_k("rollout/parse_dropped_ambiguous")] = float(dropped_ambiguous_total)
        metrics[_k("rollout/parse_truncated_rate")] = (
            float(trunc_samples / n_samples) if n_samples > 0 else 0.0
        )

        metrics[_k("rollout/sample_valid_pred_rate")] = (
            float(n_samples_valid_pred / n_samples) if n_samples > 0 else 0.0
        )
        metrics[_k("rollout/sample_any_match_rate")] = (
            float(n_samples_any_match / n_samples) if n_samples > 0 else 0.0
        )

        metrics[_k("rollout/matched_maskiou_mean")] = (
            float(matched_iou_sum / matched_iou_count) if matched_iou_count > 0 else 0.0
        )

        # Desc monitor outputs (matched pairs only).
        if desc_enabled:
            metrics[_k("rollout/desc_pairs_total")] = float(desc_pairs_total)
            exact_acc = (
                float(desc_exact_ok_total / desc_pairs_total)
                if desc_pairs_total > 0
                else 1.0
            )
            metrics[_k("rollout/desc_exact_acc_on_matched")] = float(exact_acc)

            sem_enabled = bool(sem_loaded_sum >= float(world_size) - 0.5)
            metrics[_k("rollout/desc_sem_enabled")] = float(1.0 if sem_enabled else 0.0)
            if sem_enabled:
                sem_acc = (
                    float(desc_sem_ok_total / desc_pairs_total)
                    if desc_pairs_total > 0
                    else 1.0
                )
                metrics[_k("rollout/desc_sem_acc_on_matched")] = float(sem_acc)
                if desc_sem_sim_count_total > 0:
                    metrics[_k("rollout/desc_sem_sim_mean")] = float(
                        desc_sem_sim_sum_total / desc_sem_sim_count_total
                    )
                    metrics[_k("rollout/desc_sem_sim_count")] = float(
                        desc_sem_sim_count_total
                    )

        # Mirror HF Trainer.evaluate(): log metrics and trigger callbacks.
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics
        )

        if was_training:
            self.model.train()

        return metrics

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only: bool = False,
        ignore_keys: Optional[List[str]] = None,
    ):
        # Handle the case where inputs is a list of raw samples during evaluation.
        # This can happen when using identity collator or during eval with rollout matching.
        if isinstance(inputs, list):
            inputs = self._prepare_batch_inputs(inputs)

        # Call the parent prediction_step with properly formatted inputs.
        return super().prediction_step(
            model=model,
            inputs=inputs,
            prediction_loss_only=prediction_loss_only,
            ignore_keys=ignore_keys,
        )

    def training_step(self, model, inputs, *args, **kwargs):
        # When using identity collator, `inputs` is a list of raw samples.
        if not isinstance(inputs, list):
            return super().training_step(model, inputs, *args, **kwargs)

        self._validate_rollout_matching_cfg()

        prepared = self._prepare_batch_inputs(inputs)

        return super().training_step(model, prepared, *args, **kwargs)

    # ------------------------ target construction ------------------------ #
    @staticmethod
    def _bbox_corners(points_xyxy: Sequence[int]) -> np.ndarray:
        x1, y1, x2, y2 = [float(v) for v in points_xyxy]
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)

    def _build_prefix_targets(
        self,
        *,
        pred_obj: GTObject,
        gt_obj: GTObject,
        pred_coord_indices: Sequence[int],
        ot_epsilon: float,
        ot_iters: int,
        ot_cost: Literal["l1", "l2"],
    ) -> Optional[List[int]]:
        """Compute GT-aware target bins for prefix coord supervision.

        - bbox<->bbox: direct targets.
        - otherwise: Sinkhorn OT + barycentric projection (no mixture).
        """

        if pred_obj.geom_type == "bbox_2d" and gt_obj.geom_type == "bbox_2d":
            if len(gt_obj.points_norm1000) != 4 or len(pred_coord_indices) != 4:
                return None
            return [int(min(max(v, 0), 999)) for v in gt_obj.points_norm1000]

        # Build point sets for OT in norm1000 space.
        if pred_obj.geom_type == "poly":
            pts = pred_obj.points_norm1000
            if len(pts) < 6 or len(pts) % 2 != 0:
                return None
            pred_pts = np.array(list(zip(pts[0::2], pts[1::2])), dtype=np.float32)
        else:
            if len(pred_obj.points_norm1000) != 4:
                return None
            pred_pts = self._bbox_corners(pred_obj.points_norm1000)

        if gt_obj.geom_type == "poly":
            pts = gt_obj.points_norm1000
            if len(pts) < 6 or len(pts) % 2 != 0:
                return None
            gt_pts = np.array(list(zip(pts[0::2], pts[1::2])), dtype=np.float32)
        else:
            if len(gt_obj.points_norm1000) != 4:
                return None
            gt_pts = self._bbox_corners(gt_obj.points_norm1000)

        g_hat = _sinkhorn_barycentric_targets(
            pred_points=pred_pts,
            gt_points=gt_pts,
            epsilon=ot_epsilon,
            iters=ot_iters,
            cost=ot_cost,
        )

        if pred_obj.geom_type == "poly":
            flat = g_hat.reshape(-1).tolist()
            out: List[int] = []
            for v in flat:
                vi = int(round(float(v)))
                out.append(int(min(max(vi, 0), 999)))
            if len(out) != len(pred_coord_indices):
                # pred_coord_indices is 2N; ensure alignment.
                return None
            return out

        # pred is bbox: derive xyxy bbox targets from projected corners.
        x1, y1, x2, y2 = bbox_from_points(g_hat.reshape(-1).tolist())
        bbox = [x1, y1, x2, y2]
        out = []
        for v in bbox:
            vi = int(round(float(v)))
            out.append(int(min(max(vi, 0), 999)))
        if len(out) != 4 or len(pred_coord_indices) != 4:
            return None
        return out
