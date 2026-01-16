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
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask as maskUtils
from scipy.optimize import linear_sum_assignment
from swift.trainers import Seq2SeqTrainer
from swift.trainers.rlhf_trainer.utils import replace_assistant_response_with_ids
from swift.utils import get_logger, unwrap_model_for_generation

from src.common.geometry import bbox_from_points, bbox_to_quadrilateral, flatten_points
from src.coord_tokens.codec import get_coord_token_ids, token_to_int, value_in_coord_range
from src.coord_tokens.soft_ce_w1 import coord_soft_ce_w1

logger = get_logger()


GeomType = Literal["bbox_2d", "poly"]


_OBJECT_KEY_RE = re.compile(r"^object_(\d+)$")
_IM_END = "<|im_end|>"


@dataclass
class ParsedPredObject:
    key: str
    index: int
    geom_type: GeomType
    coord_token_indices: List[int]  # indices into rollout response_token_ids (assistant-local)
    value_span: Tuple[int, int]  # [char_start, char_end) span of the object value dict


@dataclass
class RolloutParseResult:
    response_token_ids: List[int]  # stripped stop tokens, full rollout (assistant-local)
    response_text: str
    prefix_token_ids: List[int]  # suffix-trimmed prefix (assistant-local, append-ready)
    prefix_text: str
    max_object_index_in_prefix: Optional[int]
    valid_objects: List[ParsedPredObject]
    dropped_invalid: int
    dropped_ambiguous: int
    truncated: bool


@dataclass
class GTObject:
    index: int
    geom_type: GeomType
    points_norm1000: List[int]  # bbox: [x1,y1,x2,y2]; poly: flat [x,y,...]
    desc: str


@dataclass
class MatchResult:
    matched_pairs: List[Tuple[int, int]]  # (pred_idx, gt_idx)
    fn_gt_indices: List[int]
    fp_pred_indices: List[int]
    gating_rejections: int


def _coerce_int(value: Any) -> Optional[int]:
    try:
        v = int(round(float(value)))
    except Exception:
        return None
    if not value_in_coord_range(v):
        return None
    return v


def _decode_pieces(tokenizer: Any, token_ids: Sequence[int]) -> List[str]:
    # Token-level decode (no cleanup) to preserve exact token boundaries.
    return [
        tokenizer.decode(
            [int(t)],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        for t in token_ids
    ]


def _build_prefix_from_char_cut(
    *,
    tokenizer: Any,
    token_ids: Sequence[int],
    pieces: Sequence[str],
    token_start_chars: Sequence[int],
    cut_char_pos: int,
) -> List[int]:
    if cut_char_pos <= 0:
        return []
    if cut_char_pos >= (token_start_chars[-1] + len(pieces[-1])):
        return list(token_ids)

    # Find token i such that token_start_chars[i] <= cut < token_start_chars[i+1]
    lo = 0
    hi = len(token_start_chars) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        start = token_start_chars[mid]
        next_start = token_start_chars[mid + 1]
        if start <= cut_char_pos < next_start:
            lo = mid
            break
        if cut_char_pos < start:
            hi = mid - 1
        else:
            lo = mid + 1
    i = lo
    start = token_start_chars[i]
    offset = max(0, min(int(cut_char_pos - start), len(pieces[i])))

    prefix = list(token_ids[:i])
    if offset == 0:
        return prefix
    if offset == len(pieces[i]):
        prefix.append(int(token_ids[i]))
        return prefix

    piece_prefix = pieces[i][:offset]
    # Replace ONLY the final token with a shorter tokenization (allowed by spec).
    new_tail = tokenizer.encode(piece_prefix, add_special_tokens=False)
    decoded = tokenizer.decode(
        new_tail, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    if decoded != piece_prefix:
        raise ValueError(
            "Failed to retokenize a token-internal cut boundary. "
            f"expected={piece_prefix!r} got={decoded!r}"
        )
    prefix.extend([int(t) for t in new_tail])
    return prefix


def _scan_rollout_tokens(
    *,
    tokenizer: Any,
    response_token_ids: List[int],
    coord_id_set: set[int],
) -> Tuple[List[ParsedPredObject], Optional[int], int, bool, int]:
    """Token-aligned rollout scanner.

    Returns:
      (objects_raw, max_object_index, first_open_brace_pos, truncated, last_object_end_pos)

    Notes:
    - Objects are returned in appearance order.
    - Validation is performed later using the captured value_span.
    """

    pieces = _decode_pieces(tokenizer, response_token_ids)
    token_start_chars: List[int] = []
    cursor = 0
    for p in pieces:
        token_start_chars.append(cursor)
        cursor += len(p)
    text = "".join(pieces)

    brace_depth = 0
    bracket_depth = 0
    in_string = False
    escape = False
    expecting_key: Dict[int, bool] = {}

    current_string: List[str] = []
    last_key: Optional[str] = None

    pending_object_key: Optional[Tuple[str, int]] = None
    current_object: Optional[ParsedPredObject] = None
    # Geometry capture state (per current_object)
    capture_active = False
    capture_target_depth: Optional[int] = None

    objects: List[ParsedPredObject] = []
    max_index: Optional[int] = None
    first_open_brace_pos: int = -1
    last_complete_object_end: int = -1

    # Map from token index -> current capture_active state is handled by scanning
    # per token; coord tokens do not contain structural characters, so the state at
    # token-start is sufficient.

    global_char_pos = 0
    for tok_i, (tok_id, piece) in enumerate(zip(response_token_ids, pieces)):
        # Coord-token capture happens at token granularity.
        if capture_active and int(tok_id) in coord_id_set and current_object is not None:
            current_object.coord_token_indices.append(int(tok_i))

        for ch in piece:
            if in_string:
                if escape:
                    escape = False
                    current_string.append(ch)
                elif ch == "\\":
                    escape = True
                    current_string.append(ch)
                elif ch == '"':
                    in_string = False
                    s = "".join(current_string)
                    current_string = []
                    if expecting_key.get(brace_depth, False) and bracket_depth == 0:
                        last_key = s
                        if brace_depth == 1:
                            m = _OBJECT_KEY_RE.match(s)
                            if m:
                                n = int(m.group(1))
                                if max_index is None or n > max_index:
                                    max_index = n
                                pending_object_key = (s, n)
                        # For brace_depth==2, last_key is used to arm geometry capture.
                else:
                    current_string.append(ch)
                global_char_pos += 1
                continue

            # Not in string
            if ch == '"':
                in_string = True
                escape = False
                current_string = []
                global_char_pos += 1
                continue

            if ch == "{":
                brace_depth += 1
                expecting_key[brace_depth] = True
                if brace_depth == 1 and first_open_brace_pos < 0:
                    first_open_brace_pos = global_char_pos + 1
                elif brace_depth == 2:
                    if pending_object_key is None:
                        # Nested dict without an object key -> invalid; keep scanning.
                        current_object = None
                    else:
                        key, n = pending_object_key
                        pending_object_key = None
                        current_object = ParsedPredObject(
                            key=key,
                            index=n,
                            geom_type="bbox_2d",  # placeholder until validated
                            coord_token_indices=[],
                            value_span=(global_char_pos, -1),
                        )
                        objects.append(current_object)
                        last_key = None
                        capture_active = False
                        capture_target_depth = None
                elif brace_depth > 2:
                    # Nested objects are unsupported in strict parsing.
                    # Keep scanning to find stable boundaries, but we will drop later.
                    pass
                global_char_pos += 1
                continue

            if ch == "}":
                if brace_depth == 2 and current_object is not None:
                    # End of current object value dict.
                    current_object.value_span = (current_object.value_span[0], global_char_pos + 1)
                    last_complete_object_end = global_char_pos + 1
                    current_object = None
                    last_key = None
                    capture_active = False
                    capture_target_depth = None
                if brace_depth > 0:
                    expecting_key.pop(brace_depth, None)
                    brace_depth -= 1
                global_char_pos += 1
                continue

            if ch == "[":
                bracket_depth += 1
                if (
                    brace_depth == 2
                    and current_object is not None
                    and last_key in {"bbox_2d", "poly"}
                    and bracket_depth >= 1
                ):
                    # Arm capture on first array after the geometry key.
                    if capture_target_depth is None:
                        capture_active = True
                        capture_target_depth = bracket_depth - 1
                global_char_pos += 1
                continue

            if ch == "]":
                if bracket_depth > 0:
                    bracket_depth -= 1
                if capture_active and capture_target_depth is not None:
                    if bracket_depth <= capture_target_depth:
                        capture_active = False
                        capture_target_depth = None
                global_char_pos += 1
                continue

            if ch == ",":
                if brace_depth >= 1 and bracket_depth == 0:
                    expecting_key[brace_depth] = True
                global_char_pos += 1
                continue

            if ch == ":":
                if brace_depth >= 1 and bracket_depth == 0:
                    expecting_key[brace_depth] = False
                global_char_pos += 1
                continue

            global_char_pos += 1

    truncated = False
    if brace_depth != 0 or bracket_depth != 0 or in_string:
        truncated = True

    return objects, max_index, first_open_brace_pos, truncated, last_complete_object_end


def _validate_objects_strict(
    *,
    tokenizer: Any,
    response_text: str,
    objects_raw: Sequence[ParsedPredObject],
    prefix_char_end: int,
) -> Tuple[List[ParsedPredObject], int]:
    """Strict validation: drop malformed objects (no repair)."""

    valid: List[ParsedPredObject] = []
    dropped = 0
    for obj in objects_raw:
        start, end = obj.value_span
        if end <= 0 or end > prefix_char_end:
            dropped += 1
            continue
        snippet = response_text[start:end]
        try:
            parsed = json.loads(snippet)
        except Exception:
            dropped += 1
            continue
        if not isinstance(parsed, dict):
            dropped += 1
            continue
        desc = parsed.get("desc")
        if not isinstance(desc, str) or not desc.strip():
            dropped += 1
            continue
        geom_keys = [k for k in ("bbox_2d", "poly") if k in parsed]
        if len(geom_keys) != 1:
            dropped += 1
            continue
        geom_key = geom_keys[0]

        # Reject unexpected keys to keep strictness (no nested/unexpected).
        allowed = {"desc", "bbox_2d", "poly"}
        if any(k not in allowed for k in parsed.keys()):
            dropped += 1
            continue

        # Geometry values must be coord tokens (strict; no ints in rollout-matching).
        flat = flatten_points(parsed.get(geom_key))
        if flat is None or len(flat) % 2 != 0:
            dropped += 1
            continue
        token_bins: List[int] = []
        ok = True
        for v in flat:
            if not isinstance(v, str):
                ok = False
                break
            try:
                token_bins.append(int(token_to_int(v)))
            except Exception:
                ok = False
                break
        if not ok:
            dropped += 1
            continue

        # Ensure we captured coord-token indices and they match geometry arity.
        coord_idx = list(obj.coord_token_indices)
        if geom_key == "bbox_2d":
            if len(coord_idx) != 4 or len(token_bins) != 4:
                dropped += 1
                continue
        else:  # poly
            if len(coord_idx) < 6 or (len(coord_idx) % 2 != 0) or len(token_bins) != len(coord_idx):
                dropped += 1
                continue

        valid.append(
            ParsedPredObject(
                key=obj.key,
                index=int(obj.index),
                geom_type=geom_key,  # type: ignore[assignment]
                coord_token_indices=coord_idx,
                value_span=obj.value_span,
            )
        )
    return valid, dropped


def parse_rollout_for_matching(
    *,
    tokenizer: Any,
    response_token_ids: List[int],
) -> RolloutParseResult:
    coord_token_ids = get_coord_token_ids(tokenizer)
    coord_id_set = set(int(t) for t in coord_token_ids if int(t) >= 0)

    # Decode full response text (token-aligned, no cleanup).
    pieces = _decode_pieces(tokenizer, response_token_ids)
    token_start_chars: List[int] = []
    cursor = 0
    for p in pieces:
        token_start_chars.append(cursor)
        cursor += len(p)
    response_text = "".join(pieces)

    # Treat <|im_end|> as a hard stop (common in current offline rollouts).
    # This is suffix-only trimming; tokens before the cut remain unchanged, except for a
    # possible token-internal cut on the final token.
    stop_pos = response_text.find(_IM_END)
    if stop_pos >= 0:
        response_token_ids = _build_prefix_from_char_cut(
            tokenizer=tokenizer,
            token_ids=response_token_ids,
            pieces=pieces,
            token_start_chars=token_start_chars,
            cut_char_pos=int(stop_pos),
        )
        pieces = _decode_pieces(tokenizer, response_token_ids)
        token_start_chars = []
        cursor = 0
        for p in pieces:
            token_start_chars.append(cursor)
            cursor += len(p)
        response_text = "".join(pieces)

    objects_raw, max_index, first_open_brace_pos, truncated, last_obj_end = _scan_rollout_tokens(
        tokenizer=tokenizer, response_token_ids=response_token_ids, coord_id_set=coord_id_set
    )

    # If completely malformed (no top-level '{'), fall back to empty prefix so FN append can proceed.
    if first_open_brace_pos < 0:
        prefix_token_ids = [int(t) for t in tokenizer.encode("{", add_special_tokens=False)]
        prefix_text = tokenizer.decode(
            prefix_token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        return RolloutParseResult(
            response_token_ids=list(response_token_ids),
            response_text=response_text,
            prefix_token_ids=prefix_token_ids,
            prefix_text=prefix_text,
            max_object_index_in_prefix=None,
            valid_objects=[],
            dropped_invalid=0,
            dropped_ambiguous=0,
            truncated=True,
        )

    # Determine a safe prefix cut: keep up to last complete object, or just "{".
    if last_obj_end > 0:
        cut_char = last_obj_end
    elif first_open_brace_pos > 0:
        cut_char = first_open_brace_pos
    else:
        cut_char = 0

    prefix_token_ids = _build_prefix_from_char_cut(
        tokenizer=tokenizer,
        token_ids=response_token_ids,
        pieces=pieces,
        token_start_chars=token_start_chars,
        cut_char_pos=cut_char,
    )
    prefix_text = tokenizer.decode(
        prefix_token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )

    valid_objects, dropped_invalid = _validate_objects_strict(
        tokenizer=tokenizer,
        response_text=response_text,
        objects_raw=objects_raw,
        prefix_char_end=cut_char,
    )

    # In this MVP implementation, coord-slot alignment ambiguity is treated as invalid.
    dropped_ambiguous = 0

    # Best-effort: max index only among objects that survived the prefix cut.
    max_in_prefix = None
    for obj in valid_objects:
        if max_in_prefix is None or obj.index > max_in_prefix:
            max_in_prefix = obj.index

    return RolloutParseResult(
        response_token_ids=list(response_token_ids),
        response_text=response_text,
        prefix_token_ids=prefix_token_ids,
        prefix_text=prefix_text,
        max_object_index_in_prefix=max_in_prefix,
        valid_objects=valid_objects,
        dropped_invalid=int(dropped_invalid),
        dropped_ambiguous=int(dropped_ambiguous),
        truncated=bool(truncated),
    )


def _points_from_coord_tokens(
    *,
    response_token_ids: Sequence[int],
    coord_token_indices: Sequence[int],
    coord_id_to_bin: Mapping[int, int],
) -> Optional[List[int]]:
    out: List[int] = []
    for idx in coord_token_indices:
        if idx < 0 or idx >= len(response_token_ids):
            return None
        tok_id = int(response_token_ids[int(idx)])
        bin_idx = coord_id_to_bin.get(tok_id)
        if bin_idx is None:
            return None
        out.append(int(bin_idx))
    return out


def _bbox_xyxy_from_norm(points: Sequence[int], kind: GeomType) -> Tuple[float, float, float, float]:
    if kind == "bbox_2d":
        if len(points) != 4:
            return 0.0, 0.0, 0.0, 0.0
        x1, y1, x2, y2 = [float(v) for v in points]
        return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
    x1, y1, x2, y2 = bbox_from_points([float(v) for v in points])
    return float(x1), float(y1), float(x2), float(y2)


def _bbox_iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0.0 else 0.0


def _mask_iou_norm1000(
    *,
    pred_kind: GeomType,
    pred_points: Sequence[int],
    gt_kind: GeomType,
    gt_points: Sequence[int],
    resolution: int,
) -> float:
    """maskIoU in norm1000 space on a virtual RxR canvas."""
    r = int(resolution)
    if r <= 0:
        return 0.0

    def _clamp01k(values: Sequence[int]) -> List[float]:
        return [float(min(max(int(v), 0), 999)) for v in values]

    def _project(values: Sequence[float]) -> List[float]:
        # Project [0,999] -> [0,R) continuous coordinates.
        scale = float(r) / 1000.0
        return [float(v) * scale for v in values]

    def _as_poly(kind: GeomType, pts: Sequence[int]) -> List[float]:
        if kind == "bbox_2d":
            if len(pts) != 4:
                return []
            x1, y1, x2, y2 = [float(v) for v in pts]
            quad = bbox_to_quadrilateral([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)])
            return [float(v) for v in quad]
        return [float(v) for v in pts]

    p_poly = _project(_clamp01k(_as_poly(pred_kind, pred_points)))
    g_poly = _project(_clamp01k(_as_poly(gt_kind, gt_points)))
    if len(p_poly) < 6 or len(g_poly) < 6:
        return 0.0

    try:
        rle_p = maskUtils.frPyObjects([p_poly], r, r)
        rle_g = maskUtils.frPyObjects([g_poly], r, r)
        if isinstance(rle_p, list):
            rle_p = maskUtils.merge(rle_p)
        if isinstance(rle_g, list):
            rle_g = maskUtils.merge(rle_g)
        ious = maskUtils.iou([rle_p], [rle_g], [0])
        return float(ious[0][0]) if getattr(ious, "size", 0) else 0.0
    except Exception:
        return 0.0


def hungarian_match_maskiou(
    *,
    preds: Sequence[GTObject],
    gts: Sequence[GTObject],
    top_k: int,
    gate_threshold: float,
    mask_resolution: int,
    fp_cost: float,
    fn_cost: float,
) -> MatchResult:
    pred_n = len(preds)
    gt_n = len(gts)
    if pred_n == 0:
        return MatchResult(matched_pairs=[], fn_gt_indices=list(range(gt_n)), fp_pred_indices=[], gating_rejections=0)
    if gt_n == 0:
        return MatchResult(matched_pairs=[], fn_gt_indices=[], fp_pred_indices=list(range(pred_n)), gating_rejections=0)

    k = max(1, int(top_k))
    gate = float(gate_threshold)
    inf = 1e6

    gt_boxes = [_bbox_xyxy_from_norm(gt.points_norm1000, gt.geom_type) for gt in gts]
    pred_boxes = [_bbox_xyxy_from_norm(pr.points_norm1000, pr.geom_type) for pr in preds]

    # Candidate pruning per pred.
    cand: List[List[int]] = []
    for pb in pred_boxes:
        ious = [(_bbox_iou_xyxy(pb, gb), j) for j, gb in enumerate(gt_boxes)]
        ious.sort(key=lambda t: (-t[0], t[1]))
        best = [j for _, j in ious[:k]]
        if ious and ious[0][0] <= 0.0:
            # Fallback: center distance.
            pcx = 0.5 * (pb[0] + pb[2])
            pcy = 0.5 * (pb[1] + pb[3])
            dists = []
            for j, gb in enumerate(gt_boxes):
                gcx = 0.5 * (gb[0] + gb[2])
                gcy = 0.5 * (gb[1] + gb[3])
                d = (pcx - gcx) ** 2 + (pcy - gcy) ** 2
                dists.append((float(d), j))
            dists.sort(key=lambda t: (t[0], t[1]))
            best = [j for _, j in dists[:k]]
        cand.append(best)

    gating_rejections = 0
    cost_pg = np.full((pred_n, gt_n), inf, dtype=np.float64)
    for i, pr in enumerate(preds):
        for j in cand[i]:
            iou = _mask_iou_norm1000(
                pred_kind=pr.geom_type,
                pred_points=pr.points_norm1000,
                gt_kind=gts[j].geom_type,
                gt_points=gts[j].points_norm1000,
                resolution=mask_resolution,
            )
            if iou < gate:
                gating_rejections += 1
                continue
            cost_pg[i, j] = 1.0 - float(iou)

    # Dummy-augmented square matrix.
    n = pred_n + gt_n
    cost = np.full((n, n), 0.0, dtype=np.float64)
    cost[:pred_n, :gt_n] = cost_pg
    cost[:pred_n, gt_n:] = float(fp_cost)
    cost[pred_n:, :gt_n] = float(fn_cost)
    cost[pred_n:, gt_n:] = 0.0

    row_ind, col_ind = linear_sum_assignment(cost)
    assign = {int(r): int(c) for r, c in zip(row_ind, col_ind)}

    matched_pairs: List[Tuple[int, int]] = []
    fp_preds: List[int] = []
    matched_gt: set[int] = set()

    for i in range(pred_n):
        c = assign.get(i)
        if c is None:
            fp_preds.append(i)
            continue
        if c < gt_n and cost_pg[i, c] < inf * 0.5:
            matched_pairs.append((i, c))
            matched_gt.add(c)
        else:
            fp_preds.append(i)

    fn_gts = [j for j in range(gt_n) if j not in matched_gt]
    return MatchResult(
        matched_pairs=matched_pairs,
        fn_gt_indices=fn_gts,
        fp_pred_indices=fp_preds,
        gating_rejections=int(gating_rejections),
    )


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
    objs: List[GTObject] = []
    for key, entry in payload.items():
        if not isinstance(key, str):
            continue
        m = _OBJECT_KEY_RE.match(key)
        if not m:
            continue
        idx = int(m.group(1))
        if not isinstance(entry, Mapping):
            continue
        desc = entry.get("desc")
        if not isinstance(desc, str) or not desc.strip():
            continue
        geom_keys = [k for k in ("bbox_2d", "poly") if k in entry and entry[k] is not None]
        if len(geom_keys) != 1:
            continue
        geom_key = geom_keys[0]
        raw_pts = flatten_points(entry.get(geom_key))
        if raw_pts is None or len(raw_pts) % 2 != 0:
            continue
        pts: List[int] = []
        ok = True
        for v in raw_pts:
            if isinstance(v, str) and v.startswith("<|coord_"):
                try:
                    pts.append(int(token_to_int(v)))
                except Exception:
                    ok = False
                    break
            else:
                vi = _coerce_int(v)
                if vi is None:
                    ok = False
                    break
                pts.append(int(vi))
        if not ok:
            continue
        if geom_key == "bbox_2d" and len(pts) != 4:
            continue
        if geom_key == "poly" and (len(pts) < 6 or len(pts) % 2 != 0):
            continue
        objs.append(GTObject(index=idx, geom_type=geom_key, points_norm1000=pts, desc=desc.strip()))
    objs.sort(key=lambda o: o.index)
    return objs


def _serialize_append_fragment(
    *,
    fn_objects: Sequence[GTObject],
    start_index: int,
    prefix_text: str,
) -> str:
    # Determine last non-whitespace char.
    tail = prefix_text.rstrip()
    if not tail:
        raise ValueError("empty rollout prefix is not append-ready")
    last = tail[-1]
    if last not in {"{", "}", ","}:
        raise ValueError(f"rollout prefix is not append-ready; last_char={last!r}")

    if not fn_objects:
        if last == ",":
            raise ValueError("rollout prefix ends with ',' but FN set is empty; invalid JSON")
        return "}"

    leading = ""
    if last == "}":
        leading = ", "

    entries: List[str] = []
    n = int(start_index)
    for obj in fn_objects:
        payload: Dict[str, Any] = {"desc": obj.desc}
        if obj.geom_type == "bbox_2d":
            if len(obj.points_norm1000) != 4:
                continue
            payload["bbox_2d"] = [f"<|coord_{int(v)}|>" for v in obj.points_norm1000]
        else:
            pts = obj.points_norm1000
            pairs = [[f"<|coord_{int(pts[i])}|>", f"<|coord_{int(pts[i + 1])}|>"] for i in range(0, len(pts), 2)]
            payload["poly"] = pairs
        entries.append(f"\"object_{n}\": {json.dumps(payload, ensure_ascii=False, separators=(', ', ': '))}")
        n += 1

    return leading + ", ".join(entries) + "}"


def _find_desc_value_char_spans(text: str) -> List[Tuple[int, int]]:
    """Return [start,end) spans of the *value content* inside `"desc": "<VALUE>"`.

    Stage_2 rollout-matching intentionally ignores desc value supervision to avoid
    amplifying GT noise; this helper supports masking those tokens from CE.
    """
    spans: List[Tuple[int, int]] = []
    i = 0
    needle = '"desc"'
    n = len(text)
    while i < n:
        k = text.find(needle, i)
        if k < 0:
            break
        j = k + len(needle)
        # Skip whitespace, then require ':'.
        while j < n and text[j].isspace():
            j += 1
        if j >= n or text[j] != ":":
            i = k + 1
            continue
        j += 1
        while j < n and text[j].isspace():
            j += 1
        # Require opening quote of the string value.
        if j >= n or text[j] != '"':
            i = k + 1
            continue
        j += 1
        start = j
        esc = False
        while j < n:
            ch = text[j]
            if esc:
                esc = False
                j += 1
                continue
            if ch == "\\":
                esc = True
                j += 1
                continue
            if ch == '"':
                spans.append((start, j))
                j += 1
                break
            j += 1
        i = max(j, k + 1)
    return spans


def _find_desc_value_token_positions(*, tokenizer: Any, token_ids: Sequence[int]) -> List[int]:
    """Return token indices (0-based, relative to token_ids) overlapping desc-value spans."""
    ids = [int(t) for t in token_ids]
    pieces = _decode_pieces(tokenizer, ids)
    token_start_chars: List[int] = []
    cursor = 0
    for p in pieces:
        token_start_chars.append(cursor)
        cursor += len(p)
    text = "".join(pieces)
    spans = _find_desc_value_char_spans(text)
    if not spans:
        return []
    out: List[int] = []
    for ti, (start, piece) in enumerate(zip(token_start_chars, pieces)):
        end = start + len(piece)
        for s, e in spans:
            if start < e and end > s:
                out.append(int(ti))
                break
    return out


def _coord_vocab_gate_loss(
    *, logits_full: torch.Tensor, logits_coord: torch.Tensor, temperature: float
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    full = torch.nan_to_num(logits_full.float(), nan=0.0, posinf=1e4, neginf=-1e4).clamp(
        min=-1e4, max=1e4
    ) / float(temperature)
    coord = torch.nan_to_num(logits_coord.float(), nan=0.0, posinf=1e4, neginf=-1e4).clamp(
        min=-1e4, max=1e4
    ) / float(temperature)
    lse_all = torch.logsumexp(full, dim=-1)
    lse_coord = torch.logsumexp(coord, dim=-1)
    loss = (lse_all - lse_coord).clamp(min=0.0)
    return torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=0.0)


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
    tail_start = max(1, min(int(tail_start), seq_len))  # p-1 must be valid for logits_next gather
    tail_end = max(tail_start, min(int(tail_end), seq_len))

    ignore_set = set(int(i) for i in (tail_ignore_pos or []) if int(i) >= 0)
    for p in range(tail_start, tail_end):
        if p < assistant_start or p >= assistant_end:
            raise ValueError(f"tail supervision index out of assistant span: p={p} span=[{assistant_start},{assistant_end})")
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
        raise ValueError("prefix_coord_pos and prefix_coord_target_bins must have identical length")
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


class RolloutMatchingSFTTrainer(Seq2SeqTrainer):
    """Rollout-matching (stage_2) trainer variant."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._coord_token_ids: Optional[List[int]] = None
        self._coord_id_to_bin: Optional[Dict[int, int]] = None
        self._debug_dump_count: int = 0

        # Mutable config injected by src/sft.py after construction.
        self.rollout_matching_cfg: Mapping[str, Any] = {}

    # ------------------------ config helpers ------------------------ #
    def _cfg(self, key: str, default: Any) -> Any:
        try:
            cfg = self.rollout_matching_cfg
            if isinstance(cfg, Mapping) and key in cfg:
                return cfg[key]
        except Exception:
            pass
        return default

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
        images = sample.get("images") if isinstance(sample.get("images"), list) else None
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
        ids = get_coord_token_ids(tok)
        self._coord_token_ids = [int(i) for i in ids]
        self._coord_id_to_bin = {int(tok_id): int(i) for i, tok_id in enumerate(ids)}
        return self._coord_token_ids

    def _coord_id_map(self) -> Dict[int, int]:
        if self._coord_id_to_bin is None:
            _ = self._get_coord_token_ids()
        return self._coord_id_to_bin or {}

    # ------------------------ rollout + batch prep ------------------------ #
    @torch.no_grad()
    def _rollout_one(
        self, sample: Mapping[str, Any]
    ) -> Tuple[List[int], str, str, List[int]]:
        """Generate a single rollout response.

        Returns:
          (response_token_ids, decoded_text, decode_mode, prompt_token_ids)
        """
        template = self.template
        tok = template.tokenizer
        decode_mode = str(self._cfg("decode_mode", "greedy")).lower()
        max_new_tokens = int(self._cfg("max_new_tokens", 512))
        num_beams = int(self._cfg("num_beams", 1))
        temperature = float(self._cfg("temperature", 0.0))

        with template.generate_context():
            encoded = template.encode(dict(sample), return_length=True)
        batch = template.data_collator([encoded])
        from swift.llm import to_device

        batch = to_device(batch, self.model.device)
        prompt_len = int(batch["input_ids"].shape[1])
        prompt_ids = batch["input_ids"][0].detach().cpu().tolist()

        # Build GenerationConfig from model defaults.
        gen_cfg = getattr(self.model, "generation_config", None)
        if gen_cfg is None:
            from transformers import GenerationConfig

            gen_cfg = GenerationConfig()
        gen_cfg = gen_cfg.clone()
        gen_cfg.max_new_tokens = max_new_tokens
        gen_cfg.do_sample = bool(temperature > 0.0)
        gen_cfg.temperature = max(1e-4, temperature) if gen_cfg.do_sample else 1.0
        if decode_mode == "beam":
            gen_cfg.num_beams = max(1, num_beams)
            gen_cfg.num_return_sequences = max(1, int(self._cfg("num_return_sequences", gen_cfg.num_beams)))
        else:
            gen_cfg.num_beams = 1
            gen_cfg.num_return_sequences = 1

        model_inputs = {k: v for k, v in batch.items() if k != "labels"}
        model_inputs.pop("position_ids", None)
        with unwrap_model_for_generation(
            self.model_wrapped,
            self.accelerator,
            gather_deepspeed3_params=getattr(self.args, "ds3_gather_for_generation", False),
        ) as unwrapped:
            unwrapped.eval()
            with template.generate_context():
                if getattr(self.model, "model_meta", None) is not None and self.model.model_meta.is_multimodal:
                    _, model_inputs = template.pre_forward_hook(unwrapped, None, model_inputs)
                out = template.generate(
                    unwrapped,
                    **model_inputs,
                    generation_config=gen_cfg,
                    return_dict_in_generate=True,
                )
            unwrapped.train()

        sequences = out.sequences  # [R, T]
        if sequences.ndim != 2:
            raise ValueError("unexpected generate output shape")
        if sequences.shape[0] > 1 and hasattr(out, "sequences_scores") and out.sequences_scores is not None:
            best = int(torch.argmax(out.sequences_scores).item())
        else:
            best = 0
        seq = sequences[best]
        resp_ids = seq[prompt_len:].tolist()
        resp_ids = template.skip_stop_tokens(resp_ids, is_finished=True)
        text = template.decode(resp_ids, is_finished=True, first_token=True, clean_up_tokenization_spaces=False)
        return resp_ids, text, decode_mode, [int(t) for t in prompt_ids]

    def _prepare_batch_inputs(self, inputs: List[Mapping[str, Any]]) -> Dict[str, Any]:
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

        encoded_batch: List[Dict[str, Any]] = []
        meta: List[Dict[str, Any]] = []

        for sample in inputs:
            if "messages" not in sample:
                raise ValueError("rollout-matching requires 'messages' in dataset samples")

            # 1) Rollout
            resp_ids, resp_text, decode_mode, prompt_ids = self._rollout_one(sample)

            # 2) Strict token-aligned parsing + suffix-only prefix trimming
            parse = parse_rollout_for_matching(tokenizer=tok, response_token_ids=resp_ids)
            self._maybe_debug_dump_parse_failure(
                sample=sample,
                response_text=resp_text,
                prefix_text=parse.prefix_text,
                dropped_invalid=int(parse.dropped_invalid),
                dropped_ambiguous=int(parse.dropped_ambiguous),
                truncated=bool(parse.truncated),
                decode_mode=str(decode_mode),
            )

            # 3) Extract predicted objects (valid only) and map coord tokens -> bins
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
                preds.append(GTObject(index=int(pobj.index), geom_type=pobj.geom_type, points_norm1000=pts, desc=""))
                pred_meta.append(pobj)

            # 4) Extract GT objects and match
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
            # 4.1) Build self-context supervision targets for matched pairs.
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
                except Exception:
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

            fn_gt_indices_final = [i for i in range(len(gts)) if i not in matched_gt_for_supervision]
            fn_objs = [gts[i] for i in fn_gt_indices_final]

            # 5) Serialize append fragment (mandatory FN append) and build Y_train ids
            max_idx = parse.max_object_index_in_prefix
            start_idx = (max_idx + 1) if max_idx is not None else 1
            append_text = _serialize_append_fragment(
                fn_objects=fn_objs, start_index=start_idx, prefix_text=parse.prefix_text
            )
            append_ids = tok.encode(append_text, add_special_tokens=False)
            # Ignore desc value tokens in the appended tail for CE (GT desc can be noisy).
            tail_ignore_pos = _find_desc_value_token_positions(tokenizer=tok, token_ids=append_ids)
            y_train_ids = list(parse.prefix_token_ids) + [int(t) for t in append_ids]

            # 6) Teacher-forced encoding using the exact token ids (no re-tokenization)
            data_for_encode = dict(sample)
            # Deepcopy messages to avoid in-place mutations across dataloader workers.
            messages = json.loads(json.dumps(sample["messages"]))
            data_for_encode["messages"] = replace_assistant_response_with_ids(messages, y_train_ids)
            encoded = template.encode(data_for_encode, return_length=True)
            encoded_batch.append(encoded)

            meta.append(
                {
                    "prompt_len": int(len(prompt_ids)),
                    "prompt_ids": prompt_ids,
                    "prefix_len": int(len(parse.prefix_token_ids)),
                    "train_len": int(len(y_train_ids)),
                    "decode_mode": decode_mode,
                    "parse_dropped_invalid": int(parse.dropped_invalid),
                    "parse_dropped_ambiguous": int(parse.dropped_ambiguous),
                    "parse_truncated": bool(parse.truncated),
                    "valid_pred_objects": int(len(parse.valid_objects)),
                    "matched_pairs": match.matched_pairs,
                    "matched_for_supervision": int(len(matched_gt_for_supervision)),
                    "gt_objects": int(len(gts)),
                    "fn_count": int(len(fn_objs)),
                    "gating_rejections": int(match.gating_rejections),
                    "excluded_from_supervision": int(excluded),
                    "prefix_coord_pos": prefix_pos,
                    "prefix_coord_target_bins": prefix_target_bins,
                    "tail_ignore_pos": tail_ignore_pos,
                }
            )

        from swift.llm import to_device

        batch = to_device(template.data_collator(encoded_batch), self.model.device)
        batch["_rollout_matching_meta"] = meta
        return batch

    # ------------------------ loss ------------------------ #
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        meta = inputs.pop("_rollout_matching_meta", None)
        if not isinstance(meta, list):
            raise ValueError("rollout-matching trainer requires _rollout_matching_meta")

        # Always compute logits; do not rely on model.loss (we need custom masking).
        # NOTE: ms-swift's Seq2SeqTrainer/_prepare_inputs may inject helper keys
        # like compute_loss_func/loss_scale/channel. Strip them before model forward.
        ignored_keys = {"labels", "compute_loss_func", "loss_scale", "text_position_ids", "channel"}
        inputs_for_model = {k: v for k, v in inputs.items() if k not in ignored_keys}
        outputs = model(**inputs_for_model)
        logits = outputs.logits
        if logits is None:
            raise ValueError("model did not return logits")

        bsz, seq_len, vocab = logits.shape
        coord_token_ids = self._get_coord_token_ids()
        coord_ids_t = torch.tensor(coord_token_ids, device=logits.device, dtype=torch.long)
        coord_id_set = set(int(i) for i in coord_token_ids if int(i) >= 0)
        coord_id_to_bin = self._coord_id_map()

        # Build custom labels for CE (tail non-coord tokens only) and collect
        # coord supervision targets (prefix self-context + tail GT).
        input_ids = inputs["input_ids"]
        labels_masked = torch.full_like(input_ids, -100)

        supervised_batch: List[int] = []
        supervised_pos: List[int] = []  # full-seq positions (token positions)
        supervised_bin: List[int] = []  # target bins in 0..999
        supervised_is_prefix: List[bool] = []

        for b in range(bsz):
            m = meta[b]
            prompt_len = int(m["prompt_len"])
            prefix_len = int(m["prefix_len"])
            train_len = int(m["train_len"])
            prompt_ids = m.get("prompt_ids")

            # Sanity: prompt prefix matches (avoid silent misalignment).
            if prompt_len <= 0 or prompt_len >= seq_len:
                raise ValueError(f"invalid prompt_len={prompt_len} for seq_len={seq_len}")
            if isinstance(prompt_ids, list):
                teacher_prefix = input_ids[b, :prompt_len].detach().cpu().tolist()
                if teacher_prefix != prompt_ids:
                    raise ValueError("prompt tokenization mismatch between generation and teacher-forced encoding")

            prefix_pos_local = m.get("prefix_coord_pos") or []
            prefix_bins = m.get("prefix_coord_target_bins") or []
            tail_ignore_pos = m.get("tail_ignore_pos") or []
            labels_1d, cpos, cbins, cis_prefix = _build_labels_and_coord_targets_for_sample(
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
            labels_masked[b] = labels_1d
            for p, tbin, is_pref in zip(cpos, cbins, cis_prefix):
                supervised_batch.append(b)
                supervised_pos.append(int(p))
                supervised_bin.append(int(tbin))
                supervised_is_prefix.append(bool(is_pref))

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
            bin_t = torch.tensor(supervised_bin, device=logits.device, dtype=torch.long).clamp(min=0, max=999)
            is_prefix_t = torch.tensor(supervised_is_prefix, device=logits.device, dtype=torch.bool)

            logit_pos = (pos_t - 1).clamp(min=0, max=seq_len - 2)
            logits_full = logits_next[b_t, logit_pos, :]  # [N, V]
            logits_coord = logits_full.index_select(-1, coord_ids_t)  # [N, 1000]

            # Loss weights come from rollout cfg, falling back to coord_soft_ce_w1_cfg.
            cfg = getattr(self, "coord_soft_ce_w1_cfg", None)
            sigma = float(self._cfg("target_sigma", float(getattr(cfg, "target_sigma", 2.0))))
            truncate = self._cfg("target_truncate", getattr(cfg, "target_truncate", None))
            temperature = float(self._cfg("temperature_coord", float(getattr(cfg, "temperature", 1.0))))
            soft_w = float(self._cfg("soft_ce_weight", float(getattr(cfg, "soft_ce_weight", 1.0))))
            w1_w = float(self._cfg("w1_weight", float(getattr(cfg, "w1_weight", 1.0))))
            gate_w = float(self._cfg("gate_weight", float(getattr(cfg, "gate_weight", 0.0))))

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
                _coord_vocab_gate_loss(logits_full=logits_full, logits_coord=logits_coord, temperature=temperature)
                if gate_w != 0.0
                else logits_full.new_zeros((logits_full.shape[0],), dtype=torch.float32)
            )

            per_tok = soft_w * out.soft_ce_per_token + w1_w * out.w1_per_token + gate_w * gate_per
            denom = per_tok.numel()
            if denom > 0:
                coord_loss = per_tok.mean().to(dtype=ce_loss.dtype)
            if is_prefix_t.any().item():
                prefix_coord_mean = per_tok[is_prefix_t].mean().to(dtype=ce_loss.dtype)
            if (~is_prefix_t).any().item():
                tail_coord_mean = per_tok[~is_prefix_t].mean().to(dtype=ce_loss.dtype)

        total = ce_loss + coord_loss

        # Lightweight counters (no geometry metric logging).
        try:
            gt_total = float(sum(int(m.get("gt_objects", 0)) for m in meta))
            matched_total = float(sum(int(m.get("matched_for_supervision", 0)) for m in meta))
            self.log(
                {
                    f"rollout/parse_dropped_invalid": float(
                        sum(int(m.get("parse_dropped_invalid", 0)) for m in meta)
                    ),
                    f"rollout/parse_dropped_ambiguous": float(
                        sum(int(m.get("parse_dropped_ambiguous", 0)) for m in meta)
                    ),
                    f"rollout/parse_truncated": float(
                        sum(1 for m in meta if m.get("parse_truncated"))
                    ),
                    f"rollout/fn_appended": float(sum(int(m.get("fn_count", 0)) for m in meta)),
                    f"rollout/gating_rejections": float(
                        sum(int(m.get("gating_rejections", 0)) for m in meta)
                    ),
                    f"rollout/valid_pred_objects": float(sum(int(m.get("valid_pred_objects", 0)) for m in meta)),
                    f"rollout/gt_objects": gt_total,
                    f"rollout/match_rate": (matched_total / gt_total) if gt_total > 0 else 0.0,
                    f"rollout/decode_greedy": float(sum(1 for m in meta if m.get("decode_mode") == "greedy")),
                    f"rollout/decode_beam": float(sum(1 for m in meta if m.get("decode_mode") == "beam")),
                    f"rollout/matched_for_supervision": float(
                        sum(int(m.get("matched_for_supervision", 0)) for m in meta)
                    ),
                    f"rollout/excluded_from_supervision": float(
                        sum(int(m.get("excluded_from_supervision", 0)) for m in meta)
                    ),
                    f"loss/ce": float(ce_loss.detach().cpu().item()),
                    f"loss/coord": float(coord_loss.detach().cpu().item()),
                    f"loss/coord_prefix": float(prefix_coord_mean.detach().cpu().item()),
                    f"loss/coord_tail": float(tail_coord_mean.detach().cpu().item()),
                }
            )
        except Exception:
            pass

        return (total, outputs) if return_outputs else total

    def training_step(self, model, inputs, *args, **kwargs):
        # When using identity collator, `inputs` is a list of raw samples.
        if isinstance(inputs, list):
            batch = self._prepare_batch_inputs(inputs)
            return super().training_step(model, batch, *args, **kwargs)
        return super().training_step(model, inputs, *args, **kwargs)

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
