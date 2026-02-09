"""Rollout-matching SFT trainer (stage_2).

Implements the OpenSpec change:
  openspec/changes/2026-01-15-add-rollout-matching-trainer

High-level loop per batch:
  rollout (no grad) -> strict token-aligned parse -> Hungarian match -> build Y_train
  -> one teacher-forced forward -> masked CE + distributional coord losses.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import time
from contextlib import contextmanager, nullcontext
from copy import copy as shallow_copy
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask as maskUtils
from scipy.optimize import linear_sum_assignment
from swift.trainers import Seq2SeqTrainer
from swift.trainers.rlhf_trainer.utils import (
    get_gather_if_zero3_context,
    replace_assistant_response_with_ids,
)
from swift.utils import get_logger, unwrap_model_for_generation

from src.common.geometry import bbox_from_points, bbox_to_quadrilateral, flatten_points
from src.coord_tokens.codec import (
    get_coord_token_ids,
    token_to_int,
    value_in_coord_range,
)
from src.coord_tokens.soft_ce_w1 import coord_soft_ce_w1

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
    output formatting stable (e.g., top-level object_k keys).

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


GeomType = Literal["bbox_2d", "poly"]


_OBJECT_KEY_RE = re.compile(r"^object_(\d+)$")
_IM_END = "<|im_end|>"


class _ForceEosOnRepeatGuard:
    """HF rollout-time guardrail: force EOS when generation becomes degenerate.

    This is intentionally lightweight and batch-friendly:
    - It does NOT stop the entire batch (which would harm other samples).
    - Instead, it forces EOS for only the offending sequences by overriding logits.

    Motivation:
      Some rollouts can get stuck emitting repetitive garbage until `max_new_tokens`,
      which makes stage_2 extremely slow and produces unusable supervision.
    """

    def __init__(
        self,
        *,
        eos_token_id: int,
        prompt_len: int,
        min_new_tokens: int,
        max_consecutive_token_repeats: int,
        ngram_size: int,
        ngram_repeats: int,
        max_object_keys: Optional[int],
        object_key_prefix_token_ids: Optional[List[int]],
    ) -> None:
        self.eos_token_id = int(eos_token_id)
        self.prompt_len = int(prompt_len)
        self.min_new_tokens = int(min_new_tokens)
        self.max_consecutive_token_repeats = int(max_consecutive_token_repeats)
        self.ngram_size = int(ngram_size)
        self.ngram_repeats = int(ngram_repeats)
        self.max_object_keys = (
            int(max_object_keys) if max_object_keys is not None else None
        )
        self.object_key_prefix_token_ids = (
            [int(x) for x in object_key_prefix_token_ids]
            if isinstance(object_key_prefix_token_ids, list)
            and object_key_prefix_token_ids
            else None
        )

        # Lazily initialized per batch (batch size is only known at generation time).
        self._processed_lens: Optional[List[int]] = None
        self._obj_counts: Optional[List[int]] = None
        self._obj_match_idx: Optional[List[int]] = None

    def _init_state_if_needed(self, batch_size: int) -> None:
        if self._processed_lens is not None and len(self._processed_lens) == batch_size:
            return
        self._processed_lens = [int(self.prompt_len) for _ in range(batch_size)]
        self._obj_counts = [0 for _ in range(batch_size)]
        self._obj_match_idx = [0 for _ in range(batch_size)]

    def _update_object_key_counts(self, input_ids: torch.Tensor) -> None:
        """Incrementally count '"object_' occurrences in the generated tail."""
        if self.max_object_keys is None or not self.object_key_prefix_token_ids:
            return
        if (
            self._processed_lens is None
            or self._obj_counts is None
            or self._obj_match_idx is None
        ):
            return

        pat = self.object_key_prefix_token_ids
        bs = int(input_ids.shape[0])
        cur_len = int(input_ids.shape[1])

        for i in range(bs):
            start = int(self._processed_lens[i])
            if start < self.prompt_len:
                start = int(self.prompt_len)
            if start >= cur_len:
                continue

            match = int(self._obj_match_idx[i])
            count = int(self._obj_counts[i])
            # Usually only 0..1 tokens per call, but keep it general.
            for pos in range(start, cur_len):
                tid = int(input_ids[i, pos].item())
                if tid == pat[match]:
                    match += 1
                    if match >= len(pat):
                        count += 1
                        match = 0
                else:
                    match = 1 if tid == pat[0] else 0

            self._obj_match_idx[i] = match
            self._obj_counts[i] = count
            self._processed_lens[i] = cur_len

    def _should_force_eos_for_seq(self, seq: torch.Tensor, *, obj_count: int) -> bool:
        gen_len = int(seq.numel()) - int(self.prompt_len)
        if gen_len < int(self.min_new_tokens):
            return False

        # 1) Hard cap on the number of '"object_' keys (prevents runaway object spam).
        if self.max_object_keys is not None and int(obj_count) >= int(
            self.max_object_keys
        ):
            return True

        # 2) Trivial single-token loops (e.g. repeating the same token forever).
        if int(self.max_consecutive_token_repeats) > 0 and seq.numel() > 0:
            last = int(seq[-1].item())
            run = 1
            # Bound the scan to avoid O(L) work.
            limit = min(int(self.max_consecutive_token_repeats), int(seq.numel()))
            for j in range(2, limit + 1):
                if int(seq[-j].item()) != last:
                    break
                run += 1
            if run >= int(self.max_consecutive_token_repeats):
                return True

        # 3) Detect consecutive repeated n-grams: (tail == prev == prev2 ...).
        n = int(self.ngram_size)
        reps = int(self.ngram_repeats)
        if n > 0 and reps >= 2 and gen_len >= n * reps:
            gen = seq[int(self.prompt_len) :]
            if int(gen.numel()) >= n * reps:
                tail = gen[-n:]
                ok = True
                for k in range(2, reps + 1):
                    prev = gen[-k * n : -(k - 1) * n]
                    if prev.numel() != tail.numel() or not torch.equal(prev, tail):
                        ok = False
                        break
                if ok:
                    return True

        return False

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        # Signature matches transformers LogitsProcessor protocol.
        if self.eos_token_id < 0:
            return scores
        if input_ids.ndim != 2 or scores.ndim != 2:
            return scores
        bs = int(input_ids.shape[0])
        self._init_state_if_needed(bs)
        self._update_object_key_counts(input_ids)

        force = torch.zeros((bs,), dtype=torch.bool, device=scores.device)
        obj_counts = self._obj_counts or [0 for _ in range(bs)]
        for i in range(bs):
            if self._should_force_eos_for_seq(
                input_ids[i], obj_count=int(obj_counts[i])
            ):
                force[i] = True

        if not bool(force.any().item()):
            return scores

        # Force EOS for the flagged sequences by overriding logits.
        scores = scores.clone()
        scores[force, :] = -float("inf")
        scores[force, int(self.eos_token_id)] = 0.0
        return scores


@dataclass
class ParsedPredObject:
    key: str
    index: int
    desc: str
    geom_type: GeomType
    coord_token_indices: List[
        int
    ]  # indices into rollout response_token_ids (assistant-local)
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
    dropped_invalid_by_reason: Dict[str, int] = field(default_factory=dict)
    dropped_ambiguous: int = 0
    truncated: bool = False


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
    # Sum/count over matched pairs (maskIoU in norm1000 space). Used for lightweight monitoring.
    matched_maskiou_sum: float
    matched_maskiou_count: int


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
        if (
            capture_active
            and int(tok_id) in coord_id_set
            and current_object is not None
        ):
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
                            desc="",
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
                    current_object.value_span = (
                        current_object.value_span[0],
                        global_char_pos + 1,
                    )
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
) -> Tuple[List[ParsedPredObject], int, Dict[str, int]]:
    """Strict validation: drop malformed objects (no repair).

    Returns:
      valid: objects that pass strict parsing/shape constraints
      dropped: total count of dropped objects
      dropped_by_reason: coarse reason buckets (diagnostics)
    """

    valid: List[ParsedPredObject] = []
    dropped = 0
    dropped_by_reason: Dict[str, int] = {}

    def _drop(reason: str) -> None:
        nonlocal dropped
        dropped += 1
        dropped_by_reason[str(reason)] = int(dropped_by_reason.get(str(reason), 0)) + 1

    for obj in objects_raw:
        start, end = obj.value_span
        if end <= 0 or end > prefix_char_end:
            _drop("key_invalid")
            continue
        snippet = response_text[start:end]
        try:
            parsed = json.loads(snippet)
        except Exception:
            _drop("key_invalid")
            continue
        if not isinstance(parsed, dict):
            _drop("key_invalid")
            continue

        desc = parsed.get("desc")
        if not isinstance(desc, str) or not desc.strip():
            _drop("missing_desc")
            continue

        non_desc_keys = [k for k in parsed.keys() if k != "desc"]
        if not non_desc_keys:
            _drop("missing_geom")
            continue

        known_geom = {"bbox_2d", "poly"}
        geom_keys = [k for k in non_desc_keys if k in known_geom]
        extra_keys = [k for k in parsed.keys() if k not in {"desc", *known_geom}]

        if len(geom_keys) == 0:
            # Some kind of geometry was present but not recognized.
            _drop("unknown_geom")
            continue
        if len(geom_keys) != 1:
            _drop("key_invalid")
            continue
        if extra_keys:
            _drop("key_invalid")
            continue

        geom_key = geom_keys[0]

        # Geometry values must be coord tokens (strict; no ints in rollout-matching).
        flat = flatten_points(parsed.get(geom_key))
        if flat is None or len(flat) % 2 != 0:
            _drop("wrong_arity")
            continue
        token_bins: List[int] = []
        ok = True
        for v in flat:
            if not isinstance(v, str):
                ok = False
                break
            try:
                token_bins.append(int(token_to_int(str(v).strip())))
            except Exception:
                ok = False
                break
        if not ok:
            _drop("non_coord_token")
            continue

        # Ensure we captured coord-token indices and they match geometry arity.
        coord_idx = list(obj.coord_token_indices)
        if geom_key == "bbox_2d":
            if len(coord_idx) != 4 or len(token_bins) != 4:
                _drop("wrong_arity")
                continue
            # Basic bbox sanity: must be in-range and canonical (x2>=x1, y2>=y1).
            if any((t < 0) or (t > 999) for t in token_bins):
                _drop("bbox_invalid")
                continue
            x1, y1, x2, y2 = token_bins
            if x2 < x1 or y2 < y1:
                _drop("bbox_invalid")
                continue
        else:  # poly
            if (
                len(coord_idx) < 6
                or (len(coord_idx) % 2 != 0)
                or len(token_bins) != len(coord_idx)
            ):
                _drop("wrong_arity")
                continue

        valid.append(
            ParsedPredObject(
                key=obj.key,
                index=int(obj.index),
                desc=str(desc).strip(),
                geom_type=geom_key,  # type: ignore[assignment]
                coord_token_indices=coord_idx,
                value_span=obj.value_span,
            )
        )

    return valid, dropped, dropped_by_reason


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

    objects_raw, _, first_open_brace_pos, truncated, last_obj_end = (
        _scan_rollout_tokens(
            tokenizer=tokenizer,
            response_token_ids=response_token_ids,
            coord_id_set=coord_id_set,
        )
    )

    # If completely malformed (no top-level '{'), fall back to empty prefix so FN append can proceed.
    if first_open_brace_pos < 0:
        prefix_token_ids = [
            int(t) for t in tokenizer.encode("{", add_special_tokens=False)
        ]
        prefix_text = tokenizer.decode(
            prefix_token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        return RolloutParseResult(
            response_token_ids=list(response_token_ids),
            response_text=response_text,
            prefix_token_ids=prefix_token_ids,
            prefix_text=prefix_text,
            max_object_index_in_prefix=None,
            valid_objects=[],
            dropped_invalid=0,
            dropped_invalid_by_reason={},
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

    valid_objects, dropped_invalid, dropped_invalid_by_reason = _validate_objects_strict(
        tokenizer=tokenizer,
        response_text=response_text,
        objects_raw=objects_raw,
        prefix_char_end=cut_char,
    )

    # In this MVP implementation, coord-slot alignment ambiguity is treated as invalid.
    dropped_ambiguous = 0

    # Key allocation must reserve ids based on all retained object_N keys in prefix,
    # even when strict object validation later drops those entries.
    max_in_prefix: Optional[int] = None
    for obj in objects_raw:
        end = int(obj.value_span[1]) if isinstance(obj.value_span, tuple) else -1
        if end <= 0 or end > int(cut_char):
            continue
        if max_in_prefix is None or int(obj.index) > int(max_in_prefix):
            max_in_prefix = int(obj.index)

    return RolloutParseResult(
        response_token_ids=list(response_token_ids),
        response_text=response_text,
        prefix_token_ids=prefix_token_ids,
        prefix_text=prefix_text,
        max_object_index_in_prefix=max_in_prefix,
        valid_objects=valid_objects,
        dropped_invalid=int(dropped_invalid),
        dropped_invalid_by_reason=dict(dropped_invalid_by_reason),
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


def _bbox_xyxy_from_norm(
    points: Sequence[int], kind: GeomType
) -> Tuple[float, float, float, float]:
    if kind == "bbox_2d":
        if len(points) != 4:
            return 0.0, 0.0, 0.0, 0.0
        x1, y1, x2, y2 = [float(v) for v in points]
        return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
    x1, y1, x2, y2 = bbox_from_points([float(v) for v in points])
    return float(x1), float(y1), float(x2), float(y2)


def _bbox_iou_xyxy(
    a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]
) -> float:
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
        # Project [0,999] -> [0,R-1] continuous coordinates.
        # Mirror ints_to_pixels_norm1000: frac=v/999, then scale by (R-1).
        denom = 999.0
        scale = float(max(r - 1, 1)) / denom
        return [float(v) * scale for v in values]

    def _as_poly(kind: GeomType, pts: Sequence[int]) -> List[float]:
        if kind == "bbox_2d":
            if len(pts) != 4:
                return []
            x1, y1, x2, y2 = [float(v) for v in pts]
            quad = bbox_to_quadrilateral(
                [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
            )
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
        return MatchResult(
            matched_pairs=[],
            fn_gt_indices=list(range(gt_n)),
            fp_pred_indices=[],
            gating_rejections=0,
            matched_maskiou_sum=0.0,
            matched_maskiou_count=0,
        )
    if gt_n == 0:
        return MatchResult(
            matched_pairs=[],
            fn_gt_indices=[],
            fp_pred_indices=list(range(pred_n)),
            gating_rejections=0,
            matched_maskiou_sum=0.0,
            matched_maskiou_count=0,
        )

    k = max(1, int(top_k))
    gate = float(gate_threshold)
    inf = 1e6

    gt_boxes = [_bbox_xyxy_from_norm(gt.points_norm1000, gt.geom_type) for gt in gts]
    pred_boxes = [
        _bbox_xyxy_from_norm(pr.points_norm1000, pr.geom_type) for pr in preds
    ]

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
    matched_maskiou_sum = 0.0
    matched_maskiou_count = 0
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
            # cost_pg is (1 - iou) for allowed candidates.
            iou = 1.0 - float(cost_pg[i, c])
            if iou < 0.0:
                iou = 0.0
            if iou > 1.0:
                iou = 1.0
            matched_maskiou_sum += float(iou)
            matched_maskiou_count += 1
        else:
            fp_preds.append(i)

    fn_gts = [j for j in range(gt_n) if j not in matched_gt]
    return MatchResult(
        matched_pairs=matched_pairs,
        fn_gt_indices=fn_gts,
        fp_pred_indices=fp_preds,
        gating_rejections=int(gating_rejections),
        matched_maskiou_sum=float(matched_maskiou_sum),
        matched_maskiou_count=int(matched_maskiou_count),
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
        geom_keys = [
            k for k in ("bbox_2d", "poly") if k in entry and entry[k] is not None
        ]
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
        objs.append(
            GTObject(
                index=idx, geom_type=geom_key, points_norm1000=pts, desc=desc.strip()
            )
        )
    objs.sort(key=lambda o: o.index)
    return objs


def _serialize_append_fragment(
    *,
    fn_objects: Sequence[GTObject],
    start_index: int,
    prefix_text: str,
) -> str:
    tail = prefix_text.rstrip()
    if not tail:
        raise ValueError("empty rollout prefix is not append-ready")

    last = tail[-1]
    if last not in {"{", "}", ","}:
        raise ValueError(f"rollout prefix is not append-ready; last_char={last!r}")

    # Deterministically validate the retained prefix as an open top-level JSON object
    # and detect whether it already contains at least one object entry.
    def _prefix_has_object_entries(text: str) -> bool:
        depth = 0
        in_string = False
        escape = False
        expecting_key: Dict[int, bool] = {}
        current_string: List[str] = []
        saw_root_open = False
        object_keys = 0

        for ch in text:
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
                    if expecting_key.get(depth, False) and depth == 1:
                        if _OBJECT_KEY_RE.match(s):
                            object_keys += 1
                else:
                    current_string.append(ch)
                continue

            if ch == '"':
                in_string = True
                escape = False
                current_string = []
                continue

            if ch == "{":
                depth += 1
                expecting_key[depth] = True
                if depth == 1:
                    saw_root_open = True
                continue

            if ch == "}":
                if depth > 0:
                    expecting_key.pop(depth, None)
                    depth -= 1
                continue

            if ch == ",":
                if depth >= 1:
                    expecting_key[depth] = True
                continue

            if ch == ":":
                if depth >= 1:
                    expecting_key[depth] = False
                continue

        if not saw_root_open:
            raise ValueError("rollout prefix missing top-level '{'")
        if depth != 1:
            raise ValueError(
                "rollout prefix must end with an open top-level object before FN append"
            )
        if in_string:
            raise ValueError("rollout prefix ends inside a quoted string")

        return bool(object_keys > 0)

    has_entries = _prefix_has_object_entries(tail)

    if not fn_objects:
        if last == ",":
            raise ValueError(
                "rollout prefix ends with ',' but FN set is empty; invalid JSON"
            )
        return "}"

    leading = ""
    if has_entries and last != ",":
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
            pairs = [
                [f"<|coord_{int(pts[i])}|>", f"<|coord_{int(pts[i + 1])}|>"]
                for i in range(0, len(pts), 2)
            ]
            payload["poly"] = pairs
        entries.append(
            f'"object_{n}": {json.dumps(payload, ensure_ascii=False, separators=(", ", ": "))}'
        )
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


def _find_desc_value_token_positions(
    *, tokenizer: Any, token_ids: Sequence[int]
) -> List[int]:
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
    full = torch.nan_to_num(
        logits_full.float(), nan=0.0, posinf=1e4, neginf=-1e4
    ).clamp(min=-1e4, max=1e4) / float(temperature)
    coord = torch.nan_to_num(
        logits_coord.float(), nan=0.0, posinf=1e4, neginf=-1e4
    ).clamp(min=-1e4, max=1e4) / float(temperature)
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
    typically use an explicit `rollout_generate_batch_size`.
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
        except Exception:
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
        except Exception:
            base_raw_batch_size = 1

        n_raw = int(n_micro) * int(base_raw_batch_size)
        target = int(self._target_microbatch_size())
        return int((n_raw + target - 1) // target)

    def __getattr__(self, name: str):
        return getattr(self.dataloader, name)


class _RolloutMatchingPackWindow:
    """Holds one accumulation window worth of raw micro-batches and cached prepared batches.

    This enables window-aware scheduling (lookahead) without changing the Trainer's
    micro-step interface: each yielded micro-batch is still a list, but carries a
    pointer to the full window.
    """

    def __init__(self, *, raw_micro_batches: List[List[Any]]):
        self.raw_micro_batches = raw_micro_batches
        self._prepared_micro_batches: Optional[List[Dict[str, Any]]] = None

    @property
    def gas(self) -> int:
        return int(len(self.raw_micro_batches))

    def get_prepared(
        self,
        *,
        idx: int,
        build_all_prepared: Any,
    ) -> Dict[str, Any]:
        if self._prepared_micro_batches is None:
            self._prepared_micro_batches = list(build_all_prepared())
        if not isinstance(self._prepared_micro_batches, list):
            raise ValueError("prepared window batches must be a list")
        if idx < 0 or idx >= len(self._prepared_micro_batches):
            raise IndexError("prepared window batch index out of range")
        prepared = self._prepared_micro_batches[idx]
        if not isinstance(prepared, dict):
            raise ValueError("prepared batch must be a dict")
        return prepared


class _WindowedMicroBatch(list):
    """List-like micro-batch carrying a reference to its full accumulation window."""

    def __init__(self, raw: List[Any], *, window: _RolloutMatchingPackWindow, idx: int):
        super().__init__(raw)
        self.rm_window = window
        self.rm_window_idx = int(idx)


class _AccumulationWindowLookahead:
    """Prefetch `gas` micro-batches so the trainer can schedule within the full window."""

    def __init__(self, dataloader, *, gas: int):
        self.dataloader = dataloader
        self.gas = int(gas)

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        it = iter(self.dataloader)
        while True:
            raw_window: List[List[Any]] = []
            try:
                for _ in range(int(self.gas)):
                    b = next(it)
                    # Identity collator yields a list of raw samples.
                    if not isinstance(b, list):
                        raise ValueError(
                            "window lookahead expects identity-collated train batches (list of raw samples)"
                        )
                    raw_window.append(b)
            except StopIteration:
                # Partial final window: yield whatever is left without window context.
                for b in raw_window:
                    yield b
                break

            window = _RolloutMatchingPackWindow(raw_micro_batches=raw_window)
            for i, b in enumerate(raw_window):
                yield _WindowedMicroBatch(b, window=window, idx=int(i))


class _DropRemainderAccumulationWindow:
    """Drop the final partial gradient-accumulation window.

    HF/Swift will still perform an optimizer step on a partial accumulation window at
    the end of an epoch. For stage_2 step-budgeted trainers we typically want fixed
    raw-sample budgets per optimizer step, so when `training.dataloader_drop_last` is
    enabled we truncate the train dataloader to a multiple of
    `gradient_accumulation_steps`.

    This is intentionally a lightweight wrapper that preserves epoch semantics by
    dropping the remainder *per epoch*.
    """

    def __init__(self, dataloader, *, gas: int):
        self.dataloader = dataloader
        self.gas = max(1, int(gas))

    def __len__(self) -> int:
        inner_len = getattr(self.dataloader, "__len__", None)
        if inner_len is None:
            raise TypeError("wrapped dataloader has no __len__")
        n = int(len(self.dataloader))
        return int((n // int(self.gas)) * int(self.gas))

    def __iter__(self):
        n_keep = int(len(self))
        for i, b in enumerate(self.dataloader):
            if i >= n_keep:
                break
            yield b

    def __getattr__(self, name: str):
        return getattr(self.dataloader, name)



def _slim_rollout_meta_for_logging(meta: Mapping[str, Any]) -> Dict[str, Any]:
    """Drop large fields (e.g. token id lists) from rollout meta before buffering for logs.

    Rollout-matching uses gradient accumulation; we want ONE log point per optimizer step.
    We therefore buffer meta across micro-batches, but keep it lightweight.
    """

    def _as_int(x: Any, default: int = 0) -> int:
        try:
            return int(x)
        except Exception:
            return int(default)

    def _as_float(x: Any, default: float = 0.0) -> float:
        try:
            return float(x)
        except Exception:
            return float(default)

    out: Dict[str, Any] = {
        "prompt_len": _as_int(meta.get("prompt_len", 0)),
        "rollout_len": _as_int(meta.get("rollout_len", 0)),
        "prefix_len": _as_int(meta.get("prefix_len", 0)),
        "train_len": _as_int(meta.get("train_len", 0)),
        "encoded_len": _as_int(meta.get("encoded_len", 0)),
        "decode_mode": str(meta.get("decode_mode", "")),
        "parse_dropped_invalid": _as_int(meta.get("parse_dropped_invalid", 0)),
        "parse_dropped_ambiguous": _as_int(meta.get("parse_dropped_ambiguous", 0)),
        "parse_truncated": bool(meta.get("parse_truncated", False)),
        "valid_pred_objects": _as_int(meta.get("valid_pred_objects", 0)),
        "matched_for_supervision": _as_int(meta.get("matched_for_supervision", 0)),
        "matched_maskiou_sum": _as_float(meta.get("matched_maskiou_sum", 0.0)),
        "matched_maskiou_count": _as_int(meta.get("matched_maskiou_count", 0)),
        "gt_objects": _as_int(meta.get("gt_objects", 0)),
        "fn_count": _as_int(meta.get("fn_count", 0)),
        "gating_rejections": _as_int(meta.get("gating_rejections", 0)),
        "excluded_from_supervision": _as_int(meta.get("excluded_from_supervision", 0)),
        # For token masking diagnostics.
        "prefix_coord_target_bins": list(meta.get("prefix_coord_target_bins") or []),
        "tail_ignore_pos": list(meta.get("tail_ignore_pos") or []),
        # Desc monitor fields (optional).
        "desc_monitor_ran": bool(meta.get("desc_monitor_ran", False)),
        "desc_pairs_total": _as_int(meta.get("desc_pairs_total", 0)),
        "desc_exact_ok": _as_int(meta.get("desc_exact_ok", 0)),
        "desc_sem_ok": _as_int(meta.get("desc_sem_ok", 0)),
        "desc_sem_sim_sum": _as_float(meta.get("desc_sem_sim_sum", 0.0)),
        "desc_sem_sim_count": _as_int(meta.get("desc_sem_sim_count", 0)),
        "desc_sem_enabled": _as_int(meta.get("desc_sem_enabled", 0)),
    }

    return out


@dataclass
class _PendingTrainRolloutLog:
    """Accumulate rollout-matching logs across micro-batches for ONE optimizer step."""

    meta: List[Dict[str, Any]] = field(default_factory=list)

    ce_loss_sum: float = 0.0
    coord_loss_sum: float = 0.0
    coord_prefix_sum: float = 0.0
    coord_tail_sum: float = 0.0
    n_micro: int = 0

    time_forward_s: float = 0.0
    time_mask_build_s: float = 0.0

    # Rollout pipeline timings.
    time_rollout_generate_s: float = 0.0
    time_rollout_parse_match_s: float = 0.0
    time_rollout_teacher_encode_s: float = 0.0

    # Packing/collation timing.
    time_post_rollout_pack_s: float = 0.0

    # Packing stats (optional; only populated when packing is enabled).
    packing_fill_sum: float = 0.0
    packing_selected_total_len_sum: float = 0.0
    packing_segments_sum: float = 0.0
    packing_count: int = 0
    packing_buffer_last: float = 0.0


    def add_micro(
        self,
        *,
        meta: List[Mapping[str, Any]],
        ce_loss: float,
        coord_loss: float,
        coord_prefix: float,
        coord_tail: float,
        time_forward_s: float,
        time_mask_build_s: float,
        batch_metrics: Optional[Mapping[str, Any]],
    ) -> None:
        self.n_micro += 1
        self.ce_loss_sum += float(ce_loss)
        self.coord_loss_sum += float(coord_loss)
        self.coord_prefix_sum += float(coord_prefix)
        self.coord_tail_sum += float(coord_tail)
        self.time_forward_s += float(time_forward_s)
        self.time_mask_build_s += float(time_mask_build_s)

        for m in meta:
            if isinstance(m, Mapping):
                self.meta.append(_slim_rollout_meta_for_logging(m))

        if not isinstance(batch_metrics, Mapping):
            return

        # Timings.
        self.time_rollout_generate_s += float(
            batch_metrics.get("time/rollout_generate_s", 0.0) or 0.0
        )
        self.time_rollout_parse_match_s += float(
            batch_metrics.get("time/rollout_parse_match_s", 0.0) or 0.0
        )
        self.time_rollout_teacher_encode_s += float(
            batch_metrics.get("time/rollout_teacher_encode_s", 0.0) or 0.0
        )
        self.time_post_rollout_pack_s += float(
            batch_metrics.get("time/post_rollout_pack_s", 0.0) or 0.0
        )

        # Packing stats.
        if "packing/post_rollout_fill" in batch_metrics:
            self.packing_fill_sum += float(
                batch_metrics.get("packing/post_rollout_fill", 0.0) or 0.0
            )
            self.packing_selected_total_len_sum += float(
                batch_metrics.get("packing/post_rollout_selected_total_len", 0.0) or 0.0
            )
            self.packing_segments_sum += float(
                batch_metrics.get("packing/post_rollout_segments", 0.0) or 0.0
            )
            self.packing_buffer_last = float(
                batch_metrics.get(
                    "packing/post_rollout_buffer", self.packing_buffer_last
                )
                or self.packing_buffer_last
            )
            self.packing_count += 1



def schedule_post_rollout_segment_indices_window(
    *,
    encoded_lens: Sequence[int],
    packing_length: int,
    gas: int,
    select_indices_fn: Any,
) -> List[List[int]]:
    """Schedule post-rollout segments into exactly `gas` micro-packs.

    This is used when `post_rollout_pack_scope == "window"`: we already have one
    gradient-accumulation window worth of segments and must emit exactly one packed
    batch per micro-step.

    Important: this differs from the "micro" packer, which tries to greedily fill a
    single pack to maximize utilization. For "window" packing we *must* distribute
    segments across a fixed number of micro-steps.

    `select_indices_fn` is accepted for backward compatibility but is not used in the
    window scheduler.
    """

    _ = select_indices_fn

    packing_length = int(packing_length)
    gas = int(gas)
    if packing_length <= 0:
        raise ValueError("packing_length must be positive")
    if gas <= 0:
        raise ValueError("gas must be positive")

    lens = [int(x) for x in encoded_lens]
    if not lens:
        raise ValueError("window scheduling requires at least one segment")

    for sl in lens:
        if sl <= 0:
            raise ValueError("encoded_len must be positive")
        if sl > packing_length:
            raise ValueError(
                f"post-rollout window packing cannot fit a single segment: encoded_len={sl} > packing_length={packing_length}. "
                "Mitigations: increase global_max_length/template.max_length, reduce max_new_tokens, or disable packing."
            )

    total = int(sum(lens))
    if total > gas * packing_length:
        raise ValueError(
            f"infeasible window: sum_encoded_len={total} > gas*packing_length={gas * packing_length}"
        )

    n = int(len(lens))
    if n < gas:
        raise ValueError(
            f"window post-rollout packing requires at least gas segments: segments={n} < gas={gas}. "
            "Mitigations: increase rollout_generate_batch_size (more raw samples per micro-step) or disable window packing."
        )

    # Greedy LPT-style load balancing with a hard capacity constraint.
    # Items are (index, encoded_len).
    items: List[Tuple[int, int]] = [(int(i), int(sl)) for i, sl in enumerate(lens)]
    items.sort(key=lambda t: (-int(t[1]), int(t[0])))

    seeds = items[:gas]
    remaining = items[gas:]

    bins: List[List[int]] = [[int(idx)] for idx, _ in seeds]
    bin_sums: List[int] = [int(sl) for _, sl in seeds]

    for idx, sl in remaining:
        best_bin: Optional[int] = None
        best_sum: Optional[int] = None
        for b in range(gas):
            if int(bin_sums[b]) + int(sl) <= packing_length:
                s = int(bin_sums[b])
                if best_bin is None or best_sum is None or s < best_sum:
                    best_bin = int(b)
                    best_sum = int(s)
        if best_bin is None:
            raise ValueError(
                "window post-rollout packing could not fit all segments into the fixed number of micro-packs. "
                "Even if sum_encoded_len <= gas*packing_length, indivisible segments can still make packing infeasible. "
                "Mitigations: reduce max_new_tokens, increase packing_length (global_max_length), or switch post_rollout_pack_scope to 'micro'."
            )
        bins[int(best_bin)].append(int(idx))
        bin_sums[int(best_bin)] += int(sl)

    packs: List[List[int]] = [sorted(b) for b in bins]

    # Sanity checks: non-empty packs and exact cover.
    if any(len(p) == 0 for p in packs):
        raise ValueError("window post-rollout packing produced an empty micro-pack")

    used: List[int] = sorted(int(i) for p in packs for i in p)
    if used != list(range(n)):
        raise ValueError(
            "window post-rollout packing produced an invalid segment index cover"
        )

    for p in packs:
        p_total = int(sum(int(lens[i]) for i in p))
        if p_total > packing_length:
            raise ValueError("window post-rollout packing overflowed packing_length")

    return packs


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
        try:
            cfg = self.rollout_matching_cfg
            if isinstance(cfg, Mapping) and key in cfg:
                return cfg[key]
        except Exception:
            pass
        return default

    def _validate_rollout_matching_cfg(self) -> None:
        cfg = getattr(self, "rollout_matching_cfg", None)
        if cfg is None:
            return
        if not isinstance(cfg, Mapping):
            raise TypeError(
                "rollout_matching_cfg must be a mapping (injected from custom.extra.rollout_matching)"
            )

        legacy = [
            k
            for k in ("temperature", "top_p", "top_k", "rollout_buffer")
            if k in cfg
        ]
        if legacy:
            legacy_s = ", ".join(f"custom.extra.rollout_matching.{k}" for k in legacy)
            raise ValueError(
                "Legacy rollout-matching keys have been removed: "
                f"{legacy_s}. (No backward compatibility.)"
            )

        dec = cfg.get("decoding", None)
        if dec is None:
            dec = {}
        if not isinstance(dec, Mapping):
            raise TypeError(
                "custom.extra.rollout_matching.decoding must be a mapping when provided"
            )

        # Validate decoding ranges (robust defaults).
        try:
            temperature = float(dec.get("temperature", 0.0) or 0.0)
        except Exception as exc:
            raise TypeError(
                "custom.extra.rollout_matching.decoding.temperature must be a float"
            ) from exc
        if temperature < 0.0:
            raise ValueError(
                "custom.extra.rollout_matching.decoding.temperature must be >= 0"
            )

        try:
            top_p = float(dec.get("top_p", 1.0) if dec.get("top_p", None) is not None else 1.0)
        except Exception as exc:
            raise TypeError(
                "custom.extra.rollout_matching.decoding.top_p must be a float"
            ) from exc
        if not (0.0 < top_p <= 1.0):
            raise ValueError(
                "custom.extra.rollout_matching.decoding.top_p must be in (0, 1]"
            )

        top_k_raw = dec.get("top_k", -1)
        try:
            top_k = int(top_k_raw)
        except Exception as exc:
            raise TypeError(
                "custom.extra.rollout_matching.decoding.top_k must be an int"
            ) from exc
        if top_k != -1 and top_k < 1:
            raise ValueError(
                "custom.extra.rollout_matching.decoding.top_k must be -1 (disabled) or >= 1"
            )

    def _decoding_cfg(self) -> Mapping[str, Any]:
        # `rollout_matching_cfg` is injected in src/sft.py. Use a nested dict for decoding.
        raw = self._cfg("decoding", {})
        if raw is None:
            return {}
        if not isinstance(raw, Mapping):
            raise TypeError(
                "custom.extra.rollout_matching.decoding must be a mapping when provided"
            )
        return raw

    def _decoding_params(self) -> Tuple[float, float, int]:
        dec = self._decoding_cfg()

        temperature_raw = dec.get("temperature", 0.0)
        try:
            temperature = float(temperature_raw or 0.0)
        except Exception as exc:
            raise TypeError(
                "custom.extra.rollout_matching.decoding.temperature must be a float"
            ) from exc
        if temperature < 0.0:
            raise ValueError(
                "custom.extra.rollout_matching.decoding.temperature must be >= 0"
            )

        top_p_raw = dec.get("top_p", 1.0)
        try:
            top_p = float(top_p_raw if top_p_raw is not None else 1.0)
        except Exception as exc:
            raise TypeError(
                "custom.extra.rollout_matching.decoding.top_p must be a float"
            ) from exc
        if not (0.0 < top_p <= 1.0):
            raise ValueError(
                "custom.extra.rollout_matching.decoding.top_p must be in (0, 1]"
            )

        top_k_raw = dec.get("top_k", -1)
        try:
            top_k = int(top_k_raw)
        except Exception as exc:
            raise TypeError(
                "custom.extra.rollout_matching.decoding.top_k must be an int"
            ) from exc
        if top_k != -1 and top_k < 1:
            raise ValueError(
                "custom.extra.rollout_matching.decoding.top_k must be -1 (disabled) or >= 1"
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
        except Exception:
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
            except Exception:
                pass
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
        try:
            s = str(text)
        except Exception:
            return ""
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
            except Exception:
                pass

    def _format_monitor_dump_markdown(self, payload: Mapping[str, Any]) -> str:
        # Human-readable dump; keep it ASCII-safe to avoid surprising tooling issues.
        max_chars = int(self._monitor_dump_cfg().get("max_text_chars", 4000) or 4000)

        def _j(obj: Any) -> str:
            try:
                return json.dumps(obj, ensure_ascii=True, indent=2)
            except Exception:
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
            raise ValueError("custom.extra.rollout_matching.offload must be a mapping")

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
                "Mitigations: disable custom.extra.rollout_matching.offload, switch rollout_backend=hf, "
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
                f"custom.extra.rollout_matching.rollout_backend must be one of {{hf,vllm}}, got {backend!r}"
            )
        return backend  # type: ignore[return-value]

    def _vllm_mode(self) -> Literal["colocate", "server"]:
        """vLLM integration mode.

        - `colocate` (default): instantiate a local vLLM engine.
        - `server`: connect to a pre-launched ms-swift rollout server.
        """
        vcfg_raw = self._cfg("vllm", {}) or {}
        if not isinstance(vcfg_raw, Mapping):
            raise ValueError("custom.extra.rollout_matching.vllm must be a mapping")
        mode = str(vcfg_raw.get("mode", "colocate") or "colocate").strip().lower()
        if mode not in {"colocate", "server"}:
            raise ValueError(
                "custom.extra.rollout_matching.vllm.mode must be 'colocate' or 'server'; "
                f"got {mode!r}"
            )
        return mode  # type: ignore[return-value]

    def _vllm_server_cfg(self) -> Mapping[str, Any]:
        vcfg_raw = self._cfg("vllm", {}) or {}
        if not isinstance(vcfg_raw, Mapping):
            raise ValueError("custom.extra.rollout_matching.vllm must be a mapping")
        scfg_raw = vcfg_raw.get("server", {}) or {}
        if not isinstance(scfg_raw, Mapping):
            raise ValueError(
                "custom.extra.rollout_matching.vllm.server must be a mapping"
            )
        return scfg_raw

    def _vllm_server_specs(self) -> List[Dict[str, Any]]:
        """Normalize server list config to a list of {base_url, group_port} dicts.

        Spec contract: config MUST be expressed in exactly one of:
        - `server.servers: [...]` (preferred)
        - legacy paired lists: `server.base_url` + `server.group_port`
        """
        scfg = self._vllm_server_cfg()

        servers_raw = scfg.get("servers", None)
        base_url_raw = scfg.get("base_url", None)
        group_port_raw = scfg.get("group_port", None)

        if servers_raw is not None:
            if base_url_raw is not None or group_port_raw is not None:
                raise ValueError(
                    "custom.extra.rollout_matching.vllm.server must define exactly one of 'servers' or ('base_url','group_port'), not both"
                )

            if not isinstance(servers_raw, list) or not servers_raw:
                raise ValueError(
                    "custom.extra.rollout_matching.vllm.server.servers must be a non-empty list"
                )
            out: List[Dict[str, Any]] = []
            for i, s in enumerate(servers_raw):
                if not isinstance(s, Mapping):
                    raise ValueError(
                        "custom.extra.rollout_matching.vllm.server.servers[%d] must be a mapping"
                        % int(i)
                    )
                base_url = s.get("base_url")
                if not isinstance(base_url, str) or not base_url.strip():
                    raise ValueError(
                        "custom.extra.rollout_matching.vllm.server.servers[%d].base_url must be a non-empty string"
                        % int(i)
                    )
                group_port_entry_raw = s.get("group_port")
                try:
                    group_port_entry = int(group_port_entry_raw)
                except Exception as exc:
                    raise ValueError(
                        "custom.extra.rollout_matching.vllm.server.servers[%d].group_port must be an int"
                        % int(i)
                    ) from exc
                if group_port_entry <= 0:
                    raise ValueError(
                        "custom.extra.rollout_matching.vllm.server.servers[%d].group_port must be > 0"
                        % int(i)
                    )
                out.append(
                    {
                        "base_url": base_url.strip().rstrip("/"),
                        "group_port": int(group_port_entry),
                    }
                )
            return out

        # Legacy paired-list form.
        if base_url_raw is None or group_port_raw is None:
            raise ValueError(
                "custom.extra.rollout_matching.vllm.server must define either 'servers' or ('base_url','group_port')"
            )

        if isinstance(base_url_raw, str):
            if not isinstance(group_port_raw, int):
                raise ValueError(
                    "custom.extra.rollout_matching.vllm.server.group_port must be an int when base_url is a string"
                )
            if group_port_raw <= 0:
                raise ValueError(
                    "custom.extra.rollout_matching.vllm.server.group_port must be > 0"
                )
            return [
                {
                    "base_url": base_url_raw.strip().rstrip("/"),
                    "group_port": int(group_port_raw),
                }
            ]

        if not isinstance(base_url_raw, list) or not all(
            isinstance(u, str) and u.strip() for u in base_url_raw
        ):
            raise ValueError(
                "custom.extra.rollout_matching.vllm.server.base_url must be a string or a list of non-empty strings"
            )
        base_urls = [u.strip().rstrip("/") for u in base_url_raw]

        group_ports: List[int] = []
        if isinstance(group_port_raw, int):
            if group_port_raw <= 0:
                raise ValueError(
                    "custom.extra.rollout_matching.vllm.server.group_port must be > 0"
                )
            group_ports = [int(group_port_raw) + i for i in range(len(base_urls))]
        elif isinstance(group_port_raw, list):
            if len(group_port_raw) != len(base_urls):
                raise ValueError(
                    "custom.extra.rollout_matching.vllm.server.group_port list length must match base_url list length"
                )
            for i, gp_raw in enumerate(group_port_raw):
                try:
                    gp = int(gp_raw)
                except Exception as exc:
                    raise ValueError(
                        "custom.extra.rollout_matching.vllm.server.group_port[%d] must be an int"
                        % int(i)
                    ) from exc
                if gp <= 0:
                    raise ValueError(
                        "custom.extra.rollout_matching.vllm.server.group_port[%d] must be > 0"
                        % int(i)
                    )
                group_ports.append(int(gp))
        else:
            raise ValueError(
                "custom.extra.rollout_matching.vllm.server.group_port must be an int or list of ints"
            )

        return [
            {"base_url": u, "group_port": int(gp)}
            for u, gp in zip(base_urls, group_ports)
        ]

    def _vllm_server_timeouts(self) -> Tuple[float, float]:
        scfg = self._vllm_server_cfg()

        timeout_raw = scfg.get("timeout_s", None)
        if timeout_raw is None:
            timeout_s = 240.0
        else:
            try:
                timeout_s = float(timeout_raw)
            except Exception as exc:
                raise ValueError(
                    "custom.extra.rollout_matching.vllm.server.timeout_s must be a float/int"
                ) from exc
        if timeout_s <= 0:
            raise ValueError(
                "custom.extra.rollout_matching.vllm.server.timeout_s must be > 0"
            )

        infer_timeout_raw = scfg.get("infer_timeout_s", None)
        if infer_timeout_raw is None:
            infer_timeout_s = float(timeout_s)
        else:
            try:
                infer_timeout_s = float(infer_timeout_raw)
            except Exception as exc:
                raise ValueError(
                    "custom.extra.rollout_matching.vllm.server.infer_timeout_s must be null or a float/int"
                ) from exc
            if infer_timeout_s <= 0:
                infer_timeout_s = float(timeout_s)

        return float(timeout_s), float(infer_timeout_s)

    def _vllm_server_sync_cfg(self) -> Tuple[str, bool]:
        vcfg_raw = self._cfg("vllm", {}) or {}
        if not isinstance(vcfg_raw, Mapping):
            raise ValueError("custom.extra.rollout_matching.vllm must be a mapping")
        sync_raw = vcfg_raw.get("sync", {}) or {}
        if not isinstance(sync_raw, Mapping):
            raise ValueError(
                "custom.extra.rollout_matching.vllm.sync must be a mapping"
            )

        mode = str(sync_raw.get("mode", "full") or "full").strip().lower()
        if mode not in {"full", "adapter", "auto"}:
            raise ValueError(
                "custom.extra.rollout_matching.vllm.sync.mode must be one of: full|adapter|auto"
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

    def _rollout_generate_batch_size(self) -> int:
        raw = self._cfg("rollout_generate_batch_size", 1)
        try:
            v = int(raw)
        except Exception as exc:
            raise ValueError(
                "custom.extra.rollout_matching.rollout_generate_batch_size must be an int"
            ) from exc
        return max(1, v)

    def _packing_enabled(self) -> bool:
        return bool(self._cfg("packing_enabled", False))

    def _packing_length(self) -> int:
        try:
            return int(self._cfg("packing_length", 0) or 0)
        except Exception as exc:
            raise ValueError("packing_length must be an int") from exc

    def _assert_single_packed_forward(self, batch: Mapping[str, Any], *, where: str) -> None:
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
        except Exception:
            max_len = 0
        if int(max_len) > 0 and int(seq_len) > int(max_len):
            raise ValueError(
                f"{where}: packed seq_len={int(seq_len)} exceeds packing_length/global_max_length={int(max_len)}"
            )

    def _packing_buffer_cap(self) -> int:
        try:
            return int(self._cfg("packing_buffer", 0) or 0)
        except Exception as exc:
            raise ValueError("packing_buffer must be an int") from exc

    def _packing_min_fill_ratio(self) -> float:
        try:
            v = float(self._cfg("packing_min_fill_ratio", 0.65))
        except Exception as exc:
            raise ValueError("packing_min_fill_ratio must be a float") from exc
        if not (0 < v <= 1):
            raise ValueError("packing_min_fill_ratio must be in (0, 1]")
        return float(v)

    def _packing_drop_last(self) -> bool:
        return bool(self._cfg("packing_drop_last", True))

    def _post_rollout_pack_scope(self) -> str:
        """Scope for post-rollout packing.

        - "micro" (default): pack within the current micro-batch (no dataloader lookahead).
        - "window": pack across one gradient-accumulation window (requires lookahead wrapper).

        Config key: custom.extra.rollout_matching.post_rollout_pack_scope
        """

        scope_raw = self._cfg("post_rollout_pack_scope", "micro")
        scope = str(scope_raw).strip().lower()
        if scope in {"window"}:
            return "window"
        if scope in {"micro", "step", "batch"}:
            return "micro"
        logger.warning(
            "Unknown post_rollout_pack_scope=%r; defaulting to 'micro'", scope_raw
        )
        return "micro"

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
            except Exception:
                pass
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
                except Exception:
                    pass
            if padding_free is not None:
                try:
                    template.padding_free = bool(padding_free)
                except Exception:
                    pass
            if mode is not None and old_mode is not None and hasattr(template, "set_mode"):
                try:
                    if str(old_mode) != str(mode):
                        template.set_mode(str(mode))
                except Exception:
                    pass

            yield
        finally:
            try:
                old_padding_free, old_packing, old_mode = stack.pop()
            except Exception:
                old_padding_free, old_packing, old_mode = False, False, None

            if old_mode is not None and hasattr(template, "set_mode"):
                try:
                    template.set_mode(old_mode)
                except Exception:
                    pass
            try:
                template.padding_free = old_padding_free
            except Exception:
                pass
            try:
                template.packing = old_packing
            except Exception:
                pass

            try:
                tls.depth = int(getattr(tls, "depth", 1) or 1) - 1
            except Exception:
                tls.depth = 0

            if int(getattr(tls, "depth", 0) or 0) <= 0:
                tls.depth = 0
                try:
                    tls.stack = []
                except Exception:
                    pass
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
        temperature, top_p, top_k = self._decoding_params()
        repetition_penalty = float(self._cfg("repetition_penalty", 1.0) or 1.0)
        if repetition_penalty <= 0:
            raise ValueError(
                "custom.extra.rollout_matching.repetition_penalty must be > 0"
            )

        # Rollout generation MUST run without sequence packing/padding-free.
        with self._template_packing_disabled():
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
        # GenerationConfig doesn't provide a stable `.clone()` across transformers versions.
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

        model_inputs = {k: v for k, v in batch.items() if k != "labels"}
        model_inputs.pop("position_ids", None)
        model_inputs.pop("text_position_ids", None)
        logits_processor = None
        repeat_cfg_raw = self._cfg("repeat_terminate", None)
        if isinstance(repeat_cfg_raw, Mapping) and bool(
            repeat_cfg_raw.get("enabled", False)
        ):
            min_new_tokens = int(repeat_cfg_raw.get("min_new_tokens", 256) or 0)
            max_consecutive = int(
                repeat_cfg_raw.get("max_consecutive_token_repeats", 64) or 0
            )
            ngram_size = int(repeat_cfg_raw.get("ngram_size", 64) or 0)
            ngram_repeats = int(repeat_cfg_raw.get("ngram_repeats", 2) or 2)
            max_object_keys_raw = repeat_cfg_raw.get("max_object_keys")
            max_object_keys = (
                int(max_object_keys_raw) if max_object_keys_raw is not None else None
            )

            obj_prefix_ids = None
            if max_object_keys is not None:
                try:
                    obj_prefix_ids = tok.encode('"object_', add_special_tokens=False)
                except Exception:
                    obj_prefix_ids = None

            eos_id = int(getattr(tok, "eos_token_id", -1) or -1)
            if eos_id >= 0:
                guard = _ForceEosOnRepeatGuard(
                    eos_token_id=eos_id,
                    prompt_len=prompt_len,
                    min_new_tokens=min_new_tokens,
                    max_consecutive_token_repeats=max_consecutive,
                    ngram_size=ngram_size,
                    ngram_repeats=ngram_repeats,
                    max_object_keys=max_object_keys,
                    object_key_prefix_token_ids=obj_prefix_ids,
                )
                try:
                    from transformers.generation.logits_process import (
                        LogitsProcessorList,
                    )

                    logits_processor = LogitsProcessorList([guard])
                except Exception:
                    logits_processor = [guard]
        with unwrap_model_for_generation(
            self.model_wrapped,
            self.accelerator,
            gather_deepspeed3_params=getattr(
                self.args, "ds3_gather_for_generation", False
            ),
        ) as unwrapped:
            unwrapped.eval()
            # Keep packing disabled inside rollout generate.
            with self._template_packing_disabled():
                with template.generate_context():
                    if (
                        getattr(self.model, "model_meta", None) is not None
                        and self.model.model_meta.is_multimodal
                    ):
                        _, model_inputs = template.pre_forward_hook(
                            unwrapped, None, model_inputs
                        )
                    # Some templates add auxiliary position ids that the underlying model.generate
                    # doesn't accept; strip them after any pre_forward_hook edits.
                    model_inputs.pop("position_ids", None)
                    model_inputs.pop("text_position_ids", None)
                    out = template.generate(
                        unwrapped,
                        **model_inputs,
                        generation_config=gen_cfg,
                        return_dict_in_generate=True,
                        logits_processor=logits_processor,
                    )
            unwrapped.train()

        sequences = out.sequences  # [R, T]
        if sequences.ndim != 2:
            raise ValueError("unexpected generate output shape")
        if (
            sequences.shape[0] > 1
            and hasattr(out, "sequences_scores")
            and out.sequences_scores is not None
        ):
            best = int(torch.argmax(out.sequences_scores).item())
        else:
            best = 0
        seq = sequences[best]
        resp_ids = seq[prompt_len:].tolist()
        resp_ids = template.skip_stop_tokens(resp_ids, is_finished=True)
        text = template.decode(
            resp_ids,
            is_finished=True,
            first_token=True,
            clean_up_tokenization_spaces=False,
        )
        return resp_ids, text, decode_mode, [int(t) for t in prompt_ids]

    # ---- rollout backends -------------------------------------------------
    def _ensure_vllm_engine(self) -> Any:
        """Initialize a colocated vLLM engine (lazy)."""
        if self._vllm_engine is not None:
            return self._vllm_engine

        vcfg_raw = self._cfg("vllm", {}) or {}
        if not isinstance(vcfg_raw, Mapping):
            raise ValueError("custom.extra.rollout_matching.vllm must be a mapping")
        vcfg = dict(vcfg_raw)

        try:
            import torch.distributed as dist
        except Exception:
            dist = None  # type: ignore[assignment]

        world_size = 1
        if dist is not None and dist.is_available() and dist.is_initialized():
            world_size = int(dist.get_world_size())

        # Defaults: TP=4 on 4-GPU runs; otherwise TP=1 unless explicitly set.
        default_tp = 4 if world_size == 4 else 1
        tp_size = int(vcfg.get("tensor_parallel_size", default_tp))
        if tp_size <= 0:
            raise ValueError(
                "custom.extra.rollout_matching.vllm.tensor_parallel_size must be > 0"
            )
        if world_size % tp_size != 0:
            raise ValueError(
                f"vLLM colocate requires world_size % tp == 0; world_size={world_size} tp={tp_size}"
            )

        max_model_len_raw = vcfg.get("max_model_len", None)
        if max_model_len_raw is None:
            raise ValueError(
                "custom.extra.rollout_matching.vllm.max_model_len is required when rollout_backend=vllm "
                "(it must cover prompt_len + max_new_tokens)."
            )
        max_model_len = int(max_model_len_raw)
        if max_model_len <= 0:
            raise ValueError(
                "custom.extra.rollout_matching.vllm.max_model_len must be > 0"
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
                "custom.extra.rollout_matching.vllm.load_format must be a string"
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
            except Exception as exc:
                raise ValueError(
                    "custom.extra.rollout_matching.vllm.max_num_seqs must be an int"
                ) from exc
            if max_num_seqs <= 0:
                raise ValueError(
                    "custom.extra.rollout_matching.vllm.max_num_seqs must be > 0"
                )

        # Extra vLLM engine kwargs (passed through to vLLM EngineArgs by ms-swift VllmEngine).
        # This is useful to avoid hard-coded vLLM defaults that can break long-context multimodal rollouts.
        vllm_engine_kwargs: Dict[str, Any] = {}
        limit_mm_per_prompt: Optional[Dict[str, int]] = None
        max_num_batched_tokens_raw = vcfg.get("max_num_batched_tokens", None)
        if max_num_batched_tokens_raw is not None:
            try:
                max_num_batched_tokens = int(max_num_batched_tokens_raw)
            except Exception as exc:
                raise ValueError(
                    "custom.extra.rollout_matching.vllm.max_num_batched_tokens must be an int"
                ) from exc
            if max_num_batched_tokens <= 0:
                raise ValueError(
                    "custom.extra.rollout_matching.vllm.max_num_batched_tokens must be > 0"
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
                    "custom.extra.rollout_matching.vllm.kv_cache_dtype must be a string"
                )
            vllm_engine_kwargs["kv_cache_dtype"] = kv_cache_dtype
        if "cpu_offload_gb" in vcfg and vcfg.get("cpu_offload_gb") is not None:
            try:
                cpu_offload_gb = float(vcfg.get("cpu_offload_gb"))
            except Exception as exc:
                raise ValueError(
                    "custom.extra.rollout_matching.vllm.cpu_offload_gb must be a float"
                ) from exc
            if cpu_offload_gb < 0:
                raise ValueError(
                    "custom.extra.rollout_matching.vllm.cpu_offload_gb must be >= 0"
                )
            vllm_engine_kwargs["cpu_offload_gb"] = cpu_offload_gb
        if "swap_space" in vcfg and vcfg.get("swap_space") is not None:
            try:
                swap_space = float(vcfg.get("swap_space"))
            except Exception as exc:
                raise ValueError(
                    "custom.extra.rollout_matching.vllm.swap_space must be a float"
                ) from exc
            if swap_space < 0:
                raise ValueError(
                    "custom.extra.rollout_matching.vllm.swap_space must be >= 0"
                )
            vllm_engine_kwargs["swap_space"] = swap_space
        if (
            "limit_mm_per_prompt" in vcfg
            and vcfg.get("limit_mm_per_prompt") is not None
        ):
            limit_raw = vcfg.get("limit_mm_per_prompt")
            if not isinstance(limit_raw, Mapping):
                raise ValueError(
                    "custom.extra.rollout_matching.vllm.limit_mm_per_prompt must be a mapping"
                )
            limit_parsed: Dict[str, int] = {}
            for k, v in limit_raw.items():
                if not isinstance(k, str):
                    raise ValueError(
                        "custom.extra.rollout_matching.vllm.limit_mm_per_prompt keys must be strings"
                    )
                try:
                    limit_parsed[k] = int(v)
                except Exception as exc:
                    raise ValueError(
                        "custom.extra.rollout_matching.vllm.limit_mm_per_prompt values must be ints"
                    ) from exc
            # NOTE: ms-swift's `VllmEngine` already exposes `limit_mm_per_prompt` as a
            # top-level kwarg, and forwards it to `_prepare_engine_kwargs`. Passing it
            # again via `engine_kwargs` would cause a `got multiple values` TypeError.
            limit_mm_per_prompt = limit_parsed

        if "mm_encoder_tp_mode" in vcfg and vcfg.get("mm_encoder_tp_mode") is not None:
            mm_encoder_tp_mode = vcfg.get("mm_encoder_tp_mode")
            if not isinstance(mm_encoder_tp_mode, str):
                raise ValueError(
                    "custom.extra.rollout_matching.vllm.mm_encoder_tp_mode must be a string"
                )
            mm_encoder_tp_mode = mm_encoder_tp_mode.strip().lower()
            if mm_encoder_tp_mode not in {"weights", "data"}:
                raise ValueError(
                    "custom.extra.rollout_matching.vllm.mm_encoder_tp_mode must be 'weights' or 'data'"
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
            except Exception as exc:
                raise RuntimeError(
                    "vLLM rollout backend is enabled but vLLM is unavailable or incompatible. "
                    "Install/upgrade vLLM in the ms env, or set custom.extra.rollout_matching.rollout_backend: hf."
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
                except Exception:
                    pass

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
        except Exception:
            pass

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
        except Exception as exc:
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
            except Exception:
                pass

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
        except Exception:
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
            except Exception:
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
                    except Exception as exc:
                        raise RuntimeError(
                            "vLLM LoRA is disabled, but we failed to merge the adapter weights from the training "
                            "model. Mitigations: set custom.extra.rollout_matching.vllm.enable_lora=true "
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
        except Exception:
            pass

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
        except Exception:
            if hasattr(cfg0, "to_dict"):
                peft_cfg_dict = cfg0.to_dict()  # type: ignore[assignment]
            else:
                peft_cfg_dict = {}

        try:
            from peft.utils.save_and_load import get_peft_model_state_dict
        except Exception as exc:
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
        except Exception as exc:
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
            except Exception:
                pass
        except Exception as exc:
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
            except Exception:
                dist = None  # type: ignore[assignment]

            if dist is not None and dist.is_available() and dist.is_initialized():
                try:
                    rank = int(dist.get_rank())
                    world_size = int(dist.get_world_size())
                except Exception:
                    rank = 0
                    world_size = 1

            servers = self._vllm_server_specs()
            timeout_s, _infer_timeout_s = self._vllm_server_timeouts()

            try:
                from swift.trainers.rlhf_trainer.vllm_client import VLLMClient
            except Exception as exc:
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
            except Exception as exc:
                raise RuntimeError(
                    "Failed to connect to vLLM rollout server(s). "
                    "Check custom.extra.rollout_matching.vllm.server (base_url/group_port) and ensure /health/ is reachable."
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
            except Exception:
                pass

            self._vllm_server_client = client
            return client

    def _ensure_vllm_server_communicator_rank0(self, client: Any) -> None:
        """Initialize vLLM server NCCL communicator (rank0-only under DDP)."""
        if bool(getattr(self, "_vllm_server_comm_inited", False)):
            return

        rank = 0
        try:
            import torch.distributed as dist
        except Exception:
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
        except Exception as exc:
            raise RuntimeError(
                "Failed to initialize NCCL communicator with vLLM rollout server(s). "
                "Mitigations: verify group_port reachability, set NCCL env, or increase vllm.server.timeout_s."
            ) from exc

        setattr(self, "_vllm_server_comm_inited", True)

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
    ) -> None:
        """Optional raw rollout dump for diagnosing vLLM server-mode formatting.

        Controlled via:
          custom.extra.rollout_matching.vllm.server.debug_dump:
            enabled: true
            every_steps: 10           # defaults to args.logging_steps
            dump_first_step: false    # defaults to args.logging_first_step
            only_world_process_zero: true
            max_events: 3
            max_samples: 1
            max_chars: 4000
            out_dir: null  # defaults to <output_dir>/vllm_server_debug

        Notes:
        - In DDP, the default is rank0-only dumping to avoid I/O storms.
        - If only_world_process_zero=false, dumps go to per-rank subdirectories.
        """

        try:
            scfg_raw = self._vllm_server_cfg()
        except Exception:
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
        except Exception:
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
        max_chars = int(debug_raw.get("max_chars", 4000) or 4000)

        tok = self.template.tokenizer

        samples_dump: List[Dict[str, Any]] = []
        for i, (req, out) in enumerate(zip(infer_requests, outputs)):
            if i >= max_samples:
                break

            resp_ids, resp_text, decode_mode, prompt_ids = out

            parse_dump: Dict[str, Any] = {}
            try:
                parse = parse_rollout_for_matching(
                    tokenizer=tok, response_token_ids=list(resp_ids)
                )
                parse_dump = {
                    "truncated": bool(parse.truncated),
                    "dropped_invalid": int(parse.dropped_invalid),
                    "dropped_ambiguous": int(parse.dropped_ambiguous),
                    "response_len": int(len(parse.response_token_ids)),
                    "prefix_len": int(len(parse.prefix_token_ids)),
                    "response_text_head": str(parse.response_text)[:max_chars],
                    "prefix_text_head": str(parse.prefix_text)[:max_chars],
                    "valid_objects": [
                        {
                            "key": str(o.key),
                            "index": int(o.index),
                            "geom_type": str(o.geom_type),
                            "coord_token_count": int(len(o.coord_token_indices)),
                            "coord_token_indices_head": [
                                int(x) for x in list(o.coord_token_indices)[:32]
                            ],
                        }
                        for o in list(parse.valid_objects)[:16]
                    ],
                }
            except Exception as exc:
                parse_dump = {"error": repr(exc)}

            samples_dump.append(
                {
                    "i": int(i),
                    "messages": req.get("messages"),
                    "images": req.get("images"),
                    "decode_mode": str(decode_mode),
                    "seed_base": int(seed_base),
                    "prompt_len": int(len(prompt_ids)),
                    "prompt_ids_head": [int(x) for x in list(prompt_ids)[:64]],
                    "resp_text_head": str(resp_text)[:max_chars],
                    "resp_len": int(len(resp_ids)),
                    "resp_ids_head": [int(x) for x in list(resp_ids)[:256]],
                    "parse": parse_dump,
                }
            )

        payload = {
            "global_step": int(gs),
            "seed_base": int(seed_base),
            "event": int(event),
            "rank": int(rank),
            "world_size": int(world_size),
            "max_samples": int(max_samples),
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
        except Exception:
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
        except Exception:
            dist = None  # type: ignore[assignment]

        if dist is not None and dist.is_available() and dist.is_initialized():
            rank = int(dist.get_rank())
            world_size = int(dist.get_world_size())

        last = int(getattr(self, "_vllm_server_last_synced_step", -1))
        need_sync = int(step != last)

        # Under DDP, ensure all ranks take the same early-return decision.
        if dist is not None and dist.is_available() and dist.is_initialized() and int(world_size) > 1:
            try:
                try:
                    backend = str(dist.get_backend()).lower()
                except Exception:
                    backend = ""

                reduce_device = torch.device("cpu")
                if backend == "nccl" and torch.cuda.is_available():
                    reduce_device = self.model.device

                flag = torch.tensor([need_sync], device=reduce_device, dtype=torch.int32)
                dist.broadcast(flag, src=0)
                need_sync = int(flag.item())
            except Exception:
                need_sync = int(step != last)

        if need_sync == 0:
            return

        eff_mode = self._effective_vllm_server_sync_mode()
        if int(world_size) > 1 and eff_mode != "full":
            raise ValueError(
                "custom.extra.rollout_matching.vllm.sync.mode must resolve to 'full' under multi-process learners "
                f"(world_size={int(world_size)}). Got effective sync mode={eff_mode!r}."
            )

        # Single-process learner: allow adapter/full sync modes.
        if dist is None or (not dist.is_available()) or (not dist.is_initialized()) or int(world_size) == 1:
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
                    "custom.extra.rollout_matching.vllm.sync.mode=adapter requires custom.extra.rollout_matching.vllm.enable_lora: true"
                )

            client = self._ensure_vllm_server_client()
            if not bool(getattr(self, "_vllm_server_comm_inited", False)):
                try:
                    client.init_communicator(device=0)
                    setattr(self, "_vllm_server_comm_inited", True)
                except Exception as exc:
                    raise RuntimeError(
                        "Failed to initialize NCCL communicator with vLLM rollout server(s). "
                        "Mitigations: verify group_port reachability, set NCCL env, or increase vllm.server.timeout_s."
                    ) from exc

            if eff_mode == "adapter":
                # Optional runtime sanity check (server must have vLLM LoRA enabled).
                try:
                    info = client.get_engine_type()
                    if isinstance(info, dict) and not bool(info.get("enable_lora", False)):
                        raise RuntimeError(
                            "vLLM server reports enable_lora=false, but adapter-only sync was requested. "
                            "Launch the rollout server with vLLM LoRA enabled (e.g. swift rollout --vllm_enable_lora true), "
                            "or set vllm.sync.mode=full."
                        )
                except Exception as exc:
                    # If the check itself fails, continue; sync will fail with a clearer error.
                    logger.warning(
                        "Unable to verify rollout server LoRA capability: %s", exc
                    )

                try:
                    self._sync_vllm_server_adapter(client)
                except Exception as exc:
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
        except Exception:
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
            except Exception:
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
                    except Exception as exc:
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
        except Exception as exc:
            logger.warning(
                "Failed to reset vLLM server prefix cache after full sync: %s", exc
            )

    def _vllm_server_update_state_dict(
        self, client: Any, state_dict: Mapping[str, Any]
    ) -> None:
        """Bucket + broadcast a state_dict into the vLLM server via NCCL."""
        try:
            from swift.trainers.rlhf_trainer.utils import FlattenedTensorBucket
        except Exception as exc:
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
        except Exception:
            if hasattr(cfg0, "to_dict"):
                peft_cfg_dict = cfg0.to_dict()  # type: ignore[assignment]
            else:
                peft_cfg_dict = {}

        try:
            from peft.utils.save_and_load import get_peft_model_state_dict
        except Exception as exc:
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
        except Exception as exc:
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
        except Exception as exc:
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
                except Exception as exc:
                    raise ValueError(
                        "custom.extra.rollout_matching.vllm.infer_batch_size must be an int"
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
                "custom.extra.rollout_matching.repetition_penalty must be > 0"
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
        mb = self._rollout_generate_batch_size()

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
            repeat_cfg_raw = self._cfg("repeat_terminate", None)
            if isinstance(repeat_cfg_raw, Mapping) and bool(
                repeat_cfg_raw.get("enabled", False)
            ):
                min_new_tokens = int(repeat_cfg_raw.get("min_new_tokens", 256) or 0)
                max_consecutive = int(
                    repeat_cfg_raw.get("max_consecutive_token_repeats", 64) or 0
                )
                ngram_size = int(repeat_cfg_raw.get("ngram_size", 64) or 0)
                ngram_repeats = int(repeat_cfg_raw.get("ngram_repeats", 2) or 2)
                max_object_keys_raw = repeat_cfg_raw.get("max_object_keys")
                max_object_keys = (
                    int(max_object_keys_raw)
                    if max_object_keys_raw is not None
                    else None
                )

                obj_prefix_ids = None
                if max_object_keys is not None:
                    try:
                        obj_prefix_ids = tok.encode(
                            '"object_', add_special_tokens=False
                        )
                    except Exception:
                        obj_prefix_ids = None

                eos_id = int(getattr(tok, "eos_token_id", -1) or -1)
                if eos_id >= 0:
                    guard = _ForceEosOnRepeatGuard(
                        eos_token_id=eos_id,
                        prompt_len=prompt_pad_len,
                        min_new_tokens=min_new_tokens,
                        max_consecutive_token_repeats=max_consecutive,
                        ngram_size=ngram_size,
                        ngram_repeats=ngram_repeats,
                        max_object_keys=max_object_keys,
                        object_key_prefix_token_ids=obj_prefix_ids,
                    )
                    try:
                        from transformers.generation.logits_process import (
                            LogitsProcessorList,
                        )

                        logits_processor = LogitsProcessorList([guard])
                    except Exception:
                        # Fallback: transformers may accept a plain list.
                        logits_processor = [guard]

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
        self, samples: Sequence[Mapping[str, Any]]
    ) -> List[Tuple[List[int], str, str, List[int]]]:
        """vLLM rollout backend (colocate default, server optional)."""
        mode = self._vllm_mode()
        if mode == "server":
            return self._rollout_many_vllm_server(samples)
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
                "custom.extra.rollout_matching.repetition_penalty must be > 0"
            )

        try:
            from swift.llm import RequestConfig
        except Exception as exc:
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
        except Exception as exc:
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
            except Exception:
                text = ""
                token_ids = []
                prompt_ids = []
            results.append((token_ids, text, decode_mode, prompt_ids))
        return results

    @torch.no_grad()
    def _rollout_many_vllm_server(
        self, samples: Sequence[Mapping[str, Any]]
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
        # Stage2-AB async actor-learner issues server infer calls from background
        # prefetch workers; those paths MUST NOT invoke DDP collectives here.
        if not bool(getattr(self, "_stage2_async_skip_vllm_server_sync", False)):
            self._sync_vllm_server_rollout_model_if_needed()

        max_new_tokens = int(self._cfg("max_new_tokens", 512))
        temperature, top_p, top_k = self._decoding_params()
        repetition_penalty = float(self._cfg("repetition_penalty", 1.0) or 1.0)
        if repetition_penalty <= 0:
            raise ValueError(
                "custom.extra.rollout_matching.repetition_penalty must be > 0"
            )

        try:
            from swift.llm import RequestConfig
        except Exception as exc:
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
            except Exception as exc:
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

        # Deterministic contiguous chunking across servers.
        chunks = _contiguous_chunk_slices(int(len(infer_requests)), int(len(servers)))

        # Log reproducibility metadata once per optimizer step (E-steps only).
        if gs != int(getattr(self, "_vllm_server_last_logged_step", -1)):
            seed_plan: List[Dict[str, Any]] = []
            for i, (start, end) in enumerate(chunks):
                if start >= end:
                    continue
                seed_plan.append(
                    {
                        "server_idx": int(i),
                        "base_url": str(servers[i].get("base_url", "")),
                        "start": int(start),
                        "end": int(end),
                        # Effective per-server-call seed used for RequestConfig.seed:
                        # seed = rollout_seed_base + chunk_start
                        "seed": int((seed_base + int(start)) & 0x7FFFFFFF),
                    }
                )
            logger.info(
                "vLLM server rollout metadata: servers=%s sync_mode=%s request_n=%s rollout_seed_base=%s seed_plan=%s",
                servers,
                self._effective_vllm_server_sync_mode(),
                int(len(infer_requests)),
                int(seed_base),
                seed_plan,
            )
            self._vllm_server_last_logged_step = int(gs)

        results: List[Optional[Tuple[List[int], str, str, List[int]]]] = [None] * len(
            infer_requests
        )

        def _parse_output(raw: Any) -> Tuple[List[int], str, List[int]]:
            # Support both direct ChatCompletionResponse dicts and wrapped RolloutOutput dicts.
            if isinstance(raw, dict) and isinstance(raw.get("response"), dict):
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
            req_cfg["seed"] = int((seed_base + int(start)) & 0x7FFFFFFF)

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
            req_timeout_s = float(infer_timeout_s)
            # Use a (connect, read) timeout tuple to prevent indefinite hangs on broken keep-alive sockets.
            req_timeout = (min(10.0, req_timeout_s), req_timeout_s)
            try:
                with self._vllm_server_infer_guard():
                    resp = session.post(url, json=payload, timeout=req_timeout)
            except Exception as exc:
                # Retry once with a fresh session. This helps when the server was idle for A steps
                # (AAB schedules) and the underlying keep-alive connection was dropped.
                try:
                    import requests

                    client.sessions[server_idx] = requests.Session()
                    session = client.sessions[server_idx]
                    with self._vllm_server_infer_guard():
                        resp = session.post(url, json=payload, timeout=req_timeout)
                except Exception as exc2:
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
                results[int(start + j)] = (token_ids, text, decode_mode, prompt_ids)

        # Parallelize across servers.
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=int(len(servers))) as ex:
            futs = []
            for i, (start, end) in enumerate(chunks):
                if start >= end:
                    continue
                futs.append(ex.submit(_infer_on_server, int(i), int(start), int(end)))
            for f in futs:
                f.result()

        out: List[Tuple[List[int], str, str, List[int]]] = []
        for r in results:
            if r is None:
                raise RuntimeError(
                    "vLLM server failed to produce outputs for all requests"
                )
            out.append(r)

        self._maybe_debug_dump_vllm_server_rollouts(
            global_step=gs,
            seed_base=seed_base,
            infer_requests=infer_requests,
            outputs=out,
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
                except Exception:
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
            return self._rollout_many_vllm(samples_for_rollout)
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
        except Exception:
            pass

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
        except Exception:
            pass

        return selected, pack_metrics

    def _schedule_post_rollout_packs_window(
        self,
        *,
        window_segments: List[Tuple[Dict[str, Any], Dict[str, Any], int]],
        gas: int,
    ) -> Tuple[
        List[List[Tuple[Dict[str, Any], Dict[str, Any], int]]], Dict[str, float]
    ]:
        packing_length = int(self._packing_length())
        gas = int(gas)
        encoded_lens = [int(sl) for _, _, sl in window_segments]

        pack_indices = schedule_post_rollout_segment_indices_window(
            encoded_lens=encoded_lens,
            packing_length=packing_length,
            gas=gas,
            select_indices_fn=self._select_post_rollout_segment_indices,
        )

        packs: List[List[Tuple[Dict[str, Any], Dict[str, Any], int]]] = []
        fill_sum = 0.0
        selected_total_len_sum = 0.0
        segments_sum = 0

        for sel in pack_indices:
            selected = [window_segments[i] for i in sel]
            sel_total = int(sum(int(encoded_lens[i]) for i in sel))
            fill = (
                float(sel_total) / float(packing_length) if packing_length > 0 else 0.0
            )
            packs.append(selected)
            fill_sum += float(fill)
            selected_total_len_sum += float(sel_total)
            segments_sum += int(len(selected))

        total_len = int(sum(encoded_lens))
        metrics: Dict[str, float] = {
            "packing/window_segments_total": float(len(window_segments)),
            "packing/window_sum_encoded_len": float(total_len),
            "packing/window_packs": float(gas),
            "packing/window_nonempty_packs": float(sum(1 for p in packs if p)),
            "packing/window_avg_fill": float(fill_sum / float(max(1, gas))),
            "packing/window_selected_total_len_sum": float(selected_total_len_sum),
            "packing/window_segments_sum": float(segments_sum),
        }
        return packs, metrics

    def _prepare_window_packed_batches(
        self,
        *,
        window_raw_micro_batches: List[List[Mapping[str, Any]]],
        global_step: int,
    ) -> List[Dict[str, Any]]:
        """Build packed prepared batches for a full accumulation window.

        Implementation note: to keep changes localized, we reuse `_prepare_batch_inputs`
        to build per-micro post-rollout segments (without packing), then schedule and
        repack within the window.
        """
        gas = int(len(window_raw_micro_batches))
        if gas <= 0:
            raise ValueError("window_raw_micro_batches is empty")

        # Collect segments for the whole window.
        window_segments: List[Tuple[Dict[str, Any], Dict[str, Any], int]] = []
        per_micro_metrics: List[Dict[str, float]] = []

        for mb in window_raw_micro_batches:
            segs, bm = self._prepare_batch_inputs(mb, _segments_only=True)
            window_segments.extend(segs)
            per_micro_metrics.append(bm)

        # Schedule segments into exactly `gas` micro-packs.
        t_pack0 = time.perf_counter()
        packs, window_pack_metrics = self._schedule_post_rollout_packs_window(
            window_segments=window_segments,
            gas=gas,
        )
        t_pack_s = float(time.perf_counter() - t_pack0)

        template = self.template
        from swift.llm import to_device

        prepared: List[Dict[str, Any]] = []
        packing_length = int(self._packing_length())

        for i, selected in enumerate(packs):
            if not selected:
                raise ValueError(
                    "window post-rollout packing produced an empty micro-pack. "
                    "This indicates the window did not produce enough post-rollout segments."
                )

            with self._template_packing_enabled():
                packed = template.data_collator([enc for enc, _, _ in selected])
            batch = to_device(packed, self.model.device)
            self._assert_single_packed_forward(batch, where="rollout_matching/window_pack")
            batch["_rollout_matching_meta"] = [m for _, m, _ in selected]

            bm = dict(per_micro_metrics[i]) if i < len(per_micro_metrics) else {}
            bm["time/post_rollout_pack_s"] = float(t_pack_s if i == 0 else 0.0)

            sel_total = int(sum(int(sl) for _, _, sl in selected))
            fill = (
                float(sel_total) / float(packing_length) if packing_length > 0 else 0.0
            )
            bm.update(
                {
                    "packing/post_rollout_fill": float(fill),
                    "packing/post_rollout_selected_total_len": float(sel_total),
                    "packing/post_rollout_segments": float(len(selected)),
                    "packing/post_rollout_buffer": float(0.0),
                }
            )

            if i == 0:
                bm.update(window_pack_metrics)
                bm["packing/post_rollout_scope_window"] = 1.0
            else:
                bm["packing/post_rollout_scope_window"] = 0.0

            self._merge_rollout_matching_batch_metrics(batch, bm)
            prepared.append(batch)

        return prepared

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
                tokenizer=tok, response_token_ids=resp_ids
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
                    except Exception:
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
                        except Exception:
                            enc = None

                        if enc is not None:
                            # Best-effort: if model load fails (missing cache/network), skip semantics.
                            try:
                                emb = enc.encode_norm_texts(sorted(uniq))
                            except Exception:
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

            fn_gt_indices_final = [
                i for i in range(len(gts)) if i not in matched_gt_for_supervision
            ]
            fn_objs = [gts[i] for i in fn_gt_indices_final]

            # 4) Serialize append fragment (mandatory FN append) and build Y_train ids
            max_idx = parse.max_object_index_in_prefix
            start_idx = (max_idx + 1) if max_idx is not None else 1
            append_text = _serialize_append_fragment(
                fn_objects=fn_objs, start_index=start_idx, prefix_text=parse.prefix_text
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
            except Exception:
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
                except Exception:
                    pass

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
            except Exception:
                pass

        if packing_enabled:
            self._append_post_rollout_segments(segments)

            t_pack0 = time.perf_counter()
            selected, pack_metrics = self._pop_post_rollout_pack()
            with self._template_packing_enabled():
                packed = template.data_collator([enc for enc, _, _ in selected])
            batch = to_device(packed, self.model.device)
            self._assert_single_packed_forward(batch, where="rollout_matching/_prepare_batch_inputs")
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
                    sample_step = int(len(getattr(pending, "meta", []) or []))
                    try:
                        self._rm_train_samples_seen = (
                            int(getattr(self, "_rm_train_samples_seen", 0) or 0)
                            + sample_step
                        )
                    except Exception:
                        self._rm_train_samples_seen = sample_step

                    logs.update(self._build_train_rollout_log_payload(pending))
                    logs["train/samples_seen"] = float(
                        getattr(self, "_rm_train_samples_seen", 0) or 0
                    )
        except Exception:
            pass

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
                except Exception:
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
            except Exception:
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
            # Backward-compat alias: this is recall (GT coverage).
            "rollout/match_rate": float(recall),
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
            "rollout/decode_greedy": float(
                sum(1 for m in meta if m.get("decode_mode") == "greedy")
            ),
            "rollout/decode_beam": float(
                sum(1 for m in meta if m.get("decode_mode") == "beam")
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
        except Exception:
            pass

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
        except Exception:
            pass

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
        except Exception:
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
        except Exception:
            pass

        return (total, outputs) if return_outputs else total

    def get_train_dataloader(self):
        dl = super().get_train_dataloader()

        try:
            per_dev = int(getattr(self.args, "per_device_train_batch_size", 1) or 1)
        except Exception:
            per_dev = 1

        # Fixed raw batching for post-rollout packing.
        #
        # Stage_2 trainers use identity collator, so the dataloader yields lists of raw
        # samples. With per_device_train_batch_size==1 this would otherwise produce only
        # one segment per micro-step, making it impossible to reach reasonable packing
        # fill ratios.
        #
        # We use `custom.extra.rollout_matching.rollout_generate_batch_size` to control
        # how many raw samples feed into ONE packed sequence (learner microbatch stays 1).
        try:
            rollout_gen_bs = int(self._cfg("rollout_generate_batch_size", 1) or 1)
        except Exception:
            rollout_gen_bs = 1
        if self._packing_enabled() and per_dev == 1 and int(rollout_gen_bs) > 1:
            dl = _FixedRawMicroBatchStacker(
                dl,
                target_raw_batch_size=int(rollout_gen_bs),
                base_raw_batch_size=int(per_dev),
            )


        gas = int(getattr(self.args, "gradient_accumulation_steps", 1) or 1)

        # Optional window-aware post-rollout packing requires lookahead over one accumulation window.
        if self._packing_enabled() and self._post_rollout_pack_scope() == "window":
            dl = _AccumulationWindowLookahead(dl, gas=gas)

        # Drop the final partial accumulation window when requested.
        #
        # This keeps optimizer-step semantics consistent for step-budgeted stage_2 runs
        # (fixed samples per step) and avoids a trailing underfull/no-op step.
        try:
            drop_last = bool(getattr(self.args, "dataloader_drop_last", False))
        except Exception:
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
        except Exception:
            normalize_desc = None  # type: ignore[assignment]

        sem_loaded_local = 0.0
        sem_encoder = None
        if desc_enabled and desc_mode in {"semantic", "both"}:
            try:
                sem_encoder = self._get_desc_semantic_encoder(desc_cfg)
            except Exception:
                sem_encoder = None
            if sem_encoder is not None:
                # Probe once so failures are detected consistently across ranks.
                try:
                    _ = sem_encoder.encode_norm_texts(["__probe__"])
                    sem_loaded_local = 1.0
                except Exception:
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
                        tokenizer=tok, response_token_ids=resp_ids
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
                            except Exception:
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
        except Exception:
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

        metrics: Dict[str, float] = {}
        metrics[f"{metric_key_prefix}_runtime"] = float(runtime)
        if runtime > 0:
            metrics[f"{metric_key_prefix}_samples_per_second"] = float(
                n_samples / runtime
            )
            metrics[f"{metric_key_prefix}_steps_per_second"] = float(n_steps / runtime)

        metrics[f"{metric_key_prefix}_rollout_precision"] = float(precision)
        metrics[f"{metric_key_prefix}_rollout_recall"] = float(recall)
        metrics[f"{metric_key_prefix}_rollout_f1"] = float(f1)

        metrics[f"{metric_key_prefix}_rollout_pred_objects"] = float(pred_total)
        metrics[f"{metric_key_prefix}_rollout_gt_objects"] = float(gt_total)
        metrics[f"{metric_key_prefix}_rollout_matched"] = float(matched_total)
        metrics[f"{metric_key_prefix}_rollout_fp"] = float(fp_total)
        metrics[f"{metric_key_prefix}_rollout_fn"] = float(fn_total)
        metrics[f"{metric_key_prefix}_rollout_gating_rejections"] = float(
            gating_rejections_total
        )

        metrics[f"{metric_key_prefix}_rollout_parse_dropped_invalid"] = float(
            dropped_invalid_total
        )
        metrics[f"{metric_key_prefix}_rollout_parse_dropped_ambiguous"] = float(
            dropped_ambiguous_total
        )
        metrics[f"{metric_key_prefix}_rollout_parse_truncated_rate"] = (
            float(trunc_samples / n_samples) if n_samples > 0 else 0.0
        )

        metrics[f"{metric_key_prefix}_rollout_sample_valid_pred_rate"] = (
            float(n_samples_valid_pred / n_samples) if n_samples > 0 else 0.0
        )
        metrics[f"{metric_key_prefix}_rollout_sample_any_match_rate"] = (
            float(n_samples_any_match / n_samples) if n_samples > 0 else 0.0
        )

        metrics[f"{metric_key_prefix}_rollout_matched_maskiou_mean"] = (
            float(matched_iou_sum / matched_iou_count) if matched_iou_count > 0 else 0.0
        )

        # Desc monitor outputs (matched pairs only).
        if desc_enabled:
            metrics[f"{metric_key_prefix}_rollout_desc_pairs_total"] = float(
                desc_pairs_total
            )
            exact_acc = (
                float(desc_exact_ok_total / desc_pairs_total)
                if desc_pairs_total > 0
                else 1.0
            )
            metrics[f"{metric_key_prefix}_rollout_desc_exact_acc_on_matched"] = float(
                exact_acc
            )

            sem_enabled = bool(sem_loaded_sum >= float(world_size) - 0.5)
            metrics[f"{metric_key_prefix}_rollout_desc_sem_enabled"] = float(
                1.0 if sem_enabled else 0.0
            )
            if sem_enabled:
                sem_acc = (
                    float(desc_sem_ok_total / desc_pairs_total)
                    if desc_pairs_total > 0
                    else 1.0
                )
                metrics[f"{metric_key_prefix}_rollout_desc_sem_acc_on_matched"] = float(
                    sem_acc
                )
                if desc_sem_sim_count_total > 0:
                    metrics[f"{metric_key_prefix}_rollout_desc_sem_sim_mean"] = float(
                        desc_sem_sim_sum_total / desc_sem_sim_count_total
                    )
                    metrics[f"{metric_key_prefix}_rollout_desc_sem_sim_count"] = float(
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

        gs = int(getattr(getattr(self, "state", None), "global_step", 0) or 0)

        packing_enabled = bool(self._packing_enabled())
        if (
            packing_enabled
            and self._post_rollout_pack_scope() == "window"
            and isinstance(inputs, _WindowedMicroBatch)
        ):
            win = inputs.rm_window
            idx = int(getattr(inputs, "rm_window_idx", 0) or 0)

            def _build_all() -> List[Dict[str, Any]]:
                return self._prepare_window_packed_batches(
                    window_raw_micro_batches=win.raw_micro_batches,
                    global_step=gs,
                )

            prepared = win.get_prepared(idx=idx, build_all_prepared=_build_all)
        else:
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
