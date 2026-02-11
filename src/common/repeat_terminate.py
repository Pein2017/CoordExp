from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Sequence

import torch


@dataclass(frozen=True)
class RepeatTerminateConfig:
    enabled: bool
    min_new_tokens: int
    max_consecutive_token_repeats: int
    ngram_size: int
    ngram_repeats: int
    max_object_keys: Optional[int]


def parse_repeat_terminate_config(raw_cfg: Any) -> RepeatTerminateConfig:
    if not isinstance(raw_cfg, Mapping):
        return RepeatTerminateConfig(
            enabled=False,
            min_new_tokens=0,
            max_consecutive_token_repeats=0,
            ngram_size=0,
            ngram_repeats=0,
            max_object_keys=None,
        )

    enabled = bool(raw_cfg.get("enabled", False))
    min_new_tokens = int(raw_cfg.get("min_new_tokens", 256) or 0)
    max_consecutive = int(raw_cfg.get("max_consecutive_token_repeats", 64) or 0)
    ngram_size = int(raw_cfg.get("ngram_size", 64) or 0)
    ngram_repeats = int(raw_cfg.get("ngram_repeats", 2) or 0)
    max_object_keys_raw = raw_cfg.get("max_object_keys")
    max_object_keys = (
        int(max_object_keys_raw) if max_object_keys_raw is not None else None
    )

    return RepeatTerminateConfig(
        enabled=enabled,
        min_new_tokens=min_new_tokens,
        max_consecutive_token_repeats=max_consecutive,
        ngram_size=ngram_size,
        ngram_repeats=ngram_repeats,
        max_object_keys=max_object_keys,
    )


def encode_object_key_prefix(tokenizer: Any) -> Optional[List[int]]:
    try:
        ids = tokenizer.encode('"object_', add_special_tokens=False)
    except Exception:
        return None
    if not isinstance(ids, list) or not ids:
        return None
    try:
        return [int(x) for x in ids]
    except Exception:
        return None


def _count_subsequence_occurrences(
    token_ids: Sequence[int],
    pattern: Optional[Sequence[int]],
) -> int:
    if not pattern:
        return 0
    pat = [int(x) for x in pattern]
    if not pat:
        return 0
    n = len(pat)
    if len(token_ids) < n:
        return 0

    count = 0
    for i in range(0, len(token_ids) - n + 1):
        ok = True
        for j in range(n):
            if int(token_ids[i + j]) != int(pat[j]):
                ok = False
                break
        if ok:
            count += 1
    return count


def should_trigger_repeat_terminate(
    *,
    generated_token_ids: Sequence[int],
    cfg: RepeatTerminateConfig,
    object_key_prefix_token_ids: Optional[Sequence[int]],
) -> bool:
    gen_len = int(len(generated_token_ids))
    if gen_len < int(cfg.min_new_tokens):
        return False

    if cfg.max_object_keys is not None and cfg.max_object_keys >= 0:
        obj_count = _count_subsequence_occurrences(
            generated_token_ids,
            object_key_prefix_token_ids,
        )
        if int(obj_count) >= int(cfg.max_object_keys):
            return True

    if int(cfg.max_consecutive_token_repeats) > 0 and gen_len > 0:
        last = int(generated_token_ids[-1])
        run = 1
        limit = min(int(cfg.max_consecutive_token_repeats), int(gen_len))
        for j in range(2, limit + 1):
            if int(generated_token_ids[-j]) != last:
                break
            run += 1
        if run >= int(cfg.max_consecutive_token_repeats):
            return True

    n = int(cfg.ngram_size)
    reps = int(cfg.ngram_repeats)
    if n > 0 and reps >= 2 and gen_len >= n * reps:
        tail = [int(x) for x in generated_token_ids[-n:]]
        ok = True
        for k in range(2, reps + 1):
            start = int(gen_len - k * n)
            end = int(gen_len - (k - 1) * n)
            prev = [int(x) for x in generated_token_ids[start:end]]
            if prev != tail:
                ok = False
                break
        if ok:
            return True

    return False


class ForceEosOnRepeatBatchGuard:
    """Transformers logits-processor that forces EOS for offending sequences only."""

    def __init__(
        self,
        *,
        eos_token_id: int,
        prompt_len: int,
        cfg: RepeatTerminateConfig,
        object_key_prefix_token_ids: Optional[List[int]],
    ) -> None:
        self.eos_token_id = int(eos_token_id)
        self.prompt_len = int(prompt_len)
        self.cfg = cfg
        self.object_key_prefix_token_ids = object_key_prefix_token_ids

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
        if self.cfg.max_object_keys is None or not self.object_key_prefix_token_ids:
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
            for pos in range(start, cur_len):
                tid = int(input_ids[i, pos].item())
                if tid == int(pat[match]):
                    match += 1
                    if match >= len(pat):
                        count += 1
                        match = 0
                else:
                    match = 1 if tid == int(pat[0]) else 0

            self._obj_match_idx[i] = match
            self._obj_counts[i] = count
            self._processed_lens[i] = cur_len

    def _should_force_eos_for_seq(self, seq: torch.Tensor, *, obj_count: int) -> bool:
        gen_len = int(seq.numel()) - int(self.prompt_len)
        if gen_len < int(self.cfg.min_new_tokens):
            return False

        if self.cfg.max_object_keys is not None and int(obj_count) >= int(
            self.cfg.max_object_keys
        ):
            return True

        if int(self.cfg.max_consecutive_token_repeats) > 0 and seq.numel() > 0:
            last = int(seq[-1].item())
            run = 1
            limit = min(int(self.cfg.max_consecutive_token_repeats), int(seq.numel()))
            for j in range(2, limit + 1):
                if int(seq[-j].item()) != last:
                    break
                run += 1
            if run >= int(self.cfg.max_consecutive_token_repeats):
                return True

        n = int(self.cfg.ngram_size)
        reps = int(self.cfg.ngram_repeats)
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
            if self._should_force_eos_for_seq(input_ids[i], obj_count=int(obj_counts[i])):
                force[i] = True

        if not bool(force.any().item()):
            return scores

        scores_out = scores.clone()
        scores_out[force, :] = -float("inf")
        scores_out[force, int(self.eos_token_id)] = 0.0
        return scores_out


class ForceEosOnRepeatSequenceGuard:
    """vLLM logits-processor variant (single-sequence signature)."""

    def __init__(
        self,
        *,
        eos_token_id: int,
        cfg: RepeatTerminateConfig,
        object_key_prefix_token_ids: Optional[Sequence[int]],
    ) -> None:
        self.eos_token_id = int(eos_token_id)
        self.cfg = cfg
        self.object_key_prefix_token_ids = (
            [int(x) for x in object_key_prefix_token_ids]
            if object_key_prefix_token_ids is not None
            else None
        )
        self.triggered = False

        self._processed_len = 0
        self._obj_count = 0
        self._obj_match_idx = 0

    def _update_object_key_count(self, generated_token_ids: Sequence[int]) -> None:
        if self.cfg.max_object_keys is None or not self.object_key_prefix_token_ids:
            return

        pat = self.object_key_prefix_token_ids
        start = int(min(max(self._processed_len, 0), len(generated_token_ids)))
        match = int(self._obj_match_idx)
        count = int(self._obj_count)

        for pos in range(start, len(generated_token_ids)):
            tid = int(generated_token_ids[pos])
            if tid == int(pat[match]):
                match += 1
                if match >= len(pat):
                    count += 1
                    match = 0
            else:
                match = 1 if tid == int(pat[0]) else 0

        self._obj_match_idx = int(match)
        self._obj_count = int(count)
        self._processed_len = int(len(generated_token_ids))

    def _should_force(self, generated_token_ids: Sequence[int]) -> bool:
        gen_len = int(len(generated_token_ids))
        if gen_len < int(self.cfg.min_new_tokens):
            return False

        if self.cfg.max_object_keys is not None and int(self._obj_count) >= int(
            self.cfg.max_object_keys
        ):
            return True

        if int(self.cfg.max_consecutive_token_repeats) > 0 and gen_len > 0:
            last = int(generated_token_ids[-1])
            run = 1
            limit = min(int(self.cfg.max_consecutive_token_repeats), int(gen_len))
            for j in range(2, limit + 1):
                if int(generated_token_ids[-j]) != last:
                    break
                run += 1
            if run >= int(self.cfg.max_consecutive_token_repeats):
                return True

        n = int(self.cfg.ngram_size)
        reps = int(self.cfg.ngram_repeats)
        if n > 0 and reps >= 2 and gen_len >= n * reps:
            tail = [int(x) for x in generated_token_ids[-n:]]
            ok = True
            for k in range(2, reps + 1):
                start = int(gen_len - k * n)
                end = int(gen_len - (k - 1) * n)
                prev = [int(x) for x in generated_token_ids[start:end]]
                if prev != tail:
                    ok = False
                    break
            if ok:
                return True

        return False

    def __call__(self, a0: Any, a1: Any, a2: Any | None = None):
        if a2 is None:
            generated_token_ids = a0
            scores = a1
        else:
            generated_token_ids = a1
            scores = a2

        if self.eos_token_id < 0:
            return scores

        if not isinstance(generated_token_ids, Sequence):
            return scores
        if not isinstance(scores, torch.Tensor):
            return scores

        generated = [int(x) for x in generated_token_ids]
        self._update_object_key_count(generated)

        if not self._should_force(generated):
            return scores

        self.triggered = True
        scores_out = scores.clone()
        scores_out[:] = -float("inf")
        scores_out[int(self.eos_token_id)] = 0.0
        return scores_out
