from __future__ import annotations

import torch

from src.common.repeat_terminate import (
    ForceEosOnRepeatBatchGuard,
    ForceEosOnRepeatSequenceGuard,
    RepeatTerminateConfig,
    should_trigger_repeat_terminate,
)


def _cfg(**overrides) -> RepeatTerminateConfig:
    base = {
        "enabled": True,
        "min_new_tokens": 0,
        "max_consecutive_token_repeats": 4,
        "ngram_size": 3,
        "ngram_repeats": 2,
        "max_object_keys": None,
    }
    base.update(overrides)
    return RepeatTerminateConfig(**base)


def test_should_trigger_repeat_terminate_on_consecutive_repeats() -> None:
    cfg = _cfg(max_consecutive_token_repeats=4, ngram_size=0, ngram_repeats=0)
    generated = [1, 2, 9, 9, 9, 9]
    assert (
        should_trigger_repeat_terminate(
            generated_token_ids=generated,
            cfg=cfg,
            object_key_prefix_token_ids=None,
        )
        is True
    )


def test_should_trigger_repeat_terminate_on_repeated_ngrams() -> None:
    cfg = _cfg(max_consecutive_token_repeats=0, ngram_size=2, ngram_repeats=3)
    generated = [5, 7, 8, 7, 8, 7, 8]
    assert (
        should_trigger_repeat_terminate(
            generated_token_ids=generated,
            cfg=cfg,
            object_key_prefix_token_ids=None,
        )
        is True
    )


def test_should_trigger_repeat_terminate_on_object_key_cap() -> None:
    cfg = _cfg(
        max_consecutive_token_repeats=0,
        ngram_size=0,
        ngram_repeats=0,
        max_object_keys=2,
    )
    generated = [11, 12, 3, 4, 11, 12, 6, 7]
    assert (
        should_trigger_repeat_terminate(
            generated_token_ids=generated,
            cfg=cfg,
            object_key_prefix_token_ids=[11, 12],
        )
        is True
    )


def test_force_eos_batch_guard_only_affects_offending_sequences() -> None:
    guard = ForceEosOnRepeatBatchGuard(
        eos_token_id=0,
        prompt_len=2,
        cfg=_cfg(max_consecutive_token_repeats=3, ngram_size=0, ngram_repeats=0),
        object_key_prefix_token_ids=None,
    )

    input_ids = torch.tensor(
        [
            [101, 102, 9, 9, 9],
            [101, 102, 3, 4, 5],
        ],
        dtype=torch.long,
    )
    scores = torch.zeros((2, 16), dtype=torch.float32)

    out = guard(input_ids, scores)
    assert torch.isneginf(out[0]).all().item() is False
    assert float(out[0, 0].item()) == 0.0
    assert torch.equal(out[1], scores[1])


def test_force_eos_sequence_guard_sets_trigger_flag() -> None:
    guard = ForceEosOnRepeatSequenceGuard(
        eos_token_id=0,
        cfg=_cfg(max_consecutive_token_repeats=3, ngram_size=0, ngram_repeats=0),
        object_key_prefix_token_ids=None,
    )

    prompt_ids = [101, 102]
    generated_ids = [9, 9, 9]
    scores = torch.zeros((16,), dtype=torch.float32)

    out = guard(prompt_ids, generated_ids, scores)
    assert bool(guard.triggered) is True
    assert float(out[0].item()) == 0.0
    assert torch.isneginf(out[1:]).all().item() is True
