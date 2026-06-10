from __future__ import annotations

import torch
import pytest

from src.infer.constraints import (
    build_compact_full_grammar_logits_processor,
    build_compact_grammar_logits_processor,
)


class _DummyTokenizer:
    eos_token_id = 2

    def __init__(self) -> None:
        self._vocab = {
            "<|im_end|>": 2,
            "\n": 3,
            "<|object_ref_start|>": 4,
            "<|box_start|>": 5,
            "person": 6,
            "bad": 7,
        }
        for idx in range(1000):
            self._vocab[f"<|coord_{idx}|>"] = 100 + idx

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return [self._vocab[text]]


def _scores(vocab_size: int = 1200) -> torch.Tensor:
    values = torch.zeros((1, vocab_size), dtype=torch.float32)
    values[0, 7] = 100.0
    return values


def test_compact_grammar_forces_four_coord_tokens_after_box_start() -> None:
    processor = build_compact_full_grammar_logits_processor(
        tokenizer=_DummyTokenizer(),
        prompt_lengths=[2],
    )
    input_ids = torch.tensor([[80, 81, 4, 6, 5]], dtype=torch.long)

    processed = processor(input_ids, _scores())

    assert torch.isneginf(processed[0, 7])
    assert processed[0, 100] == 0.0
    assert processed[0, 1099] == 0.0


def test_compact_grammar_forces_next_object_or_stop_after_four_coords() -> None:
    processor = build_compact_full_grammar_logits_processor(
        tokenizer=_DummyTokenizer(),
        prompt_lengths=[1],
    )
    input_ids = torch.tensor(
        [[80, 4, 6, 5, 100, 101, 102, 103]],
        dtype=torch.long,
    )

    processed = processor(input_ids, _scores())

    assert torch.isneginf(processed[0, 7])
    assert torch.isneginf(processed[0, 3])
    assert processed[0, 4] == 0.0
    assert processed[0, 2] == 0.0


def test_compact_grammar_forces_object_start_or_stop_after_newline() -> None:
    processor = build_compact_full_grammar_logits_processor(
        tokenizer=_DummyTokenizer(),
        prompt_lengths=[1],
    )
    input_ids = torch.tensor(
        [[80, 4, 6, 5, 100, 101, 102, 103, 3]],
        dtype=torch.long,
    )

    processed = processor(input_ids, _scores())

    assert torch.isneginf(processed[0, 7])
    assert processed[0, 4] == 0.0
    assert processed[0, 2] == 0.0


def test_compact_grammar_im_end_eos_excludes_text_eos() -> None:
    tokenizer = _DummyTokenizer()
    tokenizer.eos_token_id = 99
    processor = build_compact_full_grammar_logits_processor(
        tokenizer=tokenizer,
        prompt_lengths=[1],
    )
    input_ids = torch.tensor(
        [[80, 4, 6, 5, 100, 101, 102, 103, 3]],
        dtype=torch.long,
    )

    processed = processor(input_ids, _scores())

    assert processed[0, 2] == 0.0
    assert torch.isneginf(processed[0, 99])


def test_compact_grammar_requires_im_end_even_when_tokenizer_eos_exists() -> None:
    tokenizer = _DummyTokenizer()
    tokenizer.eos_token_id = 99
    del tokenizer._vocab["<|im_end|>"]

    with pytest.raises(ValueError, match=r"<\|im_end\|>"):
        build_compact_full_grammar_logits_processor(
            tokenizer=tokenizer,
            prompt_lengths=[1],
        )


def test_compact_grammar_wrapper_accepts_compact_full_only() -> None:
    processor = build_compact_grammar_logits_processor(
        tokenizer=_DummyTokenizer(),
        prompt_lengths=[1],
        detection_sequence_format="compact_full",
    )

    input_ids = torch.tensor([[80]], dtype=torch.long)
    processed = processor(input_ids, _scores())

    assert torch.isneginf(processed[0, 7])
    assert processed[0, 4] == 0.0
    assert processed[0, 2] == 0.0


def test_compact_grammar_prompt_length_is_absolute_offset_under_left_padding() -> None:
    input_ids = torch.tensor([[0, 0, 80, 81]], dtype=torch.long)

    wrong_unpadded_length_processor = build_compact_full_grammar_logits_processor(
        tokenizer=_DummyTokenizer(),
        prompt_lengths=[2],
    )
    wrong_processed = wrong_unpadded_length_processor(input_ids, _scores())

    assert wrong_processed[0, 7] == 100.0

    correct_padded_offset_processor = build_compact_full_grammar_logits_processor(
        tokenizer=_DummyTokenizer(),
        prompt_lengths=[4],
    )
    processed = correct_padded_offset_processor(input_ids, _scores())

    assert torch.isneginf(processed[0, 7])
    assert processed[0, 4] == 0.0
    assert processed[0, 2] == 0.0


def test_compact_grammar_wrapper_rejects_non_compact_format() -> None:
    with pytest.raises(ValueError, match="detection_sequence_format=compact_full"):
        build_compact_grammar_logits_processor(
            tokenizer=_DummyTokenizer(),
            prompt_lengths=[1],
            detection_sequence_format="stage1_json_pretty",
        )


def test_compact_grammar_requires_all_coord_tokens() -> None:
    tokenizer = _DummyTokenizer()
    del tokenizer._vocab["<|coord_999|>"]

    with pytest.raises(ValueError, match="requires all 1000 coord tokens"):
        build_compact_full_grammar_logits_processor(
            tokenizer=tokenizer,
            prompt_lengths=[1],
        )
