from __future__ import annotations

import torch

from src.infer.compact_grammar import build_compact_full_grammar_logits_processor


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


def test_compact_grammar_forces_row_delimiter_after_four_coords() -> None:
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
    assert processed[0, 3] == 0.0
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
