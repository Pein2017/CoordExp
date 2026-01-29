import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.coord_tokens.codec import (  # noqa: E402
    build_coord_token_id_mask,
    int_to_token,
    is_coord_token,
    normalized_from_ints,
    token_to_int,
    tokens_to_ints,
)


class _DummyTokenizer:
    def __init__(self) -> None:
        limit = 1000
        self.vocab = {int_to_token(i): i for i in range(limit)}
        self.vocab_size = limit + 5  # pad a little

    def convert_tokens_to_ids(self, tokens):
        return [self.vocab.get(tok, -1) for tok in tokens]


def test_token_round_trip_default_range():
    token = int_to_token(123)
    assert token == "<|coord_123|>"
    assert token_to_int(token) == 123
    assert normalized_from_ints([123]) == pytest.approx([123.0 / 999.0])


def test_token_1000_rejected():
    with pytest.raises(ValueError):
        int_to_token(1000)
    with pytest.raises(ValueError):
        token_to_int("<|coord_1000|>")


def test_tokens_to_ints_accepts_tokens_and_numbers():
    ints = tokens_to_ints(["<|coord_1|>", 2, "3"], require_even=False)
    assert ints == [1, 2, 3]


def test_coord_mask_matches_vocab_size():
    torch = pytest.importorskip("torch")
    tok = _DummyTokenizer()
    mask = build_coord_token_id_mask(tok)
    assert mask.shape[0] == tok.vocab_size
    assert mask.sum().item() == 1000
    assert mask[0] and mask[999]
    assert not mask[-1]


def test_is_coord_token_recognizes_format():
    assert is_coord_token("<|coord_42|>")
    assert not is_coord_token("coord_42")
