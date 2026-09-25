"""Pure character/literal-token alignment contract; no tokenizer downloads."""
import subprocess
import sys

import pytest

from src.inference.token_text import (
    character_span_to_token_interval,
    decode_literal_ids,
    exact_token_text_frame,
)


class LiteralTokenizer:
    pieces = {1: "<obj>", 2: "物体", 3: "</obj>", 4: ""}

    def decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces=False):
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        return "".join(self.pieces[i] for i in ids)

    def encode(self, *_args, **_kwargs):
        raise AssertionError("literal history must never be re-encoded")


def test_exact_unicode_frame_and_original_token_interval():
    text, spans = exact_token_text_frame([1, 2, 3], LiteralTokenizer())
    assert text == "<obj>物体</obj>"
    assert spans == [(0, 5), (5, 7), (7, 13)]
    assert character_span_to_token_interval(5, 7, text=text, token_spans=spans) == (1, 2)
    assert character_span_to_token_interval(0, 13, text=text, token_spans=spans) == (0, 3)


def test_minimal_tokenizer_signature_keeps_the_existing_decode_fallback():
    class Minimal:
        def decode(self, ids, *, skip_special_tokens):
            assert not skip_special_tokens
            return "".join(str(i) for i in ids)

    assert exact_token_text_frame([1, 2], Minimal()) == ("12", [(0, 1), (1, 2)])


def test_nonadditive_decoder_fails_without_retokenizing():
    class Nonadditive(LiteralTokenizer):
        def decode(self, ids, **kwargs):
            return super().decode(ids, **kwargs) + ("!" if len(ids) > 1 else "")

    with pytest.raises(ValueError, match="does not reconstruct full decode"):
        exact_token_text_frame([1, 2], Nonadditive())


@pytest.mark.parametrize("start,end", [(True, 7), (0, False), (-1, 5), (0, 0), (7, 5), (0, 99)])
def test_invalid_character_span_is_rejected(start, end):
    text, spans = exact_token_text_frame([1, 2, 3], LiteralTokenizer())
    with pytest.raises(ValueError, match="parser character span is invalid"):
        character_span_to_token_interval(start, end, text=text, token_spans=spans)


def test_nonunique_and_midtoken_boundaries_are_not_guessed():
    with pytest.raises(ValueError, match="unique original token boundaries"):
        character_span_to_token_interval(0, 1, text="a", token_spans=[(0, 1), (0, 1)])
    with pytest.raises(ValueError, match="unique original token boundaries"):
        character_span_to_token_interval(1, 5, text="<obj>", token_spans=[(0, 5)])


def test_empty_decode_piece_does_not_become_a_false_boundary():
    text, spans = exact_token_text_frame([4, 2, 4], LiteralTokenizer())
    assert spans == [(0, 0), (0, 2), (2, 2)]
    assert character_span_to_token_interval(0, 2, text=text, token_spans=spans) == (1, 2)


def test_missing_decode_method_fails_explicitly():
    with pytest.raises(TypeError, match="must expose decode"):
        decode_literal_ids(object(), [1])


def test_import_does_not_load_model_libraries_or_direction_packages():
    command = (
        "import sys; import src.inference.token_text; "
        "assert 'torch' not in sys.modules; "
        "assert 'transformers' not in sys.modules; "
        "assert not any(k == 'probes' or k.startswith('probes.') for k in sys.modules)"
    )
    subprocess.run([sys.executable, "-B", "-c", command], check=True, timeout=15)
