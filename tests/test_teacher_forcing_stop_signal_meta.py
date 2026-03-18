from __future__ import annotations

import re

import pytest

from src.trainers.rollout_matching.parsing import decode_pieces
from src.trainers.teacher_forcing.rollout_meta import semantic_stop_branch_metadata


class _MergedBoundaryTokenizer:
    def __init__(self) -> None:
        self._piece_to_id = {
            "<|im_end|>": 9000,
            "]},": 9001,
            "]}": 9002,
        }
        self._id_to_piece = {v: k for k, v in self._piece_to_id.items()}
        self._char_to_id: dict[str, int] = {}
        self._id_to_char: dict[int, str] = {}
        self._next_char_id = 100

    def _char_id(self, ch: str) -> int:
        if ch not in self._char_to_id:
            idx = int(self._next_char_id)
            self._next_char_id += 1
            self._char_to_id[ch] = idx
            self._id_to_char[idx] = ch
        return int(self._char_to_id[ch])

    def convert_tokens_to_ids(self, token):
        if isinstance(token, list):
            return [self.convert_tokens_to_ids(t) for t in token]
        tok = str(token)
        if tok in self._piece_to_id:
            return int(self._piece_to_id[tok])
        m = re.fullmatch(r"<\|coord_(\d+)\|>", tok)
        if m:
            return 1000 + int(m.group(1))
        return self._char_id(tok)

    def encode(self, text: str, add_special_tokens: bool = False):
        s = str(text)
        out: list[int] = []
        i = 0
        while i < len(s):
            if s.startswith("<|im_end|>", i):
                out.append(self._piece_to_id["<|im_end|>"])
                i += len("<|im_end|>")
                continue
            if s.startswith("]},", i):
                out.append(self._piece_to_id["]},"])
                i += len("]},")
                continue
            if s.startswith("]}", i):
                out.append(self._piece_to_id["]}"])
                i += len("]}")
                continue
            if s.startswith("<|coord_", i):
                j = s.find("|>", i)
                if j >= 0:
                    token = s[i : j + 2]
                    out.append(self.convert_tokens_to_ids(token))
                    i = j + 2
                    continue
            out.append(self._char_id(s[i]))
            i += 1
        return out

    def decode(
        self,
        token_ids,
        *,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ):
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        out: list[str] = []
        for tok_id in token_ids:
            tok_i = int(tok_id)
            if tok_i in self._id_to_piece:
                out.append(self._id_to_piece[tok_i])
            elif tok_i >= 1000:
                out.append(f"<|coord_{tok_i - 1000}|>")
            else:
                out.append(self._id_to_char[tok_i])
        return "".join(out)


class _CharOnlyTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False):
        return [ord(ch) for ch in str(text)]

    def decode(
        self,
        token_ids,
        *,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ):
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        return "".join(chr(int(tok)) for tok in token_ids)


def test_semantic_stop_branch_metadata_returns_terminal_stop_and_closure_tail() -> None:
    tok = _MergedBoundaryTokenizer()
    text = (
        '{"objects":[{"desc":"a","bbox_2d":[<|coord_0|>,<|coord_1|>,<|coord_2|>,<|coord_3|>]},'
        '{"desc":"b","bbox_2d":[<|coord_4|>,<|coord_5|>,<|coord_6|>,<|coord_7|>]}]}'
    )
    assistant_ids = list(tok.encode(text)) + [tok.convert_tokens_to_ids("<|im_end|>")]

    meta = semantic_stop_branch_metadata(
        tokenizer=tok,
        assistant_span_ids=assistant_ids,
        prefix_len=0,
    )

    pieces = decode_pieces(tok, assistant_ids)
    stop_idx = int(meta["stop_rel_pos"])

    assert pieces[stop_idx] == "]}"
    assert pieces[stop_idx + 1] == "]}"
    assert [pieces[idx] for idx in meta["tail_closure_pos"]] == ["]}", "<|im_end|>"]
    assert meta["stop_token_id"] == tok.convert_tokens_to_ids("]}")
    assert meta["continue_token_id"] == tok.convert_tokens_to_ids("]},")


def test_semantic_stop_branch_metadata_requires_exact_single_piece_tokens() -> None:
    tok = _CharOnlyTokenizer()
    text = '{"objects":[{"desc":"x","bbox_2d":[0,0,1,1]}]}'
    assistant_ids = list(tok.encode(text)) + list(tok.encode("<|im_end|>"))

    with pytest.raises(ValueError, match=r"exact single-token piece"):
        semantic_stop_branch_metadata(
            tokenizer=tok,
            assistant_span_ids=assistant_ids,
            prefix_len=0,
        )
