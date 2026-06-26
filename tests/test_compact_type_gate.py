from __future__ import annotations

import re
from types import SimpleNamespace

import pytest
import torch

from src.common.detection_sequence import (
    BOX_END_TOKEN,
    BOX_START_TOKEN,
    OBJECT_REF_END_TOKEN,
    OBJECT_REF_START_TOKEN,
)
from src.detection.token_types import (
    allowed_type_token_ids_for_target,
    build_compact_token_type_groups,
    combine_main_and_type_losses,
)

_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|]+\|>")


class TypeGateTokenizer:
    unk_token_id = 0
    pad_token_id = 99
    eos_token = "<|endoftext|>"

    def __init__(self) -> None:
        self._token_to_id: dict[str, int] = {
            "<|im_start|>": 1,
            "<|im_end|>": 2,
            "<|endoftext|>": 3,
            "<|end_of_text|>": 4,
            OBJECT_REF_START_TOKEN: 5,
            BOX_START_TOKEN: 6,
            "<|vision_start|>": 7,
            OBJECT_REF_END_TOKEN: 8,
            BOX_END_TOKEN: 9,
            "<|coord_0|>": 10,
            "<|coord_999|>": 1009,
            "cat": 2000,
            "dog": 2001,
            "\n": 2002,
            " ": 2003,
            "<pad>": self.pad_token_id,
        }
        self.eos_token_id = self._token_to_id[self.eos_token]

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._token_to_id.get(token, self.unk_token_id)

    def get_vocab(self) -> dict[str, int]:
        return dict(self._token_to_id)

    def get_added_vocab(self) -> dict[str, int]:
        return {}

    @property
    def special_tokens_map(self) -> dict[str, str]:
        return {
            "im_start": "<|im_start|>",
            "im_end": "<|im_end|>",
            "vision_start": "<|vision_start|>",
        }

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        match = _SPECIAL_TOKEN_RE.fullmatch(text)
        if match is not None:
            return [self.convert_tokens_to_ids(text)]
        return [self._token_to_id.setdefault(text, len(self._token_to_id) + 1)]

    def apply_chat_template(self, messages, *, tokenize: bool, add_generation_prompt: bool):
        assert add_generation_prompt is False
        rendered = "".join(
            f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
            for message in messages
        )
        if tokenize:
            return [self.convert_tokens_to_ids("<|im_end|>")]
        return rendered


def test_compact_token_groups_classify_struct_coord_desc_and_eos() -> None:
    tokenizer = TypeGateTokenizer()
    groups = build_compact_token_type_groups(tokenizer)

    assert tokenizer.convert_tokens_to_ids(OBJECT_REF_START_TOKEN) in groups.struct
    assert tokenizer.convert_tokens_to_ids(OBJECT_REF_END_TOKEN) in groups.struct
    assert tokenizer.convert_tokens_to_ids(BOX_START_TOKEN) in groups.struct
    assert tokenizer.convert_tokens_to_ids(BOX_END_TOKEN) in groups.struct
    assert tokenizer.convert_tokens_to_ids("<|coord_0|>") in groups.coord
    assert tokenizer.convert_tokens_to_ids("<|coord_999|>") in groups.coord
    assert tokenizer.convert_tokens_to_ids("<|im_end|>") in groups.eos
    assert tokenizer.convert_tokens_to_ids("<|endoftext|>") not in groups.eos
    assert tokenizer.convert_tokens_to_ids("<|end_of_text|>") not in groups.eos
    assert tokenizer.convert_tokens_to_ids("<|endoftext|>") not in groups.desc
    assert tokenizer.convert_tokens_to_ids("<|end_of_text|>") not in groups.desc
    assert tokenizer.pad_token_id not in groups.desc
    assert tokenizer.convert_tokens_to_ids("<|im_start|>") not in groups.desc
    assert tokenizer.convert_tokens_to_ids("<|vision_start|>") not in groups.desc
    assert tokenizer.convert_tokens_to_ids("cat") in groups.desc
    assert tokenizer.convert_tokens_to_ids("\n") in groups.struct


def test_type_gate_uses_union_of_positive_child_types() -> None:
    tokenizer = TypeGateTokenizer()
    groups = build_compact_token_type_groups(tokenizer)
    desc_id = tokenizer.convert_tokens_to_ids("cat")
    struct_id = tokenizer.convert_tokens_to_ids(BOX_START_TOKEN)
    target = SimpleNamespace(positive_token_ids=(desc_id, struct_id))

    allowed = allowed_type_token_ids_for_target(target, groups)

    assert desc_id in allowed
    assert struct_id in allowed
    assert tokenizer.convert_tokens_to_ids("dog") in allowed


def test_type_gate_weight_is_added_to_position_loss() -> None:
    got = combine_main_and_type_losses(
        main_loss=torch.tensor(1.25),
        type_loss=torch.tensor(0.5),
        type_weight=0.2,
    )

    assert got.item() == pytest.approx(1.35)


def test_positive_tokens_are_subset_of_expanded_allowed_types() -> None:
    tokenizer = TypeGateTokenizer()
    groups = build_compact_token_type_groups(tokenizer)
    desc_id = tokenizer.convert_tokens_to_ids("cat")
    struct_id = tokenizer.convert_tokens_to_ids(BOX_START_TOKEN)
    target = SimpleNamespace(positive_token_ids=(desc_id, struct_id))

    allowed = allowed_type_token_ids_for_target(target, groups)

    assert set(target.positive_token_ids).issubset(allowed)


def test_type_gate_rejects_unclassified_positive_token_even_with_valid_sibling() -> None:
    tokenizer = TypeGateTokenizer()
    groups = build_compact_token_type_groups(tokenizer)
    desc_id = tokenizer.convert_tokens_to_ids("cat")
    target = SimpleNamespace(positive_token_ids=(desc_id, tokenizer.pad_token_id))

    with pytest.raises(ValueError, match="positive token id .*compact token type"):
        allowed_type_token_ids_for_target(target, groups)
