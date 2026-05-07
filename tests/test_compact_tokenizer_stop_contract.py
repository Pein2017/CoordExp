from __future__ import annotations

from typing import Any

import pytest

from src.detection.tokenizer_contract import (
    CompactTrainingStopContract,
    resolve_compact_training_stop_contract,
)


class FakeQwenTokenizer:
    def __init__(
        self,
        *,
        token_to_id: dict[str, int],
        eos_token: str | None = "<|im_end|>",
        unk_token_id: int | None = None,
        encode_result_for_im_end: list[int] | None = None,
        chat_template_stop: str = "<|im_end|>",
        prepend_default_system: bool = False,
    ) -> None:
        self._token_to_id = dict(token_to_id)
        self.eos_token = eos_token
        self.unk_token_id = unk_token_id
        self.eos_token_id = (
            self._token_to_id[eos_token]
            if isinstance(eos_token, str) and eos_token in self._token_to_id
            else None
        )
        self._encode_result_for_im_end = encode_result_for_im_end
        self._chat_template_stop = chat_template_stop
        self._prepend_default_system = prepend_default_system

    def convert_tokens_to_ids(self, token: str) -> int:
        fallback = self.unk_token_id if self.unk_token_id is not None else -1
        return self._token_to_id.get(token, fallback)

    def get_vocab(self) -> dict[str, int]:
        return dict(self._token_to_id)

    def get_added_vocab(self) -> dict[str, int]:
        return {}

    @property
    def special_tokens_map(self) -> dict[str, str]:
        return {"im_end": "<|im_end|>"}

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        if text == "<|im_end|>" and self._encode_result_for_im_end is not None:
            return list(self._encode_result_for_im_end)
        return self._encode_text(text)

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str | list[int]:
        assert add_generation_prompt is False
        if self._prepend_default_system and not any(
            message.get("role") == "system" for message in messages
        ):
            messages = [{"role": "system", "content": "You are helpful."}, *messages]
        rendered = "".join(
            f"<|im_start|>{message['role']}\n"
            f"{message['content']}{self._chat_template_stop}\n"
            for message in messages
        )
        if not tokenize:
            return rendered
        return self._encode_text(rendered)

    def _encode_text(self, text: str) -> list[int]:
        ids: list[int] = []
        cursor = 0
        special_tokens = sorted(self._token_to_id, key=len, reverse=True)
        while cursor < len(text):
            for token in special_tokens:
                if text.startswith(token, cursor):
                    ids.append(self._token_to_id[token])
                    cursor += len(token)
                    break
            else:
                ids.append(self.convert_tokens_to_ids(text[cursor]))
                cursor += 1
        return ids


def test_compact_training_stop_token_is_im_end_only() -> None:
    tokenizer = FakeQwenTokenizer(
        token_to_id={
            "<|im_start|>": 6,
            "<|im_end|>": 7,
            "<|endoftext|>": 8,
            "<|end_of_text|>": 9,
        },
        eos_token="<|im_end|>",
    )

    contract = resolve_compact_training_stop_contract(tokenizer)

    assert contract == CompactTrainingStopContract(
        im_end_token_id=7,
        training_eos_token_text="<|im_end|>",
        training_eos_token_ids=frozenset({7}),
        tokenizer_eos_token_text="<|im_end|>",
        tokenizer_eos_token_id=7,
    )


def test_compact_training_ignores_text_level_tokenizer_eos() -> None:
    tokenizer = FakeQwenTokenizer(
        token_to_id={
            "<|im_start|>": 6,
            "<|im_end|>": 7,
            "<|endoftext|>": 8,
        },
        eos_token="<|endoftext|>",
    )

    contract = resolve_compact_training_stop_contract(tokenizer)

    assert contract.im_end_token_id == 7
    assert contract.training_eos_token_ids == frozenset({7})
    assert contract.tokenizer_eos_token_text == "<|endoftext|>"
    assert contract.tokenizer_eos_token_id == 8


def test_missing_im_end_that_resolves_to_unk_is_rejected() -> None:
    tokenizer = FakeQwenTokenizer(
        token_to_id={"<|im_start|>": 6, "<|endoftext|>": 8},
        unk_token_id=0,
        eos_token="<|endoftext|>",
    )

    with pytest.raises(ValueError, match="im_end.*unk"):
        resolve_compact_training_stop_contract(tokenizer)


def test_im_end_must_encode_as_single_special_token() -> None:
    tokenizer = FakeQwenTokenizer(
        token_to_id={"<|im_start|>": 6, "<|im_end|>": 7},
        encode_result_for_im_end=[101, 102],
    )

    with pytest.raises(ValueError, match="single token"):
        resolve_compact_training_stop_contract(tokenizer)


def test_closed_qwen_chat_template_supplies_exactly_one_im_end_id() -> None:
    tokenizer = FakeQwenTokenizer(
        token_to_id={"<|im_start|>": 6, "<|im_end|>": 7},
    )

    contract = resolve_compact_training_stop_contract(tokenizer)
    encoded = tokenizer.apply_chat_template(
        [{"role": "assistant", "content": "payload"}],
        tokenize=True,
        add_generation_prompt=False,
    )

    assert isinstance(encoded, list)
    assert encoded.count(contract.im_end_token_id) == 1
    assert encoded[-2] == contract.im_end_token_id


def test_default_system_turn_does_not_break_assistant_im_end_contract() -> None:
    tokenizer = FakeQwenTokenizer(
        token_to_id={"<|im_start|>": 6, "<|im_end|>": 7},
        prepend_default_system=True,
    )

    contract = resolve_compact_training_stop_contract(tokenizer)
    text = tokenizer.apply_chat_template(
        [{"role": "assistant", "content": "payload"}],
        tokenize=False,
        add_generation_prompt=False,
    )
    encoded = tokenizer.apply_chat_template(
        [{"role": "assistant", "content": "payload"}],
        tokenize=True,
        add_generation_prompt=False,
    )

    assert isinstance(text, str)
    assert isinstance(encoded, list)
    assert text.count("<|im_end|>") == 2
    assert encoded.count(contract.im_end_token_id) == 2
    assert text.rfind("payload<|im_end|>") >= 0


def test_chat_template_without_im_end_after_payload_is_rejected() -> None:
    tokenizer = FakeQwenTokenizer(
        token_to_id={"<|im_start|>": 6, "<|im_end|>": 7, "<|endoftext|>": 8},
        chat_template_stop="<|endoftext|>",
    )

    with pytest.raises(ValueError, match="chat template.*<\\|im_end\\|>"):
        resolve_compact_training_stop_contract(tokenizer)
