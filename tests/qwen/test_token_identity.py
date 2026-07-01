from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.common.errors import EncodingContractError
from src.config.loader import load_train_config
from src.qwen.loading import load_qwen_components
from src.qwen.tokens import (
    DEFAULT_COORDINATE_TOKENS,
    DEFAULT_WRAPPER_TOKENS,
    IM_END_SUFFIX,
    reject_invalid_qwen_aliases,
    validate_qwen_token_identity,
)


FIXTURE_CONFIG = Path("tests/fixtures/smoke/qwen3_vl_single_image_pack/config.yaml")


class FakeTokenizer:
    def __init__(
        self,
        *,
        token_ids: dict[str, int],
        encodings: dict[str, list[int]] | None = None,
    ) -> None:
        self.token_ids = token_ids
        self.encodings = dict(encodings or {})

    def convert_tokens_to_ids(self, token: str) -> int | None:
        return self.token_ids.get(token)

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        if add_special_tokens:
            raise AssertionError("Qwen preflight must encode without extra special tokens")
        if text in self.encodings:
            return self.encodings[text]
        if text in self.token_ids:
            return [self.token_ids[text]]
        return [27, 91, 999, 91, 29]

    def __len__(self) -> int:
        return max(self.token_ids.values(), default=0) + 1


def test_fake_tokenizer_validates_coordexp_token_identity() -> None:
    tokenizer = _fake_tokenizer()

    identity = validate_qwen_token_identity(tokenizer)

    assert identity.wrapper_token_ids == {
        "<|object_ref_start|>": 151646,
        "<|object_ref_end|>": 151647,
        "<|box_start|>": 151648,
        "<|box_end|>": 151649,
    }
    assert identity.coordinate_token_ids[0] == 151670
    assert identity.coordinate_token_ids[-1] == 152669
    assert len(identity.coordinate_token_ids) == 1000
    assert identity.im_end_token_ids == (151645,)
    assert identity.newline_token_ids == (198,)
    assert identity.im_end_newline_token_ids == (151645, 198)
    assert identity.to_artifact_dict()["coord_token_count"] == 1000


def test_missing_coordinate_token_fails_before_qwen_forward() -> None:
    token_ids = _token_ids()
    del token_ids["<|coord_999|>"]
    tokenizer = _fake_tokenizer(token_ids=token_ids)

    with pytest.raises(EncodingContractError, match="<\\|coord_999\\|>"):
        validate_qwen_token_identity(tokenizer)


def test_im_end_newline_merge_fails_before_qwen_forward() -> None:
    tokenizer = _fake_tokenizer(
        encodings={
            "<|im_end|>\n": [424242],
            "\n": [198],
        }
    )

    with pytest.raises(EncodingContractError, match="separate newline"):
        validate_qwen_token_identity(tokenizer)


def test_multi_token_newline_fails_before_qwen_forward() -> None:
    tokenizer = _fake_tokenizer(
        encodings={
            "\n": [198, 199],
            "<|im_end|>\n": [151645, 198, 199],
        }
    )

    with pytest.raises(EncodingContractError, match="one token"):
        validate_qwen_token_identity(tokenizer)


def test_invalid_wrapper_alias_is_rejected_when_referenced() -> None:
    with pytest.raises(EncodingContractError, match="<\\|object_start\\|>"):
        reject_invalid_qwen_aliases(
            ["<|object_ref_start|>", "<|object_start|>"],
            context_label="template.schema_tokens",
        )


def test_invalid_wrapper_alias_is_rejected_inside_text() -> None:
    with pytest.raises(EncodingContractError, match="<\\|object_end\\|>"):
        reject_invalid_qwen_aliases(
            "Describe this <|object_end|> alias.",
            context_label="template.prompt.user",
        )


def test_real_local_qwen_components_load_without_model_and_preflight_tokens() -> None:
    resolved = load_train_config(FIXTURE_CONFIG)

    components = load_qwen_components(resolved.config, load_model=False)

    assert components.model is None
    assert components.model_identity.model_type == "qwen3_vl"
    assert components.model_identity.architectures == ("Qwen3VLForConditionalGeneration",)
    assert components.model_identity.tie_word_embeddings is True
    assert components.model_identity.text_vocab_size == 152670
    assert components.processor_identity.processor_class == "Qwen3VLProcessor"
    assert components.processor_identity.tokenizer_class == "Qwen2TokenizerFast"
    assert components.processor_identity.image_processor_class == "Qwen2VLImageProcessorFast"
    assert components.processor_identity.patch_size == 16
    assert components.processor_identity.merge_size == 2
    assert components.processor_identity.temporal_patch_size == 2
    assert tuple(components.token_identity.required_tokens) == (
        *DEFAULT_WRAPPER_TOKENS,
        *DEFAULT_COORDINATE_TOKENS,
    )
    assert components.token_identity.im_end_newline_text == IM_END_SUFFIX
    assert components.token_identity.wrapper_token_ids["<|box_end|>"] == 151649
    assert components.token_identity.coordinate_token_ids[0] == 151670
    assert components.token_identity.coordinate_token_ids[-1] == 152669
    assert components.token_identity.im_end_newline_token_ids == (151645, 198)
    artifact = components.to_artifact_dict()
    assert artifact["load_model"] is False
    patch_receipt = artifact["runtime_patches"]["qwen3_vl_patch_embed_linearization"]
    assert patch_receipt["applied"] is False
    assert patch_receipt["reason"] == "model_not_loaded"


def _fake_tokenizer(
    *,
    token_ids: dict[str, int] | None = None,
    encodings: dict[str, list[int]] | None = None,
) -> FakeTokenizer:
    return FakeTokenizer(
        token_ids=dict(token_ids or _token_ids()),
        encodings={
            "<|im_end|>\n": [151645, 198],
            "\n": [198],
            **dict(encodings or {}),
        },
    )


def _token_ids() -> dict[str, int]:
    token_ids: dict[str, int] = {
        "<|im_end|>": 151645,
        "<|object_ref_start|>": 151646,
        "<|object_ref_end|>": 151647,
        "<|box_start|>": 151648,
        "<|box_end|>": 151649,
    }
    token_ids.update(
        {
            f"<|coord_{index}|>": 151670 + index
            for index in range(1000)
        }
    )
    return token_ids
