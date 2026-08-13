from __future__ import annotations

from types import SimpleNamespace

import torch

from scripts.research.human13_forced_continuation import (
    forced_complete_row_then_natural_continuation,
    source_continuation_cap,
)
from scripts.research.run_local_branch_causal_value import hash_prefix_token_ids


class _Model:
    def __init__(self, generated: tuple[int, ...] = (40, 41, 99)) -> None:
        self.calls: list[dict] = []
        self.generated = generated

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        prefix = kwargs["input_ids"]
        generated = torch.tensor([self.generated], dtype=prefix.dtype)
        return SimpleNamespace(sequences=torch.cat((prefix, generated), dim=1))


class _Tokenizer:
    def decode(self, token_ids, **_):
        return (
            "<|object_ref_start|>person<|object_ref_end|>"
            "<|box_start|><|coord_1|><|coord_2|><|coord_3|><|coord_4|>"
            "<|box_end|><|im_end|>"
        )


class _Session:
    def __init__(self, generated: tuple[int, ...] = (40, 41, 99)) -> None:
        self._model = _Model(generated)

    @staticmethod
    def _im_end_token_id():
        return 99

    @staticmethod
    def _pad_token_id():
        return 0


def test_forced_complete_row_releases_one_unconstrained_natural_suffix() -> None:
    session = _Session()
    result = forced_complete_row_then_natural_continuation(
        session=session,
        native_inputs={
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.ones((1, 2), dtype=torch.long),
        },
        natural_prefix_token_ids=(10, 11),
        forced_row_token_ids=(30, 31),
        tokenizer=_Tokenizer(),
        image_width=100,
        image_height=100,
        repetition_penalty=1.0,
        continuation_cap=520,
        source_row_count=2,
        source_token_count=6,
        current_checkpoint_payload_sha256="c" * 64,
    )

    assert session._model.calls[0]["input_ids"].tolist() == [[1, 2, 10, 11, 30, 31]]
    assert result.forced_row_token_ids == (30, 31)
    assert result.released_token_ids == (40, 41, 99)
    assert result.termination_status == "natural_im_end"
    assert result.cap_hit is False
    assert result.forced_row_parse_evidence["parse_status"] == "accepted"
    assert len(result.forced_row_parse_evidence["predictions"]) == 1
    assert result.parse_evidence["parse_status"] == "accepted"
    assert result.natural_prefix_token_ids_sha256 == hash_prefix_token_ids((10, 11))
    assert result.forced_row_token_ids_sha256 == hash_prefix_token_ids((30, 31))
    assert result.forced_context_sha256 == hash_prefix_token_ids((10, 11, 30, 31))
    assert result.requested_continuation_cap == 520
    assert result.minimum_continuation_cap == 518
    assert result.repetition_penalty == 1.0
    assert result.current_checkpoint_payload_sha256 == "c" * 64


def test_continuation_cap_hit_is_explicit_harm() -> None:
    session = _Session((40,) * 511 + (99,))
    result = forced_complete_row_then_natural_continuation(
        session=session,
        native_inputs={"input_ids": torch.tensor([[1, 2]])},
        natural_prefix_token_ids=(),
        forced_row_token_ids=(30,),
        tokenizer=_Tokenizer(),
        image_width=100,
        image_height=100,
        repetition_penalty=1.0,
        continuation_cap=512,
        source_row_count=0,
        source_token_count=0,
        current_checkpoint_payload_sha256="c" * 64,
    )

    assert result.cap_hit is True
    assert result.termination_status == "cap_hit"


def test_source_continuation_cap_is_derived_and_enforced_before_generation() -> None:
    assert source_continuation_cap(source_row_count=4, source_token_count=30) == 542
    session = _Session()

    try:
        forced_complete_row_then_natural_continuation(
            session=session,
            native_inputs={"input_ids": torch.tensor([[1, 2]])},
            natural_prefix_token_ids=(),
            forced_row_token_ids=(30,),
            tokenizer=_Tokenizer(),
            image_width=100,
            image_height=100,
            repetition_penalty=1.0,
            continuation_cap=541,
            source_row_count=4,
            source_token_count=30,
            current_checkpoint_payload_sha256="c" * 64,
        )
    except ValueError as exc:
        assert "Source-derived minimum" in str(exc)
    else:
        raise AssertionError("undersized continuation cap was accepted")
    assert session._model.calls == []
