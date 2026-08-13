from __future__ import annotations

from types import SimpleNamespace

import torch

from scripts.research.human13_forced_continuation import (
    forced_complete_row_then_natural_continuation,
)


class _Model:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        prefix = kwargs["input_ids"]
        generated = torch.tensor([[40, 41, 99]], dtype=prefix.dtype)
        return SimpleNamespace(sequences=torch.cat((prefix, generated), dim=1))


class _Tokenizer:
    def decode(self, token_ids, **_):
        assert token_ids == [40, 41, 99]
        return "person<|object_ref_end|><|box_start|>(1,2),(3,4)<|box_end|><|im_end|>"


class _Session:
    def __init__(self) -> None:
        self._model = _Model()

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
        continuation_cap=8,
    )

    assert session._model.calls[0]["input_ids"].tolist() == [[1, 2, 10, 11, 30, 31]]
    assert result.forced_row_token_ids == (30, 31)
    assert result.released_token_ids == (40, 41, 99)
    assert result.termination_status == "natural_im_end"
    assert result.cap_hit is False


def test_continuation_cap_hit_is_explicit_harm() -> None:
    session = _Session()
    result = forced_complete_row_then_natural_continuation(
        session=session,
        native_inputs={"input_ids": torch.tensor([[1, 2]])},
        natural_prefix_token_ids=(),
        forced_row_token_ids=(30,),
        tokenizer=_Tokenizer(),
        image_width=100,
        image_height=100,
        repetition_penalty=1.0,
        continuation_cap=3,
    )

    assert result.cap_hit is True
    assert result.termination_status == "cap_hit"
