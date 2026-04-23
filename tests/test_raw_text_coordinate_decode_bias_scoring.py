from __future__ import annotations

import pytest
import torch
from transformers import RepetitionPenaltyLogitsProcessor

from src.analysis.raw_text_coordinate_decode_bias_scoring import (
    group_raw_text_token_rows,
    inspect_processed_position,
    score_processed_span_token_rows,
)


def test_group_raw_text_token_rows_partitions_desc_digit_and_structure() -> None:
    class _Tokenizer:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            return [ord(ch) for ch in text]

    candidate_assistant_text = '{"desc":"cup","bbox_2d":[12,34,56,78]}'
    token_rows = [
        {
            "assistant_relative_position": position,
            "token_text": char,
        }
        for position, char in enumerate(candidate_assistant_text)
    ]

    grouped = group_raw_text_token_rows(
        tokenizer=_Tokenizer(),
        candidate_assistant_text=candidate_assistant_text,
        token_rows=token_rows,
    )

    assert sorted(grouped) == ["desc", "digit", "structure"]
    assert "".join(row["token_text"] for row in grouped["desc"]) == "cup"
    assert "".join(row["token_text"] for row in grouped["digit"]) == "12345678"
    assert grouped["structure"][0]["token_text"] == "{"
    assert sum(len(rows) for rows in grouped.values()) == len(token_rows)


def test_group_raw_text_token_rows_requires_rebase_for_absolute_positions() -> None:
    class _Tokenizer:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            return [ord(ch) for ch in text]

    candidate_assistant_text = '{"desc":"cup","bbox_2d":[12,34,56,78]}'
    token_rows = [
        {
            "position": 5 + position,
            "token_text": char,
        }
        for position, char in enumerate(candidate_assistant_text)
    ]

    with pytest.raises(KeyError, match="position_base"):
        group_raw_text_token_rows(
            tokenizer=_Tokenizer(),
            candidate_assistant_text=candidate_assistant_text,
            token_rows=token_rows,
        )

    grouped = group_raw_text_token_rows(
        tokenizer=_Tokenizer(),
        candidate_assistant_text=candidate_assistant_text,
        token_rows=token_rows,
        position_base=5,
    )

    assert "".join(row["token_text"] for row in grouped["desc"]) == "cup"
    assert "".join(row["token_text"] for row in grouped["digit"]) == "12345678"


def test_score_processed_span_token_rows_uses_full_history_and_records_rows() -> None:
    class _Tokenizer:
        def decode(
            self,
            token_ids: list[int],
            *,
            skip_special_tokens: bool = False,
            clean_up_tokenization_spaces: bool = False,
        ) -> str:
            del skip_special_tokens, clean_up_tokenization_spaces
            return "|".join(str(int(token_id)) for token_id in token_ids)

    input_ids = torch.tensor([[5, 7, 5, 9]], dtype=torch.long)
    logits = torch.full((1, 4, 16), -20.0)
    logits[0, 1, 5] = 6.0
    logits[0, 1, 8] = 5.0
    logits[0, 2, 9] = 6.0
    logits[0, 2, 8] = 5.0

    rows = score_processed_span_token_rows(
        logits=logits,
        input_ids=input_ids,
        batch_idx=0,
        positions=[2, 3],
        tokenizer=_Tokenizer(),
        logits_processors=[RepetitionPenaltyLogitsProcessor(2.0)],
    )

    assert [row["position"] for row in rows] == [2, 3]
    assert [row["token_id"] for row in rows] == [5, 9]
    assert [row["token_text"] for row in rows] == ["5", "9"]
    assert rows[0]["history_ids"] == [5, 7]
    assert rows[1]["history_ids"] == [5, 7, 5]
    assert "raw_logprob" in rows[0]
    assert "processed_logprob" in rows[0]
    assert float(rows[0]["processed_logprob"]) < float(rows[0]["raw_logprob"])
    assert float(rows[1]["processed_logprob"]) == float(rows[1]["raw_logprob"])


def test_inspect_processed_position_reports_focus_tokens_top_tokens_and_group_summaries() -> None:
    class _Tokenizer:
        _DECODE = {
            0: "hist",
            1: "]",
            2: ", {",
            3: ', "poly"',
            4: "}",
            5: "x",
            6: ",",
            7: "<|im_end|>",
        }

        def decode(
            self,
            token_ids: list[int],
            *,
            skip_special_tokens: bool = False,
            clean_up_tokenization_spaces: bool = False,
        ) -> str:
            del skip_special_tokens, clean_up_tokenization_spaces
            return "".join(self._DECODE[int(token_id)] for token_id in token_ids)

    input_ids = torch.tensor([[0, 5, 1]], dtype=torch.long)
    logits = torch.full((1, 3, 8), -20.0)
    logits[0, 1, 1] = 7.0
    logits[0, 1, 2] = 6.0
    logits[0, 1, 3] = 5.0
    logits[0, 1, 4] = 4.0

    row = inspect_processed_position(
        logits=logits,
        input_ids=input_ids,
        batch_idx=0,
        position=2,
        tokenizer=_Tokenizer(),
        top_k=3,
        focus_token_ids={
            "close_now": 1,
            "next_object": 2,
        },
        group_token_ids={
            "close_now": [1, 4],
            "next_object": [2, 6],
            "wrong_schema": [3],
        },
    )

    assert row["position"] == 2
    assert row["token_id"] == 1
    assert row["token_text"] == "]"
    assert row["history_ids"] == [0, 5]
    assert [token["token_id"] for token in row["top_tokens"]] == [1, 2, 3]
    assert row["focus_tokens"]["close_now"]["token_text"] == "]"
    assert row["focus_tokens"]["next_object"]["token_text"] == ", {"
    assert (
        row["group_summaries"]["close_now"]["raw_top_token_text"] == "]"
    )
    assert (
        row["group_summaries"]["wrong_schema"]["raw_top_token_text"] == ', "poly"'
    )
    assert (
        float(row["group_summaries"]["close_now"]["raw_prob_mass"])
        > float(row["group_summaries"]["next_object"]["raw_prob_mass"])
        > float(row["group_summaries"]["wrong_schema"]["raw_prob_mass"])
    )
