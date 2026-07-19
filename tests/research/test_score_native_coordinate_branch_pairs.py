"""Unit tests for native sampled coordinate branch validation and scoring."""

from __future__ import annotations

import copy
import math
import re

import pytest

from scripts.research.score_native_coordinate_branch_pairs import (
    coordinate_token_ids,
    score_native_coordinate_states,
    validate_native_case,
)


class FakeTokenizer:
    """Small deterministic tokenizer for the structural tests."""

    def __init__(self) -> None:
        self.special = {
            "<|object_ref_start|>": 100,
            "<|object_ref_end|>": 101,
            "<|box_start|>": 102,
            "<|box_end|>": 107,
        }
        self.special.update({f"<|coord_{index}|>": 110 + index for index in range(1000)})
        self.words = {"person": 200, "chair": 201}

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.special.get(token, self.words.get(token, -1))

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        pieces = re.findall(r"<\|[^|]+\|>|[^<]+", text)
        result: list[int] = []
        for piece in pieces:
            if piece.startswith("<|"):
                result.append(self.special[piece])
            elif piece.strip():
                result.append(self.words[piece.strip()])
        return result


def _span(category: str, bins: list[int]) -> str:
    return (
        "<|object_ref_start|>"
        + category
        + "<|object_ref_end|><|box_start|>"
        + "".join(f"<|coord_{value}|>" for value in bins)
        + "<|box_end|>"
    )


def _document(clean_bins: list[int] | None = None, degraded_bins: list[int] | None = None) -> dict:
    clean_bins = clean_bins or [10, 20, 30, 40]
    degraded_bins = degraded_bins or [10, 20, 31, 41]
    clean_span = _span("person", clean_bins)
    degraded_span = _span("person", degraded_bins)
    tokenizer = FakeTokenizer()
    clean_ids = tokenizer.encode(clean_span)
    degraded_ids = tokenizer.encode(degraded_span)
    return {
        "schema_version": "current_seeded_sampled_rollouts.v1",
        "model_identity": {
            "model_identity": {
                "base": {"path": "/models/base"},
                "adapter": {"adapter_path": "/models/adapter", "adapter_type": "dora", "adapter_name": "default"},
                "embedding_delta": {"identity": {"delta_path": "/models/delta"}},
            }
        },
        "rollouts": [
            {
                "image_id": 7,
                "seed": 11,
                "prompt_token_ids": [1, 2, 3],
                "generated_token_ids": clean_ids,
                "predictions": {"predictions": [{"description": "person", "generated_order": 0, "coord_bins": clean_bins, "raw_span_text": clean_span}]},
            },
            {
                "image_id": 7,
                "seed": 12,
                "prompt_token_ids": [1, 2, 3],
                "generated_token_ids": degraded_ids,
                "predictions": {"predictions": [{"description": "person", "generated_order": 0, "coord_bins": degraded_bins, "raw_span_text": degraded_span}]},
            },
        ],
    }


def _case(clean_seed: int = 11, degraded_seed: int = 12) -> dict:
    return {
        "name": "native-person-7",
        "role": "primary",
        "image_id": 7,
        "category": "person",
        "reference_box_xyxy": [10, 20, 30, 40],
        "clean": {"seed": clean_seed, "generated_order": 0},
        "degraded": {"seed": degraded_seed, "generated_order": 0},
    }


def _expected_composition() -> dict[str, str]:
    return {
        "base_model_path": "/models/base",
        "adapter_path": "/models/adapter",
        "adapter_type": "dora",
        "adapter_name": "default",
        "embedding_delta_path": "/models/delta",
    }


def test_native_validation_accepts_exact_shared_prefix_and_records_first_slot() -> None:
    result = validate_native_case(_case(), _document(), tokenizer=FakeTokenizer(), expected_model_composition=_expected_composition())
    assert result["first_differing_coordinate_slot"] == "x2"
    assert result["clean"]["coordinate_generated_steps"] == [4, 5, 6, 7]
    assert result["clean"]["coord_token_ids"][-1] == 150
    assert result["counterfactual_model_composition"]["enabled"] is False


def test_native_validation_rejects_prefix_mismatch_before_first_difference() -> None:
    document = _document()
    document["rollouts"][1]["generated_token_ids"] = [999] + document["rollouts"][1]["generated_token_ids"]
    with pytest.raises(ValueError, match="steps differ"):
        validate_native_case(_case(), document, tokenizer=FakeTokenizer())


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda doc: doc["rollouts"][0]["predictions"]["predictions"][0].update({"description": "chair"}), "category"),
        (lambda doc: doc["rollouts"][0]["predictions"]["predictions"][0].update({"raw_span_text": "not-a-row"}), "tokenizer|subsequence"),
        (lambda doc: doc["rollouts"][0]["predictions"]["predictions"][0].update({"coord_bins": [10, 20, 30, 41]}), "disagree"),
    ],
)
def test_native_validation_rejects_bad_category_span_or_coordinate(mutate, message: str) -> None:
    document = _document()
    mutate(document)
    with pytest.raises(ValueError, match=message):
        validate_native_case(_case(), document, tokenizer=FakeTokenizer())


def test_native_validation_rejects_unknown_seed_or_prediction_order() -> None:
    with pytest.raises(ValueError, match="resolves to 0"):
        validate_native_case(_case(clean_seed=99), _document(), tokenizer=FakeTokenizer())
    document = _document()
    document["rollouts"][1]["predictions"]["predictions"][0]["generated_order"] = 1
    with pytest.raises(ValueError, match="generated_order 0 resolves to 0"):
        validate_native_case(_case(), document, tokenizer=FakeTokenizer())


def test_counterfactual_requires_explicit_purpose_and_records_opt_in() -> None:
    with pytest.raises(ValueError, match="non-empty purpose"):
        validate_native_case(
            _case(),
            _document(),
            tokenizer=FakeTokenizer(),
            expected_model_composition={**_expected_composition(), "adapter_path": "/models/other"},
            allow_counterfactual_model_composition=True,
        )
    result = validate_native_case(
        _case(),
        _document(),
        tokenizer=FakeTokenizer(),
        expected_model_composition={**_expected_composition(), "adapter_path": "/models/other"},
        allow_counterfactual_model_composition=True,
        counterfactual_purpose="Compare native trajectory source with another checkpoint.",
    )
    assert result["counterfactual_model_composition"] == {
        "enabled": True,
        "source_and_executed_composition_differ": True,
        "purpose": "Compare native trajectory source with another checkpoint.",
    }


def test_coordinate_path_log_probability_is_aggregated_from_raw_float32_probabilities() -> None:
    validated = validate_native_case(_case(), _document(), tokenizer=FakeTokenizer())
    token_map = coordinate_token_ids(FakeTokenizer())
    rows = []
    for slot in range(4):
        logits = [0.0] * 1200
        logits[token_map[validated["clean"]["coord_bins"][slot]]] = 4.0 + slot
        rows.append(logits)
    result = score_native_coordinate_states(
        validated,
        {"clean": rows, "degraded": copy.deepcopy(rows)},
        coordinate_ids=token_map,
        reference_bins=[10, 20, 30, 40],
    )
    expected = 0.0
    for slot, logits in enumerate(rows):
        maximum = max(logits)
        denominator = sum(math.exp(value - maximum) for value in logits)
        expected += math.log(math.exp(logits[token_map[validated["clean"]["coord_bins"][slot]]] - maximum) / denominator)
    assert result["clean"]["coordinate_path_log_probability_raw_temperature_1"] == pytest.approx(expected)
    assert len(result["paired_by_slot"]) == 4
    assert result["clean"]["coordinate_states"][0]["source_token_rank_raw_temperature_1"] == 1


def test_coordinate_token_ids_are_unique_and_complete() -> None:
    values = coordinate_token_ids(FakeTokenizer())
    assert len(values) == 1000
    assert len(set(values.values())) == 1000
