from __future__ import annotations

import pytest
import torch

from src.qwen.native import combine_singleton_native_inputs


def singleton(ids, pixels):
    return {
        "input_ids": torch.tensor([ids]),
        "attention_mask": torch.ones(1, len(ids)),
        "pixel_values": torch.tensor(pixels).float(),
        "image_grid_thw": torch.tensor([[1, 2, 2]]),
    }


def test_combining_singletons_preserves_media_order_and_replaces_only_history():
    first = singleton([10], [[1, 2], [3, 4]])
    second = singleton([20, 21], [[5, 6]])
    combined = combine_singleton_native_inputs(
        [first, second], prompt_token_ids=[[10], [20, 21]]
    )
    assert combined["input_ids"].tolist() == [[0, 10], [20, 21]]
    assert combined["pixel_values"].tolist() == [[1, 2], [3, 4], [5, 6]]
    assert combined["image_grid_thw"].tolist() == [[1, 2, 2], [1, 2, 2]]
    assert "attention_mask" not in combined  # exact_history_inputs reconstructs it.
    assert first["input_ids"].tolist() == [[10]]
    assert second["input_ids"].tolist() == [[20, 21]]


@pytest.mark.parametrize(
    "value, message", [("unsupported", "non-tensor"), (torch.tensor(1), "scalar")]
)
def test_unknown_native_field_semantics_fail_before_a_model_call(value, message):
    row = {**singleton([10], [[1, 2]]), "unknown_field": value}
    with pytest.raises(ValueError, match=message):
        combine_singleton_native_inputs([row], prompt_token_ids=[[10]])


def test_only_singletons_with_matching_prompt_cardinality_are_admitted():
    with pytest.raises(ValueError, match="cardinality"):
        combine_singleton_native_inputs([], prompt_token_ids=[])
    row = singleton([10], [[1, 2]])
    with pytest.raises(ValueError, match="cardinality"):
        combine_singleton_native_inputs([row], prompt_token_ids=[])
    row["input_ids"] = torch.tensor([[10], [11]])
    with pytest.raises(ValueError, match="singleton"):
        combine_singleton_native_inputs([row], prompt_token_ids=[[10]])
