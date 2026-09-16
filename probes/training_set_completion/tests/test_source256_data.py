from __future__ import annotations

import copy

import pytest

from probes.training_set_completion import source256_data as data


class Tokenizer:
    eos_token_id = data.EOS

    def __init__(self) -> None:
        self.by_text = {"cat": [101], "dog": [102]}
        self.by_ids = {tuple(value): key for key, value in self.by_text.items()}

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return list(self.by_text[text])

    def decode(self, ids, **kwargs):
        return self.by_ids[tuple(ids)]


def owner(owner_id: str, description: str, box: list[int]) -> dict:
    return {
        "owner_id": owner_id,
        "description": description,
        "normalized_description": description,
        "coord_bins": box,
    }


def prediction(index: int, description: str, box: list[int], start: int, text: str) -> dict:
    return {
        "bbox": box,
        "description": description,
        "generated_order": index,
        "object_span_id": f"span-{index}",
        "raw_span_sha256": str(index) * 64,
        "char_start": start,
        "char_end": start + len(text),
    }


def test_longest_prefix_stops_at_first_unmatched_row_without_skipping_later_match():
    texts = ["row-0<|box_end|>", "row-1<|box_end|>", "row-2<|box_end|>"]
    starts = [0, len(texts[0]), len(texts[0]) + len(texts[1])]
    row = {
        "row_id": "case",
        "image_width": 1000,
        "image_height": 1000,
        "pred": [
            prediction(0, "cat", [0, 0, 100, 100], starts[0], texts[0]),
            prediction(1, "cat", [300, 300, 400, 400], starts[1], texts[1]),
            prediction(2, "dog", [500, 500, 600, 600], starts[2], texts[2]),
        ],
        "dropped_predictions": [],
        "decode_stop_reason": "im_end",
        "raw_decode_text": "".join(texts),
    }
    traces = []
    for index, text in enumerate(texts):
        traces.extend(
            [
                {"token_id": data.ROW_START, "token_text": text[: -len("<|box_end|>")]},
                {"token_id": data.ROW_END, "token_text": "<|box_end|>"},
            ]
        )
    evidence = data._prefix_evidence(
        row,
        traces,
        [owner("one", "cat", [0, 0, 100, 100]), owner("two", "dog", [500, 500, 600, 600])],
    )
    assert evidence["prefix_owner_ids"] == ["one"]
    assert evidence["remaining_owner_ids"] == ["two"]
    assert evidence["prefix_token_ids"] == [data.ROW_START, data.ROW_END]
    assert evidence["boundary"]["pred_index"] == 1
    assert evidence["structural_candidate"] is True


def _route_inputs():
    source = {"image_id": 7}
    group = {
        "image_id": "7",
        "example_id": "image-7",
        "image_path": "/tmp/image-7.jpg",
        "image_content_sha256": "a" * 64,
        "executed_media_sha256": "b" * 64,
        "observed_image_grid_thw": [1, 2, 3],
        "prompt_token_ids": [10, 11],
    }
    greedy = {"active_traces": [{"token_id": data.ROW_START}, {"token_id": data.ROW_END}]}
    owners = [owner("one", "cat", [1, 2, 3, 4]), owner("two", "dog", [5, 6, 7, 8])]
    prefix = {
        "prefix_token_ids": [data.ROW_START, 101, 151647, 151648, 151671, 151672, 151673, 151674, data.ROW_END],
        "prefix_owner_ids": ["one"],
    }
    return source, group, greedy, owners, prefix


def test_completion_route_masks_prefix_and_supervises_all_suffix_tokens():
    source, group, greedy, owners, prefix = _route_inputs()
    route = data._route(
        tokenizer=Tokenizer(),
        route_kind="fixed_source_prefix_completion",
        source=source,
        group=group,
        greedy=greedy,
        owners=owners,
        prefix=prefix,
    )
    n = len(prefix["prefix_token_ids"])
    assert route["continuation_token_ids"][:n] == prefix["prefix_token_ids"]
    assert route["ce_weights"][:n] == [0] * n
    assert route["labels"][:n] == [data.IGNORE_INDEX] * n
    assert route["geometry_weights"][:n] == [0] * n
    assert all(route["ce_weights"][n:])
    assert route["continuation_token_ids"][-1] == data.EOS
    assert route["labels"][-1] == data.EOS
    assert route["provenance"]["prefix_owner_ids"] == ["one"]
    assert route["provenance"]["suffix_owner_ids"] == ["two"]
    assert len(route["trusted_boxes"]) == 1
    data._validate_route_extras(route)


def test_canonical_fallback_is_full_bank_and_has_no_masked_target_tokens():
    source, group, greedy, owners, _ = _route_inputs()
    route = data._route(
        tokenizer=Tokenizer(),
        route_kind="canonical",
        source=source,
        group=group,
        greedy=greedy,
        owners=owners,
        prefix=None,
    )
    assert route["provenance"]["prefix_owner_ids"] == []
    assert route["provenance"]["suffix_owner_ids"] == ["one", "two"]
    assert route["ce_weights"] == [1] * len(route["continuation_token_ids"])
    assert route["labels"] == route["continuation_token_ids"]
    assert len(route["trusted_boxes"]) == 2
    data._validate_route_extras(route)


def test_route_extra_validator_rejects_geometry_on_masked_prefix():
    source, group, greedy, owners, prefix = _route_inputs()
    route = data._route(
        tokenizer=Tokenizer(),
        route_kind="fixed_source_prefix_completion",
        source=source,
        group=group,
        greedy=greedy,
        owners=owners,
        prefix=prefix,
    )
    broken = copy.deepcopy(route)
    broken["geometry_weights"][0] = 1
    with pytest.raises(ValueError, match="geometry weights"):
        data._validate_route_extras(broken)


def test_schedule_has_exact_full_owner_exposure_shape():
    schedule = data._schedule(list(range(256)), seed=19)
    assert len(schedule["updates"]) == 64
    common = [image for update in schedule["updates"] for image in update["common_image_ids"]]
    variable = [image for update in schedule["updates"] for image in update["variable_image_ids"]]
    assert len(common) == len(variable) == 2048
    assert set(common.count(image) for image in range(256)) == {8}
    assert set(variable.count(image) for image in range(256)) == {8}
