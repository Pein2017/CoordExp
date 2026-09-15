from __future__ import annotations

import copy

import pytest

from probes.training_set_completion.coco227_data import EOS, _literal_coco80, reassemble_route
from probes.training_set_completion.complete_bank import _field_ids
from probes.training_set_completion.training import validate_route


class _Tokenizer:
    _special = {
        "<|object_ref_start|>": 1001,
        "<|object_ref_end|>": 1002,
        "<|box_start|>": 1003,
        "<|box_end|>": 1004,
    }
    _words = {"person": [11], "donut": [12], "chair": [13]}

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return self._words[text]

    def decode(self, ids: list[int], *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is False
        return {tuple(value): key for key, value in self._words.items()}[tuple(ids)]

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._special[token]


def _old_route() -> dict:
    tokenizer, coordinates = _Tokenizer(), list(range(1000))
    bins = [1, 2, 3, 4]
    return {
        "route_id": "old:image-000000000007",
        "image_id": 7,
        "example_id": "example-7",
        "case": {
            "image_path": "/tmp/example.jpg",
            "image_plan": {"image_content_sha256": "image", "executed_media_sha256": "media", "observed_image_grid_thw": [1, 2, 3]},
        },
        "prompt_token_ids": [99],
        "continuation_token_ids": _field_ids(tokenizer, "person", bins, coordinates) + [EOS],
        "ce_weights": [1] * 10,
        "trusted_boxes": [{"x1_position": 4, "y1_position": 5, "x2_position": 6, "y2_position": 7, "expected_bins": bins}],
        "provenance": {
            "trace": [
                {
                    "owner_id": "old-person",
                    "order": 0,
                    "source_kind": "old",
                    "source_order": 0,
                    "source_decision": {"decision": "included"},
                    "edited_fields": {
                        "selected_description": "person",
                        "description_token_positions": [1],
                        "catalog_reference_coord_bins_1000": bins,
                    },
                }
            ]
        },
    }


def test_reassemble_appends_only_after_old_literal_prefix_and_rebuilds_positions() -> None:
    old = _old_route()
    addition = {
        "image_id": 7,
        "owner_id": "new-donut",
        "description": "donut",
        "reference_coord_bins_1000": [10, 20, 30, 40],
        "source_current_known248": {"image_id": 7, "owner_id": "new-donut", "description": "donut"},
    }
    route = reassemble_route(old, [addition], tokenizer=_Tokenizer(), coordinate_ids=list(range(1000)))

    assert route["continuation_token_ids"][: -1][: -9] == old["continuation_token_ids"][:-1]
    assert route["provenance"]["trace"][:1] == old["provenance"]["trace"]
    assert route["provenance"]["new9_owner_ids"] == ["new-donut"]
    assert route["trusted_boxes"][1] == {"x1_position": 13, "y1_position": 14, "x2_position": 15, "y2_position": 16, "expected_bins": [10, 20, 30, 40]}

    shifted = copy.deepcopy(route)
    shifted["trusted_boxes"][1]["x1_position"] += 1
    with pytest.raises(ValueError, match="positions"):
        validate_route(shifted, eos_token_id=EOS, coordinate_token_ids=list(range(1000)))


def test_literal_coco80_boundary_rejects_nonmember_without_normalization() -> None:
    with pytest.raises(ValueError, match="outside literal COCO-80"):
        _literal_coco80("Figurine", owner_id="bad-case")
    with pytest.raises(ValueError, match="outside literal COCO-80"):
        _literal_coco80("figurine", owner_id="bad-class")
