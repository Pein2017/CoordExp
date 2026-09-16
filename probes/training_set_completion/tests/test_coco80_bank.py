from __future__ import annotations

import copy

import pytest

from probes.training_set_completion.coco80_bank import EOS, _selected_by_image, reassemble_route
from probes.training_set_completion.training import validate_route


class _Tokenizer:
    _special = {
        "<|object_ref_start|>": 1001,
        "<|object_ref_end|>": 1002,
        "<|box_start|>": 1003,
        "<|box_end|>": 1004,
    }
    _words = {"alpha": [11], "beta gamma": [12, 13]}

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return self._words[text]

    def decode(self, ids: list[int], *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is False
        inverse = {tuple(value): key for key, value in self._words.items()}
        return inverse[tuple(ids)]

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._special[token]


def _source_route() -> dict:
    case = {
        "image_path": "/tmp/example.jpg",
        "image_plan": {
            "image_content_sha256": "a",
            "executed_media_sha256": "b",
            "observed_image_grid_thw": [1, 2, 3],
        },
    }

    def card(owner: str, description: str, bins: list[int], order: int) -> dict:
        return {
            "owner_id": owner,
            "order": order,
            "source_kind": "source",
            "source_decision": {"proposal_id": owner},
            "edited_fields": {
                "selected_description": description,
                "catalog_reference_coord_bins_1000": bins,
                "description_source_proposal_id": owner,
                "geometry_replaced": False,
            },
        }

    return {
        "route_id": "source:image-000000000007",
        "image_id": 7,
        "example_id": "example-7",
        "case": case,
        "prompt_token_ids": [99],
        "provenance": {
            "trace": [
                card("owner-a", "alpha", [1, 2, 3, 4], 0),
                card("owner-b", "beta gamma", [5, 6, 7, 8], 1),
            ]
        },
    }


def _selected() -> list[dict]:
    return [
        {
            "image_id": 7,
            "owner_id": "owner-a",
            "source_order": 0,
            "source_description": "alpha",
            "source_description_ce_positive": True,
            "reference_bins": [1, 2, 3, 4],
            "decision": "included",
        },
        {
            "image_id": 7,
            "owner_id": "owner-b",
            "source_order": 1,
            "source_description": "beta gamma",
            "source_description_ce_positive": True,
            "reference_bins": [5, 6, 7, 8],
            "decision": "included",
        },
    ]


def test_reassemble_route_rebuilds_positions_and_preserves_owner_order() -> None:
    route = reassemble_route(
        _source_route(), _selected(), tokenizer=_Tokenizer(), coordinate_ids=list(range(1000))
    )

    assert route["continuation_token_ids"][-1] == EOS
    assert route["provenance"]["selected_owner_ids"] == ["owner-a", "owner-b"]
    assert route["trusted_boxes"] == [
        {"x1_position": 4, "y1_position": 5, "x2_position": 6, "y2_position": 7, "expected_bins": [1, 2, 3, 4]},
        {"x1_position": 14, "y1_position": 15, "x2_position": 16, "y2_position": 17, "expected_bins": [5, 6, 7, 8]},
    ]
    shifted = copy.deepcopy(route)
    shifted["trusted_boxes"][0]["x1_position"] += 4
    with pytest.raises(ValueError, match="trusted box positions"):
        validate_route(shifted, eos_token_id=EOS, coordinate_token_ids=list(range(1000)))


def test_reassemble_route_rejects_cross_object_description_or_geometry() -> None:
    source = _source_route()
    source["provenance"]["trace"][1]["edited_fields"]["selected_description"] = "alpha"
    with pytest.raises(ValueError, match="cross-object description"):
        reassemble_route(source, _selected(), tokenizer=_Tokenizer(), coordinate_ids=list(range(1000)))

    source = _source_route()
    source["provenance"]["trace"][1]["edited_fields"]["catalog_reference_coord_bins_1000"] = [1, 2, 3, 4]
    with pytest.raises(ValueError, match="cross-object geometry"):
        reassemble_route(source, _selected(), tokenizer=_Tokenizer(), coordinate_ids=list(range(1000)))


def test_selected_teacher_rejects_literal_outside_coco80_before_reassembly() -> None:
    preparation = {
        "owners": [
            {
                "image_id": 7,
                "owner_id": "out-of-scope-figurine",
                "source_order": 0,
                "source_description": "figurine",
                "source_description_ce_positive": True,
                "reference_bins": [1, 2, 3, 4],
                "decision": "included",
            }
        ]
    }
    with pytest.raises(ValueError, match="outside literal COCO-80"):
        _selected_by_image(preparation)
