"""Pure contracts for the human-refined completion causal micro-panel."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts/research/run_human_refined_completion_causal_micro_panel.py"
)
SPEC = importlib.util.spec_from_file_location("human_refined_causal_micro_panel", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _manifest() -> dict[str, object]:
    return {
        "schema_version": MODULE.SCHEMA_VERSION,
        "cases": [
            {
                "case_id": "case-5001",
                "image_id": "5001",
                "parent_prefix_owner_ids": ["parent-1"],
                "released_row_budget": 1,
                "branch_arms": [
                    {"kind": "description_gt_owner", "owner_id": "remaining-1"},
                    {"kind": "first_coordinate_gt_owner", "owner_id": "remaining-1"},
                    {"kind": "first_two_coordinates_gt_owner", "owner_id": "remaining-1"},
                    {"kind": "first_three_coordinates_gt_owner", "owner_id": "remaining-1"},
                    {"kind": "complete_gt_owner", "owner_id": "remaining-2"},
                    {"kind": "covered_gt_owner", "owner_id": "parent-1"},
                ],
            }
        ],
    }


def test_manifest_validation_normalizes_v1_and_rejects_bad_budget_or_kind() -> None:
    normalized = MODULE.validate_case_manifest(_manifest())
    assert normalized["schema_version"] == MODULE.SCHEMA_VERSION
    case = normalized["cases"][0]
    assert case["released_row_budget"] == 1
    assert case["branch_arms"][0]["kind"] == "description_gt_owner"
    assert [arm["owner_id"] for arm in case["branch_arms"][:4]] == [
        "remaining-1",
        "remaining-1",
        "remaining-1",
        "remaining-1",
    ]
    assert all(arm["operational_meaning"] for arm in case["branch_arms"])

    invalid_budget = _manifest()
    invalid_budget["cases"][0]["released_row_budget"] = 9
    with pytest.raises(ValueError, match="released_row_budget"):
        MODULE.validate_case_manifest(invalid_budget)

    invalid_kind = _manifest()
    invalid_kind["cases"][0]["branch_arms"][0]["kind"] = "unknown"
    with pytest.raises(ValueError, match="unsupported kind"):
        MODULE.validate_case_manifest(invalid_kind)


class _Tokenizer:
    ids = {
        "<|object_ref_start|>": 151646,
        "<|object_ref_end|>": 151647,
        "person": [701, 702],
    }

    def convert_tokens_to_ids(self, token: str) -> int:
        value = self.ids[token]
        assert isinstance(value, int)
        return value

    def __call__(self, text: str, *, add_special_tokens: bool = False) -> dict[str, list[int]]:
        assert add_special_tokens is False
        return {"input_ids": list(self.ids[text])}


def test_description_prefix_uses_exact_gt_description_tokens_and_wrappers() -> None:
    row = {"owner_id": "owner-1", "description": "person"}
    assert MODULE.build_description_prefix_token_ids(row, _Tokenizer()) == [151646, 701, 702, 151647]


def test_coordinate_branch_prefixes_slice_exact_validated_gt_row_tokens() -> None:
    row = [
        MODULE.OBJECT_REF_START_TOKEN_ID,
        701,
        702,
        MODULE.OBJECT_REF_END_TOKEN_ID,
        MODULE.BOX_START_TOKEN_ID,
        MODULE.COORDINATE_TOKEN_MIN_ID + 11,
        MODULE.COORDINATE_TOKEN_MIN_ID + 22,
        MODULE.COORDINATE_TOKEN_MIN_ID + 33,
        MODULE.COORDINATE_TOKEN_MIN_ID + 44,
        MODULE.BOX_END_TOKEN_ID,
    ]
    assert MODULE.build_partial_gt_owner_prefix_token_ids(
        row, kind="first_coordinate_gt_owner", owner_id="owner-1"
    ) == row[:6]
    assert MODULE.build_partial_gt_owner_prefix_token_ids(
        row, kind="first_two_coordinates_gt_owner", owner_id="owner-1"
    ) == row[:7]
    assert MODULE.build_partial_gt_owner_prefix_token_ids(
        row, kind="first_three_coordinates_gt_owner", owner_id="owner-1"
    ) == row[:8]


@pytest.mark.parametrize(
    "malformed",
    [
        # Only three coordinates precede box_end.
        [151646, 701, 151647, 151648, 151670, 151671, 151672, 151649],
        # A lexical token is present where a coordinate is required.
        [151646, 701, 151647, 151648, 151670, 12, 151672, 151673, 151649],
        # Two box_start markers make the row boundary ambiguous.
        [151646, 701, 151647, 151648, 151648, 151670, 151671, 151672, 151673, 151649],
        # The complete row is prefixed by an unrelated token.
        [99, 151646, 701, 151647, 151648, 151670, 151671, 151672, 151673, 151649],
    ],
)
def test_coordinate_branch_prefix_rejects_malformed_complete_gt_rows(
    malformed: list[int],
) -> None:
    with pytest.raises(ValueError, match="invalid|must contain|structural|start"):
        MODULE.build_partial_gt_owner_prefix_token_ids(
            malformed,
            kind="first_coordinate_gt_owner",
            owner_id="owner-1",
        )


def test_branch_and_suffix_partition_excludes_branch_from_released_discovery() -> None:
    result = MODULE.partition_branch_suffix_owner_matches(
        ["branch-owner"],
        ["branch-owner", "parent-owner", "remaining-owner"],
        ["parent-owner"],
        ["branch-owner", "remaining-owner"],
    )
    assert result["branch_owner_credits"] == ["branch-owner"]
    assert result["released_only_remaining_owner_ids"] == ["remaining-owner"]
    assert result["total_new_remaining_owner_ids_including_branch"] == [
        "branch-owner",
        "remaining-owner",
    ]
    assert result["forced_context_repeats"] == ["parent-owner"]
    assert result["completion_against_remaining_owner_set"] is True


def test_unmatched_forced_owner_cannot_leak_from_suffix_into_discovery() -> None:
    result = MODULE.partition_branch_suffix_owner_matches(
        [],
        ["forced-owner", "other-owner"],
        [],
        ["forced-owner", "other-owner"],
        ["forced-owner"],
    )
    assert result["released_only_remaining_owner_ids"] == ["other-owner"]
    assert result["total_new_remaining_owner_ids_including_branch"] == [
        "other-owner"
    ]
    assert result["released_declared_forced_owner_revisits"] == ["forced-owner"]
    assert result["completion_against_remaining_owner_set"] is False


def test_branch_suffix_matching_is_global_across_all_rows() -> None:
    owners = [
        {"owner_id": "owner-a", "category": "person", "bbox": [0, 0, 100, 100]},
        {"owner_id": "owner-b", "category": "person", "bbox": [200, 0, 300, 100]},
    ]
    branch = {
        "parsed_predictions": [
            {
                "prediction_id": "branch-a",
                "generated_row_index": 0,
                "category": "person",
                "bbox": [0, 0, 100, 100],
            }
        ]
    }
    suffix = [
        {
            "parsed_predictions": [
                {
                    "prediction_id": "suffix-a-repeat",
                    "generated_row_index": 1,
                    "category": "person",
                    "bbox": [0, 0, 100, 100],
                }
            ]
        },
        {
            "parsed_predictions": [
                {
                    "prediction_id": "suffix-b",
                    "generated_row_index": 2,
                    "category": "person",
                    "bbox": [200, 0, 300, 100],
                }
            ]
        },
    ]
    result = MODULE._branch_and_suffix_partition(
        branch,
        suffix,
        owner_rows=owners,
        parent_owner_ids=[],
        remaining_owner_ids=["owner-a", "owner-b"],
    )
    assert result["branch_owner_credits"] == ["owner-a"]
    assert result["released_only_remaining_owner_ids"] == ["owner-b"]
    assert result["global_branch_suffix_owner_matching"][
        "conservative_lower_bound_coverage"
    ] == 2


def test_raw_noop_parity_is_exact_and_reports_a_failure() -> None:
    natural = [
        {"raw_generated_token_ids": [1, 2]},
        {"raw_generated_token_ids": [3, 4]},
    ]
    assert MODULE.compare_raw_noop_parity(natural, [dict(row) for row in natural])["passed"] is True
    failed = MODULE.compare_raw_noop_parity(
        natural,
        [{"raw_generated_token_ids": [1, 2]}, {"raw_generated_token_ids": [3, 5]}],
    )
    assert failed["passed"] is False
    assert failed["first_failure_row_index"] == 1


def test_causal_admissibility_requires_a_valid_noop() -> None:
    assert MODULE.causal_admissibility_from_noop({"status": "valid"}) == (True, None)
    admissible, reason = MODULE.causal_admissibility_from_noop(
        {"status": "not_applicable"}
    )
    assert admissible is False
    assert reason == "native_complete_row_noop_not_applicable"
