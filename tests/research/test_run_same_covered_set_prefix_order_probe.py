from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import pytest

import scripts.research.run_same_covered_set_prefix_order_probe as probe
from scripts.research.run_same_covered_set_prefix_order_probe import (
    ARMS,
    CASE_SCHEMA_VERSION,
    CASE_SCHEMA_VERSION_V2,
    SCHEMA_VERSION,
    _sha256_json,
    build_prefix_arms,
    extract_row_stop,
    materialize_entity_rows,
    validate_generated_row_boundary,
    validate_artifact_payload,
    validate_case_spec,
    validate_same_coverage_order_invariants,
    build_named_prefix_arms,
    match_predictions_to_entities,
    validate_named_prefix_invariants,
)


def _row(description: str, offset: int) -> list[int]:
    # The values are already-tokenized fixtures.  The runner must preserve
    # these ids exactly and must not decode/re-tokenize them.
    return [10 + offset, 20 + offset, 30 + offset, 40 + offset]


def _spec() -> dict:
    return {
        "schema_version": CASE_SCHEMA_VERSION,
        "image_id": "image-1",
        "entities": [
            {"entity_id": "A", "description": "person", "bbox_norm1000": [0, 0, 100, 100], "row_token_ids": _row("person", 1)},
            {"entity_id": "B", "description": "person", "bbox_norm1000": [200, 0, 300, 100], "row_token_ids": _row("person", 2)},
            {"entity_id": "C", "description": "person", "bbox_norm1000": [400, 0, 500, 100], "row_token_ids": _row("person", 3)},
        ],
        "cases": [{"case_id": "case-1", "a_entity_id": "A", "b_entity_id": "B", "c_entity_id": "C"}],
    }


def test_parse_case_ids_supports_comma_separated_repeated_arguments() -> None:
    assert probe.parse_case_ids("case-b,case-a") == ("case-b", "case-a")
    assert probe.parse_case_ids(["case-b,case-a", "case-c"]) == ("case-b", "case-a", "case-c")
    with pytest.raises(ValueError, match="unique"):
        probe.parse_case_ids(["case-a", "case-a"])


def test_same_coverage_arms_have_same_row_multiset_and_final_c_row() -> None:
    spec = validate_case_spec(_spec())
    rows = materialize_entity_rows(spec)
    arms = build_prefix_arms(spec["cases"][0], rows)

    left = arms["a_then_b_then_c"]
    right = arms["b_then_a_then_c"]
    assert Counter(tuple(row) for row in left["row_token_ids"]) == Counter(tuple(row) for row in right["row_token_ids"])
    assert set(left["covered_entity_ids"]) == set(right["covered_entity_ids"]) == {"A", "B", "C"}
    assert left["row_count"] == right["row_count"] == 3
    assert left["final_row_token_ids"] == right["final_row_token_ids"] == rows["C"]["row_token_ids"]
    assert left["prefix_token_ids"] != right["prefix_token_ids"]
    assert left["entity_ids"][:2] == ["A", "B"]
    assert right["entity_ids"][:2] == ["B", "A"]


def test_b_c_is_an_activation_coverage_control() -> None:
    spec = validate_case_spec(_spec())
    arms = build_prefix_arms(spec["cases"][0], materialize_entity_rows(spec))
    control = arms["b_then_c_coverage_control"]
    assert control["entity_ids"] == ["B", "C"]
    assert control["covered_entity_ids"] == ["B", "C"]
    assert control["row_count"] == 2
    assert control["prefix_token_ids"] == arms["b_then_a_then_c"]["row_token_ids"][0] + arms["b_then_a_then_c"]["row_token_ids"][2]


def test_repeated_a_b_c_identity_is_rejected_before_model_execution() -> None:
    spec = _spec()
    spec["cases"][0]["c_entity_id"] = "A"
    with pytest.raises(ValueError, match="distinct A/B/C"):
        validate_case_spec(spec)


def test_row_stop_extracts_one_complete_row_and_terminal_or_malformed_states() -> None:
    row = "<|object_ref_start|>person<|object_ref_end|><|box_start|><|coord_1|><|coord_2|><|coord_3|><|coord_4|><|box_end|>"
    complete = extract_row_stop("noise" + row + "tail")
    assert complete["stop_reason"] == "complete_row"
    assert complete["row_text"] == row
    assert complete["char_start"] == len("noise")
    exact = extract_row_stop(row)
    assert validate_generated_row_boundary(row, exact)["stop_reason"] == "complete_row"
    assert extract_row_stop("<|im_end|>")["stop_reason"] == "terminal"
    boundary = validate_generated_row_boundary("noise" + row, complete)
    assert boundary["stop_reason"] == "contaminated_complete_row"
    trailing = extract_row_stop(row + "<|im_end|>")
    assert validate_generated_row_boundary(row + "<|im_end|>", trailing)["stop_reason"] == "contaminated_complete_row"
    malformed = extract_row_stop("<|object_ref_start|>person\n<|object_ref_start|>person", malformed_limit=2)
    assert malformed["stop_reason"] == "malformed_limit"


def test_same_seed_parse_payload_has_required_artifact_shape() -> None:
    spec = validate_case_spec(_spec())
    rows = materialize_entity_rows(spec)
    arms = build_prefix_arms(spec["cases"][0], rows)
    parse_evidence = {"parse_status": "accepted", "predictions": [{"description": "person", "bbox": [0, 0, 10, 10]}], "dropped_predictions": []}
    runs = [{
        "mode": "sample",
        "seed": 11,
        "status": "success",
        "raw_generated_token_ids": [1, 2],
        "raw_generated_text": "raw",
        "row_stop": {"stop_reason": "length", "row_text": None, "row_text_sha256": None},
        "parse_evidence": parse_evidence,
    }]
    artifact_arms = {name: {**arm, "runs": runs} for name, arm in arms.items()}
    payload = {
        "schema_version": SCHEMA_VERSION,
        "config": {"seeds": [11]},
        "cases": [{
            "case_id": "case-1",
            "image_id": "image-1",
            "entity_ledger": list(rows.values()),
            "invariants": validate_same_coverage_order_invariants(artifact_arms, rows),
            "arms": artifact_arms,
        }],
    }
    validate_artifact_payload(payload)
    assert _sha256_json([1, 2]) == _sha256_json([1, 2])
    payload["cases"][0]["arms"][ARMS[0]]["runs"][0].pop("parse_evidence")
    with pytest.raises(ValueError, match="parse_evidence"):
        validate_artifact_payload(payload)


def test_v2_named_arms_validate_same_set_and_two_row_suffix() -> None:
    spec = {
        "schema_version": CASE_SCHEMA_VERSION_V2,
        "image_id": "image-v2",
        "entities": [
            {"entity_id": name, "description": "person", "bbox_norm1000": [offset * 100, 0, offset * 100 + 80, 100], "row_token_ids": _row(name, offset)}
            for name, offset in (("A", 1), ("B", 2), ("C", 3), ("D", 4))
        ],
        "cases": [{
            "case_id": "v2-case",
            "arms": {
                "canonical": {
                    "entity_ids": ["A", "B", "C", "D"],
                    "permutation_inversion_count_relative_to_canonical_order": 0,
                    "prefix_plausibility_status": "unavailable_in_this_pilot",
                },
                "swap": {
                    "entity_ids": ["B", "A", "C", "D"],
                    "permutation_inversion_count_relative_to_canonical_order": 1,
                    "prefix_plausibility_status": "unavailable_in_this_pilot",
                },
            },
            "comparisons": [{"comparison_id": "swap-ab", "arm_names": ["canonical", "swap"], "shared_suffix_length": 2, "rollout_horizon_rows": 4}],
        }],
    }
    checked = validate_case_spec(spec)
    rows = materialize_entity_rows(checked)
    case = checked["cases"][0]
    arms = build_named_prefix_arms(case, rows)
    invariants = validate_named_prefix_invariants(arms, rows, comparison=case["comparisons"][0])
    assert invariants["same_row_token_multiset"] is True
    assert invariants["shared_suffix_length"] == 2
    assert arms["canonical"]["row_token_ids"][-2:] == arms["swap"]["row_token_ids"][-2:]
    assert arms["canonical"]["permutation_inversion_count_relative_to_canonical_order"] == 0
    assert arms["swap"]["permutation_inversion_count_relative_to_canonical_order"] == 1
    assert arms["canonical"]["prefix_plausibility_status"] == "unavailable_in_this_pilot"


def test_v2_materialized_row_text_can_be_reused_without_double_field_rejection() -> None:
    row_text = "<|object_ref_start|>person<|object_ref_end|><|box_start|><|coord_1|><|coord_2|><|coord_3|><|coord_4|><|box_end|>"
    spec = {
        "schema_version": CASE_SCHEMA_VERSION_V2,
        "image_id": "image-row-text",
        "entities": [
            {"entity_id": name, "description": "person", "bbox_norm1000": [offset, 0, offset + 80, 100], "row_text": row_text}
            for name, offset in (("A", 0), ("B", 200), ("C", 400))
        ],
        "cases": [{
            "case_id": "text-case",
            "arms": {"left": {"entity_ids": ["A", "B", "C"]}, "right": {"entity_ids": ["B", "A", "C"]}},
            "comparisons": [{"comparison_id": "swap", "arm_names": ["left", "right"], "shared_suffix_length": 1, "rollout_horizon_rows": 1}],
        }],
    }
    checked = validate_case_spec(spec)

    class Tokenizer:
        def __init__(self) -> None:
            self.calls = 0

        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            assert text == row_text and add_special_tokens is False
            self.calls += 1
            return [self.calls, 2, 3]

    rows = materialize_entity_rows(checked, tokenizer=Tokenizer())
    arms = build_named_prefix_arms(checked["cases"][0], rows)
    assert arms["left"]["row_token_ids"] == [[1, 2, 3], [2, 2, 3], [3, 2, 3]]


def test_v2_rejects_invalid_horizon_and_suffix() -> None:
    spec = _spec()
    spec.update({"schema_version": CASE_SCHEMA_VERSION_V2})
    spec["cases"] = [{
        "case_id": "bad",
        "arms": {"left": {"entity_ids": ["A", "B", "C"]}, "right": {"entity_ids": ["B", "A", "C"]}},
        "comparisons": [{"comparison_id": "bad", "arm_names": ["left", "right"], "shared_suffix_length": 3, "rollout_horizon_rows": 5}],
    }]
    with pytest.raises(ValueError, match="shared_suffix_length"):
        validate_case_spec(spec)


def test_v2_accepts_named_arm_list_form() -> None:
    spec = _spec()
    spec.update({"schema_version": CASE_SCHEMA_VERSION_V2})
    spec["cases"] = [{
        "case_id": "list-arms",
        "arms": [{"arm_name": "left", "entity_ids": ["A", "B", "C"]}, {"arm_name": "right", "entity_ids": ["B", "A", "C"]}],
        "comparisons": [{"comparison_id": "swap", "arm_names": ["left", "right"], "shared_suffix_length": 1, "rollout_horizon_rows": 1}],
    }]
    checked = validate_case_spec(spec)
    assert set(checked["cases"][0]["arms"]) == {"left", "right"}


def test_v2_owner_matching_supports_non_person_categories_and_same_class_iou() -> None:
    entities = [
        {"entity_id": "cup-1", "description": "cup", "bbox_norm1000": [0, 0, 200, 200]},
        {"entity_id": "cup-2", "description": "cup", "bbox_norm1000": [400, 0, 600, 200]},
        {"entity_id": "laptop-1", "description": "laptop", "bbox_norm1000": [0, 400, 300, 700]},
    ]
    matches = match_predictions_to_entities(
        [
            {"description": "cup", "bbox": [0, 0, 20, 20]},
            {"description": "laptop", "bbox": [0, 40, 30, 70]},
            {"description": "bottle", "bbox": [0, 0, 20, 20]},
        ],
        entities,
        image_width=100,
        image_height=100,
        restrict_to_person=False,
    )
    assert matches[0]["status"] == "matched"
    assert matches[0]["matched_entity_id"] == "cup-1"
    assert matches[0]["top_same_category_candidate_id"] == "cup-1"
    assert matches[0]["top_same_category_candidate_iou"] == pytest.approx(1.0)
    assert matches[0]["top_same_category_candidate_center_distance_norm"] == pytest.approx(0.0)
    assert matches[1]["status"] == "matched"
    assert matches[1]["matched_entity_id"] == "laptop-1"
    assert matches[2]["status"] == "unmatched"
    assert matches[2]["reason"] == "no_same_description_entity"


def test_below_threshold_match_keeps_same_category_geometry_diagnostic() -> None:
    matches = match_predictions_to_entities(
        [{"description": "cup", "bbox": [70, 70, 90, 90]}],
        [{"entity_id": "cup-1", "description": "cup", "bbox_norm1000": [0, 0, 200, 200]}],
        image_width=100,
        image_height=100,
        restrict_to_person=False,
    )
    assert matches[0]["status"] == "unmatched"
    assert matches[0]["top_same_category_candidate_id"] == "cup-1"
    assert matches[0]["top_same_category_candidate_iou"] < 0.5
    assert matches[0]["top_same_category_candidate_center_distance_norm"] > 0


def test_horizon_partitions_covered_and_uncovered_owners(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_generate_one(**kwargs: object) -> dict:
        return {
            "mode": kwargs["mode"], "seed": kwargs["seed"], "status": "success",
            "raw_generated_token_ids": [123], "raw_generated_text": "row",
            "row_stop": {"stop_reason": "complete_row"}, "parse_evidence": {"parse_status": "accepted"},
            "parsed_predictions": [{"description": "cup", "bbox": [40, 0, 60, 20]}],
        }

    monkeypatch.setattr(probe, "_generate_one", fake_generate_one)
    result = probe._generate_horizon(
        session=object(), native_inputs={}, prefix_token_ids=[1], tokenizer=object(), image_width=100, image_height=100,
        mode="greedy", seed=None, temperature=0.2, top_p=0.95, repetition_penalty=1.0,
        max_new_tokens=16, malformed_limit=2, horizon_rows=1,
        entity_rows={
            "cup-covered": {"entity_id": "cup-covered", "description": "cup", "bbox_norm1000": [0, 0, 200, 200], "row_token_ids": [1]},
            "cup-new": {"entity_id": "cup-new", "description": "cup", "bbox_norm1000": [400, 0, 600, 200], "row_token_ids": [2]},
        },
        covered_entity_ids=["cup-covered"],
    )
    row = result["rows"][0]
    assert row["strict_matched_owner_ids"] == ["cup-new"]
    assert row["covered_prefix_owner_ids"] == []
    assert row["uncovered_ledger_owner_ids"] == ["cup-new"]


def test_horizon_appends_real_row_ids_and_seeds_only_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[int | None, tuple[int, ...], int]] = []
    seeds: list[int] = []

    def fake_generate_one(**kwargs: object) -> dict:
        calls.append((kwargs["seed"], tuple(kwargs["prefix_token_ids"]), int(kwargs["row_index"])))
        token = 900 + int(kwargs["row_index"])
        return {
            "mode": kwargs["mode"],
            "seed": kwargs["seed"],
            "status": "success",
            "raw_generated_token_ids": [token],
            "raw_generated_text": "raw",
            "row_stop": {"stop_reason": "complete_row"},
            "parse_evidence": {"parse_status": "accepted"},
            "parsed_predictions": [],
        }

    monkeypatch.setattr(probe, "_generate_one", fake_generate_one)
    monkeypatch.setattr(probe, "_seed_torch", lambda seed: seeds.append(int(seed)))
    result = probe._generate_horizon(
        session=object(), native_inputs={}, prefix_token_ids=[1, 2], tokenizer=object(), image_width=100, image_height=100,
        mode="sample", seed=123, temperature=0.2, top_p=0.95, repetition_penalty=1.0,
        max_new_tokens=16, malformed_limit=2, horizon_rows=3,
        entity_rows={"A": {"entity_id": "A", "bbox_norm1000": [0, 0, 100, 100], "row_token_ids": [1]}},
    )
    assert [call[0] for call in calls] == [None, None, None]
    assert seeds == [123]
    assert [call[1] for call in calls] == [(1, 2), (1, 2, 900), (1, 2, 900, 901)]
    assert [call[2] for call in calls] == [0, 1, 2]
    assert result["final_prefix_token_ids"] == [1, 2, 900, 901, 902]
    assert result["horizon_complete"] is True


@pytest.mark.parametrize("stop_reason", ["terminal", "malformed_limit", "contaminated_complete_row"])
def test_horizon_does_not_append_noncomplete_attempts(monkeypatch: pytest.MonkeyPatch, stop_reason: str) -> None:
    def fake_generate_one(**kwargs: object) -> dict:
        return {
            "mode": kwargs["mode"], "seed": kwargs["seed"], "status": "success",
            "raw_generated_token_ids": [999], "raw_generated_text": "attempt",
            "row_stop": {"stop_reason": stop_reason}, "parse_evidence": {"parse_status": "not_run"},
            "parsed_predictions": [],
        }

    monkeypatch.setattr(probe, "_generate_one", fake_generate_one)
    result = probe._generate_horizon(
        session=object(), native_inputs={}, prefix_token_ids=[1, 2], tokenizer=object(), image_width=100, image_height=100,
        mode="greedy", seed=None, temperature=0.2, top_p=0.95, repetition_penalty=1.0,
        max_new_tokens=16, malformed_limit=2, horizon_rows=2,
        entity_rows={"A": {"entity_id": "A", "bbox_norm1000": [0, 0, 100, 100], "row_token_ids": [1]}},
    )
    assert result["status"] == "failed"
    assert result["horizon_rows_generated"] == 0
    assert result["final_prefix_token_ids"] == [1, 2]
    assert result["rows"][0]["accepted_complete_row"] is False


def test_generate_one_marks_leading_contamination_before_parsing() -> None:
    import torch

    row = "<|object_ref_start|>person<|object_ref_end|><|box_start|><|coord_1|><|coord_2|><|coord_3|><|coord_4|><|box_end|>"

    class Tokenizer:
        def decode(self, values: object, **_: object) -> str:
            return "noise" + row

    class Model:
        def generate(self, **kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(sequences=torch.tensor([[1, 2]], dtype=torch.long))

    class Session:
        _model = Model()
        _tokenizer = Tokenizer()

        def _im_end_token_id(self) -> int:
            return 3

        def _pad_token_id(self) -> int:
            return 0

    result = probe._generate_one(
        session=Session(), native_inputs={"input_ids": torch.tensor([[1]], dtype=torch.long)}, prefix_token_ids=[],
        tokenizer=Tokenizer(), image_width=100, image_height=100, mode="greedy", seed=None,
        temperature=0.2, top_p=0.95, repetition_penalty=1.0, max_new_tokens=16, malformed_limit=2,
    )
    assert result["status"] == "failed"
    assert result["row_stop"]["stop_reason"] == "contaminated_complete_row"
