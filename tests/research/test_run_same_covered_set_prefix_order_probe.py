from __future__ import annotations

from collections import Counter

import pytest

from scripts.research.run_same_covered_set_prefix_order_probe import (
    ARMS,
    CASE_SCHEMA_VERSION,
    SCHEMA_VERSION,
    _sha256_json,
    build_prefix_arms,
    extract_row_stop,
    materialize_entity_rows,
    validate_artifact_payload,
    validate_case_spec,
    validate_same_coverage_order_invariants,
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
    assert extract_row_stop("<|im_end|>")["stop_reason"] == "terminal"
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
