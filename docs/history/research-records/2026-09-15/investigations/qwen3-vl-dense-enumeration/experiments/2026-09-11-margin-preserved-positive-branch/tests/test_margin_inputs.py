from __future__ import annotations

import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("prepare_margins", HERE / "prepare_margins.py")
assert SPEC is not None and SPEC.loader is not None
MARGIN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MARGIN)


def test_strict_threshold_and_floor_fixture() -> None:
    assert MARGIN.classify_margin(source_is_argmax=True, margin=0.001) == "original_mask_near_tie"
    assert MARGIN.classify_margin(source_is_argmax=True, margin=0.0010001) == "eligible"
    assert MARGIN.margin_floor(0.0010001) == 0.00050005
    assert MARGIN.margin_floor(1.0) == 0.1
    assert MARGIN.classify_margin(source_is_argmax=False, margin=-0.25) == "source_literal_not_argmax"


def test_source_population_and_frozen_counts() -> None:
    packet = MARGIN.build_packet()
    assert packet["schema"] == "margin_preserved_positive_branch.margin_inputs.v1"
    assert packet["status"] == "prepared_no_model_execution"
    assert packet["counts"] == {
        "cases": 56,
        "action_tokens": 6056,
        "original_mask_positions": 6047,
        "eligible_positions": 6030,
        "original_mask_near_ties": 17,
        "invalid_geometry_tokens": 9,
        "source_literal_nonargmax": 0,
        "normal_count": 56,
        "normal_action_tokens": 6056,
        "normal_kl_positions": 6047,
        "eligible_margin_positions": 6030,
        "retained_kl_only_positions": 17,
    }
    assert packet["excluded_reason_counts"] == {
        "original_mask_near_tie": 17,
        "invalid_geometry_token": 9,
        "source_literal_not_argmax": 0,
    }
    assert len(packet["cases"]) == 56


def test_parallel_margin_arrays_and_original_masks() -> None:
    packet = MARGIN.build_packet()
    total_eligible = 0
    total_near = 0
    total_invalid = 0
    for case in packet["cases"]:
        parallel = [
            case["eligible_positions"],
            case["target_ids"],
            case["literal_target_ids"],
            case["source_margins"],
            case["floors"],
            case["token_categories"],
        ]
        assert len({len(values) for values in parallel}) == 1
        assert case["eligible_positions"]
        assert set(case["eligible_positions"]).issubset(set(case["original_mask_positions"]))
        for margin, floor in zip(case["source_margins"], case["floors"]):
            assert margin > MARGIN.EPSILON
            assert floor == min(0.1, 0.5 * margin)
        total_eligible += case["counts"]["eligible_positions"]
        total_near += case["counts"]["original_mask_near_ties"]
        total_invalid += case["counts"]["invalid_geometry_tokens"]
    assert (total_eligible, total_near, total_invalid) == (6030, 17, 9)


def test_invalid_geometry_positions_and_no_selection_labels() -> None:
    packet = MARGIN.build_packet()
    case = next(row for row in packet["cases"] if row["image_id"] == "360573")
    invalid = [row for row in case["excluded_positions"] if row["reason"] == "invalid_geometry_token"]
    assert [row["position"] for row in invalid] == list(range(75, 84))
    assert set(row["position"] for row in invalid).isdisjoint(case["eligible_positions"])
    assert packet["claim_boundary"]["not_selected_by"] == [
        "A flips", "GT", "owner loss", "natural endpoint outputs"
    ]


def test_identity_bindings_and_cold_readback(tmp_path: Path) -> None:
    result = MARGIN.write_outputs(tmp_path)
    packet_path = tmp_path / "inputs.json"
    receipt_path = tmp_path / "preparation-receipt.json"
    fixture_path = tmp_path / "schema-fixture.json"
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert result["packet"]["content_sha256"] == packet["content_sha256"]
    assert receipt["cold_readback"]["content_sha256_verified"] is True
    assert receipt["cold_readback"]["counts_verified"] is True
    assert fixture["threshold"]["strict_boundary"]["m0_equal_epsilon"]["eligible"] is False
    assert fixture["invalid_geometry_fixture"]["positions"] == list(range(75, 84))
    assert packet["source_bindings"]["microscope_input"]["sha256"] == MARGIN.EXPECTED_HASHES["microscope_input"]
    assert packet["trainer_input_identity"]["manifest_sha256"] == MARGIN.EXPECTED_HASHES["candidate_manifest"]
    # A second call must not rewrite a different packet under the immutable-output rule.
    MARGIN.write_outputs(tmp_path)
