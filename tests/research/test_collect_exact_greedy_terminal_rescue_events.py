from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from src.config.fingerprint import sha256_json
from src.rollout_calibration import CheckpointIdentity, assemble_state_bank, load_state_bank
from scripts.research import collect_exact_greedy_terminal_rescue_events as collector


def _target(owner: str, category: str, box: list[int]) -> dict[str, object]:
    return {"object_id": owner, "description": category, "bbox": box}


def test_trace_terminal_prefix_excludes_im_end_and_preserves_order() -> None:
    rows = [
        {"row_id": "image-a", "trace_type": "generated_token", "generated_step_index": 0, "token_id": 151646, "is_stop": False},
        {"row_id": "image-a", "trace_type": "generated_token", "generated_step_index": 1, "token_id": 151670, "is_stop": False},
        {"row_id": "image-a", "trace_type": "generated_token", "generated_step_index": 2, "token_id": 151645, "is_stop": True},
        {"row_id": "image-a", "trace_type": "generated_token", "generated_step_index": 3, "token_id": 151643, "is_stop": False, "is_pad": True},
    ]
    result = collector.trace_terminal_prefix(rows, image_id="image-a")
    assert result["prefix_token_ids"] == [151646, 151670]
    assert result["generated_token_ids"] == [151646, 151670, 151645]
    assert result["terminal_token_ids"] == [151645]


def test_target_noncoverage_records_every_prior_row_and_rejects_overlap() -> None:
    target = _target("target", "person", [400, 400, 500, 600])
    prior = [
        {"description": "person", "coord_bins": [0, 0, 100, 100]},
        {"description": "chair", "coord_bins": [380, 390, 510, 610]},
    ]
    allowed, receipts = collector.target_noncoverage_receipts(prior, target)
    assert not allowed
    assert len(receipts) == 2
    assert receipts[0]["excluded"] is True
    assert receipts[1]["excluded"] is False
    assert "intersection_over_smaller_area" in receipts[1]["plausible_reasons"]


def test_target_noncoverage_accepts_disjoint_prior_boxes() -> None:
    target = _target("target", "person", [700, 700, 800, 900])
    prior = [
        {"description": "person", "coord_bins": [0, 0, 100, 100]},
        {"description": "chair", "coord_bins": [300, 300, 400, 500]},
    ]
    allowed, receipts = collector.target_noncoverage_receipts(prior, target)
    assert allowed
    assert all(receipt["excluded"] for receipt in receipts)


def test_target_noncoverage_rejects_invalid_prior_geometry() -> None:
    target = _target("target", "person", [700, 700, 800, 900])
    allowed, receipts = collector.target_noncoverage_receipts(
        [{"description": "person", "coord_bins": None}], target
    )
    assert not allowed
    assert receipts[0]["plausible"] is True
    assert "prior_prediction_has_no_valid_box" in receipts[0]["plausible_reasons"]


def test_match_target_requires_clear_same_category_best_match() -> None:
    targets = [
        _target("a", "person", [100, 100, 300, 400]),
        _target("b", "person", [110, 110, 310, 410]),
    ]
    assert collector.match_target(
        {"description": "person", "coord_bins": [105, 105, 305, 405]},
        targets,
        minimum_iou=0.5,
        minimum_margin=0.05,
    ) is None

    unique = collector.match_target(
        {"description": "person", "coord_bins": [100, 100, 295, 395]},
        [_target("a", "person", [100, 100, 300, 400]), _target("b", "person", [700, 700, 800, 900])],
        minimum_iou=0.5,
        minimum_margin=0.05,
    )
    assert unique is not None
    assert unique["owner_id"] == "a"
    assert unique["margin"] >= 0.05


def test_selected_sites_use_schema_description_and_coordinate_types() -> None:
    tokens = [151646, 42, 151647, 151648, 151670, 151671, 151672, 151673, 151649]
    sites = collector.selected_sites(tokens)
    assert [site["intended_token_type"] for site in sites] == [
        "schema", "desc_text", "schema", "schema", "coordinate", "coordinate", "coordinate", "coordinate", "schema"
    ]
    assert collector.description_end_interval(tokens) == 3


def test_released_suffix_budget_includes_prefix_and_forced_candidate() -> None:
    assert collector.released_suffix_row_count(16, 0) == 15
    assert collector.released_suffix_row_count(16, 4) == 11
    assert collector.released_suffix_row_count(1, 0) == 0
    with pytest.raises(ValueError):
        collector.released_suffix_row_count(0, 0)


def test_sampled_match_must_be_unique_against_full_physical_ledger() -> None:
    targets = [
        _target("a", "person", [100, 100, 300, 400]),
        _target("b", "person", [110, 110, 310, 410]),
    ]
    # A singleton target check would accept this row.  The full ledger check
    # correctly rejects it because the second same-class owner is nearly tied.
    assert collector.match_target(
        {"description": "person", "coord_bins": [105, 105, 305, 405]},
        targets,
        minimum_iou=0.5,
        minimum_margin=0.05,
    ) is None


def test_target_noncoverage_proof_has_assembler_shape() -> None:
    target = _target("target", "person", [700, 700, 800, 900])
    allowed, receipts = collector.target_noncoverage_receipts(
        [{"description": "person", "coord_bins": [0, 0, 100, 100]}],
        target,
        same_class_targets=[target],
    )
    assert allowed
    proof = collector.build_target_exclusion_review(
        target,
        receipts,
        proof_stratum="target_scoped_noncoverage",
        native_terminal_generated_step=12,
    )
    assert set(proof) == {
        "target_owner_id",
        "native_terminal_generated_step",
        "prior_row_count",
        "thresholds",
        "prior_row_exclusions",
    }
    assert proof["prior_row_exclusions"][0]["row_index"] == 0
    assert proof["prior_row_exclusions"][0]["plausible_target_association"] is False


def test_source_prompt_hash_parity_is_checked() -> None:
    prompt = [1, 2, 3]
    manifest = {
        "prompt_trace": [
            {
                "row_id": "image-a",
                "backend_executed_prompt_token_ids_sha256": collector.hash_prefix_token_ids(prompt),
            }
        ]
    }
    result = collector.validate_source_prompt_trace(manifest, "image-a", prompt)
    assert result["prompt_token_parity"] == "verified"
    with pytest.raises(ValueError):
        collector.validate_source_prompt_trace(manifest, "image-a", [1, 2, 4])


def test_collector_event_assembles_and_loads_state_bank(tmp_path: Path) -> None:
    checkpoint_identity = CheckpointIdentity(
        adapter_fingerprint="1" * 64,
        embedding_delta_fingerprint="2" * 64,
        base_config_sha256="3" * 64,
        tokenizer_sha256="4" * 64,
        token_identity_sha256="5" * 64,
        special_token_identity_sha256="6" * 64,
        processor_identity_sha256="7" * 64,
    )
    image_path = tmp_path / "image.bin"
    Image.new("RGB", (64, 64), color=(12, 34, 56)).save(image_path.with_suffix(".png"))
    image_path = image_path.with_suffix(".png")
    source_gt_path = tmp_path / "source.jsonl"
    source_gt_path.write_text("{}\n", encoding="utf-8")
    source_checkpoint_id = sha256_json(checkpoint_identity.to_artifact_dict())
    prompt_ids = [10, 151655, 151655, 11]
    prefix_ids = [151646, 17, 151647, 151648, 100, 100, 200, 200, 151649]
    positive_ids = [151646, 18, 151647, 151648, 300, 300, 500, 500, 151649]
    record = {
        "example_id": "synthetic-image-999",
        "source_row": {
            "gt": [
                _target("covered", "person", [100, 100, 200, 200]),
                _target("target", "person", [300, 300, 500, 500]),
            ]
        },
        "terminal": {
            "prefix_token_ids": prefix_ids,
            "terminal_generated_step_index": 9,
        },
        "prefix_object_row_count": 1,
        "prefix_row_matches": [
            {"status": "resolved", "owner_id": "covered", "iou": 1.0, "margin": 1.0}
        ],
        "covered_owner_ids": ["covered"],
    }
    target_choice = {
        "target": _target("target", "person", [300, 300, 500, 500]),
        "noncoverage_receipts": [],
        "proof_stratum": "full_resolved",
    }
    sampled = {
        "seed": 0,
        "raw_generated_token_ids": positive_ids,
        "raw_generated_text": "<|object_ref_start|>person<|object_ref_end|>",
        "target_match": {
            "owner_id": "target",
            "iou": 1.0,
            "second_iou": 0.0,
            "margin": 1.0,
            "prediction_box": [300, 300, 500, 500],
            "target_box": [300, 300, 500, 500],
        },
    }
    admission = {
        "row_budget": {"native": 16, "counterfactual": 16},
        "generated_token_budget": {"native": 512, "counterfactual": 512},
        "target_owner_retained": True,
        "verified_owner_delta": {"added_owner_ids": ["target"], "removed_owner_ids": []},
        "confirmed_new_duplicate_count": 0,
        "confirmed_new_malformed_count": 0,
        "confirmed_new_unsupported_entity_count": 0,
        "unknown_suffix_neutral": True,
        "unknown_suffix_provenance": {"policy": "synthetic test"},
    }
    rollout, review = collector._build_event(
        record,
        target_choice,
        sampled,
        prompt_ids=prompt_ids,
        image_pad_interval=[1, 3],
        image_id=999,
        image_path=image_path,
        image_width=64,
        image_height=64,
        source_gt_path=source_gt_path,
        checkpoint_id=source_checkpoint_id,
        temperature=0.4,
        top_p=0.95,
        repetition_penalty=1.0,
        counterfactual_admission=admission,
        split="train",
    )
    output_dir = tmp_path / "state-bank"
    prompt_identity = sha256_json(prompt_ids)
    manifest = assemble_state_bank(
        output_dir=output_dir,
        rollout_rows=[rollout],
        review_rows=[review],
        source_checkpoint=checkpoint_identity,
        prompt_identity_sha256=prompt_identity,
        source_artifacts=[{"artifact_id": "collector-test", "sha256": "a" * 64}],
    )
    loaded = load_state_bank(
        output_dir / "manifest.json",
        expected_source_checkpoint=checkpoint_identity,
        expected_prompt_identity_sha256=prompt_identity,
    )
    assert manifest.source_checkpoint_id == source_checkpoint_id
    assert len(loaded.records) == 1
    assert loaded.records[0].counterfactual_admission_present
