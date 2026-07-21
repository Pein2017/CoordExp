"""Focused contract tests for selected-route owner transfer receipts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.research.analyze_selected_route_added_owner_transfer import (
    _state_bank_image_ids,
    analyze_selected_route_added_owner_transfer,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _row(pred: list[dict[str, object]]) -> dict[str, object]:
    return {
        "row_id": "coco2017_train_000000000001",
        "image_path": "/tmp/000000000001.jpg",
        "image_width": 100,
        "image_height": 100,
        "gt": [
            {"object_id": "1", "description": "person", "bbox": [0, 0, 500, 500]},
            {"object_id": "2", "description": "cat", "bbox": [500, 500, 999, 999]},
            {"object_id": "3", "description": "dog", "bbox": [0, 500, 500, 999]},
        ],
        "pred": pred,
    }


def _person_prediction() -> dict[str, object]:
    return {"description": "person", "bbox": [0, 0, 50, 50], "coord_bins": [0, 0, 500, 500]}


def _cat_prediction() -> dict[str, object]:
    return {"description": "cat", "bbox": [50, 50, 100, 100], "coord_bins": [500, 500, 999, 999]}


def _dog_prediction() -> dict[str, object]:
    return {"description": "dog", "bbox": [0, 50, 50, 100], "coord_bins": [0, 500, 500, 999]}


def test_selected_route_transfer_receipt_tracks_scope_and_alignment(tmp_path: Path) -> None:
    source = tmp_path / "source" / "gt_vs_pred.jsonl"
    step10 = tmp_path / "step10" / "gt_vs_pred.jsonl"
    step15 = tmp_path / "step15" / "gt_vs_pred.jsonl"
    step16 = tmp_path / "step16" / "gt_vs_pred.jsonl"
    _write_jsonl(source, [_row([_person_prediction()])])
    _write_jsonl(step10, [_row([_cat_prediction()])])
    _write_jsonl(step15, [_row([_person_prediction(), _cat_prediction(), _dog_prediction()])])
    _write_jsonl(step16, [_row([_person_prediction(), _cat_prediction(), _dog_prediction()])])

    source_panel = tmp_path / "source-panel.json"
    source_panel.write_text("{}\n", encoding="utf-8")
    source_panel_hash = hashlib.sha256(source_panel.read_bytes()).hexdigest()

    route_greedy = tmp_path / "route" / "greedy-all256.json"
    shared_model = {
        "base": {"path": "base"},
        "adapter": {"adapter_path": "adapter"},
        "embedding_delta": {"identity": {"delta_path": "embedding-delta"}},
    }
    effective_settings = {"backend_options": {"hf": {"attn_implementation": "sdpa"}}, "batch_size": 1}
    _write_json(
        route_greedy,
        {
            "schema_version": "greedy.v1",
            "rollout_count": 1,
            "config": {
                "decode_mode": "greedy",
                "max_new_tokens": 512,
                "temperature": 0.0,
                "top_p": 1.0,
                "repetition_penalty": 1.0,
                "resolved_fingerprint": "route-fp",
            },
            "model_identity": {
                "generation_config_fingerprint": "generation-fp",
                "effective_settings": effective_settings,
                "model_identity": shared_model,
            },
            "rollouts": [
                {
                    "image_id": 1,
                    "predictions": {"predictions": [{"description": "person", "coord_bins": [0, 0, 500, 500]}]},
                }
            ],
        },
    )

    _write_json(
        source.parent / "run_manifest.json",
        {
            "backend": "hf",
            "backend_mode": "generate",
            "generation_policy": {
                "do_sample": False,
                "max_new_tokens": 512,
                "temperature": 0.0,
                "top_p": 1.0,
                "repetition_penalty": 1.0,
            },
            "generation_config_fingerprint": "generation-fp",
            "backend_session": {"effective_settings": effective_settings},
            "model_identity": shared_model,
            "resolved_config_fingerprints": {"infer_config": "clean-fp"},
        },
    )

    assembly = tmp_path / "assembly-receipt.json"
    _write_json(
        assembly,
        {
            "route_selection": [
                {
                    "image_id": 1,
                    "admissible": True,
                    "selected_route_id": "seed-1",
                    "selected_seed": 1,
                    "candidate_routes": {
                        "seed-1": {
                            "seed": 1,
                            "owner_ids": ["1:1", "1:2", "1:3"],
                            "greedy_owner_ids": ["1:1"],
                            "added_owner_ids": ["1:2", "1:3"],
                            "last_added_owner_row_index": 2,
                        }
                    },
                }
            ],
            "source_artifacts": [{"artifact_id": "trajectory-analysis", "sha256": source_panel_hash}],
        },
    )

    records = tmp_path / "state-bank" / "records.jsonl"
    _write_jsonl(
        records,
        [
            {
                "event_id": "event-1",
                "image": {"image_id": 1},
                "candidates": [{"physical_owner_id": "1:2", "role": "positive"}],
            }
        ],
    )
    records_hash = hashlib.sha256(records.read_bytes()).hexdigest()
    manifest = tmp_path / "state-bank" / "manifest.json"
    _write_json(manifest, {"record_count": 1, "records_sha256": records_hash})

    receipt = analyze_selected_route_added_owner_transfer(
        assembly_receipt_path=assembly,
        state_bank_manifest_path=manifest,
        state_bank_records_path=records,
        source_panel_path=source_panel,
        route_analysis_greedy_rollout_path=route_greedy,
        clean_rollout_paths={
            "source": source,
            "step10": step10,
            "step15": step15,
            "step16": step16,
        },
    )

    assert receipt["scope"]["state_bank_image_count"] == 1
    assert receipt["scope"]["selected_route_added_owner_count"] == 2
    assert receipt["scope"]["selected_route_added_direct_positive_event_target_owner_count"] == 1
    assert receipt["scope"]["selected_route_added_non_direct_positive_event_target_owner_count"] == 1
    assert receipt["scope"]["selected_route_ordinary_non_added_owner_count"] == 1
    assert receipt["arms"]["source"]["added"]["matched_owner_count"] == 0
    assert receipt["arms"]["source"]["added_direct_positive_event_target"]["owner_count"] == 1
    assert receipt["arms"]["source"]["added_direct_positive_event_target"]["matched_owner_count"] == 0
    assert receipt["arms"]["source"]["added_non_direct_positive_event_target"]["owner_count"] == 1
    assert receipt["arms"]["source"]["added_non_direct_positive_event_target"]["matched_owner_count"] == 0
    assert receipt["arms"]["step10"]["added"]["matched_owner_count"] == 1
    assert receipt["arms"]["step10"]["added_direct_positive_event_target"]["matched_owner_count"] == 1
    assert receipt["arms"]["step10"]["added_non_direct_positive_event_target"]["matched_owner_count"] == 0
    assert receipt["arms"]["step10"]["added"]["relative_to_source"]["gained_owner_count"] == 1
    assert receipt["arms"]["step10"]["added_direct_positive_event_target"]["relative_to_source"]["gained_owner_count"] == 1
    assert receipt["arms"]["step10"]["added_non_direct_positive_event_target"]["relative_to_source"]["gained_owner_count"] == 0
    assert receipt["arms"]["step10"]["added"]["relative_to_source"]["paired_image_cluster_bootstrap"] == {
        "method": "percentile bootstrap of the total matched-owner count change with image as the resampling cluster",
        "seed": 20260721,
        "replicate_count": 20_000,
        "confidence_level": 0.95,
        "lower_total_owner_change": 1,
        "upper_total_owner_change": 1,
        "positive_image_count": 1,
        "negative_image_count": 0,
        "zero_image_count": 0,
    }
    assert receipt["arms"]["step10"]["ordinary_non_added"]["relative_to_source"]["lost_owner_count"] == 1
    assert receipt["arms"]["step15"]["added"]["matched_owner_count"] == 2
    assert receipt["arms"]["step15"]["added_direct_positive_event_target"]["matched_owner_count"] == 1
    assert receipt["arms"]["step15"]["added_non_direct_positive_event_target"]["matched_owner_count"] == 1
    assert receipt["arms"]["step15"]["added_non_direct_positive_event_target"]["relative_to_source"]["gained_owner_count"] == 1
    assert receipt["route_owner_universe"]["selected_route_owner_union_count"] == 3
    assert receipt["route_owner_universe"]["matched_selected_route_owner_union_count_by_arm"] == {
        "source": 1,
        "step10": 1,
        "step15": 3,
        "step16": 3,
    }
    assert receipt["route_owner_universe"]["matched_selected_route_added_direct_positive_event_target_owner_count_by_arm"] == {
        "source": 0,
        "step10": 1,
        "step15": 1,
        "step16": 1,
    }
    assert receipt["route_owner_universe"]["matched_selected_route_added_non_direct_positive_event_target_owner_count_by_arm"] == {
        "source": 0,
        "step10": 0,
        "step15": 1,
        "step16": 1,
    }
    admitted = receipt["all_annotated_owner_coverage_by_admission_scope"]
    assert admitted["source"]["admitted_state_bank_118"]["gt_owner_count"] == 3
    assert admitted["source"]["admitted_state_bank_118"]["matched_owner_count"] == 1
    assert admitted["source"]["non_admitted_138"]["gt_owner_count"] == 0
    alignment = receipt["source_artifact_alignment"]
    assert alignment["identity_comparison"]["stable_effective_backend_settings_equal"] is True
    assert alignment["identity_comparison"]["resolved_config_fingerprint_equal"] is False
    assert alignment["decoded_output_comparison"]["exact_parsed_prediction_rows"] == 1
    assert receipt["per_image"]["1"]["added_direct_positive_event_target_owner_ids"] == ["1:2"]
    assert receipt["per_image"]["1"]["added_non_direct_positive_event_target_owner_ids"] == ["1:3"]


def test_state_bank_positive_target_partition_uses_stable_physical_owner_ids(tmp_path: Path) -> None:
    records = tmp_path / "records.jsonl"
    _write_jsonl(
        records,
        [
            {
                "event_id": "event-1",
                "image": {"image_id": 1},
                "candidates": [
                    {"physical_owner_id": "1:positive", "role": "positive"},
                    {"physical_owner_id": "1:negative", "role": "negative"},
                    {"physical_owner_id": "1:legacy-no-role"},
                ],
            }
        ],
    )
    records_hash = hashlib.sha256(records.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    _write_json(manifest, {"record_count": 1, "records_sha256": records_hash})

    (
        image_ids,
        record_count,
        all_event_owner_ids,
        positive_by_image,
    ) = _state_bank_image_ids(records, json.loads(manifest.read_text()))

    assert image_ids == {"1"}
    assert record_count == 1
    assert all_event_owner_ids == {"1:positive", "1:negative", "1:legacy-no-role"}
    assert positive_by_image == {"1": {"1:positive", "1:legacy-no-role"}}
