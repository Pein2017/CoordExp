from __future__ import annotations

import json
from pathlib import Path

from scripts.research import build_image2299_current_native_ledger as subject
from scripts.research import build_sorted_owner_basin_census as census
from scripts.research import build_sorted_owner_accessibility_census_plan as legacy
from src.config.inference import load_infer_config


RANDOM_CONFIG = Path(
    "configs/coordexp_swift/infer/"
    "qwen3_vl_2b_desc_first_random_step4887_human_refined13_hf_fp32_rp1p0.yaml"
)
PANEL = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-08-04-sorted-prospective-13-image-panel-admission/"
    "evaluation-inputs/human-refined-13.coord.jsonl"
)
PANEL_RECEIPT = PANEL.parents[1] / "receipt.json"


def _write_current_rollout(path: Path) -> None:
    resolved = load_infer_config(RANDOM_CONFIG)
    config = resolved.config
    _, owners_by_image, _ = census._load_panel(PANEL)
    owner = owners_by_image[subject.TARGET_IMAGE_ID][0]
    prompt_ids = [11, 12, 13]
    generated_ids = [151646, 8987, 151647, 151648, 151670, 151671, 151672, 151673, 151649]
    example_id = "coco2017_val_000000002299"
    identity = {
        "backend": "hf",
        "model_identity": {
            "base": {"path": str(config.model.base_model)},
            "adapter": {"adapter_path": str(config.adapter.path)},
            "embedding_delta": {
                "identity": {"delta_path": str(config.embedding_delta.path)}
            },
        },
        "tokenizer_identity": {"synthetic": True},
        "processor_identity": {"synthetic": True},
        "generation_config_fingerprint": "synthetic",
        "execution_model_identity": {"synthetic": True},
        "likelihood_semantics": {"policy": "synthetic"},
    }
    payload = {
        "schema_version": census.ROLLOUT_SCHEMA_VERSION,
        "experiment_mode": "experiment_local_sampled_counterfactual_to_canonical_deterministic_backend",
        "config": {
            "decode_mode": "greedy",
            "device": "cuda:0",
            "image_ids": [example_id],
            "infer_config_path": str(RANDOM_CONFIG.resolve()),
            "max_new_tokens": 3084,
            "model_dtype": "fp32",
            "repetition_penalty": 1.0,
            "resolved_fingerprint": resolved.fingerprint,
            "sampling_is_not_infer_config": True,
            "seeds": [0],
            "temperature": 0.0,
            "top_p": 1.0,
        },
        "model_identity": identity,
        "prompt_metadata": {
            example_id: {
                "prompt_token_ids": prompt_ids,
                "chat_text_sha256": "a" * 64,
                "image_sha256": subject.EXPECTED_IMAGE_SHA256,
            }
        },
        "replay_check": {"enabled": False, "status": "not_requested", "checked_images": []},
        "rollout_count": 1,
        "rollouts": [
            {
                "image_id": 2299,
                "example_id": example_id,
                "seed": 0,
                "decode_mode": "greedy",
                "generated_token_ids": generated_ids,
                "generated_token_ids_sha256": legacy.sha256_json(generated_ids),
                "generated_text": "synthetic",
                "stop_reason": "im_end",
                "prompt_token_ids": prompt_ids,
                "prompt_token_ids_sha256": legacy.sha256_json(prompt_ids),
                "observed_image_grid_thw": [1, 46, 76],
                "executed_media_sha256": subject.EXPECTED_EXECUTED_MEDIA_SHA256,
                "predictions": {
                    "parse_status": "accepted",
                    "metric_bearing": True,
                    "valid_prediction_count": 1,
                    "dropped_prediction_count": 0,
                    "dropped_predictions": [],
                    "predictions": [
                        {
                            "generated_order": 0,
                            "description": owner["normalized_description"],
                            "bbox": list(owner["bbox_xyxy"]),
                            "object_span_id": "span:0",
                            "raw_span_sha256": legacy.sha256_json(generated_ids),
                        }
                    ],
                },
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_builds_random_current_native_ledgers(tmp_path: Path) -> None:
    rollout = tmp_path / "greedy.json"
    _write_current_rollout(rollout)
    owner_rows, prediction_rows, greedy, receipt = subject.build_artifacts(
        panel_path=PANEL,
        panel_receipt_path=PANEL_RECEIPT,
        rollout_path=rollout,
        infer_config_path=RANDOM_CONFIG,
        checkpoint_role="random_step4887",
    )

    assert len(owner_rows) == 46
    assert len(prediction_rows) == 1
    assert prediction_rows[0]["pred_row_id"].startswith("pred:random:")
    assert prediction_rows[0]["strict_match_status"] == "matched"
    assert receipt["counts"] == {
        "owner_count": 46,
        "prediction_row_count": 1,
        "strict_matched_owner_count": 1,
        "strict_false_negative_owner_count": 45,
        "strict_unmatched_prediction_count": 0,
        "ambiguity_neutral_owner_count": 0,
    }
    assert greedy["rollouts"][0]["stop_reason"] == "im_end"

    output = tmp_path / "s0"
    subject.commit_artifacts(owner_rows, prediction_rows, greedy, receipt, output)
    subject.commit_artifacts(owner_rows, prediction_rows, greedy, receipt, output)
    assert sorted(path.name for path in output.iterdir()) == [
        "greedy.json",
        "owner-ledger.jsonl",
        "prediction-row-ledger.jsonl",
        "receipt.json",
    ]
