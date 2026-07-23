from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from scripts.research.analyze_source_b16_treatment_owner_ledger import (
    OwnerLedgerError,
    analyze_source_b16_treatments,
)


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prediction(index: int, category: str, box: list[float]) -> dict[str, Any]:
    return {
        "description": category,
        "generated_order": index,
        "bbox": box,
        "object_span_id": f"source-b16:span-{index}",
    }


def _model_identity(
    payload_name: str,
    *,
    base_fingerprint: str = "base-model",
) -> dict[str, Any]:
    semantic = {
        "base_config_sha256": "base-config",
        "tokenizer_sha256": "tokenizer-files",
        "semantics": "additive_delta",
        "tensor_dtype": "float32",
        "tensor_key": "shared_embed_delta",
        "tensor_shape": [2, 4],
        "tie_word_embeddings": True,
        "token_ids": [101, 102],
        "token_strings": ["<|object_ref_start|>", "<|object_ref_end|>"],
    }
    execution = {
        "composition_key": f"composition-{payload_name}",
        "snapshot_fingerprint": f"snapshot-{payload_name}",
        "receipt_fingerprint": f"receipt-{payload_name}",
        "source_identity": {
            "base": {"fingerprint": base_fingerprint, "version": "base-v1"},
            "adapter": {"fingerprint": f"adapter-{payload_name}"},
            "embedding_delta": {
                "fingerprint": f"embedding-{payload_name}",
                "semantic_identity": semantic,
            },
        },
    }
    return {
        "execution_model_identity": execution,
        "generation_config_fingerprint": "generation-policy",
        "tokenizer_identity": {
            "required_token_count": 2,
            "wrapper_token_ids": {
                "<|object_ref_start|>": 101,
                "<|object_ref_end|>": 102,
            },
        },
        "processor_identity": {
            "processor_class": "SyntheticProcessor",
            "patch_size": 16,
        },
    }


def _rollout(
    image_id: int,
    predictions: list[dict[str, Any]],
    *,
    status: str = "accepted_natural_end",
    prompt_variant: int = 0,
    stop_reason: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    example_id = f"example-{image_id}"
    prompt_token_ids = [image_id, 700 + prompt_variant]
    prompt_hash = _json_sha256(prompt_token_ids)
    source_image_hash = f"source-image-{image_id}"
    projected_token_ids = [image_id, len(predictions)]
    projected_parser = {
        "parse_status": "accepted" if predictions else "empty",
        "valid_prediction_count": len(predictions),
        "dropped_prediction_count": 0,
        "dropped_predictions": [],
        "predictions": predictions,
    }
    failure_parser = None
    raw_parser = dict(projected_parser)
    if status == "failed_invalid_before_budget":
        dropped_predictions = [
            {"generated_order": len(predictions), "reason": "geometry_invalid"}
        ]
        failure_parser = {
            "parse_status": "accepted_with_drops",
            "valid_prediction_count": len(predictions),
            "dropped_prediction_count": len(dropped_predictions),
            "dropped_predictions": dropped_predictions,
            "predictions": predictions,
        }
        raw_parser = dict(failure_parser)
    if stop_reason is None:
        stop_reason = (
            "length" if status == "failed_token_limit_before_budget" else "im_end"
        )
    generated_token_ids = [image_id, 800]
    row = {
        "image_id": image_id,
        "example_id": example_id,
        "trajectory_id": "source-b16",
        "decode_mode": "source_b16",
        "stop_reason": stop_reason,
        "prompt_token_ids": prompt_token_ids,
        "prompt_token_ids_sha256": prompt_hash,
        "source_image_file_sha256": source_image_hash,
        "executed_rgb_sha256": f"executed-rgb-{image_id}",
        "image_width": 100,
        "image_height": 100,
        "generated_token_ids": generated_token_ids,
        "generated_token_ids_sha256": _json_sha256(generated_token_ids),
        # Deliberately unrelated raw parser evidence: the analyzer must use
        # only source_b16.projected_parser_evidence below.
        "predictions": {
            "parse_status": "accepted",
            "valid_prediction_count": 1,
            "predictions": [_prediction(99, "raw-only", [1, 1, 99, 99])],
        },
        "source_b16": {
            "status": status,
            "row_budget": 16,
            "raw_valid_complete_row_count": len(predictions),
            "projected_valid_complete_row_count": len(predictions),
            "projected_token_end_offset_exclusive": len(projected_token_ids),
            "projected_token_ids": projected_token_ids,
            "projected_token_ids_sha256": _json_sha256(projected_token_ids),
            "projected_text": f"projected-{image_id}",
            "projected_parser_evidence": projected_parser,
            "token_limit_before_budget": stop_reason == "length"
            and len(predictions) < 16,
            "natural_end_before_budget": stop_reason == "im_end"
            and len(predictions) < 16,
            "raw_parser_evidence": raw_parser,
        },
    }
    if failure_parser is not None:
        row["source_b16"]["failure_parser_evidence"] = failure_parser
    metadata = {
        "prompt_token_ids": prompt_token_ids,
        "prompt_token_ids_sha256": prompt_hash,
        "source_image_file_sha256": source_image_hash,
        "width": 100,
        "height": 100,
    }
    return row, metadata


def _status_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = Counter(row["source_b16"]["status"] for row in rows)
    ineligible = [
        row["image_id"]
        for row in rows
        if row["source_b16"]["status"]
        in {"failed_invalid_before_budget", "failed_token_limit_before_budget"}
    ]
    return {
        "status_counts": dict(sorted(statuses.items())),
        "accepted_count": sum(
            statuses[status] for status in ("accepted_budget", "accepted_natural_end")
        ),
        "ineligible_count": len(ineligible),
        "ineligible_image_ids": ineligible,
    }


def _generation_health(rows: list[dict[str, Any]]) -> dict[str, Any]:
    stop_counts = Counter(row["stop_reason"] for row in rows)
    return {
        "completion_count": len(rows),
        "generated_token_count": sum(len(row["generated_token_ids"]) for row in rows),
        "elapsed_seconds": 1.0,
        "stop_reason_counts": {
            "im_end": stop_counts["im_end"],
            "length": stop_counts["length"],
        },
        "natural_closure_count": stop_counts["im_end"],
        "parser_status_counts": {},
    }


def _write_panel(
    root: Path,
    specs: list[dict[str, Any]],
    *,
    payload_name: str,
    prompt_variant: int = 0,
    base_fingerprint: str = "base-model",
    config_overrides: dict[str, Any] | None = None,
) -> Path:
    worker = root / "worker-00-of-01"
    worker.mkdir(parents=True)
    model_identity = _model_identity(
        payload_name,
        base_fingerprint=base_fingerprint,
    )
    batches: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    for batch_index, spec in enumerate(specs):
        row, metadata = _rollout(
            int(spec["image_id"]),
            list(spec.get("predictions", [])),
            status=str(spec.get("status", "accepted_natural_end")),
            prompt_variant=prompt_variant,
            stop_reason=spec.get("stop_reason"),
        )
        rows = [row]
        config = {
            "infer_config_path": f"/synthetic/{payload_name}.yaml",
            "resolved_fingerprint": f"config-{payload_name}",
            "model_dtype": "bf16",
            "backend": "vllm",
            "decode_mode": "source_b16",
            "panel_mode": "source_b16",
            "temperature": 0.0,
            "top_p": 1.0,
            "repetition_penalty": 1.0,
            "max_new_tokens": 2048,
            "sample_count": 1,
            "sample_index_range": None,
            "sampling_is_not_infer_config": True,
            "sampling_order": "request_major",
            "worker_index": 0,
            "worker_count": 1,
            "image_batch_size": 16,
            "max_num_seqs": 32,
            "source_b16_row_budget": 16,
        }
        config.update(config_overrides or {})
        artifact = {
            "schema_version": "coordexp_vllm_trajectory_panel.v2",
            "experiment_mode": "experiment_local_vllm_trajectory_panel",
            "config": config,
            "model_identity": model_identity,
            "prompt_metadata": {row["example_id"]: metadata},
            "rollout_count": 1,
            "rollouts": rows,
        }
        artifact_path = worker / f"source_b16-batch-{batch_index:05d}.json"
        artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
        batch_summary = _status_summary(rows)
        batches.append(
            {
                "batch_index": batch_index,
                "image_ids": [row["image_id"]],
                "artifacts": {
                    "source_b16": {
                        "path": artifact_path.name,
                        "sha256": _file_sha256(artifact_path),
                    }
                },
                "generation_health": {"source_b16": _generation_health(rows)},
                "source_b16": batch_summary,
            }
        )
        all_rows.extend(rows)
    source_summary = _status_summary(all_rows)
    stop_counts = Counter(row["stop_reason"] for row in all_rows)
    manifest = {
        "schema_version": "vllm_trajectory_panel_worker_manifest.v1",
        "worker_index": 0,
        "worker_count": 1,
        "image_count": len(all_rows),
        "image_batch_size": 16,
        "max_num_seqs": 32,
        "panel_mode": "source_b16",
        "decode_modes": ["source_b16"],
        "source_b16_row_budget": 16,
        "completed_image_count": len(all_rows),
        "status": (
            "completed_with_source_b16_ineligible"
            if source_summary["ineligible_count"]
            else "completed"
        ),
        "batches": batches,
        "source_b16": source_summary,
        "completion_health": {
            "stop_reason_counts": {
                "im_end": stop_counts["im_end"],
                "length": stop_counts["length"],
            },
            "natural_closure_count": stop_counts["im_end"],
            "parser_status_counts": {},
        },
    }
    (worker / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def _write_candidate_jsonl(
    path: Path,
    rows: list[dict[str, Any]],
) -> Path:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _candidate(
    image_id: int,
    objects: list[tuple[str, str, list[float]]],
) -> dict[str, Any]:
    return {
        "image_id": image_id,
        "width": 100,
        "height": 100,
        "objects": [
            {"owner_id": owner_id, "category": category, "bbox": box}
            for owner_id, category, box in objects
        ],
    }


def _extra_panel_spec() -> dict[str, Any]:
    return {"image_id": 99, "predictions": []}


def test_reports_retained_lost_gained_and_uses_candidate_as_analysis_filter(
    tmp_path: Path,
) -> None:
    candidate = _write_candidate_jsonl(
        tmp_path / "development.jsonl",
        [
            _candidate(
                1,
                [
                    ("retained", "traffic_light", [0, 0, 10, 10]),
                    ("lost", "dog", [20, 0, 30, 10]),
                    ("gained", "bird", [40, 0, 50, 10]),
                ],
            )
        ],
    )
    source = _write_panel(
        tmp_path / "source",
        [
            {
                "image_id": 1,
                "predictions": [
                    _prediction(0, "traffic light", [0, 0, 10, 10]),
                    _prediction(1, "dog", [20, 0, 30, 10]),
                ],
            },
            _extra_panel_spec(),
        ],
        payload_name="source",
    )
    treatment = _write_panel(
        tmp_path / "treatment",
        [
            {
                "image_id": 1,
                "predictions": [
                    _prediction(0, "traffic_light", [0, 0, 10, 10]),
                    _prediction(1, "bird", [40, 0, 50, 10]),
                    _prediction(2, "bird", [40, 0, 50, 10]),
                    _prediction(3, "boat", [70, 0, 80, 10]),
                ],
            },
            _extra_panel_spec(),
        ],
        payload_name="treatment",
    )
    result = analyze_source_b16_treatments(
        candidate_jsonl=candidate,
        source_panel_root=source,
        treatment_panel_roots={"broad_seed_19": treatment},
        require_full_panel=False,
    )

    arm = result["treatments"]["broad_seed_19"]
    assert result["inputs"]["source_panel"]["full_panel_image_count"] == 2
    assert arm["panel"]["full_panel_image_count"] == 2
    assert [row["image_id"] for row in arm["per_image"]] == ["1"]
    assert (
        result["inputs"]["source_panel"]["model_payload_identity"]
        != arm["panel"]["model_payload_identity"]
    )
    for threshold in ("0.30", "0.50"):
        comparison = arm["per_image"][0]["owner_comparison"][
            "by_intersection_over_union"
        ][threshold]
        assert comparison["retained_source_owner_ids"] == ["1:retained"]
        assert comparison["lost_source_owner_ids"] == ["1:lost"]
        assert comparison["gained_annotated_owner_ids"] == ["1:gained"]
        assert comparison["net_owner_delta"] == 0
        treatment_matching = arm["per_image"][0]["treatment"][
            "matching_by_intersection_over_union"
        ][threshold]
        assert treatment_matching["duplicate_prediction_count"] == 1
        assert treatment_matching["review_needed_prediction_count"] == 2
        aggregate = arm["aggregate"]["owner_comparison_by_intersection_over_union"][
            threshold
        ]
        assert aggregate["retained_source_owner_ids"] == ["1:retained"]
        assert aggregate["lost_source_owner_ids"] == ["1:lost"]
        assert aggregate["gained_annotated_owner_ids"] == ["1:gained"]


def test_source_ineligibility_is_excluded_without_becoming_empty(
    tmp_path: Path,
) -> None:
    candidate = _write_candidate_jsonl(
        tmp_path / "heldout.jsonl",
        [_candidate(1, [("owner", "cat", [0, 0, 10, 10])])],
    )
    source = _write_panel(
        tmp_path / "source",
        [
            {
                "image_id": 1,
                "predictions": [_prediction(0, "cat", [0, 0, 10, 10])],
                "status": "failed_token_limit_before_budget",
            },
            _extra_panel_spec(),
        ],
        payload_name="source",
    )
    treatment = _write_panel(
        tmp_path / "treatment",
        [
            {
                "image_id": 1,
                "predictions": [_prediction(0, "cat", [0, 0, 10, 10])],
            },
            _extra_panel_spec(),
        ],
        payload_name="treatment",
    )

    result = analyze_source_b16_treatments(
        candidate_jsonl=candidate,
        source_panel_root=source,
        treatment_panel_roots={"broad": treatment},
        require_full_panel=False,
    )["treatments"]["broad"]

    image = result["per_image"][0]
    assert image["source"]["matching_by_intersection_over_union"] is None
    assert image["owner_comparison"] == {
        "eligible": False,
        "exclusion_reason": "source_ineligible",
        "by_intersection_over_union": None,
    }
    assert result["aggregate"]["joined_owner_comparison_eligible_image_count"] == 0
    assert (
        result["aggregate"]["source_ineligible_excluded_from_owner_denominator_count"]
        == 1
    )
    assert (
        result["aggregate"]["source_panel_health"]["token_limit_before_budget_count"]
        == 1
    )
    assert (
        result["aggregate"]["owner_comparison_by_intersection_over_union"]["0.50"][
            "lost_source_owner_ids"
        ]
        == []
    )


def test_treatment_ineligibility_is_health_not_owner_loss(tmp_path: Path) -> None:
    candidate = _write_candidate_jsonl(
        tmp_path / "development.jsonl",
        [_candidate(1, [("owner", "cat", [0, 0, 10, 10])])],
    )
    valid_spec = {
        "image_id": 1,
        "predictions": [_prediction(0, "cat", [0, 0, 10, 10])],
    }
    source = _write_panel(
        tmp_path / "source",
        [valid_spec, _extra_panel_spec()],
        payload_name="source",
    )
    treatment = _write_panel(
        tmp_path / "treatment",
        [
            {**valid_spec, "status": "failed_invalid_before_budget"},
            _extra_panel_spec(),
        ],
        payload_name="treatment",
    )
    treatment_artifact = json.loads(
        next(treatment.glob("worker-*/source_b16-batch-00000.json")).read_text(
            encoding="utf-8"
        )
    )
    treatment_receipt = treatment_artifact["rollouts"][0]["source_b16"]
    assert (
        treatment_receipt["projected_parser_evidence"]["dropped_prediction_count"] == 0
    )
    assert treatment_receipt["failure_parser_evidence"]["dropped_prediction_count"] == 1

    result = analyze_source_b16_treatments(
        candidate_jsonl=candidate,
        source_panel_root=source,
        treatment_panel_roots={"concentrated": treatment},
        require_full_panel=False,
    )["treatments"]["concentrated"]

    image = result["per_image"][0]
    assert image["owner_comparison"]["exclusion_reason"] == "treatment_ineligible"
    assert image["treatment"]["matching_by_intersection_over_union"] is None
    health = result["aggregate"]["treatment_panel_health"]
    assert health["invalid_before_budget_count"] == 1
    assert health["dropped_prediction_count"] == 1
    assert health["malformed_row_count"] == 1
    assert (
        result["aggregate"]["owner_comparison_by_intersection_over_union"]["0.30"][
            "lost_source_owner_count"
        ]
        == 0
    )


def test_degenerate_projected_box_stays_review_needed_health(
    tmp_path: Path,
) -> None:
    candidate = _write_candidate_jsonl(
        tmp_path / "development.jsonl",
        [_candidate(1, [("owner", "book", [0, 0, 10, 10])])],
    )
    panel_spec = {
        "image_id": 1,
        "predictions": [
            _prediction(0, "book", [0, 0, 10, 10]),
            _prediction(1, "book", [20, 20, 30, 20]),
        ],
    }
    source = _write_panel(
        tmp_path / "source",
        [panel_spec, _extra_panel_spec()],
        payload_name="source",
    )
    treatment = _write_panel(
        tmp_path / "treatment",
        [panel_spec, _extra_panel_spec()],
        payload_name="treatment",
    )

    result = analyze_source_b16_treatments(
        candidate_jsonl=candidate,
        source_panel_root=source,
        treatment_panel_roots={"replay": treatment},
        require_full_panel=False,
    )["treatments"]["replay"]

    source_image = result["per_image"][0]["source"]
    assert source_image["health"]["prediction_count"] == 2
    assert source_image["health"]["owner_matching_eligible_prediction_count"] == 1
    assert source_image["health"]["owner_matching_ineligible_prediction_count"] == 1
    for threshold in ("0.30", "0.50"):
        matching = source_image["matching_by_intersection_over_union"][threshold]
        assert matching["matched_prediction_count"] == 1
        assert matching["review_needed_prediction_ids"] == ["source-b16:span-1"]
        assert matching["owner_matching_ineligible_prediction_count"] == 1


def test_accepted_natural_end_with_length_stop_is_rejected(tmp_path: Path) -> None:
    candidate = _write_candidate_jsonl(
        tmp_path / "development.jsonl",
        [_candidate(1, [("owner", "cat", [0, 0, 10, 10])])],
    )
    contradictory_spec = {
        "image_id": 1,
        "predictions": [_prediction(0, "cat", [0, 0, 10, 10])],
        "status": "accepted_natural_end",
        "stop_reason": "length",
    }
    source = _write_panel(
        tmp_path / "source",
        [contradictory_spec, _extra_panel_spec()],
        payload_name="source",
    )
    treatment = _write_panel(
        tmp_path / "treatment",
        [contradictory_spec, _extra_panel_spec()],
        payload_name="treatment",
    )

    with pytest.raises(OwnerLedgerError, match="lacks im_end stop"):
        analyze_source_b16_treatments(
            candidate_jsonl=candidate,
            source_panel_root=source,
            treatment_panel_roots={"contradictory": treatment},
            require_full_panel=False,
        )


@pytest.mark.parametrize(
    ("field", "drifted_value"),
    (("backend", "not-vllm"), ("model_dtype", "float32")),
)
def test_execution_config_drift_is_rejected(
    tmp_path: Path,
    field: str,
    drifted_value: str,
) -> None:
    candidate = _write_candidate_jsonl(
        tmp_path / "development.jsonl",
        [_candidate(1, [("owner", "cat", [0, 0, 10, 10])])],
    )
    specs = [
        {
            "image_id": 1,
            "predictions": [_prediction(0, "cat", [0, 0, 10, 10])],
        },
        _extra_panel_spec(),
    ]
    source = _write_panel(
        tmp_path / "source",
        specs,
        payload_name="source",
    )
    treatment = _write_panel(
        tmp_path / "treatment",
        specs,
        payload_name="treatment",
        config_overrides={field: drifted_value},
    )

    with pytest.raises(OwnerLedgerError, match="non-canonical Source@B16 config"):
        analyze_source_b16_treatments(
            candidate_jsonl=candidate,
            source_panel_root=source,
            treatment_panel_roots={"drifted": treatment},
            require_full_panel=False,
        )


def test_repetition_penalty_1p10_requires_a_matched_source_treatment_contract(
    tmp_path: Path,
) -> None:
    candidate = _write_candidate_jsonl(
        tmp_path / "development.jsonl",
        [_candidate(1, [("owner", "cat", [0, 0, 10, 10])])],
    )
    specs = [
        {
            "image_id": 1,
            "predictions": [_prediction(0, "cat", [0, 0, 10, 10])],
        },
        _extra_panel_spec(),
    ]
    source = _write_panel(
        tmp_path / "source-rp1p10",
        specs,
        payload_name="source-rp1p10",
        config_overrides={"repetition_penalty": 1.1},
    )
    matched = _write_panel(
        tmp_path / "matched-rp1p10",
        specs,
        payload_name="matched-rp1p10",
        config_overrides={"repetition_penalty": 1.1},
    )
    result = analyze_source_b16_treatments(
        candidate_jsonl=candidate,
        source_panel_root=source,
        treatment_panel_roots={"matched": matched},
        require_full_panel=False,
    )
    assert (
        result["inputs"]["source_panel"]["config_identity"]["repetition_penalty"]
        == 1.1
    )

    neutral = _write_panel(
        tmp_path / "neutral-rp1p00",
        specs,
        payload_name="neutral-rp1p00",
    )
    with pytest.raises(OwnerLedgerError, match="execution contract mismatch"):
        analyze_source_b16_treatments(
            candidate_jsonl=candidate,
            source_panel_root=source,
            treatment_panel_roots={"mismatched": neutral},
            require_full_panel=False,
        )


def test_full_panel_cohort_mismatch_fails_even_when_analysis_subset_is_present(
    tmp_path: Path,
) -> None:
    candidate = _write_candidate_jsonl(
        tmp_path / "heldout.jsonl",
        [_candidate(1, [("owner", "cat", [0, 0, 10, 10])])],
    )
    spec = {
        "image_id": 1,
        "predictions": [_prediction(0, "cat", [0, 0, 10, 10])],
    }
    source = _write_panel(
        tmp_path / "source",
        [spec, _extra_panel_spec()],
        payload_name="source",
    )
    treatment = _write_panel(
        tmp_path / "treatment",
        [spec],
        payload_name="treatment",
    )

    with pytest.raises(OwnerLedgerError, match="full-panel cohort mismatch"):
        analyze_source_b16_treatments(
            candidate_jsonl=candidate,
            source_panel_root=source,
            treatment_panel_roots={"broad": treatment},
            require_full_panel=False,
        )


def test_prompt_and_model_invariant_drift_fail_closed(tmp_path: Path) -> None:
    candidate = _write_candidate_jsonl(
        tmp_path / "development.jsonl",
        [_candidate(1, [("owner", "cat", [0, 0, 10, 10])])],
    )
    specs = [
        {
            "image_id": 1,
            "predictions": [_prediction(0, "cat", [0, 0, 10, 10])],
        },
        _extra_panel_spec(),
    ]
    source = _write_panel(
        tmp_path / "source",
        specs,
        payload_name="source",
    )
    prompt_drift = _write_panel(
        tmp_path / "prompt-drift",
        specs,
        payload_name="prompt-drift",
        prompt_variant=1,
    )
    with pytest.raises(OwnerLedgerError, match="image or prompt mismatch"):
        analyze_source_b16_treatments(
            candidate_jsonl=candidate,
            source_panel_root=source,
            treatment_panel_roots={"prompt_drift": prompt_drift},
            require_full_panel=False,
        )

    invariant_drift = _write_panel(
        tmp_path / "invariant-drift",
        specs,
        payload_name="invariant-drift",
        base_fingerprint="different-base-model",
    )
    with pytest.raises(OwnerLedgerError, match="model invariant mismatch"):
        analyze_source_b16_treatments(
            candidate_jsonl=candidate,
            source_panel_root=source,
            treatment_panel_roots={"invariant_drift": invariant_drift},
            require_full_panel=False,
        )
