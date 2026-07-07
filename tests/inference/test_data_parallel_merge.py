from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from src.common.errors import ArtifactContractError
from src.eval.detection_consumer import evaluate_scored_detection_artifacts
from src.inference.artifacts import (
    IMAGE_PLAN_NAME,
    MANIFEST_NAME,
    PARSE_DIAGNOSTICS_NAME,
    PROVENANCE_NAME,
    RAW_NAME,
    SCORED_NAME,
    SUMMARY_NAME,
    TOKEN_TRACE_NAME,
    recompute_scores_from_artifacts,
    sha256_file,
    validate_scored_artifact_set,
    write_inference_artifacts,
)
from src.inference.backend import DecodeResult, TokenTrace
from src.inference.data_parallel import DataParallelPlan, RankShardPlan, plan_data_parallel_shards
from src.inference.parsing import parse_compact_object_box_closed
from src.inference.scoring import SCORE_POLICY_FINGERPRINT


OBJECT_TEXT = (
    "<|object_ref_start|>cat<|object_ref_end|>"
    "<|box_start|><|coord_100|><|coord_200|><|coord_300|><|coord_400|><|box_end|>"
)
REQUIRED_SHARD_ARTIFACTS = {
    RAW_NAME,
    SCORED_NAME,
    PROVENANCE_NAME,
    TOKEN_TRACE_NAME,
    PARSE_DIAGNOSTICS_NAME,
    IMAGE_PLAN_NAME,
    SUMMARY_NAME,
    MANIFEST_NAME,
}


def test_rank_local_shard_dirs_contain_complete_scored_artifact_family(tmp_path: Path) -> None:
    plan = _plan(("row-0", "row-1", "row-2", "row-3"))
    shard_dirs = _write_plan_shards(tmp_path, plan)

    assert [path.relative_to(tmp_path).as_posix() for path in shard_dirs] == [
        "shards/rank-000",
        "shards/rank-001",
    ]
    for shard_dir in shard_dirs:
        assert REQUIRED_SHARD_ARTIFACTS.issubset({path.name for path in shard_dir.iterdir()})
        validate_scored_artifact_set(shard_dir)


def test_strict_merge_restores_original_order_and_regenerates_bound_provenance(tmp_path: Path) -> None:
    from src.inference.merge import merge_shard_artifacts

    row_ids = ("row-0", "row-1", "row-2", "row-3")
    plan = _plan(row_ids)
    shard_dirs = _write_plan_shards(tmp_path, plan)

    paths = merge_shard_artifacts(
        output_dir=tmp_path,
        shard_dirs=tuple(reversed(shard_dirs)),
        expected_row_ids=row_ids,
        metadata=_merge_metadata(plan),
        plan=plan,
    )

    raw_rows = _read_jsonl(paths.raw_jsonl)
    scored_rows = _read_jsonl(paths.scored_jsonl)
    image_plan_rows = _read_jsonl(paths.image_plan_jsonl)
    token_trace_rows = _read_jsonl(paths.token_trace_jsonl)
    diagnostic_rows = _read_jsonl(paths.parse_diagnostics_jsonl)
    provenance = _read_json(paths.provenance_json)
    manifest = _read_json(paths.run_manifest_json)

    assert [row["row_id"] for row in raw_rows] == list(row_ids)
    assert [row["row_id"] for row in scored_rows] == list(row_ids)
    assert [row["row_id"] for row in image_plan_rows] == list(row_ids)
    assert [row["row_index"] for row in raw_rows] == [0, 1, 2, 3]
    assert all("rank" in row for row in token_trace_rows)
    assert all("assigned_parent_visible_device_token" in row for row in token_trace_rows)
    assert {row["rank"] for row in diagnostic_rows} == {0, 1}
    assert "rank" not in scored_rows[0]

    assert provenance["raw_artifact"]["path"] == RAW_NAME
    assert provenance["scored_artifact"]["path"] == SCORED_NAME
    assert provenance["score_policy_fingerprint"] == SCORE_POLICY_FINGERPRINT
    assert provenance["model_identity_fingerprint"] == "model-fp"
    assert provenance["processor_identity_fingerprint"] == "processor-fp"
    assert provenance["tokenizer_identity"] == {"tokenizer_sha256": "tok-fp"}
    assert provenance["adapter_identity"]["status"] == "loaded"
    assert provenance["embedding_delta_identity"]["status"] == "loaded"
    assert provenance["composition_mode"] == "canonical_handoff"
    assert provenance["handoff_readiness"] == "handoff"
    assert provenance["checkpoint_handoff"]["path"] == (
        "checkpoints/step-5/checkpoint_handoff.json"
    )
    assert manifest["composition_mode"] == "canonical_handoff"
    assert manifest["handoff_readiness"] == "handoff"
    assert manifest["checkpoint_handoff"]["fingerprint"] == "handoff-fp"
    assert provenance["parallelism"]["merge_status"] == "completed"
    assert provenance["parallelism"]["active_ranks"] == 2
    assert provenance["parallelism"]["visible_cuda_tokens"] == ["0", "1"]
    assert provenance["parallelism"]["per_device_batch_size"] == 1
    assert provenance["parallelism"]["shard_plan_fingerprint"] == plan.fingerprint
    assert provenance["parallelism"]["rank_to_device"] == {"0": "0", "1": "1"}
    assert provenance["parallelism"]["row_coverage"]["row_ids"] == list(row_ids)
    assert len(provenance["parallelism"]["shards"]) == 2
    assert set(provenance["parallelism"]["merged_artifacts"]) >= {
        RAW_NAME,
        SCORED_NAME,
        TOKEN_TRACE_NAME,
        PARSE_DIAGNOSTICS_NAME,
        IMAGE_PLAN_NAME,
    }
    for pred in scored_rows[0]["pred"]:
        assert pred["pred_score_source"]["row_id"] == "row-0"
        assert pred["pred_score_source"]["score_policy_fingerprint"] == SCORE_POLICY_FINGERPRINT

    recomputed = recompute_scores_from_artifacts(
        scored_jsonl=paths.scored_jsonl,
        token_trace_jsonl=paths.token_trace_jsonl,
    )
    assert recomputed[("row-0", "row-0:span-0")] == pytest.approx(
        scored_rows[0]["pred"][0]["score"]
    )
    evaluate_result = evaluate_scored_detection_artifacts(
        artifact_dir=tmp_path,
        output_dir=tmp_path / "eval",
    )
    assert evaluate_result.metrics["metric_family"] == "coordexp_swift_detection_coco_bbox_v1"


def test_strict_merge_rejects_missing_duplicate_order_failed_worker_and_identity_mismatches(
    tmp_path: Path,
) -> None:
    row_ids = ("row-0", "row-1", "row-2", "row-3")

    _assert_merge_failure(
        tmp_path / "missing-row",
        row_ids=row_ids,
        mutate=lambda shard_dirs, plan: None,
        expected_row_ids=row_ids + ("row-4",),
        expected_code="merge.missing_row",
    )
    _assert_merge_failure(
        tmp_path / "duplicate-row",
        row_ids=row_ids,
        mutate=lambda shard_dirs, plan: _copy_first_row_between_shards(
            shard_dirs[0], shard_dirs[1]
        ),
        expected_code="merge.rank_assignment_mismatch",
    )
    _assert_merge_failure(
        tmp_path / "row-order",
        row_ids=row_ids,
        mutate=lambda shard_dirs, plan: _rewrite_row_identity_across_artifacts(
            shard_dirs[0],
            row_id="row-0",
            field="row_index",
            value=99,
        ),
        expected_code="merge.rank_assignment_mismatch",
    )
    _assert_merge_failure(
        tmp_path / "missing-artifact",
        row_ids=row_ids,
        mutate=lambda shard_dirs, plan: (shard_dirs[0] / TOKEN_TRACE_NAME).unlink(),
        expected_code="merge.missing_shard_artifact",
    )
    _assert_merge_failure(
        tmp_path / "failed-worker",
        row_ids=row_ids,
        mutate=lambda shard_dirs, plan: None,
        worker_statuses={1: "failed"},
        expected_code="merge.worker_status_mismatch",
    )
    _assert_merge_failure(
        tmp_path / "summary-failed-receipt-completed",
        row_ids=row_ids,
        mutate=lambda shard_dirs, plan: _rewrite_json(
            shard_dirs[0] / SUMMARY_NAME,
            lambda payload: {**payload, "terminal_status": "failed"},
        ),
        worker_statuses={0: "completed", 1: "completed"},
        expected_code="merge.worker_status_mismatch",
    )
    result = _assert_merge_failure(
        tmp_path / "identity",
        row_ids=row_ids,
        mutate=lambda shard_dirs, plan: _rewrite_json(
            shard_dirs[1] / MANIFEST_NAME,
            lambda payload: {**payload, "model_identity_fingerprint": "different-model"},
        ),
        expected_code="merge.identity_mismatch",
    )
    assert result.context["field"] == "model_identity_fingerprint"
    result = _assert_merge_failure(
        tmp_path / "handoff-identity",
        row_ids=row_ids,
        mutate=lambda shard_dirs, plan: _rewrite_json(
            shard_dirs[1] / MANIFEST_NAME,
            lambda payload: {
                **payload,
                "checkpoint_handoff": {
                    **payload["checkpoint_handoff"],
                    "fingerprint": "different-handoff",
                },
            },
        ),
        expected_code="merge.identity_mismatch",
    )
    assert result.context["field"] == "checkpoint_handoff"
    _assert_merge_failure(
        tmp_path / "controller-identity",
        row_ids=row_ids,
        mutate=lambda shard_dirs, plan: None,
        metadata_transform=lambda metadata: {
            **metadata,
            "model_identity_fingerprint": "wrong-controller-model-fp",
        },
        expected_code="merge.controller_identity_mismatch",
    )
    _assert_merge_failure(
        tmp_path / "controller-model-identity-payload",
        row_ids=row_ids,
        mutate=lambda shard_dirs, plan: None,
        metadata_transform=lambda metadata: {
            **metadata,
            "model_identity": {"family": "wrong-but-same-fingerprint"},
        },
        expected_code="merge.controller_identity_mismatch",
    )
    result = _assert_merge_failure(
        tmp_path / "controller-composition-mode",
        row_ids=row_ids,
        mutate=lambda shard_dirs, plan: None,
        metadata_transform=lambda metadata: {
            **metadata,
            "composition_mode": "research_manual",
        },
        expected_code="merge.controller_identity_mismatch",
    )
    assert result.context["field"] == "composition_mode"


def test_strict_merge_rejects_missing_required_shard_metadata(tmp_path: Path) -> None:
    row_ids = ("row-0", "row-1")

    _assert_merge_failure(
        tmp_path / "missing-plan-fingerprint",
        row_ids=row_ids,
        mutate=lambda shard_dirs, plan: _rewrite_manifest_parallelism(
            shard_dirs[0],
            lambda parallelism: {
                key: value
                for key, value in parallelism.items()
                if key != "shard_plan_fingerprint"
            },
        ),
        expected_code="merge.shard_plan_missing",
    )
    _assert_merge_failure(
        tmp_path / "missing-terminal-status",
        row_ids=row_ids,
        mutate=lambda shard_dirs, plan: _rewrite_json(
            shard_dirs[0] / SUMMARY_NAME,
            lambda payload: {
                key: value for key, value in payload.items() if key != "terminal_status"
            },
        ),
        expected_code="merge.terminal_status_missing",
    )
    _assert_merge_failure(
        tmp_path / "missing-worker-device",
        row_ids=row_ids,
        mutate=lambda shard_dirs, plan: _rewrite_manifest_parallelism(
            shard_dirs[0],
            lambda parallelism: {
                **parallelism,
                "worker": {
                    key: value
                    for key, value in parallelism["worker"].items()
                    if key != "worker_cuda_visible_devices"
                },
            },
        ),
        expected_code="merge.worker_metadata_missing",
    )
    _assert_merge_failure(
        tmp_path / "worker-device-mismatch",
        row_ids=row_ids,
        mutate=lambda shard_dirs, plan: _rewrite_manifest_parallelism(
            shard_dirs[0],
            lambda parallelism: {
                **parallelism,
                "worker": {
                    **parallelism["worker"],
                    "parent_visible_device_token": "wrong-device",
                    "worker_cuda_visible_devices": "wrong-device",
                },
            },
        ),
        expected_code="merge.worker_metadata_mismatch",
    )
    _assert_merge_failure(
        tmp_path / "trace-device-mismatch",
        row_ids=row_ids,
        mutate=lambda shard_dirs, plan: _rewrite_first_trace_row(
            shard_dirs[0] / TOKEN_TRACE_NAME,
            lambda row: {
                **row,
                "assigned_parent_visible_device_token": "wrong-device",
            },
        ),
        expected_code="merge.worker_metadata_mismatch",
    )
    _assert_merge_failure(
        tmp_path / "rank-assignment-swapped-rows",
        row_ids=("row-0", "row-1", "row-2", "row-3"),
        mutate=lambda shard_dirs, plan: _swap_row_artifact_payloads(
            shard_dirs[0],
            shard_dirs[1],
        ),
        expected_code="merge.rank_assignment_mismatch",
    )


def test_strict_merge_rejects_missing_replay_and_duplicate_token_trace_keys(tmp_path: Path) -> None:
    row_ids = ("row-0", "row-1")

    _assert_merge_failure(
        tmp_path / "missing-replay",
        row_ids=row_ids,
        mutate=lambda shard_dirs, plan: _drop_replay_rows(shard_dirs[0] / TOKEN_TRACE_NAME),
        expected_code="artifacts.selected_replay_missing",
    )
    _assert_merge_failure(
        tmp_path / "duplicate-generated",
        row_ids=row_ids,
        mutate=lambda shard_dirs, plan: _duplicate_first_generated_trace(
            shard_dirs[0] / TOKEN_TRACE_NAME
        ),
        expected_code="merge.duplicate_generated_trace",
    )
    _assert_merge_failure(
        tmp_path / "bad-row-local-score-policy",
        row_ids=row_ids,
        mutate=lambda shard_dirs, plan: _rewrite_first_prediction_score_source(
            shard_dirs[0],
            lambda source: {**source, "score_policy_fingerprint": "bad-score-policy"},
        ),
        expected_code="merge.row_score_policy_mismatch",
    )


def test_strict_merge_rejects_semantically_malformed_rank_local_jsonl(tmp_path: Path) -> None:
    row_ids = ("row-0", "row-1")

    result = _assert_merge_failure(
        tmp_path / "bad-row-index",
        row_ids=row_ids,
        mutate=lambda shard_dirs, plan: _rewrite_row_identity_across_artifacts(
            shard_dirs[0],
            row_id="row-0",
            field="row_index",
            value="not-an-int",
        ),
        expected_code="merge.invalid_integer_field",
    )
    assert result.context["field"] == "row_index"
    assert result.context["artifact"] == RAW_NAME

    result = _assert_merge_failure(
        tmp_path / "bad-generated-step",
        row_ids=row_ids,
        mutate=lambda shard_dirs, plan: _rewrite_first_trace_row(
            shard_dirs[0] / TOKEN_TRACE_NAME,
            lambda row: {
                **row,
                "generated_step_index": "not-an-int",
            }
            if row.get("trace_type") == "generated_token"
            else row,
        ),
        expected_code="merge.invalid_integer_field",
    )
    assert result.context["field"] == "generated_step_index"
    assert result.context["artifact"] == TOKEN_TRACE_NAME

    result = _assert_merge_failure(
        tmp_path / "bad-pred-score",
        row_ids=row_ids,
        mutate=lambda shard_dirs, plan: _rewrite_first_prediction(
            shard_dirs[0],
            lambda pred: {**pred, "score": "not-a-float"},
        ),
        expected_code="merge.invalid_float_field",
    )
    assert result.context["field"] == "score"
    assert result.context["artifact"] == SCORED_NAME

    result = _assert_merge_failure(
        tmp_path / "missing-object-span",
        row_ids=row_ids,
        mutate=lambda shard_dirs, plan: _rewrite_first_prediction(
            shard_dirs[0],
            lambda pred: {
                key: value for key, value in pred.items() if key != "object_span_id"
            },
        ),
        expected_code="merge.invalid_string_field",
    )
    assert result.context["field"] == "object_span_id"
    assert result.context["artifact"] == SCORED_NAME

    result = _assert_merge_failure(
        tmp_path / "bad-replay-selected-count",
        row_ids=row_ids,
        mutate=lambda shard_dirs, plan: _rewrite_first_replay_row(
            shard_dirs[0] / TOKEN_TRACE_NAME,
            lambda payload: {**payload, "selected_count": "not-an-int"},
        ),
        expected_code="merge.invalid_integer_field",
    )
    assert result.context["field"] == "selected_count"
    assert result.context["artifact"] == TOKEN_TRACE_NAME

    result = _assert_merge_failure(
        tmp_path / "bad-replay-logprob",
        row_ids=row_ids,
        mutate=lambda shard_dirs, plan: _rewrite_first_replay_row(
            shard_dirs[0] / TOKEN_TRACE_NAME,
            lambda payload: {
                **payload,
                "selected_logprobs": ["not-a-float", *payload["selected_logprobs"][1:]],
            },
        ),
        expected_code="merge.invalid_float_field",
    )
    assert result.context["field"] == "selected_logprobs[0]"
    assert result.context["artifact"] == TOKEN_TRACE_NAME


def test_merge_failure_publishes_only_terminal_status_and_preserves_shards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.inference import merge as merge_module

    row_ids = ("row-0", "row-1")
    plan = _plan(row_ids)
    shard_dirs = _write_plan_shards(tmp_path, plan)
    stale_eval_dir = tmp_path / "eval_detection"
    stale_eval_dir.mkdir()
    for name in ("metrics.json", "coco_gt.json", "coco_predictions.json"):
        (stale_eval_dir / name).write_text(
            json.dumps({"benchmark_metric": True}) + "\n",
            encoding="utf-8",
        )
    for name in ("metrics.json", "evaluation_receipt.json", "coco_gt.json", "coco_predictions.json"):
        (tmp_path / name).write_text(
            json.dumps({"benchmark_metric": True}) + "\n",
            encoding="utf-8",
        )

    def fail_publish(*args: object, **kwargs: object) -> None:
        raise ArtifactContractError(
            "injected publication failure",
            code="merge.injected_publish_failure",
        )

    monkeypatch.setattr(merge_module, "_publish_staged_artifacts", fail_publish)

    with pytest.raises(ArtifactContractError) as exc_info:
        merge_module.merge_shard_artifacts(
            output_dir=tmp_path,
            shard_dirs=shard_dirs,
            expected_row_ids=row_ids,
            metadata=_merge_metadata(plan),
            plan=plan,
        )

    assert exc_info.value.code == "merge.injected_publish_failure"
    assert (tmp_path / SUMMARY_NAME).is_file()
    assert (tmp_path / MANIFEST_NAME).is_file()
    assert not (tmp_path / RAW_NAME).exists()
    assert not (tmp_path / SCORED_NAME).exists()
    assert not (tmp_path / PROVENANCE_NAME).exists()
    assert not (tmp_path / TOKEN_TRACE_NAME).exists()
    assert not (tmp_path / PARSE_DIAGNOSTICS_NAME).exists()
    assert not (tmp_path / IMAGE_PLAN_NAME).exists()
    assert not stale_eval_dir.exists()
    assert not (tmp_path / "metrics.json").exists()
    assert not (tmp_path / "evaluation_receipt.json").exists()
    assert not (tmp_path / "coco_gt.json").exists()
    assert not (tmp_path / "coco_predictions.json").exists()
    assert all(shard_dir.is_dir() for shard_dir in shard_dirs)
    assert _read_json(tmp_path / SUMMARY_NAME)["benchmark_eligible"] is False


def _assert_merge_failure(
    output_dir: Path,
    *,
    row_ids: tuple[str, ...],
    mutate: Any,
    expected_code: str,
    expected_row_ids: tuple[str, ...] | None = None,
    worker_statuses: dict[int, str] | None = None,
    metadata_transform: Any | None = None,
) -> ArtifactContractError:
    from src.inference.merge import merge_shard_artifacts

    plan = _plan(row_ids)
    shard_dirs = _write_plan_shards(output_dir, plan)
    mutate(shard_dirs, plan)
    stale_eval_dir = output_dir / "eval_detection"
    stale_eval_dir.mkdir()
    for name in ("metrics.json", "evaluation_receipt.json", "coco_gt.json", "coco_predictions.json"):
        (stale_eval_dir / name).write_text(
            json.dumps({"benchmark_metric": True}) + "\n",
            encoding="utf-8",
        )
        (output_dir / name).write_text(
            json.dumps({"benchmark_metric": True}) + "\n",
            encoding="utf-8",
        )
    with pytest.raises(ArtifactContractError) as exc_info:
        merge_shard_artifacts(
            output_dir=output_dir,
            shard_dirs=shard_dirs,
            expected_row_ids=expected_row_ids or row_ids,
            metadata=(metadata_transform or (lambda metadata: metadata))(
                _merge_metadata(plan)
            ),
            plan=plan,
            worker_statuses=worker_statuses,
        )
    assert exc_info.value.code == expected_code
    assert (output_dir / SUMMARY_NAME).is_file()
    assert (output_dir / MANIFEST_NAME).is_file()
    assert _read_json(output_dir / SUMMARY_NAME)["benchmark_eligible"] is False
    assert _read_json(output_dir / MANIFEST_NAME)["terminal_status"] == "failed"
    assert not (output_dir / RAW_NAME).exists()
    assert not (output_dir / SCORED_NAME).exists()
    assert not stale_eval_dir.exists()
    assert not (output_dir / "metrics.json").exists()
    assert not (output_dir / "evaluation_receipt.json").exists()
    assert not (output_dir / "coco_gt.json").exists()
    assert not (output_dir / "coco_predictions.json").exists()
    return exc_info.value


def _write_plan_shards(root: Path, plan: DataParallelPlan) -> tuple[Path, ...]:
    shard_dirs = []
    for rank_plan in plan.ranks:
        shard_dirs.append(_write_shard(root, plan=plan, rank_plan=rank_plan))
    return tuple(shard_dirs)


def _write_shard(root: Path, *, plan: DataParallelPlan, rank_plan: RankShardPlan) -> Path:
    shard_dir = root / "shards" / rank_plan.shard_dir_name
    rows = [
        _raw_row(row_id=row_id, row_index=row_index)
        for row_id, row_index in zip(rank_plan.row_ids, rank_plan.row_indices, strict=True)
    ]
    paths = write_inference_artifacts(
        output_dir=shard_dir,
        rows=rows,
        decode_results={row["row_id"]: _decode_result(row["row_id"]) for row in rows},
        image_plan_rows=[
            {"row_id": row["row_id"], "row_index": row["row_index"]} for row in rows
        ],
        metadata=_shard_metadata(plan=plan, rank_plan=rank_plan),
    )
    _add_rank_metadata_to_jsonl(paths.token_trace_jsonl, rank_plan=rank_plan)
    _append_ranked_diagnostic_rows(paths.parse_diagnostics_jsonl, rank_plan=rank_plan)
    return shard_dir


def _plan(row_ids: tuple[str, ...]) -> DataParallelPlan:
    return plan_data_parallel_shards(
        row_ids=row_ids,
        per_device_batch_size=1,
        visible_cuda_tokens=("0", "1"),
    )


def _raw_row(*, row_id: str, row_index: int) -> dict[str, Any]:
    parse_row = parse_compact_object_box_closed(
        OBJECT_TEXT,
        row_id=row_id,
        row_index=row_index,
        image_width=1000,
        image_height=1000,
    )
    return {
        "row_id": row_id,
        "row_index": row_index,
        "example_id": row_id,
        "image_path": f"{row_id}.jpg",
        "image_width": 1000,
        "image_height": 1000,
        "gt": [{"description": "cat", "bbox": [100, 200, 300, 400]}],
        "raw_decode_text": OBJECT_TEXT,
        "decode_stop_reason": "length",
        "parse": parse_row,
    }


def _decode_result(row_id: str) -> DecodeResult:
    traces = [
        TokenTrace(
            step_index=index,
            token_id=151646 + index,
            token_text=piece,
            logprob=math.log(0.25),
            is_stop=False,
            is_pad=False,
            backend="hf",
            backend_mode="generate",
            response_family="hf",
        )
        for index, piece in enumerate(
            [
                "<|object_ref_start|>",
                "cat",
                "<|object_ref_end|>",
                "<|box_start|>",
                "<|coord_100|>",
                "<|coord_200|>",
                "<|coord_300|>",
                "<|coord_400|>",
                "<|box_end|>",
            ]
        )
    ]
    return DecodeResult(
        request_id=row_id,
        backend="hf",
        backend_mode="generate",
        response_family="hf",
        prompt_token_ids=[11, 12],
        generated_token_ids=[trace.token_id for trace in traces],
        raw_generated_text=OBJECT_TEXT,
        parser_text=OBJECT_TEXT,
        strip_policy="none",
        stop_reason="length",
        model_identity={"family": "unit"},
        tokenizer_identity={"tokenizer_sha256": "tok-fp"},
        generation_config_fingerprint="gen-fp",
        token_trace=traces,
    )


def _base_metadata() -> dict[str, Any]:
    return {
        "artifact_schema_version": 1,
        "resolved_config_fingerprints": {"infer_config": "infer-fp"},
        "detection_template_id": "compact-object-box-closed",
        "prompt_policy_fingerprint": "prompt-fp",
        "generation_config_fingerprint": "gen-fp",
        "generation_policy": {
            "batch_size": 1,
            "max_new_tokens": 64,
            "temperature": 0.0,
            "top_p": 1.0,
            "repetition_penalty": 1.0,
        },
        "model_identity": {"family": "unit", "base_model": "unit-qwen"},
        "model_identity_fingerprint": "model-fp",
        "processor_identity": {"processor_class": "unit-processor"},
        "processor_identity_fingerprint": "processor-fp",
        "tokenizer_identity": {"tokenizer_sha256": "tok-fp"},
        "adapter_identity": {
            "status": "loaded",
            "active_adapter": "llm-dora",
            "fingerprint": "adapter-fp",
        },
        "embedding_delta_identity": {
            "status": "loaded",
            "fingerprint": "embed-delta-fp",
        },
        "composition_mode": "canonical_handoff",
        "handoff_readiness": "handoff",
        "checkpoint_handoff": {
            "path": "checkpoints/step-5/checkpoint_handoff.json",
            "fingerprint": "handoff-fp",
            "adapter_identity": {"fingerprint": "adapter-fp"},
            "special_token_embedding_identity": {"fingerprint": "embed-delta-fp"},
        },
        "template_identity": {
            "id": "compact-object-box-closed",
            "object_field_order": "desc_first",
            "object_ordering": "geo_sorted",
            "assistant_format": "object_box_closed",
        },
        "parser_policy": "compact_object_box_closed_only",
        "dataset_identity": {"input_jsonl": "unit.jsonl", "row_count": 4},
        "backend": "hf",
        "backend_mode": "generate",
        "response_family": "hf",
        "pipeline_counters": {
            "terminal_status": "completed",
            "decode_success_count": 1,
            "parser_failure_count": 0,
            "dropped_prediction_count": 0,
            "truncated_decode_count": 1,
            "decode_stop_reasons": {"length": 1},
            "image_validation_failure_count": 0,
            "score_failure_count": 0,
        },
    }


def _shard_metadata(*, plan: DataParallelPlan, rank_plan: RankShardPlan) -> dict[str, Any]:
    metadata = _base_metadata()
    metadata["parallelism"] = {
        "execution_mode": "rank_local_shard",
        "shard_plan_fingerprint": plan.fingerprint,
        "worker": {
            "rank": rank_plan.rank,
            "world_size": rank_plan.world_size,
            "parent_visible_device_token": rank_plan.parent_visible_device_token,
            "worker_cuda_visible_devices": rank_plan.parent_visible_device_token,
            "worker_logical_device": "cuda:0",
            "cuda_device_count": 1,
            "cuda_current_device": 0,
            "model_first_parameter_device": "cuda:0",
        },
        "shard_assignment": {
            "assigned_row_indices": list(rank_plan.row_indices),
            "assigned_row_ids": list(rank_plan.row_ids),
        },
    }
    return metadata


def _merge_metadata(plan: DataParallelPlan) -> dict[str, Any]:
    metadata = _base_metadata()
    metadata["parallelism"] = {
        "execution_mode": "controller_worker",
        "active_ranks": plan.active_ranks,
        "visible_cuda_tokens": list(plan.visible_cuda_tokens),
        "per_device_batch_size": plan.per_device_batch_size,
        "shard_plan_fingerprint": plan.fingerprint,
        "plan": plan.to_artifact_dict(),
    }
    return metadata


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rewrite_json(path: Path, transform: Any) -> None:
    path.write_text(
        json.dumps(transform(_read_json(path)), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _rewrite_manifest_parallelism(shard_dir: Path, transform: Any) -> None:
    _rewrite_json(
        shard_dir / MANIFEST_NAME,
        lambda payload: {
            **payload,
            "parallelism": transform(dict(payload["parallelism"])),
        },
    )


def _add_rank_metadata_to_jsonl(path: Path, *, rank_plan: RankShardPlan) -> None:
    rows = []
    for row in _read_jsonl(path):
        rows.append(
            {
                **row,
                "rank": rank_plan.rank,
                "world_size": rank_plan.world_size,
                "assigned_parent_visible_device_token": rank_plan.parent_visible_device_token,
                "worker_cuda_visible_devices": rank_plan.parent_visible_device_token,
                "worker_logical_device": "cuda:0",
            }
        )
    _write_jsonl(path, rows)


def _append_ranked_diagnostic_rows(path: Path, *, rank_plan: RankShardPlan) -> None:
    rows = _read_jsonl(path)
    for row_id, row_index in zip(rank_plan.row_ids, rank_plan.row_indices, strict=True):
        rows.append(
            {
                "row_id": row_id,
                "row_index": row_index,
                "diagnostic_type": "rank_metadata_probe",
                "code": "unit.rank_probe",
                "rank": rank_plan.rank,
                "world_size": rank_plan.world_size,
                "assigned_parent_visible_device_token": rank_plan.parent_visible_device_token,
                "worker_cuda_visible_devices": rank_plan.parent_visible_device_token,
                "worker_logical_device": "cuda:0",
            }
        )
    _write_jsonl(path, rows)


def _copy_first_row_between_shards(source_dir: Path, target_dir: Path) -> None:
    for name in (RAW_NAME, SCORED_NAME, IMAGE_PLAN_NAME):
        source_rows = _read_jsonl(source_dir / name)
        target_rows = _read_jsonl(target_dir / name)
        _write_jsonl(target_dir / name, [source_rows[0], *target_rows])
    _rewrite_json(
        target_dir / PROVENANCE_NAME,
        lambda payload: {
            **payload,
            "raw_artifact": {
                **payload["raw_artifact"],
                "sha256": sha256_file(target_dir / RAW_NAME),
            },
            "scored_artifact": {
                **payload["scored_artifact"],
                "sha256": sha256_file(target_dir / SCORED_NAME),
            },
        },
    )


def _rewrite_row_identity(path: Path, *, row_id: str, field: str, value: Any) -> None:
    rows = []
    for row in _read_jsonl(path):
        if row["row_id"] == row_id:
            row = {**row, field: value}
        rows.append(row)
    _write_jsonl(path, rows)


def _rewrite_row_identity_across_artifacts(
    shard_dir: Path,
    *,
    row_id: str,
    field: str,
    value: Any,
) -> None:
    for name in (RAW_NAME, SCORED_NAME, IMAGE_PLAN_NAME):
        _rewrite_row_identity(shard_dir / name, row_id=row_id, field=field, value=value)
    _rewrite_json(
        shard_dir / PROVENANCE_NAME,
        lambda payload: {
            **payload,
            "raw_artifact": {
                **payload["raw_artifact"],
                "sha256": sha256_file(shard_dir / RAW_NAME),
            },
            "scored_artifact": {
                **payload["scored_artifact"],
                "sha256": sha256_file(shard_dir / SCORED_NAME),
            },
        },
    )


def _drop_replay_rows(path: Path) -> None:
    _write_jsonl(
        path,
        [row for row in _read_jsonl(path) if row.get("trace_type") != "selected_token_replay"],
    )


def _duplicate_first_generated_trace(path: Path) -> None:
    rows = _read_jsonl(path)
    generated = next(row for row in rows if row.get("trace_type") == "generated_token")
    _write_jsonl(path, [generated, *rows])


def _rewrite_first_trace_row(path: Path, transform: Any) -> None:
    rows = _read_jsonl(path)
    rows[0] = transform(dict(rows[0]))
    _write_jsonl(path, rows)


def _rewrite_first_replay_row(path: Path, transform: Any) -> None:
    rows = _read_jsonl(path)
    for index, row in enumerate(rows):
        if row.get("trace_type") == "selected_token_replay":
            rows[index] = transform(dict(row))
            break
    else:
        raise AssertionError("expected selected-token replay row in fixture")
    _write_jsonl(path, rows)


def _rewrite_first_prediction(shard_dir: Path, transform: Any) -> None:
    rows = _read_jsonl(shard_dir / SCORED_NAME)
    rows[0]["pred"][0] = transform(dict(rows[0]["pred"][0]))
    _write_jsonl(shard_dir / SCORED_NAME, rows)
    _rewrite_json(
        shard_dir / PROVENANCE_NAME,
        lambda payload: {
            **payload,
            "scored_artifact": {
                **payload["scored_artifact"],
                "sha256": sha256_file(shard_dir / SCORED_NAME),
            },
        },
    )


def _rewrite_first_prediction_score_source(shard_dir: Path, transform: Any) -> None:
    rows = _read_jsonl(shard_dir / SCORED_NAME)
    source = dict(rows[0]["pred"][0]["pred_score_source"])
    rows[0]["pred"][0]["pred_score_source"] = transform(source)
    _write_jsonl(shard_dir / SCORED_NAME, rows)
    _rewrite_json(
        shard_dir / PROVENANCE_NAME,
        lambda payload: {
            **payload,
            "scored_artifact": {
                **payload["scored_artifact"],
                "sha256": sha256_file(shard_dir / SCORED_NAME),
            },
        },
    )


def _swap_row_artifact_payloads(left_dir: Path, right_dir: Path) -> None:
    for name in (RAW_NAME, SCORED_NAME, IMAGE_PLAN_NAME, TOKEN_TRACE_NAME, PARSE_DIAGNOSTICS_NAME):
        left_rows = _read_jsonl(left_dir / name)
        right_rows = _read_jsonl(right_dir / name)
        _write_jsonl(left_dir / name, right_rows)
        _write_jsonl(right_dir / name, left_rows)
    for shard_dir in (left_dir, right_dir):
        _rewrite_json(
            shard_dir / PROVENANCE_NAME,
            lambda payload, shard_dir=shard_dir: {
                **payload,
                "raw_artifact": {
                    **payload["raw_artifact"],
                    "sha256": sha256_file(shard_dir / RAW_NAME),
                },
                "scored_artifact": {
                    **payload["scored_artifact"],
                    "sha256": sha256_file(shard_dir / SCORED_NAME),
                },
            },
        )
