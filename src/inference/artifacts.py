"""Deterministic inference artifact writers for Wave 5 scoring output."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.common.errors import ArtifactContractError
from src.inference.backend import DecodeResult, TokenTrace
from src.inference.scoring import SCORE_POLICY_FINGERPRINT, score_prediction


RAW_NAME = "gt_vs_pred.jsonl"
SCORED_NAME = "gt_vs_pred_scored.jsonl"
PROVENANCE_NAME = "gt_vs_pred_scored.jsonl.provenance.json"
TOKEN_TRACE_NAME = "pred_token_trace.jsonl"
PARSE_DIAGNOSTICS_NAME = "parse_diagnostics.jsonl"
IMAGE_PLAN_NAME = "image_plan.jsonl"
SUMMARY_NAME = "summary.json"
MANIFEST_NAME = "run_manifest.json"


@dataclass(frozen=True)
class InferenceArtifactPaths:
    output_dir: Path
    raw_jsonl: Path
    scored_jsonl: Path
    provenance_json: Path
    token_trace_jsonl: Path
    parse_diagnostics_jsonl: Path
    image_plan_jsonl: Path
    summary_json: Path
    run_manifest_json: Path


def write_inference_artifacts(
    *,
    output_dir: Path,
    rows: list[dict[str, Any]],
    decode_results: dict[str, DecodeResult],
    image_plan_rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> InferenceArtifactPaths:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = InferenceArtifactPaths(
        output_dir=output_dir,
        raw_jsonl=output_dir / RAW_NAME,
        scored_jsonl=output_dir / SCORED_NAME,
        provenance_json=output_dir / PROVENANCE_NAME,
        token_trace_jsonl=output_dir / TOKEN_TRACE_NAME,
        parse_diagnostics_jsonl=output_dir / PARSE_DIAGNOSTICS_NAME,
        image_plan_jsonl=output_dir / IMAGE_PLAN_NAME,
        summary_json=output_dir / SUMMARY_NAME,
        run_manifest_json=output_dir / MANIFEST_NAME,
    )
    _require_traces_for_predicted_rows(rows, decode_results)
    _validate_image_plan_rows(rows, image_plan_rows)
    raw_model_logprob_status = _raw_model_logprob_status(metadata)
    _validate_decode_results(
        rows,
        decode_results,
        raw_model_logprob_required=raw_model_logprob_status == "available",
    )

    raw_rows: list[dict[str, Any]] = []
    scored_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    token_trace_rows: list[dict[str, Any]] = []
    scoreable_prediction_count = 0
    score_failure_count = 0

    for row in rows:
        parse_row = row["parse"]
        raw_row = _raw_artifact_row(row)
        raw_rows.append(raw_row)
        diagnostic_rows.extend(_diagnostic_rows(parse_row))

        decode_result = decode_results.get(str(row["row_id"]))
        if decode_result is not None:
            token_trace_rows.extend(
                _token_trace_rows(
                    row_id=str(row["row_id"]),
                    result=decode_result,
                    raw_model_logprob_status=raw_model_logprob_status,
                )
            )
        scored_pred: list[dict[str, Any]] = []
        if decode_result is not None and parse_row.predictions:
            for prediction in parse_row.predictions:
                try:
                    scored = score_prediction(
                        row_id=str(row["row_id"]),
                        prediction=prediction,
                        token_trace=list(decode_result.token_trace),
                    )
                except ArtifactContractError as exc:
                    score_failure_count += 1
                    diagnostic_rows.append(
                        {
                            "row_id": row["row_id"],
                            "row_index": row["row_index"],
                            "object_span_id": prediction.get("object_span_id"),
                            "diagnostic_type": "scoring",
                            "code": exc.code,
                            "context": exc.context,
                        }
                    )
                    continue
                scored_pred.append(scored.prediction)
                token_trace_rows.append(_replay_trace_row(scored.replay))
        scoreable_prediction_count += len(scored_pred)
        scored_rows.append(_scored_artifact_row(row, pred=scored_pred))

    staging_dir = Path(tempfile.mkdtemp(prefix=".wave5-artifacts-", dir=output_dir))
    staged_paths = InferenceArtifactPaths(
        output_dir=staging_dir,
        raw_jsonl=staging_dir / RAW_NAME,
        scored_jsonl=staging_dir / SCORED_NAME,
        provenance_json=staging_dir / PROVENANCE_NAME,
        token_trace_jsonl=staging_dir / TOKEN_TRACE_NAME,
        parse_diagnostics_jsonl=staging_dir / PARSE_DIAGNOSTICS_NAME,
        image_plan_jsonl=staging_dir / IMAGE_PLAN_NAME,
        summary_json=staging_dir / SUMMARY_NAME,
        run_manifest_json=staging_dir / MANIFEST_NAME,
    )
    try:
        _write_jsonl(staged_paths.raw_jsonl, raw_rows)
        _write_jsonl(staged_paths.scored_jsonl, scored_rows)
        _write_jsonl(staged_paths.token_trace_jsonl, token_trace_rows)
        _write_jsonl(staged_paths.parse_diagnostics_jsonl, diagnostic_rows)
        _write_jsonl(staged_paths.image_plan_jsonl, image_plan_rows)

        raw_sha = sha256_file(staged_paths.raw_jsonl)
        scored_sha = sha256_file(staged_paths.scored_jsonl)
        provenance = _provenance(
            metadata=metadata,
            raw_sha=raw_sha,
            scored_sha=scored_sha,
            row_ids=[str(row["row_id"]) for row in rows],
        )
        _write_json(staged_paths.provenance_json, provenance)
        summary = {
            "row_count": len(rows),
            "raw_row_count": len(raw_rows),
            "scored_row_count": len(scored_rows),
            "scoreable_prediction_count": scoreable_prediction_count,
            "diagnostic_row_count": len(diagnostic_rows),
            "trace_row_count": len(token_trace_rows),
            "scored_artifact_materialized": True,
            "benchmark_eligible": False,
            "generation_policy": dict(metadata.get("generation_policy") or {}),
            "likelihood_semantics": dict(
                metadata.get("likelihood_semantics") or {}
            ),
            "raw_model_logprob_status": _raw_model_logprob_status(metadata),
            **dict(metadata.get("pipeline_counters") or {}),
            "score_failure_count": score_failure_count,
        }
        _write_json(staged_paths.summary_json, summary)
        _write_json(staged_paths.run_manifest_json, _manifest(metadata=metadata, summary=summary))
        validate_scored_artifact_set(staging_dir)
        _replace_final_artifacts(staged_paths, paths)
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)
    validate_scored_artifact_set(output_dir)
    return paths


def write_terminal_status_artifacts(
    *,
    output_dir: Path,
    metadata: dict[str, Any],
    summary: dict[str, Any],
) -> InferenceArtifactPaths:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = InferenceArtifactPaths(
        output_dir=output_dir,
        raw_jsonl=output_dir / RAW_NAME,
        scored_jsonl=output_dir / SCORED_NAME,
        provenance_json=output_dir / PROVENANCE_NAME,
        token_trace_jsonl=output_dir / TOKEN_TRACE_NAME,
        parse_diagnostics_jsonl=output_dir / PARSE_DIAGNOSTICS_NAME,
        image_plan_jsonl=output_dir / IMAGE_PLAN_NAME,
        summary_json=output_dir / SUMMARY_NAME,
        run_manifest_json=output_dir / MANIFEST_NAME,
    )
    terminal_summary = {
        "row_count": 0,
        "raw_row_count": 0,
        "scored_row_count": 0,
        "scoreable_prediction_count": 0,
        "diagnostic_row_count": 0,
        "trace_row_count": 0,
        "scored_artifact_materialized": False,
        "benchmark_eligible": False,
        **dict(summary),
    }
    staging_dir = Path(tempfile.mkdtemp(prefix=".terminal-status-", dir=output_dir))
    staged_paths = InferenceArtifactPaths(
        output_dir=staging_dir,
        raw_jsonl=staging_dir / RAW_NAME,
        scored_jsonl=staging_dir / SCORED_NAME,
        provenance_json=staging_dir / PROVENANCE_NAME,
        token_trace_jsonl=staging_dir / TOKEN_TRACE_NAME,
        parse_diagnostics_jsonl=staging_dir / PARSE_DIAGNOSTICS_NAME,
        image_plan_jsonl=staging_dir / IMAGE_PLAN_NAME,
        summary_json=staging_dir / SUMMARY_NAME,
        run_manifest_json=staging_dir / MANIFEST_NAME,
    )
    try:
        _write_json(staged_paths.summary_json, terminal_summary)
        _write_json(
            staged_paths.run_manifest_json,
            _terminal_manifest(metadata=metadata, summary=terminal_summary),
        )
        _replace_terminal_status_artifacts(staged_paths, paths)
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)
    return paths


def validate_scored_artifact_set(output_dir: Path) -> None:
    required = [
        RAW_NAME,
        SCORED_NAME,
        TOKEN_TRACE_NAME,
        PROVENANCE_NAME,
        PARSE_DIAGNOSTICS_NAME,
        IMAGE_PLAN_NAME,
        SUMMARY_NAME,
        MANIFEST_NAME,
    ]
    for name in required:
        path = output_dir / name
        if not path.is_file():
            code = "artifacts.missing_provenance" if name == PROVENANCE_NAME else "artifacts.missing_artifact"
            raise ArtifactContractError(
                "scored artifact set is missing a required artifact",
                code=code,
                context={"artifact": name, "output_dir": str(output_dir)},
            )
    provenance = json.loads((output_dir / PROVENANCE_NAME).read_text(encoding="utf-8"))
    if provenance["raw_artifact"]["sha256"] != sha256_file(output_dir / RAW_NAME):
        raise ArtifactContractError(
            "raw artifact sha does not match provenance",
            code="artifacts.raw_sha_mismatch",
        )
    if provenance["scored_artifact"]["sha256"] != sha256_file(output_dir / SCORED_NAME):
        raise ArtifactContractError(
            "scored artifact sha does not match provenance",
            code="artifacts.scored_sha_mismatch",
        )


def recompute_scores_from_artifacts(
    *,
    scored_jsonl: Path,
    token_trace_jsonl: Path,
) -> dict[tuple[str, str], float]:
    replay_rows: dict[tuple[str, str], dict[str, Any]] = {}
    generated_rows: dict[tuple[str, int], dict[str, Any]] = {}
    for row in _read_jsonl(token_trace_jsonl):
        if row.get("trace_type") == "selected_token_replay":
            replay_rows[(row["row_id"], row["object_span_id"])] = row
        elif row.get("trace_type") == "generated_token":
            generated_rows[(row["row_id"], int(row["generated_step_index"]))] = row
    recomputed: dict[tuple[str, str], float] = {}
    for row in _read_jsonl(scored_jsonl):
        for pred in row.get("pred", []):
            key = (row["row_id"], pred["object_span_id"])
            if key not in replay_rows:
                raise ArtifactContractError(
                    "selected-token replay row is missing",
                    code="artifacts.selected_replay_missing",
                    context={"row_id": key[0], "object_span_id": key[1]},
                )
            replay = replay_rows[key]
            logprobs = _verified_selected_logprobs_from_generated_trace(
                row_id=key[0],
                object_span_id=key[1],
                replay=replay,
                generated_rows=generated_rows,
            )
            recomputed[key] = math.exp(sum(logprobs) / len(logprobs))
    return recomputed


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_traces_for_predicted_rows(
    rows: list[dict[str, Any]],
    decode_results: dict[str, DecodeResult],
) -> None:
    for row in rows:
        parse_row = row["parse"]
        if parse_row.predictions and str(row["row_id"]) not in decode_results:
            raise ArtifactContractError(
                "scored artifact production requires trace for every row with predictions",
                code="artifacts.missing_trace",
                context={"row_id": row["row_id"]},
            )


def _validate_image_plan_rows(
    rows: list[dict[str, Any]],
    image_plan_rows: list[dict[str, Any]],
) -> None:
    expected = [
        {"row_id": str(row["row_id"]), "row_index": int(row["row_index"])}
        for row in rows
    ]
    observed = [
        {
            "row_id": str(row.get("row_id")),
            "row_index": int(row.get("row_index", -1)),
        }
        for row in image_plan_rows
    ]
    if observed != expected:
        raise ArtifactContractError(
            "image_plan.jsonl rows must preserve raw row identity and order",
            code="artifacts.image_plan_row_mismatch",
            context={"expected_rows": expected, "observed_rows": observed},
        )


def _validate_decode_results(
    rows: list[dict[str, Any]],
    decode_results: dict[str, DecodeResult],
    *,
    raw_model_logprob_required: bool,
) -> None:
    for row in rows:
        row_id = str(row["row_id"])
        result = decode_results.get(row_id)
        if result is None:
            continue
        if result.request_id != row_id:
            raise ArtifactContractError(
                "decode result request_id must match artifact row_id",
                code="artifacts.decode_result_row_mismatch",
                context={"row_id": row_id, "request_id": result.request_id},
            )
        try:
            result.validate_for_scored(
                raw_model_logprob_required=raw_model_logprob_required
            )
        except ArtifactContractError:
            raise
        except Exception as exc:
            raise ArtifactContractError(
                "decode result failed scored trace validation",
                code="artifacts.decode_result_invalid",
                context={"row_id": row_id},
                cause=exc,
            ) from exc
        for trace in result.token_trace:
            if trace.logprob is not None and not math.isfinite(float(trace.logprob)):
                raise ArtifactContractError(
                    "generated-token logprob must be finite before artifact writing",
                    code="artifacts.non_finite_trace_logprob",
                    context={
                        "row_id": row_id,
                        "generated_step_index": trace.step_index,
                        "token_id": trace.token_id,
                    },
                )


def _raw_artifact_row(row: dict[str, Any]) -> dict[str, Any]:
    parse_row = row["parse"]
    return {
        "row_id": row["row_id"],
        "row_index": row["row_index"],
        "example_id": row.get("example_id", row["row_id"]),
        "image_path": row["image_path"],
        "image_width": row["image_width"],
        "image_height": row["image_height"],
        "gt": _json_safe(row.get("gt", [])),
        "pred": _json_safe(parse_row.predictions),
        "raw_decode_text": row.get("raw_decode_text", ""),
        "decode_stop_reason": row.get("decode_stop_reason", ""),
        "parser_id": parse_row.parser_id,
        "parser_policy": parse_row.parser_policy,
        "metric_bearing": parse_row.metric_bearing,
        "parse_status": parse_row.parse_status,
        "valid_prediction_count": parse_row.valid_prediction_count,
        "dropped_prediction_count": parse_row.dropped_prediction_count,
        "dropped_predictions": _json_safe(parse_row.dropped_predictions),
    }


def _scored_artifact_row(row: dict[str, Any], *, pred: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "row_id": row["row_id"],
        "row_index": row["row_index"],
        "example_id": row.get("example_id", row["row_id"]),
        "image_path": row["image_path"],
        "image_width": row["image_width"],
        "image_height": row["image_height"],
        "gt": _json_safe(row.get("gt", [])),
        "pred": _json_safe(pred),
    }


def _diagnostic_rows(parse_row: Any) -> list[dict[str, Any]]:
    return [_json_safe(item) for item in parse_row.diagnostics]


def _token_trace_rows(
    *,
    row_id: str,
    result: DecodeResult,
    raw_model_logprob_status: str,
) -> list[dict[str, Any]]:
    rows = []
    for trace in result.token_trace:
        rows.append(
            {
                "trace_type": "generated_token",
                "row_id": row_id,
                "generated_step_index": trace.step_index,
                "token_id": trace.token_id,
                "token_text": trace.token_text,
                "logprob": trace.logprob,
                "raw_model_logprob": trace.raw_model_logprob,
                "raw_model_logprob_status": raw_model_logprob_status,
                "is_stop": trace.is_stop,
                "is_pad": trace.is_pad,
                "backend": trace.backend,
                "backend_mode": trace.backend_mode,
                "response_family": trace.response_family,
            }
        )
    return rows


def _replay_trace_row(replay: dict[str, Any]) -> dict[str, Any]:
    return {
        "trace_type": "selected_token_replay",
        "row_id": replay["row_id"],
        "object_span_id": replay["object_span_id"],
        "generated_step_indices": replay["generated_step_indices"],
        "token_ids": replay["token_ids"],
        "token_text": replay["token_text"],
        "selected_logprobs": replay["selected_logprobs"],
        "selected_count": replay["selected_count"],
        "score": replay["score"],
        "score_policy_fingerprint": replay["score_policy_fingerprint"],
    }


def _raw_model_logprob_status(metadata: dict[str, Any]) -> str:
    enabled = metadata.get("raw_model_logprob_enabled", False)
    if not isinstance(enabled, bool):
        raise ArtifactContractError(
            "raw-model likelihood enablement must be boolean",
            code="artifacts.invalid_raw_model_logprob_status",
            context={"raw_model_logprob_enabled": enabled},
        )
    return "available" if enabled else "disabled"


def _provenance(
    *,
    metadata: dict[str, Any],
    raw_sha: str,
    scored_sha: str,
    row_ids: list[str],
) -> dict[str, Any]:
    return {
        "artifact_schema_version": int(metadata.get("artifact_schema_version", 1)),
        "raw_artifact": {"path": RAW_NAME, "sha256": raw_sha},
        "scored_artifact": {"path": SCORED_NAME, "sha256": scored_sha},
        "detection_template_id": metadata["detection_template_id"],
        "prompt_policy_fingerprint": metadata["prompt_policy_fingerprint"],
        "decode_policy_fingerprint": metadata["generation_config_fingerprint"],
        "generation_config_fingerprint": metadata["generation_config_fingerprint"],
        "generation_policy": dict(metadata.get("generation_policy") or {}),
        "backend_session": dict(metadata.get("backend_session") or {}),
        "likelihood_semantics": dict(metadata.get("likelihood_semantics") or {}),
        "execution_model_identity": metadata.get("execution_model_identity"),
        "frontend_identity": dict(metadata.get("frontend_identity") or {}),
        "raw_model_logprob_status": _raw_model_logprob_status(metadata),
        "media_identity": dict(metadata.get("media_identity") or {}),
        "parallelism": dict(metadata.get("parallelism") or {}),
        "model_identity": dict(metadata.get("model_identity") or {}),
        "model_identity_fingerprint": metadata["model_identity_fingerprint"],
        "processor_identity": dict(metadata.get("processor_identity") or {}),
        "processor_identity_fingerprint": metadata["processor_identity_fingerprint"],
        "tokenizer_identity": dict(metadata.get("tokenizer_identity") or {}),
        "adapter_identity": metadata.get("adapter_identity"),
        "embedding_delta_identity": metadata.get("embedding_delta_identity"),
        "template_identity": metadata["template_identity"],
        "parser_policy": metadata["parser_policy"],
        "score_policy_fingerprint": SCORE_POLICY_FINGERPRINT,
        "row_binding": {
            "row_count": len(row_ids),
            "row_ids_sha256": hashlib.sha256(
                json.dumps(row_ids, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
        },
    }


def _manifest(*, metadata: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "artifact_schema_version": int(metadata.get("artifact_schema_version", 1)),
        "artifacts": {
            "gt_vs_pred": RAW_NAME,
            "gt_vs_pred_scored": SCORED_NAME,
            "gt_vs_pred_scored_provenance": PROVENANCE_NAME,
            "pred_token_trace": TOKEN_TRACE_NAME,
            "parse_diagnostics": PARSE_DIAGNOSTICS_NAME,
            "image_plan": IMAGE_PLAN_NAME,
            "summary": SUMMARY_NAME,
        },
        "resolved_config_fingerprints": metadata.get("resolved_config_fingerprints", {}),
        "model_identity": metadata.get("model_identity", {}),
        "model_identity_fingerprint": metadata["model_identity_fingerprint"],
        "processor_identity": metadata.get("processor_identity", {}),
        "adapter_identity": metadata.get("adapter_identity"),
        "embedding_delta_identity": metadata.get("embedding_delta_identity"),
        "tokenizer_identity": metadata.get("tokenizer_identity", {}),
        "backend": metadata["backend"],
        "backend_mode": metadata["backend_mode"],
        "response_family": metadata["response_family"],
        "dataset_identity": metadata["dataset_identity"],
        "generation_config_fingerprint": metadata["generation_config_fingerprint"],
        "generation_policy": dict(metadata.get("generation_policy") or {}),
        "backend_session": dict(metadata.get("backend_session") or {}),
        "likelihood_semantics": dict(metadata.get("likelihood_semantics") or {}),
        "execution_model_identity": metadata.get("execution_model_identity"),
        "frontend_identity": dict(metadata.get("frontend_identity") or {}),
        "raw_model_logprob_status": _raw_model_logprob_status(metadata),
        "media_identity": dict(metadata.get("media_identity") or {}),
        "parallelism": dict(metadata.get("parallelism") or {}),
        "score_policy_fingerprint": SCORE_POLICY_FINGERPRINT,
        "trace_scoring_status": "scored",
        "prompt_policy_fingerprint": metadata["prompt_policy_fingerprint"],
        "template_identity": metadata["template_identity"],
        "processor_identity_fingerprint": metadata["processor_identity_fingerprint"],
        "evaluator_consumer_status": str(
            metadata.get("evaluator_consumer_status", "available_not_run")
        ),
        "scored_artifact_materialized": bool(summary["scored_artifact_materialized"]),
        "benchmark_eligible": bool(summary["benchmark_eligible"]),
    }


def _terminal_manifest(*, metadata: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "artifact_schema_version": int(metadata.get("artifact_schema_version", 1)),
        "artifacts": {
            "summary": SUMMARY_NAME,
        },
        "resolved_config_fingerprints": metadata.get("resolved_config_fingerprints", {}),
        "model_identity": metadata.get("model_identity", {}),
        "model_identity_fingerprint": metadata.get("model_identity_fingerprint"),
        "processor_identity": metadata.get("processor_identity", {}),
        "adapter_identity": metadata.get("adapter_identity"),
        "embedding_delta_identity": metadata.get("embedding_delta_identity"),
        "tokenizer_identity": metadata.get("tokenizer_identity", {}),
        "backend": metadata.get("backend"),
        "backend_mode": metadata.get("backend_mode"),
        "response_family": metadata.get("response_family"),
        "dataset_identity": metadata.get("dataset_identity", {}),
        "generation_config_fingerprint": metadata.get("generation_config_fingerprint"),
        "generation_policy": dict(metadata.get("generation_policy") or {}),
        "backend_session": dict(metadata.get("backend_session") or {}),
        "likelihood_semantics": dict(metadata.get("likelihood_semantics") or {}),
        "execution_model_identity": metadata.get("execution_model_identity"),
        "frontend_identity": dict(metadata.get("frontend_identity") or {}),
        "raw_model_logprob_status": (
            "failed"
            if metadata.get("raw_model_logprob_enabled")
            else "disabled"
        ),
        "parallelism": dict(metadata.get("parallelism") or {}),
        "score_policy_fingerprint": SCORE_POLICY_FINGERPRINT,
        "trace_scoring_status": "not_materialized",
        "prompt_policy_fingerprint": metadata.get("prompt_policy_fingerprint"),
        "template_identity": metadata.get("template_identity", {}),
        "processor_identity_fingerprint": metadata.get("processor_identity_fingerprint"),
        "evaluator_consumer_status": "not_run",
        "scored_artifact_materialized": False,
        "benchmark_eligible": False,
        "terminal_status": summary.get("terminal_status", "failed"),
        "failure_class": summary.get("failure_class"),
    }


def _verified_selected_logprobs_from_generated_trace(
    *,
    row_id: str,
    object_span_id: str,
    replay: dict[str, Any],
    generated_rows: dict[tuple[str, int], dict[str, Any]],
) -> list[float]:
    steps = replay["generated_step_indices"]
    token_ids = replay["token_ids"]
    token_text = replay["token_text"]
    logprobs = replay["selected_logprobs"]
    selected_count = int(replay["selected_count"])
    if not (
        len(steps)
        == len(token_ids)
        == len(token_text)
        == len(logprobs)
        == selected_count
    ):
        raise ArtifactContractError(
            "selected-token replay fields disagree on selected count",
            code="artifacts.selected_replay_shape_mismatch",
            context={"row_id": row_id, "object_span_id": object_span_id},
        )
    verified: list[float] = []
    for index, step in enumerate(steps):
        generated = generated_rows.get((row_id, int(step)))
        if generated is None:
            raise ArtifactContractError(
                "selected generated-token trace row is missing",
                code="artifacts.generated_trace_missing",
                context={
                    "row_id": row_id,
                    "object_span_id": object_span_id,
                    "generated_step_index": step,
                },
            )
        expected = {
            "token_id": token_ids[index],
            "token_text": token_text[index],
            "logprob": logprobs[index],
        }
        observed = {
            "token_id": generated.get("token_id"),
            "token_text": generated.get("token_text"),
            "logprob": generated.get("logprob"),
        }
        if observed != expected:
            raise ArtifactContractError(
                "selected generated-token trace row disagrees with replay evidence",
                code="artifacts.generated_trace_mismatch",
                context={
                    "row_id": row_id,
                    "object_span_id": object_span_id,
                    "generated_step_index": step,
                    "expected": expected,
                    "observed": observed,
                },
            )
        verified.append(float(generated["logprob"]))
    return verified


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row_index, row in enumerate(rows):
            try:
                line = json.dumps(
                    _json_safe(row),
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            except (TypeError, ValueError) as exc:
                raise ArtifactContractError(
                    "artifact JSONL row is not strict JSON serializable",
                    code="artifacts.strict_json_failed",
                    context={"path": str(path), "row_index": row_index},
                    cause=exc,
                ) from exc
            handle.write(line + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    try:
        text = json.dumps(_json_safe(payload), sort_keys=True, indent=2, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ArtifactContractError(
            "artifact JSON is not strict JSON serializable",
            code="artifacts.strict_json_failed",
            context={"path": str(path)},
            cause=exc,
        ) from exc
    path.write_text(text + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str, allow_nan=False))


def _replace_final_artifacts(staged: InferenceArtifactPaths, final: InferenceArtifactPaths) -> None:
    pairs = [
        (staged.raw_jsonl, final.raw_jsonl),
        (staged.scored_jsonl, final.scored_jsonl),
        (staged.provenance_json, final.provenance_json),
        (staged.token_trace_jsonl, final.token_trace_jsonl),
        (staged.parse_diagnostics_jsonl, final.parse_diagnostics_jsonl),
        (staged.image_plan_jsonl, final.image_plan_jsonl),
        (staged.summary_json, final.summary_json),
        (staged.run_manifest_json, final.run_manifest_json),
    ]
    backup_dir = Path(tempfile.mkdtemp(prefix=".wave5-artifact-backup-", dir=final.output_dir))
    backups: dict[Path, Path | None] = {}
    replaced: list[Path] = []
    try:
        for _, final_path in pairs:
            if final_path.exists():
                backup_path = backup_dir / final_path.name
                os.replace(final_path, backup_path)
                backups[final_path] = backup_path
            else:
                backups[final_path] = None
        for staged_path, final_path in pairs:
            os.replace(staged_path, final_path)
            replaced.append(final_path)
    except OSError as exc:
        for final_path in replaced:
            try:
                if final_path.exists():
                    final_path.unlink()
            except OSError:
                pass
        for final_path, backup_path in backups.items():
            if backup_path is None:
                continue
            try:
                shutil.move(str(backup_path), str(final_path))
            except OSError:
                pass
        raise ArtifactContractError(
            "failed to publish complete inference artifact set",
            code="artifacts.publish_failed",
            context={"failed_after": [path.name for path in replaced]},
            cause=exc,
        ) from exc
    finally:
        shutil.rmtree(backup_dir, ignore_errors=True)


def _replace_terminal_status_artifacts(
    staged: InferenceArtifactPaths,
    final: InferenceArtifactPaths,
) -> None:
    pairs = [
        (staged.summary_json, final.summary_json),
        (staged.run_manifest_json, final.run_manifest_json),
    ]
    backup_dir = Path(tempfile.mkdtemp(prefix=".terminal-status-backup-", dir=final.output_dir))
    backups: dict[Path, Path | None] = {}
    replaced: list[Path] = []
    try:
        for _, final_path in pairs:
            if final_path.exists():
                backup_path = backup_dir / final_path.name
                os.replace(final_path, backup_path)
                backups[final_path] = backup_path
            else:
                backups[final_path] = None
        for staged_path, final_path in pairs:
            os.replace(staged_path, final_path)
            replaced.append(final_path)
    except OSError as exc:
        for final_path in replaced:
            try:
                if final_path.exists():
                    final_path.unlink()
            except OSError:
                pass
        for final_path, backup_path in backups.items():
            if backup_path is None:
                continue
            try:
                shutil.move(str(backup_path), str(final_path))
            except OSError:
                pass
        raise ArtifactContractError(
            "failed to publish complete terminal status artifact pair",
            code="artifacts.terminal_publish_failed",
            context={"failed_after": [path.name for path in replaced]},
            cause=exc,
        ) from exc
    finally:
        shutil.rmtree(backup_dir, ignore_errors=True)
