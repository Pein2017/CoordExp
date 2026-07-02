"""Deterministic inference artifact writers for Wave 5 scoring output."""

from __future__ import annotations

import hashlib
import json
import math
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

    raw_rows: list[dict[str, Any]] = []
    scored_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    token_trace_rows: list[dict[str, Any]] = []
    scoreable_prediction_count = 0

    for row in rows:
        parse_row = row["parse"]
        raw_row = _raw_artifact_row(row)
        raw_rows.append(raw_row)
        diagnostic_rows.extend(_diagnostic_rows(parse_row))

        decode_result = decode_results.get(str(row["row_id"]))
        if decode_result is not None:
            token_trace_rows.extend(_token_trace_rows(row_id=str(row["row_id"]), result=decode_result))
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

    _write_jsonl(paths.raw_jsonl, raw_rows)
    _write_jsonl(paths.scored_jsonl, scored_rows)
    _write_jsonl(paths.token_trace_jsonl, token_trace_rows)
    _write_jsonl(paths.parse_diagnostics_jsonl, diagnostic_rows)
    _write_jsonl(paths.image_plan_jsonl, image_plan_rows)

    raw_sha = sha256_file(paths.raw_jsonl)
    scored_sha = sha256_file(paths.scored_jsonl)
    provenance = _provenance(
        metadata=metadata,
        raw_sha=raw_sha,
        scored_sha=scored_sha,
        row_ids=[str(row["row_id"]) for row in rows],
    )
    _write_json(paths.provenance_json, provenance)
    summary = {
        "row_count": len(rows),
        "raw_row_count": len(raw_rows),
        "scored_row_count": len(scored_rows),
        "scoreable_prediction_count": scoreable_prediction_count,
        "diagnostic_row_count": len(diagnostic_rows),
        "trace_row_count": len(token_trace_rows),
        "benchmark_eligible": True,
    }
    _write_json(paths.summary_json, summary)
    _write_json(paths.run_manifest_json, _manifest(metadata=metadata, summary=summary))
    validate_scored_artifact_set(output_dir)
    return paths


def validate_scored_artifact_set(output_dir: Path) -> None:
    required = [
        RAW_NAME,
        SCORED_NAME,
        TOKEN_TRACE_NAME,
        PROVENANCE_NAME,
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
    for row in _read_jsonl(token_trace_jsonl):
        if row.get("trace_type") == "selected_token_replay":
            replay_rows[(row["row_id"], row["object_span_id"])] = row
    recomputed: dict[tuple[str, str], float] = {}
    for row in _read_jsonl(scored_jsonl):
        for pred in row.get("pred", []):
            key = (row["row_id"], pred["object_span_id"])
            replay = replay_rows[key]
            logprobs = replay["selected_logprobs"]
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


def _token_trace_rows(*, row_id: str, result: DecodeResult) -> list[dict[str, Any]]:
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
        "model_identity_fingerprint": metadata["model_identity_fingerprint"],
        "processor_identity_fingerprint": metadata["processor_identity_fingerprint"],
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
        "model_identity_fingerprint": metadata["model_identity_fingerprint"],
        "adapter_identity": metadata.get("adapter_identity"),
        "backend": metadata["backend"],
        "backend_mode": metadata["backend_mode"],
        "response_family": metadata["response_family"],
        "dataset_identity": metadata["dataset_identity"],
        "generation_config_fingerprint": metadata["generation_config_fingerprint"],
        "score_policy_fingerprint": SCORE_POLICY_FINGERPRINT,
        "trace_scoring_status": "scored",
        "prompt_policy_fingerprint": metadata["prompt_policy_fingerprint"],
        "template_identity": metadata["template_identity"],
        "processor_identity_fingerprint": metadata["processor_identity_fingerprint"],
        "evaluator_consumer_status": "not_implemented_wave_5",
        "benchmark_eligible": bool(summary["benchmark_eligible"]),
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_json_safe(row), sort_keys=True, separators=(",", ":")) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_safe(payload), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))
