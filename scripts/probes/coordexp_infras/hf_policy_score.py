#!/usr/bin/env python
"""Attest policy-owned prediction scoring with raw likelihood enabled."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def main() -> int:
    args = _parse_args()
    repo_root = Path.cwd().resolve()
    run_root = Path(args.run_root).resolve()
    manifest = _read_json(run_root / "run_manifest.json")
    summary = _read_json(run_root / "summary.json")
    scored_rows = _read_jsonl(run_root / "gt_vs_pred_scored.jsonl")
    trace_rows = _read_jsonl(run_root / "pred_token_trace.jsonl")
    if manifest.get("raw_model_logprob_status") != "available":
        raise SystemExit("policy-score probe requires available raw likelihood")
    if summary.get("terminal_status") != "completed":
        raise SystemExit("policy-score probe inference did not complete")
    payload_receipt_path = Path(args.execution_payload_receipt).resolve()
    payload_receipt = _read_json(payload_receipt_path)
    execution_payload_identity = payload_receipt.get("runtime", {}).get(
        "execution_payload_identity"
    )
    if not isinstance(execution_payload_identity, dict):
        raise SystemExit("execution-payload receipt is missing composed identity")
    _validate_execution_payload_files(execution_payload_identity)
    _validate_execution_payload_paths(
        manifest=manifest,
        execution_payload_identity=execution_payload_identity,
    )

    generated = {
        (row["row_id"], int(row["generated_step_index"])): row
        for row in trace_rows
        if row.get("trace_type") == "generated_token" and not row.get("is_pad")
    }
    replay = {
        (row["row_id"], row["object_span_id"]): row
        for row in trace_rows
        if row.get("trace_type") == "selected_token_replay"
    }
    checks = []
    for scored_row in scored_rows:
        row_id = scored_row["row_id"]
        for prediction in scored_row["pred"]:
            source = prediction["pred_score_source"]
            key = (row_id, prediction["object_span_id"])
            if replay.get(key) != {
                "trace_type": "selected_token_replay",
                **{
                    field: source[field]
                    for field in (
                        "row_id",
                        "object_span_id",
                        "generated_step_indices",
                        "token_ids",
                        "token_text",
                        "selected_logprobs",
                        "selected_count",
                        "score_policy_fingerprint",
                    )
                },
                "score": prediction["score"],
            }:
                raise SystemExit("selected-token replay disagrees with score source")
            policy = []
            raw = []
            for step_index, expected_token_id in zip(
                source["generated_step_indices"],
                source["token_ids"],
                strict=True,
            ):
                token = generated[(row_id, int(step_index))]
                if token["token_id"] != expected_token_id:
                    raise SystemExit("selected-token replay token id is misaligned")
                policy.append(_finite_nonpositive(token["logprob"], "policy"))
                raw.append(
                    _finite_nonpositive(token["raw_model_logprob"], "raw_model")
                )
            if policy != source["selected_logprobs"]:
                raise SystemExit("prediction source is not backed by policy logprobs")
            policy_score = math.exp(sum(policy) / len(policy))
            raw_score = math.exp(sum(raw) / len(raw))
            if not math.isclose(prediction["score"], policy_score, abs_tol=1e-12):
                raise SystemExit("stored prediction score is not policy-derived")
            checks.append(
                {
                    "row_id": row_id,
                    "object_span_id": prediction["object_span_id"],
                    "stored_score": prediction["score"],
                    "policy_score": policy_score,
                    "raw_counterfactual_score": raw_score,
                    "absolute_policy_raw_score_delta": abs(policy_score - raw_score),
                }
            )
    if not checks:
        raise SystemExit("policy-score probe produced no scoreable predictions")
    if not any(item["absolute_policy_raw_score_delta"] > 0 for item in checks):
        raise SystemExit("policy-score probe did not distinguish policy and raw scores")

    eval_root = run_root / "eval_detection"
    metrics = _read_json(eval_root / "metrics.json")
    evaluation_receipt = _read_json(eval_root / "evaluation_receipt.json")
    if metrics.get("row_count") != len(scored_rows):
        raise SystemExit("evaluator did not consume the scored probe rows")
    if metrics.get("scored_pred_count") != len(checks):
        raise SystemExit("evaluator prediction count disagrees with scored artifacts")

    script_path = Path(__file__).resolve()
    payload = {
        "schema_version": 1,
        "status": "passed",
        "run_root": _display_path(run_root, repo_root),
        "raw_model_logprob_status": "available",
        "execution_payload_evidence": {
            "receipt_path": _display_path(payload_receipt_path, repo_root),
            "receipt_sha256": _sha256_file(payload_receipt_path),
            "fingerprint": execution_payload_identity["fingerprint"],
            "identity": execution_payload_identity,
        },
        "generated_token_count": len(generated),
        "scoreable_prediction_count": len(checks),
        "policy_raw_distinct_prediction_count": sum(
            item["absolute_policy_raw_score_delta"] > 0 for item in checks
        ),
        "maximum_policy_raw_score_delta": max(
            item["absolute_policy_raw_score_delta"] for item in checks
        ),
        "prediction_checks": checks,
        "evaluator": {
            "metrics_path": _display_path(eval_root / "metrics.json", repo_root),
            "metrics_sha256": _sha256_file(eval_root / "metrics.json"),
            "evaluation_receipt_path": _display_path(
                eval_root / "evaluation_receipt.json", repo_root
            ),
            "evaluation_receipt_sha256": _sha256_file(
                eval_root / "evaluation_receipt.json"
            ),
            "metric_family": metrics["metric_family"],
            "row_count": metrics["row_count"],
            "scored_pred_count": metrics["scored_pred_count"],
            "input_artifacts": evaluation_receipt["artifacts"],
        },
        "artifacts": {
            name: _sha256_file(run_root / name)
            for name in (
                "gt_vs_pred.jsonl",
                "gt_vs_pred_scored.jsonl",
                "gt_vs_pred_scored.jsonl.provenance.json",
                "pred_token_trace.jsonl",
                "run_manifest.json",
                "summary.json",
            )
        },
        "source": {
            "verifier": {
                "path": _display_path(script_path, repo_root),
                "sha256": _sha256_file(script_path),
            },
            "files": {
                relative: _sha256_file(repo_root / relative)
                for relative in (
                    "src/eval/detection_consumer.py",
                    "src/inference/artifacts.py",
                    "src/inference/backend.py",
                    "src/inference/scoring.py",
                )
            },
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output.resolve())
    return 0


def _validate_execution_payload_paths(
    *,
    manifest: dict[str, Any],
    execution_payload_identity: dict[str, Any],
) -> None:
    model_identity = manifest.get("model_identity")
    if not isinstance(model_identity, dict):
        raise SystemExit("scored run is missing model identity")
    adapter = model_identity.get("adapter")
    embedding_delta = model_identity.get("embedding_delta")
    if not isinstance(adapter, dict) or not isinstance(embedding_delta, dict):
        raise SystemExit("scored run did not load adapter-plus-delta composition")
    observed_adapter_path = Path(str(adapter.get("adapter_path"))).resolve()
    observed_delta_path = Path(
        str(embedding_delta.get("identity", {}).get("delta_path"))
    ).resolve()
    expected_adapter = execution_payload_identity.get("adapter")
    expected_delta = execution_payload_identity.get("embedding_delta")
    if not isinstance(expected_adapter, dict) or not isinstance(expected_delta, dict):
        raise SystemExit("execution-payload receipt is not adapter-plus-delta")
    if observed_adapter_path != Path(expected_adapter["root"]).resolve():
        raise SystemExit("scored-run adapter path differs from execution receipt")
    if observed_delta_path != Path(expected_delta["root"]).resolve():
        raise SystemExit("scored-run embedding-delta path differs from execution receipt")


def _validate_execution_payload_files(identity: dict[str, Any]) -> None:
    for role, expected_names in (
        ("adapter", {"adapter_config.json", "adapter_model.safetensors"}),
        (
            "embedding_delta",
            {
                "special_token_embeddings.json",
                "special_token_embeddings.safetensors",
            },
        ),
    ):
        component = identity.get(role)
        if not isinstance(component, dict):
            raise SystemExit(f"execution-payload identity is missing {role}")
        files = component.get("files")
        if not isinstance(files, dict) or set(files) != expected_names:
            raise SystemExit(f"execution-payload {role} file set differs")
        for filename, evidence in files.items():
            if not isinstance(evidence, dict):
                raise SystemExit(f"execution-payload {role}/{filename} evidence differs")
            path = Path(str(evidence.get("path"))).resolve()
            if not path.is_file():
                raise SystemExit(f"execution-payload file is missing: {path}")
            if path.stat().st_size != evidence.get("size_bytes"):
                raise SystemExit(f"execution-payload file size differs: {path}")
            if _sha256_file(path) != evidence.get("sha256"):
                raise SystemExit(f"execution-payload file hash differs: {path}")
        determinants = {
            key: value for key, value in component.items() if key != "fingerprint"
        }
        if _sha256_json(determinants) != component.get("fingerprint"):
            raise SystemExit(f"execution-payload {role} fingerprint differs")
    determinants = {
        "adapter": identity["adapter"],
        "embedding_delta": identity["embedding_delta"],
    }
    if _sha256_json(determinants) != identity.get("fingerprint"):
        raise SystemExit("execution-payload combined fingerprint differs")


def _finite_nonpositive(value: Any, channel: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise SystemExit(f"{channel} likelihood is not numeric")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric > 0:
        raise SystemExit(f"{channel} likelihood is not finite and non-positive")
    return numeric


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument(
        "--execution-payload-receipt",
        required=True,
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
