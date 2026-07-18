"""Read-only HF/vLLM inference artifact parity receipts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


RECEIPT_VERSION = "coordexp-swift-backend-parity-v1"
POLICY_MEDIAN_THRESHOLD = 0.002
POLICY_P99_THRESHOLD = 0.02
POLICY_MAX_THRESHOLD = 0.05
OBJECT_LOG_SCORE_THRESHOLD = 0.01
QUANTILE_METHOD = "linear interpolation with h=(n-1)q (Hyndman-Fan type 7)"

RAW_NAME = "gt_vs_pred.jsonl"
SCORED_NAME = "gt_vs_pred_scored.jsonl"
PROVENANCE_NAME = "gt_vs_pred_scored.jsonl.provenance.json"
TOKEN_TRACE_NAME = "pred_token_trace.jsonl"
PARSE_DIAGNOSTICS_NAME = "parse_diagnostics.jsonl"
IMAGE_PLAN_NAME = "image_plan.jsonl"
SUMMARY_NAME = "summary.json"
MANIFEST_NAME = "run_manifest.json"

_JSONL_NAMES = (
    RAW_NAME,
    SCORED_NAME,
    TOKEN_TRACE_NAME,
    PARSE_DIAGNOSTICS_NAME,
    IMAGE_PLAN_NAME,
)
_JSON_NAMES = (PROVENANCE_NAME, SUMMARY_NAME, MANIFEST_NAME)
_SOURCE_NAMES = (*_JSONL_NAMES, *_JSON_NAMES)
_PROMPT_ID_FIELDS = {
    "prompt_token_ids",
    "input_prompt_token_ids",
    "expected_executed_prompt_token_ids",
    "executed_prompt_token_ids",
}


class BackendParityInputError(ValueError):
    """A source run is not a well-formed completed inference artifact set."""


def build_backend_parity_receipt(
    *,
    hf_run_dir: str | Path,
    vllm_run_dir: str | Path,
    require_raw: bool = False,
) -> dict[str, Any]:
    """Validate two completed runs and return a deterministic parity receipt.

    Exact mismatches and numeric threshold failures produce a ``hold`` receipt.
    Filesystem, JSON, or source-artifact schema corruption raises
    :class:`BackendParityInputError` so callers cannot publish a receipt for
    malformed input.
    """

    if not isinstance(require_raw, bool):
        raise BackendParityInputError("require_raw must be boolean")
    hf = _load_run(Path(hf_run_dir), role="hf")
    vllm = _load_run(Path(vllm_run_dir), role="vllm")
    gates: list[dict[str, Any]] = []

    _boolean_gate(
        gates,
        "backend_roles",
        hf.manifest.get("backend") == "hf"
        and vllm.manifest.get("backend") == "vllm",
        {
            "expected": {"hf": "hf", "vllm": "vllm"},
            "observed": {
                "hf": hf.manifest.get("backend"),
                "vllm": vllm.manifest.get("backend"),
            },
        },
    )
    _exact_gate(
        gates,
        "ordered_row_request_identity",
        _ordered_row_identity(hf),
        _ordered_row_identity(vllm),
    )
    _exact_gate(
        gates,
        "image_gt_identity",
        _image_gt_identity(hf),
        _image_gt_identity(vllm),
    )
    _exact_gate(
        gates,
        "prompt_token_ids",
        _prompt_id_evidence(hf),
        _prompt_id_evidence(vllm),
    )
    generated_ids_equal = _exact_gate(
        gates,
        "generated_token_ids",
        _generated_token_identity(hf),
        _generated_token_identity(vllm),
    )
    _exact_gate(
        gates,
        "stop_reason",
        _stop_identity(hf),
        _stop_identity(vllm),
    )
    _exact_gate(
        gates,
        "parser_output",
        _parser_output(hf),
        _parser_output(vllm),
    )
    scored_structure_equal = _exact_gate(
        gates,
        "scored_rows",
        _scored_structure(hf.scored_rows),
        _scored_structure(vllm.scored_rows),
    )

    policy_metrics, policy_ok = _likelihood_metrics(
        hf,
        vllm,
        channel="policy_logprob",
        aligned=generated_ids_equal,
    )
    _boolean_gate(
        gates,
        "policy_logprob_numeric",
        policy_ok,
        _numeric_gate_details(policy_metrics),
    )

    raw_metrics, raw_ok = _raw_likelihood_metrics(
        hf,
        vllm,
        require_raw=require_raw,
        aligned=generated_ids_equal,
    )
    _boolean_gate(
        gates,
        "raw_model_logprob_numeric",
        raw_ok,
        _numeric_gate_details(raw_metrics),
    )

    object_metrics, object_ok = _object_log_score_metrics(
        hf,
        vllm,
        aligned=scored_structure_equal,
    )
    _boolean_gate(
        gates,
        "per_object_selected_log_score",
        object_ok,
        {
            "count": object_metrics["count"],
            "max": object_metrics["max"],
            "threshold": OBJECT_LOG_SCORE_THRESHOLD,
            "issues": object_metrics["issues"],
        },
    )

    failed_gates = [gate["name"] for gate in gates if gate["status"] == "failed"]
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "receipt_version": RECEIPT_VERSION,
        "status": "hold" if failed_gates else "passed",
        "failed_gates": failed_gates,
        "policy": {
            "require_raw": require_raw,
            "policy_artifact_field": "logprob",
            "policy_semantic_channel": "policy_logprob",
            "raw_artifact_field": "raw_model_logprob",
            "quantile_method": QUANTILE_METHOD,
            "thresholds": {
                "likelihood_abs_delta": {
                    "median": POLICY_MEDIAN_THRESHOLD,
                    "p99": POLICY_P99_THRESHOLD,
                    "max": POLICY_MAX_THRESHOLD,
                },
                "per_object_selected_log_score_abs_delta": (
                    OBJECT_LOG_SCORE_THRESHOLD
                ),
            },
            "scored_row_exact_projection": {
                "excluded_numeric_fields": [
                    "pred[*].score",
                    "pred[*].pred_score_source.selected_logprobs",
                ]
            },
        },
        "sources": {
            "hf": hf.source_identity(),
            "vllm": vllm.source_identity(),
        },
        "gates": gates,
        "metrics": {
            "policy_logprob_abs_delta": policy_metrics,
            "raw_model_logprob_abs_delta": raw_metrics,
            "per_object_selected_log_score_abs_delta": object_metrics,
        },
    }
    receipt["digest"] = _sha256_json(receipt)
    return receipt


def write_backend_parity_receipt(
    *,
    hf_run_dir: str | Path,
    vllm_run_dir: str | Path,
    receipt_path: str | Path,
    require_raw: bool = False,
) -> dict[str, Any]:
    """Build and atomically publish a parity receipt."""

    receipt = build_backend_parity_receipt(
        hf_run_dir=hf_run_dir,
        vllm_run_dir=vllm_run_dir,
        require_raw=require_raw,
    )
    _atomic_write_json(Path(receipt_path), receipt)
    return receipt


class _RunArtifacts:
    def __init__(
        self,
        *,
        role: str,
        run_dir: Path,
        hashes: dict[str, dict[str, Any]],
        jsonl: dict[str, list[dict[str, Any]]],
        json_objects: dict[str, dict[str, Any]],
    ) -> None:
        self.role = role
        self.run_dir = run_dir
        self.hashes = hashes
        self.jsonl = jsonl
        self.json_objects = json_objects
        self.raw_rows = jsonl[RAW_NAME]
        self.scored_rows = jsonl[SCORED_NAME]
        self.trace_rows = jsonl[TOKEN_TRACE_NAME]
        self.parse_diagnostics = jsonl[PARSE_DIAGNOSTICS_NAME]
        self.image_plan_rows = jsonl[IMAGE_PLAN_NAME]
        self.provenance = json_objects[PROVENANCE_NAME]
        self.summary = json_objects[SUMMARY_NAME]
        self.manifest = json_objects[MANIFEST_NAME]

    def source_identity(self) -> dict[str, Any]:
        return {
            "run_dir": str(self.run_dir),
            "artifacts": self.hashes,
        }


def _load_run(run_dir: Path, *, role: str) -> _RunArtifacts:
    run_dir = run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        _malformed(role, f"run directory does not exist: {run_dir}")
    hashes: dict[str, dict[str, Any]] = {}
    raw_bytes: dict[str, bytes] = {}
    for name in _SOURCE_NAMES:
        path = run_dir / name
        if not path.is_file():
            _malformed(role, f"missing required artifact: {name}")
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise BackendParityInputError(
                f"{role}: cannot read {path}: {exc}"
            ) from exc
        raw_bytes[name] = payload
        hashes[name] = {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "byte_size": len(payload),
        }

    jsonl = {
        name: _parse_jsonl(raw_bytes[name], role=role, artifact=name)
        for name in _JSONL_NAMES
    }
    json_objects = {
        name: _parse_json_object(raw_bytes[name], role=role, artifact=name)
        for name in _JSON_NAMES
    }
    run = _RunArtifacts(
        role=role,
        run_dir=run_dir,
        hashes=hashes,
        jsonl=jsonl,
        json_objects=json_objects,
    )
    _validate_completed_run(run)
    return run


def _validate_completed_run(run: _RunArtifacts) -> None:
    if run.summary.get("scored_artifact_materialized") is not True:
        _malformed(run.role, "summary does not describe a completed scored run")
    terminal_status = run.summary.get("terminal_status")
    if terminal_status is not None and terminal_status != "completed":
        _malformed(run.role, f"summary terminal_status is {terminal_status!r}")
    if run.manifest.get("scored_artifact_materialized") is not True:
        _malformed(run.role, "manifest does not describe a completed scored run")

    raw_binding = _require_mapping(
        run.provenance.get("raw_artifact"), run.role, "provenance.raw_artifact"
    )
    scored_binding = _require_mapping(
        run.provenance.get("scored_artifact"),
        run.role,
        "provenance.scored_artifact",
    )
    if raw_binding.get("sha256") != run.hashes[RAW_NAME]["sha256"]:
        _malformed(run.role, "raw artifact SHA-256 disagrees with provenance")
    if scored_binding.get("sha256") != run.hashes[SCORED_NAME]["sha256"]:
        _malformed(run.role, "scored artifact SHA-256 disagrees with provenance")

    raw_identity = _validated_row_identity(run.raw_rows, run.role, RAW_NAME)
    scored_identity = _validated_row_identity(
        run.scored_rows, run.role, SCORED_NAME
    )
    image_identity = _validated_row_identity(
        run.image_plan_rows, run.role, IMAGE_PLAN_NAME
    )
    if raw_identity != scored_identity or raw_identity != image_identity:
        _malformed(run.role, "raw, scored, and image-plan row identity/order differ")

    row_ids = [item["row_id"] for item in raw_identity]
    row_binding = run.provenance.get("row_binding")
    if row_binding is not None:
        binding = _require_mapping(
            row_binding, run.role, "provenance.row_binding"
        )
        if binding.get("row_count") != len(row_ids):
            _malformed(run.role, "provenance row count is inconsistent")
        expected = hashlib.sha256(
            json.dumps(row_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if binding.get("row_ids_sha256") != expected:
            _malformed(run.role, "provenance row-id binding is inconsistent")
    _validate_trace_schema(run, set(row_ids))


def _validated_row_identity(
    rows: Sequence[Mapping[str, Any]], role: str, artifact: str
) -> list[dict[str, Any]]:
    identity: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        row_id = _require_string(row.get("row_id"), role, f"{artifact}[{index}].row_id")
        row_index = row.get("row_index")
        if isinstance(row_index, bool) or not isinstance(row_index, int):
            _malformed(role, f"{artifact}[{index}].row_index must be an integer")
        example_id = _require_string(
            row.get("example_id"), role, f"{artifact}[{index}].example_id"
        )
        if row_id in seen:
            _malformed(role, f"{artifact} contains duplicate row_id {row_id!r}")
        seen.add(row_id)
        identity.append(
            {"row_id": row_id, "row_index": row_index, "example_id": example_id}
        )
    return identity


def _validate_trace_schema(run: _RunArtifacts, row_ids: set[str]) -> None:
    generated_by_row: dict[str, list[int]] = {row_id: [] for row_id in row_ids}
    generated_keys: set[tuple[str, int]] = set()
    replay_keys: set[tuple[str, str]] = set()
    for index, row in enumerate(run.trace_rows):
        trace_type = row.get("trace_type")
        row_id = _require_string(
            row.get("row_id"), run.role, f"{TOKEN_TRACE_NAME}[{index}].row_id"
        )
        if row_id not in row_ids:
            _malformed(run.role, f"trace row references unknown row_id {row_id!r}")
        if trace_type == "generated_token":
            step = row.get("generated_step_index")
            token_id = row.get("token_id")
            if isinstance(step, bool) or not isinstance(step, int) or step < 0:
                _malformed(run.role, "generated_step_index must be a non-negative integer")
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                _malformed(run.role, "generated token_id must be an integer")
            key = (row_id, step)
            if key in generated_keys:
                _malformed(run.role, f"duplicate generated-token trace key {key!r}")
            generated_keys.add(key)
            generated_by_row[row_id].append(step)
            if not isinstance(row.get("is_stop"), bool) or not isinstance(
                row.get("is_pad"), bool
            ):
                _malformed(run.role, "generated trace stop/pad flags must be boolean")
        elif trace_type == "selected_token_replay":
            object_span_id = _require_string(
                row.get("object_span_id"),
                run.role,
                f"{TOKEN_TRACE_NAME}[{index}].object_span_id",
            )
            key = (row_id, object_span_id)
            if key in replay_keys:
                _malformed(run.role, f"duplicate selected replay trace key {key!r}")
            replay_keys.add(key)
        else:
            _malformed(run.role, f"unknown trace_type {trace_type!r}")
    for row_id, steps in generated_by_row.items():
        if steps != list(range(len(steps))):
            _malformed(run.role, f"generated steps are not contiguous for {row_id!r}")


def _ordered_row_identity(run: _RunArtifacts) -> list[dict[str, Any]]:
    return _validated_row_identity(run.raw_rows, run.role, RAW_NAME)


def _image_gt_identity(run: _RunArtifacts) -> list[dict[str, Any]]:
    plans = {str(row["row_id"]): row for row in run.image_plan_rows}
    result = []
    for row in run.raw_rows:
        row_id = str(row["row_id"])
        plan = plans[row_id]
        result.append(
            {
                "row_id": row_id,
                "example_id": row["example_id"],
                "image_path": row.get("image_path"),
                "image_width": row.get("image_width"),
                "image_height": row.get("image_height"),
                "gt": row.get("gt"),
                "image_content_sha256": plan.get("image_content_sha256"),
                "executed_media_sha256": plan.get("executed_media_sha256"),
                "declared_width": plan.get("declared_width"),
                "declared_height": plan.get("declared_height"),
                "decoded_width": plan.get("decoded_width"),
                "decoded_height": plan.get("decoded_height"),
                "logical_transform_id": plan.get("logical_transform_id"),
                "expected_image_grid_thw": plan.get("expected_image_grid_thw"),
                "merged_visual_tokens": plan.get("merged_visual_tokens"),
            }
        )
    return result


def _prompt_id_evidence(run: _RunArtifacts) -> list[dict[str, Any]]:
    prompt_trace = run.provenance.get("prompt_trace")
    if (
        isinstance(prompt_trace, list)
        and prompt_trace
        and isinstance(prompt_trace[0], Mapping)
        and any(
            str(key).endswith("_token_ids_sha256")
            for key in prompt_trace[0]
        )
    ):
        if not isinstance(prompt_trace, list) or not prompt_trace:
            _malformed(run.role, "provenance prompt_trace must be a non-empty list")
        normalized: list[dict[str, Any]] = []
        seen: set[str] = set()
        for index, row in enumerate(prompt_trace):
            mapping = _require_mapping(
                row,
                run.role,
                f"provenance.prompt_trace[{index}]",
            )
            row_id = _require_string(
                mapping.get("row_id"),
                run.role,
                f"provenance.prompt_trace[{index}].row_id",
            )
            if row_id in seen:
                _malformed(run.role, f"prompt_trace repeats row_id {row_id!r}")
            seen.add(row_id)
            fields: dict[str, Any] = {"row_id": row_id}
            for prefix in (
                "input_prompt",
                "expected_executed_prompt",
                "backend_executed_prompt",
            ):
                count_key = f"{prefix}_token_count"
                digest_key = f"{prefix}_token_ids_sha256"
                count = mapping.get(count_key)
                digest = mapping.get(digest_key)
                if (
                    isinstance(count, bool)
                    or not isinstance(count, int)
                    or count <= 0
                ):
                    _malformed(run.role, f"{count_key} must be a positive integer")
                if (
                    not isinstance(digest, str)
                    or len(digest) != 64
                    or any(char not in "0123456789abcdef" for char in digest)
                ):
                    _malformed(run.role, f"{digest_key} must be a SHA-256 digest")
                fields[count_key] = count
                fields[digest_key] = digest
            if (
                fields["expected_executed_prompt_token_count"]
                != fields["backend_executed_prompt_token_count"]
                or fields["expected_executed_prompt_token_ids_sha256"]
                != fields["backend_executed_prompt_token_ids_sha256"]
                or mapping.get("prompt_token_parity") != "verified"
            ):
                _malformed(run.role, f"prompt parity is not verified for {row_id!r}")
            normalized.append(fields)
        expected_row_ids = [str(row["row_id"]) for row in run.raw_rows]
        if [row["row_id"] for row in normalized] != expected_row_ids:
            _malformed(run.role, "prompt_trace row identity/order differs from raw rows")
        return normalized

    evidence: list[dict[str, Any]] = []
    _collect_prompt_ids(
        run.provenance, artifact=PROVENANCE_NAME, owner=None, out=evidence
    )
    _collect_prompt_ids(
        run.trace_rows, artifact=TOKEN_TRACE_NAME, owner=None, out=evidence
    )
    if not evidence:
        _malformed(run.role, "completed run has no exact prompt-token evidence")
    return evidence


def _collect_prompt_ids(
    value: Any,
    *,
    artifact: str,
    owner: str | None,
    out: list[dict[str, Any]],
) -> None:
    if isinstance(value, Mapping):
        local_owner = owner
        for owner_key in ("request_id", "row_id"):
            if isinstance(value.get(owner_key), str):
                local_owner = str(value[owner_key])
                break
        for key, child in value.items():
            if key in _PROMPT_ID_FIELDS or key.endswith("_prompt_token_ids"):
                if not isinstance(child, list) or any(
                    isinstance(item, bool) or not isinstance(item, int)
                    for item in child
                ):
                    raise BackendParityInputError(
                        f"{artifact}: prompt token ids at {key!r} must be integer lists"
                    )
                out.append(
                    {
                        "artifact": artifact,
                        "owner": local_owner,
                        "field": key,
                        "token_ids": child,
                    }
                )
            else:
                _collect_prompt_ids(
                    child,
                    artifact=artifact,
                    owner=local_owner,
                    out=out,
                )
    elif isinstance(value, list):
        for child in value:
            _collect_prompt_ids(
                child, artifact=artifact, owner=owner, out=out
            )


def _generated_rows(run: _RunArtifacts) -> list[dict[str, Any]]:
    return [
        row for row in run.trace_rows if row.get("trace_type") == "generated_token"
    ]


def _generated_token_identity(run: _RunArtifacts) -> list[dict[str, Any]]:
    return [
        {
            "row_id": row["row_id"],
            "generated_step_index": row["generated_step_index"],
            "token_id": row["token_id"],
        }
        for row in _generated_rows(run)
    ]


def _stop_identity(run: _RunArtifacts) -> list[dict[str, Any]]:
    stop_steps: dict[str, list[int]] = {}
    for row in _generated_rows(run):
        if row["is_stop"]:
            stop_steps.setdefault(str(row["row_id"]), []).append(
                int(row["generated_step_index"])
            )
    return [
        {
            "row_id": row["row_id"],
            "decode_stop_reason": row.get("decode_stop_reason"),
            "stop_step_indices": stop_steps.get(str(row["row_id"]), []),
        }
        for row in run.raw_rows
    ]


def _parser_output(run: _RunArtifacts) -> dict[str, Any]:
    fields = (
        "row_id",
        "row_index",
        "pred",
        "raw_decode_text",
        "parser_id",
        "parser_policy",
        "metric_bearing",
        "parse_status",
        "valid_prediction_count",
        "dropped_prediction_count",
        "dropped_predictions",
    )
    diagnostic_fields = (
        "row_id",
        "row_index",
        "parser_id",
        "parse_status",
        "valid_prediction_count",
        "dropped_prediction_count",
        "dropped_predictions",
    )
    return {
        "rows": [{field: row.get(field) for field in fields} for row in run.raw_rows],
        "parse_diagnostics": [
            {field: row.get(field) for field in diagnostic_fields}
            for row in run.parse_diagnostics
        ],
    }


def _scored_structure(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    projected = json.loads(json.dumps(rows, allow_nan=False))
    for row in projected:
        for pred in row.get("pred", []):
            pred.pop("score", None)
            source = pred.get("pred_score_source")
            if isinstance(source, dict):
                source.pop("selected_logprobs", None)
    return projected


def _likelihood_metrics(
    hf: _RunArtifacts,
    vllm: _RunArtifacts,
    *,
    channel: str,
    aligned: bool,
) -> tuple[dict[str, Any], bool]:
    if not aligned:
        metrics = _empty_numeric_metrics(
            status="not_evaluated", issues=["generated token alignment failed"]
        )
        return metrics, False
    hf_rows = _generated_rows(hf)
    vllm_rows = _generated_rows(vllm)
    if len(hf_rows) != len(vllm_rows):
        metrics = _empty_numeric_metrics(
            status="not_evaluated", issues=["generated trace row counts differ"]
        )
        return metrics, False

    deltas: list[float] = []
    issues: list[str] = []
    for hf_row, vllm_row in zip(hf_rows, vllm_rows, strict=True):
        if bool(hf_row["is_pad"]) != bool(vllm_row["is_pad"]):
            issues.append(_trace_label(hf_row) + ": pad flags differ")
            continue
        if hf_row["is_pad"]:
            continue
        hf_value, hf_issue = _channel_value(hf_row, channel)
        vllm_value, vllm_issue = _channel_value(vllm_row, channel)
        if hf_issue:
            issues.append("hf " + _trace_label(hf_row) + ": " + hf_issue)
        if vllm_issue:
            issues.append("vllm " + _trace_label(vllm_row) + ": " + vllm_issue)
        if hf_issue or vllm_issue:
            continue
        assert hf_value is not None and vllm_value is not None
        deltas.append(abs(hf_value - vllm_value))
    metrics = _numeric_metrics(deltas, issues=issues)
    passed = (
        not issues
        and metrics["count"] > 0
        and metrics["median"] <= POLICY_MEDIAN_THRESHOLD
        and metrics["p99"] <= POLICY_P99_THRESHOLD
        and metrics["max"] <= POLICY_MAX_THRESHOLD
    )
    return metrics, passed


def _raw_likelihood_metrics(
    hf: _RunArtifacts,
    vllm: _RunArtifacts,
    *,
    require_raw: bool,
    aligned: bool,
) -> tuple[dict[str, Any], bool]:
    hf_status, hf_issues = _raw_status(hf)
    vllm_status, vllm_issues = _raw_status(vllm)
    issues = [*("hf: " + item for item in hf_issues), *("vllm: " + item for item in vllm_issues)]
    if hf_status == "disabled" and vllm_status == "disabled" and not require_raw:
        metrics = _empty_numeric_metrics(status="not_required", issues=issues)
        metrics["availability"] = {"hf": hf_status, "vllm": vllm_status}
        return metrics, not issues
    if hf_status != "available" or vllm_status != "available":
        if require_raw:
            issues.append("raw likelihood is required but not available in both runs")
        else:
            issues.append("raw likelihood availability differs between runs")
        metrics = _empty_numeric_metrics(status="not_evaluated", issues=issues)
        metrics["availability"] = {"hf": hf_status, "vllm": vllm_status}
        return metrics, False
    metrics, passed = _likelihood_metrics(
        hf, vllm, channel="raw_model_logprob", aligned=aligned
    )
    metrics["availability"] = {"hf": hf_status, "vllm": vllm_status}
    if issues:
        metrics["issues"] = [*issues, *metrics["issues"]]
        passed = False
    return metrics, passed


def _raw_status(run: _RunArtifacts) -> tuple[str | None, list[str]]:
    statuses = [
        value
        for value in (
            run.provenance.get("raw_model_logprob_status"),
            run.manifest.get("raw_model_logprob_status"),
        )
        if value is not None
    ]
    trace_statuses = {
        row.get("raw_model_logprob_status") for row in _generated_rows(run)
    }
    statuses.extend(sorted(value for value in trace_statuses if value is not None))
    unique = set(statuses)
    issues: list[str] = []
    if len(unique) != 1:
        issues.append(f"raw likelihood status disagrees across artifacts: {sorted(unique)!r}")
        return None, issues
    status = next(iter(unique), None)
    if status not in {"available", "disabled"}:
        issues.append(f"unsupported raw likelihood status {status!r}")
        return status, issues
    has_value = any(
        row.get("raw_model_logprob") is not None for row in _generated_rows(run)
    )
    if status == "disabled" and has_value:
        issues.append("disabled raw channel contains values")
    return status, issues


def _channel_value(
    row: Mapping[str, Any], channel: str
) -> tuple[float | None, str | None]:
    if channel == "policy_logprob":
        explicit = row.get("policy_logprob")
        compatibility = row.get("logprob")
        if explicit is not None and compatibility is not None and explicit != compatibility:
            return None, "policy_logprob and compatibility logprob disagree"
        value = explicit if explicit is not None else compatibility
    else:
        value = row.get("raw_model_logprob")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None, f"{channel} is missing or non-numeric"
    numeric = float(value)
    if not math.isfinite(numeric):
        return None, f"{channel} is non-finite"
    if numeric > 0.0:
        return None, f"{channel} is positive"
    return numeric, None


def _object_log_score_metrics(
    hf: _RunArtifacts,
    vllm: _RunArtifacts,
    *,
    aligned: bool,
) -> tuple[dict[str, Any], bool]:
    if not aligned:
        return {
            "status": "not_evaluated",
            "count": 0,
            "max": None,
            "objects": [],
            "issues": ["scored row structure failed exact comparison"],
        }, False
    objects: list[dict[str, Any]] = []
    issues: list[str] = []
    for hf_row, vllm_row in zip(hf.scored_rows, vllm.scored_rows, strict=True):
        for hf_pred, vllm_pred in zip(
            hf_row.get("pred", []), vllm_row.get("pred", []), strict=True
        ):
            row_id = str(hf_row["row_id"])
            object_span_id = str(hf_pred.get("object_span_id"))
            hf_log_score, hf_issue = _authoritative_log_score(hf_pred)
            vllm_log_score, vllm_issue = _authoritative_log_score(vllm_pred)
            if hf_issue:
                issues.append(f"hf {row_id}/{object_span_id}: {hf_issue}")
            if vllm_issue:
                issues.append(f"vllm {row_id}/{object_span_id}: {vllm_issue}")
            if hf_issue or vllm_issue:
                continue
            assert hf_log_score is not None and vllm_log_score is not None
            objects.append(
                {
                    "row_id": row_id,
                    "object_span_id": object_span_id,
                    "hf_log_score": hf_log_score,
                    "vllm_log_score": vllm_log_score,
                    "abs_delta": abs(hf_log_score - vllm_log_score),
                }
            )
    max_delta = max((item["abs_delta"] for item in objects), default=0.0)
    metrics = {
        "status": "computed" if not issues else "invalid_evidence",
        "count": len(objects),
        "max": max_delta,
        "objects": objects,
        "issues": issues,
    }
    return metrics, not issues and max_delta <= OBJECT_LOG_SCORE_THRESHOLD


def _authoritative_log_score(
    pred: Mapping[str, Any],
) -> tuple[float | None, str | None]:
    score = pred.get("score")
    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(float(score))
        or float(score) <= 0.0
        or float(score) > 1.0
    ):
        return None, "evaluator-authoritative score must be finite in (0, 1]"
    source = pred.get("pred_score_source")
    if not isinstance(source, Mapping):
        return None, "pred_score_source is missing"
    if source.get("kind") != "token_trace_selected_logprob_mean":
        return None, "pred_score_source is not policy selected-token likelihood"
    selected = source.get("selected_logprobs")
    if not isinstance(selected, list) or not selected:
        return None, "selected policy logprobs are missing"
    values: list[float] = []
    for value in selected:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) > 0.0
        ):
            return None, "selected policy logprobs are invalid"
        values.append(float(value))
    if source.get("selected_count") != len(values):
        return None, "selected_count disagrees with selected policy evidence"
    log_score = math.log(float(score))
    selected_mean = sum(values) / len(values)
    if not math.isclose(log_score, selected_mean, rel_tol=1e-12, abs_tol=1e-12):
        return None, "score does not equal exp(mean(selected policy logprobs))"
    return log_score, None


def _numeric_metrics(
    deltas: Sequence[float], *, issues: Sequence[str]
) -> dict[str, Any]:
    if not deltas:
        return _empty_numeric_metrics(
            status="invalid_evidence" if issues else "computed", issues=list(issues)
        )
    ordered = sorted(float(value) for value in deltas)
    return {
        "status": "invalid_evidence" if issues else "computed",
        "count": len(ordered),
        "median": _linear_quantile(ordered, 0.5),
        "p99": _linear_quantile(ordered, 0.99),
        "max": ordered[-1],
        "issues": list(issues),
    }


def _empty_numeric_metrics(*, status: str, issues: Sequence[str]) -> dict[str, Any]:
    return {
        "status": status,
        "count": 0,
        "median": None,
        "p99": None,
        "max": None,
        "issues": list(issues),
    }


def _linear_quantile(ordered: Sequence[float], q: float) -> float:
    if not ordered:
        raise ValueError("quantile requires at least one value")
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] + fraction * (ordered[upper] - ordered[lower]))


def _exact_gate(
    gates: list[dict[str, Any]],
    name: str,
    hf_value: Any,
    vllm_value: Any,
    *,
    absent_is_pass: bool = False,
) -> bool:
    passed = hf_value == vllm_value
    details: dict[str, Any] = {
        "hf_sha256": _sha256_json(hf_value),
        "vllm_sha256": _sha256_json(vllm_value),
    }
    if absent_is_pass and not hf_value and not vllm_value:
        details["evidence_status"] = "absent_in_both_runs"
    elif not passed:
        details["first_difference"] = _first_difference(hf_value, vllm_value)
    _boolean_gate(gates, name, passed, details)
    return passed


def _boolean_gate(
    gates: list[dict[str, Any]], name: str, passed: bool, details: Mapping[str, Any]
) -> None:
    gates.append(
        {"name": name, "status": "passed" if passed else "failed", "details": dict(details)}
    )


def _numeric_gate_details(metrics: Mapping[str, Any]) -> dict[str, Any]:
    details = {
        "status": metrics.get("status"),
        "count": metrics.get("count"),
        "median": metrics.get("median"),
        "p99": metrics.get("p99"),
        "max": metrics.get("max"),
        "thresholds": {
            "median": POLICY_MEDIAN_THRESHOLD,
            "p99": POLICY_P99_THRESHOLD,
            "max": POLICY_MAX_THRESHOLD,
        },
        "issues": metrics.get("issues", []),
    }
    if "availability" in metrics:
        details["availability"] = metrics["availability"]
    return details


def _first_difference(hf_value: Any, vllm_value: Any, path: str = "$") -> dict[str, Any]:
    if type(hf_value) is not type(vllm_value):
        return {"path": path, "hf": _brief(hf_value), "vllm": _brief(vllm_value)}
    if isinstance(hf_value, Mapping):
        hf_keys = list(hf_value)
        vllm_keys = list(vllm_value)
        if hf_keys != vllm_keys:
            return {"path": path, "hf_keys": hf_keys, "vllm_keys": vllm_keys}
        for key in hf_keys:
            if hf_value[key] != vllm_value[key]:
                return _first_difference(hf_value[key], vllm_value[key], f"{path}.{key}")
    elif isinstance(hf_value, list):
        if len(hf_value) != len(vllm_value):
            return {"path": path, "hf_length": len(hf_value), "vllm_length": len(vllm_value)}
        for index, (hf_item, vllm_item) in enumerate(zip(hf_value, vllm_value, strict=True)):
            if hf_item != vllm_item:
                return _first_difference(hf_item, vllm_item, f"{path}[{index}]")
    return {"path": path, "hf": _brief(hf_value), "vllm": _brief(vllm_value)}


def _brief(value: Any) -> Any:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    if len(encoded) <= 256:
        return value
    return {"sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(), "length": len(encoded)}


def _trace_label(row: Mapping[str, Any]) -> str:
    return f"{row.get('row_id')}/{row.get('generated_step_index')}"


def _parse_jsonl(payload: bytes, *, role: str, artifact: str) -> list[dict[str, Any]]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise BackendParityInputError(f"{role}: {artifact} is not UTF-8") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            _malformed(role, f"{artifact}:{line_number} is blank")
        value = _loads_json(line, role=role, artifact=f"{artifact}:{line_number}")
        if not isinstance(value, dict):
            _malformed(role, f"{artifact}:{line_number} must contain a JSON object")
        rows.append(value)
    return rows


def _parse_json_object(payload: bytes, *, role: str, artifact: str) -> dict[str, Any]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise BackendParityInputError(f"{role}: {artifact} is not UTF-8") from exc
    value = _loads_json(text, role=role, artifact=artifact)
    if not isinstance(value, dict):
        _malformed(role, f"{artifact} must contain a JSON object")
    return value


def _loads_json(text: str, *, role: str, artifact: str) -> Any:
    try:
        return json.loads(
            text,
            parse_constant=lambda value: (_raise_nonfinite(value)),
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise BackendParityInputError(f"{role}: malformed {artifact}: {exc}") from exc


def _raise_nonfinite(value: str) -> Any:
    raise ValueError(f"non-finite JSON number {value}")


def _require_mapping(value: Any, role: str, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _malformed(role, f"{field} must be an object")
    return value


def _require_string(value: Any, role: str, field: str) -> str:
    if not isinstance(value, str) or not value:
        _malformed(role, f"{field} must be a non-empty string")
    return value


def _malformed(role: str, message: str) -> None:
    raise BackendParityInputError(f"{role}: {message}")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    serialized = json.dumps(
        payload,
        sort_keys=True,
        indent=2,
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
