from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from src.inference.backend_parity import (
    BackendParityInputError,
    build_backend_parity_receipt,
    write_backend_parity_receipt,
)


def test_backend_parity_passes_exact_and_numeric_contract(tmp_path: Path) -> None:
    hf, vllm = _matched_runs(tmp_path, raw=True)

    receipt = build_backend_parity_receipt(
        hf_run_dir=hf,
        vllm_run_dir=vllm,
        require_raw=True,
    )

    assert receipt["status"] == "passed"
    assert receipt["failed_gates"] == []
    assert receipt["metrics"]["policy_logprob_abs_delta"] == {
        "status": "computed",
        "count": 8,
        "median": pytest.approx(0.001),
        "p99": pytest.approx(0.001),
        "max": pytest.approx(0.001),
        "issues": [],
    }
    assert receipt["metrics"]["raw_model_logprob_abs_delta"]["count"] == 8
    assert (
        receipt["metrics"]["per_object_selected_log_score_abs_delta"]["max"]
        == pytest.approx(0.001)
    )


def test_backend_parity_numeric_threshold_failure_is_hold(tmp_path: Path) -> None:
    hf = _write_run(tmp_path / "hf", backend="hf", policy=-1.0, raw=-0.8)
    vllm = _write_run(
        tmp_path / "vllm",
        backend="vllm",
        policy=-1.06,
        raw=-0.8,
    )

    receipt = build_backend_parity_receipt(
        hf_run_dir=hf,
        vllm_run_dir=vllm,
        require_raw=True,
    )

    assert receipt["status"] == "hold"
    assert "policy_logprob_numeric" in receipt["failed_gates"]
    assert receipt["metrics"]["policy_logprob_abs_delta"]["max"] == pytest.approx(
        0.06
    )


def test_backend_parity_exact_generated_id_mismatch_cannot_pass(
    tmp_path: Path,
) -> None:
    hf, vllm = _matched_runs(tmp_path, raw=True)
    trace_path = vllm / "pred_token_trace.jsonl"
    trace = _read_jsonl(trace_path)
    trace[0]["token_id"] += 1
    _write_jsonl(trace_path, trace)

    receipt = build_backend_parity_receipt(
        hf_run_dir=hf,
        vllm_run_dir=vllm,
        require_raw=True,
    )

    assert receipt["status"] == "hold"
    assert "generated_token_ids" in receipt["failed_gates"]
    assert "policy_logprob_numeric" in receipt["failed_gates"]


def test_backend_parity_rejects_generation_contract_mismatch(
    tmp_path: Path,
) -> None:
    hf, vllm = _matched_runs(tmp_path, raw=True)
    manifest_path = vllm / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["generation_config_fingerprint"] = "different-generation"
    _write_json(manifest_path, manifest)

    receipt = build_backend_parity_receipt(
        hf_run_dir=hf,
        vllm_run_dir=vllm,
        require_raw=True,
    )

    assert receipt["status"] == "hold"
    assert "semantic_execution_contract" in receipt["failed_gates"]


def test_backend_parity_ignores_legal_hf_post_stop_padding(
    tmp_path: Path,
) -> None:
    hf, vllm = _matched_runs(tmp_path, raw=True)
    trace_path = hf / "pred_token_trace.jsonl"
    trace = _read_jsonl(trace_path)
    generated = [row for row in trace if row.get("trace_type") == "generated_token"]
    pad = dict(generated[-1])
    pad.update(
        {
            "generated_step_index": len(generated),
            "token_id": 0,
            "token_text": "<|pad|>",
            "logprob": None,
            "raw_model_logprob": None,
            "is_stop": False,
            "is_pad": True,
        }
    )
    trace.insert(len(generated), pad)
    _write_jsonl(trace_path, trace)

    receipt = build_backend_parity_receipt(
        hf_run_dir=hf,
        vllm_run_dir=vllm,
        require_raw=True,
    )

    assert receipt["status"] == "passed"


def test_backend_parity_requires_nonempty_scored_object_population(
    tmp_path: Path,
) -> None:
    hf, vllm = _matched_runs(tmp_path, raw=True)
    for run_dir in (hf, vllm):
        raw_path = run_dir / "gt_vs_pred.jsonl"
        raw_rows = _read_jsonl(raw_path)
        raw_rows[0]["pred"] = []
        _write_jsonl(raw_path, raw_rows)
        scored_path = run_dir / "gt_vs_pred_scored.jsonl"
        scored_rows = _read_jsonl(scored_path)
        scored_rows[0]["pred"] = []
        _write_jsonl(scored_path, scored_rows)
        trace_path = run_dir / "pred_token_trace.jsonl"
        trace_rows = [
            row
            for row in _read_jsonl(trace_path)
            if row.get("trace_type") != "selected_token_replay"
        ]
        _write_jsonl(trace_path, trace_rows)
        provenance_path = run_dir / "gt_vs_pred_scored.jsonl.provenance.json"
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        provenance["raw_artifact"]["sha256"] = _sha256_file(raw_path)
        provenance["scored_artifact"]["sha256"] = _sha256_file(scored_path)
        _write_json(provenance_path, provenance)

    receipt = build_backend_parity_receipt(
        hf_run_dir=hf,
        vllm_run_dir=vllm,
        require_raw=True,
    )

    assert receipt["status"] == "hold"
    assert "per_object_selected_log_score" in receipt["failed_gates"]


def test_backend_parity_raw_requirement_and_missing_enabled_evidence_hold(
    tmp_path: Path,
) -> None:
    disabled_hf, disabled_vllm = _matched_runs(
        tmp_path / "disabled", raw=False
    )
    optional = build_backend_parity_receipt(
        hf_run_dir=disabled_hf,
        vllm_run_dir=disabled_vllm,
        require_raw=False,
    )
    required = build_backend_parity_receipt(
        hf_run_dir=disabled_hf,
        vllm_run_dir=disabled_vllm,
        require_raw=True,
    )

    assert optional["status"] == "passed"
    assert optional["metrics"]["raw_model_logprob_abs_delta"]["status"] == (
        "not_required"
    )
    assert required["status"] == "hold"
    assert "raw_model_logprob_numeric" in required["failed_gates"]

    enabled_hf, enabled_vllm = _matched_runs(tmp_path / "enabled", raw=True)
    trace_path = enabled_vllm / "pred_token_trace.jsonl"
    trace = _read_jsonl(trace_path)
    trace[3]["raw_model_logprob"] = None
    _write_jsonl(trace_path, trace)
    missing = build_backend_parity_receipt(
        hf_run_dir=enabled_hf,
        vllm_run_dir=enabled_vllm,
        require_raw=False,
    )

    assert missing["status"] == "hold"
    assert "raw_model_logprob_numeric" in missing["failed_gates"]
    assert any(
        "missing" in issue
        for issue in missing["metrics"]["raw_model_logprob_abs_delta"]["issues"]
    )


def test_backend_parity_receipt_is_deterministic_and_records_source_hashes(
    tmp_path: Path,
) -> None:
    hf, vllm = _matched_runs(tmp_path, raw=True)
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"

    first = write_backend_parity_receipt(
        hf_run_dir=hf,
        vllm_run_dir=vllm,
        receipt_path=first_path,
        require_raw=True,
    )
    second = write_backend_parity_receipt(
        hf_run_dir=hf,
        vllm_run_dir=vllm,
        receipt_path=second_path,
        require_raw=True,
    )

    assert first == second
    assert first_path.read_bytes() == second_path.read_bytes()
    expected = hashlib.sha256((hf / "gt_vs_pred.jsonl").read_bytes()).hexdigest()
    assert (
        first["sources"]["hf"]["artifacts"]["gt_vs_pred.jsonl"]["sha256"]
        == expected
    )
    assert set(first["sources"]["hf"]["artifacts"]) == {
        "gt_vs_pred.jsonl",
        "gt_vs_pred_scored.jsonl",
        "gt_vs_pred_scored.jsonl.provenance.json",
        "pred_token_trace.jsonl",
        "parse_diagnostics.jsonl",
        "image_plan.jsonl",
        "summary.json",
        "run_manifest.json",
    }


def test_backend_parity_malformed_input_does_not_publish_or_replace_receipt(
    tmp_path: Path,
) -> None:
    hf, vllm = _matched_runs(tmp_path, raw=True)
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text("existing\n", encoding="utf-8")
    (vllm / "gt_vs_pred.jsonl").write_text("{not-json}\n", encoding="utf-8")

    with pytest.raises(BackendParityInputError, match="malformed"):
        write_backend_parity_receipt(
            hf_run_dir=hf,
            vllm_run_dir=vllm,
            receipt_path=receipt_path,
            require_raw=True,
        )

    assert receipt_path.read_text(encoding="utf-8") == "existing\n"
    assert not list(tmp_path.glob(".receipt.json.*.tmp"))


def test_backend_parity_cli_accepts_contract_flags(tmp_path: Path) -> None:
    from scripts.probes.coordexp_swift.backend_parity import main

    hf, vllm = _matched_runs(tmp_path, raw=True)
    receipt_path = tmp_path / "cli-receipt.json"

    exit_code = main(
        [
            "--hf-run-dir",
            str(hf),
            "--vllm-run-dir",
            str(vllm),
            "--receipt",
            str(receipt_path),
            "--require-raw",
        ]
    )

    assert exit_code == 0
    assert json.loads(receipt_path.read_text(encoding="utf-8"))["status"] == "passed"


def test_backend_parity_ignores_boolean_composition_prompt_check(
    tmp_path: Path,
) -> None:
    hf, vllm = _matched_runs(tmp_path, raw=True)
    provenance_path = vllm / "gt_vs_pred_scored.jsonl.provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["execution_model_identity"] = {
        "composition_fidelity": {
            "receipt": {"comparison": {"composition_checks": {"prompt_ids": True}}}
        }
    }
    provenance_path.write_text(
        json.dumps(provenance, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    receipt = build_backend_parity_receipt(
        hf_run_dir=hf,
        vllm_run_dir=vllm,
        require_raw=True,
    )

    assert receipt["status"] == "passed"


def test_backend_parity_rejects_completed_runs_without_prompt_identity(
    tmp_path: Path,
) -> None:
    hf, vllm = _matched_runs(tmp_path, raw=True)
    provenance_path = hf / "gt_vs_pred_scored.jsonl.provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance.pop("prompt_trace")
    provenance_path.write_text(
        json.dumps(provenance, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(BackendParityInputError, match="no exact prompt-token evidence"):
        build_backend_parity_receipt(
            hf_run_dir=hf,
            vllm_run_dir=vllm,
            require_raw=True,
        )


def test_backend_parity_ignores_rank_local_parser_runtime_metadata(
    tmp_path: Path,
) -> None:
    hf, vllm = _matched_runs(tmp_path, raw=True)
    diagnostic = {
        "row_id": "row-0",
        "row_index": 0,
        "parser_id": "compact-object-box-closed-v1",
        "parse_status": "accepted",
        "valid_prediction_count": 1,
        "dropped_prediction_count": 0,
        "dropped_predictions": [],
    }
    _write_jsonl(hf / "parse_diagnostics.jsonl", [diagnostic])
    vllm_diagnostic = dict(diagnostic)
    vllm_diagnostic.update(
        {
            "rank": 0,
            "world_size": 1,
            "worker_cuda_visible_devices": "0",
            "worker_logical_device": "cuda:0",
        }
    )
    _write_jsonl(vllm / "parse_diagnostics.jsonl", [vllm_diagnostic])

    receipt = build_backend_parity_receipt(
        hf_run_dir=hf,
        vllm_run_dir=vllm,
        require_raw=True,
    )

    assert receipt["status"] == "passed"


def _matched_runs(root: Path, *, raw: bool) -> tuple[Path, Path]:
    raw_value = -0.8 if raw else None
    return (
        _write_run(root / "hf", backend="hf", policy=-1.0, raw=raw_value),
        _write_run(
            root / "vllm",
            backend="vllm",
            policy=-1.001,
            raw=None if raw_value is None else -0.801,
        ),
    )


def _write_run(
    run_dir: Path,
    *,
    backend: str,
    policy: float,
    raw: float | None,
) -> Path:
    run_dir.mkdir(parents=True)
    row_id = "row-0"
    object_span_id = "row-0:object:0"
    token_ids = list(range(101, 109))
    token_text = [
        "<|object_ref_start|>",
        "cat",
        "<|object_ref_end|>",
        "<|box_start|>",
        "<|coord_100|>",
        "<|coord_200|>",
        "<|coord_300|>",
        "<|box_end|>",
    ]
    raw_text = "".join(token_text)
    raw_rows = [
        {
            "row_id": row_id,
            "row_index": 0,
            "example_id": row_id,
            "image_path": "fixture.png",
            "image_width": 640,
            "image_height": 480,
            "gt": [{"description": "cat", "bbox": [100, 200, 300, 400]}],
            "pred": [
                {
                    "object_span_id": object_span_id,
                    "description": "cat",
                    "bbox": [64.0, 96.0, 192.0, 192.0],
                }
            ],
            "raw_decode_text": raw_text,
            "decode_stop_reason": "im_end",
            "parser_id": "compact-object-box-closed-v1",
            "parser_policy": "compact_object_box_closed_only",
            "metric_bearing": True,
            "parse_status": "accepted",
            "valid_prediction_count": 1,
            "dropped_prediction_count": 0,
            "dropped_predictions": [],
        }
    ]
    score = math.exp(policy)
    scored_rows = [
        {
            "row_id": row_id,
            "row_index": 0,
            "example_id": row_id,
            "image_path": "fixture.png",
            "image_width": 640,
            "image_height": 480,
            "gt": raw_rows[0]["gt"],
            "pred": [
                {
                    "object_span_id": object_span_id,
                    "description": "cat",
                    "bbox": [64.0, 96.0, 192.0, 192.0],
                    "score": score,
                    "pred_score_version": 1,
                    "pred_score_source": {
                        "kind": "token_trace_selected_logprob_mean",
                        "row_id": row_id,
                        "object_span_id": object_span_id,
                        "generated_step_indices": list(range(8)),
                        "token_ids": token_ids,
                        "token_text": token_text,
                        "selected_logprobs": [policy] * 8,
                        "selected_count": 8,
                        "score_policy_fingerprint": "score-policy",
                    },
                }
            ],
        }
    ]
    raw_status = "available" if raw is not None else "disabled"
    generated_trace = [
        {
            "trace_type": "generated_token",
            "row_id": row_id,
            "generated_step_index": index,
            "token_id": token_id,
            "token_text": text,
            "logprob": policy,
            "raw_model_logprob": raw,
            "raw_model_logprob_status": raw_status,
            "is_stop": index == 7,
            "is_pad": False,
            "backend": backend,
            "backend_mode": "generate" if backend == "hf" else "offline_generate",
            "response_family": backend,
        }
        for index, (token_id, text) in enumerate(
            zip(token_ids, token_text, strict=True)
        )
    ]
    trace_rows = [
        *generated_trace,
        {
            "trace_type": "selected_token_replay",
            "row_id": row_id,
            "object_span_id": object_span_id,
            "generated_step_indices": list(range(8)),
            "token_ids": token_ids,
            "token_text": token_text,
            "selected_logprobs": [policy] * 8,
            "selected_count": 8,
            "score": score,
            "score_policy_fingerprint": "score-policy",
        },
    ]
    image_rows = [
        {
            "row_id": row_id,
            "row_index": 0,
            "example_id": row_id,
            "image_path": "fixture.png",
            "declared_width": 640,
            "declared_height": 480,
            "decoded_width": 640,
            "decoded_height": 480,
            "image_content_sha256": "a" * 64,
            "executed_media_sha256": "b" * 64,
            "logical_transform_id": "identity",
            "expected_image_grid_thw": [1, 30, 40],
            "merged_visual_tokens": 300,
            "backend_projection_evidence_kind": (
                "hf_executed_tensors" if backend == "hf" else "vllm_executed_media"
            ),
        }
    ]
    _write_jsonl(run_dir / "gt_vs_pred.jsonl", raw_rows)
    _write_jsonl(run_dir / "gt_vs_pred_scored.jsonl", scored_rows)
    _write_jsonl(run_dir / "pred_token_trace.jsonl", trace_rows)
    _write_jsonl(run_dir / "parse_diagnostics.jsonl", [])
    _write_jsonl(run_dir / "image_plan.jsonl", image_rows)

    row_ids_sha = hashlib.sha256(b'["row-0"]').hexdigest()
    provenance = {
        "artifact_schema_version": 1,
        "raw_artifact": {
            "path": "gt_vs_pred.jsonl",
            "sha256": _sha256_file(run_dir / "gt_vs_pred.jsonl"),
        },
        "scored_artifact": {
            "path": "gt_vs_pred_scored.jsonl",
            "sha256": _sha256_file(run_dir / "gt_vs_pred_scored.jsonl"),
        },
        "row_binding": {"row_count": 1, "row_ids_sha256": row_ids_sha},
        "raw_model_logprob_status": raw_status,
        "prompt_trace": [
            {
                "row_id": row_id,
                "executed_prompt_token_ids": [11, 12, 13],
            }
        ],
    }
    _write_json(run_dir / "gt_vs_pred_scored.jsonl.provenance.json", provenance)
    _write_json(
        run_dir / "summary.json",
        {
            "terminal_status": "completed",
            "scored_artifact_materialized": True,
            "row_count": 1,
        },
    )
    _write_json(
        run_dir / "run_manifest.json",
        {
            "backend": backend,
            "dataset_identity": "dataset-identity",
            "generation_config_fingerprint": "generation-config",
            "generation_policy": {"temperature": 0.0, "top_p": 1.0},
            "prompt_policy_fingerprint": "prompt-policy",
            "template_identity": {"assistant_format": "coordexp"},
            "parser_policy": {"parser": "coordexp"},
            "score_policy_fingerprint": "score-policy",
            "processor_identity_fingerprint": "processor-identity",
            "raw_model_logprob_status": raw_status,
            "scored_artifact_materialized": True,
        },
    )
    return run_dir


def test_backend_parity_rejects_missing_semantic_execution_identity(
    tmp_path: Path,
) -> None:
    hf, vllm = _matched_runs(tmp_path, raw=True)
    manifest_path = hf / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("template_identity")
    _write_json(manifest_path, manifest)

    with pytest.raises(BackendParityInputError, match="template_identity"):
        build_backend_parity_receipt(
            hf_run_dir=hf,
            vllm_run_dir=vllm,
            require_raw=True,
        )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
