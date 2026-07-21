from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.probes.coordexp_swift import vllm_concurrency as probe
from src.common.errors import RuntimeContractError


EXECUTION_MODEL = {
    "mode": "base_only",
    "model_path": "/fake/model",
    "composition_key": "a" * 64,
    "snapshot_fingerprint": "b" * 64,
    "receipt_fingerprint": "c" * 64,
    "source_identity": {"base": {"fingerprint": "base-fingerprint"}},
}


class FakeSession:
    def __init__(self, calls: list[str]) -> None:
        self._calls = calls

    def close(self) -> None:
        self._calls.append("session_close")


def test_cli_requires_config_output_dir_and_receipt() -> None:
    with pytest.raises(SystemExit):
        probe.parse_args([])
    with pytest.raises(SystemExit):
        probe.parse_args(["--config", "infer.yaml", "--output-dir", "out"])


@pytest.mark.parametrize(
    ("backend", "batch_size", "smoke", "dry_run", "row_count", "code"),
    [
        ("hf", 2, True, False, 2, "vllm_concurrency.backend"),
        ("vllm", 0, True, False, 1, "vllm_concurrency.batch_size"),
        ("vllm", 2, False, False, 2, "vllm_concurrency.smoke"),
        ("vllm", 2, True, True, 2, "vllm_concurrency.smoke"),
        ("vllm", 3, True, False, 2, "vllm_concurrency.input_row_count"),
    ],
)
def test_probe_rejects_invalid_config_or_input_scope(
    tmp_path: Path,
    backend: str,
    batch_size: int,
    smoke: bool,
    dry_run: bool,
    row_count: int,
    code: str,
) -> None:
    resolved = _resolved_config(
        tmp_path,
        backend=backend,
        batch_size=batch_size,
        smoke=smoke,
        dry_run=dry_run,
    )
    calls: list[str] = []
    dependencies = _fake_dependencies(
        resolved=resolved,
        row_count=row_count,
        calls=calls,
    )
    receipt_path = tmp_path / "qualification.json"

    with pytest.raises(RuntimeContractError) as exc_info:
        probe.run_probe(
            config_path=resolved.entry_config_path,
            output_dir=tmp_path / "artifacts",
            receipt_path=receipt_path,
            dependencies=dependencies,
        )

    assert exc_info.value.code == code
    assert not receipt_path.exists()
    assert "run_shard" not in calls


def test_failed_run_never_writes_passed_receipt(tmp_path: Path) -> None:
    resolved = _resolved_config(tmp_path)
    calls: list[str] = []
    dependencies = _fake_dependencies(
        resolved=resolved,
        row_count=2,
        calls=calls,
        fail_run=True,
    )
    receipt_path = tmp_path / "qualification.json"

    with pytest.raises(RuntimeError, match="injected pipeline failure"):
        probe.run_probe(
            config_path=resolved.entry_config_path,
            output_dir=tmp_path / "artifacts",
            receipt_path=receipt_path,
            dependencies=dependencies,
        )

    assert calls == ["resolve_execution_model", "session_open", "run_shard"]
    assert not receipt_path.exists()


def test_passed_receipt_construction_is_deterministic_from_fake_artifacts(
    tmp_path: Path,
) -> None:
    resolved = _resolved_config(tmp_path)
    calls: list[str] = []
    dependencies = _fake_dependencies(
        resolved=resolved,
        row_count=2,
        calls=calls,
    )

    first = probe.run_probe(
        config_path=resolved.entry_config_path,
        output_dir=tmp_path / "artifacts-a",
        receipt_path=tmp_path / "receipt-a.json",
        dependencies=dependencies,
    )
    second = probe.run_probe(
        config_path=resolved.entry_config_path,
        output_dir=tmp_path / "artifacts-b",
        receipt_path=tmp_path / "receipt-b.json",
        dependencies=dependencies,
    )

    assert first == second
    assert first["status"] == "passed"
    assert first["schema_version"] == 1
    assert first["version"] == probe.RECEIPT_VERSION
    assert first["vllm_version"] == "0.14.1"
    assert first["max_num_seqs"] == 2
    assert first["request_ids"] == ["row-0", "row-1"]
    assert first["input_source"]["row_count"] == 2
    assert first["input_source"]["selected_prefix_row_count"] == 2
    assert [row["generated_token_ids"] for row in first["requests"]] == [
        [100, 101],
        [110, 111],
    ]
    assert [row["stop_reason"] for row in first["requests"]] == [
        "im_end",
        "length",
    ]
    assert first["raw_model_logprob_status"] == "available"
    assert first["execution_model"]["snapshot_fingerprint"] == "b" * 64
    assert first["backend_session_engine_kwargs"]["max_num_seqs"] == 2
    assert [row["row_id"] for row in first["prompt_trace"]["rows"]] == [
        "row-0",
        "row-1",
    ]
    assert [row["row_id"] for row in first["raw_replay"]["rows"]] == [
        "row-0",
        "row-1",
    ]
    assert first["raw_replay"]["settings"]["status"] == "completed"
    assert (
        first["execution_model"]["source_base_snapshot_fingerprint"]
        == "base-fingerprint"
    )
    assert first["process_cuda_binding"]["cuda"]["logical_device"] == "cuda:0"
    assert all(len(item["sha256"]) == 64 for item in first["artifacts"])
    assert (
        json.loads((tmp_path / "receipt-a.json").read_text(encoding="utf-8")) == first
    )
    assert (
        json.loads((tmp_path / "receipt-b.json").read_text(encoding="utf-8")) == second
    )
    assert calls.count("session_open") == 2
    assert calls.count("run_shard") == 2


def _resolved_config(
    tmp_path: Path,
    *,
    backend: str = "vllm",
    batch_size: int = 2,
    smoke: bool = True,
    dry_run: bool = False,
) -> Any:
    config_path = tmp_path / "infer.yaml"
    config_path.write_text("schema_version: 1\n", encoding="utf-8")
    input_jsonl = tmp_path / "fixture.jsonl"
    input_jsonl.write_text(
        json.dumps({"example_id": "row-0"})
        + "\n"
        + json.dumps({"example_id": "row-1"})
        + "\n",
        encoding="utf-8",
    )
    return SimpleNamespace(
        config=SimpleNamespace(
            backend=SimpleNamespace(type=backend),
            generation=SimpleNamespace(batch_size=batch_size),
            debug=SimpleNamespace(smoke=smoke, dry_run=dry_run),
            data=SimpleNamespace(input_jsonl=str(input_jsonl)),
        ),
        fingerprint="d" * 64,
        entry_config_path=config_path.resolve(),
        sources=(
            SimpleNamespace(
                path=config_path.resolve(),
                sha256=_sha256(config_path),
            ),
        ),
    )


def _fake_dependencies(
    *,
    resolved: Any,
    row_count: int,
    calls: list[str],
    fail_run: bool = False,
) -> probe.ProbeDependencies:
    rows = tuple(
        SimpleNamespace(example_id=f"row-{index}") for index in range(row_count)
    )

    def resolve_execution_model(_: Any) -> dict[str, Any]:
        calls.append("resolve_execution_model")
        return dict(EXECUTION_MODEL)

    def write_resolved_config(_: Any, output_dir: Path) -> None:
        config_dir = output_dir / "configs"
        config_dir.mkdir(parents=True, exist_ok=True)
        (config_dir / "resolved.json").write_text(
            json.dumps({"fingerprint": resolved.fingerprint}, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def write_execution_model(*, run_dir: Path, execution_model: Any) -> None:
        (run_dir / "execution_model.json").write_text(
            json.dumps(execution_model, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def open_session(_: Any) -> FakeSession:
        calls.append("session_open")
        return FakeSession(calls)

    def run_shard(**kwargs: Any) -> int:
        session = kwargs["session_opener"](SimpleNamespace(backend="vllm"))
        assert isinstance(session, FakeSession)
        assert kwargs["row_indices"] == tuple(range(row_count))
        assert kwargs["execution_model"] == EXECUTION_MODEL
        if fail_run:
            calls.append("run_shard")
            raise RuntimeError("injected pipeline failure")
        _write_fake_artifacts(
            Path(kwargs["output_dir"]),
            request_ids=[row.example_id for row in rows],
            execution_model=kwargs["execution_model"],
        )
        calls.append("run_shard")
        session.close()
        return 0

    return probe.ProbeDependencies(
        load_config=lambda _: resolved,
        load_rows=lambda _: rows,
        resolve_execution_model=resolve_execution_model,
        validate_execution_model=lambda value: dict(value),
        write_resolved_config_artifacts=write_resolved_config,
        write_execution_model_artifact=write_execution_model,
        run_shard=run_shard,
        validate_artifact_set=lambda _: None,
        session_opener=open_session,
        inspect_process_cuda_binding=lambda: {
            "process": {"pid": 123, "ppid": 45},
            "cuda": {
                "CUDA_VISIBLE_DEVICES": "7",
                "visible_cuda_tokens": ["7"],
                "cuda_available": True,
                "cuda_device_count": 1,
                "cuda_current_device": 0,
                "logical_device": "cuda:0",
                "device_name": "fake-gpu",
                "torch_cuda_version": "12.8",
            },
        },
        package_version=lambda name: "0.14.1" if name == "vllm" else "unknown",
    )


def _write_fake_artifacts(
    output_dir: Path,
    *,
    request_ids: list[str],
    execution_model: dict[str, Any],
) -> None:
    raw_rows = [
        {
            "row_id": request_id,
            "row_index": index,
            "decode_stop_reason": "im_end" if index == 0 else "length",
        }
        for index, request_id in enumerate(request_ids)
    ]
    trace_rows = [
        {
            "trace_type": "generated_token",
            "row_id": request_id,
            "generated_step_index": step,
            "token_id": 100 + index * 10 + step,
            "logprob": -0.1,
            "raw_model_logprob": -0.2,
        }
        for index, request_id in enumerate(request_ids)
        for step in range(2)
    ]
    image_plan_rows = [
        {
            "row_id": request_id,
            "row_index": index,
            "image_content_sha256": f"{index + 1:064x}",
            "executed_media_sha256": f"{index + 11:064x}",
            "expected_image_grid_thw": [1, 4, 6],
            "backend_prompt_token_count": 31 + index,
            "backend_image_placeholder_ranges": [{"offset": 8, "length": 6}],
        }
        for index, request_id in enumerate(request_ids)
    ]
    summary = {
        "terminal_status": "completed",
        "row_count": len(request_ids),
        "raw_model_logprob_status": "available",
    }
    runtime_qualification = {
        "status": "production_baseline_passed_concurrency_under_probe",
        "qualified_baseline_max_num_seqs": 1,
        "probed_max_num_seqs": len(request_ids),
        "override_scope": "max_num_seqs_only",
    }
    prompt_trace = [
        {
            "row_id": request_id,
            "input_prompt_token_count": 2,
            "input_prompt_token_ids_sha256": hashlib.sha256(
                f"input:{request_id}".encode()
            ).hexdigest(),
            "expected_executed_prompt_token_count": 4,
            "expected_executed_prompt_token_ids_sha256": hashlib.sha256(
                f"prompt:{request_id}".encode()
            ).hexdigest(),
            "backend_executed_prompt_token_count": 4,
            "backend_executed_prompt_token_ids_sha256": hashlib.sha256(
                f"prompt:{request_id}".encode()
            ).hexdigest(),
            "prompt_token_parity": "verified",
        }
        for request_id in request_ids
    ]
    raw_replay_trace = [
        {
            "row_id": request_id,
            "status": "verified",
            "prompt_token_count": 4,
            "prompt_token_ids_sha256": hashlib.sha256(
                f"prompt:{request_id}".encode()
            ).hexdigest(),
            "generated_token_count": 2,
            "generated_token_ids_sha256": hashlib.sha256(
                f"generated:{request_id}".encode()
            ).hexdigest(),
            "finish_reason": "stop" if index == 0 else "length",
            "native_stop_reason": None,
        }
        for index, request_id in enumerate(request_ids)
    ]
    raw_replay_by_id = {
        row["row_id"]: {
            key: value for key, value in row.items() if key != "row_id"
        }
        for row in raw_replay_trace
    }
    manifest = {
        "backend": "vllm",
        "scored_artifact_materialized": True,
        "raw_model_logprob_status": "available",
        "backend_session": {
            "backend_version": "0.14.1",
            "execution_model_identity": execution_model,
            "effective_settings": {
                "batch_size": len(request_ids),
                "engine_kwargs": {
                    "max_num_seqs": len(request_ids),
                    "tensor_parallel_size": 1,
                    "data_parallel_size": 1,
                },
                "runtime_qualification": runtime_qualification,
                "raw_replay": {
                    "status": "completed",
                    "logprobs_mode": "raw_logprobs",
                    "max_num_seqs": len(request_ids),
                    "forced_logits_processor": {
                        "module": "src.inference.vllm_forced_replay",
                        "qualname": "CoordExpForcedSequenceLogitsProcessor",
                        "source_path": "/repo/src/inference/vllm_forced_replay.py",
                        "source_sha256": "f" * 64,
                    },
                    "qualification": {
                        "status": "qualification_probe_under_renewal",
                        "probe_source_sha256": _sha256(Path(probe.__file__)),
                        "source_base_snapshot_fingerprint": "base-fingerprint",
                        "processor_source_sha256": "f" * 64,
                    },
                    "request_count": len(request_ids),
                    "row_evidence_sha256": _sha256_json(raw_replay_by_id),
                },
            },
        },
        "prompt_trace": prompt_trace,
        "raw_replay_trace": raw_replay_trace,
    }
    _write_jsonl(output_dir / "gt_vs_pred.jsonl", raw_rows)
    _write_jsonl(output_dir / "pred_token_trace.jsonl", trace_rows)
    _write_jsonl(output_dir / "image_plan.jsonl", image_plan_rows)
    (output_dir / "gt_vs_pred_scored.jsonl").write_text("", encoding="utf-8")
    (output_dir / "parse_diagnostics.jsonl").write_text("", encoding="utf-8")
    (output_dir / "gt_vs_pred_scored.jsonl.provenance.json").write_text(
        "{}\n", encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_json(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
