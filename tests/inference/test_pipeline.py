from __future__ import annotations

import ast
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import yaml
from PIL import Image

from src.common.errors import (
    ArtifactContractError,
    EncodingContractError,
    RuntimeContractError,
)
from src.inference.backend import (
    BackendLaunch,
    BackendSessionReceipt,
    DecodeResult,
    LikelihoodPair,
    TokenTrace,
)
from src.inference.runtime import InferenceFrontend
from src.qwen.loading import QwenProcessorIdentity


OBJECT_TEXT = (
    "<|object_ref_start|>cat<|object_ref_end|>"
    "<|box_start|><|coord_100|><|coord_200|><|coord_300|><|coord_400|><|box_end|>"
)


class FakeTokenizer:
    image_pad_id = 151655

    def __call__(self, text: str, *, add_special_tokens: bool = False, **_: Any) -> dict[str, list[int]]:
        assert add_special_tokens is False
        ids: list[int] = []
        marker = "<|image_pad|>"
        while marker in text:
            prefix, text = text.split(marker, 1)
            ids.extend(ord(char) for char in prefix)
            ids.append(self.image_pad_id)
        ids.extend(ord(char) for char in text)
        return {"input_ids": ids}

    def convert_tokens_to_ids(self, token: str) -> int:
        if token == "<|image_pad|>":
            return self.image_pad_id
        raise KeyError(token)


class FakeProcessor:
    def __init__(self) -> None:
        self.tokenizer = FakeTokenizer()
        self.image_processor = FakeImageProcessor()

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        **_: Any,
    ) -> str | list[int]:
        pieces: list[str] = []
        for message in messages:
            pieces.append(f"<|im_start|>{message['role']}\n")
            for item in message["content"]:
                if item["type"] == "image":
                    pieces.append("<|vision_start|><|image_pad|><|vision_end|>")
                elif item["type"] == "text":
                    pieces.append(item["text"])
            pieces.append("<|im_end|>\n")
        if add_generation_prompt:
            pieces.append("<|im_start|>assistant\n")
        text = "".join(pieces)
        if tokenize:
            return [ord(char) for char in text]
        return text


class FakeImageProcessor:
    def __call__(self, **kwargs: Any) -> dict[str, torch.Tensor]:
        image_count = len(kwargs.get("images") or [None])
        return {
            "image_grid_thw": torch.tensor([[1, 4, 6]] * image_count, dtype=torch.long),
            "pixel_values": torch.zeros((24 * image_count, 1536), dtype=torch.float32),
        }


class FakeBackend:
    def __init__(
        self,
        calls: list[list[str]],
        *,
        launch: BackendLaunch,
        model_identity: dict[str, Any] | None = None,
    ) -> None:
        self.calls = calls
        self.launch = launch
        self.closed = False
        self.receipt = BackendSessionReceipt(
            backend="hf",
            backend_mode="generate",
            response_family="hf",
            backend_version="test-transformers",
            model_identity=dict(
                model_identity
                or {"family": "unit", "base": {"path": "fake-qwen"}}
            ),
            tokenizer_identity={"sha256": "tok"},
            processor_identity={
                "processor_class": "FakeQwen3VLProcessor",
                "tokenizer_class": "FakeTokenizer",
                "image_processor_class": "FakeQwen2VLImageProcessorFast",
                "patch_size": 16,
                "merge_size": 2,
                "temporal_patch_size": 2,
            },
            generation_config_fingerprint=launch.generation_config_fingerprint,
            effective_settings={"batch_size": launch.batch_size, "device": "cuda:0"},
            likelihood_semantics={
                "policy": "fp32_log_softmax_after_active_generation_processors",
                "raw": "fp32_log_softmax_unmodified_lm_head_logits",
                "score_owned_channel": "policy_logprob",
            },
        )

    def decode(self, requests: list[Any]) -> list[DecodeResult]:
        results: list[DecodeResult] = []
        for start in range(0, len(requests), self.launch.batch_size):
            results.extend(
                self.generate_batch(requests[start : start + self.launch.batch_size])
            )
        return results

    def close(self) -> None:
        self.closed = True

    def generate_batch(
        self,
        requests: list[Any],
        **_: Any,
    ) -> list[DecodeResult]:
        self.calls.append([request.request_id for request in requests])
        for request in requests:
            assert not hasattr(request, "model_inputs")
            assert request.image_sha256
            assert request.chat_text
            assert request.logical_transform_id in {
                "identity",
                "hflip",
                "vflip",
                "hvflip",
            }
        return [
            _decode_result(
                request.request_id,
                prompt_token_ids=list(request.prompt_token_ids),
                observed_image_grid_thw=request.expected_image_grid_thw,
            )
            for request in requests
        ]


@pytest.fixture(autouse=True)
def _fake_visible_cuda_for_non_dry_pipeline_unit_tests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.inference import data_parallel

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(data_parallel, "detect_cuda_device_count", lambda: 1)


def test_pipeline_orchestrates_batched_decode_and_artifact_writing(tmp_path: Path) -> None:
    from src.inference import pipeline

    config_path = _write_config(tmp_path, batch_size=2, row_count=3)
    backend_calls: list[list[str]] = []

    result = pipeline.run(
        config_path=config_path,
        frontend_factory=_frontend_factory(),
        session_opener=lambda launch: FakeBackend(backend_calls, launch=launch),
    )

    run_dir = tmp_path / "outputs" / "wave6-pipeline"
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    raw_rows = _read_jsonl(run_dir / "gt_vs_pred.jsonl")
    scored_rows = _read_jsonl(run_dir / "gt_vs_pred_scored.jsonl")
    image_plan_rows = _read_jsonl(run_dir / "image_plan.jsonl")

    assert result == 0
    assert backend_calls == [["row-0", "row-1"], ["row-2"]]
    assert [row["row_id"] for row in raw_rows] == ["row-0", "row-1", "row-2"]
    assert [row["row_id"] for row in scored_rows] == ["row-0", "row-1", "row-2"]
    assert image_plan_rows[0]["logical_transform_id"] == "identity"
    assert image_plan_rows[0]["image_content_sha256"]
    assert image_plan_rows[0]["backend_projection_evidence_kind"] == "hf_executed_tensors"
    assert image_plan_rows[0]["executed_media_sha256"] == "a" * 64
    assert summary["row_count"] == 3
    assert summary["decode_success_count"] == 3
    assert summary["parser_failure_count"] == 0
    assert summary["truncated_decode_count"] == 3
    assert summary["decode_stop_reasons"] == {"length": 3}
    assert summary["generation_policy"]["repetition_penalty"] == pytest.approx(1.0)
    assert summary["generation_policy"]["do_sample"] is False
    assert summary["terminal_status"] == "completed"
    assert manifest["trace_scoring_status"] == "scored"
    assert manifest["backend"] == "hf"
    assert manifest["generation_policy"]["repetition_penalty"] == pytest.approx(1.0)
    assert manifest["parallelism"]["execution_mode"] == "direct_single_process"
    assert manifest["parallelism"]["plan"]["active_ranks"] == 1
    assert manifest["parallelism"]["plan"]["per_device_batch_size"] == 2
    assert manifest["parallelism"]["plan"]["decode_batch_count"] == 2
    assert manifest["parallelism"]["direct_runtime"]["logical_device"] == "cuda:0"
    assert manifest["parallelism"]["direct_runtime"]["visible_cuda_token_count"] == 1
    assert manifest["parallelism"]["direct_runtime"]["active_ranks"] == 1
    assert (
        manifest["parallelism"]["direct_runtime"]["model_first_parameter_device"]
        == "cuda:0"
    )
    assert manifest["evaluator_consumer_status"] == "available_not_run"
    assert (run_dir / "configs" / "resolved.json").is_file()
    provenance = json.loads(
        (run_dir / "gt_vs_pred_scored.jsonl.provenance.json").read_text(
            encoding="utf-8"
        )
    )
    assert provenance["generation_policy"]["max_new_tokens"] == 64
    assert provenance["parallelism"] == manifest["parallelism"]
    assert provenance["prompt_trace"] == manifest["prompt_trace"]
    assert [row["row_id"] for row in provenance["prompt_trace"]] == [
        "row-0",
        "row-1",
        "row-2",
    ]
    for prompt_trace in provenance["prompt_trace"]:
        assert prompt_trace["prompt_token_parity"] == "verified"
        assert (
            prompt_trace["expected_executed_prompt_token_ids_sha256"]
            == prompt_trace["backend_executed_prompt_token_ids_sha256"]
        )
        assert (
            prompt_trace["expected_executed_prompt_token_count"]
            == prompt_trace["backend_executed_prompt_token_count"]
        )
    assert raw_rows[0]["decode_stop_reason"] == "length"


def test_pipeline_keeps_tiny_non_smoke_run_out_of_benchmark_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.config import inference as inference_config
    from src.inference import pipeline

    monkeypatch.setattr(
        inference_config,
        "_validate_canonical_namespace",
        lambda config, entry_path: None,
    )

    config_path = _write_config(
        tmp_path,
        batch_size=2,
        row_count=2,
        smoke=False,
    )

    result = pipeline.run(
        config_path=config_path,
        frontend_factory=_frontend_factory(),
        session_opener=lambda launch: FakeBackend([], launch=launch),
    )

    run_dir = tmp_path / "outputs" / "wave6-pipeline"
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (run_dir / "run_manifest.json").read_text(encoding="utf-8")
    )

    assert result == 0
    assert summary["terminal_status"] == "completed"
    assert summary["benchmark_eligible"] is False
    assert manifest["benchmark_eligible"] is False


def test_benchmark_scope_requires_non_smoke_val200_or_larger() -> None:
    from src.inference.pipeline import _benchmark_scope_eligible

    assert (
        _benchmark_scope_eligible(
            config=SimpleNamespace(debug=SimpleNamespace(smoke=False)),
            row_count=200,
        )
        is True
    )
    assert (
        _benchmark_scope_eligible(
            config=SimpleNamespace(debug=SimpleNamespace(smoke=False)),
            row_count=199,
        )
        is False
    )
    assert (
        _benchmark_scope_eligible(
            config=SimpleNamespace(debug=SimpleNamespace(smoke=True)),
            row_count=200,
        )
        is False
    )


def test_pipeline_keeps_direct_path_when_only_one_active_rank(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.inference import data_parallel
    from src.inference import pipeline

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setattr(data_parallel, "detect_cuda_device_count", lambda: 2)
    config_path = _write_config(tmp_path, batch_size=4, row_count=2)

    def fail_worker_launcher(**kwargs: Any) -> object:
        raise AssertionError("worker launcher must not be used for active_ranks == 1")

    result = pipeline.run(
        config_path=config_path,
        frontend_factory=_frontend_factory(),
        session_opener=lambda launch: FakeBackend([], launch=launch),
        worker_launcher=fail_worker_launcher,
    )

    run_dir = tmp_path / "outputs" / "wave6-pipeline"
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    raw_rows = _read_jsonl(run_dir / "gt_vs_pred.jsonl")

    assert result == 0
    assert manifest["parallelism"]["execution_mode"] == "direct_single_process"
    assert manifest["parallelism"]["plan"]["active_ranks"] == 1
    assert [row["row_id"] for row in raw_rows] == ["row-0", "row-1"]


def test_pipeline_uses_controller_workers_and_merges_when_active_ranks_exceeds_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.config.inference import load_infer_config
    from src.inference import data_parallel
    from src.inference import pipeline

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setattr(data_parallel, "detect_cuda_device_count", lambda: 2)
    config_path = _write_config(tmp_path, batch_size=1, row_count=2)
    resolved = load_infer_config(config_path)
    expected_plan = data_parallel.plan_data_parallel_shards(
        row_ids=("row-0", "row-1"),
        per_device_batch_size=1,
        visible_cuda_tokens=("0", "1"),
    )
    launched: list[dict[str, Any]] = []

    class FakeProcess:
        def __init__(self) -> None:
            self.returncode: int | None = None

        def wait(self) -> int:
            self.returncode = 0
            return 0

    def fake_worker_launcher(**kwargs: Any) -> FakeProcess:
        rank = int(kwargs["rank"])
        rank_plan = expected_plan.ranks[rank]
        launched.append(dict(kwargs))
        pipeline.run_shard(
            resolved=resolved,
            output_dir=Path(kwargs["output_dir"]),
            row_indices=rank_plan.row_indices,
            worker_metadata={
                "shard_plan_fingerprint": expected_plan.fingerprint,
                "rank": rank_plan.rank,
                "world_size": rank_plan.world_size,
                "parent_visible_device_token": rank_plan.parent_visible_device_token,
                "worker_cuda_visible_devices": rank_plan.parent_visible_device_token,
                "worker_logical_device": "cuda:0",
                "cuda_device_count": 1,
                "cuda_current_device": 0,
                "model_first_parameter_device": "cuda:0",
                "per_device_batch_size": rank_plan.per_device_batch_size,
                "batch_ids": list(rank_plan.batch_ids),
            },
            rank_plan=rank_plan,
            execution_model=None,
            frontend_factory=_frontend_factory(),
            session_opener=lambda launch: FakeBackend([], launch=launch),
        )
        return FakeProcess()

    result = pipeline.run(
        config_path=config_path,
        frontend_factory=_frontend_factory(),
        session_opener=lambda launch: FakeBackend([], launch=launch),
        worker_launcher=fake_worker_launcher,
    )

    run_dir = tmp_path / "outputs" / "wave6-pipeline"
    raw_rows = _read_jsonl(run_dir / "gt_vs_pred.jsonl")
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    provenance = json.loads(
        (run_dir / "gt_vs_pred_scored.jsonl.provenance.json").read_text(
            encoding="utf-8"
        )
    )

    assert result == 0
    assert [call["rank"] for call in launched] == [0, 1]
    assert all(Path(call["resolved_config_json"]).is_file() for call in launched)
    assert all(Path(call["shard_plan_json"]).is_file() for call in launched)
    assert all(call.get("execution_model_json") is None for call in launched)
    assert [row["row_id"] for row in raw_rows] == ["row-0", "row-1"]
    assert manifest["parallelism"]["execution_mode"] == "controller_worker"
    assert manifest["parallelism"]["merge_status"] == "completed"
    assert manifest["parallelism"]["active_ranks"] == 2
    assert provenance["parallelism"]["rank_to_device"] == {"0": "0", "1": "1"}
    assert not (run_dir / "metrics.json").exists()


def test_pipeline_partial_worker_launch_failure_terminates_started_workers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.inference import data_parallel
    from src.inference import pipeline

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setattr(data_parallel, "detect_cuda_device_count", lambda: 2)
    config_path = _write_config(tmp_path, batch_size=1, row_count=2)

    class LiveProcess:
        pid = None
        returncode: int | None = None
        terminated = False

        def terminate(self) -> None:
            self.terminated = True
            self.returncode = -15

        def wait(self, *, timeout: float | None = None) -> int:
            assert timeout is not None
            assert self.returncode is not None
            return self.returncode

    first = LiveProcess()
    cache_root = tmp_path / "partial-launch-worker-cache"
    cache_root.mkdir()
    first._coordexp_runtime_cache_root = str(cache_root)

    def partial_launcher(**kwargs: Any) -> LiveProcess:
        if kwargs["rank"] == 0:
            return first
        raise RuntimeError("injected rank-1 launch failure")

    with pytest.raises(RuntimeContractError) as exc_info:
        pipeline.run(
            config_path=config_path,
            frontend_factory=_frontend_factory(),
            session_opener=lambda launch: FakeBackend([], launch=launch),
            worker_launcher=partial_launcher,
        )

    run_dir = tmp_path / "outputs" / "wave6-pipeline"
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert exc_info.value.code == "pipeline.worker_launch_failed"
    assert first.terminated is True
    assert first.returncode == -15
    assert not cache_root.exists()
    assert summary["terminal_status"] == "failed"
    assert manifest["terminal_status"] == "failed"
    assert not (run_dir / "gt_vs_pred.jsonl").exists()
    assert not (run_dir / "gt_vs_pred_scored.jsonl").exists()
    assert not (run_dir / "gt_vs_pred_scored.jsonl.provenance.json").exists()


def test_pipeline_cleanup_failure_still_publishes_terminal_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.inference import data_parallel, pipeline, worker

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setattr(data_parallel, "detect_cuda_device_count", lambda: 2)
    config_path = _write_config(tmp_path, batch_size=1, row_count=2)
    process = SimpleNamespace(pid=None, returncode=None)

    def partial_launcher(**kwargs: Any) -> object:
        if kwargs["rank"] == 0:
            return process
        raise RuntimeError("injected rank-1 launch failure")

    def fail_termination(_: object) -> int:
        raise RuntimeContractError(
            "survived",
            code="inference.worker_process_tree_survived",
        )

    monkeypatch.setattr(worker, "_terminate_worker_process_tree", fail_termination)

    with pytest.raises(RuntimeContractError) as exc_info:
        pipeline.run(
            config_path=config_path,
            frontend_factory=_frontend_factory(),
            session_opener=lambda launch: FakeBackend([], launch=launch),
            worker_launcher=partial_launcher,
        )

    run_dir = tmp_path / "outputs" / "wave6-pipeline"
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))

    assert exc_info.value.code == "inference.worker_process_tree_survived"
    assert summary["terminal_status"] == "failed"
    assert summary["error"]["code"] == "inference.worker_process_tree_survived"
    assert summary["error"]["context"]["controller_phase"] == "worker_launch"
    assert summary["error"]["context"]["trigger_exception_type"] == "RuntimeError"
    assert manifest["terminal_status"] == "failed"
    assert not (run_dir / "gt_vs_pred.jsonl").exists()
    assert not (run_dir / "gt_vs_pred_scored.jsonl").exists()


@pytest.mark.parametrize("failure_kind", ("worker_timeout", "orphan_process"))
def test_pipeline_controller_wait_failure_cleans_owned_workers_and_publishes_only_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
) -> None:
    from src.inference import data_parallel, pipeline, worker

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setattr(data_parallel, "detect_cuda_device_count", lambda: 2)
    config_path = _write_config(tmp_path, batch_size=1, row_count=2)
    processes: list[Any] = []

    class FakeProcess:
        def __init__(self, rank: int) -> None:
            self.rank = rank
            self.pid = 5678 if failure_kind == "orphan_process" and rank == 0 else None
            self.returncode: int | None = (
                0 if failure_kind == "orphan_process" and rank == 0 else None
            )
            self.terminated = False

        def poll(self) -> int | None:
            return self.returncode

        def terminate(self) -> None:
            self.terminated = True
            self.returncode = -15

        def wait(self, *, timeout: float | None = None) -> int:
            if self.returncode is None:
                raise worker.subprocess.TimeoutExpired("worker", timeout)
            return self.returncode

    def launcher(**kwargs: Any) -> FakeProcess:
        rank = int(kwargs["rank"])
        process = FakeProcess(rank)
        cache_root = tmp_path / f"wait-failure-cache-{rank}"
        cache_root.mkdir()
        process._coordexp_runtime_cache_root = str(cache_root)
        processes.append(process)
        return process

    original_wait = worker.wait_for_worker_processes
    if failure_kind == "worker_timeout":
        monkeypatch.setattr(
            worker,
            "wait_for_worker_processes",
            lambda launched: original_wait(launched, timeout_seconds=1e-9),
        )
    else:
        process_group_alive = True

        def process_group_exists(process_group_id: int) -> bool:
            return process_group_id == 5678 and process_group_alive

        def signal_process_group(process_group_id: int, signal: object) -> None:
            nonlocal process_group_alive
            assert process_group_id == 5678
            process_group_alive = False

        monkeypatch.setattr(worker, "WORKER_TERMINATION_GRACE_SECONDS", 0.0)
        monkeypatch.setattr(worker, "_process_group_exists", process_group_exists)
        monkeypatch.setattr(worker, "_signal_process_group", signal_process_group)

    with pytest.raises(RuntimeContractError) as exc_info:
        pipeline.run(
            config_path=config_path,
            frontend_factory=_frontend_factory(),
            session_opener=lambda launch: FakeBackend([], launch=launch),
            worker_launcher=launcher,
        )

    expected_code = {
        "worker_timeout": "inference.worker_timeout",
        "orphan_process": "inference.worker_orphan_process",
    }[failure_kind]
    assert exc_info.value.code == expected_code
    assert len(processes) == 2
    assert all(
        not (tmp_path / f"wait-failure-cache-{rank}").exists()
        for rank in (0, 1)
    )
    if failure_kind == "worker_timeout":
        assert all(process.terminated for process in processes)
    else:
        assert processes[1].terminated is True
        assert process_group_alive is False

    run_dir = tmp_path / "outputs" / "wave6-pipeline"
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert summary["terminal_status"] == "failed"
    assert summary["error"]["code"] == expected_code
    assert manifest["terminal_status"] == "failed"
    assert manifest["benchmark_eligible"] is False
    assert not (run_dir / "gt_vs_pred.jsonl").exists()
    assert not (run_dir / "gt_vs_pred_scored.jsonl").exists()
    assert not (run_dir / "gt_vs_pred_scored.jsonl.provenance.json").exists()


def test_pipeline_controller_interrupt_reaps_every_launched_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.inference import data_parallel, pipeline, worker

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setattr(data_parallel, "detect_cuda_device_count", lambda: 2)
    config_path = _write_config(tmp_path, batch_size=1, row_count=2)
    processes: list[Any] = []

    class LiveProcess:
        pid = None
        returncode: int | None = None
        terminated = False

        def terminate(self) -> None:
            self.terminated = True
            self.returncode = -15

        def wait(self, *, timeout: float | None = None) -> int:
            assert self.returncode is not None
            return self.returncode

    def launcher(**kwargs: Any) -> LiveProcess:
        process = LiveProcess()
        cache_root = tmp_path / f"interrupt-cache-{kwargs['rank']}"
        cache_root.mkdir()
        process._coordexp_runtime_cache_root = str(cache_root)
        processes.append(process)
        return process

    monkeypatch.setattr(
        worker,
        "wait_for_worker_processes",
        lambda launched: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    with pytest.raises(KeyboardInterrupt):
        pipeline.run(
            config_path=config_path,
            frontend_factory=_frontend_factory(),
            session_opener=lambda launch: FakeBackend([], launch=launch),
            worker_launcher=launcher,
        )

    assert len(processes) == 2
    assert all(process.terminated for process in processes)
    assert not any((tmp_path / f"interrupt-cache-{rank}").exists() for rank in (0, 1))
    run_dir = tmp_path / "outputs" / "wave6-pipeline"
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert summary["terminal_status"] == "failed"
    assert summary["error"]["code"] == "pipeline.controller_interrupted"
    assert summary["error"]["context"]["phase"] == "worker_wait"
    assert manifest["terminal_status"] == "failed"
    assert manifest["benchmark_eligible"] is False
    assert not (run_dir / "gt_vs_pred.jsonl").exists()
    assert not (run_dir / "gt_vs_pred_scored.jsonl").exists()
    assert not (run_dir / "gt_vs_pred_scored.jsonl.provenance.json").exists()


def test_pipeline_controller_missing_rank_zero_artifacts_writes_terminal_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.config.inference import load_infer_config
    from src.inference import data_parallel
    from src.inference import pipeline

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setattr(data_parallel, "detect_cuda_device_count", lambda: 2)
    config_path = _write_config(tmp_path, batch_size=1, row_count=2)
    resolved = load_infer_config(config_path)
    expected_plan = data_parallel.plan_data_parallel_shards(
        row_ids=("row-0", "row-1"),
        per_device_batch_size=1,
        visible_cuda_tokens=("0", "1"),
    )

    class FakeProcess:
        returncode = 0

        def wait(self) -> int:
            return 0

    def fake_worker_launcher(**kwargs: Any) -> FakeProcess:
        rank = int(kwargs["rank"])
        rank_plan = expected_plan.ranks[rank]
        if rank == 1:
            pipeline.run_shard(
                resolved=resolved,
                output_dir=Path(kwargs["output_dir"]),
                row_indices=rank_plan.row_indices,
                worker_metadata={
                    "shard_plan_fingerprint": expected_plan.fingerprint,
                    "rank": rank_plan.rank,
                    "world_size": rank_plan.world_size,
                    "parent_visible_device_token": rank_plan.parent_visible_device_token,
                    "worker_cuda_visible_devices": rank_plan.parent_visible_device_token,
                    "worker_logical_device": "cuda:0",
                    "cuda_device_count": 1,
                    "cuda_current_device": 0,
                    "model_first_parameter_device": "cuda:0",
                    "per_device_batch_size": rank_plan.per_device_batch_size,
                    "batch_ids": list(rank_plan.batch_ids),
                },
                rank_plan=rank_plan,
                frontend_factory=_frontend_factory(),
                session_opener=lambda launch: FakeBackend([], launch=launch),
            )
        return FakeProcess()

    with pytest.raises(ArtifactContractError) as exc_info:
        pipeline.run(
            config_path=config_path,
            frontend_factory=_frontend_factory(),
            session_opener=lambda launch: FakeBackend([], launch=launch),
            worker_launcher=fake_worker_launcher,
        )

    run_dir = tmp_path / "outputs" / "wave6-pipeline"
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))

    assert exc_info.value.code == "merge.missing_shard_artifact"
    assert summary["terminal_status"] == "failed"
    assert summary["failure_class"] == "merge_failure"
    assert summary["benchmark_eligible"] is False
    assert manifest["terminal_status"] == "failed"
    assert manifest["benchmark_eligible"] is False
    assert not (run_dir / "gt_vs_pred.jsonl").exists()
    assert not (run_dir / "gt_vs_pred_scored.jsonl").exists()
    assert not (run_dir / "gt_vs_pred_scored.jsonl.provenance.json").exists()
    assert (run_dir / "shards" / "rank-001").is_dir()


def test_pipeline_controller_malformed_rank_zero_manifest_writes_terminal_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.config.inference import load_infer_config
    from src.inference import data_parallel
    from src.inference import pipeline

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setattr(data_parallel, "detect_cuda_device_count", lambda: 2)
    config_path = _write_config(tmp_path, batch_size=1, row_count=2)
    resolved = load_infer_config(config_path)
    expected_plan = data_parallel.plan_data_parallel_shards(
        row_ids=("row-0", "row-1"),
        per_device_batch_size=1,
        visible_cuda_tokens=("0", "1"),
    )

    class FakeProcess:
        returncode = 0

        def wait(self) -> int:
            return 0

    def fake_worker_launcher(**kwargs: Any) -> FakeProcess:
        rank = int(kwargs["rank"])
        rank_plan = expected_plan.ranks[rank]
        pipeline.run_shard(
            resolved=resolved,
            output_dir=Path(kwargs["output_dir"]),
            row_indices=rank_plan.row_indices,
            worker_metadata={
                "shard_plan_fingerprint": expected_plan.fingerprint,
                "rank": rank_plan.rank,
                "world_size": rank_plan.world_size,
                "parent_visible_device_token": rank_plan.parent_visible_device_token,
                "worker_cuda_visible_devices": rank_plan.parent_visible_device_token,
                "worker_logical_device": "cuda:0",
                "cuda_device_count": 1,
                "cuda_current_device": 0,
                "model_first_parameter_device": "cuda:0",
                "per_device_batch_size": rank_plan.per_device_batch_size,
                "batch_ids": list(rank_plan.batch_ids),
            },
            rank_plan=rank_plan,
            frontend_factory=_frontend_factory(),
            session_opener=lambda launch: FakeBackend([], launch=launch),
        )
        if rank == 0:
            (Path(kwargs["output_dir"]) / "run_manifest.json").write_text(
                "{not valid json\n",
                encoding="utf-8",
            )
        return FakeProcess()

    with pytest.raises(ArtifactContractError) as exc_info:
        pipeline.run(
            config_path=config_path,
            frontend_factory=_frontend_factory(),
            session_opener=lambda launch: FakeBackend([], launch=launch),
            worker_launcher=fake_worker_launcher,
        )

    run_dir = tmp_path / "outputs" / "wave6-pipeline"
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))

    assert exc_info.value.code == "merge.json_decode"
    assert summary["terminal_status"] == "failed"
    assert summary["failure_class"] == "merge_failure"
    assert summary["benchmark_eligible"] is False
    assert manifest["terminal_status"] == "failed"
    assert not (run_dir / "gt_vs_pred.jsonl").exists()
    assert not (run_dir / "gt_vs_pred_scored.jsonl").exists()
    assert (run_dir / "shards" / "rank-000").is_dir()


def test_pipeline_manifest_records_embedding_delta_load_receipt(tmp_path: Path) -> None:
    from src.inference import pipeline

    config_path = _write_config(tmp_path, batch_size=1, row_count=1)
    adapter_identity = {
        "status": "validated",
        "adapter_path": "checkpoints/step-5/adapter",
    }
    embedding_delta_identity = {
        "status": "loaded",
        "identity": {"status": "validated", "metadata_path": "delta/special_token_embeddings.json"},
        "load": {"loaded": True, "tensor_shape": [1004, 2048]},
    }
    model_identity = {
        "family": "base-plus-adapter-plus-delta",
        "base": {"path": "fake-qwen"},
        "adapter": adapter_identity,
        "embedding_delta": embedding_delta_identity,
    }

    pipeline.run(
        config_path=config_path,
        frontend_factory=_frontend_factory(),
        session_opener=lambda launch: FakeBackend(
            [], launch=launch, model_identity=model_identity
        ),
    )

    manifest = json.loads(
        (tmp_path / "outputs" / "wave6-pipeline" / "run_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["model_identity"]["embedding_delta"]["status"] == "loaded"
    assert manifest["model_identity"]["embedding_delta"]["load"]["loaded"] is True
    assert manifest["adapter_identity"] == adapter_identity
    assert manifest["embedding_delta_identity"] == embedding_delta_identity


def test_pipeline_terminal_artifact_failure_writes_status_without_row_artifacts(tmp_path: Path) -> None:
    from src.inference import pipeline

    config_path = _write_config(tmp_path, batch_size=1, row_count=1)

    class BadTraceBackend(FakeBackend):
        def generate_batch(self, requests: list[Any], **kwargs: Any) -> list[DecodeResult]:
            self.calls.append([request.request_id for request in requests])
            bad = _decode_result(
                requests[0].request_id,
                prompt_token_ids=list(requests[0].prompt_token_ids),
            )
            bad_trace = list(bad.token_trace)
            bad_trace[4] = TokenTrace(
                **{
                    **bad_trace[4].__dict__,
                    "likelihood": LikelihoodPair(
                        policy_logprob=float("nan"),
                        raw_model_logprob=None,
                    ),
                }
            )
            return [DecodeResult(**{**bad.__dict__, "token_trace": bad_trace})]

    with pytest.raises(RuntimeContractError) as exc_info:
        pipeline.run(
            config_path=config_path,
            frontend_factory=_frontend_factory(),
            session_opener=lambda launch: BadTraceBackend([], launch=launch),
        )

    run_dir = tmp_path / "outputs" / "wave6-pipeline"
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))

    assert exc_info.value.code == "backend_trace.invalid_likelihood"
    assert summary["terminal_status"] == "failed"
    assert summary["failure_class"] == "contract_failure"
    assert summary["contract_failure_count"] == 1
    assert summary["benchmark_eligible"] is False
    assert manifest["terminal_status"] == "failed"
    assert manifest["benchmark_eligible"] is False
    assert not (run_dir / "gt_vs_pred.jsonl").exists()
    assert not (run_dir / "gt_vs_pred_scored.jsonl").exists()


@pytest.mark.parametrize(
    ("error", "failure_class", "error_code"),
    [
        (
            torch.OutOfMemoryError("injected CUDA OOM"),
            "cuda_oom",
            "inference.cuda_oom",
        ),
        (
            RuntimeError("injected backend startup failure"),
            "runtime_failure",
            "inference.unhandled_runtime_failure",
        ),
    ],
    ids=("cuda-oom", "backend-startup"),
)
def test_pipeline_unhandled_runtime_failure_writes_terminal_diagnostics_only(
    tmp_path: Path,
    error: Exception,
    failure_class: str,
    error_code: str,
) -> None:
    from src.inference import pipeline

    config_path = _write_config(tmp_path, batch_size=1, row_count=1)

    def fail_session_open(launch: Any) -> Any:
        raise error

    with pytest.raises(type(error), match="injected"):
        pipeline.run(
            config_path=config_path,
            frontend_factory=_frontend_factory(),
            session_opener=fail_session_open,
        )

    run_dir = tmp_path / "outputs" / "wave6-pipeline"
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))

    assert summary["terminal_status"] == "failed"
    assert summary["failure_class"] == failure_class
    assert summary["error"]["code"] == error_code
    assert summary["error"]["context"]["exception_type"] == type(error).__name__
    assert manifest["terminal_status"] == "failed"
    assert manifest["benchmark_eligible"] is False
    assert not (run_dir / "gt_vs_pred.jsonl").exists()
    assert not (run_dir / "gt_vs_pred_scored.jsonl").exists()
    assert not (run_dir / "gt_vs_pred_scored.jsonl.provenance.json").exists()


@pytest.mark.parametrize(
    ("error", "failure_class", "error_code"),
    [
        (
            torch.OutOfMemoryError("injected vLLM worker CUDA OOM"),
            "cuda_oom",
            "inference.cuda_oom",
        ),
        (
            RuntimeError("injected vLLM worker engine startup failure"),
            "runtime_failure",
            "inference.unhandled_runtime_failure",
        ),
    ],
    ids=("cuda-oom", "engine-startup"),
)
def test_pipeline_vllm_controller_worker_failures_publish_shard_and_controller_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    failure_class: str,
    error_code: str,
) -> None:
    from src.config.inference import load_infer_config
    from src.inference import data_parallel, pipeline

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(data_parallel, "detect_cuda_device_count", lambda: 1)
    config_path = _write_config(
        tmp_path,
        batch_size=1,
        row_count=1,
        backend_type="vllm",
    )
    resolved = load_infer_config(config_path)
    plan = data_parallel.plan_data_parallel_shards(
        row_ids=("row-0",),
        per_device_batch_size=1,
        visible_cuda_tokens=("0",),
    )
    execution_model = {
        "mode": "base_only",
        "model_path": str(tmp_path / "model_cache" / "qwen"),
    }
    monkeypatch.setattr(
        pipeline,
        "_resolve_execution_model_for_run",
        lambda _: execution_model,
    )
    monkeypatch.setattr(
        pipeline,
        "validate_execution_model_receipt",
        lambda receipt: dict(receipt),
    )

    class FailedWorkerProcess:
        pid = None
        returncode = 1

        def poll(self) -> int:
            return self.returncode

    def fail_session_open(_: Any) -> Any:
        raise error

    def launcher(**kwargs: Any) -> FailedWorkerProcess:
        rank_plan = plan.ranks[int(kwargs["rank"])]
        assert kwargs["execution_model_json"] is not None
        with pytest.raises(type(error), match="injected vLLM worker"):
            pipeline.run_shard(
                resolved=resolved,
                output_dir=Path(kwargs["output_dir"]),
                row_indices=rank_plan.row_indices,
                worker_metadata={
                    "shard_plan_fingerprint": plan.fingerprint,
                    "rank": rank_plan.rank,
                    "world_size": rank_plan.world_size,
                    "parent_visible_device_token": rank_plan.parent_visible_device_token,
                    "worker_cuda_visible_devices": rank_plan.parent_visible_device_token,
                    "worker_logical_device": "cuda:0",
                    "cuda_device_count": 1,
                    "cuda_current_device": 0,
                    "model_first_parameter_device": "cuda:0",
                    "per_device_batch_size": rank_plan.per_device_batch_size,
                    "batch_ids": list(rank_plan.batch_ids),
                },
                rank_plan=rank_plan,
                frontend_factory=_frontend_factory(),
                session_opener=fail_session_open,
                execution_model=execution_model,
            )
        return FailedWorkerProcess()

    with pytest.raises(RuntimeContractError) as exc_info:
        pipeline.run(
            config_path=config_path,
            worker_launcher=launcher,
        )

    run_dir = tmp_path / "outputs" / "wave6-pipeline"
    shard_summary = json.loads(
        (run_dir / "shards" / "rank-000" / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))

    assert exc_info.value.code == "pipeline.worker_failed"
    assert shard_summary["failure_class"] == failure_class
    assert shard_summary["error"]["code"] == error_code
    assert summary["terminal_status"] == "failed"
    assert summary["error"]["code"] == "pipeline.worker_failed"
    assert manifest["terminal_status"] == "failed"
    assert manifest["benchmark_eligible"] is False
    assert not (run_dir / "gt_vs_pred.jsonl").exists()
    assert not (run_dir / "gt_vs_pred_scored.jsonl").exists()


def test_pipeline_rejects_backend_batch_with_too_few_results(tmp_path: Path) -> None:
    from src.inference import pipeline

    config_path = _write_config(tmp_path, batch_size=2, row_count=2)

    class TooFewBackend(FakeBackend):
        def generate_batch(self, requests: list[Any], **kwargs: Any) -> list[DecodeResult]:
            self.calls.append([request.request_id for request in requests])
            return [
                _decode_result(
                    requests[0].request_id,
                    prompt_token_ids=list(requests[0].prompt_token_ids),
                )
            ]

    with pytest.raises(RuntimeContractError) as exc_info:
        pipeline.run(
            config_path=config_path,
            frontend_factory=_frontend_factory(),
            session_opener=lambda launch: TooFewBackend([], launch=launch),
        )

    run_dir = tmp_path / "outputs" / "wave6-pipeline"
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))

    assert exc_info.value.code == "backend_contract.result_set"
    assert exc_info.value.context["requested_request_ids"] == ["row-0", "row-1"]
    assert exc_info.value.context["observed_request_ids"] == ["row-0"]
    assert summary["terminal_status"] == "failed"
    assert summary["failure_class"] == "contract_failure"
    assert manifest["terminal_status"] == "failed"
    assert manifest["benchmark_eligible"] is False
    assert not (run_dir / "gt_vs_pred.jsonl").exists()
    assert not (run_dir / "gt_vs_pred_scored.jsonl").exists()


def test_pipeline_rejects_backend_batch_with_extra_duplicate_result(tmp_path: Path) -> None:
    from src.inference import pipeline

    config_path = _write_config(tmp_path, batch_size=2, row_count=2)

    class DuplicateBackend(FakeBackend):
        def generate_batch(self, requests: list[Any], **kwargs: Any) -> list[DecodeResult]:
            self.calls.append([request.request_id for request in requests])
            return [
                _decode_result(
                    requests[0].request_id,
                    prompt_token_ids=list(requests[0].prompt_token_ids),
                ),
                _decode_result(
                    requests[1].request_id,
                    prompt_token_ids=list(requests[1].prompt_token_ids),
                ),
                _decode_result(
                    requests[0].request_id,
                    prompt_token_ids=list(requests[0].prompt_token_ids),
                ),
            ]

    with pytest.raises(RuntimeContractError) as exc_info:
        pipeline.run(
            config_path=config_path,
            frontend_factory=_frontend_factory(),
            session_opener=lambda launch: DuplicateBackend([], launch=launch),
        )

    run_dir = tmp_path / "outputs" / "wave6-pipeline"
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))

    assert exc_info.value.code == "backend_contract.result_set"
    assert exc_info.value.context["requested_request_ids"] == ["row-0", "row-1"]
    assert exc_info.value.context["observed_request_ids"] == ["row-0", "row-1", "row-0"]
    assert summary["terminal_status"] == "failed"
    assert summary["failure_class"] == "contract_failure"
    assert manifest["terminal_status"] == "failed"
    assert manifest["benchmark_eligible"] is False
    assert not (run_dir / "gt_vs_pred.jsonl").exists()
    assert not (run_dir / "gt_vs_pred_scored.jsonl").exists()


def test_pipeline_records_parser_and_score_counters_without_metric_reduction(tmp_path: Path) -> None:
    from src.inference import pipeline

    config_path = _write_config(tmp_path, batch_size=2, row_count=2)

    class MixedBackend(FakeBackend):
        def generate_batch(self, requests: list[Any], **kwargs: Any) -> list[DecodeResult]:
            self.calls.append([request.request_id for request in requests])
            results = []
            for request in requests:
                if request.request_id == "row-1":
                    results.append(
                        _decode_result(
                            request.request_id,
                            text="not a compact object",
                            prompt_token_ids=list(request.prompt_token_ids),
                        )
                    )
                else:
                    results.append(
                        _decode_result(
                            request.request_id,
                            prompt_token_ids=list(request.prompt_token_ids),
                        )
                    )
            return results

    pipeline.run(
        config_path=config_path,
        frontend_factory=_frontend_factory(),
        session_opener=lambda launch: MixedBackend([], launch=launch),
    )

    run_dir = tmp_path / "outputs" / "wave6-pipeline"
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

    assert summary["parser_failure_count"] == 1
    assert summary["scoreable_prediction_count"] == 1
    assert not (run_dir / "metrics.json").exists()
    infer_tree = ast.parse(Path("src/infer.py").read_text(encoding="utf-8"))
    assert not any(
        isinstance(node, ast.Attribute) and node.attr.startswith("evaluate")
        for node in ast.walk(infer_tree)
    )


def test_inference_modules_do_not_import_eval_metric_reduction() -> None:
    paths = [Path("src/infer.py"), *sorted(Path("src/inference").glob("*.py"))]
    forbidden_modules = {"src.eval", "src.eval.detection_consumer"}
    forbidden_names = {"evaluate_scored_detection_artifacts", "mAP", "mRecall"}
    violations: list[str] = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module in forbidden_modules or module.startswith("src.eval."):
                    violations.append(f"{path}:{module}")
                for alias in node.names:
                    if alias.name in forbidden_names:
                        violations.append(f"{path}:{alias.name}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in forbidden_modules or alias.name.startswith("src.eval."):
                        violations.append(f"{path}:{alias.name}")
            elif isinstance(node, ast.Name) and node.id in forbidden_names:
                violations.append(f"{path}:{node.id}")

    assert violations == []


def test_pipeline_terminal_image_failure_writes_status_without_row_artifacts(tmp_path: Path) -> None:
    from src.inference import pipeline

    config_path = _write_config(tmp_path, batch_size=1, row_count=1, invalid_image=True)

    with pytest.raises(EncodingContractError) as exc_info:
        pipeline.run(
            config_path=config_path,
            frontend_factory=_frontend_factory(),
            session_opener=lambda launch: FakeBackend([], launch=launch),
        )

    run_dir = tmp_path / "outputs" / "wave6-pipeline"
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))

    assert exc_info.value.code == "qwen.image_no_resize_dimensions"
    assert summary["terminal_status"] == "failed"
    assert summary["failure_class"] == "image_validation_failure"
    assert summary["image_validation_failure_count"] == 1
    assert summary["benchmark_eligible"] is False
    assert manifest["benchmark_eligible"] is False
    assert manifest["terminal_status"] == "failed"
    assert not (run_dir / "gt_vs_pred.jsonl").exists()
    assert not (run_dir / "gt_vs_pred_scored.jsonl").exists()


def test_pipeline_writes_resolved_config_before_backend_generation(tmp_path: Path) -> None:
    from src.inference import pipeline

    config_path = _write_config(tmp_path, batch_size=1, row_count=1)
    events: list[str] = []

    class OrderBackend(FakeBackend):
        def generate_batch(self, requests: list[Any], **kwargs: Any) -> list[DecodeResult]:
            assert (tmp_path / "outputs" / "wave6-pipeline" / "configs" / "resolved.json").is_file()
            events.append("backend_generate")
            return super().generate_batch(requests, **kwargs)

    pipeline.run(
        config_path=config_path,
        frontend_factory=_frontend_factory(),
        session_opener=lambda launch: OrderBackend([], launch=launch),
    )

    assert events == ["backend_generate"]


def test_shard_primitive_processes_only_assigned_rows_and_preserves_original_indices(
    tmp_path: Path,
) -> None:
    from src.config.inference import load_infer_config
    from src.inference import pipeline

    config_path = _write_config(tmp_path, batch_size=2, row_count=3)
    resolved = load_infer_config(config_path)
    shard_dir = tmp_path / "manual-root" / "shards" / "rank-000"
    backend_calls: list[list[str]] = []

    pipeline.run_shard(
        resolved=resolved,
        output_dir=shard_dir,
        row_indices=(0, 2),
        worker_metadata={"rank": 0, "world_size": 2},
        frontend_factory=_frontend_factory(),
        session_opener=lambda launch: FakeBackend(backend_calls, launch=launch),
    )

    raw_rows = _read_jsonl(shard_dir / "gt_vs_pred.jsonl")
    scored_rows = _read_jsonl(shard_dir / "gt_vs_pred_scored.jsonl")
    image_plan_rows = _read_jsonl(shard_dir / "image_plan.jsonl")
    manifest = json.loads((shard_dir / "run_manifest.json").read_text(encoding="utf-8"))

    assert backend_calls == [["row-0", "row-2"]]
    assert [row["row_id"] for row in raw_rows] == ["row-0", "row-2"]
    assert [row["row_index"] for row in raw_rows] == [0, 2]
    assert [row["row_id"] for row in scored_rows] == ["row-0", "row-2"]
    assert [row["row_index"] for row in image_plan_rows] == [0, 2]
    assert manifest["parallelism"]["shard_assignment"] == {
        "assigned_row_indices": [0, 2],
        "assigned_row_ids": ["row-0", "row-2"],
    }
    assert "row-1" not in {row["row_id"] for row in raw_rows}


def test_shard_primitive_writes_only_fixed_shard_output_dir(tmp_path: Path) -> None:
    from src.config.inference import load_infer_config
    from src.inference import pipeline

    config_path = _write_config(tmp_path, batch_size=1, row_count=2)
    resolved = load_infer_config(config_path)
    root_dir = tmp_path / "manual-root"
    shard_dir = root_dir / "shards" / "rank-001"

    pipeline.run_shard(
        resolved=resolved,
        output_dir=shard_dir,
        row_indices=(1,),
        worker_metadata={"rank": 1, "world_size": 2},
        frontend_factory=_frontend_factory(),
        session_opener=lambda launch: FakeBackend([], launch=launch),
    )

    assert (shard_dir / "gt_vs_pred.jsonl").is_file()
    assert not (root_dir / "gt_vs_pred.jsonl").exists()
    assert not (root_dir / "gt_vs_pred_scored.jsonl").exists()
    assert not (root_dir / "configs" / "resolved.json").exists()


def test_shard_primitive_does_not_resolve_collision_policy_run_directory(
    tmp_path: Path,
) -> None:
    from src.config.inference import load_infer_config
    from src.inference import pipeline

    config_path = _write_config(tmp_path, batch_size=1, row_count=1)
    resolved = load_infer_config(config_path)
    configured_run_dir = tmp_path / "outputs" / "wave6-pipeline"
    configured_run_dir.mkdir(parents=True)
    shard_dir = tmp_path / "manual-root" / "shards" / "rank-000"

    pipeline.run_shard(
        resolved=resolved,
        output_dir=shard_dir,
        row_indices=(0,),
        worker_metadata={"rank": 0, "world_size": 1},
        frontend_factory=_frontend_factory(),
        session_opener=lambda launch: FakeBackend([], launch=launch),
    )

    assert (shard_dir / "run_manifest.json").is_file()
    assert not (configured_run_dir / "run_manifest.json").exists()


@pytest.mark.parametrize("row_index", [True, "1.5"])
def test_shard_primitive_rejects_non_integral_row_indices(
    tmp_path: Path,
    row_index: object,
) -> None:
    from src.common.errors import RuntimeContractError
    from src.config.inference import load_infer_config
    from src.inference import pipeline

    config_path = _write_config(tmp_path, batch_size=1, row_count=2)
    resolved = load_infer_config(config_path)

    with pytest.raises(RuntimeContractError) as exc_info:
        pipeline.run_shard(
            resolved=resolved,
            output_dir=tmp_path / "manual-root" / "shards" / "rank-000",
            row_indices=(row_index,),  # type: ignore[arg-type]
            worker_metadata={"rank": 0, "world_size": 1},
            frontend_factory=_frontend_factory(),
            session_opener=lambda launch: FakeBackend([], launch=launch),
        )

    assert exc_info.value.code == "pipeline.invalid_shard_row_index"


def test_shard_primitive_rejects_rank_plan_row_mismatch(tmp_path: Path) -> None:
    from src.common.errors import RuntimeContractError
    from src.config.inference import load_infer_config
    from src.inference import data_parallel
    from src.inference import pipeline

    config_path = _write_config(tmp_path, batch_size=1, row_count=3)
    resolved = load_infer_config(config_path)
    plan = data_parallel.plan_data_parallel_shards(
        row_ids=("row-0", "row-1", "row-2"),
        per_device_batch_size=1,
        visible_cuda_tokens=("0", "1"),
    )

    with pytest.raises(RuntimeContractError) as exc_info:
        pipeline.run_shard(
            resolved=resolved,
            output_dir=tmp_path / "manual-root" / "shards" / "rank-000",
            row_indices=(2,),
            rank_plan=plan.ranks[0],
            worker_metadata={"rank": 0, "world_size": 2},
            frontend_factory=_frontend_factory(),
            session_opener=lambda launch: FakeBackend([], launch=launch),
        )

    assert exc_info.value.code == "pipeline.rank_plan_row_mismatch"


def test_data_parallel_controller_uses_shard_primitive_for_each_rank_and_restores_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.config.inference import load_infer_config
    from src.inference import data_parallel
    from src.inference import pipeline

    config_path = _write_config(tmp_path, batch_size=2, row_count=5)
    resolved = load_infer_config(config_path)
    plan = data_parallel.plan_data_parallel_shards(
        row_ids=tuple(f"row-{index}" for index in range(5)),
        per_device_batch_size=2,
        visible_cuda_tokens=("0", "1"),
    )
    calls: list[dict[str, object]] = []

    def fake_run_shard(**kwargs: object) -> int:
        rank_plan = kwargs["rank_plan"]
        output_dir = Path(kwargs["output_dir"])
        calls.append(
            {
                "rank": rank_plan.rank,
                "row_indices": rank_plan.row_indices,
                "output_dir": output_dir,
            }
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        rows = [
            {"row_id": row_id, "row_index": row_index}
            for row_id, row_index in zip(
                rank_plan.row_ids,
                rank_plan.row_indices,
                strict=True,
            )
        ]
        (output_dir / "gt_vs_pred.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(pipeline, "run_shard", fake_run_shard)

    result = pipeline.run_data_parallel_shards(
        resolved=resolved,
        run_dir=tmp_path / "dp-run",
        plan=plan,
        frontend_factory=_frontend_factory(),
        session_opener=lambda launch: FakeBackend([], launch=launch),
    )

    assert [call["rank"] for call in calls] == [0, 1]
    assert [call["row_indices"] for call in calls] == [(0, 1, 4), (2, 3)]
    assert calls[0]["output_dir"] == tmp_path / "dp-run" / "shards" / "rank-000"
    assert calls[1]["output_dir"] == tmp_path / "dp-run" / "shards" / "rank-001"
    assert [row["row_id"] for row in result.raw_rows] == [
        "row-0",
        "row-1",
        "row-2",
        "row-3",
        "row-4",
    ]


def test_data_parallel_shards_reject_vllm_in_process_execution(tmp_path: Path) -> None:
    from src.inference import pipeline

    resolved = SimpleNamespace(
        config=SimpleNamespace(backend=SimpleNamespace(type="vllm"))
    )

    with pytest.raises(RuntimeContractError) as exc_info:
        pipeline.run_data_parallel_shards(
            resolved=resolved,
            run_dir=tmp_path / "dp-run",
            plan=SimpleNamespace(),
        )

    assert exc_info.value.code == "pipeline.vllm_in_process_forbidden"


def test_data_parallel_shard_manifest_records_plan_and_worker_device_metadata(
    tmp_path: Path,
) -> None:
    from src.config.inference import load_infer_config
    from src.inference import data_parallel
    from src.inference import pipeline

    config_path = _write_config(tmp_path, batch_size=1, row_count=2)
    resolved = load_infer_config(config_path)
    plan = data_parallel.plan_data_parallel_shards(
        row_ids=("row-0", "row-1"),
        per_device_batch_size=1,
        visible_cuda_tokens=("0", "1"),
    )

    result = pipeline.run_data_parallel_shards(
        resolved=resolved,
        run_dir=tmp_path / "dp-real-shards",
        plan=plan,
        frontend_factory=_frontend_factory(),
        session_opener=lambda launch: FakeBackend([], launch=launch),
    )

    assert len(result.shard_dirs) == 2
    for rank_plan, shard_dir in zip(plan.ranks, result.shard_dirs, strict=True):
        manifest = json.loads((shard_dir / "run_manifest.json").read_text(encoding="utf-8"))
        parallelism = manifest["parallelism"]
        worker = parallelism["worker"]
        assert parallelism["shard_plan_fingerprint"] == plan.fingerprint
        assert parallelism["shard_assignment"]["assigned_row_indices"] == list(
            rank_plan.row_indices
        )
        assert worker["rank"] == rank_plan.rank
        assert worker["world_size"] == rank_plan.world_size
        assert worker["parent_visible_device_token"] == rank_plan.parent_visible_device_token
        assert worker["worker_cuda_visible_devices"] == rank_plan.parent_visible_device_token
        assert worker["worker_logical_device"] == "cuda:0"
        assert worker["cuda_device_count"] == 1
        assert worker["cuda_current_device"] == 0
        assert worker["model_first_parameter_device"] == "cuda:0"


def test_shard_primitive_fills_worker_model_device_after_runtime_load(
    tmp_path: Path,
) -> None:
    from src.config.inference import load_infer_config
    from src.inference import data_parallel
    from src.inference import pipeline

    config_path = _write_config(tmp_path, batch_size=1, row_count=1)
    resolved = load_infer_config(config_path)
    plan = data_parallel.plan_data_parallel_shards(
        row_ids=("row-0",),
        per_device_batch_size=1,
        visible_cuda_tokens=("0",),
    )
    pipeline.run_shard(
        resolved=resolved,
        output_dir=tmp_path / "manual-root" / "shards" / "rank-000",
        row_indices=(0,),
        worker_metadata={
            "shard_plan_fingerprint": plan.fingerprint,
            "rank": 0,
            "world_size": 1,
            "parent_visible_device_token": "0",
            "worker_cuda_visible_devices": "0",
            "worker_logical_device": "cuda:0",
            "cuda_device_count": 1,
            "cuda_current_device": 0,
            "per_device_batch_size": 1,
            "batch_ids": [0],
        },
        rank_plan=plan.ranks[0],
        frontend_factory=_frontend_factory(),
        session_opener=lambda launch: FakeBackend([], launch=launch),
    )

    manifest = json.loads(
        (
            tmp_path
            / "manual-root"
            / "shards"
            / "rank-000"
            / "run_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert (
        manifest["parallelism"]["worker"]["model_first_parameter_device"]
        == "cuda:0"
    )


def _frontend_factory() -> Any:
    def factory(
        config: Any,
        *,
        generation_config_fingerprint: str,
        execution_model: Any | None = None,
    ) -> InferenceFrontend:
        return _frontend(
            config=config,
            generation_config_fingerprint=generation_config_fingerprint,
            execution_model=execution_model,
        )

    return factory


def _frontend(
    *,
    config: Any,
    generation_config_fingerprint: str,
    execution_model: Any | None = None,
) -> InferenceFrontend:
    processor_identity = QwenProcessorIdentity(
        processor_class="FakeQwen3VLProcessor",
        tokenizer_class="FakeTokenizer",
        image_processor_class="FakeQwen2VLImageProcessorFast",
        patch_size=16,
        merge_size=2,
        temporal_patch_size=2,
    )
    qwen = SimpleNamespace(
        processor=FakeProcessor(),
        processor_identity=processor_identity,
        config=SimpleNamespace(
            vision_config=SimpleNamespace(
                patch_size=16,
                spatial_merge_size=2,
                temporal_patch_size=2,
            )
        ),
        model=None,
        to_artifact_dict=lambda: {"frontend": "unit"},
    )
    if config.backend.type == "hf":
        model_path = config.model.base_model
        backend_options = {"hf": config.backend.hf.model_dump(mode="json")}
        execution_model_identity = None
    else:
        assert execution_model is not None
        model_path = execution_model["model_path"]
        backend_options = {"vllm": config.backend.vllm.model_dump(mode="json")}
        execution_model_identity = dict(execution_model)
    return InferenceFrontend(
        qwen=qwen,
        launch=BackendLaunch(
            backend=config.backend.type,
            model_path=model_path,
            model_dtype=config.model.dtype,
            batch_size=config.generation.batch_size,
            generation_config_fingerprint=generation_config_fingerprint,
            backend_options=backend_options,
            execution_model_identity=execution_model_identity,
        ),
    )


def _decode_result(
    row_id: str,
    *,
    text: str = OBJECT_TEXT,
    prompt_token_ids: list[int] | None = None,
    observed_image_grid_thw: tuple[int, int, int] | None = (1, 4, 6),
) -> DecodeResult:
    pieces = _token_pieces(text)
    traces = [
        TokenTrace(
            step_index=index,
            token_id=151646 + index,
            token_text=piece,
            likelihood=LikelihoodPair(
                policy_logprob=math.log(0.25),
                raw_model_logprob=None,
            ),
            is_stop=False,
            is_pad=False,
            backend="hf",
            backend_mode="generate",
            response_family="hf",
        )
        for index, piece in enumerate(pieces)
    ]
    return DecodeResult(
        request_id=row_id,
        backend="hf",
        backend_mode="generate",
        response_family="hf",
        executed_prompt_token_ids=tuple(prompt_token_ids or [11, 12]),
        generated_token_ids=tuple(trace.token_id for trace in traces),
        raw_generated_text=text,
        parser_text=text,
        strip_policy="none",
        stop_reason="length",
        token_trace=tuple(traces),
        observed_image_grid_thw=observed_image_grid_thw,
        executed_media_sha256="a" * 64,
    )


def _token_pieces(text: str) -> list[str]:
    if text != OBJECT_TEXT:
        return [text]
    return [
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


def _write_config(
    tmp_path: Path,
    *,
    batch_size: int,
    row_count: int,
    invalid_image: bool = False,
    smoke: bool = True,
    backend_type: str = "hf",
) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(row_count):
        width = 96
        height = 65 if invalid_image else 64
        image_path = data_dir / f"row-{index}.jpg"
        Image.new("RGB", (width, height), color=(12, 34, 56)).save(image_path)
        rows.append(
            {
                "example_id": f"row-{index}",
                "image": {"path": image_path.name, "width": width, "height": height},
                "objects": [
                    {
                        "object_id": f"object-{index}",
                        "description": "cat",
                        "bbox": [100, 200, 300, 400],
                        "metadata": {},
                    }
                ],
                "metadata": {},
            }
        )
    input_jsonl = data_dir / "examples.jsonl"
    input_jsonl.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    config = {
        "schema_version": 1,
        "run": {
            "name": "wave6-pipeline",
            "artifact_root": str(tmp_path / "outputs"),
            "collision_policy": "fail",
        },
        "model": {
            "base_model": str(tmp_path / "model_cache" / "qwen"),
            "dtype": "bf16",
            "processor": {"do_resize": False},
        },
        "data": {"input_jsonl": str(input_jsonl)},
        "template": {
            "object_field_order": "desc_first",
            "object_ordering": "source_order",
            "assistant_format": "object_box_closed",
            "prompt": {"user": "Describe objects."},
        },
        "backend": (
            {
                "type": "hf",
                "hf": {
                    "attn_implementation": "flash_attention_2",
                    "patch_embed_linearization": "enabled",
                },
            }
            if backend_type == "hf"
            else {
                "type": "vllm",
                "vllm": {"gpu_memory_utilization": 0.7},
            }
        ),
        "generation": {
            "batch_size": batch_size,
            "max_new_tokens": 64,
            "temperature": 0.0,
            "top_p": 1.0,
            "n": 1,
        },
        "scoring": {"enabled": True},
        "artifacts": {"write_token_trace": True, "write_parse_diagnostics": True},
        "debug": {"smoke": smoke, "dry_run": False},
    }
    config_path = tmp_path / "infer.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
