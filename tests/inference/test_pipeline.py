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

from src.common.errors import ArtifactContractError, EncodingContractError
from src.inference.backend import DecodeResult, TokenTrace
from src.qwen.loading import QwenProcessorIdentity


OBJECT_TEXT = (
    "<|object_ref_start|>cat<|object_ref_end|>"
    "<|box_start|><|coord_100|><|coord_200|><|coord_300|><|coord_400|><|box_end|>"
)


class FakeTokenizer:
    def __call__(self, text: str, *, add_special_tokens: bool = False, **_: Any) -> dict[str, list[int]]:
        assert add_special_tokens is False
        return {"input_ids": [ord(char) for char in text]}


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
    def __init__(self, calls: list[list[str]]) -> None:
        self.calls = calls

    def generate_batch(
        self,
        requests: list[Any],
        *,
        model_identity: dict[str, Any],
        tokenizer_identity: dict[str, Any],
        generation_config_fingerprint: str,
    ) -> list[DecodeResult]:
        self.calls.append([request.request_id for request in requests])
        for request in requests:
            assert set(request.model_inputs) == {"pixel_values", "image_grid_thw"}
            assert tuple(request.model_inputs["pixel_values"].shape) == (24, 1536)
            assert tuple(request.model_inputs["image_grid_thw"].shape) == (1, 3)
        return [
            _decode_result(
                request.request_id,
                prompt_token_ids=list(request.prompt_token_ids),
                model_identity=model_identity,
                tokenizer_identity=tokenizer_identity,
                generation_config_fingerprint=generation_config_fingerprint,
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
    runtime = _runtime()
    runtime.qwen.model.parameters = lambda: iter([SimpleNamespace(device="cuda:0")])

    result = pipeline.run(
        config_path=config_path,
        runtime_factory=lambda config: runtime,
        backend_factory=lambda runtime, config: FakeBackend(backend_calls),
    )

    run_dir = tmp_path / "outputs" / "wave6-pipeline"
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    raw_rows = _read_jsonl(run_dir / "gt_vs_pred.jsonl")
    scored_rows = _read_jsonl(run_dir / "gt_vs_pred_scored.jsonl")

    assert result == 0
    assert backend_calls == [["row-0", "row-1"], ["row-2"]]
    assert [row["row_id"] for row in raw_rows] == ["row-0", "row-1", "row-2"]
    assert [row["row_id"] for row in scored_rows] == ["row-0", "row-1", "row-2"]
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
    assert raw_rows[0]["decode_stop_reason"] == "length"


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
        runtime_factory=lambda config: _runtime(),
        backend_factory=lambda runtime, config: FakeBackend([]),
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
            runtime_factory=lambda config: _runtime(),
            backend_factory=lambda runtime, config: FakeBackend([]),
        )
        return FakeProcess()

    result = pipeline.run(
        config_path=config_path,
        runtime_factory=lambda config: _runtime(),
        backend_factory=lambda runtime, config: FakeBackend([]),
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
    assert [row["row_id"] for row in raw_rows] == ["row-0", "row-1"]
    assert manifest["parallelism"]["execution_mode"] == "controller_worker"
    assert manifest["parallelism"]["merge_status"] == "completed"
    assert manifest["parallelism"]["active_ranks"] == 2
    assert provenance["parallelism"]["rank_to_device"] == {"0": "0", "1": "1"}
    assert not (run_dir / "metrics.json").exists()


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
                runtime_factory=lambda config: _runtime(),
                backend_factory=lambda runtime, config: FakeBackend([]),
            )
        return FakeProcess()

    with pytest.raises(ArtifactContractError) as exc_info:
        pipeline.run(
            config_path=config_path,
            runtime_factory=lambda config: _runtime(),
            backend_factory=lambda runtime, config: FakeBackend([]),
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
            runtime_factory=lambda config: _runtime(),
            backend_factory=lambda runtime, config: FakeBackend([]),
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
            runtime_factory=lambda config: _runtime(),
            backend_factory=lambda runtime, config: FakeBackend([]),
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
    runtime = _runtime()
    runtime.adapter_receipt = {
        "status": "validated",
        "adapter_path": "checkpoints/step-5/adapter",
    }
    runtime.model_identity["embedding_delta"] = {
        "status": "loaded",
        "identity": {"status": "validated", "metadata_path": "delta/special_token_embeddings.json"},
        "load": {"loaded": True, "tensor_shape": [1004, 2048]},
    }
    runtime.embedding_delta_receipt = runtime.model_identity["embedding_delta"]

    pipeline.run(
        config_path=config_path,
        runtime_factory=lambda config: runtime,
        backend_factory=lambda runtime, config: FakeBackend([]),
    )

    manifest = json.loads(
        (tmp_path / "outputs" / "wave6-pipeline" / "run_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["model_identity"]["embedding_delta"]["status"] == "loaded"
    assert manifest["model_identity"]["embedding_delta"]["load"]["loaded"] is True
    assert manifest["adapter_identity"] == runtime.adapter_receipt
    assert manifest["embedding_delta_identity"] == runtime.embedding_delta_receipt


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
            bad_trace[4] = TokenTrace(**{**bad_trace[4].__dict__, "logprob": float("nan")})
            return [DecodeResult(**{**bad.__dict__, "token_trace": bad_trace})]

    with pytest.raises(ArtifactContractError) as exc_info:
        pipeline.run(
            config_path=config_path,
            runtime_factory=lambda config: _runtime(),
            backend_factory=lambda runtime, config: BadTraceBackend([]),
        )

    run_dir = tmp_path / "outputs" / "wave6-pipeline"
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))

    assert exc_info.value.code == "artifacts.non_finite_trace_logprob"
    assert summary["terminal_status"] == "failed"
    assert summary["failure_class"] == "artifact_contract_failure"
    assert summary["artifact_contract_failure_count"] == 1
    assert summary["benchmark_eligible"] is False
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

    with pytest.raises(ArtifactContractError) as exc_info:
        pipeline.run(
            config_path=config_path,
            runtime_factory=lambda config: _runtime(),
            backend_factory=lambda runtime, config: TooFewBackend([]),
        )

    run_dir = tmp_path / "outputs" / "wave6-pipeline"
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))

    assert exc_info.value.code == "pipeline.backend_result_set_mismatch"
    assert exc_info.value.context["requested_request_ids"] == ["row-0", "row-1"]
    assert exc_info.value.context["observed_request_ids"] == ["row-0"]
    assert exc_info.value.context["missing_request_ids"] == ["row-1"]
    assert summary["terminal_status"] == "failed"
    assert summary["failure_class"] == "artifact_contract_failure"
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

    with pytest.raises(ArtifactContractError) as exc_info:
        pipeline.run(
            config_path=config_path,
            runtime_factory=lambda config: _runtime(),
            backend_factory=lambda runtime, config: DuplicateBackend([]),
        )

    run_dir = tmp_path / "outputs" / "wave6-pipeline"
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))

    assert exc_info.value.code == "pipeline.backend_result_set_mismatch"
    assert exc_info.value.context["requested_request_ids"] == ["row-0", "row-1"]
    assert exc_info.value.context["observed_request_ids"] == ["row-0", "row-1", "row-0"]
    assert exc_info.value.context["duplicate_result_ids"] == ["row-0"]
    assert exc_info.value.context["extra_result_ids"] == ["row-0"]
    assert summary["terminal_status"] == "failed"
    assert summary["failure_class"] == "artifact_contract_failure"
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
        runtime_factory=lambda config: _runtime(),
        backend_factory=lambda runtime, config: MixedBackend([]),
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
            runtime_factory=lambda config: _runtime(),
            backend_factory=lambda runtime, config: FakeBackend([]),
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
        runtime_factory=lambda config: _runtime(),
        backend_factory=lambda runtime, config: OrderBackend([]),
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
        runtime_factory=lambda config: _runtime(),
        backend_factory=lambda runtime, config: FakeBackend(backend_calls),
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
        runtime_factory=lambda config: _runtime(),
        backend_factory=lambda runtime, config: FakeBackend([]),
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
        runtime_factory=lambda config: _runtime(),
        backend_factory=lambda runtime, config: FakeBackend([]),
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
            runtime_factory=lambda config: _runtime(),
            backend_factory=lambda runtime, config: FakeBackend([]),
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
            runtime_factory=lambda config: _runtime(),
            backend_factory=lambda runtime, config: FakeBackend([]),
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
        runtime_factory=lambda config: _runtime(),
        backend_factory=lambda runtime, config: FakeBackend([]),
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
        runtime_factory=lambda config: _runtime(),
        backend_factory=lambda runtime, config: FakeBackend([]),
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
    runtime = _runtime()
    parameter = SimpleNamespace(device="cuda:0")
    runtime.qwen.model.parameters = lambda: iter([parameter])

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
        runtime_factory=lambda config: runtime,
        backend_factory=lambda runtime, config: FakeBackend([]),
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


def _runtime() -> SimpleNamespace:
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
        model=SimpleNamespace(
            config=SimpleNamespace(
                vision_config=SimpleNamespace(
                    patch_size=16,
                    spatial_merge_size=2,
                    temporal_patch_size=2,
                )
            )
        ),
    )
    return SimpleNamespace(
        qwen=qwen,
        adapter_receipt=None,
        embedding_delta_receipt=None,
        model_identity={"family": "unit", "base": {"path": "fake-qwen"}},
    )


def _decode_result(
    row_id: str,
    *,
    text: str = OBJECT_TEXT,
    prompt_token_ids: list[int] | None = None,
    model_identity: dict[str, Any] | None = None,
    tokenizer_identity: dict[str, Any] | None = None,
    generation_config_fingerprint: str = "gen-fp",
) -> DecodeResult:
    pieces = _token_pieces(text)
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
        for index, piece in enumerate(pieces)
    ]
    return DecodeResult(
        request_id=row_id,
        backend="hf",
        backend_mode="generate",
        response_family="hf",
        prompt_token_ids=list(prompt_token_ids or [11, 12]),
        generated_token_ids=[trace.token_id for trace in traces],
        raw_generated_text=text,
        parser_text=text,
        strip_policy="none",
        stop_reason="length",
        model_identity=dict(model_identity or {"family": "unit"}),
        tokenizer_identity=dict(tokenizer_identity or {"sha256": "tok"}),
        generation_config_fingerprint=generation_config_fingerprint,
        token_trace=traces,
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
            "attn_implementation": "flash_attention_2",
            "processor": {"do_resize": False},
            "runtime_patches": {"patch_embed_linearization": "enabled"},
        },
        "data": {"input_jsonl": str(input_jsonl)},
        "template": {
            "object_field_order": "desc_first",
            "object_ordering": "source_order",
            "assistant_format": "object_box_closed",
            "prompt": {"user": "Describe objects."},
        },
        "backend": {"type": "hf"},
        "generation": {
            "batch_size": batch_size,
            "max_new_tokens": 64,
            "temperature": 0.0,
            "top_p": 1.0,
        },
        "scoring": {"enabled": True},
        "artifacts": {"write_token_trace": True, "write_parse_diagnostics": True},
        "debug": {"smoke": True, "dry_run": False},
    }
    config_path = tmp_path / "infer.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
