"""CoordExp-swift offline inference pipeline orchestration."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from src.common.errors import (
    ArtifactContractError,
    CoordExpError,
    EncodingContractError,
    RuntimeContractError,
)
from src.config.inference import InferConfig, ResolvedInferConfig, load_infer_config, resolve_infer_run_directory
from src.config.models import ProcessorConfig, TemplateConfig, TemplatePromptConfig
from src.config.writer import write_resolved_config_artifacts
from src.data import RawExample, load_raw_examples
from src.inference.artifacts import (
    RAW_NAME,
    write_inference_artifacts,
    write_terminal_status_artifacts,
)
from src.inference.backend import DecodeRequest, HFGenerateBackend
from src.inference.data_parallel import (
    DataParallelPlan,
    RankShardPlan,
    plan_data_parallel_shards,
    require_visible_cuda_for_inference,
    sort_rows_by_index,
)
from src.inference.image_plan import materialize_image_plan_batch, verify_processor_model_vision_parity
from src.inference.merge import merge_shard_artifacts
from src.inference.parsing import PARSER_POLICY, parse_compact_object_box_closed
from src.inference.prompt import TEMPLATE_ID, build_prompt_record, verify_prompt_token_parity
from src.inference.runtime import InferenceRuntime, assemble_runtime
from src.inference.scoring import SCORE_POLICY_FINGERPRINT


RuntimeFactory = Callable[[InferConfig], Any]
BackendFactory = Callable[[Any, InferConfig], Any]
WorkerLauncher = Callable[..., Any]


@dataclass(frozen=True)
class DataParallelShardRunResult:
    raw_rows: list[dict[str, Any]]
    shard_dirs: list[Path]


def run(
    *,
    config_path: str | Path,
    runtime_factory: RuntimeFactory | None = None,
    backend_factory: BackendFactory | None = None,
    worker_launcher: WorkerLauncher | None = None,
) -> int:
    resolved = load_infer_config(config_path)
    run_dir = resolve_infer_run_directory(
        resolved.config,
        timestamp=_timestamp_suffix(),
    ).run_dir
    run_dir.mkdir(parents=True, exist_ok=False)
    write_resolved_config_artifacts(resolved, run_dir)
    metadata = _base_metadata(resolved=resolved)

    if resolved.config.debug.dry_run:
        return 0

    try:
        visible_cuda_tokens = require_visible_cuda_for_inference(
            debug_dry_run=resolved.config.debug.dry_run,
        )
        raw_examples = list(load_raw_examples(resolved.config.data.input_jsonl))
        data_parallel_plan = plan_data_parallel_shards(
            row_ids=tuple(example.example_id for example in raw_examples),
            per_device_batch_size=resolved.config.generation.batch_size,
            visible_cuda_tokens=visible_cuda_tokens,
        )
        metadata["parallelism"] = {
            "execution_mode": (
                "direct_single_process"
                if data_parallel_plan.active_ranks == 1
                else "controller_worker"
            ),
            "plan": data_parallel_plan.to_artifact_dict(),
        }
    except CoordExpError as exc:
        _write_terminal_contract_failure(
            output_dir=run_dir,
            metadata=metadata,
            error=exc,
        )
        raise

    if data_parallel_plan.active_ranks == 1:
        _execute_indexed_rows_with_terminal_status(
            resolved=resolved,
            output_dir=run_dir,
            indexed_raw_examples=tuple(enumerate(raw_examples)),
            metadata=metadata,
            runtime_factory=runtime_factory,
            backend_factory=backend_factory,
        )
        return 0

    _execute_controller_worker_path(
        resolved=resolved,
        run_dir=run_dir,
        raw_examples=tuple(raw_examples),
        metadata=metadata,
        plan=data_parallel_plan,
        worker_launcher=worker_launcher,
    )
    return 0


def run_shard(
    *,
    resolved: ResolvedInferConfig,
    output_dir: Path,
    row_indices: Sequence[int],
    worker_metadata: dict[str, Any] | None = None,
    rank_plan: RankShardPlan | None = None,
    runtime_factory: RuntimeFactory | None = None,
    backend_factory: BackendFactory | None = None,
) -> int:
    raw_examples = list(load_raw_examples(resolved.config.data.input_jsonl))
    indexed_raw_examples = _select_indexed_raw_examples(
        raw_examples=raw_examples,
        row_indices=row_indices,
    )
    if rank_plan is not None:
        _validate_rank_plan_matches_assignment(
            rank_plan=rank_plan,
            row_indices=tuple(index for index, _ in indexed_raw_examples),
        )
    metadata = _base_metadata(resolved=resolved)
    shard_worker_metadata = dict(worker_metadata or {})
    metadata["parallelism"] = {
        "execution_mode": "rank_local_shard",
        "shard_plan_fingerprint": shard_worker_metadata.pop(
            "shard_plan_fingerprint",
            None,
        ),
        "worker": shard_worker_metadata,
        "shard_assignment": {
            "assigned_row_indices": [index for index, _ in indexed_raw_examples],
            "assigned_row_ids": [
                raw_example.example_id for _, raw_example in indexed_raw_examples
            ],
        },
    }
    _execute_indexed_rows_with_terminal_status(
        resolved=resolved,
        output_dir=output_dir,
        indexed_raw_examples=indexed_raw_examples,
        metadata=metadata,
        runtime_factory=runtime_factory,
        backend_factory=backend_factory,
    )
    return 0


def run_data_parallel_shards(
    *,
    resolved: ResolvedInferConfig,
    run_dir: Path,
    plan: DataParallelPlan,
    runtime_factory: RuntimeFactory | None = None,
    backend_factory: BackendFactory | None = None,
) -> DataParallelShardRunResult:
    shard_dirs: list[Path] = []
    raw_rows: list[dict[str, Any]] = []
    for rank_plan in plan.ranks:
        shard_dir = run_dir / "shards" / rank_plan.shard_dir_name
        shard_dirs.append(shard_dir)
        run_shard(
            resolved=resolved,
            output_dir=shard_dir,
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
            runtime_factory=runtime_factory,
            backend_factory=backend_factory,
        )
        raw_rows.extend(_read_jsonl(shard_dir / RAW_NAME))
    return DataParallelShardRunResult(
        raw_rows=sort_rows_by_index(raw_rows),
        shard_dirs=shard_dirs,
    )


def _execute_controller_worker_path(
    *,
    resolved: ResolvedInferConfig,
    run_dir: Path,
    raw_examples: tuple[RawExample, ...],
    metadata: dict[str, Any],
    plan: DataParallelPlan,
    worker_launcher: WorkerLauncher | None,
) -> None:
    from src.inference import worker as worker_module

    plan_json = _write_data_parallel_plan_artifact(run_dir=run_dir, plan=plan)
    resolved_config_json = run_dir / "configs" / "resolved.json"
    launcher = worker_launcher or worker_module.launch_worker_subprocess
    launched: list[tuple[RankShardPlan, Any]] = []
    for rank_plan in plan.ranks:
        shard_dir = run_dir / "shards" / rank_plan.shard_dir_name
        process = launcher(
            rank=rank_plan.rank,
            world_size=rank_plan.world_size,
            parent_visible_device_token=rank_plan.parent_visible_device_token,
            resolved_config_json=resolved_config_json,
            shard_plan_json=plan_json,
            output_dir=shard_dir,
        )
        launched.append((rank_plan, process))

    worker_statuses: dict[int, str] = {}
    return_codes: dict[int, int | None] = {}
    for rank_plan, process in launched:
        wait = getattr(process, "wait", None)
        if callable(wait):
            code = wait()
        else:
            code = getattr(process, "returncode", None)
        if code is None:
            code = getattr(process, "returncode", None)
        code_int = None if code is None else int(code)
        return_codes[rank_plan.rank] = code_int
        worker_statuses[rank_plan.rank] = "completed" if code_int == 0 else "failed"

    failed = {
        rank: code
        for rank, code in sorted(return_codes.items())
        if code != 0
    }
    if failed:
        error = RuntimeContractError(
            "one or more inference workers failed",
            code="pipeline.worker_failed",
            context={"worker_return_codes": failed},
        )
        _write_terminal_contract_failure(
            output_dir=run_dir,
            metadata=metadata,
            error=error,
        )
        raise error

    shard_dirs = [run_dir / "shards" / rank_plan.shard_dir_name for rank_plan in plan.ranks]
    merge_shard_artifacts(
        output_dir=run_dir,
        shard_dirs=tuple(shard_dirs),
        expected_row_ids=tuple(example.example_id for example in raw_examples),
        metadata=metadata,
        plan=plan,
        worker_statuses=worker_statuses,
    )


def _write_data_parallel_plan_artifact(*, run_dir: Path, plan: DataParallelPlan) -> Path:
    path = run_dir / "shards" / "data_parallel_plan.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, plan.to_artifact_dict())
    return path


def _execute_indexed_rows_with_terminal_status(
    *,
    resolved: ResolvedInferConfig,
    output_dir: Path,
    indexed_raw_examples: tuple[tuple[int, RawExample], ...],
    metadata: dict[str, Any],
    runtime_factory: RuntimeFactory | None,
    backend_factory: BackendFactory | None,
) -> None:
    runtime = (runtime_factory or assemble_runtime)(resolved.config)
    qwen = runtime.qwen
    metadata.update(_runtime_metadata(runtime=runtime, qwen=qwen))
    _fill_worker_runtime_device_metadata(metadata=metadata, qwen=qwen)
    try:
        _execute_indexed_rows(
            resolved=resolved,
            output_dir=output_dir,
            indexed_raw_examples=indexed_raw_examples,
            metadata=metadata,
            runtime=runtime,
            qwen=qwen,
            backend_factory=backend_factory,
        )
    except EncodingContractError as exc:
        write_terminal_status_artifacts(
            output_dir=output_dir,
            metadata=metadata,
            summary={
                "terminal_status": "failed",
                "failure_class": "image_validation_failure",
                "image_validation_failure_count": 1,
                "error": {"code": exc.code, "message": exc.message, "context": exc.context},
            },
        )
        raise
    except CoordExpError as exc:
        _write_terminal_contract_failure(
            output_dir=output_dir,
            metadata=metadata,
            error=exc,
        )
        raise


def _execute_indexed_rows(
    *,
    resolved: ResolvedInferConfig,
    output_dir: Path,
    indexed_raw_examples: tuple[tuple[int, RawExample], ...],
    metadata: dict[str, Any],
    runtime: Any,
    qwen: Any,
    backend_factory: BackendFactory | None,
) -> None:
    verify_processor_model_vision_parity(
        processor_identity=qwen.processor_identity,
        model_config=_model_config(qwen),
    )
    raw_examples = [raw_example for _, raw_example in indexed_raw_examples]
    image_plan_batch = materialize_image_plan_batch(
        raw_examples,
        components=qwen,
        processor_config=_processor_config(resolved.config),
        materialize=True,
        row_indices=[row_index for row_index, _ in indexed_raw_examples],
    )
    backend = (backend_factory or _default_backend_factory)(runtime, resolved.config)
    prompt_records = [
        build_prompt_record(
            raw_example,
            _template_config(resolved.config),
            processor=qwen.processor,
            row_index=row_index,
        )
        for row_index, raw_example in indexed_raw_examples
    ]
    requests = [
        DecodeRequest(
            request_id=record.row_id,
            prompt_token_ids=list(record.prompt_token_ids),
            model_inputs=image_plan_batch.model_inputs_by_row_id[record.row_id],
            max_new_tokens=resolved.config.generation.max_new_tokens,
            repetition_penalty=resolved.config.generation.repetition_penalty,
        )
        for record in prompt_records
    ]

    decode_results = {}
    for batch in _batches(requests, size=resolved.config.generation.batch_size):
        batch_results = backend.generate_batch(
            list(batch),
            model_identity=metadata["model_identity"],
            tokenizer_identity=metadata["tokenizer_identity"],
            generation_config_fingerprint=metadata["generation_config_fingerprint"],
        )
        _validate_backend_result_set(requests=list(batch), results=list(batch_results))
        for result in batch_results:
            record = _prompt_record_by_id(prompt_records, result.request_id)
            verify_prompt_token_parity(
                record,
                backend_prompt_token_ids=list(result.prompt_token_ids),
            )
            decode_results[result.request_id] = result

    rows = [
        _artifact_input_row(
            raw_example=raw_example,
            row_index=row_index,
            decode_result=decode_results[raw_example.example_id],
        )
        for row_index, raw_example in indexed_raw_examples
    ]
    counters = _pipeline_counters(rows=rows, decode_success_count=len(decode_results))
    metadata["pipeline_counters"] = counters
    write_inference_artifacts(
        output_dir=output_dir,
        rows=rows,
        decode_results=decode_results,
        image_plan_rows=[row.to_artifact_dict() for row in image_plan_batch.rows],
        metadata=metadata,
    )


def _select_indexed_raw_examples(
    *,
    raw_examples: Sequence[RawExample],
    row_indices: Sequence[int],
) -> tuple[tuple[int, RawExample], ...]:
    indices = tuple(_coerce_shard_row_index(index) for index in row_indices)
    if not indices:
        raise RuntimeContractError(
            "shard execution requires at least one assigned row",
            code="pipeline.empty_shard_assignment",
        )
    if len(set(indices)) != len(indices):
        raise RuntimeContractError(
            "shard row indices must be unique",
            code="pipeline.duplicate_shard_row_index",
            context={"row_indices": list(indices)},
        )
    max_index = len(raw_examples) - 1
    out_of_range = [index for index in indices if index < 0 or index > max_index]
    if out_of_range:
        raise RuntimeContractError(
            "shard row index is outside the input row range",
            code="pipeline.shard_row_index_out_of_range",
            context={"row_indices": list(indices), "row_count": len(raw_examples)},
        )
    return tuple((index, raw_examples[index]) for index in indices)


def _coerce_shard_row_index(index: Any) -> int:
    if isinstance(index, bool) or not isinstance(index, int):
        raise RuntimeContractError(
            "shard row indices must be integers",
            code="pipeline.invalid_shard_row_index",
            context={"row_index": index, "row_index_type": type(index).__name__},
        )
    return index


def _validate_rank_plan_matches_assignment(
    *,
    rank_plan: RankShardPlan,
    row_indices: tuple[int, ...],
) -> None:
    if tuple(rank_plan.row_indices) != row_indices:
        raise RuntimeContractError(
            "rank plan row indices must match shard assignment",
            code="pipeline.rank_plan_row_mismatch",
            context={
                "rank": rank_plan.rank,
                "rank_plan_row_indices": list(rank_plan.row_indices),
                "row_indices": list(row_indices),
            },
        )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    text = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False)
    path.write_text(text + "\n", encoding="utf-8")


def _default_backend_factory(runtime: InferenceRuntime, config: InferConfig) -> HFGenerateBackend:
    if config.backend.type != "hf":
        raise ArtifactContractError(
            "only HF backend is implemented for CoordExp-swift V1 pipeline",
            code="pipeline.backend_not_implemented",
            context={"backend": config.backend.type},
        )
    return HFGenerateBackend(model=runtime.qwen.model, tokenizer=runtime.qwen.tokenizer)


def _validate_backend_result_set(
    *,
    requests: list[DecodeRequest],
    results: list[Any],
) -> None:
    requested_ids = [request.request_id for request in requests]
    observed_ids = [str(getattr(result, "request_id", "")) for result in results]
    requested_counts = Counter(requested_ids)
    observed_counts = Counter(observed_ids)
    missing_ids = _counter_delta(requested_counts, observed_counts, requested_ids)
    extra_ids = _counter_delta(observed_counts, requested_counts, observed_ids)
    duplicate_ids = _duplicates_in_order(observed_ids)
    unknown_ids = [row_id for row_id in observed_ids if row_id not in requested_counts]
    if missing_ids or extra_ids or duplicate_ids or unknown_ids:
        raise ArtifactContractError(
            "backend decode result set must match requested batch request ids exactly",
            code="pipeline.backend_result_set_mismatch",
            context={
                "requested_request_ids": requested_ids,
                "observed_request_ids": observed_ids,
                "missing_request_ids": missing_ids,
                "extra_result_ids": extra_ids,
                "duplicate_result_ids": duplicate_ids,
                "unknown_result_ids": unknown_ids,
            },
        )


def _counter_delta(
    left: Counter[str],
    right: Counter[str],
    order: list[str],
) -> list[str]:
    remaining = left.copy()
    for row_id, count in right.items():
        remaining[row_id] -= count
    values: list[str] = []
    emitted: Counter[str] = Counter()
    for row_id in order:
        allowed = max(remaining[row_id], 0)
        if emitted[row_id] < allowed:
            values.append(row_id)
            emitted[row_id] += 1
    return values


def _duplicates_in_order(values: list[str]) -> list[str]:
    counts = Counter(values)
    seen: set[str] = set()
    duplicates = []
    for value in values:
        if counts[value] > 1 and value not in seen:
            duplicates.append(value)
            seen.add(value)
    return duplicates


def _artifact_input_row(
    *,
    raw_example: RawExample,
    row_index: int,
    decode_result: Any,
) -> dict[str, Any]:
    parse_row = parse_compact_object_box_closed(
        decode_result.parser_text,
        row_id=raw_example.example_id,
        row_index=row_index,
        image_width=raw_example.image.width,
        image_height=raw_example.image.height,
    )
    return {
        "row_id": raw_example.example_id,
        "row_index": row_index,
        "example_id": raw_example.example_id,
        "image_path": str(raw_example.image.path),
        "image_width": raw_example.image.width,
        "image_height": raw_example.image.height,
        "gt": [obj.to_artifact_dict() for obj in raw_example.objects],
        "raw_decode_text": decode_result.raw_generated_text,
        "decode_stop_reason": str(getattr(decode_result, "stop_reason", "")),
        "parse": parse_row,
    }


def _pipeline_counters(*, rows: list[dict[str, Any]], decode_success_count: int) -> dict[str, Any]:
    parser_failure_count = sum(
        1
        for row in rows
        if row["parse"].parse_status not in {"accepted", "accepted_with_drops"}
    )
    dropped_prediction_count = sum(row["parse"].dropped_prediction_count for row in rows)
    decode_stop_reasons = Counter(str(row.get("decode_stop_reason", "")) for row in rows)
    return {
        "terminal_status": "completed",
        "decode_success_count": decode_success_count,
        "parser_failure_count": parser_failure_count,
        "dropped_prediction_count": dropped_prediction_count,
        "truncated_decode_count": int(decode_stop_reasons.get("length", 0)),
        "decode_stop_reasons": dict(sorted(decode_stop_reasons.items())),
        "image_validation_failure_count": 0,
        "score_failure_count": 0,
    }


def _write_terminal_contract_failure(
    *,
    output_dir: Path,
    metadata: dict[str, Any],
    error: CoordExpError,
) -> None:
    failure_class = _failure_class(error)
    write_terminal_status_artifacts(
        output_dir=output_dir,
        metadata=metadata,
        summary={
            "terminal_status": "failed",
            "failure_class": failure_class,
            f"{failure_class}_count": 1,
            "error": {"code": error.code, "message": error.message, "context": error.context},
        },
    )


def _failure_class(error: CoordExpError) -> str:
    if isinstance(error, ArtifactContractError):
        return "artifact_contract_failure"
    return "contract_failure"


def _base_metadata(*, resolved: ResolvedInferConfig) -> dict[str, Any]:
    return {
        "artifact_schema_version": 1,
        "resolved_config_fingerprints": {"infer_config": resolved.fingerprint},
        "detection_template_id": TEMPLATE_ID,
        "prompt_policy_fingerprint": _fingerprint(
            {
                "template": resolved.config.template.model_dump(mode="json"),
                "template_id": TEMPLATE_ID,
            }
        ),
        "generation_config_fingerprint": _fingerprint(
            resolved.config.generation.model_dump(mode="json")
        ),
        "generation_policy": _generation_policy(resolved.config),
        "model_identity_fingerprint": "unknown-before-runtime",
        "processor_identity_fingerprint": "unknown-before-runtime",
        "template_identity": {
            "id": TEMPLATE_ID,
            "object_field_order": resolved.config.template.object_field_order,
            "object_ordering": resolved.config.template.object_ordering,
            "assistant_format": resolved.config.template.assistant_format,
        },
        "parser_policy": PARSER_POLICY,
        "score_policy_fingerprint": SCORE_POLICY_FINGERPRINT,
        "dataset_identity": {
            "input_jsonl": resolved.config.data.input_jsonl,
        },
        "backend": resolved.config.backend.type,
        "backend_mode": "generate",
        "response_family": resolved.config.backend.type,
        "model_identity": {},
        "tokenizer_identity": {},
    }


def _runtime_metadata(*, runtime: Any, qwen: Any) -> dict[str, Any]:
    model_identity = dict(getattr(runtime, "model_identity", {}) or {})
    tokenizer_identity = _tokenizer_identity(qwen)
    processor_identity = qwen.processor_identity.to_artifact_dict()
    checkpoint_handoff = model_identity.get("checkpoint_handoff")
    return {
        "model_identity": model_identity,
        "tokenizer_identity": tokenizer_identity,
        "processor_identity": processor_identity,
        "model_identity_fingerprint": _fingerprint(model_identity),
        "processor_identity_fingerprint": _fingerprint(processor_identity),
        "adapter_identity": getattr(runtime, "adapter_receipt", None),
        "embedding_delta_identity": getattr(runtime, "embedding_delta_receipt", None),
        "composition_mode": (
            "canonical_handoff"
            if isinstance(checkpoint_handoff, dict)
            else (
                "research_manual"
                if getattr(runtime, "adapter_receipt", None) is not None
                or getattr(runtime, "embedding_delta_receipt", None) is not None
                else "base_only"
            )
        ),
        "checkpoint_handoff": checkpoint_handoff,
    }


def _fill_worker_runtime_device_metadata(*, metadata: dict[str, Any], qwen: Any) -> None:
    parallelism = metadata.get("parallelism")
    if not isinstance(parallelism, dict):
        return
    model = getattr(qwen, "model", None)
    device = _model_first_parameter_device(model)
    worker = parallelism.get("worker")
    if isinstance(worker, dict):
        if worker.get("model_first_parameter_device"):
            return
        if device is not None:
            worker["model_first_parameter_device"] = device
        return

    direct_runtime = parallelism.setdefault("direct_runtime", {})
    if not isinstance(direct_runtime, dict):
        return
    direct_runtime.setdefault("logical_device", "cuda:0")
    plan = parallelism.get("plan")
    if isinstance(plan, dict):
        visible_tokens = plan.get("visible_cuda_tokens") or []
        direct_runtime.setdefault("visible_cuda_token_count", len(visible_tokens))
        direct_runtime.setdefault("active_ranks", plan.get("active_ranks"))
    if device is not None:
        direct_runtime.setdefault("model_first_parameter_device", device)
    try:
        direct_runtime.setdefault("cuda_available", bool(torch.cuda.is_available()))
        direct_runtime.setdefault("cuda_device_count", int(torch.cuda.device_count()))
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            direct_runtime.setdefault("cuda_current_device", int(torch.cuda.current_device()))
    except Exception as exc:  # pragma: no cover - defensive CUDA probe evidence.
        direct_runtime.setdefault("cuda_probe_error", type(exc).__name__)


def _model_first_parameter_device(model: Any | None) -> str | None:
    if model is None:
        return None
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        return None
    first_param = next(iter(parameters()), None)
    device = getattr(first_param, "device", None)
    return None if device is None else str(device)


def _tokenizer_identity(qwen: Any) -> dict[str, Any]:
    token_identity = getattr(qwen, "token_identity", None)
    if token_identity is not None and hasattr(token_identity, "to_artifact_dict"):
        return token_identity.to_artifact_dict()
    tokenizer_sha = getattr(qwen, "tokenizer_sha256", None)
    if tokenizer_sha:
        return {"tokenizer_sha256": tokenizer_sha}
    return {"identity": "test-tokenizer"}


def _model_config(qwen: Any) -> Any:
    if hasattr(qwen, "config"):
        return qwen.config
    model = getattr(qwen, "model", None)
    if model is not None and hasattr(model, "config"):
        return model.config
    return qwen


def _processor_config(config: InferConfig) -> ProcessorConfig:
    return ProcessorConfig(
        do_resize=config.model.processor.do_resize,
        max_raw_pixels=1_000_000_000,
        max_merged_visual_tokens=1_000_000,
    )


def _template_config(config: InferConfig) -> TemplateConfig:
    return TemplateConfig(
        object_field_order=config.template.object_field_order,
        object_ordering=config.template.object_ordering,
        assistant_format=config.template.assistant_format,
        prompt=TemplatePromptConfig(
            system=config.template.prompt.system,
            user=config.template.prompt.user,
        ),
    )


def _generation_policy(config: InferConfig) -> dict[str, Any]:
    return {
        "batch_size": int(config.generation.batch_size),
        "max_new_tokens": int(config.generation.max_new_tokens),
        "temperature": float(config.generation.temperature),
        "top_p": float(config.generation.top_p),
        "repetition_penalty": float(config.generation.repetition_penalty),
        "do_sample": False,
        "return_dict_in_generate": True,
        "output_scores": True,
        "stop_policy": "qwen_im_end",
    }


def _prompt_record_by_id(records: Sequence[Any], row_id: str) -> Any:
    for record in records:
        if record.row_id == row_id:
            return record
    raise ArtifactContractError(
        "backend returned decode result for an unknown request",
        code="pipeline.unknown_decode_result",
        context={"request_id": row_id},
    )


def _batches(values: Sequence[Any], *, size: int) -> list[Sequence[Any]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def _fingerprint(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _timestamp_suffix() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
