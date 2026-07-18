"""CoordExp-swift offline inference pipeline orchestration."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
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
    benchmark_scope_eligible,
    write_inference_artifacts,
    write_terminal_status_artifacts,
)
from src.inference.backend import (
    BackendSessionOpener,
    DecodeRequest,
    GenerationPolicy,
    open_backend_session,
    token_ids_sha256,
    validate_decode_results,
)
from src.inference.data_parallel import (
    DataParallelPlan,
    RankShardPlan,
    plan_data_parallel_shards,
    require_visible_cuda_for_inference,
    sort_rows_by_index,
)
from src.inference.execution_model import (
    resolve_execution_model,
    validate_execution_model_receipt,
)
from src.inference.image_plan import plan_image_batch, verify_processor_model_vision_parity
from src.inference.merge import merge_shard_artifacts
from src.inference.parsing import PARSER_POLICY, parse_compact_object_box_closed
from src.inference.prompt import TEMPLATE_ID, build_prompt_record, verify_prompt_token_parity
from src.inference.runtime import InferenceFrontend, assemble_frontend
from src.inference.scoring import SCORE_POLICY_FINGERPRINT


FrontendFactory = Callable[..., InferenceFrontend]
WorkerLauncher = Callable[..., Any]


@dataclass(frozen=True)
class DataParallelShardRunResult:
    raw_rows: list[dict[str, Any]]
    shard_dirs: list[Path]


def run(
    *,
    config_path: str | Path,
    frontend_factory: FrontendFactory | None = None,
    session_opener: BackendSessionOpener | None = None,
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
        metadata["benchmark_eligible"] = _benchmark_scope_eligible(
            config=resolved.config,
            row_count=len(raw_examples),
        )
        data_parallel_plan = plan_data_parallel_shards(
            row_ids=tuple(example.example_id for example in raw_examples),
            per_device_batch_size=resolved.config.generation.batch_size,
            visible_cuda_tokens=visible_cuda_tokens,
        )
        use_controller_worker = (
            data_parallel_plan.active_ranks > 1
            or resolved.config.backend.type == "vllm"
        )
        metadata["parallelism"] = {
            "execution_mode": (
                "controller_worker"
                if use_controller_worker
                else "direct_single_process"
            ),
            "plan": data_parallel_plan.to_artifact_dict(),
        }
        execution_model = _resolve_execution_model_for_run(resolved)
        if execution_model is not None:
            metadata["execution_model"] = execution_model
    except CoordExpError as exc:
        _write_terminal_contract_failure(
            output_dir=run_dir,
            metadata=metadata,
            error=exc,
        )
        raise

    if not use_controller_worker:
        _execute_indexed_rows_with_terminal_status(
            resolved=resolved,
            output_dir=run_dir,
            indexed_raw_examples=tuple(enumerate(raw_examples)),
            metadata=metadata,
            frontend_factory=frontend_factory,
            session_opener=session_opener,
            execution_model=execution_model,
        )
        return 0

    _execute_controller_worker_path(
        resolved=resolved,
        run_dir=run_dir,
        raw_examples=tuple(raw_examples),
        metadata=metadata,
        plan=data_parallel_plan,
        worker_launcher=worker_launcher,
        execution_model=execution_model,
    )
    return 0


def run_shard(
    *,
    resolved: ResolvedInferConfig,
    output_dir: Path,
    row_indices: Sequence[int],
    worker_metadata: dict[str, Any] | None = None,
    rank_plan: RankShardPlan | None = None,
    frontend_factory: FrontendFactory | None = None,
    session_opener: BackendSessionOpener | None = None,
    execution_model: dict[str, Any] | None = None,
) -> int:
    raw_examples = list(load_raw_examples(resolved.config.data.input_jsonl))
    metadata = _base_metadata(resolved=resolved)
    metadata["benchmark_eligible"] = _benchmark_scope_eligible(
        config=resolved.config,
        row_count=len(raw_examples),
    )
    indexed_raw_examples = _select_indexed_raw_examples(
        raw_examples=raw_examples,
        row_indices=row_indices,
    )
    if rank_plan is not None:
        _validate_rank_plan_matches_assignment(
            rank_plan=rank_plan,
            row_indices=tuple(index for index, _ in indexed_raw_examples),
        )
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
        frontend_factory=frontend_factory,
        session_opener=session_opener,
        execution_model=execution_model,
    )
    return 0


def run_data_parallel_shards(
    *,
    resolved: ResolvedInferConfig,
    run_dir: Path,
    plan: DataParallelPlan,
    frontend_factory: FrontendFactory | None = None,
    session_opener: BackendSessionOpener | None = None,
    execution_model: dict[str, Any] | None = None,
) -> DataParallelShardRunResult:
    if resolved.config.backend.type == "vllm":
        raise RuntimeContractError(
            "vLLM shards must run through fresh rank-local worker processes",
            code="pipeline.vllm_in_process_forbidden",
        )
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
            frontend_factory=frontend_factory,
            session_opener=session_opener,
            execution_model=execution_model,
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
    execution_model: dict[str, Any] | None,
) -> None:
    from src.inference import worker as worker_module

    plan_json = _write_data_parallel_plan_artifact(run_dir=run_dir, plan=plan)
    resolved_config_json = run_dir / "configs" / "resolved.json"
    launcher = worker_launcher or worker_module.launch_worker_subprocess
    execution_model_json = _write_execution_model_artifact(
        run_dir=run_dir,
        execution_model=execution_model,
    )
    launched: list[tuple[RankShardPlan, Any]] = []
    try:
        for rank_plan in plan.ranks:
            shard_dir = run_dir / "shards" / rank_plan.shard_dir_name
            process = launcher(
                rank=rank_plan.rank,
                world_size=rank_plan.world_size,
                parent_visible_device_token=rank_plan.parent_visible_device_token,
                resolved_config_json=resolved_config_json,
                shard_plan_json=plan_json,
                output_dir=shard_dir,
                execution_model_json=execution_model_json,
            )
            launched.append((rank_plan, process))
    except BaseException as cause:
        return_codes = _terminate_owned_workers_after_controller_failure(
            worker_module=worker_module,
            launched=launched,
            run_dir=run_dir,
            metadata=metadata,
            phase="worker_launch",
            cause=cause,
        )
        if isinstance(cause, (KeyboardInterrupt, SystemExit)):
            error = RuntimeContractError(
                "inference controller was interrupted during worker launch",
                code="pipeline.controller_interrupted",
                context={
                    "phase": "worker_launch",
                    "launched_ranks": [rank_plan.rank for rank_plan, _ in launched],
                    "worker_return_codes": return_codes,
                },
                cause=cause,
            )
            _write_terminal_contract_failure(
                output_dir=run_dir,
                metadata=metadata,
                error=error,
            )
            raise
        error = RuntimeContractError(
            "an inference worker failed during controller launch",
            code="pipeline.worker_launch_failed",
            context={
                "launched_ranks": [rank_plan.rank for rank_plan, _ in launched],
                "worker_return_codes": return_codes,
            },
            cause=cause,
        )
        _write_terminal_contract_failure(
            output_dir=run_dir,
            metadata=metadata,
            error=error,
        )
        raise error from cause

    try:
        return_codes = worker_module.wait_for_worker_processes(
            tuple((rank_plan.rank, process) for rank_plan, process in launched)
        )
    except RuntimeContractError as error:
        _write_terminal_contract_failure(
            output_dir=run_dir,
            metadata=metadata,
            error=error,
        )
        raise
    except BaseException as cause:
        return_codes = _terminate_owned_workers_after_controller_failure(
            worker_module=worker_module,
            launched=launched,
            run_dir=run_dir,
            metadata=metadata,
            phase="worker_wait",
            cause=cause,
        )
        error = RuntimeContractError(
            "inference controller was interrupted while waiting for workers",
            code="pipeline.controller_interrupted",
            context={
                "phase": "worker_wait",
                "worker_return_codes": return_codes,
            },
            cause=cause,
        )
        _write_terminal_contract_failure(
            output_dir=run_dir,
            metadata=metadata,
            error=error,
        )
        raise
    worker_statuses = {
        rank: "completed" if code == 0 else "failed"
        for rank, code in sorted(return_codes.items())
    }

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


def _terminate_owned_workers_after_controller_failure(
    *,
    worker_module: Any,
    launched: Sequence[tuple[RankShardPlan, Any]],
    run_dir: Path,
    metadata: dict[str, Any],
    phase: str,
    cause: BaseException,
) -> dict[int, int | None]:
    try:
        return worker_module.terminate_worker_processes(
            tuple((rank_plan.rank, process) for rank_plan, process in launched)
        )
    except RuntimeContractError as cleanup_error:
        error = RuntimeContractError(
            "inference controller could not fully terminate its owned workers",
            code=cleanup_error.code,
            context={
                **cleanup_error.context,
                "controller_phase": phase,
                "trigger_exception_type": type(cause).__name__,
                "trigger_error": str(cause),
            },
            cause=cleanup_error,
        )
        _write_terminal_contract_failure(
            output_dir=run_dir,
            metadata=metadata,
            error=error,
        )
        raise error from cause


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
    frontend_factory: FrontendFactory | None,
    session_opener: BackendSessionOpener | None,
    execution_model: dict[str, Any] | None,
) -> None:
    try:
        factory = frontend_factory or assemble_frontend
        frontend_kwargs: dict[str, Any] = {
            "generation_config_fingerprint": metadata[
                "generation_config_fingerprint"
            ]
        }
        if execution_model is not None:
            frontend_kwargs["execution_model"] = execution_model
        frontend = factory(resolved.config, **frontend_kwargs)
        _execute_indexed_rows(
            resolved=resolved,
            output_dir=output_dir,
            indexed_raw_examples=indexed_raw_examples,
            metadata=metadata,
            frontend=frontend,
            session_opener=session_opener,
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
    except torch.OutOfMemoryError as exc:
        _write_terminal_unhandled_failure(
            output_dir=output_dir,
            metadata=metadata,
            failure_class="cuda_oom",
            error_code="inference.cuda_oom",
            error=exc,
        )
        raise
    except Exception as exc:
        _write_terminal_unhandled_failure(
            output_dir=output_dir,
            metadata=metadata,
            failure_class="runtime_failure",
            error_code="inference.unhandled_runtime_failure",
            error=exc,
        )
        raise


def _resolve_execution_model_for_run(
    resolved: ResolvedInferConfig,
) -> dict[str, Any] | None:
    config = resolved.config
    if config.backend.type != "vllm":
        return None
    return resolve_execution_model(
        base_model_path=config.model.base_model,
        target_dtype=config.model.dtype,
        adapter_path=None if config.adapter is None else config.adapter.path,
        adapter_name="default" if config.adapter is None else config.adapter.name,
        embedding_delta_path=(
            None if config.embedding_delta is None else config.embedding_delta.path
        ),
    )


def _write_execution_model_artifact(
    *,
    run_dir: Path,
    execution_model: dict[str, Any] | None,
) -> Path | None:
    if execution_model is None:
        return None
    validated = validate_execution_model_receipt(execution_model)
    path = run_dir / "execution_model.json"
    _write_json(path, validated)
    return path


def _execute_indexed_rows(
    *,
    resolved: ResolvedInferConfig,
    output_dir: Path,
    indexed_raw_examples: tuple[tuple[int, RawExample], ...],
    metadata: dict[str, Any],
    frontend: InferenceFrontend,
    session_opener: BackendSessionOpener | None,
) -> None:
    qwen = frontend.qwen
    verify_processor_model_vision_parity(
        processor_identity=qwen.processor_identity,
        model_config=_model_config(qwen),
    )
    raw_examples = [raw_example for _, raw_example in indexed_raw_examples]
    image_plan_batch = plan_image_batch(
        raw_examples,
        components=qwen,
        processor_config=_processor_config(resolved.config),
        row_indices=[row_index for row_index, _ in indexed_raw_examples],
    )
    image_plan_by_row_id = {
        row.row_id: row for row in image_plan_batch.rows
    }
    prompt_records = [
        build_prompt_record(
            raw_example,
            _template_config(resolved.config),
            processor=qwen.processor,
            row_index=row_index,
            merged_visual_tokens=image_plan_by_row_id[
                raw_example.example_id
            ].merged_visual_tokens,
        )
        for row_index, raw_example in indexed_raw_examples
    ]
    generation_policy = GenerationPolicy(
        max_new_tokens=resolved.config.generation.max_new_tokens,
        repetition_penalty=resolved.config.generation.repetition_penalty,
        temperature=resolved.config.generation.temperature,
        top_p=resolved.config.generation.top_p,
        include_raw_model_logprob=(
            resolved.config.artifacts.include_raw_model_logprob
        ),
    )
    requests = [
        DecodeRequest(
            request_id=record.row_id,
            chat_text=record.chat_text,
            input_prompt_token_ids=tuple(record.input_prompt_token_ids),
            expected_executed_prompt_token_ids=tuple(
                record.expected_executed_prompt_token_ids
            ),
            image_path=image_plan_by_row_id[record.row_id].image_path,
            declared_image_width=image_plan_by_row_id[record.row_id].declared_width,
            declared_image_height=image_plan_by_row_id[record.row_id].declared_height,
            decoded_image_width=image_plan_by_row_id[record.row_id].decoded_width,
            decoded_image_height=image_plan_by_row_id[record.row_id].decoded_height,
            image_sha256=image_plan_by_row_id[
                record.row_id
            ].image_content_sha256,
            expected_image_grid_thw=tuple(
                image_plan_by_row_id[record.row_id].expected_image_grid_thw
            ),
            logical_transform_id=image_plan_by_row_id[
                record.row_id
            ].logical_transform_id,
            generation_policy=generation_policy,
        )
        for record in prompt_records
    ]
    metadata["media_identity"] = {
        "rows": [
            {
                "row_id": request.request_id,
                "image_sha256": request.image_sha256,
                "decoded_width": request.decoded_image_width,
                "decoded_height": request.decoded_image_height,
                "expected_image_grid_thw": list(
                    request.expected_image_grid_thw or ()
                ),
                "logical_transform_id": request.logical_transform_id,
            }
            for request in requests
        ]
    }

    with open_backend_session(frontend.launch, opener=session_opener) as session:
        metadata.update(
            _session_metadata(frontend=frontend, receipt=session.receipt)
        )
        _fill_worker_runtime_device_metadata(
            metadata=metadata,
            session_receipt=session.receipt,
        )
        results = validate_decode_results(
            requests=requests,
            results=session.decode(requests),
            receipt=session.receipt,
        )
        # Decode may add executed backend evidence, such as vLLM raw replay.
        metadata.update(
            _session_metadata(frontend=frontend, receipt=session.receipt)
        )
        metadata["prompt_trace"] = _prompt_trace(
            requests=requests,
            results=results,
        )
        raw_replay_trace = _raw_replay_trace(results=results)
        if raw_replay_trace:
            metadata["raw_replay_trace"] = raw_replay_trace

    decode_results = {result.request_id: result for result in results}
    for result in results:
        record = _prompt_record_by_id(prompt_records, result.request_id)
        verify_prompt_token_parity(
            record,
            backend_prompt_token_ids=list(result.prompt_token_ids),
        )

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
        image_plan_rows=[
            _image_plan_artifact_dict(row=row)
            for row in _image_plan_rows_with_backend_evidence(
                rows=image_plan_batch.rows,
                decode_results=decode_results,
                backend=str(metadata["backend"]),
            )
        ],
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


def _write_terminal_unhandled_failure(
    *,
    output_dir: Path,
    metadata: dict[str, Any],
    failure_class: str,
    error_code: str,
    error: Exception,
) -> None:
    write_terminal_status_artifacts(
        output_dir=output_dir,
        metadata=metadata,
        summary={
            "terminal_status": "failed",
            "failure_class": failure_class,
            f"{failure_class}_count": 1,
            "error": {
                "code": error_code,
                "message": str(error),
                "context": {"exception_type": type(error).__name__},
            },
        },
    )


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
        "raw_model_logprob_enabled": bool(
            resolved.config.artifacts.include_raw_model_logprob
        ),
        "benchmark_eligible": False,
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
        "backend_mode": (
            "generate"
            if resolved.config.backend.type == "hf"
            else "offline_generate"
        ),
        "response_family": resolved.config.backend.type,
        "model_identity": {},
        "tokenizer_identity": {},
    }


def _benchmark_scope_eligible(*, config: InferConfig, row_count: int) -> bool:
    """V1 benchmark claims require the accepted val200-or-larger scope."""

    return not config.debug.smoke and benchmark_scope_eligible(row_count)


def _session_metadata(*, frontend: InferenceFrontend, receipt: Any) -> dict[str, Any]:
    model_identity = dict(receipt.model_identity)
    tokenizer_identity = dict(receipt.tokenizer_identity)
    processor_identity = dict(receipt.processor_identity)
    return {
        "backend": receipt.backend,
        "backend_mode": receipt.backend_mode,
        "response_family": receipt.response_family,
        "model_identity": model_identity,
        "tokenizer_identity": tokenizer_identity,
        "processor_identity": processor_identity,
        "model_identity_fingerprint": _fingerprint(model_identity),
        "processor_identity_fingerprint": _fingerprint(processor_identity),
        "adapter_identity": model_identity.get("adapter"),
        "embedding_delta_identity": model_identity.get("embedding_delta"),
        "backend_session": receipt.to_artifact_dict(),
        "likelihood_semantics": dict(receipt.likelihood_semantics),
        "execution_model_identity": receipt.execution_model_identity,
        "frontend_identity": frontend.qwen.to_artifact_dict(),
    }


def _prompt_trace(
    *,
    requests: Sequence[DecodeRequest],
    results: Sequence[Any],
) -> list[dict[str, Any]]:
    results_by_id = {result.request_id: result for result in results}
    rows: list[dict[str, Any]] = []
    for request in requests:
        result = results_by_id[request.request_id]
        input_ids = request.input_prompt_token_ids
        expected_ids = request.expected_executed_prompt_token_ids
        executed_ids = result.executed_prompt_token_ids
        rows.append(
            {
                "row_id": request.request_id,
                "input_prompt_token_count": len(input_ids),
                "input_prompt_token_ids_sha256": token_ids_sha256(input_ids),
                "expected_executed_prompt_token_count": len(expected_ids),
                "expected_executed_prompt_token_ids_sha256": token_ids_sha256(
                    expected_ids
                ),
                "backend_executed_prompt_token_count": len(executed_ids),
                "backend_executed_prompt_token_ids_sha256": token_ids_sha256(
                    executed_ids
                ),
                "prompt_token_parity": "verified",
            }
        )
    return rows


def _raw_replay_trace(*, results: Sequence[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        evidence = result.native_evidence.get("raw_replay")
        if evidence is None:
            continue
        rows.append({"row_id": result.request_id, **dict(evidence)})
    return rows


def _fill_worker_runtime_device_metadata(
    *,
    metadata: dict[str, Any],
    session_receipt: Any,
) -> None:
    parallelism = metadata.get("parallelism")
    if not isinstance(parallelism, dict):
        return
    effective_settings = dict(session_receipt.effective_settings)
    device_value = effective_settings.get("device")
    device = None if device_value is None else str(device_value)
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


def _image_plan_rows_with_backend_evidence(
    *,
    rows: Sequence[Any],
    decode_results: dict[str, Any],
    backend: str,
) -> list[Any]:
    evidence_kind = (
        "hf_executed_tensors" if backend == "hf" else "vllm_executed_media"
    )
    return [
        replace(
            row,
            observed_image_grid_thw=(
                None
                if decode_results[row.row_id].observed_image_grid_thw is None
                else list(decode_results[row.row_id].observed_image_grid_thw)
            ),
            backend_projection_evidence_kind=evidence_kind,
            executed_media_sha256=decode_results[
                row.row_id
            ].executed_media_sha256,
            backend_prompt_token_count=len(
                decode_results[row.row_id].executed_prompt_token_ids
            ),
            backend_image_placeholder_ranges=list(
                decode_results[row.row_id].native_evidence.get(
                    "image_placeholder_ranges",
                    [],
                )
            ),
        )
        for row in rows
    ]


def _image_plan_artifact_dict(*, row: Any) -> dict[str, Any]:
    return row.to_artifact_dict()


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
        "include_raw_model_logprob": bool(
            config.artifacts.include_raw_model_logprob
        ),
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


def _fingerprint(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _timestamp_suffix() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
