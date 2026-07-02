"""CoordExp-swift offline inference pipeline orchestration."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.common.errors import ArtifactContractError, EncodingContractError
from src.config.inference import InferConfig, ResolvedInferConfig, load_infer_config, resolve_infer_run_directory
from src.config.models import ProcessorConfig, TemplateConfig, TemplatePromptConfig
from src.config.writer import write_resolved_config_artifacts
from src.data import RawExample, load_raw_examples
from src.inference.artifacts import write_inference_artifacts, write_terminal_status_artifacts
from src.inference.backend import DecodeRequest, HFGenerateBackend
from src.inference.image_plan import materialize_image_plan_rows, verify_processor_model_vision_parity
from src.inference.parsing import PARSER_POLICY, parse_compact_object_box_closed
from src.inference.prompt import TEMPLATE_ID, build_prompt_record, verify_prompt_token_parity
from src.inference.runtime import InferenceRuntime, assemble_runtime
from src.inference.scoring import SCORE_POLICY_FINGERPRINT


RuntimeFactory = Callable[[InferConfig], Any]
BackendFactory = Callable[[Any, InferConfig], Any]


def run(
    *,
    config_path: str | Path,
    runtime_factory: RuntimeFactory | None = None,
    backend_factory: BackendFactory | None = None,
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

    runtime = (runtime_factory or assemble_runtime)(resolved.config)
    qwen = runtime.qwen
    metadata.update(_runtime_metadata(runtime=runtime, qwen=qwen))
    try:
        verify_processor_model_vision_parity(
            processor_identity=qwen.processor_identity,
            model_config=_model_config(qwen),
        )
        raw_examples = list(load_raw_examples(resolved.config.data.input_jsonl))
        image_plan_rows = materialize_image_plan_rows(
            raw_examples,
            components=qwen,
            processor_config=_processor_config(resolved.config),
            materialize=True,
        )
    except EncodingContractError as exc:
        write_terminal_status_artifacts(
            output_dir=run_dir,
            metadata=metadata,
            summary={
                "terminal_status": "failed",
                "failure_class": "image_validation_failure",
                "image_validation_failure_count": 1,
                "error": {"code": exc.code, "message": exc.message, "context": exc.context},
            },
        )
        raise

    backend = (backend_factory or _default_backend_factory)(runtime, resolved.config)
    prompt_records = [
        build_prompt_record(
            raw_example,
            _template_config(resolved.config),
            processor=qwen.processor,
            row_index=index,
        )
        for index, raw_example in enumerate(raw_examples)
    ]
    requests = [
        DecodeRequest(
            request_id=record.row_id,
            prompt_token_ids=list(record.prompt_token_ids),
            model_inputs={},
            max_new_tokens=resolved.config.generation.max_new_tokens,
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
        for result in batch_results:
            record = _prompt_record_by_id(prompt_records, result.request_id)
            verify_prompt_token_parity(record, backend_prompt_token_ids=list(result.prompt_token_ids))
            decode_results[result.request_id] = result

    rows = [
        _artifact_input_row(
            raw_example=raw_example,
            row_index=index,
            decode_result=decode_results[raw_example.example_id],
        )
        for index, raw_example in enumerate(raw_examples)
    ]
    counters = _pipeline_counters(rows=rows, decode_success_count=len(decode_results))
    metadata["pipeline_counters"] = counters
    write_inference_artifacts(
        output_dir=run_dir,
        rows=rows,
        decode_results=decode_results,
        image_plan_rows=[row.to_artifact_dict() for row in image_plan_rows],
        metadata=metadata,
    )
    return 0


def _default_backend_factory(runtime: InferenceRuntime, config: InferConfig) -> HFGenerateBackend:
    if config.backend.type != "hf":
        raise ArtifactContractError(
            "only HF backend is implemented for CoordExp-swift V1 pipeline",
            code="pipeline.backend_not_implemented",
            context={"backend": config.backend.type},
        )
    return HFGenerateBackend(model=runtime.qwen.model, tokenizer=runtime.qwen.tokenizer)


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
        "parse": parse_row,
    }


def _pipeline_counters(*, rows: list[dict[str, Any]], decode_success_count: int) -> dict[str, Any]:
    parser_failure_count = sum(
        1
        for row in rows
        if row["parse"].parse_status not in {"accepted", "accepted_with_drops"}
    )
    dropped_prediction_count = sum(row["parse"].dropped_prediction_count for row in rows)
    return {
        "terminal_status": "completed",
        "decode_success_count": decode_success_count,
        "parser_failure_count": parser_failure_count,
        "dropped_prediction_count": dropped_prediction_count,
        "image_validation_failure_count": 0,
        "score_failure_count": 0,
    }


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
    return {
        "model_identity": model_identity,
        "tokenizer_identity": tokenizer_identity,
        "model_identity_fingerprint": _fingerprint(model_identity),
        "processor_identity_fingerprint": _fingerprint(processor_identity),
        "adapter_identity": getattr(runtime, "adapter_receipt", None),
    }


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
