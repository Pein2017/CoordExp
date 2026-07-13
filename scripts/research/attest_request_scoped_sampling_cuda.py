#!/usr/bin/env python3
"""Run or validate typed Compute Unified Device Architecture (CUDA) and
Qwen3-VL request-scoped sampling attestation.

This command is a mechanics gate only.  In one model-loading process it executes
the four frozen calibration images at temperatures 0.2, 0.4, and 0.6.  Every
policy runs batch-size-four and batch-size-three forward/reverse layouts, stock
versus custom processed-logit parity, capability minting, and one admitted
sampled-production replay.  Only after all three policies pass does it write one
typed, fingerprinted, append-only JavaScript Object Notation aggregate.  It
never writes metric-bearing inference artifacts.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterator

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.analysis.spatial_scope_history import derive_sampling_seed  # noqa: E402
from src.config.fingerprint import sha256_file, sha256_json  # noqa: E402
from src.config.inference import load_infer_config  # noqa: E402
from src.data import load_raw_examples  # noqa: E402
from src.inference.backend import (  # noqa: E402
    ADMITTED_PRODUCTION_REPLAY_SCHEMA_VERSION,
    EXPECTED_ATTESTED_RUNTIME_PAYLOAD_BYTE_COUNT,
    EXPECTED_ATTESTED_RUNTIME_PAYLOAD_TENSOR_COUNT,
    SAMPLED_RUNTIME_ATTESTATION_AGGREGATE_OUTPUT_SCHEMA_VERSION,
    SAMPLED_RUNTIME_ATTESTATION_POLICY_ENTRY_SCHEMA_VERSION,
    DecodeGenerationPolicy,
    DecodeRequest,
    DecodeResult,
    ExecutedSampledRuntimeAttestationCase,
    HFGenerateBackend,
    SampledRuntimeAttestationBundle,
    VerifiedSampledRuntimeAttestation,
    _compact_object_selected_token_score_replay,
    validate_sampled_runtime_attestation_aggregate_output,
    verify_checkpoint_payload_identity,
)
from src.inference.image_plan import materialize_image_plan_batch  # noqa: E402
from src.inference.pipeline import (  # noqa: E402
    _processor_config,
    _template_config,
    _tokenizer_identity,
)
from src.inference.prompt import build_prompt_record  # noqa: E402
from src.inference.runtime import assemble_runtime  # noqa: E402


CALIBRATION_COHORT_ID = "sampling-calibration-12"
CALIBRATION_MANIFEST_SHA256 = (
    "73edd29504dc526f54f36c543dfeb4ca03bf6fd15c82449116e7e4b546f58496"
)
PRIMARY_CONFIG_SHA256 = (
    "f3000588accbcf1d9ada3b2f3e0b3324d660b4810b75d8f5d050d9f184f9ca80"
)
PRIMARY_CHECKPOINT_SHA256 = (
    "613e5d97f4a7a53d6325b5c1909813d6bb82e72622942b225df9724b5556a536"
)
AUTHORIZED_CONFIG_TO_CHECKPOINT_SHA256 = {
    PRIMARY_CONFIG_SHA256: PRIMARY_CHECKPOINT_SHA256,
}
ROOT_SEED = 2026071301
TEMPERATURES = (0.2, 0.4, 0.6)
MAX_NEW_TOKENS = 512
TOP_P = 0.95
REPETITION_PENALTY = 1.0
SOURCE_VALIDATION_JSONL = Path(
    "/data/CoordExp/public_data/coco/rescale_32_1024_bbox_max60/val.coord.jsonl"
)
SOURCE_VALIDATION_SHA256 = (
    "18a3cad3b7ad847ecf39949fe751d963008dbb7707f796c742c3fa23ae3c8e8b"
)
@contextmanager
def _temporary_cwd(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _require_file(path: Path, *, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"{label} does not exist: {path}")
    return path


def _load_calibration_records(path: Path) -> list[dict[str, Any]]:
    rows = _read_jsonl(path)
    if len(rows) != 12:
        raise RuntimeError(f"calibration manifest must contain 12 rows, got {len(rows)}")
    records: list[dict[str, Any]] = []
    for row in rows:
        if row.get("cohort_id") != CALIBRATION_COHORT_ID:
            raise RuntimeError("calibration manifest contains an unexpected cohort")
        record = row.get("record")
        if not isinstance(record, dict):
            raise RuntimeError("calibration manifest row is missing record")
        records.append(record)
    records.sort(key=lambda record: int(record["frozen_order"]))
    selected = records[:4]
    if [int(record["frozen_order"]) for record in selected] != [0, 1, 2, 3]:
        raise RuntimeError("calibration manifest does not begin with frozen orders 0..3")
    return selected


def _load_source_examples(records: list[dict[str, Any]]) -> list[Any]:
    source = _require_file(SOURCE_VALIDATION_JSONL, label="processed validation source")
    if sha256_file(source) != SOURCE_VALIDATION_SHA256:
        raise RuntimeError("processed validation source digest is not frozen")
    if {
        str(record["source_dataset_sha256"])
        for record in records
    } != {SOURCE_VALIDATION_SHA256}:
        raise RuntimeError("calibration records disagree with the frozen source digest")
    by_image_id: dict[int, Any] = {}
    for example in load_raw_examples(source):
        source_metadata = example.metadata.get("source", {})
        image_id = source_metadata.get("image_id")
        if isinstance(image_id, int):
            by_image_id[image_id] = example
    selected: list[Any] = []
    for record in records:
        image_id = int(record["image_id"])
        example = by_image_id.get(image_id)
        if example is None:
            raise RuntimeError(f"calibration image {image_id} is absent from {source}")
        if Path(record["image_path"]).name != example.image.path.name:
            raise RuntimeError(f"calibration image filename mismatch for image {image_id}")
        if sha256_file(example.image.path) != str(record["image_sha256"]):
            raise RuntimeError(f"calibration image digest mismatch for image {image_id}")
        if example.source.row_number != int(record["source_row_index"]) + 1:
            raise RuntimeError(f"calibration source row mismatch for image {image_id}")
        if example.source.row_sha256 != str(record["source_row_sha256"]):
            raise RuntimeError(f"calibration source row digest mismatch for image {image_id}")
        if example.image.width != int(record["source_width"]):
            raise RuntimeError(f"calibration source width mismatch for image {image_id}")
        if example.image.height != int(record["source_height"]):
            raise RuntimeError(f"calibration source height mismatch for image {image_id}")
        selected.append(example)
    return selected


def _validate_config(
    resolved: Any,
    *,
    config_path: Path,
    checkpoint_manifest: Path,
) -> None:
    config = resolved.config
    config_digest = sha256_file(config_path)
    expected_checkpoint_digest = AUTHORIZED_CONFIG_TO_CHECKPOINT_SHA256.get(
        config_digest
    )
    if expected_checkpoint_digest is None:
        raise RuntimeError("config is not an authorized frozen Qwen3-VL lineage")
    if sha256_file(checkpoint_manifest) != expected_checkpoint_digest:
        raise RuntimeError("config and checkpoint manifest are not the same lineage")
    verify_checkpoint_payload_identity(checkpoint_manifest)
    expected = {
        "dtype": "bf16",
        "attention": "sdpa",
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_resize": False,
    }
    observed = {
        "dtype": config.model.dtype,
        "attention": config.model.attn_implementation,
        "max_new_tokens": config.generation.max_new_tokens,
        "do_resize": bool(config.model.processor.do_resize),
    }
    if observed != expected:
        raise RuntimeError(f"config violates frozen sampling factors: {observed}")
    if config.backend.type != "hf":
        raise RuntimeError("sampling attestation requires the Hugging Face backend")


def _build_requests(
    *,
    examples: list[Any],
    resolved: Any,
    qwen: Any,
    temperature: float,
) -> tuple[list[DecodeRequest], dict[str, Any]]:
    image_plan = materialize_image_plan_batch(
        examples,
        components=qwen,
        processor_config=_processor_config(resolved.config),
        materialize=True,
        row_indices=list(range(len(examples))),
    )
    records = [
        build_prompt_record(
            example,
            _template_config(resolved.config),
            processor=qwen.processor,
            row_index=index,
        )
        for index, example in enumerate(examples)
    ]
    policy = DecodeGenerationPolicy.sampled(
        max_new_tokens=MAX_NEW_TOKENS,
        repetition_penalty=REPETITION_PENALTY,
        temperature=temperature,
        top_p=TOP_P,
    )
    requests: list[DecodeRequest] = []
    for index, record in enumerate(records):
        image_id = int(examples[index].metadata["source"]["image_id"])
        requests.append(
            DecodeRequest(
                request_id=record.row_id,
                prompt_token_ids=list(record.prompt_token_ids),
                model_inputs=image_plan.model_inputs_by_row_id[record.row_id],
                generation_policy=policy,
                sampling_seed=derive_sampling_seed(
                    root_seed=ROOT_SEED,
                    role="temperature-calibration",
                    image_id=image_id,
                    cell_or_call_label=f"call-{index:02d}",
                ),
            )
        )
    return requests, {
        "image_plan_rows": [row.to_artifact_dict() for row in image_plan.rows],
        "prompt_records": [record.to_artifact_dict() for record in records],
    }


def _lineage(
    *,
    config_path: Path,
    checkpoint_manifest: Path,
    calibration_manifest: Path,
    resolved: Any,
    qwen: Any,
    records: list[dict[str, Any]],
    prompt_artifacts: dict[str, Any],
    temperature: float,
    request_ids: list[str],
) -> dict[str, Any]:
    return {
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "checkpoint_manifest_path": str(checkpoint_manifest),
        "checkpoint_manifest_sha256": sha256_file(checkpoint_manifest),
        "checkpoint_payload_identity": verify_checkpoint_payload_identity(
            checkpoint_manifest
        ),
        "calibration_manifest_path": str(calibration_manifest),
        "calibration_manifest_sha256": sha256_file(calibration_manifest),
        "calibration_request_ids": request_ids,
        "calibration_image_ids": [int(record["image_id"]) for record in records],
        "calibration_image_sha256": [
            str(record["image_sha256"]) for record in records
        ],
        "calibration_prompt_records_fingerprint": sha256_json(
            prompt_artifacts["prompt_records"]
        ),
        "calibration_image_plan_fingerprint": sha256_json(
            prompt_artifacts["image_plan_rows"]
        ),
        "model_dtype": resolved.config.model.dtype,
        "attention_implementation": resolved.config.model.attn_implementation,
        "qwen_runtime_identity": qwen.to_artifact_dict(),
        "temperature": temperature,
        "root_seed": ROOT_SEED,
        "frozen_sampling_factors": {
            "max_new_tokens": MAX_NEW_TOKENS,
            "top_p": TOP_P,
            "repetition_penalty": REPETITION_PENALTY,
        },
    }


def _write_canonical(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite append-only attestation: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def _validate_runtime_state_seal_diagnostics(
    diagnostics: Mapping[str, Any] | None,
    *,
    expected_tensor_count: int = EXPECTED_ATTESTED_RUNTIME_PAYLOAD_TENSOR_COUNT,
    expected_payload_byte_count: int = EXPECTED_ATTESTED_RUNTIME_PAYLOAD_BYTE_COUNT,
) -> dict[str, Any]:
    if diagnostics is None:
        raise RuntimeError("admitted production call did not emit live-state diagnostics")
    payload = dict(diagnostics)
    expected = {
        "measurement_scope": "physical_backend_call_pre_generation",
        "adapter_and_selected_embedding_tensor_count": expected_tensor_count,
        "adapter_and_selected_embedding_payload_byte_count": (
            expected_payload_byte_count
        ),
        "base_model_payload_hashed": False,
    }
    mismatches = {
        field: {"expected": value, "observed": payload.get(field)}
        for field, value in expected.items()
        if payload.get(field) != value
    }
    timing_fields = (
        "adapter_and_selected_embedding_hash_elapsed_seconds",
        "live_runtime_state_seal_elapsed_seconds",
    )
    invalid_timings = {
        field: payload.get(field)
        for field in timing_fields
        if not isinstance(payload.get(field), (int, float))
        or isinstance(payload.get(field), bool)
        or not math.isfinite(float(payload[field]))
        or float(payload[field]) < 0.0
    }
    if (
        not invalid_timings
        and float(payload["live_runtime_state_seal_elapsed_seconds"])
        < float(payload["adapter_and_selected_embedding_hash_elapsed_seconds"])
    ):
        invalid_timings["live_runtime_state_seal_elapsed_seconds"] = payload[
            "live_runtime_state_seal_elapsed_seconds"
        ]
    if mismatches or invalid_timings:
        raise RuntimeError(
            "admitted production live-state diagnostics violate the frozen contract: "
            f"mismatches={mismatches}, invalid_timings={invalid_timings}"
        )
    return payload


def _build_admitted_production_replay_evidence(
    *,
    requests: Sequence[DecodeRequest],
    results: Sequence[DecodeResult],
    attestation_case: ExecutedSampledRuntimeAttestationCase,
    image_sizes_by_request: Mapping[str, tuple[int, int]],
    runtime_state_seal_diagnostics: Mapping[str, Any],
    capability_bundle_payload_fingerprint: str,
    decode_generation_policy_fingerprint: str,
    portable_runtime_state_seal: Mapping[str, Any],
) -> dict[str, Any]:
    request_ids = [request.request_id for request in requests]
    result_ids = [result.request_id for result in results]
    if result_ids != request_ids:
        raise RuntimeError(
            "admitted production result order does not match frozen request order: "
            f"requests={request_ids}, results={result_ids}"
        )
    case_artifact = attestation_case.to_artifact_dict()
    if attestation_case.case_name != "batch_size_four_forward":
        raise RuntimeError("admitted production replay requires the four-forward case")
    if list(attestation_case.request_ids) != request_ids:
        raise RuntimeError("four-forward attestation request order is not canonical")
    expected_artifacts = {
        str(artifact["request_id"]): artifact
        for artifact in case_artifact["result_artifacts"]
    }
    expected_selected_replays = case_artifact[
        "compact_object_selected_token_score_replays_by_request"
    ]
    replay_rows = []
    for execution_index, (request, result) in enumerate(
        zip(requests, results, strict=True)
    ):
        result.validate_for_scored()
        expected_artifact = expected_artifacts.get(request.request_id)
        if expected_artifact is None:
            raise RuntimeError(
                f"attestation four-forward case lacks {request.request_id}"
            )
        actual_artifact = result.to_artifact_dict()
        exact_result_artifact_replay = actual_artifact == expected_artifact
        image_size = image_sizes_by_request.get(request.request_id)
        if image_size is None:
            raise RuntimeError(
                f"admitted production replay lacks image size for {request.request_id}"
            )
        actual_selected_replay = _compact_object_selected_token_score_replay(
            result,
            image_width=image_size[0],
            image_height=image_size[1],
        )
        expected_selected_replay = expected_selected_replays.get(request.request_id)
        exact_selected_token_score_replay = (
            actual_selected_replay == expected_selected_replay
        )
        receipt = result.execution_receipt
        if receipt is None:
            raise RuntimeError(
                f"admitted production result lacks receipt for {request.request_id}"
            )
        receipt_binding_valid = (
            receipt.request_id == request.request_id
            and receipt.request_execution_index == execution_index
            and receipt.sampling_seed == request.sampling_seed
            and receipt.random_generator_initial_seed == request.sampling_seed
        )
        if not (
            exact_result_artifact_replay
            and exact_selected_token_score_replay
            and receipt_binding_valid
        ):
            raise RuntimeError(
                "admitted production result did not exactly replay the attested "
                f"four-forward request {request.request_id}: "
                f"result={exact_result_artifact_replay}, "
                f"selected_score={exact_selected_token_score_replay}, "
                f"receipt={receipt_binding_valid}"
            )
        replay_rows.append(
            {
                "request_id": request.request_id,
                "execution_index": execution_index,
                "sampling_seed": request.sampling_seed,
                "exact_result_artifact_replay": True,
                "exact_selected_token_score_replay": True,
                "receipt_binding_valid": True,
                "attestation_result_artifact_fingerprint": sha256_json(
                    expected_artifact
                ),
                "admitted_result_artifact_fingerprint": sha256_json(
                    actual_artifact
                ),
                "receipt_fingerprint": receipt.receipt_fingerprint,
                "generated_token_identifiers_hash": (
                    receipt.generated_token_identifiers_hash
                ),
                "canonical_float32_score_trace_hash": (
                    receipt.canonical_float32_score_trace_hash
                ),
                "compact_object_selected_token_score_replay": (
                    actual_selected_replay
                ),
            }
        )
    payload: dict[str, Any] = {
        "schema_version": ADMITTED_PRODUCTION_REPLAY_SCHEMA_VERSION,
        "capability_gated_backend_api": (
            "HFGenerateBackend.generate_batch_with_verified_runtime_attestation"
        ),
        "capability_bundle_payload_fingerprint": (
            capability_bundle_payload_fingerprint
        ),
        "decode_generation_policy_fingerprint": (
            decode_generation_policy_fingerprint
        ),
        "compared_attestation_case": attestation_case.case_name,
        "request_ids": request_ids,
        "exact_request_order_replay": True,
        "exact_result_artifact_replay": True,
        "exact_selected_token_score_replay": True,
        "runtime_state_seal_diagnostics": dict(runtime_state_seal_diagnostics),
        "portable_runtime_state_seal": dict(portable_runtime_state_seal),
        "result_replays": replay_rows,
    }
    payload["replay_payload_fingerprint"] = sha256_json(payload)
    return payload


def _execute_admitted_production_replay(
    *,
    backend: HFGenerateBackend,
    requests: Sequence[DecodeRequest],
    bundle: SampledRuntimeAttestationBundle,
    capability: VerifiedSampledRuntimeAttestation,
    model_identity: Mapping[str, Any],
    tokenizer_identity: Mapping[str, Any],
    generation_config_fingerprint: str,
    image_sizes_by_request: Mapping[str, tuple[int, int]],
    expected_tensor_count: int = EXPECTED_ATTESTED_RUNTIME_PAYLOAD_TENSOR_COUNT,
    expected_payload_byte_count: int = EXPECTED_ATTESTED_RUNTIME_PAYLOAD_BYTE_COUNT,
) -> dict[str, Any]:
    if capability.bundle_payload_fingerprint != bundle.bundle_payload_fingerprint:
        raise RuntimeError("production capability is not bound to this bundle")
    admitted_results = backend.generate_batch_with_verified_runtime_attestation(
        requests,
        model_identity=model_identity,
        tokenizer_identity=tokenizer_identity,
        generation_config_fingerprint=generation_config_fingerprint,
        verified_runtime_attestation=capability,
    )
    diagnostics = _validate_runtime_state_seal_diagnostics(
        backend.last_runtime_state_seal_diagnostics,
        expected_tensor_count=expected_tensor_count,
        expected_payload_byte_count=expected_payload_byte_count,
    )
    cases = {
        case.case_name: case
        for case in bundle.executed_cases
    }
    four_forward = cases.get("batch_size_four_forward")
    if four_forward is None:
        raise RuntimeError("attestation bundle lacks the four-forward replay source")
    return _build_admitted_production_replay_evidence(
        requests=requests,
        results=admitted_results,
        attestation_case=four_forward,
        image_sizes_by_request=image_sizes_by_request,
        runtime_state_seal_diagnostics=diagnostics,
        capability_bundle_payload_fingerprint=(
            capability.bundle_payload_fingerprint
        ),
        decode_generation_policy_fingerprint=requests[
            0
        ].generation_policy.fingerprint,
        portable_runtime_state_seal=(
            capability.portable_runtime_state_seal_artifact()
        ),
    )


def _build_policy_attestation_entry(
    *,
    policy: DecodeGenerationPolicy,
    bundle: SampledRuntimeAttestationBundle,
    admitted_production_replay: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        admitted_production_replay.get("capability_bundle_payload_fingerprint")
        != bundle.bundle_payload_fingerprint
        or admitted_production_replay.get(
            "decode_generation_policy_fingerprint"
        )
        != policy.fingerprint
    ):
        raise RuntimeError(
            "policy attestation entry does not bind its bundle and admitted replay"
        )
    payload: dict[str, Any] = {
        "schema_version": SAMPLED_RUNTIME_ATTESTATION_POLICY_ENTRY_SCHEMA_VERSION,
        "temperature": float(policy.temperature),
        "decode_generation_policy": policy.to_artifact_dict(),
        "decode_generation_policy_fingerprint": policy.fingerprint,
        "bundle_payload_fingerprint": bundle.bundle_payload_fingerprint,
        "attestation_bundle": bundle.to_artifact_dict(),
        "admitted_production_replay": dict(admitted_production_replay),
    }
    payload["entry_payload_fingerprint"] = sha256_json(payload)
    return payload


def _build_aggregate_attestation_output(
    policy_attestations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    entries = [dict(entry) for entry in policy_attestations]
    temperatures = [entry.get("temperature") for entry in entries]
    if temperatures != list(TEMPERATURES):
        raise RuntimeError(
            "production attestation requires the exact ordered temperature set "
            f"{TEMPERATURES}, got {temperatures}"
        )
    payload: dict[str, Any] = {
        "schema_version": (
            SAMPLED_RUNTIME_ATTESTATION_AGGREGATE_OUTPUT_SCHEMA_VERSION
        ),
        "policy_attestations": entries,
    }
    payload["aggregate_payload_fingerprint"] = sha256_json(payload)
    return payload


def _attest_all_policies_then_write_output(
    *,
    output: Path,
    policy_attestation_factory: Callable[[float], Mapping[str, Any]],
) -> dict[str, Any]:
    """Run every frozen policy before the aggregate's only append-only write."""

    entries = [
        dict(policy_attestation_factory(temperature))
        for temperature in TEMPERATURES
    ]
    artifact = _build_aggregate_attestation_output(entries)
    _write_canonical(output, artifact)
    return artifact


def _execute_single_policy_attestation(
    *,
    temperature: float,
    backend: HFGenerateBackend,
    examples: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    resolved: Any,
    qwen: Any,
    config_path: Path,
    checkpoint_manifest: Path,
    calibration_manifest: Path,
    model_identity: Mapping[str, Any],
    tokenizer_identity: Mapping[str, Any],
    generation_config_fingerprint: str,
    image_sizes_by_request: Mapping[str, tuple[int, int]],
) -> dict[str, Any]:
    requests, prompt_artifacts = _build_requests(
        examples=list(examples),
        resolved=resolved,
        qwen=qwen,
        temperature=temperature,
    )
    lineage = _lineage(
        config_path=config_path,
        checkpoint_manifest=checkpoint_manifest,
        calibration_manifest=calibration_manifest,
        resolved=resolved,
        qwen=qwen,
        records=list(records),
        prompt_artifacts=prompt_artifacts,
        temperature=temperature,
        request_ids=[request.request_id for request in requests],
    )
    bundle, capability = backend.run_request_scoped_sampling_attestation(
        requests,
        lineage=lineage,
        model_identity=model_identity,
        tokenizer_identity=tokenizer_identity,
        generation_config_fingerprint=generation_config_fingerprint,
    )
    admitted = _execute_admitted_production_replay(
        backend=backend,
        requests=requests,
        bundle=bundle,
        capability=capability,
        model_identity=model_identity,
        tokenizer_identity=tokenizer_identity,
        generation_config_fingerprint=generation_config_fingerprint,
        image_sizes_by_request=image_sizes_by_request,
    )
    return _build_policy_attestation_entry(
        policy=requests[0].generation_policy,
        bundle=bundle,
        admitted_production_replay=admitted,
    )


def _validate_only(output: Path) -> None:
    validation = validate_sampled_runtime_attestation_aggregate_output(output)
    print(
        json.dumps(
            {
                "status": "verified",
                "aggregate_payload_fingerprint": validation[
                    "aggregate_payload_fingerprint"
                ],
                "production_admission": validation["production_admission"],
                "decode_generation_policy_fingerprints": validation[
                    "decode_generation_policy_fingerprints"
                ],
                "output": str(output),
            },
            sort_keys=True,
        )
    )


def run(arguments: argparse.Namespace) -> None:
    output = Path(arguments.output).expanduser().resolve()
    if arguments.validate_only:
        _require_file(output, label="attestation bundle")
        _validate_only(output)
        return
    config_path = _require_file(Path(arguments.config), label="inference config")
    checkpoint_manifest = _require_file(
        Path(arguments.checkpoint_manifest), label="checkpoint manifest"
    )
    calibration_manifest = _require_file(
        Path(arguments.calibration_manifest), label="calibration manifest"
    )
    if sha256_file(calibration_manifest) != CALIBRATION_MANIFEST_SHA256:
        raise RuntimeError("calibration manifest digest is not the frozen readiness artifact")
    config_root = config_path.parents[3] if len(config_path.parents) >= 4 else REPOSITORY_ROOT
    with _temporary_cwd(config_root):
        resolved = load_infer_config(config_path)
    _validate_config(
        resolved,
        config_path=config_path,
        checkpoint_manifest=checkpoint_manifest,
    )
    records = _load_calibration_records(calibration_manifest)
    examples = _load_source_examples(records)
    runtime = assemble_runtime(resolved.config, source_gate_root=config_root)
    qwen = runtime.qwen
    model_identity = dict(runtime.model_identity)
    tokenizer_identity = _tokenizer_identity(qwen)
    generation_fingerprint = sha256_json(resolved.config.generation.model_dump(mode="json"))
    backend = HFGenerateBackend(
        model=qwen.model,
        tokenizer=qwen.tokenizer,
        model_identity=model_identity,
        tokenizer_identity=tokenizer_identity,
        generation_config_fingerprint=generation_fingerprint,
    )
    image_sizes_by_request = {
        f"coco2017_val_{int(record['image_id']):012d}": (
            int(record["source_width"]),
            int(record["source_height"]),
        )
        for record in records
    }
    artifact = _attest_all_policies_then_write_output(
        output=output,
        policy_attestation_factory=lambda temperature: (
            _execute_single_policy_attestation(
                temperature=temperature,
                backend=backend,
                examples=examples,
                records=records,
                resolved=resolved,
                qwen=qwen,
                config_path=config_path,
                checkpoint_manifest=checkpoint_manifest,
                calibration_manifest=calibration_manifest,
                model_identity=model_identity,
                tokenizer_identity=tokenizer_identity,
                generation_config_fingerprint=generation_fingerprint,
                image_sizes_by_request=image_sizes_by_request,
            )
        ),
    )
    summaries = [
        {
            "temperature": entry["temperature"],
            "decode_generation_policy_fingerprint": entry[
                "decode_generation_policy_fingerprint"
            ],
            "bundle_payload_fingerprint": entry["bundle_payload_fingerprint"],
            "admitted_production_replay_fingerprint": entry[
                "admitted_production_replay"
            ]["replay_payload_fingerprint"],
            "runtime_state_seal_diagnostics": entry[
                "admitted_production_replay"
            ]["runtime_state_seal_diagnostics"],
        }
        for entry in artifact["policy_attestations"]
    ]
    print(
        json.dumps(
            {
                "status": "verified_and_written",
                "aggregate_payload_fingerprint": artifact[
                    "aggregate_payload_fingerprint"
                ],
                "output": str(output),
                "policy_attestations": summaries,
            },
            sort_keys=True,
        )
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-manifest", required=True)
    parser.add_argument("--calibration-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main() -> int:
    try:
        arguments = _parser().parse_args()
        run(arguments)
    except Exception as exc:
        print(f"attestation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
