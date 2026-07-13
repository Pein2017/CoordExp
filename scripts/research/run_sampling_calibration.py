#!/usr/bin/env python3
"""Execute the frozen sampled calibration protocol and write terminal bundles.

This is the producer for ``calibrate_sampling_policy.py``.  It intentionally
does one real Hugging Face generation process and records only the canonical
terminal bundles; all gate counters are reconstructed by the consumer.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
from typing import Any, TypeVar

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.analysis.spatial_scope_history.calibration import (  # noqa: E402
    CALIBRATION_MAX_CALL_COUNT,
    CALIBRATION_TERMINAL_BUNDLE_COLLECTION_SCHEMA_VERSION,
    CALIBRATION_TEMPERATURES,
    FROZEN_SPATIAL_GRID_SPEC_SHA256,
    FROZEN_CHECKPOINT_MANIFEST_SHA256,
    FROZEN_RESOLVED_INFERENCE_CONFIG_SHA256,
    CalibrationRequest,
    CalibrationTerminalBundle,
    build_calibration_requests,
    load_attested_sampling_policy_set,
    load_calibration_backend_attestation_binding,
    reconstruct_calibration_observation,
    select_sampling_calibration_from_terminal_bundles,
    summarize_calibration_panel,
    write_immutable_json,
)
from src.analysis.spatial_scope_history.cohort_ledger import (  # noqa: E402
    CohortLedger,
    sha256_payload,
)
from src.analysis.spatial_scope_history.production_executor import (  # noqa: E402
    _load_exact_raw_examples,
)
from src.analysis.spatial_scope_history.spatial import (  # noqa: E402
    MaterializedVisualInput,
)
from src.config.fingerprint import sha256_file, sha256_json  # noqa: E402
from src.config.inference import load_infer_config  # noqa: E402
from src.inference.backend import (  # noqa: E402
    DecodeGenerationPolicy,
    DecodeRequest,
    HFGenerateBackend,
    load_and_rebind_sampled_runtime_attestation_aggregate,
    verify_checkpoint_payload_identity,
)
from src.inference.pipeline import _template_config, _tokenizer_identity  # noqa: E402
from src.inference.prompt import build_prompt_record  # noqa: E402
from src.inference.runtime import assemble_runtime  # noqa: E402


T = TypeVar("T")


@contextmanager
def _temporary_cwd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def execute_frozen_protocol(
    *,
    requests_by_temperature: Mapping[float, Sequence[Sequence[T]]],
    execute_batch: Callable[[float, str, int, Sequence[T]], Sequence[T]],
    initial_gate: Callable[[float, Sequence[T]], bool],
) -> tuple[T, ...]:
    """Run the exact twelve-image, batch-four, first-passing protocol.

    ``execute_batch`` is the only injected seam.  Production passes a real HF
    backend callback; tests pass a recorder.  This keeps protocol ordering and
    early-stop behavior testable without constructing a model.
    """

    terminal: list[T] = []
    total_calls = 0
    for temperature in CALIBRATION_TEMPERATURES:
        batches = requests_by_temperature.get(temperature)
        if (
            batches is None
            or len(batches) != 12
            or any(len(batch) != 4 for batch in batches)
        ):
            raise RuntimeError(
                "frozen calibration requires twelve batch-size-four groups"
            )
        initial_rows: list[T] = []
        for batch_index, batch in enumerate(batches):
            rows = tuple(
                execute_batch(temperature, "initial", batch_index, tuple(batch))
            )
            if len(rows) != 4:
                raise RuntimeError("calibration backend returned a non-four-row batch")
            initial_rows.extend(rows)
            terminal.extend(rows)
            total_calls += 4
        if not initial_gate(temperature, tuple(initial_rows)):
            continue
        for panel_kind, physical_batches in (
            ("exact_replay", batches),
            ("reversed_order", tuple(tuple(reversed(batch)) for batch in batches)),
        ):
            for batch_index, batch in enumerate(physical_batches):
                rows = tuple(
                    execute_batch(temperature, panel_kind, batch_index, tuple(batch))
                )
                if len(rows) != 4:
                    raise RuntimeError(
                        "calibration backend returned a non-four-row batch"
                    )
                terminal.extend(rows)
                total_calls += 4
        if total_calls > CALIBRATION_MAX_CALL_COUNT:
            raise RuntimeError("frozen calibration protocol exceeded its 240-call cap")
        return tuple(terminal)
    raise RuntimeError("no sampled calibration temperature passed the initial panel")


def _build_collection_payload(
    *,
    terminal_bundles: Sequence[CalibrationTerminalBundle],
    calibration_cohort: CohortLedger,
    validation_cohort: CohortLedger,
    aggregate_sha256: str,
    aggregate_fingerprint: str,
    reference_source_sha256: str,
) -> dict[str, Any]:
    return {
        "artifact_role": "canonical executed non-metric calibration terminal bundles",
        "attestation_aggregate_artifact_sha256": aggregate_sha256,
        "attestation_aggregate_payload_fingerprint": aggregate_fingerprint,
        "calibration_cohort_sha256": calibration_cohort.fingerprint,
        "calibration_reference_source_sha256": reference_source_sha256,
        "metric_eligible": False,
        "schema_version": CALIBRATION_TERMINAL_BUNDLE_COLLECTION_SCHEMA_VERSION,
        "terminal_bundles": [bundle.to_artifact_dict() for bundle in terminal_bundles],
        "validation_cohort_sha256": validation_cohort.fingerprint,
    }


def _main(arguments: argparse.Namespace) -> int:
    infer_config_path = arguments.infer_config.expanduser().resolve()
    checkpoint_manifest = arguments.checkpoint_manifest.expanduser().resolve()
    aggregate_path = (
        arguments.sampled_runtime_attestation_aggregate.expanduser().resolve()
    )
    if sha256_file(infer_config_path) != FROZEN_RESOLVED_INFERENCE_CONFIG_SHA256:
        raise RuntimeError(
            "inference config differs from the frozen calibration runtime"
        )
    if sha256_file(checkpoint_manifest) != FROZEN_CHECKPOINT_MANIFEST_SHA256:
        raise RuntimeError(
            "checkpoint manifest differs from the frozen calibration runtime"
        )
    verify_checkpoint_payload_identity(checkpoint_manifest)
    calibration_cohort = CohortLedger.from_jsonl_bytes(
        arguments.calibration_cohort_manifest.read_bytes()
    )
    validation_cohort = CohortLedger.from_jsonl_bytes(
        arguments.validation_cohort_manifest.read_bytes()
    )
    raw_examples = _load_exact_raw_examples(
        source_path=arguments.calibration_reference_source,
        cohort=calibration_cohort,
    )
    policy_set = load_attested_sampling_policy_set(aggregate_path)
    backend_binding = load_calibration_backend_attestation_binding(aggregate_path)
    reference_loader = __import__(
        "scripts.research.calibrate_sampling_policy",
        fromlist=["_load_official_references"],
    )
    references = reference_loader._load_official_references(
        source_path=arguments.calibration_reference_source,
        cohort=calibration_cohort,
    )
    with _temporary_cwd(infer_config_path.parents[3]):
        resolved = load_infer_config(infer_config_path)
    runtime = assemble_runtime(resolved.config)
    qwen = runtime.qwen
    model_identity = dict(runtime.model_identity)
    tokenizer_identity = _tokenizer_identity(qwen)
    generation_fingerprint = sha256_json(
        resolved.config.generation.model_dump(mode="json")
    )
    backend = HFGenerateBackend(
        model=qwen.model,
        tokenizer=qwen.tokenizer,
        model_identity=model_identity,
        tokenizer_identity=tokenizer_identity,
        generation_config_fingerprint=generation_fingerprint,
    )
    processor = qwen.processor
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        raise RuntimeError("Qwen runtime processor lacks an image processor")

    prepared: dict[
        str, tuple[CalibrationRequest, Mapping[str, Any], MaterializedVisualInput]
    ] = {}
    prepared_images: dict[int, tuple[Mapping[str, Any], MaterializedVisualInput]] = {}
    for record in calibration_cohort.records:
        raw = raw_examples[record.image_id]
        prompt = build_prompt_record(
            raw,
            _template_config(resolved.config),
            processor=processor,
            row_index=record.frozen_order,
        )
        materialization = MaterializedVisualInput.from_full_image_path(
            source_image_path=record.image_path,
            expected_source_image_sha256=record.image_sha256,
            image_processor=image_processor,
            processor_contract_sha256=FROZEN_SPATIAL_GRID_SPEC_SHA256,
        )
        prepared_images[record.image_id] = (
            prompt.to_artifact_dict(),
            materialization,
        )
    requests_by_temperature: dict[float, list[tuple[CalibrationRequest, ...]]] = {}
    capabilities: dict[float, Any] = {}
    for temperature in CALIBRATION_TEMPERATURES:
        policy = DecodeGenerationPolicy.sampled(
            max_new_tokens=512,
            repetition_penalty=1.0,
            temperature=temperature,
            top_p=0.95,
        )
        if policy.fingerprint != policy_set.policy_fingerprint(temperature):
            raise RuntimeError(
                "attested sampling policy differs from the frozen policy"
            )
        capabilities[temperature] = (
            load_and_rebind_sampled_runtime_attestation_aggregate(
                aggregate_path,
                decode_generation_policy_fingerprint=policy.fingerprint,
                backend=backend,
            )
        )
        requests = build_calibration_requests(
            cohort=calibration_cohort,
            temperature=temperature,
            decode_generation_policy_fingerprint=policy.fingerprint,
        )
        grouped: list[tuple[CalibrationRequest, ...]] = []
        for image_index in range(12):
            batch = tuple(requests[image_index * 4 : image_index * 4 + 4])
            grouped.append(batch)
            record = calibration_cohort.records[image_index]
            prompt_payload, materialization = prepared_images[record.image_id]
            for request in batch:
                prepared[request.request_id] = (
                    request,
                    prompt_payload,
                    materialization,
                )
        requests_by_temperature[temperature] = grouped

    def execute_batch(
        temperature: float,
        panel_kind: str,
        batch_index: int,
        requests: Sequence[CalibrationRequest],
    ) -> tuple[CalibrationTerminalBundle, ...]:
        policy = DecodeGenerationPolicy.sampled(
            max_new_tokens=512,
            repetition_penalty=1.0,
            temperature=temperature,
            top_p=0.95,
        )
        decode_requests = []
        for request in requests:
            _, prompt_payload, materialization = prepared[request.request_id]
            decode_requests.append(
                DecodeRequest(
                    request_id=request.request_id,
                    prompt_token_ids=list(prompt_payload["prompt_token_ids"]),
                    model_inputs={
                        "pixel_values": materialization.pixel_values,
                        "image_grid_thw": materialization.image_grid_thw,
                    },
                    generation_policy=policy,
                    sampling_seed=request.sampling_seed,
                )
            )
        results = backend.generate_batch_with_verified_runtime_attestation(
            decode_requests,
            model_identity=model_identity,
            tokenizer_identity=tokenizer_identity,
            generation_config_fingerprint=generation_fingerprint,
            verified_runtime_attestation=capabilities[temperature],
        )
        if len(results) != 4:
            raise RuntimeError("HF backend returned a non-four-row result set")
        bundles: list[CalibrationTerminalBundle] = []
        for execution_index, (request, result) in enumerate(
            zip(requests, results, strict=True)
        ):
            _, prompt_payload, materialization = prepared[request.request_id]
            bundles.append(
                CalibrationTerminalBundle(
                    panel_kind=panel_kind,  # type: ignore[arg-type]
                    request=request,
                    physical_batch_index=batch_index,
                    request_execution_index=execution_index,
                    prompt_record=prompt_payload,
                    visual_input_materialization_receipt=materialization.receipt.to_artifact_dict(),
                    model_input_sha256=sha256_payload(
                        {
                            "prompt_token_ids": prompt_payload["prompt_token_ids"],
                            "visual_input_materialization_receipt": materialization.receipt.to_artifact_dict(),
                        }
                    ),
                    decode_result=result,
                    backend_attestation_aggregate_sha256=backend_binding.aggregate_artifact_sha256,
                    backend_attestation_aggregate_fingerprint=backend_binding.aggregate_payload_fingerprint,
                )
            )
        return tuple(bundles)

    def initial_gate(
        temperature: float, bundles: Sequence[CalibrationTerminalBundle]
    ) -> bool:
        observations = [
            reconstruct_calibration_observation(
                terminal_bundle=bundle,
                backend_attestation=backend_binding,
                source_width=calibration_cohort.records[
                    bundle.request.image_frozen_order
                ].source_width,
                source_height=calibration_cohort.records[
                    bundle.request.image_frozen_order
                ].source_height,
                official_reference_objects=references[bundle.request.image_id],
            )
            for bundle in bundles
        ]
        return summarize_calibration_panel(observations).gate_passed

    terminal = execute_frozen_protocol(
        requests_by_temperature=requests_by_temperature,
        execute_batch=execute_batch,
        initial_gate=initial_gate,
    )
    # Fail before writing if replay/order confirmation, validation-cohort
    # separation, or any terminal-bundle identity cannot produce the canonical
    # terminal-only selection receipt.
    selection = select_sampling_calibration_from_terminal_bundles(
        calibration_cohort=calibration_cohort,
        validation_cohort=validation_cohort,
        attested_policy_set=policy_set,
        backend_attestation=backend_binding,
        terminal_bundles=terminal,
        official_reference_objects_by_image_id=references,
    )
    payload = _build_collection_payload(
        terminal_bundles=terminal,
        calibration_cohort=calibration_cohort,
        validation_cohort=validation_cohort,
        aggregate_sha256=policy_set.aggregate_artifact_sha256,
        aggregate_fingerprint=policy_set.aggregate_payload_fingerprint,
        reference_source_sha256=sha256_file(arguments.calibration_reference_source),
    )
    write_immutable_json(arguments.output, payload)
    print(
        json.dumps(
            {
                "collection_sha256": sha256_payload(payload),
                "output": str(arguments.output.expanduser().resolve()),
                "selected_temperature": selection.selected_temperature,
                "status": "verified_and_written",
                "terminal_bundle_count": len(terminal),
            },
            sort_keys=True,
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--infer-config", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--calibration-cohort-manifest", type=Path, required=True)
    parser.add_argument("--validation-cohort-manifest", type=Path, required=True)
    parser.add_argument(
        "--sampled-runtime-attestation-aggregate", type=Path, required=True
    )
    parser.add_argument("--calibration-reference-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return _main(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
