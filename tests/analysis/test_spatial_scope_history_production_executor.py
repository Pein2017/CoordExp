from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from PIL import Image
import pytest
import torch

from src.analysis.spatial_scope_history.cohort_ledger import (
    AttemptLedger,
    AttemptRecord,
    CohortImageRecord,
    CohortLedger,
    ExecutionIdentityBundle,
    canonical_json_text,
    sha256_file,
    sha256_payload,
)
from src.analysis.spatial_scope_history.production_executor import (
    ACCEPTED_ROW_PREFIX_STATE_SCHEMA_VERSION,
    ProductionBatchExecutor,
    ProductionExecutorContractError,
    ProductionExecutorFactoryConfig,
    ProductionRuntimeBinding,
    _read_accepted_row_state,
)
from src.analysis.spatial_scope_history.runner import (
    BatchDispatch,
    BatchExecutionContext,
    RequestArtifactJournal,
    WorkerDeviceAssignment,
)
from src.analysis.spatial_scope_history.schedule import (
    DecodeProvenance,
    GridProvenance,
    ResearchSchedule,
)
from src.analysis.spatial_scope_history.spatial import SpatialGridSpec
from src.data import ImageRef, RawExample, RawObject, SourceProvenance
from src.inference.backend import (
    DecodeGenerationPolicy,
    DecodeRequest,
    DecodeResult,
    TokenTrace,
    batch_request_order_fingerprint,
    build_decode_execution_receipt,
    effective_generation_arguments,
)
from src.inference.prompt import PromptRecord


def _digest(label: str) -> str:
    return sha256_payload({"label": label})


class _ProcessorIdentity:
    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "merge_size": 2,
            "patch_size": 16,
            "processor_class": "SyntheticNoResizeProcessor",
            "temporal_patch_size": 2,
        }


class _NoResizeImageProcessor:
    def __init__(self) -> None:
        self.executed_sizes: list[tuple[int, int]] = []

    def __call__(self, *, images, return_tensors, do_resize):
        assert return_tensors == "pt"
        assert do_resize is False
        assert len(images) == 1
        width, height = images[0].size
        self.executed_sizes.append((width, height))
        raw_patch_rows = (width // 16) * (height // 16)
        return {
            "image_grid_thw": torch.tensor([[1, height // 16, width // 16]]),
            "pixel_values": torch.zeros((raw_patch_rows, 1536)),
        }


class _SyntheticProcessor:
    def __init__(self) -> None:
        self.image_processor = _NoResizeImageProcessor()


def _prompt_record(
    example: RawExample,
    *,
    row_index: int,
    continuation_text: str = "",
) -> PromptRecord:
    ordinary_chat_text = (
        "<|im_start|>user\n<|image_pad|>detect<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    chat_text = ordinary_chat_text + continuation_text
    prompt_ids = [11, 12, 13] + ([14] if continuation_text else [])
    continuation_sha256 = (
        hashlib.sha256(continuation_text.encode("utf-8")).hexdigest()
        if continuation_text
        else None
    )
    continuation_fields: dict[str, Any] = {}
    if continuation_text:
        start = len(ordinary_chat_text)
        continuation_fields = {
            "continuation_byte_span": (start, len(chat_text)),
            "continuation_character_span": (start, len(chat_text)),
            "continuation_token_impact_span": (3, 4),
            "image_placeholder_count": 1,
            "open_assistant_content_start_byte": start,
            "open_assistant_content_start_character": start,
            "open_assistant_interval_verified": True,
        }
    fingerprint = hashlib.sha256(
        json.dumps(
            {"full_chat_text": chat_text, "prompt_token_ids": prompt_ids},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return PromptRecord(
        row_id=example.example_id,
        row_index=row_index,
        example_id=example.example_id,
        messages=(),
        prompt_text="detect every Common Objects in Context class",
        chat_text=chat_text,
        prompt_token_ids=prompt_ids,
        template_id="synthetic-object-box-closed",
        template_fingerprint=_digest("template"),
        object_ordering="geometry_sorted",
        object_field_order="desc_first",
        assistant_format="object_box_closed",
        realized_object_order=[],
        full_prompt_fingerprint=fingerprint,
        continuation_text_sha256=continuation_sha256,
        **continuation_fields,
    )


def _primary_fixture(tmp_path: Path):
    records: list[CohortImageRecord] = []
    examples: dict[int, RawExample] = {}
    for frozen_order in range(4):
        image_id = 70_000 + frozen_order
        image_path = tmp_path / f"image-{image_id}.png"
        image = Image.new(
            "RGB",
            (128, 128),
            color=(frozen_order + 1, frozen_order + 2, frozen_order + 3),
        )
        image.save(image_path)
        image.close()
        row_sha256 = _digest(f"source-row-{frozen_order}")
        image_sha256 = sha256_file(image_path)
        records.append(
            CohortImageRecord(
                image_id=image_id,
                frozen_order=frozen_order,
                source_row_index=frozen_order,
                image_path=str(image_path),
                image_sha256=image_sha256,
                source_width=128,
                source_height=128,
                raw_width=128,
                raw_height=128,
                source_row_sha256=row_sha256,
                source_dataset_sha256=_digest("source-dataset"),
                raw_annotation_sha256=_digest(f"annotation-{frozen_order}"),
                noncrowd_annotated_object_count=12,
                annotated_person_count=8,
                annotated_food_tableware_count=0,
                source_crowd_annotation_count=0,
                cohort_memberships=("synthetic-dense-four",),
                density_tags=("annotated-count-density",),
            )
        )
        examples[image_id] = RawExample(
            example_id=f"coco2017_val_{image_id:012d}",
            image=ImageRef(
                declared_path=image_path.name,
                path=image_path,
                width=128,
                height=128,
                stat={},
            ),
            objects=(
                RawObject(
                    object_id="object-0",
                    description="person",
                    bbox=(100, 100, 500, 500),
                    metadata={},
                ),
            ),
            metadata={"source": {"image_id": image_id}},
            source=SourceProvenance(
                source_path=tmp_path / "source.jsonl",
                row_number=frozen_order + 1,
                row_sha256=row_sha256,
                source_format="canonical_raw_example",
            ),
        )
    cohort = CohortLedger(
        cohort_id="synthetic-dense-four",
        full_name="Synthetic Dense Four-Image Cohort",
        operational_meaning="Exercise production materialization contracts.",
        records=tuple(records),
    )
    generation_policy = DecodeGenerationPolicy.sampled(
        max_new_tokens=512,
        repetition_penalty=1.0,
        temperature=0.4,
        top_p=0.95,
    )
    schedule = ResearchSchedule.build_primary(
        unit_id="production-executor-test",
        run_id="production-executor-test-run",
        cohort=cohort,
        root_seed=2026071301,
        decode=DecodeProvenance(
            temperature=0.4,
            canonical_generation_policy_sha256=generation_policy.fingerprint,
            sampled_runtime_attestation_sha256=_digest("attestation"),
        ),
        execution_identity=ExecutionIdentityBundle(
            code_sha256=_digest("code"),
            config_sha256=_digest("config"),
            ledger_sha256=_digest("ledger"),
            runtime_sha256=_digest("runtime"),
        ),
        grid=GridProvenance(
            canonical_spatial_spec_sha256=SpatialGridSpec().fingerprint,
            canonical_spatial_receipt_contract_sha256=_digest(
                "processor-contract"
            ),
        ),
    )
    processor = _SyntheticProcessor()
    runtime = SimpleNamespace(
        qwen={
            "processor": processor,
            "processor_identity": _ProcessorIdentity(),
        }
    )
    binding = ProductionRuntimeBinding(
        runtime=runtime,
        backend=SimpleNamespace(),
        model_identity={"family": "synthetic"},
        tokenizer_identity={"family": "synthetic"},
        generation_config_fingerprint=_digest("generation-config"),
        verified_runtime_attestation=SimpleNamespace(),
    )
    infer_config = SimpleNamespace(
        generation=SimpleNamespace(
            batch_size=4,
            max_new_tokens=512,
            repetition_penalty=1.0,
            temperature=0.4,
            top_p=0.95,
        ),
        template=SimpleNamespace(
            assistant_format="object_box_closed",
            object_field_order="desc_first",
            object_ordering="geo_sorted",
            prompt=SimpleNamespace(system=None, user="detect"),
        ),
    )
    executor = ProductionBatchExecutor(
        assignment=WorkerDeviceAssignment(worker_index=0, physical_gpu_token="0"),
        schedule=schedule,
        cohort=cohort,
        raw_examples_by_image_id=examples,
        infer_config=infer_config,
        runtime_binding=binding,
        attempt_ledger_path=tmp_path / "attempts.jsonl",
    )
    return schedule, cohort, examples, processor, executor


def _request(
    schedule: ResearchSchedule,
    arm_code: str,
    *,
    cell_index: int | None = None,
    image_order: int = 0,
):
    return next(
        request
        for request in schedule.requests
        if request.arm.arm_code == arm_code
        and request.cell_index == cell_index
        and request.image_frozen_order == image_order
    )


def _empty_attempt_ledger(schedule: ResearchSchedule) -> AttemptLedger:
    return AttemptLedger(
        run_id=schedule.identity.run_id,
        schedule_sha256=schedule.fingerprint,
        execution_identity=schedule.identity.execution_identity,
        records=(),
    )


def test_factory_config_rejects_unknown_or_missing_paths() -> None:
    payload = {
        "attempt_ledger_path": "/tmp/attempts.jsonl",
        "calibration_selection_receipt_path": "/tmp/calibration-selection.json",
        "cohort_ledger_path": "/tmp/cohort.jsonl",
        "infer_config_path": "/tmp/infer.yaml",
        "primary_schedule_artifact_path": "/tmp/schedule.json",
        "sampled_runtime_attestation_path": "/tmp/attestation.json",
        "schema_version": "spatial_scope_history.production_executor_factory_config.v2",
        "source_jsonl_path": "/tmp/source.jsonl",
        "source_runtime_identity_receipt_path": "/tmp/runtime-identity.json",
    }
    assert (
        ProductionExecutorFactoryConfig.from_mapping(
            payload
        ).primary_schedule_artifact_path
        == "/tmp/schedule.json"
    )

    with pytest.raises(ProductionExecutorContractError):
        ProductionExecutorFactoryConfig.from_mapping({**payload, "extra": "no"})
    with pytest.raises(ProductionExecutorContractError):
        ProductionExecutorFactoryConfig.from_mapping(
            {
                key: value
                for key, value in payload.items()
                if key != "primary_schedule_artifact_path"
            }
        )


def test_prepare_materializes_full_tile_and_mask_without_resize(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schedule, _, _, processor, executor = _primary_fixture(tmp_path)
    prompt_visual_inputs: list[tuple[int, int] | None] = []

    def prompt_record(
        example,
        template,
        *,
        processor,
        row_index,
        visual_input_image=None,
    ):
        del template, processor
        prompt_visual_inputs.append(
            None if visual_input_image is None else visual_input_image.size
        )
        return _prompt_record(example, row_index=row_index)

    monkeypatch.setattr(
        "src.analysis.spatial_scope_history.production_executor.build_prompt_record",
        prompt_record,
    )
    ledger = _empty_attempt_ledger(schedule)

    full = executor._prepare_request(
        _request(schedule, "FULL_SINGLE"), attempt_ledger=ledger
    )
    tile = executor._prepare_request(
        _request(schedule, "TILE_RESET", cell_index=0), attempt_ledger=ledger
    )
    masked = executor._prepare_request(
        _request(schedule, "MASK_RESET", cell_index=0), attempt_ledger=ledger
    )
    cumulative_first = executor._prepare_request(
        _request(schedule, "MASK_CUMULATIVE", cell_index=0),
        attempt_ledger=ledger,
    )

    assert full.visual_materialization.receipt.input_kind == "full_image"
    assert full.image_plan_row is not None
    assert full.image_plan_row.example_id == "coco2017_val_000000070000"
    assert tile.visual_materialization.spatial_image_encoding is not None
    assert (
        tile.visual_materialization.spatial_image_encoding.plan.variant_mode
        == "tile_reset"
    )
    assert masked.visual_materialization.receipt.input_width == 128
    assert masked.visual_materialization.receipt.input_height == 128
    assert (
        masked.visual_materialization.spatial_image_encoding.plan.variant_mode
        == "mask_reset"
    )
    assert cumulative_first.predecessor_attempt is None
    assert cumulative_first.predecessor_accepted_rows == ()
    assert (
        cumulative_first.visual_materialization.spatial_image_encoding.plan.variant_mode
        == "mask_cumulative"
    )
    assert len(processor.image_processor.executed_sizes) == 4
    assert processor.image_processor.executed_sizes[0] == (128, 128)
    assert all(width <= 128 and height <= 128 for width, height in processor.image_processor.executed_sizes)
    assert prompt_visual_inputs == [
        None,
        (
            tile.visual_materialization.receipt.input_width,
            tile.visual_materialization.receipt.input_height,
        ),
        (
            masked.visual_materialization.receipt.input_width,
            masked.visual_materialization.receipt.input_height,
        ),
        (
            cumulative_first.visual_materialization.receipt.input_width,
            cumulative_first.visual_materialization.receipt.input_height,
        ),
    ]
    assert all(
        request.decode_request.generation_policy
        == DecodeGenerationPolicy.sampled(
            max_new_tokens=512,
            repetition_penalty=1.0,
            temperature=0.4,
            top_p=0.95,
        )
        for request in (full, tile, masked, cumulative_first)
    )


def test_deterministic_infer_defaults_do_not_override_sealed_request_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schedule, cohort, examples, _, executor = _primary_fixture(tmp_path)
    deterministic_infer_config = SimpleNamespace(
        generation=SimpleNamespace(
            batch_size=4,
            max_new_tokens=3084,
            repetition_penalty=1.1,
            temperature=0.0,
            top_p=1.0,
        ),
        template=executor._infer_config.template,
    )
    executor = ProductionBatchExecutor(
        assignment=WorkerDeviceAssignment(worker_index=0, physical_gpu_token="0"),
        schedule=schedule,
        cohort=cohort,
        raw_examples_by_image_id=examples,
        infer_config=deterministic_infer_config,
        runtime_binding=executor._binding,
        attempt_ledger_path=tmp_path / "deterministic-default-attempts.jsonl",
    )
    monkeypatch.setattr(
        "src.analysis.spatial_scope_history.production_executor.build_prompt_record",
        lambda example, template, *, processor, row_index: _prompt_record(
            example, row_index=row_index
        ),
    )

    prepared = executor._prepare_request(
        _request(schedule, "FULL_SINGLE"),
        attempt_ledger=_empty_attempt_ledger(schedule),
    )

    assert prepared.decode_request.generation_policy == DecodeGenerationPolicy.sampled(
        max_new_tokens=512,
        repetition_penalty=1.0,
        temperature=schedule.identity.decode.temperature,
        top_p=0.95,
    )


def test_cumulative_prompt_reads_only_durable_predecessor_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schedule, _, _, _, executor = _primary_fixture(tmp_path)
    predecessor = _request(schedule, "MASK_CUMULATIVE", cell_index=0)
    successor = _request(schedule, "MASK_CUMULATIVE", cell_index=1)
    assert successor.predecessor_request_id == predecessor.request_id
    accepted_row = (
        "<|object_ref_start|>person<|object_ref_end|>"
        "<|box_start|><|coord_100|><|coord_100|>"
        "<|coord_400|><|coord_400|><|box_end|>"
    )
    state_path = tmp_path / "accepted-row-prefix-state.json"
    state_path.write_text(
        canonical_json_text(
            {
                "accepted_global_coordinate_rows": [accepted_row],
                "schema_version": ACCEPTED_ROW_PREFIX_STATE_SCHEMA_VERSION,
            }
        ),
        encoding="utf-8",
    )
    predecessor_dependency = next(
        dependency
        for dependency in schedule.attempt_dependencies
        if dependency.request_id == predecessor.request_id
    )
    predecessor_attempt = AttemptRecord(
        run_id=schedule.identity.run_id,
        schedule_sha256=schedule.fingerprint,
        request_id=predecessor.request_id,
        physical_batch_plan_sha256=(
            predecessor_dependency.physical_batch_plan_sha256
        ),
        physical_batch_sha256=predecessor_dependency.physical_batch_sha256,
        physical_batch_index=predecessor_dependency.physical_batch_index,
        attempt_status="completed",
        started_at_utc="2026-07-13T00:00:00Z",
        finished_at_utc="2026-07-13T00:00:01Z",
        execution_identity=schedule.identity.execution_identity,
        output_artifact_sha256=_digest("predecessor-output"),
        expected_cumulative_state_sha256=(
            predecessor.initial_cumulative_state_sha256
        ),
        produced_cumulative_state_sha256=sha256_file(state_path),
        produced_cumulative_state_artifact_path=str(state_path),
    )
    ledger = replace(
        _empty_attempt_ledger(schedule),
        records=(predecessor_attempt,),
    )
    observed_continuations: list[str] = []

    def cumulative_prompt(
        example,
        template,
        *,
        processor,
        row_index,
        accepted_global_coordinate_rows,
        visual_input_image,
    ):
        del template, processor
        assert visual_input_image.size == (128, 128)
        observed_continuations.append(accepted_global_coordinate_rows)
        return _prompt_record(
            example,
            row_index=row_index,
            continuation_text=accepted_global_coordinate_rows,
        )

    monkeypatch.setattr(
        "src.analysis.spatial_scope_history.production_executor.build_cumulative_prompt_record",
        cumulative_prompt,
    )
    prepared = executor._prepare_request(successor, attempt_ledger=ledger)

    assert prepared.predecessor_attempt == predecessor_attempt
    assert prepared.predecessor_accepted_rows == (accepted_row,)
    assert observed_continuations == [accepted_row]
    assert prepared.prompt_record.continuation_text_sha256 == hashlib.sha256(
        accepted_row.encode("utf-8")
    ).hexdigest()

    state_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ProductionExecutorContractError) as exc_info:
        executor._prepare_request(successor, attempt_ledger=ledger)
    assert exc_info.value.code == "production_executor.predecessor_state_digest"


def test_accepted_row_state_schema_fails_closed(tmp_path: Path) -> None:
    valid = tmp_path / "valid.json"
    valid.write_text(
        canonical_json_text(
            {
                "accepted_global_coordinate_rows": ["row-a", "row-b"],
                "schema_version": ACCEPTED_ROW_PREFIX_STATE_SCHEMA_VERSION,
            }
        ),
        encoding="utf-8",
    )
    assert _read_accepted_row_state(valid) == ("row-a", "row-b")

    invalid = tmp_path / "invalid.json"
    invalid.write_text(
        canonical_json_text(
            {
                "accepted_global_coordinate_rows": ["row-a"],
                "schema_version": ACCEPTED_ROW_PREFIX_STATE_SCHEMA_VERSION,
                "unexpected": True,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ProductionExecutorContractError):
        _read_accepted_row_state(invalid)


class _RecordingBackend:
    def __init__(self, results: list[Any]) -> None:
        self.results = results
        self.calls: list[tuple[DecodeRequest, ...]] = []

    def generate_batch_with_verified_runtime_attestation(self, requests, **kwargs):
        del kwargs
        self.calls.append(tuple(requests))
        return list(self.results)


class _ReceiptBuildingBackend:
    def __init__(
        self,
        *,
        model_identity: dict[str, Any],
        tokenizer_identity: dict[str, Any],
        generation_config_fingerprint: str,
        coordinate_bins_by_request_id: dict[str, tuple[int, int, int, int]]
        | None = None,
    ) -> None:
        self.model_identity = model_identity
        self.tokenizer_identity = tokenizer_identity
        self.generation_config_fingerprint = generation_config_fingerprint
        self.coordinate_bins_by_request_id = coordinate_bins_by_request_id or {}
        self.calls: list[tuple[DecodeRequest, ...]] = []

    def generate_batch_with_verified_runtime_attestation(self, requests, **kwargs):
        assert kwargs["model_identity"] == self.model_identity
        assert kwargs["tokenizer_identity"] == self.tokenizer_identity
        assert (
            kwargs["generation_config_fingerprint"]
            == self.generation_config_fingerprint
        )
        ordered = tuple(requests)
        self.calls.append(ordered)
        order_fingerprint = batch_request_order_fingerprint(ordered)
        results: list[DecodeResult] = []
        for execution_index, request in enumerate(ordered):
            coordinate_bins = self.coordinate_bins_by_request_id.get(
                request.request_id,
                (100, 100, 500, 500),
            )
            pieces = (
                "<|object_ref_start|>",
                "person",
                "<|object_ref_end|>",
                "<|box_start|>",
                *(f"<|coord_{value}|>" for value in coordinate_bins),
                "<|box_end|>",
            )
            raw_row = "".join(pieces)
            generated_ids = [1000 + index for index in range(len(pieces))]
            trace = [
                TokenTrace(
                    step_index=index,
                    token_id=generated_ids[index],
                    token_text=piece,
                    logprob=-0.25,
                    is_stop=False,
                    is_pad=False,
                    backend="hf",
                    backend_mode="generate",
                    response_family="hf",
                )
                for index, piece in enumerate(pieces)
            ]
            generator = torch.Generator(device="cpu").manual_seed(
                request.sampling_seed
            )
            receipt = build_decode_execution_receipt(
                request=request,
                generated_token_ids=generated_ids,
                token_trace=trace,
                stop_reason="im_end",
                model_identity=self.model_identity,
                tokenizer_identity=self.tokenizer_identity,
                generation_config_fingerprint=(
                    self.generation_config_fingerprint
                ),
                executed_generation_arguments=effective_generation_arguments(
                    request.generation_policy,
                    eos_token_id=151645,
                    pad_token_id=0,
                ),
                request_execution_index=execution_index,
                batch_request_order_fingerprint=order_fingerprint,
                request_generator=generator,
                runtime_identity={"runtime": "synthetic"},
                custom_sampler_executed=True,
            )
            results.append(
                DecodeResult(
                    request_id=request.request_id,
                    backend="hf",
                    backend_mode="generate",
                    response_family="hf",
                    prompt_token_ids=list(request.prompt_token_ids),
                    generated_token_ids=generated_ids,
                    raw_generated_text=raw_row,
                    parser_text=raw_row,
                    strip_policy="none",
                    stop_reason="im_end",
                    model_identity=self.model_identity,
                    tokenizer_identity=self.tokenizer_identity,
                    generation_config_fingerprint=(
                        self.generation_config_fingerprint
                    ),
                    token_trace=trace,
                    execution_receipt=receipt,
                )
            )
        return results


def test_batch_call_is_indivisible_and_persists_only_after_all_derivations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schedule, _, _, _, executor = _primary_fixture(tmp_path)
    batch = schedule.batches()[0]
    assert batch.cardinality == 4
    backend_results = [
        SimpleNamespace(request_id=request.request_id)
        for request in batch.requests
    ]
    backend = _RecordingBackend(backend_results)
    executor._binding = replace(executor._binding, backend=backend)
    decode_requests = {
        request.request_id: DecodeRequest(
            request_id=request.request_id,
            prompt_token_ids=[request.schedule_index + 1],
            model_inputs={},
            generation_policy=executor._generation_policy,
            sampling_seed=request.sampling_seed,
        )
        for request in batch.requests
    }
    monkeypatch.setattr(executor, "_read_attempt_ledger", lambda: object())
    monkeypatch.setattr(
        executor,
        "_prepare_request",
        lambda request, *, attempt_ledger: SimpleNamespace(
            scheduled_request=request,
            decode_request=decode_requests[request.request_id],
        ),
    )
    monkeypatch.setattr(
        "src.analysis.spatial_scope_history.production_executor._validate_result_batch",
        lambda *, decode_requests, decode_results: {
            result.request_id: result for result in decode_results
        },
    )
    derivation_order: list[str] = []
    persistence_order: list[str] = []

    def derive(prepared, **kwargs):
        del kwargs
        derivation_order.append(prepared.scheduled_request.request_id)
        return SimpleNamespace()

    monkeypatch.setattr(executor, "_derive_completed_artifacts", derive)
    monkeypatch.setattr(
        executor,
        "_persist_completed_request",
        lambda *, prepared, **kwargs: persistence_order.append(
            prepared.scheduled_request.request_id
        ),
    )
    dispatch = BatchDispatch(
        wave_index=0,
        execution_wave_partition=batch.execution_wave_partition,
        batch=batch,
        worker=WorkerDeviceAssignment(worker_index=0, physical_gpu_token="0"),
        run_id=schedule.identity.run_id,
        schedule_sha256=schedule.fingerprint,
    )
    context = BatchExecutionContext(dispatch=dispatch, journals={})
    executor(context)

    assert len(backend.calls) == 1
    assert tuple(request.request_id for request in backend.calls[0]) == batch.request_ids
    assert derivation_order == list(batch.request_ids)
    assert persistence_order == list(batch.request_ids)

    persistence_order.clear()
    derivation_order.clear()

    def fail_second_derivation(prepared, **kwargs):
        del kwargs
        derivation_order.append(prepared.scheduled_request.request_id)
        if len(derivation_order) == 2:
            raise ProductionExecutorContractError(
                "injected peer derivation failure",
                code="production_executor.test_peer_derivation",
            )
        return SimpleNamespace()

    monkeypatch.setattr(
        executor, "_derive_completed_artifacts", fail_second_derivation
    )
    with pytest.raises(ProductionExecutorContractError):
        executor(context)
    assert persistence_order == []


def test_completed_full_image_batch_persists_terminal_evidence_and_attempts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schedule, _, _, _, executor = _primary_fixture(tmp_path)
    monkeypatch.setattr(
        "src.analysis.spatial_scope_history.production_executor.build_prompt_record",
        lambda example, template, *, processor, row_index, visual_input_image=None: _prompt_record(
            example, row_index=row_index
        ),
    )
    backend = _ReceiptBuildingBackend(
        model_identity=dict(executor._binding.model_identity),
        tokenizer_identity=dict(executor._binding.tokenizer_identity),
        generation_config_fingerprint=(
            executor._binding.generation_config_fingerprint
        ),
    )
    executor._binding = replace(executor._binding, backend=backend)
    batch = schedule.batches()[0]
    dispatch = BatchDispatch(
        wave_index=0,
        execution_wave_partition=batch.execution_wave_partition,
        batch=batch,
        worker=WorkerDeviceAssignment(worker_index=0, physical_gpu_token="0"),
        run_id=schedule.identity.run_id,
        schedule_sha256=schedule.fingerprint,
    )
    artifact_root = tmp_path / "artifacts"
    journals = {}
    for request in batch.requests:
        RequestArtifactJournal(
            root=artifact_root,
            dispatch=dispatch,
            request_id=request.request_id,
        ).write_call_intent()
        journals[request.request_id] = RequestArtifactJournal.open_after_call_intent(
            root=artifact_root,
            dispatch=dispatch,
            request_id=request.request_id,
        )
    executor(BatchExecutionContext(dispatch=dispatch, journals=journals))

    assert len(backend.calls) == 1
    assert tuple(request.request_id for request in backend.calls[0]) == batch.request_ids
    attempts = executor._read_attempt_ledger()
    assert tuple(record.request_id for record in attempts.records) == batch.request_ids
    assert all(record.attempt_status == "completed" for record in attempts.records)
    for request in batch.requests:
        journal = journals[request.request_id]
        assert journal.terminal_written is True
        assert journal.terminal_status == "completed"
        assert (journal.request_directory / "terminal-output-bundle.json").is_file()
        assert (journal.request_directory / "04-parse_score.json").is_file()
        assert not (
            journal.request_directory / "05-cumulative_state.json"
        ).exists()


def test_cumulative_successor_reloads_predecessor_state_across_executor_instances(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schedule, cohort, examples, _, first_executor = _primary_fixture(tmp_path)
    monkeypatch.setattr(
        "src.analysis.spatial_scope_history.production_executor.build_prompt_record",
        lambda example, template, *, processor, row_index, visual_input_image=None: _prompt_record(
            example, row_index=row_index
        ),
    )
    monkeypatch.setattr(
        "src.analysis.spatial_scope_history.production_executor.build_cumulative_prompt_record",
        lambda example, template, *, processor, row_index, accepted_global_coordinate_rows, visual_input_image: _prompt_record(
            example,
            row_index=row_index,
            continuation_text=accepted_global_coordinate_rows,
        ),
    )

    cell_zero_batch = next(
        batch
        for batch in schedule.batches()
        if batch.execution_wave_partition == "cumulative-cell-00"
    )
    cell_one_batch = next(
        batch
        for batch in schedule.batches()
        if batch.execution_wave_partition == "cumulative-cell-01"
    )
    cell_zero_coordinates = {
        request.request_id: (50, 50, 200, 200)
        for request in cell_zero_batch.requests
    }
    first_backend = _ReceiptBuildingBackend(
        model_identity=dict(first_executor._binding.model_identity),
        tokenizer_identity=dict(first_executor._binding.tokenizer_identity),
        generation_config_fingerprint=(
            first_executor._binding.generation_config_fingerprint
        ),
        coordinate_bins_by_request_id=cell_zero_coordinates,
    )
    first_executor._binding = replace(
        first_executor._binding,
        backend=first_backend,
    )
    artifact_root = tmp_path / "cumulative-artifacts"

    def execution_context_for(batch, *, wave_index):
        dispatch = BatchDispatch(
            wave_index=wave_index,
            execution_wave_partition=batch.execution_wave_partition,
            batch=batch,
            worker=WorkerDeviceAssignment(
                worker_index=wave_index,
                physical_gpu_token=str(wave_index),
            ),
            run_id=schedule.identity.run_id,
            schedule_sha256=schedule.fingerprint,
        )
        journals = {}
        for request in batch.requests:
            RequestArtifactJournal(
                root=artifact_root,
                dispatch=dispatch,
                request_id=request.request_id,
            ).write_call_intent()
            journals[request.request_id] = (
                RequestArtifactJournal.open_after_call_intent(
                    root=artifact_root,
                    dispatch=dispatch,
                    request_id=request.request_id,
                )
            )
        return BatchExecutionContext(dispatch=dispatch, journals=journals)

    first_executor(execution_context_for(cell_zero_batch, wave_index=0))
    first_attempts = first_executor._read_attempt_ledger()
    assert all(
        first_attempts.records_by_request_id[request.request_id].produced_cumulative_state_sha256
        is not None
        for request in cell_zero_batch.requests
    )

    second_processor = _SyntheticProcessor()
    second_binding = ProductionRuntimeBinding(
        runtime=SimpleNamespace(
            qwen={
                "processor": second_processor,
                "processor_identity": _ProcessorIdentity(),
            }
        ),
        backend=SimpleNamespace(),
        model_identity=dict(first_executor._binding.model_identity),
        tokenizer_identity=dict(first_executor._binding.tokenizer_identity),
        generation_config_fingerprint=(
            first_executor._binding.generation_config_fingerprint
        ),
        verified_runtime_attestation=SimpleNamespace(),
    )
    second_executor = ProductionBatchExecutor(
        assignment=WorkerDeviceAssignment(worker_index=1, physical_gpu_token="1"),
        schedule=schedule,
        cohort=cohort,
        raw_examples_by_image_id=examples,
        infer_config=first_executor._infer_config,
        runtime_binding=second_binding,
        attempt_ledger_path=first_executor._attempt_ledger_path,
    )
    cell_one_coordinates = {
        request.request_id: (300, 50, 450, 200)
        for request in cell_one_batch.requests
    }
    second_backend = _ReceiptBuildingBackend(
        model_identity=dict(second_executor._binding.model_identity),
        tokenizer_identity=dict(second_executor._binding.tokenizer_identity),
        generation_config_fingerprint=(
            second_executor._binding.generation_config_fingerprint
        ),
        coordinate_bins_by_request_id=cell_one_coordinates,
    )
    second_executor._binding = replace(
        second_executor._binding,
        backend=second_backend,
    )
    second_executor(execution_context_for(cell_one_batch, wave_index=1))

    final_attempts = second_executor._read_attempt_ledger()
    for request in cell_one_batch.requests:
        record = final_attempts.records_by_request_id[request.request_id]
        assert record.produced_cumulative_state_artifact_path is not None
        rows = _read_accepted_row_state(
            Path(record.produced_cumulative_state_artifact_path)
        )
        assert len(rows) == 2
        assert "<|coord_50|>" in rows[0]
        assert "<|coord_300|>" in rows[1]
