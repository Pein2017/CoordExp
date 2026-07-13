"""Executed-evidence fixtures shared by spatial-scope analysis tests."""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Sequence

from PIL import Image
import torch

from src.analysis.spatial_scope_history.cohort_ledger import (
    AttemptDependencyContract,
    AttemptRecord,
    ExecutionIdentityBundle,
    sha256_file,
    sha256_payload,
)
from src.analysis.spatial_scope_history.execution_evidence import (
    ExecutionEvidenceEnvelope,
)
from src.analysis.spatial_scope_history.schedule import (
    DecodeProvenance,
    EMPTY_ACCEPTED_ROW_PREFIX_STATE_SHA256,
    GridProvenance,
    PhysicalBatchPlan,
    RequestBatch,
    ScheduledRequest,
    primary_arm_definition,
)
from src.analysis.spatial_scope_history.spatial import (
    MaterializedVisualInput,
    SpatialGrid,
    SpatialGridSpec,
)
from src.inference.backend import (
    DecodeGenerationPolicy,
    DecodeRequest,
    DecodeResult,
    TokenTrace,
    batch_request_order_fingerprint,
    build_decode_execution_receipt,
    effective_generation_arguments,
)
from src.inference.image_plan import ImagePlanRow
from src.inference.prompt import PromptRecord


_ACTIVE_ROOT: ContextVar[Path] = ContextVar("spatial_scope_test_root")


def install_execution_evidence_test_root(path: Path) -> Token[Path]:
    return _ACTIVE_ROOT.set(path)


def reset_execution_evidence_test_root(token: Token[Path]) -> None:
    _ACTIVE_ROOT.reset(token)


def build_test_execution_evidence(
    *,
    arm_code: str = "FULL_SINGLE",
    image_id: int = 1,
    cell_index: int | None = None,
    sampling_seed: int = 137,
    source_width: int = 128,
    source_height: int = 128,
    raw_generated_text: str | None = None,
    parser_text: str | None = None,
    generated_token_ids: Sequence[int] = (101,),
    token_trace: Sequence[TokenTrace] | None = None,
    cumulative_accepted_rows: Sequence[str] | None = None,
) -> ExecutionEvidenceEnvelope:
    evidence, _ = build_test_execution_evidence_with_result(
        arm_code=arm_code,
        image_id=image_id,
        cell_index=cell_index,
        sampling_seed=sampling_seed,
        source_width=source_width,
        source_height=source_height,
        raw_generated_text=raw_generated_text,
        parser_text=parser_text,
        generated_token_ids=generated_token_ids,
        token_trace=token_trace,
        cumulative_accepted_rows=cumulative_accepted_rows,
    )
    return evidence


def build_test_execution_evidence_with_result(
    *,
    arm_code: str = "FULL_SINGLE",
    image_id: int = 1,
    cell_index: int | None = None,
    sampling_seed: int = 137,
    source_width: int = 128,
    source_height: int = 128,
    raw_generated_text: str | None = None,
    parser_text: str | None = None,
    generated_token_ids: Sequence[int] = (101,),
    token_trace: Sequence[TokenTrace] | None = None,
    cumulative_accepted_rows: Sequence[str] | None = None,
) -> tuple[ExecutionEvidenceEnvelope, DecodeResult]:
    """Build one envelope from a canonical sealed batch and return its exact result."""

    root = _ACTIVE_ROOT.get()
    arm = primary_arm_definition(arm_code)
    if arm_code == "FULL_SINGLE":
        resolved_cell_index = None
        call_label = "single"
        seed_role = "baseline"
    else:
        resolved_cell_index = 0 if cell_index is None else cell_index
        call_label = f"cell-{resolved_cell_index:02d}"
        seed_role = "paired-cell"

    image_path = root / f"source-{image_id}-{source_width}x{source_height}.png"
    if not image_path.exists():
        image = Image.new(
            "RGB",
            (source_width, source_height),
            color=(image_id % 251, (image_id * 3) % 251, (image_id * 7) % 251),
        )
        image.save(image_path)
        image.close()
    image_sha256 = sha256_file(image_path)

    policy = DecodeGenerationPolicy.sampled(
        max_new_tokens=64,
        repetition_penalty=1.0,
        temperature=0.4,
        top_p=0.95,
    )
    decode = DecodeProvenance(
        temperature=0.4,
        canonical_generation_policy_sha256=policy.fingerprint,
        sampled_runtime_attestation_sha256=_digest("runtime-attestation"),
    )
    grid_provenance = GridProvenance(
        canonical_spatial_spec_sha256=SpatialGridSpec().fingerprint,
        canonical_spatial_receipt_contract_sha256=_digest("spatial-receipt-contract"),
    )
    predecessor_request_id = None
    initial_state_sha256 = None
    if arm.cumulative_dependency:
        if resolved_cell_index == 0:
            initial_state_sha256 = EMPTY_ACCEPTED_ROW_PREFIX_STATE_SHA256
        else:
            predecessor_request_id = "spatial-scope-history-request:" + "f" * 64
    schedule_identity_sha256 = _digest("schedule")
    execution_identity_sha256 = _digest("execution")

    def make_request(*, seed: int, schedule_index: int) -> ScheduledRequest:
        candidate = ScheduledRequest(
            request_id="spatial-scope-history-request:" + "0" * 64,
            schedule_index=schedule_index,
            image_id=image_id,
            image_frozen_order=0,
            image_sha256=image_sha256,
            arm=arm,
            call_label=call_label,
            cell_index=resolved_cell_index,
            seed_role=seed_role,
            sampling_seed=seed,
            schedule_identity_sha256=schedule_identity_sha256,
            grid_sha256=grid_provenance.fingerprint,
            decode_sha256=decode.fingerprint,
            execution_identity_sha256=execution_identity_sha256,
            predecessor_request_id=predecessor_request_id,
            initial_cumulative_state_sha256=initial_state_sha256,
        )
        return replace(
            candidate,
            request_id=(
                "spatial-scope-history-request:"
                + sha256_payload(candidate.identity_payload())
            ),
        )

    requests = tuple(
        make_request(seed=sampling_seed + offset, schedule_index=offset)
        for offset in range(4)
    )
    request = requests[0]

    processor_digest = _digest("test-processor-contract")

    def test_image_processor(*, images, return_tensors, do_resize):
        assert return_tensors == "pt"
        assert do_resize is False
        assert len(images) == 1
        width, height = images[0].size
        return {
            "pixel_values": torch.zeros(((height // 16) * (width // 16), 1536)),
            "image_grid_thw": torch.tensor([[1, height // 16, width // 16]]),
        }

    if arm.input_policy == "complete_source_canvas":
        visual_materialization = MaterializedVisualInput.from_full_image_path(
            source_image_path=image_path,
            expected_source_image_sha256=image_sha256,
            image_processor=test_image_processor,
            processor_contract_sha256=processor_digest,
        )
        encoding = None
    else:
        mode = {
            "TILE_RESET": "tile_reset",
            "MASK_RESET": "mask_reset",
            "MASK_CUMULATIVE": "mask_cumulative",
        }[arm_code]
        assert resolved_cell_index is not None
        plan = SpatialGrid.build(
            source_width=source_width,
            source_height=source_height,
        ).plan(
            cell_index=resolved_cell_index,
            variant_mode=mode,  # type: ignore[arg-type]
        )
        visual_materialization = MaterializedVisualInput.from_spatial_plan_path(
            plan=plan,
            source_image_path=image_path,
            expected_source_image_sha256=image_sha256,
            image_processor=test_image_processor,
            processor_contract_sha256=processor_digest,
        )
        encoding = visual_materialization.spatial_image_encoding
        assert encoding is not None

    current_pixel_values = visual_materialization.pixel_values
    current_grid_thw = visual_materialization.image_grid_thw

    decode_requests = tuple(
        DecodeRequest(
            request_id=batch_request.request_id,
            prompt_token_ids=[11, 12 + offset],
            model_inputs={
                "pixel_values": current_pixel_values.clone(),
                "image_grid_thw": current_grid_thw.clone(),
            },
            generation_policy=policy,
            sampling_seed=batch_request.sampling_seed,
        )
        for offset, batch_request in enumerate(requests)
    )
    decode_request = decode_requests[0]
    physical_batch_plan = PhysicalBatchPlan.build(
        schedule_identity_sha256=schedule_identity_sha256,
        request_ids=[item.request_id for item in requests],
    )
    physical_batch = physical_batch_plan.batches[0]
    request_batch = RequestBatch(
        batch_index=0,
        requests=requests,
        physical_batch_sha256=physical_batch.fingerprint,
        physical_batch_plan_sha256=physical_batch_plan.fingerprint,
    )
    attempt_dependency = AttemptDependencyContract(
        request_id=request.request_id,
        physical_batch_plan_sha256=physical_batch_plan.fingerprint,
        physical_batch_sha256=physical_batch.fingerprint,
        physical_batch_index=0,
        predecessor_request_id=request.predecessor_request_id,
        initial_cumulative_state_sha256=request.initial_cumulative_state_sha256,
    )
    resolved_raw_text = raw_generated_text or (
        "<|object_ref_start|>person<|object_ref_end|>"
        "<|box_start|><|coord_100|><|coord_100|>"
        "<|coord_500|><|coord_500|><|box_end|>"
    )
    resolved_parser_text = resolved_raw_text if parser_text is None else parser_text
    resolved_token_ids = [int(value) for value in generated_token_ids]
    resolved_trace = (
        list(token_trace)
        if token_trace is not None
        else [
            TokenTrace(
                step_index=index,
                token_id=token_id,
                token_text=(resolved_raw_text if index == 0 else f"token-{token_id}"),
                logprob=-0.25,
                is_stop=False,
                is_pad=False,
                backend="hf",
                backend_mode="generate",
                response_family="hf",
            )
            for index, token_id in enumerate(resolved_token_ids)
        ]
    )
    model_identity = {"family": "test"}
    tokenizer_identity = {"sha256": "test-tokenizer"}
    generation_config_fingerprint = "test-generation-config"
    generator = torch.Generator(device="cpu").manual_seed(sampling_seed)
    receipt = build_decode_execution_receipt(
        request=decode_request,
        generated_token_ids=resolved_token_ids,
        token_trace=resolved_trace,
        stop_reason="length",
        model_identity=model_identity,
        tokenizer_identity=tokenizer_identity,
        generation_config_fingerprint=generation_config_fingerprint,
        executed_generation_arguments=effective_generation_arguments(
            policy,
            eos_token_id=151645,
            pad_token_id=0,
        ),
        request_generator=generator,
        request_execution_index=0,
        batch_request_order_fingerprint=batch_request_order_fingerprint(
            decode_requests
        ),
        runtime_identity={"runtime": "test"},
        custom_sampler_executed=True,
    )
    decode_result = DecodeResult(
        request_id=request.request_id,
        backend="hf",
        backend_mode="generate",
        response_family="hf",
        prompt_token_ids=list(decode_request.prompt_token_ids),
        generated_token_ids=resolved_token_ids,
        raw_generated_text=resolved_raw_text,
        parser_text=resolved_parser_text,
        strip_policy="none",
        stop_reason="length",
        model_identity=model_identity,
        tokenizer_identity=tokenizer_identity,
        generation_config_fingerprint=generation_config_fingerprint,
        token_trace=resolved_trace,
        execution_receipt=receipt,
    )

    continuation_text = ""
    continuation_sha256 = None
    continuation_fields: dict[str, object] = {}
    predecessor_attempt = None
    if arm.cumulative_dependency and resolved_cell_index != 0:
        accepted_rows = tuple(
            cumulative_accepted_rows
            if cumulative_accepted_rows is not None
            else (resolved_raw_text,)
        )
        continuation_text = "".join(accepted_rows)
        continuation_sha256 = (
            hashlib.sha256(continuation_text.encode("utf-8")).hexdigest()
            if continuation_text
            else None
        )
        state_path = root / f"state-{request.request_id.rsplit(':', 1)[-1]}.json"
        state_path.write_text(
            json.dumps(
                {
                    "accepted_global_coordinate_rows": list(accepted_rows),
                    "schema_version": "accepted_row_prefix_state.v1",
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        state_sha256 = sha256_file(state_path)
        predecessor_attempt = AttemptRecord(
            run_id="test-run",
            schedule_sha256=_digest("schedule-artifact"),
            request_id=request.predecessor_request_id or "missing",
            physical_batch_plan_sha256=physical_batch_plan.fingerprint,
            physical_batch_sha256=physical_batch.fingerprint,
            physical_batch_index=0,
            attempt_status="completed",
            started_at_utc="2026-07-13T00:00:00Z",
            finished_at_utc="2026-07-13T00:00:01Z",
            execution_identity=ExecutionIdentityBundle(
                code_sha256=_digest("code"),
                config_sha256=_digest("config"),
                ledger_sha256=_digest("ledger"),
                runtime_sha256=_digest("runtime"),
            ),
            output_artifact_sha256=_digest("predecessor-output"),
            expected_cumulative_state_sha256=_digest("predecessor-input"),
            produced_cumulative_state_sha256=state_sha256,
            produced_cumulative_state_artifact_path=str(state_path),
        )
    ordinary_chat_text = (
        "<|im_start|>user\n<|image_pad|>detect<|im_end|>\n<|im_start|>assistant\n"
    )
    chat_text = ordinary_chat_text + continuation_text
    if continuation_text:
        continuation_character_start = len(ordinary_chat_text)
        continuation_byte_start = len(ordinary_chat_text.encode("utf-8"))
        continuation_fields = {
            "open_assistant_content_start_character": continuation_character_start,
            "open_assistant_content_start_byte": continuation_byte_start,
            "continuation_character_span": (
                continuation_character_start,
                len(chat_text),
            ),
            "continuation_byte_span": (
                continuation_byte_start,
                len(chat_text.encode("utf-8")),
            ),
            "continuation_token_impact_span": (
                0,
                len(decode_request.prompt_token_ids),
            ),
            "image_placeholder_count": 1,
            "open_assistant_interval_verified": True,
        }
    full_prompt_fingerprint = hashlib.sha256(
        json.dumps(
            {
                "full_chat_text": chat_text,
                "prompt_token_ids": list(decode_request.prompt_token_ids),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    prompt_record = PromptRecord(
        row_id=str(image_id),
        row_index=0,
        example_id=str(image_id),
        messages=(),
        prompt_text="detect",
        chat_text=chat_text,
        prompt_token_ids=list(decode_request.prompt_token_ids),
        template_id="test-template",
        template_fingerprint=_digest("template"),
        object_ordering="geometry_sorted",
        object_field_order="description_first",
        assistant_format="compact",
        realized_object_order=[],
        full_prompt_fingerprint=full_prompt_fingerprint,
        continuation_text_sha256=continuation_sha256,
        **continuation_fields,  # type: ignore[arg-type]
    )

    if arm.input_policy == "complete_source_canvas":
        image_plan = ImagePlanRow(
            row_id=str(image_id),
            row_index=0,
            example_id=str(image_id),
            image_path=str(image_path),
            declared_width=source_width,
            declared_height=source_height,
            decoded_width=source_width,
            decoded_height=source_height,
            patch_size=16,
            merge_size=2,
            temporal_patch_size=2,
            expected_image_grid_thw=[1, source_height // 16, source_width // 16],
            observed_image_grid_thw=[1, source_height // 16, source_width // 16],
            raw_patch_rows=(source_height // 16) * (source_width // 16),
            merged_visual_tokens=(source_height // 32) * (source_width // 32),
            do_resize=False,
            status="ok",
            error=None,
        )
        envelope = ExecutionEvidenceEnvelope.from_full_image_execution(
            scheduled_request=request,
            grid_provenance=grid_provenance,
            decode_provenance=decode,
            prompt_record=prompt_record,
            decode_request_batch=decode_requests,
            request_batch=request_batch,
            physical_batch_plan=physical_batch_plan,
            attempt_dependency=attempt_dependency,
            predecessor_attempt=predecessor_attempt,
            image_plan_row=image_plan,
            visual_materialization=visual_materialization,
            decode_result=decode_result,
        )
        return envelope, decode_result

    envelope = ExecutionEvidenceEnvelope.from_spatial_execution(
        scheduled_request=request,
        grid_provenance=grid_provenance,
        decode_provenance=decode,
        prompt_record=prompt_record,
        decode_request_batch=decode_requests,
        request_batch=request_batch,
        physical_batch_plan=physical_batch_plan,
        attempt_dependency=attempt_dependency,
        predecessor_attempt=predecessor_attempt,
        visual_materialization=visual_materialization,
        decode_result=decode_result,
    )
    return envelope, decode_result


def _digest(label: str) -> str:
    return sha256_payload({"label": label})
