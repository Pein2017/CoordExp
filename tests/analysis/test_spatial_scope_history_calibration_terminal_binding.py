from __future__ import annotations

import json
import math
from pathlib import Path

from PIL import Image
import pytest
import torch

from src.analysis.spatial_scope_history.calibration import (
    CalibrationBackendAttestationBinding,
    CalibrationRequest,
    CalibrationTerminalBundle,
    _decode_receipt_runtime_contract,
    load_calibration_backend_attestation_binding,
    reconstruct_calibration_observation,
)
from src.analysis.spatial_scope_history.cohort_ledger import sha256_file, sha256_payload
from src.analysis.spatial_scope_history.metrics import ReferenceObject
from src.analysis.spatial_scope_history.schedule import (
    PRIMARY_ROOT_SEED,
    derive_sampling_seed,
)
from src.analysis.spatial_scope_history.spatial import MaterializedVisualInput
from src.common.errors import ArtifactContractError, RuntimeContractError
from src.data.geometry import coord_bins_to_pixel_xyxy
from src.eval.detection_categories import (
    COCO_80_CATEGORY_NAMESPACE_SHA256,
    COCO_80_EVALUATOR_LOCAL_CATEGORY_ID_BY_NAME,
    COCO_80_OFFICIAL_CATEGORY_ID_BY_NAME,
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


def _digest(label: str) -> str:
    return sha256_payload({"label": label})


def _request(
    *, image_sha256: str, call_index: int, temperature: float = 0.4
) -> CalibrationRequest:
    policy = DecodeGenerationPolicy.sampled(
        max_new_tokens=512,
        repetition_penalty=1.0,
        temperature=temperature,
        top_p=0.95,
    )
    call_label = f"call-{call_index:02d}"
    payload = {
        "calibration_cohort_sha256": _digest("calibration-cohort"),
        "call_index": call_index,
        "call_label": call_label,
        "decode_generation_policy_fingerprint": policy.fingerprint,
        "image_id": 42,
        "image_sha256": image_sha256,
        "sampling_seed": derive_sampling_seed(
            root_seed=PRIMARY_ROOT_SEED,
            role="temperature-calibration",
            image_id=42,
            cell_or_call_label=call_label,
        ),
        "schema_version": "spatial_scope_history.calibration_request.v1",
        "temperature": temperature,
    }
    return CalibrationRequest(
        **payload,
        image_frozen_order=0,
        request_id=(
            "spatial-scope-history-calibration-request:" + sha256_payload(payload)
        ),
    )


def _terminal_bundle(
    tmp_path: Path,
    *,
    call_index: int,
    temperature: float = 0.4,
    prompt_suffix_count: int = 0,
) -> CalibrationTerminalBundle:
    image_path = tmp_path / "image.png"
    if not image_path.exists():
        image = Image.new("RGB", (128, 128), color=(11, 23, 37))
        image.save(image_path)
        image.close()
    request = _request(
        image_sha256=sha256_file(image_path),
        call_index=call_index,
        temperature=temperature,
    )

    def image_processor(*, images, return_tensors, do_resize):
        assert len(images) == 1
        assert return_tensors == "pt"
        assert do_resize is False
        return {
            "pixel_values": torch.zeros((64, 1536), dtype=torch.float32),
            "image_grid_thw": torch.tensor([[1, 8, 8]], dtype=torch.int64),
        }

    visual = MaterializedVisualInput.from_full_image_path(
        source_image_path=image_path,
        expected_source_image_sha256=request.image_sha256,
        image_processor=image_processor,
        processor_contract_sha256=_digest("processor-contract"),
    )
    prompt_ids = [11, 12, call_index + 13, *range(20, 20 + prompt_suffix_count)]
    policy = DecodeGenerationPolicy.sampled(
        max_new_tokens=512,
        repetition_penalty=1.0,
        temperature=temperature,
        top_p=0.95,
    )
    decode_request = DecodeRequest(
        request_id=request.request_id,
        prompt_token_ids=prompt_ids,
        model_inputs={
            "pixel_values": visual.pixel_values,
            "image_grid_thw": visual.image_grid_thw,
        },
        generation_policy=policy,
        sampling_seed=request.sampling_seed,
    )
    pieces = [
        "<|object_ref_start|>",
        "cat",
        "<|object_ref_end|>",
        "<|box_start|>",
        "<|coord_100|>",
        "<|coord_200|>",
        "<|coord_300|>",
        "<|coord_400|>",
        "<|box_end|>",
        "<|im_end|>",
    ]
    generated_ids = list(range(100, 100 + len(pieces)))
    trace = [
        TokenTrace(
            step_index=index,
            token_id=generated_ids[index],
            token_text=piece,
            logprob=math.log(0.5),
            is_stop=piece == "<|im_end|>",
            is_pad=False,
            backend="hf",
            backend_mode="generate",
            response_family="hf",
        )
        for index, piece in enumerate(pieces)
    ]
    generator = torch.Generator(device="cpu").manual_seed(request.sampling_seed)
    model_identity = {"family": "test-qwen"}
    tokenizer_identity = {"fingerprint": "test-tokenizer"}
    generation_config_fingerprint = _digest("generation-config")
    receipt = build_decode_execution_receipt(
        request=decode_request,
        generated_token_ids=generated_ids,
        token_trace=trace,
        stop_reason="im_end",
        model_identity=model_identity,
        tokenizer_identity=tokenizer_identity,
        generation_config_fingerprint=generation_config_fingerprint,
        executed_generation_arguments=effective_generation_arguments(
            policy,
            eos_token_id=151645,
            pad_token_id=0,
        ),
        request_generator=generator,
        request_execution_index=call_index,
        batch_request_order_fingerprint=batch_request_order_fingerprint(
            [decode_request]
        ),
        runtime_identity={"runtime": "test"},
        attention_implementation="sdpa",
        custom_sampler_executed=True,
    )
    row_text = "".join(pieces[:-1])
    result = DecodeResult(
        request_id=request.request_id,
        backend="hf",
        backend_mode="generate",
        response_family="hf",
        prompt_token_ids=prompt_ids,
        generated_token_ids=generated_ids,
        raw_generated_text="".join(pieces),
        parser_text=row_text,
        strip_policy="terminal_im_end",
        stop_reason="im_end",
        model_identity=model_identity,
        tokenizer_identity=tokenizer_identity,
        generation_config_fingerprint=generation_config_fingerprint,
        token_trace=trace,
        execution_receipt=receipt,
    )
    prompt_record = {
        "assistant_format": "compact",
        "example_id": "42",
        "full_prompt_fingerprint": _digest(f"prompt-{call_index}"),
        "object_field_order": "description_first",
        "object_ordering": "geometry_sorted",
        "prompt_text": "detect",
        "prompt_token_count": len(prompt_ids),
        "prompt_token_ids": prompt_ids,
        "realized_object_order": [],
        "row_id": "42",
        "row_index": 0,
        "template_fingerprint": _digest("template"),
        "template_id": "test-template",
    }
    visual_receipt = visual.receipt.to_artifact_dict()
    return CalibrationTerminalBundle(
        panel_kind="initial",
        request=request,
        physical_batch_index=0,
        request_execution_index=call_index,
        prompt_record=prompt_record,
        visual_input_materialization_receipt=visual_receipt,
        model_input_sha256=sha256_payload(
            {
                "prompt_token_ids": prompt_ids,
                "visual_input_materialization_receipt": visual_receipt,
            }
        ),
        decode_result=result,
        backend_attestation_aggregate_sha256=_digest("aggregate-file"),
        backend_attestation_aggregate_fingerprint=_digest("aggregate-payload"),
    )


def _attestation(
    bundle: CalibrationTerminalBundle,
) -> CalibrationBackendAttestationBinding:
    receipt = bundle.decode_result.execution_receipt
    assert receipt is not None
    base = _decode_receipt_runtime_contract(receipt)
    contracts = []
    for temperature in (0.2, 0.4, 0.6):
        policy = DecodeGenerationPolicy.sampled(
            max_new_tokens=512,
            repetition_penalty=1.0,
            temperature=temperature,
            top_p=0.95,
        )
        contract = {**base, "decode_generation_policy_fingerprint": policy.fingerprint}
        contracts.append((policy.fingerprint, tuple(sorted(contract.items()))))
    return CalibrationBackendAttestationBinding(
        aggregate_artifact_sha256=bundle.backend_attestation_aggregate_sha256,
        aggregate_payload_fingerprint=(
            bundle.backend_attestation_aggregate_fingerprint
        ),
        execution_contracts_by_policy=tuple(contracts),
    )


def _reference() -> ReferenceObject:
    category = "cat"
    return ReferenceObject(
        image_id="42",
        ledger_scope="official_annotation",
        reference_id="coco-ann:1",
        normalized_category_name=category,
        evaluator_category_id=COCO_80_EVALUATOR_LOCAL_CATEGORY_ID_BY_NAME[category],
        official_coco_category_id=COCO_80_OFFICIAL_CATEGORY_ID_BY_NAME[category],
        category_namespace_sha256=COCO_80_CATEGORY_NAMESPACE_SHA256,
        source_canvas_bbox_xyxy=tuple(
            float(value)
            for value in coord_bins_to_pixel_xyxy(
                (100, 200, 300, 400),
                image_width=128,
                image_height=128,
                field="test.coordinate_bins",
            )
        ),
        state="accepted",
        provenance="official_annotation",
    )


def test_terminal_bundle_reconstructs_gate_inputs_from_result_evidence(
    tmp_path: Path,
) -> None:
    bundle = _terminal_bundle(tmp_path, call_index=0)
    observation = reconstruct_calibration_observation(
        terminal_bundle=bundle,
        backend_attestation=_attestation(bundle),
        source_width=128,
        source_height=128,
        official_reference_objects=(_reference(),),
    )
    assert observation.natural_closure
    assert observation.parse_without_call_level_failure
    assert observation.detected_official_reference_object_ids == ("coco-ann:1",)


def test_swapped_or_recomputed_response_cannot_enter_calibration(
    tmp_path: Path,
) -> None:
    first = _terminal_bundle(tmp_path, call_index=0)
    second = _terminal_bundle(tmp_path, call_index=1)
    payload = first.to_artifact_dict()
    payload["decode_result"] = second.decode_result.to_artifact_dict()
    payload["bundle_sha256"] = None
    identity = dict(payload)
    identity.pop("bundle_sha256")
    payload["bundle_sha256"] = sha256_payload(identity)
    with pytest.raises(
        (ArtifactContractError, RuntimeContractError), match="receipt|binding"
    ) as exc_info:
        CalibrationTerminalBundle.from_artifact_dict(payload)
    assert "execution" in str(exc_info.value) or "receipt" in str(exc_info.value)

    altered = first.to_artifact_dict()
    altered["decode_result"]["token_trace"][4]["logprob"] = math.log(0.9)
    altered["bundle_sha256"] = None
    altered_identity = dict(altered)
    altered_identity.pop("bundle_sha256")
    altered["bundle_sha256"] = sha256_payload(altered_identity)
    with pytest.raises(RuntimeContractError, match="receipt|score|binding"):
        CalibrationTerminalBundle.from_artifact_dict(altered)


def test_terminal_bundle_unknown_field_is_rejected(tmp_path: Path) -> None:
    payload = _terminal_bundle(tmp_path, call_index=0).to_artifact_dict()
    payload["trusted_gate_counter"] = 48
    with pytest.raises(ArtifactContractError, match="keys"):
        CalibrationTerminalBundle.from_artifact_dict(payload)


def test_backend_binding_accepts_validated_backend_serialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundles = [
        _terminal_bundle(
            tmp_path,
            call_index=index,
            temperature=temperature,
            prompt_suffix_count=index,
        )
        for index, temperature in enumerate((0.2, 0.4, 0.6))
    ]
    receipts = [bundle.decode_result.execution_receipt for bundle in bundles]
    assert all(receipt is not None for receipt in receipts)
    aggregate_fingerprint = _digest("aggregate-payload")
    artifact = {
        "policy_attestations": [
            {
                "decode_generation_policy_fingerprint": (
                    bundle.request.decode_generation_policy_fingerprint
                ),
                "attestation_bundle": {
                    "executed_cases": [
                        {
                            "result_artifacts": [
                                bundle.decode_result.to_artifact_dict()
                            ]
                        }
                    ]
                },
            }
            for bundle in bundles
        ]
    }
    path = tmp_path / "backend-attestation.json"
    path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        "src.analysis.spatial_scope_history.calibration."
        "validate_sampled_runtime_attestation_aggregate_output",
        lambda _path: {"aggregate_payload_fingerprint": aggregate_fingerprint},
    )

    binding = load_calibration_backend_attestation_binding(path)

    assert binding.aggregate_payload_fingerprint == aggregate_fingerprint
    assert binding.aggregate_artifact_sha256 == sha256_file(path)
    assert [policy for policy, _contract in binding.execution_contracts_by_policy] == [
        bundle.request.decode_generation_policy_fingerprint for bundle in bundles
    ]
