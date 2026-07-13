from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from src.analysis.spatial_scope_history.cohort_ledger import sha256_payload
from src.analysis.spatial_scope_history.execution_evidence import (
    ExecutedRequestBatchEvidence,
)
from src.analysis.spatial_scope_history.spatial import (
    ExecutedVisualTensorReceipt,
)
from src.common.errors import DataContractError
from spatial_scope_history_fixtures import build_test_execution_evidence


def test_same_dimension_source_image_swap_fails_before_prediction_normalization() -> (
    None
):
    first = build_test_execution_evidence(
        arm_code="FULL_SINGLE",
        image_id=1,
        sampling_seed=101,
    )
    second = build_test_execution_evidence(
        arm_code="FULL_SINGLE",
        image_id=2,
        sampling_seed=101,
    )

    with pytest.raises(
        DataContractError,
        match="dependency_request|source_mismatch",
    ):
        replace(
            first,
            scheduled_request=second.scheduled_request,
            request_fingerprint=second.request_fingerprint,
            envelope_fingerprint=None,
        )


def test_same_cell_request_and_seed_swap_fails_against_embedded_decode_receipt() -> (
    None
):
    first = build_test_execution_evidence(
        arm_code="FULL_BAG_K",
        image_id=1,
        cell_index=4,
        sampling_seed=201,
    )
    second = build_test_execution_evidence(
        arm_code="FULL_BAG_K",
        image_id=1,
        cell_index=4,
        sampling_seed=202,
    )

    with pytest.raises(
        DataContractError,
        match="dependency_request|decode_request|sampling_seed",
    ):
        replace(
            first,
            scheduled_request=second.scheduled_request,
            request_fingerprint=second.request_fingerprint,
            envelope_fingerprint=None,
        )


@pytest.mark.parametrize(
    ("field_name", "expected_code"),
    [
        ("processor_receipt_sha256", "processor_digest"),
        ("decode_receipt_fingerprint", "decode_receipt"),
        ("generated_output_token_sha256", "output_tokens"),
        ("token_trace_sha256", "token_trace"),
    ],
)
def test_mutated_execution_digest_fails_before_merge_evidence(
    field_name: str,
    expected_code: str,
) -> None:
    evidence = build_test_execution_evidence(
        arm_code="TILE_RESET",
        image_id=1,
        cell_index=3,
        sampling_seed=303,
    )

    with pytest.raises(DataContractError, match=expected_code):
        replace(
            evidence,
            **{field_name: "f" * 64, "envelope_fingerprint": None},
        )


def test_same_shape_wrong_visual_tensor_values_fail_closed() -> None:
    evidence = build_test_execution_evidence(
        arm_code="MASK_RESET",
        image_id=1,
        cell_index=5,
        sampling_seed=401,
    )
    executed = evidence.executed_visual_tensor_receipt
    wrong_values = ExecutedVisualTensorReceipt.from_tensors(
        pixel_values=torch.ones(executed.pixel_values.shape, dtype=torch.float32),
        image_grid_thw=torch.tensor([[1, 8, 8]], dtype=torch.int64),
    )

    assert wrong_values.pixel_values.shape == executed.pixel_values.shape
    assert wrong_values.image_grid_thw.shape == executed.image_grid_thw.shape
    with pytest.raises(DataContractError, match="visual_tensor_mismatch"):
        replace(
            evidence,
            executed_visual_tensor_receipt=wrong_values,
            envelope_fingerprint=None,
        )


def test_reset_prompt_cannot_substitute_for_cumulative_prompt_state() -> None:
    cumulative = build_test_execution_evidence(
        arm_code="MASK_CUMULATIVE",
        image_id=1,
        cell_index=2,
        sampling_seed=402,
    )
    reset = build_test_execution_evidence(
        arm_code="MASK_RESET",
        image_id=1,
        cell_index=2,
        sampling_seed=402,
    )

    assert (
        cumulative.executed_prompt_evidence.prompt_token_ids
        == reset.executed_prompt_evidence.prompt_token_ids
    )
    assert (
        cumulative.executed_prompt_evidence.full_prompt_fingerprint
        != reset.executed_prompt_evidence.full_prompt_fingerprint
    )
    with pytest.raises(DataContractError, match="cumulative_prompt_binding"):
        replace(
            cumulative,
            executed_prompt_evidence=reset.executed_prompt_evidence,
            envelope_fingerprint=None,
        )


def test_cumulative_predecessor_artifact_mutation_invalidates_envelope() -> None:
    evidence = build_test_execution_evidence(
        arm_code="MASK_CUMULATIVE",
        image_id=1,
        cell_index=2,
        sampling_seed=403,
    )
    cumulative = evidence.cumulative_state_dependency_evidence
    assert cumulative is not None
    assert cumulative.predecessor_state_artifact_path is not None
    state_path = cumulative.predecessor_state_artifact_path
    with open(state_path, "a", encoding="utf-8") as stream:
        stream.write("\n")

    with pytest.raises(DataContractError, match="cumulative_artifact_digest"):
        replace(evidence, envelope_fingerprint=None)


def test_empty_cumulative_state_preserves_fresh_prompt_with_dependency() -> None:
    evidence = build_test_execution_evidence(
        arm_code="MASK_CUMULATIVE",
        image_id=1,
        cell_index=2,
        sampling_seed=407,
        cumulative_accepted_rows=(),
    )

    assert evidence.cumulative_state_dependency_evidence is not None
    assert evidence.executed_prompt_evidence.continuation_text_sha256 is None


def test_reordered_backend_batch_cannot_rebind_execution_index() -> None:
    evidence = build_test_execution_evidence(
        arm_code="FULL_SINGLE",
        image_id=1,
        sampling_seed=404,
    )
    batch = evidence.executed_request_batch_evidence
    request_ids = list(batch.batch_request_ids)
    prompt_hashes = list(batch.batch_prompt_token_ids_sha256)
    request_ids[0], request_ids[1] = request_ids[1], request_ids[0]
    prompt_hashes[0], prompt_hashes[1] = prompt_hashes[1], prompt_hashes[0]
    reordered = _rebuild_batch_evidence(
        batch,
        request_ids=tuple(request_ids),
        prompt_hashes=tuple(prompt_hashes),
        request_execution_index=1,
        decode_policy_sha256=(
            evidence.decode_provenance.canonical_generation_policy_sha256
        ),
    )

    with pytest.raises(DataContractError, match="backend_batch_fingerprint"):
        replace(
            evidence,
            executed_request_batch_evidence=reordered,
            envelope_fingerprint=None,
        )


def test_cross_batch_evidence_cannot_substitute_for_sealed_batch() -> None:
    first = build_test_execution_evidence(
        arm_code="FULL_SINGLE",
        image_id=1,
        sampling_seed=405,
    )
    other = build_test_execution_evidence(
        arm_code="FULL_SINGLE",
        image_id=2,
        sampling_seed=405,
    )

    with pytest.raises(DataContractError, match="batch_dependency_binding"):
        replace(
            first,
            executed_request_batch_evidence=(other.executed_request_batch_evidence),
            envelope_fingerprint=None,
        )


def test_singleton_executed_batch_evidence_is_rejected() -> None:
    evidence = build_test_execution_evidence(
        arm_code="FULL_SINGLE",
        image_id=1,
        sampling_seed=406,
    )
    batch = evidence.executed_request_batch_evidence

    with pytest.raises(DataContractError, match="batch_cardinality"):
        ExecutedRequestBatchEvidence(
            schedule_identity_sha256=batch.schedule_identity_sha256,
            physical_batch_plan_sha256=batch.physical_batch_plan_sha256,
            physical_batch_sha256=batch.physical_batch_sha256,
            physical_batch_index=batch.physical_batch_index,
            batch_request_ids=(batch.batch_request_ids[0],),
            batch_cardinality=1,
            request_execution_index=0,
            backend_batch_order_fingerprint=(batch.backend_batch_order_fingerprint),
            batch_prompt_token_ids_sha256=(batch.batch_prompt_token_ids_sha256[0],),
            evidence_sha256="0" * 64,
        )


def _rebuild_batch_evidence(
    original: ExecutedRequestBatchEvidence,
    *,
    request_ids: tuple[str, ...],
    prompt_hashes: tuple[str, ...],
    request_execution_index: int,
    decode_policy_sha256: str,
) -> ExecutedRequestBatchEvidence:
    backend_fingerprint = sha256_payload(
        [
            {
                "decode_generation_policy_fingerprint": decode_policy_sha256,
                "prompt_token_identifiers_hash": prompt_sha256,
                "request_id": request_id,
            }
            for request_id, prompt_sha256 in zip(
                request_ids,
                prompt_hashes,
                strict=True,
            )
        ]
    )
    identity = {
        "backend_batch_order_fingerprint": backend_fingerprint,
        "batch_cardinality": len(request_ids),
        "batch_prompt_token_ids_sha256": list(prompt_hashes),
        "batch_request_ids": list(request_ids),
        "physical_batch_index": original.physical_batch_index,
        "physical_batch_plan_sha256": original.physical_batch_plan_sha256,
        "physical_batch_sha256": original.physical_batch_sha256,
        "request_execution_index": request_execution_index,
        "schedule_identity_sha256": original.schedule_identity_sha256,
        "schema_version": original.schema_version,
    }
    return ExecutedRequestBatchEvidence(
        schedule_identity_sha256=original.schedule_identity_sha256,
        physical_batch_plan_sha256=original.physical_batch_plan_sha256,
        physical_batch_sha256=original.physical_batch_sha256,
        physical_batch_index=original.physical_batch_index,
        batch_request_ids=request_ids,
        batch_cardinality=len(request_ids),
        request_execution_index=request_execution_index,
        backend_batch_order_fingerprint=backend_fingerprint,
        batch_prompt_token_ids_sha256=prompt_hashes,
        evidence_sha256=sha256_payload(identity),
        schema_version=original.schema_version,
    )
