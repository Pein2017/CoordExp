from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

import scripts.research.collect_human13_rp_crossover as adapter


def _request_receipt(request: adapter.AcquisitionRequest) -> adapter.NativeRequestReceipt:
    return adapter.NativeRequestReceipt(
        request_id=request.request_id,
        seed=request.seed,
        physical_batch_index=request.physical_batch_index,
        request_order_in_batch=request.request_order_in_batch,
        sampling_params=adapter.expected_native_sampling_evidence(request),
        prompt_token_ids_sha256=adapter.token_ids_sha256((1, 2)),
        model_id="model",
        model_identity_sha256="c" * 64,
        session_identity_sha256="d" * 64,
    )


def _output_receipt(request: adapter.AcquisitionRequest) -> adapter.NativeOutputReceipt:
    native = _request_receipt(request)
    return adapter.NativeOutputReceipt(
        native_request_receipt_sha256=native.content_sha256,
        request_id=request.request_id,
        seed=request.seed,
        physical_batch_index=request.physical_batch_index,
        request_order_in_batch=request.request_order_in_batch,
        prompt_token_ids=(1, 2),
        source_sha256="a" * 64,
        manifest_sha256="b" * 64,
        model_id="model",
        tokenizer_id="tokenizer",
        processor_id="processor",
        processor_order=("repetition_penalty", "temperature", "log_softmax"),
        sampler_backend_id="vllm:test",
        generated_token_ids=(151645,),
        processed_logprobs=(-math.log(151646),),
        terminal_kind="natural_stop",
    )


def _batch_receipt(batch: adapter.AcquisitionBatch) -> adapter.NativeBatchReceipt:
    requests = tuple(_request_receipt(request) for request in batch.requests)
    outputs = tuple(_output_receipt(request) for request in batch.requests)
    return adapter.NativeBatchReceipt(requests=requests, outputs=outputs)


def _execute(plan: adapter.AcquisitionGroupPlan) -> adapter.AcquisitionExecution:
    return adapter.execute_acquisition_group(
        plan=plan,
        execute_batch=lambda batch, params: _batch_receipt(batch),
    )


def test_plan_seals_all_four_seed_groups_batch_four_and_clear_panel_totals() -> None:
    plan = adapter.plan_acquisition_group(
        image_id=1584, repetition_penalty=1.10, seed_group_id="matrix_b"
    )
    assert adapter.QUALIFICATION_SEEDS == tuple(range(30001, 30017))
    assert adapter.MATRIX_SEED_GROUPS == {
        "matrix_a": tuple(range(31001, 31017)),
        "matrix_b": tuple(range(32001, 32017)),
        "matrix_c": tuple(range(33001, 33017)),
    }
    assert [tuple(item.seed for item in batch.requests) for batch in plan.batches] == [
        (32001, 32002, 32003, 32004),
        (32005, 32006, 32007, 32008),
        (32009, 32010, 32011, 32012),
        (32013, 32014, 32015, 32016),
    ]
    assert len(plan.requests) == len(set(item.request_id for item in plan.requests)) == 16
    assert all(item.sampling["top_p"] == 1.0 and item.sampling["top_k"] is None for item in plan.requests)
    dry = adapter.dry_run_plan()
    assert dry["panel_image_count"] == 13
    assert dry["group_physical_batch_count"] == 4
    assert dry["group_request_count"] == 16
    assert dry["panel_physical_batch_count"] == 52
    assert dry["panel_request_count"] == 208


def test_vllm_sampling_params_captures_exact_native_no_top_k_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeSamplingParams:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)
            self.__dict__.update(kwargs)

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=FakeSamplingParams))
    request = adapter.plan_acquisition_group(image_id=1584, repetition_penalty=1.10, seed_group_id="qualification").requests[0]
    params = adapter.vllm_sampling_params(request)
    assert adapter.native_sampling_evidence(params) == adapter.expected_native_sampling_evidence(request)
    assert captured["top_p"] == 1.0
    assert captured["top_k"] == 0
    assert captured["logprobs"] == 1
    assert captured["stop_token_ids"] == [151645]


def test_execute_requires_receipt_captured_lineage_in_exact_order() -> None:
    plan = adapter.plan_acquisition_group(image_id=1584, repetition_penalty=1.0, seed_group_id="qualification")
    execution = _execute(plan)
    assert execution.group.seed_group_id == "qualification"
    assert execution.plan_sha256 == plan.content_sha256
    assert len(execution.native_batch_receipts) == 4
    assert tuple(item.identity.request_id for item in execution.group.trajectories) == tuple(item.request_id for item in plan.requests)


def test_direct_execution_construction_rebuilds_canonical_native_group() -> None:
    plan = adapter.plan_acquisition_group(image_id=1584, repetition_penalty=1.0, seed_group_id="qualification")
    execution = _execute(plan)
    changed_request = replace(execution.native_batch_receipts[0].requests[0], session_identity_sha256="e" * 64)
    changed_output = replace(
        execution.native_batch_receipts[0].outputs[0], native_request_receipt_sha256=changed_request.content_sha256
    )
    changed_batch = adapter.NativeBatchReceipt(
        requests=(changed_request, *execution.native_batch_receipts[0].requests[1:]),
        outputs=(changed_output, *execution.native_batch_receipts[0].outputs[1:]),
    )
    with pytest.raises(ValueError, match="session"):
        replace(execution, native_batch_receipts=(changed_batch, *execution.native_batch_receipts[1:]))

    cross_rp_group = _execute(
        adapter.plan_acquisition_group(image_id=1584, repetition_penalty=1.10, seed_group_id="qualification")
    ).group
    with pytest.raises(ValueError, match="canonical native group"):
        replace(execution, group=cross_rp_group)


def test_frozen_execution_and_native_artifact_snapshot_caller_lists(tmp_path: Path) -> None:
    from scripts.research.human13_rp_policy import validate_acquisition_group_replay

    execution = _execute(
        adapter.plan_acquisition_group(image_id=1584, repetition_penalty=1.0, seed_group_id="qualification")
    )
    caller_receipts = list(execution.native_batch_receipts)
    sealed = adapter.AcquisitionExecution(
        plan=execution.plan,
        plan_sha256=execution.plan_sha256,
        plan_request_ids=execution.plan_request_ids,
        native_batch_receipts=caller_receipts,  # type: ignore[arg-type]
        group=execution.group,
    )
    caller_artifact_batches = list(execution.native_batch_receipts)
    artifact = adapter.NativeReceiptsArtifact(
        plan_sha256=execution.plan_sha256,
        plan_request_ids=execution.plan_request_ids,
        batch_receipts=caller_artifact_batches,  # type: ignore[arg-type]
    )
    execution_hash = sealed.native_receipts_artifact.content_sha256
    artifact_hash = artifact.content_sha256
    caller_receipts.clear()
    caller_artifact_batches.clear()
    assert isinstance(sealed.native_batch_receipts, tuple)
    assert isinstance(artifact.batch_receipts, tuple)
    assert len(sealed.native_batch_receipts) == len(artifact.batch_receipts) == 4
    assert sealed.native_receipts_artifact.content_sha256 == execution_hash
    assert artifact.content_sha256 == artifact_hash
    receipt = validate_acquisition_group_replay(sealed.group, sealed.group, adapter.ReplayTolerance())
    output = tmp_path / "sealed-after-list-mutation"
    adapter.publish_acquisition_group(output_root=output, execution=sealed, replayed=sealed.group, replay_receipt=receipt)
    assert adapter.load_published_acquisition(output, plan=sealed.plan).binding.native_receipts_sha256 == execution_hash


@pytest.mark.parametrize("mutation, message", [
    (lambda receipt: replace(receipt, request_id="historical-claim"), "request identity"),
    (lambda receipt: replace(receipt, seed=99999), "seed"),
    (lambda receipt: replace(receipt, request_order_in_batch=3), "request order"),
    (lambda receipt: replace(receipt, sampling_params={**dict(receipt.sampling_params), "top_p": 0.95}), "sampling"),
])
def test_execute_rejects_unsealed_or_wrong_native_request_receipt(
    mutation: object, message: str
) -> None:
    plan = adapter.plan_acquisition_group(image_id=1584, repetition_penalty=1.0, seed_group_id="qualification")

    def execute(batch: adapter.AcquisitionBatch, params: tuple[object, ...]) -> adapter.NativeBatchReceipt:
        receipt = _batch_receipt(batch)
        changed = mutation(receipt.requests[0])  # type: ignore[operator]
        return adapter.NativeBatchReceipt(requests=(changed, *receipt.requests[1:]), outputs=receipt.outputs)

    with pytest.raises(ValueError, match=message):
        adapter.execute_acquisition_group(plan=plan, execute_batch=execute)


def test_execute_rejects_reversed_outputs_and_mixed_prompt_session_model() -> None:
    plan = adapter.plan_acquisition_group(image_id=1584, repetition_penalty=1.0, seed_group_id="qualification")

    def reversed_outputs(batch: adapter.AcquisitionBatch, params: tuple[object, ...]) -> adapter.NativeBatchReceipt:
        receipt = _batch_receipt(batch)
        return adapter.NativeBatchReceipt(requests=receipt.requests, outputs=tuple(reversed(receipt.outputs)))

    with pytest.raises(ValueError, match="output ordering"):
        adapter.execute_acquisition_group(plan=plan, execute_batch=reversed_outputs)

    def mixed_prompt(batch: adapter.AcquisitionBatch, params: tuple[object, ...]) -> adapter.NativeBatchReceipt:
        receipt = _batch_receipt(batch)
        changed = replace(receipt.requests[0], prompt_token_ids_sha256="e" * 64)
        changed_output = replace(
            receipt.outputs[0], native_request_receipt_sha256=changed.content_sha256
        )
        return adapter.NativeBatchReceipt(
            requests=(changed, *receipt.requests[1:]),
            outputs=(changed_output, *receipt.outputs[1:]),
        )

    with pytest.raises(ValueError, match="prompt"):
        adapter.execute_acquisition_group(plan=plan, execute_batch=mixed_prompt)

    def mixed_model_session(batch: adapter.AcquisitionBatch, params: tuple[object, ...]) -> adapter.NativeBatchReceipt:
        receipt = _batch_receipt(batch)
        changed = replace(receipt.requests[0], model_identity_sha256="e" * 64)
        changed_output = replace(
            receipt.outputs[0], native_request_receipt_sha256=changed.content_sha256
        )
        return adapter.NativeBatchReceipt(
            requests=(changed, *receipt.requests[1:]),
            outputs=(changed_output, *receipt.outputs[1:]),
        )

    with pytest.raises(ValueError, match="model or session"):
        adapter.execute_acquisition_group(plan=plan, execute_batch=mixed_model_session)


def test_execute_rejects_duplicate_seed_or_request_identity_in_the_plan() -> None:
    plan = adapter.plan_acquisition_group(image_id=1584, repetition_penalty=1.0, seed_group_id="qualification")
    duplicate = replace(plan.batches[0].requests[1], seed=plan.batches[0].requests[0].seed)
    bad_batch = replace(plan.batches[0], requests=(plan.batches[0].requests[0], duplicate, *plan.batches[0].requests[2:]))
    bad_plan = replace(plan, batches=(bad_batch, *plan.batches[1:]))
    with pytest.raises(ValueError, match="seed ordering"):
        adapter.execute_acquisition_group(plan=bad_plan, execute_batch=lambda batch, params: _batch_receipt(batch))


def test_cap_processor_and_group_coverage_fail_closed() -> None:
    plan = adapter.plan_acquisition_group(image_id=1584, repetition_penalty=1.0, seed_group_id="qualification")

    def bad_cap(batch: adapter.AcquisitionBatch, params: tuple[object, ...]) -> adapter.NativeBatchReceipt:
        receipt = _batch_receipt(batch)
        output = replace(receipt.outputs[0], generated_token_ids=(1,), terminal_kind="cap_stop")
        return adapter.NativeBatchReceipt(requests=receipt.requests, outputs=(output, *receipt.outputs[1:]))

    with pytest.raises(ValueError, match="cap stop"):
        adapter.execute_acquisition_group(plan=plan, execute_batch=bad_cap)

    def missing(batch: adapter.AcquisitionBatch, params: tuple[object, ...]) -> adapter.NativeBatchReceipt:
        receipt = _batch_receipt(batch)
        return adapter.NativeBatchReceipt(requests=receipt.requests[:-1], outputs=receipt.outputs[:-1])

    with pytest.raises(ValueError, match="four"):
        adapter.execute_acquisition_group(plan=plan, execute_batch=missing)


def test_native_output_requires_the_exact_processor_order() -> None:
    request = adapter.plan_acquisition_group(
        image_id=1584, repetition_penalty=1.0, seed_group_id="qualification"
    ).requests[0]
    with pytest.raises(ValueError, match="processor order"):
        replace(
            _output_receipt(request),
            processor_order=("temperature", "repetition_penalty", "log_softmax"),
        )


def test_native_output_model_must_match_the_captured_native_request() -> None:
    plan = adapter.plan_acquisition_group(
        image_id=1584, repetition_penalty=1.0, seed_group_id="qualification"
    )

    def changed_model(batch: adapter.AcquisitionBatch, params: tuple[object, ...]) -> adapter.NativeBatchReceipt:
        receipt = _batch_receipt(batch)
        captured = replace(receipt.requests[0], model_id="other-model")
        output = replace(
            receipt.outputs[0], native_request_receipt_sha256=captured.content_sha256
        )
        return adapter.NativeBatchReceipt(
            requests=(captured, *receipt.requests[1:]),
            outputs=(output, *receipt.outputs[1:]),
        )

    with pytest.raises(ValueError, match="model identity"):
        adapter.execute_acquisition_group(plan=plan, execute_batch=changed_model)


def test_replay_requires_ordered_packed_rows_and_strict_parity() -> None:
    execution = _execute(adapter.plan_acquisition_group(image_id=1584, repetition_penalty=1.0, seed_group_id="qualification"))
    sampled = execution.group
    packed = adapter.PackedRawLogits(
        request_ids=tuple(item.identity.request_id for item in sampled.trajectories),
        token_indices=(0,) * 16,
        logits=torch.zeros((16, 151646), dtype=torch.float32),
    )
    replayed, receipt = adapter.replay_acquisition_group(sampled=sampled, packed=packed)
    assert type(receipt).__name__ == "AcquisitionGroupParityReceipt"
    assert replayed.content_sha256 != sampled.content_sha256
    with pytest.raises(ValueError, match="sealed request order"):
        adapter.replay_acquisition_group(
            sampled=sampled,
            packed=replace(packed, request_ids=tuple(reversed(packed.request_ids))),
        )


def test_publish_requires_typed_plan_bound_parity_and_is_atomic(tmp_path: Path) -> None:
    execution = _execute(adapter.plan_acquisition_group(image_id=1584, repetition_penalty=1.0, seed_group_id="qualification"))
    sampled = execution.group
    packed = adapter.PackedRawLogits(
        request_ids=tuple(item.identity.request_id for item in sampled.trajectories),
        token_indices=(0,) * 16,
        logits=torch.zeros((16, 151646), dtype=torch.float32),
    )
    replayed, receipt = adapter.replay_acquisition_group(sampled=sampled, packed=packed)
    output = tmp_path / "acquisition"
    with pytest.raises(ValueError, match="exact AcquisitionGroupParityReceipt"):
        adapter.publish_acquisition_group(output_root=output, execution=execution, replayed=replayed, replay_receipt=SimpleNamespace(**receipt.to_dict()))
    with pytest.raises(ValueError, match="replayed group"):
        adapter.publish_acquisition_group(output_root=output, execution=execution, replayed=sampled, replay_receipt=receipt)
    with pytest.raises(ValueError, match="plan SHA"):
        adapter.publish_acquisition_group(
            output_root=output,
            execution=replace(execution, plan_sha256="e" * 64),
            replayed=replayed,
            replay_receipt=receipt,
        )

    writes = 0

    def fail_second(path: Path, payload: bytes) -> None:
        nonlocal writes
        writes += 1
        if writes == 2:
            raise OSError("injected write failure")
        path.write_bytes(payload)

    with pytest.raises(OSError, match="injected"):
        adapter.publish_acquisition_group(output_root=output, execution=execution, replayed=replayed, replay_receipt=receipt, write_bytes=fail_second)
    assert not output.exists()
    published = adapter.publish_acquisition_group(output_root=output, execution=execution, replayed=replayed, replay_receipt=receipt)
    assert published == output
    assert (output / "publication-binding.json").is_file()
    assert (output / "native-receipts.json").is_file()
    restored = adapter.load_published_acquisition(output, plan=execution.plan)
    assert restored.execution.group.content_sha256 == execution.group.content_sha256
    (output / "native-receipts.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="native receipt"):
        adapter.load_published_acquisition(output, plan=execution.plan)
    (output / "native-receipts.json").unlink()
    with pytest.raises(FileNotFoundError, match="native-receipts"):
        adapter.load_published_acquisition(output, plan=execution.plan)
    with pytest.raises(FileExistsError, match="overwrite"):
        adapter.publish_acquisition_group(output_root=output, execution=execution, replayed=replayed, replay_receipt=receipt)


def test_load_requires_and_validates_replayed_and_parity_artifacts(tmp_path: Path) -> None:
    execution = _execute(adapter.plan_acquisition_group(image_id=1584, repetition_penalty=1.0, seed_group_id="qualification"))
    packed = adapter.PackedRawLogits(
        request_ids=tuple(item.identity.request_id for item in execution.group.trajectories),
        token_indices=(0,) * 16,
        logits=torch.zeros((16, 151646), dtype=torch.float32),
    )
    replayed, receipt = adapter.replay_acquisition_group(sampled=execution.group, packed=packed)
    output = tmp_path / "full-publication"
    adapter.publish_acquisition_group(output_root=output, execution=execution, replayed=replayed, replay_receipt=receipt)
    admitted = adapter.load_published_acquisition(output, plan=execution.plan)
    assert admitted.replayed_group.content_sha256 == replayed.content_sha256
    assert admitted.parity_receipt.content_sha256 == receipt.content_sha256

    replayed_path = output / "replayed-group.json"
    replayed_bytes = replayed_path.read_bytes()
    replayed_path.unlink()
    with pytest.raises(FileNotFoundError, match="replayed-group"):
        adapter.load_published_acquisition(output, plan=execution.plan)
    replayed_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="replayed acquisition group"):
        adapter.load_published_acquisition(output, plan=execution.plan)
    replayed_path.write_bytes(replayed_bytes)

    parity_path = output / "replay-receipt.json"
    parity_bytes = parity_path.read_bytes()
    parity_path.unlink()
    with pytest.raises(FileNotFoundError, match="replay-receipt"):
        adapter.load_published_acquisition(output, plan=execution.plan)
    parity_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="parity receipt"):
        adapter.load_published_acquisition(output, plan=execution.plan)
    parity_path.write_bytes(parity_bytes)


def test_publication_accepts_perfect_parity_with_identical_group_hashes(tmp_path: Path) -> None:
    from scripts.research.human13_rp_policy import validate_acquisition_group_replay

    execution = _execute(adapter.plan_acquisition_group(image_id=1584, repetition_penalty=1.0, seed_group_id="qualification"))
    receipt = validate_acquisition_group_replay(
        execution.group, execution.group, adapter.ReplayTolerance()
    )
    output = tmp_path / "perfect-parity"
    adapter.publish_acquisition_group(
        output_root=output,
        execution=execution,
        replayed=execution.group,
        replay_receipt=receipt,
    )
    assert (output / "publication-binding.json").is_file()


def test_dry_run_is_zero_action_without_runtime_import_or_artifact_write() -> None:
    plan = adapter.dry_run_plan()
    assert plan["status"] == "plan_only"
    assert plan["actions"] == {
        "model_imports": 0,
        "model_loads": 0,
        "engine_opens": 0,
        "gpu_allocations": 0,
        "artifact_writes": 0,
    }
