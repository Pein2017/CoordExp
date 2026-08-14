from __future__ import annotations

import math

import pytest
import torch


def test_plan_seals_all_four_seed_groups_and_batch_four_coverage() -> None:
    from scripts.research.collect_human13_rp_crossover import (
        MATRIX_SEED_GROUPS,
        QUALIFICATION_SEEDS,
        plan_acquisition_group,
        plan_panel_acquisition,
    )

    plan = plan_acquisition_group(
        image_id=1584, repetition_penalty=1.10, seed_group_id="matrix_b"
    )
    assert QUALIFICATION_SEEDS == tuple(range(30001, 30017))
    assert MATRIX_SEED_GROUPS == {
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
    requests = [item for batch in plan.batches for item in batch.requests]
    assert len(requests) == len(set(item.request_id for item in requests)) == 16
    assert all(
        item.sampling
        == {
            "n": 1,
            "temperature": 0.4,
            "top_p": 1.0,
            "top_k": None,
            "repetition_penalty": 1.10,
            "max_new_tokens": 512,
            "stop_token_ids": (151645,),
            "ignore_eos": False,
        }
        for item in requests
    )
    assert len(plan_panel_acquisition(repetition_penalty=1.0, seed_group_id="qualification")) == 13


def test_native_evidence_rejects_incomplete_or_misordered_score_history() -> None:
    from scripts.research.collect_human13_rp_crossover import (
        native_trajectory_evidence,
        plan_acquisition_group,
    )

    request = plan_acquisition_group(
        image_id=1584, repetition_penalty=1.0, seed_group_id="qualification"
    ).batches[0].requests[0]
    common = {
        "source_sha256": "a" * 64,
        "manifest_sha256": "b" * 64,
        "model_id": "model",
        "tokenizer_id": "tokenizer",
        "processor_id": "processor",
        "sampler_backend_id": "vllm:test",
        "evidence_origin": "fresh_native",
        "processor_order": ("repetition_penalty", "temperature", "log_softmax"),
        "sampling": request.sampling,
        "prompt_token_ids": (1, 2),
        "generated_token_ids": (7, 151645),
        "processed_logprobs": (-1.0, -2.0),
        "terminal_kind": "natural_stop",
    }
    trajectory = native_trajectory_evidence(request=request, native=common)
    assert [item.history_token_ids for item in trajectory.generated_tokens] == [
        (1, 2),
        (1, 2, 7),
    ]
    with pytest.raises(ValueError, match="complete chosen-token log probabilities"):
        native_trajectory_evidence(
            request=request, native={**common, "processed_logprobs": (-1.0,)}
        )
    with pytest.raises(ValueError, match="natural stop"):
        native_trajectory_evidence(
            request=request,
            native={**common, "generated_token_ids": (7, 8), "terminal_kind": "natural_stop"},
        )


def test_replay_accepts_only_packed_tensor_rows_and_emits_one_group_receipt() -> None:
    from scripts.research.collect_human13_rp_crossover import (
        PackedRawLogits,
        acquisition_group_from_native,
        plan_acquisition_group,
        replay_acquisition_group,
    )

    plan = plan_acquisition_group(
        image_id=1584, repetition_penalty=1.0, seed_group_id="qualification"
    )
    native = {
        request.request_id: {
            "source_sha256": "a" * 64,
            "manifest_sha256": "b" * 64,
            "model_id": "model",
            "tokenizer_id": "tokenizer",
            "processor_id": "processor",
            "sampler_backend_id": "vllm:test",
            "evidence_origin": "fresh_native",
            "processor_order": ("repetition_penalty", "temperature", "log_softmax"),
            "sampling": request.sampling,
            "prompt_token_ids": (1, 2),
            "generated_token_ids": (151645,),
            "processed_logprobs": (-math.log(151646),),
            "terminal_kind": "natural_stop",
        }
        for batch in plan.batches
        for request in batch.requests
    }
    sampled = acquisition_group_from_native(plan=plan, native_by_request_id=native)
    packed = PackedRawLogits(
        request_ids=tuple(item.identity.request_id for item in sampled.trajectories),
        token_indices=(0,) * 16,
        logits=torch.zeros((16, 151646), dtype=torch.float32),
    )
    replayed, receipt = replay_acquisition_group(sampled=sampled, packed=packed)
    assert receipt.admitted is True
    assert receipt.token_count == 16
    assert replayed.content_sha256 != sampled.content_sha256
    with pytest.raises(ValueError, match="two-dimensional tensor"):
        PackedRawLogits(
            request_ids=packed.request_ids,
            token_indices=packed.token_indices,
            logits=tuple(tuple(0.0 for _ in range(16)) for _ in range(16)),  # type: ignore[arg-type]
        )


def test_dry_run_is_zero_action_without_runtime_import_or_artifact_write() -> None:
    from scripts.research.collect_human13_rp_crossover import dry_run_plan

    plan = dry_run_plan()
    assert plan["status"] == "plan_only"
    assert plan["actions"] == {
        "model_imports": 0,
        "model_loads": 0,
        "engine_opens": 0,
        "gpu_allocations": 0,
        "artifact_writes": 0,
    }
    assert plan["image_count"] == 13
    assert plan["physical_batch_count_per_group"] == 52
    assert plan["request_count_per_group"] == 208
