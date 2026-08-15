"""CPU-only contract tests for the Human-13 all-HF shared surface."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from math import inf, nan
from typing import Callable

import pytest

from scripts.research.human13_hf_shared_surface import (
    GradientReplayGroup,
    HFSharedSurfaceIdentity,
    HFSharedSurfaceParityReceipt,
    HFSharedSurfacePolicy,
    HFSharedSurfaceResourceEstimate,
    SampledHFRequest,
    SampledHFToken,
    SharedSurfaceContractError,
    admit_gradient_replay,
    admit_sampled_group,
    dry_run_image1584_k16,
    estimate_image1584_k16_resources,
    plan_image1584_k16,
    repetition_penalty_then_temperature,
)


def _digest(label: str) -> str:
    return sha256(label.encode("utf-8")).hexdigest()


def valid_identity(**changes: object) -> HFSharedSurfaceIdentity:
    values: dict[str, object] = {
        "checkpoint_payload_sha256": _digest("checkpoint"),
        "model_object_id": 31337,
        "parameter_state_sha256": _digest("parameters"),
        "adapter_sha256": _digest("adapter"),
        "embedding_delta_sha256": _digest("embedding-delta"),
        "dtype": "bfloat16",
        "attention_backend": "flash_attention_2",
        "model_mode": "eval",
        "tokenizer_sha256": _digest("tokenizer"),
        "prompt_sha256": _digest("prompt"),
        "image_sha256": _digest("image-1584"),
        "use_cache": False,
    }
    values.update(changes)
    return HFSharedSurfaceIdentity(**values)  # type: ignore[arg-type]


def valid_policy(**changes: object) -> HFSharedSurfacePolicy:
    values: dict[str, object] = {
        "repetition_penalty": 1.0,
        "repetition_penalty_before_temperature": True,
        "prompt_inclusive_history": True,
        "temperature": 0.4,
        "top_p": 1.0,
        "top_k": None,
        "max_new_tokens": 512,
        "stop_token": "<|im_end|>",
        "use_cache": False,
    }
    values.update(changes)
    return HFSharedSurfacePolicy(**values)  # type: ignore[arg-type]


def valid_requests(**changes: object) -> tuple[SampledHFRequest, ...]:
    tokens = (
        SampledHFToken(
            token_index=0,
            history_sha256=_digest("history:request-0:0"),
            chosen_token_id=10,
            processed_logp=-0.125,
        ),
        SampledHFToken(
            token_index=1,
            history_sha256=_digest("history:request-0:1"),
            chosen_token_id=11,
            processed_logp=-0.250,
        ),
    )
    requests = tuple(
        SampledHFRequest(
            request_id=f"image-1584-seed-{seed}",
            image_id=1584,
            seed=seed,
            prompt_history_sha256=_digest(f"prompt:{seed}"),
            tokens=tokens if index == 0 else replace_tokens(tokens, index),
            processor_order=("repetition_penalty", "temperature", "top_p"),
            use_cache=False,
        )
        for index, seed in enumerate(range(35001, 35005))
    )
    return tuple(replace(request, **changes) for request in requests)


def replace_tokens(
    tokens: tuple[SampledHFToken, ...], request_index: int
) -> tuple[SampledHFToken, ...]:
    return tuple(
        replace(
            token,
            history_sha256=_digest(f"history:request-{request_index}:{token.token_index}"),
        )
        for token in tokens
    )


def valid_group_kwargs(**changes: object) -> dict[str, object]:
    values: dict[str, object] = {
        "plan": plan_image1584_k16(),
        "group_index": 0,
        "expected_identity": valid_identity(),
        "identity": valid_identity(),
        "policy": valid_policy(),
        "requests": valid_requests(),
    }
    values.update(changes)
    return values


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda: {"identity": valid_identity(model_object_id=31338)}, "shared surface identity"),
        (lambda: {"identity": valid_identity(parameter_state_sha256=_digest("changed"))}, "shared surface identity"),
        (lambda: {"identity": valid_identity(adapter_sha256=_digest("changed"))}, "shared surface identity"),
        (lambda: {"identity": valid_identity(embedding_delta_sha256=_digest("changed"))}, "shared surface identity"),
        (lambda: {"identity": valid_identity(dtype="float32")}, "shared surface identity"),
        (lambda: {"identity": valid_identity(attention_backend="sdpa")}, "shared surface identity"),
        (lambda: {"identity": valid_identity(model_mode="train")}, "shared surface identity"),
        (lambda: {"identity": valid_identity(tokenizer_sha256=_digest("changed"))}, "shared surface identity"),
        (lambda: {"identity": valid_identity(prompt_sha256=_digest("changed"))}, "shared surface identity"),
        (lambda: {"identity": valid_identity(image_sha256=_digest("changed"))}, "shared surface identity"),
        (lambda: {"policy": valid_policy(use_cache=True)}, "cache is forbidden"),
        (lambda: {"requests": valid_requests(use_cache=True)}, "cache is forbidden"),
        (lambda: {"requests": valid_requests(processor_order=("temperature", "repetition_penalty", "top_p"))}, "processor order"),
        (lambda: {"requests": valid_requests(tokens=())}, "history/token lineage"),
        (lambda: {"requests": valid_requests(tokens=(SampledHFToken(1, _digest("h"), 99, -0.1),))}, "history/token lineage"),
        (lambda: {"requests": valid_requests(tokens=(SampledHFToken(0, _digest("h"), 10, nan),))}, "finite"),
    ],
    ids=[
        "model-object",
        "parameters",
        "adapter",
        "embedding-delta",
        "dtype",
        "attention",
        "mode",
        "tokenizer",
        "prompt",
        "image",
        "policy-cache",
        "request-cache",
        "processor-order",
        "missing-history",
        "forged-token",
        "nonfinite-logp",
    ],
)
def test_shared_surface_admission_rejects_failure_mode(
    mutation: Callable[[], dict[str, object]], message: str
) -> None:
    """Catches any surface, lineage, cache, processor, or finiteness bypass."""
    with pytest.raises(SharedSurfaceContractError, match=message):
        admit_sampled_group(**(valid_group_kwargs() | mutation()))  # type: ignore[arg-type]


def test_plan_is_exactly_image1584_k16_in_four_ordered_seed_groups() -> None:
    """Catches accidental broadening, reordering, or regrouping of the frozen plan."""
    plan = plan_image1584_k16()
    assert plan.image_id == 1584
    assert plan.seed_groups == (
        (35001, 35002, 35003, 35004),
        (35005, 35006, 35007, 35008),
        (35009, 35010, 35011, 35012),
        (35013, 35014, 35015, 35016),
    )
    assert plan.policy == valid_policy()


def test_admission_rejects_wrong_seed_coverage_and_group_order() -> None:
    """Catches a sampling group that cannot be part of the exact K16 plan."""
    with pytest.raises(SharedSurfaceContractError, match="seed coverage"):
        admit_sampled_group(
            **valid_group_kwargs(
                requests=tuple(
                    replace(request, seed=request.seed + 1)
                    for request in valid_requests()
                )
            )
        )
    with pytest.raises(SharedSurfaceContractError, match="seed coverage"):
        admit_sampled_group(
            **valid_group_kwargs(group_index=1, requests=valid_requests())
        )


def test_admitted_group_round_trips_canonically_and_rejects_forged_copy() -> None:
    """Catches lossy serialization and unadmitted reconstructed scientific values."""
    group = admit_sampled_group(**valid_group_kwargs())
    assert type(group).from_dict(group.to_dict()) == group
    assert type(group).from_dict(group.to_dict()).content_sha256 == group.content_sha256
    forged = group.to_dict() | {"content_sha256": _digest("forged")}
    with pytest.raises(SharedSurfaceContractError, match="content SHA-256"):
        type(group).from_dict(forged)
    malformed = group.to_dict()
    malformed["requests"][0]["tokens"].append("not-a-token")  # type: ignore[index]
    with pytest.raises(SharedSurfaceContractError, match="strict value objects"):
        type(group).from_dict(malformed)


def test_gradient_replay_requires_exact_token_lineage_and_parity_thresholds() -> None:
    """Catches parity admission that ignores token identity or accepts excess error."""
    sampled = admit_sampled_group(**valid_group_kwargs())
    replayed = tuple(
        replace(token, processed_logp=token.processed_logp + 0.001)
        for request in sampled.requests
        for token in request.tokens
    )
    admitted = admit_gradient_replay(
        sampled_group=sampled,
        replay_identity=sampled.identity,
        replayed_tokens=replayed,
    )
    assert isinstance(admitted, GradientReplayGroup)
    assert admitted.parity.max_abs_error == pytest.approx(0.001)
    assert admitted.parity.mean_abs_error == pytest.approx(0.001)

    with pytest.raises(SharedSurfaceContractError, match="history/token lineage"):
        admit_gradient_replay(
            sampled_group=sampled,
            replay_identity=sampled.identity,
            replayed_tokens=(replace(replayed[0], chosen_token_id=777), *replayed[1:]),
        )
    with pytest.raises(SharedSurfaceContractError, match="parity thresholds"):
        admit_gradient_replay(
            sampled_group=sampled,
            replay_identity=sampled.identity,
            replayed_tokens=(replace(replayed[0], processed_logp=0.0), *replayed[1:]),
        )


def test_gradient_replay_rejects_nonfinite_values_and_forged_receipt() -> None:
    """Catches numeric poisoning and receipt reconstruction that bypasses admission."""
    sampled = admit_sampled_group(**valid_group_kwargs())
    replayed = tuple(
        replace(token, processed_logp=token.processed_logp)
        for request in sampled.requests
        for token in request.tokens
    )
    with pytest.raises(SharedSurfaceContractError, match="finite"):
        admit_gradient_replay(
            sampled_group=sampled,
            replay_identity=sampled.identity,
            replayed_tokens=(replace(replayed[0], processed_logp=inf), *replayed[1:]),
        )
    replay = admit_gradient_replay(
        sampled_group=sampled,
        replay_identity=sampled.identity,
        replayed_tokens=replayed,
    )
    forged = replay.parity.to_dict() | {"content_sha256": _digest("forged")}
    with pytest.raises(SharedSurfaceContractError, match="content SHA-256"):
        HFSharedSurfaceParityReceipt.from_dict(forged)


def test_gradient_replay_rejects_reloaded_surface_and_mean_only_parity_failure() -> None:
    """Catches replay on a changed model surface or a passing max but failing mean."""
    sampled = admit_sampled_group(**valid_group_kwargs())
    replayed = tuple(
        replace(token, processed_logp=token.processed_logp + 0.003)
        for request in sampled.requests
        for token in request.tokens
    )
    with pytest.raises(SharedSurfaceContractError, match="shared surface identity"):
        admit_gradient_replay(
            sampled_group=sampled,
            replay_identity=replace(sampled.identity, parameter_state_sha256=_digest("reloaded")),
            replayed_tokens=replayed,
        )
    with pytest.raises(SharedSurfaceContractError, match="parity thresholds"):
        admit_gradient_replay(
            sampled_group=sampled,
            replay_identity=sampled.identity,
            replayed_tokens=replayed,
        )


def test_sign_aware_repetition_penalty_precedes_temperature() -> None:
    """Catches reversed processor order and the wrong branch for negative logits."""
    assert repetition_penalty_then_temperature(
        2.0, token_was_seen=True, repetition_penalty=2.0, temperature=0.4
    ) == pytest.approx(2.5)
    assert repetition_penalty_then_temperature(
        -2.0, token_was_seen=True, repetition_penalty=2.0, temperature=0.4
    ) == pytest.approx(-10.0)
    assert repetition_penalty_then_temperature(
        2.0, token_was_seen=False, repetition_penalty=2.0, temperature=0.4
    ) == pytest.approx(5.0)


def test_pure_resource_estimate_and_dry_run_bound_the_frozen_vertical() -> None:
    """Catches an unbounded dry run or an estimate that claims model/GPU actions."""
    roots = ("/tmp/private-proposal", "/tmp/private-audit")
    estimate = estimate_image1584_k16_resources(output_roots=roots)
    assert isinstance(estimate, HFSharedSurfaceResourceEstimate)
    assert estimate.image_prompt_forwards == 2048
    assert estimate.replay_forwards == 4
    assert estimate.backward_count == 1
    assert estimate.expected_token_cap == 8192
    assert estimate.required_gpu_roles == (
        "gpu0:shared-bf16-fa2-training",
        "gpu1:fp32-sdpa-audit",
    )
    dry_run = dry_run_image1584_k16(output_roots=roots)
    assert dry_run.resource_estimate == estimate
    assert dry_run.model_gpu_actions == 0
    assert dry_run.output_roots == roots
