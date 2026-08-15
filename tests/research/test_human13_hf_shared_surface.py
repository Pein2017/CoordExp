"""CPU-only contract tests for the Human-13 all-HF shared surface."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from math import inf, nan
from typing import Callable

import pytest

from scripts.research.human13_hf_shared_surface import (
    GradientReplayGroup,
    HFActiveBatchStep,
    HFReplayCausalGather,
    HFSharedSurfaceCloseReceipt,
    HFSharedSurfaceDryRunReceipt,
    HFSharedSurfaceIdentity,
    HFSharedSurfaceParityReceipt,
    HFSharedSurfacePolicy,
    HFSharedSurfaceResourceEstimate,
    SampledHFGroup,
    SampledHFRequest,
    SampledHFToken,
    SharedSurfaceContractError,
    admit_gradient_replay,
    admit_sampled_group,
    admit_shared_surface_close,
    causal_history_sha256,
    dry_run_image1584_k16,
    estimate_image1584_k16_resources,
    plan_image1584_k16,
    repetition_penalty_then_temperature,
    require_finite_optimizer_delta,
    require_positive_objective_denominator,
    require_private_audit_reference,
    require_rollback_reference,
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


def valid_request(seed: int, *, token_count: int = 2, **changes: object) -> SampledHFRequest:
    request_id = f"image-1584-seed-{seed}"
    prompt = _digest(f"prompt:{seed}")
    token_ids = tuple(range(10, 10 + token_count))
    tokens = tuple(
        SampledHFToken(
            request_id=request_id,
            token_index=index,
            history_sha256=causal_history_sha256(prompt, token_ids[:index]),
            chosen_token_id=token_id,
            raw_chosen_logit=1.25 + index,
            processed_logp=-0.125 * (index + 1),
            causal_logit_index=20 + index,
        )
        for index, token_id in enumerate(token_ids)
    )
    values: dict[str, object] = {
        "request_id": request_id,
        "image_id": 1584,
        "seed": seed,
        "prompt_history_sha256": prompt,
        "tokens": tokens,
        "processor_order": ("repetition_penalty", "temperature", "top_p"),
        "stop_reason": "im_end",
        "use_cache": False,
    }
    values.update(changes)
    return SampledHFRequest(**values)  # type: ignore[arg-type]


def valid_requests(**changes: object) -> tuple[SampledHFRequest, ...]:
    return tuple(
        replace(valid_request(seed), **changes) for seed in range(35001, 35005)
    )


def valid_active_batch_steps(
    requests: tuple[SampledHFRequest, ...],
) -> tuple[HFActiveBatchStep, ...]:
    return tuple(
        HFActiveBatchStep(
            token_index=index,
            active_request_ids=tuple(request.request_id for request in requests),
            active_history_sha256s=tuple(
                request.tokens[index].history_sha256 for request in requests
            ),
            batch_shape=(len(requests), 32 + index),
            rng_before_sha256=_digest(f"rng:{index}:before"),
            rng_after_sha256=_digest(f"rng:{index}:after"),
        )
        for index in range(2)
    )


def valid_group_kwargs(**changes: object) -> dict[str, object]:
    requests = valid_requests()
    values: dict[str, object] = {
        "plan": plan_image1584_k16(),
        "group_index": 0,
        "expected_identity": valid_identity(),
        "identity": valid_identity(),
        "policy": valid_policy(),
        "requests": requests,
        "active_batch_steps": valid_active_batch_steps(requests),
    }
    values.update(changes)
    return values


def valid_replayed(group: SampledHFGroup, delta: float = 0.0) -> tuple[SampledHFToken, ...]:
    return tuple(
        replace(token, processed_logp=token.processed_logp + delta)
        for request in group.requests
        for token in request.tokens
    )


def valid_gathers(group: SampledHFGroup) -> tuple[HFReplayCausalGather, ...]:
    return tuple(
        HFReplayCausalGather(
            request_id=token.request_id,
            token_index=token.token_index,
            history_sha256=token.history_sha256,
            chosen_token_id=token.chosen_token_id,
            causal_logit_index=token.causal_logit_index,
        )
        for request in group.requests
        for token in request.tokens
    )


# The active Task 1.1 matrix names the actual value-contract owner and executes
# a minimal invalid input for every invariant, including later-wave handoff guards.
FAILURE_MODE_MATRIX: tuple[tuple[str, str, Callable[[], None]], ...] = (
    ("surface identity", "admit_sampled_group", lambda: admit_sampled_group(**valid_group_kwargs(identity=valid_identity(dtype="float32")))),
    ("request/history lineage", "admit_sampled_group", lambda: admit_sampled_group(**valid_group_kwargs(requests=valid_requests(request_id="")))),
    ("processor order", "admit_sampled_group", lambda: admit_sampled_group(**valid_group_kwargs(requests=valid_requests(processor_order=("temperature", "repetition_penalty", "top_p"))))),
    ("parity", "admit_gradient_replay", lambda: admit_gradient_replay(sampled_group=admit_sampled_group(**valid_group_kwargs()), replay_identity=valid_identity(), replayed_tokens=valid_replayed(admit_sampled_group(**valid_group_kwargs()), 0.03), replay_processor_order=("repetition_penalty", "temperature", "top_p"), causal_gathers=valid_gathers(admit_sampled_group(**valid_group_kwargs())))),
    ("objective denominator", "Task 3 objective composer", lambda: require_positive_objective_denominator(0)),
    ("optimizer delta", "Task 3 proposal apply", lambda: require_finite_optimizer_delta(nan)),
    ("private audit", "Task 4 dual-RP audit", lambda: require_private_audit_reference("not-a-digest")),
    ("rollback", "Task 3 transaction finalizer", lambda: require_rollback_reference("not-a-digest")),
)


@pytest.mark.parametrize(("invariant", "owner", "counterexample"), FAILURE_MODE_MATRIX)
def test_failure_mode_matrix_rejects_minimal_counterexample(
    invariant: str, owner: str, counterexample: Callable[[], None]
) -> None:
    """Catches a missing executable guard named by the authoritative Task 1.1 matrix."""
    assert owner
    with pytest.raises(SharedSurfaceContractError):
        counterexample()


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
    ],
)
def test_shared_surface_admission_rejects_failure_mode(
    mutation: Callable[[], dict[str, object]], message: str
) -> None:
    with pytest.raises(SharedSurfaceContractError, match=message):
        admit_sampled_group(**(valid_group_kwargs() | mutation()))  # type: ignore[arg-type]


def test_plan_is_exactly_image1584_k16_in_four_ordered_seed_groups() -> None:
    plan = plan_image1584_k16()
    assert plan.image_id == 1584
    assert plan.seed_groups == ((35001, 35002, 35003, 35004), (35005, 35006, 35007, 35008), (35009, 35010, 35011, 35012), (35013, 35014, 35015, 35016))
    assert plan.policy == valid_policy()


def test_admission_rejects_wrong_seed_coverage_and_group_order() -> None:
    with pytest.raises(SharedSurfaceContractError, match="seed coverage"):
        admit_sampled_group(**valid_group_kwargs(requests=tuple(replace(request, seed=request.seed + 1) for request in valid_requests())))
    with pytest.raises(SharedSurfaceContractError, match="seed coverage"):
        admit_sampled_group(**valid_group_kwargs(group_index=1))


def test_lineage_requires_observed_request_ids_causal_histories_ids_and_cap() -> None:
    request = valid_request(35001)
    with pytest.raises(SharedSurfaceContractError, match="request identity"):
        admit_sampled_group(**valid_group_kwargs(requests=(replace(request, request_id=""), *valid_requests()[1:])))
    with pytest.raises(SharedSurfaceContractError, match="history/token lineage"):
        bad = replace(request, tokens=(replace(request.tokens[0], history_sha256=_digest("forged")), *request.tokens[1:]))
        admit_sampled_group(**valid_group_kwargs(requests=(bad, *valid_requests()[1:])))
    with pytest.raises(SharedSurfaceContractError, match="chosen token"):
        bad = replace(request, tokens=(replace(request.tokens[0], chosen_token_id=-1), *request.tokens[1:]))
        admit_sampled_group(**valid_group_kwargs(requests=(bad, *valid_requests()[1:])))
    assert len(valid_request(35001, token_count=512).tokens) == 512
    with pytest.raises(SharedSurfaceContractError, match="token cap"):
        valid_request(35001, token_count=513)


def test_acquisition_evidence_binds_active_history_rng_stop_raw_logit_and_shape() -> None:
    group = admit_sampled_group(**valid_group_kwargs())
    assert group.active_batch_steps[0].batch_shape == (4, 32)
    assert group.requests[0].tokens[0].raw_chosen_logit == pytest.approx(1.25)
    assert group.requests[0].stop_reason == "im_end"
    bad_steps = (*group.active_batch_steps[:-1], replace(group.active_batch_steps[-1], active_history_sha256s=(_digest("wrong"),) * 4))
    with pytest.raises(SharedSurfaceContractError, match="active-batch history"):
        admit_sampled_group(**valid_group_kwargs(active_batch_steps=bad_steps))


def test_admitted_values_reject_direct_marker_injection_and_replacement() -> None:
    group = admit_sampled_group(**valid_group_kwargs())
    with pytest.raises(TypeError):
        SampledHFGroup(**group.__dict__, _admitted=True)  # type: ignore[arg-type]
    with pytest.raises(SharedSurfaceContractError, match="not admitted"):
        replace(group, group_index=3).to_dict()
    with pytest.raises(SharedSurfaceContractError, match="not admitted"):
        HFSharedSurfaceParityReceipt(_digest("s"), _digest("r"), (0.0,), 0.0, 0.0).to_dict()


def test_every_admitted_scientific_copy_loses_its_seal() -> None:
    """Catches dataclasses.replace preserving an admission claim after mutation."""
    group = admit_sampled_group(**valid_group_kwargs())
    replay = admit_gradient_replay(sampled_group=group, replay_identity=group.identity, replayed_tokens=valid_replayed(group), replay_processor_order=("repetition_penalty", "temperature", "top_p"), causal_gathers=valid_gathers(group))
    close = admit_shared_surface_close(identity=group.identity, replay_group_sha256=replay.content_sha256, close_reason="completed")
    estimate = estimate_image1584_k16_resources(output_roots=("/tmp/proposal", "/tmp/audit"))
    dry_run = dry_run_image1584_k16(output_roots=("/tmp/proposal", "/tmp/audit"))
    copies = (
        replace(group, group_index=3),
        replace(replay.parity, max_abs_error=0.0),
        replace(replay, replay_processor_order=("temperature", "repetition_penalty", "top_p")),
        replace(close, close_reason="failed"),
        replace(estimate, output_roots=("/tmp/proposal", "/tmp/other")),
        replace(dry_run, output_roots=("/tmp/proposal", "/tmp/other")),
    )
    for copied in copies:
        with pytest.raises(SharedSurfaceContractError, match="not admitted"):
            copied.to_dict()


def test_admitted_group_round_trips_canonically_and_rejects_forged_copy() -> None:
    group = admit_sampled_group(**valid_group_kwargs())
    assert type(group).from_dict(group.to_dict()) == group
    assert type(group).from_dict(group.to_dict()).content_sha256 == group.content_sha256
    with pytest.raises(SharedSurfaceContractError, match="content SHA-256"):
        type(group).from_dict(group.to_dict() | {"content_sha256": _digest("forged")})


def test_gradient_replay_requires_exact_lineage_processor_gather_and_parity_distribution() -> None:
    sampled = admit_sampled_group(**valid_group_kwargs())
    replayed = valid_replayed(sampled, 0.001)
    admitted = admit_gradient_replay(sampled_group=sampled, replay_identity=sampled.identity, replayed_tokens=replayed, replay_processor_order=("repetition_penalty", "temperature", "top_p"), causal_gathers=valid_gathers(sampled))
    assert isinstance(admitted, GradientReplayGroup)
    assert admitted.parity.absolute_errors == pytest.approx((0.001,) * 8)
    assert admitted.parity.max_abs_error == pytest.approx(0.001)
    with pytest.raises(SharedSurfaceContractError, match="processor order"):
        admit_gradient_replay(sampled_group=sampled, replay_identity=sampled.identity, replayed_tokens=replayed, replay_processor_order=("temperature", "repetition_penalty", "top_p"), causal_gathers=valid_gathers(sampled))
    with pytest.raises(SharedSurfaceContractError, match="causal gather"):
        admit_gradient_replay(sampled_group=sampled, replay_identity=sampled.identity, replayed_tokens=replayed, replay_processor_order=("repetition_penalty", "temperature", "top_p"), causal_gathers=(replace(valid_gathers(sampled)[0], chosen_token_id=777), *valid_gathers(sampled)[1:]))
    with pytest.raises(SharedSurfaceContractError, match="history/token lineage"):
        admit_gradient_replay(sampled_group=sampled, replay_identity=sampled.identity, replayed_tokens=(replace(replayed[0], chosen_token_id=777), *replayed[1:]), replay_processor_order=("repetition_penalty", "temperature", "top_p"), causal_gathers=valid_gathers(sampled))


def test_gradient_replay_rejects_nonfinite_surface_and_both_parity_boundaries() -> None:
    sampled = admit_sampled_group(**valid_group_kwargs())
    replayed = valid_replayed(sampled)
    with pytest.raises(SharedSurfaceContractError, match="finite"):
        admit_gradient_replay(sampled_group=sampled, replay_identity=sampled.identity, replayed_tokens=(replace(replayed[0], processed_logp=inf), *replayed[1:]), replay_processor_order=("repetition_penalty", "temperature", "top_p"), causal_gathers=valid_gathers(sampled))
    with pytest.raises(SharedSurfaceContractError, match="shared surface identity"):
        admit_gradient_replay(sampled_group=sampled, replay_identity=replace(sampled.identity, parameter_state_sha256=_digest("reloaded")), replayed_tokens=replayed, replay_processor_order=("repetition_penalty", "temperature", "top_p"), causal_gathers=valid_gathers(sampled))
    assert admit_gradient_replay(sampled_group=sampled, replay_identity=sampled.identity, replayed_tokens=valid_replayed(sampled, 0.002), replay_processor_order=("repetition_penalty", "temperature", "top_p"), causal_gathers=valid_gathers(sampled)).parity.mean_abs_error == pytest.approx(0.002)
    with pytest.raises(SharedSurfaceContractError, match="parity thresholds"):
        admit_gradient_replay(sampled_group=sampled, replay_identity=sampled.identity, replayed_tokens=valid_replayed(sampled, 0.003), replay_processor_order=("repetition_penalty", "temperature", "top_p"), causal_gathers=valid_gathers(sampled))


def test_replay_and_close_receipts_round_trip_and_reject_forged_hashes() -> None:
    sampled = admit_sampled_group(**valid_group_kwargs())
    replay = admit_gradient_replay(sampled_group=sampled, replay_identity=sampled.identity, replayed_tokens=valid_replayed(sampled), replay_processor_order=("repetition_penalty", "temperature", "top_p"), causal_gathers=valid_gathers(sampled))
    assert GradientReplayGroup.from_dict(replay.to_dict()) == replay
    close = admit_shared_surface_close(identity=sampled.identity, replay_group_sha256=replay.content_sha256, close_reason="completed")
    assert isinstance(close, HFSharedSurfaceCloseReceipt)
    assert HFSharedSurfaceCloseReceipt.from_dict(close.to_dict()) == close
    with pytest.raises(SharedSurfaceContractError, match="content SHA-256"):
        HFSharedSurfaceCloseReceipt.from_dict(close.to_dict() | {"content_sha256": _digest("forged")})


def test_sign_aware_repetition_penalty_precedes_temperature() -> None:
    assert repetition_penalty_then_temperature(2.0, token_was_seen=True, repetition_penalty=2.0, temperature=0.4) == pytest.approx(2.5)
    assert repetition_penalty_then_temperature(-2.0, token_was_seen=True, repetition_penalty=2.0, temperature=0.4) == pytest.approx(-10.0)


def test_resource_and_dry_run_are_admitted_reloadable_and_zero_action() -> None:
    roots = ("/tmp/private-proposal", "/tmp/private-audit")
    estimate = estimate_image1584_k16_resources(output_roots=roots)
    assert isinstance(estimate, HFSharedSurfaceResourceEstimate)
    assert estimate.image_prompt_forwards == 2048
    assert HFSharedSurfaceResourceEstimate.from_dict(estimate.to_dict()) == estimate
    dry_run = dry_run_image1584_k16(output_roots=roots)
    assert isinstance(dry_run, HFSharedSurfaceDryRunReceipt)
    assert HFSharedSurfaceDryRunReceipt.from_dict(dry_run.to_dict()) == dry_run
    with pytest.raises(SharedSurfaceContractError, match="content SHA-256"):
        HFSharedSurfaceDryRunReceipt.from_dict(dry_run.to_dict() | {"content_sha256": _digest("forged")})
    bad_action = dry_run.to_dict() | {"model_gpu_actions": 1}
    bad_action["content_sha256"] = _digest("recomputed-but-bad")
    with pytest.raises(SharedSurfaceContractError, match="zero model/GPU"):
        HFSharedSurfaceDryRunReceipt.from_dict(bad_action)
    mismatched = dry_run.to_dict() | {"output_roots": ["/tmp/private-proposal", "/tmp/other"]}
    mismatched["content_sha256"] = _digest("recomputed-but-bad")
    with pytest.raises(SharedSurfaceContractError, match="output roots"):
        HFSharedSurfaceDryRunReceipt.from_dict(mismatched)
