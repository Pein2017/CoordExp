from __future__ import annotations

import math

import pytest
import torch

from scripts.research.human13_k_trajectory_contracts import (
    AcquisitionGroup,
    ArtifactIdentity,
    CompleteTrajectoryEvidence,
    GeneratedTokenEvidence,
    PolicyContract,
    ReplayTolerance,
)
from scripts.research.human13_rp_policy import (
    PolicyReplayError,
    processed_policy_logprobs,
    validate_policy_replay,
)


def _identity(
    *, generated: tuple[int, ...] = (), request_id: str = "request:seed-11:image-7"
) -> ArtifactIdentity:
    return ArtifactIdentity(
        source_sha256="a" * 64,
        manifest_sha256="b" * 64,
        request_id=request_id,
        model_id="Qwen/Qwen3-VL-8B-Instruct",
        tokenizer_id="qwen3-vl-tokenizer:sha256:c",
        processor_id="qwen3-vl-processor:sha256:d",
        prompt_token_ids=(101, 102, 103),
        generated_token_ids=generated,
    )


def _contract(
    *, repetition_penalty: float = 1.10, temperature: float = 0.4
) -> PolicyContract:
    return PolicyContract(
        identity=_identity(),
        repetition_penalty=repetition_penalty,
        temperature=temperature,
        natural_stop_token_id=2,
        max_new_tokens=4,
    )


def _trajectory(
    *,
    generated: tuple[int, ...] = (4, 2),
    logprobs: tuple[float, ...] = (-0.25, -0.75),
    contract: PolicyContract | None = None,
    terminal_kind: str = "natural_stop",
) -> CompleteTrajectoryEvidence:
    contract = contract or _contract()
    identity = ArtifactIdentity(
        source_sha256=contract.identity.source_sha256,
        manifest_sha256=contract.identity.manifest_sha256,
        request_id=contract.identity.request_id,
        model_id=contract.identity.model_id,
        tokenizer_id=contract.identity.tokenizer_id,
        processor_id=contract.identity.processor_id,
        prompt_token_ids=contract.identity.prompt_token_ids,
        generated_token_ids=generated,
    )
    tokens = tuple(
        GeneratedTokenEvidence(
            identity=identity,
            policy_contract_sha256=contract.content_sha256,
            token_index=index,
            history_token_ids=(*identity.prompt_token_ids, *generated[:index]),
            chosen_token_id=token_id,
            processed_logprob=logprobs[index],
        )
        for index, token_id in enumerate(generated)
    )
    return CompleteTrajectoryEvidence(
        identity=identity,
        policy_contract=contract,
        generated_tokens=tokens,
        terminal_kind=terminal_kind,
    )


def test_rp110_is_sign_aware_on_unique_complete_history_tokens() -> None:
    # Catches changing the negative-logit branch from multiplication to division.
    actual = processed_policy_logprobs(
        torch.tensor([4.0, -2.0, 1.0, -3.0]), (0, 1, 1), _contract()
    )
    expected = torch.log_softmax(
        torch.tensor([4.0 / 1.10, -2.0 * 1.10, 1.0, -3.0]) / 0.4, dim=-1
    )
    assert actual.dtype == torch.float32
    assert torch.allclose(actual, expected, atol=1e-7, rtol=0)


def test_rp100_is_identity_before_temperature_and_full_support_normalization() -> None:
    # Catches adding an RP branch when the contract declares RP 1.0.
    raw = torch.tensor([1.0, 0.0, -1.0], dtype=torch.float64)
    actual = processed_policy_logprobs(raw, (0, 2), _contract(repetition_penalty=1.0))
    expected = torch.log_softmax(torch.tensor([2.5, 0.0, -2.5]), dim=-1)
    assert torch.allclose(actual, expected, atol=1e-7, rtol=0)
    assert torch.exp(actual).sum().item() == pytest.approx(1.0, abs=1e-7)
    assert actual[1].item() < 0.0


def test_temperature_is_applied_after_rp_and_chosen_token_is_gathered() -> None:
    # Catches applying temperature before repetition penalty or returning raw scores.
    contract = _contract(temperature=0.5)
    logprobs = processed_policy_logprobs(
        torch.tensor([-2.0, 1.0, 0.0]), (0,), contract
    )
    expected = torch.log_softmax(torch.tensor([-4.4, 2.0, 0.0]), dim=-1)
    assert logprobs[1].item() == pytest.approx(expected[1].item(), abs=1e-7)


@pytest.mark.parametrize(
    ("raw_logits", "history", "match"),
    [
        (torch.tensor([0.0, math.nan]), (), "finite"),
        (torch.tensor([0.0, math.inf]), (), "finite"),
        (torch.tensor([0.0, 1.0]), (2,), "vocabulary"),
        (torch.tensor([[0.0, 1.0]]), (), "one-dimensional"),
    ],
)
def test_policy_transform_rejects_nonfinite_or_malformed_inputs(
    raw_logits: torch.Tensor, history: tuple[int, ...], match: str
) -> None:
    # Catches silently normalizing malformed model outputs or histories.
    with pytest.raises(ValueError, match=match):
        processed_policy_logprobs(raw_logits, history, _contract())


def test_content_addressed_records_round_trip_and_are_immutable() -> None:
    # Catches incomplete canonical serialization or mutable evidence state.
    trajectory = _trajectory()
    restored = CompleteTrajectoryEvidence.from_dict(trajectory.to_dict())
    group = AcquisitionGroup(
        identity=_identity(),
        policy_contract=_contract(),
        trajectories=(trajectory,),
        seed_group_id="qualification",
    )
    assert restored == trajectory
    assert restored.content_sha256 == trajectory.content_sha256
    assert AcquisitionGroup.from_dict(group.to_dict()) == group
    assert len(group.content_sha256) == 64
    with pytest.raises(AttributeError):
        trajectory.terminal_kind = "cap_stop"  # type: ignore[misc]


def test_acquisition_group_preserves_distinct_request_bound_trajectories() -> None:
    # Catches collapsing K requests into one request identity at group sealing.
    first_contract = PolicyContract(
        identity=_identity(request_id="request:seed-11:image-7"),
        repetition_penalty=1.10,
        temperature=0.4,
        natural_stop_token_id=2,
        max_new_tokens=4,
    )
    second_contract = PolicyContract(
        identity=_identity(request_id="request:seed-12:image-7"),
        repetition_penalty=1.10,
        temperature=0.4,
        natural_stop_token_id=2,
        max_new_tokens=4,
    )
    first = _trajectory(contract=first_contract)
    second = _trajectory(contract=second_contract)
    group_contract = PolicyContract(
        identity=_identity(request_id="group:qualification:image-7"),
        repetition_penalty=1.10,
        temperature=0.4,
        natural_stop_token_id=2,
        max_new_tokens=4,
    )
    group = AcquisitionGroup(
        identity=group_contract.identity,
        policy_contract=group_contract,
        trajectories=(first, second),
        seed_group_id="qualification",
    )
    assert [item.identity.request_id for item in group.trajectories] == [
        "request:seed-11:image-7",
        "request:seed-12:image-7",
    ]


def test_complete_trajectory_requires_full_prompt_and_generated_history() -> None:
    # Catches accepting token evidence whose causal history omits the prompt.
    contract = _contract()
    identity = _identity(generated=(4, 2))
    broken = GeneratedTokenEvidence(
        identity=identity,
        policy_contract_sha256=contract.content_sha256,
        token_index=0,
        history_token_ids=(),
        chosen_token_id=4,
        processed_logprob=-0.25,
    )
    with pytest.raises(ValueError, match="history"):
        CompleteTrajectoryEvidence(
            identity=identity,
            policy_contract=contract,
            generated_tokens=(broken,),
            terminal_kind="natural_stop",
        )


@pytest.mark.parametrize(
    ("generated", "terminal_kind", "match"),
    [
        ((4, 2), "cap_stop", "cap_stop"),
        ((4, 5, 6), "natural_stop", "natural stop"),
        ((4, 5, 6, 2), "cap_stop", "natural stop"),
    ],
)
def test_trajectory_keeps_natural_and_cap_stops_exact(
    generated: tuple[int, ...], terminal_kind: str, match: str
) -> None:
    # Catches relabeling a cap as a natural terminal event (or vice versa).
    logprobs = tuple(-0.1 for _ in generated)
    with pytest.raises(ValueError, match=match):
        _trajectory(
            generated=generated,
            logprobs=logprobs,
            terminal_kind=terminal_kind,
        )


def test_replay_admits_fixed_per_token_and_group_mean_parity() -> None:
    # Catches substituting a looser aggregate-only replay gate.
    sampled = _trajectory(logprobs=(-0.400, -0.800))
    replayed = _trajectory(logprobs=(-0.4019, -0.7981))
    receipt = validate_policy_replay(sampled, replayed, ReplayTolerance())
    assert receipt.admitted
    assert receipt.per_token_absolute_error_nats == pytest.approx((0.0019, 0.0019))
    assert receipt.mean_absolute_error_nats == pytest.approx(0.0019)


@pytest.mark.parametrize(
    ("replayed", "tolerance", "match"),
    [
        (_trajectory(logprobs=(-0.421, -0.800)), ReplayTolerance(), "per-token"),
        (_trajectory(logprobs=(-0.403, -0.803)), ReplayTolerance(), "group mean"),
        (_trajectory(contract=_contract(repetition_penalty=1.0)), ReplayTolerance(), "contract"),
    ],
)
def test_replay_fails_closed_on_parity_or_exact_contract_mismatch(
    replayed: CompleteTrajectoryEvidence, tolerance: ReplayTolerance, match: str
) -> None:
    # Catches admitting a tolerance breach or a different sealed RP contract.
    with pytest.raises(PolicyReplayError, match=match):
        validate_policy_replay(_trajectory(logprobs=(-0.400, -0.800)), replayed, tolerance)


def test_replay_rejects_unsealed_tolerances() -> None:
    # Catches caller-controlled widening of the declared numeric gate.
    with pytest.raises(ValueError, match="fixed"):
        validate_policy_replay(
            _trajectory(),
            _trajectory(),
            ReplayTolerance(per_token_nats=0.03, group_mean_nats=0.003),
        )
