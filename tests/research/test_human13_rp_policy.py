from __future__ import annotations

from dataclasses import replace
import hashlib
import json
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
    AcquisitionGroupParityReceipt,
    PolicyReplayError,
    validate_acquisition_group_replay,
    processed_policy_logprobs,
    validate_policy_replay,
)


def _identity(
    *,
    generated: tuple[int, ...] = (),
    request_id: str = "request:seed-11:image-7",
    prompt: tuple[int, ...] = (101, 102, 103),
) -> ArtifactIdentity:
    return ArtifactIdentity(
        source_sha256="a" * 64,
        manifest_sha256="b" * 64,
        request_id=request_id,
        model_id="Qwen/Qwen3-VL-8B-Instruct",
        tokenizer_id="qwen3-vl-tokenizer:sha256:c",
        processor_id="qwen3-vl-processor:sha256:d",
        prompt_token_ids=prompt,
        generated_token_ids=generated,
    )


def _contract(
    *,
    repetition_penalty: float = 1.10,
    temperature: float = 0.4,
    identity: ArtifactIdentity | None = None,
    top_p: float = 1.0,
    top_k: int | None = None,
) -> PolicyContract:
    return PolicyContract(
        identity=identity or _identity(),
        repetition_penalty=repetition_penalty,
        temperature=temperature,
        natural_stop_token_id=2,
        max_new_tokens=4,
        sampler_backend_id="vllm:0.8.5",
        top_p=top_p,
        top_k=top_k,
        n=1,
        min_new_tokens=0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        ignore_eos=False,
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


def _token_evidence(
    contract: PolicyContract,
    *,
    generated: tuple[int, ...] = (1, 2),
    token_index: int = 0,
    history: tuple[int, ...] | None = None,
    identity: ArtifactIdentity | None = None,
) -> GeneratedTokenEvidence:
    source = contract.identity
    identity = identity or ArtifactIdentity(
        source_sha256=source.source_sha256,
        manifest_sha256=source.manifest_sha256,
        request_id=source.request_id,
        model_id=source.model_id,
        tokenizer_id=source.tokenizer_id,
        processor_id=source.processor_id,
        prompt_token_ids=source.prompt_token_ids,
        generated_token_ids=generated,
    )
    return GeneratedTokenEvidence(
        identity=identity,
        policy_contract_sha256=contract.content_sha256,
        token_index=token_index,
        history_token_ids=history
        if history is not None
        else (*identity.prompt_token_ids, *generated[:token_index]),
        chosen_token_id=generated[token_index],
        processed_logprob=-0.25,
    )


def test_rp110_is_sign_aware_on_unique_complete_history_tokens() -> None:
    # Catches changing the negative-logit branch from multiplication to division.
    contract = _contract(identity=_identity(prompt=(0, 1)))
    token = _token_evidence(contract, generated=(1, 2), token_index=1)
    actual = processed_policy_logprobs(torch.tensor([4.0, -2.0, 1.0, -3.0]), token, contract)
    expected = torch.log_softmax(
        torch.tensor([4.0 / 1.10, -2.0 * 1.10, 1.0, -3.0]) / 0.4, dim=-1
    )
    assert actual.dtype == torch.float32
    assert torch.allclose(actual, expected, atol=1e-7, rtol=0)


def test_rp100_is_identity_before_temperature_and_full_support_normalization() -> None:
    # Catches adding an RP branch when the contract declares RP 1.0.
    raw = torch.tensor([1.0, 0.0, -1.0], dtype=torch.float64)
    contract = _contract(repetition_penalty=1.0, identity=_identity(prompt=(0, 2)))
    actual = processed_policy_logprobs(raw, _token_evidence(contract), contract)
    expected = torch.log_softmax(torch.tensor([2.5, 0.0, -2.5]), dim=-1)
    assert torch.allclose(actual, expected, atol=1e-7, rtol=0)
    assert torch.exp(actual).sum().item() == pytest.approx(1.0, abs=1e-7)
    assert actual[1].item() < 0.0


def test_temperature_is_applied_after_rp_and_chosen_token_is_gathered() -> None:
    # Catches applying temperature before repetition penalty or returning raw scores.
    compact = _contract(temperature=0.5, identity=_identity(prompt=(0,)))
    logprobs = processed_policy_logprobs(
        torch.tensor([-2.0, 1.0, 0.0]), _token_evidence(compact), compact
    )
    expected = torch.log_softmax(torch.tensor([-4.4, 2.0, 0.0]), dim=-1)
    assert logprobs[1].item() == pytest.approx(expected[1].item(), abs=1e-7)


@pytest.mark.parametrize(
    ("raw_logits", "match"),
    [
        (torch.tensor([0.0, math.nan]), "finite"),
        (torch.tensor([0.0, math.inf]), "finite"),
        (torch.tensor([[0.0, 1.0]]), "one-dimensional"),
    ],
)
def test_policy_transform_rejects_nonfinite_or_malformed_inputs(
    raw_logits: torch.Tensor, match: str
) -> None:
    # Catches silently normalizing malformed model outputs or histories.
    with pytest.raises(ValueError, match=match):
        contract = _contract(identity=_identity(prompt=(0,)))
        processed_policy_logprobs(raw_logits, _token_evidence(contract), contract)


def test_policy_transform_rejects_token_evidence_with_prompt_or_prefix_mismatch() -> None:
    # Catches scoring a tuple which does not equal the sealed evidence history.
    contract = _contract(identity=_identity(prompt=(0, 1)))
    with pytest.raises(ValueError, match="prompt"):
        processed_policy_logprobs(
            torch.tensor([0.0, 1.0, 2.0]),
            _token_evidence(contract, history=(9, 1)),
            contract,
        )
    with pytest.raises(ValueError, match="generated prefix"):
        processed_policy_logprobs(
            torch.tensor([0.0, 1.0, 2.0]),
            _token_evidence(contract, generated=(1, 2), token_index=1, history=(0, 1, 0)),
            contract,
        )


def test_sampler_lineage_is_sealed_and_changes_the_contract_hash() -> None:
    # Catches treating sampler settings as non-identifying metadata.
    baseline = _contract()
    assert baseline.content_sha256 != _contract(top_p=0.95).content_sha256
    assert baseline.content_sha256 != _contract(top_k=20).content_sha256
    assert baseline.to_dict()["sampler_backend_id"] == "vllm:0.8.5"
    assert baseline.to_dict()["top_k"] is None


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
    first_contract = _contract(
        identity=_identity(request_id="request:seed-11:image-7")
    )
    second_contract = _contract(
        identity=_identity(request_id="request:seed-12:image-7")
    )
    first = _trajectory(contract=first_contract)
    second = _trajectory(contract=second_contract)
    group_contract = _contract(
        identity=_identity(request_id="group:qualification:image-7")
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
    assert receipt.trajectory_mean_absolute_error_nats == pytest.approx(0.0019)


@pytest.mark.parametrize(
    ("replayed", "tolerance", "match"),
    [
        (_trajectory(logprobs=(-0.421, -0.800)), ReplayTolerance(), "per-token"),
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


def _group(*trajectories: CompleteTrajectoryEvidence) -> AcquisitionGroup:
    group_contract = _contract(identity=_identity(request_id="group:qualification:image-7"))
    return AcquisitionGroup(
        identity=group_contract.identity,
        policy_contract=group_contract,
        trajectories=trajectories,
        seed_group_id="qualification",
    )


def test_group_replay_uses_one_global_mean_not_a_trajectory_local_mean() -> None:
    # Catches accepting a K group because only one trajectory-local average passes.
    first_contract = _contract(identity=_identity(request_id="request:seed-11:image-7"))
    second_contract = _contract(identity=_identity(request_id="request:seed-12:image-7"))
    sampled = _group(
        _trajectory(contract=first_contract, logprobs=(-0.4, -0.8)),
        _trajectory(contract=second_contract, logprobs=(-0.4, -0.8)),
    )
    replayed = _group(
        _trajectory(contract=first_contract, logprobs=(-0.401, -0.801)),
        _trajectory(contract=second_contract, logprobs=(-0.404, -0.804)),
    )
    with pytest.raises(PolicyReplayError, match="group mean"):
        validate_acquisition_group_replay(sampled, replayed, ReplayTolerance())


def test_group_replay_can_admit_when_one_local_mean_exceeds_group_mean() -> None:
    # Catches adding an undeclared per-trajectory mean gate to a group receipt.
    first_contract = _contract(identity=_identity(request_id="request:seed-11:image-7"))
    second_contract = _contract(identity=_identity(request_id="request:seed-12:image-7"))
    sampled = _group(
        _trajectory(generated=(4, 5, 6, 2), logprobs=(-0.4, -0.4, -0.4, -0.4), contract=first_contract),
        _trajectory(contract=second_contract),
    )
    replayed = _group(
        _trajectory(generated=(4, 5, 6, 2), logprobs=(-0.4029, -0.4029, -0.4029, -0.4029), contract=first_contract),
        _trajectory(contract=second_contract),
    )
    receipt = validate_acquisition_group_replay(sampled, replayed, ReplayTolerance())
    assert receipt.group_mean_absolute_error_nats == pytest.approx(0.0019333333333333333)
    assert receipt.content_sha256 == receipt.content_sha256


def test_group_replay_fails_closed_on_group_lineage_mismatch() -> None:
    # Catches pooling trajectories from a sampler contract with a different top-p.
    contract = _contract()
    sampled = _group(_trajectory(contract=contract))
    changed_group_contract = _contract(
        identity=_identity(request_id="group:qualification:image-7"), top_p=0.95
    )
    changed_trajectory_contract = _contract(
        identity=_identity(request_id="request:seed-11:image-7"), top_p=0.95
    )
    replayed = AcquisitionGroup(
        identity=changed_group_contract.identity,
        policy_contract=changed_group_contract,
        trajectories=(_trajectory(contract=changed_trajectory_contract),),
        seed_group_id="qualification",
    )
    with pytest.raises(PolicyReplayError, match="group lineage"):
        validate_acquisition_group_replay(sampled, replayed, ReplayTolerance())


def _parity_receipt() -> AcquisitionGroupParityReceipt:
    return AcquisitionGroupParityReceipt(
        admitted=True,
        tolerance_sha256=ReplayTolerance().content_sha256,
        sampled_group_sha256="b" * 64,
        replayed_group_sha256="c" * 64,
        request_ids=("request:seed-11:image-7", "request:seed-12:image-7"),
        token_count=2,
        per_token_absolute_error_nats=(0.001, 0.003),
        group_mean_absolute_error_nats=0.002,
    )


def test_group_parity_receipt_is_a_canonical_round_trippable_record() -> None:
    # Catches a receipt whose persisted payload can change or cannot be verified.
    receipt = _parity_receipt()
    encoded = json.dumps(receipt.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    restored = AcquisitionGroupParityReceipt.from_dict(json.loads(encoded))
    assert restored == receipt
    assert restored.content_sha256 == hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    assert restored.token_count == len(restored.per_token_absolute_error_nats)


@pytest.mark.parametrize(
    ("replacement", "match"),
    [
        ({"tolerance_sha256": "not-a-digest"}, "SHA-256"),
        ({"token_count": 0}, "token_count"),
        ({"token_count": 1}, "token_count"),
        ({"per_token_absolute_error_nats": (math.nan, 0.003)}, "finite"),
        ({"per_token_absolute_error_nats": (-0.001, 0.003)}, "nonnegative"),
        ({"group_mean_absolute_error_nats": math.inf}, "finite"),
    ],
)
def test_group_parity_receipt_rejects_invalid_sealed_values(
    replacement: dict[str, object], match: str
) -> None:
    # Catches persisted numerical or identity corruption before a receipt is used.
    with pytest.raises(ValueError, match=match):
        replace(_parity_receipt(), **replacement)


@pytest.mark.parametrize("mutation", ("missing", "extra", "forged"))
def test_group_parity_receipt_rejects_noncanonical_persisted_fields(
    mutation: str,
) -> None:
    # Catches accepting ambiguous or forged serialized receipt shapes.
    payload = _parity_receipt().to_dict()
    if mutation == "missing":
        del payload["token_count"]
    elif mutation == "extra":
        payload["unexpected"] = True
    else:
        payload["group_mean_absolute_error_nats"] = 0.001
    with pytest.raises(ValueError, match="fields|mean"):
        AcquisitionGroupParityReceipt.from_dict(payload)


@pytest.mark.parametrize(
    ("replacement", "match"),
    [
        ({"admitted": False}, "admitted"),
        ({"tolerance_sha256": "d" * 64}, "tolerance"),
        (
            {
                "per_token_absolute_error_nats": (0.021, 0.003),
                "group_mean_absolute_error_nats": 0.012,
            },
            "per-token",
        ),
        (
            {
                "per_token_absolute_error_nats": (0.003, 0.003),
                "group_mean_absolute_error_nats": 0.003,
            },
            "group mean",
        ),
    ],
)
def test_admitted_group_parity_receipt_enforces_the_fixed_replay_gate(
    replacement: dict[str, object], match: str
) -> None:
    # Catches forging an admitted receipt that the replay gate itself would reject.
    with pytest.raises(ValueError, match=match):
        replace(_parity_receipt(), **replacement)


@pytest.mark.parametrize("field", ("tolerance_sha256", "per_token_absolute_error_nats", "group_mean_absolute_error_nats"))
def test_deserialized_group_parity_receipt_rechecks_the_admission_gate(field: str) -> None:
    # Catches treating persisted receipt bytes as trusted after they are altered.
    payload = _parity_receipt().to_dict()
    if field == "tolerance_sha256":
        payload[field] = "d" * 64
    elif field == "per_token_absolute_error_nats":
        payload[field] = [0.021, 0.003]
        payload["group_mean_absolute_error_nats"] = 0.012
    else:
        payload["per_token_absolute_error_nats"] = [0.003, 0.003]
        payload[field] = 0.003
    with pytest.raises(ValueError, match="tolerance|per-token|group mean"):
        AcquisitionGroupParityReceipt.from_dict(payload)
