#!/usr/bin/env python3
"""Fresh, RP-aware Human-13 K16 acquisition for the crossover screen.

This module owns a new request surface.  It deliberately does not use the
historical discovery planner because that planner seals ``top_p=.95``.  Live
execution is injected so imports, model creation, and artifact publication are
unreachable from the plan-only path.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

from scripts.research.build_human13_k_union_manifest import EXPECTED_IMAGE_IDENTITIES
from scripts.research.human13_k_trajectory_contracts import (
    AcquisitionGroup,
    ArtifactIdentity,
    CompleteTrajectoryEvidence,
    GeneratedTokenEvidence,
    PolicyContract,
    ReplayTolerance,
)


REQUESTS_PER_BATCH = 4
BATCHES_PER_IMAGE = 4
REQUESTS_PER_IMAGE = REQUESTS_PER_BATCH * BATCHES_PER_IMAGE
NATURAL_STOP_TOKEN_ID = 151645
MAX_NEW_TOKENS = 512
PROCESSOR_ORDER = ("repetition_penalty", "temperature", "log_softmax")
QUALIFICATION_SEEDS = tuple(range(30001, 30017))
MATRIX_SEED_GROUPS = {
    "matrix_a": tuple(range(31001, 31017)),
    "matrix_b": tuple(range(32001, 32017)),
    "matrix_c": tuple(range(33001, 33017)),
}
_SEED_GROUPS = {"qualification": QUALIFICATION_SEEDS, **MATRIX_SEED_GROUPS}
_SAMPLING_BASE = {
    "n": 1,
    "temperature": 0.4,
    "top_p": 1.0,
    "top_k": None,
    "max_new_tokens": MAX_NEW_TOKENS,
    "stop_token_ids": (NATURAL_STOP_TOKEN_ID,),
    "ignore_eos": False,
}


@dataclass(frozen=True)
class AcquisitionRequest:
    image_id: int
    seed_group_id: str
    seed: int
    physical_batch_index: int
    request_order_in_batch: int
    request_id: str
    sampling: Mapping[str, object]


@dataclass(frozen=True)
class AcquisitionBatch:
    image_id: int
    seed_group_id: str
    batch_index: int
    requests: tuple[AcquisitionRequest, ...]


@dataclass(frozen=True)
class AcquisitionGroupPlan:
    image_id: int
    seed_group_id: str
    repetition_penalty: float
    batches: tuple[AcquisitionBatch, ...]

    @property
    def requests(self) -> tuple[AcquisitionRequest, ...]:
        return tuple(request for batch in self.batches for request in batch.requests)


@dataclass(frozen=True)
class PackedRawLogits:
    """Ephemeral compact replay input; full vocabulary rows remain tensors."""

    request_ids: tuple[str, ...]
    token_indices: tuple[int, ...]
    logits: Any

    def __post_init__(self) -> None:
        import torch

        if not isinstance(self.logits, torch.Tensor) or self.logits.ndim != 2:
            raise ValueError("packed raw logits must be a two-dimensional tensor")
        if len(self.request_ids) != len(self.token_indices) or len(self.request_ids) != self.logits.shape[0]:
            raise ValueError("packed raw logits rows do not cover the declared causal positions")
        if not self.request_ids or any(not isinstance(value, str) or not value for value in self.request_ids):
            raise ValueError("packed raw logits require nonempty request identities")
        if any(isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in self.token_indices):
            raise ValueError("packed raw logits token indexes must be nonnegative integers")
        if not bool(torch.isfinite(self.logits).all().item()):
            raise ValueError("packed raw logits must be finite")


def _request_id(*, image_id: int, seed_group_id: str, seed: int) -> str:
    return f"human13:{image_id}:rp-crossover:{seed_group_id}:{seed}"


def _seed_group(seed_group_id: str) -> tuple[int, ...]:
    try:
        return _SEED_GROUPS[seed_group_id]
    except KeyError as exc:
        raise ValueError("seed group is not frozen for Human-13 RP crossover") from exc


def _validate_plan(plan: AcquisitionGroupPlan) -> None:
    expected_ids = tuple(image_id for image_id, _ in EXPECTED_IMAGE_IDENTITIES)
    if plan.image_id not in expected_ids:
        raise ValueError("acquisition image is outside canonical Human-13 panel")
    if plan.repetition_penalty not in (1.0, 1.10):
        raise ValueError("training repetition penalty must be exactly 1.0 or 1.10")
    seeds = _seed_group(plan.seed_group_id)
    if len(plan.batches) != BATCHES_PER_IMAGE:
        raise ValueError("each acquisition image requires four physical batches")
    if [batch.batch_index for batch in plan.batches] != list(range(BATCHES_PER_IMAGE)):
        raise ValueError("physical batches must retain canonical order")
    if any(batch.image_id != plan.image_id or batch.seed_group_id != plan.seed_group_id for batch in plan.batches):
        raise ValueError("acquisition batch lineage differs from its group")
    requests = plan.requests
    if len(requests) != REQUESTS_PER_IMAGE or any(len(batch.requests) != REQUESTS_PER_BATCH for batch in plan.batches):
        raise ValueError("each image requires four batches of four requests")
    if tuple(request.seed for request in requests) != seeds:
        raise ValueError("acquisition seed ordering differs from its frozen group")
    if len({request.seed for request in requests}) != REQUESTS_PER_IMAGE:
        raise ValueError("acquisition group contains duplicate seeds")
    if len({request.request_id for request in requests}) != REQUESTS_PER_IMAGE:
        raise ValueError("acquisition group contains duplicate request identities")
    expected_sampling = {**_SAMPLING_BASE, "repetition_penalty": plan.repetition_penalty}
    for index, request in enumerate(requests):
        if (
            request.image_id != plan.image_id
            or request.seed_group_id != plan.seed_group_id
            or request.physical_batch_index != index // REQUESTS_PER_BATCH
            or request.request_order_in_batch != index % REQUESTS_PER_BATCH
            or request.request_id != _request_id(image_id=plan.image_id, seed_group_id=plan.seed_group_id, seed=request.seed)
            or dict(request.sampling) != expected_sampling
        ):
            raise ValueError("acquisition request differs from the sealed batch-four policy")


def plan_acquisition_group(*, image_id: int, repetition_penalty: float, seed_group_id: str) -> AcquisitionGroupPlan:
    """Return one immutable K16 plan; request identity and order are explicit."""

    seeds = _seed_group(seed_group_id)
    policy = MappingProxyType({**_SAMPLING_BASE, "repetition_penalty": repetition_penalty})
    batches = tuple(
        AcquisitionBatch(
            image_id=image_id,
            seed_group_id=seed_group_id,
            batch_index=batch_index,
            requests=tuple(
                AcquisitionRequest(
                    image_id=image_id,
                    seed_group_id=seed_group_id,
                    seed=seed,
                    physical_batch_index=batch_index,
                    request_order_in_batch=order,
                    request_id=_request_id(image_id=image_id, seed_group_id=seed_group_id, seed=seed),
                    sampling=policy,
                )
                for order, seed in enumerate(seeds[batch_index * 4 : (batch_index + 1) * 4])
            ),
        )
        for batch_index in range(BATCHES_PER_IMAGE)
    )
    plan = AcquisitionGroupPlan(image_id, seed_group_id, float(repetition_penalty), batches)
    _validate_plan(plan)
    return plan


def plan_panel_acquisition(*, repetition_penalty: float, seed_group_id: str) -> tuple[AcquisitionGroupPlan, ...]:
    plans = tuple(
        plan_acquisition_group(image_id=image_id, repetition_penalty=repetition_penalty, seed_group_id=seed_group_id)
        for image_id, _ in EXPECTED_IMAGE_IDENTITIES
    )
    if len(plans) != 13:
        raise ValueError("canonical Human-13 panel must contain exactly thirteen images")
    return plans


def dry_run_plan() -> dict[str, object]:
    """Plan only: no vLLM import, model construction, engine open, or write."""

    return {
        "schema_version": "human13_rp_crossover_acquisition.v1",
        "status": "plan_only",
        "image_count": 13,
        "physical_batch_count_per_group": 52,
        "request_count_per_group": 208,
        "requests_per_image": REQUESTS_PER_IMAGE,
        "seed_groups": {key: list(value) for key, value in _SEED_GROUPS.items()},
        "sampling": {**_SAMPLING_BASE, "repetition_penalty": "explicit: 1.0 or 1.10"},
        "actions": {"model_imports": 0, "model_loads": 0, "engine_opens": 0, "gpu_allocations": 0, "artifact_writes": 0},
    }


def vllm_sampling_params(request: AcquisitionRequest) -> Any:
    """Build the native request only on explicit live execution.

    vLLM represents disabled top-k as zero.  The sealed artifact policy uses
    ``None`` to make the no-truncation meaning explicit.
    """

    from vllm import SamplingParams

    return SamplingParams(
        n=1, seed=request.seed, temperature=0.4, top_p=1.0, top_k=0,
        repetition_penalty=_request_repetition_penalty(request), max_tokens=MAX_NEW_TOKENS,
        logprobs=1, stop_token_ids=[NATURAL_STOP_TOKEN_ID], ignore_eos=False,
        detokenize=True, skip_special_tokens=False, spaces_between_special_tokens=True,
    )


def _native_field(native: Mapping[str, object], name: str) -> object:
    if name not in native:
        raise ValueError(f"native evidence lacks {name}")
    return native[name]


def _native_sequence(native: Mapping[str, object], name: str) -> tuple[object, ...]:
    value = _native_field(native, name)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"native evidence {name} must be a sequence")
    return tuple(value)


def _request_repetition_penalty(request: AcquisitionRequest) -> float:
    value = request.sampling["repetition_penalty"]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("request repetition penalty must be numeric")
    return float(value)


def _token_ids(value: tuple[object, ...], *, field: str) -> tuple[int, ...]:
    if any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in value):
        raise ValueError(f"native evidence {field} must contain nonnegative token ids")
    return tuple(cast(int, token) for token in value)


def _scores(value: tuple[object, ...]) -> tuple[float, ...]:
    if any(isinstance(score, bool) or not isinstance(score, (int, float)) for score in value):
        raise ValueError("native evidence processed log probabilities must be numeric")
    return tuple(float(cast(int | float, score)) for score in value)


def native_trajectory_evidence(*, request: AcquisitionRequest, native: Mapping[str, object]) -> CompleteTrajectoryEvidence:
    """Seal one fresh native output, rejecting trace gaps before group assembly."""

    generated = _token_ids(_native_sequence(native, "generated_token_ids"), field="generated_token_ids")
    scores = _scores(_native_sequence(native, "processed_logprobs"))
    prompt = _token_ids(_native_sequence(native, "prompt_token_ids"), field="prompt_token_ids")
    if not generated or len(generated) != len(scores):
        raise ValueError("native evidence requires complete chosen-token log probabilities")
    if native.get("evidence_origin") != "fresh_native":
        raise ValueError("historical traces cannot supply score-function evidence")
    processor_order = native.get("processor_order")
    if not isinstance(processor_order, Sequence) or isinstance(processor_order, (str, bytes)) or tuple(processor_order) != PROCESSOR_ORDER:
        raise ValueError("native processor order differs from the sealed policy")
    sampling = _native_field(native, "sampling")
    if not isinstance(sampling, Mapping) or dict(sampling) != dict(request.sampling):
        raise ValueError("native sampler parameters differ from the sealed request")
    terminal_kind = _native_field(native, "terminal_kind")
    if terminal_kind == "natural_stop":
        if generated[-1] != NATURAL_STOP_TOKEN_ID:
            raise ValueError("natural stop must end with im_end")
    elif terminal_kind == "cap_stop":
        if len(generated) != MAX_NEW_TOKENS or NATURAL_STOP_TOKEN_ID in generated:
            raise ValueError("cap stop must retain all 512 tokens before im_end")
    else:
        raise ValueError("native terminal kind must be natural_stop or cap_stop")
    identity = ArtifactIdentity(
        source_sha256=str(_native_field(native, "source_sha256")), manifest_sha256=str(_native_field(native, "manifest_sha256")),
        request_id=request.request_id, model_id=str(_native_field(native, "model_id")), tokenizer_id=str(_native_field(native, "tokenizer_id")),
        processor_id=str(_native_field(native, "processor_id")), prompt_token_ids=prompt, generated_token_ids=generated,
    )
    contract = PolicyContract(
        identity=identity, repetition_penalty=_request_repetition_penalty(request), temperature=0.4,
        natural_stop_token_id=NATURAL_STOP_TOKEN_ID, max_new_tokens=MAX_NEW_TOKENS, processor_order=PROCESSOR_ORDER,
        sampler_backend_id=str(_native_field(native, "sampler_backend_id")), top_p=1.0, top_k=None, n=1,
    )
    tokens = tuple(
        GeneratedTokenEvidence(
            identity=identity, policy_contract_sha256=contract.content_sha256, token_index=index,
            history_token_ids=(*prompt, *generated[:index]), chosen_token_id=token, processed_logprob=score,
        )
        for index, (token, score) in enumerate(zip(generated, scores, strict=True))
    )
    return CompleteTrajectoryEvidence(identity=identity, policy_contract=contract, generated_tokens=tokens, terminal_kind=str(terminal_kind))


def acquisition_group_from_native(*, plan: AcquisitionGroupPlan, native_by_request_id: Mapping[str, Mapping[str, object]]) -> AcquisitionGroup:
    """Bind exactly the sealed K16 fresh outputs in request order."""

    _validate_plan(plan)
    expected = tuple(request.request_id for request in plan.requests)
    if set(native_by_request_id) != set(expected) or len(native_by_request_id) != REQUESTS_PER_IMAGE:
        raise ValueError("native results must cover every sealed request exactly once")
    trajectories = tuple(native_trajectory_evidence(request=request, native=native_by_request_id[request.request_id]) for request in plan.requests)
    return AcquisitionGroup(identity=trajectories[0].identity, policy_contract=trajectories[0].policy_contract, trajectories=trajectories, seed_group_id=plan.seed_group_id)


def execute_acquisition_group(*, plan: AcquisitionGroupPlan, execute_batch: Callable[[AcquisitionBatch, tuple[Any, ...]], Mapping[str, Mapping[str, object]]]) -> AcquisitionGroup:
    """Injected live executor around a caller-owned vLLM session; never opens one."""

    _validate_plan(plan)
    native: dict[str, Mapping[str, object]] = {}
    for batch in plan.batches:
        results = execute_batch(batch, tuple(vllm_sampling_params(request) for request in batch.requests))
        expected = {request.request_id for request in batch.requests}
        if set(results) != expected or set(native).intersection(results):
            raise ValueError("live batch result coverage or ordering differs from sealed requests")
        native.update(results)
    return acquisition_group_from_native(plan=plan, native_by_request_id=native)


def replay_acquisition_group(*, sampled: AcquisitionGroup, packed: PackedRawLogits) -> tuple[AcquisitionGroup, Any]:
    """Replay packed causal rows and issue one strict Task-1 group parity receipt."""

    from scripts.research.human13_rp_policy import processed_policy_logprobs, validate_acquisition_group_replay

    expected = tuple((trajectory.identity.request_id, token.token_index) for trajectory in sampled.trajectories for token in trajectory.generated_tokens)
    observed = tuple(zip(packed.request_ids, packed.token_indices, strict=True))
    if observed != expected:
        raise ValueError("packed raw logits must cover generated causal positions in sealed request order")
    rows = iter(packed.logits.unbind(0))
    replayed_trajectories: list[CompleteTrajectoryEvidence] = []
    for trajectory in sampled.trajectories:
        tokens = tuple(
            GeneratedTokenEvidence(
                identity=trajectory.identity, policy_contract_sha256=trajectory.policy_contract.content_sha256,
                token_index=token.token_index, history_token_ids=token.history_token_ids, chosen_token_id=token.chosen_token_id,
                processed_logprob=float(processed_policy_logprobs(next(rows), token, trajectory.policy_contract)[token.chosen_token_id].item()),
            )
            for token in trajectory.generated_tokens
        )
        replayed_trajectories.append(CompleteTrajectoryEvidence(identity=trajectory.identity, policy_contract=trajectory.policy_contract, generated_tokens=tokens, terminal_kind=trajectory.terminal_kind))
    replayed = AcquisitionGroup(identity=sampled.identity, policy_contract=sampled.policy_contract, trajectories=tuple(replayed_trajectories), seed_group_id=sampled.seed_group_id)
    receipt = validate_acquisition_group_replay(sampled, replayed, ReplayTolerance())
    return replayed, receipt


def publish_acquisition_group(*, output_root: str | Path, sampled: AcquisitionGroup, replay_receipt: Any) -> Path:
    """Publish only an already-admitted complete group; never persist raw logits."""

    if getattr(replay_receipt, "admitted", False) is not True or getattr(replay_receipt, "sampled_group_sha256", None) != sampled.content_sha256:
        raise ValueError("parity receipt must admit this exact acquisition group before publication")
    target = Path(output_root).resolve()
    if target.exists():
        raise FileExistsError("refusing to overwrite an acquisition artifact")
    target.mkdir(parents=True)
    try:
        (target / "acquisition-group.json").write_text(json.dumps(sampled.to_dict(), sort_keys=True) + "\n", encoding="utf-8")
        (target / "replay-receipt.json").write_text(json.dumps(replay_receipt.to_dict(), sort_keys=True) + "\n", encoding="utf-8")
    except BaseException:
        raise
    return target


__all__ = [
    "AcquisitionBatch", "AcquisitionGroupPlan", "AcquisitionRequest", "MATRIX_SEED_GROUPS", "PackedRawLogits",
    "QUALIFICATION_SEEDS", "acquisition_group_from_native", "dry_run_plan", "execute_acquisition_group",
    "native_trajectory_evidence", "plan_acquisition_group", "plan_panel_acquisition", "publish_acquisition_group",
    "replay_acquisition_group", "vllm_sampling_params",
]
