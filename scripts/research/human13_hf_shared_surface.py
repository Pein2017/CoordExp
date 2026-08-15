"""CPU-only sealed value contracts for the Human-13 all-HF shared surface."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from math import isfinite
from typing import Any, Callable, Iterable, Literal, Mapping, TypeVar
from weakref import ReferenceType, ref

from src.artifacts.json_values import json_sha256


_SEED_GROUPS = ((35001, 35002, 35003, 35004), (35005, 35006, 35007, 35008), (35009, 35010, 35011, 35012), (35013, 35014, 35015, 35016))
_PROCESSOR_ORDER = ("repetition_penalty", "temperature", "top_p")
_SEALED: dict[int, tuple[ReferenceType[object], str]] = {}
_T = TypeVar("_T", bound="_Sealed")


class SharedSurfaceContractError(ValueError):
    """A value cannot cross a shared-surface contract boundary."""


def _digest(value: object, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise SharedSurfaceContractError(f"{label} must be a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise SharedSurfaceContractError(f"{label} must be a SHA-256 digest") from exc
    return value


def _finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise SharedSurfaceContractError(f"{label} must be finite")
    return float(value)


def _keys(value: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    if set(value) != expected:
        raise SharedSurfaceContractError(f"{label} fields differ from canonical schema")


class _Sealed:
    def _payload(self) -> dict[str, object]:
        raise NotImplementedError

    @property
    def content_sha256(self) -> str:
        _require_admitted(self, label=type(self).__name__)
        return json_sha256(self._payload())

    def to_dict(self) -> dict[str, object]:
        _require_admitted(self, label=type(self).__name__)
        return self._payload() | {"content_sha256": self.content_sha256}


def _admit(value: _T, validator: Callable[[_T], None]) -> _T:
    """The sole sealing choke point for scientific/published values."""
    validator(value)
    fingerprint = json_sha256(value._payload())
    identity = id(value)

    def cleanup(_: ReferenceType[object], *, key: int = identity) -> None:
        _SEALED.pop(key, None)

    _SEALED[identity] = (ref(value, cleanup), fingerprint)
    return value


def _require_admitted(value: _Sealed, *, label: str) -> None:
    entry = _SEALED.get(id(value))
    if entry is None or entry[0]() is not value or entry[1] != json_sha256(value._payload()):
        raise SharedSurfaceContractError(f"{label} was not admitted through the shared-surface choke point")


@dataclass(frozen=True)
class HFSharedSurfaceIdentity:
    checkpoint_payload_sha256: str
    model_object_id: int
    parameter_state_sha256: str
    adapter_sha256: str
    embedding_delta_sha256: str
    dtype: Literal["bfloat16"]
    attention_backend: Literal["flash_attention_2"]
    model_mode: Literal["eval"]
    tokenizer_sha256: str
    prompt_sha256: str
    image_sha256: str
    use_cache: Literal[False]

    def __post_init__(self) -> None:
        for field in ("checkpoint_payload_sha256", "parameter_state_sha256", "adapter_sha256", "embedding_delta_sha256", "tokenizer_sha256", "prompt_sha256", "image_sha256"):
            _digest(getattr(self, field), label=field)
        if isinstance(self.model_object_id, bool) or not isinstance(self.model_object_id, int):
            raise SharedSurfaceContractError("model object identity must be an integer")
        if self.dtype != "bfloat16" or self.attention_backend != "flash_attention_2" or self.model_mode != "eval":
            raise SharedSurfaceContractError("shared surface identity differs from BF16/FA2/eval")
        if self.use_cache is not False:
            raise SharedSurfaceContractError("cache is forbidden")

    def to_dict(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> HFSharedSurfaceIdentity:
        _keys(value, set(cls.__dataclass_fields__), label="identity")
        return cls(**dict(value))


@dataclass(frozen=True)
class HFSharedSurfacePolicy:
    repetition_penalty: float
    repetition_penalty_before_temperature: Literal[True]
    prompt_inclusive_history: Literal[True]
    temperature: float
    top_p: float
    top_k: None
    max_new_tokens: int
    stop_token: str
    use_cache: Literal[False]

    def __post_init__(self) -> None:
        if _finite(self.repetition_penalty, label="repetition penalty") != 1.0 or self.repetition_penalty_before_temperature is not True:
            raise SharedSurfaceContractError("processor order requires frozen RP-before-temperature")
        if self.prompt_inclusive_history is not True:
            raise SharedSurfaceContractError("history must include prompt")
        if _finite(self.temperature, label="temperature") != 0.4 or _finite(self.top_p, label="top_p") != 1.0 or self.top_k is not None:
            raise SharedSurfaceContractError("initial sampling policy differs")
        if self.max_new_tokens != 512 or self.stop_token != "<|im_end|>":
            raise SharedSurfaceContractError("initial cap or stop policy differs")
        if self.use_cache is not False:
            raise SharedSurfaceContractError("cache is forbidden")

    def to_dict(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> HFSharedSurfacePolicy:
        _keys(value, set(cls.__dataclass_fields__), label="policy")
        return cls(**dict(value))


@dataclass(frozen=True)
class HFSharedSurfacePlan:
    image_id: Literal[1584]
    seed_groups: tuple[tuple[int, int, int, int], ...]
    policy: HFSharedSurfacePolicy

    def __post_init__(self) -> None:
        if self.image_id != 1584 or self.seed_groups != _SEED_GROUPS:
            raise SharedSurfaceContractError("seed coverage differs from frozen image-1584 K16 plan")

    def to_dict(self) -> dict[str, object]:
        return {"image_id": self.image_id, "seed_groups": [list(group) for group in self.seed_groups], "policy": self.policy.to_dict()}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> HFSharedSurfacePlan:
        _keys(value, {"image_id", "seed_groups", "policy"}, label="plan")
        if not isinstance(value["seed_groups"], list) or not isinstance(value["policy"], Mapping):
            raise SharedSurfaceContractError("plan must contain strict value objects")
        return cls(image_id=value["image_id"], seed_groups=tuple(tuple(group) for group in value["seed_groups"]), policy=HFSharedSurfacePolicy.from_dict(value["policy"]))  # type: ignore[arg-type]


def plan_image1584_k16() -> HFSharedSurfacePlan:
    return HFSharedSurfacePlan(1584, _SEED_GROUPS, HFSharedSurfacePolicy(1.0, True, True, 0.4, 1.0, None, 512, "<|im_end|>", False))


def causal_history_sha256(prompt_history_sha256: str, prior_token_ids: tuple[int, ...]) -> str:
    """The canonical prompt-inclusive causal history at one sampled action."""
    _digest(prompt_history_sha256, label="prompt_history_sha256")
    if any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in prior_token_ids):
        raise SharedSurfaceContractError("history/token lineage requires non-negative prior tokens")
    if not prior_token_ids:
        return prompt_history_sha256
    return json_sha256({"prompt_history_sha256": prompt_history_sha256, "prior_chosen_token_ids": list(prior_token_ids)})


@dataclass(frozen=True)
class SampledHFToken:
    request_id: str
    token_index: int
    history_sha256: str
    chosen_token_id: int
    raw_chosen_logit: float
    processed_logp: float
    causal_logit_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id:
            raise SharedSurfaceContractError("history/token lineage requires observed request identity")
        if isinstance(self.token_index, bool) or not isinstance(self.token_index, int) or self.token_index < 0:
            raise SharedSurfaceContractError("history/token lineage requires non-negative token index")
        _digest(self.history_sha256, label="history_sha256")
        if isinstance(self.chosen_token_id, bool) or not isinstance(self.chosen_token_id, int) or self.chosen_token_id < 0:
            raise SharedSurfaceContractError("chosen token id must be non-negative")
        _finite(self.raw_chosen_logit, label="raw chosen logit")
        _finite(self.processed_logp, label="processed log probability")
        if isinstance(self.causal_logit_index, bool) or not isinstance(self.causal_logit_index, int) or self.causal_logit_index < 0:
            raise SharedSurfaceContractError("causal gather index must be non-negative")

    def to_dict(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SampledHFToken:
        _keys(value, set(cls.__dataclass_fields__), label="sampled token")
        return cls(**dict(value))


@dataclass(frozen=True)
class SampledHFRequest:
    request_id: str
    image_id: int
    seed: int
    prompt_history_sha256: str
    tokens: tuple[SampledHFToken, ...]
    processor_order: tuple[str, str, str]
    stop_reason: Literal["im_end", "cap"]
    use_cache: Literal[False]

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id:
            raise SharedSurfaceContractError("history/token lineage requires observed request identity")
        _digest(self.prompt_history_sha256, label="prompt_history_sha256")
        if self.image_id != 1584 or isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise SharedSurfaceContractError("seed coverage differs from frozen plan")
        if self.processor_order != _PROCESSOR_ORDER:
            raise SharedSurfaceContractError("processor order differs from frozen policy")
        if self.stop_reason not in ("im_end", "cap"):
            raise SharedSurfaceContractError("stop reason differs from frozen policy")
        if self.use_cache is not False:
            raise SharedSurfaceContractError("cache is forbidden")
        if not self.tokens or len(self.tokens) > 512:
            raise SharedSurfaceContractError("token cap must be within 1..512")
        if tuple(token.token_index for token in self.tokens) != tuple(range(len(self.tokens))):
            raise SharedSurfaceContractError("history/token lineage must be contiguous")
        for index, token in enumerate(self.tokens):
            if token.request_id != self.request_id or token.history_sha256 != causal_history_sha256(self.prompt_history_sha256, tuple(item.chosen_token_id for item in self.tokens[:index])):
                raise SharedSurfaceContractError("history/token lineage differs from canonical causal history")
        if self.stop_reason == "cap" and len(self.tokens) != 512:
            raise SharedSurfaceContractError("cap stop reason requires 512 tokens")

    def to_dict(self) -> dict[str, object]:
        return {"request_id": self.request_id, "image_id": self.image_id, "seed": self.seed, "prompt_history_sha256": self.prompt_history_sha256, "tokens": [token.to_dict() for token in self.tokens], "processor_order": list(self.processor_order), "stop_reason": self.stop_reason, "use_cache": self.use_cache}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SampledHFRequest:
        _keys(value, set(cls.__dataclass_fields__), label="sampled request")
        if not isinstance(value["tokens"], list) or not isinstance(value["processor_order"], list) or not all(isinstance(token, Mapping) for token in value["tokens"]):
            raise SharedSurfaceContractError("sampled request must contain strict value objects")
        return cls(request_id=value["request_id"], image_id=value["image_id"], seed=value["seed"], prompt_history_sha256=value["prompt_history_sha256"], tokens=tuple(SampledHFToken.from_dict(token) for token in value["tokens"]), processor_order=tuple(value["processor_order"]), stop_reason=value["stop_reason"], use_cache=value["use_cache"])  # type: ignore[arg-type]


@dataclass(frozen=True)
class HFActiveBatchStep:
    token_index: int
    active_request_ids: tuple[str, ...]
    active_history_sha256s: tuple[str, ...]
    batch_shape: tuple[int, int]
    rng_before_sha256: str
    rng_after_sha256: str

    def __post_init__(self) -> None:
        if self.token_index < 0 or not self.active_request_ids or len(self.active_request_ids) != len(self.active_history_sha256s):
            raise SharedSurfaceContractError("active-batch history is malformed")
        if len(set(self.active_request_ids)) != len(self.active_request_ids) or any(not request for request in self.active_request_ids):
            raise SharedSurfaceContractError("active-batch history is malformed")
        if self.batch_shape[0] != len(self.active_request_ids) or self.batch_shape[1] < 1:
            raise SharedSurfaceContractError("active-batch shape differs from active requests")
        for history in self.active_history_sha256s:
            _digest(history, label="active history")
        _digest(self.rng_before_sha256, label="RNG transition")
        _digest(self.rng_after_sha256, label="RNG transition")

    def to_dict(self) -> dict[str, object]:
        return {"token_index": self.token_index, "active_request_ids": list(self.active_request_ids), "active_history_sha256s": list(self.active_history_sha256s), "batch_shape": list(self.batch_shape), "rng_before_sha256": self.rng_before_sha256, "rng_after_sha256": self.rng_after_sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> HFActiveBatchStep:
        _keys(value, set(cls.__dataclass_fields__), label="active batch step")
        if not all(isinstance(value[field], list) for field in ("active_request_ids", "active_history_sha256s", "batch_shape")):
            raise SharedSurfaceContractError("active batch step must contain lists")
        return cls(value["token_index"], tuple(value["active_request_ids"]), tuple(value["active_history_sha256s"]), tuple(value["batch_shape"]), value["rng_before_sha256"], value["rng_after_sha256"])  # type: ignore[arg-type]


@dataclass(frozen=True)
class HFReplayCausalGather:
    request_id: str
    token_index: int
    history_sha256: str
    chosen_token_id: int
    causal_logit_index: int

    def __post_init__(self) -> None:
        SampledHFToken(self.request_id, self.token_index, self.history_sha256, self.chosen_token_id, 0.0, 0.0, self.causal_logit_index)

    def to_dict(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> HFReplayCausalGather:
        _keys(value, set(cls.__dataclass_fields__), label="causal gather")
        return cls(**dict(value))


@dataclass(frozen=True)
class HFSharedSurfaceParityReceipt(_Sealed):
    sampled_group_sha256: str
    replayed_tokens_sha256: str
    absolute_errors: tuple[float, ...]
    max_abs_error: float
    mean_abs_error: float

    def __post_init__(self) -> None:
        _digest(self.sampled_group_sha256, label="sampled_group_sha256")
        _digest(self.replayed_tokens_sha256, label="replayed_tokens_sha256")
        if not self.absolute_errors or any(_finite(error, label="parity error") < 0 for error in self.absolute_errors):
            raise SharedSurfaceContractError("parity error distribution is invalid")
        if self.max_abs_error < 0 or self.mean_abs_error < 0:
            raise SharedSurfaceContractError("parity error cannot be negative")

    def _payload(self) -> dict[str, object]:
        return {"sampled_group_sha256": self.sampled_group_sha256, "replayed_tokens_sha256": self.replayed_tokens_sha256, "absolute_errors": list(self.absolute_errors), "max_abs_error": self.max_abs_error, "mean_abs_error": self.mean_abs_error}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> HFSharedSurfaceParityReceipt:
        _keys(value, set(cls.__dataclass_fields__) | {"content_sha256"}, label="parity receipt")
        receipt = _admit(cls(value["sampled_group_sha256"], value["replayed_tokens_sha256"], tuple(value["absolute_errors"]), value["max_abs_error"], value["mean_abs_error"]), _validate_parity)  # type: ignore[arg-type]
        _check_hash(value, receipt, label="parity receipt")
        return receipt


@dataclass(frozen=True)
class SampledHFGroup(_Sealed):
    plan: HFSharedSurfacePlan
    group_index: int
    identity: HFSharedSurfaceIdentity
    policy: HFSharedSurfacePolicy
    requests: tuple[SampledHFRequest, ...]
    active_batch_steps: tuple[HFActiveBatchStep, ...]

    def _payload(self) -> dict[str, object]:
        return {"plan": self.plan.to_dict(), "group_index": self.group_index, "identity": self.identity.to_dict(), "policy": self.policy.to_dict(), "requests": [request.to_dict() for request in self.requests], "active_batch_steps": [step.to_dict() for step in self.active_batch_steps]}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SampledHFGroup:
        _keys(value, set(cls.__dataclass_fields__) | {"content_sha256"}, label="sampled group")
        if not all(isinstance(value[field], Mapping) for field in ("plan", "identity", "policy")) or not all(isinstance(value[field], list) for field in ("requests", "active_batch_steps")):
            raise SharedSurfaceContractError("sampled group must contain strict value objects")
        if not all(isinstance(item, Mapping) for item in value["requests"] + value["active_batch_steps"]):
            raise SharedSurfaceContractError("sampled group must contain strict value objects")
        group = _admit(cls(HFSharedSurfacePlan.from_dict(value["plan"]), value["group_index"], HFSharedSurfaceIdentity.from_dict(value["identity"]), HFSharedSurfacePolicy.from_dict(value["policy"]), tuple(SampledHFRequest.from_dict(item) for item in value["requests"]), tuple(HFActiveBatchStep.from_dict(item) for item in value["active_batch_steps"])), _validate_group)  # type: ignore[arg-type]
        _check_hash(value, group, label="sampled group")
        return group


@dataclass(frozen=True)
class GradientReplayGroup(_Sealed):
    sampled_group: SampledHFGroup
    replayed_tokens: tuple[SampledHFToken, ...]
    replay_processor_order: tuple[str, str, str]
    causal_gathers: tuple[HFReplayCausalGather, ...]
    parity: HFSharedSurfaceParityReceipt

    def _payload(self) -> dict[str, object]:
        return {"sampled_group": self.sampled_group.to_dict(), "replayed_tokens": [token.to_dict() for token in self.replayed_tokens], "replay_processor_order": list(self.replay_processor_order), "causal_gathers": [gather.to_dict() for gather in self.causal_gathers], "parity": self.parity.to_dict()}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> GradientReplayGroup:
        _keys(value, set(cls.__dataclass_fields__) | {"content_sha256"}, label="gradient replay group")
        if not isinstance(value["sampled_group"], Mapping) or not isinstance(value["parity"], Mapping) or not all(isinstance(value[field], list) for field in ("replayed_tokens", "replay_processor_order", "causal_gathers")):
            raise SharedSurfaceContractError("gradient replay group must contain strict value objects")
        if not all(isinstance(item, Mapping) for item in value["replayed_tokens"] + value["causal_gathers"]):
            raise SharedSurfaceContractError("gradient replay group must contain strict value objects")
        replay = _admit(cls(SampledHFGroup.from_dict(value["sampled_group"]), tuple(SampledHFToken.from_dict(item) for item in value["replayed_tokens"]), tuple(value["replay_processor_order"]), tuple(HFReplayCausalGather.from_dict(item) for item in value["causal_gathers"]), HFSharedSurfaceParityReceipt.from_dict(value["parity"])), _validate_replay)  # type: ignore[arg-type]
        _check_hash(value, replay, label="gradient replay group")
        return replay


@dataclass(frozen=True)
class HFSharedSurfaceCloseReceipt(_Sealed):
    identity: HFSharedSurfaceIdentity
    replay_group_sha256: str
    close_reason: Literal["completed", "failed"]

    def _payload(self) -> dict[str, object]:
        return {"identity": self.identity.to_dict(), "replay_group_sha256": self.replay_group_sha256, "close_reason": self.close_reason}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> HFSharedSurfaceCloseReceipt:
        _keys(value, set(cls.__dataclass_fields__) | {"content_sha256"}, label="close receipt")
        if not isinstance(value["identity"], Mapping):
            raise SharedSurfaceContractError("close receipt identity must be strict")
        receipt = _admit(cls(HFSharedSurfaceIdentity.from_dict(value["identity"]), value["replay_group_sha256"], value["close_reason"]), _validate_close)  # type: ignore[arg-type]
        _check_hash(value, receipt, label="close receipt")
        return receipt


@dataclass(frozen=True)
class HFSharedSurfaceResourceEstimate(_Sealed):
    image_prompt_forwards: int
    replay_forwards: int
    backward_count: int
    expected_token_cap: int
    required_gpu_roles: tuple[str, str]
    output_roots: tuple[str, str]

    def _payload(self) -> dict[str, object]:
        return {"image_prompt_forwards": self.image_prompt_forwards, "replay_forwards": self.replay_forwards, "backward_count": self.backward_count, "expected_token_cap": self.expected_token_cap, "required_gpu_roles": list(self.required_gpu_roles), "output_roots": list(self.output_roots)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> HFSharedSurfaceResourceEstimate:
        _keys(value, set(cls.__dataclass_fields__) | {"content_sha256"}, label="resource estimate")
        estimate = _admit(cls(value["image_prompt_forwards"], value["replay_forwards"], value["backward_count"], value["expected_token_cap"], tuple(value["required_gpu_roles"]), tuple(value["output_roots"])), _validate_resources)  # type: ignore[arg-type]
        _check_hash(value, estimate, label="resource estimate")
        return estimate


@dataclass(frozen=True)
class HFSharedSurfaceDryRunReceipt(_Sealed):
    resource_estimate: HFSharedSurfaceResourceEstimate
    model_gpu_actions: Literal[0]
    output_roots: tuple[str, str]

    def _payload(self) -> dict[str, object]:
        return {"resource_estimate": self.resource_estimate.to_dict(), "model_gpu_actions": self.model_gpu_actions, "output_roots": list(self.output_roots)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> HFSharedSurfaceDryRunReceipt:
        _keys(value, set(cls.__dataclass_fields__) | {"content_sha256"}, label="dry-run receipt")
        if not isinstance(value["resource_estimate"], Mapping):
            raise SharedSurfaceContractError("dry-run resource estimate must be strict")
        receipt = _admit(cls(HFSharedSurfaceResourceEstimate.from_dict(value["resource_estimate"]), value["model_gpu_actions"], tuple(value["output_roots"])), _validate_dry_run)  # type: ignore[arg-type]
        _check_hash(value, receipt, label="dry-run receipt")
        return receipt


def _check_hash(value: Mapping[str, Any], loaded: _Sealed, *, label: str) -> None:
    if value["content_sha256"] != loaded.content_sha256:
        raise SharedSurfaceContractError(f"{label} content SHA-256 differs")


def _validate_group(group: SampledHFGroup) -> None:
    if type(group.plan) is not HFSharedSurfacePlan or type(group.identity) is not HFSharedSurfaceIdentity or type(group.policy) is not HFSharedSurfacePolicy or group.policy != group.plan.policy:
        raise SharedSurfaceContractError("shared surface identity/policy differs")
    if group.group_index not in range(4) or len(group.requests) != 4 or tuple(request.seed for request in group.requests) != group.plan.seed_groups[group.group_index]:
        raise SharedSurfaceContractError("seed coverage differs from frozen plan")
    if len({request.request_id for request in group.requests}) != 4:
        raise SharedSurfaceContractError("history/token lineage requires unique request identity")
    expected = {(token.request_id, token.token_index, token.history_sha256) for request in group.requests for token in request.tokens}
    observed: set[tuple[str, int, str]] = set()
    for step in group.active_batch_steps:
        for request_id, history in zip(step.active_request_ids, step.active_history_sha256s, strict=True):
            key = (request_id, step.token_index, history)
            if key not in expected:
                raise SharedSurfaceContractError("active-batch history differs from sampled trajectory")
            observed.add(key)
    if observed != expected:
        raise SharedSurfaceContractError("active-batch history does not cover sampled trajectories")


def _validate_parity(receipt: HFSharedSurfaceParityReceipt) -> None:
    if receipt.max_abs_error != max(receipt.absolute_errors) or receipt.mean_abs_error != sum(receipt.absolute_errors) / len(receipt.absolute_errors):
        raise SharedSurfaceContractError("parity error distribution differs from summary")
    if receipt.max_abs_error > 0.02 or receipt.mean_abs_error > 0.002:
        raise SharedSurfaceContractError("parity thresholds exceeded")


def _validate_replay(replay: GradientReplayGroup) -> None:
    _require_admitted(replay.sampled_group, label="sampled group")
    _require_admitted(replay.parity, label="parity receipt")
    if replay.replay_processor_order != _PROCESSOR_ORDER:
        raise SharedSurfaceContractError("processor order differs between sampling and replay")
    expected = tuple(token for request in replay.sampled_group.requests for token in request.tokens)
    if len(replay.replayed_tokens) != len(expected) or len(replay.causal_gathers) != len(expected):
        raise SharedSurfaceContractError("history/token lineage token count differs")
    for sampled, replayed, gather in zip(expected, replay.replayed_tokens, replay.causal_gathers, strict=True):
        line = (sampled.request_id, sampled.token_index, sampled.history_sha256, sampled.chosen_token_id)
        if line != (replayed.request_id, replayed.token_index, replayed.history_sha256, replayed.chosen_token_id):
            raise SharedSurfaceContractError("history/token lineage differs")
        if line != (gather.request_id, gather.token_index, gather.history_sha256, gather.chosen_token_id) or gather.causal_logit_index != replayed.causal_logit_index:
            raise SharedSurfaceContractError("causal gather differs from sampled chosen token")
    errors = tuple(_abs_error(sampled.processed_logp, replayed.processed_logp) for sampled, replayed in zip(expected, replay.replayed_tokens, strict=True))
    if replay.parity.absolute_errors != errors or replay.parity.sampled_group_sha256 != replay.sampled_group.content_sha256 or replay.parity.replayed_tokens_sha256 != _tokens_sha256(replay.replayed_tokens):
        raise SharedSurfaceContractError("parity lineage differs")
    _validate_parity(replay.parity)


def _validate_close(receipt: HFSharedSurfaceCloseReceipt) -> None:
    _digest(receipt.replay_group_sha256, label="replay_group_sha256")
    if receipt.close_reason not in ("completed", "failed"):
        raise SharedSurfaceContractError("close reason differs from contract")


def _roots(roots: object) -> tuple[str, str]:
    if not isinstance(roots, tuple) or len(roots) != 2 or any(not isinstance(root, str) or not root.startswith("/") for root in roots) or roots[0] == roots[1]:
        raise SharedSurfaceContractError("resource output roots must be two distinct absolute paths")
    return roots


def _validate_resources(estimate: HFSharedSurfaceResourceEstimate) -> None:
    if (estimate.image_prompt_forwards, estimate.replay_forwards, estimate.backward_count, estimate.expected_token_cap) != (2048, 4, 1, 8192):
        raise SharedSurfaceContractError("resource estimate differs from frozen K16 bounds")
    if estimate.required_gpu_roles != ("gpu0:shared-bf16-fa2-training", "gpu1:fp32-sdpa-audit"):
        raise SharedSurfaceContractError("resource estimate GPU roles differ from frozen vertical")
    _roots(estimate.output_roots)


def _validate_dry_run(receipt: HFSharedSurfaceDryRunReceipt) -> None:
    _require_admitted(receipt.resource_estimate, label="resource estimate")
    if receipt.model_gpu_actions != 0:
        raise SharedSurfaceContractError("dry run must perform zero model/GPU actions")
    if _roots(receipt.output_roots) != receipt.resource_estimate.output_roots:
        raise SharedSurfaceContractError("dry-run output roots differ from resource estimate")


def _tokens_sha256(tokens: Iterable[SampledHFToken]) -> str:
    return json_sha256([token.to_dict() for token in tokens])


def _abs_error(left: float, right: float) -> float:
    """Avoid a binary-rounding artifact at the frozen decimal parity boundary."""
    return float(abs(Decimal(str(left)) - Decimal(str(right))))


def admit_sampled_group(*, plan: HFSharedSurfacePlan, group_index: int, expected_identity: HFSharedSurfaceIdentity, identity: HFSharedSurfaceIdentity, policy: HFSharedSurfacePolicy, requests: tuple[SampledHFRequest, ...], active_batch_steps: tuple[HFActiveBatchStep, ...]) -> SampledHFGroup:
    if type(expected_identity) is not HFSharedSurfaceIdentity or identity != expected_identity:
        raise SharedSurfaceContractError("shared surface identity differs from expected live surface")
    return _admit(SampledHFGroup(plan, group_index, identity, policy, requests, active_batch_steps), _validate_group)


def admit_gradient_replay(*, sampled_group: SampledHFGroup, replay_identity: HFSharedSurfaceIdentity, replayed_tokens: tuple[SampledHFToken, ...], replay_processor_order: tuple[str, str, str], causal_gathers: tuple[HFReplayCausalGather, ...]) -> GradientReplayGroup:
    _require_admitted(sampled_group, label="sampled group")
    if type(replay_identity) is not HFSharedSurfaceIdentity or replay_identity != sampled_group.identity:
        raise SharedSurfaceContractError("shared surface identity differs between sampling and replay")
    expected = tuple(token for request in sampled_group.requests for token in request.tokens)
    if len(expected) != len(replayed_tokens):
        raise SharedSurfaceContractError("history/token lineage token count differs")
    errors = tuple(_abs_error(left.processed_logp, right.processed_logp) for left, right in zip(expected, replayed_tokens, strict=True))
    parity = _admit(HFSharedSurfaceParityReceipt(sampled_group.content_sha256, _tokens_sha256(replayed_tokens), errors, max(errors), sum(errors) / len(errors)), _validate_parity)
    return _admit(GradientReplayGroup(sampled_group, replayed_tokens, replay_processor_order, causal_gathers, parity), _validate_replay)


def admit_shared_surface_close(*, identity: HFSharedSurfaceIdentity, replay_group_sha256: str, close_reason: Literal["completed", "failed"]) -> HFSharedSurfaceCloseReceipt:
    return _admit(HFSharedSurfaceCloseReceipt(identity, replay_group_sha256, close_reason), _validate_close)


def repetition_penalty_then_temperature(logit: float, *, token_was_seen: bool, repetition_penalty: float, temperature: float) -> float:
    source, penalty, scale = _finite(logit, label="logit"), _finite(repetition_penalty, label="repetition penalty"), _finite(temperature, label="temperature")
    if penalty <= 0 or scale <= 0:
        raise SharedSurfaceContractError("repetition penalty and temperature must be positive")
    return (source / penalty if token_was_seen and source > 0 else source * penalty if token_was_seen else source) / scale


def estimate_image1584_k16_resources(*, output_roots: tuple[str, str]) -> HFSharedSurfaceResourceEstimate:
    return _admit(HFSharedSurfaceResourceEstimate(2048, 4, 1, 8192, ("gpu0:shared-bf16-fa2-training", "gpu1:fp32-sdpa-audit"), output_roots), _validate_resources)


def dry_run_image1584_k16(*, output_roots: tuple[str, str]) -> HFSharedSurfaceDryRunReceipt:
    estimate = estimate_image1584_k16_resources(output_roots=output_roots)
    return _admit(HFSharedSurfaceDryRunReceipt(estimate, 0, output_roots), _validate_dry_run)


def require_positive_objective_denominator(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SharedSurfaceContractError("objective denominator must be positive")
    return value


def require_finite_optimizer_delta(value: object) -> float:
    return _finite(value, label="optimizer delta")


def require_private_audit_reference(value: object) -> str:
    return _digest(value, label="private audit reference")


def require_rollback_reference(value: object) -> str:
    return _digest(value, label="rollback reference")
