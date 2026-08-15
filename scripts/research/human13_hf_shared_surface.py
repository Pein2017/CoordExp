"""CPU-only immutable contracts for the Human-13 all-HF shared surface.

This module deliberately contains neither model-framework imports nor runtime
actions.  The later HF owner supplies observations to these value contracts;
they bind those observations before a replay can be used by a loss.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import isfinite
from typing import Any, Iterable, Literal, Mapping

from src.artifacts.json_values import json_sha256


_DIGEST_LENGTH = 64
_SEED_GROUPS = (
    (35001, 35002, 35003, 35004),
    (35005, 35006, 35007, 35008),
    (35009, 35010, 35011, 35012),
    (35013, 35014, 35015, 35016),
)
_PROCESSOR_ORDER = ("repetition_penalty", "temperature", "top_p")


class SharedSurfaceContractError(ValueError):
    """A value cannot cross the shared sampling/replay admission boundary."""


def _require_digest(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or len(value) != _DIGEST_LENGTH:
        raise SharedSurfaceContractError(f"{field_name} must be a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise SharedSurfaceContractError(
            f"{field_name} must be a SHA-256 digest"
        ) from exc
    return value


def _require_exact_keys(value: Mapping[str, Any], keys: set[str], *, label: str) -> None:
    if set(value) != keys:
        raise SharedSurfaceContractError(f"{label} fields differ from canonical schema")


def _require_finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise SharedSurfaceContractError(f"{label} must be finite")
    return float(value)


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
        for name in (
            "checkpoint_payload_sha256",
            "parameter_state_sha256",
            "adapter_sha256",
            "embedding_delta_sha256",
            "tokenizer_sha256",
            "prompt_sha256",
            "image_sha256",
        ):
            _require_digest(getattr(self, name), field_name=name)
        if isinstance(self.model_object_id, bool) or not isinstance(self.model_object_id, int):
            raise SharedSurfaceContractError("model object identity must be an integer")
        if self.dtype != "bfloat16" or self.attention_backend != "flash_attention_2":
            raise SharedSurfaceContractError("shared surface identity requires BF16/FA2")
        if self.model_mode != "eval":
            raise SharedSurfaceContractError("shared surface identity requires eval mode")
        if self.use_cache is not False:
            raise SharedSurfaceContractError("cache is forbidden")

    def to_dict(self) -> dict[str, object]:
        return {
            "checkpoint_payload_sha256": self.checkpoint_payload_sha256,
            "model_object_id": self.model_object_id,
            "parameter_state_sha256": self.parameter_state_sha256,
            "adapter_sha256": self.adapter_sha256,
            "embedding_delta_sha256": self.embedding_delta_sha256,
            "dtype": self.dtype,
            "attention_backend": self.attention_backend,
            "model_mode": self.model_mode,
            "tokenizer_sha256": self.tokenizer_sha256,
            "prompt_sha256": self.prompt_sha256,
            "image_sha256": self.image_sha256,
            "use_cache": self.use_cache,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> HFSharedSurfaceIdentity:
        _require_exact_keys(value, set(cls.__dataclass_fields__) - {"_admitted"}, label="identity")
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
        if _require_finite(self.repetition_penalty, label="repetition penalty") != 1.0:
            raise SharedSurfaceContractError("initial repetition penalty must be 1.0")
        if self.repetition_penalty_before_temperature is not True:
            raise SharedSurfaceContractError("processor order requires RP-before-temperature")
        if self.prompt_inclusive_history is not True:
            raise SharedSurfaceContractError("history must include prompt")
        if _require_finite(self.temperature, label="temperature") != 0.4:
            raise SharedSurfaceContractError("initial temperature must be 0.4")
        if _require_finite(self.top_p, label="top_p") != 1.0 or self.top_k is not None:
            raise SharedSurfaceContractError("initial top-p/top-k policy differs")
        if self.max_new_tokens != 512 or self.stop_token != "<|im_end|>":
            raise SharedSurfaceContractError("initial cap or stop policy differs")
        if self.use_cache is not False:
            raise SharedSurfaceContractError("cache is forbidden")

    def to_dict(self) -> dict[str, object]:
        return {
            "repetition_penalty": self.repetition_penalty,
            "repetition_penalty_before_temperature": self.repetition_penalty_before_temperature,
            "prompt_inclusive_history": self.prompt_inclusive_history,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_new_tokens": self.max_new_tokens,
            "stop_token": self.stop_token,
            "use_cache": self.use_cache,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> HFSharedSurfacePolicy:
        _require_exact_keys(value, set(cls.__dataclass_fields__), label="policy")
        return cls(**dict(value))


@dataclass(frozen=True)
class HFSharedSurfacePlan:
    image_id: Literal[1584]
    seed_groups: tuple[tuple[int, int, int, int], ...]
    policy: HFSharedSurfacePolicy

    def __post_init__(self) -> None:
        if self.image_id != 1584 or self.seed_groups != _SEED_GROUPS:
            raise SharedSurfaceContractError("seed coverage differs from frozen image-1584 K16 plan")
        if type(self.policy) is not HFSharedSurfacePolicy:
            raise SharedSurfaceContractError("plan policy must be exact")

    def to_dict(self) -> dict[str, object]:
        return {
            "image_id": self.image_id,
            "seed_groups": [list(group) for group in self.seed_groups],
            "policy": self.policy.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> HFSharedSurfacePlan:
        _require_exact_keys(value, {"image_id", "seed_groups", "policy"}, label="plan")
        groups = value["seed_groups"]
        if not isinstance(groups, list) or any(not isinstance(group, list) for group in groups):
            raise SharedSurfaceContractError("seed coverage differs from frozen image-1584 K16 plan")
        policy = value["policy"]
        if not isinstance(policy, Mapping):
            raise SharedSurfaceContractError("plan policy must be a mapping")
        return cls(
            image_id=value["image_id"],
            seed_groups=tuple(tuple(group) for group in groups),  # type: ignore[arg-type]
            policy=HFSharedSurfacePolicy.from_dict(policy),
        )


def plan_image1584_k16() -> HFSharedSurfacePlan:
    """Return the only acquisition plan permitted by the initial vertical."""
    return HFSharedSurfacePlan(
        image_id=1584,
        seed_groups=_SEED_GROUPS,
        policy=HFSharedSurfacePolicy(
            repetition_penalty=1.0,
            repetition_penalty_before_temperature=True,
            prompt_inclusive_history=True,
            temperature=0.4,
            top_p=1.0,
            top_k=None,
            max_new_tokens=512,
            stop_token="<|im_end|>",
            use_cache=False,
        ),
    )


@dataclass(frozen=True)
class SampledHFToken:
    token_index: int
    history_sha256: str
    chosen_token_id: int
    processed_logp: float
    request_id: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.token_index, bool) or not isinstance(self.token_index, int) or self.token_index < 0:
            raise SharedSurfaceContractError("history/token lineage requires non-negative token indexes")
        _require_digest(self.history_sha256, field_name="history_sha256")
        if isinstance(self.chosen_token_id, bool) or not isinstance(self.chosen_token_id, int):
            raise SharedSurfaceContractError("history/token lineage requires integer chosen tokens")
        _require_finite(self.processed_logp, label="processed log probability")
        if not isinstance(self.request_id, str):
            raise SharedSurfaceContractError("history/token lineage requires request identity")

    def to_dict(self) -> dict[str, object]:
        return {
            "token_index": self.token_index,
            "history_sha256": self.history_sha256,
            "chosen_token_id": self.chosen_token_id,
            "processed_logp": self.processed_logp,
            "request_id": self.request_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SampledHFToken:
        _require_exact_keys(value, set(cls.__dataclass_fields__), label="sampled token")
        return cls(**dict(value))


@dataclass(frozen=True)
class SampledHFRequest:
    request_id: str
    image_id: int
    seed: int
    prompt_history_sha256: str
    tokens: tuple[SampledHFToken, ...]
    processor_order: tuple[str, str, str]
    use_cache: Literal[False]

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id:
            raise SharedSurfaceContractError("history/token lineage requires request identity")
        if self.image_id != 1584 or isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise SharedSurfaceContractError("seed coverage differs from frozen plan")
        _require_digest(self.prompt_history_sha256, field_name="prompt_history_sha256")
        if self.processor_order != _PROCESSOR_ORDER:
            raise SharedSurfaceContractError("processor order differs from frozen policy")
        if self.use_cache is not False:
            raise SharedSurfaceContractError("cache is forbidden")
        if not self.tokens or tuple(token.token_index for token in self.tokens) != tuple(range(len(self.tokens))):
            raise SharedSurfaceContractError("history/token lineage must be contiguous and non-empty")
        if any(token.request_id not in ("", self.request_id) for token in self.tokens):
            raise SharedSurfaceContractError("history/token lineage request identity differs")

    def with_bound_token_requests(self) -> SampledHFRequest:
        return replace(
            self,
            tokens=tuple(
                token if token.request_id == self.request_id else replace(token, request_id=self.request_id)
                for token in self.tokens
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "image_id": self.image_id,
            "seed": self.seed,
            "prompt_history_sha256": self.prompt_history_sha256,
            "tokens": [token.to_dict() for token in self.tokens],
            "processor_order": list(self.processor_order),
            "use_cache": self.use_cache,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SampledHFRequest:
        _require_exact_keys(value, set(cls.__dataclass_fields__), label="sampled request")
        tokens = value["tokens"]
        order = value["processor_order"]
        if not isinstance(tokens, list) or not isinstance(order, list):
            raise SharedSurfaceContractError("sampled request token/order fields must be lists")
        if not all(isinstance(token, Mapping) for token in tokens):
            raise SharedSurfaceContractError("sampled request must contain strict value objects")
        return cls(
            request_id=value["request_id"],
            image_id=value["image_id"],
            seed=value["seed"],
            prompt_history_sha256=value["prompt_history_sha256"],
            tokens=tuple(SampledHFToken.from_dict(token) for token in tokens),
            processor_order=tuple(order),  # type: ignore[arg-type]
            use_cache=value["use_cache"],
        )


@dataclass(frozen=True)
class HFSharedSurfaceParityReceipt:
    sampled_group_sha256: str
    replayed_tokens_sha256: str
    token_count: int
    max_abs_error: float
    mean_abs_error: float
    _admitted: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _require_digest(self.sampled_group_sha256, field_name="sampled_group_sha256")
        _require_digest(self.replayed_tokens_sha256, field_name="replayed_tokens_sha256")
        if isinstance(self.token_count, bool) or not isinstance(self.token_count, int) or self.token_count <= 0:
            raise SharedSurfaceContractError("parity token count must be positive")
        _require_finite(self.max_abs_error, label="parity error")
        _require_finite(self.mean_abs_error, label="parity error")
        if self.max_abs_error < 0 or self.mean_abs_error < 0:
            raise SharedSurfaceContractError("parity error cannot be negative")

    def _payload(self) -> dict[str, object]:
        return {
            "sampled_group_sha256": self.sampled_group_sha256,
            "replayed_tokens_sha256": self.replayed_tokens_sha256,
            "token_count": self.token_count,
            "max_abs_error": self.max_abs_error,
            "mean_abs_error": self.mean_abs_error,
        }

    @property
    def content_sha256(self) -> str:
        _require_admitted(self, label="parity receipt")
        return json_sha256(self._payload())

    def to_dict(self) -> dict[str, object]:
        _require_admitted(self, label="parity receipt")
        return self._payload() | {"content_sha256": self.content_sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> HFSharedSurfaceParityReceipt:
        _require_exact_keys(value, set(cls.__dataclass_fields__) - {"_admitted"} | {"content_sha256"}, label="parity receipt")
        payload = {key: value[key] for key in cls.__dataclass_fields__ if key != "_admitted"}
        receipt = _admit(cls(**payload), _validate_parity_thresholds)
        if value["content_sha256"] != receipt.content_sha256:
            raise SharedSurfaceContractError("parity receipt content SHA-256 differs")
        return receipt


@dataclass(frozen=True)
class SampledHFGroup:
    plan: HFSharedSurfacePlan
    group_index: int
    identity: HFSharedSurfaceIdentity
    policy: HFSharedSurfacePolicy
    requests: tuple[SampledHFRequest, ...]
    _admitted: bool = field(default=False, repr=False, compare=False)

    def _payload(self) -> dict[str, object]:
        return {
            "plan": self.plan.to_dict(),
            "group_index": self.group_index,
            "identity": self.identity.to_dict(),
            "policy": self.policy.to_dict(),
            "requests": [request.to_dict() for request in self.requests],
        }

    @property
    def content_sha256(self) -> str:
        _require_admitted(self, label="sampled group")
        return json_sha256(self._payload())

    def to_dict(self) -> dict[str, object]:
        _require_admitted(self, label="sampled group")
        return self._payload() | {"content_sha256": self.content_sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SampledHFGroup:
        _require_exact_keys(value, set(cls.__dataclass_fields__) - {"_admitted"} | {"content_sha256"}, label="sampled group")
        plan = value["plan"]
        identity = value["identity"]
        policy = value["policy"]
        requests = value["requests"]
        if not all(isinstance(item, Mapping) for item in (plan, identity, policy)) or not isinstance(requests, list):
            raise SharedSurfaceContractError("sampled group must contain strict value objects")
        if not all(isinstance(request, Mapping) for request in requests):
            raise SharedSurfaceContractError("sampled group must contain strict value objects")
        loaded = _admit(
            cls(
                plan=HFSharedSurfacePlan.from_dict(plan),
                group_index=value["group_index"],
                identity=HFSharedSurfaceIdentity.from_dict(identity),
                policy=HFSharedSurfacePolicy.from_dict(policy),
                requests=tuple(SampledHFRequest.from_dict(request) for request in requests),
            ),
            _validate_sampled_group,
        )
        if value["content_sha256"] != loaded.content_sha256:
            raise SharedSurfaceContractError("sampled group content SHA-256 differs")
        return loaded


@dataclass(frozen=True)
class GradientReplayGroup:
    sampled_group: SampledHFGroup
    replayed_tokens: tuple[SampledHFToken, ...]
    parity: HFSharedSurfaceParityReceipt
    _admitted: bool = field(default=False, repr=False, compare=False)

    def _payload(self) -> dict[str, object]:
        return {
            "sampled_group": self.sampled_group.to_dict(),
            "replayed_tokens": [token.to_dict() for token in self.replayed_tokens],
            "parity": self.parity.to_dict(),
        }

    @property
    def content_sha256(self) -> str:
        _require_admitted(self, label="gradient replay group")
        return json_sha256(self._payload())

    def to_dict(self) -> dict[str, object]:
        _require_admitted(self, label="gradient replay group")
        return self._payload() | {"content_sha256": self.content_sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> GradientReplayGroup:
        _require_exact_keys(value, set(cls.__dataclass_fields__) - {"_admitted"} | {"content_sha256"}, label="gradient replay group")
        sampled = value["sampled_group"]
        parity = value["parity"]
        replayed = value["replayed_tokens"]
        if not isinstance(sampled, Mapping) or not isinstance(parity, Mapping) or not isinstance(replayed, list):
            raise SharedSurfaceContractError("gradient replay group must contain strict value objects")
        if not all(isinstance(token, Mapping) for token in replayed):
            raise SharedSurfaceContractError("gradient replay group must contain strict value objects")
        loaded = _admit(
            cls(
                sampled_group=SampledHFGroup.from_dict(sampled),
                replayed_tokens=tuple(SampledHFToken.from_dict(token) for token in replayed),
                parity=HFSharedSurfaceParityReceipt.from_dict(parity),
            ),
            _validate_replay_group,
        )
        if value["content_sha256"] != loaded.content_sha256:
            raise SharedSurfaceContractError("gradient replay group content SHA-256 differs")
        return loaded


def _require_admitted(value: object, *, label: str) -> None:
    if getattr(value, "_admitted", False) is not True:
        raise SharedSurfaceContractError(f"{label} was not admitted through the shared-surface choke point")


def _admit(value: Any, validator: Any) -> Any:
    """The sole path that marks a scientific value usable by consumers."""
    validator(value)
    return replace(value, _admitted=True)


def _validate_sampled_group(group: SampledHFGroup) -> None:
    if type(group.plan) is not HFSharedSurfacePlan or type(group.identity) is not HFSharedSurfaceIdentity:
        raise SharedSurfaceContractError("shared surface identity must use exact value types")
    if type(group.policy) is not HFSharedSurfacePolicy or group.policy != group.plan.policy:
        raise SharedSurfaceContractError("processor order/policy differs from frozen plan")
    if isinstance(group.group_index, bool) or group.group_index not in range(len(group.plan.seed_groups)):
        raise SharedSurfaceContractError("seed coverage differs from frozen plan")
    expected_seeds = group.plan.seed_groups[group.group_index]
    if len(group.requests) != 4 or tuple(request.seed for request in group.requests) != expected_seeds:
        raise SharedSurfaceContractError("seed coverage differs from frozen plan")
    if len({request.request_id for request in group.requests}) != 4:
        raise SharedSurfaceContractError("history/token lineage requires unique request identities")
    if any(type(request) is not SampledHFRequest for request in group.requests):
        raise SharedSurfaceContractError("history/token lineage requires exact request values")
    if any(request.image_id != group.plan.image_id for request in group.requests):
        raise SharedSurfaceContractError("shared surface identity image differs from plan")
    if any(request.processor_order != _PROCESSOR_ORDER for request in group.requests):
        raise SharedSurfaceContractError("processor order differs from frozen policy")
    if any(request.use_cache is not False for request in group.requests):
        raise SharedSurfaceContractError("cache is forbidden")


def _validate_parity_thresholds(receipt: HFSharedSurfaceParityReceipt) -> None:
    if receipt.max_abs_error > 0.02 or receipt.mean_abs_error > 0.002:
        raise SharedSurfaceContractError("parity thresholds exceeded")


def _validate_replay_group(replay: GradientReplayGroup) -> None:
    _require_admitted(replay.sampled_group, label="sampled group")
    _require_admitted(replay.parity, label="parity receipt")
    expected = tuple(token for request in replay.sampled_group.requests for token in request.tokens)
    _validate_replay_lineage(expected, replay.replayed_tokens)
    if replay.parity.sampled_group_sha256 != replay.sampled_group.content_sha256:
        raise SharedSurfaceContractError("shared surface lineage differs from sampled group")
    if replay.parity.replayed_tokens_sha256 != _tokens_sha256(replay.replayed_tokens):
        raise SharedSurfaceContractError("history/token lineage digest differs")
    _validate_parity_thresholds(replay.parity)


def _validate_replay_lineage(
    expected: tuple[SampledHFToken, ...], replayed: tuple[SampledHFToken, ...]
) -> None:
    if len(expected) != len(replayed):
        raise SharedSurfaceContractError("history/token lineage token count differs")
    for sampled, replay in zip(expected, replayed, strict=True):
        if (
            sampled.request_id,
            sampled.token_index,
            sampled.history_sha256,
            sampled.chosen_token_id,
        ) != (
            replay.request_id,
            replay.token_index,
            replay.history_sha256,
            replay.chosen_token_id,
        ):
            raise SharedSurfaceContractError("history/token lineage differs")
        _require_finite(replay.processed_logp, label="replayed processed log probability")


def _tokens_sha256(tokens: Iterable[SampledHFToken]) -> str:
    return json_sha256([token.to_dict() for token in tokens])


def admit_sampled_group(
    *,
    plan: HFSharedSurfacePlan,
    group_index: int,
    expected_identity: HFSharedSurfaceIdentity,
    identity: HFSharedSurfaceIdentity,
    policy: HFSharedSurfacePolicy,
    requests: tuple[SampledHFRequest, ...],
) -> SampledHFGroup:
    """Bind one observed four-request acquisition to its expected live surface."""
    if type(expected_identity) is not HFSharedSurfaceIdentity or identity != expected_identity:
        raise SharedSurfaceContractError("shared surface identity differs from expected live surface")
    if identity.use_cache is not False:
        raise SharedSurfaceContractError("cache is forbidden")
    bound_requests = tuple(request.with_bound_token_requests() for request in requests)
    return _admit(
        SampledHFGroup(
            plan=plan,
            group_index=group_index,
            identity=identity,
            policy=policy,
            requests=bound_requests,
        ),
        _validate_sampled_group,
    )


def admit_gradient_replay(
    *,
    sampled_group: SampledHFGroup,
    replay_identity: HFSharedSurfaceIdentity,
    replayed_tokens: tuple[SampledHFToken, ...],
) -> GradientReplayGroup:
    """Admit one replay only after exact lineage and frozen numerical parity."""
    _require_admitted(sampled_group, label="sampled group")
    if type(replay_identity) is not HFSharedSurfaceIdentity or replay_identity != sampled_group.identity:
        raise SharedSurfaceContractError("shared surface identity differs between sampling and replay")
    expected = tuple(token for request in sampled_group.requests for token in request.tokens)
    _validate_replay_lineage(expected, replayed_tokens)
    errors = tuple(abs(sampled.processed_logp - replay.processed_logp) for sampled, replay in zip(expected, replayed_tokens, strict=True))
    receipt = _admit(
        HFSharedSurfaceParityReceipt(
            sampled_group_sha256=sampled_group.content_sha256,
            replayed_tokens_sha256=_tokens_sha256(replayed_tokens),
            token_count=len(errors),
            max_abs_error=max(errors),
            mean_abs_error=sum(errors) / len(errors),
        ),
        _validate_parity_thresholds,
    )
    return _admit(
        GradientReplayGroup(
            sampled_group=sampled_group,
            replayed_tokens=replayed_tokens,
            parity=receipt,
        ),
        _validate_replay_group,
    )


def repetition_penalty_then_temperature(
    logit: float, *, token_was_seen: bool, repetition_penalty: float, temperature: float
) -> float:
    """Apply the HF sign-aware repetition transform before temperature scaling."""
    source = _require_finite(logit, label="logit")
    penalty = _require_finite(repetition_penalty, label="repetition penalty")
    scale = _require_finite(temperature, label="temperature")
    if penalty <= 0 or scale <= 0:
        raise SharedSurfaceContractError("repetition penalty and temperature must be positive")
    processed = source / penalty if token_was_seen and source > 0 else source * penalty if token_was_seen else source
    return processed / scale


@dataclass(frozen=True)
class HFSharedSurfaceResourceEstimate:
    """Static upper bounds for the first vertical; this value performs no I/O."""

    image_prompt_forwards: int
    replay_forwards: int
    backward_count: int
    expected_token_cap: int
    required_gpu_roles: tuple[str, str]
    output_roots: tuple[str, str]

    def __post_init__(self) -> None:
        if (
            self.image_prompt_forwards != 2048
            or self.replay_forwards != 4
            or self.backward_count != 1
            or self.expected_token_cap != 8192
        ):
            raise SharedSurfaceContractError("resource estimate differs from frozen K16 bounds")
        if self.required_gpu_roles != (
            "gpu0:shared-bf16-fa2-training",
            "gpu1:fp32-sdpa-audit",
        ):
            raise SharedSurfaceContractError("resource estimate GPU roles differ from frozen vertical")
        _validate_output_roots(self.output_roots)

    def to_dict(self) -> dict[str, object]:
        return {
            "image_prompt_forwards": self.image_prompt_forwards,
            "replay_forwards": self.replay_forwards,
            "backward_count": self.backward_count,
            "expected_token_cap": self.expected_token_cap,
            "required_gpu_roles": list(self.required_gpu_roles),
            "output_roots": list(self.output_roots),
        }


@dataclass(frozen=True)
class HFSharedSurfaceDryRunReceipt:
    """A resource-only preflight receipt that explicitly records zero execution."""

    resource_estimate: HFSharedSurfaceResourceEstimate
    model_gpu_actions: Literal[0]
    output_roots: tuple[str, str]

    def __post_init__(self) -> None:
        if type(self.resource_estimate) is not HFSharedSurfaceResourceEstimate:
            raise SharedSurfaceContractError("dry-run resource estimate must be exact")
        if self.model_gpu_actions != 0:
            raise SharedSurfaceContractError("dry run must perform zero model/GPU actions")
        _validate_output_roots(self.output_roots)
        if self.output_roots != self.resource_estimate.output_roots:
            raise SharedSurfaceContractError("dry-run output roots differ from resource estimate")

    def to_dict(self) -> dict[str, object]:
        return {
            "resource_estimate": self.resource_estimate.to_dict(),
            "model_gpu_actions": self.model_gpu_actions,
            "output_roots": list(self.output_roots),
        }


def _validate_output_roots(output_roots: object) -> None:
    if (
        not isinstance(output_roots, tuple)
        or len(output_roots) != 2
        or any(not isinstance(root, str) or not root.startswith("/") for root in output_roots)
        or output_roots[0] == output_roots[1]
    ):
        raise SharedSurfaceContractError("resource output roots must be two distinct absolute paths")


def estimate_image1584_k16_resources(
    *, output_roots: tuple[str, str]
) -> HFSharedSurfaceResourceEstimate:
    """Bound four no-cache batch-four groups and one replay/update without executing."""
    plan = plan_image1584_k16()
    return HFSharedSurfaceResourceEstimate(
        # At most 512 full-history forwards per logical group, one batched call each.
        image_prompt_forwards=len(plan.seed_groups) * plan.policy.max_new_tokens,
        replay_forwards=len(plan.seed_groups),
        backward_count=1,
        expected_token_cap=sum(len(group) for group in plan.seed_groups) * plan.policy.max_new_tokens,
        required_gpu_roles=(
            "gpu0:shared-bf16-fa2-training",
            "gpu1:fp32-sdpa-audit",
        ),
        output_roots=output_roots,
    )


def dry_run_image1584_k16(*, output_roots: tuple[str, str]) -> HFSharedSurfaceDryRunReceipt:
    """Return a receipt for a zero-model/GPU-action preflight only."""
    estimate = estimate_image1584_k16_resources(output_roots=output_roots)
    return HFSharedSurfaceDryRunReceipt(
        resource_estimate=estimate,
        model_gpu_actions=0,
        output_roots=output_roots,
    )
