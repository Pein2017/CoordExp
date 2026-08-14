"""Immutable, content-addressed policy evidence for the Human-13 RP screen."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Mapping


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PROCESSOR_ORDER = ("repetition_penalty", "temperature", "log_softmax")
_FIXED_PER_TOKEN_NATS = 0.02
_FIXED_GROUP_MEAN_NATS = 0.002


def _sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _token_ids(value: tuple[int, ...] | list[int], *, field: str) -> tuple[int, ...]:
    tokens = tuple(value)
    if any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in tokens):
        raise ValueError(f"{field} must contain nonnegative integer token ids")
    return tokens


def _nonempty(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a nonempty string")
    return value


def _digest(value: str, *, field: str) -> str:
    value = _nonempty(value, field=field)
    if not _SHA256.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True)
class ArtifactIdentity:
    """The immutable source/request identity plus its complete token history."""

    source_sha256: str
    manifest_sha256: str
    request_id: str
    model_id: str
    tokenizer_id: str
    processor_id: str
    prompt_token_ids: tuple[int, ...]
    generated_token_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_sha256", _digest(self.source_sha256, field="source_sha256"))
        object.__setattr__(self, "manifest_sha256", _digest(self.manifest_sha256, field="manifest_sha256"))
        for field in ("request_id", "model_id", "tokenizer_id", "processor_id"):
            object.__setattr__(self, field, _nonempty(getattr(self, field), field=field))
        prompt = _token_ids(self.prompt_token_ids, field="prompt_token_ids")
        if not prompt:
            raise ValueError("prompt_token_ids must not be empty")
        object.__setattr__(self, "prompt_token_ids", prompt)
        object.__setattr__(
            self,
            "generated_token_ids",
            _token_ids(self.generated_token_ids, field="generated_token_ids"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "human13_k_trajectory_artifact_identity.v1",
            "source_sha256": self.source_sha256,
            "manifest_sha256": self.manifest_sha256,
            "request_id": self.request_id,
            "model_id": self.model_id,
            "tokenizer_id": self.tokenizer_id,
            "processor_id": self.processor_id,
            "prompt_token_ids": list(self.prompt_token_ids),
            "generated_token_ids": list(self.generated_token_ids),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ArtifactIdentity":
        if value.get("schema_version") != "human13_k_trajectory_artifact_identity.v1":
            raise ValueError("artifact identity schema_version differs")
        return cls(
            source_sha256=value["source_sha256"],
            manifest_sha256=value["manifest_sha256"],
            request_id=value["request_id"],
            model_id=value["model_id"],
            tokenizer_id=value["tokenizer_id"],
            processor_id=value["processor_id"],
            prompt_token_ids=tuple(value["prompt_token_ids"]),
            generated_token_ids=tuple(value["generated_token_ids"]),
        )

    @property
    def content_sha256(self) -> str:
        return _sha256(self.to_dict())


@dataclass(frozen=True)
class PolicyContract:
    """The sealed sampling policy; RP, temperature, and processor order are literal."""

    identity: ArtifactIdentity
    repetition_penalty: float
    temperature: float
    natural_stop_token_id: int
    max_new_tokens: int
    processor_order: tuple[str, ...] = _PROCESSOR_ORDER
    sampler_backend_id: str = ""
    top_p: float = 1.0
    top_k: int | None = None
    n: int = 1
    min_new_tokens: int = 0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    ignore_eos: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.identity, ArtifactIdentity):
            raise ValueError("identity must be an ArtifactIdentity")
        rp = float(self.repetition_penalty)
        temperature = float(self.temperature)
        if rp not in (1.0, 1.10) or not math.isfinite(rp):
            raise ValueError("repetition_penalty must be exactly 1.0 or 1.10")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if self.processor_order != _PROCESSOR_ORDER:
            raise ValueError("processor_order must be repetition_penalty, temperature, log_softmax")
        if isinstance(self.natural_stop_token_id, bool) or not isinstance(self.natural_stop_token_id, int) or self.natural_stop_token_id < 0:
            raise ValueError("natural_stop_token_id must be a nonnegative integer")
        if isinstance(self.max_new_tokens, bool) or not isinstance(self.max_new_tokens, int) or self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be a positive integer")
        if isinstance(self.min_new_tokens, bool) or not isinstance(self.min_new_tokens, int) or not 0 <= self.min_new_tokens <= self.max_new_tokens:
            raise ValueError("min_new_tokens must be an integer from zero to max_new_tokens")
        if isinstance(self.n, bool) or not isinstance(self.n, int) or self.n <= 0:
            raise ValueError("n must be a positive integer")
        top_p = float(self.top_p)
        if not math.isfinite(top_p) or not 0 < top_p <= 1:
            raise ValueError("top_p must be finite and in (0, 1]")
        if self.top_k is not None and (
            isinstance(self.top_k, bool) or not isinstance(self.top_k, int) or self.top_k <= 0
        ):
            raise ValueError("top_k must be None or a positive integer")
        frequency_penalty = float(self.frequency_penalty)
        presence_penalty = float(self.presence_penalty)
        if not math.isfinite(frequency_penalty) or not math.isfinite(presence_penalty):
            raise ValueError("frequency_penalty and presence_penalty must be finite")
        if not isinstance(self.ignore_eos, bool):
            raise ValueError("ignore_eos must be a boolean")
        object.__setattr__(self, "sampler_backend_id", _nonempty(self.sampler_backend_id, field="sampler_backend_id"))
        object.__setattr__(self, "repetition_penalty", rp)
        object.__setattr__(self, "temperature", temperature)
        object.__setattr__(self, "processor_order", tuple(self.processor_order))
        object.__setattr__(self, "top_p", top_p)
        object.__setattr__(self, "frequency_penalty", frequency_penalty)
        object.__setattr__(self, "presence_penalty", presence_penalty)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "human13_rp_policy_contract.v1",
            "identity": self.identity.to_dict(),
            "repetition_penalty": self.repetition_penalty,
            "temperature": self.temperature,
            "natural_stop_token_id": self.natural_stop_token_id,
            "max_new_tokens": self.max_new_tokens,
            "processor_order": list(self.processor_order),
            "sampler_backend_id": self.sampler_backend_id,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "n": self.n,
            "min_new_tokens": self.min_new_tokens,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "ignore_eos": self.ignore_eos,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PolicyContract":
        if value.get("schema_version") != "human13_rp_policy_contract.v1":
            raise ValueError("policy contract schema_version differs")
        return cls(
            identity=ArtifactIdentity.from_dict(value["identity"]),
            repetition_penalty=value["repetition_penalty"],
            temperature=value["temperature"],
            natural_stop_token_id=value["natural_stop_token_id"],
            max_new_tokens=value["max_new_tokens"],
            processor_order=tuple(value["processor_order"]),
            sampler_backend_id=value["sampler_backend_id"],
            top_p=value["top_p"],
            top_k=value["top_k"],
            n=value["n"],
            min_new_tokens=value["min_new_tokens"],
            frequency_penalty=value["frequency_penalty"],
            presence_penalty=value["presence_penalty"],
            ignore_eos=value["ignore_eos"],
        )

    @property
    def content_sha256(self) -> str:
        return _sha256(self.to_dict())


@dataclass(frozen=True)
class GeneratedTokenEvidence:
    identity: ArtifactIdentity
    policy_contract_sha256: str
    token_index: int
    history_token_ids: tuple[int, ...]
    chosen_token_id: int
    processed_logprob: float

    def __post_init__(self) -> None:
        if not isinstance(self.identity, ArtifactIdentity):
            raise ValueError("identity must be an ArtifactIdentity")
        object.__setattr__(self, "policy_contract_sha256", _digest(self.policy_contract_sha256, field="policy_contract_sha256"))
        if isinstance(self.token_index, bool) or not isinstance(self.token_index, int) or self.token_index < 0:
            raise ValueError("token_index must be a nonnegative integer")
        object.__setattr__(self, "history_token_ids", _token_ids(self.history_token_ids, field="history_token_ids"))
        if isinstance(self.chosen_token_id, bool) or not isinstance(self.chosen_token_id, int) or self.chosen_token_id < 0:
            raise ValueError("chosen_token_id must be a nonnegative integer")
        score = float(self.processed_logprob)
        if not math.isfinite(score):
            raise ValueError("processed_logprob must be finite")
        object.__setattr__(self, "processed_logprob", score)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "human13_generated_token_evidence.v1",
            "identity": self.identity.to_dict(),
            "policy_contract_sha256": self.policy_contract_sha256,
            "token_index": self.token_index,
            "history_token_ids": list(self.history_token_ids),
            "chosen_token_id": self.chosen_token_id,
            "processed_logprob": self.processed_logprob,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GeneratedTokenEvidence":
        if value.get("schema_version") != "human13_generated_token_evidence.v1":
            raise ValueError("generated token evidence schema_version differs")
        return cls(
            identity=ArtifactIdentity.from_dict(value["identity"]),
            policy_contract_sha256=value["policy_contract_sha256"],
            token_index=value["token_index"],
            history_token_ids=tuple(value["history_token_ids"]),
            chosen_token_id=value["chosen_token_id"],
            processed_logprob=value["processed_logprob"],
        )

    @property
    def content_sha256(self) -> str:
        """Token evidence is independently addressable and trajectory-addressed."""
        return _sha256(self.to_dict())


def _same_static_identity(left: ArtifactIdentity, right: ArtifactIdentity) -> bool:
    return (
        left.source_sha256,
        left.manifest_sha256,
        left.request_id,
        left.model_id,
        left.tokenizer_id,
        left.processor_id,
        left.prompt_token_ids,
    ) == (
        right.source_sha256,
        right.manifest_sha256,
        right.request_id,
        right.model_id,
        right.tokenizer_id,
        right.processor_id,
        right.prompt_token_ids,
    )


@dataclass(frozen=True)
class CompleteTrajectoryEvidence:
    identity: ArtifactIdentity
    policy_contract: PolicyContract
    generated_tokens: tuple[GeneratedTokenEvidence, ...]
    terminal_kind: str

    def __post_init__(self) -> None:
        if not isinstance(self.identity, ArtifactIdentity) or not isinstance(self.policy_contract, PolicyContract):
            raise ValueError("trajectory identity and policy_contract must be sealed records")
        if not _same_static_identity(self.identity, self.policy_contract.identity):
            raise ValueError("trajectory identity differs from policy contract identity")
        tokens = tuple(self.generated_tokens)
        if len(tokens) != len(self.identity.generated_token_ids):
            raise ValueError("generated token evidence count differs from generated history")
        if not tokens:
            raise ValueError("trajectory must contain at least one generated token")
        if len(tokens) > self.policy_contract.max_new_tokens:
            raise ValueError("generated token evidence exceeds max_new_tokens")
        expected_prefix = self.identity.prompt_token_ids
        for index, token in enumerate(tokens):
            if not isinstance(token, GeneratedTokenEvidence):
                raise ValueError("generated_tokens must contain GeneratedTokenEvidence")
            if token.identity != self.identity or token.policy_contract_sha256 != self.policy_contract.content_sha256:
                raise ValueError("generated token evidence identity or contract differs")
            if token.token_index != index or token.chosen_token_id != self.identity.generated_token_ids[index]:
                raise ValueError("generated token evidence index or chosen token differs")
            if token.history_token_ids != (*expected_prefix, *self.identity.generated_token_ids[:index]):
                raise ValueError("generated token evidence history is not the complete causal history")
        if self.terminal_kind == "natural_stop":
            if self.identity.generated_token_ids[-1] != self.policy_contract.natural_stop_token_id:
                raise ValueError("natural stop must end with the sealed natural stop token")
            if self.policy_contract.natural_stop_token_id in self.identity.generated_token_ids[:-1]:
                raise ValueError("natural stop token may occur only at the natural stop")
        elif self.terminal_kind == "cap_stop":
            if len(tokens) != self.policy_contract.max_new_tokens:
                raise ValueError("cap_stop must reach max_new_tokens")
            if self.policy_contract.natural_stop_token_id in self.identity.generated_token_ids:
                raise ValueError("cap_stop must occur before the sealed natural stop")
        else:
            raise ValueError("terminal_kind must be natural_stop or cap_stop")
        object.__setattr__(self, "generated_tokens", tokens)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "human13_complete_trajectory_evidence.v1",
            "identity": self.identity.to_dict(),
            "policy_contract": self.policy_contract.to_dict(),
            "generated_tokens": [token.to_dict() for token in self.generated_tokens],
            "terminal_kind": self.terminal_kind,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CompleteTrajectoryEvidence":
        if value.get("schema_version") != "human13_complete_trajectory_evidence.v1":
            raise ValueError("complete trajectory evidence schema_version differs")
        return cls(
            identity=ArtifactIdentity.from_dict(value["identity"]),
            policy_contract=PolicyContract.from_dict(value["policy_contract"]),
            generated_tokens=tuple(GeneratedTokenEvidence.from_dict(token) for token in value["generated_tokens"]),
            terminal_kind=value["terminal_kind"],
        )

    @property
    def content_sha256(self) -> str:
        return _sha256(self.to_dict())


@dataclass(frozen=True)
class AcquisitionGroup:
    identity: ArtifactIdentity
    policy_contract: PolicyContract
    trajectories: tuple[CompleteTrajectoryEvidence, ...]
    seed_group_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.identity, ArtifactIdentity) or not isinstance(self.policy_contract, PolicyContract):
            raise ValueError("acquisition group identity and contract must be sealed records")
        if self.identity != self.policy_contract.identity:
            raise ValueError("acquisition group identity differs from policy contract identity")
        if not isinstance(self.seed_group_id, str) or not self.seed_group_id:
            raise ValueError("seed_group_id must be a nonempty string")
        trajectories = tuple(self.trajectories)
        if not trajectories:
            raise ValueError("acquisition group must contain trajectories")
        group_surface = (
            self.identity.source_sha256,
            self.identity.manifest_sha256,
            self.identity.model_id,
            self.identity.tokenizer_id,
            self.identity.processor_id,
            self.policy_contract.repetition_penalty,
            self.policy_contract.temperature,
            self.policy_contract.natural_stop_token_id,
            self.policy_contract.max_new_tokens,
            self.policy_contract.processor_order,
            self.policy_contract.sampler_backend_id,
            self.policy_contract.top_p,
            self.policy_contract.top_k,
            self.policy_contract.n,
            self.policy_contract.min_new_tokens,
            self.policy_contract.frequency_penalty,
            self.policy_contract.presence_penalty,
            self.policy_contract.ignore_eos,
        )
        for item in trajectories:
            if not isinstance(item, CompleteTrajectoryEvidence):
                raise ValueError("acquisition group trajectories must be complete evidence")
            item_surface = (
                item.identity.source_sha256,
                item.identity.manifest_sha256,
                item.identity.model_id,
                item.identity.tokenizer_id,
                item.identity.processor_id,
                item.policy_contract.repetition_penalty,
                item.policy_contract.temperature,
                item.policy_contract.natural_stop_token_id,
                item.policy_contract.max_new_tokens,
                item.policy_contract.processor_order,
                item.policy_contract.sampler_backend_id,
                item.policy_contract.top_p,
                item.policy_contract.top_k,
                item.policy_contract.n,
                item.policy_contract.min_new_tokens,
                item.policy_contract.frequency_penalty,
                item.policy_contract.presence_penalty,
                item.policy_contract.ignore_eos,
            )
            if item_surface != group_surface:
                raise ValueError("acquisition group trajectory contract differs")
        hashes = tuple(item.content_sha256 for item in trajectories)
        if len(set(hashes)) != len(hashes):
            raise ValueError("acquisition group trajectories must be distinct")
        request_ids = tuple(item.identity.request_id for item in trajectories)
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("acquisition group trajectories must have distinct request identities")
        object.__setattr__(self, "trajectories", trajectories)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "human13_acquisition_group.v1",
            "identity": self.identity.to_dict(),
            "policy_contract": self.policy_contract.to_dict(),
            "trajectories": [item.to_dict() for item in self.trajectories],
            "seed_group_id": self.seed_group_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AcquisitionGroup":
        if value.get("schema_version") != "human13_acquisition_group.v1":
            raise ValueError("acquisition group schema_version differs")
        return cls(
            identity=ArtifactIdentity.from_dict(value["identity"]),
            policy_contract=PolicyContract.from_dict(value["policy_contract"]),
            trajectories=tuple(CompleteTrajectoryEvidence.from_dict(item) for item in value["trajectories"]),
            seed_group_id=value["seed_group_id"],
        )

    @property
    def content_sha256(self) -> str:
        return _sha256(self.to_dict())


@dataclass(frozen=True)
class ReplayTolerance:
    per_token_nats: float = _FIXED_PER_TOKEN_NATS
    group_mean_nats: float = _FIXED_GROUP_MEAN_NATS

    def __post_init__(self) -> None:
        if self.per_token_nats != _FIXED_PER_TOKEN_NATS or self.group_mean_nats != _FIXED_GROUP_MEAN_NATS:
            raise ValueError("replay tolerance must use the fixed sealed thresholds")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "human13_rp_replay_tolerance.v1",
            "per_token_nats": self.per_token_nats,
            "group_mean_nats": self.group_mean_nats,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ReplayTolerance":
        if value.get("schema_version") != "human13_rp_replay_tolerance.v1":
            raise ValueError("replay tolerance schema_version differs")
        return cls(
            per_token_nats=value["per_token_nats"],
            group_mean_nats=value["group_mean_nats"],
        )

    @property
    def content_sha256(self) -> str:
        return _sha256(self.to_dict())


__all__ = [
    "AcquisitionGroup",
    "ArtifactIdentity",
    "CompleteTrajectoryEvidence",
    "GeneratedTokenEvidence",
    "PolicyContract",
    "ReplayTolerance",
]
