"""Backend-neutral inference contracts and session lifecycle validation."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from src.common.errors import RuntimeContractError


BackendName = Literal["hf", "vllm"]
ALLOWED_STRIP_POLICIES = {"none", "terminal_im_end"}
ALLOWED_LOGICAL_IMAGE_TRANSFORMS = {"identity", "hflip", "vflip", "hvflip"}
POLICY_LIKELIHOOD_DEFINITION = "fp32_log_softmax_after_active_generation_processors"
RAW_LIKELIHOOD_DEFINITION = "fp32_log_softmax_unmodified_lm_head_logits"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class GenerationPolicy:
    """Shared deterministic generation and trace policy."""

    max_new_tokens: int
    repetition_penalty: float = 1.0
    temperature: float = 0.0
    top_p: float = 1.0
    include_raw_model_logprob: bool = False

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_new_tokens, bool)
            or not isinstance(self.max_new_tokens, int)
            or self.max_new_tokens <= 0
        ):
            _fail(
                "generation max_new_tokens must be a positive integer",
                code="backend_contract.generation_policy",
                field="max_new_tokens",
            )
        if (
            isinstance(self.repetition_penalty, bool)
            or not isinstance(self.repetition_penalty, (int, float))
            or not math.isfinite(self.repetition_penalty)
            or self.repetition_penalty <= 0
        ):
            _fail(
                "generation repetition_penalty must be finite and positive",
                code="backend_contract.generation_policy",
                field="repetition_penalty",
            )
        if (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or isinstance(self.top_p, bool)
            or not isinstance(self.top_p, (int, float))
            or self.temperature != 0.0
            or self.top_p != 1.0
        ):
            _fail(
                "canonical backend sessions require deterministic generation",
                code="backend_contract.nondeterministic_generation",
                field="temperature/top_p",
            )
        if not isinstance(self.include_raw_model_logprob, bool):
            _fail(
                "include_raw_model_logprob must be boolean",
                code="backend_contract.generation_policy",
                field="include_raw_model_logprob",
            )

    def to_artifact_dict(self) -> dict[str, object]:
        return {
            "max_new_tokens": self.max_new_tokens,
            "repetition_penalty": self.repetition_penalty,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "include_raw_model_logprob": self.include_raw_model_logprob,
        }


@dataclass(frozen=True)
class DecodeRequest:
    """One backend-neutral, single-image decode request."""

    request_id: str
    chat_text: str
    input_prompt_token_ids: tuple[int, ...]
    expected_executed_prompt_token_ids: tuple[int, ...]
    image_path: str
    declared_image_width: int
    declared_image_height: int
    decoded_image_width: int
    decoded_image_height: int
    image_sha256: str
    generation_policy: GenerationPolicy
    expected_image_grid_thw: tuple[int, int, int] | None = None
    logical_transform_id: str = "identity"

    def __post_init__(self) -> None:
        for field_name in ("request_id", "chat_text", "image_path"):
            if not isinstance(getattr(self, field_name), str) or not getattr(
                self, field_name
            ):
                _fail(
                    "decode request string fields must be non-empty",
                    code="backend_contract.decode_request",
                    field=field_name,
                    request_id=self.request_id,
                )
        _validate_token_ids(
            self.input_prompt_token_ids,
            field_name="input_prompt_token_ids",
            request_id=self.request_id,
        )
        object.__setattr__(
            self,
            "input_prompt_token_ids",
            tuple(self.input_prompt_token_ids),
        )
        object.__setattr__(
            self,
            "expected_executed_prompt_token_ids",
            tuple(self.expected_executed_prompt_token_ids),
        )
        _validate_token_ids(
            self.expected_executed_prompt_token_ids,
            field_name="expected_executed_prompt_token_ids",
            request_id=self.request_id,
        )
        dimensions = {
            "declared_image_width": self.declared_image_width,
            "declared_image_height": self.declared_image_height,
            "decoded_image_width": self.decoded_image_width,
            "decoded_image_height": self.decoded_image_height,
        }
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in dimensions.values()
        ):
            _fail(
                "decode request image dimensions must be positive integers",
                code="backend_contract.image_dimensions",
                field="image_dimensions",
                request_id=self.request_id,
            )
        if (
            self.declared_image_width,
            self.declared_image_height,
        ) != (self.decoded_image_width, self.decoded_image_height):
            raise RuntimeContractError(
                "declared and decoded image dimensions must match before backend launch",
                code="backend_contract.image_dimensions_mismatch",
                context={"request_id": self.request_id, **dimensions},
            )
        if (
            not isinstance(self.image_sha256, str)
            or _SHA256_PATTERN.fullmatch(self.image_sha256) is None
        ):
            _fail(
                "decode request image_sha256 must be a lowercase SHA-256 digest",
                code="backend_contract.image_sha256",
                field="image_sha256",
                request_id=self.request_id,
            )
        if not isinstance(self.generation_policy, GenerationPolicy):
            _fail(
                "decode request generation_policy must be a GenerationPolicy",
                code="backend_contract.generation_policy",
                field="generation_policy",
                request_id=self.request_id,
            )
        if self.logical_transform_id not in ALLOWED_LOGICAL_IMAGE_TRANSFORMS:
            _fail(
                "decode request logical image transform is unsupported",
                code="backend_contract.logical_transform_id",
                field="logical_transform_id",
                request_id=self.request_id,
            )
        if self.expected_image_grid_thw is not None and (
            not isinstance(self.expected_image_grid_thw, (list, tuple))
            or len(self.expected_image_grid_thw) != 3
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in self.expected_image_grid_thw
            )
        ):
            _fail(
                "expected_image_grid_thw must contain three positive integers",
                code="backend_contract.image_grid",
                field="expected_image_grid_thw",
                request_id=self.request_id,
            )
        if self.expected_image_grid_thw is not None:
            object.__setattr__(
                self,
                "expected_image_grid_thw",
                tuple(self.expected_image_grid_thw),
            )

    @property
    def prompt_token_ids(self) -> list[int]:
        """Temporary artifact-facing alias for the executed prompt ids."""

        return list(self.expected_executed_prompt_token_ids)


@dataclass(frozen=True)
class LikelihoodPair:
    """Aligned policy and optional raw-model likelihood channels."""

    policy_logprob: float | None
    raw_model_logprob: float | None

    def validate(
        self,
        *,
        request_id: str,
        step_index: int,
        is_pad: bool,
        raw_required: bool,
    ) -> None:
        if is_pad:
            if self.policy_logprob is not None or self.raw_model_logprob is not None:
                _fail_likelihood(
                    "padding trace rows must not carry likelihoods",
                    request_id=request_id,
                    step_index=step_index,
                )
            return
        _validate_logprob(
            self.policy_logprob,
            channel="policy_logprob",
            request_id=request_id,
            step_index=step_index,
            required=True,
        )
        _validate_logprob(
            self.raw_model_logprob,
            channel="raw_model_logprob",
            request_id=request_id,
            step_index=step_index,
            required=raw_required,
        )
        if not raw_required and self.raw_model_logprob is not None:
            _fail_likelihood(
                "raw-model likelihood was returned when the request disabled it",
                request_id=request_id,
                step_index=step_index,
            )


@dataclass(frozen=True)
class TokenTrace:
    step_index: int
    token_id: int
    token_text: str
    likelihood: LikelihoodPair
    is_stop: bool
    is_pad: bool
    backend: str
    backend_mode: str
    response_family: str

    @property
    def policy_logprob(self) -> float | None:
        return self.likelihood.policy_logprob

    @property
    def raw_model_logprob(self) -> float | None:
        return self.likelihood.raw_model_logprob

    @property
    def logprob(self) -> float | None:
        """Artifact V1 compatibility alias; policy likelihood owns scoring."""

        return self.policy_logprob


@dataclass(frozen=True)
class DecodeResult:
    request_id: str
    backend: str
    backend_mode: str
    response_family: str
    executed_prompt_token_ids: tuple[int, ...]
    generated_token_ids: tuple[int, ...]
    raw_generated_text: str
    parser_text: str
    strip_policy: str
    stop_reason: str
    token_trace: tuple[TokenTrace, ...]
    executed_media_sha256: str
    observed_image_grid_thw: tuple[int, int, int] | None = None
    native_generated_text: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "executed_prompt_token_ids",
            tuple(self.executed_prompt_token_ids),
        )
        object.__setattr__(
            self,
            "generated_token_ids",
            tuple(self.generated_token_ids),
        )
        object.__setattr__(self, "token_trace", tuple(self.token_trace))
        if self.observed_image_grid_thw is not None:
            object.__setattr__(
                self,
                "observed_image_grid_thw",
                tuple(self.observed_image_grid_thw),
            )
        if (
            not isinstance(self.executed_media_sha256, str)
            or _SHA256_PATTERN.fullmatch(self.executed_media_sha256) is None
        ):
            _fail_result(
                "decode result executed media identity must be a SHA-256 digest",
                field="executed_media_sha256",
                request_id=self.request_id,
            )

    @property
    def prompt_token_ids(self) -> list[int]:
        """Temporary artifact-facing alias retained during pipeline migration."""

        return list(self.executed_prompt_token_ids)

    def validate_for_request(
        self,
        request: DecodeRequest,
        receipt: BackendSessionReceipt,
    ) -> None:
        if self.request_id != request.request_id:
            _fail_result(
                "decode result request id does not match its request",
                field="request_id",
                request_id=request.request_id,
            )
        expected_identity = (
            receipt.backend,
            receipt.backend_mode,
            receipt.response_family,
        )
        if (self.backend, self.backend_mode, self.response_family) != expected_identity:
            _fail_result(
                "decode result backend identity does not match the session receipt",
                field="backend_identity",
                request_id=request.request_id,
            )
        if self.executed_prompt_token_ids != request.expected_executed_prompt_token_ids:
            _fail_result(
                "decode result executed prompt ids do not match expected expansion",
                field="executed_prompt_token_ids",
                request_id=request.request_id,
            )
        if request.expected_image_grid_thw is not None and (
            self.observed_image_grid_thw != request.expected_image_grid_thw
        ):
            _fail_result(
                "decode result image grid does not match shared no-resize evidence",
                field="observed_image_grid_thw",
                request_id=request.request_id,
            )
        if not self.generated_token_ids or not self.token_trace or not self.stop_reason:
            _fail_result(
                "decode result is missing generated trace evidence",
                field="generated_trace",
                request_id=request.request_id,
            )
        if self.strip_policy not in ALLOWED_STRIP_POLICIES:
            _fail_result(
                "decode result has an invalid strip policy",
                field="strip_policy",
                request_id=request.request_id,
            )
        non_pad_ids: list[int] = []
        stop_count = 0
        seen_stop = False
        for index, trace in enumerate(self.token_trace):
            if trace.step_index != index:
                _fail_result(
                    "token trace step indices must be contiguous",
                    field="token_trace.step_index",
                    request_id=request.request_id,
                )
            if (
                trace.backend,
                trace.backend_mode,
                trace.response_family,
            ) != expected_identity:
                _fail_result(
                    "token trace backend identity does not match the session receipt",
                    field="token_trace.backend_identity",
                    request_id=request.request_id,
                )
            if not trace.token_text:
                _fail_result(
                    "token trace token text must be non-empty",
                    field="token_trace.token_text",
                    request_id=request.request_id,
                )
            if (
                isinstance(trace.token_id, bool)
                or not isinstance(trace.token_id, int)
                or trace.token_id < 0
            ):
                _fail_result(
                    "token trace token ids must be non-negative integers",
                    field="token_trace.token_id",
                    request_id=request.request_id,
                )
            if trace.is_pad and not seen_stop:
                _fail_result(
                    "padding trace rows may occur only after a stop token",
                    field="token_trace.is_pad",
                    request_id=request.request_id,
                )
            if seen_stop and not trace.is_pad:
                _fail_result(
                    "non-padding trace rows may not occur after a stop token",
                    field="token_trace.is_pad",
                    request_id=request.request_id,
                )
            trace.likelihood.validate(
                request_id=request.request_id,
                step_index=index,
                is_pad=trace.is_pad,
                raw_required=request.generation_policy.include_raw_model_logprob,
            )
            if not trace.is_pad:
                non_pad_ids.append(trace.token_id)
            stop_count += int(trace.is_stop)
            seen_stop = seen_stop or trace.is_stop
        if tuple(non_pad_ids) != self.generated_token_ids:
            _fail_result(
                "generated ids and non-padding token trace ids are misaligned",
                field="generated_token_ids",
                request_id=request.request_id,
            )
        if stop_count > 1:
            _fail_result(
                "decode result contains multiple semantic stop tokens",
                field="token_trace.is_stop",
                request_id=request.request_id,
            )
        if (self.stop_reason == "im_end") != (stop_count == 1):
            _fail_result(
                "decode stop reason and stop-token trace evidence disagree",
                field="stop_reason",
                request_id=request.request_id,
            )

    def validate_for_scored(self, *, raw_model_logprob_required: bool) -> None:
        """Compatibility validation for artifact callers during Wave 1 migration."""

        if not self.generated_token_ids or not self.token_trace or not self.stop_reason:
            _fail_result(
                "scored decode result is missing generated trace evidence",
                field="generated_trace",
                request_id=self.request_id,
            )
        for trace in self.token_trace:
            trace.likelihood.validate(
                request_id=self.request_id,
                step_index=trace.step_index,
                is_pad=trace.is_pad,
                raw_required=raw_model_logprob_required,
            )


@dataclass(frozen=True)
class BackendLaunch:
    """Serializable backend launch contract; contains no native runtime objects."""

    backend: BackendName
    model_path: str
    model_dtype: Literal["bf16", "fp16", "fp32"]
    batch_size: int
    generation_config_fingerprint: str
    backend_options: Mapping[str, object] = field(default_factory=dict)
    expected_model_identity: Mapping[str, object] = field(default_factory=dict)
    execution_model_identity: Mapping[str, object] | None = None
    adapter: Mapping[str, object] | None = None
    embedding_delta: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if self.backend not in ("hf", "vllm"):
            _fail(
                "backend launch names an unknown backend",
                code="backend_contract.launch",
                field="backend",
            )
        if self.model_dtype not in ("bf16", "fp16", "fp32"):
            _fail(
                "backend launch model_dtype is unsupported",
                code="backend_contract.launch",
                field="model_dtype",
            )
        if (
            not isinstance(self.model_path, str)
            or not self.model_path
            or not isinstance(self.generation_config_fingerprint, str)
            or not self.generation_config_fingerprint
        ):
            _fail(
                "backend launch identity fields must be non-empty",
                code="backend_contract.launch",
                field="model_path/generation_config_fingerprint",
            )
        if (
            isinstance(self.batch_size, bool)
            or not isinstance(self.batch_size, int)
            or self.batch_size <= 0
        ):
            _fail(
                "backend launch batch_size must be a positive integer",
                code="backend_contract.launch",
                field="batch_size",
            )
        for field_name, value in {
            "backend_options": self.backend_options,
            "expected_model_identity": self.expected_model_identity,
        }.items():
            if not isinstance(value, Mapping):
                _fail_native(field_name, type(value).__name__)
        _require_backend_neutral(self.backend_options, field_name="backend_options")
        _require_backend_neutral(
            self.expected_model_identity,
            field_name="expected_model_identity",
        )
        if self.execution_model_identity is not None:
            if not isinstance(self.execution_model_identity, Mapping):
                _fail_native(
                    "execution_model_identity",
                    type(self.execution_model_identity).__name__,
                )
            _require_backend_neutral(
                self.execution_model_identity,
                field_name="execution_model_identity",
            )
        for field_name, value in {
            "adapter": self.adapter,
            "embedding_delta": self.embedding_delta,
        }.items():
            if value is None:
                continue
            if not isinstance(value, Mapping):
                _fail_native(field_name, type(value).__name__)
            _require_backend_neutral(value, field_name=field_name)


@dataclass(frozen=True)
class BackendSessionReceipt:
    backend: BackendName
    backend_mode: str
    response_family: str
    backend_version: str
    model_identity: Mapping[str, object]
    tokenizer_identity: Mapping[str, object]
    processor_identity: Mapping[str, object]
    generation_config_fingerprint: str
    effective_settings: Mapping[str, object]
    likelihood_semantics: Mapping[str, object]
    execution_model_identity: Mapping[str, object] | None = None

    def validate_for_launch(self, launch: BackendLaunch) -> None:
        required_strings = {
            "backend_mode": self.backend_mode,
            "response_family": self.response_family,
            "backend_version": self.backend_version,
            "generation_config_fingerprint": self.generation_config_fingerprint,
        }
        if self.backend != launch.backend:
            _fail_receipt("backend", launch=launch, receipt=self)
        for field_name, value in required_strings.items():
            if not isinstance(value, str) or not value:
                _fail_receipt(field_name, launch=launch, receipt=self)
        if self.generation_config_fingerprint != launch.generation_config_fingerprint:
            _fail_receipt("generation_config_fingerprint", launch=launch, receipt=self)
        for field_name, value in {
            "model_identity": self.model_identity,
            "tokenizer_identity": self.tokenizer_identity,
            "processor_identity": self.processor_identity,
            "effective_settings": self.effective_settings,
            "likelihood_semantics": self.likelihood_semantics,
        }.items():
            if not isinstance(value, Mapping) or not value:
                _fail_receipt(field_name, launch=launch, receipt=self)
            _require_backend_neutral(value, field_name=field_name)
        if self.effective_settings.get("batch_size") != launch.batch_size:
            _fail_receipt("effective_settings.batch_size", launch=launch, receipt=self)
        if self.likelihood_semantics.get("policy") != POLICY_LIKELIHOOD_DEFINITION:
            _fail_receipt("likelihood_semantics.policy", launch=launch, receipt=self)
        if self.likelihood_semantics.get("raw") != RAW_LIKELIHOOD_DEFINITION:
            _fail_receipt("likelihood_semantics.raw", launch=launch, receipt=self)
        if launch.expected_model_identity and (
            dict(self.model_identity) != dict(launch.expected_model_identity)
        ):
            _fail_receipt("model_identity", launch=launch, receipt=self)
        if self.execution_model_identity is not None:
            if not isinstance(self.execution_model_identity, Mapping):
                _fail_native(
                    "execution_model_identity",
                    type(self.execution_model_identity).__name__,
                )
            _require_backend_neutral(
                self.execution_model_identity,
                field_name="execution_model_identity",
            )
        if launch.execution_model_identity is not None and (
            self.execution_model_identity is None
            or dict(self.execution_model_identity)
            != dict(launch.execution_model_identity)
        ):
            _fail_receipt("execution_model_identity", launch=launch, receipt=self)

    def to_artifact_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "backend_mode": self.backend_mode,
            "response_family": self.response_family,
            "backend_version": self.backend_version,
            "model_identity": dict(self.model_identity),
            "tokenizer_identity": dict(self.tokenizer_identity),
            "processor_identity": dict(self.processor_identity),
            "generation_config_fingerprint": self.generation_config_fingerprint,
            "effective_settings": dict(self.effective_settings),
            "likelihood_semantics": dict(self.likelihood_semantics),
            "execution_model_identity": (
                None
                if self.execution_model_identity is None
                else dict(self.execution_model_identity)
            ),
        }


@runtime_checkable
class BackendSession(Protocol):
    @property
    def receipt(self) -> BackendSessionReceipt: ...

    def decode(self, requests: Sequence[DecodeRequest]) -> Sequence[DecodeResult]: ...

    def close(self) -> None: ...


BackendSessionOpener = Callable[[BackendLaunch], BackendSession]


@contextmanager
def open_backend_session(
    launch: BackendLaunch,
    *,
    opener: BackendSessionOpener | None = None,
) -> Iterator[BackendSession]:
    """Open an injectable backend session and guarantee explicit cleanup."""

    session = (opener or _default_session_opener)(launch)
    try:
        if not isinstance(session, BackendSession):
            raise RuntimeContractError(
                "backend opener did not return a BackendSession",
                code="backend_contract.session_protocol",
                context={"value_type": type(session).__name__},
            )
        session.receipt.validate_for_launch(launch)
        yield session
    finally:
        close = getattr(session, "close", None)
        if callable(close):
            close()


def validate_decode_results(
    *,
    requests: Sequence[DecodeRequest],
    results: Sequence[DecodeResult],
    receipt: BackendSessionReceipt,
) -> tuple[DecodeResult, ...]:
    """Validate exact request coverage and restore request order."""

    requested = [request.request_id for request in requests]
    observed = [result.request_id for result in results]
    requested_counts = Counter(requested)
    observed_counts = Counter(observed)
    if requested_counts != observed_counts or any(
        count != 1 for count in observed_counts.values()
    ):
        raise RuntimeContractError(
            "backend results must cover every request id exactly once",
            code="backend_contract.result_set",
            context={
                "requested_request_ids": requested,
                "observed_request_ids": observed,
            },
        )
    request_by_id = {request.request_id: request for request in requests}
    result_by_id = {result.request_id: result for result in results}
    for request_id in requested:
        result_by_id[request_id].validate_for_request(
            request_by_id[request_id],
            receipt,
        )
    return tuple(result_by_id[request_id] for request_id in requested)


def _default_session_opener(launch: BackendLaunch) -> BackendSession:
    if launch.backend == "hf":
        from src.inference.hf_backend import open_hf_backend_session

        return open_hf_backend_session(launch)
    if launch.backend == "vllm":
        try:
            from src.inference.vllm_backend import open_vllm_backend_session
        except ImportError as exc:
            raise RuntimeContractError(
                "vLLM backend session implementation is not available",
                code="backend_contract.backend_not_implemented",
                context={"backend": launch.backend},
                cause=exc,
            ) from exc
        return open_vllm_backend_session(launch)
    raise AssertionError(f"unreachable backend: {launch.backend}")


def _validate_token_ids(
    values: Sequence[int],
    *,
    field_name: str,
    request_id: str,
) -> None:
    if (
        not isinstance(values, (list, tuple))
        or not values
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in values
        )
    ):
        _fail(
            "decode request token ids must be a non-empty sequence of non-negative integers",
            code="backend_contract.token_ids",
            field=field_name,
            request_id=request_id,
        )


def _validate_logprob(
    value: float | None,
    *,
    channel: str,
    request_id: str,
    step_index: int,
    required: bool,
) -> None:
    if value is None:
        if required:
            _fail_likelihood(
                f"generated token is missing {channel}",
                request_id=request_id,
                step_index=step_index,
            )
        return
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value > 0
    ):
        _fail_likelihood(
            f"generated token has invalid {channel}",
            request_id=request_id,
            step_index=step_index,
        )


def _require_backend_neutral(value: object, *, field_name: str) -> None:
    if value is None or isinstance(value, (str, int, bool)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            _fail_native(field_name, type(value).__name__)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                _fail_native(field_name, type(key).__name__)
            _require_backend_neutral(item, field_name=field_name)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _require_backend_neutral(item, field_name=field_name)
        return
    _fail_native(field_name, type(value).__name__)


def _fail_native(field_name: str, value_type: str) -> None:
    raise RuntimeContractError(
        "backend launch and receipt fields must contain only serializable semantic values",
        code="backend_contract.native_object",
        context={"field": field_name, "value_type": value_type},
    )


def _fail(
    message: str,
    *,
    code: str,
    field: str,
    request_id: str | None = None,
) -> None:
    context = {"field": field}
    if request_id is not None:
        context["request_id"] = request_id
    raise RuntimeContractError(message, code=code, context=context)


def _fail_likelihood(message: str, *, request_id: str, step_index: int) -> None:
    raise RuntimeContractError(
        message,
        code="backend_trace.invalid_likelihood",
        context={"request_id": request_id, "step_index": step_index},
    )


def _fail_result(message: str, *, field: str, request_id: str) -> None:
    raise RuntimeContractError(
        message,
        code="backend_trace.invalid_result",
        context={"field": field, "request_id": request_id},
    )


def _fail_receipt(
    field_name: str,
    *,
    launch: BackendLaunch,
    receipt: BackendSessionReceipt,
) -> None:
    raise RuntimeContractError(
        "backend session receipt does not match its launch contract",
        code="backend_contract.session_receipt",
        context={
            "field": field_name,
            "launch_backend": launch.backend,
            "receipt_backend": receipt.backend,
        },
    )
