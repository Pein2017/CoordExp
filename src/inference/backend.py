"""Backend-neutral decode records and HF scored-generation tracing."""

from __future__ import annotations

import base64
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import struct
import time
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import torch
from torch.nn import functional as F

from src.common.errors import RuntimeContractError


BackendName = Literal["hf", "vllm"]
DecodeMode = Literal["greedy", "sampled"]
ALLOWED_STRIP_POLICIES = {"none", "terminal_im_end"}
SAMPLING_PROFILE = "temperature_top_p_categorical_v1"
CUSTOM_SAMPLER_IDENTITY = "coordexp_hf_request_scoped_categorical_v1"
DECODE_RECEIPT_SCHEMA_VERSION = "decode_execution_receipt.v1"
SAMPLED_RUNTIME_ATTESTATION_BUNDLE_SCHEMA_VERSION = (
    "typed_executed_cuda_qwen_attestation_bundle.v1"
)
SAMPLED_RUNTIME_ATTESTATION_CASE_SCHEMA_VERSION = (
    "executed_sampled_runtime_attestation_case.v1"
)
SAMPLED_RUNTIME_ATTESTATION_CROSS_CARDINALITY_SCHEMA_VERSION = (
    "cross_cardinality_request_replay_attestation.v1"
)
SAMPLED_RUNTIME_ATTESTATION_PROCESSED_LOGIT_SCHEMA_VERSION = (
    "processed_logit_parity_attestation.v1"
)
SAMPLED_RUNTIME_ATTESTATION_AGGREGATE_OUTPUT_SCHEMA_VERSION = (
    "request_scoped_sampling_attestation_aggregate_output.v1"
)
SAMPLED_RUNTIME_ATTESTATION_POLICY_ENTRY_SCHEMA_VERSION = (
    "request_scoped_sampling_policy_attestation_entry.v1"
)
ADMITTED_PRODUCTION_REPLAY_SCHEMA_VERSION = (
    "capability_admitted_sampled_production_replay.v1"
)
PORTABLE_RUNTIME_STATE_SEAL_SCHEMA_VERSION = "portable_sampled_runtime_state_seal.v1"
EXPECTED_ATTESTED_RUNTIME_PAYLOAD_TENSOR_COUNT = 589
EXPECTED_ATTESTED_RUNTIME_PAYLOAD_BYTE_COUNT = 80_248_832
MAX_SIGNED_64_BIT_SEED = (1 << 63) - 1
_AUTHORIZED_ATTESTATION_CONFIG_TO_CHECKPOINT_SHA256 = {
    "f3000588accbcf1d9ada3b2f3e0b3324d660b4810b75d8f5d050d9f184f9ca80": (
        "613e5d97f4a7a53d6325b5c1909813d6bb82e72622942b225df9724b5556a536"
    ),
}
_FROZEN_ATTESTATION_CALIBRATION_MANIFEST_SHA256 = (
    "73edd29504dc526f54f36c543dfeb4ca03bf6fd15c82449116e7e4b546f58496"
)
_FROZEN_ATTESTATION_CALIBRATION_FIRST_FOUR = (
    (
        563648,
        "coco2017_val_000000563648",
        "fdc8c2e998fd85ed47303a69381ca7614b1dbf3d702a24fce6c3a69ac52144f5",
        864,
        1184,
    ),
    (
        303713,
        "coco2017_val_000000303713",
        "b696adefee080884449a3c299bc69396ccf813e6ea0d946608404b8f881f28bd",
        832,
        1248,
    ),
    (
        529148,
        "coco2017_val_000000529148",
        "989ba2bd49ac496e195f8c1ea98c7c161cf2a38ef7bdb359a5b00ace7d13c148",
        1248,
        832,
    ),
    (
        538236,
        "coco2017_val_000000538236",
        "b97cec610ef5686ccbcc9bbd932199df72ce6621be419ba5cfc36d1037e133be",
        1248,
        832,
    ),
)
_FROZEN_ATTESTATION_CALIBRATION_SEEDS = (
    3565713206559848094,
    2992931333390152433,
    6011931503842164206,
    7384003725263415097,
)
_FROZEN_ATTESTATION_PROMPT_TOKEN_HASHES = (
    "f8a8df1bc644f6f4ba5c023fc572717b1b5f328decd05544dc48442d20ecbed3",
    "dc21acac635d4fddfeca65c2572e453ce5624dab403351da6973b434623b5ca1",
    "dc21acac635d4fddfeca65c2572e453ce5624dab403351da6973b434623b5ca1",
    "dc21acac635d4fddfeca65c2572e453ce5624dab403351da6973b434623b5ca1",
)
_FROZEN_ATTESTATION_MODEL_INPUT_FINGERPRINTS = (
    "2c4757463789bd2d3bea86af83cd43cfe8d87e0007bbd934c6786146fecd68f2",
    "933dcc2982328d39129a825bf14c045f8258bca48b8dbeb4e937d03948f9241c",
    "556988f4c6344da24d46091bdb1072ad34bfacffda5dd6b15badf621a92e0761",
    "35847a2e491308e8ba8381a17e66aa22a917f1a3b27e86fcf9483ee416f4d8fe",
)
_ALLOWED_ATTESTATION_TEMPERATURES = frozenset({0.2, 0.4, 0.6})

_REQUIRED_SAMPLED_RUNTIME_ATTESTATION_CASES = frozenset(
    {
        "batch_size_three_forward",
        "batch_size_three_reversed",
        "batch_size_four_forward",
        "batch_size_four_reversed",
        "cross_cardinality_shared_three",
    }
)
_REQUIRED_EXECUTED_SAMPLED_RUNTIME_ATTESTATION_CASES = frozenset(
    _REQUIRED_SAMPLED_RUNTIME_ATTESTATION_CASES
    - {"cross_cardinality_shared_three"}
)
_REQUIRED_SAMPLED_RUNTIME_ATTESTATION_CHECKS = frozenset(
    {
        "attention_implementation_known_and_equal",
        "cache_update_semantics",
        "cuda_device_and_hardware_identity",
        "float32_score_and_transition_score_parity",
        "generator_device_seed_and_execution_index_binding",
        "model_evaluation_mode",
        "output_type_and_shape",
        "processed_logit_parity_before_categorical_selection",
        "qwen_model_and_token_identity",
        "qwen_im_end_stop_reason",
        "request_identity_replay_across_order_and_cardinality",
        "sample_then_pad_semantics",
        "score_step_alignment",
    }
)

_GENERATION_CONFIG_PROFILE_FIELDS = frozenset(
    {
        "max_length",
        "max_new_tokens",
        "min_length",
        "min_new_tokens",
        "early_stopping",
        "max_time",
        "stop_strings",
        "do_sample",
        "num_beams",
        "use_cache",
        "cache_implementation",
        "cache_config",
        "return_legacy_cache",
        "prefill_chunk_size",
        "temperature",
        "top_k",
        "top_p",
        "min_p",
        "typical_p",
        "epsilon_cutoff",
        "eta_cutoff",
        "repetition_penalty",
        "encoder_repetition_penalty",
        "length_penalty",
        "no_repeat_ngram_size",
        "bad_words_ids",
        "renormalize_logits",
        "forced_bos_token_id",
        "forced_eos_token_id",
        "remove_invalid_values",
        "exponential_decay_length_penalty",
        "suppress_tokens",
        "begin_suppress_tokens",
        "sequence_bias",
        "token_healing",
        "guidance_scale",
        "watermarking_config",
        "num_return_sequences",
        "output_attentions",
        "output_hidden_states",
        "output_scores",
        "output_logits",
        "return_dict_in_generate",
        "pad_token_id",
        "bos_token_id",
        "eos_token_id",
        "encoder_no_repeat_ngram_size",
        "decoder_start_token_id",
        "is_assistant",
        "num_assistant_tokens",
        "num_assistant_tokens_schedule",
        "assistant_confidence_threshold",
        "prompt_lookup_num_tokens",
        "max_matching_ngram_size",
        "assistant_early_exit",
        "assistant_lookbehind",
        "target_lookbehind",
        "compile_config",
        "disable_compile",
        "low_memory",
        "penalty_alpha",
        "dola_layers",
        "diversity_penalty",
        "num_beam_groups",
        "constraints",
        "force_words_ids",
        "_from_model_config",
        "transformers_version",
    }
)

_NEUTRAL_GENERATION_CONFIG_VALUES: Mapping[str, Any] = MappingProxyType(
    {
        "min_length": 0,
        "min_new_tokens": None,
        "early_stopping": False,
        "max_time": None,
        "stop_strings": None,
        "num_beams": 1,
        "use_cache": True,
        "cache_implementation": None,
        "cache_config": None,
        "return_legacy_cache": None,
        "prefill_chunk_size": None,
        "top_k": 0,
        "min_p": None,
        "typical_p": 1.0,
        "epsilon_cutoff": 0.0,
        "eta_cutoff": 0.0,
        "encoder_repetition_penalty": 1.0,
        "length_penalty": 1.0,
        "no_repeat_ngram_size": 0,
        "bad_words_ids": None,
        "renormalize_logits": False,
        "forced_bos_token_id": None,
        "forced_eos_token_id": None,
        "remove_invalid_values": False,
        "exponential_decay_length_penalty": None,
        "suppress_tokens": None,
        "begin_suppress_tokens": None,
        "sequence_bias": None,
        "token_healing": False,
        "guidance_scale": None,
        "watermarking_config": None,
        "num_return_sequences": 1,
        "output_attentions": False,
        "output_hidden_states": False,
        "output_scores": True,
        "output_logits": False,
        "return_dict_in_generate": True,
        "encoder_no_repeat_ngram_size": 0,
        "is_assistant": False,
        "num_assistant_tokens": 20,
        "num_assistant_tokens_schedule": "constant",
        "assistant_confidence_threshold": 0.4,
        "prompt_lookup_num_tokens": None,
        "max_matching_ngram_size": None,
        "assistant_early_exit": None,
        "assistant_lookbehind": 10,
        "target_lookbehind": 10,
        "compile_config": None,
        "disable_compile": False,
        "low_memory": None,
        "penalty_alpha": None,
        "dola_layers": None,
        "diversity_penalty": 0.0,
        "num_beam_groups": 1,
        "constraints": None,
        "force_words_ids": None,
        "_from_model_config": False,
    }
)


@dataclass
class _SamplingTimingRecorder:
    categorical_draw_call_count: int = 0
    categorical_draw_cpu_dispatch_seconds: float = 0.0

    def to_artifact_dict(
        self,
        *,
        full_batch_elapsed_seconds: float,
        batch_size: int,
        score_steps: int,
        device: torch.device,
    ) -> dict[str, Any]:
        return {
            "measurement_scope": "synchronized_sampling_attestation",
            "device": str(device),
            "full_batch_elapsed_seconds": float(full_batch_elapsed_seconds),
            "batch_size": batch_size,
            "score_steps": score_steps,
            "generated_positions_including_padding": batch_size * score_steps,
            "categorical_draw_call_count": self.categorical_draw_call_count,
            "categorical_draw_cpu_enqueue_or_dispatch_seconds": float(
                self.categorical_draw_cpu_dispatch_seconds
            ),
        }


@dataclass
class _GenerationExecutionCapture:
    prepared_generation_profile: dict[str, Any] | None = None
    custom_sampler_executed: bool = False
    processed_score_shapes: list[tuple[int, ...]] | None = None
    first_processed_scores_float32: torch.Tensor | None = None
    cache_steps: list[dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        if self.processed_score_shapes is None:
            self.processed_score_shapes = []
        if self.cache_steps is None:
            self.cache_steps = []

    def record_processed_scores(self, scores: torch.Tensor) -> None:
        assert self.processed_score_shapes is not None
        self.processed_score_shapes.append(tuple(int(value) for value in scores.shape))
        if self.first_processed_scores_float32 is None:
            self.first_processed_scores_float32 = (
                scores.detach().to(device="cpu", dtype=torch.float32).contiguous()
            )

    def record_cache_step(self, *, outputs: Any, model_kwargs: Mapping[str, Any]) -> None:
        assert self.cache_steps is not None
        cache = model_kwargs.get("past_key_values")
        self.cache_steps.append(
            {
                "step_index": len(self.cache_steps),
                "output_cache_present": getattr(outputs, "past_key_values", None)
                is not None,
                "updated_cache_present": cache is not None,
                "updated_cache_type": None if cache is None else type(cache).__name__,
                "updated_cache_sequence_length": _cache_sequence_length(cache),
            }
        )


@dataclass
class _ProcessedLogitCapture:
    first_scores_float32: torch.Tensor | None = None

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        del input_ids
        if self.first_scores_float32 is None:
            self.first_scores_float32 = (
                scores.detach().to(device="cpu", dtype=torch.float32).contiguous()
            )
        return scores


@dataclass(frozen=True)
class DecodeGenerationPolicy:
    """Complete backend-neutral generation behavior for one decode request."""

    mode: DecodeMode
    sampling_profile: str
    max_new_tokens: int
    repetition_penalty: float
    temperature: float
    top_p: float

    def __post_init__(self) -> None:
        if self.mode not in {"greedy", "sampled"}:
            raise RuntimeContractError(
                "decode generation policy has an unknown mode",
                code="backend_policy.invalid_mode",
                context={"mode": self.mode},
            )
        if self.sampling_profile != SAMPLING_PROFILE:
            raise RuntimeContractError(
                "decode generation policy has an unsupported sampling profile",
                code="backend_policy.invalid_sampling_profile",
                context={"sampling_profile": self.sampling_profile},
            )
        if (
            not isinstance(self.max_new_tokens, int)
            or isinstance(self.max_new_tokens, bool)
            or self.max_new_tokens <= 0
        ):
            raise RuntimeContractError(
                "decode generation policy requires positive max_new_tokens",
                code="backend_policy.invalid_max_new_tokens",
                context={"max_new_tokens": self.max_new_tokens},
            )
        if not _is_positive_finite(self.repetition_penalty):
            raise RuntimeContractError(
                "decode generation policy requires a positive finite repetition penalty",
                code="backend_policy.invalid_repetition_penalty",
                context={"repetition_penalty": self.repetition_penalty},
            )
        if (
            not isinstance(self.top_p, (int, float))
            or isinstance(self.top_p, bool)
            or not math.isfinite(float(self.top_p))
            or not 0.0 < float(self.top_p) <= 1.0
        ):
            raise RuntimeContractError(
                "decode generation policy requires top_p in (0, 1]",
                code="backend_policy.invalid_top_p",
                context={"top_p": self.top_p},
            )
        if (
            not isinstance(self.temperature, (int, float))
            or isinstance(self.temperature, bool)
            or not math.isfinite(float(self.temperature))
        ):
            raise RuntimeContractError(
                "decode generation policy requires a finite temperature",
                code="backend_policy.invalid_temperature",
                context={"temperature": self.temperature},
            )
        if self.mode == "greedy" and (
            float(self.temperature) != 0.0 or float(self.top_p) != 1.0
        ):
            raise RuntimeContractError(
                "greedy decode policy requires neutral sampling fields",
                code="backend_policy.non_neutral_greedy",
                context={"temperature": self.temperature, "top_p": self.top_p},
            )
        if self.mode == "sampled" and float(self.temperature) <= 0.0:
            raise RuntimeContractError(
                "sampled decode policy requires positive temperature",
                code="backend_policy.invalid_temperature",
                context={"temperature": self.temperature},
            )

    @classmethod
    def greedy(
        cls,
        *,
        max_new_tokens: int,
        repetition_penalty: float = 1.0,
    ) -> DecodeGenerationPolicy:
        return cls(
            mode="greedy",
            sampling_profile=SAMPLING_PROFILE,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            temperature=0.0,
            top_p=1.0,
        )

    @classmethod
    def sampled(
        cls,
        *,
        max_new_tokens: int,
        repetition_penalty: float,
        temperature: float,
        top_p: float,
    ) -> DecodeGenerationPolicy:
        return cls(
            mode="sampled",
            sampling_profile=SAMPLING_PROFILE,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            top_p=top_p,
        )

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "sampling_profile": self.sampling_profile,
            "max_new_tokens": self.max_new_tokens,
            "repetition_penalty": float(self.repetition_penalty),
            "temperature": float(self.temperature),
            "top_p": float(self.top_p),
        }

    @property
    def fingerprint(self) -> str:
        return _sha256_json(self.to_artifact_dict())


@dataclass(frozen=True)
class DecodeRequest:
    request_id: str
    prompt_token_ids: list[int]
    model_inputs: Mapping[str, Any]
    generation_policy: DecodeGenerationPolicy
    sampling_seed: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id.strip():
            raise RuntimeContractError(
                "decode request identity must be a nonempty string",
                code="backend_policy.invalid_request_id",
                context={"request_id": self.request_id},
            )
        if not isinstance(self.generation_policy, DecodeGenerationPolicy):
            raise RuntimeContractError(
                "decode request requires a DecodeGenerationPolicy",
                code="backend_policy.missing_generation_policy",
                context={"request_id": self.request_id},
            )
        if self.generation_policy.mode == "sampled":
            if self.sampling_seed is None:
                raise RuntimeContractError(
                    "sampled decode request requires a request-owned seed",
                    code="backend_policy.sampled_seed_required",
                    context={"request_id": self.request_id},
                )
            if not _is_valid_sampling_seed(self.sampling_seed):
                raise RuntimeContractError(
                    "sampled decode request seed is outside signed 64-bit range",
                    code="backend_policy.invalid_sampling_seed",
                    context={
                        "request_id": self.request_id,
                        "sampling_seed": self.sampling_seed,
                    },
                )
        elif self.sampling_seed is not None:
            raise RuntimeContractError(
                "greedy decode request forbids a scientific sampling seed",
                code="backend_policy.greedy_seed_forbidden",
                context={"request_id": self.request_id},
            )


@dataclass(frozen=True)
class DecodeExecutionReceipt:
    """Canonical immutable execution evidence bound to one decode result."""

    schema_version: str
    request_id: str
    decode_generation_policy: Mapping[str, Any]
    decode_generation_policy_fingerprint: str
    generation_config_fingerprint: str
    sampling_seed: int | None
    random_generator_kind: str | None
    random_generator_device: str | None
    random_generator_initial_seed: int | None
    request_execution_index: int
    batch_request_order_fingerprint: str
    executed_generation_arguments: Mapping[str, Any]
    effective_generation_profile_fingerprint: str
    prompt_token_count: int
    prompt_token_identifiers_hash: str
    generated_token_count: int
    score_trace_count: int
    generated_token_identifiers_hash: str
    canonical_float32_score_trace_hash: str
    backend: str
    backend_mode: str
    response_family: str
    sampling_profile: str
    sampling_profile_fingerprint: str
    custom_sampler_identity: str | None
    custom_sampler_code_hash: str | None
    attention_implementation: str
    runtime_identity: Mapping[str, Any]
    installed_runtime_identity_fingerprint: str
    model_eval_mode: bool
    model_identity_fingerprint: str
    tokenizer_identity_fingerprint: str
    stop_reason: str
    receipt_fingerprint: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "decode_generation_policy",
            _freeze_json(self.decode_generation_policy),
        )
        object.__setattr__(
            self,
            "executed_generation_arguments",
            _freeze_json(self.executed_generation_arguments),
        )
        object.__setattr__(
            self, "runtime_identity", _freeze_json(self.runtime_identity)
        )
        if self.schema_version != DECODE_RECEIPT_SCHEMA_VERSION:
            raise RuntimeContractError(
                "decode receipt schema version is unsupported",
                code="backend_receipt.invalid_schema_version",
                context={"schema_version": self.schema_version},
            )
        if self.receipt_fingerprint != self.recompute_fingerprint():
            raise RuntimeContractError(
                "decode receipt fingerprint does not match its canonical payload",
                code="backend_receipt.fingerprint_mismatch",
                context={"request_id": self.request_id},
            )

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "decode_generation_policy": _thaw_json(self.decode_generation_policy),
            "decode_generation_policy_fingerprint": self.decode_generation_policy_fingerprint,
            "generation_config_fingerprint": self.generation_config_fingerprint,
            "sampling_seed": self.sampling_seed,
            "random_generator_kind": self.random_generator_kind,
            "random_generator_device": self.random_generator_device,
            "random_generator_initial_seed": self.random_generator_initial_seed,
            "request_execution_index": self.request_execution_index,
            "batch_request_order_fingerprint": self.batch_request_order_fingerprint,
            "executed_generation_arguments": _thaw_json(
                self.executed_generation_arguments
            ),
            "effective_generation_profile_fingerprint": self.effective_generation_profile_fingerprint,
            "prompt_token_count": self.prompt_token_count,
            "prompt_token_identifiers_hash": self.prompt_token_identifiers_hash,
            "generated_token_count": self.generated_token_count,
            "score_trace_count": self.score_trace_count,
            "generated_token_identifiers_hash": self.generated_token_identifiers_hash,
            "canonical_float32_score_trace_hash": self.canonical_float32_score_trace_hash,
            "backend": self.backend,
            "backend_mode": self.backend_mode,
            "response_family": self.response_family,
            "sampling_profile": self.sampling_profile,
            "sampling_profile_fingerprint": self.sampling_profile_fingerprint,
            "custom_sampler_identity": self.custom_sampler_identity,
            "custom_sampler_code_hash": self.custom_sampler_code_hash,
            "attention_implementation": self.attention_implementation,
            "runtime_identity": _thaw_json(self.runtime_identity),
            "installed_runtime_identity_fingerprint": self.installed_runtime_identity_fingerprint,
            "model_eval_mode": self.model_eval_mode,
            "model_identity_fingerprint": self.model_identity_fingerprint,
            "tokenizer_identity_fingerprint": self.tokenizer_identity_fingerprint,
            "stop_reason": self.stop_reason,
            "receipt_fingerprint": self.receipt_fingerprint,
        }

    @classmethod
    def from_artifact_dict(cls, payload: Mapping[str, Any]) -> DecodeExecutionReceipt:
        return cls(**dict(payload))

    def recompute_fingerprint(self) -> str:
        payload = self.to_artifact_dict()
        payload.pop("receipt_fingerprint")
        return _sha256_json(payload)


@dataclass(frozen=True)
class TokenTrace:
    step_index: int
    token_id: int
    token_text: str
    logprob: float | None
    is_stop: bool
    is_pad: bool
    backend: str
    backend_mode: str
    response_family: str


@dataclass(frozen=True)
class DecodeResult:
    request_id: str
    backend: str
    backend_mode: str
    response_family: str
    prompt_token_ids: list[int]
    generated_token_ids: list[int]
    raw_generated_text: str
    parser_text: str
    strip_policy: str
    stop_reason: str
    model_identity: Mapping[str, Any]
    tokenizer_identity: Mapping[str, Any]
    generation_config_fingerprint: str
    token_trace: list[TokenTrace]
    execution_receipt: DecodeExecutionReceipt | None = None
    execution_contract_anchor: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        anchor = self.execution_contract_anchor
        if anchor is None and self.execution_receipt is not None:
            anchor = _execution_contract_anchor(self.execution_receipt)
        if anchor is not None:
            object.__setattr__(self, "execution_contract_anchor", _freeze_json(anchor))

    def to_artifact_dict(self) -> dict[str, Any]:
        """Serialize the complete result and its result-bound execution evidence."""

        return {
            "request_id": self.request_id,
            "backend": self.backend,
            "backend_mode": self.backend_mode,
            "response_family": self.response_family,
            "prompt_token_ids": [int(value) for value in self.prompt_token_ids],
            "generated_token_ids": [int(value) for value in self.generated_token_ids],
            "raw_generated_text": self.raw_generated_text,
            "parser_text": self.parser_text,
            "strip_policy": self.strip_policy,
            "stop_reason": self.stop_reason,
            "model_identity": _thaw_json(self.model_identity),
            "tokenizer_identity": _thaw_json(self.tokenizer_identity),
            "generation_config_fingerprint": self.generation_config_fingerprint,
            "token_trace": [
                {
                    "step_index": trace.step_index,
                    "token_id": trace.token_id,
                    "token_text": trace.token_text,
                    "logprob": trace.logprob,
                    "is_stop": trace.is_stop,
                    "is_pad": trace.is_pad,
                    "backend": trace.backend,
                    "backend_mode": trace.backend_mode,
                    "response_family": trace.response_family,
                }
                for trace in self.token_trace
            ],
            "execution_receipt": None
            if self.execution_receipt is None
            else self.execution_receipt.to_artifact_dict(),
            "execution_contract_anchor": None
            if self.execution_contract_anchor is None
            else _thaw_json(self.execution_contract_anchor),
        }

    @classmethod
    def from_artifact_dict(cls, payload: Mapping[str, Any]) -> DecodeResult:
        """Reload a result without trusting its serialized receipt or anchor."""

        values = dict(payload)
        values["prompt_token_ids"] = [int(value) for value in values["prompt_token_ids"]]
        values["generated_token_ids"] = [
            int(value) for value in values["generated_token_ids"]
        ]
        values["token_trace"] = [TokenTrace(**dict(row)) for row in values["token_trace"]]
        receipt = values.get("execution_receipt")
        values["execution_receipt"] = (
            None
            if receipt is None
            else DecodeExecutionReceipt.from_artifact_dict(receipt)
        )
        return cls(**values)

    def validate_for_scored(self) -> None:
        required = {
            "prompt_token_ids": self.prompt_token_ids,
            "generated_token_ids": self.generated_token_ids,
            "token_trace": self.token_trace,
            "stop_reason": self.stop_reason,
            "backend": self.backend,
            "backend_mode": self.backend_mode,
            "response_family": self.response_family,
            "strip_policy": self.strip_policy,
            "model_identity": self.model_identity,
            "tokenizer_identity": self.tokenizer_identity,
            "generation_config_fingerprint": self.generation_config_fingerprint,
        }
        for field, value in required.items():
            if not value:
                raise RuntimeContractError(
                    "scored decode result is missing a required trace field",
                    code="backend_trace.missing_field",
                    context={"field": field, "request_id": self.request_id},
                )
        if self.strip_policy not in ALLOWED_STRIP_POLICIES:
            raise RuntimeContractError(
                "scored decode result has an invalid strip policy",
                code="backend_trace.invalid_strip_policy",
                context={
                    "strip_policy": self.strip_policy,
                    "allowed": sorted(ALLOWED_STRIP_POLICIES),
                    "request_id": self.request_id,
                },
            )
        for index, trace in enumerate(self.token_trace):
            trace_required = {
                "token_text": trace.token_text,
                "backend": trace.backend,
                "backend_mode": trace.backend_mode,
                "response_family": trace.response_family,
            }
            for field, value in trace_required.items():
                if not value:
                    raise RuntimeContractError(
                        "scored token trace is missing a required field",
                        code="backend_trace.missing_field",
                        context={
                            "field": f"token_trace.{index}.{field}",
                            "request_id": self.request_id,
                        },
                    )
            if trace.logprob is None and not trace.is_pad:
                raise RuntimeContractError(
                    "scored token trace is missing logprob for generated content",
                    code="backend_trace.missing_field",
                    context={
                        "field": f"token_trace.{index}.logprob",
                        "request_id": self.request_id,
                    },
                )
            if not trace.is_pad and trace.logprob is not None:
                canonical_float32_logprob(
                    trace.logprob,
                    error_code="backend_trace.non_finite_logprob",
                    context={
                        "field": f"token_trace.{index}.logprob",
                        "request_id": self.request_id,
                    },
                )
        self._validate_execution_receipt_binding()

    def _validate_execution_receipt_binding(self) -> None:
        receipt = self.execution_receipt
        if receipt is None:
            raise RuntimeContractError(
                "scored decode result is missing its execution receipt",
                code="backend_trace.missing_field",
                context={"field": "execution_receipt", "request_id": self.request_id},
            )
        anchor = self.execution_contract_anchor
        if anchor is None:
            raise RuntimeContractError(
                "scored decode result is missing its independent execution contract anchor",
                code="backend_trace.missing_field",
                context={
                    "field": "execution_contract_anchor",
                    "request_id": self.request_id,
                },
            )
        expected = {
            "request_id": self.request_id,
            "generation_config_fingerprint": self.generation_config_fingerprint,
            "prompt_token_count": len(self.prompt_token_ids),
            "prompt_token_identifiers_hash": _token_identifiers_hash(
                self.prompt_token_ids
            ),
            "generated_token_count": len(self.generated_token_ids),
            "score_trace_count": len(self.token_trace),
            "generated_token_identifiers_hash": _token_identifiers_hash(
                self.generated_token_ids
            ),
            "canonical_float32_score_trace_hash": _canonical_float32_score_trace_hash(
                self.token_trace
            ),
            "backend": self.backend,
            "backend_mode": self.backend_mode,
            "response_family": self.response_family,
            "model_identity_fingerprint": _sha256_json(self.model_identity),
            "tokenizer_identity_fingerprint": _sha256_json(self.tokenizer_identity),
            "stop_reason": self.stop_reason,
        }
        expected.update(_thaw_json(anchor))
        mismatches = {
            field: {"expected": expected_value, "observed": getattr(receipt, field)}
            for field, expected_value in expected.items()
            if getattr(receipt, field) != expected_value
        }
        if mismatches:
            raise RuntimeContractError(
                "decode execution receipt is not bound to its enclosing result",
                code="backend_receipt.binding_mismatch",
                context={"request_id": self.request_id, "mismatches": mismatches},
            )


@dataclass(frozen=True)
class ExecutedSampledRuntimeAttestationCase:
    """Typed evidence emitted by one actual sampled generation call."""

    schema_version: str
    case_name: str
    request_ids: tuple[str, ...]
    sampling_seeds: tuple[int, ...]
    prompt_token_identifier_hashes: tuple[str, ...]
    model_input_fingerprints: tuple[str, ...]
    prompt_width: int
    batch_size: int
    score_step_count: int
    score_tensor_shapes: tuple[tuple[int, ...], ...]
    sequence_shape: tuple[int, ...]
    generated_suffix_shape: tuple[int, ...]
    transition_score_shape: tuple[int, ...]
    output_type: str
    categorical_draw_call_count: int
    cache_steps: tuple[Mapping[str, Any], ...]
    output_cache_present: bool
    execution_device_identity: Mapping[str, Any]
    prepared_generation_profile: Mapping[str, Any]
    model_identity: Mapping[str, Any]
    tokenizer_identity: Mapping[str, Any]
    generation_config_fingerprint: str
    attention_implementation: str
    model_evaluation_mode: bool
    runtime_identity: Mapping[str, Any]
    custom_sampler_code_hash: str
    result_artifacts: tuple[Mapping[str, Any], ...]
    compact_object_selected_token_score_replays_by_request: Mapping[str, Any]
    case_payload_fingerprint: str

    def __post_init__(self) -> None:
        for field in ("cache_steps", "result_artifacts"):
            object.__setattr__(
                self,
                field,
                tuple(_freeze_json(value) for value in getattr(self, field)),
            )
        for field in (
            "execution_device_identity",
            "prepared_generation_profile",
            "model_identity",
            "tokenizer_identity",
            "runtime_identity",
            "compact_object_selected_token_score_replays_by_request",
        ):
            object.__setattr__(self, field, _freeze_json(getattr(self, field)))
        if self.schema_version != SAMPLED_RUNTIME_ATTESTATION_CASE_SCHEMA_VERSION:
            raise RuntimeContractError(
                "sampled runtime attestation case schema is unsupported",
                code="backend_sampling.attestation_case_schema_invalid",
                context={"schema_version": self.schema_version},
            )
        if self.case_payload_fingerprint != self.recompute_fingerprint():
            raise RuntimeContractError(
                "sampled runtime attestation case fingerprint is invalid",
                code="backend_sampling.attestation_case_fingerprint_mismatch",
                context={"case_name": self.case_name},
            )

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "case_name": self.case_name,
            "request_ids": list(self.request_ids),
            "sampling_seeds": list(self.sampling_seeds),
            "prompt_token_identifier_hashes": list(
                self.prompt_token_identifier_hashes
            ),
            "model_input_fingerprints": list(self.model_input_fingerprints),
            "prompt_width": self.prompt_width,
            "batch_size": self.batch_size,
            "score_step_count": self.score_step_count,
            "score_tensor_shapes": [list(shape) for shape in self.score_tensor_shapes],
            "sequence_shape": list(self.sequence_shape),
            "generated_suffix_shape": list(self.generated_suffix_shape),
            "transition_score_shape": list(self.transition_score_shape),
            "output_type": self.output_type,
            "categorical_draw_call_count": self.categorical_draw_call_count,
            "cache_steps": [_thaw_json(value) for value in self.cache_steps],
            "output_cache_present": self.output_cache_present,
            "execution_device_identity": _thaw_json(self.execution_device_identity),
            "prepared_generation_profile": _thaw_json(
                self.prepared_generation_profile
            ),
            "model_identity": _thaw_json(self.model_identity),
            "tokenizer_identity": _thaw_json(self.tokenizer_identity),
            "generation_config_fingerprint": self.generation_config_fingerprint,
            "attention_implementation": self.attention_implementation,
            "model_evaluation_mode": self.model_evaluation_mode,
            "runtime_identity": _thaw_json(self.runtime_identity),
            "custom_sampler_code_hash": self.custom_sampler_code_hash,
            "result_artifacts": [
                _thaw_json(value) for value in self.result_artifacts
            ],
            "compact_object_selected_token_score_replays_by_request": _thaw_json(
                self.compact_object_selected_token_score_replays_by_request
            ),
            "case_payload_fingerprint": self.case_payload_fingerprint,
        }

    @classmethod
    def from_artifact_dict(
        cls, payload: Mapping[str, Any]
    ) -> ExecutedSampledRuntimeAttestationCase:
        values = dict(payload)
        values["request_ids"] = tuple(str(value) for value in values["request_ids"])
        values["sampling_seeds"] = tuple(
            int(value) for value in values["sampling_seeds"]
        )
        values["prompt_token_identifier_hashes"] = tuple(
            str(value) for value in values["prompt_token_identifier_hashes"]
        )
        values["model_input_fingerprints"] = tuple(
            str(value) for value in values["model_input_fingerprints"]
        )
        values["score_tensor_shapes"] = tuple(
            tuple(int(item) for item in row) for row in values["score_tensor_shapes"]
        )
        for field in (
            "sequence_shape",
            "generated_suffix_shape",
            "transition_score_shape",
        ):
            values[field] = tuple(int(item) for item in values[field])
        values["cache_steps"] = tuple(values["cache_steps"])
        values["result_artifacts"] = tuple(values["result_artifacts"])
        return cls(**values)

    def recompute_fingerprint(self) -> str:
        payload = self.to_artifact_dict()
        payload.pop("case_payload_fingerprint")
        return _sha256_json(payload)


@dataclass(frozen=True)
class CrossCardinalityRequestReplayAttestation:
    """Persisted B3/B4 shared-request comparison recomputed by the verifier."""

    schema_version: str
    case_name: str
    shared_request_ids: tuple[str, ...]
    compared_case_names: tuple[str, ...]
    maximum_absolute_score_difference: float
    maximum_relative_score_difference: float
    exact_generated_token_replay: bool
    comparison_payload_fingerprint: str

    def __post_init__(self) -> None:
        if (
            self.schema_version
            != SAMPLED_RUNTIME_ATTESTATION_CROSS_CARDINALITY_SCHEMA_VERSION
        ):
            raise RuntimeContractError(
                "cross-cardinality attestation schema is unsupported",
                code="backend_sampling.attestation_cross_schema_invalid",
            )
        if self.comparison_payload_fingerprint != self.recompute_fingerprint():
            raise RuntimeContractError(
                "cross-cardinality attestation fingerprint is invalid",
                code="backend_sampling.attestation_cross_fingerprint_mismatch",
            )

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "case_name": self.case_name,
            "shared_request_ids": list(self.shared_request_ids),
            "compared_case_names": list(self.compared_case_names),
            "maximum_absolute_score_difference": self.maximum_absolute_score_difference,
            "maximum_relative_score_difference": self.maximum_relative_score_difference,
            "exact_generated_token_replay": self.exact_generated_token_replay,
            "comparison_payload_fingerprint": self.comparison_payload_fingerprint,
        }

    @classmethod
    def from_artifact_dict(
        cls, payload: Mapping[str, Any]
    ) -> CrossCardinalityRequestReplayAttestation:
        values = dict(payload)
        values["shared_request_ids"] = tuple(values["shared_request_ids"])
        values["compared_case_names"] = tuple(values["compared_case_names"])
        return cls(**values)

    def recompute_fingerprint(self) -> str:
        payload = self.to_artifact_dict()
        payload.pop("comparison_payload_fingerprint")
        return _sha256_json(payload)


@dataclass(frozen=True)
class ProcessedLogitParityAttestation:
    """Full float32 stock/custom first-step processed-logit tensors."""

    schema_version: str
    request_ids: tuple[str, ...]
    comparison_scope: str
    custom_tensor: Mapping[str, Any]
    stock_tensor: Mapping[str, Any]
    absolute_tolerance: float
    relative_tolerance: float
    parity_payload_fingerprint: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "custom_tensor", _freeze_json(self.custom_tensor))
        object.__setattr__(self, "stock_tensor", _freeze_json(self.stock_tensor))
        if (
            self.schema_version
            != SAMPLED_RUNTIME_ATTESTATION_PROCESSED_LOGIT_SCHEMA_VERSION
        ):
            raise RuntimeContractError(
                "processed-logit parity schema is unsupported",
                code="backend_sampling.attestation_logit_schema_invalid",
            )
        if self.parity_payload_fingerprint != self.recompute_fingerprint():
            raise RuntimeContractError(
                "processed-logit parity fingerprint is invalid",
                code="backend_sampling.attestation_logit_fingerprint_mismatch",
            )

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "request_ids": list(self.request_ids),
            "comparison_scope": self.comparison_scope,
            "custom_tensor": _thaw_json(self.custom_tensor),
            "stock_tensor": _thaw_json(self.stock_tensor),
            "absolute_tolerance": self.absolute_tolerance,
            "relative_tolerance": self.relative_tolerance,
            "parity_payload_fingerprint": self.parity_payload_fingerprint,
        }

    @classmethod
    def from_artifact_dict(
        cls, payload: Mapping[str, Any]
    ) -> ProcessedLogitParityAttestation:
        values = dict(payload)
        values["request_ids"] = tuple(values["request_ids"])
        return cls(**values)

    def recompute_fingerprint(self) -> str:
        payload = self.to_artifact_dict()
        payload.pop("parity_payload_fingerprint")
        return _sha256_json(payload)


@dataclass(frozen=True)
class SampledRuntimeAttestationBundle:
    """Reloadable evidence bundle; this object alone never grants admission."""

    schema_version: str
    lineage: Mapping[str, Any]
    executed_cases: tuple[ExecutedSampledRuntimeAttestationCase, ...]
    cross_cardinality: CrossCardinalityRequestReplayAttestation
    processed_logit_parity: ProcessedLogitParityAttestation
    bundle_payload_fingerprint: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "lineage", _freeze_json(self.lineage))
        if self.schema_version != SAMPLED_RUNTIME_ATTESTATION_BUNDLE_SCHEMA_VERSION:
            raise RuntimeContractError(
                "sampled runtime attestation bundle schema is unsupported",
                code="backend_sampling.attestation_bundle_schema_invalid",
            )
        if self.bundle_payload_fingerprint != self.recompute_fingerprint():
            raise RuntimeContractError(
                "sampled runtime attestation bundle fingerprint is invalid",
                code="backend_sampling.attestation_bundle_fingerprint_mismatch",
            )

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "lineage": _thaw_json(self.lineage),
            "executed_cases": [case.to_artifact_dict() for case in self.executed_cases],
            "cross_cardinality": self.cross_cardinality.to_artifact_dict(),
            "processed_logit_parity": self.processed_logit_parity.to_artifact_dict(),
            "bundle_payload_fingerprint": self.bundle_payload_fingerprint,
        }

    @classmethod
    def from_artifact_dict(
        cls, payload: Mapping[str, Any]
    ) -> SampledRuntimeAttestationBundle:
        values = dict(payload)
        values["executed_cases"] = tuple(
            ExecutedSampledRuntimeAttestationCase.from_artifact_dict(case)
            for case in values["executed_cases"]
        )
        values["cross_cardinality"] = (
            CrossCardinalityRequestReplayAttestation.from_artifact_dict(
                values["cross_cardinality"]
            )
        )
        values["processed_logit_parity"] = (
            ProcessedLogitParityAttestation.from_artifact_dict(
                values["processed_logit_parity"]
            )
        )
        return cls(**values)

    def recompute_fingerprint(self) -> str:
        payload = self.to_artifact_dict()
        payload.pop("bundle_payload_fingerprint")
        return _sha256_json(payload)


_VERIFIED_SAMPLED_RUNTIME_ATTESTATION_SENTINEL = object()


class VerifiedSampledRuntimeAttestation:
    """Verifier-minted, process-local sampled-production admission capability."""

    __slots__ = (
        "_admission_contract",
        "_backend_object_id",
        "_bundle_payload_fingerprint",
        "_model_object_id",
        "_runtime_state_seal",
        "_tokenizer_object_id",
    )

    def __init__(
        self,
        *,
        bundle_payload_fingerprint: str,
        admission_contract: Mapping[str, Any],
        backend_object_id: int,
        model_object_id: int,
        tokenizer_object_id: int,
        _sentinel: object,
        runtime_state_seal: Mapping[str, Any] | None = None,
    ) -> None:
        if _sentinel is not _VERIFIED_SAMPLED_RUNTIME_ATTESTATION_SENTINEL:
            raise RuntimeContractError(
                "sampled runtime admission can only be minted by the verifier",
                code="backend_sampling.attestation_capability_not_verified",
            )
        self._bundle_payload_fingerprint = bundle_payload_fingerprint
        self._admission_contract = _freeze_json(admission_contract)
        self._backend_object_id = int(backend_object_id)
        self._model_object_id = int(model_object_id)
        self._tokenizer_object_id = int(tokenizer_object_id)
        self._runtime_state_seal = _freeze_json(runtime_state_seal or {})

    @property
    def bundle_payload_fingerprint(self) -> str:
        return self._bundle_payload_fingerprint

    @property
    def admission_contract(self) -> Mapping[str, Any]:
        return self._admission_contract

    @property
    def runtime_state_seal(self) -> Mapping[str, Any]:
        return self._runtime_state_seal

    def portable_runtime_state_seal_artifact(self) -> dict[str, Any]:
        """Return cross-process runtime identity without process-local pointers."""

        return _portable_runtime_state_seal_artifact(self._runtime_state_seal)

    def is_bound_to(self, backend: HFGenerateBackend) -> bool:
        return (
            self._backend_object_id == id(backend)
            and self._model_object_id == id(backend.model)
            and self._tokenizer_object_id == id(backend.tokenizer)
        )


def _execution_contract_anchor(
    receipt: DecodeExecutionReceipt,
) -> dict[str, Any]:
    """Copy non-output execution facts outside the replaceable receipt."""

    fields = (
        "decode_generation_policy",
        "decode_generation_policy_fingerprint",
        "sampling_seed",
        "random_generator_kind",
        "random_generator_device",
        "random_generator_initial_seed",
        "request_execution_index",
        "batch_request_order_fingerprint",
        "executed_generation_arguments",
        "effective_generation_profile_fingerprint",
        "sampling_profile",
        "sampling_profile_fingerprint",
        "custom_sampler_identity",
        "custom_sampler_code_hash",
        "attention_implementation",
        "runtime_identity",
        "installed_runtime_identity_fingerprint",
        "model_eval_mode",
    )
    return {field: _thaw_json(getattr(receipt, field)) for field in fields}


class HFGenerateBackend:
    backend = "hf"
    backend_mode = "generate"
    response_family = "hf"

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        model_identity: Mapping[str, Any] | None = None,
        tokenizer_identity: Mapping[str, Any] | None = None,
        generation_config_fingerprint: str | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self._bound_model_identity = (
            None if model_identity is None else _freeze_json(model_identity)
        )
        self._bound_tokenizer_identity = (
            None if tokenizer_identity is None else _freeze_json(tokenizer_identity)
        )
        self._bound_generation_config_fingerprint = (
            generation_config_fingerprint
        )
        self._last_sampling_attestation_diagnostics: Mapping[str, Any] | None = None
        self._last_runtime_state_seal_diagnostics: Mapping[str, Any] | None = None
        self._last_sampling_attestation_case: (
            ExecutedSampledRuntimeAttestationCase | None
        ) = None
        self._last_sampling_attestation_first_processed_scores: (
            torch.Tensor | None
        ) = None

    @property
    def last_sampling_attestation_diagnostics(self) -> Mapping[str, Any] | None:
        """Nondeterministic timing evidence kept outside stable result receipts."""

        return self._last_sampling_attestation_diagnostics

    @property
    def last_runtime_state_seal_diagnostics(self) -> Mapping[str, Any] | None:
        """Return bounded live-state hashing cost from the latest seal check."""

        return self._last_runtime_state_seal_diagnostics

    @property
    def last_sampling_attestation_case(
        self,
    ) -> ExecutedSampledRuntimeAttestationCase | None:
        """Return typed evidence for the latest explicitly named probe call."""

        return self._last_sampling_attestation_case

    def _bound_runtime_identities(self) -> tuple[dict[str, Any], dict[str, Any], str]:
        if (
            self._bound_model_identity is None
            or self._bound_tokenizer_identity is None
            or not isinstance(self._bound_generation_config_fingerprint, str)
            or not self._bound_generation_config_fingerprint
        ):
            raise RuntimeContractError(
                "sampled runtime rebind requires backend-owned runtime identities",
                code="backend_sampling.attestation_backend_identity_unbound",
            )
        return (
            _thaw_json(self._bound_model_identity),
            _thaw_json(self._bound_tokenizer_identity),
            self._bound_generation_config_fingerprint,
        )

    def generate_batch(
        self,
        requests: Sequence[DecodeRequest],
        *,
        model_identity: Mapping[str, Any],
        tokenizer_identity: Mapping[str, Any],
        generation_config_fingerprint: str,
    ) -> list[DecodeResult]:
        return self._generate_batch(
            requests,
            model_identity=model_identity,
            tokenizer_identity=tokenizer_identity,
            generation_config_fingerprint=generation_config_fingerprint,
        )

    def run_request_scoped_sampling_attestation(
        self,
        requests: Sequence[DecodeRequest],
        *,
        lineage: Mapping[str, Any],
        model_identity: Mapping[str, Any],
        tokenizer_identity: Mapping[str, Any],
        generation_config_fingerprint: str,
    ) -> tuple[SampledRuntimeAttestationBundle, VerifiedSampledRuntimeAttestation]:
        """Execute the complete frozen gate and mint one process-local capability."""

        canonical_requests = list(requests)
        enriched_lineage = _bind_sampling_attestation_requests(
            lineage,
            requests=canonical_requests,
        )
        _validate_sampling_attestation_execution_request(
            enriched_lineage,
            requests=canonical_requests,
            model=self.model,
            tokenizer=self.tokenizer,
            tokenizer_identity=tokenizer_identity,
        )
        cases: list[ExecutedSampledRuntimeAttestationCase] = []
        image_sizes_by_request = {
            str(row["request_id"]): (
                int(row["image_width"]),
                int(row["image_height"]),
            )
            for row in enriched_lineage["calibration_request_plan"]
        }

        def execute(
            case_name: str, case_requests: Sequence[DecodeRequest]
        ) -> None:
            self._generate_batch(
                case_requests,
                model_identity=model_identity,
                tokenizer_identity=tokenizer_identity,
                generation_config_fingerprint=generation_config_fingerprint,
                execution_context="sampling_attestation",
                sampling_attestation_case_name=case_name,
                attestation_image_sizes_by_request=image_sizes_by_request,
            )
            case = self._last_sampling_attestation_case
            if case is None or case.case_name != case_name:
                raise RuntimeContractError(
                    "bounded attestation call did not emit its named case",
                    code="backend_sampling.attestation_case_incomplete",
                    context={"case_name": case_name},
                )
            cases.append(case)

        execute("batch_size_four_forward", canonical_requests)
        processed_logit_parity = self._attest_processed_logit_parity_against_stock(
            canonical_requests
        )
        execute("batch_size_four_reversed", list(reversed(canonical_requests)))
        execute("batch_size_three_forward", canonical_requests[:3])
        execute(
            "batch_size_three_reversed",
            list(reversed(canonical_requests[:3])),
        )
        bundle = build_sampled_runtime_attestation_bundle(
            lineage=enriched_lineage,
            executed_cases=cases,
            processed_logit_parity=processed_logit_parity,
        )
        reloaded = SampledRuntimeAttestationBundle.from_artifact_dict(
            bundle.to_artifact_dict()
        )
        parsed, admission_contract = _validate_sampled_runtime_attestation_bundle(
            reloaded
        )
        capability = _mint_verified_sampled_runtime_attestation(
            parsed,
            admission_contract=admission_contract,
            backend=self,
            model_identity=model_identity,
            tokenizer_identity=tokenizer_identity,
            generation_config_fingerprint=generation_config_fingerprint,
        )
        return parsed, capability

    def generate_batch_with_verified_runtime_attestation(
        self,
        requests: Sequence[DecodeRequest],
        *,
        model_identity: Mapping[str, Any],
        tokenizer_identity: Mapping[str, Any],
        generation_config_fingerprint: str,
        verified_runtime_attestation: VerifiedSampledRuntimeAttestation,
    ) -> list[DecodeResult]:
        """Run sampled production only with a verifier-minted capability."""

        return self._generate_batch(
            requests,
            model_identity=model_identity,
            tokenizer_identity=tokenizer_identity,
            generation_config_fingerprint=generation_config_fingerprint,
            execution_context="production",
            verified_runtime_attestation=verified_runtime_attestation,
        )

    def _attest_processed_logit_parity_against_stock(
        self,
        requests: Sequence[DecodeRequest],
        *,
        absolute_tolerance: float = 1e-6,
        relative_tolerance: float = 1e-6,
    ) -> ProcessedLogitParityAttestation:
        """Compare the actual custom and stock first-step processed logits."""

        case = self._last_sampling_attestation_case
        custom_scores = self._last_sampling_attestation_first_processed_scores
        request_ids = tuple(request.request_id for request in requests)
        if (
            case is None
            or case.case_name != "batch_size_four_forward"
            or case.request_ids != request_ids
            or custom_scores is None
        ):
            raise RuntimeContractError(
                "stock parity requires the immediately preceding named four-row custom call",
                code="backend_sampling.attestation_logit_missing_custom_baseline",
                context={"request_ids": list(request_ids)},
            )
        policy = validate_decode_batch(requests)
        if policy.mode != "sampled" or len(requests) != 4:
            raise RuntimeContractError(
                "processed-logit parity requires four sampled requests",
                code="backend_sampling.attestation_logit_invalid_batch",
            )
        prompt_width = max(len(request.prompt_token_ids) for request in requests)
        target_device = self._target_device(requests)
        input_ids, attention_mask = self._padded_prompt_tensors(
            requests, prompt_width, device=target_device
        )
        generate_inputs = self._collate_generate_inputs(requests, device=target_device)
        generate_inputs["input_ids"] = input_ids
        generate_inputs["attention_mask"] = attention_mask
        arguments = effective_generation_arguments(
            policy,
            eos_token_id=self._im_end_token_id(),
            pad_token_id=self._pad_token_id(),
            bos_token_id=self._bos_token_id(),
            decoder_start_token_id=self._decoder_start_token_id(),
        )
        generation_config = _generation_config_from_arguments(arguments, policy=policy)
        generation_config.max_new_tokens = 1
        generation_config.max_length = prompt_width + 1
        capture = _ProcessedLogitCapture()
        try:
            from transformers.generation.logits_process import LogitsProcessorList
        except ImportError as exc:
            raise RuntimeContractError(
                "installed Transformers logits processors are unavailable",
                code="backend_sampling.transformers_unavailable",
                cause=exc,
            ) from exc
        self.model.generate(
            **generate_inputs,
            generation_config=generation_config,
            logits_processor=LogitsProcessorList([capture]),
            use_model_defaults=False,
        )
        stock_scores = capture.first_scores_float32
        if stock_scores is None:
            raise RuntimeContractError(
                "stock generation did not expose first-step processed logits",
                code="backend_sampling.attestation_logit_stock_capture_missing",
            )
        return _build_processed_logit_parity_attestation(
            request_ids=request_ids,
            custom_scores=custom_scores,
            stock_scores=stock_scores,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        )

    def _generate_batch(
        self,
        requests: Sequence[DecodeRequest],
        *,
        model_identity: Mapping[str, Any],
        tokenizer_identity: Mapping[str, Any],
        generation_config_fingerprint: str,
        execution_context: Literal["production", "sampling_attestation"] = "production",
        sampling_attestation_case_name: str | None = None,
        verified_runtime_attestation: VerifiedSampledRuntimeAttestation | None = None,
        attestation_image_sizes_by_request: Mapping[str, tuple[int, int]] | None = None,
    ) -> list[DecodeResult]:
        if not requests:
            return []
        self._last_sampling_attestation_case = None
        self._last_sampling_attestation_first_processed_scores = None
        if execution_context not in {"production", "sampling_attestation"}:
            raise RuntimeContractError(
                "decode execution context is unsupported",
                code="backend_sampling.invalid_execution_context",
                context={"execution_context": execution_context},
            )
        policy = validate_decode_batch(requests)
        if sampling_attestation_case_name is not None and (
            execution_context != "sampling_attestation"
            or sampling_attestation_case_name
            not in _REQUIRED_EXECUTED_SAMPLED_RUNTIME_ATTESTATION_CASES
        ):
            raise RuntimeContractError(
                "sampling attestation case name is not an authorized executed case",
                code="backend_sampling.attestation_case_name_invalid",
                context={"case_name": sampling_attestation_case_name},
            )
        if (
            policy.mode == "sampled"
            and execution_context != "sampling_attestation"
            and verified_runtime_attestation is None
        ):
            raise RuntimeContractError(
                "request-scoped sampled generation requires a verifier-minted capability "
                "from a typed executed CUDA/Qwen attestation bundle",
                code="backend_sampling.runtime_not_attested",
                context={
                    "request_ids": [request.request_id for request in requests],
                    "required_bundle_schema_version": SAMPLED_RUNTIME_ATTESTATION_BUNDLE_SCHEMA_VERSION,
                    "required_cases": sorted(
                        _REQUIRED_SAMPLED_RUNTIME_ATTESTATION_CASES
                    ),
                    "required_checks": sorted(
                        _REQUIRED_SAMPLED_RUNTIME_ATTESTATION_CHECKS
                    ),
                    "production_admission_api_status": "verified_capability_required",
                },
            )
        prompt_width = max(len(request.prompt_token_ids) for request in requests)
        target_device = self._target_device(requests)
        input_ids, attention_mask = self._padded_prompt_tensors(
            requests,
            prompt_width,
            device=target_device,
        )
        generate_inputs = self._collate_generate_inputs(requests, device=target_device)
        generate_inputs["input_ids"] = input_ids
        generate_inputs["attention_mask"] = attention_mask
        requested_generation_arguments = effective_generation_arguments(
            policy,
            eos_token_id=self._im_end_token_id(),
            pad_token_id=self._pad_token_id(),
            bos_token_id=self._bos_token_id(),
            decoder_start_token_id=self._decoder_start_token_id(),
        )
        generation_config = _generation_config_from_arguments(
            requested_generation_arguments,
            policy=policy,
        )
        # Own the post-preparation length explicitly instead of relying on the
        # installed helper to mutate the default max_length behind the receipt.
        generation_config.max_length = prompt_width + policy.max_new_tokens
        if policy.mode == "sampled" and execution_context == "production":
            assert verified_runtime_attestation is not None
            _validate_verified_sampled_runtime_for_active_call(
                verified_runtime_attestation,
                backend=self,
                requests=requests,
                model_identity=model_identity,
                tokenizer_identity=tokenizer_identity,
                generation_config_fingerprint=generation_config_fingerprint,
                attention_implementation=_attention_implementation(self.model),
                model_evaluation_mode=not bool(
                    getattr(self.model, "training", False)
                ),
                runtime_identity=_runtime_identity(),
                execution_device_identity=_execution_device_identity(target_device),
                prepared_generation_profile=_authoritative_generation_profile(
                    generation_config,
                    policy=policy,
                    use_model_defaults=False,
                ),
                prompt_width=prompt_width,
            )
        execution_capture = _GenerationExecutionCapture()
        request_generators: tuple[torch.Generator, ...] | None = None
        sampling_timing_recorder: _SamplingTimingRecorder | None = None
        full_batch_elapsed_seconds: float | None = None
        if policy.mode == "sampled":
            request_generators = tuple(
                torch.Generator(device=target_device).manual_seed(
                    int(request.sampling_seed)
                )
                for request in requests
            )
            validate_request_generators(
                requests, request_generators, device=target_device
            )
            sampling_timing_recorder = _SamplingTimingRecorder()
            _synchronize_device_for_timing(target_device)
            started_at = time.perf_counter()
            outputs = custom_generate(
                self.model,
                request_generators=request_generators,
                sampling_timing=sampling_timing_recorder,
                execution_capture=execution_capture,
                generation_policy=policy,
                **generate_inputs,
                generation_config=generation_config,
            )
            _synchronize_device_for_timing(target_device)
            full_batch_elapsed_seconds = time.perf_counter() - started_at
        else:
            outputs = self.model.generate(
                **generate_inputs,
                generation_config=generation_config,
                use_model_defaults=False,
            )
        if policy.mode == "sampled":
            if (
                not execution_capture.custom_sampler_executed
                or execution_capture.prepared_generation_profile is None
            ):
                raise RuntimeContractError(
                    "sampled generation did not execute the request-scoped callable",
                    code="backend_sampling.custom_callable_not_executed",
                    context={
                        "request_ids": [request.request_id for request in requests]
                    },
                )
            executed_arguments = execution_capture.prepared_generation_profile
        else:
            executed_arguments = _authoritative_generation_profile(
                generation_config,
                policy=policy,
                use_model_defaults=False,
            )
        scores = getattr(outputs, "scores", None)
        if scores is None:
            raise RuntimeContractError(
                "HF generation returned no per-step scores for scored inference",
                code="backend_trace.missing_scores",
                context={"backend": self.backend},
            )
        scores = tuple(scores)
        if not scores:
            raise RuntimeContractError(
                "HF generation returned an empty score trace",
                code="backend_trace.missing_scores",
                context={"backend": self.backend},
            )
        score_tensors = self._validate_score_tensors(scores, batch_size=len(requests))
        sequences = getattr(outputs, "sequences", None)
        if sequences is None:
            raise RuntimeContractError(
                "HF generation returned no sequences",
                code="backend_trace.missing_sequences",
                context={"backend": self.backend},
            )
        sequences = _as_tensor(sequences)
        if sequences.ndim != 2 or sequences.shape[0] != len(requests):
            raise RuntimeContractError(
                "HF generation sequence shape does not match the decode batch",
                code="backend_trace.shape_mismatch",
                context={
                    "sequence_shape": tuple(sequences.shape),
                    "batch_size": len(requests),
                },
            )
        if sequences.shape[1] != prompt_width + len(scores):
            raise RuntimeContractError(
                "HF generation sequence length must equal prompt width plus score steps",
                code="backend_trace.shape_mismatch",
                context={
                    "sequence_length": int(sequences.shape[1]),
                    "expected_sequence_length": prompt_width + len(scores),
                    "prompt_width": prompt_width,
                    "score_steps": len(scores),
                },
            )
        generated_suffix = sequences[:, prompt_width : prompt_width + len(scores)]
        transition_scores = self._transition_scores(
            sequences,
            generated_suffix=generated_suffix,
            scores=score_tensors,
        )
        if transition_scores.shape != (len(requests), len(scores)):
            raise RuntimeContractError(
                "HF transition score shape does not match generated score steps",
                code="backend_trace.shape_mismatch",
                context={
                    "transition_shape": tuple(transition_scores.shape),
                    "batch_size": len(requests),
                    "score_steps": len(scores),
                },
            )
        batch_request_order_fingerprint = _batch_request_order_fingerprint(
            requests,
            policy=policy,
        )
        self._last_sampling_attestation_diagnostics = (
            None
            if sampling_timing_recorder is None
            else _freeze_json(
                sampling_timing_recorder.to_artifact_dict(
                    full_batch_elapsed_seconds=float(full_batch_elapsed_seconds),
                    batch_size=len(requests),
                    score_steps=len(scores),
                    device=target_device,
                )
            )
        )
        results = [
            self._materialize_result(
                request=request,
                row_index=row,
                generated_ids=[
                    int(token_id) for token_id in generated_suffix[row].tolist()
                ],
                transition_logprobs=[
                    float(value) for value in transition_scores[row].tolist()
                ],
                model_identity=model_identity,
                tokenizer_identity=tokenizer_identity,
                generation_config_fingerprint=generation_config_fingerprint,
                executed_generation_arguments=executed_arguments,
                batch_request_order_fingerprint=batch_request_order_fingerprint,
                request_generator=None
                if request_generators is None
                else request_generators[row],
                custom_sampler_executed=execution_capture.custom_sampler_executed,
            )
            for row, request in enumerate(requests)
        ]
        validate_decode_execution_batch(
            requests,
            results,
            model_identity=model_identity,
            tokenizer_identity=tokenizer_identity,
            generation_config_fingerprint=generation_config_fingerprint,
            executed_generation_arguments=executed_arguments,
            backend=self.backend,
            backend_mode=self.backend_mode,
            response_family=self.response_family,
            attention_implementation=_attention_implementation(self.model),
            model_eval_mode=not bool(getattr(self.model, "training", False)),
            runtime_identity=_runtime_identity(),
            request_generators=request_generators,
        )
        if policy.mode == "sampled":
            first_scores = execution_capture.first_processed_scores_float32
            self._last_sampling_attestation_first_processed_scores = (
                None if first_scores is None else first_scores.clone()
            )
        if sampling_attestation_case_name is not None:
            assert sampling_timing_recorder is not None
            self._last_sampling_attestation_case = _build_executed_attestation_case(
                case_name=sampling_attestation_case_name,
                requests=requests,
                results=results,
                prompt_width=prompt_width,
                scores=score_tensors,
                sequences=sequences,
                generated_suffix=generated_suffix,
                transition_scores=transition_scores,
                outputs=outputs,
                execution_capture=execution_capture,
                sampling_timing=sampling_timing_recorder,
                target_device=target_device,
                model_identity=model_identity,
                tokenizer_identity=tokenizer_identity,
                generation_config_fingerprint=generation_config_fingerprint,
                attention_implementation=_attention_implementation(self.model),
                model_evaluation_mode=not bool(
                    getattr(self.model, "training", False)
                ),
                image_sizes_by_request=attestation_image_sizes_by_request
                or {request.request_id: (1000, 1000) for request in requests},
            )
        return results

    def _batch_max_new_tokens(self, requests: Sequence[DecodeRequest]) -> int:
        values = {request.generation_policy.max_new_tokens for request in requests}
        if len(values) != 1:
            raise RuntimeContractError(
                "all decode requests in a V1 HF batch must use the same max_new_tokens",
                code="backend_trace.max_new_tokens_mismatch",
                context={"max_new_tokens": sorted(values)},
            )
        return values.pop()

    def _batch_repetition_penalty(self, requests: Sequence[DecodeRequest]) -> float:
        values = {
            float(request.generation_policy.repetition_penalty) for request in requests
        }
        if len(values) != 1:
            raise RuntimeContractError(
                "all decode requests in a V1 HF batch must use the same repetition_penalty",
                code="backend_trace.repetition_penalty_mismatch",
                context={"repetition_penalty": sorted(values)},
            )
        return values.pop()

    def _target_device(self, requests: Sequence[DecodeRequest]) -> torch.device:
        devices = {
            value.device
            for request in requests
            for value in request.model_inputs.values()
            if isinstance(value, torch.Tensor)
        }
        model_device = self._model_device()
        non_cpu_devices = {device for device in devices if device.type != "cpu"}
        if model_device is not None:
            if non_cpu_devices and non_cpu_devices != {model_device}:
                raise RuntimeContractError(
                    "decode request tensors must match the HF model device before generation",
                    code="backend_trace.device_mismatch",
                    context={
                        "model_device": str(model_device),
                        "devices": sorted(str(device) for device in devices),
                    },
                )
            return model_device
        if len(devices) > 1:
            raise RuntimeContractError(
                "decode request tensors must be on one device before HF generation",
                code="backend_trace.device_mismatch",
                context={"devices": sorted(str(device) for device in devices)},
            )
        if devices:
            return next(iter(devices))
        return torch.device("cpu")

    def _model_device(self) -> torch.device | None:
        if hasattr(self.model, "parameters"):
            try:
                first_param = next(iter(self.model.parameters()))
            except StopIteration:
                first_param = None
            if first_param is not None:
                return first_param.device
        return None

    def _padded_prompt_tensors(
        self,
        requests: Sequence[DecodeRequest],
        prompt_width: int,
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pad_id = self._pad_token_id()
        rows = []
        masks = []
        for request in requests:
            row = list(request.prompt_token_ids)
            if len(row) > prompt_width:
                raise RuntimeContractError(
                    "prompt token length exceeds padded prompt width",
                    code="backend_trace.shape_mismatch",
                    context={
                        "request_id": request.request_id,
                        "prompt_length": len(row),
                        "prompt_width": prompt_width,
                    },
                )
            padding = [pad_id] * (prompt_width - len(row))
            rows.append(padding + row)
            masks.append([0] * len(padding) + [1] * len(row))
        return (
            torch.tensor(rows, dtype=torch.long, device=device),
            torch.tensor(masks, dtype=torch.long, device=device),
        )

    def _collate_generate_inputs(
        self,
        requests: Sequence[DecodeRequest],
        *,
        device: torch.device,
    ) -> dict[str, Any]:
        collated: dict[str, Any] = {}
        reserved = {"input_ids", "attention_mask"}
        keys = {
            key
            for request in requests
            for key in request.model_inputs
            if key not in reserved
        }
        for key in sorted(keys):
            values = [request.model_inputs.get(key) for request in requests]
            present = [value for value in values if value is not None]
            if not present:
                continue
            if len(present) != len(requests):
                raise RuntimeContractError(
                    "all decode requests in a batch must provide the same model input keys",
                    code="backend_trace.model_input_mismatch",
                    context={"field": key},
                )
            collated[key] = _move_to_device(
                _collate_model_input_values(present, key=key),
                device=device,
            )
        return collated

    def _validate_score_tensors(
        self,
        scores: tuple[Any, ...],
        *,
        batch_size: int,
    ) -> tuple[torch.Tensor, ...]:
        validated = []
        for step_index, score in enumerate(scores):
            score_tensor = _as_tensor(score)
            if score_tensor.ndim != 2 or score_tensor.shape[0] != batch_size:
                raise RuntimeContractError(
                    "HF generation score tensor shape does not match decode batch",
                    code="backend_trace.score_shape_mismatch",
                    context={
                        "step_index": step_index,
                        "score_shape": tuple(score_tensor.shape),
                        "batch_size": batch_size,
                    },
                )
            validated.append(score_tensor)
        return tuple(validated)

    def _transition_scores(
        self,
        sequences: torch.Tensor,
        *,
        generated_suffix: torch.Tensor,
        scores: tuple[Any, ...],
    ) -> torch.Tensor:
        compute_transition_scores = getattr(
            self.model, "compute_transition_scores", None
        )
        if callable(compute_transition_scores):
            try:
                return _as_tensor(
                    compute_transition_scores(
                        sequences,
                        scores,
                        normalize_logits=True,
                    )
                )
            except Exception as exc:
                raise RuntimeContractError(
                    "HF transition score extraction failed",
                    code="backend_trace.transition_scores_failed",
                    context={
                        "sequence_shape": tuple(sequences.shape),
                        "score_shapes": [tuple(score.shape) for score in scores],
                    },
                    cause=exc,
                ) from exc
        rows = []
        for step_index, step_scores in enumerate(scores):
            logprobs = F.log_softmax(_as_tensor(step_scores), dim=-1)
            rows.append(
                logprobs.gather(1, generated_suffix[:, step_index : step_index + 1])
            )
        return torch.cat(rows, dim=1)

    def _materialize_result(
        self,
        *,
        request: DecodeRequest,
        row_index: int,
        generated_ids: list[int],
        transition_logprobs: list[float],
        model_identity: Mapping[str, Any],
        tokenizer_identity: Mapping[str, Any],
        generation_config_fingerprint: str,
        executed_generation_arguments: Mapping[str, Any],
        batch_request_order_fingerprint: str,
        request_generator: torch.Generator | None,
        custom_sampler_executed: bool,
    ) -> DecodeResult:
        stop_id = self._im_end_token_id()
        pad_id = self._pad_token_id()
        kept_ids: list[int] = []
        traces: list[TokenTrace] = []
        seen_stop = False
        stop_reason = "length"
        for step_index, (token_id, logprob) in enumerate(
            zip(generated_ids, transition_logprobs, strict=True)
        ):
            if seen_stop and token_id != pad_id:
                raise RuntimeContractError(
                    "HF generation emitted a non-padding token after im_end",
                    code="backend_trace.non_pad_after_stop",
                    context={
                        "row_index": row_index,
                        "request_id": request.request_id,
                        "step_index": step_index,
                        "token_id": token_id,
                    },
                )
            is_post_stop_pad = seen_stop and token_id == pad_id
            if token_id == pad_id and not seen_stop:
                raise RuntimeContractError(
                    "HF generation emitted a pad token before any stop token",
                    code="backend_trace.unexpected_pad_token",
                    context={
                        "row_index": row_index,
                        "request_id": request.request_id,
                        "step_index": step_index,
                        "token_id": token_id,
                    },
                )
            is_pad = is_post_stop_pad
            is_stop = token_id == stop_id and not seen_stop
            token_text = self._decode_token(token_id)
            token_logprob = None if is_pad else logprob
            traces.append(
                TokenTrace(
                    step_index=step_index,
                    token_id=token_id,
                    token_text=token_text,
                    logprob=token_logprob,
                    is_stop=is_stop,
                    is_pad=is_pad,
                    backend=self.backend,
                    backend_mode=self.backend_mode,
                    response_family=self.response_family,
                )
            )
            if is_post_stop_pad or is_pad:
                continue
            kept_ids.append(token_id)
            if is_stop:
                seen_stop = True
                stop_reason = "im_end"
        raw_generated_text = self._decode_tokens(kept_ids)
        parser_text, strip_policy = _strip_terminal_im_end(
            raw_generated_text,
            stop_id=stop_id,
            kept_ids=kept_ids,
            stop_text=self._decode_token(stop_id),
        )
        execution_receipt = self._execution_receipt(
            request=request,
            row_index=row_index,
            generated_token_ids=kept_ids,
            token_trace=traces,
            stop_reason=stop_reason,
            model_identity=model_identity,
            tokenizer_identity=tokenizer_identity,
            generation_config_fingerprint=generation_config_fingerprint,
            executed_generation_arguments=executed_generation_arguments,
            batch_request_order_fingerprint=batch_request_order_fingerprint,
            request_generator=request_generator,
            custom_sampler_executed=custom_sampler_executed,
        )
        result = DecodeResult(
            request_id=request.request_id,
            backend=self.backend,
            backend_mode=self.backend_mode,
            response_family=self.response_family,
            prompt_token_ids=list(request.prompt_token_ids),
            generated_token_ids=kept_ids,
            raw_generated_text=raw_generated_text,
            parser_text=parser_text,
            strip_policy=strip_policy,
            stop_reason=stop_reason,
            model_identity=dict(model_identity),
            tokenizer_identity=dict(tokenizer_identity),
            generation_config_fingerprint=generation_config_fingerprint,
            token_trace=traces,
            execution_receipt=execution_receipt,
        )
        result.validate_for_scored()
        return result

    def _execution_receipt(
        self,
        *,
        request: DecodeRequest,
        row_index: int,
        generated_token_ids: list[int],
        token_trace: list[TokenTrace],
        stop_reason: str,
        model_identity: Mapping[str, Any],
        tokenizer_identity: Mapping[str, Any],
        generation_config_fingerprint: str,
        executed_generation_arguments: Mapping[str, Any],
        batch_request_order_fingerprint: str,
        request_generator: torch.Generator | None,
        custom_sampler_executed: bool,
    ) -> DecodeExecutionReceipt:
        return build_decode_execution_receipt(
            request=request,
            generated_token_ids=generated_token_ids,
            token_trace=token_trace,
            stop_reason=stop_reason,
            model_identity=model_identity,
            tokenizer_identity=tokenizer_identity,
            generation_config_fingerprint=generation_config_fingerprint,
            executed_generation_arguments=executed_generation_arguments,
            backend=self.backend,
            backend_mode=self.backend_mode,
            response_family=self.response_family,
            request_execution_index=row_index,
            batch_request_order_fingerprint=batch_request_order_fingerprint,
            request_generator=request_generator,
            custom_sampler_executed=custom_sampler_executed,
            attention_implementation=_attention_implementation(self.model),
            model_eval_mode=not bool(getattr(self.model, "training", False)),
        )

    def _im_end_token_id(self) -> int:
        if hasattr(self.tokenizer, "convert_tokens_to_ids"):
            token_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
            if token_id is not None:
                return int(token_id)
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_token_id is None:
            raise RuntimeContractError(
                "tokenizer does not expose Qwen im_end/eos token id",
                code="backend_trace.missing_stop_token",
            )
        return int(eos_token_id)

    def _pad_token_id(self) -> int:
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            raise RuntimeContractError(
                "tokenizer does not expose pad token id",
                code="backend_trace.missing_pad_token",
            )
        return int(pad_token_id)

    def _bos_token_id(self) -> int | None:
        token_id = getattr(self.tokenizer, "bos_token_id", None)
        if token_id is None:
            generation_config = getattr(self.model, "generation_config", None)
            token_id = getattr(generation_config, "bos_token_id", None)
        return None if token_id is None else int(token_id)

    def _decoder_start_token_id(self) -> int | None:
        generation_config = getattr(self.model, "generation_config", None)
        token_id = getattr(generation_config, "decoder_start_token_id", None)
        return None if token_id is None else int(token_id)

    def _decode_token(self, token_id: int) -> str:
        return self._decode_tokens([token_id])

    def _decode_tokens(self, token_ids: list[int]) -> str:
        if not token_ids:
            return ""
        return str(self.tokenizer.decode(token_ids, skip_special_tokens=False))


def validate_decode_batch(requests: Sequence[DecodeRequest]) -> DecodeGenerationPolicy:
    if not requests:
        raise RuntimeContractError(
            "decode batch validation requires at least one request",
            code="backend_policy.empty_batch",
        )
    request_ids = [request.request_id for request in requests]
    duplicate_ids = sorted(
        {request_id for request_id in request_ids if request_ids.count(request_id) > 1}
    )
    if duplicate_ids:
        raise RuntimeContractError(
            "decode batch request identities must be unique",
            code="backend_policy.duplicate_request_id",
            context={"duplicate_request_ids": duplicate_ids},
        )
    policy = requests[0].generation_policy
    baseline = policy.to_artifact_dict()
    differing_fields = sorted(
        {
            field
            for request in requests[1:]
            for field, value in request.generation_policy.to_artifact_dict().items()
            if baseline[field] != value
        }
    )
    if differing_fields:
        raise RuntimeContractError(
            "all requests in one HF call must use identical generation policies",
            code="backend_policy.incompatible_batch",
            context={
                "request_ids": request_ids,
                "differing_fields": differing_fields,
                "policies": [
                    request.generation_policy.to_artifact_dict() for request in requests
                ],
            },
        )
    return policy


def effective_generation_arguments(
    policy: DecodeGenerationPolicy,
    *,
    eos_token_id: int,
    pad_token_id: int,
    bos_token_id: int | None = None,
    decoder_start_token_id: int | None = None,
) -> dict[str, Any]:
    """Return the complete fail-closed Transformers 4.57.1 generation profile."""

    arguments = {
        "max_length": 20,
        "max_new_tokens": policy.max_new_tokens,
        "min_length": 0,
        "min_new_tokens": None,
        "early_stopping": False,
        "max_time": None,
        "stop_strings": None,
        "do_sample": policy.mode == "sampled",
        "num_beams": 1,
        "use_cache": True,
        "cache_implementation": None,
        "cache_config": None,
        "return_legacy_cache": None,
        "prefill_chunk_size": None,
        "repetition_penalty": float(policy.repetition_penalty),
        "temperature": float(policy.temperature),
        "top_k": 0,
        "top_p": float(policy.top_p),
        "min_p": None,
        "typical_p": 1.0,
        "epsilon_cutoff": 0.0,
        "eta_cutoff": 0.0,
        "encoder_repetition_penalty": 1.0,
        "length_penalty": 1.0,
        "no_repeat_ngram_size": 0,
        "bad_words_ids": None,
        "renormalize_logits": False,
        "forced_bos_token_id": None,
        "forced_eos_token_id": None,
        "remove_invalid_values": False,
        "exponential_decay_length_penalty": None,
        "suppress_tokens": None,
        "begin_suppress_tokens": None,
        "sequence_bias": None,
        "token_healing": False,
        "guidance_scale": None,
        "watermarking_config": None,
        "num_return_sequences": 1,
        "output_attentions": False,
        "output_hidden_states": False,
        "output_scores": True,
        "output_logits": False,
        "return_dict_in_generate": True,
        "pad_token_id": int(pad_token_id),
        "bos_token_id": None if bos_token_id is None else int(bos_token_id),
        "eos_token_id": int(eos_token_id),
        "encoder_no_repeat_ngram_size": 0,
        "decoder_start_token_id": (
            None if decoder_start_token_id is None else int(decoder_start_token_id)
        ),
        "is_assistant": False,
        "num_assistant_tokens": 20,
        "num_assistant_tokens_schedule": "constant",
        "assistant_confidence_threshold": 0.4,
        "prompt_lookup_num_tokens": None,
        "max_matching_ngram_size": None,
        "assistant_early_exit": None,
        "assistant_lookbehind": 10,
        "target_lookbehind": 10,
        "compile_config": None,
        "disable_compile": False,
        "low_memory": None,
        "penalty_alpha": None,
        "dola_layers": None,
        "diversity_penalty": 0.0,
        "num_beam_groups": 1,
        "constraints": None,
        "force_words_ids": None,
        "_from_model_config": False,
        "transformers_version": _package_version("transformers"),
        "use_model_defaults": False,
    }
    expected_fields = _GENERATION_CONFIG_PROFILE_FIELDS | {"use_model_defaults"}
    if set(arguments) != expected_fields:
        raise RuntimeContractError(
            "authored generation profile does not cover the authoritative field set",
            code="backend_policy.incomplete_generation_profile",
            context={
                "missing_fields": sorted(expected_fields - set(arguments)),
                "unknown_fields": sorted(set(arguments) - expected_fields),
            },
        )
    return arguments


def _generation_config_from_arguments(
    arguments: Mapping[str, Any],
    *,
    policy: DecodeGenerationPolicy,
) -> Any:
    try:
        from transformers import GenerationConfig
    except ImportError as exc:
        raise RuntimeContractError(
            "Hugging Face GenerationConfig is unavailable",
            code="backend_sampling.transformers_unavailable",
            cause=exc,
        ) from exc
    config_arguments = dict(arguments)
    if config_arguments.pop("use_model_defaults", None) is not False:
        raise RuntimeContractError(
            "generation profile must explicitly disable model-default merging",
            code="backend_policy.model_defaults_not_disabled",
        )
    config = GenerationConfig(**config_arguments)
    observed = _authoritative_generation_profile(
        config,
        policy=policy,
        use_model_defaults=False,
    )
    if observed != dict(arguments):
        raise RuntimeContractError(
            "GenerationConfig did not preserve the authored complete profile",
            code="backend_policy.generation_profile_drift",
            context={"mismatches": _mapping_mismatches(arguments, observed)},
        )
    return config


def _authoritative_generation_profile(
    generation_config: Any,
    *,
    policy: DecodeGenerationPolicy,
    use_model_defaults: bool,
) -> dict[str, Any]:
    """Serialize and validate the complete active GenerationConfig contract."""

    if use_model_defaults is not False:
        raise RuntimeContractError(
            "generation execution must disable model-default merging",
            code="backend_policy.model_defaults_not_disabled",
        )
    payload = dict(generation_config.to_dict())
    for private_field, public_field in (
        ("_bos_token_tensor", "bos_token_id"),
        ("_eos_token_tensor", "eos_token_id"),
        ("_pad_token_tensor", "pad_token_id"),
        ("_decoder_start_token_tensor", "decoder_start_token_id"),
    ):
        private_value = payload.pop(private_field, None)
        if private_value is not None:
            observed_token_value = _canonical_special_token_value(private_value)
            expected_token_value = _canonical_special_token_value(
                payload.get(public_field)
            )
            if observed_token_value != expected_token_value:
                raise RuntimeContractError(
                    "prepared special-token tensor disagrees with GenerationConfig",
                    code="backend_policy.generation_profile_drift",
                    context={
                        "private_field": private_field,
                        "public_field": public_field,
                        "expected": expected_token_value,
                        "observed": observed_token_value,
                    },
                )
    compile_config = getattr(generation_config, "compile_config", None)
    payload["compile_config"] = (
        None
        if compile_config is None
        else compile_config.to_dict()
        if callable(getattr(compile_config, "to_dict", None))
        else repr(compile_config)
    )
    payload["use_model_defaults"] = False
    _validate_serialized_generation_profile(payload, policy=policy)
    return payload


def _validate_serialized_generation_profile(
    payload: Mapping[str, Any],
    *,
    policy: DecodeGenerationPolicy,
) -> None:
    observed_fields = set(payload) - {"use_model_defaults"}
    missing_fields = _GENERATION_CONFIG_PROFILE_FIELDS - observed_fields
    unknown_fields = observed_fields - _GENERATION_CONFIG_PROFILE_FIELDS
    unknown_non_neutral = {
        field: payload[field]
        for field in unknown_fields
        if not _is_neutral_unknown_generation_value(payload[field])
    }
    if payload.get("use_model_defaults") is not False:
        unknown_non_neutral["use_model_defaults"] = payload.get("use_model_defaults")
    if missing_fields or unknown_non_neutral:
        raise RuntimeContractError(
            "active GenerationConfig violates the fail-closed field contract",
            code="backend_policy.unknown_generation_behavior",
            context={
                "missing_fields": sorted(missing_fields),
                "unknown_non_neutral_fields": unknown_non_neutral,
            },
        )
    neutral_mismatches = {
        field: {"expected": expected, "observed": payload.get(field)}
        for field, expected in _NEUTRAL_GENERATION_CONFIG_VALUES.items()
        if payload.get(field) != expected
    }
    controlled_expected = {
        "max_new_tokens": policy.max_new_tokens,
        "do_sample": policy.mode == "sampled",
        "temperature": float(policy.temperature),
        "top_p": float(policy.top_p),
        "repetition_penalty": float(policy.repetition_penalty),
    }
    controlled_mismatches = {
        field: {"expected": expected, "observed": payload.get(field)}
        for field, expected in controlled_expected.items()
        if payload.get(field) != expected
    }
    max_length = payload.get("max_length")
    if (
        not isinstance(max_length, int)
        or isinstance(max_length, bool)
        or max_length <= 0
    ):
        controlled_mismatches["max_length"] = {
            "expected": "positive integer",
            "observed": max_length,
        }
    for field in ("eos_token_id", "pad_token_id"):
        value = payload.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            controlled_mismatches[field] = {
                "expected": "non-negative integer",
                "observed": value,
            }
    for field in ("bos_token_id", "decoder_start_token_id"):
        value = payload.get(field)
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or value < 0
        ):
            controlled_mismatches[field] = {
                "expected": "null or non-negative integer",
                "observed": value,
            }
    expected_transformers_version = _package_version("transformers")
    if payload.get("transformers_version") != expected_transformers_version:
        controlled_mismatches["transformers_version"] = {
            "expected": expected_transformers_version,
            "observed": payload.get("transformers_version"),
        }
    if neutral_mismatches or controlled_mismatches:
        raise RuntimeContractError(
            "active GenerationConfig contains undeclared token behavior",
            code="backend_policy.generation_profile_drift",
            context={
                "neutral_mismatches": neutral_mismatches,
                "controlled_mismatches": controlled_mismatches,
            },
        )


def _is_neutral_unknown_generation_value(value: Any) -> bool:
    if value is None or value is False:
        return True
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value == 0
    if isinstance(value, (list, tuple, dict, set)):
        return not value
    return False


def _canonical_special_token_value(value: Any) -> int | list[int] | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        canonical = [int(item) for item in value]
        return canonical[0] if len(canonical) == 1 else canonical
    return int(value)


def _mapping_mismatches(
    expected: Mapping[str, Any], observed: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        field: {"expected": expected.get(field), "observed": observed.get(field)}
        for field in sorted(set(expected) | set(observed))
        if expected.get(field) != observed.get(field)
    }


def validate_request_generators(
    requests: Sequence[DecodeRequest],
    request_generators: Sequence[torch.Generator],
    *,
    device: torch.device,
) -> None:
    if len(requests) != len(request_generators):
        raise RuntimeContractError(
            "request generator count must equal decode batch size",
            code="backend_sampling.generator_count_mismatch",
            context={
                "request_count": len(requests),
                "generator_count": len(request_generators),
            },
        )
    for execution_index, (request, generator) in enumerate(
        zip(requests, request_generators, strict=True)
    ):
        expected_seed = request.sampling_seed
        actual_seed = int(generator.initial_seed())
        if expected_seed != actual_seed:
            raise RuntimeContractError(
                "request generator seed does not match its execution index",
                code="backend_sampling.generator_seed_mismatch",
                context={
                    "execution_index": execution_index,
                    "request_id": request.request_id,
                    "expected_seed": expected_seed,
                    "actual_seed": actual_seed,
                },
            )
        generator_device = torch.device(generator.device)
        if generator_device.type != device.type or (
            generator_device.type == "cuda"
            and generator_device.index is not None
            and device.index is not None
            and generator_device.index != device.index
        ):
            raise RuntimeContractError(
                "request generator device does not match generation device",
                code="backend_sampling.generator_device_mismatch",
                context={
                    "execution_index": execution_index,
                    "request_id": request.request_id,
                    "generator_device": str(generator_device),
                    "generation_device": str(device),
                },
            )


def request_scoped_categorical_draw(
    scores: torch.Tensor,
    *,
    request_generators: Sequence[torch.Generator],
    sampling_timing: _SamplingTimingRecorder | None = None,
) -> torch.Tensor:
    """Draw one categorical token per row from its request-owned generator."""

    if scores.ndim != 2 or scores.shape[0] != len(request_generators):
        raise RuntimeContractError(
            "categorical score rows must match request generators",
            code="backend_sampling.score_generator_shape_mismatch",
            context={
                "score_shape": tuple(scores.shape),
                "generator_count": len(request_generators),
            },
        )
    probabilities = F.softmax(scores, dim=-1)
    sampled_rows = []
    for row_index, generator in enumerate(request_generators):
        started_at = time.perf_counter()
        sampled_rows.append(
            torch.multinomial(
                probabilities[row_index],
                num_samples=1,
                generator=generator,
            ).squeeze(0)
        )
        if sampling_timing is not None:
            sampling_timing.categorical_draw_call_count += 1
            sampling_timing.categorical_draw_cpu_dispatch_seconds += (
                time.perf_counter() - started_at
            )
    return torch.stack(sampled_rows)


def custom_generate(
    model: Any,
    *,
    request_generators: Sequence[torch.Generator],
    generation_policy: DecodeGenerationPolicy,
    sampling_timing: _SamplingTimingRecorder | None = None,
    execution_capture: _GenerationExecutionCapture | None = None,
    **generate_kwargs: Any,
) -> Any:
    """Run HF's supported custom callable with request-owned categorical draws."""

    capture = execution_capture or _GenerationExecutionCapture()
    timing = sampling_timing or _SamplingTimingRecorder()
    draw_count_before = timing.categorical_draw_call_count
    generate_parameters = inspect.signature(model.generate).parameters
    if "custom_generate" not in generate_parameters:
        # Narrow fixture seam only. It is accepted only when the fixture proves
        # it consumed the supplied generators through the shared draw helper.
        result = model.generate(
            request_generators=tuple(request_generators),
            sampling_timing=timing,
            use_model_defaults=False,
            **generate_kwargs,
        )
        if timing.categorical_draw_call_count <= draw_count_before:
            raise RuntimeContractError(
                "fixture generation did not consume request-owned generators",
                code="backend_sampling.custom_callable_not_executed",
            )
        capture.prepared_generation_profile = _authoritative_generation_profile(
            generate_kwargs["generation_config"],
            policy=generation_policy,
            use_model_defaults=False,
        )
        capture.custom_sampler_executed = True
        return result

    def request_scoped_decoding_method(
        owner: Any,
        input_ids: torch.Tensor,
        logits_processor: Any,
        stopping_criteria: Any,
        generation_config: Any,
        synced_gpus: bool = False,
        streamer: Any = None,
        **model_kwargs: Any,
    ) -> Any:
        capture.prepared_generation_profile = _authoritative_generation_profile(
            generation_config,
            policy=generation_policy,
            use_model_defaults=False,
        )
        capture.custom_sampler_executed = True
        return _request_scoped_sample(
            owner,
            input_ids,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            generation_config=generation_config,
            synced_gpus=synced_gpus,
            streamer=streamer,
            request_generators=request_generators,
            sampling_timing=timing,
            execution_capture=capture,
            **model_kwargs,
        )

    result = model.generate(
        **generate_kwargs,
        use_model_defaults=False,
        custom_generate=request_scoped_decoding_method,
    )
    if (
        not capture.custom_sampler_executed
        or timing.categorical_draw_call_count <= draw_count_before
    ):
        raise RuntimeContractError(
            "HF custom generation callable did not execute request-owned draws",
            code="backend_sampling.custom_callable_not_executed",
        )
    return result


def _request_scoped_sample(
    model: Any,
    input_ids: torch.Tensor,
    logits_processor: Any,
    stopping_criteria: Any,
    generation_config: Any,
    synced_gpus: bool = False,
    streamer: Any = None,
    *,
    request_generators: Sequence[torch.Generator],
    sampling_timing: _SamplingTimingRecorder | None = None,
    execution_capture: _GenerationExecutionCapture | None = None,
    **model_kwargs: Any,
) -> Any:
    """Transformers 4.57.1 `_sample` with only the categorical draw replaced."""

    try:
        from transformers.generation.utils import (
            GenerateDecoderOnlyOutput,
            GenerateEncoderDecoderOutput,
        )
    except ImportError as exc:
        raise RuntimeContractError(
            "installed Transformers generation outputs are unavailable",
            code="backend_sampling.transformers_unavailable",
            cause=exc,
        ) from exc
    pad_token_id = generation_config._pad_token_tensor
    output_attentions = generation_config.output_attentions
    output_hidden_states = generation_config.output_hidden_states
    output_scores = generation_config.output_scores
    output_logits = generation_config.output_logits
    return_dict_in_generate = generation_config.return_dict_in_generate
    has_eos_stopping_criteria = any(
        hasattr(criteria, "eos_token_id") for criteria in stopping_criteria
    )

    scores = () if (return_dict_in_generate and output_scores) else None
    raw_logits = () if (return_dict_in_generate and output_logits) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = (
        () if (return_dict_in_generate and output_hidden_states) else None
    )
    if return_dict_in_generate and model.config.is_encoder_decoder:
        encoder_attentions = (
            model_kwargs["encoder_outputs"].get("attentions")
            if output_attentions
            else None
        )
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states")
            if output_hidden_states
            else None
        )

    batch_size, cur_len = input_ids.shape[:2]
    if batch_size != len(request_generators):
        raise RuntimeContractError(
            "custom sampler generator count must equal the prepared HF batch size",
            code="backend_sampling.generator_count_mismatch",
            context={
                "batch_size": batch_size,
                "generator_count": len(request_generators),
            },
        )
    this_peer_finished = False
    unfinished_sequences = torch.ones(
        batch_size, dtype=torch.long, device=input_ids.device
    )
    model_kwargs = model._get_initial_cache_position(
        cur_len, input_ids.device, model_kwargs
    )
    model_forward = model.__call__
    compile_forward = model._valid_auto_compile_criteria(
        model_kwargs, generation_config
    )
    if compile_forward:
        os.environ["TOKENIZERS_PARALLELISM"] = "0"
        if (
            getattr(model.config, "_attn_implementation", None) == "flash_attention_2"
            and generation_config.compile_config is not None
            and generation_config.compile_config.fullgraph
        ):
            generation_config.compile_config.fullgraph = False
        model_forward = model.get_compiled_call(generation_config.compile_config)

    if generation_config.prefill_chunk_size is not None:
        model_kwargs = model._prefill_chunking(
            input_ids, generation_config, **model_kwargs
        )
        is_prefill = False
    else:
        is_prefill = True

    while model._has_unfinished_sequences(
        this_peer_finished, synced_gpus, device=input_ids.device
    ):
        model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)
        if is_prefill:
            outputs = model(**model_inputs, return_dict=True)
            is_prefill = False
        else:
            outputs = model_forward(**model_inputs, return_dict=True)
        model_kwargs = model._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=model.config.is_encoder_decoder,
        )
        if execution_capture is not None:
            execution_capture.record_cache_step(
                outputs=outputs,
                model_kwargs=model_kwargs,
            )
        if synced_gpus and this_peer_finished:
            continue
        next_token_logits = outputs.logits[:, -1, :].to(
            copy=True,
            dtype=torch.float32,
            device=input_ids.device,
        )
        next_token_scores = logits_processor(input_ids, next_token_logits)
        if execution_capture is not None:
            execution_capture.record_processed_scores(next_token_scores)
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_logits:
                raw_logits += (next_token_logits,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,)
                    if model.config.is_encoder_decoder
                    else (outputs.attentions,)
                )
                if model.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)
            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if model.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        next_tokens = request_scoped_categorical_draw(
            next_token_scores,
            request_generators=request_generators,
            sampling_timing=sampling_timing,
        )
        if has_eos_stopping_criteria:
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (
                1 - unfinished_sequences
            )
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())
        unfinished_sequences = unfinished_sequences & ~stopping_criteria(
            input_ids, scores
        )
        this_peer_finished = unfinished_sequences.max() == 0
        cur_len += 1
        del outputs

    if streamer is not None:
        streamer.end()
    if not return_dict_in_generate:
        return input_ids
    if model.config.is_encoder_decoder:
        return GenerateEncoderDecoderOutput(
            sequences=input_ids,
            scores=scores,
            logits=raw_logits,
            encoder_attentions=encoder_attentions,
            encoder_hidden_states=encoder_hidden_states,
            decoder_attentions=decoder_attentions,
            cross_attentions=cross_attentions,
            decoder_hidden_states=decoder_hidden_states,
            past_key_values=model_kwargs.get("past_key_values"),
        )
    return GenerateDecoderOnlyOutput(
        sequences=input_ids,
        scores=scores,
        logits=raw_logits,
        attentions=decoder_attentions,
        hidden_states=decoder_hidden_states,
        past_key_values=model_kwargs.get("past_key_values"),
    )


def batch_request_order_fingerprint(requests: Sequence[DecodeRequest]) -> str:
    policy = validate_decode_batch(requests)
    return _batch_request_order_fingerprint(requests, policy=policy)


def validate_decode_execution_batch(
    requests: Sequence[DecodeRequest],
    results: Sequence[DecodeResult],
    *,
    model_identity: Mapping[str, Any],
    tokenizer_identity: Mapping[str, Any],
    generation_config_fingerprint: str,
    executed_generation_arguments: Mapping[str, Any],
    backend: str,
    backend_mode: str,
    response_family: str,
    attention_implementation: str,
    model_eval_mode: bool,
    runtime_identity: Mapping[str, Any] | None = None,
    request_generators: Sequence[torch.Generator] | None = None,
) -> None:
    """Validate receipts against the independent active batch execution contract."""

    policy = validate_decode_batch(requests)
    if len(results) != len(requests):
        raise RuntimeContractError(
            "decode results do not match the ordered request batch",
            code="backend_receipt.batch_contract_mismatch",
            context={
                "request_count": len(requests),
                "result_count": len(results),
            },
        )
    generators = tuple(request_generators or ())
    if policy.mode == "sampled" and len(generators) != len(requests):
        raise RuntimeContractError(
            "sampled batch validation requires the actual ordered generators",
            code="backend_receipt.batch_contract_mismatch",
            context={
                "request_count": len(requests),
                "generator_count": len(generators),
            },
        )
    if policy.mode == "greedy" and generators:
        raise RuntimeContractError(
            "greedy batch validation forbids generator evidence",
            code="backend_receipt.batch_contract_mismatch",
            context={"generator_count": len(generators)},
        )
    runtime_payload = dict(
        _runtime_identity() if runtime_identity is None else runtime_identity
    )
    ordered_fingerprint = _batch_request_order_fingerprint(requests, policy=policy)
    expected_profile = _thaw_json(executed_generation_arguments)
    expected_profile_fingerprint = _sha256_json(expected_profile)
    expected_custom_identity = (
        CUSTOM_SAMPLER_IDENTITY if policy.mode == "sampled" else None
    )
    expected_custom_code_hash = (
        custom_sampler_code_hash() if policy.mode == "sampled" else None
    )
    expected_model_fingerprint = _sha256_json(model_identity)
    expected_tokenizer_fingerprint = _sha256_json(tokenizer_identity)
    expected_runtime_fingerprint = _sha256_json(runtime_payload)

    for execution_index, (request, result) in enumerate(
        zip(requests, results, strict=True)
    ):
        result.validate_for_scored()
        receipt = result.execution_receipt
        assert receipt is not None
        generator = None if not generators else generators[execution_index]
        expected_generator_seed = (
            None if generator is None else int(generator.initial_seed())
        )
        expected_generator_device = None if generator is None else str(generator.device)
        expected = {
            "request_id": request.request_id,
            "decode_generation_policy": policy.to_artifact_dict(),
            "decode_generation_policy_fingerprint": policy.fingerprint,
            "generation_config_fingerprint": generation_config_fingerprint,
            "sampling_seed": request.sampling_seed,
            "random_generator_kind": (None if generator is None else "torch.Generator"),
            "random_generator_device": expected_generator_device,
            "random_generator_initial_seed": expected_generator_seed,
            "request_execution_index": execution_index,
            "batch_request_order_fingerprint": ordered_fingerprint,
            "executed_generation_arguments": expected_profile,
            "effective_generation_profile_fingerprint": expected_profile_fingerprint,
            "backend": backend,
            "backend_mode": backend_mode,
            "response_family": response_family,
            "sampling_profile": policy.sampling_profile,
            "sampling_profile_fingerprint": _sampling_profile_fingerprint(
                policy.sampling_profile
            ),
            "custom_sampler_identity": expected_custom_identity,
            "custom_sampler_code_hash": expected_custom_code_hash,
            "attention_implementation": str(attention_implementation),
            "runtime_identity": runtime_payload,
            "installed_runtime_identity_fingerprint": expected_runtime_fingerprint,
            "model_eval_mode": bool(model_eval_mode),
            "model_identity_fingerprint": expected_model_fingerprint,
            "tokenizer_identity_fingerprint": expected_tokenizer_fingerprint,
        }
        mismatches = {
            field: {
                "expected": expected_value,
                "observed": _thaw_json(getattr(receipt, field)),
            }
            for field, expected_value in expected.items()
            if _thaw_json(getattr(receipt, field)) != expected_value
        }
        result_mismatches = {}
        if result.request_id != request.request_id:
            result_mismatches["request_id"] = {
                "expected": request.request_id,
                "observed": result.request_id,
            }
        if list(result.prompt_token_ids) != list(request.prompt_token_ids):
            result_mismatches["prompt_token_ids"] = {
                "expected": list(request.prompt_token_ids),
                "observed": list(result.prompt_token_ids),
            }
        if mismatches or result_mismatches:
            raise RuntimeContractError(
                "decode receipt disagrees with the active batch execution contract",
                code="backend_receipt.batch_contract_mismatch",
                context={
                    "request_id": request.request_id,
                    "execution_index": execution_index,
                    "receipt_mismatches": mismatches,
                    "result_mismatches": result_mismatches,
                },
            )


def build_decode_execution_receipt(
    *,
    request: DecodeRequest,
    generated_token_ids: Sequence[int],
    token_trace: Sequence[TokenTrace],
    stop_reason: str,
    model_identity: Mapping[str, Any],
    tokenizer_identity: Mapping[str, Any],
    generation_config_fingerprint: str,
    executed_generation_arguments: Mapping[str, Any],
    backend: str = "hf",
    backend_mode: str = "generate",
    response_family: str = "hf",
    request_execution_index: int = 0,
    batch_request_order_fingerprint: str | None = None,
    request_generator: torch.Generator | None = None,
    attention_implementation: str = "unknown",
    model_eval_mode: bool = True,
    runtime_identity: Mapping[str, Any] | None = None,
    custom_sampler_executed: bool = False,
) -> DecodeExecutionReceipt:
    """Build canonical result-bound execution evidence for any backend adapter."""

    policy = request.generation_policy
    _validate_serialized_generation_profile(
        executed_generation_arguments,
        policy=policy,
    )
    if policy.mode == "sampled" and request_generator is None:
        raise RuntimeContractError(
            "sampled execution receipt requires the actual request generator",
            code="backend_receipt.missing_generator",
            context={"request_id": request.request_id},
        )
    if policy.mode == "sampled" and not custom_sampler_executed:
        raise RuntimeContractError(
            "sampled receipt requires evidence from the executed custom callable",
            code="backend_receipt.custom_sampler_not_executed",
            context={"request_id": request.request_id},
        )
    if policy.mode == "greedy" and request_generator is not None:
        raise RuntimeContractError(
            "greedy execution receipt forbids random-generator evidence",
            code="backend_receipt.unexpected_generator",
            context={"request_id": request.request_id},
        )
    if policy.mode == "greedy" and custom_sampler_executed:
        raise RuntimeContractError(
            "greedy receipt forbids custom-sampler execution evidence",
            code="backend_receipt.unexpected_custom_sampler",
            context={"request_id": request.request_id},
        )
    runtime_payload = dict(
        _runtime_identity() if runtime_identity is None else runtime_identity
    )
    custom_sampler_identity = (
        CUSTOM_SAMPLER_IDENTITY if policy.mode == "sampled" else None
    )
    custom_code_hash = custom_sampler_code_hash() if policy.mode == "sampled" else None
    generator_kind = None if request_generator is None else "torch.Generator"
    generator_device = (
        None if request_generator is None else str(request_generator.device)
    )
    generator_seed = (
        None if request_generator is None else int(request_generator.initial_seed())
    )
    if request.sampling_seed != generator_seed:
        raise RuntimeContractError(
            "receipt generator seed does not match the request-owned seed",
            code="backend_sampling.generator_seed_mismatch",
            context={
                "request_id": request.request_id,
                "execution_index": request_execution_index,
                "expected_seed": request.sampling_seed,
                "actual_seed": generator_seed,
            },
        )
    order_fingerprint = batch_request_order_fingerprint or _sha256_json(
        [
            {
                "request_id": request.request_id,
                "prompt_token_identifiers_hash": _token_identifiers_hash(
                    request.prompt_token_ids
                ),
                "decode_generation_policy_fingerprint": policy.fingerprint,
            }
        ]
    )
    payload: dict[str, Any] = {
        "schema_version": DECODE_RECEIPT_SCHEMA_VERSION,
        "request_id": request.request_id,
        "decode_generation_policy": policy.to_artifact_dict(),
        "decode_generation_policy_fingerprint": policy.fingerprint,
        "generation_config_fingerprint": generation_config_fingerprint,
        "sampling_seed": request.sampling_seed,
        "random_generator_kind": generator_kind,
        "random_generator_device": generator_device,
        "random_generator_initial_seed": generator_seed,
        "request_execution_index": request_execution_index,
        "batch_request_order_fingerprint": order_fingerprint,
        "executed_generation_arguments": dict(executed_generation_arguments),
        "effective_generation_profile_fingerprint": _sha256_json(
            executed_generation_arguments
        ),
        "prompt_token_count": len(request.prompt_token_ids),
        "prompt_token_identifiers_hash": _token_identifiers_hash(
            request.prompt_token_ids
        ),
        "generated_token_count": len(generated_token_ids),
        "score_trace_count": len(token_trace),
        "generated_token_identifiers_hash": _token_identifiers_hash(
            generated_token_ids
        ),
        "canonical_float32_score_trace_hash": _canonical_float32_score_trace_hash(
            token_trace
        ),
        "backend": backend,
        "backend_mode": backend_mode,
        "response_family": response_family,
        "sampling_profile": policy.sampling_profile,
        "sampling_profile_fingerprint": _sampling_profile_fingerprint(
            policy.sampling_profile
        ),
        "custom_sampler_identity": custom_sampler_identity,
        "custom_sampler_code_hash": custom_code_hash,
        "attention_implementation": str(attention_implementation),
        "runtime_identity": runtime_payload,
        "installed_runtime_identity_fingerprint": _sha256_json(runtime_payload),
        "model_eval_mode": bool(model_eval_mode),
        "model_identity_fingerprint": _sha256_json(model_identity),
        "tokenizer_identity_fingerprint": _sha256_json(tokenizer_identity),
        "stop_reason": stop_reason,
    }
    return DecodeExecutionReceipt(
        **payload,
        receipt_fingerprint=_sha256_json(payload),
    )


def _batch_request_order_fingerprint(
    requests: Sequence[DecodeRequest],
    *,
    policy: DecodeGenerationPolicy,
) -> str:
    return _sha256_json(
        [
            {
                "request_id": request.request_id,
                "prompt_token_identifiers_hash": _token_identifiers_hash(
                    request.prompt_token_ids
                ),
                "decode_generation_policy_fingerprint": policy.fingerprint,
            }
            for request in requests
        ]
    )


def custom_sampler_code_hash() -> str:
    sources = [
        inspect.getsource(custom_generate),
        inspect.getsource(_request_scoped_sample),
        inspect.getsource(request_scoped_categorical_draw),
    ]
    return hashlib.sha256("\n".join(sources).encode("utf-8")).hexdigest()


def _runtime_identity() -> dict[str, Any]:
    return {
        "transformers_version": _package_version("transformers"),
        "torch_version": str(torch.__version__),
        "flash_attention_version": _package_version("flash-attn"),
        "cuda_runtime_version": None
        if torch.version.cuda is None
        else str(torch.version.cuda),
    }


def _named_runtime_tensors(model: Any) -> list[tuple[str, torch.Tensor, str]]:
    """Return the live parameter/buffer inventory without reading tensor payloads."""

    tensors: list[tuple[str, torch.Tensor, str]] = []
    named_parameters = getattr(model, "named_parameters", None)
    if callable(named_parameters):
        try:
            tensors.extend(
                (str(name), value, "parameter")
                for name, value in named_parameters()
                if isinstance(value, torch.Tensor)
            )
        except TypeError:
            tensors.extend(
                (str(name), value, "parameter")
                for name, value in named_parameters(recurse=True)
                if isinstance(value, torch.Tensor)
            )
    else:
        parameters = getattr(model, "parameters", None)
        if callable(parameters):
            tensors.extend(
                (f"parameter:{index}", value, "parameter")
                for index, value in enumerate(parameters())
                if isinstance(value, torch.Tensor)
            )
    named_buffers = getattr(model, "named_buffers", None)
    if callable(named_buffers):
        try:
            tensors.extend(
                (str(name), value, "buffer")
                for name, value in named_buffers()
                if isinstance(value, torch.Tensor)
            )
        except TypeError:
            tensors.extend(
                (str(name), value, "buffer")
                for name, value in named_buffers(recurse=True)
                if isinstance(value, torch.Tensor)
            )
    return sorted(tensors, key=lambda item: (item[2], item[0]))


def _model_tensor_inventory(model: Any) -> dict[str, Any]:
    """Build a bounded live-state seal using runtime-owned tensor versions.

    Reading tensor metadata and PyTorch version counters is O(number of tensors),
    not O(number of parameter values).  Ordinary in-place tensor mutation bumps
    the version counter, while dtype/storage replacement changes the remaining
    inventory fields.
    """

    rows = []
    for name, value, kind in _named_runtime_tensors(model):
        try:
            version: int | str = int(value._version)
        except RuntimeError:
            version = "unavailable_for_inference_tensor"
        rows.append(
            {
                "kind": kind,
                "name": name,
                "object_id": id(value),
                "data_pointer": int(value.data_ptr()),
                "dtype": str(value.dtype),
                "device": str(value.device),
                "shape": [int(item) for item in value.shape],
                "stride": [int(item) for item in value.stride()],
                "storage_offset": int(value.storage_offset()),
                "requires_grad": bool(getattr(value, "requires_grad", False)),
                "version": version,
            }
        )
    return {
        "tensor_count": len(rows),
        "inventory_fingerprint": _sha256_json(rows),
    }


def _model_tensor_structure_identity(model: Any) -> dict[str, Any]:
    """Return cross-process tensor structure without pointers or version counters."""

    rows = [
        {
            "kind": kind,
            "name": name,
            "dtype": str(value.dtype),
            "device": str(value.device),
            "shape": [int(item) for item in value.shape],
            "stride": [int(item) for item in value.stride()],
            "storage_offset": int(value.storage_offset()),
            "requires_grad": bool(getattr(value, "requires_grad", False)),
        }
        for name, value, kind in _named_runtime_tensors(model)
    ]
    return {
        "tensor_count": len(rows),
        "structure_fingerprint": _sha256_json(rows),
    }


def _model_live_metadata_state(model: Any) -> dict[str, Any]:
    config = getattr(model, "config", None)
    to_dict = getattr(config, "to_dict", None)
    config_payload = to_dict() if callable(to_dict) else vars(config or {})
    canonical_config = _json_state_value(config_payload)
    return {
        "model_type": f"{type(model).__module__}.{type(model).__qualname__}",
        "config_type": f"{type(config).__module__}.{type(config).__qualname__}",
        "config_fingerprint": _sha256_json(canonical_config),
        "training": bool(getattr(model, "training", False)),
    }


def _adapter_live_state(model: Any) -> dict[str, Any]:
    active_adapter = getattr(model, "active_adapter", None)
    active_adapters = getattr(model, "active_adapters", None)
    if callable(active_adapters):
        active_adapters = active_adapters()
    peft_config = getattr(model, "peft_config", None)
    configs: dict[str, Any] = {}
    if isinstance(peft_config, Mapping):
        for name, config in sorted(peft_config.items(), key=lambda item: str(item[0])):
            to_dict = getattr(config, "to_dict", None)
            payload = to_dict() if callable(to_dict) else vars(config)
            configs[str(name)] = _json_state_value(payload)
    module_states = []
    named_modules = getattr(model, "named_modules", None)
    if callable(named_modules):
        for module_name, module in named_modules():
            state: dict[str, Any] = {}
            for field in (
                "active_adapter",
                "_active_adapter",
                "disable_adapters",
                "_disable_adapters",
                "merged",
                "merged_adapters",
            ):
                if hasattr(module, field):
                    value = getattr(module, field)
                    if not callable(value):
                        state[field] = _json_state_value(value)
            if state:
                module_states.append({"module": str(module_name), "state": state})
    return {
        "active_adapter": _json_state_value(active_adapter),
        "active_adapters": _json_state_value(active_adapters),
        "peft_config": configs,
        "module_states": module_states,
    }


def _json_state_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _json_state_value(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (set, frozenset)):
        return [_json_state_value(item) for item in sorted(value, key=str)]
    if isinstance(value, (list, tuple)):
        return [_json_state_value(item) for item in value]
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}


def _tokenizer_live_state(tokenizer: Any) -> dict[str, Any]:
    get_vocab = getattr(tokenizer, "get_vocab", None)
    vocab = get_vocab() if callable(get_vocab) else None
    get_added_vocab = getattr(tokenizer, "get_added_vocab", None)
    added_vocab = get_added_vocab() if callable(get_added_vocab) else None
    fast_backend = getattr(tokenizer, "backend_tokenizer", None)
    if fast_backend is None:
        fast_backend = getattr(tokenizer, "_tokenizer", None)
    fast_backend_state = None
    if fast_backend is not None:
        to_str = getattr(fast_backend, "to_str", None)
        serialized = to_str() if callable(to_str) else None
        serialized_bytes = (
            None if serialized is None else str(serialized).encode("utf-8")
        )
        fast_backend_state = {
            "backend_type": (
                f"{type(fast_backend).__module__}.{type(fast_backend).__qualname__}"
            ),
            "serialized_byte_count": (
                None if serialized_bytes is None else len(serialized_bytes)
            ),
            "serialized_sha256": (
                None
                if serialized_bytes is None
                else hashlib.sha256(serialized_bytes).hexdigest()
            ),
        }
    return {
        "tokenizer_type": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
        "is_fast": getattr(tokenizer, "is_fast", None),
        "vocab_fingerprint": None if vocab is None else _sha256_json(vocab),
        "added_vocab_fingerprint": None
        if added_vocab is None
        else _sha256_json(added_vocab),
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "length": len(tokenizer) if hasattr(tokenizer, "__len__") else None,
        "special_tokens_map": _json_state_value(
            getattr(tokenizer, "special_tokens_map", None)
        ),
        "all_special_ids": _json_state_value(
            getattr(tokenizer, "all_special_ids", None)
        ),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "bos_token_id": getattr(tokenizer, "bos_token_id", None),
        "chat_template": getattr(tokenizer, "chat_template", None),
        "clean_up_tokenization_spaces": getattr(
            tokenizer, "clean_up_tokenization_spaces", None
        ),
        "padding_side": getattr(tokenizer, "padding_side", None),
        "truncation_side": getattr(tokenizer, "truncation_side", None),
        "model_max_length": getattr(tokenizer, "model_max_length", None),
        "split_special_tokens": getattr(tokenizer, "split_special_tokens", None),
        "add_prefix_space": getattr(tokenizer, "add_prefix_space", None),
        "model_input_names": _json_state_value(
            getattr(tokenizer, "model_input_names", None)
        ),
        "fast_backend_state": fast_backend_state,
    }


def _attested_payload_value_identity(
    model: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Hash bounded adapter and selected-token embedding payloads by value."""

    started = time.perf_counter()
    digest = hashlib.sha256()
    selected = [
        (name, value, kind)
        for name, value, kind in _named_runtime_tensors(model)
        if any(
            marker in name.lower()
            for marker in (
                "lora_",
                "dora",
                "shared_embed_delta",
                "special_token",
            )
        )
    ]
    payload_byte_count = 0
    for name, value, kind in selected:
        canonical = value.detach().to(device="cpu").contiguous()
        payload_bytes = canonical.view(torch.uint8).numpy().tobytes(order="C")
        payload_byte_count += len(payload_bytes)
        digest.update(kind.encode("utf-8"))
        digest.update(b"\0")
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(canonical.dtype).encode("utf-8"))
        digest.update(b"\0")
        digest.update(json.dumps(list(canonical.shape)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(payload_bytes)
    digest.update(str(len(selected)).encode("ascii"))
    identity = {
        "fingerprint": digest.hexdigest(),
        "tensor_count": len(selected),
        "payload_byte_count": payload_byte_count,
    }
    diagnostics = {
        "adapter_and_selected_embedding_hash_elapsed_seconds": max(
            0.0, time.perf_counter() - started
        ),
        "adapter_and_selected_embedding_tensor_count": len(selected),
        "adapter_and_selected_embedding_payload_byte_count": payload_byte_count,
        "base_model_payload_hashed": False,
    }
    return identity, diagnostics


def _live_runtime_state_seal_with_diagnostics(
    model: Any,
    tokenizer: Any,
    *,
    hash_payloads: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.perf_counter()
    seal = {
        "model_tensor_inventory": _model_tensor_inventory(model),
        "model_tensor_structure_identity": _model_tensor_structure_identity(model),
        "model_live_metadata_state": _model_live_metadata_state(model),
        "active_adapter_state": _adapter_live_state(model),
        "tokenizer_live_state": _tokenizer_live_state(tokenizer),
    }
    diagnostics: dict[str, Any] = {
        "adapter_and_selected_embedding_hash_elapsed_seconds": 0.0,
        "adapter_and_selected_embedding_tensor_count": 0,
        "adapter_and_selected_embedding_payload_byte_count": 0,
        "base_model_payload_hashed": False,
    }
    if hash_payloads:
        payload_identity, payload_diagnostics = _attested_payload_value_identity(model)
        seal["attested_adapter_and_embedding_payload_value_identity"] = (
            payload_identity
        )
        diagnostics.update(payload_diagnostics)
    diagnostics["live_runtime_state_seal_elapsed_seconds"] = max(
        0.0, time.perf_counter() - started
    )
    return seal, diagnostics


def _portable_runtime_state_seal_artifact(
    runtime_state_seal: Mapping[str, Any],
) -> dict[str, Any]:
    required = (
        "model_tensor_structure_identity",
        "model_live_metadata_state",
        "active_adapter_state",
        "tokenizer_live_state",
        "attested_adapter_and_embedding_payload_value_identity",
    )
    missing = [field for field in required if field not in runtime_state_seal]
    if missing:
        raise RuntimeContractError(
            "runtime state seal cannot be rebound across processes",
            code="backend_sampling.attestation_portable_state_incomplete",
            context={"missing_fields": missing},
        )
    return {
        "schema_version": PORTABLE_RUNTIME_STATE_SEAL_SCHEMA_VERSION,
        **{
            field: _thaw_json(runtime_state_seal[field])
            for field in required
        },
    }


def _live_runtime_state_seal(
    model: Any, tokenizer: Any, *, hash_payloads: bool
) -> dict[str, Any]:
    seal, _ = _live_runtime_state_seal_with_diagnostics(
        model,
        tokenizer,
        hash_payloads=hash_payloads,
    )
    return seal


def _cache_sequence_length(cache: Any) -> int | None:
    get_seq_length = getattr(cache, "get_seq_length", None)
    if callable(get_seq_length):
        try:
            return int(get_seq_length())
        except (TypeError, ValueError):
            return None
    try:
        first_layer = cache[0]
        key = first_layer[0]
        return int(key.shape[-2])
    except (IndexError, KeyError, TypeError, AttributeError):
        return None


def _execution_device_identity(device: torch.device) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "device_type": device.type,
        "logical_device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "cuda_runtime_version": None
        if torch.version.cuda is None
        else str(torch.version.cuda),
    }
    if device.type != "cuda" or not torch.cuda.is_available():
        payload.update(
            {
                "cuda_current_device": None,
                "cuda_device_name": None,
                "cuda_compute_capability": None,
                "cuda_total_memory_bytes": None,
            }
        )
        return payload
    index = device.index
    if index is None:
        index = int(torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(index)
    payload.update(
        {
            "cuda_current_device": int(torch.cuda.current_device()),
            "cuda_device_name": str(properties.name),
            "cuda_compute_capability": [int(properties.major), int(properties.minor)],
            "cuda_total_memory_bytes": int(properties.total_memory),
        }
    )
    return payload


def _compact_object_selected_token_score_replay(
    result: DecodeResult,
    *,
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    """Replay the canonical parser and selected-token object score in float32."""

    from src.inference.parsing import PARSER_ID, PARSER_POLICY, parse_compact_object_box_closed
    from src.inference.scoring import SCORE_POLICY_FINGERPRINT, score_prediction

    canonical_trace = [
        TokenTrace(
            step_index=trace.step_index,
            token_id=trace.token_id,
            token_text=trace.token_text,
            logprob=None
            if trace.logprob is None
            else canonical_float32_logprob(trace.logprob),
            is_stop=trace.is_stop,
            is_pad=trace.is_pad,
            backend=trace.backend,
            backend_mode=trace.backend_mode,
            response_family=trace.response_family,
        )
        for trace in result.token_trace
    ]
    parsed = parse_compact_object_box_closed(
        result.parser_text,
        row_id=result.request_id,
        row_index=0,
        image_width=image_width,
        image_height=image_height,
    )
    replays = []
    for prediction in parsed.predictions:
        scored = score_prediction(
            row_id=result.request_id,
            prediction=prediction,
            token_trace=canonical_trace,
        )
        replay = dict(scored.replay)
        selected_logprobs = [
            canonical_float32_logprob(value)
            for value in replay["selected_logprobs"]
        ]
        score = math.exp(sum(selected_logprobs) / len(selected_logprobs))
        replay["selected_logprobs"] = selected_logprobs
        replay["score"] = canonical_float32_logprob(score)
        replays.append(replay)
    return {
        "parser_id": PARSER_ID,
        "parser_policy": PARSER_POLICY,
        "parse_status": parsed.parse_status,
        "valid_prediction_count": parsed.valid_prediction_count,
        "dropped_prediction_count": parsed.dropped_prediction_count,
        "score_policy_fingerprint": SCORE_POLICY_FINGERPRINT,
        "prediction_replays": replays,
    }


def _require_scoreable_compact_object_replay(
    replay: Mapping[str, Any], *, request_id: str
) -> None:
    if not replay.get("prediction_replays"):
        raise RuntimeContractError(
            "attestation requires at least one scoreable compact object per request",
            code="backend_sampling.attestation_row_score_empty",
            context={"request_id": request_id},
        )


def _build_executed_attestation_case(
    *,
    case_name: str,
    requests: Sequence[DecodeRequest],
    results: Sequence[DecodeResult],
    prompt_width: int,
    scores: Sequence[torch.Tensor],
    sequences: torch.Tensor,
    generated_suffix: torch.Tensor,
    transition_scores: torch.Tensor,
    outputs: Any,
    execution_capture: _GenerationExecutionCapture,
    sampling_timing: _SamplingTimingRecorder,
    target_device: torch.device,
    model_identity: Mapping[str, Any],
    tokenizer_identity: Mapping[str, Any],
    generation_config_fingerprint: str,
    attention_implementation: str,
    model_evaluation_mode: bool,
    image_sizes_by_request: Mapping[str, tuple[int, int]],
) -> ExecutedSampledRuntimeAttestationCase:
    if execution_capture.prepared_generation_profile is None:
        raise RuntimeContractError(
            "attestation case requires the executed generation profile",
            code="backend_sampling.attestation_case_incomplete",
            context={"case_name": case_name},
        )
    cache_steps = tuple(execution_capture.cache_steps or ())
    processed_shapes = tuple(execution_capture.processed_score_shapes or ())
    if processed_shapes != tuple(tuple(int(v) for v in score.shape) for score in scores):
        raise RuntimeContractError(
            "captured processed-score shapes disagree with returned scores",
            code="backend_sampling.attestation_case_incomplete",
            context={"case_name": case_name},
        )
    payload: dict[str, Any] = {
        "schema_version": SAMPLED_RUNTIME_ATTESTATION_CASE_SCHEMA_VERSION,
        "case_name": case_name,
        "request_ids": tuple(request.request_id for request in requests),
        "sampling_seeds": tuple(int(request.sampling_seed) for request in requests),
        "prompt_token_identifier_hashes": tuple(
            _token_identifiers_hash(request.prompt_token_ids) for request in requests
        ),
        "model_input_fingerprints": tuple(
            _model_inputs_fingerprint(request.model_inputs) for request in requests
        ),
        "prompt_width": int(prompt_width),
        "batch_size": len(requests),
        "score_step_count": len(scores),
        "score_tensor_shapes": tuple(
            tuple(int(value) for value in score.shape) for score in scores
        ),
        "sequence_shape": tuple(int(value) for value in sequences.shape),
        "generated_suffix_shape": tuple(
            int(value) for value in generated_suffix.shape
        ),
        "transition_score_shape": tuple(
            int(value) for value in transition_scores.shape
        ),
        "output_type": type(outputs).__name__,
        "categorical_draw_call_count": sampling_timing.categorical_draw_call_count,
        "cache_steps": cache_steps,
        "output_cache_present": getattr(outputs, "past_key_values", None) is not None,
        "execution_device_identity": _execution_device_identity(target_device),
        "prepared_generation_profile": execution_capture.prepared_generation_profile,
        "model_identity": dict(model_identity),
        "tokenizer_identity": dict(tokenizer_identity),
        "generation_config_fingerprint": generation_config_fingerprint,
        "attention_implementation": attention_implementation,
        "model_evaluation_mode": bool(model_evaluation_mode),
        "runtime_identity": _runtime_identity(),
        "custom_sampler_code_hash": custom_sampler_code_hash(),
        "result_artifacts": tuple(result.to_artifact_dict() for result in results),
        "compact_object_selected_token_score_replays_by_request": {
            result.request_id: _compact_object_selected_token_score_replay(
                result,
                image_width=image_sizes_by_request[result.request_id][0],
                image_height=image_sizes_by_request[result.request_id][1],
            )
            for result in results
        },
    }
    payload["case_payload_fingerprint"] = _sha256_json(payload)
    return ExecutedSampledRuntimeAttestationCase(**payload)


def _float32_tensor_artifact(tensor: torch.Tensor) -> dict[str, Any]:
    canonical = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    raw = canonical.numpy().tobytes(order="C")
    return {
        "dtype": "float32",
        "shape": [int(value) for value in canonical.shape],
        "byte_order": "little" if struct.pack("=I", 1)[0] == 1 else "big",
        "compression": "zlib",
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "compressed_base64": base64.b64encode(zlib.compress(raw, level=9)).decode(
            "ascii"
        ),
    }


def _float32_tensor_from_artifact(payload: Mapping[str, Any]) -> torch.Tensor:
    native_byte_order = "little" if struct.pack("=I", 1)[0] == 1 else "big"
    if (
        payload.get("dtype") != "float32"
        or payload.get("compression") != "zlib"
        or payload.get("byte_order") != native_byte_order
    ):
        raise RuntimeContractError(
            "attestation tensor encoding is unsupported",
            code="backend_sampling.attestation_tensor_encoding_invalid",
        )
    try:
        compressed = base64.b64decode(str(payload["compressed_base64"]), validate=True)
        raw = zlib.decompress(compressed)
    except (KeyError, ValueError, zlib.error) as exc:
        raise RuntimeContractError(
            "attestation tensor payload cannot be decoded",
            code="backend_sampling.attestation_tensor_decode_failed",
            cause=exc,
        ) from exc
    if hashlib.sha256(raw).hexdigest() != payload.get("raw_sha256"):
        raise RuntimeContractError(
            "attestation tensor digest is invalid",
            code="backend_sampling.attestation_tensor_digest_mismatch",
        )
    shape = tuple(int(value) for value in payload.get("shape", ()))
    expected_bytes = math.prod(shape) * 4
    if len(raw) != expected_bytes:
        raise RuntimeContractError(
            "attestation tensor byte count does not match its shape",
            code="backend_sampling.attestation_tensor_shape_mismatch",
        )
    values = torch.frombuffer(bytearray(raw), dtype=torch.float32)
    return values.reshape(shape).clone()


def _build_processed_logit_parity_attestation(
    *,
    request_ids: tuple[str, ...],
    custom_scores: torch.Tensor,
    stock_scores: torch.Tensor,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> ProcessedLogitParityAttestation:
    payload: dict[str, Any] = {
        "schema_version": SAMPLED_RUNTIME_ATTESTATION_PROCESSED_LOGIT_SCHEMA_VERSION,
        "request_ids": request_ids,
        "comparison_scope": "first_generation_step_after_all_logit_processors_before_categorical_selection",
        "custom_tensor": _float32_tensor_artifact(custom_scores),
        "stock_tensor": _float32_tensor_artifact(stock_scores),
        "absolute_tolerance": float(absolute_tolerance),
        "relative_tolerance": float(relative_tolerance),
    }
    payload["parity_payload_fingerprint"] = _sha256_json(payload)
    return ProcessedLogitParityAttestation(**payload)


def _case_results_by_request(
    case: ExecutedSampledRuntimeAttestationCase,
) -> dict[str, DecodeResult]:
    results = [
        DecodeResult.from_artifact_dict(value) for value in case.result_artifacts
    ]
    return {result.request_id: result for result in results}


def _score_trace_differences(
    left: DecodeResult,
    right: DecodeResult,
) -> tuple[bool, float, float]:
    exact_tokens = left.generated_token_ids == right.generated_token_ids
    left_trace_values = [trace for trace in left.token_trace if not trace.is_pad]
    right_trace_values = [trace for trace in right.token_trace if not trace.is_pad]
    if len(left_trace_values) != len(right_trace_values):
        return False, math.inf, math.inf
    maximum_absolute = 0.0
    maximum_relative = 0.0
    for left_trace, right_trace in zip(
        left_trace_values, right_trace_values, strict=True
    ):
        if (
            left_trace.token_id != right_trace.token_id
            or left_trace.is_stop != right_trace.is_stop
            or left_trace.is_pad != right_trace.is_pad
        ):
            exact_tokens = False
        left_score = left_trace.logprob
        right_score = right_trace.logprob
        if left_score is None or right_score is None:
            if left_score != right_score:
                return False, math.inf, math.inf
            continue
        left_value = canonical_float32_logprob(left_score)
        right_value = canonical_float32_logprob(right_score)
        absolute = abs(left_value - right_value)
        relative = absolute / max(abs(left_value), abs(right_value), 1e-12)
        maximum_absolute = max(maximum_absolute, absolute)
        maximum_relative = max(maximum_relative, relative)
    return exact_tokens, maximum_absolute, maximum_relative


def _selected_token_replay_differences(
    left_case: ExecutedSampledRuntimeAttestationCase,
    right_case: ExecutedSampledRuntimeAttestationCase,
    request_id: str,
) -> tuple[bool, float, float]:
    left = _thaw_json(
        left_case.compact_object_selected_token_score_replays_by_request.get(
            request_id
        )
    )
    right = _thaw_json(
        right_case.compact_object_selected_token_score_replays_by_request.get(
            request_id
        )
    )
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        return False, math.inf, math.inf
    metadata_fields = (
        "parser_id",
        "parser_policy",
        "parse_status",
        "valid_prediction_count",
        "dropped_prediction_count",
        "score_policy_fingerprint",
    )
    exact = all(left.get(field) == right.get(field) for field in metadata_fields)
    left_replays = list(left.get("prediction_replays", ()))
    right_replays = list(right.get("prediction_replays", ()))
    if len(left_replays) != len(right_replays):
        return False, math.inf, math.inf
    maximum_absolute = 0.0
    maximum_relative = 0.0
    for left_replay, right_replay in zip(left_replays, right_replays, strict=True):
        for field in (
            "object_span_id",
            "generated_step_indices",
            "token_ids",
            "selected_count",
            "score_policy_fingerprint",
        ):
            exact = exact and left_replay.get(field) == right_replay.get(field)
        left_values = list(left_replay.get("selected_logprobs", ())) + [
            left_replay.get("score")
        ]
        right_values = list(right_replay.get("selected_logprobs", ())) + [
            right_replay.get("score")
        ]
        if len(left_values) != len(right_values) or any(
            value is None for value in left_values + right_values
        ):
            return False, math.inf, math.inf
        for left_value, right_value in zip(left_values, right_values, strict=True):
            left_float = canonical_float32_logprob(left_value)
            right_float = canonical_float32_logprob(right_value)
            absolute = abs(left_float - right_float)
            relative = absolute / max(abs(left_float), abs(right_float), 1e-12)
            maximum_absolute = max(maximum_absolute, absolute)
            maximum_relative = max(maximum_relative, relative)
    return exact, maximum_absolute, maximum_relative


def _compute_cross_cardinality_comparison(
    cases_by_name: Mapping[str, ExecutedSampledRuntimeAttestationCase],
) -> tuple[tuple[str, ...], bool, float, float]:
    four_forward = cases_by_name["batch_size_four_forward"]
    four_reversed = cases_by_name["batch_size_four_reversed"]
    three_forward = cases_by_name["batch_size_three_forward"]
    three_reversed = cases_by_name["batch_size_three_reversed"]
    shared_ids = four_forward.request_ids[:3]
    exact = True
    maximum_absolute = 0.0
    maximum_relative = 0.0
    comparisons = (
        (four_forward, four_reversed, four_forward.request_ids),
        (three_forward, three_reversed, three_forward.request_ids),
        (four_forward, three_forward, shared_ids),
        (four_reversed, three_reversed, shared_ids),
    )
    for left_case, right_case, request_ids in comparisons:
        left_results = _case_results_by_request(left_case)
        right_results = _case_results_by_request(right_case)
        for request_id in request_ids:
            if request_id not in left_results or request_id not in right_results:
                return shared_ids, False, math.inf, math.inf
            replay, absolute, relative = _score_trace_differences(
                left_results[request_id], right_results[request_id]
            )
            selected_replay, selected_absolute, selected_relative = (
                _selected_token_replay_differences(
                    left_case,
                    right_case,
                    request_id,
                )
            )
            exact = exact and replay and selected_replay
            maximum_absolute = max(
                maximum_absolute, absolute, selected_absolute
            )
            maximum_relative = max(
                maximum_relative, relative, selected_relative
            )
    return shared_ids, exact, maximum_absolute, maximum_relative


def build_sampled_runtime_attestation_bundle(
    *,
    lineage: Mapping[str, Any],
    executed_cases: Sequence[ExecutedSampledRuntimeAttestationCase],
    processed_logit_parity: ProcessedLogitParityAttestation,
) -> SampledRuntimeAttestationBundle:
    """Assemble untrusted evidence; only verification can mint admission."""

    cases = tuple(executed_cases)
    cases_by_name = {case.case_name: case for case in cases}
    if set(cases_by_name) != _REQUIRED_EXECUTED_SAMPLED_RUNTIME_ATTESTATION_CASES:
        raise RuntimeContractError(
            "attestation bundle does not contain the exact executed case set",
            code="backend_sampling.attestation_case_set_mismatch",
            context={"case_names": sorted(cases_by_name)},
        )
    shared_ids, exact, maximum_absolute, maximum_relative = (
        _compute_cross_cardinality_comparison(cases_by_name)
    )
    cross_payload: dict[str, Any] = {
        "schema_version": SAMPLED_RUNTIME_ATTESTATION_CROSS_CARDINALITY_SCHEMA_VERSION,
        "case_name": "cross_cardinality_shared_three",
        "shared_request_ids": shared_ids,
        "compared_case_names": (
            "batch_size_three_forward",
            "batch_size_three_reversed",
            "batch_size_four_forward",
            "batch_size_four_reversed",
        ),
        "maximum_absolute_score_difference": maximum_absolute,
        "maximum_relative_score_difference": maximum_relative,
        "exact_generated_token_replay": exact,
    }
    cross_payload["comparison_payload_fingerprint"] = _sha256_json(cross_payload)
    cross = CrossCardinalityRequestReplayAttestation(**cross_payload)
    payload: dict[str, Any] = {
        "schema_version": SAMPLED_RUNTIME_ATTESTATION_BUNDLE_SCHEMA_VERSION,
        "lineage": dict(lineage),
        "executed_cases": cases,
        "cross_cardinality": cross,
        "processed_logit_parity": processed_logit_parity,
    }
    canonical = {
        "schema_version": payload["schema_version"],
        "lineage": payload["lineage"],
        "executed_cases": [case.to_artifact_dict() for case in cases],
        "cross_cardinality": cross.to_artifact_dict(),
        "processed_logit_parity": processed_logit_parity.to_artifact_dict(),
    }
    payload["bundle_payload_fingerprint"] = _sha256_json(canonical)
    return SampledRuntimeAttestationBundle(**payload)


def _normalized_attested_generation_profile(
    profile: Mapping[str, Any], *, prompt_width: int
) -> dict[str, Any]:
    normalized = _thaw_json(profile)
    expected_max_length = prompt_width + int(normalized["max_new_tokens"])
    if normalized.get("max_length") != expected_max_length:
        raise RuntimeContractError(
            "prepared max_length is not prompt width plus max_new_tokens",
            code="backend_sampling.attestation_profile_length_mismatch",
            context={
                "prompt_width": prompt_width,
                "max_new_tokens": normalized.get("max_new_tokens"),
                "max_length": normalized.get("max_length"),
            },
        )
    normalized["max_length"] = "padded_prompt_width_plus_max_new_tokens"
    return normalized


def _validate_attestation_lineage(lineage: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "config_path",
        "config_sha256",
        "checkpoint_manifest_path",
        "checkpoint_manifest_sha256",
        "calibration_manifest_path",
        "calibration_manifest_sha256",
        "calibration_request_ids",
        "calibration_image_ids",
        "calibration_image_sha256",
        "calibration_prompt_token_identifier_hashes",
        "calibration_model_input_fingerprints",
        "calibration_prompt_records_fingerprint",
        "calibration_image_plan_fingerprint",
        "calibration_request_plan",
        "calibration_request_plan_fingerprint",
        "checkpoint_payload_identity",
        "model_dtype",
        "attention_implementation",
        "qwen_runtime_identity",
        "temperature",
        "root_seed",
        "frozen_sampling_factors",
    }
    missing = sorted(required - set(lineage))
    if missing:
        raise RuntimeContractError(
            "attestation lineage is incomplete",
            code="backend_sampling.attestation_lineage_incomplete",
            context={"missing_fields": missing},
        )
    for field in (
        "config_sha256",
        "checkpoint_manifest_sha256",
        "calibration_manifest_sha256",
        "calibration_prompt_records_fingerprint",
        "calibration_image_plan_fingerprint",
        "calibration_request_plan_fingerprint",
    ):
        value = lineage[field]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise RuntimeContractError(
                "attestation lineage digest is not canonical SHA-256",
                code="backend_sampling.attestation_lineage_digest_invalid",
                context={"field": field},
            )
    calibration_vectors = (
        "calibration_request_ids",
        "calibration_image_ids",
        "calibration_image_sha256",
        "calibration_prompt_token_identifier_hashes",
        "calibration_model_input_fingerprints",
    )
    if any(
        not isinstance(lineage[field], (list, tuple))
        or len(lineage[field]) != 4
        for field in calibration_vectors
    ):
        raise RuntimeContractError(
            "attestation lineage must bind exactly four calibration requests",
            code="backend_sampling.attestation_lineage_request_mismatch",
        )
    for field in (
        "calibration_image_sha256",
        "calibration_prompt_token_identifier_hashes",
        "calibration_model_input_fingerprints",
    ):
        if any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in lineage[field]
        ):
            raise RuntimeContractError(
                "attestation calibration identity is not canonical SHA-256",
                code="backend_sampling.attestation_lineage_digest_invalid",
                context={"field": field},
            )
    if (
        len(set(lineage["calibration_request_ids"])) != 4
        or len(set(lineage["calibration_image_ids"])) != 4
        or any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in lineage["calibration_image_ids"]
        )
    ):
        raise RuntimeContractError(
            "attestation calibration request or image identifiers are invalid",
            code="backend_sampling.attestation_lineage_request_mismatch",
        )
    if lineage["model_dtype"] != "bf16":
        raise RuntimeContractError(
            "attestation must execute the frozen bfloat16 model runtime",
            code="backend_sampling.attestation_model_dtype_mismatch",
        )
    if lineage["attention_implementation"] != "sdpa":
        raise RuntimeContractError(
            "attestation lineage requires the frozen SDPA implementation",
            code="backend_sampling.attestation_attention_mismatch",
        )
    if (
        lineage["temperature"] not in _ALLOWED_ATTESTATION_TEMPERATURES
        or lineage["root_seed"] != 2026071301
        or _thaw_json(lineage["frozen_sampling_factors"])
        != {
            "max_new_tokens": 512,
            "top_p": 0.95,
            "repetition_penalty": 1.0,
        }
    ):
        raise RuntimeContractError(
            "attestation lineage violates the frozen sampling protocol",
            code="backend_sampling.attestation_frozen_profile_mismatch",
        )
    config_digest = lineage["config_sha256"]
    expected_checkpoint_digest = (
        _AUTHORIZED_ATTESTATION_CONFIG_TO_CHECKPOINT_SHA256.get(config_digest)
    )
    if (
        expected_checkpoint_digest is None
        or lineage["checkpoint_manifest_sha256"] != expected_checkpoint_digest
        or lineage["calibration_manifest_sha256"]
        != _FROZEN_ATTESTATION_CALIBRATION_MANIFEST_SHA256
    ):
        raise RuntimeContractError(
            "attestation lineage is not the authorized primary checkpoint protocol",
            code="backend_sampling.attestation_primary_lineage_mismatch",
        )
    for path_field, digest_field in (
        ("config_path", "config_sha256"),
        ("checkpoint_manifest_path", "checkpoint_manifest_sha256"),
        ("calibration_manifest_path", "calibration_manifest_sha256"),
    ):
        path = Path(str(lineage[path_field])).expanduser().resolve()
        if not path.is_file() or _sha256_file(path) != lineage[digest_field]:
            raise RuntimeContractError(
                "attestation lineage file is absent or no longer matches its digest",
                code="backend_sampling.attestation_lineage_file_mismatch",
                context={"path_field": path_field, "path": str(path)},
            )
    checkpoint_payload_identity = verify_checkpoint_payload_identity(
        Path(str(lineage["checkpoint_manifest_path"]))
    )
    if _thaw_json(lineage["checkpoint_payload_identity"]) != checkpoint_payload_identity:
        raise RuntimeContractError(
            "attestation checkpoint payload identity was not verifier-derived",
            code="backend_sampling.attestation_checkpoint_payload_mismatch",
        )
    frozen_calibration = _verify_frozen_calibration_manifest(
        Path(str(lineage["calibration_manifest_path"]))
    )
    expected_calibration_core = [
        {
            "image_id": row[0],
            "request_id": row[1],
            "image_sha256": row[2],
            "image_width": row[3],
            "image_height": row[4],
            "sampling_seed": _FROZEN_ATTESTATION_CALIBRATION_SEEDS[index],
            "batch_size_four_forward_execution_index": index,
            "batch_size_four_reversed_execution_index": 3 - index,
            "batch_size_three_forward_execution_index": index
            if index < 3
            else None,
            "batch_size_three_reversed_execution_index": 2 - index
            if index < 3
            else None,
            "prompt_token_identifiers_hash": (
                _FROZEN_ATTESTATION_PROMPT_TOKEN_HASHES[index]
            ),
            "model_input_fingerprint": (
                _FROZEN_ATTESTATION_MODEL_INPUT_FINGERPRINTS[index]
            ),
        }
        for index, row in enumerate(frozen_calibration)
    ]
    request_plan = _thaw_json(lineage["calibration_request_plan"])
    if (
        not isinstance(request_plan, list)
        or len(request_plan) != 4
        or any(
            any(plan.get(field) != expected.get(field) for field in expected)
            for plan, expected in zip(request_plan, expected_calibration_core, strict=True)
        )
        or _sha256_json(request_plan)
        != lineage["calibration_request_plan_fingerprint"]
    ):
        raise RuntimeContractError(
            "attestation request plan is not the frozen first-four calibration plan",
            code="backend_sampling.attestation_request_plan_invalid",
        )
    if (
        [row["request_id"] for row in request_plan]
        != list(lineage["calibration_request_ids"])
        or [row["image_id"] for row in request_plan]
        != list(lineage["calibration_image_ids"])
        or [row["image_sha256"] for row in request_plan]
        != list(lineage["calibration_image_sha256"])
        or [row["prompt_token_identifiers_hash"] for row in request_plan]
        != list(lineage["calibration_prompt_token_identifier_hashes"])
        or [row["model_input_fingerprint"] for row in request_plan]
        != list(lineage["calibration_model_input_fingerprints"])
    ):
        raise RuntimeContractError(
            "attestation request plan vectors disagree with their canonical plan",
            code="backend_sampling.attestation_request_plan_invalid",
        )
    qwen_identity = lineage["qwen_runtime_identity"]
    if not isinstance(qwen_identity, Mapping):
        raise RuntimeContractError(
            "attestation lineage requires a structured Qwen runtime identity",
            code="backend_sampling.attestation_qwen_identity_invalid",
        )
    model_identity = qwen_identity.get("model")
    token_identity = qwen_identity.get("tokens")
    processor_identity = qwen_identity.get("processor")
    im_end_token_ids = (
        token_identity.get("im_end_token_ids")
        if isinstance(token_identity, Mapping)
        else None
    )
    if (
        qwen_identity.get("load_model") is not True
        or qwen_identity.get("attn_implementation")
        != lineage["attention_implementation"]
        or not isinstance(model_identity, Mapping)
        or model_identity.get("model_type") != "qwen3_vl"
        or not isinstance(processor_identity, Mapping)
        or not processor_identity.get("processor_class")
        or not isinstance(token_identity, Mapping)
        or token_identity.get("coord_token_count") != 1000
        or token_identity.get("coord_token_ids_contiguous") is not True
        or not isinstance(token_identity.get("wrapper_token_ids"), Mapping)
        or not isinstance(im_end_token_ids, (list, tuple))
        or len(im_end_token_ids) != 1
        or not isinstance(im_end_token_ids[0], int)
    ):
        raise RuntimeContractError(
            "attestation lineage is not an executed Qwen3-VL runtime identity",
            code="backend_sampling.attestation_qwen_identity_invalid",
        )
    return _thaw_json(qwen_identity)


def verify_checkpoint_payload_identity(checkpoint_manifest: Path) -> dict[str, Any]:
    """Rehash the adapter and selected-embedding payloads named by a checkpoint."""

    checkpoint_manifest = checkpoint_manifest.expanduser().resolve()
    try:
        manifest = json.loads(checkpoint_manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeContractError(
            "checkpoint manifest cannot be parsed for payload verification",
            code="backend_sampling.attestation_checkpoint_manifest_invalid",
            cause=exc,
        ) from exc
    run_root = checkpoint_manifest.parents[2]
    declared: dict[str, str] = {}
    adapter_identity = manifest.get("adapter", {}).get("identity", {})
    adapter_files = adapter_identity.get("file_sha256", {})
    if not isinstance(adapter_files, Mapping) or not adapter_files:
        raise RuntimeContractError(
            "checkpoint manifest lacks adapter payload hashes",
            code="backend_sampling.attestation_checkpoint_payload_incomplete",
        )
    declared.update({str(path): str(digest) for path, digest in adapter_files.items()})
    embedding_identity = manifest.get("special_token_embeddings", {}).get(
        "identity", {}
    )
    for path_field, digest_field in (
        ("metadata_path", "metadata_sha256"),
        ("tensor_path", "tensor_sha256"),
    ):
        path = embedding_identity.get(path_field)
        digest = embedding_identity.get(digest_field)
        if not isinstance(path, str) or not isinstance(digest, str):
            raise RuntimeContractError(
                "checkpoint manifest lacks selected-embedding payload hashes",
                code="backend_sampling.attestation_checkpoint_payload_incomplete",
            )
        declared[path] = digest
    actual: dict[str, str] = {}
    for relative_path, expected_digest in sorted(declared.items()):
        payload_path = (run_root / relative_path).resolve()
        try:
            payload_path.relative_to(run_root)
        except ValueError as exc:
            raise RuntimeContractError(
                "checkpoint payload escapes its run root",
                code="backend_sampling.attestation_checkpoint_payload_invalid",
                cause=exc,
            ) from exc
        if not payload_path.is_file():
            raise RuntimeContractError(
                "checkpoint payload is absent",
                code="backend_sampling.attestation_checkpoint_payload_missing",
                context={"path": str(payload_path)},
            )
        observed_digest = _sha256_file(payload_path)
        if observed_digest != expected_digest:
            raise RuntimeContractError(
                "checkpoint payload digest disagrees with its manifest",
                code="backend_sampling.attestation_checkpoint_payload_mismatch",
                context={"path": str(payload_path)},
            )
        actual[relative_path] = observed_digest
    payload = {
        "checkpoint_manifest_sha256": _sha256_file(checkpoint_manifest),
        "verified_file_sha256": actual,
        "adapter_identity_fingerprint": adapter_identity.get("fingerprint"),
        "selected_embedding_identity_fingerprint": embedding_identity.get(
            "fingerprint"
        ),
    }
    payload["payload_identity_fingerprint"] = _sha256_json(payload)
    return payload


def _verify_frozen_calibration_manifest(
    calibration_manifest: Path,
) -> tuple[tuple[int, str, str, int, int], ...]:
    calibration_manifest = calibration_manifest.expanduser().resolve()
    try:
        rows = [
            json.loads(line)
            for line in calibration_manifest.read_text(encoding="utf-8").splitlines()
            if line
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeContractError(
            "calibration manifest cannot be parsed",
            code="backend_sampling.attestation_calibration_manifest_invalid",
            cause=exc,
        ) from exc
    records = sorted(
        (row.get("record", {}) for row in rows),
        key=lambda record: int(record.get("frozen_order", -1)),
    )[:4]
    observed = tuple(
        (
            int(record.get("image_id", -1)),
            f"coco2017_val_{int(record.get('image_id', -1)):012d}",
            str(record.get("image_sha256", "")),
            int(record.get("source_width", -1)),
            int(record.get("source_height", -1)),
        )
        for record in records
    )
    if observed != _FROZEN_ATTESTATION_CALIBRATION_FIRST_FOUR:
        raise RuntimeContractError(
            "calibration manifest first four records are not frozen",
            code="backend_sampling.attestation_calibration_manifest_invalid",
        )
    for record, expected in zip(records, observed, strict=True):
        image_path = Path(str(record.get("image_path", ""))).expanduser().resolve()
        if not image_path.is_file() or _sha256_file(image_path) != expected[2]:
            raise RuntimeContractError(
                "calibration image payload is absent or changed",
                code="backend_sampling.attestation_calibration_image_mismatch",
                context={"path": str(image_path)},
            )
    return observed


def _bind_sampling_attestation_requests(
    lineage: Mapping[str, Any],
    *,
    requests: Sequence[DecodeRequest],
) -> dict[str, Any]:
    if len(requests) != 4:
        raise RuntimeContractError(
            "bounded runtime attestation requires exactly four canonical requests",
            code="backend_sampling.attestation_request_plan_invalid",
        )
    observed = {
        "calibration_request_ids": [request.request_id for request in requests],
        "calibration_prompt_token_identifier_hashes": [
            _token_identifiers_hash(request.prompt_token_ids) for request in requests
        ],
        "calibration_model_input_fingerprints": [
            _model_inputs_fingerprint(request.model_inputs) for request in requests
        ],
    }
    frozen = _verify_frozen_calibration_manifest(
        Path(str(lineage["calibration_manifest_path"]))
    )
    if any(
        request.request_id != frozen_row[1]
        for request, frozen_row in zip(requests, frozen, strict=True)
    ) or tuple(int(request.sampling_seed) for request in requests) != (
        _FROZEN_ATTESTATION_CALIBRATION_SEEDS
    ) or tuple(observed["calibration_prompt_token_identifier_hashes"]) != (
        _FROZEN_ATTESTATION_PROMPT_TOKEN_HASHES
    ) or tuple(observed["calibration_model_input_fingerprints"]) != (
        _FROZEN_ATTESTATION_MODEL_INPUT_FINGERPRINTS
    ):
        raise RuntimeContractError(
            "attestation requests are not the frozen first-four identities",
            code="backend_sampling.attestation_request_plan_invalid",
        )
    request_plan = [
        {
            "image_id": image_id,
            "request_id": request.request_id,
            "image_sha256": image_sha256,
            "image_width": image_width,
            "image_height": image_height,
            "prompt_token_identifiers_hash": observed[
                "calibration_prompt_token_identifier_hashes"
            ][index],
            "model_input_fingerprint": observed[
                "calibration_model_input_fingerprints"
            ][index],
            "sampling_seed": int(request.sampling_seed),
            "batch_size_four_forward_execution_index": index,
            "batch_size_four_reversed_execution_index": 3 - index,
            "batch_size_three_forward_execution_index": index if index < 3 else None,
            "batch_size_three_reversed_execution_index": 2 - index
            if index < 3
            else None,
        }
        for index, (request, (image_id, _, image_sha256, image_width, image_height))
        in enumerate(zip(requests, frozen, strict=True))
    ]
    observed["calibration_request_plan"] = request_plan
    observed["calibration_request_plan_fingerprint"] = _sha256_json(request_plan)
    bound = dict(lineage)
    mismatches = {
        field: {"declared": _thaw_json(bound[field]), "observed": value}
        for field, value in observed.items()
        if field in bound and _thaw_json(bound[field]) != value
    }
    if mismatches:
        raise RuntimeContractError(
            "attestation request plan disagrees with its frozen lineage",
            code="backend_sampling.attestation_request_plan_invalid",
            context={"mismatches": mismatches},
        )
    bound.update(observed)
    return bound


def _validate_sampling_attestation_execution_request(
    lineage: Mapping[str, Any],
    *,
    requests: Sequence[DecodeRequest],
    model: Any,
    tokenizer: Any,
    tokenizer_identity: Mapping[str, Any],
) -> None:
    qwen_identity = _validate_attestation_lineage(lineage)
    policy = validate_decode_batch(requests)
    if (
        policy.mode != "sampled"
        or policy.max_new_tokens != 512
        or policy.temperature != lineage["temperature"]
        or policy.temperature not in _ALLOWED_ATTESTATION_TEMPERATURES
        or policy.top_p != 0.95
        or policy.repetition_penalty != 1.0
    ):
        raise RuntimeContractError(
            "attestation request plan violates the frozen sampling policy",
            code="backend_sampling.attestation_frozen_profile_mismatch",
        )
    if (
        _attention_implementation(model) != "sdpa"
        or bool(getattr(model, "training", False))
        or _model_type(model) != "qwen3_vl"
        or _thaw_json(tokenizer_identity) != qwen_identity["tokens"]
        or _tokenizer_im_end_token_id(tokenizer)
        != qwen_identity["tokens"]["im_end_token_ids"][0]
    ):
        raise RuntimeContractError(
            "active model or tokenizer is not the frozen Qwen3-VL runtime",
            code="backend_sampling.attestation_active_runtime_mismatch",
        )


def _validate_attestation_case(
    case: ExecutedSampledRuntimeAttestationCase,
    *,
    image_sizes_by_request: Mapping[str, tuple[int, int]],
) -> tuple[DecodeGenerationPolicy, dict[str, Any]]:
    expected_batch_size = (
        4 if "batch_size_four" in case.case_name else 3
    )
    if case.batch_size != expected_batch_size:
        raise RuntimeContractError(
            "attestation case cardinality disagrees with its typed case name",
            code="backend_sampling.attestation_case_cardinality_mismatch",
            context={"case_name": case.case_name, "batch_size": case.batch_size},
        )
    if (
        len(case.request_ids) != case.batch_size
        or len(set(case.request_ids)) != case.batch_size
        or len(case.sampling_seeds) != case.batch_size
        or len(set(case.sampling_seeds)) != case.batch_size
        or len(case.prompt_token_identifier_hashes) != case.batch_size
        or len(case.model_input_fingerprints) != case.batch_size
    ):
        raise RuntimeContractError(
            "attestation case request identity vectors are invalid",
            code="backend_sampling.attestation_case_request_binding_invalid",
            context={"case_name": case.case_name},
        )
    if case.score_step_count <= 0:
        raise RuntimeContractError(
            "attestation case has no score steps",
            code="backend_sampling.attestation_score_step_invalid",
        )
    if (
        len(case.score_tensor_shapes) != case.score_step_count
        or any(len(shape) != 2 for shape in case.score_tensor_shapes)
    ):
        raise RuntimeContractError(
            "attestation score tensor count or rank is inconsistent",
            code="backend_sampling.attestation_output_shape_invalid",
            context={"case_name": case.case_name},
        )
    expected_score_shapes = tuple(
        (case.batch_size, case.score_tensor_shapes[0][1])
        for _ in range(case.score_step_count)
    )
    if (
        case.score_tensor_shapes != expected_score_shapes
        or case.sequence_shape
        != (case.batch_size, case.prompt_width + case.score_step_count)
        or case.generated_suffix_shape
        != (case.batch_size, case.score_step_count)
        or case.transition_score_shape
        != (case.batch_size, case.score_step_count)
        or case.output_type
        not in {"GenerateDecoderOnlyOutput", "GenerateEncoderDecoderOutput"}
    ):
        raise RuntimeContractError(
            "attestation output and score shapes are inconsistent",
            code="backend_sampling.attestation_output_shape_invalid",
            context={"case_name": case.case_name},
        )
    if case.categorical_draw_call_count != case.batch_size * case.score_step_count:
        raise RuntimeContractError(
            "attestation categorical draw count is not one draw per row per step",
            code="backend_sampling.attestation_draw_count_invalid",
            context={"case_name": case.case_name},
        )
    if len(case.cache_steps) != case.score_step_count or not case.output_cache_present:
        raise RuntimeContractError(
            "attestation cache evidence is incomplete",
            code="backend_sampling.attestation_cache_invalid",
            context={"case_name": case.case_name},
        )
    previous_cache_length: int | None = None
    for step_index, cache_step in enumerate(case.cache_steps):
        current_length = cache_step.get("updated_cache_sequence_length")
        if (
            cache_step.get("step_index") != step_index
            or cache_step.get("output_cache_present") is not True
            or cache_step.get("updated_cache_present") is not True
            or not cache_step.get("updated_cache_type")
            or not isinstance(current_length, int)
            or current_length <= 0
            or (
                previous_cache_length is not None
                and current_length != previous_cache_length + 1
            )
        ):
            raise RuntimeContractError(
                "attestation cache update sequence is invalid",
                code="backend_sampling.attestation_cache_invalid",
                context={"case_name": case.case_name, "step_index": step_index},
            )
        previous_cache_length = current_length
    device = case.execution_device_identity
    if (
        device.get("device_type") != "cuda"
        or device.get("cuda_available") is not True
        or device.get("cuda_device_count") != 1
        or device.get("cuda_current_device") != 0
        or device.get("logical_device") not in {"cuda", "cuda:0"}
        or not device.get("cuda_device_name")
        or not device.get("cuda_compute_capability")
        or not isinstance(device.get("cuda_total_memory_bytes"), int)
    ):
        raise RuntimeContractError(
            "only an executed single-visible-device CUDA case can attest production",
            code="backend_sampling.attestation_cuda_required",
            context={"case_name": case.case_name},
        )
    if not torch.cuda.is_available():
        raise RuntimeContractError(
            "attestation verification requires the executed CUDA runtime to be visible",
            code="backend_sampling.attestation_cuda_runtime_unavailable",
            context={"case_name": case.case_name},
        )
    live_device_identity = _execution_device_identity(torch.device("cuda:0"))
    if _thaw_json(device) != live_device_identity:
        raise RuntimeContractError(
            "attested CUDA hardware identity does not match the verifier runtime",
            code="backend_sampling.attestation_cuda_identity_mismatch",
            context={"case_name": case.case_name},
        )
    if not case.model_evaluation_mode:
        raise RuntimeContractError(
            "attestation model must execute in evaluation mode",
            code="backend_sampling.attestation_model_not_eval",
        )
    if case.attention_implementation in {"", "unknown"}:
        raise RuntimeContractError(
            "attestation attention implementation is unknown",
            code="backend_sampling.attestation_attention_unknown",
        )
    if case.custom_sampler_code_hash != custom_sampler_code_hash():
        raise RuntimeContractError(
            "attestation sampler code does not match the installed implementation",
            code="backend_sampling.attestation_sampler_code_mismatch",
        )
    if len(case.result_artifacts) != case.batch_size:
        raise RuntimeContractError(
            "attestation result count does not match its batch",
            code="backend_sampling.attestation_result_count_mismatch",
        )
    results = [
        DecodeResult.from_artifact_dict(value) for value in case.result_artifacts
    ]
    if tuple(result.request_id for result in results) != case.request_ids:
        raise RuntimeContractError(
            "attestation result order does not match request order",
            code="backend_sampling.attestation_result_order_mismatch",
        )
    policy: DecodeGenerationPolicy | None = None
    reconstructed_requests: list[DecodeRequest] = []
    for execution_index, (result, seed, prompt_hash) in enumerate(
        zip(
            results,
            case.sampling_seeds,
            case.prompt_token_identifier_hashes,
            strict=True,
        )
    ):
        result.validate_for_scored()
        receipt = result.execution_receipt
        assert receipt is not None
        result_policy = DecodeGenerationPolicy(**dict(receipt.decode_generation_policy))
        if policy is None:
            policy = result_policy
        elif result_policy != policy:
            raise RuntimeContractError(
                "attestation case receipts contain different policies",
                code="backend_sampling.attestation_policy_mismatch",
            )
        reconstructed_requests.append(
            DecodeRequest(
                request_id=result.request_id,
                prompt_token_ids=list(result.prompt_token_ids),
                model_inputs={},
                generation_policy=result_policy,
                sampling_seed=seed,
            )
        )
        image_size = image_sizes_by_request.get(result.request_id)
        if image_size is None:
            raise RuntimeContractError(
                "attestation result lacks a frozen calibration image size",
                code="backend_sampling.attestation_request_plan_invalid",
                context={"request_id": result.request_id},
            )
        expected_row_score = _compact_object_selected_token_score_replay(
            result,
            image_width=image_size[0],
            image_height=image_size[1],
        )
        observed_row_score = (
            case.compact_object_selected_token_score_replays_by_request.get(
                result.request_id
            )
        )
        _require_scoreable_compact_object_replay(
            expected_row_score,
            request_id=result.request_id,
        )
        if _thaw_json(observed_row_score) != expected_row_score:
            raise RuntimeContractError(
                "attestation compact-object selected-token score does not replay",
                code="backend_sampling.attestation_row_score_mismatch",
                context={"request_id": result.request_id},
            )
        if (
            _token_identifiers_hash(result.prompt_token_ids) != prompt_hash
            or len(result.token_trace) != case.score_step_count
            or receipt.score_trace_count != case.score_step_count
            or receipt.sampling_seed != seed
            or receipt.random_generator_initial_seed != seed
            or receipt.random_generator_kind != "torch.Generator"
            or receipt.random_generator_device not in {"cuda", "cuda:0"}
            or receipt.request_execution_index != execution_index
            or _thaw_json(receipt.executed_generation_arguments)
            != _thaw_json(case.prepared_generation_profile)
            or receipt.attention_implementation != case.attention_implementation
            or receipt.model_eval_mode is not True
            or _thaw_json(receipt.runtime_identity)
            != _thaw_json(case.runtime_identity)
            or receipt.custom_sampler_code_hash != case.custom_sampler_code_hash
            or _thaw_json(result.model_identity) != _thaw_json(case.model_identity)
            or _thaw_json(result.tokenizer_identity)
            != _thaw_json(case.tokenizer_identity)
            or result.generation_config_fingerprint
            != case.generation_config_fingerprint
        ):
            raise RuntimeContractError(
                "attestation result or receipt does not match executed case evidence",
                code="backend_sampling.attestation_result_binding_mismatch",
                context={"request_id": result.request_id},
            )
        stop_indices = [
            trace.step_index for trace in result.token_trace if trace.is_stop
        ]
        pad_id = int(case.prepared_generation_profile["pad_token_id"])
        stop_id = int(case.prepared_generation_profile["eos_token_id"])
        if len(stop_indices) > 1:
            raise RuntimeContractError(
                "attestation trace contains multiple Qwen stop tokens",
                code="backend_sampling.attestation_stop_invalid",
            )
        if result.stop_reason == "im_end":
            if len(stop_indices) != 1:
                raise RuntimeContractError(
                    "im_end stop reason lacks one Qwen stop token",
                    code="backend_sampling.attestation_stop_invalid",
                )
            stop_index = stop_indices[0]
            if result.token_trace[stop_index].token_id != stop_id or any(
                not trace.is_pad or trace.token_id != pad_id
                for trace in result.token_trace[stop_index + 1 :]
            ):
                raise RuntimeContractError(
                    "attestation trace violates sample-then-pad Qwen stop semantics",
                    code="backend_sampling.attestation_sample_then_pad_invalid",
                )
        elif result.stop_reason != "length" or stop_indices:
            raise RuntimeContractError(
                "attestation stop reason is inconsistent with its trace",
                code="backend_sampling.attestation_stop_invalid",
            )
    assert policy is not None
    validate_decode_batch(reconstructed_requests)
    expected_order_fingerprint = _batch_request_order_fingerprint(
        reconstructed_requests, policy=policy
    )
    if any(
        result.execution_receipt.batch_request_order_fingerprint
        != expected_order_fingerprint
        for result in results
    ):
        raise RuntimeContractError(
            "attestation receipt batch-order fingerprint does not replay",
            code="backend_sampling.attestation_order_fingerprint_mismatch",
        )
    _validate_serialized_generation_profile(
        case.prepared_generation_profile, policy=policy
    )
    normalized_profile = _normalized_attested_generation_profile(
        case.prepared_generation_profile, prompt_width=case.prompt_width
    )
    return policy, normalized_profile


def _validate_sampled_runtime_attestation_bundle(
    bundle: SampledRuntimeAttestationBundle | Mapping[str, Any],
) -> tuple[SampledRuntimeAttestationBundle, dict[str, Any]]:
    """Fully revalidate serialized evidence without granting admission."""

    parsed = (
        bundle
        if isinstance(bundle, SampledRuntimeAttestationBundle)
        else SampledRuntimeAttestationBundle.from_artifact_dict(bundle)
    )
    qwen_runtime_identity = _validate_attestation_lineage(parsed.lineage)
    image_sizes_by_request = {
        str(row["request_id"]): (int(row["image_width"]), int(row["image_height"]))
        for row in parsed.lineage["calibration_request_plan"]
    }
    cases_by_name = {case.case_name: case for case in parsed.executed_cases}
    if (
        set(cases_by_name) != _REQUIRED_EXECUTED_SAMPLED_RUNTIME_ATTESTATION_CASES
        or len(cases_by_name) != len(parsed.executed_cases)
    ):
        raise RuntimeContractError(
            "attestation bundle does not contain the exact unique executed case set",
            code="backend_sampling.attestation_case_set_mismatch",
        )
    policies: list[DecodeGenerationPolicy] = []
    normalized_profiles: list[dict[str, Any]] = []
    for case_name in sorted(cases_by_name):
        policy, normalized_profile = _validate_attestation_case(
            cases_by_name[case_name],
            image_sizes_by_request=image_sizes_by_request,
        )
        policies.append(policy)
        normalized_profiles.append(normalized_profile)
    replay_results = [
        DecodeResult.from_artifact_dict(result)
        for case in parsed.executed_cases
        for result in case.result_artifacts
    ]
    if not any(result.stop_reason == "im_end" for result in replay_results):
        raise RuntimeContractError(
            "attestation cases did not execute a natural Qwen im_end stop",
            code="backend_sampling.attestation_im_end_not_observed",
        )
    if not any(
        trace.is_pad for result in replay_results for trace in result.token_trace
    ):
        raise RuntimeContractError(
            "attestation cases did not exercise post-stop sample-then-pad behavior",
            code="backend_sampling.attestation_sample_then_pad_not_observed",
        )
    if any(policy != policies[0] for policy in policies[1:]) or any(
        profile != normalized_profiles[0] for profile in normalized_profiles[1:]
    ):
        raise RuntimeContractError(
            "attestation cases do not share one generation policy and profile",
            code="backend_sampling.attestation_profile_mismatch",
        )
    policy = policies[0]
    if (
        policy.mode != "sampled"
        or policy.max_new_tokens != 512
        or policy.temperature != parsed.lineage["temperature"]
        or policy.temperature not in _ALLOWED_ATTESTATION_TEMPERATURES
        or policy.top_p != 0.95
        or policy.repetition_penalty != 1.0
    ):
        raise RuntimeContractError(
            "executed policy violates the frozen request-scoped sampling protocol",
            code="backend_sampling.attestation_frozen_profile_mismatch",
        )
    qwen_token_identity = qwen_runtime_identity["tokens"]
    if (
        _thaw_json(cases_by_name["batch_size_four_forward"].tokenizer_identity)
        != qwen_token_identity
        or normalized_profiles[0].get("eos_token_id")
        != qwen_token_identity["im_end_token_ids"][0]
    ):
        raise RuntimeContractError(
            "executed tokenizer or stop token does not match Qwen runtime identity",
            code="backend_sampling.attestation_qwen_token_mismatch",
        )
    four_forward = cases_by_name["batch_size_four_forward"]
    four_reversed = cases_by_name["batch_size_four_reversed"]
    three_forward = cases_by_name["batch_size_three_forward"]
    three_reversed = cases_by_name["batch_size_three_reversed"]
    if (
        tuple(parsed.lineage["calibration_request_ids"])
        != four_forward.request_ids
        or tuple(parsed.lineage["calibration_prompt_token_identifier_hashes"])
        != four_forward.prompt_token_identifier_hashes
        or tuple(parsed.lineage["calibration_model_input_fingerprints"])
        != four_forward.model_input_fingerprints
    ):
        raise RuntimeContractError(
            "attestation lineage calibration identifiers do not bind the executed requests",
            code="backend_sampling.attestation_lineage_request_mismatch",
        )
    if (
        four_reversed.request_ids != tuple(reversed(four_forward.request_ids))
        or four_reversed.sampling_seeds
        != tuple(reversed(four_forward.sampling_seeds))
        or four_reversed.prompt_token_identifier_hashes
        != tuple(reversed(four_forward.prompt_token_identifier_hashes))
        or four_reversed.model_input_fingerprints
        != tuple(reversed(four_forward.model_input_fingerprints))
        or three_forward.request_ids != four_forward.request_ids[:3]
        or three_forward.sampling_seeds != four_forward.sampling_seeds[:3]
        or three_forward.prompt_token_identifier_hashes
        != four_forward.prompt_token_identifier_hashes[:3]
        or three_forward.model_input_fingerprints
        != four_forward.model_input_fingerprints[:3]
        or three_reversed.request_ids != tuple(reversed(three_forward.request_ids))
        or three_reversed.sampling_seeds
        != tuple(reversed(three_forward.sampling_seeds))
        or three_reversed.prompt_token_identifier_hashes
        != tuple(reversed(three_forward.prompt_token_identifier_hashes))
        or three_reversed.model_input_fingerprints
        != tuple(reversed(three_forward.model_input_fingerprints))
    ):
        raise RuntimeContractError(
            "attestation forward, reversed, and shared-cardinality identities disagree",
            code="backend_sampling.attestation_request_replay_layout_invalid",
        )
    shared_ids, exact, maximum_absolute, maximum_relative = (
        _compute_cross_cardinality_comparison(cases_by_name)
    )
    cross = parsed.cross_cardinality
    if (
        cross.case_name != "cross_cardinality_shared_three"
        or cross.shared_request_ids != shared_ids
        or set(cross.compared_case_names)
        != _REQUIRED_EXECUTED_SAMPLED_RUNTIME_ATTESTATION_CASES
        or cross.exact_generated_token_replay != exact
        or cross.maximum_absolute_score_difference != maximum_absolute
        or cross.maximum_relative_score_difference != maximum_relative
        or not exact
        or maximum_absolute > 1e-6
        or maximum_relative > 1e-6
    ):
        raise RuntimeContractError(
            "attestation request replay or float32 score parity failed",
            code="backend_sampling.attestation_request_replay_failed",
            context={
                "maximum_absolute_score_difference": maximum_absolute,
                "maximum_relative_score_difference": maximum_relative,
            },
        )
    parity = parsed.processed_logit_parity
    if (
        parity.request_ids != four_forward.request_ids
        or parity.comparison_scope
        != "first_generation_step_after_all_logit_processors_before_categorical_selection"
        or parity.absolute_tolerance != 1e-6
        or parity.relative_tolerance != 1e-6
    ):
        raise RuntimeContractError(
            "processed-logit parity does not bind the forward four-row case",
            code="backend_sampling.attestation_logit_binding_mismatch",
        )
    custom_tensor = _float32_tensor_from_artifact(parity.custom_tensor)
    stock_tensor = _float32_tensor_from_artifact(parity.stock_tensor)
    if (
        tuple(custom_tensor.shape) != four_forward.score_tensor_shapes[0]
        or stock_tensor.shape != custom_tensor.shape
        or not torch.isfinite(custom_tensor).all()
        or not torch.isfinite(stock_tensor).all()
        or not torch.allclose(
            custom_tensor,
            stock_tensor,
            atol=parity.absolute_tolerance,
            rtol=parity.relative_tolerance,
        )
    ):
        raise RuntimeContractError(
            "custom and stock processed logits differ before categorical selection",
            code="backend_sampling.attestation_processed_logit_parity_failed",
        )
    identity_fields = (
        "model_identity",
        "tokenizer_identity",
        "generation_config_fingerprint",
        "attention_implementation",
        "runtime_identity",
        "custom_sampler_code_hash",
        "execution_device_identity",
    )
    for field in identity_fields:
        expected = _thaw_json(getattr(four_forward, field))
        if any(
            _thaw_json(getattr(case, field)) != expected
            for case in parsed.executed_cases
        ):
            raise RuntimeContractError(
                "attestation execution identity changed between cases",
                code="backend_sampling.attestation_execution_identity_mismatch",
                context={"field": field},
            )
    if parsed.lineage["attention_implementation"] != four_forward.attention_implementation:
        raise RuntimeContractError(
            "attestation lineage and executed attention implementation disagree",
            code="backend_sampling.attestation_attention_mismatch",
        )
    admission_contract = {
        "bundle_payload_fingerprint": parsed.bundle_payload_fingerprint,
        "model_identity": _thaw_json(four_forward.model_identity),
        "tokenizer_identity": _thaw_json(four_forward.tokenizer_identity),
        "generation_config_fingerprint": four_forward.generation_config_fingerprint,
        "decode_generation_policy": policies[0].to_artifact_dict(),
        "normalized_prepared_generation_profile": normalized_profiles[0],
        "attention_implementation": four_forward.attention_implementation,
        "runtime_identity": _thaw_json(four_forward.runtime_identity),
        "custom_sampler_code_hash": four_forward.custom_sampler_code_hash,
        "execution_device_identity": _thaw_json(
            four_forward.execution_device_identity
        ),
        "verified_checks": sorted(_REQUIRED_SAMPLED_RUNTIME_ATTESTATION_CHECKS),
    }
    return parsed, admission_contract


def validate_sampled_runtime_attestation_bundle(
    bundle: SampledRuntimeAttestationBundle | Mapping[str, Any],
) -> dict[str, Any]:
    """Validate serialized evidence, explicitly without minting capability."""

    parsed, admission_contract = _validate_sampled_runtime_attestation_bundle(bundle)
    return {
        "bundle_payload_fingerprint": parsed.bundle_payload_fingerprint,
        "verified_checks": list(admission_contract["verified_checks"]),
        "production_admission": False,
    }


def _load_sampled_runtime_attestation_aggregate_payload(
    source: str | Path | Mapping[str, Any],
) -> dict[str, Any]:
    if isinstance(source, Mapping):
        return _thaw_json(source)
    try:
        path = Path(source).expanduser().resolve()
    except TypeError as exc:
        raise RuntimeContractError(
            "sampled runtime attestation aggregate source is unsupported",
            code="backend_sampling.attestation_aggregate_source_invalid",
            context={"source_type": type(source).__name__},
            cause=exc,
        ) from exc
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeContractError(
            "sampled runtime attestation aggregate cannot be loaded",
            code="backend_sampling.attestation_aggregate_load_failed",
            context={"path": str(path)},
            cause=exc,
        ) from exc
    if not isinstance(payload, Mapping):
        raise RuntimeContractError(
            "sampled runtime attestation aggregate root must be an object",
            code="backend_sampling.attestation_aggregate_schema_invalid",
            context={"path": str(path)},
        )
    return dict(payload)


def _validate_portable_runtime_state_seal_artifact(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    expected_fields = {
        "schema_version",
        "model_tensor_structure_identity",
        "model_live_metadata_state",
        "active_adapter_state",
        "tokenizer_live_state",
        "attested_adapter_and_embedding_payload_value_identity",
    }
    if (
        set(payload) != expected_fields
        or payload.get("schema_version")
        != PORTABLE_RUNTIME_STATE_SEAL_SCHEMA_VERSION
    ):
        raise RuntimeContractError(
            "portable runtime state seal schema is invalid",
            code="backend_sampling.attestation_portable_state_invalid",
        )
    portable = _portable_runtime_state_seal_artifact(payload)
    structure = portable["model_tensor_structure_identity"]
    metadata = portable["model_live_metadata_state"]
    adapter = portable["active_adapter_state"]
    tokenizer = portable["tokenizer_live_state"]
    fast_backend = (
        tokenizer.get("fast_backend_state")
        if isinstance(tokenizer, Mapping)
        else None
    )
    value_identity = portable[
        "attested_adapter_and_embedding_payload_value_identity"
    ]
    if (
        not isinstance(structure, Mapping)
        or set(structure) != {"tensor_count", "structure_fingerprint"}
        or not isinstance(structure.get("tensor_count"), int)
        or structure["tensor_count"] < 0
        or not _is_canonical_sha256(structure.get("structure_fingerprint"))
        or not isinstance(metadata, Mapping)
        or set(metadata)
        != {"model_type", "config_type", "config_fingerprint", "training"}
        or not isinstance(metadata.get("model_type"), str)
        or not isinstance(metadata.get("config_type"), str)
        or not _is_canonical_sha256(metadata.get("config_fingerprint"))
        or not isinstance(metadata.get("training"), bool)
        or not isinstance(adapter, Mapping)
        or set(adapter)
        != {"active_adapter", "active_adapters", "peft_config", "module_states"}
        or not isinstance(adapter.get("peft_config"), Mapping)
        or not isinstance(adapter.get("module_states"), list)
        or not isinstance(tokenizer, Mapping)
        or set(tokenizer)
        != {
            "tokenizer_type",
            "is_fast",
            "vocab_fingerprint",
            "added_vocab_fingerprint",
            "vocab_size",
            "length",
            "special_tokens_map",
            "all_special_ids",
            "pad_token_id",
            "eos_token_id",
            "bos_token_id",
            "chat_template",
            "clean_up_tokenization_spaces",
            "padding_side",
            "truncation_side",
            "model_max_length",
            "split_special_tokens",
            "add_prefix_space",
            "model_input_names",
            "fast_backend_state",
        }
        or not isinstance(tokenizer.get("tokenizer_type"), str)
        or any(
            digest is not None and not _is_canonical_sha256(digest)
            for digest in (
                tokenizer.get("vocab_fingerprint"),
                tokenizer.get("added_vocab_fingerprint"),
            )
        )
        or (
            fast_backend is not None
            and (
                not isinstance(fast_backend, Mapping)
                or set(fast_backend)
                != {"backend_type", "serialized_byte_count", "serialized_sha256"}
                or not isinstance(fast_backend.get("backend_type"), str)
                or not isinstance(fast_backend.get("serialized_byte_count"), int)
                or fast_backend["serialized_byte_count"] < 0
                or not _is_canonical_sha256(fast_backend.get("serialized_sha256"))
            )
        )
        or not isinstance(value_identity, Mapping)
        or set(value_identity) != {"fingerprint", "tensor_count", "payload_byte_count"}
        or value_identity.get("tensor_count")
        != EXPECTED_ATTESTED_RUNTIME_PAYLOAD_TENSOR_COUNT
        or value_identity.get("payload_byte_count")
        != EXPECTED_ATTESTED_RUNTIME_PAYLOAD_BYTE_COUNT
        or not _is_canonical_sha256(value_identity.get("fingerprint"))
    ):
        raise RuntimeContractError(
            "portable runtime state seal does not bind the frozen live payload",
            code="backend_sampling.attestation_portable_state_invalid",
        )
    return portable


def _validate_admitted_production_replay_artifact(
    payload: Mapping[str, Any],
    *,
    bundle: SampledRuntimeAttestationBundle,
    decode_generation_policy_fingerprint: str,
) -> dict[str, Any]:
    replay = dict(payload)
    replay_fingerprint = replay.pop("replay_payload_fingerprint", None)
    if replay_fingerprint != _sha256_json(replay):
        raise RuntimeContractError(
            "admitted production replay fingerprint is invalid",
            code="backend_sampling.attestation_admitted_replay_fingerprint_mismatch",
        )
    cases = {case.case_name: case for case in bundle.executed_cases}
    four_forward = cases.get("batch_size_four_forward")
    rows = payload.get("result_replays")
    diagnostics = payload.get("runtime_state_seal_diagnostics")
    if four_forward is None or not isinstance(rows, list) or not isinstance(
        diagnostics, Mapping
    ):
        raise RuntimeContractError(
            "admitted production replay evidence is incomplete",
            code="backend_sampling.attestation_admitted_replay_invalid",
        )
    expected_request_ids = list(four_forward.request_ids)
    expected_artifacts = {
        str(artifact["request_id"]): _thaw_json(artifact)
        for artifact in four_forward.result_artifacts
    }
    expected_selected_replays = _thaw_json(
        four_forward.compact_object_selected_token_score_replays_by_request
    )
    replay_mismatch = (
        payload.get("schema_version")
        != ADMITTED_PRODUCTION_REPLAY_SCHEMA_VERSION
        or payload.get("capability_gated_backend_api")
        != "HFGenerateBackend.generate_batch_with_verified_runtime_attestation"
        or payload.get("capability_bundle_payload_fingerprint")
        != bundle.bundle_payload_fingerprint
        or payload.get("decode_generation_policy_fingerprint")
        != decode_generation_policy_fingerprint
        or payload.get("compared_attestation_case")
        != "batch_size_four_forward"
        or payload.get("request_ids") != expected_request_ids
        or payload.get("exact_request_order_replay") is not True
        or payload.get("exact_result_artifact_replay") is not True
        or payload.get("exact_selected_token_score_replay") is not True
        or len(rows) != len(expected_request_ids)
        or [row.get("request_id") for row in rows] != expected_request_ids
    )
    if replay_mismatch:
        raise RuntimeContractError(
            "admitted production replay does not bind its policy and bundle",
            code="backend_sampling.attestation_admitted_replay_invalid",
        )
    for execution_index, row in enumerate(rows):
        request_id = expected_request_ids[execution_index]
        expected_artifact = expected_artifacts[request_id]
        result = DecodeResult.from_artifact_dict(expected_artifact)
        receipt = result.execution_receipt
        assert receipt is not None
        expected_artifact_fingerprint = _sha256_json(expected_artifact)
        if (
            row.get("execution_index") != execution_index
            or row.get("sampling_seed")
            != four_forward.sampling_seeds[execution_index]
            or row.get("exact_result_artifact_replay") is not True
            or row.get("exact_selected_token_score_replay") is not True
            or row.get("receipt_binding_valid") is not True
            or row.get("attestation_result_artifact_fingerprint")
            != expected_artifact_fingerprint
            or row.get("admitted_result_artifact_fingerprint")
            != expected_artifact_fingerprint
            or row.get("receipt_fingerprint") != receipt.receipt_fingerprint
            or row.get("generated_token_identifiers_hash")
            != receipt.generated_token_identifiers_hash
            or row.get("canonical_float32_score_trace_hash")
            != receipt.canonical_float32_score_trace_hash
            or row.get("compact_object_selected_token_score_replay")
            != expected_selected_replays[request_id]
        ):
            raise RuntimeContractError(
                "admitted production result row does not replay its attested result",
                code="backend_sampling.attestation_admitted_replay_result_mismatch",
                context={"request_id": request_id},
            )
    diagnostics_mismatch = (
        diagnostics.get("measurement_scope")
        != "physical_backend_call_pre_generation"
        or diagnostics.get("adapter_and_selected_embedding_tensor_count")
        != EXPECTED_ATTESTED_RUNTIME_PAYLOAD_TENSOR_COUNT
        or diagnostics.get(
            "adapter_and_selected_embedding_payload_byte_count"
        )
        != EXPECTED_ATTESTED_RUNTIME_PAYLOAD_BYTE_COUNT
        or diagnostics.get("base_model_payload_hashed") is not False
    )
    timing_fields = (
        "adapter_and_selected_embedding_hash_elapsed_seconds",
        "live_runtime_state_seal_elapsed_seconds",
    )
    timings = [diagnostics.get(field) for field in timing_fields]
    timings_invalid = any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) < 0.0
        for value in timings
    )
    if not timings_invalid and float(timings[1]) < float(timings[0]):
        timings_invalid = True
    if diagnostics_mismatch or timings_invalid:
        raise RuntimeContractError(
            "admitted production live-state diagnostics are invalid",
            code="backend_sampling.attestation_admitted_replay_diagnostics_invalid",
        )
    portable = payload.get("portable_runtime_state_seal")
    if not isinstance(portable, Mapping):
        raise RuntimeContractError(
            "admitted production replay lacks a portable runtime state seal",
            code="backend_sampling.attestation_portable_state_incomplete",
        )
    return {
        "replay_payload_fingerprint": replay_fingerprint,
        "portable_runtime_state_seal": (
            _validate_portable_runtime_state_seal_artifact(portable)
        ),
    }


def _parse_sampled_runtime_attestation_aggregate_output(
    source: str | Path | Mapping[str, Any],
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    payload = _load_sampled_runtime_attestation_aggregate_payload(source)
    if (
        payload.get("schema_version")
        != SAMPLED_RUNTIME_ATTESTATION_AGGREGATE_OUTPUT_SCHEMA_VERSION
    ):
        raise RuntimeContractError(
            "only a typed three-policy attestation aggregate can be rebound",
            code="backend_sampling.attestation_aggregate_required",
            context={"schema_version": payload.get("schema_version")},
        )
    aggregate = dict(payload)
    aggregate_fingerprint = aggregate.pop("aggregate_payload_fingerprint", None)
    if aggregate_fingerprint != _sha256_json(aggregate):
        raise RuntimeContractError(
            "sampled runtime attestation aggregate fingerprint is invalid",
            code="backend_sampling.attestation_aggregate_fingerprint_mismatch",
        )
    entries = payload.get("policy_attestations")
    if not isinstance(entries, list) or len(entries) != len(
        _ALLOWED_ATTESTATION_TEMPERATURES
    ):
        raise RuntimeContractError(
            "sampled runtime attestation aggregate requires exactly three policies",
            code="backend_sampling.attestation_aggregate_policy_set_invalid",
        )
    parsed_entries: list[dict[str, Any]] = []
    observed_temperatures: list[float] = []
    observed_policy_fingerprints: list[str] = []
    observed_bundle_fingerprints: list[str] = []
    observed_replay_fingerprints: list[str] = []
    for entry_payload in entries:
        if not isinstance(entry_payload, Mapping):
            raise RuntimeContractError(
                "sampled runtime policy attestation entry must be an object",
                code="backend_sampling.attestation_policy_entry_invalid",
            )
        entry = dict(entry_payload)
        entry_fingerprint = entry.pop("entry_payload_fingerprint", None)
        if (
            entry.get("schema_version")
            != SAMPLED_RUNTIME_ATTESTATION_POLICY_ENTRY_SCHEMA_VERSION
            or entry_fingerprint != _sha256_json(entry)
        ):
            raise RuntimeContractError(
                "sampled runtime policy attestation entry is invalid",
                code="backend_sampling.attestation_policy_entry_invalid",
            )
        policy_payload = entry.get("decode_generation_policy")
        bundle_payload = entry.get("attestation_bundle")
        admitted_payload = entry.get("admitted_production_replay")
        if not all(
            isinstance(value, Mapping)
            for value in (policy_payload, bundle_payload, admitted_payload)
        ):
            raise RuntimeContractError(
                "sampled runtime policy entry lacks typed nested evidence",
                code="backend_sampling.attestation_policy_entry_invalid",
            )
        policy = DecodeGenerationPolicy(**dict(policy_payload))
        bundle = SampledRuntimeAttestationBundle.from_artifact_dict(bundle_payload)
        temperature_value = entry.get("temperature")
        if (
            not isinstance(temperature_value, (int, float))
            or isinstance(temperature_value, bool)
            or not math.isfinite(float(temperature_value))
        ):
            raise RuntimeContractError(
                "sampled runtime policy entry temperature is invalid",
                code="backend_sampling.attestation_policy_binding_mismatch",
            )
        temperature = float(temperature_value)
        policy_fingerprint = entry.get("decode_generation_policy_fingerprint")
        if (
            policy.mode != "sampled"
            or policy.max_new_tokens != 512
            or policy.temperature != temperature
            or temperature not in _ALLOWED_ATTESTATION_TEMPERATURES
            or policy.top_p != 0.95
            or policy.repetition_penalty != 1.0
            or policy_fingerprint != policy.fingerprint
            or entry.get("bundle_payload_fingerprint")
            != bundle.bundle_payload_fingerprint
            or bundle.lineage.get("temperature") != temperature
        ):
            raise RuntimeContractError(
                "sampled runtime policy entry does not bind its exact policy",
                code="backend_sampling.attestation_policy_binding_mismatch",
            )
        replay_validation = _validate_admitted_production_replay_artifact(
            admitted_payload,
            bundle=bundle,
            decode_generation_policy_fingerprint=policy.fingerprint,
        )
        observed_temperatures.append(temperature)
        observed_policy_fingerprints.append(policy.fingerprint)
        observed_bundle_fingerprints.append(bundle.bundle_payload_fingerprint)
        observed_replay_fingerprints.append(
            replay_validation["replay_payload_fingerprint"]
        )
        parsed_entries.append(
            {
                "temperature": temperature,
                "decode_generation_policy": policy,
                "decode_generation_policy_fingerprint": policy.fingerprint,
                "attestation_bundle": bundle,
                "admitted_production_replay": dict(admitted_payload),
                "portable_runtime_state_seal": replay_validation[
                    "portable_runtime_state_seal"
                ],
                "entry_payload_fingerprint": entry_fingerprint,
            }
        )
    expected_temperatures = sorted(_ALLOWED_ATTESTATION_TEMPERATURES)
    if (
        observed_temperatures != expected_temperatures
        or len(set(observed_policy_fingerprints)) != len(expected_temperatures)
        or len(set(observed_bundle_fingerprints)) != len(expected_temperatures)
        or len(set(observed_replay_fingerprints)) != len(expected_temperatures)
    ):
        raise RuntimeContractError(
            "sampled runtime attestation aggregate policy set is not exact and unique",
            code="backend_sampling.attestation_aggregate_policy_set_invalid",
        )
    return payload, tuple(parsed_entries)


def validate_sampled_runtime_attestation_aggregate_output(
    source: str | Path | Mapping[str, Any],
) -> dict[str, Any]:
    """Fully validate a persisted three-policy output without minting capability."""

    payload, entries = _parse_sampled_runtime_attestation_aggregate_output(source)
    validations = [
        validate_sampled_runtime_attestation_bundle(entry["attestation_bundle"])
        for entry in entries
    ]
    return {
        "aggregate_payload_fingerprint": payload[
            "aggregate_payload_fingerprint"
        ],
        "decode_generation_policy_fingerprints": [
            entry["decode_generation_policy_fingerprint"] for entry in entries
        ],
        "bundle_payload_fingerprints": [
            validation["bundle_payload_fingerprint"] for validation in validations
        ],
        "production_admission": False,
    }


def load_and_rebind_sampled_runtime_attestation_aggregate(
    source: str | Path,
    *,
    decode_generation_policy_fingerprint: str,
    backend: HFGenerateBackend,
) -> VerifiedSampledRuntimeAttestation:
    """Mint one new process-local capability from persisted verified evidence."""

    _, entries = _parse_sampled_runtime_attestation_aggregate_output(source)
    validations = [
        _validate_sampled_runtime_attestation_bundle(entry["attestation_bundle"])
        for entry in entries
    ]
    matches = [
        entry
        for entry in entries
        if entry["decode_generation_policy_fingerprint"]
        == decode_generation_policy_fingerprint
    ]
    if len(matches) != 1:
        raise RuntimeContractError(
            "rebind requires exactly one attested decode policy fingerprint",
            code="backend_sampling.attestation_rebind_policy_not_found",
            context={
                "decode_generation_policy_fingerprint": (
                    decode_generation_policy_fingerprint
                ),
                "match_count": len(matches),
            },
        )
    selected = matches[0]
    selected_index = entries.index(selected)
    parsed, admission_contract = validations[selected_index]
    (
        model_identity,
        tokenizer_identity,
        generation_config_fingerprint,
    ) = backend._bound_runtime_identities()
    capability = _mint_verified_sampled_runtime_attestation(
        parsed,
        admission_contract=admission_contract,
        backend=backend,
        model_identity=model_identity,
        tokenizer_identity=tokenizer_identity,
        generation_config_fingerprint=generation_config_fingerprint,
    )
    active_portable_state = capability.portable_runtime_state_seal_artifact()
    expected_portable_state = selected["portable_runtime_state_seal"]
    if active_portable_state != expected_portable_state:
        raise RuntimeContractError(
            "active runtime state does not match persisted attested state",
            code="backend_sampling.attestation_rebind_live_state_mismatch",
            context={
                "mismatched_fields": sorted(
                    field
                    for field in set(active_portable_state)
                    | set(expected_portable_state)
                    if active_portable_state.get(field)
                    != expected_portable_state.get(field)
                )
            },
        )
    return capability


def _mint_verified_sampled_runtime_attestation(
    parsed: SampledRuntimeAttestationBundle,
    *,
    admission_contract: Mapping[str, Any],
    backend: HFGenerateBackend,
    model_identity: Mapping[str, Any],
    tokenizer_identity: Mapping[str, Any],
    generation_config_fingerprint: str,
) -> VerifiedSampledRuntimeAttestation:
    if any(
        value is not None
        for value in (
            backend._bound_model_identity,
            backend._bound_tokenizer_identity,
            backend._bound_generation_config_fingerprint,
        )
    ):
        bound_model, bound_tokenizer, bound_generation = (
            backend._bound_runtime_identities()
        )
        if (
            _thaw_json(model_identity) != bound_model
            or _thaw_json(tokenizer_identity) != bound_tokenizer
            or generation_config_fingerprint != bound_generation
        ):
            raise RuntimeContractError(
                "caller identities disagree with backend-owned runtime identities",
                code="backend_sampling.attestation_backend_identity_mismatch",
            )
    model_device = backend._model_device()
    active_device_identity = (
        None if model_device is None else _execution_device_identity(model_device)
    )
    qwen_identity = _thaw_json(parsed.lineage["qwen_runtime_identity"])
    mismatches: dict[str, Any] = {}
    expected_values = {
        "model_identity": admission_contract["model_identity"],
        "tokenizer_identity": admission_contract["tokenizer_identity"],
        "generation_config_fingerprint": admission_contract[
            "generation_config_fingerprint"
        ],
        "attention_implementation": admission_contract[
            "attention_implementation"
        ],
        "execution_device_identity": admission_contract[
            "execution_device_identity"
        ],
        "runtime_identity": admission_contract["runtime_identity"],
        "custom_sampler_code_hash": admission_contract["custom_sampler_code_hash"],
    }
    active_values = {
        "model_identity": dict(model_identity),
        "tokenizer_identity": dict(tokenizer_identity),
        "generation_config_fingerprint": generation_config_fingerprint,
        "attention_implementation": _attention_implementation(backend.model),
        "execution_device_identity": active_device_identity,
        "runtime_identity": _runtime_identity(),
        "custom_sampler_code_hash": custom_sampler_code_hash(),
    }
    for field, expected in expected_values.items():
        if _thaw_json(active_values[field]) != _thaw_json(expected):
            mismatches[field] = {
                "expected": _thaw_json(expected),
                "observed": _thaw_json(active_values[field]),
            }
    if (
        bool(getattr(backend.model, "training", False))
        or _model_type(backend.model) != "qwen3_vl"
        or _tokenizer_im_end_token_id(backend.tokenizer)
        != qwen_identity["tokens"]["im_end_token_ids"][0]
    ):
        mismatches["runtime_owned_qwen_identity"] = {
            "expected": "eval Qwen3-VL with the attested im_end token",
            "observed": {
                "model_type": _model_type(backend.model),
                "model_training": bool(getattr(backend.model, "training", False)),
                "im_end_token_id": _tokenizer_im_end_token_id(backend.tokenizer),
            },
        }
    if mismatches:
        raise RuntimeContractError(
            "live attestation backend does not match the executed bundle",
            code="backend_sampling.attestation_active_runtime_mismatch",
            context={"mismatches": mismatches},
        )
    runtime_state_seal, runtime_state_diagnostics = (
        _live_runtime_state_seal_with_diagnostics(
            backend.model,
            backend.tokenizer,
            hash_payloads=True,
        )
    )
    backend._last_runtime_state_seal_diagnostics = _freeze_json(
        {
            "measurement_scope": "capability_mint",
            **runtime_state_diagnostics,
        }
    )
    return VerifiedSampledRuntimeAttestation(
        bundle_payload_fingerprint=parsed.bundle_payload_fingerprint,
        admission_contract=admission_contract,
        backend_object_id=id(backend),
        model_object_id=id(backend.model),
        tokenizer_object_id=id(backend.tokenizer),
        runtime_state_seal=runtime_state_seal,
        _sentinel=_VERIFIED_SAMPLED_RUNTIME_ATTESTATION_SENTINEL,
    )


def _validate_verified_sampled_runtime_for_active_call(
    verified: VerifiedSampledRuntimeAttestation,
    *,
    backend: HFGenerateBackend,
    requests: Sequence[DecodeRequest],
    model_identity: Mapping[str, Any],
    tokenizer_identity: Mapping[str, Any],
    generation_config_fingerprint: str,
    attention_implementation: str,
    model_evaluation_mode: bool,
    runtime_identity: Mapping[str, Any],
    execution_device_identity: Mapping[str, Any],
    prepared_generation_profile: Mapping[str, Any],
    prompt_width: int,
) -> None:
    if not isinstance(verified, VerifiedSampledRuntimeAttestation):
        raise RuntimeContractError(
            "sampled production requires a verifier-minted capability object",
            code="backend_sampling.attestation_capability_not_verified",
        )
    if not verified.is_bound_to(backend):
        raise RuntimeContractError(
            "sampled runtime capability is not bound to this backend/model/tokenizer",
            code="backend_sampling.attestation_active_runtime_mismatch",
        )
    expected_runtime_state = _thaw_json(verified.runtime_state_seal)
    active_runtime_state, runtime_state_diagnostics = (
        _live_runtime_state_seal_with_diagnostics(
            backend.model,
            backend.tokenizer,
            hash_payloads=True,
        )
    )
    backend._last_runtime_state_seal_diagnostics = _freeze_json(
        {
            "measurement_scope": "physical_backend_call_pre_generation",
            **runtime_state_diagnostics,
        }
    )
    runtime_state_mismatches = {
        field: {
            "expected": expected_runtime_state.get(field),
            "observed": active_runtime_state.get(field),
        }
        for field in sorted(set(expected_runtime_state) | set(active_runtime_state))
        if expected_runtime_state.get(field) != active_runtime_state.get(field)
    }
    if runtime_state_mismatches:
        raise RuntimeContractError(
            "verified sampled runtime state changed after capability minting",
            code="backend_sampling.attestation_live_state_changed",
            context={"mismatches": runtime_state_mismatches},
        )
    policy = validate_decode_batch(requests)
    contract = verified.admission_contract
    active = {
        "model_identity": dict(model_identity),
        "tokenizer_identity": dict(tokenizer_identity),
        "generation_config_fingerprint": generation_config_fingerprint,
        "decode_generation_policy": policy.to_artifact_dict(),
        "normalized_prepared_generation_profile": _normalized_attested_generation_profile(
            prepared_generation_profile, prompt_width=prompt_width
        ),
        "attention_implementation": attention_implementation,
        "runtime_identity": dict(runtime_identity),
        "custom_sampler_code_hash": custom_sampler_code_hash(),
        "execution_device_identity": dict(execution_device_identity),
    }
    mismatches = {
        field: {
            "expected": _thaw_json(contract.get(field)),
            "observed": _thaw_json(value),
        }
        for field, value in active.items()
        if _thaw_json(contract.get(field)) != _thaw_json(value)
    }
    if any(
        value is not None
        for value in (
            backend._bound_model_identity,
            backend._bound_tokenizer_identity,
            backend._bound_generation_config_fingerprint,
        )
    ):
        bound_model, bound_tokenizer, bound_generation = (
            backend._bound_runtime_identities()
        )
        caller_owned_identity_values = {
            "model_identity": dict(model_identity),
            "tokenizer_identity": dict(tokenizer_identity),
            "generation_config_fingerprint": generation_config_fingerprint,
        }
        backend_owned_identity_values = {
            "model_identity": bound_model,
            "tokenizer_identity": bound_tokenizer,
            "generation_config_fingerprint": bound_generation,
        }
        for field, backend_owned_value in backend_owned_identity_values.items():
            if _thaw_json(caller_owned_identity_values[field]) != _thaw_json(
                backend_owned_value
            ):
                mismatches[f"backend_owned_{field}"] = {
                    "expected": _thaw_json(backend_owned_value),
                    "observed": _thaw_json(caller_owned_identity_values[field]),
                }
    if not model_evaluation_mode:
        mismatches["model_evaluation_mode"] = {
            "expected": True,
            "observed": False,
        }
    if mismatches:
        raise RuntimeContractError(
            "verified sampled runtime does not match the active production call",
            code="backend_sampling.attestation_active_runtime_mismatch",
            context={"mismatches": mismatches},
        )


def _synchronize_device_for_timing(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _attention_implementation(model: Any) -> str:
    config = getattr(model, "config", None)
    value = getattr(config, "_attn_implementation", None)
    return "unknown" if value is None else str(value)


def _model_type(model: Any) -> str:
    config = getattr(model, "config", None)
    value = getattr(config, "model_type", None)
    return "unknown" if value is None else str(value)


def _tokenizer_im_end_token_id(tokenizer: Any) -> int:
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if callable(convert):
        value = convert("<|im_end|>")
        if value is not None:
            return int(value)
    value = getattr(tokenizer, "eos_token_id", None)
    if value is None:
        raise RuntimeContractError(
            "tokenizer does not expose Qwen im_end/eos token id",
            code="backend_trace.missing_stop_token",
        )
    return int(value)


def _token_identifiers_hash(token_ids: Sequence[int]) -> str:
    return _sha256_json([int(token_id) for token_id in token_ids])


def _model_inputs_fingerprint(model_inputs: Mapping[str, Any]) -> str:
    """Hash the exact non-text request inputs consumed by batch collation."""

    canonical = {
        str(key): _canonical_model_input_value(value)
        for key, value in model_inputs.items()
        if key not in {"input_ids", "attention_mask"}
    }
    return _sha256_json(canonical)


def _canonical_model_input_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().to(device="cpu").contiguous()
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
        return {
            "kind": "tensor",
            "dtype": str(tensor.dtype),
            "shape": [int(dimension) for dimension in tensor.shape],
            "raw_sha256": hashlib.sha256(raw).hexdigest(),
        }
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_model_input_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_model_input_value(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise RuntimeContractError(
        "request model input cannot be fingerprinted canonically",
        code="backend_sampling.attestation_model_input_unsupported",
        context={"value_type": type(value).__name__},
    )


def _canonical_float32_score_trace_hash(
    token_trace: Sequence[TokenTrace],
) -> str:
    canonical: list[dict[str, Any]] = []
    for trace in token_trace:
        if trace.is_pad:
            logprob = None
        else:
            if trace.logprob is None:
                raise RuntimeContractError(
                    "generated non-padding token requires a finite selected-token logprob",
                    code="backend_receipt.non_finite_selected_logprob",
                    context={
                        "step_index": trace.step_index,
                        "token_id": trace.token_id,
                    },
                )
            logprob = canonical_float32_logprob(
                trace.logprob,
                error_code="backend_receipt.non_finite_selected_logprob",
                context={
                    "step_index": trace.step_index,
                    "token_id": trace.token_id,
                },
            )
        canonical.append(
            {
                "step_index": trace.step_index,
                "token_id": trace.token_id,
                "logprob": logprob,
                "is_stop": trace.is_stop,
                "is_pad": trace.is_pad,
            }
        )
    return _sha256_json(canonical)


def _sampling_profile_fingerprint(sampling_profile: str) -> str:
    return _sha256_json({"sampling_profile": sampling_profile})


def canonical_float32_logprob(
    value: Any,
    *,
    error_code: str = "backend_receipt.non_finite_selected_logprob",
    context: Mapping[str, Any] | None = None,
) -> float:
    """Promote one score to the exact IEEE-754 binary32 value used by replay."""

    try:
        source = float(value)
        canonical = struct.unpack(">f", struct.pack(">f", source))[0]
    except (OverflowError, TypeError, ValueError, struct.error) as exc:
        raise RuntimeContractError(
            "selected-token logprob cannot be represented as finite float32",
            code=error_code,
            context=dict(context or {}),
            cause=exc,
        ) from exc
    if not math.isfinite(source) or not math.isfinite(canonical):
        raise RuntimeContractError(
            "selected-token logprob cannot be represented as finite float32",
            code=error_code,
            context=dict(context or {}),
        )
    return canonical


def _sha256_json(payload: Any) -> str:
    try:
        encoded = json.dumps(
            _thaw_json(payload),
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RuntimeContractError(
            "canonical backend payload is not finite JSON",
            code="backend_receipt.non_json_value",
            context={"payload_type": type(payload).__name__},
            cause=exc,
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_canonical_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RuntimeContractError(
                "receipt canonical payload contains a non-finite float",
                code="backend_receipt.non_finite_value",
            )
        return value
    raise RuntimeContractError(
        "receipt canonical payload contains a non-JSON value",
        code="backend_receipt.non_json_value",
        context={"value_type": type(value).__name__},
    )


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    if isinstance(value, list):
        return [_thaw_json(item) for item in value]
    return value


def _is_positive_finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def _is_valid_sampling_seed(value: Any) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value <= MAX_SIGNED_64_BIT_SEED
    )


def create_backend(
    backend: BackendName | str, *, model: Any, tokenizer: Any
) -> HFGenerateBackend:
    if backend == "hf":
        return HFGenerateBackend(model=model, tokenizer=tokenizer)
    if backend == "vllm":
        raise RuntimeContractError(
            "vLLM backend is reserved but not implemented for CoordExp-swift V1",
            code="backend_trace.backend_not_implemented",
            context={"backend": backend},
        )
    raise RuntimeContractError(
        "unknown inference backend",
        code="backend_trace.unknown_backend",
        context={"backend": backend},
    )


def _as_tensor(value: Any) -> torch.Tensor:
    return value if isinstance(value, torch.Tensor) else torch.as_tensor(value)


def _collate_model_input_values(values: list[Any], key: str) -> Any:
    if key in {"pixel_values", "pixel_values_videos"}:
        return _cat_patch_values(values, field=key)
    if key in {"image_grid_thw", "video_grid_thw"}:
        return _cat_grid_thw(values, field=key)
    if len(values) == 1:
        return values[0]
    return values


def _move_to_device(value: Any, *, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, Mapping):
        return {
            key: _move_to_device(item, device=device) for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device=device) for item in value)
    if isinstance(value, list):
        return [_move_to_device(item, device=device) for item in value]
    return value


def _cat_patch_values(values: list[Any], *, field: str) -> torch.Tensor:
    tensors = _require_tensor_values(values, field=field)
    first = tensors[0]
    if first.ndim < 2:
        raise RuntimeContractError(
            "Qwen patch values must be at least rank 2",
            code="backend_trace.model_input_shape",
            context={"field": field, "shape": tuple(first.shape)},
        )
    trailing_shape = tuple(first.shape[1:])
    for index, tensor in enumerate(tensors):
        if tensor.ndim < 2 or tuple(tensor.shape[1:]) != trailing_shape:
            raise RuntimeContractError(
                "Qwen patch values must share trailing dimensions",
                code="backend_trace.model_input_shape",
                context={
                    "field": field,
                    "index": index,
                    "shape": tuple(tensor.shape),
                    "expected_trailing_shape": trailing_shape,
                },
            )
    return torch.cat(tensors, dim=0)


def _cat_grid_thw(values: list[Any], *, field: str) -> torch.Tensor:
    tensors = _require_tensor_values(values, field=field)
    for index, tensor in enumerate(tensors):
        if tensor.ndim != 2 or tensor.shape[1] != 3:
            raise RuntimeContractError(
                "Qwen grid THW inputs must be rank 2 with width 3",
                code="backend_trace.model_input_shape",
                context={"field": field, "index": index, "shape": tuple(tensor.shape)},
            )
    return torch.cat(tensors, dim=0)


def _require_tensor_values(values: list[Any], *, field: str) -> list[torch.Tensor]:
    if not all(isinstance(value, torch.Tensor) for value in values):
        raise RuntimeContractError(
            "Qwen model input collation requires tensor values",
            code="backend_trace.model_input_type",
            context={"field": field},
        )
    return values


def _strip_terminal_im_end(
    text: str,
    *,
    stop_id: int,
    kept_ids: list[int],
    stop_text: str,
) -> tuple[str, str]:
    if kept_ids and kept_ids[-1] == stop_id and text.endswith(stop_text):
        return text[: -len(stop_text)], "terminal_im_end"
    return text, "none"
