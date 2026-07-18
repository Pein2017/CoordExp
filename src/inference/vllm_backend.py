"""Offline vLLM backend session for deterministic Qwen3-VL inference."""

from __future__ import annotations

import gc
import hashlib
import inspect
import json
import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from importlib import metadata
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image
import torch

from src.common.errors import RuntimeContractError
from src.config.inference import validate_vllm_runtime_version
from src.inference.backend import (
    POLICY_LIKELIHOOD_DEFINITION,
    RAW_LIKELIHOOD_DEFINITION,
    BackendLaunch,
    BackendSessionReceipt,
    DecodeRequest,
    DecodeResult,
    LikelihoodPair,
    TokenTrace,
    cuda_peak_memory_snapshot,
    synchronize_cuda_for_timing,
    token_ids_sha256,
    update_decode_performance_receipt,
    validate_decode_results,
)
from src.qwen.images import apply_logical_image_transform, rgb_image_sha256
from src.qwen.runtime_loading import QwenLoadOptions, load_qwen_components_from_options


VLLM_MAX_MODEL_LEN = 2048
_FORCED_REPLAY_IDS_KEY = "coordexp_expected_token_ids"
EngineFactory = Callable[[Mapping[str, object]], Any]
ComponentsLoader = Callable[[BackendLaunch], Any]
RawReplayQualifier = Callable[[BackendLaunch, Mapping[str, object]], Mapping[str, object]]


class VLLMBackendSession:
    backend = "vllm"
    backend_mode = "offline_generate"
    response_family = "vllm"

    def __init__(
        self,
        *,
        launch: BackendLaunch,
        engine: Any,
        engine_factory: EngineFactory,
        engine_kwargs: Mapping[str, object],
        tokenizer: Any,
        receipt: BackendSessionReceipt,
        forced_replay_processor: type[Any] | None = None,
        raw_replay_qualifier: RawReplayQualifier | None = None,
    ) -> None:
        receipt.validate_for_launch(launch)
        self._launch = launch
        self._engine = engine
        self._engine_factory = engine_factory
        self._engine_kwargs = dict(engine_kwargs)
        self._tokenizer = tokenizer
        self._receipt = receipt
        self._forced_replay_processor = forced_replay_processor
        self._raw_replay_qualifier = raw_replay_qualifier
        self._raw_replay_evidence: dict[str, dict[str, object]] = {}
        self._closed = False

    @property
    def receipt(self) -> BackendSessionReceipt:
        return self._receipt

    def decode(self, requests: Sequence[DecodeRequest]) -> tuple[DecodeResult, ...]:
        if self._closed:
            raise RuntimeContractError(
                "vLLM backend session is already closed",
                code="vllm_backend.session_closed",
            )
        checked = tuple(requests)
        if not checked:
            return ()
        request_ids = [request.request_id for request in checked]
        if len(set(request_ids)) != len(request_ids):
            raise RuntimeContractError(
                "vLLM backend session requires unique request ids",
                code="vllm_backend.duplicate_request_id",
                context={"request_ids": request_ids},
            )
        synchronize_cuda_for_timing(torch)
        started_at = time.perf_counter()
        policy = _require_shared_generation_policy(checked)
        prompts, media_hashes = self._generation_prompts(checked)
        try:
            outputs = self._engine.generate(
                prompts,
                _generation_sampling_params(
                    policy=policy,
                    stop_token_id=self._im_end_token_id(),
                ),
                use_tqdm=False,
            )
        finally:
            _close_prompt_images(prompts)
        ordered_outputs = _restore_native_request_order(outputs, len(checked))
        raw_channels = (
            self._raw_replay(checked, ordered_outputs)
            if policy.include_raw_model_logprob
            else (None,) * len(checked)
        )
        results = tuple(
            self._materialize_result(
                request=request,
                native_output=native,
                executed_media_sha256=media_hash,
                raw_logprobs=raw_logprobs,
            )
            for request, native, media_hash, raw_logprobs in zip(
                checked,
                ordered_outputs,
                media_hashes,
                raw_channels,
                strict=True,
            )
        )
        validated = validate_decode_results(
            requests=checked,
            results=results,
            receipt=self.receipt,
        )
        synchronize_cuda_for_timing(torch)
        allocated, reserved = cuda_peak_memory_snapshot(torch)
        self._receipt = update_decode_performance_receipt(
            self._receipt,
            request_count=len(checked),
            generated_token_count=sum(len(result.generated_token_ids) for result in validated),
            elapsed_seconds=time.perf_counter() - started_at,
            peak_cuda_memory_allocated_bytes=allocated,
            peak_cuda_memory_reserved_bytes=reserved,
        )
        return validated

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        engine = self._engine
        self._engine = None
        self._tokenizer = None
        _close_vllm_engine(engine)

    def _generation_prompts(
        self,
        requests: Sequence[DecodeRequest],
    ) -> tuple[list[Any], tuple[str, ...]]:
        from vllm.inputs import TextPrompt

        prompts: list[Any] = []
        media_hashes: list[str] = []
        for request in requests:
            image = _open_verified_rgb_image(request)
            media_hashes.append(rgb_image_sha256(image))
            prompts.append(
                TextPrompt(
                    prompt=request.chat_text,
                    multi_modal_data={"image": image},
                    mm_processor_kwargs={"do_resize": False},
                )
            )
        return prompts, tuple(media_hashes)

    def _raw_replay(
        self,
        requests: Sequence[DecodeRequest],
        generation_outputs: Sequence[Any],
    ) -> tuple[tuple[float, ...], ...]:
        generated_by_request = [
            _generated_token_ids(native) for native in generation_outputs
        ]
        policy = _require_shared_generation_policy(requests)
        processor = self._resolve_forced_replay_processor()
        processor_identity = _processor_source_identity(processor)
        qualification = self._qualify_raw_replay(processor_identity)
        processed_engine = self._engine
        self._engine = None
        _close_vllm_engine(processed_engine)
        raw_engine_kwargs = _raw_replay_engine_kwargs(
            self._engine_kwargs,
            forced_replay_processor=processor,
        )
        self._engine = self._engine_factory(raw_engine_kwargs)
        prompts, _ = self._generation_prompts(requests)
        sampling_params = [
            _raw_forced_replay_sampling_params(
                policy=policy,
                expected_token_ids=generated,
                stop_token_id=self._im_end_token_id(),
            )
            for generated in generated_by_request
        ]
        try:
            replay_outputs = self._engine.generate(
                prompts,
                sampling_params,
                use_tqdm=False,
            )
        finally:
            _close_prompt_images(prompts)
        ordered = _restore_native_request_order(replay_outputs, len(requests))
        channels: list[tuple[float, ...]] = []
        replay_evidence: dict[str, dict[str, object]] = {}
        for request, generated, generation, replay in zip(
            requests,
            generated_by_request,
            generation_outputs,
            ordered,
            strict=True,
        ):
            expected_ids = request.expected_executed_prompt_token_ids
            observed_ids = tuple(int(value) for value in (replay.prompt_token_ids or ()))
            if observed_ids != expected_ids:
                raise RuntimeContractError(
                    "vLLM raw replay prompt ids differ from authoritative generation",
                    code="vllm_backend.raw_replay_prompt_mismatch",
                    context={"request_id": request.request_id},
                )
            replay_generated = _generated_token_ids(replay)
            if replay_generated != generated:
                raise RuntimeContractError(
                    "vLLM raw replay did not retain the authoritative continuation",
                    code="vllm_backend.raw_replay_token_mismatch",
                    context={
                        "request_id": request.request_id,
                        "expected": list(generated),
                        "observed": list(replay_generated),
                    },
                )
            generation_completion = _single_completion(
                generation,
                request_id=request.request_id,
                code="vllm_backend.raw_replay_alignment",
            )
            replay_completion = _single_completion(
                replay,
                request_id=request.request_id,
                code="vllm_backend.raw_replay_alignment",
            )
            generation_finish = str(
                getattr(generation_completion, "finish_reason", "") or ""
            )
            replay_finish = str(
                getattr(replay_completion, "finish_reason", "") or ""
            )
            generation_native_stop = getattr(
                generation_completion, "stop_reason", None
            )
            replay_native_stop = getattr(replay_completion, "stop_reason", None)
            if (
                replay_finish != generation_finish
                or replay_native_stop != generation_native_stop
            ):
                raise RuntimeContractError(
                    "vLLM raw replay stop evidence differs from authoritative generation",
                    code="vllm_backend.raw_replay_stop_mismatch",
                    context={
                        "request_id": request.request_id,
                        "generation_finish_reason": generation_finish,
                        "replay_finish_reason": replay_finish,
                        "generation_native_stop_reason": generation_native_stop,
                        "replay_native_stop_reason": replay_native_stop,
                    },
                )
            completions = getattr(replay, "outputs", None)
            if not isinstance(completions, Sequence) or len(completions) != 1:
                _native_failure(
                    "vLLM raw replay returned an invalid completion count",
                    code="vllm_backend.raw_replay_alignment",
                    request_id=request.request_id,
                )
            channels.append(
                _chosen_logprobs(
                    token_ids=generated,
                    positions=getattr(completions[0], "logprobs", None),
                    channel="raw_model",
                    request_id=request.request_id,
                )
            )
            replay_evidence[request.request_id] = {
                "prompt_token_count": len(observed_ids),
                "prompt_token_ids_sha256": token_ids_sha256(observed_ids),
                "generated_token_count": len(replay_generated),
                "generated_token_ids_sha256": token_ids_sha256(replay_generated),
                "finish_reason": replay_finish,
                "native_stop_reason": replay_native_stop,
                "status": "verified",
            }
        self._raw_replay_evidence = replay_evidence
        self._receipt = replace(
            self._receipt,
            effective_settings={
                **dict(self._receipt.effective_settings),
                "raw_replay": {
                    "status": "completed",
                    "logprobs_mode": raw_engine_kwargs["logprobs_mode"],
                    "max_num_seqs": raw_engine_kwargs.get(
                        "max_num_seqs", self._launch.batch_size
                    ),
                    "gpu_memory_utilization": raw_engine_kwargs[
                        "gpu_memory_utilization"
                    ],
                    "kv_cache_memory_bytes": raw_engine_kwargs[
                        "kv_cache_memory_bytes"
                    ],
                    "forced_logits_processor": _processor_source_identity(
                        processor
                    ),
                    "qualification": dict(qualification),
                    "request_count": len(replay_evidence),
                    "row_evidence_sha256": _sha256_json(replay_evidence),
                },
            },
        )
        return tuple(channels)

    def _qualify_raw_replay(
        self,
        processor_identity: Mapping[str, object],
    ) -> Mapping[str, object]:
        qualifier = self._raw_replay_qualifier
        if qualifier is None:
            from src.inference.vllm_qualification import (
                validate_vllm_forced_replay_qualification,
            )

            qualifier = lambda launch, identity: validate_vllm_forced_replay_qualification(
                launch=launch,
                processor_identity=identity,
            )
        return qualifier(self._launch, processor_identity)

    def _resolve_forced_replay_processor(self) -> type[Any]:
        if self._forced_replay_processor is None:
            from src.inference.vllm_forced_replay import (
                CoordExpForcedSequenceLogitsProcessor,
            )

            self._forced_replay_processor = CoordExpForcedSequenceLogitsProcessor
        return self._forced_replay_processor

    def _materialize_result(
        self,
        *,
        request: DecodeRequest,
        native_output: Any,
        executed_media_sha256: str,
        raw_logprobs: tuple[float, ...] | None,
    ) -> DecodeResult:
        candidates = getattr(native_output, "outputs", None)
        if not isinstance(candidates, Sequence) or len(candidates) != 1:
            _native_failure(
                "vLLM must return exactly one completion per request",
                code="vllm_backend.output_count",
                request_id=request.request_id,
            )
        completion = candidates[0]
        prompt_ids = tuple(
            int(value) for value in (native_output.prompt_token_ids or ())
        )
        if prompt_ids != request.expected_executed_prompt_token_ids:
            _native_failure(
                "vLLM executed prompt ids differ from shared Qwen expansion",
                code="vllm_backend.prompt_token_mismatch",
                request_id=request.request_id,
            )
        (
            image_placeholder_ranges,
            backend_reported_placeholders,
        ) = _validate_image_placeholder_evidence(
            native_output=native_output,
            prompt_ids=prompt_ids,
            image_pad_token_id=self._image_pad_token_id(),
            request_id=request.request_id,
        )
        generated = _generated_token_ids(native_output)
        if not generated:
            _native_failure(
                "vLLM returned no generated tokens",
                code="vllm_backend.empty_generation",
                request_id=request.request_id,
            )
        policy_logprobs = _chosen_logprobs(
            token_ids=generated,
            positions=getattr(completion, "logprobs", None),
            channel="policy",
            request_id=request.request_id,
        )
        if raw_logprobs is not None and len(raw_logprobs) != len(generated):
            _native_failure(
                "vLLM raw likelihood channel is not token aligned",
                code="vllm_backend.raw_replay_alignment",
                request_id=request.request_id,
            )
        stop_id = self._im_end_token_id()
        stop_positions = [index for index, token_id in enumerate(generated) if token_id == stop_id]
        if stop_positions and stop_positions != [len(generated) - 1]:
            _native_failure(
                "vLLM emitted a non-terminal im_end token",
                code="vllm_backend.post_stop_content",
                request_id=request.request_id,
            )
        finish_reason = str(getattr(completion, "finish_reason", "") or "")
        if finish_reason == "stop" and stop_positions != [len(generated) - 1]:
            _native_failure(
                "vLLM stop completion did not retain terminal im_end",
                code="vllm_backend.stop_token_missing",
                request_id=request.request_id,
            )
        if finish_reason == "length" and stop_positions:
            _native_failure(
                "vLLM retained im_end but reported a length finish",
                code="vllm_backend.finish_reason",
                request_id=request.request_id,
            )
        if finish_reason not in {"stop", "length"}:
            _native_failure(
                "vLLM returned an unsupported finish reason",
                code="vllm_backend.finish_reason",
                request_id=request.request_id,
            )
        native_stop_reason = getattr(completion, "stop_reason", None)
        if native_stop_reason is not None:
            _native_failure(
                "vLLM native stop reason differs from the qualified tuple",
                code="vllm_backend.native_stop_reason",
                request_id=request.request_id,
            )
        stop_reason = "im_end" if stop_positions else "length"
        raw_text = self._decode_tokens(generated)
        stop_text = self._decode_tokens((stop_id,))
        if stop_reason == "im_end" and raw_text.endswith(stop_text):
            parser_text = raw_text[: -len(stop_text)]
            strip_policy = "terminal_im_end"
        else:
            parser_text = raw_text
            strip_policy = "none"
        traces = tuple(
            TokenTrace(
                step_index=index,
                token_id=token_id,
                token_text=self._decode_tokens((token_id,)),
                likelihood=LikelihoodPair(
                    policy_logprob=policy_logprobs[index],
                    raw_model_logprob=(
                        None if raw_logprobs is None else raw_logprobs[index]
                    ),
                ),
                is_stop=token_id == stop_id,
                is_pad=False,
                backend=self.backend,
                backend_mode=self.backend_mode,
                response_family=self.response_family,
            )
            for index, token_id in enumerate(generated)
        )
        expected_native_text = self._decode_tokens(
            generated[:-1] if stop_reason == "im_end" else generated
        )
        native_text = str(getattr(completion, "text", ""))
        if native_text != expected_native_text:
            _native_failure(
                "vLLM native text differs from authoritative token-id decoding",
                code="vllm_backend.native_text_mismatch",
                request_id=request.request_id,
            )
        return DecodeResult(
            request_id=request.request_id,
            backend=self.backend,
            backend_mode=self.backend_mode,
            response_family=self.response_family,
            executed_prompt_token_ids=prompt_ids,
            generated_token_ids=generated,
            raw_generated_text=raw_text,
            parser_text=parser_text,
            strip_policy=strip_policy,
            stop_reason=stop_reason,
            token_trace=traces,
            executed_media_sha256=executed_media_sha256,
            observed_image_grid_thw=None,
            native_generated_text=native_text,
            native_evidence={
                "native_request_id": str(native_output.request_id),
                "finish_reason": finish_reason,
                "native_stop_reason": native_stop_reason,
                "image_placeholder_ranges": image_placeholder_ranges,
                "backend_reported_multi_modal_placeholders": (
                    backend_reported_placeholders
                ),
                "prompt_token_count": len(prompt_ids),
                "do_resize": False,
                "raw_replay": self._raw_replay_evidence.get(request.request_id),
            },
        )

    def _im_end_token_id(self) -> int:
        convert = getattr(self._tokenizer, "convert_tokens_to_ids", None)
        if not callable(convert):
            raise RuntimeContractError(
                "vLLM tokenizer does not expose token conversion",
                code="vllm_backend.stop_token_missing",
            )
        value = convert("<|im_end|>")
        if value is None:
            raise RuntimeContractError(
                "vLLM tokenizer does not expose the Qwen im_end token",
                code="vllm_backend.stop_token_missing",
            )
        return int(value)

    def _decode_tokens(self, token_ids: Sequence[int]) -> str:
        return str(
            self._tokenizer.decode(
                list(token_ids),
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        )

    def _image_pad_token_id(self) -> int:
        value = self._tokenizer.convert_tokens_to_ids("<|image_pad|>")
        if value is None:
            raise RuntimeContractError(
                "vLLM tokenizer does not expose the Qwen image-pad token",
                code="vllm_backend.image_pad_token_missing",
            )
        return int(value)


def open_vllm_backend_session(
    launch: BackendLaunch,
    *,
    engine_factory: EngineFactory | None = None,
    components_loader: ComponentsLoader | None = None,
    raw_replay_qualifier: RawReplayQualifier | None = None,
) -> VLLMBackendSession:
    """Open one rank-local vLLM engine over an immutable execution model."""

    if launch.backend != "vllm":
        raise RuntimeContractError(
            "vLLM session opener received a non-vLLM launch",
            code="vllm_backend.launch_backend",
            context={"backend": launch.backend},
        )
    if launch.execution_model_identity is None:
        raise RuntimeContractError(
            "vLLM session requires a validated execution-model identity",
            code="vllm_backend.execution_model_required",
        )
    observed_version = metadata.version("vllm")
    validate_vllm_runtime_version(observed_version=observed_version)
    _configure_process_mode()
    _validate_rank_local_cuda()
    options = _vllm_options(launch)
    engine_kwargs = _engine_kwargs(launch, options=options)
    if engine_factory is None:
        from src.inference.vllm_qualification import (
            validate_vllm_runtime_qualification,
        )

        qualification = validate_vllm_runtime_qualification(
            launch=launch,
            engine_kwargs=engine_kwargs,
        )
    else:
        qualification = {"status": "injected_test_engine"}
    resolved_engine_factory = engine_factory or _default_engine_factory
    engine = resolved_engine_factory(engine_kwargs)
    try:
        components = (components_loader or _load_processor_components)(launch)
        tokenizer = components.tokenizer
        model_identity = _model_identity(launch)
        tokenizer_identity = _identity_dict(
            getattr(components, "token_identity", None),
            fallback={"tokenizer_class": type(tokenizer).__name__},
        )
        processor_identity = _identity_dict(
            getattr(components, "processor_identity", None),
            fallback={"processor_class": type(components.processor).__name__},
        )
        receipt = BackendSessionReceipt(
            backend="vllm",
            backend_mode=VLLMBackendSession.backend_mode,
            response_family=VLLMBackendSession.response_family,
            backend_version=observed_version,
            model_identity=model_identity,
            tokenizer_identity=tokenizer_identity,
            processor_identity=processor_identity,
            generation_config_fingerprint=launch.generation_config_fingerprint,
            effective_settings={
                "batch_size": launch.batch_size,
                "device": "cuda:0",
                "engine_kwargs": engine_kwargs,
                "scheduler_owned_batching": True,
                "runtime_qualification": qualification,
            },
            likelihood_semantics={
                "policy": POLICY_LIKELIHOOD_DEFINITION,
                "raw": RAW_LIKELIHOOD_DEFINITION,
                "score_owned_channel": "policy_logprob",
                "raw_channel_strategy": "forced_decode_raw_logprobs_replay",
            },
            execution_model_identity=launch.execution_model_identity,
        )
        return VLLMBackendSession(
            launch=launch,
            engine=engine,
            engine_factory=resolved_engine_factory,
            engine_kwargs=engine_kwargs,
            tokenizer=tokenizer,
            receipt=receipt,
            raw_replay_qualifier=raw_replay_qualifier,
        )
    except BaseException:
        _close_vllm_engine(engine)
        raise


def _engine_kwargs(
    launch: BackendLaunch,
    *,
    options: Mapping[str, object],
) -> dict[str, object]:
    dtype = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}[
        launch.model_dtype
    ]
    return {
        "model": launch.model_path,
        "tokenizer": launch.model_path,
        "trust_remote_code": True,
        "tensor_parallel_size": 1,
        "data_parallel_size": 1,
        "dtype": dtype,
        "seed": 0,
        "gpu_memory_utilization": options["gpu_memory_utilization"],
        "max_model_len": VLLM_MAX_MODEL_LEN,
        "max_num_seqs": launch.batch_size,
        "disable_custom_all_reduce": True,
        "disable_log_stats": True,
        "generation_config": "vllm",
        "logprobs_mode": "processed_logprobs",
        "limit_mm_per_prompt": {"image": 1, "video": 0},
        "mm_processor_kwargs": {"do_resize": False},
    }


def _generation_sampling_params(*, policy: Any, stop_token_id: int) -> Any:
    from vllm import SamplingParams

    return SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        repetition_penalty=policy.repetition_penalty,
        max_tokens=policy.max_new_tokens,
        logprobs=0,
        stop_token_ids=[stop_token_id],
        ignore_eos=False,
        detokenize=True,
        skip_special_tokens=False,
        spaces_between_special_tokens=True,
    )


def _raw_forced_replay_sampling_params(
    *,
    policy: Any,
    expected_token_ids: Sequence[int],
    stop_token_id: int,
) -> Any:
    from vllm import SamplingParams

    return SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        repetition_penalty=policy.repetition_penalty,
        max_tokens=len(expected_token_ids),
        logprobs=0,
        stop_token_ids=[stop_token_id],
        ignore_eos=False,
        detokenize=True,
        skip_special_tokens=False,
        spaces_between_special_tokens=True,
        extra_args={_FORCED_REPLAY_IDS_KEY: list(expected_token_ids)},
    )


def _raw_replay_engine_kwargs(
    engine_kwargs: Mapping[str, object],
    *,
    forced_replay_processor: type[Any],
) -> dict[str, object]:
    return {
        **dict(engine_kwargs),
        "logprobs_mode": "raw_logprobs",
        "logits_processors": [forced_replay_processor],
        "gpu_memory_utilization": 0.20,
        "kv_cache_memory_bytes": 1024 * 1024 * 1024,
    }


def _single_completion(
    native_output: Any,
    *,
    request_id: str,
    code: str,
) -> Any:
    completions = getattr(native_output, "outputs", None)
    if not isinstance(completions, Sequence) or len(completions) != 1:
        _native_failure(
            "vLLM output does not contain exactly one completion",
            code=code,
            request_id=request_id,
        )
    return completions[0]


def _processor_source_identity(processor: type[Any]) -> dict[str, object]:
    try:
        source_path = inspect.getsourcefile(processor)
    except (OSError, TypeError):
        source_path = None
    if source_path is None:
        return {
            "module": processor.__module__,
            "qualname": processor.__qualname__,
            "source_path": None,
            "source_sha256": None,
        }
    path = Path(source_path).resolve()
    return {
        "module": processor.__module__,
        "qualname": processor.__qualname__,
        "source_path": str(path),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _sha256_json(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _generated_token_ids(native_output: Any) -> tuple[int, ...]:
    completions = getattr(native_output, "outputs", None)
    if not isinstance(completions, Sequence) or len(completions) != 1:
        raise RuntimeContractError(
            "vLLM output does not contain exactly one completion",
            code="vllm_backend.output_count",
        )
    return tuple(int(value) for value in completions[0].token_ids)


def _chosen_logprobs(
    *,
    token_ids: Sequence[int],
    positions: Any,
    channel: str,
    request_id: str,
) -> tuple[float, ...]:
    if positions is None or len(positions) != len(token_ids):
        _native_failure(
            f"vLLM {channel} likelihood positions are not token aligned",
            code="vllm_backend.likelihood_alignment",
            request_id=request_id,
        )
    values: list[float] = []
    for index, (token_id, candidates) in enumerate(
        zip(token_ids, positions, strict=True)
    ):
        if candidates is None or token_id not in candidates:
            _native_failure(
                f"vLLM {channel} likelihood is missing the chosen token",
                code="vllm_backend.likelihood_token_missing",
                request_id=request_id,
                context={
                    "channel": channel,
                    "generated_step_index": index,
                    "token_id": token_id,
                },
            )
        value = float(candidates[token_id].logprob)
        if not math.isfinite(value) or value > 1e-6:
            _native_failure(
                f"vLLM {channel} likelihood is non-finite or positive",
                code="vllm_backend.likelihood_value",
                request_id=request_id,
                context={
                    "channel": channel,
                    "generated_step_index": index,
                    "token_id": token_id,
                    "logprob": value,
                },
            )
        values.append(value)
    return tuple(values)


def _restore_native_request_order(outputs: Any, expected_count: int) -> tuple[Any, ...]:
    if not isinstance(outputs, Sequence) or len(outputs) != expected_count:
        raise RuntimeContractError(
            "vLLM returned the wrong number of request outputs",
            code="vllm_backend.result_set",
            context={"expected_count": expected_count},
        )
    try:
        ordered = tuple(sorted(outputs, key=lambda item: int(item.request_id)))
        native_ids = [int(item.request_id) for item in ordered]
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeContractError(
            "vLLM request outputs have invalid native request ids",
            code="vllm_backend.native_request_id",
            cause=exc,
        ) from exc
    if len(set(native_ids)) != expected_count:
        raise RuntimeContractError(
            "vLLM request outputs contain duplicate native request ids",
            code="vllm_backend.native_request_id",
            context={"request_ids": native_ids},
        )
    if native_ids != list(range(native_ids[0], native_ids[0] + expected_count)):
        raise RuntimeContractError(
            "vLLM native request ids are not one contiguous call-local range",
            code="vllm_backend.native_request_id",
            context={"request_ids": native_ids},
        )
    return ordered


def _validate_image_placeholder_evidence(
    *,
    native_output: Any,
    prompt_ids: Sequence[int],
    image_pad_token_id: int,
    request_id: str,
) -> tuple[list[dict[str, int]], dict[str, list[dict[str, int]]]]:
    expected = _contiguous_token_ranges(
        prompt_ids,
        token_id=image_pad_token_id,
    )
    placeholders = getattr(native_output, "multi_modal_placeholders", None)
    if not isinstance(placeholders, Mapping):
        _native_failure(
            "vLLM returned invalid multimodal-placeholder evidence",
            code="vllm_backend.image_placeholder_evidence",
            request_id=request_id,
        )
    image_ranges = placeholders.get("image")
    if image_ranges is None and not placeholders:
        return expected, {}
    if not isinstance(image_ranges, Sequence):
        _native_failure(
            "vLLM returned no image-placeholder evidence",
            code="vllm_backend.image_placeholder_evidence",
            request_id=request_id,
        )
    observed: list[dict[str, int]] = []
    for item in image_ranges:
        if isinstance(item, Mapping):
            offset = item.get("offset")
            length = item.get("length")
        else:
            offset = getattr(item, "offset", None)
            length = getattr(item, "length", None)
        if (
            isinstance(offset, bool)
            or not isinstance(offset, int)
            or isinstance(length, bool)
            or not isinstance(length, int)
            or offset < 0
            or length <= 0
        ):
            _native_failure(
                "vLLM returned malformed image-placeholder evidence",
                code="vllm_backend.image_placeholder_evidence",
                request_id=request_id,
            )
        observed.append({"offset": offset, "length": length})
    if observed != expected:
        raise RuntimeContractError(
            "vLLM image-placeholder evidence disagrees with returned prompt ids",
            code="vllm_backend.image_placeholder_mismatch",
            context={
                "request_id": request_id,
                "expected": expected,
                "observed": observed,
            },
        )
    return observed, {"image": observed}


def _contiguous_token_ranges(
    token_ids: Sequence[int],
    *,
    token_id: int,
) -> list[dict[str, int]]:
    ranges: list[dict[str, int]] = []
    cursor = 0
    while cursor < len(token_ids):
        if token_ids[cursor] != token_id:
            cursor += 1
            continue
        offset = cursor
        while cursor < len(token_ids) and token_ids[cursor] == token_id:
            cursor += 1
        ranges.append({"offset": offset, "length": cursor - offset})
    if len(ranges) != 1:
        raise RuntimeContractError(
            "vLLM prompt must contain one contiguous image-placeholder range",
            code="vllm_backend.image_placeholder_evidence",
            context={"ranges": ranges},
        )
    return ranges


def _open_verified_rgb_image(request: DecodeRequest) -> Image.Image:
    path = Path(request.image_path)
    try:
        image_bytes = path.read_bytes()
    except OSError as exc:
        raise RuntimeContractError(
            "vLLM backend could not reopen request image",
            code="vllm_backend.image_read",
            context={"request_id": request.request_id, "image_path": str(path)},
            cause=exc,
        ) from exc
    observed_sha256 = hashlib.sha256(image_bytes).hexdigest()
    if observed_sha256 != request.image_sha256:
        raise RuntimeContractError(
            "request image bytes changed before vLLM native projection",
            code="vllm_backend.image_sha256_mismatch",
            context={"request_id": request.request_id},
        )
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            expected_size = (
                request.decoded_image_width,
                request.decoded_image_height,
            )
            if tuple(image.size) != expected_size:
                raise RuntimeContractError(
                    "request image dimensions changed before vLLM native projection",
                    code="vllm_backend.image_dimensions_mismatch",
                    context={"request_id": request.request_id},
                )
            return apply_logical_image_transform(
                image.convert("RGB"),
                request.logical_transform_id,
                example_id=request.request_id,
                image_path=path,
            )
    except RuntimeContractError:
        raise
    except OSError as exc:
        raise RuntimeContractError(
            "vLLM backend could not decode request image",
            code="vllm_backend.image_decode",
            context={"request_id": request.request_id},
            cause=exc,
        ) from exc


def _require_shared_generation_policy(requests: Sequence[DecodeRequest]) -> Any:
    policies = {request.generation_policy for request in requests}
    if len(policies) != 1:
        raise RuntimeContractError(
            "requests in one vLLM scheduler submission must share generation policy",
            code="vllm_backend.generation_policy_mismatch",
            context={"request_ids": [request.request_id for request in requests]},
        )
    policy = policies.pop()
    longest = max(len(request.expected_executed_prompt_token_ids) for request in requests)
    if longest + policy.max_new_tokens > VLLM_MAX_MODEL_LEN:
        raise RuntimeContractError(
            "vLLM prompt plus generation exceeds the qualified model length",
            code="vllm_backend.model_length",
            context={
                "prompt_tokens": longest,
                "max_new_tokens": policy.max_new_tokens,
                "max_model_len": VLLM_MAX_MODEL_LEN,
            },
        )
    return policy


def _vllm_options(launch: BackendLaunch) -> Mapping[str, object]:
    options = launch.backend_options.get("vllm")
    if not isinstance(options, Mapping):
        raise RuntimeContractError(
            "vLLM launch is missing its strict backend options",
            code="vllm_backend.launch_options",
        )
    value = options.get("gpu_memory_utilization")
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not 0 < value <= 1
    ):
        raise RuntimeContractError(
            "vLLM gpu_memory_utilization must be between zero and one",
            code="vllm_backend.launch_options",
        )
    return options


def _default_engine_factory(kwargs: Mapping[str, object]) -> Any:
    from vllm import LLM

    return LLM(**dict(kwargs))


def _configure_process_mode() -> None:
    observed = os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING")
    if observed is None:
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        return
    if observed != "0":
        raise RuntimeContractError(
            "vLLM worker requires in-process V1 engine mode",
            code="vllm_backend.process_mode",
            context={"VLLM_ENABLE_V1_MULTIPROCESSING": observed},
        )


def _load_processor_components(launch: BackendLaunch) -> Any:
    return load_qwen_components_from_options(
        QwenLoadOptions(
            base_model=launch.model_path,
            dtype=launch.model_dtype,
            attn_implementation="eager",
            patch_embed_linearization="disabled",
            load_model=False,
        )
    )


def _model_identity(launch: BackendLaunch) -> dict[str, object]:
    identity = dict(launch.execution_model_identity or {})
    return {
        "composition_key": identity.get("composition_key"),
        "snapshot_fingerprint": identity.get("snapshot_fingerprint"),
        "mode": identity.get("mode"),
        "model_path": launch.model_path,
    }


def _identity_dict(value: object, *, fallback: Mapping[str, object]) -> dict[str, object]:
    if value is not None:
        converter = getattr(value, "to_artifact_dict", None)
        if callable(converter):
            payload = converter()
            if isinstance(payload, Mapping) and payload:
                return dict(payload)
        if isinstance(value, Mapping) and value:
            return dict(value)
    return dict(fallback)


def _validate_rank_local_cuda() -> None:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeContractError(
            "vLLM backend requires torch CUDA runtime",
            code="vllm_backend.cuda_unavailable",
            cause=exc,
        ) from exc
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeContractError(
            "vLLM worker requires exactly one visible CUDA device",
            code="vllm_backend.cuda_binding",
            context={"device_count": torch.cuda.device_count()},
        )
    if torch.cuda.current_device() != 0:
        raise RuntimeContractError(
            "vLLM worker must bind its visible GPU as logical cuda:0",
            code="vllm_backend.cuda_binding",
        )


def _close_vllm_engine(engine: Any | None) -> None:
    if engine is not None:
        llm_engine = getattr(engine, "llm_engine", None)
        engine_core = getattr(llm_engine, "engine_core", None)
        shutdown = getattr(engine_core, "shutdown", None)
        if callable(shutdown):
            shutdown()
        if llm_engine is not None:
            llm_engine.engine_core = None
        try:
            engine.llm_engine = None
        except (AttributeError, TypeError):
            pass
    try:
        from vllm.distributed import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )

        destroy_model_parallel()
        destroy_distributed_environment()
    except ImportError:
        pass
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass


def _close_prompt_images(prompts: Sequence[Any]) -> None:
    for prompt in prompts:
        data = prompt.get("multi_modal_data") if isinstance(prompt, Mapping) else None
        image = data.get("image") if isinstance(data, Mapping) else None
        close = getattr(image, "close", None)
        if callable(close):
            close()


def _native_failure(
    message: str,
    *,
    code: str,
    request_id: str,
    context: Mapping[str, object] | None = None,
) -> None:
    raise RuntimeContractError(
        message,
        code=code,
        context={"request_id": request_id, **dict(context or {})},
    )
