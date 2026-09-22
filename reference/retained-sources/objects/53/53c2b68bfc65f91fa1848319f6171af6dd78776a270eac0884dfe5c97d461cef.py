"""Hugging Face backend-owned inference session and likelihood tracing."""

from __future__ import annotations

import gc
import hashlib
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata
from io import BytesIO
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, cast
from weakref import WeakKeyDictionary

import torch
from PIL import Image
from torch.nn import functional as F

from src.adapters.dora import load_inference_dora_adapter
from src.common.errors import RuntimeContractError
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
from src.qwen.special_token_embeddings import load_inference_embedding_delta


ComponentsLoader = Callable[[BackendLaunch], Any]
_POLICY_SCORE_STEP_CHUNK_SIZE = 32


@dataclass(frozen=True)
class TeacherForcedComparison:
    compared_steps: int
    max_absolute_difference: float
    atol: float
    rtol: float


@dataclass(frozen=True, eq=False)
class HFExactHistory:
    """Audit-safe identity for one session-owned exact token history."""

    request_id: str
    conditioning_token_ids: tuple[int, ...]
    conditioning_token_ids_sha256: str


@dataclass(frozen=True)
class HFChosenTokenEvidence:
    """Raw evidence for one caller-selected continuation token."""

    token_id: int
    raw_model_logprob: float
    candidate_vocab_rank: int


@dataclass(frozen=True)
class _HFExactHistoryState:
    request_id: str
    conditioning_token_ids: tuple[int, ...]
    conditioning_token_ids_sha256: str
    context: _HFExactHistoryContext


@dataclass(eq=False)
class _HFExactHistoryContext:
    native_inputs: Mapping[str, Any] | None


class HFBackendSession:
    backend = "hf"
    backend_mode = "generate"
    response_family = "hf"

    def __init__(
        self,
        *,
        launch: BackendLaunch,
        model: Any,
        processor: Any,
        tokenizer: Any,
        receipt: BackendSessionReceipt,
    ) -> None:
        receipt.validate_for_launch(launch)
        _configure_left_padding(processor=processor, tokenizer=tokenizer)
        self._launch = launch
        self._model = model
        self._processor = processor
        self._tokenizer = tokenizer
        self._receipt = receipt
        self._closed = False
        self._exact_history_states: WeakKeyDictionary[
            HFExactHistory,
            _HFExactHistoryState,
        ] = WeakKeyDictionary()

    @property
    def receipt(self) -> BackendSessionReceipt:
        return self._receipt

    @property
    def special_token_ids(self) -> Mapping[str, int]:
        self._require_live_session()
        return MappingProxyType(
            {
                "im_end": self._im_end_token_id(),
                "pad": self._pad_token_id(),
            }
        )

    def prepare_exact_history(self, request: DecodeRequest) -> HFExactHistory:
        """Materialize and retain one verified multimodal request."""

        self._require_live_session()
        (
            native_inputs,
            executed_prompt_ids,
            _observed_grids,
            _executed_media_sha256,
        ) = self._materialize_native_inputs((request,))
        conditioning_token_ids = self._validated_token_ids(
            executed_prompt_ids[0],
            field="conditioning_token_ids",
        )
        context = _HFExactHistoryContext(native_inputs=dict(native_inputs))
        return self._new_exact_history(
            request_id=request.request_id,
            conditioning_token_ids=conditioning_token_ids,
            context=context,
        )

    def extend_exact_history(
        self,
        history: HFExactHistory,
        token_ids: Sequence[int],
    ) -> HFExactHistory:
        """Append literal vocabulary IDs without decoding or re-tokenizing."""

        state = self._validated_exact_history(history)
        appended = self._validated_token_ids(token_ids, field="token_ids")
        return self._new_exact_history(
            request_id=state.request_id,
            conditioning_token_ids=(*state.conditioning_token_ids, *appended),
            context=state.context,
        )

    def teacher_forced_evidence(
        self,
        history: HFExactHistory,
        continuation_token_ids: Sequence[int],
    ) -> tuple[HFChosenTokenEvidence, ...]:
        """Observe raw FP32 evidence for one non-empty literal continuation."""

        state = self._validated_exact_history(history)
        continuation = self._validated_token_ids(
            continuation_token_ids,
            field="continuation_token_ids",
        )
        if not continuation:
            raise RuntimeContractError(
                "teacher-forced evidence requires a non-empty continuation",
                code="hf_backend.teacher_forced_empty",
            )
        context = state.context
        if context.native_inputs is None:
            raise RuntimeContractError(
                "HF exact history no longer has live multimodal context",
                code="hf_backend.exact_history_session",
                context={"request_id": state.request_id},
            )
        model = self._model
        if model is None:
            raise RuntimeContractError(
                "HF exact-history evidence has no live model",
                code="hf_backend.session_closed",
            )
        full_token_ids = (*state.conditioning_token_ids, *continuation)
        device = _model_device(model)
        input_ids = torch.tensor(
            [full_token_ids],
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        image_grid_thw = context.native_inputs.get("image_grid_thw")
        if not isinstance(image_grid_thw, torch.Tensor):
            raise RuntimeContractError(
                "HF exact-history evidence requires image_grid_thw",
                code="hf_backend.exact_history_image_grid",
                context={"request_id": state.request_id},
            )
        position_ids = _derive_qwen_position_ids(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_grid_thw=image_grid_thw.to(device=device),
            video_grid_thw=_tensor_to_device_or_none(
                context.native_inputs.get("video_grid_thw"),
                device=device,
            ),
        )
        forward_inputs = {
            key: value
            for key, value in context.native_inputs.items()
            if key
            not in {
                "input_ids",
                "attention_mask",
                "position_ids",
                "token_type_ids",
                "cache_position",
                "rope_deltas",
            }
        }
        forward_inputs.update(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "use_cache": False,
                "return_dict": True,
                "logits_to_keep": 0,
            }
        )
        with torch.inference_mode():
            outputs = model(**forward_inputs)
        logits = _require_rank_three_tensor(
            getattr(outputs, "logits", None),
            field="logits",
        )
        boundary = len(state.conditioning_token_ids) - 1
        selected_logits = logits[
            0,
            boundary : boundary + len(continuation),
            :,
        ].detach().to(device="cpu", dtype=torch.float32).contiguous()
        if selected_logits.shape[0] != len(continuation):
            raise RuntimeContractError(
                "teacher-forced logits do not cover every continuation token",
                code="hf_backend.teacher_forced_alignment",
                context={
                    "continuation_steps": len(continuation),
                    "logit_steps": int(selected_logits.shape[0]),
                },
            )
        selected_ids = torch.tensor(
            continuation,
            dtype=torch.long,
            device=selected_logits.device,
        )
        selected_values = selected_logits.gather(1, selected_ids.unsqueeze(1))
        logprobs = F.log_softmax(selected_logits, dim=-1).gather(
            1,
            selected_ids.unsqueeze(1),
        )
        ranks = (selected_logits > selected_values).sum(dim=1) + 1
        return tuple(
            HFChosenTokenEvidence(
                token_id=token_id,
                raw_model_logprob=float(logprobs[index, 0].item()),
                candidate_vocab_rank=int(ranks[index].item()),
            )
            for index, token_id in enumerate(continuation)
        )

    def decode(self, requests: Sequence[DecodeRequest]) -> tuple[DecodeResult, ...]:
        if self._closed:
            raise RuntimeContractError(
                "HF backend session is already closed",
                code="hf_backend.session_closed",
            )
        checked = tuple(requests)
        if not checked:
            return ()
        request_ids = [request.request_id for request in checked]
        if len(set(request_ids)) != len(request_ids):
            raise RuntimeContractError(
                "HF backend session requires unique request ids",
                code="hf_backend.duplicate_request_id",
                context={"request_ids": request_ids},
            )
        synchronize_cuda_for_timing(torch)
        started_at = time.perf_counter()
        results: list[DecodeResult] = []
        for native_batch in _native_batch_groups(
            checked,
            batch_size=self._launch.batch_size,
        ):
            results.extend(self._decode_native_batch(native_batch))
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
        seen_contexts: set[int] = set()
        for state in tuple(self._exact_history_states.values()):
            context_identity = id(state.context)
            if context_identity in seen_contexts:
                continue
            seen_contexts.add(context_identity)
            state.context.native_inputs = None
        self._exact_history_states.clear()
        self._model = None
        self._processor = None
        self._tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _decode_native_batch(
        self,
        requests: Sequence[DecodeRequest],
    ) -> tuple[DecodeResult, ...]:
        policy = _require_shared_generation_policy(requests)
        (
            native_inputs,
            executed_prompt_ids,
            observed_grids,
            executed_media_sha256,
        ) = self._materialize_native_inputs(requests)
        prompt_width = int(native_inputs["input_ids"].shape[1])
        generate_kwargs: dict[str, Any] = {
            **native_inputs,
            "max_new_tokens": policy.max_new_tokens,
            "repetition_penalty": policy.repetition_penalty,
            "eos_token_id": self._im_end_token_id(),
            "pad_token_id": self._pad_token_id(),
            "do_sample": False,
            "return_dict_in_generate": True,
            "output_scores": True,
        }
        if policy.include_raw_model_logprob:
            generate_kwargs["output_logits"] = True
        with torch.inference_mode():
            outputs = self._model.generate(**generate_kwargs)
        scores = _require_step_tensors(
            getattr(outputs, "scores", None),
            batch_size=len(requests),
            field="scores",
        )
        raw_logits = (
            _require_step_tensors(
                getattr(outputs, "logits", None),
                batch_size=len(requests),
                field="logits",
            )
            if policy.include_raw_model_logprob
            else None
        )
        if raw_logits is not None and len(raw_logits) != len(scores):
            raise RuntimeContractError(
                "HF raw logits and policy scores have different step counts",
                code="hf_backend.raw_logit_alignment",
                context={
                    "score_steps": len(scores),
                    "raw_logit_steps": len(raw_logits),
                },
            )
        sequences = _require_sequences(
            getattr(outputs, "sequences", None),
            batch_size=len(requests),
            expected_width=prompt_width + len(scores),
        )
        generated = sequences[:, prompt_width:]
        policy_logprobs = _policy_chosen_token_logprobs(
            model=self._model,
            sequences=sequences,
            scores=scores,
            generated=generated,
        )
        raw_logprobs = (
            _chosen_token_logprobs(raw_logits, generated)
            if raw_logits is not None
            else None
        )
        return tuple(
            self._materialize_result(
                request=request,
                generated_ids=tuple(int(value) for value in generated[row].tolist()),
                policy_logprobs=tuple(
                    float(value) for value in policy_logprobs[row].tolist()
                ),
                raw_logprobs=(
                    None
                    if raw_logprobs is None
                    else tuple(float(value) for value in raw_logprobs[row].tolist())
                ),
                executed_prompt_ids=executed_prompt_ids[row],
                observed_image_grid_thw=observed_grids[row],
                executed_media_sha256=executed_media_sha256[row],
            )
            for row, request in enumerate(requests)
        )

    def _materialize_native_inputs(
        self,
        requests: Sequence[DecodeRequest],
    ) -> tuple[
        dict[str, Any],
        tuple[tuple[int, ...], ...],
        tuple[tuple[int, int, int] | None, ...],
        tuple[str, ...],
    ]:
        images = [_open_verified_rgb_image(request) for request in requests]
        executed_media_sha256 = tuple(rgb_image_sha256(image) for image in images)
        try:
            encoded = self._processor(
                text=[request.chat_text for request in requests],
                images=images,
                padding=True,
                return_tensors="pt",
                do_resize=False,
            )
        finally:
            for image in images:
                image.close()
        if not isinstance(encoded, Mapping):
            try:
                encoded = dict(encoded)
            except (TypeError, ValueError) as exc:
                raise RuntimeContractError(
                    "HF processor did not return mapping-like native inputs",
                    code="hf_backend.processor_output",
                    context={"value_type": type(encoded).__name__},
                    cause=exc,
                ) from exc
        native_inputs = dict(encoded)
        input_ids = _require_rank_two_tensor(
            native_inputs.get("input_ids"), field="input_ids"
        )
        if input_ids.shape[0] != len(requests):
            raise RuntimeContractError(
                "HF processor input_ids batch dimension does not match requests",
                code="hf_backend.processor_batch_shape",
                context={"batch_size": len(requests), "shape": tuple(input_ids.shape)},
            )
        attention_mask = native_inputs.get("attention_mask")
        executed_prompt_ids = _unpadded_prompt_rows(input_ids, attention_mask)
        for request, observed in zip(requests, executed_prompt_ids, strict=True):
            if observed != request.expected_executed_prompt_token_ids:
                raise RuntimeContractError(
                    "HF processor executed prompt ids differ from expected expansion",
                    code="hf_backend.prompt_token_mismatch",
                    context={
                        "request_id": request.request_id,
                        "expected_count": len(
                            request.expected_executed_prompt_token_ids
                        ),
                        "observed_count": len(observed),
                    },
                )
        observed_grids = _observed_image_grids(
            native_inputs.get("image_grid_thw"),
            batch_size=len(requests),
        )
        for request, observed in zip(requests, observed_grids, strict=True):
            if request.expected_image_grid_thw is not None and (
                observed != request.expected_image_grid_thw
            ):
                raise RuntimeContractError(
                    "HF processor image grid differs from expected no-resize plan",
                    code="hf_backend.image_grid_mismatch",
                    context={
                        "request_id": request.request_id,
                        "expected": list(request.expected_image_grid_thw),
                        "observed": None if observed is None else list(observed),
                    },
                )
        device = _model_device(self._model)
        return (
            {
                key: _move_to_device(value, device=device)
                for key, value in native_inputs.items()
            },
            executed_prompt_ids,
            observed_grids,
            executed_media_sha256,
        )

    def _materialize_result(
        self,
        *,
        request: DecodeRequest,
        generated_ids: tuple[int, ...],
        policy_logprobs: tuple[float, ...],
        raw_logprobs: tuple[float, ...] | None,
        executed_prompt_ids: tuple[int, ...],
        observed_image_grid_thw: tuple[int, int, int] | None,
        executed_media_sha256: str,
    ) -> DecodeResult:
        stop_id = self._im_end_token_id()
        pad_id = self._pad_token_id()
        kept_ids: list[int] = []
        traces: list[TokenTrace] = []
        seen_stop = False
        stop_reason = "length"
        for step_index, token_id in enumerate(generated_ids):
            if seen_stop and token_id != pad_id:
                raise RuntimeContractError(
                    "HF generation emitted non-padding content after a stop token",
                    code="hf_backend.post_stop_content",
                    context={
                        "request_id": request.request_id,
                        "step_index": step_index,
                    },
                )
            if token_id == pad_id and not seen_stop:
                raise RuntimeContractError(
                    "HF generation emitted a pad token before a stop token",
                    code="hf_backend.unexpected_pad_token",
                    context={
                        "request_id": request.request_id,
                        "step_index": step_index,
                    },
                )
            is_pad = seen_stop and token_id == pad_id
            is_stop = not seen_stop and token_id == stop_id
            traces.append(
                TokenTrace(
                    step_index=step_index,
                    token_id=token_id,
                    token_text=self._decode_tokens((token_id,)),
                    likelihood=LikelihoodPair(
                        policy_logprob=None if is_pad else policy_logprobs[step_index],
                        raw_model_logprob=(
                            None
                            if is_pad or raw_logprobs is None
                            else raw_logprobs[step_index]
                        ),
                    ),
                    is_stop=is_stop,
                    is_pad=is_pad,
                    backend=self.backend,
                    backend_mode=self.backend_mode,
                    response_family=self.response_family,
                )
            )
            if is_pad:
                continue
            kept_ids.append(token_id)
            if is_stop:
                seen_stop = True
                stop_reason = "im_end"
        raw_text = self._decode_tokens(tuple(kept_ids))
        stop_text = self._decode_tokens((stop_id,))
        if kept_ids and kept_ids[-1] == stop_id and raw_text.endswith(stop_text):
            parser_text = raw_text[: -len(stop_text)]
            strip_policy = "terminal_im_end"
        else:
            parser_text = raw_text
            strip_policy = "none"
        return DecodeResult(
            request_id=request.request_id,
            backend=self.backend,
            backend_mode=self.backend_mode,
            response_family=self.response_family,
            executed_prompt_token_ids=executed_prompt_ids,
            generated_token_ids=tuple(kept_ids),
            raw_generated_text=raw_text,
            parser_text=parser_text,
            strip_policy=strip_policy,
            stop_reason=stop_reason,
            token_trace=tuple(traces),
            observed_image_grid_thw=observed_image_grid_thw,
            executed_media_sha256=executed_media_sha256,
            native_generated_text=raw_text,
        )

    def _im_end_token_id(self) -> int:
        convert = getattr(self._tokenizer, "convert_tokens_to_ids", None)
        if callable(convert):
            token_id = convert("<|im_end|>")
            if token_id is not None:
                return int(token_id)
        eos_token_id = getattr(self._tokenizer, "eos_token_id", None)
        if eos_token_id is None:
            raise RuntimeContractError(
                "HF tokenizer does not expose the Qwen im_end token",
                code="hf_backend.stop_token_missing",
            )
        return int(eos_token_id)

    def _pad_token_id(self) -> int:
        pad_token_id = getattr(self._tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            raise RuntimeContractError(
                "HF tokenizer does not expose a pad token",
                code="hf_backend.pad_token_missing",
            )
        return int(pad_token_id)

    def _decode_tokens(self, token_ids: tuple[int, ...]) -> str:
        if not token_ids:
            return ""
        return str(self._tokenizer.decode(list(token_ids), skip_special_tokens=False))

    def _require_live_session(self) -> None:
        if self._closed:
            raise RuntimeContractError(
                "HF backend session is already closed",
                code="hf_backend.session_closed",
            )

    def _validated_token_ids(
        self,
        token_ids: Sequence[int],
        *,
        field: str,
    ) -> tuple[int, ...]:
        vocab_size = _model_vocabulary_size(self._model, self._tokenizer)
        if vocab_size is None:
            raise RuntimeContractError(
                "HF exact-history token validation cannot establish vocabulary size",
                code="hf_backend.vocab_size_unavailable",
                context={"field": field},
            )
        validated: list[int] = []
        for index, value in enumerate(token_ids):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                or (vocab_size is not None and value >= vocab_size)
            ):
                raise RuntimeContractError(
                    "HF exact-history token ID is outside the model vocabulary",
                    code="hf_backend.invalid_token_id",
                    context={
                        "field": field,
                        "index": index,
                        "value": value,
                        "vocab_size": vocab_size,
                    },
                )
            validated.append(int(value))
        return tuple(validated)

    def _new_exact_history(
        self,
        *,
        request_id: str,
        conditioning_token_ids: tuple[int, ...],
        context: _HFExactHistoryContext,
    ) -> HFExactHistory:
        digest = token_ids_sha256(conditioning_token_ids)
        history = HFExactHistory(
            request_id=request_id,
            conditioning_token_ids=conditioning_token_ids,
            conditioning_token_ids_sha256=digest,
        )
        self._exact_history_states[history] = _HFExactHistoryState(
            request_id=request_id,
            conditioning_token_ids=conditioning_token_ids,
            conditioning_token_ids_sha256=digest,
            context=context,
        )
        return history

    def _validated_exact_history(
        self,
        history: HFExactHistory,
    ) -> _HFExactHistoryState:
        self._require_live_session()
        if not isinstance(history, HFExactHistory):
            raise RuntimeContractError(
                "HF exact history was not created by this live session",
                code="hf_backend.exact_history_session",
            )
        state = self._exact_history_states.get(history)
        if state is None or (
            history.request_id != state.request_id
            or history.conditioning_token_ids != state.conditioning_token_ids
            or history.conditioning_token_ids_sha256
            != state.conditioning_token_ids_sha256
            or token_ids_sha256(history.conditioning_token_ids)
            != state.conditioning_token_ids_sha256
        ):
            raise RuntimeContractError(
                "HF exact history identity does not match live session state",
                code="hf_backend.exact_history_session",
                context={"request_id": getattr(history, "request_id", None)},
            )
        return state


def open_hf_backend_session(
    launch: BackendLaunch,
    *,
    components_loader: ComponentsLoader | None = None,
) -> HFBackendSession:
    """Load the actual HF runtime and return its backend-owned session."""

    if launch.backend != "hf":
        raise RuntimeContractError(
            "HF session opener received a non-HF launch",
            code="hf_backend.launch_backend",
            context={"backend": launch.backend},
        )
    loaded = (components_loader or _load_hf_components)(launch)
    qwen = loaded.qwen
    model = qwen.model
    if model is None:
        raise RuntimeContractError(
            "HF session loading did not produce a model",
            code="hf_backend.model_missing",
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    model_identity = _actual_model_identity(launch, loaded)
    tokenizer_identity = _identity_dict(
        getattr(qwen, "token_identity", None),
        fallback={"tokenizer_sha256": str(getattr(qwen, "tokenizer_sha256", ""))},
    )
    processor_identity = _identity_dict(
        getattr(qwen, "processor_identity", None),
        fallback={"processor_class": type(qwen.processor).__name__},
    )
    receipt = BackendSessionReceipt(
        backend="hf",
        backend_mode=HFBackendSession.backend_mode,
        response_family=HFBackendSession.response_family,
        backend_version=metadata.version("transformers"),
        model_identity=model_identity,
        tokenizer_identity=tokenizer_identity,
        processor_identity=processor_identity,
        generation_config_fingerprint=launch.generation_config_fingerprint,
        effective_settings={
            "batch_size": launch.batch_size,
            "device": str(device),
            "backend_options": dict(launch.backend_options),
            "text_padding_side": "left",
            "output_scores": True,
            "raw_output_logits": "per_request",
            "observed_model_dtype": _observed_model_dtype(model),
            "observed_attn_implementation": _observed_attn_implementation(model),
        },
        likelihood_semantics={
            "policy": POLICY_LIKELIHOOD_DEFINITION,
            "raw": RAW_LIKELIHOOD_DEFINITION,
            "score_owned_channel": "policy_logprob",
        },
        execution_model_identity=launch.execution_model_identity,
    )
    return HFBackendSession(
        launch=launch,
        model=model,
        processor=qwen.processor,
        tokenizer=qwen.tokenizer,
        receipt=receipt,
    )


_QWEN_ROPE_OWNER_MAX_MODEL_DEPTH = 8


def _resolve_qwen_get_rope_index(model: Any) -> Callable[..., tuple[Any, Any]]:
    """Bind the real Qwen ``get_rope_index`` beneath wrapper ``.model`` levels.

    Adapter wrappers (for example PEFT) and the transformers Qwen3-VL layout
    interpose a session-dependent number of ``.model`` levels above the module
    that owns exact multimodal MRoPE derivation; the owner is located, never
    reimplemented.
    """

    candidate = model
    searched: list[str] = []
    seen_ids: set[int] = set()
    while (
        candidate is not None
        and id(candidate) not in seen_ids
        and len(searched) < _QWEN_ROPE_OWNER_MAX_MODEL_DEPTH
    ):
        seen_ids.add(id(candidate))
        searched.append(type(candidate).__name__)
        get_rope_index_value = getattr(candidate, "get_rope_index", None)
        if callable(get_rope_index_value):
            return cast(Callable[..., tuple[Any, Any]], get_rope_index_value)
        candidate = getattr(candidate, "model", None)
    raise RuntimeContractError(
        "HF model does not expose Qwen get_rope_index",
        code="hf_backend.position_ids_unavailable",
        context={"searched_model_chain": searched},
    )


def _derive_qwen_position_ids(
    *,
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    image_grid_thw: torch.Tensor,
    video_grid_thw: torch.Tensor | None,
) -> torch.Tensor:
    get_rope_index = _resolve_qwen_get_rope_index(model)
    try:
        with torch.inference_mode():
            position_ids, _rope_deltas = get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                attention_mask=attention_mask,
            )
    except (RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeContractError(
            "HF model failed to derive Qwen position IDs",
            code="hf_backend.position_ids_invalid",
            cause=exc,
        ) from exc
    if not isinstance(position_ids, torch.Tensor) or position_ids.ndim != 3:
        raise RuntimeContractError(
            "Qwen get_rope_index returned invalid position IDs",
            code="hf_backend.position_ids_invalid",
            context={
                "value_type": type(position_ids).__name__,
                "shape": (
                    tuple(position_ids.shape)
                    if isinstance(position_ids, torch.Tensor)
                    else None
                ),
            },
        )
    return position_ids


def _tensor_to_device_or_none(value: Any, *, device: torch.device) -> torch.Tensor | None:
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        raise RuntimeContractError(
            "HF native grid input is not a tensor",
            code="hf_backend.exact_history_image_grid",
            context={"value_type": type(value).__name__},
        )
    return value.to(device=device)


def _model_vocabulary_size(model: Any, tokenizer: Any) -> int | None:
    for method_name in ("get_output_embeddings", "get_input_embeddings"):
        method = getattr(model, method_name, None)
        if not callable(method):
            continue
        try:
            embedding = method()
        except (AttributeError, RuntimeError, TypeError, ValueError):
            continue
        weight = getattr(embedding, "weight", None)
        shape = getattr(weight, "shape", ())
        if shape and isinstance(shape[0], int) and shape[0] > 0:
            return int(shape[0])
    try:
        tokenizer_length = len(tokenizer)
    except (AttributeError, TypeError):
        tokenizer_length = None
    if (
        isinstance(tokenizer_length, int)
        and not isinstance(tokenizer_length, bool)
        and tokenizer_length > 0
    ):
        return int(tokenizer_length)
    for value in (
        getattr(tokenizer, "vocab_size", None),
        getattr(getattr(model, "config", None), "vocab_size", None),
    ):
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return int(value)
    return None


def _observed_model_dtype(model: Any) -> dict[str, object] | None:
    parameters_value = getattr(model, "parameters", None)
    if not callable(parameters_value):
        return None
    parameters = cast(Callable[[], Iterable[Any]], parameters_value)
    counter: Counter[str] = Counter()
    try:
        for parameter in parameters():
            dtype = getattr(parameter, "dtype", None)
            numel_value = getattr(parameter, "numel", None)
            if dtype is None or not callable(numel_value):
                return None
            numel = cast(Callable[[], int], numel_value)
            counter[str(dtype)] += int(numel())
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None
    if not counter:
        return None
    return {
        "parameter_dtype_counts": dict(sorted(counter.items())),
        "parameter_dtype_names": sorted(counter),
    }


def _observed_attn_implementation(model: Any) -> str | None:
    value = getattr(getattr(model, "config", None), "_attn_implementation", None)
    if value is None or value == "":
        return None
    return str(value)


def teacher_forced_chosen_token_logprobs(
    *,
    model: Any,
    native_prompt_inputs: Mapping[str, Any],
    generated_token_ids: Sequence[int],
) -> torch.Tensor:
    """Compute an independent FP32 chosen-token reference from one full forward."""

    input_ids = _require_rank_two_tensor(
        native_prompt_inputs.get("input_ids"),
        field="input_ids",
    )
    if input_ids.shape[0] != 1:
        raise RuntimeContractError(
            "teacher-forced comparison helper accepts exactly one prompt",
            code="hf_backend.teacher_forced_batch",
            context={"shape": tuple(input_ids.shape)},
        )
    generated = torch.tensor(
        [int(token_id) for token_id in generated_token_ids],
        dtype=input_ids.dtype,
        device=input_ids.device,
    ).unsqueeze(0)
    if generated.shape[1] == 0:
        raise RuntimeContractError(
            "teacher-forced comparison requires generated token ids",
            code="hf_backend.teacher_forced_empty",
        )
    full_input_ids = torch.cat((input_ids, generated), dim=1)
    forward_inputs = dict(native_prompt_inputs)
    forward_inputs["input_ids"] = full_input_ids
    attention_mask = forward_inputs.get("attention_mask")
    if isinstance(attention_mask, torch.Tensor):
        extension = torch.ones(
            (1, generated.shape[1]),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        forward_inputs["attention_mask"] = torch.cat((attention_mask, extension), dim=1)
    with torch.inference_mode():
        outputs = model(**forward_inputs, return_dict=True, use_cache=False)
    logits = _require_rank_three_tensor(
        getattr(outputs, "logits", None), field="logits"
    )
    prompt_width = input_ids.shape[1]
    prediction_logits = logits[
        :, prompt_width - 1 : prompt_width - 1 + generated.shape[1], :
    ]
    if prediction_logits.shape[1] != generated.shape[1]:
        raise RuntimeContractError(
            "teacher-forced logits do not cover every generated token",
            code="hf_backend.teacher_forced_alignment",
            context={
                "generated_steps": int(generated.shape[1]),
                "logit_steps": int(prediction_logits.shape[1]),
            },
        )
    return (
        F.log_softmax(prediction_logits.float(), dim=-1)
        .gather(
            2,
            generated.unsqueeze(-1),
        )
        .squeeze(0)
        .squeeze(-1)
    )


def compare_raw_generation_to_teacher_forced(
    generation_raw_logprobs: Sequence[float] | torch.Tensor,
    teacher_forced_logprobs: Sequence[float] | torch.Tensor,
    *,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> TeacherForcedComparison:
    generated = (
        torch.as_tensor(generation_raw_logprobs)
        .detach()
        .to(device="cpu", dtype=torch.float32)
    )
    reference = (
        torch.as_tensor(teacher_forced_logprobs)
        .detach()
        .to(device="cpu", dtype=torch.float32)
    )
    if (
        generated.shape != reference.shape
        or generated.ndim != 1
        or generated.numel() == 0
    ):
        raise RuntimeContractError(
            "raw generation and teacher-forced likelihood shapes must match",
            code="hf_backend.teacher_forced_shape",
            context={
                "generation_shape": tuple(generated.shape),
                "reference_shape": tuple(reference.shape),
            },
        )
    max_difference = float(torch.max(torch.abs(generated - reference)).item())
    if not torch.allclose(generated, reference, atol=atol, rtol=rtol):
        raise RuntimeContractError(
            "raw generation likelihood differs from teacher-forced FP32 reference",
            code="hf_backend.teacher_forced_mismatch",
            context={
                "compared_steps": int(generated.numel()),
                "max_absolute_difference": max_difference,
                "atol": atol,
                "rtol": rtol,
            },
        )
    return TeacherForcedComparison(
        compared_steps=int(generated.numel()),
        max_absolute_difference=max_difference,
        atol=atol,
        rtol=rtol,
    )


def _load_hf_components(launch: BackendLaunch) -> Any:
    options = _hf_options(launch)
    qwen = load_qwen_components_from_options(
        QwenLoadOptions(
            base_model=launch.model_path,
            dtype=launch.model_dtype,
            attn_implementation=str(options["attn_implementation"]),
            patch_embed_linearization=str(options["patch_embed_linearization"]),
            load_model=True,
        )
    )
    adapter = _namespace_or_none(launch.adapter)
    embedding_delta = _namespace_or_none(launch.embedding_delta)
    config = SimpleNamespace(adapter=adapter, embedding_delta=embedding_delta)
    adapter_receipt = (
        load_inference_dora_adapter(config=config, qwen=qwen)
        if adapter is not None
        else None
    )
    embedding_delta_receipt = (
        load_inference_embedding_delta(config=config, qwen=qwen)
        if embedding_delta is not None
        else None
    )
    return SimpleNamespace(
        qwen=qwen,
        adapter_receipt=adapter_receipt,
        embedding_delta_receipt=embedding_delta_receipt,
    )


def _hf_options(launch: BackendLaunch) -> Mapping[str, object]:
    nested = launch.backend_options.get("hf")
    options = nested if isinstance(nested, Mapping) else launch.backend_options
    missing = [
        field
        for field in ("attn_implementation", "patch_embed_linearization")
        if field not in options
    ]
    if missing:
        raise RuntimeContractError(
            "HF backend launch is missing required execution options",
            code="hf_backend.launch_options",
            context={"missing": missing},
        )
    if options["attn_implementation"] not in {
        "flash_attention_2",
        "sdpa",
        "eager",
    } or options["patch_embed_linearization"] not in {"enabled", "disabled"}:
        raise RuntimeContractError(
            "HF backend launch contains unsupported execution options",
            code="hf_backend.launch_options",
            context={
                "attn_implementation": options["attn_implementation"],
                "patch_embed_linearization": options["patch_embed_linearization"],
            },
        )
    return options


def _actual_model_identity(launch: BackendLaunch, loaded: Any) -> dict[str, object]:
    if (
        loaded.adapter_receipt is not None
        and loaded.embedding_delta_receipt is not None
    ):
        family = "base-plus-adapter-plus-delta"
    elif loaded.adapter_receipt is not None:
        family = "base-plus-adapter"
    elif loaded.embedding_delta_receipt is not None:
        family = "base-plus-delta"
    else:
        family = "base-only"
    qwen_identity = _identity_dict(
        getattr(loaded.qwen, "model_identity", None),
        fallback={"model_class": type(loaded.qwen.model).__name__},
    )
    return {
        "family": family,
        "base": {"path": str(Path(launch.model_path).expanduser().resolve())},
        "qwen": qwen_identity,
        "adapter": loaded.adapter_receipt,
        "embedding_delta": loaded.embedding_delta_receipt,
    }


def _identity_dict(value: Any, *, fallback: Mapping[str, object]) -> dict[str, object]:
    if value is not None and callable(getattr(value, "to_artifact_dict", None)):
        return dict(value.to_artifact_dict())
    return dict(fallback)


def _configure_left_padding(*, processor: Any, tokenizer: Any) -> None:
    """Keep heterogeneous decoder-only batches aligned with the legacy HF path."""

    candidates = (tokenizer, getattr(processor, "tokenizer", None))
    seen: set[int] = set()
    for candidate in candidates:
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        try:
            candidate.padding_side = "left"
        except (AttributeError, TypeError) as exc:
            raise RuntimeContractError(
                "HF tokenizer does not permit decoder-only left padding",
                code="hf_backend.padding_side_unsupported",
                context={"tokenizer_class": type(candidate).__name__},
                cause=exc,
            ) from exc
        if getattr(candidate, "padding_side", None) != "left":
            raise RuntimeContractError(
                "HF tokenizer did not retain decoder-only left padding",
                code="hf_backend.padding_side_unsupported",
                context={"tokenizer_class": type(candidate).__name__},
            )


def _namespace_or_none(value: object) -> SimpleNamespace | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise RuntimeContractError(
            "HF adapter and embedding-delta launch options must be mappings",
            code="hf_backend.launch_options",
            context={"value_type": type(value).__name__},
        )
    return SimpleNamespace(**dict(value))


def _open_verified_rgb_image(request: DecodeRequest) -> Image.Image:
    path = Path(request.image_path)
    try:
        image_bytes = path.read_bytes()
    except OSError as exc:
        raise RuntimeContractError(
            "HF backend could not reopen request image",
            code="hf_backend.image_read",
            context={"request_id": request.request_id, "image_path": str(path)},
            cause=exc,
        ) from exc
    observed_sha256 = hashlib.sha256(image_bytes).hexdigest()
    if observed_sha256 != request.image_sha256:
        raise RuntimeContractError(
            "request image bytes changed before HF native projection",
            code="hf_backend.image_sha256_mismatch",
            context={
                "request_id": request.request_id,
                "expected_sha256": request.image_sha256,
                "observed_sha256": observed_sha256,
            },
        )
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            observed_size = tuple(int(value) for value in image.size)
            expected_size = (request.decoded_image_width, request.decoded_image_height)
            if observed_size != expected_size:
                raise RuntimeContractError(
                    "request image dimensions changed before HF native projection",
                    code="hf_backend.image_dimensions_mismatch",
                    context={
                        "request_id": request.request_id,
                        "expected": list(expected_size),
                        "observed": list(observed_size),
                    },
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
            "HF backend could not decode request image",
            code="hf_backend.image_decode",
            context={"request_id": request.request_id, "image_path": str(path)},
            cause=exc,
        ) from exc


def _require_shared_generation_policy(requests: Sequence[DecodeRequest]) -> Any:
    policies = {request.generation_policy for request in requests}
    if len(policies) != 1:
        raise RuntimeContractError(
            "requests in one HF native batch must share generation policy",
            code="hf_backend.generation_policy_mismatch",
            context={"request_ids": [request.request_id for request in requests]},
        )
    return policies.pop()


def _native_batch_groups(
    requests: Sequence[DecodeRequest],
    *,
    batch_size: int,
) -> tuple[tuple[DecodeRequest, ...], ...]:
    """Keep repetition-penalty conditioning independent of batch-only pads."""

    groups: list[tuple[DecodeRequest, ...]] = []
    for offset in range(0, len(requests), batch_size):
        chunk = tuple(requests[offset : offset + batch_size])
        policy = _require_shared_generation_policy(chunk)
        if policy.repetition_penalty == 1.0:
            groups.append(chunk)
            continue
        by_prompt_width: dict[int, list[DecodeRequest]] = {}
        for request in chunk:
            width = len(request.expected_executed_prompt_token_ids)
            by_prompt_width.setdefault(width, []).append(request)
        groups.extend(tuple(group) for group in by_prompt_width.values())
    return tuple(groups)


def _require_step_tensors(
    values: Any,
    *,
    batch_size: int,
    field: str,
) -> tuple[torch.Tensor, ...]:
    if values is None:
        raise RuntimeContractError(
            f"HF generation returned no {field}",
            code=f"hf_backend.missing_{field}",
        )
    tensors = tuple(torch.as_tensor(value) for value in values)
    if not tensors:
        raise RuntimeContractError(
            f"HF generation returned empty {field}",
            code=f"hf_backend.missing_{field}",
        )
    for step_index, tensor in enumerate(tensors):
        if tensor.ndim != 2 or tensor.shape[0] != batch_size:
            raise RuntimeContractError(
                f"HF generation {field} shape does not match the native batch",
                code="hf_backend.step_shape",
                context={
                    "field": field,
                    "step_index": step_index,
                    "shape": tuple(tensor.shape),
                    "batch_size": batch_size,
                },
            )
    return tensors


def _require_sequences(
    value: Any, *, batch_size: int, expected_width: int
) -> torch.Tensor:
    sequences = _require_rank_two_tensor(value, field="sequences")
    if sequences.shape != (batch_size, expected_width):
        raise RuntimeContractError(
            "HF generation sequence shape does not align with prompt and score steps",
            code="hf_backend.sequence_shape",
            context={
                "shape": tuple(sequences.shape),
                "expected_shape": (batch_size, expected_width),
            },
        )
    return sequences


def _chosen_token_logprobs(
    step_logits: Sequence[torch.Tensor] | None,
    generated: torch.Tensor,
) -> torch.Tensor:
    if step_logits is None:
        raise AssertionError("step logits are required")
    columns = []
    for step_index, logits in enumerate(step_logits):
        chosen = generated[:, step_index : step_index + 1].to(logits.device)
        columns.append(F.log_softmax(logits.float(), dim=-1).gather(1, chosen))
    return torch.cat(columns, dim=1)


def _policy_chosen_token_logprobs(
    *,
    model: Any,
    sequences: torch.Tensor,
    scores: Sequence[torch.Tensor],
    generated: torch.Tensor,
) -> torch.Tensor:
    compute_transition_scores = getattr(model, "compute_transition_scores", None)
    if callable(compute_transition_scores):
        prompt_width = sequences.shape[1] - len(scores)
        chunks: list[torch.Tensor] = []
        for start in range(0, len(scores), _POLICY_SCORE_STEP_CHUNK_SIZE):
            end = min(start + _POLICY_SCORE_STEP_CHUNK_SIZE, len(scores))
            chunk_scores = tuple(scores[start:end])
            chunk_sequences = sequences[:, : prompt_width + end]
            try:
                values = compute_transition_scores(
                    chunk_sequences,
                    chunk_scores,
                    normalize_logits=True,
                )
            except Exception as exc:
                raise RuntimeContractError(
                    "HF policy likelihood extraction failed",
                    code="hf_backend.policy_logprob_extraction",
                    context={
                        "sequence_shape": tuple(sequences.shape),
                        "score_steps": len(scores),
                        "chunk_start": start,
                        "chunk_end": end,
                    },
                    cause=exc,
                ) from exc
            chunk = torch.as_tensor(values)
            expected_chunk_shape = (generated.shape[0], end - start)
            if chunk.shape != expected_chunk_shape:
                raise RuntimeContractError(
                    "HF policy likelihood chunk does not align with generated ids",
                    code="hf_backend.policy_logprob_alignment",
                    context={
                        "likelihood_shape": tuple(chunk.shape),
                        "expected_shape": expected_chunk_shape,
                        "chunk_start": start,
                        "chunk_end": end,
                    },
                )
            chunks.append(chunk)
        result = torch.cat(chunks, dim=1)
        if result.shape != generated.shape:
            raise RuntimeContractError(
                "HF policy likelihood shape does not align with generated ids",
                code="hf_backend.policy_logprob_alignment",
                context={
                    "likelihood_shape": tuple(result.shape),
                    "generated_shape": tuple(generated.shape),
                },
            )
        return result
    return _chosen_token_logprobs(scores, generated)


def _unpadded_prompt_rows(
    input_ids: torch.Tensor,
    attention_mask: Any,
) -> tuple[tuple[int, ...], ...]:
    if attention_mask is None:
        return tuple(tuple(int(value) for value in row.tolist()) for row in input_ids)
    mask = _require_rank_two_tensor(attention_mask, field="attention_mask")
    if mask.shape != input_ids.shape:
        raise RuntimeContractError(
            "HF processor attention mask shape does not match input_ids",
            code="hf_backend.processor_attention_shape",
            context={
                "input_shape": tuple(input_ids.shape),
                "mask_shape": tuple(mask.shape),
            },
        )
    return tuple(
        tuple(
            int(value)
            for value, keep in zip(row.tolist(), row_mask.tolist(), strict=True)
            if keep
        )
        for row, row_mask in zip(input_ids, mask, strict=True)
    )


def _observed_image_grids(
    value: Any,
    *,
    batch_size: int,
) -> tuple[tuple[int, int, int] | None, ...]:
    if value is None:
        return tuple(None for _ in range(batch_size))
    tensor = _require_rank_two_tensor(value, field="image_grid_thw")
    if tensor.shape != (batch_size, 3):
        raise RuntimeContractError(
            "HF processor image_grid_thw shape does not match requests",
            code="hf_backend.image_grid_shape",
            context={"shape": tuple(tensor.shape), "batch_size": batch_size},
        )
    return tuple(tuple(int(item) for item in row.tolist()) for row in tensor)


def _model_device(model: Any) -> torch.device:
    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        first = next(iter(parameters()), None)
        if first is not None:
            return torch.device(first.device)
    return torch.device("cpu")


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


def _require_rank_two_tensor(value: Any, *, field: str) -> torch.Tensor:
    if value is None:
        raise RuntimeContractError(
            f"HF {field} is missing",
            code="hf_backend.tensor_missing",
            context={"field": field},
        )
    try:
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeContractError(
            f"HF {field} is not tensor-like",
            code="hf_backend.tensor_type",
            context={"field": field, "value_type": type(value).__name__},
            cause=exc,
        ) from exc
    if tensor.ndim != 2:
        raise RuntimeContractError(
            f"HF {field} must be rank two",
            code="hf_backend.tensor_shape",
            context={"field": field, "shape": tuple(tensor.shape)},
        )
    return tensor


def _require_rank_three_tensor(value: Any, *, field: str) -> torch.Tensor:
    if value is None:
        raise RuntimeContractError(
            f"HF {field} is missing",
            code="hf_backend.tensor_missing",
            context={"field": field},
        )
    try:
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeContractError(
            f"HF {field} is not tensor-like",
            code="hf_backend.tensor_type",
            context={"field": field, "value_type": type(value).__name__},
            cause=exc,
        ) from exc
    if tensor.ndim != 3:
        raise RuntimeContractError(
            f"HF {field} must be rank three",
            code="hf_backend.tensor_shape",
            context={"field": field, "shape": tuple(tensor.shape)},
        )
    return tensor
