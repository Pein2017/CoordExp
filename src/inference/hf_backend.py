"""Hugging Face backend-owned inference session and likelihood tracing."""

from __future__ import annotations

import gc
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, cast
from weakref import WeakKeyDictionary

import torch
from torch.nn import functional as F

from src.adapters.dora import attach_dora_adapter
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
from src.qwen.native import (
    NativeRequest, NativeBatch, prepare_native_inputs, prepare_replay,
    configure_left_padding, _require_rank_two_tensor, model_device,
)
from src.qwen.generation import NativeGenerationPolicy, generate_continuations
from src.qwen.runtime_loading import QwenLoadOptions, load_qwen_components_from_options
from src.qwen.special_token_embeddings import attach_embedding_delta


ComponentsLoader = Callable[[BackendLaunch], Any]


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
        configure_left_padding(processor=processor, tokenizer=tokenizer)
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
        replay = prepare_replay(
            model, context.native_inputs,
            prompt_token_ids=state.conditioning_token_ids,
            continuation_token_ids=continuation,
            compact_logits=False,
        )
        with torch.inference_mode():
            outputs = model(**replay.inputs)
        logits = _require_rank_three_tensor(getattr(outputs, "logits", None), field="logits")
        selected_logits = replay.aligned_logits(logits).detach().to(
            device="cpu", dtype=torch.float32,
        ).contiguous()
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
        for offset in range(0, len(checked), self._launch.batch_size):
            results.extend(self._decode_native_batch(checked[offset:offset + self._launch.batch_size]))
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
        generated = generate_continuations(
            self._model,
            NativeBatch(native_inputs, tuple(request.request_id for request in requests)),
            extensions=[() for _ in requests],
            budgets=[policy.max_new_tokens for _ in requests],
            eos_token_id=self._im_end_token_id(), pad_token_id=self._pad_token_id(),
            policy=NativeGenerationPolicy(repetition_penalty=policy.repetition_penalty),
            trace="raw_and_policy" if policy.include_raw_model_logprob else "policy",
        )
        assert all(result.trace is not None for result in generated)
        return tuple(
            self._materialize_result(
                request=request,
                generated_ids=result.trace.token_ids,
                policy_logprobs=result.trace.policy_logprobs,
                raw_logprobs=result.trace.raw_logprobs,
                executed_prompt_ids=executed_prompt_ids[row],
                observed_image_grid_thw=observed_grids[row],
                executed_media_sha256=executed_media_sha256[row],
            )
            for row, (request, result) in enumerate(zip(requests, generated, strict=True))
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
        batch = prepare_native_inputs(
            self._processor,
            tuple(NativeRequest(
                request_id=request.request_id,
                chat_text=request.chat_text,
                image=request.image_path,
                expected_token_ids=request.expected_executed_prompt_token_ids,
                expected_image_grid=request.expected_image_grid_thw,
                expected_image_size=(request.decoded_image_width, request.decoded_image_height),
                image_sha256=request.image_sha256,
                logical_transform=request.logical_transform_id,
            ) for request in requests),
            device=model_device(self._model),
            record_media_identity=True,
        )
        assert batch.media_sha256 is not None
        return dict(batch.inputs), batch.prompt_token_ids, batch.image_grids, batch.media_sha256

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
    adapter_receipt = (
        attach_dora_adapter(qwen.model, adapter_path=adapter.path,
                            base_model_path=qwen.base_model_path, adapter_name=adapter.name)
        if adapter is not None else None
    )
    embedding_delta_receipt = (
        attach_embedding_delta(delta_path=embedding_delta.path, qwen=qwen,
                               source_gate_root=getattr(embedding_delta, "source_gate_root", None))
        if embedding_delta is not None else None
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


def _require_shared_generation_policy(requests: Sequence[DecodeRequest]) -> Any:
    policies = {request.generation_policy for request in requests}
    if len(policies) != 1:
        raise RuntimeContractError(
            "requests in one HF native batch must share generation policy",
            code="hf_backend.generation_policy_mismatch",
            context={"request_ids": [request.request_id for request in requests]},
        )
    return policies.pop()


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
