"""Hugging Face backend-owned inference session and likelihood tracing."""

from __future__ import annotations

import gc
import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

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
    validate_decode_results,
)
from src.qwen.images import apply_logical_image_transform, rgb_image_sha256
from src.qwen.runtime_loading import QwenLoadOptions, load_qwen_components_from_options
from src.qwen.special_token_embeddings import load_inference_embedding_delta


ComponentsLoader = Callable[[BackendLaunch], Any]


@dataclass(frozen=True)
class TeacherForcedComparison:
    compared_steps: int
    max_absolute_difference: float
    atol: float
    rtol: float


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

    @property
    def receipt(self) -> BackendSessionReceipt:
        return self._receipt

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
        results: list[DecodeResult] = []
        for offset in range(0, len(checked), self._launch.batch_size):
            results.extend(
                self._decode_native_batch(
                    checked[offset : offset + self._launch.batch_size]
                )
            )
        return validate_decode_results(
            requests=checked,
            results=results,
            receipt=self.receipt,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
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
        try:
            values = compute_transition_scores(
                sequences,
                tuple(scores),
                normalize_logits=True,
            )
        except Exception as exc:
            raise RuntimeContractError(
                "HF policy likelihood extraction failed",
                code="hf_backend.policy_logprob_extraction",
                context={
                    "sequence_shape": tuple(sequences.shape),
                    "score_steps": len(scores),
                },
                cause=exc,
            ) from exc
        result = torch.as_tensor(values)
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
