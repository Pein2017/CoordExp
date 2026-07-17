"""Backend-neutral decode records and HF scored-generation tracing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch.nn import functional as F

from src.common.errors import RuntimeContractError


BackendName = Literal["hf", "vllm"]
ALLOWED_STRIP_POLICIES = {"none", "terminal_im_end"}


@dataclass(frozen=True)
class DecodeRequest:
    request_id: str
    prompt_token_ids: list[int]
    model_inputs: Mapping[str, Any]
    max_new_tokens: int
    repetition_penalty: float = 1.0
    temperature: float = 0.0
    top_p: float = 1.0


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


class HFGenerateBackend:
    backend = "hf"
    backend_mode = "generate"
    response_family = "hf"

    def __init__(self, *, model: Any, tokenizer: Any) -> None:
        self.model = model
        self.tokenizer = tokenizer

    def generate_batch(
        self,
        requests: Sequence[DecodeRequest],
        *,
        model_identity: Mapping[str, Any],
        tokenizer_identity: Mapping[str, Any],
        generation_config_fingerprint: str,
    ) -> list[DecodeResult]:
        if not requests:
            return []
        max_new_tokens = self._batch_max_new_tokens(requests)
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
        temperature, top_p = self._batch_sampling_parameters(requests)
        sampling_kwargs: dict[str, Any] = {"do_sample": temperature > 0.0}
        if temperature > 0.0:
            sampling_kwargs.update(temperature=temperature, top_p=top_p)
        outputs = self.model.generate(
            **generate_inputs,
            max_new_tokens=max_new_tokens,
            repetition_penalty=self._batch_repetition_penalty(requests),
            eos_token_id=self._im_end_token_id(),
            pad_token_id=self._pad_token_id(),
            return_dict_in_generate=True,
            output_scores=True,
            **sampling_kwargs,
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
        return [
            self._materialize_result(
                request=request,
                row_index=row,
                generated_ids=[int(token_id) for token_id in generated_suffix[row].tolist()],
                transition_logprobs=[
                    float(value) for value in transition_scores[row].tolist()
                ],
                model_identity=model_identity,
                tokenizer_identity=tokenizer_identity,
                generation_config_fingerprint=generation_config_fingerprint,
            )
            for row, request in enumerate(requests)
        ]

    def _batch_max_new_tokens(self, requests: Sequence[DecodeRequest]) -> int:
        values = {request.max_new_tokens for request in requests}
        if len(values) != 1:
            raise RuntimeContractError(
                "all decode requests in a V1 HF batch must use the same max_new_tokens",
                code="backend_trace.max_new_tokens_mismatch",
                context={"max_new_tokens": sorted(values)},
            )
        return values.pop()

    def _batch_repetition_penalty(self, requests: Sequence[DecodeRequest]) -> float:
        values = {float(request.repetition_penalty) for request in requests}
        if len(values) != 1:
            raise RuntimeContractError(
                "all decode requests in a V1 HF batch must use the same repetition_penalty",
                code="backend_trace.repetition_penalty_mismatch",
                context={"repetition_penalty": sorted(values)},
            )
        return values.pop()

    def _batch_sampling_parameters(
        self, requests: Sequence[DecodeRequest]
    ) -> tuple[float, float]:
        temperatures = {float(request.temperature) for request in requests}
        top_ps = {float(request.top_p) for request in requests}
        if len(temperatures) != 1 or len(top_ps) != 1:
            raise RuntimeContractError(
                "all decode requests in a V1 HF batch must use the same sampling parameters",
                code="backend_trace.sampling_parameter_mismatch",
                context={
                    "temperature": sorted(temperatures),
                    "top_p": sorted(top_ps),
                },
            )
        temperature = temperatures.pop()
        top_p = top_ps.pop()
        if temperature < 0.0 or not 0.0 < top_p <= 1.0:
            raise RuntimeContractError(
                "HF sampling parameters are outside their supported ranges",
                code="backend_trace.sampling_parameter_invalid",
                context={"temperature": temperature, "top_p": top_p},
            )
        return temperature, top_p

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
        compute_transition_scores = getattr(self.model, "compute_transition_scores", None)
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
        )
        result.validate_for_scored()
        return result

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

    def _decode_token(self, token_id: int) -> str:
        return self._decode_tokens([token_id])

    def _decode_tokens(self, token_ids: list[int]) -> str:
        if not token_ids:
            return ""
        return str(self.tokenizer.decode(token_ids, skip_special_tokens=False))


def create_backend(backend: BackendName | str, *, model: Any, tokenizer: Any) -> HFGenerateBackend:
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
        return {key: _move_to_device(item, device=device) for key, item in value.items()}
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
