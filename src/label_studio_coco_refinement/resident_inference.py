"""Single-GPU resident ROI inference over the current CoordExp runtime owners."""

from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import torch
import transformers
from PIL import Image
from transformers import StoppingCriteria, StoppingCriteriaList

from src.common.errors import RuntimeContractError
from src.config.fingerprint import sha256_json
from src.config.inference import InferConfig, ResolvedInferConfig
from src.config.models import ProcessorConfig, TemplateConfig, TemplatePromptConfig
from src.inference.backend import (
    DecodeRequest,
    DecodeResult,
    HFGenerateBackend,
    TokenTrace,
)
from src.inference.data_parallel import resolve_visible_cuda_tokens
from src.inference.image_plan import verify_processor_model_vision_parity
from src.inference.parsing import (
    PARSER_ID,
    PARSER_POLICY,
    ParseRow,
    parse_compact_object_box_closed,
)
from src.inference.prompt import (
    ImagePromptRecord,
    build_image_prompt_record,
    fingerprint_prompt_policy,
    verify_prompt_token_parity,
)
from src.inference.runtime import InferenceRuntime, assemble_runtime
from src.label_studio_coco_refinement.inference_results import RequestTarget
from src.label_studio_coco_refinement.roi_transform import (
    ROI_TRANSFORM_ID,
    RoiLetterboxTransform,
)
from src.qwen.images import (
    QWEN_IMAGE_PROCESSOR_KWARGS,
    QwenImageEncoding,
    encode_qwen_image_canvas,
)


_RESIDENT_PROFILE_SCHEMA = "coordexp-resident-roi-binding-v1"
_MAPPING_STATUS = "deferred_to_inference_results_owner"


@dataclass(frozen=True)
class ImmutableRgbCanvas:
    """Owned RGB bytes for a canvas that cannot be mutated after submission."""

    width: int
    height: int
    rgb_bytes: bytes
    sha256: str

    @classmethod
    def from_image(cls, image: Image.Image) -> "ImmutableRgbCanvas":
        if not isinstance(image, Image.Image):
            raise RuntimeContractError(
                "resident canvas must be a Pillow image",
                code="resident.canvas_type",
                context={"value_type": type(image).__name__},
            )
        if image.mode != "RGB":
            raise RuntimeContractError(
                "resident canvas must already be RGB",
                code="resident.canvas_mode",
                context={"mode": image.mode},
            )
        payload = image.tobytes()
        return cls(
            width=image.width,
            height=image.height,
            rgb_bytes=payload,
            sha256=_canvas_sha256(image.width, image.height, payload),
        )

    def __post_init__(self) -> None:
        if (
            isinstance(self.width, bool)
            or not isinstance(self.width, int)
            or self.width <= 0
            or isinstance(self.height, bool)
            or not isinstance(self.height, int)
            or self.height <= 0
        ):
            raise RuntimeContractError(
                "resident canvas dimensions must be positive integers",
                code="resident.canvas_dimensions",
                context={"width": self.width, "height": self.height},
            )
        if not isinstance(self.rgb_bytes, bytes):
            raise RuntimeContractError(
                "resident canvas storage must be immutable bytes",
                code="resident.canvas_storage",
                context={"value_type": type(self.rgb_bytes).__name__},
            )
        expected_bytes = self.width * self.height * 3
        if len(self.rgb_bytes) != expected_bytes:
            raise RuntimeContractError(
                "resident canvas byte count does not match RGB dimensions",
                code="resident.canvas_byte_count",
                context={"expected": expected_bytes, "observed": len(self.rgb_bytes)},
            )
        expected_sha256 = _canvas_sha256(self.width, self.height, self.rgb_bytes)
        if self.sha256 != expected_sha256:
            raise RuntimeContractError(
                "resident canvas fingerprint does not match its bytes",
                code="resident.canvas_fingerprint",
                context={"expected": expected_sha256, "observed": self.sha256},
            )

    def to_image(self) -> Image.Image:
        return Image.frombytes("RGB", (self.width, self.height), self.rgb_bytes)

    def to_receipt_dict(self) -> dict[str, Any]:
        return {
            "mode": "RGB",
            "width": self.width,
            "height": self.height,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class ResidentProfileBinding:
    """ROI sidecar identity kept strictly outside :class:`InferConfig`."""

    name: str
    resolved_infer_config_fingerprint: str
    prompt_policy_fingerprint: str
    generation_config_fingerprint: str
    runtime_identity_fingerprint: str
    processor_identity_fingerprint: str
    tokenizer_identity_fingerprint: str
    transformers_version: str
    processor_kwargs_json: str
    processor_factor: int
    default_width: int
    default_height: int
    min_axis_pixels: int
    max_axis_pixels: int
    max_total_pixels: int
    deadline_seconds: float
    schema_version: str = _RESIDENT_PROFILE_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise RuntimeContractError(
                "resident profile name must be non-empty text",
                code="resident.profile_name",
            )
        if self.schema_version != _RESIDENT_PROFILE_SCHEMA:
            raise RuntimeContractError(
                "resident profile schema is unsupported",
                code="resident.profile_schema",
                context={"schema_version": self.schema_version},
            )
        for field in (
            "resolved_infer_config_fingerprint",
            "prompt_policy_fingerprint",
            "generation_config_fingerprint",
            "runtime_identity_fingerprint",
            "processor_identity_fingerprint",
            "tokenizer_identity_fingerprint",
        ):
            _require_sha256(getattr(self, field), field=field)
        if (
            not isinstance(self.transformers_version, str)
            or not self.transformers_version.strip()
        ):
            raise RuntimeContractError(
                "resident profile transformers version must be non-empty text",
                code="resident.profile_transformers_version",
            )
        try:
            processor_kwargs = json.loads(self.processor_kwargs_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise RuntimeContractError(
                "resident profile processor kwargs must be canonical JSON",
                code="resident.profile_processor_kwargs",
            ) from exc
        if processor_kwargs != dict(
            QWEN_IMAGE_PROCESSOR_KWARGS
        ) or self.processor_kwargs_json != json.dumps(
            processor_kwargs,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ):
            raise RuntimeContractError(
                "resident profile processor kwargs do not match executed Qwen kwargs",
                code="resident.profile_processor_kwargs",
            )
        for field in (
            "processor_factor",
            "default_width",
            "default_height",
            "min_axis_pixels",
            "max_axis_pixels",
            "max_total_pixels",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise RuntimeContractError(
                    "resident profile integer bounds must be positive",
                    code="resident.profile_bound",
                    context={"field": field, "value": value},
                )
        if self.min_axis_pixels > self.max_axis_pixels:
            raise RuntimeContractError(
                "resident profile minimum axis exceeds maximum axis",
                code="resident.profile_axis_bounds",
            )
        if (
            isinstance(self.deadline_seconds, bool)
            or not isinstance(self.deadline_seconds, (int, float))
            or not math.isfinite(self.deadline_seconds)
            or self.deadline_seconds <= 0.0
        ):
            raise RuntimeContractError(
                "resident profile deadline must be finite and positive",
                code="resident.profile_deadline",
                context={"deadline_seconds": self.deadline_seconds},
            )
        self.validate_canvas(self.default_width, self.default_height)

    @property
    def fingerprint(self) -> str:
        return sha256_json(self.identity_payload())

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "resolved_infer_config_fingerprint": self.resolved_infer_config_fingerprint,
            "prompt_policy_fingerprint": self.prompt_policy_fingerprint,
            "generation_config_fingerprint": self.generation_config_fingerprint,
            "runtime_identity_fingerprint": self.runtime_identity_fingerprint,
            "processor_identity_fingerprint": self.processor_identity_fingerprint,
            "tokenizer_identity_fingerprint": self.tokenizer_identity_fingerprint,
            "transformers_version": self.transformers_version,
            "parser": {"id": PARSER_ID, "policy": PARSER_POLICY},
            "transform_id": ROI_TRANSFORM_ID,
            "processor": {
                "factor": self.processor_factor,
                "do_resize": False,
                "kwargs": json.loads(self.processor_kwargs_json),
                "default_canvas": [self.default_width, self.default_height],
                "axis_bounds": [self.min_axis_pixels, self.max_axis_pixels],
                "max_total_pixels": self.max_total_pixels,
            },
            "deadline_seconds": float(self.deadline_seconds),
        }

    def to_receipt_dict(self) -> dict[str, Any]:
        return {**self.identity_payload(), "profile_fingerprint": self.fingerprint}

    def validate_canvas(self, width: int, height: int) -> tuple[int, int]:
        if (
            isinstance(width, bool)
            or not isinstance(width, int)
            or isinstance(height, bool)
            or not isinstance(height, int)
        ):
            raise RuntimeContractError(
                "resident canvas dimensions must be integers",
                code="resident.profile_canvas_type",
            )
        if not (
            self.min_axis_pixels <= width <= self.max_axis_pixels
            and self.min_axis_pixels <= height <= self.max_axis_pixels
        ):
            raise RuntimeContractError(
                "resident canvas is outside profile axis bounds",
                code="resident.profile_canvas_axis",
                context={"width": width, "height": height},
            )
        if width % self.processor_factor or height % self.processor_factor:
            raise RuntimeContractError(
                "resident canvas is not divisible by the processor factor",
                code="resident.profile_canvas_factor",
                context={
                    "width": width,
                    "height": height,
                    "processor_factor": self.processor_factor,
                },
            )
        if width * height > self.max_total_pixels:
            raise RuntimeContractError(
                "resident canvas exceeds the profile pixel budget",
                code="resident.profile_canvas_pixels",
                context={"pixels": width * height, "maximum": self.max_total_pixels},
            )
        return width, height


@dataclass(frozen=True)
class ResidentInferenceRequest:
    target: RequestTarget
    transform: RoiLetterboxTransform
    canvas: ImmutableRgbCanvas

    def validate_for_profile(self, profile: ResidentProfileBinding) -> None:
        if self.target.profile_fingerprint != profile.fingerprint:
            raise RuntimeContractError(
                "request target is bound to a different resident profile",
                code="resident.request_profile_mismatch",
                context={
                    "request": self.target.profile_fingerprint,
                    "engine": profile.fingerprint,
                },
            )
        if self.target.transform_fingerprint != self.transform.fingerprint:
            raise RuntimeContractError(
                "request target transform fingerprint does not match the transform",
                code="resident.request_transform_mismatch",
            )
        if self.transform.transform_id != ROI_TRANSFORM_ID:
            raise RuntimeContractError(
                "request uses an unsupported ROI transform",
                code="resident.request_transform_id",
                context={"transform_id": self.transform.transform_id},
            )
        if self.transform.processor_do_resize is not False:
            raise RuntimeContractError(
                "resident ROI requests must preserve no-resize processor semantics",
                code="resident.request_resize_enabled",
            )
        if (self.canvas.width, self.canvas.height) != self.transform.canvas_size:
            raise RuntimeContractError(
                "immutable canvas dimensions do not match the ROI transform",
                code="resident.request_canvas_mismatch",
                context={
                    "canvas": [self.canvas.width, self.canvas.height],
                    "transform": list(self.transform.canvas_size),
                },
            )
        profile.validate_canvas(self.canvas.width, self.canvas.height)


@dataclass(frozen=True)
class CudaBinding:
    visible_cuda_tokens: tuple[str, ...]
    cuda_available: bool
    device_count: int
    current_device: int | None

    def to_receipt_dict(self) -> dict[str, Any]:
        return {
            "visible_cuda_tokens": list(self.visible_cuda_tokens),
            "cuda_available": self.cuda_available,
            "device_count": self.device_count,
            "current_device": self.current_device,
            "logical_device": "cuda:0",
        }


def probe_cuda_binding() -> CudaBinding:
    """Observe, but never rewrite, the process CUDA binding."""

    tokens = resolve_visible_cuda_tokens(
        environ=os.environ,
        cuda_device_count=torch.cuda.device_count,
    )
    available = bool(torch.cuda.is_available())
    count = int(torch.cuda.device_count())
    current = int(torch.cuda.current_device()) if available and count > 0 else None
    return CudaBinding(
        visible_cuda_tokens=tokens,
        cuda_available=available,
        device_count=count,
        current_device=current,
    )


class CancellationToken:
    """Thread-safe caller cancellation; the first reason wins."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._reason: str | None = None

    def cancel(self, reason: str = "client_cancelled") -> None:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("cancellation reason must be non-empty text")
        with self._lock:
            if self._event.is_set():
                return
            self._reason = reason
            self._event.set()

    @property
    def requested(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> str | None:
        with self._lock:
            return self._reason


@dataclass(frozen=True)
class CancellationMetadata:
    requested: bool
    reason: str | None
    observed_by_backend: bool
    backend_started: bool
    cuda_synchronized: bool
    deadline_seconds: float

    def to_receipt_dict(self) -> dict[str, Any]:
        return {
            "requested": self.requested,
            "reason": self.reason,
            "observed_by_backend": self.observed_by_backend,
            "backend_started": self.backend_started,
            "cuda_synchronized": self.cuda_synchronized,
            "deadline_seconds": self.deadline_seconds,
        }


class ResidentInferenceCancelled(RuntimeContractError):
    def __init__(self, metadata: CancellationMetadata) -> None:
        self.metadata = metadata
        super().__init__(
            "resident inference was cancelled at a cooperative boundary",
            code="resident.cancelled",
            context=metadata.to_receipt_dict(),
        )


@dataclass(frozen=True)
class ResidentInferenceResult:
    target: RequestTarget
    transform: RoiLetterboxTransform
    profile: ResidentProfileBinding
    prompt: ImagePromptRecord
    image_encoding: QwenImageEncoding
    decode: DecodeResult
    parse: ParseRow
    cancellation: CancellationMetadata
    cuda_binding: CudaBinding

    @property
    def raw_generated_text(self) -> str:
        return self.decode.raw_generated_text

    @property
    def parser_text(self) -> str:
        return self.decode.parser_text

    def to_receipt_dict(self) -> dict[str, Any]:
        return {
            "request_target": self.target.to_receipt_dict(),
            "profile": self.profile.to_receipt_dict(),
            "transform": self.transform.to_receipt_dict(),
            "prompt": self.prompt.to_artifact_dict(),
            "image": self.image_encoding.to_artifact_dict(),
            "decode": {
                "request_id": self.decode.request_id,
                "backend": self.decode.backend,
                "backend_mode": self.decode.backend_mode,
                "response_family": self.decode.response_family,
                "prompt_token_ids": list(self.decode.prompt_token_ids),
                "generated_token_ids": list(self.decode.generated_token_ids),
                "raw_generated_text": self.decode.raw_generated_text,
                "parser_text": self.decode.parser_text,
                "strip_policy": self.decode.strip_policy,
                "stop_reason": self.decode.stop_reason,
                "model_identity": dict(self.decode.model_identity),
                "tokenizer_identity": dict(self.decode.tokenizer_identity),
                "generation_config_fingerprint": (
                    self.decode.generation_config_fingerprint
                ),
                "token_trace": [
                    _token_trace_dict(item) for item in self.decode.token_trace
                ],
            },
            "parse": {
                **self.parse.to_artifact_dict(),
                "diagnostics": list(self.parse.diagnostics),
            },
            "mapping_status": _MAPPING_STATUS,
            "cancellation": self.cancellation.to_receipt_dict(),
            "cuda_binding": self.cuda_binding.to_receipt_dict(),
        }


class ResidentDecodeBackend(Protocol):
    def generate_batch(
        self,
        requests: Sequence[DecodeRequest],
        *,
        model_identity: Mapping[str, Any],
        tokenizer_identity: Mapping[str, Any],
        generation_config_fingerprint: str,
        cancellation_control: "GenerationControl",
    ) -> Sequence[DecodeResult]: ...


class GenerationControl:
    """Backend polling seam for deadline and caller cancellation."""

    def __init__(
        self,
        *,
        deadline_seconds: float,
        cancellation_token: CancellationToken | None,
        clock: Callable[[], float],
    ) -> None:
        self.deadline_seconds = float(deadline_seconds)
        self._token = cancellation_token
        self._clock = clock
        self._deadline_at = clock() + self.deadline_seconds
        self._observation_lock = threading.Lock()
        self._observed_reason: str | None = None

    def requested_reason(self) -> str | None:
        if self._token is not None and self._token.requested:
            return self._token.reason or "client_cancelled"
        if self._clock() >= self._deadline_at:
            return "deadline_exceeded"
        return None

    def backend_should_stop(self) -> bool:
        reason = self.requested_reason()
        if reason is None:
            return False
        with self._observation_lock:
            if self._observed_reason is None:
                self._observed_reason = reason
        return True

    @property
    def observed_by_backend(self) -> bool:
        with self._observation_lock:
            return self._observed_reason is not None


class ResidentRoiInferenceEngine:
    """Load one current runtime once and serve one ROI generation at a time."""

    def __init__(
        self,
        *,
        resolved: ResolvedInferConfig,
        profile: ResidentProfileBinding,
        runtime: InferenceRuntime,
        backend: ResidentDecodeBackend,
        cuda_binding: CudaBinding,
        cuda_synchronize: Callable[[], None],
        clock: Callable[[], float],
    ) -> None:
        self.resolved = resolved
        self.profile = profile
        self.runtime = runtime
        self.backend = backend
        self.cuda_binding = cuda_binding
        self._cuda_synchronize = cuda_synchronize
        self._clock = clock
        self.transformers_version = str(transformers.__version__)
        self.processor_kwargs = dict(QWEN_IMAGE_PROCESSOR_KWARGS)
        self._single_flight = threading.Lock()
        self._template_config = _template_config(resolved.config)
        self._tokenizer_identity = _tokenizer_identity(runtime.qwen)
        self._processor_config = ProcessorConfig(
            do_resize=False,
            max_raw_pixels=profile.max_total_pixels,
            max_merged_visual_tokens=max(
                1, profile.max_total_pixels // (profile.processor_factor**2)
            ),
        )

    @classmethod
    def load(
        cls,
        *,
        resolved: ResolvedInferConfig,
        profile: ResidentProfileBinding,
        runtime_factory: Callable[[InferConfig], InferenceRuntime] = assemble_runtime,
        backend_factory: Callable[[InferenceRuntime], ResidentDecodeBackend]
        | None = None,
        cuda_probe: Callable[[], CudaBinding] = probe_cuda_binding,
        cuda_synchronize: Callable[[], None] = torch.cuda.synchronize,
        clock: Callable[[], float] = time.monotonic,
    ) -> "ResidentRoiInferenceEngine":
        _validate_config_binding(resolved=resolved, profile=profile)
        cuda_binding = cuda_probe()
        _require_exactly_one_cuda(cuda_binding)

        runtime = runtime_factory(resolved.config)
        _validate_loaded_runtime(runtime=runtime, profile=profile)
        backend = (
            _CooperativeHFBackend(runtime)
            if backend_factory is None
            else backend_factory(runtime)
        )
        return cls(
            resolved=resolved,
            profile=profile,
            runtime=runtime,
            backend=backend,
            cuda_binding=cuda_binding,
            cuda_synchronize=cuda_synchronize,
            clock=clock,
        )

    def infer_one(
        self,
        request: ResidentInferenceRequest,
        *,
        cancellation_token: CancellationToken | None = None,
    ) -> ResidentInferenceResult:
        if not self._single_flight.acquire(blocking=False):
            raise RuntimeContractError(
                "resident inference engine already has an in-flight request",
                code="resident.single_flight_busy",
                context={"request_id": request.target.request_id},
            )
        try:
            return self._infer_locked(request, cancellation_token=cancellation_token)
        finally:
            self._single_flight.release()

    def _infer_locked(
        self,
        request: ResidentInferenceRequest,
        *,
        cancellation_token: CancellationToken | None,
    ) -> ResidentInferenceResult:
        request.validate_for_profile(self.profile)
        control = GenerationControl(
            deadline_seconds=self.profile.deadline_seconds,
            cancellation_token=cancellation_token,
            clock=self._clock,
        )
        self._raise_if_cancelled_before_backend(control)

        image = request.canvas.to_image()
        try:
            prompt = build_image_prompt_record(
                example_id=request.target.request_id,
                image=image,
                template_config=self._template_config,
                processor=self.runtime.qwen.processor,
            )
            encoding = encode_qwen_image_canvas(
                example_id=request.target.request_id,
                image=image,
                components=self.runtime.qwen,
                processor_config=self._processor_config,
            )
        finally:
            image.close()

        request.transform.assert_no_resize_processor_canvas(
            do_resize=False,
            observed_width=encoding.width,
            observed_height=encoding.height,
        )
        expected_grid = request.transform.expected_grid_thw(
            patch_size=self.runtime.qwen.processor_identity.patch_size
        )
        if encoding.image_grid_thw != expected_grid:
            raise RuntimeContractError(
                "resident Qwen image grid does not match the immutable ROI transform",
                code="resident.image_grid_mismatch",
                context={
                    "expected": list(expected_grid),
                    "observed": list(encoding.image_grid_thw),
                },
            )
        self._raise_if_cancelled_before_backend(control)

        decode_request = DecodeRequest(
            request_id=request.target.request_id,
            prompt_token_ids=list(prompt.prompt_token_ids),
            model_inputs={
                "pixel_values": encoding.pixel_values,
                "image_grid_thw": encoding.image_grid_thw_tensor,
            },
            max_new_tokens=self.resolved.config.generation.max_new_tokens,
            repetition_penalty=self.resolved.config.generation.repetition_penalty,
        )
        try:
            results = self.backend.generate_batch(
                [decode_request],
                model_identity=self.runtime.model_identity,
                tokenizer_identity=self._tokenizer_identity,
                generation_config_fingerprint=self.profile.generation_config_fingerprint,
                cancellation_control=control,
            )
        except Exception:
            self._raise_if_cancelled_after_backend(control)
            raise
        self._raise_if_cancelled_after_backend(control)
        terminal_cancellation = CancellationMetadata(
            requested=False,
            reason=None,
            observed_by_backend=control.observed_by_backend,
            backend_started=True,
            cuda_synchronized=False,
            deadline_seconds=control.deadline_seconds,
        )
        decode = _one_decode_result(
            request_id=request.target.request_id,
            results=results,
        )
        decode.validate_for_scored()
        _validate_decode_identity(
            decode=decode,
            config=self.resolved.config,
            runtime=self.runtime,
            tokenizer_identity=self._tokenizer_identity,
            profile=self.profile,
        )
        verify_prompt_token_parity(
            prompt,
            backend_prompt_token_ids=decode.prompt_token_ids,
        )
        parsed = parse_compact_object_box_closed(
            decode.parser_text,
            row_id=request.target.request_id,
            row_index=prompt.row_index,
            image_width=request.canvas.width,
            image_height=request.canvas.height,
        )
        return ResidentInferenceResult(
            target=request.target,
            transform=request.transform,
            profile=self.profile,
            prompt=prompt,
            image_encoding=encoding,
            decode=decode,
            parse=parsed,
            cancellation=terminal_cancellation,
            cuda_binding=self.cuda_binding,
        )

    def _raise_if_cancelled_before_backend(self, control: GenerationControl) -> None:
        if control.requested_reason() is None:
            return
        raise ResidentInferenceCancelled(
            _cancellation_metadata(
                control,
                backend_started=False,
                cuda_synchronized=False,
            )
        )

    def _raise_if_cancelled_after_backend(self, control: GenerationControl) -> None:
        reason = control.requested_reason()
        if reason is None:
            return
        try:
            self._cuda_synchronize()
        except Exception as exc:
            raise RuntimeContractError(
                "CUDA synchronization failed after resident cancellation",
                code="resident.cancel_synchronize_failed",
                context={"reason": reason},
                cause=exc,
            ) from exc
        metadata = _cancellation_metadata(
            control,
            backend_started=True,
            cuda_synchronized=True,
        )
        if not control.observed_by_backend:
            raise RuntimeContractError(
                "resident backend returned after cancellation without observing it",
                code="resident.cancellation_unobserved",
                context=metadata.to_receipt_dict(),
            )
        raise ResidentInferenceCancelled(metadata)


class _CancellationStoppingCriteria(StoppingCriteria):
    def __init__(self, control: GenerationControl) -> None:
        self.control = control

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        **_: Any,
    ) -> torch.BoolTensor:
        return torch.full(
            (input_ids.shape[0],),
            self.control.backend_should_stop(),
            device=input_ids.device,
            dtype=torch.bool,
        )


class _CooperativeModelProxy:
    def __init__(self, model: Any) -> None:
        self._model = model
        self._control: GenerationControl | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

    def bind(self, control: GenerationControl) -> None:
        if self._control is not None:
            raise RuntimeContractError(
                "resident model proxy already has a cancellation binding",
                code="resident.backend_reentrant",
            )
        self._control = control

    def clear(self) -> None:
        self._control = None

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        control = self._control
        if control is None:
            raise RuntimeContractError(
                "resident model generation has no cancellation binding",
                code="resident.backend_control_missing",
            )
        if kwargs.get("stopping_criteria") is not None:
            raise RuntimeContractError(
                "resident generation cannot replace an existing stopping criteria owner",
                code="resident.backend_stopping_collision",
            )
        kwargs["stopping_criteria"] = StoppingCriteriaList(
            [_CancellationStoppingCriteria(control)]
        )
        return self._model.generate(*args, **kwargs)


class _CooperativeHFBackend:
    def __init__(self, runtime: InferenceRuntime) -> None:
        qwen = runtime.qwen
        if qwen.model is None:
            raise RuntimeContractError(
                "resident HF backend requires a loaded model",
                code="resident.model_missing",
            )
        self._model = _CooperativeModelProxy(qwen.model)
        self._delegate = HFGenerateBackend(model=self._model, tokenizer=qwen.tokenizer)

    def generate_batch(
        self,
        requests: Sequence[DecodeRequest],
        *,
        model_identity: Mapping[str, Any],
        tokenizer_identity: Mapping[str, Any],
        generation_config_fingerprint: str,
        cancellation_control: GenerationControl,
    ) -> Sequence[DecodeResult]:
        self._model.bind(cancellation_control)
        try:
            return self._delegate.generate_batch(
                requests,
                model_identity=model_identity,
                tokenizer_identity=tokenizer_identity,
                generation_config_fingerprint=generation_config_fingerprint,
            )
        finally:
            self._model.clear()


def _validate_config_binding(
    *,
    resolved: ResolvedInferConfig,
    profile: ResidentProfileBinding,
) -> None:
    if not isinstance(resolved, ResolvedInferConfig):
        raise RuntimeContractError(
            "resident runtime requires a resolved strict inference config",
            code="resident.config_type",
            context={"value_type": type(resolved).__name__},
        )
    if str(transformers.__version__) != profile.transformers_version:
        raise RuntimeContractError(
            "installed Transformers version does not match the resident profile",
            code="resident.transformers_version_mismatch",
            context={
                "expected": profile.transformers_version,
                "observed": str(transformers.__version__),
            },
        )
    strict_payload = resolved.config.model_dump(mode="json")
    if "roi_inference" in strict_payload or "roi_inference" in resolved.config_dict:
        raise RuntimeContractError(
            "ROI settings must remain in the resident profile sidecar",
            code="resident.config_roi_leak",
        )
    if resolved.config_dict != strict_payload or resolved.fingerprint != sha256_json(
        strict_payload
    ):
        raise RuntimeContractError(
            "resolved inference config payload and fingerprint are inconsistent",
            code="resident.config_fingerprint_invalid",
        )
    if profile.resolved_infer_config_fingerprint != resolved.fingerprint:
        raise RuntimeContractError(
            "resident profile is bound to a different strict inference config",
            code="resident.config_profile_mismatch",
        )
    template = _template_config(resolved.config)
    if profile.prompt_policy_fingerprint != fingerprint_prompt_policy(template):
        raise RuntimeContractError(
            "resident profile prompt policy does not match the inference config",
            code="resident.prompt_profile_mismatch",
        )
    expected_generation = sha256_json(
        resolved.config.generation.model_dump(mode="json")
    )
    if profile.generation_config_fingerprint != expected_generation:
        raise RuntimeContractError(
            "resident profile generation policy does not match the inference config",
            code="resident.generation_profile_mismatch",
        )


def _require_exactly_one_cuda(binding: CudaBinding) -> None:
    if (
        len(binding.visible_cuda_tokens) != 1
        or binding.cuda_available is not True
        or binding.device_count != 1
        or binding.current_device != 0
    ):
        raise RuntimeContractError(
            "resident inference requires exactly one configured CUDA device",
            code="resident.cuda_binding",
            context=binding.to_receipt_dict(),
        )


def _validate_loaded_runtime(
    *,
    runtime: InferenceRuntime,
    profile: ResidentProfileBinding,
) -> None:
    qwen = runtime.qwen
    verify_processor_model_vision_parity(
        processor_identity=qwen.processor_identity,
        model_config=_model_config(qwen),
    )
    observed_factor = (
        qwen.processor_identity.patch_size * qwen.processor_identity.merge_size
    )
    if observed_factor != profile.processor_factor:
        raise RuntimeContractError(
            "loaded processor factor does not match the resident profile",
            code="resident.processor_factor_mismatch",
            context={"expected": profile.processor_factor, "observed": observed_factor},
        )
    checks = {
        "runtime_identity_fingerprint": sha256_json(dict(runtime.model_identity)),
        "processor_identity_fingerprint": sha256_json(
            qwen.processor_identity.to_artifact_dict()
        ),
        "tokenizer_identity_fingerprint": sha256_json(_tokenizer_identity(qwen)),
    }
    if str(transformers.__version__) != profile.transformers_version:
        raise RuntimeContractError(
            "loaded Transformers version does not match the resident profile",
            code="resident.transformers_version_mismatch",
            context={
                "expected": profile.transformers_version,
                "observed": str(transformers.__version__),
            },
        )
    for field, observed in checks.items():
        expected = getattr(profile, field)
        if observed != expected:
            raise RuntimeContractError(
                "loaded resident runtime identity does not match the profile",
                code="resident.runtime_identity_mismatch",
                context={"field": field, "expected": expected, "observed": observed},
            )
    model = getattr(qwen, "model", None)
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        raise RuntimeContractError(
            "resident runtime model does not expose parameters",
            code="resident.model_device_missing",
        )
    first_parameter = next(iter(parameters()), None)
    device = getattr(first_parameter, "device", None)
    try:
        normalized_device = None if device is None else torch.device(device)
    except (TypeError, ValueError, RuntimeError):
        normalized_device = None
    if normalized_device != torch.device("cuda:0"):
        raise RuntimeContractError(
            "resident runtime model is not on its sole logical CUDA device",
            code="resident.model_device",
            context={"device": None if device is None else str(device)},
        )


def _template_config(config: InferConfig) -> TemplateConfig:
    return TemplateConfig(
        object_field_order=config.template.object_field_order,
        object_ordering=config.template.object_ordering,
        assistant_format=config.template.assistant_format,
        prompt=TemplatePromptConfig(
            system=config.template.prompt.system,
            user=config.template.prompt.user,
        ),
    )


def _tokenizer_identity(qwen: Any) -> dict[str, Any]:
    token_identity = getattr(qwen, "token_identity", None)
    if token_identity is not None and hasattr(token_identity, "to_artifact_dict"):
        return dict(token_identity.to_artifact_dict())
    tokenizer_sha256 = getattr(qwen, "tokenizer_sha256", None)
    if tokenizer_sha256:
        return {"tokenizer_sha256": str(tokenizer_sha256)}
    raise RuntimeContractError(
        "resident runtime does not expose a tokenizer identity",
        code="resident.tokenizer_identity_missing",
    )


def _model_config(qwen: Any) -> Any:
    if hasattr(qwen, "config"):
        return qwen.config
    model = getattr(qwen, "model", None)
    if model is not None and hasattr(model, "config"):
        return model.config
    raise RuntimeContractError(
        "resident runtime does not expose a model config",
        code="resident.model_config_missing",
    )


def _one_decode_result(
    *,
    request_id: str,
    results: Sequence[DecodeResult],
) -> DecodeResult:
    if len(results) != 1:
        raise RuntimeContractError(
            "resident backend must return exactly one decode result",
            code="resident.decode_cardinality",
            context={"request_id": request_id, "result_count": len(results)},
        )
    result = results[0]
    if result.request_id != request_id:
        raise RuntimeContractError(
            "resident backend returned a result for a different request",
            code="resident.decode_request_mismatch",
            context={"expected": request_id, "observed": result.request_id},
        )
    return result


def _validate_decode_identity(
    *,
    decode: DecodeResult,
    config: InferConfig,
    runtime: InferenceRuntime,
    tokenizer_identity: Mapping[str, Any],
    profile: ResidentProfileBinding,
) -> None:
    expected_backend = config.backend.type
    if (
        decode.backend != expected_backend
        or decode.backend_mode != "generate"
        or decode.response_family != expected_backend
    ):
        raise RuntimeContractError(
            "resident backend identity does not match the configured decode path",
            code="resident.decode_backend_identity",
            context={
                "backend": decode.backend,
                "backend_mode": decode.backend_mode,
                "response_family": decode.response_family,
            },
        )
    if dict(decode.model_identity) != dict(runtime.model_identity):
        raise RuntimeContractError(
            "resident decode model identity differs from the loaded runtime",
            code="resident.decode_model_identity",
        )
    if dict(decode.tokenizer_identity) != dict(tokenizer_identity):
        raise RuntimeContractError(
            "resident decode tokenizer identity differs from the loaded runtime",
            code="resident.decode_tokenizer_identity",
        )
    if decode.generation_config_fingerprint != profile.generation_config_fingerprint:
        raise RuntimeContractError(
            "backend generation fingerprint does not match the resident profile",
            code="resident.generation_fingerprint_mismatch",
        )


def _cancellation_metadata(
    control: GenerationControl,
    *,
    backend_started: bool,
    cuda_synchronized: bool,
) -> CancellationMetadata:
    reason = control.requested_reason()
    return CancellationMetadata(
        requested=reason is not None,
        reason=reason,
        observed_by_backend=control.observed_by_backend,
        backend_started=backend_started,
        cuda_synchronized=cuda_synchronized,
        deadline_seconds=control.deadline_seconds,
    )


def _canvas_sha256(width: int, height: int, payload: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(f"RGB:{width}x{height}\0".encode("ascii"))
    digest.update(payload)
    return digest.hexdigest()


def _require_sha256(value: Any, *, field: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise RuntimeContractError(
            "resident profile identity must be a lowercase SHA-256",
            code="resident.profile_fingerprint",
            context={"field": field},
        )


def _token_trace_dict(trace: TokenTrace) -> dict[str, Any]:
    return {
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


__all__ = [
    "CancellationMetadata",
    "CancellationToken",
    "CudaBinding",
    "GenerationControl",
    "ImmutableRgbCanvas",
    "ResidentInferenceCancelled",
    "ResidentInferenceRequest",
    "ResidentInferenceResult",
    "ResidentProfileBinding",
    "ResidentRoiInferenceEngine",
    "probe_cuda_binding",
]
