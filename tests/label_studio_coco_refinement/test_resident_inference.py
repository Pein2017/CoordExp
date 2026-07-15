from __future__ import annotations

import ast
import json
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import transformers
from PIL import Image

from src.common.errors import RuntimeContractError
from src.config.fingerprint import sha256_json
from src.config.inference import (
    INFER_CONFIG_LOADER_VERSION,
    InferConfig,
    ResolvedInferConfig,
)
from src.config.models import TemplateConfig, TemplatePromptConfig
from src.inference.backend import DecodeResult, TokenTrace
from src.inference.prompt import fingerprint_prompt_policy
from src.inference.runtime import InferenceRuntime
from src.label_studio_coco_refinement.inference_results import RequestTarget
from src.label_studio_coco_refinement.resident_inference import (
    CancellationToken,
    CudaBinding,
    GenerationControl,
    ImmutableRgbCanvas,
    ResidentInferenceCancelled,
    ResidentInferenceRequest,
    ResidentProfileBinding,
    ResidentRoiInferenceEngine,
)
from src.label_studio_coco_refinement.roi_transform import RoiLetterboxTransform
from src.qwen.runtime_loading import QwenProcessorIdentity


_PARSER_TEXT = (
    "<|object_ref_start|>cat<|object_ref_end|>"
    "<|box_start|><|coord_100|><|coord_200|>"
    "<|coord_700|><|coord_800|><|box_end|>"
)


class _FakeTokenizer:
    pad_token_id = 0


class _FakeImageProcessor:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> dict[str, torch.Tensor]:
        self.calls.append(kwargs)
        image = kwargs["images"][0]
        rows = image.width * image.height // (16**2)
        return {
            "pixel_values": torch.zeros((rows, 1536), dtype=torch.float32),
            "image_grid_thw": torch.tensor(
                [[1, image.height // 16, image.width // 16]], dtype=torch.long
            ),
        }


class _FakeProcessor:
    def __init__(self) -> None:
        self.tokenizer = _FakeTokenizer()
        self.image_processor = _FakeImageProcessor()
        self.chat_calls: list[dict[str, Any]] = []

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        **kwargs: Any,
    ) -> str | list[int]:
        self.chat_calls.append(
            {
                "messages": messages,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
                **kwargs,
            }
        )
        assert add_generation_prompt is True
        assert messages[-1]["content"][0]["type"] == "image"
        if tokenize:
            assert kwargs["do_resize"] is False
            return [11, 12, 13]
        return "<|im_start|>user\n<|vision_start|>detect<|im_end|>\n"


class _FakeTokenIdentity:
    def to_artifact_dict(self) -> dict[str, Any]:
        return {"tokenizer_sha256": "fake-tokenizer"}


class _FakeModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            vision_config=SimpleNamespace(
                patch_size=16,
                spatial_merge_size=2,
                temporal_patch_size=2,
            )
        )
        self._parameter = SimpleNamespace(device=torch.device("cuda:0"))

    def parameters(self) -> list[Any]:
        return [self._parameter]


class _FakeBackend:
    def __init__(self) -> None:
        self.calls = 0
        self.mode = "success"
        self.entered = threading.Event()
        self.release = threading.Event()

    def generate_batch(
        self,
        requests: list[Any],
        *,
        model_identity: dict[str, Any],
        tokenizer_identity: dict[str, Any],
        generation_config_fingerprint: str,
        cancellation_control: GenerationControl,
    ) -> list[DecodeResult]:
        self.calls += 1
        request = requests[0]
        if self.mode == "wait_for_cancellation":
            while not cancellation_control.backend_should_stop():
                time.sleep(0.001)
        elif self.mode == "ignore_cancellation":
            time.sleep(cancellation_control.deadline_seconds * 2.0)
        elif self.mode == "block":
            self.entered.set()
            assert self.release.wait(timeout=1.0)
        return [
            _decode_result(
                request=request,
                model_identity=model_identity,
                tokenizer_identity=tokenizer_identity,
                generation_config_fingerprint=generation_config_fingerprint,
            )
        ]


def test_loads_runtime_once_and_returns_current_dual_view_parse_contract() -> None:
    resolved = _resolved_config()
    runtime = _runtime()
    profile = _profile(resolved, runtime)
    backend = _FakeBackend()
    loads: list[InferConfig] = []
    backend_loads: list[InferenceRuntime] = []

    def runtime_factory(config: InferConfig) -> InferenceRuntime:
        loads.append(config)
        return runtime

    def backend_factory(observed: InferenceRuntime) -> _FakeBackend:
        backend_loads.append(observed)
        return backend

    engine = ResidentRoiInferenceEngine.load(
        resolved=resolved,
        profile=profile,
        runtime_factory=runtime_factory,
        backend_factory=backend_factory,
        cuda_probe=_one_cuda,
    )
    first = engine.infer_one(_request(profile, request_id="request-1"))
    second = engine.infer_one(_request(profile, request_id="request-2"))

    assert loads == [resolved.config]
    assert backend_loads == [runtime]
    assert backend.calls == 2
    assert "roi_inference" not in resolved.config_dict
    assert first.raw_generated_text == _PARSER_TEXT + "<|im_end|>"
    assert first.parser_text == _PARSER_TEXT
    assert first.parse.parse_status == "accepted"
    assert first.parse.predictions[0]["description"] == "cat"
    assert first.decode.stop_reason == "eos_token"
    assert second.target.request_id == "request-2"
    receipt = first.to_receipt_dict()
    assert receipt["decode"]["raw_generated_text"] != receipt["decode"]["parser_text"]
    assert receipt["decode"]["strip_policy"] == "terminal_im_end"
    assert receipt["parse"]["parser_id"] == "compact-object-box-closed-v1"
    assert receipt["mapping_status"] == "deferred_to_inference_results_owner"
    assert receipt["cancellation"]["requested"] is False
    assert receipt["image"]["image_path"] is None


@pytest.mark.parametrize(
    "binding",
    [
        CudaBinding((), False, 0, None),
        CudaBinding(("0", "1"), True, 2, 0),
        CudaBinding(("0",), True, 1, 1),
    ],
)
def test_cuda_gate_requires_exactly_one_logical_gpu_before_runtime_load(
    binding: CudaBinding,
) -> None:
    resolved = _resolved_config()
    runtime = _runtime()
    loads = 0

    def runtime_factory(_: InferConfig) -> InferenceRuntime:
        nonlocal loads
        loads += 1
        return runtime

    with pytest.raises(RuntimeContractError) as exc_info:
        ResidentRoiInferenceEngine.load(
            resolved=resolved,
            profile=_profile(resolved, runtime),
            runtime_factory=runtime_factory,
            backend_factory=lambda _: _FakeBackend(),
            cuda_probe=lambda: binding,
        )

    assert exc_info.value.code == "resident.cuda_binding"
    assert loads == 0


def test_profile_and_strict_config_mismatch_fails_before_runtime_load() -> None:
    resolved = _resolved_config()
    runtime = _runtime()
    profile = replace(
        _profile(resolved, runtime),
        resolved_infer_config_fingerprint="0" * 64,
    )
    loads = 0

    def runtime_factory(_: InferConfig) -> InferenceRuntime:
        nonlocal loads
        loads += 1
        return runtime

    with pytest.raises(RuntimeContractError) as exc_info:
        ResidentRoiInferenceEngine.load(
            resolved=resolved,
            profile=profile,
            runtime_factory=runtime_factory,
            backend_factory=lambda _: _FakeBackend(),
            cuda_probe=_one_cuda,
        )

    assert exc_info.value.code == "resident.config_profile_mismatch"
    assert loads == 0


def test_transformers_version_and_full_processor_kwargs_are_profile_bound() -> None:
    resolved = _resolved_config()
    runtime = _runtime()
    profile = _profile(resolved, runtime)

    with pytest.raises(RuntimeContractError) as exc_info:
        ResidentRoiInferenceEngine.load(
            resolved=resolved,
            profile=replace(profile, transformers_version="0.0.invalid"),
            runtime_factory=lambda _: runtime,
            backend_factory=lambda _: _FakeBackend(),
            cuda_probe=_one_cuda,
        )
    assert exc_info.value.code == "resident.transformers_version_mismatch"

    with pytest.raises(RuntimeContractError) as exc_info:
        replace(profile, processor_kwargs_json='{"do_resize":false}')
    assert exc_info.value.code == "resident.profile_processor_kwargs"


def test_deadline_cancellation_synchronizes_then_reuses_same_backend() -> None:
    resolved = _resolved_config()
    runtime = _runtime()
    profile = _profile(resolved, runtime, deadline_seconds=0.02)
    backend = _FakeBackend()
    backend.mode = "wait_for_cancellation"
    synchronizations: list[str] = []
    loads = 0

    def runtime_factory(_: InferConfig) -> InferenceRuntime:
        nonlocal loads
        loads += 1
        return runtime

    engine = ResidentRoiInferenceEngine.load(
        resolved=resolved,
        profile=profile,
        runtime_factory=runtime_factory,
        backend_factory=lambda _: backend,
        cuda_probe=_one_cuda,
        cuda_synchronize=lambda: synchronizations.append("sync"),
    )

    with pytest.raises(ResidentInferenceCancelled) as exc_info:
        engine.infer_one(_request(profile, request_id="cancelled"))

    assert exc_info.value.metadata.reason == "deadline_exceeded"
    assert exc_info.value.metadata.observed_by_backend is True
    assert exc_info.value.metadata.cuda_synchronized is True
    assert synchronizations == ["sync"]
    backend.mode = "success"
    result = engine.infer_one(_request(profile, request_id="after-cancel"))
    assert result.parse.parse_status == "accepted"
    assert loads == 1
    assert backend.calls == 2


def test_backend_that_ignores_cancellation_cannot_claim_safe_cancel() -> None:
    resolved = _resolved_config()
    runtime = _runtime()
    profile = _profile(resolved, runtime, deadline_seconds=1.0)
    backend = _FakeBackend()
    backend.mode = "block"
    synchronizations: list[str] = []
    token = CancellationToken()
    engine = ResidentRoiInferenceEngine.load(
        resolved=resolved,
        profile=profile,
        runtime_factory=lambda _: runtime,
        backend_factory=lambda _: backend,
        cuda_probe=_one_cuda,
        cuda_synchronize=lambda: synchronizations.append("sync"),
        clock=lambda: 0.0,
    )
    errors: list[BaseException] = []

    def run() -> None:
        try:
            engine.infer_one(
                _request(profile, request_id="ignored"),
                cancellation_token=token,
            )
        except BaseException as exc:  # pragma: no cover - asserted below.
            errors.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    assert backend.entered.wait(timeout=1.0)
    token.cancel("client_cancelled")
    backend.release.set()
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeContractError)
    assert errors[0].code == "resident.cancellation_unobserved"
    assert synchronizations == ["sync"]


def test_single_flight_rejects_a_second_request() -> None:
    resolved = _resolved_config()
    runtime = _runtime()
    profile = _profile(resolved, runtime, deadline_seconds=1.0)
    backend = _FakeBackend()
    backend.mode = "block"
    engine = ResidentRoiInferenceEngine.load(
        resolved=resolved,
        profile=profile,
        runtime_factory=lambda _: runtime,
        backend_factory=lambda _: backend,
        cuda_probe=_one_cuda,
    )
    errors: list[BaseException] = []

    def first_request() -> None:
        try:
            engine.infer_one(_request(profile, request_id="first"))
        except BaseException as exc:  # pragma: no cover - asserted below.
            errors.append(exc)

    thread = threading.Thread(target=first_request)
    thread.start()
    assert backend.entered.wait(timeout=1.0)
    try:
        with pytest.raises(RuntimeContractError) as exc_info:
            engine.infer_one(_request(profile, request_id="second"))
        assert exc_info.value.code == "resident.single_flight_busy"
    finally:
        backend.release.set()
        thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert errors == []


def test_canvas_snapshot_is_immutable_and_does_not_retain_mutable_pillow_state() -> (
    None
):
    image = Image.new("RGB", (32, 32), color=(1, 2, 3))
    canvas = ImmutableRgbCanvas.from_image(image)
    image.paste((9, 9, 9), (0, 0, 32, 32))

    restored = canvas.to_image()
    try:
        assert restored.getpixel((0, 0)) == (1, 2, 3)
        assert canvas.to_receipt_dict() == {
            "mode": "RGB",
            "width": 32,
            "height": 32,
            "sha256": canvas.sha256,
        }
    finally:
        restored.close()


def test_resident_source_has_no_offline_pipeline_or_historical_infer_imports() -> None:
    source_path = Path("src/label_studio_coco_refinement/resident_inference.py")
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    pipeline_aliases: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name != "src.infer"
                assert not alias.name.startswith("src.infer.")
                assert alias.name != "src.inference.pipeline"
                if alias.name == "src.inference.pipeline":
                    pipeline_aliases.add(alias.asname or "pipeline")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert module != "src.infer"
            assert not module.startswith("src.infer.")
            assert module != "src.inference.pipeline"
            if module == "src.inference":
                for alias in node.names:
                    assert alias.name != "pipeline"
                    if alias.name == "pipeline":
                        pipeline_aliases.add(alias.asname or alias.name)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if isinstance(node.func.value, ast.Name):
            assert not (
                node.func.value.id in pipeline_aliases and node.func.attr == "run"
            )

    assert "src.inference.pipeline" not in source


def _resolved_config() -> ResolvedInferConfig:
    config = InferConfig.model_validate(
        {
            "schema_version": 1,
            "run": {
                "name": "resident-test",
                "artifact_root": "/tmp/resident-test",
                "collision_policy": "fail",
            },
            "model": {
                "base_model": "/tmp/fake-model",
                "dtype": "bf16",
                "attn_implementation": "eager",
                "processor": {"do_resize": False},
                "runtime_patches": {"patch_embed_linearization": "enabled"},
            },
            "data": {"input_jsonl": "/tmp/unused.jsonl"},
            "template": {
                "object_field_order": "desc_first",
                "object_ordering": "source_order",
                "assistant_format": "object_box_closed",
                "prompt": {"system": "detect", "user": "find every object"},
            },
            "backend": {"type": "hf"},
            "generation": {
                "batch_size": 1,
                "max_new_tokens": 64,
                "temperature": 0.0,
                "top_p": 1.0,
                "repetition_penalty": 1.0,
            },
            "scoring": {"enabled": True},
            "artifacts": {
                "write_token_trace": True,
                "write_parse_diagnostics": True,
            },
            "debug": {"smoke": True, "dry_run": False},
        }
    )
    config_dict = config.model_dump(mode="json")
    return ResolvedInferConfig(
        config=config,
        config_dict=config_dict,
        fingerprint=sha256_json(config_dict),
        schema_version=1,
        loader_version=INFER_CONFIG_LOADER_VERSION,
        entry_config_path=Path("/tmp/resident-test.yaml"),
        sources=(),
        path_origins={},
    )


def _runtime() -> InferenceRuntime:
    processor = _FakeProcessor()
    qwen = SimpleNamespace(
        processor=processor,
        tokenizer=processor.tokenizer,
        model=_FakeModel(),
        config=SimpleNamespace(
            vision_config=SimpleNamespace(
                patch_size=16,
                spatial_merge_size=2,
                temporal_patch_size=2,
            )
        ),
        processor_identity=QwenProcessorIdentity(
            processor_class="FakeProcessor",
            tokenizer_class="FakeTokenizer",
            image_processor_class="FakeImageProcessor",
            patch_size=16,
            merge_size=2,
            temporal_patch_size=2,
        ),
        token_identity=_FakeTokenIdentity(),
        tokenizer_sha256="fake-tokenizer",
    )
    return InferenceRuntime(
        qwen=qwen,
        adapter_receipt=None,
        embedding_delta_receipt=None,
        model_identity={"family": "fake-resident", "base": {"path": "/tmp/fake"}},
    )


def _profile(
    resolved: ResolvedInferConfig,
    runtime: InferenceRuntime,
    *,
    deadline_seconds: float = 0.2,
) -> ResidentProfileBinding:
    template = TemplateConfig(
        object_field_order=resolved.config.template.object_field_order,
        object_ordering=resolved.config.template.object_ordering,
        assistant_format=resolved.config.template.assistant_format,
        prompt=TemplatePromptConfig(
            system=resolved.config.template.prompt.system,
            user=resolved.config.template.prompt.user,
        ),
    )
    tokenizer_identity = runtime.qwen.token_identity.to_artifact_dict()
    return ResidentProfileBinding(
        name="fake-step-917",
        resolved_infer_config_fingerprint=resolved.fingerprint,
        prompt_policy_fingerprint=fingerprint_prompt_policy(template),
        generation_config_fingerprint=sha256_json(
            resolved.config.generation.model_dump(mode="json")
        ),
        runtime_identity_fingerprint=sha256_json(runtime.model_identity),
        processor_identity_fingerprint=sha256_json(
            runtime.qwen.processor_identity.to_artifact_dict()
        ),
        tokenizer_identity_fingerprint=sha256_json(tokenizer_identity),
        transformers_version=str(transformers.__version__),
        processor_kwargs_json=json.dumps(
            {"do_resize": False, "return_tensors": "pt"},
            sort_keys=True,
            separators=(",", ":"),
        ),
        processor_factor=32,
        default_width=96,
        default_height=64,
        min_axis_pixels=32,
        max_axis_pixels=1024,
        max_total_pixels=1_048_576,
        deadline_seconds=deadline_seconds,
    )


def _request(
    profile: ResidentProfileBinding,
    *,
    request_id: str,
) -> ResidentInferenceRequest:
    transform = RoiLetterboxTransform.from_label_studio_roi(
        source_width=96,
        source_height=64,
        roi=(0.0, 0.0, 100.0, 100.0),
        canvas_width=96,
        canvas_height=64,
    )
    source = Image.new("RGB", (96, 64), color=(4, 5, 6))
    canvas_image = transform.prepare_image(source)
    try:
        canvas = ImmutableRgbCanvas.from_image(canvas_image)
    finally:
        source.close()
        canvas_image.close()
    target = RequestTarget(
        request_id=request_id,
        project_id="project-1",
        task_id="task-1",
        task_epoch="epoch-1",
        image_id="image-1",
        annotation_id="annotation-1",
        annotation_revision="revision-1",
        current_user_id="reviewer-1",
        draft_id="draft-1",
        draft_revision="draft-revision-1",
        profile_fingerprint=profile.fingerprint,
        project_generation=0,
        transform_fingerprint=transform.fingerprint,
        preexisting_draft_dirty=False,
    )
    return ResidentInferenceRequest(target=target, transform=transform, canvas=canvas)


def _decode_result(
    *,
    request: Any,
    model_identity: dict[str, Any],
    tokenizer_identity: dict[str, Any],
    generation_config_fingerprint: str,
) -> DecodeResult:
    raw = _PARSER_TEXT + "<|im_end|>"
    return DecodeResult(
        request_id=request.request_id,
        backend="hf",
        backend_mode="generate",
        response_family="hf",
        prompt_token_ids=list(request.prompt_token_ids),
        generated_token_ids=[101],
        raw_generated_text=raw,
        parser_text=_PARSER_TEXT,
        strip_policy="terminal_im_end",
        stop_reason="eos_token",
        model_identity=dict(model_identity),
        tokenizer_identity=dict(tokenizer_identity),
        generation_config_fingerprint=generation_config_fingerprint,
        token_trace=[
            TokenTrace(
                step_index=0,
                token_id=101,
                token_text=raw,
                logprob=-0.1,
                is_stop=True,
                is_pad=False,
                backend="hf",
                backend_mode="generate",
                response_family="hf",
            )
        ],
    )


def _one_cuda() -> CudaBinding:
    return CudaBinding(
        visible_cuda_tokens=("7",),
        cuda_available=True,
        device_count=1,
        current_device=0,
    )
