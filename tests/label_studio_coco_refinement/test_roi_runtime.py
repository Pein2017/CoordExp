from __future__ import annotations

import gc
import json
import weakref
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import transformers
from PIL import Image

from src.config.fingerprint import sha256_json
from src.config.inference import (
    INFER_CONFIG_LOADER_VERSION,
    InferConfig,
    ResolvedInferConfig,
)
from src.inference.parsing import parse_compact_object_box_closed
from src.label_studio_coco_refinement.inference_profiles import (
    EngineProfile,
    EngineProfileStore,
    canonical_json,
    fingerprint_json,
)
from src.label_studio_coco_refinement.inference_results import (
    CurrentTarget,
    RequestTarget,
)
from src.label_studio_coco_refinement.resident_inference import (
    CancellationMetadata,
    ResidentInferenceCancelled,
)
from src.label_studio_coco_refinement.roi_runtime import (
    InferenceReceiptStore,
    ReceiptStoreError,
    RoiInferenceService,
    RoiRuntimeError,
    build_resident_profile_binding,
)
from src.label_studio_coco_refinement.roi_transform import (
    ROI_TRANSFORM_ID,
    RoiLetterboxTransform,
)
from src.templates.renderer import (
    BOX_END_TOKEN,
    BOX_START_TOKEN,
    OBJECT_REF_END_TOKEN,
    OBJECT_REF_START_TOKEN,
)


REQUEST_ID = "12345678-1234-5678-9234-567812345678"


def _object(name: str, coords: tuple[int, int, int, int]) -> str:
    coord_text = "".join(f"<|coord_{value}|>" for value in coords)
    return (
        f"{OBJECT_REF_START_TOKEN}{name}{OBJECT_REF_END_TOKEN}"
        f"{BOX_START_TOKEN}{coord_text}{BOX_END_TOKEN}"
    )


class _ReceiptPart:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def to_receipt_dict(self) -> dict[str, Any]:
        return dict(self.payload)

    def to_artifact_dict(self) -> dict[str, Any]:
        return dict(self.payload)


class _TargetProvider:
    def __init__(self) -> None:
        self.override: dict[str, Any] = {}
        self.calls = 0

    def current_target(self, frozen: RequestTarget) -> CurrentTarget:
        self.calls += 1
        return CurrentTarget(**{**frozen.binding_payload(), **self.override})


class _FakeEngine:
    def __init__(
        self,
        profile: Any,
        *,
        canonical_profile: EngineProfile,
        parser_text: str,
    ) -> None:
        self.profile = profile
        self.resolved = _resolved_from_profile(canonical_profile)
        self.transformers_version = canonical_profile.transformers_version
        self.processor_kwargs = json.loads(canonical_profile.processor_kwargs_json)
        self.runtime = SimpleNamespace(
            model_identity={"family": "fake"},
            qwen=SimpleNamespace(
                processor_identity=_ReceiptPart(
                    {
                        "class": "FakeProcessor",
                        "patch_size": 16,
                        "merge_size": 2,
                    }
                ),
                token_identity=_ReceiptPart({"sha256": "tokenizer"}),
            ),
        )
        self.parser_text = parser_text
        self.raw_generated_text = parser_text + "<|im_end|>"
        self.calls = 0
        self.failure: Exception | None = None
        self.result_reference: weakref.ReferenceType[Any] | None = None

    def infer_one(self, request: Any, *, cancellation_token: Any = None) -> Any:
        del cancellation_token
        self.calls += 1
        if self.failure is not None:
            raise self.failure
        parsed = parse_compact_object_box_closed(
            self.parser_text,
            row_id=request.target.request_id,
            row_index=7,
            image_width=request.canvas.width,
            image_height=request.canvas.height,
        )
        decode = SimpleNamespace(
            backend="hf",
            backend_mode="generate",
            response_family="hf",
            raw_generated_text=self.raw_generated_text,
            parser_text=self.parser_text,
            strip_policy="terminal_im_end",
            stop_reason="eos_token",
            generation_config_fingerprint=self.profile.generation_config_fingerprint,
            model_identity={"family": "fake"},
            tokenizer_identity={"sha256": "tokenizer"},
        )
        result = _FakeResult()
        result.target = request.target
        result.transform = request.transform
        result.profile = self.profile
        result.parse = parsed
        result.decode = decode
        result.raw_generated_text = self.raw_generated_text
        result.parser_text = self.parser_text
        result.image_encoding = SimpleNamespace(tensor=object())
        result.cancellation = _ReceiptPart(
            {
                "requested": False,
                "reason": None,
                "observed_by_backend": False,
                "backend_started": True,
                "cuda_synchronized": False,
                "deadline_seconds": self.profile.deadline_seconds,
            }
        )
        result.cuda_binding = _ReceiptPart(
            {
                "visible_cuda_tokens": ["0"],
                "cuda_available": True,
                "device_count": 1,
                "current_device": 0,
                "logical_device": "cuda:0",
            }
        )
        self.result_reference = weakref.ref(result)
        return result


class _FakeResult:
    pass


def _profile(tmp_path: Path, *, temperature: float = 0.0) -> EngineProfile:
    artifacts = tmp_path / "profile-artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    base = artifacts / "base"
    base.mkdir(exist_ok=True)
    (base / "weights.bin").write_bytes(b"weights")
    model_config = artifacts / "config.json"
    model_config.write_text(
        json.dumps(
            {
                "model_type": "qwen3_vl",
                "vision_config": {"patch_size": 16, "spatial_merge_size": 2},
            }
        ),
        encoding="utf-8",
    )
    tokenizer = artifacts / "tokenizer.json"
    tokenizer.write_text('{"vocab": {"a": 1}}', encoding="utf-8")
    processor = artifacts / "preprocessor_config.json"
    processor.write_text(
        json.dumps({"patch_size": 16, "merge_size": 2}), encoding="utf-8"
    )
    resolved = {
        "schema_version": 1,
        "run": {"name": "roi", "artifact_root": "/ignored", "collision_policy": "fail"},
        "model": {
            "base_model": str(base),
            "dtype": "bf16",
            "attn_implementation": "eager",
            "processor": {"do_resize": False},
            "runtime_patches": {"patch_embed_linearization": "enabled"},
        },
        "data": {"input_jsonl": "/ignored/source.jsonl"},
        "template": {
            "object_field_order": "desc_first",
            "object_ordering": "source_order",
            "assistant_format": "object_box_closed",
            "prompt": {"system": "detect", "user": "find all objects"},
        },
        "backend": {"type": "hf"},
        "generation": {
            "batch_size": 1,
            "max_new_tokens": 64,
            "temperature": temperature,
            "top_p": 1.0,
            "repetition_penalty": 1.0,
        },
        "scoring": {"enabled": True},
        "artifacts": {"write_token_trace": True, "write_parse_diagnostics": True},
        "debug": {"smoke": True, "dry_run": False},
        "roi_inference": {
            "processor_factor": 32,
            "default_width": 64,
            "default_height": 64,
            "min_axis_pixels": 32,
            "max_axis_pixels": 1024,
            "max_total_pixels": 1_048_576,
            "deadline_seconds": 20.0,
        },
    }
    roi_inference = resolved.pop("roi_inference")
    config = InferConfig.model_validate(resolved)
    config_dict = config.model_dump(mode="json")
    resolved_config = ResolvedInferConfig(
        config=config,
        config_dict=config_dict,
        fingerprint=sha256_json(config_dict),
        schema_version=1,
        loader_version=INFER_CONFIG_LOADER_VERSION,
        entry_config_path=tmp_path / "infer.yaml",
        sources=(),
        path_origins={},
    )
    return EngineProfile.capture(
        name="fake-profile",
        endpoint="http://127.0.0.1:8123/infer",
        artifact_paths={
            "base_weights": base,
            "model_config": model_config,
            "processor": processor,
            "tokenizer": tokenizer,
        },
        resolved_config=resolved_config,
        roi_inference=roi_inference,
        parser_identity={
            "id": "compact-object-box-closed-v1",
            "policy": "compact_object_box_closed_only",
        },
        adapter_identity={"id": "coordexp-resident-roi-v1"},
        transform_identity={"id": ROI_TRANSFORM_ID},
        transformers_version=str(transformers.__version__),
        processor_kwargs={"do_resize": False, "return_tensors": "pt"},
        runtime_identity={
            "model": {"family": "fake"},
            "processor": {
                "class": "FakeProcessor",
                "patch_size": 16,
                "merge_size": 2,
            },
            "tokenizer": {"sha256": "tokenizer"},
        },
    )


def _resolved_from_profile(profile: EngineProfile) -> ResolvedInferConfig:
    payload = json.loads(profile.resolved_config_json)
    payload.pop("roi_inference")
    config = InferConfig.model_validate(payload)
    return ResolvedInferConfig(
        config=config,
        config_dict=payload,
        fingerprint=profile.resolved_infer_config_fingerprint,
        schema_version=1,
        loader_version=INFER_CONFIG_LOADER_VERSION,
        entry_config_path=Path("/tmp/fake-roi-infer.yaml"),
        sources=(),
        path_origins={},
    )


def _transform() -> RoiLetterboxTransform:
    return RoiLetterboxTransform.from_label_studio_roi(
        source_width=96,
        source_height=64,
        roi=(0.0, 0.0, 100.0, 100.0),
        canvas_width=64,
        canvas_height=64,
    )


def _target(
    profile: EngineProfile,
    transform: RoiLetterboxTransform,
    *,
    request_id: str = REQUEST_ID,
) -> RequestTarget:
    return RequestTarget(
        request_id=request_id,
        project_id="project-1",
        task_id="task-42",
        task_epoch="epoch-3",
        image_id="42",
        annotation_id="annotation-9",
        annotation_revision="revision-11",
        current_user_id="reviewer-1",
        draft_id="draft-9",
        draft_revision="2026-07-15T00:00:11Z",
        profile_fingerprint=profile.fingerprint,
        project_generation=8,
        transform_fingerprint=transform.fingerprint,
        preexisting_draft_dirty=True,
    )


def _service(
    tmp_path: Path,
    profile: EngineProfile,
    provider: _TargetProvider | None = None,
) -> tuple[RoiInferenceService, InferenceReceiptStore, _TargetProvider]:
    profiles = EngineProfileStore(tmp_path / "profiles.json")
    profiles.save(profile)
    profiles.activate("project-1", profile.name)
    receipts = InferenceReceiptStore(tmp_path / "receipts.jsonl")
    provider = provider or _TargetProvider()
    ticks = iter(index / 1000 for index in range(10_000))
    return (
        RoiInferenceService(
            profiles=profiles,
            receipts=receipts,
            current_targets=provider,
            clock=lambda: next(ticks),
        ),
        receipts,
        provider,
    )


def _run(
    tmp_path: Path,
    *,
    text: str,
    provider: _TargetProvider | None = None,
) -> tuple[dict[str, Any], InferenceReceiptStore, _FakeEngine, EngineProfile]:
    profile = _profile(tmp_path)
    transform = _transform()
    service, receipts, _ = _service(tmp_path, profile, provider)
    engine = _FakeEngine(
        build_resident_profile_binding(profile),
        canonical_profile=profile,
        parser_text=text,
    )
    image = Image.new("RGB", (96, 64), color=(3, 4, 5))
    try:
        response = service.infer(
            image=image,
            target=_target(profile, transform),
            transform=transform,
            engine=engine,
        )
    finally:
        image.close()
    return response, receipts, engine, profile


def test_bridge_preserves_canonical_profile_and_attests_resident_components(
    tmp_path: Path,
) -> None:
    profile = _profile(tmp_path)
    binding = build_resident_profile_binding(profile)
    strict = json.loads(profile.resolved_config_json)
    strict.pop("roi_inference")

    assert binding.fingerprint != profile.fingerprint
    assert binding.resolved_infer_config_fingerprint == sha256_json(strict)
    assert (
        binding.prompt_policy_fingerprint
        == profile.identity_fingerprints["prompt_policy"]
    )
    assert binding.processor_factor == profile.processor_factor == 32
    assert binding.default_width == profile.default_width == 64


def test_bridge_rejects_sampling_policy_not_executed_by_resident_adapter(
    tmp_path: Path,
) -> None:
    profile = _profile(tmp_path, temperature=0.7)

    with pytest.raises(RoiRuntimeError, match="greedy") as exc_info:
        build_resident_profile_binding(profile)

    assert exc_info.value.code == "profile.unsupported_sampling_policy"


@pytest.mark.parametrize(
    ("text", "status", "clear", "inserted"),
    [
        (_object("person", (100, 100, 700, 800)), "accepted", True, 1),
        (
            _object("person", (100, 100, 400, 400))
            + _object("Person", (500, 500, 900, 900)),
            "accepted_with_drops",
            True,
            1,
        ),
        ("", "empty", True, 0),
        (_object("spaceship", (100, 100, 700, 800)), "all_rejected", True, 0),
        ("malformed response", "response_failure", False, 0),
        ('{"objects": []}', "response_failure", False, 0),
    ],
)
def test_every_parser_outcome_is_durable_and_replayable(
    tmp_path: Path,
    text: str,
    status: str,
    clear: bool,
    inserted: int,
) -> None:
    response, receipts, engine, _ = _run(tmp_path, text=text)

    assert response["terminal_status"] == status
    assert response["clear_roi"] is clear
    assert response.get("counts", {}).get("inserted", 0) == inserted
    replayed = receipts.replay()
    assert len(replayed) == 1
    assert replayed[0]["response"] == response
    execution = replayed[0]["execution"]
    assert execution["decode"]["raw_generated_text"] == text + "<|im_end|>"
    assert execution["decode"]["parser_text"] == text
    canonical_result = replayed[0]["attempt"]["result"]
    if canonical_result is not None:
        assert canonical_result["raw_response_text"] == text
    assert not any(
        key in str(execution).lower()
        for key in ("pixel_values", "image_path", "rgb_bytes")
    )
    gc.collect()
    assert engine.result_reference is not None
    assert engine.result_reference() is None


def test_accepted_result_uses_stable_safe_keys_and_resolves_for_commit(
    tmp_path: Path,
) -> None:
    response, receipts, _, profile = _run(
        tmp_path, text=_object("stop sign", (100, 100, 800, 900))
    )

    region = response["insertion_payload"]["regions"][0]
    assert region["region_link"] == f"roi:{REQUEST_ID}:1"
    assert region["category_id"] == 13
    link = receipts.resolve(response["receipt_id"])
    assert link is not None
    assert link.image_id == 42
    assert link.current_user_id == "reviewer-1"
    assert link.draft_id == "draft-9"
    assert link.draft_revision == "2026-07-15T00:00:11Z"
    assert link.result_region_keys == {f"{REQUEST_ID}:result-0": f"roi:{REQUEST_ID}:1"}
    record = receipts.get(response["receipt_id"])
    assert record is not None
    assert record["attempt"]["profile"] == profile.to_receipt_dict()
    assert record["attempt"]["request"]["profile_fingerprint"] == profile.fingerprint
    assert (
        record["execution"]["resident_profile"]["profile_fingerprint"]
        != profile.fingerprint
    )


def test_parser_text_not_raw_generated_text_is_the_strict_replay_authority(
    tmp_path: Path,
) -> None:
    text = _object("cat", (100, 100, 600, 700))
    response, receipts, _, _ = _run(tmp_path, text=text)
    record = receipts.get(response["receipt_id"])
    assert record is not None

    assert record["attempt"]["result"]["raw_response_text"] == text
    assert record["execution"]["decode"]["raw_generated_text"] == text + "<|im_end|>"
    restarted = InferenceReceiptStore(receipts.path)
    replay = restarted.get(response["receipt_id"])
    assert replay == record
    assert replay["attempt"]["result"] == record["attempt"]["result"]


def test_target_is_reread_after_inference_and_mismatch_is_abandoned(
    tmp_path: Path,
) -> None:
    provider = _TargetProvider()
    provider.override = {"annotation_revision": "newer-revision"}
    response, receipts, engine, _ = _run(
        tmp_path,
        text=_object("person", (100, 100, 700, 800)),
        provider=provider,
    )

    assert engine.calls == 1
    assert provider.calls == 1
    assert response["terminal_status"] == "abandoned_before_insertion"
    assert response["insertion_payload"] is None
    assert receipts.resolve(response["receipt_id"]) is None
    record = receipts.get(response["receipt_id"])
    assert record is not None
    assert record["execution"]["decode"]["parser_text"]
    assert record["attempt"]["result"] is None


@pytest.mark.parametrize(
    "override",
    [
        {"current_user_id": "reviewer-2"},
        {"draft_id": "draft-replacement"},
        {"draft_revision": "2026-07-15T00:00:12Z"},
    ],
)
def test_current_user_draft_identity_drift_abandons_before_insertion(
    tmp_path: Path,
    override: dict[str, Any],
) -> None:
    provider = _TargetProvider()
    provider.override = override
    response, receipts, engine, _ = _run(
        tmp_path,
        text=_object("person", (100, 100, 700, 800)),
        provider=provider,
    )

    assert engine.calls == 1
    assert response["terminal_status"] == "abandoned_before_insertion"
    assert response["insertion_payload"] is None
    assert receipts.resolve(response["receipt_id"]) is None


def test_runtime_failure_and_deadline_cancellation_are_durable(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    transform = _transform()
    service, receipts, _ = _service(tmp_path, profile)
    engine = _FakeEngine(
        build_resident_profile_binding(profile),
        canonical_profile=profile,
        parser_text="",
    )
    engine.failure = RuntimeError("boom")
    image = Image.new("RGB", (96, 64))
    try:
        failed = service.infer(
            image=image,
            target=_target(profile, transform),
            transform=transform,
            engine=engine,
        )
    finally:
        image.close()
    assert failed["terminal_status"] == "runtime_failure"
    assert len(receipts.replay()) == 1

    other_root = tmp_path / "deadline"
    deadline_profile = _profile(other_root)
    deadline_transform = _transform()
    deadline_service, deadline_receipts, _ = _service(other_root, deadline_profile)
    deadline_engine = _FakeEngine(
        build_resident_profile_binding(deadline_profile),
        canonical_profile=deadline_profile,
        parser_text="",
    )
    deadline_engine.failure = ResidentInferenceCancelled(
        CancellationMetadata(
            requested=True,
            reason="deadline_exceeded",
            observed_by_backend=True,
            backend_started=True,
            cuda_synchronized=True,
            deadline_seconds=20.0,
        )
    )
    image = Image.new("RGB", (96, 64))
    try:
        timed_out = deadline_service.infer(
            image=image,
            target=_target(deadline_profile, deadline_transform),
            transform=deadline_transform,
            engine=deadline_engine,
        )
    finally:
        image.close()
    assert timed_out["terminal_status"] == "timeout_failure"
    assert len(deadline_receipts.replay()) == 1


def test_profile_and_request_fingerprint_mismatches_fail_closed(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    transform = _transform()
    service, receipts, _ = _service(tmp_path, profile)
    wrong_engine = _FakeEngine(
        replace(build_resident_profile_binding(profile), max_total_pixels=524_288),
        canonical_profile=profile,
        parser_text="",
    )
    image = Image.new("RGB", (96, 64))
    try:
        response = service.infer(
            image=image,
            target=_target(profile, transform),
            transform=transform,
            engine=wrong_engine,
        )
    finally:
        image.close()
    assert response["terminal_status"] == "profile_failure"
    assert wrong_engine.calls == 0
    assert len(receipts.replay()) == 1

    image = Image.new("RGB", (96, 64))
    with pytest.raises(RoiRuntimeError, match="active canonical"):
        try:
            service.infer(
                image=image,
                target=replace(
                    _target(profile, transform), profile_fingerprint="f" * 64
                ),
                transform=transform,
                engine=wrong_engine,
            )
        finally:
            image.close()


@pytest.mark.parametrize("field", ["transformers_version", "processor_kwargs"])
def test_loaded_execution_policy_mismatch_fails_before_generation(
    tmp_path: Path,
    field: str,
) -> None:
    profile = _profile(tmp_path)
    transform = _transform()
    service, receipts, _ = _service(tmp_path, profile)
    engine = _FakeEngine(
        build_resident_profile_binding(profile),
        canonical_profile=profile,
        parser_text="",
    )
    if field == "transformers_version":
        engine.transformers_version = "0.0.invalid"
    else:
        engine.processor_kwargs = {"do_resize": False}
    image = Image.new("RGB", (96, 64))
    try:
        response = service.infer(
            image=image,
            target=_target(profile, transform),
            transform=transform,
            engine=engine,
        )
    finally:
        image.close()

    assert response["terminal_status"] == "profile_failure"
    assert engine.calls == 0
    assert len(receipts.replay()) == 1


def test_artifact_drift_fails_before_engine_and_is_durably_receipted(
    tmp_path: Path,
) -> None:
    profile = _profile(tmp_path)
    transform = _transform()
    service, receipts, _ = _service(tmp_path, profile)
    engine = _FakeEngine(
        build_resident_profile_binding(profile),
        canonical_profile=profile,
        parser_text="",
    )
    Path(profile.artifacts[0].path, "drift.bin").write_bytes(b"drift")
    image = Image.new("RGB", (96, 64))
    try:
        response = service.infer(
            image=image,
            target=_target(profile, transform),
            transform=transform,
            engine=engine,
        )
    finally:
        image.close()

    assert response["terminal_status"] == "profile_failure"
    assert engine.calls == 0
    record = receipts.get(response["receipt_id"])
    assert record is not None
    assert record["attempt"]["profile"] == profile.to_receipt_dict()


def test_service_accepts_no_path_url_or_non_rgb_image_api(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    transform = _transform()
    service, _, _ = _service(tmp_path, profile)
    engine = _FakeEngine(
        build_resident_profile_binding(profile),
        canonical_profile=profile,
        parser_text="",
    )
    target = _target(profile, transform)

    for invalid in (Path("/tmp/image.jpg"), "file:///tmp/image.jpg", "https://x/y.jpg"):
        with pytest.raises(RoiRuntimeError, match="Pillow RGB"):
            service.infer(
                image=invalid,  # type: ignore[arg-type]
                target=target,
                transform=transform,
                engine=engine,
            )
    grayscale = Image.new("L", (96, 64))
    try:
        with pytest.raises(RoiRuntimeError, match="Pillow RGB"):
            service.infer(
                image=grayscale,
                target=target,
                transform=transform,
                engine=engine,
            )
    finally:
        grayscale.close()
    assert engine.calls == 0


def test_transform_prepares_once_and_idempotent_restart_does_not_reexecute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = _profile(tmp_path)
    transform = _transform()
    service, receipts, provider = _service(tmp_path, profile)
    engine = _FakeEngine(
        build_resident_profile_binding(profile),
        canonical_profile=profile,
        parser_text=_object("person", (100, 100, 700, 800)),
    )
    calls = 0
    original = RoiLetterboxTransform.prepare_image

    def counted(self: RoiLetterboxTransform, image: Image.Image) -> Image.Image:
        nonlocal calls
        calls += 1
        return original(self, image)

    monkeypatch.setattr(RoiLetterboxTransform, "prepare_image", counted)
    image = Image.new("RGB", (96, 64), color=(9, 8, 7))
    target = _target(profile, transform)
    try:
        first = service.infer(
            image=image, target=target, transform=transform, engine=engine
        )
        restarted_service = RoiInferenceService(
            profiles=service.profiles,
            receipts=InferenceReceiptStore(receipts.path),
            current_targets=provider,
            clock=lambda: 1.0,
        )
        second = restarted_service.infer(
            image=image, target=target, transform=transform, engine=engine
        )
    finally:
        image.close()

    assert first == second
    assert engine.calls == 1
    assert calls == 2  # once per HTTP attempt, never twice in one attempt
    assert len(receipts.replay()) == 1


def test_request_id_conflict_on_different_canvas_fails_closed(tmp_path: Path) -> None:
    response, receipts, engine, profile = _run(
        tmp_path, text=_object("person", (100, 100, 700, 800))
    )
    assert response["terminal_status"] == "accepted"
    service = RoiInferenceService(
        profiles=EngineProfileStore(tmp_path / "profiles.json"),
        receipts=receipts,
        current_targets=_TargetProvider(),
        clock=lambda: 1.0,
    )
    image = Image.new("RGB", (96, 64), color=(200, 1, 1))
    try:
        with pytest.raises(ReceiptStoreError, match="differs"):
            service.infer(
                image=image,
                target=_target(profile, _transform()),
                transform=_transform(),
                engine=engine,
            )
    finally:
        image.close()
    assert engine.calls == 1


def test_receipt_store_detects_tamper_and_torn_tail_on_restart(tmp_path: Path) -> None:
    _, receipts, _, _ = _run(tmp_path, text=_object("person", (100, 100, 700, 800)))
    original = receipts.path.read_bytes()
    receipts.path.write_bytes(original.replace(b'"accepted"', b'"acceptXd"', 1))
    with pytest.raises(ReceiptStoreError, match="hash chain"):
        InferenceReceiptStore(receipts.path)

    receipts.path.write_bytes(original[:-1])
    with pytest.raises(ReceiptStoreError, match="torn"):
        InferenceReceiptStore(receipts.path)


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong_request",
        "wrong_status",
        "wrong_count_type",
        "wrong_count_value",
        "negative_count",
        "incomplete_counts",
        "clear_roi_type",
        "insertion_type",
        "forged_insertion",
        "extra_field",
        "credential_field",
        "failure_credential",
    ],
)
def test_resigned_service_response_cannot_diverge_from_reconstructed_attempt(
    tmp_path: Path,
    mutation: str,
) -> None:
    _, receipts, _, _ = _run(
        tmp_path,
        text=_object("person", (100, 100, 700, 800)),
    )
    entry = json.loads(receipts.path.read_text(encoding="utf-8"))
    response = entry["record"]["response"]
    if mutation == "wrong_request":
        response["request_id"] = "12345678-1234-5678-9234-567812345679"
    elif mutation == "wrong_status":
        response["terminal_status"] = "empty"
    elif mutation == "wrong_count_type":
        response["counts"]["inserted"] = "1"
    elif mutation == "wrong_count_value":
        response["counts"]["inserted"] = 9
    elif mutation == "negative_count":
        response["counts"]["rejected"] = -1
    elif mutation == "incomplete_counts":
        response["counts"].pop("parsed")
    elif mutation == "clear_roi_type":
        response["clear_roi"] = {}
    elif mutation == "insertion_type":
        response["insertion_payload"] = []
    elif mutation == "forged_insertion":
        response["insertion_payload"]["regions"][0]["category_id"] = 18
    elif mutation == "extra_field":
        response["unexpected"] = True
    elif mutation == "credential_field":
        response["authorization"] = "Bearer forged"
    else:
        response["failure"] = {"authorization": "Bearer forged"}
    _resign_single_entry(receipts.path, entry)

    error = (
        "credential-bearing"
        if mutation in {"credential_field", "failure_credential"}
        else "exactly match the reconstructed"
    )
    with pytest.raises(ReceiptStoreError, match=error):
        InferenceReceiptStore(receipts.path)


def test_resigned_failure_response_shape_is_derived_from_terminal_attempt(
    tmp_path: Path,
) -> None:
    profile = _profile(tmp_path)
    transform = _transform()
    service, receipts, _ = _service(tmp_path, profile)
    engine = _FakeEngine(
        build_resident_profile_binding(profile),
        canonical_profile=profile,
        parser_text="",
    )
    engine.failure = RuntimeError("boom")
    image = Image.new("RGB", (96, 64))
    try:
        service.infer(
            image=image,
            target=_target(profile, transform),
            transform=transform,
            engine=engine,
        )
    finally:
        image.close()
    entry = json.loads(receipts.path.read_text(encoding="utf-8"))
    entry["record"]["response"]["counts"] = {
        "parsed": 0,
        "inserted": 0,
        "rejected": 0,
    }
    _resign_single_entry(receipts.path, entry)

    with pytest.raises(ReceiptStoreError, match="exactly match the reconstructed"):
        InferenceReceiptStore(receipts.path)


def _resign_single_entry(path: Path, entry: dict[str, Any]) -> None:
    unsigned = {key: value for key, value in entry.items() if key != "entry_sha256"}
    entry["entry_sha256"] = fingerprint_json(unsigned)
    path.write_text(canonical_json(entry) + "\n", encoding="utf-8")


def test_two_store_instances_serialize_append_with_one_valid_hash_chain(
    tmp_path: Path,
) -> None:
    profile = _profile(tmp_path)
    transform = _transform()
    profiles = EngineProfileStore(tmp_path / "profiles.json")
    profiles.save(profile)
    profiles.activate("project-1", profile.name)
    receipt_path = tmp_path / "receipts.jsonl"

    def run(request_id: str) -> dict[str, Any]:
        service = RoiInferenceService(
            profiles=profiles,
            receipts=InferenceReceiptStore(receipt_path),
            current_targets=_TargetProvider(),
            clock=iter(index / 1000 for index in range(10_000)).__next__,
        )
        engine = _FakeEngine(
            build_resident_profile_binding(profile),
            canonical_profile=profile,
            parser_text=_object("person", (100, 100, 700, 800)),
        )
        image = Image.new("RGB", (96, 64), color=(3, 4, 5))
        try:
            return service.infer(
                image=image,
                target=_target(profile, transform, request_id=request_id),
                transform=transform,
                engine=engine,
            )
        finally:
            image.close()

    ids = (
        "12345678-1234-5678-9234-567812345671",
        "12345678-1234-5678-9234-567812345672",
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(run, ids))

    assert {item["request_id"] for item in responses} == set(ids)
    replay = InferenceReceiptStore(receipt_path).replay()
    assert len(replay) == 2
    assert {item["request_id"] for item in replay} == set(ids)


def test_nonaccepted_receipt_never_resolves_result_links(tmp_path: Path) -> None:
    response, receipts, _, _ = _run(tmp_path, text="")

    assert response["terminal_status"] == "empty"
    assert receipts.resolve(response["receipt_id"]) is None
