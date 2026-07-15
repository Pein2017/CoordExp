"""Target-bound ROI orchestration with append-only durable inference receipts.

The persisted :class:`EngineProfile` is the product authority.  The resident
profile is a narrowly derived execution binding and is never substituted for
the canonical profile in request or attempt receipts.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import uuid
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, Protocol

from PIL import Image

from src.config.fingerprint import sha256_json
from src.config.inference import InferConfig, ResolvedInferConfig
from src.config.models import TemplateConfig, TemplatePromptConfig
from src.inference.parsing import (
    PARSER_ID,
    PARSER_POLICY,
    parse_compact_object_box_closed,
)
from src.inference.prompt import fingerprint_prompt_policy
from src.label_studio_coco_refinement.inference_profiles import (
    EngineProfile,
    EngineProfileStore,
    canonical_json,
    fingerprint_json,
)
from src.label_studio_coco_refinement.inference_results import (
    ClassifiedInferenceResult,
    CurrentTarget,
    InferenceAttemptReceipt,
    RequestLifecycle,
    RequestState,
    RequestTarget,
    bind_for_insertion,
    classify_parser_result,
    finalize_region_links,
    terminal_state_for_result,
)
from src.label_studio_coco_refinement.resident_inference import (
    CancellationToken,
    ImmutableRgbCanvas,
    ResidentInferenceCancelled,
    ResidentInferenceRequest,
    ResidentProfileBinding,
)
from src.label_studio_coco_refinement.roi_transform import (
    ROI_TRANSFORM_ID,
    RoiLetterboxTransform,
)
from src.label_studio_coco_refinement.store import InferenceReceiptLink
from src.qwen.images import QWEN_IMAGE_PROCESSOR_KWARGS


RECEIPT_STORE_SCHEMA = "coordexp-roi-receipt-chain-v1"
RECEIPT_RECORD_SCHEMA = "coordexp-roi-attempt-record-v1"
EXECUTION_ENVELOPE_SCHEMA = "coordexp-resident-roi-execution-v1"
RESIDENT_ADAPTER_ID = "coordexp-resident-roi-v1"
_ZERO_HASH = "0" * 64
_ACCEPTED_STATUSES = {"accepted", "accepted_with_drops"}


class RoiRuntimeError(RuntimeError):
    """A request cannot safely cross the ROI orchestration boundary."""

    def __init__(self, message: str, *, code: str, stage: str) -> None:
        self.code = code
        self.stage = stage
        super().__init__(message)


class ReceiptStoreError(RoiRuntimeError):
    """The append-only receipt authority is corrupt or conflicting."""


class CurrentTargetProvider(Protocol):
    """Re-read the mutable browser/annotation target immediately before insert."""

    def current_target(self, frozen: RequestTarget) -> CurrentTarget: ...


class ResidentEngine(Protocol):
    profile: ResidentProfileBinding

    def infer_one(
        self,
        request: ResidentInferenceRequest,
        *,
        cancellation_token: CancellationToken | None = None,
    ) -> Any: ...


def build_resident_profile_binding(profile: EngineProfile) -> ResidentProfileBinding:
    """Derive the exact resident bridge from explicit canonical identity fields.

    No artifact path or live runtime object is inspected to invent component
    identity.  The canonical runtime identity must explicitly provide the
    model, processor, and tokenizer identity payloads that the resident loader
    fingerprints after load.
    """

    if not isinstance(profile, EngineProfile):
        raise RoiRuntimeError(
            "resident bridge requires a canonical EngineProfile",
            code="profile.type",
            stage="profile",
        )
    resolved = _json_mapping(profile.resolved_config_json, field="resolved_config")
    prompt = _json_mapping(profile.prompt_policy_json, field="prompt_policy")
    parser = _json_mapping(profile.parser_identity_json, field="parser_identity")
    adapter = _json_mapping(profile.adapter_identity_json, field="adapter_identity")
    transform = _json_mapping(
        profile.transform_identity_json, field="transform_identity"
    )
    runtime = _json_mapping(profile.runtime_identity_json, field="runtime_identity")

    if parser != {"id": PARSER_ID, "policy": PARSER_POLICY}:
        raise RoiRuntimeError(
            "canonical parser identity does not match the current resident parser",
            code="profile.parser_identity",
            stage="profile",
        )
    if adapter != {"id": RESIDENT_ADAPTER_ID}:
        raise RoiRuntimeError(
            "canonical adapter identity does not match the resident ROI adapter",
            code="profile.adapter_identity",
            stage="profile",
        )
    if transform != {"id": ROI_TRANSFORM_ID}:
        raise RoiRuntimeError(
            "canonical transform identity does not match the ROI transform",
            code="profile.transform_identity",
            stage="profile",
        )
    if set(runtime) != {"model", "processor", "tokenizer"} or any(
        not isinstance(runtime[field], dict) for field in runtime
    ):
        raise RoiRuntimeError(
            "runtime_identity must explicitly contain only model/processor/tokenizer mappings",
            code="profile.runtime_identity",
            stage="profile",
        )
    strict_resolved = dict(resolved)
    roi = strict_resolved.pop("roi_inference", None)
    if not isinstance(roi, dict):
        raise RoiRuntimeError(
            "resolved profile is missing the explicit roi_inference sidecar",
            code="profile.roi_sidecar",
            stage="profile",
        )
    generation = strict_resolved.get("generation")
    if not isinstance(generation, dict):
        raise RoiRuntimeError(
            "resolved profile is missing generation identity",
            code="profile.generation_identity",
            stage="profile",
        )
    _validate_executed_generation_policy(generation)
    try:
        strict_config = InferConfig.model_validate(strict_resolved)
    except Exception as exc:
        raise RoiRuntimeError(
            "canonical profile does not contain a strict InferConfig",
            code="profile.strict_config",
            stage="profile",
        ) from exc
    if strict_config.model_dump(mode="json") != strict_resolved:
        raise RoiRuntimeError(
            "canonical profile InferConfig omitted defaults or contains unknown fields",
            code="profile.strict_config_canonical",
            stage="profile",
        )
    if sha256_json(strict_resolved) != profile.resolved_infer_config_fingerprint:
        raise RoiRuntimeError(
            "canonical profile resolved InferConfig fingerprint is inconsistent",
            code="profile.resolved_fingerprint",
            stage="profile",
        )
    expected_prompt_fingerprint = profile.identity_fingerprints["prompt_policy"]
    if fingerprint_json(prompt) != expected_prompt_fingerprint:
        raise RoiRuntimeError(
            "prompt component fingerprint is inconsistent",
            code="profile.prompt_fingerprint",
            stage="profile",
        )
    executed_prompt_fingerprint = fingerprint_prompt_policy(
        _template_config(strict_config)
    )
    if expected_prompt_fingerprint != executed_prompt_fingerprint:
        raise RoiRuntimeError(
            "canonical prompt identity does not match the executed prompt policy",
            code="profile.prompt_execution_mismatch",
            stage="profile",
        )
    expected_bounds = {
        "processor_factor": profile.processor_factor,
        "default_width": profile.default_width,
        "default_height": profile.default_height,
        "min_axis_pixels": profile.min_axis_pixels,
        "max_axis_pixels": profile.max_axis_pixels,
        "max_total_pixels": profile.max_total_pixels,
        "deadline_seconds": profile.deadline_seconds,
    }
    if roi != expected_bounds:
        raise RoiRuntimeError(
            "resident bounds do not exactly match the canonical EngineProfile",
            code="profile.bounds_mismatch",
            stage="profile",
        )
    return ResidentProfileBinding(
        name=profile.name,
        resolved_infer_config_fingerprint=profile.resolved_infer_config_fingerprint,
        prompt_policy_fingerprint=expected_prompt_fingerprint,
        generation_config_fingerprint=sha256_json(generation),
        runtime_identity_fingerprint=sha256_json(runtime["model"]),
        processor_identity_fingerprint=sha256_json(runtime["processor"]),
        tokenizer_identity_fingerprint=sha256_json(runtime["tokenizer"]),
        transformers_version=profile.transformers_version,
        processor_kwargs_json=profile.processor_kwargs_json,
        processor_factor=profile.processor_factor,
        default_width=profile.default_width,
        default_height=profile.default_height,
        min_axis_pixels=profile.min_axis_pixels,
        max_axis_pixels=profile.max_axis_pixels,
        max_total_pixels=profile.max_total_pixels,
        deadline_seconds=profile.deadline_seconds,
    )


class InferenceReceiptStore:
    """Append-only, flock-serialized, hash-chained terminal attempt authority."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.replay()

    def replay(self) -> tuple[dict[str, Any], ...]:
        if not self.path.exists():
            return ()
        with self.path.open("rb") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
            try:
                return tuple(self._read_locked(handle))
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def by_request(self, request_id: str) -> dict[str, Any] | None:
        for record in self.replay():
            if record["request_id"] == request_id:
                return _json_copy(record)
        return None

    def get(self, receipt_id: str) -> dict[str, Any] | None:
        for record in self.replay():
            if record["receipt_id"] == receipt_id:
                return _json_copy(record)
        return None

    def append(
        self,
        *,
        attempt: InferenceAttemptReceipt,
        execution: Mapping[str, Any],
        response: Mapping[str, Any],
    ) -> str:
        receipt_id = _receipt_id(attempt.target.request_id)
        record = {
            "schema_version": RECEIPT_RECORD_SCHEMA,
            "receipt_id": receipt_id,
            "request_id": attempt.target.request_id,
            "attempt": attempt.to_dict(),
            "execution": _validate_execution_envelope(execution),
            "response": _strict_service_response(
                response,
                receipt_id=receipt_id,
                attempt=attempt,
            ),
        }
        # Prove replayability before publishing the terminal record.
        _validate_attempt_record(record)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        created = not self.path.exists()
        with self.path.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                records = self._read_locked(handle)
                for prior in records:
                    if prior["request_id"] != record["request_id"]:
                        continue
                    if canonical_json(prior) == canonical_json(record):
                        return receipt_id
                    raise ReceiptStoreError(
                        "request ID already has a different durable terminal receipt",
                        code="receipt.request_conflict",
                        stage="receipt",
                    )
                previous_hash = _ZERO_HASH
                sequence = 0
                if records:
                    handle.seek(0)
                    entries = _decode_entries(handle.read())
                    sequence = len(entries)
                    previous_hash = entries[-1]["entry_sha256"]
                unsigned = {
                    "schema_version": RECEIPT_STORE_SCHEMA,
                    "sequence": sequence,
                    "previous_entry_sha256": previous_hash,
                    "record": record,
                }
                entry = {**unsigned, "entry_sha256": fingerprint_json(unsigned)}
                handle.seek(0, os.SEEK_END)
                handle.write((canonical_json(entry) + "\n").encode("utf-8"))
                handle.flush()
                os.fsync(handle.fileno())
                _fsync_directory(self.path.parent)
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        if created:
            _fsync_directory(self.path.parent)
        return receipt_id

    def resolve(self, receipt_id: str) -> InferenceReceiptLink | None:
        record = self.get(receipt_id)
        if record is None:
            return None
        attempt = _attempt_from_record(record)
        status = attempt.lifecycle.state.value
        if status not in _ACCEPTED_STATUSES or attempt.result is None:
            return None
        payload = attempt.result.insertion_payload
        if payload is None:
            return None
        image_id = attempt.target.image_id
        if not image_id.isdecimal():
            raise ReceiptStoreError(
                "accepted receipt image_id is not a decimal dataset identity",
                code="receipt.image_id",
                stage="receipt",
            )
        links = {
            region.result_id: region.region_link
            for region in payload.regions
            if region.region_link is not None
        }
        if len(links) != len(payload.regions):
            raise ReceiptStoreError(
                "accepted receipt is missing finalized region links",
                code="receipt.region_links",
                stage="receipt",
            )
        return InferenceReceiptLink(
            receipt_id=receipt_id,
            request_id=attempt.target.request_id,
            project_id=attempt.target.project_id,
            task_id=attempt.target.task_id,
            image_id=int(image_id),
            annotation_id=attempt.target.annotation_id,
            current_user_id=attempt.target.current_user_id,
            draft_id=attempt.target.draft_id,
            draft_revision=attempt.target.draft_revision,
            terminal_status=status,
            result_region_keys=links,
        )

    def _read_locked(self, handle: Any) -> list[dict[str, Any]]:
        handle.seek(0)
        entries = _decode_entries(handle.read())
        previous = _ZERO_HASH
        records: list[dict[str, Any]] = []
        requests: set[str] = set()
        receipts: set[str] = set()
        for sequence, entry in enumerate(entries):
            if set(entry) != {
                "schema_version",
                "sequence",
                "previous_entry_sha256",
                "record",
                "entry_sha256",
            }:
                raise _receipt_corrupt("entry fields are not allowlisted")
            unsigned = {key: entry[key] for key in entry if key != "entry_sha256"}
            if (
                entry["schema_version"] != RECEIPT_STORE_SCHEMA
                or entry["sequence"] != sequence
                or entry["previous_entry_sha256"] != previous
                or entry["entry_sha256"] != fingerprint_json(unsigned)
            ):
                raise _receipt_corrupt("receipt hash chain does not replay")
            record = _validate_attempt_record(entry["record"])
            if record["request_id"] in requests or record["receipt_id"] in receipts:
                raise _receipt_corrupt("receipt request and receipt IDs must be unique")
            requests.add(record["request_id"])
            receipts.add(record["receipt_id"])
            records.append(record)
            previous = entry["entry_sha256"]
        return records


class RoiInferenceService:
    """Run one target-bound resident request and durably publish its outcome."""

    def __init__(
        self,
        *,
        profiles: EngineProfileStore,
        receipts: InferenceReceiptStore,
        current_targets: CurrentTargetProvider,
        clock: Any,
    ) -> None:
        self.profiles = profiles
        self.receipts = receipts
        self.current_targets = current_targets
        self.clock = clock

    def infer(
        self,
        *,
        image: Image.Image,
        target: RequestTarget,
        transform: RoiLetterboxTransform,
        engine: ResidentEngine,
        cancellation_token: CancellationToken | None = None,
    ) -> dict[str, Any]:
        """Return only ordinary JSON after its terminal receipt is durable."""

        if not isinstance(target, RequestTarget):
            raise RoiRuntimeError(
                "target must be a validated RequestTarget",
                code="request.target_type",
                stage="request",
            )
        _canonical_request_uuid(target.request_id)
        if not target.image_id.isdecimal():
            raise RoiRuntimeError(
                "target image_id must be a decimal dataset identity",
                code="request.image_id",
                stage="request",
            )
        if not isinstance(transform, RoiLetterboxTransform):
            raise RoiRuntimeError(
                "transform must be a validated RoiLetterboxTransform",
                code="request.transform_type",
                stage="request",
            )
        if target.transform_fingerprint != transform.fingerprint:
            raise RoiRuntimeError(
                "request target transform fingerprint does not match",
                code="request.transform_mismatch",
                stage="request",
            )
        if not isinstance(image, Image.Image) or image.mode != "RGB":
            raise RoiRuntimeError(
                "only vendor-resolved Pillow RGB images are accepted",
                code="request.image_boundary",
                stage="request",
            )
        if image.size != (transform.source_width, transform.source_height):
            raise RoiRuntimeError(
                "resolved image dimensions do not match the frozen transform",
                code="request.image_dimensions",
                stage="request",
            )

        # Load the immutable persisted authority first. Artifact verification is
        # part of the durable attempt below so drift gets a terminal receipt.
        profile = self.profiles.active(target.project_id, verify=False)
        if target.profile_fingerprint != profile.fingerprint:
            raise RoiRuntimeError(
                "request target is not bound to the active canonical EngineProfile",
                code="request.profile_mismatch",
                stage="request",
            )
        profile.validate_canvas(transform.canvas_width, transform.canvas_height)

        # Pixel preparation has one owner and executes exactly once.
        prepared = transform.prepare_image(image)
        try:
            canvas = ImmutableRgbCanvas.from_image(prepared)
        finally:
            prepared.close()

        prior = self.receipts.by_request(target.request_id)
        if prior is not None:
            _validate_retry(prior, target=target, transform=transform, canvas=canvas)
            return _json_copy(prior["response"])

        started = float(self.clock())
        lifecycle = RequestLifecycle()
        profile_receipt = profile.to_receipt_dict()
        transform_receipt = transform.to_receipt_dict()
        try:
            profile.verify_artifacts()
            binding = build_resident_profile_binding(profile)
            if (
                not isinstance(engine.profile, ResidentProfileBinding)
                or engine.profile != binding
            ):
                raise RoiRuntimeError(
                    "loaded resident engine does not match the canonical profile bridge",
                    code="profile.resident_binding_mismatch",
                    stage="profile",
                )
            _attest_loaded_engine(
                engine=engine,
                profile=profile,
                binding=binding,
            )
        except Exception as exc:
            terminal = lifecycle.transition(
                RequestState.PROFILE_FAILURE,
                at_seconds=_elapsed(self.clock, started),
                reason=_failure_code(exc, default="profile.invalid"),
            )
            return self._persist_failure(
                target=target,
                lifecycle=terminal,
                profile_receipt=profile_receipt,
                transform_receipt=transform_receipt,
                execution=_failure_execution(
                    canonical_profile=profile,
                    canvas=canvas,
                    stage="profile",
                    code=_failure_code(exc, default="profile.invalid"),
                ),
                stage="profile",
                code=_failure_code(exc, default="profile.invalid"),
            )

        lifecycle = lifecycle.transition(
            RequestState.RUNNING, at_seconds=_elapsed(self.clock, started)
        )
        resident_target = replace(target, profile_fingerprint=binding.fingerprint)
        resident_request = ResidentInferenceRequest(
            target=resident_target,
            transform=transform,
            canvas=canvas,
        )
        execution: dict[str, Any] | None = None
        classified: ClassifiedInferenceResult | None = None
        try:
            resident_result = engine.infer_one(
                resident_request,
                cancellation_token=cancellation_token,
            )
            try:
                _validate_resident_result(
                    resident_result,
                    target=resident_target,
                    transform=transform,
                    binding=binding,
                )
                execution = _execution_envelope(
                    canonical_profile=profile,
                    binding=binding,
                    canvas=canvas,
                    result=resident_result,
                )
                # parser_text is the exact replay input; raw_generated_text is
                # retained separately in the execution envelope.
                classified = classify_parser_result(
                    parse_row=resident_result.parse,
                    raw_response_text=resident_result.parser_text,
                    target=target,
                    transform=transform,
                )
            finally:
                # Drop the sole result reference holding Qwen tensor encodings.
                del resident_result
        except ResidentInferenceCancelled as exc:
            reason = exc.metadata.reason or "cancelled"
            cancelling = lifecycle.transition(
                RequestState.CANCELLING,
                at_seconds=_elapsed(self.clock, started),
                reason=reason,
            )
            state = (
                RequestState.TIMEOUT_FAILURE
                if reason == "deadline_exceeded"
                else RequestState.CANCELLED
            )
            terminal = cancelling.transition(
                state,
                at_seconds=_elapsed(self.clock, started),
                reason=reason,
            )
            return self._persist_failure(
                target=target,
                lifecycle=terminal,
                profile_receipt=profile_receipt,
                transform_receipt=transform_receipt,
                execution=_failure_execution(
                    canonical_profile=profile,
                    binding=binding,
                    canvas=canvas,
                    stage="deadline"
                    if state is RequestState.TIMEOUT_FAILURE
                    else "cancel",
                    code=reason,
                ),
                stage="deadline" if state is RequestState.TIMEOUT_FAILURE else "cancel",
                code=reason,
            )
        except Exception as exc:
            code = _failure_code(exc, default="resident.runtime_failure")
            terminal = lifecycle.transition(
                RequestState.RUNTIME_FAILURE,
                at_seconds=_elapsed(self.clock, started),
                reason=code,
            )
            return self._persist_failure(
                target=target,
                lifecycle=terminal,
                profile_receipt=profile_receipt,
                transform_receipt=transform_receipt,
                execution=(
                    execution
                    if execution is not None
                    else _failure_execution(
                        canonical_profile=profile,
                        binding=binding,
                        canvas=canvas,
                        stage="runtime",
                        code=code,
                    )
                ),
                stage="runtime",
                code=code,
            )

        assert classified is not None
        assert execution is not None
        try:
            current = self.current_targets.current_target(target)
            decision = bind_for_insertion(classified, current)
        except Exception as exc:
            code = _failure_code(exc, default="target.lookup_failed")
            terminal = lifecycle.transition(
                RequestState.RUNTIME_FAILURE,
                at_seconds=_elapsed(self.clock, started),
                reason=code,
            )
            return self._persist_failure(
                target=target,
                lifecycle=terminal,
                profile_receipt=profile_receipt,
                transform_receipt=transform_receipt,
                execution=execution,
                stage="target",
                code=code,
            )
        if decision.status != "bound":
            code = "target.mismatch:" + ",".join(decision.mismatches)
            terminal = lifecycle.transition(
                RequestState.ABANDONED_BEFORE_INSERTION,
                at_seconds=_elapsed(self.clock, started),
                reason=code,
            )
            return self._persist_failure(
                target=target,
                lifecycle=terminal,
                profile_receipt=profile_receipt,
                transform_receipt=transform_receipt,
                execution=execution,
                stage="target",
                code="target.mismatch",
            )

        if classified.insertion_payload is not None:
            links = {
                region.result_id: f"roi:{target.request_id}:{ordinal}"
                for ordinal, region in enumerate(
                    classified.insertion_payload.regions, start=1
                )
            }
            classified = finalize_region_links(
                classified,
                current,
                region_links=links,
            )
        terminal = lifecycle.transition(
            terminal_state_for_result(classified),
            at_seconds=_elapsed(self.clock, started),
        )
        attempt = InferenceAttemptReceipt(
            target=target,
            lifecycle=terminal,
            profile_receipt=profile_receipt,
            transform_receipt=transform_receipt,
            result=classified,
        )
        receipt_id = _receipt_id(target.request_id)
        response = _result_response(
            receipt_id=receipt_id,
            result=classified,
        )
        self.receipts.append(
            attempt=attempt,
            execution=execution,
            response=response,
        )
        del resident_request, canvas
        return response

    def _persist_failure(
        self,
        *,
        target: RequestTarget,
        lifecycle: RequestLifecycle,
        profile_receipt: Mapping[str, Any],
        transform_receipt: Mapping[str, Any],
        execution: Mapping[str, Any],
        stage: str,
        code: str,
    ) -> dict[str, Any]:
        attempt = InferenceAttemptReceipt(
            target=target,
            lifecycle=lifecycle,
            profile_receipt=profile_receipt,
            transform_receipt=transform_receipt,
            result=None,
            failure_stage=stage,
            failure_code=code,
        )
        receipt_id = _receipt_id(target.request_id)
        response = {
            "receipt_id": receipt_id,
            "request_id": target.request_id,
            "terminal_status": lifecycle.state.value,
            "clear_roi": False,
            "insertion_payload": None,
            "failure": {"stage": stage, "code": code},
        }
        self.receipts.append(attempt=attempt, execution=execution, response=response)
        return response


def _execution_envelope(
    *,
    canonical_profile: EngineProfile,
    binding: ResidentProfileBinding,
    canvas: ImmutableRgbCanvas,
    result: Any,
) -> dict[str, Any]:
    decode = result.decode
    parse = result.parse
    envelope = {
        "schema_version": EXECUTION_ENVELOPE_SCHEMA,
        "canonical_profile_fingerprint": canonical_profile.fingerprint,
        "resident_profile": binding.to_receipt_dict(),
        "canvas": canvas.to_receipt_dict(),
        "decode": {
            "backend": decode.backend,
            "backend_mode": decode.backend_mode,
            "response_family": decode.response_family,
            "raw_generated_text": result.raw_generated_text,
            "raw_generated_sha256": _sha256_text(result.raw_generated_text),
            "parser_text": result.parser_text,
            "parser_text_sha256": _sha256_text(result.parser_text),
            "strip_policy": decode.strip_policy,
            "stop_reason": decode.stop_reason,
            "generation_config_fingerprint": decode.generation_config_fingerprint,
            "model_identity_sha256": sha256_json(dict(decode.model_identity)),
            "tokenizer_identity_sha256": sha256_json(dict(decode.tokenizer_identity)),
        },
        "parse": {
            "row_index": parse.row_index,
            "status": parse.parse_status,
            "artifact_sha256": sha256_json(
                {"artifact": parse.to_artifact_dict(), "diagnostics": parse.diagnostics}
            ),
        },
        "cancellation": result.cancellation.to_receipt_dict(),
        "cuda": result.cuda_binding.to_receipt_dict(),
    }
    return _validate_execution_envelope(envelope)


def _failure_execution(
    *,
    canonical_profile: EngineProfile,
    canvas: ImmutableRgbCanvas,
    stage: str,
    code: str,
    binding: ResidentProfileBinding | None = None,
) -> dict[str, Any]:
    return _validate_execution_envelope(
        {
            "schema_version": EXECUTION_ENVELOPE_SCHEMA,
            "canonical_profile_fingerprint": canonical_profile.fingerprint,
            "resident_profile": (
                None if binding is None else binding.to_receipt_dict()
            ),
            "canvas": canvas.to_receipt_dict(),
            "decode": None,
            "parse": None,
            "cancellation": None,
            "cuda": None,
            "failure": {"stage": stage, "code": code},
        }
    )


def _validate_execution_envelope(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = _json_copy(value)
    common = {
        "schema_version",
        "canonical_profile_fingerprint",
        "resident_profile",
        "canvas",
        "decode",
        "parse",
        "cancellation",
        "cuda",
    }
    if set(payload) not in {frozenset(common), frozenset(common | {"failure"})}:
        raise _receipt_corrupt("execution envelope fields are not allowlisted")
    if payload["schema_version"] != EXECUTION_ENVELOPE_SCHEMA:
        raise _receipt_corrupt("execution envelope schema is unsupported")
    _require_sha256(payload["canonical_profile_fingerprint"], field="profile")
    resident_profile = payload["resident_profile"]
    if resident_profile is not None:
        _resident_binding_from_receipt(resident_profile)
    canvas = payload["canvas"]
    if not isinstance(canvas, dict) or set(canvas) != {
        "mode",
        "width",
        "height",
        "sha256",
    }:
        raise _receipt_corrupt("execution canvas receipt is invalid")
    if canvas["mode"] != "RGB":
        raise _receipt_corrupt("execution canvas must be RGB")
    _require_sha256(canvas["sha256"], field="canvas")
    forbidden_keys = {"path", "image_path", "bytes", "pixel_values", "tensor"}
    for key in _walk_keys(payload):
        lowered = key.lower()
        if (
            lowered in forbidden_keys
            or lowered.endswith("_path")
            or lowered.endswith("_bytes")
        ):
            raise _receipt_corrupt(
                "execution envelope cannot contain paths, bytes, or tensors"
            )
    if payload["decode"] is not None:
        decode = payload["decode"]
        expected = {
            "backend",
            "backend_mode",
            "response_family",
            "raw_generated_text",
            "raw_generated_sha256",
            "parser_text",
            "parser_text_sha256",
            "strip_policy",
            "stop_reason",
            "generation_config_fingerprint",
            "model_identity_sha256",
            "tokenizer_identity_sha256",
        }
        if not isinstance(decode, dict) or set(decode) != expected:
            raise _receipt_corrupt("execution decode fields are not allowlisted")
        if decode["raw_generated_sha256"] != _sha256_text(decode["raw_generated_text"]):
            raise _receipt_corrupt("raw generated text hash mismatch")
        if decode["parser_text_sha256"] != _sha256_text(decode["parser_text"]):
            raise _receipt_corrupt("parser text hash mismatch")
        for field in (
            "generation_config_fingerprint",
            "model_identity_sha256",
            "tokenizer_identity_sha256",
        ):
            _require_sha256(decode[field], field=field)
    failure = payload.get("failure")
    if failure is not None:
        if not isinstance(failure, dict) or set(failure) != {"stage", "code"}:
            raise _receipt_corrupt("execution failure fields are not allowlisted")
        for field in ("stage", "code"):
            value = failure[field]
            if (
                not isinstance(value, str)
                or not value
                or len(value) > 128
                or any(
                    not (character.isalnum() or character in "_.:-")
                    for character in value
                )
            ):
                raise _receipt_corrupt("execution failure identifier is unsafe")
    return payload


def _resident_binding_from_receipt(value: Any) -> ResidentProfileBinding:
    payload = _json_copy(value)
    expected = {
        "schema_version",
        "name",
        "resolved_infer_config_fingerprint",
        "prompt_policy_fingerprint",
        "generation_config_fingerprint",
        "runtime_identity_fingerprint",
        "processor_identity_fingerprint",
        "tokenizer_identity_fingerprint",
        "transformers_version",
        "parser",
        "transform_id",
        "processor",
        "deadline_seconds",
        "profile_fingerprint",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise _receipt_corrupt("resident profile receipt fields are not allowlisted")
    if payload["parser"] != {"id": PARSER_ID, "policy": PARSER_POLICY}:
        raise _receipt_corrupt("resident profile parser identity is invalid")
    if payload["transform_id"] != ROI_TRANSFORM_ID:
        raise _receipt_corrupt("resident profile transform identity is invalid")
    processor = payload["processor"]
    if not isinstance(processor, dict) or set(processor) != {
        "factor",
        "do_resize",
        "kwargs",
        "default_canvas",
        "axis_bounds",
        "max_total_pixels",
    }:
        raise _receipt_corrupt("resident profile processor receipt is invalid")
    if processor["do_resize"] is not False:
        raise _receipt_corrupt("resident profile must attest do_resize=false")
    if processor["kwargs"] != dict(QWEN_IMAGE_PROCESSOR_KWARGS):
        raise _receipt_corrupt("resident profile processor kwargs are invalid")
    try:
        binding = ResidentProfileBinding(
            schema_version=payload["schema_version"],
            name=payload["name"],
            resolved_infer_config_fingerprint=payload[
                "resolved_infer_config_fingerprint"
            ],
            prompt_policy_fingerprint=payload["prompt_policy_fingerprint"],
            generation_config_fingerprint=payload["generation_config_fingerprint"],
            runtime_identity_fingerprint=payload["runtime_identity_fingerprint"],
            processor_identity_fingerprint=payload["processor_identity_fingerprint"],
            tokenizer_identity_fingerprint=payload["tokenizer_identity_fingerprint"],
            transformers_version=payload["transformers_version"],
            processor_kwargs_json=canonical_json(processor["kwargs"]),
            processor_factor=processor["factor"],
            default_width=processor["default_canvas"][0],
            default_height=processor["default_canvas"][1],
            min_axis_pixels=processor["axis_bounds"][0],
            max_axis_pixels=processor["axis_bounds"][1],
            max_total_pixels=processor["max_total_pixels"],
            deadline_seconds=payload["deadline_seconds"],
        )
    except Exception as exc:
        raise _receipt_corrupt(
            "resident profile receipt cannot be reconstructed"
        ) from exc
    if canonical_json(binding.to_receipt_dict()) != canonical_json(payload):
        raise _receipt_corrupt("resident profile receipt fingerprint is inconsistent")
    return binding


def _validate_attempt_record(value: Any) -> dict[str, Any]:
    record = _json_copy(value)
    if set(record) != {
        "schema_version",
        "receipt_id",
        "request_id",
        "attempt",
        "execution",
        "response",
    }:
        raise _receipt_corrupt("attempt record fields are not allowlisted")
    if record["schema_version"] != RECEIPT_RECORD_SCHEMA:
        raise _receipt_corrupt("attempt record schema is unsupported")
    if record["receipt_id"] != _receipt_id(record["request_id"]):
        raise _receipt_corrupt("receipt ID does not match request ID")
    attempt = _attempt_from_record(record)
    if attempt.target.request_id != record["request_id"]:
        raise _receipt_corrupt("attempt request identity mismatch")
    execution = _validate_execution_envelope(record["execution"])
    if execution["canonical_profile_fingerprint"] != attempt.target.profile_fingerprint:
        raise _receipt_corrupt("execution and canonical profile fingerprints differ")
    record["execution"] = execution
    record["response"] = _strict_service_response(
        record["response"],
        receipt_id=record["receipt_id"],
        attempt=attempt,
    )
    return record


def _attempt_from_record(record: Mapping[str, Any]) -> InferenceAttemptReceipt:
    raw = record.get("attempt")
    if not isinstance(raw, Mapping):
        raise _receipt_corrupt("attempt payload must be a mapping")
    try:
        request = raw["request"]
        target = RequestTarget(
            request_id=request["request_id"],
            project_id=request["project_id"],
            task_id=request["task_id"],
            task_epoch=request["task_epoch"],
            image_id=request["image_id"],
            annotation_id=request["annotation_id"],
            annotation_revision=request["annotation_revision"],
            current_user_id=request["current_user_id"],
            draft_id=request["draft_id"],
            draft_revision=request["draft_revision"],
            profile_fingerprint=request["profile_fingerprint"],
            project_generation=request["project_generation"],
            transform_fingerprint=request["transform_fingerprint"],
            preexisting_draft_dirty=request["preexisting_draft_dirty"],
        )
        transform = RoiLetterboxTransform.from_receipt_dict(raw["transform"])
        lifecycle = RequestLifecycle()
        for event in raw["state_transitions"]:
            if event["from"] != lifecycle.state.value:
                raise _receipt_corrupt("lifecycle transition source is inconsistent")
            lifecycle = lifecycle.transition(
                RequestState(event["to"]),
                at_seconds=event["at_seconds"],
                reason=event["reason"],
            )
        result: ClassifiedInferenceResult | None = None
        if raw["result"] is not None:
            execution = record["execution"]
            parse_meta = execution["parse"]
            decode = execution["decode"]
            if not isinstance(parse_meta, dict) or not isinstance(decode, dict):
                raise _receipt_corrupt(
                    "result receipt lacks replayable parser execution"
                )
            parser_text = decode["parser_text"]
            parsed = parse_compact_object_box_closed(
                parser_text,
                row_id=target.request_id,
                row_index=parse_meta["row_index"],
                image_width=transform.canvas_width,
                image_height=transform.canvas_height,
            )
            result = classify_parser_result(
                parse_row=parsed,
                raw_response_text=parser_text,
                target=target,
                transform=transform,
            )
            raw_insertion = raw["result"].get("insertion_payload")
            if raw_insertion is not None:
                links = {
                    item["result_id"]: item["region_link"]
                    for item in raw_insertion["regions"]
                }
                result = finalize_region_links(
                    result,
                    CurrentTarget(**target.binding_payload()),
                    region_links=links,
                )
            if canonical_json(result.to_receipt_dict()) != canonical_json(
                raw["result"]
            ):
                raise _receipt_corrupt("persisted result does not replay exactly")
        failure = raw.get("failure")
        attempt = InferenceAttemptReceipt(
            target=target,
            lifecycle=lifecycle,
            profile_receipt=raw["profile"],
            transform_receipt=raw["transform"],
            result=result,
            failure_stage=None if failure is None else failure["stage"],
            failure_code=None if failure is None else failure["code"],
        )
    except ReceiptStoreError:
        raise
    except Exception as exc:
        raise _receipt_corrupt("terminal attempt cannot be reconstructed") from exc
    if canonical_json(attempt.to_dict()) != canonical_json(raw):
        raise _receipt_corrupt("terminal attempt is not canonical")
    return attempt


def _result_response(
    *, receipt_id: str, result: ClassifiedInferenceResult
) -> dict[str, Any]:
    payload = None
    if result.insertion_payload is not None:
        payload = result.insertion_payload.to_dict()
    return {
        "receipt_id": receipt_id,
        "request_id": result.target.request_id,
        "terminal_status": terminal_state_for_result(result).value,
        "clear_roi": result.clear_roi,
        "insertion_payload": payload,
        "failure": None,
        "counts": {
            "parsed": result.parsed_count,
            "inserted": result.inserted_count,
            "rejected": result.rejected_count,
        },
    }


def _strict_service_response(
    value: Mapping[str, Any],
    *,
    receipt_id: str,
    attempt: InferenceAttemptReceipt,
) -> dict[str, Any]:
    payload = _json_copy(value)
    expected = _canonical_service_response(receipt_id=receipt_id, attempt=attempt)
    _reject_service_response_credentials(payload)
    if canonical_json(payload) != canonical_json(expected):
        raise _receipt_corrupt(
            "service response does not exactly match the reconstructed terminal attempt"
        )
    return payload


def _canonical_service_response(
    *,
    receipt_id: str,
    attempt: InferenceAttemptReceipt,
) -> dict[str, Any]:
    if receipt_id != _receipt_id(attempt.target.request_id):
        raise _receipt_corrupt("service response receipt ID mismatch")
    if attempt.result is not None:
        return _result_response(receipt_id=receipt_id, result=attempt.result)
    failure = {
        "stage": attempt.failure_stage,
        "code": attempt.failure_code,
    }
    if not all(isinstance(value, str) and value for value in failure.values()):
        raise _receipt_corrupt("failure response lacks canonical stage/code")
    return {
        "receipt_id": receipt_id,
        "request_id": attempt.target.request_id,
        "terminal_status": attempt.lifecycle.state.value,
        "clear_roi": False,
        "insertion_payload": None,
        "failure": failure,
    }


def _reject_service_response_credentials(value: Any) -> None:
    forbidden = {
        "authorization",
        "cookie",
        "cookies",
        "password",
        "secret",
        "api_key",
        "apikey",
        "access_token",
        "refresh_token",
        "credential",
        "credentials",
        "private_key",
    }
    for key in _walk_keys(value):
        normalized = key.casefold().replace("-", "_")
        if (
            normalized in forbidden
            or "authorization" in normalized
            or "credential" in normalized
            or "private_key" in normalized
            or normalized.endswith(
                ("_password", "_secret", "_token", "_api_key", "_cookie")
            )
        ):
            raise _receipt_corrupt(
                "service response contains a credential-bearing field"
            )


def _validate_retry(
    record: Mapping[str, Any],
    *,
    target: RequestTarget,
    transform: RoiLetterboxTransform,
    canvas: ImmutableRgbCanvas,
) -> None:
    attempt = _attempt_from_record(record)
    execution = _validate_execution_envelope(record["execution"])
    if (
        attempt.target != target
        or canonical_json(_ordinary_json(attempt.transform_receipt))
        != canonical_json(transform.to_receipt_dict())
        or execution["canvas"] != canvas.to_receipt_dict()
    ):
        raise ReceiptStoreError(
            "request retry differs from its durable target/transform/canvas",
            code="receipt.request_conflict",
            stage="receipt",
        )


def _validate_resident_result(
    result: Any,
    *,
    target: RequestTarget,
    transform: RoiLetterboxTransform,
    binding: ResidentProfileBinding,
) -> None:
    if (
        result.target != target
        or result.transform != transform
        or result.profile != binding
    ):
        raise RoiRuntimeError(
            "resident result identity does not match its execution request",
            code="resident.result_identity",
            stage="runtime",
        )
    if result.parse.row_id != target.request_id:
        raise RoiRuntimeError(
            "resident parser row does not match request identity",
            code="resident.parse_identity",
            stage="runtime",
        )


def _attest_loaded_engine(
    *,
    engine: ResidentEngine,
    profile: EngineProfile,
    binding: ResidentProfileBinding,
) -> None:
    """Re-attest exact owner-exposed runtime identities against persistence."""

    resolved = getattr(engine, "resolved", None)
    if not isinstance(resolved, ResolvedInferConfig) or not isinstance(
        resolved.config, InferConfig
    ):
        raise RoiRuntimeError(
            "loaded engine does not expose a strict ResolvedInferConfig",
            code="profile.loaded_config_mismatch",
            stage="profile",
        )
    strict_payload = resolved.config.model_dump(mode="json")
    if (
        resolved.config_dict != strict_payload
        or resolved.fingerprint != sha256_json(strict_payload)
        or resolved.fingerprint != binding.resolved_infer_config_fingerprint
        or fingerprint_prompt_policy(_template_config(resolved.config))
        != binding.prompt_policy_fingerprint
    ):
        raise RoiRuntimeError(
            "loaded engine resolved config or prompt policy does not match the resident bridge",
            code="profile.loaded_config_mismatch",
            stage="profile",
        )
    if getattr(
        engine, "transformers_version", None
    ) != binding.transformers_version or getattr(
        engine, "processor_kwargs", None
    ) != dict(QWEN_IMAGE_PROCESSOR_KWARGS):
        raise RoiRuntimeError(
            "loaded engine package version or processor kwargs do not match the resident bridge",
            code="profile.loaded_execution_policy_mismatch",
            stage="profile",
        )
    runtime = getattr(engine, "runtime", None)
    qwen = getattr(runtime, "qwen", None)
    model_identity = getattr(runtime, "model_identity", None)
    processor_identity = getattr(qwen, "processor_identity", None)
    token_identity = getattr(qwen, "token_identity", None)
    if (
        not isinstance(model_identity, Mapping)
        or processor_identity is None
        or not callable(getattr(processor_identity, "to_artifact_dict", None))
        or token_identity is None
        or not callable(getattr(token_identity, "to_artifact_dict", None))
    ):
        raise RoiRuntimeError(
            "loaded engine lacks exact model/processor/tokenizer identity owners",
            code="profile.loaded_identity_missing",
            stage="profile",
        )
    observed = {
        "model": dict(model_identity),
        "processor": processor_identity.to_artifact_dict(),
        "tokenizer": token_identity.to_artifact_dict(),
    }
    profile.verify_runtime_identity(observed)
    component_checks = {
        "runtime_identity_fingerprint": sha256_json(observed["model"]),
        "processor_identity_fingerprint": sha256_json(observed["processor"]),
        "tokenizer_identity_fingerprint": sha256_json(observed["tokenizer"]),
    }
    for field, observed_fingerprint in component_checks.items():
        if getattr(binding, field) != observed_fingerprint:
            raise RoiRuntimeError(
                "loaded runtime component does not match the canonical bridge",
                code="profile.loaded_component_mismatch",
                stage="profile",
            )


def _validate_executed_generation_policy(generation: Mapping[str, Any]) -> None:
    temperature = generation.get("temperature")
    top_p = generation.get("top_p")
    if temperature != 0.0 or top_p != 1.0:
        raise RoiRuntimeError(
            "resident bridge supports only executed greedy temperature=0/top_p=1",
            code="profile.unsupported_sampling_policy",
            stage="profile",
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


def _decode_entries(payload: bytes) -> list[dict[str, Any]]:
    if not payload:
        return []
    if not payload.endswith(b"\n"):
        raise _receipt_corrupt("receipt store has a torn final record")
    entries: list[dict[str, Any]] = []
    for raw_line in payload.splitlines(keepends=True):
        try:
            text = raw_line.decode("utf-8")
            value = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise _receipt_corrupt(
                "receipt store contains invalid strict JSON"
            ) from exc
        if not isinstance(value, dict) or text != canonical_json(value) + "\n":
            raise _receipt_corrupt("receipt entry is not canonical JSON")
        entries.append(value)
    return entries


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _json_mapping(serialized: str, *, field: str) -> dict[str, Any]:
    value = json.loads(serialized)
    if not isinstance(value, dict):
        raise RoiRuntimeError(
            f"{field} must be a mapping",
            code=f"profile.{field}",
            stage="profile",
        )
    return value


def _json_copy(value: Any) -> Any:
    try:
        return json.loads(
            canonical_json(_ordinary_json(value)),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise _receipt_corrupt("receipt payload is not finite ordinary JSON") from exc


def _ordinary_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _ordinary_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_ordinary_json(item) for item in value]
    return value


def _walk_keys(value: Any):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _walk_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_keys(item)


def _receipt_id(request_id: str) -> str:
    _canonical_request_uuid(request_id)
    return f"roi-receipt:{request_id}"


def _canonical_request_uuid(value: str) -> str:
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, ValueError) as exc:
        raise RoiRuntimeError(
            "ROI request_id must be a canonical UUID",
            code="request.uuid",
            stage="request",
        ) from exc
    if str(parsed) != value:
        raise RoiRuntimeError(
            "ROI request_id must use canonical lowercase UUID text",
            code="request.uuid",
            stage="request",
        )
    return value


def _elapsed(clock: Any, started: float) -> float:
    value = float(clock()) - started
    if not math.isfinite(value):
        raise RoiRuntimeError(
            "request clock produced a non-finite timestamp",
            code="runtime.clock",
            stage="runtime",
        )
    return max(0.0, value)


def _failure_code(exc: Exception, *, default: str) -> str:
    value = getattr(exc, "code", default)
    if not isinstance(value, str) or not value:
        return default
    safe = "".join(
        char if (char.isalnum() or char in "_.:-") else "_" for char in value
    )
    return safe[:128] or default


def _sha256_text(value: str) -> str:
    if not isinstance(value, str):
        raise _receipt_corrupt("execution text must be text")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _require_sha256(value: Any, *, field: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise _receipt_corrupt(f"{field} must be a lowercase SHA-256")


def _receipt_corrupt(message: str) -> ReceiptStoreError:
    return ReceiptStoreError(
        message,
        code="receipt.corrupt",
        stage="receipt",
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "CurrentTargetProvider",
    "InferenceReceiptStore",
    "ReceiptStoreError",
    "RoiInferenceService",
    "RoiRuntimeError",
    "build_resident_profile_binding",
]
