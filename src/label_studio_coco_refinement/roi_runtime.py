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
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, replace
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
    AuthoritativeAbandonmentProof,
    AuthoritativeInsertionProof,
    ClassifiedInferenceResult,
    CurrentTarget,
    InferenceAttemptReceipt,
    InferenceResultContractError,
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


RECEIPT_STORE_SCHEMA = "coordexp-roi-receipt-chain-v2"
RECEIPT_RECORD_SCHEMA = "coordexp-roi-attempt-record-v2"
RECEIPT_DISPOSITION_RECORD_SCHEMA = "coordexp-roi-disposition-record-v1"
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


@dataclass(frozen=True)
class ResolvedInferenceReceiptLink(InferenceReceiptLink):
    """Commit link plus the vendor-locked post-insertion Draft attestation."""

    source_annotation_revision: str
    observed_annotation_revision: str
    inserted_draft_revision: str
    inserted_draft_updated_at: str
    saved_full_result_sha256: str
    saved_semantic_result_sha256: str


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
    """Append-only two-phase ROI receipt authority.

    A model result with candidate boxes first appends one ``produced`` attempt.
    A second authoritative append records either exact insertion or permanent
    abandonment.  Only the former is resolvable by the working-data store.
    """

    def __init__(self, path: str | Path, *, clock: Any = time.time) -> None:
        self.path = Path(path)
        self.clock = clock
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
            if (
                record["record_kind"] == "attempt"
                and record["request_id"] == request_id
            ):
                return _json_copy(record)
        return None

    def get(self, receipt_id: str) -> dict[str, Any] | None:
        for record in self.replay():
            if (
                record["record_kind"] == "attempt"
                and record["receipt_id"] == receipt_id
            ):
                return _json_copy(record)
        return None

    def disposition(self, receipt_id: str) -> dict[str, Any] | None:
        for record in self.replay():
            if (
                record["record_kind"] == "disposition"
                and record["receipt_id"] == receipt_id
            ):
                return _json_copy(record)
        return None

    def response(self, receipt_id: str) -> dict[str, Any] | None:
        """Return an idempotent response reflecting the latest durable phase."""

        attempt = self.get(receipt_id)
        if attempt is None:
            return None
        disposition = self.disposition(receipt_id)
        if disposition is None:
            return _json_copy(attempt["response"])
        response = _json_copy(attempt["response"])
        response.update(
            {
                "request_state": disposition["terminal_status"],
                "terminal_status": disposition["terminal_status"],
                "clear_roi": disposition["disposition"] == "inserted",
                "insertion_payload": None,
            }
        )
        attempt_counts = response.get("counts")
        if not isinstance(attempt_counts, dict):
            raise _receipt_corrupt("produced response lacks canonical counts")
        response["counts"] = {
            "parsed": attempt_counts["parsed"],
            "inserted": (
                attempt_counts["produced"]
                if disposition["disposition"] == "inserted"
                else 0
            ),
            "rejected": attempt_counts["rejected"],
        }
        if disposition["disposition"] == "inserted":
            response["failure"] = None
            response["result_region_keys"] = dict(
                disposition["proof"]["result_region_keys"]
            )
            response["insertion_attestation"] = {
                field: disposition["proof"][field]
                for field in (
                    "source_annotation_revision",
                    "observed_annotation_revision",
                    "source_draft_revision",
                    "inserted_draft_revision",
                    "inserted_draft_updated_at",
                    "saved_full_result_sha256",
                    "saved_semantic_result_sha256",
                )
            }
        else:
            response["failure"] = {
                "stage": "insertion",
                "code": disposition["proof"]["reason"],
            }
        _reject_service_response_credentials(response)
        return response

    def append(
        self,
        *,
        attempt: InferenceAttemptReceipt,
        execution: Mapping[str, Any],
        response: Mapping[str, Any],
        produced_ttl_seconds: float | None = None,
    ) -> str:
        receipt_id = _receipt_id(attempt.target.request_id)
        recorded_at = _finite_timestamp(self.clock(), field="recorded_at_seconds")
        if attempt.lifecycle.state is RequestState.PRODUCED:
            if produced_ttl_seconds is None:
                expires_at = None
            else:
                if (
                    isinstance(produced_ttl_seconds, bool)
                    or not isinstance(produced_ttl_seconds, (int, float))
                    or not math.isfinite(produced_ttl_seconds)
                    or produced_ttl_seconds <= 0
                ):
                    raise ReceiptStoreError(
                        "produced receipt TTL must be finite and positive",
                        code="receipt.produced_ttl",
                        stage="receipt",
                    )
                expires_at = recorded_at + float(produced_ttl_seconds)
        else:
            if produced_ttl_seconds is not None:
                raise ReceiptStoreError(
                    "terminal non-produced receipt cannot carry a produced TTL",
                    code="receipt.produced_ttl",
                    stage="receipt",
                )
            expires_at = None
        record = {
            "schema_version": RECEIPT_RECORD_SCHEMA,
            "record_kind": "attempt",
            "receipt_id": receipt_id,
            "request_id": attempt.target.request_id,
            "recorded_at_seconds": recorded_at,
            "expires_at_seconds": expires_at,
            "attempt": attempt.to_dict(),
            "execution": _validate_execution_envelope(execution),
            "response": _strict_service_response(
                response,
                receipt_id=receipt_id,
                attempt=attempt,
            ),
        }
        record = _validate_attempt_record(record)
        with self._exclusive_handle() as handle:
            records = self._read_locked(handle)
            for prior in records:
                if (
                    prior["record_kind"] != "attempt"
                    or prior["request_id"] != record["request_id"]
                ):
                    continue
                if canonical_json(_attempt_comparable(prior)) == canonical_json(
                    _attempt_comparable(record)
                ):
                    return receipt_id
                raise _receipt_conflict(
                    "request ID already has a different durable attempt receipt"
                )
            self._append_locked(handle, records=records, record=record)
        return receipt_id

    def finalize_inserted(self, proof: AuthoritativeInsertionProof) -> str:
        """Acknowledge exact durable Draft insertion and enable resolution."""

        if not isinstance(proof, AuthoritativeInsertionProof):
            raise ReceiptStoreError(
                "inserted finalization requires AuthoritativeInsertionProof",
                code="receipt.proof_type",
                stage="receipt",
            )
        return self._finalize_disposition(
            receipt_id=proof.receipt_id,
            request_id=proof.request_id,
            disposition="inserted",
            proof=proof.to_dict(),
        )

    def finalize_abandoned(self, proof: AuthoritativeAbandonmentProof) -> str:
        """Permanently abandon one produced candidate; it can never resolve."""

        if not isinstance(proof, AuthoritativeAbandonmentProof):
            raise ReceiptStoreError(
                "abandoned finalization requires AuthoritativeAbandonmentProof",
                code="receipt.proof_type",
                stage="receipt",
            )
        return self._finalize_disposition(
            receipt_id=proof.receipt_id,
            request_id=proof.request_id,
            disposition="abandoned",
            proof=proof.to_dict(),
        )

    def expire_produced(self, *, now: float | None = None) -> tuple[str, ...]:
        """Append abandonment for every overdue produced attempt, without a thread."""

        cutoff = _finite_timestamp(
            self.clock() if now is None else now,
            field="expiry cutoff",
        )
        expired: list[str] = []
        with self._exclusive_handle() as handle:
            records = self._read_locked(handle)
            finalized = {
                record["receipt_id"]
                for record in records
                if record["record_kind"] == "disposition"
            }
            for attempt_record in tuple(records):
                if (
                    attempt_record["record_kind"] != "attempt"
                    or attempt_record["receipt_id"] in finalized
                    or attempt_record["attempt"]["request_state"] != "produced"
                    or attempt_record["expires_at_seconds"] is None
                    or attempt_record["expires_at_seconds"] > cutoff
                ):
                    continue
                attempt = _attempt_from_record(attempt_record)
                target = attempt.target
                proof = AuthoritativeAbandonmentProof(
                    receipt_id=attempt_record["receipt_id"],
                    request_id=target.request_id,
                    project_id=target.project_id,
                    task_id=target.task_id,
                    task_epoch=target.task_epoch,
                    image_id=target.image_id,
                    annotation_id=target.annotation_id,
                    current_user_id=target.current_user_id,
                    draft_id=target.draft_id,
                    source_draft_revision=target.draft_revision,
                    reason="produced_expired",
                )
                disposition_record = _build_disposition_record(
                    disposition="abandoned",
                    proof=proof.to_dict(),
                    attempt=attempt,
                    recorded_at_seconds=_finite_timestamp(
                        self.clock(),
                        field="recorded_at_seconds",
                    ),
                )
                self._append_locked(
                    handle,
                    records=records,
                    record=disposition_record,
                )
                records.append(disposition_record)
                finalized.add(attempt_record["receipt_id"])
                expired.append(attempt_record["receipt_id"])
        return tuple(expired)

    def resolve(self, receipt_id: str) -> InferenceReceiptLink | None:
        record = self.get(receipt_id)
        disposition = self.disposition(receipt_id)
        if (
            record is None
            or disposition is None
            or disposition["disposition"] != "inserted"
        ):
            return None
        attempt = _attempt_from_record(record)
        if (
            attempt.lifecycle.state is not RequestState.PRODUCED
            or attempt.result is None
        ):
            raise _receipt_corrupt("inserted disposition lacks a produced attempt")
        image_id = attempt.target.image_id
        if not image_id.isdecimal():
            raise ReceiptStoreError(
                "inserted receipt image_id is not a decimal dataset identity",
                code="receipt.image_id",
                stage="receipt",
            )
        proof = disposition["proof"]
        return ResolvedInferenceReceiptLink(
            receipt_id=receipt_id,
            request_id=attempt.target.request_id,
            project_id=attempt.target.project_id,
            task_id=attempt.target.task_id,
            image_id=int(image_id),
            annotation_id=attempt.target.annotation_id,
            current_user_id=attempt.target.current_user_id,
            draft_id=attempt.target.draft_id,
            draft_revision=attempt.target.draft_revision,
            terminal_status=disposition["terminal_status"],
            result_region_keys=dict(proof["result_region_keys"]),
            source_annotation_revision=proof["source_annotation_revision"],
            observed_annotation_revision=proof["observed_annotation_revision"],
            inserted_draft_revision=proof["inserted_draft_revision"],
            inserted_draft_updated_at=proof["inserted_draft_updated_at"],
            saved_full_result_sha256=proof["saved_full_result_sha256"],
            saved_semantic_result_sha256=proof["saved_semantic_result_sha256"],
        )

    def _finalize_disposition(
        self,
        *,
        receipt_id: str,
        request_id: str,
        disposition: str,
        proof: Mapping[str, Any],
    ) -> str:
        recorded_at = _finite_timestamp(self.clock(), field="recorded_at_seconds")
        with self._exclusive_handle() as handle:
            records = self._read_locked(handle)
            attempt_record = next(
                (
                    record
                    for record in records
                    if record["record_kind"] == "attempt"
                    and record["receipt_id"] == receipt_id
                ),
                None,
            )
            if attempt_record is None:
                raise ReceiptStoreError(
                    "cannot finalize an unknown receipt",
                    code="receipt.unknown",
                    stage="receipt",
                )
            attempt = _attempt_from_record(attempt_record)
            if request_id != attempt.target.request_id or receipt_id != _receipt_id(
                request_id
            ):
                raise _receipt_conflict("finalization request/receipt identity differs")
            if attempt.lifecycle.state is not RequestState.PRODUCED:
                raise _receipt_conflict("only produced attempts accept a disposition")
            record = _build_disposition_record(
                disposition=disposition,
                proof=proof,
                attempt=attempt,
                recorded_at_seconds=recorded_at,
            )
            prior = next(
                (
                    item
                    for item in records
                    if item["record_kind"] == "disposition"
                    and item["receipt_id"] == receipt_id
                ),
                None,
            )
            if prior is not None:
                comparable = {
                    key: value
                    for key, value in record.items()
                    if key != "recorded_at_seconds"
                }
                prior_comparable = {
                    key: value
                    for key, value in prior.items()
                    if key != "recorded_at_seconds"
                }
                if canonical_json(prior_comparable) == canonical_json(comparable):
                    return receipt_id
                raise _receipt_conflict(
                    "receipt disposition or authoritative proof conflicts"
                )
            self._append_locked(handle, records=records, record=record)
        return receipt_id

    def _read_locked(self, handle: Any) -> list[dict[str, Any]]:
        handle.seek(0)
        entries = _decode_entries(handle.read())
        previous = _ZERO_HASH
        records: list[dict[str, Any]] = []
        attempts: dict[str, InferenceAttemptReceipt] = {}
        dispositions: set[str] = set()
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
            raw_record = entry["record"]
            if not isinstance(raw_record, Mapping):
                raise _receipt_corrupt("receipt record must be a mapping")
            if raw_record.get("record_kind") == "attempt":
                record = _validate_attempt_record(raw_record)
                if record["request_id"] in attempts:
                    raise _receipt_corrupt(
                        "attempt request and receipt IDs must be unique"
                    )
                attempts[record["request_id"]] = _attempt_from_record(record)
            elif raw_record.get("record_kind") == "disposition":
                request_id = raw_record.get("request_id")
                attempt = attempts.get(request_id)
                if attempt is None:
                    raise _receipt_corrupt("disposition precedes its produced attempt")
                record = _validate_disposition_record(raw_record, attempt=attempt)
                if record["receipt_id"] in dispositions:
                    raise _receipt_corrupt("receipt has more than one disposition")
                dispositions.add(record["receipt_id"])
            else:
                raise _receipt_corrupt("receipt record kind is unsupported")
            records.append(record)
            previous = entry["entry_sha256"]
        return records

    def _append_locked(
        self,
        handle: Any,
        *,
        records: list[dict[str, Any]],
        record: Mapping[str, Any],
    ) -> None:
        handle.seek(0)
        entries = _decode_entries(handle.read())
        previous_hash = _ZERO_HASH if not entries else entries[-1]["entry_sha256"]
        unsigned = {
            "schema_version": RECEIPT_STORE_SCHEMA,
            "sequence": len(records),
            "previous_entry_sha256": previous_hash,
            "record": record,
        }
        entry = {**unsigned, "entry_sha256": fingerprint_json(unsigned)}
        handle.seek(0, os.SEEK_END)
        handle.write((canonical_json(entry) + "\n").encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())
        _fsync_directory(self.path.parent)

    def _exclusive_handle(self):
        return _ExclusiveReceiptHandle(self.path)


class RoiInferenceService:
    """Run one target-bound resident request and durably publish its outcome."""

    def __init__(
        self,
        *,
        profiles: EngineProfileStore,
        receipts: InferenceReceiptStore,
        current_targets: CurrentTargetProvider,
        clock: Any,
        insertion_ack_timeout_seconds: float | None,
    ) -> None:
        if insertion_ack_timeout_seconds is not None and (
            isinstance(insertion_ack_timeout_seconds, bool)
            or not isinstance(insertion_ack_timeout_seconds, (int, float))
            or not math.isfinite(insertion_ack_timeout_seconds)
            or insertion_ack_timeout_seconds <= 0
        ):
            raise RoiRuntimeError(
                "insertion acknowledgement timeout must be positive or explicitly None",
                code="runtime.insertion_ack_timeout",
                stage="runtime",
            )
        self.profiles = profiles
        self.receipts = receipts
        self.current_targets = current_targets
        self.clock = clock
        self.insertion_ack_timeout_seconds = (
            None
            if insertion_ack_timeout_seconds is None
            else float(insertion_ack_timeout_seconds)
        )

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
            response = self.receipts.response(prior["receipt_id"])
            assert response is not None
            return response

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
        if classified.insertion_payload is not None:
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
            produced_ttl_seconds=(
                self.insertion_ack_timeout_seconds
                if terminal.state is RequestState.PRODUCED
                else None
            ),
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
            "request_state": lifecycle.state.value,
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
        "record_kind",
        "receipt_id",
        "request_id",
        "recorded_at_seconds",
        "expires_at_seconds",
        "attempt",
        "execution",
        "response",
    }:
        raise _receipt_corrupt("attempt record fields are not allowlisted")
    if record["schema_version"] != RECEIPT_RECORD_SCHEMA:
        raise _receipt_corrupt("attempt record schema is unsupported")
    if record["record_kind"] != "attempt":
        raise _receipt_corrupt("attempt record kind is invalid")
    _finite_timestamp(record["recorded_at_seconds"], field="recorded_at_seconds")
    if record["receipt_id"] != _receipt_id(record["request_id"]):
        raise _receipt_corrupt("receipt ID does not match request ID")
    attempt = _attempt_from_record(record)
    if attempt.target.request_id != record["request_id"]:
        raise _receipt_corrupt("attempt request identity mismatch")
    execution = _validate_execution_envelope(record["execution"])
    if execution["canonical_profile_fingerprint"] != attempt.target.profile_fingerprint:
        raise _receipt_corrupt("execution and canonical profile fingerprints differ")
    if attempt.lifecycle.state is RequestState.PRODUCED:
        if record["expires_at_seconds"] is not None:
            expires_at = _finite_timestamp(
                record["expires_at_seconds"],
                field="expires_at_seconds",
            )
            if expires_at <= record["recorded_at_seconds"]:
                raise _receipt_corrupt("produced receipt expiry must follow production")
    elif record["expires_at_seconds"] is not None:
        raise _receipt_corrupt("non-produced attempt cannot carry an expiry")
    record["execution"] = execution
    record["response"] = _strict_service_response(
        record["response"],
        receipt_id=record["receipt_id"],
        attempt=attempt,
    )
    return record


def _attempt_comparable(record: Mapping[str, Any]) -> dict[str, Any]:
    comparable = {
        key: value
        for key, value in record.items()
        if key not in {"recorded_at_seconds", "expires_at_seconds"}
    }
    expires_at = record.get("expires_at_seconds")
    comparable["produced_ttl_seconds"] = (
        None
        if expires_at is None
        else float(expires_at) - float(record["recorded_at_seconds"])
    )
    return comparable


def _build_disposition_record(
    *,
    disposition: str,
    proof: Mapping[str, Any],
    attempt: InferenceAttemptReceipt,
    recorded_at_seconds: float,
) -> dict[str, Any]:
    terminal_status = _validate_authoritative_proof(
        disposition=disposition,
        proof=proof,
        attempt=attempt,
    )
    return _validate_disposition_record(
        {
            "schema_version": RECEIPT_DISPOSITION_RECORD_SCHEMA,
            "record_kind": "disposition",
            "receipt_id": _receipt_id(attempt.target.request_id),
            "request_id": attempt.target.request_id,
            "recorded_at_seconds": recorded_at_seconds,
            "disposition": disposition,
            "terminal_status": terminal_status,
            "proof": proof,
        },
        attempt=attempt,
    )


def _validate_disposition_record(
    value: Any,
    *,
    attempt: InferenceAttemptReceipt,
) -> dict[str, Any]:
    record = _json_copy(value)
    if set(record) != {
        "schema_version",
        "record_kind",
        "receipt_id",
        "request_id",
        "recorded_at_seconds",
        "disposition",
        "terminal_status",
        "proof",
    }:
        raise _receipt_corrupt("disposition record fields are not allowlisted")
    if (
        record["schema_version"] != RECEIPT_DISPOSITION_RECORD_SCHEMA
        or record["record_kind"] != "disposition"
        or record["disposition"] not in {"inserted", "abandoned"}
    ):
        raise _receipt_corrupt("disposition record schema or kind is unsupported")
    _finite_timestamp(record["recorded_at_seconds"], field="recorded_at_seconds")
    if (
        record["request_id"] != attempt.target.request_id
        or record["receipt_id"] != _receipt_id(attempt.target.request_id)
        or attempt.lifecycle.state is not RequestState.PRODUCED
    ):
        raise _receipt_corrupt("disposition is not bound to its produced attempt")
    expected_status = _validate_authoritative_proof(
        disposition=record["disposition"],
        proof=record["proof"],
        attempt=attempt,
    )
    if record["terminal_status"] != expected_status:
        raise _receipt_corrupt("disposition terminal status is inconsistent")
    return record


def _validate_authoritative_proof(
    *,
    disposition: str,
    proof: Mapping[str, Any],
    attempt: InferenceAttemptReceipt,
) -> str:
    try:
        if disposition == "inserted":
            typed_proof: AuthoritativeInsertionProof | AuthoritativeAbandonmentProof = (
                AuthoritativeInsertionProof(**dict(proof))
            )
        elif disposition == "abandoned":
            typed_proof = AuthoritativeAbandonmentProof(**dict(proof))
        else:
            raise InferenceResultContractError("unsupported disposition")
    except Exception as exc:
        raise _receipt_corrupt("authoritative disposition proof is invalid") from exc
    if canonical_json(typed_proof.to_dict()) != canonical_json(proof):
        raise _receipt_corrupt("authoritative disposition proof is not canonical")
    target = attempt.target
    common_expected = {
        "receipt_id": _receipt_id(target.request_id),
        "request_id": target.request_id,
        "project_id": target.project_id,
        "task_id": target.task_id,
        "task_epoch": target.task_epoch,
        "image_id": target.image_id,
        "annotation_id": target.annotation_id,
        "current_user_id": target.current_user_id,
        "draft_id": target.draft_id,
        "source_draft_revision": target.draft_revision,
    }
    if disposition == "inserted":
        common_expected.update(
            {
                "source_annotation_revision": target.annotation_revision,
                "observed_annotation_revision": target.annotation_revision,
            }
        )
    observed = typed_proof.to_dict()
    mismatches = tuple(
        field
        for field, expected in common_expected.items()
        if observed[field] != expected
    )
    if mismatches:
        raise _receipt_conflict(
            "authoritative proof target differs: " + ",".join(mismatches)
        )
    if disposition == "abandoned":
        return RequestState.ABANDONED_BEFORE_INSERTION.value
    if attempt.result is None or attempt.result.insertion_payload is None:
        raise _receipt_corrupt("inserted proof lacks a candidate payload")
    expected_mapping = {
        region.result_id: region.region_key
        for region in attempt.result.insertion_payload.regions
    }
    if any(value is None for value in expected_mapping.values()):
        raise _receipt_corrupt("produced candidate lacks planned region keys")
    if observed["result_region_keys"] != expected_mapping:
        raise _receipt_conflict("authoritative result-to-region mapping differs")
    if attempt.result.outcome.value not in _ACCEPTED_STATUSES:
        raise _receipt_corrupt("inserted disposition has a non-accepted parser outcome")
    return attempt.result.outcome.value


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
                    item["result_id"]: item["region_key"]
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
    state = terminal_state_for_result(result)
    return {
        "receipt_id": receipt_id,
        "request_id": result.target.request_id,
        "request_state": state.value,
        "terminal_status": state.value if state.terminal else None,
        "clear_roi": result.clear_roi if state.terminal else False,
        "insertion_payload": payload,
        "failure": None,
        "counts": {
            "parsed": result.parsed_count,
            "produced": result.produced_count,
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
        "request_state": attempt.lifecycle.state.value,
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


def _finite_timestamp(value: Any, *, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise _receipt_corrupt(f"{field} must be finite and non-negative")
    return float(value)


def _receipt_conflict(message: str) -> ReceiptStoreError:
    return ReceiptStoreError(
        message,
        code="receipt.request_conflict",
        stage="receipt",
    )


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


class _ExclusiveReceiptHandle:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle: Any = None

    def __enter__(self) -> Any:
        self.handle = self.path.open("a+b")
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX)
        return self.handle

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()


__all__ = [
    "CurrentTargetProvider",
    "InferenceReceiptStore",
    "ReceiptStoreError",
    "RoiInferenceService",
    "RoiRuntimeError",
    "build_resident_profile_binding",
]
