"""Non-metric sampling calibration and primary-schedule materialization contracts.

The calibration lane deliberately contains no primary-arm, attempt-ledger, or
metric-admission surface.  It selects the first frozen sampled policy whose
initial panel passes the predeclared mechanics gates and whose replay panels are
request-stable.  The selected policy may then be bound into a primary research
schedule by a separate immutable artifact.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from types import MappingProxyType
from typing import Any, Literal

from src.analysis.spatial_scope_history.cohort_ledger import (
    CohortLedger,
    ExecutionIdentityBundle,
    canonical_json_text,
    sha256_file,
    sha256_payload,
)
from src.analysis.spatial_scope_history.schedule import (
    DecodeProvenance,
    GridProvenance,
    PRIMARY_ROOT_SEED,
    ResearchSchedule,
    derive_sampling_seed,
)
from src.analysis.spatial_scope_history.spatial import (
    EXECUTED_VISUAL_TENSOR_RECEIPT_SCHEMA_VERSION,
    TENSOR_CONTENT_RECEIPT_SCHEMA_VERSION,
    VISUAL_INPUT_MATERIALIZATION_RECEIPT_SCHEMA_VERSION,
    SpatialGridSpec,
)
from src.common.errors import ArtifactContractError, DataContractError
from src.inference.backend import (
    DecodeGenerationPolicy,
    DecodeResult,
    TokenTrace,
    canonical_float32_logprob,
    validate_sampled_runtime_attestation_aggregate_output,
    verify_checkpoint_payload_identity,
)
from src.inference.parsing import parse_compact_object_box_closed
from src.inference.scoring import score_prediction


CALIBRATION_REQUEST_SCHEMA_VERSION = "spatial_scope_history.calibration_request.v1"
CALIBRATION_OBSERVATION_SCHEMA_VERSION = (
    "spatial_scope_history.calibration_call_observation.v1"
)
CALIBRATION_TERMINAL_BUNDLE_SCHEMA_VERSION = (
    "spatial_scope_history.calibration_terminal_bundle.v1"
)
CALIBRATION_TERMINAL_BUNDLE_COLLECTION_SCHEMA_VERSION = (
    "spatial_scope_history.calibration_terminal_bundle_collection.v1"
)
CALIBRATION_PANEL_SCHEMA_VERSION = "spatial_scope_history.calibration_panel.v1"
CALIBRATION_CANDIDATE_SCHEMA_VERSION = (
    "spatial_scope_history.calibration_candidate_evaluation.v1"
)
CALIBRATION_SELECTION_SCHEMA_VERSION = (
    "spatial_scope_history.calibration_selection_receipt.v1"
)
ATTESTED_POLICY_SET_SCHEMA_VERSION = (
    "spatial_scope_history.attested_sampling_policy_set.v1"
)
SOURCE_RUNTIME_IDENTITY_SCHEMA_VERSION = (
    "spatial_scope_history.source_runtime_identity_receipt.v2"
)
PRIMARY_SCHEDULE_ARTIFACT_SCHEMA_VERSION = (
    "spatial_scope_history.primary_schedule_artifact.v1"
)

CALIBRATION_COHORT_ID = "sampling-calibration-12"
VALIDATION_COHORT_ID = "validation-200"
DENSE_COHORT_ID = "dense-union-51"
CALIBRATION_TEMPERATURES = (0.2, 0.4, 0.6)
CALIBRATION_SEEDS_PER_IMAGE = 4
CALIBRATION_IMAGE_COUNT = 12
CALIBRATION_INITIAL_CALL_COUNT = 48
CALIBRATION_CONFIRMATION_CALL_COUNT = 48
CALIBRATION_MAX_CALL_COUNT = 240
CALIBRATION_PARSE_MINIMUM = 46
CALIBRATION_NATURAL_CLOSURE_MINIMUM = 39
CALIBRATION_DISTINCT_IMAGE_MINIMUM = 9
CALIBRATION_DIVERSITY_MINIMUM = 0.10
FROZEN_READINESS_LEDGER_SEAL_SHA256 = (
    "c177bc2b12f06fed559660bb5e1d7ff24fa19ee5eb898e642cae27d6991ebb68"
)
FROZEN_RESOLVED_INFERENCE_CONFIG_SHA256 = (
    "f3000588accbcf1d9ada3b2f3e0b3324d660b4810b75d8f5d050d9f184f9ca80"
)
FROZEN_CHECKPOINT_MANIFEST_SHA256 = (
    "613e5d97f4a7a53d6325b5c1909813d6bb82e72622942b225df9724b5556a536"
)
FROZEN_SPATIAL_GRID_SPEC_SHA256 = (
    "0e67df2ff9c0b39cff80e1b7ebe94ce12855a9b113497ec70b5197ea68b9e9b6"
)
_SOURCE_RUNTIME_CRITICAL_PATHS = (
    "scripts/research/attest_request_scoped_sampling_cuda.py",
    "scripts/research/attest_request_scoped_sampling_source.py",
    "src/inference/backend.py",
)

_PROMPT_RECORD_ARTIFACT_FIELDS = {
    "assistant_format",
    "example_id",
    "full_prompt_fingerprint",
    "object_field_order",
    "object_ordering",
    "prompt_text",
    "prompt_token_count",
    "prompt_token_ids",
    "realized_object_order",
    "row_id",
    "row_index",
    "template_fingerprint",
    "template_id",
}
_VISUAL_INPUT_RECEIPT_FIELDS = {
    "executed_visual_tensors",
    "input_height",
    "input_kind",
    "input_rgb_sha256",
    "input_width",
    "processor_contract_sha256",
    "receipt_sha256",
    "schema_version",
    "source_image_sha256",
    "spatial_image_encoding_sha256",
}

CalibrationPanelKind = Literal["initial", "exact_replay", "reversed_order"]
_PANEL_KINDS = frozenset({"initial", "exact_replay", "reversed_order"})


@dataclass(frozen=True)
class AttestedSamplingPolicySet:
    """Three exact sampled policies authorized by one aggregate attestation."""

    aggregate_artifact_sha256: str
    aggregate_payload_fingerprint: str
    policy_fingerprints_by_temperature: tuple[tuple[float, str], ...]
    schema_version: str = ATTESTED_POLICY_SET_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTESTED_POLICY_SET_SCHEMA_VERSION:
            _artifact_error(
                "attested policy set schema is unsupported", "policy_set_schema"
            )
        _require_sha256(self.aggregate_artifact_sha256, "aggregate_artifact_sha256")
        _require_sha256(
            self.aggregate_payload_fingerprint, "aggregate_payload_fingerprint"
        )
        observed = tuple(
            temperature for temperature, _ in self.policy_fingerprints_by_temperature
        )
        if observed != CALIBRATION_TEMPERATURES:
            _artifact_error(
                "attested policy set must contain the exact frozen temperature order",
                "policy_set_temperature_order",
            )
        for _, fingerprint in self.policy_fingerprints_by_temperature:
            _require_sha256(fingerprint, "decode_generation_policy_fingerprint")
        if (
            len(
                {
                    fingerprint
                    for _, fingerprint in self.policy_fingerprints_by_temperature
                }
            )
            != 3
        ):
            _artifact_error(
                "attested policy fingerprints must be unique", "policy_set_duplicate"
            )

    def policy_fingerprint(self, temperature: float) -> str:
        matches = [
            fingerprint
            for candidate, fingerprint in self.policy_fingerprints_by_temperature
            if candidate == temperature
        ]
        if len(matches) != 1:
            _artifact_error(
                "temperature is absent from the attested policy set", "policy_missing"
            )
        return matches[0]

    @property
    def fingerprint(self) -> str:
        return sha256_payload(self.to_artifact_dict())

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "aggregate_artifact_sha256": self.aggregate_artifact_sha256,
            "aggregate_payload_fingerprint": self.aggregate_payload_fingerprint,
            "policy_fingerprints_by_temperature": [
                {
                    "temperature": temperature,
                    "decode_generation_policy_fingerprint": fingerprint,
                }
                for temperature, fingerprint in self.policy_fingerprints_by_temperature
            ],
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_artifact_dict(cls, value: Mapping[str, Any]) -> AttestedSamplingPolicySet:
        _require_exact_keys(
            value,
            {
                "aggregate_artifact_sha256",
                "aggregate_payload_fingerprint",
                "policy_fingerprints_by_temperature",
                "schema_version",
            },
            "attested sampling policy set",
        )
        pairs = value["policy_fingerprints_by_temperature"]
        if not isinstance(pairs, list):
            _artifact_error("attested policy list must be an array", "policy_set_type")
        parsed: list[tuple[float, str]] = []
        for row in pairs:
            _require_exact_keys(
                row,
                {"temperature", "decode_generation_policy_fingerprint"},
                "attested policy row",
            )
            parsed.append(
                (row["temperature"], row["decode_generation_policy_fingerprint"])
            )
        return cls(
            aggregate_artifact_sha256=value["aggregate_artifact_sha256"],
            aggregate_payload_fingerprint=value["aggregate_payload_fingerprint"],
            policy_fingerprints_by_temperature=tuple(parsed),
            schema_version=value["schema_version"],
        )


@dataclass(frozen=True)
class CalibrationRequest:
    """One non-metric full-image calibration request with a domain-separated seed."""

    request_id: str
    image_id: int
    image_frozen_order: int
    image_sha256: str
    call_index: int
    call_label: str
    sampling_seed: int
    temperature: float
    decode_generation_policy_fingerprint: str
    calibration_cohort_sha256: str
    schema_version: str = CALIBRATION_REQUEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CALIBRATION_REQUEST_SCHEMA_VERSION:
            _data_error("calibration request schema is unsupported", "request_schema")
        if self.temperature not in CALIBRATION_TEMPERATURES:
            _data_error(
                "calibration request temperature is not frozen", "request_temperature"
            )
        if not 0 <= self.call_index < CALIBRATION_SEEDS_PER_IMAGE:
            _data_error(
                "calibration call index is outside the frozen panel",
                "request_call_index",
            )
        if self.call_label != f"call-{self.call_index:02d}":
            _data_error(
                "calibration call label differs from its index", "request_call_label"
            )
        expected_seed = derive_sampling_seed(
            root_seed=PRIMARY_ROOT_SEED,
            role="temperature-calibration",
            image_id=self.image_id,
            cell_or_call_label=self.call_label,
        )
        if self.sampling_seed != expected_seed:
            _data_error(
                "calibration seed differs from frozen derivation", "request_seed"
            )
        for field_name in (
            "image_sha256",
            "decode_generation_policy_fingerprint",
            "calibration_cohort_sha256",
        ):
            _require_sha256(getattr(self, field_name), field_name)
        expected_id = "spatial-scope-history-calibration-request:" + sha256_payload(
            self.identity_payload()
        )
        if self.request_id != expected_id:
            _data_error(
                "calibration request identifier does not bind its payload", "request_id"
            )

    def identity_payload(self) -> dict[str, Any]:
        return {
            "calibration_cohort_sha256": self.calibration_cohort_sha256,
            "call_index": self.call_index,
            "call_label": self.call_label,
            "decode_generation_policy_fingerprint": self.decode_generation_policy_fingerprint,
            "image_id": self.image_id,
            "image_sha256": self.image_sha256,
            "sampling_seed": self.sampling_seed,
            "schema_version": self.schema_version,
            "temperature": self.temperature,
        }

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            **self.identity_payload(),
            "image_frozen_order": self.image_frozen_order,
            "request_id": self.request_id,
        }

    @classmethod
    def from_artifact_dict(cls, value: Mapping[str, Any]) -> CalibrationRequest:
        _require_exact_keys(
            value,
            {
                "calibration_cohort_sha256",
                "call_index",
                "call_label",
                "decode_generation_policy_fingerprint",
                "image_frozen_order",
                "image_id",
                "image_sha256",
                "request_id",
                "sampling_seed",
                "schema_version",
                "temperature",
            },
            "calibration request",
        )
        return cls(**dict(value))


@dataclass(frozen=True)
class CalibrationBackendAttestationBinding:
    """Exact execution contracts extracted from one validated aggregate."""

    aggregate_artifact_sha256: str
    aggregate_payload_fingerprint: str
    execution_contracts_by_policy: tuple[tuple[str, tuple[tuple[str, Any], ...]], ...]

    def __post_init__(self) -> None:
        _require_sha256(self.aggregate_artifact_sha256, "aggregate_artifact_sha256")
        _require_sha256(
            self.aggregate_payload_fingerprint, "aggregate_payload_fingerprint"
        )
        if len(self.execution_contracts_by_policy) != len(CALIBRATION_TEMPERATURES):
            _artifact_error(
                "calibration backend binding requires exactly three policies",
                "backend_binding_policy_count",
            )
        for policy_fingerprint, contract in self.execution_contracts_by_policy:
            _require_sha256(policy_fingerprint, "decode_generation_policy_fingerprint")
            if not contract:
                _artifact_error(
                    "calibration backend binding has an empty execution contract",
                    "backend_binding_contract",
                )

    def execution_contract(self, policy_fingerprint: str) -> Mapping[str, Any]:
        matches = [
            dict(contract)
            for candidate, contract in self.execution_contracts_by_policy
            if candidate == policy_fingerprint
        ]
        if len(matches) != 1:
            _artifact_error(
                "calibration response policy is absent from the runtime attestation",
                "backend_binding_policy",
            )
        return matches[0]


@dataclass(frozen=True)
class CalibrationTerminalBundle:
    """Canonical completed calibration call, without precomputed gate counters."""

    panel_kind: CalibrationPanelKind
    request: CalibrationRequest
    physical_batch_index: int
    request_execution_index: int
    prompt_record: Mapping[str, Any]
    visual_input_materialization_receipt: Mapping[str, Any]
    model_input_sha256: str
    decode_result: DecodeResult
    backend_attestation_aggregate_sha256: str
    backend_attestation_aggregate_fingerprint: str
    bundle_sha256: str | None = None
    schema_version: str = CALIBRATION_TERMINAL_BUNDLE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CALIBRATION_TERMINAL_BUNDLE_SCHEMA_VERSION:
            _artifact_error(
                "calibration terminal bundle schema is unsupported",
                "terminal_bundle_schema",
            )
        if self.panel_kind not in _PANEL_KINDS:
            _artifact_error(
                "calibration terminal bundle panel is unknown",
                "terminal_bundle_panel",
            )
        if self.physical_batch_index < 0 or self.request_execution_index not in range(
            4
        ):
            _artifact_error(
                "calibration terminal bundle execution position is invalid",
                "terminal_bundle_position",
            )
        for field_name in (
            "model_input_sha256",
            "backend_attestation_aggregate_sha256",
            "backend_attestation_aggregate_fingerprint",
        ):
            _require_sha256(getattr(self, field_name), field_name)
        _validate_prompt_record_artifact(self.prompt_record, request=self.request)
        _validate_visual_input_receipt(
            self.visual_input_materialization_receipt,
            request=self.request,
        )
        object.__setattr__(self, "prompt_record", _freeze_mapping(self.prompt_record))
        object.__setattr__(
            self,
            "visual_input_materialization_receipt",
            _freeze_mapping(self.visual_input_materialization_receipt),
        )
        self.decode_result.validate_for_scored()
        receipt = self.decode_result.execution_receipt
        assert receipt is not None
        expected = {
            "request_id": self.request.request_id,
            "sampling_seed": self.request.sampling_seed,
            "decode_generation_policy_fingerprint": (
                self.request.decode_generation_policy_fingerprint
            ),
            "request_execution_index": self.request_execution_index,
        }
        mismatches = {
            field: {"expected": value, "observed": getattr(receipt, field)}
            for field, value in expected.items()
            if getattr(receipt, field) != value
        }
        if self.decode_result.request_id != self.request.request_id:
            mismatches["decode_result.request_id"] = {
                "expected": self.request.request_id,
                "observed": self.decode_result.request_id,
            }
        prompt_token_ids = list(self.prompt_record["prompt_token_ids"])
        if self.decode_result.prompt_token_ids != prompt_token_ids:
            mismatches["prompt_token_ids"] = {
                "expected": prompt_token_ids,
                "observed": self.decode_result.prompt_token_ids,
            }
        policy = DecodeGenerationPolicy(**dict(receipt.decode_generation_policy))
        if (
            policy.fingerprint != self.request.decode_generation_policy_fingerprint
            or float(policy.temperature) != self.request.temperature
        ):
            mismatches["decode_generation_policy"] = {
                "expected_temperature": self.request.temperature,
                "observed": policy.to_artifact_dict(),
            }
        expected_model_input_sha256 = sha256_payload(
            {
                "prompt_token_ids": prompt_token_ids,
                "visual_input_materialization_receipt": _thaw_mapping(
                    self.visual_input_materialization_receipt
                ),
            }
        )
        if self.model_input_sha256 != expected_model_input_sha256:
            mismatches["model_input_sha256"] = {
                "expected": expected_model_input_sha256,
                "observed": self.model_input_sha256,
            }
        if mismatches:
            _artifact_error(
                "calibration terminal bundle artifacts are not one execution",
                "terminal_bundle_binding",
                mismatches=mismatches,
            )
        if self.bundle_sha256 is None:
            object.__setattr__(
                self, "bundle_sha256", sha256_payload(self.identity_payload())
            )
        _require_sha256(self.bundle_sha256, "bundle_sha256")
        if self.bundle_sha256 != sha256_payload(self.identity_payload()):
            _artifact_error(
                "calibration terminal bundle fingerprint is invalid",
                "terminal_bundle_fingerprint",
            )

    @property
    def fingerprint(self) -> str:
        assert self.bundle_sha256 is not None
        return self.bundle_sha256

    def identity_payload(self) -> dict[str, Any]:
        payload = self.to_artifact_dict()
        payload.pop("bundle_sha256")
        return payload

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "backend_attestation_aggregate_fingerprint": (
                self.backend_attestation_aggregate_fingerprint
            ),
            "backend_attestation_aggregate_sha256": (
                self.backend_attestation_aggregate_sha256
            ),
            "bundle_sha256": self.bundle_sha256,
            "decode_result": self.decode_result.to_artifact_dict(),
            "model_input_sha256": self.model_input_sha256,
            "panel_kind": self.panel_kind,
            "physical_batch_index": self.physical_batch_index,
            "prompt_record": _thaw_mapping(self.prompt_record),
            "request": self.request.to_artifact_dict(),
            "request_execution_index": self.request_execution_index,
            "schema_version": self.schema_version,
            "visual_input_materialization_receipt": _thaw_mapping(
                self.visual_input_materialization_receipt
            ),
        }

    @classmethod
    def from_artifact_dict(cls, value: Mapping[str, Any]) -> CalibrationTerminalBundle:
        _require_exact_keys(
            value,
            {
                "backend_attestation_aggregate_fingerprint",
                "backend_attestation_aggregate_sha256",
                "bundle_sha256",
                "decode_result",
                "model_input_sha256",
                "panel_kind",
                "physical_batch_index",
                "prompt_record",
                "request",
                "request_execution_index",
                "schema_version",
                "visual_input_materialization_receipt",
            },
            "calibration terminal bundle",
        )
        payload = dict(value)
        payload["request"] = CalibrationRequest.from_artifact_dict(payload["request"])
        payload["decode_result"] = DecodeResult.from_artifact_dict(
            payload["decode_result"]
        )
        return cls(**payload)


@dataclass(frozen=True)
class CalibrationCallObservation:
    """Derived response summary minted only from a canonical terminal bundle."""

    panel_kind: CalibrationPanelKind
    request: CalibrationRequest
    physical_batch_index: int
    request_execution_index: int
    raw_response_bytes_sha256: str
    parse_without_call_level_failure: bool
    natural_closure: bool
    detected_official_reference_object_ids: tuple[str, ...]
    decode_receipt_fingerprint: str
    terminal_bundle_sha256: str
    schema_version: str = CALIBRATION_OBSERVATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CALIBRATION_OBSERVATION_SCHEMA_VERSION:
            _artifact_error(
                "calibration observation schema is unsupported", "observation_schema"
            )
        if self.panel_kind not in _PANEL_KINDS:
            _artifact_error("calibration panel kind is unknown", "observation_panel")
        if self.physical_batch_index < 0 or self.request_execution_index not in range(
            4
        ):
            _artifact_error(
                "calibration execution position is invalid", "observation_position"
            )
        for field_name in (
            "raw_response_bytes_sha256",
            "decode_receipt_fingerprint",
            "terminal_bundle_sha256",
        ):
            _require_sha256(getattr(self, field_name), field_name)
        if not isinstance(
            self.parse_without_call_level_failure, bool
        ) or not isinstance(self.natural_closure, bool):
            _artifact_error(
                "calibration gate flags must be booleans", "observation_flags"
            )
        if tuple(sorted(set(self.detected_official_reference_object_ids))) != (
            self.detected_official_reference_object_ids
        ):
            _artifact_error(
                "detected official reference identifiers must be sorted and unique",
                "observation_reference_order",
            )
        if any(not value for value in self.detected_official_reference_object_ids):
            _artifact_error(
                "detected reference identifiers must be nonempty",
                "observation_reference",
            )

    @property
    def fingerprint(self) -> str:
        return sha256_payload(self.to_artifact_dict())

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "decode_receipt_fingerprint": self.decode_receipt_fingerprint,
            "detected_official_reference_object_ids": list(
                self.detected_official_reference_object_ids
            ),
            "natural_closure": self.natural_closure,
            "panel_kind": self.panel_kind,
            "parse_without_call_level_failure": self.parse_without_call_level_failure,
            "physical_batch_index": self.physical_batch_index,
            "raw_response_bytes_sha256": self.raw_response_bytes_sha256,
            "request": self.request.to_artifact_dict(),
            "request_execution_index": self.request_execution_index,
            "schema_version": self.schema_version,
            "terminal_bundle_sha256": self.terminal_bundle_sha256,
        }


@dataclass(frozen=True)
class _CalibrationMatchedPrediction:
    image_id: str
    prediction_id: str
    normalized_category_name: str
    global_bbox_xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class CalibrationPanelReceipt:
    """Frozen gate statistics and ordered evidence for one 48-call panel."""

    panel_kind: CalibrationPanelKind
    temperature: float
    decode_generation_policy_fingerprint: str
    ordered_observation_sha256s: tuple[str, ...]
    parse_success_count: int
    natural_closure_count: int
    images_with_multiple_distinct_raw_outputs: int
    prediction_set_diversity: float
    prediction_set_diversity_image_count: int
    gate_passed: bool
    schema_version: str = CALIBRATION_PANEL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CALIBRATION_PANEL_SCHEMA_VERSION:
            _artifact_error("calibration panel schema is unsupported", "panel_schema")
        if (
            self.panel_kind not in _PANEL_KINDS
            or self.temperature not in CALIBRATION_TEMPERATURES
        ):
            _artifact_error("calibration panel identity is invalid", "panel_identity")
        _require_sha256(
            self.decode_generation_policy_fingerprint,
            "decode_generation_policy_fingerprint",
        )
        if len(self.ordered_observation_sha256s) != CALIBRATION_INITIAL_CALL_COUNT:
            _artifact_error(
                "calibration panel must contain exactly 48 calls", "panel_call_count"
            )
        for value in self.ordered_observation_sha256s:
            _require_sha256(value, "ordered_observation_sha256")
        for value, upper in (
            (self.parse_success_count, 48),
            (self.natural_closure_count, 48),
            (self.images_with_multiple_distinct_raw_outputs, 12),
            (self.prediction_set_diversity_image_count, 12),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value <= upper
            ):
                _artifact_error("calibration panel count is invalid", "panel_count")
        if (
            not math.isfinite(self.prediction_set_diversity)
            or not 0.0 <= self.prediction_set_diversity <= 1.0
        ):
            _artifact_error(
                "prediction-set diversity is outside [0, 1]", "panel_diversity"
            )
        expected_pass = (
            self.parse_success_count >= CALIBRATION_PARSE_MINIMUM
            and self.natural_closure_count >= CALIBRATION_NATURAL_CLOSURE_MINIMUM
            and self.images_with_multiple_distinct_raw_outputs
            >= CALIBRATION_DISTINCT_IMAGE_MINIMUM
            and self.prediction_set_diversity_image_count > 0
            and self.prediction_set_diversity >= CALIBRATION_DIVERSITY_MINIMUM
        )
        if self.gate_passed != expected_pass:
            _artifact_error("calibration gate result does not recompute", "panel_gate")

    @property
    def fingerprint(self) -> str:
        return sha256_payload(self.to_artifact_dict())

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "decode_generation_policy_fingerprint": self.decode_generation_policy_fingerprint,
            "gate_passed": self.gate_passed,
            "images_with_multiple_distinct_raw_outputs": self.images_with_multiple_distinct_raw_outputs,
            "natural_closure_count": self.natural_closure_count,
            "ordered_observation_sha256s": list(self.ordered_observation_sha256s),
            "panel_kind": self.panel_kind,
            "parse_success_count": self.parse_success_count,
            "prediction_set_diversity": self.prediction_set_diversity,
            "prediction_set_diversity_image_count": self.prediction_set_diversity_image_count,
            "schema_version": self.schema_version,
            "temperature": self.temperature,
        }

    @classmethod
    def from_artifact_dict(cls, value: Mapping[str, Any]) -> CalibrationPanelReceipt:
        _require_exact_keys(
            value,
            {
                "decode_generation_policy_fingerprint",
                "gate_passed",
                "images_with_multiple_distinct_raw_outputs",
                "natural_closure_count",
                "ordered_observation_sha256s",
                "panel_kind",
                "parse_success_count",
                "prediction_set_diversity",
                "prediction_set_diversity_image_count",
                "schema_version",
                "temperature",
            },
            "calibration panel receipt",
        )
        payload = dict(value)
        payload["ordered_observation_sha256s"] = tuple(
            payload["ordered_observation_sha256s"]
        )
        return cls(**payload)


@dataclass(frozen=True)
class CalibrationCandidateEvaluation:
    """One initial candidate decision and optional exact confirmation panels."""

    temperature: float
    decode_generation_policy_fingerprint: str
    initial_panel: CalibrationPanelReceipt
    exact_replay_panel: CalibrationPanelReceipt | None
    reversed_order_panel: CalibrationPanelReceipt | None
    exact_replay_byte_identical: bool | None
    reversed_order_byte_identical: bool | None
    selected: bool
    schema_version: str = CALIBRATION_CANDIDATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CALIBRATION_CANDIDATE_SCHEMA_VERSION:
            _artifact_error(
                "candidate evaluation schema is unsupported", "candidate_schema"
            )
        if self.temperature != self.initial_panel.temperature:
            _artifact_error(
                "candidate and panel temperatures differ", "candidate_temperature"
            )
        if (
            self.decode_generation_policy_fingerprint
            != self.initial_panel.decode_generation_policy_fingerprint
        ):
            _artifact_error("candidate and panel policies differ", "candidate_policy")
        confirmation = (self.exact_replay_panel, self.reversed_order_panel)
        flags = (self.exact_replay_byte_identical, self.reversed_order_byte_identical)
        if self.initial_panel.gate_passed:
            if any(panel is None for panel in confirmation) or any(
                flag is None for flag in flags
            ):
                _artifact_error(
                    "passing candidate lacks confirmation panels",
                    "candidate_confirmation",
                )
            assert self.exact_replay_panel is not None
            assert self.reversed_order_panel is not None
            if (
                self.exact_replay_panel.panel_kind != "exact_replay"
                or self.reversed_order_panel.panel_kind != "reversed_order"
                or self.exact_replay_panel.temperature != self.temperature
                or self.reversed_order_panel.temperature != self.temperature
                or self.exact_replay_panel.decode_generation_policy_fingerprint
                != self.decode_generation_policy_fingerprint
                or self.reversed_order_panel.decode_generation_policy_fingerprint
                != self.decode_generation_policy_fingerprint
                or not self.exact_replay_panel.gate_passed
                or not self.reversed_order_panel.gate_passed
            ):
                _artifact_error(
                    "candidate confirmation panel identity drifted",
                    "candidate_confirmation",
                )
            if self.selected != (flags == (True, True)):
                _artifact_error(
                    "candidate selection differs from replay invariance",
                    "candidate_selection",
                )
        elif (
            any(value is not None for value in (*confirmation, *flags)) or self.selected
        ):
            _artifact_error(
                "failed candidate cannot carry confirmation evidence",
                "candidate_failed",
            )

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "decode_generation_policy_fingerprint": self.decode_generation_policy_fingerprint,
            "exact_replay_byte_identical": self.exact_replay_byte_identical,
            "exact_replay_panel": None
            if self.exact_replay_panel is None
            else self.exact_replay_panel.to_artifact_dict(),
            "initial_panel": self.initial_panel.to_artifact_dict(),
            "reversed_order_byte_identical": self.reversed_order_byte_identical,
            "reversed_order_panel": None
            if self.reversed_order_panel is None
            else self.reversed_order_panel.to_artifact_dict(),
            "schema_version": self.schema_version,
            "selected": self.selected,
            "temperature": self.temperature,
        }

    @classmethod
    def from_artifact_dict(
        cls, value: Mapping[str, Any]
    ) -> CalibrationCandidateEvaluation:
        _require_exact_keys(
            value,
            {
                "decode_generation_policy_fingerprint",
                "exact_replay_byte_identical",
                "exact_replay_panel",
                "initial_panel",
                "reversed_order_byte_identical",
                "reversed_order_panel",
                "schema_version",
                "selected",
                "temperature",
            },
            "calibration candidate evaluation",
        )
        payload = dict(value)
        payload["initial_panel"] = CalibrationPanelReceipt.from_artifact_dict(
            payload["initial_panel"]
        )
        for field in ("exact_replay_panel", "reversed_order_panel"):
            if payload[field] is not None:
                payload[field] = CalibrationPanelReceipt.from_artifact_dict(
                    payload[field]
                )
        return cls(**payload)


@dataclass(frozen=True)
class CalibrationSelectionReceipt:
    """Immutable first-passing policy selection from the separate non-metric lane."""

    calibration_cohort_sha256: str
    validation_cohort_sha256: str
    attested_policy_set: AttestedSamplingPolicySet
    candidate_evaluations: tuple[CalibrationCandidateEvaluation, ...]
    calibration_observations_sha256: str
    selected_temperature: float
    selected_decode_generation_policy_fingerprint: str
    total_model_call_count: int
    metric_eligible: Literal[False] = False
    artifact_role: str = "non-metric sampling calibration selection receipt"
    receipt_sha256: str | None = None
    schema_version: str = CALIBRATION_SELECTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CALIBRATION_SELECTION_SCHEMA_VERSION:
            _artifact_error(
                "calibration selection schema is unsupported", "selection_schema"
            )
        for field_name in (
            "calibration_cohort_sha256",
            "calibration_observations_sha256",
            "validation_cohort_sha256",
        ):
            _require_sha256(getattr(self, field_name), field_name)
        if self.metric_eligible is not False:
            _artifact_error(
                "calibration receipt must remain non-metric", "selection_metric_lane"
            )
        if self.artifact_role != "non-metric sampling calibration selection receipt":
            _artifact_error(
                "calibration artifact role drifted", "selection_artifact_role"
            )
        temperatures = tuple(item.temperature for item in self.candidate_evaluations)
        expected_prefix = CALIBRATION_TEMPERATURES[: len(temperatures)]
        if not self.candidate_evaluations or temperatures != expected_prefix:
            _artifact_error(
                "candidate evaluations do not follow frozen order", "selection_order"
            )
        selected_indexes = [
            index
            for index, item in enumerate(self.candidate_evaluations)
            if item.selected
        ]
        if selected_indexes != [len(self.candidate_evaluations) - 1]:
            _artifact_error(
                "selection must be the first and final passing candidate",
                "selection_first_pass",
            )
        if any(
            item.initial_panel.gate_passed for item in self.candidate_evaluations[:-1]
        ):
            _artifact_error(
                "a higher candidate was evaluated after an earlier pass",
                "selection_post_pass",
            )
        chosen = self.candidate_evaluations[-1]
        if (
            self.selected_temperature != chosen.temperature
            or self.selected_decode_generation_policy_fingerprint
            != chosen.decode_generation_policy_fingerprint
            or self.attested_policy_set.policy_fingerprint(self.selected_temperature)
            != self.selected_decode_generation_policy_fingerprint
        ):
            _artifact_error(
                "selected policy does not bind its attestation", "selection_policy"
            )
        expected_calls = len(self.candidate_evaluations) * 48 + 96
        if (
            self.total_model_call_count != expected_calls
            or expected_calls > CALIBRATION_MAX_CALL_COUNT
        ):
            _artifact_error(
                "calibration call budget does not recompute", "selection_call_budget"
            )
        if self.receipt_sha256 is None:
            object.__setattr__(
                self, "receipt_sha256", sha256_payload(self.identity_payload())
            )
        _require_sha256(self.receipt_sha256, "receipt_sha256")
        if self.receipt_sha256 != sha256_payload(self.identity_payload()):
            _artifact_error(
                "calibration selection fingerprint is invalid", "selection_fingerprint"
            )

    def identity_payload(self) -> dict[str, Any]:
        payload = self.to_artifact_dict()
        payload.pop("receipt_sha256")
        return payload

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "artifact_role": self.artifact_role,
            "attested_policy_set": self.attested_policy_set.to_artifact_dict(),
            "calibration_cohort_sha256": self.calibration_cohort_sha256,
            "calibration_observations_sha256": self.calibration_observations_sha256,
            "candidate_evaluations": [
                item.to_artifact_dict() for item in self.candidate_evaluations
            ],
            "metric_eligible": self.metric_eligible,
            "receipt_sha256": self.receipt_sha256,
            "schema_version": self.schema_version,
            "selected_decode_generation_policy_fingerprint": self.selected_decode_generation_policy_fingerprint,
            "selected_temperature": self.selected_temperature,
            "total_model_call_count": self.total_model_call_count,
            "validation_cohort_sha256": self.validation_cohort_sha256,
        }

    @classmethod
    def from_artifact_dict(
        cls, value: Mapping[str, Any]
    ) -> CalibrationSelectionReceipt:
        _require_exact_keys(
            value,
            {
                "artifact_role",
                "attested_policy_set",
                "calibration_cohort_sha256",
                "calibration_observations_sha256",
                "candidate_evaluations",
                "metric_eligible",
                "receipt_sha256",
                "schema_version",
                "selected_decode_generation_policy_fingerprint",
                "selected_temperature",
                "total_model_call_count",
                "validation_cohort_sha256",
            },
            "calibration selection receipt",
        )
        payload = dict(value)
        payload["attested_policy_set"] = AttestedSamplingPolicySet.from_artifact_dict(
            payload["attested_policy_set"]
        )
        payload["candidate_evaluations"] = tuple(
            CalibrationCandidateEvaluation.from_artifact_dict(row)
            for row in payload["candidate_evaluations"]
        )
        return cls(**payload)


def load_attested_sampling_policy_set(
    aggregate_path: Path,
) -> AttestedSamplingPolicySet:
    """Validate and reduce one typed three-policy runtime attestation aggregate."""

    from src.inference.backend import (  # imported lazily to keep CPU contracts light
        validate_sampled_runtime_attestation_aggregate_output,
    )

    path = aggregate_path.expanduser().resolve()
    validation = validate_sampled_runtime_attestation_aggregate_output(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("policy_attestations")
    if not isinstance(entries, list):
        _artifact_error(
            "attestation aggregate has no policy entries", "aggregate_entries"
        )
    pairs = tuple(
        (
            entry.get("temperature"),
            entry.get("decode_generation_policy_fingerprint"),
        )
        for entry in entries
    )
    return AttestedSamplingPolicySet(
        aggregate_artifact_sha256=sha256_file(path),
        aggregate_payload_fingerprint=validation["aggregate_payload_fingerprint"],
        policy_fingerprints_by_temperature=pairs,  # type: ignore[arg-type]
    )


def load_calibration_backend_attestation_binding(
    aggregate_path: Path,
) -> CalibrationBackendAttestationBinding:
    """Load exact backend identities from result-bound aggregate evidence."""

    path = aggregate_path.expanduser().resolve()
    validation = validate_sampled_runtime_attestation_aggregate_output(path)
    # The inference backend owns and validates this artifact.  Its producer uses
    # the backend's human-readable aggregate serialization, not this module's
    # compact research-receipt serialization.
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("policy_attestations")
    if not isinstance(entries, list):
        _artifact_error(
            "runtime attestation aggregate has no policy entries",
            "backend_binding_entries",
        )
    contracts: list[tuple[str, tuple[tuple[str, Any], ...]]] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            _artifact_error(
                "runtime attestation policy entry is not an object",
                "backend_binding_entry",
            )
        policy_fingerprint = entry.get("decode_generation_policy_fingerprint")
        _require_sha256(policy_fingerprint, "decode_generation_policy_fingerprint")
        bundle = entry.get("attestation_bundle")
        if not isinstance(bundle, Mapping):
            _artifact_error(
                "runtime attestation policy lacks its executed bundle",
                "backend_binding_bundle",
            )
        cases = bundle.get("executed_cases")
        if not isinstance(cases, list):
            _artifact_error(
                "runtime attestation bundle lacks executed cases",
                "backend_binding_cases",
            )
        observed_contracts: list[dict[str, Any]] = []
        for case in cases:
            if not isinstance(case, Mapping):
                _artifact_error(
                    "runtime attestation case is not an object",
                    "backend_binding_case",
                )
            results = case.get("result_artifacts")
            if not isinstance(results, list):
                _artifact_error(
                    "runtime attestation case lacks result artifacts",
                    "backend_binding_results",
                )
            for artifact in results:
                if not isinstance(artifact, Mapping):
                    _artifact_error(
                        "runtime attestation result is not an object",
                        "backend_binding_result",
                    )
                result = DecodeResult.from_artifact_dict(artifact)
                result.validate_for_scored()
                receipt = result.execution_receipt
                assert receipt is not None
                if receipt.decode_generation_policy_fingerprint != policy_fingerprint:
                    _artifact_error(
                        "runtime attestation result uses another policy",
                        "backend_binding_result_policy",
                    )
                observed_contracts.append(_decode_receipt_runtime_contract(receipt))
        if not observed_contracts or any(
            contract != observed_contracts[0] for contract in observed_contracts[1:]
        ):
            _artifact_error(
                "runtime attestation results do not share one execution identity",
                "backend_binding_result_identity",
            )
        contracts.append(
            (
                policy_fingerprint,
                tuple(sorted(observed_contracts[0].items())),
            )
        )
    return CalibrationBackendAttestationBinding(
        aggregate_artifact_sha256=sha256_file(path),
        aggregate_payload_fingerprint=validation["aggregate_payload_fingerprint"],
        execution_contracts_by_policy=tuple(contracts),
    )


def build_calibration_requests(
    *,
    cohort: CohortLedger,
    temperature: float,
    decode_generation_policy_fingerprint: str,
) -> tuple[CalibrationRequest, ...]:
    """Build twelve same-image batch-size-four groups in immutable image order."""

    _validate_calibration_cohort(cohort)
    requests: list[CalibrationRequest] = []
    for image in cohort.records:
        for call_index in range(CALIBRATION_SEEDS_PER_IMAGE):
            call_label = f"call-{call_index:02d}"
            payload = {
                "calibration_cohort_sha256": cohort.fingerprint,
                "call_index": call_index,
                "call_label": call_label,
                "decode_generation_policy_fingerprint": decode_generation_policy_fingerprint,
                "image_id": image.image_id,
                "image_sha256": image.image_sha256,
                "sampling_seed": derive_sampling_seed(
                    root_seed=PRIMARY_ROOT_SEED,
                    role="temperature-calibration",
                    image_id=image.image_id,
                    cell_or_call_label=call_label,
                ),
                "schema_version": CALIBRATION_REQUEST_SCHEMA_VERSION,
                "temperature": temperature,
            }
            requests.append(
                CalibrationRequest(
                    **payload,
                    image_frozen_order=image.frozen_order,
                    request_id="spatial-scope-history-calibration-request:"
                    + sha256_payload(payload),
                )
            )
    return tuple(requests)


def reconstruct_calibration_observation(
    *,
    terminal_bundle: CalibrationTerminalBundle,
    backend_attestation: CalibrationBackendAttestationBinding,
    source_width: int,
    source_height: int,
    official_reference_objects: Sequence[Any],
) -> CalibrationCallObservation:
    """Derive every calibration gate input from canonical execution artifacts."""

    if (
        terminal_bundle.backend_attestation_aggregate_sha256
        != backend_attestation.aggregate_artifact_sha256
        or terminal_bundle.backend_attestation_aggregate_fingerprint
        != backend_attestation.aggregate_payload_fingerprint
    ):
        _artifact_error(
            "calibration terminal bundle uses another runtime attestation",
            "terminal_bundle_attestation",
        )
    result = terminal_bundle.decode_result
    receipt = result.execution_receipt
    assert receipt is not None
    expected_runtime_contract = backend_attestation.execution_contract(
        terminal_bundle.request.decode_generation_policy_fingerprint
    )
    observed_runtime_contract = _decode_receipt_runtime_contract(receipt)
    mismatches = {
        field: {"expected": expected, "observed": observed_runtime_contract.get(field)}
        for field, expected in expected_runtime_contract.items()
        if observed_runtime_contract.get(field) != expected
    }
    if mismatches:
        _artifact_error(
            "calibration decode result differs from attested backend identity",
            "terminal_bundle_runtime_binding",
            mismatches=mismatches,
        )
    visual_receipt = terminal_bundle.visual_input_materialization_receipt
    if (
        visual_receipt["input_width"] != source_width
        or visual_receipt["input_height"] != source_height
    ):
        _artifact_error(
            "calibration visual input dimensions differ from the sealed cohort",
            "terminal_bundle_source_frame",
        )
    parsed = parse_compact_object_box_closed(
        result.parser_text,
        row_id=terminal_bundle.request.request_id,
        row_index=terminal_bundle.request.image_frozen_order,
        image_width=source_width,
        image_height=source_height,
    )
    canonical_trace = tuple(
        _canonical_calibration_trace(row) for row in result.token_trace
    )
    predictions: list[_CalibrationMatchedPrediction] = []
    from src.eval.detection_categories import normalize_coco_category_name

    for prediction in parsed.predictions:
        scored = score_prediction(
            row_id=terminal_bundle.request.request_id,
            prediction=prediction,
            token_trace=list(canonical_trace),
        )
        # Force selected-token score replay before any row can affect diversity.
        replay = scored.replay
        if not isinstance(replay, Mapping) or not replay.get("selected_logprobs"):
            _artifact_error(
                "calibration prediction lacks canonical selected-token scores",
                "terminal_bundle_selected_scores",
            )
        try:
            normalized_category = normalize_coco_category_name(
                str(prediction["description"])
            )
        except Exception:
            continue
        predictions.append(
            _CalibrationMatchedPrediction(
                image_id=str(terminal_bundle.request.image_id),
                prediction_id=str(prediction["object_span_id"]),
                normalized_category_name=normalized_category,
                global_bbox_xyxy=tuple(float(value) for value in prediction["bbox"]),  # type: ignore[arg-type]
            )
        )
    from src.analysis.spatial_scope_history.metrics import exact_reference_match

    match = exact_reference_match(predictions, official_reference_objects)
    return CalibrationCallObservation(
        panel_kind=terminal_bundle.panel_kind,
        request=terminal_bundle.request,
        physical_batch_index=terminal_bundle.physical_batch_index,
        request_execution_index=terminal_bundle.request_execution_index,
        raw_response_bytes_sha256=hashlib.sha256(
            result.raw_generated_text.encode("utf-8")
        ).hexdigest(),
        parse_without_call_level_failure=True,
        natural_closure=result.stop_reason == "im_end",
        detected_official_reference_object_ids=match.matched_reference_ids,
        decode_receipt_fingerprint=receipt.receipt_fingerprint,
        terminal_bundle_sha256=terminal_bundle.fingerprint,
    )


def select_sampling_calibration_from_terminal_bundles(
    *,
    calibration_cohort: CohortLedger,
    validation_cohort: CohortLedger,
    attested_policy_set: AttestedSamplingPolicySet,
    backend_attestation: CalibrationBackendAttestationBinding,
    terminal_bundles: Sequence[CalibrationTerminalBundle],
    official_reference_objects_by_image_id: Mapping[int, Sequence[Any]],
) -> CalibrationSelectionReceipt:
    """Select only after reconstructing every gate input from terminal artifacts."""

    _validate_calibration_and_validation_cohorts(
        calibration_cohort=calibration_cohort,
        validation_cohort=validation_cohort,
    )
    if (
        backend_attestation.aggregate_artifact_sha256
        != attested_policy_set.aggregate_artifact_sha256
        or backend_attestation.aggregate_payload_fingerprint
        != attested_policy_set.aggregate_payload_fingerprint
    ):
        _artifact_error(
            "calibration policy and response attestations differ",
            "terminal_bundle_attestation_set",
        )
    cohort_by_image_id = {row.image_id: row for row in calibration_cohort.records}
    rows: list[CalibrationCallObservation] = []
    for bundle in terminal_bundles:
        cohort_row = cohort_by_image_id.get(bundle.request.image_id)
        if cohort_row is None or (
            cohort_row.image_sha256 != bundle.request.image_sha256
            or cohort_row.frozen_order != bundle.request.image_frozen_order
        ):
            _artifact_error(
                "calibration terminal bundle belongs to another cohort",
                "terminal_bundle_cohort",
            )
        references = official_reference_objects_by_image_id.get(bundle.request.image_id)
        if references is None:
            _artifact_error(
                "calibration terminal bundle lacks sealed official references",
                "terminal_bundle_references",
            )
        rows.append(
            reconstruct_calibration_observation(
                terminal_bundle=bundle,
                backend_attestation=backend_attestation,
                source_width=cohort_row.source_width,
                source_height=cohort_row.source_height,
                official_reference_objects=references,
            )
        )
    return _select_sampling_calibration_from_reconstructed_observations(
        calibration_cohort=calibration_cohort,
        validation_cohort=validation_cohort,
        attested_policy_set=attested_policy_set,
        observations=rows,
    )


def _select_sampling_calibration_from_reconstructed_observations(
    *,
    calibration_cohort: CohortLedger,
    validation_cohort: CohortLedger,
    attested_policy_set: AttestedSamplingPolicySet,
    observations: Sequence[CalibrationCallObservation],
) -> CalibrationSelectionReceipt:
    """Pure selection core; callers must reconstruct rows from terminal bundles."""

    _validate_calibration_and_validation_cohorts(
        calibration_cohort=calibration_cohort,
        validation_cohort=validation_cohort,
    )
    rows = tuple(observations)
    grouped: dict[
        tuple[float, CalibrationPanelKind], list[CalibrationCallObservation]
    ] = {}
    for row in rows:
        grouped.setdefault((row.request.temperature, row.panel_kind), []).append(row)
    evaluations: list[CalibrationCandidateEvaluation] = []
    executed_observations: list[CalibrationCallObservation] = []
    for temperature in CALIBRATION_TEMPERATURES:
        policy = DecodeGenerationPolicy.sampled(
            max_new_tokens=512,
            repetition_penalty=1.0,
            temperature=temperature,
            top_p=0.95,
        )
        if policy.fingerprint != attested_policy_set.policy_fingerprint(temperature):
            _artifact_error(
                "frozen policy differs from aggregate attestation",
                "policy_fingerprint",
            )
        requests = build_calibration_requests(
            cohort=calibration_cohort,
            temperature=temperature,
            decode_generation_policy_fingerprint=policy.fingerprint,
        )
        initial = _ordered_materialized_panel(
            grouped=grouped,
            requests=requests,
            panel_kind="initial",
        )
        executed_observations.extend(initial)
        initial_panel = summarize_calibration_panel(initial)
        if not initial_panel.gate_passed:
            evaluations.append(
                CalibrationCandidateEvaluation(
                    temperature=temperature,
                    decode_generation_policy_fingerprint=policy.fingerprint,
                    initial_panel=initial_panel,
                    exact_replay_panel=None,
                    reversed_order_panel=None,
                    exact_replay_byte_identical=None,
                    reversed_order_byte_identical=None,
                    selected=False,
                )
            )
            continue
        replay = _ordered_materialized_panel(
            grouped=grouped,
            requests=requests,
            panel_kind="exact_replay",
        )
        reversed_rows = _ordered_materialized_panel(
            grouped=grouped,
            requests=requests,
            panel_kind="reversed_order",
        )
        executed_observations.extend(replay)
        executed_observations.extend(reversed_rows)
        if not _responses_are_byte_identical(initial, replay) or not (
            _responses_are_byte_identical(initial, reversed_rows)
        ):
            _artifact_error(
                "sampled backend failed exact same-request replay or order invariance",
                "calibration_replay_invariance",
            )
        evaluations.append(
            CalibrationCandidateEvaluation(
                temperature=temperature,
                decode_generation_policy_fingerprint=policy.fingerprint,
                initial_panel=initial_panel,
                exact_replay_panel=summarize_calibration_panel(replay),
                reversed_order_panel=summarize_calibration_panel(reversed_rows),
                exact_replay_byte_identical=True,
                reversed_order_byte_identical=True,
                selected=True,
            )
        )
        receipt = CalibrationSelectionReceipt(
            calibration_cohort_sha256=calibration_cohort.fingerprint,
            validation_cohort_sha256=validation_cohort.fingerprint,
            attested_policy_set=attested_policy_set,
            candidate_evaluations=tuple(evaluations),
            calibration_observations_sha256=sha256_payload(
                [row.terminal_bundle_sha256 for row in executed_observations]
            ),
            selected_temperature=temperature,
            selected_decode_generation_policy_fingerprint=policy.fingerprint,
            total_model_call_count=len(executed_observations),
        )
        expected_panel_keys = {
            (evaluation.temperature, "initial") for evaluation in evaluations
        } | {
            (temperature, "exact_replay"),
            (temperature, "reversed_order"),
        }
        if set(grouped) != expected_panel_keys or len(rows) != len(
            executed_observations
        ):
            _artifact_error(
                "terminal bundles include missing or post-selection panels",
                "terminal_bundle_collection_scope",
            )
        return receipt
    _artifact_error(
        "no frozen temperature passed the calibration mechanics gates",
        "calibration_no_candidate",
    )


def summarize_calibration_panel(
    observations: Sequence[CalibrationCallObservation],
) -> CalibrationPanelReceipt:
    """Validate one exact panel and compute only predeclared mechanics gates."""

    rows = tuple(observations)
    if len(rows) != CALIBRATION_INITIAL_CALL_COUNT:
        _artifact_error("calibration panel must contain 48 observations", "panel_size")
    panel_kinds = {row.panel_kind for row in rows}
    temperatures = {row.request.temperature for row in rows}
    policies = {row.request.decode_generation_policy_fingerprint for row in rows}
    if len(panel_kinds) != 1 or len(temperatures) != 1 or len(policies) != 1:
        _artifact_error("calibration panel mixed identities", "panel_mixed_identity")
    if len({row.request.request_id for row in rows}) != CALIBRATION_INITIAL_CALL_COUNT:
        _artifact_error(
            "calibration panel repeats a request identity", "panel_request_identity"
        )
    expected_image_orders = tuple(
        image_order
        for image_order in range(CALIBRATION_IMAGE_COUNT)
        for _ in range(CALIBRATION_SEEDS_PER_IMAGE)
    )
    if tuple(row.request.image_frozen_order for row in rows) != expected_image_orders:
        _artifact_error(
            "calibration image order differs from the frozen cohort",
            "panel_image_order",
        )
    panel_kind = next(iter(panel_kinds))
    expected_execution_indexes = (0, 1, 2, 3)
    by_image: dict[int, list[CalibrationCallObservation]] = {}
    for row in rows:
        by_image.setdefault(row.request.image_id, []).append(row)
    if len(by_image) != CALIBRATION_IMAGE_COUNT:
        _artifact_error("calibration panel must contain twelve images", "panel_images")
    for batch_index, image_rows in enumerate(by_image.values()):
        if (
            len(image_rows) != 4
            or {row.physical_batch_index for row in image_rows} != {batch_index}
            or {row.request.image_frozen_order for row in image_rows} != {batch_index}
        ):
            _artifact_error(
                "each calibration image must occupy one batch-size-four batch",
                "panel_batch",
            )
        observed_call_indexes = tuple(row.request.call_index for row in image_rows)
        expected_call_indexes = (
            (3, 2, 1, 0) if panel_kind == "reversed_order" else (0, 1, 2, 3)
        )
        if observed_call_indexes != expected_call_indexes:
            _artifact_error(
                "calibration request order differs from panel contract", "panel_order"
            )
        if (
            tuple(row.request_execution_index for row in image_rows)
            != expected_execution_indexes
        ):
            _artifact_error(
                "calibration execution index differs from request order",
                "panel_execution_order",
            )
    parse_count = sum(row.parse_without_call_level_failure for row in rows)
    closure_count = sum(row.natural_closure for row in rows)
    distinct_images = sum(
        len({row.raw_response_bytes_sha256 for row in image_rows}) >= 2
        for image_rows in by_image.values()
    )
    per_image_diversities: list[float] = []
    for image_rows in by_image.values():
        sets = [set(row.detected_official_reference_object_ids) for row in image_rows]
        if not set().union(*sets):
            continue
        similarities: list[float] = []
        for left_index in range(4):
            for right_index in range(left_index + 1, 4):
                union = sets[left_index] | sets[right_index]
                similarity = (
                    1.0
                    if not union
                    else len(sets[left_index] & sets[right_index]) / len(union)
                )
                similarities.append(similarity)
        per_image_diversities.append(1.0 - sum(similarities) / len(similarities))
    diversity = (
        0.0
        if not per_image_diversities
        else sum(per_image_diversities) / len(per_image_diversities)
    )
    passed = (
        parse_count >= CALIBRATION_PARSE_MINIMUM
        and closure_count >= CALIBRATION_NATURAL_CLOSURE_MINIMUM
        and distinct_images >= CALIBRATION_DISTINCT_IMAGE_MINIMUM
        and bool(per_image_diversities)
        and diversity >= CALIBRATION_DIVERSITY_MINIMUM
    )
    return CalibrationPanelReceipt(
        panel_kind=panel_kind,
        temperature=next(iter(temperatures)),
        decode_generation_policy_fingerprint=next(iter(policies)),
        ordered_observation_sha256s=tuple(row.fingerprint for row in rows),
        parse_success_count=parse_count,
        natural_closure_count=closure_count,
        images_with_multiple_distinct_raw_outputs=distinct_images,
        prediction_set_diversity=diversity,
        prediction_set_diversity_image_count=len(per_image_diversities),
        gate_passed=passed,
    )


def _source_runtime_repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _canonical_pretty_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _run_git(*arguments: str, repository_root: Path) -> bytes:
    completed = subprocess.run(
        ("git", "-C", str(repository_root), *arguments),
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        _artifact_error(
            "source/runtime identity cannot verify committed source",
            "identity_git_verification",
            git_arguments=list(arguments),
            stderr=completed.stderr.decode("utf-8", errors="replace"),
        )
    return completed.stdout


def _live_committed_source_identity() -> tuple[str, tuple[tuple[str, str], ...]]:
    repository_root = _source_runtime_repository_root()
    git_head_commit = (
        _run_git("rev-parse", "--verify", "HEAD", repository_root=repository_root)
        .decode("ascii")
        .strip()
    )
    if len(git_head_commit) != 40 or any(
        character not in "0123456789abcdef" for character in git_head_commit
    ):
        _artifact_error(
            "source/runtime identity requires a canonical committed git HEAD",
            "identity_git_head",
            git_head_commit=git_head_commit,
        )
    critical_hashes: list[tuple[str, str]] = []
    for relative_path in _SOURCE_RUNTIME_CRITICAL_PATHS:
        path = repository_root / relative_path
        if not path.is_file():
            _artifact_error(
                "source/runtime critical source is absent",
                "identity_critical_source",
                relative_path=relative_path,
            )
        committed_bytes = _run_git(
            "show",
            f"{git_head_commit}:{relative_path}",
            repository_root=repository_root,
        )
        current_bytes = path.read_bytes()
        if current_bytes != committed_bytes:
            _artifact_error(
                "source/runtime critical source differs from committed HEAD",
                "identity_critical_source_dirty",
                relative_path=relative_path,
            )
        critical_hashes.append(
            (relative_path, hashlib.sha256(current_bytes).hexdigest())
        )
    return git_head_commit, tuple(critical_hashes)


def _live_source_attestation_bytes() -> bytes:
    from scripts.research.attest_request_scoped_sampling_source import (
        build_source_attestation,
    )

    return _canonical_pretty_json_bytes(build_source_attestation())


def _derived_source_code_sha256(
    *,
    git_head_commit: str,
    critical_source_sha256s: Sequence[tuple[str, str]],
    source_attestation_sha256: str,
) -> str:
    return sha256_payload(
        {
            "critical_source_sha256s": dict(critical_source_sha256s),
            "git_head_commit": git_head_commit,
            "source_attestation_sha256": source_attestation_sha256,
        }
    )


def _derived_source_runtime_sha256(
    *,
    source_attestation_sha256: str,
    aggregate_sha256: str,
    aggregate_fingerprint: str,
    config_sha256: str,
    checkpoint_manifest_sha256: str,
) -> str:
    return sha256_payload(
        {
            "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
            "config_sha256": config_sha256,
            "sampled_runtime_attestation_aggregate_fingerprint": (
                aggregate_fingerprint
            ),
            "sampled_runtime_attestation_aggregate_sha256": aggregate_sha256,
            "source_attestation_sha256": source_attestation_sha256,
        }
    )


@dataclass(frozen=True)
class SourceRuntimeIdentityReceipt:
    """Verifier-derived identity for the exact committed sampling runtime."""

    code_sha256: str
    config_sha256: str
    checkpoint_manifest_sha256: str
    ledger_seal_sha256: str
    runtime_sha256: str
    source_attestation_sha256: str
    sampled_runtime_attestation_aggregate_sha256: str
    sampled_runtime_attestation_aggregate_fingerprint: str
    processor_contract_sha256: str
    git_head_commit: str
    critical_source_sha256s: tuple[tuple[str, str], ...]
    receipt_sha256: str | None = None
    schema_version: str = SOURCE_RUNTIME_IDENTITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SOURCE_RUNTIME_IDENTITY_SCHEMA_VERSION:
            _artifact_error(
                "source/runtime identity schema is unsupported", "identity_schema"
            )
        for field_name in (
            "code_sha256",
            "config_sha256",
            "checkpoint_manifest_sha256",
            "ledger_seal_sha256",
            "runtime_sha256",
            "source_attestation_sha256",
            "sampled_runtime_attestation_aggregate_sha256",
            "sampled_runtime_attestation_aggregate_fingerprint",
            "processor_contract_sha256",
        ):
            _require_sha256(getattr(self, field_name), field_name)
        if self.config_sha256 != FROZEN_RESOLVED_INFERENCE_CONFIG_SHA256:
            _artifact_error(
                "source/runtime identity uses another resolved inference config",
                "identity_config_pin",
            )
        if self.checkpoint_manifest_sha256 != FROZEN_CHECKPOINT_MANIFEST_SHA256:
            _artifact_error(
                "source/runtime identity uses another checkpoint manifest",
                "identity_checkpoint_pin",
            )
        if self.ledger_seal_sha256 != FROZEN_READINESS_LEDGER_SEAL_SHA256:
            _artifact_error(
                "source/runtime identity uses another readiness ledger seal",
                "identity_ledger_pin",
            )
        if (
            SpatialGridSpec().fingerprint != FROZEN_SPATIAL_GRID_SPEC_SHA256
            or self.processor_contract_sha256 != FROZEN_SPATIAL_GRID_SPEC_SHA256
        ):
            _artifact_error(
                "source/runtime identity uses another processor contract",
                "identity_processor_pin",
            )
        if len(self.git_head_commit) != 40 or any(
            character not in "0123456789abcdef" for character in self.git_head_commit
        ):
            _artifact_error(
                "source/runtime identity git HEAD is invalid", "identity_git_head"
            )
        expected_paths = tuple(sorted(_SOURCE_RUNTIME_CRITICAL_PATHS))
        observed_paths = tuple(path for path, _ in self.critical_source_sha256s)
        if observed_paths != expected_paths:
            _artifact_error(
                "source/runtime identity critical source set is not exact",
                "identity_critical_source_set",
            )
        for path, digest in self.critical_source_sha256s:
            if not isinstance(path, str) or not path:
                _artifact_error(
                    "source/runtime critical source path is invalid",
                    "identity_critical_source_set",
                )
            _require_sha256(digest, f"critical_source_sha256s[{path}]")
        if self.code_sha256 != _derived_source_code_sha256(
            git_head_commit=self.git_head_commit,
            critical_source_sha256s=self.critical_source_sha256s,
            source_attestation_sha256=self.source_attestation_sha256,
        ):
            _artifact_error(
                "source/runtime code identity is not verifier-derived",
                "identity_code_derivation",
            )
        if self.runtime_sha256 != _derived_source_runtime_sha256(
            source_attestation_sha256=self.source_attestation_sha256,
            aggregate_sha256=self.sampled_runtime_attestation_aggregate_sha256,
            aggregate_fingerprint=(
                self.sampled_runtime_attestation_aggregate_fingerprint
            ),
            config_sha256=self.config_sha256,
            checkpoint_manifest_sha256=self.checkpoint_manifest_sha256,
        ):
            _artifact_error(
                "source/runtime runtime identity is not verifier-derived",
                "identity_runtime_derivation",
            )
        if self.receipt_sha256 is None:
            object.__setattr__(
                self, "receipt_sha256", sha256_payload(self.identity_payload())
            )
        _require_sha256(self.receipt_sha256, "receipt_sha256")
        if self.receipt_sha256 != sha256_payload(self.identity_payload()):
            _artifact_error(
                "source/runtime identity fingerprint is invalid", "identity_fingerprint"
            )

    @property
    def execution_identity(self) -> ExecutionIdentityBundle:
        return ExecutionIdentityBundle(
            code_sha256=self.code_sha256,
            config_sha256=self.config_sha256,
            ledger_sha256=self.ledger_seal_sha256,
            runtime_sha256=self.runtime_sha256,
        )

    def identity_payload(self) -> dict[str, Any]:
        payload = self.to_artifact_dict()
        payload.pop("receipt_sha256")
        return payload

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_manifest_sha256": self.checkpoint_manifest_sha256,
            "code_sha256": self.code_sha256,
            "config_sha256": self.config_sha256,
            "critical_source_sha256s": {
                path: digest for path, digest in self.critical_source_sha256s
            },
            "git_head_commit": self.git_head_commit,
            "ledger_seal_sha256": self.ledger_seal_sha256,
            "processor_contract_sha256": self.processor_contract_sha256,
            "receipt_sha256": self.receipt_sha256,
            "runtime_sha256": self.runtime_sha256,
            "sampled_runtime_attestation_aggregate_fingerprint": self.sampled_runtime_attestation_aggregate_fingerprint,
            "sampled_runtime_attestation_aggregate_sha256": self.sampled_runtime_attestation_aggregate_sha256,
            "schema_version": self.schema_version,
            "source_attestation_sha256": self.source_attestation_sha256,
        }

    def validate_live_source(self) -> None:
        live_source_attestation = _live_source_attestation_bytes()
        if hashlib.sha256(live_source_attestation).hexdigest() != (
            self.source_attestation_sha256
        ):
            _artifact_error(
                "source attestation no longer matches live installed source",
                "identity_source_attestation_live",
            )
        git_head_commit, critical_source_sha256s = _live_committed_source_identity()
        if (
            git_head_commit != self.git_head_commit
            or critical_source_sha256s != self.critical_source_sha256s
        ):
            _artifact_error(
                "source/runtime identity no longer matches committed source",
                "identity_committed_source_live",
            )

    @classmethod
    def from_verified_artifacts(
        cls,
        *,
        source_attestation_path: Path,
        sampled_runtime_attestation_aggregate_path: Path,
        resolved_inference_config_path: Path,
        checkpoint_manifest_path: Path,
        ledger_seal_sha256: str,
        processor_contract_sha256: str,
    ) -> SourceRuntimeIdentityReceipt:
        source_path = source_attestation_path.expanduser().resolve()
        aggregate_path = (
            sampled_runtime_attestation_aggregate_path.expanduser().resolve()
        )
        config_path = resolved_inference_config_path.expanduser().resolve()
        checkpoint_path = checkpoint_manifest_path.expanduser().resolve()
        for path, label in (
            (source_path, "source attestation"),
            (aggregate_path, "sampled runtime attestation aggregate"),
            (config_path, "resolved inference config"),
            (checkpoint_path, "checkpoint manifest"),
        ):
            if not path.is_file():
                _artifact_error(
                    f"{label} is absent", "identity_source_artifact", path=str(path)
                )
        config_sha256 = sha256_file(config_path)
        checkpoint_manifest_sha256 = sha256_file(checkpoint_path)
        if config_sha256 != FROZEN_RESOLVED_INFERENCE_CONFIG_SHA256:
            _artifact_error(
                "resolved inference config is not the frozen primary config",
                "identity_config_pin",
            )
        if checkpoint_manifest_sha256 != FROZEN_CHECKPOINT_MANIFEST_SHA256:
            _artifact_error(
                "checkpoint manifest is not the frozen primary checkpoint",
                "identity_checkpoint_pin",
            )
        if ledger_seal_sha256 != FROZEN_READINESS_LEDGER_SEAL_SHA256:
            _artifact_error(
                "readiness ledger seal is not the frozen reviewed seal",
                "identity_ledger_pin",
            )
        if (
            SpatialGridSpec().fingerprint != FROZEN_SPATIAL_GRID_SPEC_SHA256
            or processor_contract_sha256 != FROZEN_SPATIAL_GRID_SPEC_SHA256
        ):
            _artifact_error(
                "processor contract is not the frozen spatial grid",
                "identity_processor_pin",
            )

        live_source_attestation = _live_source_attestation_bytes()
        if source_path.read_bytes() != live_source_attestation:
            _artifact_error(
                "source attestation artifact is not the exact live source attestation",
                "identity_source_attestation_artifact",
            )
        source_attestation_sha256 = hashlib.sha256(live_source_attestation).hexdigest()
        git_head_commit, critical_source_sha256s = _live_committed_source_identity()

        aggregate_validation = validate_sampled_runtime_attestation_aggregate_output(
            aggregate_path
        )
        aggregate_bytes = aggregate_path.read_bytes()
        aggregate_payload = json.loads(aggregate_bytes.decode("utf-8"))
        if not isinstance(aggregate_payload, Mapping):
            _artifact_error(
                "sampled runtime attestation aggregate is not an object",
                "identity_aggregate_shape",
            )
        _require_exact_keys(
            aggregate_payload,
            {
                "aggregate_payload_fingerprint",
                "policy_attestations",
                "schema_version",
            },
            "sampled runtime attestation aggregate",
        )
        if aggregate_bytes != _canonical_pretty_json_bytes(aggregate_payload):
            _artifact_error(
                "sampled runtime attestation aggregate is not canonical",
                "identity_aggregate_canonical",
            )
        entries = aggregate_payload.get("policy_attestations")
        if not isinstance(entries, list) or len(entries) != 3:
            _artifact_error(
                "sampled runtime attestation aggregate policy set is not exact",
                "identity_aggregate_shape",
            )
        checkpoint_payload_identity = verify_checkpoint_payload_identity(
            checkpoint_path
        )
        for entry_index, entry in enumerate(entries):
            if not isinstance(entry, Mapping):
                _artifact_error(
                    "sampled runtime attestation policy entry is invalid",
                    "identity_aggregate_shape",
                    entry_index=entry_index,
                )
            _require_exact_keys(
                entry,
                {
                    "admitted_production_replay",
                    "attestation_bundle",
                    "bundle_payload_fingerprint",
                    "decode_generation_policy",
                    "decode_generation_policy_fingerprint",
                    "entry_payload_fingerprint",
                    "schema_version",
                    "temperature",
                },
                "sampled runtime policy attestation entry",
            )
            bundle = entry.get("attestation_bundle")
            lineage = bundle.get("lineage") if isinstance(bundle, Mapping) else None
            if not isinstance(lineage, Mapping):
                _artifact_error(
                    "sampled runtime attestation lineage is absent",
                    "identity_aggregate_lineage",
                    entry_index=entry_index,
                )
            if (
                lineage.get("config_sha256") != config_sha256
                or lineage.get("checkpoint_manifest_sha256")
                != checkpoint_manifest_sha256
                or lineage.get("checkpoint_payload_identity")
                != checkpoint_payload_identity
            ):
                _artifact_error(
                    "sampled runtime attestation lineage uses different sources",
                    "identity_aggregate_lineage",
                    entry_index=entry_index,
                )
        aggregate_sha256 = sha256_file(aggregate_path)
        aggregate_fingerprint = aggregate_validation["aggregate_payload_fingerprint"]
        code_sha256 = _derived_source_code_sha256(
            git_head_commit=git_head_commit,
            critical_source_sha256s=critical_source_sha256s,
            source_attestation_sha256=source_attestation_sha256,
        )
        runtime_sha256 = _derived_source_runtime_sha256(
            source_attestation_sha256=source_attestation_sha256,
            aggregate_sha256=aggregate_sha256,
            aggregate_fingerprint=aggregate_fingerprint,
            config_sha256=config_sha256,
            checkpoint_manifest_sha256=checkpoint_manifest_sha256,
        )
        return cls(
            checkpoint_manifest_sha256=checkpoint_manifest_sha256,
            code_sha256=code_sha256,
            config_sha256=config_sha256,
            critical_source_sha256s=critical_source_sha256s,
            git_head_commit=git_head_commit,
            ledger_seal_sha256=ledger_seal_sha256,
            processor_contract_sha256=processor_contract_sha256,
            runtime_sha256=runtime_sha256,
            sampled_runtime_attestation_aggregate_fingerprint=(aggregate_fingerprint),
            sampled_runtime_attestation_aggregate_sha256=aggregate_sha256,
            source_attestation_sha256=source_attestation_sha256,
        )

    @classmethod
    def from_artifact_dict(
        cls, value: Mapping[str, Any]
    ) -> SourceRuntimeIdentityReceipt:
        _require_exact_keys(
            value,
            {
                "checkpoint_manifest_sha256",
                "code_sha256",
                "config_sha256",
                "critical_source_sha256s",
                "git_head_commit",
                "ledger_seal_sha256",
                "processor_contract_sha256",
                "receipt_sha256",
                "runtime_sha256",
                "sampled_runtime_attestation_aggregate_fingerprint",
                "sampled_runtime_attestation_aggregate_sha256",
                "schema_version",
                "source_attestation_sha256",
            },
            "source runtime identity receipt",
        )
        critical = value["critical_source_sha256s"]
        if not isinstance(critical, Mapping):
            _artifact_error(
                "source/runtime critical source hashes must be an object",
                "identity_critical_source_set",
            )
        payload = dict(value)
        payload["critical_source_sha256s"] = tuple(sorted(critical.items()))
        receipt = cls(**payload)
        receipt.validate_live_source()
        return receipt


@dataclass(frozen=True)
class PrimaryScheduleArtifact:
    """Immutable schedule plus every source hash needed to validate its origin."""

    schedule: ResearchSchedule
    cohort_artifact_name: str
    cohort_artifact_sha256: str
    readiness_ledger_seal_sha256: str
    calibration_selection_receipt_sha256: str
    calibration_selection_fingerprint: str
    source_runtime_identity_receipt_sha256: str
    source_runtime_identity_fingerprint: str
    source_hashes: tuple[tuple[str, str], ...]
    artifact_sha256: str | None = None
    schema_version: str = PRIMARY_SCHEDULE_ARTIFACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PRIMARY_SCHEDULE_ARTIFACT_SCHEMA_VERSION:
            _artifact_error(
                "primary schedule artifact schema is unsupported",
                "primary_schedule_schema",
            )
        if self.cohort_artifact_name not in {
            "cohort-manifest.jsonl",
            "dense-union-51-manifest.jsonl",
        }:
            _artifact_error(
                "primary schedule cohort artifact is unsupported",
                "primary_schedule_cohort",
            )
        for field_name in (
            "cohort_artifact_sha256",
            "readiness_ledger_seal_sha256",
            "calibration_selection_receipt_sha256",
            "calibration_selection_fingerprint",
            "source_runtime_identity_receipt_sha256",
            "source_runtime_identity_fingerprint",
        ):
            _require_sha256(getattr(self, field_name), field_name)
        if tuple(sorted(self.source_hashes)) != self.source_hashes or len(
            dict(self.source_hashes)
        ) != len(self.source_hashes):
            _artifact_error(
                "primary schedule source hashes must be sorted and unique",
                "primary_schedule_sources",
            )
        for name, digest in self.source_hashes:
            if not name:
                _artifact_error(
                    "primary schedule source name is empty", "primary_schedule_sources"
                )
            _require_sha256(digest, "source_hash")
        if self.artifact_sha256 is None:
            object.__setattr__(
                self, "artifact_sha256", sha256_payload(self.identity_payload())
            )
        _require_sha256(self.artifact_sha256, "artifact_sha256")
        if self.artifact_sha256 != sha256_payload(self.identity_payload()):
            _artifact_error(
                "primary schedule artifact fingerprint is invalid",
                "primary_schedule_fingerprint",
            )

    def identity_payload(self) -> dict[str, Any]:
        payload = self.to_artifact_dict()
        payload.pop("artifact_sha256")
        return payload

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "artifact_sha256": self.artifact_sha256,
            "calibration_selection_fingerprint": self.calibration_selection_fingerprint,
            "calibration_selection_receipt_sha256": self.calibration_selection_receipt_sha256,
            "cohort_artifact_name": self.cohort_artifact_name,
            "cohort_artifact_sha256": self.cohort_artifact_sha256,
            "readiness_ledger_seal_sha256": self.readiness_ledger_seal_sha256,
            "schedule": self.schedule.to_artifact_dict(),
            "schema_version": self.schema_version,
            "source_hashes": {name: digest for name, digest in self.source_hashes},
            "source_runtime_identity_fingerprint": self.source_runtime_identity_fingerprint,
            "source_runtime_identity_receipt_sha256": self.source_runtime_identity_receipt_sha256,
        }

    @classmethod
    def from_artifact_dict(cls, value: Mapping[str, Any]) -> PrimaryScheduleArtifact:
        _require_exact_keys(
            value,
            {
                "artifact_sha256",
                "calibration_selection_fingerprint",
                "calibration_selection_receipt_sha256",
                "cohort_artifact_name",
                "cohort_artifact_sha256",
                "readiness_ledger_seal_sha256",
                "schedule",
                "schema_version",
                "source_hashes",
                "source_runtime_identity_fingerprint",
                "source_runtime_identity_receipt_sha256",
            },
            "primary schedule artifact",
        )
        payload = dict(value)
        payload["schedule"] = ResearchSchedule.from_artifact_dict(payload["schedule"])
        if not isinstance(payload["source_hashes"], Mapping):
            _artifact_error(
                "primary schedule source hashes must be an object",
                "primary_schedule_sources",
            )
        payload["source_hashes"] = tuple(sorted(payload["source_hashes"].items()))
        return cls(**payload)


def materialize_primary_schedule_artifact(
    *,
    unit_id: str,
    run_id: str,
    cohort: CohortLedger,
    cohort_artifact_name: str,
    cohort_artifact_sha256: str,
    readiness_ledger_seal_sha256: str,
    calibration_selection: CalibrationSelectionReceipt,
    calibration_selection_receipt_sha256: str,
    source_runtime_identity: SourceRuntimeIdentityReceipt,
    source_runtime_identity_receipt_sha256: str,
    source_hashes: Mapping[str, str],
    root_seed: int = PRIMARY_ROOT_SEED,
) -> PrimaryScheduleArtifact:
    """Build and validate the exact 51-image smoke or 200-image primary schedule."""

    expected_count = {DENSE_COHORT_ID: 51, VALIDATION_COHORT_ID: 200}.get(
        cohort.cohort_id
    )
    if expected_count is None or len(cohort.records) != expected_count:
        _data_error(
            "schedule cohort is not exact Dense-Union-51 or Validation-200",
            "primary_schedule_cohort",
        )
    if source_runtime_identity.ledger_seal_sha256 != readiness_ledger_seal_sha256:
        _artifact_error(
            "source/runtime identity and readiness seal differ",
            "primary_schedule_ledger",
        )
    if (
        calibration_selection.attested_policy_set.aggregate_artifact_sha256
        != source_runtime_identity.sampled_runtime_attestation_aggregate_sha256
        or calibration_selection.attested_policy_set.aggregate_payload_fingerprint
        != source_runtime_identity.sampled_runtime_attestation_aggregate_fingerprint
    ):
        _artifact_error(
            "calibration and runtime identity use different attestation aggregates",
            "primary_schedule_attestation",
        )
    required_source_hashes = {
        cohort_artifact_name: cohort_artifact_sha256,
        "ledger-seal.json": readiness_ledger_seal_sha256,
        "calibration-selection-receipt.json": (calibration_selection_receipt_sha256),
        "source-runtime-identity-receipt.json": (
            source_runtime_identity_receipt_sha256
        ),
    }
    source_mismatches = {
        name: {"expected": expected, "observed": source_hashes.get(name)}
        for name, expected in required_source_hashes.items()
        if source_hashes.get(name) != expected
    }
    if source_mismatches:
        _artifact_error(
            "primary schedule source hash map is incomplete or inconsistent",
            "primary_schedule_source_binding",
            mismatches=source_mismatches,
        )
    if root_seed != PRIMARY_ROOT_SEED and cohort.cohort_id != DENSE_COHORT_ID:
        _data_error(
            "the second root seed is reserved for Dense-Union-51 replication",
            "primary_schedule_second_root_scope",
        )
    if calibration_selection.validation_cohort_sha256 != (
        cohort.fingerprint
        if cohort.cohort_id == VALIDATION_COHORT_ID
        else source_hashes.get("validation_cohort_fingerprint")
    ):
        _artifact_error(
            "calibration receipt and Validation-200 identity differ",
            "primary_schedule_validation_cohort",
        )
    decode = DecodeProvenance(
        temperature=calibration_selection.selected_temperature,
        canonical_generation_policy_sha256=calibration_selection.selected_decode_generation_policy_fingerprint,
        sampled_runtime_attestation_sha256=source_runtime_identity.sampled_runtime_attestation_aggregate_sha256,
    )
    grid = GridProvenance(
        canonical_spatial_spec_sha256=SpatialGridSpec().fingerprint,
        canonical_spatial_receipt_contract_sha256=source_runtime_identity.processor_contract_sha256,
    )
    schedule = ResearchSchedule.build_primary(
        unit_id=unit_id,
        run_id=run_id,
        cohort=cohort,
        root_seed=root_seed,
        decode=decode,
        execution_identity=source_runtime_identity.execution_identity,
        grid=grid,
    )
    expected = (
        (3_315, 833, 816, 17) if expected_count == 51 else (13_000, 3_250, 3_250, 0)
    )
    observed = (
        len(schedule.requests),
        len(schedule.batches()),
        sum(batch.cardinality == 4 for batch in schedule.batches()),
        sum(batch.cardinality == 3 for batch in schedule.batches()),
    )
    if observed != expected:
        _artifact_error(
            "primary schedule cardinalities differ from frozen contract",
            "primary_schedule_cardinality",
        )
    return PrimaryScheduleArtifact(
        schedule=schedule,
        cohort_artifact_name=cohort_artifact_name,
        cohort_artifact_sha256=cohort_artifact_sha256,
        readiness_ledger_seal_sha256=readiness_ledger_seal_sha256,
        calibration_selection_receipt_sha256=calibration_selection_receipt_sha256,
        calibration_selection_fingerprint=calibration_selection.receipt_sha256,
        source_runtime_identity_receipt_sha256=source_runtime_identity_receipt_sha256,
        source_runtime_identity_fingerprint=source_runtime_identity.receipt_sha256,
        source_hashes=tuple(sorted(source_hashes.items())),
    )


def write_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically install one canonical JavaScript Object Notation artifact once."""

    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    text = canonical_json_text(payload) + "\n"
    temporary = destination.with_name(
        f".{destination.name}.tmp-{os.getpid()}-{hashlib.sha256(text.encode()).hexdigest()[:12]}"
    )
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise ArtifactContractError(
                "refusing to overwrite immutable research artifact",
                code="analysis.calibration_immutable_output_exists",
                context={"path": str(destination)},
                cause=exc,
            ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def load_canonical_json(path: Path) -> Mapping[str, Any]:
    """Load only canonical object-form JavaScript Object Notation."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactContractError(
            "research receipt cannot be loaded",
            code="analysis.calibration_receipt_load",
            context={"path": str(path)},
            cause=exc,
        ) from exc
    if not isinstance(payload, Mapping) or canonical_json_text(
        payload
    ) + "\n" != path.read_text(encoding="utf-8"):
        _artifact_error(
            "research receipt must be a canonical JavaScript Object Notation object",
            "receipt_canonical",
        )
    return payload


def _responses_are_byte_identical(
    initial: Sequence[CalibrationCallObservation],
    comparison: Sequence[CalibrationCallObservation],
) -> bool:
    initial_by_request = {row.request.request_id: row for row in initial}
    comparison_by_request = {row.request.request_id: row for row in comparison}
    if set(initial_by_request) != set(comparison_by_request):
        return False
    return all(
        initial_by_request[request_id].request.sampling_seed
        == comparison_by_request[request_id].request.sampling_seed
        and initial_by_request[request_id].raw_response_bytes_sha256
        == comparison_by_request[request_id].raw_response_bytes_sha256
        and initial_by_request[request_id].parse_without_call_level_failure
        == comparison_by_request[request_id].parse_without_call_level_failure
        and initial_by_request[request_id].natural_closure
        == comparison_by_request[request_id].natural_closure
        and initial_by_request[request_id].detected_official_reference_object_ids
        == comparison_by_request[request_id].detected_official_reference_object_ids
        for request_id in initial_by_request
    )


def _validate_calibration_cohort(cohort: CohortLedger) -> None:
    if (
        cohort.cohort_id != CALIBRATION_COHORT_ID
        or len(cohort.records) != CALIBRATION_IMAGE_COUNT
    ):
        _data_error(
            "calibration cohort must be exact Sampling Calibration-12",
            "calibration_cohort",
        )


def _validate_calibration_and_validation_cohorts(
    *,
    calibration_cohort: CohortLedger,
    validation_cohort: CohortLedger,
) -> None:
    _validate_calibration_cohort(calibration_cohort)
    if (
        validation_cohort.cohort_id != VALIDATION_COHORT_ID
        or len(validation_cohort.records) != 200
    ):
        _data_error(
            "validation cohort must be exact Validation-200",
            "validation_cohort",
        )
    overlap = sorted(
        {row.image_id for row in calibration_cohort.records}
        & {row.image_id for row in validation_cohort.records}
    )
    if overlap:
        _data_error(
            "Sampling Calibration-12 overlaps Validation-200",
            "calibration_validation_overlap",
            overlapping_image_ids=overlap,
        )


def _ordered_materialized_panel(
    *,
    grouped: Mapping[
        tuple[float, CalibrationPanelKind], Sequence[CalibrationCallObservation]
    ],
    requests: Sequence[CalibrationRequest],
    panel_kind: CalibrationPanelKind,
) -> tuple[CalibrationCallObservation, ...]:
    """Rebuild one exact physical panel from canonical terminal bundles."""

    if len(requests) != CALIBRATION_INITIAL_CALL_COUNT:
        _artifact_error(
            "calibration request panel does not contain 48 calls",
            "terminal_bundle_request_count",
        )
    temperature = requests[0].temperature
    rows = tuple(grouped.get((temperature, panel_kind), ()))
    if len(rows) != CALIBRATION_INITIAL_CALL_COUNT:
        _artifact_error(
            "terminal bundle panel does not contain 48 calls",
            "terminal_bundle_panel_count",
            panel_kind=panel_kind,
            temperature=temperature,
            observed_count=len(rows),
        )
    expected_by_request = {row.request_id: row for row in requests}
    observed_by_request: dict[str, CalibrationCallObservation] = {}
    for row in rows:
        request_id = row.request.request_id
        expected = expected_by_request.get(request_id)
        if (
            request_id in observed_by_request
            or expected is None
            or row.request != expected
        ):
            _artifact_error(
                "terminal bundle request differs from the frozen panel",
                "terminal_bundle_request_identity",
                request_id=request_id,
            )
        observed_by_request[request_id] = row
    if set(observed_by_request) != set(expected_by_request):
        _artifact_error(
            "terminal bundle panel omits frozen requests",
            "terminal_bundle_request_identity",
        )
    ordered: list[CalibrationCallObservation] = []
    for batch_index in range(CALIBRATION_IMAGE_COUNT):
        canonical = tuple(requests[batch_index * 4 : (batch_index + 1) * 4])
        physical = (
            tuple(reversed(canonical)) if panel_kind == "reversed_order" else canonical
        )
        batch_fingerprints: set[str] = set()
        for execution_index, request in enumerate(physical):
            row = observed_by_request[request.request_id]
            if (
                row.physical_batch_index != batch_index
                or row.request_execution_index != execution_index
            ):
                _artifact_error(
                    "terminal bundle execution order differs from the frozen panel",
                    "terminal_bundle_execution_order",
                    request_id=request.request_id,
                )
            batch_fingerprints.add(row.decode_receipt_fingerprint)
            ordered.append(row)
        # Receipt fingerprints are request-specific; execution indexes above are
        # the exact cross-request batch-order binding used by this calibration.
        if len(batch_fingerprints) != CALIBRATION_SEEDS_PER_IMAGE:
            _artifact_error(
                "terminal bundle batch repeats a decode receipt",
                "terminal_bundle_receipt_identity",
            )
    return tuple(ordered)


def _validate_prompt_record_artifact(
    value: Mapping[str, Any], *, request: CalibrationRequest
) -> None:
    _require_exact_keys(value, _PROMPT_RECORD_ARTIFACT_FIELDS, "prompt record")
    prompt_token_ids = value["prompt_token_ids"]
    if (
        not isinstance(prompt_token_ids, list)
        or not prompt_token_ids
        or any(
            isinstance(token_id, bool) or not isinstance(token_id, int)
            for token_id in prompt_token_ids
        )
        or value["prompt_token_count"] != len(prompt_token_ids)
    ):
        _artifact_error(
            "prompt record token identifiers are invalid",
            "terminal_bundle_prompt_tokens",
        )
    if value["row_index"] != request.image_frozen_order:
        _artifact_error(
            "prompt record row index differs from the calibration image",
            "terminal_bundle_prompt_row",
        )
    for field in (
        "assistant_format",
        "example_id",
        "object_field_order",
        "object_ordering",
        "prompt_text",
        "row_id",
        "template_id",
    ):
        if not isinstance(value[field], str) or not value[field]:
            _artifact_error(
                "prompt record contains an empty identity field",
                "terminal_bundle_prompt_identity",
                field=field,
            )
    for field in ("full_prompt_fingerprint", "template_fingerprint"):
        _require_sha256(value[field], field)
    if not isinstance(value["realized_object_order"], list):
        _artifact_error(
            "prompt record realized object order is not an array",
            "terminal_bundle_prompt_order",
        )


def _validate_visual_input_receipt(
    value: Mapping[str, Any], *, request: CalibrationRequest
) -> None:
    _require_exact_keys(
        value,
        _VISUAL_INPUT_RECEIPT_FIELDS,
        "visual input materialization receipt",
    )
    if (
        value["schema_version"] != VISUAL_INPUT_MATERIALIZATION_RECEIPT_SCHEMA_VERSION
        or value["input_kind"] != "full_image"
        or value["spatial_image_encoding_sha256"] is not None
    ):
        _artifact_error(
            "calibration requires a no-resize full-image materialization receipt",
            "terminal_bundle_visual_kind",
        )
    if value["source_image_sha256"] != request.image_sha256:
        _artifact_error(
            "visual receipt source image differs from the calibration request",
            "terminal_bundle_visual_source",
        )
    for field in (
        "source_image_sha256",
        "input_rgb_sha256",
        "processor_contract_sha256",
        "receipt_sha256",
    ):
        _require_sha256(value[field], field)
    if any(
        isinstance(value[field], bool)
        or not isinstance(value[field], int)
        or value[field] <= 0
        for field in ("input_width", "input_height")
    ):
        _artifact_error(
            "visual receipt dimensions are invalid",
            "terminal_bundle_visual_dimensions",
        )
    tensors = value["executed_visual_tensors"]
    _require_exact_keys(
        tensors,
        {"image_grid_thw", "pixel_values", "receipt_sha256", "schema_version"},
        "executed visual tensor receipt",
    )
    if tensors["schema_version"] != EXECUTED_VISUAL_TENSOR_RECEIPT_SCHEMA_VERSION:
        _artifact_error(
            "executed visual tensor receipt schema is unsupported",
            "terminal_bundle_visual_tensor_schema",
        )
    for name in ("image_grid_thw", "pixel_values"):
        tensor = tensors[name]
        _require_exact_keys(
            tensor,
            {"canonical_content_sha256", "dtype", "schema_version", "shape"},
            f"{name} tensor receipt",
        )
        if (
            tensor["schema_version"] != TENSOR_CONTENT_RECEIPT_SCHEMA_VERSION
            or not isinstance(tensor["dtype"], str)
            or not tensor["dtype"]
            or not isinstance(tensor["shape"], list)
            or any(
                isinstance(size, bool) or not isinstance(size, int) or size < 0
                for size in tensor["shape"]
            )
        ):
            _artifact_error(
                "executed visual tensor metadata is invalid",
                "terminal_bundle_visual_tensor",
                field=name,
            )
        _require_sha256(tensor["canonical_content_sha256"], "canonical_content_sha256")
    _require_sha256(tensors["receipt_sha256"], "executed_visual_tensors.receipt_sha256")
    tensor_identity = dict(tensors)
    tensor_identity.pop("receipt_sha256")
    if tensors["receipt_sha256"] != sha256_payload(tensor_identity):
        _artifact_error(
            "executed visual tensor receipt digest is invalid",
            "terminal_bundle_visual_tensor_digest",
        )
    identity = dict(value)
    identity.pop("receipt_sha256")
    if value["receipt_sha256"] != sha256_payload(identity):
        _artifact_error(
            "visual input materialization receipt digest is invalid",
            "terminal_bundle_visual_digest",
        )


def _decode_receipt_runtime_contract(receipt: Any) -> dict[str, Any]:
    artifact = receipt.to_artifact_dict()
    fields = (
        "attention_implementation",
        "backend",
        "backend_mode",
        "custom_sampler_code_hash",
        "custom_sampler_identity",
        "decode_generation_policy_fingerprint",
        "effective_generation_profile_fingerprint",
        "executed_generation_arguments",
        "generation_config_fingerprint",
        "installed_runtime_identity_fingerprint",
        "model_eval_mode",
        "model_identity_fingerprint",
        "random_generator_device",
        "random_generator_kind",
        "response_family",
        "runtime_identity",
        "sampling_profile",
        "sampling_profile_fingerprint",
        "tokenizer_identity_fingerprint",
    )
    return {field: artifact[field] for field in fields}


def _canonical_calibration_trace(trace: TokenTrace) -> TokenTrace:
    return replace(
        trace,
        logprob=(
            None
            if trace.logprob is None
            else canonical_float32_logprob(
                trace.logprob,
                error_code="analysis.calibration_non_finite_logprob",
                context={"step_index": trace.step_index},
            )
        ),
    )


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(
        {str(key): _freeze_json_value(item) for key, item in value.items()}
    )


def _freeze_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json_value(item) for item in value)
    return value


def _thaw_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _thaw_json_value(item) for key, item in value.items()}


def _thaw_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _thaw_mapping(value)
    if isinstance(value, tuple):
        return [_thaw_json_value(item) for item in value]
    return value


def _require_sha256(value: Any, field: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _artifact_error(
            "identity field must be lowercase Secure Hash Algorithm 256-bit hexadecimal",
            "sha256",
            field=field,
        )


def _require_exact_keys(value: Any, expected: set[str], record_name: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        _artifact_error(
            f"{record_name} keys differ from its schema",
            "record_keys",
            missing=sorted(
                expected - set(value) if isinstance(value, Mapping) else expected
            ),
            unknown=sorted(set(value) - expected if isinstance(value, Mapping) else ()),
        )


def _data_error(message: str, suffix: str, **context: Any) -> None:
    raise DataContractError(
        message, code=f"analysis.calibration_{suffix}", context=context
    )


def _artifact_error(message: str, suffix: str, **context: Any) -> None:
    raise ArtifactContractError(
        message, code=f"analysis.calibration_{suffix}", context=context
    )
