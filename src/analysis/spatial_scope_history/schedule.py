"""Deterministic request schedule for spatial-scope and history disentanglement."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Literal

from src.analysis.spatial_scope_history.cohort_ledger import (
    AttemptLedger,
    AttemptDependencyContract,
    AttemptRecord,
    CohortLedger,
    ExecutionIdentityBundle,
    dependency_skip_failure_code,
    dependency_state_unavailable_failure_code,
    sha256_file,
    sha256_payload,
)
from src.common.errors import ArtifactContractError, DataContractError


SCHEDULE_SCHEMA_VERSION = "spatial_scope_history.schedule.v2"
REQUEST_SCHEMA_VERSION = "spatial_scope_history.request.v1"
GRID_SCHEMA_VERSION = "spatial_scope_history.grid_provenance.v1"
DECODE_SCHEMA_VERSION = "spatial_scope_history.decode_provenance.v1"
PHYSICAL_BATCH_SCHEMA_VERSION = "spatial_scope_history.physical_batch.v2"
PHYSICAL_BATCH_PLAN_SCHEMA_VERSION = "spatial_scope_history.physical_batch_plan.v2"
SEED_NAMESPACE = "coordexp-dense-enumeration-seeds-v1"
PRIMARY_ROOT_SEED = 2026071301
SECOND_ROOT_SEED = 2026071302
PRIMARY_BATCH_SIZE = 4
GRID_CELL_COUNT = 16
INDEPENDENT_EXECUTION_WAVE_PARTITION = "independent"
CUMULATIVE_EXECUTION_WAVE_PARTITION_PREFIX = "cumulative-cell-"
EMPTY_ACCEPTED_ROW_PREFIX_STATE_SHA256 = sha256_payload(
    {
        "accepted_global_coordinate_rows": [],
        "schema_version": "accepted_row_prefix_state.v1",
    }
)
PrimaryArmCode = Literal[
    "FULL_SINGLE", "FULL_BAG_K", "TILE_RESET", "MASK_RESET", "MASK_CUMULATIVE"
]


def cumulative_execution_wave_partition(cell_index: int) -> str:
    """Return the sealed execution-wave identity for one cumulative cell."""

    if (
        isinstance(cell_index, bool)
        or not isinstance(cell_index, int)
        or not 0 <= cell_index < GRID_CELL_COUNT
    ):
        _data_error(
            "cumulative execution-wave cell index is outside the canonical grid",
            code="analysis.execution_wave_cell_index",
            cell_index=cell_index,
        )
    return f"{CUMULATIVE_EXECUTION_WAVE_PARTITION_PREFIX}{cell_index:02d}"


def execution_wave_partition_for_request(request: ScheduledRequest) -> str:
    """Map one request to its immutable physical batching partition."""

    if not request.arm.cumulative_dependency:
        return INDEPENDENT_EXECUTION_WAVE_PARTITION
    if request.cell_index is None:
        _data_error(
            "cumulative request lacks a canonical cell index",
            code="analysis.execution_wave_request_cell",
            request_id=request.request_id,
        )
    return cumulative_execution_wave_partition(request.cell_index)


def _validate_execution_wave_partition(value: str) -> None:
    if value == INDEPENDENT_EXECUTION_WAVE_PARTITION:
        return
    if value in {
        cumulative_execution_wave_partition(cell_index)
        for cell_index in range(GRID_CELL_COUNT)
    }:
        return
    _artifact_error(
        "physical batch names an unknown execution-wave partition",
        code="analysis.physical_batch_execution_wave",
        execution_wave_partition=value,
    )


@dataclass(frozen=True)
class ResearchArmDefinition:
    """Complete name and operational behavior of one experimental arm."""

    arm_code: PrimaryArmCode
    full_name: str
    operational_meaning: str
    input_policy: str
    history_policy: str
    calls_per_image: int
    uses_spatial_cell: bool
    cumulative_dependency: bool

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "arm_code": self.arm_code,
            "calls_per_image": self.calls_per_image,
            "cumulative_dependency": self.cumulative_dependency,
            "full_name": self.full_name,
            "history_policy": self.history_policy,
            "input_policy": self.input_policy,
            "operational_meaning": self.operational_meaning,
            "uses_spatial_cell": self.uses_spatial_cell,
        }


PRIMARY_ARM_DEFINITIONS: tuple[ResearchArmDefinition, ...] = (
    ResearchArmDefinition(
        arm_code="FULL_SINGLE",
        full_name="Full-Image Single Rollout",
        operational_meaning=(
            "One sampled rollout over the complete image from the fresh base prompt "
            "using the independent baseline seed."
        ),
        input_policy="complete_source_canvas",
        history_policy="fresh_base_prompt",
        calls_per_image=1,
        uses_spatial_cell=False,
        cumulative_dependency=False,
    ),
    ResearchArmDefinition(
        arm_code="FULL_BAG_K",
        full_name="Full-Image K-Rollout Independent Bagging",
        operational_meaning=(
            "Sixteen independent complete-image rollouts from fresh prompts using the "
            "paired canonical-cell seed vector."
        ),
        input_policy="complete_source_canvas",
        history_policy="fresh_base_prompt_per_call",
        calls_per_image=GRID_CELL_COUNT,
        uses_spatial_cell=True,
        cumulative_dependency=False,
    ),
    ResearchArmDefinition(
        arm_code="TILE_RESET",
        full_name="Native-Scale Tile with Per-Tile Reset",
        operational_meaning=(
            "Sixteen unresized core-plus-halo tile calls, each decoded from a fresh prompt."
        ),
        input_policy="native_scale_core_plus_halo_tile",
        history_policy="fresh_base_prompt_per_tile",
        calls_per_image=GRID_CELL_COUNT,
        uses_spatial_cell=True,
        cumulative_dependency=False,
    ),
    ResearchArmDefinition(
        arm_code="MASK_RESET",
        full_name="Full-Canvas Masked Region with Per-Region Reset",
        operational_meaning=(
            "Sixteen full-size core-plus-halo masked canvases, each decoded from a fresh prompt."
        ),
        input_policy="full_canvas_core_plus_halo_mask",
        history_policy="fresh_base_prompt_per_region",
        calls_per_image=GRID_CELL_COUNT,
        uses_spatial_cell=True,
        cumulative_dependency=False,
    ),
    ResearchArmDefinition(
        arm_code="MASK_CUMULATIVE",
        full_name="Full-Canvas Masked Region with Cumulative Accepted-Row Prefix",
        operational_meaning=(
            "Sixteen full-size masked canvases decoded in canonical cell order while "
            "accepted owning global-coordinate rows accumulate inside the open assistant turn."
        ),
        input_policy="full_canvas_core_plus_halo_mask",
        history_policy="cumulative_accepted_global_row_prefix",
        calls_per_image=GRID_CELL_COUNT,
        uses_spatial_cell=True,
        cumulative_dependency=True,
    ),
)
_ARM_BY_CODE: Mapping[str, ResearchArmDefinition] = {
    definition.arm_code: definition for definition in PRIMARY_ARM_DEFINITIONS
}
PRIMARY_ARM_CODES: tuple[str, ...] = tuple(_ARM_BY_CODE)


def primary_arm_definition(arm_code: str) -> ResearchArmDefinition:
    """Return one frozen primary arm declaration or fail closed."""

    try:
        return _ARM_BY_CODE[arm_code]
    except KeyError:
        _data_error(
            "experimental arm is not in the frozen five-arm primary panel",
            code="analysis.schedule_unknown_primary_arm",
            arm_code=arm_code,
        )


@dataclass(frozen=True)
class GridProvenance:
    """Content binding to the canonical spatial specification and receipt contract."""

    canonical_spatial_spec_sha256: str
    canonical_spatial_receipt_contract_sha256: str
    schema_version: str = GRID_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != GRID_SCHEMA_VERSION:
            _data_error(
                "grid provenance has an unsupported schema version",
                code="analysis.schedule_grid_schema",
            )
        _require_sha256(
            self.canonical_spatial_spec_sha256,
            field="canonical_spatial_spec_sha256",
        )
        _require_sha256(
            self.canonical_spatial_receipt_contract_sha256,
            field="canonical_spatial_receipt_contract_sha256",
        )

    @property
    def cell_count(self) -> int:
        """The frozen primary four-by-four grid contains sixteen cells."""

        return GRID_CELL_COUNT

    @property
    def fingerprint(self) -> str:
        return sha256_payload(self.to_artifact_dict())

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "canonical_spatial_receipt_contract_sha256": (
                self.canonical_spatial_receipt_contract_sha256
            ),
            "canonical_spatial_spec_sha256": self.canonical_spatial_spec_sha256,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_artifact_dict(cls, value: Mapping[str, Any]) -> GridProvenance:
        fields = {
            "canonical_spatial_receipt_contract_sha256",
            "canonical_spatial_spec_sha256",
            "schema_version",
        }
        _require_exact_keys(value, fields=fields, record_name="grid provenance")
        return cls(**dict(value))


@dataclass(frozen=True)
class DecodeProvenance:
    """Research temperature bound to canonical backend policy and attestation receipts."""

    temperature: float
    canonical_generation_policy_sha256: str
    sampled_runtime_attestation_sha256: str
    schema_version: str = DECODE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != DECODE_SCHEMA_VERSION:
            _data_error(
                "decode provenance has an unsupported schema version",
                code="analysis.schedule_decode_schema",
            )
        if self.temperature not in {0.2, 0.4, 0.6} or not math.isfinite(
            self.temperature
        ):
            _data_error(
                "temperature must be one frozen calibration candidate",
                code="analysis.schedule_temperature",
                temperature=self.temperature,
            )
        _require_sha256(
            self.canonical_generation_policy_sha256,
            field="canonical_generation_policy_sha256",
        )
        _require_sha256(
            self.sampled_runtime_attestation_sha256,
            field="sampled_runtime_attestation_sha256",
        )

    @property
    def fingerprint(self) -> str:
        return sha256_payload(self.to_artifact_dict())

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "canonical_generation_policy_sha256": self.canonical_generation_policy_sha256,
            "sampled_runtime_attestation_sha256": self.sampled_runtime_attestation_sha256,
            "schema_version": self.schema_version,
            "temperature": self.temperature,
        }

    @classmethod
    def from_artifact_dict(cls, value: Mapping[str, Any]) -> DecodeProvenance:
        fields = {
            "canonical_generation_policy_sha256",
            "sampled_runtime_attestation_sha256",
            "schema_version",
            "temperature",
        }
        _require_exact_keys(value, fields=fields, record_name="decode provenance")
        return cls(**dict(value))


@dataclass(frozen=True)
class ScheduleIdentity:
    """All identities that must remain equal for same-run resume."""

    unit_id: str
    run_id: str
    cohort_sha256: str
    root_seed: int
    execution_identity: ExecutionIdentityBundle
    grid: GridProvenance
    decode: DecodeProvenance
    seed_namespace: str = SEED_NAMESPACE
    schema_version: str = SCHEDULE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name in ("unit_id", "run_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                _data_error(
                    "schedule identity string must be nonempty",
                    code="analysis.schedule_identity_string",
                    field=field_name,
                )
        _require_sha256(self.cohort_sha256, field="cohort_sha256")
        if self.root_seed not in {PRIMARY_ROOT_SEED, SECOND_ROOT_SEED}:
            _data_error(
                "root seed is not authorized for this research unit",
                code="analysis.schedule_root_seed",
                root_seed=self.root_seed,
            )
        if (
            self.seed_namespace != SEED_NAMESPACE
            or self.schema_version != SCHEDULE_SCHEMA_VERSION
        ):
            _data_error(
                "schedule namespace or schema differs from the frozen contract",
                code="analysis.schedule_namespace_schema",
            )

    @property
    def fingerprint(self) -> str:
        return sha256_payload(self.to_artifact_dict())

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "cohort_sha256": self.cohort_sha256,
            "decode": self.decode.to_artifact_dict(),
            "execution_identity": self.execution_identity.to_artifact_dict(),
            "grid": self.grid.to_artifact_dict(),
            "root_seed": self.root_seed,
            "run_id": self.run_id,
            "schema_version": self.schema_version,
            "seed_namespace": self.seed_namespace,
            "unit_id": self.unit_id,
        }

    @classmethod
    def from_artifact_dict(cls, value: Mapping[str, Any]) -> ScheduleIdentity:
        fields = {
            "cohort_sha256",
            "decode",
            "execution_identity",
            "grid",
            "root_seed",
            "run_id",
            "schema_version",
            "seed_namespace",
            "unit_id",
        }
        _require_exact_keys(value, fields=fields, record_name="schedule identity")
        payload = dict(value)
        payload["execution_identity"] = ExecutionIdentityBundle.from_artifact_dict(
            payload["execution_identity"]
        )
        payload["grid"] = GridProvenance.from_artifact_dict(payload["grid"])
        payload["decode"] = DecodeProvenance.from_artifact_dict(payload["decode"])
        return cls(**payload)


@dataclass(frozen=True)
class ScheduledRequest:
    """One pre-materialized scientific request, independent of batch placement."""

    request_id: str
    schedule_index: int
    image_id: int
    image_frozen_order: int
    image_sha256: str
    arm: ResearchArmDefinition
    call_label: str
    cell_index: int | None
    seed_role: str
    sampling_seed: int
    schedule_identity_sha256: str
    grid_sha256: str
    decode_sha256: str
    execution_identity_sha256: str
    predecessor_request_id: str | None
    initial_cumulative_state_sha256: str | None
    schema_version: str = REQUEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != REQUEST_SCHEMA_VERSION:
            _data_error(
                "request schema differs from the frozen contract",
                code="analysis.request_schema",
            )
        if (
            self.arm.arm_code not in _ARM_BY_CODE
            or self.arm != _ARM_BY_CODE[self.arm.arm_code]
        ):
            _data_error(
                "request arm definition differs from the frozen primary panel",
                code="analysis.request_arm_drift",
                arm_code=self.arm.arm_code,
            )
        if not isinstance(self.request_id, str) or not self.request_id.startswith(
            "spatial-scope-history-request:"
        ):
            _data_error(
                "request identity has an invalid namespace",
                code="analysis.request_id_namespace",
                request_id=self.request_id,
            )
        for field_name in (
            "schedule_index",
            "image_id",
            "image_frozen_order",
            "sampling_seed",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                _data_error(
                    "request integer field must be nonnegative",
                    code="analysis.request_integer",
                    field=field_name,
                )
        if self.sampling_seed >= 1 << 63:
            _data_error(
                "sampling seed must be an unsigned 63-bit integer",
                code="analysis.request_seed_range",
                sampling_seed=self.sampling_seed,
            )
        for field_name in (
            "image_sha256",
            "schedule_identity_sha256",
            "grid_sha256",
            "decode_sha256",
            "execution_identity_sha256",
        ):
            _require_sha256(getattr(self, field_name), field=field_name)
        if self.arm.arm_code == "FULL_SINGLE":
            if (self.cell_index, self.seed_role, self.call_label) != (
                None,
                "baseline",
                "single",
            ):
                _data_error(
                    "full-image single rollout requires baseline seed provenance",
                    code="analysis.request_baseline_provenance",
                )
        else:
            expected_label = (
                None if self.cell_index is None else f"cell-{self.cell_index:02d}"
            )
            if (
                self.cell_index is None
                or not 0 <= self.cell_index < GRID_CELL_COUNT
                or self.seed_role != "paired-cell"
                or self.call_label != expected_label
            ):
                _data_error(
                    "K-call request requires canonical paired-cell seed provenance",
                    code="analysis.request_cell_provenance",
                )
        if self.arm.cumulative_dependency:
            if self.cell_index == 0:
                if (
                    self.predecessor_request_id is not None
                    or self.initial_cumulative_state_sha256
                    != EMPTY_ACCEPTED_ROW_PREFIX_STATE_SHA256
                ):
                    _data_error(
                        "first cumulative cell requires the frozen empty initial state",
                        code="analysis.request_cumulative_initial_state",
                        request_id=self.request_id,
                    )
            elif (
                self.predecessor_request_id is None
                or self.initial_cumulative_state_sha256 is not None
            ):
                _data_error(
                    "later cumulative cell requires exactly one predecessor",
                    code="analysis.request_cumulative_predecessor",
                    request_id=self.request_id,
                )
        elif (
            self.predecessor_request_id is not None
            or self.initial_cumulative_state_sha256 is not None
        ):
            _data_error(
                "non-cumulative request cannot carry a cumulative dependency",
                code="analysis.request_unexpected_cumulative_dependency",
                request_id=self.request_id,
            )

    def identity_payload(self) -> dict[str, Any]:
        """Return identity-bearing fields, deliberately excluding execution order."""

        return {
            "arm": self.arm.to_artifact_dict(),
            "call_label": self.call_label,
            "cell_index": self.cell_index,
            "decode_sha256": self.decode_sha256,
            "execution_identity_sha256": self.execution_identity_sha256,
            "grid_sha256": self.grid_sha256,
            "image_id": self.image_id,
            "image_sha256": self.image_sha256,
            "initial_cumulative_state_sha256": self.initial_cumulative_state_sha256,
            "predecessor_request_id": self.predecessor_request_id,
            "sampling_seed": self.sampling_seed,
            "schedule_identity_sha256": self.schedule_identity_sha256,
            "schema_version": self.schema_version,
            "seed_role": self.seed_role,
        }

    def validate_request_id(self) -> None:
        expected = "spatial-scope-history-request:" + sha256_payload(
            self.identity_payload()
        )
        if self.request_id != expected:
            _data_error(
                "request identifier does not match its identity-bearing provenance",
                code="analysis.request_id_drift",
                request_id=self.request_id,
                expected_request_id=expected,
            )

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            **self.identity_payload(),
            "image_frozen_order": self.image_frozen_order,
            "request_id": self.request_id,
            "schedule_index": self.schedule_index,
        }

    @classmethod
    def from_artifact_dict(cls, value: Mapping[str, Any]) -> ScheduledRequest:
        fields = {
            "arm",
            "call_label",
            "cell_index",
            "decode_sha256",
            "execution_identity_sha256",
            "grid_sha256",
            "image_frozen_order",
            "image_id",
            "image_sha256",
            "initial_cumulative_state_sha256",
            "predecessor_request_id",
            "request_id",
            "sampling_seed",
            "schedule_identity_sha256",
            "schedule_index",
            "schema_version",
            "seed_role",
        }
        _require_exact_keys(value, fields=fields, record_name="scheduled request")
        payload = dict(value)
        arm_payload = payload.pop("arm")
        _require_exact_keys(
            arm_payload,
            fields=set(PRIMARY_ARM_DEFINITIONS[0].to_artifact_dict()),
            record_name="research arm definition",
        )
        arm_code = arm_payload["arm_code"]
        if (
            arm_code not in _ARM_BY_CODE
            or arm_payload != _ARM_BY_CODE[arm_code].to_artifact_dict()
        ):
            _artifact_error(
                "scheduled request contains an unknown or drifted arm definition",
                code="analysis.request_arm_drift",
                arm_code=arm_code,
            )
        payload["arm"] = _ARM_BY_CODE[arm_code]
        request = cls(**payload)
        request.validate_request_id()
        return request


@dataclass(frozen=True)
class PhysicalBatchDefinition:
    """One content-addressed ordered request group in the sealed physical plan."""

    batch_index: int
    request_ids: tuple[str, ...]
    execution_wave_partition: str = INDEPENDENT_EXECUTION_WAVE_PARTITION
    schema_version: str = PHYSICAL_BATCH_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PHYSICAL_BATCH_SCHEMA_VERSION:
            _artifact_error(
                "physical batch has an unsupported schema version",
                code="analysis.physical_batch_schema",
                batch_index=self.batch_index,
            )
        if (
            isinstance(self.batch_index, bool)
            or not isinstance(self.batch_index, int)
            or self.batch_index < 0
        ):
            _artifact_error(
                "physical batch index must be a nonnegative integer",
                code="analysis.physical_batch_index",
            )
        if len(self.request_ids) not in {3, 4}:
            _artifact_error(
                "physical batch cardinality is outside the attested plan",
                code="analysis.physical_batch_cardinality",
                batch_index=self.batch_index,
                cardinality=len(self.request_ids),
            )
        if len(self.request_ids) != len(set(self.request_ids)):
            _artifact_error(
                "physical batch repeats a request identity",
                code="analysis.physical_batch_duplicate_request",
                batch_index=self.batch_index,
            )
        for request_id in self.request_ids:
            if not isinstance(request_id, str) or not request_id:
                _artifact_error(
                    "physical batch request identity must be nonempty",
                    code="analysis.physical_batch_request_id",
                    batch_index=self.batch_index,
                )
        _validate_execution_wave_partition(self.execution_wave_partition)

    @property
    def cardinality(self) -> int:
        return len(self.request_ids)

    @property
    def fingerprint(self) -> str:
        return sha256_payload(self.identity_payload())

    def identity_payload(self) -> dict[str, Any]:
        return {
            "batch_index": self.batch_index,
            "execution_wave_partition": self.execution_wave_partition,
            "request_ids": list(self.request_ids),
            "schema_version": self.schema_version,
        }

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            **self.identity_payload(),
            "physical_batch_sha256": self.fingerprint,
        }

    @classmethod
    def from_artifact_dict(cls, value: Mapping[str, Any]) -> PhysicalBatchDefinition:
        fields = {
            "batch_index",
            "execution_wave_partition",
            "physical_batch_sha256",
            "request_ids",
            "schema_version",
        }
        _require_exact_keys(value, fields=fields, record_name="physical batch")
        batch = cls(
            batch_index=value["batch_index"],
            execution_wave_partition=value["execution_wave_partition"],
            request_ids=tuple(value["request_ids"]),
            schema_version=value["schema_version"],
        )
        if value["physical_batch_sha256"] != batch.fingerprint:
            _artifact_error(
                "physical batch fingerprint does not match its ordered membership",
                code="analysis.physical_batch_fingerprint",
                batch_index=batch.batch_index,
            )
        return batch


@dataclass(frozen=True)
class PhysicalBatchPlan:
    """Content-addressed physical batches sealed before any attempt executes."""

    schedule_identity_sha256: str
    primary_batch_size: int
    batches: tuple[PhysicalBatchDefinition, ...]
    schema_version: str = PHYSICAL_BATCH_PLAN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_sha256(
            self.schedule_identity_sha256,
            field="schedule_identity_sha256",
        )
        if self.schema_version != PHYSICAL_BATCH_PLAN_SCHEMA_VERSION:
            _artifact_error(
                "physical batch plan has an unsupported schema version",
                code="analysis.physical_batch_plan_schema",
            )
        if self.primary_batch_size != PRIMARY_BATCH_SIZE:
            _artifact_error(
                "physical batch plan differs from the frozen primary batch size",
                code="analysis.physical_batch_plan_size",
                primary_batch_size=self.primary_batch_size,
            )
        if not self.batches:
            _artifact_error(
                "physical batch plan cannot be empty",
                code="analysis.physical_batch_plan_empty",
            )
        if [batch.batch_index for batch in self.batches] != list(
            range(len(self.batches))
        ):
            _artifact_error(
                "physical batch plan indexes must be contiguous",
                code="analysis.physical_batch_plan_order",
            )
        observed_partition_order: list[str] = []
        batches_by_partition: dict[str, list[PhysicalBatchDefinition]] = {}
        for batch in self.batches:
            if batch.execution_wave_partition not in batches_by_partition:
                observed_partition_order.append(batch.execution_wave_partition)
                batches_by_partition[batch.execution_wave_partition] = []
            elif (
                observed_partition_order[-1] != batch.execution_wave_partition
            ):
                _artifact_error(
                    "execution-wave partition must occupy one contiguous batch range",
                    code="analysis.physical_batch_plan_partition_contiguity",
                    execution_wave_partition=batch.execution_wave_partition,
                )
            batches_by_partition[batch.execution_wave_partition].append(batch)
        expected_partition_order = [
            INDEPENDENT_EXECUTION_WAVE_PARTITION,
            *(
                cumulative_execution_wave_partition(cell_index)
                for cell_index in range(GRID_CELL_COUNT)
            ),
        ]
        if observed_partition_order != expected_partition_order[
            : len(observed_partition_order)
        ]:
            _artifact_error(
                "execution-wave partitions differ from canonical dependency order",
                code="analysis.physical_batch_plan_partition_order",
                observed_partition_order=observed_partition_order,
            )
        for partition, partition_batches in batches_by_partition.items():
            if any(
                batch.cardinality != PRIMARY_BATCH_SIZE
                for batch in partition_batches[:-1]
            ):
                _artifact_error(
                    "only the final batch of one execution-wave partition may be a natural tail",
                    code="analysis.physical_batch_plan_partition_nonfinal_tail",
                    execution_wave_partition=partition,
                )
        request_ids = [
            request_id for batch in self.batches for request_id in batch.request_ids
        ]
        if len(request_ids) != len(set(request_ids)):
            _artifact_error(
                "physical batch plan assigns one request more than once",
                code="analysis.physical_batch_plan_duplicate_request",
            )

    @property
    def fingerprint(self) -> str:
        return sha256_payload(self.identity_payload())

    def identity_payload(self) -> dict[str, Any]:
        return {
            "batches": [batch.to_artifact_dict() for batch in self.batches],
            "primary_batch_size": self.primary_batch_size,
            "schedule_identity_sha256": self.schedule_identity_sha256,
            "schema_version": self.schema_version,
        }

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            **self.identity_payload(),
            "physical_batch_plan_sha256": self.fingerprint,
        }

    @classmethod
    def from_artifact_dict(cls, value: Mapping[str, Any]) -> PhysicalBatchPlan:
        fields = {
            "batches",
            "physical_batch_plan_sha256",
            "primary_batch_size",
            "schedule_identity_sha256",
            "schema_version",
        }
        _require_exact_keys(value, fields=fields, record_name="physical batch plan")
        plan = cls(
            schedule_identity_sha256=value["schedule_identity_sha256"],
            primary_batch_size=value["primary_batch_size"],
            batches=tuple(
                PhysicalBatchDefinition.from_artifact_dict(batch)
                for batch in value["batches"]
            ),
            schema_version=value["schema_version"],
        )
        if value["physical_batch_plan_sha256"] != plan.fingerprint:
            _artifact_error(
                "physical batch plan fingerprint does not match its ordered batches",
                code="analysis.physical_batch_plan_fingerprint",
            )
        return plan

    @classmethod
    def build(
        cls,
        *,
        schedule_identity_sha256: str,
        request_ids: Sequence[str],
        execution_wave_partitions: Sequence[str] | None = None,
    ) -> PhysicalBatchPlan:
        partitions = (
            tuple(INDEPENDENT_EXECUTION_WAVE_PARTITION for _ in request_ids)
            if execution_wave_partitions is None
            else tuple(execution_wave_partitions)
        )
        if len(partitions) != len(request_ids):
            _artifact_error(
                "request identifiers and execution-wave partitions differ in length",
                code="analysis.physical_batch_plan_partition_length",
            )
        batches: list[PhysicalBatchDefinition] = []
        partition_start = 0
        while partition_start < len(request_ids):
            partition = partitions[partition_start]
            _validate_execution_wave_partition(partition)
            partition_end = partition_start + 1
            while (
                partition_end < len(request_ids)
                and partitions[partition_end] == partition
            ):
                partition_end += 1
            for start in range(partition_start, partition_end, PRIMARY_BATCH_SIZE):
                batches.append(
                    PhysicalBatchDefinition(
                        batch_index=len(batches),
                        request_ids=tuple(
                            request_ids[
                                start : min(start + PRIMARY_BATCH_SIZE, partition_end)
                            ]
                        ),
                        execution_wave_partition=partition,
                    )
                )
            partition_start = partition_end
        return cls(
            schedule_identity_sha256=schedule_identity_sha256,
            primary_batch_size=PRIMARY_BATCH_SIZE,
            batches=tuple(batches),
        )


@dataclass(frozen=True)
class RequestBatch:
    """One materialized execution batch preserving sealed physical membership."""

    batch_index: int
    requests: tuple[ScheduledRequest, ...]
    physical_batch_sha256: str
    physical_batch_plan_sha256: str
    execution_wave_partition: str = INDEPENDENT_EXECUTION_WAVE_PARTITION

    def __post_init__(self) -> None:
        expected_physical_batch_sha256 = PhysicalBatchDefinition(
            batch_index=self.batch_index,
            request_ids=tuple(request.request_id for request in self.requests),
            execution_wave_partition=self.execution_wave_partition,
        ).fingerprint
        if self.physical_batch_sha256 != expected_physical_batch_sha256:
            _artifact_error(
                "materialized batch membership differs from its sealed fingerprint",
                code="analysis.request_batch_fingerprint",
                batch_index=self.batch_index,
            )
        _require_sha256(self.physical_batch_sha256, field="physical_batch_sha256")
        _require_sha256(
            self.physical_batch_plan_sha256,
            field="physical_batch_plan_sha256",
        )
        _validate_execution_wave_partition(self.execution_wave_partition)

    @property
    def cardinality(self) -> int:
        return len(self.requests)

    @property
    def request_ids(self) -> tuple[str, ...]:
        return tuple(request.request_id for request in self.requests)


@dataclass(frozen=True)
class BlockedDependency:
    """One request that cannot execute because its cumulative chain is broken."""

    request_id: str
    predecessor_request_id: str
    blocking_request_id: str
    reason: str
    required_terminal_skip_failure_code: str


@dataclass(frozen=True)
class DeferredDependency:
    """One request waiting for an unattempted predecessor in a later wave."""

    request_id: str
    predecessor_request_id: str
    reason: str = "predecessor_not_attempted"


@dataclass(frozen=True)
class ResumePlan:
    """Whole sealed batches currently runnable without changing membership."""

    physical_batch_plan_sha256: str
    runnable_batches: tuple[RequestBatch, ...]
    held_batch: RequestBatch | None
    completed_physical_batch_sha256s: tuple[str, ...]
    blocked_dependencies: tuple[BlockedDependency, ...]
    deferred_dependencies: tuple[DeferredDependency, ...]
    already_attempted_request_ids: tuple[str, ...]

    @property
    def runnable_request_ids(self) -> tuple[str, ...]:
        return tuple(
            request.request_id
            for batch in self.runnable_batches
            for request in batch.requests
        )


@dataclass(frozen=True)
class ResearchSchedule:
    """Immutable primary request universe and canonical execution order."""

    identity: ScheduleIdentity
    requests: tuple[ScheduledRequest, ...]
    physical_batch_plan: PhysicalBatchPlan
    primary_batch_size: int = PRIMARY_BATCH_SIZE

    def __post_init__(self) -> None:
        if self.primary_batch_size != PRIMARY_BATCH_SIZE:
            _data_error(
                "research schedule requires the frozen primary batch size",
                code="analysis.schedule_batch_size",
                primary_batch_size=self.primary_batch_size,
            )
        expected_indexes = list(range(len(self.requests)))
        observed_indexes = [request.schedule_index for request in self.requests]
        if observed_indexes != expected_indexes:
            _data_error(
                "scheduled requests must follow contiguous canonical order",
                code="analysis.schedule_order",
            )
        request_ids = [request.request_id for request in self.requests]
        if len(request_ids) != len(set(request_ids)):
            _data_error(
                "research schedule contains duplicate request identities",
                code="analysis.schedule_duplicate_request",
            )
        for request in self.requests:
            request.validate_request_id()
            if request.schedule_identity_sha256 != self.identity.fingerprint:
                _data_error(
                    "request contains a different schedule identity",
                    code="analysis.request_schedule_identity_drift",
                    request_id=request.request_id,
                )
        if (
            self.physical_batch_plan.schedule_identity_sha256
            != self.identity.fingerprint
        ):
            _data_error(
                "physical batch plan contains a different schedule identity",
                code="analysis.physical_batch_plan_schedule_identity",
            )
        planned_request_ids = tuple(
            request_id
            for batch in self.physical_batch_plan.batches
            for request_id in batch.request_ids
        )
        if planned_request_ids != tuple(request_ids):
            _data_error(
                "physical batch plan membership differs from canonical request order",
                code="analysis.physical_batch_plan_membership",
            )
        request_by_id = {request.request_id: request for request in self.requests}
        for batch in self.physical_batch_plan.batches:
            expected_partitions = {
                execution_wave_partition_for_request(request_by_id[request_id])
                for request_id in batch.request_ids
            }
            if expected_partitions != {batch.execution_wave_partition}:
                _data_error(
                    "physical batch membership differs from its execution-wave partition",
                    code="analysis.physical_batch_request_wave_consistency",
                    batch_index=batch.batch_index,
                    execution_wave_partition=batch.execution_wave_partition,
                    expected_execution_wave_partitions=sorted(expected_partitions),
                )
        self._validate_primary_matrix()

    def _validate_primary_matrix(self) -> None:
        baseline = [
            request
            for request in self.requests
            if request.arm.arm_code == "FULL_SINGLE"
        ]
        if not baseline:
            _data_error(
                "primary schedule must contain the full-image single-rollout arm",
                code="analysis.schedule_missing_baseline",
            )
        image_rows = [
            (request.image_frozen_order, request.image_id, request.image_sha256)
            for request in baseline
        ]
        if [row[0] for row in image_rows] != list(range(len(image_rows))):
            _data_error(
                "baseline requests must establish contiguous frozen image order",
                code="analysis.schedule_image_order",
            )
        if len({row[1] for row in image_rows}) != len(image_rows):
            _data_error(
                "baseline requests contain duplicate image identifiers",
                code="analysis.schedule_duplicate_image",
            )
        expected: list[tuple[str, int | None, int, int, str]] = []
        for arm in PRIMARY_ARM_DEFINITIONS:
            cell_indexes: Sequence[int | None] = (
                tuple(range(GRID_CELL_COUNT)) if arm.uses_spatial_cell else (None,)
            )
            for cell_index in cell_indexes:
                expected.extend(
                    (arm.arm_code, cell_index, image_order, image_id, image_sha256)
                    for image_order, image_id, image_sha256 in image_rows
                )
        observed = [
            (
                request.arm.arm_code,
                request.cell_index,
                request.image_frozen_order,
                request.image_id,
                request.image_sha256,
            )
            for request in self.requests
        ]
        if observed != expected:
            _data_error(
                "request order or membership differs from the frozen five-arm primary matrix",
                code="analysis.schedule_primary_matrix",
                observed_request_count=len(observed),
                expected_request_count=len(expected),
            )
        cumulative_by_image: dict[int, list[ScheduledRequest]] = {}
        for request in self.requests:
            if request.arm.arm_code == "MASK_CUMULATIVE":
                cumulative_by_image.setdefault(request.image_id, []).append(request)
        for image_id, chain in cumulative_by_image.items():
            previous_request_id: str | None = None
            for cell_index, request in enumerate(chain):
                if request.cell_index != cell_index:
                    _data_error(
                        "cumulative chain does not follow contiguous canonical cells",
                        code="analysis.schedule_cumulative_cell_order",
                        image_id=image_id,
                    )
                expected_initial_state = (
                    EMPTY_ACCEPTED_ROW_PREFIX_STATE_SHA256 if cell_index == 0 else None
                )
                if (
                    request.predecessor_request_id != previous_request_id
                    or request.initial_cumulative_state_sha256 != expected_initial_state
                ):
                    _data_error(
                        "cumulative predecessor chain drifted from canonical image-local order",
                        code="analysis.schedule_cumulative_dependency_chain",
                        image_id=image_id,
                        cell_index=cell_index,
                    )
                previous_request_id = request.request_id
        expected_partitions = (
            INDEPENDENT_EXECUTION_WAVE_PARTITION,
            *(
                cumulative_execution_wave_partition(cell_index)
                for cell_index in range(GRID_CELL_COUNT)
            ),
        )
        observed_partitions = tuple(
            dict.fromkeys(
                batch.execution_wave_partition
                for batch in self.physical_batch_plan.batches
            )
        )
        if observed_partitions != expected_partitions:
            _data_error(
                "primary schedule must seal the independent and sixteen cumulative execution-wave partitions",
                code="analysis.schedule_execution_wave_partitions",
                observed_execution_wave_partitions=list(observed_partitions),
            )

    @property
    def fingerprint(self) -> str:
        return sha256_payload(self.to_artifact_dict())

    @property
    def request_ids(self) -> tuple[str, ...]:
        return tuple(request.request_id for request in self.requests)

    @property
    def attempt_dependencies(self) -> tuple[AttemptDependencyContract, ...]:
        physical_batch_plan_sha256 = self.physical_batch_plan.fingerprint
        physical_batch_by_request_id = {
            request_id: batch
            for batch in self.physical_batch_plan.batches
            for request_id in batch.request_ids
        }
        return tuple(
            AttemptDependencyContract(
                request_id=request.request_id,
                physical_batch_plan_sha256=physical_batch_plan_sha256,
                physical_batch_sha256=physical_batch_by_request_id[
                    request.request_id
                ].fingerprint,
                physical_batch_index=physical_batch_by_request_id[
                    request.request_id
                ].batch_index,
                predecessor_request_id=request.predecessor_request_id,
                initial_cumulative_state_sha256=(
                    request.initial_cumulative_state_sha256
                ),
            )
            for request in self.requests
        )

    def batches(self) -> tuple[RequestBatch, ...]:
        requests_by_id = {request.request_id: request for request in self.requests}
        physical_batch_plan_sha256 = self.physical_batch_plan.fingerprint
        return tuple(
            RequestBatch(
                batch_index=batch.batch_index,
                requests=tuple(
                    requests_by_id[request_id] for request_id in batch.request_ids
                ),
                physical_batch_sha256=batch.fingerprint,
                physical_batch_plan_sha256=physical_batch_plan_sha256,
                execution_wave_partition=batch.execution_wave_partition,
            )
            for batch in self.physical_batch_plan.batches
        )

    def resume_batches(
        self,
        attempt_ledger: AttemptLedger,
        *,
        attested_batch_cardinalities: frozenset[int] = frozenset({3, 4}),
    ) -> ResumePlan:
        """Return only whole sealed batches on the current dependency-safe frontier."""

        if (
            attempt_ledger.run_id != self.identity.run_id
            or attempt_ledger.schedule_sha256 != self.fingerprint
            or attempt_ledger.execution_identity != self.identity.execution_identity
        ):
            _artifact_error(
                "resume identities differ from the materialized schedule",
                code="analysis.resume_identity_drift",
            )
        attempt_ledger.validate_dependencies(
            self.attempt_dependencies,
            verify_cumulative_state_artifacts=False,
        )
        attempts = attempt_ledger.records_by_request_id
        requests_by_id = {request.request_id: request for request in self.requests}
        physical_batches = self.batches()
        for batch in physical_batches:
            if batch.cardinality not in attested_batch_cardinalities:
                _artifact_error(
                    "sealed physical batch cardinality was not attested",
                    code="analysis.resume_unattested_batch_cardinality",
                    physical_batch_sha256=batch.physical_batch_sha256,
                    batch_index=batch.batch_index,
                    batch_cardinality=batch.cardinality,
                )

        completed_physical_batch_sha256s: list[str] = []
        for batch in physical_batches:
            terminal_request_ids = tuple(
                request_id for request_id in batch.request_ids if request_id in attempts
            )
            if terminal_request_ids and len(terminal_request_ids) != batch.cardinality:
                _artifact_error(
                    "same-run resume found a partially terminal physical batch; seal a continuation plan under a new run identity",
                    code="analysis.resume_partial_physical_batch",
                    physical_batch_plan_sha256=(self.physical_batch_plan.fingerprint),
                    physical_batch_sha256=batch.physical_batch_sha256,
                    batch_index=batch.batch_index,
                    terminal_request_ids=list(terminal_request_ids),
                    pending_request_ids=sorted(
                        set(batch.request_ids) - set(terminal_request_ids)
                    ),
                )
            if len(terminal_request_ids) == batch.cardinality:
                completed_physical_batch_sha256s.append(batch.physical_batch_sha256)

        blocked: list[BlockedDependency] = []
        deferred: list[DeferredDependency] = []
        classification_cache: dict[str, tuple[str, str | None, str | None]] = {}

        def classify(request: ScheduledRequest) -> tuple[str, str | None, str | None]:
            cached = classification_cache.get(request.request_id)
            if cached is not None:
                return cached
            predecessor_id = request.predecessor_request_id
            if predecessor_id is None:
                result = ("runnable", None, None)
            elif predecessor_id in attempts:
                predecessor = attempts[predecessor_id]
                if predecessor.attempt_status != "completed":
                    result = (
                        "blocked",
                        predecessor_id,
                        dependency_skip_failure_code(
                            predecessor_request_id=predecessor_id,
                            predecessor_status=predecessor.attempt_status,
                        ),
                    )
                elif not _cumulative_state_artifact_available(predecessor):
                    result = (
                        "blocked",
                        predecessor_id,
                        dependency_state_unavailable_failure_code(
                            predecessor_request_id=predecessor_id
                        ),
                    )
                else:
                    result = ("runnable", None, None)
            else:
                predecessor_request = requests_by_id[predecessor_id]
                predecessor_class, blocking_id, _ = classify(predecessor_request)
                if predecessor_class == "blocked":
                    result = (
                        "blocked",
                        blocking_id,
                        dependency_skip_failure_code(
                            predecessor_request_id=predecessor_id,
                            predecessor_status="skipped",
                        ),
                    )
                else:
                    result = ("deferred", None, None)
            classification_cache[request.request_id] = result
            return result

        disposition_by_request_id: dict[str, str] = {}
        for request in self.requests:
            if request.request_id in attempts:
                continue
            disposition, blocking_id, failure_code = classify(request)
            disposition_by_request_id[request.request_id] = disposition
            if disposition == "blocked":
                assert request.predecessor_request_id is not None
                assert blocking_id is not None
                assert failure_code is not None
                blocked.append(
                    BlockedDependency(
                        request_id=request.request_id,
                        predecessor_request_id=request.predecessor_request_id,
                        blocking_request_id=blocking_id,
                        reason=(
                            "predecessor_state_unavailable"
                            if "state_unavailable" in failure_code
                            else "predecessor_chain_not_completed"
                        ),
                        required_terminal_skip_failure_code=failure_code,
                    )
                )
            elif disposition == "deferred":
                assert request.predecessor_request_id is not None
                deferred.append(
                    DeferredDependency(
                        request_id=request.request_id,
                        predecessor_request_id=request.predecessor_request_id,
                    )
                )

        if blocked:
            blocked_ids = {item.request_id for item in blocked}
            blocked_batches = [
                batch
                for batch in physical_batches
                if blocked_ids.intersection(batch.request_ids)
            ]
            _artifact_error(
                "same-run resume contains a broken cumulative chain; seal an explicit continuation plan under a new run identity",
                code="analysis.resume_requires_continuation_plan",
                physical_batch_plan_sha256=self.physical_batch_plan.fingerprint,
                blocked_request_ids=sorted(blocked_ids),
                blocked_physical_batch_sha256s=[
                    batch.physical_batch_sha256 for batch in blocked_batches
                ],
                blocking_request_ids=sorted(
                    {item.blocking_request_id for item in blocked}
                ),
                required_terminal_skip_failure_codes=sorted(
                    {item.required_terminal_skip_failure_code for item in blocked}
                ),
            )

        runnable_batches: list[RequestBatch] = []
        held_batch: RequestBatch | None = None
        completed_batch_ids = set(completed_physical_batch_sha256s)
        for batch in physical_batches:
            if batch.physical_batch_sha256 in completed_batch_ids:
                continue
            batch_dispositions = {
                disposition_by_request_id[request_id]
                for request_id in batch.request_ids
            }
            if batch_dispositions == {"runnable"}:
                runnable_batches.append(batch)
                continue
            held_batch = batch
            break

        return ResumePlan(
            physical_batch_plan_sha256=self.physical_batch_plan.fingerprint,
            runnable_batches=tuple(runnable_batches),
            held_batch=held_batch,
            completed_physical_batch_sha256s=tuple(completed_physical_batch_sha256s),
            blocked_dependencies=tuple(blocked),
            deferred_dependencies=tuple(deferred),
            already_attempted_request_ids=tuple(
                request.request_id
                for request in self.requests
                if request.request_id in attempts
            ),
        )

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "identity": self.identity.to_artifact_dict(),
            "physical_batch_plan": self.physical_batch_plan.to_artifact_dict(),
            "primary_batch_size": self.primary_batch_size,
            "requests": [request.to_artifact_dict() for request in self.requests],
        }

    @classmethod
    def from_artifact_dict(cls, value: Mapping[str, Any]) -> ResearchSchedule:
        fields = {
            "identity",
            "physical_batch_plan",
            "primary_batch_size",
            "requests",
        }
        _require_exact_keys(value, fields=fields, record_name="research schedule")
        return cls(
            identity=ScheduleIdentity.from_artifact_dict(value["identity"]),
            physical_batch_plan=PhysicalBatchPlan.from_artifact_dict(
                value["physical_batch_plan"]
            ),
            primary_batch_size=value["primary_batch_size"],
            requests=tuple(
                ScheduledRequest.from_artifact_dict(request)
                for request in value["requests"]
            ),
        )

    @classmethod
    def build_primary(
        cls,
        *,
        unit_id: str,
        run_id: str,
        cohort: CohortLedger,
        root_seed: int,
        decode: DecodeProvenance,
        execution_identity: ExecutionIdentityBundle,
        grid: GridProvenance,
    ) -> ResearchSchedule:
        resolved_grid = grid
        identity = ScheduleIdentity(
            unit_id=unit_id,
            run_id=run_id,
            cohort_sha256=cohort.fingerprint,
            root_seed=root_seed,
            execution_identity=execution_identity,
            grid=resolved_grid,
            decode=decode,
        )
        requests: list[ScheduledRequest] = []
        cumulative_predecessor_by_image: dict[int, str] = {}
        for arm in PRIMARY_ARM_DEFINITIONS:
            cell_indexes: Sequence[int | None] = (
                tuple(range(resolved_grid.cell_count))
                if arm.uses_spatial_cell
                else (None,)
            )
            for cell_index in cell_indexes:
                for image in cohort.records:
                    call_label = (
                        "single" if cell_index is None else f"cell-{cell_index:02d}"
                    )
                    seed_role = "baseline" if cell_index is None else "paired-cell"
                    sampling_seed = derive_sampling_seed(
                        root_seed=root_seed,
                        role=seed_role,
                        image_id=image.image_id,
                        cell_or_call_label=call_label,
                    )
                    predecessor_request_id = None
                    initial_cumulative_state_sha256 = None
                    if arm.cumulative_dependency:
                        predecessor_request_id = cumulative_predecessor_by_image.get(
                            image.image_id
                        )
                        if cell_index == 0:
                            initial_cumulative_state_sha256 = (
                                EMPTY_ACCEPTED_ROW_PREFIX_STATE_SHA256
                            )
                    request = _build_request(
                        schedule_index=len(requests),
                        image_id=image.image_id,
                        image_frozen_order=image.frozen_order,
                        image_sha256=image.image_sha256,
                        arm=arm,
                        call_label=call_label,
                        cell_index=cell_index,
                        seed_role=seed_role,
                        sampling_seed=sampling_seed,
                        schedule_identity=identity,
                        predecessor_request_id=predecessor_request_id,
                        initial_cumulative_state_sha256=(
                            initial_cumulative_state_sha256
                        ),
                    )
                    requests.append(request)
                    if arm.cumulative_dependency:
                        cumulative_predecessor_by_image[image.image_id] = (
                            request.request_id
                        )
        request_tuple = tuple(requests)
        physical_batch_plan = PhysicalBatchPlan.build(
            schedule_identity_sha256=identity.fingerprint,
            request_ids=tuple(request.request_id for request in request_tuple),
            execution_wave_partitions=tuple(
                execution_wave_partition_for_request(request)
                for request in request_tuple
            ),
        )
        return cls(
            identity=identity,
            requests=request_tuple,
            physical_batch_plan=physical_batch_plan,
        )


def derive_sampling_seed(
    *, root_seed: int, role: str, image_id: int, cell_or_call_label: str
) -> int:
    """Derive the frozen unsigned 63-bit per-request sampling seed."""

    if root_seed not in {PRIMARY_ROOT_SEED, SECOND_ROOT_SEED}:
        _data_error(
            "root seed is not authorized for this research unit",
            code="analysis.schedule_root_seed",
            root_seed=root_seed,
        )
    if role not in {
        "baseline",
        "paired-cell",
        "temperature-calibration",
        "image-bootstrap",
    }:
        _data_error(
            "sampling seed role is unknown",
            code="analysis.seed_role",
            role=role,
        )
    if isinstance(image_id, bool) or not isinstance(image_id, int) or image_id < 0:
        _data_error(
            "image identifier must be nonnegative", code="analysis.seed_image_id"
        )
    if not isinstance(cell_or_call_label, str) or not cell_or_call_label:
        _data_error("seed call label must be nonempty", code="analysis.seed_call_label")
    domain = (
        f"{SEED_NAMESPACE}\0{root_seed}\0{role}\0{image_id}\0{cell_or_call_label}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(domain).digest()[:8], "big") & ((1 << 63) - 1)


def _build_request(
    *,
    schedule_index: int,
    image_id: int,
    image_frozen_order: int,
    image_sha256: str,
    arm: ResearchArmDefinition,
    call_label: str,
    cell_index: int | None,
    seed_role: str,
    sampling_seed: int,
    schedule_identity: ScheduleIdentity,
    predecessor_request_id: str | None,
    initial_cumulative_state_sha256: str | None,
) -> ScheduledRequest:
    provisional = ScheduledRequest(
        request_id="spatial-scope-history-request:" + "0" * 64,
        schedule_index=schedule_index,
        image_id=image_id,
        image_frozen_order=image_frozen_order,
        image_sha256=image_sha256,
        arm=arm,
        call_label=call_label,
        cell_index=cell_index,
        seed_role=seed_role,
        sampling_seed=sampling_seed,
        schedule_identity_sha256=schedule_identity.fingerprint,
        grid_sha256=schedule_identity.grid.fingerprint,
        decode_sha256=schedule_identity.decode.fingerprint,
        execution_identity_sha256=schedule_identity.execution_identity.fingerprint,
        predecessor_request_id=predecessor_request_id,
        initial_cumulative_state_sha256=initial_cumulative_state_sha256,
    )
    return ScheduledRequest(
        **{
            **provisional.__dict__,
            "request_id": "spatial-scope-history-request:"
            + sha256_payload(provisional.identity_payload()),
        }
    )


def _cumulative_state_artifact_available(record: AttemptRecord) -> bool:
    if (
        record.produced_cumulative_state_sha256 is None
        or record.produced_cumulative_state_artifact_path is None
    ):
        return False
    path = Path(record.produced_cumulative_state_artifact_path)
    return (
        path.is_file() and sha256_file(path) == record.produced_cumulative_state_sha256
    )


def _require_exact_keys(
    value: Mapping[str, Any], *, fields: set[str], record_name: str
) -> None:
    if not isinstance(value, Mapping):
        _artifact_error(
            f"{record_name} must be a mapping",
            code="analysis.schedule_record_type",
        )
    observed = set(value)
    if observed != fields:
        _artifact_error(
            f"{record_name} keys do not match its schema",
            code="analysis.schedule_record_keys",
            record_name=record_name,
            missing_keys=sorted(fields - observed),
            unknown_keys=sorted(observed - fields),
        )


def _require_sha256(value: Any, *, field: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _artifact_error(
            "identity field must be a lowercase SHA-256 hexadecimal digest",
            code="analysis.schedule_sha256",
            field=field,
        )


def _data_error(message: str, *, code: str, **context: Any) -> None:
    raise DataContractError(message, code=code, context=context)


def _artifact_error(message: str, *, code: str, **context: Any) -> None:
    raise ArtifactContractError(message, code=code, context=context)
