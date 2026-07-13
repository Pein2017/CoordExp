"""Exact object matching and recomputable metrics for spatial-scope research.

This module is intentionally pure CPU code.  It stores sufficient statistics
instead of evaluator-shaped summaries so that rescue, retention, merge, safety,
and budget views can be recomputed without reading model outputs again.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
import heapq
import json
import math
from pathlib import Path
import random
from typing import Any, Literal

from src.analysis.spatial_scope_history.merge import (
    ImageMergeResult,
    NormalizedCallPrediction,
    PixelBox,
    StrictDuplicateComponent,
)
from src.analysis.spatial_scope_history.execution_evidence import (
    TerminalCallOutputBundle,
)
from src.analysis.spatial_scope_history.cohort_ledger import (
    AttemptLedger,
    AttemptStatus,
    CohortLedger,
    canonical_json_text,
    sha256_file,
    sha256_payload,
)
from src.analysis.spatial_scope_history.schedule import (
    GRID_CELL_COUNT,
    PRIMARY_ARM_DEFINITIONS,
    PRIMARY_ROOT_SEED,
    ResearchSchedule,
    ScheduledRequest,
    derive_sampling_seed,
)
from src.analysis.spatial_scope_history.spatial import SpatialGrid, SpatialGridSpec
from src.common.errors import DataContractError
from src.eval.detection_categories import (
    COCO_80_CATEGORY_NAMESPACE_SHA256,
    COCO_80_EVALUATOR_LOCAL_CATEGORY_ID_BY_NAME,
    COCO_80_OFFICIAL_CATEGORY_ID_BY_NAME,
    normalize_coco_category_name,
)
from src.vis.matching import iou_xyxy


PRIMARY_MATCH_INTERSECTION_OVER_UNION_THRESHOLD = 0.50
LOCALIZATION_SENSITIVITY_INTERSECTION_OVER_UNION_THRESHOLD = 0.75
IGNORE_INTERSECTION_OVER_PREDICTION_AREA_THRESHOLD = 0.50
EXACT_MATCH_SOLVER = "custom-successive-shortest-augmenting-path-exact-binary64-v1"
EXACT_MATCH_OBJECTIVE_TOLERANCE = 0.0
DEFAULT_BOOTSTRAP_REPLICATES = 10_000
DEFAULT_MINIMUM_APPLICABLE_BOOTSTRAP_REPLICATES = 9_500

ReferenceLedgerScope = Literal["official_annotation", "audit_augmented"]
ReferenceState = Literal["accepted", "ambiguous", "partial", "crowd", "out_of_scope"]
MetricStatus = Literal["ok", "not_applicable"]
BootstrapEstimator = Literal["rate", "paired_rate_difference", "count_ratio"]
BootstrapEvidenceStatus = Literal["metric_bearing", "synthetic_only"]
RowParseStatus = Literal["parsed", "malformed"]
RowValidityStatus = Literal["valid", "invalid", "not_applicable"]
RowOwnershipStatus = Literal["owned", "non_owning", "not_applicable"]


@dataclass(frozen=True)
class MetricReadinessIdentity:
    """Sealed pre-output identities required before metric admission."""

    readiness_seal_sha256: str
    annotation_derived_cohort_seal_sha256: str
    readiness_root: str
    cohort_artifact_name: str
    cohort_artifact_sha256: str
    cohort_sha256: str
    reference_ledger_sha256: str
    spatial_grid_spec_sha256: str
    category_namespace_sha256: str = COCO_80_CATEGORY_NAMESPACE_SHA256

    @classmethod
    def from_verified_artifacts(
        cls,
        *,
        readiness_root: str | Path,
        spatial_grid_spec: SpatialGridSpec,
        cohort_artifact_name: str,
        cohort_artifact_sha256: str,
    ) -> MetricReadinessIdentity:
        root = Path(readiness_root).resolve()
        derived = _derive_readiness_artifacts(
            root,
            cohort_artifact_name=cohort_artifact_name,
            cohort_artifact_sha256=cohort_artifact_sha256,
        )
        return cls(
            readiness_seal_sha256=derived["readiness_seal_sha256"],
            annotation_derived_cohort_seal_sha256=derived[
                "annotation_derived_cohort_seal_sha256"
            ],
            readiness_root=str(root),
            cohort_artifact_name=derived["cohort_artifact_name"],
            cohort_artifact_sha256=derived["cohort_artifact_sha256"],
            cohort_sha256=derived["cohort_sha256"],
            reference_ledger_sha256=derived["reference_ledger_sha256"],
            spatial_grid_spec_sha256=spatial_grid_spec.fingerprint,
            category_namespace_sha256=derived["category_namespace_sha256"],
        )

    def __post_init__(self) -> None:
        for field_name in (
            "readiness_seal_sha256",
            "annotation_derived_cohort_seal_sha256",
            "cohort_artifact_sha256",
            "cohort_sha256",
            "reference_ledger_sha256",
            "spatial_grid_spec_sha256",
            "category_namespace_sha256",
        ):
            _require_sha256(getattr(self, field_name), field=field_name)
        _require_nonempty(self.readiness_root, field="readiness_root")
        if (
            not isinstance(self.cohort_artifact_name, str)
            or not self.cohort_artifact_name
            or Path(self.cohort_artifact_name).name != self.cohort_artifact_name
        ):
            _fail(
                "metric cohort artifact name must be root-local",
                "analysis.metrics_readiness_cohort_artifact_name",
                cohort_artifact_name=self.cohort_artifact_name,
            )
        if self.category_namespace_sha256 != COCO_80_CATEGORY_NAMESPACE_SHA256:
            _fail(
                "readiness identity uses a noncanonical category namespace",
                "analysis.metrics_admission_category_namespace",
            )
        self.verify_current_artifacts()

    def verify_current_artifacts(self) -> None:
        root = Path(self.readiness_root)
        derived = _derive_readiness_artifacts(
            root,
            cohort_artifact_name=self.cohort_artifact_name,
            cohort_artifact_sha256=self.cohort_artifact_sha256,
        )
        if any(
            derived[field] != getattr(self, field)
            for field in (
                "readiness_seal_sha256",
                "annotation_derived_cohort_seal_sha256",
                "cohort_artifact_name",
                "cohort_artifact_sha256",
                "cohort_sha256",
                "reference_ledger_sha256",
                "category_namespace_sha256",
            )
        ):
            _fail(
                "readiness component identities changed after verification",
                "analysis.metrics_readiness_component_drift",
            )

    def reference_objects(
        self,
        *,
        image_id: str,
        ledger_scope: ReferenceLedgerScope,
    ) -> tuple[ReferenceObject, ...]:
        """Reconstruct the exact metric reference set from sealed ledger bytes."""

        self.verify_current_artifacts()
        return _reference_objects_from_readiness(
            readiness=self,
            image_id=image_id,
            ledger_scope=ledger_scope,
        )

    @property
    def fingerprint(self) -> str:
        return sha256_payload(self.to_json_dict())

    def to_json_dict(self) -> dict[str, str]:
        return {
            "annotation_derived_cohort_seal_sha256": (
                self.annotation_derived_cohort_seal_sha256
            ),
            "category_namespace_sha256": self.category_namespace_sha256,
            "cohort_artifact_name": self.cohort_artifact_name,
            "cohort_artifact_sha256": self.cohort_artifact_sha256,
            "cohort_sha256": self.cohort_sha256,
            "readiness_seal_sha256": self.readiness_seal_sha256,
            "readiness_root": self.readiness_root,
            "reference_ledger_sha256": self.reference_ledger_sha256,
            "spatial_grid_spec_sha256": self.spatial_grid_spec_sha256,
        }


@dataclass(frozen=True)
class MetricAdmittedImage:
    """One immutable source image and executed spatial grid in metric scope."""

    image_id: str
    image_frozen_order: int
    source_image_sha256: str
    source_width: int
    source_height: int
    spatial_grid: SpatialGrid

    def __post_init__(self) -> None:
        _require_nonempty(self.image_id, field="image_id")
        _require_nonnegative_integer(
            self.image_frozen_order, field="image_frozen_order"
        )
        _require_sha256(self.source_image_sha256, field="source_image_sha256")
        _require_positive_integer(self.source_width, field="source_width")
        _require_positive_integer(self.source_height, field="source_height")
        if (self.spatial_grid.source_width, self.spatial_grid.source_height) != (
            self.source_width,
            self.source_height,
        ):
            _fail(
                "admitted spatial grid differs from the source-image frame",
                "analysis.metrics_admission_grid_frame",
                image_id=self.image_id,
            )

    def to_identity_dict(self) -> dict[str, Any]:
        return {
            "image_frozen_order": self.image_frozen_order,
            "image_id": self.image_id,
            "source_height": self.source_height,
            "source_image_sha256": self.source_image_sha256,
            "source_width": self.source_width,
            "spatial_grid_spec_sha256": self.spatial_grid.spec.fingerprint,
        }


@dataclass(frozen=True)
class MetricAdmittedRequest:
    """One scheduled request joined to its unique terminal attempt."""

    request: ScheduledRequest
    attempt_status: AttemptStatus
    output_artifact_sha256: str | None
    output_artifact_path: str | None
    failure_code: str | None
    produced_cumulative_state_sha256: str | None
    produced_cumulative_state_artifact_path: str | None

    def __post_init__(self) -> None:
        if self.attempt_status not in {
            "completed",
            "failed",
            "skipped",
            "capped",
            "invalid",
        }:
            _fail(
                "admitted request has an unknown terminal status",
                "analysis.metrics_admission_attempt_status",
                request_id=self.request.request_id,
            )
        if self.output_artifact_sha256 is not None:
            _require_sha256(
                self.output_artifact_sha256, field="output_artifact_sha256"
            )
        if self.output_artifact_path is not None:
            _require_nonempty(self.output_artifact_path, field="output_artifact_path")
        if self.output_artifact_sha256 is None or self.output_artifact_path is None:
            _fail(
                "every admitted terminal request requires its canonical output bundle",
                "analysis.metrics_admission_terminal_bundle",
                request_id=self.request.request_id,
            )
        if (self.produced_cumulative_state_sha256 is None) != (
            self.produced_cumulative_state_artifact_path is None
        ):
            _fail(
                "produced cumulative state requires both artifact path and digest",
                "analysis.metrics_admission_terminal_bundle_cumulative_state",
                request_id=self.request.request_id,
            )
        if self.produced_cumulative_state_sha256 is not None:
            _require_sha256(
                self.produced_cumulative_state_sha256,
                field="produced_cumulative_state_sha256",
            )
            _require_nonempty(
                self.produced_cumulative_state_artifact_path,
                field="produced_cumulative_state_artifact_path",
            )
        bundle = TerminalCallOutputBundle.from_path(self.output_artifact_path)
        if bundle.bundle_sha256 != self.output_artifact_sha256:
            _fail(
                "terminal output bundle differs from attempt digest",
                "analysis.metrics_admission_terminal_bundle_digest",
                request_id=self.request.request_id,
            )
        if bundle.payload.get("request_id") != self.request.request_id or bundle.payload.get(
            "attempt_status"
        ) != self.attempt_status:
            _fail(
                "terminal output bundle differs from admitted request",
                "analysis.metrics_admission_terminal_bundle_request",
                request_id=self.request.request_id,
            )
        if self.attempt_status == "completed":
            if self.failure_code is not None:
                _fail(
                    "completed admitted request requires output and no failure code",
                    "analysis.metrics_admission_completed_attempt",
                    request_id=self.request.request_id,
                )
        elif not self.failure_code or bundle.payload.get("failure_code") != self.failure_code:
            _fail(
                "non-completed admitted request requires matching failure evidence",
                "analysis.metrics_admission_noncompleted_attempt",
                request_id=self.request.request_id,
            )
        expected_cumulative_product = (
            None
            if self.produced_cumulative_state_sha256 is None
            else {
                "artifact_path": self.produced_cumulative_state_artifact_path,
                "artifact_sha256": self.produced_cumulative_state_sha256,
            }
        )
        if bundle.payload.get("cumulative_state_product") != expected_cumulative_product:
            _fail(
                "terminal bundle cumulative state differs from attempt record",
                "analysis.metrics_admission_terminal_bundle_cumulative_state",
                request_id=self.request.request_id,
            )

    def to_identity_dict(self) -> dict[str, Any]:
        return {
            "attempt_status": self.attempt_status,
            "failure_code": self.failure_code,
            "output_artifact_sha256": self.output_artifact_sha256,
            "output_artifact_path": self.output_artifact_path,
            "produced_cumulative_state_artifact_path": (
                self.produced_cumulative_state_artifact_path
            ),
            "produced_cumulative_state_sha256": (
                self.produced_cumulative_state_sha256
            ),
            "request_id": self.request.request_id,
        }


@dataclass(frozen=True)
class MetricAdmissionContract:
    """Immutable whole-run gate preceding every metric-bearing primitive."""

    schedule_sha256: str
    schedule_identity_sha256: str
    physical_batch_plan_sha256: str
    grid_provenance_sha256: str
    cohort_sha256: str
    readiness: MetricReadinessIdentity
    images: tuple[MetricAdmittedImage, ...]
    requests: tuple[MetricAdmittedRequest, ...]

    def __post_init__(self) -> None:
        for field_name in (
            "schedule_sha256",
            "schedule_identity_sha256",
            "physical_batch_plan_sha256",
            "grid_provenance_sha256",
            "cohort_sha256",
        ):
            _require_sha256(getattr(self, field_name), field=field_name)
        image_orders = [image.image_frozen_order for image in self.images]
        if image_orders != list(range(len(self.images))):
            _fail(
                "metric admission image order is not the frozen contiguous order",
                "analysis.metrics_admission_image_order",
            )
        image_ids = [image.image_id for image in self.images]
        if len(image_ids) != len(set(image_ids)):
            _fail(
                "metric admission repeats an image identity",
                "analysis.metrics_admission_duplicate_image",
            )
        request_ids = [item.request.request_id for item in self.requests]
        if len(request_ids) != len(set(request_ids)):
            _fail(
                "metric admission repeats a request identity",
                "analysis.metrics_admission_duplicate_request",
            )
        if self.readiness.cohort_sha256 != self.cohort_sha256:
            _fail(
                "admission readiness and cohort identities differ",
                "analysis.metrics_admission_internal_cohort",
            )
        images_by_id = {image.image_id: image for image in self.images}
        expected_schedule_indexes = list(range(len(self.requests)))
        observed_schedule_indexes = [
            item.request.schedule_index for item in self.requests
        ]
        if observed_schedule_indexes != expected_schedule_indexes:
            _fail(
                "admitted requests do not retain canonical schedule order",
                "analysis.metrics_admission_request_order",
            )
        for item in self.requests:
            request = item.request
            image = images_by_id.get(str(request.image_id))
            if image is None:
                _fail(
                    "admitted request references an image outside the cohort",
                    "analysis.metrics_admission_request_image",
                    request_id=request.request_id,
                )
            if (
                request.schedule_identity_sha256 != self.schedule_identity_sha256
                or request.grid_sha256 != self.grid_provenance_sha256
                or request.image_frozen_order != image.image_frozen_order
                or request.image_sha256 != image.source_image_sha256
            ):
                _fail(
                    "admitted request identity differs from schedule, grid, or source image",
                    "analysis.metrics_admission_request_provenance",
                    request_id=request.request_id,
                )
        for image in self.images:
            for arm in PRIMARY_ARM_DEFINITIONS:
                requests = tuple(
                    item.request
                    for item in self.requests
                    if str(item.request.image_id) == image.image_id
                    and item.request.arm.arm_code == arm.arm_code
                )
                expected_cells = (
                    tuple(range(GRID_CELL_COUNT)) if arm.uses_spatial_cell else (None,)
                )
                if tuple(request.cell_index for request in requests) != expected_cells:
                    _fail(
                        "admitted image arm is missing or repeats scheduled calls",
                        "analysis.metrics_admission_primary_matrix",
                        image_id=image.image_id,
                        arm_identifier=arm.arm_code,
                    )

    @property
    def fingerprint(self) -> str:
        return sha256_payload(self.to_identity_dict())

    def to_identity_dict(self) -> dict[str, Any]:
        return {
            "cohort_sha256": self.cohort_sha256,
            "images": [image.to_identity_dict() for image in self.images],
            "physical_batch_plan_sha256": self.physical_batch_plan_sha256,
            "grid_provenance_sha256": self.grid_provenance_sha256,
            "readiness": self.readiness.to_json_dict(),
            "requests": [request.to_identity_dict() for request in self.requests],
            "schedule_identity_sha256": self.schedule_identity_sha256,
            "schedule_sha256": self.schedule_sha256,
        }

    def image(self, image_id: str) -> MetricAdmittedImage:
        matches = tuple(image for image in self.images if image.image_id == image_id)
        if len(matches) != 1:
            _fail(
                "image is absent from the sealed metric-admission universe",
                "analysis.metrics_admission_unknown_image",
                image_id=image_id,
            )
        return matches[0]

    def image_arm_requests(
        self, *, image_id: str, arm_identifier: str
    ) -> tuple[MetricAdmittedRequest, ...]:
        matches = tuple(
            item
            for item in self.requests
            if str(item.request.image_id) == image_id
            and item.request.arm.arm_code == arm_identifier
        )
        if not matches:
            _fail(
                "image arm is absent from the sealed metric-admission universe",
                "analysis.metrics_admission_unknown_image_arm",
                image_id=image_id,
                arm_identifier=arm_identifier,
            )
        return matches


def admit_metric_schedule(
    *,
    schedule: ResearchSchedule,
    cohort: CohortLedger,
    readiness: MetricReadinessIdentity,
    attempt_ledger: AttemptLedger,
    spatial_grids_by_image_id: Mapping[str, SpatialGrid],
) -> MetricAdmissionContract:
    """Seal the complete terminal request universe before metric construction."""

    readiness.verify_current_artifacts()

    if schedule.identity.cohort_sha256 != cohort.fingerprint:
        _fail(
            "schedule and cohort identities differ at metric admission",
            "analysis.metrics_admission_cohort_schedule",
        )
    if readiness.cohort_sha256 != cohort.fingerprint:
        _fail(
            "readiness and cohort identities differ at metric admission",
            "analysis.metrics_admission_cohort_readiness",
        )
    if (
        readiness.readiness_seal_sha256
        != schedule.identity.execution_identity.ledger_sha256
    ):
        _fail(
            "readiness ledger differs from the scheduled execution identity",
            "analysis.metrics_admission_readiness_ledger",
        )
    if (
        readiness.spatial_grid_spec_sha256
        != schedule.identity.grid.canonical_spatial_spec_sha256
    ):
        _fail(
            "readiness grid differs from the scheduled grid provenance",
            "analysis.metrics_admission_readiness_grid",
        )
    if (
        attempt_ledger.run_id != schedule.identity.run_id
        or attempt_ledger.schedule_sha256 != schedule.fingerprint
        or attempt_ledger.execution_identity != schedule.identity.execution_identity
    ):
        _fail(
            "attempt ledger belongs to another sealed schedule",
            "analysis.metrics_admission_attempt_schedule",
        )
    scheduled_request_ids = frozenset(schedule.request_ids)
    attempted_request_ids = attempt_ledger.attempted_request_ids
    missing = sorted(scheduled_request_ids - attempted_request_ids)
    extra = sorted(attempted_request_ids - scheduled_request_ids)
    if missing or extra:
        _fail(
            "metric admission requires exactly one terminal attempt for every scheduled request",
            "analysis.metrics_admission_terminal_universe",
            missing_request_ids=missing,
            extra_request_ids=extra,
        )
    attempt_ledger.validate_dependencies(
        schedule.attempt_dependencies,
        verify_cumulative_state_artifacts=False,
    )
    expected_image_ids = tuple(str(record.image_id) for record in cohort.records)
    if set(spatial_grids_by_image_id) != set(expected_image_ids):
        _fail(
            "spatial-grid universe differs from the frozen cohort",
            "analysis.metrics_admission_grid_universe",
            missing_image_ids=sorted(
                set(expected_image_ids) - set(spatial_grids_by_image_id)
            ),
            extra_image_ids=sorted(
                set(spatial_grids_by_image_id) - set(expected_image_ids)
            ),
        )
    admitted_images: list[MetricAdmittedImage] = []
    for record in cohort.records:
        grid = spatial_grids_by_image_id[str(record.image_id)]
        if grid.spec.fingerprint != readiness.spatial_grid_spec_sha256:
            _fail(
                "executed spatial grid differs from the frozen readiness grid",
                "analysis.metrics_admission_executed_grid",
                image_id=record.image_id,
            )
        admitted_images.append(
            MetricAdmittedImage(
                image_id=str(record.image_id),
                image_frozen_order=record.frozen_order,
                source_image_sha256=record.image_sha256,
                source_width=record.source_width,
                source_height=record.source_height,
                spatial_grid=grid,
            )
        )
    attempts_by_id = attempt_ledger.records_by_request_id
    admitted_requests = tuple(
        MetricAdmittedRequest(
            request=request,
            attempt_status=attempts_by_id[request.request_id].attempt_status,
            output_artifact_sha256=(
                attempts_by_id[request.request_id].output_artifact_sha256
            ),
            output_artifact_path=attempts_by_id[request.request_id].output_artifact_path,
            failure_code=attempts_by_id[request.request_id].failure_code,
            produced_cumulative_state_sha256=(
                attempts_by_id[request.request_id].produced_cumulative_state_sha256
            ),
            produced_cumulative_state_artifact_path=(
                attempts_by_id[
                    request.request_id
                ].produced_cumulative_state_artifact_path
            ),
        )
        for request in schedule.requests
    )
    return MetricAdmissionContract(
        schedule_sha256=schedule.fingerprint,
        schedule_identity_sha256=schedule.identity.fingerprint,
        physical_batch_plan_sha256=schedule.physical_batch_plan.fingerprint,
        grid_provenance_sha256=schedule.identity.grid.fingerprint,
        cohort_sha256=cohort.fingerprint,
        readiness=readiness,
        images=tuple(admitted_images),
        requests=admitted_requests,
    )


@dataclass(frozen=True)
class ReferenceObject:
    """One immutable accepted, ignore, uncertain, or out-of-scope reference."""

    image_id: str
    ledger_scope: ReferenceLedgerScope
    reference_id: str
    normalized_category_name: str | None
    evaluator_category_id: int | None
    official_coco_category_id: int | None
    category_namespace_sha256: str
    source_canvas_bbox_xyxy: PixelBox | None
    state: ReferenceState
    provenance: str
    candidate_category_names: tuple[str, ...] = ()
    owner_cell_index: int | None = None
    core_interior_for_mask_harm: bool = False

    def __post_init__(self) -> None:
        _require_nonempty(self.image_id, field="image_id")
        _require_nonempty(self.reference_id, field="reference_id")
        _require_nonempty(self.provenance, field="provenance")
        if self.ledger_scope not in {"official_annotation", "audit_augmented"}:
            _fail("unknown reference-ledger scope", "analysis.metrics_ledger_scope")
        if self.state not in {
            "accepted",
            "ambiguous",
            "partial",
            "crowd",
            "out_of_scope",
        }:
            _fail("unknown reference state", "analysis.metrics_reference_state")
        if self.ledger_scope == "official_annotation":
            if self.provenance != "official_annotation":
                _fail(
                    "official ledger records require official-annotation provenance",
                    "analysis.metrics_official_provenance",
                    reference_id=self.reference_id,
                )
            if self.state not in {"accepted", "crowd"}:
                _fail(
                    "official ledger may contain only individuals or crowd ignores",
                    "analysis.metrics_official_state",
                    reference_id=self.reference_id,
                )
            expected_prefix = "coco-crowd:" if self.state == "crowd" else "coco-ann:"
            if not self.reference_id.startswith(expected_prefix):
                _fail(
                    "official reference identifier uses the wrong object namespace",
                    "analysis.metrics_official_identifier",
                    reference_id=self.reference_id,
                )
        if self.normalized_category_name is None:
            normalized = None
            expected_evaluator_category_id = None
            expected_official_coco_category_id = None
        else:
            normalized = normalize_coco_category_name(self.normalized_category_name)
            expected_evaluator_category_id = (
                COCO_80_EVALUATOR_LOCAL_CATEGORY_ID_BY_NAME.get(normalized)
            )
            expected_official_coco_category_id = (
                COCO_80_OFFICIAL_CATEGORY_ID_BY_NAME.get(normalized)
            )
        if self.category_namespace_sha256 != COCO_80_CATEGORY_NAMESPACE_SHA256:
            _fail(
                "reference category namespace digest is not canonical",
                "analysis.metrics_reference_category_namespace",
                reference_id=self.reference_id,
            )
        if self.state == "out_of_scope":
            if (
                self.normalized_category_name is not None
                or self.evaluator_category_id is not None
                or self.official_coco_category_id is not None
            ):
                _fail(
                    "out-of-scope references must not claim a COCO-80 category",
                    "analysis.metrics_out_of_scope_category",
                    reference_id=self.reference_id,
                )
        elif self.state == "ambiguous" and self.normalized_category_name is None:
            if (
                self.evaluator_category_id is not None
                or self.official_coco_category_id is not None
            ):
                _fail(
                    "category-unresolved ambiguous references cannot claim category identifiers",
                    "analysis.metrics_reference_category",
                    reference_id=self.reference_id,
                )
            if not self.candidate_category_names:
                _fail(
                    "category-unresolved ambiguous references require candidate categories",
                    "analysis.metrics_ambiguous_candidate_categories",
                    reference_id=self.reference_id,
                )
        elif (
            expected_evaluator_category_id is None
            or expected_evaluator_category_id != self.evaluator_category_id
            or expected_official_coco_category_id is None
            or expected_official_coco_category_id != self.official_coco_category_id
        ):
            _fail(
                "reference category name, evaluator identifier, and official identifier disagree",
                "analysis.metrics_reference_category",
                reference_id=self.reference_id,
            )
        if normalized != self.normalized_category_name:
            _fail(
                "reference category must already be normalized",
                "analysis.metrics_reference_category_not_normalized",
                reference_id=self.reference_id,
            )
        if tuple(sorted(set(self.candidate_category_names))) != (
            self.candidate_category_names
        ):
            _fail(
                "candidate categories must be an immutable sorted unique tuple",
                "analysis.metrics_candidate_categories",
                reference_id=self.reference_id,
            )
        for candidate in self.candidate_category_names:
            normalized_candidate = normalize_coco_category_name(candidate)
            if (
                COCO_80_EVALUATOR_LOCAL_CATEGORY_ID_BY_NAME.get(normalized_candidate)
                is None
                or normalized_candidate != candidate
            ):
                _fail(
                    "candidate category is outside COCO-80",
                    "analysis.metrics_candidate_category",
                    reference_id=self.reference_id,
                )
        if self.state in {"accepted", "ambiguous", "crowd"} and (
            self.source_canvas_bbox_xyxy is None
        ):
            _fail(
                "reference state requires a source-canvas box",
                "analysis.metrics_reference_box_missing",
                reference_id=self.reference_id,
            )
        if self.state != "accepted" and (
            self.owner_cell_index is not None or self.core_interior_for_mask_harm
        ):
            _fail(
                "only accepted individual references may own cells or mask-harm status",
                "analysis.metrics_nonindividual_ownership",
                reference_id=self.reference_id,
            )
        if self.source_canvas_bbox_xyxy is not None:
            _validate_box(self.source_canvas_bbox_xyxy, field="reference_bbox")
        if self.owner_cell_index is not None and not (
            0 <= self.owner_cell_index < GRID_CELL_COUNT
        ):
            _fail(
                "owner cell index must be in the frozen four-by-four grid",
                "analysis.metrics_owner_cell",
                reference_id=self.reference_id,
            )

    @property
    def is_individual_rescue_reference(self) -> bool:
        return self.state == "accepted"

    @property
    def categories_for_uncertainty_ignore(self) -> frozenset[str]:
        if self.candidate_category_names:
            return frozenset(self.candidate_category_names)
        if self.normalized_category_name is None:
            return frozenset()
        return frozenset({self.normalized_category_name})

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "candidate_category_names": list(self.candidate_category_names),
            "category_namespace_sha256": self.category_namespace_sha256,
            "core_interior_for_mask_harm": self.core_interior_for_mask_harm,
            "evaluator_category_id": self.evaluator_category_id,
            "image_id": self.image_id,
            "ledger_scope": self.ledger_scope,
            "normalized_category_name": self.normalized_category_name,
            "official_coco_category_id": self.official_coco_category_id,
            "owner_cell_index": self.owner_cell_index,
            "provenance": self.provenance,
            "reference_id": self.reference_id,
            "source_canvas_bbox_xyxy": (
                list(self.source_canvas_bbox_xyxy)
                if self.source_canvas_bbox_xyxy is not None
                else None
            ),
            "state": self.state,
        }


@dataclass(frozen=True)
class ReferenceMatch:
    """One exact one-to-one prediction/reference match."""

    prediction_id: str
    reference_id: str
    normalized_category_name: str
    intersection_over_union: float

    def __post_init__(self) -> None:
        _require_nonempty(self.prediction_id, field="prediction_id")
        _require_nonempty(self.reference_id, field="reference_id")
        if not math.isfinite(self.intersection_over_union) or not (
            0.0 <= self.intersection_over_union <= 1.0
        ):
            _fail(
                "match Intersection over Union must be finite and bounded",
                "analysis.metrics_match_iou",
            )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "intersection_over_union": self.intersection_over_union,
            "normalized_category_name": self.normalized_category_name,
            "prediction_id": self.prediction_id,
            "reference_id": self.reference_id,
        }


@dataclass(frozen=True)
class ReferenceMatchResult:
    """Exact deterministic assignment plus unmatched identifiers."""

    threshold: float
    matches: tuple[ReferenceMatch, ...]
    unmatched_prediction_ids: tuple[str, ...]
    unmatched_reference_ids: tuple[str, ...]
    solver: str = EXACT_MATCH_SOLVER
    objective_tolerance: float = EXACT_MATCH_OBJECTIVE_TOLERANCE

    @property
    def matched_prediction_ids(self) -> tuple[str, ...]:
        return tuple(sorted(match.prediction_id for match in self.matches))

    @property
    def matched_reference_ids(self) -> tuple[str, ...]:
        return tuple(sorted(match.reference_id for match in self.matches))

    @property
    def total_intersection_over_union(self) -> float:
        return math.fsum(match.intersection_over_union for match in self.matches)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "matches": [match.to_json_dict() for match in self.matches],
            "objective_tolerance": self.objective_tolerance,
            "solver": self.solver,
            "threshold": self.threshold,
            "total_intersection_over_union": self.total_intersection_over_union,
            "unmatched_prediction_ids": list(self.unmatched_prediction_ids),
            "unmatched_reference_ids": list(self.unmatched_reference_ids),
        }


@dataclass(frozen=True)
class RowMetricRecord:
    """One row-level sufficient-statistics record."""

    canonical_call_id: str
    generated_row_index: int
    prediction_id: str | None
    parse_status: RowParseStatus
    validity_status: RowValidityStatus
    ownership_status: RowOwnershipStatus
    matched_reference_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_nonempty(self.canonical_call_id, field="canonical_call_id")
        _require_nonnegative_integer(
            self.generated_row_index, field="generated_row_index"
        )
        _validate_sorted_unique(self.matched_reference_ids, "matched_reference_ids")
        if self.parse_status not in {"parsed", "malformed"}:
            _fail(
                "row parse status is not normalized",
                "analysis.metrics_row_parse_status",
                canonical_call_id=self.canonical_call_id,
            )
        if self.validity_status not in {"valid", "invalid", "not_applicable"}:
            _fail(
                "row validity status is not normalized",
                "analysis.metrics_row_validity_status",
                canonical_call_id=self.canonical_call_id,
            )
        if self.ownership_status not in {"owned", "non_owning", "not_applicable"}:
            _fail(
                "row ownership status is not normalized",
                "analysis.metrics_row_ownership_status",
                canonical_call_id=self.canonical_call_id,
            )
        if self.parse_status == "malformed":
            expected_state = (None, "not_applicable", "not_applicable", ())
        elif self.validity_status == "invalid":
            expected_state = (None, "invalid", "not_applicable", ())
        elif self.validity_status == "valid":
            if self.prediction_id is None:
                _fail(
                    "valid row requires a prediction identifier",
                    "analysis.metrics_row_prediction_missing",
                    canonical_call_id=self.canonical_call_id,
                )
            return
        else:
            expected_state = (None, "invalid", "not_applicable", ())
        observed = (
            self.prediction_id,
            self.validity_status,
            self.ownership_status,
            self.matched_reference_ids,
        )
        if observed != expected_state:
            _fail(
                "row parse, validity, prediction, ownership, and match states disagree",
                "analysis.metrics_row_state",
                canonical_call_id=self.canonical_call_id,
                generated_row_index=self.generated_row_index,
            )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "canonical_call_id": self.canonical_call_id,
            "generated_row_index": self.generated_row_index,
            "matched_reference_ids": list(self.matched_reference_ids),
            "ownership_status": self.ownership_status,
            "parse_status": self.parse_status,
            "prediction_id": self.prediction_id,
            "validity_status": self.validity_status,
        }


@dataclass(frozen=True)
class CallMetricRecord:
    """One call-level sufficient-statistics and realized-budget record."""

    canonical_call_id: str
    canonical_cell_index: int | None
    sampling_seed: int
    raw_any_call_matched_reference_ids: tuple[str, ...]
    raw_owning_call_matched_reference_ids: tuple[str, ...]
    valid_prediction_ids: tuple[str, ...]
    attempted_row_count: int
    malformed_row_count: int
    invalid_row_count: int
    non_owning_prediction_count: int
    natural_closure_count: int
    controller_cap_count: int
    token_cap_count: int
    error_count: int
    prompt_token_count: int
    image_token_count: int
    generated_token_count: int
    wall_time_seconds: float
    peak_device_memory_bytes: int
    attempt_status: AttemptStatus = "completed"
    source_image_sha256: str | None = None
    image_frozen_order: int | None = None
    terminal_attempt_output_artifact_sha256: str | None = None
    terminal_attempt_output_artifact_path: str | None = None
    terminal_attempt_failure_code: str | None = None

    def __post_init__(self) -> None:
        _require_nonempty(self.canonical_call_id, field="canonical_call_id")
        if self.canonical_cell_index is not None:
            _require_nonnegative_integer(
                self.canonical_cell_index, field="canonical_cell_index"
            )
        _require_nonnegative_integer(self.sampling_seed, field="sampling_seed")
        _validate_sorted_unique(
            self.raw_any_call_matched_reference_ids,
            "raw_any_call_matched_reference_ids",
        )
        _validate_sorted_unique(
            self.raw_owning_call_matched_reference_ids,
            "raw_owning_call_matched_reference_ids",
        )
        _validate_sorted_unique(self.valid_prediction_ids, "valid_prediction_ids")
        for field_name in (
            "attempted_row_count",
            "malformed_row_count",
            "invalid_row_count",
            "non_owning_prediction_count",
            "natural_closure_count",
            "controller_cap_count",
            "token_cap_count",
            "error_count",
            "prompt_token_count",
            "image_token_count",
            "generated_token_count",
            "peak_device_memory_bytes",
        ):
            _require_nonnegative_integer(getattr(self, field_name), field=field_name)
        if not math.isfinite(self.wall_time_seconds) or self.wall_time_seconds < 0:
            _fail("wall time must be finite and nonnegative", "analysis.metrics_wall")
        if self.attempt_status not in {
            "completed",
            "failed",
            "skipped",
            "capped",
            "invalid",
        }:
            _fail(
                "call record has an unknown terminal attempt status",
                "analysis.metrics_call_attempt_status",
                canonical_call_id=self.canonical_call_id,
            )
        if self.source_image_sha256 is not None:
            _require_sha256(self.source_image_sha256, field="source_image_sha256")
        if self.image_frozen_order is not None:
            _require_nonnegative_integer(
                self.image_frozen_order, field="image_frozen_order"
            )
        if self.terminal_attempt_output_artifact_sha256 is not None:
            _require_sha256(
                self.terminal_attempt_output_artifact_sha256,
                field="terminal_attempt_output_artifact_sha256",
            )
        if self.terminal_attempt_output_artifact_path is not None:
            _require_nonempty(
                self.terminal_attempt_output_artifact_path,
                field="terminal_attempt_output_artifact_path",
            )
        if self.terminal_attempt_failure_code is not None:
            _require_nonempty(
                self.terminal_attempt_failure_code,
                field="terminal_attempt_failure_code",
            )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "attempted_row_count": self.attempted_row_count,
            "attempt_status": self.attempt_status,
            "canonical_call_id": self.canonical_call_id,
            "canonical_cell_index": self.canonical_cell_index,
            "controller_cap_count": self.controller_cap_count,
            "error_count": self.error_count,
            "generated_token_count": self.generated_token_count,
            "image_token_count": self.image_token_count,
            "image_frozen_order": self.image_frozen_order,
            "invalid_row_count": self.invalid_row_count,
            "malformed_row_count": self.malformed_row_count,
            "raw_any_call_matched_reference_ids": list(
                self.raw_any_call_matched_reference_ids
            ),
            "raw_owning_call_matched_reference_ids": list(
                self.raw_owning_call_matched_reference_ids
            ),
            "natural_closure_count": self.natural_closure_count,
            "non_owning_prediction_count": self.non_owning_prediction_count,
            "peak_device_memory_bytes": self.peak_device_memory_bytes,
            "prompt_token_count": self.prompt_token_count,
            "sampling_seed": self.sampling_seed,
            "source_image_sha256": self.source_image_sha256,
            "terminal_attempt_failure_code": self.terminal_attempt_failure_code,
            "terminal_attempt_output_artifact_sha256": (
                self.terminal_attempt_output_artifact_sha256
            ),
            "terminal_attempt_output_artifact_path": (
                self.terminal_attempt_output_artifact_path
            ),
            "token_cap_count": self.token_cap_count,
            "valid_prediction_ids": list(self.valid_prediction_ids),
            "wall_time_seconds": self.wall_time_seconds,
        }


@dataclass(frozen=True)
class CellMetricRecord:
    """One canonical cell's ownership and raw-call reference outcomes."""

    canonical_cell_index: int
    owned_reference_ids: tuple[str, ...]
    raw_owning_call_matched_reference_ids: tuple[str, ...]
    canonical_call_id: str

    def __post_init__(self) -> None:
        _require_nonnegative_integer(
            self.canonical_cell_index, field="canonical_cell_index"
        )
        _require_nonempty(self.canonical_call_id, field="canonical_call_id")
        _validate_sorted_unique(self.owned_reference_ids, "owned_reference_ids")
        _validate_sorted_unique(
            self.raw_owning_call_matched_reference_ids,
            "raw_owning_call_matched_reference_ids",
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "canonical_call_id": self.canonical_call_id,
            "canonical_cell_index": self.canonical_cell_index,
            "raw_owning_call_matched_reference_ids": list(
                self.raw_owning_call_matched_reference_ids
            ),
            "owned_reference_ids": list(self.owned_reference_ids),
        }


@dataclass(frozen=True)
class ImageMetricPrimitive:
    """Recomputable per-image primitive for one arm, scope, and IoU threshold."""

    image_id: str
    metric_admission_sha256: str
    schedule_sha256: str
    cohort_sha256: str
    readiness_seal_sha256: str
    source_image_sha256: str
    image_frozen_order: int
    spatial_grid_spec_sha256: str
    arm_identifier: str
    source_width: int
    source_height: int
    ledger_scope: ReferenceLedgerScope
    category_namespace_sha256: str
    threshold: float
    accepted_reference_ids: tuple[str, ...]
    core_interior_reference_ids: tuple[str, ...]
    raw_owning_match: ReferenceMatchResult
    raw_any_call_union_match: ReferenceMatchResult
    pre_merge_match: ReferenceMatchResult
    post_merge_match: ReferenceMatchResult
    merge_created_reference_ids: tuple[str, ...]
    merge_destroyed_reference_ids: tuple[str, ...]
    crowd_ignored_prediction_ids: tuple[str, ...]
    uncertainty_ignored_prediction_ids: tuple[str, ...]
    unmatched_valid_prediction_ids: tuple[str, ...]
    malformed_row_count: int
    invalid_row_count: int
    non_owning_prediction_count: int
    attempted_row_count: int
    attempted_call_count: int
    natural_closure_count: int
    controller_cap_count: int
    token_cap_count: int
    error_count: int
    invalid_call_count: int
    final_valid_prediction_count: int
    pre_merge_strict_duplicate_components: tuple[StrictDuplicateComponent, ...]
    post_merge_strict_duplicate_components: tuple[StrictDuplicateComponent, ...]
    configured_call_budget: int
    realized_call_count: int
    prompt_token_count: int
    image_token_count: int
    generated_token_count: int
    wall_time_seconds: float
    peak_device_memory_bytes: int
    reference_object_count_stratum_value: int
    prompt_token_count_stratum_value: int
    attempted_row_count_stratum_value: int
    density_tags: tuple[str, ...]
    cell_records: tuple[CellMetricRecord, ...]
    call_records: tuple[CallMetricRecord, ...]
    row_records: tuple[RowMetricRecord, ...]

    def __post_init__(self) -> None:
        _require_nonempty(self.image_id, field="image_id")
        for field_name in (
            "metric_admission_sha256",
            "schedule_sha256",
            "cohort_sha256",
            "readiness_seal_sha256",
            "source_image_sha256",
            "spatial_grid_spec_sha256",
        ):
            _require_sha256(getattr(self, field_name), field=field_name)
        _require_nonnegative_integer(
            self.image_frozen_order, field="image_frozen_order"
        )
        _require_nonempty(self.arm_identifier, field="arm_identifier")
        _require_positive_integer(self.source_width, field="source_width")
        _require_positive_integer(self.source_height, field="source_height")
        if self.ledger_scope not in {"official_annotation", "audit_augmented"}:
            _fail("unknown reference-ledger scope", "analysis.metrics_ledger_scope")
        if self.category_namespace_sha256 != COCO_80_CATEGORY_NAMESPACE_SHA256:
            _fail(
                "metric primitive category namespace digest is not canonical",
                "analysis.metrics_primitive_category_namespace",
                image_id=self.image_id,
            )
        if self.threshold not in {
            PRIMARY_MATCH_INTERSECTION_OVER_UNION_THRESHOLD,
            LOCALIZATION_SENSITIVITY_INTERSECTION_OVER_UNION_THRESHOLD,
        }:
            _fail(
                "unsupported primitive match threshold",
                "analysis.metrics_match_threshold",
            )
        _validate_sorted_unique(self.accepted_reference_ids, "accepted_reference_ids")
        if not set(self.core_interior_reference_ids).issubset(
            self.accepted_reference_ids
        ):
            _fail(
                "mask-harm references must be accepted individual references",
                "analysis.metrics_core_interior_subset",
                image_id=self.image_id,
            )
        for field_name in (
            "merge_created_reference_ids",
            "merge_destroyed_reference_ids",
            "crowd_ignored_prediction_ids",
            "uncertainty_ignored_prediction_ids",
            "unmatched_valid_prediction_ids",
            "core_interior_reference_ids",
            "density_tags",
        ):
            _validate_sorted_unique(getattr(self, field_name), field_name)
        for match_result in (
            self.raw_owning_match,
            self.raw_any_call_union_match,
            self.pre_merge_match,
            self.post_merge_match,
        ):
            if match_result.threshold != self.threshold:
                _fail(
                    "image primitive match thresholds drifted",
                    "analysis.metrics_primitive_threshold",
                    image_id=self.image_id,
                )
        for field_name in (
            "malformed_row_count",
            "invalid_row_count",
            "non_owning_prediction_count",
            "attempted_row_count",
            "attempted_call_count",
            "natural_closure_count",
            "controller_cap_count",
            "token_cap_count",
            "error_count",
            "invalid_call_count",
            "final_valid_prediction_count",
            "configured_call_budget",
            "realized_call_count",
            "prompt_token_count",
            "image_token_count",
            "generated_token_count",
            "peak_device_memory_bytes",
            "reference_object_count_stratum_value",
            "prompt_token_count_stratum_value",
            "attempted_row_count_stratum_value",
        ):
            _require_nonnegative_integer(getattr(self, field_name), field=field_name)
        if not math.isfinite(self.wall_time_seconds) or self.wall_time_seconds < 0:
            _fail(
                "wall time must be finite and nonnegative",
                "analysis.metrics_wall",
                image_id=self.image_id,
            )
        if self.reference_object_count_stratum_value != len(
            self.accepted_reference_ids
        ):
            _fail(
                "reference-count stratum must preserve the exact accepted count",
                "analysis.metrics_reference_count_stratum",
                image_id=self.image_id,
            )
        if self.prompt_token_count_stratum_value != self.prompt_token_count:
            _fail(
                "prompt-token stratum must preserve the exact prompt-token count",
                "analysis.metrics_prompt_count_stratum",
                image_id=self.image_id,
            )
        if self.attempted_row_count_stratum_value != self.attempted_row_count:
            _fail(
                "row-count stratum must preserve the exact attempted-row count",
                "analysis.metrics_row_count_stratum",
                image_id=self.image_id,
            )

    @property
    def post_merge_matched_reference_ids(self) -> tuple[str, ...]:
        return self.post_merge_match.matched_reference_ids

    @property
    def pre_merge_duplicate_excess_count(self) -> int:
        return sum(
            len(component.member_prediction_ids) - 1
            for component in self.pre_merge_strict_duplicate_components
        )

    @property
    def post_merge_duplicate_excess_count(self) -> int:
        return sum(
            len(component.member_prediction_ids) - 1
            for component in self.post_merge_strict_duplicate_components
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "accepted_reference_ids": list(self.accepted_reference_ids),
            "arm_identifier": self.arm_identifier,
            "attempted_call_count": self.attempted_call_count,
            "attempted_row_count": self.attempted_row_count,
            "attempted_row_count_stratum_value": self.attempted_row_count_stratum_value,
            "call_records": [record.to_json_dict() for record in self.call_records],
            "category_namespace_sha256": self.category_namespace_sha256,
            "cohort_sha256": self.cohort_sha256,
            "cell_records": [record.to_json_dict() for record in self.cell_records],
            "configured_call_budget": self.configured_call_budget,
            "controller_cap_count": self.controller_cap_count,
            "core_interior_reference_ids": list(self.core_interior_reference_ids),
            "crowd_ignored_prediction_ids": list(self.crowd_ignored_prediction_ids),
            "density_tags": list(self.density_tags),
            "error_count": self.error_count,
            "final_valid_prediction_count": self.final_valid_prediction_count,
            "generated_token_count": self.generated_token_count,
            "image_id": self.image_id,
            "image_frozen_order": self.image_frozen_order,
            "metric_admission_sha256": self.metric_admission_sha256,
            "image_token_count": self.image_token_count,
            "invalid_row_count": self.invalid_row_count,
            "invalid_call_count": self.invalid_call_count,
            "ledger_scope": self.ledger_scope,
            "malformed_row_count": self.malformed_row_count,
            "merge_created_reference_ids": list(self.merge_created_reference_ids),
            "merge_destroyed_reference_ids": list(self.merge_destroyed_reference_ids),
            "natural_closure_count": self.natural_closure_count,
            "non_owning_prediction_count": self.non_owning_prediction_count,
            "peak_device_memory_bytes": self.peak_device_memory_bytes,
            "post_merge_match": self.post_merge_match.to_json_dict(),
            "post_merge_strict_duplicate_components": [
                component.to_json_dict()
                for component in self.post_merge_strict_duplicate_components
            ],
            "pre_merge_match": self.pre_merge_match.to_json_dict(),
            "pre_merge_strict_duplicate_components": [
                component.to_json_dict()
                for component in self.pre_merge_strict_duplicate_components
            ],
            "prompt_token_count": self.prompt_token_count,
            "prompt_token_count_stratum_value": self.prompt_token_count_stratum_value,
            "raw_any_call_union_match": self.raw_any_call_union_match.to_json_dict(),
            "raw_owning_match": self.raw_owning_match.to_json_dict(),
            "realized_call_count": self.realized_call_count,
            "reference_object_count_stratum_value": (
                self.reference_object_count_stratum_value
            ),
            "row_records": [record.to_json_dict() for record in self.row_records],
            "readiness_seal_sha256": self.readiness_seal_sha256,
            "schedule_sha256": self.schedule_sha256,
            "source_image_sha256": self.source_image_sha256,
            "source_height": self.source_height,
            "source_width": self.source_width,
            "spatial_grid_spec_sha256": self.spatial_grid_spec_sha256,
            "threshold": self.threshold,
            "token_cap_count": self.token_cap_count,
            "uncertainty_ignored_prediction_ids": list(
                self.uncertainty_ignored_prediction_ids
            ),
            "unmatched_valid_prediction_ids": list(self.unmatched_valid_prediction_ids),
            "wall_time_seconds": self.wall_time_seconds,
        }


@dataclass(frozen=True)
class MetricEstimate:
    """One named aggregate with explicit numerator and denominator."""

    metric_name: str
    numerator: float
    denominator: float
    estimate: float | None
    status: MetricStatus
    scope: str
    threshold: float | None
    comparator: str | None
    comparator_numerator: float | None = None
    comparator_denominator: float | None = None

    def __post_init__(self) -> None:
        for field_name in ("numerator", "denominator"):
            value = getattr(self, field_name)
            if not math.isfinite(value):
                _fail(
                    "metric counts must be finite",
                    "analysis.metrics_nonfinite",
                    field=field_name,
                )
        if self.denominator < 0:
            _fail(
                "metric denominator must be nonnegative", "analysis.metrics_denominator"
            )
        if self.estimate is not None and not math.isfinite(self.estimate):
            _fail("metric estimate must be finite", "analysis.metrics_nonfinite")
        if (self.comparator_numerator is None) != (self.comparator_denominator is None):
            _fail(
                "comparator numerator and denominator must be present together",
                "analysis.metrics_comparator_counts",
            )
        if self.comparator_numerator is not None:
            assert self.comparator_denominator is not None
            if (
                not math.isfinite(self.comparator_numerator)
                or not math.isfinite(self.comparator_denominator)
                or self.comparator_numerator < 0
                or self.comparator_denominator < 0
            ):
                _fail(
                    "comparator counts must be finite and nonnegative",
                    "analysis.metrics_comparator_counts",
                )
        expected_status: MetricStatus = (
            "ok" if self.estimate is not None else "not_applicable"
        )
        if self.status != expected_status:
            _fail("metric status disagrees with estimate", "analysis.metrics_status")

    def to_json_dict(self) -> dict[str, Any]:
        payload = {
            "comparator": self.comparator,
            "denominator": self.denominator,
            "estimate": self.estimate,
            "metric_name": self.metric_name,
            "numerator": self.numerator,
            "scope": self.scope,
            "status": self.status,
            "threshold": self.threshold,
        }
        if self.comparator_numerator is not None:
            payload["comparator_numerator"] = self.comparator_numerator
            payload["comparator_denominator"] = self.comparator_denominator
        return payload


@dataclass(frozen=True)
class OwningSeedPairRecord:
    """Exact request and seed evidence for one image/cell paired comparison."""

    image_id: str
    canonical_cell_index: int
    candidate_request_id: str
    full_bag_request_id: str
    sampling_seed: int
    owned_reference_ids: tuple[str, ...]
    baseline_missed_reference_ids: tuple[str, ...]
    candidate_raw_owning_matched_reference_ids: tuple[str, ...]
    full_bag_raw_call_matched_reference_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_nonempty(self.image_id, field="image_id")
        _require_nonnegative_integer(
            self.canonical_cell_index, field="canonical_cell_index"
        )
        _require_nonempty(self.candidate_request_id, field="candidate_request_id")
        _require_nonempty(self.full_bag_request_id, field="full_bag_request_id")
        _require_nonnegative_integer(self.sampling_seed, field="sampling_seed")
        for field_name in (
            "owned_reference_ids",
            "baseline_missed_reference_ids",
            "candidate_raw_owning_matched_reference_ids",
            "full_bag_raw_call_matched_reference_ids",
        ):
            _validate_sorted_unique(getattr(self, field_name), field_name)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "baseline_missed_reference_ids": list(self.baseline_missed_reference_ids),
            "candidate_raw_owning_matched_reference_ids": list(
                self.candidate_raw_owning_matched_reference_ids
            ),
            "candidate_request_id": self.candidate_request_id,
            "canonical_cell_index": self.canonical_cell_index,
            "full_bag_raw_call_matched_reference_ids": list(
                self.full_bag_raw_call_matched_reference_ids
            ),
            "full_bag_request_id": self.full_bag_request_id,
            "image_id": self.image_id,
            "owned_reference_ids": list(self.owned_reference_ids),
            "sampling_seed": self.sampling_seed,
        }


@dataclass(frozen=True)
class AggregateMetricReport:
    """Named aggregate views over a paired image cohort."""

    arm_identifier: str
    ledger_scope: ReferenceLedgerScope
    category_namespace_sha256: str
    scope: str
    threshold: float
    metrics: tuple[MetricEstimate, ...]
    owning_seed_pair_records: tuple[OwningSeedPairRecord, ...] = ()

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "arm_identifier": self.arm_identifier,
            "category_namespace_sha256": self.category_namespace_sha256,
            "ledger_scope": self.ledger_scope,
            "metrics": [metric.to_json_dict() for metric in self.metrics],
            "owning_seed_pair_records": [
                record.to_json_dict() for record in self.owning_seed_pair_records
            ],
            "scope": self.scope,
            "threshold": self.threshold,
        }


@dataclass(frozen=True)
class BootstrapImageRecord:
    """One image bundle's paired sufficient statistics for a bootstrap metric."""

    image_id: str
    candidate_numerator: float
    candidate_denominator: float
    comparator_numerator: float = 0.0
    comparator_denominator: float = 0.0

    def __post_init__(self) -> None:
        _require_nonempty(self.image_id, field="image_id")
        for field_name in (
            "candidate_numerator",
            "candidate_denominator",
            "comparator_numerator",
            "comparator_denominator",
        ):
            value = getattr(self, field_name)
            if not math.isfinite(value) or value < 0:
                _fail(
                    "bootstrap sufficient statistics must be finite and nonnegative",
                    "analysis.metrics_bootstrap_value",
                    field=field_name,
                )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "candidate_denominator": self.candidate_denominator,
            "candidate_numerator": self.candidate_numerator,
            "comparator_denominator": self.comparator_denominator,
            "comparator_numerator": self.comparator_numerator,
            "image_id": self.image_id,
        }


@dataclass(frozen=True)
class BootstrapReplicateRecord:
    """One image-clustered bootstrap replicate."""

    replicate_index: int
    sampled_image_indices: tuple[int, ...]
    numerator: float
    denominator: float
    comparator_numerator: float | None
    comparator_denominator: float | None
    estimate: float | None
    status: MetricStatus

    def to_json_dict(self) -> dict[str, Any]:
        payload = {
            "denominator": self.denominator,
            "estimate": self.estimate,
            "numerator": self.numerator,
            "replicate_index": self.replicate_index,
            "sampled_image_indices": list(self.sampled_image_indices),
            "status": self.status,
        }
        if self.comparator_numerator is not None:
            payload["comparator_numerator"] = self.comparator_numerator
            payload["comparator_denominator"] = self.comparator_denominator
        return payload


@dataclass(frozen=True)
class BootstrapMetricReport:
    """Percentile interval with explicit applicability accounting."""

    metric_name: str
    category_namespace_sha256: str
    estimator: BootstrapEstimator
    point_estimate: MetricEstimate
    lower_bound: float | None
    upper_bound: float | None
    applicable_replicates: int
    total_replicates: int
    minimum_applicable_replicates: int
    confidence_level: float
    sampling_seed: int
    bootstrap_algorithm: str
    evidence_status: BootstrapEvidenceStatus
    records: tuple[BootstrapReplicateRecord, ...]

    def __post_init__(self) -> None:
        _require_nonempty(self.metric_name, field="metric_name")
        _require_nonempty(self.bootstrap_algorithm, field="bootstrap_algorithm")
        for field_name in (
            "applicable_replicates",
            "total_replicates",
            "minimum_applicable_replicates",
            "sampling_seed",
        ):
            _require_nonnegative_integer(getattr(self, field_name), field=field_name)
        if (
            self.total_replicates == 0
            or self.applicable_replicates > self.total_replicates
            or self.minimum_applicable_replicates > self.total_replicates
            or self.confidence_level != 0.95
            or self.point_estimate.metric_name != self.metric_name
        ):
            _fail(
                "bootstrap counts, confidence level, or point-estimate identity drifted",
                "analysis.metrics_bootstrap_report_contract",
            )
        interval_expected = (
            self.applicable_replicates >= self.minimum_applicable_replicates
            and self.point_estimate.status == "ok"
        )
        if (self.lower_bound is None) != (self.upper_bound is None) or (
            interval_expected != (self.lower_bound is not None)
        ):
            _fail(
                "bootstrap interval availability disagrees with applicability",
                "analysis.metrics_bootstrap_interval_contract",
            )
        if self.lower_bound is not None:
            assert self.upper_bound is not None
            if (
                not math.isfinite(self.lower_bound)
                or not math.isfinite(self.upper_bound)
                or self.lower_bound > self.upper_bound
            ):
                _fail(
                    "bootstrap interval must be finite and ordered",
                    "analysis.metrics_bootstrap_interval_contract",
                )
        if self.category_namespace_sha256 != COCO_80_CATEGORY_NAMESPACE_SHA256:
            _fail(
                "bootstrap category namespace digest is not canonical",
                "analysis.metrics_bootstrap_category_namespace",
            )
        if self.evidence_status == "metric_bearing":
            if (
                self.total_replicates != DEFAULT_BOOTSTRAP_REPLICATES
                or self.minimum_applicable_replicates
                != DEFAULT_MINIMUM_APPLICABLE_BOOTSTRAP_REPLICATES
                or self.bootstrap_algorithm != "image_clustered_percentile_95_v1"
                or self.records
            ):
                _fail(
                    "metric-bearing bootstrap must use the frozen compact contract",
                    "analysis.metrics_bootstrap_evidence_contract",
                )
        elif self.evidence_status != "synthetic_only":
            _fail(
                "unknown bootstrap evidence status",
                "analysis.metrics_bootstrap_evidence_status",
            )
        elif len(self.records) != self.total_replicates:
            _fail(
                "synthetic bootstrap records must cover every replicate",
                "analysis.metrics_bootstrap_synthetic_records",
            )

    @property
    def status(self) -> MetricStatus:
        if (
            self.point_estimate.status == "ok"
            and self.applicable_replicates >= self.minimum_applicable_replicates
        ):
            return "ok"
        return "not_applicable"

    def to_json_dict(self) -> dict[str, Any]:
        if self.evidence_status != "metric_bearing":
            _fail(
                "synthetic bootstrap helpers cannot serialize as metric evidence",
                "analysis.metrics_synthetic_bootstrap_evidence",
            )
        return {
            "applicable_replicates": self.applicable_replicates,
            "bootstrap_algorithm": self.bootstrap_algorithm,
            "category_namespace_sha256": self.category_namespace_sha256,
            "confidence_level": self.confidence_level,
            "estimator": self.estimator,
            "evidence_status": self.evidence_status,
            "lower_bound": self.lower_bound,
            "metric_name": self.metric_name,
            "minimum_applicable_replicates": self.minimum_applicable_replicates,
            "point_estimate": self.point_estimate.to_json_dict(),
            "sampling_seed": self.sampling_seed,
            "status": self.status,
            "total_replicates": self.total_replicates,
            "upper_bound": self.upper_bound,
        }


def exact_reference_match(
    predictions: Sequence[NormalizedCallPrediction],
    references: Sequence[ReferenceObject],
    *,
    threshold: float = PRIMARY_MATCH_INTERSECTION_OVER_UNION_THRESHOLD,
) -> ReferenceMatchResult:
    """Return the exact cardinality, IoU, then lexicographic assignment.

    The custom unit-capacity min-cost-flow solver sends maximum feasible flow.
    Edge benefits encode exact binary64 IoU sums above a lexicographic bitset,
    so cardinality is optimized by flow value, exact IoU by the high-order
    integer term, and the lexicographically earliest immutable edge set last.
    """

    if threshold not in {
        PRIMARY_MATCH_INTERSECTION_OVER_UNION_THRESHOLD,
        LOCALIZATION_SENSITIVITY_INTERSECTION_OVER_UNION_THRESHOLD,
    }:
        _fail("unsupported match threshold", "analysis.metrics_match_threshold")
    ordered_predictions = tuple(
        sorted(predictions, key=lambda item: item.prediction_id)
    )
    accepted = tuple(
        sorted(
            (reference for reference in references if reference.state == "accepted"),
            key=lambda item: item.reference_id,
        )
    )
    _validate_prediction_reference_universe(ordered_predictions, references)
    candidate_rows: list[tuple[int, int, float]] = []
    for prediction_index, prediction in enumerate(ordered_predictions):
        for reference_index, reference in enumerate(accepted):
            if (
                prediction.normalized_category_name
                != reference.normalized_category_name
            ):
                continue
            assert reference.source_canvas_bbox_xyxy is not None
            overlap = float(
                iou_xyxy(prediction.global_bbox_xyxy, reference.source_canvas_bbox_xyxy)
            )
            if overlap >= threshold:
                candidate_rows.append((prediction_index, reference_index, overlap))
    if not candidate_rows:
        return ReferenceMatchResult(
            threshold=threshold,
            matches=(),
            unmatched_prediction_ids=tuple(
                prediction.prediction_id for prediction in ordered_predictions
            ),
            unmatched_reference_ids=tuple(
                reference.reference_id for reference in accepted
            ),
        )

    candidate_rows.sort(
        key=lambda row: (
            ordered_predictions[row[0]].prediction_id,
            accepted[row[1]].reference_id,
        )
    )
    denominators = [overlap.as_integer_ratio()[1] for _, _, overlap in candidate_rows]
    common_denominator = max(denominators)
    lexicographic_base = 1 << len(candidate_rows)
    benefits: list[int] = []
    for edge_index, (_, _, overlap) in enumerate(candidate_rows):
        numerator, denominator = overlap.as_integer_ratio()
        exact_iou_integer = numerator * (common_denominator // denominator)
        lexicographic_bit = 1 << (len(candidate_rows) - edge_index - 1)
        benefits.append(exact_iou_integer * lexicographic_base + lexicographic_bit)
    selected_edge_indexes = _exact_maximum_flow_assignment(
        prediction_count=len(ordered_predictions),
        reference_count=len(accepted),
        candidate_rows=candidate_rows,
        benefits=benefits,
    )
    matches = tuple(
        ReferenceMatch(
            prediction_id=ordered_predictions[candidate_rows[index][0]].prediction_id,
            reference_id=accepted[candidate_rows[index][1]].reference_id,
            normalized_category_name=ordered_predictions[
                candidate_rows[index][0]
            ].normalized_category_name,
            intersection_over_union=candidate_rows[index][2],
        )
        for index in sorted(selected_edge_indexes)
    )
    matched_prediction_ids = {match.prediction_id for match in matches}
    matched_reference_ids = {match.reference_id for match in matches}
    return ReferenceMatchResult(
        threshold=threshold,
        matches=matches,
        unmatched_prediction_ids=tuple(
            prediction.prediction_id
            for prediction in ordered_predictions
            if prediction.prediction_id not in matched_prediction_ids
        ),
        unmatched_reference_ids=tuple(
            reference.reference_id
            for reference in accepted
            if reference.reference_id not in matched_reference_ids
        ),
    )


def build_image_metric_primitive(
    *,
    admission: MetricAdmissionContract,
    merge_result: ImageMergeResult,
    references: Sequence[ReferenceObject],
    ledger_scope: ReferenceLedgerScope,
    threshold: float = PRIMARY_MATCH_INTERSECTION_OVER_UNION_THRESHOLD,
    configured_call_budget: int,
    density_tags: tuple[str, ...] = (),
    call_records: tuple[CallMetricRecord, ...],
    row_records: tuple[RowMetricRecord, ...],
) -> ImageMetricPrimitive:
    """Build one metric-bearing primitive from a fully admitted run universe."""

    admitted_image = admission.image(merge_result.image_id)
    admitted_requests = admission.image_arm_requests(
        image_id=merge_result.image_id,
        arm_identifier=merge_result.arm.arm_code,
    )
    if (
        merge_result.source_width != admitted_image.source_width
        or merge_result.source_height != admitted_image.source_height
    ):
        _fail(
            "merge result frame differs from the admitted source image",
            "analysis.metrics_admission_merge_frame",
            image_id=merge_result.image_id,
        )
    expected_call_budget = len(admitted_requests)
    if configured_call_budget != expected_call_budget:
        _fail(
            "configured call budget differs from the complete scheduled image-arm universe",
            "analysis.metrics_admission_call_budget",
            image_id=merge_result.image_id,
            arm_identifier=merge_result.arm.arm_code,
            expected_call_budget=expected_call_budget,
            observed_call_budget=configured_call_budget,
        )
    _validate_metric_call_admission(
        admitted_requests=admitted_requests,
        merge_result=merge_result,
        call_records=call_records,
        row_records=row_records,
    )
    sealed_references = admission.readiness.reference_objects(
        image_id=merge_result.image_id,
        ledger_scope=ledger_scope,
    )
    _validate_exact_sealed_reference_set(
        supplied=references,
        expected=sealed_references,
        image_id=merge_result.image_id,
        ledger_scope=ledger_scope,
    )
    scoped_references = _validate_and_derive_reference_spatial_contract(
        references=references,
        admitted_image=admitted_image,
    )
    return _build_image_metric_primitive(
        merge_result=merge_result,
        references=scoped_references,
        ledger_scope=ledger_scope,
        threshold=threshold,
        configured_call_budget=configured_call_budget,
        density_tags=density_tags,
        call_records=call_records,
        row_records=row_records,
        admission=admission,
        admitted_image=admitted_image,
    )


def _build_image_metric_primitive(
    *,
    merge_result: ImageMergeResult,
    references: Sequence[ReferenceObject],
    ledger_scope: ReferenceLedgerScope,
    threshold: float = PRIMARY_MATCH_INTERSECTION_OVER_UNION_THRESHOLD,
    configured_call_budget: int,
    density_tags: tuple[str, ...] = (),
    call_records: tuple[CallMetricRecord, ...],
    row_records: tuple[RowMetricRecord, ...],
    admission: MetricAdmissionContract | None = None,
    admitted_image: MetricAdmittedImage | None = None,
) -> ImageMetricPrimitive:
    """Internal formula helper; public callers must pass the admission gate."""

    scoped_references = tuple(references)
    if any(reference.ledger_scope != ledger_scope for reference in scoped_references):
        _fail(
            "official and audit-augmented ledgers must never be mixed",
            "analysis.metrics_ledger_mixed",
            image_id=merge_result.image_id,
        )
    if any(
        reference.image_id != merge_result.image_id for reference in scoped_references
    ):
        _fail(
            "reference image does not match merge result",
            "analysis.metrics_reference_image",
            image_id=merge_result.image_id,
        )
    accepted_ids = tuple(
        sorted(
            reference.reference_id
            for reference in scoped_references
            if reference.state == "accepted"
        )
    )
    core_interior_ids = tuple(
        sorted(
            reference.reference_id
            for reference in scoped_references
            if reference.state == "accepted" and reference.core_interior_for_mask_harm
        )
    )
    raw_owning = exact_reference_match(
        merge_result.raw_owning_call.predictions,
        scoped_references,
        threshold=threshold,
    )
    raw_union = exact_reference_match(
        merge_result.raw_any_call.predictions,
        scoped_references,
        threshold=threshold,
    )
    pre_merge = exact_reference_match(
        merge_result.pre_merge.predictions,
        scoped_references,
        threshold=threshold,
    )
    post_merge = exact_reference_match(
        merge_result.post_merge.predictions,
        scoped_references,
        threshold=threshold,
    )
    prediction_by_id = {
        prediction.prediction_id: prediction
        for prediction in merge_result.post_merge.predictions
    }
    crowd_regions = tuple(
        reference for reference in scoped_references if reference.state == "crowd"
    )
    uncertainty_regions = tuple(
        reference
        for reference in scoped_references
        if reference.state in {"ambiguous", "partial"}
        and reference.source_canvas_bbox_xyxy is not None
    )
    crowd_ignored: list[str] = []
    uncertainty_ignored: list[str] = []
    unmatched: list[str] = []
    for prediction_id in post_merge.unmatched_prediction_ids:
        prediction = prediction_by_id[prediction_id]
        if _prediction_overlaps_ignore_region(prediction, crowd_regions):
            crowd_ignored.append(prediction_id)
        elif _prediction_overlaps_ignore_region(prediction, uncertainty_regions):
            uncertainty_ignored.append(prediction_id)
        else:
            unmatched.append(prediction_id)
    non_owning_prediction_count = len(merge_result.non_owning_diagnostic.predictions)
    (
        derived_cell_records,
        malformed_row_count,
        invalid_row_count,
        attempted_row_count,
        attempted_call_count,
        natural_closure_count,
        controller_cap_count,
        token_cap_count,
        error_count,
        invalid_call_count,
        prompt_token_count,
        image_token_count,
        generated_token_count,
        wall_time_seconds,
        peak_device_memory_bytes,
    ) = _validate_and_derive_execution_records(
        merge_result=merge_result,
        references=scoped_references,
        threshold=threshold,
        configured_call_budget=configured_call_budget,
        call_records=call_records,
        row_records=row_records,
    )
    return ImageMetricPrimitive(
        image_id=merge_result.image_id,
        metric_admission_sha256=(
            admission.fingerprint if admission is not None else "0" * 64
        ),
        schedule_sha256=(
            admission.schedule_sha256 if admission is not None else "0" * 64
        ),
        cohort_sha256=(
            admission.cohort_sha256 if admission is not None else "0" * 64
        ),
        readiness_seal_sha256=(
            admission.readiness.readiness_seal_sha256
            if admission is not None
            else "0" * 64
        ),
        source_image_sha256=(
            admitted_image.source_image_sha256
            if admitted_image is not None
            else "0" * 64
        ),
        image_frozen_order=(
            admitted_image.image_frozen_order if admitted_image is not None else 0
        ),
        spatial_grid_spec_sha256=(
            admitted_image.spatial_grid.spec.fingerprint
            if admitted_image is not None
            else "0" * 64
        ),
        arm_identifier=merge_result.arm.arm_code,
        source_width=merge_result.source_width,
        source_height=merge_result.source_height,
        ledger_scope=ledger_scope,
        category_namespace_sha256=COCO_80_CATEGORY_NAMESPACE_SHA256,
        threshold=threshold,
        accepted_reference_ids=accepted_ids,
        core_interior_reference_ids=core_interior_ids,
        raw_owning_match=raw_owning,
        raw_any_call_union_match=raw_union,
        pre_merge_match=pre_merge,
        post_merge_match=post_merge,
        merge_created_reference_ids=tuple(
            sorted(
                set(post_merge.matched_reference_ids)
                - set(pre_merge.matched_reference_ids)
            )
        ),
        merge_destroyed_reference_ids=tuple(
            sorted(
                set(pre_merge.matched_reference_ids)
                - set(post_merge.matched_reference_ids)
            )
        ),
        crowd_ignored_prediction_ids=tuple(sorted(crowd_ignored)),
        uncertainty_ignored_prediction_ids=tuple(sorted(uncertainty_ignored)),
        unmatched_valid_prediction_ids=tuple(sorted(unmatched)),
        malformed_row_count=malformed_row_count,
        invalid_row_count=invalid_row_count,
        non_owning_prediction_count=non_owning_prediction_count,
        attempted_row_count=attempted_row_count,
        attempted_call_count=attempted_call_count,
        natural_closure_count=natural_closure_count,
        controller_cap_count=controller_cap_count,
        token_cap_count=token_cap_count,
        error_count=error_count,
        invalid_call_count=invalid_call_count,
        final_valid_prediction_count=len(merge_result.post_merge.predictions),
        pre_merge_strict_duplicate_components=(
            merge_result.pre_merge_strict_duplicate_components
        ),
        post_merge_strict_duplicate_components=(
            merge_result.post_merge_strict_duplicate_components
        ),
        configured_call_budget=configured_call_budget,
        realized_call_count=attempted_call_count,
        prompt_token_count=prompt_token_count,
        image_token_count=image_token_count,
        generated_token_count=generated_token_count,
        wall_time_seconds=wall_time_seconds,
        peak_device_memory_bytes=peak_device_memory_bytes,
        reference_object_count_stratum_value=len(accepted_ids),
        prompt_token_count_stratum_value=prompt_token_count,
        attempted_row_count_stratum_value=attempted_row_count,
        density_tags=tuple(sorted(set(density_tags))),
        cell_records=derived_cell_records,
        call_records=call_records,
        row_records=row_records,
    )


def aggregate_metric_report(
    candidate: Sequence[ImageMetricPrimitive],
    baseline: Sequence[ImageMetricPrimitive],
    *,
    scope: str,
    paired_full_bag: Sequence[ImageMetricPrimitive] | None = None,
) -> AggregateMetricReport:
    """Aggregate required rescue, transition, safety, and budget views."""

    candidate_by_image, baseline_by_image = _paired_primitive_maps(candidate, baseline)
    if not candidate_by_image:
        _fail("aggregate requires at least one paired image", "analysis.metrics_empty")
    first = next(iter(candidate_by_image.values()))
    _require_arm(candidate_by_image.values(), expected=None, role="candidate")
    _require_arm(baseline_by_image.values(), expected="FULL_SINGLE", role="baseline")
    _require_nonempty(scope, field="scope")
    bag_by_image: Mapping[str, ImageMetricPrimitive] | None = None
    if paired_full_bag is not None:
        bag_by_image, _ = _paired_primitive_maps(paired_full_bag, baseline)
        _require_arm(
            bag_by_image.values(), expected="FULL_BAG_K", role="paired comparator"
        )
    missed = 0
    rescued_post = 0
    rescued_raw_union = 0
    retained_denominator = 0
    retained_numerator = 0
    mask_harm_denominator = 0
    mask_harm_numerator = 0
    accepted_total = 0
    matched_total = 0
    precision_unmatched = 0
    uncertainty_ignored = 0
    merge_created = 0
    merge_destroyed = 0
    pre_duplicate_excess = 0
    pre_prediction_count = 0
    post_duplicate_excess = 0
    post_prediction_count = 0
    invalid_rows = 0
    invalid_calls = 0
    attempted_rows = 0
    natural_closures = 0
    attempted_calls = 0
    final_predictions = 0
    baseline_final_predictions = 0
    owning_seed_numerator = 0
    owning_seed_comparator_numerator = 0
    owning_seed_denominator = 0
    owning_seed_pairs: list[OwningSeedPairRecord] = []
    budgets = {
        "realized_call_count": 0.0,
        "configured_call_budget": 0.0,
        "prompt_token_count": 0.0,
        "image_token_count": 0.0,
        "generated_token_count": 0.0,
        "wall_time_seconds": 0.0,
        "peak_device_memory_bytes": 0.0,
    }
    for image_id in sorted(candidate_by_image):
        current = candidate_by_image[image_id]
        base = baseline_by_image[image_id]
        accepted = set(current.accepted_reference_ids)
        base_detected = set(base.post_merge_matched_reference_ids)
        current_post = set(current.post_merge_matched_reference_ids)
        current_union = set(current.raw_any_call_union_match.matched_reference_ids)
        missed_ids = accepted - base_detected
        missed += len(missed_ids)
        rescued_post += len(missed_ids & current_post)
        rescued_raw_union += len(missed_ids & current_union)
        retained_denominator += len(base_detected)
        retained_numerator += len(base_detected & current_post)
        core_baseline_detected = (
            set(current.core_interior_reference_ids) & base_detected
        )
        mask_harm_denominator += len(core_baseline_detected)
        mask_harm_numerator += len(core_baseline_detected & current_post)
        accepted_total += len(accepted)
        matched_total += len(current_post)
        precision_unmatched += len(current.unmatched_valid_prediction_ids)
        uncertainty_ignored += len(current.uncertainty_ignored_prediction_ids)
        merge_created += len(current.merge_created_reference_ids)
        merge_destroyed += len(current.merge_destroyed_reference_ids)
        pre_duplicate_excess += current.pre_merge_duplicate_excess_count
        pre_prediction_count += len(
            current.pre_merge_match.matched_prediction_ids
        ) + len(current.pre_merge_match.unmatched_prediction_ids)
        post_duplicate_excess += current.post_merge_duplicate_excess_count
        post_prediction_count += current.final_valid_prediction_count
        invalid_rows += current.malformed_row_count + current.invalid_row_count
        invalid_calls += current.invalid_call_count
        attempted_rows += current.attempted_row_count
        natural_closures += current.natural_closure_count
        attempted_calls += current.attempted_call_count
        final_predictions += current.final_valid_prediction_count
        baseline_final_predictions += base.final_valid_prediction_count
        for key in budgets:
            if key == "peak_device_memory_bytes":
                budgets[key] = max(budgets[key], float(getattr(current, key)))
            else:
                budgets[key] += float(getattr(current, key))
        if bag_by_image is not None:
            bag = bag_by_image[image_id]
            image_pairs = _validated_owning_seed_pairs(
                candidate=current,
                full_bag=bag,
                baseline_missed_reference_ids=tuple(sorted(missed_ids)),
            )
            owning_seed_pairs.extend(image_pairs)
            for pair in image_pairs:
                missed_owned = set(pair.baseline_missed_reference_ids)
                owning_seed_denominator += len(missed_owned)
                owning_seed_numerator += len(
                    missed_owned & set(pair.candidate_raw_owning_matched_reference_ids)
                )
                owning_seed_comparator_numerator += len(
                    missed_owned & set(pair.full_bag_raw_call_matched_reference_ids)
                )

    comparator = "FULL_SINGLE"
    metrics = [
        _metric(
            "post_merge_local_rescue_rate",
            rescued_post,
            missed,
            scope,
            first.threshold,
            comparator,
        ),
        _metric(
            "overall_retention",
            retained_numerator,
            retained_denominator,
            scope,
            first.threshold,
            comparator,
        ),
        _metric(
            "mask_harm_retention",
            mask_harm_numerator,
            mask_harm_denominator,
            scope,
            first.threshold,
            comparator,
        ),
        _metric(
            "raw_any_call_union_rescue_rate",
            rescued_raw_union,
            missed,
            scope,
            first.threshold,
            comparator,
        ),
        _metric(
            "merge_created_reference_match_rate",
            merge_created,
            accepted_total,
            scope,
            first.threshold,
            None,
        ),
        _metric(
            "merge_destroyed_reference_match_rate",
            merge_destroyed,
            accepted_total,
            scope,
            first.threshold,
            None,
        ),
        _metric(
            "manual_unique_recall",
            matched_total,
            accepted_total,
            scope,
            first.threshold,
            None,
        ),
        _metric(
            "manual_precision",
            matched_total,
            matched_total + precision_unmatched,
            scope,
            first.threshold,
            None,
        ),
        _metric(
            "manual_precision_uncertainty_ignored_as_unmatched",
            matched_total,
            matched_total + precision_unmatched + uncertainty_ignored,
            scope,
            first.threshold,
            None,
        ),
        _metric(
            "pre_merge_strict_duplicate_rate",
            pre_duplicate_excess,
            pre_prediction_count,
            scope,
            0.85,
            None,
        ),
        _metric(
            "post_merge_strict_duplicate_rate",
            post_duplicate_excess,
            post_prediction_count,
            scope,
            0.85,
            None,
        ),
        _metric("invalid_row_rate", invalid_rows, attempted_rows, scope, None, None),
        _metric("invalid_call_rate", invalid_calls, attempted_calls, scope, None, None),
        _metric(
            "natural_closure_rate", natural_closures, attempted_calls, scope, None, None
        ),
        _metric(
            "prediction_count_inflation",
            final_predictions,
            baseline_final_predictions,
            scope,
            None,
            comparator,
        ),
    ]
    if bag_by_image is not None:
        metrics.extend(
            (
                _metric(
                    "owning_seed_raw_rescue_rate",
                    owning_seed_numerator,
                    owning_seed_denominator,
                    scope,
                    first.threshold,
                    "FULL_BAG_K paired canonical cell",
                ),
                _metric(
                    "paired_full_image_raw_rescue_rate",
                    owning_seed_comparator_numerator,
                    owning_seed_denominator,
                    scope,
                    first.threshold,
                    first.arm_identifier,
                ),
                _difference_metric(
                    "owning_seed_raw_rescue_difference",
                    owning_seed_numerator,
                    owning_seed_comparator_numerator,
                    owning_seed_denominator,
                    scope,
                    first.threshold,
                    "FULL_BAG_K paired canonical cell",
                ),
            )
        )
    for key, value in budgets.items():
        metrics.append(_metric(f"realized_budget_{key}", value, 1.0, scope, None, None))
    return AggregateMetricReport(
        arm_identifier=first.arm_identifier,
        ledger_scope=first.ledger_scope,
        category_namespace_sha256=first.category_namespace_sha256,
        scope=scope,
        threshold=first.threshold,
        metrics=tuple(metrics),
        owning_seed_pair_records=tuple(owning_seed_pairs),
    )


def image_clustered_bootstrap(
    records: Sequence[BootstrapImageRecord],
    *,
    metric_name: str,
    estimator: BootstrapEstimator,
    scope: str,
    threshold: float | None,
    comparator: str | None,
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    minimum_applicable_replicates: int = (
        DEFAULT_MINIMUM_APPLICABLE_BOOTSTRAP_REPLICATES
    ),
    root_seed: int = PRIMARY_ROOT_SEED,
    synthetic_only: bool = False,
) -> BootstrapMetricReport:
    """Run a flexible synthetic-only bootstrap for unit and method tests.

    Metric-bearing research must use :func:`bootstrap_named_metric_report`,
    which derives contributions from validated image primitives and freezes the
    replicate/applicability contract.
    """

    if not synthetic_only:
        _fail(
            "arbitrary bootstrap records are synthetic-only",
            "analysis.metrics_arbitrary_bootstrap_evidence",
        )
    return _run_image_clustered_bootstrap(
        records,
        metric_name=metric_name,
        estimator=estimator,
        scope=scope,
        threshold=threshold,
        comparator=comparator,
        replicates=replicates,
        minimum_applicable_replicates=minimum_applicable_replicates,
        root_seed=root_seed,
        evidence_status="synthetic_only",
        retain_replicate_records=True,
    )


def bootstrap_named_metric_report(
    candidate: Sequence[ImageMetricPrimitive],
    baseline: Sequence[ImageMetricPrimitive],
    *,
    metric_name: str,
    scope: str,
    paired_full_bag: Sequence[ImageMetricPrimitive] | None = None,
    root_seed: int = PRIMARY_ROOT_SEED,
) -> BootstrapMetricReport:
    """Run the frozen metric-bearing bootstrap from validated image primitives."""

    records, estimator, threshold, comparator = _named_bootstrap_records(
        candidate,
        baseline,
        metric_name=metric_name,
        paired_full_bag=paired_full_bag,
    )
    return _run_image_clustered_bootstrap(
        records,
        metric_name=metric_name,
        estimator=estimator,
        scope=scope,
        threshold=threshold,
        comparator=comparator,
        replicates=DEFAULT_BOOTSTRAP_REPLICATES,
        minimum_applicable_replicates=(DEFAULT_MINIMUM_APPLICABLE_BOOTSTRAP_REPLICATES),
        root_seed=root_seed,
        evidence_status="metric_bearing",
        retain_replicate_records=False,
    )


def bootstrap_paired_arm_metric_report(
    candidate: Sequence[ImageMetricPrimitive],
    comparator: Sequence[ImageMetricPrimitive],
    full_single_baseline: Sequence[ImageMetricPrimitive],
    *,
    metric_name: str,
    scope: str,
    root_seed: int = PRIMARY_ROOT_SEED,
) -> BootstrapMetricReport:
    """Bootstrap one frozen candidate-minus-comparator arm difference.

    Post-merge Local Rescue Rate uses the same per-image set of accepted
    references missed by ``FULL_SINGLE`` for both arms.  The remaining names
    are the predeclared paired safety-guardrail differences.
    """

    candidate_by_image, baseline_by_image = _paired_primitive_maps(
        candidate, full_single_baseline
    )
    comparator_by_image, _ = _paired_primitive_maps(
        comparator, full_single_baseline
    )
    candidate_arm = _require_arm(
        candidate_by_image.values(), expected=None, role="paired candidate"
    )
    comparator_arm = _require_arm(
        comparator_by_image.values(), expected=None, role="paired comparator"
    )
    _require_arm(
        baseline_by_image.values(),
        expected="FULL_SINGLE",
        role="shared rescue baseline",
    )
    allowed_arm_pairs = {
        ("MASK_RESET", "FULL_BAG_K"),
        ("MASK_RESET", "MASK_CUMULATIVE"),
        ("MASK_CUMULATIVE", "MASK_RESET"),
        ("TILE_RESET", "MASK_RESET"),
    }
    if (candidate_arm, comparator_arm) not in allowed_arm_pairs:
        _fail(
            "paired bootstrap arm direction is not in the frozen registry",
            "analysis.metrics_paired_bootstrap_arm_pair",
            candidate_arm=candidate_arm,
            comparator_arm=comparator_arm,
        )
    allowed_metric_names = {
        "post_merge_local_rescue_rate_difference",
        "manual_precision_difference",
        "post_merge_strict_duplicate_rate_difference",
        "invalid_row_rate_difference",
        "invalid_call_rate_difference",
        "natural_closure_rate_difference",
    }
    if metric_name not in allowed_metric_names:
        _fail(
            "metric name is not a declared paired-arm estimand or guardrail",
            "analysis.metrics_paired_bootstrap_metric_name",
            metric_name=metric_name,
        )

    primitive_universe = (
        *candidate_by_image.values(),
        *comparator_by_image.values(),
        *baseline_by_image.values(),
    )
    if (
        len({primitive.ledger_scope for primitive in primitive_universe}) != 1
        or len({primitive.threshold for primitive in primitive_universe}) != 1
        or len(
            {
                primitive.category_namespace_sha256
                for primitive in primitive_universe
            }
        )
        != 1
    ):
        _fail(
            "paired bootstrap scope, threshold, or category namespace drifted",
            "analysis.metrics_paired_bootstrap_scope",
        )

    records: list[BootstrapImageRecord] = []
    for image_id in sorted(candidate_by_image):
        current = candidate_by_image[image_id]
        paired = comparator_by_image[image_id]
        baseline = baseline_by_image[image_id]
        if metric_name == "post_merge_local_rescue_rate_difference":
            baseline_missed_reference_ids = set(
                baseline.accepted_reference_ids
            ) - set(baseline.post_merge_matched_reference_ids)
            candidate_numerator = len(
                baseline_missed_reference_ids
                & set(current.post_merge_matched_reference_ids)
            )
            comparator_numerator = len(
                baseline_missed_reference_ids
                & set(paired.post_merge_matched_reference_ids)
            )
            candidate_denominator = len(baseline_missed_reference_ids)
            comparator_denominator = len(baseline_missed_reference_ids)
        elif metric_name == "manual_precision_difference":
            candidate_numerator = len(current.post_merge_matched_reference_ids)
            candidate_denominator = candidate_numerator + len(
                current.unmatched_valid_prediction_ids
            )
            comparator_numerator = len(paired.post_merge_matched_reference_ids)
            comparator_denominator = comparator_numerator + len(
                paired.unmatched_valid_prediction_ids
            )
        elif metric_name == "post_merge_strict_duplicate_rate_difference":
            candidate_numerator = current.post_merge_duplicate_excess_count
            candidate_denominator = current.final_valid_prediction_count
            comparator_numerator = paired.post_merge_duplicate_excess_count
            comparator_denominator = paired.final_valid_prediction_count
        elif metric_name == "invalid_row_rate_difference":
            candidate_numerator = current.malformed_row_count + current.invalid_row_count
            candidate_denominator = current.attempted_row_count
            comparator_numerator = paired.malformed_row_count + paired.invalid_row_count
            comparator_denominator = paired.attempted_row_count
        elif metric_name == "invalid_call_rate_difference":
            candidate_numerator = current.invalid_call_count
            candidate_denominator = current.attempted_call_count
            comparator_numerator = paired.invalid_call_count
            comparator_denominator = paired.attempted_call_count
        else:
            assert metric_name == "natural_closure_rate_difference"
            candidate_numerator = current.natural_closure_count
            candidate_denominator = current.attempted_call_count
            comparator_numerator = paired.natural_closure_count
            comparator_denominator = paired.attempted_call_count
        records.append(
            BootstrapImageRecord(
                image_id=image_id,
                candidate_numerator=float(candidate_numerator),
                candidate_denominator=float(candidate_denominator),
                comparator_numerator=float(comparator_numerator),
                comparator_denominator=float(comparator_denominator),
            )
        )

    threshold = next(iter(candidate_by_image.values())).threshold
    return _run_image_clustered_bootstrap(
        records,
        metric_name=metric_name,
        estimator="paired_rate_difference",
        scope=scope,
        threshold=threshold,
        comparator=comparator_arm,
        replicates=DEFAULT_BOOTSTRAP_REPLICATES,
        minimum_applicable_replicates=(DEFAULT_MINIMUM_APPLICABLE_BOOTSTRAP_REPLICATES),
        root_seed=root_seed,
        evidence_status="metric_bearing",
        retain_replicate_records=False,
    )


def _run_image_clustered_bootstrap(
    records: Sequence[BootstrapImageRecord],
    *,
    metric_name: str,
    estimator: BootstrapEstimator,
    scope: str,
    threshold: float | None,
    comparator: str | None,
    replicates: int,
    minimum_applicable_replicates: int,
    root_seed: int,
    evidence_status: BootstrapEvidenceStatus,
    retain_replicate_records: bool,
) -> BootstrapMetricReport:
    """Bootstrap paired image bundles without object- or call-level resampling."""

    if not records:
        _fail("bootstrap requires image records", "analysis.metrics_bootstrap_empty")
    _require_nonnegative_integer(replicates, field="replicates")
    _require_nonnegative_integer(
        minimum_applicable_replicates,
        field="minimum_applicable_replicates",
    )
    _require_nonnegative_integer(root_seed, field="root_seed")
    if replicates == 0 or minimum_applicable_replicates > replicates:
        _fail("invalid bootstrap replicate counts", "analysis.metrics_bootstrap_count")
    ordered = tuple(sorted(records, key=lambda record: record.image_id))
    if len({record.image_id for record in ordered}) != len(ordered):
        _fail(
            "bootstrap image identifiers must be unique",
            "analysis.metrics_bootstrap_image",
        )
    seed = derive_sampling_seed(
        root_seed=root_seed,
        role="image-bootstrap",
        image_id=0,
        cell_or_call_label=f"replicates-{replicates}",
    )
    (
        point_numerator,
        point_denominator,
        point_comparator_numerator,
        point_comparator_denominator,
        point_value,
    ) = _bootstrap_estimate(ordered, estimator)
    point = _metric(
        metric_name,
        point_numerator,
        point_denominator,
        scope,
        threshold,
        comparator,
        estimate_override=point_value,
        comparator_numerator=(
            point_comparator_numerator
            if estimator == "paired_rate_difference"
            else None
        ),
        comparator_denominator=(
            point_comparator_denominator
            if estimator == "paired_rate_difference"
            else None
        ),
    )
    generator = random.Random(seed)
    replicate_records: list[BootstrapReplicateRecord] = []
    applicable_values: list[float] = []
    for replicate_index in range(replicates):
        indexes = tuple(generator.randrange(len(ordered)) for _ in ordered)
        sampled = tuple(ordered[index] for index in indexes)
        (
            numerator,
            denominator,
            comparator_numerator,
            comparator_denominator,
            value,
        ) = _bootstrap_estimate(sampled, estimator)
        status: MetricStatus = "ok" if value is not None else "not_applicable"
        if value is not None:
            applicable_values.append(value)
        if retain_replicate_records:
            replicate_records.append(
                BootstrapReplicateRecord(
                    replicate_index=replicate_index,
                    sampled_image_indices=indexes,
                    numerator=numerator,
                    denominator=denominator,
                    comparator_numerator=(
                        comparator_numerator
                        if estimator == "paired_rate_difference"
                        else None
                    ),
                    comparator_denominator=(
                        comparator_denominator
                        if estimator == "paired_rate_difference"
                        else None
                    ),
                    estimate=value,
                    status=status,
                )
            )
    interval_valid = bool(applicable_values) and (
        len(applicable_values) >= minimum_applicable_replicates
    )
    lower = _percentile(applicable_values, 0.025) if interval_valid else None
    upper = _percentile(applicable_values, 0.975) if interval_valid else None
    return BootstrapMetricReport(
        metric_name=metric_name,
        category_namespace_sha256=COCO_80_CATEGORY_NAMESPACE_SHA256,
        estimator=estimator,
        point_estimate=point,
        lower_bound=lower,
        upper_bound=upper,
        applicable_replicates=len(applicable_values),
        total_replicates=replicates,
        minimum_applicable_replicates=minimum_applicable_replicates,
        confidence_level=0.95,
        sampling_seed=seed,
        bootstrap_algorithm="image_clustered_percentile_95_v1",
        evidence_status=evidence_status,
        records=tuple(replicate_records) if retain_replicate_records else (),
    )


@dataclass
class _FlowEdge:
    to_node: int
    reverse_index: int
    capacity: int
    cost: int
    candidate_index: int | None


def _exact_maximum_flow_assignment(
    *,
    prediction_count: int,
    reference_count: int,
    candidate_rows: Sequence[tuple[int, int, float]],
    benefits: Sequence[int],
) -> frozenset[int]:
    source = 0
    prediction_offset = 1
    reference_offset = prediction_offset + prediction_count
    sink = reference_offset + reference_count
    graph: list[list[_FlowEdge]] = [[] for _ in range(sink + 1)]

    def add_edge(
        source_node: int,
        target_node: int,
        capacity: int,
        cost: int,
        candidate_index: int | None = None,
    ) -> None:
        forward = _FlowEdge(
            target_node,
            len(graph[target_node]),
            capacity,
            cost,
            candidate_index,
        )
        backward = _FlowEdge(
            source_node,
            len(graph[source_node]),
            0,
            -cost,
            None,
        )
        graph[source_node].append(forward)
        graph[target_node].append(backward)

    for prediction_index in range(prediction_count):
        add_edge(source, prediction_offset + prediction_index, 1, 0)
    for reference_index in range(reference_count):
        add_edge(reference_offset + reference_index, sink, 1, 0)
    for candidate_index, ((prediction_index, reference_index, _), benefit) in enumerate(
        zip(candidate_rows, benefits, strict=True)
    ):
        add_edge(
            prediction_offset + prediction_index,
            reference_offset + reference_index,
            1,
            -benefit,
            candidate_index,
        )

    # Exact shortest-path potentials for the acyclic initial graph make all
    # residual reduced costs nonnegative. Subsequent Dijkstra passes preserve
    # exact integer objectives while avoiding a Bellman-Ford scan per match.
    potentials = [0] * len(graph)
    reference_has_candidate = [False] * reference_count
    for candidate_index, (_, reference_index, _) in enumerate(candidate_rows):
        reference_has_candidate[reference_index] = True
        reference_node = reference_offset + reference_index
        candidate_cost = -benefits[candidate_index]
        potentials[reference_node] = min(potentials[reference_node], candidate_cost)
    reachable_reference_potentials = [
        potentials[reference_offset + reference_index]
        for reference_index in range(reference_count)
        if reference_has_candidate[reference_index]
    ]
    potentials[sink] = min(reachable_reference_potentials)

    while True:
        distance: list[int | None] = [None] * len(graph)
        previous: list[tuple[int, int] | None] = [None] * len(graph)
        distance[source] = 0
        queue: list[tuple[int, int]] = [(0, source)]
        while queue:
            current_distance, node = heapq.heappop(queue)
            if current_distance != distance[node]:
                continue
            for edge_index, edge in enumerate(graph[node]):
                if edge.capacity <= 0:
                    continue
                reduced_cost = edge.cost + potentials[node] - potentials[edge.to_node]
                if reduced_cost < 0:
                    _fail(
                        "exact matcher produced a negative reduced cost",
                        "analysis.metrics_match_solver_invariant",
                    )
                candidate_distance = current_distance + reduced_cost
                if (
                    distance[edge.to_node] is None
                    or candidate_distance < distance[edge.to_node]
                ):
                    distance[edge.to_node] = candidate_distance
                    previous[edge.to_node] = (node, edge_index)
                    heapq.heappush(queue, (candidate_distance, edge.to_node))
        if previous[sink] is None:
            break
        for node, node_distance in enumerate(distance):
            if node_distance is not None:
                potentials[node] += node_distance
        node = sink
        while node != source:
            prior_node, edge_index = previous[node]  # type: ignore[misc]
            edge = graph[prior_node][edge_index]
            edge.capacity -= 1
            graph[node][edge.reverse_index].capacity += 1
            node = prior_node

    selected: set[int] = set()
    for prediction_index in range(prediction_count):
        node = prediction_offset + prediction_index
        for edge in graph[node]:
            if edge.candidate_index is not None and edge.capacity == 0:
                selected.add(edge.candidate_index)
    return frozenset(selected)


def _prediction_overlaps_ignore_region(
    prediction: NormalizedCallPrediction,
    references: Sequence[ReferenceObject],
) -> bool:
    prediction_area = _box_area(prediction.global_bbox_xyxy)
    for reference in references:
        if reference.source_canvas_bbox_xyxy is None:
            continue
        if reference.state in {"ambiguous", "partial"}:
            uncertainty_categories = reference.categories_for_uncertainty_ignore
            if uncertainty_categories and (
                prediction.normalized_category_name not in uncertainty_categories
            ):
                continue
        elif prediction.normalized_category_name != reference.normalized_category_name:
            continue
        overlap = _intersection_area(
            prediction.global_bbox_xyxy, reference.source_canvas_bbox_xyxy
        )
        if (
            overlap / prediction_area
            >= IGNORE_INTERSECTION_OVER_PREDICTION_AREA_THRESHOLD
        ):
            return True
    return False


def _validate_metric_call_admission(
    *,
    admitted_requests: tuple[MetricAdmittedRequest, ...],
    merge_result: ImageMergeResult,
    call_records: tuple[CallMetricRecord, ...],
    row_records: tuple[RowMetricRecord, ...],
) -> None:
    expected_by_id = {
        item.request.request_id: item for item in admitted_requests
    }
    observed_by_id = {record.canonical_call_id: record for record in call_records}
    if len(observed_by_id) != len(call_records):
        _fail(
            "metric call records repeat a scheduled request identity",
            "analysis.metrics_admission_duplicate_call",
            image_id=merge_result.image_id,
        )
    missing = sorted(set(expected_by_id) - set(observed_by_id))
    extra = sorted(set(observed_by_id) - set(expected_by_id))
    if missing or extra:
        _fail(
            "metric calls differ from the complete scheduled image-arm universe",
            "analysis.metrics_admission_call_universe",
            image_id=merge_result.image_id,
            arm_identifier=merge_result.arm.arm_code,
            missing_request_ids=missing,
            extra_request_ids=extra,
        )
    rows_by_call: dict[str, list[RowMetricRecord]] = {
        request_id: [] for request_id in expected_by_id
    }
    for row in row_records:
        if row.canonical_call_id not in rows_by_call:
            _fail(
                "metric row names a request outside the admitted image arm",
                "analysis.metrics_admission_row_call",
                canonical_call_id=row.canonical_call_id,
            )
        rows_by_call[row.canonical_call_id].append(row)
    prediction_ids_by_call: dict[str, list[str]] = {
        request_id: [] for request_id in expected_by_id
    }
    for prediction in merge_result.raw_any_call.predictions:
        if prediction.canonical_call_id not in prediction_ids_by_call:
            _fail(
                "merge prediction names a request outside the admitted image arm",
                "analysis.metrics_admission_prediction_call",
                prediction_id=prediction.prediction_id,
            )
        prediction_ids_by_call[prediction.canonical_call_id].append(
            prediction.prediction_id
        )
    for request_id, item in expected_by_id.items():
        request = item.request
        call = observed_by_id[request_id]
        if (
            call.canonical_cell_index != request.cell_index
            or call.sampling_seed != request.sampling_seed
            or call.attempt_status != item.attempt_status
            or call.source_image_sha256 != request.image_sha256
            or call.image_frozen_order != request.image_frozen_order
            or call.terminal_attempt_output_artifact_sha256
            != item.output_artifact_sha256
            or call.terminal_attempt_output_artifact_path != item.output_artifact_path
            or call.terminal_attempt_failure_code != item.failure_code
        ):
            _fail(
                "metric call provenance differs from its scheduled terminal request",
                "analysis.metrics_admission_call_provenance",
                request_id=request_id,
            )
        call_predictions = tuple(
            sorted(
                (
                    prediction
                    for prediction in merge_result.raw_any_call.predictions
                    if prediction.canonical_call_id == request_id
                ),
                key=lambda prediction: (
                    prediction.generated_row_index,
                    prediction.prediction_id,
                ),
            )
        )
        call_rows = tuple(
            sorted(
                rows_by_call[request_id],
                key=lambda row: (row.generated_row_index, row.prediction_id or ""),
            )
        )
        _validate_call_against_terminal_bundle(
            call=call,
            admitted_request=item,
            predictions=call_predictions,
            row_records=call_rows,
        )
        if item.attempt_status == "completed":
            if call.error_count != 0:
                _fail(
                    "completed attempt cannot be represented as an execution error",
                    "analysis.metrics_admission_completed_error",
                    request_id=request_id,
                )
            continue
        if (
            call.attempted_row_count != 0
            or call.malformed_row_count != 0
            or call.invalid_row_count != 0
            or call.non_owning_prediction_count != 0
            or call.valid_prediction_ids
            or call.raw_any_call_matched_reference_ids
            or call.raw_owning_call_matched_reference_ids
            or rows_by_call[request_id]
            or prediction_ids_by_call[request_id]
        ):
            _fail(
                "non-completed attempt cannot contribute generated rows or predictions",
                "analysis.metrics_admission_noncompleted_output",
                request_id=request_id,
                attempt_status=item.attempt_status,
            )
        if item.attempt_status in {"failed", "skipped", "invalid"}:
            if (
                call.error_count != 1
                or call.natural_closure_count != 0
                or call.controller_cap_count != 0
                or call.token_cap_count != 0
            ):
                _fail(
                    "failed, skipped, or invalid attempt requires one explicit error terminal",
                    "analysis.metrics_admission_failure_terminal",
                    request_id=request_id,
                    attempt_status=item.attempt_status,
                )
        elif item.attempt_status == "capped" and (
            call.error_count != 0
            or call.natural_closure_count != 0
            or call.controller_cap_count + call.token_cap_count != 1
        ):
            _fail(
                "capped attempt requires exactly one explicit controller or token cap",
                "analysis.metrics_admission_cap_terminal",
                request_id=request_id,
            )


def _validate_and_derive_reference_spatial_contract(
    *,
    references: Sequence[ReferenceObject],
    admitted_image: MetricAdmittedImage,
) -> tuple[ReferenceObject, ...]:
    result: list[ReferenceObject] = []
    for reference in references:
        if reference.image_id != admitted_image.image_id:
            _fail(
                "reference image differs from the admitted source image",
                "analysis.metrics_admission_reference_image",
                reference_id=reference.reference_id,
            )
        if reference.state != "accepted":
            expected_owner = None
            expected_core_interior = False
        else:
            assert reference.source_canvas_bbox_xyxy is not None
            ownership = admitted_image.spatial_grid.ownership(
                reference.source_canvas_bbox_xyxy
            )
            if not ownership.is_valid or ownership.owner_cell_index is None:
                _fail(
                    "accepted reference has no owner in the frozen spatial grid",
                    "analysis.metrics_admission_reference_owner",
                    reference_id=reference.reference_id,
                )
            expected_owner = ownership.owner_cell_index
            expected_core_interior = _is_mask_harm_core_interior(
                reference.source_canvas_bbox_xyxy,
                grid=admitted_image.spatial_grid,
                owner_cell_index=expected_owner,
            )
        if (
            reference.owner_cell_index != expected_owner
            or reference.core_interior_for_mask_harm != expected_core_interior
        ):
            _fail(
                "reference ownership or core-interior status differs from frozen geometry",
                "analysis.metrics_admission_reference_spatial_derivation",
                reference_id=reference.reference_id,
                expected_owner_cell_index=expected_owner,
                observed_owner_cell_index=reference.owner_cell_index,
                expected_core_interior_for_mask_harm=expected_core_interior,
                observed_core_interior_for_mask_harm=(
                    reference.core_interior_for_mask_harm
                ),
            )
        result.append(
            replace(
                reference,
                owner_cell_index=expected_owner,
                core_interior_for_mask_harm=expected_core_interior,
            )
        )
    return tuple(result)


def _is_mask_harm_core_interior(
    box: PixelBox,
    *,
    grid: SpatialGrid,
    owner_cell_index: int,
) -> bool:
    cell = grid.cell(owner_cell_index)
    left, top, right, bottom = cell.core_pixel_xyxy
    x1, y1, x2, y2 = box
    if not (left <= x1 < x2 <= right and top <= y1 < y2 <= bottom):
        return False
    quantum = grid.spec.visual_quantum_pixels
    if left > 0 and x1 - left < quantum:
        return False
    if right < grid.source_width and right - x2 < quantum:
        return False
    if top > 0 and y1 - top < quantum:
        return False
    if bottom < grid.source_height and bottom - y2 < quantum:
        return False
    return True


def _validate_and_derive_execution_records(
    *,
    merge_result: ImageMergeResult,
    references: Sequence[ReferenceObject],
    threshold: float,
    configured_call_budget: int,
    call_records: tuple[CallMetricRecord, ...],
    row_records: tuple[RowMetricRecord, ...],
) -> tuple[
    tuple[CellMetricRecord, ...],
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    float,
    int,
]:
    _require_nonnegative_integer(configured_call_budget, field="configured_call_budget")
    if not call_records:
        _fail(
            "metric primitive requires terminal records for every attempted call",
            "analysis.metrics_call_records_empty",
            image_id=merge_result.image_id,
        )
    calls_by_id = {record.canonical_call_id: record for record in call_records}
    if len(calls_by_id) != len(call_records):
        _fail(
            "call identifiers must be unique within an image arm",
            "analysis.metrics_call_identifier",
            image_id=merge_result.image_id,
        )
    if configured_call_budget < len(call_records):
        _fail(
            "realized calls exceed the configured call budget",
            "analysis.metrics_call_budget",
            image_id=merge_result.image_id,
        )
    predictions_by_call: dict[str, list[NormalizedCallPrediction]] = {
        call_id: [] for call_id in calls_by_id
    }
    for prediction in merge_result.raw_any_call.predictions:
        if prediction.canonical_call_id not in predictions_by_call:
            _fail(
                "raw prediction names a call absent from call records",
                "analysis.metrics_prediction_call",
                prediction_id=prediction.prediction_id,
            )
        predictions_by_call[prediction.canonical_call_id].append(prediction)
    rows_by_call: dict[str, list[RowMetricRecord]] = {
        call_id: [] for call_id in calls_by_id
    }
    for row in row_records:
        if row.canonical_call_id not in rows_by_call:
            _fail(
                "row names a call absent from call records",
                "analysis.metrics_row_call",
                canonical_call_id=row.canonical_call_id,
            )
        rows_by_call[row.canonical_call_id].append(row)

    cells: list[CellMetricRecord] = []
    observed_cells: set[int] = set()
    for call in call_records:
        terminal_count = (
            call.natural_closure_count
            + call.controller_cap_count
            + call.token_cap_count
            + call.error_count
        )
        if terminal_count != 1:
            _fail(
                "each attempted call requires exactly one terminal outcome",
                "analysis.metrics_terminal_outcome",
                canonical_call_id=call.canonical_call_id,
                terminal_count=terminal_count,
            )
        call_rows = rows_by_call[call.canonical_call_id]
        if len(call_rows) != call.attempted_row_count:
            _fail(
                "call attempted-row count disagrees with row records",
                "analysis.metrics_call_row_count",
                canonical_call_id=call.canonical_call_id,
            )
        row_indexes = [row.generated_row_index for row in call_rows]
        if len(set(row_indexes)) != len(row_indexes):
            _fail(
                "generated row indexes must be unique within a call",
                "analysis.metrics_row_index_reconciliation",
                canonical_call_id=call.canonical_call_id,
            )
        malformed = sum(row.parse_status == "malformed" for row in call_rows)
        invalid = sum(
            row.parse_status != "malformed" and row.validity_status == "invalid"
            for row in call_rows
        )
        non_owning = sum(row.ownership_status == "non_owning" for row in call_rows)
        if (
            malformed != call.malformed_row_count
            or invalid != call.invalid_row_count
            or non_owning != call.non_owning_prediction_count
        ):
            _fail(
                "call counters disagree with row records",
                "analysis.metrics_call_counter_reconciliation",
                canonical_call_id=call.canonical_call_id,
            )
        row_prediction_ids = tuple(
            sorted(
                row.prediction_id
                for row in call_rows
                if row.validity_status == "valid" and row.prediction_id is not None
            )
        )
        expected_prediction_ids = tuple(
            sorted(
                prediction.prediction_id
                for prediction in predictions_by_call[call.canonical_call_id]
            )
        )
        if row_prediction_ids != call.valid_prediction_ids or (
            expected_prediction_ids != call.valid_prediction_ids
        ):
            _fail(
                "call valid predictions disagree with row or merge records",
                "analysis.metrics_call_prediction_reconciliation",
                canonical_call_id=call.canonical_call_id,
            )
        expected_match_result = exact_reference_match(
            predictions_by_call[call.canonical_call_id],
            references,
            threshold=threshold,
        )
        expected_matches = expected_match_result.matched_reference_ids
        if expected_matches != call.raw_any_call_matched_reference_ids:
            _fail(
                "call reference matches disagree with exact raw-call matching",
                "analysis.metrics_call_match_reconciliation",
                canonical_call_id=call.canonical_call_id,
            )
        expected_raw_owning_matches = exact_reference_match(
            tuple(
                prediction
                for prediction in predictions_by_call[call.canonical_call_id]
                if prediction.ownership_status != "non_owning"
            ),
            references,
            threshold=threshold,
        ).matched_reference_ids
        if expected_raw_owning_matches != call.raw_owning_call_matched_reference_ids:
            _fail(
                "call raw-owning matches disagree with exact ownership-filtered matching",
                "analysis.metrics_call_raw_owning_match_reconciliation",
                canonical_call_id=call.canonical_call_id,
            )
        expected_row_matches: dict[str, tuple[str, ...]] = {
            prediction_id: () for prediction_id in expected_prediction_ids
        }
        for match in expected_match_result.matches:
            expected_row_matches[match.prediction_id] = (match.reference_id,)
        prediction_by_id = {
            prediction.prediction_id: prediction
            for prediction in predictions_by_call[call.canonical_call_id]
        }
        for row in call_rows:
            if row.validity_status != "valid":
                continue
            assert row.prediction_id is not None
            prediction = prediction_by_id[row.prediction_id]
            if (
                prediction.generated_row_index != row.generated_row_index
                or prediction.ownership_status != row.ownership_status
                or expected_row_matches[row.prediction_id] != row.matched_reference_ids
            ):
                _fail(
                    "row evidence disagrees with its exact prediction and match record",
                    "analysis.metrics_row_prediction_reconciliation",
                    canonical_call_id=call.canonical_call_id,
                    generated_row_index=row.generated_row_index,
                )
        if call.canonical_cell_index is not None:
            if call.canonical_cell_index in observed_cells:
                _fail(
                    "canonical cell has multiple call records",
                    "analysis.metrics_duplicate_cell_call",
                    canonical_cell_index=call.canonical_cell_index,
                )
            observed_cells.add(call.canonical_cell_index)
            owned_ids = tuple(
                sorted(
                    reference.reference_id
                    for reference in references
                    if reference.state == "accepted"
                    and reference.owner_cell_index == call.canonical_cell_index
                )
            )
            cells.append(
                CellMetricRecord(
                    canonical_cell_index=call.canonical_cell_index,
                    owned_reference_ids=owned_ids,
                    raw_owning_call_matched_reference_ids=(expected_raw_owning_matches),
                    canonical_call_id=call.canonical_call_id,
                )
            )

    malformed_total = sum(call.malformed_row_count for call in call_records)
    invalid_total = sum(call.invalid_row_count for call in call_records)
    non_owning_total = sum(call.non_owning_prediction_count for call in call_records)
    if non_owning_total != len(merge_result.non_owning_diagnostic.predictions):
        _fail(
            "non-owning counts disagree with merge records",
            "analysis.metrics_non_owning_reconciliation",
            image_id=merge_result.image_id,
        )
    return (
        tuple(sorted(cells, key=lambda record: record.canonical_cell_index)),
        malformed_total,
        invalid_total,
        len(row_records),
        len(call_records),
        sum(call.natural_closure_count for call in call_records),
        sum(call.controller_cap_count for call in call_records),
        sum(call.token_cap_count for call in call_records),
        sum(call.error_count for call in call_records),
        sum(
            call.attempt_status != "completed"
            or (call.malformed_row_count + call.invalid_row_count) > 0
            for call in call_records
        ),
        sum(call.prompt_token_count for call in call_records),
        sum(call.image_token_count for call in call_records),
        sum(call.generated_token_count for call in call_records),
        math.fsum(call.wall_time_seconds for call in call_records),
        max(call.peak_device_memory_bytes for call in call_records),
    )


def _paired_primitive_maps(
    candidate: Sequence[ImageMetricPrimitive],
    comparator: Sequence[ImageMetricPrimitive],
) -> tuple[dict[str, ImageMetricPrimitive], dict[str, ImageMetricPrimitive]]:
    candidate_map = {primitive.image_id: primitive for primitive in candidate}
    comparator_map = {primitive.image_id: primitive for primitive in comparator}
    if len(candidate_map) != len(candidate) or len(comparator_map) != len(comparator):
        _fail(
            "primitive image identifiers must be unique",
            "analysis.metrics_duplicate_image",
        )
    if set(candidate_map) != set(comparator_map):
        _fail("paired metric image universes differ", "analysis.metrics_image_universe")
    for image_id in candidate_map:
        left = candidate_map[image_id]
        right = comparator_map[image_id]
        if (
            left.metric_admission_sha256 != right.metric_admission_sha256
            or left.schedule_sha256 != right.schedule_sha256
            or left.cohort_sha256 != right.cohort_sha256
            or left.readiness_seal_sha256 != right.readiness_seal_sha256
            or left.source_image_sha256 != right.source_image_sha256
            or left.image_frozen_order != right.image_frozen_order
            or left.spatial_grid_spec_sha256 != right.spatial_grid_spec_sha256
            or left.source_width != right.source_width
            or left.source_height != right.source_height
            or left.ledger_scope != right.ledger_scope
            or left.category_namespace_sha256 != right.category_namespace_sha256
            or left.threshold != right.threshold
            or left.accepted_reference_ids != right.accepted_reference_ids
            or left.core_interior_reference_ids != right.core_interior_reference_ids
        ):
            _fail(
                "paired primitive reference contracts differ",
                "analysis.metrics_pair_contract",
                image_id=image_id,
            )
    return candidate_map, comparator_map


def _require_arm(
    primitives: Iterable[ImageMetricPrimitive],
    *,
    expected: str | None,
    role: str,
) -> str:
    arm_identifiers = {primitive.arm_identifier for primitive in primitives}
    if not arm_identifiers:
        _fail(
            "metric input arm is empty",
            "analysis.metrics_empty",
            role=role,
        )
    if len(arm_identifiers) != 1:
        _fail(
            "metric input mixes experimental arms",
            "analysis.metrics_mixed_arms",
            role=role,
            arm_identifiers=sorted(arm_identifiers),
        )
    arm_identifier = arm_identifiers.pop()
    if expected is not None and arm_identifier != expected:
        _fail(
            "metric input uses the wrong frozen comparator arm",
            "analysis.metrics_wrong_comparator_arm",
            role=role,
            expected=expected,
            observed=arm_identifier,
        )
    return arm_identifier


def _validated_owning_seed_pairs(
    *,
    candidate: ImageMetricPrimitive,
    full_bag: ImageMetricPrimitive,
    baseline_missed_reference_ids: tuple[str, ...],
) -> tuple[OwningSeedPairRecord, ...]:
    expected_cells = set(range(16))
    candidate_cells = {
        record.canonical_cell_index: record for record in candidate.cell_records
    }
    candidate_calls = {
        record.canonical_cell_index: record
        for record in candidate.call_records
        if record.canonical_cell_index is not None
    }
    bag_calls = {
        record.canonical_cell_index: record
        for record in full_bag.call_records
        if record.canonical_cell_index is not None
    }
    for name, observed in (
        ("candidate cells", set(candidate_cells)),
        ("candidate calls", set(candidate_calls)),
        ("FULL_BAG_K calls", set(bag_calls)),
    ):
        if observed != expected_cells:
            _fail(
                "owning-seed evidence requires exactly all sixteen canonical cells",
                "analysis.metrics_owning_seed_cells",
                image_id=candidate.image_id,
                record_family=name,
                missing=sorted(expected_cells - observed),
                extra=sorted(observed - expected_cells),
            )
    owner_occurrences: dict[str, list[int]] = {}
    candidate_raw_matches: set[str] = set()
    bag_raw_matches: set[str] = set()
    pair_records: list[OwningSeedPairRecord] = []
    missed = set(baseline_missed_reference_ids)
    for cell_index in range(16):
        cell = candidate_cells[cell_index]
        candidate_call = candidate_calls[cell_index]
        bag_call = bag_calls[cell_index]
        if cell.canonical_call_id != candidate_call.canonical_call_id:
            _fail(
                "cell evidence does not bind its exact candidate request",
                "analysis.metrics_owning_seed_request",
                image_id=candidate.image_id,
                canonical_cell_index=cell_index,
            )
        if candidate_call.sampling_seed != bag_call.sampling_seed:
            _fail(
                "paired canonical-cell calls require the same sampling seed",
                "analysis.metrics_owning_seed_seed",
                image_id=candidate.image_id,
                canonical_cell_index=cell_index,
                candidate_seed=candidate_call.sampling_seed,
                full_bag_seed=bag_call.sampling_seed,
            )
        if (
            cell.raw_owning_call_matched_reference_ids
            != candidate_call.raw_owning_call_matched_reference_ids
        ):
            _fail(
                "cell matches do not equal exact candidate raw-owning-call matches",
                "analysis.metrics_owning_seed_candidate_match",
                image_id=candidate.image_id,
                canonical_cell_index=cell_index,
            )
        for reference_id in cell.owned_reference_ids:
            owner_occurrences.setdefault(reference_id, []).append(cell_index)
        candidate_raw_matches.update(cell.raw_owning_call_matched_reference_ids)
        bag_raw_matches.update(bag_call.raw_any_call_matched_reference_ids)
        owned = set(cell.owned_reference_ids)
        pair_records.append(
            OwningSeedPairRecord(
                image_id=candidate.image_id,
                canonical_cell_index=cell_index,
                candidate_request_id=candidate_call.canonical_call_id,
                full_bag_request_id=bag_call.canonical_call_id,
                sampling_seed=candidate_call.sampling_seed,
                owned_reference_ids=cell.owned_reference_ids,
                baseline_missed_reference_ids=tuple(sorted(missed & owned)),
                candidate_raw_owning_matched_reference_ids=tuple(
                    sorted(set(cell.raw_owning_call_matched_reference_ids) & owned)
                ),
                full_bag_raw_call_matched_reference_ids=tuple(
                    sorted(set(bag_call.raw_any_call_matched_reference_ids) & owned)
                ),
            )
        )
    accepted = set(candidate.accepted_reference_ids)
    if set(owner_occurrences) != accepted or any(
        len(cells) != 1 for cells in owner_occurrences.values()
    ):
        _fail(
            "every accepted reference requires exactly one canonical-cell owner",
            "analysis.metrics_reference_owner_partition",
            image_id=candidate.image_id,
            missing=sorted(accepted - set(owner_occurrences)),
            extra=sorted(set(owner_occurrences) - accepted),
            duplicates={
                key: value
                for key, value in owner_occurrences.items()
                if len(value) != 1
            },
        )
    if candidate_raw_matches != set(candidate.raw_owning_match.matched_reference_ids):
        _fail(
            "candidate cell matches do not reconcile to raw-owning matches",
            "analysis.metrics_candidate_raw_owning_reconciliation",
            image_id=candidate.image_id,
        )
    if bag_raw_matches != set(full_bag.raw_owning_match.matched_reference_ids):
        _fail(
            "FULL_BAG_K cell matches do not reconcile to raw-call union matches",
            "analysis.metrics_full_bag_raw_reconciliation",
            image_id=candidate.image_id,
        )
    if sum(len(record.baseline_missed_reference_ids) for record in pair_records) != len(
        missed
    ):
        _fail(
            "owning-seed denominator lost baseline-missed references",
            "analysis.metrics_owning_seed_denominator",
            image_id=candidate.image_id,
        )
    return tuple(pair_records)


def _named_bootstrap_records(
    candidate: Sequence[ImageMetricPrimitive],
    baseline: Sequence[ImageMetricPrimitive],
    *,
    metric_name: str,
    paired_full_bag: Sequence[ImageMetricPrimitive] | None,
) -> tuple[
    tuple[BootstrapImageRecord, ...],
    BootstrapEstimator,
    float | None,
    str | None,
]:
    candidate_by_image, baseline_by_image = _paired_primitive_maps(candidate, baseline)
    _require_arm(candidate_by_image.values(), expected=None, role="candidate")
    _require_arm(baseline_by_image.values(), expected="FULL_SINGLE", role="baseline")
    bag_by_image: Mapping[str, ImageMetricPrimitive] | None = None
    if paired_full_bag is not None:
        bag_by_image, _ = _paired_primitive_maps(paired_full_bag, baseline)
        _require_arm(
            bag_by_image.values(), expected="FULL_BAG_K", role="paired comparator"
        )
    difference_names = {
        "manual_precision_difference",
        "post_merge_strict_duplicate_rate_difference",
        "invalid_row_rate_difference",
        "invalid_call_rate_difference",
        "natural_closure_rate_difference",
        "owning_seed_raw_rescue_difference",
    }
    if metric_name in difference_names and bag_by_image is None:
        _fail(
            "named paired-difference bootstrap requires FULL_BAG_K primitives",
            "analysis.metrics_bootstrap_comparator_missing",
            metric_name=metric_name,
        )
    records: list[BootstrapImageRecord] = []
    threshold = next(iter(candidate_by_image.values())).threshold
    estimator: BootstrapEstimator = "rate"
    comparator: str | None = None
    for image_id in sorted(candidate_by_image):
        current = candidate_by_image[image_id]
        base = baseline_by_image[image_id]
        bag = bag_by_image[image_id] if bag_by_image is not None else None
        accepted = set(current.accepted_reference_ids)
        base_detected = set(base.post_merge_matched_reference_ids)
        current_detected = set(current.post_merge_matched_reference_ids)
        missed = accepted - base_detected
        candidate_numerator = 0.0
        candidate_denominator = 0.0
        comparator_numerator = 0.0
        comparator_denominator = 0.0
        if metric_name == "post_merge_local_rescue_rate":
            candidate_numerator = len(missed & current_detected)
            candidate_denominator = len(missed)
            comparator = "FULL_SINGLE"
        elif metric_name == "raw_any_call_union_rescue_rate":
            candidate_numerator = len(
                missed & set(current.raw_any_call_union_match.matched_reference_ids)
            )
            candidate_denominator = len(missed)
            comparator = "FULL_SINGLE"
        elif metric_name == "overall_retention":
            candidate_numerator = len(base_detected & current_detected)
            candidate_denominator = len(base_detected)
            comparator = "FULL_SINGLE"
        elif metric_name == "mask_harm_retention":
            eligible = base_detected & set(current.core_interior_reference_ids)
            candidate_numerator = len(eligible & current_detected)
            candidate_denominator = len(eligible)
            comparator = "FULL_SINGLE"
        elif metric_name in {"manual_precision", "manual_precision_difference"}:
            candidate_numerator = len(current_detected)
            candidate_denominator = len(current_detected) + len(
                current.unmatched_valid_prediction_ids
            )
            if metric_name.endswith("_difference"):
                assert bag is not None
                comparator_detected = set(bag.post_merge_matched_reference_ids)
                comparator_numerator = len(comparator_detected)
                comparator_denominator = len(comparator_detected) + len(
                    bag.unmatched_valid_prediction_ids
                )
        elif metric_name == "manual_precision_uncertainty_ignored_as_unmatched":
            candidate_numerator = len(current_detected)
            candidate_denominator = (
                len(current_detected)
                + len(current.unmatched_valid_prediction_ids)
                + len(current.uncertainty_ignored_prediction_ids)
            )
        elif metric_name in {
            "post_merge_strict_duplicate_rate",
            "post_merge_strict_duplicate_rate_difference",
        }:
            candidate_numerator = current.post_merge_duplicate_excess_count
            candidate_denominator = current.final_valid_prediction_count
            if metric_name.endswith("_difference"):
                assert bag is not None
                comparator_numerator = bag.post_merge_duplicate_excess_count
                comparator_denominator = bag.final_valid_prediction_count
        elif metric_name in {"invalid_row_rate", "invalid_row_rate_difference"}:
            candidate_numerator = (
                current.malformed_row_count + current.invalid_row_count
            )
            candidate_denominator = current.attempted_row_count
            if metric_name.endswith("_difference"):
                assert bag is not None
                comparator_numerator = bag.malformed_row_count + bag.invalid_row_count
                comparator_denominator = bag.attempted_row_count
        elif metric_name in {"invalid_call_rate", "invalid_call_rate_difference"}:
            candidate_numerator = current.invalid_call_count
            candidate_denominator = current.attempted_call_count
            if metric_name.endswith("_difference"):
                assert bag is not None
                comparator_numerator = bag.invalid_call_count
                comparator_denominator = bag.attempted_call_count
        elif metric_name in {
            "natural_closure_rate",
            "natural_closure_rate_difference",
        }:
            candidate_numerator = current.natural_closure_count
            candidate_denominator = current.attempted_call_count
            if metric_name.endswith("_difference"):
                assert bag is not None
                comparator_numerator = bag.natural_closure_count
                comparator_denominator = bag.attempted_call_count
        elif metric_name == "prediction_count_inflation":
            estimator = "count_ratio"
            candidate_numerator = current.final_valid_prediction_count
            candidate_denominator = 1
            comparator_numerator = base.final_valid_prediction_count
            comparator_denominator = 1
            comparator = "FULL_SINGLE"
        elif metric_name == "owning_seed_raw_rescue_difference":
            assert bag is not None
            estimator = "paired_rate_difference"
            pairs = _validated_owning_seed_pairs(
                candidate=current,
                full_bag=bag,
                baseline_missed_reference_ids=tuple(sorted(missed)),
            )
            candidate_numerator = sum(
                len(
                    set(pair.baseline_missed_reference_ids)
                    & set(pair.candidate_raw_owning_matched_reference_ids)
                )
                for pair in pairs
            )
            comparator_numerator = sum(
                len(
                    set(pair.baseline_missed_reference_ids)
                    & set(pair.full_bag_raw_call_matched_reference_ids)
                )
                for pair in pairs
            )
            candidate_denominator = len(missed)
            comparator_denominator = len(missed)
        else:
            _fail(
                "metric name is not in the frozen bootstrap registry",
                "analysis.metrics_bootstrap_metric_name",
                metric_name=metric_name,
            )
        if metric_name in difference_names:
            estimator = "paired_rate_difference"
            comparator = "FULL_BAG_K"
        records.append(
            BootstrapImageRecord(
                image_id=image_id,
                candidate_numerator=float(candidate_numerator),
                candidate_denominator=float(candidate_denominator),
                comparator_numerator=float(comparator_numerator),
                comparator_denominator=float(comparator_denominator),
            )
        )
    return tuple(records), estimator, threshold, comparator


def _metric(
    name: str,
    numerator: float,
    denominator: float,
    scope: str,
    threshold: float | None,
    comparator: str | None,
    *,
    estimate_override: float | None = None,
    comparator_numerator: float | None = None,
    comparator_denominator: float | None = None,
) -> MetricEstimate:
    if denominator == 0:
        return MetricEstimate(
            name,
            numerator,
            denominator,
            None,
            "not_applicable",
            scope,
            threshold,
            comparator,
            comparator_numerator,
            comparator_denominator,
        )
    estimate = (
        numerator / denominator if estimate_override is None else estimate_override
    )
    return MetricEstimate(
        name,
        numerator,
        denominator,
        estimate,
        "ok",
        scope,
        threshold,
        comparator,
        comparator_numerator,
        comparator_denominator,
    )


def _difference_metric(
    name: str,
    candidate_numerator: float,
    comparator_numerator: float,
    denominator: float,
    scope: str,
    threshold: float | None,
    comparator: str,
) -> MetricEstimate:
    if denominator == 0:
        return MetricEstimate(
            name,
            candidate_numerator,
            0,
            None,
            "not_applicable",
            scope,
            threshold,
            comparator,
            comparator_numerator,
            0,
        )
    return MetricEstimate(
        name,
        candidate_numerator,
        denominator,
        (candidate_numerator - comparator_numerator) / denominator,
        "ok",
        scope,
        threshold,
        comparator,
        comparator_numerator,
        denominator,
    )


def _bootstrap_estimate(
    records: Sequence[BootstrapImageRecord], estimator: BootstrapEstimator
) -> tuple[float, float, float, float, float | None]:
    candidate_numerator = math.fsum(item.candidate_numerator for item in records)
    candidate_denominator = math.fsum(item.candidate_denominator for item in records)
    comparator_numerator = math.fsum(item.comparator_numerator for item in records)
    comparator_denominator = math.fsum(item.comparator_denominator for item in records)
    if estimator == "rate":
        if candidate_denominator == 0:
            return candidate_numerator, candidate_denominator, 0.0, 0.0, None
        return (
            candidate_numerator,
            candidate_denominator,
            0.0,
            0.0,
            candidate_numerator / candidate_denominator,
        )
    if estimator == "paired_rate_difference":
        if candidate_denominator == 0 or comparator_denominator == 0:
            return (
                candidate_numerator,
                candidate_denominator,
                comparator_numerator,
                comparator_denominator,
                None,
            )
        value = (
            candidate_numerator / candidate_denominator
            - comparator_numerator / comparator_denominator
        )
        return (
            candidate_numerator,
            candidate_denominator,
            comparator_numerator,
            comparator_denominator,
            value,
        )
    if estimator == "count_ratio":
        if comparator_numerator == 0:
            return candidate_numerator, comparator_numerator, 0.0, 0.0, None
        return (
            candidate_numerator,
            comparator_numerator,
            0.0,
            0.0,
            candidate_numerator / comparator_numerator,
        )
    _fail("unknown bootstrap estimator", "analysis.metrics_bootstrap_estimator")


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    weight = position - lower_index
    return ordered[lower_index] * (1.0 - weight) + ordered[upper_index] * weight


def _validate_prediction_reference_universe(
    predictions: Sequence[NormalizedCallPrediction],
    references: Sequence[ReferenceObject],
) -> None:
    if len({item.prediction_id for item in predictions}) != len(predictions):
        _fail("prediction identifiers must be unique", "analysis.metrics_prediction_id")
    if len({item.reference_id for item in references}) != len(references):
        _fail("reference identifiers must be unique", "analysis.metrics_reference_id")
    image_ids = {item.image_id for item in predictions} | {
        item.image_id for item in references
    }
    if len(image_ids) > 1:
        _fail("matching is image-local", "analysis.metrics_match_image")
    scopes = {item.ledger_scope for item in references}
    if len(scopes) > 1:
        _fail("matching cannot mix ledgers", "analysis.metrics_ledger_mixed")


def _box_area(box: PixelBox) -> float:
    return (box[2] - box[0]) * (box[3] - box[1])


def _intersection_area(first: PixelBox, second: PixelBox) -> float:
    width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    return width * height


def _validate_box(box: PixelBox, *, field: str) -> None:
    if len(box) != 4 or any(not math.isfinite(float(value)) for value in box):
        _fail(
            "box must contain four finite coordinates",
            "analysis.metrics_box",
            field=field,
        )
    if box[0] >= box[2] or box[1] >= box[3]:
        _fail("box must have positive area", "analysis.metrics_box_area", field=field)


def _validate_sorted_unique(values: tuple[str, ...], field: str) -> None:
    if tuple(sorted(set(values))) != values:
        _fail(
            "identifier tuple must be sorted and unique",
            "analysis.metrics_sorted_unique",
            field=field,
        )


def _validate_call_against_terminal_bundle(
    *,
    call: CallMetricRecord,
    admitted_request: MetricAdmittedRequest,
    predictions: Sequence[NormalizedCallPrediction] = (),
    row_records: Sequence[RowMetricRecord] = (),
) -> None:
    path = call.terminal_attempt_output_artifact_path
    digest = call.terminal_attempt_output_artifact_sha256
    if path is None or digest is None:
        _fail(
            "metric call is missing its canonical terminal bundle",
            "analysis.metrics_admission_call_terminal_bundle",
            canonical_call_id=call.canonical_call_id,
        )
    bundle = TerminalCallOutputBundle.from_path(path)
    if bundle.bundle_sha256 != digest or digest != admitted_request.output_artifact_sha256:
        _fail(
            "metric call and terminal attempt do not bind the same bundle",
            "analysis.metrics_admission_call_terminal_bundle_digest",
            canonical_call_id=call.canonical_call_id,
        )
    payload = bundle.payload
    expected_diagnostics = {
        "attempted_row_count": call.attempted_row_count,
        "controller_cap_count": call.controller_cap_count,
        "error_count": call.error_count,
        "generated_token_count": call.generated_token_count,
        "image_token_count": call.image_token_count,
        "invalid_row_count": call.invalid_row_count,
        "malformed_row_count": call.malformed_row_count,
        "natural_closure_count": call.natural_closure_count,
        "non_owning_prediction_count": call.non_owning_prediction_count,
        "peak_device_memory_bytes": call.peak_device_memory_bytes,
        "prompt_token_count": call.prompt_token_count,
        "token_cap_count": call.token_cap_count,
        "valid_prediction_ids": list(call.valid_prediction_ids),
        "wall_time_seconds": call.wall_time_seconds,
    }
    if payload.get("call_diagnostics") != expected_diagnostics:
        _fail(
            "metric call diagnostics do not reconstruct from terminal bundle",
            "analysis.metrics_admission_call_terminal_diagnostics",
            canonical_call_id=call.canonical_call_id,
        )
    expected_parse_receipts = [
        prediction.parse_score_receipt.to_artifact_dict()
        for prediction in predictions
    ]
    if payload.get("parse_score_receipts") != expected_parse_receipts:
        _fail(
            "metric predictions do not reconstruct terminal parse receipts",
            "analysis.metrics_admission_call_terminal_parse_receipts",
            canonical_call_id=call.canonical_call_id,
        )
    terminal_row_diagnostics = payload.get("row_diagnostics")
    if not isinstance(terminal_row_diagnostics, list) or any(
        not isinstance(row, dict) or row.get("matched_reference_ids") != []
        for row in terminal_row_diagnostics
    ):
        _fail(
            "terminal row diagnostics must remain pre-metric and unmatched",
            "analysis.metrics_admission_call_terminal_row_diagnostics",
            canonical_call_id=call.canonical_call_id,
        )
    expected_row_diagnostics = [
        replace(row, matched_reference_ids=()).to_json_dict() for row in row_records
    ]
    if terminal_row_diagnostics != expected_row_diagnostics:
        _fail(
            "metric rows do not reconstruct terminal row diagnostics",
            "analysis.metrics_admission_call_terminal_row_diagnostics",
            canonical_call_id=call.canonical_call_id,
        )


def _validate_exact_sealed_reference_set(
    *,
    supplied: Sequence[ReferenceObject],
    expected: Sequence[ReferenceObject],
    image_id: str,
    ledger_scope: ReferenceLedgerScope,
) -> None:
    supplied_identities = [_reference_source_identity(item) for item in supplied]
    expected_identities = [_reference_source_identity(item) for item in expected]
    supplied_ids = [item.reference_id for item in supplied]
    if len(supplied_ids) != len(set(supplied_ids)):
        _fail(
            "metric references repeat a sealed ledger identifier",
            "analysis.metrics_admission_reference_duplicate",
            image_id=image_id,
            ledger_scope=ledger_scope,
        )
    if sorted(supplied_identities, key=canonical_json_text) != sorted(
        expected_identities,
        key=canonical_json_text,
    ):
        _fail(
            "metric references differ from the sealed ledger content",
            "analysis.metrics_admission_reference_set",
            image_id=image_id,
            ledger_scope=ledger_scope,
            expected_reference_ids=sorted(item.reference_id for item in expected),
            supplied_reference_ids=sorted(supplied_ids),
        )


def _reference_source_identity(reference: ReferenceObject) -> dict[str, Any]:
    return {
        "candidate_category_names": list(reference.candidate_category_names),
        "category_namespace_sha256": reference.category_namespace_sha256,
        "evaluator_category_id": reference.evaluator_category_id,
        "image_id": reference.image_id,
        "ledger_scope": reference.ledger_scope,
        "normalized_category_name": reference.normalized_category_name,
        "official_coco_category_id": reference.official_coco_category_id,
        "provenance": reference.provenance,
        "reference_id": reference.reference_id,
        "source_canvas_bbox_xyxy": (
            None
            if reference.source_canvas_bbox_xyxy is None
            else list(reference.source_canvas_bbox_xyxy)
        ),
        "state": reference.state,
    }


def _reference_objects_from_readiness(
    *,
    readiness: MetricReadinessIdentity,
    image_id: str,
    ledger_scope: ReferenceLedgerScope,
) -> tuple[ReferenceObject, ...]:
    root = Path(readiness.readiness_root)
    cohort = CohortLedger.from_jsonl_bytes(
        (root / readiness.cohort_artifact_name).read_bytes()
    )
    try:
        cohort_image = next(
            record for record in cohort.records if str(record.image_id) == image_id
        )
    except StopIteration:
        _fail(
            "metric image is absent from the sealed cohort",
            "analysis.metrics_admission_reference_image",
            image_id=image_id,
        )
    if ledger_scope == "audit_augmented":
        rows = _read_canonical_jsonl(root / "audit-augmented-ledger.jsonl")
        adjudication_rows = _adjudication_rows_by_identifier(
            _read_canonical_jsonl(root / "adjudication.jsonl")
        )
        references = tuple(
            _audit_reference_from_row(
                row=row,
                adjudication_row=_joined_adjudication_row(
                    audit_row=row,
                    adjudication_rows=adjudication_rows,
                ),
                cohort_image=cohort_image,
            )
            for row in rows
            if str(row.get("image_id")) == image_id
        )
    elif ledger_scope == "official_annotation":
        individual_rows = _read_canonical_jsonl(
            root / "official-individual-ledger.jsonl"
        )
        crowd_rows = _read_canonical_jsonl(
            root / "official-crowd-ignore-ledger.jsonl"
        )
        references = tuple(
            _official_reference_from_row(
                row=row,
                cohort_image=cohort_image,
                state="accepted",
            )
            for row in individual_rows
            if str(row.get("image_id")) == image_id
        ) + tuple(
            _official_reference_from_row(
                row=row,
                cohort_image=cohort_image,
                state="crowd",
            )
            for row in crowd_rows
            if str(row.get("image_id")) == image_id
        )
    else:
        _fail("unknown reference-ledger scope", "analysis.metrics_ledger_scope")
    identifiers = [reference.reference_id for reference in references]
    if len(identifiers) != len(set(identifiers)):
        _fail(
            "sealed reference ledgers repeat an object identifier",
            "analysis.metrics_readiness_reference_duplicate",
            image_id=image_id,
            ledger_scope=ledger_scope,
        )
    return tuple(sorted(references, key=lambda reference: reference.reference_id))


def _read_canonical_jsonl(path: Path) -> tuple[Mapping[str, Any], ...]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise DataContractError(
            "sealed reference ledger is unreadable",
            code="analysis.metrics_readiness_reference_ledger",
            context={"path": str(path)},
            cause=exc,
        ) from exc
    rows: list[Mapping[str, Any]] = []
    for line_number, raw_line in enumerate(raw.splitlines(), start=1):
        if not raw_line:
            _fail(
                "sealed reference ledger contains a blank row",
                "analysis.metrics_readiness_reference_jsonl",
                path=str(path),
                line_number=line_number,
            )
        try:
            row = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DataContractError(
                "sealed reference ledger row is invalid JSON",
                code="analysis.metrics_readiness_reference_jsonl",
                context={"path": str(path), "line_number": line_number},
                cause=exc,
            ) from exc
        if not isinstance(row, dict) or canonical_json_text(row).encode("utf-8") != raw_line:
            _fail(
                "sealed reference ledger row is not canonical",
                "analysis.metrics_readiness_reference_jsonl",
                path=str(path),
                line_number=line_number,
            )
        rows.append(row)
    return tuple(rows)


def _audit_reference_from_row(
    *,
    row: Mapping[str, Any],
    adjudication_row: Mapping[str, Any],
    cohort_image: Any,
) -> ReferenceObject:
    if row.get("image_sha256") != cohort_image.image_sha256:
        _fail(
            "audit reference image digest differs from sealed cohort",
            "analysis.metrics_readiness_reference_image_digest",
            reference_id=row.get("object_identifier"),
        )
    state = _normalize_audit_reference_state(row.get("final_state"))
    adjudication_state = _normalize_audit_reference_state(
        adjudication_row.get("final_state")
    )
    if adjudication_state != state:
        _fail(
            "audit reference state differs from its sealed adjudication decision",
            "analysis.metrics_readiness_adjudication_state",
            reference_id=row.get("object_identifier"),
        )
    candidate_category_names = _candidate_category_names_from_adjudication(
        adjudication_row
    )
    return ReferenceObject(
        image_id=str(row.get("image_id")),
        ledger_scope="audit_augmented",
        reference_id=row.get("object_identifier"),
        normalized_category_name=row.get("normalized_category_name"),
        evaluator_category_id=row.get("evaluator_local_category_id"),
        official_coco_category_id=row.get("official_coco_category_id"),
        category_namespace_sha256=row.get("category_namespace_sha256"),
        source_canvas_bbox_xyxy=_optional_reference_box(
            row.get("source_canvas_box_xyxy")
        ),
        state=state,
        provenance=row.get("provenance"),
        candidate_category_names=candidate_category_names,
    )


def _normalize_audit_reference_state(value: Any) -> ReferenceState:
    if value == "out-of-scope":
        return "out_of_scope"
    if value in {"accepted", "ambiguous", "partial", "crowd", "out_of_scope"}:
        return value  # type: ignore[return-value]
    _fail(
        "sealed audit reference uses an unknown final state",
        "analysis.metrics_readiness_reference_state",
        final_state=value,
    )


def _adjudication_rows_by_identifier(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        identifier = row.get("adjudication_identifier")
        _require_nonempty(identifier, field="adjudication_identifier")
        if identifier in indexed:
            _fail(
                "sealed adjudication ledger repeats an identifier",
                "analysis.metrics_readiness_adjudication_duplicate",
                adjudication_identifier=identifier,
            )
        indexed[identifier] = row
    return indexed


def _joined_adjudication_row(
    *,
    audit_row: Mapping[str, Any],
    adjudication_rows: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    identifier = audit_row.get("adjudication_identifier")
    _require_nonempty(identifier, field="adjudication_identifier")
    adjudication_row = adjudication_rows.get(identifier)
    if adjudication_row is None:
        _fail(
            "audit reference lacks its exact sealed adjudication decision",
            "analysis.metrics_readiness_adjudication_join",
            adjudication_identifier=identifier,
            reference_id=audit_row.get("object_identifier"),
        )
    return adjudication_row


def _candidate_category_names_from_adjudication(
    row: Mapping[str, Any],
) -> tuple[str, ...]:
    candidates = row.get("candidate_categories")
    if not isinstance(candidates, list):
        _fail(
            "sealed adjudication candidate categories must be a list",
            "analysis.metrics_readiness_adjudication_candidates",
            adjudication_identifier=row.get("adjudication_identifier"),
        )
    names: list[str] = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            _fail(
                "sealed adjudication candidate category must be an object",
                "analysis.metrics_readiness_adjudication_candidates",
                adjudication_identifier=row.get("adjudication_identifier"),
            )
        normalized_name = candidate.get("normalized_category_name")
        official_category_id = candidate.get("official_coco_category_id")
        _require_nonempty(
            normalized_name,
            field="candidate_categories.normalized_category_name",
        )
        if (
            normalize_coco_category_name(normalized_name) != normalized_name
            or COCO_80_OFFICIAL_CATEGORY_ID_BY_NAME.get(normalized_name)
            != official_category_id
        ):
            _fail(
                "sealed adjudication candidate category is outside canonical COCO-80",
                "analysis.metrics_readiness_adjudication_candidates",
                adjudication_identifier=row.get("adjudication_identifier"),
            )
        names.append(normalized_name)
    if names != sorted(set(names)):
        _fail(
            "sealed adjudication candidate categories must be sorted and unique",
            "analysis.metrics_readiness_adjudication_candidates",
            adjudication_identifier=row.get("adjudication_identifier"),
        )
    return tuple(names)


def _official_reference_from_row(
    *,
    row: Mapping[str, Any],
    cohort_image: Any,
    state: Literal["accepted", "crowd"],
) -> ReferenceObject:
    if row.get("source_image_sha256") != cohort_image.image_sha256:
        _fail(
            "official reference image digest differs from sealed cohort",
            "analysis.metrics_readiness_reference_image_digest",
            reference_id=row.get("object_or_region_identifier"),
        )
    geometry = row.get("geometry")
    if not isinstance(geometry, Mapping):
        _fail(
            "official reference geometry is missing",
            "analysis.metrics_readiness_reference_geometry",
            reference_id=row.get("object_or_region_identifier"),
        )
    return ReferenceObject(
        image_id=str(row.get("image_id")),
        ledger_scope="official_annotation",
        reference_id=row.get("object_or_region_identifier"),
        normalized_category_name=row.get("normalized_category_name"),
        evaluator_category_id=row.get("evaluator_local_category_id"),
        official_coco_category_id=row.get("official_coco_category_id"),
        category_namespace_sha256=row.get("coco_80_category_namespace_sha256"),
        source_canvas_bbox_xyxy=_optional_reference_box(
            geometry.get("clipped_source_corners_xyxy")
        ),
        state=state,
        provenance="official_annotation",
    )


def _optional_reference_box(value: Any) -> PixelBox | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 4:
        _fail(
            "sealed reference box must contain four coordinates",
            "analysis.metrics_readiness_reference_geometry",
        )
    return tuple(float(coordinate) for coordinate in value)  # type: ignore[return-value]


def _derive_readiness_artifacts(
    root: Path,
    *,
    cohort_artifact_name: str,
    cohort_artifact_sha256: str,
) -> dict[str, str]:
    if (
        not isinstance(cohort_artifact_name, str)
        or not cohort_artifact_name
        or Path(cohort_artifact_name).name != cohort_artifact_name
    ):
        _fail(
            "metric cohort artifact name must be root-local",
            "analysis.metrics_readiness_cohort_artifact_name",
            cohort_artifact_name=cohort_artifact_name,
        )
    _require_sha256(cohort_artifact_sha256, field="cohort_artifact_sha256")
    annotation_path = root / "annotation-derived-cohort-seal.json"
    final_path = root / "ledger-seal.json"
    annotation = _load_and_verify_readiness_seal(
        path=annotation_path,
        root=root,
        required_artifacts={
            cohort_artifact_name,
            "coco-80-category-namespace.json",
            "official-individual-ledger.jsonl",
            "official-crowd-ignore-ledger.jsonl",
        },
    )
    final = _load_and_verify_readiness_seal(
        path=final_path,
        root=root,
        required_artifacts={
            "adjudication.jsonl",
            "audit-augmented-ledger.jsonl",
        },
    )
    category_digest = annotation["artifact_digests"][
        "coco-80-category-namespace.json"
    ]
    if category_digest != COCO_80_CATEGORY_NAMESPACE_SHA256 or final.get(
        "category_namespace_sha256"
    ) != COCO_80_CATEGORY_NAMESPACE_SHA256:
        _fail(
            "readiness seals do not bind the canonical COCO-80 category namespace",
            "analysis.metrics_readiness_category",
        )
    final_source_digests = final.get("source_digests")
    official_ledger_crosslinks = {
        "official_individual_ledger_jsonl": "official-individual-ledger.jsonl",
        "official_crowd_ledger_jsonl": "official-crowd-ignore-ledger.jsonl",
    }
    if not isinstance(final_source_digests, Mapping) or any(
        final_source_digests.get(source_name)
        != annotation["artifact_digests"][artifact_name]
        for source_name, artifact_name in official_ledger_crosslinks.items()
    ):
        _fail(
            "final readiness source digests differ from annotation-sealed official ledgers",
            "analysis.metrics_readiness_official_ledger_source_crosslink",
        )
    sealed_cohort_artifact_sha256 = annotation["artifact_digests"][cohort_artifact_name]
    if sealed_cohort_artifact_sha256 != cohort_artifact_sha256:
        _fail(
            "schedule-selected cohort digest differs from the annotation-derived seal",
            "analysis.metrics_readiness_cohort_artifact_digest",
            cohort_artifact_name=cohort_artifact_name,
        )
    cohort = CohortLedger.from_jsonl_bytes((root / cohort_artifact_name).read_bytes())
    return {
        "annotation_derived_cohort_seal_sha256": sha256_file(annotation_path),
        "category_namespace_sha256": category_digest,
        "cohort_artifact_name": cohort_artifact_name,
        "cohort_artifact_sha256": sealed_cohort_artifact_sha256,
        "cohort_sha256": cohort.fingerprint,
        "readiness_seal_sha256": sha256_file(final_path),
        "reference_ledger_sha256": final["artifact_digests"][
            "audit-augmented-ledger.jsonl"
        ],
    }


def _load_and_verify_readiness_seal(
    *,
    path: Path,
    root: Path,
    required_artifacts: set[str],
) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise DataContractError(
            "required readiness seal is absent or unreadable",
            code="analysis.metrics_readiness_seal_missing",
            context={"path": str(path)},
            cause=exc,
        ) from exc
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DataContractError(
            "readiness seal must be valid JSON",
            code="analysis.metrics_readiness_seal_json",
            context={"path": str(path)},
            cause=exc,
        ) from exc
    if not isinstance(payload, dict):
        _fail(
            "readiness seal must be a JSON object",
            "analysis.metrics_readiness_seal_json",
            path=str(path),
        )
    artifact_digests = payload.get("artifact_digests")
    if not isinstance(artifact_digests, dict) or not required_artifacts.issubset(
        artifact_digests
    ):
        _fail(
            "readiness seal omits required artifact digests",
            "analysis.metrics_readiness_artifact_set",
            path=str(path),
            missing=sorted(required_artifacts - set(artifact_digests or {})),
        )
    for artifact_name, expected_digest in artifact_digests.items():
        if (
            not isinstance(artifact_name, str)
            or Path(artifact_name).name != artifact_name
        ):
            _fail(
                "readiness seal artifact name must be root-local",
                "analysis.metrics_readiness_artifact_name",
                artifact_name=artifact_name,
            )
        _require_sha256(expected_digest, field=f"artifact_digests.{artifact_name}")
        artifact_path = root / artifact_name
        if not artifact_path.is_file() or sha256_file(artifact_path) != expected_digest:
            _fail(
                "readiness seal artifact content differs from its digest",
                "analysis.metrics_readiness_artifact_digest",
                artifact_name=artifact_name,
            )
    return payload


def _require_nonempty(value: object, *, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        _fail(
            "field must be a nonempty string", "analysis.metrics_nonempty", field=field
        )


def _require_sha256(value: object, *, field: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(
            "field must be a lowercase Secure Hash Algorithm 256-bit digest",
            "analysis.metrics_sha256",
            field=field,
        )


def _require_nonnegative_integer(value: object, *, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail(
            "field must be a nonnegative integer",
            "analysis.metrics_nonnegative",
            field=field,
        )


def _require_positive_integer(value: object, *, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _fail(
            "field must be a positive integer",
            "analysis.metrics_positive",
            field=field,
        )


def _fail(message: str, code: str, **context: Any) -> None:
    raise DataContractError(message, code=code, context=context)
