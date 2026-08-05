#!/usr/bin/env python3
"""Pure candidate-bank and basin accounting for the sorted-owner landscape.

This module intentionally does not import a model, tokenizer, torch, or
Transformers.  A later execution runner owns prompt construction and
teacher-forced forwards; it supplies raw selected-token log probabilities (and,
when required, the four pre-softmax vocabulary rows).  Keeping this layer pure
makes the geometry proposal, score accumulation, cluster identity, and mass
receipts independently reproducible on CPU.

Decision-bearing numerical choices do not live here.  Anchor margins, proposal
weights, clustering radii, shape rules, and the prominence functional are
required in a validated rules mapping.  This code consequently cannot silently
turn a provisional research convention into an outcome-sensitive default.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
import hashlib
import json
import math
from typing import Any, Literal, TypeGuard


RULES_SCHEMA_VERSION = "sorted_owner_basin_landscape_rules.v2"
SEMANTIC_CORE_SCHEMA_VERSION = "sorted_owner_basin_semantic_core.v1"
GEOMETRY_IDENTITY_SCHEMA = "canonical_round_bin_times_extent_over_1000.v1"
COORDINATE_BIN_MIN = 0
COORDINATE_BIN_MAX = 999
COORDINATE_NAMES = ("x1", "y1", "x2", "y2")
POLICY_SCORE_LABEL = "non_likelihood_repetition_penalty_policy_score"

# These fields are deliberately outside the scientific-semantic digest.  They
# choose which already-frozen owner/context rows are executed and bind the
# resulting files; none is allowed to redefine a score, geometry, registry,
# calibration, or interpretation convention.
SEMANTIC_CORE_OUTER_EXECUTION_FIELDS = (
    "structural_status",
    "task6_context_selection",
    "sealed_inputs",
    "non_c_smoke_freeze_receipt",
    "candidate_materializer",
    "output_plan_identity",
)


def _canonical_json_digest(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    except (TypeError, ValueError) as exc:
        raise ValueError("semantic-core payload must be canonically JSON serializable") from exc
    return hashlib.sha256(encoded).hexdigest()


def build_semantic_core_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    """Project every outcome-changing convention from an authored rule plan."""

    rules = _require_mapping(value, "landscape decision rules")
    free_tree = _require_mapping(
        rules.get("free_coordinate_tree"), "rules.free_coordinate_tree"
    )
    ablation = _require_mapping(rules.get("ablation"), "rules.ablation")
    calibration = _require_mapping(rules.get("calibration"), "rules.calibration")
    materializer_value = rules.get("candidate_materializer")
    materializer = (
        None
        if materializer_value is None
        else _require_mapping(materializer_value, "rules.candidate_materializer")
    )
    analysis_channels = (
        {}
        if materializer is None
        else _require_mapping(
            materializer.get("analysis_channels"),
            "rules.candidate_materializer.analysis_channels",
        )
    )
    return {
        "schema_version": SEMANTIC_CORE_SCHEMA_VERSION,
        "contract_mode": rules.get("contract_mode"),
        "geometry_and_token_rules": {
            "geometry_identity": rules.get("geometry_identity"),
            "coordinate_space_contract": rules.get("coordinate_space_contract"),
            "coordinate_bins": rules.get("coordinate_bins"),
            "target_anchor": rules.get("target_anchor"),
            "extent_grid_policy": rules.get("extent_grid_policy"),
            "schema_tokens": rules.get("schema_tokens"),
            "token_registry": rules.get("token_registry"),
            "structural_row_wrapper": rules.get("structural_row_wrapper"),
        },
        "global_foil_role_and_description_registry": rules.get(
            "global_foil_role_and_description_registry"
        ),
        "proposal_measures": rules.get("proposal_measures"),
        "bank_proposal_measure": rules.get("bank_proposal_measure"),
        "candidate_weights": rules.get("candidate_weights"),
        "candidate_materialization_rules": {
            "p_x1_y1_pruning": rules.get("p_x1_y1_pruning"),
            "target_bank_name": rules.get("target_bank_name"),
            "coco_namespace": rules.get("coco_namespace"),
            "materializer_schema_version": (
                None if materializer is None else materializer.get("schema_version")
            ),
            "seal_scope": (
                None if materializer is None else materializer.get("seal_scope")
            ),
            "analysis_channel_owners": {
                str(name): _require_mapping(
                    channel, f"candidate materializer analysis channel {name}"
                ).get("owner")
                for name, channel in sorted(analysis_channels.items())
            },
        },
        "free_coordinate_tree": {
            "budget": free_tree.get("budget"),
            "selector": free_tree.get("selector"),
            "candidate_membership_surface": free_tree.get(
                "candidate_membership_surface"
            ),
        },
        "spatial_clustering": rules.get("spatial_clustering"),
        "shape": rules.get("shape"),
        "declared_extent_submodes": rules.get("declared_extent_submodes"),
        "prominence": rules.get("prominence"),
        "score_channels": rules.get("score_channels"),
        "registered_sampling_policy": rules.get("registered_sampling_policy"),
        "loose_support": rules.get("loose_support"),
        "ablation": dict(ablation),
        "calibration": {
            "algorithm": calibration.get("algorithm"),
            "quantile": calibration.get("quantile"),
            "minimum_roles": calibration.get("minimum_roles"),
            "metric_reducers": calibration.get("metric_reducers"),
            "control_ids": calibration.get("control_ids"),
        },
        "invalidation_rule": rules.get("invalidation_rule"),
        "control_hierarchy": rules.get("control_hierarchy"),
        "canonical_description_source": rules.get("canonical_description_source"),
        "canonical_alias_policy": rules.get("canonical_alias_policy"),
        "model_tokenizer_runtime_vocabulary_identity": rules.get(
            "model_tokenizer_runtime_vocabulary_identity"
        ),
    }


def semantic_core_payload(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return and validate the canonical outcome-changing rule payload.

    The payload is authored explicitly instead of being obtained by subtracting
    a growing deny-list from the outer plan.  This keeps the digest surface
    reviewable and prevents a new execution-plan field from silently becoming
    scientific semantics (or vice versa).
    """

    rules = _require_mapping(value, "landscape decision rules")
    core_record = _require_mapping(rules.get("semantic_core"), "rules.semantic_core")
    if core_record.get("schema_version") != SEMANTIC_CORE_SCHEMA_VERSION:
        raise ValueError(
            f"rules.semantic_core.schema_version must be {SEMANTIC_CORE_SCHEMA_VERSION!r}"
        )
    payload = _require_mapping(core_record.get("payload"), "rules.semantic_core.payload")
    declared = _require_sha256_digest(
        core_record.get("sha256"), "rules.semantic_core.sha256"
    )
    observed = _canonical_json_digest(payload)
    if declared != observed:
        raise ValueError("rules.semantic_core.sha256 is stale")
    if dict(payload) != build_semantic_core_payload(rules):
        raise ValueError(
            "rules.semantic_core.payload is not the canonical projection of the operational rules"
        )
    if "candidate_materializer" in rules:
        materializer = _require_mapping(
            rules.get("candidate_materializer"), "rules.candidate_materializer"
        )
        global_registry = _require_mapping(
            rules.get("global_foil_role_and_description_registry"),
            "rules.global_foil_role_and_description_registry",
        )
        duplicate_bindings = (
            (
                materializer.get("coordinate_space"),
                rules.get("coordinate_space_contract"),
                "candidate-materializer coordinate space",
            ),
            (
                materializer.get("bank_roles"),
                rules.get("bank_roles"),
                "candidate-materializer bank roles",
            ),
            (
                materializer.get("foil_set"),
                global_registry.get("foil_set"),
                "candidate-materializer global foil set",
            ),
            (
                materializer.get("p_x1_y1_pruning"),
                rules.get("p_x1_y1_pruning"),
                "candidate-materializer pruning",
            ),
            (
                materializer.get("free_search_budget"),
                rules.get("free_search"),
                "candidate-materializer free-search budget",
            ),
            (
                materializer.get("token_registry"),
                rules.get("token_registry"),
                "candidate-materializer token registry",
            ),
            (
                materializer.get("structural_row_wrapper"),
                rules.get("structural_row_wrapper"),
                "candidate-materializer row wrapper",
            ),
            (
                materializer.get("target_bank_name"),
                rules.get("target_bank_name"),
                "candidate-materializer target bank",
            ),
            (
                materializer.get("coco_namespace"),
                rules.get("coco_namespace"),
                "candidate-materializer COCO namespace",
            ),
            (
                rules.get("registered_basin_roles"),
                global_registry.get("registered_basin_roles"),
                "global registered basin roles",
            ),
            (
                rules.get("owner_canonical_descriptions"),
                global_registry.get("owner_canonical_descriptions"),
                "global canonical descriptions",
            ),
        )
        for observed_binding, semantic_binding, label in duplicate_bindings:
            if observed_binding != semantic_binding:
                raise ValueError(f"{label} drifts from the canonical semantic core")
    calibration_contract = _require_mapping(
        rules.get("calibration_contract"), "rules.calibration_contract"
    )
    semantic_calibration = _require_mapping(
        payload.get("calibration"), "rules.semantic_core.payload.calibration"
    )
    for key in ("algorithm", "quantile", "minimum_roles", "metric_reducers", "control_ids"):
        if calibration_contract.get(key) != semantic_calibration.get(key):
            raise ValueError(
                f"calibration_contract.{key} drifts from the canonical semantic core"
            )
    if (
        tuple(core_record.get("excluded_outer_execution_fields", ()))
        != SEMANTIC_CORE_OUTER_EXECUTION_FIELDS
    ):
        raise ValueError(
            "rules.semantic_core must declare the exact outer execution-plan exclusions"
        )
    return payload


def semantic_core_digest(value: Mapping[str, Any]) -> str:
    """Return the canonical semantic digest for a two-stage decision plan."""

    semantic_core_payload(value)
    return _require_sha256_digest(
        _require_mapping(value.get("semantic_core"), "rules.semantic_core").get("sha256"),
        "rules.semantic_core.sha256",
    )


def _is_int(value: Any) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _require_sha256_digest(value: Any, label: str) -> str:
    digest = _require_nonempty_string(value, label)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{label} must be a lowercase hexadecimal SHA-256 digest")
    return digest


def _require_finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _require_sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be a sequence")
    return value


@dataclass(frozen=True, order=True)
class CoordinateBin:
    """One Qwen coordinate-vocabulary bin, structurally bounded to 0 through 999."""

    value: int

    def __post_init__(self) -> None:
        if not _is_int(self.value):
            raise ValueError("coordinate bin must be an integer")
        if not COORDINATE_BIN_MIN <= self.value <= COORDINATE_BIN_MAX:
            raise ValueError(
                f"coordinate bin {self.value} is outside "
                f"[{COORDINATE_BIN_MIN}, {COORDINATE_BIN_MAX}]"
            )


@dataclass(frozen=True)
class CoordinateBox:
    """A valid, non-empty xyxy box expressed in coordinate bins."""

    x1: CoordinateBin
    y1: CoordinateBin
    x2: CoordinateBin
    y2: CoordinateBin

    def __post_init__(self) -> None:
        if self.x1.value >= self.x2.value:
            raise ValueError("coordinate box requires x1 < x2")
        if self.y1.value >= self.y2.value:
            raise ValueError("coordinate box requires y1 < y2")

    @classmethod
    def from_values(cls, x1: int, y1: int, x2: int, y2: int) -> "CoordinateBox":
        return cls(CoordinateBin(x1), CoordinateBin(y1), CoordinateBin(x2), CoordinateBin(y2))

    @property
    def width(self) -> int:
        return self.x2.value - self.x1.value

    @property
    def height(self) -> int:
        return self.y2.value - self.y1.value

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1.value + self.x2.value) / 2.0, (self.y1.value + self.y2.value) / 2.0)

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x1.value, self.y1.value, self.x2.value, self.y2.value)


@dataclass(frozen=True)
class RawCoordinateLogprobs:
    """Raw-model selected-token log probabilities for a complete box.

    These are likelihood terms, unlike ``PolicyCoordinateScore`` below.  A
    valid log probability cannot be positive beyond floating-point noise.
    """

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        for name in COORDINATE_NAMES:
            value = _require_finite_number(getattr(self, name), f"raw {name} logprob")
            if value > 1e-12:
                raise ValueError(f"raw {name} logprob must be non-positive")
            object.__setattr__(self, name, value)

    @property
    def total(self) -> float:
        return self.x1 + self.y1 + self.x2 + self.y2

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass(frozen=True)
class TargetAnchorRules:
    margin_fraction: float
    min_margin_bins: int
    max_margin_bins: int


@dataclass(frozen=True)
class ProposalMeasure:
    """Explicit normalized proposal measure for one candidate-bank family."""

    measure_id: str
    comparability_group: str
    normalization: Literal["full_domain_normalized_weighted_sum"]
    bank_weights: tuple[tuple[str, float], ...]

    def weight_for(self, bank_name: str) -> float:
        weights = dict(self.bank_weights)
        try:
            return weights[bank_name]
        except KeyError as exc:
            raise ValueError(
                f"proposal measure {self.measure_id!r} has no weight for bank {bank_name!r}"
            ) from exc


@dataclass(frozen=True)
class SpatialClusteringRules:
    owner_link_iou_min: float
    owner_link_center_distance_max: float
    extent_submode_iou_min: float


@dataclass(frozen=True)
class ShapeRules:
    near_peak_logprob_delta: float
    wide_ridge_min_candidates: int
    multi_submode_min: int
    merged_extent_submodes: tuple[str, ...]
    scan_bank_names: tuple[str, ...]


@dataclass(frozen=True)
class RegisteredBasinRole:
    """One frozen target or foil role admitted by the decision rules."""

    role_id: str
    kind: Literal["target", "foil"]
    foil_set_id: str
    identity_kind: Literal["reviewed_physical_owner", "registered_geometry"]
    allowed_bank_names: tuple[str, ...]


@dataclass(frozen=True)
class LandscapeRules:
    """Validated, frozen-rule projection used by all pure operations."""

    coordinate_min: int
    coordinate_max: int
    contract_mode: Literal["production", "test_fixture"]
    geometry_identity_schema: Literal["canonical_round_bin_times_extent_over_1000.v1"]
    rule_digest: str
    target_anchor: TargetAnchorRules
    bank_order: tuple[str, ...]
    bank_proposal_measure: tuple[tuple[str, str], ...]
    proposal_measures: tuple[ProposalMeasure, ...]
    spatial_clustering: SpatialClusteringRules
    shape: ShapeRules
    declared_extent_submodes: tuple[str, ...]
    registered_basin_roles: tuple[RegisteredBasinRole, ...]
    prominence_functional: Literal["peak_height_difference"]

    def bank_rank(self, bank_name: str) -> int:
        try:
            return self.bank_order.index(bank_name)
        except ValueError as exc:
            raise ValueError(f"unknown deterministic bank {bank_name!r}") from exc

    def proposal_measure_for_bank(self, bank_name: str) -> ProposalMeasure:
        mapping = dict(self.bank_proposal_measure)
        try:
            measure_id = mapping[bank_name]
        except KeyError as exc:
            raise ValueError(f"bank {bank_name!r} has no proposal measure") from exc
        for measure in self.proposal_measures:
            if measure.measure_id == measure_id:
                return measure
        raise AssertionError(f"validated rules lost proposal measure {measure_id!r}")

    def basin_role(self, role_id: str) -> RegisteredBasinRole:
        for role in self.registered_basin_roles:
            if role.role_id == role_id:
                return role
        raise ValueError(f"unregistered basin role {role_id!r}")

    @property
    def can_emit_production_attestation(self) -> bool:
        return self.contract_mode == "production"


def validate_rule_mapping(value: Mapping[str, Any]) -> LandscapeRules:
    """Validate every non-model rule required by this CPU core.

    This parser deliberately has no fallback values.  A caller must freeze all
    outcome-sensitive conventions before candidate or mass receipts are made.
    """

    rules = _require_mapping(value, "landscape decision rules")
    if rules.get("schema_version") != RULES_SCHEMA_VERSION:
        raise ValueError(f"rules.schema_version must be {RULES_SCHEMA_VERSION!r}")
    contract_mode = rules.get("contract_mode")
    if contract_mode not in {"production", "test_fixture"}:
        raise ValueError("rules.contract_mode must be 'production' or 'test_fixture'")
    if "semantic_core" in rules:
        rule_digest = semantic_core_digest(rules)
    else:
        try:
            rule_digest = _canonical_json_digest(rules)
        except ValueError as exc:
            raise ValueError(
                "landscape decision rules must be JSON serializable for their digest"
            ) from exc
    geometry_raw = _require_mapping(rules.get("geometry_identity"), "rules.geometry_identity")
    if geometry_raw.get("schema") != GEOMETRY_IDENTITY_SCHEMA:
        raise ValueError(
            f"rules.geometry_identity.schema must be {GEOMETRY_IDENTITY_SCHEMA!r}"
        )
    if geometry_raw.get("coordinate_denominator") != 1000:
        raise ValueError("rules.geometry_identity.coordinate_denominator must be 1000")

    coordinate_bins = _require_mapping(rules.get("coordinate_bins"), "rules.coordinate_bins")
    coordinate_min = coordinate_bins.get("min")
    coordinate_max = coordinate_bins.get("max")
    if not _is_int(coordinate_min) or not _is_int(coordinate_max):
        raise ValueError("rules.coordinate_bins min and max must be integers")
    if not COORDINATE_BIN_MIN <= coordinate_min < coordinate_max <= COORDINATE_BIN_MAX:
        raise ValueError("rules.coordinate_bins must be an increasing subset of the coordinate vocabulary")
    if contract_mode == "production" and (
        coordinate_min != COORDINATE_BIN_MIN or coordinate_max != COORDINATE_BIN_MAX
    ):
        raise ValueError("production rules require the complete coordinate vocabulary 0..999")

    target_raw = _require_mapping(rules.get("target_anchor"), "rules.target_anchor")
    margin_fraction = _require_finite_number(
        target_raw.get("margin_fraction"), "rules.target_anchor.margin_fraction"
    )
    min_margin = target_raw.get("min_margin_bins")
    max_margin = target_raw.get("max_margin_bins")
    if margin_fraction < 0.0:
        raise ValueError("rules.target_anchor.margin_fraction must be non-negative")
    if not _is_int(min_margin) or not _is_int(max_margin) or min_margin < 0 or max_margin < min_margin:
        raise ValueError("rules.target_anchor margins must be non-negative increasing integers")
    target_anchor = TargetAnchorRules(margin_fraction, min_margin, max_margin)

    bank_order_raw = _require_sequence(rules.get("bank_order"), "rules.bank_order")
    bank_order = tuple(_require_nonempty_string(item, "rules.bank_order entry") for item in bank_order_raw)
    if not bank_order or len(set(bank_order)) != len(bank_order):
        raise ValueError("rules.bank_order must contain unique bank names")

    measures_raw = _require_mapping(rules.get("proposal_measures"), "rules.proposal_measures")
    measures: list[ProposalMeasure] = []
    for measure_id in sorted(measures_raw):
        specification = _require_mapping(measures_raw[measure_id], f"proposal measure {measure_id!r}")
        measure_name = _require_nonempty_string(measure_id, "proposal measure id")
        comparability_group = _require_nonempty_string(
            specification.get("comparability_group"),
            f"proposal measure {measure_id!r}.comparability_group",
        )
        if specification.get("normalization") != "full_domain_normalized_weighted_sum":
            raise ValueError(
                f"proposal measure {measure_id!r}.normalization must be "
                "'full_domain_normalized_weighted_sum'"
            )
        weights_raw = _require_mapping(
            specification.get("bank_weights"), f"proposal measure {measure_id!r}.bank_weights"
        )
        weights: list[tuple[str, float]] = []
        for bank_name in sorted(weights_raw):
            bank = _require_nonempty_string(bank_name, "proposal measure bank name")
            weight = _require_finite_number(weights_raw[bank_name], f"proposal weight for {bank!r}")
            if weight <= 0.0:
                raise ValueError(f"proposal weight for {bank!r} must be positive")
            weights.append((bank, weight))
        if not weights:
            raise ValueError(f"proposal measure {measure_id!r} has no bank weights")
        measures.append(
            ProposalMeasure(
                measure_id=measure_name,
                comparability_group=comparability_group,
                normalization="full_domain_normalized_weighted_sum",
                bank_weights=tuple(weights),
            )
        )
    if not measures:
        raise ValueError("rules.proposal_measures must not be empty")
    measure_names = {measure.measure_id for measure in measures}

    bank_measure_raw = _require_mapping(
        rules.get("bank_proposal_measure"), "rules.bank_proposal_measure"
    )
    if set(bank_measure_raw) != set(bank_order):
        raise ValueError("rules.bank_proposal_measure must name exactly the deterministic banks")
    bank_measures: list[tuple[str, str]] = []
    measures_by_name = {measure.measure_id: measure for measure in measures}
    for bank in bank_order:
        measure_id = _require_nonempty_string(
            bank_measure_raw[bank], f"proposal measure for bank {bank!r}"
        )
        if measure_id not in measure_names:
            raise ValueError(f"bank {bank!r} references unknown proposal measure {measure_id!r}")
        if bank not in dict(measures_by_name[measure_id].bank_weights):
            raise ValueError(
                f"proposal measure {measure_id!r} must explicitly weight its bank {bank!r}"
            )
        bank_measures.append((bank, measure_id))

    clustering_raw = _require_mapping(rules.get("spatial_clustering"), "rules.spatial_clustering")
    owner_iou = _require_finite_number(
        clustering_raw.get("owner_link_iou_min"), "rules.spatial_clustering.owner_link_iou_min"
    )
    owner_center = _require_finite_number(
        clustering_raw.get("owner_link_center_distance_max"),
        "rules.spatial_clustering.owner_link_center_distance_max",
    )
    submode_iou = _require_finite_number(
        clustering_raw.get("extent_submode_iou_min"),
        "rules.spatial_clustering.extent_submode_iou_min",
    )
    if not 0.0 < owner_iou <= 1.0 or owner_center < 0.0 or not 0.0 < submode_iou <= 1.0:
        raise ValueError("spatial clustering thresholds are outside their valid ranges")
    clustering = SpatialClusteringRules(owner_iou, owner_center, submode_iou)

    shape_raw = _require_mapping(rules.get("shape"), "rules.shape")
    near_peak_delta = _require_finite_number(
        shape_raw.get("near_peak_logprob_delta"), "rules.shape.near_peak_logprob_delta"
    )
    wide_ridge_min = shape_raw.get("wide_ridge_min_candidates")
    multi_submode_min = shape_raw.get("multi_submode_min")
    if near_peak_delta < 0.0 or not _is_int(wide_ridge_min) or wide_ridge_min < 2:
        raise ValueError("rules.shape near-peak and ridge settings are invalid")
    if not _is_int(multi_submode_min) or multi_submode_min < 2:
        raise ValueError("rules.shape.multi_submode_min must be at least two")
    merged_raw = _require_sequence(shape_raw.get("merged_extent_submodes"), "rules.shape.merged_extent_submodes")
    scan_raw = _require_sequence(shape_raw.get("scan_bank_names"), "rules.shape.scan_bank_names")
    merged = tuple(sorted({_require_nonempty_string(item, "merged extent submode") for item in merged_raw}))
    scan_banks = tuple(sorted({_require_nonempty_string(item, "scan bank name") for item in scan_raw}))
    unknown_scan_banks = set(scan_banks).difference(bank_order)
    if unknown_scan_banks:
        raise ValueError(f"rules.shape.scan_bank_names has unknown banks: {sorted(unknown_scan_banks)}")
    shape = ShapeRules(near_peak_delta, wide_ridge_min, multi_submode_min, merged, scan_banks)

    submodes_raw = _require_sequence(
        rules.get("declared_extent_submodes"), "rules.declared_extent_submodes"
    )
    declared_submodes = tuple(
        sorted({_require_nonempty_string(item, "declared extent submode") for item in submodes_raw})
    )
    if not declared_submodes:
        raise ValueError("rules.declared_extent_submodes must not be empty")
    unknown_merged = set(merged).difference(declared_submodes)
    if unknown_merged:
        raise ValueError(
            f"rules.shape.merged_extent_submodes are not declared extent submodes: {sorted(unknown_merged)}"
        )

    roles_raw = _require_mapping(rules.get("registered_basin_roles"), "rules.registered_basin_roles")
    registered_roles: list[RegisteredBasinRole] = []
    for role_id in sorted(roles_raw):
        specification = _require_mapping(roles_raw[role_id], f"registered basin role {role_id!r}")
        name = _require_nonempty_string(role_id, "registered basin role id")
        kind = specification.get("kind")
        if kind not in {"target", "foil"}:
            raise ValueError(f"registered basin role {name!r}.kind must be 'target' or 'foil'")
        foil_set_id = _require_nonempty_string(
            specification.get("foil_set_id"), f"registered basin role {name!r}.foil_set_id"
        )
        identity_kind = specification.get("identity_kind")
        if identity_kind not in {"reviewed_physical_owner", "registered_geometry"}:
            raise ValueError(
                f"registered basin role {name!r}.identity_kind must be "
                "'reviewed_physical_owner' or 'registered_geometry'"
            )
        allowed_raw = _require_sequence(
            specification.get("allowed_bank_names"),
            f"registered basin role {name!r}.allowed_bank_names",
        )
        allowed_banks = tuple(
            sorted({_require_nonempty_string(item, "registered role bank name") for item in allowed_raw})
        )
        if not allowed_banks or set(allowed_banks).difference(bank_order):
            raise ValueError(
                f"registered basin role {name!r} must declare known allowed bank names"
            )
        if kind == "target" and identity_kind != "reviewed_physical_owner":
            raise ValueError("target basin roles require reviewed physical-owner identity")
        if kind == "target" and (
            "target" not in allowed_banks
            or {"covered", "background", "scan"}.intersection(allowed_banks)
        ):
            raise ValueError(
                "target basin roles must admit target geometry and cannot admit foil banks"
            )
        if identity_kind == "registered_geometry" and {"target", "covered"}.intersection(
            allowed_banks
        ):
            raise ValueError(
                "target and covered-owner banks cannot use owner-neutral registered geometry"
            )
        if identity_kind == "registered_geometry" and not set(allowed_banks).issubset(
            {"background", "scan"}
        ):
            raise ValueError(
                "owner-neutral registered geometry is restricted to background and scan foil banks"
            )
        registered_roles.append(
            RegisteredBasinRole(name, kind, foil_set_id, identity_kind, allowed_banks)
        )
    if not registered_roles or {role.kind for role in registered_roles} != {"target", "foil"}:
        raise ValueError("rules.registered_basin_roles must declare at least one target and one foil role")

    prominence_raw = _require_mapping(rules.get("prominence"), "rules.prominence")
    if prominence_raw.get("functional") != "peak_height_difference":
        raise ValueError("rules.prominence.functional must be 'peak_height_difference'")

    return LandscapeRules(
        coordinate_min=coordinate_min,
        coordinate_max=coordinate_max,
        contract_mode=contract_mode,
        geometry_identity_schema=GEOMETRY_IDENTITY_SCHEMA,
        rule_digest=rule_digest,
        target_anchor=target_anchor,
        bank_order=bank_order,
        bank_proposal_measure=tuple(bank_measures),
        proposal_measures=tuple(measures),
        spatial_clustering=clustering,
        shape=shape,
        declared_extent_submodes=declared_submodes,
        registered_basin_roles=tuple(registered_roles),
        prominence_functional="peak_height_difference",
    )


def _validate_box_in_rules(box: CoordinateBox, rules: LandscapeRules, label: str) -> None:
    if box.x1.value < rules.coordinate_min or box.y1.value < rules.coordinate_min:
        raise ValueError(f"{label} lies below the declared coordinate range")
    if box.x2.value > rules.coordinate_max or box.y2.value > rules.coordinate_max:
        raise ValueError(f"{label} lies above the declared coordinate range")


@dataclass(frozen=True)
class CanonicalGeometryIdentity:
    """Pixel-space identity derived by the production coordinate contract."""

    schema: Literal["canonical_round_bin_times_extent_over_1000.v1"]
    coordinate_box: CoordinateBox
    image_width: int
    image_height: int
    pixel_box_xyxy: tuple[int, int, int, int]
    identity_digest: str


def canonical_geometry_coordinate(
    coordinate: CoordinateBin, extent: int, rules: LandscapeRules
) -> int:
    """Apply the frozen ``round(value * extent / 1000)`` coordinate identity."""

    if rules.geometry_identity_schema != GEOMETRY_IDENTITY_SCHEMA:
        raise ValueError("rules do not declare the canonical geometry identity schema")
    if not _is_int(extent) or extent <= 0:
        raise ValueError("geometry extent must be a positive integer")
    return round(coordinate.value * extent / 1000)


def canonical_geometry_identity(
    box: CoordinateBox,
    *,
    image_width: int,
    image_height: int,
    rules: LandscapeRules,
) -> CanonicalGeometryIdentity:
    """Return a digestible canonical geometry receipt for a coordinate box."""

    _validate_box_in_rules(box, rules, "canonical geometry box")
    pixels = (
        canonical_geometry_coordinate(box.x1, image_width, rules),
        canonical_geometry_coordinate(box.y1, image_height, rules),
        canonical_geometry_coordinate(box.x2, image_width, rules),
        canonical_geometry_coordinate(box.y2, image_height, rules),
    )
    payload = {
        "coordinate_box": list(box.as_tuple()),
        "image_height": image_height,
        "image_width": image_width,
        "pixel_box_xyxy": list(pixels),
        "schema": GEOMETRY_IDENTITY_SCHEMA,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return CanonicalGeometryIdentity(
        schema=GEOMETRY_IDENTITY_SCHEMA,
        coordinate_box=box,
        image_width=image_width,
        image_height=image_height,
        pixel_box_xyxy=pixels,
        identity_digest=digest,
    )


def _size_aware_margin(length: int, anchor_rules: TargetAnchorRules) -> int:
    # ceil preserves the declared positive fractional band for small GT boxes.
    requested = math.ceil(length * anchor_rules.margin_fraction)
    return min(anchor_rules.max_margin_bins, max(anchor_rules.min_margin_bins, requested))


def enumerate_target_x1_anchors(gt_box: CoordinateBox, rules: LandscapeRules) -> tuple[CoordinateBin, ...]:
    """Return every target x1 anchor in the full GT interior plus frozen margin."""

    _validate_box_in_rules(gt_box, rules, "ground-truth box")
    margin = _size_aware_margin(gt_box.width, rules.target_anchor)
    lower = max(rules.coordinate_min, gt_box.x1.value - margin)
    # x1 needs one later representable x2, independently of a particular extent seed.
    upper = min(rules.coordinate_max - 1, gt_box.x2.value - 1 + margin)
    return tuple(CoordinateBin(value) for value in range(lower, upper + 1))


def enumerate_target_y1_anchors(gt_box: CoordinateBox, rules: LandscapeRules) -> tuple[CoordinateBin, ...]:
    """Return every target y1 anchor in the full GT interior plus frozen margin."""

    _validate_box_in_rules(gt_box, rules, "ground-truth box")
    margin = _size_aware_margin(gt_box.height, rules.target_anchor)
    lower = max(rules.coordinate_min, gt_box.y1.value - margin)
    upper = min(rules.coordinate_max - 1, gt_box.y2.value - 1 + margin)
    return tuple(CoordinateBin(value) for value in range(lower, upper + 1))


@dataclass(frozen=True)
class ConditionalY1Entry:
    """One y1 vocabulary item scored after one supplied x1 anchor."""

    x1: CoordinateBin
    y1: CoordinateBin
    is_target_anchor: bool
    can_form_valid_box: bool
    invalid_box_reason: str | None


@dataclass(frozen=True)
class ConditionalY1ScoreReceipt:
    """One raw selected-token y1 score in a complete conditional vocabulary row."""

    x1: CoordinateBin
    y1: CoordinateBin
    raw_selected_token_logprob: float
    can_form_valid_box: bool
    invalid_box_reason: str | None

    def __post_init__(self) -> None:
        value = _require_finite_number(
            self.raw_selected_token_logprob, "conditional y1 raw selected-token logprob"
        )
        if value > 1e-12:
            raise ValueError("conditional y1 raw selected-token logprob must be non-positive")
        object.__setattr__(self, "raw_selected_token_logprob", value)


@dataclass(frozen=True)
class ConditionalY1CompletenessAttestation:
    """Fully identity-bound digest of complete y1 rows for every admitted x1."""

    attestation_kind: Literal["production", "test_fixture"]
    diagnostic_owner_id: str
    gt_owner_id: str
    image_identity: str
    gt_box: CoordinateBox
    canonical_description_text: str
    canonical_description_token_digest: str
    context_id: str
    context_token_digest: str
    tokenizer_identity: str
    model_identity: str
    runtime_identity: str
    geometry_identity_schema: Literal["canonical_round_bin_times_extent_over_1000.v1"]
    rule_digest: str
    declared_x1_bins: tuple[int, ...]
    per_x1_receipt_digests: tuple[tuple[int, str], ...]
    completeness_digest: str

    def __post_init__(self) -> None:
        if self.attestation_kind not in {"production", "test_fixture"}:
            raise ValueError("conditional y1 attestation kind must be production or test_fixture")
        _require_nonempty_string(self.diagnostic_owner_id, "conditional y1 diagnostic owner ID")
        _require_nonempty_string(self.gt_owner_id, "conditional y1 GT owner ID")
        _require_nonempty_string(self.image_identity, "conditional y1 image identity")
        _require_nonempty_string(
            self.canonical_description_text, "conditional y1 canonical description text"
        )
        _require_sha256_digest(
            self.canonical_description_token_digest,
            "conditional y1 canonical description token digest",
        )
        _require_nonempty_string(self.context_id, "conditional y1 completeness context_id")
        _require_sha256_digest(
            self.context_token_digest, "conditional y1 exact context-token digest"
        )
        _require_nonempty_string(self.tokenizer_identity, "conditional y1 tokenizer identity")
        _require_nonempty_string(self.model_identity, "conditional y1 model identity")
        _require_nonempty_string(self.runtime_identity, "conditional y1 runtime identity")
        if self.geometry_identity_schema != GEOMETRY_IDENTITY_SCHEMA:
            raise ValueError("conditional y1 attestation has unknown geometry identity schema")
        _require_nonempty_string(self.rule_digest, "conditional y1 completeness rule_digest")
        _require_nonempty_string(self.completeness_digest, "conditional y1 completeness digest")
        if not self.declared_x1_bins:
            raise ValueError("conditional y1 completeness must declare at least one x1 bin")
        if tuple(sorted(set(self.declared_x1_bins))) != self.declared_x1_bins:
            raise ValueError("conditional y1 completeness x1 bins must be sorted and unique")
        expected_bins = tuple(item[0] for item in self.per_x1_receipt_digests)
        if expected_bins != self.declared_x1_bins:
            raise ValueError("conditional y1 completeness per-x1 digests must cover declared x1 bins")
        for x1, digest in self.per_x1_receipt_digests:
            if not _is_int(x1) or not COORDINATE_BIN_MIN <= x1 <= COORDINATE_BIN_MAX:
                raise ValueError("conditional y1 completeness has invalid x1 bin")
            _require_nonempty_string(digest, "conditional y1 per-x1 receipt digest")


def enumerate_complete_conditional_y1(
    x1: CoordinateBin, gt_box: CoordinateBox, rules: LandscapeRules
) -> tuple[ConditionalY1Entry, ...]:
    """Enumerate the complete declared y1 vocabulary for an admitted x1.

    ``is_target_anchor`` marks the GT-interior-plus-margin subset.  Returning
    all bins, rather than only that subset, makes the required conditional-y1
    upper-bound score surface explicit and auditable.
    """

    if not rules.coordinate_min <= x1.value < rules.coordinate_max:
        raise ValueError("admitted x1 is outside the declared usable coordinate range")
    target_y = {item.value for item in enumerate_target_y1_anchors(gt_box, rules)}
    return tuple(
        ConditionalY1Entry(
            x1=x1,
            y1=CoordinateBin(y1),
            is_target_anchor=y1 in target_y,
            # The final vocabulary item is still scored for the required full
            # conditional surface, but no positive-height box can continue from it.
            can_form_valid_box=y1 < rules.coordinate_max,
            invalid_box_reason=(
                None if y1 < rules.coordinate_max else "no_later_representable_y2"
            ),
        )
        for y1 in range(rules.coordinate_min, rules.coordinate_max + 1)
    )


def validate_complete_conditional_y1_scores(
    *,
    x1: CoordinateBin,
    gt_box: CoordinateBox,
    scores: Sequence[ConditionalY1ScoreReceipt],
    rules: LandscapeRules,
) -> tuple[ConditionalY1ScoreReceipt, ...]:
    """Require every y1 bin exactly once for one admitted x1 score surface.

    The terminal y1 vocabulary item remains mandatory even though its receipt
    records that it cannot form a non-empty box.  Missing or repeated bins are
    not a sparse approximation; they are an invalid conditional landscape.
    """

    expected = enumerate_complete_conditional_y1(x1, gt_box, rules)
    by_y1: dict[int, ConditionalY1ScoreReceipt] = {}
    for score in scores:
        if not isinstance(score, ConditionalY1ScoreReceipt):
            raise ValueError("conditional y1 scores must use ConditionalY1ScoreReceipt")
        if score.x1 != x1:
            raise ValueError("conditional y1 score receipt has a different x1 anchor")
        y1 = score.y1.value
        if y1 in by_y1:
            raise ValueError(f"conditional y1 score receipts repeat y1 bin {y1}")
        by_y1[y1] = score
    expected_by_y1 = {item.y1.value: item for item in expected}
    missing = sorted(set(expected_by_y1).difference(by_y1))
    extra = sorted(set(by_y1).difference(expected_by_y1))
    if missing or extra:
        raise ValueError(
            "conditional y1 score receipts must cover every declared y1 bin exactly once "
            f"(missing={missing}, extra={extra})"
        )
    ordered: list[ConditionalY1ScoreReceipt] = []
    for y1, expected_entry in sorted(expected_by_y1.items()):
        score = by_y1[y1]
        if (
            score.can_form_valid_box != expected_entry.can_form_valid_box
            or score.invalid_box_reason != expected_entry.invalid_box_reason
        ):
            raise ValueError(f"conditional y1 score receipt has invalid-box audit mismatch at y1={y1}")
        ordered.append(score)
    return tuple(ordered)


def _conditional_y1_receipt_digest(
    x1: CoordinateBin, scores: Sequence[ConditionalY1ScoreReceipt]
) -> str:
    payload = {
        "x1": x1.value,
        "scores": [
            {
                "can_form_valid_box": score.can_form_valid_box,
                "invalid_box_reason": score.invalid_box_reason,
                "raw_selected_token_logprob": score.raw_selected_token_logprob,
                "y1": score.y1.value,
            }
            for score in scores
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _conditional_y1_completeness_digest(
    *,
    attestation_kind: str,
    diagnostic_owner_id: str,
    gt_owner_id: str,
    image_identity: str,
    gt_box: CoordinateBox,
    canonical_description_text: str,
    canonical_description_token_digest: str,
    context_id: str,
    context_token_digest: str,
    tokenizer_identity: str,
    model_identity: str,
    runtime_identity: str,
    geometry_identity_schema: str,
    rule_digest: str,
    declared_x1_bins: Sequence[int],
    per_x1_receipt_digests: Sequence[tuple[int, str]],
) -> str:
    payload = {
        "attestation_kind": attestation_kind,
        "canonical_description_text": canonical_description_text,
        "canonical_description_token_digest": canonical_description_token_digest,
        "context_id": context_id,
        "context_token_digest": context_token_digest,
        "declared_x1_bins": list(declared_x1_bins),
        "diagnostic_owner_id": diagnostic_owner_id,
        "geometry_identity_schema": geometry_identity_schema,
        "gt_box": list(gt_box.as_tuple()),
        "gt_owner_id": gt_owner_id,
        "image_identity": image_identity,
        "model_identity": model_identity,
        "per_x1_receipt_digests": [list(item) for item in per_x1_receipt_digests],
        "rule_digest": rule_digest,
        "runtime_identity": runtime_identity,
        "tokenizer_identity": tokenizer_identity,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def attest_complete_conditional_y1_scores(
    *,
    diagnostic_owner_id: str,
    gt_owner_id: str,
    image_identity: str,
    context_id: str,
    canonical_description_text: str,
    canonical_description_token_digest: str,
    context_token_digest: str,
    tokenizer_identity: str,
    model_identity: str,
    runtime_identity: str,
    gt_box: CoordinateBox,
    scores_by_x1: Mapping[CoordinateBin, Sequence[ConditionalY1ScoreReceipt]],
    rules: LandscapeRules,
) -> ConditionalY1CompletenessAttestation:
    """Validate and bind every admitted x1's complete conditional y1 receipt.

    A decision-bearing landscape cannot stand on selected y1 rows alone.  This
    attestation commits to the full target x1 set, every complete y1 row, its
    exact context, and the frozen rules used to declare the vocabulary domain.
    """

    diagnostic_owner = _require_nonempty_string(
        diagnostic_owner_id, "conditional y1 diagnostic owner ID"
    )
    gt_owner = _require_nonempty_string(gt_owner_id, "conditional y1 GT owner ID")
    image = _require_nonempty_string(image_identity, "conditional y1 image identity")
    context = _require_nonempty_string(context_id, "conditional y1 context_id")
    description = _require_nonempty_string(
        canonical_description_text, "conditional y1 canonical description text"
    )
    description_tokens = _require_sha256_digest(
        canonical_description_token_digest,
        "conditional y1 canonical description token digest",
    )
    context_tokens = _require_sha256_digest(
        context_token_digest, "conditional y1 exact context-token digest"
    )
    tokenizer = _require_nonempty_string(tokenizer_identity, "conditional y1 tokenizer identity")
    model = _require_nonempty_string(model_identity, "conditional y1 model identity")
    runtime = _require_nonempty_string(runtime_identity, "conditional y1 runtime identity")
    _validate_box_in_rules(gt_box, rules, "conditional y1 ground-truth box")
    admitted = enumerate_target_x1_anchors(gt_box, rules)
    expected_keys = set(admitted)
    observed_keys = set(scores_by_x1)
    if observed_keys != expected_keys:
        missing = sorted(item.value for item in expected_keys.difference(observed_keys))
        extra = sorted(
            item.value if isinstance(item, CoordinateBin) else str(item)
            for item in observed_keys.difference(expected_keys)
        )
        raise ValueError(
            "conditional y1 completeness must cover exactly the declared x1 anchors "
            f"(missing={missing}, extra={extra})"
        )
    per_x1: list[tuple[int, str]] = []
    for x1 in admitted:
        validated = validate_complete_conditional_y1_scores(
            x1=x1, gt_box=gt_box, scores=scores_by_x1[x1], rules=rules
        )
        per_x1.append((x1.value, _conditional_y1_receipt_digest(x1, validated)))
    declared_x1 = tuple(item.value for item in admitted)
    digest = _conditional_y1_completeness_digest(
        attestation_kind=rules.contract_mode,
        diagnostic_owner_id=diagnostic_owner,
        gt_owner_id=gt_owner,
        image_identity=image,
        gt_box=gt_box,
        canonical_description_text=description,
        canonical_description_token_digest=description_tokens,
        context_id=context,
        context_token_digest=context_tokens,
        tokenizer_identity=tokenizer,
        model_identity=model,
        runtime_identity=runtime,
        geometry_identity_schema=rules.geometry_identity_schema,
        rule_digest=rules.rule_digest,
        declared_x1_bins=declared_x1,
        per_x1_receipt_digests=per_x1,
    )
    return ConditionalY1CompletenessAttestation(
        attestation_kind=rules.contract_mode,
        diagnostic_owner_id=diagnostic_owner,
        gt_owner_id=gt_owner,
        image_identity=image,
        gt_box=gt_box,
        canonical_description_text=description,
        canonical_description_token_digest=description_tokens,
        context_id=context,
        context_token_digest=context_tokens,
        tokenizer_identity=tokenizer,
        model_identity=model,
        runtime_identity=runtime,
        geometry_identity_schema=rules.geometry_identity_schema,
        rule_digest=rules.rule_digest,
        declared_x1_bins=declared_x1,
        per_x1_receipt_digests=tuple(per_x1),
        completeness_digest=digest,
    )


def _validate_conditional_y1_completeness_attestation(
    attestation: ConditionalY1CompletenessAttestation, rules: LandscapeRules
) -> None:
    if not isinstance(attestation, ConditionalY1CompletenessAttestation):
        raise ValueError("decision-bearing basin measurements require ConditionalY1CompletenessAttestation")
    if attestation.rule_digest != rules.rule_digest:
        raise ValueError("conditional y1 completeness attestation rule digest is stale")
    if attestation.attestation_kind != rules.contract_mode:
        raise ValueError("conditional y1 completeness attestation contract mode is stale or rebound")
    if attestation.geometry_identity_schema != rules.geometry_identity_schema:
        raise ValueError("conditional y1 completeness geometry identity schema is stale or rebound")
    expected = _conditional_y1_completeness_digest(
        attestation_kind=attestation.attestation_kind,
        diagnostic_owner_id=attestation.diagnostic_owner_id,
        gt_owner_id=attestation.gt_owner_id,
        image_identity=attestation.image_identity,
        gt_box=attestation.gt_box,
        canonical_description_text=attestation.canonical_description_text,
        canonical_description_token_digest=attestation.canonical_description_token_digest,
        context_id=attestation.context_id,
        context_token_digest=attestation.context_token_digest,
        tokenizer_identity=attestation.tokenizer_identity,
        model_identity=attestation.model_identity,
        runtime_identity=attestation.runtime_identity,
        geometry_identity_schema=attestation.geometry_identity_schema,
        rule_digest=attestation.rule_digest,
        declared_x1_bins=attestation.declared_x1_bins,
        per_x1_receipt_digests=attestation.per_x1_receipt_digests,
    )
    if attestation.completeness_digest != expected:
        raise ValueError("conditional y1 completeness attestation digest is invalid or stale")


@dataclass(frozen=True)
class TargetAnchorPair:
    x1: CoordinateBin
    y1: CoordinateBin


def enumerate_target_anchor_pairs(
    gt_box: CoordinateBox, rules: LandscapeRules
) -> tuple[TargetAnchorPair, ...]:
    """Deterministic cartesian target-anchor set, sorted x1 then y1."""

    x_values = enumerate_target_x1_anchors(gt_box, rules)
    y_values = enumerate_target_y1_anchors(gt_box, rules)
    return tuple(TargetAnchorPair(x1, y1) for x1 in x_values for y1 in y_values)


@dataclass(frozen=True)
class ExtentBankSeed:
    """Caller-supplied extent candidate before deterministic expansion.

    ``anchor_translate`` preserves this seed's width and height at every target
    anchor.  ``exact`` admits the supplied complete box once.  Thus GT, part,
    whole, merged, covered-owner, background, and scan seeds have no hidden
    construction policy in the core; the frozen rules and caller provide it.
    """

    bank_name: str
    source_id: str
    box: CoordinateBox
    extent_submode: str
    expansion: Literal["anchor_translate", "exact"]
    physical_owner_hint: str | None = None

    def __post_init__(self) -> None:
        _require_nonempty_string(self.bank_name, "extent bank name")
        _require_nonempty_string(self.source_id, "extent source id")
        _require_nonempty_string(self.extent_submode, "extent submode")
        if self.expansion not in {"anchor_translate", "exact"}:
            raise ValueError("extent expansion must be 'anchor_translate' or 'exact'")
        if self.physical_owner_hint is not None:
            _require_nonempty_string(self.physical_owner_hint, "physical owner hint")


def complete_box_candidate_id(
    *, bank_name: str, source_id: str, box: CoordinateBox, anchor: TargetAnchorPair | None
) -> str:
    """A collision-free, JSON-canonical identifier for one exact complete box."""

    payload = {
        "anchor": None if anchor is None else [anchor.x1.value, anchor.y1.value],
        "bank": _require_nonempty_string(bank_name, "candidate bank name"),
        "box": list(box.as_tuple()),
        "source_id": _require_nonempty_string(source_id, "candidate source id"),
    }
    return "complete-box:" + json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass(frozen=True)
class CompleteBoxCandidate:
    candidate_id: str
    bank_name: str
    source_id: str
    box: CoordinateBox
    extent_submode: str
    proposal_measure_id: str
    anchor: TargetAnchorPair | None
    physical_owner_hint: str | None = None

    def __post_init__(self) -> None:
        _require_nonempty_string(self.candidate_id, "candidate id")
        _require_nonempty_string(self.bank_name, "candidate bank name")
        _require_nonempty_string(self.source_id, "candidate source id")
        _require_nonempty_string(self.extent_submode, "candidate extent submode")
        _require_nonempty_string(self.proposal_measure_id, "candidate proposal measure id")
        if self.physical_owner_hint is not None:
            _require_nonempty_string(self.physical_owner_hint, "candidate physical owner hint")


@dataclass(frozen=True)
class CandidateExpansionReceipt:
    bank_name: str
    source_id: str
    expansion: str
    created_candidate_ids: tuple[str, ...]
    skipped_anchor_count: int


@dataclass(frozen=True)
class ExtentBankExpansion:
    candidates: tuple[CompleteBoxCandidate, ...]
    receipts: tuple[CandidateExpansionReceipt, ...]


def _candidate_sort_key(candidate: CompleteBoxCandidate, rules: LandscapeRules) -> tuple[Any, ...]:
    return (
        rules.bank_rank(candidate.bank_name),
        candidate.box.as_tuple(),
        candidate.source_id,
        candidate.candidate_id,
    )


def _validate_candidate_under_rules(candidate: CompleteBoxCandidate, rules: LandscapeRules) -> None:
    rules.bank_rank(candidate.bank_name)
    _validate_box_in_rules(candidate.box, rules, f"candidate {candidate.candidate_id!r}")
    if candidate.extent_submode not in rules.declared_extent_submodes:
        raise ValueError(
            f"candidate {candidate.candidate_id!r} has undeclared extent submode "
            f"{candidate.extent_submode!r}"
        )
    expected_measure = rules.proposal_measure_for_bank(candidate.bank_name).measure_id
    if candidate.proposal_measure_id != expected_measure:
        raise ValueError(
            f"candidate {candidate.candidate_id!r} proposal_measure_id "
            f"{candidate.proposal_measure_id!r} does not match bank {candidate.bank_name!r} "
            f"rule {expected_measure!r}"
        )


def expand_extent_banks(
    anchors: Sequence[TargetAnchorPair],
    seeds: Sequence[ExtentBankSeed],
    rules: LandscapeRules,
) -> ExtentBankExpansion:
    """Expand caller-supplied banks into exact valid complete-box candidates."""

    sorted_anchors = tuple(sorted(set(anchors), key=lambda item: (item.x1.value, item.y1.value)))
    for anchor in sorted_anchors:
        if not rules.coordinate_min <= anchor.x1.value < rules.coordinate_max:
            raise ValueError("target anchor x1 is outside the usable coordinate range")
        if not rules.coordinate_min <= anchor.y1.value < rules.coordinate_max:
            raise ValueError("target anchor y1 is outside the usable coordinate range")
    sorted_seeds = tuple(
        sorted(
            seeds,
            key=lambda seed: (
                rules.bank_rank(seed.bank_name),
                seed.source_id,
                seed.box.as_tuple(),
                seed.extent_submode,
                seed.expansion,
            ),
        )
    )
    candidates: list[CompleteBoxCandidate] = []
    receipts: list[CandidateExpansionReceipt] = []
    seen_ids: set[str] = set()
    for seed in sorted_seeds:
        rules.bank_rank(seed.bank_name)
        _validate_box_in_rules(seed.box, rules, f"extent seed {seed.source_id!r}")
        if seed.extent_submode not in rules.declared_extent_submodes:
            raise ValueError(
                f"extent seed {seed.source_id!r} has undeclared extent submode "
                f"{seed.extent_submode!r}"
            )
        measure = rules.proposal_measure_for_bank(seed.bank_name)
        created: list[str] = []
        skipped = 0
        placements: Sequence[TargetAnchorPair | None]
        placements = sorted_anchors if seed.expansion == "anchor_translate" else (None,)
        for anchor in placements:
            if anchor is None:
                candidate_box = seed.box
            else:
                try:
                    candidate_box = CoordinateBox.from_values(
                        anchor.x1.value,
                        anchor.y1.value,
                        anchor.x1.value + seed.box.width,
                        anchor.y1.value + seed.box.height,
                    )
                except ValueError:
                    skipped += 1
                    continue
            if (
                candidate_box.x1.value < rules.coordinate_min
                or candidate_box.y1.value < rules.coordinate_min
                or candidate_box.x2.value > rules.coordinate_max
                or candidate_box.y2.value > rules.coordinate_max
            ):
                skipped += 1
                continue
            candidate_id = complete_box_candidate_id(
                bank_name=seed.bank_name,
                source_id=seed.source_id,
                box=candidate_box,
                anchor=anchor,
            )
            if candidate_id in seen_ids:
                raise ValueError(f"duplicate complete-box candidate id {candidate_id!r}")
            seen_ids.add(candidate_id)
            candidate = CompleteBoxCandidate(
                candidate_id=candidate_id,
                bank_name=seed.bank_name,
                source_id=seed.source_id,
                box=candidate_box,
                extent_submode=seed.extent_submode,
                proposal_measure_id=measure.measure_id,
                anchor=anchor,
                physical_owner_hint=seed.physical_owner_hint,
            )
            candidates.append(candidate)
            created.append(candidate_id)
        receipts.append(
            CandidateExpansionReceipt(
                bank_name=seed.bank_name,
                source_id=seed.source_id,
                expansion=seed.expansion,
                created_candidate_ids=tuple(sorted(created)),
                skipped_anchor_count=skipped,
            )
        )
    return ExtentBankExpansion(
        candidates=tuple(sorted(candidates, key=lambda item: _candidate_sort_key(item, rules))),
        receipts=tuple(receipts),
    )


@dataclass(frozen=True)
class PolicyCoordinateScore:
    """A decoding-policy score, explicitly not a raw model likelihood."""

    label: Literal["non_likelihood_repetition_penalty_policy_score"]
    repetition_penalty: float
    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if self.label != POLICY_SCORE_LABEL:
            raise ValueError("policy score label must state that it is non-likelihood")
        penalty = _require_finite_number(self.repetition_penalty, "repetition penalty")
        if penalty <= 0.0:
            raise ValueError("repetition penalty must be positive")
        object.__setattr__(self, "repetition_penalty", penalty)
        for name in COORDINATE_NAMES:
            value = _require_finite_number(getattr(self, name), f"policy {name} logprob")
            if value > 1e-12:
                raise ValueError(f"policy {name} logprob must be non-positive")
            object.__setattr__(self, name, value)

    @property
    def total(self) -> float:
        return self.x1 + self.y1 + self.x2 + self.y2


@dataclass(frozen=True)
class CandidateScore:
    candidate: CompleteBoxCandidate
    raw_coordinate_logprobs: RawCoordinateLogprobs
    policy_coordinate_score: PolicyCoordinateScore | None = None

    @property
    def raw_complete_box_logprob(self) -> float:
        return self.raw_coordinate_logprobs.total


def score_complete_candidates(
    candidates: Sequence[CompleteBoxCandidate],
    raw_logprobs_by_candidate_id: Mapping[str, RawCoordinateLogprobs],
    rules: LandscapeRules,
) -> tuple[CandidateScore, ...]:
    """Join one exact raw x1/y1/x2/y2 likelihood receipt to every candidate."""

    ids = [candidate.candidate_id for candidate in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("complete candidates must have unique IDs")
    provided = set(raw_logprobs_by_candidate_id)
    missing = set(ids).difference(provided)
    extra = provided.difference(ids)
    if missing or extra:
        raise ValueError(
            "raw coordinate score IDs must exactly match complete candidates "
            f"(missing={sorted(missing)}, extra={sorted(extra)})"
        )
    result: list[CandidateScore] = []
    for candidate in sorted(candidates, key=lambda item: item.candidate_id):
        _validate_candidate_under_rules(candidate, rules)
        raw = raw_logprobs_by_candidate_id[candidate.candidate_id]
        if not isinstance(raw, RawCoordinateLogprobs):
            raise ValueError("raw candidate scores must use RawCoordinateLogprobs")
        result.append(CandidateScore(candidate, raw))
    return tuple(result)


def _logsumexp(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("logsumexp requires at least one value")
    maximum = max(values)
    if maximum == -math.inf:
        return -math.inf
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def transformers_repetition_penalty_transform(
    raw_logits: Mapping[int, float], previous_token_ids: Sequence[int], repetition_penalty: float
) -> dict[int, float]:
    """Apply the Hugging Face repetition-penalty rule to one vocabulary row.

    For each previously seen token, non-negative logits are divided by the
    penalty and negative logits are multiplied by it.  This is the behavior of
    ``transformers.RepetitionPenaltyLogitsProcessor`` for a batch item, but is
    implemented here without importing Transformers so this module stays pure.
    """

    penalty = _require_finite_number(repetition_penalty, "repetition penalty")
    if penalty <= 0.0:
        raise ValueError("repetition penalty must be positive")
    transformed: dict[int, float] = {}
    for token_id, logit in raw_logits.items():
        if not _is_int(token_id) or token_id < 0:
            raise ValueError("logit vocabulary IDs must be non-negative integers")
        transformed[int(token_id)] = _require_finite_number(logit, f"raw logit for token {token_id}")
    for token_id in set(previous_token_ids):
        if not _is_int(token_id) or token_id < 0:
            raise ValueError("previous token IDs must be non-negative integers")
        if token_id not in transformed:
            raise ValueError(f"previous token {token_id} is absent from the supplied vocabulary row")
        value = transformed[token_id]
        transformed[token_id] = value * penalty if value < 0.0 else value / penalty
    return transformed


def _log_softmax_selected(logits: Mapping[int, float], selected_token_id: int) -> float:
    if selected_token_id not in logits:
        raise ValueError(f"selected token {selected_token_id} is absent from the supplied vocabulary row")
    values = list(logits.values())
    return logits[selected_token_id] - _logsumexp(values)


def full_vocabulary_id_digest(vocabulary_size: int) -> str:
    """Digest the contiguous Transformers vocabulary ID domain ``0..V-1``."""

    if not _is_int(vocabulary_size) or vocabulary_size <= 0:
        raise ValueError("vocabulary size must be a positive integer")
    return hashlib.sha256(
        json.dumps(list(range(vocabulary_size)), separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True)
class PolicyRuntimeIdentity:
    """Frozen tokenizer/model/runtime identity supplied by the execution runner."""

    tokenizer_identity: str
    model_identity: str
    runtime_rule_digest: str

    def __post_init__(self) -> None:
        _require_nonempty_string(self.tokenizer_identity, "policy tokenizer identity")
        _require_nonempty_string(self.model_identity, "policy model identity")
        _require_nonempty_string(self.runtime_rule_digest, "policy runtime rule digest")


@dataclass(frozen=True)
class FullVocabularyAttestation:
    """Attests a full vocabulary row for one frozen tokenizer/model/runtime identity."""

    expected_vocabulary_size: int
    contiguous_token_id_digest: str
    tokenizer_identity: str
    model_identity: str
    runtime_rule_digest: str

    def __post_init__(self) -> None:
        if not _is_int(self.expected_vocabulary_size) or self.expected_vocabulary_size <= 0:
            raise ValueError("expected vocabulary size must be a positive integer")
        expected = full_vocabulary_id_digest(self.expected_vocabulary_size)
        if self.contiguous_token_id_digest != expected:
            raise ValueError(
                "full-vocabulary attestation digest does not match the contiguous expected ID domain"
            )
        _require_nonempty_string(self.tokenizer_identity, "full-vocabulary tokenizer identity")
        _require_nonempty_string(self.model_identity, "full-vocabulary model identity")
        _require_nonempty_string(self.runtime_rule_digest, "full-vocabulary runtime rule digest")


def _validate_full_vocabulary_row(
    row: Mapping[int, float], attestation: FullVocabularyAttestation, row_index: int
) -> None:
    expected_size = attestation.expected_vocabulary_size
    if any(not _is_int(token_id) or token_id < 0 for token_id in row):
        raise ValueError(f"policy vocabulary row {row_index} has invalid token IDs")
    token_ids = sorted(row)
    if len(token_ids) != expected_size or token_ids != list(range(expected_size)):
        raise ValueError(
            f"policy vocabulary row {row_index} is not the attested full contiguous vocabulary "
            f"of size {expected_size}"
        )
    observed_digest = hashlib.sha256(
        json.dumps(token_ids, separators=(",", ":")).encode()
    ).hexdigest()
    if observed_digest != attestation.contiguous_token_id_digest:
        raise ValueError(f"policy vocabulary row {row_index} does not match the attested token-ID digest")


def repetition_penalty_policy_score(
    *,
    raw_step_logits: Sequence[Mapping[int, float]],
    coordinate_token_ids: Sequence[int],
    prefix_token_ids: Sequence[int],
    repetition_penalty: float,
    vocabulary_attestation: FullVocabularyAttestation,
    runtime_identity: PolicyRuntimeIdentity,
    rules: LandscapeRules,
) -> PolicyCoordinateScore:
    """Derive four policy scores from raw logits while retaining their label.

    ``raw_step_logits[i]`` must be the raw full-vocabulary row obtained after
    forcing the description and coordinate prefix through coordinate ``i - 1``.
    The returned value must never be substituted for model likelihood.
    """

    if len(raw_step_logits) != 4 or len(coordinate_token_ids) != 4:
        raise ValueError("policy scoring requires exactly four coordinate vocabulary rows and token IDs")
    if not isinstance(vocabulary_attestation, FullVocabularyAttestation):
        raise ValueError("policy scoring requires a FullVocabularyAttestation")
    if not isinstance(runtime_identity, PolicyRuntimeIdentity):
        raise ValueError("policy scoring requires a PolicyRuntimeIdentity runner receipt")
    if runtime_identity.runtime_rule_digest != rules.rule_digest:
        raise ValueError("policy runner receipt runtime rule digest does not match frozen rules")
    if (
        vocabulary_attestation.tokenizer_identity != runtime_identity.tokenizer_identity
        or vocabulary_attestation.model_identity != runtime_identity.model_identity
        or vocabulary_attestation.runtime_rule_digest != runtime_identity.runtime_rule_digest
    ):
        raise ValueError(
            "full-vocabulary attestation tokenizer/model/runtime identity does not match "
            "the runner receipt"
        )
    history = list(prefix_token_ids)
    values: list[float] = []
    for index, (row, selected) in enumerate(zip(raw_step_logits, coordinate_token_ids)):
        if not _is_int(selected) or selected < 0:
            raise ValueError("coordinate token IDs must be non-negative integers")
        _validate_full_vocabulary_row(row, vocabulary_attestation, index)
        transformed = transformers_repetition_penalty_transform(row, history, repetition_penalty)
        values.append(_log_softmax_selected(transformed, selected))
        history.append(selected)
    return PolicyCoordinateScore(POLICY_SCORE_LABEL, repetition_penalty, *values)


def _intersection_over_union(left: CoordinateBox, right: CoordinateBox) -> float:
    x1 = max(left.x1.value, right.x1.value)
    y1 = max(left.y1.value, right.y1.value)
    x2 = min(left.x2.value, right.x2.value)
    y2 = min(left.y2.value, right.y2.value)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = left.area + right.area - intersection
    return intersection / union


def _center_distance(left: CoordinateBox, right: CoordinateBox) -> float:
    left_x, left_y = left.center
    right_x, right_y = right.center
    return math.hypot(left_x - right_x, left_y - right_y)


class _DisjointSet:
    def __init__(self, values: Sequence[str]) -> None:
        self._parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self._parent[value]
        if parent != value:
            self._parent[value] = self.find(parent)
        return self._parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self._parent[max(left_root, right_root)] = min(left_root, right_root)


def _components(
    candidates: Sequence[CandidateScore], link: Any, *, hard_owner_hint_separation: bool = False
) -> tuple[tuple[CandidateScore, ...], ...]:
    ordered = tuple(sorted(candidates, key=lambda item: item.candidate.candidate_id))
    disjoint = _DisjointSet([item.candidate.candidate_id for item in ordered])
    component_hints: dict[str, set[str]] = {
        item.candidate.candidate_id: (
            set()
            if item.candidate.physical_owner_hint is None
            else {item.candidate.physical_owner_hint}
        )
        for item in ordered
    }
    for index, left in enumerate(ordered):
        for right in ordered[index + 1 :]:
            if link(left.candidate.box, right.candidate.box):
                combined_hints: set[str] = set()
                if hard_owner_hint_separation:
                    left_root = disjoint.find(left.candidate.candidate_id)
                    right_root = disjoint.find(right.candidate.candidate_id)
                    combined_hints = component_hints[left_root].union(component_hints[right_root])
                    # A null-hint bridge may attach to the first deterministic
                    # hint component it encounters, but can never transitively
                    # merge two known physical owners.
                    if len(combined_hints) > 1:
                        continue
                disjoint.union(left.candidate.candidate_id, right.candidate.candidate_id)
                if hard_owner_hint_separation:
                    merged_root = disjoint.find(left.candidate.candidate_id)
                    component_hints[merged_root] = combined_hints
    groups: dict[str, list[CandidateScore]] = {}
    for item in ordered:
        groups.setdefault(disjoint.find(item.candidate.candidate_id), []).append(item)
    return tuple(
        tuple(sorted(group, key=lambda item: item.candidate.candidate_id))
        for _, group in sorted(groups.items())
    )


def _physical_component_sort_key(component: Sequence[CandidateScore]) -> tuple[Any, ...]:
    return min(
        (
            item.candidate.box.y1.value,
            item.candidate.box.x1.value,
            item.candidate.box.y2.value,
            item.candidate.box.x2.value,
            item.candidate.candidate_id,
        )
        for item in component
    )


@dataclass(frozen=True)
class ExtentSubmodeCluster:
    submode_id: str
    candidate_ids: tuple[str, ...]


@dataclass(frozen=True)
class PhysicalBasinCluster:
    basin_id: str
    candidate_ids: tuple[str, ...]
    extent_submodes: tuple[ExtentSubmodeCluster, ...]
    reviewed_physical_owner_id: str | None
    owner_identity_status: Literal["reviewed", "unresolved"]


def cluster_physical_basins(
    candidate_scores: Sequence[CandidateScore], rules: LandscapeRules
) -> tuple[PhysicalBasinCluster, ...]:
    """Cluster physical geometry first, then split each basin into extent modes.

    Distinct supplied ``physical_owner_hint`` values are hard boundaries, even
    where their crowded boxes overlap or an unhinted candidate would otherwise
    create an IoU/center-distance chain.  With no hints, the geometry route is
    intentionally deterministic single-link clustering: a chain of overlapping
    boxes can form one physical candidate basin and therefore remains a review
    limitation rather than an implicit owner truth.
    """

    if not candidate_scores:
        return ()
    seen_ids: set[str] = set()
    for item in candidate_scores:
        if item.candidate.candidate_id in seen_ids:
            raise ValueError("candidate scores must have unique candidate IDs")
        seen_ids.add(item.candidate.candidate_id)
        _validate_candidate_under_rules(item.candidate, rules)
    threshold = rules.spatial_clustering
    physical_components = _components(
        candidate_scores,
        lambda left, right: _intersection_over_union(left, right) >= threshold.owner_link_iou_min
        or _center_distance(left, right) <= threshold.owner_link_center_distance_max,
        hard_owner_hint_separation=True,
    )
    clusters: list[PhysicalBasinCluster] = []
    for basin_index, component in enumerate(
        sorted(physical_components, key=_physical_component_sort_key), start=1
    ):
        basin_id = f"physical-basin-{basin_index:03d}"
        # Extent labels are frozen semantics, so never allow a high-IoU face,
        # torso, whole, or merged seed to erase a declared submode boundary.
        submode_components: list[tuple[CandidateScore, ...]] = []
        for label in rules.declared_extent_submodes:
            same_label = tuple(
                item for item in component if item.candidate.extent_submode == label
            )
            if same_label:
                submode_components.extend(
                    _components(
                        same_label,
                        lambda left, right: _intersection_over_union(left, right)
                        >= threshold.extent_submode_iou_min,
                    )
                )
        ordered_submodes = sorted(submode_components, key=_physical_component_sort_key)
        submodes = tuple(
            ExtentSubmodeCluster(
                submode_id=f"{basin_id}:extent-submode-{submode_index:03d}",
                candidate_ids=tuple(item.candidate.candidate_id for item in submode),
            )
            for submode_index, submode in enumerate(ordered_submodes, start=1)
        )
        owner_hints = {item.candidate.physical_owner_hint for item in component}
        reviewed_owner_ids = {hint for hint in owner_hints if hint is not None}
        if len(reviewed_owner_ids) == 1 and None not in owner_hints:
            reviewed_physical_owner_id = next(iter(reviewed_owner_ids))
            owner_status: Literal["reviewed", "unresolved"] = "reviewed"
        else:
            reviewed_physical_owner_id = None
            owner_status = "unresolved"
        clusters.append(
            PhysicalBasinCluster(
                basin_id=basin_id,
                candidate_ids=tuple(item.candidate.candidate_id for item in component),
                extent_submodes=submodes,
                reviewed_physical_owner_id=reviewed_physical_owner_id,
                owner_identity_status=owner_status,
            )
        )
    return tuple(clusters)


def _score_map(candidate_scores: Sequence[CandidateScore]) -> dict[str, CandidateScore]:
    result = {item.candidate.candidate_id: item for item in candidate_scores}
    if len(result) != len(candidate_scores):
        raise ValueError("candidate scores must have unique IDs")
    return result


def count_distinct_physical_basins_in_top_n(
    candidate_scores: Sequence[CandidateScore],
    clusters: Sequence[PhysicalBasinCluster],
    top_n: int,
) -> int:
    """Count owner-level peaks, rather than treating a token top-N as owners."""

    if not _is_int(top_n) or top_n <= 0:
        raise ValueError("top_n must be a positive integer")
    score_map = _score_map(candidate_scores)
    candidate_to_basin: dict[str, str] = {}
    for cluster in clusters:
        for candidate_id in cluster.candidate_ids:
            if candidate_id in candidate_to_basin:
                raise ValueError("a candidate belongs to more than one physical basin")
            candidate_to_basin[candidate_id] = cluster.basin_id
    if set(candidate_to_basin) != set(score_map):
        raise ValueError("clusters must cover exactly the scored candidates")
    ranked = sorted(
        candidate_scores,
        key=lambda item: (-item.raw_complete_box_logprob, item.candidate.candidate_id),
    )
    selected_basin_ids = {candidate_to_basin[item.candidate.candidate_id] for item in ranked[:top_n]}
    clusters_by_id = {cluster.basin_id: cluster for cluster in clusters}
    unresolved = [
        basin_id
        for basin_id in sorted(selected_basin_ids)
        if clusters_by_id[basin_id].owner_identity_status != "reviewed"
    ]
    if unresolved:
        raise ValueError(
            "top-N physical-owner counting cannot use unresolved physical basins: "
            f"{unresolved}"
        )
    return len(selected_basin_ids)


@dataclass(frozen=True)
class BasinMeasurement:
    basin_id: str
    role_id: str
    role_kind: Literal["target", "foil"]
    identity_kind: Literal["reviewed_physical_owner", "registered_geometry"]
    reviewed_physical_owner_id: str | None
    registered_geometry_id: str | None
    candidate_bank_names: tuple[str, ...]
    context_id: str
    foil_set_id: str
    rule_digest: str
    conditional_y1_completeness_digest: str
    candidate_ids: tuple[str, ...]
    mass_unique_box_candidate_ids: tuple[str, ...]
    extent_submode_count: int
    peak_candidate_id: str
    peak_height: float
    normalized_basin_log_mass: float
    normalized_basin_mass: float
    proposal_measure_id: str
    proposal_comparability_group: str
    proposal_domain_candidate_ids: tuple[str, ...]
    proposal_domain_log_weight_normalizer: float
    shape: str


@dataclass(frozen=True)
class BasinRegistration:
    """Frozen identity binding required before a basin becomes decision-bearing."""

    basin_id: str
    role_id: str
    identity_kind: Literal["reviewed_physical_owner", "registered_geometry"]
    reviewed_physical_owner_id: str | None
    registered_geometry_id: str | None
    context_id: str
    foil_set_id: str
    rule_digest: str
    conditional_y1_completeness_digest: str

    def __post_init__(self) -> None:
        _require_nonempty_string(self.basin_id, "basin registration basin_id")
        _require_nonempty_string(self.role_id, "basin registration role_id")
        if self.identity_kind == "reviewed_physical_owner":
            _require_nonempty_string(
                self.reviewed_physical_owner_id,
                "basin registration reviewed_physical_owner_id",
            )
            if self.registered_geometry_id is not None:
                raise ValueError("reviewed physical-owner registration cannot also carry geometry ID")
        elif self.identity_kind == "registered_geometry":
            _require_nonempty_string(
                self.registered_geometry_id, "basin registration registered_geometry_id"
            )
            if self.reviewed_physical_owner_id is not None:
                raise ValueError("owner-neutral geometry registration cannot carry physical-owner ID")
        else:
            raise ValueError(
                "basin registration identity_kind must be reviewed_physical_owner or registered_geometry"
            )
        _require_nonempty_string(self.context_id, "basin registration context_id")
        _require_nonempty_string(self.foil_set_id, "basin registration foil_set_id")
        _require_nonempty_string(self.rule_digest, "basin registration rule_digest")
        _require_nonempty_string(
            self.conditional_y1_completeness_digest,
            "basin registration conditional_y1_completeness_digest",
        )


@dataclass(frozen=True)
class ProposalDomainBoxReceipt:
    """One unique coordinate box and all candidate-source multiplicity behind it."""

    box: CoordinateBox
    canonical_candidate_id: str
    multiplicity_candidate_ids: tuple[str, ...]
    proposal_weight: float
    raw_complete_box_logprob: float


@dataclass(frozen=True)
class ProposalDomainReceipt:
    """One full candidate domain used to normalize a declared proposal measure."""

    proposal_measure_id: str
    proposal_comparability_group: str
    candidate_ids: tuple[str, ...]
    unique_coordinate_boxes: tuple[ProposalDomainBoxReceipt, ...]
    raw_total_weight: float
    log_weight_normalizer: float


def proposal_domain_receipts(
    candidate_scores: Sequence[CandidateScore], rules: LandscapeRules
) -> tuple[ProposalDomainReceipt, ...]:
    """Freeze each measure's full compared domain before basin mass is summed.

    Normalization is deliberately over every scored candidate governed by the
    measure, not separately inside an owner basin.  This preserves the mass of
    a wider or multi-mode basin under the declared proposal measure.
    """

    score_map = _score_map(candidate_scores)
    by_measure: dict[str, list[CandidateScore]] = {}
    for item in score_map.values():
        _validate_candidate_under_rules(item.candidate, rules)
        by_measure.setdefault(item.candidate.proposal_measure_id, []).append(item)
    receipts: list[ProposalDomainReceipt] = []
    for measure_id, members in sorted(by_measure.items()):
        measures = [measure for measure in rules.proposal_measures if measure.measure_id == measure_id]
        if len(measures) != 1:
            raise ValueError(f"candidate references unknown proposal measure {measure_id!r}")
        measure = measures[0]
        ordered = tuple(sorted(members, key=lambda item: item.candidate.candidate_id))
        by_box: dict[tuple[int, int, int, int], list[CandidateScore]] = {}
        for item in ordered:
            by_box.setdefault(item.candidate.box.as_tuple(), []).append(item)
        unique_boxes: list[ProposalDomainBoxReceipt] = []
        for box_key, box_members in sorted(by_box.items()):
            canonical = min(box_members, key=lambda item: item.candidate.candidate_id)
            coordinate_score_signatures = {
                item.raw_coordinate_logprobs.as_tuple() for item in box_members
            }
            if len(coordinate_score_signatures) != 1:
                raise ValueError(
                    "duplicate exact coordinate box has inconsistent raw selected-token logprobs "
                    f"under proposal measure {measure_id!r}: {box_key}"
                )
            weights = {measure.weight_for(item.candidate.bank_name) for item in box_members}
            if len(weights) != 1:
                raise ValueError(
                    "duplicate exact coordinate box has unequal proposal weights "
                    f"under proposal measure {measure_id!r}: {box_key}"
                )
            unique_boxes.append(
                ProposalDomainBoxReceipt(
                    box=canonical.candidate.box,
                    canonical_candidate_id=canonical.candidate.candidate_id,
                    multiplicity_candidate_ids=tuple(
                        item.candidate.candidate_id
                        for item in sorted(box_members, key=lambda item: item.candidate.candidate_id)
                    ),
                    proposal_weight=next(iter(weights)),
                    raw_complete_box_logprob=canonical.raw_complete_box_logprob,
                )
            )
        log_weights = [math.log(item.proposal_weight) for item in unique_boxes]
        log_normalizer = _logsumexp(log_weights)
        receipts.append(
            ProposalDomainReceipt(
                proposal_measure_id=measure.measure_id,
                proposal_comparability_group=measure.comparability_group,
                candidate_ids=tuple(item.candidate.candidate_id for item in ordered),
                unique_coordinate_boxes=tuple(unique_boxes),
                raw_total_weight=sum(item.proposal_weight for item in unique_boxes),
                log_weight_normalizer=log_normalizer,
            )
        )
    return tuple(receipts)


def _measure_for_basin(
    cluster: PhysicalBasinCluster, score_map: Mapping[str, CandidateScore], rules: LandscapeRules
) -> ProposalMeasure:
    measure_ids = {score_map[candidate_id].candidate.proposal_measure_id for candidate_id in cluster.candidate_ids}
    if len(measure_ids) != 1:
        raise ValueError(
            f"basin {cluster.basin_id!r} mixes unequal proposal measures: {sorted(measure_ids)}"
        )
    measure_id = next(iter(measure_ids))
    for measure in rules.proposal_measures:
        if measure.measure_id == measure_id:
            return measure
    raise ValueError(f"basin {cluster.basin_id!r} references unknown proposal measure {measure_id!r}")


def _classify_shape(
    cluster: PhysicalBasinCluster,
    score_map: Mapping[str, CandidateScore],
    peak_height: float,
    rules: LandscapeRules,
) -> str:
    near_peak = [
        score_map[candidate_id]
        for candidate_id in cluster.candidate_ids
        if score_map[candidate_id].raw_complete_box_logprob
        >= peak_height - rules.shape.near_peak_logprob_delta
    ]
    near_submodes: set[str] = set()
    for submode in cluster.extent_submodes:
        if set(submode.candidate_ids).intersection(item.candidate.candidate_id for item in near_peak):
            near_submodes.add(submode.submode_id)
    if any(item.candidate.extent_submode in rules.shape.merged_extent_submodes for item in near_peak):
        return "merged_extent"
    if any(item.candidate.bank_name in rules.shape.scan_bank_names for item in near_peak):
        return "scan_direction_ridge"
    if len(near_submodes) >= rules.shape.multi_submode_min:
        return "part_or_whole_lobes"
    if len(near_peak) >= rules.shape.wide_ridge_min_candidates:
        return "wide_ridge"
    return "localized_peak"


def _validated_basin_registrations(
    clusters: Sequence[PhysicalBasinCluster],
    registrations: Sequence[BasinRegistration],
    conditional_y1_attestation: ConditionalY1CompletenessAttestation,
    score_map: Mapping[str, CandidateScore],
    rules: LandscapeRules,
) -> dict[str, tuple[BasinRegistration, RegisteredBasinRole]]:
    by_basin: dict[str, BasinRegistration] = {}
    for registration in registrations:
        if not isinstance(registration, BasinRegistration):
            raise ValueError("basin registrations must use BasinRegistration")
        if registration.basin_id in by_basin:
            raise ValueError(f"duplicate basin registration for {registration.basin_id!r}")
        by_basin[registration.basin_id] = registration
    cluster_ids = {cluster.basin_id for cluster in clusters}
    if set(by_basin) != cluster_ids:
        raise ValueError(
            "basin registrations must cover exactly the physical clusters "
            f"(missing={sorted(cluster_ids.difference(by_basin))}, "
            f"extra={sorted(set(by_basin).difference(cluster_ids))})"
        )
    validated: dict[str, tuple[BasinRegistration, RegisteredBasinRole]] = {}
    for cluster in clusters:
        registration = by_basin[cluster.basin_id]
        if registration.rule_digest != rules.rule_digest:
            raise ValueError(f"basin registration rule digest does not match frozen rules for {cluster.basin_id!r}")
        if registration.context_id != conditional_y1_attestation.context_id:
            raise ValueError(
                f"basin registration context does not match conditional y1 completeness attestation "
                f"for {cluster.basin_id!r}"
            )
        if (
            registration.conditional_y1_completeness_digest
            != conditional_y1_attestation.completeness_digest
        ):
            raise ValueError(
                f"basin registration conditional y1 completeness digest does not match "
                f"the validated attestation for {cluster.basin_id!r}"
            )
        role = rules.basin_role(registration.role_id)
        if registration.foil_set_id != role.foil_set_id:
            raise ValueError(
                f"basin registration foil-set ID does not match registered role for {cluster.basin_id!r}"
            )
        if registration.identity_kind != role.identity_kind:
            raise ValueError(
                f"basin registration identity kind does not match registered role for {cluster.basin_id!r}"
            )
        cluster_banks = {
            score_map[candidate_id].candidate.bank_name for candidate_id in cluster.candidate_ids
        }
        if not cluster_banks or not cluster_banks.issubset(role.allowed_bank_names):
            raise ValueError(
                f"basin {cluster.basin_id!r} banks {sorted(cluster_banks)} are not admitted by "
                f"registered role {role.role_id!r}"
            )
        if role.identity_kind == "reviewed_physical_owner":
            if (
                cluster.owner_identity_status != "reviewed"
                or cluster.reviewed_physical_owner_id is None
            ):
                raise ValueError(
                    f"role {role.role_id!r} requires a reviewed physical-owner binding, but "
                    f"basin {cluster.basin_id!r} is unresolved"
                )
            if registration.reviewed_physical_owner_id != cluster.reviewed_physical_owner_id:
                raise ValueError(
                    f"basin registration physical owner does not match reviewed cluster "
                    f"for {cluster.basin_id!r}"
                )
            if (
                role.kind == "target"
                and registration.reviewed_physical_owner_id
                != conditional_y1_attestation.gt_owner_id
            ):
                raise ValueError(
                    "target basin owner does not match the GT owner bound by the "
                    "conditional y1 attestation"
                )
        elif role.kind != "foil":
            raise ValueError("owner-neutral registered geometry can only be used by a foil role")
        validated[cluster.basin_id] = (registration, role)
    target_bindings = [
        (registration, role)
        for registration, role in validated.values()
        if role.kind == "target"
    ]
    covered_bindings = [
        (registration, role)
        for registration, role in validated.values()
        if "covered" in role.allowed_bank_names
    ]
    for target_registration, _ in target_bindings:
        for covered_registration, _ in covered_bindings:
            if (
                target_registration.context_id == covered_registration.context_id
                and target_registration.foil_set_id == covered_registration.foil_set_id
                and target_registration.reviewed_physical_owner_id
                == covered_registration.reviewed_physical_owner_id
            ):
                raise ValueError(
                    "target and covered-owner foil must bind distinct reviewed physical owners"
                )
    return validated


def compute_basin_measurements(
    candidate_scores: Sequence[CandidateScore],
    clusters: Sequence[PhysicalBasinCluster],
    registrations: Sequence[BasinRegistration],
    conditional_y1_attestation: ConditionalY1CompletenessAttestation,
    rules: LandscapeRules,
) -> tuple[BasinMeasurement, ...]:
    """Compute registered raw peaks and proposal mass per reviewed physical basin."""

    score_map = _score_map(candidate_scores)
    cluster_ids = {candidate_id for cluster in clusters for candidate_id in cluster.candidate_ids}
    if cluster_ids != set(score_map):
        raise ValueError("physical clusters must cover exactly the scored candidates")
    _validate_conditional_y1_completeness_attestation(conditional_y1_attestation, rules)
    bindings = _validated_basin_registrations(
        clusters, registrations, conditional_y1_attestation, score_map, rules
    )
    domains_by_measure = {
        receipt.proposal_measure_id: receipt
        for receipt in proposal_domain_receipts(candidate_scores, rules)
    }
    unique_box_by_candidate: dict[str, tuple[str, ProposalDomainBoxReceipt]] = {}
    for measure_id, domain in domains_by_measure.items():
        for unique_box in domain.unique_coordinate_boxes:
            for candidate_id in unique_box.multiplicity_candidate_ids:
                if candidate_id in unique_box_by_candidate:
                    raise AssertionError(f"candidate {candidate_id!r} appeared in multiple proposal domains")
                unique_box_by_candidate[candidate_id] = (measure_id, unique_box)
    unique_box_owners: dict[tuple[str, str], set[str]] = {}
    for cluster in clusters:
        for candidate_id in cluster.candidate_ids:
            try:
                measure_id, unique_box = unique_box_by_candidate[candidate_id]
            except KeyError as exc:
                raise AssertionError(f"candidate {candidate_id!r} is missing from its proposal domain") from exc
            unique_box_owners.setdefault((measure_id, unique_box.canonical_candidate_id), set()).add(
                cluster.basin_id
            )
    ambiguous_unique_boxes = {
        key: sorted(basin_ids)
        for key, basin_ids in unique_box_owners.items()
        if len(basin_ids) != 1
    }
    if ambiguous_unique_boxes:
        raise ValueError(
            "one exact coordinate box cannot support multiple physical basins under one proposal measure: "
            f"{ambiguous_unique_boxes}"
        )
    measurements: list[BasinMeasurement] = []
    for cluster in sorted(clusters, key=lambda item: item.basin_id):
        if not cluster.candidate_ids:
            raise ValueError("a physical basin must contain candidates")
        measure = _measure_for_basin(cluster, score_map, rules)
        members = [score_map[candidate_id] for candidate_id in cluster.candidate_ids]
        registration, role = bindings[cluster.basin_id]
        peak = min(
            members,
            key=lambda item: (-item.raw_complete_box_logprob, item.candidate.candidate_id),
        )
        try:
            domain = domains_by_measure[measure.measure_id]
        except KeyError as exc:
            raise AssertionError(f"missing full proposal domain for {measure.measure_id!r}") from exc
        unique_box_ids = tuple(
            sorted(
                {
                    unique_box_by_candidate[item.candidate.candidate_id][1].canonical_candidate_id
                    for item in members
                }
            )
        )
        unique_boxes = [
            next(
                box
                for box in domain.unique_coordinate_boxes
                if box.canonical_candidate_id == candidate_id
            )
            for candidate_id in unique_box_ids
        ]
        weighted_scores = [
            item.raw_complete_box_logprob + math.log(item.proposal_weight)
            for item in unique_boxes
        ]
        log_mass = _logsumexp(weighted_scores) - domain.log_weight_normalizer
        measurements.append(
            BasinMeasurement(
                basin_id=cluster.basin_id,
                role_id=registration.role_id,
                role_kind=role.kind,
                identity_kind=registration.identity_kind,
                reviewed_physical_owner_id=registration.reviewed_physical_owner_id,
                registered_geometry_id=registration.registered_geometry_id,
                candidate_bank_names=tuple(
                    sorted({item.candidate.bank_name for item in members})
                ),
                context_id=registration.context_id,
                foil_set_id=registration.foil_set_id,
                rule_digest=registration.rule_digest,
                conditional_y1_completeness_digest=(
                    registration.conditional_y1_completeness_digest
                ),
                candidate_ids=tuple(sorted(cluster.candidate_ids)),
                mass_unique_box_candidate_ids=unique_box_ids,
                extent_submode_count=len(cluster.extent_submodes),
                peak_candidate_id=peak.candidate.candidate_id,
                peak_height=peak.raw_complete_box_logprob,
                normalized_basin_log_mass=log_mass,
                normalized_basin_mass=math.exp(log_mass),
                proposal_measure_id=measure.measure_id,
                proposal_comparability_group=measure.comparability_group,
                proposal_domain_candidate_ids=domain.candidate_ids,
                proposal_domain_log_weight_normalizer=domain.log_weight_normalizer,
                shape=_classify_shape(cluster, score_map, peak.raw_complete_box_logprob, rules),
            )
        )
    return tuple(measurements)


def assert_basin_mass_comparable(measurements: Sequence[BasinMeasurement]) -> None:
    """Reject cross-basin mass arithmetic unless measures were explicitly matched."""

    if len(measurements) < 2:
        raise ValueError("basin-mass comparison requires at least two basin measurements")
    measure_ids = {item.proposal_measure_id for item in measurements}
    domains = {
        (
            item.proposal_measure_id,
            item.proposal_domain_candidate_ids,
            item.proposal_domain_log_weight_normalizer,
        )
        for item in measurements
    }
    if len(measure_ids) != 1 or len(domains) != 1:
        details = sorted(
            (
                item.basin_id,
                item.proposal_measure_id,
                item.proposal_comparability_group,
                item.proposal_domain_candidate_ids,
            )
            for item in measurements
        )
        raise ValueError(f"basin mass is not comparable under unequal proposal measures: {details}")


@dataclass(frozen=True)
class PeakProminenceReceipt:
    target_basin_id: str
    foil_basin_id: str
    target_role_id: str
    foil_role_id: str
    target_identity_kind: Literal["reviewed_physical_owner", "registered_geometry"]
    foil_identity_kind: Literal["reviewed_physical_owner", "registered_geometry"]
    target_reviewed_physical_owner_id: str | None
    foil_reviewed_physical_owner_id: str | None
    foil_registered_geometry_id: str | None
    context_id: str
    foil_set_id: str
    rule_digest: str
    conditional_y1_completeness_digest: str
    functional: Literal["peak_height_difference"]
    target_peak_height: float
    foil_peak_height: float
    peak_prominence: float


def compute_peak_prominence(
    target: BasinMeasurement,
    foil: BasinMeasurement,
    conditional_y1_attestation: ConditionalY1CompletenessAttestation,
    rules: LandscapeRules,
) -> PeakProminenceReceipt:
    """Compute the frozen peak-height-difference functional without a verdict threshold."""

    if target.basin_id == foil.basin_id:
        raise ValueError("peak prominence requires distinct target and foil basins")
    if rules.prominence_functional != "peak_height_difference":
        raise ValueError("unsupported frozen prominence functional")
    _validate_conditional_y1_completeness_attestation(conditional_y1_attestation, rules)
    for measurement in (target, foil):
        if measurement.rule_digest != rules.rule_digest:
            raise ValueError("peak prominence measurement rule digest does not match frozen rules")
        role = rules.basin_role(measurement.role_id)
        if role.kind != measurement.role_kind or role.foil_set_id != measurement.foil_set_id:
            raise ValueError("peak prominence measurement has an unregistered role or foil-set binding")
        if role.identity_kind != measurement.identity_kind:
            raise ValueError("peak prominence measurement identity kind does not match registered role")
        if not set(measurement.candidate_bank_names).issubset(role.allowed_bank_names):
            raise ValueError("peak prominence measurement banks do not match registered role")
        if measurement.context_id != conditional_y1_attestation.context_id:
            raise ValueError("peak prominence measurement context does not match conditional y1 completeness attestation")
        if (
            measurement.conditional_y1_completeness_digest
            != conditional_y1_attestation.completeness_digest
        ):
            raise ValueError(
                "peak prominence measurement conditional y1 completeness digest does not match "
                "the validated attestation"
            )
    if target.role_kind != "target" or foil.role_kind != "foil":
        raise ValueError("peak prominence requires one registered target basin and one registered foil basin")
    if (
        target.identity_kind != "reviewed_physical_owner"
        or target.reviewed_physical_owner_id is None
    ):
        raise ValueError("peak prominence target requires a reviewed physical owner")
    if foil.identity_kind == "reviewed_physical_owner":
        if foil.reviewed_physical_owner_id is None:
            raise ValueError("reviewed-owner foil is missing its physical owner")
        if foil.reviewed_physical_owner_id == target.reviewed_physical_owner_id:
            raise ValueError("target and covered-owner foil must bind distinct reviewed physical owners")
    elif foil.registered_geometry_id is None:
        raise ValueError("owner-neutral foil is missing its registered geometry identity")
    if target.context_id != foil.context_id:
        raise ValueError("peak prominence requires target and foil from the same exact context")
    if target.foil_set_id != foil.foil_set_id:
        raise ValueError("peak prominence requires target and foil from the same frozen foil set")
    return PeakProminenceReceipt(
        target_basin_id=target.basin_id,
        foil_basin_id=foil.basin_id,
        target_role_id=target.role_id,
        foil_role_id=foil.role_id,
        target_identity_kind=target.identity_kind,
        foil_identity_kind=foil.identity_kind,
        target_reviewed_physical_owner_id=target.reviewed_physical_owner_id,
        foil_reviewed_physical_owner_id=foil.reviewed_physical_owner_id,
        foil_registered_geometry_id=foil.registered_geometry_id,
        context_id=target.context_id,
        foil_set_id=target.foil_set_id,
        rule_digest=target.rule_digest,
        conditional_y1_completeness_digest=target.conditional_y1_completeness_digest,
        functional="peak_height_difference",
        target_peak_height=target.peak_height,
        foil_peak_height=foil.peak_height,
        peak_prominence=target.peak_height - foil.peak_height,
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, CoordinateBin):
        return value.value
    if isinstance(value, CoordinateBox):
        return {"x1": value.x1.value, "y1": value.y1.value, "x2": value.x2.value, "y2": value.y2.value}
    if is_dataclass(value):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("JSON receipt contains a non-finite float")
    return value


def json_serializable_receipt(value: Any) -> dict[str, Any] | list[Any] | str | int | float | bool | None:
    """Validate and normalize a stable JSON-compatible receipt value."""

    normalized = _jsonable(value)
    # Round-trip catches unsupported leaf values and freezes deterministic key ordering
    # for callers that persist the following stable_json_dumps representation.
    return json.loads(json.dumps(normalized, sort_keys=True, ensure_ascii=False, allow_nan=False))


def stable_json_dumps(value: Any) -> str:
    """Canonical UTF-8-safe JSON encoding used for deterministic CPU receipts."""

    return json.dumps(json_serializable_receipt(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
