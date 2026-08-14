#!/usr/bin/env python3
"""Fail-closed analyzer for the complete Human-13 RP crossover matrix.

The analyzer consumes the shared Wave-5 matrix records; it does not define a
second cell or plan schema.  All public input forms are canonicalized through
``MatrixAnalysisInput.from_dict`` before any audit JSONL is opened.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from scripts.research.analyze_human13_k_union import (
    _manifest_sha256,
    _match_prefix,
    _ordered_predictions,
)
from scripts.research.build_human13_k_union_manifest import (
    Human13KUnionManifest,
    ImageRecord,
    load_manifest,
)
from scripts.research.human13_rp_crossover_matrix_contracts import (
    ARM_IDS,
    CANONICAL_IMAGE_IDS,
    DRY_RUN_COUNTER_KEYS,
    EVALUATION_RPS,
    MATRIX_SEED_GROUPS,
    PHASE_MATRIX,
    TRAINING_RPS,
    CellReceipt,
    CellSpec,
    MatrixPlan,
    NodeTerminalReceipt,
    SourceBaselineRef,
    matrix_identity,
    validate_matrix_receipts,
)


INPUT_SCHEMA = "human13_rp_crossover_analysis_input.v1"
SURFACE_SCHEMA = "human13_rp_crossover_surface_analysis.v1"
PAIRED_SCHEMA = "human13_rp_crossover_paired_change.v1"
SUCCESS_SCHEMA = "human13_rp_crossover_success.v1"
ANALYSIS_SCHEMA = "human13_rp_crossover_analysis.v1"
FROZEN_MANIFEST_SHA256 = (
    "a8f88716c1227054ab29dc698f89462c9369c47c8d6415de3783c0937f60a6fb"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CAP_STOP_REASONS = {"length", "max_tokens", "max_new_tokens"}
_NATURAL_STOP_REASONS = {"eos", "im_end"}


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _digest(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _nonempty(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a nonempty string")
    return value


def _rp(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be exactly 1.0 or 1.10")
    result = float(value)
    if result not in EVALUATION_RPS:
        raise ValueError(f"{field} must be exactly 1.0 or 1.10")
    return result


def _integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return value


@dataclass(frozen=True)
class MatrixAnalysisInput:
    """Analyzer-only aggregate joining a shared plan, receipts, and baselines.

    Construction is the single public admission gate: direct construction, the
    node-terminal publisher, and ``from_dict`` reload all re-enter the shared
    ``validate_matrix_receipts`` choke point before any audit file is opened.
    """

    plan: MatrixPlan
    cell_receipts: tuple[CellReceipt, ...]
    source_output_paths: tuple[tuple[float, str], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.plan, MatrixPlan):
            raise ValueError("analysis input plan must be a MatrixPlan")
        receipts = tuple(self.cell_receipts)
        if any(not isinstance(item, CellReceipt) for item in receipts):
            raise ValueError("analysis input receipts must be CellReceipt records")
        paths = tuple(
            (
                _rp(rp, field="source output RP"),
                _nonempty(path, field="source output path"),
            )
            for rp, path in self.source_output_paths
        )
        if len(paths) != len(EVALUATION_RPS) or {rp for rp, _ in paths} != set(
            EVALUATION_RPS
        ):
            raise ValueError(
                "source output paths must cover both evaluation RPs exactly"
            )
        if len({path for _, path in paths}) != len(paths):
            raise ValueError("source output paths must be unique")
        validate_matrix_receipts(self.plan, receipts)
        object.__setattr__(self, "cell_receipts", receipts)
        object.__setattr__(self, "source_output_paths", paths)

    def _preimage(self) -> dict[str, Any]:
        return {
            "schema_version": INPUT_SCHEMA,
            "plan": self.plan.to_dict(),
            "cell_receipts": [item.to_dict() for item in self.cell_receipts],
            "source_output_paths": [
                {"evaluation_rp": rp, "output_path": path}
                for rp, path in self.source_output_paths
            ],
        }

    @property
    def content_sha256(self) -> str:
        return _sha256(self._preimage())

    def to_dict(self) -> dict[str, Any]:
        value = self._preimage()
        return {**value, "content_sha256": _sha256(value)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MatrixAnalysisInput":
        required = {
            "schema_version",
            "plan",
            "cell_receipts",
            "source_output_paths",
            "content_sha256",
        }
        if set(value) != required or value.get("schema_version") != INPUT_SCHEMA:
            raise ValueError("analysis input schema differs")
        result = cls(
            plan=MatrixPlan.from_dict(value["plan"]),
            cell_receipts=tuple(
                CellReceipt.from_dict(item) for item in value["cell_receipts"]
            ),
            source_output_paths=tuple(
                (item["evaluation_rp"], item["output_path"])
                for item in value["source_output_paths"]
            ),
        )
        if value["content_sha256"] != result.content_sha256:
            raise ValueError("analysis input content SHA-256 differs")
        return result


@dataclass(frozen=True)
class CellSurfaceAnalysis:
    training_rp: float
    seed_group_id: str
    arm_id: str
    evaluation_rp: float
    trusted_gain_owner_ids: tuple[str, ...]
    baseline_loss_owner_ids: tuple[str, ...]
    historical_g_owner_ids: tuple[str, ...]
    historical_h_owner_ids: tuple[str, ...]
    historical_m_owner_ids: tuple[str, ...]
    incidental_m_recovery_owner_ids: tuple[str, ...]
    protected_but_undefendable_m_owner_ids: tuple[str, ...]
    legacy_union_at_k_diagnostic_owner_ids: tuple[str, ...]
    union_at_k_is_success_evidence: bool
    duplicate_rows: int
    malformed_rows: int
    invalid_rows: int
    unmatched_rows: int
    prediction_rows: int
    generated_tokens: int
    stop_reason_counts: tuple[tuple[str, int], ...]
    natural_stops: int
    cap_stops: int
    update_count: int
    retry_count: int
    rollback_confirmed: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "training_rp", _rp(self.training_rp, field="training RP")
        )
        object.__setattr__(
            self, "evaluation_rp", _rp(self.evaluation_rp, field="evaluation RP")
        )
        if self.seed_group_id not in MATRIX_SEED_GROUPS or self.arm_id not in ARM_IDS:
            raise ValueError("surface identity is outside the canonical matrix")
        tuple_fields = (
            "trusted_gain_owner_ids",
            "baseline_loss_owner_ids",
            "historical_g_owner_ids",
            "historical_h_owner_ids",
            "historical_m_owner_ids",
            "incidental_m_recovery_owner_ids",
            "protected_but_undefendable_m_owner_ids",
            "legacy_union_at_k_diagnostic_owner_ids",
        )
        for field in tuple_fields:
            values = tuple(getattr(self, field))
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{field} must be sorted and unique")
            object.__setattr__(self, field, values)
        if self.union_at_k_is_success_evidence is not False:
            raise ValueError("union@K must remain diagnostic only")
        for field in (
            "duplicate_rows",
            "malformed_rows",
            "invalid_rows",
            "unmatched_rows",
            "prediction_rows",
            "generated_tokens",
            "natural_stops",
            "cap_stops",
            "update_count",
            "retry_count",
        ):
            _integer(getattr(self, field), field=field)
        if self.update_count != 1 or self.retry_count != 0:
            raise ValueError("surface must bind one update and zero retries")
        if self.rollback_confirmed is not True:
            raise ValueError("surface must bind complete rollback")
        reasons = tuple(
            (str(reason), int(count)) for reason, count in self.stop_reason_counts
        )
        if reasons != tuple(sorted(reasons)) or any(count <= 0 for _, count in reasons):
            raise ValueError("stop reason counts must be sorted and positive")
        object.__setattr__(self, "stop_reason_counts", reasons)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SURFACE_SCHEMA,
            "training_rp": self.training_rp,
            "seed_group_id": self.seed_group_id,
            "arm_id": self.arm_id,
            "evaluation_rp": self.evaluation_rp,
            "trusted_gain_owner_ids": list(self.trusted_gain_owner_ids),
            "baseline_loss_owner_ids": list(self.baseline_loss_owner_ids),
            "historical_g_owner_ids": list(self.historical_g_owner_ids),
            "historical_h_owner_ids": list(self.historical_h_owner_ids),
            "historical_m_owner_ids": list(self.historical_m_owner_ids),
            "incidental_m_recovery_owner_ids": list(
                self.incidental_m_recovery_owner_ids
            ),
            "protected_but_undefendable_m_owner_ids": list(
                self.protected_but_undefendable_m_owner_ids
            ),
            "legacy_union_at_k_diagnostic_owner_ids": list(
                self.legacy_union_at_k_diagnostic_owner_ids
            ),
            "union_at_k_is_success_evidence": self.union_at_k_is_success_evidence,
            "duplicate_rows": self.duplicate_rows,
            "malformed_rows": self.malformed_rows,
            "invalid_rows": self.invalid_rows,
            "unmatched_rows": self.unmatched_rows,
            "prediction_rows": self.prediction_rows,
            "generated_tokens": self.generated_tokens,
            "stop_reason_counts": [list(item) for item in self.stop_reason_counts],
            "natural_stops": self.natural_stops,
            "cap_stops": self.cap_stops,
            "update_count": self.update_count,
            "retry_count": self.retry_count,
            "rollback_confirmed": self.rollback_confirmed,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CellSurfaceAnalysis":
        if value.get("schema_version") != SURFACE_SCHEMA:
            raise ValueError("surface analysis schema differs")
        fields = {key: item for key, item in value.items() if key != "schema_version"}
        for key in (
            "trusted_gain_owner_ids",
            "baseline_loss_owner_ids",
            "historical_g_owner_ids",
            "historical_h_owner_ids",
            "historical_m_owner_ids",
            "incidental_m_recovery_owner_ids",
            "protected_but_undefendable_m_owner_ids",
            "legacy_union_at_k_diagnostic_owner_ids",
            "stop_reason_counts",
        ):
            fields[key] = tuple(
                tuple(item) if key == "stop_reason_counts" else item
                for item in fields[key]
            )
        return cls(**fields)


@dataclass(frozen=True)
class PairedArmChange:
    training_rp: float
    seed_group_id: str
    evaluation_rp: float
    from_arm: str
    to_arm: str
    added_trusted_gain_owner_ids: tuple[str, ...]
    removed_trusted_gain_owner_ids: tuple[str, ...]
    added_baseline_loss_owner_ids: tuple[str, ...]
    resolved_baseline_loss_owner_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "training_rp", _rp(self.training_rp, field="training RP")
        )
        object.__setattr__(
            self, "evaluation_rp", _rp(self.evaluation_rp, field="evaluation RP")
        )
        if self.seed_group_id not in MATRIX_SEED_GROUPS or (
            self.from_arm,
            self.to_arm,
        ) not in {
            ("A", "B"),
            ("B", "C"),
        }:
            raise ValueError("paired arm change must stay within one canonical group")
        for field in (
            "added_trusted_gain_owner_ids",
            "removed_trusted_gain_owner_ids",
            "added_baseline_loss_owner_ids",
            "resolved_baseline_loss_owner_ids",
        ):
            values = tuple(getattr(self, field))
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{field} must be sorted and unique")
            object.__setattr__(self, field, values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PAIRED_SCHEMA,
            "training_rp": self.training_rp,
            "seed_group_id": self.seed_group_id,
            "evaluation_rp": self.evaluation_rp,
            "from_arm": self.from_arm,
            "to_arm": self.to_arm,
            "added_trusted_gain_owner_ids": list(self.added_trusted_gain_owner_ids),
            "removed_trusted_gain_owner_ids": list(self.removed_trusted_gain_owner_ids),
            "added_baseline_loss_owner_ids": list(self.added_baseline_loss_owner_ids),
            "resolved_baseline_loss_owner_ids": list(
                self.resolved_baseline_loss_owner_ids
            ),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PairedArmChange":
        if value.get("schema_version") != PAIRED_SCHEMA:
            raise ValueError("paired change schema differs")
        return cls(
            training_rp=value["training_rp"],
            seed_group_id=value["seed_group_id"],
            evaluation_rp=value["evaluation_rp"],
            from_arm=value["from_arm"],
            to_arm=value["to_arm"],
            added_trusted_gain_owner_ids=tuple(value["added_trusted_gain_owner_ids"]),
            removed_trusted_gain_owner_ids=tuple(
                value["removed_trusted_gain_owner_ids"]
            ),
            added_baseline_loss_owner_ids=tuple(value["added_baseline_loss_owner_ids"]),
            resolved_baseline_loss_owner_ids=tuple(
                value["resolved_baseline_loss_owner_ids"]
            ),
        )


@dataclass(frozen=True)
class TrainingRPSuccess:
    training_rp: float
    contract_local_passing_seed_groups: tuple[str, ...]
    contract_local_success: bool
    rp_robust_passing_seed_groups: tuple[str, ...]
    rp_robust_success: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "training_rp", _rp(self.training_rp, field="training RP")
        )
        for field in (
            "contract_local_passing_seed_groups",
            "rp_robust_passing_seed_groups",
        ):
            values = tuple(getattr(self, field))
            if values != tuple(
                group for group in MATRIX_SEED_GROUPS if group in values
            ):
                raise ValueError(f"{field} must follow canonical seed order")
            object.__setattr__(self, field, values)
        if self.contract_local_success is not (
            len(self.contract_local_passing_seed_groups) >= 2
        ) or self.rp_robust_success is not (
            len(self.rp_robust_passing_seed_groups) >= 2
        ):
            raise ValueError("success booleans differ from the exact two-seed rule")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SUCCESS_SCHEMA,
            "training_rp": self.training_rp,
            "contract_local_passing_seed_groups": list(
                self.contract_local_passing_seed_groups
            ),
            "contract_local_success": self.contract_local_success,
            "rp_robust_passing_seed_groups": list(self.rp_robust_passing_seed_groups),
            "rp_robust_success": self.rp_robust_success,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TrainingRPSuccess":
        if value.get("schema_version") != SUCCESS_SCHEMA:
            raise ValueError("success schema differs")
        return cls(
            training_rp=value["training_rp"],
            contract_local_passing_seed_groups=tuple(
                value["contract_local_passing_seed_groups"]
            ),
            contract_local_success=value["contract_local_success"],
            rp_robust_passing_seed_groups=tuple(value["rp_robust_passing_seed_groups"]),
            rp_robust_success=value["rp_robust_success"],
        )


@dataclass(frozen=True)
class MatrixAnalysisReceipt:
    input_sha256: str
    source_sha256: str
    manifest_sha256: str
    surfaces: tuple[CellSurfaceAnalysis, ...]
    paired_changes: tuple[PairedArmChange, ...]
    success: tuple[TrainingRPSuccess, ...]
    bi_policy_replication: bool
    qualification_outcomes_included: bool = False

    def __post_init__(self) -> None:
        for field in ("input_sha256", "source_sha256", "manifest_sha256"):
            object.__setattr__(self, field, _digest(getattr(self, field), field=field))
        surfaces = tuple(self.surfaces)
        paired = tuple(self.paired_changes)
        success = tuple(self.success)
        if len(surfaces) != 36 or len(paired) != 24 or len(success) != 2:
            raise ValueError(
                "analysis receipt must retain the complete matrix projection"
            )
        expected_surfaces = tuple(
            (training_rp, seed_group, arm_id, evaluation_rp)
            for training_rp in TRAINING_RPS
            for seed_group in MATRIX_SEED_GROUPS
            for arm_id in ARM_IDS
            for evaluation_rp in EVALUATION_RPS
        )
        observed_surfaces = tuple(
            (
                item.training_rp,
                item.seed_group_id,
                item.arm_id,
                item.evaluation_rp,
            )
            for item in surfaces
        )
        if observed_surfaces != expected_surfaces:
            raise ValueError("analysis receipt complete surface identities differ")
        expected_paired = tuple(
            (training_rp, seed_group, evaluation_rp, from_arm, to_arm)
            for training_rp in TRAINING_RPS
            for seed_group in MATRIX_SEED_GROUPS
            for evaluation_rp in EVALUATION_RPS
            for from_arm, to_arm in (("A", "B"), ("B", "C"))
        )
        observed_paired = tuple(
            (
                item.training_rp,
                item.seed_group_id,
                item.evaluation_rp,
                item.from_arm,
                item.to_arm,
            )
            for item in paired
        )
        if observed_paired != expected_paired:
            raise ValueError("analysis receipt paired identities differ")
        if tuple(item.training_rp for item in success) != TRAINING_RPS:
            raise ValueError("analysis receipt success identities differ")
        if self.bi_policy_replication is not all(
            item.rp_robust_success for item in success
        ):
            raise ValueError("bi-policy replication differs from RP-robust success")
        if self.qualification_outcomes_included is not False:
            raise ValueError("qualification outcomes must not enter matrix disposition")
        object.__setattr__(self, "surfaces", surfaces)
        object.__setattr__(self, "paired_changes", paired)
        object.__setattr__(self, "success", success)

    def _preimage(self) -> dict[str, Any]:
        return {
            "schema_version": ANALYSIS_SCHEMA,
            "input_sha256": self.input_sha256,
            "source_sha256": self.source_sha256,
            "manifest_sha256": self.manifest_sha256,
            "surfaces": [item.to_dict() for item in self.surfaces],
            "paired_changes": [item.to_dict() for item in self.paired_changes],
            "success": [item.to_dict() for item in self.success],
            "bi_policy_replication": self.bi_policy_replication,
            "qualification_outcomes_included": self.qualification_outcomes_included,
        }

    @property
    def content_sha256(self) -> str:
        return _sha256(self._preimage())

    def to_dict(self) -> dict[str, Any]:
        value = self._preimage()
        return {**value, "content_sha256": _sha256(value)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MatrixAnalysisReceipt":
        required = {
            "schema_version",
            "input_sha256",
            "source_sha256",
            "manifest_sha256",
            "surfaces",
            "paired_changes",
            "success",
            "bi_policy_replication",
            "qualification_outcomes_included",
            "content_sha256",
        }
        if set(value) != required or value.get("schema_version") != ANALYSIS_SCHEMA:
            raise ValueError("analysis receipt schema differs")
        result = cls(
            input_sha256=value["input_sha256"],
            source_sha256=value["source_sha256"],
            manifest_sha256=value["manifest_sha256"],
            surfaces=tuple(
                CellSurfaceAnalysis.from_dict(item) for item in value["surfaces"]
            ),
            paired_changes=tuple(
                PairedArmChange.from_dict(item) for item in value["paired_changes"]
            ),
            success=tuple(
                TrainingRPSuccess.from_dict(item) for item in value["success"]
            ),
            bi_policy_replication=value["bi_policy_replication"],
            qualification_outcomes_included=value["qualification_outcomes_included"],
        )
        if value["content_sha256"] != result.content_sha256:
            raise ValueError("analysis receipt content SHA-256 differs")
        return result


def _load_node_terminal(
    value: NodeTerminalReceipt | Mapping[str, Any] | str | Path,
) -> NodeTerminalReceipt:
    if isinstance(value, NodeTerminalReceipt):
        return value
    if isinstance(value, Mapping):
        return NodeTerminalReceipt.from_dict(value)
    path = Path(value)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("node terminal receipt is unavailable or invalid") from error
    if not isinstance(payload, Mapping):
        raise ValueError("node terminal receipt must contain an object")
    return NodeTerminalReceipt.from_dict(payload)


def publish_matrix_analysis_input(
    node_terminals: Sequence[NodeTerminalReceipt | Mapping[str, Any] | str | Path],
    *,
    source_baselines: Sequence[SourceBaselineRef],
    source_output_paths: Sequence[tuple[float, str]],
    concurrency_cap: int = 8,
    output_path: str | Path | None = None,
) -> MatrixAnalysisInput:
    """Finalize the six matrix node terminals into one analyzable matrix.

    This is the only aggregate publisher: it consumes the exact typed specs and
    receipts the nodes persisted plus the two immutable Source baselines,
    excludes qualification, and re-enters the shared ``MatrixPlan`` and
    ``MatrixAnalysisInput`` constructors, so the published artifact and any
    later reload are admitted by the same choke point.
    """

    terminals = tuple(_load_node_terminal(item) for item in node_terminals)
    if any(item.phase != PHASE_MATRIX for item in terminals):
        raise ValueError(
            "qualification node terminals are excluded from the matrix aggregate"
        )
    expected_keys = {
        (training_rp, seed_group)
        for training_rp in TRAINING_RPS
        for seed_group in MATRIX_SEED_GROUPS
    }
    by_key = {
        (
            item.acquisition_key.training_rp,
            item.acquisition_key.seed_group_id,
        ): item
        for item in terminals
    }
    if len(terminals) != len(expected_keys) or set(by_key) != expected_keys:
        raise ValueError(
            "the matrix aggregate requires exactly six matrix node terminals"
        )
    if any(item.status != "succeeded" for item in terminals):
        raise ValueError("every matrix node terminal must have succeeded")

    ordered = [
        by_key[(training_rp, seed_group)]
        for training_rp in TRAINING_RPS
        for seed_group in MATRIX_SEED_GROUPS
    ]
    cells = tuple(spec for item in ordered for spec in item.cell_specs)
    receipts = tuple(receipt for item in ordered for receipt in item.cell_receipts)
    plan = MatrixPlan(
        acquisitions=tuple(item.acquisition_key for item in ordered),
        cells=cells,
        source_baselines=tuple(source_baselines),
        dependency_edges=tuple(
            (cell.cell_key.acquisition_key.content_sha256, cell.content_sha256)
            for cell in cells
        ),
        concurrency_cap=concurrency_cap,
        dry_run_counters={key: 0 for key in DRY_RUN_COUNTER_KEYS},
    )
    published = MatrixAnalysisInput(plan, receipts, tuple(source_output_paths))
    if output_path is not None:
        path = Path(output_path)
        if path.exists():
            raise FileExistsError(f"refusing to overwrite analysis input: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(published.to_dict(), sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
    return published


def _load_input(
    value: MatrixAnalysisInput | Mapping[str, Any] | str | Path,
) -> MatrixAnalysisInput:
    if isinstance(value, MatrixAnalysisInput):
        payload = value.to_dict()
    elif isinstance(value, Mapping):
        payload = dict(value)
    else:
        path = Path(value)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("analysis input file is unavailable or invalid") from error
        if not isinstance(payload, Mapping):
            raise ValueError("analysis input file must contain an object")
    return MatrixAnalysisInput.from_dict(payload)


def _admit_complete_matrix(
    manifest: Human13KUnionManifest,
    inputs: MatrixAnalysisInput,
    *,
    expected_manifest_sha256: str | None,
) -> tuple[dict[tuple[float, str, str], CellSpec], str, str]:
    """Bind one admitted matrix to this exact manifest and Source lineage.

    Cell/receipt shape, identity, and nested-objective admission belong to the
    shared contract validator; only the manifest-bound lineage is analyzer
    business.
    """

    if not isinstance(manifest, Human13KUnionManifest):
        raise ValueError("matrix analyzer requires a Human13KUnionManifest")
    if (
        tuple(sorted(image.image_id for image in manifest.images))
        != CANONICAL_IMAGE_IDS
    ):
        raise ValueError("manifest must cover the sealed thirteen image IDs exactly")
    matcher = manifest.binding.matcher
    if (
        matcher.algorithm != "cardinality_first_max_total_iou"
        or matcher.same_category is not True
        or matcher.duplicate_comparison != "strictly_greater"
    ):
        raise ValueError("manifest must bind the canonical matcher identity")
    manifest_sha = _manifest_sha256(manifest)
    if (
        expected_manifest_sha256 is not None
        and manifest_sha != expected_manifest_sha256
    ):
        raise ValueError("manifest differs from the exact frozen manifest SHA-256")

    validate_matrix_receipts(inputs.plan, inputs.cell_receipts)
    planned = {matrix_identity(cell): cell for cell in inputs.plan.cells}
    source_ids = {cell.shared_evidence.source_sha256 for cell in inputs.plan.cells}
    manifest_ids = {cell.shared_evidence.manifest_sha256 for cell in inputs.plan.cells}
    if len(source_ids) != 1 or manifest_ids != {manifest_sha}:
        raise ValueError("matrix Source or manifest lineage is mixed")
    source_sha = source_ids.pop()
    if {item.checkpoint_sha256 for item in inputs.plan.source_baselines} != {
        source_sha
    }:
        raise ValueError("matrix Source checkpoint lineage is mixed")
    return planned, source_sha, manifest_sha


def _load_jsonl(path_value: str, expected_sha256: str) -> tuple[Mapping[str, Any], ...]:
    path = Path(path_value)
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ValueError(f"audit output path is unavailable: {path}") from error
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError(f"audit output SHA-256 differs for exact path {path}")
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number} is invalid JSON") from error
        if not isinstance(item, Mapping):
            raise ValueError(f"{path}:{line_number} must be an object")
        rows.append(item)
    return tuple(rows)


def _owner_universes(
    manifest: Human13KUnionManifest,
) -> tuple[set[str], set[str], set[str]]:
    return (
        {owner for image in manifest.images for owner in image.g_owner_ids},
        {owner for image in manifest.images for owner in image.h_owner_ids},
        {owner for image in manifest.images for owner in image.m_owner_ids},
    )


def _project_output_rows(
    manifest: Human13KUnionManifest,
    rows: Sequence[Mapping[str, Any]],
    *,
    evaluation_rp: float,
    checkpoint_sha256: str,
    generation_sha256: str | None,
    source_sha256: str,
    manifest_sha256: str,
    policy_sha256: str | None,
    config_sha256: str | None,
    training_rp: float | None,
    seed_group_id: str | None,
    arm_id: str | None,
) -> dict[str, Any]:
    if len(rows) != len(CANONICAL_IMAGE_IDS):
        raise ValueError("audit output must contain the exact thirteen image IDs")
    images = {image.image_id: image for image in manifest.images}
    observed_ids: list[int] = []
    final_owner_ids: set[str] = set()
    burdens: Counter[str] = Counter()
    stop_reasons: Counter[str] = Counter()
    group_signatures: set[tuple[Any, ...]] = set()
    matcher = manifest.binding.matcher
    for row in rows:
        image_id = _integer(row.get("image_id"), field="audit image_id")
        observed_ids.append(image_id)
        image: ImageRecord | None = images.get(image_id)
        if image is None:
            raise ValueError("audit output contains an image outside the manifest")
        if row.get("decode_mode") != "original_prompt_clean_greedy":
            raise ValueError("audit output is not original-prompt clean greedy")
        if _rp(row.get("repetition_penalty"), field="generation RP") != evaluation_rp:
            raise ValueError("audit generation RP differs from its AuditRef")
        if (
            generation_sha256 is not None
            and row.get("generation_policy_receipt_sha256") != generation_sha256
        ):
            raise ValueError("audit generation-policy receipt differs")
        provenance = row.get("provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError("audit provenance must be an object")
        if provenance.get("manifest_sha256") != manifest_sha256:
            raise ValueError("audit manifest lineage differs")
        if provenance.get("source_checkpoint_sha256") != source_sha256:
            raise ValueError("audit Source lineage differs")
        if provenance.get("checkpoint_payload_sha256") != checkpoint_sha256:
            raise ValueError("audit evaluated checkpoint differs")
        if (
            policy_sha256 is not None
            and provenance.get("policy_contract_sha256") != policy_sha256
        ):
            raise ValueError("audit policy lineage differs")
        if (
            config_sha256 is not None
            and provenance.get("resolved_config_sha256") != config_sha256
        ):
            raise ValueError("audit config lineage differs")
        if (
            training_rp is not None
            and _rp(provenance.get("training_rp"), field="audit training RP")
            != training_rp
        ):
            raise ValueError("audit training RP lineage differs")
        if (
            seed_group_id is not None
            and provenance.get("seed_group_id") != seed_group_id
        ):
            raise ValueError("audit seed-group lineage differs")
        if arm_id is not None and provenance.get("arm_id") != arm_id:
            raise ValueError("audit arm lineage differs")
        group_signatures.add(
            (
                provenance.get("manifest_sha256"),
                provenance.get("source_checkpoint_sha256"),
                provenance.get("checkpoint_payload_sha256"),
                provenance.get("policy_contract_sha256"),
                provenance.get("resolved_config_sha256"),
                provenance.get("training_rp"),
                provenance.get("seed_group_id"),
                provenance.get("arm_id"),
            )
        )
        predictions = _ordered_predictions(row)
        projected = _match_prefix(
            image,
            predictions,
            duplicate_iou_threshold=matcher.duplicate_iou_threshold,
            owner_iou_threshold=matcher.owner_iou_threshold,
        )
        final_owner_ids.update(projected["owner_matches"])
        malformed = _integer(
            row.get("malformed_row_count", 0), field="malformed_row_count"
        )
        token_ids = row.get("generated_token_ids")
        if not isinstance(token_ids, list) or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in token_ids
        ):
            raise ValueError("audit generated_token_ids must be nonnegative integers")
        stop_reason = _nonempty(row.get("stop_reason"), field="stop_reason")
        burdens.update(
            duplicate_rows=len(projected["duplicate_rows"]),
            invalid_rows=len(projected["invalid_rows"]),
            unmatched_rows=len(projected["unmatched_rows"]),
            malformed_rows=malformed,
            prediction_rows=len(predictions) + malformed,
            generated_tokens=len(token_ids),
            natural_stops=int(stop_reason in _NATURAL_STOP_REASONS),
            cap_stops=int(stop_reason in _CAP_STOP_REASONS),
        )
        stop_reasons[stop_reason] += 1
    if tuple(sorted(observed_ids)) != CANONICAL_IMAGE_IDS or len(
        set(observed_ids)
    ) != len(observed_ids):
        raise ValueError("audit output must contain the exact thirteen image IDs")
    if len(group_signatures) != 1:
        raise ValueError("audit output lineage is mixed within its thirteen rows")
    return {
        "owner_ids": final_owner_ids,
        **burdens,
        "stop_reason_counts": tuple(sorted(stop_reasons.items())),
    }


def _surface_result(
    *,
    receipt: CellReceipt,
    audit_projection: Mapping[str, Any],
    source_owner_ids: set[str],
    g: set[str],
    h: set[str],
    m: set[str],
    evaluation_rp: float,
) -> CellSurfaceAnalysis:
    final = set(audit_projection["owner_ids"])
    acquisition = receipt.cell_key.acquisition_key
    return CellSurfaceAnalysis(
        training_rp=acquisition.training_rp,
        seed_group_id=acquisition.seed_group_id,
        arm_id=receipt.cell_key.arm_id,
        evaluation_rp=evaluation_rp,
        trusted_gain_owner_ids=tuple(sorted((final & (g | h)) - source_owner_ids)),
        baseline_loss_owner_ids=tuple(sorted(source_owner_ids - final)),
        historical_g_owner_ids=tuple(sorted(final & g)),
        historical_h_owner_ids=tuple(sorted(final & h)),
        historical_m_owner_ids=tuple(sorted(final & m)),
        incidental_m_recovery_owner_ids=tuple(sorted((final & m) - source_owner_ids)),
        protected_but_undefendable_m_owner_ids=tuple(sorted(source_owner_ids & m)),
        legacy_union_at_k_diagnostic_owner_ids=tuple(sorted(g | h)),
        union_at_k_is_success_evidence=False,
        duplicate_rows=int(audit_projection["duplicate_rows"]),
        malformed_rows=int(audit_projection["malformed_rows"]),
        invalid_rows=int(audit_projection["invalid_rows"]),
        unmatched_rows=int(audit_projection["unmatched_rows"]),
        prediction_rows=int(audit_projection["prediction_rows"]),
        generated_tokens=int(audit_projection["generated_tokens"]),
        stop_reason_counts=tuple(audit_projection["stop_reason_counts"]),
        natural_stops=int(audit_projection["natural_stops"]),
        cap_stops=int(audit_projection["cap_stops"]),
        update_count=receipt.update_count,
        retry_count=0,
        rollback_confirmed=receipt.rollback_confirmed,
    )


def _paired_changes(
    surfaces: Sequence[CellSurfaceAnalysis],
) -> tuple[PairedArmChange, ...]:
    by_key = {
        (item.training_rp, item.seed_group_id, item.arm_id, item.evaluation_rp): item
        for item in surfaces
    }
    result: list[PairedArmChange] = []
    for training_rp in TRAINING_RPS:
        for seed_group in MATRIX_SEED_GROUPS:
            for evaluation_rp in EVALUATION_RPS:
                for from_arm, to_arm in (("A", "B"), ("B", "C")):
                    before = by_key[(training_rp, seed_group, from_arm, evaluation_rp)]
                    after = by_key[(training_rp, seed_group, to_arm, evaluation_rp)]
                    before_gain = set(before.trusted_gain_owner_ids)
                    after_gain = set(after.trusted_gain_owner_ids)
                    before_loss = set(before.baseline_loss_owner_ids)
                    after_loss = set(after.baseline_loss_owner_ids)
                    result.append(
                        PairedArmChange(
                            training_rp,
                            seed_group,
                            evaluation_rp,
                            from_arm,
                            to_arm,
                            tuple(sorted(after_gain - before_gain)),
                            tuple(sorted(before_gain - after_gain)),
                            tuple(sorted(after_loss - before_loss)),
                            tuple(sorted(before_loss - after_loss)),
                        )
                    )
    return tuple(result)


def _success(surfaces: Sequence[CellSurfaceAnalysis]) -> tuple[TrainingRPSuccess, ...]:
    c_surfaces = {
        (item.training_rp, item.seed_group_id, item.evaluation_rp): item
        for item in surfaces
        if item.arm_id == "C"
    }
    result: list[TrainingRPSuccess] = []
    for training_rp in TRAINING_RPS:
        passed_by_surface = {
            evaluation_rp: {
                seed_group
                for seed_group in MATRIX_SEED_GROUPS
                if (
                    c_surfaces[
                        (training_rp, seed_group, evaluation_rp)
                    ].trusted_gain_owner_ids
                    and not c_surfaces[
                        (training_rp, seed_group, evaluation_rp)
                    ].baseline_loss_owner_ids
                )
            }
            for evaluation_rp in EVALUATION_RPS
        }
        local = tuple(
            group
            for group in MATRIX_SEED_GROUPS
            if group in passed_by_surface[training_rp]
        )
        robust_set = set.intersection(*passed_by_surface.values())
        robust = tuple(group for group in MATRIX_SEED_GROUPS if group in robust_set)
        result.append(
            TrainingRPSuccess(
                training_rp,
                local,
                len(local) >= 2,
                robust,
                len(robust) >= 2,
            )
        )
    return tuple(result)


def _analyze_matrix_impl(
    manifest: Human13KUnionManifest,
    value: MatrixAnalysisInput | Mapping[str, Any] | str | Path,
    *,
    expected_manifest_sha256: str | None,
) -> MatrixAnalysisReceipt:
    """Admit, load, and analyze exactly one complete eighteen-cell matrix."""

    inputs = _load_input(value)
    planned, source_sha, manifest_sha = _admit_complete_matrix(
        manifest,
        inputs,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    g, h, m = _owner_universes(manifest)

    source_paths = dict(inputs.source_output_paths)
    baselines = {item.evaluation_rp: item for item in inputs.plan.source_baselines}
    source_owners: dict[float, set[str]] = {}
    for evaluation_rp in EVALUATION_RPS:
        baseline = baselines[evaluation_rp]
        rows = _load_jsonl(source_paths[evaluation_rp], baseline.output_a_sha256)
        projected = _project_output_rows(
            manifest,
            rows,
            evaluation_rp=evaluation_rp,
            checkpoint_sha256=baseline.checkpoint_sha256,
            generation_sha256=None,
            source_sha256=source_sha,
            manifest_sha256=manifest_sha,
            policy_sha256=None,
            config_sha256=None,
            training_rp=None,
            seed_group_id=None,
            arm_id=None,
        )
        source_owners[evaluation_rp] = set(projected["owner_ids"])

    receipts = {matrix_identity(item): item for item in inputs.cell_receipts}
    surfaces: list[CellSurfaceAnalysis] = []
    for training_rp in TRAINING_RPS:
        for seed_group in MATRIX_SEED_GROUPS:
            for arm_id in ARM_IDS:
                key = (training_rp, seed_group, arm_id)
                receipt = receipts[key]
                cell = planned[key]
                audits = {audit.evaluation_rp: audit for audit in receipt.audits}
                for evaluation_rp in EVALUATION_RPS:
                    audit = audits[evaluation_rp]
                    rows = _load_jsonl(audit.output_path, audit.output_sha256)
                    projected = _project_output_rows(
                        manifest,
                        rows,
                        evaluation_rp=evaluation_rp,
                        checkpoint_sha256=audit.evaluated_checkpoint_sha256,
                        generation_sha256=audit.generation_policy_receipt_sha256,
                        source_sha256=source_sha,
                        manifest_sha256=manifest_sha,
                        policy_sha256=cell.shared_evidence.policy_contract_sha256,
                        config_sha256=cell.leaf_config_sha256,
                        training_rp=training_rp,
                        seed_group_id=seed_group,
                        arm_id=arm_id,
                    )
                    surfaces.append(
                        _surface_result(
                            receipt=receipt,
                            audit_projection=projected,
                            source_owner_ids=source_owners[evaluation_rp],
                            g=g,
                            h=h,
                            m=m,
                            evaluation_rp=evaluation_rp,
                        )
                    )
    surfaces_tuple = tuple(surfaces)
    success = _success(surfaces_tuple)
    return MatrixAnalysisReceipt(
        input_sha256=inputs.content_sha256,
        source_sha256=source_sha,
        manifest_sha256=manifest_sha,
        surfaces=surfaces_tuple,
        paired_changes=_paired_changes(surfaces_tuple),
        success=success,
        bi_policy_replication=all(item.rp_robust_success for item in success),
        qualification_outcomes_included=False,
    )


def _analyze_matrix_for_test(
    manifest: Human13KUnionManifest,
    value: MatrixAnalysisInput | Mapping[str, Any] | str | Path,
) -> MatrixAnalysisReceipt:
    """Exercise matrix semantics with a synthetic manifest in CPU-only tests."""

    return _analyze_matrix_impl(
        manifest,
        value,
        expected_manifest_sha256=None,
    )


def analyze_matrix(
    manifest: Human13KUnionManifest,
    value: MatrixAnalysisInput | Mapping[str, Any] | str | Path,
) -> MatrixAnalysisReceipt:
    """Admit and analyze the matrix bound to the exact frozen human13 manifest."""

    return _analyze_matrix_impl(
        manifest,
        value,
        expected_manifest_sha256=FROZEN_MANIFEST_SHA256,
    )


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    result = analyze_matrix(load_manifest(args.manifest), Path(args.input))
    print(json.dumps(result.to_dict(), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CellSurfaceAnalysis",
    "MatrixAnalysisInput",
    "MatrixAnalysisReceipt",
    "PairedArmChange",
    "TrainingRPSuccess",
    "analyze_matrix",
    "publish_matrix_analysis_input",
]
