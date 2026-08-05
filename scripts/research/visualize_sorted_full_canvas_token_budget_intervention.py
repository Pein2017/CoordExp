#!/usr/bin/env python3
"""Paired owner-local atlas for the sorted full-canvas token-budget probe.

The intervention analyzer owns owner outcomes.  This adapter consumes its
``intervention-report.json`` and renders only deterministic, report-selected
examples.  It never recomputes support, thresholds, or a scientific verdict.

Each figure uses one physical owner's frozen local candidate bank and one
identical context in both arms.  Confidence is independently normalized inside
each arm over that same owner-local bank.  Raw log probabilities are neither
emitted nor compared across arms.  The two panels share one score-independent
crop in the predecessor one-times pixel frame.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research import build_sorted_owner_accessibility_census_plan as planner  # noqa: E402
from scripts.research import merge_sorted_owner_accessibility_census_shards as merge  # noqa: E402
from scripts.research import prepare_sorted_full_canvas_token_budget_intervention as prepare  # noqa: E402
from scripts.research import analyze_sorted_full_canvas_token_budget_intervention as analyzer  # noqa: E402

UNIT_ID = prepare.INTERVENTION_UNIT_ID
REPORT_SCHEMA_VERSION = "sorted-full-canvas-visual-token-budget-intervention-report.v1"
VISUAL_SCHEMA_VERSION = "sorted-full-canvas-visual-token-budget-paired-owner.v1"
MANIFEST_SCHEMA_VERSION = "sorted-full-canvas-visual-token-budget-paired-manifest.v1"
MANIFEST_NAME = "visual-manifest.json"
SPECS_NAME = "paired-owner-specs.jsonl"
SCORES_NAME = "census-scores.jsonl"
RECEIPT_NAME = "shard-receipt.json"
PEAK_COUNT = 4
PEAK_NMS_IOU = 0.50
CROP_PADDING_PIXELS = 64
MIN_CROP_EXTENT_PIXELS = 192
DEFAULT_MAX_PER_ROLE = 12

canonical_json_bytes = planner.canonical_json_bytes
sha256_json = planner.sha256_json


class VisualContractError(RuntimeError):
    """A required artifact, provenance identity, or exact join is invalid."""


@dataclass(frozen=True)
class Case:
    owner_id: str
    role: str
    context_id: str
    matched_recovered_owner_id: str | None = None


@dataclass(frozen=True)
class Inputs:
    report_path: Path
    overlay_path: Path
    plan_dir: Path
    baseline_shard_root: Path
    treatment_shard_root: Path
    report: dict[str, Any]
    overlay: dict[str, Any]
    baseline_plan: merge.PlanBundle
    treatment_plan: merge.PlanBundle
    treatment_shards: Mapping[str, analyzer.TreatmentShard]


def _fail(message: str) -> None:
    raise VisualContractError(message)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VisualContractError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        _fail(f"{label} {path} is not a JSON object")
    return value


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                _fail(f"{label} {path}:{line_number} is not a JSON object")
            rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise VisualContractError(f"cannot read {label} {path}: {exc}") from exc
    return rows


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{label} is missing or is not an object")
    return value


def _require_sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        _fail(f"{label} is missing or is not a list")
    return value


def validate_report_provenance(
    report: Mapping[str, Any],
    overlay: Mapping[str, Any],
    plan: merge.PlanBundle,
) -> None:
    """Bind the analyzer report to the sealed overlay and predecessor plan."""

    expected = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "complete_uniform_capture_identity",
        "intervention_unit_id": UNIT_ID,
        "arm_id": str(overlay["arm_id"]),
        "baseline_arm_id": str(overlay["baseline_arm_id"]),
        "overlay_content_sha256": str(overlay["overlay_content_sha256"]),
        "base_plan_receipt_content_sha256": str(plan.receipt["receipt_content_sha256"]),
        "predecessor_run_root": str(overlay["base"]["predecessor_run_root"]),
    }
    for key, wanted in expected.items():
        if str(report.get(key)) != str(wanted):
            _fail(f"report {key} {report.get(key)!r} does not match {wanted!r}")

    comparison = _require_mapping(report.get("comparison_semantics"), "comparison_semantics")
    if comparison.get("raw_logprob_compared_across_arms") is not False:
        _fail("report does not forbid cross-arm raw-logprob comparison")
    if comparison.get("compared_quantity") != "support_disposition_under_arm_local_calibration":
        _fail("report comparison semantics do not use arm-local calibration")

    captured = {str(value) for value in _require_sequence(report.get("captured_images"), "captured_images")}
    overlay_images = {str(value) for value in _require_mapping(overlay.get("images"), "overlay.images")}
    if captured != overlay_images:
        _fail("report captured_images do not exactly equal the sealed overlay image set")

    admission = _require_mapping(
        report.get("owner_role_context_admission"), "owner_role_context_admission"
    )
    if admission.get("selection_basis") != (
        "pre_treatment_overlay_and_predecessor_owner_summaries_only"
    ) or admission.get("treatment_scores_used_for_context_selection") is not False:
        _fail("report owner/context selection is not sealed and score-independent")
    _require_mapping(admission.get("owners"), "owner_role_context_admission.owners")

    capture_identity = _require_mapping(report.get("capture_identity"), "capture_identity")
    if capture_identity.get("status") != "complete_uniform_capture_identity":
        _fail("report capture_identity is not complete_uniform_capture_identity")
    if int(capture_identity.get("image_count", -1)) != len(overlay_images):
        _fail("report capture_identity image_count does not cover the sealed overlay")

    seals = _require_mapping(report.get("capture_artifact_seals"), "capture_artifact_seals")
    if seals.get("schema_version") != analyzer.CAPTURE_ARTIFACT_SEALS_SCHEMA_VERSION:
        _fail("report capture_artifact_seals have an unexpected schema_version")
    reconstructed = sha256_json(
        {key: value for key, value in seals.items() if key != "capture_artifact_seals_sha256"}
    )
    if seals.get("capture_artifact_seals_sha256") != reconstructed:
        _fail("report capture_artifact_seals do not reconstruct their own digest")
    sealed_images = _require_mapping(seals.get("images"), "capture_artifact_seals.images")
    if {str(key) for key in sealed_images} != overlay_images:
        _fail("report capture artifact seal image set does not equal the overlay")


def verify_capture_artifact_seals(
    report: Mapping[str, Any], treatment_shard_root: Path
) -> None:
    """Verify exact treatment bytes before any treatment score row is parsed."""

    seals = _require_mapping(report.get("capture_artifact_seals"), "capture_artifact_seals")
    images = _require_mapping(seals.get("images"), "capture_artifact_seals.images")
    root = Path(treatment_shard_root).resolve()
    for image_id, raw in sorted(images.items()):
        block = _require_mapping(raw, f"capture artifact seal {image_id}")
        directory = None
        for name in (str(image_id), f"shard-{image_id}"):
            candidate = root / name
            if candidate.is_dir():
                directory = candidate.resolve()
                break
        if directory is None:
            _fail(f"sealed treatment shard {image_id!r} is absent under {root}")
        expected = (
            ("receipt_path", "receipt_sha256", directory / RECEIPT_NAME),
            ("census_scores_path", "census_scores_sha256", directory / SCORES_NAME),
        )
        for path_key, digest_key, actual_path in expected:
            if Path(str(block.get(path_key))).resolve() != actual_path:
                _fail(f"sealed {image_id} {path_key} does not name the supplied shard root")
            if not actual_path.is_file():
                _fail(f"sealed treatment artifact is missing: {actual_path}")
            if str(block.get(digest_key)) != planner.sha256_file(actual_path):
                _fail(f"sealed treatment artifact digest changed: {actual_path}")


def verify_stable_plan_semantics(
    baseline: merge.PlanBundle, treatment: merge.PlanBundle
) -> None:
    """Prove the two plan views differ only in presentation/prefix identity."""

    for label, baseline_rows, treatment_rows, stable_fields in (
        (
            "owner",
            baseline.owners,
            treatment.owners,
            (
                "image_id",
                "normalized_description",
                "bbox_pixel_xyxy",
                "candidate_bank",
            ),
        ),
        (
            "candidate",
            baseline.candidates,
            treatment.candidates,
            (
                "image_id",
                "normalized_description",
                "coord_token_ids",
                "coord_token_ids_sha256",
                "decoded_bbox_pixel_xyxy",
                "representative_role",
                "generator_gt_owner_ids",
            ),
        ),
        (
            "context",
            baseline.contexts,
            treatment.contexts,
            ("image_id", "generated_prefix_token_ids", "generated_prefix_token_ids_sha256"),
        ),
        (
            "query group",
            baseline.query_groups,
            treatment.query_groups,
            (
                "image_id",
                "context_id",
                "normalized_description",
                "candidate_ids",
                "query_suffix_token_ids",
                "query_suffix_token_ids_sha256",
            ),
        ),
    ):
        if set(baseline_rows) != set(treatment_rows):
            _fail(f"{label} IDs differ between baseline and treatment plan views")
        for row_id in sorted(baseline_rows):
            before = baseline_rows[row_id]
            after = treatment_rows[row_id]
            for field in stable_fields:
                if before.get(field) != after.get(field):
                    _fail(
                        f"{label} {row_id!r} field {field!r} drifted across plan views"
                    )

    if set(baseline.images) != set(treatment.images):
        _fail("image IDs differ between baseline and treatment plan views")
    if baseline.categories != treatment.categories:
        _fail("category semantics drifted across baseline and treatment plan views")


def load_inputs(
    *,
    report_path: Path,
    overlay_path: Path,
    plan_dir: Path,
    baseline_shard_root: Path,
    treatment_shard_root: Path,
) -> Inputs:
    try:
        overlay = prepare.load_overlay(Path(overlay_path))
        baseline_plan = merge.load_plan(Path(plan_dir))
        treatment_plan = merge.load_plan(Path(plan_dir))
        prepare.assert_overlay_binds_plan(
            overlay,
            plan_receipt_content_sha256=str(
                baseline_plan.receipt["receipt_content_sha256"]
            ),
            capture_rules_sha256=str(baseline_plan.receipt["capture_rules_sha256"]),
        )
        prepare.apply_overlay(
            overlay,
            images=treatment_plan.images,
            contexts=treatment_plan.contexts,
            query_groups=treatment_plan.query_groups,
            categories=treatment_plan.categories,
            candidates=treatment_plan.candidates,
            owners=treatment_plan.owners,
        )
        verify_stable_plan_semantics(baseline_plan, treatment_plan)
    except (prepare.PrepareContractError, merge.MergeContractError) as exc:
        raise VisualContractError(str(exc)) from exc
    report = _read_json(Path(report_path), "intervention report")
    validate_report_provenance(report, overlay, baseline_plan)
    verify_capture_artifact_seals(report, treatment_shard_root)
    try:
        treatment_shards = analyzer.load_treatment_shards(
            Path(treatment_shard_root), overlay
        )
        # Runs the exact analyzer event admission, including redundant row-ID
        # joins, prefix/suffix admissions, full row-stamp equality, and score
        # completeness.  The resulting features are deliberately discarded:
        # the report, not this presentation adapter, owns support semantics.
        analyzer.build_treatment_owner_contexts(treatment_plan, treatment_shards)
        current_identity = analyzer.capture_identity_summary(treatment_shards, overlay)
        current_seals = analyzer.build_capture_artifact_seals(treatment_shards)
    except (analyzer.AnalysisContractError, merge.MergeContractError) as exc:
        raise VisualContractError(str(exc)) from exc
    if current_identity != report["capture_identity"]:
        _fail("current treatment capture identity differs from the analyzed report")
    if current_seals != report["capture_artifact_seals"]:
        _fail("current treatment receipt/score seals differ from the analyzed report")
    return Inputs(
        report_path=Path(report_path).resolve(),
        overlay_path=Path(overlay_path).resolve(),
        plan_dir=Path(plan_dir).resolve(),
        baseline_shard_root=Path(baseline_shard_root).resolve(),
        treatment_shard_root=Path(treatment_shard_root).resolve(),
        report=report,
        overlay=overlay,
        baseline_plan=baseline_plan,
        treatment_plan=treatment_plan,
        treatment_shards=treatment_shards,
    )


def _owner_area(owner: Mapping[str, Any]) -> float:
    x1, y1, x2, y2 = (float(v) for v in owner["bbox_pixel_xyxy"])
    return max(1.0, (x2 - x1) * (y2 - y1))


def _owner_sort_key(owner: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        int(owner["image_id"]),
        *tuple(float(v) for v in owner.get("owner_sort_key", [])),
        str(owner["gt_owner_id"]),
    )


def _admitted_context(report: Mapping[str, Any], owner_id: str) -> str:
    owners = _require_mapping(
        _require_mapping(report["owner_role_context_admission"], "admission").get("owners"),
        "admission.owners",
    )
    block = _require_mapping(owners.get(owner_id), f"admission owner {owner_id}")
    contexts = sorted(
        str(value)
        for value in _require_sequence(block.get("allowed_context_ids"), f"{owner_id} contexts")
    )
    if not contexts:
        _fail(f"owner {owner_id!r} has no admitted visualization context")
    return contexts[0]


def select_cases(
    report: Mapping[str, Any],
    owners: Mapping[str, Mapping[str, Any]],
    *,
    max_per_role: int = DEFAULT_MAX_PER_ROLE,
) -> list[Case]:
    """Select report outcomes deterministically, without reading score rows."""

    if max_per_role <= 0:
        _fail("max_per_role must be positive")
    primary = _require_mapping(report.get("primary"), "primary")
    primary_rows = list(_require_sequence(primary.get("owners"), "primary.owners"))
    seen_primary: set[str] = set()
    for row in primary_rows:
        row = _require_mapping(row, "primary owner")
        owner_id = str(row.get("gt_owner_id"))
        if owner_id in seen_primary:
            _fail(f"report primary owners contain duplicate {owner_id!r}")
        seen_primary.add(owner_id)
        if owner_id not in owners:
            _fail(f"report primary owner {owner_id!r} is absent from the plan")
        support = row.get("treatment_support_u")
        if not isinstance(support, bool):
            _fail(f"report primary owner {owner_id!r} has non-boolean treatment support")
        allowed = set(
            str(value)
            for value in _require_sequence(
                _require_mapping(
                    _require_mapping(
                        report["owner_role_context_admission"], "owner context admission"
                    ).get("owners"),
                    "owner context admission owners",
                ).get(owner_id, {}).get("allowed_context_ids"),
                f"{owner_id} allowed contexts",
            )
        )
        support_contexts = {
            str(value)
            for value in _require_sequence(
                row.get("treatment_support_context_ids_u"),
                f"{owner_id} treatment support contexts",
            )
        }
        if not support_contexts <= allowed:
            _fail(f"report primary owner {owner_id!r} support context is not frozen-allowed")
        if support and not support_contexts:
            _fail(f"recovered owner {owner_id!r} has no treatment support context")
        if not support and support_contexts:
            _fail(f"nonrecovered owner {owner_id!r} unexpectedly lists support contexts")

    recovered_rows = [row for row in primary_rows if row.get("treatment_support_u") is True]
    nonrecovered_rows = [row for row in primary_rows if row.get("treatment_support_u") is False]
    recovered_rows.sort(key=lambda row: _owner_sort_key(owners[str(row["gt_owner_id"])]))
    nonrecovered_rows.sort(key=lambda row: _owner_sort_key(owners[str(row["gt_owner_id"])]))
    selected_recovered = recovered_rows[:max_per_role]

    cases: list[Case] = []
    for row in selected_recovered:
        owner_id = str(row["gt_owner_id"])
        support_contexts = sorted(str(v) for v in row.get("treatment_support_context_ids_u", []))
        if not support_contexts:
            _fail(f"recovered owner {owner_id!r} has no treatment support context")
        cases.append(Case(owner_id, "recovered_persistent", support_contexts[0]))

    unused = {str(row["gt_owner_id"]) for row in nonrecovered_rows}
    for recovered in selected_recovered:
        if not unused:
            break
        recovered_id = str(recovered["gt_owner_id"])
        source = owners[recovered_id]
        source_area = _owner_area(source)

        def distance(candidate_id: str) -> tuple[Any, ...]:
            candidate = owners[candidate_id]
            return (
                str(candidate["image_id"]) != str(source["image_id"]),
                str(candidate["normalized_description"])
                != str(source["normalized_description"]),
                abs(math.log(_owner_area(candidate) / source_area)),
                _owner_sort_key(candidate),
            )

        control_id = min(unused, key=distance)
        unused.remove(control_id)
        cases.append(
            Case(
                control_id,
                "matched_nonrecovered_control",
                _admitted_context(report, control_id),
                matched_recovered_owner_id=recovered_id,
            )
        )

    retention = _require_mapping(report.get("retention"), "retention")
    for key, role in (
        ("confirmation_true_positive", "confirmation_retention_loss"),
        ("resolved_support", "resolved_retention_loss"),
    ):
        block = _require_mapping(retention.get(key), f"retention.{key}")
        owner_ids = sorted(str(v) for v in _require_sequence(block.get("lost"), f"{key}.lost"))
        for owner_id in owner_ids[:max_per_role]:
            if owner_id not in owners:
                _fail(f"retention-loss owner {owner_id!r} is absent from the plan")
            cases.append(Case(owner_id, role, _admitted_context(report, owner_id)))

    order = {
        "recovered_persistent": 0,
        "matched_nonrecovered_control": 1,
        "confirmation_retention_loss": 2,
        "resolved_retention_loss": 3,
    }
    return sorted(
        cases,
        key=lambda case: (order[case.role], _owner_sort_key(owners[case.owner_id])),
    )


def within_arm_confidence(logprobs: Mapping[str, float]) -> dict[str, float]:
    """Softmax over one arm's complete owner-local bank only."""

    if not logprobs:
        _fail("cannot normalize an empty owner-local bank")
    if any(not math.isfinite(float(value)) for value in logprobs.values()):
        _fail("owner-local bank contains a non-finite score")
    peak = max(float(value) for value in logprobs.values())
    exp = {key: math.exp(float(value) - peak) for key, value in logprobs.items()}
    total = math.fsum(exp.values())
    if not math.isfinite(total) or total <= 0.0:
        _fail("owner-local confidence normalization is not finite")
    return {key: value / total for key, value in exp.items()}


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = (float(v) for v in a)
    bx1, by1, bx2, by2 = (float(v) for v in b)
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return 0.0 if union <= 0.0 else intersection / union


def select_spatial_peaks(
    confidence: Mapping[str, float],
    boxes: Mapping[str, Sequence[float]],
    *,
    max_peaks: int = PEAK_COUNT,
    nms_iou: float = PEAK_NMS_IOU,
) -> list[str]:
    """Greedy score-ordered spatial NMS with deterministic ID tie-breaking."""

    if set(confidence) != set(boxes):
        _fail("confidence and coordinate candidate IDs differ")
    ordered = sorted(confidence, key=lambda key: (-float(confidence[key]), key))
    selected: list[str] = []
    for candidate_id in ordered:
        if all(_iou(boxes[candidate_id], boxes[prior]) < nms_iou for prior in selected):
            selected.append(candidate_id)
            if len(selected) == max_peaks:
                break
    return selected


def shared_local_crop(
    boxes: Sequence[Sequence[float]],
    *,
    width: int,
    height: int,
    padding: int = CROP_PADDING_PIXELS,
    minimum_extent: int = MIN_CROP_EXTENT_PIXELS,
) -> dict[str, Any]:
    """One score-independent crop shared byte-for-byte by both panels."""

    if not boxes:
        _fail("cannot construct an owner-local crop without geometry")
    x1 = max(0.0, min(float(box[0]) for box in boxes) - padding)
    y1 = max(0.0, min(float(box[1]) for box in boxes) - padding)
    x2 = min(float(width), max(float(box[2]) for box in boxes) + padding)
    y2 = min(float(height), max(float(box[3]) for box in boxes) + padding)

    def expand(lo: float, hi: float, bound: int) -> tuple[float, float]:
        if hi - lo >= minimum_extent:
            return lo, hi
        centre = (lo + hi) / 2.0
        lo = max(0.0, centre - minimum_extent / 2.0)
        hi = min(float(bound), lo + minimum_extent)
        lo = max(0.0, hi - minimum_extent)
        return lo, hi

    x1, x2 = expand(x1, x2, width)
    y1, y2 = expand(y1, y2, height)
    window = [int(math.floor(x1)), int(math.floor(y1)), int(math.ceil(x2)), int(math.ceil(y2))]
    return {
        "window_pixel_xyxy": window,
        "shared_across_arms": True,
        "derived_from": "gt_and_frozen_owner_local_candidate_geometry_never_scores",
        "presentation_only": True,
        "model_input": "separate_full_canvas_arm_rasters_never_this_crop",
        "padding_pixels": padding,
        "minimum_extent_pixels": minimum_extent,
    }


def _validate_receipt(
    receipt: Mapping[str, Any],
    *,
    image_id: str,
    plan_digest: str,
    overlay: Mapping[str, Any],
    treatment: bool,
) -> None:
    if receipt.get("status") != "captured" or str(receipt.get("image_id")) != image_id:
        _fail(f"{image_id} shard receipt is not a captured receipt for that image")
    plan_block = _require_mapping(receipt.get("plan"), f"{image_id} receipt.plan")
    if str(plan_block.get("receipt_content_sha256")) != plan_digest:
        _fail(f"{image_id} shard receipt belongs to another predecessor plan")
    if treatment:
        stamp = _require_mapping(receipt.get("intervention"), f"{image_id} intervention stamp")
        for key, wanted in (
            ("intervention_unit_id", UNIT_ID),
            ("arm_id", overlay["arm_id"]),
            ("baseline_arm_id", overlay["baseline_arm_id"]),
            ("overlay_content_sha256", overlay["overlay_content_sha256"]),
            ("base_plan_receipt_content_sha256", plan_digest),
        ):
            if str(stamp.get(key)) != str(wanted):
                _fail(f"{image_id} treatment receipt {key} does not match the overlay")
        completeness = _require_mapping(
            receipt.get("intervention_completeness"), f"{image_id} completeness"
        )
        wanted = {
            str(value)
            for value in _require_sequence(
                _require_mapping(
                    overlay.get("query_group_selection"), "overlay query-group selection"
                )
                .get("query_group_ids_by_image", {})
                .get(image_id),
                f"overlay selected groups for {image_id}",
            )
        }
        if completeness.get("status") != "complete_frozen_overlay_selection":
            _fail(f"{image_id} treatment shard is not complete for the frozen overlay")
        if completeness.get("is_complete_frozen_overlay_selection") is not True:
            _fail(f"{image_id} treatment completeness boolean is inconsistent")
        for key in ("missing_query_group_ids", "extra_query_group_ids"):
            values = _require_sequence(completeness.get(key), f"{image_id} completeness {key}")
            if values:
                _fail(f"{image_id} treatment completeness declares nonempty {key}")
        if int(completeness.get("frozen_expected_query_group_count", -1)) != len(wanted):
            _fail(f"{image_id} treatment completeness expected-group count is inconsistent")
        if int(completeness.get("executed_query_group_count", -1)) != len(wanted):
            _fail(f"{image_id} treatment completeness executed-group count is inconsistent")
        mode = _require_mapping(receipt.get("intervention_capture_mode"), f"{image_id} mode")
        if (
            mode.get("score_only") is not True
            or mode.get("behavior_sidecars_captured") is not False
            or mode.get("behavior_sidecars_intentionally_disabled") is not True
        ):
            _fail(f"{image_id} treatment shard is not score-only")
    elif receipt.get("capture_completeness") != "complete_shard":
        _fail(f"{image_id} baseline shard is not a complete predecessor shard")


def _score_index_for_group(
    *,
    shard_root: Path,
    image_id: str,
    query_group_id: str,
    candidate_ids: Sequence[str],
    plan: merge.PlanBundle,
    overlay: Mapping[str, Any],
    treatment: bool,
    validated_treatment_shard: analyzer.TreatmentShard | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    directory = (
        Path(validated_treatment_shard.directory)
        if validated_treatment_shard is not None
        else Path(shard_root) / image_id
    )
    receipt_path = directory / RECEIPT_NAME
    score_path = directory / SCORES_NAME
    receipt = (
        validated_treatment_shard.receipt
        if validated_treatment_shard is not None
        else _read_json(receipt_path, "shard receipt")
    )
    plan_digest = str(plan.receipt["receipt_content_sha256"])
    _validate_receipt(
        receipt,
        image_id=image_id,
        plan_digest=plan_digest,
        overlay=overlay,
        treatment=treatment,
    )
    wanted = set(candidate_ids)
    rows: dict[str, dict[str, Any]] = {}
    score_rows = (
        validated_treatment_shard.scores
        if validated_treatment_shard is not None
        else _read_jsonl(score_path, "census scores")
    )
    for row in score_rows:
        if str(row.get("query_group_id")) != query_group_id:
            continue
        candidate_id = str(row.get("candidate_id"))
        if candidate_id not in wanted:
            continue
        if candidate_id in rows:
            _fail(f"duplicate score row for {query_group_id!r}, {candidate_id!r}")
        if str(row.get("image_id")) != image_id:
            _fail(f"score row {candidate_id!r} joins to a foreign image")
        if str(row.get("context_id")) != query_group_id.rsplit("|", 1)[0]:
            _fail(f"score row {candidate_id!r} joins to a foreign context")
        group = plan.query_groups[query_group_id]
        if str(row.get("request_id")) != f"{query_group_id}|{candidate_id}":
            _fail(f"score row {candidate_id!r} has a noncanonical request_id")
        if str(row.get("normalized_description")) != str(group["normalized_description"]):
            _fail(f"score row {candidate_id!r} joins to a foreign description")
        expected_rank_key = {
            "image_id": image_id,
            "context_id": str(group["context_id"]),
            "normalized_description": str(group["normalized_description"]),
        }
        if not isinstance(row.get("rank_key"), Mapping) or dict(row["rank_key"]) != (
            expected_rank_key
        ):
            _fail(f"score row {candidate_id!r} has a foreign rank_key")
        if str(row.get("plan_receipt_content_sha256")) != plan_digest:
            _fail(f"score row {candidate_id!r} belongs to another plan")
        candidate = plan.candidates[candidate_id]
        if list(row.get("coord_token_ids") or []) != list(candidate["coord_token_ids"]):
            _fail(f"score row {candidate_id!r} coordinate token IDs drifted")
        if str(row.get("coord_token_ids_sha256")) != str(candidate["coord_token_ids_sha256"]):
            _fail(f"score row {candidate_id!r} coordinate digest drifted")
        if treatment:
            stamp = _require_mapping(row.get("intervention"), "treatment score stamp")
            receipt_stamp = _require_mapping(receipt.get("intervention"), "receipt stamp")
            if dict(stamp) != dict(receipt_stamp):
                _fail(f"treatment score row {candidate_id!r} stamp differs from its receipt")
        elif "intervention" in row:
            _fail(f"baseline score row {candidate_id!r} unexpectedly carries an intervention")
        rows[candidate_id] = row
    missing = sorted(wanted - set(rows))
    if missing:
        _fail(f"{query_group_id!r} is missing {len(missing)} owner-local rows: {missing[:3]!r}")
    if not treatment:
        # Reuse the predecessor's exact suffix/prefix/admission/candidate event
        # checks for the selected baseline rows.  Row counts outside this
        # visualization slice are intentionally not reinterpreted here.
        artifact = merge.ShardArtifacts(
            image_id=image_id,
            split=str(plan.images[image_id]["split"]),
            status="captured",
            directory=directory,
            receipt=receipt,
            scores=list(rows.values()),
            evidence_loaded=False,
        )
        try:
            admissions = merge.build_admission_index(plan, artifact)
            admitted = merge.admit_score_rows(plan, artifact, admissions)
        except merge.MergeContractError as exc:
            raise VisualContractError(str(exc)) from exc
        if len(admitted) != len(rows):
            _fail(f"baseline query group {query_group_id!r} did not admit every selected row")
    return rows, {
        "receipt": str(receipt_path.resolve()),
        "scores": str(score_path.resolve()),
        "receipt_sha256": planner.sha256_file(receipt_path),
        "scores_sha256": planner.sha256_file(score_path),
    }


def _owner_bank_ids(owner: Mapping[str, Any]) -> list[str]:
    bank = _require_mapping(owner.get("candidate_bank"), "owner candidate_bank")
    values = [
        str(value)
        for value in _require_sequence(
            bank.get("generator_local_landscape_candidate_ids"),
            "generator_local_landscape_candidate_ids",
        )
    ]
    if not values or len(values) != len(set(values)):
        _fail("owner local bank is empty or contains duplicate candidate IDs")
    return values


def build_case_spec(inputs: Inputs, case: Case) -> dict[str, Any]:
    owner = inputs.baseline_plan.owners.get(case.owner_id)
    if not isinstance(owner, Mapping):
        _fail(f"owner {case.owner_id!r} is absent from the plan")
    image_id = str(owner["image_id"])
    description = str(owner["normalized_description"])
    context = inputs.baseline_plan.contexts.get(case.context_id)
    if not isinstance(context, Mapping) or str(context.get("image_id")) != image_id:
        _fail(f"owner {case.owner_id!r} context is absent or belongs to another image")
    query_group_id = f"{case.context_id}|{description}"
    group = inputs.baseline_plan.query_groups.get(query_group_id)
    if not isinstance(group, Mapping) or group.get("status") != "admitted":
        _fail(f"query group {query_group_id!r} is absent or not admitted")

    candidate_ids = _owner_bank_ids(owner)
    group_ids = {str(value) for value in group.get("candidate_ids", [])}
    if not set(candidate_ids) <= group_ids:
        _fail(f"owner {case.owner_id!r} local bank is not contained in {query_group_id!r}")
    candidates: dict[str, Mapping[str, Any]] = {}
    for candidate_id in candidate_ids:
        candidate = inputs.baseline_plan.candidates.get(candidate_id)
        if not isinstance(candidate, Mapping):
            _fail(f"owner {case.owner_id!r} names unknown candidate {candidate_id!r}")
        if str(candidate.get("image_id")) != image_id or str(
            candidate.get("normalized_description")
        ) != description:
            _fail(f"candidate {candidate_id!r} joins to a foreign image or description")
        if case.owner_id not in {str(v) for v in candidate.get("generator_gt_owner_ids", [])}:
            _fail(f"candidate {candidate_id!r} is not generated by owner {case.owner_id!r}")
        candidates[candidate_id] = candidate

    baseline_rows, baseline_sources = _score_index_for_group(
        shard_root=inputs.baseline_shard_root,
        image_id=image_id,
        query_group_id=query_group_id,
        candidate_ids=candidate_ids,
        plan=inputs.baseline_plan,
        overlay=inputs.overlay,
        treatment=False,
    )
    treatment_rows, treatment_sources = _score_index_for_group(
        shard_root=inputs.treatment_shard_root,
        image_id=image_id,
        query_group_id=query_group_id,
        candidate_ids=candidate_ids,
        plan=inputs.treatment_plan,
        overlay=inputs.overlay,
        treatment=True,
        validated_treatment_shard=inputs.treatment_shards[image_id],
    )
    boxes = {
        candidate_id: [float(v) for v in candidates[candidate_id]["decoded_bbox_pixel_xyxy"]]
        for candidate_id in candidate_ids
    }

    def arm_spec(arm_id: str, rows: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
        confidence = within_arm_confidence(
            {
                candidate_id: float(rows[candidate_id]["complete_box_logprob_sum"])
                for candidate_id in candidate_ids
            }
        )
        peak_ids = select_spatial_peaks(confidence, boxes)
        display_max = max(confidence[candidate_id] for candidate_id in peak_ids)
        return {
            "arm_id": arm_id,
            "confidence_normalization": (
                "independent_softmax_over_this_arms_same_frozen_owner_local_bank"
            ),
            "confidence_cross_arm_comparable": False,
            "raw_logprob_emitted": False,
            "local_bank_candidate_count": len(candidate_ids),
            "spatial_peak_policy": {
                "max_peaks": PEAK_COUNT,
                "greedy_nms_iou_strictly_below": PEAK_NMS_IOU,
            },
            "peaks": [
                {
                    "candidate_id": candidate_id,
                    "bbox_pixel_xyxy": boxes[candidate_id],
                    "within_arm_confidence": confidence[candidate_id],
                    "within_arm_display_intensity": confidence[candidate_id] / display_max,
                    "representative_role": str(candidates[candidate_id]["representative_role"]),
                }
                for candidate_id in peak_ids
            ],
        }

    image = inputs.baseline_plan.images[image_id]
    crop = shared_local_crop(
        [owner["bbox_pixel_xyxy"], *boxes.values()],
        width=int(image["image_width"]),
        height=int(image["image_height"]),
    )
    return {
        "schema_version": VISUAL_SCHEMA_VERSION,
        "product": "paired_owner_local_landscape",
        "figure_id": f"paired_owner:{case.role}:{case.owner_id}:{case.context_id}",
        "selection_role": case.role,
        "gt_owner_id": case.owner_id,
        "matched_recovered_owner_id": case.matched_recovered_owner_id,
        "image_id": image_id,
        "split": str(owner["split"]),
        "context_id": case.context_id,
        "query_group_id": query_group_id,
        "normalized_description": description,
        "gt_bbox_pixel_xyxy": [float(v) for v in owner["bbox_pixel_xyxy"]],
        "canvas": {"width": int(image["image_width"]), "height": int(image["image_height"])},
        "crop": crop,
        "arms": [
            arm_spec(str(inputs.overlay["baseline_arm_id"]), baseline_rows),
            arm_spec(str(inputs.overlay["arm_id"]), treatment_rows),
        ],
        "comparison_semantics": {
            "same_owner": True,
            "same_context": True,
            "same_candidate_ids_and_coordinates": True,
            "same_crop_window": True,
            "raw_cross_arm_score_comparison": False,
            "scientific_outcome_source": "intervention_report_never_this_visualizer",
        },
        "provenance": {
            "report_path": str(inputs.report_path),
            "report_sha256": planner.sha256_file(inputs.report_path),
            "overlay_path": str(inputs.overlay_path),
            "overlay_content_sha256": str(inputs.overlay["overlay_content_sha256"]),
            "plan_receipt_content_sha256": str(
                inputs.baseline_plan.receipt["receipt_content_sha256"]
            ),
            "prompt_identity_by_arm": {
                "baseline": {
                    "observed_prefix_sha256": str(group["observed_prefix_sha256"]),
                    "query_prefix_sha256": str(group["query_prefix_sha256"]),
                },
                "treatment": {
                    "observed_prefix_sha256": str(
                        inputs.treatment_plan.query_groups[query_group_id][
                            "observed_prefix_sha256"
                        ]
                    ),
                    "query_prefix_sha256": str(
                        inputs.treatment_plan.query_groups[query_group_id][
                            "query_prefix_sha256"
                        ]
                    ),
                },
            },
            "baseline": baseline_sources,
            "treatment": treatment_sources,
            "row_ids": {
                "owner": case.owner_id,
                "context": case.context_id,
                "query_group": query_group_id,
                "candidates": candidate_ids,
                "baseline_requests": sorted(str(row["request_id"]) for row in baseline_rows.values()),
                "treatment_requests": sorted(str(row["request_id"]) for row in treatment_rows.values()),
            },
        },
    }


def _resolve_image_path(inputs: Inputs, image_id: str, image_root: Path) -> Path:
    image = inputs.baseline_plan.images[image_id]
    path = Path(image_root) / str(image["file_name"])
    if not path.is_file():
        _fail(f"presentation image does not exist: {path}")
    return path


def render_spec(spec: Mapping[str, Any], inputs: Inputs, output_dir: Path, image_root: Path) -> str:
    """Render a large paired crop with GT and independently shaded arm peaks."""

    from PIL import Image, ImageDraw, ImageFont

    image_id = str(spec["image_id"])
    source_path = _resolve_image_path(inputs, image_id, image_root)
    base = Image.open(source_path).convert("RGB")
    canvas = spec["canvas"]
    expected_size = (int(canvas["width"]), int(canvas["height"]))
    if base.size != expected_size:
        _fail(f"presentation image {source_path} size {base.size} != plan canvas {expected_size}")
    window = tuple(int(v) for v in spec["crop"]["window_pixel_xyxy"])
    crop = base.crop(window)
    panel_size = (640, 640)
    gap = 24
    header = 104
    footer = 92
    figure = Image.new("RGB", (panel_size[0] * 2 + gap, header + panel_size[1] + footer), (18, 18, 22))
    draw = ImageDraw.Draw(figure)
    font = ImageFont.load_default()
    scale_x = panel_size[0] / crop.width
    scale_y = panel_size[1] / crop.height

    def projected(box: Sequence[float], offset_x: int) -> list[float]:
        return [
            offset_x + (float(box[0]) - window[0]) * scale_x,
            header + (float(box[1]) - window[1]) * scale_y,
            offset_x + (float(box[2]) - window[0]) * scale_x,
            header + (float(box[3]) - window[1]) * scale_y,
        ]

    resized = crop.resize(panel_size, Image.Resampling.LANCZOS)
    for arm_index, arm in enumerate(spec["arms"]):
        offset_x = arm_index * (panel_size[0] + gap)
        figure.paste(resized, (offset_x, header))
        title = "baseline 1x" if arm_index == 0 else "treatment ~2x visual tokens"
        draw.text((offset_x + 8, 50), title, fill=(238, 238, 242), font=font)
        draw.text(
            (offset_x + 8, 68),
            "confidence normalized inside this arm only",
            fill=(185, 185, 195),
            font=font,
        )
        for rank, peak in enumerate(arm["peaks"], 1):
            intensity = float(peak["within_arm_display_intensity"])
            colour = (int(70 + 185 * intensity), int(70 + 135 * intensity), 35)
            box = projected(peak["bbox_pixel_xyxy"], offset_x)
            draw.rectangle(box, outline=colour, width=4)
            draw.text((box[0] + 3, box[1] + 3), f"P{rank}", fill=(255, 255, 245), font=font)
        draw.rectangle(projected(spec["gt_bbox_pixel_xyxy"], offset_x), outline=(55, 240, 120), width=5)

    draw.text(
        (8, 10),
        f"{spec['selection_role']} | owner {spec['gt_owner_id']} | "
        f"{spec['normalized_description']} | context {spec['context_id']}",
        fill=(245, 245, 248),
        font=font,
    )
    draw.text((8, header + panel_size[1] + 14), "green = exact GT; P1..P4 = arm-local spatial peaks", fill=(220, 220, 228), font=font)
    draw.text(
        (8, header + panel_size[1] + 34),
        "same crop and candidate coordinates; raw likelihoods are not cross-arm comparable",
        fill=(185, 185, 195),
        font=font,
    )
    if spec.get("matched_recovered_owner_id"):
        draw.text(
            (8, header + panel_size[1] + 54),
            f"matched recovered owner: {spec['matched_recovered_owner_id']}",
            fill=(185, 185, 195),
            font=font,
        )
    safe = str(spec["figure_id"]).replace(":", "__").replace("|", "__").replace("/", "_")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{safe}.png"
    figure.save(path)
    return str(path.resolve())


def build_manifest(inputs: Inputs, specs: Sequence[Mapping[str, Any]], rendered: Mapping[str, str]) -> dict[str, Any]:
    figures = []
    for spec in specs:
        figure_id = str(spec["figure_id"])
        provenance = _require_mapping(spec.get("provenance"), f"{figure_id}.provenance")
        figures.append(
            {
                "figure_id": figure_id,
                "selection_role": str(spec["selection_role"]),
                "gt_owner_id": str(spec["gt_owner_id"]),
                "context_id": str(spec["context_id"]),
                "spec_sha256": sha256_json(spec),
                "rendered_path": rendered.get(figure_id),
                "provenance": provenance,
            }
        )
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "intervention_unit_id": UNIT_ID,
        "authority": "analyzer_report_plus_sealed_plan_and_captured_score_rows",
        "reads_model": False,
        "recomputes_support_or_threshold": False,
        "recomputes_scientific_outcome": False,
        "raw_cross_arm_logprob_comparison": False,
        "confidence_normalization": "independent_within_each_arm_owner_local_bank",
        "same_crop_window_across_arms": True,
        "fixed_spatial_peak_count_maximum": PEAK_COUNT,
        "figure_count": len(figures),
        "selection_counts": {
            role: sum(1 for spec in specs if spec["selection_role"] == role)
            for role in sorted({str(spec["selection_role"]) for spec in specs})
        },
        "figures": figures,
        "report_path": str(inputs.report_path),
        "report_sha256": planner.sha256_file(inputs.report_path),
        "overlay_path": str(inputs.overlay_path),
        "overlay_content_sha256": str(inputs.overlay["overlay_content_sha256"]),
        "plan_receipt_content_sha256": str(
            inputs.baseline_plan.receipt["receipt_content_sha256"]
        ),
    }
    manifest["manifest_content_sha256"] = sha256_json(manifest)
    return manifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--plan-dir", type=Path, required=True)
    parser.add_argument("--baseline-shard-root", type=Path, required=True)
    parser.add_argument("--treatment-shard-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--image-root",
        type=Path,
        default=Path("/data/CoordExp/public_data/coco/rescale_32_1024_bbox"),
    )
    parser.add_argument("--max-per-role", type=int, default=DEFAULT_MAX_PER_ROLE)
    parser.add_argument("--specs-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        inputs = load_inputs(
            report_path=args.report,
            overlay_path=args.overlay,
            plan_dir=args.plan_dir,
            baseline_shard_root=args.baseline_shard_root,
            treatment_shard_root=args.treatment_shard_root,
        )
        cases = select_cases(
            inputs.report, inputs.baseline_plan.owners, max_per_role=args.max_per_role
        )
        specs = [build_case_spec(inputs, case) for case in cases]
        rendered: dict[str, str] = {}
        if not args.specs_only:
            for spec in specs:
                rendered[str(spec["figure_id"])] = render_spec(
                    spec, inputs, args.output_dir, args.image_root
                )
        manifest = build_manifest(inputs, specs, rendered)
    except VisualContractError as exc:
        print(f"visual contract error: {exc}", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / SPECS_NAME).write_bytes(
        b"".join(canonical_json_bytes(spec) + b"\n" for spec in specs)
    )
    (args.output_dir / MANIFEST_NAME).write_bytes(canonical_json_bytes(manifest) + b"\n")
    print(f"{len(specs)} paired owner figures; manifest {args.output_dir / MANIFEST_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
