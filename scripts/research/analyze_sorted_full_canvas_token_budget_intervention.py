#!/usr/bin/env python3
"""Analyzer for the sorted full-canvas visual-token-budget intervention.

Consumes the predecessor census presentation (owner summaries and owner-context
features) plus the *complete* treatment shards and the sealed overlay, and
answers one question: under a ~2x merged visual-token presentation of the same
raw optical information, how many owners the predecessor called
``persistent_no_tested_localization_support`` now carry tested localization
support?

Support semantics are the predecessor's, re-used rather than re-stated:
``merge_sorted_owner_accessibility_census_shards`` supplies the event table, the
owner-context feature reconstruction, ``peak_lift`` / ``local_concentration``,
the ``_summarize_bound`` support projection, the sealed epsilon (0.002), the
conjunction rule and the q10 quantile.  What this analyzer adds is that the
thresholds are re-derived **on the treatment arm's own discovery true-positive
due-context observations** -- because a threshold calibrated on 1x scores would
silently import a cross-arm comparison.

Raw log probabilities are never compared across arms.  Every cross-arm statement
this analyzer makes is a comparison of *dispositions* produced by two
independently calibrated support tests.

Primary report
--------------
* persistent owners outside image 4134: overall (63 expected) and person-only
  (51 expected), recovered support, image spread, Wilson intervals and
  leave-one-image-out sensitivity;
* retention gates: confirmation true-positive controls >= 90%, resolved-support
  owners >= 90%;
* image 4134's restricted stratum, descriptive only;
* a minimum-decisive outcome block.  Meeting it does **not** launch anything:
  scale-up is a lead decision, not an analyzer side effect.

A0 identity parity (that the 1x presentation still reproduces the predecessor's
own numbers on this checkout) is a separate smoke gate.  Baseline dispositions
here come from the predecessor presentation, never from a fresh 1x rerun.
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
from scripts.research import score_sorted_owner_accessibility_census_shard as census  # noqa: E402
from scripts.research import (  # noqa: E402
    score_sorted_full_canvas_token_budget_intervention_shard as treatment_scorer,
)

INTERVENTION_UNIT_ID = prepare.INTERVENTION_UNIT_ID
ARM_ID = prepare.ARM_ID
RESTRICTED_IMAGE_ID = prepare.RESTRICTED_IMAGE_ID
PERSISTENT_DISPOSITION = prepare.PERSISTENT_DISPOSITION
RESOLVED_DISPOSITION = prepare.RESOLVED_DISPOSITION
ROW_STAMP_KEY = treatment_scorer.ROW_STAMP_KEY
COMPLETENESS_KEY = treatment_scorer.COMPLETENESS_KEY
COMPLETENESS_COMPLETE = treatment_scorer.COMPLETENESS_COMPLETE

REPORT_SCHEMA_VERSION = "sorted-full-canvas-visual-token-budget-intervention-report.v1"

REPORT_NAME = "intervention-report.json"
MARKDOWN_NAME = "intervention-report.md"

#: Cohort sizes the predecessor published.  A drift here means the presentation
#: this analyzer was pointed at is not the one the probe was designed against,
#: so it is a fail-closed condition rather than a warning.
EXPECTED_PERSISTENT_OUTSIDE_RESTRICTED = 63
EXPECTED_PERSISTENT_OUTSIDE_RESTRICTED_PERSON = 51

#: The frozen retention cohorts.  Retention is measured over the *complete*
#: cohort, so both its size and its coverage under the treatment arm are gates:
#: a rate over "whatever happened to be captured" would rise every time a
#: control that would have failed went missing.
EXPECTED_CONFIRMATION_TP_CONTROLS = 71
EXPECTED_RESOLVED_SUPPORT_OWNERS = 114

#: The frozen calibration population: discovery-half native true positives at
#: their deterministic due contexts.  The unit freezes exactly this many
#: observations, so a short calibration set moves both thresholds and is
#: refused rather than silently re-quantiled.
EXPECTED_DISCOVERY_TP_CALIBRATION_OBSERVATIONS = 70

#: Minimum-decisive outcome.  Not a launch trigger.
MIN_DECISIVE_OVERALL = 13
MIN_DECISIVE_PERSON_ONLY = 10
MIN_DECISIVE_IMAGE_SPREAD = 3
RETENTION_GATE = 0.90

#: The primary bound.  ``l`` is reported alongside; a disposition that only
#: holds under one bound is never presented as settled.
PRIMARY_BOUND = "u"


class AnalysisContractError(RuntimeError):
    """Raised when an analysis precondition fails; the report is not written."""


def _fail(message: str) -> None:
    raise AnalysisContractError(message)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not Path(path).is_file():
        _fail(f"{label} is missing at {path}")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    if not Path(path).is_file():
        _fail(f"{label} is missing at {path}")
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise AnalysisContractError(f"{label} line {index} is not valid JSON") from exc
    return rows


@dataclass(frozen=True)
class TreatmentShard:
    image_id: str
    directory: Path
    receipt: dict[str, Any]
    scores: list[dict[str, Any]]
    proposals: list[dict[str, Any]]
    x1: list[dict[str, Any]]


CAPTURE_ARTIFACT_SEALS_SCHEMA_VERSION = (
    "sorted-full-canvas-visual-token-budget-capture-artifact-seals.v1"
)


def build_capture_artifact_seals(
    shards: Mapping[str, TreatmentShard],
) -> dict[str, Any]:
    """Seal the exact receipt and score bytes consumed by this analyzer.

    The score file is the conclusion-bearing numerical input and the receipt
    owns its capture/admission identity.  Both are sealed after shard admission
    and published in the report so a downstream presentation tool cannot read
    changed bytes under a stale scientific outcome.
    """

    per_image: dict[str, Any] = {}
    for image_id, shard in sorted(shards.items()):
        receipt_path = (Path(shard.directory) / "shard-receipt.json").resolve()
        scores_path = (Path(shard.directory) / "census-scores.jsonl").resolve()
        if not receipt_path.is_file() or not scores_path.is_file():
            _fail(f"shard {image_id} cannot seal its consumed receipt and score files")
        per_image[image_id] = {
            "image_id": image_id,
            "receipt_path": str(receipt_path),
            "receipt_sha256": planner.sha256_file(receipt_path),
            "census_scores_path": str(scores_path),
            "census_scores_sha256": planner.sha256_file(scores_path),
            "localization_score_row_count": len(shard.scores),
            "receipt_intervention_stamp_sha256": planner.sha256_json(
                shard.receipt.get(ROW_STAMP_KEY)
            ),
        }
    payload = {
        "schema_version": CAPTURE_ARTIFACT_SEALS_SCHEMA_VERSION,
        "role": "exact_receipt_and_score_bytes_consumed_by_this_analyzer",
        "image_count": len(per_image),
        "images": per_image,
    }
    payload["capture_artifact_seals_sha256"] = planner.sha256_json(payload)
    return payload


_MISSING = object()


def _nested(value: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = value
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return _MISSING
        current = current[key]
    return current


def _identity_digest(value: Any) -> str:
    return planner.sha256_json(value)


def _assert_expected_identity(
    actual: Mapping[str, Any], expected: Mapping[str, Any], *, label: str
) -> None:
    for key, expected_value in expected.items():
        if key not in actual:
            _fail(f"{label} is missing conclusion-bearing field {key!r}")
        if actual[key] != expected_value:
            _fail(
                f"{label}.{key}={actual[key]!r} disagrees with the sealed overlay "
                f"value {expected_value!r}"
            )


def capture_identity_summary(
    shards: Mapping[str, TreatmentShard], overlay: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate and freeze the conclusion-bearing identity of all captures.

    The default conclusion-bearing path requires the complete real-HF receipt
    identity.  Compact/non-HF fixtures may bypass this function only through
    ``load_treatment_shards(..., validate_capture_identity=False)``.
    """

    common_paths = (
        ("intervention_schema_version",),
        ("intervention_source_sha256",),
        ("code",),
        ("plan",),
        ("backend_identity", "backend"),
        ("backend_identity", "infer_config"),
        ("backend_identity", "model_identity"),
        ("backend_identity", "tokenizer_identity"),
        ("backend_identity", "adapter_identity"),
        ("backend_identity", "repetition_penalty_stratum"),
        ("backend_identity", "session_scope"),
        ("backend_identity", "is_real_model"),
        ("backend_identity", "usable_as_evidence"),
        ("backend_identity", "uses_model_generate"),
        ("granularity", "candidate_batch_size"),
        ("granularity", "bulk_scoring_path"),
        ("granularity", "candidate_batch_scope"),
        ("numerics", "matmul_precision"),
        ("numerics", "batched_path_max_abs_diff_bound"),
        ("runtime_invariants",),
    )
    required_per_shard_paths = (
        *common_paths,
        ("code", "executed_source_sha256"),
        ("code", "planner_source_sha256"),
        ("code", "runtime_seam_source_sha256"),
        ("plan", "plan_schema_version"),
        ("plan", "receipt_content_sha256"),
        ("plan", "capture_rules_sha256"),
        ("backend_identity", "executed_media_sha256"),
        ("backend_identity", "executed_prompt_token_count"),
        ("backend_identity", "image_grid_thw"),
        ("numerics", "batched_path_parity"),
        ("checks",),
        ("admission",),
        ("counts",),
        ("phase_order",),
        ("intervention_selection",),
    )
    reference: dict[tuple[str, ...], Any] = {}
    per_image: dict[str, Any] = {}
    images = overlay.get("images")

    for image_id, shard in sorted(shards.items()):
        receipt = shard.receipt
        stamp = receipt[ROW_STAMP_KEY]
        missing_identity_paths = [
            ".".join(path)
            for path in required_per_shard_paths
            if _nested(receipt, path) is _MISSING
        ]
        if missing_identity_paths:
            _fail(
                f"shard {image_id} is missing required conclusion-bearing capture identity "
                f"fields {missing_identity_paths!r}"
            )
        for field in (
            "code",
            "plan",
            "backend_identity",
            "granularity",
            "numerics",
            "runtime_invariants",
            "checks",
            "admission",
            "counts",
            "phase_order",
            "intervention_selection",
        ):
            if not isinstance(receipt[field], Mapping):
                _fail(f"shard {image_id} capture identity field {field!r} is not a mapping")
        if not isinstance(images, Mapping) or image_id not in images:
            _fail(f"sealed overlay carries no image identity for shard {image_id}")
        image_block = images[image_id]
        if not isinstance(image_block, Mapping):
            _fail(f"sealed overlay image identity for {image_id} is not a mapping")
        treatment = image_block.get("treatment")
        current = image_block.get("current")
        if not isinstance(treatment, Mapping) or not isinstance(current, Mapping):
            _fail(f"sealed overlay image {image_id} has no current/treatment identity")

        expected_stamp = {
            "intervention_unit_id": INTERVENTION_UNIT_ID,
            "arm_id": str(overlay["arm_id"]),
            "baseline_arm_id": str(overlay["baseline_arm_id"]),
            "overlay_content_sha256": str(overlay["overlay_content_sha256"]),
            "base_plan_receipt_content_sha256": str(
                overlay["base"]["plan_receipt_content_sha256"]
            ),
            "max_pixels": int(overlay["max_pixels"]),
            "image_grid_thw": list(treatment["image_grid_thw"]),
            "merged_visual_tokens": int(treatment["merged_visual_tokens"]),
            "baseline_merged_visual_tokens": int(current["merged_visual_tokens"]),
            "executed_media_sha256": str(treatment["executed_media_sha256"]),
            "media_width": int(treatment["width"]),
            "media_height": int(treatment["height"]),
            "prompt_token_ids_sha256": str(treatment["prompt_token_ids_sha256"]),
            "prompt_token_count": int(treatment["prompt_token_count"]),
            "baseline_prompt_token_ids_sha256": str(current["prompt_token_ids_sha256"]),
            "cross_arm_raw_logprob_comparison": "forbidden",
        }
        _assert_expected_identity(stamp, expected_stamp, label=f"shard {image_id} stamp")

        if (
            "intervention_schema_version" in receipt
            and receipt["intervention_schema_version"] != treatment_scorer.RECEIPT_SCHEMA_VERSION
        ):
            _fail(f"shard {image_id} carries an unrecognized intervention receipt schema")
        if (
            "intervention_source_sha256" in receipt
            and receipt["intervention_source_sha256"]
            != treatment_scorer.EXECUTED_SOURCE_SHA256
        ):
            _fail(f"shard {image_id} intervention_source_sha256 is not the frozen wrapper")
        receipt_code = receipt.get("code")
        if isinstance(receipt_code, Mapping):
            _assert_expected_identity(
                receipt_code,
                {
                    "executed_source_sha256": census.EXECUTED_SOURCE_SHA256,
                    "planner_source_sha256": planner.sha256_file(Path(planner.__file__).resolve()),
                    "runtime_seam_source_sha256": planner.sha256_file(
                        REPO_ROOT / "scripts/research/score_sorted_owner_basin_landscape.py"
                    ),
                },
                label=f"shard {image_id} capture code",
            )
        receipt_plan = receipt.get("plan")
        if isinstance(receipt_plan, Mapping):
            plan_expected = {
                "receipt_content_sha256": str(
                    overlay["base"]["plan_receipt_content_sha256"]
                )
            }
            for key in ("capture_rules_sha256", "plan_schema_version"):
                if key in overlay["base"]:
                    plan_expected[key] = overlay["base"][key]
            _assert_expected_identity(
                receipt_plan, plan_expected, label=f"shard {image_id} plan identity"
            )

        backend = receipt.get("backend_identity")
        if isinstance(backend, Mapping):
            if backend["backend"] != "hf":
                _fail(f"shard {image_id} backend_identity.backend is not 'hf'")
            sealed_infer_config = overlay["base"].get("infer_config")
            if sealed_infer_config is None:
                _fail("sealed overlay base carries no infer_config identity")
            if backend["infer_config"] != sealed_infer_config:
                _fail(
                    f"shard {image_id} backend_identity.infer_config disagrees with the "
                    "sealed overlay base infer config"
                )
            repetition_penalty = backend["repetition_penalty_stratum"]
            if (
                isinstance(repetition_penalty, bool)
                or not isinstance(repetition_penalty, (int, float))
                or float(repetition_penalty) != 1.0
            ):
                _fail(
                    f"shard {image_id} backend_identity.repetition_penalty_stratum is not 1.0"
                )
            if backend["is_real_model"] is not True:
                _fail(f"shard {image_id} backend_identity.is_real_model is not true")
            if backend["usable_as_evidence"] is not True:
                _fail(f"shard {image_id} backend_identity.usable_as_evidence is not true")
            if backend["uses_model_generate"] is not False:
                _fail(f"shard {image_id} backend_identity.uses_model_generate is not false")
            for field, expected_value in (
                ("executed_media_sha256", expected_stamp["executed_media_sha256"]),
                ("image_grid_thw", expected_stamp["image_grid_thw"]),
                ("executed_prompt_token_count", expected_stamp["prompt_token_count"]),
            ):
                if field in backend and backend[field] != expected_value:
                    _fail(
                        f"shard {image_id} backend_identity.{field} disagrees with its "
                        "sealed per-image intervention stamp"
                    )
        checks = receipt.get("checks")
        if isinstance(checks, Mapping) and any(value is not True for value in checks.values()):
            _fail(f"shard {image_id} has a failed conclusion-bearing capture check")
        admission = receipt.get("admission")
        if isinstance(admission, Mapping) and admission.get("all_admitted") is not True:
            _fail(f"shard {image_id} did not admit every executed query group")
        counts = receipt.get("counts")
        if isinstance(counts, Mapping):
            expected_counts = {
                "query_group_count": len({str(row["query_group_id"]) for row in shard.scores}),
                "localization_score_rows": len(shard.scores),
                "proposal_surface_rows": len(shard.proposals),
                "x1_diagnostic_rows": len(shard.x1),
                "free_decode_sidecar_rows": 0,
            }
            _assert_expected_identity(
                counts, expected_counts, label=f"shard {image_id} captured row counts"
            )
        phase_order = receipt.get("phase_order")
        if isinstance(phase_order, Mapping) and (
            phase_order.get("behavior_sidecars_captured") is not False
            or phase_order.get("behavior_sidecars_intentionally_disabled") is not True
        ):
            _fail(f"shard {image_id} did not preserve the score-only phase contract")
        intervention_selection = receipt.get("intervention_selection")
        if isinstance(intervention_selection, Mapping) and (
            intervention_selection.get("selection_is_frozen_in_the_overlay") is not True
            or intervention_selection.get("selection_uses_treatment_scores") is not False
        ):
            _fail(f"shard {image_id} selection is not frozen and score-independent")
        granularity = receipt.get("granularity")
        if isinstance(granularity, Mapping) and int(granularity.get("candidate_batch_size", -1)) != 16:
            _fail(f"shard {image_id} was not captured with the frozen candidate batch 16")
        numerics = receipt.get("numerics")
        if isinstance(numerics, Mapping):
            parity = numerics.get("batched_path_parity")
            bound = numerics.get("batched_path_max_abs_diff_bound")
            if isinstance(parity, Mapping):
                if parity.get("admitted") is not True or parity.get("all_argmax_parity") is not True:
                    _fail(f"shard {image_id} failed batched-path parity admission")
                if bound is None or float(parity.get("max_abs_diff_observed", math.inf)) > float(bound):
                    _fail(f"shard {image_id} exceeded the frozen batched-path tolerance")

        for rows, label in (
            (shard.scores, "score"),
            (shard.proposals, "proposal"),
            (shard.x1, "x1"),
        ):
            for row in rows:
                row_stamp = row.get(ROW_STAMP_KEY)
                if not isinstance(row_stamp, Mapping) or dict(row_stamp) != dict(stamp):
                    _fail(f"shard {image_id} has a {label} row with mixed intervention identity")

        for path in common_paths:
            value = _nested(receipt, path)
            if value is _MISSING:
                continue
            if path not in reference:
                reference[path] = value
            elif value != reference[path]:
                _fail(
                    f"mixed capture identity at {'.'.join(path)}: shard {image_id} "
                    "does not match the other captured shards"
                )

        per_image[image_id] = {
            "stamp": dict(stamp),
            "backend_executed_media_sha256": (
                None if not isinstance(backend, Mapping) else backend.get("executed_media_sha256")
            ),
            "backend_image_grid_thw": (
                None if not isinstance(backend, Mapping) else backend.get("image_grid_thw")
            ),
            "backend_executed_prompt_token_count": (
                None
                if not isinstance(backend, Mapping)
                else backend.get("executed_prompt_token_count")
            ),
        }

    # A field present in any receipt must be present in all receipts.
    for path in reference:
        missing = [
            image_id
            for image_id, shard in sorted(shards.items())
            if _nested(shard.receipt, path) is _MISSING
        ]
        if missing:
            _fail(
                f"capture identity field {'.'.join(path)} is missing from shards {missing!r}"
            )

    common_summary: dict[str, Any] = {}
    for path, value in sorted(reference.items()):
        key = ".".join(path)
        common_summary[key] = (
            value
            if path in {("code",), ("plan",)}
            or isinstance(value, (str, int, float, bool))
            or value is None
            else {"sha256": _identity_digest(value)}
        )
    return {
        "status": "complete_uniform_capture_identity",
        "image_count": len(shards),
        "common": common_summary,
        "per_image": per_image,
    }


def load_treatment_shards(
    shard_root: Path,
    overlay: Mapping[str, Any],
    *,
    validate_capture_identity: bool = True,
) -> dict[str, TreatmentShard]:
    """Load every selected shard, failing closed on anything incomplete or foreign."""

    shard_root = Path(shard_root)
    selection = overlay["query_group_selection"]["query_group_ids_by_image"]
    overlay_digest = str(overlay["overlay_content_sha256"])
    base_digest = str(overlay["base"]["plan_receipt_content_sha256"])
    arm_id = str(overlay["arm_id"])

    shards: dict[str, TreatmentShard] = {}
    for image_id in sorted(selection):
        directory = None
        for name in (image_id, f"shard-{image_id}"):
            candidate = shard_root / name
            if candidate.is_dir():
                directory = candidate
                break
        if directory is None:
            _fail(f"treatment shard for image {image_id!r} is missing under {shard_root}")
        if (directory / census.QUARANTINE_NAME).is_file():
            _fail(f"treatment shard for image {image_id!r} is quarantined; it carries no evidence")

        receipt = _read_json(directory / "shard-receipt.json", f"shard {image_id} receipt")
        stamp = receipt.get(ROW_STAMP_KEY)
        if not isinstance(stamp, Mapping):
            _fail(f"shard {image_id} receipt carries no intervention stamp")
        if str(stamp.get("intervention_unit_id")) != INTERVENTION_UNIT_ID:
            _fail(f"shard {image_id} belongs to another intervention unit")
        if str(stamp.get("arm_id")) != arm_id:
            _fail(
                f"shard {image_id} was captured under arm {stamp.get('arm_id')!r}, not "
                f"{arm_id!r}; arms are never pooled"
            )
        if str(stamp.get("overlay_content_sha256")) != overlay_digest:
            _fail(f"shard {image_id} was captured under a different sealed overlay")
        if str(stamp.get("base_plan_receipt_content_sha256")) != base_digest:
            _fail(f"shard {image_id} was captured against a different predecessor plan")
        if receipt.get("status") != "captured":
            _fail(f"shard {image_id} is not a captured shard")
        capture_mode = receipt.get("intervention_capture_mode") or {}
        if not capture_mode.get("score_only"):
            _fail(f"shard {image_id} is not a score-only capture")

        scores = _read_jsonl(directory / "census-scores.jsonl", f"shard {image_id} scores")
        proposals = _read_jsonl(
            directory / "proposal-surface.jsonl", f"shard {image_id} proposal surface"
        )
        x1 = _read_jsonl(directory / "x1-distributions.jsonl", f"shard {image_id} x1")

        executed = {str(row["query_group_id"]) for row in scores}
        wanted = {str(value) for value in selection[image_id]}

        # Intervention-relative completeness, declared by the capture itself.
        # The predecessor's ``capture_completeness`` is base-plan-relative and
        # reads "subset_smoke" even for a complete intervention capture, so it
        # is never consulted here -- but the successor must have declared this
        # block, and must have declared it complete.  A shard written before
        # this field existed fails closed rather than being assumed complete.
        completeness = receipt.get(COMPLETENESS_KEY)
        if not isinstance(completeness, Mapping):
            _fail(
                f"shard {image_id} carries no {COMPLETENESS_KEY!r} block; its coverage of "
                "the frozen overlay selection cannot be established (a capture predating "
                "this contract is not assumed complete)"
            )
        status = str(completeness.get("status"))
        if status != COMPLETENESS_COMPLETE:
            _fail(
                f"shard {image_id} declares {COMPLETENESS_KEY}.status={status!r}; a "
                f"conclusion-bearing analysis requires {COMPLETENESS_COMPLETE!r}"
            )
        if completeness.get("is_complete_frozen_overlay_selection") is not True:
            _fail(
                f"shard {image_id} {COMPLETENESS_KEY} block is internally inconsistent: "
                f"status is {COMPLETENESS_COMPLETE!r} but "
                "is_complete_frozen_overlay_selection is not true"
            )
        for field, expected in (
            ("missing_query_group_ids", "missing"),
            ("extra_query_group_ids", "extra"),
        ):
            declared = completeness.get(field)
            if not isinstance(declared, Sequence) or isinstance(declared, (str, bytes)):
                _fail(f"shard {image_id} {COMPLETENESS_KEY}.{field} is not a list")
            if declared:
                _fail(
                    f"shard {image_id} declares itself complete but lists {len(declared)} "
                    f"{expected} query groups"
                )
        if int(completeness.get("frozen_expected_query_group_count", -1)) != len(wanted):
            _fail(
                f"shard {image_id} was captured against a frozen selection of "
                f"{completeness.get('frozen_expected_query_group_count')!r} query groups, "
                f"but this overlay freezes {len(wanted)} for that image"
            )
        if int(completeness.get("executed_query_group_count", -1)) != len(executed):
            _fail(
                f"shard {image_id} declares {completeness.get('executed_query_group_count')!r} "
                f"executed query groups but its score rows cover {len(executed)}"
            )

        # The declared block is a claim; these two set comparisons are the
        # independent check of it, and are kept exactly as strict as before.
        missing = sorted(wanted - executed)
        if missing:
            _fail(
                f"shard {image_id} is incomplete: {len(missing)} selected query groups have no "
                f"score row (first: {missing[:3]!r})"
            )
        extra = sorted(executed - wanted)
        if extra:
            _fail(
                f"shard {image_id} scored {len(extra)} query groups outside the sealed "
                f"selection (first: {extra[:3]!r})"
            )

        for row in scores:
            row_stamp = row.get(ROW_STAMP_KEY)
            if not isinstance(row_stamp, Mapping):
                _fail(f"shard {image_id} has a score row without an intervention stamp")
            if str(row_stamp.get("arm_id")) != arm_id or str(
                row_stamp.get("overlay_content_sha256")
            ) != overlay_digest:
                _fail(f"shard {image_id} mixes arm or overlay identity inside one shard")
            value = row.get("complete_box_logprob_sum")
            if value is None or not math.isfinite(float(value)):
                _fail(f"shard {image_id} carries a non-finite complete_box_logprob_sum")

        shards[image_id] = TreatmentShard(
            image_id=image_id,
            directory=directory,
            receipt=receipt,
            scores=scores,
            proposals=proposals,
            x1=x1,
        )
    if validate_capture_identity:
        capture_identity_summary(shards, overlay)
    return shards


# ---------------------------------------------------------------------------
# Treatment feature reconstruction
# ---------------------------------------------------------------------------


def build_treatment_owner_contexts(
    plan: merge.PlanBundle, shards: Mapping[str, TreatmentShard]
) -> list[dict[str, Any]]:
    """Rebuild owner-context features from treatment shards, predecessor code path."""

    events: list[merge.ScoreEvent] = []
    proposal_surfaces: dict[str, Mapping[str, Any]] = {}
    for image_id in sorted(shards):
        shard = shards[image_id]
        validate_treatment_score_row_joins(plan, shard)
        artifacts = merge.ShardArtifacts(
            image_id=image_id,
            split=str(plan.shards[image_id]["split"]),
            status="captured",
            directory=shard.directory,
            receipt=shard.receipt,
            scores=shard.scores,
            proposals=shard.proposals,
            free_decodes=[],
            file_digests={},
        )
        admissions = merge.build_admission_index(plan, artifacts)
        events.extend(merge.admit_score_rows(plan, artifacts, admissions))
        for context_id, surface in merge.admit_proposal_rows(
            plan, artifacts, admissions
        ).items():
            proposal_surfaces[str(context_id)] = surface

    event_rows = merge.build_event_table(plan, events, sidecars={})
    return merge.build_owner_context_features(plan, event_rows, proposal_surfaces)


def validate_treatment_score_row_joins(
    plan: merge.PlanBundle, shard: TreatmentShard
) -> None:
    """Validate redundant row identity before predecessor event admission.

    ``merge.admit_score_rows`` owns suffix/prefix/admission/rank/candidate and
    coordinate semantics.  These extra equality checks cover the presentation
    identifiers that are intentionally redundant in each row, preventing a
    downstream exact-row join from accepting a foreign request or description
    merely because its coordinate tuple is otherwise admissible.
    """

    for index, row in enumerate(shard.scores):
        label = f"shard {shard.image_id} score row {index}"
        group_id = str(row.get("query_group_id"))
        group = plan.query_groups.get(group_id)
        if not isinstance(group, Mapping):
            _fail(f"{label} names unknown query group {group_id!r}")
        candidate_id = str(row.get("candidate_id"))
        expected_request_id = f"{group_id}|{candidate_id}"
        if str(row.get("request_id")) != expected_request_id:
            _fail(f"{label} request_id is not the canonical query-group/candidate ID")
        expected_description = str(group["normalized_description"])
        if str(row.get("normalized_description")) != expected_description:
            _fail(f"{label} normalized_description does not match its plan query group")
        expected_rank_key = {
            "image_id": shard.image_id,
            "context_id": str(group["context_id"]),
            "normalized_description": expected_description,
        }
        rank_key = row.get("rank_key")
        if not isinstance(rank_key, Mapping) or dict(rank_key) != expected_rank_key:
            _fail(f"{label} rank_key is not the exact plan query-group key")


def filter_owner_contexts_to_frozen_roles(
    plan: merge.PlanBundle,
    summaries: Sequence[Mapping[str, Any]],
    owner_contexts: Sequence[Mapping[str, Any]],
    *,
    overlay: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep only each owner's pre-treatment context set for its frozen role."""

    selection = overlay.get("query_group_selection")
    if not isinstance(selection, Mapping):
        _fail("overlay carries no frozen query-group selection")
    if selection.get("selection_policy") != (
        "frozen_before_any_treatment_score_never_score_selected"
    ):
        _fail("query-group selection is not sealed as pre-treatment and score-independent")
    cohorts = selection.get("cohort_owner_ids")
    due = selection.get("due_context_by_owner")
    selected_by_image = selection.get("query_group_ids_by_image")
    if not isinstance(cohorts, Mapping) or not isinstance(due, Mapping):
        _fail("overlay cannot reconstruct frozen cohort roles or due-context mappings")
    if not isinstance(selected_by_image, Mapping):
        _fail("overlay carries no selected query-group IDs by image")

    role_names = (
        "persistent_outside_restricted",
        "persistent_restricted",
        "discovery_tp_calibration",
        "confirmation_tp_retention",
        "resolved_retention",
    )
    missing_roles = [role for role in role_names if role not in cohorts]
    if missing_roles:
        _fail(f"overlay cannot reconstruct frozen owner roles {missing_roles!r}")

    by_owner: dict[str, Mapping[str, Any]] = {}
    for summary in summaries:
        owner_id = str(summary.get("gt_owner_id"))
        if owner_id in by_owner:
            _fail(f"predecessor summaries contain duplicate owner {owner_id!r}")
        by_owner[owner_id] = summary

    roles_by_owner: dict[str, list[str]] = {}
    for role in role_names:
        values = cohorts[role]
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            _fail(f"overlay cohort {role!r} is not a list")
        for value in values:
            owner_id = str(value)
            if owner_id not in by_owner:
                _fail(f"overlay cohort {role!r} names unknown owner {owner_id!r}")
            roles_by_owner.setdefault(owner_id, []).append(role)
    overlapping = {
        owner_id: roles for owner_id, roles in roles_by_owner.items() if len(roles) != 1
    }
    if overlapping:
        _fail(f"owner role mapping is not unique: {list(sorted(overlapping.items()))[:3]!r}")

    observed_by_owner: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in owner_contexts:
        owner_id = str(row["gt_owner_id"])
        context_id = str(row["context_id"])
        owner_rows = observed_by_owner.setdefault(owner_id, {})
        if context_id in owner_rows:
            _fail(f"owner {owner_id!r} has duplicate treatment context {context_id!r}")
        owner_rows[context_id] = row

    def _view(summary: Mapping[str, Any], name: str) -> str | None:
        block = summary.get("upper_bound_u")
        if not isinstance(block, Mapping):
            _fail(f"owner {summary.get('gt_owner_id')!r} has no frozen upper_bound_u summary")
        view = block.get(name)
        if view is None:
            return None
        if not isinstance(view, Mapping) or view.get("context_id") is None:
            _fail(
                f"owner {summary.get('gt_owner_id')!r} has an unresolved frozen {name!r} view"
            )
        return str(view["context_id"])

    def _usable(summary: Mapping[str, Any]) -> set[str]:
        block = summary.get("upper_bound_u")
        if not isinstance(block, Mapping):
            _fail(f"owner {summary.get('gt_owner_id')!r} has no frozen upper_bound_u summary")
        values = block.get("usable_support_context_ids")
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            _fail(
                f"owner {summary.get('gt_owner_id')!r} has no frozen usable-support context list"
            )
        return {str(value) for value in values}

    def _selected(image_id: str) -> set[str]:
        values = selected_by_image.get(image_id)
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            _fail(f"overlay has no selected query-group list for image {image_id!r}")
        return {str(value) for value in values}

    filtered: list[dict[str, Any]] = []
    audit_owners: dict[str, Any] = {}
    for owner_id, roles in sorted(roles_by_owner.items()):
        role = roles[0]
        summary = by_owner[owner_id]
        image_id = str(summary["image_id"])
        description = str(summary["normalized_description"])
        allowed: set[str]

        if role == "persistent_outside_restricted":
            allowed = {
                str(group["context_id"])
                for group in plan.query_groups.values()
                if group.get("status") == "admitted"
                and str(group["image_id"]) == image_id
                and str(group["normalized_description"]) == description
                and not bool(
                    plan.contexts[str(group["context_id"])]["loop_marking"]["loop_tail"]
                )
            }
        elif role == "persistent_restricted":
            allowed = {f"{RESTRICTED_IMAGE_ID}:boundary-000"}
            for name in (
                "primary_first_non_loop_minimal_abs_frontier",
                "diagnostic_best_all",
            ):
                context_id = _view(summary, name)
                if context_id is not None:
                    allowed.add(context_id)
        elif role == "discovery_tp_calibration":
            if owner_id not in due:
                _fail(f"calibration owner {owner_id!r} has no frozen due-context mapping")
            allowed = {str(due[owner_id])}
        elif role == "confirmation_tp_retention":
            allowed = _usable(summary)
            if owner_id in due:
                allowed.add(str(due[owner_id]))
        else:
            allowed = _usable(summary)
            if not allowed:
                _fail(f"resolved owner {owner_id!r} has no frozen usable-support context")
            if owner_id in due:
                allowed.add(str(due[owner_id]))
            frontier = _view(summary, "primary_first_non_loop_minimal_abs_frontier")
            if frontier is not None:
                allowed.add(frontier)

        if not allowed:
            _fail(f"owner {owner_id!r} role {role!r} resolves to no allowed contexts")
        selected = _selected(image_id)
        for context_id in sorted(allowed):
            context = plan.contexts.get(context_id)
            if not isinstance(context, Mapping) or str(context.get("image_id")) != image_id:
                _fail(
                    f"owner {owner_id!r} allowed context {context_id!r} is absent or foreign"
                )
            group_id = f"{context_id}|{description}"
            group = plan.query_groups.get(group_id)
            if not isinstance(group, Mapping) or group.get("status") != "admitted":
                _fail(
                    f"owner {owner_id!r} allowed context {context_id!r} has no admitted "
                    f"predecessor query group for {description!r}"
                )
            if group_id not in selected:
                _fail(
                    f"owner {owner_id!r} intended context {context_id!r} is missing from "
                    "the sealed overlay selection"
                )

        observed = observed_by_owner.get(owner_id, {})
        missing = sorted(allowed - set(observed))
        if missing:
            _fail(
                f"owner {owner_id!r} is missing {len(missing)} intended treatment contexts "
                f"(first: {missing[:3]!r})"
            )
        excluded = sorted(set(observed) - allowed)
        for context_id in sorted(allowed):
            filtered.append(dict(observed[context_id]))
        role_audit = {
            "allowed_context_ids": sorted(allowed),
            "observed_context_ids": sorted(allowed & set(observed)),
            "excluded_superset_context_ids": excluded,
            "allowed_context_count": len(allowed),
            "observed_context_count": len(allowed & set(observed)),
            "excluded_superset_context_count": len(excluded),
        }
        audit_owners[owner_id] = {
            "image_id": image_id,
            "normalized_description": description,
            "roles": {role: role_audit},
            **role_audit,
        }

    return filtered, {
        "selection_basis": "pre_treatment_overlay_and_predecessor_owner_summaries_only",
        "treatment_scores_used_for_context_selection": False,
        "owner_count": len(audit_owners),
        "role_owner_counts": {
            role: len({str(value) for value in cohorts[role]}) for role in role_names
        },
        "owners": audit_owners,
    }


def calibrate_treatment_support(
    plan: merge.PlanBundle,
    owner_contexts: Sequence[Mapping[str, Any]],
    *,
    overlay: Mapping[str, Any],
) -> merge.SupportCalibration:
    """Re-derive the q10 support thresholds on the *treatment* arm's own controls.

    Same population and same rule as the predecessor's Phase B: discovery-half
    native true positives, at their deterministic due context, excluding loop
    tails, calibrated on the minimum across the two ambiguity bounds, at the
    sealed primary quantile with the sealed epsilon.  Only the scores are the
    treatment arm's, which is exactly the point: no threshold crosses an arm.
    """

    contract = merge.load_support_contract(plan)
    due = merge.due_context_index(plan)
    contexts_by_owner: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in owner_contexts:
        contexts_by_owner.setdefault(str(row["gt_owner_id"]), {})[str(row["context_id"])] = row

    observations: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for owner_id, owner in sorted(plan.owners.items()):
        if not owner.get("native_true_positive"):
            continue
        if str(owner["split"]) != "discovery":
            continue
        entry = due.get(owner_id)
        if entry is None or entry["excluded"]:
            exclusions.append(entry or {"gt_owner_id": owner_id, "reason": "no_due_mapping"})
            continue
        row = contexts_by_owner.get(owner_id, {}).get(str(entry["due_context_id"]))
        if row is None:
            exclusions.append(
                {**entry, "excluded": True, "reason": "owner_not_tested_at_its_due_context"}
            )
            continue
        if row["loop_marking"]["loop_tail"]:
            exclusions.append({**entry, "excluded": True, "reason": "due_context_is_a_loop_tail"})
            continue
        bounds = row["localization"]["generator_local_max_excluding_other_owner_strict"]
        lifts = [
            bounds[key]["peak_lift"] for key in ("ambiguity_excluded_l", "ambiguity_included_u")
        ]
        concentrations = [
            bounds[key]["local_concentration"]
            for key in ("ambiguity_excluded_l", "ambiguity_included_u")
        ]
        if any(value is None for value in (*lifts, *concentrations)):
            exclusions.append(
                {**entry, "excluded": True, "reason": "no_observation_under_one_of_the_bounds"}
            )
            continue
        if not all(math.isfinite(float(value)) for value in (*lifts, *concentrations)):
            _fail(
                f"treatment calibration observation for {owner_id!r} is not finite; refusing "
                "to derive a threshold from it"
            )
        observations.append(
            {
                "gt_owner_id": owner_id,
                "image_id": str(owner["image_id"]),
                "normalized_description": str(owner["normalized_description"]),
                "due_context_id": str(entry["due_context_id"]),
                "peak_lift_min_over_bounds": min(float(value) for value in lifts),
                "local_concentration_min_over_bounds": min(
                    float(value) for value in concentrations
                ),
            }
        )

    if not observations:
        _fail(
            "no discovery native true positive produced a treatment due-context observation "
            "under both ambiguity bounds; the treatment thresholds cannot be calibrated and "
            "no support statement can be made"
        )

    # The calibration population is frozen, not "whatever survived".  Both
    # thresholds are quantiles of this set, so a missing control moves them --
    # and a missing *weak* control moves them upward, making the treatment arm
    # look harder to pass than the predecessor's.  Refuse rather than re-quantile.
    declared_cohort = (
        (overlay.get("query_group_selection") or {}).get("cohort_owner_ids") or {}
    ).get("discovery_tp_calibration")
    expected_owner_ids: set[str] | None = None
    if declared_cohort is not None:
        expected_owner_ids = {str(value) for value in declared_cohort}
        if len(expected_owner_ids) != EXPECTED_DISCOVERY_TP_CALIBRATION_OBSERVATIONS:
            _fail(
                f"the overlay freezes {len(expected_owner_ids)} discovery true-positive "
                f"calibration owners, but this unit freezes "
                f"{EXPECTED_DISCOVERY_TP_CALIBRATION_OBSERVATIONS}"
            )
    observed_owner_ids = {str(row["gt_owner_id"]) for row in observations}
    if len(observations) != EXPECTED_DISCOVERY_TP_CALIBRATION_OBSERVATIONS:
        _fail(
            f"the treatment arm produced {len(observations)} discovery true-positive "
            f"due-context calibration observations, but this unit freezes "
            f"{EXPECTED_DISCOVERY_TP_CALIBRATION_OBSERVATIONS}; the q"
            f"{merge.load_support_contract(plan).primary_quantile} thresholds would be taken "
            f"over a different population "
            f"({len(exclusions)} recorded exclusions)"
        )
    if expected_owner_ids is not None and observed_owner_ids != expected_owner_ids:
        missing = sorted(expected_owner_ids - observed_owner_ids)
        unexpected = sorted(observed_owner_ids - expected_owner_ids)
        _fail(
            "the treatment calibration population is not the overlay's frozen discovery "
            f"true-positive cohort (missing={missing[:3]!r}, unexpected={unexpected[:3]!r})"
        )

    lifts = [row["peak_lift_min_over_bounds"] for row in observations]
    concentrations = [row["local_concentration_min_over_bounds"] for row in observations]
    per_category: dict[str, int] = {}
    for row in observations:
        key = str(row["normalized_description"])
        per_category[key] = per_category.get(key, 0) + 1

    return merge.SupportCalibration(
        theta_peak_lift=merge._quantile(lifts, contract.primary_quantile),  # noqa: SLF001
        theta_local_concentration=merge._quantile(  # noqa: SLF001
            concentrations, contract.primary_quantile
        ),
        epsilon=contract.support_epsilon,
        quantile=contract.primary_quantile,
        observation_count=len(observations),
        per_category_counts=per_category,
        sensitivity={
            "role": "report_only_never_moves_a_threshold",
            "quantiles": {
                str(level): {
                    "theta_peak_lift": merge._quantile(lifts, level),  # noqa: SLF001
                    "theta_local_concentration": merge._quantile(  # noqa: SLF001
                        concentrations, level
                    ),
                }
                for level in contract.sensitivity_quantiles
            },
            "per_category": {},
        },
        exclusions=tuple(exclusions),
        consumed_shard_digests=(),
        capture_manifest_sha256=str(overlay["overlay_content_sha256"]),
        category_contribution_min=contract.category_contribution_min,
        underrepresented_flag=contract.underrepresented_flag,
        cross_context_delta_epsilon=contract.cross_context_delta_epsilon,
        statistics=contract.statistics,
    )


def summarize_owner_support(
    owner_contexts: Sequence[Mapping[str, Any]], *, calibration: merge.SupportCalibration
) -> dict[str, dict[str, Any]]:
    """Per-owner treatment support under both bounds, via the predecessor projection."""

    by_owner: dict[str, list[Mapping[str, Any]]] = {}
    for row in owner_contexts:
        by_owner.setdefault(str(row["gt_owner_id"]), []).append(row)

    out: dict[str, dict[str, Any]] = {}
    for owner_id, rows in sorted(by_owner.items()):
        ordered = sorted(rows, key=lambda row: int(row["boundary_index"]))
        out[owner_id] = {
            "lower_bound_l": merge._summarize_bound(  # noqa: SLF001
                ordered, bound="l", calibration=calibration
            ),
            "upper_bound_u": merge._summarize_bound(  # noqa: SLF001
                ordered, bound="u", calibration=calibration
            ),
            "tested_context_ids": [str(row["context_id"]) for row in ordered],
        }
    return out


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def wilson_interval(successes: int, total: int, *, z: float = 1.959963984540054) -> dict[str, Any]:
    """Wilson score interval for a binomial proportion."""

    if total <= 0:
        return {"point": None, "low": None, "high": None, "successes": successes, "total": total}
    phat = successes / total
    denominator = 1.0 + (z * z) / total
    centre = (phat + (z * z) / (2 * total)) / denominator
    spread = (
        z
        * math.sqrt((phat * (1.0 - phat) + (z * z) / (4 * total)) / total)
        / denominator
    )
    return {
        "point": phat,
        "low": max(0.0, centre - spread),
        "high": min(1.0, centre + spread),
        "successes": successes,
        "total": total,
        "method": "wilson_score_interval_z_1p96",
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _recovered(support: Mapping[str, Any] | None, bound: str) -> bool | None:
    if support is None:
        return None
    block = support["lower_bound_l" if bound == "l" else "upper_bound_u"]
    return None if block["usable_support"] is None else bool(block["usable_support"])


def build_report(
    *,
    overlay: Mapping[str, Any],
    summaries: Sequence[Mapping[str, Any]],
    treatment_support: Mapping[str, Any],
    calibration: merge.SupportCalibration,
    shards: Mapping[str, TreatmentShard],
    context_admission: Mapping[str, Any] | None = None,
    capture_identity: Mapping[str, Any] | None = None,
    capture_artifact_seals: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the machine-readable report; every gate is fail-closed above."""

    if not isinstance(capture_identity, Mapping) or capture_identity.get("status") != (
        "complete_uniform_capture_identity"
    ):
        _fail("report requires a complete_uniform_capture_identity capture receipt")
    if not isinstance(capture_artifact_seals, Mapping):
        _fail("report requires exact capture artifact seals")
    if capture_artifact_seals.get("schema_version") != (
        CAPTURE_ARTIFACT_SEALS_SCHEMA_VERSION
    ):
        _fail("report capture artifact seals have an unexpected schema_version")
    expected_seal = planner.sha256_json(
        {
            key: value
            for key, value in capture_artifact_seals.items()
            if key != "capture_artifact_seals_sha256"
        }
    )
    if capture_artifact_seals.get("capture_artifact_seals_sha256") != expected_seal:
        _fail("report capture artifact seals do not reconstruct their own digest")
    sealed_images = capture_artifact_seals.get("images")
    if not isinstance(sealed_images, Mapping):
        _fail("capture artifact seals images is missing or is not a mapping")
    if set(str(key) for key in sealed_images) != set(str(key) for key in shards):
        _fail("capture artifact seal image set does not match the analyzed shards")

    by_owner = {str(row["gt_owner_id"]): row for row in summaries}

    persistent_outside = sorted(
        owner_id
        for owner_id, summary in by_owner.items()
        if str(summary["disposition"]) == PERSISTENT_DISPOSITION
        and str(summary["image_id"]) != RESTRICTED_IMAGE_ID
    )
    persistent_person = [
        owner_id
        for owner_id in persistent_outside
        if str(by_owner[owner_id]["normalized_description"]) == "person"
    ]
    persistent_restricted = sorted(
        owner_id
        for owner_id, summary in by_owner.items()
        if str(summary["disposition"]) == PERSISTENT_DISPOSITION
        and str(summary["image_id"]) == RESTRICTED_IMAGE_ID
    )

    if len(persistent_outside) != EXPECTED_PERSISTENT_OUTSIDE_RESTRICTED:
        _fail(
            f"cohort drift: the predecessor presentation carries {len(persistent_outside)} "
            f"persistent owners outside image {RESTRICTED_IMAGE_ID}, expected "
            f"{EXPECTED_PERSISTENT_OUTSIDE_RESTRICTED}"
        )
    if len(persistent_person) != EXPECTED_PERSISTENT_OUTSIDE_RESTRICTED_PERSON:
        _fail(
            f"cohort drift: {len(persistent_person)} person-only persistent owners outside "
            f"image {RESTRICTED_IMAGE_ID}, expected "
            f"{EXPECTED_PERSISTENT_OUTSIDE_RESTRICTED_PERSON}"
        )

    def _cohort_rows(owner_ids: Sequence[str]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for owner_id in owner_ids:
            summary = by_owner[owner_id]
            support = treatment_support.get(owner_id)
            rows.append(
                {
                    "gt_owner_id": owner_id,
                    "image_id": str(summary["image_id"]),
                    "normalized_description": str(summary["normalized_description"]),
                    "split": str(summary["split"]),
                    "baseline_disposition": str(summary["disposition"]),
                    "baseline_usable_support_u": bool(
                        summary["upper_bound_u"]["usable_support"]
                    ),
                    "treatment_tested": support is not None,
                    "treatment_tested_context_count": (
                        0 if support is None else len(support["tested_context_ids"])
                    ),
                    "treatment_support_u": _recovered(support, "u"),
                    "treatment_support_l": _recovered(support, "l"),
                    "treatment_support_context_ids_u": (
                        []
                        if support is None
                        else list(support["upper_bound_u"]["usable_support_context_ids"])
                    ),
                }
            )
        return rows

    persistent_rows = _cohort_rows(persistent_outside)
    untested = [row["gt_owner_id"] for row in persistent_rows if not row["treatment_tested"]]
    if untested:
        _fail(
            f"{len(untested)} persistent owners outside image {RESTRICTED_IMAGE_ID} were never "
            f"tested under the treatment arm (first: {untested[:3]!r}); the selected shards "
            "are incomplete for the primary cohort"
        )

    recovered_rows = [row for row in persistent_rows if row["treatment_support_u"]]
    recovered_person = [
        row for row in recovered_rows if row["normalized_description"] == "person"
    ]
    recovery_images = sorted({row["image_id"] for row in recovered_rows})

    def _loio() -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for image_id in sorted({row["image_id"] for row in persistent_rows}):
            kept = [row for row in persistent_rows if row["image_id"] != image_id]
            kept_recovered = [row for row in kept if row["treatment_support_u"]]
            kept_person = [row for row in kept if row["normalized_description"] == "person"]
            out.append(
                {
                    "held_out_image_id": image_id,
                    "cohort_size": len(kept),
                    "recovered": len(kept_recovered),
                    "recovered_person_only": sum(
                        1 for row in kept_person if row["treatment_support_u"]
                    ),
                    "person_only_cohort_size": len(kept_person),
                    "rate": (len(kept_recovered) / len(kept)) if kept else None,
                }
            )
        return out

    def _retention(
        owner_ids: Sequence[str], label: str, *, expected_size: int
    ) -> dict[str, Any]:
        """Retention over the *complete* frozen cohort, never over what was tested.

        A rate taken over the tested subset silently rewards an incomplete
        capture: dropping the controls that would have failed raises the rate.
        So the cohort size is checked against the formal freeze, an untested
        cohort member is a contract error rather than a smaller denominator,
        and the denominator published is the whole cohort.
        """

        rows = _cohort_rows(sorted(owner_ids))
        if len(rows) != expected_size:
            _fail(
                f"cohort drift: the predecessor presentation carries {len(rows)} "
                f"{label!r} owners, expected {expected_size}"
            )
        untested = [row["gt_owner_id"] for row in rows if not row["treatment_tested"]]
        if untested:
            _fail(
                f"{len(untested)} {label!r} owners were never tested under the treatment arm "
                f"(first: {untested[:3]!r}); the retention gate must be measured over the "
                "complete frozen cohort, so an incomplete capture cannot be scored"
            )
        retained = [row for row in rows if row["treatment_support_u"]]
        rate = len(retained) / len(rows)
        return {
            "cohort": label,
            "cohort_size": len(rows),
            "denominator": "complete_frozen_cohort_never_the_tested_subset",
            "tested_under_treatment": len(rows),
            "untested_under_treatment": [],
            "retained": len(retained),
            "retention_rate": rate,
            "gate": RETENTION_GATE,
            "gate_met": bool(rate >= RETENTION_GATE),
            "lost": [row["gt_owner_id"] for row in rows if not row["treatment_support_u"]],
            "wilson": wilson_interval(len(retained), len(rows)),
        }

    confirmation_tp = [
        owner_id
        for owner_id, summary in by_owner.items()
        if summary.get("native_true_positive") and str(summary["split"]) == "confirmation"
    ]
    resolved = [
        owner_id
        for owner_id, summary in by_owner.items()
        if str(summary["disposition"]) == RESOLVED_DISPOSITION
    ]
    confirmation_retention = _retention(
        confirmation_tp,
        "confirmation_true_positive_control",
        expected_size=EXPECTED_CONFIRMATION_TP_CONTROLS,
    )
    resolved_retention = _retention(
        resolved,
        "resolved_tested_localization_support",
        expected_size=EXPECTED_RESOLVED_SUPPORT_OWNERS,
    )

    restricted_rows = _cohort_rows(persistent_restricted)
    restricted = {
        "image_id": RESTRICTED_IMAGE_ID,
        "role": "descriptive_only_never_a_primary_estimate",
        "context_union": "root_plus_frozen_u_frontier_plus_frozen_u_best_diagnostic",
        "cohort_size": len(restricted_rows),
        "tested_under_treatment": sum(1 for row in restricted_rows if row["treatment_tested"]),
        "recovered": sum(1 for row in restricted_rows if row["treatment_support_u"]),
        "owners": restricted_rows,
    }

    gates_intact = bool(
        confirmation_retention["gate_met"] and resolved_retention["gate_met"]
    )
    decision = {
        "role": "minimum_decisive_outcome_never_an_automatic_scale_up_trigger",
        "thresholds": {
            "recovered_overall_at_least": MIN_DECISIVE_OVERALL,
            "recovered_person_only_at_least": MIN_DECISIVE_PERSON_ONLY,
            "images_with_recovery_at_least": MIN_DECISIVE_IMAGE_SPREAD,
            "retention_gates_intact": True,
        },
        "observed": {
            "recovered_overall": len(recovered_rows),
            "recovered_person_only": len(recovered_person),
            "images_with_recovery": len(recovery_images),
            "retention_gates_intact": gates_intact,
        },
        "minimum_decisive_met": bool(
            len(recovered_rows) >= MIN_DECISIVE_OVERALL
            and len(recovered_person) >= MIN_DECISIVE_PERSON_ONLY
            and len(recovery_images) >= MIN_DECISIVE_IMAGE_SPREAD
            and gates_intact
        ),
        "next_step_is_a_lead_decision": True,
    }

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "complete_uniform_capture_identity",
        "intervention_unit_id": INTERVENTION_UNIT_ID,
        "arm_id": str(overlay["arm_id"]),
        "baseline_arm_id": str(overlay["baseline_arm_id"]),
        "overlay_content_sha256": str(overlay["overlay_content_sha256"]),
        "base_plan_receipt_content_sha256": str(
            overlay["base"]["plan_receipt_content_sha256"]
        ),
        "predecessor_run_root": str(overlay["base"]["predecessor_run_root"]),
        "comparison_semantics": {
            "raw_logprob_compared_across_arms": False,
            "compared_quantity": "support_disposition_under_arm_local_calibration",
            "primary_bound": PRIMARY_BOUND,
            "baseline_dispositions_source": "predecessor_presentation_never_a_fresh_1x_rerun",
            "a0_identity_parity_role": "separate_smoke_gate_not_part_of_this_report",
            "claim_scope": str(overlay["claim_scope"]),
        },
        "treatment_calibration": calibration.describe(),
        "presentation": {
            image_id: {
                "baseline_merged_visual_tokens": int(
                    block["current"]["merged_visual_tokens"]
                ),
                "treatment_merged_visual_tokens": int(
                    block["treatment"]["merged_visual_tokens"]
                ),
                "ratio": block["treatment"]["merged_visual_token_ratio_vs_current"],
            }
            for image_id, block in sorted((overlay["images"] or {}).items())
        },
        "captured_images": sorted(shards),
        "primary": {
            "cohort": "persistent_no_tested_localization_support_outside_restricted_image",
            "cohort_size": len(persistent_rows),
            "recovered": len(recovered_rows),
            "recovered_rate": wilson_interval(len(recovered_rows), len(persistent_rows)),
            "person_only_cohort_size": len(persistent_person),
            "recovered_person_only": len(recovered_person),
            "recovered_person_only_rate": wilson_interval(
                len(recovered_person), len(persistent_person)
            ),
            "images_with_recovery": recovery_images,
            "image_spread": len(recovery_images),
            "recovered_by_image": {
                image_id: sum(1 for row in recovered_rows if row["image_id"] == image_id)
                for image_id in sorted({row["image_id"] for row in persistent_rows})
            },
            "leave_one_image_out": _loio(),
            "owners": persistent_rows,
        },
        "retention": {
            "confirmation_true_positive": confirmation_retention,
            "resolved_support": resolved_retention,
            "gates_intact": gates_intact,
        },
        "restricted_stratum": restricted,
        "minimum_decisive_outcome": decision,
        "capture_identity": dict(capture_identity),
        "capture_artifact_seals": dict(capture_artifact_seals),
        "analyzer_source_sha256": planner.sha256_file(Path(__file__).resolve()),
        "merge_source_sha256": merge.MERGE_SOURCE_SHA256,
    }
    if context_admission is not None:
        report["owner_role_context_admission"] = dict(context_admission)
    return report


def render_markdown(report: Mapping[str, Any]) -> str:
    primary = report["primary"]
    retention = report["retention"]
    decision = report["minimum_decisive_outcome"]
    calibration = report["treatment_calibration"]

    def _pct(block: Mapping[str, Any]) -> str:
        if block.get("point") is None:
            return "n/a"
        return (
            f"{block['successes']}/{block['total']} = {block['point']:.3f} "
            f"[{block['low']:.3f}, {block['high']:.3f}]"
        )

    lines = [
        f"# Full-canvas token-budget intervention ({report['arm_id']})",
        "",
        f"- Baseline arm: `{report['baseline_arm_id']}` (dispositions from the predecessor "
        "presentation, not a fresh 1x rerun)",
        f"- Overlay: `{report['overlay_content_sha256'][:16]}...`",
        f"- Claim scope: {report['comparison_semantics']['claim_scope']}",
        "- Raw log probabilities are never compared across arms; only support dispositions "
        "under arm-local calibration are.",
        "",
        "## Treatment calibration (re-derived on this arm)",
        "",
        f"- discovery TP due-context observations: {calibration['observation_count']}",
        f"- theta_peak_lift = {calibration['theta_peak_lift']:.6f}, "
        f"theta_local_concentration = {calibration['theta_local_concentration']:.6f}, "
        f"epsilon = {calibration['epsilon']}, q = {calibration['quantile']}",
        "",
        "## Primary: persistent owners outside image 4134",
        "",
        f"- overall recovered: {_pct(primary['recovered_rate'])}",
        f"- person-only recovered: {_pct(primary['recovered_person_only_rate'])}",
        f"- image spread: {primary['image_spread']} "
        f"({', '.join(primary['images_with_recovery']) or 'none'})",
        "",
        "| held-out image | cohort | recovered | rate |",
        "| --- | --- | --- | --- |",
    ]
    for row in primary["leave_one_image_out"]:
        rate = "n/a" if row["rate"] is None else f"{row['rate']:.3f}"
        lines.append(
            f"| {row['held_out_image_id']} | {row['cohort_size']} | {row['recovered']} | {rate} |"
        )

    lines += [
        "",
        "## Retention gates",
        "",
        "| cohort | tested | retained | rate | gate met |",
        "| --- | --- | --- | --- | --- |",
    ]
    for key in ("confirmation_true_positive", "resolved_support"):
        block = retention[key]
        rate = "n/a" if block["retention_rate"] is None else f"{block['retention_rate']:.3f}"
        lines.append(
            f"| {block['cohort']} | {block['tested_under_treatment']} | {block['retained']} | "
            f"{rate} | {block['gate_met']} |"
        )

    restricted = report["restricted_stratum"]
    lines += [
        "",
        f"## Restricted stratum (image {restricted['image_id']}, descriptive only)",
        "",
        f"- cohort {restricted['cohort_size']}, tested {restricted['tested_under_treatment']}, "
        f"recovered {restricted['recovered']}",
        f"- context union: {restricted['context_union']}",
        "",
        "## Minimum-decisive outcome",
        "",
        f"- met: **{decision['minimum_decisive_met']}** "
        f"(>= {decision['thresholds']['recovered_overall_at_least']} overall, "
        f">= {decision['thresholds']['recovered_person_only_at_least']} person-only, "
        f">= {decision['thresholds']['images_with_recovery_at_least']} images, retention intact)",
        f"- observed: {json.dumps(decision['observed'], sort_keys=True)}",
        "- This is not a scale-up trigger. The next step is a lead decision.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def analyze(
    *,
    base_plan_dir: Path,
    overlay_path: Path,
    treatment_shard_root: Path,
    owner_summaries_jsonl: Path,
    output_dir: Path | None,
) -> dict[str, Any]:
    try:
        overlay = prepare.load_overlay(Path(overlay_path))
    except prepare.PrepareContractError as exc:
        raise AnalysisContractError(str(exc)) from exc

    plan = merge.load_plan(Path(base_plan_dir))
    try:
        prepare.assert_overlay_binds_plan(
            overlay,
            plan_receipt_content_sha256=str(plan.receipt["receipt_content_sha256"]),
            capture_rules_sha256=str(plan.receipt["capture_rules_sha256"]),
        )
        prepare.apply_overlay(
            overlay,
            images=plan.images,
            contexts=plan.contexts,
            query_groups=plan.query_groups,
            categories=plan.categories,
            candidates=plan.candidates,
            owners=plan.owners,
        )
    except prepare.PrepareContractError as exc:
        raise AnalysisContractError(str(exc)) from exc

    summaries = _read_jsonl(Path(owner_summaries_jsonl), "predecessor owner summaries")
    if str(overlay["base"]["owner_summaries_sha256"]) != planner.sha256_file(
        Path(owner_summaries_jsonl)
    ):
        _fail(
            "the predecessor owner summaries differ from the ones the overlay was sealed "
            "against; the cohort would not be the one this probe selected for"
        )

    shards = load_treatment_shards(Path(treatment_shard_root), overlay)
    capture_identity = capture_identity_summary(shards, overlay)
    capture_artifact_seals = build_capture_artifact_seals(shards)
    owner_contexts = build_treatment_owner_contexts(plan, shards)
    owner_contexts, context_admission = filter_owner_contexts_to_frozen_roles(
        plan, summaries, owner_contexts, overlay=overlay
    )
    calibration = calibrate_treatment_support(plan, owner_contexts, overlay=overlay)
    treatment_support = summarize_owner_support(owner_contexts, calibration=calibration)

    report = build_report(
        overlay=overlay,
        summaries=summaries,
        treatment_support=treatment_support,
        calibration=calibration,
        shards=shards,
        context_admission=context_admission,
        capture_identity=capture_identity,
        capture_artifact_seals=capture_artifact_seals,
    )

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / REPORT_NAME).write_bytes(
            planner.canonical_json_bytes(report) + b"\n"
        )
        (output_dir / MARKDOWN_NAME).write_text(render_markdown(report), encoding="utf-8")
    return report


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--base-plan-dir", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--treatment-shard-root", type=Path, required=True)
    parser.add_argument(
        "--owner-summaries",
        type=Path,
        required=True,
        help="predecessor phases/presentation/owner-summaries.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        report = analyze(
            base_plan_dir=args.base_plan_dir,
            overlay_path=args.overlay,
            treatment_shard_root=args.treatment_shard_root,
            owner_summaries_jsonl=args.owner_summaries,
            output_dir=args.output_dir,
        )
    except AnalysisContractError as exc:
        print(f"analysis error: {exc}", file=sys.stderr)
        return 2
    print(render_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
