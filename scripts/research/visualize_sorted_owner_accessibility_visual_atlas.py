#!/usr/bin/env python3
"""Visual atlas for the sorted owner accessibility phenotype census
(``2026-08-03-sorted-owner-accessibility-phenotype-census``).

Contract item 14: **JSONL artifacts and receipts are the authority.**  This
tool reads only the sealed plan registries and the captured shard/merged JSONL
rows.  It never opens a model, never recomputes a score, and never derives a
number that is not already present in an artifact row.  Every rendered figure
records, in ``visual-manifest.json``, the exact artifact files and row IDs it
was built from.

Five products
-------------
``owner_map``
    Per image: every censused owner, drawn with a short ID and a side legend.
``proposal_map``
    Per context: the boundary gate and the within-context category-routing
    ranks.  Proposal quantities are per context and per category only -- never
    per owner.
``localization_landscape``
    Per query group: the candidate bank drawn with within-context normalized
    confidence.  Never a 2-D heatmap of the ``x1`` distribution.
``owner_card``
    Per owner: bank adequacy, frontier summary, the calibrated local-peak
    support evidence (``peak_lift``, ``local_concentration``, disposition), and
    -- kept visibly apart -- the routing/competition rank and margin.
``feature_overview``
    Continuous feature distributions, reported separately for the discovery and
    confirmation halves.

Presentation rules
------------------
* Box labels are short IDs (``O3``, ``C7``); the full IDs live in a side legend.
* Confidence is **within-context normalized** (a softmax over one query group's
  complete-box log-likelihoods).  Raw log probabilities are never drawn and
  never compared across images.
* Presentation-only crops use fixed padding and always include the competition
  neighbourhood.  The model always saw the original full image; every cropped
  figure says so.
* **Support is never recomputed here.**  Whether an owner has support, and
  against which thresholds, is decided by the merge tool's sealed discovery
  calibration.  This atlas copies ``peak_lift``, ``local_concentration``, the
  calibrated support status, and the disposition verbatim.  It never applies a
  quantile, never compares a statistic to a threshold, and never reads rank
  ``1`` as support -- rank and margin are presented as a separate routing and
  competition surface.

Design
------
The data layer (``build_*_spec``) is pure and testable without any image
backend; the rendering layer turns a spec into a PNG.  ``--specs-only`` emits
the specs and the manifest without rendering.
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

UNIT_ID = planner.UNIT_ID
VISUAL_SCHEMA_VERSION = "sorted-owner-accessibility-census-visual.v1"
MANIFEST_SCHEMA_VERSION = "sorted-owner-accessibility-census-visual-manifest.v1"
MANIFEST_NAME = "visual-manifest.json"
#: Emitted by the merge tool, which owns this schema.
OWNER_SUMMARY_NAME = "owner-summaries.jsonl"

#: Shard files emitted by the scorer, which owns their schema.
SHARD_SCORES_NAME = "census-scores.jsonl"
SHARD_PROPOSAL_NAME = "proposal-surface.jsonl"
SHARD_FREE_DECODE_NAME = "free-decode-sidecars.jsonl"
SHARD_QUARANTINE_NAME = "shard-quarantine.json"

#: ``free-decode-sidecars.jsonl`` carries two different sidecar kinds; the free
#: *next row* is the per-context behavioural one, keyed by (image, context).
FREE_NEXT_ROW_KIND = "census_free_next_row_sidecar"
FREE_GREEDY_BOX_KIND = "census_free_greedy_box_sidecar"

#: Continuous local-peak statistics that decide support.  Copied verbatim from
#: the merge tool's owner summaries; never recomputed here.
SUPPORT_STATISTICS: tuple[str, ...] = ("peak_lift", "local_concentration")
#: Routing/competition fields, presented separately and never as support.
ROUTING_FIELDS: tuple[str, ...] = (
    "rank_within_group",
    "owner_rank_within_group",
    "owner_population_size",
    "margin_to_best_owner_in_group",
    "margin_semantics",
)

PRODUCTS: tuple[str, ...] = (
    "owner_map",
    "proposal_map",
    "localization_landscape",
    "owner_card",
    "feature_overview",
)

#: Fixed presentation-only crop padding, in pixels.  Never adaptive: a crop
#: that tightened around the subject would silently change how crowded a
#: neighbourhood looks between figures.
CROP_PADDING_PIXELS = 96

sha256_json = planner.sha256_json
canonical_json_bytes = planner.canonical_json_bytes


class VisualContractError(RuntimeError):
    """Raised when a visualization precondition or provenance rule fails."""


# ---------------------------------------------------------------------------
# Artifact loading (JSONL only)
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise VisualContractError(f"{label} is missing at {path}")
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise VisualContractError(f"{label} line {index} is not valid JSON") from exc
    return rows


@dataclass(frozen=True)
class Artifacts:
    """Every row this tool is allowed to read, with its provenance path."""

    plan_dir: Path
    sources: dict[str, str]
    images: list[dict[str, Any]]
    owners: list[dict[str, Any]]
    categories: list[dict[str, Any]]
    contexts: list[dict[str, Any]]
    candidates: list[dict[str, Any]]
    query_groups: list[dict[str, Any]]
    scores: list[dict[str, Any]]
    proposals: list[dict[str, Any]]
    free_next_rows: list[dict[str, Any]]
    owner_records: list[dict[str, Any]]

    def free_next_row(self, image_id: str, context_id: str) -> dict[str, Any] | None:
        """Join the free next-row sidecar by ``(image_id, context_id)``.

        Returns ``None`` when the shard did not publish one.  A missing sidecar
        is omitted from the figure, never fabricated or inferred.
        """

        for row in self.free_next_rows:
            if (
                str(row.get("image_id")) == str(image_id)
                and str(row.get("context_id")) == str(context_id)
            ):
                return row
        return None

    def owner_by_id(self) -> dict[str, dict[str, Any]]:
        return {str(row["gt_owner_id"]): row for row in self.owners}

    def candidate_by_id(self) -> dict[str, dict[str, Any]]:
        return {str(row["candidate_id"]): row for row in self.candidates}

    def image_by_id(self) -> dict[str, dict[str, Any]]:
        return {str(row["image_id"]): row for row in self.images}

    def context_by_id(self) -> dict[str, dict[str, Any]]:
        return {str(row["context_id"]): row for row in self.contexts}


def load_artifacts(
    plan_dir: Path, *, shard_root: Path | None = None, merged_dir: Path | None = None
) -> Artifacts:
    """Load the sealed plan and, when present, captured shard/merged rows.

    Verifies the plan receipt's digests first: a figure must never be rendered
    from a plan that does not reconstruct its own seal.
    """

    receipt_path = plan_dir / "receipt.json"
    if not receipt_path.is_file():
        raise VisualContractError(f"plan receipt is missing at {receipt_path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("unit_id") != UNIT_ID:
        raise VisualContractError("plan receipt unit_id does not match this unit")
    recomputed = sha256_json(
        {key: value for key, value in receipt.items() if key != "receipt_content_sha256"}
    )
    if recomputed != receipt.get("receipt_content_sha256"):
        raise VisualContractError("plan receipt does not reconstruct its own digest")

    sources = {"plan_receipt": str(receipt_path)}
    def load(name: str) -> list[dict[str, Any]]:
        sources[name] = str(plan_dir / name)
        return _read_jsonl(plan_dir / name, name)

    scores: list[dict[str, Any]] = []
    proposals: list[dict[str, Any]] = []
    free_next_rows: list[dict[str, Any]] = []
    if shard_root is not None:
        for shard_dir in sorted(Path(shard_root).glob("*")):
            if not shard_dir.is_dir():
                continue
            if (shard_dir / SHARD_QUARANTINE_NAME).is_file():
                # Contract item 15: evidence from a failed shard is never used.
                continue
            score_path = shard_dir / SHARD_SCORES_NAME
            proposal_path = shard_dir / SHARD_PROPOSAL_NAME
            free_decode_path = shard_dir / SHARD_FREE_DECODE_NAME
            if score_path.is_file():
                sources[f"scores:{shard_dir.name}"] = str(score_path)
                scores.extend(_read_jsonl(score_path, SHARD_SCORES_NAME))
            if proposal_path.is_file():
                sources[f"proposals:{shard_dir.name}"] = str(proposal_path)
                proposals.extend(_read_jsonl(proposal_path, SHARD_PROPOSAL_NAME))
            if free_decode_path.is_file():
                sources[f"free_decodes:{shard_dir.name}"] = str(free_decode_path)
                # Both sidecar kinds share this file; only the per-context free
                # next row is a proposal-map input.
                free_next_rows.extend(
                    row
                    for row in _read_jsonl(free_decode_path, SHARD_FREE_DECODE_NAME)
                    if str(row.get("row_kind")) == FREE_NEXT_ROW_KIND
                )

    owner_records: list[dict[str, Any]] = []
    if merged_dir is not None:
        # The merge tool owns this schema; ``owner-summaries.jsonl`` is its
        # emitted name.  This atlas reads it and never writes it.
        path = Path(merged_dir) / OWNER_SUMMARY_NAME
        if path.is_file():
            sources["owner_summaries"] = str(path)
            owner_records = _read_jsonl(path, OWNER_SUMMARY_NAME)

    return Artifacts(
        plan_dir=plan_dir,
        sources=sources,
        images=load("image-registry.jsonl"),
        owners=load("owner-registry.jsonl"),
        categories=load("category-registry.jsonl"),
        contexts=load("context-registry.jsonl"),
        candidates=load("candidate-bank.jsonl"),
        query_groups=load("query-group-registry.jsonl"),
        scores=scores,
        proposals=proposals,
        free_next_rows=free_next_rows,
        owner_records=owner_records,
    )


# ---------------------------------------------------------------------------
# Short IDs and within-context normalized confidence
# ---------------------------------------------------------------------------


def assign_short_ids(full_ids: Sequence[str], *, prefix: str) -> dict[str, str]:
    """Deterministic short label per full ID, with a legend mapping back.

    Box labels must stay short enough not to occlude the image; the full ID is
    recovered from the side legend, never from the box.
    """

    return {full_id: f"{prefix}{index}" for index, full_id in enumerate(full_ids)}


def within_context_confidence(logprobs: Mapping[str, float]) -> dict[str, float]:
    """Softmax over one query group's complete-box log-likelihoods.

    This is the only confidence this atlas draws.  It is normalized *within one
    ``(image, context, category)`` group*, so it is comparable inside a figure
    and deliberately not comparable across images -- which is exactly what
    contract item 15's ban on raw cross-image log-probability comparison
    requires.
    """

    if not logprobs:
        return {}
    values = list(logprobs.values())
    peak = max(values)
    exponentials = {key: math.exp(float(value) - peak) for key, value in logprobs.items()}
    total = math.fsum(exponentials.values())
    if total <= 0.0:
        uniform = 1.0 / len(logprobs)
        return {key: uniform for key in logprobs}
    return {key: value / total for key, value in exponentials.items()}


def fixed_padding_crop(
    boxes: Sequence[Sequence[float]], *, width: int, height: int, padding: int = CROP_PADDING_PIXELS
) -> dict[str, Any]:
    """Presentation-only crop with fixed padding around a competition neighbourhood.

    The crop covers *every* supplied box, so the competition neighbourhood is
    always included and a figure can never imply an owner was alone.  The model
    always saw the original full image; the returned spec says so explicitly.
    """

    if not boxes:
        window = [0, 0, int(width), int(height)]
    else:
        x1 = min(float(box[0]) for box in boxes)
        y1 = min(float(box[1]) for box in boxes)
        x2 = max(float(box[2]) for box in boxes)
        y2 = max(float(box[3]) for box in boxes)
        window = [
            int(max(0, math.floor(x1) - padding)),
            int(max(0, math.floor(y1) - padding)),
            int(min(width, math.ceil(x2) + padding)),
            int(min(height, math.ceil(y2) + padding)),
        ]
    return {
        "window_pixel_xyxy": window,
        "padding_pixels": int(padding),
        "padding_policy": "fixed_never_adaptive",
        "includes_competition_neighbourhood": True,
        "presentation_only": True,
        "model_input": "original_full_image_never_this_crop",
    }


# ---------------------------------------------------------------------------
# Spec builders (pure; no image backend)
# ---------------------------------------------------------------------------


def build_owner_map_spec(artifacts: Artifacts, image_id: str) -> dict[str, Any]:
    """Every censused owner in one image, with short IDs and a side legend."""

    image = artifacts.image_by_id().get(image_id)
    if image is None:
        raise VisualContractError(f"image {image_id!r} is not in the plan")
    owners = sorted(
        (row for row in artifacts.owners if str(row["image_id"]) == image_id),
        key=lambda row: (row["owner_sort_key"], str(row["gt_owner_id"])),
    )
    short = assign_short_ids([str(row["gt_owner_id"]) for row in owners], prefix="O")
    entries = []
    for row in owners:
        owner_id = str(row["gt_owner_id"])
        bank = row["candidate_bank"]
        entries.append(
            {
                "gt_owner_id": owner_id,
                "short_id": short[owner_id],
                "normalized_description": str(row["normalized_description"]),
                "bbox_pixel_xyxy": list(row["bbox_pixel_xyxy"]),
                "owner_sort_key": list(row["owner_sort_key"]),
                "native_true_positive": bool(row["native_true_positive"]),
                "calibration_role": str(row["calibration_role"]),
                "greedy_eligible": bool(row["greedy_eligible"]),
                "bank_coverage_status": str(bank["bank_coverage_status"]),
                "disposition_eligible": bool(bank["disposition_eligible"]),
            }
        )
    return {
        "schema_version": VISUAL_SCHEMA_VERSION,
        "product": "owner_map",
        "figure_id": f"owner_map:{image_id}",
        "image_id": image_id,
        "split": str(image["split"]),
        "canvas": {"width": int(image["image_width"]), "height": int(image["image_height"])},
        "owners": entries,
        "legend": [
            {"short_id": entry["short_id"], "full_id": entry["gt_owner_id"],
             "normalized_description": entry["normalized_description"]}
            for entry in entries
        ],
        "label_policy": "short_ids_on_boxes_full_ids_in_side_legend",
        "provenance": {
            "artifact_files": [artifacts.sources["owner-registry.jsonl"],
                               artifacts.sources["image-registry.jsonl"]],
            "row_ids": [entry["gt_owner_id"] for entry in entries],
        },
    }


def build_proposal_map_spec(artifacts: Artifacts, context_id: str) -> dict[str, Any]:
    """One context's boundary gate and within-context category routing ranks."""

    rows = [row for row in artifacts.proposals if str(row["context_id"]) == context_id]
    if not rows:
        raise VisualContractError(
            f"no captured proposal-surface row for context {context_id!r}"
        )
    row = rows[0]
    context = artifacts.context_by_id()[context_id]
    routing = sorted(
        row["category_routing_event"], key=lambda entry: int(entry["within_context_rank"])
    )
    free_next_row = artifacts.free_next_row(str(row["image_id"]), context_id)
    return {
        "schema_version": VISUAL_SCHEMA_VERSION,
        "product": "proposal_map",
        "figure_id": f"proposal_map:{context_id}",
        "context_id": context_id,
        "image_id": str(row["image_id"]),
        "split": str(row["split"]),
        "context_role": str(context["context_role"]),
        "loop_marking": context["loop_marking"],
        # (a) gate -- never described as description accessibility.
        "boundary_gate": {
            "continue_probability": float(row["boundary_gate"]["continue_probability"]),
            "stop_probability": float(row["boundary_gate"]["stop_probability"]),
            # The scorer emits a *log probability* margin, not a logit margin.
            "continue_vs_stop_logprob_margin": float(
                row["boundary_gate"]["continue_vs_stop_logprob_margin"]
            ),
            "margin_units": "natural_log_probability_difference",
            "margin_semantics": "logprob(object_ref_start) - logprob(im_end)",
            "semantics": str(row["boundary_gate"]["semantics"]),
        },
        # (b) category routing -- raw sequence sums and within-context ranks.
        "category_routing": [
            {
                "normalized_description": str(entry["normalized_description"]),
                "within_context_rank": int(entry["within_context_rank"]),
                "raw_sequence_logprob_sum": float(entry["raw_sequence_logprob_sum"]),
                "aggregation": str(entry["aggregation"]),
            }
            for entry in routing
        ],
        # Joined from free-decode-sidecars.jsonl by (image, context).  It is
        # behaviour, not a probability, and it is omitted rather than invented
        # when the shard published none.
        "free_next_row": _free_next_row_view(free_next_row),
        "free_next_row_available": free_next_row is not None,
        "emits_per_owner_proposal_probability": False,
        "includes_coordinate_scores": False,
        "provenance": {
            "artifact_files": [
                path for key, path in artifacts.sources.items() if key.startswith("proposals:")
            ]
            + (
                [
                    path
                    for key, path in artifacts.sources.items()
                    if key.startswith("free_decodes:")
                ]
                if free_next_row is not None
                else []
            ),
            "row_ids": [context_id]
            + ([str(free_next_row["sidecar_id"])] if free_next_row is not None else []),
        },
    }


def _free_next_row_view(row: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Copy the free next-row sidecar's behavioural fields, verbatim."""

    if row is None:
        return None
    return {
        "sidecar_id": row.get("sidecar_id"),
        "row_kind": row.get("row_kind"),
        "token_ids": list(row.get("token_ids") or []),
        "token_count": row.get("token_count"),
        "stop_reason": row.get("stop_reason"),
        "decode_mode": row.get("decode_mode"),
        "reached_im_end": row.get("reached_im_end"),
        "reached_box_end": row.get("reached_box_end"),
        "truncated_at_cap": row.get("truncated_at_cap"),
        "is_sidecar": True,
        "enters_core_ranks": False,
        "is_behavior_not_probability": True,
    }


def build_localization_landscape_spec(
    artifacts: Artifacts, query_group_id: str
) -> dict[str, Any]:
    """One query group's candidate bank under within-context normalized confidence."""

    rows = [row for row in artifacts.scores if str(row["query_group_id"]) == query_group_id]
    if not rows:
        raise VisualContractError(f"no captured score rows for query group {query_group_id!r}")
    candidates = artifacts.candidate_by_id()
    logprobs = {
        str(row["candidate_id"]): float(row["complete_box_logprob_sum"]) for row in rows
    }
    confidence = within_context_confidence(logprobs)
    ordered = sorted(rows, key=lambda row: (-float(row["complete_box_logprob_sum"]),
                                            str(row["candidate_id"])))
    short = assign_short_ids([str(row["candidate_id"]) for row in ordered], prefix="C")

    entries = []
    boxes: list[Sequence[float]] = []
    for row in ordered:
        candidate_id = str(row["candidate_id"])
        candidate = candidates[candidate_id]
        box = [float(v) for v in candidate["decoded_bbox_pixel_xyxy"]]
        boxes.append(box)
        entries.append(
            {
                "candidate_id": candidate_id,
                "short_id": short[candidate_id],
                "representative_role": str(candidate["representative_role"]),
                "candidate_class": str(candidate["candidate_class"]),
                "candidate_provenance": str(candidate["candidate_provenance"]),
                "bbox_pixel_xyxy": box,
                "within_context_confidence": confidence[candidate_id],
                "competition_rank": int(row["competition"]["rank"]),
                "generator_gt_owner_ids": list(candidate["generator_gt_owner_ids"]),
                "cross_owner_generated": bool(candidate["cross_owner_generated"]),
                "strict_assignment_status": str(candidate["strict_assignment_status"]),
                "strict_assignment_gt_owner_id": candidate["strict_assignment_gt_owner_id"],
                "is_sidecar": bool(row.get("is_sidecar", False)),
            }
        )

    first = rows[0]
    image = artifacts.image_by_id()[str(first["image_id"])]
    return {
        "schema_version": VISUAL_SCHEMA_VERSION,
        "product": "localization_landscape",
        "figure_id": f"localization_landscape:{query_group_id}",
        "query_group_id": query_group_id,
        "rank_key": first["rank_key"],
        "image_id": str(first["image_id"]),
        "context_id": str(first["context_id"]),
        "normalized_description": str(first["normalized_description"]),
        "canvas": {"width": int(image["image_width"]), "height": int(image["image_height"])},
        "candidates": entries,
        "legend": [
            {
                "short_id": entry["short_id"],
                "full_id": entry["candidate_id"],
                "role": entry["representative_role"],
            }
            for entry in entries
        ],
        "confidence_normalization": "within_context_softmax_over_one_query_group",
        "raw_logprobs_drawn": False,
        "cross_image_comparison": False,
        "x1_distribution_rendered": False,
        "x1_distribution_policy": "diagnostic_only_never_a_2d_heatmap",
        "crop": fixed_padding_crop(
            boxes, width=int(image["image_width"]), height=int(image["image_height"])
        ),
        "label_policy": "short_ids_on_boxes_full_ids_in_side_legend",
        "provenance": {
            "artifact_files": [
                path for key, path in artifacts.sources.items() if key.startswith("scores:")
            ]
            + [artifacts.sources["candidate-bank.jsonl"]],
            "row_ids": [str(row["request_id"]) for row in ordered],
        },
    }


def build_owner_card_spec(artifacts: Artifacts, gt_owner_id: str) -> dict[str, Any]:
    """One owner's bank adequacy, frontier summary, and per-context maxima."""

    owner = artifacts.owner_by_id().get(gt_owner_id)
    if owner is None:
        raise VisualContractError(f"owner {gt_owner_id!r} is not in the plan")
    bank = owner["candidate_bank"]
    record = next(
        (row for row in artifacts.owner_records if str(row.get("gt_owner_id")) == gt_owner_id),
        None,
    )
    owner_scores = [
        row for row in artifacts.scores if _score_row_generated_by(row, gt_owner_id)
    ]
    cross_owner_scores = [
        row for row in owner_scores if len(_generator_owner_ids(row)) > 1
    ]

    spec: dict[str, Any] = {
        "schema_version": VISUAL_SCHEMA_VERSION,
        "product": "owner_card",
        "figure_id": f"owner_card:{gt_owner_id}",
        "gt_owner_id": gt_owner_id,
        "image_id": str(owner["image_id"]),
        "split": str(owner["split"]),
        "normalized_description": str(owner["normalized_description"]),
        "bbox_pixel_xyxy": list(owner["bbox_pixel_xyxy"]),
        "native_true_positive": bool(owner["native_true_positive"]),
        "calibration_role": str(owner["calibration_role"]),
        "bank": {
            "logical_role_count": int(bank["logical_role_count"]),
            "distinct_physical_candidate_count": int(bank["distinct_physical_candidate_count"]),
            "uniquely_assigned_candidate_count": int(
                bank["strict_assignment_coverage"]["uniquely_assigned_candidate_count"]
            ),
            "other_owner_strict_count": int(bank["other_owner_strict_count"]),
            "bank_coverage_status": str(bank["bank_coverage_status"]),
            "disposition_eligible": bool(bank["disposition_eligible"]),
        },
        "captured_score_row_count": len(owner_scores),
        "score_attribution": {
            "rule": "exact_membership_in_generator_gt_owner_ids",
            "representative_generator_shortcut": False,
            "cross_owner_shared_score_row_count": len(cross_owner_scores),
            "cross_owner_shared_candidate_ids": sorted(
                {str(row["candidate_id"]) for row in cross_owner_scores}
            ),
        },
        "provenance": {
            "artifact_files": [artifacts.sources["owner-registry.jsonl"]],
            "row_ids": [gt_owner_id],
        },
    }
    if record is not None:
        # The merge tool owns the frontier, support, and disposition semantics.
        # Everything below is copied verbatim; nothing is recomputed and no
        # threshold is applied here.
        spec["owner_record"] = {
            key: record[key]
            for key in (
                "loop_tail_only_support",
                "never_frontier_tested",
                "frontier_tested",
                "has_non_loop_primary_context",
                "tested_context_count",
                "non_loop_tested_context_count",
                "exact_anchor_score_reported",
                "ambiguity_bound_disposition_flip",
                "disposition",
                "disposition_blockers",
                "support_criterion",
                "bank_adequacy",
                "bank_report",
                "threshold_category_support",
            )
            if key in record
        }
        spec["calibrated_support"] = _calibrated_support_view(record)
        # Presented separately, and explicitly not a support input.
        spec["routing_surface"] = {
            **(record.get("routing_summary") or {}),
            "role": "routing_and_competition_surface_never_a_support_input",
        }
        spec["provenance"]["artifact_files"].append(artifacts.sources["owner_summaries"])
    spec["recomputed_any_statistic"] = False
    spec["applied_any_threshold"] = False
    spec["support_source"] = "merge_sealed_discovery_calibration"
    return spec


def _generator_owner_ids(row: Mapping[str, Any]) -> list[str]:
    """The owners that generated a localization score row's candidate.

    Score rows carry ``generator_gt_owner_ids`` (a list), never a scalar
    ``gt_owner_id``: after cross-owner collapse one scored candidate can belong
    to several same-category owners at once.
    """

    return [str(value) for value in (row.get("generator_gt_owner_ids") or [])]


def _score_row_generated_by(row: Mapping[str, Any], gt_owner_id: str) -> bool:
    """Exact membership in ``generator_gt_owner_ids``.

    Deliberately not a representative-generator shortcut: a candidate shared by
    two owners is attributed to *both*, so a cross-owner candidate stays
    visible on each owner's card instead of silently belonging to whichever
    generator happened to sort first.
    """

    return str(gt_owner_id) in _generator_owner_ids(row)


def _bound_view(bound: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Copy one ambiguity bound's local-peak evidence, verbatim."""

    if not bound:
        return None
    view: dict[str, Any] = {
        "bound": bound.get("bound"),
        "usable_support": bound.get("usable_support"),
        "support_calibrated": bound.get("support_calibrated"),
        "support_criterion_id": bound.get("support_criterion_id"),
        "never_frontier_tested": bound.get("never_frontier_tested"),
        "loop_tail_only_support": bound.get("loop_tail_only_support"),
        "tested_context_count": bound.get("tested_context_count"),
        "non_loop_tested_context_count": bound.get("non_loop_tested_context_count"),
    }
    for key in ("primary_best_non_loop", "primary_first_non_loop_minimal_abs_frontier"):
        projected = bound.get(key)
        if not projected:
            view[key] = None
            continue
        view[key] = {
            "context_id": projected.get("context_id"),
            "loop_tail": projected.get("loop_tail"),
            **{name: projected.get(name) for name in SUPPORT_STATISTICS},
            "routing": {name: projected.get(name) for name in ROUTING_FIELDS},
            "exact_anchor_score": projected.get("exact_anchor_score"),
        }
    # Diagnostic only: includes loop-tail contexts, so it never sits beside the
    # primary summaries without saying what it is.
    diagnostic = bound.get("diagnostic_best_all")
    view["diagnostic_best_all"] = (
        {
            "context_id": diagnostic.get("context_id"),
            "loop_tail": diagnostic.get("loop_tail"),
            **{name: diagnostic.get(name) for name in SUPPORT_STATISTICS},
        }
        if diagnostic
        else None
    )
    view["diagnostic_best_all_role"] = bound.get(
        "diagnostic_best_all_role", "diagnostic_only_includes_loop_tail_contexts"
    )
    return view


def _calibrated_support_view(record: Mapping[str, Any]) -> dict[str, Any]:
    """Both ambiguity bounds' calibrated support evidence, copied verbatim."""

    return {
        "statistics": list(SUPPORT_STATISTICS),
        "lower_bound_l": _bound_view(record.get("lower_bound_l")),
        "upper_bound_u": _bound_view(record.get("upper_bound_u")),
        "disposition": record.get("disposition"),
        "ambiguity_bound_disposition_flip": record.get("ambiguity_bound_disposition_flip"),
        "thresholds_applied_here": False,
        "thresholds_owner": "merge_sealed_discovery_calibration_receipt",
        "rank_is_support_criterion": False,
    }


def build_feature_overview_spec(artifacts: Artifacts) -> dict[str, Any]:
    """Continuous feature distributions, reported per split and never pooled."""

    def describe(values: Sequence[float]) -> dict[str, Any]:
        """Plain descriptive spread. Never a quantile threshold, never a cut."""

        ordered = sorted(float(value) for value in values)
        if not ordered:
            return {"count": 0}
        return {
            "count": len(ordered),
            "min": ordered[0],
            "median": ordered[len(ordered) // 2],
            "max": ordered[-1],
            "mean": math.fsum(ordered) / len(ordered),
        }

    def support_statistics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Distributions of the calibrated local-peak statistics, as reported.

        Read from the merge tool's upper-bound primary non-loop summary; this
        function never compares a value to a threshold.
        """

        collected: dict[str, list[float]] = {name: [] for name in SUPPORT_STATISTICS}
        for row in rows:
            projected = (row.get("upper_bound_u") or {}).get("primary_best_non_loop") or {}
            for name in SUPPORT_STATISTICS:
                value = projected.get(name)
                if isinstance(value, (int, float)):
                    collected[name].append(float(value))
        return {name: describe(values) for name, values in collected.items()}

    def disposition_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in rows:
            key = str(row.get("disposition", "unreported"))
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items()))

    per_split: dict[str, Any] = {}
    for split in ("discovery", "confirmation"):
        owners = [row for row in artifacts.owners if str(row["split"]) == split]
        contexts = [row for row in artifacts.contexts if str(row["split"]) == split]
        records = [row for row in artifacts.owner_records if str(row.get("split")) == split]
        per_split[split] = {
            "image_ids": sorted(
                {str(row["image_id"]) for row in artifacts.images if str(row["split"]) == split},
                key=int,
            ),
            "owner_count": len(owners),
            "true_positive_owner_count": sum(1 for row in owners if row["native_true_positive"]),
            "context_count": len(contexts),
            "loop_tail_context_count": sum(
                1 for row in contexts if row["loop_marking"]["loop_tail"]
            ),
            "non_loop_context_count": sum(
                1 for row in contexts if not row["loop_marking"]["loop_tail"]
            ),
            "bank_coverage_status_counts": {
                status: sum(
                    1 for row in owners if row["candidate_bank"]["bank_coverage_status"] == status
                )
                for status in planner.BANK_COVERAGE_STATUSES
            },
            "owner_summary_count": len(records),
            # Continuous features first: the local-peak statistics that decide
            # support, reported as distributions, never as a pass/fail count
            # derived here.
            "support_statistic_distributions": support_statistics(records),
            "disposition_counts": disposition_counts(records),
            "underrepresented_categories": sorted(
                {
                    str(row["normalized_description"])
                    for row in records
                    if isinstance(row.get("threshold_category_support"), Mapping)
                    and row["threshold_category_support"].get("flag")
                    == "pooled_underrepresented"
                }
            ),
        }
    return {
        "schema_version": VISUAL_SCHEMA_VERSION,
        "product": "feature_overview",
        "figure_id": "feature_overview",
        "per_split": per_split,
        "splits_pooled": False,
        "loop_tail_pooled_with_ordinary_contexts": False,
        "continuous_features_first": True,
        "support_statistics": list(SUPPORT_STATISTICS),
        "thresholds_applied_here": False,
        "recomputed_any_support": False,
        "provenance": {
            "artifact_files": sorted(
                {
                    artifacts.sources["owner-registry.jsonl"],
                    artifacts.sources["context-registry.jsonl"],
                    artifacts.sources["image-registry.jsonl"],
                }
            ),
            "row_ids": ["*"],
        },
    }


# ---------------------------------------------------------------------------
# Atlas assembly and manifest
# ---------------------------------------------------------------------------


def build_atlas_specs(
    artifacts: Artifacts,
    *,
    image_ids: Sequence[str] | None = None,
    max_proposal_maps: int = 12,
    max_landscapes: int = 12,
    max_owner_cards: int = 24,
) -> list[dict[str, Any]]:
    """Build every available product, skipping ones with no captured evidence."""

    wanted = (
        [str(v) for v in image_ids]
        if image_ids is not None
        else sorted({str(row["image_id"]) for row in artifacts.images}, key=int)
    )
    specs: list[dict[str, Any]] = [
        build_owner_map_spec(artifacts, image_id) for image_id in wanted
    ]

    proposal_contexts = sorted(
        {str(row["context_id"]) for row in artifacts.proposals if str(row["image_id"]) in wanted}
    )[:max_proposal_maps]
    specs.extend(build_proposal_map_spec(artifacts, context_id) for context_id in proposal_contexts)

    groups = sorted(
        {str(row["query_group_id"]) for row in artifacts.scores if str(row["image_id"]) in wanted}
    )[:max_landscapes]
    specs.extend(build_localization_landscape_spec(artifacts, group) for group in groups)

    owner_ids = sorted(
        {str(row["gt_owner_id"]) for row in artifacts.owners if str(row["image_id"]) in wanted}
    )[:max_owner_cards]
    specs.extend(build_owner_card_spec(artifacts, owner_id) for owner_id in owner_ids)

    specs.append(build_feature_overview_spec(artifacts))
    return specs


def build_manifest(
    artifacts: Artifacts, specs: Sequence[Mapping[str, Any]], *, rendered: Mapping[str, str]
) -> dict[str, Any]:
    """Bind every figure to the artifact files and row IDs it was built from."""

    figures = []
    for spec in specs:
        figure_id = str(spec["figure_id"])
        provenance = spec["provenance"]
        if not provenance.get("artifact_files"):
            raise VisualContractError(f"figure {figure_id!r} has no artifact provenance")
        figures.append(
            {
                "figure_id": figure_id,
                "product": str(spec["product"]),
                "rendered_path": rendered.get(figure_id),
                "spec_sha256": sha256_json(spec),
                "artifact_files": sorted(set(provenance["artifact_files"])),
                "row_ids": list(provenance["row_ids"]),
            }
        )
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "authority": "jsonl_artifacts_and_receipts",
        "reads_any_model": False,
        "recomputes_any_score": False,
        "recomputes_any_support_or_threshold": False,
        "support_statistics": list(SUPPORT_STATISTICS),
        "support_thresholds_owner": "merge_sealed_discovery_calibration_receipt",
        "rank_presented_as": "routing_and_competition_surface_never_support",
        "products": sorted({str(spec["product"]) for spec in specs}),
        "expected_products": list(PRODUCTS),
        "figure_count": len(figures),
        "figures": figures,
        "source_artifacts": dict(sorted(artifacts.sources.items())),
        "presentation_rules": {
            "box_labels": "short_ids_with_side_legend",
            "confidence": "within_context_normalized",
            "crop_padding_pixels": CROP_PADDING_PIXELS,
            "crop_padding_policy": "fixed_never_adaptive",
            "crops_include_competition_neighbourhood": True,
            "model_input": "original_full_image",
            "x1_distribution_rendered_as_2d_heatmap": False,
            "raw_cross_image_logprob_comparison": False,
            "evidence_from_quarantined_shards": False,
        },
    }
    manifest["manifest_content_sha256"] = sha256_json(manifest)
    return manifest


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _resolve_image_path(artifacts: Artifacts, image_id: str) -> Path | None:
    row = artifacts.image_by_id().get(image_id)
    if row is None:
        return None
    file_name = str(row.get("file_name") or "")
    if not file_name:
        return None
    candidate = Path("/data/CoordExp/public_data/coco/rescale_32_1024_bbox") / file_name
    return candidate if candidate.is_file() else None


def render_spec(spec: Mapping[str, Any], artifacts: Artifacts, output_dir: Path) -> str:
    """Render one spec to a PNG.

    Rendering is intentionally thin: everything decision-bearing already lives
    in the spec, so a figure cannot disagree with its manifest.
    """

    from PIL import Image, ImageDraw

    product = str(spec["product"])
    figure_id = str(spec["figure_id"])
    safe = figure_id.replace(":", "__").replace("|", "__").replace("/", "_")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{safe}.png"

    legend_width = 320
    if product in {"owner_map", "localization_landscape"}:
        canvas = spec["canvas"]
        width, height = int(canvas["width"]), int(canvas["height"])
        source = _resolve_image_path(artifacts, str(spec["image_id"]))
        if source is not None:
            base = Image.open(source).convert("RGB").resize((width, height))
        else:
            base = Image.new("RGB", (width, height), (28, 28, 32))
        figure = Image.new("RGB", (width + legend_width, height), (18, 18, 20))
        figure.paste(base, (0, 0))
        draw = ImageDraw.Draw(figure)

        entries = spec.get("owners") or spec.get("candidates") or []
        for entry in entries:
            box = [float(v) for v in entry["bbox_pixel_xyxy"]]
            confidence = float(entry.get("within_context_confidence", 1.0))
            intensity = int(80 + 175 * max(0.0, min(1.0, confidence)))
            colour = (intensity, 90, 255 - intensity)
            draw.rectangle(box, outline=colour, width=3)
            draw.text((box[0] + 3, box[1] + 3), str(entry["short_id"]), fill=(255, 255, 255))

        draw.text((width + 12, 10), figure_id, fill=(230, 230, 230))
        for index, item in enumerate(spec.get("legend", [])[:40]):
            text = f"{item['short_id']}  {item['full_id']}"
            draw.text((width + 12, 34 + index * 14), text[:44], fill=(190, 190, 200))
        figure.save(path)
        return str(path)

    # Text products: proposal maps, owner cards, and the feature overview.
    lines = [figure_id, ""]
    for key, value in spec.items():
        if key in {"provenance", "figure_id", "schema_version"}:
            continue
        rendered = json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else str(value)
        lines.append(f"{key}: {rendered[:160]}")
    figure = Image.new("RGB", (1100, max(220, 24 + 16 * len(lines))), (18, 18, 20))
    draw = ImageDraw.Draw(figure)
    for index, line in enumerate(lines):
        draw.text((12, 12 + index * 16), line[:150], fill=(220, 220, 225))
    figure.save(path)
    return str(path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--plan-dir", type=Path, required=True)
    parser.add_argument("--shard-root", type=Path, default=None)
    parser.add_argument("--merged-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-id", action="append", default=None)
    parser.add_argument(
        "--specs-only",
        action="store_true",
        help="emit specs and the manifest without rendering any image",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        artifacts = load_artifacts(
            args.plan_dir, shard_root=args.shard_root, merged_dir=args.merged_dir
        )
        specs = build_atlas_specs(artifacts, image_ids=args.image_id)
        rendered: dict[str, str] = {}
        if not args.specs_only:
            for spec in specs:
                rendered[str(spec["figure_id"])] = render_spec(spec, artifacts, args.output_dir)
        manifest = build_manifest(artifacts, specs, rendered=rendered)
    except VisualContractError as exc:
        print(f"visual contract error: {exc}", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "visual-specs.jsonl").write_bytes(
        b"".join(canonical_json_bytes(spec) + b"\n" for spec in specs)
    )
    (args.output_dir / MANIFEST_NAME).write_bytes(canonical_json_bytes(manifest) + b"\n")
    print(
        f"{manifest['figure_count']} figures; products {manifest['products']}; "
        f"manifest {args.output_dir / MANIFEST_NAME}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
