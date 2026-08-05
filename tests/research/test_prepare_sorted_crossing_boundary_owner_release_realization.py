"""Tests for the sorted crossing-boundary owner release/realization CPU planner.

Every test builds a self-contained synthetic census + prevalence run under
``tmp_path``: no test reads the real immutable predecessor runs, and no test
loads a model, a tokenizer, or a score artifact.

The fixture reproduces the exact frozen shape the planner enforces -- ``114``
supported false-negative owners, ``141`` native true positives, ``26`` U-bound
crossing owners (``12`` strict-matched ``E`` rows, ``14`` unmatched), ``25``
under L, ``24`` under both at the same crossing context, ``14`` disjoint timing
controls, and one due-boundary true-positive replay control per image -- so a
single mutation can be used to prove each fail-closed guard.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping
import copy
import json
from pathlib import Path
from typing import Any

import pytest

from scripts.research import (
    analyze_sorted_supported_fn_native_prefix_reachability_prevalence as prevalence,
)
from scripts.research import build_sorted_owner_accessibility_census_plan as planner
from scripts.research import merge_sorted_owner_accessibility_census_shards as merge
from scripts.research import prepare_sorted_crossing_boundary_owner_release_realization as builder

# ---------------------------------------------------------------------------
# Frozen fixture geometry
# ---------------------------------------------------------------------------

IMAGE_IDS: tuple[str, ...] = tuple(str(700 + index) for index in range(12))

#: 12 * 12 - 3 = 141 native true positives, one strict-matched native row each.
TP_PER_IMAGE: dict[str, int] = {
    image_id: (12 if index < 9 else 11) for index, image_id in enumerate(IMAGE_IDS)
}
#: 2 + 2 + 10 = 14 unmatched native rows, one per unmatched-E primary owner.
UNMATCHED_ROWS_PER_IMAGE: dict[str, int] = {
    image_id: (2 if index < 2 else 1) for index, image_id in enumerate(IMAGE_IDS)
}
#: 2 + 2 + 10 = 14 disjoint timing controls.
TIMING_PER_IMAGE: dict[str, int] = {
    image_id: (2 if index < 2 else 1) for index, image_id in enumerate(IMAGE_IDS)
}
#: 11 * 6 + 7 = 73 inert false-negative owners.
FILLER_PER_IMAGE: dict[str, int] = {
    image_id: (6 if index < 11 else 7) for index, image_id in enumerate(IMAGE_IDS)
}

WRAPPERS: dict[str, int] = {
    "object_ref_start": 151646,
    "object_ref_end": 151647,
    "box_start": 151648,
    "box_end": 151649,
    "im_end": 151645,
}
COORDINATE_TOKEN_IDS: dict[str, int] = {"start": 151670, "end_inclusive": 152669, "bin_count": 1000}

DESCRIPTION_TOKENS: dict[str, list[int]] = {
    "person": [1001],
    "kite": [1002, 1003],
    "clock": [1004],
}

#: Fixture support-calibration constants.  They are deliberately *not* the real
#: sealed numbers: every assertion below proves the planner copied these exact
#: values out of the fixture receipt, which it could not do if it carried a
#: threshold of its own.
FIXTURE_THETA_PEAK_LIFT = 2.5
FIXTURE_THETA_LOCAL_CONCENTRATION = 1.25
FIXTURE_SUPPORT_EPSILON = 0.002
FIXTURE_CROSS_CONTEXT_DELTA_EPSILON = 0.004
FIXTURE_PRIMARY_QUANTILE = 0.1
FIXTURE_SENSITIVITY_QUANTILES: tuple[float, ...] = (0.05, 0.25)
FIXTURE_CATEGORY_CONTRIBUTION_MIN = 20
FIXTURE_UNDERREPRESENTED_FLAG = "pooled_underrepresented"
FIXTURE_CALIBRATION_OBSERVATIONS = 70

#: The partition-to-bound map the frozen census classifier produces.  The
#: fixture seals it so a drifted classifier fails the build instead of silently
#: reshaping an owner's U or L bank.
FIXTURE_PARTITION_BOUNDS: dict[str, list[str]] = {
    "strict_assigned_self": ["lower", "upper"],
    "ambiguous_upper": ["upper"],
    "unmatched_generator_local": ["upper"],
    "other_owner_strict": [],
    "not_generated_by_owner": [],
}

#: The timing-control owner is favorable and calibrated-supported at exactly
#: these two boundaries; the planner must select the later one.
TIMING_QUALIFYING_BOUNDARIES: tuple[int, int] = (3, 5)
#: The due-boundary true-positive replay control of each image.
TP_CONTROL_ORDINAL = 1


def _sha_json(value: Any) -> str:
    return merge.sha256_json(value)


def _coords(seed: int) -> list[int]:
    start = COORDINATE_TOKEN_IDS["start"]
    span = COORDINATE_TOKEN_IDS["bin_count"]
    return [start + (seed * (step + 3) * 7 + step * 11) % span for step in range(4)]


def _row_token_block(description: str, coord_token_ids: list[int]) -> list[int]:
    return (
        [WRAPPERS["object_ref_start"]]
        + DESCRIPTION_TOKENS[description]
        + [WRAPPERS["object_ref_end"], WRAPPERS["box_start"]]
        + coord_token_ids
        + [WRAPPERS["box_end"]]
    )


def _query_suffix(description: str) -> list[int]:
    return (
        [WRAPPERS["object_ref_start"]]
        + DESCRIPTION_TOKENS[description]
        + [WRAPPERS["object_ref_end"], WRAPPERS["box_start"]]
    )


# ---------------------------------------------------------------------------
# Fixture construction
# ---------------------------------------------------------------------------


class _ImagePlan:
    """The owner/row layout of one synthetic image."""

    def __init__(self, image_id: str, index: int) -> None:
        self.image_id = image_id
        self.index = index
        self.tp_count = TP_PER_IMAGE[image_id]
        self.unmatched_count = UNMATCHED_ROWS_PER_IMAGE[image_id]
        self.row_count = self.tp_count + self.unmatched_count
        self.rows: list[dict[str, Any]] = []
        self.owners: list[dict[str, Any]] = []
        self._ordinal = 0

    def next_owner_id(self) -> str:
        owner_id = f"gt:{self.image_id}:{self._ordinal}"
        self._ordinal += 1
        return owner_id


def _build_image_plan(image_id: str, index: int) -> _ImagePlan:
    plan = _ImagePlan(image_id, index)

    # Native true positives own the strict-matched rows 0 .. tp_count - 1.
    for ordinal in range(plan.tp_count):
        owner_id = plan.next_owner_id()
        description = "kite" if ordinal == 2 else "person"
        plan.owners.append(
            {
                "gt_owner_id": owner_id,
                "role": "native_true_positive",
                "normalized_description": description,
                "due_row_index": ordinal,
                "due_supported": ordinal == TP_CONTROL_ORDINAL,
            }
        )
        plan.rows.append(
            {
                "row_index": ordinal,
                "normalized_description": description,
                "strict_match_status": "matched",
                "strict_match_gt_owner_id": owner_id,
            }
        )

    # Unmatched native rows are the E rows of the unmatched-E primary owners.
    for offset in range(plan.unmatched_count):
        plan.rows.append(
            {
                "row_index": plan.tp_count + offset,
                "normalized_description": "kite" if offset == 0 else "person",
                "strict_match_status": "unmatched",
                "strict_match_gt_owner_id": None,
            }
        )

    # One matched-E primary crossing at row 0.  Image 0's primary carries a
    # singleton description so its same-category competitor family is empty and
    # its description differs from the E row it is skipped by.
    plan.owners.append(
        {
            "gt_owner_id": plan.next_owner_id(),
            "role": "primary_matched_e",
            "normalized_description": "clock" if index == 0 else "person",
            "crossing_boundary": 0,
            "bound": "u_and_l" if index != 0 else "u_only",
        }
    )
    # Unmatched-E primaries crossing at the unmatched rows.  The last unmatched
    # row of every image leaves P+E terminal, so F is absent there.
    for offset in range(plan.unmatched_count):
        plan.owners.append(
            {
                "gt_owner_id": plan.next_owner_id(),
                "role": "primary_unmatched_e",
                "normalized_description": "person",
                "crossing_boundary": plan.tp_count + offset,
                "bound": "u_only" if (index == 1 and offset == 0) else "u_and_l",
            }
        )
    if index == 0:
        plan.owners.append(
            {
                "gt_owner_id": plan.next_owner_id(),
                "role": "l_only_crossing",
                "normalized_description": "person",
                "crossing_boundary": 1,
                "bound": "l_only",
            }
        )
    for _ in range(TIMING_PER_IMAGE[image_id]):
        plan.owners.append(
            {
                "gt_owner_id": plan.next_owner_id(),
                "role": "timing_control",
                "normalized_description": "person",
            }
        )
    for _ in range(FILLER_PER_IMAGE[image_id]):
        plan.owners.append(
            {
                "gt_owner_id": plan.next_owner_id(),
                "role": "filler",
                "normalized_description": "person",
            }
        )
    return plan


def _competition(rank: int, owner_id: str, other_owner_id: str) -> dict[str, Any]:
    return {
        "rank": rank,
        "best_gt_owner_id": owner_id if rank == 1 else other_owner_id,
        "margin_to_best_owner": 0.0 if rank == 1 else -1.5,
        "population_size": 3,
        "population": "owners_of_this_image_context_category",
    }


def _owner_context_row(
    *,
    owner_id: str,
    image_id: str,
    boundary_index: int,
    context_role: str,
    passed_state: str,
    gate_open: bool,
    category_rank: int,
    u_rank: int,
    l_rank: int,
    other_owner_id: str,
) -> dict[str, Any]:
    context_id = builder.context_id_for(image_id, boundary_index)
    return {
        "schema_version": merge.OWNER_CONTEXT_SCHEMA_VERSION,
        "row_kind": "census_owner_context",
        "owner_context_id": f"{owner_id}@{context_id}",
        "gt_owner_id": owner_id,
        "image_id": image_id,
        "context_id": context_id,
        "boundary_index": boundary_index,
        "context_role": context_role,
        "loop_marking": {"loop_tail": False, "flag_is_not_a_mechanism_label": True},
        "frontier_features": {"passed_state": passed_state, "frontier_present": boundary_index > 0},
        "category_proposal_channel": {
            "boundary_gate": {"continue_vs_stop_logprob_margin": 1.25 if gate_open else -1.25},
            "category_routing_event": {"within_context_rank": category_rank},
        },
        "owner_competition_u": _competition(u_rank, owner_id, other_owner_id),
        "owner_competition_l": _competition(l_rank, owner_id, other_owner_id),
    }


def build_fixture_rows() -> dict[str, list[dict[str, Any]]]:
    """Every census row of the synthetic predecessor, ready to be mutated."""

    plans = [_build_image_plan(image_id, index) for index, image_id in enumerate(IMAGE_IDS)]

    contexts: list[dict[str, Any]] = []
    sidecars: list[dict[str, Any]] = []
    owner_registry: list[dict[str, Any]] = []
    owner_summaries: list[dict[str, Any]] = []
    owner_contexts: list[dict[str, Any]] = []
    categories: list[dict[str, Any]] = []
    query_groups: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    images: list[dict[str, Any]] = []

    for plan in plans:
        image_id = plan.image_id
        other_owner_id = f"gt:{image_id}:0"

        images.append(
            {
                "schema_version": planner.PLAN_SCHEMA_VERSION,
                "row_kind": "census_image",
                "image_id": image_id,
                "native_complete_row_count": plan.row_count,
                "wrapper_token_ids": dict(WRAPPERS),
                "coordinate_token_ids": dict(COORDINATE_TOKEN_IDS),
            }
        )

        prefix_tokens: list[int] = []
        prefix_rows: list[dict[str, Any]] = []
        for row in plan.rows:
            row_index = int(row["row_index"])
            coord_token_ids = _coords(int(image_id) * 100 + row_index)
            raw_span_sha256 = _sha_json(["raw-span", image_id, row_index])
            contexts.append(
                {
                    "schema_version": planner.PLAN_SCHEMA_VERSION,
                    "row_kind": "census_context",
                    "context_id": builder.context_id_for(image_id, row_index),
                    "image_id": image_id,
                    "boundary_index": row_index,
                    "context_role": "root" if row_index == 0 else "row_boundary",
                    "terminal_kind": None,
                    "frontier": None,
                    "generated_prefix_token_ids": list(prefix_tokens),
                    "generated_prefix_token_ids_sha256": _sha_json(list(prefix_tokens)),
                    "prefix_rows": copy.deepcopy(prefix_rows),
                    "prefix_row_indices": [int(item["row_index"]) for item in prefix_rows],
                    "prefix_admission": {
                        "forced_continue_rows_excluded": True,
                        "retokenized": False,
                        "source": "native_greedy_complete_rows_only",
                    },
                    "total_complete_row_count": plan.row_count,
                }
            )
            sidecars.append(
                {
                    "schema_version": planner.PLAN_SCHEMA_VERSION,
                    "row_kind": "native_sidecar",
                    "sidecar_id": f"native:{image_id}:{row_index}",
                    "pred_row_id": f"pred:sorted:greedy:0:{image_id}:{row_index}",
                    "image_id": image_id,
                    "row_index": row_index,
                    "normalized_description": row["normalized_description"],
                    "strict_match_status": row["strict_match_status"],
                    "strict_match_gt_owner_id": row["strict_match_gt_owner_id"],
                    "raw_span_sha256": raw_span_sha256,
                    "coord_token_ids": coord_token_ids,
                    "coord_token_ids_sha256": _sha_json(coord_token_ids),
                    "bbox_pixel_xyxy": [10.0, 20.0, 30.0, 40.0],
                }
            )
            prefix_tokens = prefix_tokens + _row_token_block(
                row["normalized_description"], coord_token_ids
            )
            prefix_rows = prefix_rows + [
                {
                    "row_index": row_index,
                    "pred_row_id": f"pred:sorted:greedy:0:{image_id}:{row_index}",
                    "description": row["normalized_description"],
                    "raw_span_sha256": raw_span_sha256,
                    "strict_match_status": row["strict_match_status"],
                    "strict_match_gt_owner_id": row["strict_match_gt_owner_id"],
                }
            ]
        contexts.append(
            {
                "schema_version": planner.PLAN_SCHEMA_VERSION,
                "row_kind": "census_context",
                "context_id": builder.context_id_for(image_id, plan.row_count),
                "image_id": image_id,
                "boundary_index": plan.row_count,
                "context_role": "terminal",
                "terminal_kind": "natural_stop",
                "frontier": None,
                "generated_prefix_token_ids": list(prefix_tokens),
                "generated_prefix_token_ids_sha256": _sha_json(list(prefix_tokens)),
                "prefix_rows": copy.deepcopy(prefix_rows),
                "prefix_row_indices": [int(item["row_index"]) for item in prefix_rows],
                "prefix_admission": {
                    "forced_continue_rows_excluded": True,
                    "retokenized": False,
                    "source": "native_greedy_complete_rows_only",
                },
                "total_complete_row_count": plan.row_count,
            }
        )

        boundary_indices = list(range(plan.row_count + 1))
        terminal_context_id = builder.context_id_for(image_id, plan.row_count)

        for ordinal, owner in enumerate(plan.owners):
            owner_id = str(owner["gt_owner_id"])
            description = str(owner["normalized_description"])
            role = str(owner["role"])
            # Three sealed target-local candidates per owner, one in each
            # support partition the census classifier distinguishes: the exact
            # anchor is strictly self-assigned (counts under both bounds), the
            # shifted probe is ambiguity-neutral (U only), and the collided
            # probe is strictly assigned to another owner (excluded from both
            # and moved to the collision diagnostic).
            owner_candidate_ids: list[str] = []
            for index, (suffix, transform_role, status, assigned) in enumerate(
                (
                    ("anchor", "exact_gt_anchor", "matched", owner_id),
                    ("shift", "translate_left", "ambiguous_neutral", None),
                    ("collide", "translate_right", "matched", f"gt:{image_id}:collision"),
                )
            ):
                coord_token_ids = _coords(int(image_id) * 1000 + ordinal * 5 + index + 1)
                candidate_id = f"cand:{image_id}:{ordinal}:{suffix}"
                owner_candidate_ids.append(candidate_id)
                candidates.append(
                    {
                        "schema_version": planner.PLAN_SCHEMA_VERSION,
                        "row_kind": "physical_candidate",
                        "candidate_id": candidate_id,
                        "image_id": image_id,
                        "normalized_description": description,
                        "coord_token_ids": coord_token_ids,
                        "coord_token_ids_sha256": _sha_json(coord_token_ids),
                        "decoded_bbox_pixel_xyxy": [1, 2, 3, 4],
                        "generator_gt_owner_ids": [owner_id],
                        "strict_assignment_scope": "same_normalized_description_only",
                        "strict_assignment_status": status,
                        "strict_assignment_gt_owner_id": assigned,
                        "ambiguity_owner_ids": (
                            [owner_id] if status == "ambiguous_neutral" else []
                        ),
                        "representative_role": transform_role,
                        "generators": [
                            {
                                "generator_gt_owner_id": owner_id,
                                "logical_transform_role": transform_role,
                            }
                        ],
                    }
                )
            owner_registry.append(
                {
                    "schema_version": planner.PLAN_SCHEMA_VERSION,
                    "row_kind": "census_owner",
                    "gt_owner_id": owner_id,
                    "image_id": image_id,
                    "normalized_description": description,
                    "official_coco_category_id": 1,
                    "bbox_pixel_xyxy": [1.0, 2.0, 3.0, 4.0],
                    "owner_sort_key": [float(ordinal), float(ordinal)],
                    "candidate_bank": {
                        "logical_role_count": planner.LOGICAL_ROLE_COUNT,
                        "distinct_physical_candidate_count": len(owner_candidate_ids),
                        "bank_coverage_status": "full",
                        "physical_candidate_ids": list(owner_candidate_ids),
                    },
                }
            )

            crossing_boundary = owner.get("crossing_boundary")
            bound = owner.get("bound")
            if role == "native_true_positive":
                usable_u = (
                    [builder.context_id_for(image_id, int(owner["due_row_index"]))]
                    if owner["due_supported"]
                    else []
                )
                usable_l = list(usable_u)
                favorable_boundaries: set[int] = set()
            elif role in {"primary_matched_e", "primary_unmatched_e", "l_only_crossing"}:
                crossing_context_id = builder.context_id_for(image_id, int(crossing_boundary))
                favorable_boundaries = {int(crossing_boundary)}
                usable_u = [] if bound == "l_only" else [crossing_context_id]
                usable_l = [] if bound == "u_only" else [crossing_context_id]
            elif role == "timing_control":
                favorable_boundaries = set(TIMING_QUALIFYING_BOUNDARIES)
                usable_u = [
                    builder.context_id_for(image_id, index)
                    for index in TIMING_QUALIFYING_BOUNDARIES
                ]
                usable_l = list(usable_u)
            else:
                favorable_boundaries = {plan.row_count}
                usable_u = [terminal_context_id]
                usable_l = [terminal_context_id]

            owner_summaries.append(
                {
                    "schema_version": merge.OWNER_SUMMARY_SCHEMA_VERSION,
                    "gt_owner_id": owner_id,
                    "image_id": image_id,
                    "normalized_description": description,
                    "native_true_positive": role == "native_true_positive",
                    "disposition": (
                        "native_true_positive_calibration_control"
                        if role == "native_true_positive"
                        else merge.DISPOSITION_RESOLVED
                    ),
                    "upper_bound_u": {"bound": "u", "usable_support_context_ids": usable_u},
                    "lower_bound_l": {"bound": "l", "usable_support_context_ids": usable_l},
                }
            )

            for boundary_index in boundary_indices:
                if boundary_index == 0:
                    context_role = "root"
                elif boundary_index == plan.row_count:
                    context_role = "terminal"
                else:
                    context_role = "row_boundary"
                if crossing_boundary is not None and boundary_index > int(crossing_boundary):
                    passed_state = "passed_by_frontier"
                elif boundary_index == 0:
                    passed_state = "root_no_frontier"
                else:
                    passed_state = "ahead_of_frontier"
                favorable = boundary_index in favorable_boundaries
                owner_contexts.append(
                    _owner_context_row(
                        owner_id=owner_id,
                        image_id=image_id,
                        boundary_index=boundary_index,
                        context_role=context_role,
                        passed_state=passed_state,
                        gate_open=favorable,
                        category_rank=1 if favorable else 9,
                        u_rank=1 if (favorable and bound != "l_only") else 2,
                        l_rank=1 if (favorable and bound != "u_only") else 2,
                        other_owner_id=other_owner_id,
                    )
                )

        descriptions = sorted(
            {str(owner["normalized_description"]) for owner in plan.owners}
            | {str(row["normalized_description"]) for row in plan.rows}
        )
        for description in descriptions:
            owner_ids = sorted(
                str(owner["gt_owner_id"])
                for owner in plan.owners
                if str(owner["normalized_description"]) == description
            )
            suffix_tokens = _query_suffix(description)
            categories.append(
                {
                    "schema_version": planner.PLAN_SCHEMA_VERSION,
                    "row_kind": "census_category",
                    "category_query_id": f"{image_id}:{description}",
                    "image_id": image_id,
                    "normalized_description": description,
                    "status": "admitted",
                    "owner_ids": owner_ids,
                    "owner_count_in_image": len(owner_ids),
                    "category_token_ids": list(DESCRIPTION_TOKENS[description]),
                    "category_token_ids_sha256": _sha_json(list(DESCRIPTION_TOKENS[description])),
                    "query_suffix_token_ids": suffix_tokens,
                    "query_suffix_token_ids_sha256": _sha_json(suffix_tokens),
                }
            )
            population_ids = sorted(
                str(candidate["candidate_id"])
                for candidate in candidates
                if str(candidate["image_id"]) == image_id
                and str(candidate["normalized_description"]) == description
            )
            for boundary_index in boundary_indices:
                context_id = builder.context_id_for(image_id, boundary_index)
                query_groups.append(
                    {
                        "schema_version": planner.PLAN_SCHEMA_VERSION,
                        "row_kind": "census_query_group",
                        "query_group_id": f"{context_id}|{description}",
                        "context_id": context_id,
                        "image_id": image_id,
                        "normalized_description": description,
                        "status": "admitted",
                        "candidate_ids": list(population_ids),
                        "candidate_count": len(population_ids),
                        "unique_coordinate_tuple_count": len(population_ids),
                        "query_suffix_token_ids": suffix_tokens,
                        "query_suffix_token_ids_sha256": _sha_json(suffix_tokens),
                        "admission_receipt_id": f"admission:{context_id}|query_suffix",
                    }
                )

    return {
        "contexts": contexts,
        "sidecars": sidecars,
        "owner_registry": owner_registry,
        "owner_summaries": owner_summaries,
        "owner_contexts": owner_contexts,
        "categories": categories,
        "query_groups": query_groups,
        "candidates": candidates,
        "images": images,
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode("utf-8")
    path.write_bytes(payload)
    return merge.sha256_bytes(payload)


def _write_json(path: Path, value: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True).encode("utf-8")
    path.write_bytes(payload)
    return merge.sha256_bytes(payload)


def _seal(receipt: MutableMapping[str, Any]) -> dict[str, Any]:
    sealed = dict(receipt)
    sealed["receipt_content_sha256"] = merge.sha256_json(dict(receipt))
    return sealed


def build_support_calibration() -> dict[str, Any]:
    """The sealed calibration receipt, produced by the census merge's own writer.

    Using ``SupportCalibration.describe`` rather than a hand-written literal
    means the fixture receipt carries a real self-consistent
    ``calibration_sha256`` and the live ``merge_source_sha256``, so the
    planner's provenance checks are exercised instead of bypassed.
    """

    return merge.SupportCalibration(
        theta_peak_lift=FIXTURE_THETA_PEAK_LIFT,
        theta_local_concentration=FIXTURE_THETA_LOCAL_CONCENTRATION,
        epsilon=FIXTURE_SUPPORT_EPSILON,
        quantile=FIXTURE_PRIMARY_QUANTILE,
        observation_count=FIXTURE_CALIBRATION_OBSERVATIONS,
        per_category_counts={"person": 30, "kite": 25, "clock": 15},
        sensitivity={
            "role": "report_only_never_moves_a_threshold",
            "quantiles": {
                "0.05": {"theta_peak_lift": 2.0, "theta_local_concentration": 1.0},
                "0.25": {"theta_peak_lift": 2.9, "theta_local_concentration": 1.6},
            },
        },
        exclusions=(),
        consumed_shard_digests=("fixture-shard-a", "fixture-shard-b"),
        capture_manifest_sha256="fixture-capture-manifest",
        category_contribution_min=FIXTURE_CATEGORY_CONTRIBUTION_MIN,
        underrepresented_flag=FIXTURE_UNDERREPRESENTED_FLAG,
        cross_context_delta_epsilon=FIXTURE_CROSS_CONTEXT_DELTA_EPSILON,
        statistics=merge.SUPPORT_FEATURE_NAMES,
    ).describe()


def build_capture_rules() -> dict[str, Any]:
    """A capture-rules file whose ``owner_support`` block is the frozen one."""

    return {
        "capture_rules_sha256": "fixture",
        "admission": {"channels": ["query_suffix"]},
        "owner_support": {
            "primary_neighbourhood": "generator_local_landscape",
            "other_owner_strict_candidate": (
                "excluded_from_target_support_moved_to_collision_diagnostic"
            ),
            "partition_bounds": copy.deepcopy(FIXTURE_PARTITION_BOUNDS),
            "epsilons": {
                "adaptive": False,
                "support_epsilon": FIXTURE_SUPPORT_EPSILON,
                "cross_context_delta_epsilon": FIXTURE_CROSS_CONTEXT_DELTA_EPSILON,
            },
            "support_calibration": {
                "primary_quantile": FIXTURE_PRIMARY_QUANTILE,
                "sensitivity_quantiles": list(FIXTURE_SENSITIVITY_QUANTILES),
                "category_contribution_min": FIXTURE_CATEGORY_CONTRIBUTION_MIN,
                "underrepresented_flag": FIXTURE_UNDERREPRESENTED_FLAG,
                "population": "discovery_native_true_positive_owners",
                "stratification": "pooled",
                "threshold_search_or_tuning": "forbidden",
            },
            "support_definition": {
                "statistics": list(merge.SUPPORT_FEATURE_NAMES),
                "rank_is_support_criterion": False,
                "rank_one_as_support": "forbidden",
                "evaluated_under_both_ambiguity_bounds": True,
                "usable_support_combinator": "conjunction_both_statistics",
                "usable_support_rule": (
                    "peak_lift >= threshold + epsilon AND "
                    "local_concentration >= threshold + epsilon"
                ),
                "computed_on": "generator_local_max_excluding_other_owner_strict",
                "peak_lift": {
                    "formula": (
                        "best_logprob - logsumexp(all unique candidate logprobs in the same "
                        "query group) + log(unique_population_size)"
                    ),
                    "population": "all_unique_candidates_in_the_query_group",
                    "not": "best_minus_an_owner_local_reference",
                    "shift_invariant": True,
                },
                "local_concentration": {
                    "formula": (
                        "best exclusion-filtered generator-local score - median(that owner's "
                        "own exclusion-filtered bank scores)"
                    ),
                    "reference_population": "owner_own_exclusion_filtered_bank_scores",
                    "reference_statistic": "median_mean_of_two_central_values_when_even",
                    "not": "normalized_mass_share",
                },
            },
        },
    }


def build_support_semantics(calibration: Mapping[str, Any]) -> dict[str, Any]:
    """The presentation-phase declaration that must agree with the receipt."""

    return {
        "constants_source": "sealed_capture_rules_owner_support",
        "criterion_id": calibration["criterion_id"],
        "inputs": list(calibration["statistics"]),
        "epsilon": calibration["epsilon"],
        "cross_context_delta_epsilon": calibration["cross_context_delta_epsilon"],
        "primary_quantile": calibration["quantile"],
        "rank_is_not_a_support_input": True,
        "rank_and_margin_role": (
            "published as routing/competition features, never a support input"
        ),
        "observed_drift_role": "compliance_signal_only_never_widens_epsilon",
    }


def materialize_fixture(
    tmp_path: Path,
    *,
    mutate_rows: Callable[[dict[str, list[dict[str, Any]]]], None] | None = None,
    mutate_receipts: Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], None] | None = None,
    mutate_support: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[Path, Path]:
    """Write a complete synthetic census + prevalence pair; return both roots.

    ``mutate_support`` receives ``{"capture_rules", "calibration", "applied",
    "semantics"}`` before anything is written, where ``applied`` is the
    ``support_criterion`` block the presentation merge receipt will publish.
    Tampering with any one of them must fail the build closed.
    """

    rows = build_fixture_rows()
    if mutate_rows is not None:
        mutate_rows(rows)

    calibration = build_support_calibration()
    support = {
        "capture_rules": build_capture_rules(),
        "calibration": calibration,
        "applied": copy.deepcopy(calibration),
        "semantics": build_support_semantics(calibration),
    }
    if mutate_support is not None:
        mutate_support(support)

    census_root = tmp_path / "census"
    digests: dict[str, str] = {}
    digests["plan/context-registry.jsonl"] = _write_jsonl(
        census_root / "plan/context-registry.jsonl", rows["contexts"]
    )
    digests["plan/native-sidecar-registry.jsonl"] = _write_jsonl(
        census_root / "plan/native-sidecar-registry.jsonl", rows["sidecars"]
    )
    digests["plan/owner-registry.jsonl"] = _write_jsonl(
        census_root / "plan/owner-registry.jsonl", rows["owner_registry"]
    )
    digests["plan/category-registry.jsonl"] = _write_jsonl(
        census_root / "plan/category-registry.jsonl", rows["categories"]
    )
    digests["plan/query-group-registry.jsonl"] = _write_jsonl(
        census_root / "plan/query-group-registry.jsonl", rows["query_groups"]
    )
    digests["plan/candidate-bank.jsonl"] = _write_jsonl(
        census_root / "plan/candidate-bank.jsonl", rows["candidates"]
    )
    digests["plan/image-registry.jsonl"] = _write_jsonl(
        census_root / "plan/image-registry.jsonl", rows["images"]
    )
    digests["plan/capture-rules.json"] = _write_json(
        census_root / "plan/capture-rules.json", support["capture_rules"]
    )
    _write_json(
        census_root / builder.SUPPORT_CALIBRATION_REL, support["calibration"]
    )
    digests["phases/presentation/owner-summaries.jsonl"] = _write_jsonl(
        census_root / "phases/presentation/owner-summaries.jsonl", rows["owner_summaries"]
    )
    digests["phases/presentation/owner-context-features.jsonl"] = _write_jsonl(
        census_root / "phases/presentation/owner-context-features.jsonl", rows["owner_contexts"]
    )

    plan_receipt = _seal(
        {
            "schema_version": planner.PLAN_SCHEMA_VERSION,
            "unit_id": builder.CENSUS_UNIT_ID,
            "output_file_digests": {
                name.split("/", 1)[1]: digest
                for name, digest in digests.items()
                if name.startswith("plan/")
            },
            "score_input_policy": {
                "candidate_selection_uses_scores": False,
                "reads_any_score_artifact": False,
            },
            "query_suffix_contract": {
                "shape": ["object_ref_start", "category_token_ids", "object_ref_end", "box_start"],
                "wrapper_token_ids": dict(WRAPPERS),
                "on_mismatch": "fail_closed",
            },
        }
    )
    merge_receipt = _seal(
        {
            "schema_version": merge.MERGE_SCHEMA_VERSION,
            "unit_id": builder.CENSUS_UNIT_ID,
            "phase": merge.PHASE_PRESENTATION,
            "usable_as_census_conclusion": True,
            "support_criterion": support["applied"],
            "support_semantics": support["semantics"],
            "plan": {"receipt_content_sha256": plan_receipt["receipt_content_sha256"]},
            "counts": {
                "owner_summary_row_count": len(rows["owner_summaries"]),
                "owner_context_row_count": len(rows["owner_contexts"]),
            },
            "output_file_digests": {
                "owner-summaries.jsonl": digests["phases/presentation/owner-summaries.jsonl"],
                "owner-context-features.jsonl": digests[
                    "phases/presentation/owner-context-features.jsonl"
                ],
            },
        }
    )

    prevalence_root = tmp_path / "prevalence"
    report_json_sha = _write_json(prevalence_root / "analysis/report.json", {"unit_id": "fixture"})
    (prevalence_root / "analysis/report.md").write_bytes(b"# fixture\n")
    report_md_sha = merge.sha256_bytes((prevalence_root / "analysis/report.md").read_bytes())
    owner_records_sha = _write_jsonl(
        prevalence_root / "analysis/owner-records.jsonl", [{"gt_owner_id": "fixture"}]
    )
    prevalence_receipt = {
        "schema_version": prevalence.RECEIPT_SCHEMA_VERSION,
        "unit_id": builder.PREVALENCE_UNIT_ID,
        "predecessor_run_root": str(census_root),
        "predecessor_unit_id": builder.CENSUS_UNIT_ID,
        "report_json_sha256": report_json_sha,
        "report_md_sha256": report_md_sha,
        "owner_records_jsonl_sha256": owner_records_sha,
        "input_file_sha256": {
            name: digests[name]
            for name in (
                "plan/context-registry.jsonl",
                "plan/native-sidecar-registry.jsonl",
                "phases/presentation/owner-summaries.jsonl",
                "phases/presentation/owner-context-features.jsonl",
            )
        },
        "validation": {"usable_as_census_conclusion": True},
    }

    if mutate_receipts is not None:
        mutate_receipts(plan_receipt, merge_receipt, prevalence_receipt)
        plan_receipt = _seal(
            {key: value for key, value in plan_receipt.items() if key != "receipt_content_sha256"}
        )
        merge_receipt["plan"] = {"receipt_content_sha256": plan_receipt["receipt_content_sha256"]}
        merge_receipt = _seal(
            {key: value for key, value in merge_receipt.items() if key != "receipt_content_sha256"}
        )

    plan_receipt_sha = _write_json(census_root / "plan/receipt.json", plan_receipt)
    merge_receipt_sha = _write_json(
        census_root / "phases/presentation/merge-receipt.json", merge_receipt
    )
    prevalence_receipt["input_file_sha256"]["plan/receipt.json"] = plan_receipt_sha
    prevalence_receipt["input_file_sha256"][
        "phases/presentation/merge-receipt.json"
    ] = merge_receipt_sha
    _write_json(prevalence_root / "analysis/receipt.json", _seal(prevalence_receipt))
    return prevalence_root, census_root


def _build(tmp_path: Path, **kwargs: Any) -> dict[str, Any]:
    prevalence_root, census_root = materialize_fixture(tmp_path, **kwargs)
    return builder.build_plan(
        prevalence_run_root=prevalence_root,
        census_run_root=census_root,
        output_root=tmp_path / "out",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


@pytest.fixture(scope="module")
def sealed_plan(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """One clean build, reused by every non-mutating assertion."""

    tmp_path = tmp_path_factory.mktemp("sealed")
    manifest = _build(tmp_path)
    plan_dir = tmp_path / "out" / builder.PLAN_DIR_NAME
    return {
        "manifest": manifest,
        "plan_dir": plan_dir,
        "cohort": _read_jsonl(plan_dir / builder.COHORT_REGISTRY_NAME),
        "controls": _read_jsonl(plan_dir / builder.CONTROL_REGISTRY_NAME),
        "requests": _read_jsonl(plan_dir / builder.REQUEST_PLAN_NAME),
    }


# ---------------------------------------------------------------------------
# Cohort semantics
# ---------------------------------------------------------------------------


def test_frozen_cohort_denominators_are_reproduced(sealed_plan: dict[str, Any]) -> None:
    counts = sealed_plan["manifest"]["cohort_counts"]
    assert counts["u_bound_crossing_count"] == builder.EXPECTED_U_CROSSING_COUNT == 26
    assert counts["l_bound_crossing_count"] == builder.EXPECTED_L_CROSSING_COUNT == 25
    assert (
        counts["exact_same_context_u_and_l_count"]
        == builder.EXPECTED_EXACT_UL_CROSSING_COUNT
        == 24
    )
    assert counts["matched_e_count"] == builder.EXPECTED_MATCHED_E_COUNT == 12
    assert counts["unmatched_e_count"] == builder.EXPECTED_UNMATCHED_E_COUNT == 14
    assert len(sealed_plan["cohort"]) == 26
    # Every resolved false-negative owner that crosses: the 26 U-bound primaries plus
    # the single L-only crossing owner.  It is never the primary denominator.
    assert counts["all_resolved_owner_crossing_count"] == 27
    assert sum(counts["per_image_owner_counts"].values()) == 26


def test_e_strata_are_preserved_not_selected_away(sealed_plan: dict[str, Any]) -> None:
    strata = [row["e_row"]["stratum"] for row in sealed_plan["cohort"]]
    assert strata.count(builder.STRATUM_MATCHED_E) == 12
    assert strata.count(builder.STRATUM_UNMATCHED_E) == 14
    for row in sealed_plan["cohort"]:
        if row["e_row"]["stratum"] == builder.STRATUM_MATCHED_E:
            assert row["e_row"]["strict_match_status"] == "matched"
            assert row["e_row"]["strict_match_gt_owner_id"] != row["gt_owner_id"]
        else:
            assert row["e_row"]["strict_match_status"] != "matched"


def test_boundary_convention_binds_e_to_row_b_and_pe_to_boundary_b_plus_one(
    sealed_plan: dict[str, Any],
) -> None:
    for row in sealed_plan["cohort"]:
        boundary = row["crossing"]["boundary_index_b"]
        image_id = row["image_id"]
        assert row["crossing"]["p_context_id"] == builder.context_id_for(image_id, boundary)
        assert row["crossing"]["pe_context_id"] == builder.context_id_for(image_id, boundary + 1)
        assert row["e_row"]["row_index"] == boundary
        assert row["crossing"]["p_passed_state"] in builder.ROOT_OR_AHEAD_STATES


def test_full_row_tokens_come_only_from_the_adjacent_context_suffix(
    tmp_path: Path, sealed_plan: dict[str, Any]
) -> None:
    contexts = {
        row["context_id"]: row
        for row in _read_jsonl(
            Path(sealed_plan["manifest"]["lineage"]["census_run_root"])
            / "plan/context-registry.jsonl"
        )
    }
    for row in sealed_plan["cohort"]:
        p_tokens = contexts[row["crossing"]["p_context_id"]]["generated_prefix_token_ids"]
        pe_tokens = contexts[row["crossing"]["pe_context_id"]]["generated_prefix_token_ids"]
        assert pe_tokens[: len(p_tokens)] == p_tokens
        assert row["e_row"]["full_row_token_ids"] == pe_tokens[len(p_tokens) :]
        assert row["e_row"]["full_row_token_ids_sha256"] == merge.sha256_json(
            row["e_row"]["full_row_token_ids"]
        )
        offset = row["e_row"]["coord_token_offset_in_row"]
        tokens = row["e_row"]["full_row_token_ids"]
        assert tokens[offset : offset + 4] == row["e_row"]["coord_token_ids"]
        assert tokens[offset - 1] == WRAPPERS["box_start"]
        assert tokens[-1] == WRAPPERS["box_end"]
        assert "full_row_token_ids" not in row["e_row"]["sidecar_role"]
        assert row["e_row"]["full_row_token_source"].startswith("literal_adjacent_context")


def test_f_row_is_bound_only_when_a_further_native_row_exists(
    sealed_plan: dict[str, Any],
) -> None:
    counts = sealed_plan["manifest"]["cohort_counts"]
    # Every image's last unmatched row leaves P+E terminal, so F cannot exist.
    assert counts["terminal_p_plus_e_count"] == 12
    assert counts["f_row_present_count"] == 14
    for row in sealed_plan["cohort"]:
        if row["crossing"]["pe_context_role"] == "terminal":
            assert row["f_row"] is None
            assert row["f_row_present"] is False
            assert row["native_next_action_at_pe"]["kind"] == builder.NATIVE_ACTION_STOP
            assert row["native_next_action_at_pe"]["token_ids"] == [WRAPPERS["im_end"]]
        else:
            assert row["f_row"]["row_index"] == row["crossing"]["boundary_index_b"] + 1
            assert row["f_row"]["post_e_boundary_context_id"] == builder.context_id_for(
                row["image_id"], row["crossing"]["boundary_index_b"] + 2
            )
            assert row["native_next_action_at_pe"]["kind"] == builder.NATIVE_ACTION_ROW


def test_same_description_cases_are_tagged_as_construction_determined(
    sealed_plan: dict[str, Any],
) -> None:
    same = [
        row
        for row in sealed_plan["cohort"]
        if row["description_observability"] == builder.OBSERVABILITY_SAME_DESCRIPTION
    ]
    different = [
        row
        for row in sealed_plan["cohort"]
        if row["description_observability"] == builder.OBSERVABILITY_DIFFERENT_DESCRIPTION
    ]
    assert len(same) + len(different) == 26
    assert same and different
    for row in same:
        assert row["same_description_as_e"] is True
        assert row["normalized_description"] == row["e_row"]["normalized_description"]
        assert "construction-determined" in row["same_description_note"]
    for row in different:
        assert row["same_description_as_e"] is False
        assert row["same_description_note"] is None


def test_bounds_record_u_as_primary_and_l_as_sensitivity(sealed_plan: dict[str, Any]) -> None:
    u_only = 0
    for row in sealed_plan["cohort"]:
        bounds = row["bounds"]
        assert bounds["u_favorable_top3_supported_at_p"] is True
        assert bounds["primary_bound"] == "u"
        assert bounds["u_channel"]["bound"] == "u"
        assert bounds["l_channel"]["bound"] == "l"
        assert bounds["exact_same_context_u_and_l"] == bounds["l_favorable_top3_supported_at_p"]
        if not bounds["l_favorable_top3_supported_at_p"]:
            u_only += 1
    assert u_only == 2


def test_target_description_path_and_inserted_row_use_only_sealed_tokens(
    sealed_plan: dict[str, Any],
) -> None:
    for row in sealed_plan["cohort"]:
        path = row["target_description_path"]
        assert path["query_suffix_token_ids"][0] == WRAPPERS["object_ref_start"]
        assert path["query_suffix_token_ids"][-1] == WRAPPERS["box_start"]
        assert path["query_suffix_token_ids"][-2] == WRAPPERS["object_ref_end"]
        assert path["query_suffix_token_ids_sha256"] == merge.sha256_json(
            path["query_suffix_token_ids"]
        )
        assert len(path["predecessor_query_group_ids"]) == 2
        inserted = row["inserted_clean_row_c"]
        assert inserted["token_ids"] == (
            path["query_suffix_token_ids"] + inserted["coord_token_ids"] + [WRAPPERS["box_end"]]
        )
        assert len(inserted["coord_token_ids"]) == 4
        for token in inserted["coord_token_ids"]:
            assert COORDINATE_TOKEN_IDS["start"] <= token <= COORDINATE_TOKEN_IDS["end_inclusive"]
        families = row["candidate_families"]
        assert (
            inserted["exact_gt_anchor_candidate_id"]
            in families[builder.FAMILY_TARGET_LOCAL]["candidate_ids"]
        )
        assert families[builder.FAMILY_SAME_CATEGORY_OTHER_OWNER]["candidate_ids_sha256"] == (
            merge.sha256_json(families[builder.FAMILY_SAME_CATEGORY_OTHER_OWNER]["candidate_ids"])
        )


def test_singleton_category_owner_has_an_empty_competitor_family(
    sealed_plan: dict[str, Any],
) -> None:
    singleton = [
        row for row in sealed_plan["cohort"] if row["normalized_description"] == "clock"
    ]
    assert len(singleton) == 1
    families = singleton[0]["candidate_families"]
    assert families[builder.FAMILY_SAME_CATEGORY_OTHER_OWNER]["candidate_count"] == 0
    assert families[builder.FAMILY_TARGET_LOCAL]["candidate_count"] == 3
    # A singleton category still has a unique population: its own family.
    assert families["unique_population"]["unique_population_size"] == 3


def test_owner_and_image_isolation_holds_across_every_registry_row(
    sealed_plan: dict[str, Any],
) -> None:
    for row in list(sealed_plan["cohort"]) + list(sealed_plan["controls"]):
        image_id = row["image_id"]
        assert row["gt_owner_id"].startswith(f"gt:{image_id}:")
        for value in json.dumps(row).split('"'):
            if value.startswith(f"{image_id}:boundary-") or ":boundary-" not in value:
                continue
            raise AssertionError(f"row for image {image_id} references foreign context {value}")


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------


def test_controls_are_frozen_disjoint_and_deterministic(sealed_plan: dict[str, Any]) -> None:
    manifest = sealed_plan["manifest"]
    timing = [
        row
        for row in sealed_plan["controls"]
        if row["cohort"] == builder.TIMING_CONTROL_COHORT
    ]
    replay = [
        row
        for row in sealed_plan["controls"]
        if row["cohort"] == builder.TP_REPLAY_CONTROL_COHORT
    ]
    assert len(timing) == builder.EXPECTED_TIMING_CONTROL_COUNT == 14
    assert len(replay) == builder.EXPECTED_TP_REPLAY_CONTROL_COUNT == 12
    assert {row["image_id"] for row in replay} == set(IMAGE_IDS)
    primary_ids = {row["gt_owner_id"] for row in sealed_plan["cohort"]}
    control_ids = {row["gt_owner_id"] for row in sealed_plan["controls"]}
    assert not primary_ids & control_ids
    assert len(control_ids) == len(sealed_plan["controls"])
    assert manifest["control_counts"]["disjoint_from_primary_cohort"] is True


def test_timing_control_selects_the_latest_qualifying_boundary(
    sealed_plan: dict[str, Any],
) -> None:
    timing = [
        row
        for row in sealed_plan["controls"]
        if row["cohort"] == builder.TIMING_CONTROL_COHORT
    ]
    for row in timing:
        assert row["qualifying_boundary_indices"] == list(TIMING_QUALIFYING_BOUNDARIES)
        assert row["selected_boundary_index"] == max(TIMING_QUALIFYING_BOUNDARIES)
        assert row["context_id"] == builder.context_id_for(
            row["image_id"], row["selected_boundary_index"]
        )
        assert row["next_context_id"] == builder.context_id_for(
            row["image_id"], row["selected_boundary_index"] + 1
        )
        assert row["next_row"]["strict_match_status"] == "matched"
        assert row["role"].startswith("descriptive_only")


def test_timing_controls_exclude_the_owner_s_own_crossing_boundary(tmp_path: Path) -> None:
    """The noncrossing rule, proven without relying on primary exclusion."""

    prevalence_root, census_root = materialize_fixture(tmp_path)
    inputs = builder.load_plan_inputs(prevalence_root, census_root)
    builder.validate_plan_inputs(inputs)
    cases = builder.derive_crossing_cases(inputs)
    crossing_owner_ids = {case.gt_owner_id for case in cases if case.u_favorable}
    # Deliberately exclude nobody: only the noncrossing rule can now keep the
    # matched-E primaries (favorable, supported, matched next row) out.
    controls = builder.build_timing_controls(
        inputs, cases, frozenset(), builder.bind_support_calibration(inputs)
    )
    control_ids = {row["gt_owner_id"] for row in controls}
    assert not control_ids & crossing_owner_ids
    for row in controls:
        if row["own_crossing_boundary_index"] is not None:
            assert row["own_crossing_boundary_index"] not in row["qualifying_boundary_indices"]


def test_tp_replay_control_prefers_due_supported_non_singleton(
    sealed_plan: dict[str, Any],
) -> None:
    replay = [
        row
        for row in sealed_plan["controls"]
        if row["cohort"] == builder.TP_REPLAY_CONTROL_COHORT
    ]
    for row in replay:
        assert row["due_supported"] is True
        assert row["non_singleton_category"] is True
        assert row["preference_tier"] == 0
        assert row["row_index"] == TP_CONTROL_ORDINAL
        assert row["due_context_id"] == builder.context_id_for(row["image_id"], row["row_index"])
        assert row["replaced_native_row"]["strict_match_gt_owner_id"] == row["gt_owner_id"]
        assert row["following_native_action"]["kind"] == builder.NATIVE_ACTION_ROW


def test_tp_replay_control_breaks_ties_by_native_row_index(tmp_path: Path) -> None:
    def promote_a_later_true_positive(rows: dict[str, list[dict[str, Any]]]) -> None:
        for summary in rows["owner_summaries"]:
            if summary["gt_owner_id"] == "gt:700:3":
                summary["upper_bound_u"]["usable_support_context_ids"] = [
                    builder.context_id_for("700", 3)
                ]

    manifest = _build(tmp_path, mutate_rows=promote_a_later_true_positive)
    controls = _read_jsonl(tmp_path / "out" / builder.PLAN_DIR_NAME / builder.CONTROL_REGISTRY_NAME)
    chosen = [
        row
        for row in controls
        if row["cohort"] == builder.TP_REPLAY_CONTROL_COHORT and row["image_id"] == "700"
    ]
    assert len(chosen) == 1
    assert chosen[0]["image_candidate_count"] == TP_PER_IMAGE["700"]
    # Both owners are due-supported non-singletons; the lower native row index wins.
    assert chosen[0]["row_index"] == TP_CONTROL_ORDINAL
    assert chosen[0]["gt_owner_id"] == "gt:700:1"
    assert manifest["control_counts"]["tp_replay_control_count"] == 12


# ---------------------------------------------------------------------------
# Request plan
# ---------------------------------------------------------------------------


def test_request_plan_covers_both_ladders_at_both_contexts(sealed_plan: dict[str, Any]) -> None:
    counts = sealed_plan["manifest"]["request_counts"]
    assert counts["total"] == len(sealed_plan["requests"])
    # 26 primaries * (2 contexts * 5 requests) + 26 P+C->E + 14 P+E+C->F.
    assert counts["by_cohort"][builder.PRIMARY_COHORT] == 26 * 10 + 26 + 14
    assert counts["by_cohort"][builder.TIMING_CONTROL_COHORT] == 14 * 10
    assert counts["by_cohort"][builder.TP_REPLAY_CONTROL_COHORT] == 12 * 4
    assert counts["optional"] == 14

    primary = [
        request
        for request in sealed_plan["requests"]
        if request["cohort"] == builder.PRIMARY_COHORT
    ]
    for owner_row in sealed_plan["cohort"]:
        owner_requests = [
            request for request in primary if request["gt_owner_id"] == owner_row["gt_owner_id"]
        ]
        variants = {request["variant"] for request in owner_requests}
        assert {"at_p", "at_p_plus_e", "p_plus_c_then_e"} <= variants
        assert ("p_plus_e_plus_c_then_f" in variants) is owner_row["f_row_present"]
        for family in (
            builder.REQUEST_NATURAL_RELEASE,
            builder.REQUEST_NATIVE_NEXT_ACTION,
            builder.REQUEST_COORDINATE_TARGET_LOCAL,
            builder.REQUEST_COORDINATE_COMPETITOR,
            builder.REQUEST_COORDINATE_GREEDY,
        ):
            at_contexts = {
                request["context_id"]
                for request in owner_requests
                if request["request_family"] == family
            }
            assert at_contexts == {
                owner_row["crossing"]["p_context_id"],
                owner_row["crossing"]["pe_context_id"],
            }
        assert sorted(request["request_id"] for request in owner_requests) == owner_row[
            "request_ids"
        ]


def test_request_identities_are_unique_stable_and_score_blind(
    sealed_plan: dict[str, Any],
) -> None:
    ids = [request["request_id"] for request in sealed_plan["requests"]]
    keys = [request["request_key"] for request in sealed_plan["requests"]]
    assert len(set(ids)) == len(ids)
    assert len(set(keys)) == len(keys)
    for request in sealed_plan["requests"]:
        assert request["request_id"].startswith("req:")
        assert request["prefix"]["retokenized"] is False
        assert request["prefix"]["appended_token_ids_sha256"] == merge.sha256_json(
            request["prefix"]["appended_token_ids"]
        )
        assert request["inspects_new_model_logits"] is False
        assert request["score_blind_plan"] is True
        assert request["branch_schema_id"] == builder.BRANCH_SCHEMA_ID
        assert request["request_family"] in builder.REQUEST_FAMILIES


def test_coordinate_requests_force_d_c_and_declare_the_frozen_grammar(
    sealed_plan: dict[str, Any],
) -> None:
    greedy = [
        request
        for request in sealed_plan["requests"]
        if request["request_family"] == builder.REQUEST_COORDINATE_GREEDY
    ]
    assert greedy
    for request in greedy:
        grammar = request["decode"]["grammar"]
        assert grammar["coordinate_token_count"] == 4
        assert grammar["terminal_token_id"] == WRAPPERS["box_end"]
        assert grammar["max_new_tokens"] == 5
        assert grammar["coordinate_token_id_start"] == COORDINATE_TOKEN_IDS["start"]
        assert grammar["coordinate_token_id_end_inclusive"] == COORDINATE_TOKEN_IDS["end_inclusive"]
        assert request["decode"]["on_grammar_violation"] == "malformed_never_silently_repaired"
        assert request["prefix"]["appended_role"] == "forced_target_description_path_d_c"
        assert request["prefix"]["appended_token_ids"][-1] == WRAPPERS["box_start"]
    for request in sealed_plan["requests"]:
        if request["request_family"] not in {
            builder.REQUEST_COORDINATE_TARGET_LOCAL,
            builder.REQUEST_COORDINATE_COMPETITOR,
        }:
            continue
        assert request["prefix"]["appended_role"] == "forced_target_description_path_d_c"
        assert request["candidate_family"]["candidate_ids_sha256"] == merge.sha256_json(
            request["candidate_family"]["candidate_ids"]
        )
        description = request["prefix"]["appended_token_ids"]
        assert request["predecessor_query_group_id"].startswith(f"{request['context_id']}|")
        assert description[0] == WRAPPERS["object_ref_start"]


def test_compatibility_requests_are_secondary_and_carry_the_exact_next_row(
    sealed_plan: dict[str, Any],
) -> None:
    compat = {
        (request["gt_owner_id"], request["variant"]): request
        for request in sealed_plan["requests"]
        if request["request_family"] == builder.REQUEST_DOWNSTREAM_COMPATIBILITY
    }
    assert compat
    for request in compat.values():
        assert request["readout_tier"].startswith("secondary")
        assert request["branch_inputs"] == []
        assert request["prefix"]["appended_role"] == "inserted_exact_clean_gt_row_c"
    for owner_row in sealed_plan["cohort"]:
        owner_id = owner_row["gt_owner_id"]
        p_request = compat[(owner_id, "p_plus_c_then_e")]
        assert p_request["context_id"] == owner_row["crossing"]["p_context_id"]
        assert p_request["scored_target"]["token_ids"] == owner_row["e_row"]["full_row_token_ids"]
        assert p_request["optional"] is False
        if owner_row["f_row_present"]:
            f_request = compat[(owner_id, "p_plus_e_plus_c_then_f")]
            assert f_request["context_id"] == owner_row["crossing"]["pe_context_id"]
            assert (
                f_request["scored_target"]["token_ids"] == owner_row["f_row"]["full_row_token_ids"]
            )
            assert f_request["optional"] is True


def test_native_next_action_requests_carry_the_exact_native_alternative(
    sealed_plan: dict[str, Any],
) -> None:
    by_key = {
        (request["gt_owner_id"], request["variant"]): request
        for request in sealed_plan["requests"]
        if request["request_family"] == builder.REQUEST_NATIVE_NEXT_ACTION
    }
    for owner_row in sealed_plan["cohort"]:
        at_p = by_key[(owner_row["gt_owner_id"], "at_p")]
        assert at_p["scored_target"]["token_ids"] == owner_row["e_row"]["full_row_token_ids"]
        assert at_p["scored_target"]["kind"] == builder.NATIVE_ACTION_ROW
        at_pe = by_key[(owner_row["gt_owner_id"], "at_p_plus_e")]
        assert at_pe["scored_target"]["kind"] == owner_row["native_next_action_at_pe"]["kind"]


def test_plan_declares_the_branch_schema_without_evaluating_it(
    sealed_plan: dict[str, Any],
) -> None:
    schema = sealed_plan["manifest"]["branch_schema"]
    assert schema["exhaustive_order"] == [
        "displaced",
        "release_lost",
        "realization_fail",
        "ambiguous",
    ]
    assert schema["sub_tags"]["displaced"] == ["likelihood_displaced", "greedy_displaced"]
    assert schema["evaluated_here"] is False
    for row in sealed_plan["cohort"]:
        assert row["branch_order"] == schema["exhaustive_order"]
        assert "primary_branch" not in row
    policy = sealed_plan["manifest"]["score_input_policy"]
    assert policy["inspects_new_model_logits"] is False
    assert policy["loads_model_or_tokenizer"] is False
    assert policy["cohort_selection_uses_new_scores"] is False


# ---------------------------------------------------------------------------
# Target-local support calibration
# ---------------------------------------------------------------------------


def test_support_calibration_carries_the_exact_sealed_values_and_lineage(
    sealed_plan: dict[str, Any],
) -> None:
    """Every published threshold is the sealed receipt's, byte-traceable."""

    support = sealed_plan["manifest"]["support_calibration"]
    census_root = Path(sealed_plan["manifest"]["lineage"]["census_run_root"])
    source_path = census_root / builder.SUPPORT_CALIBRATION_REL
    payload = source_path.read_bytes()
    receipt = json.loads(payload)

    seal = support["source"]
    assert seal["path"] == builder.SUPPORT_CALIBRATION_REL
    assert seal["byte_size"] == len(payload)
    assert seal["sha256"] == merge.sha256_bytes(payload)
    assert seal["calibration_sha256"] == receipt["calibration_sha256"]
    assert seal["calibration_schema_version"] == merge.CALIBRATION_SCHEMA_VERSION
    assert seal["calibration_unit_id"] == builder.CENSUS_UNIT_ID
    assert seal["merge_source_sha256"] == merge.MERGE_SOURCE_SHA256
    assert seal["applied_by_phase"] == merge.PHASE_PRESENTATION

    thresholds = support["thresholds"]
    assert thresholds["theta_peak_lift"] == FIXTURE_THETA_PEAK_LIFT
    assert thresholds["theta_local_concentration"] == FIXTURE_THETA_LOCAL_CONCENTRATION
    assert thresholds["epsilon"] == FIXTURE_SUPPORT_EPSILON
    assert thresholds["quantile"] == FIXTURE_PRIMARY_QUANTILE
    assert thresholds["cross_context_delta_epsilon"] == FIXTURE_CROSS_CONTEXT_DELTA_EPSILON
    assert thresholds["epsilon_is_adaptive"] is False
    assert thresholds["quantile_is_primary_and_fixed"] is True
    assert thresholds["calibration_stratum"] == receipt["calibration_stratum"]
    # Nothing was re-derived, re-fit, or taken from this unit's own scores.
    assert support["derived_here"] is False
    assert support["uses_crossing_unit_scores"] is False
    assert support["criterion_id"] == merge.SUPPORT_CRITERION_ID
    assert support["rule"]["statistics"] == list(merge.SUPPORT_FEATURE_NAMES)
    assert support["rule"]["text"] == receipt["rule"]
    assert support["rule"]["evaluated_under_both_ambiguity_bounds"] is True


def test_support_calibration_binds_both_ambiguity_bounds(
    sealed_plan: dict[str, Any],
) -> None:
    """U owns the primary branch; L is a reproducible sensitivity, not a stand-in."""

    bounds = sealed_plan["manifest"]["support_calibration"]["bounds"]
    assert bounds["paths"] == {"u": "ambiguity_included_u", "l": "ambiguity_excluded_l"}
    assert bounds["primary_bound"] == "u"
    assert bounds["sensitivity_bound"] == "l"
    assert "never_changes_a_u_branch" in bounds["sensitivity_bound_role"]
    assert bounds["partition_bounds"] == FIXTURE_PARTITION_BOUNDS
    assert bounds["partition_bounds_source"] == "frozen_census_classify_candidate_for_owner"


def test_target_local_family_publishes_reproducible_u_and_l_banks(
    sealed_plan: dict[str, Any],
) -> None:
    """Both bounds are membership lists derived from sealed strict assignment."""

    checked = 0
    for row in list(sealed_plan["cohort"]) + list(sealed_plan["controls"]):
        family = row["candidate_families"][builder.FAMILY_TARGET_LOCAL]
        members = {member["candidate_id"]: member for member in family["members"]}
        assert set(members) == set(family["candidate_ids"])
        assert family["logical_role_count"] == planner.LOGICAL_ROLE_COUNT

        expected_u = [
            candidate_id
            for candidate_id in family["candidate_ids"]
            if members[candidate_id]["counts_toward_upper_bound_u"]
        ]
        expected_l = [
            candidate_id
            for candidate_id in family["candidate_ids"]
            if members[candidate_id]["counts_toward_lower_bound_l"]
        ]
        assert family["bounds"]["u"]["candidate_ids"] == expected_u
        assert family["bounds"]["l"]["candidate_ids"] == expected_l
        assert family["bounds"]["u"]["owner_context_path"] == "ambiguity_included_u"
        assert family["bounds"]["l"]["owner_context_path"] == "ambiguity_excluded_l"
        assert expected_l, "the L sensitivity must be evaluable"
        # L is a strict subset relation, never a separate selection.
        assert set(expected_l) <= set(expected_u)

        # The fixture partitions are exactly one per bound state.
        partitions = {member["partition"] for member in family["members"]}
        assert partitions == {
            "strict_assigned_self",
            "ambiguous_upper",
            "other_owner_strict",
        }
        excluded = family["excluded_other_owner_strict_candidate_ids"]
        assert excluded == [
            candidate_id
            for candidate_id in family["candidate_ids"]
            if members[candidate_id]["partition"] == "other_owner_strict"
        ]
        for candidate_id in excluded:
            assert candidate_id not in expected_u
            assert candidate_id not in expected_l
        checked += 1
    assert checked == 26 + 26


def test_support_membership_is_never_derived_from_rank_or_margin(
    sealed_plan: dict[str, Any],
) -> None:
    """No rank, margin, or score field may reach a support decision."""

    support = sealed_plan["manifest"]["support_calibration"]
    assert support["rank_and_margin"]["rank_is_a_support_input"] is False
    assert support["rank_and_margin"]["margin_is_a_support_input"] is False
    assert support["rank_and_margin"]["rank_one_as_support"] == "forbidden"

    forbidden = ("rank", "margin", "logprob", "score", "posterior")
    for row in list(sealed_plan["cohort"]) + list(sealed_plan["controls"]):
        family = row["candidate_families"][builder.FAMILY_TARGET_LOCAL]
        assert (
            family["support_membership_source"]
            == "sealed_strict_assignment_never_a_rank_or_margin"
        )
        for member in family["members"]:
            assert member["partition_source"] == "frozen_census_classify_candidate_for_owner"
            # Membership is a function of the sealed assignment alone.
            status = member["strict_assignment_status"]
            assigned = member["strict_assignment_gt_owner_id"]
            owner_id = row["gt_owner_id"]
            if status == "matched" and assigned != owner_id:
                assert member["counts_toward_upper_bound_u"] is False
                assert member["counts_toward_lower_bound_l"] is False
            elif status == "matched":
                assert member["counts_toward_upper_bound_u"] is True
                assert member["counts_toward_lower_bound_l"] is True
            else:
                assert member["counts_toward_upper_bound_u"] is True
                assert member["counts_toward_lower_bound_l"] is False
            for key in member:
                assert not any(token in key for token in forbidden), key
        for bound_block in family["bounds"].values():
            for key in bound_block:
                assert not any(token in key for token in forbidden), key


def test_planner_declares_no_threshold_of_its_own() -> None:
    """A threshold literal in this module would be an invented calibration."""

    source = Path(builder.__file__).read_text(encoding="utf-8")
    for token in ("theta_peak_lift =", "theta_local_concentration =", "epsilon ="):
        assert token not in source
    assert str(FIXTURE_THETA_PEAK_LIFT) not in source
    assert str(FIXTURE_SUPPORT_EPSILON) not in source


def test_unique_population_is_verified_against_the_sealed_query_groups(
    sealed_plan: dict[str, Any],
) -> None:
    """``peak_lift``'s denominator is proven at every context that is scored."""

    for row in list(sealed_plan["cohort"]) + list(sealed_plan["controls"]):
        families = row["candidate_families"]
        population = families["unique_population"]
        target_local = families[builder.FAMILY_TARGET_LOCAL]["candidate_ids"]
        competitor = families[builder.FAMILY_SAME_CATEGORY_OTHER_OWNER]["candidate_ids"]
        assert set(population["candidate_ids"]) == set(target_local) | set(competitor)
        assert population["unique_population_size"] == len(population["candidate_ids"])
        assert population["scope"] == "image_context_normalized_description"
        assert population["source"] == "sealed_census_query_group_registry"
        assert population["verified_query_group_ids"]
        assert families[builder.FAMILY_SAME_CATEGORY_OTHER_OWNER]["support_role"] == (
            "enters_the_unique_peak_lift_population_only_never_the_target_local_"
            "concentration_bank"
        )


def test_coordinate_requests_carry_the_support_calibration_binding(
    sealed_plan: dict[str, Any],
) -> None:
    """A scorer reading one request alone can still reproduce support."""

    support = sealed_plan["manifest"]["support_calibration"]
    target_local_requests = 0
    for request in sealed_plan["requests"]:
        if request["request_family"] not in {
            builder.REQUEST_COORDINATE_TARGET_LOCAL,
            builder.REQUEST_COORDINATE_COMPETITOR,
        }:
            assert "support_evaluation" not in request["scored_target"]
            continue
        evaluation = request["scored_target"]["support_evaluation"]
        reference = evaluation["support_calibration"]
        assert reference["calibration_sha256"] == support["source"]["calibration_sha256"]
        assert reference["theta_peak_lift"] == support["thresholds"]["theta_peak_lift"]
        assert (
            reference["theta_local_concentration"]
            == support["thresholds"]["theta_local_concentration"]
        )
        assert reference["epsilon"] == support["thresholds"]["epsilon"]
        assert reference["primary_bound"] == "u"
        assert reference["rank_is_a_support_input"] is False
        population = evaluation["unique_population"]
        assert population["query_group_id"] == request["predecessor_query_group_id"]
        assert population["query_group_id"].startswith(f"{request['context_id']}|")
        assert population["scope"] == "image_context_normalized_description"
        assert population["unique_population_size"] > 0
        if request["request_family"] == builder.REQUEST_COORDINATE_TARGET_LOCAL:
            target_local_requests += 1
            bounds = evaluation["target_local_bounds"]
            assert set(bounds) == {"u", "l"}
            assert bounds["u"]["owner_context_path"] == "ambiguity_included_u"
            assert bounds["l"]["owner_context_path"] == "ambiguity_excluded_l"
            assert (
                evaluation["family_role"]
                == "target_local_support_bank_and_peak_lift_numerator"
            )
        else:
            assert evaluation["target_local_bounds"] is None
    assert target_local_requests == 92


# ---------------------------------------------------------------------------
# Sealing and determinism
# ---------------------------------------------------------------------------


def test_manifest_seals_every_input_and_output_and_self_seals(
    sealed_plan: dict[str, Any],
) -> None:
    manifest = sealed_plan["manifest"]
    plan_dir: Path = sealed_plan["plan_dir"]
    reconstructed = merge.sha256_json(
        {key: value for key, value in manifest.items() if key != "manifest_content_sha256"}
    )
    assert reconstructed == manifest["manifest_content_sha256"]
    for name, seal in manifest["output_file_digests"].items():
        payload = (plan_dir / name).read_bytes()
        assert seal["sha256"] == merge.sha256_bytes(payload)
        assert seal["byte_size"] == len(payload)
        assert seal["row_count"] == len(payload.decode("utf-8").splitlines())
    lineage = manifest["lineage"]
    for relative_name in prevalence.AUTHORITATIVE_INPUT_FILES:
        assert relative_name in lineage["census_input_files"]
    for relative_name in builder.EXTRA_CENSUS_PLAN_FILES:
        assert relative_name in lineage["census_input_files"]
    for relative_name in builder.PREVALENCE_INPUT_FILES:
        assert relative_name in lineage["prevalence_input_files"]
    assert lineage["prevalence_unit_id"] == builder.PREVALENCE_UNIT_ID
    assert lineage["census_unit_id"] == builder.CENSUS_UNIT_ID
    assert manifest["unit_id"] == builder.UNIT_ID
    assert manifest["schema_version"] == builder.MANIFEST_SCHEMA_VERSION
    # The scalar digest fields of the v1 manifest no longer exist.
    assert "builder_source_sha256" not in manifest
    assert "census_input_file_sha256" not in lineage
    assert "prevalence_input_file_sha256" not in lineage


def test_every_input_and_source_seal_carries_path_byte_size_and_sha256(
    sealed_plan: dict[str, Any],
) -> None:
    """The seal fields are complete *and* describe the live bytes on disk."""

    manifest = sealed_plan["manifest"]
    lineage = manifest["lineage"]
    assert lineage["seal_fields"] == ["path", "byte_size", "sha256"]
    assert builder.SEAL_FIELDS == ("path", "byte_size", "sha256")

    sealed_groups = (
        (Path(lineage["prevalence_run_root"]), lineage["prevalence_input_files"]),
        (Path(lineage["census_run_root"]), lineage["census_input_files"]),
    )
    sealed_file_count = 0
    for root, seals in sealed_groups:
        assert seals
        for relative_name, seal in seals.items():
            assert set(seal) == set(builder.SEAL_FIELDS)
            assert seal["path"] == relative_name
            payload = (root / seal["path"]).read_bytes()
            assert seal["byte_size"] == len(payload)
            assert seal["sha256"] == merge.sha256_bytes(payload)
            sealed_file_count += 1
    assert sealed_file_count == len(builder.PREVALENCE_INPUT_FILES) + len(
        prevalence.AUTHORITATIVE_INPUT_FILES
    ) + len(builder.EXTRA_CENSUS_PLAN_FILES) + len(builder.EXTRA_CENSUS_PHASE_FILES)
    assert builder.SUPPORT_CALIBRATION_REL in lineage["census_input_files"]

    source_seal = manifest["builder_source"]
    assert set(source_seal) == set(builder.SEAL_FIELDS)
    source_path = Path(builder.__file__).resolve()
    assert source_seal["path"] == str(source_path.relative_to(builder.REPO_ROOT))
    source_bytes = source_path.read_bytes()
    assert source_seal["byte_size"] == len(source_bytes)
    assert source_seal["sha256"] == merge.sha256_bytes(source_bytes)

    for name, seal in manifest["output_file_digests"].items():
        assert set(builder.SEAL_FIELDS) <= set(seal)
        assert seal["path"] == name


def test_an_input_that_changes_between_validation_and_sealing_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prevalence_root, census_root = materialize_fixture(tmp_path)
    real_validate = builder.validate_plan_inputs

    def validate_then_tamper(inputs: Any) -> Any:
        validation = real_validate(inputs)
        target = census_root / "plan/capture-rules.json"
        target.write_bytes(target.read_bytes() + b" ")
        return validation

    monkeypatch.setattr(builder, "validate_plan_inputs", validate_then_tamper)
    with pytest.raises(
        builder.PlanContractError, match="changed on disk between validation and sealing"
    ):
        builder.build_plan(
            prevalence_run_root=prevalence_root,
            census_run_root=census_root,
            output_root=tmp_path / "out",
        )


def test_two_builds_from_the_same_inputs_are_byte_identical(tmp_path: Path) -> None:
    prevalence_root, census_root = materialize_fixture(tmp_path)
    first = builder.build_plan(
        prevalence_run_root=prevalence_root,
        census_run_root=census_root,
        output_root=tmp_path / "first",
    )
    second = builder.build_plan(
        prevalence_run_root=prevalence_root,
        census_run_root=census_root,
        output_root=tmp_path / "second",
    )
    assert first["manifest_content_sha256"] == second["manifest_content_sha256"]
    assert first["lineage"] == second["lineage"]
    assert first["builder_source"] == second["builder_source"]
    assert first["support_calibration"] == second["support_calibration"]
    for name in (
        builder.MANIFEST_NAME,
        builder.COHORT_REGISTRY_NAME,
        builder.CONTROL_REGISTRY_NAME,
        builder.REQUEST_PLAN_NAME,
    ):
        left = (tmp_path / "first" / builder.PLAN_DIR_NAME / name).read_bytes()
        right = (tmp_path / "second" / builder.PLAN_DIR_NAME / name).read_bytes()
        assert left == right


def test_existing_output_is_not_silently_overwritten(tmp_path: Path) -> None:
    prevalence_root, census_root = materialize_fixture(tmp_path)
    builder.build_plan(
        prevalence_run_root=prevalence_root,
        census_run_root=census_root,
        output_root=tmp_path / "out",
    )
    with pytest.raises(builder.PlanContractError, match="already exists and is not empty"):
        builder.build_plan(
            prevalence_run_root=prevalence_root,
            census_run_root=census_root,
            output_root=tmp_path / "out",
        )
    builder.build_plan(
        prevalence_run_root=prevalence_root,
        census_run_root=census_root,
        output_root=tmp_path / "out",
        overwrite=True,
    )


# ---------------------------------------------------------------------------
# Fail-closed guards
# ---------------------------------------------------------------------------


def test_edited_calibration_threshold_fails_closed(tmp_path: Path) -> None:
    """A raised threshold that keeps the old digest is a tampered receipt."""

    def raise_the_threshold(support: dict[str, Any]) -> None:
        support["calibration"]["theta_peak_lift"] = 9.0
        support["applied"]["theta_peak_lift"] = 9.0

    with pytest.raises(
        builder.PlanContractError, match="does not reconstruct its own digest"
    ):
        _build(tmp_path, mutate_support=raise_the_threshold)


def test_resealed_calibration_that_the_phase_never_applied_fails_closed(
    tmp_path: Path,
) -> None:
    """Re-digesting an edited receipt does not make it the applied one."""

    def reseal_a_lower_threshold(support: dict[str, Any]) -> None:
        edited = dict(support["calibration"])
        edited["theta_local_concentration"] = 0.0
        edited.pop("calibration_sha256")
        edited["calibration_sha256"] = merge.sha256_json(edited)
        support["calibration"] = edited

    with pytest.raises(
        builder.PlanContractError, match="not the receipt the presentation phase applied"
    ):
        _build(tmp_path, mutate_support=reseal_a_lower_threshold)


def test_calibration_from_a_foreign_analyzer_source_fails_closed(tmp_path: Path) -> None:
    def forge_the_merge_source(support: dict[str, Any]) -> None:
        edited = dict(support["calibration"])
        edited["merge_source_sha256"] = "f" * 64
        edited.pop("calibration_sha256")
        edited["calibration_sha256"] = merge.sha256_json(edited)
        support["calibration"] = edited
        support["applied"] = copy.deepcopy(edited)

    with pytest.raises(builder.PlanContractError, match="derived by a different analyzer source"):
        _build(tmp_path, mutate_support=forge_the_merge_source)


def test_calibration_schema_drift_fails_closed(tmp_path: Path) -> None:
    def bump_the_schema(support: dict[str, Any]) -> None:
        edited = dict(support["calibration"])
        edited["schema_version"] = "sorted-owner-accessibility-census-support-calibration.v2"
        edited.pop("calibration_sha256")
        edited["calibration_sha256"] = merge.sha256_json(edited)
        support["calibration"] = edited
        support["applied"] = copy.deepcopy(edited)

    with pytest.raises(builder.PlanContractError, match="unexpected schema_version"):
        _build(tmp_path, mutate_support=bump_the_schema)


def test_missing_calibration_artifact_fails_closed(tmp_path: Path) -> None:
    prevalence_root, census_root = materialize_fixture(tmp_path)
    (census_root / builder.SUPPORT_CALIBRATION_REL).unlink()
    with pytest.raises(builder.PlanContractError, match="census plan file is missing"):
        builder.build_plan(
            prevalence_run_root=prevalence_root,
            census_run_root=census_root,
            output_root=tmp_path / "out",
        )


def test_presentation_receipt_without_a_support_criterion_fails_closed(
    tmp_path: Path,
) -> None:
    def drop_the_applied_block(
        plan_receipt: dict[str, Any],
        merge_receipt: dict[str, Any],
        prevalence_receipt: dict[str, Any],
    ) -> None:
        del plan_receipt, prevalence_receipt
        merge_receipt.pop("support_criterion")

    with pytest.raises(builder.PlanContractError, match="carries no support_criterion block"):
        _build(tmp_path, mutate_receipts=drop_the_applied_block)


def test_capture_rules_epsilon_drift_fails_closed(tmp_path: Path) -> None:
    """A widened epsilon in the frozen rules can never move a sealed threshold."""

    def widen_the_epsilon(support: dict[str, Any]) -> None:
        support["capture_rules"]["owner_support"]["epsilons"]["support_epsilon"] = 0.05

    with pytest.raises(
        builder.PlanContractError, match="support_epsilon .* disagrees with the sealed"
    ):
        _build(tmp_path, mutate_support=widen_the_epsilon)


def test_capture_rules_quantile_drift_fails_closed(tmp_path: Path) -> None:
    def move_the_quantile(support: dict[str, Any]) -> None:
        support["capture_rules"]["owner_support"]["support_calibration"][
            "primary_quantile"
        ] = 0.25

    with pytest.raises(
        builder.PlanContractError, match="primary_quantile .* disagrees with the sealed"
    ):
        _build(tmp_path, mutate_support=move_the_quantile)


def test_capture_rules_that_allow_a_rank_criterion_fail_closed(tmp_path: Path) -> None:
    def allow_rank_support(support: dict[str, Any]) -> None:
        support["capture_rules"]["owner_support"]["support_definition"][
            "rank_is_support_criterion"
        ] = True

    with pytest.raises(builder.PlanContractError, match="does not forbid a rank criterion"):
        _build(tmp_path, mutate_support=allow_rank_support)


def test_partition_bound_drift_fails_closed(tmp_path: Path) -> None:
    """Promoting an ambiguity-neutral candidate into L would reshape the L bank."""

    def promote_ambiguous_to_l(support: dict[str, Any]) -> None:
        support["capture_rules"]["owner_support"]["partition_bounds"]["ambiguous_upper"] = [
            "lower",
            "upper",
        ]

    with pytest.raises(
        builder.PlanContractError, match="partition_bounds map disagrees with the frozen"
    ):
        _build(tmp_path, mutate_support=promote_ambiguous_to_l)


def test_support_semantics_that_disagree_with_the_receipt_fail_closed(
    tmp_path: Path,
) -> None:
    def drift_the_declared_quantile(support: dict[str, Any]) -> None:
        support["semantics"]["primary_quantile"] = 0.25

    with pytest.raises(
        builder.PlanContractError, match="support_semantics.primary_quantile disagrees"
    ):
        _build(tmp_path, mutate_support=drift_the_declared_quantile)


def test_support_definition_on_the_wrong_neighbourhood_fails_closed(
    tmp_path: Path,
) -> None:
    def move_the_neighbourhood(support: dict[str, Any]) -> None:
        support["capture_rules"]["owner_support"]["support_definition"][
            "computed_on"
        ] = "generator_local_max"

    with pytest.raises(builder.PlanContractError, match="not the exclusion-filtered"):
        _build(tmp_path, mutate_support=move_the_neighbourhood)


def test_query_group_population_that_is_not_the_candidate_families_fails_closed(
    tmp_path: Path,
) -> None:
    def drop_a_candidate_from_one_group(rows: dict[str, list[dict[str, Any]]]) -> None:
        for row in rows["query_groups"]:
            if row["image_id"] == "702":
                row["candidate_ids"] = row["candidate_ids"][:-1]
                row["unique_coordinate_tuple_count"] = len(row["candidate_ids"])

    with pytest.raises(builder.PlanContractError, match="is not the target-local plus"):
        _build(tmp_path, mutate_rows=drop_a_candidate_from_one_group)


def test_non_admitted_query_group_cannot_own_a_support_population(
    tmp_path: Path,
) -> None:
    def withdraw_admission(rows: dict[str, list[dict[str, Any]]]) -> None:
        for row in rows["query_groups"]:
            if row["image_id"] == "702":
                row["status"] = "not_admitted"

    with pytest.raises(builder.PlanContractError, match="is not admitted"):
        _build(tmp_path, mutate_rows=withdraw_admission)


def test_target_local_family_that_disagrees_with_the_sealed_bank_fails_closed(
    tmp_path: Path,
) -> None:
    def shrink_the_sealed_bank(rows: dict[str, list[dict[str, Any]]]) -> None:
        for row in rows["owner_registry"]:
            bank = row["candidate_bank"]
            bank["physical_candidate_ids"] = bank["physical_candidate_ids"][:1]

    with pytest.raises(
        builder.PlanContractError, match="disagrees with the sealed\n?\\s*candidate_bank"
    ):
        _build(tmp_path, mutate_rows=shrink_the_sealed_bank)


def test_owner_without_an_evaluable_lower_bound_bank_fails_closed(
    tmp_path: Path,
) -> None:
    """No strict self-assignment means no L bank, so the sensitivity is unprovable."""

    def strip_every_strict_self_assignment(rows: dict[str, list[dict[str, Any]]]) -> None:
        for row in rows["candidates"]:
            if row["strict_assignment_status"] == "matched" and str(
                row["strict_assignment_gt_owner_id"]
            ).endswith(row["candidate_id"].split(":")[2]):
                row["strict_assignment_status"] = "ambiguous_neutral"
                row["strict_assignment_gt_owner_id"] = None

    with pytest.raises(
        builder.PlanContractError, match="empty L-bound exclusion-filtered target-local bank"
    ):
        _build(tmp_path, mutate_rows=strip_every_strict_self_assignment)


def test_missing_sidecar_row_fails_closed(tmp_path: Path) -> None:
    def drop_a_crossing_sidecar(rows: dict[str, list[dict[str, Any]]]) -> None:
        rows["sidecars"] = [
            row
            for row in rows["sidecars"]
            if not (row["image_id"] == "702" and row["row_index"] == 0)
        ]

    with pytest.raises(builder.PlanContractError, match="native sidecar row 0 of image '702' is missing"):
        _build(tmp_path, mutate_rows=drop_a_crossing_sidecar)


def test_sidecar_coordinate_digest_mismatch_fails_closed(tmp_path: Path) -> None:
    def corrupt_digest(rows: dict[str, list[dict[str, Any]]]) -> None:
        for row in rows["sidecars"]:
            if row["image_id"] == "702" and row["row_index"] == 0:
                row["coord_token_ids_sha256"] = "0" * 64

    with pytest.raises(builder.PlanContractError, match="coordinate digest does not reconstruct"):
        _build(tmp_path, mutate_rows=corrupt_digest)


def test_sidecar_coordinate_tokens_absent_from_the_row_suffix_fail_closed(
    tmp_path: Path,
) -> None:
    def rewrite_sidecar_coordinates(rows: dict[str, list[dict[str, Any]]]) -> None:
        for row in rows["sidecars"]:
            if row["image_id"] == "702" and row["row_index"] == 0:
                tokens = [COORDINATE_TOKEN_IDS["start"] + offset for offset in (1, 2, 3, 4)]
                row["coord_token_ids"] = tokens
                row["coord_token_ids_sha256"] = merge.sha256_json(tokens)

    with pytest.raises(builder.PlanContractError, match="occur 0 times in the literal"):
        _build(tmp_path, mutate_rows=rewrite_sidecar_coordinates)


def test_coordinate_tokens_outside_the_frozen_range_fail_closed(tmp_path: Path) -> None:
    def move_tokens_out_of_range(rows: dict[str, list[dict[str, Any]]]) -> None:
        for row in rows["candidates"]:
            if row["candidate_id"].endswith(":anchor") and row["image_id"] == "702":
                row["coord_token_ids"] = [10, 11, 12, 13]
                row["coord_token_ids_sha256"] = merge.sha256_json([10, 11, 12, 13])

    with pytest.raises(builder.PlanContractError, match="outside the frozen coordinate-token range"):
        _build(tmp_path, mutate_rows=move_tokens_out_of_range)


def test_non_prefix_adjacent_contexts_fail_closed(tmp_path: Path) -> None:
    # The terminal boundary of image 702 is the P+E of its unmatched-E primary,
    # so its predecessor prefix is non-empty and the prefix relation is real.
    broken_boundary = TP_PER_IMAGE["702"] + UNMATCHED_ROWS_PER_IMAGE["702"]

    def break_the_prefix_relation(rows: dict[str, list[dict[str, Any]]]) -> None:
        for row in rows["contexts"]:
            if row["context_id"] == builder.context_id_for("702", broken_boundary):
                tokens = [999] + list(row["generated_prefix_token_ids"])
                row["generated_prefix_token_ids"] = tokens
                row["generated_prefix_token_ids_sha256"] = merge.sha256_json(tokens)

    with pytest.raises(builder.PlanContractError, match="is not a literal token prefix of"):
        _build(tmp_path, mutate_rows=break_the_prefix_relation)


def test_context_prefix_digest_mismatch_fails_closed(tmp_path: Path) -> None:
    def corrupt_prefix_digest(rows: dict[str, list[dict[str, Any]]]) -> None:
        for row in rows["contexts"]:
            if row["context_id"] == builder.context_id_for("702", 0):
                row["generated_prefix_token_ids_sha256"] = "1" * 64

    with pytest.raises(
        builder.PlanContractError, match="do not reconstruct their own declared digest"
    ):
        _build(tmp_path, mutate_rows=corrupt_prefix_digest)


def test_sidecar_raw_span_disagreement_with_the_context_row_fails_closed(
    tmp_path: Path,
) -> None:
    def desynchronize_raw_span(rows: dict[str, list[dict[str, Any]]]) -> None:
        for row in rows["sidecars"]:
            if row["image_id"] == "702" and row["row_index"] == 0:
                row["raw_span_sha256"] = "2" * 64

    with pytest.raises(
        builder.PlanContractError, match="disagrees with the context-registry prefix row"
    ):
        _build(tmp_path, mutate_rows=desynchronize_raw_span)


def test_owner_context_row_from_another_image_fails_closed(tmp_path: Path) -> None:
    def relabel_one_owner_context(rows: dict[str, list[dict[str, Any]]]) -> None:
        for row in rows["owner_contexts"]:
            if row["owner_context_id"] == f"gt:702:{TP_PER_IMAGE['702']}@702:boundary-000":
                row["image_id"] = "703"

    with pytest.raises(builder.PlanContractError, match="belongs to image '703' but owner"):
        _build(tmp_path, mutate_rows=relabel_one_owner_context)


def test_query_group_from_another_image_fails_closed(tmp_path: Path) -> None:
    def relabel_one_query_group(rows: dict[str, list[dict[str, Any]]]) -> None:
        for row in rows["query_groups"]:
            if row["query_group_id"] == f"{builder.context_id_for('702', 0)}|person":
                row["image_id"] = "703"

    with pytest.raises(builder.PlanContractError, match="belongs to another image"):
        _build(tmp_path, mutate_rows=relabel_one_query_group)


def test_malformed_query_suffix_shape_fails_closed(tmp_path: Path) -> None:
    def drop_the_box_start(rows: dict[str, list[dict[str, Any]]]) -> None:
        for row in rows["categories"]:
            if row["category_query_id"] == "702:person":
                tokens = list(row["query_suffix_token_ids"])[:-1]
                row["query_suffix_token_ids"] = tokens
                row["query_suffix_token_ids_sha256"] = merge.sha256_json(tokens)

    with pytest.raises(builder.PlanContractError, match="not the frozen"):
        _build(tmp_path, mutate_rows=drop_the_box_start)


def test_missing_exact_gt_anchor_candidate_fails_closed(tmp_path: Path) -> None:
    def drop_one_anchor(rows: dict[str, list[dict[str, Any]]]) -> None:
        target = f"gt:702:{TP_PER_IMAGE['702']}"
        rows["candidates"] = [
            row
            for row in rows["candidates"]
            if not (
                row["generator_gt_owner_ids"] == [target]
                and row["generators"][0]["logical_transform_role"] == "exact_gt_anchor"
            )
        ]

    with pytest.raises(builder.PlanContractError, match="no exact_gt_anchor candidate"):
        _build(tmp_path, mutate_rows=drop_one_anchor)


def test_duplicate_exact_gt_anchor_candidate_fails_closed(tmp_path: Path) -> None:
    def duplicate_one_anchor(rows: dict[str, list[dict[str, Any]]]) -> None:
        target = f"gt:702:{TP_PER_IMAGE['702']}"
        for row in list(rows["candidates"]):
            if (
                row["generator_gt_owner_ids"] == [target]
                and row["generators"][0]["logical_transform_role"] == "exact_gt_anchor"
            ):
                clone = copy.deepcopy(row)
                clone["candidate_id"] = row["candidate_id"] + ":clone"
                rows["candidates"].append(clone)
                break

    with pytest.raises(builder.PlanContractError, match="more than one exact_gt_anchor candidate"):
        _build(tmp_path, mutate_rows=duplicate_one_anchor)


def test_matched_e_stratum_mismatch_fails_closed(tmp_path: Path) -> None:
    def make_one_unmatched_e_look_matched(rows: dict[str, list[dict[str, Any]]]) -> None:
        for row in rows["sidecars"]:
            if row["image_id"] == "702" and row["row_index"] == TP_PER_IMAGE["702"]:
                row["strict_match_status"] = "matched"
                row["strict_match_gt_owner_id"] = "gt:702:0"
        for row in rows["contexts"]:
            for prefix_row in row["prefix_rows"]:
                if row["image_id"] == "702" and prefix_row["row_index"] == TP_PER_IMAGE["702"]:
                    prefix_row["strict_match_status"] = "matched"
                    prefix_row["strict_match_gt_owner_id"] = "gt:702:0"

    with pytest.raises(builder.PlanContractError, match="the matched-E stratum is 13, not the frozen 12"):
        _build(tmp_path, mutate_rows=make_one_unmatched_e_look_matched)


def test_u_cohort_count_mismatch_fails_closed(tmp_path: Path) -> None:
    def withdraw_one_primary_support(rows: dict[str, list[dict[str, Any]]]) -> None:
        for summary in rows["owner_summaries"]:
            if summary["gt_owner_id"] == f"gt:702:{TP_PER_IMAGE['702']}":
                summary["upper_bound_u"]["usable_support_context_ids"] = []

    with pytest.raises(
        builder.PlanContractError, match="the U-bound crossing cohort is 25, not the frozen 26"
    ):
        _build(tmp_path, mutate_rows=withdraw_one_primary_support)


def test_l_sensitivity_count_mismatch_fails_closed(tmp_path: Path) -> None:
    def withdraw_one_l_support(rows: dict[str, list[dict[str, Any]]]) -> None:
        for summary in rows["owner_summaries"]:
            if summary["gt_owner_id"] == f"gt:702:{TP_PER_IMAGE['702']}":
                summary["lower_bound_l"]["usable_support_context_ids"] = []

    with pytest.raises(
        builder.PlanContractError, match="the L-bound crossing cohort is 24, not the frozen 25"
    ):
        _build(tmp_path, mutate_rows=withdraw_one_l_support)


def test_timing_control_count_mismatch_is_reported_never_patched(tmp_path: Path) -> None:
    def demote_one_timing_control(rows: dict[str, list[dict[str, Any]]]) -> None:
        target = f"gt:702:{TP_PER_IMAGE['702'] + UNMATCHED_ROWS_PER_IMAGE['702'] + 1}"
        for summary in rows["owner_summaries"]:
            if summary["gt_owner_id"] == target:
                assert summary["upper_bound_u"]["usable_support_context_ids"]
                summary["upper_bound_u"]["usable_support_context_ids"] = []

    with pytest.raises(
        builder.PlanContractError,
        match="the disjoint timing-control registry is 13, not the frozen 14",
    ):
        _build(tmp_path, mutate_rows=demote_one_timing_control)


def test_tp_replay_control_count_mismatch_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(builder, "EXPECTED_TP_REPLAY_CONTROL_COUNT", 13)
    with pytest.raises(
        builder.PlanContractError,
        match="the native true-positive replay-control registry is 12, not the frozen 13",
    ):
        _build(tmp_path)


def test_tampered_prevalence_receipt_fails_closed(tmp_path: Path) -> None:
    prevalence_root, census_root = materialize_fixture(tmp_path)
    receipt_path = prevalence_root / builder.PREVALENCE_RECEIPT_REL
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["predecessor_unit_id"] = "tampered"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    with pytest.raises(
        builder.PlanContractError, match="does not reconstruct its own receipt_content_sha256"
    ):
        builder.build_plan(
            prevalence_run_root=prevalence_root,
            census_run_root=census_root,
            output_root=tmp_path / "out",
        )


def test_census_bytes_that_disagree_with_the_prevalence_receipt_fail_closed(
    tmp_path: Path,
) -> None:
    def desynchronize_the_receipt(
        plan_receipt: dict[str, Any],
        merge_receipt: dict[str, Any],
        prevalence_receipt: dict[str, Any],
    ) -> None:
        prevalence_receipt["input_file_sha256"]["plan/context-registry.jsonl"] = "3" * 64

    prevalence_root, census_root = materialize_fixture(
        tmp_path, mutate_receipts=desynchronize_the_receipt
    )
    with pytest.raises(builder.PlanContractError, match="the two predecessor runs disagree"):
        builder.build_plan(
            prevalence_run_root=prevalence_root,
            census_run_root=census_root,
            output_root=tmp_path / "out",
        )


def test_unsealed_extra_plan_file_fails_closed(tmp_path: Path) -> None:
    def break_one_plan_digest(
        plan_receipt: dict[str, Any],
        merge_receipt: dict[str, Any],
        prevalence_receipt: dict[str, Any],
    ) -> None:
        plan_receipt["output_file_digests"]["candidate-bank.jsonl"] = "4" * 64

    prevalence_root, census_root = materialize_fixture(
        tmp_path, mutate_receipts=break_one_plan_digest
    )
    with pytest.raises(
        builder.PlanContractError,
        match="plan/candidate-bank.jsonl bytes do not match the digest sealed",
    ):
        builder.build_plan(
            prevalence_run_root=prevalence_root,
            census_run_root=census_root,
            output_root=tmp_path / "out",
        )


def test_predecessor_contract_failures_surface_as_plan_contract_errors(tmp_path: Path) -> None:
    prevalence_root, census_root = materialize_fixture(tmp_path)
    receipt_path = census_root / "phases/presentation/merge-receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["counts"]["owner_summary_row_count"] = 1
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    with pytest.raises(builder.PlanContractError, match="sealed census validation"):
        builder.build_plan(
            prevalence_run_root=prevalence_root,
            census_run_root=census_root,
            output_root=tmp_path / "out",
        )


def test_mixed_token_registries_across_images_fail_closed(tmp_path: Path) -> None:
    def shift_one_image_token_registry(rows: dict[str, list[dict[str, Any]]]) -> None:
        for row in rows["images"]:
            if row["image_id"] == "703":
                row["wrapper_token_ids"] = dict(WRAPPERS) | {"box_end": 99999}

    with pytest.raises(builder.PlanContractError, match="refuses to mix token identities"):
        _build(tmp_path, mutate_rows=shift_one_image_token_registry)


# ---------------------------------------------------------------------------
# Crossing-boundary unit semantics
# ---------------------------------------------------------------------------


def _states(*states: str) -> dict[int, dict[str, Any]]:
    return {
        index: {"frontier_features": {"passed_state": state}} for index, state in enumerate(states)
    }


def test_derive_crossing_boundary_takes_the_last_root_or_ahead_boundary() -> None:
    rows = _states(
        "root_no_frontier", "ahead_of_frontier", "ahead_of_frontier", "passed_by_frontier"
    )
    assert builder.derive_crossing_boundary(rows) == 2


def test_derive_crossing_boundary_uses_the_first_crossing_only() -> None:
    rows = _states(
        "root_no_frontier",
        "ahead_of_frontier",
        "passed_by_frontier",
        "ahead_of_frontier",
        "passed_by_frontier",
    )
    assert builder.derive_crossing_boundary(rows) == 1


def test_owner_that_never_passes_the_frontier_has_no_crossing() -> None:
    assert builder.derive_crossing_boundary(_states("root_no_frontier", "ahead_of_frontier")) is None


def test_at_frontier_predecessor_is_not_a_crossing() -> None:
    rows = _states("root_no_frontier", "at_frontier", "passed_by_frontier")
    assert builder.derive_crossing_boundary(rows) is None


def test_owner_passed_at_the_root_has_no_crossing() -> None:
    assert builder.derive_crossing_boundary(_states("passed_by_frontier")) is None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_writes_the_plan_with_explicit_paths(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    prevalence_root, census_root = materialize_fixture(tmp_path)
    exit_code = builder.main(
        [
            "--prevalence-run-root",
            str(prevalence_root),
            "--census-run-root",
            str(census_root),
            "--output-root",
            str(tmp_path / "cli"),
        ]
    )
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "U=26 L=25 U&L=24 matched-E=12 unmatched-E=14" in captured.out
    plan_dir = tmp_path / "cli" / builder.PLAN_DIR_NAME
    for name in (
        builder.MANIFEST_NAME,
        builder.COHORT_REGISTRY_NAME,
        builder.CONTROL_REGISTRY_NAME,
        builder.REQUEST_PLAN_NAME,
    ):
        assert (plan_dir / name).is_file()


def test_cli_returns_one_on_a_contract_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    prevalence_root, _census_root = materialize_fixture(tmp_path)
    exit_code = builder.main(
        [
            "--prevalence-run-root",
            str(prevalence_root),
            "--census-run-root",
            str(tmp_path / "does-not-exist"),
            "--output-root",
            str(tmp_path / "cli"),
        ]
    )
    assert exit_code == 1
    assert "FAIL-CLOSED" in capsys.readouterr().err


def test_census_run_root_defaults_to_the_sealed_predecessor(tmp_path: Path) -> None:
    prevalence_root, census_root = materialize_fixture(tmp_path)
    manifest = builder.build_plan(
        prevalence_run_root=prevalence_root, output_root=tmp_path / "defaulted"
    )
    assert manifest["lineage"]["census_run_root"] == str(census_root)
