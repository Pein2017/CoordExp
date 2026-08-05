"""Contract tests for the sorted owner accessibility census shard merge.

Every fixture here is synthetic and CPU-only.  The plan directory is built with
the *real* planner helpers (``build_candidate_bank``, ``build_capture_rules``,
``build_query_suffix``) so the candidate bank, the alias collapse, the strict
assignment and the sealed adequacy bands under test are the production ones;
only the twelve-image panel, the contexts and the scores are stand-ins.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import copy
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research import build_sorted_owner_accessibility_census_plan as planner  # noqa: E402
from scripts.research import (  # noqa: E402
    merge_sorted_owner_accessibility_census_shards as merge,
)


IMAGE_IDS: tuple[str, ...] = tuple(sorted(planner.SPLIT_BY_IMAGE_ID, key=int))
CANVAS_WIDTH = 200
CANVAS_HEIGHT = 200

#: Distinct-candidate geometry, probed against the real candidate bank:
#: a centred box realizes all seventeen roles, a corner box clips several
#: translations into the exact anchor and lands in the ``adequate_reduced`` band.
FULL_BANK_BOX = (40, 40, 120, 120)
REDUCED_BANK_BOX = (0, 0, 40, 40)
SECOND_FULL_BANK_BOX = (60, 60, 140, 140)
#: Two fully covered owners far enough apart that no fixed role of one aliases
#: onto the other, so each owner's exact anchor is its own physical candidate.
SEPARATED_BOX_A = (10, 10, 70, 70)
SEPARATED_BOX_B = (120, 120, 180, 180)

CATEGORY_TOKEN_IDS: Mapping[str, list[int]] = {
    "person": [700, 701],
    "car": [800],
}


# ---------------------------------------------------------------------------
# Fixture specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OwnerSpec:
    owner_id: str
    box: tuple[int, int, int, int]
    description: str = "person"
    native_true_positive: bool = False
    greedy_eligible: bool = True


@dataclass(frozen=True)
class ContextSpec:
    boundary_index: int
    frontier_box: tuple[int, int, int, int] | None = None
    frontier_description: str = "person"
    loop_tail: bool = False
    consecutive_run: int = 1


@dataclass(frozen=True)
class ImageSpec:
    owners: tuple[OwnerSpec, ...]
    contexts: tuple[ContextSpec, ...]


def default_image_spec(image_id: str) -> ImageSpec:
    """One well-covered owner, a root context and one frontier context."""

    return ImageSpec(
        owners=(OwnerSpec(owner_id=f"own-{image_id}-a", box=FULL_BANK_BOX),),
        contexts=(
            ContextSpec(boundary_index=0),
            ContextSpec(boundary_index=1, frontier_box=(10, 10, 30, 30)),
        ),
    )


# ---------------------------------------------------------------------------
# Plan construction
# ---------------------------------------------------------------------------


def _prompt_tokens(image_id: str) -> list[int]:
    seed = int(image_id) % 97
    return [9000 + seed, 9001 + seed, 9002 + seed]


def _generated_prefix(image_id: str, boundary_index: int) -> list[int]:
    return [7000 + int(image_id) % 53 + step for step in range(boundary_index)]


def build_plan_dir(tmp_path: Path, overrides: Mapping[str, ImageSpec] | None = None) -> Path:
    """Materialize a sealed twelve-image plan directory."""

    specs = {image_id: default_image_spec(image_id) for image_id in IMAGE_IDS}
    specs.update(overrides or {})

    panel = {
        image_id: {"width": CANVAS_WIDTH, "height": CANVAS_HEIGHT} for image_id in IMAGE_IDS
    }
    owners_by_image: dict[str, list[dict[str, object]]] = {}
    for image_id, spec in specs.items():
        owners_by_image[image_id] = [
            {
                "gt_owner_id": owner.owner_id,
                "image_id": image_id,
                "normalized_description": owner.description,
                "description": owner.description,
                "bbox_pixel_xyxy": list(owner.box),
                "owner_sort_key": [owner.box[1], owner.box[0]],
                "greedy_eligible": owner.greedy_eligible,
                "greedy_eligibility_status": (
                    "eligible" if owner.greedy_eligible else "globally_ambiguous_neutral"
                ),
                "split": planner.SPLIT_BY_IMAGE_ID[image_id],
            }
            for owner in spec.owners
        ]

    candidates, accounting = planner.build_candidate_bank(owners_by_image, panel)

    owner_rows = [
        {
            "schema_version": planner.PLAN_SCHEMA_VERSION,
            "row_kind": "census_owner",
            **owner,
            "native_strict_match_pred_row_ids": (
                [f"pred-{image_id}-tp"] if spec_owner.native_true_positive else []
            ),
            "native_true_positive": spec_owner.native_true_positive,
            "calibration_role": (
                "native_true_positive_calibration"
                if spec_owner.native_true_positive
                else "native_false_negative"
            ),
            "excluded_from_census": False,
            "candidate_bank": accounting[str(owner["gt_owner_id"])],
        }
        for image_id, spec in specs.items()
        for owner, spec_owner in zip(owners_by_image[image_id], spec.owners, strict=True)
    ]

    image_rows = [
        {
            "schema_version": planner.PLAN_SCHEMA_VERSION,
            "row_kind": "census_image",
            "image_id": image_id,
            "split": planner.SPLIT_BY_IMAGE_ID[image_id],
            "image_width": CANVAS_WIDTH,
            "image_height": CANVAS_HEIGHT,
            "prompt_token_ids": _prompt_tokens(image_id),
            "prompt_token_ids_sha256": planner.sha256_json(_prompt_tokens(image_id)),
        }
        for image_id in IMAGE_IDS
    ]

    category_rows: list[dict[str, object]] = []
    for image_id, spec in specs.items():
        for description in sorted({owner.description for owner in spec.owners}):
            tokens = list(CATEGORY_TOKEN_IDS[description])
            suffix = planner.build_query_suffix(tokens)
            category_rows.append(
                {
                    "schema_version": planner.PLAN_SCHEMA_VERSION,
                    "row_kind": "census_category",
                    "category_query_id": f"{image_id}:{description}",
                    "image_id": image_id,
                    "split": planner.SPLIT_BY_IMAGE_ID[image_id],
                    "normalized_description": description,
                    "status": "admitted",
                    "category_token_ids": tokens,
                    "query_suffix_token_ids": suffix,
                    "query_suffix_token_ids_sha256": planner.sha256_json(suffix),
                }
            )

    context_rows: list[dict[str, object]] = []
    for image_id, spec in specs.items():
        for context in spec.contexts:
            frontier = None
            if context.frontier_box is not None:
                box = list(context.frontier_box)
                frontier = {
                    "row_index": context.boundary_index - 1,
                    "pred_row_id": f"pred-{image_id}-{context.boundary_index}",
                    "description": context.frontier_description,
                    "bbox_pixel_xyxy": [float(value) for value in box],
                    "sort_key": [float(box[1]), float(box[0])],
                    "strict_match_status": "unmatched",
                    "strict_match_gt_owner_id": None,
                }
            context_rows.append(
                {
                    "schema_version": planner.PLAN_SCHEMA_VERSION,
                    "row_kind": "census_context",
                    "context_id": f"{image_id}:boundary-{context.boundary_index:03d}",
                    "image_id": image_id,
                    "split": planner.SPLIT_BY_IMAGE_ID[image_id],
                    "boundary_index": context.boundary_index,
                    "context_role": "root" if context.boundary_index == 0 else "row_boundary",
                    "generated_prefix_token_ids": _generated_prefix(
                        image_id, context.boundary_index
                    ),
                    "loop_marking": {
                        "prior_identical_row_count": 0,
                        "consecutive_identical_row_run_length": context.consecutive_run,
                        "repeated_raw_span_sha256": None,
                        "loop_tail": context.loop_tail,
                        "flag_is_not_a_mechanism_label": True,
                    },
                    "frontier": frontier,
                    "canvas": {"width": CANVAS_WIDTH, "height": CANVAS_HEIGHT},
                }
            )

    bank_by_category: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for candidate in candidates:
        bank_by_category.setdefault(
            (str(candidate["image_id"]), str(candidate["normalized_description"])), []
        ).append(candidate)

    categories_by_image: dict[str, list[Mapping[str, object]]] = {}
    for row in category_rows:
        categories_by_image.setdefault(str(row["image_id"]), []).append(row)

    query_group_rows: list[dict[str, object]] = []
    for context in context_rows:
        image_id = str(context["image_id"])
        observed = [*_prompt_tokens(image_id), *context["generated_prefix_token_ids"]]
        for category in categories_by_image[image_id]:
            description = str(category["normalized_description"])
            suffix = list(category["query_suffix_token_ids"])
            full_prefix = [*observed, *suffix]
            observed_sha = planner.sha256_json(observed)
            query_sha = planner.sha256_json(full_prefix)
            bank = bank_by_category.get((image_id, description), [])
            query_group_rows.append(
                {
                    "schema_version": planner.PLAN_SCHEMA_VERSION,
                    "row_kind": "census_query_group",
                    "query_group_id": f"{context['context_id']}|{description}",
                    "image_id": image_id,
                    "split": planner.SPLIT_BY_IMAGE_ID[image_id],
                    "context_id": context["context_id"],
                    "category_query_id": category["category_query_id"],
                    "normalized_description": description,
                    "status": "admitted",
                    "query_suffix_token_ids": suffix,
                    "query_suffix_token_ids_sha256": planner.sha256_json(suffix),
                    "observed_prefix_sha256": observed_sha,
                    "query_prefix_sha256": query_sha,
                    "candidate_ids": sorted(str(row["candidate_id"]) for row in bank),
                    "candidate_count": len(bank),
                    "admission_receipt_id": planner.admission_receipt_id(
                        context_id=str(context["context_id"]),
                        channel=planner.CHANNEL_QUERY_SUFFIX,
                        prefix_sha256=query_sha,
                    ),
                    "proposal_boundary_gate_admission_receipt_id": (
                        planner.admission_receipt_id(
                            context_id=str(context["context_id"]),
                            channel=planner.CHANNEL_PROPOSAL_BOUNDARY_GATE,
                            prefix_sha256=observed_sha,
                        )
                    ),
                    "proposal_route_admission_receipt_id": (
                        planner.proposal_route_admission_receipt_id(
                            context_id=str(context["context_id"]),
                            observed_prefix_sha256=observed_sha,
                            category_token_ids=category["category_token_ids"],
                        )
                    ),
                }
            )

    sidecar_rows: list[dict[str, object]] = []
    for image_id in IMAGE_IDS:
        for description in sorted({str(row["normalized_description"]) for row in category_rows if str(row["image_id"]) == image_id}):
            bank = bank_by_category.get((image_id, description), [])
            if not bank:
                continue
            first = bank[0]
            sidecar_rows.append(
                {
                    "schema_version": planner.PLAN_SCHEMA_VERSION,
                    "row_kind": "native_sidecar",
                    "sidecar_id": f"native:{image_id}:0",
                    "image_id": image_id,
                    "pred_row_id": f"pred-{image_id}-tp",
                    "row_index": 0,
                    "split": planner.SPLIT_BY_IMAGE_ID[image_id],
                    "normalized_description": description,
                    "coord_token_ids": list(first["coord_token_ids"]),
                    "excluded_from_core_ranks": True,
                }
            )

    shard_rows = [
        {
            "schema_version": planner.PLAN_SCHEMA_VERSION,
            "row_kind": "census_shard",
            "shard_id": f"shard-{image_id}",
            "image_id": image_id,
            "split": planner.SPLIT_BY_IMAGE_ID[image_id],
            "estimated_work_units": 1,
            "dispatch_order": index,
        }
        for index, image_id in enumerate(IMAGE_IDS)
    ]

    capture_rules = planner.build_capture_rules()

    files: dict[str, bytes] = {
        "image-registry.jsonl": _jsonl(image_rows),
        "owner-registry.jsonl": _jsonl(owner_rows),
        "category-registry.jsonl": _jsonl(category_rows),
        "context-registry.jsonl": _jsonl(context_rows),
        "candidate-bank.jsonl": _jsonl(candidates),
        "query-group-registry.jsonl": _jsonl(query_group_rows),
        "native-sidecar-registry.jsonl": _jsonl(sidecar_rows),
        "shard-manifest.jsonl": _jsonl(shard_rows),
        planner.CAPTURE_RULES_NAME: planner.canonical_json_bytes(capture_rules) + b"\n",
    }

    receipt = {
        "schema_version": planner.PLAN_SCHEMA_VERSION,
        "unit_id": planner.UNIT_ID,
        "capture_rules_sha256": capture_rules["capture_rules_sha256"],
        "output_file_digests": {
            name: planner.hashlib.sha256(content).hexdigest()
            for name, content in sorted(files.items())
        },
    }
    receipt["receipt_content_sha256"] = planner.sha256_json(
        {key: value for key, value in receipt.items() if key != "receipt_content_sha256"}
    )

    plan_dir = tmp_path / "plan"
    plan_dir.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (plan_dir / name).write_bytes(content)
    (plan_dir / "receipt.json").write_bytes(planner.canonical_json_bytes(receipt) + b"\n")
    return plan_dir


def _jsonl(rows: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(planner.canonical_json_bytes(row) + b"\n" for row in rows)


# ---------------------------------------------------------------------------
# Shard construction
# ---------------------------------------------------------------------------

ScoreFn = Callable[[Mapping[str, object], Mapping[str, object]], float]


def default_score(group: Mapping[str, object], candidate: Mapping[str, object]) -> float:
    """Exact anchors strongest, then by fixed role order.  Score-independent."""

    return -float(candidate["representative_role_ordinal"])


@dataclass
class ShardOptions:
    score_fn: ScoreFn = default_score
    quarantined: frozenset[str] = frozenset()
    omitted: frozenset[str] = frozenset()
    mutate_scores: Callable[[str, list[dict]], list[dict]] | None = None
    mutate_receipt: Callable[[str, dict], dict] | None = None
    mutate_proposals: Callable[[str, list[dict]], list[dict]] | None = None
    mutate_free_decodes: Callable[[str, list[dict]], list[dict]] | None = None
    model_identity: Callable[[str], dict] = field(
        default=lambda image_id: {"path": "model-a", "sha256": "m" * 8}
    )


def write_shards(
    plan: merge.PlanBundle, shard_root: Path, options: ShardOptions | None = None
) -> Path:
    """Write one directory per planned image in the scorer's published shape."""

    options = options or ShardOptions()
    shard_root.mkdir(parents=True, exist_ok=True)

    for image_id in sorted(plan.images, key=int):
        if image_id in options.omitted:
            continue
        directory = shard_root / image_id
        directory.mkdir(parents=True, exist_ok=True)

        if image_id in options.quarantined:
            payload = {
                "schema_version": merge.EXPECTED_QUARANTINE_SCHEMA,
                "unit_id": planner.UNIT_ID,
                "image_id": image_id,
                "status": "quarantined",
                "reason": "synthetic_quarantine",
                "detail": "test fixture",
                "scope": "this_image_only",
                "evidence_usable": False,
            }
            (directory / merge.SHARD_QUARANTINE_NAME).write_bytes(
                planner.canonical_json_bytes(payload) + b"\n"
            )
            continue

        groups = [
            row
            for row in plan.query_groups.values()
            if str(row["image_id"]) == image_id and row["status"] == "admitted"
        ]
        groups.sort(key=lambda row: str(row["query_group_id"]))

        score_rows: list[dict] = []
        free_rows: list[dict] = []
        admissions: dict[str, dict] = {}
        for group in groups:
            context_id = str(group["context_id"])
            query_sha = str(group["query_prefix_sha256"])
            observed_sha = str(group["observed_prefix_sha256"])
            query_receipt = planner.admission_receipt_id(
                context_id=context_id,
                channel=planner.CHANNEL_QUERY_SUFFIX,
                prefix_sha256=query_sha,
            )
            gate_receipt = planner.admission_receipt_id(
                context_id=context_id,
                channel=planner.CHANNEL_PROPOSAL_BOUNDARY_GATE,
                prefix_sha256=observed_sha,
            )
            category = plan.categories[str(group["category_query_id"])]
            route_receipt = planner.proposal_route_admission_receipt_id(
                context_id=context_id,
                observed_prefix_sha256=observed_sha,
                category_token_ids=category["category_token_ids"],
            )
            admissions[query_receipt] = {
                "schema_version": merge.EXPECTED_ADMISSION_SCHEMA,
                "admission_receipt_id": query_receipt,
                "channel": planner.CHANNEL_QUERY_SUFFIX,
                "context_id": context_id,
                "query_group_id": str(group["query_group_id"]),
                "admission_key": "exact_query_prefix_sha256",
                "observed_prefix_sha256": observed_sha,
                "query_prefix_sha256": query_sha,
                "query_suffix_token_ids_sha256": str(
                    group["query_suffix_token_ids_sha256"]
                ),
                "inherited_from_another_prefix": False,
                "admitted": True,
            }
            admissions[gate_receipt] = {
                "schema_version": merge.EXPECTED_ADMISSION_SCHEMA,
                "admission_receipt_id": gate_receipt,
                "channel": planner.CHANNEL_PROPOSAL_BOUNDARY_GATE,
                "proposal_scope": "boundary_gate",
                "context_id": context_id,
                "category_query_id": None,
                "admission_key": "exact_observed_prefix_sha256",
                "observed_prefix_sha256": observed_sha,
                "query_prefix_sha256": observed_sha,
                "inherited_from_another_prefix": False,
                "covered_by_a_query_suffix_admission": False,
                "admitted": True,
            }
            admissions[route_receipt] = {
                "schema_version": merge.EXPECTED_ADMISSION_SCHEMA,
                "admission_receipt_id": route_receipt,
                "channel": planner.CHANNEL_PROPOSAL_CATEGORY_ROUTE,
                "proposal_scope": "category_routing_path",
                "context_id": context_id,
                "category_query_id": str(group["category_query_id"]),
                "normalized_description": str(group["normalized_description"]),
                "admission_key": "exact_executed_routing_path_sha256",
                "observed_prefix_sha256": observed_sha,
                "query_prefix_sha256": observed_sha,
                "routing_path_digest": planner.proposal_route_digest(
                    category["category_token_ids"]
                ),
                "executed_routing_prefix_sha256": planner.sha256_json(
                    [observed_sha, str(group["query_group_id"])]
                ),
                "inherited_from_another_prefix": False,
                "inherited_from_another_category": False,
                "covered_by_a_query_suffix_admission": False,
                "admitted": True,
            }

            for candidate_id in group["candidate_ids"]:
                candidate = plan.candidates[str(candidate_id)]
                score_rows.append(
                    {
                        "schema_version": merge.EXPECTED_SCORE_SCHEMA,
                        "row_kind": "census_localization_score",
                        "request_id": f"{group['query_group_id']}|{candidate_id}",
                        "query_group_id": str(group["query_group_id"]),
                        "image_id": image_id,
                        "context_id": context_id,
                        "normalized_description": str(group["normalized_description"]),
                        "candidate_id": str(candidate_id),
                        "coord_token_ids": list(candidate["coord_token_ids"]),
                        "complete_box_logprob_sum": float(
                            options.score_fn(group, candidate)
                        ),
                        "observed_prefix_sha256": observed_sha,
                        "query_prefix_sha256": query_sha,
                        "query_suffix_token_ids_sha256": str(
                            group["query_suffix_token_ids_sha256"]
                        ),
                        "admission_receipt_id": query_receipt,
                        "rank_key": {
                            "image_id": image_id,
                            "context_id": context_id,
                            "normalized_description": str(group["normalized_description"]),
                        },
                        "is_sidecar": False,
                        "enters_core_ranks": True,
                    }
                )
            if group["candidate_ids"]:
                first = plan.candidates[str(group["candidate_ids"][0])]
                free_rows.append(
                    {
                        "schema_version": merge.EXPECTED_FREE_DECODE_SCHEMA,
                        "row_kind": merge.FREE_BOX_ROW_KIND,
                        "well_formed_box": True,
                        "sidecar_id": f"free:{group['query_group_id']}",
                        "query_group_id": str(group["query_group_id"]),
                        "image_id": image_id,
                        "context_id": context_id,
                        "normalized_description": str(group["normalized_description"]),
                        "coord_token_ids": list(first["coord_token_ids"]),
                        "is_sidecar": True,
                        "enters_core_ranks": False,
                    }
                )

        proposal_rows: list[dict] = []
        for context_id in sorted({str(row["context_id"]) for row in groups}):
            context = plan.contexts[context_id]
            observed_sha = next(
                str(row["observed_prefix_sha256"])
                for row in groups
                if str(row["context_id"]) == context_id
            )
            descriptions = sorted(
                {
                    str(row["normalized_description"])
                    for row in groups
                    if str(row["context_id"]) == context_id
                }
            )
            proposal_rows.append(
                {
                    "schema_version": merge.EXPECTED_PROPOSAL_SCHEMA,
                    "row_kind": "census_proposal_surface",
                    "context_id": context_id,
                    "image_id": image_id,
                    "split": str(context["split"]),
                    "context_role": str(context["context_role"]),
                    "observed_prefix_sha256": observed_sha,
                    "boundary_gate_admission_receipt_id": planner.admission_receipt_id(
                        context_id=context_id,
                        channel=planner.CHANNEL_PROPOSAL_BOUNDARY_GATE,
                        prefix_sha256=observed_sha,
                    ),
                    "boundary_gate": {
                        "continue_logprob": -0.5,
                        "stop_logprob": -2.5,
                        "continue_vs_stop_logprob_margin": 2.0,
                    },
                    "category_routing_event": [
                        {
                            "normalized_description": description,
                            "channel": planner.CHANNEL_PROPOSAL_CATEGORY_ROUTE,
                            "routing_path_digest": planner.proposal_route_digest(
                                CATEGORY_TOKEN_IDS[description]
                            ),
                            "admission_receipt_id": (
                                planner.proposal_route_admission_receipt_id(
                                    context_id=context_id,
                                    observed_prefix_sha256=observed_sha,
                                    category_token_ids=CATEGORY_TOKEN_IDS[description],
                                )
                            ),
                            "raw_sequence_logprob_sum": -1.0 - index,
                            "within_context_rank": index + 1,
                            "within_context_population": len(descriptions),
                            "row_prefix_block_raw_sequence_logprob_sum": -2.0 - index,
                        }
                        for index, description in enumerate(descriptions)
                    ],
                }
            )

        if options.mutate_scores is not None:
            score_rows = options.mutate_scores(image_id, score_rows)
        if options.mutate_proposals is not None:
            proposal_rows = options.mutate_proposals(image_id, proposal_rows)
        if options.mutate_free_decodes is not None:
            free_rows = options.mutate_free_decodes(image_id, free_rows)

        receipt = {
            "schema_version": merge.EXPECTED_SHARD_RECEIPT_SCHEMA,
            "unit_id": planner.UNIT_ID,
            "image_id": image_id,
            "split": str(plan.shards[image_id]["split"]),
            "status": "captured",
            "capture_completeness": merge.COMPLETE_SHARD,
            "subset_capture": {
                "is_subset": False,
                "usable_as_complete_shard_evidence": True,
            },
            "code": {
                "executed_source_sha256": "scorer-source",
                "planner_source_sha256": "planner-source",
            },
            "plan": {
                "plan_dir": str(plan.plan_dir),
                "receipt_content_sha256": plan.receipt["receipt_content_sha256"],
                "capture_rules_sha256": plan.capture_rules["capture_rules_sha256"],
            },
            "backend_identity": {
                "backend": "fake",
                "is_real_model": True,
                "usable_as_evidence": True,
                "model_identity": options.model_identity(image_id),
                "tokenizer_identity": {"path": "tokenizer-a"},
            },
            # The scorer's authoritative admission location.
            "admission": {
                "channels": list(planner.ADMISSION_CHANNELS),
                "all_admitted": all(row["admitted"] for row in admissions.values()),
                "receipts": [admissions[key] for key in sorted(admissions)],
            },
            "scalar_admission": {"target_blind": True},
            "counts": {
                "localization_score_rows": len(score_rows),
                "proposal_surface_rows": len(proposal_rows),
            },
        }
        if options.mutate_receipt is not None:
            receipt = options.mutate_receipt(image_id, receipt)

        x1_rows = [
            {
                "schema_version": merge.EXPECTED_SCORE_SCHEMA,
                "row_kind": "census_x1_distribution",
                "query_group_id": str(group["query_group_id"]),
                "role": "diagnostic_only_never_a_2d_heatmap_never_a_rank",
            }
            for group in groups
        ]

        (directory / merge.SHARD_SCORES_NAME).write_bytes(_jsonl(score_rows))
        (directory / merge.SHARD_X1_NAME).write_bytes(_jsonl(x1_rows))
        (directory / merge.SHARD_PROPOSAL_NAME).write_bytes(_jsonl(proposal_rows))
        (directory / merge.SHARD_FREE_DECODE_NAME).write_bytes(_jsonl(free_rows))
        (directory / merge.SHARD_RECEIPT_NAME).write_bytes(
            planner.canonical_json_bytes(receipt) + b"\n"
        )
    return shard_root


def fixed_calibration(
    *, theta_lift: float = 0.0, theta_conc: float = -1.0
) -> merge.SupportCalibration:
    """A synthetic sealed calibration for disposition-ladder tests.

    ``theta_conc`` sits below zero because an owner whose L population is a
    single strictly-assigned candidate has a local concentration of exactly
    zero by construction; a real calibration derives these from discovery TPs.
    """

    sealed = planner.build_capture_rules()["owner_support"]
    return merge.SupportCalibration(
        theta_peak_lift=theta_lift,
        theta_local_concentration=theta_conc,
        epsilon=float(sealed["epsilons"]["support_epsilon"]),
        quantile=float(sealed["support_calibration"]["primary_quantile"]),
        observation_count=64,
        per_category_counts={"person": 64},
        sensitivity={"role": "report_only_never_moves_a_threshold"},
        exclusions=(),
        consumed_shard_digests=(),
        capture_manifest_sha256="0" * 64,
        category_contribution_min=int(
            sealed["support_calibration"]["category_contribution_min"]
        ),
        underrepresented_flag=str(sealed["support_calibration"]["underrepresented_flag"]),
        cross_context_delta_epsilon=float(
            sealed["epsilons"]["cross_context_delta_epsilon"]
        ),
        statistics=tuple(sealed["support_definition"]["statistics"]),
    )


def build_merged(
    tmp_path: Path,
    *,
    overrides: Mapping[str, ImageSpec] | None = None,
    options: ShardOptions | None = None,
    calibration: merge.SupportCalibration | None = None,
    phase: str = merge.PHASE_FULL,
    allowlist: Sequence[str] | None = None,
) -> merge.MergedCensus:
    plan_dir = build_plan_dir(tmp_path, overrides)
    plan = merge.load_plan(plan_dir)
    write_shards(plan, tmp_path / "shards", options)
    return merge.merge_census(
        plan,
        tmp_path / "shards",
        phase=phase,
        allowlist=allowlist,
        calibration=calibration,
    )


# ---------------------------------------------------------------------------
# Happy path: events, ranks, posteriors
# ---------------------------------------------------------------------------


def test_merge_publishes_ranked_events_over_unique_coordinate_tuples(tmp_path: Path) -> None:
    merged = build_merged(tmp_path)

    assert merged.events, "the merge published no events"
    by_group: dict[tuple[str, str, str], list[dict]] = {}
    for row in merged.events:
        by_group.setdefault(
            (row["image_id"], row["context_id"], row["normalized_description"]), []
        ).append(row)

    for key, rows in by_group.items():
        ranks = sorted(int(row["competition"]["rank"]) for row in rows)
        assert ranks == list(range(1, len(rows) + 1)), f"{key} ranks are not a permutation"
        # One mass per unique coordinate tuple.
        tuples = [tuple(row["coord_token_ids"]) for row in rows]
        assert len(set(tuples)) == len(tuples)
        posterior = sum(float(row["competition"]["within_group_posterior"]) for row in rows)
        assert posterior == pytest.approx(1.0)
        best = [row for row in rows if int(row["competition"]["rank"]) == 1][0]
        assert best["competition"]["margin_to_group_best"] == pytest.approx(0.0)
        for row in rows:
            assert row["competition"]["is_model_probability"] is False
            assert row["competition"]["sidecars_excluded_from_population"] is True


def test_ranks_never_span_two_contexts_or_categories(tmp_path: Path) -> None:
    merged = build_merged(tmp_path)
    for row in merged.events:
        assert row["rank_key"] == {
            "image_id": row["image_id"],
            "context_id": row["context_id"],
            "normalized_description": row["normalized_description"],
        }
        assert int(row["competition"]["population_size"]) == len(
            [
                other
                for other in merged.events
                if other["rank_key"] == row["rank_key"]
            ]
        )


def test_sidecars_join_provenance_but_add_no_rank_mass(tmp_path: Path) -> None:
    merged = build_merged(tmp_path)
    joined = [
        row
        for row in merged.events
        if row["sidecar_provenance"]["native_sidecar_ids"]
        or row["sidecar_provenance"]["free_decode_sidecar_ids"]
    ]
    assert joined, "fixture did not exercise a sidecar join"
    for row in joined:
        assert row["sidecar_provenance"]["join_semantics"] == (
            "joins_provenance_never_adds_rank_mass"
        )
    # A joined sidecar must not inflate the population it joins.
    for row in joined:
        population = int(row["competition"]["population_size"])
        siblings = [
            other for other in merged.events if other["rank_key"] == row["rank_key"]
        ]
        assert population == len(siblings)


# ---------------------------------------------------------------------------
# Quarantine and the global stop policy
# ---------------------------------------------------------------------------


def test_two_quarantined_images_are_ignored_not_fatal(tmp_path: Path) -> None:
    quarantined = frozenset({IMAGE_IDS[0], IMAGE_IDS[1]})
    merged = build_merged(tmp_path, options=ShardOptions(quarantined=quarantined))

    ledger = merged.receipt["quarantine_ledger"]
    assert ledger["global_stop_triggered"] is False
    assert set(ledger["quarantined_image_ids"]) == quarantined
    assert ledger["evidence_from_failed_shards"] == "forbidden_never_read"
    # No event, owner-context row or summary may come from a quarantined image.
    for row in merged.events:
        assert row["image_id"] not in quarantined
    for row in merged.owner_summaries:
        assert row["image_id"] not in quarantined


def test_more_than_two_image_quarantines_triggers_global_stop(tmp_path: Path) -> None:
    quarantined = frozenset({IMAGE_IDS[0], IMAGE_IDS[1], IMAGE_IDS[2]})
    plan_dir = build_plan_dir(tmp_path)
    plan = merge.load_plan(plan_dir)
    write_shards(plan, tmp_path / "shards", ShardOptions(quarantined=quarantined))

    with pytest.raises(merge.GlobalStopError) as excinfo:
        merge.merge_census(plan, tmp_path / "shards")
    assert "global stop" in str(excinfo.value).lower()


def test_missing_shards_count_toward_the_same_unusable_budget(tmp_path: Path) -> None:
    plan_dir = build_plan_dir(tmp_path)
    plan = merge.load_plan(plan_dir)
    write_shards(
        plan,
        tmp_path / "shards",
        ShardOptions(quarantined=frozenset({IMAGE_IDS[0]}), omitted=frozenset({IMAGE_IDS[1], IMAGE_IDS[2]})),
    )
    with pytest.raises(merge.GlobalStopError):
        merge.merge_census(plan, tmp_path / "shards")


# ---------------------------------------------------------------------------
# Admission, suffix, and duplicate-event rejection
# ---------------------------------------------------------------------------


def test_row_without_query_suffix_digest_is_rejected_as_pre_p0(tmp_path: Path) -> None:
    def strip_suffix(image_id: str, rows: list[dict]) -> list[dict]:
        if image_id == IMAGE_IDS[0] and rows:
            rows[0] = {
                key: value
                for key, value in rows[0].items()
                if key != "query_suffix_token_ids_sha256"
            }
        return rows

    plan_dir = build_plan_dir(tmp_path)
    plan = merge.load_plan(plan_dir)
    write_shards(plan, tmp_path / "shards", ShardOptions(mutate_scores=strip_suffix))
    with pytest.raises(merge.MergeContractError) as excinfo:
        merge.merge_census(plan, tmp_path / "shards")
    assert "pre-P0" in str(excinfo.value)


def test_row_with_foreign_query_suffix_digest_is_rejected(tmp_path: Path) -> None:
    def retag(image_id: str, rows: list[dict]) -> list[dict]:
        if image_id == IMAGE_IDS[0] and rows:
            rows[0] = {**rows[0], "query_suffix_token_ids_sha256": "0" * 64}
        return rows

    plan_dir = build_plan_dir(tmp_path)
    plan = merge.load_plan(plan_dir)
    write_shards(plan, tmp_path / "shards", ShardOptions(mutate_scores=retag))
    with pytest.raises(merge.MergeContractError, match="query suffix digest"):
        merge.merge_census(plan, tmp_path / "shards")


def test_score_row_without_covering_admission_receipt_is_rejected(tmp_path: Path) -> None:
    def drop_query_admissions(image_id: str, receipt: dict) -> dict:
        if image_id != IMAGE_IDS[0]:
            return receipt
        receipt = copy.deepcopy(receipt)
        receipt["admission"]["receipts"] = [
            row
            for row in receipt["admission"]["receipts"]
            if row["channel"] != planner.CHANNEL_QUERY_SUFFIX
        ]
        return receipt

    plan_dir = build_plan_dir(tmp_path)
    plan = merge.load_plan(plan_dir)
    write_shards(plan, tmp_path / "shards", ShardOptions(mutate_receipt=drop_query_admissions))
    with pytest.raises(merge.MergeContractError, match="no covering admission receipt"):
        merge.merge_census(plan, tmp_path / "shards")


def test_unadmitted_context_cannot_carry_scored_rows(tmp_path: Path) -> None:
    def refuse_admission(image_id: str, receipt: dict) -> dict:
        if image_id != IMAGE_IDS[0]:
            return receipt
        receipt = copy.deepcopy(receipt)
        for row in receipt["admission"]["receipts"]:
            if row["channel"] == planner.CHANNEL_QUERY_SUFFIX:
                row["admitted"] = False
        return receipt

    plan_dir = build_plan_dir(tmp_path)
    plan = merge.load_plan(plan_dir)
    write_shards(plan, tmp_path / "shards", ShardOptions(mutate_receipt=refuse_admission))
    with pytest.raises(merge.MergeContractError, match="did not admit its context"):
        merge.merge_census(plan, tmp_path / "shards")


@pytest.mark.parametrize(
    "dropped_channel",
    [planner.CHANNEL_PROPOSAL_BOUNDARY_GATE, planner.CHANNEL_PROPOSAL_CATEGORY_ROUTE],
)
def test_an_admission_on_one_channel_never_covers_another(
    tmp_path: Path, dropped_channel: str
) -> None:
    """Every proposal channel needs its own receipt.

    Dropping one channel's receipts leaves the other proposal channel and every
    query-suffix receipt for the same contexts intact, so the merge can only
    pass if it is genuinely keying coverage on the channel.
    """

    def drop_channel(image_id: str, receipt: dict) -> dict:
        if image_id != IMAGE_IDS[0]:
            return receipt
        receipt = copy.deepcopy(receipt)
        receipt["admission"]["receipts"] = [
            row
            for row in receipt["admission"]["receipts"]
            if row["channel"] != dropped_channel
        ]
        return receipt

    plan_dir = build_plan_dir(tmp_path)
    plan = merge.load_plan(plan_dir)
    write_shards(plan, tmp_path / "shards", ShardOptions(mutate_receipt=drop_channel))
    with pytest.raises(merge.MergeContractError) as excinfo:
        merge.merge_census(plan, tmp_path / "shards")
    message = str(excinfo.value)
    assert dropped_channel in message
    assert "another channel can never cover it" in message


def test_two_categories_never_share_a_proposal_route_admission(tmp_path: Path) -> None:
    """A routing event may not inherit another category's route receipt."""

    image_id = IMAGE_IDS[0]
    overrides = {
        image_id: ImageSpec(
            owners=(
                OwnerSpec(owner_id="own-person", box=FULL_BANK_BOX, description="person"),
                OwnerSpec(
                    owner_id="own-car", box=SECOND_FULL_BANK_BOX, description="car"
                ),
            ),
            contexts=(ContextSpec(boundary_index=1, frontier_box=(10, 10, 30, 30)),),
        )
    }

    def swap_route_receipts(shard_image_id: str, rows: list[dict]) -> list[dict]:
        if shard_image_id != image_id:
            return rows
        for row in rows:
            events = row["category_routing_event"]
            if len(events) > 1:
                events[0] = {
                    **events[0],
                    "admission_receipt_id": events[1]["admission_receipt_id"],
                }
        return rows

    plan_dir = build_plan_dir(tmp_path, overrides)
    plan = merge.load_plan(plan_dir)
    write_shards(
        plan, tmp_path / "shards", ShardOptions(mutate_proposals=swap_route_receipts)
    )
    with pytest.raises(merge.MergeContractError, match="channel/prefix require"):
        merge.merge_census(plan, tmp_path / "shards")


def test_row_referencing_another_contexts_admission_is_rejected(tmp_path: Path) -> None:
    def relabel(image_id: str, rows: list[dict]) -> list[dict]:
        if image_id == IMAGE_IDS[0] and len(rows) > 1:
            foreign = next(
                row["admission_receipt_id"]
                for row in rows
                if row["context_id"] != rows[0]["context_id"]
            )
            rows[0] = {**rows[0], "admission_receipt_id": foreign}
        return rows

    plan_dir = build_plan_dir(tmp_path)
    plan = merge.load_plan(plan_dir)
    write_shards(plan, tmp_path / "shards", ShardOptions(mutate_scores=relabel))
    with pytest.raises(merge.MergeContractError, match="channel/prefix require"):
        merge.merge_census(plan, tmp_path / "shards")


def test_duplicate_context_category_coordinate_event_is_rejected(tmp_path: Path) -> None:
    def duplicate(image_id: str, rows: list[dict]) -> list[dict]:
        if image_id == IMAGE_IDS[0] and rows:
            rows.append(dict(rows[0]))
        return rows

    plan_dir = build_plan_dir(tmp_path)
    plan = merge.load_plan(plan_dir)
    write_shards(plan, tmp_path / "shards", ShardOptions(mutate_scores=duplicate))
    with pytest.raises(merge.MergeContractError, match="duplicates the .* event"):
        merge.merge_census(plan, tmp_path / "shards")


# ---------------------------------------------------------------------------
# Plan, capture-rule, and runtime digest binding
# ---------------------------------------------------------------------------


def test_tampered_plan_file_fails_its_sealed_digest(tmp_path: Path) -> None:
    plan_dir = build_plan_dir(tmp_path)
    path = plan_dir / "owner-registry.jsonl"
    path.write_bytes(path.read_bytes() + b'{"gt_owner_id":"smuggled"}\n')
    with pytest.raises(merge.MergeContractError, match="sealed digest"):
        merge.load_plan(plan_dir)


def test_shard_captured_against_other_capture_rules_is_rejected(tmp_path: Path) -> None:
    def drift(image_id: str, receipt: dict) -> dict:
        if image_id != IMAGE_IDS[0]:
            return receipt
        receipt = copy.deepcopy(receipt)
        receipt["plan"]["capture_rules_sha256"] = "f" * 64
        return receipt

    plan_dir = build_plan_dir(tmp_path)
    plan = merge.load_plan(plan_dir)
    write_shards(plan, tmp_path / "shards", ShardOptions(mutate_receipt=drift))
    with pytest.raises(merge.MergeContractError, match="different capture rules"):
        merge.merge_census(plan, tmp_path / "shards")


def test_shards_from_two_model_identities_cannot_be_merged(tmp_path: Path) -> None:
    def identity(image_id: str) -> dict:
        return {"path": "model-b"} if image_id == IMAGE_IDS[0] else {"path": "model-a"}

    plan_dir = build_plan_dir(tmp_path)
    plan = merge.load_plan(plan_dir)
    write_shards(plan, tmp_path / "shards", ShardOptions(model_identity=identity))
    with pytest.raises(merge.MergeContractError, match="model_identity_sha256"):
        merge.merge_census(plan, tmp_path / "shards")


def test_bank_adequacy_thresholds_must_be_bound_from_the_capture_rules(
    tmp_path: Path,
) -> None:
    """The merge must never substitute an adequacy threshold of its own."""

    plan_dir = build_plan_dir(tmp_path)
    plan = merge.load_plan(plan_dir)
    stripped = {
        key: value for key, value in plan.capture_rules.items() if key != "owner_support"
    }
    crippled = merge.PlanBundle(
        plan_dir=plan.plan_dir,
        receipt=plan.receipt,
        capture_rules=stripped,
        images=plan.images,
        owners=plan.owners,
        categories=plan.categories,
        contexts=plan.contexts,
        candidates=plan.candidates,
        query_groups=plan.query_groups,
        native_sidecars=plan.native_sidecars,
        shards=plan.shards,
    )
    with pytest.raises(merge.MergeContractError, match="will not substitute"):
        merge.load_bank_adequacy_rule(crippled)

    rule = merge.load_bank_adequacy_rule(plan)
    sealed = plan.capture_rules["owner_support"]["bank_adequacy_rule"]
    assert rule.full_at_least == sealed["full_at_least"]
    assert rule.adequate_at_least == sealed["adequate_at_least"]
    assert "adequate_reduced" in rule.eligible_statuses


# ---------------------------------------------------------------------------
# Owner-context continuous features
# ---------------------------------------------------------------------------


def test_owner_context_reconstructs_signed_frontier_geometry(tmp_path: Path) -> None:
    image_id = IMAGE_IDS[0]
    overrides = {
        image_id: ImageSpec(
            owners=(
                OwnerSpec(owner_id="own-ahead", box=(100, 100, 160, 160)),
                OwnerSpec(owner_id="own-behind", box=(10, 10, 50, 50)),
            ),
            contexts=(
                ContextSpec(boundary_index=0),
                ContextSpec(boundary_index=1, frontier_box=(20, 60, 60, 100)),
            ),
        )
    }
    merged = build_merged(tmp_path, overrides=overrides)
    rows = {
        (row["gt_owner_id"], row["boundary_index"]): row
        for row in merged.owner_contexts
        if row["image_id"] == image_id
    }

    root = rows[("own-ahead", 0)]["frontier_features"]
    assert root["frontier_present"] is False
    assert root["signed_frontier_ordinal_distance"] is None
    assert root["passed_state"] == "root_no_frontier"

    # Frontier sort key is [y1, x1] = [60, 20]; the two owners sort at [100, 100]
    # and [10, 10], so exactly one owner lies on each side and neither sits at
    # the frontier.  Zero is reserved for an owner that sorts exactly there.
    ahead = rows[("own-ahead", 1)]["frontier_features"]
    behind = rows[("own-behind", 1)]["frontier_features"]
    assert ahead["signed_frontier_ordinal_distance"] == 1
    assert ahead["passed_state"] == "ahead_of_frontier"
    assert behind["signed_frontier_ordinal_distance"] == -1
    assert behind["passed_state"] == "passed_by_frontier"
    assert ahead["signed_frontier_pixel_distance_y"] == pytest.approx(40.0)
    assert behind["signed_frontier_pixel_distance_y"] == pytest.approx(-50.0)
    assert ahead["abs_signed_frontier_ordinal_distance"] == abs(
        ahead["signed_frontier_ordinal_distance"]
    )
    assert behind["same_description_owners_ahead_of_frontier"] == 1
    for row in (ahead, behind):
        overlap = row["frontier_overlap"]
        assert overlap["intersection_over_union"] is not None
        assert overlap["center_offset_pixels"] is not None
        assert overlap["extent_ratio"] is not None


def test_category_proposal_channel_is_reported_separately(tmp_path: Path) -> None:
    merged = build_merged(tmp_path)
    row = merged.owner_contexts[0]
    channel = row["category_proposal_channel"]
    # The frozen split: a context-level boundary gate and a per-(context,
    # category) routing channel, admitted separately from each other and from
    # the query-suffix channel.
    assert channel["channels"] == [
        planner.CHANNEL_PROPOSAL_BOUNDARY_GATE,
        planner.CHANNEL_PROPOSAL_CATEGORY_ROUTE,
    ]
    assert planner.CHANNEL_QUERY_SUFFIX not in channel["channels"]
    assert channel["separate_from_localization"] is True
    assert channel["combined_with_localization"] is False
    assert channel["boundary_gate"]["semantics"] == (
        "gate_only_never_description_accessibility"
    )
    assert channel["category_routing_event"]["aggregation"] == (
        "raw_sequence_sum_no_token_mean"
    )
    # Neither proposal quantity leaks into any localization quantity.
    assert "boundary_gate" not in row["localization"]
    assert "category_routing_event" not in row["localization"]


def test_owner_context_publishes_every_sealed_required_quantity(tmp_path: Path) -> None:
    merged = build_merged(tmp_path)
    required = merged.plan.capture_rules["owner_support"]["required_per_owner_context_fields"]
    for row in merged.owner_contexts:
        for name in required:
            assert name in row["localization"], f"{name} missing from owner-context row"
        assert row["localization"]["exclusion_filtered_primary_max"] is not None


# ---------------------------------------------------------------------------
# Cross-owner aliasing, collisions, and the L/U bounds
# ---------------------------------------------------------------------------


def _twin_overrides(image_id: str) -> dict[str, ImageSpec]:
    """Two same-description owners on the identical box.

    Their fixed roles realize identical coordinate tuples, so alias collapse
    makes them one physical candidate with two generators, and every candidate
    is ambiguity-neutral rather than strictly assigned.
    """

    return {
        image_id: ImageSpec(
            owners=(
                OwnerSpec(owner_id="own-twin-a", box=FULL_BANK_BOX),
                OwnerSpec(owner_id="own-twin-b", box=FULL_BANK_BOX),
            ),
            contexts=(
                ContextSpec(boundary_index=0),
                ContextSpec(boundary_index=1, frontier_box=(10, 10, 30, 30)),
            ),
        )
    }


def test_cross_owner_alias_carries_a_single_rank_mass(tmp_path: Path) -> None:
    image_id = IMAGE_IDS[0]
    merged = build_merged(tmp_path, overrides=_twin_overrides(image_id))

    shared = [
        row
        for row in merged.events
        if row["image_id"] == image_id and row["candidate_provenance"]["cross_owner_generated"]
    ]
    assert shared, "fixture did not produce a cross-owner physical candidate"
    for row in shared:
        assert row["candidate_provenance"]["generator_owner_count"] == 2
        assert set(row["candidate_provenance"]["generator_gt_owner_ids"]) == {
            "own-twin-a",
            "own-twin-b",
        }
        assert row["candidate_provenance"]["role"] == (
            "provenance_only_never_rank_or_assignment"
        )

    # One coordinate tuple, one event, in each group.
    for group_key in {
        (row["image_id"], row["context_id"], row["normalized_description"])
        for row in shared
    }:
        rows = [
            row
            for row in merged.events
            if (row["image_id"], row["context_id"], row["normalized_description"])
            == group_key
        ]
        tuples = [tuple(row["coord_token_ids"]) for row in rows]
        assert len(set(tuples)) == len(tuples)

    # Both owners see the shared candidate in their generator-local neighbourhood.
    for owner_id in ("own-twin-a", "own-twin-b"):
        contexts = [
            row for row in merged.owner_contexts if row["gt_owner_id"] == owner_id
        ]
        assert contexts
        assert all(
            row["localization"]["counts"]["generator_local_event_count"] > 0
            for row in contexts
        )


def test_ambiguous_candidates_enter_only_the_upper_bound(tmp_path: Path) -> None:
    image_id = IMAGE_IDS[0]
    merged = build_merged(tmp_path, overrides=_twin_overrides(image_id))
    rows = [row for row in merged.owner_contexts if row["gt_owner_id"] == "own-twin-a"]
    assert rows
    for row in rows:
        bounds = row["localization"]["generator_local_max_excluding_other_owner_strict"]
        assert row["localization"]["ambiguous_upper_max"]["count"] > 0
        assert row["localization"]["strict_assigned_max"]["value"] is None
        # Ambiguity-neutral mass exists only in the U bound.
        assert bounds["ambiguity_included_u"]["count"] > bounds["ambiguity_excluded_l"]["count"]


def test_other_owner_strict_candidate_moves_to_the_collision_diagnostic(
    tmp_path: Path,
) -> None:
    image_id = IMAGE_IDS[0]
    overrides = {
        image_id: ImageSpec(
            owners=(
                OwnerSpec(owner_id="own-left", box=FULL_BANK_BOX),
                OwnerSpec(owner_id="own-right", box=SECOND_FULL_BANK_BOX),
            ),
            contexts=(ContextSpec(boundary_index=1, frontier_box=(10, 10, 30, 30)),),
        )
    }
    merged = build_merged(tmp_path, overrides=overrides)
    rows = [row for row in merged.owner_contexts if row["image_id"] == image_id]
    assert rows

    collided = [
        row for row in rows if row["collision_diagnostic"]["other_owner_strict_event_count"] > 0
    ]
    assert collided, "fixture did not produce a cross-owner strict collision"
    for row in collided:
        diagnostic = row["collision_diagnostic"]
        assert diagnostic["role"] == (
            "excluded_from_target_support_never_evidence_against_target"
        )
        # The excluded candidates are not counted in the exclusion-filtered set.
        counts = row["localization"]["counts"]
        assert counts["exclusion_filtered_event_count"] == (
            counts["generator_local_event_count"]
            - diagnostic["other_owner_strict_event_count"]
        )
        assert row["gt_owner_id"] not in diagnostic["other_owner_strict_gt_owner_ids"]


def test_strict_table_is_parallel_and_never_the_primary(tmp_path: Path) -> None:
    merged = build_merged(tmp_path)
    for row in merged.owner_contexts:
        assert row["localization"]["primary_neighbourhood"] == "generator_local_landscape"
        assert row["strict_table"]["role"] == (
            "parallel_owner_identifiable_lower_bound_never_bank_membership"
        )
        assert "owner_competition" in row
        assert "strict_table_competition" in row


# ---------------------------------------------------------------------------
# Loop-tail exclusion and the frozen minimal-frontier tie rule
# ---------------------------------------------------------------------------


def test_loop_tail_contexts_are_excluded_from_the_primary_views(tmp_path: Path) -> None:
    image_id = IMAGE_IDS[0]
    overrides = {
        image_id: ImageSpec(
            owners=(OwnerSpec(owner_id="own-loop", box=FULL_BANK_BOX),),
            contexts=(
                ContextSpec(boundary_index=1, frontier_box=(10, 10, 30, 30)),
                ContextSpec(
                    boundary_index=2,
                    frontier_box=(12, 12, 32, 32),
                    loop_tail=True,
                    consecutive_run=4,
                ),
            ),
        )
    }

    def loop_scores(group: Mapping[str, object], candidate: Mapping[str, object]) -> float:
        # Make the loop-tail context carry the strongest score in the census.
        boost = 10.0 if str(group["context_id"]).endswith("boundary-002") else 0.0
        return boost - float(candidate["representative_role_ordinal"])

    merged = build_merged(
        tmp_path, overrides=overrides, options=ShardOptions(score_fn=loop_scores)
    )
    summary = next(
        row for row in merged.owner_summaries if row["gt_owner_id"] == "own-loop"
    )
    upper = summary["upper_bound_u"]

    assert upper["diagnostic_best_all"]["loop_tail"] is True
    assert upper["diagnostic_best_all_role"] == (
        "diagnostic_only_includes_loop_tail_contexts"
    )
    assert upper["primary_best_non_loop"]["loop_tail"] is False
    assert upper["primary_best_non_loop"]["boundary_index"] == 1
    assert upper["primary_first_non_loop_minimal_abs_frontier"]["loop_tail"] is False
    assert upper["non_loop_tested_context_count"] == 1
    assert upper["tested_context_count"] == 2


def test_minimal_frontier_tie_takes_the_first_non_loop_boundary(tmp_path: Path) -> None:
    image_id = IMAGE_IDS[0]
    # Two non-loop contexts whose frontiers sit symmetrically around the owner,
    # so both attain the same absolute signed ordinal distance.
    overrides = {
        image_id: ImageSpec(
            owners=(
                OwnerSpec(owner_id="own-a", box=(20, 20, 60, 60)),
                OwnerSpec(owner_id="own-b", box=(120, 120, 160, 160)),
            ),
            contexts=(
                ContextSpec(boundary_index=1, frontier_box=(10, 10, 30, 30)),
                ContextSpec(boundary_index=2, frontier_box=(10, 10, 30, 30)),
                ContextSpec(
                    boundary_index=3,
                    frontier_box=(10, 10, 30, 30),
                    loop_tail=True,
                    consecutive_run=5,
                ),
            ),
        )
    }
    merged = build_merged(tmp_path, overrides=overrides)
    summary = next(row for row in merged.owner_summaries if row["gt_owner_id"] == "own-a")
    chosen = summary["upper_bound_u"]["primary_first_non_loop_minimal_abs_frontier"]

    assert chosen["boundary_index"] == 1, "the tie must resolve to the first boundary"
    assert chosen["loop_tail"] is False
    assert summary["upper_bound_u"]["minimal_frontier_tie_rule"] == (
        planner.MINIMAL_FRONTIER_TIE_RULE
    )


# ---------------------------------------------------------------------------
# Dispositions: protections and the persistent-negative preconditions
# ---------------------------------------------------------------------------


def _summary(merged: merge.MergedCensus, owner_id: str) -> dict:
    return next(row for row in merged.owner_summaries if row["gt_owner_id"] == owner_id)


def test_never_frontier_tested_owner_cannot_close_persistent_negative(
    tmp_path: Path,
) -> None:
    image_id = IMAGE_IDS[0]
    overrides = {
        image_id: ImageSpec(
            owners=(
                OwnerSpec(owner_id="own-strong", box=FULL_BANK_BOX),
                OwnerSpec(owner_id="own-weak", box=SECOND_FULL_BANK_BOX),
            ),
            # Root only: no context in this image has a frontier.
            contexts=(ContextSpec(boundary_index=0),),
        )
    }

    def weak_last(group: Mapping[str, object], candidate: Mapping[str, object]) -> float:
        owners = candidate["generator_gt_owner_ids"]
        penalty = -50.0 if "own-weak" in owners else 0.0
        return penalty - float(candidate["representative_role_ordinal"])

    merged = build_merged(
        tmp_path,
        overrides=overrides,
        options=ShardOptions(score_fn=weak_last),
        calibration=fixed_calibration(),
    )
    weak = _summary(merged, "own-weak")

    assert weak["never_frontier_tested"] is True
    assert weak["frontier_tested"] is False
    assert weak["upper_bound_u"]["usable_support"] is False
    assert weak["disposition"] != merge.DISPOSITION_PERSISTENT_NEGATIVE
    assert weak["disposition"] == merge.DISPOSITION_UNRESOLVED
    assert "never_frontier_tested" in weak["disposition_blockers"]
    assert weak["persistent_negative_preconditions"]["frontier_tested"] is False
    assert weak["disposition_semantics"]["is_causal_label"] is False


def test_frontier_tested_owner_without_support_closes_persistent_negative(
    tmp_path: Path,
) -> None:
    image_id = IMAGE_IDS[0]
    overrides = {
        image_id: ImageSpec(
            owners=(
                OwnerSpec(owner_id="own-strong", box=FULL_BANK_BOX),
                OwnerSpec(owner_id="own-weak", box=SECOND_FULL_BANK_BOX),
            ),
            contexts=(ContextSpec(boundary_index=1, frontier_box=(10, 10, 30, 30)),),
        )
    }

    def weak_last(group: Mapping[str, object], candidate: Mapping[str, object]) -> float:
        owners = candidate["generator_gt_owner_ids"]
        penalty = -50.0 if "own-weak" in owners else 0.0
        return penalty - float(candidate["representative_role_ordinal"])

    merged = build_merged(
        tmp_path,
        overrides=overrides,
        options=ShardOptions(score_fn=weak_last),
        calibration=fixed_calibration(),
    )
    weak = _summary(merged, "own-weak")
    strong = _summary(merged, "own-strong")

    assert weak["frontier_tested"] is True
    assert weak["bank_adequacy"]["adequate"] is True
    assert weak["exact_anchor_score_reported"] is True
    assert weak["disposition_blockers"] == []
    assert weak["disposition"] == merge.DISPOSITION_PERSISTENT_NEGATIVE
    assert weak["persistent_negative_preconditions"][
        "no_usable_support_in_any_non_loop_context_under_u"
    ] is True
    assert strong["disposition"] == merge.DISPOSITION_RESOLVED


def test_bank_undercoverage_floors_an_owner_to_unresolved(tmp_path: Path) -> None:
    """A twin pair fails the mandatory exact-anchor self-localization gate."""

    image_id = IMAGE_IDS[0]
    merged = build_merged(tmp_path, overrides=_twin_overrides(image_id))
    twin = _summary(merged, "own-twin-a")

    assert twin["bank_adequacy"]["exact_anchor_uniquely_self_assigned"] is False
    assert twin["bank_adequacy"]["status"] == "undercovered_unresolved_only"
    assert twin["bank_adequacy"]["adequate"] is False
    assert "bank_undercoverage" in twin["disposition_blockers"]
    assert twin["disposition"] != merge.DISPOSITION_PERSISTENT_NEGATIVE
    assert twin["disposition_semantics"]["undercovered_owner_floor"] == (
        "unresolved_only_never_persistent_no_tested_localization_support"
    )


def test_adequate_reduced_bank_stays_eligible_and_flagged(tmp_path: Path) -> None:
    """The reduced band is flagged, never floored: it must remain dispositionable."""

    image_id = IMAGE_IDS[0]
    overrides = {
        image_id: ImageSpec(
            owners=(OwnerSpec(owner_id="own-reduced", box=REDUCED_BANK_BOX),),
            contexts=(ContextSpec(boundary_index=1, frontier_box=(150, 150, 180, 180)),),
        )
    }
    merged = build_merged(
        tmp_path, overrides=overrides, calibration=fixed_calibration()
    )
    reduced = _summary(merged, "own-reduced")
    adequacy = reduced["bank_adequacy"]

    assert adequacy["status"] == "adequate_reduced"
    assert adequacy["distinct_physical_candidate_count"] < adequacy["full_at_least"]
    assert adequacy["distinct_physical_candidate_count"] >= adequacy["adequate_at_least"]
    assert adequacy["adequate"] is True
    assert adequacy["flagged_reduced"] is True
    assert adequacy["threshold_source"] == (
        "sealed_capture_rules_owner_support_bank_adequacy_rule"
    )
    assert "bank_undercoverage" not in reduced["disposition_blockers"]
    assert reduced["disposition"] == merge.DISPOSITION_RESOLVED


def test_owner_report_carries_the_sealed_required_counts(tmp_path: Path) -> None:
    merged = build_merged(tmp_path)
    required = merged.plan.capture_rules["owner_support"]["required_per_owner_report_fields"]
    for row in merged.owner_summaries:
        for name in required:
            assert name in row["bank_report"], f"{name} missing from owner bank report"


# ---------------------------------------------------------------------------
# Split separation and the two-phase rule contract
# ---------------------------------------------------------------------------


def test_discovery_and_confirmation_owners_stay_separated(tmp_path: Path) -> None:
    merged = build_merged(tmp_path)
    discovery = merged.summaries_for_split("discovery")
    confirmation = merged.summaries_for_split("confirmation")

    assert discovery and confirmation
    assert {row["image_id"] for row in discovery} <= set(planner.DISCOVERY_IMAGE_IDS)
    assert {row["image_id"] for row in confirmation} <= set(planner.CONFIRMATION_IMAGE_IDS)
    assert not (
        {row["image_id"] for row in discovery} & {row["image_id"] for row in confirmation}
    )
    assert merged.receipt["counts"]["discovery_owner_summary_count"] == len(discovery)
    assert merged.receipt["counts"]["confirmation_owner_summary_count"] == len(confirmation)


def test_discovery_description_is_diagnostic_only_and_carries_no_cut(
    tmp_path: Path,
) -> None:
    merged = build_merged(tmp_path)
    described = merge.describe_discovery_distributions(merged)

    assert described["split"] == "discovery"
    assert described["is_a_rule"] is False
    assert described["decision_bearing"] is False
    assert described["role"] == "diagnostic_suggestion_only_never_a_phenotype_cut"

    # It reports distributions and nothing decision-bearing.  Asserted on the
    # emitted key structure rather than a raw substring search, because the unit
    # id itself legitimately contains the word "phenotype".
    def _keys(value: object) -> set[str]:
        if isinstance(value, dict):
            return set(value) | {
                key for item in value.values() for key in _keys(item)
            }
        if isinstance(value, list):
            return {key for item in value for key in _keys(item)}
        return set()

    emitted = _keys(described)
    forbidden = {
        "threshold",
        "thresholds",
        "conditions",
        "phenotype",
        "phenotype_counts",
        "phenotype_when_all_conditions_hold",
        "assignments",
        "discovery_rule_sha256",
    }
    assert not (emitted & forbidden), sorted(emitted & forbidden)
    assert set(described["strata"]) == {
        "all",
        "native_true_positive",
        "native_false_negative",
    }
    # A distribution description cannot be used as a rule.
    with pytest.raises(merge.MergeContractError):
        merge.apply_confirmation(
            merged, described, expect_discovery_rule_sha256="whatever"
        )


def _rule_spec(**overrides: object) -> dict:
    spec: dict = {
        "rule_id": "test-rule",
        "bound": "u",
        "view": "primary_best_non_loop",
        "conditions": [
            {
                "feature": "margin_to_best_owner_in_group",
                "direction": "at_least",
                "threshold": -0.5,
                "rationale": "owner is at or near the group best",
            }
        ],
        "phenotype_when_all_conditions_hold": "tested_localization_support",
        "phenotype_otherwise": "no_tested_localization_support_under_this_rule",
        "provenance": {
            "constructed_after_inspecting": "discovery owner-summary distributions",
            "author": "test",
            "causal_claim_asserted": False,
        },
    }
    spec.update(overrides)
    return spec


def test_sealed_rule_is_explicit_and_derives_no_threshold(tmp_path: Path) -> None:
    merged = build_merged(tmp_path)
    rule = merge.seal_discovery_rule(_rule_spec(), merged)

    assert rule["authored"] == "explicit_post_discovery_never_auto_derived"
    assert rule["phase"] == "discovery"
    assert rule["phenotype_labels_are_causal"] is False
    assert rule["discovery_image_ids_frozen"] == list(planner.DISCOVERY_IMAGE_IDS)
    assert set(rule["discovery_image_ids_used"]) <= set(planner.DISCOVERY_IMAGE_IDS)
    # The threshold is exactly what was authored, never re-derived.
    assert rule["conditions"][0]["threshold"] == pytest.approx(-0.5)
    assert rule["conditions"][0]["feature_definition"]
    assert rule["discovery_rule_sha256"] == merge.sha256_json(
        {key: value for key, value in rule.items() if key != "discovery_rule_sha256"}
    )


def test_sealed_rule_rejects_unpublished_features_and_missing_provenance(
    tmp_path: Path,
) -> None:
    merged = build_merged(tmp_path)

    with pytest.raises(merge.MergeContractError, match="does not publish"):
        merge.seal_discovery_rule(
            _rule_spec(
                conditions=[
                    {"feature": "vibes", "direction": "at_least", "threshold": 0.0}
                ]
            ),
            merged,
        )
    with pytest.raises(merge.MergeContractError, match="provenance"):
        merge.seal_discovery_rule(_rule_spec(provenance={}), merged)
    with pytest.raises(merge.MergeContractError, match="causal claim"):
        merge.seal_discovery_rule(
            _rule_spec(
                provenance={
                    "constructed_after_inspecting": "distributions",
                    "causal_claim_asserted": True,
                }
            ),
            merged,
        )
    with pytest.raises(merge.MergeContractError, match="direction"):
        merge.seal_discovery_rule(
            _rule_spec(
                conditions=[
                    {
                        "feature": "margin_to_best_owner_in_group",
                        "direction": "roughly",
                        "threshold": 0.0,
                    }
                ]
            ),
            merged,
        )


def test_confirmation_binds_the_exact_rule_digest_and_uses_held_out_images(
    tmp_path: Path,
) -> None:
    merged = build_merged(tmp_path)
    rule = merge.seal_discovery_rule(_rule_spec(), merged)
    report = merge.apply_confirmation(
        merged, rule, expect_discovery_rule_sha256=rule["discovery_rule_sha256"]
    )

    assert report["bound_discovery_rule_sha256"] == rule["discovery_rule_sha256"]
    assert report["rule_retuned"] is False
    assert report["conditions_applied"] == rule["conditions"]
    assert set(report["confirmation_image_ids_used"]) <= set(
        planner.CONFIRMATION_IMAGE_IDS
    )
    assert not set(report["confirmation_image_ids_used"]) & set(
        planner.DISCOVERY_IMAGE_IDS
    )
    assert all(
        row["split"] == "confirmation" for row in report["assignments"]
    )
    assert report["phenotype_labels_are_causal"] is False


def test_confirmation_refuses_every_retuning_route(tmp_path: Path) -> None:
    merged = build_merged(tmp_path)
    rule = merge.seal_discovery_rule(_rule_spec(), merged)
    digest = rule["discovery_rule_sha256"]

    # 1. An explicit override request.
    with pytest.raises(merge.MergeContractError, match="may not retune"):
        merge.apply_confirmation(
            merged,
            rule,
            expect_discovery_rule_sha256=digest,
            overrides={"margin_to_best_owner_in_group": 0.0},
        )

    # 2. A threshold edited after sealing: the digest no longer reconstructs.
    edited = copy.deepcopy(rule)
    edited["conditions"][0]["threshold"] = 99.0
    with pytest.raises(merge.MergeContractError, match="edited after sealing"):
        merge.apply_confirmation(
            merged, edited, expect_discovery_rule_sha256=edited["discovery_rule_sha256"]
        )

    # 3. A re-sealed rule that no longer matches the digest the caller expects.
    resealed = merge.seal_discovery_rule(
        _rule_spec(
            conditions=[
                {
                    "feature": "margin_to_best_owner_in_group",
                    "direction": "at_least",
                    "threshold": 99.0,
                }
            ]
        ),
        merged,
    )
    with pytest.raises(merge.MergeContractError, match="was asked to bind"):
        merge.apply_confirmation(
            merged, resealed, expect_discovery_rule_sha256=digest
        )

    # 4. No expected digest named at all.
    with pytest.raises(merge.MergeContractError, match="named explicitly"):
        merge.apply_confirmation(merged, rule, expect_discovery_rule_sha256="")


def test_feature_catalog_names_the_true_margin_and_the_cross_context_delta(
    tmp_path: Path,
) -> None:
    """A cross-context delta must never be published as a competition margin."""

    merged = build_merged(tmp_path)
    summary = merged.owner_summaries[0]
    vector = merge.owner_feature_vector(summary, bound="u")
    assert vector is not None

    assert "margin_to_best_owner_in_group" in merge.OWNER_FEATURE_CATALOG
    assert "cross_context_delta_primary_best_vs_minimal_frontier" in vector
    assert (
        "NOT a competition margin"
        in merge.OWNER_FEATURE_CATALOG[
            "cross_context_delta_primary_best_vs_minimal_frontier"
        ]
    )
    # The true margin comes from a single (context, category) rank population.
    projected = summary["upper_bound_u"]["primary_best_non_loop"]
    assert projected["margin_semantics"] == (
        "within_same_context_and_category_owner_competition"
    )
    assert projected["margin_to_best_owner_in_group"] is not None


# ---------------------------------------------------------------------------
# Receipt and commit
# ---------------------------------------------------------------------------


def test_merge_receipt_reconstructs_and_commit_is_create_or_identical(
    tmp_path: Path,
) -> None:
    merged = build_merged(tmp_path)
    receipt = merged.receipt
    assert receipt["receipt_content_sha256"] == merge.sha256_json(
        {key: value for key, value in receipt.items() if key != "receipt_content_sha256"}
    )
    assert receipt["causal_labels"]["emitted"] is False
    assert receipt["rank_contract"]["duplicate_event_policy"] == "fail_closed"
    assert receipt["quarantined_score_semantics"]["reads_quarantined_shard_evidence"] is False

    out = tmp_path / "merged"
    first = merge.commit_merge(merged, out)
    second = merge.commit_merge(merged, out)
    assert first == second

    (out / merge.EVENTS_NAME).write_bytes(b'{"tampered":true}\n')
    with pytest.raises(merge.MergeContractError, match="refusing to overwrite"):
        merge.commit_merge(merged, out)


# ---------------------------------------------------------------------------
# Local-peak support: exact formulas, mixed-pass cases, and bound partitions
# ---------------------------------------------------------------------------


def _support_block(row: Mapping[str, object], bound: str) -> dict:
    key = "ambiguity_excluded_l" if bound == "l" else "ambiguity_included_u"
    return row["localization"]["generator_local_max_excluding_other_owner_strict"][key]


def test_peak_lift_and_local_concentration_match_the_ruled_formulas(
    tmp_path: Path,
) -> None:
    """``peak_lift = best - logsumexp(group) + log(N)``; concentration = best - median(own bank)."""

    merged = build_merged(tmp_path)
    events_by_group: dict[tuple[str, str, str], list[dict]] = {}
    for row in merged.events:
        events_by_group.setdefault(
            (row["image_id"], row["context_id"], row["normalized_description"]), []
        ).append(row)

    checked = 0
    for row in merged.owner_contexts:
        group = events_by_group[
            (row["image_id"], row["context_id"], row["normalized_description"])
        ]
        group_scores = [float(item["complete_box_logprob_sum"]) for item in group]
        population = len(group_scores)
        peak = max(group_scores)
        # Recomputed the plain way, independent of the module's log-softmax.
        logsumexp = peak + math.log(sum(math.exp(value - peak) for value in group_scores))
        for bound in ("l", "u"):
            block = _support_block(row, bound)
            if block["value"] is None:
                continue
            best = float(block["value"])
            assert block["peak_lift"] == pytest.approx(
                best - logsumexp + math.log(population)
            )
            assert block["unique_population_size"] == population
            assert block["local_concentration"] == pytest.approx(
                best - float(block["bank_median"])
            )
            checked += 1
    assert checked, "no owner-context carried a support statistic"


def test_support_requires_both_statistics_mixed_pass_is_not_support(
    tmp_path: Path,
) -> None:
    """One statistic clearing its threshold is never support on its own."""

    merged = build_merged(tmp_path)
    row = next(
        row
        for row in merged.owner_contexts
        if _support_block(row, "u")["value"] is not None
    )
    block = _support_block(row, "u")
    lift = float(block["peak_lift"])
    concentration = float(block["local_concentration"])
    epsilon = float(
        planner.build_capture_rules()["owner_support"]["epsilons"]["support_epsilon"]
    )

    both = fixed_calibration(theta_lift=lift - 1.0, theta_conc=concentration - 1.0)
    lift_only = fixed_calibration(theta_lift=lift - 1.0, theta_conc=concentration + 1.0)
    conc_only = fixed_calibration(theta_lift=lift + 1.0, theta_conc=concentration - 1.0)
    neither = fixed_calibration(theta_lift=lift + 1.0, theta_conc=concentration + 1.0)

    assert both.clears(block) is True
    assert lift_only.clears(block) is False
    assert conc_only.clears(block) is False
    assert neither.clears(block) is False

    # The epsilon guard is applied to both statistics, on the strict side.
    exactly_at = fixed_calibration(theta_lift=lift, theta_conc=concentration)
    assert exactly_at.clears(block) is False, "epsilon guard was not applied"
    just_under = fixed_calibration(
        theta_lift=lift - epsilon, theta_conc=concentration - epsilon
    )
    assert just_under.clears(block) is True


def test_strong_localized_rank_two_owner_is_supported(tmp_path: Path) -> None:
    """Support is local-peak evidence, so a rank-2 owner with a real peak passes.

    This is the case the rejected rank==1 criterion got wrong: owner B is
    routed second within the group, yet its own landscape carries a sharp peak
    at its geometry.
    """

    image_id = IMAGE_IDS[0]
    overrides = {
        image_id: ImageSpec(
            owners=(
                OwnerSpec(owner_id="own-a", box=SEPARATED_BOX_A),
                OwnerSpec(owner_id="own-b", box=SEPARATED_BOX_B),
            ),
            contexts=(ContextSpec(boundary_index=1, frontier_box=(90, 90, 110, 110)),),
        )
    }

    def sharp_peaks(group: Mapping[str, object], candidate: Mapping[str, object]) -> float:
        # Each owner's exact anchor is a sharp peak; everything else is flat and
        # far below.  Owner B's peak is slightly under owner A's, so B ranks 2.
        owners = candidate["generator_gt_owner_ids"]
        if candidate["representative_role"] != "exact_gt_anchor":
            return -20.0
        return 0.0 if "own-a" in owners else -0.5

    merged = build_merged(
        tmp_path,
        overrides=overrides,
        options=ShardOptions(score_fn=sharp_peaks),
        calibration=fixed_calibration(theta_lift=0.0, theta_conc=1.0),
    )
    rows = {
        row["gt_owner_id"]: row
        for row in merged.owner_contexts
        if row["image_id"] == image_id
    }
    b_row = rows["own-b"]
    b_block = _support_block(b_row, "u")

    # Owner B is genuinely not the group winner...
    assert int(b_block["rank"]) > 1
    assert b_block["rank_role"] if "rank_role" in b_block else True
    assert b_row["owner_competition_u"]["rank"] == 2
    # ...yet it carries a real local peak and is supported.
    assert float(b_block["peak_lift"]) > 0.0
    assert float(b_block["local_concentration"]) > 1.0

    summary = _summary(merged, "own-b")
    assert summary["upper_bound_u"]["usable_support"] is True
    assert summary["disposition"] != merge.DISPOSITION_PERSISTENT_NEGATIVE
    assert summary["routing_summary"]["ever_rank1"] is False
    assert summary["routing_summary"]["role"] == (
        "routing_and_competition_surface_never_a_support_input"
    )


def test_unmatched_candidate_is_u_only_and_cannot_create_l_support(
    tmp_path: Path,
) -> None:
    """A high-scoring unmatched probe may never manufacture conservative support.

    The owner's strongest candidate is one of its *unmatched* perturbations,
    which the sealed partition admits under U only.  It must not enter L, and
    the resulting bound disagreement must close the owner unresolved rather
    than resolved.
    """

    image_id = IMAGE_IDS[0]
    overrides = {
        image_id: ImageSpec(
            owners=(OwnerSpec(owner_id="own-solo", box=FULL_BANK_BOX),),
            contexts=(ContextSpec(boundary_index=1, frontier_box=(10, 10, 30, 30)),),
        )
    }

    def unmatched_wins(group: Mapping[str, object], candidate: Mapping[str, object]) -> float:
        # Reward exactly the candidates that landed on no owner.
        if candidate["strict_assignment_status"] == "matched":
            return -30.0
        return 0.0

    merged = build_merged(
        tmp_path,
        overrides=overrides,
        options=ShardOptions(score_fn=unmatched_wins),
        calibration=fixed_calibration(theta_lift=0.0, theta_conc=-1.0),
    )
    row = next(r for r in merged.owner_contexts if r["gt_owner_id"] == "own-solo")
    lower = _support_block(row, "l")
    upper = _support_block(row, "u")

    assert row["localization"]["counts"]["unassigned_event_count"] > 0
    assert row["localization"]["support_partition"]["unmatched_generator_local"] == (
        "counts_under_u_only"
    )
    # The unmatched peak is visible under U and absent from L.
    assert float(upper["value"]) > float(lower["value"])
    assert float(upper["peak_lift"]) > float(lower["peak_lift"])
    # The whole L population is strictly-assigned-self only.
    assert lower["count"] == row["localization"]["counts"]["strict_assigned_event_count"]

    summary = _summary(merged, "own-solo")
    assert summary["upper_bound_u"]["usable_support"] is True
    assert summary["lower_bound_l"]["usable_support"] is False
    assert summary["ambiguity_bound_disposition_flip"] is True
    assert summary["disposition"] == merge.DISPOSITION_UNRESOLVED_FLIP


def test_ambiguity_bound_negative_invariance(tmp_path: Path) -> None:
    """An owner with no support under optimistic U has none under L either.

    L is a subset of U, so a negative close cannot depend on which bound is
    read; the persistent-negative gate is therefore bound-invariant.
    """

    image_id = IMAGE_IDS[0]
    overrides = {
        image_id: ImageSpec(
            owners=(
                OwnerSpec(owner_id="own-strong", box=FULL_BANK_BOX),
                OwnerSpec(owner_id="own-weak", box=SECOND_FULL_BANK_BOX),
            ),
            contexts=(ContextSpec(boundary_index=1, frontier_box=(10, 10, 30, 30)),),
        )
    }

    def weak_last(group: Mapping[str, object], candidate: Mapping[str, object]) -> float:
        owners = candidate["generator_gt_owner_ids"]
        penalty = -50.0 if "own-weak" in owners else 0.0
        return penalty - float(candidate["representative_role_ordinal"])

    merged = build_merged(
        tmp_path,
        overrides=overrides,
        options=ShardOptions(score_fn=weak_last),
        calibration=fixed_calibration(),
    )
    weak = _summary(merged, "own-weak")

    assert weak["upper_bound_u"]["usable_support"] is False
    assert weak["lower_bound_l"]["usable_support"] is False
    assert weak["ambiguity_bound_disposition_flip"] is False
    assert weak["disposition"] == merge.DISPOSITION_PERSISTENT_NEGATIVE
    assert weak["persistent_negative_preconditions"][
        "no_usable_support_in_any_non_loop_context_under_u"
    ] is True
    assert weak["persistent_negative_preconditions"]["optimistic_bound"] == "u"


def test_nothing_closes_without_a_sealed_calibration(tmp_path: Path) -> None:
    merged = build_merged(tmp_path)
    for row in merged.owner_summaries:
        assert row["disposition"] == merge.DISPOSITION_UNCALIBRATED
        assert row["upper_bound_u"]["usable_support"] is None
        assert row["upper_bound_u"]["support_calibrated"] is False
    assert merged.receipt["dispositions_closed"] is False


def test_support_constants_are_bound_from_the_sealed_capture_rules(
    tmp_path: Path,
) -> None:
    plan = merge.load_plan(build_plan_dir(tmp_path))
    contract = merge.load_support_contract(plan)
    sealed = plan.capture_rules["owner_support"]

    assert contract.support_epsilon == sealed["epsilons"]["support_epsilon"]
    assert contract.cross_context_delta_epsilon == (
        sealed["epsilons"]["cross_context_delta_epsilon"]
    )
    assert contract.primary_quantile == sealed["support_calibration"]["primary_quantile"]
    assert set(contract.statistics) == set(merge.SUPPORT_FEATURE_NAMES)
    assert sealed["support_definition"]["rank_is_support_criterion"] is False

    stripped = {k: v for k, v in plan.capture_rules.items() if k != "owner_support"}
    crippled = merge.PlanBundle(
        plan_dir=plan.plan_dir,
        receipt=plan.receipt,
        capture_rules=stripped,
        images=plan.images,
        owners=plan.owners,
        categories=plan.categories,
        contexts=plan.contexts,
        candidates=plan.candidates,
        query_groups=plan.query_groups,
        native_sidecars=plan.native_sidecars,
        shards=plan.shards,
    )
    with pytest.raises(merge.MergeContractError, match="will not substitute"):
        merge.load_support_contract(crippled)


# ---------------------------------------------------------------------------
# Blind-analysis phase DAG
# ---------------------------------------------------------------------------


def test_capture_manifest_hashes_every_shard_without_parsing_a_score_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Phase A must be metadata-only.

    ``_read_jsonl`` is the single parsing seam for score, proposal and
    free-decode files, so hard-failing it proves the manifest never interprets
    evidence -- while the manifest still hashes every one of those files.
    """

    plan_dir = build_plan_dir(tmp_path)
    plan = merge.load_plan(plan_dir)
    write_shards(plan, tmp_path / "shards")

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("the capture-manifest phase parsed an evidence file")

    monkeypatch.setattr(merge, "_read_jsonl", forbidden)
    manifest = merge.build_capture_manifest(plan, tmp_path / "shards")

    assert manifest["reads_score_rows"] is False
    assert manifest["score_derived_interpretation_performed"] is False
    assert manifest["quarantine_dispositions_frozen"] is True
    assert len(manifest["images"]) == len(IMAGE_IDS)

    # Every shard file is nonetheless content-addressed.
    for entry in manifest["images"]:
        for name in (
            merge.SHARD_RECEIPT_NAME,
            merge.SHARD_SCORES_NAME,
            merge.SHARD_PROPOSAL_NAME,
            merge.SHARD_FREE_DECODE_NAME,
        ):
            digest = entry["file_digests"].get(name)
            assert digest, f"{entry['image_id']} did not hash {name}"
            observed = merge.sha256_bytes(
                (tmp_path / "shards" / entry["image_id"] / name).read_bytes()
            )
            assert digest == observed
        assert entry["lineage"]["evidence_loaded"] is False
        assert entry["lineage"]["score_row_count"] is None

    assert manifest["capture_manifest_sha256"] == merge.sha256_json(
        {k: v for k, v in manifest.items() if k != "capture_manifest_sha256"}
    )
    assert set(manifest["discovery_shard_digests"]).isdisjoint(
        manifest["confirmation_shard_digests"]
    )


def test_discovery_phase_never_opens_a_confirmation_shard_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The blind boundary is mechanical: the files are never opened.

    Every filesystem read in this module goes through ``Path.read_bytes`` or
    ``Path.read_text``; both are recorded here, and no recorded path may live
    under a confirmation shard directory.
    """

    plan_dir = build_plan_dir(tmp_path)
    plan = merge.load_plan(plan_dir)
    shard_root = tmp_path / "shards"
    write_shards(plan, shard_root)
    manifest = merge.build_capture_manifest(plan, shard_root)

    touched: list[Path] = []
    real_bytes = Path.read_bytes
    real_text = Path.read_text

    def record_bytes(self: Path, *args: object, **kwargs: object) -> bytes:
        touched.append(self)
        return real_bytes(self, *args, **kwargs)

    def record_text(self: Path, *args: object, **kwargs: object) -> str:
        touched.append(self)
        return real_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", record_bytes)
    monkeypatch.setattr(Path, "read_text", record_text)

    merged = merge.merge_census(
        plan,
        shard_root,
        manifest=manifest,
        allowlist=merge._allowlist_for_phase(merge.PHASE_DISCOVERY),
        phase=merge.PHASE_DISCOVERY,
    )

    confirmation_dirs = {
        (shard_root / image_id).resolve() for image_id in planner.CONFIRMATION_IMAGE_IDS
    }
    leaked = [
        path
        for path in touched
        if path.resolve().parent in confirmation_dirs
    ]
    assert not leaked, f"discovery opened confirmation files: {leaked}"

    assert merged.receipt["phase"] == merge.PHASE_DISCOVERY
    assert set(merged.receipt["image_allowlist"]) == set(planner.DISCOVERY_IMAGE_IDS)
    assert not merged.summaries_for_split("confirmation")
    assert merged.summaries_for_split("discovery")


def test_discovery_phase_fails_closed_on_a_confirmation_digest(tmp_path: Path) -> None:
    plan_dir = build_plan_dir(tmp_path)
    plan = merge.load_plan(plan_dir)
    shard_root = tmp_path / "shards"
    write_shards(plan, shard_root)
    manifest = merge.build_capture_manifest(plan, shard_root)

    # Assert every discovery digest into the confirmation set: the allowlist
    # still permits the files, so only the digest proof can catch this.
    poisoned = dict(manifest)
    poisoned["confirmation_shard_digests"] = sorted(
        set(manifest["confirmation_shard_digests"]) | set(manifest["discovery_shard_digests"])
    )
    # Re-seal so the self-digest and byte-binding checks pass and the only
    # thing left to fail is split disjointness.
    poisoned.pop("capture_manifest_sha256")
    poisoned["capture_manifest_sha256"] = merge.sha256_json(poisoned)
    with pytest.raises(merge.MergeContractError, match="blind-analysis boundary is broken"):
        merge.merge_census(
            plan,
            shard_root,
            manifest=poisoned,
            allowlist=merge._allowlist_for_phase(merge.PHASE_DISCOVERY),
            phase=merge.PHASE_DISCOVERY,
        )


def test_calibration_is_discovery_only_and_seals_its_own_digest(tmp_path: Path) -> None:
    image_id = planner.DISCOVERY_IMAGE_IDS[0]
    overrides = {
        image_id: ImageSpec(
            owners=(
                OwnerSpec(
                    owner_id="own-tp", box=SEPARATED_BOX_A, native_true_positive=True
                ),
                OwnerSpec(owner_id="own-fn", box=SEPARATED_BOX_B),
            ),
            contexts=(
                ContextSpec(boundary_index=0),
                ContextSpec(boundary_index=1, frontier_box=(90, 90, 110, 110)),
            ),
        )
    }
    plan_dir = build_plan_dir(tmp_path, overrides)
    plan = merge.load_plan(plan_dir)
    shard_root = tmp_path / "shards"
    write_shards(plan, shard_root)
    manifest = merge.build_capture_manifest(plan, shard_root)

    merged = merge.merge_census(
        plan,
        shard_root,
        manifest=manifest,
        allowlist=merge._allowlist_for_phase(merge.PHASE_DISCOVERY),
        phase=merge.PHASE_DISCOVERY,
    )
    calibration = merge.calibrate_support(merged, manifest)
    receipt = calibration.describe()

    assert receipt["calibration_stratum"] == "pooled_discovery_native_true_positives"
    assert receipt["quantile"] == plan.capture_rules["owner_support"][
        "support_calibration"
    ]["primary_quantile"]
    assert receipt["epsilon"] == plan.capture_rules["owner_support"]["epsilons"][
        "support_epsilon"
    ]
    assert receipt["confirmation_evidence_consumed"] is False
    assert set(receipt["consumed_shard_digests"]).isdisjoint(
        manifest["confirmation_shard_digests"]
    )
    assert receipt["category_stratification_role"] == (
        "report_only_sensitivity_never_a_threshold"
    )
    assert set(receipt["sensitivity"]["quantiles"]) == {
        str(level)
        for level in plan.capture_rules["owner_support"]["support_calibration"][
            "sensitivity_quantiles"
        ]
    }
    # Round-trips through its sealed digest.
    rebuilt = merge.calibration_from_receipt(receipt)
    assert rebuilt.theta_peak_lift == calibration.theta_peak_lift
    assert rebuilt.theta_local_concentration == calibration.theta_local_concentration

    edited = dict(receipt)
    edited["theta_peak_lift"] = 99.0
    with pytest.raises(merge.MergeContractError, match="does not reconstruct"):
        merge.calibration_from_receipt(edited)


def test_calibration_refuses_a_non_discovery_merge(tmp_path: Path) -> None:
    plan_dir = build_plan_dir(tmp_path)
    plan = merge.load_plan(plan_dir)
    shard_root = tmp_path / "shards"
    write_shards(plan, shard_root)
    manifest = merge.build_capture_manifest(plan, shard_root)
    full = merge.merge_census(plan, shard_root, manifest=manifest, phase=merge.PHASE_FULL)
    with pytest.raises(merge.MergeContractError, match="discovery-phase merge"):
        merge.calibrate_support(full, manifest)


def test_due_context_comes_from_the_native_row_index_not_from_scores(
    tmp_path: Path,
) -> None:
    image_id = planner.DISCOVERY_IMAGE_IDS[0]
    overrides = {
        image_id: ImageSpec(
            owners=(
                OwnerSpec(
                    owner_id="own-tp", box=SEPARATED_BOX_A, native_true_positive=True
                ),
            ),
            contexts=(
                ContextSpec(boundary_index=0),
                ContextSpec(boundary_index=1, frontier_box=(90, 90, 110, 110)),
            ),
        )
    }
    plan = merge.load_plan(build_plan_dir(tmp_path, overrides))
    due = merge.due_context_index(plan)

    entry = due["own-tp"]
    assert entry["excluded"] is False
    # The fixture's sidecar maps pred_row_id -> row_index 0, so the due context
    # is the boundary before that row.
    assert entry["row_index"] == 0
    assert entry["due_context_id"] == f"{image_id}:boundary-000"
    assert entry["due_context_id"] in plan.contexts
    # Only true positives are mapped at all.
    assert all(
        plan.owners[owner_id]["native_true_positive"] for owner_id in due
    )


def test_smoke_admit_is_single_image_and_never_a_census_conclusion(
    tmp_path: Path,
) -> None:
    plan_dir = build_plan_dir(tmp_path)
    plan = merge.load_plan(plan_dir)
    shard_root = tmp_path / "shards"
    # Only the representative image exists: the launch gate runs *before* the
    # other eleven shards are captured.
    write_shards(
        plan,
        shard_root,
        ShardOptions(
            omitted=frozenset(set(IMAGE_IDS) - {merge.SMOKE_IMAGE_ID}),
        ),
    )
    merged = merge.merge_census(
        plan,
        shard_root,
        allowlist=merge._allowlist_for_phase(merge.PHASE_SMOKE_ADMIT),
        phase=merge.PHASE_SMOKE_ADMIT,
        usable_as_census_conclusion=False,
    )

    assert merge.SMOKE_IMAGE_ID in planner.DISCOVERY_IMAGE_IDS
    assert merged.receipt["phase"] == merge.PHASE_SMOKE_ADMIT
    assert merged.receipt["usable_as_census_conclusion"] is False
    assert merged.receipt["image_allowlist"] == [merge.SMOKE_IMAGE_ID]
    assert {row["image_id"] for row in merged.events} == {merge.SMOKE_IMAGE_ID}
    # It reuses the real admission/event path rather than bypassing it.
    assert merged.events
    assert all(row["identity"]["admission_receipt_id"] for row in merged.events)
    # It concludes nothing, takes no calibration, and cannot be relabelled.
    assert merged.receipt["quarantine_ledger"]["global_stop_evaluated"] is False
    assert all(
        row["disposition"] == merge.DISPOSITION_UNCALIBRATED
        for row in merged.owner_summaries
    )
    with pytest.raises(merge.MergeContractError, match="never be marked usable"):
        merge.merge_census(
            plan,
            shard_root,
            allowlist=[merge.SMOKE_IMAGE_ID],
            phase=merge.PHASE_SMOKE_ADMIT,
            usable_as_census_conclusion=True,
        )
    with pytest.raises(merge.MergeContractError, match="exactly image"):
        merge.merge_census(
            plan,
            shard_root,
            allowlist=list(IMAGE_IDS),
            phase=merge.PHASE_SMOKE_ADMIT,
            usable_as_census_conclusion=False,
        )
    # And a full capture over the same root still fails: eleven shards are absent.
    with pytest.raises(merge.GlobalStopError):
        merge.build_capture_manifest(plan, shard_root)


def test_smoke_admit_does_not_weaken_the_global_stop_policy(tmp_path: Path) -> None:
    """A smoke run cannot mask an over-budget capture: the manifest fails first."""

    plan_dir = build_plan_dir(tmp_path)
    plan = merge.load_plan(plan_dir)
    shard_root = tmp_path / "shards"
    write_shards(
        plan,
        shard_root,
        ShardOptions(quarantined=frozenset(IMAGE_IDS[:3])),
    )
    with pytest.raises(merge.GlobalStopError):
        merge.build_capture_manifest(plan, shard_root)


def test_manifest_fails_closed_on_cross_split_runtime_drift(tmp_path: Path) -> None:
    """Uniformity must be proven across both splits, at manifest time.

    Each later phase only sees its own half, so a discovery/confirmation model
    or code divergence would otherwise survive while confirmation still bound
    the same manifest digest.
    """

    plan_dir = build_plan_dir(tmp_path)
    plan = merge.load_plan(plan_dir)

    def split_model(image_id: str) -> dict:
        return {
            "path": "model-discovery"
            if image_id in planner.DISCOVERY_IMAGE_IDS
            else "model-confirmation"
        }

    shard_root = tmp_path / "shards"
    write_shards(plan, shard_root, ShardOptions(model_identity=split_model))
    with pytest.raises(merge.MergeContractError, match="model_identity_sha256"):
        merge.build_capture_manifest(plan, shard_root)

    # Each split is internally uniform, so a per-split merge would not catch it.
    for phase in (merge.PHASE_DISCOVERY, merge.PHASE_CONFIRMATION):
        merge.merge_census(
            plan,
            shard_root,
            allowlist=merge._allowlist_for_phase(phase),
            phase=phase,
        )


def test_manifest_seals_the_uniform_runtime_identity(tmp_path: Path) -> None:
    plan_dir = build_plan_dir(tmp_path)
    plan = merge.load_plan(plan_dir)
    shard_root = tmp_path / "shards"
    write_shards(plan, shard_root)
    manifest = merge.build_capture_manifest(plan, shard_root)

    assert manifest["runtime_uniform_across_both_splits"] is True
    assert manifest["runtime_identity"]["model_identity_sha256"]
    assert manifest["runtime_identity"]["tokenizer_identity_sha256"]
    assert manifest["runtime_identity_sha256"] == merge.sha256_json(
        manifest["runtime_identity"]
    )


def test_shard_missing_an_evidence_file_is_not_captured(tmp_path: Path) -> None:
    """A receipt without its full evidence set is incomplete, never captured."""

    plan_dir = build_plan_dir(tmp_path)
    plan = merge.load_plan(plan_dir)
    shard_root = tmp_path / "shards"
    write_shards(plan, shard_root)
    (shard_root / IMAGE_IDS[0] / merge.SHARD_X1_NAME).unlink()

    shards = merge.discover_shards(plan, shard_root, load_evidence=False)
    incomplete = next(row for row in shards if row.image_id == IMAGE_IDS[0])
    assert incomplete.status == "incomplete"
    assert merge.SHARD_X1_NAME in incomplete.detail

    manifest = merge.build_capture_manifest(plan, shard_root)
    ledger = manifest["quarantine_ledger"]
    assert ledger["incomplete_image_ids"] == [IMAGE_IDS[0]]
    assert IMAGE_IDS[0] in ledger["unusable_image_ids"]
    assert IMAGE_IDS[0] not in ledger["captured_image_ids"]

    # And incompleteness spends the same unusable budget as a quarantine.
    (shard_root / IMAGE_IDS[1] / merge.SHARD_PROPOSAL_NAME).unlink()
    two = merge.build_capture_manifest(plan, shard_root)["quarantine_ledger"]
    assert two["incomplete_image_count"] == 2
    assert two["unusable_image_count"] == 2
    assert two["global_stop_triggered"] is False

    (shard_root / IMAGE_IDS[2] / merge.SHARD_FREE_DECODE_NAME).unlink()
    with pytest.raises(merge.GlobalStopError) as excinfo:
        merge.build_capture_manifest(plan, shard_root)
    assert "3 images are unusable" in str(excinfo.value)


def test_more_than_two_incomplete_shards_alone_trigger_the_global_stop(
    tmp_path: Path,
) -> None:
    """Incompleteness must spend the stop budget on its own.

    No shard here is quarantined or absent, so if ``incomplete`` were dropped
    from the unusable ledger the count would read zero and the census would
    merge a partial capture into a conclusion.
    """

    plan_dir = build_plan_dir(tmp_path)
    plan = merge.load_plan(plan_dir)
    shard_root = tmp_path / "shards"
    write_shards(plan, shard_root)

    # Two incomplete shards sit exactly at the frozen threshold.
    for image_id in IMAGE_IDS[:2]:
        (shard_root / image_id / merge.SHARD_X1_NAME).unlink()
    ledger = merge.build_capture_manifest(plan, shard_root)["quarantine_ledger"]
    assert ledger["quarantined_image_count"] == 0
    assert ledger["missing_image_count"] == 0
    assert ledger["incomplete_image_ids"] == sorted(IMAGE_IDS[:2], key=int)
    assert ledger["unusable_image_ids"] == sorted(IMAGE_IDS[:2], key=int)
    assert ledger["global_stop_triggered"] is False
    # An incomplete shard is never counted as captured.
    assert not set(ledger["captured_image_ids"]) & set(IMAGE_IDS[:2])
    assert len(ledger["captured_image_ids"]) == len(IMAGE_IDS) - 2

    # A third one crosses it, with nothing quarantined and nothing missing.
    (shard_root / IMAGE_IDS[2] / merge.SHARD_X1_NAME).unlink()
    with pytest.raises(merge.GlobalStopError) as excinfo:
        merge.build_capture_manifest(plan, shard_root)
    message = str(excinfo.value)
    assert "3 images are unusable" in message
    assert "above the frozen global stop threshold of 2" in message

    # The same budget is enforced on the plain merge path, which builds its own
    # ledger rather than binding a manifest.
    with pytest.raises(merge.GlobalStopError):
        merge.merge_census(plan, shard_root)


def test_receipt_declaring_an_unpublished_output_digest_is_rejected(
    tmp_path: Path,
) -> None:
    def declare_phantom(image_id: str, receipt: dict) -> dict:
        if image_id != IMAGE_IDS[0]:
            return receipt
        receipt = copy.deepcopy(receipt)
        receipt["output_file_digests"] = {"phantom-output.jsonl": "a" * 64}
        return receipt

    plan_dir = build_plan_dir(tmp_path)
    plan = merge.load_plan(plan_dir)
    write_shards(plan, tmp_path / "shards", ShardOptions(mutate_receipt=declare_phantom))
    with pytest.raises(merge.MergeContractError, match="no such file was published"):
        merge.merge_census(plan, tmp_path / "shards")


def test_tampered_capture_rules_cannot_silently_substitute_a_support_constant(
    tmp_path: Path,
) -> None:
    """Either the sealed value is used exactly, or the merge fails closed.

    Rewriting ``owner_support`` in the sealed plan changes the epsilon the merge
    applies -- proving the merge reads it rather than a private copy -- and the
    tampering is independently caught by the plan's own file digest.
    """

    plan_dir = build_plan_dir(tmp_path)
    original = merge.load_plan(plan_dir)
    sealed_epsilon = original.capture_rules["owner_support"]["epsilons"]["support_epsilon"]
    assert merge.load_support_contract(original).support_epsilon == sealed_epsilon

    # The value the merge applies tracks the sealed value exactly.
    moved = copy.deepcopy(dict(original.capture_rules))
    moved["owner_support"]["epsilons"]["support_epsilon"] = sealed_epsilon + 0.5
    tampered = merge.PlanBundle(
        plan_dir=original.plan_dir,
        receipt=original.receipt,
        capture_rules=moved,
        images=original.images,
        owners=original.owners,
        categories=original.categories,
        contexts=original.contexts,
        candidates=original.candidates,
        query_groups=original.query_groups,
        native_sidecars=original.native_sidecars,
        shards=original.shards,
    )
    assert merge.load_support_contract(tampered).support_epsilon == sealed_epsilon + 0.5

    # And tampering on disk is caught before any of it is read.
    rules_path = plan_dir / planner.CAPTURE_RULES_NAME
    payload = json.loads(rules_path.read_text(encoding="utf-8"))
    payload["owner_support"]["epsilons"]["support_epsilon"] = 0.5
    rules_path.write_bytes(planner.canonical_json_bytes(payload) + b"\n")
    with pytest.raises(merge.MergeContractError, match="sealed digest"):
        merge.load_plan(plan_dir)


def test_support_definition_forbidding_rank_is_enforced(tmp_path: Path) -> None:
    plan = merge.load_plan(build_plan_dir(tmp_path))
    weakened = copy.deepcopy(dict(plan.capture_rules))
    weakened["owner_support"]["support_definition"]["rank_is_support_criterion"] = True
    bundle = merge.PlanBundle(
        plan_dir=plan.plan_dir,
        receipt=plan.receipt,
        capture_rules=weakened,
        images=plan.images,
        owners=plan.owners,
        categories=plan.categories,
        contexts=plan.contexts,
        candidates=plan.candidates,
        query_groups=plan.query_groups,
        native_sidecars=plan.native_sidecars,
        shards=plan.shards,
    )
    with pytest.raises(merge.MergeContractError, match="forbid a rank criterion"):
        merge.load_support_contract(bundle)


# ---------------------------------------------------------------------------
# Round trip against the real scorer
# ---------------------------------------------------------------------------


def _score_with_real_scorer(tmp_path: Path, *, image_id: str, group_limit: int | None):
    """Run the *actual* scorer over the fixture plan and return its shard directory."""

    scorer = pytest.importorskip(
        "scripts.research.score_sorted_owner_accessibility_census_shard"
    )
    plan_dir = build_plan_dir(tmp_path)
    scorer_plan = scorer.load_plan(plan_dir)
    groups = [
        str(row["query_group_id"])
        for row in scorer_plan.image_query_groups(image_id)
        if row["status"] == "admitted"
    ]
    selected = groups if group_limit is None else groups[:group_limit]
    out = tmp_path / "shards" / image_id
    scorer.run_shard(
        scorer_plan,
        image_id=image_id,
        backend=scorer.FakeCensusBackend(),
        output_dir=out,
        query_group_ids=selected,
    )
    return plan_dir, out, groups, selected


def test_merge_admits_real_scorer_output_without_schema_drift(tmp_path: Path) -> None:
    """The merge must consume what the current scorer actually writes.

    This runs the real ``run_shard`` (deterministic stub backend, no GPU) and
    pushes its published artifacts through the merge's admission, event and
    feature code.  A schema divergence between the two modules fails here
    rather than at launch.
    """

    image_id = IMAGE_IDS[0]
    plan_dir, shard_dir, groups, selected = _score_with_real_scorer(
        tmp_path, image_id=image_id, group_limit=None
    )
    assert selected == groups

    plan = merge.load_plan(plan_dir)
    shards = merge.discover_shards(plan, tmp_path / "shards", allowlist=[image_id])
    shard = next(row for row in shards if row.image_id == image_id)
    assert shard.status == "captured", shard.detail
    assert shard.scores and shard.proposals

    # Admission receipts reconstruct channel-specifically from the scorer's own
    # authoritative `admission.receipts` block.
    admissions = merge.build_admission_index(plan, shard)
    assert admissions.by_receipt_id
    channels = {row["channel"] for row in admissions.by_receipt_id.values()}
    assert channels == set(planner.ADMISSION_CHANNELS)

    # Every published score row and proposal row is admitted.
    events = merge.admit_score_rows(plan, shard, admissions)
    assert len(events) == len(shard.scores)
    surfaces = merge.admit_proposal_rows(plan, shard, admissions)
    assert len(surfaces) == len(shard.proposals)

    # And the downstream views build from that real output.
    rows = merge.build_event_table(plan, events, merge.index_sidecars(plan, [shard]))
    owner_contexts = merge.build_owner_context_features(plan, rows, surfaces)
    assert rows and owner_contexts
    merge.assert_required_fields(plan, owner_contexts)
    block = owner_contexts[0]["localization"][
        "generator_local_max_excluding_other_owner_strict"
    ]["ambiguity_included_u"]
    assert block["peak_lift"] is not None
    assert block["local_concentration"] is not None
    assert owner_contexts[0]["category_proposal_channel"]["boundary_gate"] is not None


def test_stub_backend_shard_is_refused_as_evidence_not_as_schema(
    tmp_path: Path,
) -> None:
    """A stub-backend shard parses cleanly and is refused on provenance alone.

    The distinction matters: a schema rejection would mean the two modules
    disagree, whereas this rejection is exactly the intended one -- the fake
    backend declares itself unusable as evidence.
    """

    image_id = IMAGE_IDS[0]
    plan_dir, _shard_dir, _groups, _selected = _score_with_real_scorer(
        tmp_path, image_id=image_id, group_limit=1
    )
    plan = merge.load_plan(plan_dir)
    shard = next(
        row
        for row in merge.discover_shards(plan, tmp_path / "shards", allowlist=[image_id])
        if row.image_id == image_id
    )

    # Schema-level admission succeeds...
    admissions = merge.build_admission_index(plan, shard)
    assert merge.admit_score_rows(plan, shard, admissions)
    # ...and the only rejection is the provenance one.
    assert shard.receipt["backend_identity"]["usable_as_evidence"] is False
    with pytest.raises(merge.MergeContractError, match="unusable as evidence"):
        merge.validate_shard_lineage(plan, shard)


# ---------------------------------------------------------------------------
# Cohort domain: eligibility and the true-positive control population
# ---------------------------------------------------------------------------


def test_non_greedy_eligible_owner_is_floored_to_unresolved(tmp_path: Path) -> None:
    """An owner outside the native matching universe can never close negative."""

    image_id = IMAGE_IDS[0]
    overrides = {
        image_id: ImageSpec(
            owners=(
                OwnerSpec(owner_id="own-strong", box=SEPARATED_BOX_A),
                OwnerSpec(
                    owner_id="own-ambiguous",
                    box=SEPARATED_BOX_B,
                    greedy_eligible=False,
                ),
            ),
            contexts=(ContextSpec(boundary_index=1, frontier_box=(90, 90, 110, 110)),),
        )
    }

    def weak_last(group: Mapping[str, object], candidate: Mapping[str, object]) -> float:
        owners = candidate["generator_gt_owner_ids"]
        penalty = -50.0 if "own-ambiguous" in owners else 0.0
        return penalty - float(candidate["representative_role_ordinal"])

    merged = build_merged(
        tmp_path,
        overrides=overrides,
        options=ShardOptions(score_fn=weak_last),
        calibration=fixed_calibration(),
    )
    row = _summary(merged, "own-ambiguous")

    # It stays in the census with its continuous evidence intact...
    assert row["upper_bound_u"]["primary_best_non_loop"]["peak_lift"] is not None
    assert row["upper_bound_u"]["usable_support"] is False
    # ...but is floored, and is not an FN denominator member.
    assert row["greedy_eligible"] is False
    assert row["disposition"] == merge.DISPOSITION_NOT_ELIGIBLE
    assert row["disposition"] != merge.DISPOSITION_PERSISTENT_NEGATIVE
    assert "not_greedy_eligible" in row["disposition_blockers"]
    assert row["in_false_negative_prevalence_denominator"] is False
    assert row["cohort"] == "outside_native_matching_universe"
    assert row["persistent_negative_preconditions"]["greedy_eligible"] is False


def test_true_positive_below_threshold_is_a_control_not_a_false_negative(
    tmp_path: Path,
) -> None:
    """A calibration TP that fails the support test is never an FN negative.

    The frozen q10 threshold guarantees a share of TPs fail by construction, so
    labelling them with the false-negative disposition would count the control
    population as false negatives.
    """

    image_id = IMAGE_IDS[0]
    overrides = {
        image_id: ImageSpec(
            owners=(
                OwnerSpec(owner_id="own-strong", box=SEPARATED_BOX_A),
                OwnerSpec(
                    owner_id="own-tp-weak",
                    box=SEPARATED_BOX_B,
                    native_true_positive=True,
                ),
            ),
            contexts=(ContextSpec(boundary_index=1, frontier_box=(90, 90, 110, 110)),),
        )
    }

    def tp_below_threshold(
        group: Mapping[str, object], candidate: Mapping[str, object]
    ) -> float:
        owners = candidate["generator_gt_owner_ids"]
        penalty = -50.0 if "own-tp-weak" in owners else 0.0
        return penalty - float(candidate["representative_role_ordinal"])

    merged = build_merged(
        tmp_path,
        overrides=overrides,
        options=ShardOptions(score_fn=tp_below_threshold),
        calibration=fixed_calibration(),
    )
    row = _summary(merged, "own-tp-weak")

    assert row["native_true_positive"] is True
    # It genuinely fails the support test...
    assert row["upper_bound_u"]["usable_support"] is False
    assert row["frontier_tested"] is True
    assert row["bank_adequacy"]["adequate"] is True
    # ...and is still a control, never a false-negative finding.
    assert row["disposition"] == merge.DISPOSITION_TP_CONTROL
    assert row["disposition"] != merge.DISPOSITION_PERSISTENT_NEGATIVE
    assert row["cohort"] == "native_true_positive_control"
    assert row["in_false_negative_prevalence_denominator"] is False
    assert row["persistent_negative_preconditions"]["is_native_false_negative"] is False
    # Its continuous calibration evidence is retained.
    assert row["upper_bound_u"]["primary_best_non_loop"]["peak_lift"] is not None
    assert row["upper_bound_u"]["primary_best_non_loop"]["local_concentration"] is not None


def test_subset_smoke_shard_is_refused_by_every_conclusion_bearing_path(
    tmp_path: Path,
) -> None:
    """A smoke capture dropped into a full shard root must not count as captured."""

    def as_subset_smoke(image_id: str, receipt: dict) -> dict:
        if image_id != IMAGE_IDS[0]:
            return receipt
        receipt = copy.deepcopy(receipt)
        receipt["capture_completeness"] = merge.SUBSET_SMOKE
        receipt["subset_capture"] = {
            "is_subset": True,
            "planned_admitted_query_group_count": 4,
            "executed_query_group_count": 1,
            "usable_as_complete_shard_evidence": False,
        }
        return receipt

    plan_dir = build_plan_dir(tmp_path)
    plan = merge.load_plan(plan_dir)
    shard_root = tmp_path / "shards"
    write_shards(plan, shard_root, ShardOptions(mutate_receipt=as_subset_smoke))

    for phase, allowlist in (
        (merge.PHASE_FULL, None),
        (merge.PHASE_DISCOVERY, merge._allowlist_for_phase(merge.PHASE_DISCOVERY)),
    ):
        if allowlist is not None and IMAGE_IDS[0] not in allowlist:
            continue
        with pytest.raises(merge.MergeContractError, match="only a complete shard"):
            merge.merge_census(plan, shard_root, allowlist=allowlist, phase=phase)

    with pytest.raises(merge.MergeContractError, match="only a complete shard"):
        merge.build_capture_manifest(plan, shard_root)

    # The smoke-admit path alone accepts it, and labels it non-conclusion.
    smoke_root = tmp_path / "smoke"
    write_shards(
        plan,
        smoke_root,
        ShardOptions(
            omitted=frozenset(set(IMAGE_IDS) - {merge.SMOKE_IMAGE_ID}),
            mutate_receipt=lambda image_id, receipt: as_subset_smoke(IMAGE_IDS[0], receipt)
            if image_id == merge.SMOKE_IMAGE_ID
            else receipt,
        ),
    )
    merged = merge.merge_census(
        plan,
        smoke_root,
        allowlist=[merge.SMOKE_IMAGE_ID],
        phase=merge.PHASE_SMOKE_ADMIT,
        usable_as_census_conclusion=False,
    )
    assert merged.receipt["usable_as_census_conclusion"] is False
    lineage = merged.receipt["shard_lineage"][0]
    assert lineage["capture_completeness"] == merge.SUBSET_SMOKE
    assert lineage["usable_as_complete_shard_evidence"] is False


def test_discovery_output_is_recommitted_with_the_sealed_calibration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Discovery must publish *calibrated* support states, not the derivation pass.

    The calibration is derived from the uncalibrated continuous rows and then
    re-applied to the same discovery half, so a downstream discovery cohort or
    rule inspects calibrated evidence.  The second pass must still never open a
    confirmation file.
    """

    image_id = planner.DISCOVERY_IMAGE_IDS[0]
    overrides = {
        image_id: ImageSpec(
            owners=(
                OwnerSpec(
                    owner_id="own-tp", box=SEPARATED_BOX_A, native_true_positive=True
                ),
                OwnerSpec(owner_id="own-fn", box=SEPARATED_BOX_B),
            ),
            contexts=(
                ContextSpec(boundary_index=0),
                ContextSpec(boundary_index=1, frontier_box=(90, 90, 110, 110)),
            ),
        )
    }
    plan_dir = build_plan_dir(tmp_path, overrides)
    plan = merge.load_plan(plan_dir)
    shard_root = tmp_path / "shards"
    write_shards(plan, shard_root)
    manifest = merge.build_capture_manifest(plan, shard_root)

    first = merge.merge_census(
        plan,
        shard_root,
        manifest=manifest,
        allowlist=merge._allowlist_for_phase(merge.PHASE_DISCOVERY),
        phase=merge.PHASE_DISCOVERY,
    )
    assert first.receipt["dispositions_closed"] is False
    calibration = merge.calibrate_support(first, manifest)

    touched: list[Path] = []
    real_bytes, real_text = Path.read_bytes, Path.read_text
    monkeypatch.setattr(
        Path, "read_bytes", lambda self, *a, **k: (touched.append(self), real_bytes(self, *a, **k))[1]
    )
    monkeypatch.setattr(
        Path, "read_text", lambda self, *a, **k: (touched.append(self), real_text(self, *a, **k))[1]
    )
    second = merge.merge_census(
        plan,
        shard_root,
        manifest=manifest,
        allowlist=merge._allowlist_for_phase(merge.PHASE_DISCOVERY),
        phase=merge.PHASE_DISCOVERY,
        calibration=calibration,
    )

    confirmation_dirs = {
        (shard_root / cid).resolve() for cid in planner.CONFIRMATION_IMAGE_IDS
    }
    assert not [p for p in touched if p.resolve().parent in confirmation_dirs]

    assert second.receipt["dispositions_closed"] is True
    assert second.receipt["support_criterion"]["calibration_sha256"] == (
        calibration.describe()["calibration_sha256"]
    )
    assert not second.summaries_for_split("confirmation")
    for row in second.summaries_for_split("discovery"):
        assert row["disposition"] != merge.DISPOSITION_UNCALIBRATED
        assert row["upper_bound_u"]["usable_support"] is not None
        assert row["upper_bound_u"]["support_calibrated"] is True


# ---------------------------------------------------------------------------
# Sidecar undercoverage diagnostics
# ---------------------------------------------------------------------------


def test_out_of_bank_free_box_stays_visible_but_never_enters_ranks(
    tmp_path: Path,
) -> None:
    """A strong free box outside the fixed bank must remain diagnosable.

    It is the sharpest evidence that the seventeen-role bank under-covers the
    owner's landscape, so it may not vanish -- and it may not enter any rank,
    posterior or support test either.
    """

    image_id = IMAGE_IDS[0]
    overrides = {
        image_id: ImageSpec(
            owners=(OwnerSpec(owner_id="own-solo", box=FULL_BANK_BOX),),
            contexts=(ContextSpec(boundary_index=1, frontier_box=(10, 10, 30, 30)),),
        )
    }

    out_of_bank_bins = (5, 7, 900, 950)

    def out_of_bank_free_box(shard_image_id: str, rows: list[dict]) -> list[dict]:
        if shard_image_id != image_id:
            return rows
        for row in rows:
            row["coord_bins"] = list(out_of_bank_bins)
            row["coord_token_ids"] = [
                planner.COORD_TOKEN_START + value for value in out_of_bank_bins
            ]
            row["complete_box_logprob_sum"] = 25.0
            row["well_formed_box"] = True
        return rows

    plan_dir = build_plan_dir(tmp_path, overrides)
    plan = merge.load_plan(plan_dir)
    shard_root = tmp_path / "shards"
    write_shards(
        plan, shard_root, ShardOptions(mutate_free_decodes=out_of_bank_free_box)
    )
    merged = merge.merge_census(plan, shard_root, calibration=fixed_calibration())

    diagnostics = [
        row
        for row in merged.sidecar_diagnostics
        if row["image_id"] == image_id and row["sidecar_kind"] == "free_greedy_box"
    ]
    assert diagnostics, "the out-of-bank free box vanished from the merged outputs"
    entry = diagnostics[0]
    assert entry["inside_fixed_bank"] is False
    assert entry["joins_fixed_bank_candidate_id"] is None
    assert entry["complete_box_logprob_sum"] == 25.0
    assert entry["strict_assignment_status"] in {"matched", "ambiguous_neutral", "unmatched"}
    assert entry["decoded_bbox_pixel_xyxy"]
    assert entry["enters_core_ranks"] is False
    assert entry["enters_support_test"] is False

    # The signed gap against the owner's fixed-bank best is published and, here,
    # positive: the free box outscores everything the bank contains.
    gap = next(row for row in entry["owner_gaps"] if row["gt_owner_id"] == "own-solo")
    assert gap["signed_gap_vs_fixed_bank_best_u"] > 0.0
    assert gap["fixed_bank_best_u"] is not None
    assert gap["intersection_over_union_with_owner"] is not None
    assert gap["extent_ratio"] is not None

    # It changed no rank, no posterior and no support outcome.
    assert all(row["candidate_id"] != entry["sidecar_id"] for row in merged.events)
    for key in {
        (row["image_id"], row["context_id"], row["normalized_description"])
        for row in merged.events
    }:
        rows = [
            row
            for row in merged.events
            if (row["image_id"], row["context_id"], row["normalized_description"]) == key
        ]
        assert sum(row["competition"]["within_group_posterior"] for row in rows) == (
            pytest.approx(1.0)
        )

    # Native sidecars keep their join/outside-bank provenance too.
    native = [
        row for row in merged.sidecar_diagnostics if row["sidecar_kind"] == "native_emitted_box"
    ]
    assert native
    assert all("inside_fixed_bank" in row for row in native)

    # And the surface is receipted.
    assert merged.receipt["counts"]["sidecar_diagnostic_row_count"] == len(
        merged.sidecar_diagnostics
    )
    assert merge.SIDECAR_DIAGNOSTIC_NAME in merged.receipt["output_file_digests"]


# ---------------------------------------------------------------------------
# Candidate-vs-owner geometry (local part / whole object / oversized drift)
# ---------------------------------------------------------------------------


def test_owner_local_maxima_carry_candidate_versus_owner_geometry(
    tmp_path: Path,
) -> None:
    merged = build_merged(tmp_path)
    row = next(
        row
        for row in merged.owner_contexts
        if row["localization"]["exclusion_filtered_primary_max"]["value"] is not None
    )
    localization = row["localization"]
    for key in (
        "generator_local_max",
        "exclusion_filtered_primary_max",
        "exact_anchor_score",
    ):
        geometry = localization[key]["geometry"]
        assert geometry is not None, key
        assert geometry["intersection_over_union_with_owner"] is not None
        assert len(geometry["signed_center_offset_pixels"]) == 2
        assert geometry["width_extent_ratio"] is not None
        assert geometry["height_extent_ratio"] is not None
        assert geometry["area_ratio"] is not None
        assert geometry["role"] == (
            "continuous_geometry_no_threshold_no_phenotype_label"
        )
    for bound in ("ambiguity_excluded_l", "ambiguity_included_u"):
        block = localization["generator_local_max_excluding_other_owner_strict"][bound]
        if block["value"] is not None:
            assert block["geometry"] is not None

    # The exact anchor is the owner's own box, so its geometry is the identity.
    anchor = localization["exact_anchor_score"]["geometry"]
    assert anchor["intersection_over_union_with_owner"] == pytest.approx(1.0, abs=0.05)
    assert anchor["width_extent_ratio"] == pytest.approx(1.0, abs=0.05)
    assert anchor["area_ratio"] == pytest.approx(1.0, abs=0.1)


def test_extent_ratios_separate_local_part_from_oversized_drift(
    tmp_path: Path,
) -> None:
    """Shrink roles read below one, expand roles above one, on the same owner."""

    image_id = IMAGE_IDS[0]
    overrides = {
        image_id: ImageSpec(
            owners=(OwnerSpec(owner_id="own-solo", box=FULL_BANK_BOX),),
            contexts=(ContextSpec(boundary_index=1, frontier_box=(10, 10, 30, 30)),),
        )
    }

    def by_role(role_name: str):
        def _score(group: Mapping[str, object], candidate: Mapping[str, object]) -> float:
            return 0.0 if candidate["representative_role"] == role_name else -30.0

        return _score

    ratios: dict[str, dict] = {}
    for role in ("isotropic_shrink", "isotropic_expand"):
        merged = build_merged(
            tmp_path / role,
            overrides=overrides,
            options=ShardOptions(score_fn=by_role(role)),
        )
        row = next(r for r in merged.owner_contexts if r["gt_owner_id"] == "own-solo")
        ratios[role] = row["localization"]["generator_local_max"]["geometry"]

    shrunk = ratios["isotropic_shrink"]
    grown = ratios["isotropic_expand"]

    # A local-part style box covers a fraction of the owner in both axes...
    assert shrunk["width_extent_ratio"] < 1.0
    assert shrunk["height_extent_ratio"] < 1.0
    assert shrunk["area_ratio"] < 1.0
    # ...and an oversized drift overshoots it in both.
    assert grown["width_extent_ratio"] > 1.0
    assert grown["height_extent_ratio"] > 1.0
    assert grown["area_ratio"] > 1.0
    # Both remain centred on the owner, so extent - not offset - is what
    # distinguishes them.
    for geometry in (shrunk, grown):
        assert abs(geometry["signed_center_offset_pixels"][0]) < 5.0
        assert abs(geometry["signed_center_offset_pixels"][1]) < 5.0


def test_other_owner_strict_candidates_stay_collision_only_with_geometry(
    tmp_path: Path,
) -> None:
    """Geometry never promotes an other-owner-strict candidate into support."""

    image_id = IMAGE_IDS[0]
    overrides = {
        image_id: ImageSpec(
            owners=(
                OwnerSpec(owner_id="own-left", box=FULL_BANK_BOX),
                OwnerSpec(owner_id="own-right", box=SECOND_FULL_BANK_BOX),
            ),
            contexts=(ContextSpec(boundary_index=1, frontier_box=(10, 10, 30, 30)),),
        )
    }
    merged = build_merged(tmp_path, overrides=overrides)
    collided = [
        row
        for row in merged.owner_contexts
        if row["collision_diagnostic"]["other_owner_strict_event_count"] > 0
    ]
    assert collided

    for row in collided:
        excluded = set(row["collision_diagnostic"]["other_owner_strict_candidate_ids"])
        for key in (
            "exclusion_filtered_primary_max",
            "strict_assigned_max",
            "ambiguous_upper_max",
        ):
            assert row["localization"][key].get("candidate_id") not in excluded
        for bound in ("ambiguity_excluded_l", "ambiguity_included_u"):
            block = row["localization"][
                "generator_local_max_excluding_other_owner_strict"
            ][bound]
            assert block.get("candidate_id") not in excluded
        assert row["collision_diagnostic"]["role"] == (
            "excluded_from_target_support_never_evidence_against_target"
        )


# ---------------------------------------------------------------------------
# Geometry provenance: the sealed plan is the single source
# ---------------------------------------------------------------------------


def _shared_candidate_score(shared_id: str):
    def _score(group: Mapping[str, object], candidate: Mapping[str, object]) -> float:
        return 0.0 if str(candidate["candidate_id"]) == shared_id else -30.0

    return _score


def test_cross_owner_alias_geometry_comes_from_each_owners_generator_entry(
    tmp_path: Path,
) -> None:
    """One collapsed candidate, two owners, two different geometries.

    ``FULL_BANK_BOX``'s exact anchor and ``SECOND_FULL_BANK_BOX``'s
    ``translate_up_left`` realize the identical coordinate tuple, so alias
    collapse leaves one physical candidate with two generators whose GT boxes
    differ.  Each owner must read *its own* generator entry; the candidate-level
    representative summary names only the first generator and must never be
    reused for the second.
    """

    image_id = IMAGE_IDS[0]
    overrides = {
        image_id: ImageSpec(
            owners=(
                OwnerSpec(owner_id="own-a", box=FULL_BANK_BOX),
                OwnerSpec(owner_id="own-b", box=SECOND_FULL_BANK_BOX),
            ),
            contexts=(ContextSpec(boundary_index=1, frontier_box=(10, 10, 30, 30)),),
        )
    }
    plan_dir = build_plan_dir(tmp_path, overrides)
    plan = merge.load_plan(plan_dir)

    shared = next(
        candidate
        for candidate in plan.candidates.values()
        if candidate["cross_owner_generated"]
        and {"own-a", "own-b"} <= set(candidate["generator_gt_owner_ids"])
    )
    shared_id = str(shared["candidate_id"])

    write_shards(
        plan, tmp_path / "shards", ShardOptions(score_fn=_shared_candidate_score(shared_id))
    )
    merged = merge.merge_census(plan, tmp_path / "shards")

    rows = {
        row["gt_owner_id"]: row
        for row in merged.owner_contexts
        if row["image_id"] == image_id
    }
    geometries = {}
    for owner_id in ("own-a", "own-b"):
        block = rows[owner_id]["localization"]["generator_local_max"]
        assert block["candidate_id"] == shared_id, owner_id
        geometries[owner_id] = block["geometry"]

    a_geom, b_geom = geometries["own-a"], geometries["own-b"]

    # Each block is tied to its own generator entry.
    assert a_geom["generator_gt_owner_id"] == "own-a"
    assert b_geom["generator_gt_owner_id"] == "own-b"
    assert a_geom["logical_transform_role"] != b_geom["logical_transform_role"]

    # And the numbers genuinely differ: a representative-summary shortcut would
    # have given own-b own-a's identity geometry.
    assert a_geom["intersection_over_union_with_owner"] == pytest.approx(1.0)
    assert b_geom["intersection_over_union_with_owner"] < 0.5
    assert a_geom["signed_center_offset_pixels"] == [0.0, 0.0]
    assert b_geom["signed_center_offset_pixels"] != [0.0, 0.0]
    assert a_geom["owner_bbox_pixel_xyxy"] != b_geom["owner_bbox_pixel_xyxy"]

    # The candidate-level representative view names only the first generator.
    representative = shared["representative_generator_geometry"]
    assert representative["generator_gt_owner_id"] == "own-a"
    assert b_geom["intersection_over_union_with_owner"] != (
        representative["intersection_over_union_with_generator"]
    )


def test_geometry_is_read_from_the_plan_and_never_recomputed(tmp_path: Path) -> None:
    """Every published field is a passthrough of the sealed generator entry."""

    image_id = IMAGE_IDS[0]
    overrides = {
        image_id: ImageSpec(
            owners=(
                OwnerSpec(owner_id="own-a", box=FULL_BANK_BOX),
                OwnerSpec(owner_id="own-b", box=SECOND_FULL_BANK_BOX),
            ),
            contexts=(ContextSpec(boundary_index=1, frontier_box=(10, 10, 30, 30)),),
        )
    }
    plan_dir = build_plan_dir(tmp_path, overrides)
    plan = merge.load_plan(plan_dir)
    write_shards(plan, tmp_path / "shards")
    merged = merge.merge_census(plan, tmp_path / "shards")

    checked = 0
    for row in merged.owner_contexts:
        owner_id = str(row["gt_owner_id"])
        for key in (
            "generator_local_max",
            "exclusion_filtered_primary_max",
            "strict_assigned_max",
            "ambiguous_upper_max",
            "exact_anchor_score",
        ):
            block = row["localization"][key]
            geometry = block.get("geometry")
            if geometry is None:
                continue
            candidate = plan.candidates[str(block["candidate_id"])]
            sealed = next(
                entry["geometry"]
                for entry in candidate["generators"]
                if str(entry["generator_gt_owner_id"]) == owner_id
            )
            assert geometry["geometry_provenance"] == (
                "sealed_plan_per_generator_geometry"
            )
            assert geometry["authority"] == "candidate.generators[].geometry"
            assert geometry["recomputed_in_merge"] is False
            # Exact passthrough of the sealed numbers.
            assert geometry["intersection_over_union_with_owner"] == (
                sealed["intersection_over_union_with_generator"]
            )
            assert geometry["signed_center_offset_pixels"] == (
                sealed["center_offset_pixels"]
            )
            assert geometry["width_extent_ratio"] == sealed["extent_ratio"][0]
            assert geometry["height_extent_ratio"] == sealed["extent_ratio"][1]
            assert geometry["area_ratio"] == sealed["area_ratio"]
            assert geometry["owner_bbox_pixel_xyxy"] == (
                sealed["generator_bbox_pixel_xyxy"]
            )
            # The normalized view is derived from sealed fields only.
            extent = sealed["generator_extent_pixels"]
            assert geometry["signed_center_offset_owner_relative"][0] == pytest.approx(
                sealed["center_offset_pixels"][0] / extent[0]
            )
            assert geometry["signed_center_offset_owner_relative"][1] == pytest.approx(
                sealed["center_offset_pixels"][1] / extent[1]
            )
            checked += 1
    assert checked, "no owner-local block carried geometry"


def test_geometry_fails_closed_without_this_owners_generator_provenance() -> None:
    candidate = {
        "candidate_id": "cand:abc",
        "decoded_bbox_pixel_xyxy": [0, 0, 10, 10],
        "generators": [
            {
                "generator_gt_owner_id": "own-other",
                "logical_transform_role": "exact_gt_anchor",
                "geometry": {
                    "generator_bbox_pixel_xyxy": [0, 0, 10, 10],
                    "intersection_over_union_with_generator": 1.0,
                    "center_offset_pixels": [0.0, 0.0],
                    "extent_ratio": [1.0, 1.0],
                    "candidate_extent_pixels": [10.0, 10.0],
                    "generator_extent_pixels": [10.0, 10.0],
                    "area_ratio": 1.0,
                },
            }
        ],
    }
    with pytest.raises(merge.MergeContractError, match="no generator geometry for owner"):
        merge.candidate_owner_geometry(candidate, "own-target")


def test_geometry_fails_closed_on_a_generator_entry_without_geometry() -> None:
    candidate = {
        "candidate_id": "cand:abc",
        "decoded_bbox_pixel_xyxy": [0, 0, 10, 10],
        "generators": [
            {
                "generator_gt_owner_id": "own-target",
                "logical_transform_role": "translate_left",
                "geometry": None,
            }
        ],
    }
    with pytest.raises(merge.MergeContractError, match="without sealed geometry"):
        merge.candidate_owner_geometry(candidate, "own-target")


def test_geometry_fails_closed_on_conflicting_generator_geometry() -> None:
    def entry(role: str, iou: float) -> dict:
        return {
            "generator_gt_owner_id": "own-target",
            "logical_transform_role": role,
            "geometry": {
                "generator_bbox_pixel_xyxy": [0, 0, 10, 10],
                "intersection_over_union_with_generator": iou,
                "center_offset_pixels": [0.0, 0.0],
                "extent_ratio": [1.0, 1.0],
                "candidate_extent_pixels": [10.0, 10.0],
                "generator_extent_pixels": [10.0, 10.0],
                "area_ratio": 1.0,
            },
        }

    # Two roles reaching one tuple for one owner is legal and must agree.
    agreeing = {
        "candidate_id": "cand:abc",
        "decoded_bbox_pixel_xyxy": [0, 0, 10, 10],
        "generators": [entry("translate_left", 1.0), entry("translate_up", 1.0)],
    }
    resolved = merge.candidate_owner_geometry(agreeing, "own-target")
    assert resolved["generator_entry_count_for_owner"] == 2
    assert resolved["intersection_over_union_with_owner"] == 1.0

    conflicting = {
        "candidate_id": "cand:abc",
        "decoded_bbox_pixel_xyxy": [0, 0, 10, 10],
        "generators": [entry("translate_left", 1.0), entry("translate_up", 0.25)],
    }
    with pytest.raises(merge.MergeContractError, match="conflicting generator geometries"):
        merge.candidate_owner_geometry(conflicting, "own-target")


def test_geometry_never_enters_score_rank_support_or_population(
    tmp_path: Path,
) -> None:
    """Geometry is descriptive: perturbing it must move no decision."""

    image_id = IMAGE_IDS[0]
    overrides = {
        image_id: ImageSpec(
            owners=(OwnerSpec(owner_id="own-solo", box=FULL_BANK_BOX),),
            contexts=(ContextSpec(boundary_index=1, frontier_box=(10, 10, 30, 30)),),
        )
    }
    merged = build_merged(
        tmp_path, overrides=overrides, calibration=fixed_calibration()
    )
    row = next(r for r in merged.owner_contexts if r["gt_owner_id"] == "own-solo")
    summary = _summary(merged, "own-solo")
    block = row["localization"]["generator_local_max_excluding_other_owner_strict"][
        "ambiguity_included_u"
    ]

    assert block["geometry"]["enters_score_rank_support_or_population"] is False
    # Support and ranks are functions of the score statistics alone.
    assert block["peak_lift"] is not None
    assert block["local_concentration"] is not None
    assert summary["upper_bound_u"]["usable_support"] is not None

    # Events carry no geometry-derived rank input, and the population is the
    # collapsed unique candidate set regardless of geometry.
    for event in merged.events:
        assert "geometry" not in event["competition"]
        assert set(event["competition"]) == {
            "population",
            "population_size",
            "rank",
            "margin_to_group_best",
            "margin_to_runner_up",
            "within_group_log_posterior",
            "within_group_posterior",
            "posterior_semantics",
            "is_model_probability",
            "sidecars_excluded_from_population",
        }


# ---------------------------------------------------------------------------
# Capture-manifest byte binding
# ---------------------------------------------------------------------------


def _sealed_root(tmp_path: Path, overrides: Mapping[str, ImageSpec] | None = None):
    plan_dir = build_plan_dir(tmp_path, overrides)
    plan = merge.load_plan(plan_dir)
    shard_root = tmp_path / "shards"
    write_shards(plan, shard_root)
    manifest = merge.build_capture_manifest(plan, shard_root)
    return plan, shard_root, manifest


@pytest.mark.parametrize(
    "mutated_file",
    [
        merge.SHARD_SCORES_NAME,
        merge.SHARD_PROPOSAL_NAME,
        merge.SHARD_FREE_DECODE_NAME,
        merge.SHARD_X1_NAME,
        merge.SHARD_RECEIPT_NAME,
    ],
)
def test_discovery_fails_closed_when_a_consumed_shard_changes_after_sealing(
    tmp_path: Path, mutated_file: str
) -> None:
    """A shard rewritten after manifest time must not be analyzable.

    Split-disjointness would pass here: the mutated file still belongs to the
    discovery split. Only byte binding catches it.
    """

    plan, shard_root, manifest = _sealed_root(tmp_path)
    image_id = planner.DISCOVERY_IMAGE_IDS[0]
    path = shard_root / image_id / mutated_file
    path.write_bytes(path.read_bytes() + b"\n")

    with pytest.raises(merge.MergeContractError) as excinfo:
        merge.merge_census(
            plan,
            shard_root,
            manifest=manifest,
            allowlist=merge._allowlist_for_phase(merge.PHASE_DISCOVERY),
            phase=merge.PHASE_DISCOVERY,
        )
    message = str(excinfo.value)
    assert "bytes changed after the capture manifest was sealed" in message
    assert f"changed=['{mutated_file}']" in message
    assert image_id in message


def test_confirmation_fails_closed_when_a_consumed_shard_changes_after_sealing(
    tmp_path: Path,
) -> None:
    plan, shard_root, manifest = _sealed_root(tmp_path)
    image_id = planner.CONFIRMATION_IMAGE_IDS[0]
    path = shard_root / image_id / merge.SHARD_SCORES_NAME
    path.write_bytes(path.read_bytes() + b"\n")

    with pytest.raises(merge.MergeContractError, match="bytes changed after the capture"):
        merge.merge_census(
            plan,
            shard_root,
            manifest=manifest,
            allowlist=merge._allowlist_for_phase(merge.PHASE_CONFIRMATION),
            phase=merge.PHASE_CONFIRMATION,
        )

    # A discovery-side mutation does not implicate the confirmation phase,
    # because confirmation never opens those files.
    discovery_path = (
        shard_root / planner.DISCOVERY_IMAGE_IDS[0] / merge.SHARD_SCORES_NAME
    )
    discovery_path.write_bytes(discovery_path.read_bytes() + b"\n")
    path.write_bytes(path.read_bytes()[:-1])
    merge.merge_census(
        plan,
        shard_root,
        manifest=manifest,
        allowlist=merge._allowlist_for_phase(merge.PHASE_CONFIRMATION),
        phase=merge.PHASE_CONFIRMATION,
    )


def test_manifest_binding_rejects_a_deleted_file_and_a_status_drift(
    tmp_path: Path,
) -> None:
    plan, shard_root, manifest = _sealed_root(tmp_path)
    image_id = planner.DISCOVERY_IMAGE_IDS[0]

    # Deleting an evidence file both removes a sealed digest and flips the
    # shard's status from captured to incomplete.
    (shard_root / image_id / merge.SHARD_X1_NAME).unlink()
    with pytest.raises(merge.MergeContractError) as excinfo:
        merge.merge_census(
            plan,
            shard_root,
            manifest=manifest,
            allowlist=merge._allowlist_for_phase(merge.PHASE_DISCOVERY),
            phase=merge.PHASE_DISCOVERY,
        )
    assert "status drifted from the frozen manifest disposition" in str(excinfo.value)


def test_manifest_binding_rejects_an_edited_or_foreign_manifest(
    tmp_path: Path,
) -> None:
    plan, shard_root, manifest = _sealed_root(tmp_path)
    allowlist = merge._allowlist_for_phase(merge.PHASE_DISCOVERY)

    def _run(candidate: Mapping[str, object]) -> None:
        merge.merge_census(
            plan,
            shard_root,
            manifest=candidate,
            allowlist=allowlist,
            phase=merge.PHASE_DISCOVERY,
        )

    # The sealed manifest binds cleanly.
    bound = merge.merge_census(
        plan, shard_root, manifest=manifest, allowlist=allowlist,
        phase=merge.PHASE_DISCOVERY,
    ).receipt["capture_manifest_binding"]
    assert bound["byte_identity_verified"] is True
    assert bound["capture_manifest_sha256"] == manifest["capture_manifest_sha256"]
    assert set(bound["bound_image_ids"]) == set(planner.DISCOVERY_IMAGE_IDS)

    # Edited after sealing.  Must be a *discovery* entry: the discovery phase
    # only binds the shards it actually consumes, so tampering with a
    # confirmation entry is caught by disjointness rather than by binding.
    edited = copy.deepcopy(manifest)
    discovery_entry = next(
        entry
        for entry in edited["images"]
        if entry["image_id"] == planner.DISCOVERY_IMAGE_IDS[0]
    )
    discovery_entry["file_digests"][merge.SHARD_SCORES_NAME] = "0" * 64
    with pytest.raises(merge.MergeContractError, match="does not reconstruct its own digest"):
        _run(edited)

    # Re-sealed but describing the wrong bytes: binding still catches it.
    resealed = copy.deepcopy(edited)
    resealed.pop("capture_manifest_sha256")
    resealed["capture_manifest_sha256"] = merge.sha256_json(resealed)
    with pytest.raises(merge.MergeContractError, match="bytes changed after the capture"):
        _run(resealed)

    # A manifest sealed against another plan.
    foreign = copy.deepcopy(manifest)
    foreign["plan_receipt_content_sha256"] = "f" * 64
    foreign.pop("capture_manifest_sha256")
    foreign["capture_manifest_sha256"] = merge.sha256_json(foreign)
    with pytest.raises(merge.MergeContractError, match="different plan receipt"):
        _run(foreign)

    # A duplicated image entry.
    duplicated = copy.deepcopy(manifest)
    duplicated["images"].append(copy.deepcopy(duplicated["images"][0]))
    duplicated.pop("capture_manifest_sha256")
    duplicated["capture_manifest_sha256"] = merge.sha256_json(duplicated)
    with pytest.raises(merge.MergeContractError, match="lists image .* twice"):
        _run(duplicated)

    # A manifest that does not cover every planned shard.
    partial = copy.deepcopy(manifest)
    partial["images"] = [
        entry
        for entry in partial["images"]
        if entry["image_id"] != planner.CONFIRMATION_IMAGE_IDS[0]
    ]
    partial.pop("capture_manifest_sha256")
    partial["capture_manifest_sha256"] = merge.sha256_json(partial)
    with pytest.raises(merge.MergeContractError, match="must cover every planned shard"):
        _run(partial)


# ---------------------------------------------------------------------------
# Analyzer source provenance
# ---------------------------------------------------------------------------


def _merge_source_sha256() -> str:
    """Hash the merge module's own file, independently of the module constant."""

    import hashlib

    return hashlib.sha256(
        Path(merge.__file__).resolve().read_bytes()
    ).hexdigest()


def test_capture_manifest_content_addresses_the_analyzer_source(
    tmp_path: Path,
) -> None:
    plan, shard_root, manifest = _sealed_root(tmp_path)
    expected = _merge_source_sha256()

    assert manifest["code"]["merge_source_sha256"] == expected
    assert manifest["code"]["merge_source_sha256"] == merge.MERGE_SOURCE_SHA256
    assert manifest["code"]["role"] == (
        "sealed_the_metadata_and_hash_policy_provenance_only"
    )
    # It participates in the manifest's own digest, so it cannot be swapped
    # after sealing without detection.
    assert manifest["capture_manifest_sha256"] == merge.sha256_json(
        {k: v for k, v in manifest.items() if k != "capture_manifest_sha256"}
    )
    tampered = copy.deepcopy(manifest)
    tampered["code"]["merge_source_sha256"] = "0" * 64
    assert merge.sha256_json(
        {k: v for k, v in tampered.items() if k != "capture_manifest_sha256"}
    ) != manifest["capture_manifest_sha256"]

    # Stable across repeated sealing of the same bytes.
    again = merge.build_capture_manifest(plan, shard_root)
    assert again["code"]["merge_source_sha256"] == expected
    assert again["capture_manifest_sha256"] == manifest["capture_manifest_sha256"]


def test_merged_receipt_content_addresses_the_analyzer_source(
    tmp_path: Path,
) -> None:
    merged = build_merged(tmp_path, calibration=fixed_calibration())
    code = merged.receipt["code"]
    expected = _merge_source_sha256()

    assert code["merge_source_sha256"] == expected
    assert code["role"] == "interpreted_scores_and_applied_calibration_provenance_only"
    # The scorer and planner sources stay bound alongside it.
    assert code["scorer_source_sha256"] == (
        merged.receipt["runtime_identity"]["executed_source_sha256"]
    )
    assert code["planner_source_sha256"] == (
        merged.receipt["runtime_identity"]["planner_source_sha256"]
    )
    # It participates in the receipt's own digest.
    assert merged.receipt["receipt_content_sha256"] == merge.sha256_json(
        {
            key: value
            for key, value in merged.receipt.items()
            if key != "receipt_content_sha256"
        }
    )
    # Present in the uncalibrated phase too.
    uncalibrated = build_merged(tmp_path / "second")
    assert uncalibrated.receipt["code"]["merge_source_sha256"] == expected


def test_analyzer_source_provenance_is_never_a_support_input(tmp_path: Path) -> None:
    """The hash is recorded and nothing reads it as evidence."""

    merged = build_merged(tmp_path, calibration=fixed_calibration())
    assert merged.receipt["code"]["is_support_input"] is False

    digest = merge.MERGE_SOURCE_SHA256
    # It appears nowhere in the per-owner or per-event decision surfaces.
    for row in merged.owner_summaries:
        for key in ("lower_bound_l", "upper_bound_u", "bank_adequacy", "routing_summary"):
            assert digest not in json.dumps(row[key])
        assert digest not in json.dumps(row["persistent_negative_preconditions"])
    for row in merged.events:
        assert digest not in json.dumps(row["competition"])
    for row in merged.owner_contexts:
        assert digest not in json.dumps(row["localization"])

    # And the support contract's declared inputs are the two score statistics.
    assert merged.receipt["support_semantics"]["inputs"] == list(
        merge.SUPPORT_FEATURE_NAMES
    )
    assert merged.receipt["support_semantics"]["rank_is_not_a_support_input"] is True


def _discovery_calibration(tmp_path: Path):
    """A real sealed calibration, derived on the discovery half."""

    image_id = planner.DISCOVERY_IMAGE_IDS[0]
    overrides = {
        image_id: ImageSpec(
            owners=(
                OwnerSpec(
                    owner_id="own-tp", box=SEPARATED_BOX_A, native_true_positive=True
                ),
                OwnerSpec(owner_id="own-fn", box=SEPARATED_BOX_B),
            ),
            contexts=(
                ContextSpec(boundary_index=0),
                ContextSpec(boundary_index=1, frontier_box=(90, 90, 110, 110)),
            ),
        )
    }
    plan, shard_root, manifest = _sealed_root(tmp_path, overrides)
    merged = merge.merge_census(
        plan,
        shard_root,
        manifest=manifest,
        allowlist=merge._allowlist_for_phase(merge.PHASE_DISCOVERY),
        phase=merge.PHASE_DISCOVERY,
    )
    return merge.calibrate_support(merged, manifest)


def test_calibration_receipt_binds_the_analyzer_source(tmp_path: Path) -> None:
    calibration = _discovery_calibration(tmp_path)
    receipt = calibration.describe()
    expected = _merge_source_sha256()

    assert receipt["merge_source_sha256"] == expected
    assert receipt["merge_source_sha256"] == merge.MERGE_SOURCE_SHA256
    assert receipt["merge_source_role"] == "derived_these_thresholds_provenance_only"

    # It participates in calibration_sha256.
    assert receipt["calibration_sha256"] == merge.sha256_json(
        {k: v for k, v in receipt.items() if k != "calibration_sha256"}
    )
    swapped = {**receipt, "merge_source_sha256": "0" * 64}
    assert merge.sha256_json(
        {k: v for k, v in swapped.items() if k != "calibration_sha256"}
    ) != receipt["calibration_sha256"]

    # Round-trips under the current analyzer.
    rebuilt = merge.calibration_from_receipt(receipt)
    assert rebuilt.theta_peak_lift == calibration.theta_peak_lift
    assert rebuilt.theta_local_concentration == calibration.theta_local_concentration
    assert rebuilt.describe()["calibration_sha256"] == receipt["calibration_sha256"]


def test_calibration_from_a_foreign_analyzer_source_is_refused(
    tmp_path: Path,
) -> None:
    """Thresholds derived by other analysis code may not be applied here."""

    receipt = _discovery_calibration(tmp_path).describe()

    # Legitimately sealed by a different analyzer: the self-digest reconstructs,
    # so only the source binding can catch it.
    foreign = {**receipt, "merge_source_sha256": "b" * 64}
    foreign.pop("calibration_sha256")
    foreign["calibration_sha256"] = merge.sha256_json(foreign)
    assert foreign["calibration_sha256"] == merge.sha256_json(
        {k: v for k, v in foreign.items() if k != "calibration_sha256"}
    )
    with pytest.raises(merge.MergeContractError, match="derived by a different analyzer"):
        merge.calibration_from_receipt(foreign)

    # Absent entirely.
    missing = {k: v for k, v in receipt.items() if k != "merge_source_sha256"}
    missing.pop("calibration_sha256")
    missing["calibration_sha256"] = merge.sha256_json(missing)
    with pytest.raises(merge.MergeContractError, match="does not record the analyzer source"):
        merge.calibration_from_receipt(missing)

    # Edited in place without re-sealing still fails on the digest first.
    edited = {**receipt, "merge_source_sha256": "c" * 64}
    with pytest.raises(merge.MergeContractError, match="does not reconstruct its own digest"):
        merge.calibration_from_receipt(edited)


def test_analyzer_source_binding_changes_no_threshold(tmp_path: Path) -> None:
    """Provenance only: the sealed numbers are identical with and without it."""

    calibration = _discovery_calibration(tmp_path)
    receipt = calibration.describe()
    sealed = merge.load_plan(tmp_path / "plan").capture_rules["owner_support"]

    assert receipt["quantile"] == sealed["support_calibration"]["primary_quantile"]
    assert receipt["epsilon"] == sealed["epsilons"]["support_epsilon"]
    assert receipt["theta_peak_lift"] == calibration.theta_peak_lift
    assert receipt["theta_local_concentration"] == calibration.theta_local_concentration

    # Stripping the provenance fields leaves every decision value untouched.
    decisions = {
        key: value
        for key, value in receipt.items()
        if key
        not in {
            "merge_source_sha256",
            "merge_source_role",
            "calibration_sha256",
        }
    }
    rebuilt = merge.calibration_from_receipt(receipt)
    for key, value in decisions.items():
        assert rebuilt.describe()[key] == value

    # And the support test itself reads only the two score statistics.
    block = {"peak_lift": 10.0, "local_concentration": 10.0}
    assert rebuilt.clears(block) == calibration.clears(block)


# ---------------------------------------------------------------------------
# CLI summary rendering per phase
# ---------------------------------------------------------------------------


def _smoke_only_root(tmp_path: Path) -> tuple[Path, Path]:
    """A shard root holding *only* the representative smoke image, as on disk
    after a real subset-smoke capture and before the full capture exists."""

    plan_dir = build_plan_dir(tmp_path)
    plan = merge.load_plan(plan_dir)
    shard_root = tmp_path / "shards"
    write_shards(
        plan,
        shard_root,
        ShardOptions(omitted=frozenset(set(IMAGE_IDS) - {merge.SMOKE_IMAGE_ID})),
    )
    return plan_dir, shard_root


def test_smoke_admit_main_commits_and_summarizes_without_crashing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The public CLI path must survive its own ledger schema.

    ``main`` commits the merged outputs and *then* prints ``summarize``, so a
    summary that assumed the full-census ledger shape crashed after the
    artifacts were already on disk -- the run looked failed while its outputs
    were complete.
    """

    plan_dir, shard_root = _smoke_only_root(tmp_path)
    output_dir = tmp_path / "out"

    exit_code = merge.main(
        [
            "--plan-dir",
            str(plan_dir),
            "--shard-root",
            str(shard_root),
            "--phase",
            merge.PHASE_SMOKE_ADMIT,
            "--output-dir",
            str(output_dir),
        ]
    )
    assert exit_code == 0

    printed = capsys.readouterr().out
    assert f"phase: {merge.PHASE_SMOKE_ADMIT}" in printed
    assert "single-image launch gate -- NOT a census conclusion" in printed
    assert "usable as conclusion     False" in printed
    assert merge.SMOKE_IMAGE_ID in printed
    assert "single_image_launch_gate" in printed
    # No census-wide aggregate is invented for a capture that has not happened.
    assert "quarantined" not in printed
    assert "missing" not in printed
    assert "captured images" not in printed

    # And the artifacts it committed before summarizing are intact.
    for name in (
        merge.EVENTS_NAME,
        merge.OWNER_CONTEXT_NAME,
        merge.OWNER_SUMMARY_NAME,
        merge.SIDECAR_DIAGNOSTIC_NAME,
        merge.MERGE_RECEIPT_NAME,
    ):
        assert (output_dir / name).is_file(), name
    receipt = json.loads((output_dir / merge.MERGE_RECEIPT_NAME).read_text())
    assert receipt["phase"] == merge.PHASE_SMOKE_ADMIT
    assert receipt["usable_as_census_conclusion"] is False
    assert receipt["dispositions_closed"] is False


def test_summarize_renders_the_smoke_ledger_without_full_census_keys(
    tmp_path: Path,
) -> None:
    plan_dir, shard_root = _smoke_only_root(tmp_path)
    plan = merge.load_plan(plan_dir)
    merged = merge.merge_census(
        plan,
        shard_root,
        allowlist=[merge.SMOKE_IMAGE_ID],
        phase=merge.PHASE_SMOKE_ADMIT,
        usable_as_census_conclusion=False,
    )

    # The ledger genuinely lacks the full-census aggregates; summarize must not
    # reach for them.
    ledger = merged.receipt["quarantine_ledger"]
    for absent in (
        "quarantined_image_count",
        "missing_image_count",
        "incomplete_image_count",
        "captured_image_ids",
    ):
        assert absent not in ledger

    text = merge.summarize(merged)
    assert "single-image launch gate" in text
    assert f"events                     {merged.receipt['counts']['event_count']}" in text
    assert "dispositions (closed: False)" in text
    # Every disposition on a smoke run is the uncalibrated one.
    assert merge.DISPOSITION_UNCALIBRATED in text


def test_summarize_still_reports_full_census_shard_dispositions(
    tmp_path: Path,
) -> None:
    """The conclusion-bearing phases keep their aggregate reporting."""

    plan_dir = build_plan_dir(tmp_path)
    plan = merge.load_plan(plan_dir)
    shard_root = tmp_path / "shards"
    write_shards(plan, shard_root, ShardOptions(quarantined=frozenset({IMAGE_IDS[0]})))
    (shard_root / IMAGE_IDS[1] / merge.SHARD_X1_NAME).unlink()

    merged = merge.merge_census(plan, shard_root)
    text = merge.summarize(merged)

    assert f"phase: {merge.PHASE_FULL}" in text
    assert "captured images            10" in text
    assert f"  quarantined              1 ['{IMAGE_IDS[0]}']" in text
    assert "  missing                  0 []" in text
    assert f"  incomplete               1 ['{IMAGE_IDS[1]}']" in text
    assert "single-image launch gate" not in text


def test_main_full_phase_summary_path_is_exercised(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Guard the other CLI branch that reaches summarize()."""

    plan_dir = build_plan_dir(tmp_path)
    plan = merge.load_plan(plan_dir)
    shard_root = tmp_path / "shards"
    write_shards(plan, shard_root)

    exit_code = merge.main(
        [
            "--plan-dir",
            str(plan_dir),
            "--shard-root",
            str(shard_root),
            "--phase",
            merge.PHASE_CAPTURE_MANIFEST,
            "--output-dir",
            str(tmp_path / "manifest-out"),
        ]
    )
    assert exit_code == 0
    printed = capsys.readouterr().out
    # The manifest phase builds no MergedCensus, so it prints no summary.
    assert "capture manifest sealed" in printed
    assert "phase: " not in printed
