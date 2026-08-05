"""Focused tests for the sorted owner accessibility census shard scorer.

Everything here is CPU-only.  The deterministic fake backend returns real
``torch`` tensors and is driven through the *same* production seams the GPU run
uses -- including ``run_cache_parity_gate`` -- so the contract assertions are
exercised rather than mocked away.  The real-HF wiring is covered separately by
a recording Qwen-like model that drives ``prefill_context`` /
``HFCacheBackend`` / ``_build_full_reforward_closure`` verbatim.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research import build_sorted_owner_accessibility_census_plan as planner  # noqa: E402
from scripts.research import score_sorted_owner_accessibility_census_shard as scorer  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic sealed plan
# ---------------------------------------------------------------------------

IMAGE_ID = "6040"
PROMPT_TOKEN_IDS = [11, 12, 13, 14]
#: "person" is one description token; "dining table" is three.  Their canonical
#: suffixes therefore have different lengths, which is what makes the
#: variable-length suffix and shape-coverage contracts testable.
CATEGORY_TOKENS = {"person": [900], "dining table": [901, 902, 903]}


def _coord(bin_index: int) -> int:
    return planner.COORD_TOKEN_START + int(bin_index)


def _candidate(
    *,
    description: str,
    bins: list[int],
    generators: list[tuple[str, str]],
    assigned_owner: str | None,
    status: str = "matched",
) -> dict[str, Any]:
    tokens = [_coord(v) for v in bins]
    return {
        "schema_version": planner.PLAN_SCHEMA_VERSION,
        "row_kind": "physical_candidate",
        "candidate_id": planner.physical_candidate_id(
            image_id=IMAGE_ID, normalized_description=description, tokens=tokens
        ),
        "image_id": IMAGE_ID,
        "normalized_description": description,
        "coord_bins": list(bins),
        "coord_token_ids": tokens,
        "coord_token_ids_sha256": planner.sha256_json(tokens),
        "decoded_bbox_pixel_xyxy": [float(v) for v in bins],
        "identity_rule": "digest_image_category_coord_tokens",
        "collapse_scope": "image_and_normalized_description",
        "selection_policy": "fixed_seventeen_role_order_without_scores",
        "generators": [
            {
                "generator_gt_owner_id": owner,
                "logical_transform_role": role,
                "role_ordinal": planner.ROLE_ORDINAL[role],
                "candidate_class": planner.ROLE_CLASS[role],
            }
            for owner, role in generators
        ],
        "generator_gt_owner_ids": sorted({owner for owner, _ in generators}),
        "generator_owner_count": len({owner for owner, _ in generators}),
        "cross_owner_generated": len({owner for owner, _ in generators}) > 1,
        "representative_role": generators[0][1],
        "representative_role_ordinal": planner.ROLE_ORDINAL[generators[0][1]],
        "candidate_class": planner.ROLE_CLASS[generators[0][1]],
        "candidate_provenance": "exact" if generators[0][1] == "exact_gt_anchor" else "neighborhood",
        "generator_provenance_role": "provenance_only_never_rank_or_assignment",
        "sidecar_provenance": [],
        "strict_assignment_scope": "same_normalized_description_only",
        "strict_assignment_status": status,
        "strict_assignment_gt_owner_id": assigned_owner,
        "ambiguity_owner_ids": [],
        "any_category_assignment_status": status,
        "any_category_assignment_gt_owner_id": assigned_owner,
        "any_category_assignment_role": "diagnostic_only_never_primary_rank_or_ambiguity",
    }


def _owner_bank(*, reached: list[str], uniquely_assigned: int) -> dict[str, Any]:
    status = (
        "full"
        if len(reached) >= planner.BANK_COVERAGE_FULL_REACHED_AT_LEAST
        else "adequate_reduced"
        if len(reached) >= planner.BANK_COVERAGE_ADEQUATE_REACHED_AT_LEAST
        else "undercovered_unresolved_only"
    )
    eligible = status in planner.DISPOSITION_ELIGIBLE_BANK_STATUSES
    return {
        "logical_role_count": planner.LOGICAL_ROLE_COUNT,
        "core_role_count": planner.CORE_ROLE_COUNT,
        "extension_role_count": planner.EXTENSION_ROLE_COUNT,
        "admitted_role_count": planner.LOGICAL_ROLE_COUNT,
        "not_admitted_role_count": 0,
        "distinct_physical_candidate_count": len(reached),
        "distinct_physical_candidates_reached": len(reached),
        "physical_candidate_ids": sorted(reached),
        "cross_owner_shared_candidate_count": 0,
        "exact_anchor_admitted": True,
        "exact_anchor_uniquely_self_assigned": True,
        "generator_local_bank_adequacy": {
            "status": status,
            "measured_on": "distinct_physical_candidate_count",
            "distinct_physical_candidate_count": len(reached),
            "exact_anchor_uniquely_self_assigned": True,
            "full_at_least": planner.BANK_COVERAGE_FULL_REACHED_AT_LEAST,
            "adequate_at_least": planner.BANK_COVERAGE_ADEQUATE_REACHED_AT_LEAST,
            "disposition_eligible": eligible,
            "threshold_status": "frozen_before_any_score",
        },
        "strict_assignment_coverage": {
            "uniquely_assigned_candidate_count": int(uniquely_assigned),
            "roles_lost_to_other_owner_assignment": 0,
            "roles_lost_to_ambiguous_assignment": 0,
            "roles_lost_to_unmatched_assignment": 0,
            "role": "separate_lower_bound_view_never_the_adequacy_gate",
        },
        "bank_coverage_status": status,
        "disposition_eligible": eligible,
        "undercovered": not eligible,
        "disposition_floor": (
            "no_floor" if eligible else scorer.UNDERCOVERED_DISPOSITION_FLOOR
        ),
        "logical_roles": [],
    }


def build_plan_rows() -> dict[str, list[dict[str, Any]]]:
    """One image, two contexts, two categories of different suffix lengths."""

    images = [
        {
            "schema_version": planner.PLAN_SCHEMA_VERSION,
            "row_kind": "census_image",
            "image_id": IMAGE_ID,
            "split": "discovery",
            "image_width": 640,
            "image_height": 480,
            "file_name": "6040.jpg",
            "prompt_token_ids": list(PROMPT_TOKEN_IDS),
            "prompt_token_ids_sha256": planner.sha256_json(PROMPT_TOKEN_IDS),
            "generated_token_ids_sha256": planner.sha256_json([]),
            "executed_media_sha256": "media-digest",
            "native_complete_row_count": 1,
            "native_stop_reason": "im_end",
            "native_seed": 0,
            "wrapper_token_ids": dict(planner.WRAPPER_TOKEN_IDS),
            "coordinate_token_ids": {
                "start": planner.COORD_TOKEN_START,
                "end_inclusive": planner.COORD_TOKEN_END,
                "bin_count": planner.COORD_BIN_COUNT,
            },
        }
    ]

    # Context 000 is the root (empty generated prefix); context 001 carries one
    # complete native row, so the two contexts have *different* observed
    # prefixes -- which is what the cross-context isolation test relies on.
    native_row = [
        planner.OBJECT_REF_START,
        *CATEGORY_TOKENS["person"],
        planner.OBJECT_REF_END,
        planner.BOX_START,
        _coord(10),
        _coord(20),
        _coord(30),
        _coord(40),
        planner.BOX_END,
    ]
    contexts = []
    for boundary_index, generated_prefix in enumerate(([], native_row)):
        contexts.append(
            {
                "schema_version": planner.PLAN_SCHEMA_VERSION,
                "row_kind": "census_context",
                "context_id": f"{IMAGE_ID}:boundary-{boundary_index:03d}",
                "image_id": IMAGE_ID,
                "split": "discovery",
                "boundary_index": boundary_index,
                "total_complete_row_count": 1,
                "context_role": "root" if boundary_index == 0 else "terminal",
                "terminal_kind": None if boundary_index == 0 else "natural_stop",
                "generated_prefix_token_ids": list(generated_prefix),
                "generated_prefix_token_ids_sha256": planner.sha256_json(list(generated_prefix)),
                "loop_marking": {
                    "prior_identical_row_count": None,
                    "consecutive_identical_row_run_length": None,
                    "repeated_raw_span_sha256": None,
                    "loop_tail": False,
                    "loop_tail_rule": "consecutive_identical_row_run_length >= 3",
                    "flag_is_not_a_mechanism_label": True,
                },
                "frontier": None,
                "image_owner_count": 2,
                "canvas": {"width": 640, "height": 480},
            }
        )

    categories = [
        {
            "schema_version": planner.PLAN_SCHEMA_VERSION,
            "row_kind": "census_category",
            "category_query_id": f"{IMAGE_ID}:{description}",
            "image_id": IMAGE_ID,
            "split": "discovery",
            "normalized_description": description,
            "owner_count_in_image": 1,
            "owner_ids": [f"gt:{IMAGE_ID}:{index}"],
            "token_source": "observed_native_row",
            "estimand_name": "category_field_support_at_owner_geometry",
            "status": "admitted",
            "category_token_ids": tokens,
            "category_token_ids_sha256": planner.sha256_json(tokens),
            "query_suffix_token_ids": planner.build_query_suffix(tokens),
            "query_suffix_token_ids_sha256": planner.sha256_json(
                planner.build_query_suffix(tokens)
            ),
        }
        for index, (description, tokens) in enumerate(sorted(CATEGORY_TOKENS.items()))
    ]

    candidates = [
        _candidate(
            description="person",
            bins=[10, 20, 30, 40],
            generators=[(f"gt:{IMAGE_ID}:1", "exact_gt_anchor")],
            assigned_owner=f"gt:{IMAGE_ID}:1",
        ),
        # One physical candidate that two owners of the same category both
        # generate: collapsed once, provenance retained.
        _candidate(
            description="person",
            bins=[11, 21, 31, 41],
            generators=[
                (f"gt:{IMAGE_ID}:1", "translate_left"),
                (f"gt:{IMAGE_ID}:9", "translate_right"),
            ],
            assigned_owner=f"gt:{IMAGE_ID}:1",
        ),
        _candidate(
            description="dining table",
            bins=[50, 60, 70, 80],
            generators=[(f"gt:{IMAGE_ID}:0", "exact_gt_anchor")],
            assigned_owner=f"gt:{IMAGE_ID}:0",
        ),
    ]
    by_description: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        by_description.setdefault(candidate["normalized_description"], []).append(candidate)

    owners = [
        {
            "schema_version": planner.PLAN_SCHEMA_VERSION,
            "row_kind": "census_owner",
            "gt_owner_id": f"gt:{IMAGE_ID}:{index}",
            "image_id": IMAGE_ID,
            "split": "discovery",
            "normalized_description": description,
            "bbox_pixel_xyxy": [1.0, 2.0, 3.0, 4.0],
            "native_strict_match_pred_row_ids": [],
            "native_true_positive": False,
            "calibration_role": "native_false_negative",
            "excluded_from_census": False,
            "candidate_bank": _owner_bank(
                reached=[row["candidate_id"] for row in by_description[description]],
                uniquely_assigned=len(by_description[description]),
            ),
        }
        for index, description in enumerate(sorted(CATEGORY_TOKENS))
    ]

    query_groups = []
    for context in contexts:
        observed = [*PROMPT_TOKEN_IDS, *context["generated_prefix_token_ids"]]
        observed_sha = planner.sha256_json(observed)
        for category in categories:
            description = category["normalized_description"]
            suffix = list(category["query_suffix_token_ids"])
            full_prefix = [*observed, *suffix]
            query_sha = planner.sha256_json(full_prefix)
            query_groups.append(
                {
                    "schema_version": planner.PLAN_SCHEMA_VERSION,
                    "row_kind": "census_query_group",
                    "query_group_id": f"{context['context_id']}|{description}",
                    "image_id": IMAGE_ID,
                    "split": "discovery",
                    "context_id": context["context_id"],
                    "category_query_id": category["category_query_id"],
                    "normalized_description": description,
                    "status": "admitted",
                    "query_suffix_token_ids": suffix,
                    "query_suffix_token_ids_sha256": planner.sha256_json(suffix),
                    "suffix_shape_class": planner.suffix_shape_class(
                        category["category_token_ids"]
                    ),
                    "observed_prefix_sha256": observed_sha,
                    "query_prefix_sha256": query_sha,
                    "query_prefix_token_count": len(full_prefix),
                    "observed_prefix_token_count": len(observed),
                    "admission_receipt_id": planner.admission_receipt_id(
                        context_id=context["context_id"],
                        channel=planner.CHANNEL_QUERY_SUFFIX,
                        prefix_sha256=query_sha,
                    ),
                    "proposal_boundary_gate_admission_receipt_id": (
                        planner.admission_receipt_id(
                            context_id=context["context_id"],
                            channel=planner.CHANNEL_PROPOSAL_BOUNDARY_GATE,
                            prefix_sha256=observed_sha,
                        )
                    ),
                    "proposal_route_admission_receipt_id": (
                        planner.proposal_route_admission_receipt_id(
                            context_id=context["context_id"],
                            observed_prefix_sha256=observed_sha,
                            category_token_ids=category["category_token_ids"],
                        )
                    ),
                    "proposal_route_token_ids": planner.proposal_route_token_ids(
                        category["category_token_ids"]
                    ),
                    "proposal_route_digest": planner.proposal_route_digest(
                        category["category_token_ids"]
                    ),
                    "rank_key": {
                        "image_id": IMAGE_ID,
                        "context_id": context["context_id"],
                        "normalized_description": description,
                    },
                    "candidate_ids": sorted(
                        row["candidate_id"] for row in by_description[description]
                    ),
                    "candidate_count": len(by_description[description]),
                    "unique_coordinate_tuple_count": len(by_description[description]),
                    "request_policy": "exactly_one_request_per_unique_coordinate_tuple",
                }
            )

    shards = [
        {
            "schema_version": planner.PLAN_SCHEMA_VERSION,
            "row_kind": "census_shard",
            "image_id": IMAGE_ID,
            "split": "discovery",
            "estimated_work_units": len(query_groups) * len(candidates),
        }
    ]
    return {
        "image-registry.jsonl": images,
        "owner-registry.jsonl": owners,
        "category-registry.jsonl": categories,
        "context-registry.jsonl": contexts,
        "candidate-bank.jsonl": candidates,
        "query-group-registry.jsonl": query_groups,
        "native-sidecar-registry.jsonl": [],
        "shard-manifest.jsonl": shards,
    }


def seal_plan(plan_dir: Path, rows: dict[str, list[dict[str, Any]]]) -> Path:
    plan_dir.mkdir(parents=True, exist_ok=True)
    digests: dict[str, str] = {}
    for name in planner.PLAN_JSONL_NAMES:
        content = b"".join(planner.canonical_json_bytes(row) + b"\n" for row in rows[name])
        (plan_dir / name).write_bytes(content)
        digests[name] = hashlib.sha256(content).hexdigest()

    capture_rules = planner.build_capture_rules()
    capture_bytes = planner.canonical_json_bytes(capture_rules) + b"\n"
    (plan_dir / planner.CAPTURE_RULES_NAME).write_bytes(capture_bytes)
    digests[planner.CAPTURE_RULES_NAME] = hashlib.sha256(capture_bytes).hexdigest()

    receipt = {
        "schema_version": planner.PLAN_SCHEMA_VERSION,
        "unit_id": planner.UNIT_ID,
        "output_file_digests": digests,
        "capture_rules_sha256": capture_rules["capture_rules_sha256"],
    }
    receipt["receipt_content_sha256"] = planner.sha256_json(receipt)
    (plan_dir / "receipt.json").write_bytes(planner.canonical_json_bytes(receipt) + b"\n")
    return plan_dir


@pytest.fixture
def plan_dir(tmp_path: Path) -> Path:
    return seal_plan(tmp_path / "plan", build_plan_rows())


@pytest.fixture
def plan(plan_dir: Path):
    return scorer.load_plan(plan_dir)


@pytest.fixture
def backend():
    return scorer.FakeCensusBackend()


def reseal(tmp_path: Path, mutate) -> Path:
    rows = build_plan_rows()
    mutate(rows)
    return seal_plan(tmp_path / "mutated-plan", rows)


# ---------------------------------------------------------------------------
# Plan loading and pre-P0 refusal
# ---------------------------------------------------------------------------


def test_load_plan_reconstructs_every_sealed_digest(plan):
    assert plan.receipt_content_sha256
    assert plan.capture_rules_sha256
    assert set(plan.query_groups) == {
        f"{IMAGE_ID}:boundary-000|dining table",
        f"{IMAGE_ID}:boundary-000|person",
        f"{IMAGE_ID}:boundary-001|dining table",
        f"{IMAGE_ID}:boundary-001|person",
    }


def test_load_plan_refuses_a_foreign_plan_schema_version(plan_dir: Path):
    receipt = json.loads((plan_dir / "receipt.json").read_text())
    receipt["schema_version"] = "sorted-owner-accessibility-census-plan.v0-pre-p0"
    receipt["receipt_content_sha256"] = planner.sha256_json(
        {k: v for k, v in receipt.items() if k != "receipt_content_sha256"}
    )
    (plan_dir / "receipt.json").write_bytes(planner.canonical_json_bytes(receipt) + b"\n")
    with pytest.raises(scorer.ShardContractError, match="pre-P0 plan"):
        scorer.load_plan(plan_dir)


def test_load_plan_refuses_a_tampered_plan_file(plan_dir: Path):
    path = plan_dir / "candidate-bank.jsonl"
    path.write_bytes(path.read_bytes() + b'{"candidate_id": "smuggled"}\n')
    with pytest.raises(scorer.ShardContractError, match="sealed digest"):
        scorer.load_plan(plan_dir)


def test_pre_p0_score_row_is_rejected_by_the_row_binding_gate(plan, backend):
    result = scorer.run_shard(plan, image_id=IMAGE_ID, backend=backend, output_dir=None)
    good = result.scores[0]
    scorer.assert_row_bindings(good, label="p0 row")

    # A pre-P0 row: same score payload, none of the P0 capture bindings.
    pre_p0 = {
        key: value
        for key, value in good.items()
        if key
        not in {
            "row_contract",
            "channel",
            "admission_receipt_id",
            "capture_rules_sha256",
            "plan_receipt_content_sha256",
            "plan_schema_version",
        }
    }
    with pytest.raises(scorer.ShardContractError, match="pre-P0 rows are mechanically unjoinable"):
        scorer.assert_row_bindings(pre_p0, label="pre-P0 row")

    # A row that merely *claims* a different contract token is also refused.
    wrong_contract = dict(good, row_contract="pre-p0-shape-class-admission")
    with pytest.raises(scorer.ShardContractError, match="declares contract"):
        scorer.assert_row_bindings(wrong_contract, label="pre-P0 row")


def test_every_emitted_row_carries_the_full_binding_set(plan, backend):
    result = scorer.run_shard(plan, image_id=IMAGE_ID, backend=backend, output_dir=None)
    everything = [*result.scores, *result.x1, *result.proposals, *result.free_decodes]
    assert everything
    for row in everything:
        for key in scorer.REQUIRED_ROW_BINDING_FIELDS:
            assert row.get(key) not in (None, ""), (key, row["row_kind"])
        assert row["capture_rules_sha256"] == plan.capture_rules_sha256
        assert row["plan_receipt_content_sha256"] == plan.receipt_content_sha256


# ---------------------------------------------------------------------------
# Variable-length canonical suffix
# ---------------------------------------------------------------------------


def test_multi_token_category_suffix_is_accepted_and_shape_classed(plan):
    item = scorer.resolve_work_item(plan, f"{IMAGE_ID}:boundary-000|dining table")
    assert item.category_token_ids == CATEGORY_TOKENS["dining table"]
    assert item.query_suffix_token_ids == [
        planner.OBJECT_REF_START,
        *CATEGORY_TOKENS["dining table"],
        planner.OBJECT_REF_END,
        planner.BOX_START,
    ]
    assert item.query_prefix_token_ids[-1] == planner.BOX_START
    assert item.suffix_shape_class == "suffix_len_6"

    single = scorer.resolve_work_item(plan, f"{IMAGE_ID}:boundary-000|person")
    assert single.suffix_shape_class == "suffix_len_4"
    assert single.suffix_shape_class != item.suffix_shape_class


@pytest.mark.parametrize(
    ("label", "mutate_suffix"),
    [
        ("not_terminating_at_box_start", lambda suffix: suffix[:-1]),
        ("extra_token_before_box_start", lambda suffix: [*suffix[:-1], 999, suffix[-1]]),
        ("wrong_category_tokens", lambda suffix: [suffix[0], 777, *suffix[2:]]),
        ("truncated_multi_token_category", lambda suffix: [suffix[0], suffix[1], *suffix[-2:]]),
    ],
)
def test_non_canonical_suffix_is_rejected(tmp_path: Path, label, mutate_suffix):
    def mutate(rows):
        for group in rows["query-group-registry.jsonl"]:
            if group["normalized_description"] != "dining table":
                continue
            suffix = mutate_suffix(list(group["query_suffix_token_ids"]))
            observed = [
                *PROMPT_TOKEN_IDS,
                *next(
                    context["generated_prefix_token_ids"]
                    for context in rows["context-registry.jsonl"]
                    if context["context_id"] == group["context_id"]
                ),
            ]
            full_prefix = [*observed, *suffix]
            group["query_suffix_token_ids"] = suffix
            group["query_suffix_token_ids_sha256"] = planner.sha256_json(suffix)
            group["query_prefix_sha256"] = planner.sha256_json(full_prefix)
            group["admission_receipt_id"] = planner.admission_receipt_id(
                context_id=group["context_id"],
                channel=planner.CHANNEL_QUERY_SUFFIX,
                prefix_sha256=group["query_prefix_sha256"],
            )

    mutated = scorer.load_plan(reseal(tmp_path, mutate))
    with pytest.raises(scorer.ShardContractError):
        scorer.resolve_work_item(mutated, f"{IMAGE_ID}:boundary-000|dining table")


def test_suffix_of_another_category_is_rejected_even_when_shape_matches(tmp_path: Path):
    """A same-length suffix carrying different tokens must still fail closed."""

    def mutate(rows):
        for group in rows["query-group-registry.jsonl"]:
            if group["normalized_description"] != "dining table":
                continue
            suffix = [
                planner.OBJECT_REF_START,
                901,
                902,
                904,  # one token differs; the length class is unchanged
                planner.OBJECT_REF_END,
                planner.BOX_START,
            ]
            observed = [
                *PROMPT_TOKEN_IDS,
                *next(
                    context["generated_prefix_token_ids"]
                    for context in rows["context-registry.jsonl"]
                    if context["context_id"] == group["context_id"]
                ),
            ]
            group["query_suffix_token_ids"] = suffix
            group["query_suffix_token_ids_sha256"] = planner.sha256_json(suffix)
            group["query_prefix_sha256"] = planner.sha256_json([*observed, *suffix])
            group["admission_receipt_id"] = planner.admission_receipt_id(
                context_id=group["context_id"],
                channel=planner.CHANNEL_QUERY_SUFFIX,
                prefix_sha256=group["query_prefix_sha256"],
            )

    mutated = scorer.load_plan(reseal(tmp_path, mutate))
    with pytest.raises(scorer.ShardContractError, match="canonical query suffix"):
        scorer.resolve_work_item(mutated, f"{IMAGE_ID}:boundary-000|dining table")


# ---------------------------------------------------------------------------
# Cross-owner collapse and rank population
# ---------------------------------------------------------------------------


def test_cross_owner_identical_tuple_is_one_physical_candidate(plan, backend):
    item = scorer.resolve_work_item(plan, f"{IMAGE_ID}:boundary-000|person")
    shared = [row for row in item.candidates if row["cross_owner_generated"]]
    assert len(shared) == 1
    assert shared[0]["generator_owner_count"] == 2
    assert len(shared[0]["generators"]) == 2

    result = scorer.run_shard(plan, image_id=IMAGE_ID, backend=backend, output_dir=None)
    rows = [
        row
        for row in result.scores
        if row["query_group_id"] == f"{IMAGE_ID}:boundary-000|person"
    ]
    tuples = [tuple(row["coord_token_ids"]) for row in rows]
    assert len(tuples) == len(set(tuples)) == 2
    shared_row = next(row for row in rows if row["cross_owner_generated"])
    assert shared_row["generator_owner_count"] == 2
    assert shared_row["competition"]["population_size"] == 2
    assert shared_row["generator_provenance_role"] == "provenance_only_never_rank_or_assignment"


def test_duplicate_coordinate_tuple_in_a_group_fails_closed(tmp_path: Path):
    def mutate(rows):
        original = rows["candidate-bank.jsonl"][0]
        clone = copy.deepcopy(original)
        clone["candidate_id"] = original["candidate_id"] + ":clone"
        rows["candidate-bank.jsonl"].append(clone)
        for group in rows["query-group-registry.jsonl"]:
            if group["normalized_description"] == "person":
                group["candidate_ids"] = sorted([*group["candidate_ids"], clone["candidate_id"]])
                group["candidate_count"] = len(group["candidate_ids"])

    mutated = scorer.load_plan(reseal(tmp_path, mutate))
    with pytest.raises(scorer.ShardContractError, match="share coordinate"):
        scorer.resolve_work_item(mutated, f"{IMAGE_ID}:boundary-000|person")


def test_repeated_candidate_id_in_a_group_fails_closed(tmp_path: Path):
    def mutate(rows):
        for group in rows["query-group-registry.jsonl"]:
            if group["normalized_description"] == "person":
                group["candidate_ids"] = [*group["candidate_ids"], group["candidate_ids"][0]]
                group["candidate_count"] = len(group["candidate_ids"])

    mutated = scorer.load_plan(reseal(tmp_path, mutate))
    with pytest.raises(scorer.ShardContractError, match="appears more than once"):
        scorer.resolve_work_item(mutated, f"{IMAGE_ID}:boundary-000|person")


def test_ranks_are_context_and_category_local_and_exclude_sidecars(plan, backend):
    result = scorer.run_shard(plan, image_id=IMAGE_ID, backend=backend, output_dir=None)
    by_group: dict[str, list[dict[str, Any]]] = {}
    for row in result.scores:
        by_group.setdefault(row["query_group_id"], []).append(row)
    assert len(by_group) == 4
    for group_id, rows in by_group.items():
        keys = {
            (row["image_id"], row["context_id"], row["normalized_description"]) for row in rows
        }
        assert len(keys) == 1, group_id
        ranks = sorted(row["competition"]["rank"] for row in rows)
        assert ranks == list(range(1, len(rows) + 1))
        best = max(rows, key=lambda row: row["complete_box_logprob_sum"])
        assert best["competition"]["rank"] == 1
        assert best["competition"]["margin_to_group_best"] == 0.0
        for row in rows:
            assert row["competition"]["population"] == (
                "collapsed_unique_physical_candidates_only"
            )
            assert row["competition"]["sidecars_excluded"] is True
            assert row["competition"]["generator_identity_excluded"] is True

    for row in result.free_decodes:
        assert row["enters_core_ranks"] is False
        assert row["is_sidecar"] is True
        assert "competition" not in row


def test_rank_population_refuses_duplicate_coordinate_sequences():
    rows = [
        {
            "candidate_id": "cand:a",
            "coord_token_ids": [scorer.COORD_TOKEN_START] * 4,
            "complete_box_logprob_sum": -1.0,
        },
        {
            "candidate_id": "cand:b",
            "coord_token_ids": [scorer.COORD_TOKEN_START] * 4,
            "complete_box_logprob_sum": -2.0,
        },
    ]
    with pytest.raises(scorer.ShardContractError, match="duplicate coordinate sequences"):
        scorer.attach_competition_ranks(rows)


def test_sidecar_identical_tuple_joins_provenance_without_rank_mass(plan, backend, monkeypatch):
    """A free box that lands on a bank tuple joins it; it never adds a rank."""

    item = scorer.resolve_work_item(plan, f"{IMAGE_ID}:boundary-000|person")
    target = item.candidates[0]
    tokens = [*(int(v) for v in target["coord_token_ids"]), planner.BOX_END]
    step = {"index": 0}

    def fake_readout(logits, *, selected_token_id=None, top_k=scorer.DEFAULT_TOP_K, **kwargs):
        if selected_token_id is None:
            chosen = tokens[min(step["index"], len(tokens) - 1)]
            step["index"] += 1
            return scorer.Readout(-0.5, chosen, chosen, -0.5, [chosen], [-0.5], None)
        return real_readout(logits, selected_token_id=selected_token_id, top_k=top_k, **kwargs)

    real_readout = scorer.readout
    guard = scorer.PhaseGuard()
    guard.open_generation_phase(backend=backend)
    monkeypatch.setattr(scorer, "readout", fake_readout)
    sidecar = scorer.free_greedy_box_sidecar(
        backend,
        plan,
        item,
        guard=guard,
        admission={"admission_receipt_id": item.admission_receipt_id, "admitted": True},
        bank_tokens={
            tuple(int(v) for v in target["coord_token_ids"]): str(target["candidate_id"])
        },
    )
    assert sidecar["well_formed_box"] is True
    assert sidecar["joins_physical_candidate_id"] == str(target["candidate_id"])
    assert sidecar["join_semantics"] == "joins_provenance_never_adds_rank_mass"
    assert sidecar["enters_core_ranks"] is False


# ---------------------------------------------------------------------------
# Admission: exact prefix, channel separation, coverage
# ---------------------------------------------------------------------------


def test_box_admission_is_keyed_on_the_exact_query_prefix(plan, backend):
    guard = scorer.PhaseGuard()
    person = scorer.resolve_work_item(plan, f"{IMAGE_ID}:boundary-000|person")
    table = scorer.resolve_work_item(plan, f"{IMAGE_ID}:boundary-000|dining table")
    other_context = scorer.resolve_work_item(plan, f"{IMAGE_ID}:boundary-001|person")

    # Same context, different category -> different exact prefix -> different receipt.
    assert person.context_id == table.context_id
    assert person.admission_receipt_id != table.admission_receipt_id
    # Same category, different context -> different receipt too.
    assert person.admission_receipt_id != other_context.admission_receipt_id

    receipt = scorer.run_box_channel_admission(backend, person, guard=guard)
    assert receipt["admitted"] is True
    assert receipt["admission_key"] == "exact_query_prefix_sha256"
    assert receipt["query_prefix_sha256"] == person.query_prefix_sha256
    assert receipt["depth_count"] == 4
    assert receipt["parity_gate"]["status"] == "passed"
    assert len(receipt["parity_gate"]["raw_logit_parity_steps"]) == 4
    assert receipt["parity_gate"]["branch_reverse_order_invariance"]["cache_length_restored"]
    assert receipt["inherited_from_another_prefix"] is False


def test_box_admission_probe_is_score_blind_and_deterministic(plan, backend):
    guard = scorer.PhaseGuard()
    item = scorer.resolve_work_item(plan, f"{IMAGE_ID}:boundary-000|person")
    first = scorer.run_box_channel_admission(backend, item, guard=guard)
    second = scorer.run_box_channel_admission(backend, item, guard=guard)
    assert first["probe_candidate_id"] == second["probe_candidate_id"]
    assert first["probe_candidate_id"] == min(
        str(row["candidate_id"]) for row in item.candidates
    )
    assert first["probe_is_score_blind"] is True


def test_proposal_channel_has_its_own_separate_admission(plan, backend):
    guard = scorer.PhaseGuard()
    context_id = f"{IMAGE_ID}:boundary-000"
    box = scorer.run_box_channel_admission(
        backend, scorer.resolve_work_item(plan, f"{context_id}|person"), guard=guard
    )
    boundary = scorer.run_proposal_boundary_admission(
        backend, plan, context_id=context_id, guard=guard
    )
    assert box["channel"] == planner.CHANNEL_QUERY_SUFFIX
    assert boundary["channel"] == planner.CHANNEL_PROPOSAL_BOUNDARY_GATE
    assert boundary["admission_receipt_id"] != box["admission_receipt_id"]
    assert boundary["admission_key"] == "exact_observed_prefix_sha256"
    assert boundary["query_suffix_token_ids_sha256"] == scorer.EMPTY_SUFFIX_SHA256
    assert boundary["covered_by_a_query_suffix_admission"] is False
    assert boundary["authorizes_category_routing_paths"] is False
    assert boundary["admitted"] is True
    assert boundary["depth_count"] == 1
    # Observed prefix only: it must not be the query prefix of any box group.
    assert boundary["observed_prefix_sha256"] != box["query_prefix_sha256"]


def test_each_category_routing_path_has_its_own_admission(plan, backend):
    """A first-category probe must never authorize another category's path."""

    guard = scorer.PhaseGuard()
    context_id = f"{IMAGE_ID}:boundary-000"
    categories = plan.image_categories(IMAGE_ID)
    assert [row["normalized_description"] for row in categories] == [
        "dining table",
        "person",
    ]
    receipts = {
        str(category["category_query_id"]): scorer.run_proposal_path_admission(
            backend, plan, context_id=context_id, category=category, guard=guard
        )
        for category in categories
    }
    table = receipts[f"{IMAGE_ID}:dining table"]
    person = receipts[f"{IMAGE_ID}:person"]

    assert table["channel"] == planner.CHANNEL_PROPOSAL_CATEGORY_ROUTE
    assert table["admission_receipt_id"] != person["admission_receipt_id"]
    assert table["routing_path_digest"] != person["routing_path_digest"]
    # Different token *content* and different length: three category tokens
    # versus one.
    assert table["routing_path_token_count"] == 5
    assert person["routing_path_token_count"] == 3
    assert table["routing_path_shape_class"] != person["routing_path_shape_class"]
    # Every depth of the executed path is parity-checked, plus the terminal
    # box_start readout the row_prefix_block consumes.
    assert table["depth_count"] == 6
    assert person["depth_count"] == 4
    assert table["depths"][-1]["selected_token_id"] == planner.BOX_START
    assert all(row["admitted"] for row in receipts.values())

    boundary = scorer.run_proposal_boundary_admission(
        backend, plan, context_id=context_id, guard=guard
    )
    assert boundary["admission_receipt_id"] not in {
        row["admission_receipt_id"] for row in receipts.values()
    }


def test_proposal_scoring_refuses_a_missing_or_foreign_route_admission(plan, backend):
    guard = scorer.PhaseGuard()
    context_id = f"{IMAGE_ID}:boundary-000"
    categories = plan.image_categories(IMAGE_ID)
    boundary = scorer.run_proposal_boundary_admission(
        backend, plan, context_id=context_id, guard=guard
    )
    receipts = {
        str(category["category_query_id"]): scorer.run_proposal_path_admission(
            backend, plan, context_id=context_id, category=category, guard=guard
        )
        for category in categories
    }

    # Missing: the first category's receipt may not stand in for the second.
    only_first = {f"{IMAGE_ID}:dining table": receipts[f"{IMAGE_ID}:dining table"]}
    with pytest.raises(scorer.ShardContractError, match="no proposal-route admission"):
        scorer.score_proposal_surface(
            backend,
            plan,
            context_id=context_id,
            guard=guard,
            boundary_admission=boundary,
            route_admissions=only_first,
        )

    # Swapped: each category's receipt must key on its own path.
    swapped = {
        f"{IMAGE_ID}:dining table": receipts[f"{IMAGE_ID}:person"],
        f"{IMAGE_ID}:person": receipts[f"{IMAGE_ID}:dining table"],
    }
    with pytest.raises(scorer.ShardContractError, match="routing admission for another path"):
        scorer.score_proposal_surface(
            backend,
            plan,
            context_id=context_id,
            guard=guard,
            boundary_admission=boundary,
            route_admissions=swapped,
        )

    # The boundary receipt is not a routing receipt.
    with pytest.raises(scorer.ShardContractError, match="routing admission for another path"):
        scorer.score_proposal_surface(
            backend,
            plan,
            context_id=context_id,
            guard=guard,
            boundary_admission=boundary,
            route_admissions={
                str(category["category_query_id"]): boundary for category in categories
            },
        )


def test_missing_suffix_shape_admission_coverage_fails(plan, backend):
    """Dropping the multi-token category's admission must fail the coverage audit."""

    guard = scorer.PhaseGuard()
    items = [
        scorer.resolve_work_item(plan, f"{IMAGE_ID}:boundary-000|person"),
        scorer.resolve_work_item(plan, f"{IMAGE_ID}:boundary-000|dining table"),
    ]
    admissions = {}
    for item in items:
        receipt = scorer.run_box_channel_admission(backend, item, guard=guard)
        admissions[receipt["admission_receipt_id"]] = receipt

    coverage = scorer.assert_suffix_shape_coverage(admissions, items)
    assert coverage["distinct_suffix_shape_classes"] == ["suffix_len_4", "suffix_len_6"]

    dropped = {
        key: value
        for key, value in admissions.items()
        if value["suffix_shape_class"] != "suffix_len_6"
    }
    with pytest.raises(scorer.ShardContractError, match="does not cover every executed"):
        scorer.assert_suffix_shape_coverage(dropped, items)

    # A failed (not-admitted) receipt does not count as coverage either.
    failed = dict(admissions)
    for key, value in failed.items():
        if value["suffix_shape_class"] == "suffix_len_6":
            failed[key] = {**value, "admitted": False}
    with pytest.raises(scorer.ShardContractError, match="does not cover every executed"):
        scorer.assert_suffix_shape_coverage(failed, items)


def test_scoring_refuses_an_admission_receipt_from_another_prefix(plan, backend):
    guard = scorer.PhaseGuard()
    person = scorer.resolve_work_item(plan, f"{IMAGE_ID}:boundary-000|person")
    table = scorer.resolve_work_item(plan, f"{IMAGE_ID}:boundary-000|dining table")
    foreign = scorer.run_box_channel_admission(backend, table, guard=guard)
    with pytest.raises(scorer.ShardContractError, match="another exact prefix"):
        scorer.score_query_group(
            backend,
            plan,
            person,
            guard=guard,
            admission=foreign,
            scalar_modulus=64,
        )


def test_shard_receipt_has_one_box_admission_per_group_and_one_proposal_per_context(
    plan, backend
):
    result = scorer.run_shard(plan, image_id=IMAGE_ID, backend=backend, output_dir=None)
    admission = result.receipt["admission"]
    assert admission["box_channel_receipt_count"] == 4
    assert admission["proposal_boundary_gate_receipt_count"] == 2
    # Two contexts x two categories: no routing path shares an admission.
    assert admission["proposal_route_receipt_count"] == 4
    receipt_ids = [row["admission_receipt_id"] for row in admission["receipts"]]
    assert len(receipt_ids) == len(set(receipt_ids)) == 10
    assert admission["all_admitted"] is True


# ---------------------------------------------------------------------------
# Cache isolation
# ---------------------------------------------------------------------------


def test_cross_context_cache_is_never_reused(plan, backend):
    """Every group gets a fresh cache rooted at its own exact prefix."""

    scorer.run_shard(plan, image_id=IMAGE_ID, backend=backend, output_dir=None)
    prefills = backend.prefill_calls
    assert prefills
    for tokens in prefills:
        assert tokens[: len(PROMPT_TOKEN_IDS)] == PROMPT_TOKEN_IDS
    # Two distinct observed prefixes exist (root and terminal contexts); no
    # prefill ever roots at a prefix belonging to neither.
    contexts = {
        tuple([*PROMPT_TOKEN_IDS, *row["generated_prefix_token_ids"]])
        for row in plan.contexts.values()
    }
    for tokens in prefills:
        assert any(tuple(tokens[: len(prefix)]) == prefix for prefix in contexts)
    assert backend.live_prefill_count == 0


def test_a_second_live_cache_is_refused(plan, backend):
    item = scorer.resolve_work_item(plan, f"{IMAGE_ID}:boundary-000|person")
    first = backend.prefill(item.query_prefix_token_ids)
    try:
        with pytest.raises(scorer.ShardContractError, match="second KV cache"):
            backend.prefill(item.observed_prefix_token_ids)
    finally:
        first.close()
    assert backend.live_prefill_count == 0


def test_poisoned_cross_context_cache_is_detected_not_scored(plan, backend, monkeypatch):
    """A cache carrying another context's state must fail the rooted assertion.

    Simulates the exact failure the discipline exists to prevent: a prefill
    handed back at the *wrong* root length, i.e. holding tokens from a
    different context.
    """

    other = scorer.resolve_work_item(plan, f"{IMAGE_ID}:boundary-001|person")
    target = scorer.resolve_work_item(plan, f"{IMAGE_ID}:boundary-000|person")
    real_prefill = backend.prefill

    def poisoned(token_ids):
        handle = real_prefill(token_ids)
        # Leak the other context's tokens into the cache backend.
        handle.cache_backend.step(other.query_prefix_token_ids[:3])
        return handle

    monkeypatch.setattr(backend, "prefill", poisoned)
    guard = scorer.PhaseGuard()
    with pytest.raises(scorer.ShardContractError, match="does not match the prefill length"):
        scorer.run_box_channel_admission(backend, target, guard=guard)
    assert backend.live_prefill_count == 0


def test_branch_state_never_leaks_between_candidates(plan, backend):
    """Scoring is order-independent: the cache is cropped back to the root."""

    guard = scorer.PhaseGuard()
    item = scorer.resolve_work_item(plan, f"{IMAGE_ID}:boundary-000|person")
    admission = scorer.run_box_channel_admission(backend, item, guard=guard)
    rows, _diag, _comp = scorer.score_query_group(
        backend, plan, item, guard=guard, admission=admission, scalar_modulus=64
    )
    forward = {row["candidate_id"]: row["complete_box_logprob_sum"] for row in rows}

    reversed_item = scorer.WorkItem(
        **{
            **item.__dict__,
            "candidates": list(reversed(item.candidates)),
        }
    )
    rows_reversed, _d, _c = scorer.score_query_group(
        backend,
        plan,
        reversed_item,
        guard=guard,
        admission=admission,
        scalar_modulus=64,
    )
    backward = {row["candidate_id"]: row["complete_box_logprob_sum"] for row in rows_reversed}
    assert forward == pytest.approx(backward, abs=0.0)


# ---------------------------------------------------------------------------
# Scalar reference and epsilon receipt
# ---------------------------------------------------------------------------


def test_scalar_spot_check_selection_is_digest_based_and_never_empty():
    request_ids = [f"group|cand:{index}" for index in range(6)]
    selected, forced = scorer.select_scalar_reference_request_ids(
        request_ids, modulus=10**9
    )
    assert len(selected) == 1
    assert forced in selected
    assert forced == min(
        request_ids,
        key=lambda rid: (
            int.from_bytes(hashlib.sha256(rid.encode()).digest()[:8], "big"),
            rid,
        ),
    )

    always, forced_none = scorer.select_scalar_reference_request_ids(request_ids, modulus=1)
    assert always == set(request_ids)
    assert forced_none is None


def test_every_query_group_carries_at_least_one_scalar_spot_check(plan, backend):
    result = scorer.run_shard(plan, image_id=IMAGE_ID, backend=backend, output_dir=None)
    groups = {row["query_group_id"] for row in result.scores}
    with_check = {
        row["query_group_id"] for row in result.scores if row["scalar_reference"] is not None
    }
    assert groups == with_check
    assert result.receipt["scalar_admission"]["every_group_has_a_spot_check"] is True
    for row in result.scores:
        if row["scalar_reference"] is not None:
            assert row["scalar_reference"]["target_blind"] is True
            assert row["scalar_reference"]["abs_diff_vs_cached"] == pytest.approx(0.0, abs=1e-6)


def test_image_epsilon_receipt_is_target_blind_and_eight_repeats(plan, backend):
    result = scorer.run_shard(plan, image_id=IMAGE_ID, backend=backend, output_dir=None)
    epsilon = result.receipt["scalar_admission"]["image_epsilon_receipt"]
    assert epsilon["scope"] == "image"
    assert epsilon["target_blind"] is True
    assert epsilon["reads_any_candidate"] is False
    assert epsilon["repeat_count"] == planner.SCALAR_REFERENCE_REPEAT_COUNT == 8
    assert len(epsilon["repeat_logprobs"]) == 8
    assert epsilon["probe_token_id"] == scorer.COORD_TOKEN_START
    assert epsilon["argmax_stable"] is True
    # One per image, not one per context/category.
    assert isinstance(epsilon, dict)


# ---------------------------------------------------------------------------
# Diagnostics, proposal semantics, phase order
# ---------------------------------------------------------------------------


def test_full_x1_distribution_is_captured_at_one_thousand_bins(plan, backend):
    result = scorer.run_shard(plan, image_id=IMAGE_ID, backend=backend, output_dir=None)
    assert len(result.x1) == 4
    for row in result.x1:
        assert len(row["x1_logprobs"]) == planner.COORD_BIN_COUNT == 1000
        assert row["read_point"] == "immediately_after_box_start"
        assert row["role"] == "diagnostic_only_never_a_2d_heatmap_never_a_rank"


def test_proposal_surface_keeps_three_named_quantities_separate(plan, backend):
    result = scorer.run_shard(plan, image_id=IMAGE_ID, backend=backend, output_dir=None)
    assert len(result.proposals) == 2
    for row in result.proposals:
        assert row["channel"] == planner.CHANNEL_PROPOSAL_BOUNDARY_GATE
        assert row["boundary_gate_admission_authorizes_routing_paths"] is False
        assert row["emits_per_owner_proposal_probability"] is False
        assert row["includes_coordinate_scores"] is False
        gate = row["boundary_gate"]
        assert gate["nothing_forced"] is True
        assert gate["semantics"] == "gate_only_never_description_accessibility"
        routing = row["category_routing_event"]
        assert {entry["normalized_description"] for entry in routing} == set(CATEGORY_TOKENS)
        route_ids = {entry["admission_receipt_id"] for entry in routing}
        assert len(route_ids) == len(routing)
        assert row["admission_receipt_id"] not in route_ids
        for entry in routing:
            assert entry["channel"] == planner.CHANNEL_PROPOSAL_CATEGORY_ROUTE
            assert entry["aggregation"] == "raw_sequence_sum_no_token_mean"
            expected = pytest.approx(
                sum(token["selected_logprob"] for token in entry["per_token"])
            )
            assert entry["raw_sequence_logprob_sum"] == expected
            assert entry["row_prefix_block_raw_sequence_logprob_sum"] == pytest.approx(
                gate["continue_logprob"]
                + entry["raw_sequence_logprob_sum"]
                + entry["box_start_logprob"]
            )
        assert sorted(entry["within_context_rank"] for entry in routing) == [1, 2]


def test_free_sidecars_run_only_after_decision_scoring(plan, backend):
    result = scorer.run_shard(plan, image_id=IMAGE_ID, backend=backend, output_dir=None)
    phases = result.receipt["phase_order"]
    assert phases["decision_scoring_complete_before_generation"] is True
    assert phases["generation_phase_after_decision_scoring"] is True
    assert phases["likelihood_scored_after_generation_phase"] is False
    assert phases["free_box_max_tokens"] == scorer.FREE_BOX_MAX_TOKENS == 5
    assert phases["free_row_max_tokens"] == scorer.FREE_ROW_MAX_TOKENS
    kinds = {row["row_kind"] for row in result.free_decodes}
    assert kinds == {
        "census_free_greedy_box_sidecar",
        "census_free_next_row_sidecar",
    }
    for row in result.free_decodes:
        assert row["uses_model_generate"] is False
        assert row["decode_mode"] == "greedy_explicit_position_cache"
        assert row["generation_phase_after_decision_scoring"] is True
        assert row["token_count"] <= row["max_tokens"]


def test_phase_guard_refuses_likelihood_scoring_after_generation_opens(plan, backend):
    guard = scorer.PhaseGuard()
    item = scorer.resolve_work_item(plan, f"{IMAGE_ID}:boundary-000|person")
    admission = scorer.run_box_channel_admission(backend, item, guard=guard)
    guard.open_generation_phase(backend=backend)
    with pytest.raises(scorer.ShardContractError, match="after the terminal generation phase"):
        scorer.score_query_group(
            backend, plan, item, guard=guard, admission=admission, scalar_modulus=64
        )
    with pytest.raises(scorer.ShardContractError, match="after the terminal generation phase"):
        scorer.run_proposal_boundary_admission(
            backend, plan, context_id=item.context_id, guard=guard
        )
    with pytest.raises(scorer.ShardContractError, match="after the terminal generation phase"):
        scorer.run_proposal_path_admission(
            backend,
            plan,
            context_id=item.context_id,
            category=plan.image_categories(IMAGE_ID)[0],
            guard=guard,
        )


def test_generation_phase_cannot_open_while_a_cache_is_live(plan, backend):
    guard = scorer.PhaseGuard()
    item = scorer.resolve_work_item(plan, f"{IMAGE_ID}:boundary-000|person")
    handle = backend.prefill(item.query_prefix_token_ids)
    try:
        with pytest.raises(scorer.ShardContractError, match="decision-phase cache is still live"):
            guard.open_generation_phase(backend=backend)
    finally:
        handle.close()
    guard.open_generation_phase(backend=backend)


def test_free_sidecars_are_refused_outside_the_generation_phase(plan, backend):
    guard = scorer.PhaseGuard()
    item = scorer.resolve_work_item(plan, f"{IMAGE_ID}:boundary-000|person")
    with pytest.raises(scorer.ShardContractError, match="terminal generation phase"):
        scorer.free_greedy_box_sidecar(
            backend,
            plan,
            item,
            guard=guard,
            admission={"admission_receipt_id": item.admission_receipt_id, "admitted": True},
            bank_tokens={},
        )


# ---------------------------------------------------------------------------
# Owner bank accounting
# ---------------------------------------------------------------------------


def test_owner_bank_accounting_is_propagated_with_its_disposition_floor(plan, backend):
    result = scorer.run_shard(plan, image_id=IMAGE_ID, backend=backend, output_dir=None)
    accounting = result.receipt["owner_bank_accounting"]
    assert {row["gt_owner_id"] for row in accounting} == {
        f"gt:{IMAGE_ID}:0",
        f"gt:{IMAGE_ID}:1",
    }
    for row in accounting:
        assert row["logical_role_count"] == planner.LOGICAL_ROLE_COUNT == 17
        assert row["undercovered"] is True
        assert row["disposition_floor"] == scorer.UNDERCOVERED_DISPOSITION_FLOOR
        assert row["physical_candidates_scored_in_shard"] >= 1
        assert row["physical_candidates_reached_but_unscored"] == []


def test_undercovered_owner_without_the_floor_fails_closed(tmp_path: Path):
    def mutate(rows):
        rows["owner-registry.jsonl"][0]["candidate_bank"]["disposition_floor"] = (
            "persistent_no_tested_localization_support"
        )

    mutated = scorer.load_plan(reseal(tmp_path, mutate))
    with pytest.raises(scorer.ShardContractError, match="unresolved-only disposition floor"):
        scorer.build_owner_bank_accounting(mutated, IMAGE_ID, scored_candidate_ids=[])


def test_owner_bank_accounting_requires_every_contract_key(tmp_path: Path):
    def mutate(rows):
        rows["owner-registry.jsonl"][0]["candidate_bank"]["strict_assignment_coverage"].pop(
            "uniquely_assigned_candidate_count"
        )

    mutated = scorer.load_plan(reseal(tmp_path, mutate))
    with pytest.raises(scorer.ShardContractError, match="missing"):
        scorer.build_owner_bank_accounting(mutated, IMAGE_ID, scored_candidate_ids=[])


# ---------------------------------------------------------------------------
# CLI: contract-only validation, smoke subset, quarantine
# ---------------------------------------------------------------------------


def test_validate_contract_only_loads_no_model(plan_dir: Path, monkeypatch, capsys):
    def explode(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("--validate-contract-only must not build a session")

    monkeypatch.setattr(scorer, "build_hf_session_spec", explode)
    monkeypatch.setattr(scorer, "open_hf_backend", explode)
    exit_code = scorer.main(
        ["--plan-dir", str(plan_dir), "--image-id", IMAGE_ID, "--validate-contract-only"]
    )
    assert exit_code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "contract_ok"
    assert report["loads_any_model"] is False
    assert report["admitted_query_group_count"] == 4
    assert report["distinct_suffix_shape_classes"] == ["suffix_len_4", "suffix_len_6"]
    assert report["box_channel_admission_receipt_count"] == 4
    assert report["proposal_boundary_gate_admission_receipt_count"] == 2
    assert report["proposal_route_admission_receipt_count"] == 4
    assert report["cross_owner_tuple_collapse"] == "verified_for_every_admitted_group"


def test_smoke_subset_scores_fewer_groups_without_touching_the_plan(
    plan, plan_dir: Path, backend, tmp_path: Path
):
    before = {
        name: (plan_dir / name).read_bytes()
        for name in (*planner.PLAN_FILE_NAMES, "receipt.json", planner.CAPTURE_RULES_NAME)
    }
    result = scorer.run_shard(
        plan,
        image_id=IMAGE_ID,
        backend=backend,
        output_dir=tmp_path / "smoke",
        max_query_groups=1,
    )
    assert result.receipt["capture_completeness"] == "subset_smoke"
    assert result.receipt["subset_capture"]["is_subset"] is True
    assert result.receipt["subset_capture"]["executed_query_group_count"] == 1
    assert result.receipt["subset_capture"]["planned_admitted_query_group_count"] == 4
    assert result.receipt["subset_capture"]["usable_as_complete_shard_evidence"] is False
    assert len(result.x1) == 1
    for name, content in before.items():
        assert (plan_dir / name).read_bytes() == content


def test_complete_capture_is_marked_complete(plan, backend, tmp_path: Path):
    result = scorer.run_shard(
        plan, image_id=IMAGE_ID, backend=backend, output_dir=tmp_path / "full"
    )
    assert result.receipt["capture_completeness"] == "complete_shard"
    assert result.receipt["subset_capture"]["is_subset"] is False
    for name in (
        scorer.SCORES_NAME,
        scorer.X1_NAME,
        scorer.PROPOSAL_NAME,
        scorer.FREE_DECODE_NAME,
        scorer.RECEIPT_NAME,
    ):
        assert (tmp_path / "full" / name).is_file()


def test_failure_writes_a_quarantine_and_no_cached_primary_rows(
    plan_dir: Path, tmp_path: Path, monkeypatch
):
    output_dir = tmp_path / "failed"

    def failing(*args, **kwargs):
        raise scorer.ShardContractError("synthetic mid-shard failure")

    monkeypatch.setattr(scorer, "score_query_group", failing)
    exit_code = scorer.main(
        [
            "--plan-dir",
            str(plan_dir),
            "--image-id",
            IMAGE_ID,
            "--backend",
            "fake",
            "--output-dir",
            str(output_dir),
        ]
    )
    assert exit_code == 3
    quarantine = json.loads((output_dir / scorer.QUARANTINE_NAME).read_text())
    assert quarantine["status"] == "quarantined"
    assert quarantine["scope"] == "this_image_only"
    assert quarantine["evidence_usable"] is False
    assert quarantine["primary_rows_written"] is False
    assert quarantine["primary_output_residue"] == []
    assert quarantine["staging_directory_residue"] == []
    assert quarantine["residue_origin"] == "none"
    for name in scorer.PRIMARY_OUTPUT_NAMES:
        assert not (output_dir / name).exists()
    assert not (output_dir / scorer.RECEIPT_NAME).exists()
    assert scorer.find_staging_residue(output_dir) == []


def test_failed_admission_writes_no_primary_rows(plan_dir: Path, tmp_path: Path, monkeypatch):
    output_dir = tmp_path / "failed-admission"
    real = scorer.run_box_channel_admission

    def not_admitted(*args, **kwargs):
        return {**real(*args, **kwargs), "admitted": False}

    monkeypatch.setattr(scorer, "run_box_channel_admission", not_admitted)
    exit_code = scorer.main(
        [
            "--plan-dir",
            str(plan_dir),
            "--image-id",
            IMAGE_ID,
            "--backend",
            "fake",
            "--output-dir",
            str(output_dir),
        ]
    )
    assert exit_code == 3
    quarantine = json.loads((output_dir / scorer.QUARANTINE_NAME).read_text())
    assert "parity admission" in quarantine["detail"]
    for name in scorer.PRIMARY_OUTPUT_NAMES:
        assert not (output_dir / name).exists()


def test_unknown_image_is_not_a_planned_shard(plan, backend):
    with pytest.raises(scorer.ShardContractError, match="not a planned shard"):
        scorer.run_shard(plan, image_id="99999", backend=backend, output_dir=None)


# ---------------------------------------------------------------------------
# Repetition-penalty stratum policy
# ---------------------------------------------------------------------------


def test_default_infer_config_is_the_rp1p0_override():
    assert scorer.DEFAULT_INFER_CONFIG.name.endswith("_rp1p0.yaml")
    assert scorer.DEFAULT_INFER_CONFIG.is_file()
    text = scorer.DEFAULT_INFER_CONFIG.read_text(encoding="utf-8")
    assert "repetition_penalty: 1.0" in text
    assert "extends: qwen3_vl_2b_desc_first_geo_sorted_step4887_human_refined12_hf_fp32.yaml" in text


def test_rp1p0_config_resolves_to_the_scored_stratum():
    """Resolve the real config: the leaf must actually carry rp1.0."""

    from src.config.inference import load_infer_config

    resolved = load_infer_config(scorer.DEFAULT_INFER_CONFIG.resolve(strict=True))
    assert resolved.config.backend.type == "hf"
    assert float(resolved.config.generation.repetition_penalty) == 1.0
    assert resolved.config.run.name.endswith("-rp1p0")
    assert (
        scorer.assert_repetition_penalty_stratum(
            resolved.config.generation.repetition_penalty, label="resolved"
        )
        == planner.NATIVE_REPETITION_PENALTY_STRATUM
    )
    # Model identity is unchanged: only the stratum and the run name differ.
    legacy = load_infer_config(
        (
            scorer.DEFAULT_INFER_CONFIG.parent
            / "qwen3_vl_2b_desc_first_geo_sorted_step4887_human_refined12_hf_fp32.yaml"
        ).resolve(strict=True)
    )
    assert resolved.config.adapter.path == legacy.config.adapter.path
    assert resolved.config.embedding_delta.path == legacy.config.embedding_delta.path
    assert resolved.config.data.input_jsonl == legacy.config.data.input_jsonl
    assert float(legacy.config.generation.repetition_penalty) == 1.1
    with pytest.raises(scorer.ShardContractError, match="scores only 1.0"):
        scorer.assert_repetition_penalty_stratum(
            legacy.config.generation.repetition_penalty, label="legacy"
        )


def test_repetition_penalty_stratum_gate_fails_closed_on_rp1p10():
    assert scorer.assert_repetition_penalty_stratum(1.0, label="probe") == 1.0
    with pytest.raises(scorer.ShardContractError, match="scores only 1.0"):
        scorer.assert_repetition_penalty_stratum(1.10, label="probe")
    with pytest.raises(scorer.ShardContractError, match="not numeric"):
        scorer.assert_repetition_penalty_stratum("nope", label="probe")


def test_inherited_rp1p10_config_is_still_present_and_untouched():
    legacy = (
        REPO_ROOT
        / "configs/coordexp_swift/infer"
        / "qwen3_vl_2b_desc_first_geo_sorted_step4887_human_refined12_hf_fp32.yaml"
    )
    assert legacy.is_file()
    assert "repetition_penalty: 1.10" in legacy.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Real-HF seam: explicit positions, no generate(), monkeypatchable construction
# ---------------------------------------------------------------------------


class _RecordingQwenModel(torch.nn.Module):
    """A tiny stand-in that drives the *real* production seams.

    It exposes exactly the surface ``prefill_context`` / ``HFCacheBackend`` /
    ``_build_full_reforward_closure`` touch: ``get_rope_index`` through a
    ``model`` attribute, ``parameters()`` for device resolution, and a forward
    that fills a real ``transformers`` ``DynamicCache``.  Every call it receives
    is recorded so the explicit-``position_ids`` invariant can be asserted
    directly rather than assumed.
    """

    def __init__(self, *, vocab_size: int = scorer.COORD_TOKEN_END + 11, layers: int = 2) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.layers = int(layers)
        self._weight = torch.nn.Parameter(torch.zeros(1))
        self.calls: list[dict[str, Any]] = []
        self.rope_deltas: Any = None
        self.generate_calls = 0

    # ``derive_prefill_position_state`` reads ``model.model.get_rope_index``.
    @property
    def model(self):
        return self

    def get_rope_index(self, input_ids, image_grid_thw, video_grid_thw, attention_mask=None):
        seq = int(input_ids.shape[1])
        positions = torch.arange(seq, dtype=torch.long).view(1, 1, -1).expand(3, 1, -1).clone()
        # A shared mutable attribute exactly like the real model's.
        self.rope_deltas = torch.zeros((1, 1), dtype=torch.long)
        return positions, self.rope_deltas

    def generate(self, *args, **kwargs):  # pragma: no cover - must never be called
        self.generate_calls += 1
        raise AssertionError("the census scorer must never call model.generate()")

    def forward(
        self,
        *,
        input_ids,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
        return_dict=True,
        logits_to_keep=1,
        **vision_kwargs,
    ):
        batch = int(input_ids.shape[0])
        length = int(input_ids.shape[1])
        new_tokens = [[int(v) for v in row] for row in input_ids.tolist()]
        past_length = (
            int(past_key_values.get_seq_length()) if past_key_values is not None else 0
        )
        self.calls.append(
            {
                "input_token_ids": list(new_tokens[0]),
                "input_token_ids_per_lane": [list(row) for row in new_tokens],
                "batch_size": batch,
                "position_ids": position_ids,
                "position_ids_is_none": position_ids is None,
                "use_cache": bool(use_cache),
                "has_pixel_values": "pixel_values" in vision_kwargs,
                "past_length": past_length if past_key_values is not None else None,
            }
        )
        if position_ids is None:
            raise AssertionError("a model call arrived without explicit position_ids")

        # The literal history is carried *inside* the KV cache, exactly as a
        # real model's is.  A lane therefore sees only its own tokens, and any
        # implementation that let lanes share key-value storage would produce a
        # visibly wrong history here rather than passing silently.
        if past_length:
            cached = past_key_values.layers[0].keys[:, 0, :past_length, 0]
            history = [[int(round(float(v))) for v in row] for row in cached.tolist()]
            if len(history) != batch:
                raise AssertionError(
                    f"cache holds {len(history)} lanes but the call passed {batch}"
                )
        else:
            history = [[] for _ in range(batch)]
        full = [[*history[lane], *new_tokens[lane]] for lane in range(batch)]

        if past_key_values is not None and use_cache:
            token_block = input_ids.to(dtype=torch.float32).view(batch, 1, length, 1)
            for layer_index in range(self.layers):
                past_key_values.update(token_block.clone(), token_block.clone(), layer_index)

        keep = int(logits_to_keep) or length
        lane_rows = []
        for lane in range(batch):
            start_index = len(full[lane]) - keep
            rows = []
            for offset in range(keep):
                prefix = full[lane][: start_index + offset + 1]
                digest = hashlib.sha256(",".join(str(v) for v in prefix).encode()).digest()
                generator = torch.Generator().manual_seed(int.from_bytes(digest[:8], "big"))
                rows.append(torch.randn(self.vocab_size, generator=generator))
            lane_rows.append(torch.stack(rows, dim=0))
        logits = torch.stack(lane_rows, dim=0)

        class _Output:
            pass

        out = _Output()
        out.logits = logits
        out.past_key_values = past_key_values
        return out


def _native_inputs(prompt_token_ids):
    return {
        "input_ids": torch.tensor([list(prompt_token_ids)], dtype=torch.long),
        "attention_mask": torch.ones((1, len(prompt_token_ids)), dtype=torch.long),
        "pixel_values": torch.zeros(4, 8),
        "image_grid_thw": torch.tensor([[1, 2, 2]], dtype=torch.long),
    }


def test_hf_backend_passes_explicit_position_ids_on_every_model_call(plan):
    model = _RecordingQwenModel()
    backend = scorer.HFCensusBackend(
        model=model,
        native_prompt_inputs=_native_inputs(PROMPT_TOKEN_IDS),
        prompt_token_ids=PROMPT_TOKEN_IDS,
        identity={"backend": "hf-recording"},
        expected_layer_count=model.layers,
    )
    item = scorer.resolve_work_item(plan, f"{IMAGE_ID}:boundary-000|person")

    handle = backend.prefill(item.query_prefix_token_ids)
    try:
        handle.assert_rooted(label="recording prefill")
        with handle.branch() as branch:
            branch.step([scorer.COORD_TOKEN_START])
            branch.step([scorer.COORD_TOKEN_START + 1])
        handle.assert_rooted(label="recording prefill after branch")
    finally:
        handle.close()
    backend.full_reforward(item.query_prefix_token_ids)

    assert model.calls
    assert model.generate_calls == 0
    for call in model.calls:
        assert call["position_ids_is_none"] is False
        assert call["position_ids"] is not None
        assert call["position_ids"].shape[0] == 3
    # Vision kwargs are consumed by the prefill/reforward calls only; the
    # continuation steps must not re-pass them.
    assert model.calls[0]["has_pixel_values"] is True
    assert model.calls[1]["has_pixel_values"] is False
    assert backend.live_prefill_count == 0


def test_hf_full_reforward_receives_the_complete_prompt_bearing_prefix(plan):
    model = _RecordingQwenModel()
    backend = scorer.HFCensusBackend(
        model=model,
        native_prompt_inputs=_native_inputs(PROMPT_TOKEN_IDS),
        prompt_token_ids=PROMPT_TOKEN_IDS,
        identity={"backend": "hf-recording"},
        expected_layer_count=model.layers,
    )
    item = scorer.resolve_work_item(plan, f"{IMAGE_ID}:boundary-000|person")
    backend.full_reforward(item.query_prefix_token_ids)

    call = model.calls[-1]
    assert call["use_cache"] is False
    # The prompt must be present: stripping it would reforward a different
    # sequence than the cached path scored.
    assert call["input_token_ids"] == item.query_prefix_token_ids
    assert call["input_token_ids"][: len(PROMPT_TOKEN_IDS)] == PROMPT_TOKEN_IDS


def test_hf_backend_rejects_a_prefix_that_is_not_prompt_rooted(plan):
    model = _RecordingQwenModel()
    backend = scorer.HFCensusBackend(
        model=model,
        native_prompt_inputs=_native_inputs(PROMPT_TOKEN_IDS),
        prompt_token_ids=PROMPT_TOKEN_IDS,
        identity={"backend": "hf-recording"},
        expected_layer_count=model.layers,
    )
    # Same *length* as the prompt but different content: a count-only check
    # would let this through.
    divergent = [*PROMPT_TOKEN_IDS[:-1], PROMPT_TOKEN_IDS[-1] + 1, planner.BOX_START]
    with pytest.raises(scorer.ShardContractError, match="does not begin with this image"):
        backend.prefill(divergent)
    with pytest.raises(scorer.ShardContractError, match="does not begin with this image"):
        backend.full_reforward(divergent)
    with pytest.raises(scorer.ShardContractError, match="at least the executed"):
        backend.prefill(PROMPT_TOKEN_IDS[:-1])
    # The prompt itself is a legitimate root prefix.
    handle = backend.prefill(PROMPT_TOKEN_IDS)
    handle.close()


def test_hf_backend_admission_runs_the_real_parity_gate(plan):
    model = _RecordingQwenModel()
    backend = scorer.HFCensusBackend(
        model=model,
        native_prompt_inputs=_native_inputs(PROMPT_TOKEN_IDS),
        prompt_token_ids=PROMPT_TOKEN_IDS,
        identity={"backend": "hf-recording"},
        expected_layer_count=model.layers,
    )
    item = scorer.resolve_work_item(plan, f"{IMAGE_ID}:boundary-000|person")
    receipt = scorer.run_box_channel_admission(backend, item, guard=scorer.PhaseGuard())
    assert receipt["parity_gate"]["cache_layers"]["observed_layer_count"] == model.layers
    assert receipt["depth_count"] == 4
    # The cache branch and the independent uncached reforward must agree at
    # every coordinate depth for the group to be admitted at all.
    assert receipt["parity_gate"]["status"] == "passed"
    assert receipt["admitted"] is True
    assert receipt["scoring_backend_selection"]["selected_backend"] == "kv_cache"
    assert model.generate_calls == 0
    for call in model.calls:
        assert call["position_ids_is_none"] is False


def test_full_shard_runs_end_to_end_on_the_real_hf_seam(plan):
    """The whole shard, driven through prefill_context/HFCacheBackend/BranchCursor."""

    model = _RecordingQwenModel()
    backend = scorer.HFCensusBackend(
        model=model,
        native_prompt_inputs=_native_inputs(PROMPT_TOKEN_IDS),
        prompt_token_ids=PROMPT_TOKEN_IDS,
        identity={"backend": "hf-recording", "is_real_model": False},
        expected_layer_count=model.layers,
    )
    result = scorer.run_shard(plan, image_id=IMAGE_ID, backend=backend, output_dir=None)
    assert result.receipt["status"] == "captured"
    assert result.receipt["admission"]["all_admitted"] is True
    assert result.receipt["counts"]["query_group_count"] == 4
    assert result.receipt["counts"]["proposal_surface_rows"] == 2
    assert result.receipt["runtime_invariants"]["uses_model_generate"] is False
    assert model.generate_calls == 0
    assert backend.live_prefill_count == 0
    for call in model.calls:
        assert call["position_ids_is_none"] is False
    for row in [*result.scores, *result.x1, *result.proposals, *result.free_decodes]:
        scorer.assert_row_bindings(row, label=row["row_kind"])


def test_open_hf_backend_is_monkeypatchable_and_verifies_the_plan(plan, monkeypatch):
    """The real construction path is explicit: session opening is injectable."""

    from src.inference import hf_backend as hf_backend_module

    model = _RecordingQwenModel()
    native = _native_inputs(PROMPT_TOKEN_IDS)

    class _FakeSession:
        def __init__(self, prompt_token_ids, media_sha256):
            self._model = model
            self._prompt = tuple(int(v) for v in prompt_token_ids)
            self._media = media_sha256

        @property
        def receipt(self):
            class _Receipt:
                @staticmethod
                def to_artifact_dict():
                    return {
                        "model_identity": {"path": "fake"},
                        "tokenizer_identity": {"path": "fake"},
                        "adapter_identity": {"path": "step-4887"},
                    }

            return _Receipt()

        def _materialize_native_inputs(self, requests):
            return native, (self._prompt,), ((1, 2, 2),), (self._media,)

    monkeypatch.setattr(hf_backend_module, "HFBackendSession", _FakeSession)

    class _Request:
        generation_policy = type("_Policy", (), {"repetition_penalty": 1.0})()

    spec = scorer.HFSessionSpec(
        image_id=IMAGE_ID,
        infer_config=Path("fake.yaml"),
        launch=object(),
        request=_Request(),
        image_grid_thw=(1, 2, 2),
        planned_prompt_token_ids=list(PROMPT_TOKEN_IDS),
        planned_executed_media_sha256="media-digest",
        repetition_penalty_stratum=1.0,
    )

    import contextlib as _contextlib

    session = _FakeSession(PROMPT_TOKEN_IDS, "media-digest")

    @_contextlib.contextmanager
    def opener(launch):
        yield session

    with scorer.open_hf_backend(spec, session_opener=opener) as backend:
        assert isinstance(backend, scorer.HFCensusBackend)
        identity = backend.identity
        assert identity["backend"] == "hf"
        assert identity["uses_model_generate"] is False
        assert identity["repetition_penalty_stratum"] == 1.0
        assert identity["adapter_identity"] == {"path": "step-4887"}
        assert identity["executed_media_sha256"] == "media-digest"

    # A divergent executed prompt must fail closed.
    bad_spec = scorer.HFSessionSpec(
        **{**spec.__dict__, "planned_prompt_token_ids": [*PROMPT_TOKEN_IDS[:-1], 999]}
    )
    with pytest.raises(scorer.ShardContractError, match="differ from the plan"):
        with scorer.open_hf_backend(bad_spec, session_opener=opener):
            pass

    # A divergent executed media digest must fail closed too.
    media_spec = scorer.HFSessionSpec(
        **{**spec.__dict__, "planned_executed_media_sha256": "other-digest"}
    )
    with pytest.raises(scorer.ShardContractError, match="media digest"):
        with scorer.open_hf_backend(media_spec, session_opener=opener):
            pass


def test_open_hf_backend_refuses_an_rp1p10_request(plan, monkeypatch):
    from src.inference import hf_backend as hf_backend_module

    monkeypatch.setattr(hf_backend_module, "HFBackendSession", object)

    class _Request:
        generation_policy = type("_Policy", (), {"repetition_penalty": 1.10})()

    spec = scorer.HFSessionSpec(
        image_id=IMAGE_ID,
        infer_config=Path("fake.yaml"),
        launch=object(),
        request=_Request(),
        image_grid_thw=(1, 2, 2),
        planned_prompt_token_ids=list(PROMPT_TOKEN_IDS),
        planned_executed_media_sha256="media-digest",
        repetition_penalty_stratum=1.0,
    )

    def opener(launch):  # pragma: no cover - must never open
        raise AssertionError("the session must not open under a foreign stratum")

    with pytest.raises(scorer.ShardContractError, match="scores only 1.0"):
        with scorer.open_hf_backend(spec, session_opener=opener):
            pass


# ---------------------------------------------------------------------------
# Root context: the observed prefix is the executed prompt itself
# ---------------------------------------------------------------------------


def test_root_context_observed_prefix_equals_the_executed_prompt(plan):
    """The root boundary has no generated rows, so it is prompt-only."""

    root = plan.contexts[f"{IMAGE_ID}:boundary-000"]
    assert root["context_role"] == "root"
    assert root["generated_prefix_token_ids"] == []
    assert scorer._observed_prefix_token_ids(plan, root["context_id"]) == PROMPT_TOKEN_IDS


def test_prompt_only_prefill_runs_on_the_production_seams(plan):
    """A prompt-only root prefill must work and stay explicitly positioned."""

    model = _RecordingQwenModel()
    backend = scorer.HFCensusBackend(
        model=model,
        native_prompt_inputs=_native_inputs(PROMPT_TOKEN_IDS),
        prompt_token_ids=PROMPT_TOKEN_IDS,
        identity={"backend": "hf-recording"},
        expected_layer_count=model.layers,
    )
    handle = backend.prefill(PROMPT_TOKEN_IDS)
    try:
        assert handle.prefill_length == len(PROMPT_TOKEN_IDS)
        handle.assert_rooted(label="prompt-only root")
        with handle.branch() as branch:
            branch.step([planner.OBJECT_REF_START])
        handle.assert_rooted(label="prompt-only root after branch")
    finally:
        handle.close()
    # The prompt-only path is a real prefill: it consumes the vision kwargs.
    assert model.calls[0]["has_pixel_values"] is True
    assert model.calls[0]["input_token_ids"] == PROMPT_TOKEN_IDS
    for call in model.calls:
        assert call["position_ids_is_none"] is False
    assert model.generate_calls == 0
    assert backend.live_prefill_count == 0


def test_full_reforward_accepts_the_prompt_only_boundary_prefix(plan):
    model = _RecordingQwenModel()
    backend = scorer.HFCensusBackend(
        model=model,
        native_prompt_inputs=_native_inputs(PROMPT_TOKEN_IDS),
        prompt_token_ids=PROMPT_TOKEN_IDS,
        identity={"backend": "hf-recording"},
        expected_layer_count=model.layers,
    )
    backend.full_reforward(PROMPT_TOKEN_IDS)
    call = model.calls[-1]
    assert call["use_cache"] is False
    assert call["input_token_ids"] == PROMPT_TOKEN_IDS


def test_root_context_proposal_and_free_next_row_work_end_to_end(plan):
    """The whole root-context proposal path, on the real seams, prompt-only."""

    model = _RecordingQwenModel()
    backend = scorer.HFCensusBackend(
        model=model,
        native_prompt_inputs=_native_inputs(PROMPT_TOKEN_IDS),
        prompt_token_ids=PROMPT_TOKEN_IDS,
        identity={"backend": "hf-recording"},
        expected_layer_count=model.layers,
    )
    guard = scorer.PhaseGuard()
    context_id = f"{IMAGE_ID}:boundary-000"
    boundary = scorer.run_proposal_boundary_admission(
        backend, plan, context_id=context_id, guard=guard
    )
    assert boundary["admitted"] is True
    routes = {
        str(category["category_query_id"]): scorer.run_proposal_path_admission(
            backend, plan, context_id=context_id, category=category, guard=guard
        )
        for category in plan.image_categories(IMAGE_ID)
    }
    row = scorer.score_proposal_surface(
        backend,
        plan,
        context_id=context_id,
        guard=guard,
        boundary_admission=boundary,
        route_admissions=routes,
    )
    scorer.assert_row_bindings(row, label="root proposal row")
    assert row["context_role"] == "root"
    assert len(row["category_routing_event"]) == 2

    guard.open_generation_phase(backend=backend)
    sidecar = scorer.free_next_row_sidecar(
        backend, plan, context_id=context_id, guard=guard, admission=boundary
    )
    scorer.assert_row_bindings(sidecar, label="root free next row")
    assert sidecar["token_count"] >= 1
    assert sidecar["channel"] == planner.CHANNEL_PROPOSAL_BOUNDARY_GATE
    assert model.generate_calls == 0
    for call in model.calls:
        assert call["position_ids_is_none"] is False


def test_full_shard_covers_the_root_context(plan, backend):
    """A full fake-backend shard includes the prompt-only root context."""

    result = scorer.run_shard(plan, image_id=IMAGE_ID, backend=backend, output_dir=None)
    roots = [
        row for row in result.proposals if row["context_id"] == f"{IMAGE_ID}:boundary-000"
    ]
    assert len(roots) == 1
    assert roots[0]["context_role"] == "root"
    assert any(
        row["context_id"] == f"{IMAGE_ID}:boundary-000"
        and row["row_kind"] == "census_free_next_row_sidecar"
        for row in result.free_decodes
    )


# ---------------------------------------------------------------------------
# Atomic shard publish
# ---------------------------------------------------------------------------


@pytest.fixture
def shard_result(plan, backend) -> scorer.ShardResult:
    return scorer.run_shard(plan, image_id=IMAGE_ID, backend=backend, output_dir=None)


def test_atomic_publish_creates_the_complete_output(shard_result, tmp_path: Path):
    output_dir = tmp_path / "runs" / "shard-6040"
    record = scorer.write_shard(output_dir, shard_result)

    assert record["published"] is True
    assert record["publish_mode"] == "atomic_staging_directory_rename"
    assert record["already_present_identical"] is False
    expected = scorer.shard_output_files(shard_result)
    assert sorted(child.name for child in output_dir.iterdir()) == sorted(expected)
    for name, content in expected.items():
        assert (output_dir / name).read_bytes() == content
    # Nothing owned by this invocation is left beside the published shard.
    assert scorer.find_staging_residue(output_dir) == []
    assert [child.name for child in output_dir.parent.iterdir()] == [output_dir.name]


def test_injected_mid_write_failure_leaves_no_final_primary_files(
    shard_result, tmp_path: Path, monkeypatch
):
    """An interruption part-way through the write must publish nothing."""

    output_dir = tmp_path / "runs" / "shard-6040"
    real_write = scorer._write_durable
    written: list[str] = []

    def failing(path: Path, content: bytes) -> None:
        if len(written) >= 2:
            raise OSError(28, "No space left on device")
        written.append(path.name)
        real_write(path, content)

    monkeypatch.setattr(scorer, "_write_durable", failing)
    with pytest.raises(OSError, match="No space left on device"):
        scorer.write_shard(output_dir, shard_result)

    # The final path was never created, so no partial primary row is visible.
    assert not output_dir.exists()
    assert scorer.find_staging_residue(output_dir) == []
    assert list(output_dir.parent.iterdir()) == []
    assert len(written) == 2  # the failure really did land mid-set


def test_failed_publish_reports_honest_residue_in_the_quarantine(
    shard_result, tmp_path: Path, monkeypatch
):
    """Residue from an *earlier* run is reported, never claimed away, never deleted."""

    output_dir = tmp_path / "runs" / "shard-6040"
    output_dir.mkdir(parents=True)
    stale = output_dir / scorer.SCORES_NAME
    stale.write_bytes(b'{"row_kind": "left over from an earlier tool"}\n')

    # A fresh capture must refuse to publish into that foreign directory.
    with pytest.raises(scorer.ShardContractError, match="not a byte-identical capture"):
        scorer.write_shard(output_dir, shard_result)
    assert stale.read_bytes() == b'{"row_kind": "left over from an earlier tool"}\n'

    payload = scorer.write_quarantine(
        output_dir, image_id=IMAGE_ID, reason="OSError", detail="synthetic"
    )
    assert payload["evidence_usable"] is False
    assert payload["primary_rows_written"] is True
    assert payload["primary_output_residue"] == [scorer.SCORES_NAME]
    assert payload["residue_left_in_place"] is True
    assert payload["residue_origin"] == (
        "not_written_by_this_invocation_shard_publish_is_atomic"
    )
    # The pre-existing file is untouched.
    assert stale.read_bytes() == b'{"row_kind": "left over from an earlier tool"}\n'


def test_identical_rerun_is_a_no_op(shard_result, tmp_path: Path):
    output_dir = tmp_path / "runs" / "shard-6040"
    scorer.write_shard(output_dir, shard_result)
    before = {
        child.name: (child.read_bytes(), child.stat().st_mtime_ns)
        for child in output_dir.iterdir()
    }

    record = scorer.write_shard(output_dir, shard_result)
    assert record["published"] is False
    assert record["publish_mode"] == "no_op_identical_rerun"
    assert record["already_present_identical"] is True

    after = {
        child.name: (child.read_bytes(), child.stat().st_mtime_ns)
        for child in output_dir.iterdir()
    }
    assert after == before
    assert scorer.find_staging_residue(output_dir) == []


def test_divergent_existing_output_fails_closed(shard_result, tmp_path: Path):
    output_dir = tmp_path / "runs" / "shard-6040"
    scorer.write_shard(output_dir, shard_result)
    target = output_dir / scorer.SCORES_NAME
    tampered = target.read_bytes() + b'{"row_kind": "smuggled"}\n'
    target.write_bytes(tampered)

    with pytest.raises(scorer.ShardContractError, match="left untouched"):
        scorer.write_shard(output_dir, shard_result)
    assert target.read_bytes() == tampered

    # Missing a file is equally foreign.
    (output_dir / scorer.X1_NAME).unlink()
    with pytest.raises(scorer.ShardContractError, match="not a byte-identical capture"):
        scorer.write_shard(output_dir, shard_result)


def test_stale_quarantine_blocks_a_silent_republish(shard_result, tmp_path: Path):
    """A prior quarantine record must not be erased by a later success."""

    output_dir = tmp_path / "runs" / "shard-6040"
    scorer.write_quarantine(
        output_dir, image_id=IMAGE_ID, reason="ShardContractError", detail="earlier failure"
    )
    quarantine_bytes = (output_dir / scorer.QUARANTINE_NAME).read_bytes()

    with pytest.raises(scorer.ShardContractError, match="fresh output directory"):
        scorer.write_shard(output_dir, shard_result)
    assert (output_dir / scorer.QUARANTINE_NAME).read_bytes() == quarantine_bytes


def test_staging_cleanup_refuses_to_delete_anything_it_does_not_own(tmp_path: Path):
    foreign = tmp_path / "operator-data"
    foreign.mkdir()
    (foreign / "notes.md").write_text("do not delete me")
    with pytest.raises(scorer.ShardContractError, match="not a shard staging directory"):
        scorer._remove_owned_staging(foreign, expected_names=(scorer.SCORES_NAME,))
    assert (foreign / "notes.md").read_text() == "do not delete me"

    staging = tmp_path / f"shard-6040{scorer.STAGING_DIR_PREFIX}1-abc"
    staging.mkdir()
    (staging / scorer.SCORES_NAME).write_bytes(b"")
    (staging / "unexpected.bin").write_bytes(b"")
    with pytest.raises(scorer.ShardContractError, match="unexpected entries"):
        scorer._remove_owned_staging(staging, expected_names=(scorer.SCORES_NAME,))
    assert (staging / "unexpected.bin").exists()


def test_cli_publishes_atomically(plan_dir: Path, tmp_path: Path, capsys):
    output_dir = tmp_path / "runs" / "shard-6040"
    exit_code = scorer.main(
        [
            "--plan-dir",
            str(plan_dir),
            "--image-id",
            IMAGE_ID,
            "--backend",
            "fake",
            "--output-dir",
            str(output_dir),
        ]
    )
    assert exit_code == 0
    capsys.readouterr()
    assert sorted(child.name for child in output_dir.iterdir()) == sorted(
        [
            scorer.SCORES_NAME,
            scorer.X1_NAME,
            scorer.PROPOSAL_NAME,
            scorer.FREE_DECODE_NAME,
            scorer.RECEIPT_NAME,
        ]
    )
    assert scorer.find_staging_residue(output_dir) == []


# ---------------------------------------------------------------------------
# Candidate mini-batching
# ---------------------------------------------------------------------------


def _rows_by_candidate(rows):
    return {row["candidate_id"]: row for row in rows}


def test_candidate_batch_size_one_is_the_default_and_preserves_bytes(
    plan, tmp_path: Path
):
    """The frozen default must produce byte-identical output to before."""

    assert scorer.DEFAULT_CANDIDATE_BATCH_SIZE == 1
    baseline = scorer.run_shard(
        plan, image_id=IMAGE_ID, backend=scorer.FakeCensusBackend(), output_dir=None
    )
    explicit = scorer.run_shard(
        plan,
        image_id=IMAGE_ID,
        backend=scorer.FakeCensusBackend(),
        output_dir=None,
        candidate_batch_size=1,
    )
    assert scorer.shard_output_files(baseline)[scorer.SCORES_NAME] == (
        scorer.shard_output_files(explicit)[scorer.SCORES_NAME]
    )
    granularity = explicit.receipt["granularity"]
    assert granularity["candidate_batch_size"] == 1
    assert granularity["candidate_batching_enabled"] is False
    assert granularity["bytes_identical_to_candidate_batch_size_one"] is True
    assert granularity["bulk_scoring_path"] == "admitted_kv_cache_single_candidate_branch"
    # No batched branch is ever opened at the default.
    assert explicit.receipt["numerics"]["batched_path_parity"] is None


@pytest.mark.parametrize("batch_size", [2, 3, 8])
def test_batched_scoring_is_numerically_identical_to_single_branch(
    plan, batch_size: int
):
    """Batch>1 must reproduce batch=1 exactly on the deterministic backend."""

    single = scorer.run_shard(
        plan, image_id=IMAGE_ID, backend=scorer.FakeCensusBackend(), output_dir=None
    )
    batched = scorer.run_shard(
        plan,
        image_id=IMAGE_ID,
        backend=scorer.FakeCensusBackend(),
        output_dir=None,
        candidate_batch_size=batch_size,
    )

    assert [row["candidate_id"] for row in batched.scores] == [
        row["candidate_id"] for row in single.scores
    ]
    left = _rows_by_candidate(single.scores)
    right = _rows_by_candidate(batched.scores)
    assert set(left) == set(right)
    for candidate_id, one in left.items():
        other = right[candidate_id]
        assert other["complete_box_logprob_sum"] == one["complete_box_logprob_sum"]
        assert other["per_coordinate"] == one["per_coordinate"]
        assert other["box_end"] == one["box_end"]
        assert other["competition"] == one["competition"]
        # Spot-check selection is digest-based, so most candidates carry None;
        # the selection itself and any selected value must both match.
        assert (other["scalar_reference"] is None) == (one["scalar_reference"] is None)
        if one["scalar_reference"] is not None:
            assert other["scalar_reference"]["complete_box_logprob_sum"] == (
                one["scalar_reference"]["complete_box_logprob_sum"]
            )
            assert other["scalar_reference"]["selection"] == one["scalar_reference"]["selection"]

    # Everything outside the box channel is untouched by the knob.
    assert scorer.shard_output_files(batched)[scorer.PROPOSAL_NAME] == (
        scorer.shard_output_files(single)[scorer.PROPOSAL_NAME]
    )
    assert scorer.shard_output_files(batched)[scorer.FREE_DECODE_NAME] == (
        scorer.shard_output_files(single)[scorer.FREE_DECODE_NAME]
    )

    # The x1 diagnostic carries the batch size truthfully, so its bytes differ
    # by exactly that field and nothing else -- including the 1000-bin curve.
    for one, other in zip(single.x1, batched.x1, strict=True):
        assert other["x1_logprobs"] == one["x1_logprobs"]
        assert other["checks"]["candidate_batch_size"] == batch_size
        assert one["checks"]["candidate_batch_size"] == 1
        assert other["checks"]["candidate_batching_enabled"] is True
        assert one["checks"]["candidate_batching_enabled"] is False
        stripped = {
            key: {k: v for k, v in value.items()
                  if k not in {"candidate_batch_size", "candidate_batching_enabled"}}
            if key == "checks"
            else value
            for key, value in other.items()
        }
        expected = {
            key: {k: v for k, v in value.items()
                  if k not in {"candidate_batch_size", "candidate_batching_enabled"}}
            if key == "checks"
            else value
            for key, value in one.items()
        }
        assert stripped == expected


def test_tail_batch_is_scored_and_widths_are_exact(plan, backend):
    """A group of 2 at batch 3, and a group of 3 at batch 2, both complete."""

    guard = scorer.PhaseGuard()
    item = scorer.resolve_work_item(plan, f"{IMAGE_ID}:boundary-000|person")
    assert len(item.candidates) == 2
    admission = scorer.run_box_channel_admission(backend, item, guard=guard)

    backend.batched_branch_widths.clear()
    rows, _diag, _comp = scorer.score_query_group(
        backend, plan, item, guard=guard, admission=admission, scalar_modulus=1,
        candidate_batch_size=3,
    )
    assert len(rows) == 2
    # Two candidates at width 3 -> one short lane group, never a padded lane.
    assert backend.batched_branch_widths == [2]

    # Three candidates at width 2 -> a full batch then a tail of one.
    wide = scorer.WorkItem(
        **{**item.__dict__, "candidates": [*item.candidates, item.candidates[0]]}
    )
    object.__setattr__(
        wide,
        "candidates",
        [
            item.candidates[0],
            item.candidates[1],
            {**item.candidates[0], "candidate_id": "cand:tail", "coord_token_ids": [
                scorer.COORD_TOKEN_START + 5,
                scorer.COORD_TOKEN_START + 6,
                scorer.COORD_TOKEN_START + 7,
                scorer.COORD_TOKEN_START + 8,
            ]},
        ],
    )
    backend.batched_branch_widths.clear()
    tail_rows, _d, _c = scorer.score_query_group(
        backend, plan, wide, guard=guard, admission=admission, scalar_modulus=1,
        candidate_batch_size=2,
    )
    assert len(tail_rows) == 3
    assert backend.batched_branch_widths == [2, 1]
    assert [row["candidate_id"] for row in tail_rows] == [
        row["candidate_id"] for row in wide.candidates
    ]


def test_batched_lanes_do_not_contaminate_each_other(plan, backend):
    """Each lane's score must depend only on its own coordinate tokens."""

    guard = scorer.PhaseGuard()
    item = scorer.resolve_work_item(plan, f"{IMAGE_ID}:boundary-000|person")
    admission = scorer.run_box_channel_admission(backend, item, guard=guard)

    together, _d, _c = scorer.score_query_group(
        backend, plan, item, guard=guard, admission=admission, scalar_modulus=1,
        candidate_batch_size=2,
    )
    # Score each candidate as the sole occupant of its own batch.
    alone: dict[str, float] = {}
    for candidate in item.candidates:
        solo = scorer.WorkItem(**{**item.__dict__, "candidates": [candidate]})
        rows, _dd, _cc = scorer.score_query_group(
            backend, plan, solo, guard=guard, admission=admission, scalar_modulus=1,
            candidate_batch_size=2,
        )
        alone[rows[0]["candidate_id"]] = rows[0]["complete_box_logprob_sum"]

    for row in together:
        assert row["complete_box_logprob_sum"] == alone[row["candidate_id"]], (
            "a lane's score changed when another candidate shared the batch"
        )


def test_batched_path_parity_probe_uses_real_width_and_distinct_lanes(plan, backend):
    item = scorer.resolve_work_item(plan, f"{IMAGE_ID}:boundary-000|person")
    parity = scorer.run_batched_path_parity(backend, item, candidate_batch_size=8)
    assert parity["admitted"] is True
    assert parity["bitwise_identical"] is True
    assert parity["lane_width"] == 2  # the group only has two candidates
    assert parity["lanes_carry_distinct_tokens"] is True
    assert parity["distinct_coordinate_tuple_count"] == 2
    assert parity["max_abs_diff_bound"] == scorer.BATCHED_PATH_MAX_ABS_DIFF == 1e-4
    assert parity["on_failure"] == "fail_closed_no_silent_path_fallback"
    assert len(parity["per_candidate"]) == 2


def test_shard_refuses_to_capture_when_batched_parity_fails(plan, monkeypatch):
    real = scorer.run_batched_path_parity

    def diverging(*args, **kwargs):
        return {**real(*args, **kwargs), "admitted": False, "max_abs_diff_observed": 0.5}

    monkeypatch.setattr(scorer, "run_batched_path_parity", diverging)
    with pytest.raises(scorer.ShardContractError, match="no silent fallback"):
        scorer.run_shard(
            plan,
            image_id=IMAGE_ID,
            backend=scorer.FakeCensusBackend(),
            output_dir=None,
            candidate_batch_size=8,
        )


def test_batched_receipt_is_truthful(plan):
    result = scorer.run_shard(
        plan,
        image_id=IMAGE_ID,
        backend=scorer.FakeCensusBackend(),
        output_dir=None,
        candidate_batch_size=8,
    )
    granularity = result.receipt["granularity"]
    assert granularity["candidate_batch_size"] == 8
    assert granularity["candidate_batching_enabled"] is True
    assert granularity["bytes_identical_to_candidate_batch_size_one"] is False
    assert granularity["bulk_scoring_path"] == "admitted_kv_cache_batched_candidate_lanes"
    assert granularity["candidate_batch_scope"] == (
        "one_exact_query_group_and_admitted_prefix_only"
    )
    assert granularity["out_of_memory_policy"] == (
        "fail_loud_quarantine_never_silent_fallback"
    )
    # The uncached reforward is a diagnostic, never a fallback bulk path.
    assert granularity["batched_full_reforward_role"] == (
        "parity_diagnostic_only_never_fallback"
    )

    numerics = result.receipt["numerics"]
    assert numerics["batched_path_parity"]["admitted"] is True
    assert numerics["batched_path_max_abs_diff_bound"] == 1e-4
    precision = numerics["matmul_precision"]
    assert precision["torch_available"] is True
    assert "float32_matmul_precision" in precision
    assert "cuda_matmul_allow_tf32" in precision

    invariants = result.receipt["runtime_invariants"]
    assert invariants["explicit_position_ids_shape"] == (
        "3_by_batch_by_time_from_prefill_rope_deltas"
    )
    assert invariants["shared_model_rope_deltas_recomputation_used"] is False
    assert invariants["candidate_lane_kv_storage"] == (
        "per_lane_materialized_never_stride_zero"
    )
    assert invariants["group_root_cache_never_expanded_in_place"] is True


def test_invalid_candidate_batch_size_fails_closed(plan, backend):
    with pytest.raises(scorer.ShardContractError, match="at least one"):
        scorer.run_shard(
            plan, image_id=IMAGE_ID, backend=backend, output_dir=None,
            candidate_batch_size=0,
        )


def test_cli_exposes_candidate_batch_size_defaulting_to_one(plan_dir: Path, tmp_path: Path, capsys):
    output_dir = tmp_path / "runs" / "bs8"
    exit_code = scorer.main(
        [
            "--plan-dir", str(plan_dir), "--image-id", IMAGE_ID,
            "--backend", "fake", "--output-dir", str(output_dir),
            "--candidate-batch-size", "8",
        ]
    )
    assert exit_code == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["granularity"]["candidate_batch_size"] == 8

    parsed = scorer._parse_args(["--plan-dir", "p", "--image-id", "x"])
    assert parsed.candidate_batch_size == 1


# --- real-HF lane mechanics (no GPU) ---------------------------------------


def test_expanded_cache_lanes_own_their_storage_and_leave_the_root_alone():
    from transformers.cache_utils import DynamicCache

    root = DynamicCache()
    for layer in range(3):
        root.update(torch.randn(1, 2, 6, 4), torch.randn(1, 2, 6, 4), layer)
    root_shapes = scorer._cache_layer_shapes(root)

    expanded = scorer.expand_cache_for_lanes(root.cache if hasattr(root, "cache") else root, 8)
    for layer in expanded.layers:
        for tensor in (layer.keys, layer.values):
            assert tensor.shape[0] == 8
            assert 0 not in tensor.stride(), "stride-0 lanes would share KV storage"
            assert tensor.untyped_storage().nbytes() >= tensor.numel() * tensor.element_size()
    # Lanes are independent memory: writing one must not move another.
    expanded.layers[0].keys[0, 0, 0, 0] = 4242.0
    assert float(expanded.layers[0].keys[1, 0, 0, 0]) != 4242.0
    # The root is untouched in shape and content.
    assert scorer._cache_layer_shapes(root) == root_shapes
    assert float(root.layers[0].keys[0, 0, 0, 0]) != 4242.0


def test_batched_branch_positions_are_explicit_3_by_batch_by_time(plan):
    model = _RecordingQwenModel()
    backend = scorer.HFCensusBackend(
        model=model,
        native_prompt_inputs=_native_inputs(PROMPT_TOKEN_IDS),
        prompt_token_ids=PROMPT_TOKEN_IDS,
        identity={"backend": "hf-recording"},
        expected_layer_count=model.layers,
    )
    item = scorer.resolve_work_item(plan, f"{IMAGE_ID}:boundary-000|person")
    prefill = backend.prefill(item.query_prefix_token_ids)
    try:
        root_shapes = scorer._cache_layer_shapes(prefill.cache_backend.cache)
        with prefill.batched_branch(4) as lanes:
            model.calls.clear()
            stepped = lanes.step([[scorer.COORD_TOKEN_START + i] for i in range(4)])
        assert stepped.shape[0] == 4
        call = model.calls[-1]
        positions = call["position_ids"]
        assert positions is not None
        assert tuple(positions.shape) == (3, 4, 1)
        # Every lane sits at the same depth, so positions replicate exactly.
        assert torch.equal(positions[0, 0], positions[0, 3])
        assert 0 not in positions.stride()
        # The group root cache is still the admitted batch-1 root.
        prefill.assert_rooted(label="root after batched branch")
        assert scorer._cache_layer_shapes(prefill.cache_backend.cache) == root_shapes
        assert all(shape[0][0] == 1 for shape in root_shapes)
    finally:
        prefill.close()


def test_real_seam_batched_and_single_paths_agree(plan):
    """End-to-end on the real seams: batch=4 must match batch=1 tightly."""

    def build():
        model = _RecordingQwenModel()
        return model, scorer.HFCensusBackend(
            model=model,
            native_prompt_inputs=_native_inputs(PROMPT_TOKEN_IDS),
            prompt_token_ids=PROMPT_TOKEN_IDS,
            identity={"backend": "hf-recording"},
            expected_layer_count=model.layers,
        )

    _m1, single_backend = build()
    single = scorer.run_shard(plan, image_id=IMAGE_ID, backend=single_backend, output_dir=None)
    model, batched_backend = build()
    batched = scorer.run_shard(
        plan, image_id=IMAGE_ID, backend=batched_backend, output_dir=None,
        candidate_batch_size=4,
    )
    assert model.generate_calls == 0
    for call in model.calls:
        assert call["position_ids_is_none"] is False

    left = _rows_by_candidate(single.scores)
    right = _rows_by_candidate(batched.scores)
    assert set(left) == set(right)
    worst = max(
        abs(left[cid]["complete_box_logprob_sum"] - right[cid]["complete_box_logprob_sum"])
        for cid in left
    )
    assert worst <= scorer.BATCHED_PATH_MAX_ABS_DIFF
    assert batched.receipt["numerics"]["batched_path_parity"]["admitted"] is True
