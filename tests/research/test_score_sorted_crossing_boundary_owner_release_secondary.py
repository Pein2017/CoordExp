"""Contracts for the secondary downstream-compatibility capture
(``scripts/research/score_sorted_crossing_boundary_owner_release_secondary.py``)
of the frozen unit:

``research/investigations/qwen3-vl-dense-enumeration/experiments/
2026-08-03-sorted-crossing-boundary-owner-release-realization/unit.md``

Discipline mirrored from the primary scorer's own test file:

- no GPU, no real model or tokenizer.  Every runtime exercise runs on the
  deterministic ``FakeCensusBackend`` whose logits are a pure function of the
  literal token sequence, which is what makes the cached-versus-uncached parity
  gate meaningful rather than vacuous;
- literal token ids and sha256-over-canonical-JSON digests only, never decoded
  prose;
- fail-closed assertions state the contract, not the implementation.

The 64-request sealed plan is built once by :func:`_synthetic_plan` and shared
by every test that needs the whole census; the runtime tests then narrow to one
image that carries all three sealed variants in three requests, so no test pays
for a 26-owner capture.
"""

from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pytest

from scripts.research import (
    analyze_sorted_crossing_boundary_owner_release as analyzer,
)
from scripts.research import (
    prepare_sorted_crossing_boundary_owner_release_realization as plan_builder,
)
from scripts.research import score_sorted_crossing_boundary_owner_release as primary
from scripts.research import (
    score_sorted_crossing_boundary_owner_release_secondary as sut,
)
from scripts.research import score_sorted_owner_accessibility_census_shard as census_shard


OBJECT_REF_START = 151646
OBJECT_REF_END = 151647
BOX_START = 151648
BOX_END = 151649
COORD_START = 151670
COORD_END_INCLUSIVE = 152669

FAKE_BACKEND_IDENTITY = {"backend": "fake", "usable_as_evidence": False}
REAL_BACKEND_IDENTITY = {"backend": "hf", "usable_as_evidence": True}

#: The retired dual-schema revision.  Named only so the tests can prove it is
#: now rejected; no production constant admits it.
RETIRED_REPORT_SCHEMA_V1 = "sorted-crossing-boundary-owner-release-report.v1"
RETIRED_OWNER_ROW_SCHEMA_V1 = "sorted-crossing-boundary-owner-release-owner-row.v1"


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


# ---------------------------------------------------------------------------
# 0. Frozen constants
# ---------------------------------------------------------------------------


def test_unit_tier_family_and_artifact_names_are_frozen() -> None:
    assert sut.UNIT_ID == "2026-08-03-sorted-crossing-boundary-owner-release-realization"
    assert sut.SECONDARY_READOUT_TIER == "secondary_sealed_after_primary_branches"
    assert sut.SECONDARY_READOUT_TIER != primary.PRIMARY_READOUT_TIER
    assert sut.REQUEST_FAMILY == plan_builder.REQUEST_DOWNSTREAM_COMPATIBILITY
    assert sut.APPENDED_ROLE == "inserted_exact_clean_gt_row_c"
    assert sut.ROWS_NAME == "secondary-compatibility-rows.jsonl"
    assert sut.RECEIPT_NAME == "secondary-compatibility-receipt.json"
    assert sut.PARITY_NAME == "secondary-compatibility-parity.json"
    assert sut.EVIDENCE_OUTPUT_NAMES == (sut.ROWS_NAME, sut.PARITY_NAME)


def test_frozen_secondary_request_census_is_26_26_12() -> None:
    assert sut.SECONDARY_VARIANTS == (
        "p_plus_c_then_e",
        "p_plus_e_plus_c_then_f",
        "benign_substitution_then_following_native_action",
    )
    assert dict(sut.EXPECTED_REQUEST_COUNT_BY_VARIANT) == {
        "p_plus_c_then_e": 26,
        "p_plus_e_plus_c_then_f": 26,
        "benign_substitution_then_following_native_action": 12,
    }
    assert sut.EXPECTED_SECONDARY_REQUEST_COUNT == 64


def test_variant_cohort_and_optionality_are_frozen() -> None:
    assert dict(sut.VARIANT_COHORT) == {
        "p_plus_c_then_e": plan_builder.PRIMARY_COHORT,
        "p_plus_e_plus_c_then_f": plan_builder.PRIMARY_COHORT,
        "benign_substitution_then_following_native_action": (
            plan_builder.TP_REPLAY_CONTROL_COHORT
        ),
    }
    # Only the late catch-up F extension is plan-optional.
    assert dict(sut.VARIANT_IS_PLAN_OPTIONAL) == {
        "p_plus_c_then_e": False,
        "p_plus_e_plus_c_then_f": True,
        "benign_substitution_then_following_native_action": False,
    }


def test_reported_segments_roots_and_claim_boundary() -> None:
    assert sut.SEGMENTS == ("description", "coordinates", "complete_row")
    assert sut.PAIRED_ROOTS == (sut.ROOT_BASELINE, sut.ROOT_MODIFIED)
    assert sut.CACHE_PARITY_MAX_SELECTED_LOGIT_ABS_DIFF == 1e-3
    assert "never final-set retention" in sut.CLAIM_BOUNDARY
    assert "free-rollout" in sut.CLAIM_BOUNDARY


# ---------------------------------------------------------------------------
# 1. Bounded synthetic plan (one 64-request census, reused everywhere)
# ---------------------------------------------------------------------------


class _StubCensus:
    def __init__(self, contexts: dict[str, Any]) -> None:
        self.contexts_by_id = contexts


class _StubInputs:
    def __init__(self, contexts: dict[str, Any], images: dict[str, Any]) -> None:
        self.census = _StubCensus(contexts)
        self.images_by_id = images


def _row_tokens(description: list[int], coordinates: list[int]) -> list[int]:
    return [OBJECT_REF_START, *description, OBJECT_REF_END, BOX_START, *coordinates, BOX_END]


def _coords(seed: int) -> list[int]:
    span = COORD_END_INCLUSIVE - COORD_START + 1
    return [COORD_START + (seed * (index + 7) * 13) % span for index in range(4)]


def _description(seed: int) -> list[int]:
    return [20000 + seed % 97] * (1 + seed % 3)


#: One primary owner per boundary, twelve images, 26 primary owners and twelve
#: TP replay controls -- exactly the sealed 26/26/12 secondary census.
_IMAGE_IDS: tuple[str, ...] = tuple(f"img-{index:02d}" for index in range(12))
_PRIMARY_OWNERS_PER_IMAGE: tuple[int, ...] = (1, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2)
_PRIMARY_BOUNDARIES: tuple[int, ...] = (3, 6, 9)
_ROWS_PER_IMAGE = 11


def _make_request(
    *,
    cohort: str,
    gt_owner_id: str,
    image_id: str,
    context_id: str,
    context_role: str,
    boundary_index: int,
    appended_token_ids: list[int],
    base_prefix_token_ids: list[int],
    scored_target: dict[str, Any],
    variant: str,
    optional: bool,
) -> dict[str, Any]:
    """Reproduce the sealed request identity the plan builder publishes."""

    base_digest = _sha256_json(base_prefix_token_ids)
    identity = {
        "unit_id": sut.UNIT_ID,
        "request_family": sut.REQUEST_FAMILY,
        "cohort": cohort,
        "gt_owner_id": gt_owner_id,
        "context_id": context_id,
        "variant": variant,
        "appended_token_ids": list(appended_token_ids),
        "base_prefix_token_ids_sha256": base_digest,
        "scored_target": dict(scored_target),
        "decode": None,
        "candidate_family": None,
    }
    digest = _sha256_json(identity)
    return {
        "schema_version": plan_builder.REQUEST_SCHEMA_VERSION,
        "row_kind": "crossing_boundary_request",
        "unit_id": sut.UNIT_ID,
        "branch_schema_id": plan_builder.BRANCH_SCHEMA_ID,
        "branch_inputs": [],
        "candidate_family": None,
        "cohort": cohort,
        "context_id": context_id,
        "context_role": context_role,
        "boundary_index": boundary_index,
        "decode": None,
        "gt_owner_id": gt_owner_id,
        "identity_digest": digest,
        "image_id": image_id,
        "inspects_new_model_logits": False,
        "optional": optional,
        "predecessor_query_group_id": None,
        "prefix": {
            "appended_role": sut.APPENDED_ROLE,
            "appended_token_count": len(appended_token_ids),
            "appended_token_ids": list(appended_token_ids),
            "appended_token_ids_sha256": _sha256_json(list(appended_token_ids)),
            "base_context_id": context_id,
            "base_prefix_token_count": len(base_prefix_token_ids),
            "base_prefix_token_ids_sha256": base_digest,
            "retokenized": False,
        },
        "readout_tier": sut.SECONDARY_READOUT_TIER,
        "request_family": sut.REQUEST_FAMILY,
        "request_id": f"req:{digest[:32]}",
        "request_key": f"{sut.REQUEST_FAMILY}|{cohort}|{gt_owner_id}|{context_id}|{variant}",
        "score_blind_plan": True,
        "scored_target": dict(scored_target),
        "variant": variant,
    }


def _sidecar(token_ids: list[int], *, row_index: int, pre: str, post: str) -> dict[str, Any]:
    return {
        "full_row_token_ids": list(token_ids),
        "full_row_token_ids_sha256": _sha256_json(list(token_ids)),
        "row_index": row_index,
        "pre_row_context_id": pre,
        "post_row_context_id": post,
        "stratum": plan_builder.STRATUM_MATCHED_E,
    }


def _synthetic_plan() -> primary.SealedPlan:
    """The whole sealed 26/26/12 secondary census, as plain dicts."""

    images: dict[str, Any] = {}
    contexts: dict[str, Any] = {}
    cohort_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []

    for image_index, image_id in enumerate(_IMAGE_IDS):
        images[image_id] = {"prompt_token_ids": [900 + image_index, 901 + image_index]}
        rows = [
            _row_tokens(
                _description(image_index * 31 + row_index),
                _coords(image_index * 17 + row_index + 1),
            )
            for row_index in range(_ROWS_PER_IMAGE)
        ]
        prefixes: list[list[int]] = [[]]
        for row in rows:
            prefixes.append(prefixes[-1] + row)
        for boundary_index, prefix in enumerate(prefixes):
            context_id = f"{image_id}:boundary-{boundary_index:03d}"
            contexts[context_id] = {
                "context_id": context_id,
                "image_id": image_id,
                "boundary_index": boundary_index,
                "context_role": "root" if boundary_index == 0 else "row_boundary",
                "generated_prefix_token_ids": list(prefix),
                "generated_prefix_token_ids_sha256": _sha256_json(list(prefix)),
            }

        def context(index: int, _image_id: str = image_id) -> str:
            return f"{_image_id}:boundary-{index:03d}"

        # --- 26 primary crossing owners ---------------------------------
        for slot in range(_PRIMARY_OWNERS_PER_IMAGE[image_index]):
            boundary = _PRIMARY_BOUNDARIES[slot]
            owner_id = f"gt:{image_id}:{boundary}"
            e_row = _sidecar(
                rows[boundary],
                row_index=boundary,
                pre=context(boundary),
                post=context(boundary + 1),
            )
            f_row = _sidecar(
                rows[boundary + 1],
                row_index=boundary + 1,
                pre=context(boundary + 1),
                post=context(boundary + 2),
            )
            inserted = _row_tokens(
                _description(image_index * 31 + boundary),
                _coords(1000 + image_index * 7 + boundary),
            )
            cohort_rows.append(
                {
                    "schema_version": plan_builder.COHORT_SCHEMA_VERSION,
                    "unit_id": sut.UNIT_ID,
                    "cohort": plan_builder.PRIMARY_COHORT,
                    "gt_owner_id": owner_id,
                    "image_id": image_id,
                    "normalized_description": f"thing-{boundary}",
                    "crossing": {
                        "p_context_id": context(boundary),
                        "pe_context_id": context(boundary + 1),
                    },
                    "e_row": e_row,
                    "f_row": f_row,
                    "f_row_present": True,
                    "inserted_clean_row_c": {
                        "token_ids": list(inserted),
                        "token_ids_sha256": _sha256_json(list(inserted)),
                    },
                }
            )
            for variant, ctx_index, target_row, optional in (
                ("p_plus_c_then_e", boundary, e_row, False),
                ("p_plus_e_plus_c_then_f", boundary + 1, f_row, True),
            ):
                requests.append(
                    _make_request(
                        cohort=plan_builder.PRIMARY_COHORT,
                        gt_owner_id=owner_id,
                        image_id=image_id,
                        context_id=context(ctx_index),
                        context_role="row_boundary",
                        boundary_index=ctx_index,
                        appended_token_ids=list(inserted),
                        base_prefix_token_ids=prefixes[ctx_index],
                        scored_target={
                            "kind": "exact_native_row",
                            "token_ids": list(target_row["full_row_token_ids"]),
                            "token_ids_sha256": target_row["full_row_token_ids_sha256"],
                            "native_row_index": target_row["row_index"],
                            "compare_against": (
                                "the same exact row scored at the unmodified native context"
                            ),
                            "report": [
                                "description_delta",
                                "coordinate_delta",
                                "complete_row_delta",
                            ],
                        },
                        variant=variant,
                        optional=optional,
                    )
                )

        # --- twelve TP replay controls (one per image) -------------------
        control_owner = f"gt:{image_id}:tp"
        twin = _row_tokens(_description(image_index * 31), _coords(2000 + image_index))
        following = {
            "context_id": context(1),
            "kind": "native_row",
            "row_index": 1,
            "token_ids": list(rows[1]),
            "token_ids_sha256": _sha256_json(list(rows[1])),
        }
        control_rows.append(
            {
                "schema_version": plan_builder.CONTROL_SCHEMA_VERSION,
                "unit_id": sut.UNIT_ID,
                "cohort": plan_builder.TP_REPLAY_CONTROL_COHORT,
                "gt_owner_id": control_owner,
                "image_id": image_id,
                "normalized_description": "tp",
                "due_context_id": context(0),
                "row_index": 0,
                "following_native_action": following,
                "inserted_clean_row_c": {
                    "token_ids": list(twin),
                    "token_ids_sha256": _sha256_json(list(twin)),
                },
            }
        )
        requests.append(
            _make_request(
                cohort=plan_builder.TP_REPLAY_CONTROL_COHORT,
                gt_owner_id=control_owner,
                image_id=image_id,
                context_id=context(0),
                context_role="root",
                boundary_index=0,
                appended_token_ids=list(twin),
                base_prefix_token_ids=prefixes[0],
                scored_target={
                    "kind": "native_row",
                    "token_ids": list(rows[1]),
                    "token_ids_sha256": _sha256_json(list(rows[1])),
                    "native_row_index": 1,
                    "compare_against": (
                        "the same exact action scored at the unmodified native context"
                    ),
                    "report": [
                        "description_delta",
                        "coordinate_delta",
                        "complete_row_delta",
                    ],
                },
                variant="benign_substitution_then_following_native_action",
                optional=False,
            )
        )

    manifest = {
        "schema_version": plan_builder.MANIFEST_SCHEMA_VERSION,
        "unit_id": sut.UNIT_ID,
        "manifest_content_sha256": "plan-manifest-digest",
        "builder_source": {"path": "builder.py", "sha256": "b" * 64},
        "cohort_counts": {
            "u_bound_crossing_count": 26,
            "l_bound_crossing_count": 25,
            "exact_same_context_u_and_l_count": 24,
            "matched_e_count": 12,
            "unmatched_e_count": 14,
            "f_row_present_count": 26,
        },
        "control_counts": {"timing_control_count": 14, "tp_replay_control_count": 12},
        "lineage": {"census_run_root": "/nonexistent/census"},
    }
    return primary.SealedPlan(
        plan_dir=Path("/nonexistent/plan"),
        manifest=manifest,
        cohort_rows=cohort_rows,
        control_rows=control_rows,
        request_rows=requests,
        plan_file_sha256={"request-plan.jsonl": "a" * 64},
        inputs=_StubInputs(contexts, images),
        candidates_by_id={},
        owners_by_image={},
    )


@pytest.fixture(scope="module")
def sealed_plan() -> primary.SealedPlan:
    return _synthetic_plan()


@pytest.fixture()
def plan(sealed_plan: primary.SealedPlan) -> primary.SealedPlan:
    """A deep-enough copy so a mutating test cannot leak into its neighbours."""

    return primary.SealedPlan(
        plan_dir=sealed_plan.plan_dir,
        manifest=copy.deepcopy(sealed_plan.manifest),
        cohort_rows=copy.deepcopy(sealed_plan.cohort_rows),
        control_rows=copy.deepcopy(sealed_plan.control_rows),
        request_rows=copy.deepcopy(sealed_plan.request_rows),
        plan_file_sha256=dict(sealed_plan.plan_file_sha256),
        inputs=_StubInputs(
            copy.deepcopy(sealed_plan.inputs.census.contexts_by_id),
            copy.deepcopy(sealed_plan.inputs.images_by_id),
        ),
        candidates_by_id={},
        owners_by_image={},
    )


#: The one image the runtime tests use: one primary owner (with F) and one TP
#: replay control, so a capture is three requests instead of a 26-owner sweep.
RUNTIME_IMAGE_ID = "img-00"


def _request_of(plan: primary.SealedPlan, *, image_id: str, variant: str) -> dict[str, Any]:
    for row in plan.request_rows:
        if str(row["image_id"]) == image_id and str(row["variant"]) == variant:
            return row
    raise AssertionError(f"no {variant!r} request on image {image_id!r}")


def _registry_of(plan: primary.SealedPlan, owner_id: str) -> dict[str, Any]:
    for row in (*plan.cohort_rows, *plan.control_rows):
        if str(row["gt_owner_id"]) == owner_id:
            return row
    raise AssertionError(f"no registry row for {owner_id!r}")


# ---------------------------------------------------------------------------
# 2. The 64-request sealed census
# ---------------------------------------------------------------------------


def test_sealed_plan_holds_exactly_the_frozen_64_secondary_requests(
    plan: primary.SealedPlan,
) -> None:
    rows = sut.secondary_request_rows(plan)
    counts = sut.validate_secondary_plan_counts(rows, manifest=plan.manifest)
    assert counts["observed_total"] == 64
    assert counts["observed_by_variant"] == dict(sut.EXPECTED_REQUEST_COUNT_BY_VARIANT)
    assert counts["expected_by_variant"] == counts["observed_by_variant"]


@pytest.mark.parametrize("variant", sut.SECONDARY_VARIANTS)
def test_a_missing_secondary_request_of_any_variant_fails_closed(
    plan: primary.SealedPlan, variant: str
) -> None:
    rows = [row for row in sut.secondary_request_rows(plan) if row["variant"] != variant]
    with pytest.raises(sut.SecondaryCompatibilityContractError) as excinfo:
        sut.validate_secondary_plan_counts(rows, manifest=plan.manifest)
    assert variant in str(excinfo.value)


def test_a_materialized_optional_f_without_a_sealed_f_row_fails_closed(
    plan: primary.SealedPlan,
) -> None:
    # The plan seals 25 owners with an F row but materialised 26 optional
    # requests: an optional readout is never materialised without its row.
    plan.manifest["cohort_counts"]["f_row_present_count"] = 25
    rows = sut.secondary_request_rows(plan)
    with pytest.raises(sut.SecondaryCompatibilityContractError) as excinfo:
        sut.validate_secondary_plan_counts(rows, manifest=plan.manifest)
    assert "never materialized without its row" in str(excinfo.value)


def test_a_duplicated_secondary_request_fails_closed(plan: primary.SealedPlan) -> None:
    plan.request_rows.append(copy.deepcopy(plan.request_rows[0]))
    with pytest.raises(sut.SecondaryCompatibilityContractError) as excinfo:
        sut.secondary_request_rows(plan)
    assert "duplicate secondary request" in str(excinfo.value)


def test_primary_tier_requests_are_never_executed_by_this_pass(
    plan: primary.SealedPlan,
) -> None:
    plan.request_rows.append(
        {
            **copy.deepcopy(plan.request_rows[0]),
            "readout_tier": primary.PRIMARY_READOUT_TIER,
            "request_id": "req:primary-tier",
        }
    )
    rows = sut.secondary_request_rows(plan)
    assert len(rows) == 64
    assert all(row["readout_tier"] == sut.SECONDARY_READOUT_TIER for row in rows)


@pytest.mark.parametrize(
    ("mutation", "fragment"),
    [
        ({"readout_tier": "primary"}, "not 'secondary_sealed_after_primary_branches'"),
        ({"request_family": "coordinate_greedy_row"}, "is family"),
        ({"variant": "p_plus_c_then_g"}, "unknown secondary variant"),
        ({"cohort": plan_builder.TP_REPLAY_CONTROL_COHORT}, "declares cohort"),
        ({"optional": True}, "declares optional="),
        ({"branch_inputs": ["displaced"]}, "never feeds a primary branch"),
        ({"unit_id": "another-unit"}, "belongs to another unit"),
    ],
)
def test_assert_secondary_request_fails_closed_on_a_drifted_contract(
    plan: primary.SealedPlan, mutation: dict[str, Any], fragment: str
) -> None:
    row = copy.deepcopy(_request_of(plan, image_id=RUNTIME_IMAGE_ID, variant="p_plus_c_then_e"))
    row.update(mutation)
    with pytest.raises(sut.SecondaryCompatibilityContractError) as excinfo:
        sut.assert_secondary_request(row)
    assert fragment in str(excinfo.value)


@pytest.mark.parametrize(
    ("prefix_mutation", "fragment"),
    [
        ({"appended_role": "forced_target_description_path_d_c"}, "appends role"),
        ({"retokenized": True}, "retokenized prefix"),
    ],
)
def test_assert_secondary_request_fails_closed_on_a_drifted_prefix(
    plan: primary.SealedPlan, prefix_mutation: dict[str, Any], fragment: str
) -> None:
    row = copy.deepcopy(_request_of(plan, image_id=RUNTIME_IMAGE_ID, variant="p_plus_c_then_e"))
    row["prefix"] = {**row["prefix"], **prefix_mutation}
    with pytest.raises(sut.SecondaryCompatibilityContractError) as excinfo:
        sut.assert_secondary_request(row)
    assert fragment in str(excinfo.value)


def test_one_image_selection_is_score_blind_and_covers_every_sealed_variant(
    plan: primary.SealedPlan,
) -> None:
    rows = sut.secondary_request_rows(plan)
    selected = sut.secondary_requests_for_image(rows, image_id=RUNTIME_IMAGE_ID)
    assert {str(row["image_id"]) for row in selected} == {RUNTIME_IMAGE_ID}
    assert sut.counts_by_variant(selected) == {
        "p_plus_c_then_e": 1,
        "p_plus_e_plus_c_then_f": 1,
        "benign_substitution_then_following_native_action": 1,
    }
    # Stable order, derived from the sealed request identity alone.
    assert [row["request_id"] for row in selected] == sorted(
        row["request_id"] for row in selected
    )


def test_an_uncovered_image_fails_closed_and_names_the_covered_ones(
    plan: primary.SealedPlan,
) -> None:
    rows = sut.secondary_request_rows(plan)
    with pytest.raises(sut.SecondaryCompatibilityContractError) as excinfo:
        sut.secondary_requests_for_image(rows, image_id="img-99")
    assert RUNTIME_IMAGE_ID in str(excinfo.value)


# ---------------------------------------------------------------------------
# 3. Exact row grammar, segment sums and paired deltas
# ---------------------------------------------------------------------------


def _valid_row() -> list[int]:
    return _row_tokens([20001, 20002], [151700, 151800, 151900, 152000])


def test_row_segments_use_the_rows_own_wrapper_boundaries() -> None:
    tokens = _valid_row()
    segments = sut.row_segments(tokens)
    assert segments.box_start_index == 4
    assert segments.description == (0, 5)
    assert segments.coordinates == (5, 9)
    assert segments.complete_row == (0, 10)
    # The description path is the primary scorer's own literal definition.
    assert tokens[slice(*segments.description)] == primary._description_path(tokens)
    assert tokens[slice(*segments.coordinates)] == [151700, 151800, 151900, 152000]
    assert tokens[segments.complete_row[1] - 1] == BOX_END


def test_row_segments_are_immutable() -> None:
    segments = sut.row_segments(_valid_row())
    with pytest.raises(FrozenInstanceError):
        segments.description = (0, 1)  # type: ignore[misc]


@pytest.mark.parametrize(
    ("tokens", "fragment"),
    [
        ([], "cannot be empty"),
        ([99, 20001, OBJECT_REF_END, BOX_START, 151700, 151800, 151900, 152000, BOX_END],
         "must open with <|object_ref_start|>"),
        ([OBJECT_REF_START, 20001, OBJECT_REF_END, 151700, 151800, 151900, 152000, BOX_END],
         "must carry <|box_start|>"),
        ([OBJECT_REF_START, 20001, BOX_START, 151700, 151800, 151900, 152000, BOX_END],
         "must close its description with <|object_ref_end|>"),
        ([OBJECT_REF_START, 20001, OBJECT_REF_END, BOX_START, 151700, 151800, 151900, BOX_END],
         "exactly 4 coordinate tokens"),
        ([OBJECT_REF_START, 20001, OBJECT_REF_END, BOX_START, 151700, 151800, 151900, 42,
          BOX_END], "out-of-domain token"),
        ([OBJECT_REF_START, 20001, OBJECT_REF_END, BOX_START, 151700, 151800, 151900, 152000,
          77], "must terminate with <|box_end|>"),
    ],
)
def test_row_segments_fail_closed_on_a_malformed_row(
    tokens: list[int], fragment: str
) -> None:
    with pytest.raises(sut.SecondaryCompatibilityContractError) as excinfo:
        sut.row_segments(tokens)
    assert fragment in str(excinfo.value)


def test_segment_sums_are_exact_over_the_literal_spans() -> None:
    tokens = _valid_row()
    segments = sut.row_segments(tokens)
    logprobs = [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0]
    sums = sut.segment_sums(segments, logprobs, label="probe")
    assert sums["description"]["sum"] == pytest.approx(-15.0)
    assert sums["description"]["token_count"] == 5
    assert sums["coordinates"]["sum"] == pytest.approx(-30.0)
    assert sums["coordinates"]["token_count"] == 4
    assert sums["complete_row"]["sum"] == pytest.approx(-55.0)
    assert sums["complete_row"]["token_count"] == 10
    assert sums["coordinates"]["token_mean"] == pytest.approx(-7.5)


def test_segment_sums_reject_a_length_mismatch_and_a_nonfinite_score() -> None:
    segments = sut.row_segments(_valid_row())
    with pytest.raises(sut.SecondaryCompatibilityContractError) as excinfo:
        sut.segment_sums(segments, [-1.0] * 9, label="probe")
    assert "do not cover the row's" in str(excinfo.value)

    values = [-1.0] * 10
    values[6] = -math.inf
    with pytest.raises(sut.SecondaryCompatibilityContractError) as excinfo:
        sut.segment_sums(segments, values, label="probe")
    assert "non-finite selected logprob" in str(excinfo.value)


def test_paired_deltas_are_modified_minus_baseline_with_exact_signs() -> None:
    segments = sut.row_segments(_valid_row())
    baseline = sut.segment_sums(segments, [-2.0] * 10, label="baseline")
    modified = sut.segment_sums(
        segments, [-1.0] * 5 + [-3.0] * 4 + [-2.0], label="modified"
    )
    deltas = sut.paired_deltas(baseline, modified)
    assert deltas["description"]["delta"] == pytest.approx(5.0)
    assert deltas["description"]["sign"] == 1
    assert deltas["coordinates"]["delta"] == pytest.approx(-4.0)
    assert deltas["coordinates"]["sign"] == -1
    assert deltas["complete_row"]["delta"] == pytest.approx(1.0)
    assert deltas["complete_row"]["delta_token_mean"] == pytest.approx(0.1)
    assert deltas["complete_row"]["baseline_sum"] == pytest.approx(-20.0)
    assert deltas["complete_row"]["modified_sum"] == pytest.approx(-19.0)


def test_paired_deltas_refuse_two_different_token_spans() -> None:
    long_segments = sut.row_segments(_valid_row())
    short_segments = sut.row_segments(
        _row_tokens([20001], [151700, 151800, 151900, 152000])
    )
    baseline = sut.segment_sums(long_segments, [-1.0] * 10, label="baseline")
    modified = sut.segment_sums(short_segments, [-1.0] * 9, label="modified")
    with pytest.raises(sut.SecondaryCompatibilityContractError) as excinfo:
        sut.paired_deltas(baseline, modified)
    assert "same token span on both roots" in str(excinfo.value)


def test_paired_roots_must_force_byte_identical_target_tokens() -> None:
    tokens = _valid_row()
    digest = sut.assert_paired_token_identity(
        tokens, list(tokens), declared_sha256=_sha256_json(tokens), label="probe"
    )
    assert digest == _sha256_json(tokens)

    with pytest.raises(sut.SecondaryCompatibilityContractError) as excinfo:
        sut.assert_paired_token_identity(
            tokens,
            [*tokens[:-1], BOX_START],
            declared_sha256=_sha256_json(tokens),
            label="probe",
        )
    assert "forced different target tokens" in str(excinfo.value)

    with pytest.raises(sut.SecondaryCompatibilityContractError) as excinfo:
        sut.assert_paired_token_identity(
            tokens, list(tokens), declared_sha256="c" * 64, label="probe"
        )
    assert "not the sealed" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 4. Paired-root resolution against the sealed registries
# ---------------------------------------------------------------------------


def test_primary_variants_append_to_the_same_boundary_they_score_against(
    plan: primary.SealedPlan,
) -> None:
    for variant, offset in (("p_plus_c_then_e", 0), ("p_plus_e_plus_c_then_f", 1)):
        request = _request_of(plan, image_id=RUNTIME_IMAGE_ID, variant=variant)
        registry = _registry_of(plan, str(request["gt_owner_id"]))
        roots = sut.resolve_paired_roots(plan, request, registry)
        assert roots.baseline_context_id == roots.modified_context_id
        assert roots.native_row_index == 3 + offset
        assert roots.successor_context_id.endswith(f"{4 + offset:03d}")
        assert roots.registry_source.startswith("cohort_registry")


def test_benign_substitution_pairs_against_the_unmodified_native_successor(
    plan: primary.SealedPlan,
) -> None:
    request = _request_of(
        plan,
        image_id=RUNTIME_IMAGE_ID,
        variant="benign_substitution_then_following_native_action",
    )
    registry = _registry_of(plan, str(request["gt_owner_id"]))
    roots = sut.resolve_paired_roots(plan, request, registry)
    # The clean twin replaces native row 0 at the due-boundary predecessor; the
    # baseline is the native successor context the plan seals.
    assert roots.modified_context_id.endswith("000")
    assert roots.baseline_context_id.endswith("001")
    assert roots.successor_context_id.endswith("002")
    assert roots.native_row_index == 1
    assert roots.registry_source == "control_registry.following_native_action"


def test_a_baseline_that_disagrees_with_the_boundary_convention_fails_closed(
    plan: primary.SealedPlan,
) -> None:
    request = _request_of(plan, image_id=RUNTIME_IMAGE_ID, variant="p_plus_c_then_e")
    registry = copy.deepcopy(_registry_of(plan, str(request["gt_owner_id"])))
    registry["e_row"]["pre_row_context_id"] = f"{RUNTIME_IMAGE_ID}:boundary-005"
    with pytest.raises(sut.SecondaryCompatibilityContractError) as excinfo:
        sut.resolve_paired_roots(plan, request, registry)
    assert "baseline context disagrees" in str(excinfo.value)


def test_a_target_that_is_not_the_literal_boundary_suffix_fails_closed(
    plan: primary.SealedPlan,
) -> None:
    request = copy.deepcopy(
        _request_of(plan, image_id=RUNTIME_IMAGE_ID, variant="p_plus_c_then_e")
    )
    registry = _registry_of(plan, str(request["gt_owner_id"]))
    tampered = _row_tokens([20001], [151700, 151800, 151900, 152000])
    request["scored_target"] = {
        **request["scored_target"],
        "token_ids": tampered,
        "token_ids_sha256": _sha256_json(tampered),
    }
    with pytest.raises(sut.SecondaryCompatibilityContractError) as excinfo:
        sut.resolve_paired_roots(plan, request, registry)
    assert "literal token suffix between sealed boundaries" in str(excinfo.value)


def test_a_materialized_f_request_without_a_sealed_f_row_fails_closed(
    plan: primary.SealedPlan,
) -> None:
    request = _request_of(plan, image_id=RUNTIME_IMAGE_ID, variant="p_plus_e_plus_c_then_f")
    registry = copy.deepcopy(_registry_of(plan, str(request["gt_owner_id"])))
    registry["f_row_present"] = False
    with pytest.raises(sut.SecondaryCompatibilityContractError) as excinfo:
        sut.resolve_paired_roots(plan, request, registry)
    assert "declares no F row" in str(excinfo.value)


def test_a_request_joined_to_the_wrong_owner_or_image_fails_closed(
    plan: primary.SealedPlan,
) -> None:
    request = _request_of(plan, image_id=RUNTIME_IMAGE_ID, variant="p_plus_c_then_e")
    other = _registry_of(plan, f"gt:{RUNTIME_IMAGE_ID}:tp")
    with pytest.raises(sut.SecondaryCompatibilityContractError) as excinfo:
        sut.resolve_paired_roots(plan, request, other)
    assert "wrong registry owner row" in str(excinfo.value)

    registry = copy.deepcopy(_registry_of(plan, str(request["gt_owner_id"])))
    registry["image_id"] = "img-01"
    with pytest.raises(sut.SecondaryCompatibilityContractError) as excinfo:
        sut.resolve_paired_roots(plan, request, registry)
    assert "is registered on image" in str(excinfo.value)


def test_an_appended_row_that_is_not_the_owners_sealed_clean_gt_row_fails_closed(
    plan: primary.SealedPlan,
) -> None:
    request = copy.deepcopy(
        _request_of(plan, image_id=RUNTIME_IMAGE_ID, variant="p_plus_c_then_e")
    )
    registry = _registry_of(plan, str(request["gt_owner_id"]))
    assert sut.assert_inserted_row_matches_registry(request, registry)
    request["prefix"] = {**request["prefix"], "appended_token_ids_sha256": "d" * 64}
    with pytest.raises(sut.SecondaryCompatibilityContractError) as excinfo:
        sut.assert_inserted_row_matches_registry(request, registry)
    assert "not the owner's sealed clean GT row" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 5. The sealed primary-branch gate
# ---------------------------------------------------------------------------


def _analysis_owner_row(index: int, *, branch: str = "displaced") -> dict[str, Any]:
    return {
        "schema_version": sut.ANALYSIS_OWNER_ROW_SCHEMA_VERSION,
        "unit_id": sut.UNIT_ID,
        "row_kind": "crossing_boundary_owner_row",
        "cohort": plan_builder.PRIMARY_COHORT,
        "gt_owner_id": f"gt:analysis:{index}",
        "image_id": f"img-{index % 12:02d}",
        "primary_branch": branch,
        "interpretable": True,
        "quarantined": False,
    }


def _write_primary_analysis(
    tmp_path: Path,
    *,
    plan_manifest_sha256: str = "plan-manifest-digest",
    owner_rows: list[dict[str, Any]] | None = None,
    receipt_overrides: dict[str, Any] | None = None,
    report_overrides: dict[str, Any] | None = None,
    merge_overrides: dict[str, Any] | None = None,
    extra_file: str | None = None,
    drop_file: str | None = None,
    reseal: bool = True,
) -> Path:
    """A complete sealed primary analysis plus the merged run it points at."""

    merged_dir = tmp_path / "primary-merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    merge_receipt: dict[str, Any] = {
        "schema_version": "sorted_crossing_boundary_owner_release_merge_receipt.v1",
        "unit_id": sut.UNIT_ID,
        "plan": {"manifest_content_sha256": plan_manifest_sha256},
        "policy": {"secondary_compatibility_merged": False, "merged_readout_tier": "primary"},
        "runtime_identity_sha256": "e" * 64,
        "scorer_source_sha256": "f" * 64,
        "merger_source_sha256": "0" * 64,
    }
    merge_receipt.update(merge_overrides or {})
    merge_receipt["receipt_content_sha256"] = _sha256_json(merge_receipt)
    merge_bytes = (
        json.dumps(merge_receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode()
    (merged_dir / "merge-receipt.json").write_bytes(merge_bytes)

    rows = owner_rows if owner_rows is not None else [_analysis_owner_row(i) for i in range(26)]
    rows_bytes = b"".join(primary.canonical_json_bytes(row) + b"\n" for row in rows)
    report: dict[str, Any] = {
        "schema_version": sut.ANALYSIS_REPORT_SCHEMA_VERSION,
        "unit_id": sut.UNIT_ID,
        "decision": "route_one_mechanism_matched_successor",
        "merged_dir": str(merged_dir),
        "merge_receipt_content_sha256": merge_receipt["receipt_content_sha256"],
        "runtime_identity_sha256": merge_receipt["runtime_identity_sha256"],
        "routing": {"routing_denominator_count": 24},
        "interpretability_gate": {"passed": True},
        "primary_cohort": {"denominator": 26},
    }
    report.update(report_overrides or {})
    report_bytes = (
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode()
    report_md_bytes = b"# report\n"

    analysis_dir = tmp_path / "primary-analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / sut.ANALYSIS_OWNER_ROWS_NAME).write_bytes(rows_bytes)
    (analysis_dir / sut.ANALYSIS_REPORT_JSON_NAME).write_bytes(report_bytes)
    (analysis_dir / sut.ANALYSIS_REPORT_MD_NAME).write_bytes(report_md_bytes)

    receipt: dict[str, Any] = {
        "schema_version": sut.ANALYSIS_RECEIPT_SCHEMA_VERSION,
        "unit_id": sut.UNIT_ID,
        # The gate now requires the analysis to have been produced by exactly the
        # analyzer this checkout would run, so the fixture seals that digest.
        "analyzer_source_sha256": sut.current_analyzer_source_sha256(),
        "scorer_source_sha256": merge_receipt["scorer_source_sha256"],
        "merger_source_sha256": merge_receipt["merger_source_sha256"],
        "merged_dir": str(merged_dir),
        "merge_receipt_content_sha256": merge_receipt["receipt_content_sha256"],
        "runtime_identity_sha256": merge_receipt["runtime_identity_sha256"],
        "input_file_sha256": {
            "merge-receipt.json": hashlib.sha256(merge_bytes).hexdigest(),
        },
        "primary_denominator": 26,
        "routing_denominator_count": report["routing"]["routing_denominator_count"],
        "interpretability_gate_passed": report["interpretability_gate"]["passed"],
        "decision": report["decision"],
        "policy": {"secondary_compatibility_read": False},
        "artifact_determinism": "no_wall_clock_or_host_specific_field_is_sealed",
        "output_file_digests": {
            sut.ANALYSIS_OWNER_ROWS_NAME: {
                "path": sut.ANALYSIS_OWNER_ROWS_NAME,
                "byte_size": len(rows_bytes),
                "row_count": len(rows),
                "sha256": hashlib.sha256(rows_bytes).hexdigest(),
            },
            sut.ANALYSIS_REPORT_JSON_NAME: {
                "path": sut.ANALYSIS_REPORT_JSON_NAME,
                "byte_size": len(report_bytes),
                "sha256": hashlib.sha256(report_bytes).hexdigest(),
            },
            sut.ANALYSIS_REPORT_MD_NAME: {
                "path": sut.ANALYSIS_REPORT_MD_NAME,
                "byte_size": len(report_md_bytes),
                "sha256": hashlib.sha256(report_md_bytes).hexdigest(),
            },
        },
    }
    receipt.update(receipt_overrides or {})
    if reseal:
        receipt["receipt_content_sha256"] = _sha256_json(receipt)
    else:
        receipt.setdefault("receipt_content_sha256", "2" * 64)
    (analysis_dir / sut.ANALYSIS_RECEIPT_NAME).write_bytes(
        (json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
    )
    if extra_file is not None:
        (analysis_dir / extra_file).write_text("stray\n")
    if drop_file is not None:
        (analysis_dir / drop_file).unlink()
    return analysis_dir


def test_only_the_live_analyzer_revision_is_admitted() -> None:
    # One pinned revision, taken from the analyzer's own constants; no dual
    # schema tuple and no lineage caveat survive.
    assert sut.ANALYSIS_RECEIPT_SCHEMA_VERSION == analyzer.RECEIPT_SCHEMA_VERSION
    assert sut.ANALYSIS_REPORT_SCHEMA_VERSION == analyzer.REPORT_SCHEMA_VERSION
    assert sut.ANALYSIS_OWNER_ROW_SCHEMA_VERSION == analyzer.OWNER_ROW_SCHEMA_VERSION
    assert sut.ANALYSIS_REPORT_SCHEMA_VERSION.endswith(".v2")
    assert sut.ANALYSIS_OWNER_ROW_SCHEMA_VERSION.endswith(".v2")
    assert sut.ANALYSIS_REPORT_SCHEMA_VERSION != RETIRED_REPORT_SCHEMA_V1
    assert sut.ANALYSIS_OWNER_ROW_SCHEMA_VERSION != RETIRED_OWNER_ROW_SCHEMA_V1
    for retired in ("ANALYSIS_ADMITTED_REPORT_SCHEMAS", "ANALYZER_SOURCE_LINEAGE_CAVEAT"):
        assert not hasattr(sut, retired), f"{retired} must not survive the tightening"
    assert sut.current_analyzer_source_sha256() == hashlib.sha256(
        Path(analyzer.__file__).resolve().read_bytes()
    ).hexdigest()


def test_a_complete_sealed_analysis_closes_the_branch_gate(
    tmp_path: Path, plan: primary.SealedPlan
) -> None:
    analysis_dir = _write_primary_analysis(tmp_path)
    analysis = sut.load_sealed_primary_analysis(analysis_dir, plan=plan)
    assert analysis.branch_row_count == 26
    assert analysis.branch_counts == {"displaced": 26}
    assert analysis.decision == "route_one_mechanism_matched_successor"
    assert analysis.binding["secondary_fields_present"] is False
    assert analysis.binding["branch_labels_used_to_select_requests"] is False
    assert analysis.binding["plan_manifest_content_sha256"] == "plan-manifest-digest"
    assert analysis.binding["analyzer_source_sha256"] == sut.current_analyzer_source_sha256()
    assert analysis.binding_sha256 == _sha256_json(analysis.binding)


def test_a_v1_report_schema_is_rejected(tmp_path: Path, plan: primary.SealedPlan) -> None:
    analysis_dir = _write_primary_analysis(
        tmp_path, report_overrides={"schema_version": RETIRED_REPORT_SCHEMA_V1}
    )
    with pytest.raises(sut.SecondaryCompatibilityContractError) as excinfo:
        sut.load_sealed_primary_analysis(analysis_dir, plan=plan)
    message = str(excinfo.value)
    assert "primary analysis report schema" in message
    assert RETIRED_REPORT_SCHEMA_V1 in message
    assert sut.ANALYSIS_REPORT_SCHEMA_VERSION in message


def test_v1_owner_rows_are_rejected(tmp_path: Path, plan: primary.SealedPlan) -> None:
    rows = [_analysis_owner_row(index) for index in range(26)]
    rows[11]["schema_version"] = RETIRED_OWNER_ROW_SCHEMA_V1
    analysis_dir = _write_primary_analysis(tmp_path, owner_rows=rows)
    with pytest.raises(sut.SecondaryCompatibilityContractError) as excinfo:
        sut.load_sealed_primary_analysis(analysis_dir, plan=plan)
    message = str(excinfo.value)
    assert "sealed analysis owner row schema" in message
    assert RETIRED_OWNER_ROW_SCHEMA_V1 in message
    assert sut.ANALYSIS_OWNER_ROW_SCHEMA_VERSION in message


def test_a_retired_receipt_schema_is_rejected(
    tmp_path: Path, plan: primary.SealedPlan
) -> None:
    analysis_dir = _write_primary_analysis(
        tmp_path,
        receipt_overrides={
            "schema_version": "sorted-crossing-boundary-owner-release-receipt.v0"
        },
    )
    with pytest.raises(sut.SecondaryCompatibilityContractError) as excinfo:
        sut.load_sealed_primary_analysis(analysis_dir, plan=plan)
    assert "primary analysis receipt schema" in str(excinfo.value)


def test_an_analysis_from_another_analyzer_revision_is_rejected(
    tmp_path: Path, plan: primary.SealedPlan
) -> None:
    # Structurally complete and self-sealed, but emitted by a different analyzer
    # source: a branch registry from another revision is a different branch
    # contract and never gates this pass.
    stale = "a1" * 32
    assert stale != sut.current_analyzer_source_sha256()
    analysis_dir = _write_primary_analysis(
        tmp_path, receipt_overrides={"analyzer_source_sha256": stale}
    )
    with pytest.raises(sut.SecondaryCompatibilityContractError) as excinfo:
        sut.load_sealed_primary_analysis(analysis_dir, plan=plan)
    message = str(excinfo.value)
    assert stale in message
    assert sut.current_analyzer_source_sha256() in message
    assert "not the one the current analyzer emits" in message


@pytest.mark.parametrize("name", sut.ANALYSIS_REQUIRED_FILES)
def test_an_incomplete_analysis_directory_fails_closed(
    tmp_path: Path, plan: primary.SealedPlan, name: str
) -> None:
    analysis_dir = _write_primary_analysis(tmp_path, drop_file=name)
    with pytest.raises(sut.SecondaryCompatibilityContractError) as excinfo:
        sut.load_sealed_primary_analysis(analysis_dir, plan=plan)
    assert "incomplete" in str(excinfo.value)
    assert name in str(excinfo.value)


def test_an_unknown_artifact_in_the_analysis_directory_fails_closed(
    tmp_path: Path, plan: primary.SealedPlan
) -> None:
    analysis_dir = _write_primary_analysis(tmp_path, extra_file="secondary-rows.jsonl")
    with pytest.raises(sut.SecondaryCompatibilityContractError) as excinfo:
        sut.load_sealed_primary_analysis(analysis_dir, plan=plan)
    assert "unknown artifact" in str(excinfo.value)


def test_an_edited_analysis_receipt_that_did_not_reseal_fails_closed(
    tmp_path: Path, plan: primary.SealedPlan
) -> None:
    analysis_dir = _write_primary_analysis(tmp_path, reseal=False)
    with pytest.raises(sut.SecondaryCompatibilityContractError) as excinfo:
        sut.load_sealed_primary_analysis(analysis_dir, plan=plan)
    assert "does not self-seal" in str(excinfo.value)


def test_a_tampered_analysis_output_digest_fails_closed(
    tmp_path: Path, plan: primary.SealedPlan
) -> None:
    analysis_dir = _write_primary_analysis(tmp_path)
    (analysis_dir / sut.ANALYSIS_REPORT_MD_NAME).write_text("# tampered\n")
    with pytest.raises(sut.SecondaryCompatibilityContractError) as excinfo:
        sut.load_sealed_primary_analysis(analysis_dir, plan=plan)
    assert "hashes to" in str(excinfo.value)


def test_fewer_than_26_sealed_branch_rows_fails_closed(
    tmp_path: Path, plan: primary.SealedPlan
) -> None:
    analysis_dir = _write_primary_analysis(
        tmp_path, owner_rows=[_analysis_owner_row(i) for i in range(25)]
    )
    with pytest.raises(sut.SecondaryCompatibilityContractError) as excinfo:
        sut.load_sealed_primary_analysis(analysis_dir, plan=plan)
    assert "25 primary branch rows, not the frozen 26" in str(excinfo.value)


def test_an_unknown_primary_branch_label_fails_closed(
    tmp_path: Path, plan: primary.SealedPlan
) -> None:
    rows = [_analysis_owner_row(i) for i in range(26)]
    rows[7]["primary_branch"] = "owner_commit_token"
    analysis_dir = _write_primary_analysis(tmp_path, owner_rows=rows)
    with pytest.raises(sut.SecondaryCompatibilityContractError) as excinfo:
        sut.load_sealed_primary_analysis(analysis_dir, plan=plan)
    assert "unknown primary branch" in str(excinfo.value)


def test_a_sealed_analysis_carrying_a_secondary_field_fails_closed(
    tmp_path: Path, plan: primary.SealedPlan
) -> None:
    rows = [_analysis_owner_row(i) for i in range(26)]
    rows[3]["secondary_compatibility"] = {"complete_row_delta": -1.0}
    analysis_dir = _write_primary_analysis(tmp_path, owner_rows=rows)
    with pytest.raises(sut.SecondaryCompatibilityContractError) as excinfo:
        sut.load_sealed_primary_analysis(analysis_dir, plan=plan)
    assert "secondary readout key" in str(excinfo.value)


def test_an_analysis_that_declares_it_read_secondary_fields_fails_closed(
    tmp_path: Path, plan: primary.SealedPlan
) -> None:
    analysis_dir = _write_primary_analysis(
        tmp_path, receipt_overrides={"policy": {"secondary_compatibility_read": True}}
    )
    with pytest.raises(sut.SecondaryCompatibilityContractError) as excinfo:
        sut.load_sealed_primary_analysis(analysis_dir, plan=plan)
    assert "secondary_compatibility_read=false" in str(excinfo.value)


def test_an_unknown_routing_decision_fails_closed(
    tmp_path: Path, plan: primary.SealedPlan
) -> None:
    analysis_dir = _write_primary_analysis(
        tmp_path,
        receipt_overrides={"decision": "promote_owner_commit_token"},
        report_overrides={"decision": "promote_owner_commit_token"},
    )
    with pytest.raises(sut.SecondaryCompatibilityContractError) as excinfo:
        sut.load_sealed_primary_analysis(analysis_dir, plan=plan)
    assert "unknown routing decision" in str(excinfo.value)


def test_an_analysis_sealed_over_another_plan_fails_closed(
    tmp_path: Path, plan: primary.SealedPlan
) -> None:
    analysis_dir = _write_primary_analysis(
        tmp_path, plan_manifest_sha256="another-plan-manifest"
    )
    with pytest.raises(sut.SecondaryCompatibilityContractError) as excinfo:
        sut.load_sealed_primary_analysis(analysis_dir, plan=plan)
    assert "decided over a different CPU plan manifest" in str(excinfo.value)


def test_a_drifted_merge_receipt_fails_closed(
    tmp_path: Path, plan: primary.SealedPlan
) -> None:
    analysis_dir = _write_primary_analysis(tmp_path)
    (tmp_path / "primary-merged" / "merge-receipt.json").write_text("{}\n")
    with pytest.raises(sut.SecondaryCompatibilityContractError) as excinfo:
        sut.load_sealed_primary_analysis(analysis_dir, plan=plan)
    assert "the merged evidence drifted after the branch pass" in str(excinfo.value)


def test_a_merge_receipt_declaring_merged_secondary_rows_fails_closed(
    tmp_path: Path, plan: primary.SealedPlan
) -> None:
    analysis_dir = _write_primary_analysis(
        tmp_path,
        merge_overrides={
            "policy": {"secondary_compatibility_merged": True, "merged_readout_tier": "primary"}
        },
    )
    with pytest.raises(sut.SecondaryCompatibilityContractError) as excinfo:
        sut.load_sealed_primary_analysis(analysis_dir, plan=plan)
    assert "must be free of them" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 6. Cache parity and quarantine
# ---------------------------------------------------------------------------


def _parity_row(
    request_id: str,
    *,
    baseline: list[float],
    modified: list[float],
    baseline_argmax: list[int] | None = None,
    modified_argmax: list[int] | None = None,
    signs: dict[str, int] | None = None,
) -> dict[str, Any]:
    tokens = list(range(len(baseline)))
    return {
        "request_id": request_id,
        "scored_token_ids": tokens,
        "roots": {
            sut.ROOT_BASELINE: {
                "selected_logprobs": baseline,
                "argmax_token_ids": baseline_argmax or tokens,
            },
            sut.ROOT_MODIFIED: {
                "selected_logprobs": modified,
                "argmax_token_ids": modified_argmax or tokens,
            },
        },
        "deltas": {
            segment: {"sign": (signs or {}).get(segment, 1)} for segment in sut.SEGMENTS
        },
    }


def test_parity_admits_cache_within_tolerance_and_full_discrete_agreement() -> None:
    cached = [_parity_row("req:a", baseline=[-1.0, -2.0], modified=[-3.0, -4.0])]
    uncached = [
        _parity_row("req:a", baseline=[-1.0005, -2.0], modified=[-3.0, -4.0002])
    ]
    result = sut.evaluate_secondary_parity(cached_rows=cached, uncached_rows=uncached)
    assert result.status == sut.CACHE_ADMITTED
    assert result.mismatched_fields == ()
    assert result.max_selected_logit_abs_diff <= 1e-3
    assert {entry.root for entry in result.per_root} == set(sut.PAIRED_ROOTS)
    assert all(entry.compared_token_count == 2 for entry in result.per_root)


def test_parity_falls_back_when_any_selected_logit_exceeds_1e_minus_3() -> None:
    cached = [_parity_row("req:a", baseline=[-1.0, -2.0], modified=[-3.0, -4.0])]
    uncached = [_parity_row("req:a", baseline=[-1.0, -2.0], modified=[-3.0, -4.01])]
    result = sut.evaluate_secondary_parity(cached_rows=cached, uncached_rows=uncached)
    assert result.status == sut.UNCACHED_FALLBACK
    assert f"selected_logit:{sut.ROOT_MODIFIED}" in result.mismatched_fields


def test_parity_falls_back_on_an_argmax_or_rank_flip_within_tolerance() -> None:
    cached = [_parity_row("req:a", baseline=[-1.0, -2.0], modified=[-3.0, -4.0])]
    uncached = [
        _parity_row(
            "req:a",
            baseline=[-1.0, -2.0],
            modified=[-3.0, -4.0],
            baseline_argmax=[0, 99],
        )
    ]
    result = sut.evaluate_secondary_parity(cached_rows=cached, uncached_rows=uncached)
    assert result.status == sut.UNCACHED_FALLBACK
    assert f"argmax:{sut.ROOT_BASELINE}" in result.mismatched_fields
    assert f"selected_is_argmax:{sut.ROOT_BASELINE}" in result.mismatched_fields


def test_parity_falls_back_on_a_segment_delta_sign_flip_alone() -> None:
    cached = [_parity_row("req:a", baseline=[-1.0, -2.0], modified=[-3.0, -4.0])]
    uncached = [
        _parity_row(
            "req:a",
            baseline=[-1.0, -2.0],
            modified=[-3.0, -4.0],
            signs={"coordinates": -1},
        )
    ]
    result = sut.evaluate_secondary_parity(cached_rows=cached, uncached_rows=uncached)
    assert result.status == sut.UNCACHED_FALLBACK
    assert "delta_sign" in result.mismatched_fields
    assert result.delta_sign_parity is False


def test_parity_refuses_misaligned_or_empty_streams() -> None:
    cached = [_parity_row("req:a", baseline=[-1.0], modified=[-2.0])]
    uncached = [_parity_row("req:b", baseline=[-1.0], modified=[-2.0])]
    result = sut.evaluate_secondary_parity(cached_rows=cached, uncached_rows=uncached)
    assert result.status == sut.UNCACHED_FALLBACK
    assert any(name.startswith("stream_alignment") for name in result.mismatched_fields)

    empty = sut.evaluate_secondary_parity(cached_rows=[], uncached_rows=[])
    assert empty.status == sut.UNCACHED_FALLBACK
    assert any(name.startswith("stream_empty") for name in empty.mismatched_fields)


def test_replay_enforcement_follows_the_backends_own_evidence_declaration() -> None:
    assert sut.replay_enforcement_for(REAL_BACKEND_IDENTITY) is True
    assert sut.replay_enforcement_for(FAKE_BACKEND_IDENTITY) is False
    assert sut.replay_enforcement_for(census_shard.FakeCensusBackend().identity) is False


def test_quarantine_ledger_is_immutable_and_append_only() -> None:
    ledger = sut.SecondaryQuarantineLedger()
    assert ledger.count == 0
    unchanged = sut.apply_secondary_quarantine(
        admitted=True,
        request_id="req:a",
        gt_owner_id="gt:x",
        reason="r",
        detail="d",
        ledger=ledger,
    )
    assert unchanged is ledger
    appended = sut.apply_secondary_quarantine(
        admitted=False,
        request_id="req:a",
        gt_owner_id="gt:x",
        reason="baseline_native_argmax_replay_mismatch",
        detail="d",
        ledger=ledger,
    )
    assert ledger.count == 0
    assert appended.count == 1
    assert appended.entries[0].request_id == "req:a"
    with pytest.raises(FrozenInstanceError):
        appended.entries[0].reason = "other"  # type: ignore[misc]


def test_a_quarantined_shard_publishes_no_partial_evidence() -> None:
    result = sut.SecondaryShardResult(
        receipt={"schema_version": sut.RECEIPT_SCHEMA_VERSION},
        rows=[{"request_id": "req:a"}],
        parity={"schema_version": sut.PARITY_SCHEMA_VERSION},
        quarantine={"schema_version": sut.QUARANTINE_SCHEMA_VERSION},
    )
    files = sut.shard_output_files(result)
    assert sorted(files) == [sut.QUARANTINE_NAME, sut.RECEIPT_NAME]
    for name in sut.EVIDENCE_OUTPUT_NAMES:
        assert name not in files


# ---------------------------------------------------------------------------
# 7. End-to-end capture on the deterministic fake backend
# ---------------------------------------------------------------------------


def _runtime_identity() -> dict[str, Any]:
    return {
        "backend": "fake",
        "model_identity": {"checkpoint": "fake"},
        "tokenizer_identity": {"tokenizer": "fake"},
        "adapter_identity": None,
        "numerics": {"explicit_position_ids": True},
        "source_identity": {"scripts.research.fake": "9" * 64},
    }


@pytest.fixture()
def sealed_analysis(tmp_path: Path, plan: primary.SealedPlan) -> sut.SealedPrimaryAnalysis:
    return sut.load_sealed_primary_analysis(_write_primary_analysis(tmp_path), plan=plan)


def _fake_backend() -> Any:
    return census_shard.FakeCensusBackend(seed="secondary-test")


def _smoke(plan: primary.SealedPlan, analysis: sut.SealedPrimaryAnalysis) -> Any:
    return sut.run_smoke_shard(
        plan=plan,
        analysis=analysis,
        backend=_fake_backend(),
        shard_id="smoke",
        image_id=RUNTIME_IMAGE_ID,
        runtime_identity=_runtime_identity(),
    )


def test_smoke_admits_the_cache_and_seals_an_admission_over_one_image(
    plan: primary.SealedPlan, sealed_analysis: sut.SealedPrimaryAnalysis
) -> None:
    result = _smoke(plan, sealed_analysis)
    admission = result.admission
    assert admission is not None
    assert admission["cache_admitted"] is True
    assert admission["smoke_variants"] == {
        "p_plus_c_then_e": 1,
        "p_plus_e_plus_c_then_f": 1,
        "benign_substitution_then_following_native_action": 1,
    }
    assert admission["primary_analysis_binding_sha256"] == sealed_analysis.binding_sha256
    assert admission["plan_manifest_content_sha256"] == "plan-manifest-digest"
    assert set(admission["root_backend"]) == set(sut.PAIRED_ROOTS)
    # A smoke publishes parity and its admission, never evidence rows.
    assert sorted(sut.shard_output_files(result)) == [
        sut.ADMISSION_NAME,
        sut.PARITY_NAME,
        sut.RECEIPT_NAME,
    ]
    assert result.receipt["executed"]["row_count"] == 0
    assert result.receipt["mode"] == "smoke"


def test_capture_scores_every_sealed_secondary_request_of_the_image(
    plan: primary.SealedPlan, sealed_analysis: sut.SealedPrimaryAnalysis
) -> None:
    admission = _smoke(plan, sealed_analysis).admission
    result = sut.run_capture_shard(
        plan=plan,
        analysis=sealed_analysis,
        backend=_fake_backend(),
        shard_id="cap",
        session_image_id=RUNTIME_IMAGE_ID,
        admission=admission,
        runtime_identity=_runtime_identity(),
    )
    assert result.quarantine is None
    assert len(result.rows) == 3
    assert sut.counts_by_variant(result.rows) == {
        "p_plus_c_then_e": 1,
        "p_plus_e_plus_c_then_f": 1,
        "benign_substitution_then_following_native_action": 1,
    }
    counts = result.receipt["secondary_request_counts"]
    assert counts["plan"]["observed_total"] == 64
    assert counts["image"]["observed_by_variant"] == counts["image"]["expected_by_variant"]
    assert result.receipt["policy"]["branch_labels_used_to_select_requests"] is False
    assert result.receipt["policy"]["primary_branches_sealed_before_this_pass"] is True
    assert result.receipt["policy"]["claim_boundary"] == sut.CLAIM_BOUNDARY
    assert (
        result.receipt["primary_analysis_binding_sha256"] == sealed_analysis.binding_sha256
    )


def test_captured_rows_pair_identical_tokens_and_report_exact_segment_math(
    plan: primary.SealedPlan, sealed_analysis: sut.SealedPrimaryAnalysis
) -> None:
    admission = _smoke(plan, sealed_analysis).admission
    result = sut.run_capture_shard(
        plan=plan,
        analysis=sealed_analysis,
        backend=_fake_backend(),
        shard_id="cap",
        session_image_id=RUNTIME_IMAGE_ID,
        admission=admission,
        runtime_identity=_runtime_identity(),
    )
    for row in result.rows:
        tokens = row["scored_token_ids"]
        segments = sut.row_segments(tokens)
        assert row["scored_token_ids_sha256"] == _sha256_json(tokens)
        assert row["claim_boundary"] == sut.CLAIM_BOUNDARY
        assert row["delta_orientation"] == "modified_minus_baseline"
        assert row["retokenized"] is False
        assert row["uses_model_generate"] is False
        assert row["primary_analysis_binding_sha256"] == sealed_analysis.binding_sha256
        baseline = row["roots"][sut.ROOT_BASELINE]
        modified = row["roots"][sut.ROOT_MODIFIED]
        # Both roots forced the same literal row and are separate cache states.
        assert len(baseline["selected_logprobs"]) == len(tokens)
        assert len(modified["selected_logprobs"]) == len(tokens)
        assert baseline["context_group_id"] != modified["context_group_id"]
        assert baseline["appended_token_count"] == 0
        assert modified["appended_token_count"] == row["inserted_clean_row_c_token_count"]
        # Argmax ids travel as diagnostics beside the selected-token channel.
        assert len(baseline["argmax_token_ids"]) == len(tokens)
        assert baseline["selected_is_argmax"] == [
            int(a) == int(b) for a, b in zip(tokens, baseline["argmax_token_ids"])
        ]
        for segment in sut.SEGMENTS:
            start, stop = segments.span(segment)
            expected_baseline = math.fsum(baseline["selected_logprobs"][start:stop])
            expected_modified = math.fsum(modified["selected_logprobs"][start:stop])
            delta = row["deltas"][segment]
            assert delta["baseline_sum"] == pytest.approx(expected_baseline)
            assert delta["modified_sum"] == pytest.approx(expected_modified)
            assert delta["delta"] == pytest.approx(expected_modified - expected_baseline)
            assert delta["sign"] == primary._margin_sign(delta["delta"])
            assert row["segments"][segment]["token_ids"] == tokens[start:stop]
        assert row["deltas"]["complete_row"]["token_count"] == len(tokens)


def test_capture_is_deterministic_and_publishes_create_or_identical(
    tmp_path: Path, plan: primary.SealedPlan, sealed_analysis: sut.SealedPrimaryAnalysis
) -> None:
    admission = _smoke(plan, sealed_analysis).admission

    def _capture() -> Any:
        return sut.run_capture_shard(
            plan=plan,
            analysis=sealed_analysis,
            backend=_fake_backend(),
            shard_id="cap",
            session_image_id=RUNTIME_IMAGE_ID,
            admission=admission,
            runtime_identity=_runtime_identity(),
        )

    first = sut.shard_output_files(_capture())
    second = sut.shard_output_files(_capture())
    assert first == second

    output_dir = tmp_path / "shard"
    published = primary._publish(output_dir, first)
    assert published["published"] is True
    again = primary._publish(output_dir, second)
    assert again["published"] is False
    assert again["publish_mode"] == "no_op_identical_rerun"

    mutated = dict(second)
    mutated[sut.ROWS_NAME] = second[sut.ROWS_NAME] + b"{}\n"
    with pytest.raises(primary.CrossingBoundaryContractError) as excinfo:
        primary._publish(output_dir, mutated)
    assert "not a byte-identical capture" in str(excinfo.value)


def test_a_context_of_another_image_is_refused_by_the_open_session(
    plan: primary.SealedPlan, sealed_analysis: sut.SealedPrimaryAnalysis
) -> None:
    admission = _smoke(plan, sealed_analysis).admission
    # Both images' root boundaries carry an empty self-prefix, so this request
    # still reconstructs every token digest; only the sealed image identity
    # separates them.  Forwarding it would pair a literal prefix with the wrong
    # picture and produce numbers that look ordinary while being meaningless.
    original = _request_of(
        plan,
        image_id=RUNTIME_IMAGE_ID,
        variant="benign_substitution_then_following_native_action",
    )
    relabelled = _make_request(
        cohort=plan_builder.TP_REPLAY_CONTROL_COHORT,
        gt_owner_id=str(original["gt_owner_id"]),
        image_id=RUNTIME_IMAGE_ID,
        context_id="img-01:boundary-000",
        context_role="root",
        boundary_index=0,
        appended_token_ids=list(original["prefix"]["appended_token_ids"]),
        base_prefix_token_ids=[],
        scored_target=dict(original["scored_target"]),
        variant="benign_substitution_then_following_native_action",
        optional=False,
    )
    plan.request_rows[plan.request_rows.index(original)] = relabelled
    with pytest.raises(primary.CrossingBoundaryContractError) as excinfo:
        sut.run_capture_shard(
            plan=plan,
            analysis=sealed_analysis,
            backend=_fake_backend(),
            shard_id="cap",
            session_image_id=RUNTIME_IMAGE_ID,
            admission=admission,
            runtime_identity=_runtime_identity(),
        )
    assert "another image's visual state" in str(excinfo.value)


def test_an_admission_sealed_against_another_branch_gate_is_refused(
    tmp_path: Path, plan: primary.SealedPlan, sealed_analysis: sut.SealedPrimaryAnalysis
) -> None:
    admission = _smoke(plan, sealed_analysis).admission
    other_dir = _write_primary_analysis(
        tmp_path / "other",
        owner_rows=[_analysis_owner_row(i, branch="release_lost") for i in range(26)],
    )
    other = sut.load_sealed_primary_analysis(other_dir, plan=plan)
    assert other.binding_sha256 != sealed_analysis.binding_sha256
    with pytest.raises(sut.SecondaryCompatibilityContractError) as excinfo:
        sut.validate_secondary_admission_receipt(
            admission, plan=plan, analysis=other, runtime_identity=_runtime_identity()
        )
    assert "different primary analysis" in str(excinfo.value)


def test_an_edited_admission_receipt_is_refused(
    plan: primary.SealedPlan, sealed_analysis: sut.SealedPrimaryAnalysis
) -> None:
    admission = dict(_smoke(plan, sealed_analysis).admission)
    assert sut.validate_secondary_admission_receipt(
        admission, plan=plan, analysis=sealed_analysis, runtime_identity=_runtime_identity()
    )
    admission["cache_admitted"] = False
    with pytest.raises(sut.SecondaryCompatibilityContractError) as excinfo:
        sut.validate_secondary_admission_receipt(
            admission,
            plan=plan,
            analysis=sealed_analysis,
            runtime_identity=_runtime_identity(),
        )
    assert "does not self-seal" in str(excinfo.value)


def test_an_admission_sealed_under_another_runtime_is_refused(
    plan: primary.SealedPlan, sealed_analysis: sut.SealedPrimaryAnalysis
) -> None:
    admission = _smoke(plan, sealed_analysis).admission
    drifted = {**_runtime_identity(), "model_identity": {"checkpoint": "other"}}
    with pytest.raises(sut.SecondaryCompatibilityContractError) as excinfo:
        sut.validate_secondary_admission_receipt(
            admission, plan=plan, analysis=sealed_analysis, runtime_identity=drifted
        )
    assert "different runtime identity" in str(excinfo.value)


def test_a_replay_mismatch_on_an_evidence_bearing_backend_quarantines_the_shard(
    plan: primary.SealedPlan, sealed_analysis: sut.SealedPrimaryAnalysis
) -> None:
    """The fake backend's argmax is a hash, so an enforced replay must quarantine.

    ``replay_enforcement_for`` keys enforcement to the backend's own
    ``usable_as_evidence`` declaration; forcing it on here is what proves the
    enforced path withholds every evidence file rather than publishing a shard
    whose baseline root did not reproduce the native row.
    """

    admission = _smoke(plan, sealed_analysis).admission

    class _EvidenceBearingFake:
        """The same deterministic backend, declaring itself evidence-bearing."""

        def __init__(self, inner: Any) -> None:
            self._inner = inner

        def __getattr__(self, name: str) -> Any:
            return getattr(self._inner, name)

        @property
        def identity(self) -> dict[str, Any]:
            return {**dict(self._inner.identity), "usable_as_evidence": True}

    result = sut.run_capture_shard(
        plan=plan,
        analysis=sealed_analysis,
        backend=_EvidenceBearingFake(_fake_backend()),
        shard_id="cap",
        session_image_id=RUNTIME_IMAGE_ID,
        admission=admission,
        runtime_identity=_runtime_identity(),
    )
    assert result.quarantine is not None
    assert result.quarantine["entries"]
    assert all(
        entry["reason"] == "baseline_native_argmax_replay_mismatch"
        for entry in result.quarantine["entries"]
    )
    files = sut.shard_output_files(result)
    assert sorted(files) == [sut.QUARANTINE_NAME, sut.RECEIPT_NAME]
    assert result.receipt["policy"]["replay_admission_enforced"] is True


def test_a_reused_root_cache_state_fails_closed(
    plan: primary.SealedPlan, sealed_analysis: sut.SealedPrimaryAnalysis
) -> None:
    # Two secondary requests that shared one logical root would mean the second
    # read a cache the first had already advanced.
    ids = ["group:a", "group:a"]
    with pytest.raises(primary.CrossingBoundaryContractError) as excinfo:
        primary.assert_fresh_context_per_owner(ids)
    assert "reused" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 8. CLI surface
# ---------------------------------------------------------------------------


def test_cli_parallels_the_primary_scorer() -> None:
    parser = sut.build_parser()
    options = {
        action.option_strings[0] for action in parser._actions if action.option_strings
    }
    for name in (
        "--plan-dir",
        "--primary-analysis-dir",
        "--prevalence-run-root",
        "--census-run-root",
        "--infer-config",
        "--runtime-identity",
        "--output-dir",
        "--shard-id",
        "--mode",
        "--image-id",
        "--admission-receipt",
        "--backend",
        "--batch-size",
    ):
        assert name in options
    args = parser.parse_args(
        [
            "--plan-dir", "/plan",
            "--primary-analysis-dir", "/analysis",
            "--output-dir", "/out",
            "--shard-id", "s",
            "--image-id", "img-00",
        ]
    )
    assert args.mode == sut.MODE_CAPTURE
    assert args.backend == "hf"
    assert args.batch_size == sut.DEFAULT_BATCH_SIZE


def test_no_public_helper_takes_a_model_tokenizer_or_device(
    plan: primary.SealedPlan,
) -> None:
    import inspect

    forbidden = {"model", "tokenizer", "device", "processor", "cuda"}
    for name, value in vars(sut).items():
        if name.startswith("_") or not inspect.isfunction(value):
            continue
        if value.__module__ != sut.__name__:
            continue
        parameters = set(inspect.signature(value).parameters)
        assert not (parameters & forbidden), f"{name} accepts {parameters & forbidden!r}"
