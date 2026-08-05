"""Contract tests for the neutral-row control paired capture.

The fixture is a bounded synthetic sealed plan carrying the whole frozen
21/21/12 census, plus a stub census context surface, so the request contract,
the paired-root resolution, the scored-token identity gate and one real
end-to-end shard are exercised without a model, a GPU or a production artifact.
The deterministic ``FakeCensusBackend`` runs the identical teacher-forcing code
path the production backend runs.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from scripts.research import prepare_sorted_crossing_neutral_row_control as plan_builder
from scripts.research import score_sorted_crossing_boundary_owner_release as crossing_scorer
from scripts.research import (
    score_sorted_crossing_boundary_owner_release_secondary as paired,
)
from scripts.research import score_sorted_crossing_neutral_row_control as sut
from scripts.research import score_sorted_owner_accessibility_census_shard as shard

OBJ_START = crossing_scorer.OBJECT_REF_START
OBJ_END = crossing_scorer.OBJECT_REF_END
BOX_START = crossing_scorer.BOX_START
BOX_END = crossing_scorer.BOX_END
COORD_START = crossing_scorer.COORDINATE_TOKEN_ID_START

FAKE_BACKEND_IDENTITY = {"backend": "fake", "usable_as_evidence": False}
REAL_BACKEND_IDENTITY = {"backend": "hf", "usable_as_evidence": True}

#: Executed crossing owners per image; 21 in total, with image ``7511`` carrying
#: only its benign reference exactly as the real cohort does.
EXECUTED_PER_IMAGE: dict[str, int] = {
    "10707": 1,
    "13348": 1,
    "13923": 2,
    "14038": 3,
    "14439": 1,
    "1584": 1,
    "16228": 4,
    "2685": 1,
    "4134": 5,
    "5001": 1,
    "6040": 1,
    "7511": 0,
}
IMAGE_IDS: tuple[str, ...] = tuple(EXECUTED_PER_IMAGE)
#: The image a capture shard opens a session for: one owner (two arms) plus its
#: benign pair, so a whole shard is three requests instead of a 21-owner sweep.
#: Exactly like its real counterpart it carries one unmatched-``E``,
#: different-description owner, so it can never carry the smoke.
RUNTIME_IMAGE_ID = "10707"
#: The one image the real sealed plan proves carries all four strata across its
#: executed owners, and therefore the only image the smoke may run on.
SMOKE_IMAGE_ID = "4134"


def _sha256_json(value: Any) -> str:
    return crossing_scorer.sha256_json(value)


def _strata(image_id: str, slot: int) -> tuple[bool, bool]:
    """``(E strict-matched, E realized C's description)`` for one executed owner.

    Only the smoke image carries all four strata; every other image stays
    unmatched-``E``/different-description, exactly as the single-owner images of
    the real sealed plan do.
    """

    if image_id != SMOKE_IMAGE_ID:
        return False, False
    return slot % 2 == 0, slot in (1, 2)


def _coords(index: int) -> list[int]:
    base = (index * 7) % 900
    return [COORD_START + base + offset for offset in (0, 1, 4, 5)]


def _description(seed: int) -> list[int]:
    return [20000 + (seed * 13) % 500] * (1 + seed % 2)


def _row(seed: int) -> list[int]:
    return [OBJ_START, *_description(seed), OBJ_END, BOX_START, *_coords(seed), BOX_END]


def _make_request(
    *,
    arm: str,
    cohort_role: str,
    gt_owner_id: str,
    image_id: str,
    context_id: str,
    context_role: str,
    boundary_index: int,
    base_prefix_token_ids: list[int],
    appended_token_ids: list[int],
    scored_target: dict[str, Any],
    baseline_context_id: str,
    sealed_reference: dict[str, Any],
) -> dict[str, Any]:
    """Reproduce the sealed request identity the plan builder publishes."""

    base_digest = _sha256_json(base_prefix_token_ids)
    family = plan_builder.REQUEST_FAMILY_BY_ARM[arm]
    identity = {
        "unit_id": sut.UNIT_ID,
        "request_family": family,
        "arm": arm,
        "cohort_role": cohort_role,
        "gt_owner_id": gt_owner_id,
        "context_id": context_id,
        "baseline_context_id": baseline_context_id,
        "appended_token_ids": list(appended_token_ids),
        "base_prefix_token_ids_sha256": base_digest,
        "scored_target": dict(scored_target),
    }
    digest = _sha256_json(identity)
    return {
        "schema_version": plan_builder.REQUEST_SCHEMA_VERSION,
        "row_kind": "neutral_row_control_request",
        "unit_id": sut.UNIT_ID,
        "request_id": f"req:{digest[:32]}",
        "request_key": f"{family}|{cohort_role}|{gt_owner_id}|{context_id}|{arm}",
        "request_family": family,
        "arm": arm,
        "cohort_role": cohort_role,
        "gt_owner_id": gt_owner_id,
        "image_id": image_id,
        "context_id": context_id,
        "context_role": context_role,
        "boundary_index": boundary_index,
        "paired_roots": {
            "baseline_context_id": baseline_context_id,
            "modified_context_id": context_id,
            "orientation": "modified_minus_baseline",
        },
        "prefix": {
            "base_context_id": context_id,
            "base_prefix_token_count": len(base_prefix_token_ids),
            "base_prefix_token_ids_sha256": base_digest,
            "appended_token_ids": list(appended_token_ids),
            "appended_token_ids_sha256": _sha256_json(list(appended_token_ids)),
            "appended_token_count": len(appended_token_ids),
            "appended_role": plan_builder.APPENDED_ROLE_BY_ARM[arm],
            "retokenized": False,
        },
        "scored_target": dict(scored_target),
        "sealed_reference": dict(sealed_reference),
        "score_blind_plan": True,
        "inspects_new_model_logits": False,
        "identity_digest": digest,
    }


class _StubCensus:
    def __init__(self, contexts: dict[str, Any]) -> None:
        self.contexts_by_id = contexts


class _StubInputs:
    def __init__(self, contexts: dict[str, Any], images: dict[str, Any]) -> None:
        self.census = _StubCensus(contexts)
        self.images_by_id = images


def _sealed_reference(seed: int) -> dict[str, Any]:
    return {
        "coordinate_delta": -1.5 + 0.1 * seed,
        "coordinate_delta_sign": -1,
        "baseline_coordinate_sum": -10.0 - seed,
        "modified_coordinate_sum": -11.5 - seed,
        "baseline_argmax_token_ids_sha256": _sha256_json([seed]),
        "modified_argmax_token_ids_sha256": _sha256_json([seed]),
        "baseline_argmax_reproduces_description_path": True,
        "baseline_argmax_reproduces_complete_row": True,
        "scored_token_ids_sha256": _sha256_json([seed]),
        "scored_token_count": 9,
        "request_id": f"req:sealed:{seed}",
        "role": "gate_reference_only_never_the_estimand",
        "materiality_cutoff_nats": -1.0,
        "replay_max_coordinate_delta_abs_diff": 0.05,
        "replay_max_selected_logit_abs_diff": 0.001,
    }


def _synthetic_plan() -> sut.SealedNeutralPlan:
    """The whole sealed 21/21/12 census, as plain rows over a stub context surface."""

    contexts: dict[str, Any] = {}
    images: dict[str, Any] = {}
    selection_rows: list[dict[str, Any]] = []
    benign_rows: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    voting_budget = plan_builder.VOTING_OWNER_COUNT

    for image_index, image_id in enumerate(IMAGE_IDS):
        images[image_id] = {"prompt_token_ids": [900 + image_index, 901 + image_index]}
        owner_count = EXECUTED_PER_IMAGE[image_id]
        row_count = 4 + 2 * max(owner_count, 1)
        rows = [_row(image_index * 31 + row_index) for row_index in range(row_count)]
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

        def _context(index: int, _image_id: str = image_id) -> str:
            return f"{_image_id}:boundary-{index:03d}"

        for slot in range(owner_count):
            boundary = 3 + 2 * slot
            gt_owner_id = f"gt:{image_id}:{boundary}"
            votes = voting_budget > 0
            voting_budget -= 1 if votes else 0
            cohort_role = (
                plan_builder.COHORT_VOTING if votes else plan_builder.COHORT_SPECIFICITY
            )
            e_tokens = rows[boundary]
            clean = _row(5000 + image_index * 7 + slot)
            neutral = _row(9000 + image_index * 7 + slot)
            reference = _sealed_reference(image_index * 7 + slot)
            e_matched, e_same_description = _strata(image_id, slot)
            c_description = f"targ-{boundary}"
            selection_rows.append(
                {
                    "schema_version": plan_builder.SELECTION_SCHEMA_VERSION,
                    "row_kind": "neutral_row_selection",
                    "unit_id": sut.UNIT_ID,
                    "gt_owner_id": gt_owner_id,
                    "image_id": image_id,
                    "normalized_description": c_description,
                    "cohort_role": cohort_role,
                    "votes": votes,
                    "executed": True,
                    "material_negative": votes,
                    "clearly_separated": votes and slot == 0,
                    "sentinel_owner": False,
                    "posthoc_support_extent_uncertain": False,
                    "crossing": {
                        "boundary_index_b": boundary,
                        "p_context_id": _context(boundary),
                    },
                    "inserted_clean_row_c": {
                        "token_ids": clean,
                        "token_ids_sha256": _sha256_json(clean),
                        "token_count": len(clean),
                    },
                    "scored_e_row": {
                        "row_index": boundary,
                        "token_ids": e_tokens,
                        "token_ids_sha256": _sha256_json(e_tokens),
                        "strict_match_status": "matched" if e_matched else "unmatched",
                        "stratum": "matched_e" if e_matched else "unmatched_e",
                        "normalized_description": (
                            c_description if e_same_description else "erow"
                        ),
                        "pre_row_context_id": _context(boundary),
                        "post_row_context_id": _context(boundary + 1),
                    },
                    "neutral_row_n": {
                        "gt_owner_id": f"gt:{image_id}:n{slot}",
                        "token_ids": neutral,
                        "token_ids_sha256": _sha256_json(neutral),
                        "token_count": len(neutral),
                        "row_length_delta_tokens": 0,
                        "rows_back_distance": 2,
                        "min_center_distance_normalized": 0.2,
                    },
                    "sealed_clean_reference": reference,
                    "request_ids": [],
                }
            )
            scored_target = {
                "kind": "exact_native_row",
                "token_ids": list(e_tokens),
                "token_ids_sha256": _sha256_json(e_tokens),
                "native_row_index": boundary,
                "baseline_context_id": _context(boundary),
                "successor_context_id": _context(boundary + 1),
                "compare_against": "the same exact E row scored at the unmodified native P",
                "report": ["description_delta", "coordinate_delta", "complete_row_delta"],
                "primary_segment": "coordinates",
            }
            for arm, appended in (
                (sut.ARM_NEUTRAL, neutral),
                (sut.ARM_CLEAN_REPLAY, clean),
            ):
                requests.append(
                    _make_request(
                        arm=arm,
                        cohort_role=cohort_role,
                        gt_owner_id=gt_owner_id,
                        image_id=image_id,
                        context_id=_context(boundary),
                        context_role="row_boundary",
                        boundary_index=boundary,
                        base_prefix_token_ids=prefixes[boundary],
                        appended_token_ids=appended,
                        scored_target=scored_target,
                        baseline_context_id=_context(boundary),
                        sealed_reference=reference,
                    )
                )

        control_owner = f"gt:{image_id}:tp"
        twin = _row(7000 + image_index)
        following = rows[1]
        reference = _sealed_reference(400 + image_index)
        benign_rows.append(
            {
                "schema_version": plan_builder.BENIGN_SCHEMA_VERSION,
                "row_kind": "benign_reference_control",
                "unit_id": sut.UNIT_ID,
                "cohort_role": plan_builder.COHORT_BENIGN,
                "gt_owner_id": control_owner,
                "image_id": image_id,
                "normalized_description": "tp",
                "due_context_id": _context(0),
                "replaced_native_row_index": 0,
                "following_native_action": {
                    "context_id": _context(1),
                    "native_row_index": 1,
                    "token_ids": following,
                    "token_ids_sha256": _sha256_json(following),
                    "token_count": len(following),
                },
                "inserted_clean_row_c": {
                    "token_ids": twin,
                    "token_ids_sha256": _sha256_json(twin),
                    "token_count": len(twin),
                },
                "sealed_benign_reference": reference,
                "request_ids": [],
            }
        )
        requests.append(
            _make_request(
                arm=sut.ARM_BENIGN,
                cohort_role=plan_builder.COHORT_BENIGN,
                gt_owner_id=control_owner,
                image_id=image_id,
                context_id=_context(0),
                context_role="root",
                boundary_index=0,
                base_prefix_token_ids=prefixes[0],
                appended_token_ids=twin,
                scored_target={
                    "kind": "exact_native_row",
                    "token_ids": list(following),
                    "token_ids_sha256": _sha256_json(following),
                    "native_row_index": 1,
                    "baseline_context_id": _context(1),
                    "successor_context_id": _context(2),
                    "compare_against": "the unmodified native successor context",
                    "report": ["description_delta", "coordinate_delta", "complete_row_delta"],
                    "primary_segment": "coordinates",
                },
                baseline_context_id=_context(1),
                sealed_reference=reference,
            )
        )

    manifest = {
        "schema_version": plan_builder.MANIFEST_SCHEMA_VERSION,
        "unit_id": sut.UNIT_ID,
        "manifest_content_sha256": "plan-manifest-digest",
        "builder_source": {"path": "builder.py", "sha256": "b" * 64},
        "cohort": {
            "voting_owner_count": 9,
            "specificity_owner_count": 12,
            "infeasible_owner_count": 5,
            "executed_owner_count": 21,
            "benign_control_count": 12,
            "image_count": 12,
            "image_ids": list(IMAGE_IDS),
        },
        "gates": {"specificity_material_max": 4},
        "routes": {"order": list(plan_builder.ROUTE_ORDER)},
        "lineage": {"crossing_plan_dir": "/nonexistent/crossing"},
    }
    crossing = crossing_scorer.SealedPlan(
        plan_dir=Path("/nonexistent/crossing"),
        manifest={"manifest_content_sha256": "crossing-digest"},
        cohort_rows=[],
        control_rows=[],
        request_rows=[],
        plan_file_sha256={},
        inputs=_StubInputs(contexts, images),
        candidates_by_id={},
        owners_by_image={},
    )
    return sut.SealedNeutralPlan(
        plan_dir=Path("/nonexistent/plan"),
        manifest=manifest,
        selection_rows=selection_rows,
        benign_rows=benign_rows,
        request_rows=requests,
        plan_file_sha256={plan_builder.REQUEST_PLAN_NAME: "a" * 64},
        crossing=crossing,
    )


@pytest.fixture(scope="module")
def sealed_plan() -> sut.SealedNeutralPlan:
    return _synthetic_plan()


@pytest.fixture()
def plan(sealed_plan: sut.SealedNeutralPlan) -> sut.SealedNeutralPlan:
    """A deep-enough copy so a mutating test cannot leak into its neighbours."""

    return sut.SealedNeutralPlan(
        plan_dir=sealed_plan.plan_dir,
        manifest=copy.deepcopy(sealed_plan.manifest),
        selection_rows=copy.deepcopy(sealed_plan.selection_rows),
        benign_rows=copy.deepcopy(sealed_plan.benign_rows),
        request_rows=copy.deepcopy(sealed_plan.request_rows),
        plan_file_sha256=dict(sealed_plan.plan_file_sha256),
        crossing=crossing_scorer.SealedPlan(
            plan_dir=sealed_plan.crossing.plan_dir,
            manifest=dict(sealed_plan.crossing.manifest),
            cohort_rows=[],
            control_rows=[],
            request_rows=[],
            plan_file_sha256={},
            inputs=_StubInputs(
                copy.deepcopy(sealed_plan.crossing.inputs.census.contexts_by_id),
                copy.deepcopy(sealed_plan.crossing.inputs.images_by_id),
            ),
            candidates_by_id={},
            owners_by_image={},
        ),
    )


def _request_of(plan: sut.SealedNeutralPlan, *, image_id: str, arm: str) -> dict[str, Any]:
    for row in plan.request_rows:
        if str(row["image_id"]) == image_id and str(row["arm"]) == arm:
            return row
    raise AssertionError(f"no {arm!r} request on image {image_id!r}")


def _registry_of(plan: sut.SealedNeutralPlan, gt_owner_id: str) -> dict[str, Any]:
    return plan.registry_by_owner[gt_owner_id]


# ---------------------------------------------------------------------------
# 1. Frozen identities
# ---------------------------------------------------------------------------


def test_unit_arms_and_artifact_names_are_frozen() -> None:
    assert sut.UNIT_ID == (
        "2026-08-04-sorted-crossing-matched-length-neutral-row-insertion-control"
    )
    assert sut.ARMS == ("p_plus_n_then_e", "p_plus_c_then_e", "benign_substitution_"
                        "then_following_native_action")
    assert sut.PAIRED_CROSSING_ARMS == (sut.ARM_NEUTRAL, sut.ARM_CLEAN_REPLAY)
    assert sut.ROWS_NAME == "neutral-row-control-rows.jsonl"
    assert sut.EVIDENCE_OUTPUT_NAMES == (sut.ROWS_NAME, sut.PARITY_NAME)
    assert sut.PAIRED_ROOTS == paired.PAIRED_ROOTS
    assert sut.SEGMENTS == paired.SEGMENTS
    assert sut.CACHE_PARITY_MAX_SELECTED_LOGIT_ABS_DIFF == 1e-3


def test_frozen_request_census_is_21_21_12(plan: sut.SealedNeutralPlan) -> None:
    counts = sut.validate_plan_counts(plan)
    assert counts["observed_total"] == 54
    assert counts["observed_by_arm"] == {
        sut.ARM_NEUTRAL: 21,
        sut.ARM_CLEAN_REPLAY: 21,
        sut.ARM_BENIGN: 12,
    }
    assert counts["expected_by_arm"] == dict(sut.EXPECTED_REQUEST_COUNT_BY_ARM)


def test_a_missing_request_of_any_arm_fails_closed(plan: sut.SealedNeutralPlan) -> None:
    plan.request_rows.pop()
    with pytest.raises(sut.NeutralRowScoreContractError, match="not the frozen 54"):
        sut.validate_plan_counts(plan)


def test_a_duplicated_request_fails_closed(plan: sut.SealedNeutralPlan) -> None:
    plan.request_rows.append(copy.deepcopy(plan.request_rows[0]))
    with pytest.raises(sut.NeutralRowScoreContractError, match="duplicate request"):
        sut.validate_plan_counts(plan)


def test_a_drifted_cohort_denominator_fails_closed(plan: sut.SealedNeutralPlan) -> None:
    plan.manifest["cohort"]["voting_owner_count"] = 8
    with pytest.raises(sut.NeutralRowScoreContractError, match="voting_owner_count"):
        sut.validate_plan_counts(plan)


def test_a_prohibited_image_in_the_plan_fails_closed(plan: sut.SealedNeutralPlan) -> None:
    plan.manifest["cohort"]["image_ids"] = [*IMAGE_IDS, "2299"]
    with pytest.raises(sut.NeutralRowScoreContractError, match="prohibited image"):
        sut.validate_plan_counts(plan)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("unit_id", "another-unit", "belongs to another unit"),
        ("arm", "invented_arm", "unknown arm"),
        ("request_family", "invented_family", "declares family"),
        ("cohort_role", plan_builder.COHORT_BENIGN, "cohort role"),
        ("inspects_new_model_logits", True, "inspected a new model logit"),
    ],
)
def test_assert_request_fails_closed_on_a_drifted_contract(
    plan: sut.SealedNeutralPlan, field: str, value: Any, match: str
) -> None:
    request = dict(_request_of(plan, image_id=RUNTIME_IMAGE_ID, arm=sut.ARM_NEUTRAL))
    request[field] = value
    with pytest.raises(sut.NeutralRowScoreContractError, match=match):
        sut.assert_request(request)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("appended_role", "something_else", "appends role"),
        ("retokenized", True, "retokenized prefix"),
    ],
)
def test_assert_request_fails_closed_on_a_drifted_prefix(
    plan: sut.SealedNeutralPlan, field: str, value: Any, match: str
) -> None:
    request = copy.deepcopy(_request_of(plan, image_id=RUNTIME_IMAGE_ID, arm=sut.ARM_NEUTRAL))
    request["prefix"][field] = value
    with pytest.raises(sut.NeutralRowScoreContractError, match=match):
        sut.assert_request(request)


def test_one_image_selection_is_reference_blind(plan: sut.SealedNeutralPlan) -> None:
    rows = sut.requests_for_image(plan.request_rows, image_id=RUNTIME_IMAGE_ID)
    assert {row["arm"] for row in rows} == set(sut.ARMS)
    assert [row["request_id"] for row in rows] == sorted(
        row["request_id"] for row in rows
    )
    # The benign-only image still yields exactly its reference pair.
    lone = sut.requests_for_image(plan.request_rows, image_id="7511")
    assert [row["arm"] for row in lone] == [sut.ARM_BENIGN]


def test_an_uncovered_image_fails_closed_and_names_the_covered_ones(
    plan: sut.SealedNeutralPlan,
) -> None:
    with pytest.raises(sut.NeutralRowScoreContractError, match="has no sealed request"):
        sut.requests_for_image(plan.request_rows, image_id="2299")


# ---------------------------------------------------------------------------
# 2. Scored-token identity across the arms
# ---------------------------------------------------------------------------


def test_scored_e_tokens_must_be_identical_across_the_two_arms(
    plan: sut.SealedNeutralPlan,
) -> None:
    result = sut.assert_scored_token_identity(plan.request_rows)
    assert result["checked_owner_count"] == 21
    assert result["identical_scored_e_tokens_across_arms"] is True


def test_a_diverged_scored_row_between_arms_fails_closed() -> None:
    rows = [
        {
            "arm": sut.ARM_NEUTRAL,
            "gt_owner_id": "gt:x:1",
            "scored_token_ids_sha256": "a" * 64,
        },
        {
            "arm": sut.ARM_CLEAN_REPLAY,
            "gt_owner_id": "gt:x:1",
            "scored_token_ids_sha256": "b" * 64,
        },
    ]
    with pytest.raises(sut.NeutralRowScoreContractError, match="different E token digests"):
        sut.assert_scored_token_identity(rows)


def test_the_benign_arm_never_participates_in_the_identity_gate() -> None:
    rows = [
        {
            "arm": sut.ARM_BENIGN,
            "gt_owner_id": "gt:x:tp",
            "scored_token_ids_sha256": "c" * 64,
        }
    ]
    assert sut.assert_scored_token_identity(rows)["checked_owner_count"] == 0


# ---------------------------------------------------------------------------
# 3. Paired-root resolution
# ---------------------------------------------------------------------------


def test_crossing_arms_append_to_the_same_boundary_they_score(
    plan: sut.SealedNeutralPlan,
) -> None:
    for arm in sut.PAIRED_CROSSING_ARMS:
        request = _request_of(plan, image_id=RUNTIME_IMAGE_ID, arm=arm)
        registry = _registry_of(plan, str(request["gt_owner_id"]))
        roots = sut.resolve_paired_roots(plan, request, registry)
        assert roots.baseline_context_id == roots.modified_context_id
        assert roots.native_row_index == int(request["boundary_index"])
        assert roots.registry_source == "selection_registry.scored_e_row"


def test_benign_pairs_against_the_unmodified_native_successor(
    plan: sut.SealedNeutralPlan,
) -> None:
    request = _request_of(plan, image_id=RUNTIME_IMAGE_ID, arm=sut.ARM_BENIGN)
    registry = _registry_of(plan, str(request["gt_owner_id"]))
    roots = sut.resolve_paired_roots(plan, request, registry)
    assert roots.modified_context_id != roots.baseline_context_id
    assert int(plan.context(roots.modified_context_id)["boundary_index"]) + 1 == (
        roots.native_row_index
    )
    assert roots.registry_source == "benign_registry.following_native_action"


def test_a_baseline_that_disagrees_with_the_boundary_convention_fails_closed(
    plan: sut.SealedNeutralPlan,
) -> None:
    request = copy.deepcopy(_request_of(plan, image_id=RUNTIME_IMAGE_ID, arm=sut.ARM_NEUTRAL))
    registry = copy.deepcopy(_registry_of(plan, str(request["gt_owner_id"])))
    registry["scored_e_row"]["pre_row_context_id"] = f"{RUNTIME_IMAGE_ID}:boundary-000"
    request["scored_target"]["baseline_context_id"] = f"{RUNTIME_IMAGE_ID}:boundary-000"
    with pytest.raises(sut.NeutralRowScoreContractError, match="boundary convention"):
        sut.resolve_paired_roots(plan, request, registry)


def test_a_request_whose_declared_baseline_leaves_the_registry_fails_closed(
    plan: sut.SealedNeutralPlan,
) -> None:
    request = copy.deepcopy(_request_of(plan, image_id=RUNTIME_IMAGE_ID, arm=sut.ARM_NEUTRAL))
    request["scored_target"]["baseline_context_id"] = f"{RUNTIME_IMAGE_ID}:boundary-002"
    registry = _registry_of(plan, str(request["gt_owner_id"]))
    with pytest.raises(sut.NeutralRowScoreContractError, match="its registry seals"):
        sut.resolve_paired_roots(plan, request, registry)


def test_a_target_that_is_not_the_literal_boundary_suffix_fails_closed(
    plan: sut.SealedNeutralPlan,
) -> None:
    request = copy.deepcopy(_request_of(plan, image_id=RUNTIME_IMAGE_ID, arm=sut.ARM_NEUTRAL))
    request["scored_target"]["token_ids"] = [*request["scored_target"]["token_ids"], BOX_END]
    registry = _registry_of(plan, str(request["gt_owner_id"]))
    with pytest.raises(sut.NeutralRowScoreContractError, match="literal token suffix"):
        sut.resolve_paired_roots(plan, request, registry)


def test_a_request_joined_to_the_wrong_owner_or_image_fails_closed(
    plan: sut.SealedNeutralPlan,
) -> None:
    request = _request_of(plan, image_id=RUNTIME_IMAGE_ID, arm=sut.ARM_NEUTRAL)
    other = copy.deepcopy(_registry_of(plan, str(request["gt_owner_id"])))
    other["gt_owner_id"] = "gt:other:1"
    with pytest.raises(sut.NeutralRowScoreContractError, match="wrong registry owner row"):
        sut.resolve_paired_roots(plan, request, other)
    other = copy.deepcopy(_registry_of(plan, str(request["gt_owner_id"])))
    other["image_id"] = "9999"
    with pytest.raises(sut.NeutralRowScoreContractError, match="registered on image"):
        sut.resolve_paired_roots(plan, request, other)


def test_a_cohort_role_that_disagrees_with_the_registry_fails_closed(
    plan: sut.SealedNeutralPlan,
) -> None:
    request = _request_of(plan, image_id=RUNTIME_IMAGE_ID, arm=sut.ARM_NEUTRAL)
    registry = copy.deepcopy(_registry_of(plan, str(request["gt_owner_id"])))
    registry["cohort_role"] = plan_builder.COHORT_SPECIFICITY
    if str(request["cohort_role"]) == plan_builder.COHORT_SPECIFICITY:
        registry["cohort_role"] = plan_builder.COHORT_VOTING
    with pytest.raises(sut.NeutralRowScoreContractError, match="registered in cohort role"):
        sut.resolve_paired_roots(plan, request, registry)


# ---------------------------------------------------------------------------
# 4. Appended-row attribution
# ---------------------------------------------------------------------------


def test_each_arm_appends_its_own_sealed_row(plan: sut.SealedNeutralPlan) -> None:
    for arm, key in (
        (sut.ARM_NEUTRAL, "neutral_row_n"),
        (sut.ARM_CLEAN_REPLAY, "inserted_clean_row_c"),
    ):
        request = _request_of(plan, image_id=RUNTIME_IMAGE_ID, arm=arm)
        registry = _registry_of(plan, str(request["gt_owner_id"]))
        digest = sut.assert_appended_row_matches_registry(request, registry)
        assert digest == registry[key]["token_ids_sha256"]


def test_an_appended_row_that_is_not_the_sealed_one_fails_closed(
    plan: sut.SealedNeutralPlan,
) -> None:
    request = copy.deepcopy(_request_of(plan, image_id=RUNTIME_IMAGE_ID, arm=sut.ARM_NEUTRAL))
    registry = _registry_of(plan, str(request["gt_owner_id"]))
    request["prefix"]["appended_token_ids_sha256"] = "f" * 64
    with pytest.raises(sut.NeutralRowScoreContractError, match="appends a row hashing to"):
        sut.assert_appended_row_matches_registry(request, registry)


def test_the_neutral_arm_never_borrows_the_clean_rows_identity(
    plan: sut.SealedNeutralPlan,
) -> None:
    request = copy.deepcopy(_request_of(plan, image_id=RUNTIME_IMAGE_ID, arm=sut.ARM_NEUTRAL))
    registry = _registry_of(plan, str(request["gt_owner_id"]))
    clean = registry["inserted_clean_row_c"]
    request["prefix"]["appended_token_ids"] = list(clean["token_ids"])
    request["prefix"]["appended_token_ids_sha256"] = clean["token_ids_sha256"]
    with pytest.raises(sut.NeutralRowScoreContractError, match="neutral control row N"):
        sut.assert_appended_row_matches_registry(request, registry)


def test_an_appended_row_without_the_frozen_grammar_fails_closed(
    plan: sut.SealedNeutralPlan,
) -> None:
    request = copy.deepcopy(_request_of(plan, image_id=RUNTIME_IMAGE_ID, arm=sut.ARM_NEUTRAL))
    registry = copy.deepcopy(_registry_of(plan, str(request["gt_owner_id"])))
    broken = list(request["prefix"]["appended_token_ids"])[:-1]
    request["prefix"]["appended_token_ids"] = broken
    request["prefix"]["appended_token_ids_sha256"] = _sha256_json(broken)
    registry["neutral_row_n"]["token_ids"] = broken
    registry["neutral_row_n"]["token_ids_sha256"] = _sha256_json(broken)
    with pytest.raises(sut.NeutralRowScoreContractError, match="complete"):
        sut.assert_appended_row_matches_registry(request, registry)


# ---------------------------------------------------------------------------
# 5. Real capture over the deterministic backend
# ---------------------------------------------------------------------------


@pytest.fixture()
def backend() -> Any:
    return shard.FakeCensusBackend(seed="neutral-row-test")


def _runtime_identity(image_id: str = RUNTIME_IMAGE_ID) -> dict[str, Any]:
    return {
        "model_identity": {"name": "fake"},
        "tokenizer_identity": {"name": "fake"},
        "source_identity": sut.source_identity(),
        "session_image_id": image_id,
    }


def _smoke(plan: sut.SealedNeutralPlan, backend: Any) -> sut.ShardResult:
    """The one real four-strata smoke every capture inherits from."""

    return sut.run_smoke_shard(
        plan=plan,
        backend=backend,
        shard_id="smoke",
        image_id=SMOKE_IMAGE_ID,
        runtime_identity=_runtime_identity(SMOKE_IMAGE_ID),
    )


def test_a_smoke_shard_proves_parity_and_seals_an_admission(
    plan: sut.SealedNeutralPlan, backend: Any
) -> None:
    result = _smoke(plan, backend)
    assert result.quarantine is None
    assert result.rows == []
    assert result.admission is not None
    assert result.admission["cache_admitted"] is True
    assert result.admission["smoke_image_id"] == SMOKE_IMAGE_ID
    assert result.admission["smoke_arms"] == {
        sut.ARM_NEUTRAL: EXECUTED_PER_IMAGE[SMOKE_IMAGE_ID],
        sut.ARM_CLEAN_REPLAY: EXECUTED_PER_IMAGE[SMOKE_IMAGE_ID],
        sut.ARM_BENIGN: 1,
    }
    files = sut.shard_output_files(result)
    assert set(files) == {sut.ADMISSION_NAME, sut.PARITY_NAME, sut.RECEIPT_NAME}
    assert sut.ROWS_NAME not in files


def test_the_smoke_seals_a_representative_owner_and_requests_per_stratum(
    plan: sut.SealedNeutralPlan, backend: Any
) -> None:
    """unit.md's four strata, each carried by a named executed owner."""

    result = _smoke(plan, backend)
    assert result.admission is not None
    strata = result.admission["smoke_strata"]
    assert strata["required"] == list(sut.REQUIRED_SMOKE_STRATA)
    assert sorted(strata["representatives"]) == sorted(sut.REQUIRED_SMOKE_STRATA)
    for stratum, representative in strata["representatives"].items():
        assert stratum in {
            representative["e_stratum"],
            representative["description_relation"],
        }
        assert representative["request_ids"] == sorted(
            str(row["request_id"])
            for row in plan.request_rows
            if str(row["gt_owner_id"]) == representative["gt_owner_id"]
            and str(row["arm"]) in sut.PAIRED_CROSSING_ARMS
        )
        assert len(representative["request_ids"]) == 2
    assert {entry["gt_owner_id"] for entry in strata["owner_strata"]} == {
        str(row["gt_owner_id"])
        for row in plan.request_rows
        if str(row["image_id"]) == SMOKE_IMAGE_ID
        and str(row["arm"]) in sut.PAIRED_CROSSING_ARMS
    }
    # The seal is inside the admission digest, so it cannot be added afterwards.
    unsealed = {
        key: value
        for key, value in result.admission.items()
        if key != "admission_content_sha256"
    }
    assert crossing_scorer.sha256_json(unsealed) == result.admission[
        "admission_content_sha256"
    ]


def test_a_smoke_image_that_misses_a_stratum_fails_closed(
    plan: sut.SealedNeutralPlan, backend: Any
) -> None:
    """The real one-owner images carry two strata, so none of them may smoke."""

    with pytest.raises(sut.NeutralRowScoreContractError, match="carries no"):
        sut.run_smoke_shard(
            plan=plan,
            backend=backend,
            shard_id="smoke",
            image_id=RUNTIME_IMAGE_ID,
            runtime_identity=_runtime_identity(),
        )


def test_a_smoke_image_missing_an_arm_fails_closed(
    plan: sut.SealedNeutralPlan, backend: Any
) -> None:
    with pytest.raises(sut.NeutralRowScoreContractError, match="no sealed request for arm"):
        sut.run_smoke_shard(
            plan=plan,
            backend=backend,
            shard_id="smoke",
            image_id="7511",
            runtime_identity=_runtime_identity(),
        )


def test_a_capture_shard_scores_both_roots_of_every_request(
    plan: sut.SealedNeutralPlan, backend: Any
) -> None:
    smoke = _smoke(plan, backend)
    result = sut.run_capture_shard(
        plan=plan,
        backend=backend,
        shard_id="capture",
        session_image_id=RUNTIME_IMAGE_ID,
        admission=smoke.admission,
        runtime_identity=_runtime_identity(),
    )
    assert result.quarantine is None
    assert len(result.rows) == 3
    assert sut.counts_by_arm(result.rows) == {
        sut.ARM_NEUTRAL: 1,
        sut.ARM_CLEAN_REPLAY: 1,
        sut.ARM_BENIGN: 1,
    }
    for row in result.rows:
        assert sorted(row["roots"]) == sorted(sut.PAIRED_ROOTS)
        assert row["retokenized"] is False
        assert row["uses_model_generate"] is False
        assert row["delta_orientation"] == "modified_minus_baseline"
        assert row["primary_segment"] == "coordinates"
        assert row["sealed_reference"]["role"] == "gate_reference_only_never_the_estimand"
        assert set(row["deltas"]) == set(sut.SEGMENTS)
        baseline = row["roots"][sut.ROOT_BASELINE]
        modified = row["roots"][sut.ROOT_MODIFIED]
        assert baseline["appended_token_count"] == 0
        assert modified["appended_token_count"] == row["appended_row_token_count"]
        for segment in sut.SEGMENTS:
            assert row["deltas"][segment]["delta"] == pytest.approx(
                modified["segment_sums"][segment]["sum"]
                - baseline["segment_sums"][segment]["sum"]
            )
    files = sut.shard_output_files(result)
    assert set(files) == {sut.ROWS_NAME, sut.PARITY_NAME, sut.RECEIPT_NAME}


def test_both_arms_of_one_owner_score_identical_tokens_in_a_real_shard(
    plan: sut.SealedNeutralPlan, backend: Any
) -> None:
    smoke = _smoke(plan, backend)
    result = sut.run_capture_shard(
        plan=plan,
        backend=backend,
        shard_id="capture",
        session_image_id=RUNTIME_IMAGE_ID,
        admission=smoke.admission,
        runtime_identity=_runtime_identity(),
    )
    crossing_rows = [row for row in result.rows if row["arm"] in sut.PAIRED_CROSSING_ARMS]
    assert len(crossing_rows) == 2
    assert len({row["scored_token_ids_sha256"] for row in crossing_rows}) == 1
    assert len({tuple(row["scored_token_ids"]) for row in crossing_rows}) == 1
    # The two arms differ only in what they append.
    assert len({row["appended_row_token_ids_sha256"] for row in crossing_rows}) == 2
    # Both arms share one unmodified native baseline root.
    assert len(
        {
            row["roots"][sut.ROOT_BASELINE]["segment_sums"]["coordinates"]["sum"]
            for row in crossing_rows
        }
    ) == 1


def test_a_capture_never_forwards_another_images_context(
    plan: sut.SealedNeutralPlan, backend: Any
) -> None:
    smoke = _smoke(plan, backend)
    foreign = copy.deepcopy(_request_of(plan, image_id="13348", arm=sut.ARM_BENIGN))
    foreign["image_id"] = RUNTIME_IMAGE_ID
    state = sut.CaptureState()
    with pytest.raises(sut.NeutralRowScoreContractError):
        sut._capture_request(
            plan,
            backend,
            state,
            shard_id="capture",
            session_image_id=RUNTIME_IMAGE_ID,
            request=foreign,
            registry_row=_registry_of(plan, str(foreign["gt_owner_id"])),
            enforce_replay=False,
        )
    assert smoke.admission is not None


def test_replay_enforcement_quarantines_and_withholds_evidence(
    plan: sut.SealedNeutralPlan, backend: Any
) -> None:
    """An evidence-bearing backend whose native replay fails publishes no rows."""

    smoke = _smoke(plan, _FakeEvidenceBackend(backend))
    assert smoke.quarantine is not None
    files = sut.shard_output_files(smoke)
    assert set(files) == {sut.QUARANTINE_NAME, sut.RECEIPT_NAME}
    assert sut.ROWS_NAME not in files
    assert files[sut.QUARANTINE_NAME]


class _FakeEvidenceBackend:
    """The deterministic backend, declared evidence-bearing so replay is enforced."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    @property
    def identity(self) -> dict[str, Any]:
        return dict(REAL_BACKEND_IDENTITY)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def test_receipt_policy_declares_the_analysis_boundary(
    plan: sut.SealedNeutralPlan, backend: Any
) -> None:
    smoke = _smoke(plan, backend)
    result = sut.run_capture_shard(
        plan=plan,
        backend=backend,
        shard_id="capture",
        session_image_id=RUNTIME_IMAGE_ID,
        admission=smoke.admission,
        runtime_identity=_runtime_identity(),
    )
    policy = result.receipt["policy"]
    assert policy["relative_estimand_computed_here"] is False
    assert policy["materiality_decided_here"] is False
    assert policy["route_decided_here"] is False
    assert policy["free_decode"] is False
    assert policy["greedy_coordinate_decode"] is False
    assert policy["scored_e_tokens_identical_across_n_and_c_arms"] is True
    assert policy["request_selection_uses_sealed_reference_values"] is False
    receipt = json.loads(json.dumps(result.receipt))
    declared = receipt.pop("receipt_content_sha256")
    assert crossing_scorer.sha256_json(receipt) == declared


def test_an_admission_from_another_plan_is_refused(
    plan: sut.SealedNeutralPlan, backend: Any
) -> None:
    smoke = _smoke(plan, backend)
    admission = dict(smoke.admission or {})
    admission["plan_manifest_content_sha256"] = "another-plan"
    admission.pop("admission_content_sha256")
    admission["admission_content_sha256"] = crossing_scorer.sha256_json(admission)
    with pytest.raises(sut.NeutralRowScoreContractError, match="different plan manifest"):
        sut.validate_admission_receipt(
            admission, plan=plan, runtime_identity=_runtime_identity()
        )


def test_an_edited_admission_that_did_not_reseal_is_refused(
    plan: sut.SealedNeutralPlan, backend: Any
) -> None:
    smoke = _smoke(plan, backend)
    admission = dict(smoke.admission or {})
    admission["cache_admitted"] = False
    with pytest.raises(sut.NeutralRowScoreContractError, match="does not self-seal"):
        sut.validate_admission_receipt(
            admission, plan=plan, runtime_identity=_runtime_identity()
        )


def test_an_admission_that_proved_fewer_strata_cannot_admit_a_capture(
    plan: sut.SealedNeutralPlan, backend: Any
) -> None:
    """A resealed receipt that drops a stratum still never admits a capture."""

    smoke = _smoke(plan, backend)
    admission = copy.deepcopy(dict(smoke.admission or {}))
    admission["smoke_strata"]["representatives"].pop(sut.REQUIRED_SMOKE_STRATA[0])
    admission.pop("admission_content_sha256")
    admission["admission_content_sha256"] = crossing_scorer.sha256_json(admission)
    with pytest.raises(sut.NeutralRowScoreContractError, match="proves strata"):
        sut.validate_admission_receipt(
            admission, plan=plan, runtime_identity=_runtime_identity()
        )
