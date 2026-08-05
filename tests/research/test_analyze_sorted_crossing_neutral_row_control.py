"""Contract tests for the neutral-row control frozen-route analysis.

Two layers:

* synthetic truth tables over the pure gate and route functions, so the frozen
  ``-1.0`` cutoff, the absolute specificity threshold, both absolute route
  thresholds and the quarantine arithmetic are pinned exhaustively; and
* an end-to-end analysis over a synthesized sealed plan and merged evidence with
  chosen deltas, so the ordered gates, the same-run benign estimand, the
  sensitivities, the strata and the published artifacts are exercised together.

No model, GPU or production artifact is touched.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Callable

import pytest

from scripts.research import analyze_sorted_crossing_neutral_row_control as sut
from scripts.research import analyze_sorted_crossing_owner_row_geometry as geometry
from scripts.research import merge_sorted_crossing_neutral_row_control as merger
from scripts.research import prepare_sorted_crossing_neutral_row_control as plan_builder
from scripts.research import score_sorted_crossing_boundary_owner_release as crossing_scorer
from scripts.research import score_sorted_crossing_neutral_row_control as scorer

OBJ_START = crossing_scorer.OBJECT_REF_START
OBJ_END = crossing_scorer.OBJECT_REF_END
BOX_START = crossing_scorer.BOX_START
BOX_END = crossing_scorer.BOX_END
COORD_START = crossing_scorer.COORDINATE_TOKEN_ID_START

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
VOTING_SLOTS: tuple[tuple[str, int], ...] = (
    ("13348", 0),
    ("13923", 0),
    ("14038", 0),
    ("1584", 0),
    ("16228", 0),
    ("16228", 1),
    ("4134", 0),
    ("4134", 1),
    ("6040", 0),
)
SEPARATED_SLOTS: frozenset[tuple[str, int]] = frozenset(
    {("13348", 0), ("16228", 0), ("4134", 0), ("4134", 1)}
)


def _sha256_json(value: Any) -> str:
    return crossing_scorer.sha256_json(value)


def _coords(index: int) -> list[int]:
    base = (index * 7) % 900
    return [COORD_START + base + offset for offset in (0, 1, 4, 5)]


def _row(seed: int) -> list[int]:
    description = [20000 + (seed * 13) % 500] * (1 + seed % 2)
    return [OBJ_START, *description, OBJ_END, BOX_START, *_coords(seed), BOX_END]


# ---------------------------------------------------------------------------
# 1. Frozen constants
# ---------------------------------------------------------------------------


def test_the_materiality_cutoff_is_imported_never_redeclared() -> None:
    assert sut.MATERIALITY_MAX_NATS is geometry.MATERIAL_NEGATIVE_MAX_NATS
    assert sut.MATERIALITY_MAX_NATS == -1.0
    assert sut.MATERIALITY_MAX_NATS == plan_builder.MATERIALITY_MAX_NATS


def test_the_gate_order_is_the_frozen_one() -> None:
    assert sut.GATE_ORDER == (
        "input_and_token_identity",
        "runtime_replay",
        "same_run_positive_controls",
        "benign_reference_replay",
        "cached_versus_uncached_parity",
        "neutral_row_specificity",
    )
    assert set(merger.MERGE_OWNED_GATES) | set(merger.ANALYSIS_OWNED_GATES) == set(
        sut.GATE_ORDER
    )


def test_every_route_carries_its_permitted_reading() -> None:
    assert set(sut.ROUTE_READINGS) == {
        sut.ROUTE_C_CONTENT_SPECIFIC,
        sut.ROUTE_GENERIC_SUFFICIENT,
        sut.ROUTE_NEUTRAL_NOT_NEUTRAL,
        sut.ROUTE_INCONCLUSIVE,
    }
    assert "never refuted" in sut.ROUTE_READINGS[sut.ROUTE_GENERIC_SUFFICIENT]
    assert "does not identify" in sut.ROUTE_READINGS[sut.ROUTE_C_CONTENT_SPECIFIC]
    assert "unknown-neutral" in sut.UNKNOWN_NEUTRAL_NOTE
    assert "coverage" in sut.LIKELIHOOD_VERSUS_COVERAGE


@pytest.mark.parametrize(
    ("relative", "expected"),
    [(-1.0001, True), (-1.0, True), (-0.9999, False), (0.0, False), (2.0, False)],
)
def test_materiality_is_at_or_below_minus_one_nat(relative: float, expected: bool) -> None:
    assert sut.is_material(relative) is expected


# ---------------------------------------------------------------------------
# 2. Synthetic owner readouts
# ---------------------------------------------------------------------------


def _readout(
    gt_owner_id: str,
    *,
    image_id: str,
    cohort_role: str,
    neutral_relative: float,
    votes: bool = False,
    clearly_separated: bool = False,
    interpretable: bool = True,
    row_length_delta_tokens: int = 0,
    posthoc_uncertain: bool = False,
    sentinel: bool = False,
    clean_relative: float = -2.0,
    clean_raw: float | None = None,
    sealed_clean_raw: float | None = None,
    sealed_material: bool | None = None,
) -> sut.OwnerReadout:
    clean_raw = clean_relative if clean_raw is None else clean_raw
    sealed_clean_raw = clean_raw if sealed_clean_raw is None else sealed_clean_raw
    sealed_material = (
        sut.is_material(clean_relative) if sealed_material is None else sealed_material
    )
    row = {
        "gt_owner_id": gt_owner_id,
        "image_id": image_id,
        "cohort_role": cohort_role,
        "votes": votes,
        "clearly_separated": clearly_separated,
        "sentinel_owner": sentinel,
        "posthoc_support_extent_uncertain": posthoc_uncertain,
        "interpretable": interpretable,
        "neutral": {
            "relative_coordinate_delta": neutral_relative,
            "material": sut.is_material(neutral_relative),
            "counts_toward_route": interpretable,
            "row_length_delta_tokens": row_length_delta_tokens,
        },
        "clean": {
            "coordinate_delta": clean_raw,
            "coordinate_delta_sign": 1 if clean_raw > 0 else -1,
            "relative_coordinate_delta": clean_relative,
            "material": sut.is_material(clean_relative),
            "counts_toward_route": interpretable,
            "sealed_coordinate_delta": sealed_clean_raw,
            "sealed_coordinate_delta_sign": 1 if sealed_clean_raw > 0 else -1,
            "sealed_material_negative": sealed_material,
        },
        "strata": {
            "e_strict_match": "unmatched",
            "e_stratum": "unmatched_e",
            "description_relation": "different_description",
            "geometric_separation": (
                "clearly_separated" if clearly_separated else "not_clearly_separated"
            ),
            "visual_adjudication": (
                "plausible_support_extent_uncertain" if posthoc_uncertain else "not_flagged"
            ),
        },
    }
    return sut.OwnerReadout(
        gt_owner_id=gt_owner_id,
        image_id=image_id,
        cohort_role=cohort_role,
        votes=votes,
        clearly_separated=clearly_separated,
        row=row,
    )


def _specificity_readouts(material_count: int, *, quarantined: int = 0) -> list[sut.OwnerReadout]:
    readouts: list[sut.OwnerReadout] = []
    for index in range(plan_builder.SPECIFICITY_OWNER_COUNT):
        readouts.append(
            _readout(
                f"gt:spec:{index}",
                image_id=f"img-{index % 4}",
                cohort_role=plan_builder.COHORT_SPECIFICITY,
                neutral_relative=-3.0 if index < material_count else 0.5,
                interpretable=index >= quarantined,
                clean_relative=0.5,
            )
        )
    return readouts


def _voting_readouts(
    *,
    nonmaterial: int,
    nonmaterial_images: int,
    nonmaterial_separated: int,
    nonmaterial_separated_images: int,
    quarantined: int = 0,
) -> list[sut.OwnerReadout]:
    """Nine voting owners with an exactly specified nonmaterial side."""

    readouts: list[sut.OwnerReadout] = []
    separated_left = nonmaterial_separated
    for index in range(plan_builder.VOTING_OWNER_COUNT):
        is_nonmaterial = index < nonmaterial
        if is_nonmaterial:
            image_index = min(index, nonmaterial_images - 1)
            separated = separated_left > 0
            if separated:
                image_index = min(
                    nonmaterial_separated - separated_left,
                    nonmaterial_separated_images - 1,
                )
                separated_left -= 1
            image_id = f"img-{image_index}"
        else:
            separated = False
            image_id = f"img-other-{index}"
        readouts.append(
            _readout(
                f"gt:vote:{index}",
                image_id=image_id,
                cohort_role=plan_builder.COHORT_VOTING,
                votes=True,
                clearly_separated=separated,
                neutral_relative=0.5 if is_nonmaterial else -3.0,
                interpretable=index >= quarantined,
                clean_relative=-2.0,
            )
        )
    return readouts


# ---------------------------------------------------------------------------
# 3. Specificity gate truth table
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("material_count", "passed"),
    [(0, True), (1, True), (2, True), (3, True), (4, False), (5, False), (12, False)],
)
def test_specificity_gate_threshold_is_the_absolute_integer_four(
    material_count: int, passed: bool
) -> None:
    gate = sut.evaluate_specificity_gate(_specificity_readouts(material_count))
    assert gate["material_count"] == material_count
    assert gate["passed"] is passed
    assert gate["threshold"] == 4
    assert gate["threshold_is_absolute"] is True
    assert gate["frozen_denominator"] == 12
    assert gate["failure_route"] == sut.ROUTE_NEUTRAL_NOT_NEUTRAL


def test_a_quarantined_specificity_owner_never_counts_as_material() -> None:
    """Four would-be material owners, two quarantined: the gate still passes."""

    gate = sut.evaluate_specificity_gate(_specificity_readouts(4, quarantined=2))
    assert gate["material_count"] == 2
    assert gate["passed"] is True
    assert gate["denominator"] == 12
    assert gate["frozen_denominator"] == 12


def test_the_specificity_gate_only_reads_the_specificity_stratum() -> None:
    readouts = [*_specificity_readouts(0), *_voting_readouts(
        nonmaterial=0,
        nonmaterial_images=1,
        nonmaterial_separated=0,
        nonmaterial_separated_images=1,
    )]
    gate = sut.evaluate_specificity_gate(readouts)
    assert gate["denominator"] == 12
    assert gate["material_count"] == 0


# ---------------------------------------------------------------------------
# 4. Route truth table
# ---------------------------------------------------------------------------


def _route_of(readouts: list[sut.OwnerReadout], *, specificity_material: int = 0) -> str:
    combined = [*readouts, *_specificity_readouts(specificity_material)]
    specificity = sut.evaluate_specificity_gate(combined)
    return str(sut.decide_route(combined, specificity)["route"])


@pytest.mark.parametrize(
    ("nonmaterial", "images", "separated", "separated_images", "expected"),
    [
        (9, 3, 4, 2, sut.ROUTE_C_CONTENT_SPECIFIC),
        (5, 3, 2, 2, sut.ROUTE_C_CONTENT_SPECIFIC),
        (4, 3, 2, 2, sut.ROUTE_INCONCLUSIVE),
        (5, 2, 2, 2, sut.ROUTE_INCONCLUSIVE),
        (5, 3, 1, 1, sut.ROUTE_INCONCLUSIVE),
        (5, 3, 2, 1, sut.ROUTE_INCONCLUSIVE),
    ],
)
def test_c_content_specific_route_thresholds_are_absolute(
    nonmaterial: int, images: int, separated: int, separated_images: int, expected: str
) -> None:
    readouts = _voting_readouts(
        nonmaterial=nonmaterial,
        nonmaterial_images=images,
        nonmaterial_separated=separated,
        nonmaterial_separated_images=separated_images,
    )
    assert _route_of(readouts) == expected


@pytest.mark.parametrize(
    ("material", "expected"),
    [(9, sut.ROUTE_GENERIC_SUFFICIENT), (7, sut.ROUTE_GENERIC_SUFFICIENT), (6, sut.ROUTE_INCONCLUSIVE)],
)
def test_generic_route_needs_seven_of_nine_material_and_three_separated(
    material: int, expected: str
) -> None:
    readouts: list[sut.OwnerReadout] = []
    for index in range(plan_builder.VOTING_OWNER_COUNT):
        is_material = index < material
        readouts.append(
            _readout(
                f"gt:vote:{index}",
                image_id=f"img-{index % 3}" if is_material else f"img-x-{index}",
                cohort_role=plan_builder.COHORT_VOTING,
                votes=True,
                clearly_separated=is_material and index < 3,
                neutral_relative=-3.0 if is_material else 0.5,
                clean_relative=-2.0,
            )
        )
    assert _route_of(readouts) == expected


def test_the_specificity_gate_pre_empts_every_other_route() -> None:
    """Even a clean C-specific pattern routes to neutral_row_not_neutral."""

    readouts = _voting_readouts(
        nonmaterial=9, nonmaterial_images=3, nonmaterial_separated=4, nonmaterial_separated_images=2
    )
    assert _route_of(readouts, specificity_material=0) == sut.ROUTE_C_CONTENT_SPECIFIC
    assert _route_of(readouts, specificity_material=4) == sut.ROUTE_NEUTRAL_NOT_NEUTRAL


def test_a_quarantined_voting_owner_satisfies_neither_side() -> None:
    readouts = _voting_readouts(
        nonmaterial=5,
        nonmaterial_images=3,
        nonmaterial_separated=2,
        nonmaterial_separated_images=2,
        quarantined=1,
    )
    combined = [*readouts, *_specificity_readouts(0)]
    decision = sut.decide_route(combined, sut.evaluate_specificity_gate(combined))
    assert decision["nonmaterial_side"]["owner_count"] == 4
    assert decision["material_side"]["owner_count"] == 4
    assert decision["route"] == sut.ROUTE_INCONCLUSIVE
    assert decision["denominators_are_absolute"] is True
    assert decision["voting_denominator"] == 9


def test_quarantine_arithmetic_partitions_each_stratum() -> None:
    readouts = [
        *_voting_readouts(
            nonmaterial=5,
            nonmaterial_images=3,
            nonmaterial_separated=2,
            nonmaterial_separated_images=2,
            quarantined=2,
        ),
        *_specificity_readouts(3, quarantined=1),
    ]
    voting = sut._quarantine_arithmetic(readouts, cohort_role=plan_builder.COHORT_VOTING)
    assert voting["denominator"] == 9
    assert voting["neither_count"] == 2
    assert (
        voting["material_count"] + voting["nonmaterial_count"] + voting["neither_count"] == 9
    )
    specificity = sut._quarantine_arithmetic(
        readouts, cohort_role=plan_builder.COHORT_SPECIFICITY
    )
    assert specificity["denominator"] == 12
    assert specificity["neither_count"] == 1
    assert "conservatively satisfies no route" in specificity["neither_semantics"]


# ---------------------------------------------------------------------------
# 5. Gate 3, the same-run positive controls
# ---------------------------------------------------------------------------


def _positive_readouts(
    *, failures: int = 0, sentinel_fails: bool = False
) -> list[sut.OwnerReadout]:
    readouts: list[sut.OwnerReadout] = []
    for index in range(plan_builder.VOTING_OWNER_COUNT):
        fails = index < failures
        readouts.append(
            _readout(
                f"gt:vote:{index}",
                image_id=f"img-{index % 3}",
                cohort_role=plan_builder.COHORT_VOTING,
                votes=True,
                neutral_relative=0.5,
                clean_relative=-2.0,
                clean_raw=-2.0,
                sealed_clean_raw=-2.5 if fails else -2.0,
                sentinel=sentinel_fails and index == 0,
            )
        )
    return readouts


def test_the_positive_control_gate_admits_a_faithful_replay() -> None:
    gate = sut.evaluate_positive_control_gate(_positive_readouts())
    assert gate["failure_count"] == 0
    assert gate["passed"] is True
    assert gate["checked_owner_count"] == 9
    assert gate["mandatory_sentinel_owner_ids"] == list(sut.SENTINEL_OWNER_IDS)
    assert all(entry["passed"] for entry in gate["owner_rows"])
    assert all(
        entry["coordinate_delta_tolerance"] == 0.05 for entry in gate["owner_rows"]
    )


@pytest.mark.parametrize("failures", [1, 2])
def test_up_to_two_positive_control_failures_are_visible_and_allowed(failures: int) -> None:
    gate = sut.evaluate_positive_control_gate(_positive_readouts(failures=failures))
    assert gate["failure_count"] == failures
    assert len(gate["failed_owner_ids"]) == failures
    assert gate["max_failures"] == 2


def test_more_than_two_positive_control_failures_stop_the_unit() -> None:
    with pytest.raises(sut.NeutralRowAnalysisContractError, match="more than the frozen 2"):
        sut.evaluate_positive_control_gate(_positive_readouts(failures=3))


def test_a_failing_sentinel_stops_the_unit_regardless_of_the_total() -> None:
    with pytest.raises(sut.NeutralRowAnalysisContractError, match="sentinel"):
        sut.evaluate_positive_control_gate(
            _positive_readouts(failures=1, sentinel_fails=True)
        )


def test_a_positive_control_that_flips_materiality_is_a_failure() -> None:
    readouts = _positive_readouts()
    flipped = list(readouts)
    flipped[0].row["clean"]["sealed_material_negative"] = False
    gate = sut.evaluate_positive_control_gate(flipped)
    assert gate["failure_count"] == 1
    entry = next(item for item in gate["owner_rows"] if not item["passed"])
    assert entry["materiality_reproduced"] is False
    assert entry["sign_reproduced"] is True


# ---------------------------------------------------------------------------
# 6. End-to-end over synthesized sealed evidence
# ---------------------------------------------------------------------------


class _Fixture:
    """A sealed plan directory plus merged evidence with chosen deltas."""

    def __init__(
        self,
        root: Path,
        *,
        neutral_relative: Callable[[str, bool], float],
        benign_delta: float = 0.0,
        benign_override: dict[str, dict[str, Any]] | None = None,
        quarantined: tuple[str, ...] = (),
        length_delta: Callable[[str, bool], int] | None = None,
    ) -> None:
        self.root = root
        self.plan_dir = root / "plan"
        self.merged_dir = root / "merged"
        self.neutral_relative = neutral_relative
        self.benign_delta = benign_delta
        self.benign_override = benign_override or {}
        self.quarantined = quarantined
        self.length_delta = length_delta or (lambda _owner, _votes: 0)
        self._build()

    def _reference(self, *, raw: float, baseline_sum: float, argmax: list[int], material: bool) -> dict[str, Any]:
        return {
            "coordinate_delta": raw,
            "coordinate_delta_sign": 1 if raw > 0 else (-1 if raw < 0 else 0),
            "baseline_coordinate_sum": baseline_sum,
            "modified_coordinate_sum": baseline_sum + raw,
            "baseline_argmax_token_ids_sha256": _sha256_json(argmax),
            "modified_argmax_token_ids_sha256": _sha256_json(argmax),
            "baseline_argmax_reproduces_description_path": True,
            "baseline_argmax_reproduces_complete_row": True,
            "scored_token_ids_sha256": _sha256_json(argmax),
            "scored_token_count": len(argmax),
            "request_id": "req:sealed",
            "material_negative": material,
            "relative_coordinate_delta": raw - self.benign_delta,
            "clearly_separated": False,
            "role": "gate_reference_only_never_the_estimand",
            "materiality_cutoff_nats": -1.0,
            "replay_max_coordinate_delta_abs_diff": 0.05,
            "replay_max_selected_logit_abs_diff": 0.001,
        }

    def _build(self) -> None:
        selection: list[dict[str, Any]] = []
        benign: list[dict[str, Any]] = []
        requests: list[dict[str, Any]] = []
        merged_rows: list[dict[str, Any]] = []
        quarantined_ids: list[str] = []

        def _request(
            *,
            arm: str,
            cohort_role: str,
            gt_owner_id: str,
            image_id: str,
            context_id: str,
            boundary_index: int,
            appended: list[int],
            scored: list[int],
            baseline_context_id: str,
            successor_context_id: str,
            reference: dict[str, Any],
        ) -> dict[str, Any]:
            family = plan_builder.REQUEST_FAMILY_BY_ARM[arm]
            scored_target = {
                "kind": "exact_native_row",
                "token_ids": list(scored),
                "token_ids_sha256": _sha256_json(scored),
                "native_row_index": boundary_index,
                "baseline_context_id": baseline_context_id,
                "successor_context_id": successor_context_id,
                "compare_against": "the unmodified native context",
                "report": ["description_delta", "coordinate_delta", "complete_row_delta"],
                "primary_segment": "coordinates",
            }
            identity = {
                "unit_id": scorer.UNIT_ID,
                "request_family": family,
                "arm": arm,
                "cohort_role": cohort_role,
                "gt_owner_id": gt_owner_id,
                "context_id": context_id,
                "baseline_context_id": baseline_context_id,
                "appended_token_ids": list(appended),
                "base_prefix_token_ids_sha256": _sha256_json([boundary_index]),
                "scored_target": dict(scored_target),
            }
            digest = _sha256_json(identity)
            return {
                "schema_version": plan_builder.REQUEST_SCHEMA_VERSION,
                "row_kind": "neutral_row_control_request",
                "unit_id": scorer.UNIT_ID,
                "request_id": f"req:{digest[:32]}",
                "request_key": f"{family}|{cohort_role}|{gt_owner_id}|{context_id}|{arm}",
                "request_family": family,
                "arm": arm,
                "cohort_role": cohort_role,
                "gt_owner_id": gt_owner_id,
                "image_id": image_id,
                "context_id": context_id,
                "context_role": "row_boundary",
                "boundary_index": boundary_index,
                "paired_roots": {
                    "baseline_context_id": baseline_context_id,
                    "modified_context_id": context_id,
                    "orientation": "modified_minus_baseline",
                },
                "prefix": {
                    "base_context_id": context_id,
                    "base_prefix_token_count": 1,
                    "base_prefix_token_ids_sha256": _sha256_json([boundary_index]),
                    "appended_token_ids": list(appended),
                    "appended_token_ids_sha256": _sha256_json(appended),
                    "appended_token_count": len(appended),
                    "appended_role": plan_builder.APPENDED_ROLE_BY_ARM[arm],
                    "retokenized": False,
                },
                "scored_target": scored_target,
                "sealed_reference": dict(reference),
                "score_blind_plan": True,
                "inspects_new_model_logits": False,
                "identity_digest": digest,
            }

        def _merged_row(request: dict[str, Any], *, raw: float, baseline_sum: float) -> dict[str, Any]:
            scored = list(request["scored_target"]["token_ids"])
            return {
                "schema_version": scorer.SCHEMA_VERSION,
                "row_kind": "neutral_row_control_row",
                "unit_id": scorer.UNIT_ID,
                "request_id": request["request_id"],
                "arm": request["arm"],
                "cohort_role": request["cohort_role"],
                "gt_owner_id": request["gt_owner_id"],
                "image_id": request["image_id"],
                "scored_token_ids": scored,
                "scored_token_ids_sha256": _sha256_json(scored),
                "scored_token_count": len(scored),
                "sealed_reference": dict(request["sealed_reference"]),
                "deltas": {
                    "coordinates": {
                        "delta": raw,
                        "sign": 1 if raw > 0 else (-1 if raw < 0 else 0),
                    },
                    "complete_row": {"delta": raw * 1.1, "sign": 1 if raw > 0 else -1},
                    "description": {"delta": raw * 0.1, "sign": 1 if raw > 0 else -1},
                },
                "roots": {
                    scorer.ROOT_BASELINE: {
                        "segment_sums": {"coordinates": {"sum": baseline_sum}},
                        "argmax_token_ids": scored,
                    },
                    scorer.ROOT_MODIFIED: {
                        "segment_sums": {"coordinates": {"sum": baseline_sum + raw}},
                        "argmax_token_ids": scored,
                    },
                },
            }

        for image_index, image_id in enumerate(IMAGE_IDS):
            for slot in range(EXECUTED_PER_IMAGE[image_id]):
                boundary = 3 + 2 * slot
                gt_owner_id = f"gt:{image_id}:{boundary}"
                votes = (image_id, slot) in VOTING_SLOTS
                cohort_role = (
                    plan_builder.COHORT_VOTING if votes else plan_builder.COHORT_SPECIFICITY
                )
                scored = _row(image_index * 31 + boundary)
                clean = _row(5000 + image_index * 7 + slot)
                neutral = _row(9000 + image_index * 7 + slot)
                baseline_sum = -10.0 - image_index
                clean_raw = (-2.0 if votes else 0.4) + self.benign_delta
                reference = self._reference(
                    raw=clean_raw,
                    baseline_sum=baseline_sum,
                    argmax=scored,
                    material=votes,
                )
                context_id = f"{image_id}:boundary-{boundary:03d}"
                selection.append(
                    {
                        "schema_version": plan_builder.SELECTION_SCHEMA_VERSION,
                        "row_kind": "neutral_row_selection",
                        "unit_id": scorer.UNIT_ID,
                        "gt_owner_id": gt_owner_id,
                        "image_id": image_id,
                        "normalized_description": f"targ-{boundary}",
                        "cohort_role": cohort_role,
                        "votes": votes,
                        "executed": True,
                        "material_negative": votes,
                        "clearly_separated": (image_id, slot) in SEPARATED_SLOTS,
                        "sentinel_owner": False,
                        "posthoc_support_extent_uncertain": (image_id, slot) == ("4134", 1),
                        "crossing": {
                            "boundary_index_b": boundary,
                            "p_context_id": context_id,
                        },
                        "inserted_clean_row_c": {
                            "token_ids": clean,
                            "token_ids_sha256": _sha256_json(clean),
                            "token_count": len(clean),
                        },
                        "scored_e_row": {
                            "row_index": boundary,
                            "token_ids": scored,
                            "token_ids_sha256": _sha256_json(scored),
                            "strict_match_status": "matched" if slot == 0 else "unmatched",
                            "normalized_description": (
                                f"targ-{boundary}" if slot == 1 else "erow"
                            ),
                            "pre_row_context_id": context_id,
                            "post_row_context_id": f"{image_id}:boundary-{boundary + 1:03d}",
                        },
                        "neutral_row_n": {
                            "gt_owner_id": f"gt:{image_id}:n{slot}",
                            "token_ids": neutral,
                            "token_ids_sha256": _sha256_json(neutral),
                            "token_count": len(neutral),
                            "row_length_delta_tokens": self.length_delta(gt_owner_id, votes),
                            "rows_back_distance": 2,
                            "min_center_distance_normalized": 0.2,
                        },
                        "sealed_clean_reference": reference,
                        "request_ids": [],
                    }
                )
                observed_baseline = (
                    baseline_sum + 4.0 if gt_owner_id in self.quarantined else baseline_sum
                )
                if gt_owner_id in self.quarantined:
                    quarantined_ids.append(gt_owner_id)
                for arm, appended, raw in (
                    (
                        scorer.ARM_NEUTRAL,
                        neutral,
                        self.neutral_relative(gt_owner_id, votes) + self.benign_delta,
                    ),
                    (scorer.ARM_CLEAN_REPLAY, clean, clean_raw),
                ):
                    request = _request(
                        arm=arm,
                        cohort_role=cohort_role,
                        gt_owner_id=gt_owner_id,
                        image_id=image_id,
                        context_id=context_id,
                        boundary_index=boundary,
                        appended=appended,
                        scored=scored,
                        baseline_context_id=context_id,
                        successor_context_id=f"{image_id}:boundary-{boundary + 1:03d}",
                        reference=reference,
                    )
                    requests.append(request)
                    merged_rows.append(
                        _merged_row(request, raw=raw, baseline_sum=observed_baseline)
                    )

            control_owner = f"gt:{image_id}:tp"
            twin = _row(7000 + image_index)
            following = _row(image_index * 31 + 1)
            override = self.benign_override.get(image_id, {})
            benign_raw = float(override.get("observed", self.benign_delta))
            sealed_raw = float(override.get("sealed", self.benign_delta))
            benign_baseline = -8.0 - image_index
            reference = self._reference(
                raw=sealed_raw,
                baseline_sum=benign_baseline,
                argmax=following,
                material=sealed_raw <= -1.0,
            )
            if "sealed_argmax" in override:
                reference["baseline_argmax_token_ids_sha256"] = override["sealed_argmax"]
            benign.append(
                {
                    "schema_version": plan_builder.BENIGN_SCHEMA_VERSION,
                    "row_kind": "benign_reference_control",
                    "unit_id": scorer.UNIT_ID,
                    "cohort_role": plan_builder.COHORT_BENIGN,
                    "gt_owner_id": control_owner,
                    "image_id": image_id,
                    "normalized_description": "tp",
                    "due_context_id": f"{image_id}:boundary-000",
                    "replaced_native_row_index": 0,
                    "following_native_action": {
                        "context_id": f"{image_id}:boundary-001",
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
            request = _request(
                arm=scorer.ARM_BENIGN,
                cohort_role=plan_builder.COHORT_BENIGN,
                gt_owner_id=control_owner,
                image_id=image_id,
                context_id=f"{image_id}:boundary-000",
                boundary_index=1,
                appended=twin,
                scored=following,
                baseline_context_id=f"{image_id}:boundary-001",
                successor_context_id=f"{image_id}:boundary-002",
                reference=reference,
            )
            requests.append(request)
            merged_rows.append(
                _merged_row(
                    request,
                    raw=benign_raw,
                    baseline_sum=benign_baseline + float(override.get("sum_drift", 0.0)),
                )
            )

        manifest = self._write_plan(selection, benign, requests)
        self._write_merged(manifest, merged_rows, quarantined_ids)

    def _write_plan(
        self,
        selection: list[dict[str, Any]],
        benign: list[dict[str, Any]],
        requests: list[dict[str, Any]],
    ) -> dict[str, Any]:
        self.plan_dir.mkdir(parents=True, exist_ok=True)
        row_sets = {
            plan_builder.SELECTION_REGISTRY_NAME: selection,
            plan_builder.BENIGN_REGISTRY_NAME: benign,
            plan_builder.REQUEST_PLAN_NAME: requests,
        }
        files = {
            name: b"".join(
                crossing_scorer.canonical_json_bytes(row) + b"\n" for row in rows
            )
            for name, rows in row_sets.items()
        }
        for name, payload in files.items():
            (self.plan_dir / name).write_bytes(payload)
        manifest = {
            "schema_version": plan_builder.MANIFEST_SCHEMA_VERSION,
            "unit_id": scorer.UNIT_ID,
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
            "materiality": {"cutoff_nats": -1.0},
            "non_voting_sensitivities": {"materiality_cutoffs_nats": [-0.75, -1.25]},
            "lineage": {"crossing_plan_dir": "/nonexistent/crossing"},
            "output_file_digests": {
                name: {
                    "path": name,
                    "byte_size": len(files[name]),
                    "sha256": crossing_scorer.sha256_bytes(files[name]),
                    "row_count": len(row_sets[name]),
                }
                for name in sorted(files)
            },
        }
        manifest["manifest_content_sha256"] = _sha256_json(manifest)
        (self.plan_dir / plan_builder.MANIFEST_NAME).write_bytes(
            crossing_scorer.canonical_json_bytes(manifest) + b"\n"
        )
        return manifest

    def _write_merged(
        self,
        manifest: dict[str, Any],
        rows: list[dict[str, Any]],
        quarantined: list[str],
    ) -> None:
        self.merged_dir.mkdir(parents=True, exist_ok=True)
        files = {
            merger.MERGED_ROWS_NAME: b"".join(
                crossing_scorer.canonical_json_bytes(row) + b"\n" for row in rows
            ),
            merger.MERGED_PARITY_NAME: b"",
        }
        for name, payload in files.items():
            (self.merged_dir / name).write_bytes(payload)
        receipt = {
            "schema_version": merger.MERGE_SCHEMA_VERSION,
            "unit_id": scorer.UNIT_ID,
            "merger_source_sha256": "m" * 64,
            "runtime_identity_sha256": "r" * 64,
            "plan": {
                "plan_dir": str(self.plan_dir),
                "manifest_content_sha256": manifest["manifest_content_sha256"],
            },
            "closure": {"row_count": len(rows)},
            "gates": {
                "order": list(merger.MERGE_OWNED_GATES),
                "deferred_to_analysis": list(merger.ANALYSIS_OWNED_GATES),
                merger.GATE_INPUT_AND_TOKEN_IDENTITY: {
                    "passed": True,
                    "checked_owner_count": 21,
                    "identical_scored_e_tokens_across_arms": True,
                },
                merger.GATE_RUNTIME_REPLAY: {
                    "passed": True,
                    "tolerance": 1e-3,
                    "quarantined_owner_count": len(quarantined),
                    "quarantined_owner_ids": sorted(quarantined),
                    "max_quarantined_owners": 2,
                    "owner_rows": [],
                },
                merger.GATE_CACHE_PARITY: {"passed": True, "tolerance": 1e-3},
            },
            "output_file_digests": {
                name: {
                    "path": name,
                    "byte_size": len(payload),
                    "sha256": crossing_scorer.sha256_bytes(payload),
                }
                for name, payload in sorted(files.items())
            },
        }
        receipt["receipt_content_sha256"] = _sha256_json(receipt)
        (self.merged_dir / merger.MERGE_RECEIPT_NAME).write_bytes(
            crossing_scorer.canonical_json_bytes(receipt) + b"\n"
        )

    def analyze(self) -> dict[str, Any]:
        return sut.run_analysis(self.merged_dir, plan_dir=self.plan_dir)


def _nonmaterial_everywhere(_owner: str, _votes: bool) -> float:
    return 0.4


def _material_everywhere(_owner: str, _votes: bool) -> float:
    return -3.0


@pytest.fixture()
def c_specific(tmp_path: Path) -> _Fixture:
    return _Fixture(tmp_path / "c-specific", neutral_relative=_nonmaterial_everywhere)


def test_end_to_end_analysis_routes_c_content_specific(c_specific: _Fixture) -> None:
    result = c_specific.analyze()
    summary = result["summary"]
    assert summary["decision"]["route"] == sut.ROUTE_C_CONTENT_SPECIFIC
    assert summary["decision"]["nonmaterial_side"]["owner_count"] == 9
    assert summary["decision"]["nonmaterial_side"]["clearly_separated_count"] == 4
    assert summary["gates"]["order"] == list(sut.GATE_ORDER)
    for gate in sut.GATE_ORDER:
        assert summary["gates"][gate]["passed"] is True
    assert summary["gates"][sut.GATE_INPUT_AND_TOKEN_IDENTITY]["owner"] == "merge"
    assert summary["gates"][sut.GATE_POSITIVE_CONTROLS]["owner"] == "analysis"
    assert len(result["owner_rows"]) == 21


def test_end_to_end_analysis_routes_neutral_row_not_neutral(tmp_path: Path) -> None:
    fixture = _Fixture(tmp_path / "not-neutral", neutral_relative=_material_everywhere)
    summary = fixture.analyze()["summary"]
    assert summary["gates"][sut.GATE_SPECIFICITY]["passed"] is False
    assert summary["gates"][sut.GATE_SPECIFICITY]["material_count"] == 12
    assert summary["decision"]["route"] == sut.ROUTE_NEUTRAL_NOT_NEUTRAL


def test_the_same_run_benign_replay_owns_the_relative_estimand(tmp_path: Path) -> None:
    """A nonzero same-run benign delta shifts every relative delta by exactly it."""

    fixture = _Fixture(
        tmp_path / "benign-shift",
        neutral_relative=_nonmaterial_everywhere,
        benign_delta=-0.75,
    )
    result = fixture.analyze()
    for row in result["owner_rows"]:
        reference = row["same_run_benign_reference"]
        assert reference["same_run_benign_coordinate_delta"] == pytest.approx(-0.75)
        assert row["neutral"]["relative_coordinate_delta"] == pytest.approx(
            row["neutral"]["coordinate_delta"] - (-0.75)
        )
        assert row["clean"]["relative_coordinate_delta"] == pytest.approx(
            row["clean"]["coordinate_delta"] - (-0.75)
        )
    summary = result["summary"]
    assert summary["materiality"]["benign_reference_source"] == "same_run_replay_only"
    assert (
        summary["materiality"]["sealed_value_role"]
        == "gate_reference_only_never_the_estimand"
    )
    assert summary["decision"]["route"] == sut.ROUTE_C_CONTENT_SPECIFIC


@pytest.mark.parametrize(
    "override",
    [
        {"observed": 0.4},  # coordinate delta drifts past 0.05 nat
        {"sum_drift": 0.5},  # selected-token sum drifts past 1e-3
        {"sealed_argmax": "e" * 64},  # compared argmax not preserved
    ],
)
def test_a_benign_mismatch_blocks_only_its_own_image(
    tmp_path: Path, override: dict[str, Any]
) -> None:
    fixture = _Fixture(
        tmp_path / f"benign-{sorted(override)[0]}",
        neutral_relative=_nonmaterial_everywhere,
        benign_override={"4134": override},
    )
    result = fixture.analyze()
    gate = result["summary"]["gates"][sut.GATE_BENIGN_REPLAY]
    assert gate["passed"] is False
    assert gate["blocked_image_ids"] == ["4134"]
    blocked = [row for row in result["owner_rows"] if row["image_id"] == "4134"]
    assert blocked and all(row["benign_blocked"] for row in blocked)
    assert all(not row["interpretable"] for row in blocked)
    others = [row for row in result["owner_rows"] if row["image_id"] != "4134"]
    assert all(row["interpretable"] for row in others)
    # Image 4134 carries two voting owners, both clearly separated, so the route
    # arithmetic loses them from *both* sides while the absolute denominators
    # stay nine and four.
    decision = result["summary"]["decision"]
    assert decision["nonmaterial_side"]["owner_count"] == 7
    assert decision["nonmaterial_side"]["clearly_separated_count"] == 2
    assert decision["voting_denominator"] == 9
    assert decision["clearly_separated_denominator"] == 4
    assert decision["route"] == sut.ROUTE_C_CONTENT_SPECIFIC


def test_enough_blocked_images_push_the_route_to_inconclusive(tmp_path: Path) -> None:
    """Blocking 4134 and 16228 leaves one clearly separated owner, below the two."""

    fixture = _Fixture(
        tmp_path / "benign-two-blocked",
        neutral_relative=_nonmaterial_everywhere,
        benign_override={"4134": {"observed": 0.4}, "13348": {"observed": 0.4}},
    )
    result = fixture.analyze()
    gate = result["summary"]["gates"][sut.GATE_BENIGN_REPLAY]
    assert gate["blocked_image_ids"] == ["13348", "4134"]
    decision = result["summary"]["decision"]
    assert decision["nonmaterial_side"]["clearly_separated_count"] == 1
    assert decision["route"] == sut.ROUTE_INCONCLUSIVE


def test_a_quarantined_owner_is_neither_material_nor_nonmaterial(tmp_path: Path) -> None:
    fixture = _Fixture(
        tmp_path / "quarantined",
        neutral_relative=_nonmaterial_everywhere,
        quarantined=("gt:4134:3",),
    )
    result = fixture.analyze()
    row = next(
        item for item in result["owner_rows"] if item["gt_owner_id"] == "gt:4134:3"
    )
    assert row["quarantined"] is True
    assert row["interpretable"] is False
    assert row["neutral"]["counts_toward_route"] is False
    arithmetic = result["summary"]["quarantine_arithmetic"][plan_builder.COHORT_VOTING]
    assert arithmetic["neither_owner_ids"] == ["gt:4134:3"]
    assert (
        arithmetic["material_count"]
        + arithmetic["nonmaterial_count"]
        + arithmetic["neither_count"]
        == 9
    )


# ---------------------------------------------------------------------------
# 7. Non-voting sensitivities and strata
# ---------------------------------------------------------------------------


def test_the_named_sensitivities_are_all_recomputed_and_non_voting(
    c_specific: _Fixture,
) -> None:
    summary = c_specific.analyze()["summary"]
    names = [entry["name"] for entry in summary["non_voting_sensitivities"]]
    assert names == [
        sut.SENSITIVITY_EXCLUDE_LENGTH_DELTA_TWO,
        sut.SENSITIVITY_EXCLUDE_POSTHOC_UNCERTAIN,
        "materiality_cutoff_-0.75",
        "materiality_cutoff_-1.25",
    ]
    for entry in summary["non_voting_sensitivities"]:
        assert entry["role"].startswith("non_voting_descriptive_recomputation")
        assert "route" in entry
    posthoc = next(
        entry
        for entry in summary["non_voting_sensitivities"]
        if entry["name"] == sut.SENSITIVITY_EXCLUDE_POSTHOC_UNCERTAIN
    )
    assert posthoc["excluded_owner_ids"] == ["gt:4134:5"]


def test_the_length_delta_two_sensitivity_drops_exactly_those_rows(
    tmp_path: Path,
) -> None:
    dropped = {"gt:13348:3", "gt:16228:3", "gt:4134:3"}
    fixture = _Fixture(
        tmp_path / "length",
        neutral_relative=_nonmaterial_everywhere,
        length_delta=lambda owner, _votes: 2 if owner in dropped else 0,
    )
    summary = fixture.analyze()["summary"]
    assert summary["decision"]["route"] == sut.ROUTE_C_CONTENT_SPECIFIC
    entry = next(
        item
        for item in summary["non_voting_sensitivities"]
        if item["name"] == sut.SENSITIVITY_EXCLUDE_LENGTH_DELTA_TWO
    )
    assert entry["excluded_owner_ids"] == sorted(dropped)
    # Three of the four clearly separated owners drop out, leaving one.
    assert entry["nonmaterial_side"]["clearly_separated_count"] == 1
    assert entry["route"] == sut.ROUTE_INCONCLUSIVE
    assert entry["route_changed"] is True
    assert entry["cutoff_fragile"] is False


def test_a_cutoff_that_changes_the_route_is_labelled_cutoff_fragile(
    tmp_path: Path,
) -> None:
    """Neutral deltas at -0.9 nat: nonmaterial at -1.0, material at -0.75."""

    fixture = _Fixture(
        tmp_path / "fragile", neutral_relative=lambda _owner, _votes: -0.9
    )
    summary = fixture.analyze()["summary"]
    assert summary["decision"]["route"] == sut.ROUTE_C_CONTENT_SPECIFIC
    loose = next(
        entry
        for entry in summary["non_voting_sensitivities"]
        if entry["name"] == "materiality_cutoff_-0.75"
    )
    assert loose["route_changed"] is True
    assert loose["cutoff_fragile"] is True
    strict = next(
        entry
        for entry in summary["non_voting_sensitivities"]
        if entry["name"] == "materiality_cutoff_-1.25"
    )
    assert strict["route"] == sut.ROUTE_C_CONTENT_SPECIFIC
    assert strict["cutoff_fragile"] is False


def test_strata_are_descriptive_and_keep_unmatched_e_unknown_neutral(
    c_specific: _Fixture,
) -> None:
    summary = c_specific.analyze()["summary"]
    strata = summary["strata"]
    assert set(sut.STRATUM_AXES).issubset(strata)
    assert strata["role"] == "descriptive_only_never_a_route"
    assert "unknown-neutral" in strata["unknown_neutral_note"]
    assert set(strata["e_strict_match"]) == {"matched", "unmatched"}
    assert set(strata["description_relation"]) == {
        "same_description",
        "different_description",
    }
    totals = sum(bucket["owner_count"] for bucket in strata["e_strict_match"].values())
    assert totals == 21


def test_likelihood_is_never_reported_as_coverage(c_specific: _Fixture) -> None:
    result = c_specific.analyze()
    separation = result["summary"]["evidence_separation"]
    assert separation["owner_coverage_measured_here"] is False
    assert separation["owner_emission_measured_here"] is False
    assert separation["strict_matching_measured_here"] is False
    for row in result["owner_rows"]:
        assert row["likelihood_versus_coverage"] == sut.LIKELIHOOD_VERSUS_COVERAGE
        assert row["strata"]["unknown_neutral_note"] == sut.UNKNOWN_NEUTRAL_NOTE
    assert any("coverage" in item for item in result["summary"]["not_claimed"])


# ---------------------------------------------------------------------------
# 8. Published artifacts
# ---------------------------------------------------------------------------


def test_published_artifacts_seal_themselves_and_their_inputs(
    c_specific: _Fixture,
) -> None:
    result = c_specific.analyze()
    files = sut.build_output_files(result)
    assert set(files) == {
        sut.OWNER_ROWS_NAME,
        sut.SUMMARY_NAME,
        sut.REPORT_MD_NAME,
        sut.RECEIPT_NAME,
    }
    receipt = json.loads(files[sut.RECEIPT_NAME])
    declared = receipt.pop("receipt_content_sha256")
    assert _sha256_json(receipt) == declared
    for name, entry in receipt["output_file_digests"].items():
        assert entry["sha256"] == crossing_scorer.sha256_bytes(files[name])
    assert receipt["materiality_cutoff_nats"] == -1.0
    assert receipt["gate_order"] == list(sut.GATE_ORDER)
    assert receipt["decision_route"] == sut.ROUTE_C_CONTENT_SPECIFIC
    assert receipt["policy"]["adaptive_threshold"] is False
    assert receipt["policy"]["second_neutral_row_choice"] is False
    assert receipt["policy"]["automatic_gpu_successor"] is False
    report = files[sut.REPORT_MD_NAME].decode("utf-8")
    assert sut.ROUTE_C_CONTENT_SPECIFIC in report
    assert "Quarantined owners: 0" in report
    assert "material + " in report


def test_the_analysis_refuses_a_merge_receipt_of_another_plan(
    c_specific: _Fixture, tmp_path: Path
) -> None:
    receipt = json.loads(
        (c_specific.merged_dir / merger.MERGE_RECEIPT_NAME).read_text()
    )
    receipt.pop("receipt_content_sha256")
    receipt["plan"]["manifest_content_sha256"] = "0" * 64
    receipt["receipt_content_sha256"] = _sha256_json(receipt)
    (c_specific.merged_dir / merger.MERGE_RECEIPT_NAME).write_bytes(
        crossing_scorer.canonical_json_bytes(receipt) + b"\n"
    )
    with pytest.raises(sut.NeutralRowAnalysisContractError, match="not the one the merge"):
        c_specific.analyze()


def test_the_analysis_refuses_merged_evidence_that_is_not_the_frozen_54(
    c_specific: _Fixture,
) -> None:
    rows = [
        json.loads(line)
        for line in (
            c_specific.merged_dir / merger.MERGED_ROWS_NAME
        ).read_text().splitlines()
    ]
    payload = b"".join(
        crossing_scorer.canonical_json_bytes(row) + b"\n" for row in rows[:-1]
    )
    (c_specific.merged_dir / merger.MERGED_ROWS_NAME).write_bytes(payload)
    receipt = json.loads(
        (c_specific.merged_dir / merger.MERGE_RECEIPT_NAME).read_text()
    )
    receipt.pop("receipt_content_sha256")
    receipt["output_file_digests"][merger.MERGED_ROWS_NAME] = {
        "path": merger.MERGED_ROWS_NAME,
        "byte_size": len(payload),
        "sha256": crossing_scorer.sha256_bytes(payload),
    }
    receipt["receipt_content_sha256"] = _sha256_json(receipt)
    (c_specific.merged_dir / merger.MERGE_RECEIPT_NAME).write_bytes(
        crossing_scorer.canonical_json_bytes(receipt) + b"\n"
    )
    with pytest.raises(sut.NeutralRowAnalysisContractError, match="not the frozen 54"):
        c_specific.analyze()


def test_the_analysis_refuses_a_merge_that_did_not_pass_its_own_gates(
    c_specific: _Fixture,
) -> None:
    receipt = json.loads(
        (c_specific.merged_dir / merger.MERGE_RECEIPT_NAME).read_text()
    )
    receipt.pop("receipt_content_sha256")
    receipt["gates"][merger.GATE_RUNTIME_REPLAY]["passed"] = False
    receipt["receipt_content_sha256"] = _sha256_json(receipt)
    (c_specific.merged_dir / merger.MERGE_RECEIPT_NAME).write_bytes(
        crossing_scorer.canonical_json_bytes(receipt) + b"\n"
    )
    with pytest.raises(sut.NeutralRowAnalysisContractError, match="does not declare gate"):
        c_specific.analyze()


def test_the_analysis_is_deterministic_across_repeated_runs(c_specific: _Fixture) -> None:
    first = sut.build_output_files(c_specific.analyze())
    second = sut.build_output_files(c_specific.analyze())
    assert first == second


def test_no_raw_delta_is_pooled_across_images(c_specific: _Fixture) -> None:
    summary = c_specific.analyze()["summary"]
    geometry.assert_no_cross_image_raw_delta_pooling(summary, label="probe")
    geometry.assert_emitted_payload(summary, label="probe")
    with pytest.raises(geometry.GeometryContractError):
        geometry.assert_no_cross_image_raw_delta_pooling(
            {"mean_coordinate_delta": 0.1}, label="probe"
        )


def test_owner_rows_disclose_the_unavoidable_neutral_row_confounds(
    c_specific: _Fixture,
) -> None:
    rows = c_specific.analyze()["owner_rows"]
    for row in rows:
        assert row["neutral"]["confounds"] == [
            "duplicate_of_an_already_emitted_row",
            "sorted_route_regression",
        ]
        assert row["neutral"]["materiality_cutoff_nats"] == -1.0
        assert row["clean"]["sealed_role"] == "gate_reference_only_never_the_estimand"
        assert row["claim_boundary"] == sut.CLAIM_BOUNDARY


def test_owner_rows_are_copy_safe_and_carry_one_benign_reference_per_image(
    c_specific: _Fixture,
) -> None:
    rows = c_specific.analyze()["owner_rows"]
    by_image: dict[str, set[str]] = {}
    for row in rows:
        by_image.setdefault(row["image_id"], set()).add(
            row["same_run_benign_reference"]["gt_owner_id"]
        )
    assert all(len(owners) == 1 for owners in by_image.values())
    snapshot = copy.deepcopy(rows)
    rows[0]["neutral"]["material"] = not rows[0]["neutral"]["material"]
    assert snapshot[0]["neutral"]["material"] != rows[0]["neutral"]["material"]
