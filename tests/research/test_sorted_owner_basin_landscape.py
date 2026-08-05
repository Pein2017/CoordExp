"""Synthetic CPU tests for sorted-owner basin-landscape candidate accounting."""

from __future__ import annotations

import json
import hashlib
from dataclasses import replace

import pytest

from scripts.research.sorted_owner_basin_landscape import (
    POLICY_SCORE_LABEL,
    RULES_SCHEMA_VERSION,
    GEOMETRY_IDENTITY_SCHEMA,
    BasinRegistration,
    CandidateScore,
    canonical_geometry_identity,
    CompleteBoxCandidate,
    CoordinateBin,
    CoordinateBox,
    ConditionalY1ScoreReceipt,
    ExtentBankSeed,
    FullVocabularyAttestation,
    PolicyRuntimeIdentity,
    RawCoordinateLogprobs,
    TargetAnchorPair,
    assert_basin_mass_comparable,
    attest_complete_conditional_y1_scores,
    cluster_physical_basins,
    complete_box_candidate_id,
    compute_basin_measurements,
    compute_peak_prominence,
    count_distinct_physical_basins_in_top_n,
    enumerate_complete_conditional_y1,
    enumerate_target_anchor_pairs,
    enumerate_target_x1_anchors,
    expand_extent_banks,
    full_vocabulary_id_digest,
    json_serializable_receipt,
    proposal_domain_receipts,
    repetition_penalty_policy_score,
    score_complete_candidates,
    stable_json_dumps,
    transformers_repetition_penalty_transform,
    validate_complete_conditional_y1_scores,
    validate_rule_mapping,
)


def _rules(*, target_measure_group: str = "matched") -> dict:
    return {
        "schema_version": RULES_SCHEMA_VERSION,
        "contract_mode": "test_fixture",
        "geometry_identity": {
            "schema": GEOMETRY_IDENTITY_SCHEMA,
            "coordinate_denominator": 1000,
        },
        "coordinate_bins": {"min": 0, "max": 99},
        "target_anchor": {
            "margin_fraction": 0.25,
            "min_margin_bins": 1,
            "max_margin_bins": 4,
        },
        "bank_order": ["target", "covered", "background", "scan", "part", "whole", "merged"],
        "proposal_measures": {
            "target_measure": {
                "comparability_group": target_measure_group,
                "normalization": "full_domain_normalized_weighted_sum",
                "bank_weights": {"target": 1.0, "part": 1.0, "whole": 1.0, "merged": 1.0},
            },
            "foil_measure": {
                "comparability_group": "matched",
                "normalization": "full_domain_normalized_weighted_sum",
                "bank_weights": {"covered": 1.0, "background": 1.0, "scan": 1.0},
            },
        },
        "bank_proposal_measure": {
            "target": "target_measure",
            "covered": "foil_measure",
            "background": "foil_measure",
            "scan": "foil_measure",
            "part": "target_measure",
            "whole": "target_measure",
            "merged": "target_measure",
        },
        "spatial_clustering": {
            "owner_link_iou_min": 0.1,
            "owner_link_center_distance_max": 8.0,
            "extent_submode_iou_min": 0.75,
        },
        "shape": {
            "near_peak_logprob_delta": 0.4,
            "wide_ridge_min_candidates": 2,
            "multi_submode_min": 2,
            "merged_extent_submodes": ["merged"],
            "scan_bank_names": ["scan"],
        },
        "declared_extent_submodes": ["gt", "part", "whole", "merged"],
        "registered_basin_roles": {
            "target_owner": {
                "kind": "target",
                "foil_set_id": "owner-vs-frozen-foils",
                "identity_kind": "reviewed_physical_owner",
                "allowed_bank_names": ["target", "part", "whole", "merged"],
            },
            "same_description_owner_foil": {
                "kind": "foil",
                "foil_set_id": "owner-vs-frozen-foils",
                "identity_kind": "reviewed_physical_owner",
                "allowed_bank_names": ["target", "part", "whole", "merged"],
            },
            "covered_foil": {
                "kind": "foil",
                "foil_set_id": "owner-vs-frozen-foils",
                "identity_kind": "reviewed_physical_owner",
                "allowed_bank_names": ["covered"],
            },
            "background_foil": {
                "kind": "foil",
                "foil_set_id": "owner-vs-frozen-foils",
                "identity_kind": "registered_geometry",
                "allowed_bank_names": ["background"],
            },
            "scan_foil": {
                "kind": "foil",
                "foil_set_id": "owner-vs-frozen-foils",
                "identity_kind": "registered_geometry",
                "allowed_bank_names": ["scan"],
            },
        },
        "prominence": {"functional": "peak_height_difference"},
    }


def _candidate(
    candidate_id: str,
    box: CoordinateBox,
    *,
    bank: str = "target",
    measure: str = "target_measure",
    submode: str = "whole",
    physical_owner_hint: str | None = None,
) -> CompleteBoxCandidate:
    return CompleteBoxCandidate(
        candidate_id=candidate_id,
        bank_name=bank,
        source_id=candidate_id,
        box=box,
        extent_submode=submode,
        proposal_measure_id=measure,
        anchor=None,
        physical_owner_hint=physical_owner_hint,
    )


def _score(candidate: CompleteBoxCandidate, total: float) -> CandidateScore:
    # The decomposition is preserved in real receipts; uniform terms make the
    # synthetic peak ordering especially readable.
    return CandidateScore(candidate, RawCoordinateLogprobs(*(total / 4.0 for _ in range(4))))


def _registrations(
    clusters,
    rules,
    completeness_attestation,
    candidate_scores,
    *,
    roles_by_owner: dict[str, str],
    neutral_roles_by_bank: dict[str, str] | None = None,
) -> tuple[BasinRegistration, ...]:
    registrations = []
    for cluster in clusters:
        banks = {
            candidate.candidate.bank_name
            for candidate in candidate_scores
            if candidate.candidate.candidate_id in cluster.candidate_ids
        }
        if cluster.reviewed_physical_owner_id is not None:
            role_id = roles_by_owner[cluster.reviewed_physical_owner_id]
        else:
            assert neutral_roles_by_bank is not None and len(banks) == 1
            role_id = neutral_roles_by_bank[next(iter(banks))]
        role = rules.basin_role(role_id)
        registrations.append(
            BasinRegistration(
                basin_id=cluster.basin_id,
                role_id=role_id,
                identity_kind=role.identity_kind,
                reviewed_physical_owner_id=(
                    cluster.reviewed_physical_owner_id
                    if role.identity_kind == "reviewed_physical_owner"
                    else None
                ),
                registered_geometry_id=(
                    f"registered-geometry:{cluster.basin_id}"
                    if role.identity_kind == "registered_geometry"
                    else None
                ),
                context_id=completeness_attestation.context_id,
                foil_set_id=rules.basin_role(role_id).foil_set_id,
                rule_digest=rules.rule_digest,
                conditional_y1_completeness_digest=completeness_attestation.completeness_digest,
            )
        )
    return tuple(registrations)


def _complete_y1_attestation(
    rules,
    *,
    diagnostic_owner_id: str,
    gt_owner_id: str | None = None,
    image_identity: str = "image:sha256:fixture",
    gt_box: CoordinateBox | None = None,
    canonical_description_text: str = "person",
    canonical_description_token_digest: str | None = None,
    context_id: str = "context:exact-prefix",
    context_token_digest: str = hashlib.sha256(b"context-token-fixture").hexdigest(),
    tokenizer_identity: str = "tokenizer:fixture",
    model_identity: str = "model:fixture",
    runtime_identity: str = "runtime:fixture",
):
    gt = gt_box or CoordinateBox.from_values(10, 20, 14, 24)
    scores_by_x1 = {}
    for x1 in enumerate_target_x1_anchors(gt, rules):
        scores_by_x1[x1] = tuple(
            ConditionalY1ScoreReceipt(
                x1=entry.x1,
                y1=entry.y1,
                raw_selected_token_logprob=-1.0,
                can_form_valid_box=entry.can_form_valid_box,
                invalid_box_reason=entry.invalid_box_reason,
            )
            for entry in enumerate_complete_conditional_y1(x1, gt, rules)
        )
    return attest_complete_conditional_y1_scores(
        diagnostic_owner_id=diagnostic_owner_id,
        gt_owner_id=gt_owner_id or diagnostic_owner_id,
        image_identity=image_identity,
        context_id=context_id,
        canonical_description_text=canonical_description_text,
        canonical_description_token_digest=(
            canonical_description_token_digest
            or hashlib.sha256(canonical_description_text.encode()).hexdigest()
        ),
        context_token_digest=context_token_digest,
        tokenizer_identity=tokenizer_identity,
        model_identity=model_identity,
        runtime_identity=runtime_identity,
        gt_box=gt,
        scores_by_x1=scores_by_x1,
        rules=rules,
    )


def test_coordinate_types_and_full_interior_margin_anchor_plan() -> None:
    rules = validate_rule_mapping(_rules())
    gt = CoordinateBox.from_values(10, 20, 14, 24)
    assert [item.value for item in enumerate_target_x1_anchors(gt, rules)] == [9, 10, 11, 12, 13, 14]
    anchors = enumerate_target_anchor_pairs(gt, rules)
    assert (anchors[0].x1.value, anchors[0].y1.value) == (9, 19)
    assert (anchors[-1].x1.value, anchors[-1].y1.value) == (14, 24)
    complete_y = enumerate_complete_conditional_y1(CoordinateBin(10), gt, rules)
    assert len(complete_y) == 100
    assert complete_y[0].is_target_anchor is False
    assert [entry.y1.value for entry in complete_y if entry.is_target_anchor] == [19, 20, 21, 22, 23, 24]
    assert complete_y[-1].y1.value == 99
    assert complete_y[-1].can_form_valid_box is False
    assert complete_y[-1].invalid_box_reason == "no_later_representable_y2"
    y_scores = tuple(
        ConditionalY1ScoreReceipt(
            x1=entry.x1,
            y1=entry.y1,
            raw_selected_token_logprob=-1.0,
            can_form_valid_box=entry.can_form_valid_box,
            invalid_box_reason=entry.invalid_box_reason,
        )
        for entry in complete_y
    )
    validated_y_scores = validate_complete_conditional_y1_scores(
        x1=CoordinateBin(10), gt_box=gt, scores=y_scores, rules=rules
    )
    assert validated_y_scores[-1].invalid_box_reason == "no_later_representable_y2"
    with pytest.raises(ValueError, match="cover every declared y1 bin"):
        validate_complete_conditional_y1_scores(
            x1=CoordinateBin(10), gt_box=gt, scores=y_scores[:-1], rules=rules
        )
    with pytest.raises(ValueError, match="repeat y1"):
        validate_complete_conditional_y1_scores(
            x1=CoordinateBin(10), gt_box=gt, scores=(*y_scores, y_scores[0]), rules=rules
        )
    declared_rows = {
        x1: tuple(
            ConditionalY1ScoreReceipt(
                x1=entry.x1,
                y1=entry.y1,
                raw_selected_token_logprob=-1.0,
                can_form_valid_box=entry.can_form_valid_box,
                invalid_box_reason=entry.invalid_box_reason,
            )
            for entry in enumerate_complete_conditional_y1(x1, gt, rules)
        )
        for x1 in enumerate_target_x1_anchors(gt, rules)
    }
    declared_rows.pop(next(iter(declared_rows)))
    with pytest.raises(ValueError, match="declared x1 anchors"):
        attest_complete_conditional_y1_scores(
            diagnostic_owner_id="gt:fixture",
            gt_owner_id="gt:fixture",
            image_identity="image:fixture",
            context_id="context:exact-prefix",
            canonical_description_text="person",
            canonical_description_token_digest=hashlib.sha256(b"person").hexdigest(),
            context_token_digest=hashlib.sha256(b"context").hexdigest(),
            tokenizer_identity="tokenizer:fixture",
            model_identity="model:fixture",
            runtime_identity="runtime:fixture",
            gt_box=gt,
            scores_by_x1=declared_rows,
            rules=rules,
        )
    with pytest.raises(ValueError, match="x1 < x2"):
        CoordinateBox.from_values(1, 2, 1, 3)


def test_production_coordinate_contract_is_full_0_999_and_fixture_is_nonproduction() -> None:
    fixture_rules = validate_rule_mapping(_rules())
    fixture_attestation = _complete_y1_attestation(
        fixture_rules, diagnostic_owner_id="gt:fixture"
    )
    assert fixture_rules.contract_mode == "test_fixture"
    assert fixture_rules.can_emit_production_attestation is False
    assert fixture_attestation.attestation_kind == "test_fixture"

    invalid_production = _rules()
    invalid_production["contract_mode"] = "production"
    with pytest.raises(ValueError, match="complete coordinate vocabulary 0..999"):
        validate_rule_mapping(invalid_production)

    production_mapping = _rules()
    production_mapping["contract_mode"] = "production"
    production_mapping["coordinate_bins"] = {"min": 0, "max": 999}
    production_rules = validate_rule_mapping(production_mapping)
    gt = CoordinateBox.from_values(10, 20, 14, 24)
    complete_y = enumerate_complete_conditional_y1(CoordinateBin(10), gt, production_rules)
    assert len(complete_y) == 1000
    assert (complete_y[0].y1.value, complete_y[-1].y1.value) == (0, 999)
    assert complete_y[-1].can_form_valid_box is False
    production_attestation = _complete_y1_attestation(
        production_rules, diagnostic_owner_id="gt:production"
    )
    assert production_rules.can_emit_production_attestation is True
    assert production_attestation.attestation_kind == "production"
    geometry = canonical_geometry_identity(
        CoordinateBox.from_values(0, 0, 999, 999),
        image_width=640,
        image_height=480,
        rules=production_rules,
    )
    assert geometry.schema == GEOMETRY_IDENTITY_SCHEMA
    assert geometry.pixel_box_xyxy == (0, 0, round(999 * 640 / 1000), round(999 * 480 / 1000))


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("diagnostic_owner_id", "gt:other-diagnostic"),
        ("gt_owner_id", "gt:other-panel-owner"),
        ("image_identity", "image:sha256:other"),
        ("gt_box", CoordinateBox.from_values(11, 20, 15, 24)),
        ("canonical_description_text", "pedestrian"),
        ("canonical_description_token_digest", hashlib.sha256(b"other-description").hexdigest()),
        ("context_id", "context:other-prefix"),
        ("context_token_digest", hashlib.sha256(b"other-context").hexdigest()),
        ("tokenizer_identity", "tokenizer:other"),
        ("model_identity", "model:other"),
        ("runtime_identity", "runtime:other"),
    ],
)
def test_conditional_y1_attestation_digest_binds_every_execution_identity(
    field: str, replacement
) -> None:
    rules = validate_rule_mapping(_rules())
    base = _complete_y1_attestation(rules, diagnostic_owner_id="gt:target")
    alternate = _complete_y1_attestation(
        rules,
        diagnostic_owner_id=(replacement if field == "diagnostic_owner_id" else "gt:target"),
        gt_owner_id=(replacement if field == "gt_owner_id" else "gt:target"),
        image_identity=(replacement if field == "image_identity" else "image:sha256:fixture"),
        gt_box=(replacement if field == "gt_box" else None),
        canonical_description_text=(
            replacement if field == "canonical_description_text" else "person"
        ),
        canonical_description_token_digest=(
            replacement
            if field == "canonical_description_token_digest"
            else hashlib.sha256(b"person").hexdigest()
        ),
        context_id=(replacement if field == "context_id" else "context:exact-prefix"),
        context_token_digest=(
            replacement
            if field == "context_token_digest"
            else hashlib.sha256(b"context-token-fixture").hexdigest()
        ),
        tokenizer_identity=(replacement if field == "tokenizer_identity" else "tokenizer:fixture"),
        model_identity=(replacement if field == "model_identity" else "model:fixture"),
        runtime_identity=(replacement if field == "runtime_identity" else "runtime:fixture"),
    )
    assert alternate.completeness_digest != base.completeness_digest


def test_deterministic_extent_expansion_has_exact_complete_box_ids_and_receipts() -> None:
    rules = validate_rule_mapping(_rules())
    anchors = (TargetAnchorPair(CoordinateBin(10), CoordinateBin(10)), TargetAnchorPair(CoordinateBin(94), CoordinateBin(94)))
    expansion = expand_extent_banks(
        anchors,
        (
            ExtentBankSeed("target", "gt-template", CoordinateBox.from_values(5, 5, 12, 14), "gt", "anchor_translate"),
            ExtentBankSeed("covered", "covered-1", CoordinateBox.from_values(40, 40, 48, 50), "whole", "exact"),
        ),
        rules,
    )
    assert len(expansion.candidates) == 2  # The 94/94 translation would exceed bin 99.
    target = next(item for item in expansion.candidates if item.bank_name == "target")
    assert target.box.as_tuple() == (10, 10, 17, 19)
    assert target.candidate_id == complete_box_candidate_id(
        bank_name="target", source_id="gt-template", box=target.box, anchor=TargetAnchorPair(CoordinateBin(10), CoordinateBin(10))
    )
    receipt = next(item for item in expansion.receipts if item.source_id == "gt-template")
    assert receipt.skipped_anchor_count == 1
    raw = {item.candidate_id: RawCoordinateLogprobs(-1.0, -2.0, -3.0, -4.0) for item in expansion.candidates}
    scores = score_complete_candidates(expansion.candidates, raw, rules)
    assert scores[0].raw_complete_box_logprob == -10.0
    with pytest.raises(ValueError, match="exactly match"):
        score_complete_candidates(expansion.candidates, {}, rules)


def test_three_same_description_owner_peaks_need_cluster_aware_counting() -> None:
    """Token top-3 can all be one person even though three physical basins exist."""

    rules = validate_rule_mapping(_rules())
    # Three high-probability extent variants occupy person A.  The next two
    # candidates are distinct people sharing the same canonical description.
    scores = (
        _score(_candidate("a-face", CoordinateBox.from_values(10, 10, 16, 16), submode="part", physical_owner_hint="gt:image:a"), -1.0),
        _score(_candidate("a-torso", CoordinateBox.from_values(8, 8, 20, 24), submode="whole", physical_owner_hint="gt:image:a"), -1.1),
        _score(_candidate("a-merged", CoordinateBox.from_values(7, 7, 22, 26), submode="merged", physical_owner_hint="gt:image:a"), -1.2),
        _score(_candidate("b-whole", CoordinateBox.from_values(40, 10, 54, 28), physical_owner_hint="gt:image:b"), -2.0),
        _score(_candidate("c-whole", CoordinateBox.from_values(70, 10, 84, 28), physical_owner_hint="gt:image:c"), -2.1),
    )
    clusters = cluster_physical_basins(scores, rules)
    assert len(clusters) == 3
    assert count_distinct_physical_basins_in_top_n(scores, clusters, 3) == 1
    completeness = _complete_y1_attestation(rules, diagnostic_owner_id="gt:image:a")
    measurements = compute_basin_measurements(
        scores,
        clusters,
        _registrations(
            clusters,
            rules,
            completeness,
            scores,
            roles_by_owner={
                "gt:image:a": "target_owner",
                "gt:image:b": "same_description_owner_foil",
                "gt:image:c": "same_description_owner_foil",
            },
        ),
        completeness,
        rules,
    )
    assert len(measurements) == 3
    assert measurements[0].extent_submode_count == 3
    assert measurements[0].shape == "merged_extent"


def test_non_null_owner_hints_hard_separate_crowded_overlapping_owners() -> None:
    rules = validate_rule_mapping(_rules())
    # The boxes overlap heavily and an unhinted candidate geometrically bridges
    # them, but two reviewed physical owners must never be joined by that chain.
    scores = (
        _score(
            _candidate(
                "a-left-owner",
                CoordinateBox.from_values(10, 10, 30, 30),
                physical_owner_hint="gt:image:10",
            ),
            -1.0,
        ),
        _score(_candidate("b-null-bridge", CoordinateBox.from_values(12, 10, 32, 30)), -1.1),
        _score(
            _candidate(
                "c-right-owner",
                CoordinateBox.from_values(14, 10, 34, 30),
                physical_owner_hint="gt:image:11",
            ),
            -1.2,
        ),
    )
    clusters = cluster_physical_basins(scores, rules)
    assert len(clusters) == 2
    membership = {cluster.basin_id: set(cluster.candidate_ids) for cluster in clusters}
    known_owner_candidates = {"a-left-owner", "c-right-owner"}
    assert known_owner_candidates == set().union(*membership.values()) & known_owner_candidates
    assert all(len(known_owner_candidates.intersection(ids)) <= 1 for ids in membership.values())
    assert any(cluster.owner_identity_status == "unresolved" for cluster in clusters)
    with pytest.raises(ValueError, match="unresolved physical basins"):
        count_distinct_physical_basins_in_top_n(scores, clusters, 2)


def test_declared_extent_submode_labels_are_hard_boundaries_before_iou_clustering() -> None:
    rules = validate_rule_mapping(_rules())
    # Identical geometry would be one geometric submode without the frozen
    # labels.  It remains two declared extent modes for one reviewed owner.
    scores = (
        _score(
            _candidate(
                "face-mode",
                CoordinateBox.from_values(10, 10, 20, 20),
                submode="part",
                physical_owner_hint="gt:one",
            ),
            -1.0,
        ),
        _score(
            _candidate(
                "whole-mode",
                CoordinateBox.from_values(10, 10, 20, 20),
                submode="whole",
                physical_owner_hint="gt:one",
            ),
            -1.0,
        ),
    )
    clusters = cluster_physical_basins(scores, rules)
    assert len(clusters) == 1
    assert clusters[0].owner_identity_status == "reviewed"
    assert len(clusters[0].extent_submodes) == 2


def test_owner_neutral_background_and_reviewed_covered_foil_identity_invariants() -> None:
    rules = validate_rule_mapping(_rules())
    target_score = _score(
        _candidate("target", CoordinateBox.from_values(10, 10, 20, 20), physical_owner_hint="gt:t"),
        -1.0,
    )
    covered_score = _score(
        _candidate(
            "covered",
            CoordinateBox.from_values(40, 10, 50, 20),
            bank="covered",
            measure="foil_measure",
            physical_owner_hint="gt:c",
        ),
        -1.5,
    )
    background_score = _score(
        _candidate(
            "background",
            CoordinateBox.from_values(70, 10, 80, 20),
            bank="background",
            measure="foil_measure",
        ),
        -2.0,
    )
    scores = (target_score, covered_score, background_score)
    clusters = cluster_physical_basins(scores, rules)
    completeness = _complete_y1_attestation(rules, diagnostic_owner_id="gt:t")
    registrations = _registrations(
        clusters,
        rules,
        completeness,
        scores,
        roles_by_owner={"gt:t": "target_owner", "gt:c": "covered_foil"},
        neutral_roles_by_bank={"background": "background_foil"},
    )
    measurements = compute_basin_measurements(
        scores, clusters, registrations, completeness, rules
    )
    target = next(item for item in measurements if item.role_id == "target_owner")
    covered = next(item for item in measurements if item.role_id == "covered_foil")
    background = next(item for item in measurements if item.role_id == "background_foil")
    assert background.identity_kind == "registered_geometry"
    assert background.reviewed_physical_owner_id is None
    assert background.registered_geometry_id is not None
    assert compute_peak_prominence(target, background, completeness, rules).peak_prominence == pytest.approx(1.0)
    assert compute_peak_prominence(target, covered, completeness, rules).peak_prominence == pytest.approx(0.5)

    background_registration = next(
        item for item in registrations if item.role_id == "background_foil"
    )
    masquerading_target = replace(
        background_registration,
        role_id="target_owner",
        identity_kind="reviewed_physical_owner",
        reviewed_physical_owner_id="gt:t",
        registered_geometry_id=None,
    )
    with pytest.raises(ValueError, match="banks .* not admitted"):
        compute_basin_measurements(
            scores,
            clusters,
            tuple(
                masquerading_target if item is background_registration else item
                for item in registrations
            ),
            completeness,
            rules,
        )

    ownerless_covered = _score(
        _candidate(
            "ownerless-covered",
            CoordinateBox.from_values(40, 10, 50, 20),
            bank="covered",
            measure="foil_measure",
        ),
        -1.5,
    )
    unresolved_scores = (target_score, ownerless_covered)
    unresolved_clusters = cluster_physical_basins(unresolved_scores, rules)
    unresolved_registrations = []
    for cluster in unresolved_clusters:
        bank = next(
            item.candidate.bank_name
            for item in unresolved_scores
            if item.candidate.candidate_id in cluster.candidate_ids
        )
        role_id = "target_owner" if bank == "target" else "covered_foil"
        unresolved_registrations.append(
            BasinRegistration(
                basin_id=cluster.basin_id,
                role_id=role_id,
                identity_kind="reviewed_physical_owner",
                reviewed_physical_owner_id="gt:t" if bank == "target" else "gt:c",
                registered_geometry_id=None,
                context_id=completeness.context_id,
                foil_set_id=rules.basin_role(role_id).foil_set_id,
                rule_digest=rules.rule_digest,
                conditional_y1_completeness_digest=completeness.completeness_digest,
            )
        )
    with pytest.raises(ValueError, match="requires a reviewed physical-owner binding"):
        compute_basin_measurements(
            unresolved_scores,
            unresolved_clusters,
            tuple(unresolved_registrations),
            completeness,
            rules,
        )

    same_owner_covered = replace(
        covered_score,
        candidate=replace(covered_score.candidate, physical_owner_hint="gt:t"),
    )
    same_owner_scores = (target_score, same_owner_covered)
    same_owner_clusters = cluster_physical_basins(same_owner_scores, rules)
    same_owner_registrations = []
    for cluster in same_owner_clusters:
        bank = next(
            item.candidate.bank_name
            for item in same_owner_scores
            if item.candidate.candidate_id in cluster.candidate_ids
        )
        role_id = "target_owner" if bank == "target" else "covered_foil"
        same_owner_registrations.append(
            BasinRegistration(
                basin_id=cluster.basin_id,
                role_id=role_id,
                identity_kind="reviewed_physical_owner",
                reviewed_physical_owner_id="gt:t",
                registered_geometry_id=None,
                context_id=completeness.context_id,
                foil_set_id=rules.basin_role(role_id).foil_set_id,
                rule_digest=rules.rule_digest,
                conditional_y1_completeness_digest=completeness.completeness_digest,
            )
        )
    with pytest.raises(ValueError, match="distinct reviewed physical owners"):
        compute_basin_measurements(
            same_owner_scores,
            same_owner_clusters,
            tuple(same_owner_registrations),
            completeness,
            rules,
        )


def test_normalized_mass_uses_unique_boxes_in_the_full_compared_domain() -> None:
    rules = validate_rule_mapping(_rules())
    # A has two equal-score coordinate boxes (with a second source for A1); B
    # has one.  The source duplicate remains in provenance but cannot double
    # count proposal mass, so A still carries exactly twice B's mass.
    a1 = _score(_candidate("a1", CoordinateBox.from_values(10, 10, 20, 20), physical_owner_hint="gt:a"), -2.0)
    a1_duplicate = _score(_candidate("a1-duplicate", CoordinateBox.from_values(10, 10, 20, 20), physical_owner_hint="gt:a"), -2.0)
    a2 = _score(_candidate("a2", CoordinateBox.from_values(11, 10, 21, 20), physical_owner_hint="gt:a"), -2.0)
    b = _score(_candidate("b", CoordinateBox.from_values(50, 10, 60, 20), physical_owner_hint="gt:b"), -2.0)
    scores = (a1, a1_duplicate, a2, b)
    clusters = cluster_physical_basins(scores, rules)
    completeness = _complete_y1_attestation(rules, diagnostic_owner_id="gt:a")
    measurements = compute_basin_measurements(
        scores,
        clusters,
        _registrations(
            clusters,
            rules,
            completeness,
            scores,
            roles_by_owner={"gt:a": "target_owner", "gt:b": "same_description_owner_foil"},
        ),
        completeness,
        rules,
    )
    assert len(measurements) == 2
    domain = proposal_domain_receipts(scores, rules)
    assert domain[0].candidate_ids == ("a1", "a1-duplicate", "a2", "b")
    assert len(domain[0].unique_coordinate_boxes) == 3
    assert domain[0].unique_coordinate_boxes[0].multiplicity_candidate_ids == ("a1", "a1-duplicate")
    assert measurements[0].mass_unique_box_candidate_ids == ("a1", "a2")
    assert measurements[0].normalized_basin_mass == pytest.approx(2.0 * measurements[1].normalized_basin_mass)
    assert_basin_mass_comparable(measurements)


def test_peak_prominence_requires_registered_target_and_frozen_foil_same_context() -> None:
    rules = validate_rule_mapping(_rules())
    target_score = _score(
        _candidate("target", CoordinateBox.from_values(10, 10, 20, 20), physical_owner_hint="gt:target"),
        -1.0,
    )
    foil_score = _score(
        _candidate(
            "foil",
            CoordinateBox.from_values(50, 10, 60, 20),
            bank="background",
            measure="foil_measure",
        ),
        -2.0,
    )
    scores = (target_score, foil_score)
    clusters = cluster_physical_basins(scores, rules)
    completeness = _complete_y1_attestation(rules, diagnostic_owner_id="gt:target")
    registrations = _registrations(
        clusters,
        rules,
        completeness,
        scores,
        roles_by_owner={"gt:target": "target_owner"},
        neutral_roles_by_bank={"background": "background_foil"},
    )
    measurements = compute_basin_measurements(
        scores,
        clusters,
        registrations,
        completeness,
        rules,
    )
    target = next(item for item in measurements if item.role_kind == "target")
    foil = next(item for item in measurements if item.role_kind == "foil")
    with pytest.raises(ValueError, match="unequal proposal measures"):
        assert_basin_mass_comparable(measurements)
    prominence = compute_peak_prominence(target, foil, completeness, rules)
    assert prominence.context_id == "context:exact-prefix"
    assert prominence.foil_set_id == "owner-vs-frozen-foils"
    assert prominence.peak_prominence == pytest.approx(1.0)
    with pytest.raises(ValueError, match="one registered target basin"):
        compute_peak_prominence(
            target, replace(target, basin_id=foil.basin_id), completeness, rules
        )
    with pytest.raises(ValueError, match="one registered target basin"):
        compute_peak_prominence(
            replace(foil, basin_id=target.basin_id), foil, completeness, rules
        )
    with pytest.raises(ValueError, match="context"):
        compute_peak_prominence(target, replace(foil, context_id="context:other"), completeness, rules)
    with pytest.raises(ValueError, match="unregistered basin role"):
        compute_peak_prominence(
            target, replace(foil, role_id="not-a-frozen-foil"), completeness, rules
        )
    bad_role_registrations = tuple(
        replace(registration, role_id="not-a-frozen-foil")
        if registration.role_id == "background_foil"
        else registration
        for registration in registrations
    )
    with pytest.raises(ValueError, match="unregistered basin role"):
        compute_basin_measurements(scores, clusters, bad_role_registrations, completeness, rules)
    bad_digest_registrations = tuple(
        replace(registration, rule_digest="not-the-frozen-rule-digest")
        if registration.role_id == "target_owner"
        else registration
        for registration in registrations
    )
    with pytest.raises(ValueError, match="rule digest"):
        compute_basin_measurements(scores, clusters, bad_digest_registrations, completeness, rules)
    with pytest.raises(ValueError, match="require ConditionalY1CompletenessAttestation"):
        compute_basin_measurements(scores, clusters, registrations, None, rules)
    stale_completeness_registrations = tuple(
        replace(registration, conditional_y1_completeness_digest="stale-completeness-digest")
        for registration in registrations
    )
    with pytest.raises(ValueError, match="conditional y1 completeness digest"):
        compute_basin_measurements(
            scores, clusters, stale_completeness_registrations, completeness, rules
        )
    stale_rules_mapping = _rules()
    stale_rules_mapping["target_anchor"]["margin_fraction"] = 0.5
    stale_rules = validate_rule_mapping(stale_rules_mapping)
    with pytest.raises(ValueError, match="attestation rule digest is stale"):
        compute_basin_measurements(scores, clusters, registrations, completeness, stale_rules)
    other_context_completeness = _complete_y1_attestation(
        rules,
        diagnostic_owner_id="gt:target",
        context_id="context:other",
        context_token_digest=hashlib.sha256(b"context:other").hexdigest(),
    )
    with pytest.raises(ValueError, match="context does not match conditional y1"):
        compute_basin_measurements(
            scores, clusters, registrations, other_context_completeness, rules
        )
    with pytest.raises(ValueError, match="measurement context does not match conditional y1"):
        compute_peak_prominence(target, foil, other_context_completeness, rules)
    with pytest.raises(ValueError, match="measurement conditional y1 completeness digest"):
        compute_peak_prominence(
            replace(target, conditional_y1_completeness_digest="stale-completeness-digest"),
            foil,
            completeness,
            rules,
        )
    rebound_image_attestation = _complete_y1_attestation(
        rules,
        diagnostic_owner_id="gt:target",
        image_identity="image:sha256:rebound",
    )
    assert rebound_image_attestation.completeness_digest != completeness.completeness_digest
    with pytest.raises(ValueError, match="conditional y1 completeness digest"):
        compute_basin_measurements(
            scores, clusters, registrations, rebound_image_attestation, rules
        )
    rebound_owner_attestation = _complete_y1_attestation(
        rules, diagnostic_owner_id="gt:other-owner"
    )
    rebound_owner_registrations = tuple(
        replace(
            registration,
            conditional_y1_completeness_digest=(
                rebound_owner_attestation.completeness_digest
            ),
        )
        for registration in registrations
    )
    with pytest.raises(ValueError, match="does not match the GT owner"):
        compute_basin_measurements(
            scores,
            clusters,
            rebound_owner_registrations,
            rebound_owner_attestation,
            rules,
        )
    with pytest.raises(ValueError, match="does not match bank"):
        cluster_physical_basins(
            (
                _score(
                    _candidate(
                        "bad-measure",
                        CoordinateBox.from_values(70, 10, 80, 20),
                        measure="foil_measure",
                        physical_owner_hint="gt:bad",
                    ),
                    -1.0,
                ),
            ),
            rules,
        )


def test_target_registration_binds_gt_owner_not_diagnostic_namespace() -> None:
    rules = validate_rule_mapping(_rules())
    target_score = _score(
        _candidate(
            "target",
            CoordinateBox.from_values(10, 10, 20, 20),
            physical_owner_hint="gt:target",
        ),
        -1.0,
    )
    scores = (target_score,)
    clusters = cluster_physical_basins(scores, rules)
    completeness = _complete_y1_attestation(
        rules,
        diagnostic_owner_id="diagnostic:gt:target",
        gt_owner_id="gt:target",
    )
    registrations = _registrations(
        clusters,
        rules,
        completeness,
        scores,
        roles_by_owner={"gt:target": "target_owner"},
    )

    measurements = compute_basin_measurements(
        scores,
        clusters,
        registrations,
        completeness,
        rules,
    )

    assert len(measurements) == 1
    assert measurements[0].reviewed_physical_owner_id == "gt:target"


def test_toy_tokenizer_policy_transform_is_identity_attested_and_non_likelihood() -> None:
    transformed = transformers_repetition_penalty_transform({1: 2.0, 2: -3.0, 3: 0.0}, [1, 2], 2.0)
    assert transformed == {1: 1.0, 2: -6.0, 3: 0.0}
    # This deliberately tiny domain is a toy tokenizer, not a production
    # Qwen vocabulary.  Its explicit identity must agree with the runner receipt.
    rules = validate_rule_mapping(_rules())
    runtime_identity = PolicyRuntimeIdentity(
        tokenizer_identity="toy-tokenizer:two-coordinate-tokens",
        model_identity="toy-model:cpu-policy-test",
        runtime_rule_digest=rules.rule_digest,
    )
    full_domain = FullVocabularyAttestation(
        2,
        full_vocabulary_id_digest(2),
        runtime_identity.tokenizer_identity,
        runtime_identity.model_identity,
        runtime_identity.runtime_rule_digest,
    )
    policy = repetition_penalty_policy_score(
        raw_step_logits=({0: 2.0, 1: 1.0}, {0: 1.0, 1: 2.0}, {0: 3.0, 1: 0.0}, {0: 0.0, 1: 3.0}),
        coordinate_token_ids=(0, 1, 0, 1),
        prefix_token_ids=(0,),
        repetition_penalty=1.1,
        vocabulary_attestation=full_domain,
        runtime_identity=runtime_identity,
        rules=rules,
    )
    assert policy.label == POLICY_SCORE_LABEL
    assert policy.total < 0.0
    with pytest.raises(ValueError, match="attested full contiguous vocabulary"):
        repetition_penalty_policy_score(
            raw_step_logits=({0: 1.0, 1: 0.0},) * 4,
            coordinate_token_ids=(0, 1, 0, 1),
            prefix_token_ids=(0,),
            repetition_penalty=1.1,
            vocabulary_attestation=FullVocabularyAttestation(
                3,
                full_vocabulary_id_digest(3),
                runtime_identity.tokenizer_identity,
                runtime_identity.model_identity,
                runtime_identity.runtime_rule_digest,
            ),
            runtime_identity=runtime_identity,
            rules=rules,
        )
    with pytest.raises(ValueError, match="identity does not match"):
        repetition_penalty_policy_score(
            raw_step_logits=({0: 1.0, 1: 0.0},) * 4,
            coordinate_token_ids=(0, 1, 0, 1),
            prefix_token_ids=(0,),
            repetition_penalty=1.1,
            vocabulary_attestation=full_domain,
            runtime_identity=replace(runtime_identity, model_identity="toy-model:other"),
            rules=rules,
        )
    with pytest.raises(ValueError, match="runtime rule digest"):
        repetition_penalty_policy_score(
            raw_step_logits=({0: 1.0, 1: 0.0},) * 4,
            coordinate_token_ids=(0, 1, 0, 1),
            prefix_token_ids=(0,),
            repetition_penalty=1.1,
            vocabulary_attestation=full_domain,
            runtime_identity=replace(runtime_identity, runtime_rule_digest="stale-runtime-rules"),
            rules=rules,
        )


def test_receipts_are_json_serializable_and_stably_ordered() -> None:
    rules = validate_rule_mapping(_rules())
    receipt = json_serializable_receipt({"rules": rules, "box": CoordinateBox.from_values(1, 2, 3, 4)})
    encoded = stable_json_dumps(receipt)
    assert json.loads(encoded)["box"] == {"x1": 1, "x2": 3, "y1": 2, "y2": 4}
    assert encoded == stable_json_dumps(receipt)
