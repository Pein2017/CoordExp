from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

import scripts.research.analyze_individual_trajectory_union_support as analyzer
from scripts.research.analyze_individual_trajectory_union_support import (
    analyze_rollout_files,
    analyze_rollout_payloads,
    match_prefix,
    parse_args,
)


def _prediction(trajectory: str, index: int, category: str, box: list[float]) -> dict[str, object]:
    return {
        "description": category,
        "generated_order": index,
        "bbox": box,
        "object_span_id": f"{trajectory}:{index}",
    }


def _rollout(
    image_id: str,
    mode: str,
    seed: int,
    predictions: list[tuple[str, list[float]]],
    *,
    stop_reason: str = "im_end",
    parse_status: str = "accepted",
) -> dict[str, object]:
    trajectory = "greedy" if mode == "greedy" else f"seed-{seed}"
    return {
        "image_id": image_id,
        "example_id": f"example-{image_id}",
        "seed": seed,
        "decode_mode": mode,
        "stop_reason": stop_reason,
        "predictions": {
            "parse_status": parse_status,
            "predictions": [_prediction(trajectory, i, category, box) for i, (category, box) in enumerate(predictions)],
        },
    }


def _artifact(mode: str, rollouts: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema_version": "current_seeded_sampled_rollouts.v1",
        "config": {
            "decode_mode": mode,
            "temperature": 0.0 if mode == "greedy" else 0.4,
            "top_p": 0.95,
            "repetition_penalty": 1.0,
            "max_new_tokens": 512,
            "resolved_fingerprint": "same-checkpoint-config",
        },
        "model_identity": {"checkpoint": "same-checkpoint"},
        "rollouts": rollouts,
    }


def _analyze(
    rollouts: list[dict[str, object]],
    owners: list[dict[str, object]],
    *,
    review_decisions: dict[str, object] | None = None,
) -> dict[str, object]:
    greedy = [row for row in rollouts if row["decode_mode"] == "greedy"]
    sampled = [row for row in rollouts if row["decode_mode"] == "sampled"]
    result = analyze_rollout_payloads(
        [_artifact("greedy", greedy), _artifact("sampled", sampled)],
        {"scene": owners},
        review_decisions=review_decisions,
    )
    return result["images"][0]


def _review(*decisions: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": "individual_trajectory_union_support_review_decisions.v1",
        "decisions": list(decisions),
    }


def test_cli_can_select_only_the_fixed_row_budget_needed_by_a_large_panel() -> None:
    args = parse_args(
        [
            "--rollout-artifact",
            "rollouts",
            "--annotations",
            "annotations.jsonl",
            "--output",
            "analysis.json",
            "--budget",
            "16",
            "--allow-incomplete-panel",
        ]
    )
    assert args.budgets == [16]


def test_incomplete_mode_reports_observed_panel_without_legacy_expectation() -> None:
    result = analyze_rollout_payloads(
        [
            _artifact("greedy", [_rollout("scene", "greedy", 31000, [])]),
            _artifact("sampled", [_rollout("scene", "sampled", 31001, [])]),
        ],
        {"scene": []},
        require_full_panel=False,
        budgets=(16,),
    )

    assert result["panel_expected"] is None
    assert result["observed_panel"] == {
        "image_count": 1,
        "greedy_seeds": [31000],
        "sampled_seeds": [31001],
        "greedy_trajectories_per_image": [1],
        "sampled_trajectories_per_image": [1],
    }


def test_file_analysis_sources_bind_current_analyzer_and_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    annotations = tmp_path / "candidate-pool.jsonl"
    annotations.write_text("candidate pool\n", encoding="utf-8")
    rollout = tmp_path / "rollout.json"
    rollout.write_text("rollout\n", encoding="utf-8")
    observed_panel = {
        "image_count": 1,
        "greedy_seeds": [31000],
        "sampled_seeds": [31001],
        "greedy_trajectories_per_image": [1],
        "sampled_trajectories_per_image": [1],
    }
    monkeypatch.setattr(
        analyzer,
        "load_rollout_artifacts",
        lambda _paths, *, require_decode_mode: [
            {
                "_source_path": str(rollout.resolve()),
                "_source_sha256": "rollout-sha",
                "rollouts": [],
            }
        ],
    )
    monkeypatch.setattr(
        analyzer, "load_generation7_annotations", lambda _path, *, image_ids: {}
    )
    monkeypatch.setattr(
        analyzer,
        "analyze_rollout_payloads",
        lambda *_args, **_kwargs: {
            "fixed_budgets": [16],
            "iou_threshold": 0.5,
            "require_full_panel": False,
            "observed_panel": observed_panel,
        },
    )

    result = analyze_rollout_files(
        [rollout],
        annotations,
        require_full_panel=False,
        budgets=(16,),
    )

    analyzer_path = Path(analyzer.__file__).resolve()
    assert result["sources"] == {
        "rollout_artifacts": [str(rollout.resolve())],
        "rollout_artifact_sha256": ["rollout-sha"],
        "annotations_path": str(annotations.resolve()),
        "annotations_sha256": hashlib.sha256(annotations.read_bytes()).hexdigest(),
        "analyzer_path": str(analyzer_path),
        "analyzer_sha256": hashlib.sha256(analyzer_path.read_bytes()).hexdigest(),
        "analysis_policy": {
            "fixed_budgets": [16],
            "iou_threshold": 0.5,
            "require_full_panel": False,
            "review_decisions_used": False,
            "observed_panel": observed_panel,
        },
    }


def test_duplicate_rows_do_not_increase_unique_owner_coverage() -> None:
    image = _analyze(
        [
            _rollout("scene", "greedy", 21000, [("cat", [0, 0, 10, 10]), ("cat", [0, 0, 10, 10])]),
            _rollout("scene", "sampled", 21001, [("cat", [0, 0, 10, 10])]),
        ],
        [{"owner_id": "cat-a", "category": "cat", "bbox": [0, 0, 10, 10]}],
    )
    budget = image["budgets"][0]
    assert budget["C_g"] == 1
    assert budget["trajectory_assignments"]["greedy"]["matched_owner_ids"] == ["cat-a"]
    assert budget["trajectory_assignments"]["greedy"]["unmatched_review_queue"][0]["entity_status"] == "duplicate"


def test_union_can_exceed_best_single_trajectory() -> None:
    image = _analyze(
        [
            _rollout("scene", "greedy", 21000, [("cat", [0, 0, 10, 10])]),
            _rollout("scene", "sampled", 21001, [("cat", [0, 0, 10, 10]), ("dog", [10, 0, 20, 10])]),
            _rollout("scene", "sampled", 21002, [("cat", [0, 0, 10, 10]), ("bird", [20, 0, 30, 10])]),
        ],
        [
            {"owner_id": "cat-a", "category": "cat", "bbox": [0, 0, 10, 10]},
            {"owner_id": "dog-b", "category": "dog", "bbox": [10, 0, 20, 10]},
            {"owner_id": "bird-c", "category": "bird", "bbox": [20, 0, 30, 10]},
        ],
    )
    budget = image["budgets"][0]
    assert budget["C_best"] == 2
    assert budget["C_union"] == 3
    assert budget["union_only_gain"] == 1
    assert budget["best_trajectory_ids"] == ["seed-21001", "seed-21002"]


def test_matching_is_recomputed_when_a_later_row_is_added() -> None:
    owners = [{"owner_id": "cat-a", "category": "cat", "bbox": [0, 0, 10, 10]}]
    rows = [
        {"prediction_id": "p0", "generated_row_index": 0, "category": "cat", "bbox": [0, 0, 9, 10]},
        {"prediction_id": "p1", "generated_row_index": 1, "category": "cat", "bbox": [0, 0, 10, 10]},
    ]
    at_one = match_prefix(rows, owners, 1)
    at_two = match_prefix(rows, owners, 2)
    assert at_one["matches"][0]["prediction_id"] == "p0"
    assert at_two["matches"][0]["prediction_id"] == "p1"
    assert at_two["row_assignment_receipts"][0]["entity_status"] == "duplicate"


def test_semantic_mismatch_stays_unresolved_for_crop_review() -> None:
    image = _analyze(
        [
            _rollout("scene", "greedy", 21000, [("dog", [0, 0, 10, 10])]),
            _rollout("scene", "sampled", 21001, [("dog", [0, 0, 10, 10])]),
        ],
        [{"owner_id": "cat-a", "category": "cat", "bbox": [0, 0, 10, 10]}],
    )
    receipt = image["budgets"][0]["unmatched_review_queue"][0]
    assert receipt["entity_status"] == "semantic_mismatch_unresolved"
    assert receipt["entity_status"] not in {"hallucination", "unsupported_hallucination"}


def test_right_censoring_removes_branch_evidence_but_keeps_lower_bounds() -> None:
    image = _analyze(
        [
            _rollout("scene", "greedy", 21000, [("cat", [0, 0, 10, 10])], stop_reason="im_end"),
            _rollout("scene", "sampled", 21001, [("cat", [0, 0, 10, 10])], stop_reason="length"),
        ],
        [{"owner_id": "cat-a", "category": "cat", "bbox": [0, 0, 10, 10]}],
    )
    first = image["budgets"][0]
    assert first["branch_evidence_status"] == "missing_right_censor"
    assert first["C_g"] is None and first["C_best"] is None and first["C_union"] is None
    assert first["lower_bounds"] == {"C_g": 1, "C_best": 1, "C_union": 1}


def test_natural_stop_before_budget_remains_branch_bearing() -> None:
    image = _analyze(
        [
            _rollout("scene", "greedy", 21000, [("cat", [0, 0, 10, 10])], stop_reason="im_end"),
            _rollout("scene", "sampled", 21001, [("cat", [0, 0, 10, 10])], stop_reason="im_end"),
        ],
        [{"owner_id": "cat-a", "category": "cat", "bbox": [0, 0, 10, 10]}],
    )
    budget = image["budgets"][0]
    assert budget["branch_evidence_status"] == "complete"
    assert budget["C_g"] == budget["C_best"] == budget["C_union"] == 1


def test_accepted_with_drops_keeps_valid_rows_and_applies_cutoff_to_malformed_count() -> None:
    greedy_row = _rollout("scene", "greedy", 21000, [("cat", [0, 0, 10, 10])])
    greedy_row["predictions"] = {
        "parse_status": "accepted_with_drops",
        "valid_prediction_count": 1,
        "dropped_prediction_count": 1,
        "dropped_predictions": [{"generated_order": 3, "reason": "geometry_invalid"}],
        "predictions": greedy_row["predictions"]["predictions"],
    }
    sampled_row = _rollout("scene", "sampled", 21001, [("cat", [0, 0, 10, 10])])
    result = analyze_rollout_payloads(
        [_artifact("greedy", [greedy_row]), _artifact("sampled", [sampled_row])],
        {"scene": [{"owner_id": "cat-a", "category": "cat", "bbox": [0, 0, 10, 10]}]},
        budgets=(1, 4),
    )
    budgets = result["images"][0]["budgets"]
    assert budgets[0]["trajectory_assignments"]["greedy"]["matched_owner_ids"] == ["cat-a"]
    assert budgets[0]["trajectory_assignments"]["greedy"]["malformed_row_count"] == 0
    assert budgets[1]["trajectory_assignments"]["greedy"]["malformed_row_count"] == 1
    assert budgets[1]["panel_harmful_row_count"] == 1


def test_review_verified_owner_adds_unresolved_owner_to_coverage() -> None:
    image = _analyze(
        [
            _rollout("scene", "greedy", 21000, [("cat", [0, 0, 10, 10]), ("dog", [20, 0, 30, 10])]),
            _rollout("scene", "sampled", 21001, [("cat", [0, 0, 10, 10])]),
        ],
        [
            {"owner_id": "cat-a", "category": "cat", "bbox": [0, 0, 10, 10]},
            {"owner_id": "dog-b", "category": "dog", "bbox": [10, 0, 20, 10]},
        ],
        review_decisions=_review(
            {
                "prediction_id": "greedy:1",
                "image_id": "scene",
                "trajectory_id": "greedy",
                "entity_status": "verified_owner",
                "owner_id": "dog-b",
                "geometry_status": "imperfect",
                "evidence": "Crop confirms the dog; its box is shifted right.",
            }
        ),
    )
    budget = image["budgets"][0]
    assignment = budget["trajectory_assignments"]["greedy"]
    assert budget["C_g"] == 2
    assert assignment["matched_owner_ids"] == ["cat-a", "dog-b"]
    assert assignment["unmatched_review_queue"] == []
    assert image["budgets"][0]["row_assignment_receipts"][1]["entity_status"] == "verified_owner"


def test_review_verified_owner_rejects_counting_an_already_assigned_owner_twice() -> None:
    with pytest.raises(ValueError, match="would count owner cat-a twice"):
        _analyze(
            [
                _rollout("scene", "greedy", 21000, [("cat", [0, 0, 10, 10]), ("cat", [20, 0, 30, 10])]),
                _rollout("scene", "sampled", 21001, [("cat", [0, 0, 10, 10])]),
            ],
            [{"owner_id": "cat-a", "category": "cat", "bbox": [0, 0, 10, 10]}],
            review_decisions=_review(
                {
                    "prediction_id": "greedy:1",
                    "image_id": "scene",
                    "trajectory_id": "greedy",
                    "entity_status": "verified_owner",
                    "owner_id": "cat-a",
                    "geometry_status": "acceptable",
                    "evidence": "The second row is the same cat.",
                }
            ),
        )


def test_review_unsupported_hallucination_counts_as_harmful_not_unresolved() -> None:
    image = _analyze(
        [
            _rollout("scene", "greedy", 21000, [("boat", [0, 0, 10, 10])]),
            _rollout("scene", "sampled", 21001, [("boat", [0, 0, 10, 10])]),
        ],
        [{"owner_id": "cat-a", "category": "cat", "bbox": [0, 0, 10, 10]}],
        review_decisions=_review(
            {
                "prediction_id": "greedy:0",
                "image_id": "scene",
                "trajectory_id": "greedy",
                "entity_status": "unsupported_hallucination",
                "evidence": "Crop contains only background.",
            }
        ),
    )
    assignment = image["budgets"][0]["trajectory_assignments"]["greedy"]
    assert assignment["harmful_row_count"] == 1
    assert assignment["row_counts"]["unsupported_hallucination"] == 1
    assert assignment["row_counts"]["unresolved"] == 0


def test_review_decision_unknown_or_unused_prediction_id_fails() -> None:
    with pytest.raises(ValueError, match="unknown review decision prediction_id"):
        _analyze(
            [
                _rollout("scene", "greedy", 21000, [("cat", [0, 0, 10, 10])]),
                _rollout("scene", "sampled", 21001, [("cat", [0, 0, 10, 10])]),
            ],
            [{"owner_id": "cat-a", "category": "cat", "bbox": [0, 0, 10, 10]}],
            review_decisions=_review(
                {
                    "prediction_id": "greedy:missing",
                    "image_id": "scene",
                    "trajectory_id": "greedy",
                    "entity_status": "uncertain",
                    "evidence": "No such prediction exists.",
                }
            ),
        )
    with pytest.raises(ValueError, match="not applied"):
        _analyze(
            [
                _rollout("scene", "greedy", 21000, [("cat", [0, 0, 10, 10])]),
                _rollout("scene", "sampled", 21001, [("cat", [0, 0, 10, 10])]),
            ],
            [{"owner_id": "cat-a", "category": "cat", "bbox": [0, 0, 10, 10]}],
            review_decisions=_review(
                {
                    "prediction_id": "greedy:0",
                    "image_id": "scene",
                    "trajectory_id": "greedy",
                    "entity_status": "uncertain",
                    "evidence": "This row is automatically matched and needs no review.",
                }
            ),
        )


def test_ambiguous_exact_category_match_enters_review_queue_until_adjudicated() -> None:
    image = _analyze(
        [
            _rollout("scene", "greedy", 21000, [("person", [0, 0, 10, 10])]),
            _rollout("scene", "sampled", 21001, [("person", [0, 0, 10, 10])]),
        ],
        [
            {"owner_id": "person-a", "category": "person", "bbox": [0, 0, 10, 10]},
            {"owner_id": "person-b", "category": "person", "bbox": [0, 0, 10, 10]},
        ],
    )
    assignment = image["budgets"][0]["trajectory_assignments"]["greedy"]
    assert assignment["matched_owner_ids"] == []
    assert assignment["unmatched_review_queue"][0]["entity_status"] == "ambiguous_matched_review"
    assert image["budgets"][0]["row_counts"]["ambiguous_matched_review"] == 2


def test_review_can_adjudicate_ambiguous_exact_category_match() -> None:
    image = _analyze(
        [
            _rollout("scene", "greedy", 21000, [("person", [0, 0, 10, 10])]),
            _rollout("scene", "sampled", 21001, [("person", [0, 0, 10, 10])]),
        ],
        [
            {"owner_id": "person-a", "category": "person", "bbox": [0, 0, 10, 10]},
            {"owner_id": "person-b", "category": "person", "bbox": [0, 0, 10, 10]},
        ],
        review_decisions=_review(
            {
                "prediction_id": "greedy:0",
                "image_id": "scene",
                "trajectory_id": "greedy",
                "entity_status": "verified_owner",
                "owner_id": "person-b",
                "geometry_status": "acceptable",
                "evidence": "Crop review assigns the box to the rear person.",
            }
        ),
    )
    assignment = image["budgets"][0]["trajectory_assignments"]["greedy"]
    assert assignment["matched_owner_ids"] == ["person-b"]
    assert assignment["unmatched_review_queue"] == []


def test_review_uncertain_remains_an_unresolved_potential() -> None:
    image = _analyze(
        [
            _rollout("scene", "greedy", 21000, [("boat", [0, 0, 10, 10])]),
            _rollout("scene", "sampled", 21001, [("boat", [0, 0, 10, 10])]),
        ],
        [],
        review_decisions=_review(
            {
                "prediction_id": "greedy:0",
                "image_id": "scene",
                "trajectory_id": "greedy",
                "entity_status": "uncertain",
                "evidence": "The crop is too small to decide whether this is a boat.",
            }
        ),
    )
    assignment = image["budgets"][0]["trajectory_assignments"]["greedy"]
    assert assignment["unmatched_review_queue"][0]["entity_status"] == "uncertain"
    assert assignment["unmatched_review_queue"][0]["review_required"] is False
    assert assignment["row_counts"]["unresolved"] == 1
    assert assignment["coverage_lower_bound"] == 0
    assert assignment["coverage_upper_bound"] == 0
    assert assignment["harmful_row_lower_bound"] == 0
    assert assignment["harmful_row_upper_bound"] == 1


def test_route_local_conservative_certificate_requires_bounded_advantage() -> None:
    image = _analyze(
        [
            _rollout("scene", "greedy", 21000, [("cat", [0, 0, 10, 10])]),
            _rollout("scene", "sampled", 21001, [("cat", [0, 0, 10, 10]), ("dog", [10, 0, 20, 10])]),
        ],
        [
            {"owner_id": "cat-a", "category": "cat", "bbox": [0, 0, 10, 10]},
            {"owner_id": "dog-b", "category": "dog", "bbox": [10, 0, 20, 10]},
        ],
    )
    certificate = image["budgets"][0]["route_local_conservative_certificates"]["seed-21001"]
    assert certificate["conservative_certificate"] is True
    assert certificate["coverage_condition"] is True
    assert certificate["harmful_condition"] is True
    assert certificate["conservative_gain_lower_bound"] == 1


def test_route_local_certificate_is_not_claimed_when_greedy_has_unresolved_potential() -> None:
    image = _analyze(
        [
            _rollout("scene", "greedy", 21000, [("cat", [0, 0, 10, 10]), ("boat", [30, 0, 40, 10])]),
            _rollout("scene", "sampled", 21001, [("cat", [0, 0, 10, 10]), ("dog", [10, 0, 20, 10])]),
        ],
        [
            {"owner_id": "cat-a", "category": "cat", "bbox": [0, 0, 10, 10]},
            {"owner_id": "dog-b", "category": "dog", "bbox": [10, 0, 20, 10]},
        ],
    )
    certificate = image["budgets"][0]["route_local_conservative_certificates"]["seed-21001"]
    assert certificate["conservative_certificate"] is False
    assert certificate["coverage_condition"] is False
    assert certificate["conservative_gain_lower_bound"] == 0
