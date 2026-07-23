from __future__ import annotations

from copy import deepcopy
import math

import pytest
import torch

from scripts.research import run_complete_candidate_row_scoring as candidate_scoring
from scripts.research import run_physical_owner_duplication_prefix_counterfactual as probe


def _row(description_token: int, coordinates: tuple[int, int, int, int]) -> list[int]:
    return [
        candidate_scoring.OBJECT_REF_START,
        description_token,
        candidate_scoring.OBJECT_REF_END,
        candidate_scoring.BOX_START,
        *coordinates,
        candidate_scoring.BOX_END,
    ]


def _hashed_row(owner_id: str, tokens: list[int]) -> dict[str, object]:
    return {
        "owner_id": owner_id,
        "row_token_ids": tokens,
        "row_token_ids_sha256": probe.sha256_json(tokens),
    }


def _case_document() -> dict[str, object]:
    coordinate_start = candidate_scoring.COORDINATE_TOKEN_START
    rows = {
        "A": _row(101, (coordinate_start + 10, coordinate_start + 20, coordinate_start + 100, coordinate_start + 200)),
        "B": _row(102, (coordinate_start + 300, coordinate_start + 40, coordinate_start + 410, coordinate_start + 210)),
        "C": _row(103, (coordinate_start + 500, coordinate_start + 300, coordinate_start + 620, coordinate_start + 440)),
        "D": _row(104, (coordinate_start + 700, coordinate_start + 500, coordinate_start + 820, coordinate_start + 680)),
    }
    ledger = [
        {
            "owner_id": owner_id,
            "description": "bottle",
            "bbox_norm1000": [float(index * 100), 0.0, float(index * 100 + 80), 100.0],
        }
        for index, owner_id in enumerate(("A", "B", "C", "D"))
    ]
    return {
        "schema_version": probe.CASE_SCHEMA_VERSION,
        "cases": [
            {
                "case_id": "case-b-duplicate",
                "image_id": "455649",
                "entity_ledger": ledger,
                "prefix_rows": [_hashed_row(owner_id, rows[owner_id]) for owner_id in ("A", "B", "C")],
                "impending_duplicate_owner_id": "B",
                "candidates": [
                    {
                        **_hashed_row("B", rows["B"]),
                        "candidate_id": "impending-b-duplicate",
                        "role": "impending_duplicate",
                    },
                    {
                        **_hashed_row("D", rows["D"]),
                        "candidate_id": "reviewed-new-d",
                        "role": "reviewed_new_owner",
                    },
                ],
                "interventions": {
                    "shuffle_owner_order": ["C", "A", "B"],
                    "replacement_owner_id": "A",
                    "coordinate_donor_owner_id": "C",
                },
            }
        ],
    }


def _validated_case() -> dict[str, object]:
    return probe.validate_case_document(_case_document())["cases"][0]


def test_all_five_prefix_arms_preserve_the_intended_single_factor() -> None:
    case = _validated_case()
    arms = probe.build_prefix_arms(case)
    exact = arms[probe.EXACT_ARM]

    assert set(arms) == set(probe.REQUIRED_ARMS)
    assert exact["prefix_owner_ids"] == ["A", "B", "C"]
    assert exact["prefix_token_ids_sha256"] == probe.sha256_json(exact["prefix_token_ids"])

    shuffled = arms[probe.SHUFFLE_ARM]
    assert shuffled["prefix_owner_ids"] == ["C", "A", "B"]
    assert sorted(row["row_token_ids_sha256"] for row in shuffled["prefix_rows"]) == sorted(
        row["row_token_ids_sha256"] for row in exact["prefix_rows"]
    )
    assert shuffled["first_divergence_from_exact"]["exact_row_index"] == 0

    removed = arms[probe.REMOVAL_ARM]
    assert removed["prefix_owner_ids"] == ["A", "C"]
    assert removed["baseline_covered_owner_ids"] == ["A", "B", "C"]
    assert len(removed["prefix_token_ids"]) < len(exact["prefix_token_ids"])

    replaced = arms[probe.REPLACEMENT_ARM]
    assert replaced["prefix_owner_ids"] == ["A", "A", "C"]
    assert replaced["prefix_rows"][1]["row_token_ids"] == exact["prefix_rows"][0]["row_token_ids"]
    assert len(replaced["prefix_token_ids"]) == len(exact["prefix_token_ids"])

    corrupted = arms[probe.CORRUPTION_ARM]
    target_original = exact["prefix_rows"][1]["row_token_ids"]
    target_corrupted = corrupted["prefix_rows"][1]["row_token_ids"]
    phases = candidate_scoring._canonical_row_phases(target_original)
    coordinate_indices = set(phases["x1"] + phases["y1"] + phases["x2"] + phases["y2"])
    assert all(
        target_corrupted[index] == token
        for index, token in enumerate(target_original)
        if index not in coordinate_indices
    )
    assert probe._row_coordinates(target_corrupted, "test") == probe._row_coordinates(
        exact["prefix_rows"][2]["row_token_ids"], "test"
    )
    assert corrupted["prefix_rows"][1]["coordinate_corruption_source"] == "owner:C"


def test_first_divergence_reports_tokens_and_length_change() -> None:
    case = _validated_case()
    arms = probe.build_prefix_arms(case)

    exact_divergence = arms[probe.EXACT_ARM]["first_divergence_from_exact"]
    assert exact_divergence["has_divergence"] is False
    assert exact_divergence["first_differing_token_index"] is None

    shuffle_divergence = arms[probe.SHUFFLE_ARM]["first_divergence_from_exact"]
    assert shuffle_divergence["has_divergence"] is True
    assert shuffle_divergence["first_differing_token_index"] == 1
    assert shuffle_divergence["exact_row_index"] == 0
    assert shuffle_divergence["modified_row_index"] == 0

    removal_divergence = arms[probe.REMOVAL_ARM]["first_divergence_from_exact"]
    assert removal_divergence["has_divergence"] is True
    assert removal_divergence["exact_row_index"] == 1
    assert removal_divergence["modified_row_index"] == 1
    assert removal_divergence["modified_prefix_token_count"] < removal_divergence["exact_prefix_token_count"]


def test_coordinate_corruption_rejects_an_unchanged_address() -> None:
    document = _case_document()
    case = document["cases"][0]
    coordinate_start = candidate_scoring.COORDINATE_TOKEN_START
    coordinates = [
        coordinate_start + 300,
        coordinate_start + 40,
        coordinate_start + 410,
        coordinate_start + 210,
    ]
    case["interventions"] = {
        "coordinate_token_ids": coordinates,
        "coordinate_token_ids_sha256": probe.sha256_json(coordinates),
    }
    checked = probe.validate_case_document(document)
    with pytest.raises(probe.PrefixCounterfactualValidationError, match="must differ"):
        probe.build_prefix_arms(checked["cases"][0])


def test_hashed_rows_are_required_before_any_model_execution() -> None:
    document = _case_document()
    document["cases"][0]["prefix_rows"][0].pop("row_token_ids_sha256")
    with pytest.raises(probe.PrefixCounterfactualValidationError, match="requires a row token hash"):
        probe.validate_case_document(document)


def test_short_continuation_summary_keeps_new_owner_and_duplicate_destinations_separate() -> None:
    run = {
        "status": "success",
        "horizon_rows_requested": 2,
        "horizon_rows_generated": 2,
        "horizon_complete": True,
        "rows": [
            {
                "accepted_complete_row": True,
                "strict_matched_owner_ids": ["B"],
                "covered_prefix_owner_ids": ["B"],
                "uncovered_ledger_owner_ids": [],
                "unmatched_or_ambiguous_prediction_indices": [],
                "row_stop": {"stop_reason": "complete_row"},
            },
            {
                "accepted_complete_row": True,
                "strict_matched_owner_ids": ["D"],
                "covered_prefix_owner_ids": [],
                "uncovered_ledger_owner_ids": ["D"],
                "unmatched_or_ambiguous_prediction_indices": [],
                "row_stop": {"stop_reason": "complete_row"},
            },
        ],
    }
    summary = probe.summarize_short_continuation(run, baseline_covered_owner_ids=["A", "B", "C"])
    assert summary["first_owner_ids"] == ["B"]
    assert summary["first_row_is_duplicate"] is True
    assert summary["first_row_is_new_owner"] is False
    assert summary["downstream_new_owner_ids"] == ["D"]
    assert summary["downstream_covered_owner_ids"] == ["B"]
    assert summary["duplicate_row_count"] == 1


def test_synthetic_receipt_validates_original_modified_tokens_and_score_slots() -> None:
    case = _validated_case()
    artifact = probe._case_artifact_skeleton(case)
    for arm in artifact["arms"].values():
        arm["raw_candidate_field_row_scores"] = [_valid_candidate_score()]
        arm["terminal_boundary_score"] = _valid_terminal_score()
        arm["short_continuations"] = [_valid_continuation()]
    payload = {
        "schema_version": probe.RECEIPT_SCHEMA_VERSION,
        "cases": [artifact],
    }
    probe.validate_receipt_payload(payload)

    bad = deepcopy(payload)
    bad["cases"][0]["arms"][probe.CORRUPTION_ARM]["prefix_token_ids_sha256"] = "0" * 64
    with pytest.raises(probe.PrefixCounterfactualValidationError, match="prefix hash mismatch"):
        probe.validate_receipt_payload(bad)


def _valid_candidate_score() -> dict[str, object]:
    field = {"sum": -1.0, "mean": -1.0, "count": 1}
    return {
        "candidate_id": "impending-b-duplicate",
        "owner_id": "B",
        "candidate_role": "impending_duplicate",
        "candidate_description": "bottle",
        "row_token_ids": _row(
            102,
            (
                candidate_scoring.COORDINATE_TOKEN_START + 300,
                candidate_scoring.COORDINATE_TOKEN_START + 40,
                candidate_scoring.COORDINATE_TOKEN_START + 410,
                candidate_scoring.COORDINATE_TOKEN_START + 210,
            ),
        ),
        "token_count": 9,
        "token_log_probabilities": [-1.0] * 9,
        "row_entry": field,
        "description": field,
        "geometry": {"sum": -5.0, "mean": -1.0, "count": 5},
        "x1": field,
        "y1": field,
        "x2": field,
        "y2": field,
        "closure": field,
        "full_row": {"sum": -9.0, "mean": -1.0, "count": 9},
    }


def _valid_terminal_score() -> dict[str, object]:
    return {
        "row_entry_token_id": candidate_scoring.OBJECT_REF_START,
        "terminal_token_id": 151645,
        "row_entry_log_probability": -1.0,
        "terminal_log_probability": -2.0,
        "row_entry_minus_terminal": 1.0,
    }


def _valid_continuation() -> dict[str, object]:
    return {
        "mode": "greedy",
        "seed": None,
        "status": "success",
        "short_continuation_outcome": {
            "continuation_status": "success",
            "first_owner_ids": ["B"],
            "downstream_new_owner_ids": [],
        },
    }


def test_receipt_rejects_descriptive_label_in_numeric_description_score() -> None:
    artifact = probe._case_artifact_skeleton(_validated_case())
    for arm in artifact["arms"].values():
        score = _valid_candidate_score()
        score["description"] = "bottle"
        arm["raw_candidate_field_row_scores"] = [score]
        arm["terminal_boundary_score"] = _valid_terminal_score()
        arm["short_continuations"] = [_valid_continuation()]
    with pytest.raises(probe.PrefixCounterfactualValidationError, match="description score"):
        probe.validate_receipt_payload({"schema_version": probe.RECEIPT_SCHEMA_VERSION, "cases": [artifact]})


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("raw_candidate_field_row_scores", [], "non-empty candidate score"),
        ("short_continuations", [], "non-empty continuation"),
    ],
)
def test_receipt_rejects_empty_score_or_continuation_slots(field: str, value: object, message: str) -> None:
    artifact = probe._case_artifact_skeleton(_validated_case())
    for arm in artifact["arms"].values():
        arm["raw_candidate_field_row_scores"] = [_valid_candidate_score()]
        arm["terminal_boundary_score"] = _valid_terminal_score()
        arm["short_continuations"] = [_valid_continuation()]
    artifact["arms"][probe.EXACT_ARM][field] = value
    with pytest.raises(probe.PrefixCounterfactualValidationError, match=message):
        probe.validate_receipt_payload({"schema_version": probe.RECEIPT_SCHEMA_VERSION, "cases": [artifact]})


def test_receipt_rejects_nonfinite_field_score() -> None:
    artifact = probe._case_artifact_skeleton(_validated_case())
    for arm in artifact["arms"].values():
        score = _valid_candidate_score()
        score["description"] = {"sum": math.nan, "mean": -1.0, "count": 1}
        arm["raw_candidate_field_row_scores"] = [score]
        arm["terminal_boundary_score"] = _valid_terminal_score()
        arm["short_continuations"] = [_valid_continuation()]
    with pytest.raises(probe.PrefixCounterfactualValidationError, match="finite"):
        probe.validate_receipt_payload({"schema_version": probe.RECEIPT_SCHEMA_VERSION, "cases": [artifact]})


def test_candidate_label_is_emitted_without_overwriting_numeric_description_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = _case_document()
    document["cases"][0]["candidates"][0]["description"] = "bottle"
    candidate = probe.validate_case_document(document)["cases"][0]["candidates"][0]
    assert candidate["candidate_description"] == "bottle"

    def fake_forward_logits(
        _model: object,
        _model_inputs: object,
        input_ids: list[int],
        _image_grid_thw: object,
    ) -> torch.Tensor:
        vocab = candidate_scoring.COORDINATE_TOKEN_END_EXCLUSIVE + 1
        logits = torch.zeros((len(input_ids), vocab), dtype=torch.float32)
        for index, token in enumerate(input_ids[1:]):
            logits[index, token] = 1.0
        return logits

    monkeypatch.setattr(candidate_scoring, "_forward_logits", fake_forward_logits)
    _, scores = probe._score_arm_candidates(
        model=object(),
        model_inputs={},
        image_grid_thw=None,
        full_prefix_token_ids=[11, 12],
        candidates=[candidate],
        terminal_token_id=13,
    )
    assert scores[0]["candidate_description"] == "bottle"
    assert isinstance(scores[0]["description"], dict)
    assert scores[0]["description"]["count"] == 1
