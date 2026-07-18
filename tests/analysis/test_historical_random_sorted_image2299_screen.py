"""Model-free tests for the historical image-2299 likelihood screen."""

from __future__ import annotations

import math

import pytest
import torch

from scripts.research.run_historical_random_sorted_image2299_screen import (
    EXPECTED_PROMPT_HASH,
    MATCH_AMBIGUITY_MARGIN,
    MATCH_IOU_THRESHOLD,
    OWNER_PERSON_RANKS,
    SAMPLING_MAX_NEW_TOKENS,
    SAMPLING_REPETITION_PENALTY,
    SAMPLING_SEEDS,
    SAMPLING_TEMPERATURE,
    SAMPLING_TOP_P,
    bbox_iou,
    build_historical_prompts,
    geometry_sorted_objects,
    legacy_sampling_suffix_is_complete,
    match_parsed_row_to_candidates,
    merge_arm_records,
    parse_legacy_generation,
    render_legacy_row,
    score_token_sequence,
    select_image2299_rows,
    verify_historical_prompt,
)


def _score(value: float) -> dict[str, object]:
    return {
        "total_log_probability": value,
        "mean_log_probability": value / 7,
        "structural_log_probability": value / 3,
        "structural_mean_log_probability": value / 6,
        "description_log_probability": value / 7,
        "coordinate_log_probability": value * 4 / 7,
        "coordinate_log_probabilities": {slot: value / 28 for slot in ("x1", "y1", "x2", "y2")},
    }


def _arm(adapter: str, owner: int, shift: float) -> dict[str, object]:
    # Match image 2299's frozen global layout: tie rows are interspersed with
    # person rows, so person-only rank 14 is global object rank 19.
    tie_global_ranks = {8, 9, 10, 11, 12, 43, 44, 45}
    person_global_ranks = [rank for rank in range(46) if rank not in tie_global_ranks]
    owner_global_rank = person_global_ranks[owner]
    candidates = []
    for rank in range(46):
        desc = "tie" if rank in tie_global_ranks else "person"
        person_rank = person_global_ranks.index(rank) if desc == "person" else None
        value = -float(rank) + shift
        candidates.append({
            "global_object_rank": rank,
            "person_only_rank": person_rank,
            "category_rank": person_rank if person_rank is not None else rank - 38,
            "desc": desc,
            "category_id": 1 if desc == "person" else 32,
            "coco_ann_id": -rank,
            "bbox": [1, 2, 3, 4],
            "emitted": rank in {0, 1, owner_global_rank},
            "uncovered": rank not in {0, 1, owner_global_rank},
            "emitted_person": desc == "person" and rank in {0, 1, owner_global_rank},
            "uncovered_person": desc == "person" and rank not in {0, 1, owner_global_rank},
            "score": _score(value),
        })
    return {
        "schema_version": "historical_random_sorted_image2299_screen.v1",
        "adapter_role": adapter,
        "owner_person_rank": owner,
        "owner_global_rank": owner_global_rank,
        "candidate_scores": candidates,
        "terminal_score": {
            "token_log_probability": -50.0 + shift,
            "object_ref_start_log_probability": -20.0 + shift,
            "continue_minus_terminal_margin": 30.0,
        },
    }


def test_legacy_row_render_has_no_separator_or_closing_marker() -> None:
    row = render_legacy_row("person", [625, 86, 711, 356])
    assert row == (
        "<|object_ref_start|>person<|box_start|>"
        "<|coord_625|><|coord_86|><|coord_711|><|coord_356|>"
    )
    assert "\n" not in row
    assert "<|object_ref_end|>" not in row
    assert "<|box_end|>" not in row


def test_sampling_parser_accepts_one_row_or_terminal_and_rejects_extra_rows() -> None:
    row = render_legacy_row("person", [625, 86, 711, 356])
    parsed = parse_legacy_generation(row + "<|im_end|>")
    assert parsed["status"] == "row"
    assert parsed["description"] == "person"
    assert parsed["bbox"] == [625, 86, 711, 356]
    assert parsed["terminal_after_row"] is True
    assert parse_legacy_generation("<|im_end|>")["status"] == "terminal"
    malformed = parse_legacy_generation(row + row)
    assert malformed["status"] == "malformed"


def test_sampling_match_requires_category_and_reports_physical_owner() -> None:
    candidates = [
        {"global_object_rank": 7, "person_only_rank": 5, "desc": "person", "coco_ann_id": -7, "bbox": [10, 10, 30, 40]},
        {"global_object_rank": 8, "person_only_rank": 6, "desc": "person", "coco_ann_id": -8, "bbox": [70, 70, 90, 100]},
        {"global_object_rank": 9, "person_only_rank": None, "desc": "tie", "coco_ann_id": -9, "bbox": [10, 10, 30, 40]},
    ]
    parsed = parse_legacy_generation(render_legacy_row("person", [10, 10, 30, 40]))
    result = match_parsed_row_to_candidates(parsed, candidates)
    assert result["status"] == "matched"
    assert result["matched_global_object_rank"] == 7
    assert result["matched_person_only_rank"] == 5
    assert bbox_iou([10, 10, 30, 40], [10, 10, 30, 40]) == pytest.approx(1.0)
    wrong_category = parse_legacy_generation(render_legacy_row("book", [70, 70, 90, 100]))
    wrong_result = match_parsed_row_to_candidates(wrong_category, candidates)
    assert wrong_result["status"] == "unmatched"
    assert wrong_result["reason"] == "no_exact_coco_category"
    assert len(wrong_result["candidate_rankings"]) == len(candidates)


def test_sampling_match_reports_ambiguous_same_category_objects() -> None:
    parsed = parse_legacy_generation(render_legacy_row("person", [10, 10, 30, 40]))
    candidates = [
        {"global_object_rank": 1, "person_only_rank": 0, "desc": "person", "bbox": [10, 10, 30, 40]},
        {"global_object_rank": 2, "person_only_rank": 1, "desc": "person", "bbox": [10, 10, 30, 40]},
    ]
    result = match_parsed_row_to_candidates(parsed, candidates)
    assert result["status"] == "ambiguous"
    assert result["ambiguous_global_object_ranks"] == [1, 2]
    assert MATCH_IOU_THRESHOLD == pytest.approx(0.5)
    assert MATCH_AMBIGUITY_MARGIN == pytest.approx(0.05)


def test_sampling_seed_contract_is_frozen_and_paired() -> None:
    assert SAMPLING_SEEDS == tuple(range(24))
    assert len(SAMPLING_SEEDS) == len(set(SAMPLING_SEEDS)) == 24
    assert SAMPLING_TEMPERATURE == pytest.approx(0.4)
    assert SAMPLING_TOP_P == pytest.approx(0.95)
    assert SAMPLING_REPETITION_PENALTY == pytest.approx(1.0)
    assert SAMPLING_MAX_NEW_TOKENS == 9


def test_sampling_stop_waits_for_complete_row_not_description_length() -> None:
    kwargs = {
        "object_ref_start_token_id": 10,
        "box_start_token_id": 11,
        "coordinate_token_ids": {20, 21, 22, 23},
        "terminal_token_id": 99,
    }
    one_token_description = [10, 12, 11, 20, 21, 22, 23]
    two_token_description = [10, 12, 13, 11, 20, 21, 22, 23]
    assert legacy_sampling_suffix_is_complete(one_token_description, **kwargs)
    assert legacy_sampling_suffix_is_complete(two_token_description, **kwargs)
    assert not legacy_sampling_suffix_is_complete([10, 12, 11, 20, 21, 22], **kwargs)
    assert legacy_sampling_suffix_is_complete([99], **kwargs)


def test_historical_sorted_prompt_hash_is_frozen() -> None:
    prompt = build_historical_prompts(ordering="sorted")
    assert prompt["hash"] == EXPECTED_PROMPT_HASH
    assert verify_historical_prompt()["hash"] == EXPECTED_PROMPT_HASH
    assert build_historical_prompts(ordering="random")["hash"] != EXPECTED_PROMPT_HASH


def test_geometry_sort_assigns_global_and_person_ranks() -> None:
    record = {
        "objects": [
            {"desc": "tie", "bbox_2d": ["<|coord_5|>", "<|coord_1|>", "<|coord_8|>", "<|coord_4|>"]},
            {"desc": "person", "bbox_2d": ["<|coord_2|>", "<|coord_1|>", "<|coord_3|>", "<|coord_4|>"]},
            {"desc": "person", "bbox_2d": ["<|coord_1|>", "<|coord_9|>", "<|coord_3|>", "<|coord_10|>"]},
        ]
    }
    objects = geometry_sorted_objects(record, require_expected_person_count=False)
    assert [(x["desc"], x["global_object_rank"]) for x in objects] == [("person", 0), ("tie", 1), ("person", 2)]
    assert [x["person_only_rank"] for x in objects if x["desc"] == "person"] == [0, 1]


def test_score_token_sequence_decomposes_all_slots() -> None:
    row_ids = [1, 2, 3, 4, 5, 6, 7]
    logits = torch.zeros((12, 20), dtype=torch.float32)
    # Give each target a distinct, known logit at its prediction position.
    for index, token in enumerate(row_ids):
        logits[4 + index, token] = float(index + 1)
    result = score_token_sequence(logits, boundary_length=5, row_token_ids=row_ids)
    expected = [float(torch.log_softmax(logits[4 + i], dim=-1)[token]) for i, token in enumerate(row_ids)]
    assert result["token_count"] == 7
    assert result["token_ids"] == row_ids
    assert result["token_log_probs"] == pytest.approx([float(v) for v in expected])
    assert result["total_log_probability"] == pytest.approx(math.fsum(float(v) for v in expected))
    assert result["description_log_probability"] == pytest.approx(float(expected[1]))
    assert result["coordinate_log_probabilities"]["x2"] == pytest.approx(float(expected[5]))


def test_tail_logits_have_same_local_score_as_full_logits() -> None:
    row_ids = [1, 2, 3, 4, 5, 6, 7]
    full = torch.zeros((30, 20), dtype=torch.float32)
    for index, token in enumerate(row_ids):
        full[22 + index, token] = float(index + 1)
    full_score = score_token_sequence(full, boundary_length=23, row_token_ids=row_ids)
    tail_score = score_token_sequence(full[-8:], boundary_length=1, row_token_ids=row_ids)
    assert tail_score["token_log_probs"] == pytest.approx(full_score["token_log_probs"])
    assert tail_score["total_log_probability"] == pytest.approx(full_score["total_log_probability"])


def test_merge_reports_candidate_mass_correlations_and_deltas() -> None:
    records = [_arm(adapter, owner, 0.0 if adapter == "random" else 0.5) for adapter in ("random", "sorted") for owner in OWNER_PERSON_RANKS]
    result = merge_arm_records(records)
    assert set(result["owners"]) == {str(owner) for owner in OWNER_PERSON_RANKS}
    for owner in OWNER_PERSON_RANKS:
        item = result["owners"][str(owner)]
        expected_global_rank = 19 if owner == 14 else owner
        assert item["owner_global_rank"] == expected_global_rank
        assert item["random"]["all_candidate_logsumexp"] > float("-inf")
        assert item["random"]["person_candidate_logsumexp"] > item["random"]["tie_logsumexp"]
        assert item["top1_agreement"] == 1.0
        assert item["top5_agreement"] == 1.0
        assert result["correlations"][str(owner)]["pearson_total_log_probability"] == pytest.approx(1.0)
        assert item["sorted_minus_random_by_candidate"]["0"]["total_log_probability"] == pytest.approx(0.5)
        for adapter in ("random", "sorted"):
            restricted = item[adapter]["candidate_restricted_probabilities"]
            assert restricted["person_probability_over_all_46"] + restricted["tie_probability_over_all_46"] == pytest.approx(1.0)
            assert restricted["emitted_probability_over_all_46"] + restricted["uncovered_probability_over_all_46"] == pytest.approx(1.0)
            assert restricted["emitted_person_probability_given_person"] + restricted["uncovered_person_probability_given_person"] == pytest.approx(1.0)


@pytest.mark.parametrize("owner", OWNER_PERSON_RANKS)
def test_image2299_packet_has_frozen_owner_identity(owner: int) -> None:
    packet = select_image2299_rows()
    assert len(packet["objects"]) == 46
    assert len(packet["people"]) == 38
    assert packet["people"][owner]["person_only_rank"] == owner
