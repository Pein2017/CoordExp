import pytest

from probes.training_set_completion import route_bank as b


def _pred():
    pieces = ["<s>", "two ", "words", "</s>", "<b>", "<c0>", "<c1>", "<c2>", "<c3>", "</b>"]
    text = "".join(pieces)
    spans, cursor = [], 0
    for piece in pieces:
        spans.append((cursor, cursor + len(piece))); cursor += len(piece)
    return text, spans, {
        "char_start": 0, "char_end": len(text), "description_predicted": "two words",
        "schema_spans": [{"char_start": spans[0][0], "char_end": spans[0][1]}, {"char_start": spans[3][0], "char_end": spans[3][1]}, {"char_start": spans[4][0], "char_end": spans[4][1]}, {"char_start": spans[9][0], "char_end": spans[9][1]}],
        "coord_token_spans": [{"char_start": spans[i][0], "char_end": spans[i][1]} for i in range(5, 9)],
        "coord_bins": [0, 1, 2, 3], "_token_ids": [90, 91, 92, 93, 94, 10, 11, 12, 13, 95],
    }


def _review(**overrides):
    return {"proposal_id": "r:p0", "root_owner_id": "owner", "reviewed_owner_id": "owner", "physical_status": "true_unique", "root_repeat_after_alias": False, "parser_status": "parsed_valid", "raw_geometry_invalid": False, "effective_extent": "reasonable", "effective_direct_CE": "positive", "class": "verified", "reviewed": True, "coord_bins_1000": [0, 1, 2, 3], **overrides}


def test_positive_multitoken_description_and_schema_masks_align_to_original_tokens():
    text, spans, pred = _pred()
    positions, card = b._mask_fields(acquisition_pred=pred, text=text, token_spans=spans, review=_review(), targets={"owner"}, coordinate_ids=[10, 11, 12, 13])
    assert positions == set(range(10))
    assert card["description_positions"] == [1, 2]
    assert card["coordinate_positions"] == [5, 6, 7, 8]
    assert card["positive_positions"] == list(range(10))


def test_invalid_row_is_retained_in_trace_but_has_no_positive_positions():
    text, spans, pred = _pred()
    positions, card = b._mask_fields(acquisition_pred=pred, text=text, token_spans=spans, review=_review(physical_status="invalid", raw_geometry_invalid=True, effective_direct_CE="mask"), targets={"owner"}, coordinate_ids=[10, 11, 12, 13])
    assert positions == set()
    assert card["row_token_positions"] == list(range(10))
    assert card["reason"] == "masked_by_review_or_nonpositive_field"


def test_dict_bbox_correct_never_becomes_whole_row_positive():
    text, spans, pred = _pred()
    positions, card = b._mask_fields(acquisition_pred=pred, text=text, token_spans=spans, review=_review(effective_direct_CE={"bbox": "correct", "description": "positive"}), targets={"owner"}, coordinate_ids=[10, 11, 12, 13])
    assert positions == set()
    assert card["positive_positions"] == []


def test_effective_class_override_masks_description_despite_raw_verified_class():
    text, spans, pred = _pred()
    positions, card = b._mask_fields(
        acquisition_pred=pred,
        text=text,
        token_spans=spans,
        review=_review(**{"class": "verified", "effective_class": "unknown"}),
        targets={"owner"},
        coordinate_ids=[10, 11, 12, 13],
    )
    assert set(card["schema_positions"]).issubset(positions)
    assert set(card["description_positions"]).isdisjoint(positions)


def test_route_selection_prioritizes_coverage_then_bad_and_unknown_debt_then_temperature():
    targets = {"a", "b"}
    base = {"physical_status": "true_unique", "parser_status": "parsed_valid", "raw_geometry_invalid": False, "effective_extent": "reasonable"}
    rows = [
        {**base, "request_id": "greedy", "temperature": 0.0, "root_owner_id": "a"},
        {**base, "request_id": "sample", "temperature": 0.3, "root_owner_id": "a"},
        {**base, "request_id": "sample", "temperature": 0.3, "root_owner_id": "b"},
        {**base, "request_id": "other1", "temperature": 0.1, "root_owner_id": "a", "physical_status": "repeat"},
        {**base, "request_id": "other2", "temperature": 0.7, "root_owner_id": "a", "physical_status": "unknown"},
    ]
    # Production data always has four policies; this fixture includes one
    # duplicate clean coverage row to make the deterministic ranking observable.
    request, card = b.choose_route(rows, targets)
    assert request == "sample"
    assert card["distinct_target_owners"] == 2
