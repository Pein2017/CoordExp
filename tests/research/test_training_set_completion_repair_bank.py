import copy

import pytest

from probes.training_set_completion import repair_bank as b


def _manifest():
    requests = [
        {"request_id": f"r{index}", "image_id": 1, "temperature": temperature, "kind": "sample", "seed": index}
        for index, temperature in enumerate((0.0, 0.1, 0.3, 0.7))
    ]
    return {
        "content_sha256": "manifest",
        "runtime": {"cap": 20, "eos": b.EOS},
        "requests": requests,
        "prefixes": {"1": {"forced_prefix": [7, 8]}},
    }


def _candidate(request, suffix):
    prefix = [7, 8]
    full = prefix + suffix
    return {
        "manifest_sha256": "manifest",
        "request": request,
        "image_id": 1,
        "forced_prefix_token_ids": prefix,
        "forced_prefix_sha256": b.digest(prefix),
        "generated_suffix_token_ids": suffix,
        "generated_suffix_sha256": b.digest(suffix),
        "continuation_token_ids": full,
        "continuation_token_ids_sha256": b.digest(full),
        "prefix_token_count": len(prefix),
        "suffix_token_count": len(suffix),
        "assistant_token_cap": 20,
        "decode_stop_reason": "im_end",
    }


def test_release_selection_requires_natural_eos_then_shortest_suffix_and_lower_temperature():
    manifest = _manifest()
    rows = [
        _candidate(manifest["requests"][0], [3, 4, b.EOS]),
        _candidate(manifest["requests"][1], [9, b.EOS]),
        _candidate(manifest["requests"][2], [6, b.EOS]),
        _candidate(manifest["requests"][3], [b.EOS]),
    ]
    rows[-1]["decode_stop_reason"] = "length"
    selected, trace = b.choose_release(rows, manifest)
    assert selected["request"]["request_id"] == "r1"
    assert trace["selected_request_id"] == "r1"
    assert next(row for row in trace["candidates"] if row["request_id"] == "r3")["rejection_reasons"] == ["not_natural_terminal_eos"]


def _frame():
    pieces = ["<s>", "donut", "</s>", "<b>", "<c0>", "<c1>", "<c2>", "<c3>", "</b>"]
    text = "".join(pieces)
    spans, cursor = [], 0
    for piece in pieces:
        spans.append((cursor, cursor + len(piece)))
        cursor += len(piece)
    pred = {
        "char_start": 0,
        "char_end": len(text),
        "description": "donut",
        "generated_order": 0,
        "schema_spans": [
            {"char_start": spans[index][0], "char_end": spans[index][1]}
            for index in (0, 2, 3, 8)
        ],
        "coord_token_spans": [
            {"char_start": spans[index][0], "char_end": spans[index][1]}
            for index in range(4, 8)
        ],
        "coord_bins": [0, 1, 2, 3],
    }
    return text, spans, pred


def _decision(owner="atomic", effective_class="verified", direct=None):
    return {
        "proposal_id": "source:p0",
        "image_id": 1,
        "generated_order": 0,
        "effective_owner_id": owner,
        "physical_status": "true_unique",
        "packet_raw_row_status": "parsed_valid",
        "effective_extent": "reasonable",
        "effective_class": effective_class,
        "effective_direct_CE": direct or {"bbox": "positive", "description": "positive"},
        "raw": {
            "coord_bins_1000": [0, 1, 2, 3],
            "description": "donut",
            "raw_axes_preserved": True,
            "status": "parsed_valid",
        },
    }


def test_positive_projection_for_non_atomic_crowd_owner_is_fully_masked(monkeypatch):
    text, spans, pred = _frame()
    monkeypatch.setattr(b, "_native_parse", lambda *args, **kwargs: (text, spans, {"pred": [pred], "dropped_predictions": []}))
    weights, trace, boxes, _ = b._mask_reviewed_prefix(
        ids=[90, 91, 92, 93, 10, 11, 12, 13, 94],
        tokenizer=object(),
        record={},
        stop_reason="im_end",
        decisions=[_decision(owner="crowd")],
        targets={"atomic"},
        prefix_source_token_count=9,
        coordinate_ids=[10, 11, 12, 13],
    )
    assert weights == [0] * 9
    assert trace[0]["reason"] == "masked_non_atomic_owner"
    assert boxes == []


def test_class_override_keeps_reviewed_geometry_but_masks_description(monkeypatch):
    text, spans, pred = _frame()
    monkeypatch.setattr(b, "_native_parse", lambda *args, **kwargs: (text, spans, {"pred": [pred], "dropped_predictions": []}))
    decision = _decision(effective_class="wrong", direct={"bbox": "positive", "description": "mask"})
    weights, trace, boxes, _ = b._mask_reviewed_prefix(
        ids=[90, 91, 92, 93, 10, 11, 12, 13, 94], tokenizer=object(), record={}, stop_reason="im_end",
        decisions=[decision], targets={"atomic"}, prefix_source_token_count=9, coordinate_ids=[10, 11, 12, 13],
    )
    assert all(weights[position] == 1 for position in trace[0]["schema_positions"])
    assert all(weights[position] == 0 for position in trace[0]["description_positions"])
    assert len(boxes) == 1 and boxes[0]["expected_bins"] == [0, 1, 2, 3]


def _known_decision(owner):
    return {
        "effective_owner_id": owner,
        "physical_status": "true_unique",
        "effective_extent": "reasonable",
        "effective_class": "verified",
        "effective_direct_CE": {"bbox": "positive", "description": "positive"},
    }


def test_323322_eos_requires_eos_only_suffix_all_seven_known_owners_and_no_debt():
    decisions = [_known_decision(str(index)) for index in range(6)]
    targets = {str(index) for index in range(7)}
    allowed, known, debt = b._allow_323322_eos(
        suffix=[b.EOS], source_decisions=decisions, forced_owner_id="6", targets=targets,
    )
    assert allowed and known == targets and not debt
    changed = copy.deepcopy(decisions)
    changed[0]["effective_extent"] = "wrong"
    assert not b._allow_323322_eos(suffix=[b.EOS], source_decisions=changed, forced_owner_id="6", targets=targets)[0]
    assert not b._allow_323322_eos(suffix=[99, b.EOS], source_decisions=decisions, forced_owner_id="6", targets=targets)[0]


def test_route_record_preserves_literal_ids_and_only_allows_declared_final_eos():
    ids = [10, 11, b.EOS]
    route = b._route_record(
        image=1, ids=ids, weights=[1, 1, 1], boxes=[], trace=[],
        source_record={"example_id": "case"}, route_id="route", eos_positive=True, provenance={},
    )
    assert route["continuation_token_ids"] == ids
    assert route["continuation_token_ids_sha256"] == b.digest(ids)
    with pytest.raises(ValueError, match="EOS mask contract"):
        b._route_record(
            image=1, ids=ids, weights=[1, 1, 0], boxes=[], trace=[],
            source_record={"example_id": "case"}, route_id="route", eos_positive=True, provenance={},
        )
