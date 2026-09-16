import pytest
from probes.training_set_completion import complete_bank as b


def test_select_requires_complete_unique_catalog_coverage_and_keeps_first_source_occurrence():
    records = [{"owner_id": "a"}, {"owner_id": "b"}]
    source = [
        {"owner_id": "a", "generated_order": 4},
        {"owner_id": "a", "generated_order": 1},
    ]
    picked = b._select(records, source)
    assert [(x[0]["owner_id"], x[1] and x[1]["generated_order"], x[2]) for x in picked] == [("a", 1, False), ("b", None, True)]
    with pytest.raises(ValueError, match="duplicate target owner"):
        b._select([{"owner_id": "a"}, {"owner_id": "a"}], source)


def test_unknown_catalog_class_retains_observed_literal_but_masks_description_ce():
    desc, positive, source = b._description({"category": None}, {"class": "unknown", "raw": {"description": "bowl"}}, None)
    assert (desc, positive, source) == ("bowl", False, "unknown_class_observed_literal_masked")
    with pytest.raises(ValueError, match="unknown class requires"):
        b._description({"category": None}, {"class": "wrong", "raw": {"description": None}}, None)

def test_root_unknown_admission_masks_description_even_if_an_older_source_was_verified():
    desc, positive, source = b._description(
        {"category": None, "legacy_observed_description": "bowl"},
        {"class": "verified", "raw": {"description": "bowl"}},
        {"class_policy": "mask_description"},
    )
    assert (desc, positive, source) == ("bowl", False, "root_unknown_observed_literal_masked")


def _candidate():
    import copy
    import json

    return copy.deepcopy(json.loads((b.ROOT / "bank.json").read_text()))


def _rehash(candidate):
    candidate["content_sha256"] = b.digest({key: value for key, value in candidate.items() if key != "content_sha256"})


def test_real_candidate_consumer_rejects_owner_coverage_and_coordinate_trace_mutations():
    missing_owner = _candidate()
    trace = missing_owner["routes"][0]["provenance"]["trace"]
    trace[1]["owner_id"] = trace[0]["owner_id"]
    _rehash(missing_owner)
    with pytest.raises(ValueError, match="route exact completeness"):
        b.validate(missing_owner)

    wrong_geometry = _candidate()
    wrong_geometry["routes"][0]["trusted_boxes"][0]["expected_bins"][0] += 1
    _rehash(wrong_geometry)
    with pytest.raises(ValueError, match="trace geometry"):
        b.validate(wrong_geometry)


def test_verified_observed_stage01_description_can_override_wrong_parent_text():
    desc, positive, source = b._description(
        {"category": None, "legacy_verified_description": "spoon", "legacy_observed_description": "spoon"},
        {"class": "wrong", "raw": {"description": "bowl"}},
        None,
    )
    assert (desc, positive, source) == ("spoon", True, "accepted_source_verified_class")


def test_real_candidate_consumer_rejects_stale_source_binding():
    stale = _candidate()
    stale["sources"]["catalog"]["sha256"] = "0" * 64
    _rehash(stale)
    with pytest.raises(ValueError, match="source identity: catalog"):
        b.validate(stale)


def test_admitted_parent_step16_owner_retains_native_generated_order_in_v2_candidate():
    bank = _candidate()
    route = next(route for route in bank["routes"] if route["image_id"] == 219546)
    card = next(card for card in route["provenance"]["trace"] if card["owner_id"] == "first-fit:new:219546:left-bread-basket")
    assert card["source_kind"] == "first_fit_step16_first_occurrence"
    assert card["source_decision"]["proposal_id"] == "first-fit:step16:image-000000219546:p3"
    assert card["source_decision"]["generated_order"] == 3


def test_step32_only_admission_remains_deterministically_appended_in_v2_candidate():
    bank = _candidate()
    route = next(route for route in bank["routes"] if route["image_id"] == 25274)
    card = next(card for card in route["provenance"]["trace"] if card["owner_id"] == "first-fit:new:25274:black-coated-person")
    assert card["source_kind"] == "appended_missing_owner"
    assert card["source_decision"] is None
