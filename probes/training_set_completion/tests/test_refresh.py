import pytest

from probes.training_set_completion import refresh as r


def test_refresh_plan_is_exactly_one_seeded_route_per_policy_per_image():
    rows = r.request_plan()
    assert len(rows) == 44
    assert [row for row in rows if row["kind"] == "greedy"][0]["seed"] == r.SEED_ROOT
    assert len({row["seed"] for row in rows if row["kind"] == "sample"}) == 33
    assert {row["image_id"] for row in rows} == set(r.IMAGE_IDS)


def test_refresh_plan_rejects_a_missing_policy_route():
    rows = r.request_plan()
    with pytest.raises(ValueError, match="denominator"):
        r.validate_request_plan(rows[:-1])


def test_greedy_comparison_requires_literal_token_identity():
    expected = {"route_id": "stage01:image-000000000001:greedy", "generated_token_ids": [1, 2], "generated_token_ids_sha256": r.digest([1, 2])}
    observed = {"request": {"request_id": "stage02:image-000000000001:greedy"}, "image_id": 1, "generated_token_ids": [1, 3], "generated_token_ids_sha256": r.digest([1, 3])}
    assert r._compare_greedy(observed, expected)["same_token_ids"] is False
