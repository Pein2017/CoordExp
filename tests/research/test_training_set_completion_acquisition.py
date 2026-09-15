import copy

import pytest

from probes.training_set_completion import acquisition as a


def test_request_plan_is_exact_unique_11_by_4_with_distinct_sample_seeds():
    rows = a.request_plan()
    assert len(rows) == 44
    assert [row["image_id"] for row in rows[:4]] == [a.IMAGE_IDS[0]] * 4
    assert [row["temperature"] for row in rows[:4]] == [0.0, 0.1, 0.3, 0.7]
    assert len({row["request_id"] for row in rows}) == 44
    assert len({row["seed"] for row in rows if row["seed"] is not None}) == 33


def test_request_plan_rejects_seed_reuse_and_missing_request():
    rows = a.request_plan()
    reused = copy.deepcopy(rows)
    reused[2]["seed"] = reused[1]["seed"]
    with pytest.raises(ValueError, match="sample seeds"):
        a.validate_request_plan(reused)
    with pytest.raises(ValueError, match="request denominator"):
        a.validate_request_plan(rows[:-1])


def _record():
    return {"example_id": "e", "image_id": a.IMAGE_IDS[0], "prompt_token_ids": [4, 5],
            "case": {"image_plan": {"executed_media_sha256": "m", "observed_image_grid_thw": [1, 2, 3]}}}


def _payload(request):
    ids = [7, a.EOS]
    return {"schema": f"{a.SCHEMA}.row", "request": dict(request), "example_id": "e",
            "image_id": a.IMAGE_IDS[0], "empty_assistant_prefix": True, "assistant_token_cap": a.CAP,
            "prompt_token_ids": [4, 5], "prompt_token_ids_sha256": a.digest([4, 5]),
            "generated_token_ids": ids, "generated_token_ids_sha256": a.digest(ids),
            "generated_token_count": 2, "raw_decode_text": "decoded", "decode_stop_reason": "im_end",
            "executed_media_sha256": "m", "observed_image_grid_thw": [1, 2, 3]}


def test_result_validation_binds_budget_terminal_token_text_prompt_and_media():
    request = a.request_plan()[0]
    a.validate_result_payload(_payload(request), request, _record(), decode=lambda ids: "decoded")
    bad = _payload(request)
    bad["generated_token_ids"] = [a.EOS, 8]
    bad["generated_token_ids_sha256"] = a.digest(bad["generated_token_ids"])
    with pytest.raises(ValueError, match="generated terminal"):
        a.validate_result_payload(bad, request, _record(), decode=lambda ids: "decoded")


def test_phase_partition_reuses_smoke_without_overlap():
    requests = a.request_plan()
    manifest = {"requests": requests, "execution": {"smoke_request_ids": [requests[0]["request_id"], requests[1]["request_id"]]}}
    smoke = a._phase_requests(manifest, "smoke", 0, 1)
    remaining = [row for shard in range(8) for row in a._phase_requests(manifest, "remaining", shard, 8)]
    assert len(smoke) == 2 and len(remaining) == 42
    assert {row["request_id"] for row in smoke}.isdisjoint(row["request_id"] for row in remaining)
    assert {row["request_id"] for row in smoke + remaining} == {row["request_id"] for row in requests}


def test_geometry_invalid_count_uses_parser_drop_reason():
    rows = [
        {"parsed": {"dropped_predictions": [{"reason": "geometry_invalid"}, {"reason": "incomplete"}]}},
        {"parsed": {"dropped_predictions": [{"reason": "geometry_invalid"}]}},
    ]
    assert a.geometry_invalid_count(rows) == 2
