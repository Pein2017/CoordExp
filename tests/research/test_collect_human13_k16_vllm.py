from __future__ import annotations

from types import SimpleNamespace

import pytest

import scripts.research.collect_human13_k16_vllm as collector


def test_one_image_is_four_successive_batches_of_four_explicit_n1_requests() -> None:
    batches = collector.plan_image_requests(image_id=1584)

    assert len(batches) == 4
    assert [batch.batch_index for batch in batches] == [0, 1, 2, 3]
    assert [[request.seed for request in batch.requests] for batch in batches] == [
        [21001, 21002, 21003, 21004],
        [21005, 21006, 21007, 21008],
        [21009, 21010, 21011, 21012],
        [21013, 21014, 21015, 21016],
    ]
    assert all(len(batch.requests) == 4 for batch in batches)
    assert all(
        request.request_id == f"human13:1584:k16:{request.seed}"
        for batch in batches
        for request in batch.requests
    )
    assert all(
        request.sampling
        == {
            "n": 1,
            "seed": request.seed,
            "temperature": 0.4,
            "top_p": 0.95,
            "repetition_penalty": 1.10,
            "max_tokens": 512,
        }
        for batch in batches
        for request in batch.requests
    )


def test_batch_execution_requires_four_independent_n1_requests() -> None:
    batch = collector.plan_image_requests(image_id=1584)[0]
    seen: list[tuple[tuple[str, ...], tuple[dict[str, object], ...]]] = []

    def execute(
        requests: tuple[collector.PlannedRequest, ...],
    ) -> list[dict[str, object]]:
        seen.append(
            (
                tuple(request.request_id for request in requests),
                tuple(request.sampling for request in requests),
            )
        )
        return [
            {
                "request_id": request.request_id,
                "seed": request.seed,
                "value": request.seed,
            }
            for request in reversed(requests)
        ]

    bound = collector.execute_batch(batch=batch, execute=execute)

    assert [result.seed for result in bound] == [21001, 21002, 21003, 21004]
    assert seen == [
        (
            tuple(request.request_id for request in batch.requests),
            tuple(request.sampling for request in batch.requests),
        )
    ]
    assert all(sampling["n"] == 1 for _, samplings in seen for sampling in samplings)


def test_result_binding_restores_seed_order_and_rejects_incomplete_or_duplicate_results() -> (
    None
):
    plan = collector.plan_image_requests(image_id=1584)
    expected = [request for batch in plan for request in batch.requests]
    reversed_results = [
        {"request_id": request.request_id, "seed": request.seed, "value": request.seed}
        for request in reversed(expected)
    ]

    bound = collector.bind_results(requests=expected, results=reversed_results)

    assert [result.seed for result in bound] == list(range(21001, 21017))
    with pytest.raises(ValueError, match="exactly once"):
        collector.bind_results(requests=expected, results=reversed_results[:-1])
    with pytest.raises(ValueError, match="exactly once"):
        collector.bind_results(
            requests=expected,
            results=[*reversed_results, reversed_results[0]],
        )


def test_result_binding_rejects_seed_or_request_identity_mismatch() -> None:
    request = collector.plan_image_requests(image_id=1584)[0].requests[0]

    with pytest.raises(ValueError, match="seed does not match"):
        collector.bind_results(
            requests=(request,),
            results=[{"request_id": request.request_id, "seed": 21002}],
        )
    with pytest.raises(ValueError, match="unexpected request"):
        collector.bind_results(
            requests=(request,),
            results=[{"request_id": "human13:1584:k16:99999", "seed": request.seed}],
        )


def test_unavailable_cache_telemetry_makes_no_reuse_claim() -> None:
    telemetry = collector.cache_telemetry(SimpleNamespace())

    assert telemetry == {
        "status": "unavailable",
        "cache_reuse_observed": False,
        "reason": "runtime exposes no encoder/prefix/cache counters",
    }


def test_dry_run_exact_panel_is_plan_only_without_runtime_import() -> None:
    summary = collector.dry_run_summary()

    assert summary["image_count"] == 13
    assert summary["physical_batch_count"] == 52
    assert summary["request_count"] == 208
    assert summary["execution"] == "not_started"
    assert summary["cache_telemetry"]["status"] == "unavailable"
    assert summary["sampling_contract"]["n"] == 1
