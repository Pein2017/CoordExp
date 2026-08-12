from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace
import sys
from types import ModuleType

import pytest

import scripts.research.collect_human13_k16_vllm as collector
from src.common.errors import RuntimeContractError


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


def test_vllm_session_seam_uses_numeric_native_ids_and_preserves_collector_attribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = collector.plan_image_requests(image_id=1584)[0]
    fake_vllm = ModuleType("vllm")

    class SamplingParams:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    fake_vllm.SamplingParams = SamplingParams  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    @dataclass(frozen=True)
    class BaseRequest:
        request_id: str
        image_sha256: str

    @dataclass(frozen=True)
    class NativeOutput:
        request_id: str

    class Engine:
        def __init__(self) -> None:
            self.calls: list[tuple[list[object], list[SamplingParams]]] = []
            self.native_ids = ("103", "100", "102", "101")

        def generate(
            self,
            prompts: list[object],
            sampling_params: list[SamplingParams],
            *,
            use_tqdm: bool,
        ) -> list[NativeOutput]:
            assert use_tqdm is False
            self.calls.append((prompts, sampling_params))
            return [NativeOutput(request_id) for request_id in self.native_ids]

    class Session:
        def __init__(self) -> None:
            self._engine = Engine()
            self.request_ids: tuple[str, ...] = ()

        def _generation_prompts(
            self, requests: tuple[BaseRequest, ...]
        ) -> tuple[list[object], list[str]]:
            self.request_ids = tuple(request.request_id for request in requests)
            return ([{} for _ in requests], ["hash" for _ in requests])

        def _im_end_token_id(self) -> int:
            return 99

    session = Session()
    expected_image_sha256 = dict(collector.EXPECTED_IMAGE_IDENTITIES)[1584]
    bound = collector.execute_vllm_batch(
        session=session,
        base_request=BaseRequest(request_id="base", image_sha256=expected_image_sha256),
        batch=batch,
    )

    assert session.request_ids == ("0", "1", "2", "3")
    assert [result.request_id for result in bound] == [
        request.request_id for request in batch.requests
    ]
    assert [result.seed for result in bound] == [21001, 21002, 21003, 21004]
    assert [result.payload["native_request_id"] for result in bound] == [
        "100",
        "101",
        "102",
        "103",
    ]
    assert [params.kwargs["n"] for params in session._engine.calls[0][1]] == [
        1,
        1,
        1,
        1,
    ]

    session._engine.native_ids = ("100", "101", "101", "102")
    with pytest.raises(RuntimeContractError, match="duplicate native request ids"):
        collector.execute_vllm_batch(
            session=session,
            base_request=BaseRequest(
                request_id="base", image_sha256=expected_image_sha256
            ),
            batch=batch,
        )


def test_batch_validation_rejects_noncanonical_seed_image_or_request_identity() -> None:
    batch = collector.plan_image_requests(image_id=1584)[1]
    first = batch.requests[0]
    wrong_seed = replace(
        first,
        seed=21001,
        request_id="human13:1584:k16:21001",
        sampling={**collector.SAMPLING, "seed": 21001},
    )
    with pytest.raises(ValueError, match="exact seed slice"):
        collector._validate_plan_for_batch(
            replace(batch, requests=(wrong_seed, *batch.requests[1:]))
        )

    wrong_image = replace(
        first,
        image_id=99999,
        request_id="human13:99999:k16:21005",
    )
    with pytest.raises(ValueError, match="single expected image"):
        collector._validate_plan_for_batch(
            replace(batch, requests=(wrong_image, *batch.requests[1:]))
        )

    wrong_request_id = replace(first, request_id="human13:1584:k16:99999")
    with pytest.raises(ValueError, match="canonical request identity"):
        collector._validate_plan_for_batch(
            replace(batch, requests=(wrong_request_id, *batch.requests[1:]))
        )


def test_vllm_batch_rejects_base_request_image_mismatch_before_runtime_action() -> None:
    batch = collector.plan_image_requests(image_id=1584)[0]

    @dataclass(frozen=True)
    class BaseRequest:
        request_id: str
        image_sha256: str

    class Session:
        def __init__(self) -> None:
            self.prompt_calls = 0
            self._engine = SimpleNamespace(generate=self._unexpected_engine_action)

        def _generation_prompts(
            self, requests: object
        ) -> tuple[list[object], list[str]]:
            del requests
            self.prompt_calls += 1
            raise AssertionError("base image mismatch reached prompt preparation")

        def _unexpected_engine_action(self, *args: object, **kwargs: object) -> None:
            del args, kwargs
            raise AssertionError("base image mismatch reached engine submission")

    session = Session()
    with pytest.raises(ValueError, match="base request image SHA-256"):
        collector.execute_vllm_batch(
            session=session,
            base_request=BaseRequest(request_id="base", image_sha256="0" * 64),
            batch=batch,
        )
    assert session.prompt_calls == 0
