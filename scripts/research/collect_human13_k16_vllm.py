#!/usr/bin/env python3
"""Collect the Human-13 K=16 sampled-discovery ledger with explicit requests.

This is an experiment-local collector.  It deliberately fan-outs every image
into four successive batches of four independent ``n=1`` requests so seed
identity never depends on a vLLM multi-completion child index.  The clean
greedy collector and final manifest admission are separate steps.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
import json
from pathlib import Path
import sys
from typing import Any
from types import MappingProxyType


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


from scripts.research.build_human13_k_union_manifest import (  # noqa: E402
    EXPECTED_IMAGE_IDENTITIES,
    EXPECTED_K_SEEDS,
)


REQUESTS_PER_BATCH = 4
PHYSICAL_BATCHES_PER_IMAGE = 4
SAMPLING = {
    "n": 1,
    "temperature": 0.4,
    "top_p": 0.95,
    "repetition_penalty": 1.10,
    "max_tokens": 512,
}


@dataclass(frozen=True)
class PlannedRequest:
    """One explicit, independently seeded vLLM request."""

    image_id: int
    seed: int
    physical_batch_index: int
    request_id: str
    sampling: Mapping[str, object]


@dataclass(frozen=True)
class PlannedBatch:
    """One physical submission, always containing four independent requests."""

    image_id: int
    batch_index: int
    requests: tuple[PlannedRequest, ...]


@dataclass(frozen=True)
class BoundResult:
    """A result bound to its declared request and restored to canonical seed order."""

    request_id: str
    seed: int
    payload: Mapping[str, object]


def _request_id(*, image_id: int, seed: int) -> str:
    return f"human13:{image_id}:k16:{seed}"


def plan_image_requests(*, image_id: int) -> tuple[PlannedBatch, ...]:
    """Return the immutable four-by-four request plan for one image."""

    batches: list[PlannedBatch] = []
    for batch_index in range(PHYSICAL_BATCHES_PER_IMAGE):
        start = batch_index * REQUESTS_PER_BATCH
        requests = tuple(
            PlannedRequest(
                image_id=image_id,
                seed=seed,
                physical_batch_index=batch_index,
                request_id=_request_id(image_id=image_id, seed=seed),
                sampling=MappingProxyType({**SAMPLING, "seed": seed}),
            )
            for seed in EXPECTED_K_SEEDS[start : start + REQUESTS_PER_BATCH]
        )
        batches.append(
            PlannedBatch(
                image_id=image_id,
                batch_index=batch_index,
                requests=requests,
            )
        )
    _validate_plan(batches)
    return tuple(batches)


def plan_panel_requests() -> tuple[PlannedBatch, ...]:
    """Return canonical Human-13 plans in the frozen panel order."""

    return tuple(
        batch
        for image_id, _ in EXPECTED_IMAGE_IDENTITIES
        for batch in plan_image_requests(image_id=image_id)
    )


def _validate_plan(batches: Sequence[PlannedBatch]) -> None:
    if len(batches) != PHYSICAL_BATCHES_PER_IMAGE:
        raise ValueError("each image requires exactly four physical batches")
    image_ids = {batch.image_id for batch in batches}
    if len(image_ids) != 1:
        raise ValueError("one image plan may not mix image identities")
    if [batch.batch_index for batch in batches] != list(
        range(PHYSICAL_BATCHES_PER_IMAGE)
    ):
        raise ValueError("physical batches must be in canonical order 0..3")
    requests = [request for batch in batches for request in batch.requests]
    if any(len(batch.requests) != REQUESTS_PER_BATCH for batch in batches):
        raise ValueError("each physical batch requires exactly four requests")
    if [request.seed for request in requests] != list(EXPECTED_K_SEEDS):
        raise ValueError("one image plan requires seeds 21001..21016 in order")
    if len({request.request_id for request in requests}) != len(requests):
        raise ValueError("one image plan contains duplicate request identities")
    for request in requests:
        if request.sampling != {**SAMPLING, "seed": request.seed}:
            raise ValueError("request sampling differs from the frozen K=16 contract")
        if request.sampling["n"] != 1:
            raise ValueError("K=16 sampling prohibits n>1 requests")


def _result_value(result: Mapping[str, object] | object, field: str) -> object:
    if isinstance(result, Mapping):
        return result.get(field)
    return getattr(result, field, None)


def bind_results(
    *,
    requests: Sequence[PlannedRequest],
    results: Sequence[Mapping[str, object] | object],
) -> tuple[BoundResult, ...]:
    """Fail closed on result coverage, then restore the declared seed order."""

    expected_ids = [request.request_id for request in requests]
    observed_ids = [str(_result_value(result, "request_id")) for result in results]
    if Counter(expected_ids) != Counter(observed_ids) or any(
        count != 1 for count in Counter(observed_ids).values()
    ):
        if any(request_id not in expected_ids for request_id in observed_ids):
            raise ValueError("result set contains an unexpected request identity")
        raise ValueError("results must cover every declared request exactly once")
    by_request_id = {
        str(_result_value(result, "request_id")): result for result in results
    }
    bound: list[BoundResult] = []
    for request in requests:
        result = by_request_id[request.request_id]
        result_seed = _result_value(result, "seed")
        if result_seed != request.seed:
            raise ValueError("result seed does not match its declared request")
        payload = (
            dict(result) if isinstance(result, Mapping) else {"native_result": result}
        )
        bound.append(
            BoundResult(
                request_id=request.request_id,
                seed=request.seed,
                payload=payload,
            )
        )
    return tuple(sorted(bound, key=lambda result: result.seed))


def execute_batch(
    *,
    batch: PlannedBatch,
    execute: Callable[
        [tuple[PlannedRequest, ...]], Sequence[Mapping[str, object] | object]
    ],
) -> tuple[BoundResult, ...]:
    """Execute exactly one declared physical batch through an injected seam."""

    _validate_plan_for_batch(batch)
    return bind_results(requests=batch.requests, results=execute(batch.requests))


def _validate_plan_for_batch(batch: PlannedBatch) -> None:
    if len(batch.requests) != REQUESTS_PER_BATCH:
        raise ValueError("each physical batch requires exactly four requests")
    if any(
        request.physical_batch_index != batch.batch_index for request in batch.requests
    ):
        raise ValueError("request physical-batch identity does not match its batch")
    if any(
        request.sampling != {**SAMPLING, "seed": request.seed}
        for request in batch.requests
    ):
        raise ValueError("request sampling differs from the frozen K=16 contract")
    if any(request.sampling["n"] != 1 for request in batch.requests):
        raise ValueError("K=16 sampling prohibits n>1 requests")


def cache_telemetry(runtime: object) -> dict[str, object]:
    """Record optional counters without inferring cache reuse from request shape."""

    counters = getattr(runtime, "cache_telemetry", None)
    if not isinstance(counters, Mapping):
        return {
            "status": "unavailable",
            "cache_reuse_observed": False,
            "reason": "runtime exposes no encoder/prefix/cache counters",
        }
    return {
        "status": "available",
        "cache_reuse_observed": False,
        "counters": dict(counters),
    }


def dry_run_summary() -> dict[str, object]:
    """Return the exact CPU-only request plan without opening a runtime session."""

    batches = plan_panel_requests()
    requests = [request for batch in batches for request in batch.requests]
    return {
        "image_count": len(EXPECTED_IMAGE_IDENTITIES),
        "physical_batch_count": len(batches),
        "request_count": len(requests),
        "execution": "not_started",
        "sampling_contract": dict(SAMPLING),
        "cache_telemetry": cache_telemetry(object()),
    }


def _sampling_params(request: PlannedRequest, *, stop_token_id: int) -> Any:
    """Import vLLM only on the explicit execution path."""

    from vllm import SamplingParams

    return SamplingParams(
        **request.sampling,
        top_k=0,
        stop_token_ids=[int(stop_token_id)],
        ignore_eos=False,
        detokenize=True,
        skip_special_tokens=False,
        spaces_between_special_tokens=True,
    )


def execute_vllm_batch(
    *,
    session: Any,
    base_request: Any,
    batch: PlannedBatch,
) -> tuple[BoundResult, ...]:
    """Submit one batch through the already-open backend-neutral vLLM session."""

    _validate_plan_for_batch(batch)
    from src.inference.vllm_backend import (
        _close_prompt_images,
        _restore_native_request_order,
    )

    decode_requests = tuple(
        replace(base_request, request_id=request.request_id)
        for request in batch.requests
    )
    prompts, media_hashes = session._generation_prompts(decode_requests)
    del media_hashes  # The later artifact projection owns executed-RGB evidence.
    try:
        outputs = session._engine.generate(
            prompts,
            [
                _sampling_params(request, stop_token_id=session._im_end_token_id())
                for request in batch.requests
            ],
            use_tqdm=False,
        )
    finally:
        _close_prompt_images(prompts)
    ordered_outputs = _restore_native_request_order(outputs, REQUESTS_PER_BATCH)
    raw_results = tuple(
        {
            "request_id": request.request_id,
            "seed": request.seed,
            "native_output": output,
        }
        for request, output in zip(batch.requests, ordered_outputs, strict=True)
    )
    return bind_results(requests=batch.requests, results=raw_results)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the exact 13-image, 52-batch, 208-request plan without runtime action.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.dry_run:
        raise SystemExit(
            "this collector requires --dry-run until explicit model execution"
        )
    print(json.dumps(dry_run_summary(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
