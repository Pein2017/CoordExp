from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from scripts.research import human13_rp_crossover_production_backend as backend_owner
from scripts.research import launch_human13_k_trajectory_rp_crossover as launcher
from scripts.research.collect_human13_rp_crossover import (
    NATURAL_STOP_TOKEN_ID,
    execute_acquisition_group,
    plan_acquisition_group,
)


_SamplerHandle = backend_owner._SamplerHandle


@dataclass(frozen=True)
class _BaseRequest:
    request_id: str
    generation_policy: object
    expected_executed_prompt_token_ids: tuple[int, ...]


@dataclass(frozen=True)
class _Result:
    request_id: str
    executed_prompt_token_ids: tuple[int, ...]
    generated_token_ids: tuple[int, ...]
    token_trace: tuple[object, ...]
    stop_reason: str

    def validate_for_request(self, request, receipt) -> None:
        del receipt
        assert self.request_id == request.request_id
        assert request.generation_policy.temperature == 0.0
        assert request.generation_policy.top_p == 1.0
        assert request.generation_policy.max_new_tokens == 512


class _Engine:
    def __init__(self) -> None:
        self.seed_batches: list[tuple[int, ...]] = []

    def generate(self, prompts, params, *, use_tqdm):
        assert use_tqdm is False
        assert len(prompts) == 4
        self.seed_batches.append(tuple(item.seed for item in params))
        return tuple(
            SimpleNamespace(request_id=str(index)) for index in reversed(range(4))
        )


class _Session:
    def __init__(self) -> None:
        self._engine = _Engine()
        self.receipt = object()

    def _generation_prompts(self, requests):
        return tuple({} for _ in requests), tuple("f" * 64 for _ in requests)

    def _materialize_result(
        self, *, request, native_output, executed_media_sha256, raw_logprobs
    ):
        del native_output, executed_media_sha256, raw_logprobs
        likelihood = SimpleNamespace(policy_logprob=-0.25)
        return _Result(
            request_id=request.request_id,
            executed_prompt_token_ids=request.expected_executed_prompt_token_ids,
            generated_token_ids=(NATURAL_STOP_TOKEN_ID,),
            token_trace=(SimpleNamespace(likelihood=likelihood),),
            stop_reason="im_end",
        )


def _sampler(image_id: int) -> _SamplerHandle:
    session = _Session()
    return _SamplerHandle(
        session=session,
        base_requests={
            image_id: _BaseRequest(
                request_id="base",
                generation_policy=object(),
                expected_executed_prompt_token_ids=(11, 12, 13),
            )
        },
        session_identity_sha256="a" * 64,
        model_id="source-model",
        model_identity_sha256="b" * 64,
        tokenizer_id="source-tokenizer",
        processor_id="source-processor",
        sampler_backend_id="vllm:test:native",
        frozen=SimpleNamespace(
            source_checkpoint_payload_sha256="c" * 64,
            manifest_sha256="d" * 64,
        ),
    )


def test_native_receipt_executor_projects_all_exact_batch_four_evidence(
    tmp_path,
) -> None:
    owner = backend_owner.Human13RPCrossoverProductionBackend(
        {"cells": ({"output_root": str(tmp_path / "node" / "cell")},)}
    )
    plan = plan_acquisition_group(
        image_id=1584,
        repetition_penalty=1.10,
        seed_group_id="qualification",
    )
    sampler = _sampler(plan.image_id)

    execution = execute_acquisition_group(
        plan=plan,
        execute_batch=lambda batch, params: owner.sample_batch(sampler, batch, params),
    )

    assert tuple(
        item.seed
        for receipt in execution.native_batch_receipts
        for item in receipt.requests
    ) == tuple(range(30001, 30017))
    assert sampler.session._engine.seed_batches == [
        tuple(range(start, start + 4)) for start in range(30001, 30017, 4)
    ]
    assert {
        item.processor_order
        for receipt in execution.native_batch_receipts
        for item in receipt.outputs
    } == {("repetition_penalty", "temperature", "log_softmax")}
    assert all(
        item.processed_logprobs == (-0.25,)
        for receipt in execution.native_batch_receipts
        for item in receipt.outputs
    )


def test_audit_burden_requires_one_unmixed_exact_thirteen_image_panel() -> None:
    outputs = tuple(
        {
            "image_id": image_id,
            "malformed_row_count": int(image_id == 7),
            "stop_reason": "length" if image_id == 8 else "im_end",
            "parser_status": ("all_spans_dropped" if image_id == 9 else "accepted"),
        }
        for image_id in range(1, 14)
    )

    assert backend_owner.Human13RPCrossoverProductionBackend._burdens(outputs) == {
        "malformed": 1,
        "cap_terminated": 1,
        "unparseable": 1,
    }
    with pytest.raises(backend_owner.ProductionBackendError, match="exact 13-image"):
        backend_owner.Human13RPCrossoverProductionBackend._burdens(
            (*outputs[:-1], outputs[0])
        )


@pytest.mark.parametrize("parser_status", ["empty", "unsupported_format"])
def test_audit_burden_counts_every_canonical_unparseable_status(
    parser_status: str,
) -> None:
    outputs = tuple(
        {
            "image_id": image_id,
            "malformed_row_count": 0,
            "stop_reason": "im_end",
            "parser_status": parser_status if image_id == 1 else "accepted",
        }
        for image_id in range(1, 14)
    )

    assert (
        backend_owner.Human13RPCrossoverProductionBackend._burdens(outputs)[
            "unparseable"
        ]
        == 1
    )

    unknown = ({**outputs[0], "parser_status": "unknown"}, *outputs[1:])
    with pytest.raises(backend_owner.ProductionBackendError, match="parser status"):
        backend_owner.Human13RPCrossoverProductionBackend._burdens(unknown)


def test_audit_delta_counts_only_new_per_image_structural_burdens() -> None:
    source = tuple(
        {
            "image_id": image_id,
            "malformed_row_count": int(image_id == 1),
            "stop_reason": "length" if image_id == 3 else "im_end",
            "parser_status": ("all_spans_dropped" if image_id == 5 else "accepted"),
        }
        for image_id in range(1, 14)
    )
    proposal = tuple(
        {
            **item,
            "malformed_row_count": (
                0 if item["image_id"] == 1 else int(item["image_id"] == 2)
            ),
            "stop_reason": (
                "im_end"
                if item["image_id"] == 3
                else "length"
                if item["image_id"] == 4
                else item["stop_reason"]
            ),
            "parser_status": (
                "accepted"
                if item["image_id"] == 5
                else (
                    "all_spans_dropped"
                    if item["image_id"] == 6
                    else item["parser_status"]
                )
            ),
        }
        for item in source
    )

    assert backend_owner.Human13RPCrossoverProductionBackend._burden_delta(
        proposal, source
    ) == {
        "malformed": 1,
        "cap_terminated": 1,
        "unparseable": 1,
    }


def test_cell_plan_projects_each_matrix_arm_without_changing_the_sealed_ray() -> None:
    leaf = next(
        item
        for item in launcher.load_leaf_configs()
        if item.training_rp == 1.0 and item.arm_id == "C"
    )
    frozen = SimpleNamespace(
        c_leaf_path=leaf.source_path,
        qualification_learning_rate_ray=(3.0e-7, 1.0e-6, 3.0e-6, 1.0e-5, 3.0e-5),
    )

    plan = backend_owner.Human13RPCrossoverProductionBackend._qualification_plan(
        frozen, learning_rate=1.0e-6, arm_id="A"
    )

    assert plan.arm_id == "A"
    assert plan.learning_rate == 1.0e-6
    assert plan.learning_rate_resolution == "provisional_qualification"
