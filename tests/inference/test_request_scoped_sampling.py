from __future__ import annotations

import copy
import json
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from src.common.errors import RuntimeContractError


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 9

    def convert_tokens_to_ids(self, token: str) -> int:
        assert token == "<|im_end|>"
        return self.eos_token_id

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = False) -> str:
        assert skip_special_tokens is False
        return "".join(
            "<|im_end|>" if token_id == 9 else str(token_id) for token_id in token_ids
        )


class TinyGenerateModel:
    """Generation fixture that exercises the public custom_generate callable."""

    def __init__(self) -> None:
        self.config = SimpleNamespace(
            is_encoder_decoder=False,
            _attn_implementation="eager",
            model_type="qwen3_vl",
        )
        self.training = False
        self.runtime_weight = torch.nn.Parameter(torch.zeros(()))
        self.seen_generators: tuple[torch.Generator, ...] | None = None
        self.seen_generation_config: Any = None

    def parameters(self) -> Any:
        return iter([self.runtime_weight])

    def generate(self, **kwargs: Any) -> SimpleNamespace:
        self.seen_generators = tuple(kwargs.pop("request_generators"))
        sampling_timing = kwargs.pop("sampling_timing")
        self.seen_generation_config = kwargs["generation_config"]
        input_ids = kwargs["input_ids"]
        logits = torch.tensor(
            [
                -100.0,
                0.0,
                0.0,
                -100.0,
                -100.0,
                -100.0,
                -100.0,
                -100.0,
                -100.0,
                -100.0,
            ],
            device=input_ids.device,
        ).repeat(input_ids.shape[0], 1)
        from src.inference.backend import request_scoped_categorical_draw

        sampled = request_scoped_categorical_draw(
            logits,
            request_generators=self.seen_generators,
            sampling_timing=sampling_timing,
        )
        return SimpleNamespace(
            sequences=torch.cat([input_ids, sampled[:, None]], dim=1),
            scores=(logits,),
        )

    def compute_transition_scores(
        self,
        sequences: torch.Tensor,
        scores: tuple[torch.Tensor, ...],
        *,
        normalize_logits: bool,
    ) -> torch.Tensor:
        assert normalize_logits is True
        generated = sequences[:, -1:]
        return torch.log_softmax(scores[0], dim=-1).gather(1, generated)


class MutableRuntimeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(2))
        self.lora_A = torch.nn.Parameter(torch.zeros(2))
        self.shared_embed_delta = torch.nn.Parameter(torch.zeros(2))
        self.register_buffer("runtime_buffer", torch.zeros(1))
        self.config = SimpleNamespace(
            is_encoder_decoder=False,
            _attn_implementation="eager",
            model_type="qwen3_vl",
        )
        self.active_adapter = "default"
        self.peft_config: dict[str, Any] = {}
        self.eval()


class MutableFastTokenizerBackend:
    def __init__(self) -> None:
        self.normalizer = "fixture-normalizer"

    def to_str(self) -> str:
        return f'{{"normalizer":"{self.normalizer}"}}'


class MutableTokenizer(TinyTokenizer):
    bos_token_id = 1
    chat_template = "fixture-template"
    clean_up_tokenization_spaces = True
    padding_side = "left"
    truncation_side = "right"
    model_max_length = 4096
    is_fast = True
    vocab_size = 4
    special_tokens_map = {"eos_token": "<|im_end|>"}
    all_special_ids = [0, 1, 9]

    def __init__(self) -> None:
        self.backend_tokenizer = MutableFastTokenizerBackend()

    def __len__(self) -> int:
        return 4

    def get_vocab(self) -> dict[str, int]:
        return {"a": 1, "b": 2, "<|im_end|>": self.eos_token_id}

    def get_added_vocab(self) -> dict[str, int]:
        return {"<|im_end|>": self.eos_token_id}


def _sampled_policy(*, temperature: float = 0.7, top_p: float = 0.9) -> Any:
    from src.inference.backend import DecodeGenerationPolicy

    return DecodeGenerationPolicy.sampled(
        max_new_tokens=8,
        repetition_penalty=1.0,
        temperature=temperature,
        top_p=top_p,
    )


def _greedy_policy() -> Any:
    from src.inference.backend import DecodeGenerationPolicy

    return DecodeGenerationPolicy.greedy(max_new_tokens=8, repetition_penalty=1.0)


def _request(request_id: str, *, policy: Any, seed: int | None) -> Any:
    from src.inference.backend import DecodeRequest

    return DecodeRequest(
        request_id=request_id,
        prompt_token_ids=[4, 5],
        model_inputs={"input_ids": torch.tensor([4, 5])},
        generation_policy=policy,
        sampling_seed=seed,
    )


def _run_unit_sampled_backend(
    backend: Any,
    requests: list[Any],
    *,
    model_identity: dict[str, Any] | None = None,
    tokenizer_identity: dict[str, Any] | None = None,
    generation_config_fingerprint: str = "authored-config-fingerprint",
    case_name: str | None = None,
) -> list[Any]:
    """Exercise the private mechanism seam without creating a public bypass."""

    return backend._generate_batch(
        requests,
        model_identity=model_identity or {"family": "tiny-model"},
        tokenizer_identity=tokenizer_identity or {"sha256": "tiny-tokenizer"},
        generation_config_fingerprint=generation_config_fingerprint,
        execution_context="sampling_attestation",
        sampling_attestation_case_name=case_name,
    )


def _new_bound_rebind_backend() -> tuple[Any, dict[str, Any], dict[str, Any], str]:
    from src.inference.backend import HFGenerateBackend

    model_identity = {"family": "portable-cpu-fixture"}
    tokenizer_identity = {"identity": "mutable-tokenizer-fixture"}
    generation_config_fingerprint = "portable-generation-config-fixture"
    backend = HFGenerateBackend(
        model=MutableRuntimeModel(),
        tokenizer=MutableTokenizer(),
        model_identity=model_identity,
        tokenizer_identity=tokenizer_identity,
        generation_config_fingerprint=generation_config_fingerprint,
    )
    return (
        backend,
        model_identity,
        tokenizer_identity,
        generation_config_fingerprint,
    )


def _payload_model_identity(root: str) -> dict[str, Any]:
    return {
        "base_model": {
            "model_type": "qwen3_vl",
            "model_path": "/models/canonical-qwen3-vl",
            "revision": "immutable-base-revision",
        },
        "adapter": {
            "adapter_path": f"{root}/adapter",
            "adapter_type": "Weight-Decomposed Low-Rank Adaptation",
            "adapter_payload_evidence": {
                "config_path": f"{root}/adapter/adapter_config.json",
                "config_sha256": "a" * 64,
                "tensor_path": f"{root}/adapter/adapter_model.safetensors",
                "tensor_sha256": "b" * 64,
            },
        },
        "embedding_delta": {
            "identity": {
                "delta_path": f"{root}/embedding/shared_embed_delta.safetensors",
                "delta_sha256": "c" * 64,
                "metadata_path": f"{root}/embedding/metadata.json",
                "metadata_sha256": "d" * 64,
            },
            "load": {
                "metadata_path": f"{root}/embedding/metadata.json",
                "metadata_sha256": "d" * 64,
                "tensor_path": f"{root}/embedding/shared_embed_delta.safetensors",
                "tensor_sha256": "c" * 64,
            },
        },
    }


def _case_with_mutated_result(
    case: Any,
    request_id: str,
    mutate_result: Any,
) -> Any:
    """Re-seal one untrusted executed-case fixture after a controlled mutation."""

    from src.inference.backend import (
        DecodeResult,
        ExecutedSampledRuntimeAttestationCase,
        _sha256_json,
    )

    payload = case.to_artifact_dict()
    mutated_results = []
    for artifact in payload["result_artifacts"]:
        result = DecodeResult.from_artifact_dict(artifact)
        if result.request_id == request_id:
            result = mutate_result(result)
        mutated_results.append(result.to_artifact_dict())
    payload["result_artifacts"] = mutated_results
    payload.pop("case_payload_fingerprint")
    payload["case_payload_fingerprint"] = _sha256_json(payload)
    return ExecutedSampledRuntimeAttestationCase.from_artifact_dict(payload)


def _first_cpu_attestation_bundle(fixture: dict[str, Any]) -> Any:
    from src.inference.backend import SampledRuntimeAttestationBundle

    return SampledRuntimeAttestationBundle.from_artifact_dict(
        fixture["artifact"]["policy_attestations"][0]["attestation_bundle"]
    )


def test_within_cardinality_order_drift_is_rejected_before_bundle_construction(
    cpu_three_policy_attestation_aggregate: dict[str, Any],
) -> None:
    from src.inference.backend import build_sampled_runtime_attestation_bundle

    bundle = _first_cpu_attestation_bundle(cpu_three_policy_attestation_aggregate)
    four_reversed = next(
        case for case in bundle.executed_cases if case.case_name == "batch_size_four_reversed"
    )

    def mutate(result: Any) -> Any:
        traces = list(result.token_trace)
        index = next(index for index, trace in enumerate(traces) if not trace.is_pad)
        traces[index] = replace(traces[index], token_id=traces[index].token_id + 1)
        return replace(result, token_trace=traces)

    mutated = _case_with_mutated_result(
        four_reversed,
        four_reversed.request_ids[0],
        mutate,
    )
    cases = tuple(
        mutated if case.case_name == mutated.case_name else case
        for case in bundle.executed_cases
    )
    with pytest.raises(RuntimeContractError) as exc_info:
        build_sampled_runtime_attestation_bundle(
            lineage=dict(bundle.lineage),
            executed_cases=cases,
            processed_logit_parity=bundle.processed_logit_parity,
        )
    assert exc_info.value.code == "backend_sampling.attestation_request_order_failed"


def test_cross_cardinality_shape_drift_is_diagnostic_and_json_safe(
    cpu_three_policy_attestation_aggregate: dict[str, Any],
) -> None:
    from src.inference.backend import build_sampled_runtime_attestation_bundle

    bundle = _first_cpu_attestation_bundle(cpu_three_policy_attestation_aggregate)
    three_forward = next(
        case for case in bundle.executed_cases if case.case_name == "batch_size_three_forward"
    )

    def drop_one_nonpad_trace(result: Any) -> Any:
        traces = list(result.token_trace)
        index = next(
            index for index in range(len(traces) - 1, -1, -1) if not traces[index].is_pad
        )
        traces.pop(index)
        return replace(result, token_trace=traces)

    mutated = _case_with_mutated_result(
        three_forward,
        three_forward.request_ids[0],
        drop_one_nonpad_trace,
    )
    three_reversed = next(
        case
        for case in bundle.executed_cases
        if case.case_name == "batch_size_three_reversed"
    )
    mutated_reversed = _case_with_mutated_result(
        three_reversed,
        three_forward.request_ids[0],
        drop_one_nonpad_trace,
    )
    cases = tuple(
        mutated
        if case.case_name == mutated.case_name
        else mutated_reversed
        if case.case_name == mutated_reversed.case_name
        else case
        for case in bundle.executed_cases
    )
    rebuilt = build_sampled_runtime_attestation_bundle(
        lineage=dict(bundle.lineage),
        executed_cases=cases,
        processed_logit_parity=bundle.processed_logit_parity,
    )
    cross = rebuilt.cross_cardinality
    assert cross.cross_cardinality_exact_generated_token_replay is False
    assert cross.cross_cardinality_maximum_absolute_score_difference is None
    assert cross.cross_cardinality_maximum_relative_score_difference is None
    assert cross.cross_cardinality_comparison_status == "shape_mismatch"
    assert "score_trace_shape_mismatch" in str(cross.cross_cardinality_mismatch_reason)
    json.dumps(rebuilt.to_artifact_dict(), allow_nan=False)


@pytest.fixture(scope="module")
def cpu_three_policy_attestation_aggregate() -> dict[str, Any]:
    """Build typed CPU evidence for aggregate/rebind contract tests only."""

    from scripts.research import attest_request_scoped_sampling_cuda as script
    from transformers import GPT2Config, GPT2LMHeadModel

    from src.inference.backend import (
        DecodeGenerationPolicy,
        DecodeRequest,
        HFGenerateBackend,
        _live_runtime_state_seal,
        _portable_runtime_state_seal_artifact,
        _thaw_json,
        build_sampled_runtime_attestation_bundle,
    )

    runtime_backend, model_identity, tokenizer_identity, generation_fingerprint = (
        _new_bound_rebind_backend()
    )
    portable_state = _portable_runtime_state_seal_artifact(
        _live_runtime_state_seal(
            runtime_backend.model,
            runtime_backend.tokenizer,
            hash_payloads=True,
        )
    )
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=10,
            n_positions=520,
            n_ctx=520,
            n_embd=8,
            n_layer=1,
            n_head=1,
            bos_token_id=1,
            eos_token_id=9,
            pad_token_id=0,
        )
    ).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        for module in model.modules():
            if isinstance(module, torch.nn.LayerNorm):
                module.weight.fill_(1.0)
        embedding = torch.tensor(
            [1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 0.75, -0.75]
        )
        for token_id in range(1, 6):
            model.transformer.wte.weight[token_id].copy_(embedding)
        model.transformer.wte.weight[9].copy_(10.0 * embedding)
    generation_backend = HFGenerateBackend(model=model, tokenizer=TinyTokenizer())
    entries = []
    for temperature in script.TEMPERATURES:
        policy = DecodeGenerationPolicy.sampled(
            max_new_tokens=512,
            repetition_penalty=1.0,
            temperature=temperature,
            top_p=0.95,
        )
        requests = [
            DecodeRequest(
                request_id=f"calibration-{index}",
                prompt_token_ids=[1, 2 + index],
                model_inputs={"input_ids": torch.tensor([1, 2 + index])},
                generation_policy=policy,
                sampling_seed=1000 + index,
            )
            for index in range(4)
        ]

        def execute(case_name: str, case_requests: list[Any]) -> Any:
            _run_unit_sampled_backend(
                generation_backend,
                case_requests,
                model_identity=model_identity,
                tokenizer_identity=tokenizer_identity,
                generation_config_fingerprint=generation_fingerprint,
                case_name=case_name,
            )
            assert generation_backend.last_sampling_attestation_case is not None
            return generation_backend.last_sampling_attestation_case

        four_forward = execute("batch_size_four_forward", requests)
        processed_logit_parity = (
            generation_backend._attest_processed_logit_parity_against_stock(
                requests
            )
        )
        four_reversed = execute(
            "batch_size_four_reversed", list(reversed(requests))
        )
        three_forward = execute("batch_size_three_forward", requests[:3])
        three_reversed = execute(
            "batch_size_three_reversed", list(reversed(requests[:3]))
        )
        bundle = build_sampled_runtime_attestation_bundle(
            lineage={
                "temperature": temperature,
                "qwen_runtime_identity": {"tokens": {"im_end_token_ids": [9]}},
            },
            executed_cases=(
                four_forward,
                four_reversed,
                three_forward,
                three_reversed,
            ),
            processed_logit_parity=processed_logit_parity,
        )
        replay_rows = []
        selected_replays = _thaw_json(
            four_forward.compact_object_selected_token_score_replays_by_request
        )
        for execution_index, artifact in enumerate(four_forward.result_artifacts):
            artifact_dict = _thaw_json(artifact)
            result = script.DecodeResult.from_artifact_dict(artifact_dict)
            receipt = result.execution_receipt
            assert receipt is not None
            artifact_fingerprint = script.sha256_json(artifact_dict)
            replay_rows.append(
                {
                    "request_id": result.request_id,
                    "execution_index": execution_index,
                    "sampling_seed": four_forward.sampling_seeds[execution_index],
                    "exact_result_artifact_replay": True,
                    "exact_selected_token_score_replay": True,
                    "receipt_binding_valid": True,
                    "attestation_result_artifact_fingerprint": artifact_fingerprint,
                    "admitted_result_artifact_fingerprint": artifact_fingerprint,
                    "receipt_fingerprint": receipt.receipt_fingerprint,
                    "generated_token_identifiers_hash": (
                        receipt.generated_token_identifiers_hash
                    ),
                    "canonical_float32_score_trace_hash": (
                        receipt.canonical_float32_score_trace_hash
                    ),
                    "compact_object_selected_token_score_replay": (
                        selected_replays[result.request_id]
                    ),
                }
            )
        admitted = {
            "schema_version": script.ADMITTED_PRODUCTION_REPLAY_SCHEMA_VERSION,
            "capability_gated_backend_api": (
                "HFGenerateBackend.generate_batch_with_verified_runtime_attestation"
            ),
            "capability_bundle_payload_fingerprint": (
                bundle.bundle_payload_fingerprint
            ),
            "decode_generation_policy_fingerprint": policy.fingerprint,
            "compared_attestation_case": "batch_size_four_forward",
            "request_ids": list(four_forward.request_ids),
            "exact_request_order_replay": True,
            "exact_result_artifact_replay": True,
            "exact_selected_token_score_replay": True,
            "runtime_state_seal_diagnostics": {
                "measurement_scope": "physical_backend_call_pre_generation",
                "adapter_and_selected_embedding_tensor_count": 2,
                "adapter_and_selected_embedding_payload_byte_count": 16,
                "adapter_and_selected_embedding_hash_elapsed_seconds": 0.0,
                "live_runtime_state_seal_elapsed_seconds": 0.0,
                "base_model_payload_hashed": False,
            },
            "portable_runtime_state_seal": portable_state,
            "result_replays": replay_rows,
        }
        admitted["replay_payload_fingerprint"] = script.sha256_json(admitted)
        entries.append(
            script._build_policy_attestation_entry(
                policy=policy,
                bundle=bundle,
                admitted_production_replay=admitted,
            )
        )
    return {
        "artifact": script._build_aggregate_attestation_output(entries),
        "model_identity": model_identity,
        "tokenizer_identity": tokenizer_identity,
        "generation_config_fingerprint": generation_fingerprint,
    }


def _refingerprint_policy_attestation_entry(entry: dict[str, Any]) -> None:
    from scripts.research import attest_request_scoped_sampling_cuda as script

    entry.pop("entry_payload_fingerprint", None)
    entry["entry_payload_fingerprint"] = script.sha256_json(entry)


def _refingerprint_attestation_aggregate(artifact: dict[str, Any]) -> None:
    from scripts.research import attest_request_scoped_sampling_cuda as script

    artifact.pop("aggregate_payload_fingerprint", None)
    artifact["aggregate_payload_fingerprint"] = script.sha256_json(artifact)


def _fake_full_bundle_validation(
    fixture: dict[str, Any],
) -> Any:
    from src.inference import backend as backend_module

    expected_runtime_identity = backend_module._runtime_identity()
    expected_device_identity = backend_module._execution_device_identity(
        torch.device("cpu")
    )

    def validate(bundle: Any) -> tuple[Any, dict[str, Any]]:
        four_forward = next(
            case
            for case in bundle.executed_cases
            if case.case_name == "batch_size_four_forward"
        )
        result = backend_module.DecodeResult.from_artifact_dict(
            four_forward.result_artifacts[0]
        )
        receipt = result.execution_receipt
        assert receipt is not None
        return bundle, {
            "bundle_payload_fingerprint": bundle.bundle_payload_fingerprint,
            "model_identity": fixture["model_identity"],
            "tokenizer_identity": fixture["tokenizer_identity"],
            "generation_config_fingerprint": fixture[
                "generation_config_fingerprint"
            ],
            "decode_generation_policy": dict(receipt.decode_generation_policy),
            "normalized_prepared_generation_profile": (
                backend_module._normalized_attested_generation_profile(
                    receipt.executed_generation_arguments,
                    prompt_width=len(result.prompt_token_ids),
                )
            ),
            "attention_implementation": "eager",
            "runtime_identity": expected_runtime_identity,
            "custom_sampler_code_hash": backend_module.custom_sampler_code_hash(),
            "execution_device_identity": expected_device_identity,
            "verified_checks": [],
        }

    return validate


def _use_cpu_fixture_payload_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.inference import backend as backend_module

    monkeypatch.setattr(
        backend_module,
        "EXPECTED_ATTESTED_RUNTIME_PAYLOAD_TENSOR_COUNT",
        2,
    )
    monkeypatch.setattr(
        backend_module,
        "EXPECTED_ATTESTED_RUNTIME_PAYLOAD_BYTE_COUNT",
        16,
    )


def test_generation_policy_is_immutable_tagged_and_canonical() -> None:
    policy = _sampled_policy()

    assert policy.mode == "sampled"
    assert policy.sampling_profile == "temperature_top_p_categorical_v1"
    assert policy.to_artifact_dict() == {
        "mode": "sampled",
        "sampling_profile": "temperature_top_p_categorical_v1",
        "max_new_tokens": 8,
        "repetition_penalty": 1.0,
        "temperature": 0.7,
        "top_p": 0.9,
    }
    assert len(policy.fingerprint) == 64
    assert policy.fingerprint == _sampled_policy().fingerprint
    with pytest.raises(FrozenInstanceError):
        policy.top_p = 0.5


@pytest.mark.parametrize(
    ("factory", "expected_code"),
    [
        (
            lambda: _sampled_policy(temperature=0.0),
            "backend_policy.invalid_temperature",
        ),
        (lambda: _sampled_policy(top_p=0.0), "backend_policy.invalid_top_p"),
    ],
)
def test_invalid_sampled_policy_fails(factory: Any, expected_code: str) -> None:
    with pytest.raises(RuntimeContractError) as exc_info:
        factory()
    assert exc_info.value.code == expected_code


def test_request_seed_contract_rejects_missing_sampled_and_present_greedy_seed() -> (
    None
):
    with pytest.raises(RuntimeContractError) as sampled_exc:
        _request("sampled", policy=_sampled_policy(), seed=None)
    assert sampled_exc.value.code == "backend_policy.sampled_seed_required"
    assert sampled_exc.value.context["request_id"] == "sampled"

    with pytest.raises(RuntimeContractError) as greedy_exc:
        _request("greedy", policy=_greedy_policy(), seed=1)
    assert greedy_exc.value.code == "backend_policy.greedy_seed_forbidden"


def test_decode_request_exposes_no_process_or_batch_owned_sampling_seed_api() -> None:
    from dataclasses import fields

    from src.inference.backend import DecodeRequest

    field_names = {field.name for field in fields(DecodeRequest)}
    assert "sampling_seed" in field_names
    assert not {
        "batch_sampling_seed",
        "global_sampling_seed",
        "process_sampling_seed",
    }.intersection(field_names)
    with pytest.raises(TypeError):
        DecodeRequest(
            request_id="forbidden-batch-seed",
            prompt_token_ids=[4, 5],
            model_inputs={"input_ids": torch.tensor([4, 5])},
            generation_policy=_sampled_policy(),
            sampling_seed=1,
            batch_sampling_seed=1,
        )


@pytest.mark.parametrize("request_id", ["", "   "])
def test_request_identity_must_be_nonempty(request_id: str) -> None:
    with pytest.raises(RuntimeContractError) as exc_info:
        _request(request_id, policy=_greedy_policy(), seed=None)
    assert exc_info.value.code == "backend_policy.invalid_request_id"


def test_batch_validation_accepts_four_distinct_request_seeds_and_rejects_duplicates() -> (
    None
):
    from src.inference.backend import validate_decode_batch

    policy = _sampled_policy()
    requests = [
        _request(f"row-{index}", policy=policy, seed=100 + index) for index in range(4)
    ]
    validate_decode_batch(requests)

    with pytest.raises(RuntimeContractError) as exc_info:
        validate_decode_batch([requests[0], requests[0]])
    assert exc_info.value.code == "backend_policy.duplicate_request_id"


def test_batch_validation_rejects_incompatible_policy_fields_before_generation() -> (
    None
):
    from src.inference.backend import validate_decode_batch

    with pytest.raises(RuntimeContractError) as exc_info:
        validate_decode_batch(
            [
                _request("row-a", policy=_sampled_policy(top_p=0.9), seed=1),
                _request("row-b", policy=_sampled_policy(top_p=0.8), seed=2),
            ]
        )
    assert exc_info.value.code == "backend_policy.incompatible_batch"
    assert exc_info.value.context["differing_fields"] == ["top_p"]


def test_sanitized_sampled_profile_disables_inherited_warpers() -> None:
    from src.inference.backend import effective_generation_arguments

    arguments = effective_generation_arguments(
        _sampled_policy(),
        eos_token_id=9,
        pad_token_id=0,
    )
    assert arguments["do_sample"] is True
    assert arguments["temperature"] == pytest.approx(0.7)
    assert arguments["top_p"] == pytest.approx(0.9)
    assert arguments["top_k"] == 0
    assert arguments["typical_p"] == pytest.approx(1.0)
    assert arguments["min_p"] is None
    assert arguments["epsilon_cutoff"] == pytest.approx(0.0)
    assert arguments["eta_cutoff"] == pytest.approx(0.0)
    assert arguments["num_beams"] == 1
    assert arguments["num_return_sequences"] == 1
    assert arguments["constraints"] is None
    assert arguments["force_words_ids"] is None
    assert arguments["watermarking_config"] is None
    assert arguments["eos_token_id"] == 9
    assert arguments["pad_token_id"] == 0


def test_request_scoped_categorical_draw_binds_each_row_to_its_generator() -> None:
    from src.inference.backend import request_scoped_categorical_draw

    scores = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    generators = tuple(torch.Generator().manual_seed(seed) for seed in (11, 29))
    expected = torch.stack(
        [
            torch.multinomial(
                torch.softmax(scores[row], dim=-1),
                num_samples=1,
                generator=torch.Generator().manual_seed(seed),
            ).squeeze(0)
            for row, seed in enumerate((11, 29))
        ]
    )

    observed = request_scoped_categorical_draw(scores, request_generators=generators)
    assert torch.equal(observed, expected)


def test_four_row_sampled_backend_receipts_bind_generators_policy_order_and_outputs() -> (
    None
):
    from src.inference.backend import HFGenerateBackend

    policy = _sampled_policy()
    requests = [
        _request(f"row-{index}", policy=policy, seed=100 + index) for index in range(4)
    ]
    model = TinyGenerateModel()
    backend = HFGenerateBackend(model=model, tokenizer=TinyTokenizer())

    results = _run_unit_sampled_backend(
        backend,
        requests,
    )

    assert model.seen_generators is not None
    assert [generator.initial_seed() for generator in model.seen_generators] == [
        100,
        101,
        102,
        103,
    ]
    assert (
        len(
            {
                result.execution_receipt.batch_request_order_fingerprint
                for result in results
            }
        )
        == 1
    )
    for index, result in enumerate(results):
        receipt = result.execution_receipt
        assert receipt.request_id == f"row-{index}"
        assert receipt.sampling_seed == 100 + index
        assert receipt.random_generator_kind == "torch.Generator"
        assert receipt.random_generator_device == "cpu"
        assert receipt.random_generator_initial_seed == 100 + index
        assert receipt.request_execution_index == index
        assert receipt.decode_generation_policy_fingerprint == policy.fingerprint
        assert (
            receipt.custom_sampler_identity
            == "coordexp_hf_request_scoped_categorical_v1"
        )
        assert len(receipt.custom_sampler_code_hash) == 64
        assert receipt.attention_implementation == "eager"
        assert receipt.model_identity_fingerprint
        assert receipt.tokenizer_identity_fingerprint
        assert receipt.installed_runtime_identity_fingerprint
        assert receipt.effective_generation_profile_fingerprint
        assert receipt.canonical_float32_score_trace_hash
        assert receipt.sampling_profile_fingerprint
        assert receipt.score_trace_count == len(result.token_trace)
        assert receipt.executed_generation_arguments["top_k"] == 0
        assert receipt.generated_token_identifiers_hash
        result.validate_for_scored()
    diagnostics = backend.last_sampling_attestation_diagnostics
    assert diagnostics is not None
    assert diagnostics["measurement_scope"] == "synchronized_sampling_attestation"
    assert diagnostics["batch_size"] == 4
    assert diagnostics["score_steps"] == 1
    assert diagnostics["categorical_draw_call_count"] == 4
    assert diagnostics["full_batch_elapsed_seconds"] >= 0.0


def test_receipt_canonical_serialization_replay_and_swap_rejection() -> None:
    from dataclasses import replace

    from src.inference.backend import HFGenerateBackend

    requests = [
        _request(f"row-{index}", policy=_sampled_policy(), seed=200 + index)
        for index in range(4)
    ]
    backend = HFGenerateBackend(
        model=TinyGenerateModel(), tokenizer=TinyTokenizer()
    )
    results = _run_unit_sampled_backend(
        backend,
        requests,
    )
    artifact = results[0].execution_receipt.to_artifact_dict()
    replay = results[0].execution_receipt.from_artifact_dict(artifact)
    assert replay == results[0].execution_receipt
    assert replay.receipt_fingerprint == artifact["receipt_fingerprint"]

    swapped = replace(results[0], execution_receipt=results[1].execution_receipt)
    with pytest.raises(RuntimeContractError) as exc_info:
        swapped.validate_for_scored()
    assert exc_info.value.code == "backend_receipt.binding_mismatch"


def test_batch_validator_rejects_recomputed_forged_execution_contract() -> None:
    from dataclasses import replace

    from src.inference.backend import (
        DecodeExecutionReceipt,
        HFGenerateBackend,
        _sha256_json,
        validate_decode_execution_batch,
    )

    policy = _sampled_policy()
    requests = [
        _request(f"row-{index}", policy=policy, seed=250 + index) for index in range(4)
    ]
    model_identity = {"family": "tiny-model"}
    tokenizer_identity = {"sha256": "tiny-tokenizer"}
    backend = HFGenerateBackend(
        model=TinyGenerateModel(), tokenizer=TinyTokenizer()
    )
    results = _run_unit_sampled_backend(
        backend,
        requests,
        model_identity=model_identity,
        tokenizer_identity=tokenizer_identity,
    )
    original_artifact = results[0].execution_receipt.to_artifact_dict()
    forged_artifact = dict(original_artifact)
    forged_policy = dict(forged_artifact["decode_generation_policy"])
    forged_policy["top_p"] = 0.5
    forged_runtime = dict(forged_artifact["runtime_identity"])
    forged_runtime["torch_version"] = "forged"
    forged_arguments = dict(forged_artifact["executed_generation_arguments"])
    forged_arguments["top_k"] = 1
    forged_artifact.update(
        {
            "decode_generation_policy": forged_policy,
            "decode_generation_policy_fingerprint": _sha256_json(forged_policy),
            "executed_generation_arguments": forged_arguments,
            "effective_generation_profile_fingerprint": _sha256_json(forged_arguments),
            "custom_sampler_identity": "forged-sampler",
            "custom_sampler_code_hash": "0" * 64,
            "attention_implementation": "forged-attention",
            "runtime_identity": forged_runtime,
            "installed_runtime_identity_fingerprint": _sha256_json(forged_runtime),
            "model_eval_mode": False,
        }
    )
    fingerprint_payload = dict(forged_artifact)
    fingerprint_payload.pop("receipt_fingerprint")
    forged_artifact["receipt_fingerprint"] = _sha256_json(fingerprint_payload)
    forged_receipt = DecodeExecutionReceipt.from_artifact_dict(forged_artifact)
    receipt_only_substitution = replace(results[0], execution_receipt=forged_receipt)
    with pytest.raises(RuntimeContractError) as result_exc_info:
        receipt_only_substitution.validate_for_scored()
    assert result_exc_info.value.code == "backend_receipt.binding_mismatch"
    assert {
        "decode_generation_policy_fingerprint",
        "executed_generation_arguments",
        "custom_sampler_identity",
        "custom_sampler_code_hash",
        "attention_implementation",
        "runtime_identity",
        "model_eval_mode",
    }.issubset(result_exc_info.value.context["mismatches"])

    forged_result_and_anchor = replace(
        results[0],
        execution_receipt=forged_receipt,
        execution_contract_anchor=None,
    )
    forged_result_and_anchor.validate_for_scored()
    forged_results = [forged_result_and_anchor, *results[1:]]

    original_receipt = results[0].execution_receipt
    with pytest.raises(RuntimeContractError) as exc_info:
        validate_decode_execution_batch(
            requests,
            forged_results,
            model_identity=model_identity,
            tokenizer_identity=tokenizer_identity,
            generation_config_fingerprint="authored-config-fingerprint",
            executed_generation_arguments=(
                original_receipt.to_artifact_dict()["executed_generation_arguments"]
            ),
            backend="hf",
            backend_mode="generate",
            response_family="hf",
            attention_implementation="eager",
            model_eval_mode=True,
            runtime_identity=original_receipt.to_artifact_dict()["runtime_identity"],
            request_generators=tuple(
                torch.Generator().manual_seed(250 + index) for index in range(4)
            ),
        )
    assert exc_info.value.code == "backend_receipt.batch_contract_mismatch"
    mismatches = exc_info.value.context["receipt_mismatches"]
    assert {
        "decode_generation_policy_fingerprint",
        "executed_generation_arguments",
        "custom_sampler_identity",
        "custom_sampler_code_hash",
        "attention_implementation",
        "runtime_identity",
        "model_eval_mode",
    }.issubset(mismatches)


@pytest.mark.parametrize(
    ("identity_field", "active_identity", "expected_mismatch"),
    [
        ("model", {"family": "other-model"}, "model_identity_fingerprint"),
        ("tokenizer", {"sha256": "other-tokenizer"}, "tokenizer_identity_fingerprint"),
    ],
)
def test_batch_validator_rejects_model_or_tokenizer_identity_swap(
    identity_field: str,
    active_identity: dict[str, Any],
    expected_mismatch: str,
) -> None:
    from src.inference.backend import HFGenerateBackend, validate_decode_execution_batch

    requests = [
        _request(f"row-{index}", policy=_sampled_policy(), seed=270 + index)
        for index in range(4)
    ]
    model_identity = {"family": "tiny-model"}
    tokenizer_identity = {"sha256": "tiny-tokenizer"}
    results = _run_unit_sampled_backend(
        HFGenerateBackend(model=TinyGenerateModel(), tokenizer=TinyTokenizer()),
        requests,
        model_identity=model_identity,
        tokenizer_identity=tokenizer_identity,
    )
    receipt = results[0].execution_receipt
    with pytest.raises(RuntimeContractError) as exc_info:
        validate_decode_execution_batch(
            requests,
            results,
            model_identity=(
                active_identity if identity_field == "model" else model_identity
            ),
            tokenizer_identity=(
                active_identity if identity_field == "tokenizer" else tokenizer_identity
            ),
            generation_config_fingerprint="authored-config-fingerprint",
            executed_generation_arguments=dict(receipt.executed_generation_arguments),
            backend="hf",
            backend_mode="generate",
            response_family="hf",
            attention_implementation="eager",
            model_eval_mode=True,
            runtime_identity=dict(receipt.runtime_identity),
            request_generators=tuple(
                torch.Generator().manual_seed(270 + index) for index in range(4)
            ),
        )
    assert exc_info.value.code == "backend_receipt.batch_contract_mismatch"
    assert expected_mismatch in exc_info.value.context["receipt_mismatches"]


def test_swapped_generator_order_rejected_before_model_generation() -> None:
    from src.inference.backend import validate_request_generators

    requests = [
        _request(f"row-{index}", policy=_sampled_policy(), seed=300 + index)
        for index in range(4)
    ]
    generators = [torch.Generator().manual_seed(300 + index) for index in range(4)]
    generators[0], generators[1] = generators[1], generators[0]

    with pytest.raises(RuntimeContractError) as exc_info:
        validate_request_generators(requests, generators, device=torch.device("cpu"))
    assert exc_info.value.code == "backend_sampling.generator_seed_mismatch"
    assert exc_info.value.context["execution_index"] == 0


def test_public_sampled_backend_remains_gated_before_cuda_attestation() -> None:
    from src.inference.backend import HFGenerateBackend

    requests = [
        _request(f"row-{index}", policy=_sampled_policy(), seed=400 + index)
        for index in range(4)
    ]
    with pytest.raises(RuntimeContractError) as exc_info:
        HFGenerateBackend(
            model=TinyGenerateModel(), tokenizer=TinyTokenizer()
        ).generate_batch(
            requests,
            model_identity={"family": "tiny-model"},
            tokenizer_identity={"sha256": "tiny-tokenizer"},
            generation_config_fingerprint="authored-config-fingerprint",
        )
    assert exc_info.value.code == "backend_sampling.runtime_not_attested"
    assert (
        exc_info.value.context["required_bundle_schema_version"]
        == "typed_executed_cuda_qwen_attestation_bundle.v1"
    )
    assert set(exc_info.value.context["required_cases"]) == {
        "batch_size_three_forward",
        "batch_size_three_reversed",
        "batch_size_four_forward",
        "batch_size_four_reversed",
        "cross_cardinality_shared_three",
    }
    assert {
        "cuda_device_and_hardware_identity",
        "float32_score_and_transition_score_parity",
        "generator_device_seed_and_execution_index_binding",
        "processed_logit_parity_before_categorical_selection",
        "qwen_im_end_stop_reason",
        "sample_then_pad_semantics",
    }.issubset(exc_info.value.context["required_checks"])
    assert (
        exc_info.value.context["production_admission_api_status"]
        == "verified_capability_required"
    )


def test_callers_cannot_construct_verified_sampled_runtime_capability() -> None:
    from src.inference.backend import VerifiedSampledRuntimeAttestation

    with pytest.raises(RuntimeContractError) as exc_info:
        VerifiedSampledRuntimeAttestation(
            bundle_payload_fingerprint="0" * 64,
            admission_contract={},
            backend_object_id=1,
            model_object_id=2,
            tokenizer_object_id=3,
            _sentinel=object(),
        )
    assert (
        exc_info.value.code
        == "backend_sampling.attestation_capability_not_verified"
    )


def _live_state_validation_fixture() -> tuple[Any, Any, Any, dict[str, Any]]:
    from src.inference.backend import (
        HFGenerateBackend,
        VerifiedSampledRuntimeAttestation,
        _VERIFIED_SAMPLED_RUNTIME_ATTESTATION_SENTINEL,
        _execution_device_identity,
        _live_runtime_state_seal,
        _normalized_attested_generation_profile,
        _runtime_identity,
        custom_sampler_code_hash,
        effective_generation_arguments,
    )

    model = MutableRuntimeModel()
    tokenizer = MutableTokenizer()
    backend = HFGenerateBackend(model=model, tokenizer=tokenizer)
    request = _request("row-live-seal", policy=_sampled_policy(), seed=17)
    profile = effective_generation_arguments(
        request.generation_policy,
        eos_token_id=9,
        pad_token_id=0,
        bos_token_id=1,
    )
    profile["max_length"] = len(request.prompt_token_ids) + 8
    model_identity = {"family": "mutable-runtime"}
    tokenizer_identity = {"sha256": "mutable-tokenizer"}
    contract = {
        "model_identity": model_identity,
        "tokenizer_identity": tokenizer_identity,
        "generation_config_fingerprint": "mutable-config",
        "decode_generation_policy": request.generation_policy.to_artifact_dict(),
        "normalized_prepared_generation_profile": _normalized_attested_generation_profile(
            profile,
            prompt_width=len(request.prompt_token_ids),
        ),
        "attention_implementation": "eager",
        "runtime_identity": _runtime_identity(),
        "custom_sampler_code_hash": custom_sampler_code_hash(),
        "execution_device_identity": _execution_device_identity(torch.device("cpu")),
    }
    capability = VerifiedSampledRuntimeAttestation(
        bundle_payload_fingerprint="f" * 64,
        admission_contract=contract,
        backend_object_id=id(backend),
        model_object_id=id(model),
        tokenizer_object_id=id(tokenizer),
        runtime_state_seal=_live_runtime_state_seal(
            model, tokenizer, hash_payloads=True
        ),
        _sentinel=_VERIFIED_SAMPLED_RUNTIME_ATTESTATION_SENTINEL,
    )
    arguments = {
        "backend": backend,
        "requests": [request],
        "model_identity": model_identity,
        "tokenizer_identity": tokenizer_identity,
        "generation_config_fingerprint": "mutable-config",
        "attention_implementation": "eager",
        "model_evaluation_mode": True,
        "runtime_identity": _runtime_identity(),
        "execution_device_identity": _execution_device_identity(torch.device("cpu")),
        "prepared_generation_profile": profile,
        "prompt_width": len(request.prompt_token_ids),
    }
    return model, tokenizer, capability, arguments


def _relocated_payload_active_call_fixture() -> tuple[Any, dict[str, Any]]:
    _, _, original_capability, arguments = (
        _live_state_validation_fixture()
    )
    capability = _capability_with_attested_model_identity(
        original_capability,
        arguments,
        _payload_model_identity("/attested/root"),
    )
    arguments["model_identity"] = _payload_model_identity("/active/root")
    return capability, arguments


def _capability_with_attested_model_identity(
    original_capability: Any,
    arguments: dict[str, Any],
    model_identity: dict[str, Any],
) -> Any:
    from src.inference import backend as backend_module

    contract = backend_module._thaw_json(original_capability.admission_contract)
    contract["model_identity"] = model_identity
    capability = backend_module.VerifiedSampledRuntimeAttestation(
        bundle_payload_fingerprint=original_capability.bundle_payload_fingerprint,
        admission_contract=contract,
        backend_object_id=id(arguments["backend"]),
        model_object_id=id(arguments["backend"].model),
        tokenizer_object_id=id(arguments["backend"].tokenizer),
        runtime_state_seal=backend_module._thaw_json(
            original_capability.runtime_state_seal
        ),
        _sentinel=backend_module._VERIFIED_SAMPLED_RUNTIME_ATTESTATION_SENTINEL,
    )
    return capability


@pytest.mark.parametrize(
    "mutation",
    [
        "parameter_value",
        "buffer_value",
        "parameter_dtype",
        "selected_embedding_delta",
        "adapter_payload_data",
        "selected_embedding_payload_data",
        "active_adapter",
        "tokenizer_special_token",
        "tokenizer_cleanup",
        "tokenizer_fast_backend",
    ],
)
def test_verified_runtime_capability_fails_closed_after_live_state_mutation(
    mutation: str,
) -> None:
    from src.inference.backend import _validate_verified_sampled_runtime_for_active_call

    model, tokenizer, capability, arguments = _live_state_validation_fixture()
    if mutation == "parameter_value":
        with torch.no_grad():
            model.weight.add_(1)
    elif mutation == "buffer_value":
        model.runtime_buffer.add_(1)
    elif mutation == "parameter_dtype":
        model.weight = torch.nn.Parameter(model.weight.detach().to(torch.float64))
    elif mutation == "selected_embedding_delta":
        with torch.no_grad():
            model.shared_embed_delta.add_(1)
    elif mutation == "adapter_payload_data":
        version = model.lora_A._version
        model.lora_A.data.add_(1)
        assert model.lora_A._version == version
    elif mutation == "selected_embedding_payload_data":
        version = model.shared_embed_delta._version
        model.shared_embed_delta.data.add_(1)
        assert model.shared_embed_delta._version == version
    elif mutation == "active_adapter":
        model.active_adapter = "other"
    elif mutation == "tokenizer_special_token":
        tokenizer.eos_token_id = 8
    elif mutation == "tokenizer_cleanup":
        tokenizer.clean_up_tokenization_spaces = False
    else:
        tokenizer.backend_tokenizer.normalizer = "mutated-normalizer"

    with pytest.raises(RuntimeContractError) as exc_info:
        _validate_verified_sampled_runtime_for_active_call(
            capability,
            **arguments,
        )
    assert exc_info.value.code == "backend_sampling.attestation_live_state_changed"


def test_live_state_value_rehash_is_bounded_and_reports_call_overhead() -> None:
    from src.inference.backend import _validate_verified_sampled_runtime_for_active_call

    _, _, capability, arguments = _live_state_validation_fixture()
    backend = arguments["backend"]
    _validate_verified_sampled_runtime_for_active_call(capability, **arguments)

    diagnostics = backend.last_runtime_state_seal_diagnostics
    assert diagnostics is not None
    assert diagnostics["measurement_scope"] == "physical_backend_call_pre_generation"
    assert diagnostics["adapter_and_selected_embedding_tensor_count"] == 2
    assert diagnostics["adapter_and_selected_embedding_payload_byte_count"] == 16
    assert diagnostics["adapter_and_selected_embedding_hash_elapsed_seconds"] >= 0.0
    assert diagnostics["live_runtime_state_seal_elapsed_seconds"] >= 0.0
    assert diagnostics["base_model_payload_hashed"] is False


def test_active_call_accepts_only_payload_location_relocation() -> None:
    from src.inference.backend import _validate_verified_sampled_runtime_for_active_call

    capability, arguments = _relocated_payload_active_call_fixture()

    _validate_verified_sampled_runtime_for_active_call(capability, **arguments)


def test_active_call_rejects_non_location_payload_identity_change() -> None:
    from src.inference.backend import _validate_verified_sampled_runtime_for_active_call

    capability, arguments = _relocated_payload_active_call_fixture()
    arguments["model_identity"]["embedding_delta"]["identity"][
        "delta_sha256"
    ] = "f" * 64

    with pytest.raises(RuntimeContractError) as error:
        _validate_verified_sampled_runtime_for_active_call(capability, **arguments)

    assert error.value.code == "backend_sampling.attestation_active_runtime_mismatch"


@pytest.mark.parametrize(
    "invalid_path",
    ["missing", "relative", "null", "non_string", "malformed_family"],
)
def test_active_call_rejects_matching_invalid_payload_paths(
    invalid_path: str,
) -> None:
    from src.inference.backend import _validate_verified_sampled_runtime_for_active_call

    capability, arguments = _relocated_payload_active_call_fixture()
    attested_identity = _payload_model_identity("/same/root")
    active_identity = copy.deepcopy(attested_identity)
    attested_adapter = attested_identity["adapter"]
    active_adapter = active_identity["adapter"]
    if invalid_path == "malformed_family":
        attested_identity["adapter"] = 17
        active_identity["adapter"] = 17
    elif invalid_path == "missing":
        attested_adapter.pop("adapter_path")
        active_adapter.pop("adapter_path")
    elif invalid_path == "relative":
        attested_adapter["adapter_path"] = "relative/adapter"
        active_adapter["adapter_path"] = "relative/adapter"
    elif invalid_path == "null":
        attested_adapter["adapter_path"] = None
        active_adapter["adapter_path"] = None
    else:
        attested_adapter["adapter_path"] = 17
        active_adapter["adapter_path"] = 17
    capability = _capability_with_attested_model_identity(
        capability,
        arguments,
        attested_identity,
    )
    arguments["model_identity"] = active_identity

    with pytest.raises(RuntimeContractError) as error:
        _validate_verified_sampled_runtime_for_active_call(capability, **arguments)

    assert (
        error.value.code
        == "backend_sampling.attestation_model_payload_path_invalid"
    )


@pytest.mark.parametrize("payload_families", ["base_only", "adapter_only", "delta_only"])
def test_active_call_accepts_valid_optional_payload_families(
    payload_families: str,
) -> None:
    from src.inference.backend import _validate_verified_sampled_runtime_for_active_call

    capability, arguments = _relocated_payload_active_call_fixture()
    attested_identity = _payload_model_identity("/attested/root")
    active_identity = _payload_model_identity("/active/root")
    if payload_families == "base_only":
        attested_identity["adapter"] = None
        active_identity["adapter"] = None
        attested_identity["embedding_delta"] = None
        active_identity["embedding_delta"] = None
    elif payload_families == "adapter_only":
        attested_identity["embedding_delta"] = None
        active_identity["embedding_delta"] = None
    else:
        attested_identity["adapter"] = None
        active_identity["adapter"] = None
    capability = _capability_with_attested_model_identity(
        capability,
        arguments,
        attested_identity,
    )
    arguments["model_identity"] = active_identity

    _validate_verified_sampled_runtime_for_active_call(capability, **arguments)


def test_active_call_rejects_optional_payload_family_presence_mismatch() -> None:
    from src.inference.backend import _validate_verified_sampled_runtime_for_active_call

    capability, arguments = _relocated_payload_active_call_fixture()
    attested_identity = _payload_model_identity("/attested/root")
    attested_identity["adapter"] = None
    capability = _capability_with_attested_model_identity(
        capability,
        arguments,
        attested_identity,
    )

    with pytest.raises(RuntimeContractError) as error:
        _validate_verified_sampled_runtime_for_active_call(capability, **arguments)

    assert error.value.code == "backend_sampling.attestation_active_runtime_mismatch"


@pytest.mark.parametrize("mutation", ["base_model_path", "unrecognized_extra_path"])
def test_active_call_does_not_normalize_other_model_identity_paths(
    mutation: str,
) -> None:
    from src.inference.backend import _validate_verified_sampled_runtime_for_active_call

    capability, arguments = _relocated_payload_active_call_fixture()
    if mutation == "base_model_path":
        arguments["model_identity"]["base_model"]["model_path"] = (
            "/models/different-qwen3-vl"
        )
    else:
        attested_identity = _payload_model_identity("/attested/root")
        attested_identity["adapter"]["unrecognized_extra_path"] = (
            "/attested/root/extra"
        )
        capability = _capability_with_attested_model_identity(
            capability,
            arguments,
            attested_identity,
        )
        arguments["model_identity"]["adapter"]["unrecognized_extra_path"] = (
            "/active/root/extra"
        )

    with pytest.raises(RuntimeContractError) as error:
        _validate_verified_sampled_runtime_for_active_call(capability, **arguments)

    assert error.value.code == "backend_sampling.attestation_active_runtime_mismatch"


def test_cuda_attestation_script_executes_real_capability_gated_cpu_replay() -> None:
    from scripts.research import attest_request_scoped_sampling_cuda as script
    from transformers import GPT2Config, GPT2LMHeadModel

    from src.inference.backend import (
        HFGenerateBackend,
        VerifiedSampledRuntimeAttestation,
        _VERIFIED_SAMPLED_RUNTIME_ATTESTATION_SENTINEL,
        _execution_device_identity,
        _live_runtime_state_seal,
        _normalized_attested_generation_profile,
    )

    requests = [
        _request(f"row-{index}", policy=_sampled_policy(), seed=500 + index)
        for index in range(4)
    ]
    model_identity = {"family": "tiny-model"}
    tokenizer_identity = {"sha256": "tiny-tokenizer"}
    generation_config_fingerprint = "authored-config-fingerprint"
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=10,
            n_positions=16,
            n_ctx=16,
            n_embd=8,
            n_layer=1,
            n_head=1,
            bos_token_id=1,
            eos_token_id=9,
            pad_token_id=9,
        )
    ).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    model.active_adapter = None
    model.active_adapters = []
    backend = HFGenerateBackend(model=model, tokenizer=TinyTokenizer())
    attested_results = _run_unit_sampled_backend(
        backend,
        requests,
        model_identity=model_identity,
        tokenizer_identity=tokenizer_identity,
        generation_config_fingerprint=generation_config_fingerprint,
        case_name="batch_size_four_forward",
    )
    attestation_case = backend.last_sampling_attestation_case
    assert attestation_case is not None
    receipt = attested_results[0].execution_receipt
    contract = {
        "model_identity": model_identity,
        "tokenizer_identity": tokenizer_identity,
        "generation_config_fingerprint": generation_config_fingerprint,
        "decode_generation_policy": requests[0].generation_policy.to_artifact_dict(),
        "normalized_prepared_generation_profile": (
            _normalized_attested_generation_profile(
                receipt.executed_generation_arguments,
                prompt_width=len(requests[0].prompt_token_ids),
            )
        ),
        "attention_implementation": receipt.attention_implementation,
        "runtime_identity": dict(receipt.runtime_identity),
        "custom_sampler_code_hash": receipt.custom_sampler_code_hash,
        "execution_device_identity": _execution_device_identity(
            torch.device("cpu")
        ),
    }
    capability = VerifiedSampledRuntimeAttestation(
        bundle_payload_fingerprint="f" * 64,
        admission_contract=contract,
        backend_object_id=id(backend),
        model_object_id=id(backend.model),
        tokenizer_object_id=id(backend.tokenizer),
        runtime_state_seal=_live_runtime_state_seal(
            backend.model,
            backend.tokenizer,
            hash_payloads=True,
        ),
        _sentinel=_VERIFIED_SAMPLED_RUNTIME_ATTESTATION_SENTINEL,
    )
    evidence = script._execute_admitted_production_replay(
        backend=backend,
        requests=requests,
        bundle=SimpleNamespace(
            executed_cases=(attestation_case,),
            bundle_payload_fingerprint="f" * 64,
        ),
        capability=capability,
        model_identity=model_identity,
        tokenizer_identity=tokenizer_identity,
        generation_config_fingerprint=generation_config_fingerprint,
        image_sizes_by_request={request.request_id: (1000, 1000) for request in requests},
        expected_tensor_count=0,
        expected_payload_byte_count=0,
    )

    assert evidence["exact_request_order_replay"] is True
    assert evidence["exact_result_artifact_replay"] is True
    assert evidence["exact_selected_token_score_replay"] is True
    assert [row["request_id"] for row in evidence["result_replays"]] == [
        request.request_id for request in requests
    ]
    assert (
        evidence["runtime_state_seal_diagnostics"]["measurement_scope"]
        == "physical_backend_call_pre_generation"
    )


@pytest.mark.parametrize(
    "diagnostics",
    [
        None,
        {
            "measurement_scope": "capability_mint",
            "adapter_and_selected_embedding_tensor_count": 0,
            "adapter_and_selected_embedding_payload_byte_count": 0,
            "adapter_and_selected_embedding_hash_elapsed_seconds": 0.0,
            "live_runtime_state_seal_elapsed_seconds": 0.0,
            "base_model_payload_hashed": False,
        },
        {
            "measurement_scope": "physical_backend_call_pre_generation",
            "adapter_and_selected_embedding_tensor_count": 1,
            "adapter_and_selected_embedding_payload_byte_count": 0,
            "adapter_and_selected_embedding_hash_elapsed_seconds": 0.0,
            "live_runtime_state_seal_elapsed_seconds": 0.0,
            "base_model_payload_hashed": False,
        },
    ],
)
def test_cuda_attestation_script_fails_closed_on_missing_or_wrong_diagnostics(
    diagnostics: dict[str, Any] | None,
) -> None:
    from scripts.research import attest_request_scoped_sampling_cuda as script

    class BackendFixture:
        def __init__(self) -> None:
            self.last_runtime_state_seal_diagnostics = diagnostics
            self.admitted_call_count = 0

        def generate_batch_with_verified_runtime_attestation(
            self, requests: Any, **kwargs: Any
        ) -> list[Any]:
            del requests, kwargs
            self.admitted_call_count += 1
            return []

    backend = BackendFixture()
    with pytest.raises(RuntimeError, match="diagnostics"):
        script._execute_admitted_production_replay(
            backend=backend,
            requests=[],
            bundle=SimpleNamespace(
                executed_cases=(), bundle_payload_fingerprint="f" * 64
            ),
            capability=SimpleNamespace(bundle_payload_fingerprint="f" * 64),
            model_identity={},
            tokenizer_identity={},
            generation_config_fingerprint="fixture",
            image_sizes_by_request={},
            expected_tensor_count=0,
            expected_payload_byte_count=0,
        )
    assert backend.admitted_call_count == 1


def test_cuda_attestation_script_attests_all_policies_before_append_only_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    from scripts.research import attest_request_scoped_sampling_cuda as script

    events: list[str] = []

    def attest(temperature: float) -> dict[str, Any]:
        events.append(f"attested:{temperature}")
        return {"temperature": temperature}

    def write(path: Any, payload: Any) -> None:
        del path, payload
        assert events == ["attested:0.2", "attested:0.4", "attested:0.6"]
        events.append("write")

    monkeypatch.setattr(script, "_write_canonical", write)
    script._attest_all_policies_then_write_output(
        output=tmp_path / "attestation.json",
        policy_attestation_factory=attest,
    )
    assert events == ["attested:0.2", "attested:0.4", "attested:0.6", "write"]


def test_cuda_attestation_script_never_writes_when_admission_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    from scripts.research import attest_request_scoped_sampling_cuda as script

    writes: list[Any] = []

    def rejected(temperature: float) -> dict[str, Any]:
        if temperature == 0.4:
            raise RuntimeError("policy attestation rejected")
        return {"temperature": temperature}

    monkeypatch.setattr(
        script,
        "_write_canonical",
        lambda path, payload: writes.append((path, payload)),
    )
    with pytest.raises(RuntimeError, match="policy attestation rejected"):
        script._attest_all_policies_then_write_output(
            output=tmp_path / "attestation.json",
            policy_attestation_factory=rejected,
        )
    assert writes == []


def test_cuda_attestation_production_cli_has_no_single_temperature_mode() -> None:
    from scripts.research import attest_request_scoped_sampling_cuda as script

    parser = script._parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--output",
                "/tmp/attestation.json",
                "--temperature",
                "0.2",
            ]
        )


def test_three_policy_aggregate_rejects_bare_and_non_exact_policy_sets(
    monkeypatch: pytest.MonkeyPatch,
    cpu_three_policy_attestation_aggregate: dict[str, Any],
) -> None:
    from src.inference import backend as backend_module

    _use_cpu_fixture_payload_contract(monkeypatch)
    artifact = cpu_three_policy_attestation_aggregate["artifact"]
    bare_bundle = artifact["policy_attestations"][0]["attestation_bundle"]
    with pytest.raises(RuntimeContractError) as bare_error:
        backend_module._parse_sampled_runtime_attestation_aggregate_output(
            bare_bundle
        )
    assert bare_error.value.code == "backend_sampling.attestation_aggregate_required"

    invalid_artifacts = []
    missing = copy.deepcopy(artifact)
    missing["policy_attestations"].pop()
    _refingerprint_attestation_aggregate(missing)
    invalid_artifacts.append(missing)

    fourth = copy.deepcopy(artifact)
    fourth["policy_attestations"].append(
        copy.deepcopy(fourth["policy_attestations"][-1])
    )
    _refingerprint_attestation_aggregate(fourth)
    invalid_artifacts.append(fourth)

    duplicate = copy.deepcopy(artifact)
    duplicate["policy_attestations"][-1] = copy.deepcopy(
        duplicate["policy_attestations"][0]
    )
    _refingerprint_attestation_aggregate(duplicate)
    invalid_artifacts.append(duplicate)

    unexpected = copy.deepcopy(artifact)
    unexpected_entry = unexpected["policy_attestations"][-1]
    unexpected_entry["temperature"] = 0.8
    _refingerprint_policy_attestation_entry(unexpected_entry)
    _refingerprint_attestation_aggregate(unexpected)
    invalid_artifacts.append(unexpected)

    for invalid in invalid_artifacts:
        with pytest.raises(RuntimeContractError):
            backend_module._parse_sampled_runtime_attestation_aggregate_output(
                invalid
            )


def test_three_policy_aggregate_rejects_cross_policy_bundle_and_replay_swaps(
    monkeypatch: pytest.MonkeyPatch,
    cpu_three_policy_attestation_aggregate: dict[str, Any],
) -> None:
    from src.inference import backend as backend_module

    _use_cpu_fixture_payload_contract(monkeypatch)
    artifact = cpu_three_policy_attestation_aggregate["artifact"]

    bundle_only = copy.deepcopy(artifact)
    first, second = bundle_only["policy_attestations"][:2]
    first["attestation_bundle"], second["attestation_bundle"] = (
        second["attestation_bundle"],
        first["attestation_bundle"],
    )
    first["bundle_payload_fingerprint"] = first["attestation_bundle"][
        "bundle_payload_fingerprint"
    ]
    second["bundle_payload_fingerprint"] = second["attestation_bundle"][
        "bundle_payload_fingerprint"
    ]
    _refingerprint_policy_attestation_entry(first)
    _refingerprint_policy_attestation_entry(second)
    _refingerprint_attestation_aggregate(bundle_only)
    with pytest.raises(RuntimeContractError):
        backend_module._parse_sampled_runtime_attestation_aggregate_output(
            bundle_only
        )

    bundle_and_replay = copy.deepcopy(artifact)
    first, second = bundle_and_replay["policy_attestations"][:2]
    for field in (
        "attestation_bundle",
        "bundle_payload_fingerprint",
        "admitted_production_replay",
    ):
        first[field], second[field] = second[field], first[field]
    _refingerprint_policy_attestation_entry(first)
    _refingerprint_policy_attestation_entry(second)
    _refingerprint_attestation_aggregate(bundle_and_replay)
    with pytest.raises(RuntimeContractError):
        backend_module._parse_sampled_runtime_attestation_aggregate_output(
            bundle_and_replay
        )


def test_three_policy_aggregate_rejects_output_bundle_replay_and_seal_tampering(
    monkeypatch: pytest.MonkeyPatch,
    cpu_three_policy_attestation_aggregate: dict[str, Any],
) -> None:
    from scripts.research import attest_request_scoped_sampling_cuda as script
    from src.inference import backend as backend_module

    _use_cpu_fixture_payload_contract(monkeypatch)
    artifact = cpu_three_policy_attestation_aggregate["artifact"]

    outer = copy.deepcopy(artifact)
    outer["policy_attestations"][0]["temperature"] = 0.25
    with pytest.raises(RuntimeContractError):
        backend_module._parse_sampled_runtime_attestation_aggregate_output(outer)

    bundle = copy.deepcopy(artifact)
    bundle["policy_attestations"][0]["attestation_bundle"]["lineage"][
        "temperature"
    ] = 0.4
    _refingerprint_policy_attestation_entry(bundle["policy_attestations"][0])
    _refingerprint_attestation_aggregate(bundle)
    with pytest.raises(RuntimeContractError):
        backend_module._parse_sampled_runtime_attestation_aggregate_output(bundle)

    replay = copy.deepcopy(artifact)
    replay_entry = replay["policy_attestations"][0]
    replay_payload = replay_entry["admitted_production_replay"]
    replay_payload["capability_gated_backend_api"] = "unverified.generate"
    replay_payload.pop("replay_payload_fingerprint")
    replay_payload["replay_payload_fingerprint"] = script.sha256_json(replay_payload)
    _refingerprint_policy_attestation_entry(replay_entry)
    _refingerprint_attestation_aggregate(replay)
    with pytest.raises(RuntimeContractError):
        backend_module._parse_sampled_runtime_attestation_aggregate_output(replay)

    portable = copy.deepcopy(artifact)
    portable_entry = portable["policy_attestations"][0]
    portable_replay = portable_entry["admitted_production_replay"]
    portable_replay["portable_runtime_state_seal"][
        "model_tensor_structure_identity"
    ]["structure_fingerprint"] = "not-a-digest"
    portable_replay.pop("replay_payload_fingerprint")
    portable_replay["replay_payload_fingerprint"] = script.sha256_json(
        portable_replay
    )
    _refingerprint_policy_attestation_entry(portable_entry)
    _refingerprint_attestation_aggregate(portable)
    with pytest.raises(RuntimeContractError):
        backend_module._parse_sampled_runtime_attestation_aggregate_output(portable)


def test_persisted_aggregate_rebinds_equivalent_runtime_and_validates_all_policies(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    cpu_three_policy_attestation_aggregate: dict[str, Any],
) -> None:
    from scripts.research import attest_request_scoped_sampling_cuda as script
    from src.inference import backend as backend_module

    _use_cpu_fixture_payload_contract(monkeypatch)
    fixture = cpu_three_policy_attestation_aggregate
    output = tmp_path / "three-policy-attestation.json"
    output.write_text(
        script.json.dumps(fixture["artifact"], sort_keys=True),
        encoding="utf-8",
    )
    validations: list[float] = []
    full_validate = _fake_full_bundle_validation(fixture)

    def counted(bundle: Any) -> Any:
        validations.append(float(bundle.lineage["temperature"]))
        return full_validate(bundle)

    monkeypatch.setattr(
        backend_module,
        "_validate_sampled_runtime_attestation_bundle",
        counted,
    )
    backend, _, _, _ = _new_bound_rebind_backend()
    selected_fingerprint = fixture["artifact"]["policy_attestations"][1][
        "decode_generation_policy_fingerprint"
    ]
    capability = backend_module.load_and_rebind_sampled_runtime_attestation_aggregate(
        output,
        decode_generation_policy_fingerprint=selected_fingerprint,
        backend=backend,
    )
    assert validations == [0.2, 0.4, 0.6]
    assert capability.is_bound_to(backend)
    assert capability.bundle_payload_fingerprint == fixture["artifact"][
        "policy_attestations"
    ][1]["bundle_payload_fingerprint"]

    with pytest.raises(RuntimeContractError) as missing_policy:
        backend_module.load_and_rebind_sampled_runtime_attestation_aggregate(
            output,
            decode_generation_policy_fingerprint="f" * 64,
            backend=backend,
        )
    assert (
        missing_policy.value.code
        == "backend_sampling.attestation_rebind_policy_not_found"
    )


def test_persisted_rebind_accepts_only_payload_location_relocation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    cpu_three_policy_attestation_aggregate: dict[str, Any],
) -> None:
    from scripts.research import attest_request_scoped_sampling_cuda as script
    from src.inference import backend as backend_module

    _use_cpu_fixture_payload_contract(monkeypatch)
    fixture = copy.deepcopy(cpu_three_policy_attestation_aggregate)
    fixture["model_identity"] = _payload_model_identity("/attested/root")
    output = tmp_path / "relocated-payload.json"
    output.write_text(script.json.dumps(fixture["artifact"]), encoding="utf-8")
    monkeypatch.setattr(
        backend_module,
        "_validate_sampled_runtime_attestation_bundle",
        _fake_full_bundle_validation(fixture),
    )
    active_model_identity = _payload_model_identity("/active/root")
    backend = backend_module.HFGenerateBackend(
        model=MutableRuntimeModel(),
        tokenizer=MutableTokenizer(),
        model_identity=active_model_identity,
        tokenizer_identity=fixture["tokenizer_identity"],
        generation_config_fingerprint=fixture["generation_config_fingerprint"],
    )

    capability = backend_module.load_and_rebind_sampled_runtime_attestation_aggregate(
        output,
        decode_generation_policy_fingerprint=fixture["artifact"][
            "policy_attestations"
        ][0]["decode_generation_policy_fingerprint"],
        backend=backend,
    )

    assert capability.is_bound_to(backend)


def test_persisted_rebind_rejects_non_location_payload_identity_change(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    cpu_three_policy_attestation_aggregate: dict[str, Any],
) -> None:
    from scripts.research import attest_request_scoped_sampling_cuda as script
    from src.inference import backend as backend_module

    _use_cpu_fixture_payload_contract(monkeypatch)
    fixture = copy.deepcopy(cpu_three_policy_attestation_aggregate)
    fixture["model_identity"] = _payload_model_identity("/attested/root")
    output = tmp_path / "changed-payload.json"
    output.write_text(script.json.dumps(fixture["artifact"]), encoding="utf-8")
    monkeypatch.setattr(
        backend_module,
        "_validate_sampled_runtime_attestation_bundle",
        _fake_full_bundle_validation(fixture),
    )
    active_model_identity = _payload_model_identity("/active/root")
    active_model_identity["adapter"]["adapter_payload_evidence"][
        "tensor_sha256"
    ] = "e" * 64
    backend = backend_module.HFGenerateBackend(
        model=MutableRuntimeModel(),
        tokenizer=MutableTokenizer(),
        model_identity=active_model_identity,
        tokenizer_identity=fixture["tokenizer_identity"],
        generation_config_fingerprint=fixture["generation_config_fingerprint"],
    )

    with pytest.raises(RuntimeContractError) as error:
        backend_module.load_and_rebind_sampled_runtime_attestation_aggregate(
            output,
            decode_generation_policy_fingerprint=fixture["artifact"][
                "policy_attestations"
            ][0]["decode_generation_policy_fingerprint"],
            backend=backend,
        )

    assert error.value.code == "backend_sampling.attestation_active_runtime_mismatch"


@pytest.mark.parametrize(
    "mismatch",
    ["model_identity", "tokenizer_identity", "generation_config"],
)
def test_persisted_rebind_rejects_backend_owned_identity_mismatch(
    mismatch: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    cpu_three_policy_attestation_aggregate: dict[str, Any],
) -> None:
    from scripts.research import attest_request_scoped_sampling_cuda as script
    from src.inference import backend as backend_module

    _use_cpu_fixture_payload_contract(monkeypatch)
    fixture = cpu_three_policy_attestation_aggregate
    output = tmp_path / f"identity-{mismatch}.json"
    output.write_text(script.json.dumps(fixture["artifact"]), encoding="utf-8")
    monkeypatch.setattr(
        backend_module,
        "_validate_sampled_runtime_attestation_bundle",
        _fake_full_bundle_validation(fixture),
    )
    model_identity = dict(fixture["model_identity"])
    tokenizer_identity = dict(fixture["tokenizer_identity"])
    generation_fingerprint = fixture["generation_config_fingerprint"]
    if mismatch == "model_identity":
        model_identity["family"] = "changed"
    elif mismatch == "tokenizer_identity":
        tokenizer_identity["identity"] = "changed"
    else:
        generation_fingerprint = "changed-generation-config"
    backend = backend_module.HFGenerateBackend(
        model=MutableRuntimeModel(),
        tokenizer=MutableTokenizer(),
        model_identity=model_identity,
        tokenizer_identity=tokenizer_identity,
        generation_config_fingerprint=generation_fingerprint,
    )
    selected_fingerprint = fixture["artifact"]["policy_attestations"][0][
        "decode_generation_policy_fingerprint"
    ]
    with pytest.raises(RuntimeContractError) as error:
        backend_module.load_and_rebind_sampled_runtime_attestation_aggregate(
            output,
            decode_generation_policy_fingerprint=selected_fingerprint,
            backend=backend,
        )
    assert error.value.code == "backend_sampling.attestation_active_runtime_mismatch"


@pytest.mark.parametrize(
    "mutation",
    ["adapter_value", "embedding_value", "tokenizer", "model_config", "model_structure"],
)
def test_persisted_rebind_rejects_live_runtime_state_mismatch(
    mutation: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    cpu_three_policy_attestation_aggregate: dict[str, Any],
) -> None:
    from scripts.research import attest_request_scoped_sampling_cuda as script
    from src.inference import backend as backend_module

    _use_cpu_fixture_payload_contract(monkeypatch)
    fixture = cpu_three_policy_attestation_aggregate
    output = tmp_path / f"live-{mutation}.json"
    output.write_text(script.json.dumps(fixture["artifact"]), encoding="utf-8")
    monkeypatch.setattr(
        backend_module,
        "_validate_sampled_runtime_attestation_bundle",
        _fake_full_bundle_validation(fixture),
    )
    backend, _, _, _ = _new_bound_rebind_backend()
    with torch.no_grad():
        if mutation == "adapter_value":
            backend.model.lora_A.data.add_(1.0)
        elif mutation == "embedding_value":
            backend.model.shared_embed_delta.data.add_(1.0)
        elif mutation == "model_structure":
            backend.model.register_buffer("new_runtime_buffer", torch.zeros(1))
    if mutation == "tokenizer":
        backend.tokenizer.backend_tokenizer.normalizer = "changed"
    elif mutation == "model_config":
        backend.model.config._attn_implementation = "changed"
    selected_fingerprint = fixture["artifact"]["policy_attestations"][0][
        "decode_generation_policy_fingerprint"
    ]
    with pytest.raises(RuntimeContractError):
        backend_module.load_and_rebind_sampled_runtime_attestation_aggregate(
            output,
            decode_generation_policy_fingerprint=selected_fingerprint,
            backend=backend,
        )


@pytest.mark.parametrize("mismatch", ["device", "runtime"])
def test_persisted_rebind_rejects_device_or_runtime_mismatch(
    mismatch: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    cpu_three_policy_attestation_aggregate: dict[str, Any],
) -> None:
    from scripts.research import attest_request_scoped_sampling_cuda as script
    from src.inference import backend as backend_module

    _use_cpu_fixture_payload_contract(monkeypatch)
    fixture = cpu_three_policy_attestation_aggregate
    output = tmp_path / f"environment-{mismatch}.json"
    output.write_text(script.json.dumps(fixture["artifact"]), encoding="utf-8")
    full_validate = _fake_full_bundle_validation(fixture)
    monkeypatch.setattr(
        backend_module,
        "_validate_sampled_runtime_attestation_bundle",
        full_validate,
    )
    if mismatch == "runtime":
        monkeypatch.setattr(
            backend_module,
            "_runtime_identity",
            lambda: {"runtime": "changed"},
        )
    else:
        observed = backend_module._execution_device_identity(torch.device("cpu"))
        changed = {**observed, "logical_device": "changed"}
        monkeypatch.setattr(
            backend_module,
            "_execution_device_identity",
            lambda device: changed,
        )
    backend, _, _, _ = _new_bound_rebind_backend()
    selected_fingerprint = fixture["artifact"]["policy_attestations"][0][
        "decode_generation_policy_fingerprint"
    ]
    with pytest.raises(RuntimeContractError) as error:
        backend_module.load_and_rebind_sampled_runtime_attestation_aggregate(
            output,
            decode_generation_policy_fingerprint=selected_fingerprint,
            backend=backend,
        )
    assert error.value.code == "backend_sampling.attestation_active_runtime_mismatch"


def test_rebound_capability_rejects_equivalent_but_different_backend_object(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    cpu_three_policy_attestation_aggregate: dict[str, Any],
) -> None:
    from scripts.research import attest_request_scoped_sampling_cuda as script
    from src.inference import backend as backend_module

    _use_cpu_fixture_payload_contract(monkeypatch)
    fixture = cpu_three_policy_attestation_aggregate
    output = tmp_path / "backend-object-swap.json"
    output.write_text(script.json.dumps(fixture["artifact"]), encoding="utf-8")
    monkeypatch.setattr(
        backend_module,
        "_validate_sampled_runtime_attestation_bundle",
        _fake_full_bundle_validation(fixture),
    )
    first_backend, _, _, _ = _new_bound_rebind_backend()
    entry = fixture["artifact"]["policy_attestations"][0]
    capability = backend_module.load_and_rebind_sampled_runtime_attestation_aggregate(
        output,
        decode_generation_policy_fingerprint=entry[
            "decode_generation_policy_fingerprint"
        ],
        backend=first_backend,
    )
    second_backend, model_identity, tokenizer_identity, generation_fingerprint = (
        _new_bound_rebind_backend()
    )
    request = backend_module.DecodeRequest(
        request_id="backend-swap",
        prompt_token_ids=[1, 2],
        model_inputs={"input_ids": torch.tensor([1, 2])},
        generation_policy=backend_module.DecodeGenerationPolicy(
            **entry["decode_generation_policy"]
        ),
        sampling_seed=1000,
    )
    with pytest.raises(RuntimeContractError) as error:
        second_backend.generate_batch_with_verified_runtime_attestation(
            [request],
            model_identity=model_identity,
            tokenizer_identity=tokenizer_identity,
            generation_config_fingerprint=generation_fingerprint,
            verified_runtime_attestation=capability,
        )
    assert error.value.code == "backend_sampling.attestation_active_runtime_mismatch"


@pytest.mark.parametrize(
    "stale_field",
    ["model_identity", "tokenizer_identity", "generation_config"],
)
def test_rebound_capability_rejects_stale_call_identity_arguments(
    stale_field: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    cpu_three_policy_attestation_aggregate: dict[str, Any],
) -> None:
    from scripts.research import attest_request_scoped_sampling_cuda as script
    from src.inference import backend as backend_module

    _use_cpu_fixture_payload_contract(monkeypatch)
    fixture = cpu_three_policy_attestation_aggregate
    output = tmp_path / f"stale-call-{stale_field}.json"
    output.write_text(script.json.dumps(fixture["artifact"]), encoding="utf-8")
    monkeypatch.setattr(
        backend_module,
        "_validate_sampled_runtime_attestation_bundle",
        _fake_full_bundle_validation(fixture),
    )
    backend, model_identity, tokenizer_identity, generation_fingerprint = (
        _new_bound_rebind_backend()
    )
    entry = fixture["artifact"]["policy_attestations"][0]
    capability = backend_module.load_and_rebind_sampled_runtime_attestation_aggregate(
        output,
        decode_generation_policy_fingerprint=entry[
            "decode_generation_policy_fingerprint"
        ],
        backend=backend,
    )
    call_model_identity = dict(model_identity)
    call_tokenizer_identity = dict(tokenizer_identity)
    call_generation_fingerprint = generation_fingerprint
    if stale_field == "model_identity":
        call_model_identity["family"] = "stale"
    elif stale_field == "tokenizer_identity":
        call_tokenizer_identity["identity"] = "stale"
    else:
        call_generation_fingerprint = "stale-generation-config"
    request = backend_module.DecodeRequest(
        request_id="stale-call-identity",
        prompt_token_ids=[1, 2],
        model_inputs={"input_ids": torch.tensor([1, 2])},
        generation_policy=backend_module.DecodeGenerationPolicy(
            **entry["decode_generation_policy"]
        ),
        sampling_seed=1000,
    )
    with pytest.raises(RuntimeContractError) as error:
        backend.generate_batch_with_verified_runtime_attestation(
            [request],
            model_identity=call_model_identity,
            tokenizer_identity=call_tokenizer_identity,
            generation_config_fingerprint=call_generation_fingerprint,
            verified_runtime_attestation=capability,
        )
    assert error.value.code == "backend_sampling.attestation_active_runtime_mismatch"


def test_no_public_sampled_attestation_result_bypass_or_serialized_mint_api() -> None:
    import src.inference.backend as backend_module

    assert not hasattr(
        backend_module.HFGenerateBackend,
        "generate_batch_for_sampling_attestation",
    )
    assert not hasattr(
        backend_module,
        "verify_sampled_runtime_attestation_bundle",
    )
    assert hasattr(
        backend_module,
        "validate_sampled_runtime_attestation_bundle",
    )


def test_receipt_payload_is_recursively_immutable_and_logprob_bound() -> None:
    from dataclasses import replace

    from src.inference.backend import (
        TokenTrace,
        build_decode_execution_receipt,
        effective_generation_arguments,
    )

    request = _request("row-immutable", policy=_greedy_policy(), seed=None)
    trace = TokenTrace(
        step_index=0,
        token_id=1,
        token_text="1",
        logprob=-0.25,
        is_stop=False,
        is_pad=False,
        backend="hf",
        backend_mode="generate",
        response_family="hf",
    )
    receipt = build_decode_execution_receipt(
        request=request,
        generated_token_ids=[1],
        token_trace=[trace],
        stop_reason="length",
        model_identity={"family": "tiny"},
        tokenizer_identity={"sha256": "tiny"},
        generation_config_fingerprint="config",
        executed_generation_arguments=effective_generation_arguments(
            request.generation_policy,
            eos_token_id=9,
            pad_token_id=0,
        ),
        runtime_identity={"versions": {"ordered": ["torch", "transformers"]}},
    )
    with pytest.raises(TypeError):
        receipt.executed_generation_arguments["top_k"] = 50
    with pytest.raises(TypeError):
        receipt.runtime_identity["versions"]["ordered"][0] = "changed"
    assert receipt.runtime_identity["versions"]["ordered"] == ("torch", "transformers")

    from src.inference.backend import DecodeResult

    result = DecodeResult(
        request_id=request.request_id,
        backend="hf",
        backend_mode="generate",
        response_family="hf",
        prompt_token_ids=list(request.prompt_token_ids),
        generated_token_ids=[1],
        raw_generated_text="1",
        parser_text="1",
        strip_policy="none",
        stop_reason="length",
        model_identity={"family": "tiny"},
        tokenizer_identity={"sha256": "tiny"},
        generation_config_fingerprint="config",
        token_trace=[replace(trace, logprob=-0.5)],
        execution_receipt=receipt,
    )
    with pytest.raises(RuntimeContractError) as exc_info:
        result.validate_for_scored()
    assert exc_info.value.code == "backend_receipt.binding_mismatch"
    assert "canonical_float32_score_trace_hash" in exc_info.value.context["mismatches"]


def test_custom_generate_uses_installed_hf_loop_with_four_generators_scores_and_cache() -> (
    None
):
    from transformers import GPT2Config, GPT2LMHeadModel

    from src.inference.backend import (
        _GenerationExecutionCapture,
        _SamplingTimingRecorder,
        _generation_config_from_arguments,
        custom_generate,
        effective_generation_arguments,
    )

    torch.manual_seed(7)
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=10,
            n_positions=16,
            n_ctx=16,
            n_embd=8,
            n_layer=1,
            n_head=1,
            bos_token_id=1,
            eos_token_id=9,
            pad_token_id=9,
        )
    ).eval()
    model.generation_config.top_k = 7
    model.generation_config.min_length = 5
    model.generation_config.token_healing = True
    policy = _sampled_policy()
    generation_config = _generation_config_from_arguments(
        effective_generation_arguments(
            policy,
            eos_token_id=9,
            pad_token_id=9,
            bos_token_id=1,
        ),
        policy=policy,
    )
    generators = tuple(torch.Generator().manual_seed(500 + index) for index in range(4))
    timing = _SamplingTimingRecorder()
    capture = _GenerationExecutionCapture()
    outputs = custom_generate(
        model,
        request_generators=generators,
        generation_policy=policy,
        sampling_timing=timing,
        execution_capture=capture,
        input_ids=torch.tensor([[1, 2], [1, 3], [1, 4], [1, 5]], dtype=torch.long),
        attention_mask=torch.ones((4, 2), dtype=torch.long),
        generation_config=generation_config,
    )

    assert outputs.sequences.shape[0] == 4
    assert outputs.sequences.shape[1] == 2 + len(outputs.scores)
    assert 1 <= len(outputs.scores) <= policy.max_new_tokens
    assert all(tuple(score.shape) == (4, 10) for score in outputs.scores)
    assert outputs.past_key_values is not None
    assert callable(model._sample)
    assert timing.categorical_draw_call_count == 4 * len(outputs.scores)
    assert capture.custom_sampler_executed is True
    assert capture.prepared_generation_profile is not None
    assert capture.prepared_generation_profile["use_model_defaults"] is False
    assert capture.prepared_generation_profile["top_k"] == 0
    assert capture.prepared_generation_profile["min_length"] == 0
    assert capture.prepared_generation_profile["token_healing"] is False


def test_stock_generation_scores_are_captured_after_sampling_warpers() -> None:
    from transformers import GPT2Config, GPT2LMHeadModel
    from transformers.generation.logits_process import (
        LogitsProcessorList,
        TemperatureLogitsWarper,
        TopPLogitsWarper,
    )

    from src.inference.backend import (
        _ProcessedLogitCapture,
        _generation_config_from_arguments,
        effective_generation_arguments,
    )

    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=10,
            n_positions=16,
            n_ctx=16,
            n_embd=8,
            n_layer=1,
            n_head=1,
            bos_token_id=1,
            eos_token_id=9,
            pad_token_id=9,
        )
    ).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    policy = _sampled_policy(temperature=0.5, top_p=0.4)
    generation_config = _generation_config_from_arguments(
        effective_generation_arguments(
            policy,
            eos_token_id=9,
            pad_token_id=9,
            bos_token_id=1,
        ),
        policy=policy,
    )
    generation_config.max_new_tokens = 1
    generation_config.max_length = 3
    pre_warper_capture = _ProcessedLogitCapture()

    outputs = model.generate(
        input_ids=torch.tensor([[1, 2]], dtype=torch.long),
        attention_mask=torch.ones((1, 2), dtype=torch.long),
        generation_config=generation_config,
        logits_processor=LogitsProcessorList([pre_warper_capture]),
        use_model_defaults=False,
    )

    assert pre_warper_capture.first_scores_float32 is not None
    pre_warper = pre_warper_capture.first_scores_float32
    post_warper = outputs.scores[0].detach().to(dtype=torch.float32, device="cpu")
    expected_post_warper = TemperatureLogitsWarper(policy.temperature)(
        torch.tensor([[1, 2]], dtype=torch.long), pre_warper.clone()
    )
    expected_post_warper = TopPLogitsWarper(policy.top_p)(
        torch.tensor([[1, 2]], dtype=torch.long), expected_post_warper
    )
    assert torch.isfinite(pre_warper).all()
    assert torch.isneginf(post_warper).any()
    assert torch.equal(post_warper, expected_post_warper)


def test_processed_logit_match_accepts_matching_negative_infinity_masks() -> None:
    from src.inference.backend import _processed_logits_match

    custom = torch.tensor([[0.0, float("-inf"), 1.0]], dtype=torch.float32)
    stock = torch.tensor([[0.0, float("-inf"), 1.0 + 1e-7]], dtype=torch.float32)
    assert _processed_logits_match(
        custom,
        stock,
        absolute_tolerance=1e-6,
        relative_tolerance=1e-6,
    )


@pytest.mark.parametrize(
    ("custom", "stock"),
    [
        (
            [0.0, float("-inf"), float("inf")],
            [0.0, float("inf"), float("-inf")],
        ),
        ([0.0, float("-inf"), 1.0], [0.0, 2.0, 1.0]),
        ([0.0, float("nan"), 1.0], [0.0, float("nan"), 1.0]),
        ([0.0, float("-inf"), 1.0], [0.0, float("-inf"), 1.01]),
    ],
)
def test_processed_logit_match_rejects_infinity_nan_and_finite_drift(
    custom: list[float], stock: list[float]
) -> None:
    from src.inference.backend import _processed_logits_match

    assert not _processed_logits_match(
        torch.tensor([custom], dtype=torch.float32),
        torch.tensor([stock], dtype=torch.float32),
        absolute_tolerance=1e-6,
        relative_tolerance=1e-6,
    )


def test_real_generation_mixin_request_seeds_ignore_global_rng_and_batch_order() -> (
    None
):
    from transformers import GPT2Config, GPT2LMHeadModel

    from src.inference.backend import (
        _GenerationExecutionCapture,
        _SamplingTimingRecorder,
        _generation_config_from_arguments,
        custom_generate,
        effective_generation_arguments,
    )

    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=10,
            n_positions=16,
            n_ctx=16,
            n_embd=8,
            n_layer=1,
            n_head=1,
            bos_token_id=1,
            eos_token_id=9,
            pad_token_id=9,
        )
    ).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    policy = _sampled_policy(temperature=1.0, top_p=1.0)

    def run(
        request_ids: list[str], seeds: list[int], *, global_seed: int
    ) -> tuple[dict[str, list[int]], int]:
        torch.manual_seed(global_seed)
        generation_config = _generation_config_from_arguments(
            effective_generation_arguments(
                policy,
                eos_token_id=9,
                pad_token_id=9,
                bos_token_id=1,
            ),
            policy=policy,
        )
        timing = _SamplingTimingRecorder()
        capture = _GenerationExecutionCapture()
        outputs = custom_generate(
            model,
            request_generators=tuple(
                torch.Generator().manual_seed(seed) for seed in seeds
            ),
            generation_policy=policy,
            sampling_timing=timing,
            execution_capture=capture,
            input_ids=torch.tensor(
                [[1, 2 + index] for index in range(len(request_ids))],
                dtype=torch.long,
            ),
            attention_mask=torch.ones((len(request_ids), 2), dtype=torch.long),
            generation_config=generation_config,
        )
        assert capture.custom_sampler_executed is True
        assert timing.categorical_draw_call_count == len(request_ids) * len(
            outputs.scores
        )
        return {
            request_id: [int(token_id) for token_id in outputs.sequences[index, 2:]]
            for index, request_id in enumerate(request_ids)
        }, timing.categorical_draw_call_count

    request_ids = ["row-0", "row-1", "row-2", "row-3"]
    seeds = [500, 501, 502, 503]
    baseline, baseline_draws = run(request_ids, seeds, global_seed=7)
    different_global, _ = run(request_ids, seeds, global_seed=987654)
    reversed_rows, _ = run(
        list(reversed(request_ids)),
        list(reversed(seeds)),
        global_seed=123,
    )
    changed_seed, _ = run(request_ids, [504, 501, 502, 503], global_seed=7)

    assert baseline_draws > 0
    assert different_global == baseline
    assert reversed_rows == baseline
    assert changed_seed["row-0"] != baseline["row-0"]
    assert {request_id: changed_seed[request_id] for request_id in request_ids[1:]} == {
        request_id: baseline[request_id] for request_id in request_ids[1:]
    }


def test_typed_attestation_bundle_round_trips_but_untrusted_lineage_cannot_validate() -> (
    None
):
    from transformers import GPT2Config, GPT2LMHeadModel

    from src.inference.backend import (
        DecodeRequest,
        HFGenerateBackend,
        SampledRuntimeAttestationBundle,
        build_sampled_runtime_attestation_bundle,
        validate_sampled_runtime_attestation_bundle,
    )

    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=10,
            n_positions=16,
            n_ctx=16,
            n_embd=8,
            n_layer=1,
            n_head=1,
            bos_token_id=1,
            eos_token_id=9,
            pad_token_id=9,
        )
    ).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    backend = HFGenerateBackend(model=model, tokenizer=TinyTokenizer())
    from src.inference.backend import DecodeGenerationPolicy

    policy = DecodeGenerationPolicy.sampled(
        max_new_tokens=1,
        repetition_penalty=1.0,
        temperature=1.0,
        top_p=1.0,
    )
    canonical = [
        DecodeRequest(
            request_id=f"calibration-{index}",
            prompt_token_ids=[1, 2 + index],
            model_inputs={"input_ids": torch.tensor([1, 2 + index])},
            generation_policy=policy,
            sampling_seed=800 + index,
        )
        for index in range(4)
    ]
    qwen_token_identity = {
        "required_token_count": 1004,
        "wrapper_token_ids": {
            "<|object_ref_start|>": 100,
            "<|object_ref_end|>": 101,
            "<|box_start|>": 102,
            "<|box_end|>": 103,
        },
        "coord_token_count": 1000,
        "coord_token_id_min": 200,
        "coord_token_id_max": 1199,
        "coord_token_ids_contiguous": True,
        "im_end_newline_text": "<|im_end|>\n",
        "im_end_token_ids": [9],
        "newline_token_ids": [8],
        "im_end_newline_token_ids": [9, 8],
        "im_end_newline_split_verified": True,
        "tokenizer_vocab_size": 1200,
    }
    identities = {
        "model_identity": {"family": "cpu-fixture"},
        "tokenizer_identity": qwen_token_identity,
        "generation_config_fingerprint": "cpu-fixture-config",
    }

    def execute(case_name: str, requests: list[Any]) -> Any:
        _run_unit_sampled_backend(
            backend,
            requests,
            case_name=case_name,
            **identities,
        )
        assert backend.last_sampling_attestation_case is not None
        return backend.last_sampling_attestation_case

    four_forward = execute("batch_size_four_forward", canonical)
    processed_logit_parity = backend._attest_processed_logit_parity_against_stock(
        canonical
    )
    four_reversed = execute("batch_size_four_reversed", list(reversed(canonical)))
    three_forward = execute("batch_size_three_forward", canonical[:3])
    three_reversed = execute(
        "batch_size_three_reversed", list(reversed(canonical[:3]))
    )
    bundle = build_sampled_runtime_attestation_bundle(
        lineage={
            "config_path": "/tmp/config.yaml",
            "config_sha256": "1" * 64,
            "checkpoint_manifest_path": "/tmp/checkpoint.json",
            "checkpoint_manifest_sha256": "2" * 64,
            "calibration_manifest_path": "/tmp/calibration.jsonl",
            "calibration_manifest_sha256": "3" * 64,
            "calibration_request_ids": [request.request_id for request in canonical],
            "calibration_image_ids": [100, 101, 102, 103],
            "calibration_image_sha256": ["6" * 64] * 4,
            "calibration_prompt_token_identifier_hashes": list(
                four_forward.prompt_token_identifier_hashes
            ),
            "calibration_model_input_fingerprints": list(
                four_forward.model_input_fingerprints
            ),
            "calibration_prompt_records_fingerprint": "7" * 64,
            "calibration_image_plan_fingerprint": "8" * 64,
            "calibration_request_plan": [],
            "calibration_request_plan_fingerprint": "9" * 64,
            "checkpoint_payload_identity": {},
            "model_dtype": "bf16",
            "attention_implementation": "sdpa",
            "temperature": 0.2,
            "root_seed": 2026071301,
            "frozen_sampling_factors": {
                "max_new_tokens": 512,
                "top_p": 0.95,
                "repetition_penalty": 1.0,
            },
            "qwen_runtime_identity": {
                "base_model_path": "/tmp/qwen3-vl",
                "base_config_sha256": "4" * 64,
                "tokenizer_sha256": "5" * 64,
                "load_model": True,
                "attn_implementation": "sdpa",
                "processor": {"processor_class": "Qwen3VLProcessor"},
                "model": {"model_type": "qwen3_vl"},
                "tokens": qwen_token_identity,
                "package_versions": {},
                "runtime_patches": {},
            },
        },
        executed_cases=(
            four_forward,
            four_reversed,
            three_forward,
            three_reversed,
        ),
        processed_logit_parity=processed_logit_parity,
    )
    replay = SampledRuntimeAttestationBundle.from_artifact_dict(
        bundle.to_artifact_dict()
    )
    assert replay.bundle_payload_fingerprint == bundle.bundle_payload_fingerprint
    with pytest.raises(RuntimeContractError) as exc_info:
        validate_sampled_runtime_attestation_bundle(replay)
    assert (
        exc_info.value.code
        == "backend_sampling.attestation_primary_lineage_mismatch"
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("min_length", 1),
        ("min_new_tokens", 1),
        ("token_healing", True),
        ("guidance_scale", 1.5),
        ("dola_layers", "low"),
        ("future_non_neutral_behavior", True),
    ],
)
def test_complete_generation_profile_rejects_undeclared_behavior(
    field: str, value: Any
) -> None:
    from src.inference.backend import (
        _generation_config_from_arguments,
        effective_generation_arguments,
    )

    policy = _sampled_policy()
    arguments = effective_generation_arguments(
        policy,
        eos_token_id=9,
        pad_token_id=0,
    )
    assert arguments["use_model_defaults"] is False
    assert arguments["min_length"] == 0
    assert arguments["min_new_tokens"] is None
    assert arguments["token_healing"] is False
    assert arguments["guidance_scale"] is None
    assert arguments["dola_layers"] is None
    arguments[field] = value
    with pytest.raises(RuntimeContractError) as exc_info:
        _generation_config_from_arguments(arguments, policy=policy)
    assert exc_info.value.code in {
        "backend_policy.generation_profile_drift",
        "backend_policy.unknown_generation_behavior",
    }


def test_receipt_logprob_hash_canonicalizes_padding_to_null_and_rejects_nonpadding_nan() -> (
    None
):
    from dataclasses import replace

    from src.inference.backend import (
        TokenTrace,
        build_decode_execution_receipt,
        effective_generation_arguments,
    )

    request = _request("row-logprobs", policy=_greedy_policy(), seed=None)
    content = TokenTrace(
        step_index=0,
        token_id=1,
        token_text="1",
        logprob=-0.1,
        is_stop=False,
        is_pad=False,
        backend="hf",
        backend_mode="generate",
        response_family="hf",
    )
    padding = TokenTrace(
        step_index=1,
        token_id=0,
        token_text="<pad>",
        logprob=None,
        is_stop=False,
        is_pad=True,
        backend="hf",
        backend_mode="generate",
        response_family="hf",
    )
    common = {
        "request": request,
        "generated_token_ids": [1],
        "stop_reason": "length",
        "model_identity": {"family": "tiny"},
        "tokenizer_identity": {"sha256": "tiny"},
        "generation_config_fingerprint": "config",
        "executed_generation_arguments": effective_generation_arguments(
            request.generation_policy,
            eos_token_id=9,
            pad_token_id=0,
        ),
    }
    null_padding = build_decode_execution_receipt(
        token_trace=[content, padding],
        **common,
    )
    nan_padding = build_decode_execution_receipt(
        token_trace=[content, replace(padding, logprob=float("nan"))],
        **common,
    )
    assert (
        null_padding.canonical_float32_score_trace_hash
        == nan_padding.canonical_float32_score_trace_hash
    )

    with pytest.raises(RuntimeContractError) as exc_info:
        build_decode_execution_receipt(
            token_trace=[replace(content, logprob=float("nan")), padding],
            **common,
        )
    assert exc_info.value.code == "backend_receipt.non_finite_selected_logprob"


def test_canonical_float32_score_trace_hash_binds_float32_and_stop_classification() -> (
    None
):
    from dataclasses import replace

    from src.inference.backend import (
        TokenTrace,
        _canonical_float32_score_trace_hash,
        canonical_float32_logprob,
    )

    base = 0.12345679104328156
    same_float32 = base + 1e-12
    different_float32 = base + 1e-5
    trace = TokenTrace(
        step_index=0,
        token_id=1,
        token_text="1",
        logprob=base,
        is_stop=False,
        is_pad=False,
        backend="hf",
        backend_mode="generate",
        response_family="hf",
    )

    assert base != same_float32
    assert canonical_float32_logprob(base) == canonical_float32_logprob(same_float32)
    assert _canonical_float32_score_trace_hash(
        [trace]
    ) == _canonical_float32_score_trace_hash([replace(trace, logprob=same_float32)])
    assert _canonical_float32_score_trace_hash(
        [trace]
    ) != _canonical_float32_score_trace_hash(
        [replace(trace, logprob=different_float32)]
    )
    assert _canonical_float32_score_trace_hash(
        [trace]
    ) != _canonical_float32_score_trace_hash([replace(trace, is_stop=True)])


def test_attestation_replays_canonical_compact_object_selected_token_score() -> None:
    import math

    from src.inference.backend import (
        DecodeResult,
        TokenTrace,
        _compact_object_selected_token_score_replay,
        canonical_float32_logprob,
    )
    from src.inference.scoring import SCORE_POLICY_FINGERPRINT

    pieces = [
        "<|object_ref_start|>",
        "person",
        "<|object_ref_end|>",
        "<|box_start|>",
        "<|coord_100|>",
        "<|coord_200|>",
        "<|coord_300|>",
        "<|coord_400|>",
        "<|box_end|>",
    ]
    logprobs = [-0.1, -100.0, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8]
    traces = [
        TokenTrace(
            step_index=index,
            token_id=100 + index,
            token_text=text,
            logprob=logprobs[index],
            is_stop=False,
            is_pad=False,
            backend="hf",
            backend_mode="generate",
            response_family="hf",
        )
        for index, text in enumerate(pieces)
    ]
    result = DecodeResult(
        request_id="compact-row",
        backend="hf",
        backend_mode="generate",
        response_family="hf",
        prompt_token_ids=[1],
        generated_token_ids=[trace.token_id for trace in traces],
        raw_generated_text="".join(pieces),
        parser_text="".join(pieces),
        strip_policy="none",
        stop_reason="length",
        model_identity={},
        tokenizer_identity={},
        generation_config_fingerprint="fixture",
        token_trace=traces,
        execution_receipt=None,
    )

    replay = _compact_object_selected_token_score_replay(
        result,
        image_width=1000,
        image_height=1000,
    )
    assert replay["score_policy_fingerprint"] == SCORE_POLICY_FINGERPRINT
    assert replay["valid_prediction_count"] == 1
    selected = replay["prediction_replays"][0]
    assert selected["generated_step_indices"] == [0, 2, 3, 4, 5, 6, 7, 8]
    assert selected["token_ids"] == [100, 102, 103, 104, 105, 106, 107, 108]
    assert -100.0 not in selected["selected_logprobs"]
    expected = math.exp(
        sum(canonical_float32_logprob(logprobs[index]) for index in [0, 2, 3, 4, 5, 6, 7, 8])
        / 8
    )
    assert selected["score"] == canonical_float32_logprob(expected)


def test_attestation_rejects_vacuous_empty_selected_token_replay() -> None:
    from src.inference.backend import _require_scoreable_compact_object_replay

    with pytest.raises(RuntimeContractError) as exc_info:
        _require_scoreable_compact_object_replay(
            {"prediction_replays": []},
            request_id="empty-row",
        )
    assert exc_info.value.code == "backend_sampling.attestation_row_score_empty"


def test_checkpoint_payload_verifier_rehashes_adapter_and_embedding_files(
    tmp_path: Any,
) -> None:
    import hashlib
    import json
    from pathlib import Path

    from src.inference.backend import verify_checkpoint_payload_identity

    run_root = Path(tmp_path) / "run"
    checkpoint_dir = run_root / "checkpoints" / "step-1"
    adapter_dir = checkpoint_dir / "adapter"
    embedding_dir = checkpoint_dir / "special_token_embeddings"
    adapter_dir.mkdir(parents=True)
    embedding_dir.mkdir(parents=True)
    payloads = {
        "checkpoints/step-1/adapter/adapter_config.json": b"adapter-config",
        "checkpoints/step-1/adapter/adapter_model.safetensors": b"adapter-model",
        "checkpoints/step-1/special_token_embeddings/special_token_embeddings.json": b"embedding-metadata",
        "checkpoints/step-1/special_token_embeddings/special_token_embeddings.safetensors": b"embedding-tensor",
    }
    digests = {}
    for relative, payload in payloads.items():
        path = run_root / relative
        path.write_bytes(payload)
        digests[relative] = hashlib.sha256(payload).hexdigest()
    manifest = {
        "adapter": {
            "identity": {
                "file_sha256": {
                    key: value for key, value in digests.items() if "/adapter/" in key
                },
                "fingerprint": "adapter-fingerprint",
            }
        },
        "special_token_embeddings": {
            "identity": {
                "metadata_path": "checkpoints/step-1/special_token_embeddings/special_token_embeddings.json",
                "metadata_sha256": digests[
                    "checkpoints/step-1/special_token_embeddings/special_token_embeddings.json"
                ],
                "tensor_path": "checkpoints/step-1/special_token_embeddings/special_token_embeddings.safetensors",
                "tensor_sha256": digests[
                    "checkpoints/step-1/special_token_embeddings/special_token_embeddings.safetensors"
                ],
                "fingerprint": "embedding-fingerprint",
            }
        },
    }
    checkpoint_manifest = checkpoint_dir / "checkpoint.json"
    checkpoint_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    identity = verify_checkpoint_payload_identity(checkpoint_manifest)
    assert identity["verified_file_sha256"] == dict(sorted(digests.items()))

    (adapter_dir / "adapter_model.safetensors").write_bytes(b"tampered")
    with pytest.raises(RuntimeContractError) as exc_info:
        verify_checkpoint_payload_identity(checkpoint_manifest)
    assert exc_info.value.code == "backend_sampling.attestation_checkpoint_payload_mismatch"


def test_attestation_request_binder_rejects_nonfrozen_first_four_identity() -> None:
    from src.inference.backend import _bind_sampling_attestation_requests

    frozen_seeds = [
        3565713206559848094,
        2992931333390152433,
        6011931503842164206,
        7384003725263415097,
    ]
    requests = [
        _request(
            request_id,
            policy=_sampled_policy(),
            seed=frozen_seeds[index],
        )
        for index, request_id in enumerate(
            [
                "not-the-frozen-first-request",
                "coco2017_val_000000303713",
                "coco2017_val_000000529148",
                "coco2017_val_000000538236",
            ]
        )
    ]
    manifest = (
        "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
        "2026-07-13-spatial-scope-history-disentanglement/readiness-v2/"
        "sampling-calibration-12-manifest.jsonl"
    )
    with pytest.raises(RuntimeContractError) as exc_info:
        _bind_sampling_attestation_requests(
            {"calibration_manifest_path": manifest},
            requests=requests,
        )
    assert exc_info.value.code == "backend_sampling.attestation_request_plan_invalid"


def test_attestation_request_binder_rejects_nonfrozen_seed_plan() -> None:
    from src.inference.backend import _bind_sampling_attestation_requests

    request_ids = [
        "coco2017_val_000000563648",
        "coco2017_val_000000303713",
        "coco2017_val_000000529148",
        "coco2017_val_000000538236",
    ]
    requests = [
        _request(request_id, policy=_sampled_policy(), seed=100 + index)
        for index, request_id in enumerate(request_ids)
    ]
    manifest = (
        "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
        "2026-07-13-spatial-scope-history-disentanglement/readiness-v2/"
        "sampling-calibration-12-manifest.jsonl"
    )
    with pytest.raises(RuntimeContractError) as exc_info:
        _bind_sampling_attestation_requests(
            {"calibration_manifest_path": manifest},
            requests=requests,
        )
    assert exc_info.value.code == "backend_sampling.attestation_request_plan_invalid"
