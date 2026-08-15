from __future__ import annotations

from dataclasses import dataclass
import gc
from types import SimpleNamespace
import weakref

import pytest

from scripts.research import human13_live_model as live_model
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


def test_parity_surface_uses_only_one_image_inference_assembly(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    leaf = next(
        item
        for item in launcher.load_leaf_configs()
        if item.training_rp == 1.0 and item.arm_id == "C"
    )
    frozen = SimpleNamespace(
        c_leaf_path=leaf.source_path,
        qualification_learning_rate_ray=(3.0e-7, 1.0e-6, 3.0e-6, 1.0e-5, 3.0e-5),
        default_qualification_learning_rate=3.0e-6,
    )
    calls: list[object] = []

    class Model:
        def __init__(self) -> None:
            self.training = True
            self.requires_grad = True

        def requires_grad_(self, value: bool):
            calls.append(("requires_grad", value))
            self.requires_grad = value
            return self

        def eval(self):
            calls.append("eval")
            self.training = False
            return self

    model = Model()

    class Accelerator:
        num_processes = 1
        process_index = 0
        device = "cuda:0"

        def prepare_model(self, candidate, *, evaluation_mode):
            calls.append(("prepare_model", evaluation_mode))
            assert candidate is model
            return candidate

    accelerator = Accelerator()
    components = SimpleNamespace(
        model=object(),
        base_model_path=live_model.SOURCE_BASE_MODEL_PATH,
        base_config_sha256=live_model.SOURCE_BASE_CONFIG_SHA256,
        tokenizer_sha256=live_model.SOURCE_TOKENIZER_SHA256,
        tokenizer=object(),
    )

    class InferenceOnlyBackend:
        def create_accelerator(self, plan):
            calls.append(("accelerator", plan.mixed_precision))
            return accelerator

        def validate_accelerator(self, candidate, plan):
            calls.append(("validate_accelerator", plan.attn_implementation))
            assert candidate is accelerator

        def load_qwen(self, plan):
            calls.append(("load_qwen", plan.mixed_precision, plan.attn_implementation))
            return components

        def warm_start_language_dora(
            self, base_model, loaded, plan, *, repo_root
        ):
            calls.append(("warm_start_dora", plan.adapter_rank, plan.adapter_alpha))
            assert base_model is components.model
            assert loaded is components
            return SimpleNamespace(model=model, receipt=object())

        def load_and_freeze_special_token_delta(
            self, candidate, loaded, plan, *, repo_root
        ):
            calls.append(("load_frozen_delta", plan.freeze_special_token_delta))
            assert candidate is model
            assert loaded is components
            return SimpleNamespace(
                model=model,
                shared_embed_delta=SimpleNamespace(requires_grad=False),
                receipt=object(),
            )

        def enable_memory_savers(self, model):
            raise AssertionError("parity must not enable training memory savers")

        def build_optimizer(self, model, adapter_result, plan):
            raise AssertionError("parity must not construct an optimizer")

        def build_trainable_surface_receipt(self, *args):
            raise AssertionError("parity must not construct a trainable surface")

        def build_runtime(self, **kwargs):
            raise AssertionError("parity must not construct TrainRuntime")

    validation = SimpleNamespace(to_artifact_dict=lambda: {"validated": True})
    monkeypatch.setattr(
        live_model,
        "validate_human13_live_model_plan",
        lambda plan: validation,
    )

    skeleton = SimpleNamespace(
        input_ids=(1, 2, 3),
        prompt_token_count=3,
        image_encoding=object(),
    )

    def build_skeleton(*, image_id, components, repo_root):
        calls.append(("skeleton", image_id))
        assert image_id == 1584
        assert components is not None
        return skeleton

    owner = backend_owner.Human13RPCrossoverProductionBackend(
        {"cells": ({"output_root": str(tmp_path / "node" / "cell")},)}
    )
    handle = owner.open_parity_surface(
        frozen,
        image_id=1584,
        _assembly_backend=InferenceOnlyBackend(),
        _skeleton_builder=build_skeleton,
    )

    assert handle.assembly.plan.learning_rate_resolution == "provisional_qualification"
    assert handle.assembly.plan.mixed_precision == "fp32"
    assert handle.assembly.plan.attn_implementation == "sdpa"
    assert handle.assembly.validation is validation
    assert tuple(handle.skeletons) == (1584,)
    assert handle.skeletons[1584] is skeleton
    assert not hasattr(skeleton, "owner_row_tokens")
    assert not hasattr(handle.assembly, "optimizer")
    assert not hasattr(handle.assembly, "scheduler")
    assert calls == [
        ("accelerator", "fp32"),
        ("validate_accelerator", "sdpa"),
        ("load_qwen", "fp32", "sdpa"),
        ("warm_start_dora", 16, 32),
        ("load_frozen_delta", True),
        ("requires_grad", False),
        "eval",
        ("prepare_model", True),
        ("skeleton", 1584),
    ]


def test_margin_surface_scorer_failure_releases_before_ownership_transfer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from scripts.research import build_human13_k_union_manifest as manifest_owner
    from scripts.research import human13_hf_census as census
    from scripts.research import human13_live_model as live_model
    from src.inference import hf_backend

    @dataclass(frozen=True)
    class Plan:
        mixed_precision: str = "bf16"
        attn_implementation: str = "flash_attention_2"

    @dataclass(frozen=True)
    class Components:
        model: object

    class Model:
        def eval(self) -> None:
            return None

    model_ref: weakref.ReferenceType[Model] | None = None

    class Assembly:
        def load_qwen(self, plan):
            nonlocal model_ref
            del plan
            model = Model()
            model_ref = weakref.ref(model)
            return Components(model=model)

        def warm_start_language_dora(self, model, components, plan, *, repo_root):
            del model, plan, repo_root
            return components

        def load_and_freeze_special_token_delta(
            self, model, components, plan, *, repo_root
        ):
            del model, plan, repo_root
            return components

    class Session:
        def __init__(self) -> None:
            self.loaded = None
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1
            self.loaded = None
            raise RuntimeError("secondary close failure")

    session = Session()

    def open_session(launch, *, components_loader):
        session.loaded = components_loader(launch)
        return session

    def fail_scorer(**kwargs):
        del kwargs
        raise ValueError("census scorer construction failed")

    owner = backend_owner.Human13RPCrossoverProductionBackend(
        {"cells": ({"output_root": str(tmp_path / "node" / "cell")},)}
    )
    monkeypatch.setattr(
        owner,
        "_qualification_plan",
        lambda frozen, *, learning_rate: Plan(),
    )
    monkeypatch.setattr(live_model, "DefaultHuman13AssemblyBackend", Assembly)
    monkeypatch.setattr(
        live_model,
        "build_human13_processor_skeletons",
        lambda manifest, components, *, repo_root: {},
    )
    monkeypatch.setattr(
        manifest_owner, "load_manifest", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(
        census,
        "_load_source_inputs",
        lambda repo_root: (SimpleNamespace(batch_size=1), {}),
    )
    monkeypatch.setattr(census, "Human13HFCensusScorer", fail_scorer)
    monkeypatch.setattr(hf_backend, "open_hf_backend_session", open_session)
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    frozen = SimpleNamespace(
        default_qualification_learning_rate=3.0e-6,
        manifest_path=tmp_path / "manifest.json",
    )
    with pytest.raises(ValueError, match="census scorer construction failed"):
        owner.open_margin_surface(frozen)

    gc.collect()
    assert session.close_calls == 1
    assert session.loaded is None
    assert model_ref is not None and model_ref() is None
