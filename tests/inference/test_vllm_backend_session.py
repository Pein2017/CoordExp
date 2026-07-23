from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from PIL import Image

from src.common.errors import RuntimeContractError


IM_END_ID = 151645
IMAGE_PAD_ID = 12


class FakeTokenizer:
    def __init__(self) -> None:
        self.id_to_text = {
            21: "A",
            22: "B",
            23: "C",
            IM_END_ID: "<|im_end|>",
        }
        self.decode_calls: list[dict[str, Any]] = []

    def convert_tokens_to_ids(self, token: str) -> int:
        return {
            "<|im_end|>": IM_END_ID,
            "<|image_pad|>": IMAGE_PAD_ID,
        }[token]

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        self.decode_calls.append(
            {
                "token_ids": list(token_ids),
                "skip_special_tokens": skip_special_tokens,
                "clean_up_tokenization_spaces": clean_up_tokenization_spaces,
            }
        )
        return "".join(self.id_to_text[token_id] for token_id in token_ids)


class FakeSamplingParams:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


class FakeTextPrompt:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


class FakeTokensPrompt:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


class FakeEngineCore:
    def __init__(self) -> None:
        self.shutdown_calls = 0

    def shutdown(self) -> None:
        self.shutdown_calls += 1


class FakeEngine:
    def __init__(self, *output_batches: list[Any]) -> None:
        self.output_batches = list(output_batches)
        self.calls: list[dict[str, Any]] = []
        self.core = FakeEngineCore()
        self.llm_engine = SimpleNamespace(engine_core=self.core)

    def generate(
        self,
        prompts: list[Any],
        sampling_params: FakeSamplingParams,
        *,
        use_tqdm: bool,
    ) -> list[Any]:
        self.calls.append(
            {
                "prompts": prompts,
                "sampling_params": sampling_params,
                "use_tqdm": use_tqdm,
            }
        )
        if not self.output_batches:
            raise AssertionError("unexpected native generate call")
        return self.output_batches.pop(0)


@pytest.fixture(autouse=True)
def fake_vllm_modules(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    cleanup = SimpleNamespace(model_parallel=0, distributed=0)
    vllm = ModuleType("vllm")
    vllm.SamplingParams = FakeSamplingParams
    inputs = ModuleType("vllm.inputs")
    inputs.TextPrompt = FakeTextPrompt
    inputs.TokensPrompt = FakeTokensPrompt
    distributed = ModuleType("vllm.distributed")

    def destroy_model_parallel() -> None:
        cleanup.model_parallel += 1

    def destroy_distributed_environment() -> None:
        cleanup.distributed += 1

    distributed.destroy_model_parallel = destroy_model_parallel
    distributed.destroy_distributed_environment = destroy_distributed_environment
    vllm.inputs = inputs
    vllm.distributed = distributed
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.inputs", inputs)
    monkeypatch.setitem(sys.modules, "vllm.distributed", distributed)
    return cleanup


def _launch(
    *,
    gpu_memory_utilization: float = 0.7,
    max_model_len: int = 2048,
) -> Any:
    from src.inference.backend import BackendLaunch

    return BackendLaunch(
        backend="vllm",
        model_path="/unused/materialized-model",
        model_dtype="bf16",
        batch_size=2,
        generation_config_fingerprint="generation-fingerprint",
        backend_options={
            "vllm": {
                "gpu_memory_utilization": gpu_memory_utilization,
                "max_model_len": max_model_len,
            }
        },
        execution_model_identity={
            "model_path": "/unused/materialized-model",
            "mode": "materialized",
            "composition_key": "a" * 64,
            "snapshot_fingerprint": "b" * 64,
            "composition_fidelity": {"digest": "c" * 64},
        },
    )


def _receipt(launch: Any | None = None) -> Any:
    from src.inference.backend import (
        POLICY_LIKELIHOOD_DEFINITION,
        RAW_LIKELIHOOD_DEFINITION,
        BackendSessionReceipt,
    )

    launch = launch or _launch()
    return BackendSessionReceipt(
        backend="vllm",
        backend_mode="offline_generate",
        response_family="vllm",
        backend_version="0.14.1",
        model_identity={"mode": "materialized"},
        tokenizer_identity={"sha256": "tokenizer"},
        processor_identity={"sha256": "processor"},
        generation_config_fingerprint=launch.generation_config_fingerprint,
        effective_settings={"batch_size": launch.batch_size},
        likelihood_semantics={
            "policy": POLICY_LIKELIHOOD_DEFINITION,
            "raw": RAW_LIKELIHOOD_DEFINITION,
            "score_owned_channel": "policy_logprob",
        },
        execution_model_identity=launch.execution_model_identity,
    )


def _request(
    image_path: Path,
    *,
    request_id: str = "row-1",
    prompt_ids: tuple[int, ...] = (11, 12),
    raw: bool = False,
) -> Any:
    from src.inference.backend import DecodeRequest, GenerationPolicy

    return DecodeRequest(
        request_id=request_id,
        chat_text=f"chat-{request_id}",
        input_prompt_token_ids=(11,),
        expected_executed_prompt_token_ids=prompt_ids,
        image_path=str(image_path),
        declared_image_width=2,
        declared_image_height=2,
        decoded_image_width=2,
        decoded_image_height=2,
        image_sha256=hashlib.sha256(image_path.read_bytes()).hexdigest(),
        generation_policy=GenerationPolicy(
            max_new_tokens=2,
            repetition_penalty=1.1,
            include_raw_model_logprob=raw,
        ),
        expected_image_grid_thw=(1, 1, 2),
    )


def _native_output(
    *,
    native_request_id: str = "0",
    prompt_ids: tuple[int, ...] = (11, 12),
    generated_ids: tuple[int, ...] = (21, IM_END_ID),
    policy_values: tuple[float, ...] = (-0.25, -0.5),
    finish_reason: str = "stop",
    stop_reason: str | None = None,
    native_text: str = "A",
) -> Any:
    logprobs = [
        {token_id: SimpleNamespace(logprob=value)}
        for token_id, value in zip(generated_ids, policy_values, strict=True)
    ]
    return SimpleNamespace(
        request_id=native_request_id,
        prompt_token_ids=list(prompt_ids),
        prompt_logprobs=None,
        multi_modal_placeholders={"image": [{"offset": 1, "length": 1}]},
        outputs=[
            SimpleNamespace(
                token_ids=list(generated_ids),
                text=native_text,
                finish_reason=finish_reason,
                stop_reason=stop_reason,
                logprobs=logprobs,
            )
        ],
    )


def _replay_output(
    *,
    prompt_ids: tuple[int, ...] = (11, 12),
    generated_ids: tuple[int, ...] = (21, IM_END_ID),
    raw_values: tuple[float, ...] = (-0.2, -0.4),
    native_request_id: str = "0",
) -> Any:
    full_ids = (*prompt_ids, *generated_ids)
    prompt_logprobs: list[Any] = [None] * len(prompt_ids)
    prompt_logprobs.extend(
        {
            token_id: SimpleNamespace(logprob=value),
            token_id + 1: SimpleNamespace(logprob=value - 3.0),
        }
        for token_id, value in zip(generated_ids, raw_values, strict=True)
    )
    return SimpleNamespace(
        request_id=native_request_id,
        prompt_token_ids=list(full_ids),
        prompt_logprobs=prompt_logprobs,
        outputs=[
            SimpleNamespace(
                token_ids=[23],
                text="",
                finish_reason="length",
                stop_reason=None,
                logprobs=None,
            )
        ],
    )


def _session(engine: FakeEngine, *, launch: Any | None = None) -> Any:
    from src.inference.vllm_backend import VLLMBackendSession

    launch = launch or _launch()
    return VLLMBackendSession(
        launch=launch,
        engine=engine,
        engine_factory=lambda _: engine,
        engine_kwargs={"logprobs_mode": "processed_logprobs"},
        tokenizer=FakeTokenizer(),
        receipt=_receipt(launch),
        forced_replay_processor=object,
        raw_replay_qualifier=lambda launch, identity: {
            "status": "passed",
            "processor_source_sha256": identity.get("source_sha256"),
        },
    )


def _qualification_case(tmp_path: Path) -> tuple[Any, dict[str, object], Path, str]:
    from src.inference import vllm_qualification

    base_fingerprint = "qualified-base-fingerprint"
    source_sha256 = "d" * 64
    launch = replace(
        _launch(),
        batch_size=1,
        execution_model_identity={
            **(_launch().execution_model_identity or {}),
            "source_identity": {
                "base": {"fingerprint": base_fingerprint},
            },
        },
    )
    engine_kwargs: dict[str, object] = {
        "model": launch.model_path,
        "tokenizer": launch.model_path,
        "tensor_parallel_size": 1,
        "data_parallel_size": 1,
        "dtype": "bfloat16",
        "logprobs_mode": "processed_logprobs",
        "generation_config": "vllm",
        "limit_mm_per_prompt": {"image": 1, "video": 0},
        "mm_processor_kwargs": {"do_resize": False},
        "gpu_memory_utilization": 0.7,
        "max_model_len": 2048,
        "max_num_seqs": 1,
    }
    receipt = {
        "status": "passed",
        "candidate_version": "0.14.1",
        "dependencies": {
            package: vllm_qualification.metadata.version(package)
            for package in ("vllm", "torch", "transformers", "peft", "qwen-vl-utils")
        },
        "generation": {
            "generated_token_ids": [21, IM_END_ID],
            "policy_logprobs": [-0.1, -0.2],
            "finish_reason": "stop",
        },
        "likelihood_alignment": {
            "finite_non_positive": True,
            "token_ids_aligned": True,
            "token_count": 2,
        },
        "cuda": {"available": True, "device_count": 1, "current_device": 0},
        "process": {
            "engine_process_mode": "uniprocess",
            "children_after_engine_open": [],
        },
        "cleanup": {
            "shutdown_called": True,
            "shutdown_error": None,
            "owned_children_after_cleanup": [],
        },
        "post_worker_exit": {
            "worker_returncode": 0,
            "worker_pid_alive_after_exit": False,
            "gpu_memory_returned_to_baseline": True,
            "children_after": [],
        },
        "probe": {
            "path": str(vllm_qualification.QUALIFICATION_PROBE.resolve()),
            "repo_relative_path": (
                vllm_qualification.QUALIFICATION_PROBE_RELATIVE_PATH.as_posix()
            ),
            "sha256": source_sha256,
        },
        "qualification_scope": {
            "source_base_snapshot_fingerprint": base_fingerprint,
        },
        "engine": {
            "qualification_argument_policy": {
                "semantic_invariants": {
                    field: engine_kwargs[field]
                    for field in (
                        "tensor_parallel_size",
                        "data_parallel_size",
                        "dtype",
                        "logprobs_mode",
                        "generation_config",
                        "limit_mm_per_prompt",
                        "mm_processor_kwargs",
                    )
                },
                "qualified_exact_values": {
                    field: engine_kwargs[field]
                    for field in ("gpu_memory_utilization", "max_model_len")
                },
            },
        },
        "installed_sources": {
            "engine_args": {
                "package": "vllm",
                "relative_path": "vllm/engine/arg_utils.py",
                "path": "/qualified/vllm/engine/arg_utils.py",
                "sha256": source_sha256,
            },
            "processor": {
                "package": "vllm",
                "relative_path": "vllm/multimodal/processing.py",
                "path": "/qualified/vllm/multimodal/processing.py",
                "sha256": source_sha256,
            },
        },
        "loaded_runtime_sources": {
            "file_count": 1,
            "files": [
                {
                    "package": "vllm",
                    "relative_path": "vllm/lazy_runtime.py",
                    "path": "/qualified/vllm/lazy_runtime.py",
                    "sha256": source_sha256,
                    "modules": ["vllm.lazy_runtime"],
                }
            ],
            "required_paths": ["vllm/lazy_runtime.py"],
            "required_paths_present": True,
            "fingerprint": vllm_qualification._sha256_json(
                [
                    {
                        "package": "vllm",
                        "relative_path": "vllm/lazy_runtime.py",
                        "sha256": source_sha256,
                        "modules": ["vllm.lazy_runtime"],
                    }
                ]
            ),
        },
    }
    receipt_path = tmp_path / "vllm-qualification.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    (tmp_path / "vllm-0.14.1-application-sources.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "version": "coordexp-swift-vllm-application-sources-v1",
                "source_sha256": {
                    path: source_sha256
                    for path in vllm_qualification.APPLICATION_EXECUTION_SOURCE_PATHS
                },
            }
        ),
        encoding="utf-8",
    )
    return launch, engine_kwargs, receipt_path, source_sha256


def test_vllm_qualification_repo_sources_resolve_from_active_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.inference import vllm_qualification

    alternate_root = tmp_path / "alternate-checkout"
    config_path = alternate_root / "configs" / "infer.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("backend: hf\n", encoding="utf-8")
    expected_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    monkeypatch.setattr(vllm_qualification, "_REPO_ROOT", alternate_root)

    vllm_qualification._validate_config_sources(
        {
            "sources": [
                {
                    "path": "/obsolete/checkout/configs/infer.yaml",
                    "repo_relative_path": "configs/infer.yaml",
                    "sha256": expected_sha256,
                }
            ]
        }
    )


def test_vllm_qualification_rejects_absolute_only_repo_source_identity() -> None:
    from src.inference import vllm_qualification

    with pytest.raises(RuntimeContractError) as exc_info:
        vllm_qualification._validate_config_sources(
            {
                "sources": [
                    {
                        "path": "/obsolete/checkout/configs/infer.yaml",
                        "sha256": "a" * 64,
                    }
                ]
            }
        )

    assert exc_info.value.code == "vllm_backend.concurrency_qualification_source_drift"


def test_open_vllm_backend_session_validates_launch_and_engine_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.inference import vllm_backend

    launch = _launch(gpu_memory_utilization=1.0, max_model_len=4096)
    captured: dict[str, Any] = {}
    engine = FakeEngine()

    def engine_factory(kwargs: Any) -> FakeEngine:
        captured.update(kwargs)
        return engine

    components = SimpleNamespace(
        tokenizer=FakeTokenizer(),
        processor=object(),
        token_identity={"sha256": "tokenizer"},
        processor_identity={"sha256": "processor"},
    )
    monkeypatch.setattr(vllm_backend.metadata, "version", lambda _: "0.14.1")
    monkeypatch.setattr(vllm_backend, "_validate_rank_local_cuda", lambda: None)

    session = vllm_backend.open_vllm_backend_session(
        launch,
        engine_factory=engine_factory,
        components_loader=lambda _: components,
    )

    assert captured["tensor_parallel_size"] == 1
    assert captured["data_parallel_size"] == 1
    assert captured["max_num_seqs"] == launch.batch_size
    assert captured["gpu_memory_utilization"] == pytest.approx(1.0)
    assert captured["max_model_len"] == 4096
    assert captured["logprobs_mode"] == "processed_logprobs"
    assert captured["mm_processor_kwargs"] == {"do_resize": False}
    assert session.receipt.execution_model_identity == launch.execution_model_identity
    session.close()


def test_open_vllm_backend_session_rejects_wrong_backend_or_missing_execution_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.inference import vllm_backend

    launch = _launch()
    monkeypatch.setattr(vllm_backend.metadata, "version", lambda _: "0.14.1")
    monkeypatch.setattr(vllm_backend, "_validate_rank_local_cuda", lambda: None)
    attempts = 0

    def engine_factory(_: Any) -> Any:
        nonlocal attempts
        attempts += 1
        raise AssertionError("invalid launch reached engine construction")

    with pytest.raises(RuntimeContractError) as wrong_backend:
        vllm_backend.open_vllm_backend_session(
            replace(launch, backend="hf"), engine_factory=engine_factory
        )
    assert wrong_backend.value.code == "vllm_backend.launch_backend"

    with pytest.raises(RuntimeContractError) as missing_execution_model:
        vllm_backend.open_vllm_backend_session(
            replace(launch, execution_model_identity=None),
            engine_factory=engine_factory,
        )
    assert missing_execution_model.value.code == "vllm_backend.execution_model_required"
    assert attempts == 0


def test_open_vllm_backend_session_records_unverified_version_and_starts_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.inference import vllm_backend

    monkeypatch.setattr(vllm_backend.metadata, "version", lambda _: "0.15.0")
    monkeypatch.setattr(vllm_backend, "_validate_rank_local_cuda", lambda: None)
    engine = FakeEngine()
    components = SimpleNamespace(
        tokenizer=FakeTokenizer(),
        processor=object(),
        token_identity={"sha256": "tokenizer"},
        processor_identity={"sha256": "processor"},
    )

    session = vllm_backend.open_vllm_backend_session(
        _launch(),
        engine_factory=lambda _: engine,
        components_loader=lambda _: components,
    )

    assert session.receipt.effective_settings["runtime_preflight"]["version"] == {
        "observed_version": "0.15.0",
        "status": "unverified",
        "known_working_versions": ["0.14.1"],
    }
    session.close()


def test_unverified_vllm_version_rejects_raw_trace_but_allows_policy_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.inference import vllm_backend

    image_path = tmp_path / "image.png"
    Image.new("RGB", (2, 2), color="white").save(image_path)
    engine = FakeEngine([_native_output()])
    components = SimpleNamespace(
        tokenizer=FakeTokenizer(),
        processor=object(),
        token_identity={"sha256": "tokenizer"},
        processor_identity={"sha256": "processor"},
    )
    monkeypatch.setattr(vllm_backend.metadata, "version", lambda _: "0.15.0")
    monkeypatch.setattr(vllm_backend, "_validate_rank_local_cuda", lambda: None)
    session = vllm_backend.open_vllm_backend_session(
        _launch(),
        engine_factory=lambda _: engine,
        components_loader=lambda _: components,
    )

    with pytest.raises(RuntimeContractError) as exc_info:
        session.decode([_request(image_path, raw=True)])

    assert exc_info.value.code == "vllm_backend.raw_replay_version_unverified"
    session.close()


def test_first_live_decode_promotes_operational_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.inference import vllm_backend

    image_path = tmp_path / "image.png"
    Image.new("RGB", (2, 2), color="white").save(image_path)
    engine = FakeEngine([_native_output()])
    components = SimpleNamespace(
        tokenizer=FakeTokenizer(),
        processor=object(),
        token_identity={"sha256": "tokenizer"},
        processor_identity={"sha256": "processor"},
    )
    monkeypatch.setattr(vllm_backend.metadata, "version", lambda _: "0.14.1")
    monkeypatch.setattr(vllm_backend, "_validate_rank_local_cuda", lambda: None)
    session = vllm_backend.open_vllm_backend_session(
        _launch(),
        engine_factory=lambda _: engine,
        components_loader=lambda _: components,
    )

    request = _request(image_path)
    session.decode([request])

    preflight = session.receipt.effective_settings["runtime_preflight"]
    assert preflight["status"] == "passed_live_decode"
    assert preflight["live_decode"]["request_count"] == 1
    assert preflight["live_decode"]["first_request_id"] == request.request_id
    session.close()
    preflight = session.receipt.effective_settings["runtime_preflight"]
    assert preflight["status"] == "passed_live_decode_and_cleanup"
    assert preflight["cleanup"]["status"] == "completed"
    assert preflight["cleanup"]["events"][-1]["scope"] == "session_close"


def test_operational_preflight_demotes_historical_source_and_argument_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.inference import vllm_qualification

    launch, engine_kwargs, receipt_path, source_sha256 = _qualification_case(tmp_path)
    engine_kwargs["gpu_memory_utilization"] = 0.4
    monkeypatch.setattr(
        vllm_qualification,
        "_sha256_file",
        lambda path: "e" * 64 if path.name == "vllm_backend.py" else source_sha256,
    )

    result = vllm_qualification.inspect_vllm_operational_preflight(
        launch=launch,
        engine_kwargs=engine_kwargs,
        observed_version="0.14.1",
        application_receipt_path=(tmp_path / "vllm-0.14.1-application-sources.json"),
    )

    assert result["status"] == "ready_for_engine_construction"
    assert result["engine_settings"]["gpu_memory_utilization"] == pytest.approx(0.4)
    assert result["historical_application_sources"]["status"] == "stale"
    assert (
        result["historical_application_sources"]["error"]["code"]
        == "vllm_backend.application_qualification_source_drift"
    )


def test_raw_replay_preflight_treats_missing_historical_receipt_as_diagnostic(
    tmp_path: Path,
) -> None:
    from src.inference import vllm_qualification

    result = vllm_qualification.inspect_vllm_raw_replay_preflight(
        launch=_launch(),
        processor_identity={
            "module": "src.inference.vllm_forced_replay",
            "qualname": "CoordExpForcedSequenceLogitsProcessor",
            "source_sha256": "f" * 64,
        },
        receipt_path=tmp_path / "missing.json",
    )

    assert result["status"] == "ready_for_live_replay"
    assert result["historical_qualification"]["status"] == "unavailable"
    assert result["processor_identity"]["source_sha256"] == "f" * 64


def test_open_vllm_backend_session_rejects_inherited_multiprocessing_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.inference import vllm_backend

    monkeypatch.setattr(vllm_backend.metadata, "version", lambda _: "0.14.1")
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "1")
    attempts = 0

    def engine_factory(_: Any) -> Any:
        nonlocal attempts
        attempts += 1
        raise AssertionError("invalid process mode reached engine construction")

    with pytest.raises(RuntimeContractError) as exc_info:
        vllm_backend.open_vllm_backend_session(
            _launch(),
            engine_factory=engine_factory,
        )

    assert exc_info.value.code == "vllm_backend.process_mode"
    assert exc_info.value.context == {"VLLM_ENABLE_V1_MULTIPROCESSING": "1"}
    assert attempts == 0


def test_validate_vllm_runtime_qualification_accepts_matching_receipt_and_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.inference import vllm_qualification

    launch, engine_kwargs, receipt_path, source_sha256 = _qualification_case(tmp_path)
    monkeypatch.setattr(
        vllm_qualification,
        "_sha256_file",
        lambda _: source_sha256,
    )

    result = vllm_qualification.validate_vllm_runtime_qualification(
        launch=launch,
        engine_kwargs=engine_kwargs,
        receipt_path=receipt_path,
    )

    assert result["status"] == "passed"
    assert result["candidate_version"] == "0.14.1"
    assert result["source_base_snapshot_fingerprint"] == (
        "qualified-base-fingerprint"
    )
    assert result["max_num_seqs"] == 1
    assert result["receipt_sha256"] == hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    assert result["application_qualification"]["source_count"] == len(
        vllm_qualification.APPLICATION_EXECUTION_SOURCE_PATHS
    )


def test_validate_vllm_runtime_qualification_rejects_incomplete_application_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.inference import vllm_qualification

    launch, engine_kwargs, receipt_path, source_sha256 = _qualification_case(tmp_path)
    application_path = tmp_path / "vllm-0.14.1-application-sources.json"
    payload = json.loads(application_path.read_text(encoding="utf-8"))
    payload["source_sha256"].pop("src/inference/pipeline.py")
    application_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(vllm_qualification, "_sha256_file", lambda _: source_sha256)

    with pytest.raises(RuntimeContractError) as exc_info:
        vllm_qualification.validate_vllm_runtime_qualification(
            launch=launch,
            engine_kwargs=engine_kwargs,
            receipt_path=receipt_path,
        )

    assert exc_info.value.code == "vllm_backend.application_qualification_receipt"
    assert exc_info.value.context["missing_paths"] == ["src/inference/pipeline.py"]


def test_validate_vllm_runtime_qualification_rejects_missing_runtime_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.inference import vllm_qualification

    launch, engine_kwargs, receipt_path, source_sha256 = _qualification_case(tmp_path)
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    payload.pop("generation")
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(vllm_qualification, "_sha256_file", lambda _: source_sha256)

    with pytest.raises(RuntimeContractError) as exc_info:
        vllm_qualification.validate_vllm_runtime_qualification(
            launch=launch,
            engine_kwargs=engine_kwargs,
            receipt_path=receipt_path,
        )

    assert exc_info.value.code == "vllm_backend.qualification_receipt"
    assert exc_info.value.context["field"] == "generation"


def test_validate_vllm_runtime_qualification_accepts_executed_concurrency_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.inference import vllm_qualification

    launch, engine_kwargs, receipt_path, source_sha256 = _qualification_case(tmp_path)
    application_path = tmp_path / "vllm-0.14.1-application-sources.json"
    application_sha256 = hashlib.sha256(application_path.read_bytes()).hexdigest()
    launch = replace(launch, batch_size=4)
    engine_kwargs["max_num_seqs"] = 4
    concurrency_receipt = {
        "status": "passed",
        "version": "coordexp-swift-vllm-concurrency-qualification-v1",
        "vllm_version": "0.14.1",
        "max_num_seqs": 4,
        "request_count": 4,
        "requests": [{"request_id": f"row-{index}"} for index in range(4)],
        "backend_session_engine_kwargs": dict(engine_kwargs),
        "runtime_qualification": {
            "baseline": {
                "receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
                "application_qualification": {
                    "status": "passed",
                    "receipt_sha256": application_sha256,
                },
            },
        },
        "execution_model": {
            "source_base_snapshot_fingerprint": "qualified-base-fingerprint",
        },
        "probe_source_sha256": source_sha256,
            "config": {
                "sources": [
                    {
                        "path": "/qualified/config.yaml",
                        "repo_relative_path": "configs/qualified.yaml",
                        "sha256": source_sha256,
                    }
                ],
            },
    }
    concurrency_path = tmp_path / "vllm-concurrency.json"
    concurrency_path.write_text(json.dumps(concurrency_receipt), encoding="utf-8")
    monkeypatch.setattr(vllm_qualification, "_sha256_file", lambda _: source_sha256)

    result = vllm_qualification.validate_vllm_runtime_qualification(
        launch=launch,
        engine_kwargs=engine_kwargs,
        receipt_path=receipt_path,
        concurrency_receipt_path=concurrency_path,
    )

    assert result["status"] == "passed"
    assert result["max_num_seqs"] == 4
    assert result["concurrency_qualification"]["status"] == "passed"
    assert result["concurrency_qualification"]["max_num_seqs"] == 4

    derivative_identity = dict(launch.execution_model_identity or {})
    derivative_identity.update(
        composition_key="different-composition",
        snapshot_fingerprint="different-snapshot",
        receipt_fingerprint="different-receipt",
    )
    derivative = vllm_qualification.validate_vllm_runtime_qualification(
        launch=replace(launch, execution_model_identity=derivative_identity),
        engine_kwargs=engine_kwargs,
        receipt_path=receipt_path,
        concurrency_receipt_path=concurrency_path,
    )
    assert derivative["concurrency_qualification"]["status"] == "passed"

    concurrency_receipt["runtime_qualification"]["baseline"][
        "application_qualification"
    ]["receipt_sha256"] = "0" * 64
    concurrency_path.write_text(json.dumps(concurrency_receipt), encoding="utf-8")
    with pytest.raises(RuntimeContractError) as exc_info:
        vllm_qualification.validate_vllm_runtime_qualification(
            launch=launch,
            engine_kwargs=engine_kwargs,
            receipt_path=receipt_path,
            concurrency_receipt_path=concurrency_path,
        )
    assert exc_info.value.code == "vllm_backend.concurrency_qualification_receipt"


def test_validate_vllm_runtime_qualification_rejects_missing_concurrency_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.inference import vllm_qualification

    launch, engine_kwargs, receipt_path, source_sha256 = _qualification_case(tmp_path)
    launch = replace(launch, batch_size=4)
    engine_kwargs["max_num_seqs"] = 4
    monkeypatch.setattr(vllm_qualification, "_sha256_file", lambda _: source_sha256)

    with pytest.raises(RuntimeContractError) as exc_info:
        vllm_qualification.validate_vllm_runtime_qualification(
            launch=launch,
            engine_kwargs=engine_kwargs,
            receipt_path=receipt_path,
            concurrency_receipt_path=tmp_path / "missing-concurrency.json",
        )

    assert exc_info.value.code == "vllm_backend.concurrency_qualification_receipt"


def test_validate_vllm_forced_replay_qualification_binds_processor_and_base(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.inference import vllm_qualification

    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text('{"status":"passed"}\n', encoding="utf-8")
    application_path = tmp_path / "vllm-0.14.1-application-sources.json"
    application_path.write_text('{"status":"passed"}\n', encoding="utf-8")
    application_sha256 = hashlib.sha256(application_path.read_bytes()).hexdigest()
    source_sha256 = "d" * 64
    processor_identity = {
        "module": "src.inference.vllm_forced_replay",
        "qualname": "CoordExpForcedSequenceLogitsProcessor",
        "source_path": "/repo/src/inference/vllm_forced_replay.py",
        "source_sha256": "f" * 64,
    }
    rows = [
        {
            "row_id": "row-0",
            "status": "verified",
            "prompt_token_count": 4,
            "prompt_token_ids_sha256": "a" * 64,
            "generated_token_count": 2,
            "generated_token_ids_sha256": "b" * 64,
            "finish_reason": "stop",
            "native_stop_reason": None,
        }
    ]
    row_map = {
        "row-0": {key: value for key, value in rows[0].items() if key != "row_id"}
    }
    payload = {
        "status": "passed",
        "version": "coordexp-swift-vllm-concurrency-qualification-v1",
        "vllm_version": "0.14.1",
        "max_num_seqs": 1,
        "request_count": 1,
        "runtime_qualification": {
                "baseline": {
                    "receipt_sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
                    "application_qualification": {
                        "status": "passed",
                        "receipt_sha256": application_sha256,
                    },
                }
        },
        "probe_source_sha256": source_sha256,
        "config": {
            "sources": [
                {
                    "path": "/config.yaml",
                    "repo_relative_path": "configs/qualified.yaml",
                    "sha256": source_sha256,
                }
            ]
        },
        "execution_model": {
            "source_base_snapshot_fingerprint": "qualified-base-fingerprint"
        },
        "raw_replay": {
            "settings": {
                "status": "completed",
                "logprobs_mode": "raw_logprobs",
                "max_num_seqs": 1,
                "request_count": 1,
                    "row_evidence_sha256": hashlib.sha256(
                        json.dumps(row_map, sort_keys=True, separators=(",", ":")).encode()
                    ).hexdigest(),
                    "forced_logits_processor": processor_identity,
                    "qualification": {
                        "status": "passed",
                        "evidence": "executed_by_this_receipt",
                        "probe_source_sha256": source_sha256,
                        "source_base_snapshot_fingerprint": (
                            "qualified-base-fingerprint"
                        ),
                        "processor_source_sha256": processor_identity[
                            "source_sha256"
                        ],
                    },
                },
            "rows": rows,
            "row_evidence_sha256": hashlib.sha256(
                json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        },
    }
    receipt_path = tmp_path / "forced-replay.json"
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(vllm_qualification, "_sha256_file", lambda _: source_sha256)
    launch = replace(
        _launch(),
        batch_size=1,
        execution_model_identity={
            **(_launch().execution_model_identity or {}),
            "source_identity": {
                "base": {"fingerprint": "qualified-base-fingerprint"}
            },
        },
    )

    result = vllm_qualification.validate_vllm_forced_replay_qualification(
        launch=launch,
        processor_identity=processor_identity,
        receipt_path=receipt_path,
        baseline_receipt_path=baseline_path,
    )

    assert result["status"] == "passed"
    assert result["processor_source_sha256"] == "f" * 64
    relocated = vllm_qualification.validate_vllm_forced_replay_qualification(
        launch=launch,
        processor_identity={
            **processor_identity,
            "source_path": "/equivalent/checkout/src/inference/vllm_forced_replay.py",
        },
        receipt_path=receipt_path,
        baseline_receipt_path=baseline_path,
    )
    assert relocated["status"] == "passed"
    with pytest.raises(RuntimeContractError) as exc_info:
        vllm_qualification.validate_vllm_forced_replay_qualification(
            launch=launch,
            processor_identity={**processor_identity, "source_sha256": "e" * 64},
            receipt_path=receipt_path,
            baseline_receipt_path=baseline_path,
        )
    assert exc_info.value.code == "vllm_backend.raw_replay_qualification_processor_drift"
    with pytest.raises(RuntimeContractError) as exc_info:
        vllm_qualification.validate_vllm_forced_replay_qualification(
            launch=replace(launch, batch_size=2),
            processor_identity=processor_identity,
            receipt_path=receipt_path,
            baseline_receipt_path=baseline_path,
        )
    assert exc_info.value.code == "vllm_backend.raw_replay_qualification_invalid"


@pytest.mark.parametrize(
    ("drift", "expected_code"),
    [
        ("model", "vllm_backend.qualification_model_family"),
        ("engine", "vllm_backend.qualification_engine_argument"),
        ("source", "vllm_backend.qualification_source_drift"),
        ("loaded_source", "vllm_backend.qualification_loaded_source_drift"),
        ("application_source", "vllm_backend.application_qualification_source_drift"),
        ("probe", "vllm_backend.qualification_probe_drift"),
    ],
)
def test_validate_vllm_runtime_qualification_rejects_runtime_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
    expected_code: str,
) -> None:
    from src.inference import vllm_qualification

    launch, engine_kwargs, receipt_path, source_sha256 = _qualification_case(tmp_path)
    if drift == "model":
        execution_identity = dict(launch.execution_model_identity or {})
        execution_identity["source_identity"] = {
            "base": {"fingerprint": "different-base-fingerprint"},
        }
        launch = replace(launch, execution_model_identity=execution_identity)
    elif drift == "engine":
        engine_kwargs["dtype"] = "float16"
    def observed_source_sha256(path: Path) -> str:
        is_probe = path.resolve() == vllm_qualification.QUALIFICATION_PROBE.resolve()
        if drift == "probe" and is_probe:
            return "e" * 64
        if drift == "loaded_source" and path.name == "lazy_runtime.py":
            return "e" * 64
        if drift == "application_source" and path.name == "vllm_backend.py":
            return "e" * 64
        if drift == "source" and not is_probe:
            return "e" * 64
        return source_sha256

    monkeypatch.setattr(
        vllm_qualification,
        "_sha256_file",
        observed_source_sha256,
    )

    with pytest.raises(RuntimeContractError) as exc_info:
        vllm_qualification.validate_vllm_runtime_qualification(
            launch=launch,
            engine_kwargs=engine_kwargs,
            receipt_path=receipt_path,
        )

    assert exc_info.value.code == expected_code


def test_vllm_session_reorders_multi_request_results_and_extracts_chosen_logprobs(
    tmp_path: Path,
) -> None:
    image_paths = [tmp_path / "first.png", tmp_path / "second.png"]
    for image_path in image_paths:
        Image.new("RGB", (2, 2), color="white").save(image_path)
    engine = FakeEngine(
        [
            _native_output(
                native_request_id="1",
                prompt_ids=(13, IMAGE_PAD_ID),
                generated_ids=(22, IM_END_ID),
                policy_values=(-0.3, -0.6),
                native_text="B",
            ),
            _native_output(native_request_id="0"),
        ]
    )

    session = _session(engine)
    results = session.decode(
        [
            _request(image_paths[0], request_id="row-first"),
            _request(
                image_paths[1],
                request_id="row-second",
                prompt_ids=(13, IMAGE_PAD_ID),
            ),
        ]
    )

    assert [result.request_id for result in results] == ["row-first", "row-second"]
    assert results[0].generated_token_ids == (21, IM_END_ID)
    assert results[1].generated_token_ids == (22, IM_END_ID)
    assert [trace.policy_logprob for trace in results[1].token_trace] == [
        pytest.approx(-0.3),
        pytest.approx(-0.6),
    ]
    assert len(engine.calls) == 1
    assert all(isinstance(prompt, FakeTextPrompt) for prompt in engine.calls[0]["prompts"])
    assert engine.calls[0]["sampling_params"].logprobs == 0
    assert engine.calls[0]["use_tqdm"] is False
    performance = session.receipt.effective_settings["performance"]
    assert performance["request_count"] == 2
    assert performance["generated_token_count"] == 4
    assert performance["decode_elapsed_seconds"] > 0


def test_vllm_session_reconstructs_text_and_normalizes_retained_im_end_stop(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (2, 2), color="white").save(image_path)
    engine = FakeEngine([_native_output(stop_reason=None, native_text="A")])

    result = _session(engine).decode([_request(image_path)])[0]

    assert result.generated_token_ids == (21, IM_END_ID)
    assert result.native_generated_text == "A"
    assert result.raw_generated_text == "A<|im_end|>"
    assert result.parser_text == "A"
    assert result.strip_policy == "terminal_im_end"
    assert result.stop_reason == "im_end"
    assert result.token_trace[-1].token_text == "<|im_end|>"
    assert result.token_trace[-1].is_stop is True


def test_vllm_session_accepts_empty_native_placeholders_with_one_prompt_pad_range(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (2, 2), color="white").save(image_path)
    prompt_ids = (11, IMAGE_PAD_ID, IMAGE_PAD_ID, 13)
    native = _native_output(prompt_ids=prompt_ids)
    native.multi_modal_placeholders = {}

    result = _session(FakeEngine([native])).decode(
        [_request(image_path, prompt_ids=prompt_ids)]
    )[0]

    assert result.native_evidence["image_placeholder_ranges"] == [
        {"offset": 1, "length": 2}
    ]
    assert result.native_evidence["backend_reported_multi_modal_placeholders"] == {}


@pytest.mark.parametrize(
    "prompt_ids",
    [
        (11, 13),
        (IMAGE_PAD_ID, 11, IMAGE_PAD_ID),
    ],
    ids=("missing", "non-contiguous"),
)
def test_vllm_session_rejects_missing_or_malformed_prompt_image_pad_evidence(
    tmp_path: Path,
    prompt_ids: tuple[int, ...],
) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (2, 2), color="white").save(image_path)
    native = _native_output(prompt_ids=prompt_ids)
    native.multi_modal_placeholders = {}

    with pytest.raises(RuntimeContractError) as exc_info:
        _session(FakeEngine([native])).decode(
            [_request(image_path, prompt_ids=prompt_ids)]
        )

    assert exc_info.value.code == "vllm_backend.image_placeholder_evidence"


@pytest.mark.parametrize("placeholders", [None, {"image": [{"offset": 1}] }])
def test_vllm_session_rejects_missing_or_malformed_native_placeholder_evidence(
    tmp_path: Path,
    placeholders: object,
) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (2, 2), color="white").save(image_path)
    native = _native_output()
    native.multi_modal_placeholders = placeholders

    with pytest.raises(RuntimeContractError) as exc_info:
        _session(FakeEngine([native])).decode([_request(image_path)])

    assert exc_info.value.code == "vllm_backend.image_placeholder_evidence"
    assert exc_info.value.context["request_id"] == "row-1"


def test_vllm_session_rejects_non_null_native_stop_reason(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (2, 2), color="white").save(image_path)

    with pytest.raises(RuntimeContractError) as exc_info:
        _session(FakeEngine([_native_output(stop_reason=IM_END_ID)])).decode(
            [_request(image_path)]
        )

    assert exc_info.value.code == "vllm_backend.native_stop_reason"
    assert exc_info.value.context["request_id"] == "row-1"


def test_vllm_session_rejects_native_completion_text_drift(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (2, 2), color="white").save(image_path)

    with pytest.raises(RuntimeContractError) as exc_info:
        _session(FakeEngine([_native_output(native_text="drifted")])).decode(
            [_request(image_path)]
        )

    assert exc_info.value.code == "vllm_backend.native_text_mismatch"
    assert exc_info.value.context["request_id"] == "row-1"


def test_vllm_session_fails_fast_on_prompt_id_mismatch(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (2, 2), color="white").save(image_path)
    engine = FakeEngine([_native_output(prompt_ids=(11, 99))])

    with pytest.raises(RuntimeContractError) as exc_info:
        _session(engine).decode([_request(image_path)])

    assert exc_info.value.code == "vllm_backend.prompt_token_mismatch"
    assert exc_info.value.context["request_id"] == "row-1"


def test_vllm_session_fails_before_native_generation_when_image_identity_changes(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (2, 2), color="white").save(image_path)
    request = _request(image_path)
    Image.new("RGB", (2, 2), color="black").save(image_path)
    engine = FakeEngine()

    with pytest.raises(RuntimeContractError) as exc_info:
        _session(engine).decode([request])

    assert exc_info.value.code == "vllm_backend.image_sha256_mismatch"
    assert exc_info.value.context["request_id"] == "row-1"
    assert engine.calls == []


def test_vllm_session_replays_raw_likelihoods_in_forced_decode_mode(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (2, 2), color="white").save(image_path)
    engine = FakeEngine(
        [_native_output()],
        [_native_output(policy_values=(-0.2, -0.4))],
    )

    session = _session(engine)
    result = session.decode([_request(image_path, raw=True)])[0]

    assert [trace.raw_model_logprob for trace in result.token_trace] == [
        pytest.approx(-0.2),
        pytest.approx(-0.4),
    ]
    assert len(engine.calls) == 2
    replay_call = engine.calls[1]
    assert isinstance(replay_call["sampling_params"], list)
    assert replay_call["sampling_params"][0].logprobs == 0
    assert replay_call["sampling_params"][0].extra_args == {
        "coordexp_expected_token_ids": [21, IM_END_ID]
    }
    assert replay_call["prompts"][0].prompt == "chat-row-1"
    assert 23 not in result.generated_token_ids
    assert result.native_evidence["raw_replay"]["status"] == "verified"
    assert result.native_evidence["raw_replay"]["prompt_token_count"] == 2
    assert result.observed_image_grid_thw is None
    raw_receipt = session.receipt.effective_settings["raw_replay"]
    assert raw_receipt["status"] == "completed"
    assert raw_receipt["logprobs_mode"] == "raw_logprobs"
    assert raw_receipt["request_count"] == 1
    assert len(raw_receipt["row_evidence_sha256"]) == 64


def test_raw_enabled_session_recreates_processed_engine_before_second_decode(
    tmp_path: Path,
) -> None:
    from src.inference.vllm_backend import VLLMBackendSession

    image_path = tmp_path / "image.png"
    Image.new("RGB", (2, 2), color="white").save(image_path)
    launch = _launch()
    first_processed = FakeEngine([_native_output(policy_values=(-0.1, -0.2))])
    second_processed = FakeEngine([_native_output(policy_values=(-0.3, -0.4))])
    raw_engines = [
        FakeEngine([_native_output(policy_values=(-1.1, -1.2))]),
        FakeEngine([_native_output(policy_values=(-1.3, -1.4))]),
    ]
    processed_engines = [second_processed]

    def engine_factory(kwargs: Any) -> FakeEngine:
        if kwargs["logprobs_mode"] == "raw_logprobs":
            return raw_engines.pop(0)
        return processed_engines.pop(0)

    session = VLLMBackendSession(
        launch=launch,
        engine=first_processed,
        engine_factory=engine_factory,
        engine_kwargs={"logprobs_mode": "processed_logprobs"},
        tokenizer=FakeTokenizer(),
        receipt=_receipt(launch),
        forced_replay_processor=object,
        raw_replay_qualifier=lambda launch, identity: {"status": "passed"},
    )

    first = session.decode([_request(image_path, raw=True)])[0]
    second = session.decode([_request(image_path, raw=True)])[0]

    assert first.token_trace[0].policy_logprob == pytest.approx(-0.1)
    assert first.token_trace[0].raw_model_logprob == pytest.approx(-1.1)
    assert second.token_trace[0].policy_logprob == pytest.approx(-0.3)
    assert second.token_trace[0].raw_model_logprob == pytest.approx(-1.3)
    assert len(first_processed.calls) == 1
    assert len(second_processed.calls) == 1


def test_raw_replay_engine_uses_raw_mode_forcing_and_bounded_kv_cache() -> None:
    from src.inference.vllm_backend import _raw_replay_engine_kwargs

    processor = type("Processor", (), {})
    kwargs = _raw_replay_engine_kwargs(
        {
            "model": "/model",
            "logprobs_mode": "processed_logprobs",
            "max_num_seqs": 4,
        },
        forced_replay_processor=processor,
    )

    assert kwargs["model"] == "/model"
    assert kwargs["max_num_seqs"] == 4
    assert kwargs["logprobs_mode"] == "raw_logprobs"
    assert kwargs["logits_processors"] == [processor]
    assert kwargs["gpu_memory_utilization"] == 0.20
    assert kwargs["kv_cache_memory_bytes"] == 1024**3


def test_vllm_session_skips_raw_replay_when_disabled(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (2, 2), color="white").save(image_path)
    engine = FakeEngine([_native_output()])

    result = _session(engine).decode([_request(image_path, raw=False)])[0]

    assert len(engine.calls) == 1
    assert all(trace.raw_model_logprob is None for trace in result.token_trace)


def test_vllm_session_uses_launch_max_model_len_for_generation_guard(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (2, 2), color="white").save(image_path)
    session = _session(FakeEngine(), launch=_launch(max_model_len=4096))

    with pytest.raises(RuntimeContractError) as exc_info:
        session.decode([_request(image_path, prompt_ids=(1,) * 4095)])

    assert exc_info.value.code == "vllm_backend.model_length"
    assert exc_info.value.context == {
        "prompt_tokens": 4095,
        "max_new_tokens": 2,
        "max_model_len": 4096,
    }


def test_vllm_session_rejects_shifted_raw_replay_alignment(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (2, 2), color="white").save(image_path)
    shifted = _native_output()
    shifted.prompt_token_ids = [11, 99]
    engine = FakeEngine([_native_output()], [shifted])

    with pytest.raises(RuntimeContractError) as exc_info:
        _session(engine).decode([_request(image_path, raw=True)])

    assert exc_info.value.code == "vllm_backend.raw_replay_prompt_mismatch"
    assert exc_info.value.context["request_id"] == "row-1"


def test_vllm_session_rejects_raw_replay_stop_drift(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (2, 2), color="white").save(image_path)
    replay = _native_output(finish_reason="length")
    engine = FakeEngine([_native_output()], [replay])

    with pytest.raises(RuntimeContractError) as exc_info:
        _session(engine).decode([_request(image_path, raw=True)])

    assert exc_info.value.code == "vllm_backend.raw_replay_stop_mismatch"


def test_vllm_session_rejects_raw_replay_native_request_id_drift(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (2, 2), color="white").save(image_path)
    generation = _native_output(native_request_id="4")
    replay = _native_output(native_request_id="0")
    engine = FakeEngine([generation], [replay])

    with pytest.raises(RuntimeContractError) as exc_info:
        _session(engine).decode([_request(image_path, raw=True)])

    assert exc_info.value.code == "vllm_backend.raw_replay_request_id_mismatch"


@pytest.mark.parametrize(
    ("positions", "expected_code"),
    [
        (None, "vllm_backend.likelihood_alignment"),
        (
            [{22: SimpleNamespace(logprob=-0.2)}, {IM_END_ID: SimpleNamespace(logprob=-0.4)}],
            "vllm_backend.likelihood_token_missing",
        ),
        (
            [{21: SimpleNamespace(logprob=math.nan)}, {IM_END_ID: SimpleNamespace(logprob=-0.4)}],
            "vllm_backend.likelihood_value",
        ),
        (
            [{21: SimpleNamespace(logprob=math.inf)}, {IM_END_ID: SimpleNamespace(logprob=-0.4)}],
            "vllm_backend.likelihood_value",
        ),
        (
            [{21: SimpleNamespace(logprob=0.1)}, {IM_END_ID: SimpleNamespace(logprob=-0.4)}],
            "vllm_backend.likelihood_value",
        ),
    ],
    ids=("missing", "chosen-token-missing", "nan", "infinite", "positive"),
)
def test_vllm_session_rejects_invalid_processed_likelihood_evidence(
    tmp_path: Path,
    positions: Any,
    expected_code: str,
) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (2, 2), color="white").save(image_path)
    native = _native_output()
    native.outputs[0].logprobs = positions

    with pytest.raises(RuntimeContractError) as exc_info:
        _session(FakeEngine([native])).decode([_request(image_path)])

    assert exc_info.value.code == expected_code
    assert exc_info.value.context["request_id"] == "row-1"
    if expected_code != "vllm_backend.likelihood_alignment":
        assert exc_info.value.context["channel"] == "policy"
        assert exc_info.value.context["generated_step_index"] == 0
        assert exc_info.value.context["token_id"] == 21


def test_vllm_session_rejects_duplicate_semantic_request_ids(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (2, 2), color="white").save(image_path)
    request = _request(image_path)
    engine = FakeEngine()

    with pytest.raises(RuntimeContractError) as exc_info:
        _session(engine).decode([request, request])

    assert exc_info.value.code == "vllm_backend.duplicate_request_id"
    assert engine.calls == []


def test_vllm_session_rejects_duplicate_native_request_ids(tmp_path: Path) -> None:
    image_paths = [tmp_path / "first.png", tmp_path / "second.png"]
    for image_path in image_paths:
        Image.new("RGB", (2, 2), color="white").save(image_path)
    engine = FakeEngine(
        [
            _native_output(native_request_id="0"),
            _native_output(
                native_request_id="0",
                prompt_ids=(13, IMAGE_PAD_ID),
            ),
        ]
    )

    with pytest.raises(RuntimeContractError) as exc_info:
        _session(engine).decode(
            [
                _request(image_paths[0], request_id="row-first"),
                _request(
                    image_paths[1],
                    request_id="row-second",
                    prompt_ids=(13, IMAGE_PAD_ID),
                ),
            ]
        )

    assert exc_info.value.code == "vllm_backend.native_request_id"


def test_vllm_session_rejects_non_contiguous_native_request_ids(
    tmp_path: Path,
) -> None:
    image_paths = [tmp_path / "first.png", tmp_path / "second.png"]
    for image_path in image_paths:
        Image.new("RGB", (2, 2), color="white").save(image_path)
    engine = FakeEngine(
        [
            _native_output(native_request_id="0"),
            _native_output(
                native_request_id="2",
                prompt_ids=(13, IMAGE_PAD_ID),
            ),
        ]
    )

    with pytest.raises(RuntimeContractError) as exc_info:
        _session(engine).decode(
            [
                _request(image_paths[0], request_id="row-first"),
                _request(
                    image_paths[1],
                    request_id="row-second",
                    prompt_ids=(13, IMAGE_PAD_ID),
                ),
            ]
        )

    assert exc_info.value.code == "vllm_backend.native_request_id"
    assert exc_info.value.context == {"request_ids": [0, 2]}


def test_vllm_session_close_is_idempotent_and_decode_after_close_fails(
    tmp_path: Path,
    fake_vllm_modules: SimpleNamespace,
) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (2, 2), color="white").save(image_path)
    engine = FakeEngine()
    session = _session(engine)

    session.close()
    session.close()

    assert engine.core.shutdown_calls == 1
    assert fake_vllm_modules.model_parallel == 1
    assert fake_vllm_modules.distributed == 1
    with pytest.raises(RuntimeContractError) as exc_info:
        session.decode([_request(image_path)])
    assert exc_info.value.code == "vllm_backend.session_closed"


def test_vllm_engine_cleanup_collects_after_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.inference import vllm_backend

    calls = 0

    def collect() -> int:
        nonlocal calls
        calls += 1
        return 0

    monkeypatch.setattr(vllm_backend.gc, "collect", collect)
    engine = FakeEngine()

    receipt = vllm_backend._close_vllm_engine(engine)

    assert receipt["status"] == "completed"
    assert receipt["shutdown_called"] is True
    assert calls == 1


def test_vllm_session_close_rejects_missing_owned_shutdown_interface() -> None:
    engine = SimpleNamespace(llm_engine=SimpleNamespace(engine_core=object()))
    session = _session(engine)

    with pytest.raises(RuntimeContractError) as exc_info:
        session.close()

    assert exc_info.value.code == "vllm_backend.cleanup_interface_missing"
    cleanup = session.receipt.effective_settings["runtime_preflight"]["cleanup"]
    assert cleanup["status"] == "failed"
