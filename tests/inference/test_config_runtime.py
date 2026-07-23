from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from src.common.errors import ConfigContractError, RuntimeContractError
from src.qwen.tokens import (
    DEFAULT_COORDINATE_TOKENS,
    DEFAULT_WRAPPER_TOKENS,
    QwenTokenIdentity,
)


def test_valid_production_infer_config_loads() -> None:
    from src.config.inference import InferConfig, load_infer_config

    resolved = load_infer_config("configs/coordexp_swift/infer/base.yaml")
    repo_root = Path.cwd().resolve()
    config_dir = (repo_root / "configs" / "coordexp_swift" / "infer").resolve()
    expected_base_model = (
        repo_root
        / "model_cache"
        / "models"
        / "Qwen"
        / "Qwen3-VL-2B-Instruct-coordexp-natural-adjacent"
    ).resolve()
    expected_input = (
        repo_root / "tests" / "fixtures" / "smoke" / "qwen3_vl_single_image_pack" / "examples.jsonl"
    ).resolve()
    expected_artifact_root = (repo_root / "outputs" / "coordexp_swift" / "infer").resolve()

    assert isinstance(resolved.config, InferConfig)
    assert resolved.config.backend.type == "hf"
    assert resolved.config.backend.hf.attn_implementation == "flash_attention_2"
    assert resolved.config.backend.hf.patch_embed_linearization == "enabled"
    assert set(type(resolved.config.model).model_fields) == {
        "base_model",
        "dtype",
        "processor",
    }
    assert resolved.config.generation.batch_size > 1
    assert resolved.config.generation.n == 1
    assert resolved.config.generation.repetition_penalty == pytest.approx(1.0)
    assert resolved.config.artifacts.include_raw_model_logprob is False
    assert resolved.config.debug.smoke is False
    assert resolved.config_dict["generation"]["batch_size"] == 2
    assert Path(resolved.config.model.base_model) == expected_base_model
    assert Path(resolved.config.data.input_jsonl) == expected_input
    assert Path(resolved.config.run.artifact_root) == expected_artifact_root
    for resolved_path in (
        Path(resolved.config.model.base_model),
        Path(resolved.config.data.input_jsonl),
        Path(resolved.config.run.artifact_root),
    ):
        with pytest.raises(ValueError):
            resolved_path.relative_to(config_dir)
    assert resolved.fingerprint


@pytest.mark.parametrize(
    "config_path",
    sorted(Path("configs/coordexp_swift/infer").glob("*.yaml")),
    ids=lambda path: path.name,
)
def test_all_canonical_infer_configs_use_strict_backend_projection(
    config_path: Path,
) -> None:
    from src.config.inference import load_infer_config

    resolved = load_infer_config(config_path)

    if resolved.config.backend.type == "hf":
        assert resolved.config.backend.hf.attn_implementation in {
            "flash_attention_2",
            "sdpa",
            "eager",
        }
        assert resolved.config.backend.hf.patch_embed_linearization in {
            "enabled",
            "disabled",
        }
        assert set(resolved.config_dict["backend"]) == {"type", "hf"}
    else:
        assert 0 < resolved.config.backend.vllm.gpu_memory_utilization <= 1
        assert resolved.config.backend.vllm.max_model_len == 2048
        assert set(resolved.config_dict["backend"]) == {"type", "vllm"}
    assert set(resolved.config_dict["model"]) == {"base_model", "dtype", "processor"}
    assert resolved.config.generation.temperature == pytest.approx(0.0)
    assert resolved.config.generation.top_p == pytest.approx(1.0)
    assert resolved.config.generation.n == 1
    assert resolved.config.scoring.enabled is True
    assert resolved.config.artifacts.write_token_trace is True
    assert resolved.config.artifacts.write_parse_diagnostics is True
    assert resolved.config.artifacts.include_raw_model_logprob is False


@pytest.mark.parametrize(
    "config_path",
    [
        "configs/coordexp_swift/infer/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_dora_r16a32_step917_val200.yaml",
        "configs/coordexp_swift/infer/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_dora_r16a32_benchmark.yaml",
        "configs/coordexp_swift/infer/wave7_real_production_adapter_smoke.yaml",
    ],
)
def test_production_aligned_infer_configs_keep_training_system_prompt(
    config_path: str,
) -> None:
    from src.config.inference import load_infer_config

    training_config = yaml.safe_load(
        Path(
            "configs/coordexp_swift/prod/"
            "qwen3_vl_2b_desc_first_geo_sorted_pure_ce_dora_r16a32_llm_12000_"
            "accelerate8_ebs64_4epoch_warmup0p1.yaml"
        ).read_text(encoding="utf-8")
    )
    expected_system = training_config["template"]["prompt"]["system"]

    resolved = load_infer_config(config_path)

    assert resolved.config.template.prompt.system == expected_system


def test_step917_val200_and_benchmark_configs_use_same_checkpoint_payloads() -> None:
    from src.config.inference import load_infer_config

    val200 = load_infer_config(
        "configs/coordexp_swift/infer/"
        "qwen3_vl_2b_desc_first_geo_sorted_pure_ce_dora_r16a32_step917_val200.yaml"
    )
    benchmark = load_infer_config(
        "configs/coordexp_swift/infer/"
        "qwen3_vl_2b_desc_first_geo_sorted_pure_ce_dora_r16a32_benchmark.yaml"
    )

    assert val200.config.adapter is not None
    assert benchmark.config.adapter is not None
    assert val200.config.embedding_delta is not None
    assert benchmark.config.embedding_delta is not None
    assert val200.config.adapter.path == benchmark.config.adapter.path
    assert val200.config.embedding_delta.path == benchmark.config.embedding_delta.path
    assert "/step-917/" in val200.config.adapter.path
    assert val200.config.generation.repetition_penalty == pytest.approx(1.10)
    assert benchmark.config.generation.repetition_penalty == pytest.approx(1.10)


@pytest.mark.parametrize("key", ["optimizer", "training", "checkpoint"])
def test_infer_config_rejects_training_only_top_level_keys(
    tmp_path: Path,
    key: str,
) -> None:
    from src.config.inference import load_infer_config

    config_path = _write_config(tmp_path, **{key: {}})

    with pytest.raises(ConfigContractError) as exc_info:
        load_infer_config(config_path)

    assert exc_info.value.code == "config.schema_validation"
    assert exc_info.value.context["field"] == key


def test_infer_config_rejects_unknown_nested_keys(tmp_path: Path) -> None:
    from src.config.inference import load_infer_config

    config_path = _write_config(
        tmp_path,
        generation={
            "batch_size": 2,
            "max_new_tokens": 64,
            "temperature": 0.0,
            "top_p": 1.0,
            "extra_decode_knob": True,
        },
    )

    with pytest.raises(ConfigContractError) as exc_info:
        load_infer_config(config_path)

    assert exc_info.value.code == "config.schema_validation"
    assert exc_info.value.context["field"] == "generation.extra_decode_knob"


def test_legacy_infer_config_path_is_rejected() -> None:
    from src.config.inference import load_infer_config

    with pytest.raises(ConfigContractError) as exc_info:
        load_infer_config("configs/infer/pipeline.yaml")

    assert exc_info.value.code == "config.legacy_infer_path"
    assert "reference-only" in exc_info.value.message


@pytest.mark.parametrize("gpu_memory_utilization", [0.01, 0.7, 1.0])
def test_vllm_backend_accepts_strict_selected_block(
    tmp_path: Path,
    gpu_memory_utilization: float,
) -> None:
    from src.config.inference import load_infer_config

    config_path = _write_vllm_config(
        tmp_path,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    resolved = load_infer_config(config_path)

    assert resolved.config.backend.type == "vllm"
    assert resolved.config.backend.vllm.gpu_memory_utilization == pytest.approx(
        gpu_memory_utilization
    )
    assert resolved.config.backend.vllm.max_model_len == 2048


def test_vllm_backend_accepts_explicit_max_model_len(tmp_path: Path) -> None:
    from src.config.inference import load_infer_config
    from src.inference.runtime import prepare_backend_launch

    resolved = load_infer_config(
        _write_vllm_config(
            tmp_path,
            gpu_memory_utilization=0.7,
            max_model_len=4096,
        )
    )
    launch = prepare_backend_launch(
        resolved.config,
        generation_config_fingerprint="generation-fingerprint",
        execution_model={
            "model_path": str(tmp_path / "snapshot"),
            "mode": "materialized",
            "composition_key": "a" * 64,
            "snapshot_fingerprint": "b" * 64,
        },
    )

    assert resolved.config.backend.vllm.max_model_len == 4096
    assert launch.backend_options["vllm"]["max_model_len"] == 4096


def test_vllm_backend_replaces_inherited_hf_discriminated_block(
    tmp_path: Path,
) -> None:
    from src.config.inference import load_infer_config

    base = tmp_path / "base.yaml"
    base.write_text(
        yaml.safe_dump(_base_config(tmp_path), sort_keys=False),
        encoding="utf-8",
    )
    child = tmp_path / "vllm.yaml"
    child.write_text(
        yaml.safe_dump(
            {
                "extends": "base.yaml",
                "backend": {
                    "type": "vllm",
                    "vllm": {"gpu_memory_utilization": 0.7},
                },
                "debug": {"smoke": True, "dry_run": True},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    resolved = load_infer_config(child)

    assert resolved.config.backend.type == "vllm"
    assert "hf" not in resolved.config_dict["backend"]


@pytest.mark.parametrize("gpu_memory_utilization", [0.0, -0.1, 1.01, float("inf")])
def test_vllm_backend_rejects_invalid_gpu_memory_utilization(
    tmp_path: Path,
    gpu_memory_utilization: float,
) -> None:
    from src.config.inference import load_infer_config

    config_path = _write_vllm_config(
        tmp_path,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    with pytest.raises(ConfigContractError) as exc_info:
        load_infer_config(config_path)

    assert exc_info.value.code == "config.schema_validation"
    assert exc_info.value.context["field"].endswith(
        "vllm.gpu_memory_utilization"
    )


@pytest.mark.parametrize("max_model_len", [0, -1, 1.5, True])
def test_vllm_backend_rejects_invalid_max_model_len(
    tmp_path: Path,
    max_model_len: object,
) -> None:
    from src.config.inference import load_infer_config

    config_path = _write_vllm_config(
        tmp_path,
        gpu_memory_utilization=0.7,
        max_model_len=max_model_len,
    )

    with pytest.raises(ConfigContractError) as exc_info:
        load_infer_config(config_path)

    assert exc_info.value.code == "config.schema_validation"
    assert exc_info.value.context["field"].endswith("vllm.max_model_len")


@pytest.mark.parametrize(
    ("backend", "field_suffix"),
    [
        (
            {
                "type": "hf",
                "hf": {
                    "attn_implementation": "sdpa",
                    "patch_embed_linearization": "enabled",
                },
                "vllm": {"gpu_memory_utilization": 0.7},
            },
            "vllm",
        ),
        (
            {
                "type": "vllm",
                "vllm": {"gpu_memory_utilization": 0.7},
                "hf": {
                    "attn_implementation": "sdpa",
                    "patch_embed_linearization": "enabled",
                },
            },
            "hf",
        ),
        ({"type": "hf"}, "hf"),
        ({"type": "vllm"}, "vllm"),
        (
            {
                "type": "hf",
                "hf": {"attn_implementation": "sdpa"},
            },
            "hf.patch_embed_linearization",
        ),
        (
            {
                "type": "hf",
                "hf": {"patch_embed_linearization": "enabled"},
            },
            "hf.attn_implementation",
        ),
        ({"type": "vllm", "vllm": {}}, "vllm.gpu_memory_utilization"),
        (
            {
                "type": "vllm",
                "vllm": {
                    "gpu_memory_utilization": 0.7,
                    "arbitrary_engine_kwarg": True,
                },
            },
            "vllm.arbitrary_engine_kwarg",
        ),
    ],
)
def test_backend_discriminator_rejects_opposite_missing_and_unknown_blocks(
    tmp_path: Path,
    backend: dict[str, Any],
    field_suffix: str,
) -> None:
    from src.config.inference import load_infer_config

    config_path = _write_config(tmp_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["backend"] = backend
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ConfigContractError) as exc_info:
        load_infer_config(config_path)

    assert exc_info.value.code == "config.schema_validation"
    assert exc_info.value.context["field"].endswith(field_suffix)


@pytest.mark.parametrize(
    ("legacy_field", "legacy_value"),
    [
        ("attn_implementation", "sdpa"),
        ("runtime_patches", {"patch_embed_linearization": "enabled"}),
    ],
)
def test_infer_model_rejects_legacy_hf_only_fields(
    tmp_path: Path,
    legacy_field: str,
    legacy_value: Any,
) -> None:
    from src.config.inference import load_infer_config

    config_path = _write_config(tmp_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["model"][legacy_field] = legacy_value
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ConfigContractError) as exc_info:
        load_infer_config(config_path)

    assert exc_info.value.code == "config.schema_validation"
    assert exc_info.value.context["field"] == f"model.{legacy_field}"


@pytest.mark.parametrize(
    ("overrides", "expected_code"),
    [
        ({"generation": {"temperature": 0.1}}, "config.deterministic_inference"),
        ({"generation": {"top_p": 0.9}}, "config.deterministic_inference"),
        ({"generation": {"n": 2}}, "config.schema_validation"),
        ({"scoring": {"enabled": False}}, "config.scoring_required"),
        (
            {"artifacts": {"write_token_trace": False}},
            "config.inference_evidence_required",
        ),
        (
            {"artifacts": {"write_parse_diagnostics": False}},
            "config.inference_evidence_required",
        ),
    ],
)
def test_canonical_inference_rejects_stochastic_or_evidence_disabling_values(
    tmp_path: Path,
    overrides: dict[str, Any],
    expected_code: str,
) -> None:
    from src.config.inference import load_infer_config

    config_path = _write_config(tmp_path, **overrides)

    with pytest.raises(ConfigContractError) as exc_info:
        load_infer_config(config_path)

    assert exc_info.value.code == expected_code


def test_raw_model_logprob_is_opt_in(tmp_path: Path) -> None:
    from src.config.inference import load_infer_config

    default_config = load_infer_config(
        _write_config(
            tmp_path / "default",
            debug={"smoke": False, "dry_run": True},
        )
    )
    enabled_config = load_infer_config(
        _write_config(
            tmp_path / "enabled",
            artifacts={
                "write_token_trace": True,
                "write_parse_diagnostics": True,
                "include_raw_model_logprob": True,
            },
            debug={"smoke": False, "dry_run": True},
        )
    )

    assert default_config.config.artifacts.include_raw_model_logprob is False
    assert enabled_config.config.artifacts.include_raw_model_logprob is True


def test_vllm_version_preflight_classifies_without_authorizing_runtime() -> None:
    from src.config.inference import inspect_vllm_runtime_version

    known = inspect_vllm_runtime_version(observed_version="0.14.1")
    unverified = inspect_vllm_runtime_version(observed_version="0.14.2")

    assert known == {
        "observed_version": "0.14.1",
        "status": "known_working",
        "known_working_versions": ["0.14.1"],
    }
    assert unverified == {
        "observed_version": "0.14.2",
        "status": "unverified",
        "known_working_versions": ["0.14.1"],
    }


def test_debug_batch_size_one_requires_explicit_smoke(tmp_path: Path) -> None:
    from src.config.inference import load_infer_config

    production_path = _write_config(
        tmp_path / "production",
        generation={"batch_size": 1, "max_new_tokens": 64, "temperature": 0.0, "top_p": 1.0},
    )
    with pytest.raises(ConfigContractError) as exc_info:
        load_infer_config(production_path)
    assert exc_info.value.code == "config.production_batch_size_one"

    debug_path = _write_config(
        tmp_path / "debug",
        generation={"batch_size": 1, "max_new_tokens": 64, "temperature": 0.0, "top_p": 1.0},
        debug={"smoke": True, "dry_run": True},
    )
    resolved = load_infer_config(debug_path)
    assert resolved.config.generation.batch_size == 1
    assert resolved.config.debug.smoke is True

    dry_run_path = _write_config(
        tmp_path / "dry_run",
        generation={"batch_size": 1, "max_new_tokens": 64, "temperature": 0.0, "top_p": 1.0},
        debug={"smoke": False, "dry_run": True},
    )
    dry_run = load_infer_config(dry_run_path)
    assert dry_run.config.generation.batch_size == 1
    assert dry_run.config.debug.dry_run is True


def test_noncanonical_production_config_path_is_rejected_but_debug_is_allowed(
    tmp_path: Path,
) -> None:
    from src.config.inference import load_infer_config

    production_path = _write_config(tmp_path / "outside" / "infer.yaml")
    with pytest.raises(ConfigContractError) as exc_info:
        load_infer_config(production_path)
    assert exc_info.value.code == "config.noncanonical_infer_path"

    debug_path = _write_config(
        tmp_path / "outside-debug" / "infer.yaml",
        debug={"smoke": False, "dry_run": True},
    )
    assert load_infer_config(debug_path).config.debug.dry_run is True


def test_resolved_config_artifacts_are_written_under_configs(tmp_path: Path) -> None:
    from src.config.inference import load_infer_config
    from src.config.writer import write_resolved_config_artifacts

    resolved = load_infer_config(_write_config(tmp_path, debug={"smoke": False, "dry_run": True}))
    artifacts = write_resolved_config_artifacts(resolved, tmp_path / "run")

    assert artifacts.json_path == tmp_path / "run" / "configs" / "resolved.json"
    assert artifacts.yaml_path == tmp_path / "run" / "configs" / "resolved.yaml"
    assert not (tmp_path / "run" / "resolved_config.json").exists()
    assert json.loads(artifacts.json_path.read_text(encoding="utf-8"))["config"][
        "schema_version"
    ] == 1
    assert yaml.safe_load(artifacts.yaml_path.read_text(encoding="utf-8"))["resolution"][
        "fingerprint"
    ] == resolved.fingerprint


def test_pipeline_dry_run_respects_fail_collision_policy(tmp_path: Path) -> None:
    from src.inference.pipeline import run

    config_path = _write_config(
        tmp_path,
        debug={"smoke": False, "dry_run": True},
    )

    assert run(config_path=config_path) == 0
    with pytest.raises(ConfigContractError) as exc_info:
        run(config_path=config_path)

    assert exc_info.value.code == "config.run_dir_exists"


def test_pipeline_dry_run_timestamp_collision_policy_chooses_timestamped_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.inference import pipeline as pipeline_module

    config_path = _write_config(
        tmp_path,
        run={
            "name": "wave2-test",
            "artifact_root": str(tmp_path / "outputs"),
            "collision_policy": "timestamp",
        },
        debug={"smoke": False, "dry_run": True},
    )
    (tmp_path / "outputs" / "wave2-test").mkdir(parents=True)
    monkeypatch.setattr(
        pipeline_module,
        "_timestamp_suffix",
        lambda: "20260702T000000Z",
    )

    assert pipeline_module.run(config_path=config_path) == 0
    assert (
        tmp_path
        / "outputs"
        / "wave2-test-20260702T000000Z"
        / "configs"
        / "resolved.json"
    ).is_file()


def test_pipeline_dry_run_does_not_overwrite_existing_resolved_config(
    tmp_path: Path,
) -> None:
    from src.inference.pipeline import run

    config_path = _write_config(
        tmp_path,
        generation={"batch_size": 2, "max_new_tokens": 64, "temperature": 0.0, "top_p": 1.0},
        debug={"smoke": False, "dry_run": True},
    )

    assert run(config_path=config_path) == 0
    resolved_path = tmp_path / "outputs" / "wave2-test" / "configs" / "resolved.json"
    first_payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    assert first_payload["config"]["generation"]["max_new_tokens"] == 64

    config_path = _write_config(
        tmp_path,
        generation={"batch_size": 2, "max_new_tokens": 128, "temperature": 0.0, "top_p": 1.0},
        debug={"smoke": False, "dry_run": True},
    )
    with pytest.raises(ConfigContractError) as exc_info:
        run(config_path=config_path)

    assert exc_info.value.code == "config.run_dir_exists"
    second_payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    assert second_payload["config"]["generation"]["max_new_tokens"] == 64


def test_frontend_loads_processor_only_and_projects_hf_launch_payloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.config.inference import load_infer_config
    from src.inference import runtime as runtime_module

    config_path = _write_config(
        tmp_path,
        adapter={"type": "dora", "path": "adapter/step-5"},
        embedding_delta={"path": "delta/special-token-delta.safetensors"},
        debug={"smoke": True, "dry_run": True},
    )
    resolved = load_infer_config(config_path)
    observed_options: list[Any] = []

    def fake_load_qwen(options: Any) -> SimpleNamespace:
        observed_options.append(options)
        return SimpleNamespace(model=None)

    monkeypatch.setattr(
        runtime_module,
        "load_qwen_components_from_options",
        fake_load_qwen,
    )

    frontend = runtime_module.assemble_frontend(
        resolved.config,
        generation_config_fingerprint="generation-fingerprint",
    )

    assert observed_options[0].load_model is False
    assert observed_options[0].attn_implementation == "flash_attention_2"
    assert observed_options[0].patch_embed_linearization == "enabled"
    assert frontend.launch.backend == "hf"
    assert frontend.launch.adapter is not None
    assert str(frontend.launch.adapter["path"]).endswith("adapter/step-5")
    assert frontend.launch.embedding_delta is not None
    assert str(frontend.launch.embedding_delta["path"]).endswith(
        "delta/special-token-delta.safetensors"
    )


def test_frontend_rejects_any_shared_executable_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.config.inference import load_infer_config
    from src.inference import runtime as runtime_module

    resolved = load_infer_config(
        _write_config(tmp_path, debug={"smoke": True, "dry_run": True})
    )
    monkeypatch.setattr(
        runtime_module,
        "load_qwen_components_from_options",
        lambda options: SimpleNamespace(model=object()),
    )

    with pytest.raises(RuntimeContractError) as exc_info:
        runtime_module.assemble_frontend(
            resolved.config,
            generation_config_fingerprint="generation-fingerprint",
        )

    assert exc_info.value.code == "inference.frontend_model_loaded"


def test_hf_launch_rejects_vllm_execution_model_receipt(tmp_path: Path) -> None:
    from src.config.inference import load_infer_config
    from src.inference.runtime import prepare_backend_launch

    resolved = load_infer_config(
        _write_config(
            tmp_path,
            adapter={"type": "dora", "path": "adapter/step-5"},
            embedding_delta={"path": "delta/special-token-delta.safetensors"},
            debug={"smoke": True, "dry_run": True},
        )
    )

    with pytest.raises(RuntimeContractError) as exc_info:
        prepare_backend_launch(
            resolved.config,
            generation_config_fingerprint="generation-fingerprint",
            execution_model={"model_path": str(tmp_path / "materialized")},
        )

    assert exc_info.value.code == "inference.hf_execution_model_forbidden"


def test_vllm_composed_launch_requires_execution_model(
    tmp_path: Path,
) -> None:
    from src.config.inference import load_infer_config
    from src.inference.runtime import prepare_backend_launch

    config_path = _write_vllm_config(tmp_path, gpu_memory_utilization=0.7)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["adapter"] = {
        "type": "dora",
        "path": str(tmp_path / "adapter"),
        "name": "default",
    }
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    resolved = load_infer_config(config_path)

    with pytest.raises(RuntimeContractError) as exc_info:
        prepare_backend_launch(
            resolved.config,
            generation_config_fingerprint="generation-fingerprint",
        )

    assert exc_info.value.code == "inference.execution_model_required"


def test_vllm_execution_model_launch_uses_materialized_path_without_live_payloads(
    tmp_path: Path,
) -> None:
    from src.config.inference import load_infer_config
    from src.inference.runtime import prepare_backend_launch

    config_path = _write_vllm_config(tmp_path, gpu_memory_utilization=0.7)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["adapter"] = {
        "type": "dora",
        "path": str(tmp_path / "adapter"),
        "name": "default",
    }
    payload["embedding_delta"] = {"path": str(tmp_path / "delta")}
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    resolved = load_infer_config(config_path)
    snapshot = tmp_path / "materialized" / "snapshot"

    launch = prepare_backend_launch(
        resolved.config,
        generation_config_fingerprint="generation-fingerprint",
        execution_model={
            "model_path": str(snapshot),
            "mode": "materialized",
            "composition_key": "a" * 64,
            "snapshot_fingerprint": "b" * 64,
            "composition_fidelity": {"digest": "c" * 64},
        },
    )

    assert launch.backend == "vllm"
    assert launch.model_path == str(snapshot.resolve())
    assert launch.adapter is None
    assert launch.embedding_delta is None
    assert launch.execution_model_identity is not None
    assert launch.execution_model_identity["composition_key"] == "a" * 64


def test_vllm_materialized_launch_accepts_structural_identity_without_comparison(
    tmp_path: Path,
) -> None:
    from src.config.inference import load_infer_config
    from src.inference.runtime import prepare_backend_launch

    config_path = _write_vllm_config(tmp_path, gpu_memory_utilization=0.7)
    resolved = load_infer_config(config_path)

    launch = prepare_backend_launch(
        resolved.config,
        generation_config_fingerprint="generation-fingerprint",
        execution_model={
            "model_path": str(tmp_path / "snapshot"),
            "mode": "materialized",
            "composition_key": "a" * 64,
            "snapshot_fingerprint": "b" * 64,
        },
    )

    assert launch.execution_model_identity is not None
    assert "composition_fidelity" not in launch.execution_model_identity


def test_missing_inference_input_reports_declaring_config_and_resolved_path(
    tmp_path: Path,
) -> None:
    from src.config.inference import load_infer_config, validate_infer_input_paths

    config_path = _write_vllm_config(tmp_path / "leaf", gpu_memory_utilization=0.7)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base_path = Path(payload["model"]["base_model"])
    data_path = Path(payload["data"]["input_jsonl"])
    base_path.mkdir(parents=True)
    data_path.parent.mkdir(parents=True)
    data_path.write_text("{}\n", encoding="utf-8")
    payload["adapter"] = {
        "type": "dora",
        "path": "../shared/missing-adapter",
        "name": "default",
    }
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    resolved = load_infer_config(config_path)

    with pytest.raises(ConfigContractError) as exc_info:
        validate_infer_input_paths(resolved)

    assert exc_info.value.code == "config.inference_input_path_missing"
    assert exc_info.value.context == {
        "field": "adapter.path",
        "declared_path": "../shared/missing-adapter",
        "declaring_config_path": str(config_path.resolve()),
        "resolved_path": str((config_path.parent / "../shared/missing-adapter").resolve()),
        "expected_kind": "directory",
    }


def test_qwen_components_shape_exposes_delta_identity_sha_fields() -> None:
    import dataclasses

    from src.qwen.runtime_loading import QwenComponents

    field_names = {field.name for field in dataclasses.fields(QwenComponents)}
    assert "base_config_sha256" in field_names
    assert "tokenizer_sha256" in field_names


def test_qwen_loading_train_wrapper_uses_single_neutral_loader() -> None:
    loading_tree = ast.parse(Path("src/qwen/loading.py").read_text(encoding="utf-8"))
    runtime_tree = ast.parse(Path("src/qwen/runtime_loading.py").read_text(encoding="utf-8"))

    loading_defs = {
        node.name
        for node in ast.walk(loading_tree)
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }
    runtime_defs = {
        node.name
        for node in ast.walk(runtime_tree)
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }
    assert "load_qwen_components_from_options" not in loading_defs
    assert "QwenLoadOptions" not in loading_defs
    assert "load_qwen_components_from_options" in runtime_defs
    assert "QwenLoadOptions" in runtime_defs

    imports_runtime_loader = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "src.qwen.runtime_loading"
        and {
            "QwenLoadOptions",
            "load_qwen_components_from_options",
        }.issubset({alias.name for alias in node.names})
        for node in ast.walk(loading_tree)
    )
    assert imports_runtime_loader


def test_infer_entry_help_resolves_without_src_infer_package() -> None:
    assert Path("src/infer.py").is_file()
    assert not Path("src/infer").exists()
    assert Path("src/inference/__init__.py").is_file()

    result = subprocess.run(
        [sys.executable, "-m", "src.infer", "--help"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 0
    assert "--config" in result.stdout


def test_inference_facing_modules_have_no_training_config_or_pipeline_residue() -> None:
    forbidden_names = {
        "TrainConfig",
        "ResolvedTrainConfig",
        "ResolvedStepSchedule",
        "load_train_config",
    }
    forbidden_training_modules = ("src.training",)
    paths = [
        Path("src/infer.py"),
        Path("src/config/inference.py"),
        Path("src/config/writer.py"),
        Path("src/qwen/runtime_loading.py"),
        Path("src/qwen/special_token_embeddings.py"),
        Path("src/adapters/dora.py"),
        *Path("src/inference").glob("*.py"),
    ]

    assert paths
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert not module.startswith(forbidden_training_modules), str(path)
                imported = {alias.name for alias in node.names}
                assert forbidden_names.isdisjoint(imported), str(path)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith(forbidden_training_modules), str(path)
            elif isinstance(node, ast.Name):
                assert node.id not in forbidden_names, str(path)


def _write_config(path_or_dir: Path, **overrides: Any) -> Path:
    directory = path_or_dir if path_or_dir.suffix == "" else path_or_dir.parent
    directory.mkdir(parents=True, exist_ok=True)
    path = path_or_dir if path_or_dir.suffix else directory / "infer.yaml"
    payload = _base_config(directory)
    _deep_update(payload, overrides)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _write_vllm_config(
    path_or_dir: Path,
    *,
    gpu_memory_utilization: float,
    max_model_len: object | None = None,
) -> Path:
    directory = path_or_dir if path_or_dir.suffix == "" else path_or_dir.parent
    directory.mkdir(parents=True, exist_ok=True)
    path = path_or_dir if path_or_dir.suffix else directory / "infer.yaml"
    payload = _base_config(directory)
    vllm: dict[str, object] = {"gpu_memory_utilization": gpu_memory_utilization}
    if max_model_len is not None:
        vllm["max_model_len"] = max_model_len
    payload["backend"] = {
        "type": "vllm",
        "vllm": vllm,
    }
    payload["debug"] = {"smoke": False, "dry_run": True}
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _base_config(directory: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run": {
            "name": "wave2-test",
            "artifact_root": str(directory / "outputs"),
            "collision_policy": "fail",
        },
        "model": {
            "base_model": str(directory / "model_cache" / "qwen"),
            "dtype": "bf16",
            "processor": {"do_resize": False},
        },
        "data": {"input_jsonl": str(directory / "data" / "examples.jsonl")},
        "template": {
            "object_field_order": "desc_first",
            "object_ordering": "source_order",
            "assistant_format": "object_box_closed",
            "prompt": {"user": "Describe objects."},
        },
        "backend": {
            "type": "hf",
            "hf": {
                "attn_implementation": "flash_attention_2",
                "patch_embed_linearization": "enabled",
            },
        },
        "generation": {
            "batch_size": 2,
            "max_new_tokens": 64,
            "temperature": 0.0,
            "top_p": 1.0,
            "n": 1,
        },
        "scoring": {"enabled": True},
        "artifacts": {"write_token_trace": True, "write_parse_diagnostics": True},
        "debug": {"smoke": False, "dry_run": False},
    }


def _deep_update(payload: dict[str, Any], updates: dict[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(payload.get(key), dict):
            _deep_update(payload[key], value)
        else:
            payload[key] = value


class FakePeftModel:
    def __init__(self) -> None:
        self.loaded_path = ""
        self.loaded_adapter_name = ""
        self.active_adapter = ""

    def load_adapter(
        self,
        path: str,
        *,
        adapter_name: str,
        is_trainable: bool,
    ) -> SimpleNamespace:
        assert is_trainable is False
        self.loaded_path = path
        self.loaded_adapter_name = adapter_name
        return SimpleNamespace(missing_keys=[], unexpected_keys=[])

    def set_adapter(self, adapter_name: str) -> None:
        self.active_adapter = adapter_name

    def get_model_status(self) -> SimpleNamespace:
        return SimpleNamespace(
            enabled=True,
            active_adapters=[self.active_adapter],
            merged_adapters=[],
            requires_grad=False,
        )


class FakeGenerationModel:
    def __init__(self) -> None:
        self.to_calls: list[str] = []
        self.eval_calls = 0

    def to(self, device: object) -> "FakeGenerationModel":
        self.to_calls.append(str(device))
        return self

    def eval(self) -> "FakeGenerationModel":
        self.eval_calls += 1
        return self


def _fake_qwen_components(config: Any) -> SimpleNamespace:
    return SimpleNamespace(
        base_model_path=config.model.base_model,
        model=FakeGenerationModel(),
        base_config_sha256="base-config-sha",
        tokenizer_sha256="tokenizer-sha",
        processor_identity=SimpleNamespace(
            to_artifact_dict=lambda: {"processor_class": "unit-processor"},
        ),
    )


def _token_identity() -> QwenTokenIdentity:
    return QwenTokenIdentity(
        required_tokens=(*DEFAULT_WRAPPER_TOKENS, *DEFAULT_COORDINATE_TOKENS),
        wrapper_token_ids={
            token: 151646 + index
            for index, token in enumerate(DEFAULT_WRAPPER_TOKENS)
        },
        coordinate_token_ids=tuple(range(151670, 152670)),
        im_end_newline_text="<|im_end|>\n",
        im_end_token_ids=(151645,),
        newline_token_ids=(198,),
        im_end_newline_token_ids=(151645, 198),
        tokenizer_vocab_size=152670,
    )


def _write_delta_metadata(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    metadata = {
        "semantics": "additive_delta",
        "tensor_key": "shared_embed_delta",
        "tensor_shape": [1004, 4],
        "tensor_dtype": "torch.float32",
        "token_strings": [*DEFAULT_WRAPPER_TOKENS, *DEFAULT_COORDINATE_TOKENS],
        "token_ids": [151646, 151647, 151648, 151649, *range(151670, 152670)],
        "base_model_path": str(path.parent / "model_cache" / "qwen"),
        "base_config_sha256": "base-config-sha",
        "tokenizer_sha256": "tokenizer-sha",
        "tie_word_embeddings": True,
    }
    (path / "special_token_embeddings.json").write_text(
        json.dumps(metadata, sort_keys=True),
        encoding="utf-8",
    )
