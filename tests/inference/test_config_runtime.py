from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.common.errors import ConfigContractError, RuntimeContractError


def test_valid_production_infer_config_loads() -> None:
    from src.config.inference import InferConfig, load_infer_config

    resolved = load_infer_config("configs/coordexp_swift/infer/base.yaml")

    assert isinstance(resolved.config, InferConfig)
    assert resolved.config.backend.type == "hf"
    assert resolved.config.generation.batch_size > 1
    assert resolved.config.debug.smoke is False
    assert resolved.config_dict["generation"]["batch_size"] == 2
    assert resolved.fingerprint


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


def test_vllm_backend_is_reserved_but_not_implemented(tmp_path: Path) -> None:
    from src.config.inference import load_infer_config

    config_path = _write_config(tmp_path, backend={"type": "vllm"})

    with pytest.raises(ConfigContractError) as exc_info:
        load_infer_config(config_path)

    assert exc_info.value.code == "config.backend_not_implemented"
    assert exc_info.value.context["backend"] == "vllm"


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


def test_runtime_assembly_uses_default_owner_wired_adapter_and_delta_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.config.inference import load_infer_config
    from src.inference import runtime as runtime_module

    config_path = _write_config(
        tmp_path,
        adapter={"type": "dora", "path": "adapter/checkpoint-final"},
        embedding_delta={"path": "delta/special-token-delta.safetensors"},
        debug={"smoke": True, "dry_run": True},
    )
    resolved = load_infer_config(config_path)

    monkeypatch.setattr(
        runtime_module,
        "load_qwen_components_from_options",
        lambda options: {
            "base_model_path": options.base_model,
            "load_model": options.load_model,
        },
    )
    monkeypatch.setattr(
        runtime_module,
        "load_inference_dora_adapter",
        lambda *, config, qwen: {
            "status": "validated",
            "adapter_path": config.adapter.path,
            "base_model_path": qwen["base_model_path"],
        },
    )
    monkeypatch.setattr(
        runtime_module,
        "validate_inference_embedding_delta_identity",
        lambda *, config, qwen: {
            "status": "validated",
            "delta_path": config.embedding_delta.path,
            "base_model_path": qwen["base_model_path"],
        },
    )

    runtime = runtime_module.assemble_runtime(resolved.config)

    assert runtime.model_identity["family"] == "base-plus-adapter-plus-delta"
    assert runtime.model_identity["base"]["path"] == resolved.config.model.base_model
    assert runtime.model_identity["adapter"]["adapter_path"].endswith(
        "adapter/checkpoint-final"
    )
    assert runtime.model_identity["embedding_delta"]["delta_path"].endswith(
        "delta/special-token-delta.safetensors"
    )


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
            "attn_implementation": "flash_attention_2",
            "processor": {"do_resize": False},
            "runtime_patches": {"patch_embed_linearization": "enabled"},
        },
        "data": {"input_jsonl": str(directory / "data" / "examples.jsonl")},
        "template": {
            "object_field_order": "desc_first",
            "object_ordering": "source_order",
            "assistant_format": "object_box_closed",
            "prompt": {"user": "Describe objects."},
        },
        "backend": {"type": "hf"},
        "generation": {
            "batch_size": 2,
            "max_new_tokens": 64,
            "temperature": 0.0,
            "top_p": 1.0,
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
