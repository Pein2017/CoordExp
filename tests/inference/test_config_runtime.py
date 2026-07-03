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
    assert resolved.config.generation.batch_size > 1
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
    assert "Wave 2" not in exc_info.value.message


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
        "load_inference_embedding_delta",
        lambda *, config, qwen: {
            "status": "loaded",
            "identity": {
                "status": "validated",
                "delta_path": config.embedding_delta.path,
                "base_model_path": qwen["base_model_path"],
            },
            "load": {"loaded": True},
        },
    )

    runtime = runtime_module.assemble_runtime(resolved.config)

    assert runtime.model_identity["family"] == "base-plus-adapter-plus-delta"
    assert runtime.model_identity["base"]["path"] == resolved.config.model.base_model
    assert runtime.model_identity["adapter"]["adapter_path"].endswith(
        "adapter/checkpoint-final"
    )
    assert runtime.model_identity["embedding_delta"]["identity"]["delta_path"].endswith(
        "delta/special-token-delta.safetensors"
    )


def test_adapter_runtime_loads_qwen_model_and_uses_adapter_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.config.inference import load_infer_config
    from src.inference import runtime as runtime_module

    config_path = _write_config(
        tmp_path,
        adapter={"type": "dora", "path": "adapter/checkpoint-final"},
        debug={"smoke": True, "dry_run": False},
    )
    resolved = load_infer_config(config_path)
    fake_model = FakePeftModel()
    observed_load_model: list[bool] = []

    def fake_load_qwen(options: Any) -> SimpleNamespace:
        observed_load_model.append(options.load_model)
        if not options.load_model:
            raise AssertionError("adapter runtime must not load Qwen with load_model=False")
        return SimpleNamespace(
            base_model_path=options.base_model,
            model=fake_model,
            token_identity=None,
            base_config_sha256="base-config-sha",
            tokenizer_sha256="tokenizer-sha",
        )

    monkeypatch.setattr(
        runtime_module,
        "load_qwen_components_from_options",
        fake_load_qwen,
    )

    runtime = runtime_module.assemble_runtime(resolved.config)

    assert observed_load_model == [True]
    assert fake_model.loaded_path.endswith("adapter/checkpoint-final")
    assert fake_model.active_adapter == "default"
    assert runtime.adapter_receipt is not None
    assert runtime.adapter_receipt["status"] == "validated"
    assert runtime.model_identity["family"] == "base-plus-adapter"


def test_base_only_runtime_loads_qwen_model_for_hf_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.config.inference import load_infer_config
    from src.inference import runtime as runtime_module

    config_path = _write_config(
        tmp_path,
        adapter=None,
        embedding_delta=None,
        debug={"smoke": True, "dry_run": False},
    )
    resolved = load_infer_config(config_path)
    observed_load_model: list[bool] = []

    def fake_load_qwen(options: Any) -> SimpleNamespace:
        observed_load_model.append(options.load_model)
        return SimpleNamespace(
            base_model_path=options.base_model,
            model=object() if options.load_model else None,
            token_identity=None,
            base_config_sha256="base-config-sha",
            tokenizer_sha256="tokenizer-sha",
        )

    monkeypatch.setattr(
        runtime_module,
        "load_qwen_components_from_options",
        fake_load_qwen,
    )

    runtime = runtime_module.assemble_runtime(resolved.config)

    assert observed_load_model == [True]
    assert runtime.qwen.model is not None
    assert runtime.model_identity["family"] == "base-only"


def test_inference_runtime_moves_loaded_model_to_cuda_for_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.config.inference import load_infer_config
    from src.inference import runtime as runtime_module

    config_path = _write_config(
        tmp_path,
        adapter=None,
        embedding_delta=None,
        debug={"smoke": True, "dry_run": False},
    )
    resolved = load_infer_config(config_path)
    fake_model = FakeGenerationModel()

    monkeypatch.setattr(runtime_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        runtime_module,
        "load_qwen_components_from_options",
        lambda options: SimpleNamespace(
            base_model_path=options.base_model,
            model=fake_model if options.load_model else None,
            token_identity=None,
            base_config_sha256="base-config-sha",
            tokenizer_sha256="tokenizer-sha",
        ),
    )

    runtime = runtime_module.assemble_runtime(resolved.config)

    assert runtime.qwen.model is fake_model
    assert fake_model.to_calls == ["cuda"]
    assert fake_model.eval_calls == 1


def test_embedding_delta_runtime_loads_and_installs_delta_with_qwen_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.config.inference import load_infer_config
    from src.inference import runtime as runtime_module

    delta_dir = tmp_path / "delta"
    config_path = _write_config(
        tmp_path,
        embedding_delta={"path": str(delta_dir)},
        debug={"smoke": True, "dry_run": True},
    )
    resolved = load_infer_config(config_path)
    calls: list[tuple[str, str]] = []

    monkeypatch.setattr(
        runtime_module,
        "load_qwen_components_from_options",
        lambda options: SimpleNamespace(
            base_model_path=resolved.config.model.base_model,
            model=object(),
            token_identity=_token_identity(),
            base_config_sha256="base-config-sha",
            tokenizer_sha256="tokenizer-sha",
        ),
    )
    monkeypatch.setattr(
        runtime_module,
        "load_inference_embedding_delta",
        lambda *, config, qwen: calls.append((config.embedding_delta.path, str(qwen.model)))
        or {
            "status": "loaded",
            "identity": {
                "status": "validated",
                "metadata": {
                    "base_config_sha256": qwen.base_config_sha256,
                    "tokenizer_sha256": qwen.tokenizer_sha256,
                },
                "metadata_path": str(delta_dir / "special_token_embeddings.json"),
            },
            "load": {"loaded": True},
        },
    )

    runtime = runtime_module.assemble_runtime(resolved.config)

    assert calls == [(str(delta_dir), str(runtime.qwen.model))]
    assert runtime.embedding_delta_receipt is not None
    assert runtime.embedding_delta_receipt["status"] == "loaded"
    assert runtime.embedding_delta_receipt["identity"]["metadata"]["base_config_sha256"] == "base-config-sha"
    assert runtime.model_identity["family"] == "base-plus-delta"
    assert runtime.model_identity["embedding_delta"]["identity"]["metadata_path"].endswith(
        "special_token_embeddings.json"
    )


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
            "repetition_penalty": 1.10,
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
