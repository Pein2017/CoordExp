from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.artifacts import provenance


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _committed_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "--quiet")
    _git(repo, "config", "user.email", "tests@example.invalid")
    _git(repo, "config", "user.name", "CoordExp Tests")
    (repo / "src").mkdir()
    (repo / "src" / "entry.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(repo, "add", "src/entry.py")
    _git(repo, "commit", "--quiet", "-m", "fixture")
    return repo


def _available(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    assert value["status"] == "available"
    return value


def test_repository_provenance_for_clean_checkout(tmp_path: Path) -> None:
    repo = _committed_repo(tmp_path)

    receipt = provenance.collect_repository_provenance(repo)

    commit = _available(receipt["commit"])
    assert len(str(commit["value"])) == 40
    assert receipt["state"] == "clean"
    assert receipt["tracked_changes_present"] is False
    assert receipt["untracked_changes_present"] is False
    assert receipt["execution_relevant_changes"] == {
        "count": 0,
        "path_classes": {},
        "truncated": False,
    }
    _available(receipt["execution_relevant_digest"])


def test_repository_provenance_for_tracked_dirty_checkout(tmp_path: Path) -> None:
    repo = _committed_repo(tmp_path)
    patch_body = "PATCH_BODY_MUST_NOT_APPEAR"
    (repo / "src" / "entry.py").write_text(
        f"VALUE = {patch_body!r}\n", encoding="utf-8"
    )

    receipt = provenance.collect_repository_provenance(repo)

    assert receipt["state"] == "dirty"
    assert receipt["tracked_changes_present"] is True
    assert receipt["untracked_changes_present"] is False
    assert receipt["execution_relevant_changes"]["path_classes"] == {
        "source": {"tracked": 1, "untracked": 0}
    }
    _available(receipt["execution_relevant_digest"])
    assert patch_body not in json.dumps(receipt, sort_keys=True)


def test_repository_provenance_for_untracked_execution_file(tmp_path: Path) -> None:
    repo = _committed_repo(tmp_path)
    (repo / "configs").mkdir()
    (repo / "configs" / "run.yaml").write_text("schema_version: 1\n", encoding="utf-8")

    receipt = provenance.collect_repository_provenance(repo)

    assert receipt["state"] == "dirty"
    assert receipt["tracked_changes_present"] is False
    assert receipt["untracked_changes_present"] is True
    assert receipt["execution_relevant_changes"]["path_classes"] == {
        "config": {"tracked": 0, "untracked": 1}
    }
    _available(receipt["execution_relevant_digest"])


def test_repository_provenance_outside_git_is_explicitly_unavailable(
    tmp_path: Path,
) -> None:
    receipt = provenance.collect_repository_provenance(tmp_path)

    assert receipt == {
        "commit": {"status": "unavailable", "reason": "not_a_git_repository"},
        "state": "unavailable",
        "tracked_changes_present": None,
        "untracked_changes_present": None,
        "execution_relevant_changes": {
            "count": 0,
            "path_classes": {},
            "truncated": False,
        },
        "execution_relevant_digest": {
            "status": "unavailable",
            "reason": "not_a_git_repository",
        },
    }


def test_dependency_provenance_records_versions_origins_and_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    modules: dict[str, object] = {}
    for _component, (
        _distribution,
        import_name,
    ) in provenance._DEPENDENCY_IMPORTS.items():
        suffix = ".so" if import_name == "flash_attn_2_cuda" else ".py"
        origin = tmp_path / f"{import_name}{suffix}"
        origin.write_bytes(f"identity:{import_name}".encode())
        module = SimpleNamespace(
            __spec__=SimpleNamespace(origin=str(origin)), __file__=str(origin)
        )
        if import_name == "torch":
            module.__version__ = "2.8.0"
            module.version = SimpleNamespace(
                cuda="12.8", hip=None, git_version="torch-git"
            )
            module.cuda = SimpleNamespace(
                is_available=lambda: True, device_count=lambda: 8
            )
            module.backends = SimpleNamespace(
                cudnn=SimpleNamespace(version=lambda: 91002)
            )
        modules[import_name] = module

    source_modules = {
        "flash_attn.flash_attn_interface",
        "transformers.modeling_utils",
        "transformers.modeling_flash_attention_utils",
        "transformers.integrations.flash_attention",
        "transformers.models.auto.image_processing_auto",
        "transformers.models.auto.configuration_auto",
        "transformers.models.auto.processing_auto",
        "transformers.models.auto.tokenization_auto",
        "transformers.models.qwen2.tokenization_qwen2",
        "transformers.models.qwen2.tokenization_qwen2_fast",
        "transformers.models.qwen2_vl.image_processing_qwen2_vl_fast",
        "transformers.models.qwen3_vl.configuration_qwen3_vl",
        "transformers.models.qwen3_vl.modeling_qwen3_vl",
        "transformers.models.qwen3_vl.processing_qwen3_vl",
        "transformers.optimization",
        "transformers.tokenization_utils",
        "transformers.tokenization_utils_fast",
        "transformers.trainer_utils",
        "accelerate.accelerator",
        "accelerate.state",
        "accelerate.utils.modeling",
        "accelerate.utils.operations",
        "peft.tuners.lora.layer",
        "peft.tuners.lora.dora",
        "peft.tuners.lora.config",
        "peft.tuners.lora.model",
        "peft.peft_model",
        "peft.mapping_func",
        "peft.tuners.tuners_utils",
        "tokenizers.tokenizers",
        "torch._C",
    }
    for import_name in source_modules:
        suffix = (
            ".so" if import_name in {"tokenizers.tokenizers", "torch._C"} else ".py"
        )
        origin = tmp_path / (import_name.replace(".", "_") + suffix)
        origin.write_bytes(f"identity:{import_name}".encode())
        module = SimpleNamespace(
            __spec__=SimpleNamespace(origin=str(origin)), __file__=str(origin)
        )
        for component_name, owners in provenance._DEPENDENCY_SOURCE_IMPORTS.items():
            for owner_name, owner_import_name in owners.items():
                if owner_import_name != import_name:
                    continue
                for symbol in provenance._DEPENDENCY_SOURCE_SYMBOLS.get(
                    component_name, {}
                ).get(owner_name, ()):
                    setattr(module, symbol, object())
        modules[import_name] = module

    native_paths = {
        (
            "nvidia-cuda-runtime-cu12",
            "nvidia/cuda_runtime/lib/libcudart.so.12",
        ),
        ("torch", "torch/lib/libtorch_cuda.so"),
        ("torch", "torch/lib/libc10_cuda.so"),
        ("nvidia-cudnn-cu12", "nvidia/cudnn/lib/libcudnn.so.9"),
        ("nvidia-cudnn-cu12", "nvidia/cudnn/lib/libcudnn_adv.so.9"),
        ("nvidia-cudnn-cu12", "nvidia/cudnn/lib/libcudnn_cnn.so.9"),
        (
            "nvidia-cudnn-cu12",
            "nvidia/cudnn/lib/libcudnn_engines_precompiled.so.9",
        ),
        (
            "nvidia-cudnn-cu12",
            "nvidia/cudnn/lib/libcudnn_engines_runtime_compiled.so.9",
        ),
        ("nvidia-cudnn-cu12", "nvidia/cudnn/lib/libcudnn_graph.so.9"),
        ("nvidia-cudnn-cu12", "nvidia/cudnn/lib/libcudnn_heuristic.so.9"),
        ("nvidia-cudnn-cu12", "nvidia/cudnn/lib/libcudnn_ops.so.9"),
    }
    resolved_native_paths = {}
    for distribution, relative_path in native_paths:
        origin = tmp_path / relative_path.replace("/", "_")
        origin.write_bytes(f"identity:{distribution}:{relative_path}".encode())
        resolved_native_paths[(distribution, relative_path)] = origin

    monkeypatch.setattr(provenance, "_import_module", modules.__getitem__)
    monkeypatch.setattr(
        provenance, "_find_module_spec", lambda name: modules[name].__spec__
    )
    monkeypatch.setattr(
        provenance,
        "_distribution_file_path",
        lambda distribution, relative_path: resolved_native_paths[
            (distribution, relative_path)
        ],
        raising=False,
    )
    monkeypatch.setattr(
        provenance,
        "_loaded_shared_object_origin",
        lambda soname: resolved_native_paths[
            (
                "nvidia-cuda-runtime-cu12",
                "nvidia/cuda_runtime/lib/libcudart.so.12",
            )
        ]
        if soname == "libcudart.so.12"
        else None,
        raising=False,
    )
    monkeypatch.setattr(
        provenance,
        "_distribution_version",
        lambda distribution: {
            "ms-swift": "4.2.2",
            "transformers": "4.57.1",
            "flash-attn": "2.8.3",
            "torch": "2.8.0",
            "nvidia-cuda-runtime-cu12": "12.8.90",
            "accelerate": "1.10.1",
            "peft": "0.17.1",
            "tokenizers": "0.22.1",
        }[distribution],
    )

    receipt = provenance.collect_dependency_provenance()

    assert tuple(receipt) == (*provenance._DEPENDENCY_IMPORTS, "cuda-runtime")
    for name, component in receipt.items():
        _available(component["distribution_version"])
        digest = _available(component["sha256"])["value"]
        assert len(str(digest)) == 64
        if name == "ms-swift":
            assert component["role"] == "reference_only_not_imported_by_training_route"
            _available(component["selected_module_origin"])
            assert component["imported_origin"] == {
                "status": "unavailable",
                "reason": "reference_only_not_imported",
            }
        else:
            assert component["role"] == "runtime_dependency"
            _available(component["imported_origin"])
    assert receipt["flash_attn_2_cuda"]["origin_kind"] == "binary"
    cuda_runtime = receipt["cuda-runtime"]
    assert cuda_runtime["distribution"] == "nvidia-cuda-runtime-cu12"
    assert cuda_runtime["origin_resolution"] == "loaded_shared_object"
    assert cuda_runtime["loaded_soname"] == "libcudart.so.12"
    assert cuda_runtime["loaded_origin_matches_distribution"] is True
    _available(cuda_runtime["distribution_origin"])
    _available(cuda_runtime["imported_origin"])
    _available(cuda_runtime["size_bytes"])
    assert len(str(_available(cuda_runtime["sha256"])["value"])) == 64
    assert receipt["transformers"]["origin_kind"] == "source"
    assert set(receipt["transformers"]["source_identities"]) == {
        "auto_configuration",
        "auto_image_processing",
        "auto_processing",
        "auto_tokenization",
        "flash_attention_integration",
        "flash_attention_utils",
        "modeling_utils",
        "optimization",
        "qwen2_tokenization",
        "qwen2_tokenization_fast",
        "qwen2_vl_image_processing_fast",
        "qwen3_vl_configuration",
        "qwen3_vl_modeling",
        "qwen3_vl_processing",
        "tokenization_utils",
        "tokenization_utils_fast",
        "trainer_utils",
    }
    assert set(receipt["accelerate"]["source_identities"]) == {
        "accelerator",
        "modeling",
        "operations",
        "state",
    }
    assert set(receipt["flash-attn"]["source_identities"]) == {"flash_attn_interface"}
    assert set(receipt["torch"]["source_identities"]) == {"torch_c_extension"}
    assert set(receipt["peft"]["source_identities"]) == {
        "dora_implementation",
        "dora_layer",
        "lora_config",
        "lora_model",
        "mapping_func",
        "peft_model",
        "tuners_utils",
    }
    assert set(receipt["tokenizers"]["source_identities"]) == {"tokenizers_extension"}
    assert receipt["transformers"]["source_identities"]["auto_processing"][
        "symbols"
    ] == ["AutoProcessor"]
    assert receipt["transformers"]["source_identities"][
        "qwen2_vl_image_processing_fast"
    ]["symbols"] == ["Qwen2VLImageProcessorFast"]
    assert receipt["transformers"]["source_identities"]["qwen2_tokenization_fast"][
        "symbols"
    ] == ["Qwen2TokenizerFast"]
    assert receipt["transformers"]["source_identities"]["qwen2_tokenization"][
        "symbols"
    ] == ["Qwen2Tokenizer"]
    assert receipt["transformers"]["source_identities"]["qwen3_vl_modeling"][
        "symbols"
    ] == [
        "Qwen3VLForConditionalGeneration",
        "Qwen3VLTextAttention",
        "Qwen3VLTextModel",
        "Qwen3VLVisionAttention",
    ]
    assert receipt["peft"]["source_identities"]["mapping_func"]["symbols"] == [
        "get_peft_model"
    ]
    assert all(
        source["symbols_available"] is True
        for component_name in ("transformers", "peft")
        for source in receipt[component_name]["source_identities"].values()
    )
    for component_name in (
        "transformers",
        "flash-attn",
        "torch",
        "accelerate",
        "peft",
        "tokenizers",
    ):
        for source in receipt[component_name]["source_identities"].values():
            _available(source["imported_origin"])
            _available(source["size_bytes"])
            assert len(str(_available(source["sha256"])["value"])) == 64
    assert set(receipt["torch"]["native_identities"]) == {
        "cudnn",
        "cudnn_adv",
        "cudnn_cnn",
        "cudnn_engines_precompiled",
        "cudnn_engines_runtime_compiled",
        "cudnn_graph",
        "cudnn_heuristic",
        "cudnn_ops",
        "libc10_cuda",
        "libtorch_cuda",
    }
    for native in receipt["torch"]["native_identities"].values():
        assert native["origin_kind"] == "binary"
        _available(native["imported_origin"])
        _available(native["size_bytes"])
        assert len(str(_available(native["sha256"])["value"])) == 64

    monkeypatch.setattr(
        provenance,
        "_nvidia_driver_version",
        lambda: {"status": "available", "value": "550.54.15"},
        raising=False,
    )
    runtime = provenance.collect_runtime_metadata(receipt)
    assert runtime["torch_cuda"] == {
        "torch_version": "2.8.0",
        "torch_git_version": "torch-git",
        "cuda_compiled_version": "12.8",
        "hip_compiled_version": None,
        "cuda_available": True,
        "cuda_device_count": 8,
        "cudnn_version": 91002,
        "nvidia_driver_version": {
            "status": "available",
            "value": "550.54.15",
        },
        "cuda_driver_runtime_applicable": True,
    }


def test_ms_swift_is_reference_only_and_resolved_without_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference_repo = tmp_path / "ms-swift-reference"
    reference_repo.mkdir()
    _git(reference_repo, "init", "--quiet")
    _git(reference_repo, "config", "user.email", "tests@example.invalid")
    _git(reference_repo, "config", "user.name", "CoordExp Tests")
    swift_root = reference_repo / "swift"
    swift_root.mkdir()
    swift_origin = swift_root / "__init__.py"
    swift_origin.write_text("__version__ = '4.2.2'\n", encoding="utf-8")
    (reference_repo / "setup.py").write_text("# reference fixture\n", encoding="utf-8")
    _git(reference_repo, "add", "swift/__init__.py", "setup.py")
    _git(reference_repo, "commit", "--quiet", "-m", "reference fixture")

    runtime_modules: dict[str, object] = {}
    for component, (
        _distribution,
        import_name,
    ) in provenance._DEPENDENCY_IMPORTS.items():
        if component == "ms-swift":
            continue
        suffix = ".so" if import_name == "flash_attn_2_cuda" else ".py"
        origin = tmp_path / f"{import_name}{suffix}"
        origin.write_bytes(f"identity:{import_name}".encode())
        runtime_modules[import_name] = SimpleNamespace(
            __spec__=SimpleNamespace(origin=str(origin)), __file__=str(origin)
        )

    imported: list[str] = []

    def import_runtime(name: str) -> object:
        imported.append(name)
        if name == "swift":
            pytest.fail("reference-only ms-swift must not be imported")
        return runtime_modules[name]

    monkeypatch.setattr(provenance, "_import_module", import_runtime)
    monkeypatch.setattr(
        provenance,
        "_find_module_spec",
        lambda name: SimpleNamespace(origin=str(swift_origin))
        if name == "swift"
        else None,
        raising=False,
    )
    monkeypatch.setattr(provenance, "_distribution_version", lambda _name: "1.0")
    monkeypatch.setattr(provenance, "_distribution_record_text", lambda _name: "record")

    receipt = provenance.collect_dependency_provenance()

    assert "swift" not in imported
    reference = receipt["ms-swift"]
    assert reference["role"] == "reference_only_not_imported_by_training_route"
    assert reference["origin_resolution"] == "import_spec_without_import"
    assert reference["imported_origin"] == {
        "status": "unavailable",
        "reason": "reference_only_not_imported",
    }
    assert _available(reference["selected_module_origin"])["value"] == str(
        swift_origin.resolve()
    )
    source_repository = _available(reference["source_repository"])["value"]
    assert source_repository["state"] == "clean"
    assert len(str(source_repository["commit"])) == 40
    assert all(
        component["role"] == "runtime_dependency"
        for name, component in receipt.items()
        if name != "ms-swift"
    )


def test_dependency_failures_are_per_component_and_do_not_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_import(_name: str) -> object:
        raise ImportError("injected import failure")

    def fail_version(_name: str) -> str:
        raise provenance.PackageNotFoundError

    monkeypatch.setattr(provenance, "_import_module", fail_import)
    monkeypatch.setattr(provenance, "_find_module_spec", lambda _name: None)
    monkeypatch.setattr(provenance, "_distribution_version", fail_version)
    monkeypatch.setattr(
        provenance,
        "_distribution_file_path",
        lambda _distribution, _relative_path: None,
        raising=False,
    )
    monkeypatch.setattr(
        provenance,
        "_loaded_shared_object_origin",
        lambda _soname: None,
        raising=False,
    )

    receipt = provenance.collect_dependency_provenance()

    assert set(receipt) == {*provenance._DEPENDENCY_IMPORTS, "cuda-runtime"}
    for name, component in receipt.items():
        assert component["distribution_version"] == {
            "status": "unavailable",
            "reason": "distribution_not_found",
        }
        if name == "ms-swift":
            assert component["role"] == "reference_only_not_imported_by_training_route"
            assert component["imported_origin"] == {
                "status": "unavailable",
                "reason": "reference_only_not_imported",
            }
            assert component["selected_module_origin"] == {
                "status": "unavailable",
                "reason": "reference_spec_resolution_failed",
            }
            assert component["sha256"] == {
                "status": "unavailable",
                "reason": "reference_spec_resolution_failed",
            }
        elif name == "cuda-runtime":
            assert component["role"] == "runtime_dependency"
            assert component["imported_origin"] == {
                "status": "unavailable",
                "reason": "loaded_shared_object_unavailable",
            }
            assert component["sha256"] == {
                "status": "unavailable",
                "reason": "loaded_shared_object_unavailable",
            }
            assert component["loaded_origin_matches_distribution"] is False
        else:
            assert component["role"] == "runtime_dependency"
            assert component["imported_origin"] == {
                "status": "unavailable",
                "reason": "import_failed",
            }
            assert component["sha256"] == {
                "status": "unavailable",
                "reason": "import_failed",
            }
            for source in component["source_identities"].values():
                assert source["sha256"] == {
                    "status": "unavailable",
                    "reason": "import_failed",
                }
            for native in component["native_identities"].values():
                assert native["sha256"] == {
                    "status": "unavailable",
                    "reason": "origin_unavailable",
                }
    runtime = provenance.collect_runtime_metadata(receipt)
    assert runtime["torch_cuda"] == {
        "status": "unavailable",
        "reason": "torch_import_failed",
    }


def test_full_provenance_is_deterministic_strict_json_and_non_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _committed_repo(tmp_path)
    secret = "SECRET_TOKEN_VALUE_SHOULD_NEVER_ESCAPE"
    (repo / "src" / "credentials.py").write_text(
        f"TOKEN = {secret!r}\n", encoding="utf-8"
    )

    def fail_import(_name: str) -> object:
        raise ImportError(secret)

    def fail_version(_name: str) -> str:
        raise RuntimeError(secret)

    monkeypatch.setattr(provenance, "_import_module", fail_import)
    monkeypatch.setattr(provenance, "_find_module_spec", fail_import)
    monkeypatch.setattr(provenance, "_distribution_version", fail_version)

    first = provenance.collect_execution_provenance(repository_root=repo)
    second = provenance.collect_execution_provenance(repository_root=repo)
    first_json = json.dumps(
        first, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    second_json = json.dumps(
        second, sort_keys=True, separators=(",", ":"), allow_nan=False
    )

    assert first_json == second_json
    assert secret not in first_json
    assert "TOKEN =" not in first_json
    assert "credentials.py" not in first_json
    assert set(first) == {"schema_version", "repository", "dependencies", "runtime"}


def test_environment_secrets_never_enter_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _committed_repo(tmp_path)
    secrets = {
        "AWS_SECRET_ACCESS_KEY": "SECRET-SENTINEL-AWS",
        "HF_TOKEN": "SECRET-SENTINEL-HF",
        "WANDB_API_KEY": "SECRET-SENTINEL-WANDB",
        "COORDEXP_TEST_PASSWORD": "SECRET-SENTINEL-PASSWORD",
    }
    for name, value in secrets.items():
        monkeypatch.setenv(name, value)

    receipt = provenance.collect_execution_provenance(repository_root=repo)
    receipt_json = json.dumps(
        receipt, sort_keys=True, separators=(",", ":"), allow_nan=False
    )

    for value in secrets.values():
        assert value not in receipt_json
    assert set(receipt) == {"schema_version", "repository", "dependencies", "runtime"}

    def _collect_lowercase_keys(node: object, keys: set[str]) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                keys.add(str(key).lower())
                _collect_lowercase_keys(value, keys)
        elif isinstance(node, list):
            for item in node:
                _collect_lowercase_keys(item, keys)

    all_keys: set[str] = set()
    _collect_lowercase_keys(receipt, all_keys)
    assert "environment" not in all_keys
    assert "env" not in all_keys


def _available_identity(value: object) -> dict[str, object]:
    return {"status": "available", "value": value}


_PINNED_NATIVE_BUILD_IDS = {
    "libtorch_cuda": "bf4c31d86c74fe1e5d5c67caea2d1a006785ce95",
    "libc10_cuda": "d9bfdd5a168655341fb2595f9dd763c9be969281",
    "cudnn": "989afd40d5f56a848608b792eda320d3c4067d4c",
    "cudnn_adv": "40dad754fc3521e5a5531381d3aecc1e0ba31455",
    "cudnn_cnn": "4d8c00415c2e0f9b967a6614588acad6bc5b3164",
    "cudnn_engines_precompiled": "0e111592d57f71d0ccc36621f67c89d9197ec74d",
    "cudnn_engines_runtime_compiled": "f8099f58a92ca1e2671c13927faef534fa9beab7",
    "cudnn_graph": "429aed8e6a0dcc9974a1371b071b8b4d590d0545",
    "cudnn_heuristic": "11532cd128025c2c19cfceea30702b0176035ea9",
    "cudnn_ops": "8ca23749d0fd44f406607efb7f09986da436204c",
}


def _runtime_dependency(
    *,
    distribution: str,
    import_name: str,
    version: str,
    origin_kind: str,
    origin_sha256: str,
    origin_size_bytes: int,
    record_sha256: str,
    record_size_bytes: int,
    source_identities: dict[str, tuple[str, str, int] | tuple[str, str, str, int]]
    | None = None,
    source_symbols: dict[str, tuple[str, ...]] | None = None,
    native_identities: dict[str, tuple[str, str, str, int]] | None = None,
) -> dict[str, object]:
    sources = {}
    for name, identity in (source_identities or {}).items():
        if len(identity) == 3:
            module_name, sha256, size_bytes = identity
            source_kind = "source"
        else:
            module_name, source_kind, sha256, size_bytes = identity
        suffix = ".so" if source_kind == "binary" else ".py"
        sources[name] = {
            "module": module_name,
            "symbols": list((source_symbols or {}).get(name, ())),
            "symbols_available": True,
            "imported_origin": _available_identity(f"/env/{module_name}{suffix}"),
            "origin_kind": source_kind,
            "sha256": _available_identity(sha256),
            "size_bytes": _available_identity(size_bytes),
        }
    natives = {}
    for name, (owner_distribution, relative_path, sha256, size_bytes) in (
        native_identities or {}
    ).items():
        natives[name] = {
            "distribution": owner_distribution,
            "relative_path": relative_path,
            "imported_origin": _available_identity(f"/env/{relative_path}"),
            "origin_kind": "binary",
            "sha256": _available_identity(sha256),
            "size_bytes": _available_identity(size_bytes),
            "elf_build_id": _available_identity(_PINNED_NATIVE_BUILD_IDS[name]),
        }
    return {
        "distribution": distribution,
        "import_name": import_name,
        "distribution_version": _available_identity(version),
        "distribution_record": _available_identity(
            {"sha256": record_sha256, "size_bytes": record_size_bytes}
        ),
        "role": "runtime_dependency",
        "origin_resolution": "imported_module",
        "imported_origin": _available_identity(f"/env/{import_name}"),
        "origin_kind": origin_kind,
        "sha256": _available_identity(origin_sha256),
        "size_bytes": _available_identity(origin_size_bytes),
        "source_repository": {
            "status": "unavailable",
            "reason": "not_source_origin",
        },
        "source_identities": sources,
        "native_identities": natives,
    }


def _pinned_runtime_observation() -> dict[str, object]:
    flash_record = "fee009739702fa997fa85c07f134ad54636d042475235515b90afdcfffd299a5"
    dependencies = {
        "ms-swift": {
            "distribution": "ms-swift",
            "import_name": "swift",
            "distribution_version": _available_identity("4.2.2"),
            "distribution_record": {
                "status": "unavailable",
                "reason": "distribution_record_absent",
            },
            "role": "reference_only_not_imported_by_training_route",
            "origin_resolution": "import_spec_without_import",
            "selected_module_origin": _available_identity(
                "/reference/swift/__init__.py"
            ),
            "imported_origin": {
                "status": "unavailable",
                "reason": "reference_only_not_imported",
            },
            "origin_kind": "source",
            "sha256": _available_identity(
                "d345fd8f68077d11730ffe56747a52b1858550e8c21067ed55a1db2f79ab5caf"
            ),
            "size_bytes": _available_identity(3529),
            "source_repository": _available_identity(
                {
                    "commit": "f2797138dba0e224cfff735cd89a528a08d8732a",
                    "state": "clean",
                }
            ),
        },
        "transformers": _runtime_dependency(
            distribution="transformers",
            import_name="transformers",
            version="4.57.1",
            origin_kind="source",
            origin_sha256="4ef5187b5f66c564aa575ddab9ce94342630d40fe874d0464d790c6f6b748647",
            origin_size_bytes=47000,
            record_sha256="025b8186b05df7f173dff14013042262c7dacdb556e98ee5c56a182c02e42728",
            record_size_bytes=411910,
            source_identities={
                "auto_configuration": (
                    "transformers.models.auto.configuration_auto",
                    "120b62cd4e13dd155762acd14cf64fc8bca17a068f5dd004ddfd30fbcb2ef4ab",
                    55606,
                ),
                "auto_image_processing": (
                    "transformers.models.auto.image_processing_auto",
                    "34e04aa2fc7e5bab3f6e918d8e144760af5eb7c9e0b910641e09bbecf228c27b",
                    39079,
                ),
                "auto_processing": (
                    "transformers.models.auto.processing_auto",
                    "e97580de189cd7e5ea3378a96e01279939f305c68a0245a9160b80cb1a938961",
                    20853,
                ),
                "auto_tokenization": (
                    "transformers.models.auto.tokenization_auto",
                    "bd040dfb212fb20f1bef1f8fedfc3b4c100fe3b57e603c738552aebf2521c51d",
                    57863,
                ),
                "flash_attention_integration": (
                    "transformers.integrations.flash_attention",
                    "850aa63f9473391188c357454cfb6a89339394d26f6024ccbe246f00f6344144",
                    3125,
                ),
                "flash_attention_utils": (
                    "transformers.modeling_flash_attention_utils",
                    "293fe81c6bd38aac8a3bc6ae086ff21b9695b349e97c9650b1fe61e42a36d710",
                    30112,
                ),
                "modeling_utils": (
                    "transformers.modeling_utils",
                    "bf1c6b2a43cf7c36fb79f37c981424dd6ae78eb863fcaa5d2a37e76c9828611d",
                    308964,
                ),
                "optimization": (
                    "transformers.optimization",
                    "41b08ffb29c2691626b225ed389a694cc8e61480818214a35f215b5c39167bfb",
                    39971,
                ),
                "qwen2_tokenization": (
                    "transformers.models.qwen2.tokenization_qwen2",
                    "23f05697fcaf26fe5e328abaf524f010510a2b9f4ba2e2edc9dca0a4015a09a2",
                    13935,
                ),
                "qwen2_tokenization_fast": (
                    "transformers.models.qwen2.tokenization_qwen2_fast",
                    "1025a3b86526283bd86a447f0fe2d991d09775b32d3fdcc233c9e447b0611049",
                    5210,
                ),
                "qwen2_vl_image_processing_fast": (
                    "transformers.models.qwen2_vl.image_processing_qwen2_vl_fast",
                    "09bfa9b17df7c3f0c6159bc34008ee50f21d2472cd5bae7e5c21ba1ca13a423c",
                    12723,
                ),
                "qwen3_vl_configuration": (
                    "transformers.models.qwen3_vl.configuration_qwen3_vl",
                    "177fd0a4dc1b08307c08ca72cf26b8d7dc028ab6ab5975bf650fc14ab8132a83",
                    14827,
                ),
                "qwen3_vl_modeling": (
                    "transformers.models.qwen3_vl.modeling_qwen3_vl",
                    "dd63ed3b124232735b3dca1bfa28f9d6b0d3f7182afcb75dde8f3e724b2b22da",
                    70877,
                ),
                "qwen3_vl_processing": (
                    "transformers.models.qwen3_vl.processing_qwen3_vl",
                    "efd8d64aaf608aad1ffb3e6d503d6a99e5227d007df95c1d9fa905d998cda4a9",
                    17149,
                ),
                "tokenization_utils": (
                    "transformers.tokenization_utils",
                    "dfcc42414037d865d22568259eec12dd10caa7375b2d7a27707b7a502a08ef80",
                    47780,
                ),
                "tokenization_utils_fast": (
                    "transformers.tokenization_utils_fast",
                    "558e454ee6850e90e8bc5d1782e35878dfda1da9bfdf997ae113fa7ff771a0f3",
                    41383,
                ),
                "trainer_utils": (
                    "transformers.trainer_utils",
                    "20e32d05ef22f366ab9ce91c6dbec2290b2e46f8da749d52eeebaf6a4349bd8e",
                    34254,
                ),
            },
            source_symbols={
                "auto_configuration": ("AutoConfig",),
                "auto_image_processing": ("AutoImageProcessor",),
                "auto_processing": ("AutoProcessor",),
                "auto_tokenization": ("AutoTokenizer", "tokenizer_class_from_name"),
                "flash_attention_integration": ("flash_attention_forward",),
                "flash_attention_utils": (
                    "FlashAttentionKwargs",
                    "_flash_attention_forward",
                    "lazy_import_flash_attention",
                    "prepare_fa_kwargs_from_position_ids",
                ),
                "modeling_utils": ("ALL_ATTENTION_FUNCTIONS",),
                "optimization": ("get_cosine_schedule_with_warmup",),
                "qwen2_tokenization": ("Qwen2Tokenizer",),
                "qwen2_tokenization_fast": ("Qwen2TokenizerFast",),
                "qwen2_vl_image_processing_fast": ("Qwen2VLImageProcessorFast",),
                "qwen3_vl_configuration": ("Qwen3VLConfig", "Qwen3VLTextConfig"),
                "qwen3_vl_modeling": (
                    "Qwen3VLForConditionalGeneration",
                    "Qwen3VLTextAttention",
                    "Qwen3VLTextModel",
                    "Qwen3VLVisionAttention",
                ),
                "qwen3_vl_processing": ("Qwen3VLProcessor",),
                "tokenization_utils": ("PreTrainedTokenizer",),
                "tokenization_utils_fast": ("PreTrainedTokenizerFast",),
                "trainer_utils": ("set_seed",),
            },
        ),
        "flash-attn": _runtime_dependency(
            distribution="flash-attn",
            import_name="flash_attn",
            version="2.8.3",
            origin_kind="source",
            origin_sha256="f1833af940ac1124e09dc05dd922308871411035e75b90bf4dbe3a831dc03b50",
            origin_size_bytes=285,
            record_sha256=flash_record,
            record_size_bytes=15390,
            source_identities={
                "flash_attn_interface": (
                    "flash_attn.flash_attn_interface",
                    "e8da8127f7ebf5c5aeb7f35b316ff96394a70553378f125717ea174af912db13",
                    60677,
                )
            },
        ),
        "flash_attn_2_cuda": _runtime_dependency(
            distribution="flash-attn",
            import_name="flash_attn_2_cuda",
            version="2.8.3",
            origin_kind="binary",
            origin_sha256="8ca052bf2d3f53baa629e22749b9622a95273c5bffb5f06cd24768ef63f65807",
            origin_size_bytes=997961816,
            record_sha256=flash_record,
            record_size_bytes=15390,
        ),
        "torch": _runtime_dependency(
            distribution="torch",
            import_name="torch",
            version="2.9.1",
            origin_kind="source",
            origin_sha256="3caf7f40140ede2465bde40b9003af10cbbc8f7bcf436fa1de026471daa1b288",
            origin_size_bytes=102969,
            record_sha256="a8a2b13b3ba6e31a168babb44dcbdda35d3f56583ae1ba9e98ad51ae809cb0e7",
            record_size_bytes=1364780,
            source_identities={
                "torch_c_extension": (
                    "torch._C",
                    "binary",
                    "90fc84350d2de15ca167734eb63b20b3a1a4ea2721d498b66e3e65174c0cff9f",
                    25521,
                )
            },
            native_identities={
                "libtorch_cuda": (
                    "torch",
                    "torch/lib/libtorch_cuda.so",
                    "02250527966ae122ccfc89d0306736874f9c619ba04431871ef76175e7253b66",
                    1022776209,
                ),
                "libc10_cuda": (
                    "torch",
                    "torch/lib/libc10_cuda.so",
                    "2f6027b42aee93b5db3f814b92dfbe179074cc928eebad85804b6a266214d5f7",
                    697169,
                ),
                "cudnn": (
                    "nvidia-cudnn-cu12",
                    "nvidia/cudnn/lib/libcudnn.so.9",
                    "3b68ea689be647bce63cb3c2d9edb589add415405a7e9f296d69ca029cae0b8b",
                    125136,
                ),
                "cudnn_adv": (
                    "nvidia-cudnn-cu12",
                    "nvidia/cudnn/lib/libcudnn_adv.so.9",
                    "9814d64e04fce1f6cbd16d02ccefd6611541b3c61736fb450e334b482635ca07",
                    285226600,
                ),
                "cudnn_cnn": (
                    "nvidia-cudnn-cu12",
                    "nvidia/cudnn/lib/libcudnn_cnn.so.9",
                    "655c2d2649466c73b6b4c19705fb6906fd95e331a1f8218314b49f37dd7cafed",
                    6301096,
                ),
                "cudnn_engines_precompiled": (
                    "nvidia-cudnn-cu12",
                    "nvidia/cudnn/lib/libcudnn_engines_precompiled.so.9",
                    "f504745d346542609f975de51227080647d3675eac576ead82c396b8124f8f28",
                    547383096,
                ),
                "cudnn_engines_runtime_compiled": (
                    "nvidia-cudnn-cu12",
                    "nvidia/cudnn/lib/libcudnn_engines_runtime_compiled.so.9",
                    "8222672743c4f2e39feb1c365d6f6b6c40f56023471506f49944910847f1ee0a",
                    23094328,
                ),
                "cudnn_graph": (
                    "nvidia-cudnn-cu12",
                    "nvidia/cudnn/lib/libcudnn_graph.so.9",
                    "787c955dce49091ead850e4536666594095ea9f92a8a08879d8ddad466674657",
                    4439912,
                ),
                "cudnn_heuristic": (
                    "nvidia-cudnn-cu12",
                    "nvidia/cudnn/lib/libcudnn_heuristic.so.9",
                    "2804eee5f6fc11d0299ba07687eda0a974d3b2436357c639706c4e282cc2cc05",
                    58726712,
                ),
                "cudnn_ops": (
                    "nvidia-cudnn-cu12",
                    "nvidia/cudnn/lib/libcudnn_ops.so.9",
                    "34f56d67f3df108949d1e5e03543064e055e217d0f5e53dc6d4f7c9454d852f8",
                    127936080,
                ),
            },
        ),
        "cuda-runtime": {
            "distribution": "nvidia-cuda-runtime-cu12",
            "import_name": None,
            "distribution_version": _available_identity("12.8.90"),
            "distribution_record": _available_identity(
                {
                    "sha256": "bef69015da09064656ac31d35fc73d6436712380c602ff38fb5fc16c30512bdf",
                    "size_bytes": 11369,
                }
            ),
            "role": "runtime_dependency",
            "origin_resolution": "loaded_shared_object",
            "loaded_soname": "libcudart.so.12",
            "distribution_relative_path": "nvidia/cuda_runtime/lib/libcudart.so.12",
            "distribution_origin": _available_identity(
                "/env/nvidia/cuda_runtime/lib/libcudart.so.12"
            ),
            "imported_origin": _available_identity(
                "/env/nvidia/cuda_runtime/lib/libcudart.so.12"
            ),
            "loaded_origin_matches_distribution": True,
            "origin_kind": "binary",
            "sha256": _available_identity(
                "c3a75b33af334a3486d197dbd1584a2985183ba4688d237a2be5f2f679329920"
            ),
            "size_bytes": _available_identity(728800),
            "elf_build_id": _available_identity(
                "7b1714ea2d766ca35afe1e3dd34a75b41b78999f"
            ),
            "source_repository": {
                "status": "unavailable",
                "reason": "not_source_origin",
            },
            "source_identities": {},
            "native_identities": {},
        },
        "accelerate": _runtime_dependency(
            distribution="accelerate",
            import_name="accelerate",
            version="1.10.1",
            origin_kind="source",
            origin_sha256="68f56baac9b078db4735567649441ef28932a946b400bf8d8c2519b0a5b94b89",
            origin_size_bytes=1555,
            record_sha256="8c84d01d529cd3d9745936fd74933835e9838e7119a96832a27123e5170500f8",
            record_size_bytes=13939,
            source_identities={
                "accelerator": (
                    "accelerate.accelerator",
                    "9005e9a74cd819701578ea9a86e8cd71c16152f0e2308ed1b6f447c0797837f9",
                    190652,
                ),
                "operations": (
                    "accelerate.utils.operations",
                    "f384a95f02ac6cd1c778f6e82fd814a8ae9b039be8d24c87fe02c07709f313fb",
                    31266,
                ),
                "modeling": (
                    "accelerate.utils.modeling",
                    "35cbf0f316b086f599bf54babf4d7691ca66fd675ef1dfcfb10821bd5e983a02",
                    95815,
                ),
                "state": (
                    "accelerate.state",
                    "dfa76df4d4205babb1f8c1ce87206d359168f28d551b8851b64f159a4911537a",
                    57976,
                ),
            },
        ),
        "peft": _runtime_dependency(
            distribution="peft",
            import_name="peft",
            version="0.17.1",
            origin_kind="source",
            origin_sha256="efabc3b44d7326eee59c07910e426fa38ee0c6419b773e8f75af87ae61d3fa32",
            origin_size_bytes=5523,
            record_sha256="258749eb655ec7ee9b6d4f6040b2a34c9207cafc9f3efc35d96559e63460a38c",
            record_size_bytes=22735,
            source_identities={
                "dora_implementation": (
                    "peft.tuners.lora.dora",
                    "07d514fd057d8aa9ec7b89bec3f3fbd04862951c1a8ff73153e833096481ecf5",
                    8480,
                ),
                "dora_layer": (
                    "peft.tuners.lora.layer",
                    "39e2f6908c9f3faf4d2de2b6ecfdf44c96d2cae67b8155ed1b51bc67fb25b734",
                    97115,
                ),
                "lora_config": (
                    "peft.tuners.lora.config",
                    "85a478bcdfd42f9398d232d5e0ddb49650bd1199148503f864834fdbeca3aebb",
                    42385,
                ),
                "lora_model": (
                    "peft.tuners.lora.model",
                    "9eec396e506134e50afa41ff21b2608a3307f8f07207f460506d2da33d7bc2f7",
                    45130,
                ),
                "mapping_func": (
                    "peft.mapping_func",
                    "c30533cb009ea60e45a35d564285be65c9778c0c41e0c281dc2dfacd69c2315d",
                    6064,
                ),
                "peft_model": (
                    "peft.peft_model",
                    "2307ebeb101baff53b19ce5a0f109885d99b64ab147d67cba19719b6097c3e10",
                    156704,
                ),
                "tuners_utils": (
                    "peft.tuners.tuners_utils",
                    "d5b6a92d8f0b5325951519e81d2ed9debcba6ace1028e887958dd5e5188be66a",
                    67995,
                ),
            },
            source_symbols={
                "dora_implementation": ("DoraLinearLayer",),
                "dora_layer": ("LoraLayer",),
                "lora_config": ("LoraConfig",),
                "lora_model": ("LoraModel",),
                "mapping_func": ("get_peft_model",),
                "peft_model": ("PeftModel", "get_model_status"),
                "tuners_utils": ("BaseTuner", "BaseTunerLayer"),
            },
        ),
        "tokenizers": _runtime_dependency(
            distribution="tokenizers",
            import_name="tokenizers",
            version="0.22.0",
            origin_kind="source",
            origin_sha256="644e596a052fa1b05272b1c141d1286e1c78c2a3346ecabd17d68b62404d8d84",
            origin_size_bytes=2615,
            record_sha256="2df0edacccd5e83cc9fde45cc5199701bc97731a676787ada63d02b8ff0f4bf0",
            record_size_bytes=3650,
            source_identities={
                "tokenizers_extension": (
                    "tokenizers.tokenizers",
                    "binary",
                    "1e1a657d75b22975395c66f0ce1b341fe9eec4b4c17a9e0da3c93a2ffddbd6db",
                    10028880,
                )
            },
        ),
    }
    return {
        "dependencies": dependencies,
        "runtime": {
            "python": {"implementation": "CPython", "version": "3.12.11"},
            "torch_cuda": {
                "torch_version": "2.9.1+cu128",
                "torch_git_version": "5811a8d7da873dd699ff6687092c225caffcf1bb",
                "cuda_compiled_version": "12.8",
                "hip_compiled_version": None,
                "cuda_available": True,
                "cuda_device_count": 1,
                "cudnn_version": 91002,
                "nvidia_driver_version": _available_identity("550.54.15"),
                "cuda_driver_runtime_applicable": True,
            },
        },
    }


def test_pinned_runtime_baseline_admits_exact_runtime_and_reports_reference_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = _pinned_runtime_observation()
    observed["dependencies"]["ms-swift"]["sha256"] = _available_identity("0" * 64)

    result = provenance.compare_pinned_runtime_baseline(
        provenance=observed,
        attention_backend="flash_attention_2",
    )

    assert result["admitted"] is True, result["mismatches"]
    assert result["mismatches"] == []
    assert result["reference_only"]["ms-swift"] == {
        "matches_recorded_reference": False,
        "mismatches": ["dependencies.ms-swift.sha256.value"],
    }
    assert result["schema_version"] == 3
    assert result["baseline_sha256"] == (
        "cc486f03edb88e4fa1c9d41dc6fa98c97f25a633400e6baf98077e8beaf2b784"
    )
    assert provenance.pinned_runtime_baseline()["schema_version"] == 3


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        ("dependencies.transformers.distribution_version.value", "4.58.0"),
        ("dependencies.flash_attn_2_cuda.sha256.value", "0" * 64),
        (
            "dependencies.accelerate.source_identities.operations.sha256.value",
            "1" * 64,
        ),
        (
            "dependencies.flash-attn.source_identities.flash_attn_interface.sha256.value",
            "2" * 64,
        ),
        (
            "dependencies.torch.source_identities.torch_c_extension.sha256.value",
            "3" * 64,
        ),
        (
            "dependencies.peft.source_identities.dora_layer.sha256.value",
            "4" * 64,
        ),
        (
            "dependencies.transformers.source_identities.auto_processing.sha256.value",
            "a" * 64,
        ),
        (
            "dependencies.transformers.source_identities.auto_processing.symbols_available",
            False,
        ),
        (
            "dependencies.transformers.source_identities.qwen2_vl_image_processing_fast.sha256.value",
            "d" * 64,
        ),
        (
            "dependencies.transformers.source_identities.qwen2_tokenization_fast.sha256.value",
            "e" * 64,
        ),
        (
            "dependencies.transformers.source_identities.qwen2_tokenization.symbols_available",
            False,
        ),
        (
            "dependencies.transformers.source_identities.tokenization_utils_fast.sha256.value",
            "f" * 64,
        ),
        (
            "dependencies.peft.source_identities.mapping_func.sha256.value",
            "b" * 64,
        ),
        (
            "dependencies.tokenizers.source_identities.tokenizers_extension.sha256.value",
            "5" * 64,
        ),
        (
            "dependencies.torch.native_identities.libtorch_cuda.sha256.value",
            "6" * 64,
        ),
        (
            "dependencies.torch.native_identities.libtorch_cuda.elf_build_id.value",
            "c" * 40,
        ),
        (
            "dependencies.torch.native_identities.libc10_cuda.sha256.value",
            "7" * 64,
        ),
        (
            "dependencies.torch.native_identities.cudnn_ops.sha256.value",
            "8" * 64,
        ),
        ("dependencies.cuda-runtime.distribution_version.value", "12.9.0"),
        ("dependencies.cuda-runtime.sha256.value", "9" * 64),
        ("dependencies.cuda-runtime.loaded_origin_matches_distribution", False),
        ("runtime.python.version", "3.12.12"),
        ("runtime.torch_cuda.cuda_compiled_version", "12.9"),
        ("runtime.torch_cuda.nvidia_driver_version.value", "570.00"),
        ("runtime.torch_cuda.cuda_driver_runtime_applicable", False),
    ],
)
def test_pinned_runtime_baseline_rejects_runtime_identity_drift(
    path: str,
    replacement: object,
) -> None:
    observed = _pinned_runtime_observation()
    owner: dict[str, object] = observed
    parts = path.split(".")
    for part in parts[:-1]:
        owner = owner[part]  # type: ignore[assignment]
    owner[parts[-1]] = replacement

    result = provenance.compare_pinned_runtime_baseline(
        provenance=observed,
        attention_backend="flash_attention_2",
    )

    assert result["admitted"] is False
    assert result["mismatches"] == [path]


def test_pinned_runtime_baseline_rejects_unknown_backend_and_unavailable_identity() -> (
    None
):
    observed = _pinned_runtime_observation()
    observed["dependencies"]["torch"]["sha256"] = {
        "status": "unavailable",
        "reason": "origin_unreadable",
    }

    result = provenance.compare_pinned_runtime_baseline(
        provenance=observed,
        attention_backend="sdpa",
    )

    assert result["admitted"] is False
    assert result["mismatches"] == [
        "attention_backend",
        "dependencies.torch.sha256.status",
        "dependencies.torch.sha256.value",
    ]
    with pytest.raises(provenance.PinnedRuntimeAdmissionError) as caught:
        provenance.require_pinned_runtime_baseline(
            provenance=observed,
            attention_backend="sdpa",
        )
    assert caught.value.result == result


def test_pinned_runtime_baseline_rejects_unavailable_native_owner() -> None:
    observed = _pinned_runtime_observation()
    observed["dependencies"]["torch"]["native_identities"]["libtorch_cuda"][
        "sha256"
    ] = {
        "status": "unavailable",
        "reason": "origin_unreadable",
    }

    result = provenance.compare_pinned_runtime_baseline(
        provenance=observed,
        attention_backend="flash_attention_2",
    )

    assert result["admitted"] is False
    assert result["mismatches"] == [
        "dependencies.torch.native_identities.libtorch_cuda.sha256.status",
        "dependencies.torch.native_identities.libtorch_cuda.sha256.value",
    ]


def test_pinned_runtime_baseline_rejects_unavailable_loaded_cuda_runtime() -> None:
    observed = _pinned_runtime_observation()
    cuda_runtime = observed["dependencies"]["cuda-runtime"]
    cuda_runtime["imported_origin"] = {
        "status": "unavailable",
        "reason": "loaded_shared_object_unavailable",
    }
    cuda_runtime["sha256"] = {
        "status": "unavailable",
        "reason": "loaded_shared_object_unavailable",
    }

    result = provenance.compare_pinned_runtime_baseline(
        provenance=observed,
        attention_backend="flash_attention_2",
    )

    assert result["admitted"] is False
    assert result["mismatches"] == [
        "dependencies.cuda-runtime.imported_origin.status",
        "dependencies.cuda-runtime.sha256.status",
        "dependencies.cuda-runtime.sha256.value",
    ]


def test_current_live_pinned_runtime_baseline_is_admitted() -> None:
    observed = provenance.collect_execution_provenance(repository_root=Path.cwd())
    torch_cuda = observed["runtime"].get("torch_cuda", {})
    if not isinstance(torch_cuda, dict) or torch_cuda.get("cuda_available") is not True:
        pytest.skip("live pinned GPU baseline requires a CUDA-visible worker")

    result = provenance.compare_pinned_runtime_baseline(
        provenance=observed,
        attention_backend="flash_attention_2",
    )

    assert result["admitted"] is True, result["mismatches"]
    cuda_runtime = observed["dependencies"]["cuda-runtime"]
    assert cuda_runtime["loaded_origin_matches_distribution"] is True
    assert (
        _available(cuda_runtime["imported_origin"])["value"]
        == _available(cuda_runtime["distribution_origin"])["value"]
    )
    assert _available(cuda_runtime["size_bytes"])["value"] == 728800
    assert _available(cuda_runtime["sha256"])["value"] == (
        "c3a75b33af334a3486d197dbd1584a2985183ba4688d237a2be5f2f679329920"
    )


def test_pinned_runtime_baseline_rejects_unapproved_runtime_component_and_source() -> (
    None
):
    observed = _pinned_runtime_observation()
    observed["dependencies"]["new-runtime"] = {"role": "runtime_dependency"}
    observed["dependencies"]["accelerate"]["source_identities"]["new_owner"] = {
        "module": "accelerate.new_owner"
    }
    observed["dependencies"]["torch"]["native_identities"]["new_native_owner"] = {
        "distribution": "torch"
    }

    result = provenance.compare_pinned_runtime_baseline(
        provenance=observed,
        attention_backend="flash_attention_2",
    )

    assert result["admitted"] is False
    assert result["mismatches"] == [
        "dependencies.accelerate.source_identities.new_owner",
        "dependencies.new-runtime",
        "dependencies.torch.native_identities.new_native_owner",
    ]


def test_pinned_runtime_comparator_fails_closed_on_malformed_source_names() -> None:
    observed = _pinned_runtime_observation()
    sources = observed["dependencies"]["accelerate"]["source_identities"]
    sources[7] = {}
    sources["unknown"] = {}
    native_owners = observed["dependencies"]["torch"]["native_identities"]
    native_owners[7] = {}
    native_owners["unknown"] = {}
    observed["dependencies"][7] = {"role": "runtime_dependency"}

    result = provenance.compare_pinned_runtime_baseline(
        provenance=observed,
        attention_backend="flash_attention_2",
    )

    assert result["admitted"] is False
    assert result["mismatches"] == [
        "dependencies.7",
        "dependencies.accelerate.source_identities.7",
        "dependencies.accelerate.source_identities.unknown",
        "dependencies.torch.native_identities.7",
        "dependencies.torch.native_identities.unknown",
    ]
