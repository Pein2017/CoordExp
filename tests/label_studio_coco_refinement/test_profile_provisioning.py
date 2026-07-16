from __future__ import annotations

import inspect
import json
import multiprocessing
import os
import stat
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

import scripts.provision_label_studio_roi_profile as cli_module
import src.label_studio_coco_refinement.inference_profiles as inference_profiles_module
import src.label_studio_coco_refinement.profile_provisioning as provisioning_module
from src.config.inference import load_infer_config
from src.inference.runtime import assemble_runtime
from src.label_studio_coco_refinement.inference_profiles import (
    EngineProfileStore,
    ProfileContractError,
)
from src.label_studio_coco_refinement.profile_provisioning import (
    PROVISIONING_RECEIPT_SCHEMA_VERSION,
    ProfileProvisioningError,
    provision_roi_profile,
    require_single_visible_cuda,
)
from src.label_studio_coco_refinement.roi_launch import (
    DEFAULT_ENGINE_FACTORY_TARGET,
    ROI_LAUNCH_SCHEMA_VERSION,
    load_roi_launch_config,
)
from src.inference.parsing import PARSER_ID, PARSER_POLICY
from src.label_studio_coco_refinement.roi_runtime import RESIDENT_ADAPTER_ID
from src.label_studio_coco_refinement.roi_transform import ROI_TRANSFORM_ID
from src.qwen.images import QWEN_IMAGE_PROCESSOR_KWARGS


class _Identity:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def to_artifact_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(self.payload))


def _fake_runtime(base: Path) -> Any:
    return SimpleNamespace(
        model_identity={"family": "base-only", "base": {"path": str(base)}},
        qwen=SimpleNamespace(
            processor_identity=_Identity(
                {
                    "processor_class": "FakeProcessor",
                    "tokenizer_class": "FakeTokenizer",
                    "image_processor_class": "FakeImageProcessor",
                    "patch_size": 16,
                    "merge_size": 2,
                    "temporal_patch_size": 2,
                }
            ),
            token_identity=_Identity(
                {
                    "vocab_size": 151936,
                    "coordinate_start": 151646,
                    "coordinate_end": 152645,
                }
            ),
        ),
    )


def _authored_infer_config(
    root: Path, *, conditionals: bool = False
) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    base = root / "base"
    base.mkdir()
    (base / "model.safetensors").write_bytes(b"model-v1")
    (base / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_vl",
                "vision_config": {"patch_size": 16, "spatial_merge_size": 2},
            }
        ),
        encoding="utf-8",
    )
    (base / "tokenizer.json").write_text(
        json.dumps({"model": {"type": "BPE", "vocab": {"x": 0}}}),
        encoding="utf-8",
    )
    (base / "preprocessor_config.json").write_text(
        json.dumps({"patch_size": 16, "merge_size": 2}),
        encoding="utf-8",
    )
    (root / "source.jsonl").write_text("{}\n", encoding="utf-8")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "run": {
            "name": "profile-provisioning-test",
            "artifact_root": "outputs",
            "collision_policy": "fail",
        },
        "model": {
            "base_model": "base",
            "dtype": "bf16",
            "attn_implementation": "eager",
            "processor": {"do_resize": False},
            "runtime_patches": {"patch_embed_linearization": "enabled"},
        },
        "data": {"input_jsonl": "source.jsonl"},
        "template": {
            "object_field_order": "desc_first",
            "object_ordering": "source_order",
            "assistant_format": "object_box_closed",
            "prompt": {
                "system": "operator-private-config-marker",
                "user": "find every object",
            },
        },
        "backend": {"type": "hf"},
        "generation": {
            "batch_size": 1,
            "max_new_tokens": 128,
            "temperature": 0.0,
            "top_p": 1.0,
            "repetition_penalty": 1.0,
        },
        "scoring": {"enabled": True},
        "artifacts": {
            "write_token_trace": True,
            "write_parse_diagnostics": True,
        },
        "debug": {"smoke": True, "dry_run": False},
    }
    if conditionals:
        adapter = root / "adapter"
        delta = root / "embedding_delta"
        adapter.mkdir()
        delta.mkdir()
        (adapter / "adapter.safetensors").write_bytes(b"adapter-v1")
        (delta / "embedding_delta.pt").write_bytes(b"delta-v1")
        payload["adapter"] = {"type": "dora", "path": "adapter", "name": "default"}
        payload["embedding_delta"] = {"path": "embedding_delta"}
    config_path = root / "authored-infer.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return config_path, base


def _provision_args(root: Path, config_path: Path) -> dict[str, Any]:
    return {
        "infer_config_path": config_path,
        "profile_store_path": (root / "state/profiles.json").resolve(),
        "launch_config_path": root / "operator/roi-launch.json",
        "receipt_store_path": root / "state/receipts.jsonl",
        "profile_name": "production-profile",
        "selector": "production",
        "bind_host": "127.0.0.1",
        "bind_port": 8123,
        "insertion_ack_timeout_seconds": 30.0,
        "min_axis_pixels": 32,
        "max_axis_pixels": 2048,
        "max_total_pixels": 2_097_152,
        "deadline_seconds": 20.0,
    }


def _named_profile_args(
    args: dict[str, Any], *, profile_name: str, selector: str
) -> dict[str, Any]:
    return {**args, "profile_name": profile_name, "selector": selector}


def _process_provision_profile(
    args: dict[str, Any],
    base: Path,
    result_queue: Any,
    slow_write_started: Any | None = None,
) -> None:
    if slow_write_started is not None:
        original_write = EngineProfileStore._write

        def delayed_write(store: EngineProfileStore, payload: Any) -> None:
            slow_write_started.set()
            time.sleep(0.5)
            original_write(store, payload)

        EngineProfileStore._write = delayed_write
    try:
        receipt = provision_roi_profile(
            **args,
            runtime_loader=lambda _config: _fake_runtime(base.resolve()),
            gpu_preflight=lambda: None,
        )
        result_queue.put(("ok", receipt["launch"]))
    except BaseException as exc:
        result_queue.put(("error", repr(exc)))
        raise


def test_provisions_exact_runtime_identity_conditional_roles_and_strict_launch(
    tmp_path: Path,
) -> None:
    config_path, base = _authored_infer_config(tmp_path / "authored", conditionals=True)
    args = _provision_args(tmp_path, config_path)
    calls: list[str] = []

    def preflight() -> None:
        calls.append("gpu")

    def runtime_loader(config: Any) -> Any:
        calls.append("runtime")
        assert config == load_infer_config(config_path).config
        return _fake_runtime(base.resolve())

    receipt = provision_roi_profile(
        **args,
        runtime_loader=runtime_loader,
        gpu_preflight=preflight,
    )

    assert calls == ["gpu", "runtime"]
    profile = EngineProfileStore(args["profile_store_path"]).profiles()[
        "production-profile"
    ]
    assert [artifact.role for artifact in profile.artifacts] == [
        "adapter",
        "base_weights",
        "embedding_delta",
        "model_config",
        "processor",
        "tokenizer",
    ]
    artifacts = {artifact.role: Path(artifact.path) for artifact in profile.artifacts}
    assert artifacts["base_weights"] == base.resolve()
    assert artifacts["model_config"] == (base / "config.json").resolve()
    assert artifacts["tokenizer"] == (base / "tokenizer.json").resolve()
    assert artifacts["processor"] == (base / "preprocessor_config.json").resolve()
    resolved = load_infer_config(config_path).config
    assert artifacts["adapter"] == Path(resolved.adapter.path)
    assert artifacts["embedding_delta"] == Path(resolved.embedding_delta.path)
    assert json.loads(profile.runtime_identity_json) == {
        "model": _fake_runtime(base.resolve()).model_identity,
        "processor": _fake_runtime(
            base.resolve()
        ).qwen.processor_identity.to_artifact_dict(),
        "tokenizer": _fake_runtime(
            base.resolve()
        ).qwen.token_identity.to_artifact_dict(),
    }
    assert json.loads(profile.parser_identity_json) == {
        "id": PARSER_ID,
        "policy": PARSER_POLICY,
    }
    assert json.loads(profile.adapter_identity_json) == {"id": RESIDENT_ADAPTER_ID}
    assert json.loads(profile.transform_identity_json) == {"id": ROI_TRANSFORM_ID}
    assert profile.transformers_version == provisioning_module.transformers.__version__
    assert json.loads(profile.processor_kwargs_json) == dict(
        QWEN_IMAGE_PROCESSOR_KWARGS
    )

    launch = load_roi_launch_config(args["launch_config_path"])
    assert launch.schema_version == ROI_LAUNCH_SCHEMA_VERSION
    assert launch.profile_store_path == args["profile_store_path"]
    assert launch.receipt_store_path == Path(args["receipt_store_path"]).resolve()
    assert dict(launch.profile_selectors) == {"production": "production-profile"}
    assert launch.engine_factory.target == DEFAULT_ENGINE_FACTORY_TARGET
    assert launch.engine_factory.config_copy() == {}

    assert receipt["schema_version"] == PROVISIONING_RECEIPT_SCHEMA_VERSION
    assert receipt["profile"]["artifact_count"] == 6
    assert receipt["profile"]["processor"]["axis_bounds"] == [32, 2048]
    serialized_receipt = json.dumps(receipt, sort_keys=True)
    for private in (
        str(tmp_path),
        str(config_path),
        str(base),
        "operator-private-config-marker",
    ):
        assert private not in serialized_receipt


def test_identical_rerun_is_idempotent_and_conflicting_launch_is_rejected_before_runtime(
    tmp_path: Path,
) -> None:
    config_path, base = _authored_infer_config(tmp_path / "authored")
    args = _provision_args(tmp_path, config_path)
    runtime_calls = 0

    def runtime_loader(_config: Any) -> Any:
        nonlocal runtime_calls
        runtime_calls += 1
        return _fake_runtime(base.resolve())

    first = provision_roi_profile(
        **args,
        runtime_loader=runtime_loader,
        gpu_preflight=lambda: None,
    )
    profile_bytes = Path(args["profile_store_path"]).read_bytes()
    launch_bytes = Path(args["launch_config_path"]).read_bytes()
    second = provision_roi_profile(
        **args,
        runtime_loader=runtime_loader,
        gpu_preflight=lambda: None,
    )

    assert first == second
    assert Path(args["profile_store_path"]).read_bytes() == profile_bytes
    assert Path(args["launch_config_path"]).read_bytes() == launch_bytes
    assert runtime_calls == 2

    payload = json.loads(launch_bytes)
    payload["bind"]["port"] = 8124
    Path(args["launch_config_path"]).write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ProfileProvisioningError, match="incompatible launch"):
        provision_roi_profile(
            **args,
            runtime_loader=runtime_loader,
            gpu_preflight=lambda: None,
        )
    assert runtime_calls == 2


def test_provisions_and_idempotently_reruns_two_profiles_in_one_strict_launch(
    tmp_path: Path,
) -> None:
    config_path, base = _authored_infer_config(tmp_path / "authored")
    args = _provision_args(tmp_path, config_path)
    profile_a = _named_profile_args(
        args, profile_name="production-profile", selector="production"
    )
    profile_b = _named_profile_args(
        args, profile_name="review-profile", selector="review"
    )
    provision = {
        "runtime_loader": lambda _config: _fake_runtime(base.resolve()),
        "gpu_preflight": lambda: None,
    }

    receipt_a = provision_roi_profile(**profile_a, **provision)
    receipt_b = provision_roi_profile(**profile_b, **provision)
    profile_bytes = Path(args["profile_store_path"]).read_bytes()
    launch_bytes = Path(args["launch_config_path"]).read_bytes()

    assert receipt_a["launch"]["selector_count"] == 1
    assert receipt_a["launch"]["profile_count"] == 1
    assert receipt_b["launch"]["selector_count"] == 2
    assert receipt_b["launch"]["profile_count"] == 2
    assert set(EngineProfileStore(args["profile_store_path"]).profiles()) == {
        "production-profile",
        "review-profile",
    }
    launch = load_roi_launch_config(args["launch_config_path"])
    assert dict(launch.profile_selectors) == {
        "production": "production-profile",
        "review": "review-profile",
    }

    rerun_a = provision_roi_profile(**profile_a, **provision)
    rerun_b = provision_roi_profile(**profile_b, **provision)
    assert Path(args["profile_store_path"]).read_bytes() == profile_bytes
    assert Path(args["launch_config_path"]).read_bytes() == launch_bytes
    assert rerun_a["launch"]["selector_count"] == 2
    assert rerun_b["launch"]["selector_count"] == 2
    assert rerun_a["launch"]["profile_count"] == 2
    assert rerun_b["launch"]["profile_count"] == 2


@pytest.mark.parametrize(
    ("profile_name", "selector", "message"),
    [
        ("other-profile", "production", "selector.*different profile"),
        ("production-profile", "other-selector", "profile.*different selector"),
    ],
)
def test_rejects_selector_or_profile_name_conflict_before_runtime(
    tmp_path: Path,
    profile_name: str,
    selector: str,
    message: str,
) -> None:
    config_path, base = _authored_infer_config(tmp_path / "authored")
    args = _provision_args(tmp_path, config_path)
    provision_roi_profile(
        **args,
        runtime_loader=lambda _config: _fake_runtime(base.resolve()),
        gpu_preflight=lambda: None,
    )
    runtime_calls = 0

    def runtime_loader(_config: Any) -> Any:
        nonlocal runtime_calls
        runtime_calls += 1
        return _fake_runtime(base.resolve())

    with pytest.raises(ProfileProvisioningError, match=message):
        provision_roi_profile(
            **_named_profile_args(args, profile_name=profile_name, selector=selector),
            runtime_loader=runtime_loader,
            gpu_preflight=lambda: None,
        )
    assert runtime_calls == 0
    assert set(EngineProfileStore(args["profile_store_path"]).profiles()) == {
        "production-profile"
    }


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("bind_port", 8124),
        ("profile_store_path", "other/profiles.json"),
        ("receipt_store_path", "other/receipts.jsonl"),
        ("insertion_ack_timeout_seconds", 31.0),
    ],
)
def test_rejects_existing_launch_global_mismatch_before_runtime(
    tmp_path: Path,
    field: str,
    replacement: Any,
) -> None:
    config_path, base = _authored_infer_config(tmp_path / "authored")
    args = _provision_args(tmp_path, config_path)
    provision_roi_profile(
        **args,
        runtime_loader=lambda _config: _fake_runtime(base.resolve()),
        gpu_preflight=lambda: None,
    )
    conflicting = _named_profile_args(
        args, profile_name="review-profile", selector="review"
    )
    conflicting[field] = (
        tmp_path / replacement if isinstance(replacement, str) else replacement
    )
    runtime_calls = 0

    def runtime_loader(_config: Any) -> Any:
        nonlocal runtime_calls
        runtime_calls += 1
        return _fake_runtime(base.resolve())

    with pytest.raises(ProfileProvisioningError, match="incompatible launch"):
        provision_roi_profile(
            **conflicting,
            runtime_loader=runtime_loader,
            gpu_preflight=lambda: None,
        )
    assert runtime_calls == 0


def test_concurrent_different_profile_additions_do_not_lose_updates(
    tmp_path: Path,
) -> None:
    config_path, base = _authored_infer_config(tmp_path / "authored")
    args = _provision_args(tmp_path, config_path)
    provision_roi_profile(
        **args,
        runtime_loader=lambda _config: _fake_runtime(base.resolve()),
        gpu_preflight=lambda: None,
    )
    ready = Barrier(2)

    def add(profile_name: str, selector: str) -> dict[str, Any]:
        def runtime_loader(_config: Any) -> Any:
            ready.wait(timeout=10)
            return _fake_runtime(base.resolve())

        return provision_roi_profile(
            **_named_profile_args(args, profile_name=profile_name, selector=selector),
            runtime_loader=runtime_loader,
            gpu_preflight=lambda: None,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(add, "review-profile", "review"),
            executor.submit(add, "audit-profile", "audit"),
        ]
        receipts = [future.result(timeout=20) for future in futures]

    assert set(EngineProfileStore(args["profile_store_path"]).profiles()) == {
        "production-profile",
        "review-profile",
        "audit-profile",
    }
    launch = load_roi_launch_config(args["launch_config_path"])
    assert dict(launch.profile_selectors) == {
        "audit": "audit-profile",
        "production": "production-profile",
        "review": "review-profile",
    }
    assert sorted(receipt["launch"]["selector_count"] for receipt in receipts) == [
        2,
        3,
    ]
    assert sorted(receipt["launch"]["profile_count"] for receipt in receipts) == [
        2,
        3,
    ]


def test_cross_process_shared_profile_store_mutations_do_not_overwrite(
    tmp_path: Path,
) -> None:
    config_path, base = _authored_infer_config(tmp_path / "authored")
    args = _provision_args(tmp_path, config_path)
    provision_roi_profile(
        **args,
        runtime_loader=lambda _config: _fake_runtime(base.resolve()),
        gpu_preflight=lambda: None,
    )
    profile_b = _named_profile_args(
        {
            **args,
            "launch_config_path": tmp_path / "operator/review-launch.json",
            "receipt_store_path": tmp_path / "state/review-receipts.jsonl",
        },
        profile_name="review-profile",
        selector="review",
    )
    profile_c = _named_profile_args(
        {
            **args,
            "launch_config_path": tmp_path / "operator/audit-launch.json",
            "receipt_store_path": tmp_path / "state/audit-receipts.jsonl",
        },
        profile_name="audit-profile",
        selector="audit",
    )
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    slow_write_started = context.Event()
    process_b = context.Process(
        target=_process_provision_profile,
        args=(profile_b, base, result_queue, slow_write_started),
    )
    process_c = context.Process(
        target=_process_provision_profile,
        args=(profile_c, base, result_queue),
    )

    process_b.start()
    assert slow_write_started.wait(timeout=10)
    process_c.start()
    for process in (process_b, process_c):
        process.join(timeout=20)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            pytest.fail("cross-process provisioning did not terminate")
        assert process.exitcode == 0
    results = [result_queue.get(timeout=5) for _ in range(2)]

    assert [status for status, _receipt in results] == ["ok", "ok"]
    assert set(EngineProfileStore(args["profile_store_path"]).profiles()) == {
        "production-profile",
        "review-profile",
        "audit-profile",
    }
    assert dict(
        load_roi_launch_config(profile_b["launch_config_path"]).profile_selectors
    ) == {"review": "review-profile"}
    assert dict(
        load_roi_launch_config(profile_c["launch_config_path"]).profile_selectors
    ) == {"audit": "audit-profile"}
    assert sorted(receipt["profile_count"] for _status, receipt in results) == [2, 3]


def test_launch_publish_failure_after_profile_save_converges_on_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, base = _authored_infer_config(tmp_path / "authored")
    args = _provision_args(tmp_path, config_path)
    provision = {
        "runtime_loader": lambda _config: _fake_runtime(base.resolve()),
        "gpu_preflight": lambda: None,
    }
    provision_roi_profile(**args, **provision)
    launch_path = Path(args["launch_config_path"]).resolve()
    launch_before = launch_path.read_bytes()
    real_replace = os.replace
    failed = False

    def fail_first_launch_replace(source: str | Path, destination: str | Path) -> None:
        nonlocal failed
        if Path(destination) == launch_path and not failed:
            failed = True
            raise OSError("injected launch publication failure")
        real_replace(source, destination)

    monkeypatch.setattr(provisioning_module.os, "replace", fail_first_launch_replace)
    profile_b = _named_profile_args(
        args, profile_name="review-profile", selector="review"
    )
    with pytest.raises(OSError, match="injected launch publication failure"):
        provision_roi_profile(**profile_b, **provision)

    assert launch_path.read_bytes() == launch_before
    assert set(EngineProfileStore(args["profile_store_path"]).profiles()) == {
        "production-profile",
        "review-profile",
    }
    receipt = provision_roi_profile(**profile_b, **provision)
    assert dict(load_roi_launch_config(launch_path).profile_selectors) == {
        "production": "production-profile",
        "review": "review-profile",
    }
    assert receipt["launch"]["selector_count"] == 2
    assert receipt["launch"]["profile_count"] == 2
    assert list(launch_path.parent.glob(f".{launch_path.name}.*.tmp")) == []


def test_launch_directory_fsync_failure_is_repaired_by_idempotent_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, base = _authored_infer_config(tmp_path / "authored")
    args = _provision_args(tmp_path, config_path)
    launch_path = Path(args["launch_config_path"]).resolve()
    provision = {
        "runtime_loader": lambda _config: _fake_runtime(base.resolve()),
        "gpu_preflight": lambda: None,
    }
    real_fsync_directory = provisioning_module._fsync_directory
    fail_once = True

    def fail_launch_directory_once(directory: Path) -> None:
        nonlocal fail_once
        if directory == launch_path.parent and fail_once:
            fail_once = False
            raise OSError("injected launch directory fsync failure")
        real_fsync_directory(directory)

    monkeypatch.setattr(
        provisioning_module, "_fsync_directory", fail_launch_directory_once
    )
    with pytest.raises(OSError, match="injected launch directory fsync failure"):
        provision_roi_profile(**args, **provision)

    launch_bytes = launch_path.read_bytes()
    fsynced: list[Path] = []

    def tracking_fsync_directory(directory: Path) -> None:
        fsynced.append(directory)
        real_fsync_directory(directory)

    monkeypatch.setattr(
        provisioning_module, "_fsync_directory", tracking_fsync_directory
    )
    receipt = provision_roi_profile(**args, **provision)
    assert launch_path.read_bytes() == launch_bytes
    assert fsynced == [launch_path.parent]
    assert receipt["launch"]["selector_count"] == 1
    assert receipt["launch"]["profile_count"] == 1


def test_stale_launch_missing_saved_profile_fails_closed_and_can_recover(
    tmp_path: Path,
) -> None:
    config_path, base = _authored_infer_config(tmp_path / "authored")
    args = _provision_args(tmp_path, config_path)
    provision = {
        "runtime_loader": lambda _config: _fake_runtime(base.resolve()),
        "gpu_preflight": lambda: None,
    }
    profile_a = _named_profile_args(
        args, profile_name="production-profile", selector="production"
    )
    profile_b = _named_profile_args(
        args, profile_name="review-profile", selector="review"
    )
    provision_roi_profile(**profile_a, **provision)
    provision_roi_profile(**profile_b, **provision)
    launch_path = Path(args["launch_config_path"])
    launch_bytes = launch_path.read_bytes()
    store_path = Path(args["profile_store_path"])
    store_payload = json.loads(store_path.read_text(encoding="utf-8"))
    del store_payload["profiles"]["production-profile"]
    store_path.write_text(json.dumps(store_payload), encoding="utf-8")

    with pytest.raises(
        ProfileProvisioningError, match="missing saved profiles: count=1"
    ):
        provision_roi_profile(**profile_b, **provision)
    assert launch_path.read_bytes() == launch_bytes

    recovered = provision_roi_profile(**profile_a, **provision)
    assert recovered["launch"]["selector_count"] == 2
    assert recovered["launch"]["profile_count"] == 2
    assert launch_path.read_bytes() == launch_bytes


@pytest.mark.parametrize("drift", ["artifact", "config"])
def test_existing_profile_rejects_artifact_or_config_drift(
    tmp_path: Path, drift: str
) -> None:
    config_path, base = _authored_infer_config(tmp_path / "authored")
    args = _provision_args(tmp_path, config_path)
    provision_roi_profile(
        **args,
        runtime_loader=lambda _config: _fake_runtime(base.resolve()),
        gpu_preflight=lambda: None,
    )
    if drift == "artifact":
        (base / "model.safetensors").write_bytes(b"model-v2")
    else:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        payload["generation"]["max_new_tokens"] = 256
        config_path.write_text(
            yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
        )

    with pytest.raises(ProfileContractError, match="immutable profile"):
        provision_roi_profile(
            **args,
            runtime_loader=lambda _config: _fake_runtime(base.resolve()),
            gpu_preflight=lambda: None,
        )


def test_launch_publication_fsyncs_file_and_directory_without_temporary_residue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, base = _authored_infer_config(tmp_path / "authored")
    args = _provision_args(tmp_path, config_path)
    provision = {
        "runtime_loader": lambda _config: _fake_runtime(base.resolve()),
        "gpu_preflight": lambda: None,
    }
    provision_roi_profile(**args, **provision)
    observed_modes: list[int] = []
    fsynced_directories: list[tuple[str, Path]] = []
    replace_calls: list[tuple[Path, Path]] = []
    real_fsync = os.fsync
    real_replace = os.replace
    real_launch_fsync_directory = provisioning_module._fsync_directory
    real_profile_fsync_directory = inference_profiles_module._fsync_directory

    def tracking_fsync(descriptor: int) -> None:
        observed_modes.append(os.fstat(descriptor).st_mode)
        real_fsync(descriptor)

    def tracking_replace(source: str | Path, destination: str | Path) -> None:
        replace_calls.append((Path(source), Path(destination)))
        real_replace(source, destination)

    def tracking_launch_fsync_directory(directory: Path) -> None:
        fsynced_directories.append(("launch", directory))
        real_launch_fsync_directory(directory)

    def tracking_profile_fsync_directory(directory: Path) -> None:
        fsynced_directories.append(("profile", directory))
        real_profile_fsync_directory(directory)

    monkeypatch.setattr(provisioning_module.os, "fsync", tracking_fsync)
    monkeypatch.setattr(provisioning_module.os, "replace", tracking_replace)
    monkeypatch.setattr(
        provisioning_module, "_fsync_directory", tracking_launch_fsync_directory
    )
    monkeypatch.setattr(
        inference_profiles_module,
        "_fsync_directory",
        tracking_profile_fsync_directory,
    )
    provision_roi_profile(
        **_named_profile_args(args, profile_name="review-profile", selector="review"),
        **provision,
    )

    profile_store_path = Path(args["profile_store_path"])
    launch_path = Path(args["launch_config_path"]).resolve()
    replace_destinations = [destination for _source, destination in replace_calls]
    assert profile_store_path in replace_destinations
    assert launch_path in replace_destinations
    assert replace_destinations.index(profile_store_path) < replace_destinations.index(
        launch_path
    )
    assert fsynced_directories == [
        ("profile", profile_store_path.parent),
        ("launch", launch_path.parent),
    ]
    assert any(stat.S_ISREG(mode) for mode in observed_modes)
    assert any(stat.S_ISDIR(mode) for mode in observed_modes)
    assert launch_path.read_bytes().endswith(b"\n")
    assert list(launch_path.parent.glob(f".{launch_path.name}.*.tmp")) == []
    assert dict(load_roi_launch_config(launch_path).profile_selectors) == {
        "production": "production-profile",
        "review": "review-profile",
    }


def test_launch_publication_drift_check_rejects_changed_input_without_replacement(
    tmp_path: Path,
) -> None:
    config_path, base = _authored_infer_config(tmp_path / "authored")
    args = _provision_args(tmp_path, config_path)
    provision_roi_profile(
        **args,
        runtime_loader=lambda _config: _fake_runtime(base.resolve()),
        gpu_preflight=lambda: None,
    )
    launch_path = Path(args["launch_config_path"]).resolve()
    expected_launch_fingerprint = provisioning_module._launch_file_fingerprint(
        launch_path
    )
    payload = json.loads(launch_path.read_text(encoding="utf-8"))
    drifted = launch_path.read_bytes() + b"\n"
    launch_path.write_bytes(drifted)

    with pytest.raises(ProfileProvisioningError, match="changed concurrently"):
        provisioning_module._publish_launch_atomic(
            launch_path,
            payload,
            expected_launch_fingerprint=expected_launch_fingerprint,
        )

    assert launch_path.read_bytes() == drifted
    assert list(launch_path.parent.glob(f".{launch_path.name}.*.tmp")) == []


@pytest.mark.parametrize(
    ("tokens", "device_count", "available"),
    [
        ((), 0, False),
        (("0", "1"), 2, True),
        (("0",), 0, False),
        (("0",), 1, False),
    ],
)
def test_single_cuda_preflight_fails_closed(
    tokens: tuple[str, ...], device_count: int, available: bool
) -> None:
    with pytest.raises(ProfileProvisioningError, match="exactly one"):
        require_single_visible_cuda(
            visible_cuda_resolver=lambda **_kwargs: tokens,
            cuda_device_count=lambda: device_count,
            cuda_is_available=lambda: available,
        )


def test_single_cuda_preflight_and_production_runtime_default_are_exact() -> None:
    assert (
        require_single_visible_cuda(
            visible_cuda_resolver=lambda **_kwargs: ("GPU-test",),
            cuda_device_count=lambda: 1,
            cuda_is_available=lambda: True,
        )
        == "GPU-test"
    )
    parameters = inspect.signature(provision_roi_profile).parameters
    assert parameters["runtime_loader"].default is assemble_runtime
    assert parameters["gpu_preflight"].default is require_single_visible_cuda


def test_relative_profile_store_is_rejected_without_loading_runtime(
    tmp_path: Path,
) -> None:
    config_path, _base = _authored_infer_config(tmp_path / "authored")
    args = _provision_args(tmp_path, config_path)
    args["profile_store_path"] = Path("relative/profiles.json")
    loaded = False

    def runtime_loader(_config: Any) -> Any:
        nonlocal loaded
        loaded = True
        raise AssertionError("runtime must not load")

    with pytest.raises(ProfileProvisioningError, match="must be absolute"):
        provision_roi_profile(
            **args,
            runtime_loader=runtime_loader,
            gpu_preflight=lambda: None,
        )
    assert loaded is False


@pytest.mark.parametrize("lock_owner", ["launch", "profile_store"])
def test_output_paths_cannot_collide_with_persistent_provision_locks(
    tmp_path: Path, lock_owner: str
) -> None:
    config_path, _base = _authored_infer_config(tmp_path / "authored")
    args = _provision_args(tmp_path, config_path)
    launch_path = Path(args["launch_config_path"]).resolve()
    profile_store = EngineProfileStore(args["profile_store_path"])
    args["receipt_store_path"] = (
        provisioning_module._launch_lock_path(launch_path)
        if lock_owner == "launch"
        else profile_store.lock_path
    )

    with pytest.raises(ProfileProvisioningError, match="collide with provision locks"):
        provision_roi_profile(
            **args,
            runtime_loader=lambda _config: pytest.fail("runtime must not load"),
            gpu_preflight=lambda: None,
        )


def test_launch_and_profile_store_lock_paths_cannot_alias(
    tmp_path: Path,
) -> None:
    config_path, _base = _authored_infer_config(tmp_path / "authored")
    args = _provision_args(tmp_path, config_path)
    args["launch_config_path"] = tmp_path / "operator/roi.json"
    args["profile_store_path"] = (tmp_path / "operator/roi.json.provision").resolve()

    with pytest.raises(ProfileProvisioningError, match="collide with provision locks"):
        provision_roi_profile(
            **args,
            runtime_loader=lambda _config: pytest.fail("runtime must not load"),
            gpu_preflight=lambda: None,
        )


def test_cli_forwards_all_explicit_bounds_and_1024_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    captured: dict[str, Any] = {}
    expected_receipt = {"schema_version": PROVISIONING_RECEIPT_SCHEMA_VERSION}

    def fake_provision(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return expected_receipt

    monkeypatch.setattr(cli_module, "provision_roi_profile", fake_provision)
    exit_code = cli_module.main(
        [
            "--infer-config",
            "infer.yaml",
            "--profile-store",
            str((tmp_path / "profiles.json").resolve()),
            "--launch-config",
            "launch.json",
            "--receipt-store",
            "receipts.jsonl",
            "--profile-name",
            "profile-a",
            "--selector",
            "safe",
            "--bind-host",
            "127.0.0.1",
            "--bind-port",
            "8123",
            "--ack-timeout-seconds",
            "30",
            "--min-axis-pixels",
            "32",
            "--max-axis-pixels",
            "2048",
            "--max-total-pixels",
            "2097152",
            "--deadline-seconds",
            "20",
        ]
    )

    assert exit_code == 0
    assert captured["default_width"] == 1024
    assert captured["default_height"] == 1024
    assert captured["min_axis_pixels"] == 32
    assert captured["max_axis_pixels"] == 2048
    assert captured["max_total_pixels"] == 2_097_152
    assert captured["deadline_seconds"] == 20.0
    assert json.loads(capsys.readouterr().out) == expected_receipt
