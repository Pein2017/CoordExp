from __future__ import annotations

import json
import multiprocessing
import time
from functools import partial
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from src.common.errors import RuntimeContractError
from src.inference.execution_model import (
    build_execution_model_composition_key,
    resolve_execution_model,
    validate_execution_model_receipt,
)
from src.inference.model_assets import build_model_snapshot_manifest


def _write_snapshot(root: Path, *, weight: bytes = b"weights") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_vl",
                "architectures": ["Qwen3VLForConditionalGeneration"],
                "tie_word_embeddings": True,
                "dtype": "bfloat16",
            }
        ),
        encoding="utf-8",
    )
    (root / "tokenizer.json").write_text("{}", encoding="utf-8")
    (root / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    save_file(
        {
            "model.language_model.embed_tokens.weight": torch.full(
                (2, 2),
                float(sum(weight) % 251),
                dtype=torch.bfloat16,
            )
        },
        root / "model.safetensors",
    )
    return root


def _identity(name: str, fingerprint: str) -> dict[str, object]:
    return {
        "kind": name,
        "fingerprint": fingerprint,
        "files": [{"relative_path": f"{name}.bin", "sha256": fingerprint}],
    }


def _concurrent_materializer(marker: Path, snapshot_root: Path) -> dict[str, object]:
    with marker.open("a", encoding="utf-8") as handle:
        handle.write("build\n")
    time.sleep(0.2)
    _write_snapshot(snapshot_root, weight=b"derived")
    return {"owner": "concurrency-test"}


def _resolve_in_subprocess(
    base: Path,
    cache_root: Path,
    marker: Path,
    queue: multiprocessing.Queue,
) -> None:
    try:
        receipt = resolve_execution_model(
            base_model_path=base,
            target_dtype="bf16",
            adapter_identity=_identity("adapter", "adapter-a"),
            cache_root=cache_root,
            materialize_snapshot=partial(_concurrent_materializer, marker),
        )
        queue.put(("ok", receipt["receipt_fingerprint"]))
    except BaseException as exc:  # pragma: no cover - surfaced in parent assertion
        queue.put(("error", repr(exc)))


def test_composition_key_ignores_source_paths_and_changes_with_payload() -> None:
    base_a = {
        "root": "/first/base",
        "version": "v1",
        "fingerprint": "base-fp",
        "file_count": 1,
        "files": [{"relative_path": "model.safetensors", "sha256": "base"}],
    }
    base_b = {**base_a, "root": "/second/base"}
    adapter_a = {
        **_identity("adapter", "adapter-a"),
        "base_model_name_or_path": "/first/base",
    }
    delta_a = {
        **_identity("delta", "delta-a"),
        "base_model_path": "/first/base",
    }
    common = {
        "adapter_identity": adapter_a,
        "embedding_delta_identity": delta_a,
        "target_dtype": "bf16",
        "package_versions": {"peft": "0.17.1", "transformers": "4.57.1"},
    }
    key_a = build_execution_model_composition_key(base_manifest=base_a, **common)
    key_b = build_execution_model_composition_key(
        base_manifest=base_b,
        **{
            **common,
            "adapter_identity": {
                **adapter_a,
                "base_model_name_or_path": "/second/base",
            },
            "embedding_delta_identity": {
                **delta_a,
                "base_model_path": "/second/base",
            },
        },
    )
    assert key_a == key_b

    changed_adapter = build_execution_model_composition_key(
        base_manifest=base_a,
        **{**common, "adapter_identity": _identity("adapter", "adapter-b")},
    )
    changed_delta = build_execution_model_composition_key(
        base_manifest=base_a,
        **{
            **common,
            "embedding_delta_identity": _identity("delta", "delta-b"),
        },
    )
    assert changed_adapter != key_a
    assert changed_delta != key_a


def test_composition_key_binds_every_materializer_dependency() -> None:
    base = {
        "root": "/base",
        "version": "v1",
        "fingerprint": "base-fp",
        "file_count": 1,
        "files": [{"relative_path": "model.safetensors", "sha256": "base"}],
    }
    versions = {
        "transformers": "4.57.1",
        "peft": "0.17.1",
        "torch": "2.9.1",
        "safetensors": "0.6.2",
    }
    baseline = build_execution_model_composition_key(
        base_manifest=base,
        adapter_identity=None,
        embedding_delta_identity=None,
        target_dtype="bf16",
        package_versions=versions,
    )
    for package in versions:
        changed = build_execution_model_composition_key(
            base_manifest=base,
            adapter_identity=None,
            embedding_delta_identity=None,
            target_dtype="bf16",
            package_versions={**versions, package: "changed"},
        )
        assert changed != baseline, package


def test_base_only_receipt_revalidates_snapshot_bytes(tmp_path: Path) -> None:
    base = _write_snapshot(tmp_path / "base")
    receipt = resolve_execution_model(
        base_model_path=base,
        target_dtype="bf16",
    )
    assert receipt["mode"] == "base_only"
    assert receipt["model_path"] == str(base.resolve())
    assert validate_execution_model_receipt(receipt) == receipt

    (base / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(RuntimeContractError, match="snapshot"):
        validate_execution_model_receipt(receipt)


def test_composed_cache_hit_reuses_published_snapshot(tmp_path: Path) -> None:
    base = _write_snapshot(tmp_path / "base")
    builds: list[Path] = []

    def materialize(snapshot_root: Path) -> dict[str, object]:
        builds.append(snapshot_root)
        _write_snapshot(snapshot_root, weight=b"derived")
        return {"owner": "test"}

    kwargs = {
        "base_model_path": base,
        "target_dtype": "bf16",
        "adapter_identity": _identity("adapter", "adapter-a"),
        "embedding_delta_identity": _identity("delta", "delta-a"),
        "cache_root": tmp_path / "cache",
        "materialize_snapshot": materialize,
    }
    first = resolve_execution_model(**kwargs)
    second = resolve_execution_model(**kwargs)

    assert first == second
    assert first["mode"] == "materialized"
    assert len(builds) == 1
    assert Path(first["receipt_path"]).is_file()
    assert validate_execution_model_receipt(first) == first


def test_corrupt_completed_cache_fails_without_rebuild(tmp_path: Path) -> None:
    base = _write_snapshot(tmp_path / "base")
    builds = 0

    def materialize(snapshot_root: Path) -> dict[str, object]:
        nonlocal builds
        builds += 1
        _write_snapshot(snapshot_root, weight=b"derived")
        return {"owner": "test"}

    kwargs = {
        "base_model_path": base,
        "target_dtype": "bf16",
        "adapter_identity": _identity("adapter", "adapter-a"),
        "cache_root": tmp_path / "cache",
        "materialize_snapshot": materialize,
    }
    receipt = resolve_execution_model(**kwargs)
    Path(receipt["model_path"]).joinpath("model.safetensors").write_bytes(b"corrupt")

    with pytest.raises(RuntimeContractError, match="snapshot"):
        resolve_execution_model(**kwargs)
    assert builds == 1


def test_failed_build_never_publishes_completed_directory(tmp_path: Path) -> None:
    base = _write_snapshot(tmp_path / "base")
    cache_root = tmp_path / "cache"

    def fail(snapshot_root: Path) -> dict[str, object]:
        _write_snapshot(snapshot_root)
        raise RuntimeError("injected failure")

    with pytest.raises(RuntimeError, match="injected failure"):
        resolve_execution_model(
            base_model_path=base,
            target_dtype="bf16",
            adapter_identity=_identity("adapter", "adapter-a"),
            cache_root=cache_root,
            materialize_snapshot=fail,
        )

    completed = [
        path
        for path in cache_root.iterdir()
        if path.is_dir() and path.name not in {".locks", ".staging"}
    ]
    assert completed == []


def test_materialized_snapshot_rejects_unreadable_model_payload(
    tmp_path: Path,
) -> None:
    base = _write_snapshot(tmp_path / "base")
    cache_root = tmp_path / "cache"

    def materialize(snapshot_root: Path) -> dict[str, object]:
        snapshot_root.mkdir(parents=True)
        (snapshot_root / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "qwen3_vl",
                    "architectures": ["Qwen3VLForConditionalGeneration"],
                    "tie_word_embeddings": True,
                    "dtype": "bfloat16",
                }
            ),
            encoding="utf-8",
        )
        (snapshot_root / "tokenizer.json").write_text("{}", encoding="utf-8")
        (snapshot_root / "preprocessor_config.json").write_text(
            "{}", encoding="utf-8"
        )
        (snapshot_root / "model.safetensors").write_bytes(b"not-a-model")
        return {"owner": "malformed-test"}

    with pytest.raises(RuntimeContractError) as exc_info:
        resolve_execution_model(
            base_model_path=base,
            target_dtype="bf16",
            adapter_identity=_identity("adapter", "adapter-a"),
            cache_root=cache_root,
            materialize_snapshot=materialize,
        )
    assert exc_info.value.code == "inference.execution_model_snapshot_tensor_invalid"
    assert not any(
        path.is_dir() and path.name not in {".locks", ".staging"}
        for path in cache_root.iterdir()
    )


def test_concurrent_cache_miss_has_exactly_one_builder(tmp_path: Path) -> None:
    base = _write_snapshot(tmp_path / "base")
    cache_root = tmp_path / "cache"
    marker = tmp_path / "builders.txt"
    context = multiprocessing.get_context("fork")
    queue = context.Queue()
    processes = [
        context.Process(
            target=_resolve_in_subprocess,
            args=(base, cache_root, marker, queue),
        )
        for _ in range(3)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0
    results = [queue.get(timeout=1) for _ in processes]

    assert {status for status, _ in results} == {"ok"}
    assert len({fingerprint for _, fingerprint in results}) == 1
    assert marker.read_text(encoding="utf-8").splitlines() == ["build"]


def test_receipt_rejects_source_identity_or_manifest_tampering(tmp_path: Path) -> None:
    base = _write_snapshot(tmp_path / "base")
    receipt = resolve_execution_model(base_model_path=base, target_dtype="bf16")
    tampered = json.loads(json.dumps(receipt))
    tampered["source_identity"]["base"]["fingerprint"] = "wrong"
    with pytest.raises(RuntimeContractError, match="identity"):
        validate_execution_model_receipt(tampered)

    manifest = build_model_snapshot_manifest(base)
    assert manifest["file_count"] == 4
