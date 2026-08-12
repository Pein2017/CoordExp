from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from src.artifacts import provenance


def _available(value: object) -> dict[str, object]:
    return {"status": "available", "value": value}


def _native_identity(path: Path, *, build_id: str) -> dict[str, object]:
    data = path.read_bytes()
    return {
        "relative_path": path.name,
        "imported_origin": _available(str(path)),
        "sha256": _available(hashlib.sha256(data).hexdigest()),
        "size_bytes": _available(len(data)),
        "elf_build_id": _available(build_id),
    }


def _native_fixture(tmp_path: Path) -> tuple[dict[str, object], dict[str, Path]]:
    paths: dict[str, Path] = {}
    for name in (
        "libtorch_cuda.so",
        "libc10_cuda.so",
        "libcudnn.so.9",
        "libcudnn_adv.so.9",
        "libcudart.so.12",
        "libcuda.so.550.54.15",
    ):
        path = tmp_path / name
        path.write_bytes(f"fixture:{name}".encode())
        paths[name] = path
    provenance_fixture: dict[str, object] = {
        "dependencies": {
            "torch": {
                "native_identities": {
                    "libtorch_cuda": _native_identity(
                        paths["libtorch_cuda.so"], build_id="build-torch"
                    ),
                    "libc10_cuda": _native_identity(
                        paths["libc10_cuda.so"], build_id="build-c10"
                    ),
                    "cudnn": _native_identity(
                        paths["libcudnn.so.9"], build_id="build-cudnn"
                    ),
                    "cudnn_adv": _native_identity(
                        paths["libcudnn_adv.so.9"], build_id="build-cudnn-adv"
                    ),
                }
            },
            "cuda-runtime": {
                **_native_identity(paths["libcudart.so.12"], build_id="build-cudart"),
                "distribution_origin": _available(str(paths["libcudart.so.12"])),
            },
        }
    }
    return provenance_fixture, paths


def _install_native_fixture(
    monkeypatch: pytest.MonkeyPatch,
    paths: dict[str, Path],
) -> None:
    monkeypatch.setattr(provenance, "_cuda_runtime_is_initialized", lambda: True)
    monkeypatch.setattr(
        provenance,
        "_mapped_shared_object_origins",
        lambda: {name: (path,) for name, path in paths.items()},
    )
    build_ids = {
        "libtorch_cuda.so": "build-torch",
        "libc10_cuda.so": "build-c10",
        "libcudnn.so.9": "build-cudnn",
        "libcudnn_adv.so.9": "build-cudnn-adv",
        "libcudart.so.12": "build-cudart",
        "libcuda.so.550.54.15": "build-driver",
    }
    monkeypatch.setattr(
        provenance,
        "_elf_build_id",
        lambda path: _available(build_ids[path.name]),
    )
    driver_path = paths["libcuda.so.550.54.15"]
    driver_data = driver_path.read_bytes()
    monkeypatch.setattr(
        provenance,
        "_PINNED_DRIVER_NATIVE_IDENTITY",
        {
            "soname_family": "libcuda.so",
            "origin": str(driver_path),
            "sha256": hashlib.sha256(driver_data).hexdigest(),
            "size_bytes": len(driver_data),
            "elf_build_id": "build-driver",
        },
    )


def test_native_execution_attestation_binds_every_mapped_cuda_dso(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preload, paths = _native_fixture(tmp_path)
    _install_native_fixture(monkeypatch, paths)

    result = provenance.collect_mapped_native_execution_attestation(provenance=preload)

    assert result["admitted"] is True, result["mismatches"]
    assert result["mismatches"] == []
    assert set(result["components"]) == {
        "libc10_cuda",
        "libcuda",
        "libcudart",
        "libcudnn",
        "libcudnn_adv",
        "libtorch_cuda",
    }
    assert result["mapped_cudnn_components"] == ["libcudnn", "libcudnn_adv"]


@pytest.mark.parametrize(
    "failure",
    [
        "mutation",
        "wrong_build_id",
        "wrong_origin",
        "missing_driver",
        "unknown_cudnn",
    ],
)
def test_native_execution_attestation_fails_closed(
    failure: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preload, paths = _native_fixture(tmp_path)
    _install_native_fixture(monkeypatch, paths)
    if failure == "mutation":
        paths["libcudart.so.12"].write_bytes(b"mutated")
    elif failure == "wrong_build_id":
        observed_build_id = provenance._elf_build_id
        monkeypatch.setattr(
            provenance,
            "_elf_build_id",
            lambda path: (
                _available("wrong-build-id")
                if path.name == "libcudart.so.12"
                else observed_build_id(path)
            ),
        )
    elif failure == "wrong_origin":
        wrong = tmp_path / "alternate" / "libcudart.so.12"
        wrong.parent.mkdir()
        wrong.write_bytes(paths["libcudart.so.12"].read_bytes())
        paths["libcudart.so.12"] = wrong
        _install_native_fixture(monkeypatch, paths)
    elif failure == "missing_driver":
        paths.pop("libcuda.so.550.54.15")
        monkeypatch.setattr(
            provenance,
            "_mapped_shared_object_origins",
            lambda: {name: (path,) for name, path in paths.items()},
        )
    else:
        unknown = tmp_path / "libcudnn_future.so.9"
        unknown.write_bytes(b"unrecognized mapped cudnn")
        paths[unknown.name] = unknown

    result = provenance.collect_mapped_native_execution_attestation(provenance=preload)

    assert result["admitted"] is False
    assert result["mismatches"]
    with pytest.raises(provenance.NativeExecutionAttestationError):
        provenance.require_mapped_native_execution_attestation(provenance=preload)


def test_native_execution_attestation_requires_initialized_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(provenance, "_cuda_runtime_is_initialized", lambda: False)

    result = provenance.collect_mapped_native_execution_attestation(provenance={})

    assert result["admitted"] is False
    assert result["mismatches"] == ["cuda_initialized"]
