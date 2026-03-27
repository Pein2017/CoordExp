from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from src.sft import _build_source_path_identity, _collect_dependency_provenance


def _assert_mod_info(value: object) -> None:
    assert isinstance(value, Mapping)
    # We either record version/file, or record an import error.
    assert (
        "error" in value
        or "version" in value
        or "file" in value
    ), f"unexpected module info shape: {value!r}"


def test_collect_dependency_provenance_smoke() -> None:
    deps = _collect_dependency_provenance()

    assert isinstance(deps, Mapping)
    assert "python" in deps

    _assert_mod_info(deps.get("transformers"))
    _assert_mod_info(deps.get("torch"))
    _assert_mod_info(deps.get("vllm"))
    _assert_mod_info(deps.get("swift"))


def test_build_source_path_identity_includes_sha256_for_small_files(
    tmp_path: Path,
) -> None:
    sample = tmp_path / "sample.jsonl"
    sample.write_text('{"id": 1}\n', encoding="utf-8")

    identity = _build_source_path_identity(sample)

    assert isinstance(identity, Mapping)
    assert identity["exists"] is True
    assert identity["resolved_path"] == str(sample.resolve())
    assert isinstance(identity.get("sha256"), str)
