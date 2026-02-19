from __future__ import annotations

from collections.abc import Mapping

from src.sft import _collect_dependency_provenance


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
