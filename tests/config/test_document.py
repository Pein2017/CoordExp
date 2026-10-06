"""Contract tests for inherited documents, independent of model execution."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from src.common.errors import ConfigContractError
from src.config.document import (
    copy_mapping,
    load_inherited_mapping,
    reject_required_placeholders,
)


def test_inheritance_preserves_authorship_and_ordered_source_identity(tmp_path: Path) -> None:
    parent = tmp_path / "base.yaml"
    parent.write_text("model:\n  path: parent/model\n  options: {dtype: bf16, tags: [old]}\n")
    leaf = tmp_path / "leaf.yaml"
    leaf.write_text("extends: base.yaml\nmodel:\n  options: {tags: [new]}\n")

    payload, origins, sources = load_inherited_mapping(leaf)
    assert payload == {"model": {"path": "parent/model", "options": {"dtype": "bf16", "tags": ["new"]}}}
    assert origins == {"model.path": parent, "model.options.dtype": parent, "model.options.tags": leaf}
    assert [source.path for source in sources] == [parent, leaf]
    assert [source.sha256 for source in sources] == [
        hashlib.sha256(path.read_bytes()).hexdigest() for path in (parent, leaf)
    ]
    payload["model"]["options"]["tags"].append("mutated")
    assert load_inherited_mapping(leaf)[0]["model"]["options"]["tags"] == ["new"]


def test_variant_replacement_is_an_explicit_inference_policy(tmp_path: Path) -> None:
    parent = tmp_path / "base.yaml"
    parent.write_text("backend: {type: hf, hf: {attention: eager}}\n")
    leaf = tmp_path / "leaf.yaml"
    leaf.write_text("extends: base.yaml\nbackend: {type: vllm, vllm: {limit: 4}}\n")

    ordinary, ordinary_origins, _ = load_inherited_mapping(leaf)
    replaced, replaced_origins, _ = load_inherited_mapping(leaf, replace_discriminated=True)
    assert ordinary["backend"]["hf"] == {"attention": "eager"}
    assert ordinary_origins["backend.hf.attention"] == parent
    assert replaced == {"backend": {"type": "vllm", "vllm": {"limit": 4}}}
    assert replaced_origins == {"backend.type": leaf, "backend.vllm.limit": leaf}


@pytest.mark.parametrize(
    ("content", "code"),
    [
        ("extends: [parent.yaml]\n", "config.extends_shape"),
        ("model: {extends: other.yaml}\n", "config.nested_extends"),
        ("items: [{extends: other.yaml}]\n", "config.nested_extends"),
        ("[not, a, mapping]\n", "config.mapping"),
        ("extends: absent.yaml\n", "config.missing_file"),
    ],
)
def test_document_failures_keep_stable_error_codes(tmp_path: Path, content: str, code: str) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(content)
    with pytest.raises(ConfigContractError) as error:
        load_inherited_mapping(path)
    assert error.value.code == code


def test_cycle_reports_the_actual_chain(tmp_path: Path) -> None:
    a, b = tmp_path / "a.yaml", tmp_path / "b.yaml"
    a.write_text("extends: b.yaml\n")
    b.write_text("extends: a.yaml\n")
    with pytest.raises(ConfigContractError) as error:
        load_inherited_mapping(a)
    assert error.value.code == "config.extends_cycle"
    assert error.value.context["cycle"] == f"{a} -> {b} -> {a}"


def test_null_cannot_erase_inherited_value(tmp_path: Path) -> None:
    parent = tmp_path / "base.yaml"
    parent.write_text("model: {dtype: bf16}\n")
    leaf = tmp_path / "leaf.yaml"
    leaf.write_text("extends: base.yaml\nmodel: {dtype: null}\n")
    with pytest.raises(ConfigContractError) as error:
        load_inherited_mapping(leaf)
    assert error.value.code == "config.null_inherited_delete"
    assert error.value.context == {"field": "model.dtype", "declaring_config_path": str(leaf)}


def test_placeholders_are_checked_after_child_override(tmp_path: Path) -> None:
    parent = tmp_path / "base.yaml"
    parent.write_text("model: {path: REQUIRED}\n")
    leaf = tmp_path / "leaf.yaml"
    leaf.write_text("extends: base.yaml\nmodel: {path: explicit/model}\n")
    reject_required_placeholders(load_inherited_mapping(leaf)[0])
    with pytest.raises(ConfigContractError) as error:
        reject_required_placeholders(load_inherited_mapping(parent)[0])
    assert error.value.code == "config.required_placeholder"
    assert error.value.context["field"] == "model.path"


def test_yaml_aliases_are_copied_per_occurrence() -> None:
    original = yaml.safe_load("a: &shared [{path: relative/model}]\nb: *shared\n")
    assert original["a"] is original["b"]
    copied = copy_mapping(original)
    copied["a"][0]["path"] = "/resolved/model"
    assert copied["b"][0]["path"] == "relative/model"
    assert original["a"][0]["path"] == "relative/model"
