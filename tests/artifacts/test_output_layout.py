import hashlib
import json
from pathlib import Path

import pytest

from src.artifacts.output_layout import scan_output_root


def _write_native_adapter_payload(root):
    adapter = root / "adapter"
    adapter.mkdir(parents=True)
    payloads = {
        "README.md": b"# Generated adapter card\n",
        "adapter_config.json": b"{}",
        "adapter_model.safetensors": b"opaque tensor payload",
    }
    entries = []
    for name, content in payloads.items():
        (adapter / name).write_bytes(content)
        entries.append(
            {
                "relative_path": name,
                "size_bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    (root / "inference_payload_manifest.json").write_text(
        json.dumps(
            {
                "schema": "coordexp-infras-inference-checkpoint-payload-manifest",
                "schema_version": 1,
                "adapter": {
                    "relative_root": "adapter",
                    "status": "present",
                    "files": entries,
                },
            }
        ),
        encoding="utf-8",
    )
    return adapter


def test_artifacts_are_allowed_but_code_prose_environments_and_bytecode_are_not(tmp_path):
    (tmp_path / "result.json").write_text("{}")
    assert scan_output_root(tmp_path)["passed"]
    for name in ["worker.py", "result.md", "launch.sh", "worker.pyc"]:
        (tmp_path / name).write_text("content")
    (tmp_path / ".venv").mkdir()
    result = scan_output_root(tmp_path)
    assert len(result["findings"]) == 5
    assert not result["passed"]


def test_links_are_reported_without_following_external_trees(tmp_path):
    root = tmp_path / "outputs"
    root.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    (external / "not_scanned.py").write_text("content")
    (root / "images").symlink_to(external, target_is_directory=True)
    assert scan_output_root(root)["passed"]
    (root / "missing").symlink_to(tmp_path / "gone")
    result = scan_output_root(root)
    assert result["symlinks_not_followed"] == 2
    assert result["findings"][0]["reason"] == "broken_symlink"


def test_missing_root_cannot_pass(tmp_path):
    with pytest.raises(FileNotFoundError):
        scan_output_root(tmp_path / "missing")


def test_native_manifest_bound_adapter_readme_is_allowed(tmp_path):
    _write_native_adapter_payload(tmp_path)

    result = scan_output_root(tmp_path)

    assert result["passed"]
    assert result["findings"] == []


def test_nested_native_adapter_cards_are_allowed_but_loose_readme_is_not(tmp_path):
    _write_native_adapter_payload(tmp_path / "checkpoint-one" / "payload")
    _write_native_adapter_payload(tmp_path / "checkpoint-two" / "payload")
    loose_readme = tmp_path / "README.md"
    loose_readme.write_text("unbound prose\n", encoding="utf-8")

    result = scan_output_root(tmp_path)

    assert not result["passed"]
    assert [Path(item["path"]) for item in result["findings"]] == [loose_readme]


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_manifest",
        "malformed_manifest",
        "wrong_schema",
        "wrong_relative_root",
        "missing_readme_entry",
        "altered_readme",
        "missing_config",
        "missing_model",
    ],
)
def test_adapter_readme_requires_exact_manifest_and_package(tmp_path, mutation):
    adapter = _write_native_adapter_payload(tmp_path)
    manifest = tmp_path / "inference_payload_manifest.json"
    readme = adapter / "README.md"
    if mutation == "missing_manifest":
        manifest.unlink()
    elif mutation == "malformed_manifest":
        manifest.write_text("{", encoding="utf-8")
    elif mutation == "wrong_schema":
        data = json.loads(manifest.read_text(encoding="utf-8"))
        data["schema"] = "other"
        manifest.write_text(json.dumps(data), encoding="utf-8")
    elif mutation == "wrong_relative_root":
        data = json.loads(manifest.read_text(encoding="utf-8"))
        data["adapter"]["relative_root"] = "../adapter"
        manifest.write_text(json.dumps(data), encoding="utf-8")
    elif mutation == "missing_readme_entry":
        data = json.loads(manifest.read_text(encoding="utf-8"))
        data["adapter"]["files"] = [
            entry
            for entry in data["adapter"]["files"]
            if entry["relative_path"] != "README.md"
        ]
        manifest.write_text(json.dumps(data), encoding="utf-8")
    elif mutation == "altered_readme":
        readme.write_bytes(readme.read_bytes() + b"changed\n")
    elif mutation == "missing_config":
        (adapter / "adapter_config.json").unlink()
    elif mutation == "missing_model":
        (adapter / "adapter_model.safetensors").unlink()

    result = scan_output_root(tmp_path)

    assert not result["passed"]
    assert {Path(item["path"]) for item in result["findings"]} >= {readme}


def test_only_adapter_card_is_exempt_and_symlink_escape_is_rejected(tmp_path):
    _write_native_adapter_payload(tmp_path)
    outside_readme = tmp_path / "README.md"
    outside_readme.write_text("outside card\n", encoding="utf-8")

    result = scan_output_root(tmp_path)

    assert [Path(item["path"]) for item in result["findings"]] == [outside_readme]

    escaped_root = tmp_path / "escaped"
    external_adapter = tmp_path / "external-adapter"
    escaped_root.mkdir()
    external_adapter.mkdir()
    (external_adapter / "README.md").write_text("external\n", encoding="utf-8")
    (escaped_root / "inference_payload_manifest.json").write_text(
        (tmp_path / "inference_payload_manifest.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (escaped_root / "adapter").symlink_to(external_adapter, target_is_directory=True)

    escaped = scan_output_root(escaped_root)

    assert not escaped["passed"]
    assert any(
        item["reason"] == "manifest_bound_adapter_symlink"
        for item in escaped["findings"]
    )


def test_nested_native_adapter_directory_symlink_is_rejected(tmp_path):
    scan_root = tmp_path / "scan"
    adapter = _write_native_adapter_payload(scan_root / "checkpoint" / "payload")
    external_adapter = tmp_path / "external-adapter"
    adapter.rename(external_adapter)
    adapter.symlink_to(external_adapter, target_is_directory=True)

    result = scan_output_root(scan_root)

    assert not result["passed"]
    assert result["findings"] == [
        {
            "path": str(adapter),
            "reason": "manifest_bound_adapter_symlink",
        }
    ]


@pytest.mark.parametrize(
    "relative_path",
    [
        "adapter/README.md",
        "adapter/adapter_config.json",
        "adapter/adapter_model.safetensors",
        "inference_payload_manifest.json",
    ],
)
def test_manifest_bound_package_members_must_not_be_symlinks(tmp_path, relative_path):
    adapter = _write_native_adapter_payload(tmp_path)
    path = tmp_path / relative_path
    target = tmp_path / "outside" / path.name
    target.parent.mkdir(exist_ok=True)
    target.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(target)

    result = scan_output_root(tmp_path)

    assert not result["passed"]
    if relative_path != "inference_payload_manifest.json":
        assert any(Path(item["path"]) == adapter / "README.md" for item in result["findings"])
