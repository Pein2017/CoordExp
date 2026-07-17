from __future__ import annotations

from pathlib import Path

import pytest

from src.common.errors import RuntimeContractError
from src.inference.model_assets import (
    build_model_snapshot_manifest,
    validate_model_snapshot_manifest,
)


def test_model_snapshot_manifest_hashes_every_regular_file(tmp_path: Path) -> None:
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    (tmp_path / "chat_template.jinja").write_text("template-a", encoding="utf-8")
    nested = tmp_path / "processor"
    nested.mkdir()
    (nested / "special_tokens_map.json").write_text("{}", encoding="utf-8")

    manifest = build_model_snapshot_manifest(tmp_path)

    assert manifest["file_count"] == 3
    assert [item["relative_path"] for item in manifest["files"]] == [
        "chat_template.jinja",
        "model.safetensors",
        "processor/special_tokens_map.json",
    ]
    assert validate_model_snapshot_manifest(manifest)["fingerprint"] == manifest[
        "fingerprint"
    ]


@pytest.mark.parametrize(
    "relative_path",
    ["chat_template.jinja", "special_tokens_map.json"],
)
def test_model_snapshot_manifest_detects_prompt_asset_drift(
    tmp_path: Path,
    relative_path: str,
) -> None:
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    (tmp_path / "chat_template.jinja").write_text("template-a", encoding="utf-8")
    (tmp_path / "special_tokens_map.json").write_text("{}", encoding="utf-8")
    manifest = build_model_snapshot_manifest(tmp_path)

    (tmp_path / relative_path).write_text("changed", encoding="utf-8")

    with pytest.raises(RuntimeContractError) as exc_info:
        validate_model_snapshot_manifest(manifest)
    assert exc_info.value.code == "inference.model_snapshot_identity_mismatch"
