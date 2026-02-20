import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from src.utils.run_manifest import (
    RUN_MANIFEST_SCHEMA_VERSION,
    collect_runtime_env_metadata,
    serialize_resolved_training_config,
    write_run_manifest_files,
)


@dataclass
class _TinyCfg:
    output_dir: Path
    template: dict[str, Any]


def test_collect_runtime_env_metadata_is_whitelisted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROOT_IMAGE_DIR", "/tmp/images")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    # Should not be included unless explicitly requested.
    monkeypatch.setenv("SHOULD_NOT_APPEAR", "secret")

    env = collect_runtime_env_metadata()
    assert env["ROOT_IMAGE_DIR"] == "/tmp/images"
    assert env["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert "SHOULD_NOT_APPEAR" not in env


def test_serialize_resolved_training_config_converts_paths_to_strings() -> None:
    cfg = _TinyCfg(output_dir=Path("out"), template={"max_pixels": 786432})
    resolved = serialize_resolved_training_config(cfg)
    assert resolved["output_dir"] == "out"
    assert resolved["template"]["max_pixels"] == 786432


def test_write_run_manifest_files_writes_required_json(tmp_path: Path) -> None:
    cfg = _TinyCfg(output_dir=Path("out"), template={"max_pixels": 786432})
    written = write_run_manifest_files(
        output_dir=tmp_path,
        training_config=cfg,
        config_path="configs/unit.yaml",
        base_config_path="configs/base.yaml",
        dataset_seed=17,
    )

    resolved_path = tmp_path / written["resolved_config"]
    env_path = tmp_path / written["runtime_env"]
    assert resolved_path.is_file()
    assert env_path.is_file()

    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    assert resolved["schema_version"] == RUN_MANIFEST_SCHEMA_VERSION
    assert resolved["dataset_seed"] == 17
    assert resolved["resolved"]["template"]["max_pixels"] == 786432

    env = json.loads(env_path.read_text(encoding="utf-8"))
    assert env["schema_version"] == RUN_MANIFEST_SCHEMA_VERSION
    assert isinstance(env["env"], dict)

