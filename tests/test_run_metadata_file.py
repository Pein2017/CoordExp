from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.bootstrap.run_metadata import write_run_metadata_file


def test_write_run_metadata_file_writes_expected_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COORDEXP_STAGE2_LAUNCHER", "scripts/train_stage2.sh")

    out_path = write_run_metadata_file(
        output_dir=tmp_path,
        config_path="configs/stage2.yaml",
        base_config_path="configs/base.yaml",
        run_name="unit-run",
        dataset_seed=23,
        repo_root=tmp_path,
        manifest_files={"resolved_config": "resolved_config.json"},
        train_cache_info={"status": "ready"},
        eval_cache_info={"status": "disabled"},
    )

    payload = json.loads(out_path.read_text(encoding="utf-8"))

    assert out_path.name == "run_metadata.json"
    assert payload["config"] == "configs/stage2.yaml"
    assert payload["base_config"] == "configs/base.yaml"
    assert payload["run_name"] == "unit-run"
    assert payload["dataset_seed"] == 23
    assert payload["run_manifest_files"]["resolved_config"] == "resolved_config.json"
    assert payload["launcher"]["COORDEXP_STAGE2_LAUNCHER"] == "scripts/train_stage2.sh"
    assert payload["encoded_sample_cache"]["train"]["status"] == "ready"
    assert payload["encoded_sample_cache"]["eval"]["status"] == "disabled"
    assert isinstance(payload["upstream"], dict)
