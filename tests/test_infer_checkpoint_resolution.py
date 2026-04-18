from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.infer.checkpoints import resolve_inference_checkpoint


def _write_adapter_checkpoint(
    path: Path,
    *,
    base_model_name_or_path: str | None = "base-model",
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    payload = {
        "modules_to_save": [],
    }
    if base_model_name_or_path is not None:
        payload["base_model_name_or_path"] = base_model_name_or_path
    (path / "adapter_config.json").write_text(
        json.dumps(payload, ensure_ascii=True),
        encoding="utf-8",
    )


def test_resolve_inference_checkpoint_keeps_full_model_inputs() -> None:
    resolved = resolve_inference_checkpoint(model_checkpoint="merged-model")

    assert resolved.checkpoint_mode == "full_model"
    assert resolved.requested_model_checkpoint == "merged-model"
    assert resolved.requested_adapter_checkpoint is None
    assert resolved.resolved_base_model_checkpoint == "merged-model"
    assert resolved.resolved_adapter_checkpoint is None
    assert resolved.adapter_info is None


def test_resolve_inference_checkpoint_rejects_explicit_adapter_checkpoint(
    tmp_path: Path,
) -> None:
    adapter_dir = tmp_path / "adapter"
    _write_adapter_checkpoint(adapter_dir, base_model_name_or_path="unused-base")

    with pytest.raises(ValueError, match="infer.adapter_checkpoint is no longer supported"):
        resolve_inference_checkpoint(
            model_checkpoint="base-model",
            adapter_checkpoint=str(adapter_dir),
        )


def test_resolve_inference_checkpoint_supports_adapter_shorthand(
    tmp_path: Path,
) -> None:
    adapter_dir = tmp_path / "adapter"
    _write_adapter_checkpoint(adapter_dir, base_model_name_or_path="base-from-config")

    resolved = resolve_inference_checkpoint(model_checkpoint=str(adapter_dir))

    assert resolved.checkpoint_mode == "adapter_shorthand"
    assert resolved.requested_model_checkpoint == str(adapter_dir)
    assert resolved.requested_adapter_checkpoint is None
    assert resolved.resolved_base_model_checkpoint == "base-from-config"
    assert resolved.resolved_adapter_checkpoint == str(adapter_dir)
    assert resolved.adapter_info is not None


def test_resolve_inference_checkpoint_rejects_explicit_adapter_on_shorthand(
    tmp_path: Path,
) -> None:
    shorthand_dir = tmp_path / "shorthand"
    explicit_dir = tmp_path / "explicit"
    _write_adapter_checkpoint(shorthand_dir, base_model_name_or_path="base-a")
    _write_adapter_checkpoint(explicit_dir, base_model_name_or_path="base-b")

    with pytest.raises(ValueError, match="infer.adapter_checkpoint is no longer supported"):
        resolve_inference_checkpoint(
            model_checkpoint=str(shorthand_dir),
            adapter_checkpoint=str(explicit_dir),
        )


def test_resolve_inference_checkpoint_rejects_shorthand_without_base_model(
    tmp_path: Path,
) -> None:
    adapter_dir = tmp_path / "adapter"
    _write_adapter_checkpoint(adapter_dir, base_model_name_or_path=None)

    with pytest.raises(ValueError, match="base_model_name_or_path"):
        resolve_inference_checkpoint(model_checkpoint=str(adapter_dir))
