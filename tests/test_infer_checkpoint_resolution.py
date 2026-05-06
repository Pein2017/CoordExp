from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.common.model_paths import canonical_coordexp_repo_root
from src.infer.checkpoints import (
    resolve_inference_checkpoint,
    validate_compact_coord_token_adapter_contract,
)


def _write_adapter_checkpoint(
    path: Path,
    *,
    base_model_name_or_path: str | None = "base-model",
    modules_to_save: list[str] | None = None,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    payload = {
        "modules_to_save": [] if modules_to_save is None else modules_to_save,
    }
    if base_model_name_or_path is not None:
        payload["base_model_name_or_path"] = base_model_name_or_path
    (path / "adapter_config.json").write_text(
        json.dumps(payload, ensure_ascii=True),
        encoding="utf-8",
    )


def _write_coord_offset_weights(
    path: Path,
    *,
    coord_ids: list[int],
    tie_head: bool = True,
) -> None:
    import torch
    from safetensors.torch import save_file

    payload = {
        "base_model.model.coord_offset_adapter.coord_ids": torch.tensor(
            coord_ids, dtype=torch.long
        ),
        "base_model.model.coord_offset_adapter.embed_offset": torch.zeros(
            len(coord_ids), 4, dtype=torch.float32
        ),
    }
    if not tie_head:
        payload["base_model.model.coord_offset_adapter.head_offset"] = torch.zeros(
            len(coord_ids), 4, dtype=torch.float32
        )
    save_file(payload, str(path / "adapter_model.safetensors"))


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


def test_resolve_inference_checkpoint_normalizes_worktree_coordexp_base_path(
    tmp_path: Path,
) -> None:
    adapter_dir = tmp_path / "adapter"
    _write_adapter_checkpoint(
        adapter_dir,
        base_model_name_or_path=(
            "/data/CoordExp/.worktrees/compact-detection-sequence/"
            "model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp"
        ),
    )

    resolved = resolve_inference_checkpoint(model_checkpoint=str(adapter_dir))

    assert resolved.resolved_base_model_checkpoint == str(
        canonical_coordexp_repo_root()
        / "model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp"
    )


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


def test_compact_coord_token_adapter_requires_saved_coord_offset(
    tmp_path: Path,
) -> None:
    adapter_dir = tmp_path / "adapter"
    _write_adapter_checkpoint(adapter_dir, base_model_name_or_path="base-model")
    resolved = resolve_inference_checkpoint(model_checkpoint=str(adapter_dir))

    with pytest.raises(ValueError, match="compact_full.*coord_offset_adapter"):
        validate_compact_coord_token_adapter_contract(
            resolved,
            detection_sequence_format="compact_full",
        )


def test_compact_coord_token_adapter_guard_normalizes_format_alias(
    tmp_path: Path,
) -> None:
    adapter_dir = tmp_path / "adapter"
    _write_adapter_checkpoint(adapter_dir, base_model_name_or_path="base-model")
    resolved = resolve_inference_checkpoint(model_checkpoint=str(adapter_dir))

    with pytest.raises(ValueError, match="compact_full.*coord_offset_adapter"):
        validate_compact_coord_token_adapter_contract(
            resolved,
            detection_sequence_format="compact-full",
        )


def test_compact_coord_token_adapter_rejects_partial_or_untied_rows(
    tmp_path: Path,
) -> None:
    adapter_dir = tmp_path / "adapter"
    _write_adapter_checkpoint(
        adapter_dir,
        base_model_name_or_path="base-model",
        modules_to_save=["coord_offset_adapter"],
    )
    _write_coord_offset_weights(
        adapter_dir,
        coord_ids=[151646, 151648, *range(151670, 152669)],
        tie_head=False,
    )
    resolved = resolve_inference_checkpoint(model_checkpoint=str(adapter_dir))

    with pytest.raises(ValueError, match="tie_head=True|1002"):
        validate_compact_coord_token_adapter_contract(
            resolved,
            detection_sequence_format="compact_full",
        )


def test_compact_coord_token_adapter_accepts_exact_tied_rows(
    tmp_path: Path,
) -> None:
    adapter_dir = tmp_path / "adapter"
    _write_adapter_checkpoint(
        adapter_dir,
        base_model_name_or_path="base-model",
        modules_to_save=["coord_offset_adapter"],
    )
    _write_coord_offset_weights(
        adapter_dir,
        coord_ids=[151646, 151648, *range(151670, 152670)],
        tie_head=True,
    )
    resolved = resolve_inference_checkpoint(model_checkpoint=str(adapter_dir))

    validate_compact_coord_token_adapter_contract(
        resolved,
        detection_sequence_format="compact_full",
    )


@pytest.mark.parametrize(
    "coord_ids",
    [
        [151646, 151648, *range(151670, 152669)],
        [151646, 151648, *range(151670, 152670), 42],
    ],
)
def test_compact_coord_token_adapter_rejects_missing_or_extra_rows(
    tmp_path: Path,
    coord_ids: list[int],
) -> None:
    adapter_dir = tmp_path / "adapter"
    _write_adapter_checkpoint(
        adapter_dir,
        base_model_name_or_path="base-model",
        modules_to_save=["coord_offset_adapter"],
    )
    _write_coord_offset_weights(
        adapter_dir,
        coord_ids=coord_ids,
        tie_head=True,
    )
    resolved = resolve_inference_checkpoint(model_checkpoint=str(adapter_dir))

    with pytest.raises(ValueError, match="exactly 1002 trainable"):
        validate_compact_coord_token_adapter_contract(
            resolved,
            detection_sequence_format="compact_full",
        )


def test_compact_coord_token_adapter_guard_does_not_apply_to_non_compact_format(
    tmp_path: Path,
) -> None:
    adapter_dir = tmp_path / "adapter"
    _write_adapter_checkpoint(adapter_dir, base_model_name_or_path="base-model")
    resolved = resolve_inference_checkpoint(model_checkpoint=str(adapter_dir))

    validate_compact_coord_token_adapter_contract(
        resolved,
        detection_sequence_format="stage1_json_pretty",
    )


def test_compact_coord_token_adapter_allows_full_or_merged_checkpoint() -> None:
    resolved = resolve_inference_checkpoint(model_checkpoint="merged-model")

    validate_compact_coord_token_adapter_contract(
        resolved,
        detection_sequence_format="compact_full",
    )
