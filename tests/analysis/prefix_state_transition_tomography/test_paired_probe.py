from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from src.analysis.prefix_state_transition_tomography.config import load_config
from src.analysis.prefix_state_transition_tomography.paired_probe import (
    _assert_prefix_token_alignment,
    _assert_rendered_continuation_context,
    _probe_prefix_state,
    _resolve_image_path,
    rows_for_shard,
    run_paired_checkpoint_probe,
    validate_sampled_image_paths,
)


def test_rows_for_shard_prefers_planned_shard_fields() -> None:
    rows = [
        {"prefix_state_id": "a", "shard_id": 1},
        {"prefix_state_id": "b", "planned_shard_id": 1},
        {"prefix_state_id": "c"},
        {"prefix_state_id": "d"},
    ]

    selected = rows_for_shard(rows, shard_id=1, num_shards=2)

    assert [row["prefix_state_id"] for row in selected] == ["a", "b", "d"]


def test_probe_prefix_state_preserves_emitted_gt_order_and_outputs_probe_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def fake_score_boundary(**kwargs: object) -> dict[str, object]:
        seen["boundary_assistant_text"] = kwargs["boundary_assistant_text"]
        return {
            "ranked_desc_scores": [
                {"desc": "person", "role": "residual", "score": 1.0, "rank": 1},
                {"desc": "dog", "role": "emitted", "score": 0.1, "rank": 2},
            ],
            "summary": {
                "boundary_alignment": "residual_favored",
                "eos_score": -2.0,
                "best_desc": "person",
                "best_role": "residual",
                "margin_best_residual_vs_eos": 3.0,
            },
        }

    def fake_score_forced_x1(**kwargs: object) -> dict[str, object]:
        return {
            "checkpoint_role": kwargs["checkpoint_role"],
            "prefix_state_id": kwargs["state"]["prefix_state_id"],  # type: ignore[index]
            "probe_desc": kwargs["desc"],
            "readout_type": "forced_desc_pre_x1",
            "forced_x1_residual_coverage": 1.0,
        }

    monkeypatch.setattr(
        "src.analysis.prefix_state_transition_tomography.paired_probe._score_boundary",
        fake_score_boundary,
    )
    monkeypatch.setattr(
        "src.analysis.prefix_state_transition_tomography.paired_probe._score_forced_x1",
        fake_score_forced_x1,
    )
    monkeypatch.setattr(
        "src.analysis.prefix_state_transition_tomography.paired_probe._resolve_image_path",
        lambda *args, **kwargs: Path("/tmp/img.jpg"),
    )

    boundary_rows, forced_rows = _probe_prefix_state(
        state=_state(),
        checkpoint_role="et_rmp_ce",
        model_handle=SimpleNamespace(),
        peak=SimpleNamespace(),
    )

    assert "<|object_ref_start|>dog<|box_start|>" in str(seen["boundary_assistant_text"])
    assert str(seen["boundary_assistant_text"]).index("dog") < str(seen["boundary_assistant_text"]).index("person")
    assert {row["readout_type"] for row in boundary_rows} == {"boundary_full_desc_span"}
    assert [row["probe_desc"] for row in boundary_rows] == ["person", "dog"]
    assert [row["probe_desc"] for row in forced_rows] == ["person", "dog"]


def test_run_paired_probe_blocks_when_index_gate_missing(tmp_path: Path) -> None:
    config = _write_config(tmp_path)

    result = run_paired_checkpoint_probe(config=config, shard_id=0)

    assert result["status"] == "blocked"
    assert result["reason"] == "missing_prefix_state_index_summary"


def test_run_paired_probe_blocks_when_index_gate_failed(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    config.artifact_root.mkdir(parents=True)
    (config.artifact_root / "prefix_state_index_summary.json").write_text(
        json.dumps({"launch_eligible": False, "failed_launch_gates": ["too_few_rows"]}),
        encoding="utf-8",
    )

    result = run_paired_checkpoint_probe(config=config, shard_id=0)

    assert result["status"] == "blocked"
    assert result["reason"] == "prefix_state_index_not_launch_eligible"
    assert result["failed_launch_gates"] == ["too_few_rows"]


def test_prefix_token_alignment_rejects_tokenizer_boundary_drift() -> None:
    import torch

    prefix_ids = torch.tensor([[1, 2, 3]])
    full_ids = torch.tensor([[1, 2, 4, 5]])

    with pytest.raises(RuntimeError, match="diverged"):
        _assert_prefix_token_alignment(prefix_ids, full_ids, suffix_start=3)


def test_rendered_context_guard_requires_pre_x1_tail() -> None:
    _assert_rendered_continuation_context(
        full_text="prefix <|object_ref_start|>person<|box_start|>",
        assistant_text="<|object_ref_start|>person<|box_start|>",
        expected_context_suffix="<|box_start|>",
    )
    with pytest.raises(RuntimeError, match="assistant continuation tail"):
        _assert_rendered_continuation_context(
            full_text="prefix <|object_ref_start|>person<|box_start|><|im_end|>",
            assistant_text="<|object_ref_start|>person<|box_start|>",
            expected_context_suffix="<|box_start|>",
        )


def test_resolve_image_path_uses_configured_root(tmp_path: Path) -> None:
    image_root = tmp_path / "images-root"
    image = image_root / "images" / "train2017" / "000000000009.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"fake")
    state = {
        "image_path": "images/train2017/000000000009.jpg",
        "source_dataset_jsonl": str(tmp_path / "train.coord.jsonl"),
    }

    assert _resolve_image_path(state, image_root=image_root, must_exist=True) == image


def test_validate_sampled_image_paths_reports_missing_images(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    config.artifact_root.mkdir(parents=True)
    from src.analysis.prefix_state_transition_tomography.jsonl import write_jsonl

    write_jsonl(
        config.artifact_root / "prefix_state_sampled_rows.jsonl",
        [
            {
                "prefix_state_id": "pst",
                "image_path": "missing.jpg",
                "source_dataset_jsonl": str(tmp_path / "train.coord.jsonl"),
            }
        ],
    )

    result = validate_sampled_image_paths(config=config)

    assert result["status"] == "missing_images"
    assert result["missing_rows"] == 1
    assert result["examples"][0]["prefix_state_id"] == "pst"


def _state() -> dict[str, object]:
    return {
        "schema_version": "a3.1.v1",
        "project_id": "prefix_state_transition_tomography",
        "phase_id": "phase_a3_1",
        "run_id": "run",
        "checkpoint_id": "paired",
        "split": "train",
        "source_dataset_jsonl": "/tmp/train.jsonl",
        "source_line_idx": 0,
        "image_id": "img-1",
        "image_path": "/tmp/img.jpg",
        "prefix_state_id": "pst-1",
        "transition_type": "same_desc_transition",
        "prefix_condition": "same_desc_prefix_k",
        "prefix_depth": "shallow_1",
        "prefix_order_policy_id": "same_desc_x1_order",
        "gt_objects": [
            {"gt_idx": 0, "desc": "person", "bbox_xyxy": [100, 10, 180, 200]},
            {"gt_idx": 1, "desc": "dog", "bbox_xyxy": [300, 20, 380, 220]},
            {"gt_idx": 2, "desc": "person", "bbox_xyxy": [500, 30, 580, 230]},
        ],
        "emitted_gt_indices": [1, 0],
        "residual_gt_indices": [2],
        "emitted_descs": ["dog", "person"],
        "residual_descs": ["person"],
        "all_descs": ["person", "dog"],
        "target_residual_desc": "person",
        "hard_competitor_desc": "dog",
        "shard_id": 0,
    }


def _write_config(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "project_id": "prefix_state_transition_tomography",
                "artifact_root": str(tmp_path / "artifacts"),
                "train_jsonl": str(tmp_path / "train.jsonl"),
                "val_jsonl": str(tmp_path / "val.jsonl"),
                "stages": ["prefix_state_index", "validate"],
                "checkpoints": {
                    "et_rmp_ce": {"checkpoint_path": "/ckpts/et/checkpoint-3664"},
                    "pure_ce": {"checkpoint_path": "/ckpts/pure/checkpoint-3664"},
                },
            }
        ),
        encoding="utf-8",
    )
    return load_config(config_path)
