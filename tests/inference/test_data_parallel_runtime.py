from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from PIL import Image

from src.common.errors import RuntimeContractError


def test_visible_cuda_tokens_preserve_external_integer_and_uuid_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.inference import data_parallel

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2, 3,GPU-abc,7")

    tokens = data_parallel.resolve_visible_cuda_tokens()

    assert tokens == ("2", "3", "GPU-abc", "7")


def test_visible_cuda_tokens_can_be_synthesized_when_environment_is_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.inference import data_parallel

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    tokens = data_parallel.resolve_visible_cuda_tokens(cuda_device_count=lambda: 3)

    assert tokens == ("0", "1", "2")


def test_non_dry_cuda_requirement_fails_when_no_devices_are_visible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.inference import data_parallel

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    with pytest.raises(RuntimeContractError) as exc_info:
        data_parallel.require_visible_cuda_for_inference(
            debug_dry_run=False,
            cuda_device_count=lambda: 0,
        )

    assert exc_info.value.code == "inference.cuda_unavailable"
    assert exc_info.value.context["debug.dry_run"] is False


def test_cuda_visible_devices_minus_one_means_no_visible_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.inference import data_parallel

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")

    with pytest.raises(RuntimeContractError) as exc_info:
        data_parallel.require_visible_cuda_for_inference(debug_dry_run=False)

    assert exc_info.value.code == "inference.cuda_unavailable"


def test_non_empty_cuda_visible_devices_still_requires_runtime_cuda_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.inference import data_parallel

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    with pytest.raises(RuntimeContractError) as exc_info:
        data_parallel.require_visible_cuda_for_inference(
            debug_dry_run=False,
            cuda_device_count=lambda: 0,
        )

    assert exc_info.value.code == "inference.cuda_unavailable"
    assert exc_info.value.context["visible_cuda_tokens"] == ["0"]


def test_dry_run_cuda_requirement_returns_no_tokens_without_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.inference import data_parallel

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    tokens = data_parallel.require_visible_cuda_for_inference(
        debug_dry_run=True,
        cuda_device_count=lambda: 0,
    )

    assert tokens == ()


def test_data_parallel_plan_uses_min_devices_and_decode_batch_count() -> None:
    from src.inference import data_parallel

    plan = data_parallel.plan_data_parallel_shards(
        row_ids=("row-0", "row-1", "row-2", "row-3", "row-4"),
        per_device_batch_size=2,
        visible_cuda_tokens=("0", "1", "2", "3", "4", "5", "6", "7"),
    )

    assert plan.decode_batch_count == 3
    assert plan.active_ranks == 3
    assert [rank.rank for rank in plan.ranks] == [0, 1, 2]
    assert [rank.parent_visible_device_token for rank in plan.ranks] == ["0", "1", "2"]


def test_generation_batch_size_is_per_device_and_not_divided_by_world_size() -> None:
    from src.inference import data_parallel

    plan = data_parallel.plan_data_parallel_shards(
        row_ids=tuple(f"row-{index}" for index in range(10)),
        per_device_batch_size=4,
        visible_cuda_tokens=("0", "1"),
    )

    assert plan.per_device_batch_size == 4
    assert all(rank.per_device_batch_size == 4 for rank in plan.ranks)
    assert [batch.row_ids for batch in plan.decode_batches] == [
        ("row-0", "row-1", "row-2", "row-3"),
        ("row-4", "row-5", "row-6", "row-7"),
        ("row-8", "row-9"),
    ]


def test_decode_batches_are_assigned_round_robin_and_restore_original_order() -> None:
    from src.inference import data_parallel

    plan = data_parallel.plan_data_parallel_shards(
        row_ids=tuple(f"row-{index}" for index in range(10)),
        per_device_batch_size=2,
        visible_cuda_tokens=("0", "1"),
    )

    assert plan.ranks[0].batch_ids == (0, 2, 4)
    assert plan.ranks[0].row_ids == ("row-0", "row-1", "row-4", "row-5", "row-8", "row-9")
    assert plan.ranks[1].batch_ids == (1, 3)
    assert plan.ranks[1].row_ids == ("row-2", "row-3", "row-6", "row-7")

    shuffled_rows = [
        {"row_index": 2, "row_id": "row-2"},
        {"row_index": 0, "row_id": "row-0"},
        {"row_index": 1, "row_id": "row-1"},
    ]
    assert data_parallel.sort_rows_by_index(shuffled_rows) == [
        {"row_index": 0, "row_id": "row-0"},
        {"row_index": 1, "row_id": "row-1"},
        {"row_index": 2, "row_id": "row-2"},
    ]


def test_data_parallel_plan_artifact_records_rank_assignments_and_fingerprint() -> None:
    from src.inference import data_parallel

    plan = data_parallel.plan_data_parallel_shards(
        row_ids=tuple(f"row-{index}" for index in range(6)),
        per_device_batch_size=2,
        visible_cuda_tokens=("2", "3"),
    )
    artifact = plan.to_artifact_dict()

    assert artifact["fingerprint"] == plan.fingerprint
    assert artifact["decode_batches"] == [
        {"batch_id": 0, "row_indices": [0, 1], "row_ids": ["row-0", "row-1"]},
        {"batch_id": 1, "row_indices": [2, 3], "row_ids": ["row-2", "row-3"]},
        {"batch_id": 2, "row_indices": [4, 5], "row_ids": ["row-4", "row-5"]},
    ]
    assert artifact["ranks"] == [
        {
            "rank": 0,
            "world_size": 2,
            "parent_visible_device_token": "2",
            "per_device_batch_size": 2,
            "batch_ids": [0, 2],
            "row_indices": [0, 1, 4, 5],
            "row_ids": ["row-0", "row-1", "row-4", "row-5"],
            "shard_dir_name": "rank-000",
        },
        {
            "rank": 1,
            "world_size": 2,
            "parent_visible_device_token": "3",
            "per_device_batch_size": 2,
            "batch_ids": [1],
            "row_indices": [2, 3],
            "row_ids": ["row-2", "row-3"],
            "shard_dir_name": "rank-001",
        },
    ]


def test_empty_input_fails_before_active_rank_planning() -> None:
    from src.inference import data_parallel

    with pytest.raises(RuntimeContractError) as exc_info:
        data_parallel.plan_data_parallel_shards(
            row_ids=(),
            per_device_batch_size=2,
            visible_cuda_tokens=("0", "1"),
        )

    assert exc_info.value.code == "inference.empty_input_jsonl"


def test_pipeline_non_dry_cuda_failure_happens_before_runtime_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.inference import data_parallel
    from src.inference import pipeline

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(data_parallel, "detect_cuda_device_count", lambda: 0)
    config_path = _write_config(tmp_path, row_count=1, dry_run=False)
    runtime_loaded = False

    def fail_if_loaded(config: Any) -> Any:
        nonlocal runtime_loaded
        runtime_loaded = True
        raise AssertionError("runtime must not load when CUDA is unavailable")

    with pytest.raises(RuntimeContractError) as exc_info:
        pipeline.run(config_path=config_path, frontend_factory=fail_if_loaded)

    run_dir = tmp_path / "outputs" / "wave8-data-parallel"
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

    assert exc_info.value.code == "inference.cuda_unavailable"
    assert runtime_loaded is False
    assert summary["benchmark_eligible"] is False
    assert not (run_dir / "gt_vs_pred_scored.jsonl").exists()


def test_pipeline_non_empty_cuda_mask_with_zero_runtime_devices_fails_before_runtime_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.inference import data_parallel
    from src.inference import pipeline

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(data_parallel, "detect_cuda_device_count", lambda: 0)
    config_path = _write_config(tmp_path, row_count=1, dry_run=False)
    runtime_loaded = False

    def fail_if_loaded(config: Any) -> Any:
        nonlocal runtime_loaded
        runtime_loaded = True
        raise AssertionError("runtime must not load when CUDA runtime sees no devices")

    with pytest.raises(RuntimeContractError) as exc_info:
        pipeline.run(config_path=config_path, frontend_factory=fail_if_loaded)

    assert exc_info.value.code == "inference.cuda_unavailable"
    assert exc_info.value.context["visible_cuda_tokens"] == ["0"]
    assert runtime_loaded is False


def test_pipeline_dry_run_does_not_require_cuda_or_load_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.inference import data_parallel
    from src.inference import pipeline

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(data_parallel, "detect_cuda_device_count", lambda: 0)
    config_path = _write_config(tmp_path, row_count=1, dry_run=True)

    def fail_if_loaded(config: Any) -> Any:
        raise AssertionError("dry-run must not load runtime")

    assert pipeline.run(config_path=config_path, frontend_factory=fail_if_loaded) == 0


def test_pipeline_empty_input_fails_before_runtime_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.inference import data_parallel
    from src.inference import pipeline

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(data_parallel, "detect_cuda_device_count", lambda: 1)
    config_path = _write_config(tmp_path, row_count=0, dry_run=False)
    runtime_loaded = False

    def fail_if_loaded(config: Any) -> Any:
        nonlocal runtime_loaded
        runtime_loaded = True
        raise AssertionError("runtime must not load for empty input")

    with pytest.raises(RuntimeContractError) as exc_info:
        pipeline.run(config_path=config_path, frontend_factory=fail_if_loaded)

    assert exc_info.value.code == "inference.empty_input_jsonl"
    assert runtime_loaded is False


def _write_config(
    tmp_path: Path,
    *,
    row_count: int,
    dry_run: bool,
) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(row_count):
        image_path = data_dir / f"row-{index}.jpg"
        Image.new("RGB", (96, 64), color=(12, 34, 56)).save(image_path)
        rows.append(
            {
                "example_id": f"row-{index}",
                "image": {"path": image_path.name, "width": 96, "height": 64},
                "objects": [
                    {
                        "object_id": f"object-{index}",
                        "description": "cat",
                        "bbox": [100, 200, 300, 400],
                        "metadata": {},
                    }
                ],
                "metadata": {},
            }
        )
    input_jsonl = data_dir / "examples.jsonl"
    input_jsonl.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    config = {
        "schema_version": 1,
        "run": {
            "name": "wave8-data-parallel",
            "artifact_root": str(tmp_path / "outputs"),
            "collision_policy": "fail",
        },
        "model": {
            "base_model": str(tmp_path / "model_cache" / "qwen"),
            "dtype": "bf16",
            "processor": {"do_resize": False},
        },
        "data": {"input_jsonl": str(input_jsonl)},
        "template": {
            "object_field_order": "desc_first",
            "object_ordering": "source_order",
            "assistant_format": "object_box_closed",
            "prompt": {"user": "Describe objects."},
        },
        "backend": {
            "type": "hf",
            "hf": {
                "attn_implementation": "flash_attention_2",
                "patch_embed_linearization": "enabled",
            },
        },
        "generation": {
            "batch_size": 1,
            "max_new_tokens": 64,
            "temperature": 0.0,
            "top_p": 1.0,
            "n": 1,
        },
        "scoring": {"enabled": True},
        "artifacts": {"write_token_trace": True, "write_parse_diagnostics": True},
        "debug": {"smoke": True, "dry_run": dry_run},
    }
    config_path = tmp_path / "infer.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path
