from __future__ import annotations

from pathlib import Path

import pytest

from probes.training_set_completion import source256_readback as readback


PREPARATION = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-16-source256-fixed-prefix-completion/preparation/"
    "source256-data-v3/preparation.json"
)


def _receipt(batch: int, *, parity: bool = True, status: str = "completed") -> dict:
    return {
        "status": status,
        "configured_batch_size": batch,
        "request_count": 4,
        "exact_reference_parity": parity,
    }


def test_plan_freezes_high_prefill_probe_and_balanced_full_cohort_shards(tmp_path) -> None:
    if not PREPARATION.is_file():
        pytest.skip("Source256 v3 preparation is unavailable")
    path = tmp_path / "readback-plan.json"
    value = readback.build_plan(preparation_path=PREPARATION, output=path)
    checked = readback.validate_plan(value)

    assert value["policy"] == readback.POLICY
    assert len(value["qualification"]["image_ids"]) == 4
    assert value["qualification"]["formal_candidates"] == [4]
    assert value["bounds"]["qualification_requests"] == 8
    assert [len(shard) for shard in value["endpoint_shards"]["train"]] == [32] * 8
    assert [len(shard) for shard in value["endpoint_shards"]["dev"]] == [16] * 8
    assert set(value["qualification"]["image_ids"]) <= set(checked["cohorts"]["train"])


def test_batch_selection_has_a_hard_user_frozen_batch4_floor() -> None:
    qualified = readback.select_qualified_batch(
        {"serial": _receipt(1), "batch4": _receipt(4)}
    )
    assert qualified["status"] == "qualified"
    assert qualified["selected_batch_size"] == 4

    blocked = readback.select_qualified_batch(
        {
            "serial": _receipt(1),
            "batch4": _receipt(4, status="rejected_oom"),
        }
    )
    assert blocked == {
        "status": "blocked_batch_below_4",
        "selected_batch_size": None,
        "reason": "batch4 did not complete with exact singleton token/stop parity",
    }


def test_formal_batching_never_silently_emits_a_small_tail() -> None:
    assert [len(chunk) for chunk in readback._chunks(list(range(16)), 8, formal=True)] == [8, 8]
    assert [len(chunk) for chunk in readback._chunks(list(range(12)), 8, formal=True)] == [8, 4]
    with pytest.raises(ValueError, match="tail batch below4"):
        readback._chunks(list(range(10)), 8, formal=True)
    with pytest.raises(ValueError, match="must be >=4"):
        readback._chunks(list(range(8)), 2, formal=True)
