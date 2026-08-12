from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from scripts.research.analyze_human13_k_union import analyze_outputs
from scripts.research.build_human13_k_union_manifest import (
    GlobalDenominatorIdentity,
    Human13KUnionManifest,
    ImageRecord,
    OwnerRecord,
    default_binding,
)


def _image(
    image_id: int,
    owners: list[tuple[str, str, str, tuple[float, float, float, float]]],
) -> ImageRecord:
    records = tuple(
        OwnerRecord(
            owner_id=owner_id,
            category=category,
            bbox=bbox,
            source_object_index=index,
            stratum=stratum,  # type: ignore[arg-type]
            source_row_ids=(),
            sampled_row_ids=(),
        )
        for index, (owner_id, category, stratum, bbox) in enumerate(owners)
    )
    return ImageRecord(
        image_id=image_id,
        panel_row_sha256=None,
        image_sha256=None,
        owners=records,
        trajectories=(),
        duplicate_events=(),
        selected_rows=(),
        g_owner_ids=tuple(item.owner_id for item in records if item.stratum == "G"),
        h_owner_ids=tuple(item.owner_id for item in records if item.stratum == "H"),
        m_owner_ids=tuple(item.owner_id for item in records if item.stratum == "M"),
        replay_row_ids=(),
        target_row_ids=(),
        candidate_row_ids=(),
    )


def _manifest(*images: ImageRecord) -> Human13KUnionManifest:
    return Human13KUnionManifest(
        schema_version="human13_k_union_manifest.v1",
        binding=default_binding(),
        images=tuple(images),
        arms=(),
        denominators=GlobalDenominatorIdentity(
            panel_image_count=13,
            target_image_count=0,
            target_owner_count=0,
            replay_image_count=0,
            replay_owner_count=0,
            duplicate_image_count=0,
            duplicate_event_count=0,
        ),
        full_panel=False,
    )


def _output(
    *,
    image_id: int,
    arm_id: str,
    predictions: list[dict[str, object]],
    milestone: int = 0,
    token_count: int = 8,
    stop_reason: str = "im_end",
    malformed_row_count: int = 0,
    runtime: dict[str, float | int] | None = None,
) -> dict[str, object]:
    return {
        "image_id": image_id,
        "arm_id": arm_id,
        "milestone": milestone,
        "decode_mode": "original_prompt_clean_greedy",
        "repetition_penalty": 1.0,
        "predictions": predictions,
        "generated_token_ids": list(range(token_count)),
        "stop_reason": stop_reason,
        "malformed_row_count": malformed_row_count,
        "runtime": runtime or {},
    }


def _pred(
    order: int,
    category: str,
    bbox: tuple[float, float, float, float] | None,
) -> dict[str, object]:
    return {"generated_order": order, "description": category, "bbox": bbox}


def _record(result: dict[str, object], *, image_id: int, arm_id: str) -> dict[str, object]:
    records = result["per_image"]
    assert isinstance(records, list)
    return next(
        item
        for item in records
        if item["image_id"] == image_id and item["arm_id"] == arm_id
    )


def _panel_record(
    result: dict[str, object], *, panel: str, arm_id: str, milestone: int
) -> dict[str, object]:
    records = result[panel]
    assert isinstance(records, list)
    return next(
        item
        for item in records
        if item["arm_id"] == arm_id and item["milestone"] == milestone
    )


def test_direct_cli_help_documents_the_raw_jsonl_contract() -> None:
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "research"
        / "analyze_human13_k_union.py"
    )

    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "original_prompt_clean_greedy" in completed.stdout
    assert "generated_token_ids" in completed.stdout


def test_chronological_class_agnostic_duplicates_are_excluded_before_matching() -> None:
    # Both predictions can form a cardinality-two assignment to these dense
    # owners. Their pred-pred IoU is > 0.95, so the later row must first be
    # removed and can never receive owner credit. The retained box deliberately
    # differs from every native selected row: final metric credit is GT-based.
    manifest = _manifest(
        _image(
            2299,
            [
                ("h-a", "person", "H", (0.0, 0.0, 10.0, 10.0)),
                ("h-b", "bus", "H", (0.2, 0.0, 10.2, 10.0)),
            ],
        )
    )
    outputs = [
        _output(image_id=2299, arm_id="frozen_source", predictions=[]),
        _output(
            image_id=2299,
            arm_id="A3",
            milestone=1,
            predictions=[
                _pred(0, " PERSON ", (0.05, 0.0, 10.05, 10.0)),
                _pred(1, "bus", (0.25, 0.0, 10.25, 10.0)),
            ],
        ),
    ]

    result = analyze_outputs(manifest, outputs, fixed_row_budgets=(1, 2))
    record = _record(result, image_id=2299, arm_id="A3")

    assert record["k_hit_gained_owner_ids"] == ["h-a"]
    assert record["final_unique_owner_ids"] == ["h-a"]
    assert record["burden"]["duplicate_rows"] == 1
    assert record["duplicate_rows"][0]["generated_order"] == 1
    assert record["fixed_budget_coverage"]["2"]["owner_ids"] == ["h-a"]


def test_matching_is_cardinality_first_then_maximum_total_iou() -> None:
    # Highest-IoU greedy would take owner-a/pred-0 and strand owner-b. The
    # declared global assignment instead returns both owners.
    manifest = _manifest(
        _image(
            1584,
            [
                ("owner-a", "person", "G", (22, 31, 83, 72)),
                ("owner-b", "person", "G", (29, 10, 79, 71)),
            ],
        )
    )
    predictions = [
        _pred(0, "person", (31, 27, 81, 88)),
        _pred(1, "person", (27, 37, 93, 87)),
    ]

    result = analyze_outputs(
        manifest,
        [
            _output(
                image_id=1584, arm_id="frozen_source", predictions=predictions
            ),
            _output(image_id=1584, arm_id="A1", predictions=predictions, milestone=1),
        ],
    )
    record = _record(result, image_id=1584, arm_id="A1")

    assert record["source_retained_owner_ids"] == ["owner-a", "owner-b"]
    assert record["burden"]["unmatched_rows"] == 0


def test_owner_exchange_burdens_and_panel_slices_remain_separate() -> None:
    manifest = _manifest(
        _image(
            1584,
            [
                ("g-1584", "bus", "G", (0, 0, 10, 10)),
                ("h-1584", "dog", "H", (20, 0, 30, 10)),
            ],
        ),
        _image(
            2299,
            [
                ("g-2299", "person", "G", (0, 20, 10, 30)),
                ("h-2299", "cat", "H", (20, 20, 30, 30)),
                ("m-2299", "chair", "M", (40, 20, 50, 30)),
            ],
        ),
    )
    outputs = [
        _output(
            image_id=1584,
            arm_id="frozen_source",
            predictions=[_pred(0, "bus", (0, 0, 10, 10))],
        ),
        _output(
            image_id=2299,
            arm_id="frozen_source",
            predictions=[_pred(0, "person", (0, 20, 10, 30))],
        ),
        _output(
            image_id=1584,
            arm_id="A3",
            milestone=1,
            predictions=[
                _pred(0, "bus", (0.2, 0, 10.2, 10)),
                _pred(1, "dog", (20, 0, 30, 10)),
            ],
            token_count=11,
            runtime={"wall_time_seconds": 2.0, "gpu_seconds": 1.5},
        ),
        _output(
            image_id=2299,
            arm_id="A3",
            milestone=1,
            predictions=[
                _pred(0, "cat", (20.1, 20, 30.1, 30)),
                _pred(1, "bus", (20.2, 20, 30.2, 30)),
                _pred(2, "chair", (40, 20, 50, 30)),
                _pred(3, "bicycle", (60, 20, 70, 30)),
                _pred(4, "cat", (5, 5, 5, 8)),
            ],
            token_count=13,
            stop_reason="max_new_tokens",
            malformed_row_count=2,
            runtime={"wall_time_seconds": 3.0, "gpu_seconds": 2.5},
        ),
    ]

    result = analyze_outputs(manifest, outputs, fixed_row_budgets=(1, 2, 4))
    exchange = _record(result, image_id=2299, arm_id="A3")

    assert exchange["k_hit_gained_owner_ids"] == ["h-2299"]
    assert exchange["source_lost_owner_ids"] == ["g-2299"]
    assert "net_owner_gain" not in exchange
    assert exchange["k_miss_incidental_gain_owner_ids"] == ["m-2299"]
    assert exchange["final_unique_owner_ids"] == ["h-2299", "m-2299"]
    assert exchange["burden"] == {
        "duplicate_rows": 1,
        "unmatched_rows": 1,
        "malformed_rows": 2,
        "invalid_rows": 1,
        "cap_stops": 1,
    }
    assert exchange["prediction_row_count"] == 7
    assert exchange["generated_token_count"] == 13
    assert exchange["fixed_budget_coverage"]["1"]["owner_ids"] == ["h-2299"]
    assert exchange["natural_stop_coverage"]["owner_ids"] == [
        "h-2299",
        "m-2299",
    ]
    assert exchange["natural_stop_reached"] is False

    assert set(result) >= {
        "per_image",
        "legacy12",
        "image2299",
        "pooled",
    }
    pooled = _panel_record(result, panel="pooled", arm_id="A3", milestone=1)
    assert pooled["k_hit_gained_count"] == 2
    assert pooled["source_retained_count"] == 1
    assert pooled["source_lost_count"] == 1
    assert pooled["k_miss_incidental_gain_count"] == 1
    assert pooled["final_unique_owner_ids"] == [
        "1584:g-1584",
        "1584:h-1584",
        "2299:h-2299",
        "2299:m-2299",
    ]
    assert pooled["burden"]["duplicate_rows"] == 1
    assert pooled["burden"]["unmatched_rows"] == 1
    assert pooled["burden"]["malformed_rows"] == 2
    assert pooled["burden"]["invalid_rows"] == 1
    assert pooled["burden"]["cap_stops"] == 1
    assert pooled["generated_token_count"] == 24
    assert pooled["fixed_budget_coverage"]["1"]["k_hit_owner_count"] == 1
    assert pooled["natural_stop_coverage"]["owner_count"] == 4
    assert pooled["runtime"]["wall_time_seconds"] == 5.0
    assert pooled["safe_in_panel_consolidation_image_count"] == 1
    assert pooled["full_h_mastery_image_count"] == 1

    legacy = _panel_record(result, panel="legacy12", arm_id="A3", milestone=1)
    special = _panel_record(result, panel="image2299", arm_id="A3", milestone=1)
    assert legacy["image_ids"] == [1584]
    assert special["image_ids"] == [2299]
    assert legacy["common_owner_iou"]["owner_count"] == 1
    assert special["common_owner_iou"]["owner_count"] == 0
