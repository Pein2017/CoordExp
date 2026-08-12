from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.research.analyze_human13_k_union import (
    _analyze_outputs_unsafe,
    _manifest_sha256,
    analyze_outputs,
)
from scripts.research.build_human13_k_union_manifest import (
    ArmIdentity,
    GlobalDenominatorIdentity,
    Human13KUnionManifest,
    ImageRecord,
    OwnerRecord,
    PredictionRowInput,
    PrefixRecord,
    RequestIdentity,
    SelectedRowRecord,
    TrajectoryRecord,
    default_binding,
)


def _image(
    image_id: int,
    owners: list[tuple[str, str, str, tuple[float, float, float, float]]],
    *,
    selected_native_box: tuple[float, float, float, float] | None = None,
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
    source_tokens = (101, 102, 999)
    source = TrajectoryRecord(
        trajectory_id=f"source:{image_id}",
        request=RequestIdentity(
            backend="hf",
            backend_version="test-hf",
            mode="source_greedy",
            n=1,
            seed=None,
            physical_batch_index=0,
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.0,
            max_new_tokens=3084,
        ),
        raw_token_ids=source_tokens,
        terminal_token_index=2,
        stop_reason="im_end",
        parser_status="complete",
        rows=(),
        prefix=PrefixRecord(
            raw_token_ids=source_tokens[:2],
            clean_token_ids=source_tokens[:2],
            removed_row_ids=(),
        ),
        retained_row_ids=(),
        duplicate_row_ids=(),
        matched_row_ids=(),
        replay_token_mask=(False, False, False),
        duplicate_target_mask=(False, False, False),
    )
    selected_rows = ()
    sampled: tuple[TrajectoryRecord, ...] = ()
    if selected_native_box is not None:
        native_row = PredictionRowInput(
            row_id="native-row",
            row_index=0,
            category=records[0].category,
            bbox=selected_native_box,
            token_start=0,
            token_end=3,
            final_coordinate_token_index=2,
        )
        sampled = (
            TrajectoryRecord(
                trajectory_id="k-21001",
                request=RequestIdentity(
                    backend="vllm",
                    backend_version="test-vllm",
                    mode="k_sampled",
                    n=1,
                    seed=21001,
                    physical_batch_index=0,
                    temperature=0.4,
                    top_p=0.95,
                    repetition_penalty=1.1,
                    max_new_tokens=512,
                ),
                raw_token_ids=(1, 2, 3, 999),
                terminal_token_index=3,
                stop_reason="im_end",
                parser_status="complete",
                rows=(native_row,),
                prefix=PrefixRecord(
                    raw_token_ids=(1, 2, 3),
                    clean_token_ids=(1, 2, 3),
                    removed_row_ids=(),
                ),
                retained_row_ids=("native-row",),
                duplicate_row_ids=(),
                matched_row_ids=("native-row",),
                replay_token_mask=(False, False, False, False),
                duplicate_target_mask=(False, False, False, False),
            ),
        )
        selected_rows = (
            SelectedRowRecord(
                owner_id=records[0].owner_id,
                row_id="native-row",
                trajectory_id="k-21001",
                seed=21001,
                row_index=0,
                owner_iou=0.75,
                token_ids=(1, 2, 3),
                target_token_mask=(True, True, True),
            ),
        )
    return ImageRecord(
        image_id=image_id,
        panel_row_sha256=None,
        image_sha256=None,
        owners=records,
        trajectories=(source,) + sampled,
        duplicate_events=(),
        selected_rows=selected_rows,
        g_owner_ids=tuple(item.owner_id for item in records if item.stratum == "G"),
        h_owner_ids=tuple(item.owner_id for item in records if item.stratum == "H"),
        m_owner_ids=tuple(item.owner_id for item in records if item.stratum == "M"),
        replay_row_ids=(),
        target_row_ids=tuple(row.row_id for row in selected_rows),
        candidate_row_ids=(),
    )


def _manifest(*images: ImageRecord) -> Human13KUnionManifest:
    return Human13KUnionManifest(
        schema_version="human13_k_union_manifest.v1",
        binding=default_binding(),
        images=tuple(images),
        arms=tuple(
            ArmIdentity(arm_id, target_scope, terminal_masked=True)
            for arm_id, target_scope in (
                ("frozen_source", "none"),
                ("full_gt_capacity", "GT"),
                ("A0", "none"),
                ("A1", "H"),
                ("A3", "H"),
                ("A4", "H"),
                ("A6", "H"),
                ("A7", "H"),
                ("A8-prime", "H"),
            )
        ),
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
    manifest: Human13KUnionManifest,
    image_id: int,
    arm_id: str,
    predictions: list[dict[str, object]],
    milestone: int = 0,
    token_count: int = 8,
    stop_reason: str = "im_end",
    malformed_row_count: int = 0,
    runtime: dict[str, float | int] | None = None,
) -> dict[str, object]:
    binding = manifest.binding
    source = next(image for image in manifest.images if image.image_id == image_id)
    source_trajectory = source.trajectories[0]
    source_checkpoint_identity = {
        "checkpoint_path": binding.source.checkpoint_path,
        "base_model_path": binding.source.base_model_path,
        "adapter_sha256": binding.source.adapter_sha256,
        "special_embedding_sha256": binding.source.special_embedding_sha256,
    }
    run_id = "frozen-source" if arm_id == "frozen_source" else f"run-{arm_id}"
    run_root = f"{binding.artifact_root}/{run_id}"
    checkpoint_path = (
        binding.source.checkpoint_path
        if arm_id == "frozen_source"
        else f"{run_root}/checkpoints/step-{milestone}"
    )
    generated_token_ids = (
        list(source_trajectory.raw_token_ids)
        if arm_id == "frozen_source"
        else list(range(token_count))
    )
    return {
        "image_id": image_id,
        "arm_id": arm_id,
        "milestone": milestone,
        "decode_mode": "original_prompt_clean_greedy",
        "repetition_penalty": 1.0,
        "predictions": predictions,
        "generated_token_ids": generated_token_ids,
        "stop_reason": stop_reason,
        "malformed_row_count": malformed_row_count,
        "runtime": runtime or {},
        "provenance": {
            "unit_id": binding.unit_id,
            "purpose": binding.purpose,
            "artifact_root": binding.artifact_root,
            "manifest_sha256": _manifest_sha256(manifest),
            "panel_sha256": binding.panel.panel_sha256,
            "image_id": image_id,
            "panel_row_sha256": source.panel_row_sha256,
            "image_sha256": source.image_sha256,
            "arm_id": arm_id,
            "milestone": milestone,
            "backend": "hf",
            "backend_version": source_trajectory.request.backend_version,
            "physical_batch_size": 1,
            "do_sample": False,
            "max_new_tokens": source_trajectory.request.max_new_tokens,
            "source_checkpoint_identity": source_checkpoint_identity,
            "checkpoint_path": checkpoint_path,
            "checkpoint_payload_sha256": hashlib.sha256(
                checkpoint_path.encode("utf-8")
            ).hexdigest(),
            "run_id": run_id,
            "run_root": run_root,
            "resolved_arm_plan_sha256": hashlib.sha256(
                f"plan:{arm_id}".encode("utf-8")
            ).hexdigest(),
            "resolved_config_sha256": hashlib.sha256(
                f"config:{arm_id}".encode("utf-8")
            ).hexdigest(),
            "prompt_policy_fingerprint": binding.surface.prompt_policy_fingerprint,
            "tokenizer_sha256": binding.surface.tokenizer_sha256,
            "tokenizer_class": binding.surface.tokenizer_class,
            "wrapper": binding.surface.wrapper,
            "parser": binding.surface.parser,
            "source_trajectory_id": source_trajectory.trajectory_id,
            "trajectory_id": (
                source_trajectory.trajectory_id
                if arm_id == "frozen_source"
                else f"{arm_id}:{milestone}:{image_id}"
            ),
        },
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
    assert "manifest_sha256" in completed.stdout
    assert "physical_batch_size=1" in completed.stdout


def test_public_api_rejects_partial_manifest_before_projecting_outputs() -> None:
    manifest = _manifest(_image(2299, [("h-a", "person", "H", (0, 0, 10, 10))]))
    outputs = [
        _output(
            manifest=manifest,
            image_id=2299,
            arm_id="frozen_source",
            predictions=[],
        )
    ]

    with pytest.raises(ValueError, match="full Human-13 manifest|mechanics-only"):
        analyze_outputs(manifest, outputs)


@pytest.mark.parametrize(
    ("field", "wrong"),
    [
        ("manifest_sha256", "0" * 64),
        ("panel_sha256", "0" * 64),
        ("panel_row_sha256", "1" * 64),
        ("image_sha256", "2" * 64),
        ("backend", "vllm"),
        ("physical_batch_size", 2),
        ("source_checkpoint_identity", {"checkpoint_path": "/wrong"}),
        ("prompt_policy_fingerprint", "2" * 64),
        ("tokenizer_sha256", "3" * 64),
        ("wrapper", "wrong_wrapper"),
        ("parser", "wrong_parser"),
    ],
)
def test_raw_outputs_reject_wrong_run_or_manifest_provenance(
    field: str, wrong: object
) -> None:
    manifest = _manifest(_image(2299, [("h-a", "person", "H", (0, 0, 10, 10))]))
    output = _output(
        manifest=manifest,
        image_id=2299,
        arm_id="frozen_source",
        predictions=[],
    )
    output["provenance"][field] = wrong

    with pytest.raises(ValueError, match=field):
        _analyze_outputs_unsafe(manifest, [output])


def test_frozen_source_binds_exact_manifest_trajectory_and_token_identity() -> None:
    manifest = _manifest(_image(2299, [("h-a", "person", "H", (0, 0, 10, 10))]))
    output = _output(
        manifest=manifest,
        image_id=2299,
        arm_id="frozen_source",
        predictions=[],
    )

    wrong_trajectory = {**output, "provenance": dict(output["provenance"])}
    wrong_trajectory["provenance"]["trajectory_id"] = "not-the-frozen-source"
    with pytest.raises(ValueError, match="trajectory_id"):
        _analyze_outputs_unsafe(manifest, [wrong_trajectory])

    wrong_tokens = {**output, "generated_token_ids": [999]}
    with pytest.raises(ValueError, match="generated_token_ids"):
        _analyze_outputs_unsafe(manifest, [wrong_tokens])

    wrong_checkpoint = {**output, "provenance": dict(output["provenance"])}
    wrong_checkpoint["provenance"]["checkpoint_path"] = "/trained/not-source"
    with pytest.raises(ValueError, match="frozen_source.*checkpoint_path"):
        _analyze_outputs_unsafe(manifest, [wrong_checkpoint])


def test_trained_arm_requires_distinct_complete_run_and_checkpoint_evidence() -> None:
    manifest = _manifest(_image(2299, [("h-a", "person", "H", (0, 0, 10, 10))]))
    source = _output(
        manifest=manifest,
        image_id=2299,
        arm_id="frozen_source",
        predictions=[],
    )
    arm = _output(
        manifest=manifest,
        image_id=2299,
        arm_id="A3",
        milestone=1,
        predictions=[],
    )

    wrong = {**arm, "provenance": dict(arm["provenance"])}
    wrong["provenance"]["checkpoint_path"] = manifest.binding.source.checkpoint_path
    with pytest.raises(ValueError, match="trained arm.*checkpoint_path"):
        _analyze_outputs_unsafe(manifest, [source, wrong])

    for field in (
        "checkpoint_payload_sha256",
        "run_id",
        "run_root",
        "resolved_arm_plan_sha256",
        "resolved_config_sha256",
    ):
        missing = {**arm, "provenance": dict(arm["provenance"])}
        del missing["provenance"][field]
        with pytest.raises(ValueError, match=field):
            _analyze_outputs_unsafe(manifest, [source, missing])


def test_arm_milestone_run_evidence_is_group_consistent_and_roots_are_unique() -> None:
    manifest = _manifest(
        _image(1584, [("h-a", "person", "H", (0, 0, 10, 10))]),
        _image(2299, [("h-b", "person", "H", (20, 0, 30, 10))]),
    )
    outputs = [
        *[
            _output(
                manifest=manifest,
                image_id=image_id,
                arm_id="frozen_source",
                predictions=[],
            )
            for image_id in (1584, 2299)
        ],
        *[
            _output(
                manifest=manifest,
                image_id=image_id,
                arm_id="A3",
                milestone=1,
                predictions=[],
            )
            for image_id in (1584, 2299)
        ],
    ]
    inconsistent = [
        {**item, "provenance": dict(item["provenance"])} for item in outputs
    ]
    inconsistent[-1]["provenance"]["checkpoint_payload_sha256"] = "4" * 64
    with pytest.raises(ValueError, match="group-consistent"):
        _analyze_outputs_unsafe(manifest, inconsistent)

    duplicate_root_arm = _output(
        manifest=manifest,
        image_id=1584,
        arm_id="A1",
        milestone=1,
        predictions=[],
    )
    duplicate_root_arm["provenance"]["run_root"] = outputs[-1]["provenance"][
        "run_root"
    ]
    duplicate_root_arm_peer = _output(
        manifest=manifest,
        image_id=2299,
        arm_id="A1",
        milestone=1,
        predictions=[],
    )
    duplicate_root_arm_peer["provenance"]["run_root"] = duplicate_root_arm[
        "provenance"
    ]["run_root"]
    with pytest.raises(ValueError, match="unique run_root"):
        _analyze_outputs_unsafe(
            manifest, outputs + [duplicate_root_arm, duplicate_root_arm_peer]
        )


def test_one_arm_run_reuses_root_across_milestones_with_free_trajectory_ids() -> None:
    manifest = _manifest(
        _image(1584, [("h-a", "person", "H", (0, 0, 10, 10))]),
        _image(2299, [("h-b", "person", "H", (20, 0, 30, 10))]),
    )
    outputs = [
        *[
            _output(
                manifest=manifest,
                image_id=image_id,
                arm_id="frozen_source",
                predictions=[],
            )
            for image_id in (1584, 2299)
        ],
        *[
            _output(
                manifest=manifest,
                image_id=image_id,
                arm_id="A3",
                milestone=milestone,
                predictions=[],
            )
            for milestone in (1, 2)
            for image_id in (1584, 2299)
        ],
    ]
    for index, output in enumerate(outputs[2:], start=1):
        output["provenance"]["trajectory_id"] = f"hf-eval-output-{index}"

    result = _analyze_outputs_unsafe(manifest, outputs)

    assert {
        (record["arm_id"], record["milestone"])
        for record in result["pooled"]
    } >= {("A3", 1), ("A3", 2)}


def test_unknown_stop_reason_is_rejected_before_safety_projection() -> None:
    manifest = _manifest(_image(2299, [("h-a", "person", "H", (0, 0, 10, 10))]))
    output = _output(
        manifest=manifest,
        image_id=2299,
        arm_id="frozen_source",
        predictions=[],
        stop_reason="im_ennd",
    )

    with pytest.raises(ValueError, match="supported enum"):
        _analyze_outputs_unsafe(manifest, [output])


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
            selected_native_box=(0.0, 0.0, 8.0, 8.0),
        )
    )
    outputs = [
        _output(
            manifest=manifest,
            image_id=2299,
            arm_id="frozen_source",
            predictions=[],
        ),
        _output(
            manifest=manifest,
            image_id=2299,
            arm_id="A3",
            milestone=1,
            predictions=[
                _pred(0, " PERSON ", (0.05, 0.0, 10.05, 10.0)),
                _pred(1, "bus", (0.25, 0.0, 10.25, 10.0)),
            ],
        ),
    ]

    result = _analyze_outputs_unsafe(manifest, outputs, fixed_row_budgets=(1, 2))
    record = _record(result, image_id=2299, arm_id="A3")

    assert manifest.images[0].selected_rows
    native_box = manifest.images[0].trajectories[1].rows[0].bbox
    assert native_box == (0.0, 0.0, 8.0, 8.0)
    assert tuple(outputs[1]["predictions"][0]["bbox"]) != native_box
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

    result = _analyze_outputs_unsafe(
        manifest,
        [
            _output(
                manifest=manifest,
                image_id=1584, arm_id="frozen_source", predictions=predictions
            ),
            _output(
                manifest=manifest,
                image_id=1584,
                arm_id="A1",
                predictions=predictions,
                milestone=1,
            ),
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
            manifest=manifest,
            image_id=1584,
            arm_id="frozen_source",
            predictions=[_pred(0, "bus", (0, 0, 10, 10))],
        ),
        _output(
            manifest=manifest,
            image_id=2299,
            arm_id="frozen_source",
            predictions=[_pred(0, "person", (0, 20, 10, 30))],
        ),
        _output(
            manifest=manifest,
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
            manifest=manifest,
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

    result = _analyze_outputs_unsafe(
        manifest, outputs, fixed_row_budgets=(1, 2, 4)
    )
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
