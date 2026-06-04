from __future__ import annotations

import json
import subprocess
import sys

import pytest

from src.analysis.sorted_random_no_newline_phenotype import fn_probe
from src.analysis.sorted_random_no_newline_phenotype.fn_probe import (
    build_fn_candidate_score_rows,
    build_fn_probe_rows,
    build_fn_slot_evidence_rows,
    r95,
)


def test_r95_rule_and_broad_radius_is_only_diagnostic() -> None:
    assert r95(10) == 0
    assert r95(50) == 2
    assert r95(100) == 4
    assert r95(200) == 8
    assert r95(400) == 8

    rows = build_fn_slot_evidence_rows(
        [
            {
                "probe_id": "probe-broad-only",
                "fn_case_id": "case-broad-only",
                "slot": "x1",
                "axis_len": 10,
                "gt_idx": 7,
                "gt_value": 5,
                "peak_value": 6,
                "score": 0.98,
            }
        ]
    )

    assert rows[0]["strict_r95_radius"] == 0
    assert rows[0]["x1_broad_near_24"] is True
    assert rows[0]["x1_strict_r95_hit"] is False
    json.dumps(rows, allow_nan=False, sort_keys=True)


def test_probe_rows_persist_hint_levels_and_prefix_conditions() -> None:
    case = _case(
        fn_case_id="case-contract",
        fn_gt_idx=1,
        fn_desc="chair",
        fn_bbox=[20, 3, 30, 12],
    )
    prefix_objects = [
        _prefix_object(
            source="gt",
            gt_idx=2,
            pred_idx=None,
            desc="future lamp",
            bbox=[40, 4, 50, 14],
            order_idx=0,
        )
    ]
    specs = []
    for index, hint_level in enumerate(fn_probe.HINT_LEVELS):
        specs.append(
            {
                "probe_id": f"probe-hint-{hint_level}",
                "fn_case_id": "case-contract",
                "hint_level": hint_level,
                "prefix_condition": fn_probe.PREFIX_CONDITIONS[index],
                "prefix_objects": prefix_objects,
                "generated_continuation": '{"desc":"chair","bbox":[20,3,30,12]}',
                "valid_parse": True,
            }
        )
    for condition in fn_probe.PREFIX_CONDITIONS[len(fn_probe.HINT_LEVELS) :]:
        specs.append(
            {
                "probe_id": f"probe-prefix-{condition}",
                "fn_case_id": "case-contract",
                "hint_level": "desc_x1_y1",
                "prefix_condition": condition,
                "prefix_objects": prefix_objects,
                "generated_continuation": '{"desc":"chair","bbox":[20,3,30,12]}',
                "valid_parse": True,
            }
        )

    rows = build_fn_probe_rows([case], probe_specs=specs)
    by_id = {row["probe_id"]: row for row in rows}

    assert set(fn_probe.HINT_LEVELS) == {"none", "desc", "desc_x1", "desc_x1_y1"}
    assert set(fn_probe.PREFIX_CONDITIONS) == {
        "empty_prefix",
        "rollout_prefix",
        "teacher_sorted_prefix",
        "teacher_oracle_remaining_prefix",
        "same_desc_removed_prefix",
        "same_desc_shuffled_prefix",
    }
    assert by_id["probe-hint-none"]["hint_coords"] == {}
    assert by_id["probe-hint-desc"]["hint_coords"] == {}
    assert by_id["probe-hint-desc_x1"]["hint_coords"] == {"x1": 20}
    assert by_id["probe-hint-desc_x1_y1"]["hint_coords"] == {"x1": 20, "y1": 3}

    for row in rows:
        assert row["hint_policy_id"] == "desc_x1_r95_ladder_v1"
        assert len(row["hint_text_sha256"]) == 64
        assert row["valid_parse"] is True
        assert row["anchor_policy"]
        assert row["prefix_gt_indices"] == [2]
        assert row["prefix_pred_indices"] == []
        assert row["prefix_len"] == 1
        assert row["contains_future_gt_after_fn"] is True
        assert len(row["rendered_assistant_prefix_sha256"]) == 64
        assert row["prefix_objects"] == prefix_objects
        assert {"source", "gt_idx", "pred_idx", "desc", "bbox", "order_idx"} <= set(
            row["prefix_objects"][0]
        )
    json.dumps(rows, allow_nan=False, sort_keys=True)


def test_desc_only_same_desc_score_does_not_count_as_residual_success() -> None:
    case = _case(
        fn_case_id="case-same-desc",
        fn_gt_idx=1,
        fn_desc="chair",
        fn_bbox=[20, 0, 30, 10],
        emitted_gt_indices=[0],
        residual_gt_indices=[1],
    )
    specs = [
        {
            "probe_id": "probe-desc-favored-emitted-x1",
            "fn_case_id": "case-same-desc",
            "hint_level": "desc_x1",
            "prefix_condition": "teacher_sorted_prefix",
            "prefix_objects": [
                _prefix_object(
                    source="pred",
                    gt_idx=0,
                    pred_idx=4,
                    desc="chair",
                    bbox=[0, 0, 10, 10],
                    order_idx=0,
                )
            ],
            "generated_continuation": '{"desc":"chair","bbox":[0,0,10,10]}',
            "valid_parse": True,
        }
    ]
    candidate_rows = build_fn_candidate_score_rows(
        [
            {
                "probe_id": "probe-desc-favored-emitted-x1",
                "fn_case_id": "case-same-desc",
                "candidate_id": "desc-chair",
                "candidate_gt_idx": None,
                "desc": "chair",
                "role": "desc_only_same_desc",
                "score": 0.91,
            },
            {
                "probe_id": "probe-desc-favored-emitted-x1",
                "fn_case_id": "case-same-desc",
                "candidate_id": "desc-table",
                "candidate_gt_idx": None,
                "desc": "table",
                "role": "hard_competitor",
                "score": 0.10,
            },
        ]
    )
    slot_rows = build_fn_slot_evidence_rows(
        [
            {
                "probe_id": "probe-desc-favored-emitted-x1",
                "fn_case_id": "case-same-desc",
                "slot": "x1",
                "axis_len": 100,
                "gt_idx": 1,
                "gt_value": 20,
                "peak_value": 0,
                "score": 0.72,
            },
            {
                "probe_id": "probe-desc-favored-emitted-x1",
                "fn_case_id": "case-same-desc",
                "slot": "x1",
                "axis_len": 100,
                "gt_idx": 0,
                "gt_value": 0,
                "peak_value": 0,
                "score": 0.88,
            },
        ]
    )

    rows = build_fn_probe_rows(
        [case],
        probe_specs=specs,
        candidate_score_rows=candidate_rows,
        slot_evidence_rows=slot_rows,
    )
    row = rows[0]

    assert row["emitted_gt_indices"] == [0]
    assert row["residual_gt_indices"] == [1]
    assert row["desc_target_favored"] is True
    assert row["emitted_same_desc_x1_strict_hit"] is True
    assert row["residual_x1_strict_hit"] is False
    assert row["residual_accounting_success"] is False
    assert row["residual_accounting_failure"] is True
    assert row["coord_binding_failure"] is True
    assert row["not_rescued_under_valid_desc_x1_controls"] is True
    assert "vision_unreachable_fn" not in row
    json.dumps({"probe": rows, "scores": candidate_rows, "slots": slot_rows}, allow_nan=False)


def test_teacher_prefix_strict_x1_hit_and_rollout_loss_sets_prefix_flip() -> None:
    case = _case(
        fn_case_id="case-prefix-flip",
        fn_gt_idx=1,
        fn_desc="chair",
        fn_bbox=[20, 0, 30, 10],
        residual_gt_indices=[1],
    )
    specs = [
        {
            "probe_id": "probe-teacher",
            "fn_case_id": "case-prefix-flip",
            "hint_level": "desc_x1",
            "prefix_condition": "teacher_sorted_prefix",
            "prefix_objects": [],
            "generated_continuation": '{"desc":"chair","bbox":[20,0,30,10]}',
            "valid_parse": True,
        },
        {
            "probe_id": "probe-rollout",
            "fn_case_id": "case-prefix-flip",
            "hint_level": "desc_x1",
            "prefix_condition": "rollout_prefix",
            "prefix_objects": [],
            "generated_continuation": '{"desc":"chair","bbox":[0,0,10,10]}',
            "valid_parse": True,
        },
    ]
    candidate_rows = build_fn_candidate_score_rows(
        [
            _score("probe-teacher", "case-prefix-flip", desc="chair", score=0.7),
            _score("probe-rollout", "case-prefix-flip", desc="chair", score=0.7),
        ]
    )
    slot_rows = build_fn_slot_evidence_rows(
        [
            {
                "probe_id": "probe-teacher",
                "fn_case_id": "case-prefix-flip",
                "slot": "x1",
                "axis_len": 100,
                "gt_idx": 1,
                "gt_value": 20,
                "peak_value": 20,
                "score": 0.90,
            },
            {
                "probe_id": "probe-teacher",
                "fn_case_id": "case-prefix-flip",
                "slot": "y1",
                "axis_len": 100,
                "gt_idx": 1,
                "gt_value": 0,
                "peak_value": 0,
                "score": 0.90,
            },
            {
                "probe_id": "probe-teacher",
                "fn_case_id": "case-prefix-flip",
                "slot": "x2",
                "axis_len": 100,
                "gt_idx": 1,
                "gt_value": 30,
                "peak_value": 30,
                "score": 0.90,
            },
            {
                "probe_id": "probe-teacher",
                "fn_case_id": "case-prefix-flip",
                "slot": "y2",
                "axis_len": 100,
                "gt_idx": 1,
                "gt_value": 10,
                "peak_value": 10,
                "score": 0.90,
            },
            {
                "probe_id": "probe-rollout",
                "fn_case_id": "case-prefix-flip",
                "slot": "x1",
                "axis_len": 100,
                "gt_idx": 1,
                "gt_value": 20,
                "peak_value": 5,
                "score": 0.90,
            },
        ]
    )

    rows = build_fn_probe_rows(
        [case],
        probe_specs=specs,
        candidate_score_rows=candidate_rows,
        slot_evidence_rows=slot_rows,
    )
    by_id = {row["probe_id"]: row for row in rows}

    assert by_id["probe-teacher"]["residual_x1_strict_hit"] is True
    assert by_id["probe-teacher"]["residual_geometry_success"] is True
    assert by_id["probe-teacher"]["prefix_suppression_flip"] is False
    assert by_id["probe-rollout"]["residual_x1_strict_hit"] is False
    assert by_id["probe-rollout"]["residual_geometry_success"] is False
    assert by_id["probe-rollout"]["prefix_suppression_flip"] is True
    assert by_id["probe-rollout"]["primary_bucket"] == "prefix_suppression_flip"
    json.dumps(rows, allow_nan=False, sort_keys=True)


def test_desc_x1_hint_control_requires_later_coordinate_convergence() -> None:
    case = _case(
        fn_case_id="case-hint-control",
        fn_gt_idx=1,
        fn_desc="chair",
        fn_bbox=[20, 0, 30, 10],
        residual_gt_indices=[1],
    )
    specs = [
        {
            "probe_id": "probe-hinted-x1-bad-later-slots",
            "fn_case_id": "case-hint-control",
            "hint_level": "desc_x1",
            "prefix_condition": "teacher_sorted_prefix",
            "prefix_objects": [],
            "generated_continuation": '{"desc":"chair","bbox":[20,70,80,90]}',
            "valid_parse": True,
        }
    ]
    candidate_rows = build_fn_candidate_score_rows(
        [
            _score(
                "probe-hinted-x1-bad-later-slots",
                "case-hint-control",
                desc="chair",
                score=0.7,
            )
        ]
    )
    slot_rows = build_fn_slot_evidence_rows(
        [
            {
                "probe_id": "probe-hinted-x1-bad-later-slots",
                "fn_case_id": "case-hint-control",
                "slot": "x1",
                "axis_len": 10,
                "gt_idx": 1,
                "gt_value": 20,
                "peak_value": 20,
                "score": None,
                "hinted_control": True,
                "model_predicted": False,
            },
            {
                "probe_id": "probe-hinted-x1-bad-later-slots",
                "fn_case_id": "case-hint-control",
                "slot": "y1",
                "axis_len": 10,
                "gt_idx": 1,
                "gt_value": 0,
                "peak_value": 70,
                "score": 0.5,
            },
            {
                "probe_id": "probe-hinted-x1-bad-later-slots",
                "fn_case_id": "case-hint-control",
                "slot": "x2",
                "axis_len": 10,
                "gt_idx": 1,
                "gt_value": 30,
                "peak_value": 80,
                "score": 0.5,
            },
            {
                "probe_id": "probe-hinted-x1-bad-later-slots",
                "fn_case_id": "case-hint-control",
                "slot": "y2",
                "axis_len": 10,
                "gt_idx": 1,
                "gt_value": 10,
                "peak_value": 90,
                "score": 0.5,
            },
        ]
    )

    rows = build_fn_probe_rows(
        [case],
        probe_specs=specs,
        candidate_score_rows=candidate_rows,
        slot_evidence_rows=slot_rows,
    )
    row = rows[0]

    assert row["desc_target_favored"] is True
    assert row["valid_residual_x1_control"] is True
    assert row["residual_x1_hint_control_hit"] is True
    assert row["residual_x1_model_strict_hit"] is False
    assert row["all_required_slots_strict_hit"] is False
    assert row["residual_geometry_success"] is False
    assert row["probe_invalid_or_unscored"] is False
    assert row["coord_binding_failure"] is True
    assert row["primary_bucket"] == "coord_binding_failure"
    json.dumps({"probe": rows, "scores": candidate_rows, "slots": slot_rows}, allow_nan=False)


def test_invalid_generated_continuation_is_invalid_or_unscored_not_vision_bucket() -> None:
    rows = build_fn_probe_rows(
        [
            _case(
                fn_case_id="case-invalid",
                fn_gt_idx=3,
                fn_desc="cup",
                fn_bbox=[8, 8, 18, 18],
            )
        ],
        probe_specs=[
            {
                "probe_id": "probe-invalid",
                "fn_case_id": "case-invalid",
                "hint_level": "desc_x1",
                "prefix_condition": "empty_prefix",
                "prefix_objects": [],
                "generated_continuation": "not parseable",
                "valid_parse": False,
            }
        ],
    )

    assert rows[0]["probe_invalid_or_unscored"] is True
    assert rows[0]["primary_bucket"] == "probe_invalid_or_unscored"
    assert "vision_unreachable_fn" not in rows[0]
    assert "vision_unreachable_fn" not in json.dumps(rows, sort_keys=True)
    json.dumps(rows, allow_nan=False, sort_keys=True)


def test_missing_or_non_x1_slot_control_is_invalid_not_coord_failure() -> None:
    case = _case(
        fn_case_id="case-missing-x1-control",
        fn_gt_idx=1,
        fn_desc="chair",
        fn_bbox=[20, 0, 30, 10],
        residual_gt_indices=[1],
    )
    specs = [
        {
            "probe_id": "probe-no-slots",
            "fn_case_id": "case-missing-x1-control",
            "hint_level": "desc_x1",
            "prefix_condition": "empty_prefix",
            "prefix_objects": [],
            "generated_continuation": '{"desc":"chair"}',
            "valid_parse": True,
        },
        {
            "probe_id": "probe-y1-only",
            "fn_case_id": "case-missing-x1-control",
            "hint_level": "desc_x1",
            "prefix_condition": "empty_prefix",
            "prefix_objects": [],
            "generated_continuation": '{"desc":"chair"}',
            "valid_parse": True,
        },
        {
            "probe_id": "probe-unsupported-slot",
            "fn_case_id": "case-missing-x1-control",
            "hint_level": "desc_x1",
            "prefix_condition": "empty_prefix",
            "prefix_objects": [],
            "generated_continuation": '{"desc":"chair"}',
            "valid_parse": True,
        },
    ]
    candidate_rows = build_fn_candidate_score_rows(
        [
            _score(probe_id, "case-missing-x1-control", desc="chair", score=0.7)
            for probe_id in ("probe-no-slots", "probe-y1-only", "probe-unsupported-slot")
        ]
    )
    slot_rows = build_fn_slot_evidence_rows(
        [
            {
                "probe_id": "probe-y1-only",
                "fn_case_id": "case-missing-x1-control",
                "slot": "y1",
                "axis_len": 80,
                "gt_idx": 1,
                "gt_value": 0,
                "peak_value": 0,
                "score": 0.9,
            },
            {
                "probe_id": "probe-unsupported-slot",
                "fn_case_id": "case-missing-x1-control",
                "slot": "center_x",
                "axis_len": 100,
                "gt_idx": 1,
                "gt_value": 20,
                "peak_value": 20,
                "score": 0.9,
            },
        ]
    )

    rows = build_fn_probe_rows(
        [case],
        probe_specs=specs,
        candidate_score_rows=candidate_rows,
        slot_evidence_rows=slot_rows,
    )

    for row in rows:
        assert row["desc_target_favored"] is True
        assert row["valid_residual_x1_control"] is False
        assert row["probe_invalid_or_unscored"] is True
        assert row["coord_binding_failure"] is False
        assert row["not_rescued_under_valid_desc_x1_controls"] is False
        assert row["primary_bucket"] == "probe_invalid_or_unscored"
    json.dumps(rows, allow_nan=False, sort_keys=True)


def test_prefix_flip_requires_valid_rollout_residual_x1_control() -> None:
    case = _case(
        fn_case_id="case-prefix-invalid-rollout",
        fn_gt_idx=1,
        fn_desc="chair",
        fn_bbox=[20, 0, 30, 10],
        residual_gt_indices=[1],
    )
    specs = [
        {
            "probe_id": "probe-teacher-valid",
            "fn_case_id": "case-prefix-invalid-rollout",
            "hint_level": "desc_x1",
            "prefix_condition": "teacher_sorted_prefix",
            "prefix_objects": [],
            "generated_continuation": '{"desc":"chair","bbox":[20,0,30,10]}',
            "valid_parse": True,
        },
        {
            "probe_id": "probe-rollout-no-slot",
            "fn_case_id": "case-prefix-invalid-rollout",
            "hint_level": "desc_x1",
            "prefix_condition": "rollout_prefix",
            "prefix_objects": [],
            "generated_continuation": '{"desc":"chair"}',
            "valid_parse": True,
        },
        {
            "probe_id": "probe-rollout-invalid",
            "fn_case_id": "case-prefix-invalid-rollout",
            "hint_level": "desc_x1",
            "prefix_condition": "rollout_prefix",
            "prefix_objects": [],
            "generated_continuation": "not parseable",
            "valid_parse": False,
        },
    ]
    candidate_rows = build_fn_candidate_score_rows(
        [
            _score("probe-teacher-valid", "case-prefix-invalid-rollout", desc="chair", score=0.7),
            _score("probe-rollout-no-slot", "case-prefix-invalid-rollout", desc="chair", score=0.7),
            _score("probe-rollout-invalid", "case-prefix-invalid-rollout", desc="chair", score=0.7),
        ]
    )
    slot_rows = build_fn_slot_evidence_rows(
        [
            {
                "probe_id": "probe-teacher-valid",
                "fn_case_id": "case-prefix-invalid-rollout",
                "slot": "x1",
                "axis_len": 100,
                "gt_idx": 1,
                "gt_value": 20,
                "peak_value": 20,
                "score": 0.9,
            }
        ]
    )

    rows = build_fn_probe_rows(
        [case],
        probe_specs=specs,
        candidate_score_rows=candidate_rows,
        slot_evidence_rows=slot_rows,
    )
    by_id = {row["probe_id"]: row for row in rows}

    assert by_id["probe-teacher-valid"]["valid_residual_x1_control"] is True
    assert by_id["probe-rollout-no-slot"]["probe_invalid_or_unscored"] is True
    assert by_id["probe-rollout-no-slot"]["prefix_suppression_flip"] is False
    assert by_id["probe-rollout-invalid"]["probe_invalid_or_unscored"] is True
    assert by_id["probe-rollout-invalid"]["prefix_suppression_flip"] is False
    json.dumps(rows, allow_nan=False, sort_keys=True)


def test_duplicate_candidate_ids_are_rejected_per_probe() -> None:
    with pytest.raises(ValueError, match="duplicate candidate_id"):
        build_fn_candidate_score_rows(
            [
                {
                    "probe_id": "probe-duplicate",
                    "fn_case_id": "case-duplicate",
                    "candidate_id": "desc-chair",
                    "desc": "chair",
                    "score": 0.9,
                },
                {
                    "probe_id": "probe-duplicate",
                    "fn_case_id": "case-duplicate",
                    "candidate_id": "desc-chair",
                    "desc": "chair",
                    "score": 0.8,
                },
            ]
        )


def test_fn_probe_import_does_not_pull_heavy_or_gpu_modules() -> None:
    script = """
import json
import sys
import src.analysis.sorted_random_no_newline_phenotype.fn_probe
blocked = ["yaml", "torch", "transformers", "PIL", "numpy"]
print(json.dumps({name: name in sys.modules for name in blocked}, sort_keys=True))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        text=True,
        capture_output=True,
    )

    assert json.loads(result.stdout) == {
        "PIL": False,
        "numpy": False,
        "torch": False,
        "transformers": False,
        "yaml": False,
    }


def _case(
    *,
    fn_case_id: str,
    fn_gt_idx: int,
    fn_desc: str,
    fn_bbox: list[int],
    emitted_gt_indices: list[int] | None = None,
    residual_gt_indices: list[int] | None = None,
) -> dict[str, object]:
    return {
        "fn_case_id": fn_case_id,
        "checkpoint_role": "fullobj_sorted_pure_ce_ckpt3668",
        "split": "val",
        "image_id": "img-7",
        "width": 100,
        "height": 80,
        "fn_gt_idx": fn_gt_idx,
        "fn_desc": fn_desc,
        "fn_bbox": fn_bbox,
        "emitted_gt_indices": [] if emitted_gt_indices is None else emitted_gt_indices,
        "residual_gt_indices": (
            [fn_gt_idx] if residual_gt_indices is None else residual_gt_indices
        ),
    }


def _prefix_object(
    *,
    source: str,
    gt_idx: int | None,
    pred_idx: int | None,
    desc: str,
    bbox: list[int],
    order_idx: int,
) -> dict[str, object]:
    return {
        "source": source,
        "gt_idx": gt_idx,
        "pred_idx": pred_idx,
        "desc": desc,
        "bbox": bbox,
        "order_idx": order_idx,
    }


def _score(
    probe_id: str,
    fn_case_id: str,
    *,
    desc: str,
    score: float,
) -> dict[str, object]:
    return {
        "probe_id": probe_id,
        "fn_case_id": fn_case_id,
        "candidate_id": f"desc-{desc}",
        "candidate_gt_idx": None,
        "desc": desc,
        "role": "desc_only_same_desc",
        "score": score,
    }
