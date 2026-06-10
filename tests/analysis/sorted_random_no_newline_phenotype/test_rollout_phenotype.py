from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.analysis.sorted_random_no_newline_phenotype.rollout_phenotype import (
    compute_rollout_phenotype,
    materialize_rollout_phenotype,
    normalize_rollout_row,
    summarize_rollout_phenotype,
)


ROLE_RANDOM = "fullobj_random_pure_ce_ckpt3668"
ROLE_SORTED = "fullobj_sorted_pure_ce_ckpt3668"
DECODE_POLICY = "free_text_unconstrained_greedy_temp0"
ARTIFACT_ROOT = (
    "/data/CoordExp/outputs/analysis/autoreg_object_rollout/"
    "sorted_random_no_newline_phenotype/"
    "fullobj_purece_random_vs_sorted_ckpt3668_phase_a3_2"
)
RANDOM_CHECKPOINT = (
    "/data/CoordExp/outputs/stage1_2b/recursive_detection_ce_latest/"
    "compact_full_fullobj_random_sft_bsz16_4epoch_tokenrows_v2/"
    "compact-full-fullobj-random-sft-bsz16-4epoch-tokenrows-v2/"
    "v1-20260601-062428/checkpoint-3668"
)
SORTED_CHECKPOINT = (
    "/data/CoordExp/outputs/stage1_2b/recursive_detection_ce_latest/"
    "compact_full_fullobj_sorted_sft_bsz16_4epoch_tokenrows_v2/"
    "compact-full-fullobj-sorted-sft-bsz16-4epoch-tokenrows-v2/"
    "v1-20260601-062429/checkpoint-3668"
)


def test_normalize_rollout_row_accepts_a3_2_free_text_contract() -> None:
    row = normalize_rollout_row(
        _rollout_row(
            image_id="img-1",
            pred_rows_ordered=[_pred("person", [10, 20, 50, 80])],
            invalid_pred_rows=[],
        )
    )

    assert row["checkpoint_role"] == ROLE_RANDOM
    assert row["image_id"] == "img-1"
    assert row["source_line_idx"] == 0
    assert row["raw_output_text_sha256"] == "0" * 64
    assert row["decode_policy"] == DECODE_POLICY
    assert row["constraint_policy"] == "none"
    assert row["native_prompt_ordering"] == "random_permutation"
    assert row["template_contract"] == {"row_separator": "none"}
    assert row["pred_rows_ordered"][0]["desc"] == "person"
    assert row["invalid_pred_rows"] == []


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"decode_policy": "compact_grammar_greedy_temp0"}, "decode_policy"),
        ({"constraint_policy": "compact_grammar"}, "constraint_policy"),
        ({"template_contract": {"row_separator": "newline"}}, "row_separator"),
    ],
)
def test_normalize_rollout_row_rejects_constrained_or_old_decode_policy(
    update: dict[str, Any],
    message: str,
) -> None:
    row = _rollout_row(image_id="img-1")
    row.update(update)

    with pytest.raises(ValueError, match=message):
        normalize_rollout_row(row)


def test_standard_eval_artifact_row_is_adapted_to_a3_2_contract() -> None:
    canonical_row = _standard_eval_row(
        checkpoint_role=ROLE_SORTED,
        native_prompt_ordering="sorted",
        source_line_idx=7,
        errors=["legacy_separator_in_new_format"],
        error_entries=[
            {
                "code": "legacy_separator_in_new_format",
                "message": "newline separator",
                "stage": "infer.parse_pred",
            }
        ],
    )

    normalized = normalize_rollout_row(canonical_row)
    rows = compute_rollout_phenotype([], [canonical_row])
    summary = summarize_rollout_phenotype([], [canonical_row])

    assert normalized["checkpoint_role"] == ROLE_SORTED
    assert normalized["image_id"] == "img-1"
    assert normalized["source_line_idx"] == 7
    assert normalized["raw_output_text_sha256"] == hashlib.sha256(
        canonical_row["raw_output_text"].encode("utf-8")
    ).hexdigest()
    assert normalized["rollout_stop_reason"] == "eos"
    assert normalized["pred_rows_ordered"] == [_pred("person", [11, 11, 59, 59])]
    assert normalized["invalid_pred_rows"] == [
        {
            "code": "legacy_separator_in_new_format",
            "message": "newline separator",
            "stage": "infer.parse_pred",
        }
    ]
    assert rows[0]["invalid_pred_row_count"] == 1
    assert rows[0]["early_eos"] is True
    assert rows[0]["gt_count"] == 1
    assert rows[0]["matched_gt_count"] == 1
    assert summary["metrics"]["instance_recall"] == pytest.approx(1.0)


def test_standard_eval_artifact_row_rejects_constrained_decode_metadata() -> None:
    row = {
        "image": "images/val/img-1.jpg",
        "width": 1024,
        "height": 768,
        "mode": "coord",
        "gt": [],
        "pred": [],
        "coord_mode": "pixel",
        "raw_output_json": {"objects": []},
        "raw_special_tokens": [],
        "raw_ends_with_im_end": False,
        "errors": [],
        "error_entries": [],
        "metadata": {
            "checkpoint_role": ROLE_RANDOM,
            "source_line_idx": 0,
            "decode_policy": "compact_grammar_greedy_temp0",
            "constraint_policy": "none",
            "native_prompt_ordering": "random_permutation",
            "template_contract": {"row_separator": "none"},
        },
    }

    with pytest.raises(ValueError, match="decode_policy"):
        normalize_rollout_row(row)


@pytest.mark.parametrize(
    ("missing_field", "message"),
    [
        ("decode_policy", "decode_policy"),
        ("constraint_policy", "constraint_policy"),
        ("template_contract", "template_contract"),
        ("source_line_idx", "source_line_idx"),
    ],
)
def test_standard_eval_artifact_row_requires_explicit_a3_2_provenance(
    missing_field: str,
    message: str,
) -> None:
    row = _standard_eval_row()
    del row["metadata"][missing_field]

    with pytest.raises(ValueError, match=message):
        normalize_rollout_row(row)


def test_summary_counts_class_any_vs_instance_recall_for_missed_instance() -> None:
    gt_rows = [
        _gt_record(
            "img-1",
            [
                _gt("person", [10, 10, 60, 60], gt_idx=0),
                _gt("person", [100, 10, 160, 60], gt_idx=1),
            ],
        )
    ]
    rollout_rows = [
        _rollout_row(
            image_id="img-1",
            pred_rows_ordered=[_pred("person", [11, 11, 59, 59])],
            rollout_stop_reason="max_new_tokens",
        )
    ]

    summary = summarize_rollout_phenotype(gt_rows, rollout_rows)
    row = compute_rollout_phenotype(gt_rows, rollout_rows)[0]

    assert summary["metrics"]["class_any_recall"] == pytest.approx(1.0)
    assert summary["metrics"]["instance_recall"] == pytest.approx(0.5)
    assert summary["metrics"]["class_found_but_instance_missed_rate"] == pytest.approx(
        1.0
    )
    assert summary["metrics"]["same_desc_fn_rate"] == pytest.approx(0.5)
    assert row["same_desc_fn_gt_indices"] == [1]


def test_same_desc_duplicate_side_label_for_near_identical_predictions() -> None:
    gt_rows = [_gt_record("img-1", [_gt("person", [10, 10, 60, 60], gt_idx=0)])]
    rollout_rows = [
        _rollout_row(
            image_id="img-1",
            pred_rows_ordered=[
                _pred("person", [10, 10, 60, 60]),
                _pred("person", [10, 10, 60, 60]),
            ],
        )
    ]

    rows = compute_rollout_phenotype(gt_rows, rollout_rows)
    summary = summarize_rollout_phenotype(gt_rows, rollout_rows)

    assert rows[0]["duplicate_side_label_count"] == 1
    assert rows[0]["duplicate_side_labels"] == [
        {
            "label": "same_desc_duplicate",
            "desc": "person",
            "pred_idx": 1,
            "duplicate_of_pred_idx": 0,
            "iou": 1.0,
        }
    ]
    assert summary["metrics"]["duplicate_side_label_count"] == 1
    assert summary["metrics"]["same_desc_duplication_rate"] == pytest.approx(0.5)


def test_eos_after_partial_coverage_is_counted() -> None:
    gt_rows = [
        _gt_record(
            "img-1",
            [
                _gt("person", [10, 10, 60, 60], gt_idx=0),
                _gt("chair", [100, 10, 160, 60], gt_idx=1),
            ],
        )
    ]
    rollout_rows = [
        _rollout_row(
            image_id="img-1",
            pred_rows_ordered=[_pred("person", [10, 10, 60, 60])],
            rollout_stop_reason="eos",
        )
    ]

    rows = compute_rollout_phenotype(gt_rows, rollout_rows)
    summary = summarize_rollout_phenotype(gt_rows, rollout_rows)

    assert rows[0]["eos_after_partial_coverage"] is True
    assert summary["metrics"]["eos_after_partial_coverage_rate"] == pytest.approx(1.0)


def test_degenerate_and_invalid_row_counters_are_phenotype_not_fatal() -> None:
    gt_rows = [_gt_record("img-1", [_gt("person", [10, 10, 60, 60], gt_idx=0)])]
    rollout_rows = [
        _rollout_row(
            image_id="img-1",
            pred_rows_ordered=[
                _pred("person", [10, 10, 60, 60]),
                _pred("chair", [100, 100, 100, 120]),
                {"bbox_xyxy": [1, 2, 3, 4]},
            ],
            invalid_pred_rows=[
                {"raw": "<bad-row>", "error": "wrong_coord_arity"},
                {"raw": "<bad-row-2>", "error": "missing_box_start"},
            ],
        )
    ]

    rows = compute_rollout_phenotype(gt_rows, rollout_rows)
    summary = summarize_rollout_phenotype(gt_rows, rollout_rows)

    assert rows[0]["invalid_pred_row_count"] == 2
    assert rows[0]["degenerate_box_count"] == 1
    assert rows[0]["malformed_pred_row_count"] == 1
    assert summary["metrics"]["parse_invalid_count"] == 2
    assert summary["metrics"]["degenerate_box_count"] == 1
    assert summary["metrics"]["malformed_pred_row_count"] == 1


def test_sorted_gt_order_agreement_uses_canonical_gt_order() -> None:
    gt_rows = [
        _gt_record(
            "img-1",
            [
                _gt("bottom", [50, 500, 90, 560], gt_idx=0),
                _gt("top-left", [10, 10, 40, 40], gt_idx=1),
                _gt("top-right", [80, 20, 120, 60], gt_idx=2),
            ],
            canonical_sorted_gt_indices=[1, 2, 0],
        )
    ]
    rollout_rows = [
        _rollout_row(
            image_id="img-1",
            checkpoint_role=ROLE_SORTED,
            native_prompt_ordering="sorted",
            pred_rows_ordered=[
                _pred("top-left", [10, 10, 40, 40]),
                _pred("top-right", [80, 20, 120, 60]),
                _pred("bottom", [50, 500, 90, 560]),
            ],
        )
    ]

    rows = compute_rollout_phenotype(gt_rows, rollout_rows)
    summary = summarize_rollout_phenotype(gt_rows, rollout_rows)

    assert rows[0]["matched_gt_indices_in_pred_order"] == [1, 2, 0]
    assert rows[0]["canonical_sorted_gt_indices"] == [1, 2, 0]
    assert rows[0]["sorted_gt_order_agreement"] is True
    assert summary["metrics"]["sorted_gt_order_agreement"] == pytest.approx(1.0)


def test_summary_outputs_are_strict_json() -> None:
    gt_rows = [_gt_record("img-1", [])]
    rollout_rows = [
        _rollout_row(
            image_id="img-1",
            pred_rows_ordered=[],
            rollout_stop_reason="eos",
        )
    ]

    rows = compute_rollout_phenotype(gt_rows, rollout_rows)
    summary = summarize_rollout_phenotype(gt_rows, rollout_rows)

    json.dumps(rows, allow_nan=False, sort_keys=True)
    json.dumps(summary, allow_nan=False, sort_keys=True)


def test_materialize_rollout_phenotype_writes_required_rollout_artifacts(
    tmp_path: Path,
) -> None:
    gt_rows: list[dict[str, Any]] = []
    rollout_row = _standard_eval_row()
    rollout_row.update(
        {
            "runtime_kind": "real_gpu_native_rollout_v1",
            "checkpoint_fingerprint": "model:test",
            "gpu_id": "0",
        }
    )
    rollout_rows = [rollout_row]

    result = materialize_rollout_phenotype(
        tmp_path,
        gt_rows,
        rollout_rows,
    )

    assert result["rows_path"] == str(tmp_path / "rollout" / "rollout_phenotype_rows.jsonl")
    assert result["summary_path"] == str(tmp_path / "rollout" / "rollout_summary.json")
    rows_path = Path(result["rows_path"])
    summary_path = Path(result["summary_path"])
    assert rows_path.is_file()
    assert summary_path.is_file()

    persisted_rows = [
        json.loads(line)
        for line in rows_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    persisted_summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert len(persisted_rows) == 1
    assert persisted_rows[0]["gt_count"] == 1
    assert persisted_rows[0]["matched_gt_count"] == 1
    assert persisted_rows[0]["source_runtime_kind"] == "real_gpu_native_rollout_v1"
    assert persisted_rows[0]["source_checkpoint_fingerprint"] == "model:test"
    assert persisted_rows[0]["source_gpu_id"] == "0"
    assert persisted_summary["row_count"] == 1
    assert persisted_summary["metrics"]["instance_recall"] == pytest.approx(1.0)
    json.dumps(persisted_rows, allow_nan=False, sort_keys=True)
    json.dumps(persisted_summary, allow_nan=False, sort_keys=True)


def test_native_greedy_yaml_specs_are_unconstrained_and_role_specific() -> None:
    cases = [
        (
            Path(
                "configs/infer/recursive_detection_ce/"
                "fullobj_random_purece_ckpt3668_a3_2_rollout1024_greedy.yaml"
            ),
            ROLE_RANDOM,
            RANDOM_CHECKPOINT,
            "random",
            "random_permutation",
        ),
        (
            Path(
                "configs/infer/recursive_detection_ce/"
                "fullobj_sorted_purece_ckpt3668_a3_2_rollout1024_greedy.yaml"
            ),
            ROLE_SORTED,
            SORTED_CHECKPOINT,
            "sorted",
            "sorted",
        ),
    ]

    for path, role, checkpoint, object_ordering, native_prompt_ordering in cases:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))

        assert config["stages"] == {"infer": True, "eval": False, "vis": False}
        assert config["artifacts"]["run_dir"] == f"{ARTIFACT_ROOT}/rollout/{role}"
        assert config["infer"]["model_checkpoint"] == checkpoint
        assert config["infer"]["gt_jsonl"] == (
            "/data/CoordExp/public_data/coco/"
            "rescale_32_1024_bbox_len12000/val.coord.jsonl"
        )
        assert config["run"]["root_image_dir"] == (
            "/data/CoordExp/public_data/coco/rescale_32_1024_bbox"
        )
        assert config["infer"]["limit"] == 1024
        assert config["infer"]["detection_sequence_format"] == "compact_full"
        assert config["infer"]["object_field_order"] == "desc_first"
        assert config["infer"]["object_ordering"] == object_ordering
        assert config["infer"]["parsing"]["compact_full"]["mode"] == (
            "marker_delimited_strict"
        )
        assert config["infer"]["generation"]["decode_mode"] == "greedy"
        assert config["infer"]["generation"]["temperature"] == 0.0
        assert config["infer"]["generation"]["do_sample"] is False
        assert config["infer"]["generation"]["top_p"] == 1.0
        assert config["infer"]["generation"]["num_beams"] == 1
        assert config["infer"]["generation"]["repetition_penalty"] == 1.0
        assert config["a3_2_launch_spec"]["checkpoint_role"] == role
        assert config["a3_2_launch_spec"]["decode_policy"] == DECODE_POLICY
        assert config["a3_2_launch_spec"]["constraint_policy"] == "none"
        assert config["a3_2_launch_spec"]["native_prompt_ordering"] == (
            native_prompt_ordering
        )
        assert config["a3_2_launch_spec"]["template_contract"]["row_separator"] == (
            "none"
        )
        assert config["a3_2_launch_spec"]["template_contract"][
            "detection_sequence_format"
        ] == "compact_full"
        assert config["a3_2_launch_spec"]["template_contract"][
            "compact_full_parse_mode"
        ] == "marker_delimited_strict"
        assert config["a3_2_launch_spec"]["output_root"] == (
            f"{ARTIFACT_ROOT}/rollout/{role}"
        )
        assert_no_enabled_decode_constraints(config)


def assert_no_enabled_decode_constraints(value: Any, path: str = "") -> None:
    forbidden_key_parts = (
        "compact_grammar",
        "grammar",
        "trie",
        "force_row",
        "forced_row",
        "constrained_decode",
        "logits_processor",
        "generation_constraints",
        "stop_pressure",
    )
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            if str(key) == "constraint_policy":
                assert child == "none"
                continue
            if any(part in str(key) for part in forbidden_key_parts):
                assert _is_disabled(child), f"{child_path} is enabled: {child!r}"
            assert_no_enabled_decode_constraints(child, child_path)
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            assert_no_enabled_decode_constraints(child, f"{path}[{idx}]")


def _is_disabled(value: Any) -> bool:
    if value in (None, False, 0, "", "none", "disabled"):
        return True
    if isinstance(value, dict):
        if value.get("enabled") is False:
            return True
        return not value
    if isinstance(value, list):
        return not value
    return False


def _rollout_row(
    *,
    image_id: str,
    checkpoint_role: str = ROLE_RANDOM,
    native_prompt_ordering: str = "random_permutation",
    pred_rows_ordered: list[Any] | None = None,
    invalid_pred_rows: list[Any] | None = None,
    rollout_stop_reason: str = "eos",
) -> dict[str, Any]:
    return {
        "checkpoint_role": checkpoint_role,
        "image_id": image_id,
        "source_line_idx": 0,
        "raw_output_text_sha256": "0" * 64,
        "pred_rows_ordered": pred_rows_ordered or [],
        "invalid_pred_rows": invalid_pred_rows or [],
        "rollout_stop_reason": rollout_stop_reason,
        "decode_policy": DECODE_POLICY,
        "constraint_policy": "none",
        "native_prompt_ordering": native_prompt_ordering,
        "template_contract": {"row_separator": "none"},
    }


def _gt_record(
    image_id: str,
    objects: list[dict[str, Any]],
    *,
    canonical_sorted_gt_indices: list[int] | None = None,
) -> dict[str, Any]:
    record = {"image_id": image_id, "objects": objects}
    if canonical_sorted_gt_indices is not None:
        record["canonical_sorted_gt_indices"] = canonical_sorted_gt_indices
    return record


def _gt(desc: str, bbox_xyxy: list[int], *, gt_idx: int) -> dict[str, Any]:
    return {"gt_idx": gt_idx, "desc": desc, "bbox_xyxy": bbox_xyxy}


def _pred(desc: str, bbox_xyxy: list[int]) -> dict[str, Any]:
    return {"desc": desc, "bbox_xyxy": bbox_xyxy}


def _standard_eval_row(
    *,
    checkpoint_role: str = ROLE_RANDOM,
    native_prompt_ordering: str = "random_permutation",
    source_line_idx: int = 0,
    errors: list[str] | None = None,
    error_entries: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    pred = _pred("person", [11, 11, 59, 59])
    return {
        "image": "images/val/img-1.jpg",
        "image_id": "img-1",
        "width": 1024,
        "height": 768,
        "mode": "coord",
        "gt": [_gt("person", [10, 10, 60, 60], gt_idx=0)],
        "pred": [pred],
        "coord_mode": "pixel",
        "raw_output_text": "<|object_ref_start|>person<|box_start|>",
        "raw_output_json": {"objects": [pred]},
        "raw_special_tokens": ["<|im_end|>"],
        "raw_ends_with_im_end": True,
        "errors": errors or [],
        "error_entries": error_entries or [],
        "metadata": {
            "checkpoint_role": checkpoint_role,
            "source_line_idx": source_line_idx,
            "decode_policy": DECODE_POLICY,
            "constraint_policy": "none",
            "native_prompt_ordering": native_prompt_ordering,
            "template_contract": {"row_separator": "none"},
        },
    }
