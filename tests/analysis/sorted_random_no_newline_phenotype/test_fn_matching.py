from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy

import pytest

from src.analysis.sorted_random_no_newline_phenotype.fn_matching import (
    MATCH_POLICY_ID,
    build_fn_case_universe,
    build_replayable_fn_cases,
    match_fn_cases_for_image,
)


def test_same_desc_match_wrong_desc_near_miss_and_duplicate_side_labels() -> None:
    gt_rows = [
        _gt(0, "chair", [0, 0, 10, 10]),
        _gt(1, "table", [20, 0, 30, 10]),
        _gt(2, "lamp", [40, 0, 50, 10]),
        _gt(3, "cup", [60, 0, 70, 10]),
        _gt(4, "bottle", [80, 0, 90, 10]),
    ]
    pred_rows = [
        _pred(0, "chair", [0, 0, 10, 10]),
        _pred(1, "plant", [20, 0, 30, 10]),
        _pred(2, "lamp", [45, 0, 55, 10]),
        _pred(3, "cup", [60, 0, 70, 10]),
        _pred(4, "cup", [60, 0, 70, 10]),
    ]

    ledger = match_fn_cases_for_image(gt_rows, pred_rows)

    by_gt = {row["gt_idx"]: row for row in ledger["gt_match_rows"]}
    assert by_gt[0]["is_fn"] is False
    assert by_gt[0]["match_pred_idx"] == 0
    assert by_gt[1]["is_fn"] is True
    assert by_gt[1]["wrong_desc_overlap"] is True
    assert by_gt[2]["is_fn"] is True
    assert by_gt[2]["near_miss"] is True
    assert by_gt[3]["is_fn"] is False
    assert by_gt[3]["same_desc_duplicate"] is True
    assert by_gt[4]["is_fn"] is True
    assert by_gt[4]["near_miss"] is False
    assert by_gt[4]["wrong_desc_overlap"] is False
    assert by_gt[4]["same_desc_duplicate"] is False
    assert {match["gt_idx"] for match in ledger["accepted_matches"]} == {0, 3}


def test_gt_points_bbox_surface_is_accepted_for_coco_jsonl_rows() -> None:
    gt_rows = [
        {
            "desc": "clock",
            "points": [873, 236, 901, 279],
            "type": "bbox_2d",
            "gt_idx": 0,
            "gt_sorted_rank": 1,
            "same_desc_gt_count": 1,
            "object_count": 20,
        }
    ]
    pred_rows = [_pred(0, "clock", [873, 236, 901, 279])]

    ledger = match_fn_cases_for_image(gt_rows, pred_rows)

    assert ledger["gt_match_rows"][0]["is_fn"] is False
    assert ledger["gt_match_rows"][0]["gt_bbox"] == [873, 236, 901, 279]
    assert ledger["accepted_matches"][0]["iou"] == pytest.approx(1.0)


def test_side_labels_do_not_change_main_fn_status() -> None:
    gt_rows = [
        _gt(0, "spoon", [0, 0, 10, 10]),
        _gt(1, "spoon", [0, 0, 10, 10]),
        _gt(2, "fork", [20, 0, 30, 10]),
    ]
    pred_rows = [
        _pred(0, "spoon", [0, 0, 10, 10]),
        _pred(1, "fork", [25, 0, 35, 10]),
        _pred(2, "knife", [20, 0, 30, 10]),
    ]

    ledger = match_fn_cases_for_image(gt_rows, pred_rows)

    by_gt = {row["gt_idx"]: row for row in ledger["gt_match_rows"]}
    assert by_gt[0]["is_fn"] is False
    assert by_gt[1]["is_fn"] is True
    assert by_gt[1]["same_desc_duplicate"] is True
    assert by_gt[2]["is_fn"] is True
    assert by_gt[2]["near_miss"] is True
    assert by_gt[2]["wrong_desc_overlap"] is True


def test_tie_breaking_uses_negative_iou_then_pred_idx_then_gt_idx() -> None:
    pred_tie = match_fn_cases_for_image(
        [_gt(0, "chair", [0, 0, 10, 10])],
        [
            _pred(5, "chair", [0, 0, 10, 10]),
            _pred(2, "chair", [0, 0, 10, 10]),
        ],
    )
    assert pred_tie["gt_match_rows"][0]["match_pred_idx"] == 2

    gt_tie = match_fn_cases_for_image(
        [
            _gt(7, "lamp", [0, 0, 10, 10]),
            _gt(3, "lamp", [0, 0, 10, 10]),
        ],
        [_pred(0, "lamp", [0, 0, 10, 10])],
    )
    by_gt = {row["gt_idx"]: row for row in gt_tie["gt_match_rows"]}
    assert by_gt[3]["match_pred_idx"] == 0
    assert by_gt[3]["is_fn"] is False
    assert by_gt[7]["match_pred_idx"] is None
    assert by_gt[7]["is_fn"] is True


def test_fn_universe_carries_both_checkpoint_statuses_and_membership() -> None:
    gt_rows = [
        _gt(0, "chair", [0, 0, 10, 10]),
        _gt(1, "table", [20, 0, 30, 10]),
        _gt(2, "lamp", [40, 0, 50, 10]),
        _gt(3, "cup", [60, 0, 70, 10]),
    ]
    random_ledger = match_fn_cases_for_image(
        gt_rows,
        [
            _pred(0, "chair", [0, 0, 10, 10]),
            _pred(1, "cup", [60, 0, 70, 10]),
        ],
        split="val",
        image_id=11,
    )
    sorted_ledger = match_fn_cases_for_image(
        gt_rows,
        [
            _pred(0, "table", [20, 0, 30, 10]),
            _pred(1, "cup", [60, 0, 70, 10]),
        ],
        split="val",
        image_id=11,
    )

    rows = build_fn_case_universe(
        gt_rows,
        random_match_ledger=random_ledger,
        sorted_match_ledger=sorted_ledger,
        split="val",
        image_id=11,
        sampled_gt_object_keys={"val:11:0", "val:11:1", "val:11:2"},
        sampling_reasons={"val:11:2": "same_desc_hard_case"},
    )

    by_gt = {row["gt_object_key"]: row for row in rows}
    assert by_gt["val:11:0"] == {
        "gt_object_key": "val:11:0",
        "is_fn_fullobj_random_pure_ce_ckpt3668": False,
        "is_fn_fullobj_sorted_pure_ce_ckpt3668": True,
        "random_match_pred_idx": 0,
        "sorted_match_pred_idx": None,
        "fn_membership": "sorted_only_fn",
        "sampled_for_probe": True,
        "sampling_reason": "selected_for_probe",
    }
    assert by_gt["val:11:1"]["fn_membership"] == "random_only_fn"
    assert by_gt["val:11:2"]["fn_membership"] == "shared_fn"
    assert by_gt["val:11:2"]["sampling_reason"] == "same_desc_hard_case"
    assert by_gt["val:11:3"]["fn_membership"] == "not_fn"
    assert by_gt["val:11:3"]["sampled_for_probe"] is False
    assert all(
        "is_fn_fullobj_random_pure_ce_ckpt3668" in row
        and "is_fn_fullobj_sorted_pure_ce_ckpt3668" in row
        for row in rows
    )
    json.dumps(rows, allow_nan=False, sort_keys=True)


def test_replayable_fn_cases_are_complete_json_safe_and_deterministic() -> None:
    gt_rows = [
        _gt(
            0,
            "chair",
            [0, 0, 10, 10],
            gt_sorted_rank=1,
            same_desc_gt_count=2,
            object_count=2,
        ),
        _gt(
            1,
            "chair",
            [0, 0, 10, 10],
            gt_sorted_rank=2,
            same_desc_gt_count=2,
            object_count=2,
        ),
    ]
    random_ledger = match_fn_cases_for_image(
        gt_rows,
        [_pred(0, "chair", [0, 0, 10, 10])],
        split="val",
        image_id="img-9",
    )
    sorted_ledger = match_fn_cases_for_image(
        gt_rows,
        [],
        split="val",
        image_id="img-9",
    )
    universe_rows = build_fn_case_universe(
        gt_rows,
        random_match_ledger=random_ledger,
        sorted_match_ledger=sorted_ledger,
        split="val",
        image_id="img-9",
        sampled_gt_object_keys={"val:img-9:1"},
        sampling_reasons={"val:img-9:1": "shared_fn_probe"},
    )
    contexts_by_role = {
        "fullobj_random_pure_ce_ckpt3668": _context(
            checkpoint_role="fullobj_random_pure_ce_ckpt3668",
            checkpoint_fingerprint="random-sha",
        ),
        "fullobj_sorted_pure_ce_ckpt3668": _context(
            checkpoint_role="fullobj_sorted_pure_ce_ckpt3668",
            checkpoint_fingerprint="sorted-sha",
        ),
    }

    rows = build_replayable_fn_cases(
        universe_rows,
        match_ledgers_by_role={
            "fullobj_random_pure_ce_ckpt3668": random_ledger,
            "fullobj_sorted_pure_ce_ckpt3668": sorted_ledger,
        },
        contexts_by_role=contexts_by_role,
    )
    rows_again = build_replayable_fn_cases(
        deepcopy(universe_rows),
        match_ledgers_by_role={
            "fullobj_random_pure_ce_ckpt3668": deepcopy(random_ledger),
            "fullobj_sorted_pure_ce_ckpt3668": deepcopy(sorted_ledger),
        },
        contexts_by_role=deepcopy(contexts_by_role),
    )

    assert rows == rows_again
    assert [row["checkpoint_role"] for row in rows] == [
        "fullobj_random_pure_ce_ckpt3668",
        "fullobj_sorted_pure_ce_ckpt3668",
    ]
    assert len({row["fn_case_id"] for row in rows}) == 2
    required_fields = {
        "fn_case_id",
        "checkpoint_role",
        "split",
        "image_id",
        "source_line_idx",
        "image_path",
        "width",
        "height",
        "coord_mode",
        "bbox_surface",
        "data_root",
        "jsonl_sha256",
        "checkpoint_fingerprint",
        "decode_policy",
        "template_contract",
        "fn_gt_idx",
        "fn_desc",
        "fn_bbox",
        "gt_sorted_rank",
        "same_desc_gt_count",
        "object_count",
        "pred_rows_ordered",
        "match_policy_id",
        "match_candidates_same_desc",
        "accepted_matches",
        "best_same_desc_iou",
        "near_miss",
        "wrong_desc_overlap",
        "same_desc_duplicate",
        "duplicate_source",
        "invalid_pred_count",
        "sample_stratum",
    }
    assert all(required_fields <= row.keys() for row in rows)
    assert rows[0]["fn_gt_idx"] == 1
    assert rows[0]["fn_desc"] == "chair"
    assert rows[0]["fn_bbox"] == [0, 0, 10, 10]
    assert rows[0]["same_desc_duplicate"] is True
    assert rows[0]["duplicate_source"] == "same_desc_iou_gt_0_pred_0"
    assert rows[0]["sample_stratum"] == "shared_fn"
    assert rows[1]["accepted_matches"] == []
    assert rows[1]["sample_stratum"] == "shared_fn"
    required_value_fields = required_fields - {"duplicate_source", "best_same_desc_iou"}
    assert all(row[field] is not None for row in rows for field in required_value_fields)
    json.dumps(rows, allow_nan=False, sort_keys=True)


def test_fn_universe_and_replay_support_multiple_full_checkpoint_roles() -> None:
    gt_rows = [
        _gt(0, "chair", [0, 0, 10, 10]),
        _gt(1, "table", [20, 0, 30, 10]),
    ]
    pure_random = match_fn_cases_for_image(
        gt_rows,
        [_pred(0, "chair", [0, 0, 10, 10])],
        split="val",
        image_id=17,
    )
    et_random = match_fn_cases_for_image(
        gt_rows,
        [_pred(0, "table", [20, 0, 30, 10])],
        split="val",
        image_id=17,
    )
    et_sorted = match_fn_cases_for_image(
        gt_rows,
        [],
        split="val",
        image_id=17,
    )
    match_ledgers_by_role = {
        "fullobj_random_pure_ce_ckpt3668": pure_random,
        "fullobj_random_et_rmp_ce_ckpt3668": et_random,
        "fullobj_sorted_et_rmp_ce_ckpt3668": et_sorted,
    }

    universe_rows = build_fn_case_universe(
        gt_rows,
        match_ledgers_by_role=match_ledgers_by_role,
        split="val",
        image_id=17,
        sampled_gt_object_keys={"val:17:0", "val:17:1"},
    )
    by_gt = {row["gt_object_key"]: row for row in universe_rows}

    assert by_gt["val:17:0"]["is_fn_fullobj_random_pure_ce_ckpt3668"] is False
    assert by_gt["val:17:0"]["is_fn_fullobj_random_et_rmp_ce_ckpt3668"] is True
    assert by_gt["val:17:0"]["is_fn_fullobj_sorted_et_rmp_ce_ckpt3668"] is True
    assert by_gt["val:17:0"]["match_pred_idx_fullobj_random_et_rmp_ce_ckpt3668"] is None
    assert by_gt["val:17:0"]["fn_membership"] == (
        "fn:fullobj_random_et_rmp_ce_ckpt3668,"
        "fullobj_sorted_et_rmp_ce_ckpt3668"
    )
    assert by_gt["val:17:1"]["is_fn_fullobj_random_pure_ce_ckpt3668"] is True
    assert by_gt["val:17:1"]["is_fn_fullobj_random_et_rmp_ce_ckpt3668"] is False

    rows = build_replayable_fn_cases(
        universe_rows,
        match_ledgers_by_role=match_ledgers_by_role,
        contexts_by_role={
            role: _context(checkpoint_role=role, checkpoint_fingerprint=f"{role}-sha")
            for role in match_ledgers_by_role
        },
    )

    assert [(row["fn_gt_idx"], row["checkpoint_role"]) for row in rows] == [
        (0, "fullobj_random_et_rmp_ce_ckpt3668"),
        (0, "fullobj_sorted_et_rmp_ce_ckpt3668"),
        (1, "fullobj_random_pure_ce_ckpt3668"),
        (1, "fullobj_sorted_et_rmp_ce_ckpt3668"),
    ]
    assert all(row["sample_stratum"].startswith("fn:") for row in rows)
    json.dumps(universe_rows, allow_nan=False, sort_keys=True)
    json.dumps(rows, allow_nan=False, sort_keys=True)


def test_replayable_fn_cases_reject_missing_expected_checkpoint_context() -> None:
    gt_rows, random_ledger, sorted_ledger, universe_rows = _shared_fn_inputs()

    with pytest.raises(ValueError, match="contexts_by_role.*missing.*sorted"):
        build_replayable_fn_cases(
            universe_rows,
            match_ledgers_by_role={
                "fullobj_random_pure_ce_ckpt3668": random_ledger,
                "fullobj_sorted_pure_ce_ckpt3668": sorted_ledger,
            },
            contexts_by_role={
                "fullobj_random_pure_ce_ckpt3668": _context(
                    checkpoint_role="fullobj_random_pure_ce_ckpt3668",
                    checkpoint_fingerprint="random-sha",
                ),
            },
        )
    assert gt_rows


@pytest.mark.parametrize(
    "missing_field",
    ["jsonl_sha256", "checkpoint_fingerprint", "image_path"],
)
def test_replayable_fn_cases_reject_missing_required_context_fields(
    missing_field: str,
) -> None:
    _, random_ledger, sorted_ledger, universe_rows = _shared_fn_inputs()
    contexts_by_role = {
        "fullobj_random_pure_ce_ckpt3668": _context(
            checkpoint_role="fullobj_random_pure_ce_ckpt3668",
            checkpoint_fingerprint="random-sha",
        ),
        "fullobj_sorted_pure_ce_ckpt3668": _context(
            checkpoint_role="fullobj_sorted_pure_ce_ckpt3668",
            checkpoint_fingerprint="sorted-sha",
        ),
    }
    contexts_by_role["fullobj_sorted_pure_ce_ckpt3668"].pop(missing_field)

    with pytest.raises(ValueError, match=missing_field):
        build_replayable_fn_cases(
            universe_rows,
            match_ledgers_by_role={
                "fullobj_random_pure_ce_ckpt3668": random_ledger,
                "fullobj_sorted_pure_ce_ckpt3668": sorted_ledger,
            },
            contexts_by_role=contexts_by_role,
        )


def test_replayable_fn_cases_reject_missing_expected_fn_field() -> None:
    _, random_ledger, sorted_ledger, universe_rows = _shared_fn_inputs()
    universe_rows[0].pop("is_fn_fullobj_sorted_pure_ce_ckpt3668")

    with pytest.raises(ValueError, match="missing.*is_fn_fullobj_sorted"):
        build_replayable_fn_cases(
            universe_rows,
            match_ledgers_by_role={
                "fullobj_random_pure_ce_ckpt3668": random_ledger,
                "fullobj_sorted_pure_ce_ckpt3668": sorted_ledger,
            },
            contexts_by_role={
                "fullobj_random_pure_ce_ckpt3668": _context(
                    checkpoint_role="fullobj_random_pure_ce_ckpt3668",
                    checkpoint_fingerprint="random-sha",
                ),
                "fullobj_sorted_pure_ce_ckpt3668": _context(
                    checkpoint_role="fullobj_sorted_pure_ce_ckpt3668",
                    checkpoint_fingerprint="sorted-sha",
                ),
            },
        )


def test_replayable_fn_cases_reject_non_bool_expected_fn_field() -> None:
    _, random_ledger, sorted_ledger, universe_rows = _shared_fn_inputs()
    universe_rows[0]["is_fn_fullobj_sorted_pure_ce_ckpt3668"] = None

    with pytest.raises(ValueError, match="is_fn_fullobj_sorted.*bool"):
        build_replayable_fn_cases(
            universe_rows,
            match_ledgers_by_role={
                "fullobj_random_pure_ce_ckpt3668": random_ledger,
                "fullobj_sorted_pure_ce_ckpt3668": sorted_ledger,
            },
            contexts_by_role={
                "fullobj_random_pure_ce_ckpt3668": _context(
                    checkpoint_role="fullobj_random_pure_ce_ckpt3668",
                    checkpoint_fingerprint="random-sha",
                ),
                "fullobj_sorted_pure_ce_ckpt3668": _context(
                    checkpoint_role="fullobj_sorted_pure_ce_ckpt3668",
                    checkpoint_fingerprint="sorted-sha",
                ),
            },
        )


def test_replayable_fn_cases_reject_sequence_match_ledgers() -> None:
    _, random_ledger, sorted_ledger, universe_rows = _shared_fn_inputs()

    with pytest.raises(ValueError, match="match_ledgers_by_role.*mapping"):
        build_replayable_fn_cases(
            universe_rows,
            match_ledgers_by_role={
                "fullobj_random_pure_ce_ckpt3668": random_ledger,
                "fullobj_sorted_pure_ce_ckpt3668": sorted_ledger["gt_match_rows"],
            },
            contexts_by_role={
                "fullobj_random_pure_ce_ckpt3668": _context(
                    checkpoint_role="fullobj_random_pure_ce_ckpt3668",
                    checkpoint_fingerprint="random-sha",
                ),
                "fullobj_sorted_pure_ce_ckpt3668": _context(
                    checkpoint_role="fullobj_sorted_pure_ce_ckpt3668",
                    checkpoint_fingerprint="sorted-sha",
                ),
            },
        )


def test_fn_universe_rejects_duplicate_missing_and_extra_ledger_keys() -> None:
    gt_rows = [_gt(0, "chair", [0, 0, 10, 10]), _gt(1, "table", [20, 0, 30, 10])]
    random_ledger = match_fn_cases_for_image(gt_rows, [], split="val", image_id=31)
    sorted_ledger = match_fn_cases_for_image(gt_rows, [], split="val", image_id=31)

    duplicate_ledger = deepcopy(random_ledger)
    duplicate_ledger["gt_match_rows"].append(deepcopy(duplicate_ledger["gt_match_rows"][0]))
    with pytest.raises(ValueError, match="duplicate.*gt_object_key"):
        build_fn_case_universe(
            gt_rows,
            random_match_ledger=duplicate_ledger,
            sorted_match_ledger=sorted_ledger,
            split="val",
            image_id=31,
        )

    missing_ledger = deepcopy(random_ledger)
    missing_ledger["gt_match_rows"] = missing_ledger["gt_match_rows"][:-1]
    with pytest.raises(ValueError, match="missing.*val:31:1"):
        build_fn_case_universe(
            gt_rows,
            random_match_ledger=missing_ledger,
            sorted_match_ledger=sorted_ledger,
            split="val",
            image_id=31,
        )

    extra_ledger = deepcopy(random_ledger)
    extra_row = deepcopy(extra_ledger["gt_match_rows"][0])
    extra_row["gt_object_key"] = "val:31:99"
    extra_ledger["gt_match_rows"].append(extra_row)
    with pytest.raises(ValueError, match="extra.*val:31:99"):
        build_fn_case_universe(
            gt_rows,
            random_match_ledger=extra_ledger,
            sorted_match_ledger=sorted_ledger,
            split="val",
            image_id=31,
        )


def test_universe_and_replay_rows_are_stably_sorted_by_gt_key_and_role() -> None:
    gt_rows = [
        _gt(2, "lamp", [40, 0, 50, 10]),
        _gt(0, "chair", [0, 0, 10, 10]),
        _gt(1, "table", [20, 0, 30, 10]),
    ]
    canonical_gt_rows = sorted(gt_rows, key=lambda row: int(row["gt_idx"]))
    random_ledger = match_fn_cases_for_image(
        canonical_gt_rows,
        [_pred(0, "chair", [0, 0, 10, 10])],
        split="val",
        image_id=41,
    )
    sorted_ledger = match_fn_cases_for_image(
        canonical_gt_rows,
        [],
        split="val",
        image_id=41,
    )

    universe_rows = build_fn_case_universe(
        gt_rows,
        random_match_ledger=random_ledger,
        sorted_match_ledger=sorted_ledger,
        split="val",
        image_id=41,
        sampled_gt_object_keys={"val:41:0", "val:41:1", "val:41:2"},
    )
    rows = build_replayable_fn_cases(
        list(reversed(universe_rows)),
        match_ledgers_by_role={
            "fullobj_sorted_pure_ce_ckpt3668": sorted_ledger,
            "fullobj_random_pure_ce_ckpt3668": random_ledger,
        },
        contexts_by_role={
            "fullobj_sorted_pure_ce_ckpt3668": _context(
                checkpoint_role="fullobj_sorted_pure_ce_ckpt3668",
                checkpoint_fingerprint="sorted-sha",
            ),
            "fullobj_random_pure_ce_ckpt3668": _context(
                checkpoint_role="fullobj_random_pure_ce_ckpt3668",
                checkpoint_fingerprint="random-sha",
            ),
        },
    )

    assert [row["gt_object_key"] for row in universe_rows] == [
        "val:41:0",
        "val:41:1",
        "val:41:2",
    ]
    assert [(row["fn_gt_idx"], row["checkpoint_role"]) for row in rows] == [
        (0, "fullobj_sorted_pure_ce_ckpt3668"),
        (1, "fullobj_random_pure_ce_ckpt3668"),
        (1, "fullobj_sorted_pure_ce_ckpt3668"),
        (2, "fullobj_random_pure_ce_ckpt3668"),
        (2, "fullobj_sorted_pure_ce_ckpt3668"),
    ]


def test_fn_matching_import_does_not_pull_heavy_modules() -> None:
    script = """
import json
import sys
import src.analysis.sorted_random_no_newline_phenotype.fn_matching
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


def _gt(
    gt_idx: int,
    desc: str,
    bbox_xyxy: list[int],
    *,
    gt_sorted_rank: int | None = None,
    same_desc_gt_count: int | None = None,
    object_count: int | None = None,
) -> dict[str, object]:
    return {
        "gt_idx": gt_idx,
        "desc": desc,
        "bbox_xyxy": bbox_xyxy,
        "gt_sorted_rank": gt_sorted_rank,
        "same_desc_gt_count": same_desc_gt_count,
        "object_count": object_count,
    }


def _pred(pred_idx: int, desc: str, bbox_xyxy: list[int]) -> dict[str, object]:
    return {"pred_idx": pred_idx, "desc": desc, "bbox_xyxy": bbox_xyxy}


def _shared_fn_inputs() -> tuple[
    list[dict[str, object]],
    dict[str, object],
    dict[str, object],
    list[dict[str, object]],
]:
    gt_rows = [
        _gt(
            0,
            "chair",
            [0, 0, 10, 10],
            gt_sorted_rank=1,
            same_desc_gt_count=1,
            object_count=1,
        )
    ]
    random_ledger = match_fn_cases_for_image(
        gt_rows,
        [],
        split="val",
        image_id="shared",
    )
    sorted_ledger = match_fn_cases_for_image(
        gt_rows,
        [],
        split="val",
        image_id="shared",
    )
    universe_rows = build_fn_case_universe(
        gt_rows,
        random_match_ledger=random_ledger,
        sorted_match_ledger=sorted_ledger,
        split="val",
        image_id="shared",
        sampled_gt_object_keys={"val:shared:0"},
    )
    return gt_rows, random_ledger, sorted_ledger, universe_rows


def _context(
    *,
    checkpoint_role: str,
    checkpoint_fingerprint: str,
) -> dict[str, object]:
    return {
        "checkpoint_role": checkpoint_role,
        "split": "val",
        "image_id": "img-9",
        "source_line_idx": 91,
        "image_path": "images/val/img-9.jpg",
        "width": 1024,
        "height": 768,
        "coord_mode": "xyxy",
        "bbox_surface": "coord_token",
        "data_root": "/tmp/data-root",
        "jsonl_sha256": "0" * 64,
        "checkpoint_fingerprint": checkpoint_fingerprint,
        "decode_policy": {"temperature": 0.0, "top_p": 1.0},
        "template_contract": {"row_separator": "none"},
        "invalid_pred_count": 0,
    }
