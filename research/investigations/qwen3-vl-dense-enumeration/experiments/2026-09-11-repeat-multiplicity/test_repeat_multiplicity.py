from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


prepare = load("prepare")
runner = load("run_probe")


def fake_row(desc: int, coords: tuple[int, int, int, int]) -> list[int]:
    return [runner.ROW_START, desc, runner.REF_END, runner.BOX_START,
            *(151670 + value for value in coords), runner.ROW_END]


def test_prefix_family_holds_length_set_recency_and_final_row():
    rows = {label: {"token_ids": fake_row(desc, coords)} for label, desc, coords in (
        ("A", 7, (0, 0, 10, 10)), ("B", 7, (20, 20, 30, 30)), ("C", 8, (40, 40, 50, 50)))}
    families = []
    for cell, block in prepare.BLOCKS.items():
        labels, ids = prepare.make_prefix(rows, block)
        families.append((cell, labels, ids))
    assert {len(ids) for _, _, ids in families} == {81}
    assert {tuple(labels[-3:]) for _, labels, _ in families} == {("A", "B", "C")}
    assert {tuple(sorted(set(labels))) for _, labels, _ in families} == {("A", "B", "C")}
    assert {labels[-3:].index("A") for _, labels, _ in families} == {0}
    assert {labels[-3:].index("B") for _, labels, _ in families} == {1}
    assert {(labels.count("A"), labels.count("B")) for _, labels, _ in families} == {(5, 3), (4, 4), (3, 5)}


def test_free_behavior_preserves_invalid_and_raw_tail():
    a = fake_row(7, (0, 0, 10, 10))
    b = fake_row(7, (20, 20, 30, 30))
    invalid = fake_row(7, (30, 30, 20, 40))
    result = runner.free_behavior(a + b, a + invalid + [runner.ROW_START, 7], {"A": {"token_ids": a}, "B": {"token_ids": b}}, "length", image_width=1000, image_height=1000)
    assert result["complete_free_rows"] == 2
    assert result["exact_candidate_recurrences"] == {"A": 1, "B": 0}
    assert result["native_pixel_class_blind_strict_repeat_rows_against_prefix_and_prior_free"] == 1
    assert result["geometry_invalid_complete_free_rows"] == 1
    assert result["raw_unparsed_tail_ids"] == [runner.ROW_START, 7]
    assert result["horizon"] and not result["eos"]


def test_free_behavior_does_not_count_terminal_eos_as_unparsed():
    a = fake_row(7, (0, 0, 10, 10))
    result = runner.free_behavior(a, a + [runner.EOS], {"A": {"token_ids": a}}, "im_end", image_width=1000, image_height=1000)
    assert result["complete_free_rows"] == 1
    assert result["raw_unparsed_tail_ids"] == []
    assert result["eos"] and not result["horizon"]


def test_primary_strict_repeat_uses_native_pixel_projection():
    a = fake_row(7, (1, 100, 41, 200))
    b = fake_row(7, (3, 100, 41, 200))
    result = runner.free_behavior(a, b + [runner.EOS], {"A": {"token_ids": a}}, "im_end", image_width=832, image_height=1248)
    assert runner.iou((1, 100, 41, 200), (3, 100, 41, 200)) == pytest.approx(0.95)
    assert result["auxiliary_coord_bin_class_blind_strict_repeat_rows_against_prefix_and_prior_free"] == 0
    assert result["native_pixel_class_blind_strict_repeat_rows_against_prefix_and_prior_free"] == 1
    assert result["native_pixel_iou_gt_0_95_candidate_recurrences_same_description"] == {"A": 1}


def test_geometric_iou_is_not_owner_admission():
    assert prepare.row_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0
    assert prepare.row_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1


def test_runner_rejects_nonfinal_packet_before_sources():
    with pytest.raises(ValueError, match="not frozen"):
        runner.validate_packet({"schema": "repeat_multiplicity.v1", "status": "draft_pending_root_visual_admission"}, "9813")
