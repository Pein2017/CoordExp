from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from src.analysis.hard_ce_coord_logit_locality import (
    SLOT_NAMES,
    attribute_lane_c_coord_distribution,
    build_lane_c_shard_plan,
    build_lane_c_prefix_match_state,
    compact_slots_from_text,
    compute_embedding_geometry_metrics,
    compute_probability_locality_metrics,
    cut_complete_compact_rows,
    distribution_metrics_from_logits,
    lane_c_record_selected,
    lane_c_shard_label,
    merge_lane_c_shards,
    normalize_lane_c_shard,
    prediction_position_for_label_position,
    resolve_coord_token_ids,
    summarize_lane_c_per_case_rows,
    select_lane_c_intended_target_gt_idx,
    select_lane_c_generated_intended_target,
    _prefix_text_at_depth,
)


class _FakeCoordTokenizer:
    unk_token_id = 0

    def __init__(self, *, include_wildcard_collision: bool = False) -> None:
        self._token_to_id = {
            f"<|coord_{index}|>": 1000 + index for index in range(1000)
        }
        self._token_to_id["<|coord_*|>"] = (
            1000 if include_wildcard_collision else 999_999
        )

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            return self._token_to_id.get(tokens, self.unk_token_id)
        return [self._token_to_id.get(token, self.unk_token_id) for token in tokens]


def test_resolve_coord_token_ids_returns_only_exact_1000_bins() -> None:
    vocab = resolve_coord_token_ids(_FakeCoordTokenizer())

    assert len(vocab.coord_token_ids) == 1000
    assert vocab.coord_token_ids[0] == 1000
    assert vocab.coord_token_ids[-1] == 1999
    assert vocab.wildcard_token_id == 999_999
    assert vocab.wildcard_token_id not in set(vocab.coord_token_ids)


def test_resolve_coord_token_ids_rejects_wildcard_collision() -> None:
    with pytest.raises(ValueError, match="wildcard"):
        resolve_coord_token_ids(
            _FakeCoordTokenizer(include_wildcard_collision=True)
        )


def test_prediction_position_for_label_position_uses_causal_shift() -> None:
    assert prediction_position_for_label_position(7) == 6
    with pytest.raises(ValueError, match="position 0"):
        prediction_position_for_label_position(0)


def test_probability_metrics_distinguish_full_mass_from_conditional_shape() -> None:
    logits = torch.full((1010,), -8.0)
    coord_ids = list(range(2, 1002))
    logits[0] = 6.0
    logits[coord_ids[0]] = 1.0
    logits[coord_ids[1]] = 5.0
    logits[coord_ids[2]] = 2.0
    logits[coord_ids[3]] = 0.0

    row = distribution_metrics_from_logits(
        logits=logits,
        coord_token_ids=coord_ids,
        gt_bin=1,
        radii=(0, 1, 2),
        top_k=3,
    )

    assert row.coord_vocab_mass < 0.5
    assert row.p_gt_cond > 0.9
    assert row.rank_gt == 1
    assert row.top1_bin == 1
    assert row.mass_by_radius["mass_at_0"] == pytest.approx(row.p_gt_cond)
    assert sum(item["prob_cond"] for item in row.top_bins) <= 1.0


def test_probability_metrics_classify_delta_uniform_and_bimodal_shapes() -> None:
    delta = np.zeros(1000, dtype=np.float64)
    delta[512] = 1.0
    delta_metrics = compute_probability_locality_metrics(delta, gt_bin=512)
    assert delta_metrics.shape_label == "sharp_delta"
    assert delta_metrics.local_maxima_count == 1
    assert delta_metrics.mass_by_radius["mass_at_1"] == pytest.approx(1.0)

    uniform = np.full(1000, 0.001, dtype=np.float64)
    uniform_metrics = compute_probability_locality_metrics(uniform, gt_bin=512)
    assert uniform_metrics.shape_label == "diffuse"
    assert uniform_metrics.effective_support > 900

    bimodal = np.zeros(1000, dtype=np.float64)
    bimodal[512] = 0.55
    bimodal[800] = 0.45
    bimodal_metrics = compute_probability_locality_metrics(bimodal, gt_bin=512)
    assert bimodal_metrics.shape_label == "bimodal_target_other"
    assert bimodal_metrics.secondary_peak_distance == 288


def test_embedding_metrics_detect_ordered_numeric_manifold() -> None:
    ordered = np.stack(
        [np.arange(32, dtype=np.float64), np.zeros(32, dtype=np.float64)],
        axis=1,
    )

    metrics = compute_embedding_geometry_metrics(
        ordered,
        neighbor_k=4,
        radii=(1, 2, 4),
    )

    assert metrics.pearson_distance_numeric > 0.99
    assert metrics.spearman_distance_numeric > 0.99
    assert metrics.knn_radius_recall["radius_4"] == pytest.approx(1.0)
    assert metrics.numeric_neighbor_rank_mean["offset_1"] <= 2.0
    assert math.isfinite(metrics.bandedness_ratio)


def test_embedding_metrics_drop_when_numeric_order_is_scrambled() -> None:
    rng = np.random.default_rng(123)
    shuffled = rng.normal(size=(32, 8))

    metrics = compute_embedding_geometry_metrics(
        shuffled,
        neighbor_k=4,
        radii=(1, 2, 4),
    )

    assert metrics.knn_radius_recall["radius_1"] < 0.35
    assert metrics.pearson_distance_numeric < 0.35


def test_cut_complete_compact_rows_strips_im_end_and_rejects_partial_tail() -> None:
    text = (
        "<|object_ref_start|>cat<|box_start|>"
        "<|coord_1|><|coord_2|><|coord_3|><|coord_4|>\n"
        "<|object_ref_start|>dog<|box_start|>"
        "<|coord_5|><|coord_6|><|coord_7|><|coord_8|><|im_end|>\n"
        "<|object_ref_start|>broken<|box_start|><|coord_9|>"
    )

    assert cut_complete_compact_rows(text, max_rows=1) == (
        "<|object_ref_start|>cat<|box_start|>"
        "<|coord_1|><|coord_2|><|coord_3|><|coord_4|>"
    )
    assert cut_complete_compact_rows(text) == (
        "<|object_ref_start|>cat<|box_start|>"
        "<|coord_1|><|coord_2|><|coord_3|><|coord_4|>\n"
        "<|object_ref_start|>dog<|box_start|>"
        "<|coord_5|><|coord_6|><|coord_7|><|coord_8|>"
    )


def test_prefix_text_at_depth_enforces_exact_prefix_row_count() -> None:
    text = (
        "<|object_ref_start|>cat<|box_start|>"
        "<|coord_1|><|coord_2|><|coord_3|><|coord_4|>\n"
        "<|object_ref_start|>ambiguous<|box_start|>"
        "<|coord_9|><|coord_9|><|coord_9|><|coord_9|>\n"
        "<|object_ref_start|>dog<|box_start|>"
        "<|coord_5|><|coord_6|><|coord_7|><|coord_8|>\n"
    )

    assert _prefix_text_at_depth(text, depth=1) == (
        "<|object_ref_start|>cat<|box_start|>"
        "<|coord_1|><|coord_2|><|coord_3|><|coord_4|>"
    )
    assert len(compact_slots_from_text(_prefix_text_at_depth(text, depth=2))) == 8


def test_compact_slots_from_text_preserves_xyxy_slot_order() -> None:
    rows = compact_slots_from_text(
        "<|object_ref_start|>cat<|box_start|>"
        "<|coord_10|><|coord_20|><|coord_30|><|coord_40|>"
    )

    assert [row.slot for row in rows] == list(SLOT_NAMES)
    assert [row.gt_bin for row in rows] == [10, 20, 30, 40]
    assert {row.object_order_index for row in rows} == {0}
    assert rows[0].desc == "cat"


def test_lane_c_fp_prefix_does_not_advance_intended_target_by_depth() -> None:
    state = build_lane_c_prefix_match_state(
        [
            {
                "raw_pred_idx": 0,
                "row_label": "unmatched_fp",
                "raw_match_label": "unmatched_fp",
                "guarded_match_label": "unmatched_fp",
                "matched_gt_idx": None,
                "guarded_matched_gt_idx": None,
                "suppressed_by_guard": False,
            }
        ],
        depth=1,
        gt_count=2,
    )

    selected = select_lane_c_intended_target_gt_idx([0, 1], state)

    assert state.prefix_quality == "fp_prefix"
    assert state.fp_prefix_object_indices == (0,)
    assert state.matched_prefix_gt_indices == ()
    assert state.remaining_gt_indices == (0, 1)
    assert selected.intended_target_gt_idx == 0
    assert selected.target_selection_rule == "first_remaining_teacher_order_guarded"


def test_lane_c_duplicate_suppressed_raw_tp_does_not_consume_guarded_gt() -> None:
    state = build_lane_c_prefix_match_state(
        [
            {
                "raw_pred_idx": 0,
                "row_label": "duplicate_suppressed",
                "raw_match_label": "tp_like",
                "guarded_match_label": None,
                "matched_gt_idx": 0,
                "guarded_matched_gt_idx": None,
                "suppressed_by_guard": True,
            }
        ],
        depth=1,
        gt_count=2,
    )

    selected = select_lane_c_intended_target_gt_idx([0, 1], state)

    assert state.prefix_quality == "duplicate_prefix"
    assert state.duplicate_prefix_object_indices == (0,)
    assert state.matched_prefix_gt_indices == ()
    assert state.remaining_gt_indices == (0, 1)
    assert selected.intended_target_gt_idx == 0


def test_lane_c_best_other_rank_uses_full_distribution_not_top_k() -> None:
    probs = np.zeros(1000, dtype=np.float64)
    probs[100] = 0.50
    for idx in range(20):
        probs[idx] = 0.01
    probs[900] = 0.005

    attr = attribute_lane_c_coord_distribution(
        probs,
        target_bin=100,
        gt_bins_by_index={0: {"x1": 100}, 1: {"x1": 900}},
        prefix_bins_by_label={},
        slot="x1",
        radii=(4, 8),
    )

    assert attr.target_rank == 1
    assert attr.best_other_gt_idx == 1
    assert attr.best_other_gt_bin == 900
    assert attr.best_other_gt_rank == 22
    assert attr.best_other_gt_rank > 12
    assert attr.target_margin_vs_best_other == pytest.approx(
        attr.mass_by_radius["mass_at_radius_4"] - probs[900] / probs.sum()
    )


def test_lane_c_attribution_labels_same_desc_and_previous_generated_peaks() -> None:
    same_desc_probs = np.zeros(1000, dtype=np.float64)
    same_desc_probs[303] = 0.80
    same_desc_probs[100] = 0.20

    same_desc_attr = attribute_lane_c_coord_distribution(
        same_desc_probs,
        target_bin=100,
        gt_bins_by_index={
            0: {"x1": 100},
            1: {"x1": 304, "same_desc_competitor": True},
        },
        prefix_bins_by_label={},
        slot="x1",
        radii=(4, 8),
    )

    previous_probs = np.zeros(1000, dtype=np.float64)
    previous_probs[701] = 0.80
    previous_probs[100] = 0.20
    previous_attr = attribute_lane_c_coord_distribution(
        previous_probs,
        target_bin=100,
        gt_bins_by_index={0: {"x1": 100}},
        prefix_bins_by_label={"previous_generated_object": [{"x1": 700}]},
        slot="x1",
        radii=(4, 8),
    )

    assert same_desc_attr.top_peak_attribution == "same_desc_competitor_gt_object"
    assert previous_attr.top_peak_attribution == "previous_generated_object"


def test_lane_c_shard_selection_uses_source_line_idx_for_full_curves() -> None:
    shard_index, num_shards, label = normalize_lane_c_shard(
        shard_index=2,
        num_shards=3,
    )
    curve_rows = [
        {"source_line_idx": 5, "prefix_depth": depth, "slot": "x1"}
        for depth in range(4)
    ]
    other_curve_row = {"source_line_idx": 4, "prefix_depth": 0, "slot": "x1"}

    assert label == lane_c_shard_label(2, 3)
    assert shard_index == 2
    assert num_shards == 3
    assert all(
        lane_c_record_selected(row, shard_index=2, num_shards=3)
        for row in curve_rows
    )
    assert not lane_c_record_selected(
        other_curve_row,
        shard_index=2,
        num_shards=3,
    )


def test_lane_c_merge_rejects_missing_unexpected_and_malformed_shard_dirs(tmp_path: Path) -> None:
    root = tmp_path / "lane_c"
    shards = root / "shards"
    shards.mkdir(parents=True)
    (shards / "lane_c_shard_000-of-002").mkdir()

    with pytest.raises(ValueError, match="missing Lane-C shard dirs"):
        merge_lane_c_shards(root, expected_shards=2)

    (shards / "lane_c_shard_001-of-002").mkdir()
    (shards / "lane_c_shard_002-of-002").mkdir()
    with pytest.raises(ValueError, match="unexpected Lane-C shard dirs"):
        merge_lane_c_shards(root, expected_shards=2)

    (shards / "lane_c_shard_002-of-002").rmdir()
    (shards / "lane_c_shard_bad").mkdir()
    with pytest.raises(ValueError, match="malformed Lane-C shard dirs"):
        merge_lane_c_shards(root, expected_shards=2)


def test_lane_c_merge_rejects_duplicate_per_slot_keys(tmp_path: Path) -> None:
    root = tmp_path / "lane_c"
    for shard_index in range(2):
        shard_dir = root / "shards" / lane_c_shard_label(shard_index, 2)
        shard_dir.mkdir(parents=True)
        row = {
            "lane_c_merge_key": "row0:self_prefix:1:0:x1:0",
            "source_line_idx": 0,
            "prefix_mode": "self_prefix",
            "prefix_depth": 1,
            "intended_target_gt_idx": 0,
            "slot": "x1",
            "slot_index": 0,
        }
        (shard_dir / "per_slot.jsonl").write_text(
            __import__("json").dumps(row) + "\n",
            encoding="utf-8",
        )
        (shard_dir / "per_case.jsonl").write_text("", encoding="utf-8")
        (shard_dir / "summary.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate Lane-C per-slot merge key"):
        merge_lane_c_shards(root, expected_shards=2)


def test_lane_c_dry_run_shard_plan_has_unique_dirs_and_merge_without_model_load(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    output_root = tmp_path / "x1_basin_attribution"
    plan = build_lane_c_shard_plan(
        config_path=config_path,
        output_root=output_root,
        num_shards=8,
        python_executable="python",
    )

    shard_dirs = [item["shard_dir"] for item in plan["shards"]]
    commands = [item["command"] for item in plan["shards"]]

    assert len(shard_dirs) == 8
    assert len(set(shard_dirs)) == 8
    assert all(f"--shard-index {index}" in commands[index] for index in range(8))
    assert all("CUDA_VISIBLE_DEVICES" not in command for command in commands)
    assert "--merge-shards" in plan["merge_command"]
    assert "x1_basin_attribution" in plan["merge_command"]


def test_lane_c_generated_prefix_target_selection_uses_lane_a_state_not_ordinal_depth() -> None:
    selected = select_lane_c_generated_intended_target(
        teacher_order_gt_indices=(0, 1, 2),
        lane_a_rows=[
            {
                "raw_pred_idx": 0,
                "row_label": "unmatched_fp",
                "raw_match_label": "unmatched_fp",
                "guarded_match_label": "unmatched_fp",
                "matched_gt_idx": None,
                "guarded_matched_gt_idx": None,
                "suppressed_by_guard": False,
            }
        ],
        depth=1,
        gt_count=3,
    )

    assert selected.intended_target_gt_idx == 0
    assert selected.target_selection_rule == "first_remaining_teacher_order_guarded"


def test_lane_c_per_case_summary_preserves_separate_slot_metrics() -> None:
    rows = [
        {
            "case_id": "row0:self_prefix:0:7",
            "source_line_idx": 0,
            "prefix_mode": "self_prefix",
            "prefix_depth": 0,
            "intended_target_gt_idx": 7,
            "slot": slot,
            "top_peak_attribution": f"{slot}_peak",
            "target_rank": index + 1,
            "best_other_gt_rank": 10 + index,
            "target_margin_vs_best_other": 0.1 * index,
            "gt_top1": index == 0,
            "top1_distance": index,
            "mass_at_radius_4": 0.2 + index,
            "mass_at_radius_8": 0.3 + index,
        }
        for index, slot in enumerate(SLOT_NAMES)
    ]

    cases = summarize_lane_c_per_case_rows(rows)

    assert len(cases) == 1
    by_slot = cases[0]["slots"]
    assert tuple(by_slot) == SLOT_NAMES
    assert by_slot["x1"]["top_peak_attribution"] == "x1_peak"
    assert by_slot["y1"]["target_rank"] == 2
    assert by_slot["x2"]["mass_at_radius_4"] == pytest.approx(2.2)
    assert by_slot["y2"]["top1_distance"] == 3
    assert cases[0]["x1"]["top_peak_attribution"] == "x1_peak"
