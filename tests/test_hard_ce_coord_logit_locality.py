from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from src.analysis.hard_ce_coord_logit_locality import (
    SLOT_NAMES,
    compact_slots_from_text,
    compute_embedding_geometry_metrics,
    compute_probability_locality_metrics,
    cut_complete_compact_rows,
    distribution_metrics_from_logits,
    prediction_position_for_label_position,
    resolve_coord_token_ids,
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
