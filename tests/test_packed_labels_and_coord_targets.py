from __future__ import annotations

import torch

from src.trainers.stage2_rollout_aligned import _build_labels_and_coord_targets_for_batch


def test_build_labels_and_coord_targets_packed_respects_segment_boundaries() -> None:
    # Two packed segments, each length 6.
    seg1 = [10, 11, 100, 5, 100, 6]
    seg2 = [20, 21, 100, 7, 100, 8]

    input_ids = torch.tensor([seg1 + seg2], dtype=torch.long)

    meta = [
        {
            "encoded_len": 6,
            "prompt_len": 2,
            "prefix_len": 0,
            "train_len": 4,
            "prompt_ids": [10, 11],
            "prefix_coord_pos": [],
            "prefix_coord_target_bins": [],
            "tail_ignore_pos": [],
        },
        {
            "encoded_len": 6,
            "prompt_len": 2,
            "prefix_len": 0,
            "train_len": 4,
            "prompt_ids": [20, 21],
            "prefix_coord_pos": [],
            "prefix_coord_target_bins": [],
            "tail_ignore_pos": [],
        },
    ]

    labels_masked, supervised_batch, supervised_pos, supervised_bin, supervised_is_prefix = (
        _build_labels_and_coord_targets_for_batch(
            input_ids=input_ids,
            meta=meta,
            coord_id_set={100},
            coord_id_to_bin={100: 0},
        )
    )

    assert labels_masked.shape == input_ids.shape

    # CE labels should be present only for non-coord tail tokens inside each segment.
    keep = torch.where(labels_masked[0] != -100)[0].tolist()
    assert keep == [3, 5, 9, 11]
    assert int(labels_masked[0, 3].item()) == 5
    assert int(labels_masked[0, 5].item()) == 6
    assert int(labels_masked[0, 9].item()) == 7
    assert int(labels_masked[0, 11].item()) == 8

    # Coord supervision positions should be the coord token positions (global indices).
    assert supervised_batch == [0, 0, 0, 0]
    assert supervised_pos == [2, 4, 8, 10]
    assert supervised_bin == [0, 0, 0, 0]
    assert supervised_is_prefix == [False, False, False, False]
