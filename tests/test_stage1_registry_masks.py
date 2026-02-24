import torch

from src.trainers.metrics.mixins import CoordSoftCEW1LossMixin
from src.trainers.teacher_forcing.stage1 import mask_stage1_coord_targets


def test_stage1_coord_mask_matches_registry_helper_on_packed_like_batch():
    mixin = CoordSoftCEW1LossMixin.__new__(CoordSoftCEW1LossMixin)

    labels = torch.tensor(
        [[101, 10, 202, 20, -100, 303, 10, 404, 505, 20, 606, -100]],
        dtype=torch.long,
    )
    coord_token_ids = [10, 20]

    out_mixin = mixin._mask_coord_targets(labels, coord_token_ids)
    out_registry = mask_stage1_coord_targets(labels, coord_token_ids)

    assert torch.equal(out_mixin, out_registry)
    assert int(out_mixin[0, 1].item()) == -100
    assert int(out_mixin[0, 3].item()) == -100
    assert int(out_mixin[0, 6].item()) == -100
    assert int(out_mixin[0, 9].item()) == -100
