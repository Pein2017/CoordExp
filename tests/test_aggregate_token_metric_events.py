import pytest
import torch

from src.data_collators.token_types import TokenType
from src.metrics.aggregate_token_metrics import (
    build_next_token_batch,
    compute_token_type_acc,
)


def test_aggregate_coord_token_acc_emits_canonical_keys_and_legacy_aliases() -> None:
    logits = torch.zeros((1, 5, 10), dtype=torch.float32)
    labels = torch.tensor([[-100, 2, 3, 4, 5]], dtype=torch.long)
    token_types = torch.tensor(
        [
            [
                TokenType.IGNORE,
                TokenType.COORD,
                TokenType.COORD,
                TokenType.DESC,
                TokenType.FORMAT,
            ]
        ],
        dtype=torch.long,
    )

    logits[0, 0, 2] = 10.0
    logits[0, 1, 7] = 10.0
    logits[0, 1, 8] = 9.0
    logits[0, 1, 9] = 8.0
    logits[0, 1, 3] = 7.0
    logits[0, 1, 6] = 6.0
    logits[0, 2, 4] = 10.0
    logits[0, 3, 5] = 10.0

    batch = build_next_token_batch(
        logits=logits,
        labels=labels,
        token_types=token_types,
        log_top5=True,
    )
    assert batch is not None

    metrics = compute_token_type_acc(batch)

    assert metrics["coord_token_acc/full_vocab/top1"] == pytest.approx(0.5)
    assert metrics["coord_token_acc"] == pytest.approx(0.5)
    assert metrics["coord_token_acc/full_vocab/top5"] == pytest.approx(1.0)
    assert metrics["coord_token_acc_top5"] == pytest.approx(1.0)
    assert metrics["coord_token_acc"] == metrics["coord_token_acc/full_vocab/top1"]
    assert (
        metrics["coord_token_acc_top5"]
        == metrics["coord_token_acc/full_vocab/top5"]
    )
