import torch

from src.data_collators.dataset_metrics import build_dataset_metrics_collator


class _DummyTemplate:
    # Only used for attribute access in build_dataset_metrics_collator.
    tokenizer = None
    template_meta = None


def _base_collator(batch):
    # Produce minimal fields expected by build_dataset_metrics_collator.
    # For packed input, `batch` is a list of packs (list[list[dict]]).
    if len(batch) > 0 and isinstance(batch[0], (list, tuple)):
        bsz = len(batch)
        seqlen = 8
        labels = torch.full((bsz, seqlen), 1, dtype=torch.long)
        attention_mask = torch.ones((bsz, seqlen), dtype=torch.long)
        return {"labels": labels, "attention_mask": attention_mask, "input_ids": labels.clone()}
    # Non-packed input: list[dict]
    bsz = len(batch)
    seqlen = 8
    labels = torch.full((bsz, seqlen), 1, dtype=torch.long)
    attention_mask = torch.ones((bsz, seqlen), dtype=torch.long)
    return {"labels": labels, "attention_mask": attention_mask, "input_ids": labels.clone()}


def test_pack_num_samples_nonpacked_is_ones() -> None:
    collator = build_dataset_metrics_collator(_DummyTemplate(), _base_collator)
    batch = [{"dataset": "lvis"}, {"dataset": "lvis"}]
    out = collator(batch)
    assert "pack_num_samples" in out
    assert out["pack_num_samples"].tolist() == [1, 1]


def test_pack_num_samples_packed_counts_members() -> None:
    collator = build_dataset_metrics_collator(_DummyTemplate(), _base_collator)
    batch = [
        [{"dataset": "lvis"}, {"dataset": "lvis"}, {"dataset": "lvis"}],
        [{"dataset": "lvis"}],
    ]
    out = collator(batch)
    assert "pack_num_samples" in out
    assert out["pack_num_samples"].tolist() == [3, 1]

