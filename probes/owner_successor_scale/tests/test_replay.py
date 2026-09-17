from types import SimpleNamespace

import pytest
import torch

from probes.owner_successor_scale import replay
from src.qwen.native import padded_histories


class FakeModel:
    """A native-shaped model whose logits encode batch and causal position."""

    device = torch.device("cpu")

    def __call__(self, **inputs):
        ids = inputs["input_ids"]
        batch, width = ids.shape
        logits = torch.empty(batch, width, 5)
        for row in range(batch):
            for position in range(width):
                logits[row, position] = torch.tensor([row, position, ids[row, position], 7, 11])
        keep = inputs.get("logits_to_keep", 0)
        return SimpleNamespace(logits=logits[:, -keep:] if keep else logits)

    def get_rope_index(self, input_ids, image_grid_thw, attention_mask, **kwargs):
        batch, width = input_ids.shape
        return torch.zeros(3, batch, width, dtype=torch.long), torch.zeros(batch, 1, dtype=torch.long)


def entry(prompt, target, pixels, grid=(1, 1, 1)):
    return {
        "prompt_ids": prompt,
        "record": {"target_token_ids": target},
        "inputs": {
            "input_ids": torch.tensor([prompt]),
            "attention_mask": torch.ones(1, len(prompt), dtype=torch.long),
            "pixel_values": torch.tensor(pixels, dtype=torch.float32).reshape(-1, 1),
            "image_grid_thw": torch.tensor([grid]),
        },
    }


def fake_exact_history_inputs(model, native_inputs, histories, *, pad_token_id, logits_to_keep):
    assert pad_token_id == 0 and logits_to_keep == 4
    assert native_inputs["pixel_values"].shape[0] == 3
    assert native_inputs["image_grid_thw"].tolist() == [[1, 1, 1], [2, 1, 1]]
    ids, mask = padded_histories(histories, pad_token_id=pad_token_id)
    return {"input_ids": ids, "attention_mask": mask, "logits_to_keep": logits_to_keep}


def test_batched_aligned_logits_preserves_order_and_causal_rows_for_mixed_lengths(monkeypatch):
    monkeypatch.setattr(replay, "exact_history_inputs", fake_exact_history_inputs)
    entries = [entry([10, 11], [20, 21, 22], [1]), entry([30], [40], [2, 3], grid=(2, 1, 1))]
    observed = replay.batched_aligned_logits(FakeModel(), entries)
    assert [value.shape for value in observed] == [(3, 5), (1, 5)]
    # Histories are left-padded to width five, then compacted to the last four
    # logits.  The selected rows are causal predecessors, not same-position
    # token rows.
    assert observed[0][:, :3].tolist() == [[0.0, 1.0, 11.0], [0.0, 2.0, 20.0], [0.0, 3.0, 21.0]]
    assert observed[1][:, :3].tolist() == [[1.0, 3.0, 30.0]]


def test_batched_replay_rejects_mixed_native_fields_before_forward(monkeypatch):
    monkeypatch.setattr(replay, "exact_history_inputs", fake_exact_history_inputs)
    first, second = entry([1], [2], [1]), entry([3], [4], [2])
    second["inputs"]["unexpected"] = "not a tensor"
    with pytest.raises(ValueError, match="different native fields"):
        replay.batched_aligned_logits(FakeModel(), [first, second])


def test_batched_replay_rejects_empty_literal_target():
    bad = entry([1], [2], [1])
    bad["record"]["target_token_ids"] = []
    with pytest.raises(ValueError, match="literal nonempty"):
        replay.batched_aligned_logits(FakeModel(), [bad])
