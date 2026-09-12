"""Strict replay keeps native DDP buckets stable across process restarts."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.checkpoint import checkpoint

from src.qwen.special_token_embeddings import (
    SelectedDeltaInputEmbedding,
    SelectedDeltaOutputHead,
    SpecialTokenSelection,
)
from src.runtime.seeding import TrainingSeedReceipt
from src.training import session


def _policy(mode: str) -> dict:
    return TrainingSeedReceipt(
        17, "pipeline_entry", mode, False
    ).to_policy_identity_dict()


@pytest.mark.parametrize("mode", ["legacy", "strict_cuda_replay_v1"])
def test_admitted_accelerator_construction_matches_recorded_ddp_policy(
    monkeypatch, mode
):
    captured = {}
    monkeypatch.setattr(
        session, "Accelerator", lambda **kwargs: captured.update(kwargs) or captured
    )
    config = SimpleNamespace(
        training=SimpleNamespace(precision="bf16"),
        runtime=SimpleNamespace(determinism=SimpleNamespace(mode=mode)),
    )
    session.open_admitted_accelerator(
        plan=SimpleNamespace(resolved_config=SimpleNamespace(config=config)),
        writer=None,
        lifecycle={},
    )
    assert captured["gradient_accumulation_steps"] == 1
    assert captured["mixed_precision"] == "bf16"
    if mode == "legacy":
        assert "kwargs_handlers" not in captured
        assert "distributed_gradient_reduction" not in _policy(mode)
    else:
        (handler,) = captured["kwargs_handlers"]
        assert handler.find_unused_parameters is True
        assert handler.static_graph is False
        assert _policy(mode)["distributed_gradient_reduction"] == {
            "schema_version": 1,
            "bucket_policy": "fixed_initial",
            "ddp_kwargs": {"find_unused_parameters": True, "static_graph": False},
        }


def test_exact_resume_identity_binds_distributed_gradient_reduction():
    common = dict(
        packing={}, input_provider={}, attention={}, profile_sync={}, eval_reduction={}
    )
    fixed = _policy("strict_cuda_replay_v1")["distributed_gradient_reduction"]
    actual = session._exact_resume_policy_payload(
        **common, distributed_gradient_reduction=fixed
    )
    assert actual["distributed_gradient_reduction"] == fixed
    assert actual != {
        key: value
        for key, value in actual.items()
        if key != "distributed_gradient_reduction"
    }


class _NegativeExampleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        selection = SpecialTokenSelection(
            token_strings=("selected-a", "selected-b"), token_ids=(6, 7)
        )
        delta = torch.nn.Parameter(torch.zeros(2, 8))
        embedding = torch.nn.Embedding(8, 8)
        head = torch.nn.Linear(8, 8, bias=False)
        embedding.weight.requires_grad_(False)
        head.weight.requires_grad_(False)
        self.embedding = SelectedDeltaInputEmbedding(embedding, selection, delta)
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(8, 8), torch.nn.Tanh(), torch.nn.Linear(8, 8)
        )
        self.head = SelectedDeltaOutputHead(head, selection, delta)

    def forward(self, token_ids):
        hidden = checkpoint(self.layers, self.embedding(token_ids), use_reentrant=False)
        return self.head(hidden)


def _negative_ddp_worker(rank: int, rendezvous: str, receipt: str):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2
    )
    try:
        torch.manual_seed(17)
        model = _NegativeExampleModel()
        reference = deepcopy(model)
        kwargs = _policy("strict_cuda_replay_v1")["distributed_gradient_reduction"][
            "ddp_kwargs"
        ]
        ddp = torch.nn.parallel.DistributedDataParallel(
            model, bucket_cap_mb=0.0001, **kwargs
        )
        token_ids = torch.tensor(
            [[0, 1]]
        )  # No selected input tokens, as in negative examples.
        targets = torch.tensor([2 + rank, 2 + rank])
        records = []
        for step in range(1, 4):
            for micro in range(6):

                def backward(owner):
                    # The model returns logits; the loss lives outside DDP, like SFT.
                    (
                        torch.nn.functional.cross_entropy(
                            owner(token_ids).view(-1, 8), targets
                        )
                        / 6
                    ).backward()

                if micro < 5:
                    with ddp.no_sync():
                        backward(ddp)
                else:
                    backward(ddp)
                backward(reference)
            expected = reference.embedding.shared_embed_delta.grad
            dist.all_reduce(expected)
            expected /= 2
            observed = model.embedding.shared_embed_delta.grad
            assert torch.count_nonzero(observed) > 0
            torch.testing.assert_close(observed, expected, rtol=0, atol=0)
            info = ddp._get_ddp_logging_data()
            assert info.get("has_rebuilt_buckets", 0) == 0
            records.append(
                {
                    "step": step,
                    "bucket_sizes": info["bucket_sizes"],
                    "has_rebuilt_buckets": info.get("has_rebuilt_buckets", 0),
                    "delta_grad_norm": float(observed.norm()),
                }
            )
            ddp.zero_grad(set_to_none=True)
            reference.zero_grad(set_to_none=True)
        assert len({row["bucket_sizes"] for row in records}) == 1
        if rank == 0:
            Path(receipt).write_text(
                json.dumps(
                    {"cuda_initialized": torch.cuda.is_initialized(), "steps": records}
                )
            )
    finally:
        dist.destroy_process_group()


def test_fixed_buckets_preserve_negative_example_shared_delta_gradients(tmp_path):
    receipt = tmp_path / "buckets.json"
    mp.spawn(
        _negative_ddp_worker,
        args=(str(tmp_path / "rendezvous"), str(receipt)),
        nprocs=2,
        join=True,
    )
    assert json.loads(receipt.read_text())["cuda_initialized"] is False
