from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from probes.dora_owner_learning import owner_outcome, sample, train
from probes.dora_owner_learning.runtime import DEFAULT_CONFIG
from src.config.inference import load_research_infer_config
from src.qwen.native import NativeBatch


class TinyReplayModel(torch.nn.Module):
    def __init__(self, vocab=11):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(vocab, 3, generator=torch.Generator().manual_seed(23)))

    def get_rope_index(self, input_ids, image_grid_thw, video_grid_thw, attention_mask):
        positions = torch.arange(input_ids.shape[1]).expand(3, *input_ids.shape)
        return positions, None

    def forward(self, input_ids, logits_to_keep=0, **kwargs):
        hidden = torch.nn.functional.embedding(input_ids, self.weight)
        logits = hidden @ self.weight.T
        return SimpleNamespace(logits=logits[:, -logits_to_keep:] if logits_to_keep else logits)


def test_profile_keeps_original_resolved_scientific_config():
    resolved = load_research_infer_config(DEFAULT_CONFIG)
    assert resolved.fingerprint == "7f8448a8d8e62442bea1e9b1ffa45921a876510d6c9f12370949c6a418d1a4bb"
    assert resolved.config.model.dtype == "fp32"
    assert resolved.config.backend.hf.attn_implementation == "sdpa"


@pytest.mark.parametrize("scope", [owner_outcome.COORDINATE_SCOPE, owner_outcome.FULL_ACTION_SCOPE])
def test_owner_outcome_replay_keeps_credit_scope_and_gradients(scope):
    cell = {"prefix_token_ids": [7], "action_token_ids": [3, 151670, 151671, 151672, 151673, 4],
            "coordinate_action_positions": [1, 2, 3, 4]}
    model = TinyReplayModel(vocab=151674)
    reference = TinyReplayModel(vocab=151674)
    loss = owner_outcome.OwnerOutcomeScorer(model)(
        {"input_ids": torch.tensor([[1, 2]]), "image_grid_thw": torch.tensor([[1, 1, 1]])}, [1, 2], cell,
        credit=-0.75, scope=scope,
    )
    full_ids = torch.tensor([[1, 2, *cell["prefix_token_ids"], *cell["action_token_ids"]]])
    logits = reference(full_ids).logits[0, 2:-1].float()
    targets = torch.tensor(cell["action_token_ids"])
    chosen = logits.log_softmax(-1).gather(1, targets[:, None]).squeeze(1)
    selected = chosen[1:5] if scope == owner_outcome.COORDINATE_SCOPE else chosen
    expected = -selected.sum() * -0.75 * (8 / (16 * 4))
    loss.backward()
    expected.backward()
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(model.weight.grad, reference.weight.grad)
    cell["coordinate_action_positions"] = [0, 1, 2, 3]
    with pytest.raises(ValueError, match="coordinate mask"):
        owner_outcome.OwnerOutcomeScorer(model)({}, [1], cell, credit=1, scope=scope)
    cell["coordinate_action_positions"] = [True, 2, 3, 4]
    with pytest.raises(ValueError, match="coordinate mask"):
        owner_outcome.OwnerOutcomeScorer(model)({}, [1], cell, credit=1, scope=scope)


def test_raw_softmax_calls_native_generation_without_traces_or_default_filters():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.calls = []

        def generate(self, **kwargs):
            self.calls.append(kwargs)
            # Literal pad before EOS is a legitimate raw-softmax sampled ID.
            return torch.tensor([[1, 2, 0, 6, 9]])

    model = Model()
    batch = NativeBatch({"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.ones(1, 2, dtype=torch.long)}, ("row",))
    tokenizer = SimpleNamespace(decode=lambda ids, **_: ",".join(map(str, ids)))
    actual = sample.sample_one(model, tokenizer, batch, seed=17, max_new_tokens=4, eos_token_id=9, pad_token_id=0)
    assert actual == ([0, 6], "0,6", "im_end")
    kwargs = model.calls[0]
    assert kwargs["use_model_defaults"] is False
    assert kwargs["generation_config"].top_k == 0
    assert kwargs["temperature"] == kwargs["top_p"] == kwargs["repetition_penalty"] == 1
    assert not kwargs["output_scores"] and not kwargs["output_logits"]


def _ddp_worker(rank, world, init_file, output_dir):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world)
    try:
        for mode in ("ce", "rloo", "all_sync_sensitivity"):
            ddp = DDP(train.TrajectoryScorer(TinyReplayModel()), broadcast_buffers=False, init_sync=False)
            hook_state = {"calls": 0}

            def communication(state, bucket):
                state["calls"] += 1
                tensor = bucket.buffer()
                dist.all_reduce(tensor)
                tensor.div_(world)
                future = torch.futures.Future()
                future.set_result(tensor)
                return future

            ddp.register_comm_hook(hook_state, communication)
            actions = ([3 + rank, 4], [5, 6, 7], [4, 3], [7, 6, 5])
            advantages = (1.0, -1.0, 2.0, -2.0)
            for index, action_ids in enumerate(actions):
                train.backward_action(
                    ddp, model_inputs={"input_ids": torch.tensor([[1, 2 + rank]])},
                    grid=torch.tensor([[1, 1, 1]]), prompt_ids=[1, 2 + rank],
                    action={"action_token_ids": action_ids, "advantage": advantages[index]},
                    arm="rloo" if mode == "rloo" else "ce", image_count=2 if mode == "rloo" else 8,
                    sync_gradients=(mode == "all_sync_sensitivity" or index == len(actions) - 1),
                    world_size=world,
                )
            torch.save({"gradient": ddp.module.model.weight.grad, "communication_calls": hook_state["calls"]},
                       Path(output_dir) / f"{mode}-rank-{rank}.pt")
    finally:
        dist.destroy_process_group()


def test_two_process_ddp_matches_global_objectives_and_detects_extra_sync(tmp_path):
    mp.spawn(_ddp_worker, args=(2, str(tmp_path / "init"), str(tmp_path)), nprocs=2, join=True)
    for arm in ("ce", "rloo"):
        reference = train.TrajectoryScorer(TinyReplayModel())
        for rank in range(2):
            actions = ([3 + rank, 4], [5, 6, 7], [4, 3], [7, 6, 5])
            for action, advantage in zip(actions, (1.0, -1.0, 2.0, -2.0), strict=True):
                chosen = reference({"input_ids": torch.tensor([[1, 2 + rank]])},
                                   torch.tensor([[1, 1, 1]]), [1, 2 + rank], action)
                objective = -chosen.mean() / 8 if arm == "ce" else -advantage * chosen.sum() / (2 * 4)
                objective.backward()
        for rank in range(2):
            observed = torch.load(tmp_path / f"{arm}-rank-{rank}.pt", weights_only=True)
            assert observed["communication_calls"] == 1
            torch.testing.assert_close(observed["gradient"], reference.model.weight.grad)
            assert not torch.allclose(observed["gradient"], reference.model.weight.grad / 2)
    # The same measurement observes four collectives when the caller omits the
    # intermediate no_sync windows, so final-only synchronization has teeth.
    for rank in range(2):
        observed = torch.load(tmp_path / f"all_sync_sensitivity-rank-{rank}.pt", weights_only=True)
        assert observed["communication_calls"] == 4
