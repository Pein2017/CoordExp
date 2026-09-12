"""Consumer-level falsification of the literal-row shared execution contract."""
from copy import deepcopy
from contextlib import nullcontext

import pytest
import torch

from probes.parallel_owner_research import training as t


def record(key="P", *, conditional=False):
    value = {"record_id": key, "example_id": "example", "image": {
        "row_id": "example", "row_index": 0, "image_id": "1", "image_path": "/not-read.jpg",
        "image_sha256": "bound", "observed_image_grid_thw": [1, 2, 2],
        "executed_media_sha256": "bound"}, "prompt_token_ids": [1, 2],
        "prompt_token_ids_sha256": t.old.digest_ids([1, 2]), "prefix_token_ids": [],
        "target_token_ids": [151646, 7, 151649]}
    if conditional:
        value.update(kl_positions=[0, 2], unknown_mask_policy="literal_positions_only")
    return value


def packet():
    return {"schema": t.SCHEMA, "normal_mask_policy": t.NORMAL_MASK,
        "margin_policy": t.MARGIN_POLICY, "normal_keys": ["n0", "n1", "n2"],
        "positive_records": [record()], "conditional_records": [record("W", conditional=True)],
        "weights": dict(positive=1, conditional_kl=10, normal_kl=100, margin=10),
        "denominators": dict(positive=1, conditional_kl=1, normal_kl=3, margin=3),
        "arms": {"A": {"steps": [[{"record_id": "P", "weight": 1}]]}},
        "runtime": {"world_sizes": [2, 8], **{k: 1000 for k in t.LIMITS}},
        "optimizer": dict(lr=1e-5, betas=[0.9, 0.999], eps=1e-8, weight_decay=0, foreach=False),
        "clip_gradient_norm": 1.0}


def test_explicit_packet_accepts_literal_empty_prefix():
    t.validate_contract(packet(), verify_images=False)


@pytest.mark.parametrize("mutation", [
    lambda p: p["denominators"].pop("positive"),
    lambda p: p["denominators"].update(normal_kl=56),
    lambda p: p["denominators"].update(conditional_kl=2),
    lambda p: p["weights"].update(positive=float("nan")),
    lambda p: p["positive_records"][0].update(kl_positions=[0]),
    lambda p: p["positive_records"][0].update(target_token_ids=[151646, 7]),
    lambda p: p["positive_records"][0].update(target_token_ids=[151646, 151649, 151646, 151649]),
    lambda p: p["conditional_records"][0].update(unknown_mask_policy="infer_from_parser"),
    lambda p: p["conditional_records"][0].update(kl_positions=[99]),
    lambda p: p["conditional_records"].clear(),
    lambda p: p["normal_keys"].append("n0"),
    lambda p: p["arms"]["A"]["steps"][0][0].update(record_id="missing"),
    lambda p: p["arms"]["A"]["steps"].append([]),
    lambda p: p["optimizer"].pop("foreach"),
    lambda p: p["runtime"].pop("max_rss_bytes"),
    lambda p: p.update(margin_policy="fixed_source_competitor"),
    lambda p: p.update(normal_mask_policy="all_tokens"),
])
def test_contract_fails_closed(mutation):
    value = packet()
    mutation(value)
    with pytest.raises((ValueError, AssertionError, RuntimeError)):
        t.validate_contract(value, verify_images=False)


def test_fixed_conditional_does_not_follow_history_schedule():
    value = packet()
    value["positive_records"].append(record("Q"))
    value["arms"] = {"fixed": {"steps": [[{"record_id": "P", "weight": 1}]] * 4},
                     "mixed": {"steps": [[{"record_id": name, "weight": 1}] for name in ("P", "Q", "P", "Q")]}}
    t.validate_contract(value, verify_images=False)
    assert value["conditional_records"][0]["record_id"] == "W"
    assert t.work_counts(value, "fixed", 2, 0) == t.work_counts(value, "mixed", 2, 0)


def test_work_counts_cover_uneven_shards_and_repeated_exposures():
    value = packet()
    value["arms"]["A"]["steps"][0].append({"record_id": "P", "weight": 0.5})
    counts = [t.work_counts(value, "A", 2, rank) for rank in (0, 1)]
    assert [c["training_replays"] for c in counts] == [5, 4]
    assert [c["synchronized_backwards"] for c in counts] == [1, 1]
    assert sum(c["final_reference_forwards"] for c in counts) == 3


def _loss_for_rows(parameter, rows, *, kind, scales):
    result = parameter.sum() * 0
    for features, targets, positions in rows:
        logits = features @ parameter
        reference = torch.log_softmax(torch.zeros(len(positions), 3), -1) if kind != "positive" else None
        margin = None
        if kind == "normal":
            margin = {"key": "toy", "eligible_positions": positions,
                      "original_kl_positions": positions, "target_ids": targets[positions].tolist(),
                      "source_margins": [0.4] * len(positions), "floors": [0.1] * len(positions)}
        loss, _ = t.item_loss(logits, targets, kind=kind, scales=scales,
            positions=positions, reference_logp=reference, margin=margin)
        result = result + loss
    return result


def _toy_world(world, *, mutated_missing_world=False):
    generator = torch.Generator().manual_seed(93)
    parameter = torch.randn(2, 3, generator=generator, requires_grad=True)
    positive = [(torch.randn(n, 2, generator=generator), torch.arange(n) % 3, []) for n in (2, 5)]
    conditional = [(torch.randn(3, 2, generator=generator), torch.tensor([0, 1, 2]), [0, 2])]
    normals = [(torch.randn(n, 2, generator=generator), torch.arange(n) % 3, list(range(0, n, 2)))
               for n in (2, 3, 4, 5, 6)]
    value = packet()
    value["denominators"].update(positive=2, normal_kl=5, margin=5)
    total = parameter.sum() * 0
    for rank in range(world):
        scales = {k: t.local_scale(value, k, world_size=world) for k in t.COMPONENTS}
        if mutated_missing_world:
            scales["normal_kl"] /= world
            scales["margin"] /= world
        total = total + (_loss_for_rows(parameter, positive, kind="positive", scales=scales)
                       + _loss_for_rows(parameter, conditional, kind="conditional", scales=scales)
                       + _loss_for_rows(parameter, normals[rank::world], kind="normal", scales=scales)) / world
    total.backward()
    gradient = parameter.grad.detach().clone()
    optimizer = torch.optim.AdamW([parameter], **value["optimizer"])
    optimizer.step()
    return total.detach(), gradient, parameter.detach()


@pytest.mark.parametrize("world", [2, 8])
def test_two_eight_rank_objective_gradient_and_update_equal_serial_uneven_lengths(world):
    expected = _toy_world(1)
    observed = _toy_world(world)
    for actual, reference in zip(observed, expected):
        torch.testing.assert_close(actual, reference, atol=3e-5, rtol=3e-6)
    # Sensitivity witness: omitting the DDP compensation really fails this consumer oracle.
    wrong = _toy_world(world, mutated_missing_world=True)
    with pytest.raises(AssertionError):
        torch.testing.assert_close(wrong[1], expected[1], atol=3e-5, rtol=3e-6)


def test_full_row_is_sum_not_token_mean_and_mask_does_not_enter_positive():
    logits = torch.zeros(5, 3, requires_grad=True)
    targets = torch.zeros(5, dtype=torch.long)
    loss, stats = t.item_loss(logits, targets, kind="positive", scales={"positive": 0.5},
                             positions=[], reference_logp=None, margin=None)
    torch.testing.assert_close(loss, torch.tensor(2.5 * math_log_three()))
    assert stats["route"]["token_count"] == 5
    with pytest.raises((ValueError, AssertionError, RuntimeError)):
        t.item_loss(logits, targets, kind="positive", scales={"positive": 1},
                    positions=[0], reference_logp=None, margin=None)


def math_log_three():
    import math
    return math.log(3)


def test_kl_uses_only_literal_positions_and_detaches_reference():
    logits = torch.randn(4, 3, requires_grad=True)
    reference = torch.log_softmax(torch.randn(2, 3), -1).requires_grad_()
    loss, _ = t.item_loss(logits, torch.tensor([0, 1, 2, 0]), kind="conditional",
        scales={"conditional_kl": 1}, positions=[0, 3], reference_logp=reference, margin=None)
    loss.backward()
    assert torch.count_nonzero(logits.grad[1:3]) == 0
    assert torch.count_nonzero(logits.grad[[0, 3]]) > 0
    assert reference.grad is None


def test_margin_tracks_new_full_vocab_competitor_not_source_runner_up():
    logits = torch.tensor([[0.6, 0.5, 1.0]], requires_grad=True)
    margin = {"key": "toy", "eligible_positions": [0], "original_kl_positions": [0],
              "target_ids": [0], "source_margins": [0.4], "floors": [0.1]}
    penalty, detail = t.margin_engine.worst_margin_penalty(logits, torch.tensor([0]), margin)
    torch.testing.assert_close(penalty, torch.tensor(0.5))
    assert detail["worst_best_other_id"] == 2
    penalty.backward()
    torch.testing.assert_close(logits.grad, torch.tensor([[-1., 0., 1.]]))


def test_qualification_zero_margin_matches_old_positive3_and_kl_scaling():
    generator = torch.Generator().manual_seed(3)
    logits = torch.randn(7, 5, generator=generator, requires_grad=True)
    targets = torch.arange(7) % 5
    loss, _ = t.item_loss(logits, targets, kind="positive", scales={"positive": 1 / 3},
        positions=[], reference_logp=None, margin=None)
    old_loss = t.old.positive_loss(logits, targets, targets.tolist())
    torch.testing.assert_close(loss, old_loss, rtol=0, atol=0)


def _ddp_uneven_worker(rank, rendezvous, output):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        torch.manual_seed(41)
        model = torch.nn.Linear(2, 3, bias=False)
        wrapper = DDP(model, broadcast_buffers=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, foreach=False)
        items = [(torch.tensor([[1., 2.]]), 0, 1.)]
        items += [(torch.tensor([[float(i), 1.]]), i % 3, 2 / 3) for i in range(3) if i % 2 == rank]
        for index, (features, target, scale) in enumerate(items):
            with nullcontext() if index == len(items) - 1 else wrapper.no_sync():
                loss = torch.nn.functional.cross_entropy(wrapper(features), torch.tensor([target])) * scale
                loss.backward()
        gradient = model.weight.grad.detach().clone()
        optimizer.step()
        torch.save({"gradient": gradient, "weight": model.weight.detach()}, output + f"-{rank}.pt")
    finally:
        dist.destroy_process_group()


def test_real_cpu_ddp_one_sync_handles_uneven_local_replay_counts(tmp_path):
    """Exercise real DDP rather than just averaging synthetic rank loss values."""
    torch.multiprocessing.spawn(_ddp_uneven_worker,
        args=(f"file://{tmp_path / 'rendezvous'}", str(tmp_path / "result")), nprocs=2, join=True)
    torch.manual_seed(41)
    model = torch.nn.Linear(2, 3, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, foreach=False)
    loss = torch.nn.functional.cross_entropy(model(torch.tensor([[1., 2.]])), torch.tensor([0]))
    for i in range(3):
        loss = loss + torch.nn.functional.cross_entropy(
            model(torch.tensor([[float(i), 1.]])), torch.tensor([i % 3])) / 3
    loss.backward()
    gradient = model.weight.grad.detach().clone()
    optimizer.step()
    for rank in (0, 1):
        result = torch.load(tmp_path / f"result-{rank}.pt", weights_only=True)
        torch.testing.assert_close(result["gradient"], gradient, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(result["weight"], model.weight, rtol=1e-6, atol=1e-7)
