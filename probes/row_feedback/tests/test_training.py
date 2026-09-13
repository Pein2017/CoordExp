from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from probes.row_feedback import training


def _distributed_records():
    bank_records = []
    for index, record_id in enumerate(("a", "b")):
        h = [7] * (2 + index) + [training.runtime.BOX_END]
        c = [11] * (1 + 2 * index) + [training.runtime.BOX_END]
        w = [13] * (3 - index) + [training.runtime.BOX_END]
        bank_records.append({
            "record_id": record_id,
            "prompt_token_ids": [5] * (7 + 3 * index),
            "literal_rows": {
                "h": {"token_ids": h},
                "c": {"token_ids": c},
                "w": {"token_ids": w},
            },
            "visible_history_token_ids": h + c,
        })
    normal_records = []
    for index in range(54):
        action = [17] * (1 + index % 7)
        if index % 3 == 0:
            action.append(training.runtime.BOX_END)
        positions = list(range(1 + index % 4))
        normal_records.append({
            "key": f"normal-{index:02d}",
            "prompt_token_ids": [19] * (3 + index % 5),
            "action_ids": action,
            "kl_positions": positions,
        })
    return {"records": bank_records}, {"records": normal_records}


def test_real_update_consumer_uses_token_mean_record_mean_and_weights(monkeypatch) -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    calls = iter([0.0, 2.0, 1.0, 3.0, *([2.0] * 54)])

    def fake_replay(_qwen, _inputs, *, target_ids, **_kwargs):
        value = next(calls)
        target = torch.tensor(list(target_ids), dtype=torch.long)
        logits = torch.zeros((target.numel(), 2), dtype=torch.float32)
        logits[:, 0] = value * parameter
        return {
            "logits": logits,
            "target_ids": target,
            "model_forwards": 1,
            "image_forwards": 1,
            "internal_slot_count": 0,
            "visible_target_tokens": target.numel(),
            "physical_token_count": target.numel(),
            "timing": {"wall_seconds": 0.0},
        }

    monkeypatch.setattr(training.runtime, "replay_visible", fake_replay)
    records = {
        key: {"record_id": key, "literal_rows": {
            "h": {"token_ids": [0]}, "c": {"token_ids": [0] if key == "a" else [0, 0, 0]},
            "w": {"token_ids": [0] if key == "a" else [0, 0, 0]},
        }, "visible_history_token_ids": [0],}
        for key in ("a", "b")
    }
    records["b"]["visible_history_token_ids"] = [0, 0]
    normal = [{"key": f"normal-{i}", "action_ids": [0], "kl_positions": [0]}
              for i in range(54)]
    teacher_log_probs = {row["key"]: torch.log_softmax(torch.tensor([[0.0, 0.0]]), dim=-1)
                         for row in normal}
    optimizer = torch.optim.AdamW([parameter], lr=1e-5, betas=(0.9, 0.999), eps=1e-8,
                                  weight_decay=0, foreach=False)
    result = training._run_update(
        qwen=object(), optimizer=optimizer, arm="F", package_ids=["a", "b"],
        bank_by_id=records, bank_entries={"a": {"inputs": {}, "prompt_ids": [0]},
                                          "b": {"inputs": {}, "prompt_ids": [0]}},
        normal_records=normal, normal_entries={row["key"]: {"inputs": {}, "prompt_ids": [0]}
                                               for row in normal},
        teacher_by_key=teacher_log_probs, named=[("parameter", parameter)],
    )
    c_mean = (math.log(2.0) + math.log(1.0 + math.exp(-1.0))) / 2
    w_mean = (math.log(1.0 + math.exp(-2.0)) + math.log(1.0 + math.exp(-3.0))) / 2
    p1 = math.exp(2.0) / (math.exp(2.0) + 1.0)
    p2 = 1.0 / (math.exp(2.0) + 1.0)
    kl = 0.5 * math.log(0.5 / p1) + 0.5 * math.log(0.5 / p2)
    expected_total = c_mean + w_mean + 100 * kl
    assert result["component_record_means"]["entry_c"] == pytest.approx(c_mean)
    assert result["component_record_means"]["post_completion_w"] == pytest.approx(w_mean)
    assert result["component_record_means"]["normal_kl"] == pytest.approx(kl, abs=1e-6)
    assert result["loss"] == pytest.approx(expected_total, abs=1e-5)
    def sigmoid(value):
        return 1.0 / (1.0 + math.exp(-value))

    expected_gradient = ((sigmoid(1.0) - 1.0) / 2
                         + (2.0 * (sigmoid(2.0) - 1.0)
                            + 3.0 * (sigmoid(3.0) - 1.0)) / 2
                         + 100.0 * (2.0 * (p1 - 0.5)))
    assert result["gradient_norm_before_clip"] == pytest.approx(abs(expected_gradient), rel=1e-6)
    # A global CE token mean would weight the three-token records three times;
    # this assertion gives the reduction check a concrete counterexample.
    assert c_mean != pytest.approx((math.log(2.0) + 3 * math.log(1.0 + math.exp(-1.0))) / 4)


def test_cyclic_pair_schedule_is_seeded_and_balanced() -> None:
    ids = [f"record-{index}" for index in range(16)]
    first = training.cyclic_pair_schedule(ids, seed=training.SEED, updates=64)
    second = training.cyclic_pair_schedule(ids, seed=training.SEED, updates=64)
    assert first == second
    assert first[:8] == first[8:16]
    assert first[:8] == first[16:24]
    exposure = {record_id: 0 for record_id in ids}
    for pair in first:
        assert len(pair) == 2 and pair[0] != pair[1]
        for record_id in pair:
            exposure[record_id] += 1
    assert set(exposure.values()) == {8}
    assert first != training.cyclic_pair_schedule(ids, seed=training.SEED + 1, updates=64)


def test_distributed_assignment_is_exact_deterministic_and_cost_balanced() -> None:
    bank, protection = _distributed_records()
    kwargs = {"schedule": [["a", "b"]], "bank": bank, "protection": protection,
              "world_size": 4}
    first, first_hash = training.distributed_assignment_binding(**kwargs)
    second, second_hash = training.distributed_assignment_binding(**kwargs)
    assert first == second
    assert first_hash == second_hash == training._canonical_hash(first)
    update = first["updates"][0]
    shards = update["shards"]
    items = [item for shard in shards for item in shard["items"]]
    assert len(items) == len({item["item_id"] for item in items}) == 58
    assert sorted(len(shard["items"]) for shard in shards) == [14, 14, 15, 15]
    assert sum(item["kind"] == "entry_c" for item in items) == 2
    assert sum(item["kind"] == "post_completion_w" for item in items) == 2
    normals = [item for item in items if item["kind"] == "normal_kl"]
    assert len(normals) == 54
    assert {item["loss_scale"] for item in items if item["kind"] == "entry_c"} == {0.5}
    assert {item["loss_scale"] for item in items if item["kind"] == "post_completion_w"} == {0.5}
    assert {item["loss_scale"] for item in normals} == {100.0 / 54}
    lpt_loads = [shard["structural_physical_tokens"] for shard in shards]
    serial_items = sorted(items, key=lambda item: item["serial_index"])
    round_robin_loads = [sum(item["structural_physical_tokens"]
                             for item in serial_items[rank::4]) for rank in range(4)]
    assert max(lpt_loads) - min(lpt_loads) < max(round_robin_loads) - min(round_robin_loads)


def test_sharded_backward_global_sum_matches_serial_update(monkeypatch) -> None:
    bank, protection = _distributed_records()
    manifest = training.build_distributed_assignment_manifest(
        schedule=[["a", "b"]], bank=bank, protection=protection, world_size=4,
    )
    assignment = manifest["updates"][0]
    bank_by_id = {record["record_id"]: record for record in bank["records"]}
    normal_by_key = {record["key"]: record for record in protection["records"]}
    bank_entries = {
        key: {"inputs": {}, "prompt_ids": record["prompt_token_ids"],
              "coefficient": 0.7 + 0.1 * index}
        for index, (key, record) in enumerate(bank_by_id.items())
    }
    normal_entries = {
        key: {"inputs": {}, "prompt_ids": record["prompt_token_ids"],
              "coefficient": 0.2 + 0.01 * index}
        for index, (key, record) in enumerate(normal_by_key.items())
    }
    teachers = {
        key: torch.zeros((len(record["kl_positions"]), 2), dtype=torch.float32)
        for key, record in normal_by_key.items()
    }

    def fake_replay_loss(*, qwen, entry, history, targets, **_kwargs):
        coefficient = entry["coefficient"] + 0.003 * len(history) + 0.002 * len(targets)
        loss = (qwen.parameter * coefficient - 0.25).square()
        physical = (len(entry["prompt_ids"]) + len(history)
                    + list(history).count(training.runtime.BOX_END) + len(targets)
                    + list(targets).count(training.runtime.BOX_END))
        replay = {
            "model_forwards": 1,
            "image_forwards": 1,
            "internal_slot_count": (list(history) + list(targets)).count(
                training.runtime.BOX_END),
            "visible_target_tokens": len(targets),
            "physical_token_count": physical,
            "timing": {"wall_seconds": 0.0},
        }
        return loss, replay, "toy"

    monkeypatch.setattr(training, "_replay_loss", fake_replay_loss)
    serial_parameter = torch.nn.Parameter(torch.tensor(1.0))
    serial_optimizer = torch.optim.AdamW(
        [serial_parameter], lr=1e-5, betas=(0.9, 0.999), eps=1e-8,
        weight_decay=0, foreach=False,
    )
    serial = training._run_update(
        qwen=SimpleNamespace(parameter=serial_parameter), optimizer=serial_optimizer,
        arm="F", package_ids=["a", "b"], bank_by_id=bank_by_id,
        bank_entries=bank_entries, normal_records=protection["records"],
        normal_entries=normal_entries, teacher_by_key=teachers,
        named=[("parameter", serial_parameter)],
    )

    local_gradients = []
    summaries = []
    for shard in assignment["shards"]:
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        summary = training._run_shard_backward(
            qwen=SimpleNamespace(parameter=parameter), arm="F", shard=shard,
            bank_by_id=bank_by_id, bank_entries=bank_entries,
            normal_by_key=normal_by_key, normal_entries=normal_entries,
            teacher_by_key=teachers,
        )
        assert parameter.grad is not None
        local_gradients.append(parameter.grad.detach().clone())
        summaries.append(summary)
    aggregate = training._aggregate_shard_summaries(summaries, assignment)
    distributed_parameter = torch.nn.Parameter(torch.tensor(1.0))
    distributed_parameter.grad = torch.stack(local_gradients).sum(dim=0)
    distributed_optimizer = torch.optim.AdamW(
        [distributed_parameter], lr=1e-5, betas=(0.9, 0.999), eps=1e-8,
        weight_decay=0, foreach=False,
    )
    step = training._apply_optimizer_step(
        distributed_optimizer, [("parameter", distributed_parameter)],
    )
    assert aggregate["counts"] == serial["counts"]
    assert aggregate["replays"] == serial["replays"] == 58
    assert aggregate["component_record_means"] == pytest.approx(
        serial["component_record_means"], rel=1e-6, abs=1e-7)
    assert aggregate["loss"] == pytest.approx(serial["loss"], rel=1e-6, abs=1e-7)
    assert step["gradient_norm_before_clip"] == pytest.approx(
        serial["gradient_norm_before_clip"], rel=1e-6)
    assert distributed_parameter.detach().item() == pytest.approx(
        serial_parameter.detach().item(), abs=1e-8)
    assert step["optimizer_steps"] == serial["optimizer_steps"] == 1
    assert distributed_optimizer.state[distributed_parameter]["step"] == 1


def test_gradient_collective_is_sum_without_world_size_average(monkeypatch) -> None:
    first = torch.nn.Parameter(torch.zeros(2))
    second = torch.nn.Parameter(torch.zeros((1, 1)))
    first.grad = torch.tensor([1.0, 2.0])
    second.grad = torch.tensor([[4.0]])
    calls = []

    def fake_all_reduce(tensor, *, op):
        calls.append((op, tensor.shape))
        tensor.add_(3.0)

    monkeypatch.setattr(training.dist, "all_reduce", fake_all_reduce)
    training._sum_parameter_gradients([("first", first), ("second", second)])
    assert calls == [(training.dist.ReduceOp.SUM, torch.Size([3]))]
    assert first.grad.tolist() == [4.0, 5.0]
    assert second.grad.tolist() == [[7.0]]


def test_synchronized_remote_failure_rejects_success_on_every_rank(monkeypatch) -> None:
    monkeypatch.setattr(training.dist, "get_world_size", lambda: 2)

    def fake_all_gather_object(rows, _local):
        rows[:] = [None, {"rank": 1, "phase": "backward", "error": "RuntimeError: failed"}]

    monkeypatch.setattr(training.dist, "all_gather_object", fake_all_gather_object)
    with pytest.raises(RuntimeError, match="distributed phase failed"):
        training._synchronize_phase_error(0, "backward", None)


def test_distributed_entry_selects_local_cuda_before_nccl_admission(monkeypatch, tmp_path) -> None:
    class ExpectedStop(RuntimeError):
        pass

    events = []
    monkeypatch.setattr(training.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        training, "_distributed_environment", lambda: (1, 2, 1, ["2", "3"]),
    )
    monkeypatch.setattr(
        training.torch.cuda, "set_device", lambda device: events.append(("device", str(device))),
    )

    def fake_init_process_group(*, backend, timeout):
        assert events == [("device", "cuda:1")]
        events.append(("init", backend, timeout))

    monkeypatch.setattr(training.dist, "init_process_group", fake_init_process_group)
    monkeypatch.setattr(training.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(training, "load_packet", lambda _path: ({}, {}, {}, {}))

    def stop_at_first_collective(_rank, phase, _error):
        assert phase == "packet_admission"
        assert events[0] == ("device", "cuda:1")
        assert events[1][0:2] == ("init", "nccl")
        raise ExpectedStop

    monkeypatch.setattr(training, "_synchronize_phase_error", stop_at_first_collective)
    with pytest.raises(ExpectedStop):
        training.run_distributed_training(
            packet_path=tmp_path / "packet.json",
            output=tmp_path / "output",
            arm="S",
            mode="cost",
        )


def test_distributed_topology_rejects_assignment_hash_drift() -> None:
    bank, protection = _distributed_records()
    manifest, digest = training.distributed_assignment_binding(
        schedule=[["a", "b"]], bank=bank, protection=protection, world_size=2,
    )
    del manifest
    packet = {
        "schedule": [["a", "b"]],
        "execution": {"distributed": {
            "schema": training.DIST_SCHEMA,
            "world_size": 2,
            "backend": "nccl",
            "gradient_reduction": "sum",
            "assignment_policy": training.DIST_ASSIGNMENT_POLICY,
            "assignment_sha256": digest,
        }},
    }
    training._validate_distributed_topology(
        packet, bank, protection, world_size=2,
    )
    packet["execution"]["distributed"]["assignment_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="assignment manifest binding changed"):
        training._validate_distributed_topology(
            packet, bank, protection, world_size=2,
        )


def test_packet_code_binding_rejects_stale_source() -> None:
    paths = {
        "runtime": Path(training.runtime.__file__).resolve(),
        "training": Path(training.__file__).resolve(),
        "data": Path(training.data.__file__).resolve(),
        "teacher": Path(training.teacher.__file__).resolve(),
    }
    packet = {
        "schema": training.SCHEMA,
        "status": "cost_only_pre_fit",
        "seed": training.SEED,
        "optimizer": copy.deepcopy(training.OPTIMIZER),
        "loss": copy.deepcopy(training.LOSS),
        "code_bindings": {
            name: {"path": str(path), "sha256": training.file_hash(path)}
            for name, path in paths.items()
        },
        "bank": {}, "protection": {}, "teacher_cache": {}, "schedule_source": {},
    }
    training.validate_packet(packet, validate_children=False)
    stale = copy.deepcopy(packet)
    stale["code_bindings"]["runtime"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="code binding drift: runtime"):
        training.validate_packet(stale, validate_children=False)


def test_loaded_anchor_verification_uses_path_and_full_descriptor(monkeypatch) -> None:
    import src.adapters.dora

    descriptor = {
        "root": "/adapter",
        "fingerprint": "frozen",
        "semantic_identity": {"base_model_name_or_path": "/base"},
    }
    packet = {"anchor_adapter": {"root": "/adapter", "fingerprint": "frozen"}}
    bank = {"source_roles": {"fit_anchor": {"adapter": descriptor}}}
    loaded_identity = {"model_identity": {"adapter": {
        "adapter_path": "/adapter", "base_model_path": "/base",
    }}}
    monkeypatch.setattr(src.adapters.dora, "inspect_dora_adapter_payload", lambda *_: descriptor)
    assert training._verify_loaded_anchor(packet, bank, loaded_identity) == descriptor
    monkeypatch.setattr(src.adapters.dora, "inspect_dora_adapter_payload",
                        lambda *_: {**descriptor, "fingerprint": "changed"})
    with pytest.raises(ValueError, match="loaded adapter payload differs"):
        training._verify_loaded_anchor(packet, bank, loaded_identity)


def test_loaded_runtime_receipt_uses_adapter_path_shape_without_fingerprint() -> None:
    receipt_path = Path(
        "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
        "2026-09-13-row-feedback-pilot/technical/runtime-f-v2/train-receipt.json"
    )
    if not receipt_path.is_file():
        pytest.skip("real technical receipt is not available in this checkout")
    loaded = json.loads(receipt_path.read_text())["loaded_identity"]
    adapter = loaded["model_identity"]["adapter"]
    assert "fingerprint" not in adapter
    assert {"adapter_path", "base_model_path", "adapter_payload_evidence"}.issubset(adapter)
