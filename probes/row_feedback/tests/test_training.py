from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest
import torch

from probes.row_feedback import training


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
    sigmoid = lambda value: 1.0 / (1.0 + math.exp(-value))
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
