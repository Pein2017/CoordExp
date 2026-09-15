from __future__ import annotations

import json
import queue
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from probes.training_set_completion import coco22_readback as readback22
from probes.training_set_completion import coco22_training as training22
from probes.training_set_completion import coco22_trial as trial22
from probes.training_set_completion import coco227_readback as previous_readback
from probes.training_set_completion import training


ACTIVE = [1 + (index * 7) % 23 for index in range(22)]


def _model() -> torch.nn.Module:
    model = torch.nn.Linear(3, 1, bias=True, dtype=torch.float64)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.2, -0.3, 0.5]], dtype=torch.float64))
        model.bias.copy_(torch.tensor([0.05], dtype=torch.float64))
    return model


def _terms(model: torch.nn.Module, index: int) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.tensor(
        [float(index + 1), float(index % 4 - 1), float(index % 3)],
        dtype=torch.float64,
    )
    value = model(x).squeeze()
    return (
        (value - (index + 1) / 9) ** 2,
        (value + (index % 5) / 7) ** 2,
    )


def _worker(rank: int, init_path: str, output: str) -> None:
    dist.init_process_group(
        "gloo", init_method=f"file://{init_path}", rank=rank, world_size=8,
    )
    try:
        model = _model()
        named = tuple(model.named_parameters())
        indices = training22.partition_route_indices(22, rank=rank, world_size=8)
        ce, hinge = zip(*[_terms(model, index) for index in indices], strict=True)
        loss = training22.objective_from_route_terms(
            ce, [ACTIVE[index] for index in indices], hinge,
            ce_reduction="sample_equal", global_ce_eligible_images=22,
            global_active_tokens=sum(ACTIVE),
        )
        loss.backward()
        training22.distributed.sum_gradients_(named)
        Path(output, f"rank-{rank}.json").write_text(
            json.dumps({
                "indices": indices,
                "gradients": {
                    name: parameter.grad.tolist() for name, parameter in named
                },
            }) + "\n"
        )
    finally:
        dist.destroy_process_group()


def test_uneven_eight_rank_sum_matches_serial_22_image_gradient(tmp_path: Path):
    if not dist.is_available():
        pytest.skip("torch.distributed unavailable")
    mp.spawn(
        _worker, args=(str(tmp_path / "init"), str(tmp_path)),
        nprocs=8, join=True,
    )
    rows = [
        json.loads((tmp_path / f"rank-{rank}.json").read_text())
        for rank in range(8)
    ]
    assert [len(row["indices"]) for row in rows] == list(training22.RANK_COUNTS)
    assert [index for row in rows for index in row["indices"]] == list(range(22))
    serial = _model()
    ce, hinge = zip(*[_terms(serial, index) for index in range(22)], strict=True)
    training22.objective_from_route_terms(
        ce, ACTIVE, hinge, ce_reduction="sample_equal",
        global_ce_eligible_images=22,
        global_active_tokens=sum(ACTIVE),
    ).backward()
    for row in rows:
        for name, parameter in serial.named_parameters():
            torch.testing.assert_close(
                torch.tensor(row["gradients"][name], dtype=torch.float64),
                parameter.grad,
            )


def test_22_image_reduction_rejects_local_mean_or_token_mean():
    values = [torch.tensor(float(i), requires_grad=True) for i in range(22)]
    hinges = [torch.tensor(float(i % 4), requires_grad=True) for i in range(22)]
    loss = training22.objective_from_route_terms(
        values, list(range(1, 23)), hinges,
        ce_reduction="sample_equal", global_ce_eligible_images=22,
        global_active_tokens=sum(range(1, 23)),
    )
    torch.testing.assert_close(
        loss, sum(values) / 22 + 0.01 * sum(hinges) / 22,
    )
    with pytest.raises(ValueError, match="sample-equal"):
        training22.objective_from_route_terms(
            values[:3], [1, 2, 3], hinges[:3],
            ce_reduction="sample_equal", global_ce_eligible_images=3,
            global_active_tokens=6,
        )
    with pytest.raises(ValueError, match="sample-equal"):
        training22.objective_from_route_terms(
            values, list(range(1, 23)), hinges,
            ce_reduction="global_active_token_equal",
            global_ce_eligible_images=22,
            global_active_tokens=sum(range(1, 23)),
        )


def test_new_torchrun_command_uses_all_eight_ranks_and_distinct_producer(tmp_path: Path):
    command = trial22.distributed_training_command(
        manifest_path=tmp_path / "manifest.json",
        output=tmp_path / "training",
    )
    assert command[:6] == [
        "python", "-m", "torch.distributed.run",
        "--standalone", "--nproc-per-node=8", "-m",
    ]
    assert command[6] == "probes.training_set_completion.coco22_training"


def test_native_scope_restores_legacy_producer_and_portable_wait():
    old_schema = previous_readback.SCHEMA
    old_count = previous_readback.IMAGE_COUNT
    old_validator = previous_readback._checkpoint_from_terminal
    with readback22._native_scope():
        assert previous_readback.SCHEMA == readback22.SCHEMA
        assert previous_readback.IMAGE_COUNT == 22
        assert previous_readback._checkpoint_from_terminal is readback22._checkpoint_from_terminal
    assert previous_readback.SCHEMA == old_schema
    assert previous_readback.IMAGE_COUNT == old_count
    assert previous_readback._checkpoint_from_terminal is old_validator
    events: queue.Queue[dict] = queue.Queue()
    process = subprocess.Popen([sys.executable, "-c", "raise SystemExit(7)"])
    readback22.start_waiter(process, events)
    event = events.get(timeout=10)
    assert event["pid"] == process.pid
    assert event["exit_code"] == 7
    assert event["wait_error"] is None


def test_cold_source0_receipt_requires_exact_adapter_identity():
    adapter = {"root": "/example/source", "fingerprint": "abc"}
    receipt = {
        "schema": f"{readback22.SCHEMA}.source0_terminal",
        "status": "cold_source_ready", "source_adapter": adapter,
    }
    readback22._checkpoint_from_terminal(receipt, step=0, adapter=adapter)
    with pytest.raises(ValueError, match="cold source0"):
        readback22._checkpoint_from_terminal(
            {**receipt, "source_adapter": {**adapter, "fingerprint": "wrong"}},
            step=0, adapter=adapter,
        )


@pytest.mark.parametrize(
    "change",
    ("iou_0_8_owner", "dropped_geometry", "different_class", "shifted_box"),
)
def test_batch_qualification_rejects_hidden_output_or_owner_changes(
    monkeypatch: pytest.MonkeyPatch, change: str,
):
    base = {
        "parse_status": "valid", "metric_bearing": True, "valid_count": 1,
        "raw": {
            "parser_dropped_total": 0, "geometry_invalid": 0,
            "malformed_non_geometry": 0, "drop_reasons": {},
        },
        "duplicate_pair_count": 0, "outside_coco80_count": 0,
        "stop": "im_end", "cap_debt": 0,
        "assignments": {
            "0.5": [("owner-a", True)], "0.8": [("owner-a", True)]
        },
        "valid": [{
            "description": "person", "coord_bins_1000": [100, 100, 200, 200]
        }],
    }
    changed = json.loads(json.dumps(base))
    if change == "iou_0_8_owner":
        changed["assignments"]["0.8"] = []
    elif change == "dropped_geometry":
        changed["raw"]["parser_dropped_total"] = 1
        changed["raw"]["geometry_invalid"] = 1
        changed["raw"]["drop_reasons"] = {"bbox_invalid": 1}
    elif change == "different_class":
        changed["valid"][0]["description"] = "bicycle"
    else:
        changed["valid"][0]["coord_bins_1000"] = [110, 100, 210, 200]
    monkeypatch.setattr(
        readback22, "_qualification_signature",
        lambda row, _route: base if row["kind"] == "reference" else changed,
    )
    comparison = readback22.strict_batch_consistency(
        {"kind": "reference"}, {"kind": "candidate"}, {"image_id": 5},
    )
    assert comparison["parity"] is False


def _resume_fixture(tmp_path: Path) -> tuple[Path, Path, dict, dict]:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}\n")
    checkpoint = tmp_path / "checkpoints" / "step-00016"
    (checkpoint / "adapter").mkdir(parents=True)
    original = {"root": "/frozen/original-s", "fingerprint": "original"}
    saved = {
        "root": str(checkpoint / "adapter"),
        "fingerprint": "changed-after-sixteen-updates",
    }
    manifest = {
        "source_adapter": original,
        "optimizer": {"lr": 1e-5},
        "runtime": {"updates": 256, "checkpoint_steps": [8, 16, 32, 64, 128, 256]},
        "model_config": {"model": {"base_model": "/frozen/base"}},
    }
    torch.save(
        {
            "schema": f"{training.SCHEMA}.checkpoint.v1",
            "manifest": training.binding(manifest_path),
            "source_adapter": original,
            "saved_adapter": saved,
            "optimizer": manifest["optimizer"],
            "step": 16,
        },
        checkpoint / "state.pt",
    )
    (checkpoint / "consensus.json").write_text(
        json.dumps({
            "step": 16, "rank_count": 8,
            "state": {"optimizer_steps": [16]},
        }) + "\n"
    )
    return manifest_path, checkpoint, manifest, saved


def test_resume_loads_checkpoint_adapter_while_preserving_original_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    manifest_path, checkpoint, manifest, saved = _resume_fixture(tmp_path)
    monkeypatch.setattr(
        training22.training, "inspect_dora_adapter_payload",
        lambda path, base: saved
        if Path(path) == checkpoint / "adapter" and base == "/frozen/base"
        else (_ for _ in ()).throw(AssertionError("wrong adapter/base")),
    )
    source, observed = training22.resume_adapter_identity(
        manifest_path=manifest_path, manifest=manifest, resume=checkpoint,
    )
    assert source == checkpoint / "adapter"
    assert observed == saved
    assert observed["fingerprint"] != manifest["source_adapter"]["fingerprint"]


@pytest.mark.parametrize("tamper", ("saved_adapter", "original_lineage", "rank_consensus"))
def test_resume_rejects_mismatched_checkpoint_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tamper: str,
):
    manifest_path, checkpoint, manifest, saved = _resume_fixture(tmp_path)
    monkeypatch.setattr(
        training22.training, "inspect_dora_adapter_payload",
        lambda *_: saved,
    )
    if tamper in ("saved_adapter", "original_lineage"):
        state = torch.load(checkpoint / "state.pt", weights_only=False)
        if tamper == "saved_adapter":
            state["saved_adapter"] = {**saved, "fingerprint": "tampered"}
        else:
            state["source_adapter"] = {**manifest["source_adapter"],
                                       "fingerprint": "wrong-initial-arm"}
        torch.save(state, checkpoint / "state.pt")
    else:
        (checkpoint / "consensus.json").write_text(
            json.dumps({"step": 16, "rank_count": 7,
                        "state": {"optimizer_steps": [16]}}) + "\n"
        )
    with pytest.raises(ValueError, match="resume"):
        training22.resume_adapter_identity(
            manifest_path=manifest_path, manifest=manifest, resume=checkpoint,
        )
