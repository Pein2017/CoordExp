import pytest
import torch

from probes.native_owner_scale.state import ARMS, _key_vector, canonical_terminal_stop, k_only_branch_slices


def captured_fixture():
    native = [
        (
            torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4) + 10 * layer,
            torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4) + 20 * layer,
        )
        for layer in range(3)
    ]
    return {
        "native": native,
        "owner_a": [(keys + 2 * (layer + 1), values - 100) for layer, (keys, values) in enumerate(native)],
        "owner_b": [(keys + 0.25, values + 300) for keys, values in native],
    }


def test_k_only_self_and_a_preserve_every_value_byte():
    captured = captured_fixture()
    self_branch, self_stats = k_only_branch_slices(captured, "native_self")
    a_branch, a_stats = k_only_branch_slices(captured, "owner_a_k")
    assert self_stats["realized_k_norm"] == 0
    for layer in range(3):
        assert torch.equal(self_branch[layer][0], captured["native"][layer][0])
        assert torch.equal(self_branch[layer][1], captured["native"][layer][1])
        assert torch.equal(a_branch[layer][0], captured["owner_a"][layer][0])
        assert torch.equal(a_branch[layer][1], captured["native"][layer][1])
    assert a_stats["norm_rule"].endswith("4 coordinate positions")


def test_wrong_owner_k_is_globally_norm_matched_not_layer_matched():
    captured = captured_fixture()
    a_branch, _ = k_only_branch_slices(captured, "owner_a_k")
    b_branch, stats = k_only_branch_slices(captured, "owner_b_k_to_a")
    native = _key_vector(captured["native"])
    assert torch.linalg.vector_norm(_key_vector(a_branch) - native) == pytest.approx(
        torch.linalg.vector_norm(_key_vector(b_branch) - native), rel=2e-6
    )
    # B has a uniform direction; one global scale preserves that cross-layer ratio.
    delta = _key_vector(b_branch) - native
    assert delta[:24].mean() == pytest.approx(delta[48:72].mean())
    assert stats["scale"] > 1
    for layer in range(3):
        assert torch.equal(b_branch[layer][1], captured["native"][layer][1])


def test_unregistered_and_zero_norm_fail_closed():
    captured = captured_fixture()
    with pytest.raises(ValueError, match="unregistered"):
        k_only_branch_slices(captured, "adaptive_a")
    captured["owner_b"] = captured["native"]
    with pytest.raises(ValueError, match="zero-norm"):
        k_only_branch_slices(captured, "owner_b_k_to_a")
    assert ARMS == ("native_self", "owner_a_k", "owner_b_k_to_a")


def test_native_im_end_is_canonical_eos_but_early_length_fails_closed():
    assert canonical_terminal_stop("im_end", 90, 3074) == "eos"
    assert canonical_terminal_stop("eos", 390, 3064) == "eos"
    assert canonical_terminal_stop("length", 3064, 3064) == "length"
    with pytest.raises(ValueError, match="before EOS/full"):
        canonical_terminal_stop("length", 100, 3064)
    with pytest.raises(ValueError, match="unknown stop"):
        canonical_terminal_stop("cancelled", 100, 3064)


def test_prepare_retains_completed_gate_after_source_migration(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    import shutil
    from probes.native_owner_scale import state

    original_root = state.RAW_ROOT
    monkeypatch.setattr(state, "RAW_ROOT", tmp_path)
    shutil.copyfile(original_root / "packet.json", tmp_path / "packet.json")
    (tmp_path / "gate-v1").mkdir()
    for name in ["gate.json", "receipt.json", "consumer.json"]:
        shutil.copyfile(original_root / "gate-v1" / name, tmp_path / "gate-v1" / name)
    monkeypatch.setattr(state, "_render_carrier_card", lambda case, output: {"case_id": case["case_id"]})
    result = state.prepare()
    packet = json.loads(Path(result["packet"]).read_text())
    receipt = json.loads((tmp_path / "gate-v1/receipt.json").read_text())
    assert packet["completed_gate"]["runner_sha256"] == receipt["runner_sha256"]
    assert not (tmp_path / "gate-v1/runner.py").exists()

    # New receipts name the external capture. A changed capture must not fall
    # back to a matching historical source.
    retained = state.SourceArchive(Path(
        "/data/CoordExp/docs/history/output-sources/2026-09-21/manifest.json"
    )).resolve(original_root / "gate-v1/runner.py", receipt["runner_sha256"])
    capture = tmp_path / "source-capture.py"
    shutil.copyfile(retained["path"], capture)
    receipt["runner_source"] = str(capture)
    (tmp_path / "gate-v1/receipt.json").write_text(json.dumps(receipt))
    state.prepare()
    capture.write_text("changed")
    with pytest.raises(ValueError, match="completed gate runner changed"):
        state.prepare()
    del receipt["runner_source"]
    receipt["runner_sha256"] = "0" * 64
    (tmp_path / "gate-v1/receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(FileNotFoundError, match="no verified source bytes"):
        state.prepare()


@pytest.mark.parametrize("entry", ["native", "instance", "amplitude"])
def test_runtime_captures_sources_outside_outputs_before_model_load(tmp_path, monkeypatch, entry):
    import json
    from pathlib import Path
    from probes.native_owner_scale import state
    from probes.parallel_owner_research import instance_state, instance_state_amplitude
    from probes.dora_owner_learning import runtime
    from src.artifacts import source_provenance
    from src.config.inference import InferConfig

    monkeypatch.setattr(source_provenance, "SOURCE_ARCHIVE_ROOT", tmp_path / "sources")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 0)
    monkeypatch.setattr(InferConfig, "model_validate", lambda value: value)

    def stop_before_model(*args, **kwargs):
        raise RuntimeError("CPU test stops before model load")

    monkeypatch.setattr(state, "_load_policy", stop_before_model)
    monkeypatch.setattr(runtime, "load_policy", stop_before_model)
    monkeypatch.setattr(instance_state_amplitude, "load_case", stop_before_model)
    source = tmp_path / "input.json"
    source.write_text("{}")
    packet = {"config": {}, "cases": [], "references": {}, "source_files": {}}
    for key in ["source_packet", "parent_panel", "carrier_manifest", "parent_packet"]:
        packet[key] = str(source)
    for key in ["source_sha256", "parent_panel_sha256", "carrier_manifest_sha256", "parent_sha256"]:
        packet[key] = state.file_hash(source)
    packet_path = tmp_path / "packet.json"
    packet_path.write_text(json.dumps(packet))
    out = tmp_path / "outputs" / entry
    with pytest.raises(RuntimeError, match="stops before model load"):
        if entry == "native":
            state.run_stage(packet_path, out, [])
        elif entry == "instance":
            instance_state.execute(packet_path, out, False)
        else:
            instance_state_amplitude.stage(packet_path, out, "capture")
    receipt = json.loads((out / "receipt.json").read_text())
    runner = Path(receipt["runner_source"])
    assert runner.is_relative_to(tmp_path / "sources")
    assert state.file_hash(runner) == receipt["runner_sha256"]
    assert not list(out.rglob("*.py"))
    assert (out / "packet.json").read_bytes() == packet_path.read_bytes()
    if entry == "amplitude":
        assert Path(receipt["dependency_source"]).read_bytes() == Path(instance_state.__file__).read_bytes()
