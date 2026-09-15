import json
import shlex
from pathlib import Path

from probes.owner_successor_scale import evaluation as e


def test_n16_anchor_packet_binds_immutable_panel_and_no_rerun_slice():
    packet = e.read(e.ROOT / "packet.json")
    baseline = e.read(e.ROOT / "baseline-packet.json")
    assert packet["schema"] == "native_owner_scale_state.evaluation.packet.v1"
    assert packet["status"] == "cpu_prepared_n16_anchor_no_model_calls"
    assert len(packet["records"]) == 256
    assert packet["n16_anchor"]["adapter_fingerprint"] == e.N16_ADAPTER_FINGERPRINT
    assert packet["n16_anchor"]["adapter_weights_sha256"] == e.N16_ADAPTER_SHA256
    assert baseline["arm"] == "N16-anchor"
    assert baseline["physical_gpus"] == [6, 7]
    assert baseline["phases"]["slice"]["image_ids"] == e.read(e.CONFIRMATION)["image_ids"][:2]


def test_phase_union_is_exact_and_generation_contract_is_frozen():
    packet = e.read(e.ROOT / "baseline-packet.json")
    selection = e.read(e.CONFIRMATION)
    left = packet["phases"]["slice"]["image_ids"]
    right = packet["phases"]["full"]["image_ids"]
    assert len(left) == 2 and len(right) == 254
    assert len(set(left + right)) == 256
    assert set(left + right) == set(selection["image_ids"])
    assert packet["generation"] == {
        "dtype": "fp32",
        "attention": "sdpa",
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "repetition_penalty": 1.0,
        "max_new_tokens": 3084,
        "eos_token_id": 151645,
        "natural_prefix_ids": [],
        "forced_credit": False,
    }


def _command_args(command: str) -> list[str]:
    parts = shlex.split(command)
    if parts[0].startswith("CUDA_VISIBLE_DEVICES="):
        assert parts[0] == "CUDA_VISIBLE_DEVICES=6,7"
        parts = parts[1:]
    assert parts[:3] == ["python", "-m", "probes.owner_successor_scale.evaluation"]
    return parts[3:]


def test_command_surface_uses_wrapper_cli_and_parses_without_legacy_gpu_argument(tmp_path):
    packet = tmp_path / "baseline-packet.json"
    phases = {
        "slice": {
            "run_root": str(tmp_path / "slice-run"),
            "consumer_root": str(tmp_path / "slice-consumer"),
        },
        "full": {
            "run_root": str(tmp_path / "full-run"),
            "consumer_root": str(tmp_path / "full-consumer"),
        },
    }
    commands = e._command_surface(packet, phases)
    parser = e._build_parser()

    for phase in ("slice", "full"):
        launch = parser.parse_args(_command_args(commands[phase]["launch"]))
        assert launch.command == "launch" and launch.phase == phase
        assert launch.packet == packet.resolve()
        assert "--gpu" not in commands[phase]["launch"]

        merge = parser.parse_args(_command_args(commands[phase]["merge"]))
        assert merge.command == "merge" and merge.phase == phase
        assert merge.packet == packet.resolve()

        consume = parser.parse_args(_command_args(commands[phase]["consume"]))
        assert consume.command == "consume" and consume.phase == phase
        assert consume.packet == packet.resolve()
        assert consume.rows == Path(phases[phase]["run_root"]) / "rows.jsonl"
        assert consume.output == Path(phases[phase]["consumer_root"])


def test_prepare_rewrites_native_helper_commands_to_wrapper(monkeypatch, tmp_path):
    packet = {
        "model": {"base": "fixture"},
        "config": {"profile": "fixture"},
        "generation": {"max_new_tokens": e.CAP},
        "n16_anchor": {"adapter": {"root": "/fixture/adapter", "fingerprint": "fixture"}},
    }

    def fake_build(output):
        e.publish(output / "packet.json", packet)
        return packet

    def fake_prepare_baseline(*, packet_path, output_path):
        phases = {
            "slice": {
                "image_ids": [1, 2],
                "run_root": str(tmp_path / "slice-run"),
                "consumer_root": str(tmp_path / "slice-consumer"),
            },
            "full": {
                "image_ids": list(range(3, 257)),
                "run_root": str(tmp_path / "full-run"),
                "consumer_root": str(tmp_path / "full-consumer"),
            },
        }
        baseline = {
            "producer": {"path": "native"},
            "arm": "Stable50",
            "model": {},
            "config": {},
            "generation": {},
            "adapter": {},
            "stored_adapter": {},
            "physical_gpus": list(e.GPUS),
            "phases": phases,
            "consumer": {
                "module": "probes.native_owner_scale.evaluation",
                "merge_command": "native merge",
                "consume_command": "native consume",
            },
            "launch_gate": {
                "launch_command": "native launch",
                "raw_to_consumer": {"slice_merge": "native merge"},
            },
        }
        e.publish(output_path, baseline)
        e.publish(
            output_path.with_name("baseline-launch-request.json"),
            {
                "slice": {"launch": "native", "merge": "native", "consume": "native"},
                "remaining254": {
                    "launch_after_slice_consumer": "native",
                    "merge": "native",
                    "consume": "native",
                },
            },
        )
        return baseline

    monkeypatch.setattr(e, "_build_evaluation_packet", fake_build)
    monkeypatch.setattr(e, "_configure_native", lambda: None)
    monkeypatch.setattr(e.native, "prepare_baseline", fake_prepare_baseline)

    result = e.prepare(output=tmp_path)
    baseline = e.read(tmp_path / "baseline-packet.json")
    request = e.read(tmp_path / "baseline-launch-request.json")
    commands = e._command_surface(tmp_path / "baseline-packet.json", baseline["phases"])

    assert result["model_calls"] == 0
    assert baseline["physical_gpus"] == [6, 7]
    assert baseline["consumer"]["module"] == "probes.owner_successor_scale.evaluation"
    assert baseline["launch_gate"]["launch_command"] == commands["slice"]["launch"]
    assert baseline["launch_gate"]["raw_to_consumer"] == {
        "slice_merge": commands["slice"]["merge"],
        "slice_consume": commands["slice"]["consume"],
        "full_launch_after_slice": commands["full"]["launch"],
        "full_merge": commands["full"]["merge"],
        "full_consume": commands["full"]["consume"],
    }
    assert request["physical_gpus"] == [6, 7]
    assert request["slice"]["launch"] == commands["slice"]["launch"]
    assert request["slice"]["merge"] == commands["slice"]["merge"]
    assert request["slice"]["consume"] == commands["slice"]["consume"]
    assert request["remaining254"]["launch_after_slice_consumer"] == commands["full"]["launch"]
    assert request["remaining254"]["merge"] == commands["full"]["merge"]
    assert request["remaining254"]["consume"] == commands["full"]["consume"]
    assert "probes.native_owner_scale.evaluation" not in json.dumps(baseline["launch_gate"])
    assert "probes.native_owner_scale.evaluation" not in json.dumps(request)


def test_cold_anchor_result_is_combined_and_raw_class_inventory_is_separate():
    result = e.read(e.ROOT / "anchor-result.json")
    assert result["status"] == "cold_verified_n16_anchor_256"
    assert result["denominators"] == {"fresh256": 256, "slice": 2, "remaining254": 254}
    assert result["burden"]["eos"] == 256
    assert result["burden"]["caps"] == 0
    assert result["costs"]["image_forwards"] == 256
    assert result["raw_class_inventory"]["raw_prediction_count"] == 1766
    assert "not silently relabeled or filtered" in result["raw_class_inventory"]["policy"]
