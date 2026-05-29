from __future__ import annotations

from pathlib import Path

from src.trainers.rollout_matching.preflight import resolve_stage2_launcher_preflight


def test_stage2_preflight_extracts_server_runtime_knobs_from_yaml() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / "configs/stage2_rollout_correction/base.yaml"

    preflight = resolve_stage2_launcher_preflight(str(config_path))

    assert preflight.get("server_torch_dtype") == "bfloat16"
    assert preflight.get("vllm_gpu_memory_utilization") == 0.85
    assert preflight.get("vllm_engine_kwargs") == {
        "enable_tower_connector_lora": True,
        "mm_processor_kwargs": {"do_resize": False},
    }
    assert preflight.get("vllm_enable_lora") is True
    assert preflight.get("vllm_max_lora_rank") == 16
