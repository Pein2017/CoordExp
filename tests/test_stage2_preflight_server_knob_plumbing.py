from __future__ import annotations

from pathlib import Path

from src.trainers.rollout_matching.preflight import resolve_stage2_launcher_preflight


def test_stage2_preflight_extracts_server_runtime_knobs_from_yaml() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / "configs/stage2_two_channel/prod/ab_mixed.yaml"

    preflight = resolve_stage2_launcher_preflight(str(config_path))

    assert preflight.get("server_torch_dtype") == "bfloat16"
    assert preflight.get("vllm_gpu_memory_utilization") == 0.85

