from __future__ import annotations

import types
from pathlib import Path

import pytest

from src.trainers.rollout_matching.preflight import (
    build_stage2_launcher_preflight,
    resolve_stage2_launcher_preflight,
)


def test_stage2_launcher_preflight_resolves_expected_fields_for_prod_cfg() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cfg = repo_root / "configs/stage2_ab/prod/ab_mixed.yaml"
    out = resolve_stage2_launcher_preflight(str(cfg))

    assert out["rollout_backend"] == "vllm"
    assert out["vllm_mode"] == "server"
    assert isinstance(out["server_base_urls"], list) and len(out["server_base_urls"]) == 1
    assert out["server_base_urls"][0].startswith("http")
    assert out["server_model"] == "output/1-26/checkpoint-1516-merged"

    root = Path(out["root_image_dir_resolved"]).resolve()
    assert root.name == "rescale_32_768_bbox_max60"
    assert "public_data" in root.parts

    assert int(out["vllm_max_model_len"]) == 12000
    assert bool(out["vllm_enable_lora"]) is False
    assert pytest.approx(float(out["vllm_gpu_memory_utilization"]), rel=1e-6) == 0.85
    assert out["server_torch_dtype"] in {"bfloat16", "bf16"}


def test_stage2_launcher_preflight_rejects_multi_server_configs() -> None:
    # build_stage2_launcher_preflight should fail fast before touching JSONL paths.
    training_config = types.SimpleNamespace(
        model={"model": "some_model"},
        custom=types.SimpleNamespace(train_jsonl="public_data/lvis/train.jsonl"),
        template={"max_pixels": 32 * 32 * 768},
        rollout_matching={
            "rollout_backend": "vllm",
            "vllm": {
                "mode": "server",
                "max_model_len": 128,
                "enable_lora": False,
                "server": {
                    "servers": [
                        {"base_url": "http://127.0.0.1:8000"},
                        {"base_url": "http://127.0.0.1:8001"},
                    ]
                },
            },
        },
    )

    with pytest.raises(ValueError, match=r"supports exactly 1 vLLM rollout server"):
        _ = build_stage2_launcher_preflight(training_config, config_path=None)


def test_stage2_launcher_preflight_requires_vllm_max_model_len() -> None:
    training_config = types.SimpleNamespace(
        model={"model": "some_model"},
        custom=types.SimpleNamespace(train_jsonl="public_data/lvis/train.jsonl"),
        template={"max_pixels": 32 * 32 * 768},
        rollout_matching={
            "rollout_backend": "vllm",
            "vllm": {
                "mode": "server",
                "server": {"servers": [{"base_url": "http://127.0.0.1:8000"}]},
            },
        },
    )

    with pytest.raises(ValueError, match=r"rollout_matching\.vllm\.max_model_len is required"):
        _ = build_stage2_launcher_preflight(training_config, config_path=None)
