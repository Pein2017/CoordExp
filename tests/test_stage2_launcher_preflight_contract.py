from __future__ import annotations

import types
from pathlib import Path

import pytest

from src.trainers.rollout_matching.preflight import build_stage2_launcher_preflight


def test_stage2_launcher_preflight_resolves_expected_fields_for_server_cfg() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    train_jsonl = (
        repo_root
        / "public_data"
        / "coco"
        / "rescale_32_768_bbox_max60"
        / "train.coord.jsonl"
    )
    val_jsonl = (
        repo_root
        / "public_data"
        / "coco"
        / "rescale_32_768_bbox_max60"
        / "val.coord.jsonl"
    )

    training_config = types.SimpleNamespace(
        model={
            "model": "output/stage1/coco_bbox_max60-coco80-desc_first/epoch_4-softce_w1-coco80-ckpt_1832-merged",
            "dtype": "bfloat16",
        },
        custom=types.SimpleNamespace(
            train_jsonl=str(train_jsonl),
            val_jsonl=str(val_jsonl),
        ),
        template={"max_pixels": 32 * 32 * 768 * 10000},
        rollout_matching={
            "rollout_backend": "hf",
            "eval_rollout_backend": "vllm",
            "vllm": {
                "mode": "server",
                "max_model_len": 14000,
                "enable_lora": False,
                "gpu_memory_utilization": 0.85,
                "server": {
                    "servers": [
                        {
                            "base_url": "http://127.0.0.1:8000",
                            "group_port": 51216,
                        }
                    ]
                },
            },
        },
    )

    out = build_stage2_launcher_preflight(training_config, config_path=None)

    assert out["rollout_backend"] == "hf"
    assert out["eval_rollout_backend"] == "vllm"
    assert out["vllm_mode"] == "server"
    assert isinstance(out["server_base_urls"], list) and len(out["server_base_urls"]) == 1
    assert out["server_base_urls"][0].startswith("http")
    assert (
        out["server_model"]
        == "output/stage1/coco_bbox_max60-coco80-desc_first/epoch_4-softce_w1-coco80-ckpt_1832-merged"
    )

    root = Path(out["root_image_dir_resolved"]).resolve()
    assert root.name == "rescale_32_768_bbox_max60"
    assert "public_data" in root.parts

    assert int(out["vllm_max_model_len"]) == 14000
    assert bool(out["vllm_enable_lora"]) is False
    assert pytest.approx(float(out["vllm_gpu_memory_utilization"]), rel=1e-6) == 0.85
    assert out["server_torch_dtype"] in {"bfloat16", "bf16"}


def test_stage2_launcher_preflight_accepts_vllm_train_backend() -> None:
    training_config = types.SimpleNamespace(
        model={"model": "some_model", "dtype": "bfloat16"},
        custom=types.SimpleNamespace(train_jsonl="public_data/lvis/train.jsonl"),
        template={"max_pixels": 32 * 32 * 768},
        rollout_matching={
            "rollout_backend": "vllm",
            "eval_rollout_backend": "vllm",
            "vllm": {
                "mode": "server",
                "max_model_len": 2048,
                "enable_lora": False,
                "server": {
                    "servers": [
                        {
                            "base_url": "http://127.0.0.1:8000",
                            "group_port": 51216,
                        }
                    ]
                },
            },
        },
    )

    out = build_stage2_launcher_preflight(training_config, config_path=None)

    assert out["rollout_backend"] == "vllm"
    assert out["eval_rollout_backend"] == "vllm"
    assert out["vllm_mode"] == "server"


def test_stage2_launcher_preflight_rejects_base_url_0_0_0_0() -> None:
    training_config = types.SimpleNamespace(
        model={"model": "some_model", "dtype": "bfloat16"},
        custom=types.SimpleNamespace(train_jsonl="public_data/lvis/train.jsonl"),
        template={"max_pixels": 32 * 32 * 768},
        rollout_matching={
            "rollout_backend": "vllm",
            "eval_rollout_backend": "vllm",
            "vllm": {
                "mode": "server",
                "max_model_len": 2048,
                "enable_lora": False,
                "server": {
                    "servers": [
                        {
                            "base_url": "http://0.0.0.0:8000",
                            "group_port": 51216,
                        }
                    ]
                },
            },
        },
    )

    with pytest.raises(ValueError, match=r"must not use 0\.0\.0\.0"):
        _ = build_stage2_launcher_preflight(training_config, config_path=None)


def test_stage2_launcher_preflight_rejects_multi_server_configs() -> None:
    # build_stage2_launcher_preflight should fail fast before touching JSONL paths.
    training_config = types.SimpleNamespace(
        model={"model": "some_model"},
        custom=types.SimpleNamespace(train_jsonl="public_data/lvis/train.jsonl"),
        template={"max_pixels": 32 * 32 * 768},
        rollout_matching={
            "rollout_backend": "hf",
            "eval_rollout_backend": "vllm",
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
            "rollout_backend": "hf",
            "eval_rollout_backend": "vllm",
            "vllm": {
                "mode": "server",
                "server": {"servers": [{"base_url": "http://127.0.0.1:8000"}]},
            },
        },
    )

    with pytest.raises(ValueError, match=r"rollout_matching\.vllm\.max_model_len is required"):
        _ = build_stage2_launcher_preflight(training_config, config_path=None)
