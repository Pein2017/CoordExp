import json
from pathlib import Path

from src.launchers.stage2_vllm_server import build_swift_rollout_cmd, parse_base_url
from src.trainers.rollout_matching.preflight import resolve_stage2_launcher_preflight


def _flag_value(cmd: list[str], flag: str) -> str:
    index = cmd.index(flag)
    return cmd[index + 1]


def test_stage2_launcher_shell_is_thin_python_delegate() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "train_stage2.sh"
    src = script.read_text(encoding="utf-8")
    assert "exec python -m src.launchers.stage2_vllm_server" in src


def test_stage2_launcher_passes_template_flags_to_swift_rollout() -> None:
    cmd = build_swift_rollout_cmd(
        server_model=Path("output/server-model"),
        base_url=parse_base_url("http://127.0.0.1:8000"),
        server_torch_dtype="bfloat16",
        vllm_dp=2,
        vllm_tp=1,
        vllm_enforce_eager=True,
        vllm_gpu_memory_utilization=0.8,
        vllm_max_model_len=8192,
        vllm_enable_lora=False,
        template="qwen3_vl",
        template_max_pixels=1280,
        template_max_length=4096,
        truncation_strategy="left",
    )

    assert _flag_value(cmd, "--template") == "qwen3_vl"
    assert _flag_value(cmd, "--max_length") == "4096"
    assert _flag_value(cmd, "--truncation_strategy") == "left"
    assert _flag_value(cmd, "--max_pixels") == "1280"
    assert _flag_value(cmd, "--vllm_tensor_parallel_size") == "1"
    assert _flag_value(cmd, "--vllm_enforce_eager") == "true"


def test_stage2_launcher_default_config_exists() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "train_stage2.sh"
    src = script.read_text(encoding="utf-8")

    assert "configs/stage2_two_channel/smoke/ab_mixed_20steps.yaml" in src
    assert (
        repo_root / "configs" / "stage2_two_channel" / "smoke" / "ab_mixed_20steps.yaml"
    ).is_file()


def test_stage2_launcher_round_trips_preflight_engine_kwargs() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / "configs/stage2_two_channel/prod/ab_mixed.yaml"
    preflight = resolve_stage2_launcher_preflight(str(config_path))

    cmd = build_swift_rollout_cmd(
        server_model=Path(preflight["server_model"]),
        base_url=parse_base_url(preflight["server_base_urls"][0]),
        server_torch_dtype=str(preflight["server_torch_dtype"]),
        vllm_dp=1,
        vllm_tp=int(preflight["vllm_tensor_parallel_size"]),
        vllm_enforce_eager=bool(preflight["vllm_enforce_eager"]),
        vllm_gpu_memory_utilization=float(preflight["vllm_gpu_memory_utilization"]),
        vllm_max_model_len=int(preflight["vllm_max_model_len"]),
        vllm_enable_lora=bool(preflight["vllm_enable_lora"]),
        template=str(preflight["server_template"]),
        template_max_pixels=int(preflight["template_max_pixels"]),
        template_max_length=int(preflight["server_max_length"]),
        truncation_strategy=str(preflight["server_truncation_strategy"]),
        vllm_engine_kwargs=preflight["vllm_engine_kwargs"],
    )

    assert json.loads(_flag_value(cmd, "--vllm_engine_kwargs")) == {
        "mm_processor_kwargs": {"do_resize": False}
    }
