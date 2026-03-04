from pathlib import Path

from src.launchers.stage2_vllm_server import build_swift_rollout_cmd, parse_base_url


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
