from __future__ import annotations

import subprocess
import sys


def test_swift_rollout_help_exposes_required_vllm_flags() -> None:
    out = subprocess.check_output(
        [sys.executable, "-m", "swift.cli.rollout", "--help"],
        text=True,
    )

    required_flags = [
        "--vllm_data_parallel_size",
        "--vllm_tensor_parallel_size",
        "--vllm_enforce_eager",
        "--vllm_gpu_memory_utilization",
        "--vllm_max_model_len",
        "--vllm_enable_lora",
        "--vllm_max_lora_rank",
        "--vllm_engine_kwargs",
        "--torch_dtype",
        "--infer_backend",
    ]
    for flag in required_flags:
        assert flag in out, f"missing expected flag in swift rollout --help: {flag}"
