from __future__ import annotations

import shutil
import subprocess

import pytest


def test_swift_rollout_help_exposes_required_vllm_flags() -> None:
    swift_bin = shutil.which("swift")
    if swift_bin is None:
        pytest.skip("swift CLI not available in this environment")

    out = subprocess.check_output([swift_bin, "rollout", "--help"], text=True)

    required_flags = [
        "--vllm_data_parallel_size",
        "--vllm_tensor_parallel_size",
        "--vllm_enforce_eager",
        "--vllm_gpu_memory_utilization",
        "--vllm_max_model_len",
        "--vllm_enable_lora",
        "--torch_dtype",
        "--infer_backend",
    ]
    for flag in required_flags:
        assert flag in out, f"missing expected flag in swift rollout --help: {flag}"
