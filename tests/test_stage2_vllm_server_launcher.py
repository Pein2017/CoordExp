from __future__ import annotations

import json
from pathlib import Path

import pytest

import src.launchers.stage2_vllm_server as launcher


def test_resolve_config_path_supports_configs_prefix(tmp_path: Path) -> None:
    (tmp_path / "configs").mkdir()
    cfg = tmp_path / "configs" / "a.yaml"
    cfg.write_text("x: 1\n", encoding="utf-8")

    got = launcher.resolve_config_path("configs/a.yaml", repo_root=tmp_path)
    assert got == cfg.resolve()


def test_resolve_config_path_appends_yaml(tmp_path: Path) -> None:
    (tmp_path / "configs").mkdir()
    cfg = tmp_path / "configs" / "a.yaml"
    cfg.write_text("x: 1\n", encoding="utf-8")

    got = launcher.resolve_config_path("a", repo_root=tmp_path)
    assert got == cfg.resolve()


def test_parse_base_url_accepts_http_host_port() -> None:
    parsed = launcher.parse_base_url("http://127.0.0.1:8000")
    assert parsed.scheme == "http"
    assert parsed.host == "127.0.0.1"
    assert parsed.port == 8000
    assert parsed.base_url == "http://127.0.0.1:8000"


def test_parse_base_url_rejects_0_0_0_0() -> None:
    with pytest.raises(ValueError, match=r"0\.0\.0\.0"):
        launcher.parse_base_url("http://0.0.0.0:8000")


def test_parse_base_url_requires_port() -> None:
    with pytest.raises(ValueError, match=r"host and port"):
        launcher.parse_base_url("http://127.0.0.1")


def test_parse_base_url_rejects_path() -> None:
    with pytest.raises(ValueError, match=r"must not include a path"):
        launcher.parse_base_url("http://127.0.0.1:8000/api")


def test_validate_gpu_split_detects_overlap() -> None:
    with pytest.raises(ValueError, match=r"disjoint"):
        launcher.validate_gpu_split(
            server_gpus_raw="0,1",
            train_gpus_raw="1,2",
            server_tp=1,
        )


def test_validate_gpu_split_requires_tp_divisibility() -> None:
    with pytest.raises(ValueError, match=r"divisible"):
        launcher.validate_gpu_split(
            server_gpus_raw="0,1,2",
            train_gpus_raw="3",
            server_tp=2,
        )


def test_build_swift_rollout_cmd_contains_required_flags(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    base = launcher.parse_base_url("http://127.0.0.1:8000")
    cmd = launcher.build_swift_rollout_cmd(
        server_model=model_dir,
        base_url=base,
        server_torch_dtype="bfloat16",
        vllm_dp=2,
        vllm_tp=1,
        vllm_enforce_eager=True,
        vllm_gpu_memory_utilization=0.8,
        vllm_max_model_len=4096,
        vllm_enable_lora=False,
        template="qwen3_vl",
        template_max_pixels=786432,
        template_max_length=2048,
        truncation_strategy="delete",
    )

    assert cmd[:2] == ["swift", "rollout"]
    assert "--infer_backend" in cmd
    assert "vllm" in cmd
    assert "--host" in cmd
    assert "127.0.0.1" in cmd
    assert "--port" in cmd
    assert "8000" in cmd
    assert "--max_pixels" in cmd
    assert "786432" in cmd


def test_build_swift_rollout_cmd_serializes_vllm_engine_kwargs(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    cmd = launcher.build_swift_rollout_cmd(
        server_model=model_dir,
        base_url=launcher.parse_base_url("http://127.0.0.1:8000"),
        server_torch_dtype="bfloat16",
        vllm_dp=1,
        vllm_tp=1,
        vllm_enforce_eager=True,
        vllm_gpu_memory_utilization=0.8,
        vllm_max_model_len=4096,
        vllm_enable_lora=False,
        template="qwen3_vl",
        template_max_pixels=786432,
        template_max_length=2048,
        truncation_strategy="delete",
        vllm_engine_kwargs={"mm_processor_kwargs": {"do_resize": False}},
    )

    idx = cmd.index("--vllm_engine_kwargs")
    assert json.loads(cmd[idx + 1]) == {"mm_processor_kwargs": {"do_resize": False}}


def test_build_torchrun_cmd_uses_src_sft(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("x: 1\n", encoding="utf-8")

    cmd = launcher.build_torchrun_cmd(
        config_path=cfg,
        num_gpus=4,
        master_addr="127.0.0.1",
        master_port=12345,
    )

    assert cmd[0] == "torchrun"
    assert "-m" in cmd
    assert "src.sft" in cmd
    assert "--config" in cmd
    assert str(cfg) in cmd


def test_launch_swift_rollout_server_cleans_up_on_health_timeout(monkeypatch) -> None:
    class FakeProc:
        pid = 123

    fake_proc = FakeProc()

    def fake_popen(*_args, **_kwargs):
        return fake_proc

    terminated: dict[str, object] = {"called": False, "proc": None}

    def fake_terminate(proc, *, label: str) -> None:
        terminated["called"] = True
        terminated["proc"] = proc
        assert label == "vLLM server"

    def fake_wait_health(*_args, **_kwargs):
        raise TimeoutError("boom")

    monkeypatch.setattr(launcher.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(launcher, "_terminate_process_group", fake_terminate)
    monkeypatch.setattr(launcher, "_wait_for_server_health", fake_wait_health)

    with pytest.raises(TimeoutError):
        launcher.launch_swift_rollout_server(
            server_cmd=["swift", "rollout"],
            server_env={},
            repo_root=Path("/tmp"),
            health_url="http://127.0.0.1:8000/health/",
            world_url="http://127.0.0.1:8000/get_world_size/",
            wait_timeout_s=0.01,
            wait_interval_s=0.01,
            expected_world_size=1,
        )

    assert terminated["called"] is True
    assert terminated["proc"] is fake_proc


def test_wait_for_server_health_bounds_probe_timeout_by_remaining_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProc:
        returncode = None

        def poll(self) -> None:
            return None

    proc = FakeProc()
    now = {"t": 0.0}
    probe_timeouts: list[float] = []

    def fake_monotonic() -> float:
        return float(now["t"])

    def fake_http_get(_url: str, *, timeout_s: float) -> tuple[int, str]:
        probe_timeouts.append(float(timeout_s))
        # Simulate a stuck probe consuming exactly the provided timeout budget.
        now["t"] += float(timeout_s)
        return 503, "pending"

    monkeypatch.setattr(launcher.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(launcher.time, "sleep", lambda _s: None)
    monkeypatch.setattr(launcher, "_http_get", fake_http_get)

    with pytest.raises(TimeoutError, match=r"did not become ready within 0.2s"):
        launcher._wait_for_server_health(
            server_proc=proc,
            health_url="http://127.0.0.1:8000/health/",
            wait_timeout_s=0.2,
            wait_interval_s=0.1,
        )

    assert probe_timeouts
    assert max(probe_timeouts) <= 0.200001
    assert max(probe_timeouts) < 5.0
