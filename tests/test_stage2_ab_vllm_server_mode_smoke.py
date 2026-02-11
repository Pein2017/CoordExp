from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

import pytest
import torch


def _env_flag(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _find_ms_swift_root(repo_root: Path) -> Optional[Path]:
    # This repo often lives next to a sibling `ms-swift/` checkout.
    for base in [repo_root.parent, *repo_root.parents]:
        candidate = (base / "ms-swift").resolve()
        if candidate.is_dir():
            return candidate
    return None


def _nvidia_gpu_count() -> int:
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True)
    except Exception:
        return 0
    return sum(1 for line in out.splitlines() if line.strip().startswith("GPU "))


def _unset_proxies(env: dict[str, str]) -> None:
    # Some environments export http(s)_proxy globally. Ensure local requests to
    # the rollout server do not leak into a proxy.
    for key in (
        "http_proxy",
        "https_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "all_proxy",
        "ALL_PROXY",
    ):
        env.pop(key, None)


def _set_no_proxy(env: dict[str, str]) -> None:
    # Learner/server processes should not route localhost through an HTTP proxy.
    existing = env.get("NO_PROXY") or env.get("no_proxy") or ""
    tokens = [t.strip() for t in existing.split(",") if t.strip()]
    for needed in ("127.0.0.1", "localhost"):
        if needed not in tokens:
            tokens.append(needed)
    value = ",".join(tokens)
    env["NO_PROXY"] = value
    env["no_proxy"] = value


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _http_get(url: str, *, timeout_s: float) -> tuple[int, bytes]:
    """HTTP GET that bypasses env proxies.

    This repo commonly runs with `http_proxy/https_proxy` set, which can break
    localhost health checks.
    """

    import urllib.request

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(url, method="GET")
    with opener.open(req, timeout=timeout_s) as resp:
        return int(resp.getcode()), resp.read()


def _wait_for_rollout_server(
    base_url: str, *, timeout_s: float, proc: Optional[subprocess.Popen[str]] = None
) -> dict[str, Any]:
    deadline = time.time() + float(timeout_s)
    last_err: Optional[BaseException] = None

    base_url = base_url.rstrip("/")
    while time.time() < deadline:
        if proc is not None:
            rc = proc.poll()
            if rc is not None:
                raise RuntimeError(
                    f"rollout server process exited early (returncode={rc})"
                )

        try:
            code, _ = _http_get(f"{base_url}/health/", timeout_s=5.0)
            if code != 200:
                time.sleep(1.0)
                continue

            code, body = _http_get(f"{base_url}/get_world_size/", timeout_s=5.0)
            if code == 200:
                return json.loads(body.decode("utf-8"))
        except BaseException as exc:
            last_err = exc

        time.sleep(1.0)

    raise TimeoutError(
        f"rollout server did not become healthy within {timeout_s:.1f}s; last_err={last_err!r}"
    )


def _tail(path: Path, *, n_lines: int = 120) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-int(n_lines) :])
    except Exception as exc:  # pragma: no cover
        return f"<failed to read {path}: {exc}>"


def _select_model_dir() -> Path:
    override = os.environ.get("COORDEXP_STAGE2_AB_MODEL")
    if override:
        return Path(override)

    # Prefer the stable stage-1 merged ckpt dir if present (mentioned in progress/prompt_for_4GPUs.md).
    candidates = [
        Path("output/1-26/checkpoint-1516-merged"),
        Path(
            "output/1-26/stage_1/poly_prefer_semantic_max60-pure_ce/mixed-merged-ckpt-1516"
        ),
        Path(
            "output/1-26/stage_1/poly_prefer_semantic_max60-pure_ce/"
            "v0-20260126-162638/epoch_4-pure_ce-LRs-2e-4_1e-4_4e-4-from-base-4B/"
            "merged_checkpoint-1516_20260128-130701"
        ),
    ]
    for path in candidates:
        if path.is_dir():
            return path

    raise FileNotFoundError(
        "No Stage-1 checkpoint directory found. Set COORDEXP_STAGE2_AB_MODEL to a local merged checkpoint folder."
    )


def test_stage2_ab_b_only_vllm_server_mode_smoke(tmp_path: Path):
    """End-to-end Stage-2 AB Channel-B smoke with vLLM **server mode**.

    This is intentionally gated behind an env flag because it requires GPUs + a local checkpoint.
    """

    if not _env_flag("COORDEXP_RUN_4GPU_SMOKE"):
        pytest.skip(
            "Set COORDEXP_RUN_4GPU_SMOKE=1 to run the 4-GPU server-mode smoke test."
        )

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available in this environment.")
    if _nvidia_gpu_count() < 2:
        pytest.skip("Need >=2 GPUs for 1 rollout server + 1 learner.")

    pytest.importorskip("vllm")

    swift_bin = shutil.which("swift")
    if not swift_bin:
        pytest.skip(
            "`swift` CLI not found on PATH; required to launch the rollout server."
        )

    repo_root = Path(__file__).resolve().parent.parent
    ms_swift_root = _find_ms_swift_root(repo_root)

    model_dir = _select_model_dir()

    server_visible = os.environ.get(
        "COORDEXP_STAGE2_AB_SERVER_CUDA_VISIBLE_DEVICES", "0"
    )
    learner_visible = os.environ.get(
        "COORDEXP_STAGE2_AB_LEARNER_CUDA_VISIBLE_DEVICES", "1"
    )
    vllm_max_model_len = int(
        os.environ.get("COORDEXP_STAGE2_AB_VLLM_MAX_MODEL_LEN", "8192")
    )
    vllm_gpu_memory_utilization = float(
        os.environ.get("COORDEXP_STAGE2_AB_VLLM_GPU_MEMORY_UTILIZATION", "0.90")
    )
    if vllm_max_model_len <= 0:
        raise ValueError("COORDEXP_STAGE2_AB_VLLM_MAX_MODEL_LEN must be a positive int")
    if not (0.0 < vllm_gpu_memory_utilization <= 1.0):
        raise ValueError(
            "COORDEXP_STAGE2_AB_VLLM_GPU_MEMORY_UTILIZATION must be in (0, 1]"
        )

    server_dp = int(os.environ.get("COORDEXP_STAGE2_AB_SERVER_DP", "1"))
    if server_dp <= 0:
        raise ValueError("COORDEXP_STAGE2_AB_SERVER_DP must be >= 1")

    server_visible_ids = [
        s.strip() for s in str(server_visible).split(",") if s.strip()
    ]
    if len(server_visible_ids) < server_dp:
        raise ValueError(
            f"COORDEXP_STAGE2_AB_SERVER_DP={server_dp} requires at least {server_dp} "
            f"GPUs in COORDEXP_STAGE2_AB_SERVER_CUDA_VISIBLE_DEVICES; got {server_visible!r}"
        )

    port = _pick_free_port()
    group_port = _pick_free_port()
    while group_port == port:
        group_port = _pick_free_port()
    base_url = f"http://127.0.0.1:{port}"

    out_root = tmp_path / "stage2_ab_out"
    tb_root = tmp_path / "stage2_ab_tb"
    out_root.mkdir(parents=True, exist_ok=True)
    tb_root.mkdir(parents=True, exist_ok=True)

    run_name = f"b_only_vllm_server_smoke_test_{int(time.time())}"

    # Override the stock smoke config with a temp config that:
    # - uses a local checkpoint (model_dir)
    # - uses a free server port + group_port
    # - writes outputs under tmp_path
    cfg_path = tmp_path / "stage2_ab_b_only_vllm_server_smoke.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                f"extends: {(repo_root / 'configs/stage2_ab/smoke/b_only.yaml').as_posix()}",
                f"global_max_length: {vllm_max_model_len}",
                "model:",
                f"  model: {model_dir}",
                "training:",
                f"  output_dir: {out_root}",
                f"  run_name: {run_name}",
                f"  logging_dir: {tb_root}",
                "  max_steps: 3",
                "  effective_batch_size: null",
                "  gradient_accumulation_steps: 1",
                "stage2_ab:",
                "  channel_b:",
                "    reordered_gt_sft: false",
                "    semantic_desc_gate:",
                "      enabled: false",
                "custom:",
                "  extra:",
                "    rollout_matching:",
                "      max_new_tokens: 256",
                "      vllm:",
                "        server:",
                "          servers:",
                f"            - base_url: {base_url}",
                f"              group_port: {group_port}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    server_log = tmp_path / "swift_rollout_server.log"
    learner_log = tmp_path / "stage2_ab_learner.log"

    server_env = os.environ.copy()
    server_env["CUDA_VISIBLE_DEVICES"] = str(server_visible)
    _unset_proxies(server_env)
    _set_no_proxy(server_env)

    # Prefer explicit PYTHONPATH so that both this repo and a sibling ms-swift checkout resolve.
    py_path_parts = [str(repo_root)]
    if ms_swift_root is not None:
        py_path_parts.append(str(ms_swift_root))
    server_env["PYTHONPATH"] = os.pathsep.join(py_path_parts)

    server_cmd = [
        swift_bin,
        "rollout",
        "--model",
        str(model_dir),
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--infer_backend",
        "vllm",
        "--vllm_data_parallel_size",
        str(server_dp),
        "--vllm_tensor_parallel_size",
        "1",
        "--vllm_gpu_memory_utilization",
        str(vllm_gpu_memory_utilization),
        "--vllm_max_model_len",
        str(vllm_max_model_len),
        "--vllm_enable_lora",
        "false",
    ]

    proc: Optional[subprocess.Popen[str]] = None
    try:
        with server_log.open("w", encoding="utf-8") as fp:
            proc = subprocess.Popen(
                server_cmd,
                cwd=str(repo_root),
                env=server_env,
                stdout=fp,
                stderr=subprocess.STDOUT,
                text=True,
            )

        # Wait until the HTTP server is healthy (model load can be slow).
        try:
            info = _wait_for_rollout_server(base_url, timeout_s=15 * 60, proc=proc)
        except BaseException as exc:
            rc = None if proc is None else proc.poll()
            raise AssertionError(
                f"rollout server failed to become healthy (returncode={rc}); last server logs:\n"
                + _tail(server_log)
            ) from exc

        assert int(info.get("world_size", -1)) == server_dp

        learner_env = os.environ.copy()
        learner_env["CUDA_VISIBLE_DEVICES"] = str(learner_visible)
        learner_env["PYTHONPATH"] = os.pathsep.join(py_path_parts)
        _unset_proxies(learner_env)
        _set_no_proxy(learner_env)

        learner_cmd = [
            sys.executable,
            "-m",
            "src.sft",
            "--config",
            str(cfg_path),
        ]

        with learner_log.open("w", encoding="utf-8") as fp:
            out = subprocess.run(
                learner_cmd,
                cwd=str(repo_root),
                env=learner_env,
                stdout=fp,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=45 * 60,
            )

        assert out.returncode == 0, (
            "stage2-ab learner failed; last learner logs:\n"
            + _tail(learner_log)
            + "\n\nlast server logs:\n"
            + _tail(server_log)
        )

        learner_text = learner_log.read_text(encoding="utf-8", errors="replace")
        assert "vLLM rollout server engine_type" in learner_text

        jsonl_paths = sorted(out_root.rglob("logging.jsonl"))
        assert jsonl_paths, (
            f"No logging.jsonl produced under {out_root}. "
            "Check learner logs for trainer init failures.\n\n" + _tail(learner_log)
        )
        if len(jsonl_paths) > 1:
            raise AssertionError(
                f"Expected 1 logging.jsonl under {out_root}, found {len(jsonl_paths)}: {jsonl_paths}"
            )

        records: list[dict[str, Any]] = []
        for line in (
            jsonl_paths[0].read_text(encoding="utf-8", errors="replace").splitlines()
        ):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                records.append(obj)

        merged: dict[str, Any] = {}
        for rec in records:
            merged.update(rec)

        b_records = [
            r
            for r in records
            if float(r.get("stage2/channel_b", 0.0)) == pytest.approx(1.0)
        ]
        assert b_records, (
            "No Channel-B logs found; async queue may have been starved. "
            "Increase max_steps or stage2_ab.channel_b.async.prefetch_target_packs for this smoke run.\n\n"
            + _tail(learner_log)
        )
        merged_b = dict(b_records[-1])

        assert float(merged_b.get("stage2/channel_b", 0.0)) == pytest.approx(1.0)
        assert float(merged_b.get("rollout/decode_non_beam_count", 0.0)) > 0.0
        assert float(merged_b.get("rollout/rollout_len_mean", 0.0)) > 0.0
        assert float(merged_b.get("rollout/f1", 0.0)) >= 0.0

        # Minimal logging contract: avoid legacy/verbose rollout config gauges.
        assert "rollout/backend_vllm" not in merged_b
        assert "rollout/decode_mode_greedy" not in merged_b

    finally:
        if proc is not None:
            try:
                proc.terminate()
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:  # pragma: no cover
                proc.kill()
                proc.wait(timeout=30)
