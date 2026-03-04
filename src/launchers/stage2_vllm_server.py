"""Stage-2 AB combined launcher (vLLM rollout server + learner).

Operator contract:
- Hyperparameters live in YAML.
- Runtime knobs are environment variables only (no positional args).

This launcher is intentionally synchronous (no asyncio) and owns process lifecycle
so we do not leak `swift rollout` servers on failure.
"""

from __future__ import annotations

import json
import os
import random
import shlex
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

from src.trainers.rollout_matching.preflight import resolve_stage2_launcher_preflight


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _format_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(str(tok)) for tok in cmd)


def _die(message: str, *, rc: int = 1) -> None:
    print(f"[ERROR] {message}", file=sys.stderr)
    raise SystemExit(rc)


def _info(message: str) -> None:
    print(f"[INFO] {message}")


def _get_env_str(*names: str, default: str | None = None) -> str | None:
    for name in names:
        raw = os.environ.get(name)
        if raw is None:
            continue
        value = str(raw).strip()
        if value:
            return value
    return default


def _parse_seconds(raw: str | None, *, default: float) -> float:
    if raw is None:
        return float(default)
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"expected seconds as float/int, got: {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"seconds must be > 0, got: {value}")
    return float(value)


def _parse_csv_tokens(raw: str, *, name: str) -> list[str]:
    tokens = [tok.strip() for tok in raw.split(",")]
    out = [tok for tok in tokens if tok]
    if not out:
        raise ValueError(f"{name} must contain at least one device id")
    return out


def validate_gpu_split(
    *,
    server_gpus_raw: str,
    train_gpus_raw: str,
    server_tp: int,
) -> tuple[list[str], list[str], int]:
    server_gpus = _parse_csv_tokens(server_gpus_raw, name="server_gpus")
    train_gpus = _parse_csv_tokens(train_gpus_raw, name="train_gpus")

    overlap_set = set(server_gpus) & set(train_gpus)
    if overlap_set:
        overlap = sorted(overlap_set)
        raise ValueError(
            "server_gpus and train_gpus must be disjoint. "
            f"Overlap: {overlap}. server_gpus={server_gpus_raw!r} train_gpus={train_gpus_raw!r}"
        )

    if int(server_tp) <= 0:
        raise ValueError(
            f"rollout_matching.vllm.tensor_parallel_size must be >= 1, got: {server_tp}"
        )

    if len(server_gpus) % int(server_tp) != 0:
        raise ValueError(
            "server_gpus count must be divisible by rollout_matching.vllm.tensor_parallel_size. "
            f"server_gpus={server_gpus_raw!r} n={len(server_gpus)} tensor_parallel_size={server_tp}"
        )

    server_dp = int(len(server_gpus) / int(server_tp))
    return server_gpus, train_gpus, server_dp


def resolve_config_path(config_raw: str, *, repo_root: Path) -> Path:
    if not config_raw.strip():
        raise ValueError("config must be non-empty")

    raw = config_raw.strip()
    if raw.startswith("/"):
        path = Path(raw)
    elif raw.startswith("configs/"):
        path = repo_root / raw
    elif raw.endswith(".yaml"):
        path = repo_root / "configs" / raw
    else:
        path = repo_root / "configs" / f"{raw}.yaml"

    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    return path


@dataclass(frozen=True)
class _BaseUrl:
    base_url: str
    scheme: str
    host: str
    port: int


def parse_base_url(base_url: str) -> _BaseUrl:
    raw = str(base_url).strip()
    if not raw:
        raise ValueError("base_url must be non-empty")

    parsed = urlparse(raw)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"base_url scheme must be http(s), got: {raw!r}")

    host = parsed.hostname
    port = parsed.port
    if host is None or port is None:
        raise ValueError(f"base_url must include host and port, got: {raw!r}")

    if host == "0.0.0.0":
        raise ValueError(
            "base_url host must not be 0.0.0.0. Use 127.0.0.1 (or an explicit interface IP)."
        )

    if parsed.path not in {"", "/"}:
        raise ValueError(
            f"base_url must not include a path (swift rollout serves at /). Got: {raw!r}"
        )

    normalized = f"{parsed.scheme}://{host}:{port}"
    return _BaseUrl(
        base_url=normalized, scheme=parsed.scheme, host=host, port=int(port)
    )


def _assert_port_free(host: str, port: int) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((host, int(port)))
    except OSError as exc:
        raise OSError(
            f"Requested rollout server endpoint is already in use: {host}:{port}"
        ) from exc
    finally:
        sock.close()


def _merge_no_proxy(existing_raw: str, *, hosts: list[str]) -> str:
    tokens = [tok.strip() for tok in existing_raw.split(",") if tok.strip()]
    for host in hosts:
        host = str(host).strip()
        if not host:
            continue
        if host not in tokens:
            tokens.append(host)
    return ",".join(tokens)


def _apply_no_proxy(env: dict[str, str], *, hosts: list[str]) -> None:
    np_raw = env.get("NO_PROXY") or env.get("no_proxy") or ""
    merged = _merge_no_proxy(np_raw, hosts=hosts)
    env["NO_PROXY"] = merged
    env["no_proxy"] = merged


def _ensure_pythonpath(env: dict[str, str], *, repo_root: Path) -> None:
    raw = env.get("PYTHONPATH", "")
    if raw:
        parts = raw.split(":")
        if str(repo_root) not in parts:
            env["PYTHONPATH"] = f"{repo_root}:{raw}"
    else:
        env["PYTHONPATH"] = str(repo_root)


def _set_default_env(env: dict[str, str], key: str, value: str) -> None:
    if key not in env or env.get(key) in (None, ""):
        env[key] = value


def build_swift_rollout_cmd(
    *,
    server_model: Path,
    base_url: _BaseUrl,
    server_torch_dtype: str,
    vllm_dp: int,
    vllm_tp: int,
    vllm_enforce_eager: bool,
    vllm_gpu_memory_utilization: float,
    vllm_max_model_len: int,
    vllm_enable_lora: bool,
    template: str | None,
    template_max_pixels: int,
    template_max_length: int | None,
    truncation_strategy: str | None,
) -> list[str]:
    cmd = [
        "swift",
        "rollout",
        "--model",
        str(server_model),
        "--host",
        base_url.host,
        "--port",
        str(base_url.port),
        "--infer_backend",
        "vllm",
        "--torch_dtype",
        str(server_torch_dtype),
        "--vllm_data_parallel_size",
        str(int(vllm_dp)),
        "--vllm_tensor_parallel_size",
        str(int(vllm_tp)),
        "--vllm_enforce_eager",
        "true" if bool(vllm_enforce_eager) else "false",
        "--vllm_gpu_memory_utilization",
        str(float(vllm_gpu_memory_utilization)),
        "--vllm_max_model_len",
        str(int(vllm_max_model_len)),
        "--vllm_enable_lora",
        "true" if bool(vllm_enable_lora) else "false",
    ]

    if template:
        cmd.extend(["--template", template])
    if template_max_length is not None:
        cmd.extend(["--max_length", str(int(template_max_length))])
    if truncation_strategy:
        cmd.extend(["--truncation_strategy", truncation_strategy])

    cmd.extend(["--max_pixels", str(int(template_max_pixels))])
    return cmd


def build_torchrun_cmd(
    *,
    config_path: Path,
    num_gpus: int,
    master_addr: str,
    master_port: int,
) -> list[str]:
    return [
        "torchrun",
        f"--nproc_per_node={int(num_gpus)}",
        f"--master_addr={master_addr}",
        f"--master_port={int(master_port)}",
        "-m",
        "src.sft",
        "--config",
        str(config_path),
    ]


def _http_get(url: str, *, timeout_s: float) -> tuple[int, str]:
    import urllib.error
    import urllib.request

    req = urllib.request.Request(url, method="GET")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    try:
        with opener.open(req, timeout=float(timeout_s)) as resp:
            status = int(getattr(resp, "status", 200))
            body = resp.read()
    except urllib.error.HTTPError as exc:
        status = int(getattr(exc, "code", 0) or 0)
        body = exc.read() if hasattr(exc, "read") else b""
    except Exception as exc:
        return 0, f"{exc.__class__.__name__}: {exc}"

    try:
        text = body.decode("utf-8", errors="replace").strip()
    except Exception:
        text = str(body)
    return status, text


def _parse_world_size(raw_text: str) -> int:
    raw = str(raw_text).strip()
    if not raw:
        raise ValueError("empty world_size payload")

    payload: Any
    try:
        payload = json.loads(raw)
    except Exception:
        payload = raw

    if isinstance(payload, int):
        return int(payload)
    if isinstance(payload, str) and payload.isdigit():
        return int(payload)
    if isinstance(payload, Mapping):
        if "world_size" in payload:
            return int(payload["world_size"])

    raise ValueError(f"unrecognized world_size payload: {raw!r}")


def _terminate_process_group(proc: subprocess.Popen[str], *, label: str) -> None:
    if proc.poll() is not None:
        return

    pid = int(proc.pid)
    _info(f"Stopping {label} (pid={pid})")

    try:
        os.killpg(pid, signal.SIGTERM)
    except Exception:
        try:
            proc.terminate()
        except Exception:
            return

    try:
        proc.wait(timeout=5)
    except Exception:
        try:
            os.killpg(pid, signal.SIGKILL)
        except Exception:
            try:
                proc.kill()
            except Exception:
                return


def _wait_for_server_health(
    *,
    server_proc: subprocess.Popen[str],
    health_url: str,
    wait_timeout_s: float,
    wait_interval_s: float,
) -> None:
    _info(f"Waiting for vLLM server readiness: {health_url}")

    start_ts = time.monotonic()
    last_status = 0
    last_body = ""
    while True:
        if server_proc.poll() is not None:
            raise RuntimeError(
                "vLLM server exited before becoming ready. "
                f"return_code={server_proc.returncode}"
            )

        last_status, last_body = _http_get(health_url, timeout_s=5.0)
        if last_status == 200:
            return
        if time.monotonic() - start_ts > wait_timeout_s:
            raise TimeoutError(
                f"vLLM server did not become ready within {wait_timeout_s}s: {health_url} "
                f"(last status={last_status}, body={last_body!r})"
            )
        time.sleep(wait_interval_s)


def _query_server_world_size(*, world_url: str) -> int:
    status, body = _http_get(world_url, timeout_s=10.0)
    if status != 200:
        raise RuntimeError(
            f"Failed to query rollout server world_size from {world_url}. "
            f"status={status} body={body!r}"
        )
    return _parse_world_size(body)


def launch_swift_rollout_server(
    *,
    server_cmd: list[str],
    server_env: Mapping[str, str],
    repo_root: Path,
    health_url: str,
    world_url: str,
    wait_timeout_s: float,
    wait_interval_s: float,
    expected_world_size: int,
) -> subprocess.Popen[str]:
    proc = subprocess.Popen(
        server_cmd,
        cwd=str(repo_root),
        env=dict(server_env),
        text=True,
        start_new_session=True,
    )

    try:
        _wait_for_server_health(
            server_proc=proc,
            health_url=health_url,
            wait_timeout_s=wait_timeout_s,
            wait_interval_s=wait_interval_s,
        )
        world_size = _query_server_world_size(world_url=world_url)
        if int(world_size) != int(expected_world_size):
            raise RuntimeError(
                f"Unexpected rollout server world_size at {world_url}: got={world_size} expected={expected_world_size}. "
                "This usually indicates a stale server or port mismatch; aborting before learner launch."
            )
        return proc
    except Exception:
        _terminate_process_group(proc, label="vLLM server")
        raise


def _run_validate_jsonl(
    *, repo_root: Path, jsonl_path: str, max_pixels: int, mode: str, n: int
) -> None:
    cmd = [
        sys.executable,
        str(repo_root / "public_data" / "scripts" / "validate_jsonl.py"),
        str(jsonl_path),
        "--max-pixels",
        str(int(max_pixels)),
        "--multiple-of",
        "32",
        "--image-check-mode",
        str(mode),
        "--enforce-rescale-images-real-dir",
        "--image-check-n",
        str(int(n)),
    ]
    subprocess.run(cmd, cwd=str(repo_root), check=True)


def main() -> int:
    try:
        if len(sys.argv) != 1:
            _die(
                "This launcher accepts environment variables only (no positional args). "
                "Invoke via scripts/train_stage2.sh.",
                rc=2,
            )

        config_raw = _get_env_str("config", "CONFIG", "CONFIG_PATH")
        if config_raw is None:
            _die(
                "Missing config. Set env var `config=...` (or `CONFIG=...`). Example: "
                "config=configs/stage2_two_channel/prod/ab_mixed.yaml",
                rc=2,
            )

        server_gpus_raw = _get_env_str("server_gpus", "SERVER_GPUS")
        train_gpus_raw = _get_env_str("train_gpus", "TRAIN_GPUS")
        if server_gpus_raw is None or train_gpus_raw is None:
            _die(
                "Missing GPU split. Set `server_gpus=...` and `train_gpus=...` (or uppercase variants).",
                rc=2,
            )

        wait_timeout_s = _parse_seconds(
            _get_env_str("wait_timeout", "WAIT_TIMEOUT"), default=900.0
        )
        wait_interval_s = _parse_seconds(
            _get_env_str("wait_interval", "WAIT_INTERVAL"), default=2.0
        )

        config_path = resolve_config_path(config_raw, repo_root=_REPO_ROOT)

        preflight = resolve_stage2_launcher_preflight(str(config_path))

        rollout_backend = preflight.get("rollout_backend")
        vllm_mode = preflight.get("vllm_mode")
        server_base_urls = preflight.get("server_base_urls")

        if rollout_backend != "vllm" or vllm_mode != "server":
            raise ValueError(
                "scripts/train_stage2.sh (server-mode combined launcher) requires YAML: "
                "rollout_matching.rollout_backend=vllm and rollout_matching.vllm.mode=server. "
                f"Got rollout_backend={rollout_backend!r} vllm_mode={vllm_mode!r}."
            )

        if not isinstance(server_base_urls, list) or len(server_base_urls) != 1:
            raise ValueError(
                "Single-server only: rollout_matching.vllm.server.servers must have exactly 1 entry. "
                f"Got server_base_urls={server_base_urls!r}."
            )

        base_url = parse_base_url(server_base_urls[0])
        if base_url.scheme != "http":
            raise ValueError(
                "Combined launcher only supports http base_url (swift rollout serves http). "
                f"Got: {base_url.base_url!r}"
            )

        _assert_port_free(base_url.host, base_url.port)

        server_model_raw = str(preflight.get("server_model") or "").strip()
        if not server_model_raw:
            raise ValueError("Preflight missing server_model")
        server_model = Path(server_model_raw)
        if not server_model.is_absolute():
            server_model = (_REPO_ROOT / server_model).resolve()
        if not server_model.is_dir():
            raise FileNotFoundError(
                "Server model directory not found: "
                f"{server_model} (ms-swift treats missing local paths as hub IDs; refusing to continue)"
            )

        root_image_dir = str(preflight.get("root_image_dir_resolved") or "").strip()
        if not root_image_dir:
            raise ValueError("Preflight missing root_image_dir_resolved")

        train_jsonl = str(preflight.get("train_jsonl_resolved") or "").strip()
        val_jsonl = str(preflight.get("val_jsonl_resolved") or "").strip()
        template_max_pixels = int(preflight["template_max_pixels"])

        if not train_jsonl:
            raise ValueError("custom.train_jsonl is required (resolved empty)")
        if not val_jsonl:
            raise ValueError("custom.val_jsonl is required (resolved empty)")

        server_tp = int(preflight.get("vllm_tensor_parallel_size") or 1)
        server_gpus, train_gpus, server_dp = validate_gpu_split(
            server_gpus_raw=server_gpus_raw,
            train_gpus_raw=train_gpus_raw,
            server_tp=server_tp,
        )

        server_torch_dtype = (
            str(preflight.get("server_torch_dtype") or "").strip() or "bfloat16"
        )
        vllm_enforce_eager = bool(preflight.get("vllm_enforce_eager"))

        vllm_max_model_len = int(preflight.get("vllm_max_model_len"))
        vllm_enable_lora = bool(preflight.get("vllm_enable_lora"))

        gpu_mem_raw = preflight.get("vllm_gpu_memory_utilization")
        vllm_gpu_memory_utilization = 0.75
        if gpu_mem_raw is not None:
            vllm_gpu_memory_utilization = float(gpu_mem_raw)

        template = str(preflight.get("server_template") or "").strip() or None

        template_max_length_raw = preflight.get("server_max_length")
        template_max_length = None
        if (
            template_max_length_raw is not None
            and str(template_max_length_raw).strip() != ""
        ):
            template_max_length = int(template_max_length_raw)

        truncation_strategy = (
            str(preflight.get("server_truncation_strategy") or "").strip() or None
        )

        _info(
            "========================================================================"
        )
        _info(
            "[PRECHECK] Validating JSONL contracts + max_pixels before launching GPUs"
        )
        _info(
            "========================================================================"
        )
        _info(f"[PRECHECK] train_jsonl: {train_jsonl}")
        _info(f"[PRECHECK] val_jsonl: {val_jsonl}")
        _info(f"[PRECHECK] max_pixels: {template_max_pixels} (expect 768*32*32=786432)")
        _info("[PRECHECK] multiple_of: 32")
        _info(
            "========================================================================"
        )

        _run_validate_jsonl(
            repo_root=_REPO_ROOT,
            jsonl_path=train_jsonl,
            max_pixels=template_max_pixels,
            mode="exists",
            n=0,
        )
        _run_validate_jsonl(
            repo_root=_REPO_ROOT,
            jsonl_path=train_jsonl,
            max_pixels=template_max_pixels,
            mode="open",
            n=256,
        )
        _run_validate_jsonl(
            repo_root=_REPO_ROOT,
            jsonl_path=val_jsonl,
            max_pixels=template_max_pixels,
            mode="open",
            n=0,
        )

        server_cmd = build_swift_rollout_cmd(
            server_model=server_model,
            base_url=base_url,
            server_torch_dtype=server_torch_dtype,
            vllm_dp=server_dp,
            vllm_tp=server_tp,
            vllm_enforce_eager=vllm_enforce_eager,
            vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
            vllm_max_model_len=vllm_max_model_len,
            vllm_enable_lora=vllm_enable_lora,
            template=template,
            template_max_pixels=template_max_pixels,
            template_max_length=template_max_length,
            truncation_strategy=truncation_strategy,
        )

        train_world_size = len(train_gpus)
        master_port_raw = os.environ.get("MASTER_PORT")
        if master_port_raw is None or str(master_port_raw).strip() == "":
            master_port = random.randint(10000, 65535)
        else:
            master_port = int(master_port_raw)

        master_addr = os.environ.get("MASTER_ADDR") or "127.0.0.1"

        learner_cmd = build_torchrun_cmd(
            config_path=config_path,
            num_gpus=train_world_size,
            master_addr=master_addr,
            master_port=master_port,
        )

        base_env = dict(os.environ)
        _ensure_pythonpath(base_env, repo_root=_REPO_ROOT)
        _apply_no_proxy(base_env, hosts=[base_url.host, "127.0.0.1", "localhost"])

        _set_default_env(
            base_env, "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
        )
        _set_default_env(base_env, "NCCL_ASYNC_ERROR_HANDLING", "1")
        _set_default_env(base_env, "TORCH_NCCL_TRACE_BUFFER_SIZE", "67108864")
        _set_default_env(base_env, "OMP_NUM_THREADS", "8")

        _set_default_env(base_env, "TORCH_NCCL_ENABLE_MONITORING", "1")
        _set_default_env(base_env, "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "120")
        _set_default_env(base_env, "TORCH_NCCL_DUMP_ON_TIMEOUT", "1")

        server_env = dict(base_env)
        server_env["CUDA_VISIBLE_DEVICES"] = server_gpus_raw
        server_env["ROOT_IMAGE_DIR"] = root_image_dir

        learner_env = dict(base_env)
        learner_env["CUDA_VISIBLE_DEVICES"] = train_gpus_raw
        learner_env["MASTER_ADDR"] = str(master_addr)
        learner_env["MASTER_PORT"] = str(master_port)
        _set_default_env(learner_env, "ROOT_IMAGE_DIR", root_image_dir)

        learner_env.update(
            {
                "COORDEXP_STAGE2_LAUNCHER": "scripts/train_stage2.sh",
                "COORDEXP_STAGE2_SERVER_BASE_URL": base_url.base_url,
                "COORDEXP_STAGE2_SERVER_MODEL": str(server_model),
                "COORDEXP_STAGE2_SERVER_TORCH_DTYPE": str(server_torch_dtype),
                "COORDEXP_STAGE2_SERVER_DP": str(server_dp),
                "COORDEXP_STAGE2_SERVER_TP": str(server_tp),
                "COORDEXP_STAGE2_SERVER_ENFORCE_EAGER": "true"
                if vllm_enforce_eager
                else "false",
                "COORDEXP_STAGE2_SERVER_GPU_MEMORY_UTILIZATION": str(
                    vllm_gpu_memory_utilization
                ),
                "COORDEXP_STAGE2_SERVER_MAX_MODEL_LEN": str(vllm_max_model_len),
                "COORDEXP_STAGE2_SERVER_ENABLE_LORA": "true"
                if vllm_enable_lora
                else "false",
                "COORDEXP_STAGE2_SERVER_GPUS": str(server_gpus_raw),
                "COORDEXP_STAGE2_LEARNER_GPUS": str(train_gpus_raw),
            }
        )

        _info(
            "========================================================================"
        )
        _info("  Stage-2 AB vLLM Server + Learner Launcher (Python)")
        _info(
            "========================================================================"
        )
        _info(f"[INFO] Config:      {config_path}")
        _info(f"[INFO] Server GPUs: {server_gpus_raw} (dp={server_dp}, tp={server_tp})")
        _info(f"[INFO] Train GPUs:  {train_gpus_raw} (world_size={train_world_size})")
        _info(f"[INFO] Server:      {base_url.host}:{base_url.port}")
        _info(f"[INFO] Model:       {server_model}")
        _info(f"[INFO] ROOT_IMAGE_DIR: {root_image_dir}")
        _info(f"[INFO] torch_dtype: {server_torch_dtype}")
        _info(f"[INFO] eager:       {vllm_enforce_eager}")
        _info(f"[INFO] max_model_len:{vllm_max_model_len}")
        _info(
            f"[INFO] enable_lora: {vllm_enable_lora} (full-sync-only; adapter sync unsupported)"
        )
        _info(
            "[INFO] nccl_monitor:%s heartbeat_s:%s dump_on_timeout:%s"
            % (
                base_env.get("TORCH_NCCL_ENABLE_MONITORING"),
                base_env.get("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"),
                base_env.get("TORCH_NCCL_DUMP_ON_TIMEOUT"),
            )
        )
        _info(
            "========================================================================"
        )

        server_proc: subprocess.Popen[str] | None = None
        learner_proc: subprocess.Popen[str] | None = None

        def cleanup() -> None:
            nonlocal server_proc, learner_proc
            if learner_proc is not None:
                _terminate_process_group(learner_proc, label="learner")
            if server_proc is not None:
                _terminate_process_group(server_proc, label="vLLM server")

        def _on_signal(signum: int, _frame: Any) -> None:
            sig = signal.Signals(signum).name
            _info(f"Received {sig}; shutting down learner + vLLM server...")
            cleanup()
            raise SystemExit(130)

        signal.signal(signal.SIGINT, _on_signal)
        signal.signal(signal.SIGTERM, _on_signal)

        try:
            _info(
                f"[RUN] (cwd={_REPO_ROOT}) CUDA_VISIBLE_DEVICES={server_gpus_raw} {_format_cmd(server_cmd)}"
            )
            health_url = f"{base_url.base_url}/health/"
            world_url = f"{base_url.base_url}/get_world_size/"

            server_proc = launch_swift_rollout_server(
                server_cmd=server_cmd,
                server_env=server_env,
                repo_root=_REPO_ROOT,
                health_url=health_url,
                world_url=world_url,
                wait_timeout_s=wait_timeout_s,
                wait_interval_s=wait_interval_s,
                expected_world_size=len(server_gpus),
            )

            _info(f"vLLM server is ready. world_size: {len(server_gpus)}")

            _info(
                f"[RUN] (cwd={_REPO_ROOT}) CUDA_VISIBLE_DEVICES={train_gpus_raw} {_format_cmd(learner_cmd)}"
            )
            learner_proc = subprocess.Popen(
                learner_cmd,
                cwd=str(_REPO_ROOT),
                env=learner_env,
                text=True,
                start_new_session=True,
            )
            rc = int(learner_proc.wait())
            return rc
        finally:
            cleanup()
    except SystemExit:
        raise
    except TimeoutError as exc:
        _die(str(exc), rc=1)
    except RuntimeError as exc:
        _die(str(exc), rc=1)
    except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
        _die(str(exc), rc=2)
    except subprocess.CalledProcessError as exc:
        _die(f"Subprocess failed (rc={exc.returncode}): {exc}")
    except Exception as exc:
        _die(f"{exc.__class__.__name__}: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
