#!/usr/bin/env python
"""Unified pipeline runner for CoordExp inference/eval/vis (YAML-first).

Primary usage:
  python scripts/run_infer.py --config configs/infer/<exp>.yaml

The YAML config is treated as a single file (no extends/inherit, no variable
interpolation). Legacy CLI flags are supported during a transition period:
- If both --config and legacy flags are provided, legacy flags override YAML.

For legacy (flag-only) inference runs (no YAML):
  python scripts/run_infer.py \
      --gt_jsonl <path> \
      --model_checkpoint <ckpt> \
      --mode coord|text|auto \
      --out <path/to/gt_vs_pred.jsonl>
"""

from __future__ import annotations

import argparse
import contextlib
import os
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path
from typing import Any, Dict

import requests
import yaml

from src.infer.pipeline import run_pipeline


def _add_legacy_infer_flags(ap: argparse.ArgumentParser, *, required: bool) -> None:
    ap.add_argument("--gt_jsonl", required=required, help="Path to ground-truth JSONL")
    ap.add_argument("--model_checkpoint", required=required, help="Checkpoint path")
    ap.add_argument(
        "--mode",
        required=required,
        choices=["coord", "text", "auto"],
        help="Model/GT mode (coord-token vs pixel GT), or auto-detect",
    )
    ap.add_argument(
        "--pred-coord-mode",
        choices=["auto", "pixel", "norm1000"],
        default=None if not required else "auto",
        help="Override how prediction coords are interpreted before scaling",
    )
    ap.add_argument("--device", default=None if not required else "cuda:0")
    ap.add_argument(
        "--limit", type=int, default=None if not required else 0, help="0 = all"
    )
    ap.add_argument(
        "--detect-samples",
        type=int,
        default=None if not required else 128,
        help="When mode=auto, how many GT records to scan",
    )

    # Artifacts (legacy-only; in YAML mode these map to artifacts.* overrides)
    ap.add_argument(
        "--out",
        default=None if not required else "gt_vs_pred.jsonl",
        help="Output JSONL path (defaults to gt_vs_pred.jsonl)",
    )
    ap.add_argument(
        "--summary",
        default=None,
        help="Optional summary path (defaults to <out_dir>/summary.json)",
    )

    # Backend
    ap.add_argument(
        "--backend",
        choices=["hf", "vllm"],
        default=None if not required else "hf",
        help="Generation backend: hf (default) or vllm",
    )
    ap.add_argument(
        "--vllm-base-url",
        default=None,
        help="(vllm backend) OpenAI-compatible base URL, e.g. http://127.0.0.1:8000",
    )
    ap.add_argument(
        "--vllm-model",
        default=None,
        help="(vllm backend) Served model name; defaults to --model_checkpoint",
    )

    # Generation flags
    ap.add_argument("--temperature", type=float, default=None if not required else 0.01)
    ap.add_argument("--top_p", type=float, default=None if not required else 0.95)
    ap.add_argument(
        "--max_new_tokens", type=int, default=None if not required else 1024
    )
    ap.add_argument(
        "--repetition_penalty", type=float, default=None if not required else 1.05
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=None if not required else 1,
        help="Generation micro-batch size (HF batched generate, vLLM request micro-batch)",
    )
    ap.add_argument("--seed", type=int, default=None)


def _yaml_overrides_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    o: Dict[str, Any] = {}

    def _set(k: str, v: Any) -> None:
        if v is not None:
            o[k] = v

    # Run/artifacts overrides
    _set("artifacts.gt_vs_pred_jsonl", args.out)
    _set("artifacts.summary_json", args.summary)

    # Infer overrides
    _set("infer.gt_jsonl", args.gt_jsonl)
    _set("infer.model_checkpoint", args.model_checkpoint)
    _set("infer.mode", args.mode)
    _set("infer.pred_coord_mode", args.pred_coord_mode)
    _set("infer.device", args.device)
    _set("infer.limit", args.limit)
    _set("infer.detect_samples", args.detect_samples)

    # Backend overrides
    _set("infer.backend.type", args.backend)
    _set("infer.backend.base_url", args.vllm_base_url)
    _set("infer.backend.model", args.vllm_model)

    # Generation overrides
    _set("infer.generation.temperature", args.temperature)
    _set("infer.generation.top_p", args.top_p)
    _set("infer.generation.max_new_tokens", args.max_new_tokens)
    _set("infer.generation.repetition_penalty", args.repetition_penalty)
    _set("infer.generation.batch_size", args.batch_size)
    _set("infer.generation.seed", args.seed)

    return o


def _run_legacy_infer(args: argparse.Namespace) -> None:
    from src.infer import GenerationConfig, InferenceConfig, InferenceEngine

    gen_cfg = GenerationConfig(
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        max_new_tokens=int(args.max_new_tokens),
        repetition_penalty=float(args.repetition_penalty),
        batch_size=int(args.batch_size),
        seed=args.seed,
    )

    backend_cfg: Dict[str, Any] = {}
    if args.backend == "vllm":
        if args.vllm_base_url:
            backend_cfg["base_url"] = str(args.vllm_base_url)
        if args.vllm_model:
            backend_cfg["model"] = str(args.vllm_model)

    inf_cfg = InferenceConfig(
        gt_jsonl=str(args.gt_jsonl),
        model_checkpoint=str(args.model_checkpoint),
        mode=str(args.mode),
        pred_coord_mode=str(args.pred_coord_mode),
        out_path=str(args.out),
        summary_path=str(args.summary) if args.summary else None,
        device=str(args.device),
        limit=int(args.limit),
        backend_type=str(args.backend),
        backend=backend_cfg,
        detect_samples=int(args.detect_samples),
    )

    engine = InferenceEngine(inf_cfg, gen_cfg)
    out_path, summary_path = engine.infer()
    print(f"Wrote predictions to {out_path} and summary to {summary_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Unified inference pipeline runner")
    ap.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML pipeline config (YAML-first).",
    )

    # If --config is provided, legacy flags become optional overrides.
    _add_legacy_infer_flags(ap, required=False)

    args = ap.parse_args()

    if args.config is not None:
        overrides = _yaml_overrides_from_args(args)
        with _maybe_launch_vllm_server(Path(args.config), overrides):
            artifacts = run_pipeline(config_path=Path(args.config), overrides=overrides)
        print(f"Pipeline complete. run_dir={artifacts.run_dir}")
        return

    # Legacy flag-only mode: require the classic inputs.
    ap2 = argparse.ArgumentParser(description="Legacy inference (no YAML)")
    _add_legacy_infer_flags(ap2, required=True)
    args2 = ap2.parse_args()
    _run_legacy_infer(args2)


def _apply_overrides(cfg: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Apply dotted overrides in-place (shallow utility; no schema awareness)."""

    for dotted_key, value in (overrides or {}).items():
        cursor = cfg
        parts = str(dotted_key).split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return cfg


def _load_cfg_with_overrides(
    config_path: Path, overrides: Dict[str, Any]
) -> Dict[str, Any]:
    raw = yaml.safe_load(config_path.read_text()) if config_path.exists() else {}
    if raw is None:
        raw = {}
    return _apply_overrides(raw, overrides)


def _base_url_to_host_port(url: str) -> tuple[str, int]:
    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8000
    return host, port


def _models_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    return (base + "/models") if base.endswith("/v1") else (base + "/v1/models")


def _is_local_base_url(base_url: str) -> bool:
    host, _ = _base_url_to_host_port(base_url)
    return host in {"127.0.0.1", "localhost", "0.0.0.0"}


def _server_cmd_from_cfg(
    model: str, host: str, port: int, server_opts: Dict[str, Any]
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model,
        "--host",
        host,
        "--port",
        str(port),
        "--trust-remote-code",
    ]

    def _add(flag: str, val: Any) -> None:
        if val is None:
            return
        cmd.extend([flag, str(val)])

    def _add_bool(flag: str, enabled: bool | None) -> None:
        if enabled is None:
            return
        cmd.append(flag if enabled else flag.replace("--", "--no-"))

    _add("--tensor-parallel-size", server_opts.get("vllm_tensor_parallel_size"))
    _add("--data-parallel-size", server_opts.get("vllm_data_parallel_size"))
    _add("--gpu-memory-utilization", server_opts.get("vllm_gpu_memory_utilization"))
    _add("--max-model-len", server_opts.get("vllm_max_model_len"))
    _add("--max-num-batched-tokens", server_opts.get("max_num_batched_tokens"))
    _add("--max-num-seqs", server_opts.get("vllm_max_num_seqs"))
    _add_bool("--enable-prefix-caching", server_opts.get("vllm_enable_prefix_caching"))

    return cmd


@contextlib.contextmanager
def _maybe_launch_vllm_server(config_path: Path, overrides: Dict[str, Any]):
    """
    Auto-launch a local vLLM server when infer.backend.type=vllm targets localhost
    and no server is reachable. Keeps CLI unchanged; tears down after pipeline.
    """

    cfg = _load_cfg_with_overrides(config_path, overrides)
    infer_cfg = cfg.get("infer") or {}
    backend_cfg = infer_cfg.get("backend") or {}
    backend_type = str(backend_cfg.get("type") or "").strip().lower()

    mode = str(backend_cfg.get("mode") or "server").strip().lower()
    if mode == "local":
        # In-process vLLM mode (no HTTP server) - nothing to auto-launch.
        yield None
        return

    base_url = str(
        backend_cfg.get("base_url") or os.environ.get("VLLM_BASE_URL") or ""
    ).strip()

    if backend_type != "vllm":
        yield None
        return

    if not base_url:
        # Let downstream validation raise the canonical error.
        yield None
        return

    if not _is_local_base_url(base_url):
        yield None
        return

    auto_launch_raw = backend_cfg.get("auto_launch", True)
    if isinstance(auto_launch_raw, str):
        auto_launch = auto_launch_raw.strip().lower() not in {"0", "false", "no", "off"}
    else:
        auto_launch = bool(auto_launch_raw)

    if not auto_launch:
        # Respect user choice: do not spawn a local server process.
        yield None
        return

    # Fast preflight: if server already up, skip auto-launch.
    try:
        resp = requests.get(_models_url(base_url), timeout=1.5)
        if resp.status_code < 400:
            yield None
            return
    except Exception:
        pass

    host, port = _base_url_to_host_port(base_url)
    model = str(
        backend_cfg.get("model") or infer_cfg.get("model_checkpoint") or ""
    ).strip()
    if not model:
        yield None
        return

    server_opts = backend_cfg.get("server_options") or {}
    cmd = _server_cmd_from_cfg(model, host, port, server_opts)

    log_path = Path("temp/vllm_server.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log_fp:
        proc = subprocess.Popen(cmd, stdout=log_fp, stderr=subprocess.STDOUT)

    try:
        deadline = time.time() + float(server_opts.get("startup_timeout_s", 120))
        last_exc: Exception | None = None
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"vLLM server exited early with code {proc.returncode}; log: {log_path}"
                )
            try:
                resp = requests.get(_models_url(base_url), timeout=2.0)
                if resp.status_code < 400:
                    break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
            time.sleep(1.0)
        else:
            raise RuntimeError(
                f"Timed out waiting for vLLM server on {base_url}; "
                f"last error={last_exc}; see log {log_path}"
            )

        yield {"process": proc, "log": log_path}
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    main()
