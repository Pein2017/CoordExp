from __future__ import annotations

import copy
import importlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def attach_encoded_sample_cache_run_metadata(
    meta: dict[str, Any],
    *,
    train_cache_info: Mapping[str, Any] | None,
    eval_cache_info: Mapping[str, Any] | None,
) -> None:
    encoded_sample_cache: dict[str, Any] = {}
    if train_cache_info is not None:
        encoded_sample_cache["train"] = copy.deepcopy(dict(train_cache_info))
    if eval_cache_info is not None:
        encoded_sample_cache["eval"] = copy.deepcopy(dict(eval_cache_info))
    if encoded_sample_cache:
        meta["encoded_sample_cache"] = encoded_sample_cache


def _safe_module_info(module_name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        return {"error": f"{exc.__class__.__name__}: {exc}"}

    version = getattr(module, "__version__", None)
    module_file = getattr(module, "__file__", None)

    info: dict[str, Any] = {}
    if version is not None:
        info["version"] = str(version)
    if module_file:
        info["file"] = str(module_file)
    return info


def _find_git_repo_root(start_dir: Path) -> Path | None:
    for parent in [start_dir, *start_dir.parents]:
        if (parent / ".git").exists():
            return parent
    return None


def _safe_git_state(repo_root: Path) -> dict[str, Any]:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.STDOUT,
            text=True,
        ).strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.STDOUT,
            text=True,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=str(repo_root),
            stderr=subprocess.STDOUT,
            text=True,
        ).strip()
        status_lines = [line for line in status.splitlines() if line.strip()]
        dirty = bool(status_lines)
        return {
            "sha": sha,
            "branch": branch,
            "dirty": dirty,
            "status_porcelain": status_lines,
        }
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"error": f"{exc.__class__.__name__}: {exc}"}


def collect_dependency_provenance() -> dict[str, Any]:
    """Collect upstream dependency provenance for paper-ready reproducibility."""

    deps: dict[str, Any] = {
        "python": sys.version,
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV")
        or os.environ.get("CONDA_ENV")
        or None,
        "transformers": _safe_module_info("transformers"),
        "torch": _safe_module_info("torch"),
        "vllm": _safe_module_info("vllm"),
        "swift": _safe_module_info("swift"),
    }

    swift_info = deps.get("swift")
    swift_file = None
    if isinstance(swift_info, Mapping):
        swift_file = swift_info.get("file")
    if isinstance(swift_file, str) and swift_file:
        repo_root = _find_git_repo_root(Path(swift_file).resolve().parent)
        if repo_root is not None:
            deps["ms_swift"] = {
                "repo_root": str(repo_root),
                **_safe_git_state(repo_root),
            }

    return deps


STAGE2_LAUNCHER_METADATA_ENV_KEYS = [
    "COORDEXP_STAGE2_LAUNCHER",
    "COORDEXP_STAGE2_SERVER_BASE_URL",
    "COORDEXP_STAGE2_SERVER_MODEL",
    "COORDEXP_STAGE2_SERVER_TORCH_DTYPE",
    "COORDEXP_STAGE2_SERVER_DP",
    "COORDEXP_STAGE2_SERVER_TP",
    "COORDEXP_STAGE2_SERVER_ENFORCE_EAGER",
    "COORDEXP_STAGE2_SERVER_GPU_MEMORY_UTILIZATION",
    "COORDEXP_STAGE2_SERVER_MAX_MODEL_LEN",
    "COORDEXP_STAGE2_SERVER_ENABLE_LORA",
    "COORDEXP_STAGE2_SERVER_GPUS",
    "COORDEXP_STAGE2_LEARNER_GPUS",
]


def collect_launcher_metadata_from_env() -> dict[str, str]:
    meta: dict[str, str] = {}
    for key in STAGE2_LAUNCHER_METADATA_ENV_KEYS:
        value = os.environ.get(key)
        if value is None:
            continue
        meta[key] = value
    return meta


def build_run_metadata_payload(
    *,
    output_dir: str | Path,
    config_path: str,
    base_config_path: str | None,
    run_name: str,
    dataset_seed: int,
    repo_root: Path,
    manifest_files: Mapping[str, Any] | None,
    train_cache_info: Mapping[str, Any] | None,
    eval_cache_info: Mapping[str, Any] | None,
) -> dict[str, Any]:
    git_state = _safe_git_state(repo_root)
    status_lines_raw = git_state.get("status_porcelain", [])
    if isinstance(status_lines_raw, list):
        status_lines = [str(line) for line in status_lines_raw[:200]]
    else:
        status_lines = []

    meta: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path or ""),
        "base_config": str(base_config_path or ""),
        "run_name": str(run_name or ""),
        "output_dir": str(output_dir),
        "git_sha": git_state.get("sha"),
        "git_branch": git_state.get("branch"),
        "git_dirty": git_state.get("dirty"),
        "git_status_porcelain": status_lines,
        "dataset_seed": int(dataset_seed),
        "upstream": collect_dependency_provenance(),
    }

    if manifest_files is not None:
        meta["run_manifest_files"] = dict(manifest_files)

    launcher_meta = collect_launcher_metadata_from_env()
    if launcher_meta:
        meta["launcher"] = launcher_meta

    attach_encoded_sample_cache_run_metadata(
        meta,
        train_cache_info=train_cache_info,
        eval_cache_info=eval_cache_info,
    )
    return meta


def write_run_metadata_file_from_payload(
    *,
    output_dir: str | Path,
    payload: Mapping[str, Any],
) -> Path:
    out_path = Path(str(output_dir)) / "run_metadata.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return out_path


def write_run_metadata_file(
    *,
    output_dir: str | Path,
    config_path: str,
    base_config_path: str | None,
    run_name: str,
    dataset_seed: int,
    repo_root: Path,
    manifest_files: Mapping[str, Any] | None,
    train_cache_info: Mapping[str, Any] | None,
    eval_cache_info: Mapping[str, Any] | None,
) -> Path:
    payload = build_run_metadata_payload(
        output_dir=output_dir,
        config_path=config_path,
        base_config_path=base_config_path,
        run_name=run_name,
        dataset_seed=dataset_seed,
        repo_root=repo_root,
        manifest_files=manifest_files,
        train_cache_info=train_cache_info,
        eval_cache_info=eval_cache_info,
    )
    return write_run_metadata_file_from_payload(
        output_dir=output_dir,
        payload=payload,
    )
