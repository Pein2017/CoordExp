from __future__ import annotations

import json
import os
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any, Mapping

from src.config.strict_dataclass import dataclass_asdict_no_none
from src.utils.logger import get_logger

logger = get_logger(__name__)

RUN_MANIFEST_SCHEMA_VERSION = 1


_DEFAULT_ENV_KEYS: list[str] = [
    # Core data path contract (learner + rollout server must agree).
    "ROOT_IMAGE_DIR",
    # GPU placement / topology diagnostics.
    "CUDA_VISIBLE_DEVICES",
    "NCCL_DEBUG",
    # Common runtime knobs that can affect determinism/perf.
    "TOKENIZERS_PARALLELISM",
    "PYTHONHASHSEED",
    "OMP_NUM_THREADS",
    # HF cache locations (affects offline reproducibility + storage).
    "HF_HOME",
    "TRANSFORMERS_CACHE",
    "TORCH_HOME",
    # Distributed training (torchrun/SLURM).
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "SLURM_JOB_ID",
    "SLURM_PROCID",
    "SLURM_LOCALID",
    "SLURM_NTASKS",
    "SLURM_NTASKS_PER_NODE",
    "SLURM_NODELIST",
    # Environment identity.
    "CONDA_DEFAULT_ENV",
    "CONDA_PREFIX",
]


def collect_runtime_env_metadata(
    *, keys: list[str] | None = None
) -> dict[str, str]:
    """Collect a small, high-signal subset of environment metadata.

    This is intentionally a whitelist (not a full env dump) to avoid leaking
    tokens/secrets into run artifacts.
    """

    selected = list(keys) if keys is not None else list(_DEFAULT_ENV_KEYS)
    out: dict[str, str] = {}
    for key in selected:
        value = os.environ.get(key)
        if value is None:
            continue
        value = str(value)
        if value.strip() == "":
            continue
        out[str(key)] = value
    return out


def _to_jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _to_jsonable(dataclass_asdict_no_none(value))
    if isinstance(value, Mapping):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, set):
        return [_to_jsonable(v) for v in sorted(value, key=lambda x: str(x))]
    # Defensive fallback: avoid hard failure on a new config type.
    return str(value)


def serialize_resolved_training_config(training_config: Any) -> dict[str, Any]:
    if is_dataclass(training_config):
        resolved = dataclass_asdict_no_none(training_config)
    elif isinstance(training_config, Mapping):
        resolved = dict(training_config)
    else:
        raise TypeError(
            "training_config must be a dataclass or mapping; "
            f"got {type(training_config).__name__}"
        )
    jsonable = _to_jsonable(resolved)
    if not isinstance(jsonable, dict):
        raise TypeError("Resolved training config serialization must yield a dict")
    return jsonable


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_to_jsonable(dict(payload)), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_run_manifest_files(
    *,
    output_dir: str | Path,
    training_config: Any,
    config_path: str,
    base_config_path: str | None,
    dataset_seed: int,
    env_keys: list[str] | None = None,
) -> dict[str, str]:
    """Write required reproducibility artifacts under `output_dir`.

    Contract: these files are required for paper-ready reproducibility and must
    be written before training starts.
    """

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    resolved_cfg = serialize_resolved_training_config(training_config)
    resolved_path = out_dir / "resolved_config.json"
    _write_json(
        resolved_path,
        {
            "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
            "config_path": str(config_path),
            "base_config_path": str(base_config_path or ""),
            "dataset_seed": int(dataset_seed),
            "resolved": resolved_cfg,
        },
    )

    env_path = out_dir / "runtime_env.json"
    _write_json(
        env_path,
        {
            "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
            "env": collect_runtime_env_metadata(keys=env_keys),
        },
    )

    # Best-effort: keep a copy of the exact YAML sources used to build the resolved config.
    # If these copies fail, training can still proceed with the resolved snapshot above.
    try:
        cfg_src = Path(str(config_path))
        if cfg_src.is_file():
            (out_dir / "config_source.yaml").write_text(
                cfg_src.read_text(encoding="utf-8"), encoding="utf-8"
            )
    except Exception as exc:
        logger.warning("Failed to persist config_source.yaml: %r", exc)

    if base_config_path:
        try:
            base_src = Path(str(base_config_path))
            if base_src.is_file():
                (out_dir / "base_config_source.yaml").write_text(
                    base_src.read_text(encoding="utf-8"), encoding="utf-8"
                )
        except Exception as exc:
            logger.warning("Failed to persist base_config_source.yaml: %r", exc)

    return {
        "resolved_config": str(resolved_path.name),
        "runtime_env": str(env_path.name),
    }

