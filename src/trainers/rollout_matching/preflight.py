"""Helpers that expose the Stage-2 rollout preflight contract."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from dataclasses import is_dataclass
from typing import Any, Optional

from src.config.loader import ConfigLoader
from src.config.schema import TrainingConfig
from src.config.strict_dataclass import dataclass_asdict_no_none

RolloutContract = dict[str, Any]
Stage2LauncherPreflight = dict[str, Any]


__all__ = [
    "RolloutContract",
    "Stage2LauncherPreflight",
    "build_rollout_matching_contract",
    "build_stage2_launcher_preflight",
    "resolve_rollout_matching_contract",
    "resolve_stage2_launcher_preflight",
]


def resolve_rollout_matching_contract(
    config_path: str, base_config_path: Optional[str] = None
) -> RolloutContract:
    """Load a config via *ConfigLoader* and build the rollout contract."""

    _, training_config = ConfigLoader.load_training_config(config_path, base_config_path)
    return build_rollout_matching_contract(training_config)


def resolve_stage2_launcher_preflight(
    config_path: str, base_config_path: Optional[str] = None
) -> Stage2LauncherPreflight:
    """Load a config via *ConfigLoader* and build launcher preflight settings."""

    _, training_config = ConfigLoader.load_training_config(config_path, base_config_path)
    return build_stage2_launcher_preflight(training_config)


def build_rollout_matching_contract(training_config: TrainingConfig) -> RolloutContract:
    """Normalize the rollout sub-namespace into the preflight contract."""

    rollout_cfg = _extract_rollout_mapping(training_config)

    backend_raw = rollout_cfg.get("rollout_backend")
    if backend_raw is None:
        raise ValueError(
            "rollout_matching.rollout_backend is required for Stage-2 rollout preflight."
        )
    backend = str(backend_raw).strip()
    if backend != "vllm":
        raise ValueError(
            "rollout_matching.rollout_backend must be 'vllm' for server-mode rollout launch."
        )

    vllm_cfg = _extract_required_mapping(rollout_cfg, "rollout_matching.vllm")

    mode_raw = vllm_cfg.get("mode")
    mode = None if mode_raw is None else str(mode_raw).strip()
    if not mode:
        raise ValueError("rollout_matching.vllm.mode must be provided (server or colocate).")
    if mode != "server":
        raise ValueError(
            "rollout_matching.vllm.mode must be 'server' for Stage-2 vLLM server-mode rollout."
        )

    server_cfg = _extract_required_mapping(vllm_cfg, "rollout_matching.vllm.server")

    servers = server_cfg.get("servers")
    if not isinstance(servers, list) or not servers:
        raise ValueError(
            "rollout_matching.vllm.server.servers must be a non-empty list for server-mode launches."
        )

    base_urls: list[str] = []
    for idx, server_entry in enumerate(servers):
        if not isinstance(server_entry, Mapping):
            raise TypeError(
                "rollout_matching.vllm.server.servers[%d] must be a mapping." % idx
            )
        base_url_raw = server_entry.get("base_url")
        if base_url_raw is None:
            raise ValueError(
                "rollout_matching.vllm.server.servers[%d].base_url is required." % idx
            )
        if not isinstance(base_url_raw, str):
            raise TypeError(
                "rollout_matching.vllm.server.servers[%d].base_url must be a string." % idx
            )
        base_url = base_url_raw.strip()
        if not base_url:
            raise ValueError(
                "rollout_matching.vllm.server.servers[%d].base_url must be non-empty." % idx
            )
        base_urls.append(base_url)

    return {
        "rollout_backend": backend,
        "vllm_mode": mode,
        "server_base_urls": base_urls,
    }


def build_stage2_launcher_preflight(
    training_config: TrainingConfig,
) -> Stage2LauncherPreflight:
    """Build the single preflight payload consumed by `scripts/train_stage2.sh`."""

    rollout_contract = build_rollout_matching_contract(training_config)
    rollout_cfg = _extract_rollout_mapping(training_config)

    model_raw = training_config.model.get("model")
    if not isinstance(model_raw, str) or not model_raw.strip():
        raise ValueError("model.model must be set (rollout model path).")
    model_path = model_raw.strip()

    train_jsonl_raw = getattr(training_config.custom, "train_jsonl", None)
    if not isinstance(train_jsonl_raw, str) or not train_jsonl_raw.strip():
        raise ValueError(
            "custom.train_jsonl must be set to resolve ROOT_IMAGE_DIR for server-mode rollouts."
        )
    train_jsonl_path = Path(train_jsonl_raw.strip())
    if not train_jsonl_path.is_absolute():
        train_jsonl_path = (Path.cwd() / train_jsonl_path).resolve()
    root_image_dir = train_jsonl_path.parent

    vllm_cfg = _extract_required_mapping(rollout_cfg, "rollout_matching.vllm")
    max_model_len_raw = vllm_cfg.get("max_model_len")
    if max_model_len_raw is None:
        raise ValueError(
            "rollout_matching.vllm.max_model_len is required for server preflight."
        )
    try:
        max_model_len = int(max_model_len_raw)
    except Exception as exc:
        raise TypeError("rollout_matching.vllm.max_model_len must be an int") from exc
    if max_model_len <= 0:
        raise ValueError("rollout_matching.vllm.max_model_len must be > 0")

    enable_lora = bool(vllm_cfg.get("enable_lora", False))

    return {
        "rollout_backend": rollout_contract["rollout_backend"],
        "vllm_mode": rollout_contract["vllm_mode"],
        "server_base_urls": list(rollout_contract["server_base_urls"]),
        "server_model": model_path,
        "root_image_dir_resolved": str(root_image_dir),
        "vllm_max_model_len": max_model_len,
        "vllm_enable_lora": enable_lora,
    }


def _extract_rollout_mapping(training_config: TrainingConfig) -> Mapping[str, Any]:
    canonical = getattr(training_config, "rollout_matching", None)
    if canonical is None:
        raise ValueError(
            "rollout_matching configuration is required (see rollout_matching.*)."
        )

    if is_dataclass(canonical):
        return dataclass_asdict_no_none(canonical)

    if not isinstance(canonical, Mapping):
        raise TypeError("rollout_matching must be a mapping when provided.")
    return dict(canonical)


def _extract_required_mapping(payload: Mapping[str, Any], key_path: str) -> Mapping[str, Any]:
    key_name = key_path.split(".")[-1]
    value = payload.get(key_name)
    if value is None:
        raise ValueError(f"{key_path} is required for rollout preflight.")
    if not isinstance(value, Mapping):
        raise TypeError(f"{key_path} must be a mapping.")
    return dict(value)
