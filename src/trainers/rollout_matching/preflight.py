"""Helpers that expose the Stage-2 rollout preflight contract."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from dataclasses import is_dataclass
from urllib.parse import urlparse
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
    """Load a config via *ConfigLoader* and build the rollout contract.

    This preflight intentionally avoids building ms-swift TrainArguments so it can run
    in CPU-only environments and from arbitrary working directories.
    """

    training_config = ConfigLoader.load_materialized_training_config(
        config_path, base_config_path
    )
    return build_rollout_matching_contract(training_config)


def resolve_stage2_launcher_preflight(
    config_path: str, base_config_path: Optional[str] = None
) -> Stage2LauncherPreflight:
    """Load a config via *ConfigLoader* and build launcher preflight settings.

    NOTE: This function must be safe to call from arbitrary working directories.
    The launcher passes absolute config paths, but JSONL/model paths inside the
    YAML are frequently repo-relative (e.g. public_data/*).

    This preflight intentionally avoids building ms-swift TrainArguments so it can run
    in CPU-only environments and without triggering hub downloads.
    """

    training_config = ConfigLoader.load_materialized_training_config(
        config_path, base_config_path
    )
    return build_stage2_launcher_preflight(training_config, config_path=config_path)


def build_rollout_matching_contract(training_config: TrainingConfig) -> RolloutContract:
    """Normalize the rollout sub-namespace into the preflight contract."""

    rollout_cfg = _extract_rollout_mapping(training_config)

    backend_raw = rollout_cfg.get("rollout_backend")
    if backend_raw is None:
        raise ValueError(
            "rollout_matching.rollout_backend is required for Stage-2 rollout preflight."
        )
    backend = str(backend_raw).strip().lower()
    if backend not in {"hf", "vllm"}:
        raise ValueError(
            "rollout_matching.rollout_backend must be one of {'hf', 'vllm'} for Stage-2 rollout preflight."
        )

    eval_backend_raw = rollout_cfg.get("eval_rollout_backend")
    if eval_backend_raw is None:
        raise ValueError(
            "rollout_matching.eval_rollout_backend must be set to 'vllm' for Stage-2 rollout preflight."
        )
    eval_backend = str(eval_backend_raw).strip().lower()
    if eval_backend != "vllm":
        raise ValueError(
            "rollout_matching.eval_rollout_backend must be 'vllm' for Stage-2 rollout preflight."
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

        parsed = urlparse(base_url)
        host = str(parsed.hostname or "").strip().lower()
        if host == "0.0.0.0":
            raise ValueError(
                "rollout_matching.vllm.server.servers[%d].base_url must not use 0.0.0.0; "
                "use 127.0.0.1 or a routable host/IP instead." % idx
            )

        base_urls.append(base_url)

    return {
        "rollout_backend": backend,
        "eval_rollout_backend": eval_backend,
        "vllm_mode": mode,
        "server_base_urls": base_urls,
    }


def build_stage2_launcher_preflight(
    training_config: TrainingConfig,
    config_path: Optional[str] = None,
) -> Stage2LauncherPreflight:
    """Build the single preflight payload consumed by `scripts/train_stage2.sh`.

    This preflight is consumed by the bash launcher, so it must:
    - resolve repo-relative paths deterministically (independent of cwd)
    - fail fast with actionable diagnostics on objective-changing misconfig
    """

    rollout_contract = build_rollout_matching_contract(training_config)
    rollout_cfg = _extract_rollout_mapping(training_config)

    server_base_urls = list(rollout_contract["server_base_urls"])
    if len(server_base_urls) != 1:
        raise ValueError(
            "scripts/train_stage2.sh supports exactly 1 vLLM rollout server (it selects urls[0]). "
            "Set rollout_matching.vllm.server.servers to a single entry or use external orchestration. "
            f"Got {len(server_base_urls)} servers."
        )

    model_raw = training_config.model.get("model")
    if not isinstance(model_raw, str) or not model_raw.strip():
        raise ValueError("model.model must be set (rollout model path).")
    model_path = model_raw.strip()

    train_jsonl_raw = getattr(training_config.custom, "train_jsonl", None)
    if not isinstance(train_jsonl_raw, str) or not train_jsonl_raw.strip():
        raise ValueError(
            "custom.train_jsonl must be set to resolve ROOT_IMAGE_DIR for server-mode rollouts."
        )
    train_jsonl_path = _resolve_path_for_config(train_jsonl_raw.strip(), config_path)
    root_image_dir = train_jsonl_path.parent

    val_jsonl_raw = getattr(training_config.custom, "val_jsonl", None)
    val_jsonl_path = None
    if isinstance(val_jsonl_raw, str) and val_jsonl_raw.strip():
        val_jsonl_path = _resolve_path_for_config(val_jsonl_raw.strip(), config_path)

    max_pixels_raw = training_config.template.get("max_pixels")
    if max_pixels_raw is None:
        raise ValueError(
            "template.max_pixels must be set for Stage-2 preflight (we treat it as a hard input constraint)."
        )
    try:
        template_max_pixels = int(max_pixels_raw)
    except Exception as exc:
        raise TypeError("template.max_pixels must be an int") from exc
    if template_max_pixels <= 0:
        raise ValueError("template.max_pixels must be > 0")

    vllm_cfg = _extract_required_mapping(rollout_cfg, "rollout_matching.vllm")
    max_model_len_raw = vllm_cfg.get("max_model_len")
    if max_model_len_raw is None:
        raise ValueError(
            "rollout_matching.vllm.max_model_len is required for server preflight."
        )
    try:
        max_model_len = int(max_model_len_raw)
    except (TypeError, ValueError) as exc:
        raise TypeError("rollout_matching.vllm.max_model_len must be an int") from exc
    if max_model_len <= 0:
        raise ValueError("rollout_matching.vllm.max_model_len must be > 0")

    enable_lora = bool(vllm_cfg.get("enable_lora", False))

    tensor_parallel_size = 1
    tp_raw = vllm_cfg.get("tensor_parallel_size")
    if tp_raw is not None:
        try:
            tensor_parallel_size = int(tp_raw)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "rollout_matching.vllm.tensor_parallel_size must be an int when provided"
            ) from exc
    if tensor_parallel_size <= 0:
        raise ValueError(
            "rollout_matching.vllm.tensor_parallel_size must be > 0 when provided"
        )

    enforce_eager = bool(vllm_cfg.get("enforce_eager", False))

    gpu_memory_utilization = None
    gpu_mem_raw = vllm_cfg.get("gpu_memory_utilization")
    if gpu_mem_raw is not None:
        try:
            gpu_memory_utilization = float(gpu_mem_raw)
        except Exception as exc:
            raise TypeError(
                "rollout_matching.vllm.gpu_memory_utilization must be a float when provided"
            ) from exc
        if not (0.0 < gpu_memory_utilization <= 1.0):
            raise ValueError(
                "rollout_matching.vllm.gpu_memory_utilization must be in (0, 1] when provided"
            )

    server_torch_dtype = None
    dtype_raw = training_config.model.get("dtype")
    if dtype_raw is None:
        dtype_raw = training_config.model.get("torch_dtype")
    if dtype_raw is not None:
        if not isinstance(dtype_raw, str):
            raise TypeError("model.dtype/torch_dtype must be a string when provided")
        server_torch_dtype = dtype_raw.strip() or None

    server_template = ""
    template_raw = None
    try:
        template_raw = training_config.template.get("template")
    except Exception:
        template_raw = None
    if isinstance(template_raw, str) and template_raw.strip():
        server_template = template_raw.strip()

    server_max_length = None
    gml_raw = getattr(training_config, "global_max_length", None)
    if gml_raw is not None:
        try:
            server_max_length = int(gml_raw)
        except Exception as exc:
            raise TypeError("global_max_length must be an int when provided") from exc
        if server_max_length <= 0:
            raise ValueError(
                f"global_max_length must be a positive int when provided, got {gml_raw!r}"
            )

    server_truncation_strategy = ""
    trunc_raw = None
    try:
        trunc_raw = training_config.template.get("truncation_strategy")
    except Exception:
        trunc_raw = None
    if isinstance(trunc_raw, str) and trunc_raw.strip():
        trunc = trunc_raw.strip().lower()
        # ms-swift CLI uses truncation_strategy='delete' to mean Template(truncation_strategy='raise').
        if trunc == "raise":
            server_truncation_strategy = "delete"
        else:
            server_truncation_strategy = trunc

    # Optional vLLM EngineArgs passthrough for server launches.
    #
    # This is the only reliable way to keep vLLM server-side prompt tokenization
    # identical to local teacher-forcing encode for Qwen3-VL, since the HF
    # processor defaults to do_resize=True (which can upscale small images and
    # change the number of `<|image_pad|>` tokens).
    vllm_engine_kwargs: dict[str, Any] = {}
    mm_processor_kwargs_raw = vllm_cfg.get("mm_processor_kwargs")
    if mm_processor_kwargs_raw is not None:
        if not isinstance(mm_processor_kwargs_raw, Mapping):
            raise TypeError(
                "rollout_matching.vllm.mm_processor_kwargs must be a mapping when provided"
            )
        mm_processor_kwargs: dict[str, Any] = dict(mm_processor_kwargs_raw)
        allowed_keys = {"do_resize"}
        unknown_keys = set(mm_processor_kwargs.keys()) - set(allowed_keys)
        if unknown_keys:
            raise ValueError(
                "rollout_matching.vllm.mm_processor_kwargs contains unsupported keys: "
                f"{sorted(unknown_keys)} (allowed: {sorted(allowed_keys)})"
            )

        if "do_resize" in mm_processor_kwargs:
            raw = mm_processor_kwargs.get("do_resize")
            if isinstance(raw, bool):
                do_resize = raw
            elif isinstance(raw, int) and raw in {0, 1}:
                do_resize = bool(raw)
            elif isinstance(raw, str):
                raw_s = raw.strip().lower()
                if raw_s in {"true", "1", "yes"}:
                    do_resize = True
                elif raw_s in {"false", "0", "no"}:
                    do_resize = False
                else:
                    raise TypeError(
                        "rollout_matching.vllm.mm_processor_kwargs.do_resize must be a bool"
                    )
            else:
                raise TypeError(
                    "rollout_matching.vllm.mm_processor_kwargs.do_resize must be a bool"
                )
            mm_processor_kwargs["do_resize"] = do_resize

        vllm_engine_kwargs["mm_processor_kwargs"] = mm_processor_kwargs

    return {
        "rollout_backend": rollout_contract["rollout_backend"],
        "eval_rollout_backend": rollout_contract["eval_rollout_backend"],
        "vllm_mode": rollout_contract["vllm_mode"],
        "server_base_urls": server_base_urls,
        "server_model": model_path,
        "train_jsonl_resolved": str(train_jsonl_path),
        "val_jsonl_resolved": "" if val_jsonl_path is None else str(val_jsonl_path),
        "template_max_pixels": template_max_pixels,
        "server_template": server_template,
        "server_max_length": server_max_length,
        "server_truncation_strategy": server_truncation_strategy,
        "root_image_dir_resolved": str(root_image_dir),
        "vllm_tensor_parallel_size": tensor_parallel_size,
        "vllm_enforce_eager": enforce_eager,
        "vllm_max_model_len": max_model_len,
        "vllm_enable_lora": enable_lora,
        "vllm_gpu_memory_utilization": gpu_memory_utilization,
        "vllm_engine_kwargs": vllm_engine_kwargs,
        "server_torch_dtype": server_torch_dtype,
    }


def _infer_repo_root_from_config_path(config_path: Path) -> Optional[Path]:
    """Best-effort repo-root inference from a config path.

    CoordExp configs live under `<repo>/configs/...`. When this holds, we treat
    non-dot-prefixed relative paths (e.g. `public_data/...`, `output/...`) as
    repo-relative.

    Returns None when the config path is outside a standard repo layout.
    """

    for parent in config_path.parents:
        if parent.name == "configs":
            return parent.parent
    return None


def _resolve_path_for_config(raw_path: str, config_path: Optional[str]) -> Path:
    """Resolve a possibly-relative path in a way that is stable across cwd.

    Resolution policy:
    - absolute paths are resolved directly
    - if `raw_path` starts with '.' (./ or ../), resolve relative to the config directory
    - otherwise, resolve relative to the repo root inferred from the config path
      (fallback: config directory)
    - fallback when config_path is missing: resolve relative to cwd
    """

    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()

    if config_path:
        config_file = Path(config_path).expanduser().resolve()
        config_dir = config_file.parent
        repo_root = _infer_repo_root_from_config_path(config_file)
        if raw_path.startswith("."):
            anchor = config_dir
        else:
            anchor = repo_root or config_dir
        return (anchor / path).resolve()

    return (Path.cwd() / path).resolve()


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
