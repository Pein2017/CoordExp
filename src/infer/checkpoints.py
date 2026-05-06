from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from src.common.model_paths import normalize_coordexp_base_model_path
from src.tokens.qwen_native import (
    EXPECTED_BOX_START_ID,
    EXPECTED_COORD_END_ID,
    EXPECTED_COORD_START_ID,
    EXPECTED_OBJECT_REF_START_ID,
)

VLLM_ADAPTER_UNSUPPORTED_MESSAGE = (
    "Adapter-based inference is supported only with infer.backend.type=hf in "
    "this repo. Current Stage-1 adapters include DoRA + "
    "coord_offset_adapter, which vLLM does not support natively; use a "
    "merged checkpoint for vLLM."
)

COMPACT_COORD_TOKEN_REQUIRED_ROW_IDS: tuple[int, ...] = (
    EXPECTED_OBJECT_REF_START_ID,
    EXPECTED_BOX_START_ID,
    *range(EXPECTED_COORD_START_ID, EXPECTED_COORD_END_ID + 1),
)


@dataclass(frozen=True)
class CoordOffsetAdapterSpec:
    coord_ids: tuple[int, ...]
    tie_head: bool


@dataclass(frozen=True)
class AdapterCheckpointInfo:
    path: str
    base_model_name_or_path: Optional[str]
    modules_to_save: tuple[str, ...]
    coord_offset_spec: Optional[CoordOffsetAdapterSpec]


@dataclass(frozen=True)
class ResolvedInferenceCheckpoint:
    checkpoint_mode: Literal["full_model", "base_plus_adapter", "adapter_shorthand"]
    requested_model_checkpoint: str
    requested_adapter_checkpoint: Optional[str]
    resolved_base_model_checkpoint: str
    resolved_adapter_checkpoint: Optional[str]
    adapter_info: Optional[AdapterCheckpointInfo]


def looks_like_local_adapter_checkpoint(path: str) -> bool:
    raw = str(path or "").strip()
    if not raw:
        return False
    candidate = Path(raw).expanduser()
    return candidate.is_dir() and (candidate / "adapter_config.json").is_file()


def _require_local_adapter_dir(path: str) -> Path:
    raw = str(path or "").strip()
    if not raw:
        raise ValueError("Adapter checkpoint path must be a non-empty string.")

    adapter_dir = Path(raw).expanduser()
    cfg_path = adapter_dir / "adapter_config.json"
    if not adapter_dir.is_dir() or not cfg_path.is_file():
        raise ValueError(
            f"Adapter checkpoint {raw!r} must be a local directory containing "
            "adapter_config.json."
        )
    return adapter_dir


def _load_coord_offset_spec(adapter_dir: Path) -> CoordOffsetAdapterSpec:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError(
            "coord_offset_adapter inference requires the 'safetensors' package "
            "in the active environment."
        ) from exc

    weights_path = adapter_dir / "adapter_model.safetensors"
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"adapter_model.safetensors not found under {adapter_dir}."
        )

    coord_key: Optional[str] = None
    embed_key: Optional[str] = None
    head_key: Optional[str] = None
    coord_ids: tuple[int, ...] = ()

    with safe_open(str(weights_path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if key.endswith("coord_offset_adapter.coord_ids"):
                coord_key = key
            elif key.endswith("coord_offset_adapter.embed_offset"):
                embed_key = key
            elif key.endswith("coord_offset_adapter.head_offset"):
                head_key = key

        if coord_key is None or embed_key is None:
            raise ValueError(
                "coord_offset_adapter was declared in modules_to_save, but "
                "adapter_model.safetensors is missing coord_ids/embed_offset."
            )

        coord_ids_tensor = handle.get_tensor(coord_key).reshape(-1).tolist()
        coord_ids = tuple(int(value) for value in coord_ids_tensor)

    if not coord_ids:
        raise ValueError("coord_offset_adapter.coord_ids must be non-empty.")

    return CoordOffsetAdapterSpec(coord_ids=coord_ids, tie_head=head_key is None)


def load_adapter_checkpoint_info(adapter_checkpoint: str) -> AdapterCheckpointInfo:
    adapter_dir = _require_local_adapter_dir(adapter_checkpoint)
    cfg_path = adapter_dir / "adapter_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"{cfg_path} must contain a JSON object.")

    base_raw = cfg.get("base_model_name_or_path")
    if base_raw is None:
        base_model_name_or_path: Optional[str] = None
    elif isinstance(base_raw, str):
        base_model_name_or_path = base_raw.strip() or None
        base_model_name_or_path = normalize_coordexp_base_model_path(
            base_model_name_or_path
        )
    else:
        raise ValueError(
            f"{cfg_path}: base_model_name_or_path must be a string when present."
        )

    modules_raw = cfg.get("modules_to_save")
    if modules_raw is None:
        modules_to_save: tuple[str, ...] = ()
    elif isinstance(modules_raw, list):
        modules_to_save = tuple(
            str(item).strip() for item in modules_raw if str(item).strip()
        )
    else:
        raise ValueError(f"{cfg_path}: modules_to_save must be a list when present.")

    coord_offset_spec: Optional[CoordOffsetAdapterSpec] = None
    if "coord_offset_adapter" in modules_to_save:
        coord_offset_spec = _load_coord_offset_spec(adapter_dir)

    return AdapterCheckpointInfo(
        path=str(adapter_checkpoint),
        base_model_name_or_path=base_model_name_or_path,
        modules_to_save=modules_to_save,
        coord_offset_spec=coord_offset_spec,
    )


def resolve_inference_checkpoint(
    *,
    model_checkpoint: str,
    adapter_checkpoint: Optional[str] = None,
) -> ResolvedInferenceCheckpoint:
    requested_model_checkpoint = str(model_checkpoint or "").strip()
    if not requested_model_checkpoint:
        raise ValueError("infer.model_checkpoint must be a non-empty string.")

    requested_adapter_checkpoint = str(adapter_checkpoint or "").strip() or None
    model_is_adapter = looks_like_local_adapter_checkpoint(requested_model_checkpoint)

    if requested_adapter_checkpoint is not None:
        raise ValueError(
            "infer.adapter_checkpoint is no longer supported. "
            "Use adapter shorthand instead: set infer.model_checkpoint "
            "to the adapter directory and let adapter_config.json resolve the base."
        )

    if model_is_adapter:
        adapter_info = load_adapter_checkpoint_info(requested_model_checkpoint)
        base_model = str(adapter_info.base_model_name_or_path or "").strip()
        if not base_model:
            raise ValueError(
                "Adapter shorthand requires "
                "adapter_config.json.base_model_name_or_path to be set."
            )
        return ResolvedInferenceCheckpoint(
            checkpoint_mode="adapter_shorthand",
            requested_model_checkpoint=requested_model_checkpoint,
            requested_adapter_checkpoint=None,
            resolved_base_model_checkpoint=base_model,
            resolved_adapter_checkpoint=requested_model_checkpoint,
            adapter_info=adapter_info,
        )

    return ResolvedInferenceCheckpoint(
        checkpoint_mode="full_model",
        requested_model_checkpoint=requested_model_checkpoint,
        requested_adapter_checkpoint=None,
        resolved_base_model_checkpoint=requested_model_checkpoint,
        resolved_adapter_checkpoint=None,
        adapter_info=None,
    )


def validate_compact_coord_token_adapter_contract(
    resolved_checkpoint: ResolvedInferenceCheckpoint,
    *,
    detection_sequence_format: str,
) -> None:
    """Fail fast when compact adapter inference would drop token-row offsets."""

    normalized_format = (
        str(detection_sequence_format).strip().lower().replace("-", "_").replace(" ", "_")
    )
    if normalized_format != "compact_full":
        return
    if resolved_checkpoint.resolved_adapter_checkpoint is None:
        # Full/merged checkpoints may already have offsets injected into weights.
        return

    adapter_info = resolved_checkpoint.adapter_info
    coord_spec = adapter_info.coord_offset_spec if adapter_info is not None else None
    if coord_spec is None:
        raise ValueError(
            "compact_full adapter inference requires adapter_config.json "
            "modules_to_save to include coord_offset_adapter and "
            "adapter_model.safetensors to contain coord_offset_adapter weights. "
            "This checkpoint would otherwise run with coordinate/token-row offsets inactive."
        )
    if not coord_spec.tie_head:
        raise ValueError(
            "compact_full adapter inference requires tied-head "
            "coord_offset_adapter checkpoints (tie_head=True; no head_offset tensor)."
        )

    actual = tuple(int(token_id) for token_id in coord_spec.coord_ids)
    actual_set = set(actual)
    required_set = set(COMPACT_COORD_TOKEN_REQUIRED_ROW_IDS)
    missing = sorted(required_set - actual_set)
    extra = sorted(actual_set - required_set)
    if len(actual) != len(required_set) or missing or extra:
        raise ValueError(
            "compact_full coord_offset_adapter must contain exactly 1002 trainable "
            "token rows: <|object_ref_start|>, <|box_start|>, and "
            "<|coord_0|>..<|coord_999|>. "
            f"got={len(actual)} unique={len(actual_set)} "
            f"missing={missing[:8]} extra={extra[:8]}"
        )
