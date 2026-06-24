from __future__ import annotations

import json
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from src.common.model_paths import normalize_coordexp_base_model_path
from src.detection.template_contracts import (
    resolve_detection_template_contract,
    required_trainable_token_row_ids,
)
from src.tokens.qwen_native import EXPECTED_COORD_END_ID, EXPECTED_COORD_START_ID

VLLM_ADAPTER_UNSUPPORTED_MESSAGE = (
    "Adapter-based inference is supported only with infer.backend.type=hf in "
    "this repo. Current Stage-1 adapters include DoRA + "
    "token_embeddings_adapter, which vLLM does not support natively; use a "
    "merged checkpoint for vLLM."
)

_STRUCTURAL_ROW_ID_TO_TOKEN = {
    151646: "<|object_ref_start|>",
    151647: "<|object_ref_end|>",
    151648: "<|box_start|>",
    151649: "<|box_end|>",
}
TRAINING_ONLY_MODULES_TO_DROP_FOR_INFERENCE = ("coverage_ledger_head",)


@dataclass(frozen=True)
class TokenEmbeddingsAdapterSpec:
    token_ids: tuple[int, ...]
    tie_head: bool
    embed_offset_rows: int
    head_offset_rows: int | None = None


@dataclass(frozen=True)
class AdapterCheckpointInfo:
    path: str
    base_model_name_or_path: Optional[str]
    modules_to_save: tuple[str, ...]
    token_embeddings_adapter_spec: Optional[TokenEmbeddingsAdapterSpec]


@dataclass(frozen=True)
class AdapterCheckpointInferenceView:
    path: str
    source_path: str
    dropped_modules_to_save: tuple[str, ...]
    dropped_tensor_keys: tuple[str, ...]


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


def _safetensor_key_belongs_to_module(key: str, module_name: str) -> bool:
    return str(module_name) in str(key).split(".")


def _filter_adapter_safetensors_for_inference(
    *,
    source_path: Path,
    target_path: Path,
    dropped_modules: tuple[str, ...],
) -> tuple[str, ...]:
    try:
        from safetensors import safe_open
        from safetensors.torch import save_file
    except ImportError as exc:
        raise RuntimeError(
            "adapter checkpoint inference filtering requires the 'safetensors' "
            "package in the active environment."
        ) from exc

    import torch

    kept: dict[str, torch.Tensor] = {}
    dropped: list[str] = []
    with safe_open(str(source_path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            key_s = str(key)
            if any(
                _safetensor_key_belongs_to_module(key_s, module)
                for module in dropped_modules
            ):
                dropped.append(key_s)
                continue
            kept[key_s] = handle.get_tensor(key_s)

    if kept:
        save_file(kept, str(target_path))
    return tuple(dropped)


def prepare_adapter_checkpoint_for_inference(
    adapter_checkpoint: str,
    *,
    cache_root: str | Path | None = None,
    training_only_modules: tuple[str, ...] = TRAINING_ONLY_MODULES_TO_DROP_FOR_INFERENCE,
) -> AdapterCheckpointInferenceView:
    """Return an adapter directory that is loadable for inference.

    Training-only PEFT modules, such as the coverage ledger auxiliary head, are
    useful for checkpoint resume/debugging but are not part of generation.
    Inference should load the LoRA/token-row state without requiring those
    auxiliary modules to exist on the base model.
    """

    adapter_dir = _require_local_adapter_dir(adapter_checkpoint)
    cfg_path = adapter_dir / "adapter_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"{cfg_path} must contain a JSON object.")

    modules_raw = cfg.get("modules_to_save")
    if modules_raw is None:
        modules_to_save: list[str] = []
    elif isinstance(modules_raw, list):
        modules_to_save = [
            str(item).strip() for item in modules_raw if str(item).strip()
        ]
    else:
        raise ValueError(f"{cfg_path}: modules_to_save must be a list when present.")

    drop_set = {str(item).strip() for item in training_only_modules if str(item).strip()}
    dropped_modules = tuple(module for module in modules_to_save if module in drop_set)
    if not dropped_modules:
        return AdapterCheckpointInferenceView(
            path=str(adapter_dir),
            source_path=str(adapter_dir),
            dropped_modules_to_save=(),
            dropped_tensor_keys=(),
        )

    if cache_root is None:
        cache_parent = Path(tempfile.gettempdir()) / "coordexp_infer_adapter_views"
    else:
        cache_parent = Path(cache_root).expanduser()
    cache_parent.mkdir(parents=True, exist_ok=True)
    view_dir = Path(
        tempfile.mkdtemp(prefix=f"{adapter_dir.name}-", dir=str(cache_parent))
    )
    shutil.copytree(
        adapter_dir,
        view_dir,
        symlinks=True,
        ignore=shutil.ignore_patterns("adapter_config.json", "adapter_model.safetensors"),
        dirs_exist_ok=True,
    )

    view_cfg = dict(cfg)
    view_cfg["modules_to_save"] = [
        module for module in modules_to_save if module not in set(dropped_modules)
    ]
    (view_dir / "adapter_config.json").write_text(
        json.dumps(view_cfg, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )

    dropped_tensor_keys: tuple[str, ...] = ()
    weights_path = adapter_dir / "adapter_model.safetensors"
    if weights_path.is_file():
        dropped_tensor_keys = _filter_adapter_safetensors_for_inference(
            source_path=weights_path,
            target_path=view_dir / "adapter_model.safetensors",
            dropped_modules=dropped_modules,
        )

    return AdapterCheckpointInferenceView(
        path=str(view_dir),
        source_path=str(adapter_dir),
        dropped_modules_to_save=dropped_modules,
        dropped_tensor_keys=dropped_tensor_keys,
    )


def _load_token_embeddings_adapter_spec(adapter_dir: Path) -> TokenEmbeddingsAdapterSpec:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError(
            "token_embeddings_adapter inference requires the 'safetensors' package "
            "in the active environment."
        ) from exc

    weights_path = adapter_dir / "adapter_model.safetensors"
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"adapter_model.safetensors not found under {adapter_dir}."
        )

    token_ids_key: Optional[str] = None
    embed_key: Optional[str] = None
    head_key: Optional[str] = None
    token_ids: tuple[int, ...] = ()
    embed_offset_rows: int | None = None
    head_offset_rows: int | None = None

    with safe_open(str(weights_path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if key.endswith("token_embeddings_adapter.token_ids"):
                token_ids_key = key
            elif key.endswith("token_embeddings_adapter.embed_offset"):
                embed_key = key
            elif key.endswith("token_embeddings_adapter.head_offset"):
                head_key = key

        if token_ids_key is None or embed_key is None:
            raise ValueError(
                "token_embeddings_adapter was declared in modules_to_save, but "
                "adapter_model.safetensors is missing token_ids/embed_offset."
            )

        token_ids_tensor = handle.get_tensor(token_ids_key).reshape(-1).tolist()
        token_ids = tuple(int(value) for value in token_ids_tensor)
        embed_offset_rows = int(handle.get_tensor(embed_key).shape[0])
        if head_key is not None:
            head_offset_rows = int(handle.get_tensor(head_key).shape[0])

    if not token_ids:
        raise ValueError("token_embeddings_adapter.token_ids must be non-empty.")

    if embed_offset_rows is None:
        raise ValueError("token_embeddings_adapter.embed_offset row count is unavailable.")

    return TokenEmbeddingsAdapterSpec(
        token_ids=token_ids,
        tie_head=head_key is None,
        embed_offset_rows=embed_offset_rows,
        head_offset_rows=head_offset_rows,
    )


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

    token_embeddings_adapter_spec: Optional[TokenEmbeddingsAdapterSpec] = None
    if "token_embeddings_adapter" in modules_to_save:
        token_embeddings_adapter_spec = _load_token_embeddings_adapter_spec(adapter_dir)

    return AdapterCheckpointInfo(
        path=str(adapter_checkpoint),
        base_model_name_or_path=base_model_name_or_path,
        modules_to_save=modules_to_save,
        token_embeddings_adapter_spec=token_embeddings_adapter_spec,
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


def validate_compact_token_embeddings_adapter_contract(
    resolved_checkpoint: ResolvedInferenceCheckpoint,
    *,
    detection_template_id: str,
) -> None:
    """Fail fast when compact adapter inference would drop token-row offsets."""

    contract = resolve_detection_template_contract(detection_template_id)
    if not contract.is_compact:
        return
    if resolved_checkpoint.resolved_adapter_checkpoint is None:
        # Full/merged checkpoints may already have offsets injected into weights.
        return

    adapter_info = resolved_checkpoint.adapter_info
    adapter_spec = (
        adapter_info.token_embeddings_adapter_spec if adapter_info is not None else None
    )
    if adapter_spec is None:
        raise ValueError(
            f"{contract.template_id} adapter inference requires adapter_config.json "
            "modules_to_save to include token_embeddings_adapter and "
            "adapter_model.safetensors to contain token_embeddings_adapter weights. "
            "This checkpoint would otherwise run with coordinate/token-row offsets inactive."
        )

    actual = tuple(int(token_id) for token_id in adapter_spec.token_ids)
    expected = required_trainable_token_row_ids(contract.template_id)
    actual_set = set(actual)
    required_set = set(expected)
    missing = sorted(required_set - actual_set)
    extra = sorted(actual_set - required_set)
    duplicates = sorted(
        token_id for token_id, count in Counter(actual).items() if count > 1
    )
    if len(actual) != len(required_set) or missing or extra or duplicates:
        raise ValueError(
            f"{contract.template_id} token_embeddings_adapter must contain exactly "
            f"{len(expected)} trainable token rows: "
            f"{_describe_required_rows(contract.template_id)}. "
            f"got={len(actual)} unique={len(actual_set)} "
            f"missing={_describe_row_id_list(missing[:8])} "
            f"extra={_describe_row_id_list(extra[:8])} "
            f"duplicates={_describe_row_id_list(duplicates[:8])}"
        )
    if adapter_spec.embed_offset_rows != len(actual):
        raise ValueError(
            f"{contract.template_id} token_embeddings_adapter embed_offset rows "
            f"must match token_ids; got embed_offset rows={adapter_spec.embed_offset_rows} "
            f"token_ids={len(actual)}"
        )
    if (
        adapter_spec.head_offset_rows is not None
        and adapter_spec.head_offset_rows != len(actual)
    ):
        raise ValueError(
            f"{contract.template_id} token_embeddings_adapter head_offset rows "
            f"must match token_ids; got head_offset rows={adapter_spec.head_offset_rows} "
            f"token_ids={len(actual)}"
        )


def _describe_required_rows(template_id: str) -> str:
    contract = resolve_detection_template_contract(template_id)
    structural = ", ".join(contract.required_structural_tokens)
    return f"{structural}, and <|coord_0|>..<|coord_999|>"


def _describe_row_id_list(row_ids: list[int]) -> list[str]:
    return [
        _STRUCTURAL_ROW_ID_TO_TOKEN.get(
            int(row_id),
            _coord_row_id_to_token_text(int(row_id)),
        )
        for row_id in row_ids
    ]


def _coord_row_id_to_token_text(row_id: int) -> str:
    if EXPECTED_COORD_START_ID <= row_id <= EXPECTED_COORD_END_ID:
        return f"<|coord_{row_id - EXPECTED_COORD_START_ID}|>"
    return str(row_id)
