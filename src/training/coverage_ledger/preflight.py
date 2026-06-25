"""Strict preflight for coverage-ledger smoke artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config.loader import ConfigLoader
from src.config.schema import CoordTokensConfig, DetectionTrainingConfig
from src.config.strict_dataclass import dataclass_asdict_no_none
from src.common.model_paths import canonical_coordexp_repo_root
from src.coord_tokens.template_adapter import apply_coord_template_adapter
from src.detection.dataset_selection import (
    DatasetRowSelectionConfig,
    SEEDED_RANDOM_WITHOUT_REPLACEMENT,
    normalize_dataset_row_selection,
    select_dataset_row_indices,
)
from src.detection.dataset import DetectionTrainingDataset
from src.detection.runtime import (
    build_detection_runtime_custom_shim,
    detection_mode,
    resolve_detection_prompts,
)
from src.training.coverage_ledger.artifacts import (
    CoverageLedgerOverlayCandidate,
    CoverageLedgerPreflightArtifactInputs,
    CoverageLedgerPreflightArtifactResult,
    write_coverage_ledger_preflight_artifacts,
)
from src.training.coverage_ledger.sidecars import CoverageLedgerSidecar
from src.training.coverage_ledger.visual_regions import (
    VisualTokenRegion,
    map_norm1000_bbox_to_visual_token_region,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
SELECTION_ALGORITHM = SEEDED_RANDOM_WITHOUT_REPLACEMENT
IMAGE_GRID_METADATA_VERSION = "qwen3_vl_image_grid_thw_v0"
DATASET_ID = "coverage_ledger_preflight"

ALLOWED_BASELINE_LEDGER_DIFFS = {
    "/objective/terms/token_type_mass/enabled",
    "/objective/terms/token_type_mass/weight",
    "/objective/terms/continuation_margin/enabled",
    "/objective/terms/continuation_margin/weight",
    "/objective/terms/bbox_positive_area/enabled",
    "/objective/terms/bbox_positive_area/weight",
    "/objective/terms/coverage_ledger/enabled",
    "/objective/terms/coverage_ledger/coverage_weight",
    "/objective/terms/coverage_ledger/region_anchor_weight",
    "/objective/terms/coverage_ledger/ledger_projection_dim",
    "/objective/terms/coverage_ledger/temperature",
    "/objective/terms/coverage_ledger/normalize_eps",
    "/objective/terms/coverage_ledger/pos_weight",
    "/objective/terms/coverage_ledger/log_auc",
    "/objective/terms/coverage_ledger/log_accuracy",
    "/objective/terms/coverage_ledger/overlay_sample_count",
    "/objective/terms/coverage_ledger/smoke_sample_count",
    "/objective/terms/coverage_ledger/smoke_sample_seed",
    "/training/run_name",
    "/training/artifact_subdir",
    "/training/output_dir",
    "/training/logging_dir",
    "/debug/train_artifact_subdir",
    "/debug/val_artifact_subdir",
    "/debug/preflight_artifact_subdir",
}


def _load_swift_template_dependencies() -> tuple[Any, Any]:
    try:
        from swift.llm import get_model_tokenizer as resolved_get_model_processor
        from swift.llm.template import get_template as resolved_get_template
    except ImportError:
        from swift.model import get_model_processor as resolved_get_model_processor
        from swift.template import get_template as resolved_get_template
    return resolved_get_model_processor, resolved_get_template


def get_model_processor(*args: Any, **kwargs: Any) -> Any:
    resolved_get_model_processor, _resolved_get_template = _load_swift_template_dependencies()
    return resolved_get_model_processor(*args, **kwargs)


def get_template(*args: Any, **kwargs: Any) -> Any:
    _resolved_get_model_processor, resolved_get_template = _load_swift_template_dependencies()
    return resolved_get_template(*args, **kwargs)


@dataclass(frozen=True, slots=True)
class CoverageLedgerPreflightResult:
    artifact_result: CoverageLedgerPreflightArtifactResult
    selected_row_indices: tuple[int, ...]
    selected_sample_ids: tuple[str, ...]


def run_coverage_ledger_preflight(
    *,
    config_path: str | Path,
    baseline_config_path: str | Path,
    output_root: str | Path,
    swift_template: Any | None = None,
) -> CoverageLedgerPreflightResult:
    """Build all coverage-ledger preflight artifacts without launching training."""

    ledger_config = _load_detection_config(config_path)
    baseline_config = _load_detection_config(baseline_config_path)
    assert_smoke_config_diff_allowed(baseline_config, ledger_config)

    ledger_term = ledger_config.objective.terms.coverage_ledger
    if not ledger_term.enabled:
        raise ValueError("coverage_ledger preflight requires ledger config enabled")
    selection = resolve_coverage_ledger_train_selection(ledger_config)
    expected_sample_count = int(selection.count)
    expected_overlay_count = int(ledger_term.overlay_sample_count)
    selection_seed = int(selection.seed)

    system_prompt, _user_prompt = resolve_detection_prompts(ledger_config)
    custom_config = build_detection_runtime_custom_shim(ledger_config)
    train_jsonl_path = _resolve_repo_path(ledger_config.data.train_jsonl)
    _require_existing_file(train_jsonl_path, "training JSONL")
    local_model_path = _resolve_local_model_path(ledger_config.model["model"])
    if local_model_path is not None:
        _require_existing_path(local_model_path, "model cache")

    template = swift_template or build_preflight_swift_template(
        ledger_config,
        system_prompt=system_prompt,
    )
    patch_size, spatial_merge_size = resolve_visual_grid_geometry(template)
    dataset = build_coverage_ledger_preflight_training_dataset(
        ledger_config,
        train_jsonl_path=train_jsonl_path,
        swift_template=template,
        selection=selection,
        system_prompt=system_prompt,
        user_prompt=custom_config.user_prompt,
        seed=int(ledger_config.training.get("seed", selection_seed)),
    )
    selected_row_indices = dataset.source_row_indices

    sidecars: list[CoverageLedgerSidecar] = []
    visual_regions_by_sample_id: dict[str, tuple[VisualTokenRegion, ...]] = {}
    visual_grid_shape_by_sample_id: dict[str, tuple[int, int]] = {}
    overlay_candidates: list[CoverageLedgerOverlayCandidate] = []
    for local_index, row_index in enumerate(selected_row_indices):
        sample = dataset[local_index]
        sidecar = _extract_coverage_ledger_sidecar(sample)
        regions = tuple(
            map_norm1000_bbox_to_visual_token_region(
                entry.bbox_norm1000_xyxy,
                image_grid_thw=sidecar.image_grid_thw,
                processed_width=sidecar.processed_width,
                processed_height=sidecar.processed_height,
                patch_size=patch_size,
                spatial_merge_size=spatial_merge_size,
            )
            for entry in sidecar.object_entries
        )
        sidecars.append(sidecar)
        visual_regions_by_sample_id[sidecar.sample_id] = regions
        _grid_t, grid_h, grid_w = sidecar.image_grid_thw
        visual_grid_shape_by_sample_id[sidecar.sample_id] = (
            grid_h // spatial_merge_size,
            grid_w // spatial_merge_size,
        )
        for entry, region in zip(sidecar.object_entries, regions, strict=True):
            if len(overlay_candidates) >= expected_overlay_count:
                break
            overlay_candidates.append(
                CoverageLedgerOverlayCandidate(
                    sample_id=sidecar.sample_id,
                    row_index=int(row_index),
                    object_entry=entry,
                    visual_region=region,
                    render_image_path=resolve_overlay_render_image_path(sample),
                )
            )

    artifact_result = write_coverage_ledger_preflight_artifacts(
        CoverageLedgerPreflightArtifactInputs(
            output_root=Path(output_root),
            source_jsonl_path=train_jsonl_path,
            dataset_id=DATASET_ID,
            split="train",
            selected_row_indices=selected_row_indices,
            selection_seed=selection_seed,
            selection_algorithm=selection.algorithm,
            template_id=ledger_config.detection_template.id,
            object_field_order=str(
                ledger_config.sample_factory.target_sequence.object_field_order
            ),
            tokenizer_id=_tokenizer_id(template),
            model_id=str(ledger_config.model["model"]),
            processor_do_resize=False,
            image_grid_metadata_version=IMAGE_GRID_METADATA_VERSION,
            sidecars=tuple(sidecars),
            visual_regions_by_sample_id=visual_regions_by_sample_id,
            overlay_candidates=tuple(overlay_candidates),
            visual_grid_shape_by_sample_id=visual_grid_shape_by_sample_id,
            expected_sample_count=expected_sample_count,
            expected_overlay_count=expected_overlay_count,
        )
    )
    return CoverageLedgerPreflightResult(
        artifact_result=artifact_result,
        selected_row_indices=selected_row_indices,
        selected_sample_ids=tuple(sidecar.sample_id for sidecar in sidecars),
    )


def assert_smoke_config_diff_allowed(
    baseline_config: DetectionTrainingConfig,
    ledger_config: DetectionTrainingConfig,
) -> None:
    """Fail if the baseline/ledger pair differs outside the Task-2 allowlist."""

    baseline_payload = dataclass_asdict_no_none(baseline_config)
    ledger_payload = dataclass_asdict_no_none(ledger_config)
    changed_paths = _changed_paths(baseline_payload, ledger_payload)
    disallowed = sorted(changed_paths - ALLOWED_BASELINE_LEDGER_DIFFS)
    if disallowed:
        raise ValueError(
            "coverage ledger smoke baseline/config diff contains disallowed "
            f"paths: {disallowed}"
        )


def select_preflight_row_indices(
    *,
    total_rows: int,
    count: int,
    seed: int,
) -> tuple[int, ...]:
    return select_dataset_row_indices(
        total_rows=total_rows,
        selection=DatasetRowSelectionConfig(
            algorithm=SELECTION_ALGORITHM,
            count=int(count),
            seed=int(seed),
        ),
    )


def resolve_coverage_ledger_train_selection(
    training_config: DetectionTrainingConfig,
) -> DatasetRowSelectionConfig:
    selection = normalize_dataset_row_selection(
        getattr(training_config.debug, "train_sample_selection", None),
        path="debug.train_sample_selection",
    )
    if selection is None:
        raise ValueError(
            "coverage ledger smoke/preflight requires debug.train_sample_selection "
            "as the single source of truth for train rows"
        )
    ledger_term = training_config.objective.terms.coverage_ledger
    if int(ledger_term.smoke_sample_count) != selection.count:
        raise ValueError(
            "objective.terms.coverage_ledger.smoke_sample_count must match "
            "debug.train_sample_selection.count"
        )
    if int(ledger_term.smoke_sample_seed) != selection.seed:
        raise ValueError(
            "objective.terms.coverage_ledger.smoke_sample_seed must match "
            "debug.train_sample_selection.seed"
        )
    return selection


def build_coverage_ledger_preflight_training_dataset(
    training_config: DetectionTrainingConfig,
    *,
    train_jsonl_path: str | Path,
    swift_template: Any,
    selection: DatasetRowSelectionConfig,
    image_root: str | Path | None = None,
    system_prompt: str | None = None,
    user_prompt: str | None = None,
    seed: int | None = None,
) -> DetectionTrainingDataset:
    """Build the preflight dataset with the same sample identity as smoke training."""

    if system_prompt is None or user_prompt is None:
        resolved_system_prompt, _resolved_user_prompt = resolve_detection_prompts(
            training_config
        )
        custom_config = build_detection_runtime_custom_shim(training_config)
        if system_prompt is None:
            system_prompt = resolved_system_prompt
        if user_prompt is None:
            user_prompt = custom_config.user_prompt
    return DetectionTrainingDataset.from_jsonl(
        train_jsonl_path,
        swift_template=swift_template,
        image_root=(
            _resolve_repo_path(training_config.data.image_root)
            if image_root is None
            else image_root
        ),
        detection_template_id=training_config.detection_template.id,
        mode=detection_mode(training_config),
        object_ordering=training_config.sample_factory.target_sequence.object_ordering,
        user_prompt=str(user_prompt),
        system_prompt=system_prompt,
        seed=int(
            seed
            if seed is not None
            else training_config.training.get("seed", selection.seed)
        ),
        state_weighting="uniform_permutation",
        normalization="semantic_image_bucket_balanced",
        object_field_order=str(
            training_config.sample_factory.target_sequence.object_field_order
        ),
        teacher_forcing_profile=str(training_config.objective.profile),
        teacher_forcing_rollin_policy=str(
            training_config.objective.target_ir.rollin_policy.name
        ),
        teacher_forcing_rollin_base_seed=int(
            training_config.objective.target_ir.rollin_policy.base_seed
        ),
        coverage_ledger_enabled=True,
        sample_limit=selection.count,
        sample_selection=selection,
        dataset_name="detection_train",
    )


def build_preflight_swift_template(
    training_config: DetectionTrainingConfig,
    *,
    system_prompt: str | None,
) -> Any:
    """Create a Swift template with no model load and coord-token adaptation."""

    import torch

    dtype = _torch_dtype(training_config.model.get("torch_dtype"))
    model_type = training_config.model.get("model_type")
    model_kwargs = {}
    if model_type is not None:
        model_kwargs["model_type"] = str(model_type)
    local_model_path = _resolve_local_model_path(training_config.model["model"])
    model_path = (
        str(local_model_path)
        if local_model_path is not None
        else str(training_config.model["model"])
    )
    _model, processor = get_model_processor(
        model_path,
        torch_dtype=dtype,
        load_model=False,
        download_model=False,
        **model_kwargs,
    )
    template = get_template(
        template_type=str(training_config.template.get("template", "qwen3_vl")),
        processor=processor,
        default_system=system_prompt,
        max_length=int(training_config.template.get("max_length", 12000)),
        truncation_strategy=str(
            training_config.template.get("truncation_strategy", "raise")
        ),
        max_pixels=int(training_config.template.get("max_pixels", 1048576)),
    )
    apply_coord_template_adapter(
        template,
        CoordTokensConfig(enabled=True, skip_bbox_norm=True),
    )
    if hasattr(template, "set_mode"):
        template.set_mode("train")
    return template


def resolve_visual_grid_geometry(swift_template: Any) -> tuple[int, int]:
    processor = getattr(swift_template, "processor", None)
    image_processor = getattr(processor, "image_processor", processor)
    patch_size = getattr(image_processor, "patch_size", None)
    spatial_merge_size = getattr(image_processor, "spatial_merge_size", None)
    if spatial_merge_size is None:
        spatial_merge_size = getattr(image_processor, "merge_size", None)
    if patch_size is None or spatial_merge_size is None:
        raise ValueError(
            "coverage ledger preflight requires processor patch_size and "
            "spatial_merge_size/merge_size for visual-region overlays"
        )
    return int(patch_size), int(spatial_merge_size)


def resolve_overlay_render_image_path(sample: Mapping[str, Any]) -> Path:
    """Return the resolved image path from the dataset-rendered chat messages."""

    messages = sample.get("messages")
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        raise ValueError("coverage ledger overlay preflight requires sample messages")
    for message in messages:
        if not isinstance(message, Mapping) or message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, Sequence) or isinstance(content, (str, bytes)):
            continue
        for item in content:
            if (
                isinstance(item, Mapping)
                and item.get("type") == "image"
                and isinstance(item.get("image"), str)
                and str(item.get("image")).strip()
            ):
                return Path(str(item["image"])).expanduser().resolve(strict=True)
    raise ValueError(
        "coverage ledger overlay preflight could not find a resolved user image path"
    )


def _load_detection_config(path: str | Path) -> DetectionTrainingConfig:
    cfg = ConfigLoader.load_materialized_training_config(str(path))
    if not isinstance(cfg, DetectionTrainingConfig):
        raise TypeError("coverage ledger preflight requires DetectionTrainingConfig")
    return cfg


def _extract_coverage_ledger_sidecar(sample: Mapping[str, Any]) -> CoverageLedgerSidecar:
    training_sidecars = sample.get("training_sidecars")
    supervision = getattr(training_sidecars, "supervision", None)
    payloads = tuple(getattr(supervision, "payloads", ()))
    sidecars = [payload for payload in payloads if isinstance(payload, CoverageLedgerSidecar)]
    if len(sidecars) != 1:
        raise ValueError(
            "coverage ledger preflight expected exactly one "
            f"CoverageLedgerSidecar; got {len(sidecars)}"
        )
    return sidecars[0]


def _resolve_repo_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve(strict=False)
    worktree_path = (REPO_ROOT / path).resolve(strict=False)
    if path.parts and path.parts[0] in {"model_cache", "public_data"}:
        return (canonical_coordexp_repo_root() / path).resolve(strict=False)
    if worktree_path.exists():
        return worktree_path
    canonical_path = (canonical_coordexp_repo_root() / path).resolve(strict=False)
    if canonical_path.exists():
        return canonical_path
    return worktree_path


def _resolve_local_model_path(path_value: str | Path) -> Path | None:
    text = str(path_value)
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve(strict=False)
    if text.startswith(("model_cache/", "./", "../")):
        return _resolve_repo_path(path)
    return None


def _require_existing_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"coverage ledger preflight {label} not found: {path}")


def _require_existing_path(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"coverage ledger preflight {label} not found: {path}")


def _tokenizer_id(swift_template: Any) -> str:
    tokenizer = getattr(swift_template, "tokenizer", None)
    name = getattr(tokenizer, "name_or_path", None)
    if isinstance(name, str) and name:
        return name
    return type(tokenizer).__name__ if tokenizer is not None else "unknown-tokenizer"


def _torch_dtype(value: Any) -> Any:
    import torch

    if value is None:
        return torch.bfloat16
    text = str(value)
    if text == "bfloat16":
        return torch.bfloat16
    if text == "float16":
        return torch.float16
    if text in {"float32", "fp32"}:
        return torch.float32
    return value


def _escape_pointer_part(part: str) -> str:
    return part.replace("~", "~0").replace("/", "~1")


def _join_pointer(parent: str, key: str) -> str:
    suffix = _escape_pointer_part(key)
    return f"/{suffix}" if parent == "" else f"{parent}/{suffix}"


def _leaf_paths(value: Any, parent: str) -> set[str]:
    if isinstance(value, Mapping):
        paths: set[str] = set()
        for key, child in value.items():
            paths.update(_leaf_paths(child, _join_pointer(parent, str(key))))
        return paths or {parent or "/"}
    if isinstance(value, list):
        paths = set()
        for index, child in enumerate(value):
            paths.update(_leaf_paths(child, _join_pointer(parent, str(index))))
        return paths or {parent or "/"}
    return {parent or "/"}


def _changed_paths(left: Any, right: Any, parent: str = "") -> set[str]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        paths: set[str] = set()
        for key in sorted(set(left) | set(right)):
            path = _join_pointer(parent, str(key))
            if key not in left:
                paths.update(_leaf_paths(right[key], path))
            elif key not in right:
                paths.update(_leaf_paths(left[key], path))
            else:
                paths.update(_changed_paths(left[key], right[key], path))
        return paths
    if isinstance(left, list) and isinstance(right, list):
        if left == right:
            return set()
        paths = set()
        for index in range(max(len(left), len(right))):
            path = _join_pointer(parent, str(index))
            if index >= len(left):
                paths.update(_leaf_paths(right[index], path))
            elif index >= len(right):
                paths.update(_leaf_paths(left[index], path))
            else:
                paths.update(_changed_paths(left[index], right[index], path))
        return paths
    return set() if left == right else {parent or "/"}


__all__ = [
    "CoverageLedgerPreflightResult",
    "assert_smoke_config_diff_allowed",
    "build_preflight_swift_template",
    "resolve_coverage_ledger_train_selection",
    "resolve_overlay_render_image_path",
    "resolve_visual_grid_geometry",
    "run_coverage_ledger_preflight",
    "select_preflight_row_indices",
]
