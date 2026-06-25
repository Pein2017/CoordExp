"""Strict preflight for coverage-ledger smoke artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from src.config.loader import ConfigLoader
from src.config.schema import CoordTokensConfig, DetectionTrainingConfig
from src.config.strict_dataclass import dataclass_asdict_no_none
from src.common.model_paths import canonical_coordexp_repo_root
from src.coord_tokens.template_adapter import apply_coord_template_adapter
from src.data_collators import build_dataset_metrics_collator
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
from src.datasets.wrappers.packed_caption import build_static_packed_dataset
from src.trainers.batch_extras import BATCH_EXTRAS_KEYS
from src.trainers.teacher_forcing.forwards import prepare_forward_inputs
from src.training.teacher_forcing.constants import TEACHER_FORCING_TARGET_IR_KEY


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
    packed_materialization_path = write_static_packed_materialization_preflight(
        ledger_config,
        dataset=dataset,
        swift_template=template,
        output_root=Path(output_root),
    )
    artifact_result = replace(
        artifact_result,
        packed_materialization_path=packed_materialization_path,
    )
    return CoverageLedgerPreflightResult(
        artifact_result=artifact_result,
        selected_row_indices=selected_row_indices,
        selected_sample_ids=tuple(sidecar.sample_id for sidecar in sidecars),
    )


def write_static_packed_materialization_preflight(
    training_config: DetectionTrainingConfig,
    *,
    dataset: DetectionTrainingDataset,
    swift_template: Any,
    output_root: Path,
) -> Path:
    """Materialize and validate one real static-packed coverage-ledger batch."""

    if not bool(training_config.training.get("packing", False)):
        raise ValueError("coverage ledger packed preflight requires training.packing=true")
    if not bool(training_config.training.get("eval_packing", False)):
        raise ValueError("coverage ledger packed preflight requires training.eval_packing=true")
    if not training_config.packing.static_packing:
        raise ValueError(
            "coverage ledger packed preflight requires packing.static_packing=true"
        )
    if training_config.packing.padding_free_packed:
        raise ValueError(
            "coverage ledger packed preflight does not support padding_free_packed=true"
        )

    packing_length = _resolve_preflight_packing_length(training_config, swift_template)
    min_fill_ratio = float(training_config.training.get("packing_min_fill_ratio", 0.65))
    packing_drop_last = bool(training_config.training.get("packing_drop_last", True))
    allow_single_long = bool(training_config.training.get("packing_allow_single_long", True))
    wait_timeout_s = float(training_config.training.get("packing_wait_timeout_s", 7200.0))
    # Preflight runs after the real Qwen3-VL processor/template is constructed.
    # Forked length workers can park inside processor state on some nodes; the
    # no-training gate values determinism over parallel cache build speed.
    length_precompute_workers = 1

    packed_dataset = build_static_packed_dataset(
        dataset,
        template=swift_template,
        packing_length=packing_length,
        min_fill_ratio=min_fill_ratio,
        packing_drop_last=packing_drop_last,
        dataloader_drop_last=False,
        allow_single_long=allow_single_long,
        cache_dir=Path(output_root) / "static_packing_preflight_cache",
        fingerprint={
            "schema_version": "coverage_ledger_static_packed_preflight_v0",
            "template_id": training_config.detection_template.id,
            "objective_id": training_config.objective.id,
            "rollin_policy": str(
                training_config.objective.target_ir.rollin_policy.name
            ),
            "packing_length": packing_length,
        },
        world_size=1,
        train_dataloader_shuffle=False,
        wait_timeout_s=wait_timeout_s,
        length_precompute_workers=length_precompute_workers,
    )

    pack_index, pack = _select_two_segment_pack(packed_dataset)
    collator = build_dataset_metrics_collator(
        swift_template,
        swift_template.data_collator,
        coverage_ledger_cfg=training_config.objective.terms.coverage_ledger,
    )
    _ensure_preflight_packing_dummy_model(training_config, swift_template)
    batch = collator([pack])
    ignored_keys = (
        "labels",
        "attention_mask",
        "sample_id",
        "training_sidecars",
        TEACHER_FORCING_TARGET_IR_KEY,
        *BATCH_EXTRAS_KEYS,
    )
    dummy_model = SimpleNamespace(config=SimpleNamespace(model_type="qwen3_vl"))
    _core_model, inputs_for_model, _model_type = prepare_forward_inputs(
        model=dummy_model,
        inputs=batch,
        ignored_keys=ignored_keys,
        packing_enabled=True,
        where="coverage ledger static packed preflight",
    )

    offsets = _packed_segment_offsets_payload(batch.get("packed_segment_offsets"))
    segment_count = len(offsets)
    if segment_count < 2:
        raise ValueError("packed preflight expected at least two packed segments")
    shifted_sidecars = _coverage_ledger_sidecars_from_batch(batch)
    if len(shifted_sidecars) != segment_count:
        raise ValueError(
            "packed preflight coverage-ledger sidecar count must match segment count; "
            f"sidecars={len(shifted_sidecars)} segments={segment_count}"
        )
    teacher_forcing_irs = batch.get(TEACHER_FORCING_TARGET_IR_KEY)
    if not isinstance(teacher_forcing_irs, tuple) or len(teacher_forcing_irs) != segment_count:
        raise ValueError(
            "packed preflight teacher_forcing_target_ir count must match segment count"
        )

    image_grid = _nested_list(batch.get("image_grid_thw"), field_name="image_grid_thw")
    if len(image_grid) != segment_count:
        raise ValueError(
            "packed preflight image_grid_thw row count must match segment count; "
            f"image_grid_rows={len(image_grid)} segments={segment_count}"
        )
    for row_index, row in enumerate(image_grid):
        if len(row) != 3 or int(row[0]) != 1:
            raise ValueError(
                "packed preflight supports one image frame per segment; "
                f"row={row_index} image_grid_thw={row}"
            )
    if batch.get("pixel_values_videos") is not None or batch.get("video_grid_thw") is not None:
        raise ValueError("packed preflight does not support video tensors")

    position_ids_shape = _shape_list(inputs_for_model.get("position_ids"))
    if len(position_ids_shape) != 3 or int(position_ids_shape[0]) != 4:
        raise ValueError(
            "packed preflight expected Qwen3-VL 4-row position_ids after forward prep; "
            f"shape={position_ids_shape}"
        )

    sample_ids = [str(sidecar.sample_id) for sidecar in shifted_sidecars]
    if sample_ids != [str(record["sample_id"]) for record in offsets]:
        raise ValueError(
            "packed preflight shifted sidecar sample order must match segment offsets"
        )

    source_row_indices = tuple(int(index) for index in getattr(dataset, "source_row_indices"))
    pack_local_indices = [int(index) for index in packed_dataset.pack_plan[int(pack_index)]]
    payload = {
        "schema_version": "coverage_ledger_static_packed_materialization_v0",
        "packing_length": int(packing_length),
        "packing_min_fill_ratio": float(min_fill_ratio),
        "pack_index": int(pack_index),
        "pack_local_indices": pack_local_indices,
        "pack_source_row_indices": [source_row_indices[index] for index in pack_local_indices],
        "segment_count": int(segment_count),
        "segment_offsets": offsets,
        "sample_ids": sample_ids,
        "input_ids_shape": _shape_list(batch.get("input_ids")),
        "labels_shape": _shape_list(batch.get("labels")),
        "attention_mask_shape": _shape_list(batch.get("attention_mask")),
        "raw_position_ids_shape": _shape_list(batch.get("position_ids")),
        "forward_position_ids_shape": position_ids_shape,
        "text_position_ids_shape": _shape_list(batch.get("text_position_ids")),
        "cu_seq_lens_q": _flat_int_list(batch.get("cu_seq_lens_q")),
        "cu_seq_lens_k": _flat_int_list(batch.get("cu_seq_lens_k")),
        "max_length_q": _maybe_int(batch.get("max_length_q")),
        "max_length_k": _maybe_int(batch.get("max_length_k")),
        "image_grid_thw": image_grid,
        "image_grid_row_count": len(image_grid),
        "teacher_forcing_target_ir_count": len(teacher_forcing_irs),
        "coverage_ledger_sidecar_count": len(shifted_sidecars),
        "shifted_sidecars": [
            {
                "sample_id": sidecar.sample_id,
                "prompt_end_position": int(sidecar.prompt_end_position),
                "image_grid_thw": list(sidecar.image_grid_thw),
                "object_count": len(sidecar.object_entries),
                "first_object_coord_label_positions": (
                    list(sidecar.object_entries[0].coord_label_positions)
                    if sidecar.object_entries
                    else []
                ),
            }
            for sidecar in shifted_sidecars
        ],
    }

    out_path = Path(output_root) / "ledger" / "packed_materialization.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return out_path


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


def _resolve_preflight_packing_length(
    training_config: DetectionTrainingConfig,
    swift_template: Any,
) -> int:
    raw = (
        getattr(swift_template, "max_length", None)
        or training_config.template.get("max_length")
        or training_config.training.get("global_max_length")
    )
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("coverage ledger packed preflight could not resolve max length") from exc
    if value <= 0:
        raise ValueError("coverage ledger packed preflight requires positive max length")
    return value


def _ensure_preflight_packing_dummy_model(
    training_config: DetectionTrainingConfig,
    swift_template: Any,
) -> None:
    if getattr(swift_template, "model", None) is not None:
        return
    if getattr(swift_template, "dummy_model", None) is not None:
        return
    if getattr(swift_template, "model_info", None) is None:
        return
    model_type = training_config.model.get("model_type")
    if model_type is None:
        model_type = getattr(swift_template.model_info, "model_type", None)
    if model_type is None:
        return

    import torch

    local_model_path = _resolve_local_model_path(training_config.model["model"])
    model_path = (
        str(local_model_path)
        if local_model_path is not None
        else str(training_config.model["model"])
    )
    with torch.device("meta"):
        dummy_model, _processor = get_model_processor(
            model_path,
            return_dummy_model=True,
            model_type=str(model_type),
            torch_dtype=_torch_dtype(training_config.model.get("torch_dtype")),
            download_model=False,
        )
    swift_template.dummy_model = dummy_model


def _select_two_segment_pack(packed_dataset: Any) -> tuple[int, Sequence[Mapping[str, Any]]]:
    for pack_index in range(len(packed_dataset)):
        pack = packed_dataset[int(pack_index)]
        if isinstance(pack, Sequence) and not isinstance(pack, (str, bytes)):
            if len(pack) >= 2:
                return int(pack_index), pack
    raise ValueError(
        "coverage ledger packed preflight could not find a two-segment static pack"
    )


def _packed_segment_offsets_payload(value: Any) -> list[dict[str, Any]]:
    if value is None:
        raise ValueError("packed preflight batch missing packed_segment_offsets")
    if isinstance(value, tuple) and all(
        hasattr(offset, "token_start") and hasattr(offset, "token_end")
        for offset in value
    ):
        return [
            {
                "sample_id": str(offset.sample_id),
                "packed_row_index": int(offset.packed_row_index),
                "segment_index": int(offset.segment_index),
                "token_start": int(offset.token_start),
                "token_end": int(offset.token_end),
            }
            for offset in value
        ]

    boundaries = _flat_int_list(value)
    if len(boundaries) < 2:
        raise ValueError("packed_segment_offsets boundaries must contain at least two values")
    return [
        {
            "sample_id": "",
            "packed_row_index": 0,
            "segment_index": index,
            "token_start": int(start),
            "token_end": int(end),
        }
        for index, (start, end) in enumerate(zip(boundaries, boundaries[1:]))
    ]


def _coverage_ledger_sidecars_from_batch(batch: Mapping[str, Any]) -> tuple[CoverageLedgerSidecar, ...]:
    training_sidecars = batch.get("training_sidecars")
    supervision = getattr(training_sidecars, "supervision", None)
    payloads = tuple(getattr(supervision, "payloads", ()))
    sidecars = tuple(
        payload for payload in payloads if isinstance(payload, CoverageLedgerSidecar)
    )
    if not sidecars:
        raise ValueError("packed preflight batch missing coverage-ledger sidecars")
    return sidecars


def _nested_list(value: Any, *, field_name: str) -> list[list[int]]:
    raw = _to_python(value)
    if not isinstance(raw, list):
        raise TypeError(f"{field_name} must be a tensor/list with rows")
    if raw and all(isinstance(item, int) for item in raw):
        return [[int(item) for item in raw]]
    rows: list[list[int]] = []
    for row_index, row in enumerate(raw):
        if not isinstance(row, list):
            raise TypeError(f"{field_name}[{row_index}] must be a list")
        rows.append([int(item) for item in row])
    return rows


def _flat_int_list(value: Any) -> list[int]:
    raw = _to_python(value)
    if isinstance(raw, list):
        if raw and isinstance(raw[0], list):
            if len(raw) != 1:
                raise ValueError("expected a flat vector or a single-row vector")
            raw = raw[0]
        return [int(item) for item in raw]
    if isinstance(raw, tuple):
        return [int(item) for item in raw]
    raise TypeError(f"expected vector-like value, got {type(value).__name__}")


def _shape_list(value: Any) -> list[int]:
    shape = getattr(value, "shape", None)
    if shape is None:
        return []
    return [int(item) for item in shape]


def _maybe_int(value: Any) -> int | None:
    if value is None:
        return None
    item = getattr(value, "item", None)
    if callable(item):
        value = item()
    return int(value)


def _to_python(value: Any) -> Any:
    if value is None:
        raise ValueError("expected value, got None")
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    if isinstance(value, tuple):
        return list(value)
    return value


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
