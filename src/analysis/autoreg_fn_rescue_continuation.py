"""Pure helpers for the FN-rescue continuation analysis.

This module intentionally contains no model loading or GPU execution.  It owns
the config surface and deterministic helper contracts that later FN-rescue
stages reuse for source selection, hint construction, parsing, scoring, and
region membership.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from numbers import Integral, Real
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import yaml

from src.coord_tokens.codec import int_to_token, token_to_int

EXPECTED_EVIDENCE_SCOPE = "val200_attention_atlas_linked_fn_stratified"
RESCUE_TIERS = {"desc_only", "desc_x1", "desc_x1_wrong_control"}
FN_RESCUE_STAGES = (
    "select_cases",
    "feasibility",
    "rescue_decode",
    "attention_replay",
    "merge",
    "report",
    "gallery",
)
FN_RESCUE_MERGE_JSONL_FILES = MappingProxyType(
    {
        "selected_rescue_cases": "selected_rescue_cases.jsonl",
        "rescue_rows": "rescue_rows.jsonl",
        "rescue_generation_rows": "rescue_generation_rows.jsonl",
        "rescue_replay_prefix_rows": "rescue_replay_prefix_rows.jsonl",
        "rescue_attention_region_rows": "rescue_attention_region_rows.jsonl",
        "rescue_decision_context_rows": "rescue_decision_context_rows.jsonl",
        "rescue_candidate_region_rows": "rescue_candidate_region_rows.jsonl",
        "wrong_control_rows": "wrong_control_rows.jsonl",
    }
)
_COORD_PREFIX_RE = re.compile(r"<\|coord_(0|[1-9]\d{0,2})\|>")
_SHARD_LABEL_RE = re.compile(r"^shard_(\d{3})-of-(\d{3})$")
_SOURCE_PATH_FIELDS = (
    "checkpoint",
    "dataset_jsonl",
    "attention_atlas_root",
    "source_selected_cases",
    "source_candidate_regions",
    "rollout_anatomy_per_row",
    "gt_vs_pred_scored",
    "pred_token_trace",
    "infer_resolved_config",
    "lane_c_study_config",
)
_ACCEPTABLE_TORCH_DTYPES = {"bfloat16", "float16", "float32"}
_GENERATION_TERMINAL_SUFFIXES = (
    "<|im_end|>",
    "<|endoftext|>",
    "<|eos|>",
    "<|eot_id|>",
    "</s>",
    "<|pad|>",
    "<pad>",
)
_DISALLOWED_GENERATE_RETURN_KWARGS = frozenset(
    {
        "output_attentions",
        "return_dict_in_generate",
        "output_hidden_states",
        "output_scores",
        "output_logits",
        "output_router_logits",
    }
)


def _load_repo_module_from_path(module_name: str, relative_path: str) -> Any:
    module_path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_compact_rows = _load_repo_module_from_path(
    "_coordexp_common_detection_compact_rows_for_fn_rescue",
    "common/detection_compact_rows.py",
)

BOX_START_TOKEN = _compact_rows.BOX_START_TOKEN
COMPACT_DESC_FORBIDDEN_SUBSTRINGS = _compact_rows.COMPACT_DESC_FORBIDDEN_SUBSTRINGS
OBJECT_REF_START_TOKEN = _compact_rows.OBJECT_REF_START_TOKEN
render_compact_row = _compact_rows.render_compact_row


@dataclass(frozen=True)
class FnRescuePaths:
    artifact_root: Path
    checkpoint: Path
    dataset_jsonl: Path
    attention_atlas_root: Path
    source_selected_cases: Path
    source_candidate_regions: Path
    rollout_anatomy_per_row: Path
    gt_vs_pred_scored: Path
    pred_token_trace: Path
    infer_resolved_config: Path | None
    lane_c_study_config: Path | None


@dataclass(frozen=True)
class FnRescueSelectionConfig:
    evidence_scope: str
    sample_limit: int
    per_stratum_cap: int
    duplicate_iou_threshold: float
    wrong_control_overlap_threshold: float
    context_expansion_norm1000: int


@dataclass(frozen=True)
class FnRescueExecutionConfig:
    attn_implementation: str
    torch_dtype: str
    decoding: str
    do_sample: bool
    num_beams: int
    max_new_tokens_desc_only: int
    max_new_tokens_desc_x1: int
    gallery_per_bucket: int


@dataclass(frozen=True)
class FnRescueConfig:
    paths: FnRescuePaths
    selection: FnRescueSelectionConfig
    execution: FnRescueExecutionConfig
    config_path: Path | None = None
    source_artifacts: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class ParsedRescueGeneration:
    valid_parse: bool
    generated_coord_tokens: tuple[str, ...]
    generated_box_xyxy: tuple[int, int, int, int] | None
    parse_errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class RescueBoxScore:
    valid_parse: bool
    target_iou: float
    success_iou30: bool
    success_iou50: bool
    success_iou75: bool
    same_desc_duplicate_iou95: bool
    max_same_desc_existing_iou: float
    duplicate_source_kind: str | None
    duplicate_source_raw_pred_idx: int | None
    duplicate_source_guarded_pred_idx: int | None
    duplicate_source_bbox_xyxy: tuple[int, int, int, int] | None
    primary_rescue_success: bool


@dataclass(frozen=True)
class WrongControlSource:
    kind: str
    bbox_xyxy: tuple[int, int, int, int] | None
    x1: int | None
    source_index: int | None = None
    rejected_candidates: tuple[MappingProxyType[str, Any], ...] = ()
    skip_reason: str | None = None


@dataclass(frozen=True)
class ChatTemplateContinuationCheck:
    feasible: bool
    status: str
    errors: tuple[str, ...] = ()
    rendered_prompt: str | None = None
    last_token_text: str | None = None
    input_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class RescueDecodeResult:
    generated_tail_ids: tuple[int, ...]
    generated_tail_text: str
    parsed_generation: ParsedRescueGeneration
    hint_x1: int | None
    generation_kwargs: Mapping[str, Any]
    full_generated_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class RescueAttentionReplayResult:
    status: str
    rows: tuple[dict[str, Any], ...]
    errors: tuple[str, ...] = ()
    query_role: str | None = None
    query_index: int | None = None
    last_token_text: str | None = None
    replay_prefix_token_ids: tuple[int, ...] = ()


def _load_geometry_helpers() -> tuple[Any, Any]:
    geometry_path = Path(__file__).resolve().parents[1] / "datasets" / "geometry.py"
    spec = importlib.util.spec_from_file_location(
        "_coordexp_datasets_geometry_for_fn_rescue", geometry_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load geometry helpers from {geometry_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.valid_xyxy_box, module.box_iou_xyxy


valid_xyxy_box, box_iou_xyxy = _load_geometry_helpers()


def load_fn_rescue_config(path: Path) -> FnRescueConfig:
    """Load and validate an FN-rescue config YAML file."""

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"FN-rescue config does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"FN-rescue config must be a mapping: {config_path}")

    paths_payload = _require_mapping(payload, "paths")
    selection_payload = _require_mapping(payload, "selection")
    execution_payload = _require_mapping(payload, "execution")

    paths = FnRescuePaths(
        artifact_root=_path_value(paths_payload, "artifact_root"),
        checkpoint=_path_value(paths_payload, "checkpoint"),
        dataset_jsonl=_path_value(paths_payload, "dataset_jsonl"),
        attention_atlas_root=_path_value(paths_payload, "attention_atlas_root"),
        source_selected_cases=_path_value(paths_payload, "source_selected_cases"),
        source_candidate_regions=_path_value(paths_payload, "source_candidate_regions"),
        rollout_anatomy_per_row=_path_value(paths_payload, "rollout_anatomy_per_row"),
        gt_vs_pred_scored=_path_value(paths_payload, "gt_vs_pred_scored"),
        pred_token_trace=_path_value(paths_payload, "pred_token_trace"),
        infer_resolved_config=_optional_path_value(
            paths_payload, "infer_resolved_config"
        ),
        lane_c_study_config=_optional_path_value(paths_payload, "lane_c_study_config"),
    )
    selection = FnRescueSelectionConfig(
        evidence_scope=str(_require_key(selection_payload, "evidence_scope")),
        sample_limit=_positive_int(selection_payload, "sample_limit"),
        per_stratum_cap=_positive_int(selection_payload, "per_stratum_cap"),
        duplicate_iou_threshold=_float_value(
            selection_payload, "duplicate_iou_threshold"
        ),
        wrong_control_overlap_threshold=_float_value(
            selection_payload, "wrong_control_overlap_threshold"
        ),
        context_expansion_norm1000=_nonnegative_int(
            selection_payload, "context_expansion_norm1000"
        ),
    )
    execution = FnRescueExecutionConfig(
        attn_implementation=str(_require_key(execution_payload, "attn_implementation")),
        torch_dtype=str(_require_key(execution_payload, "torch_dtype")),
        decoding=str(_require_key(execution_payload, "decoding")),
        do_sample=_bool_value(execution_payload, "do_sample"),
        num_beams=_int_value(execution_payload, "num_beams"),
        max_new_tokens_desc_only=_int_value(
            execution_payload, "max_new_tokens_desc_only"
        ),
        max_new_tokens_desc_x1=_int_value(execution_payload, "max_new_tokens_desc_x1"),
        gallery_per_bucket=_positive_int(execution_payload, "gallery_per_bucket"),
    )

    _validate_config(paths, selection, execution)
    return FnRescueConfig(
        paths=paths,
        selection=selection,
        execution=execution,
        config_path=config_path.resolve(strict=False),
        source_artifacts=_build_source_artifact_summary(paths),
    )


def coord_token(value: int) -> str:
    """Render one compact-full norm1000 coordinate token."""

    value = int(value)
    if value < 0 or value > 999:
        raise ValueError(f"coord token out of norm1000 range: {value}")
    return int_to_token(value)


def _clamp_norm1000_coord(value: int) -> int:
    return max(0, min(999, int(value)))


def _hint_x1_record(raw_x1: int | None) -> dict[str, Any]:
    if raw_x1 is None:
        return {
            "hint_x1": None,
            "hint_x1_raw": None,
            "hint_x1_used": None,
            "hint_x1_clamped": False,
        }
    raw_value = int(raw_x1)
    used_value = _clamp_norm1000_coord(raw_value)
    return {
        "hint_x1": used_value,
        "hint_x1_raw": raw_value,
        "hint_x1_used": used_value,
        "hint_x1_clamped": raw_value != used_value,
    }


def strip_generation_terminal(text: str) -> str:
    """Trim trailing generation terminal markers while preserving interior text."""

    out = str(text)
    while True:
        previous = out
        out = out.rstrip()
        for suffix in _GENERATION_TERMINAL_SUFFIXES:
            if out.endswith(suffix):
                out = out[: -len(suffix)]
                break
        if out == previous:
            return out


def render_rescue_hint_row_prefix(desc: str, *, tier: str, x1: int | None) -> str:
    """Render a partial compact-full assistant row for FN-rescue continuation."""

    clean_desc = str(desc).strip()
    if not clean_desc:
        raise ValueError("target desc must be nonempty")
    if any(item in clean_desc for item in COMPACT_DESC_FORBIDDEN_SUBSTRINGS):
        raise ValueError("target desc is not compact-row safe")
    if tier == "desc_only":
        if x1 is not None:
            raise ValueError("desc_only must not receive x1")
        return render_compact_row(
            clean_desc,
            (),
            include_object_ref_marker=True,
            include_bbox_start_marker=True,
        )
    if tier in {"desc_x1", "desc_x1_wrong_control"}:
        if x1 is None:
            raise ValueError(f"{tier} requires x1")
        return render_compact_row(
            clean_desc,
            (coord_token(_clamp_norm1000_coord(int(x1))),),
            include_object_ref_marker=True,
            include_bbox_start_marker=True,
        )
    raise ValueError(f"unknown rescue tier: {tier}")


def prefix_text_from_raw_compact_predictions(
    *,
    rollout_rows: Sequence[Mapping[str, Any]],
    scored_row: Mapping[str, Any],
    token_trace_row: Mapping[str, Any] | None = None,
    prefix_depth: int,
) -> str:
    """Reconstruct compact prefix text from raw coord-token predictions."""

    if int(prefix_depth) <= 0:
        return ""

    raw_output_json = scored_row.get("raw_output_json")
    if not isinstance(raw_output_json, Mapping):
        raise ValueError("scored_row.raw_output_json must be a mapping with raw objects")
    raw_objects = raw_output_json.get("objects")
    if not isinstance(raw_objects, list):
        raise ValueError("scored_row.raw_output_json.objects must be a list")

    rendered_rows: list[tuple[int, str]] = []
    for rollout_row in rollout_rows:
        raw_pred_idx = rollout_row.get("raw_pred_idx")
        if raw_pred_idx is None:
            continue
        raw_idx = int(raw_pred_idx)
        if raw_idx < 0 or raw_idx >= int(prefix_depth):
            continue
        if bool(rollout_row.get("suppressed_by_guard", False)):
            continue
        if raw_idx >= len(raw_objects):
            raise ValueError(
                f"raw_output_json.objects missing raw_pred_idx {raw_idx} for prefix reconstruction"
            )
        raw_object = raw_objects[raw_idx]
        if not isinstance(raw_object, Mapping):
            raise ValueError(f"raw_output_json.objects[{raw_idx}] must be a mapping")
        raw_desc = str(raw_object.get("desc") or "").strip()
        pred_desc = str(rollout_row.get("pred_desc") or "").strip()
        if raw_desc != pred_desc:
            raise ValueError(
                f"raw desc mismatch for raw_pred_idx {raw_idx}: {raw_desc!r} != {pred_desc!r}"
            )
        bbox_2d = raw_object.get("bbox_2d")
        if not isinstance(bbox_2d, Sequence) or isinstance(bbox_2d, (str, bytes)):
            raise ValueError(
                f"raw_output_json.objects[{raw_idx}].bbox_2d must be a coord-token sequence"
            )
        coord_tokens = tuple(str(token) for token in bbox_2d)
        if len(coord_tokens) != 4:
            raise ValueError(
                f"raw_output_json.objects[{raw_idx}].bbox_2d must contain exactly 4 tokens"
            )
        if any(not token.startswith("<|coord_") for token in coord_tokens):
            raise ValueError(
                f"raw_output_json.objects[{raw_idx}].bbox_2d must contain raw coord tokens"
            )
        rendered_rows.append(
            (
                raw_idx,
                render_compact_row(
                    pred_desc,
                    coord_tokens,
                    include_object_ref_marker=True,
                    include_bbox_start_marker=True,
                ),
            )
        )

    prefix_text = strip_generation_terminal(
        "".join(text for _, text in sorted(rendered_rows, key=lambda item: item[0]))
    )
    if token_trace_row is not None and prefix_text:
        generated_token_text = strip_generation_terminal(
            _token_text_from_row(token_trace_row, key="generated_token_text")
        )
        if prefix_text not in generated_token_text:
            raise ValueError(
                "reconstructed prefix was not provably present in generated_token_text"
            )
    return prefix_text


def build_rescue_assistant_prefix(
    prefix_text: str, target_desc: str, tier: str, hint_x1: int | None
) -> str:
    """Append a partial rescue hint row onto a compact prefix."""

    return strip_generation_terminal(prefix_text) + render_rescue_hint_row_prefix(
        target_desc,
        tier=tier,
        x1=hint_x1,
    )


def check_chat_template_continuation_feasibility(
    model_handle: Any,
    messages: Sequence[Mapping[str, Any]],
    assistant_prefix: str,
    tier: str,
    hint_x1: int | None = None,
) -> ChatTemplateContinuationCheck:
    """Check whether the chat template can continue directly from assistant text."""

    try:
        rendered_prompt = model_handle.processor.apply_chat_template(
            [dict(message) for message in messages],
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=True,
        )
    except Exception as exc:
        return ChatTemplateContinuationCheck(
            feasible=False,
            status="chat_template_error",
            errors=(str(exc),),
        )
    if not isinstance(rendered_prompt, str):
        return ChatTemplateContinuationCheck(
            feasible=False,
            status="chat_template_non_string",
            errors=("processor.apply_chat_template(..., tokenize=False) must return str",),
        )

    errors: list[str] = []
    if rendered_prompt.endswith("<|im_end|>"):
        errors.append("rendered prompt unexpectedly ends with <|im_end|>")
    if not rendered_prompt.endswith(str(assistant_prefix)):
        errors.append("rendered prompt does not end with the exact assistant_prefix")

    try:
        input_ids = _encode_text_ids(model_handle, rendered_prompt)
    except Exception as exc:
        return ChatTemplateContinuationCheck(
            feasible=False,
            status="tokenize_error",
            errors=(str(exc),),
            rendered_prompt=rendered_prompt,
        )
    tokenizer = _resolve_tokenizer(model_handle)
    trimmed_ids = _trim_trailing_pad_ids(input_ids, _resolve_pad_token_id(tokenizer))
    last_token_text = (
        _decode_single_token_text(tokenizer, trimmed_ids[-1]) if trimmed_ids else None
    )
    expected_last = (
        BOX_START_TOKEN
        if tier == "desc_only"
        else coord_token(_require_hint_x1(tier=tier, hint_x1=hint_x1))
    )
    if last_token_text != expected_last:
        errors.append(
            f"last non-pad token mismatch: expected {expected_last!r}, got {last_token_text!r}"
        )

    status = "ok"
    if errors:
        if any("exact assistant_prefix" in error for error in errors):
            status = "prompt_suffix_mismatch"
        elif any("<|im_end|>" in error for error in errors):
            status = "prompt_terminalized"
        else:
            status = "last_token_mismatch"
    return ChatTemplateContinuationCheck(
        feasible=not errors,
        status=status,
        errors=tuple(errors),
        rendered_prompt=rendered_prompt,
        last_token_text=last_token_text,
        input_ids=trimmed_ids,
    )


def parse_generated_bbox(
    *, tier: str, generated_tail_text: str, hint_x1: int | None
) -> ParsedRescueGeneration:
    """Parse the leading generated coord tokens into a rescue xyxy box."""

    if tier not in RESCUE_TIERS:
        raise ValueError(f"unknown rescue tier: {tier}")

    required_generated = 4 if tier == "desc_only" else 3
    if tier == "desc_only" and hint_x1 is not None:
        return ParsedRescueGeneration(False, (), None, ("desc_only received hint_x1",))
    if tier != "desc_only" and hint_x1 is None:
        return ParsedRescueGeneration(False, (), None, (f"{tier} requires hint_x1",))

    tokens = _leading_coord_tokens(str(generated_tail_text), required_generated)
    if len(tokens) < required_generated:
        return ParsedRescueGeneration(
            False,
            tokens,
            None,
            (f"expected {required_generated} leading coord tokens",),
        )

    try:
        generated_values = [token_to_int(token) for token in tokens]
        box = (
            generated_values
            if tier == "desc_only"
            else [int(hint_x1), *generated_values]
        )
    except ValueError as exc:
        return ParsedRescueGeneration(False, tokens, None, (str(exc),))

    if not valid_xyxy_box(box):
        return ParsedRescueGeneration(False, tokens, None, ("invalid xyxy box",))
    return ParsedRescueGeneration(True, tokens, _box_tuple(box))


def decode_rescue_tail(
    model_handle: Any,
    processor_inputs: Mapping[str, Any],
    tier: str,
    assistant_prefix_token_ids: Sequence[int] | None = None,
) -> RescueDecodeResult:
    """Run greedy continuation and parse only the generated tail tokens."""

    if tier not in RESCUE_TIERS:
        raise ValueError(f"unknown rescue tier: {tier}")
    tokenizer = _resolve_tokenizer(model_handle)
    model = getattr(model_handle, "model", None)
    if model is None or not hasattr(model, "generate"):
        raise ValueError("model_handle.model.generate is required")

    prompt_ids = _extract_first_token_row(_require_key(processor_inputs, "input_ids"))
    hint_x1 = _infer_hint_x1(
        tier=tier,
        tokenizer=tokenizer,
        assistant_prefix_token_ids=assistant_prefix_token_ids,
        prompt_ids=prompt_ids,
    )
    generate_inputs = _scrub_generate_inputs(processor_inputs)
    generation_kwargs = {
        **generate_inputs,
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": 4 if tier == "desc_only" else 3,
        "pad_token_id": _resolve_pad_token_id(tokenizer),
        "eos_token_id": _resolve_im_end_token_id(tokenizer),
    }
    generated = model.generate(**generation_kwargs)
    full_generated_ids = _extract_first_token_row(generated)
    prompt_length = len(prompt_ids)
    if len(full_generated_ids) < prompt_length:
        raise ValueError("generated sequence was shorter than the prompt")
    tail_ids = tuple(full_generated_ids[prompt_length:])
    tail_text = _decode_token_ids(tokenizer, tail_ids)
    parsed_generation = parse_generated_bbox(
        tier=tier,
        generated_tail_text=tail_text,
        hint_x1=hint_x1,
    )
    return RescueDecodeResult(
        generated_tail_ids=tail_ids,
        generated_tail_text=tail_text,
        parsed_generation=parsed_generation,
        hint_x1=hint_x1,
        generation_kwargs=MappingProxyType(dict(generation_kwargs)),
        full_generated_ids=tuple(full_generated_ids),
    )


def score_rescue_box(
    *,
    generated_box: Sequence[Any] | None,
    target_box: Sequence[Any],
    same_desc_rollout_boxes: Sequence[Any],
    duplicate_iou_threshold: float = 0.95,
) -> RescueBoxScore:
    """Score a generated rescue box against the target and duplicate policy."""

    valid_parse = generated_box is not None and valid_xyxy_box(generated_box)
    target_iou = box_iou_xyxy(generated_box or [], target_box) if valid_parse else 0.0
    duplicate = _max_duplicate_source(generated_box, same_desc_rollout_boxes)
    max_same_desc_existing_iou = duplicate["iou"]
    same_desc_duplicate = max_same_desc_existing_iou > float(duplicate_iou_threshold)

    return RescueBoxScore(
        valid_parse=valid_parse,
        target_iou=target_iou,
        success_iou30=target_iou >= 0.30,
        success_iou50=target_iou >= 0.50,
        success_iou75=target_iou >= 0.75,
        same_desc_duplicate_iou95=same_desc_duplicate,
        max_same_desc_existing_iou=max_same_desc_existing_iou,
        duplicate_source_kind=duplicate["kind"],
        duplicate_source_raw_pred_idx=duplicate["raw_pred_idx"],
        duplicate_source_guarded_pred_idx=duplicate["guarded_pred_idx"],
        duplicate_source_bbox_xyxy=duplicate["bbox_xyxy"],
        primary_rescue_success=bool(valid_parse and target_iou >= 0.50)
        and not same_desc_duplicate,
    )


def canonical_target_gt_idx(row: Mapping[str, Any]) -> int:
    """Return the canonical target GT index from linked atlas ledger rows."""

    if row.get("target_gt_idx") is not None:
        return int(row["target_gt_idx"])
    if row.get("intended_target_gt_idx") is not None:
        return int(row["intended_target_gt_idx"])
    raise KeyError("row must contain target_gt_idx or intended_target_gt_idx")


def fn_rescue_stratum_key(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    """Build the deterministic four-component FN-rescue stratum key."""

    prefix_quality_bucket = str(row["prefix_quality"])
    attribution = row.get("x1_top_peak_attribution")
    if attribution == "same_desc_competitor_gt_object":
        binding_bucket = "same_desc_competitor"
    elif attribution == "no_local_object_diffuse":
        binding_bucket = "no_local_object_diffuse"
    elif int(row.get("x1_target_rank") or 10**9) > 50:
        binding_bucket = "target_low_rank"
    else:
        binding_bucket = "other"

    depth = int(row.get("prefix_depth") or 0)
    if depth == 0:
        depth_bucket = "d0"
    elif depth <= 3:
        depth_bucket = "d1_3"
    elif depth <= 7:
        depth_bucket = "d4_7"
    else:
        depth_bucket = "d8_plus"

    gt_count = int(row.get("dataset_gt_count", row.get("gt_count", 0)) or 0)
    if gt_count <= 5:
        object_count_bucket = "gt_1_5"
    elif gt_count <= 15:
        object_count_bucket = "gt_6_15"
    else:
        object_count_bucket = "gt_16_plus"

    return (
        prefix_quality_bucket,
        binding_bucket,
        depth_bucket,
        object_count_bucket,
    )


def select_rescue_cases(
    rows: Iterable[Mapping[str, Any]],
    *,
    sample_limit: int,
    per_stratum_cap: int,
    object_counts_by_case: Mapping[Any, int] | None = None,
) -> list[dict[str, Any]]:
    """Select linked FN-rescue source rows deterministically by stratum."""

    object_counts_by_case = object_counts_by_case or {}
    normalized: list[dict[str, Any]] = []
    for row in rows:
        out = dict(row)
        out["target_gt_idx"] = canonical_target_gt_idx(out)
        case_id = out.get("case_id", out.get("source_line_idx"))
        if out.get("dataset_gt_count") is None and case_id in object_counts_by_case:
            out["dataset_gt_count"] = int(object_counts_by_case[case_id])
        out["fn_rescue_stratum"] = fn_rescue_stratum_key(out)
        normalized.append(out)

    normalized.sort(
        key=lambda item: (
            int(item.get("source_line_idx", 0)),
            int(item.get("prefix_depth", 0)),
            int(item.get("target_gt_idx", 0)),
            str(item.get("case_id", "")),
        )
    )
    selected: list[dict[str, Any]] = []
    counts: dict[tuple[str, str, str, str], int] = defaultdict(int)
    for row in normalized:
        if len(selected) >= sample_limit:
            break
        key = row["fn_rescue_stratum"]
        if counts[key] >= per_stratum_cap:
            continue
        counts[key] += 1
        selected.append(row)
    return selected


def choose_wrong_control_source(
    *,
    target_box: Sequence[Any],
    same_desc_gt_boxes: Sequence[Sequence[Any]],
    same_desc_rollout_boxes: Sequence[Sequence[Any]],
    all_gt_boxes: Sequence[Sequence[Any]],
    context_expansion_norm1000: int,
    duplicate_iou_threshold: float = 0.95,
    wrong_control_overlap_threshold: float = 0.05,
    existing_prediction_boxes: Sequence[Sequence[Any]] | None = None,
) -> WrongControlSource:
    """Pick a deterministic wrong-control x1 source region."""

    for idx, box in enumerate(same_desc_gt_boxes):
        if valid_xyxy_box(box) and box_iou_xyxy(box, target_box) < duplicate_iou_threshold:
            return _wrong_source("same_desc_competitor_gt_object", box, idx)
    for idx, box in enumerate(same_desc_rollout_boxes):
        if valid_xyxy_box(box) and box_iou_xyxy(box, target_box) < duplicate_iou_threshold:
            return _wrong_source("same_desc_rollout_prediction", box, idx)
    return _choose_far_background_source(
        target_box=target_box,
        all_gt_boxes=all_gt_boxes,
        existing_prediction_boxes=existing_prediction_boxes or same_desc_rollout_boxes,
        context_expansion_norm1000=context_expansion_norm1000,
        wrong_control_overlap_threshold=wrong_control_overlap_threshold,
    )


def build_rescue_region_membership(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, list[int]]:
    """Build per-instance and same-kind union visual-token membership."""

    membership: dict[str, list[int]] = {}
    union: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        kind = str(row["region_kind"])
        instance_id = str(row.get("region_instance_id", row.get("source_index", "0")))
        token_indices = sorted({int(value) for value in row.get("token_indices", [])})
        membership[f"instance:{kind}:{instance_id}"] = token_indices
        union[kind].update(token_indices)
    for kind, token_set in union.items():
        membership[f"union:{kind}"] = sorted(token_set)
    return membership


def _split_rescue_attention_region_key(region_key: str) -> dict[str, str]:
    if region_key.startswith("instance:"):
        parts = region_key.split(":", 2)
        if len(parts) == 3 and parts[1] and parts[2]:
            return {
                "aggregation_scope": "instance",
                "region_kind": parts[1],
                "region_instance_id": parts[2],
            }
    if region_key.startswith("union:"):
        parts = region_key.split(":", 1)
        if len(parts) == 2 and parts[1]:
            return {
                "aggregation_scope": "union",
                "region_kind": parts[1],
                "region_instance_id": f"union:{parts[1]}",
            }
    raise ValueError(f"unsupported FN-rescue attention region key: {region_key!r}")


def replay_rescue_attention_rows(
    *,
    model_handle: Any,
    processor_inputs: Mapping[str, Any],
    tier: str,
    stored_assistant_prefix_token_ids: Sequence[int],
    region_rows: Sequence[Mapping[str, Any]],
    attention_tensors: Sequence[Any] | None = None,
) -> RescueAttentionReplayResult:
    """Replay attentions for the rescue continuation query token."""

    tokenizer = _resolve_tokenizer(model_handle)
    replay_ids = _trim_trailing_pad_ids(
        _extract_first_token_row(_require_key(processor_inputs, "input_ids")),
        _resolve_pad_token_id(tokenizer),
    )
    stored_prefix_ids = tuple(int(token_id) for token_id in stored_assistant_prefix_token_ids)
    if not stored_prefix_ids:
        raise ValueError("stored assistant prefix token ids must be nonempty")
    replay_prefix_ids = replay_ids[-len(stored_prefix_ids) :]
    if replay_prefix_ids != stored_prefix_ids:
        raise ValueError("replay assistant prefix token ids do not match stored assistant prefix token ids")

    if attention_tensors is None:
        model = getattr(model_handle, "model", None)
        if model is None or not callable(getattr(model, "__call__", None)):
            raise ValueError("attention replay requires attention_tensors or a callable model_handle.model")
        attn_implementation = _model_attn_implementation(model)
        if attn_implementation is not None and attn_implementation != "eager":
            raise ValueError(
                "attention replay requires eager attention when attention_tensors are not supplied; "
                f"got {attn_implementation!r}"
            )
        outputs = model(
            **dict(processor_inputs),
            output_attentions=True,
            return_dict=True,
            use_cache=False,
        )
        attention_tensors = getattr(outputs, "attentions", None)
        if attention_tensors is None:
            raise ValueError(
                "attention replay expected model outputs to include attentions when output_attentions=True"
            )
    if not isinstance(attention_tensors, Sequence) or isinstance(
        attention_tensors, (str, bytes)
    ):
        raise ValueError("attention replay requires a sequence of attention tensors")

    find_visual_token_spans, build_patch_region_membership, aggregate_attention_for_query = (
        _load_attention_helpers()
    )
    image_token_id = _resolve_image_token_id(model_handle, processor_inputs)
    visual_spans = find_visual_token_spans(replay_ids, image_token_id=image_token_id)
    if len(visual_spans) != 1:
        raise ValueError(
            f"expected exactly 1 visual span in replay input ids; found {len(visual_spans)}"
        )
    visual_start, visual_end = visual_spans[0]
    grid_t, grid_h, grid_w = _extract_image_grid_thw(
        _require_key(processor_inputs, "image_grid_thw")
    )
    expected_visual_tokens, _, visual_grid_h, visual_grid_w = _rescue_visual_grid_from_thw(
        grid_t,
        grid_h,
        grid_w,
        merge_size=_processor_merge_size(model_handle.processor),
    )
    if (visual_end - visual_start) != expected_visual_tokens:
        raise ValueError(
            "visual token span length does not match image_grid_thw after merge_size"
        )

    instance_rows: list[dict[str, Any]] = []
    for index, region_row in enumerate(region_rows):
        membership = build_patch_region_membership(
            visual_token_start=visual_start,
            grid_h=visual_grid_h,
            grid_w=visual_grid_w,
            region_rows=[region_row],
        )
        region_kind = str(region_row["region_kind"])
        instance_rows.append(
            {
                "region_kind": region_kind,
                "region_instance_id": str(region_row.get("region_instance_id", index)),
                "token_indices": membership.get(region_kind, []),
            }
        )
    region_membership = build_rescue_region_membership(instance_rows)
    query_role = "pre_x1" if tier == "desc_only" else "pre_y1"
    query_index = len(replay_ids) - 1
    base_row = {"tier": tier, "rescue_tier": tier}
    rows: list[dict[str, Any]] = []
    for layer_index, attention in enumerate(attention_tensors):
        rows.extend(
            aggregate_attention_for_query(
                attention,
                batch_idx=0,
                query_index=query_index,
                layer_index=layer_index,
                role=query_role,
                region_membership=region_membership,
                base_row=base_row,
            )
        )
    return RescueAttentionReplayResult(
        status="ok",
        rows=tuple(rows),
        query_role=query_role,
        query_index=query_index,
        last_token_text=_decode_single_token_text(tokenizer, replay_ids[-1]),
        replay_prefix_token_ids=replay_prefix_ids,
    )


def _require_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"FN-rescue config section {key!r} must be a mapping")
    return value


def _path_value(payload: Mapping[str, Any], key: str) -> Path:
    value = payload.get(key)
    if value is None:
        raise ValueError(f"FN-rescue config missing paths.{key}")
    return Path(str(value))


def _optional_path_value(payload: Mapping[str, Any], key: str) -> Path | None:
    value = _require_key(payload, key)
    return None if value is None else Path(str(value))


def _require_key(payload: Mapping[str, Any], key: str) -> Any:
    if key not in payload:
        raise ValueError(f"FN-rescue config missing required key: {key}")
    return payload[key]


def _int_value(payload: Mapping[str, Any], key: str) -> int:
    value = _require_key(payload, key)
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{key} must be a YAML integer scalar")
    return int(value)


def _float_value(payload: Mapping[str, Any], key: str) -> float:
    value = _require_key(payload, key)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{key} must be a YAML numeric scalar")
    return float(value)


def _bool_value(payload: Mapping[str, Any], key: str) -> bool:
    value = _require_key(payload, key)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value


def _positive_int(payload: Mapping[str, Any], key: str) -> int:
    value = _int_value(payload, key)
    if value <= 0:
        raise ValueError(f"{key} must be positive")
    return value


def _nonnegative_int(payload: Mapping[str, Any], key: str) -> int:
    value = _int_value(payload, key)
    if value < 0:
        raise ValueError(f"{key} must be nonnegative")
    return value


def _validate_config(
    paths: FnRescuePaths,
    selection: FnRescueSelectionConfig,
    execution: FnRescueExecutionConfig,
) -> None:
    for field_name in _SOURCE_PATH_FIELDS:
        value = getattr(paths, field_name)
        if value is not None and not value.exists():
            raise FileNotFoundError(f"paths.{field_name} does not exist: {value}")
    if not paths.artifact_root.parent.exists():
        raise FileNotFoundError(
            f"paths.artifact_root parent does not exist: {paths.artifact_root.parent}"
        )
    if selection.evidence_scope != EXPECTED_EVIDENCE_SCOPE:
        raise ValueError(
            f"evidence_scope must be {EXPECTED_EVIDENCE_SCOPE!r}; "
            f"got {selection.evidence_scope!r}"
        )
    if not 0.0 <= selection.wrong_control_overlap_threshold <= 1.0:
        raise ValueError("wrong_control_overlap_threshold must be in [0.0, 1.0]")
    if execution.attn_implementation != "eager":
        raise ValueError("attn_implementation must be eager for attention replay")
    if execution.torch_dtype not in _ACCEPTABLE_TORCH_DTYPES:
        raise ValueError(
            "torch_dtype must be one of "
            f"{sorted(_ACCEPTABLE_TORCH_DTYPES)}; got {execution.torch_dtype!r}"
        )
    if execution.decoding != "greedy":
        raise ValueError("decoding must be greedy")
    if execution.do_sample is not False:
        raise ValueError("do_sample must be false")
    if execution.num_beams != 1:
        raise ValueError("num_beams must be 1")
    if execution.max_new_tokens_desc_only != 4:
        raise ValueError("max_new_tokens_desc_only must be 4")
    if execution.max_new_tokens_desc_x1 != 3:
        raise ValueError("max_new_tokens_desc_x1 must be 3")


def _build_source_artifact_summary(
    paths: FnRescuePaths,
) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for field_name in _SOURCE_PATH_FIELDS:
        path = getattr(paths, field_name)
        if path is None:
            summary[field_name] = {"path": None, "exists": False}
            continue
        item = {
            "path": str(path),
            "exists": path.exists(),
            "byte_size": _path_byte_size(path),
            "row_count": _jsonl_row_count(path) if path.suffix == ".jsonl" else None,
        }
        if path.is_file():
            item["hash_kind"] = "file_sha256"
            item["sha256"] = _file_sha256(path)
        elif path.is_dir():
            item["hash_kind"] = "directory_structural_fingerprint"
            item["structural_fingerprint"] = _directory_structural_fingerprint(path)
        else:
            item["hash_kind"] = None
        summary[field_name] = item
    return summary


def _path_byte_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    return 0


def _directory_structural_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(str(item.relative_to(path)).encode("utf-8"))
        digest.update(str(item.stat().st_size).encode("ascii"))
    return digest.hexdigest()


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl_row_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def _leading_coord_tokens(text: str, required: int) -> tuple[str, ...]:
    tokens: list[str] = []
    offset = 0
    for _ in range(required):
        match = _COORD_PREFIX_RE.match(text, offset)
        if match is None:
            break
        tokens.append(match.group(0))
        offset = match.end()
    return tuple(tokens)


def _token_text_from_row(row: Mapping[str, Any], *, key: str) -> str:
    tokens = row.get(key)
    if isinstance(tokens, str):
        return tokens
    if isinstance(tokens, Sequence) and not isinstance(tokens, (bytes, bytearray)):
        return "".join(str(token) for token in tokens)
    raise ValueError(f"{key} must be a string or token sequence")


def _scrub_generate_inputs(processor_inputs: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in dict(processor_inputs).items()
        if str(key) not in _DISALLOWED_GENERATE_RETURN_KWARGS
    }


def _resolve_tokenizer(model_handle: Any) -> Any:
    tokenizer = getattr(model_handle, "tokenizer", None)
    if tokenizer is not None:
        return tokenizer
    processor = getattr(model_handle, "processor", None)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("model_handle must expose tokenizer or processor.tokenizer")
    return tokenizer


def _model_attn_implementation(model: Any) -> str | None:
    config = getattr(model, "config", None)
    if config is None:
        return None
    for owner in (config, getattr(config, "text_config", None)):
        if owner is None:
            continue
        for field_name in ("_attn_implementation", "attn_implementation"):
            value = getattr(owner, field_name, None)
            if value is not None:
                return str(value)
        if isinstance(owner, Mapping):
            for field_name in ("_attn_implementation", "attn_implementation"):
                value = owner.get(field_name)
                if value is not None:
                    return str(value)
    return None


def _encode_text_ids(model_handle: Any, text: str) -> tuple[int, ...]:
    tokenizer = _resolve_tokenizer(model_handle)
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        raise ValueError("tokenizer.encode is required for continuation feasibility checks")
    encoded = encode(str(text), add_special_tokens=False)
    return tuple(int(token_id) for token_id in encoded)


def _extract_first_token_row(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError("token ids must be a sequence or tensor-like object")
    if value and isinstance(value[0], Sequence) and not isinstance(
        value[0], (str, bytes, bytearray)
    ):
        return [int(token_id) for token_id in value[0]]
    return [int(token_id) for token_id in value]


def _resolve_pad_token_id(tokenizer: Any) -> int | None:
    value = getattr(tokenizer, "pad_token_id", None)
    return None if value is None else int(value)


def _resolve_im_end_token_id(tokenizer: Any) -> int:
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if callable(convert):
        token_id = convert("<|im_end|>")
        if token_id is not None:
            return int(token_id)
    value = getattr(tokenizer, "eos_token_id", None)
    if value is None:
        raise ValueError("tokenizer must expose <|im_end|> or eos_token_id")
    return int(value)


def _trim_trailing_pad_ids(
    token_ids: Sequence[int], pad_token_id: int | None
) -> tuple[int, ...]:
    out = [int(token_id) for token_id in token_ids]
    if pad_token_id is None:
        return tuple(out)
    while out and out[-1] == int(pad_token_id):
        out.pop()
    return tuple(out)


def _decode_token_ids(tokenizer: Any, token_ids: Sequence[int]) -> str:
    decode = getattr(tokenizer, "decode", None)
    if callable(decode):
        return str(decode(list(int(token_id) for token_id in token_ids), skip_special_tokens=False))
    convert = getattr(tokenizer, "convert_ids_to_tokens", None)
    if callable(convert):
        return "".join(str(convert(int(token_id))) for token_id in token_ids)
    raise ValueError("tokenizer must expose decode or convert_ids_to_tokens")


def _decode_single_token_text(tokenizer: Any, token_id: int) -> str:
    return _decode_token_ids(tokenizer, [int(token_id)])


def _require_hint_x1(*, tier: str, hint_x1: int | None) -> int:
    if tier == "desc_only":
        if hint_x1 is not None:
            raise ValueError("desc_only must not receive hint_x1")
        return -1
    if hint_x1 is None:
        raise ValueError(f"{tier} requires hint_x1")
    return int(hint_x1)


def _infer_hint_x1(
    *,
    tier: str,
    tokenizer: Any,
    assistant_prefix_token_ids: Sequence[int] | None,
    prompt_ids: Sequence[int],
) -> int | None:
    if tier == "desc_only":
        return None
    source_ids = (
        tuple(int(token_id) for token_id in assistant_prefix_token_ids)
        if assistant_prefix_token_ids is not None
        else _trim_trailing_pad_ids(prompt_ids, _resolve_pad_token_id(tokenizer))
    )
    if not source_ids:
        raise ValueError(f"{tier} requires assistant prefix token ids ending with x1")
    last_token_text = _decode_single_token_text(tokenizer, source_ids[-1])
    if not last_token_text.startswith("<|coord_"):
        raise ValueError(f"{tier} requires the assistant prefix to end with a coord token")
    return token_to_int(last_token_text)


def _load_attention_helpers() -> tuple[Any, Any, Any]:
    module = _load_repo_module_from_path(
        "_coordexp_autoreg_attention_evidence_routing_for_fn_rescue",
        "analysis/autoreg_attention_evidence_routing.py",
    )
    return (
        module.find_visual_token_spans,
        module.build_patch_region_membership,
        module.aggregate_attention_for_query,
    )


def _extract_image_grid_thw(value: Any) -> tuple[int, int, int]:
    row = _extract_first_token_row(value)
    if len(row) != 3:
        raise ValueError("image_grid_thw must contain exactly three integers")
    return int(row[0]), int(row[1]), int(row[2])


def _processor_merge_size(processor: Any) -> int:
    image_processor = getattr(processor, "image_processor", None)
    merge_size = getattr(image_processor, "merge_size", 1)
    try:
        parsed = int(merge_size)
    except (TypeError, ValueError):
        parsed = 1
    return max(1, parsed)


def _rescue_visual_grid_from_thw(
    grid_t: int, grid_h: int, grid_w: int, *, merge_size: int
) -> tuple[int, int, int, int]:
    if int(grid_h) % int(merge_size) != 0 or int(grid_w) % int(merge_size) != 0:
        raise RuntimeError(
            "image_grid_thw spatial dimensions are not divisible by merge_size"
        )
    visual_grid_h = int(grid_h) // int(merge_size)
    visual_grid_w = int(grid_w) // int(merge_size)
    expected_visual_tokens = int(grid_t) * visual_grid_h * visual_grid_w
    return expected_visual_tokens, int(grid_t), visual_grid_h, visual_grid_w


def _resolve_image_token_id(model_handle: Any, processor_inputs: Mapping[str, Any]) -> int:
    processor = getattr(model_handle, "processor", None)
    if processor is not None and getattr(processor, "image_token_id", None) is not None:
        return int(processor.image_token_id)
    if "image_token_id" in processor_inputs:
        return int(processor_inputs["image_token_id"])
    tokenizer = _resolve_tokenizer(model_handle)
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if callable(convert):
        return int(convert("<|image_pad|>"))
    raise ValueError("unable to resolve image_token_id for attention replay")


def _max_duplicate_source(
    generated_box: Sequence[Any] | None, same_desc_rollout_boxes: Sequence[Any]
) -> dict[str, Any]:
    best = {
        "iou": 0.0,
        "kind": None,
        "raw_pred_idx": None,
        "guarded_pred_idx": None,
        "bbox_xyxy": None,
    }
    if generated_box is None or not valid_xyxy_box(generated_box):
        return best

    for idx, source in enumerate(same_desc_rollout_boxes):
        box, metadata = _duplicate_box_and_metadata(source)
        if not valid_xyxy_box(box):
            continue
        iou = box_iou_xyxy(generated_box, box)
        if iou > best["iou"]:
            best = {
                "iou": iou,
                "kind": metadata.get("kind", "same_desc_rollout_prediction"),
                "raw_pred_idx": metadata.get("raw_pred_idx", idx),
                "guarded_pred_idx": metadata.get("guarded_pred_idx"),
                "bbox_xyxy": _box_tuple(box),
            }
    return best


def _duplicate_box_and_metadata(source: Any) -> tuple[Any, dict[str, Any]]:
    if isinstance(source, Mapping):
        box = (
            source.get("bbox_xyxy")
            or source.get("bbox")
            or source.get("bbox_2d")
            or source.get("box")
        )
        return box, dict(source)
    return source, {}


def _wrong_source(
    kind: str, box: Sequence[Any], source_index: int | None
) -> WrongControlSource:
    bbox = _box_tuple(box)
    return WrongControlSource(kind=kind, bbox_xyxy=bbox, x1=bbox[0], source_index=source_index)


def _choose_far_background_source(
    *,
    target_box: Sequence[Any],
    all_gt_boxes: Sequence[Sequence[Any]],
    existing_prediction_boxes: Sequence[Sequence[Any]],
    context_expansion_norm1000: int,
    wrong_control_overlap_threshold: float,
) -> WrongControlSource:
    if not valid_xyxy_box(target_box):
        return _wrong_control_unavailable(())
    target = _box_tuple(target_box)
    width = max(16, target[2] - target[0])
    height = max(16, target[3] - target[1])
    context_ring = _expand_box(target, int(context_expansion_norm1000))
    blockers = [box for box in [*all_gt_boxes, *existing_prediction_boxes] if valid_xyxy_box(box)]
    rejected: list[MappingProxyType[str, Any]] = []

    for label, candidate in _far_background_candidates(width, height):
        reasons = []
        if box_iou_xyxy(candidate, context_ring) > wrong_control_overlap_threshold:
            reasons.append("target_context_ring_overlap")
        for idx, box in enumerate(blockers):
            if box_iou_xyxy(candidate, box) > wrong_control_overlap_threshold:
                reasons.append(f"blocker_{idx}_overlap")
        if reasons:
            rejected.append(_frozen_rejection(label, candidate, reasons))
            continue
        return WrongControlSource(
            kind="far_background_concrete",
            bbox_xyxy=candidate,
            x1=candidate[0],
            source_index=None,
            rejected_candidates=tuple(rejected),
        )

    return _wrong_control_unavailable(tuple(rejected))


def _wrong_control_unavailable(
    rejected_candidates: tuple[MappingProxyType[str, Any], ...],
) -> WrongControlSource:
    return WrongControlSource(
        kind="wrong_control_unavailable",
        bbox_xyxy=None,
        x1=None,
        source_index=None,
        rejected_candidates=rejected_candidates,
        skip_reason="wrong_control_unavailable",
    )


def _expand_box(box: Sequence[int], margin: int) -> tuple[int, int, int, int]:
    return (
        max(0, int(box[0]) - margin),
        max(0, int(box[1]) - margin),
        min(999, int(box[2]) + margin),
        min(999, int(box[3]) + margin),
    )


def _far_background_candidates(
    width: int, height: int
) -> list[tuple[str, tuple[int, int, int, int]]]:
    max_x = 999
    max_y = 999
    mid_y = max(0, (max_y - height) // 2)
    return [
        ("top_left", (0, 0, min(width, max_x), min(height, max_y))),
        ("top_right", (max(0, max_x - width), 0, max_x, min(height, max_y))),
        ("bottom_left", (0, max(0, max_y - height), min(width, max_x), max_y)),
        ("bottom_right", (max(0, max_x - width), max(0, max_y - height), max_x, max_y)),
        ("center_left", (0, mid_y, min(width, max_x), min(max_y, mid_y + height))),
        (
            "center_right",
            (max(0, max_x - width), mid_y, max_x, min(max_y, mid_y + height)),
        ),
    ]


def _box_tuple(box: Sequence[Any]) -> tuple[int, int, int, int]:
    if not valid_xyxy_box(box):
        raise ValueError(f"invalid xyxy box: {box!r}")
    x1, y1, x2, y2 = (int(round(float(value))) for value in box)
    return (x1, y1, x2, y2)


def _frozen_rejection(
    label: str, bbox_xyxy: Sequence[Any], reasons: Sequence[str]
) -> MappingProxyType[str, Any]:
    return MappingProxyType(
        {
            "label": str(label),
            "bbox_xyxy": _box_tuple(bbox_xyxy),
            "reasons": tuple(str(reason) for reason in reasons),
        }
    )


def fn_rescue_shard_label(index: int, num_shards: int) -> str:
    """Render the canonical shard label for FN-rescue outputs."""

    if isinstance(index, bool) or not isinstance(index, Integral):
        raise ValueError("shard index must be an integer")
    if isinstance(num_shards, bool) or not isinstance(num_shards, Integral):
        raise ValueError("num_shards must be an integer")
    shard_index = int(index)
    shard_count = int(num_shards)
    if shard_count <= 0:
        raise ValueError("num_shards must be positive")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(
            f"shard index must be in [0, {shard_count}); got {shard_index}"
        )
    return f"shard_{shard_index:03d}-of-{shard_count:03d}"


def normalize_fn_rescue_shard(
    shard: int | str | Mapping[str, Any], num_shards: int | None = None
) -> dict[str, Any]:
    """Normalize shard inputs into a canonical index/count/label triple."""

    if isinstance(shard, Mapping):
        if shard.get("shard_label") is not None:
            normalized = normalize_fn_rescue_shard(str(shard["shard_label"]))
            if shard.get("shard_index") is not None and int(shard["shard_index"]) != int(
                normalized["shard_index"]
            ):
                raise ValueError("mapping shard_index disagrees with shard_label")
            if shard.get("num_shards") is not None and int(shard["num_shards"]) != int(
                normalized["num_shards"]
            ):
                raise ValueError("mapping num_shards disagrees with shard_label")
            return normalized
        if shard.get("shard_index") is None or shard.get("num_shards") is None:
            raise ValueError(
                "mapping shard input must contain shard_label or both shard_index and num_shards"
            )
        shard_index = int(shard["shard_index"])
        shard_count = int(shard["num_shards"])
        return {
            "shard_index": shard_index,
            "num_shards": shard_count,
            "shard_label": fn_rescue_shard_label(shard_index, shard_count),
        }
    if isinstance(shard, str):
        match = _SHARD_LABEL_RE.match(shard.strip())
        if match is None:
            raise ValueError(f"invalid shard label: {shard!r}")
        shard_index = int(match.group(1))
        shard_count = int(match.group(2))
        return {
            "shard_index": shard_index,
            "num_shards": shard_count,
            "shard_label": fn_rescue_shard_label(shard_index, shard_count),
        }
    if num_shards is None:
        raise ValueError("num_shards is required when shard is numeric")
    shard_index = int(shard)
    shard_count = int(num_shards)
    return {
        "shard_index": shard_index,
        "num_shards": shard_count,
        "shard_label": fn_rescue_shard_label(shard_index, shard_count),
    }


def _fn_rescue_runtime_spec(config: FnRescueConfig) -> dict[str, Any]:
    payload = (
        _maybe_read_json(config.paths.infer_resolved_config)
        if config.paths.infer_resolved_config is not None
        else {}
    ) or {}
    infer_payload = payload.get("infer") if isinstance(payload, Mapping) else None
    if not isinstance(infer_payload, Mapping):
        infer_payload = {}
    root_image_dir_raw = payload.get("root_image_dir") if isinstance(payload, Mapping) else None
    root_image_dir = (
        _resolve_relative_to(config.paths.infer_resolved_config, root_image_dir_raw)
        if root_image_dir_raw is not None and config.paths.infer_resolved_config is not None
        else config.paths.dataset_jsonl.parent
    )
    processor_policy = infer_payload.get("processor_do_resize")
    if processor_policy is None and isinstance(payload, Mapping):
        processor_policy = payload.get("processor_do_resize")
    if processor_policy is None:
        processor_policy = False
    return {
        "prompt_variant": str(infer_payload.get("prompt_variant", "coco_80")),
        "object_field_order": str(infer_payload.get("object_field_order", "desc_first")),
        "bbox_format": str(infer_payload.get("bbox_format", "xyxy")),
        "detection_sequence_format": str(
            infer_payload.get("detection_sequence_format", "compact_full")
        ),
        "object_ordering": str(infer_payload.get("object_ordering", "sorted")),
        "coord_mode": str(infer_payload.get("coord_mode", "coord_tokens")),
        "root_image_dir": root_image_dir,
        "processor_do_resize_or_policy": processor_policy,
    }


def _serialize_fn_rescue_runtime_spec(runtime_spec: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "prompt_variant": str(runtime_spec["prompt_variant"]),
        "object_field_order": str(runtime_spec["object_field_order"]),
        "bbox_format": str(runtime_spec["bbox_format"]),
        "detection_sequence_format": str(runtime_spec["detection_sequence_format"]),
        "object_ordering": str(runtime_spec["object_ordering"]),
        "coord_mode": str(runtime_spec["coord_mode"]),
        "root_image_dir": str(runtime_spec["root_image_dir"]),
        "processor_do_resize_or_policy": runtime_spec["processor_do_resize_or_policy"],
    }


def _runtime_spec_from_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    runtime_spec = manifest.get("runtime_spec")
    if not isinstance(runtime_spec, Mapping):
        raise ValueError("gallery runtime unavailable: missing manifest runtime_spec")
    root_image_dir_raw = runtime_spec.get("root_image_dir")
    if not isinstance(root_image_dir_raw, str) or not root_image_dir_raw.strip():
        raise ValueError("gallery runtime unavailable: missing manifest root_image_dir")
    root_image_dir = Path(root_image_dir_raw).expanduser().resolve(strict=False)
    if not root_image_dir.exists():
        raise ValueError(
            f"gallery runtime unavailable: root_image_dir does not exist: {root_image_dir}"
        )
    return {
        "prompt_variant": str(runtime_spec.get("prompt_variant", "coco_80")),
        "object_field_order": str(runtime_spec.get("object_field_order", "desc_first")),
        "bbox_format": str(runtime_spec.get("bbox_format", "xyxy")),
        "detection_sequence_format": str(
            runtime_spec.get("detection_sequence_format", "compact_full")
        ),
        "object_ordering": str(runtime_spec.get("object_ordering", "sorted")),
        "coord_mode": str(runtime_spec.get("coord_mode", "coord_tokens")),
        "root_image_dir": root_image_dir,
        "processor_do_resize_or_policy": runtime_spec.get(
            "processor_do_resize_or_policy", False
        ),
    }


def _resolve_relative_to(reference_path: Path | None, raw_value: Any) -> Path:
    path = Path(str(raw_value)).expanduser()
    if path.is_absolute():
        return path.resolve(strict=False)
    if reference_path is not None:
        return (reference_path.parent / path).resolve(strict=False)
    return path.resolve(strict=False)


def _resolve_fn_rescue_image_path(
    dataset_row: Mapping[str, Any], *, image_root: Path
) -> Path:
    images = dataset_row.get("images")
    if isinstance(images, Sequence) and not isinstance(images, (str, bytes)) and images:
        image_path = Path(str(images[0])).expanduser()
    elif dataset_row.get("image") is not None:
        image_path = Path(str(dataset_row["image"])).expanduser()
    else:
        raise ValueError("dataset row must contain image or a nonempty images sequence")
    if not image_path.is_absolute():
        image_path = image_root / image_path
    return image_path.resolve(strict=False)


def _build_fn_rescue_messages(
    *,
    runtime_spec: Mapping[str, Any],
    image_path: Path,
    assistant_prefix_text: str,
) -> list[dict[str, Any]]:
    from src.common.detection_chat import build_detection_chat_messages
    from src.config.prompts import get_template_prompts

    system_prompt, user_prompt = get_template_prompts(
        ordering=str(runtime_spec["object_ordering"]),
        coord_mode=str(runtime_spec["coord_mode"]),
        prompt_variant=str(runtime_spec["prompt_variant"]),
        object_field_order=str(runtime_spec["object_field_order"]),
        bbox_format=str(runtime_spec["bbox_format"]),
        detection_sequence_format=str(runtime_spec["detection_sequence_format"]),
    )
    return build_detection_chat_messages(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        images=[str(image_path)],
        assistant_text=assistant_prefix_text,
    )


def _load_fn_rescue_image(path: Path) -> Any:
    from PIL import Image

    with Image.open(path) as image:
        return image.convert("RGB").copy()


def _model_device(model: Any) -> Any:
    try:
        parameters = getattr(model, "parameters", None)
        if callable(parameters):
            first_param = next(parameters())
            return getattr(first_param, "device", None)
    except (StopIteration, TypeError):
        return getattr(model, "device", None)
    return getattr(model, "device", None)


def _default_fn_rescue_processor_input_builder(
    *,
    model_handle: Any,
    assistant_prefix_text: str,
    image_path: Path,
    **_: Any,
) -> Mapping[str, Any]:
    import torch

    from src.common.qwen_generation import call_processor_with_qwen_geometry

    processor = getattr(model_handle, "processor", None)
    if processor is None:
        raise ValueError("model_handle.processor is required")
    prompt_text = processor.apply_chat_template(
        [{"role": message["role"], "content": message["content"]} for message in _["messages"]],
        tokenize=False,
        add_generation_prompt=False,
        continue_final_message=True,
    )
    inputs = call_processor_with_qwen_geometry(
        processor,
        text=[prompt_text],
        images=[_load_fn_rescue_image(image_path)],
        return_tensors="pt",
        padding=False,
    )
    model = getattr(model_handle, "model", None)
    device = _model_device(model)
    if device is None:
        return inputs
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in dict(inputs).items()
    }


def _default_fn_rescue_model_handle_loader(config: FnRescueConfig) -> Any:
    from src.analysis.hard_ce_coord_logit_locality import (
        StudyConfig,
        StudyExecutionConfig,
        StudyModelConfig,
        StudyPaths,
        load_model_handle,
    )

    runtime_spec = _fn_rescue_runtime_spec(config)
    resolved_config_path = (
        config.paths.infer_resolved_config
        or config.config_path
        or config.paths.dataset_jsonl
    )
    study = StudyConfig(
        paths=StudyPaths(
            checkpoint=config.paths.checkpoint,
            resolved_config=resolved_config_path,
            source_config=config.config_path,
            dataset_jsonl=config.paths.dataset_jsonl,
            image_root=Path(runtime_spec["root_image_dir"]),
            artifact_root=config.paths.artifact_root,
            self_rollout_root=None,
            self_rollout_regen_config=None,
        ),
        model=StudyModelConfig(
            prompt_variant=str(runtime_spec["prompt_variant"]),
            object_field_order=str(runtime_spec["object_field_order"]),
            bbox_format=str(runtime_spec["bbox_format"]),
            detection_sequence_format=str(runtime_spec["detection_sequence_format"]),
            object_ordering=str(runtime_spec["object_ordering"]),
            attn_implementation=config.execution.attn_implementation,
            torch_dtype=config.execution.torch_dtype,
        ),
        execution=StudyExecutionConfig(sample_limit=config.selection.sample_limit),
    )
    return load_model_handle(study)


def _git_metadata() -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    git_sha: str | None = None
    git_dirty: bool | None = None
    try:
        git_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            cwd=repo_root,
            text=True,
            capture_output=True,
        ).stdout.strip()
        dirty_stdout = subprocess.run(
            ["git", "status", "--short"],
            check=True,
            cwd=repo_root,
            text=True,
            capture_output=True,
        ).stdout
        git_dirty = bool(dirty_stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        pass
    return {"git_sha": git_sha, "git_dirty": git_dirty}


def _ensure_fn_rescue_manifest(
    config: FnRescueConfig,
    *,
    expected_shards: int,
    attn_implementation_selected: str | None = None,
) -> dict[str, Any]:
    runtime_spec = _fn_rescue_runtime_spec(config)
    manifest_path = config.paths.artifact_root / "shards_manifest.json"
    existing = _maybe_read_json(manifest_path) or {}
    source_artifacts = _source_artifact_summary_for_config(config)
    git_metadata = _git_metadata()
    payload = {
        "analysis_name": "autoreg_fn_rescue_continuation",
        "artifact_schema_version": 1,
        "artifact_root": str(config.paths.artifact_root),
        "config_path": None if config.config_path is None else str(config.config_path),
        "config_sha256": _config_sha256(config),
        "git_sha": git_metadata["git_sha"],
        "git_dirty": git_metadata["git_dirty"],
        "checkpoint": str(config.paths.checkpoint),
        "source_artifacts": source_artifacts,
        "source_selected_cases_sha256": source_artifacts["source_selected_cases"].get("sha256"),
        "source_candidate_regions_sha256": source_artifacts["source_candidate_regions"].get("sha256"),
        "rollout_anatomy_per_row_sha256": source_artifacts["rollout_anatomy_per_row"].get("sha256"),
        "gt_vs_pred_scored_sha256": source_artifacts["gt_vs_pred_scored"].get("sha256"),
        "pred_token_trace_sha256": source_artifacts["pred_token_trace"].get("sha256"),
        "attn_implementation_requested": config.execution.attn_implementation,
        "attn_implementation_selected": (
            attn_implementation_selected
            if attn_implementation_selected is not None
            else existing.get("attn_implementation_selected")
        ),
        "decoding": config.execution.decoding,
        "do_sample": config.execution.do_sample,
        "num_beams": config.execution.num_beams,
        "max_new_tokens_desc_only": config.execution.max_new_tokens_desc_only,
        "max_new_tokens_desc_x1": config.execution.max_new_tokens_desc_x1,
        "gallery_per_bucket": config.execution.gallery_per_bucket,
        "runtime_spec": _serialize_fn_rescue_runtime_spec(runtime_spec),
        "processor_do_resize_or_policy": runtime_spec["processor_do_resize_or_policy"],
        "pre_y1_definition": (
            "state that predicts generated y1 after a forced x1 token in the rescue prefix"
        ),
        "expected_shard_labels": [
            fn_rescue_shard_label(index, expected_shards) for index in range(expected_shards)
        ],
    }
    _write_json_atomic(manifest_path, payload)
    return payload


def _refresh_fn_rescue_shard_summary(
    config: FnRescueConfig,
    *,
    shard_root: Path,
    shard_index: int,
    num_shards: int,
    completed_stage: str,
    auxiliary_row_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    summary_path = shard_root / "summary.json"
    existing = _maybe_read_json(summary_path) or {}
    completed = [
        stage
        for stage in FN_RESCUE_STAGES
        if stage
        in {
            *(existing.get("stages_completed") or []),
            "select_cases",
            completed_stage,
        }
    ]
    row_counts = {
        key: len(read_jsonl(shard_root / filename))
        for key, filename in FN_RESCUE_MERGE_JSONL_FILES.items()
    }
    aux_counts = dict(existing.get("auxiliary_row_counts") or {})
    if auxiliary_row_counts is not None:
        aux_counts.update(
            {
                str(name): int(value)
                for name, value in auxiliary_row_counts.items()
            }
        )
    summary = {
        "analysis_name": "autoreg_fn_rescue_continuation",
        "artifact_schema_version": 1,
        "artifact_root": str(config.paths.artifact_root),
        "checkpoint": str(config.paths.checkpoint),
        "evidence_scope": config.selection.evidence_scope,
        "config_sha256": _config_sha256(config),
        "shard_index": int(shard_index),
        "num_shards": int(num_shards),
        "shard_label": fn_rescue_shard_label(shard_index, num_shards),
        "source_artifacts": _source_artifact_summary_for_config(config),
        "row_counts": row_counts,
        "stages_completed": completed,
    }
    if aux_counts:
        summary["auxiliary_row_counts"] = aux_counts
    _write_json_atomic(summary_path, summary)
    return summary


def _emitted_fn_rescue_stage_items(
    config: FnRescueConfig,
    *,
    shard_root: Path,
) -> list[dict[str, Any]]:
    runtime_spec = _fn_rescue_runtime_spec(config)
    selected_rows = read_jsonl(shard_root / FN_RESCUE_MERGE_JSONL_FILES["selected_rescue_cases"])
    rescue_rows = read_jsonl(shard_root / FN_RESCUE_MERGE_JSONL_FILES["rescue_rows"])
    candidate_rows = read_jsonl(
        shard_root / FN_RESCUE_MERGE_JSONL_FILES["rescue_candidate_region_rows"]
    )
    wrong_control_rows = read_jsonl(
        shard_root / FN_RESCUE_MERGE_JSONL_FILES["wrong_control_rows"]
    )
    dataset_rows = read_jsonl(config.paths.dataset_jsonl)
    scored_rows = read_jsonl(config.paths.gt_vs_pred_scored)
    rollout_rows = read_jsonl(config.paths.rollout_anatomy_per_row)
    selected_by_case = {
        str(row.get("case_id", row.get("source_line_idx"))): row for row in selected_rows
    }
    candidate_by_case_tier: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        candidate_by_case_tier[(str(row["case_id"]), str(row["planned_rescue_tier"]))].append(
            row
        )
    wrong_control_by_case = {str(row["case_id"]): row for row in wrong_control_rows}
    rollout_by_source_line = _group_rows_by_int_key(rollout_rows, "source_line_idx")
    items: list[dict[str, Any]] = []
    for rescue_row in rescue_rows:
        if not bool(rescue_row.get("emitted")):
            continue
        case_id = str(rescue_row["case_id"])
        selected_row = selected_by_case[case_id]
        source_line_idx = int(rescue_row["source_line_idx"])
        dataset_row = _require_line_index_row(dataset_rows, source_line_idx, "dataset_jsonl")
        scored_row = _require_line_index_row(
            scored_rows, source_line_idx, "gt_vs_pred_scored"
        )
        rollout_case_rows = list(rollout_by_source_line.get(source_line_idx, ()))
        tier = str(rescue_row["planned_rescue_tier"])
        hint_x1_raw = rescue_row.get("hint_x1")
        hint_x1 = None if hint_x1_raw is None else int(hint_x1_raw)
        assistant_prefix_text = build_rescue_assistant_prefix(
            prefix_text=str(selected_row.get("prefix_text") or ""),
            target_desc=str(rescue_row["target_desc"]),
            tier=tier,
            hint_x1=hint_x1,
        )
        image_path = _resolve_fn_rescue_image_path(
            dataset_row, image_root=Path(runtime_spec["root_image_dir"])
        )
        items.append(
            {
                "case_id": case_id,
                "selected_row": selected_row,
                "rescue_row": rescue_row,
                "dataset_row": dataset_row,
                "scored_row": scored_row,
                "rollout_case_rows": rollout_case_rows,
                "candidate_rows": list(candidate_by_case_tier.get((case_id, tier), ())),
                "wrong_control_row": wrong_control_by_case.get(case_id),
                "assistant_prefix_text": assistant_prefix_text,
                "image_path": image_path,
                "messages": _build_fn_rescue_messages(
                    runtime_spec=runtime_spec,
                    image_path=image_path,
                    assistant_prefix_text=assistant_prefix_text,
                ),
            }
        )
    return items


def _prefix_objects_from_rollout_rows(
    rollout_case_rows: Sequence[Mapping[str, Any]], *, prefix_depth: int
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in rollout_case_rows:
        raw_pred_idx = row.get("raw_pred_idx")
        if raw_pred_idx is None or bool(row.get("suppressed_by_guard", False)):
            continue
        if int(raw_pred_idx) >= int(prefix_depth):
            continue
        rows.append(
            {
                "raw_pred_idx": int(raw_pred_idx),
                "guarded_pred_idx": row.get("guarded_pred_idx"),
                "pred_desc": row.get("pred_desc"),
                "pred_points": list(_box_tuple(row["pred_points"]))
                if valid_xyxy_box(row.get("pred_points", ()))
                else None,
            }
        )
    return rows


def _generation_kwargs_artifact_summary(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key, value in kwargs.items():
        if key in {
            "do_sample",
            "num_beams",
            "max_new_tokens",
            "pad_token_id",
            "eos_token_id",
        }:
            summary[key] = value
            continue
        shape = getattr(value, "shape", None)
        if shape is not None:
            summary[f"{key}_shape"] = [int(dim) for dim in shape]
    return summary


def _run_fn_rescue_attention_probe(
    *,
    model_handle: Any,
    processor_inputs: Mapping[str, Any],
    tier: str,
    stored_assistant_prefix_token_ids: Sequence[int],
    region_rows: Sequence[Mapping[str, Any]],
    prompt_input_ids: Sequence[int] | None = None,
) -> tuple[dict[str, Any], RescueAttentionReplayResult]:
    tokenizer = _resolve_tokenizer(model_handle)
    prompt_ids = _trim_trailing_pad_ids(
        _extract_first_token_row(_require_key(processor_inputs, "input_ids")),
        _resolve_pad_token_id(tokenizer),
    )
    stored_prefix_ids = tuple(int(token_id) for token_id in stored_assistant_prefix_token_ids)
    replay_prefix_ids = prompt_ids[-len(stored_prefix_ids) :] if stored_prefix_ids else ()
    prompt_input_prefix_ids = (
        ()
        if prompt_input_ids is None
        else tuple(int(token_id) for token_id in prompt_input_ids)[
            -len(stored_prefix_ids) :
        ]
    )
    prompt_input_ids_aligned = (
        prompt_input_ids is None
        or prompt_input_prefix_ids == stored_prefix_ids
    )
    if not prompt_input_ids_aligned:
        raise ValueError(
            "feasibility prompt input ids do not end with stored assistant prefix token ids"
        )

    model = getattr(model_handle, "model", None)
    if model is None or not callable(getattr(model, "__call__", None)):
        raise ValueError("attention feasibility probe requires a callable model_handle.model")
    attn_implementation = _model_attn_implementation(model)
    if attn_implementation is not None and attn_implementation != "eager":
        raise ValueError(
            "attention feasibility probe requires eager attention; "
            f"got {attn_implementation!r}"
        )
    outputs = model(
        **dict(processor_inputs),
        output_attentions=True,
        return_dict=True,
        use_cache=False,
    )
    attention_tensors = getattr(outputs, "attentions", None)
    if attention_tensors is None:
        raise ValueError(
            "attention feasibility probe expected model outputs to include attentions"
        )
    if not isinstance(attention_tensors, Sequence) or isinstance(
        attention_tensors, (str, bytes)
    ):
        raise ValueError("attention feasibility probe requires a sequence of attention tensors")

    find_visual_token_spans, _, _ = _load_attention_helpers()
    image_token_id = _resolve_image_token_id(model_handle, processor_inputs)
    visual_spans = find_visual_token_spans(prompt_ids, image_token_id=image_token_id)
    if len(visual_spans) != 1:
        raise ValueError(
            f"expected exactly 1 visual span in feasibility input ids; found {len(visual_spans)}"
        )
    visual_start, visual_end = visual_spans[0]
    grid_t, grid_h, grid_w = _extract_image_grid_thw(
        _require_key(processor_inputs, "image_grid_thw")
    )
    expected_visual_tokens, _, visual_grid_h, visual_grid_w = _rescue_visual_grid_from_thw(
        grid_t,
        grid_h,
        grid_w,
        merge_size=_processor_merge_size(model_handle.processor),
    )
    if (visual_end - visual_start) != expected_visual_tokens:
        raise ValueError(
            "visual token span length does not match image_grid_thw after merge_size"
        )
    replay_result = replay_rescue_attention_rows(
        model_handle=model_handle,
        processor_inputs=processor_inputs,
        tier=tier,
        stored_assistant_prefix_token_ids=stored_prefix_ids,
        region_rows=region_rows,
        attention_tensors=attention_tensors,
    )
    attention_shape_0 = None
    first_attention = attention_tensors[0] if attention_tensors else None
    if first_attention is not None and hasattr(first_attention, "shape"):
        shape = getattr(first_attention, "shape")
        if len(shape) > 0:
            attention_shape_0 = int(shape[0])
    probe = {
        "replay_prefix_token_ids": list(replay_prefix_ids),
        "replay_prefix_token_match": bool(replay_prefix_ids == stored_prefix_ids),
        "prompt_input_ids_aligned": bool(prompt_input_ids_aligned),
        "visual_span_start": int(visual_start),
        "visual_span_end": int(visual_end),
        "visual_grid_h": int(visual_grid_h),
        "visual_grid_w": int(visual_grid_w),
        "attention_layer_count": len(attention_tensors),
        "attention_shape_0": attention_shape_0,
        "attn_implementation_selected": attn_implementation,
        "query_role": replay_result.query_role,
    }
    return probe, replay_result


def build_fn_rescue_dry_run_plan(
    config: FnRescueConfig,
    stages: str | Sequence[str],
    shard_index: int | str | Mapping[str, Any],
    num_shards: int | None = None,
) -> dict[str, Any]:
    """Build a deterministic dry-run payload for orchestration tooling."""

    stage_list = _normalize_fn_rescue_stages(stages)
    shard_info = normalize_fn_rescue_shard(shard_index, num_shards)
    expected_labels = [
        fn_rescue_shard_label(index, int(shard_info["num_shards"]))
        for index in range(int(shard_info["num_shards"]))
    ]
    return {
        "artifact_root": str(config.paths.artifact_root),
        "checkpoint": str(config.paths.checkpoint),
        "evidence_scope": config.selection.evidence_scope,
        "stages": stage_list,
        "available_stages": list(FN_RESCUE_STAGES),
        "shard": shard_info,
        "expected_shard_labels": expected_labels,
        "decode_settings": {
            "attn_implementation": config.execution.attn_implementation,
            "torch_dtype": config.execution.torch_dtype,
            "decoding": config.execution.decoding,
            "do_sample": config.execution.do_sample,
            "num_beams": config.execution.num_beams,
            "max_new_tokens_desc_only": config.execution.max_new_tokens_desc_only,
            "max_new_tokens_desc_x1": config.execution.max_new_tokens_desc_x1,
        },
        "source_artifacts": _source_artifact_summary_for_config(config),
        "config_sha256": _config_sha256(config),
    }


def materialize_fn_rescue_select_cases_shard(
    config: FnRescueConfig, shard_index: int, num_shards: int
) -> dict[str, Any]:
    """Select and materialize a shard-local FN-rescue denominator ledger."""

    shard_info = normalize_fn_rescue_shard(shard_index, num_shards)
    shard_label = str(shard_info["shard_label"])
    shard_root = config.paths.artifact_root / "shards" / shard_label
    shard_root.mkdir(parents=True, exist_ok=True)
    config_sha256 = _config_sha256(config)
    source_artifacts = _source_artifact_summary_for_config(config)

    source_selected_rows = read_jsonl(config.paths.source_selected_cases)
    rollout_rows = read_jsonl(config.paths.rollout_anatomy_per_row)
    rollout_by_source_line = _group_rows_by_int_key(rollout_rows, "source_line_idx")
    object_counts_by_case = _object_counts_by_case(source_selected_rows, rollout_by_source_line)
    selected_cases = select_rescue_cases(
        _augment_selected_rows_with_object_counts(
            source_selected_rows, rollout_by_source_line
        ),
        sample_limit=config.selection.sample_limit,
        per_stratum_cap=config.selection.per_stratum_cap,
        object_counts_by_case=object_counts_by_case,
    )
    shard_selected = [
        dict(row)
        for index, row in enumerate(selected_cases)
        if index % int(shard_info["num_shards"]) == int(shard_info["shard_index"])
    ]

    candidate_rows = read_jsonl(config.paths.source_candidate_regions)
    candidate_rows_by_case = _group_rows_by_key(candidate_rows, "case_id")
    scored_rows = read_jsonl(config.paths.gt_vs_pred_scored)
    token_trace_rows = _read_jsonl_indexed(
        config.paths.pred_token_trace, index_key="line_idx"
    )

    selected_outputs: list[dict[str, Any]] = []
    rescue_rows: list[dict[str, Any]] = []
    candidate_outputs: list[dict[str, Any]] = []
    wrong_control_rows: list[dict[str, Any]] = []

    for case_row in shard_selected:
        case_id = str(case_row.get("case_id", case_row.get("source_line_idx")))
        source_line_idx = int(case_row["source_line_idx"])
        target_gt_idx = canonical_target_gt_idx(case_row)
        target_desc = str(case_row["target_desc"])
        prefix_depth = int(case_row.get("prefix_depth", 0) or 0)
        stratum = tuple(str(value) for value in case_row["fn_rescue_stratum"])
        binding_bucket = stratum[1]
        depth_bucket = stratum[2]
        object_count_bucket = stratum[3]

        case_candidate_rows = list(candidate_rows_by_case.get(case_id, ()))
        _validate_candidate_join_keys(case_row, case_candidate_rows)
        target_region = _require_single_region(case_candidate_rows, "target_gt")
        target_box = _box_tuple(target_region["bbox_xyxy"])
        rollout_case_rows = list(rollout_by_source_line.get(source_line_idx, ()))
        scored_row = _require_line_index_row(
            scored_rows, source_line_idx, "gt_vs_pred_scored"
        )
        token_trace_row = token_trace_rows.get(source_line_idx)

        prefix_text: str | None
        prefix_reconstruction_error: str | None = None
        try:
            prefix_text = prefix_text_from_raw_compact_predictions(
                rollout_rows=rollout_case_rows,
                scored_row=scored_row,
                token_trace_row=token_trace_row,
                prefix_depth=prefix_depth,
            )
            prefix_reconstruction_status = "empty_prefix" if prefix_text == "" else "ok"
        except Exception as exc:  # pragma: no cover
            prefix_text = None
            prefix_reconstruction_status = "prefix_reconstruction_failed"
            prefix_reconstruction_error = f"{type(exc).__name__}: {exc}"

        same_desc_gt_boxes = [
            candidate["bbox_xyxy"]
            for candidate in case_candidate_rows
            if candidate.get("region_kind") == "same_desc_gt"
        ]
        same_desc_rollout_rows = _same_desc_rollout_rows(
            rollout_case_rows, target_desc=target_desc
        )
        same_desc_rollout_boxes = [
            row["pred_points"]
            for row in same_desc_rollout_rows
            if valid_xyxy_box(row.get("pred_points", ()))
        ]
        wrong_control_source = choose_wrong_control_source(
            target_box=target_box,
            same_desc_gt_boxes=same_desc_gt_boxes,
            same_desc_rollout_boxes=same_desc_rollout_boxes,
            all_gt_boxes=_all_gt_boxes_from_scored_row(scored_row),
            context_expansion_norm1000=config.selection.context_expansion_norm1000,
            duplicate_iou_threshold=config.selection.duplicate_iou_threshold,
            wrong_control_overlap_threshold=config.selection.wrong_control_overlap_threshold,
            existing_prediction_boxes=_all_prediction_boxes_from_scored_row(scored_row),
        )

        selected_outputs.append(
            {
                **dict(case_row),
                "case_id": case_id,
                "target_gt_idx": target_gt_idx,
                "binding_bucket": binding_bucket,
                "depth_bucket": depth_bucket,
                "object_count_bucket": object_count_bucket,
                "target_bbox_xyxy": list(target_box),
                "prefix_text": prefix_text,
                "prefix_reconstruction_status": prefix_reconstruction_status,
                "prefix_reconstruction_error": prefix_reconstruction_error,
                "config_sha256": config_sha256,
                "source_artifacts": source_artifacts,
                "shard_index": int(shard_info["shard_index"]),
                "num_shards": int(shard_info["num_shards"]),
                "shard_label": shard_label,
            }
        )

        base_candidate_rows = _build_selection_candidate_rows(
            case_row=case_row,
            case_candidate_rows=case_candidate_rows,
            rollout_case_rows=rollout_case_rows,
            target_desc=target_desc,
            prefix_depth=prefix_depth,
            shard_label=shard_label,
        )

        for tier in ("desc_only", "desc_x1", "desc_x1_wrong_control"):
            emitted = prefix_reconstruction_status != "prefix_reconstruction_failed"
            skip_reason: str | None = None
            wrong_control_status: str | None = None
            if tier == "desc_only":
                hint_x1_raw = None
            elif tier == "desc_x1":
                hint_x1_raw = target_box[0]
            else:
                hint_x1_raw = wrong_control_source.x1
                wrong_control_status = str(
                    wrong_control_source.skip_reason or wrong_control_source.kind
                )
                if wrong_control_source.skip_reason is not None:
                    emitted = False
                    skip_reason = str(wrong_control_source.skip_reason)
            hint_x1_fields = _hint_x1_record(
                None if hint_x1_raw is None else int(hint_x1_raw)
            )
            if prefix_reconstruction_status == "prefix_reconstruction_failed":
                emitted = False
                skip_reason = "prefix_reconstruction_failed"
            rescue_rows.append(
                {
                    "case_id": case_id,
                    "source_line_idx": source_line_idx,
                    "target_gt_idx": target_gt_idx,
                    "target_desc": target_desc,
                    "prefix_depth": prefix_depth,
                    "prefix_quality": str(case_row["prefix_quality"]),
                    "binding_bucket": binding_bucket,
                    "depth_bucket": depth_bucket,
                    "object_count_bucket": object_count_bucket,
                    "planned_rescue_tier": tier,
                    "attempt_status": "emitted" if emitted else "skipped",
                    "emitted": emitted,
                    "skip_reason": skip_reason,
                    **hint_x1_fields,
                    "prefix_reconstruction_status": prefix_reconstruction_status,
                    "prefix_reconstruction_error": prefix_reconstruction_error,
                    "wrong_control_status": wrong_control_status,
                    "source_artifacts": source_artifacts,
                    "config_sha256": config_sha256,
                    "shard_label": shard_label,
                }
            )
            candidate_outputs.extend(
                _candidate_rows_for_tier(
                    base_candidate_rows=base_candidate_rows,
                    case_row=case_row,
                    target_gt_idx=target_gt_idx,
                    target_desc=target_desc,
                    tier=tier,
                    wrong_control_source=wrong_control_source if emitted else None,
                )
            )
            if emitted and tier == "desc_x1_wrong_control":
                wrong_control_rows.append(
                    {
                        "case_id": case_id,
                        "source_line_idx": source_line_idx,
                        "target_gt_idx": target_gt_idx,
                        "target_desc": target_desc,
                        "rescue_tier": tier,
                        "wrong_control_source_kind": wrong_control_source.kind,
                        "wrong_control_source_bbox_xyxy": list(
                            wrong_control_source.bbox_xyxy or ()
                        )
                        if wrong_control_source.bbox_xyxy is not None
                        else None,
                        "wrong_control_source_x1": wrong_control_source.x1,
                        "source_index": wrong_control_source.source_index,
                        "rejected_candidates": [
                            dict(item) for item in wrong_control_source.rejected_candidates
                        ],
                        "source_artifacts": source_artifacts,
                        "config_sha256": config_sha256,
                        "shard_label": shard_label,
                    }
                )

    file_rows = {
        "selected_rescue_cases": selected_outputs,
        "rescue_rows": rescue_rows,
        "rescue_generation_rows": [],
        "rescue_replay_prefix_rows": [],
        "rescue_attention_region_rows": [],
        "rescue_decision_context_rows": [],
        "rescue_candidate_region_rows": candidate_outputs,
        "wrong_control_rows": wrong_control_rows,
    }
    for key, rows in file_rows.items():
        _write_jsonl_atomic(shard_root / FN_RESCUE_MERGE_JSONL_FILES[key], rows)
    _ensure_fn_rescue_manifest(
        config, expected_shards=int(shard_info["num_shards"])
    )
    summary = _refresh_fn_rescue_shard_summary(
        config,
        shard_root=shard_root,
        shard_index=int(shard_info["shard_index"]),
        num_shards=int(shard_info["num_shards"]),
        completed_stage="select_cases",
    )
    return {
        "artifact_root": str(config.paths.artifact_root),
        "shard_root": str(shard_root),
        "shard_label": shard_label,
        "config_sha256": config_sha256,
        "row_counts": summary["row_counts"],
    }


def _require_fn_rescue_select_cases_ready(
    config: FnRescueConfig,
    *,
    stage_name: str,
    shard_index: int,
    num_shards: int,
) -> tuple[Path, str, dict[str, Any]]:
    shard_info = normalize_fn_rescue_shard(shard_index, num_shards)
    shard_label = str(shard_info["shard_label"])
    shard_root = config.paths.artifact_root / "shards" / shard_label
    required_paths = [
        shard_root / "summary.json",
        shard_root / FN_RESCUE_MERGE_JSONL_FILES["selected_rescue_cases"],
        shard_root / FN_RESCUE_MERGE_JSONL_FILES["rescue_rows"],
        shard_root / FN_RESCUE_MERGE_JSONL_FILES["rescue_candidate_region_rows"],
        shard_root / FN_RESCUE_MERGE_JSONL_FILES["wrong_control_rows"],
    ]
    missing = [path for path in required_paths if not path.exists()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise RuntimeError(
            f"FN-rescue {stage_name} requires select_cases shard outputs for {shard_label}; "
            f"run select_cases first. Missing: {missing_text}"
        )
    summary = json.loads((shard_root / "summary.json").read_text(encoding="utf-8"))
    row_counts = summary.get("row_counts")
    if not isinstance(row_counts, Mapping):
        raise RuntimeError(
            f"FN-rescue {stage_name} requires row_counts in {shard_root / 'summary.json'}"
        )
    return shard_root, shard_label, dict(summary)


def materialize_fn_rescue_feasibility_shard(
    config: FnRescueConfig,
    shard_index: int,
    num_shards: int,
    *,
    model_handle_loader: Any | None = None,
    processor_input_builder: Any | None = None,
) -> dict[str, Any]:
    shard_root, shard_label, _summary = _require_fn_rescue_select_cases_ready(
        config,
        stage_name="feasibility",
        shard_index=shard_index,
        num_shards=num_shards,
    )
    loader = model_handle_loader or _default_fn_rescue_model_handle_loader
    build_inputs = processor_input_builder or _default_fn_rescue_processor_input_builder
    model_handle = loader(config)
    selected_attn = _model_attn_implementation(getattr(model_handle, "model", None))
    _ensure_fn_rescue_manifest(
        config,
        expected_shards=num_shards,
        attn_implementation_selected=selected_attn,
    )
    tokenizer = _resolve_tokenizer(model_handle)
    rows: list[dict[str, Any]] = []
    for item in _emitted_fn_rescue_stage_items(config, shard_root=shard_root):
        rescue_row = item["rescue_row"]
        selected_row = item["selected_row"]
        tier = str(rescue_row["planned_rescue_tier"])
        hint_x1_raw = rescue_row.get("hint_x1")
        hint_x1 = None if hint_x1_raw is None else int(hint_x1_raw)
        assistant_prefix_text = str(item["assistant_prefix_text"])
        assistant_prefix_token_ids = tuple(
            int(token_id)
            for token_id in tokenizer.encode(
                assistant_prefix_text,
                add_special_tokens=False,
            )
        )
        feasibility = check_chat_template_continuation_feasibility(
            model_handle,
            item["messages"],
            assistant_prefix_text,
            tier=tier,
            hint_x1=hint_x1,
        )
        processor_inputs = build_inputs(
            model_handle=model_handle,
            config=config,
            stage_name="feasibility",
            selected_row=selected_row,
            rescue_row=rescue_row,
            dataset_row=item["dataset_row"],
            scored_row=item["scored_row"],
            rollout_case_rows=item["rollout_case_rows"],
            candidate_rows=item["candidate_rows"],
            messages=item["messages"],
            assistant_prefix_text=assistant_prefix_text,
            image_path=item["image_path"],
        )
        probe, replay_result = _run_fn_rescue_attention_probe(
            model_handle=model_handle,
            processor_inputs=processor_inputs,
            tier=tier,
            stored_assistant_prefix_token_ids=assistant_prefix_token_ids,
            region_rows=item["candidate_rows"],
            prompt_input_ids=feasibility.input_ids,
        )
        runtime_spec = _fn_rescue_runtime_spec(config)
        rows.append(
            {
                "case_id": str(item["case_id"]),
                "source_line_idx": int(rescue_row["source_line_idx"]),
                "target_gt_idx": canonical_target_gt_idx(selected_row),
                "target_desc": str(rescue_row["target_desc"]),
                "rescue_tier": tier,
                "prefix_quality": str(selected_row["prefix_quality"]),
                "binding_bucket": str(selected_row["binding_bucket"]),
                "depth_bucket": str(selected_row["depth_bucket"]),
                "object_count_bucket": str(selected_row["object_count_bucket"]),
                "hint_x1": hint_x1,
                "hint_x1_raw": rescue_row.get("hint_x1_raw", hint_x1),
                "hint_x1_used": rescue_row.get("hint_x1_used", hint_x1),
                "hint_x1_clamped": bool(rescue_row.get("hint_x1_clamped", False)),
                "assistant_prefix_text": assistant_prefix_text,
                "assistant_prefix_token_ids": list(assistant_prefix_token_ids),
                "rendered_prompt": feasibility.rendered_prompt,
                "prompt_input_ids": list(feasibility.input_ids),
                "feasible": bool(feasibility.feasible and probe["replay_prefix_token_match"]),
                "status": str(feasibility.status),
                "errors": list(feasibility.errors),
                "last_token_text": feasibility.last_token_text,
                "image_path": str(item["image_path"]),
                "replay_prefix_token_match": probe["replay_prefix_token_match"],
                "prompt_input_ids_aligned": probe["prompt_input_ids_aligned"],
                "visual_span_start": probe["visual_span_start"],
                "visual_span_end": probe["visual_span_end"],
                "visual_grid_h": probe["visual_grid_h"],
                "visual_grid_w": probe["visual_grid_w"],
                "attention_layer_count": probe["attention_layer_count"],
                "attention_shape_0": probe["attention_shape_0"],
                "attn_implementation_selected": probe["attn_implementation_selected"],
                "processor_do_resize_or_policy": runtime_spec["processor_do_resize_or_policy"],
                "query_role": probe["query_role"],
                "replay_status": replay_result.status,
                "source_artifacts": _source_artifact_summary_for_config(config),
                "config_sha256": _config_sha256(config),
                "shard_label": shard_label,
            }
        )
    _write_jsonl_atomic(shard_root / "feasibility_rows.jsonl", rows)
    summary = _refresh_fn_rescue_shard_summary(
        config,
        shard_root=shard_root,
        shard_index=shard_index,
        num_shards=num_shards,
        completed_stage="feasibility",
        auxiliary_row_counts={"feasibility_rows": len(rows)},
    )
    return {
        "artifact_root": str(config.paths.artifact_root),
        "shard_root": str(shard_root),
        "shard_label": shard_label,
        "row_counts": {**summary["row_counts"], "feasibility_rows": len(rows)},
    }


def materialize_fn_rescue_decode_shard(
    config: FnRescueConfig,
    shard_index: int,
    num_shards: int,
    *,
    model_handle_loader: Any | None = None,
    processor_input_builder: Any | None = None,
) -> dict[str, Any]:
    shard_root, shard_label, _summary = _require_fn_rescue_select_cases_ready(
        config,
        stage_name="rescue_decode",
        shard_index=shard_index,
        num_shards=num_shards,
    )
    loader = model_handle_loader or _default_fn_rescue_model_handle_loader
    build_inputs = processor_input_builder or _default_fn_rescue_processor_input_builder
    model_handle = loader(config)
    selected_attn = _model_attn_implementation(getattr(model_handle, "model", None))
    _ensure_fn_rescue_manifest(
        config,
        expected_shards=num_shards,
        attn_implementation_selected=selected_attn,
    )
    tokenizer = _resolve_tokenizer(model_handle)
    source_artifacts = _source_artifact_summary_for_config(config)
    config_sha256 = _config_sha256(config)
    generation_rows: list[dict[str, Any]] = []
    for item in _emitted_fn_rescue_stage_items(config, shard_root=shard_root):
        rescue_row = item["rescue_row"]
        selected_row = item["selected_row"]
        tier = str(rescue_row["planned_rescue_tier"])
        hint_x1_raw = rescue_row.get("hint_x1")
        hint_x1 = None if hint_x1_raw is None else int(hint_x1_raw)
        assistant_prefix_text = str(item["assistant_prefix_text"])
        assistant_prefix_token_ids = tuple(
            int(token_id)
            for token_id in tokenizer.encode(
                assistant_prefix_text,
                add_special_tokens=False,
            )
        )
        feasibility = check_chat_template_continuation_feasibility(
            model_handle,
            item["messages"],
            assistant_prefix_text,
            tier=tier,
            hint_x1=hint_x1,
        )
        if not feasibility.feasible:
            raise RuntimeError(
                f"FN-rescue rescue_decode feasibility failed for case {item['case_id']} "
                f"tier {tier}: {list(feasibility.errors)!r}"
            )
        processor_inputs = build_inputs(
            model_handle=model_handle,
            config=config,
            stage_name="rescue_decode",
            selected_row=selected_row,
            rescue_row=rescue_row,
            dataset_row=item["dataset_row"],
            scored_row=item["scored_row"],
            rollout_case_rows=item["rollout_case_rows"],
            candidate_rows=item["candidate_rows"],
            messages=item["messages"],
            assistant_prefix_text=assistant_prefix_text,
            image_path=item["image_path"],
        )
        decode_result = decode_rescue_tail(
            model_handle=model_handle,
            processor_inputs=processor_inputs,
            tier=tier,
            assistant_prefix_token_ids=assistant_prefix_token_ids,
        )
        target_region = _require_single_region(item["candidate_rows"], "target_gt")
        target_box = _box_tuple(target_region["bbox_xyxy"])
        same_desc_rollout_boxes = [
            row["pred_points"]
            for row in _same_desc_rollout_rows(
                item["rollout_case_rows"], target_desc=str(rescue_row["target_desc"])
            )
            if valid_xyxy_box(row.get("pred_points", ()))
        ]
        score = score_rescue_box(
            generated_box=decode_result.parsed_generation.generated_box_xyxy,
            target_box=target_box,
            same_desc_rollout_boxes=same_desc_rollout_boxes,
            duplicate_iou_threshold=config.selection.duplicate_iou_threshold,
        )
        target_desc_preserved = (
            OBJECT_REF_START_TOKEN not in decode_result.generated_tail_text
            and str(rescue_row["target_desc"]) in assistant_prefix_text
        )
        geometry_valid = bool(score.valid_parse)
        if not target_desc_preserved:
            outcome_bucket = "target_desc_not_preserved"
        elif not geometry_valid:
            outcome_bucket = "invalid_parse"
        elif bool(score.same_desc_duplicate_iou95):
            outcome_bucket = "duplicate_rejected"
        elif bool(score.primary_rescue_success):
            outcome_bucket = "primary_rescue_success"
        else:
            outcome_bucket = "target_iou_below_0p50"
        eos_token_id = decode_result.generation_kwargs.get("eos_token_id")
        stop_reason = (
            "eos_token"
            if decode_result.generated_tail_ids
            and eos_token_id is not None
            and int(decode_result.generated_tail_ids[-1]) == int(eos_token_id)
            else "max_new_tokens"
        )
        generation_rows.append(
            {
                "case_id": str(item["case_id"]),
                "source_line_idx": int(rescue_row["source_line_idx"]),
                "target_gt_idx": canonical_target_gt_idx(selected_row),
                "target_desc": str(rescue_row["target_desc"]),
                "rescue_tier": tier,
                "prefix_quality": str(selected_row["prefix_quality"]),
                "binding_bucket": str(selected_row["binding_bucket"]),
                "depth_bucket": str(selected_row["depth_bucket"]),
                "object_count_bucket": str(selected_row["object_count_bucket"]),
                "assistant_prefix_text": assistant_prefix_text,
                "assistant_prefix_token_ids": list(assistant_prefix_token_ids),
                "generated_token_ids": list(decode_result.generated_tail_ids),
                "full_generated_ids": list(decode_result.full_generated_ids),
                "generated_tail_text": decode_result.generated_tail_text,
                "stop_reason": stop_reason,
                "hint_x1": decode_result.hint_x1,
                "hint_x1_raw": rescue_row.get("hint_x1_raw", decode_result.hint_x1),
                "hint_x1_used": rescue_row.get("hint_x1_used", decode_result.hint_x1),
                "hint_x1_clamped": bool(rescue_row.get("hint_x1_clamped", False)),
                "parse_status": (
                    "ok"
                    if decode_result.parsed_generation.valid_parse
                    else "invalid_parse"
                ),
                "parse_errors": list(decode_result.parsed_generation.parse_errors),
                "generated_coord_tokens": list(
                    decode_result.parsed_generation.generated_coord_tokens
                ),
                "generated_box_xyxy": (
                    None
                    if decode_result.parsed_generation.generated_box_xyxy is None
                    else list(decode_result.parsed_generation.generated_box_xyxy)
                ),
                "target_bbox_xyxy": list(target_box),
                "target_desc_preserved": bool(target_desc_preserved),
                "geometry_valid": bool(geometry_valid),
                "outcome_bucket": outcome_bucket,
                "valid_parse": bool(score.valid_parse),
                "target_iou": float(score.target_iou),
                "success_iou30": bool(score.success_iou30),
                "success_iou50": bool(score.success_iou50),
                "success_iou75": bool(score.success_iou75),
                "same_desc_duplicate_iou95": bool(score.same_desc_duplicate_iou95),
                "max_same_desc_existing_iou": float(score.max_same_desc_existing_iou),
                "duplicate_source_kind": score.duplicate_source_kind,
                "duplicate_source_raw_pred_idx": score.duplicate_source_raw_pred_idx,
                "duplicate_source_guarded_pred_idx": score.duplicate_source_guarded_pred_idx,
                "duplicate_source_bbox_xyxy": (
                    None
                    if score.duplicate_source_bbox_xyxy is None
                    else list(score.duplicate_source_bbox_xyxy)
                ),
                "primary_rescue_success": bool(score.primary_rescue_success),
                "generation_kwargs": _generation_kwargs_artifact_summary(
                    decode_result.generation_kwargs
                ),
                "image_path": str(item["image_path"]),
                "prefix_objects": _prefix_objects_from_rollout_rows(
                    item["rollout_case_rows"],
                    prefix_depth=int(selected_row.get("prefix_depth", 0) or 0),
                ),
                "source_artifacts": source_artifacts,
                "config_sha256": config_sha256,
                "shard_label": shard_label,
                "attn_implementation_selected": selected_attn,
            }
        )
    _write_jsonl_atomic(
        shard_root / FN_RESCUE_MERGE_JSONL_FILES["rescue_generation_rows"],
        generation_rows,
    )
    summary = _refresh_fn_rescue_shard_summary(
        config,
        shard_root=shard_root,
        shard_index=shard_index,
        num_shards=num_shards,
        completed_stage="rescue_decode",
    )
    return {
        "artifact_root": str(config.paths.artifact_root),
        "shard_root": str(shard_root),
        "shard_label": shard_label,
        "row_counts": summary["row_counts"],
    }


def materialize_fn_rescue_attention_replay_shard(
    config: FnRescueConfig,
    shard_index: int,
    num_shards: int,
    *,
    model_handle_loader: Any | None = None,
    processor_input_builder: Any | None = None,
) -> dict[str, Any]:
    shard_root, shard_label, summary = _require_fn_rescue_select_cases_ready(
        config,
        stage_name="attention_replay",
        shard_index=shard_index,
        num_shards=num_shards,
    )
    completed_stages = {str(stage) for stage in summary.get("stages_completed") or []}
    generation_path = shard_root / FN_RESCUE_MERGE_JSONL_FILES["rescue_generation_rows"]
    if "rescue_decode" not in completed_stages or not generation_path.exists():
        raise RuntimeError(
            f"FN-rescue attention_replay requires rescue_decode outputs for {shard_label}"
        )
    generation_rows = read_jsonl(generation_path)
    if not generation_rows:
        _write_jsonl_atomic(
            shard_root / FN_RESCUE_MERGE_JSONL_FILES["rescue_replay_prefix_rows"],
            [],
        )
        _write_jsonl_atomic(
            shard_root / FN_RESCUE_MERGE_JSONL_FILES["rescue_decision_context_rows"],
            [],
        )
        _write_jsonl_atomic(
            shard_root / FN_RESCUE_MERGE_JSONL_FILES["rescue_attention_region_rows"],
            [],
        )
        summary = _refresh_fn_rescue_shard_summary(
            config,
            shard_root=shard_root,
            shard_index=shard_index,
            num_shards=num_shards,
            completed_stage="attention_replay",
        )
        return {
            "artifact_root": str(config.paths.artifact_root),
            "shard_root": str(shard_root),
            "shard_label": shard_label,
            "row_counts": summary["row_counts"],
        }
    loader = model_handle_loader or _default_fn_rescue_model_handle_loader
    build_inputs = processor_input_builder or _default_fn_rescue_processor_input_builder
    model_handle = loader(config)
    selected_attn = _model_attn_implementation(getattr(model_handle, "model", None))
    _ensure_fn_rescue_manifest(
        config,
        expected_shards=num_shards,
        attn_implementation_selected=selected_attn,
    )
    source_artifacts = _source_artifact_summary_for_config(config)
    config_sha256 = _config_sha256(config)
    items = {
        (str(item["case_id"]), str(item["rescue_row"]["planned_rescue_tier"])): item
        for item in _emitted_fn_rescue_stage_items(config, shard_root=shard_root)
    }
    replay_prefix_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    attention_rows: list[dict[str, Any]] = []
    for generation_row in generation_rows:
        case_id = str(generation_row["case_id"])
        tier = str(generation_row["rescue_tier"])
        key = (case_id, tier)
        if key not in items:
            raise RuntimeError(
                f"FN-rescue attention_replay could not find emitted rescue context for {key!r}"
            )
        item = items[key]
        selected_row = item["selected_row"]
        rescue_row = item["rescue_row"]
        assistant_prefix_text = str(generation_row["assistant_prefix_text"])
        stored_assistant_prefix_token_ids = tuple(
            int(token_id) for token_id in generation_row["assistant_prefix_token_ids"]
        )
        processor_inputs = build_inputs(
            model_handle=model_handle,
            config=config,
            stage_name="attention_replay",
            selected_row=selected_row,
            rescue_row=rescue_row,
            dataset_row=item["dataset_row"],
            scored_row=item["scored_row"],
            rollout_case_rows=item["rollout_case_rows"],
            candidate_rows=item["candidate_rows"],
            messages=item["messages"],
            assistant_prefix_text=assistant_prefix_text,
            image_path=item["image_path"],
        )
        replay_result = replay_rescue_attention_rows(
            model_handle=model_handle,
            processor_inputs=processor_inputs,
            tier=tier,
            stored_assistant_prefix_token_ids=stored_assistant_prefix_token_ids,
            region_rows=item["candidate_rows"],
            attention_tensors=None,
        )
        base_row = {
            "case_id": case_id,
            "source_line_idx": int(rescue_row["source_line_idx"]),
            "target_gt_idx": canonical_target_gt_idx(selected_row),
            "target_desc": str(rescue_row["target_desc"]),
            "rescue_tier": tier,
            "prefix_quality": str(selected_row["prefix_quality"]),
            "binding_bucket": str(selected_row["binding_bucket"]),
            "depth_bucket": str(selected_row["depth_bucket"]),
            "object_count_bucket": str(selected_row["object_count_bucket"]),
            "assistant_prefix_text": assistant_prefix_text,
            "assistant_prefix_token_ids": list(stored_assistant_prefix_token_ids),
            "hint_x1": generation_row.get("hint_x1"),
            "hint_x1_raw": generation_row.get("hint_x1_raw"),
            "hint_x1_used": generation_row.get("hint_x1_used"),
            "hint_x1_clamped": bool(generation_row.get("hint_x1_clamped", False)),
            "image_path": str(item["image_path"]),
            "source_artifacts": source_artifacts,
            "config_sha256": config_sha256,
            "shard_label": shard_label,
            "attn_implementation_selected": selected_attn,
        }
        replay_prefix_rows.append(
            {
                **base_row,
                "replay_prefix_token_ids": list(replay_result.replay_prefix_token_ids),
                "replay_prefix_token_match": bool(
                    tuple(stored_assistant_prefix_token_ids)
                    == tuple(replay_result.replay_prefix_token_ids)
                ),
                "query_role": replay_result.query_role,
                "query_index": replay_result.query_index,
                "last_token_text": replay_result.last_token_text,
                "status": replay_result.status,
                "errors": list(replay_result.errors),
            }
        )
        decision_rows.append(
            {
                **base_row,
                "query_role": replay_result.query_role,
                "query_index": replay_result.query_index,
                "last_token_text": replay_result.last_token_text,
                "status": replay_result.status,
                "errors": list(replay_result.errors),
            }
        )
        for row in replay_result.rows:
            replay_row = dict(row)
            attention_rows.append(
                {
                    **base_row,
                    **replay_row,
                    **_split_rescue_attention_region_key(str(replay_row["region_kind"])),
                }
            )
    _write_jsonl_atomic(
        shard_root / FN_RESCUE_MERGE_JSONL_FILES["rescue_replay_prefix_rows"],
        replay_prefix_rows,
    )
    _write_jsonl_atomic(
        shard_root / FN_RESCUE_MERGE_JSONL_FILES["rescue_decision_context_rows"],
        decision_rows,
    )
    _write_jsonl_atomic(
        shard_root / FN_RESCUE_MERGE_JSONL_FILES["rescue_attention_region_rows"],
        attention_rows,
    )
    summary = _refresh_fn_rescue_shard_summary(
        config,
        shard_root=shard_root,
        shard_index=shard_index,
        num_shards=num_shards,
        completed_stage="attention_replay",
    )
    return {
        "artifact_root": str(config.paths.artifact_root),
        "shard_root": str(shard_root),
        "shard_label": shard_label,
        "row_counts": summary["row_counts"],
    }


def merge_fn_rescue_shards(
    root: Path | str, expected_shards: int | Sequence[int | str | Mapping[str, Any]]
) -> dict[str, Any]:
    """Merge shard-local FN-rescue artifacts into a root-level view."""

    root_path = Path(root)
    shard_labels = _expected_shard_labels(expected_shards)
    shard_summaries: list[dict[str, Any]] = []
    merged_rows: dict[str, list[dict[str, Any]]] = {
        key: [] for key in FN_RESCUE_MERGE_JSONL_FILES
    }
    merged_row_counts_from_shards: dict[str, int] = {
        key: 0 for key in FN_RESCUE_MERGE_JSONL_FILES
    }
    seen_keys: dict[str, set[tuple[Any, ...]]] = {
        key: set() for key in FN_RESCUE_MERGE_JSONL_FILES
    }

    for shard_label in shard_labels:
        shard_root = root_path / "shards" / shard_label
        if not shard_root.exists():
            raise FileNotFoundError(f"missing FN-rescue shard root: {shard_root}")
        summary_path = shard_root / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"missing FN-rescue shard summary: {summary_path}")
        shard_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        shard_summaries.append(shard_summary)
        summary_row_counts = shard_summary.get("row_counts")
        if not isinstance(summary_row_counts, Mapping):
            raise ValueError(f"shard summary row_counts missing or invalid: {summary_path}")
        for key, filename in FN_RESCUE_MERGE_JSONL_FILES.items():
            path = shard_root / filename
            if not path.exists():
                raise FileNotFoundError(f"missing FN-rescue shard file: {path}")
            rows = read_jsonl(path)
            if key not in summary_row_counts:
                raise ValueError(f"shard summary row_counts missing key {key!r}: {summary_path}")
            expected_count = int(summary_row_counts[key])
            actual_count = len(rows)
            if expected_count != actual_count:
                raise ValueError(
                    f"shard summary row_counts mismatch for {key}: expected {expected_count}, got {actual_count}"
                )
            merged_row_counts_from_shards[key] += actual_count
            for row in rows:
                row_key = _merged_row_key(key, row)
                if row_key in seen_keys[key]:
                    raise ValueError(f"duplicate key in {filename}: {row_key!r}")
                seen_keys[key].add(row_key)
                merged_rows[key].append(dict(row))

    merged_row_counts = {key: len(rows) for key, rows in merged_rows.items()}
    for key, total_count in merged_row_counts.items():
        if total_count != merged_row_counts_from_shards[key]:
            raise ValueError(
                f"merged row counts do not match shard row_counts for {key}: {total_count} != {merged_row_counts_from_shards[key]}"
            )

    config_sha256 = _singleton_value(
        shard_summaries, "config_sha256", "config hash differs across shards"
    )
    checkpoint = _singleton_value(
        shard_summaries, "checkpoint", "checkpoint differs across shards"
    )
    evidence_scope = _singleton_value(
        shard_summaries, "evidence_scope", "evidence_scope differs across shards"
    )
    source_artifacts = _singleton_value(
        shard_summaries, "source_artifacts", "source artifact summaries differ across shards"
    )

    selected_case_ids = {str(row["case_id"]) for row in merged_rows["selected_rescue_cases"]}
    rescue_rows = merged_rows["rescue_rows"]
    _validate_rescue_row_planned_tier_coverage(
        selected_case_ids=selected_case_ids, rescue_rows=rescue_rows
    )
    emitted_rows = [row for row in rescue_rows if bool(row.get("emitted"))]
    skipped_rows = [row for row in rescue_rows if not bool(row.get("emitted"))]
    if len(emitted_rows) + len(skipped_rows) != len(rescue_rows):
        raise ValueError("rescue_rows denominator ledger must be partitioned into emitted and skipped rows")
    emitted_decode_attempts = len(emitted_rows)
    emitted_wrong_control = sum(
        1
        for row in emitted_rows
        if row.get("planned_rescue_tier") == "desc_x1_wrong_control"
    )
    for key in (
        "rescue_generation_rows",
        "rescue_replay_prefix_rows",
        "rescue_decision_context_rows",
    ):
        if len(merged_rows[key]) != emitted_decode_attempts:
            raise ValueError(
                f"{FN_RESCUE_MERGE_JSONL_FILES[key]} row count must equal emitted decode attempts"
            )
    if len(merged_rows["wrong_control_rows"]) != emitted_wrong_control:
        raise ValueError(
            "wrong_control_rows row count must equal emitted desc_x1_wrong_control attempts"
        )

    temp_root = Path(tempfile.mkdtemp(prefix="fn_rescue_merge_", dir=root_path))
    try:
        for key, filename in FN_RESCUE_MERGE_JSONL_FILES.items():
            _write_jsonl_atomic(temp_root / filename, merged_rows[key])
        merge_summary = {
            "artifact_root": str(root_path),
            "checkpoint": checkpoint,
            "evidence_scope": evidence_scope,
            "config_sha256": config_sha256,
            "expected_shard_labels": shard_labels,
            "source_artifacts": source_artifacts,
            "row_counts": merged_row_counts,
            "output_files": {},
            "validation_status": "ok",
        }
        for filename in FN_RESCUE_MERGE_JSONL_FILES.values():
            (temp_root / filename).replace(root_path / filename)
        merge_summary["output_files"] = {
            key: _file_output_summary(root_path / filename)
            for key, filename in FN_RESCUE_MERGE_JSONL_FILES.items()
        }
        _write_json_atomic(root_path / "merge_summary.json", merge_summary)
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)

    summary = summarize_fn_rescue_artifacts(root_path)
    _write_json_atomic(root_path / "summary.json", summary)
    return {
        "artifact_root": str(root_path),
        "expected_shard_labels": shard_labels,
        "config_sha256": config_sha256,
        "validation_status": "ok",
        "row_counts": summary["row_counts"],
    }


def summarize_fn_rescue_artifacts(root: Path | str) -> dict[str, Any]:
    """Summarize merged FN-rescue artifacts with denominator-first statistics."""

    root_path = Path(root)
    merge_summary = _maybe_read_json(root_path / "merge_summary.json") or {}
    selected_rows = _safe_read_jsonl(
        root_path / FN_RESCUE_MERGE_JSONL_FILES["selected_rescue_cases"]
    )
    rescue_rows = _safe_read_jsonl(root_path / FN_RESCUE_MERGE_JSONL_FILES["rescue_rows"])
    generation_rows = _safe_read_jsonl(
        root_path / FN_RESCUE_MERGE_JSONL_FILES["rescue_generation_rows"]
    )
    attention_rows = _safe_read_jsonl(
        root_path / FN_RESCUE_MERGE_JSONL_FILES["rescue_attention_region_rows"]
    )
    wrong_control_rows = _safe_read_jsonl(
        root_path / FN_RESCUE_MERGE_JSONL_FILES["wrong_control_rows"]
    )

    row_counts = {
        key: len(_safe_read_jsonl(root_path / filename))
        for key, filename in FN_RESCUE_MERGE_JSONL_FILES.items()
    }
    case_rows = selected_rows or _unique_case_rows_from_rescue_rows(rescue_rows)
    selected_case_counts_by_bucket = {
        "prefix_quality": _count_by_key(case_rows, "prefix_quality"),
        "binding_bucket": _count_by_key(case_rows, "binding_bucket"),
        "depth_bucket": _count_by_key(case_rows, "depth_bucket"),
        "object_count_bucket": _count_by_key(case_rows, "object_count_bucket"),
    }
    denominator_by_tier = _denominator_stats_by_tier(
        rescue_rows, generation_rows=generation_rows
    )
    generation_stats = _generation_stats_by_tier(generation_rows)
    attention_summary = _attention_summary(attention_rows)
    wrong_control_distribution = _count_by_key(
        wrong_control_rows, "wrong_control_source_kind"
    )
    source_selected_cases_path = (
        merge_summary.get("source_artifacts", {})
        .get("source_selected_cases", {})
        .get("path")
    )
    interpretation_bounds = [
        "GT leakage remains possible because rescue continuations are conditioned on GT-derived hints.",
        "Evidence scope is linked-stratified and not the full val200 FN universe.",
        "This analysis is diagnostic only and not deployable inference behavior.",
        "Gallery rows are qualitative only and are not a metric source.",
    ]
    return {
        "artifact_root": str(root_path),
        "checkpoint": merge_summary.get("checkpoint"),
        "evidence_scope": merge_summary.get("evidence_scope"),
        "config_sha256": merge_summary.get("config_sha256"),
        "expected_shard_labels": merge_summary.get("expected_shard_labels", []),
        "source_artifacts": merge_summary.get("source_artifacts", {}),
        "source_selected_cases_path": source_selected_cases_path,
        "output_files": merge_summary.get("output_files", {}),
        "row_counts": row_counts,
        "selected_case_count": len(case_rows),
        "selected_case_counts_by_bucket": selected_case_counts_by_bucket,
        "counts_by_case_bucket": selected_case_counts_by_bucket,
        "denominator": denominator_by_tier,
        "generation_stats": generation_stats,
        "wrong_control_source_distribution": wrong_control_distribution,
        "attention_summary": attention_summary,
        "attention_union_summary": attention_summary["union_by_tier_role_layer_group"],
        "interpretation_bounds": interpretation_bounds,
        "gallery_metric_policy": "excluded",
        "validation_status": merge_summary.get("validation_status", "unknown"),
    }


def write_fn_rescue_report(root: Path | str) -> Path:
    """Write a markdown FN-rescue report grounded in merged artifacts."""

    root_path = Path(root)
    summary = summarize_fn_rescue_artifacts(root_path)
    lines = [
        "# FN-Rescue Continuation Report",
        "",
        "## Scope",
        f"- Evidence scope: `{summary.get('evidence_scope')}`",
        f"- Checkpoint: `{summary.get('checkpoint')}`",
        f"- Config hash: `{summary.get('config_sha256')}`",
        f"- Source Selected Cases Path: `{summary.get('source_selected_cases_path')}`",
        "",
        "## Denominator Ledger",
    ]
    denominator_rows: list[list[str]] = []
    for tier, stats in summary["denominator"]["by_tier"].items():
        denominator_rows.append(
            [
                tier,
                str(stats["total"]),
                str(stats["attempted"]),
                str(stats["skipped"]),
                str(stats["invalid_parse"]),
                str(stats["target_desc_not_preserved"]),
                str(stats["geometry_invalid"]),
                str(stats["wrong_control_unavailable"]),
            ]
        )
    lines.append(
        _markdown_table(
            [
                "Tier",
                "Total",
                "Attempted",
                "Skipped",
                "Invalid Parse",
                "Target Desc Not Preserved",
                "Geometry Invalid",
                "Wrong Control Unavailable",
            ],
            denominator_rows,
        )
    )
    lines.extend(
        [
            "",
            "## Case Buckets",
            _markdown_table(
                ["Bucket Family", "Counts"],
                [
                    [bucket_name, json.dumps(counts, sort_keys=True)]
                    for bucket_name, counts in summary["selected_case_counts_by_bucket"].items()
                ],
            ),
            "",
            "## Source Artifacts",
            _markdown_table(
                [
                    "Artifact",
                    "Path",
                    "Hash Kind",
                    "Hash",
                    "Byte Size",
                    "Row Count",
                ],
                [
                    [
                        name,
                        str(item.get("path")),
                        str(item.get("hash_kind")),
                        str(
                            item.get("sha256")
                            or item.get("structural_fingerprint")
                            or ""
                        ),
                        str(item.get("byte_size")),
                        str(item.get("row_count")),
                    ]
                    for name, item in summary["source_artifacts"].items()
                ],
            ),
            "",
            "## Merged Output Row Counts",
            _markdown_table(
                ["Output", "Row Count"],
                [
                    [name, str(count)]
                    for name, count in summary["row_counts"].items()
                ],
            ),
            "",
            "## Parse-Valid Rates By Tier",
            _markdown_table(
                ["Tier", "Count", "Parse Valid Count", "Parse Valid Rate"],
                [
                    [
                        tier,
                        str(stats["count"]),
                        str(stats["parse_valid_count"]),
                        f"{stats['parse_valid_rate']:.3f}",
                    ]
                    for tier, stats in summary["generation_stats"]["by_tier"].items()
                ],
            ),
            "",
            "## IoU Rates By Tier",
            _markdown_table(
                [
                    "Tier",
                    "IoU>=0.3",
                    "IoU>=0.5",
                    "IoU>=0.75",
                    "Raw Rescue Success",
                    "Primary Rescue Success",
                    "Duplicate Rejected",
                ],
                [
                    [
                        tier,
                        f"{stats['success_iou30_rate']:.3f}",
                        f"{stats['success_iou50_rate']:.3f}",
                        f"{stats['success_iou75_rate']:.3f}",
                        f"{stats['raw_rescue_success_rate']:.3f}",
                        f"{stats['primary_rescue_success_rate']:.3f}",
                        str(stats["duplicate_rejected_count"]),
                    ]
                    for tier, stats in summary["generation_stats"]["by_tier"].items()
                ],
            ),
            "",
            "## IoU Rates By Stratum",
            _markdown_table(
                [
                    "Tier",
                    "Prefix Quality",
                    "Binding",
                    "Depth",
                    "Object Count",
                    "Count",
                    "IoU>=0.3",
                    "IoU>=0.5",
                    "IoU>=0.75",
                ],
                [
                    [
                        row["rescue_tier"],
                        row["prefix_quality"],
                        row["binding_bucket"],
                        row["depth_bucket"],
                        row["object_count_bucket"],
                        str(row["count"]),
                        f"{row['success_iou30_rate']:.3f}",
                        f"{row['success_iou50_rate']:.3f}",
                        f"{row['success_iou75_rate']:.3f}",
                    ]
                    for row in summary["generation_stats"]["by_stratum"]
                ],
            ),
        ]
    )
    lines.extend(
        [
            "",
            "## Rescue Success Summary",
            _markdown_table(
                [
                    "Tier",
                    "Raw Rescue Success Count",
                    "Raw Rescue Success Rate",
                    "Primary Rescue Success Count",
                    "Primary Rescue Success Rate",
                    "Duplicate Rejected Count",
                ],
                [
                    [
                        tier,
                        str(stats["raw_rescue_success_count"]),
                        f"{stats['raw_rescue_success_rate']:.3f}",
                        str(stats["primary_rescue_success_count"]),
                        f"{stats['primary_rescue_success_rate']:.3f}",
                        str(stats["duplicate_rejected_count"]),
                    ]
                    for tier, stats in summary["generation_stats"]["by_tier"].items()
                ],
            ),
            "",
            "## Wrong-Control Source Distribution",
            _markdown_table(
                ["Source Kind", "Count"],
                [
                    [kind, str(count)]
                    for kind, count in summary["wrong_control_source_distribution"].items()
                ],
            ),
            "",
            "## Attention Union Summary",
            _markdown_table(
                [
                    "Tier",
                    "Role",
                    "Layer Group",
                    "Region Kind",
                    "Rows",
                    "Mean Attention Mass",
                ],
                [
                    [
                        row["rescue_tier"],
                        row["role"],
                        row["layer_group"],
                        row["region_kind"],
                        str(row["count"]),
                        f"{row['mean_attention_mass']:.6f}",
                    ]
                    for row in summary["attention_summary"]["union_by_tier_role_layer_group"]
                ],
            ),
            "",
            "## Per-Instance Attention Diagnostics",
            _markdown_table(
                [
                    "Tier",
                    "Role",
                    "Layer",
                    "Region Kind",
                    "Region Instance",
                    "Rows",
                    "Mean Attention Mass",
                ],
                [
                    [
                        row["rescue_tier"],
                        row["role"],
                        str(row["layer"]),
                        row["region_kind"],
                        row["region_instance_id"],
                        str(row["count"]),
                        f"{row['mean_attention_mass']:.6f}",
                    ]
                    for row in summary["attention_summary"]["instance_diagnostics"]
                ],
            ),
            "",
            "## Interpretation Bounds",
            *[f"- {item}" for item in summary["interpretation_bounds"]],
            "",
            "Gallery rows are qualitative only and are not a metric source.",
            "",
        ]
    )
    report_path = root_path / "report.md"
    _write_text_atomic(report_path, "\n".join(lines))
    _write_json_atomic(root_path / "summary.json", summary)
    return report_path


def write_fn_rescue_gallery(
    config_or_root: FnRescueConfig | Path | str, *, per_bucket: int | None = None
) -> dict[str, Any]:
    """Write deterministic qualitative gallery index and visualization resources."""

    root_path, configured_per_bucket, runtime_spec, source_gt_path = _gallery_context(
        config_or_root
    )
    bucket_cap = configured_per_bucket if per_bucket is None else int(per_bucket)
    if bucket_cap <= 0:
        raise ValueError("per_bucket must be positive")

    selected_rows = _safe_read_jsonl(
        root_path / FN_RESCUE_MERGE_JSONL_FILES["selected_rescue_cases"]
    )
    rescue_rows = _safe_read_jsonl(root_path / FN_RESCUE_MERGE_JSONL_FILES["rescue_rows"])
    generation_rows = _safe_read_jsonl(
        root_path / FN_RESCUE_MERGE_JSONL_FILES["rescue_generation_rows"]
    )
    wrong_control_rows = _safe_read_jsonl(
        root_path / FN_RESCUE_MERGE_JSONL_FILES["wrong_control_rows"]
    )

    selected_by_case = {
        str(row.get("case_id", row.get("source_line_idx"))): row for row in selected_rows
    }
    rescue_rows_by_case = _group_rows_by_key(rescue_rows, "case_id")
    generation_by_case_tier = {
        (str(row["case_id"]), str(row["rescue_tier"])): row for row in generation_rows
    }
    wrong_control_by_case = {
        str(row["case_id"]): row for row in wrong_control_rows
    }

    buckets: dict[str, list[str]] = {name: [] for name in (
        "desc_only_success",
        "desc_x1_only_success",
        "both_fail",
        "wrong_control_binds_competitor",
        "duplicate_copy_rejected",
    )}
    ordered_case_ids = sorted(
        {str(row["case_id"]) for row in rescue_rows},
        key=lambda case_id: (
            int(_gallery_case_row(selected_by_case, rescue_rows_by_case, case_id)["source_line_idx"]),
            case_id,
        ),
    )
    for case_id in ordered_case_ids:
        desc_only = generation_by_case_tier.get((case_id, "desc_only"))
        desc_x1 = generation_by_case_tier.get((case_id, "desc_x1"))
        wrong_control = generation_by_case_tier.get((case_id, "desc_x1_wrong_control"))
        if desc_only and bool(desc_only.get("primary_rescue_success")):
            buckets["desc_only_success"].append(case_id)
        if (
            desc_x1
            and bool(desc_x1.get("primary_rescue_success"))
            and not bool(desc_only and desc_only.get("primary_rescue_success"))
        ):
            buckets["desc_x1_only_success"].append(case_id)
        if (
            desc_only is not None
            and desc_x1 is not None
            and not bool(desc_only.get("primary_rescue_success"))
            and not bool(desc_x1.get("primary_rescue_success"))
        ):
            buckets["both_fail"].append(case_id)
        if wrong_control and (
            bool(wrong_control.get("same_desc_duplicate_iou95"))
            or wrong_control.get("duplicate_source_kind")
            in {"same_desc_rollout_prediction", "same_desc_competitor_gt_object"}
        ):
            buckets["wrong_control_binds_competitor"].append(case_id)
        if any(
            bool(row.get("same_desc_duplicate_iou95"))
            for (row_case_id, _), row in generation_by_case_tier.items()
            if row_case_id == case_id
        ):
            buckets["duplicate_copy_rejected"].append(case_id)

    gallery_root = root_path / "gallery"
    vis_root = gallery_root / "vis_resources"
    images_root = gallery_root / "images"
    vis_root.mkdir(parents=True, exist_ok=True)
    images_root.mkdir(parents=True, exist_ok=True)
    index_rows: list[dict[str, Any]] = []
    resource_paths: dict[str, str] = {}
    source_scored_rows = read_jsonl(source_gt_path)
    for bucket_name, case_ids in buckets.items():
        selected_case_ids = case_ids[:bucket_cap]
        resource_rows: list[dict[str, Any]] = []
        resource_path = vis_root / f"{bucket_name}.jsonl"
        resource_paths[bucket_name] = str(resource_path)
        for case_id in selected_case_ids:
            case_row = _gallery_case_row(selected_by_case, rescue_rows_by_case, case_id)
            desc_only = generation_by_case_tier.get((case_id, "desc_only"))
            desc_x1 = generation_by_case_tier.get((case_id, "desc_x1"))
            wrong_control = generation_by_case_tier.get((case_id, "desc_x1_wrong_control"))
            wrong_control_source = wrong_control_by_case.get(case_id)
            render_info = _render_fn_rescue_gallery_case_image(
                gallery_root=gallery_root,
                bucket_name=bucket_name,
                case_row=case_row,
                runtime_spec=runtime_spec,
                source_scored_rows=source_scored_rows,
                source_gt_path=source_gt_path,
                desc_only=desc_only,
                desc_x1=desc_x1,
                wrong_control_rescue=wrong_control,
                wrong_control_source=wrong_control_source,
            )
            resource_rows.append(
                {
                    "bucket_name": bucket_name,
                    "case_id": case_id,
                    "source_line_idx": int(case_row["source_line_idx"]),
                    "target_gt_idx": int(case_row["target_gt_idx"]),
                    "target_desc": str(case_row["target_desc"]),
                    "source_kind": "fn_rescue_gallery",
                    "visual_note": "qualitative_only_not_metric_source",
                    "rendering_status": render_info["rendering_status"],
                    "rendered_image": render_info["rendered_image"],
                    "render_error": render_info.get("render_error"),
                    "provenance": render_info.get("provenance"),
                    "debug": {
                        "visual_roles": render_info.get("visual_roles"),
                    },
                    "rescue_rows": {
                        "desc_only": desc_only,
                        "desc_x1": desc_x1,
                        "desc_x1_wrong_control": wrong_control,
                    },
                    "wrong_control_source": wrong_control_source,
                }
            )
            index_rows.append(
                {
                    "bucket_name": bucket_name,
                    "case_id": case_id,
                    "source_line_idx": int(case_row["source_line_idx"]),
                    "target_gt_idx": int(case_row["target_gt_idx"]),
                    "target_desc": str(case_row["target_desc"]),
                    "source_kind": "fn_rescue_gallery",
                    "visual_note": "qualitative_only_not_metric_source",
                    "rendering_status": render_info["rendering_status"],
                    "rendered_image": render_info["rendered_image"],
                    "render_error": render_info.get("render_error"),
                    "provenance": render_info.get("provenance"),
                    "debug": {"visual_roles": render_info.get("visual_roles")},
                    "resource_path": str(resource_path),
                }
            )
        _write_jsonl_atomic(resource_path, resource_rows)

    summary = {
        "artifact_root": str(root_path),
        "source_kind": "fn_rescue_gallery",
        "visual_note": "qualitative_only_not_metric_source",
        "bucket_cap": bucket_cap,
        "bucket_counts": {
            bucket_name: sum(1 for row in index_rows if row["bucket_name"] == bucket_name)
            for bucket_name in buckets
        },
        "rendered_image_count": sum(
            1 for row in index_rows if row.get("rendering_status") == "rendered"
        ),
        "resource_paths": resource_paths,
    }
    _write_jsonl_atomic(gallery_root / "gallery_index.jsonl", index_rows)
    _write_json_atomic(gallery_root / "gallery_summary.json", summary)
    return {
        "gallery_root": str(gallery_root),
        "gallery_index": str(gallery_root / "gallery_index.jsonl"),
        "gallery_summary": str(gallery_root / "gallery_summary.json"),
        "resource_paths": resource_paths,
        "bucket_cap": bucket_cap,
    }


def _normalize_fn_rescue_stages(stages: str | Sequence[str]) -> list[str]:
    if isinstance(stages, str):
        items = [item.strip() for item in stages.split(",") if item.strip()]
    else:
        items = [str(item).strip() for item in stages if str(item).strip()]
    if not items:
        raise ValueError("at least one FN-rescue stage is required")
    invalid = [item for item in items if item not in FN_RESCUE_STAGES]
    if invalid:
        raise ValueError(f"unknown FN-rescue stages: {invalid!r}")
    return items


def _source_artifact_summary_for_config(
    config: FnRescueConfig,
) -> dict[str, dict[str, Any]]:
    return dict(config.source_artifacts or _build_source_artifact_summary(config.paths))


def _config_sha256(config: FnRescueConfig) -> str:
    payload = {
        "paths": {
            field_name: (
                None
                if getattr(config.paths, field_name) is None
                else str(getattr(config.paths, field_name))
            )
            for field_name in _SOURCE_PATH_FIELDS + ("artifact_root",)
        },
        "selection": asdict(config.selection),
        "execution": asdict(config.execution),
    }
    digest = hashlib.sha256()
    digest.update(json.dumps(payload, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def _group_rows_by_key(
    rows: Iterable[Mapping[str, Any]], key: str
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(dict(row))
    return grouped


def _group_rows_by_int_key(
    rows: Iterable[Mapping[str, Any]], key: str
) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row[key])].append(dict(row))
    return grouped


def _object_counts_by_case(
    selected_rows: Sequence[Mapping[str, Any]],
    rollout_by_source_line: Mapping[int, Sequence[Mapping[str, Any]]],
) -> dict[Any, int]:
    counts: dict[Any, int] = {}
    for row in selected_rows:
        source_line_idx = int(row["source_line_idx"])
        for rollout_row in rollout_by_source_line.get(source_line_idx, ()):
            count = rollout_row.get("dataset_gt_count", rollout_row.get("gt_count"))
            if count is not None:
                counts[str(row.get("case_id", source_line_idx))] = int(count)
                counts[source_line_idx] = int(count)
                break
    return counts


def _augment_selected_rows_with_object_counts(
    selected_rows: Sequence[Mapping[str, Any]],
    rollout_by_source_line: Mapping[int, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    augmented: list[dict[str, Any]] = []
    for row in selected_rows:
        item = dict(row)
        if item.get("dataset_gt_count") is None:
            for rollout_row in rollout_by_source_line.get(int(item["source_line_idx"]), ()):
                count = rollout_row.get("dataset_gt_count", rollout_row.get("gt_count"))
                if count is not None:
                    item["dataset_gt_count"] = int(count)
                    break
        augmented.append(item)
    return augmented


def _read_jsonl_indexed(path: Path, *, index_key: str | None) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for line_index, row in enumerate(read_jsonl(path)):
        if index_key is not None and row.get(index_key) is not None:
            row_index = int(row[index_key])
        else:
            row_index = line_index
        indexed[row_index] = dict(row)
    return indexed


def _require_line_index_row(
    rows: Sequence[Mapping[str, Any]], index: int, label: str
) -> Mapping[str, Any]:
    if index < 0 or index >= len(rows):
        raise ValueError(f"{label} missing row for source_line_idx {index}")
    return rows[index]


def _validate_candidate_join_keys(
    case_row: Mapping[str, Any], candidate_rows: Sequence[Mapping[str, Any]]
) -> None:
    if not candidate_rows:
        raise ValueError(f"no candidate rows found for case_id {case_row.get('case_id')!r}")
    expected_case_id = str(case_row.get("case_id", case_row.get("source_line_idx")))
    expected_source_line_idx = int(case_row["source_line_idx"])
    expected_target_gt_idx = canonical_target_gt_idx(case_row)
    expected_target_desc = str(case_row["target_desc"])
    for row in candidate_rows:
        if str(row.get("case_id")) != expected_case_id:
            raise ValueError("candidate join mismatch on case_id")
        if int(row.get("source_line_idx")) != expected_source_line_idx:
            raise ValueError("candidate join mismatch on source_line_idx")
        if canonical_target_gt_idx(row) != expected_target_gt_idx:
            raise ValueError("candidate join mismatch on target_gt_idx")
        if str(row.get("target_desc")) != expected_target_desc:
            raise ValueError("candidate join mismatch on target_desc")
        if (
            row.get("desc") is not None
            and str(row.get("region_kind")) != "far_background"
            and str(row.get("desc")) != expected_target_desc
        ):
            raise ValueError("candidate join mismatch on target_desc")


def _require_single_region(
    candidate_rows: Sequence[Mapping[str, Any]], region_kind: str
) -> Mapping[str, Any]:
    matches = [row for row in candidate_rows if row.get("region_kind") == region_kind]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly 1 candidate region for {region_kind!r}; found {len(matches)}"
        )
    return matches[0]


def _same_desc_rollout_rows(
    rollout_rows: Sequence[Mapping[str, Any]], *, target_desc: str
) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in rollout_rows
        if str(row.get("pred_desc", "")).strip() == target_desc
        and valid_xyxy_box(row.get("pred_points", ()))
    ]


def _all_gt_boxes_from_scored_row(scored_row: Mapping[str, Any]) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    for row in scored_row.get("gt", []) or []:
        points = row.get("points") if isinstance(row, Mapping) else None
        if valid_xyxy_box(points or ()):
            boxes.append(_box_tuple(points))
    return boxes


def _all_prediction_boxes_from_scored_row(
    scored_row: Mapping[str, Any]
) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    for row in scored_row.get("pred", []) or []:
        points = row.get("points") if isinstance(row, Mapping) else None
        if valid_xyxy_box(points or ()):
            boxes.append(_box_tuple(points))
    return boxes


def _build_selection_candidate_rows(
    *,
    case_row: Mapping[str, Any],
    case_candidate_rows: Sequence[Mapping[str, Any]],
    rollout_case_rows: Sequence[Mapping[str, Any]],
    target_desc: str,
    prefix_depth: int,
    shard_label: str,
) -> list[dict[str, Any]]:
    case_id = str(case_row["case_id"])
    source_line_idx = int(case_row["source_line_idx"])
    target_gt_idx = canonical_target_gt_idx(case_row)
    base_rows: list[dict[str, Any]] = []
    for row in case_candidate_rows:
        source_kind = str(row["region_kind"])
        region_kind = (
            "same_desc_competitor_gt_object"
            if source_kind == "same_desc_gt"
            else source_kind
        )
        gt_idx = row.get("gt_idx")
        if region_kind == "target_gt":
            region_instance_id = f"gt:{target_gt_idx}"
            source_index = target_gt_idx
        elif region_kind == "context_ring":
            region_instance_id = "context_ring:0"
            source_index = 0
        elif region_kind == "far_background":
            region_instance_id = "far_background:0"
            source_index = 0
        else:
            region_instance_id = f"gt:{int(gt_idx)}"
            source_index = int(gt_idx)
        base_rows.append(
            {
                "case_id": case_id,
                "source_line_idx": source_line_idx,
                "target_gt_idx": target_gt_idx,
                "target_desc": target_desc,
                "region_kind": region_kind,
                "region_instance_id": region_instance_id,
                "source_index": source_index,
                "aggregation_scope": "instance",
                "bbox_xyxy": list(_box_tuple(row["bbox_xyxy"])),
                "source_region_kind": source_kind,
                "shard_label": shard_label,
            }
        )
    for rollout_row in rollout_case_rows:
        raw_pred_idx = rollout_row.get("raw_pred_idx")
        if raw_pred_idx is None or not valid_xyxy_box(rollout_row.get("pred_points", ())):
            continue
        pred_points = list(_box_tuple(rollout_row["pred_points"]))
        if not bool(rollout_row.get("suppressed_by_guard", False)) and int(raw_pred_idx) < int(
            prefix_depth
        ):
            base_rows.append(
                {
                    "case_id": case_id,
                    "source_line_idx": source_line_idx,
                    "target_gt_idx": target_gt_idx,
                    "target_desc": target_desc,
                    "region_kind": "previous_generated_object",
                    "region_instance_id": f"raw_pred:{int(raw_pred_idx)}",
                    "source_index": int(raw_pred_idx),
                    "aggregation_scope": "instance",
                    "bbox_xyxy": pred_points,
                    "source_region_kind": "rollout_prefix",
                    "pred_desc": str(rollout_row.get("pred_desc", "")),
                    "shard_label": shard_label,
                }
            )
        if str(rollout_row.get("pred_desc", "")).strip() == target_desc:
            base_rows.append(
                {
                    "case_id": case_id,
                    "source_line_idx": source_line_idx,
                    "target_gt_idx": target_gt_idx,
                    "target_desc": target_desc,
                    "region_kind": "same_desc_rollout_prediction",
                    "region_instance_id": f"raw_pred:{int(raw_pred_idx)}",
                    "source_index": int(raw_pred_idx),
                    "aggregation_scope": "instance",
                    "bbox_xyxy": pred_points,
                    "source_region_kind": "rollout_same_desc",
                    "pred_desc": str(rollout_row.get("pred_desc", "")),
                    "shard_label": shard_label,
                }
            )
    deduped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in base_rows:
        key = (
            row["case_id"],
            row["region_kind"],
            row["region_instance_id"],
        )
        deduped.setdefault(key, row)
    return list(deduped.values())


def _candidate_rows_for_tier(
    *,
    base_candidate_rows: Sequence[Mapping[str, Any]],
    case_row: Mapping[str, Any],
    target_gt_idx: int,
    target_desc: str,
    tier: str,
    wrong_control_source: WrongControlSource | None,
) -> list[dict[str, Any]]:
    rows = [dict(row, planned_rescue_tier=tier) for row in base_candidate_rows]
    if (
        tier == "desc_x1_wrong_control"
        and wrong_control_source is not None
        and wrong_control_source.bbox_xyxy is not None
    ):
        rows.append(
            {
                "case_id": str(case_row["case_id"]),
                "source_line_idx": int(case_row["source_line_idx"]),
                "target_gt_idx": target_gt_idx,
                "target_desc": target_desc,
                "planned_rescue_tier": tier,
                "region_kind": "wrong_control_source_region",
                "region_instance_id": (
                    f"{wrong_control_source.kind}:{0 if wrong_control_source.source_index is None else int(wrong_control_source.source_index)}"
                ),
                "source_index": 0
                if wrong_control_source.source_index is None
                else int(wrong_control_source.source_index),
                "aggregation_scope": "instance",
                "bbox_xyxy": list(wrong_control_source.bbox_xyxy),
                "source_region_kind": wrong_control_source.kind,
                "rejected_candidates": [
                    dict(item) for item in wrong_control_source.rejected_candidates
                ],
                "shard_label": str(base_candidate_rows[0]["shard_label"])
                if base_candidate_rows
                else None,
            }
        )
    return rows


def _expected_shard_labels(
    expected_shards: int | Sequence[int | str | Mapping[str, Any]]
) -> list[str]:
    if isinstance(expected_shards, Integral) and not isinstance(expected_shards, bool):
        shard_count = int(expected_shards)
        return [fn_rescue_shard_label(index, shard_count) for index in range(shard_count)]
    labels = [normalize_fn_rescue_shard(item)["shard_label"] for item in expected_shards]
    if len(set(labels)) != len(labels):
        raise ValueError("expected_shards contains duplicate shard labels")
    return labels


def _merged_row_key(file_key: str, row: Mapping[str, Any]) -> tuple[Any, ...]:
    if file_key == "selected_rescue_cases":
        return (str(row["case_id"]),)
    if file_key == "rescue_rows":
        return (str(row["case_id"]), str(row["planned_rescue_tier"]))
    if file_key in {
        "rescue_generation_rows",
        "rescue_replay_prefix_rows",
        "rescue_decision_context_rows",
        "wrong_control_rows",
    }:
        return (str(row["case_id"]), str(row["rescue_tier"]))
    if file_key == "rescue_candidate_region_rows":
        tier = row.get("rescue_tier", row.get("planned_rescue_tier"))
        return (
            str(row["case_id"]),
            str(tier),
            str(row["region_kind"]),
            str(row["region_instance_id"]),
        )
    if file_key == "rescue_attention_region_rows":
        return (
            str(row["case_id"]),
            str(row["rescue_tier"]),
            str(row["role"]),
            int(row["layer"]),
            int(row["head"]),
            str(row["aggregation_scope"]),
            str(row["region_kind"]),
            str(row["region_instance_id"]),
        )
    raise KeyError(f"unsupported merged file key: {file_key}")


def _singleton_value(
    summaries: Sequence[Mapping[str, Any]], key: str, message: str
) -> Any:
    serialized: dict[str, Any] = {}
    for item in summaries:
        value = item.get(key)
        serialized[json.dumps(value, sort_keys=True)] = value
    if len(serialized) != 1:
        raise ValueError(message)
    return next(iter(serialized.values()))


def _file_output_summary(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "byte_size": _path_byte_size(path),
        "row_count": _jsonl_row_count(path) if path.suffix == ".jsonl" else None,
        "hash_kind": "file_sha256",
        "sha256": _file_sha256(path),
    }


def _maybe_read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _safe_read_jsonl(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path) if path.exists() else []


def _unique_case_rows_from_rescue_rows(
    rescue_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for row in rescue_rows:
        deduped.setdefault(
            str(row["case_id"]),
            {
                "case_id": str(row["case_id"]),
                "source_line_idx": int(row["source_line_idx"]),
                "target_gt_idx": int(row["target_gt_idx"]),
                "target_desc": str(row["target_desc"]),
                "prefix_quality": str(row["prefix_quality"]),
                "binding_bucket": str(row["binding_bucket"]),
                "depth_bucket": str(row["depth_bucket"]),
                "object_count_bucket": str(row["object_count_bucket"]),
            },
        )
    return list(deduped.values())


def _count_by_key(
    rows: Sequence[Mapping[str, Any]], key: str
) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        value = row.get(key)
        if value is not None:
            counts[str(value)] += 1
    return dict(sorted(counts.items()))


def _validate_rescue_row_planned_tier_coverage(
    *,
    selected_case_ids: set[str],
    rescue_rows: Sequence[Mapping[str, Any]],
) -> None:
    rows_by_case: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rescue_rows:
        rows_by_case[str(row["case_id"])].append(row)
    missing_selected = sorted(selected_case_ids.difference(rows_by_case))
    if missing_selected:
        raise ValueError(
            f"planned tier coverage missing rescue rows for selected cases: {missing_selected!r}"
        )
    expected_tiers = sorted(RESCUE_TIERS)
    for case_id in sorted(selected_case_ids):
        case_rows = rows_by_case[case_id]
        tier_counts = _count_by_key(case_rows, "planned_rescue_tier")
        missing_tiers = sorted(set(expected_tiers).difference(tier_counts))
        extra_tiers = sorted(set(tier_counts).difference(expected_tiers))
        duplicate_tiers = sorted(
            tier for tier, count in tier_counts.items() if int(count) != 1
        )
        if missing_tiers or extra_tiers or duplicate_tiers:
            raise ValueError(
                "planned tier coverage invalid for "
                f"{case_id!r}: missing={missing_tiers!r} extra={extra_tiers!r} duplicate={duplicate_tiers!r}"
            )


def _denominator_stats_by_tier(
    rescue_rows: Sequence[Mapping[str, Any]],
    generation_rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    generation_rows = list(generation_rows or [])
    generation_by_key = {
        (str(row["case_id"]), str(row["rescue_tier"])): row for row in generation_rows
    }
    by_tier: dict[str, dict[str, Any]] = {}
    totals = {
        "total_rows": len(rescue_rows),
        "attempted": 0,
        "skipped": 0,
        "invalid_parse": 0,
        "target_desc_not_preserved": 0,
        "geometry_invalid": 0,
        "wrong_control_unavailable": 0,
    }
    for tier in sorted(RESCUE_TIERS):
        rows = [
            row for row in rescue_rows if str(row.get("planned_rescue_tier")) == tier
        ]
        attempted = sum(1 for row in rows if bool(row.get("emitted")))
        skipped = len(rows) - attempted
        invalid_parse = 0
        target_desc_not_preserved = 0
        geometry_invalid = 0
        for row in rows:
            generation_row = generation_by_key.get((str(row["case_id"]), tier))
            if generation_row is None:
                continue
            if not bool(generation_row.get("valid_parse", False)):
                invalid_parse += 1
            if generation_row.get("target_desc_preserved") is False:
                target_desc_not_preserved += 1
            if generation_row.get("geometry_valid") is False:
                geometry_invalid += 1
        wrong_control_unavailable = sum(
            1 for row in rows if row.get("skip_reason") == "wrong_control_unavailable"
        )
        by_tier[tier] = {
            "total": len(rows),
            "attempted": attempted,
            "emitted": attempted,
            "skipped": skipped,
            "invalid_parse": invalid_parse,
            "target_desc_not_preserved": target_desc_not_preserved,
            "geometry_invalid": geometry_invalid,
            "wrong_control_unavailable": wrong_control_unavailable,
            "skip_reasons": _count_by_key(
                [row for row in rows if row.get("skip_reason") is not None], "skip_reason"
            ),
        }
        totals["attempted"] += attempted
        totals["skipped"] += skipped
        totals["invalid_parse"] += invalid_parse
        totals["target_desc_not_preserved"] += target_desc_not_preserved
        totals["geometry_invalid"] += geometry_invalid
        totals["wrong_control_unavailable"] += wrong_control_unavailable
    return {**totals, "by_tier": by_tier}


def _generation_stats_by_tier(
    generation_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    by_tier: dict[str, dict[str, Any]] = {}
    for tier in sorted(RESCUE_TIERS):
        rows = [row for row in generation_rows if str(row["rescue_tier"]) == tier]
        count = len(rows)
        valid_parse_count = sum(1 for row in rows if bool(row.get("valid_parse")))
        success_iou30_count = sum(1 for row in rows if bool(row.get("success_iou30")))
        success_iou50_count = sum(1 for row in rows if bool(row.get("success_iou50")))
        success_iou75_count = sum(1 for row in rows if bool(row.get("success_iou75")))
        primary_rescue_success_count = sum(
            1 for row in rows if bool(row.get("primary_rescue_success"))
        )
        duplicate_rejected_count = sum(
            1 for row in rows if bool(row.get("same_desc_duplicate_iou95"))
        )
        denominator = count if count > 0 else 1
        by_tier[tier] = {
            "count": count,
            "parse_valid_count": valid_parse_count,
            "parse_valid_rate": valid_parse_count / denominator if count else 0.0,
            "valid_parse": valid_parse_count / denominator if count else 0.0,
            "success_iou30_count": success_iou30_count,
            "success_iou30_rate": success_iou30_count / denominator if count else 0.0,
            "success_iou30": success_iou30_count / denominator if count else 0.0,
            "success_iou50_count": success_iou50_count,
            "success_iou50_rate": success_iou50_count / denominator if count else 0.0,
            "success_iou50": success_iou50_count / denominator if count else 0.0,
            "success_iou75_count": success_iou75_count,
            "success_iou75_rate": success_iou75_count / denominator if count else 0.0,
            "success_iou75": success_iou75_count / denominator if count else 0.0,
            "raw_rescue_success_count": success_iou50_count,
            "raw_rescue_success_rate": success_iou50_count / denominator if count else 0.0,
            "primary_rescue_success_count": primary_rescue_success_count,
            "primary_rescue_success_rate": primary_rescue_success_count / denominator if count else 0.0,
            "primary_rescue_success": primary_rescue_success_count / denominator if count else 0.0,
            "duplicate_rejected_count": duplicate_rejected_count,
            "same_desc_duplicate_iou95": duplicate_rejected_count / denominator if count else 0.0,
            "duplicate_source_kind": _count_by_key(rows, "duplicate_source_kind"),
        }
    return {
        "total_rows": len(generation_rows),
        "row_count": len(generation_rows),
        "by_tier": by_tier,
        "by_stratum": _generation_stats_by_stratum(generation_rows),
    }


def _attention_summary(
    attention_rows: Sequence[Mapping[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    union_grouped: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    instance_grouped: dict[tuple[str, str, int, str, str], list[float]] = defaultdict(list)
    for row in attention_rows:
        if str(row.get("aggregation_scope")) == "union":
            key = (
                str(row.get("rescue_tier")),
                str(row.get("role")),
                _layer_group_label(int(row.get("layer", 0))),
                str(row.get("region_kind")),
            )
            union_grouped[key].append(float(row.get("attention_mass", 0.0)))
        elif str(row.get("aggregation_scope")) == "instance":
            key = (
                str(row.get("rescue_tier")),
                str(row.get("role")),
                int(row.get("layer", 0)),
                str(row.get("region_kind")),
                str(row.get("region_instance_id")),
            )
            instance_grouped[key].append(float(row.get("attention_mass", 0.0)))
    union_summary: list[dict[str, Any]] = []
    for key in sorted(union_grouped):
        values = union_grouped[key]
        union_summary.append(
            {
                "rescue_tier": key[0],
                "role": key[1],
                "layer_group": key[2],
                "region_kind": key[3],
                "count": len(values),
                "mean_attention_mass": sum(values) / len(values),
            }
        )
    instance_summary: list[dict[str, Any]] = []
    for key in sorted(instance_grouped):
        values = instance_grouped[key]
        instance_summary.append(
            {
                "rescue_tier": key[0],
                "role": key[1],
                "layer": key[2],
                "region_kind": key[3],
                "region_instance_id": key[4],
                "count": len(values),
                "mean_attention_mass": sum(values) / len(values),
            }
        )
    return {
        "union_by_tier_role_layer_group": union_summary,
        "instance_diagnostics": instance_summary,
    }


def _generation_stats_by_stratum(
    generation_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in generation_rows:
        key = (
            str(row.get("rescue_tier")),
            str(row.get("prefix_quality", "unknown")),
            str(row.get("binding_bucket", "unknown")),
            str(row.get("depth_bucket", "unknown")),
            str(row.get("object_count_bucket", "unknown")),
        )
        grouped[key].append(row)
    summary: list[dict[str, Any]] = []
    for key in sorted(grouped):
        rows = grouped[key]
        count = len(rows)
        denominator = count if count > 0 else 1
        summary.append(
            {
                "rescue_tier": key[0],
                "prefix_quality": key[1],
                "binding_bucket": key[2],
                "depth_bucket": key[3],
                "object_count_bucket": key[4],
                "count": count,
                "success_iou30_rate": sum(
                    1 for row in rows if bool(row.get("success_iou30"))
                )
                / denominator
                if count
                else 0.0,
                "success_iou50_rate": sum(
                    1 for row in rows if bool(row.get("success_iou50"))
                )
                / denominator
                if count
                else 0.0,
                "success_iou75_rate": sum(
                    1 for row in rows if bool(row.get("success_iou75"))
                )
                / denominator
                if count
                else 0.0,
            }
        )
    return summary


def _layer_group_label(layer: int) -> str:
    group_start = (int(layer) // 8) * 8
    group_end = group_start + 7
    return f"layers_{group_start:02d}_{group_end:02d}"


def _gallery_context(
    config_or_root: FnRescueConfig | Path | str,
) -> tuple[Path, int, Mapping[str, Any], Path]:
    if isinstance(config_or_root, FnRescueConfig):
        runtime_spec = _fn_rescue_runtime_spec(config_or_root)
        root_image_dir = Path(runtime_spec["root_image_dir"])
        if not root_image_dir.exists():
            raise ValueError(
                f"gallery runtime unavailable: root_image_dir does not exist: {root_image_dir}"
            )
        return (
            config_or_root.paths.artifact_root,
            config_or_root.execution.gallery_per_bucket,
            runtime_spec,
            config_or_root.paths.gt_vs_pred_scored,
        )
    root_path = Path(config_or_root)
    manifest = _maybe_read_json(root_path / "shards_manifest.json") or {}
    config_path = manifest.get("config_path") if isinstance(manifest, Mapping) else None
    if isinstance(config_path, str) and config_path.strip():
        try:
            config = load_fn_rescue_config(Path(config_path))
            return (
                root_path,
                config.execution.gallery_per_bucket,
                _fn_rescue_runtime_spec(config),
                config.paths.gt_vs_pred_scored,
            )
        except (OSError, ValueError):
            pass
    if not isinstance(manifest, Mapping) or not manifest:
        raise ValueError("gallery runtime unavailable: missing shards_manifest.json")
    runtime_spec = _runtime_spec_from_manifest(manifest)
    source_artifacts = manifest.get("source_artifacts")
    if not isinstance(source_artifacts, Mapping):
        raise ValueError("gallery runtime unavailable: missing manifest source_artifacts")
    gt_vs_pred_scored = source_artifacts.get("gt_vs_pred_scored")
    if not isinstance(gt_vs_pred_scored, Mapping):
        raise ValueError(
            "gallery runtime unavailable: missing manifest gt_vs_pred_scored artifact"
        )
    gt_vs_pred_scored_path = gt_vs_pred_scored.get("path")
    if not isinstance(gt_vs_pred_scored_path, str) or not gt_vs_pred_scored_path.strip():
        raise ValueError(
            "gallery runtime unavailable: missing manifest gt_vs_pred_scored path"
        )
    source_gt_path = Path(gt_vs_pred_scored_path).expanduser().resolve(strict=False)
    if not source_gt_path.exists():
        raise ValueError(
            f"gallery runtime unavailable: gt_vs_pred_scored path does not exist: {source_gt_path}"
        )
    return (
        root_path,
        int(manifest.get("gallery_per_bucket", 1) or 1),
        runtime_spec,
        source_gt_path,
    )


def _canonical_gt_vs_pred_helpers() -> tuple[Any, Any]:
    from src.vis.gt_vs_pred import ensure_gt_vs_pred_vis_resource, render_gt_vs_pred_review

    return ensure_gt_vs_pred_vis_resource, render_gt_vs_pred_review


def _gallery_overlay_bbox(box: Any) -> list[int] | None:
    valid_xyxy_box, _box_iou_xyxy = _load_geometry_helpers()
    if not valid_xyxy_box(box):
        return None
    return [int(v) for v in box]


def _build_fn_rescue_gallery_source_row(
    *,
    scored_row: Mapping[str, Any],
    case_row: Mapping[str, Any],
    bucket_name: str,
    source_gt_path: Path,
    desc_only: Mapping[str, Any] | None,
    desc_x1: Mapping[str, Any] | None,
    wrong_control_rescue: Mapping[str, Any] | None,
    wrong_control_source: Mapping[str, Any] | None,
) -> dict[str, Any]:
    source_row = dict(scored_row)
    pred_rows = [dict(obj) for obj in list(scored_row.get("pred") or [])]
    next_index = 0
    for fallback_index, obj in enumerate(pred_rows):
        raw_index = obj.get("index", fallback_index)
        try:
            next_index = max(next_index, int(raw_index) + 1)
        except (TypeError, ValueError):
            next_index = max(next_index, fallback_index + 1)

    visual_roles: dict[str, dict[str, Any]] = {}

    def _append_overlay(
        *,
        role_key: str,
        desc: str,
        bbox_xyxy: Sequence[int] | None,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        nonlocal next_index
        if bbox_xyxy is None:
            return
        pred_index = next_index
        next_index += 1
        pred_rows.append(
            {
                "index": pred_index,
                "desc": desc,
                "bbox_2d": [int(v) for v in bbox_xyxy],
            }
        )
        visual_roles[role_key] = {
            "desc": desc,
            "pred_index": pred_index,
            "bbox_xyxy": [int(v) for v in bbox_xyxy],
            **(dict(extra) if extra is not None else {}),
        }

    _append_overlay(
        role_key="rescue_desc_only",
        desc="FN rescue desc_only",
        bbox_xyxy=_gallery_overlay_bbox(desc_only.get("generated_box_xyxy") if desc_only else None),
        extra={"rescue_tier": "desc_only", "case_id": str(case_row.get("case_id"))},
    )
    _append_overlay(
        role_key="rescue_desc_x1",
        desc="FN rescue desc_x1",
        bbox_xyxy=_gallery_overlay_bbox(desc_x1.get("generated_box_xyxy") if desc_x1 else None),
        extra={"rescue_tier": "desc_x1", "case_id": str(case_row.get("case_id"))},
    )
    _append_overlay(
        role_key="rescue_desc_x1_wrong_control",
        desc="FN rescue desc_x1_wrong_control",
        bbox_xyxy=_gallery_overlay_bbox(
            wrong_control_rescue.get("generated_box_xyxy") if wrong_control_rescue else None
        ),
        extra={
            "rescue_tier": "desc_x1_wrong_control",
            "case_id": str(case_row.get("case_id")),
        },
    )
    wrong_control_kind = (
        None
        if wrong_control_source is None
        else wrong_control_source.get("wrong_control_source_kind")
    )
    _append_overlay(
        role_key="wrong_control_source",
        desc="wrong-control source",
        bbox_xyxy=_gallery_overlay_bbox(
            None
            if wrong_control_source is None
            else wrong_control_source.get("wrong_control_source_bbox_xyxy")
        ),
        extra={
            "wrong_control_source_kind": wrong_control_kind,
            "case_id": str(case_row.get("case_id")),
        },
    )

    debug_payload = dict(source_row.get("debug") or {})
    debug_payload["visual_roles"] = visual_roles
    source_row["debug"] = debug_payload
    provenance = dict(source_row.get("provenance") or {})
    provenance.update(
        {
            "gallery_case_id": str(case_row.get("case_id")),
            "gallery_bucket_name": bucket_name,
            "gallery_source_line_idx": int(case_row.get("source_line_idx", 0) or 0),
            "gallery_target_gt_idx": int(case_row.get("target_gt_idx", 0) or 0),
            "source_gt_vs_pred_scored": str(source_gt_path),
        }
    )
    source_row["provenance"] = provenance
    source_row["source_kind"] = "fn_rescue_gallery"
    source_row["pred"] = pred_rows
    return source_row


def _render_fn_rescue_gallery_case_image(
    *,
    gallery_root: Path,
    bucket_name: str,
    case_row: Mapping[str, Any],
    runtime_spec: Mapping[str, Any] | None,
    source_scored_rows: Sequence[Mapping[str, Any]] | None,
    source_gt_path: Path | None,
    desc_only: Mapping[str, Any] | None,
    desc_x1: Mapping[str, Any] | None,
    wrong_control_rescue: Mapping[str, Any] | None,
    wrong_control_source: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if runtime_spec is None or source_scored_rows is None or source_gt_path is None:
        raise ValueError("gallery runtime unavailable: missing gallery runtime context")
    try:
        source_line_idx = int(case_row["source_line_idx"])
        scored_row = _require_line_index_row(
            source_scored_rows, source_line_idx, "gallery_gt_vs_pred_scored"
        )
        case_id = str(case_row.get("case_id", source_line_idx))
        safe_case_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", case_id)
        subset_jsonl = (
            gallery_root / "vis_resources" / f"{bucket_name}_{safe_case_id}_source.jsonl"
        )
        canonical_jsonl = (
            gallery_root / "vis_resources" / f"{bucket_name}_{safe_case_id}_canonical.jsonl"
        )
        source_row = _build_fn_rescue_gallery_source_row(
            scored_row=scored_row,
            case_row=case_row,
            bucket_name=bucket_name,
            source_gt_path=source_gt_path,
            desc_only=desc_only,
            desc_x1=desc_x1,
            wrong_control_rescue=wrong_control_rescue,
            wrong_control_source=wrong_control_source,
        )
        _write_jsonl_atomic(subset_jsonl, [source_row])
        ensure_gt_vs_pred_vis_resource, render_gt_vs_pred_review = (
            _canonical_gt_vs_pred_helpers()
        )
        canonical_path = ensure_gt_vs_pred_vis_resource(
            subset_jsonl,
            output_path=canonical_jsonl,
            source_kind="fn_rescue_gallery",
            materialize_matching=True,
        )
        bucket_dir = gallery_root / "images" / bucket_name
        bucket_dir.mkdir(parents=True, exist_ok=True)
        temp_dir = bucket_dir / f".tmp_{safe_case_id}"
        render_gt_vs_pred_review(
            canonical_path,
            out_dir=temp_dir,
            limit=1,
            root_image_dir=Path(runtime_spec["root_image_dir"]),
        )
        rendered_tmp = temp_dir / "vis_0000.png"
        if not rendered_tmp.exists():
            raise FileNotFoundError(f"rendered png missing: {rendered_tmp}")
        final_path = bucket_dir / f"{source_line_idx:05d}_{safe_case_id}.png"
        rendered_tmp.replace(final_path)
        shutil.rmtree(temp_dir, ignore_errors=True)
        return {
            "rendering_status": "rendered",
            "rendered_image": str(final_path),
            "render_error": None,
            "visual_roles": dict(source_row.get("debug") or {}).get("visual_roles"),
            "provenance": dict(source_row.get("provenance") or {}),
        }
    except Exception as exc:
        return {
            "rendering_status": "render_failed",
            "rendered_image": None,
            "render_error": f"{type(exc).__name__}: {exc}",
            "visual_roles": None,
            "provenance": None,
        }


def _gallery_case_row(
    selected_by_case: Mapping[str, Mapping[str, Any]],
    rescue_rows_by_case: Mapping[str, Sequence[Mapping[str, Any]]],
    case_id: str,
) -> Mapping[str, Any]:
    case_row = selected_by_case.get(case_id)
    if case_row is not None and case_row.get("source_line_idx") is not None:
        return case_row
    return rescue_rows_by_case[case_id][0]


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    if not rows:
        rows = [["(none)"] + [""] * (len(headers) - 1)]
    header_line = "| " + " | ".join(headers) + " |"
    divider = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join([header_line, divider, *body])


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with open(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        Path(tmp_name).replace(path)
    finally:
        if Path(tmp_name).exists():
            Path(tmp_name).unlink(missing_ok=True)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    _write_text_atomic(path, json.dumps(payload, sort_keys=True, indent=2) + "\n")


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    text = "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows)
    _write_text_atomic(path, text)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file for pure selection/unit-test workflows."""

    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no} is not a JSON object")
            rows.append(value)
    return rows


__all__ = [
    "BOX_START_TOKEN",
    "ChatTemplateContinuationCheck",
    "FN_RESCUE_MERGE_JSONL_FILES",
    "FN_RESCUE_STAGES",
    "OBJECT_REF_START_TOKEN",
    "FnRescueConfig",
    "FnRescueExecutionConfig",
    "FnRescuePaths",
    "FnRescueSelectionConfig",
    "ParsedRescueGeneration",
    "RescueAttentionReplayResult",
    "RescueBoxScore",
    "RescueDecodeResult",
    "WrongControlSource",
    "build_fn_rescue_dry_run_plan",
    "build_rescue_assistant_prefix",
    "build_rescue_region_membership",
    "canonical_target_gt_idx",
    "check_chat_template_continuation_feasibility",
    "choose_wrong_control_source",
    "coord_token",
    "decode_rescue_tail",
    "fn_rescue_stratum_key",
    "fn_rescue_shard_label",
    "load_fn_rescue_config",
    "materialize_fn_rescue_attention_replay_shard",
    "materialize_fn_rescue_decode_shard",
    "materialize_fn_rescue_feasibility_shard",
    "materialize_fn_rescue_select_cases_shard",
    "merge_fn_rescue_shards",
    "normalize_fn_rescue_shard",
    "parse_generated_bbox",
    "prefix_text_from_raw_compact_predictions",
    "read_jsonl",
    "replay_rescue_attention_rows",
    "render_rescue_hint_row_prefix",
    "score_rescue_box",
    "select_rescue_cases",
    "strip_generation_terminal",
    "summarize_fn_rescue_artifacts",
    "write_fn_rescue_gallery",
    "write_fn_rescue_report",
]
