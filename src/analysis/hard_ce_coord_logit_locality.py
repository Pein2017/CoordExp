"""Hard-CE coordinate-logit and token-row locality diagnostics.

This module is intentionally split into deterministic metric utilities and
heavier checkpoint-driven stages. The pure utilities are unit-tested without a
model; the staged runner loads the hard-CE compact-full checkpoint only when a
GPU-facing stage requests it.
"""

from __future__ import annotations

import json
import math
import re
import shlex
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Iterator, Mapping, MutableMapping, Sequence

import numpy as np
import torch
import yaml

from src.common.detection_chat import build_detection_chat_messages
from src.common.detection_compact_rows import (
    BOX_START_TOKEN,
    OBJECT_REF_START_TOKEN,
    parse_compact_row,
    render_compact_row,
)
from src.config.prompts import get_template_prompts
from src.coord_tokens.codec import get_coord_token_ids, int_to_token
from src.detection.data import (
    NormalizedDetectionObject,
    ObjectOrderingPlan,
    normalize_detection_row,
    parse_raw_detection_row,
)
from src.detection.dataset import _mix_seed as _detection_mix_seed
from src.detection.objective import prepare_detection_training_example
from src.detection.template import get_detection_template
from src.detection.tokenization import TokenRole


REPO_ROOT = Path(__file__).resolve().parents[2]
SLOT_NAMES: tuple[str, str, str, str] = ("x1", "y1", "x2", "y2")
DEFAULT_RADII: tuple[int, ...] = (0, 1, 2, 4, 8, 16, 32, 64)
DEFAULT_EMBEDDING_RADII: tuple[int, ...] = (1, 2, 4, 8, 16)
COORD_TOKEN_RE = re.compile(r"<\|coord_(\d{1,3})\|>")
IM_END_TOKEN = "<|im_end|>"


@dataclass(frozen=True)
class CoordVocab:
    """Resolved coordinate-token vocabulary."""

    coord_token_ids: tuple[int, ...]
    wildcard_token_id: int | None = None


@dataclass(frozen=True)
class ProbabilityLocalityMetrics:
    """Shape metrics for a conditional distribution over coordinate bins."""

    p_gt_cond: float
    rank_gt: int
    top1_bin: int
    top1_distance: int
    expected_bin: float
    expected_abs_error: float
    entropy: float
    normalized_entropy: float
    effective_support: float
    peak_probability: float
    mass_by_radius: dict[str, float]
    local_maxima_count: int
    secondary_peak_bin: int | None
    secondary_peak_distance: int | None
    top2_top1_ratio: float | None
    local_distance_correlation: float
    gaussian_kl: float | None
    gaussian_scale: float | None
    laplacian_kl: float | None
    laplacian_scale: float | None
    shape_label: str


@dataclass(frozen=True)
class DistributionMetrics:
    """Full-vocab and conditional-coordinate distribution metrics."""

    coord_vocab_mass: float
    p_gt_full: float
    p_gt_cond: float
    rank_gt: int
    top1_bin: int
    top1_distance: int
    expected_bin: float
    expected_abs_error: float
    entropy: float
    normalized_entropy: float
    effective_support: float
    peak_probability: float
    mass_by_radius: dict[str, float]
    local_maxima_count: int
    secondary_peak_bin: int | None
    secondary_peak_distance: int | None
    top2_top1_ratio: float | None
    local_distance_correlation: float
    gaussian_kl: float | None
    gaussian_scale: float | None
    laplacian_kl: float | None
    laplacian_scale: float | None
    shape_label: str
    top_bins: list[dict[str, float | int]]
    conditional_probs: np.ndarray = field(repr=False, compare=False)
    coord_logits: np.ndarray = field(repr=False, compare=False)


@dataclass(frozen=True)
class LaneCPrefixMatchState:
    """Guarded Lane-A match state consumed by Lane-C target selection."""

    depth: int
    gt_count: int
    match_policy: str
    consumed_raw_pred_indices: tuple[int, ...]
    matched_prefix_gt_indices: tuple[int, ...]
    remaining_gt_indices: tuple[int, ...]
    fp_prefix_object_indices: tuple[int, ...]
    duplicate_prefix_object_indices: tuple[int, ...]
    invalid_prefix_object_indices: tuple[int, ...]
    ambiguous_prefix_object_indices: tuple[int, ...]
    prefix_quality: str


@dataclass(frozen=True)
class LaneCTargetSelection:
    """Selected Lane-C intended target and rule provenance."""

    intended_target_gt_idx: int | None
    target_selection_rule: str


@dataclass(frozen=True)
class LaneCCoordAttribution:
    """Attribution metrics for one Lane-C coordinate distribution."""

    target_bin: int
    top1_bin: int
    target_rank: int
    best_other_gt_idx: int | None
    best_other_gt_bin: int | None
    best_other_gt_rank: int | None
    target_margin_vs_best_other: float | None
    entropy: float
    gt_top1: bool
    top1_distance: int
    mass_by_radius: dict[str, float]
    top_peak_attribution: str


@dataclass(frozen=True)
class EmbeddingGeometryMetrics:
    """Numeric-manifold diagnostics for coordinate-token row vectors."""

    pearson_distance_numeric: float
    spearman_distance_numeric: float
    knn_radius_recall: dict[str, float]
    numeric_neighbor_rank_mean: dict[str, float]
    bandedness_ratio: float
    local_step_mean: float
    local_step_std: float
    second_difference_mean: float
    boundary_step_mean: float
    directionality_r2: float
    directionality_rank_correlation: float
    contiguous_cluster_purity: float


@dataclass(frozen=True)
class CompactSlot:
    """One coordinate slot parsed from a compact-full assistant row."""

    object_order_index: int
    desc: str
    slot: str
    gt_bin: int
    bbox_xyxy: tuple[int, int, int, int]


@dataclass(frozen=True)
class StudyPaths:
    """Filesystem inputs and outputs for one locality study."""

    checkpoint: Path
    resolved_config: Path
    source_config: Path | None
    dataset_jsonl: Path
    image_root: Path
    artifact_root: Path
    self_rollout_root: Path | None
    self_rollout_regen_config: Path | None
    base_model_for_embedding_control: Path | None = None
    lane_a_rollout_root: Path | None = None


@dataclass(frozen=True)
class StudyModelConfig:
    """Model/prompt settings shared by heavy stages."""

    prompt_variant: str = "coco_80"
    object_field_order: str = "desc_first"
    bbox_format: str = "xyxy"
    detection_sequence_format: str = "compact_full"
    object_ordering: str = "random_permutation"
    seed: int = 42
    teacher_target_kind_mode: str = "recursive_detection"
    device: str = "auto"
    attn_implementation: str = "auto"
    torch_dtype: str = "bfloat16"


@dataclass(frozen=True)
class StudyExecutionConfig:
    """Execution bounds for staged extraction."""

    sample_limit: int = 200
    batch_size: int = 1
    top_k: int = 12
    radii: tuple[int, ...] = DEFAULT_RADII
    embedding_neighbor_k: int = 16
    embedding_radii: tuple[int, ...] = DEFAULT_EMBEDDING_RADII
    max_self_prefixes_per_image: int = 8
    save_raw_arrays: bool = True
    run_infer_if_missing: bool = False


@dataclass(frozen=True)
class StudyConfig:
    """Parsed YAML configuration for the hard-CE locality study."""

    paths: StudyPaths
    model: StudyModelConfig
    execution: StudyExecutionConfig


@dataclass(frozen=True)
class PreparedForwardExample:
    """One fully rendered text/image input with coordinate slot metadata."""

    row_index: int
    image_id: int | None
    file_name: str
    image_path: Path
    width: int
    height: int
    assistant_text: str
    full_text: str
    full_input_ids: tuple[int, ...]
    coord_slots: tuple[CompactSlot, ...]
    assistant_coord_positions: tuple[int, ...]
    target_kinds: tuple[str, ...]
    target_roles: tuple[str, ...]
    prefix_condition: str
    prefix_depth: int
    prefix_quality: str
    scope_label: str
    pairing_id: str | None = None
    lane_c_metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class ModelHandle:
    """Loaded model, processor, tokenizer, and checkpoint metadata."""

    model: Any
    processor: Any
    tokenizer: Any
    resolved_checkpoint: Any
    token_embeddings_adapter: Any | None
    base_embedding_rows: np.ndarray | None = None
    base_output_rows: np.ndarray | None = None


def resolve_coord_token_ids(tokenizer: Any) -> CoordVocab:
    """Resolve exactly `<|coord_0|>` through `<|coord_999|>` token ids."""

    # validating canonical coordinate ids
    coord_ids = tuple(int(token_id) for token_id in get_coord_token_ids(tokenizer, validate=True))
    if len(coord_ids) != 1000 or len(set(coord_ids)) != 1000:
        raise ValueError("coordinate token id lookup must yield 1000 distinct ids")

    # guarding the wildcard helper token
    wildcard_raw = tokenizer.convert_tokens_to_ids("<|coord_*|>")
    wildcard_id = None if wildcard_raw is None else int(wildcard_raw)
    if wildcard_id is not None and wildcard_id in set(coord_ids):
        raise ValueError("<|coord_*|> wildcard token id collides with coordinate bins")

    return CoordVocab(coord_token_ids=coord_ids, wildcard_token_id=wildcard_id)


def prediction_position_for_label_position(label_position: int) -> int:
    """Return the causal logit row that predicts a teacher token."""

    position = int(label_position)
    if position <= 0:
        raise ValueError("label position 0 has no previous-token prediction row")
    return position - 1


def distribution_metrics_from_logits(
    *,
    logits: torch.Tensor,
    coord_token_ids: Sequence[int],
    gt_bin: int,
    radii: Sequence[int] = DEFAULT_RADII,
    top_k: int = 12,
) -> DistributionMetrics:
    """Compute full-vocab mass and conditional coordinate-shape metrics."""

    # selecting coordinate logits
    if logits.ndim != 1:
        raise ValueError("logits must be a one-dimensional vocab row")
    coord_index = torch.tensor(
        [int(token_id) for token_id in coord_token_ids],
        dtype=torch.long,
        device=logits.device,
    )
    coord_logits_t = logits.index_select(0, coord_index).float()

    # normalizing full-vocab and conditional views
    log_norm = torch.logsumexp(logits.float(), dim=0)
    coord_log_probs_full = coord_logits_t - log_norm
    coord_probs_full = torch.exp(coord_log_probs_full)
    coord_vocab_mass = float(coord_probs_full.sum().detach().cpu().item())
    if coord_vocab_mass <= 0.0 or not math.isfinite(coord_vocab_mass):
        p_cond = np.full(len(coord_token_ids), 1.0 / len(coord_token_ids), dtype=np.float64)
    else:
        p_cond = (coord_probs_full / coord_probs_full.sum()).detach().cpu().numpy().astype(np.float64)

    # summarizing shape
    locality = compute_probability_locality_metrics(
        p_cond,
        gt_bin=int(gt_bin),
        radii=tuple(int(radius) for radius in radii),
    )
    top_count = min(int(top_k), int(p_cond.shape[0]))
    top_indices = np.argsort(-p_cond, kind="stable")[:top_count]
    top_bins = [
        {
            "bin": int(index),
            "distance": int(abs(int(index) - int(gt_bin))),
            "prob_cond": float(p_cond[index]),
            "prob_full": float(coord_probs_full.detach().cpu().numpy()[index]),
            "logit": float(coord_logits_t.detach().cpu().numpy()[index]),
        }
        for index in top_indices
    ]

    return DistributionMetrics(
        coord_vocab_mass=coord_vocab_mass,
        p_gt_full=float(coord_probs_full[int(gt_bin)].detach().cpu().item()),
        p_gt_cond=locality.p_gt_cond,
        rank_gt=locality.rank_gt,
        top1_bin=locality.top1_bin,
        top1_distance=locality.top1_distance,
        expected_bin=locality.expected_bin,
        expected_abs_error=locality.expected_abs_error,
        entropy=locality.entropy,
        normalized_entropy=locality.normalized_entropy,
        effective_support=locality.effective_support,
        peak_probability=locality.peak_probability,
        mass_by_radius=locality.mass_by_radius,
        local_maxima_count=locality.local_maxima_count,
        secondary_peak_bin=locality.secondary_peak_bin,
        secondary_peak_distance=locality.secondary_peak_distance,
        top2_top1_ratio=locality.top2_top1_ratio,
        local_distance_correlation=locality.local_distance_correlation,
        gaussian_kl=locality.gaussian_kl,
        gaussian_scale=locality.gaussian_scale,
        laplacian_kl=locality.laplacian_kl,
        laplacian_scale=locality.laplacian_scale,
        shape_label=locality.shape_label,
        top_bins=top_bins,
        conditional_probs=p_cond,
        coord_logits=coord_logits_t.detach().cpu().numpy().astype(np.float32),
    )


def build_lane_c_prefix_match_state(
    lane_a_rows: Sequence[Mapping[str, Any]],
    depth: int,
    gt_count: int,
    *,
    match_policy: str = "guarded",
) -> LaneCPrefixMatchState:
    """Build guarded prefix state from Lane-A per-row rollout anatomy records.

    ``depth`` is a raw generated-object prefix length: only rows whose
    ``raw_pred_idx`` is less than ``depth`` are consumed. Under the default
    guarded policy, duplicate-suppressed raw TPs do not consume GT coverage.
    """

    policy = str(match_policy or "guarded").lower()
    if policy != "guarded":
        raise ValueError(f"unsupported Lane-C match_policy {match_policy!r}")
    depth_i = max(0, int(depth))
    gt_count_i = max(0, int(gt_count))
    prefix_rows = sorted(
        (
            row
            for row in lane_a_rows
            if _lane_c_optional_int(row.get("raw_pred_idx")) is not None
            and int(row["raw_pred_idx"]) < depth_i
        ),
        key=lambda row: int(row["raw_pred_idx"]),
    )

    matched_gt: set[int] = set()
    consumed_indices: list[int] = []
    fp_indices: list[int] = []
    duplicate_indices: list[int] = []
    invalid_indices: list[int] = []
    ambiguous_indices: list[int] = []

    for row in prefix_rows:
        raw_idx = int(row["raw_pred_idx"])
        consumed_indices.append(raw_idx)
        classes = _lane_c_prefix_problem_classes(row)
        if "fp" in classes:
            fp_indices.append(raw_idx)
        if "duplicate" in classes:
            duplicate_indices.append(raw_idx)
        if "invalid" in classes:
            invalid_indices.append(raw_idx)
        if "ambiguous" in classes:
            ambiguous_indices.append(raw_idx)

        guarded_gt_idx = _lane_c_optional_int(row.get("guarded_matched_gt_idx"))
        if (
            guarded_gt_idx is not None
            and 0 <= guarded_gt_idx < gt_count_i
            and not bool(row.get("suppressed_by_guard", False))
            and _lane_c_is_guarded_tp_like(row)
        ):
            matched_gt.add(int(guarded_gt_idx))

    problem_names = {
        name
        for name, indices in (
            ("fp", fp_indices),
            ("duplicate", duplicate_indices),
            ("invalid", invalid_indices),
            ("ambiguous", ambiguous_indices),
        )
        if indices
    }
    prefix_quality = _lane_c_prefix_quality(depth_i, problem_names)
    matched_sorted = tuple(sorted(matched_gt))
    return LaneCPrefixMatchState(
        depth=depth_i,
        gt_count=gt_count_i,
        match_policy=policy,
        consumed_raw_pred_indices=tuple(consumed_indices),
        matched_prefix_gt_indices=matched_sorted,
        remaining_gt_indices=tuple(idx for idx in range(gt_count_i) if idx not in matched_gt),
        fp_prefix_object_indices=tuple(fp_indices),
        duplicate_prefix_object_indices=tuple(duplicate_indices),
        invalid_prefix_object_indices=tuple(invalid_indices),
        ambiguous_prefix_object_indices=tuple(ambiguous_indices),
        prefix_quality=prefix_quality,
    )


def select_lane_c_intended_target_gt_idx(
    teacher_order_gt_indices: Sequence[int],
    prefix_state: LaneCPrefixMatchState | Mapping[str, Any],
) -> LaneCTargetSelection:
    """Select the first teacher-order GT index still remaining after a prefix."""

    remaining = set(
        int(item)
        for item in _lane_c_state_value(prefix_state, "remaining_gt_indices", ())
        if item is not None
    )
    for raw_idx in teacher_order_gt_indices:
        gt_idx = int(raw_idx)
        if gt_idx in remaining:
            return LaneCTargetSelection(
                intended_target_gt_idx=gt_idx,
                target_selection_rule="first_remaining_teacher_order_guarded",
            )
    return LaneCTargetSelection(
        intended_target_gt_idx=None,
        target_selection_rule="no_remaining_gt",
    )


def attribute_lane_c_coord_distribution(
    probs: Sequence[float] | np.ndarray,
    *,
    target_bin: int,
    gt_bins_by_index: Mapping[int, Any],
    prefix_bins_by_label: Mapping[str, Any] | None,
    slot: str,
    radii: Sequence[int] = (4, 8),
) -> LaneCCoordAttribution:
    """Attribute one full 1000-bin coordinate distribution to local objects."""

    p = _lane_c_normalized_probs(probs)
    target = int(target_bin)
    if target < 0 or target >= p.shape[0]:
        raise ValueError("target_bin must be in the 0..999 range")
    order = np.argsort(-p, kind="stable")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(order) + 1)
    top1_bin = int(order[0])

    radius_values = tuple(int(radius) for radius in radii)
    mass_by_radius = {
        f"mass_at_radius_{radius}": float(
            p[np.abs(np.arange(1000, dtype=np.int64) - target) <= radius].sum()
        )
        for radius in radius_values
    }
    entropy = float(-np.sum(p * np.log(np.clip(p, 1e-300, 1.0))))

    other_gt_candidates: list[tuple[int, int, int, float]] = []
    for raw_gt_idx, raw_value in gt_bins_by_index.items():
        gt_idx = int(raw_gt_idx)
        candidate_bin = _lane_c_extract_slot_bin(raw_value, slot=slot)
        if candidate_bin is None or candidate_bin == target:
            continue
        other_gt_candidates.append((int(ranks[candidate_bin]), gt_idx, candidate_bin, float(p[candidate_bin])))
    other_gt_candidates.sort(key=lambda item: (item[0], item[1]))
    if other_gt_candidates:
        best_other_rank, best_other_idx, best_other_bin, best_other_prob = other_gt_candidates[0]
        margin = float(p[target] - best_other_prob)
    else:
        best_other_rank = None
        best_other_idx = None
        best_other_bin = None
        margin = None

    return LaneCCoordAttribution(
        target_bin=target,
        top1_bin=top1_bin,
        target_rank=int(ranks[target]),
        best_other_gt_idx=best_other_idx,
        best_other_gt_bin=best_other_bin,
        best_other_gt_rank=best_other_rank,
        target_margin_vs_best_other=margin,
        entropy=entropy,
        gt_top1=bool(top1_bin == target),
        top1_distance=int(abs(top1_bin - target)),
        mass_by_radius=mass_by_radius,
        top_peak_attribution=_lane_c_top_peak_attribution(
            top1_bin=top1_bin,
            target_bin=target,
            gt_bins_by_index=gt_bins_by_index,
            prefix_bins_by_label=prefix_bins_by_label or {},
            slot=slot,
            local_radius=max(radius_values) if radius_values else 8,
        ),
    )


def lane_c_shard_label(shard_index: int, num_shards: int) -> str:
    """Return the stable Lane-C shard label."""

    return f"lane_c_shard_{int(shard_index):03d}-of-{int(num_shards):03d}"


def normalize_lane_c_shard(
    *,
    shard_index: int | None,
    num_shards: int | None,
) -> tuple[int | None, int | None, str | None]:
    """Validate optional Lane-C sharding parameters."""

    if shard_index is None and num_shards is None:
        return None, None, None
    if shard_index is None or num_shards is None:
        raise ValueError("Lane-C sharding requires both shard_index and num_shards")
    shard_count = int(num_shards)
    shard_i = int(shard_index)
    if shard_count <= 0:
        raise ValueError(f"num_shards must be positive, got {shard_count}")
    if not 0 <= shard_i < shard_count:
        raise ValueError(
            "shard_index must satisfy 0 <= shard_index < num_shards, got "
            f"shard_index={shard_i} num_shards={shard_count}"
        )
    return shard_i, shard_count, lane_c_shard_label(shard_i, shard_count)


def lane_c_record_selected(
    record: Mapping[str, Any] | int,
    *,
    limit: int | None = None,
    shard_index: int | None = None,
    num_shards: int | None = None,
) -> bool:
    """Select Lane-C records by stable source line so all prefix depths stay together."""

    source_line_idx = int(record.get("source_line_idx")) if isinstance(record, Mapping) else int(record)
    if source_line_idx < 0:
        return False
    if limit is not None and source_line_idx >= int(limit):
        return False
    normalized_shard_index, normalized_num_shards, _ = normalize_lane_c_shard(
        shard_index=shard_index,
        num_shards=num_shards,
    )
    if normalized_shard_index is None or normalized_num_shards is None:
        return True
    return source_line_idx % normalized_num_shards == normalized_shard_index


def lane_c_merge_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    """Stable merge key for Lane-C per-slot/per-case rows."""

    return (
        _lane_c_optional_int(record.get("source_line_idx")),
        str(record.get("prefix_mode", record.get("prefix_condition", ""))),
        _lane_c_optional_int(record.get("prefix_depth")),
        _lane_c_optional_int(record.get("intended_target_gt_idx")),
        str(record.get("slot", "")),
        _lane_c_optional_int(record.get("slot_index")),
    )


def select_lane_c_generated_intended_target(
    *,
    teacher_order_gt_indices: Sequence[int],
    lane_a_rows: Sequence[Mapping[str, Any]],
    depth: int,
    gt_count: int,
    match_policy: str = "guarded",
) -> LaneCTargetSelection:
    """Select a generated-prefix target from Lane-A state, never ordinal depth."""

    state = build_lane_c_prefix_match_state(
        lane_a_rows,
        depth=int(depth),
        gt_count=int(gt_count),
        match_policy=match_policy,
    )
    return select_lane_c_intended_target_gt_idx(teacher_order_gt_indices, state)


def build_lane_c_shard_plan(
    *,
    config_path: str | Path,
    output_root: str | Path,
    num_shards: int,
    python_executable: str = "python",
) -> dict[str, Any]:
    """Build CPU-only Lane-C shard and merge commands without loading a model."""

    shard_count = int(num_shards)
    if shard_count <= 0:
        raise ValueError("num_shards must be positive")
    config = Path(config_path)
    root = Path(output_root)
    script = REPO_ROOT / "scripts" / "analysis" / "run_hard_ce_coord_logit_locality.py"
    shards: list[dict[str, str | int]] = []
    for shard_index in range(shard_count):
        label = lane_c_shard_label(shard_index, shard_count)
        shard_dir = root / "shards" / label
        command = " ".join(
            [
                shlex.quote(str(python_executable)),
                shlex.quote(str(script)),
                "--config",
                shlex.quote(str(config)),
                "--stages",
                "x1_basin_attribution",
                "--shard-index",
                str(shard_index),
                "--num-shards",
                str(shard_count),
            ]
        )
        shards.append(
            {
                "shard_index": shard_index,
                "num_shards": shard_count,
                "shard_label": label,
                "shard_dir": str(shard_dir),
                "command": command,
            }
        )
    merge_command = " ".join(
        [
            shlex.quote(str(python_executable)),
            shlex.quote(str(script)),
            "--config",
            shlex.quote(str(config)),
            "--stages",
            "x1_basin_attribution",
            "--merge-shards",
            "--num-shards",
            str(shard_count),
        ]
    )
    return {
        "stage": "x1_basin_attribution",
        "output_root": str(root),
        "num_shards": shard_count,
        "shards": shards,
        "merge_command": merge_command,
    }


def summarize_lane_c_per_case_rows(
    per_slot_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize Lane-C slot rows while preserving x1/y1/x2/y2 separately."""

    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in per_slot_rows:
        key = (
            _lane_c_optional_int(row.get("source_line_idx")),
            str(row.get("prefix_mode", row.get("prefix_condition", ""))),
            _lane_c_optional_int(row.get("prefix_depth")),
            _lane_c_optional_int(row.get("intended_target_gt_idx")),
        )
        grouped[key].append(row)

    cases: list[dict[str, Any]] = []
    metric_keys = (
        "top_peak_attribution",
        "target_rank",
        "best_other_gt_rank",
        "target_margin_vs_best_other",
        "gt_top1",
        "top1_distance",
        "mass_at_radius_4",
        "mass_at_radius_8",
    )
    for key, rows in sorted(grouped.items(), key=lambda item: item[0]):
        exemplar = dict(rows[0])
        slot_rows = {
            str(row.get("slot")): row
            for row in sorted(
                rows,
                key=lambda item: SLOT_NAMES.index(str(item.get("slot")))
                if str(item.get("slot")) in SLOT_NAMES
                else 99,
            )
            if str(row.get("slot")) in SLOT_NAMES
        }
        slots = {
            slot: {metric_key: slot_rows[slot].get(metric_key) for metric_key in metric_keys}
            for slot in SLOT_NAMES
            if slot in slot_rows
        }
        case = {
            "lane_c_case_key": ":".join("" if part is None else str(part) for part in key),
            "case_id": exemplar.get("case_id"),
            "source_line_idx": key[0],
            "prefix_mode": key[1],
            "prefix_depth": key[2],
            "intended_target_gt_idx": key[3],
            "prefix_quality": exemplar.get("prefix_quality"),
            "match_policy": exemplar.get("match_policy"),
            "target_selection_rule": exemplar.get("target_selection_rule"),
            "target_object_instance_id": exemplar.get("target_object_instance_id"),
            "target_source_object_index": exemplar.get("target_source_object_index"),
            "target_desc": exemplar.get("target_desc"),
            "target_bbox_xyxy": exemplar.get("target_bbox_xyxy"),
            "slots": slots,
        }
        if "x1" in slots:
            case["x1"] = slots["x1"]
        cases.append(case)
    return cases


def merge_lane_c_shards(
    output_root: str | Path,
    *,
    expected_shards: int,
) -> dict[str, Any]:
    """Merge Lane-C shard outputs with strict stale/missing/duplicate guards."""

    root = Path(output_root)
    shard_count = int(expected_shards)
    if shard_count <= 0:
        raise ValueError("expected_shards must be positive")
    shards_dir = root / "shards"
    if not shards_dir.exists():
        raise FileNotFoundError(f"Lane-C shards directory not found: {shards_dir}")

    shard_pattern = re.compile(r"^lane_c_shard_(\d{3})-of-(\d{3})$")
    found_dirs = sorted(path for path in shards_dir.iterdir() if path.is_dir())
    malformed = [path.name for path in found_dirs if shard_pattern.fullmatch(path.name) is None]
    if malformed:
        raise ValueError(f"malformed Lane-C shard dirs: {malformed}")

    expected_names = {lane_c_shard_label(index, shard_count) for index in range(shard_count)}
    found_names = {path.name for path in found_dirs}
    unexpected = sorted(found_names - expected_names)
    if unexpected:
        raise ValueError(f"unexpected Lane-C shard dirs: {unexpected}")
    missing = sorted(expected_names - found_names)
    if missing:
        raise ValueError(f"missing Lane-C shard dirs: {missing}")

    per_slot_rows: list[dict[str, Any]] = []
    per_case_rows: list[dict[str, Any]] = []
    shard_summaries: list[dict[str, Any]] = []
    seen_slot_keys: dict[str, str] = {}
    for shard_name in sorted(expected_names):
        shard_dir = shards_dir / shard_name
        per_slot_path = shard_dir / "per_slot.jsonl"
        per_case_path = shard_dir / "per_case.jsonl"
        summary_path = shard_dir / "summary.json"
        missing_files = [
            str(path)
            for path in (per_slot_path, per_case_path, summary_path)
            if not path.exists()
        ]
        if missing_files:
            raise FileNotFoundError(f"Lane-C shard {shard_name} missing files: {missing_files}")
        for row in _read_jsonl(per_slot_path):
            key = str(row.get("lane_c_merge_key") or lane_c_merge_key(row))
            if key in seen_slot_keys:
                raise ValueError(
                    "duplicate Lane-C per-slot merge key "
                    f"{key!r} in {shard_name} and {seen_slot_keys[key]}"
                )
            seen_slot_keys[key] = shard_name
            row["lane_c_merge_key"] = key
            per_slot_rows.append(row)
        per_case_rows.extend(_read_jsonl(per_case_path))
        shard_summaries.append(json.loads(summary_path.read_text(encoding="utf-8") or "{}"))

    root.mkdir(parents=True, exist_ok=True)
    with (root / "per_slot.jsonl").open("w", encoding="utf-8") as handle:
        for row in sorted(per_slot_rows, key=lane_c_merge_key):
            handle.write(json.dumps(_jsonable(row), ensure_ascii=True) + "\n")
    if not per_case_rows:
        per_case_rows = summarize_lane_c_per_case_rows(per_slot_rows)
    with (root / "per_case.jsonl").open("w", encoding="utf-8") as handle:
        for row in per_case_rows:
            handle.write(json.dumps(_jsonable(row), ensure_ascii=True) + "\n")

    summary = _lane_c_summary(
        per_slot_rows,
        per_case_rows=per_case_rows,
        shard_metadata={"mode": "merge", "expected_shards": shard_count},
    )
    skipped_counts: Counter[str] = Counter()
    selected_record_count = 0
    planned_example_count = 0
    for shard_summary in shard_summaries:
        selected_record_count += int(shard_summary.get("selected_record_count") or 0)
        planned_example_count += int(shard_summary.get("planned_example_count") or 0)
        skipped_counts.update(
            {
                str(key): int(value)
                for key, value in (shard_summary.get("skipped_counts") or {}).items()
            }
        )
    summary["selected_record_count"] = selected_record_count
    summary["planned_example_count"] = planned_example_count
    summary["skipped_counts"] = dict(sorted(skipped_counts.items()))
    summary["shard_summaries"] = shard_summaries
    merge_summary = {
        "stage": "x1_basin_attribution",
        "output_root": str(root),
        "expected_shards": shard_count,
        "merged_shards": sorted(expected_names),
        "row_count": len(per_slot_rows),
        "case_count": len(per_case_rows),
    }
    _write_json(root / "summary.json", summary)
    _write_json(root / "merge_summary.json", merge_summary)
    return merge_summary


def compute_probability_locality_metrics(
    probabilities: Sequence[float] | np.ndarray,
    *,
    gt_bin: int,
    radii: Sequence[int] = DEFAULT_RADII,
) -> ProbabilityLocalityMetrics:
    """Compute locality, entropy, fit, and modality metrics for coord bins."""

    # normalizing the distribution
    probs = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if probs.shape[0] != 1000:
        raise ValueError(f"coordinate probability vector must have 1000 bins; got {probs.shape[0]}")
    if np.any(probs < 0.0):
        raise ValueError("coordinate probabilities must be non-negative")
    total = float(np.sum(probs))
    if total <= 0.0 or not math.isfinite(total):
        probs = np.full(1000, 0.001, dtype=np.float64)
    else:
        probs = probs / total
    gt = int(gt_bin)
    if gt < 0 or gt >= probs.shape[0]:
        raise ValueError("gt_bin must be in the 0..999 range")

    # computing rank and radial mass
    bins = np.arange(probs.shape[0], dtype=np.float64)
    distances = np.abs(bins - float(gt))
    order = np.argsort(-probs, kind="stable")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(order) + 1)
    top1_bin = int(order[0])
    mass_by_radius = {
        f"mass_at_{int(radius)}": float(probs[np.abs(np.arange(1000) - gt) <= int(radius)].sum())
        for radius in radii
    }

    # measuring entropy and expectation
    entropy = float(-np.sum(probs * np.log(np.clip(probs, 1e-300, 1.0))))
    expected_bin = float(np.sum(probs * bins))
    expected_abs_error = float(np.sum(probs * distances))
    peak_probability = float(probs[top1_bin])
    effective_support = float(math.exp(entropy))

    # counting modality after light smoothing
    smoothed = _smooth_probs(probs)
    local_maxima = _local_maxima(smoothed, min_prominence=max(1e-6, peak_probability * 0.01))
    secondary_peak_bin = int(order[1]) if len(order) > 1 else None
    secondary_peak_distance = (
        int(abs(secondary_peak_bin - gt)) if secondary_peak_bin is not None else None
    )
    top2_top1_ratio = (
        float(probs[order[1]] / max(probs[order[0]], 1e-300)) if len(order) > 1 else None
    )

    # fitting simple descriptive target-centered families
    gaussian_scale, gaussian_kl = _fit_centered_family(
        probs,
        gt_bin=gt,
        family="gaussian",
    )
    laplacian_scale, laplacian_kl = _fit_centered_family(
        probs,
        gt_bin=gt,
        family="laplacian",
    )

    return ProbabilityLocalityMetrics(
        p_gt_cond=float(probs[gt]),
        rank_gt=int(ranks[gt]),
        top1_bin=top1_bin,
        top1_distance=int(abs(top1_bin - gt)),
        expected_bin=expected_bin,
        expected_abs_error=expected_abs_error,
        entropy=entropy,
        normalized_entropy=float(entropy / math.log(1000.0)),
        effective_support=effective_support,
        peak_probability=peak_probability,
        mass_by_radius=mass_by_radius,
        local_maxima_count=int(len(local_maxima)),
        secondary_peak_bin=secondary_peak_bin,
        secondary_peak_distance=secondary_peak_distance,
        top2_top1_ratio=top2_top1_ratio,
        local_distance_correlation=_safe_corr(-distances, probs, method="pearson"),
        gaussian_kl=gaussian_kl,
        gaussian_scale=gaussian_scale,
        laplacian_kl=laplacian_kl,
        laplacian_scale=laplacian_scale,
        shape_label=_classify_probability_shape(
            probs,
            gt_bin=gt,
            local_maxima_count=len(local_maxima),
            top1_distance=abs(top1_bin - gt),
            effective_support=effective_support,
            top2_top1_ratio=top2_top1_ratio,
            secondary_peak_distance=secondary_peak_distance,
        ),
    )


def compute_embedding_geometry_metrics(
    vectors: np.ndarray | Sequence[Sequence[float]],
    *,
    neighbor_k: int = 16,
    radii: Sequence[int] = DEFAULT_EMBEDDING_RADII,
) -> EmbeddingGeometryMetrics:
    """Measure whether coordinate rows form a numeric adjacency manifold."""

    # validating row matrix
    matrix = np.asarray(vectors, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("embedding vectors must be a two-dimensional matrix")
    if matrix.shape[0] < 3:
        raise ValueError("at least three coordinate rows are required")
    row_count = int(matrix.shape[0])
    k = max(1, min(int(neighbor_k), row_count - 1))

    # computing pairwise distances
    euclidean = _pairwise_euclidean(matrix)
    numeric_distance = np.abs(
        np.arange(row_count)[:, None] - np.arange(row_count)[None, :]
    ).astype(np.float64)
    mask = np.triu(np.ones((row_count, row_count), dtype=bool), k=1)
    dist_flat = euclidean[mask]
    numeric_flat = numeric_distance[mask]

    # measuring nearest-neighbor numeric locality
    nearest = np.argsort(euclidean, axis=1, kind="stable")[:, 1 : k + 1]
    knn_radius_recall = {}
    for radius in radii:
        radius_i = int(radius)
        hits = [
            float(np.mean(np.abs(nearest[row] - row) <= radius_i))
            for row in range(row_count)
        ]
        knn_radius_recall[f"radius_{radius_i}"] = float(np.mean(hits))
    numeric_neighbor_rank_mean = {
        f"offset_{offset}": _mean_numeric_neighbor_rank(euclidean, offset=offset)
        for offset in (1, 2, 4)
        if row_count > offset
    }

    # measuring smoothness and recoverability
    local_steps = np.linalg.norm(np.diff(matrix, axis=0), axis=1)
    second_diff = np.linalg.norm(matrix[2:] - 2.0 * matrix[1:-1] + matrix[:-2], axis=1)
    boundary_steps = np.concatenate([local_steps[: min(16, len(local_steps))], local_steps[-min(16, len(local_steps)) :]])
    y = np.arange(row_count, dtype=np.float64)
    directionality_r2 = _linear_regression_r2(matrix, y)
    projection = _first_pc_projection(matrix)

    return EmbeddingGeometryMetrics(
        pearson_distance_numeric=_safe_corr(dist_flat, numeric_flat, method="pearson"),
        spearman_distance_numeric=_safe_corr(dist_flat, numeric_flat, method="spearman"),
        knn_radius_recall=knn_radius_recall,
        numeric_neighbor_rank_mean=numeric_neighbor_rank_mean,
        bandedness_ratio=_bandedness_ratio(euclidean),
        local_step_mean=float(np.mean(local_steps)),
        local_step_std=float(np.std(local_steps)),
        second_difference_mean=float(np.mean(second_diff)) if len(second_diff) else 0.0,
        boundary_step_mean=float(np.mean(boundary_steps)),
        directionality_r2=directionality_r2,
        directionality_rank_correlation=_safe_corr(projection, y, method="spearman"),
        contiguous_cluster_purity=_contiguous_cluster_purity(matrix),
    )


def cut_complete_compact_rows(text: str, *, max_rows: int | None = None) -> str:
    """Return only complete compact-full rows from generated text."""

    # removing chat stop markers before row validation
    cleaned = str(text).replace(IM_END_TOKEN, "")
    rows: list[str] = []
    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = parse_compact_row(
            line,
            require_object_ref_marker=True,
            require_bbox_start_marker=True,
        )
        if parts is None or len(parts.bbox_tokens) != 4:
            continue
        rows.append(line)
        if max_rows is not None and len(rows) >= int(max_rows):
            break
    return "\n".join(rows)


def compact_slots_from_text(text: str) -> tuple[CompactSlot, ...]:
    """Parse compact-full assistant text into xyxy coordinate slot records."""

    # parsing complete rows in rendered order
    slots: list[CompactSlot] = []
    for object_index, raw_line in enumerate(str(text).splitlines() or [str(text)]):
        line = raw_line.strip()
        if not line:
            continue
        parts = parse_compact_row(
            line,
            require_object_ref_marker=True,
            require_bbox_start_marker=True,
        )
        if parts is None or len(parts.bbox_tokens) != 4:
            continue
        bins = tuple(_coord_bin_from_token(token) for token in parts.bbox_tokens)
        for slot_name, value in zip(SLOT_NAMES, bins, strict=True):
            slots.append(
                CompactSlot(
                    object_order_index=int(object_index),
                    desc=str(parts.desc),
                    slot=str(slot_name),
                    gt_bin=int(value),
                    bbox_xyxy=tuple(int(item) for item in bins),
                )
            )
    return tuple(slots)


def load_study_config(config_path: str | Path) -> StudyConfig:
    """Load a YAML config for the locality study."""

    # reading top-level mappings
    path = Path(config_path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a YAML mapping")
    paths_raw = _required_mapping(payload, "paths")
    model_raw = dict(payload.get("model") or {})
    execution_raw = dict(payload.get("execution") or {})

    paths = StudyPaths(
        checkpoint=_resolve_repo_path(_required_str(paths_raw, "checkpoint")),
        resolved_config=_resolve_repo_path(_required_str(paths_raw, "resolved_config")),
        source_config=_optional_path(paths_raw.get("source_config")),
        dataset_jsonl=_resolve_repo_path(_required_str(paths_raw, "dataset_jsonl")),
        image_root=_resolve_repo_path(_required_str(paths_raw, "image_root")),
        artifact_root=_resolve_repo_path(_required_str(paths_raw, "artifact_root")),
        self_rollout_root=_optional_path(paths_raw.get("self_rollout_root")),
        self_rollout_regen_config=_optional_path(paths_raw.get("self_rollout_regen_config")),
        base_model_for_embedding_control=_optional_path(paths_raw.get("base_model_for_embedding_control")),
        lane_a_rollout_root=_optional_path(paths_raw.get("lane_a_rollout_root")),
    )
    model = StudyModelConfig(
        prompt_variant=str(model_raw.get("prompt_variant", "coco_80")),
        object_field_order=str(model_raw.get("object_field_order", "desc_first")),
        bbox_format=str(model_raw.get("bbox_format", "xyxy")),
        detection_sequence_format=str(model_raw.get("detection_sequence_format", "compact_full")),
        object_ordering=str(model_raw.get("object_ordering", "random_permutation")),
        seed=int(model_raw.get("seed", 42)),
        teacher_target_kind_mode=str(model_raw.get("teacher_target_kind_mode", "recursive_detection")),
        device=str(model_raw.get("device", "auto")),
        attn_implementation=str(model_raw.get("attn_implementation", "auto")),
        torch_dtype=str(model_raw.get("torch_dtype", "bfloat16")),
    )
    execution = StudyExecutionConfig(
        sample_limit=int(execution_raw.get("sample_limit", 200)),
        batch_size=max(1, int(execution_raw.get("batch_size", 1))),
        top_k=max(1, int(execution_raw.get("top_k", 12))),
        radii=tuple(int(v) for v in execution_raw.get("radii", DEFAULT_RADII)),
        embedding_neighbor_k=max(1, int(execution_raw.get("embedding_neighbor_k", 16))),
        embedding_radii=tuple(
            int(v) for v in execution_raw.get("embedding_radii", DEFAULT_EMBEDDING_RADII)
        ),
        max_self_prefixes_per_image=max(1, int(execution_raw.get("max_self_prefixes_per_image", 8))),
        save_raw_arrays=bool(execution_raw.get("save_raw_arrays", True)),
        run_infer_if_missing=bool(execution_raw.get("run_infer_if_missing", False)),
    )
    return StudyConfig(paths=paths, model=model, execution=execution)


def run_study(
    *,
    config_path: str | Path,
    stages: Sequence[str],
    limit: int | None = None,
    shard_index: int | None = None,
    num_shards: int | None = None,
    dry_run: bool = False,
    merge_shards: bool = False,
) -> dict[str, Any]:
    """Run one or more locality-study stages."""

    # preparing output directories
    config = load_study_config(config_path)
    if not dry_run and not merge_shards:
        config.paths.artifact_root.mkdir(parents=True, exist_ok=True)
        (config.paths.artifact_root / "plots").mkdir(parents=True, exist_ok=True)
        (config.paths.artifact_root / "examples").mkdir(parents=True, exist_ok=True)
    stage_summaries: list[dict[str, Any]] = []
    model_handle: ModelHandle | None = None

    # executing requested stages in order
    for raw_stage in stages:
        stage = str(raw_stage).strip().lower()
        if not stage:
            continue
        if stage == "embeddings":
            model_handle = model_handle or load_model_handle(config)
            stage_summaries.append(run_embedding_stage(config, model_handle=model_handle))
        elif stage == "teacher_forced":
            model_handle = model_handle or load_model_handle(config)
            stage_summaries.append(
                run_teacher_forced_stage(config, model_handle=model_handle, limit=limit)
            )
        elif stage == "self_prefix":
            model_handle = model_handle or load_model_handle(config)
            stage_summaries.append(
                run_self_prefix_stage(config, model_handle=model_handle, limit=limit)
            )
        elif stage == "plots":
            stage_summaries.append(run_plot_stage(config))
        elif stage == "report":
            stage_summaries.append(run_report_stage(config))
        elif stage == "x1_basin_attribution":
            if merge_shards:
                expected = num_shards if num_shards is not None else 8
                stage_summaries.append(
                    merge_lane_c_shards(config.paths.artifact_root, expected_shards=expected)
                )
                continue
            if dry_run:
                expected = num_shards if num_shards is not None else 8
                stage_summaries.append(
                    build_lane_c_shard_plan(
                        config_path=config_path,
                        output_root=config.paths.artifact_root,
                        num_shards=expected,
                    )
                )
                continue
            normalized_shard_index, normalized_num_shards, shard_label = normalize_lane_c_shard(
                shard_index=shard_index,
                num_shards=num_shards,
            )
            stage_config = config
            if shard_label is not None:
                shard_root = config.paths.artifact_root / "shards" / shard_label
                stage_config = replace(
                    config,
                    paths=replace(config.paths, artifact_root=shard_root),
                )
                stage_config.paths.artifact_root.mkdir(parents=True, exist_ok=True)
            model_handle = model_handle or load_model_handle(stage_config)
            stage_summaries.append(
                run_lane_c_x1_basin_attribution_stage(
                    stage_config,
                    model_handle=model_handle,
                    limit=limit,
                    shard_index=normalized_shard_index,
                    num_shards=normalized_num_shards,
                    shard_label=shard_label,
                )
            )
        else:
            raise ValueError(f"unknown hard-CE locality stage {stage!r}")

    summary = {"artifact_root": str(config.paths.artifact_root), "stages": stage_summaries}
    if not dry_run:
        _write_json(config.paths.artifact_root / "run_summary.json", summary)
    return summary


def load_model_handle(config: StudyConfig) -> ModelHandle:
    """Load the checkpoint with the same HF/Swift token-embeddings adapter semantics as inference."""

    # importing heavy dependencies lazily
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    from src.infer.checkpoints import (
        resolve_inference_checkpoint,
        validate_compact_coord_token_adapter_contract,
    )
    from src.tokens.row_offsets import install_token_embeddings_adapter, reattach_token_embeddings_adapter_hooks

    # resolving adapter shorthand and processor source
    resolved = resolve_inference_checkpoint(model_checkpoint=str(config.paths.checkpoint))
    validate_compact_coord_token_adapter_contract(
        resolved,
        detection_template_id="compact",
    )
    processor_source = str(resolved.resolved_base_model_checkpoint)
    processor = AutoProcessor.from_pretrained(
        processor_source,
        trust_remote_code=True,
        local_files_only=Path(processor_source).exists(),
    )
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("AutoProcessor did not expose a tokenizer")
    _configure_tokenizer_padding(tokenizer)

    # loading base model and optional adapter
    device = _resolve_device(config.model.device)
    attn = _resolve_attn(config.model.attn_implementation, device=device)
    dtype = _resolve_torch_dtype(config.model.torch_dtype)
    model = _load_qwen_with_attention_fallback(
        Qwen3VLForConditionalGeneration,
        checkpoint=str(resolved.resolved_base_model_checkpoint),
        dtype=dtype,
        requested_attn=attn,
        device=device,
    )

    token_embeddings_adapter = None
    adapter_checkpoint = str(resolved.resolved_adapter_checkpoint or "").strip()
    adapter_info = getattr(resolved, "adapter_info", None)
    token_embeddings_adapter_spec = getattr(adapter_info, "token_embeddings_adapter_spec", None) if adapter_info else None
    base_embedding_rows = None
    base_output_rows = None
    if adapter_checkpoint:
        if token_embeddings_adapter_spec is not None:
            install_token_embeddings_adapter(
                model,
                token_ids=token_embeddings_adapter_spec.token_ids,
                tie_head=token_embeddings_adapter_spec.tie_head,
            )
        vocab = resolve_coord_token_ids(tokenizer)
        base_embedding_rows, base_output_rows = _extract_base_rows(model, vocab.coord_token_ids)
        try:
            from swift import Swift
        except ImportError as exc:
            raise RuntimeError("Swift adapter loading requires the 'swift' package") from exc
        model = Swift.from_pretrained(model, model_id=adapter_checkpoint, inference_mode=True)
        if token_embeddings_adapter_spec is not None:
            token_embeddings_adapter = reattach_token_embeddings_adapter_hooks(model)
            if token_embeddings_adapter is None:
                raise RuntimeError("token_embeddings_adapter hooks could not be reattached")
    else:
        vocab = resolve_coord_token_ids(tokenizer)
        base_embedding_rows, base_output_rows = _extract_base_rows(model, vocab.coord_token_ids)

    model.eval()
    return ModelHandle(
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        resolved_checkpoint=resolved,
        token_embeddings_adapter=token_embeddings_adapter,
        base_embedding_rows=base_embedding_rows,
        base_output_rows=base_output_rows,
    )


def run_embedding_stage(config: StudyConfig, *, model_handle: ModelHandle) -> dict[str, Any]:
    """Extract coordinate-token row geometry and write metrics/arrays."""

    # extracting surfaces
    vocab = resolve_coord_token_ids(model_handle.tokenizer)
    surfaces = extract_embedding_surfaces(model_handle, vocab=vocab)
    embedding_dir = config.paths.artifact_root / "embeddings"
    embedding_dir.mkdir(parents=True, exist_ok=True)

    # computing per-surface metrics and rows
    metrics_by_surface: dict[str, Any] = {}
    row_path = config.paths.artifact_root / "embedding_rows.jsonl"
    with row_path.open("w", encoding="utf-8") as handle:
        for surface_name, matrix in sorted(surfaces.items()):
            if matrix.shape[0] != 1000:
                continue
            metrics = compute_embedding_geometry_metrics(
                matrix,
                neighbor_k=config.execution.embedding_neighbor_k,
                radii=config.execution.embedding_radii,
            )
            metrics_by_surface[surface_name] = _jsonable(asdict(metrics))
            if config.execution.save_raw_arrays:
                np.save(embedding_dir / f"{surface_name}.npy", matrix.astype(np.float32))
                np.save(
                    embedding_dir / f"{surface_name}__cosine_distance.npy",
                    _pairwise_cosine_distance(matrix).astype(np.float32),
                )
                np.save(
                    embedding_dir / f"{surface_name}__euclidean_distance.npy",
                    _pairwise_euclidean(matrix).astype(np.float32),
                )
            projection = _pca_projection(matrix, dims=2)
            nearest = np.argsort(_pairwise_euclidean(matrix), axis=1, kind="stable")[:, 1:9]
            local_steps = np.linalg.norm(np.diff(matrix.astype(np.float64), axis=0), axis=1)
            for coord_bin, token_id in enumerate(vocab.coord_token_ids):
                nearest_bins = [int(v) for v in nearest[coord_bin].tolist()]
                record = {
                    "surface": surface_name,
                    "token_text": int_to_token(coord_bin),
                    "token_id": int(token_id),
                    "coord_bin": int(coord_bin),
                    "norm": float(np.linalg.norm(matrix[coord_bin])),
                    "nearest_coord_bins": nearest_bins,
                    "nearest_coord_distances": [
                        float(_pairwise_euclidean(matrix[[coord_bin, nb]])[0, 1])
                        for nb in nearest_bins
                    ],
                    "local_step_prev": float(local_steps[coord_bin - 1]) if coord_bin > 0 else None,
                    "local_step_next": float(local_steps[coord_bin]) if coord_bin < len(local_steps) else None,
                    "pca_x": float(projection[coord_bin, 0]),
                    "pca_y": float(projection[coord_bin, 1]),
                }
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    # writing summary
    summary = {
        "stage": "embeddings",
        "coord_token_count": len(vocab.coord_token_ids),
        "wildcard_token_id": vocab.wildcard_token_id,
        "surfaces": metrics_by_surface,
        "embedding_rows": str(row_path),
    }
    _write_json(config.paths.artifact_root / "embedding_summary.json", summary)
    return summary


def extract_embedding_surfaces(
    model_handle: ModelHandle,
    *,
    vocab: CoordVocab,
) -> dict[str, np.ndarray]:
    """Extract base, offset, and effective coordinate row surfaces."""

    # reading current model rows
    coord_ids = tuple(int(token_id) for token_id in vocab.coord_token_ids)
    base_input, base_output = _extract_base_rows(model_handle.model, coord_ids)
    if model_handle.base_embedding_rows is not None:
        base_input = model_handle.base_embedding_rows
    if model_handle.base_output_rows is not None:
        base_output = model_handle.base_output_rows
    surfaces: dict[str, np.ndarray] = {
        "base_input": base_input.astype(np.float32),
        "base_output": base_output.astype(np.float32),
    }

    # applying token-embeddings adapter semantics explicitly
    adapter = model_handle.token_embeddings_adapter
    if adapter is not None:
        offset_by_id = _adapter_offsets_by_token_id(adapter)
        embed_offsets = np.stack(
            [offset_by_id.get(int(token_id), np.zeros(base_input.shape[1], dtype=np.float32)) for token_id in coord_ids],
            axis=0,
        )
        head_offsets = embed_offsets
        if not bool(getattr(adapter, "tie_head", True)) and getattr(adapter, "head_offset", None) is not None:
            head_offset_by_id = _adapter_offsets_by_token_id(adapter, use_head=True)
            head_offsets = np.stack(
                [head_offset_by_id.get(int(token_id), np.zeros(base_output.shape[1], dtype=np.float32)) for token_id in coord_ids],
                axis=0,
            )
        surfaces["token_embeddings_adapter_offset"] = embed_offsets.astype(np.float32)
        surfaces["effective_input"] = (base_input + embed_offsets).astype(np.float32)
        surfaces["effective_output"] = (base_output + head_offsets).astype(np.float32)
    else:
        surfaces["effective_input"] = base_input.astype(np.float32)
        surfaces["effective_output"] = base_output.astype(np.float32)

    return surfaces


def run_teacher_forced_stage(
    config: StudyConfig,
    *,
    model_handle: ModelHandle,
    limit: int | None = None,
) -> dict[str, Any]:
    """Extract coordinate logits under ground-truth teacher-forcing prefixes."""

    # preparing examples and forwarding batches
    examples = prepare_teacher_forced_examples(
        config,
        model_handle=model_handle,
        limit=limit,
    )
    rows = _score_forward_examples(
        config,
        model_handle=model_handle,
        examples=examples,
        output_suffix="teacher_forced",
    )
    return _append_per_coord_rows(
        config,
        rows=rows,
        stage_name="teacher_forced",
    )


def run_self_prefix_stage(
    config: StudyConfig,
    *,
    model_handle: ModelHandle,
    limit: int | None = None,
) -> dict[str, Any]:
    """Extract coordinate logits when the prefix comes from self-rollout rows."""

    # checking artifact availability
    if config.paths.self_rollout_root is None:
        raise ValueError("self_prefix stage requires paths.self_rollout_root")
    if not config.paths.self_rollout_root.exists():
        if config.execution.run_infer_if_missing and config.paths.self_rollout_regen_config is not None:
            _run_self_rollout_regeneration(config.paths.self_rollout_regen_config)
        else:
            raise FileNotFoundError(
                f"self-rollout artifact root not found: {config.paths.self_rollout_root}"
            )

    # preparing generated-prefix forced continuations
    examples = prepare_self_prefix_examples(
        config,
        model_handle=model_handle,
        limit=limit,
    )
    rows = _score_forward_examples(
        config,
        model_handle=model_handle,
        examples=examples,
        output_suffix="self_prefix",
    )
    return _append_per_coord_rows(
        config,
        rows=rows,
        stage_name="self_prefix",
    )


def run_lane_c_x1_basin_attribution_stage(
    config: StudyConfig,
    *,
    model_handle: ModelHandle,
    limit: int | None = None,
    shard_index: int | None = None,
    num_shards: int | None = None,
    shard_label: str | None = None,
) -> dict[str, Any]:
    """Score Lane-C intended-target coordinate basins for teacher/self prefixes."""

    examples, build_summary = prepare_lane_c_x1_basin_examples(
        config,
        model_handle=model_handle,
        limit=limit,
        shard_index=shard_index,
        num_shards=num_shards,
    )
    rows = _score_forward_examples(
        config,
        model_handle=model_handle,
        examples=examples,
        output_suffix="lane_c_x1_basin_attribution",
    )
    per_slot_path = config.paths.artifact_root / "per_slot.jsonl"
    with per_slot_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_jsonable(row), ensure_ascii=True) + "\n")

    per_case_rows = summarize_lane_c_per_case_rows(rows)
    per_case_path = config.paths.artifact_root / "per_case.jsonl"
    with per_case_path.open("w", encoding="utf-8") as handle:
        for row in per_case_rows:
            handle.write(json.dumps(_jsonable(row), ensure_ascii=True) + "\n")

    summary = _lane_c_summary(
        rows,
        per_case_rows=per_case_rows,
        shard_metadata={
            "mode": "shard" if shard_label else "single",
            "shard_index": shard_index,
            "num_shards": num_shards,
            "shard_label": shard_label,
        },
    )
    summary.update(build_summary)
    _write_json(config.paths.artifact_root / "summary.json", summary)
    return {
        "stage": "x1_basin_attribution",
        "row_count": len(rows),
        "case_count": len(per_case_rows),
        "per_slot": str(per_slot_path),
        "per_case": str(per_case_path),
        "summary": str(config.paths.artifact_root / "summary.json"),
        "shard_label": shard_label,
    }


def prepare_lane_c_x1_basin_examples(
    config: StudyConfig,
    *,
    model_handle: ModelHandle,
    limit: int | None,
    shard_index: int | None = None,
    num_shards: int | None = None,
) -> tuple[list[PreparedForwardExample], dict[str, Any]]:
    """Build Lane-C teacher and generated-prefix forced-continuation examples."""

    if config.paths.lane_a_rollout_root is None:
        raise ValueError("x1_basin_attribution requires paths.lane_a_rollout_root")
    if config.paths.self_rollout_root is None:
        raise ValueError("x1_basin_attribution requires paths.self_rollout_root")
    lane_a_rows = _lane_c_group_rows_by_source_line(
        config.paths.lane_a_rollout_root / "per_row.jsonl"
    )
    rows = _read_jsonl(config.paths.dataset_jsonl)
    traces = _read_jsonl(config.paths.self_rollout_root / "pred_token_trace.jsonl")
    confidences = _read_jsonl(config.paths.self_rollout_root / "pred_confidence.jsonl")
    active_limit = min(
        config.execution.sample_limit if limit is None else int(limit),
        len(rows),
        len(traces),
        len(confidences),
    )
    scope_label = f"val{active_limit}" if limit is None else f"limit={active_limit}"
    template = get_detection_template("compact")
    system_prompt, user_prompt = _resolve_prompts(config)
    vocab = resolve_coord_token_ids(model_handle.tokenizer)
    examples: list[PreparedForwardExample] = []
    skipped: Counter[str] = Counter()
    selected_record_count = 0

    for row_index in range(active_limit):
        if not lane_c_record_selected(
            row_index,
            limit=active_limit,
            shard_index=shard_index,
            num_shards=num_shards,
        ):
            continue
        selected_record_count += 1
        raw = parse_raw_detection_row(rows[row_index])
        normalized = normalize_detection_row(
            raw,
            object_ordering=_ordering_plan(config, row_index=row_index),
        )
        gt_count = len(normalized.objects)
        if gt_count <= 0:
            skipped["empty_gt"] += 1
            continue
        teacher_order_gt_indices = tuple(
            int(obj.source_object_index) for obj in normalized.objects
        )
        objects_by_source = {
            int(obj.source_object_index): obj for obj in normalized.objects
        }
        gt_bins_by_index = _lane_c_gt_bins_by_source_index(normalized)

        for depth in range(gt_count):
            prefix_state = _lane_c_teacher_prefix_state(
                teacher_order_gt_indices,
                depth=depth,
                gt_count=gt_count,
            )
            selection = select_lane_c_intended_target_gt_idx(
                teacher_order_gt_indices,
                prefix_state,
            )
            example = _lane_c_build_forward_example(
                config,
                model_handle=model_handle,
                raw=raw,
                normalized=normalized,
                source_line_idx=row_index,
                prefix_text="\n".join(
                    _render_normalized_object(obj) for obj in normalized.objects[:depth]
                ),
                prefix_mode="teacher_forced",
                prefix_depth=depth,
                prefix_state=prefix_state,
                selection=selection,
                objects_by_source=objects_by_source,
                gt_bins_by_index=gt_bins_by_index,
                scope_label=scope_label,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                vocab=vocab,
            )
            if example is None:
                skipped["teacher_forced_unrenderable"] += 1
            else:
                examples.append(example)

        trace = traces[row_index]
        confidence = confidences[row_index]
        valid_confidence_objects = _valid_confidence_objects(confidence, trace)
        generated_text = "".join(str(item) for item in trace.get("generated_token_text", []))
        complete_prefix_text = cut_complete_compact_rows(generated_text)
        prefix_rows = [line for line in complete_prefix_text.splitlines() if line.strip()]
        max_depth = min(
            len(prefix_rows),
            len(valid_confidence_objects),
            config.execution.max_self_prefixes_per_image,
        )
        for depth in range(max_depth + 1):
            prefix_state = build_lane_c_prefix_match_state(
                lane_a_rows.get(row_index, ()),
                depth=depth,
                gt_count=gt_count,
                match_policy="guarded",
            )
            selection = select_lane_c_intended_target_gt_idx(
                teacher_order_gt_indices,
                prefix_state,
            )
            if selection.intended_target_gt_idx is None:
                skipped["generated_prefix_no_remaining_gt"] += 1
                continue
            prefix_text = _generated_prefix_text_from_confidence(
                trace,
                valid_confidence_objects,
                depth=depth,
            )
            if prefix_text:
                prefix_text = _prefix_text_at_depth(prefix_text, depth=depth)
            if depth > 0 and len([line for line in prefix_text.splitlines() if line.strip()]) != depth:
                skipped["generated_prefix_unrenderable"] += 1
                continue
            example = _lane_c_build_forward_example(
                config,
                model_handle=model_handle,
                raw=raw,
                normalized=normalized,
                source_line_idx=row_index,
                prefix_text=prefix_text,
                prefix_mode="self_prefix",
                prefix_depth=depth,
                prefix_state=prefix_state,
                selection=selection,
                objects_by_source=objects_by_source,
                gt_bins_by_index=gt_bins_by_index,
                scope_label=scope_label,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                vocab=vocab,
            )
            if example is None:
                skipped["generated_prefix_unrenderable"] += 1
            else:
                examples.append(example)

    return examples, {
        "selected_record_count": selected_record_count,
        "skipped_counts": dict(skipped),
        "planned_example_count": len(examples),
    }


def prepare_teacher_forced_examples(
    config: StudyConfig,
    *,
    model_handle: ModelHandle,
    limit: int | None,
) -> list[PreparedForwardExample]:
    """Render ground-truth compact-full examples and locate coord tokens."""

    # loading and rendering dataset rows
    rows = _read_jsonl(config.paths.dataset_jsonl)
    active_limit = config.execution.sample_limit if limit is None else int(limit)
    scope_label = f"val{active_limit}" if limit is None else f"limit={active_limit}"
    examples: list[PreparedForwardExample] = []
    template = get_detection_template("compact")
    system_prompt, user_prompt = _resolve_prompts(config)
    for row_index, row in enumerate(rows[:active_limit]):
        raw = parse_raw_detection_row(row)
        normalized = normalize_detection_row(
            raw,
            object_ordering=_ordering_plan(config, row_index=row_index),
        )
        rendered = template.render_assistant(normalized)
        messages = build_detection_chat_messages(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            images=[str(_resolve_image(config, raw.images[0]))],
            assistant_text=rendered.text,
        )
        full_text = model_handle.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        full_input_ids = _processor_input_ids(
            model_handle.processor,
            full_text=full_text,
            image_path=_resolve_image(config, raw.images[0]),
        )
        assistant_ids = tuple(
            int(v)
            for v in model_handle.tokenizer.encode(
                rendered.text,
                add_special_tokens=False,
            )
        )
        assistant_start = _find_subsequence(full_input_ids, assistant_ids)
        if assistant_start is None:
            raise ValueError(f"assistant coord span not found for row {row_index}")
        slots = compact_slots_from_text(rendered.text)
        assistant_coord_positions = _assistant_coord_positions(
            assistant_ids=assistant_ids,
            slots=slots,
            coord_token_ids=resolve_coord_token_ids(model_handle.tokenizer).coord_token_ids,
            assistant_start=assistant_start,
        )
        if config.model.teacher_target_kind_mode == "all_hard_ce":
            target_kinds = tuple("hard_ce" for _ in slots)
            target_roles = tuple("coord" for _ in slots)
        else:
            target_kinds, target_roles = _target_kind_inventory(
                normalized=normalized,
                tokenizer=model_handle.tokenizer,
                messages=messages,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
        image_path = _resolve_image(config, raw.images[0])
        examples.append(
            PreparedForwardExample(
                row_index=row_index,
                image_id=int(raw.image_id),
                file_name=str(raw.file_name),
                image_path=image_path,
                width=int(raw.width),
                height=int(raw.height),
                assistant_text=rendered.text,
                full_text=str(full_text),
                full_input_ids=full_input_ids,
                coord_slots=slots,
                assistant_coord_positions=assistant_coord_positions,
                target_kinds=target_kinds,
                target_roles=target_roles,
                prefix_condition="teacher_forced",
                prefix_depth=-1,
                prefix_quality="gt_prefix",
                scope_label=scope_label,
                pairing_id=f"row{row_index}:teacher",
            )
        )
    return examples


def prepare_self_prefix_examples(
    config: StudyConfig,
    *,
    model_handle: ModelHandle,
    limit: int | None,
) -> list[PreparedForwardExample]:
    """Build forced GT continuations after complete generated compact rows."""

    # loading rollout sidecars
    if config.paths.self_rollout_root is None:
        raise ValueError("self_rollout_root is required")
    rows = _read_jsonl(config.paths.dataset_jsonl)
    traces = _read_jsonl(config.paths.self_rollout_root / "pred_token_trace.jsonl")
    confidences = _read_jsonl(config.paths.self_rollout_root / "pred_confidence.jsonl")
    matches = _load_match_rows(config.paths.self_rollout_root)
    active_limit = min(
        config.execution.sample_limit if limit is None else int(limit),
        len(rows),
        len(traces),
        len(confidences),
    )
    scope_label = f"val{active_limit}" if limit is None else f"limit={active_limit}"
    template = get_detection_template("compact")
    system_prompt, user_prompt = _resolve_prompts(config)
    vocab = resolve_coord_token_ids(model_handle.tokenizer)
    examples: list[PreparedForwardExample] = []

    # pairing generated prefixes with GT target rows
    for row_index in range(active_limit):
        raw = parse_raw_detection_row(rows[row_index])
        normalized = normalize_detection_row(
            raw,
            object_ordering=_ordering_plan(config, row_index=row_index),
        )
        trace = traces[row_index]
        confidence = confidences[row_index]
        valid_confidence_objects = _valid_confidence_objects(confidence, trace)
        generated_text = "".join(str(item) for item in trace.get("generated_token_text", []))
        complete_prefix_text = cut_complete_compact_rows(generated_text)
        prefix_rows = [line for line in complete_prefix_text.splitlines() if line.strip()]
        if not prefix_rows:
            prefix_depths = [0]
        else:
            prefix_depths = list(
                range(
                    0,
                    min(
                        len(prefix_rows),
                        len(valid_confidence_objects),
                        len(normalized.objects) - 1,
                        config.execution.max_self_prefixes_per_image,
                    )
                    + 1,
                )
            )
        for depth in prefix_depths:
            if depth >= len(normalized.objects):
                continue
            target = normalized.objects[depth]
            target_row_text = _render_normalized_object(target)
            prefix_text = _generated_prefix_text_from_confidence(
                trace,
                valid_confidence_objects,
                depth=depth,
            )
            if prefix_text:
                prefix_text = _prefix_text_at_depth(prefix_text, depth=depth)
            assistant_text = f"{prefix_text}\n{target_row_text}" if prefix_text else target_row_text
            messages = build_detection_chat_messages(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                images=[str(_resolve_image(config, raw.images[0]))],
                assistant_text=assistant_text,
            )
            full_text = model_handle.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            full_input_ids = _processor_input_ids(
                model_handle.processor,
                full_text=full_text,
                image_path=_resolve_image(config, raw.images[0]),
            )
            assistant_ids = tuple(
                int(v)
                for v in model_handle.tokenizer.encode(
                    assistant_text,
                    add_special_tokens=False,
                )
            )
            assistant_start = _find_subsequence(full_input_ids, assistant_ids)
            if assistant_start is None:
                continue
            all_slots = compact_slots_from_text(assistant_text)
            target_slots = tuple(slot for slot in all_slots if slot.object_order_index == depth)
            assistant_coord_positions = _assistant_coord_positions(
                assistant_ids=assistant_ids,
                slots=all_slots,
                coord_token_ids=vocab.coord_token_ids,
                assistant_start=assistant_start,
            )[-len(target_slots) :]
            image_path = _resolve_image(config, raw.images[0])
            examples.append(
                PreparedForwardExample(
                    row_index=row_index,
                    image_id=int(raw.image_id),
                    file_name=str(raw.file_name),
                    image_path=image_path,
                    width=int(raw.width),
                    height=int(raw.height),
                    assistant_text=assistant_text,
                    full_text=str(full_text),
                    full_input_ids=full_input_ids,
                    coord_slots=target_slots,
                    assistant_coord_positions=assistant_coord_positions,
                    target_kinds=tuple("hard_ce" for _ in target_slots),
                    target_roles=tuple("coord" for _ in target_slots),
                    prefix_condition="self_prefix",
                    prefix_depth=int(depth),
                    prefix_quality=_prefix_quality_for_depth(
                        matches.get(row_index),
                        prefix_rows=prefix_rows[:depth],
                        depth=depth,
                    ),
                    scope_label=scope_label,
                    pairing_id=f"row{row_index}:self_depth{depth}",
                )
            )
    return examples


def run_plot_stage(config: StudyConfig) -> dict[str, Any]:
    """Render diagnostic plots from existing JSONL summaries."""

    # importing plotting lazily
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = config.paths.artifact_root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_paths: list[str] = []

    # plotting embedding surfaces
    embedding_summary_path = config.paths.artifact_root / "embedding_summary.json"
    if embedding_summary_path.exists():
        embedding_rows = _read_jsonl(config.paths.artifact_root / "embedding_rows.jsonl")
        if embedding_rows:
            surface = "effective_output" if any(r.get("surface") == "effective_output" for r in embedding_rows) else str(embedding_rows[0].get("surface"))
            rows = [r for r in embedding_rows if r.get("surface") == surface]
            coords = np.asarray([int(r["coord_bin"]) for r in rows], dtype=np.int64)
            pca = np.asarray([[float(r["pca_x"]), float(r["pca_y"])] for r in rows], dtype=np.float64)
            plt.figure(figsize=(7, 5))
            sc = plt.scatter(pca[:, 0], pca[:, 1], c=coords, cmap="viridis", s=9)
            plt.colorbar(sc, label="coord bin")
            plt.title(f"Coordinate rows PCA: {surface}")
            plt.tight_layout()
            out = plot_dir / "embedding_projection_pca_umap.png"
            plt.savefig(out, dpi=180)
            plt.close()
            plot_paths.append(str(out))

            summary = json.loads(embedding_summary_path.read_text(encoding="utf-8"))
            plt.figure(figsize=(7, 4))
            for name, metric in sorted(summary.get("surfaces", {}).items()):
                recall = metric.get("knn_radius_recall", {})
                xs = [int(str(k).split("_")[-1]) for k in recall]
                ys = [float(recall[k]) for k in recall]
                if xs:
                    plt.plot(xs, ys, marker="o", label=name)
            plt.xlabel("numeric radius")
            plt.ylabel("kNN recall")
            plt.xscale("log", base=2)
            plt.ylim(0, 1.02)
            plt.legend(fontsize=7)
            plt.tight_layout()
            out = plot_dir / "embedding_knn_radius_recall.png"
            plt.savefig(out, dpi=180)
            plt.close()
            plot_paths.append(str(out))

            arrays_dir = config.paths.artifact_root / "embeddings"
            dist_path = arrays_dir / f"{surface}__cosine_distance.npy"
            if dist_path.exists():
                plt.figure(figsize=(6, 5))
                plt.imshow(np.load(dist_path), cmap="magma", aspect="auto")
                plt.colorbar(label="cosine distance")
                plt.title(f"Distance matrix: {surface}")
                plt.tight_layout()
                out = plot_dir / "embedding_distance_matrix__effective_coord_rows.png"
                plt.savefig(out, dpi=180)
                plt.close()
                plot_paths.append(str(out))

    # plotting logit locality rows
    per_coord_path = config.paths.artifact_root / "per_coord_rows.jsonl"
    if per_coord_path.exists():
        rows = _read_jsonl(per_coord_path)
        plot_paths.extend(_plot_logit_rows(rows, plot_dir=plot_dir))

    return {"stage": "plots", "plot_count": len(plot_paths), "plots": plot_paths}


def run_report_stage(config: StudyConfig) -> dict[str, Any]:
    """Assemble a Markdown report and machine-readable summary."""

    # summarizing available artifacts
    per_coord_path = config.paths.artifact_root / "per_coord_rows.jsonl"
    coord_rows = _read_jsonl(per_coord_path) if per_coord_path.exists() else []
    embedding_summary_path = config.paths.artifact_root / "embedding_summary.json"
    embedding_summary = (
        json.loads(embedding_summary_path.read_text(encoding="utf-8"))
        if embedding_summary_path.exists()
        else {}
    )
    grouped = _summarize_coord_rows(coord_rows)
    hard_ce_by_condition = _summarize_by_fields(
        coord_rows,
        fields=("prefix_condition",),
        filter_values={"target_kind": "hard_ce"},
    )
    hard_ce_by_area = _summarize_by_fields(
        coord_rows,
        fields=("prefix_condition", "bbox_area_bucket"),
        filter_values={"target_kind": "hard_ce"},
    )
    hard_ce_by_prefix_quality = _summarize_by_fields(
        coord_rows,
        fields=("prefix_quality",),
        filter_values={"target_kind": "hard_ce", "prefix_condition": "self_prefix"},
    )
    hard_ce_by_boundary = _summarize_by_fields(
        coord_rows,
        fields=("prefix_condition", "boundary_flag"),
        filter_values={"target_kind": "hard_ce"},
    )
    decision = _decision_hint(grouped, embedding_summary)
    row_scope_labels = sorted({str(row.get("scope_label", "unknown")) for row in coord_rows})
    summary = {
        "scope": {
            "sample_limit": config.execution.sample_limit,
            "row_scope_labels": row_scope_labels,
            "checkpoint": str(config.paths.checkpoint),
            "dataset_jsonl": str(config.paths.dataset_jsonl),
            "self_rollout_root": str(config.paths.self_rollout_root) if config.paths.self_rollout_root else None,
            "teacher_target_kind_mode": config.model.teacher_target_kind_mode,
        },
        "coord_rows": grouped,
        "slices": {
            "hard_ce_by_condition": hard_ce_by_condition,
            "hard_ce_by_area_bucket": hard_ce_by_area,
            "hard_ce_by_self_prefix_quality": hard_ce_by_prefix_quality,
            "hard_ce_by_boundary_flag": hard_ce_by_boundary,
        },
        "embedding_summary": embedding_summary.get("surfaces", {}),
        "decision_hint": decision,
    }
    _write_json(config.paths.artifact_root / "summary.json", summary)

    # writing report markdown
    lines = [
        "# Hard-CE Coordinate-Logit And Token-Embedding Locality Report",
        "",
        f"- Checkpoint: `{config.paths.checkpoint}`",
        f"- Dataset: `{config.paths.dataset_jsonl}`",
        f"- Scope: `val{config.execution.sample_limit}` by default unless overridden at run time",
        f"- Extracted row scopes in current artifacts: `{', '.join(row_scope_labels) if row_scope_labels else 'none'}`",
        "",
        "## Output-Distribution Locality",
        "",
    ]
    if grouped:
        for key, row in sorted(grouped.items()):
            lines.append(
                f"- `{key}`: n={row['count']}, mean mass@4={row.get('mass_at_4_mean')}, "
                f"mean top1 distance={row.get('top1_distance_mean')}, "
                f"mean entropy={row.get('entropy_mean')}, GT top1 rate={row.get('gt_top1_rate')}"
            )
    else:
        lines.append("- No coordinate-logit rows have been extracted yet.")
    lines.extend(["", "## Token-Embedding Locality", ""])
    surfaces = embedding_summary.get("surfaces", {})
    if surfaces:
        for surface, metric in sorted(surfaces.items()):
            lines.append(
                f"- `{surface}`: Pearson distance-vs-numeric={metric.get('pearson_distance_numeric')}, "
                f"Spearman={metric.get('spearman_distance_numeric')}, "
                f"kNN radius_4={metric.get('knn_radius_recall', {}).get('radius_4')}, "
                f"directionality R2={metric.get('directionality_r2')}"
            )
    else:
        lines.append("- No embedding rows have been extracted yet.")
    lines.extend(["", "## Slice Diagnostics", ""])
    if hard_ce_by_condition:
        lines.append("### Hard-CE Rows By Prefix Condition")
        lines.extend(_markdown_summary_bullets(hard_ce_by_condition))
    if hard_ce_by_area:
        lines.append("")
        lines.append("### Hard-CE Rows By Object Area")
        lines.extend(_markdown_summary_bullets(hard_ce_by_area))
    if hard_ce_by_prefix_quality:
        lines.append("")
        lines.append("### Self-Prefix Quality")
        lines.extend(_markdown_summary_bullets(hard_ce_by_prefix_quality))
    if hard_ce_by_boundary:
        lines.append("")
        lines.append("### Boundary Boxes")
        lines.extend(_markdown_summary_bullets(hard_ce_by_boundary))
    lines.extend(
        [
            "",
            "## Decision Hint",
            "",
            decision,
            "",
            "## Caveats",
            "",
            "- Embedding locality alone is not grounding evidence; use it to explain available row geometry.",
            "- Teacher-forced and self-prefix scopes must be reported separately.",
            "- Descriptive Gaussian/Laplacian fits are diagnostics, not assumptions about the true shape.",
        ]
    )
    report_path = config.paths.artifact_root / "report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"stage": "report", "report": str(report_path), "summary": str(config.paths.artifact_root / "summary.json")}


def _score_forward_examples(
    config: StudyConfig,
    *,
    model_handle: ModelHandle,
    examples: Sequence[PreparedForwardExample],
    output_suffix: str,
) -> list[dict[str, Any]]:
    """Run forward passes and collect per-coordinate distribution records."""

    # forwarding batches
    vocab = resolve_coord_token_ids(model_handle.tokenizer)
    rows: list[dict[str, Any]] = []
    batch_size = max(1, int(config.execution.batch_size))
    raw_prob_arrays: list[np.ndarray] = []
    raw_logit_arrays: list[np.ndarray] = []
    hidden_arrays: list[np.ndarray] = []
    for start in range(0, len(examples), batch_size):
        batch = list(examples[start : start + batch_size])
        images = [_load_image(example.image_path) for example in batch]
        model_inputs = model_handle.processor(
            text=[example.full_text for example in batch],
            images=images,
            return_tensors="pt",
            padding=True,
        )
        model_inputs = {
            key: value.to(_model_device(model_handle.model)) if isinstance(value, torch.Tensor) else value
            for key, value in model_inputs.items()
        }
        with torch.inference_mode():
            outputs = model_handle.model(**model_inputs, use_cache=False, output_hidden_states=True)
        logits = getattr(outputs, "logits", None)
        hidden_states = getattr(outputs, "hidden_states", None)
        if not isinstance(logits, torch.Tensor):
            raise RuntimeError("model forward did not return logits")
        input_ids = model_inputs.get("input_ids")
        if not isinstance(input_ids, torch.Tensor):
            raise RuntimeError("processor output missing input_ids")
        last_hidden = hidden_states[-1] if hidden_states is not None else None

        for batch_idx, example in enumerate(batch):
            pad_offset = _batch_pad_offset(
                input_ids=input_ids,
                batch_idx=batch_idx,
                expected_ids=example.full_input_ids,
            )
            for slot_index, (slot, position) in enumerate(zip(example.coord_slots, example.assistant_coord_positions, strict=True)):
                abs_pos = int(pad_offset + position)
                pred_pos = prediction_position_for_label_position(abs_pos)
                teacher_token_id = int(input_ids[batch_idx, abs_pos].detach().cpu().item())
                expected_token_id = int(vocab.coord_token_ids[slot.gt_bin])
                if teacher_token_id != expected_token_id:
                    raise RuntimeError(
                        f"decoded teacher token mismatch at row={example.row_index} "
                        f"slot={slot.slot}: got {teacher_token_id}, expected {expected_token_id}"
                    )
                metrics = distribution_metrics_from_logits(
                    logits=logits[batch_idx, pred_pos],
                    coord_token_ids=vocab.coord_token_ids,
                    gt_bin=slot.gt_bin,
                    radii=config.execution.radii,
                    top_k=config.execution.top_k,
                )
                raw_prob_arrays.append(metrics.conditional_probs.astype(np.float32))
                raw_logit_arrays.append(metrics.coord_logits.astype(np.float32))
                hidden_norm = None
                if isinstance(last_hidden, torch.Tensor):
                    hidden_vec = last_hidden[batch_idx, pred_pos].detach().float().cpu().numpy()
                    hidden_arrays.append(hidden_vec.astype(np.float32))
                    hidden_norm = float(np.linalg.norm(hidden_vec))
                base_row = _coord_metric_row(
                    config,
                    example=example,
                    slot=slot,
                    slot_index=slot_index,
                    abs_pos=abs_pos,
                    pred_pos=pred_pos,
                    teacher_token_id=teacher_token_id,
                    metrics=metrics,
                    hidden_norm=hidden_norm,
                    model_handle=model_handle,
                )
                rows.append(
                    _lane_c_enrich_coord_metric_row(
                        base_row,
                        slot=slot,
                        metrics=metrics,
                        metadata=example.lane_c_metadata,
                    )
                )

    # persisting dense arrays outside JSONL
    if config.execution.save_raw_arrays and raw_prob_arrays:
        arrays_dir = config.paths.artifact_root / "arrays"
        arrays_dir.mkdir(parents=True, exist_ok=True)
        np.save(arrays_dir / f"{output_suffix}__p_cond.npy", np.stack(raw_prob_arrays, axis=0))
        np.save(arrays_dir / f"{output_suffix}__coord_logits.npy", np.stack(raw_logit_arrays, axis=0))
        if hidden_arrays:
            np.save(arrays_dir / f"{output_suffix}__hidden.npy", np.stack(hidden_arrays, axis=0))
    return rows


def _coord_metric_row(
    config: StudyConfig,
    *,
    example: PreparedForwardExample,
    slot: CompactSlot,
    slot_index: int,
    abs_pos: int,
    pred_pos: int,
    teacher_token_id: int,
    metrics: DistributionMetrics,
    hidden_norm: float | None,
    model_handle: ModelHandle,
) -> dict[str, Any]:
    """Build one compact JSONL row for a scored coordinate position."""

    # deriving geometry buckets
    x1, y1, x2, y2 = slot.bbox_xyxy
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    area = width * height
    bridge = _embedding_bridge_metrics(model_handle, gt_bin=slot.gt_bin, distribution=metrics.conditional_probs, top1_bin=metrics.top1_bin)

    return {
        "checkpoint": str(config.paths.checkpoint),
        "dataset_jsonl": str(config.paths.dataset_jsonl),
        "scope_label": example.scope_label,
        "row_index": int(example.row_index),
        "image_id": example.image_id,
        "file_name": example.file_name,
        "prefix_condition": example.prefix_condition,
        "prefix_depth": int(example.prefix_depth),
        "prefix_quality": example.prefix_quality,
        "pairing_id": example.pairing_id,
        "object_order_index": int(slot.object_order_index),
        "desc": slot.desc,
        "slot": slot.slot,
        "slot_index": int(slot_index),
        "gt_bin": int(slot.gt_bin),
        "bbox_xyxy": [int(v) for v in slot.bbox_xyxy],
        "bbox_width": int(width),
        "bbox_height": int(height),
        "bbox_area": int(area),
        "bbox_min_side": int(min(width, height)),
        "bbox_area_bucket": _area_bucket(area),
        "boundary_flag": bool(min(slot.bbox_xyxy) <= 8 or max(slot.bbox_xyxy) >= 991),
        "target_kind": example.target_kinds[slot_index] if slot_index < len(example.target_kinds) else "unknown",
        "target_token_role": example.target_roles[slot_index] if slot_index < len(example.target_roles) else "unknown",
        "label_position": int(abs_pos),
        "prediction_position": int(pred_pos),
        "teacher_token_id": int(teacher_token_id),
        "coord_vocab_mass": metrics.coord_vocab_mass,
        "p_gt_full": metrics.p_gt_full,
        "p_gt_cond": metrics.p_gt_cond,
        "rank_gt": metrics.rank_gt,
        "top1_bin": metrics.top1_bin,
        "top1_distance": metrics.top1_distance,
        "expected_bin": metrics.expected_bin,
        "expected_abs_error": metrics.expected_abs_error,
        "entropy": metrics.entropy,
        "normalized_entropy": metrics.normalized_entropy,
        "effective_support": metrics.effective_support,
        "peak_probability": metrics.peak_probability,
        **metrics.mass_by_radius,
        "local_maxima_count": metrics.local_maxima_count,
        "secondary_peak_bin": metrics.secondary_peak_bin,
        "secondary_peak_distance": metrics.secondary_peak_distance,
        "top2_top1_ratio": metrics.top2_top1_ratio,
        "local_distance_correlation": metrics.local_distance_correlation,
        "gaussian_kl": metrics.gaussian_kl,
        "gaussian_scale": metrics.gaussian_scale,
        "laplacian_kl": metrics.laplacian_kl,
        "laplacian_scale": metrics.laplacian_scale,
        "shape_label": metrics.shape_label,
        "top_bins": metrics.top_bins,
        "hidden_norm": hidden_norm,
        **bridge,
    }


def _append_per_coord_rows(
    config: StudyConfig,
    *,
    rows: Sequence[dict[str, Any]],
    stage_name: str,
) -> dict[str, Any]:
    """Write scored coordinate rows and return a stage summary."""

    # overwriting the current stage rows for reproducible reruns
    stage_path = config.paths.artifact_root / f"{stage_name}_per_coord_rows.jsonl"
    with stage_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_jsonable(row), ensure_ascii=True) + "\n")

    # rebuilding the combined row view from known stage-specific files
    path = config.paths.artifact_root / "per_coord_rows.jsonl"
    combined_files = [
        config.paths.artifact_root / "teacher_forced_per_coord_rows.jsonl",
        config.paths.artifact_root / "self_prefix_per_coord_rows.jsonl",
    ]
    with path.open("w", encoding="utf-8") as out_handle:
        for source_path in combined_files:
            if not source_path.exists():
                continue
            with source_path.open("r", encoding="utf-8") as in_handle:
                for line in in_handle:
                    if line.strip():
                        out_handle.write(line)
    summary = {
        "stage": stage_name,
        "row_count": len(rows),
        "per_coord_rows": str(path),
        "stage_per_coord_rows": str(stage_path),
        "by_slot": _summarize_coord_rows(rows),
    }
    _write_json(config.paths.artifact_root / f"{stage_name}_summary.json", summary)
    return summary


def _embedding_bridge_metrics(
    model_handle: ModelHandle,
    *,
    gt_bin: int,
    distribution: np.ndarray,
    top1_bin: int,
) -> dict[str, Any]:
    """Relate a dynamic logit distribution to static effective output rows."""

    # selecting effective output geometry if available
    try:
        vocab = resolve_coord_token_ids(model_handle.tokenizer)
        surface = extract_embedding_surfaces(model_handle, vocab=vocab).get("effective_output")
    except Exception:
        surface = None
    if surface is None or surface.shape[0] != 1000:
        return {
            "embedding_gt_neighbor_radius4_recall": None,
            "embedding_top1_distance_to_gt_row": None,
            "embedding_weighted_mean_distance_to_gt_row": None,
        }
    distances = np.linalg.norm(surface - surface[int(gt_bin)][None, :], axis=1)
    nearest = np.argsort(distances, kind="stable")[1:17]
    return {
        "embedding_gt_neighbor_radius4_recall": float(np.mean(np.abs(nearest - int(gt_bin)) <= 4)),
        "embedding_top1_distance_to_gt_row": float(distances[int(top1_bin)]),
        "embedding_weighted_mean_distance_to_gt_row": float(np.sum(distribution * distances)),
    }


def _target_kind_inventory(
    *,
    normalized: Any,
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    system_prompt: str | None,
    user_prompt: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Best-effort map from compact slots to recursive target kind/role."""

    # building sidecar target order with the detection objective helper
    try:
        prepared = prepare_detection_training_example(
            normalized,
            template=get_detection_template("compact"),
            tokenizer=tokenizer,
            mode="random_permutation_et_rmp_ce",
            state_weighting="uniform_permutation",
            normalization="semantic_image_bucket_balanced",
            system_prompt=system_prompt,
            user_content=user_prompt,
            messages=messages,
        )
        targets = [
            target
            for target in (prepared.recursive_detection_targets.token_targets if prepared.recursive_detection_targets else ())
            if target.token_role is TokenRole.COORD
        ]
    except Exception:
        return (tuple("unknown" for _ in range(4 * len(normalized.objects))), tuple("unknown" for _ in range(4 * len(normalized.objects))))
    return (
        tuple(str(target.kind) for target in targets),
        tuple(_enum_value(target.token_role) for target in targets),
    )


def _assistant_coord_positions(
    *,
    assistant_ids: Sequence[int],
    slots: Sequence[CompactSlot],
    coord_token_ids: Sequence[int],
    assistant_start: int,
) -> tuple[int, ...]:
    """Find absolute full-input positions for compact coord slots."""

    # matching coord token ids in slot order
    positions: list[int] = []
    search_start = 0
    for slot in slots:
        coord_id = int(coord_token_ids[int(slot.gt_bin)])
        found = None
        for index in range(search_start, len(assistant_ids)):
            if int(assistant_ids[index]) == coord_id:
                found = index
                break
        if found is None:
            raise ValueError(f"coord token {coord_id} for bin {slot.gt_bin} not found in assistant ids")
        positions.append(int(assistant_start + found))
        search_start = found + 1
    return tuple(positions)


def _processor_input_ids(processor: Any, *, full_text: str, image_path: Path) -> tuple[int, ...]:
    """Tokenize one image/text example without padding for position alignment."""

    inputs = processor(
        text=[full_text],
        images=[_load_image(image_path)],
        return_tensors="pt",
        padding=False,
    )
    ids = inputs.get("input_ids")
    if not isinstance(ids, torch.Tensor):
        raise RuntimeError("processor output missing input_ids")
    return tuple(int(v) for v in ids[0].detach().cpu().tolist())


def _batch_pad_offset(
    *,
    input_ids: torch.Tensor,
    batch_idx: int,
    expected_ids: Sequence[int],
) -> int:
    """Return left-padding offset and verify unpadded ids align."""

    seq_len = len(expected_ids)
    padded_len = int(input_ids.shape[1])
    pad_offset = int(padded_len - seq_len)
    observed = tuple(int(v) for v in input_ids[batch_idx, pad_offset:].detach().cpu().tolist())
    if observed != tuple(int(v) for v in expected_ids):
        raise RuntimeError("batched processor input ids do not align with unpadded ids")
    return pad_offset


def _load_image(path: Path) -> Any:
    """Load an image for Qwen processor inputs."""

    from PIL import Image

    return Image.open(path).convert("RGB")


def _extract_base_rows(model: Any, coord_ids: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """Extract raw input and output row matrices from a loaded model."""

    # reading embedding and lm-head weights
    embed_module = model.get_input_embeddings()
    head_module = model.get_output_embeddings()
    embed_weight = getattr(embed_module, "weight", None)
    if not isinstance(embed_weight, torch.Tensor):
        raise RuntimeError("model input embeddings do not expose a weight tensor")
    ids = torch.tensor([int(token_id) for token_id in coord_ids], dtype=torch.long, device=embed_weight.device)
    input_rows = embed_weight.detach().index_select(0, ids).float().cpu().numpy()
    head_weight = getattr(head_module, "weight", None) if head_module is not None else None
    if isinstance(head_weight, torch.Tensor):
        out_ids = ids.to(head_weight.device)
        output_rows = head_weight.detach().index_select(0, out_ids).float().cpu().numpy()
    else:
        output_rows = input_rows.copy()
    return input_rows.astype(np.float32), output_rows.astype(np.float32)


def _load_qwen_with_attention_fallback(
    model_cls: Any,
    *,
    checkpoint: str,
    dtype: torch.dtype,
    requested_attn: str,
    device: str,
) -> Any:
    """Load Qwen with a short attention-implementation fallback chain."""

    # trying the requested implementation before portable fallbacks
    candidates: list[str] = []
    for candidate in (requested_attn, "flash_attention_2", "sdpa", "eager"):
        value = str(candidate or "").strip().lower()
        if value and value not in candidates:
            candidates.append(value)
    last_exc: Exception | None = None
    for candidate in candidates:
        try:
            return model_cls.from_pretrained(
                checkpoint,
                torch_dtype=dtype,
                attn_implementation=candidate,
            ).to(device)
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            last_exc = exc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    raise RuntimeError(f"failed to load Qwen model from {checkpoint!r}") from last_exc


def _adapter_offsets_by_token_id(adapter: Any, *, use_head: bool = False) -> dict[int, np.ndarray]:
    """Return token-embeddings adapter rows keyed by token id."""

    # extracting adapter tensors by persisted coord id order
    coord_ids = getattr(adapter, "token_ids", None)
    if not isinstance(coord_ids, torch.Tensor):
        raise RuntimeError("token_embeddings_adapter is missing token_ids")
    if use_head and getattr(adapter, "head_offset", None) is not None:
        offset = adapter.head_offset
    else:
        offset = adapter.embed_offset
    tensor = offset.detach().float().cpu().numpy()
    return {
        int(token_id): tensor[index].astype(np.float32)
        for index, token_id in enumerate(coord_ids.detach().cpu().tolist())
    }


def _pairwise_euclidean(matrix: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distances."""

    arr = np.asarray(matrix, dtype=np.float64)
    sq = np.sum(arr * arr, axis=1, keepdims=True)
    dist_sq = np.maximum(sq + sq.T - 2.0 * arr @ arr.T, 0.0)
    return np.sqrt(dist_sq)


def _pairwise_cosine_distance(matrix: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine distances."""

    arr = np.asarray(matrix, dtype=np.float64)
    norm = np.linalg.norm(arr, axis=1, keepdims=True)
    normed = arr / np.clip(norm, 1e-12, None)
    return np.clip(1.0 - normed @ normed.T, 0.0, 2.0)


def _safe_corr(x: Sequence[float] | np.ndarray, y: Sequence[float] | np.ndarray, *, method: str) -> float:
    """Return finite Pearson or Spearman correlation with safe fallbacks."""

    # normalizing vectors
    xv = np.asarray(x, dtype=np.float64).reshape(-1)
    yv = np.asarray(y, dtype=np.float64).reshape(-1)
    if xv.shape[0] != yv.shape[0] or xv.shape[0] < 2:
        return 0.0
    mask = np.isfinite(xv) & np.isfinite(yv)
    xv = xv[mask]
    yv = yv[mask]
    if xv.shape[0] < 2:
        return 0.0
    if method == "spearman":
        xv = _rankdata(xv)
        yv = _rankdata(yv)
    x_std = float(np.std(xv))
    y_std = float(np.std(yv))
    if x_std <= 0.0 or y_std <= 0.0:
        return 0.0
    value = float(np.corrcoef(xv, yv)[0, 1])
    return value if math.isfinite(value) else 0.0


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Compute average ranks for one-dimensional numeric values."""

    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = rank
        start = end
    return ranks


def _mean_numeric_neighbor_rank(distance: np.ndarray, *, offset: int) -> float:
    """Average embedding-neighbor rank of i plus/minus a numeric offset."""

    ranks: list[float] = []
    for row in range(distance.shape[0]):
        order = np.argsort(distance[row], kind="stable")
        rank_by_index = np.empty_like(order)
        rank_by_index[order] = np.arange(len(order))
        for candidate in (row - int(offset), row + int(offset)):
            if 0 <= candidate < distance.shape[0]:
                ranks.append(float(rank_by_index[candidate]))
    return float(np.mean(ranks)) if ranks else 0.0


def _bandedness_ratio(distance: np.ndarray, *, near_radius: int = 8, far_radius: int = 128) -> float:
    """Ratio of far off-diagonal distance to near-diagonal distance."""

    idx = np.arange(distance.shape[0])
    numeric = np.abs(idx[:, None] - idx[None, :])
    near = distance[(numeric > 0) & (numeric <= int(near_radius))]
    far = distance[numeric >= int(far_radius)]
    if near.size == 0 or far.size == 0:
        return 0.0
    return float(np.mean(far) / max(float(np.mean(near)), 1e-12))


def _linear_regression_r2(matrix: np.ndarray, y: np.ndarray) -> float:
    """Fit a ridge-stabilized linear probe from rows to coordinate index."""

    # solving a least-squares probe on centered data
    x = np.asarray(matrix, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64)
    x = x - np.mean(x, axis=0, keepdims=True)
    y_centered = yv - np.mean(yv)
    try:
        coef, *_ = np.linalg.lstsq(x, y_centered, rcond=1e-6)
        pred = x @ coef + np.mean(yv)
    except np.linalg.LinAlgError:
        return 0.0
    ss_res = float(np.sum((yv - pred) ** 2))
    ss_tot = float(np.sum((yv - np.mean(yv)) ** 2))
    if ss_tot <= 0.0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - ss_res / ss_tot))


def _first_pc_projection(matrix: np.ndarray) -> np.ndarray:
    """Return the first principal-component coordinate."""

    projection = _pca_projection(matrix, dims=1)
    return projection[:, 0]


def _pca_projection(matrix: np.ndarray, *, dims: int) -> np.ndarray:
    """Return a deterministic PCA projection."""

    arr = np.asarray(matrix, dtype=np.float64)
    centered = arr - np.mean(arr, axis=0, keepdims=True)
    try:
        u, s, _vh = np.linalg.svd(centered, full_matrices=False)
        out = u[:, :dims] * s[:dims]
    except np.linalg.LinAlgError:
        out = np.zeros((arr.shape[0], dims), dtype=np.float64)
    if out.shape[1] < dims:
        out = np.pad(out, ((0, 0), (0, dims - out.shape[1])))
    return out.astype(np.float64)


def _contiguous_cluster_purity(matrix: np.ndarray, *, cluster_count: int = 20) -> float:
    """Measure whether k-means clusters correspond to numeric intervals."""

    labels = _simple_kmeans_labels(matrix, k=min(int(cluster_count), matrix.shape[0]))
    purities: list[float] = []
    for cluster in sorted(set(int(v) for v in labels)):
        members = np.where(labels == cluster)[0]
        if len(members) == 0:
            continue
        span = int(np.max(members) - np.min(members) + 1)
        purities.append(float(len(members) / max(span, 1)))
    return float(np.mean(purities)) if purities else 0.0


def _simple_kmeans_labels(matrix: np.ndarray, *, k: int, iterations: int = 25) -> np.ndarray:
    """Small deterministic k-means for cluster-purity diagnostics."""

    arr = np.asarray(matrix, dtype=np.float64)
    if arr.shape[0] <= k:
        return np.arange(arr.shape[0], dtype=np.int64)
    seeds = np.linspace(0, arr.shape[0] - 1, num=k, dtype=np.int64)
    centers = arr[seeds].copy()
    labels = np.zeros(arr.shape[0], dtype=np.int64)
    for _ in range(int(iterations)):
        distances = ((arr[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = np.argmin(distances, axis=1)
        for cluster in range(k):
            members = arr[labels == cluster]
            if len(members):
                centers[cluster] = members.mean(axis=0)
    return labels


def _smooth_probs(probs: np.ndarray) -> np.ndarray:
    """Apply a tiny symmetric smoothing kernel for peak counting."""

    kernel = np.asarray([0.25, 0.5, 0.25], dtype=np.float64)
    padded = np.pad(probs, (1, 1), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _local_maxima(values: np.ndarray, *, min_prominence: float) -> list[int]:
    """Return local maxima indices above a small prominence threshold."""

    peaks: list[int] = []
    for index in range(len(values)):
        left = values[index - 1] if index > 0 else -math.inf
        right = values[index + 1] if index + 1 < len(values) else -math.inf
        if values[index] >= left and values[index] >= right and values[index] >= min_prominence:
            peaks.append(index)
    return peaks


def _fit_centered_family(
    probs: np.ndarray,
    *,
    gt_bin: int,
    family: str,
) -> tuple[float | None, float | None]:
    """Fit a target-centered Gaussian or Laplacian by moment matching."""

    bins = np.arange(len(probs), dtype=np.float64)
    distance = np.abs(bins - float(gt_bin))
    if family == "gaussian":
        scale = math.sqrt(max(float(np.sum(probs * distance * distance)), 1e-6))
        candidate = np.exp(-0.5 * (distance / max(scale, 1e-6)) ** 2)
    elif family == "laplacian":
        scale = max(float(np.sum(probs * distance)), 1e-6)
        candidate = np.exp(-distance / max(scale, 1e-6))
    else:
        raise ValueError(f"unknown family {family!r}")
    candidate = candidate / max(float(candidate.sum()), 1e-300)
    kl = float(np.sum(probs * (np.log(np.clip(probs, 1e-300, 1.0)) - np.log(np.clip(candidate, 1e-300, 1.0)))))
    return float(scale), kl if math.isfinite(kl) else None


def _classify_probability_shape(
    probs: np.ndarray,
    *,
    gt_bin: int,
    local_maxima_count: int,
    top1_distance: int,
    effective_support: float,
    top2_top1_ratio: float | None,
    secondary_peak_distance: int | None,
) -> str:
    """Assign a coarse descriptive label to a coordinate distribution."""

    if effective_support > 500:
        return "diffuse"
    if probs[int(gt_bin)] > 0.85 and top1_distance == 0 and effective_support < 3:
        return "sharp_delta"
    if top1_distance <= 4 and local_maxima_count <= 1:
        return "target_local"
    if (
        top1_distance <= 4
        and top2_top1_ratio is not None
        and top2_top1_ratio >= 0.25
        and secondary_peak_distance is not None
        and secondary_peak_distance >= 32
    ):
        return "bimodal_target_other"
    if top1_distance >= 16 and effective_support < 80:
        return "wrong_object_local"
    if local_maxima_count >= 3:
        return "multimodal_irregular"
    return "smooth_or_irregular"


def _lane_c_optional_int(value: Any) -> int | None:
    """Return an int for present scalar values, otherwise None."""

    if value is None:
        return None
    try:
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _lane_c_state_value(
    state: LaneCPrefixMatchState | Mapping[str, Any],
    key: str,
    default: Any,
) -> Any:
    """Read a field from either a Lane-C dataclass or mapping."""

    if isinstance(state, Mapping):
        return state.get(key, default)
    return getattr(state, key, default)


def _lane_c_is_guarded_tp_like(row: Mapping[str, Any]) -> bool:
    """Return whether a Lane-A row is a non-suppressed guarded TP-like row."""

    if bool(row.get("suppressed_by_guard", False)):
        return False
    guarded_label = row.get("guarded_match_label")
    if guarded_label is not None:
        return str(guarded_label).strip().lower() == "tp_like"
    return str(row.get("row_label", "")).strip().lower() == "tp_like"


def _lane_c_prefix_problem_classes(row: Mapping[str, Any]) -> set[str]:
    """Classify generated prefix rows into Lane-C problem families."""

    labels = {
        str(row.get(key, "")).strip().lower()
        for key in ("row_label", "raw_match_label", "guarded_match_label")
        if row.get(key) is not None
    }
    out: set[str] = set()
    if bool(row.get("suppressed_by_guard", False)) or any("duplicate" in label for label in labels):
        out.add("duplicate")
    if any("invalid" in label or "unparsed" in label for label in labels):
        out.add("invalid")
    if bool(row.get("ambiguous", False)) or any("ambiguous" in label for label in labels):
        out.add("ambiguous")
    if any(
        label in {"unmatched_fp", "wrong_desc_fp", "fp", "false_positive"}
        or label.endswith("_fp")
        for label in labels
    ):
        out.add("fp")
    return out


def _lane_c_prefix_quality(depth: int, problem_names: set[str]) -> str:
    """Map prefix problem families to the Lane-C quality label."""

    if int(depth) <= 0:
        return "empty_prefix"
    if not problem_names:
        return "clean_prefix"
    if len(problem_names) > 1:
        return "mixed_prefix"
    only = next(iter(problem_names))
    return {
        "fp": "fp_prefix",
        "duplicate": "duplicate_prefix",
        "invalid": "invalid_prefix",
        "ambiguous": "ambiguous_prefix",
    }.get(only, "mixed_prefix")


def _lane_c_normalized_probs(probs: Sequence[float] | np.ndarray) -> np.ndarray:
    """Validate and normalize a 1000-bin conditional probability vector."""

    p = np.asarray(probs, dtype=np.float64).reshape(-1)
    if p.shape[0] != 1000:
        raise ValueError(f"Lane-C probability vector must have 1000 bins; got {p.shape[0]}")
    if np.any(p < 0.0):
        raise ValueError("Lane-C probabilities must be non-negative")
    total = float(np.sum(p))
    if total <= 0.0 or not math.isfinite(total):
        return np.full(1000, 0.001, dtype=np.float64)
    return p / total


def _lane_c_extract_slot_bin(value: Any, *, slot: str) -> int | None:
    """Extract a slot bin from compact object-ish metadata."""

    if value is None:
        return None
    if isinstance(value, Mapping):
        for key in (slot, "bin"):
            if key in value:
                direct = _lane_c_extract_slot_bin(value[key], slot=slot)
                if direct is not None:
                    return direct
        for key in ("bins", "slot_bins", "coord_bins", "bbox_xyxy", "bbox", "points"):
            if key in value:
                nested = _lane_c_extract_slot_bin(value[key], slot=slot)
                if nested is not None:
                    return nested
        return None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) == 4 and slot in SLOT_NAMES:
            return _lane_c_valid_bin(value[SLOT_NAMES.index(slot)])
        if len(value) == 1:
            return _lane_c_valid_bin(value[0])
        return None
    return _lane_c_valid_bin(value)


def _lane_c_valid_bin(value: Any) -> int | None:
    """Return a valid coord bin or None."""

    bin_value = _lane_c_optional_int(value)
    if bin_value is None or not 0 <= bin_value < 1000:
        return None
    return int(bin_value)


def _lane_c_is_same_desc_gt(value: Any) -> bool:
    """Return whether GT metadata marks a same-description competitor."""

    if not isinstance(value, Mapping):
        return False
    if any(bool(value.get(key, False)) for key in ("same_desc", "same_desc_as_target", "same_desc_competitor")):
        return True
    attribution = str(value.get("attribution", "")).lower()
    return "same_desc" in attribution


def _lane_c_prefix_bins_by_category(
    prefix_bins_by_label: Mapping[str, Any],
    *,
    slot: str,
) -> dict[str, list[int]]:
    """Group prefix object bins into attribution categories."""

    grouped: dict[str, list[int]] = {
        "false_positive_prefix_object": [],
        "previous_generated_object": [],
    }
    for label, raw_value in prefix_bins_by_label.items():
        label_text = str(label).lower()
        category = (
            "false_positive_prefix_object"
            if "false_positive" in label_text or label_text.startswith("fp") or "_fp" in label_text
            else "previous_generated_object"
        )
        values: Iterable[Any]
        if isinstance(raw_value, Mapping):
            if any(key in raw_value for key in (slot, "bin", "bins", "slot_bins", "coord_bins", "bbox_xyxy", "bbox", "points")):
                values = (raw_value,)
            else:
                values = raw_value.values()
        elif isinstance(raw_value, Sequence) and not isinstance(raw_value, (str, bytes)):
            values = raw_value
        else:
            values = (raw_value,)
        for item in values:
            bin_value = _lane_c_extract_slot_bin(item, slot=slot)
            if bin_value is not None:
                grouped[category].append(bin_value)
    return grouped


def _lane_c_top_peak_attribution(
    *,
    top1_bin: int,
    target_bin: int,
    gt_bins_by_index: Mapping[int, Any],
    prefix_bins_by_label: Mapping[str, Any],
    slot: str,
    local_radius: int,
) -> str:
    """Attribute the top coordinate peak to the nearest known local object."""

    radius = max(0, int(local_radius))
    if abs(int(top1_bin) - int(target_bin)) <= radius:
        return "target_gt_object"

    same_desc_bins: list[int] = []
    other_gt_bins: list[int] = []
    for raw_value in gt_bins_by_index.values():
        bin_value = _lane_c_extract_slot_bin(raw_value, slot=slot)
        if bin_value is None or bin_value == int(target_bin):
            continue
        if _lane_c_is_same_desc_gt(raw_value):
            same_desc_bins.append(bin_value)
        else:
            other_gt_bins.append(bin_value)
    if any(abs(int(top1_bin) - candidate) <= radius for candidate in same_desc_bins):
        return "same_desc_competitor_gt_object"
    if any(abs(int(top1_bin) - candidate) <= radius for candidate in other_gt_bins):
        return "other_same_image_gt_object"

    prefix_bins = _lane_c_prefix_bins_by_category(prefix_bins_by_label, slot=slot)
    if any(abs(int(top1_bin) - candidate) <= radius for candidate in prefix_bins["false_positive_prefix_object"]):
        return "false_positive_prefix_object"
    if any(abs(int(top1_bin) - candidate) <= radius for candidate in prefix_bins["previous_generated_object"]):
        return "previous_generated_object"
    return "no_local_object_diffuse"


def _lane_c_group_rows_by_source_line(path: Path) -> dict[int, list[dict[str, Any]]]:
    """Load Lane-A per-row records grouped by dataset source line."""

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in _read_jsonl(path):
        source_line_idx = _lane_c_optional_int(row.get("source_line_idx"))
        if source_line_idx is None:
            continue
        grouped[int(source_line_idx)].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda item: _lane_c_optional_int(item.get("raw_pred_idx")) or 0)
    return grouped


def _lane_c_teacher_prefix_state(
    teacher_order_gt_indices: Sequence[int],
    *,
    depth: int,
    gt_count: int,
) -> LaneCPrefixMatchState:
    """Build Lane-C state for direct GT teacher-prefix coverage."""

    depth_i = max(0, int(depth))
    gt_count_i = max(0, int(gt_count))
    matched = tuple(
        sorted(
            {
                int(item)
                for item in teacher_order_gt_indices[:depth_i]
                if 0 <= int(item) < gt_count_i
            }
        )
    )
    matched_set = set(matched)
    return LaneCPrefixMatchState(
        depth=depth_i,
        gt_count=gt_count_i,
        match_policy="teacher_forced",
        consumed_raw_pred_indices=tuple(range(depth_i)),
        matched_prefix_gt_indices=matched,
        remaining_gt_indices=tuple(idx for idx in range(gt_count_i) if idx not in matched_set),
        fp_prefix_object_indices=(),
        duplicate_prefix_object_indices=(),
        invalid_prefix_object_indices=(),
        ambiguous_prefix_object_indices=(),
        prefix_quality="empty_prefix" if depth_i <= 0 else "gt_prefix",
    )


def _lane_c_gt_bins_by_source_index(normalized: Any) -> dict[int, dict[str, Any]]:
    """Return GT object metadata keyed by original source-object index."""

    desc_counts = Counter(str(obj.desc) for obj in normalized.objects)
    out: dict[int, dict[str, Any]] = {}
    for obj in normalized.objects:
        bins = tuple(_coord_bin_from_token(token) for token in obj.bbox_2d.tokens)
        out[int(obj.source_object_index)] = {
            "desc": str(obj.desc),
            "bbox_xyxy": [int(value) for value in bins],
            "x1": int(bins[0]),
            "y1": int(bins[1]),
            "x2": int(bins[2]),
            "y2": int(bins[3]),
            "object_instance_id": str(obj.object_instance_id),
            "source_object_index": int(obj.source_object_index),
            "same_desc_competitor": desc_counts[str(obj.desc)] > 1,
        }
    return out


def _lane_c_prefix_bins_by_label_from_text(
    prefix_text: str,
    prefix_state: LaneCPrefixMatchState,
) -> dict[str, list[dict[str, Any]]]:
    """Parse generated prefix boxes into attribution groups."""

    slots = compact_slots_from_text(prefix_text)
    by_object: dict[int, dict[str, Any]] = {}
    for slot in slots:
        item = by_object.setdefault(
            int(slot.object_order_index),
            {
                "desc": slot.desc,
                "bbox_xyxy": [int(value) for value in slot.bbox_xyxy],
            },
        )
        item[str(slot.slot)] = int(slot.gt_bin)
    fp_indices = set(prefix_state.fp_prefix_object_indices)
    out: dict[str, list[dict[str, Any]]] = {
        "previous_generated_object": [],
        "false_positive_prefix_object": [],
    }
    for object_index, item in sorted(by_object.items()):
        label = (
            "false_positive_prefix_object"
            if object_index in fp_indices
            else "previous_generated_object"
        )
        out[label].append(item)
    return out


def _lane_c_build_forward_example(
    config: StudyConfig,
    *,
    model_handle: ModelHandle,
    raw: Any,
    normalized: Any,
    source_line_idx: int,
    prefix_text: str,
    prefix_mode: str,
    prefix_depth: int,
    prefix_state: LaneCPrefixMatchState,
    selection: LaneCTargetSelection,
    objects_by_source: Mapping[int, NormalizedDetectionObject],
    gt_bins_by_index: Mapping[int, Any],
    scope_label: str,
    system_prompt: str | None,
    user_prompt: str,
    vocab: CoordVocab,
) -> PreparedForwardExample | None:
    """Render one Lane-C forced continuation and locate target coord positions."""

    target_idx = selection.intended_target_gt_idx
    if target_idx is None or int(target_idx) not in objects_by_source:
        return None
    target = objects_by_source[int(target_idx)]
    target_row_text = _render_normalized_object(target)
    clean_prefix = str(prefix_text).strip()
    assistant_text = f"{clean_prefix}\n{target_row_text}" if clean_prefix else target_row_text
    messages = build_detection_chat_messages(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        images=[str(_resolve_image(config, raw.images[0]))],
        assistant_text=assistant_text,
    )
    full_text = model_handle.processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    full_input_ids = _processor_input_ids(
        model_handle.processor,
        full_text=full_text,
        image_path=_resolve_image(config, raw.images[0]),
    )
    assistant_ids = tuple(
        int(v)
        for v in model_handle.tokenizer.encode(
            assistant_text,
            add_special_tokens=False,
        )
    )
    assistant_start = _find_subsequence(full_input_ids, assistant_ids)
    if assistant_start is None:
        return None
    all_slots = compact_slots_from_text(assistant_text)
    target_object_order_index = int(prefix_depth)
    target_slots = tuple(
        slot for slot in all_slots if slot.object_order_index == target_object_order_index
    )
    if len(target_slots) != len(SLOT_NAMES):
        return None
    assistant_coord_positions = _assistant_coord_positions(
        assistant_ids=assistant_ids,
        slots=all_slots,
        coord_token_ids=vocab.coord_token_ids,
        assistant_start=assistant_start,
    )[-len(target_slots) :]
    target_meta = dict(gt_bins_by_index[int(target_idx)])
    case_id = (
        f"row{int(source_line_idx)}:{prefix_mode}:"
        f"depth{int(prefix_depth)}:gt{int(target_idx)}"
    )
    metadata = {
        "case_id": case_id,
        "source_line_idx": int(source_line_idx),
        "prefix_mode": str(prefix_mode),
        "prefix_depth": int(prefix_depth),
        "prefix_quality": prefix_state.prefix_quality,
        "match_policy": prefix_state.match_policy,
        "intended_target_gt_idx": int(target_idx),
        "target_selection_rule": selection.target_selection_rule,
        "matched_prefix_gt_indices": list(prefix_state.matched_prefix_gt_indices),
        "remaining_gt_indices": list(prefix_state.remaining_gt_indices),
        "fp_prefix_object_indices": list(prefix_state.fp_prefix_object_indices),
        "duplicate_prefix_object_indices": list(prefix_state.duplicate_prefix_object_indices),
        "invalid_prefix_object_indices": list(prefix_state.invalid_prefix_object_indices),
        "ambiguous_prefix_object_indices": list(prefix_state.ambiguous_prefix_object_indices),
        "target_object_instance_id": target_meta.get("object_instance_id"),
        "target_source_object_index": int(target.source_object_index),
        "target_desc": str(target.desc),
        "target_bbox_xyxy": target_meta.get("bbox_xyxy"),
        "gt_bins_by_index": dict(gt_bins_by_index),
        "prefix_bins_by_label": _lane_c_prefix_bins_by_label_from_text(
            clean_prefix,
            prefix_state,
        ),
    }
    return PreparedForwardExample(
        row_index=int(source_line_idx),
        image_id=int(raw.image_id),
        file_name=str(raw.file_name),
        image_path=_resolve_image(config, raw.images[0]),
        width=int(raw.width),
        height=int(raw.height),
        assistant_text=assistant_text,
        full_text=str(full_text),
        full_input_ids=full_input_ids,
        coord_slots=target_slots,
        assistant_coord_positions=assistant_coord_positions,
        target_kinds=tuple("hard_ce" for _ in target_slots),
        target_roles=tuple("coord" for _ in target_slots),
        prefix_condition=str(prefix_mode),
        prefix_depth=int(prefix_depth),
        prefix_quality=prefix_state.prefix_quality,
        scope_label=scope_label,
        pairing_id=case_id,
        lane_c_metadata=metadata,
    )


def _lane_c_enrich_coord_metric_row(
    row: Mapping[str, Any],
    *,
    slot: CompactSlot,
    metrics: DistributionMetrics,
    metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Add Lane-C attribution metadata to a scored coordinate row."""

    out = dict(row)
    if not metadata:
        return out
    gt_bins_by_index = metadata.get("gt_bins_by_index") or {}
    prefix_bins_by_label = metadata.get("prefix_bins_by_label") or {}
    attr = attribute_lane_c_coord_distribution(
        metrics.conditional_probs,
        target_bin=int(slot.gt_bin),
        gt_bins_by_index=gt_bins_by_index,
        prefix_bins_by_label=prefix_bins_by_label,
        slot=str(slot.slot),
        radii=(4, 8),
    )
    out.update(_jsonable(metadata))
    out.update(
        {
            "prefix_condition": metadata.get("prefix_mode", row.get("prefix_condition")),
            "top_peak_attribution": attr.top_peak_attribution,
            "target_rank": attr.target_rank,
            "best_other_gt_idx": attr.best_other_gt_idx,
            "best_other_gt_bin": attr.best_other_gt_bin,
            "best_other_gt_rank": attr.best_other_gt_rank,
            "target_margin_vs_best_other": attr.target_margin_vs_best_other,
            "gt_top1": attr.gt_top1,
            "top1_distance": attr.top1_distance,
            "mass_at_radius_4": attr.mass_by_radius.get("mass_at_radius_4"),
            "mass_at_radius_8": attr.mass_by_radius.get("mass_at_radius_8"),
        }
    )
    out["lane_c_merge_key"] = ":".join(
        "" if part is None else str(part) for part in lane_c_merge_key(out)
    )
    return out


def _lane_c_summary(
    per_slot_rows: Sequence[Mapping[str, Any]],
    *,
    per_case_rows: Sequence[Mapping[str, Any]],
    shard_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Build Lane-C summary counts for shard or merged outputs."""

    by_prefix_mode = Counter(str(row.get("prefix_mode", row.get("prefix_condition", "unknown"))) for row in per_slot_rows)
    by_prefix_quality = Counter(str(row.get("prefix_quality", "unknown")) for row in per_slot_rows)
    by_slot = Counter(str(row.get("slot", "unknown")) for row in per_slot_rows)
    by_attribution = Counter(str(row.get("top_peak_attribution", "unknown")) for row in per_slot_rows)
    return {
        "stage": "x1_basin_attribution",
        "row_count": len(per_slot_rows),
        "case_count": len(per_case_rows),
        "counts_by_prefix_mode": dict(sorted(by_prefix_mode.items())),
        "counts_by_prefix_quality": dict(sorted(by_prefix_quality.items())),
        "counts_by_slot": dict(sorted(by_slot.items())),
        "counts_by_top_peak_attribution": dict(sorted(by_attribution.items())),
        "shard_metadata": dict(shard_metadata),
    }


def _coord_bin_from_token(token: str) -> int:
    """Parse `<|coord_N|>` into integer N."""

    match = COORD_TOKEN_RE.fullmatch(str(token))
    if match is None:
        raise ValueError(f"not a coordinate token: {token!r}")
    value = int(match.group(1))
    if value < 0 or value > 999:
        raise ValueError(f"coordinate bin outside 0..999: {token!r}")
    return value


def _resolve_prompts(config: StudyConfig) -> tuple[str | None, str]:
    """Resolve compact-full detection prompts for analysis rendering."""

    ordering = "random" if config.model.object_ordering == "random_permutation" else "sorted"
    return get_template_prompts(
        ordering=ordering,
        coord_mode="coord_tokens",
        prompt_variant=config.model.prompt_variant,
        object_field_order=config.model.object_field_order,
        bbox_format=config.model.bbox_format,
        detection_sequence_format=config.model.detection_sequence_format,
    )


def _ordering_plan(config: StudyConfig, *, row_index: int) -> ObjectOrderingPlan:
    """Mirror current DetectionTrainingDataset object ordering."""

    if config.model.object_ordering == "sorted":
        return ObjectOrderingPlan.sorted(seed_source="hard_ce_coord_logit_locality")
    return ObjectOrderingPlan.random_permutation(
        seed=_detection_mix_seed(config.model.seed, 0, int(row_index)),
        seed_source=f"hard_ce_coord_logit_locality:seed={config.model.seed}:row={row_index}",
    )


def _render_normalized_object(obj: NormalizedDetectionObject) -> str:
    """Render one normalized object as a compact-full row."""

    return render_compact_row(
        str(obj.desc),
        obj.bbox_2d.tokens,
        include_object_ref_marker=True,
        include_bbox_start_marker=True,
    )


def _resolve_image(config: StudyConfig, image: str) -> Path:
    """Resolve an image path under the configured image root."""

    candidate = Path(str(image))
    path = candidate if candidate.is_absolute() else config.paths.image_root / candidate
    return path.expanduser().resolve(strict=True)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into dictionaries."""

    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, MutableMapping):
                rows.append(dict(payload))
    return rows


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a JSON object with stable formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _jsonable(value: Any) -> Any:
    """Convert numpy/torch/dataclass values into JSON-safe objects."""

    if dataclass_isinstance(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def dataclass_isinstance(value: Any) -> bool:
    """Return whether value is a dataclass instance rather than class."""

    return hasattr(value, "__dataclass_fields__") and not isinstance(value, type)


def _required_mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """Read a required mapping value."""

    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return value


def _required_str(parent: Mapping[str, Any], key: str) -> str:
    """Read a required non-empty string."""

    value = parent.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _optional_path(value: Any) -> Path | None:
    """Resolve an optional path value."""

    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("optional path values must be non-empty strings")
    return _resolve_repo_path(value)


def _resolve_repo_path(value: str) -> Path:
    """Resolve absolute or repo-relative paths."""

    raw = Path(str(value)).expanduser()
    return raw if raw.is_absolute() else (REPO_ROOT / raw).resolve(strict=False)


def _configure_tokenizer_padding(tokenizer: Any) -> None:
    """Configure left padding for causal teacher-forced batches."""

    setattr(tokenizer, "padding_side", "left")
    if getattr(tokenizer, "pad_token_id", None) is None:
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token_id is None:
            raise RuntimeError("tokenizer.eos_token_id is required for padding")
        setattr(tokenizer, "pad_token_id", eos_token_id)


def _resolve_device(device: str) -> str:
    """Resolve `auto` device to CUDA when available."""

    requested = str(device or "auto").lower()
    if requested == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return str(device)


def _resolve_attn(attn: str, *, device: str) -> str:
    """Resolve attention implementation for model loading."""

    requested = str(attn or "auto").lower()
    if requested != "auto":
        return requested
    return "flash_attention_2" if "cuda" in str(device).lower() and torch.cuda.is_available() else "sdpa"


def _resolve_torch_dtype(dtype: str) -> torch.dtype:
    """Resolve YAML dtype names to torch dtypes."""

    normalized = str(dtype or "bfloat16").lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unsupported torch dtype {dtype!r}")


def _model_device(model: Any) -> torch.device:
    """Return the first parameter device for a loaded model."""

    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _find_subsequence(haystack: Sequence[int], needle: Sequence[int]) -> int | None:
    """Find a token subsequence start index."""

    if not needle:
        return 0
    last = len(haystack) - len(needle)
    for start in range(max(0, last + 1)):
        if tuple(haystack[start : start + len(needle)]) == tuple(needle):
            return int(start)
    return None


def _enum_value(value: Any) -> str:
    """Return enum `.value` when present, otherwise string value."""

    return str(getattr(value, "value", value))


def _area_bucket(area: int) -> str:
    """Bucket normalized 1000-space bbox area."""

    value = int(area)
    if value < 32 * 32:
        return "tiny"
    if value < 96 * 96:
        return "small"
    if value < 256 * 256:
        return "medium"
    return "large"


def _load_match_rows(root: Path) -> dict[int, dict[str, Any]]:
    """Load evaluator match rows keyed by line/image index."""

    path = root / "eval" / "matches.jsonl"
    if not path.exists():
        path = root / "eval" / "matches@0.30.jsonl"
    if not path.exists():
        return {}
    rows = _read_jsonl(path)
    return {int(index): row for index, row in enumerate(rows)}


def _valid_confidence_objects(
    confidence_row: Mapping[str, Any],
    trace_row: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    """Return pred objects with trustworthy four-coordinate token spans."""

    # validating selected-token trace alignment
    tokens = list(trace_row.get("generated_token_text") or [])
    out: list[Mapping[str, Any]] = []
    for obj in confidence_row.get("objects", []) or []:
        if not isinstance(obj, Mapping):
            continue
        details = obj.get("confidence_details")
        if not isinstance(details, Mapping):
            continue
        indices = details.get("matched_token_indices")
        if (
            obj.get("kept") is not True
            or details.get("failure_reason") is not None
            or int(details.get("ambiguous_matches") or 0) != 0
            or not isinstance(indices, Sequence)
            or isinstance(indices, (str, bytes))
            or len(indices) != 4
        ):
            continue
        coord_indices = [int(index) for index in indices]
        if any(index < 0 or index >= len(tokens) for index in coord_indices):
            continue
        if any(COORD_TOKEN_RE.fullmatch(str(tokens[index])) is None for index in coord_indices):
            continue
        out.append(obj)
    return out


def _generated_prefix_text_from_confidence(
    trace_row: Mapping[str, Any],
    confidence_objects: Sequence[Mapping[str, Any]],
    *,
    depth: int,
) -> str:
    """Cut generated prefix text at complete-object token boundaries."""

    # returning empty prefix for depth zero
    if int(depth) <= 0 or not confidence_objects:
        return ""
    tokens = [str(item) for item in trace_row.get("generated_token_text", [])]
    object_index = min(int(depth), len(confidence_objects)) - 1
    details = confidence_objects[object_index].get("confidence_details")
    if not isinstance(details, Mapping):
        return ""
    indices = details.get("matched_token_indices")
    if not isinstance(indices, Sequence) or isinstance(indices, (str, bytes)) or not indices:
        return ""
    stop = int(indices[-1]) + 1
    if stop < len(tokens) and tokens[stop] == "\n":
        stop += 1
    prefix_pieces: list[str] = []
    for piece in tokens[:stop]:
        if IM_END_TOKEN in piece or "<|endoftext|>" in piece:
            break
        prefix_pieces.append(piece)
    return "".join(prefix_pieces).strip()


def _prefix_text_at_depth(prefix_text: str, *, depth: int) -> str:
    """Return exactly ``depth`` complete generated compact rows.

    Confidence rows may skip ambiguous generated objects, so cutting at the
    ``depth``-th valid confidence object can include extra raw generated rows.
    The self-prefix analysis defines depth by compact-row count, so enforce the
    row-count contract after the token-boundary cut.
    """

    # trimming complete generated rows to the requested prefix depth
    if int(depth) <= 0:
        return ""
    return cut_complete_compact_rows(str(prefix_text), max_rows=int(depth))


def _prefix_quality_for_depth(
    match_row: Mapping[str, Any] | None,
    *,
    prefix_rows: Sequence[str],
    depth: int,
) -> str:
    """Classify generated-prefix quality from match artifacts."""

    if int(depth) <= 0:
        return "empty_prefix"
    compact_rows = [str(row).strip() for row in prefix_rows if str(row).strip()]
    if len(compact_rows) != len(set(compact_rows)):
        return "duplicate_prefix"
    if not match_row:
        return "drift_prefix"
    matched_pred = {int(item.get("pred_idx")) for item in match_row.get("matches", []) if "pred_idx" in item}
    if all(index in matched_pred for index in range(int(depth))):
        return "clean_prefix"
    return "fp_prefix"


def _run_self_rollout_regeneration(config_path: Path) -> None:
    """Regenerate self-rollout artifacts through the standard infer runner."""

    import subprocess

    subprocess.run(
        [
            "conda",
            "run",
            "-n",
            "ms",
            "python",
            "scripts/run_infer.py",
            "--config",
            str(config_path),
        ],
        cwd=str(REPO_ROOT),
        check=True,
    )


def _plot_logit_rows(rows: Sequence[Mapping[str, Any]], *, plot_dir: Path) -> list[str]:
    """Render summary plots for scored coordinate rows."""

    import matplotlib.pyplot as plt

    plot_paths: list[str] = []
    if not rows:
        return plot_paths

    # mass-by-radius curves per slot
    radii = sorted(
        {
            int(key.split("_")[-1])
            for row in rows
            for key in row
            if str(key).startswith("mass_at_")
        }
    )
    plt.figure(figsize=(7, 4))
    for slot in SLOT_NAMES:
        slot_rows = [row for row in rows if row.get("slot") == slot]
        if not slot_rows:
            continue
        ys = [float(np.mean([float(row.get(f"mass_at_{radius}", 0.0)) for row in slot_rows])) for radius in radii]
        plt.plot(radii, ys, marker="o", label=slot)
    plt.xlabel("radius around GT bin")
    plt.ylabel("conditional mass")
    plt.xscale("symlog", linthresh=1)
    plt.ylim(0, 1.02)
    plt.legend()
    plt.tight_layout()
    out = plot_dir / "locality_mass_by_radius__slot.png"
    plt.savefig(out, dpi=180)
    plt.close()
    plot_paths.append(str(out))

    # entropy versus top-1 distance
    plt.figure(figsize=(6, 4))
    colors = [SLOT_NAMES.index(str(row.get("slot"))) if row.get("slot") in SLOT_NAMES else 0 for row in rows]
    plt.scatter(
        [float(row.get("top1_distance", 0.0)) for row in rows],
        [float(row.get("entropy", 0.0)) for row in rows],
        c=colors,
        s=12,
        cmap="tab10",
    )
    plt.xlabel("top-1 distance from GT")
    plt.ylabel("conditional entropy")
    plt.tight_layout()
    out = plot_dir / "entropy_vs_top1_distance.png"
    plt.savefig(out, dpi=180)
    plt.close()
    plot_paths.append(str(out))

    # slot peak-distance histograms
    plt.figure(figsize=(7, 4))
    for slot in SLOT_NAMES:
        values = [float(row.get("top1_distance", 0.0)) for row in rows if row.get("slot") == slot]
        if values:
            plt.hist(values, bins=32, alpha=0.45, label=slot)
    plt.xlabel("top-1 distance from GT")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    out = plot_dir / "peak_distance_hist__slot.png"
    plt.savefig(out, dpi=180)
    plt.close()
    plot_paths.append(str(out))

    # teacher versus self prefix shift when both exist
    by_condition = defaultdict(list)
    for row in rows:
        by_condition[str(row.get("prefix_condition"))].append(row)
    if "teacher_forced" in by_condition and "self_prefix" in by_condition:
        labels = []
        teacher = []
        self_values = []
        for slot in SLOT_NAMES:
            labels.append(slot)
            teacher.append(float(np.mean([float(row.get("top1_distance", 0.0)) for row in by_condition["teacher_forced"] if row.get("slot") == slot] or [0.0])))
            self_values.append(float(np.mean([float(row.get("top1_distance", 0.0)) for row in by_condition["self_prefix"] if row.get("slot") == slot] or [0.0])))
        x = np.arange(len(labels))
        plt.figure(figsize=(6, 4))
        plt.bar(x - 0.18, teacher, width=0.36, label="teacher")
        plt.bar(x + 0.18, self_values, width=0.36, label="self")
        plt.xticks(x, labels)
        plt.ylabel("mean top-1 distance")
        plt.legend()
        plt.tight_layout()
        out = plot_dir / "teacher_vs_self_delta__slot.png"
        plt.savefig(out, dpi=180)
        plt.close()
        plot_paths.append(str(out))

    return plot_paths


def _summarize_coord_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate coordinate rows by prefix condition and slot."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            f"{row.get('prefix_condition', 'unknown')}::"
            f"{row.get('target_kind', 'unknown')}::"
            f"{row.get('slot', 'unknown')}"
        )
        grouped[key].append(row)
    summary: dict[str, Any] = {}
    for key, items in grouped.items():
        summary[key] = {
            "count": len(items),
            "mass_at_4_mean": _mean_key(items, "mass_at_4"),
            "mass_at_8_mean": _mean_key(items, "mass_at_8"),
            "p_gt_cond_mean": _mean_key(items, "p_gt_cond"),
            "coord_vocab_mass_mean": _mean_key(items, "coord_vocab_mass"),
            "top1_distance_mean": _mean_key(items, "top1_distance"),
            "entropy_mean": _mean_key(items, "entropy"),
            "gt_top1_rate": float(np.mean([1.0 if int(item.get("rank_gt", 0)) == 1 else 0.0 for item in items])),
            "shape_counts": dict(Counter(str(item.get("shape_label")) for item in items)),
        }
    return summary


def _summarize_by_fields(
    rows: Sequence[Mapping[str, Any]],
    *,
    fields: Sequence[str],
    filter_values: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate coordinate rows by selected metadata fields."""

    # filtering and grouping rows
    filter_values = filter_values or {}
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if any(row.get(key) != value for key, value in filter_values.items()):
            continue
        key = "::".join(str(row.get(field, "unknown")) for field in fields)
        grouped[key].append(row)

    # computing a compact metric panel
    out: dict[str, Any] = {}
    for key, items in sorted(grouped.items()):
        out[key] = {
            "count": len(items),
            "mass_at_4_mean": _mean_key(items, "mass_at_4"),
            "mass_at_8_mean": _mean_key(items, "mass_at_8"),
            "mass_at_16_mean": _mean_key(items, "mass_at_16"),
            "p_gt_cond_mean": _mean_key(items, "p_gt_cond"),
            "top1_distance_mean": _mean_key(items, "top1_distance"),
            "top1_distance_median": _median_key(items, "top1_distance"),
            "rank1_rate": float(np.mean([int(item.get("rank_gt", 0)) == 1 for item in items])),
            "rank5_rate": float(np.mean([int(item.get("rank_gt", 0)) <= 5 for item in items])),
            "entropy_mean": _mean_key(items, "entropy"),
            "coord_vocab_mass_mean": _mean_key(items, "coord_vocab_mass"),
            "shape_counts": dict(Counter(str(item.get("shape_label")) for item in items)),
        }
    return out


def _markdown_summary_bullets(summary: Mapping[str, Any]) -> list[str]:
    """Render compact metric summaries as Markdown bullets."""

    # keeping the report concise while preserving decision metrics
    lines: list[str] = []
    for key, row in sorted(summary.items()):
        lines.append(
            f"- `{key}`: n={row.get('count')}, mass@4={_fmt(row.get('mass_at_4_mean'))}, "
            f"mass@8={_fmt(row.get('mass_at_8_mean'))}, "
            f"top1 mean/median={_fmt(row.get('top1_distance_mean'))}/{_fmt(row.get('top1_distance_median'))}, "
            f"rank1={_fmt(row.get('rank1_rate'))}, rank5={_fmt(row.get('rank5_rate'))}"
        )
    return lines


def _mean_key(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    """Mean numeric row value for a key."""

    values = [float(row[key]) for row in rows if key in row and row[key] is not None]
    return float(np.mean(values)) if values else None


def _median_key(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    """Median numeric row value for a key."""

    values = [float(row[key]) for row in rows if key in row and row[key] is not None]
    return float(np.median(values)) if values else None


def _fmt(value: Any) -> str:
    """Format report floats compactly."""

    if value is None:
        return "n/a"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def _decision_hint(grouped: Mapping[str, Any], embedding_summary: Mapping[str, Any]) -> str:
    """Produce a conservative soft-CE decision hint from current artifacts."""

    # combining teacher-forced and embedding evidence
    teacher_hard_rows = [
        value
        for key, value in grouped.items()
        if str(key).startswith("teacher_forced::hard_ce::")
    ]
    teacher_rows = teacher_hard_rows or [
        value for key, value in grouped.items() if str(key).startswith("teacher_forced::")
    ]
    if teacher_rows:
        mass4 = float(np.mean([float(row.get("mass_at_4_mean") or 0.0) for row in teacher_rows]))
        topdist = float(np.mean([float(row.get("top1_distance_mean") or 999.0) for row in teacher_rows]))
    else:
        mass4 = 0.0
        topdist = 999.0
    surfaces = embedding_summary.get("surfaces", {}) if isinstance(embedding_summary, Mapping) else {}
    eff = surfaces.get("effective_output") if isinstance(surfaces, Mapping) else None
    emb_recall = None
    if isinstance(eff, Mapping):
        emb_recall = eff.get("knn_radius_recall", {}).get("radius_4")
    emb_recall_f = float(emb_recall) if emb_recall is not None else 0.0

    if mass4 >= 0.60 and topdist <= 4.0 and emb_recall_f >= 0.65:
        return "Evidence currently leans toward pausing or deprecating soft CE in ordinary clean-prefix regimes, pending self-prefix and crowded-case confirmation."
    if mass4 >= 0.45 and emb_recall_f >= 0.55:
        return "Evidence currently leans toward modifying soft CE or focusing on exposure/prefix dynamics rather than treating static coordinate adjacency as absent."
    if mass4 < 0.25 and emb_recall_f < 0.35:
        return "Evidence currently supports continued investment in soft CE or another explicit ordinal/geometric objective."
    return "Evidence is mixed; keep teacher-forced, self-prefix, and embedding surfaces separated before making a soft-CE deprecation decision."
