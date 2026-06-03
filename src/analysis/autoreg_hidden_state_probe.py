from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


LANE_D_COMPACT_ROLES = (
    "prompt_end",
    "row_start",
    "desc_end",
    "box_start",
    "pre_x1",
    "post_x1",
    "post_y1",
    "row_end_or_separator",
    "final_generated_prefix_state",
)

LANE_D_SEPARATOR_KINDS = (
    "none_marker_delimited",
    "newline",
)

LANE_D_RENDER_SOURCES = (
    "strict_compact_full",
    "lane_c_forced_continuation",
    "generated_prefix_replay",
)

LANE_D_PREFIX_STATE_KINDS = (
    "not_applicable",
    "empty_prefix_prompt_end",
    "teacher_forced_prefix_boundary",
    "generated_prefix_boundary",
    "partial_row",
)

LANE_D_GENERATED_TOKEN_ROLES = {
    "desc_end",
    "box_start",
    "pre_x1",
    "post_x1",
    "post_y1",
}

DEFAULT_LANE_D_LAYER_GROUPS: Mapping[str, tuple[int, ...]] = {
    "early": (0, 1),
    "middle": (12, 13),
    "late": (24, 25, 26, 27),
    "last": (-4, -3, -2, -1),
}

LANE_D_STAGES = (
    "select_cases",
    "position_inventory",
    "hidden_states",
    "patching",
    "merge",
    "report",
)

LANE_D_MERGE_DOMAINS = (
    "selected_cases",
    "position_inventory",
    "probe_rows",
    "patch_rows",
)

LANE_D_MERGE_JSONL_FILES: Mapping[str, str] = {
    "selected_cases": "selected_cases.jsonl",
    "position_inventory": "position_inventory.jsonl",
    "probe_rows": "probe_rows.jsonl",
    "patch_rows": "patch_rows.jsonl",
}

LANE_D_SHARD_SUMMARY_FILE = "summary.json"

LANE_D_DUPLICATE_KEY_DOMAINS: Mapping[str, tuple[str, ...]] = {
    "selected_cases": (
        "source_line_idx",
        "case_id",
    ),
    "position_inventory": (
        "source_line_idx",
        "case_id",
        "prefix_mode",
        "prefix_depth",
        "render_source",
        "role",
    ),
    "probe_rows": (
        "source_line_idx",
        "case_id",
        "prefix_mode",
        "prefix_depth",
        "task",
        "role",
        "layer_group",
        "model_layer",
        "slot",
        "target_label_source",
    ),
    "patch_rows": (
        "source_line_idx",
        "case_id",
        "prefix_mode",
        "prefix_depth",
        "patch_policy",
        "donor_policy",
        "donor_case_id",
        "role",
        "layer_group",
        "model_layer",
        "slot",
    ),
}

LANE_D_SHARED_REPRODUCIBILITY_FIELDS = (
    "config_path",
    "config_sha256",
    "checkpoint",
    "artifact_root",
    "layer_groups",
    "batch_size",
)

_LANE_D_DUPLICATE_KEY_NONNEGATIVE_INT_FIELDS = {
    "source_line_idx",
    "prefix_depth",
}

_LANE_D_DUPLICATE_KEY_INT_FIELDS = {
    "model_layer",
}

_LANE_D_DUPLICATE_KEY_STRING_FIELDS = {
    "case_id",
    "prefix_mode",
    "render_source",
    "role",
    "task",
    "layer_group",
    "slot",
    "target_label_source",
    "patch_policy",
    "donor_policy",
}

_LANE_D_SHARD_DIR_PATTERN = re.compile(r"^shard_(\d{3})-of-(\d{3})$")
_LANE_D_COORD_TOKEN_RE = re.compile(r"<\|coord_\d+\|>")
_LANE_D_COORD_VALUE_RE = re.compile(r"<\|coord_(\d+)\|>")

LANE_D_OBJECT_REF_START_TOKEN = "<|object_ref_start|>"
LANE_D_BOX_START_TOKEN = "<|box_start|>"
DEFAULT_LANE_C_STUDY_CONFIG = Path(
    "/data/CoordExp/configs/analysis/hard_ce_coord_logit_locality/"
    "ckpt3664_lane_c_val200.yaml"
)

_LANE_D_ROLE_SLOTS = {
    "pre_x1": "x1",
    "post_x1": "y1",
    "post_y1": "x2",
}


@dataclass(frozen=True)
class LaneDPaths:
    artifact_root: Path
    checkpoint: Path
    dataset_jsonl: Path
    self_rollout_root: Path
    lane_a_rollout_root: Path
    lane_c_per_case: Path
    lane_c_study_config: Path = DEFAULT_LANE_C_STUDY_CONFIG


@dataclass(frozen=True)
class LaneDSelectionConfig:
    max_cases: int = 512


@dataclass(frozen=True)
class LaneDPositionsConfig:
    roles: tuple[str, ...] = LANE_D_COMPACT_ROLES
    layer_groups: Mapping[str, tuple[int, ...]] = field(
        default_factory=lambda: dict(DEFAULT_LANE_D_LAYER_GROUPS)
    )


@dataclass(frozen=True)
class LaneDExecutionConfig:
    batch_size: int = 1
    enable_x1_logit_lens: bool = False
    x1_logit_lens_roles: tuple[str, ...] = ("desc_end", "box_start", "pre_x1")
    x1_logit_lens_top_k: int = 5
    enable_coord_slot_logit_lens: bool = False
    coord_slot_logit_lens_roles: tuple[str, ...] = ("pre_x1", "post_x1", "post_y1")
    coord_slot_logit_lens_top_k: int = 5


@dataclass(frozen=True)
class LaneDConfig:
    config_path: Path
    paths: LaneDPaths
    selection: LaneDSelectionConfig = field(default_factory=LaneDSelectionConfig)
    positions: LaneDPositionsConfig = field(default_factory=LaneDPositionsConfig)
    execution: LaneDExecutionConfig = field(default_factory=LaneDExecutionConfig)


def lane_d_shard_label(shard_index: int, num_shards: int) -> str:
    return f"shard_{shard_index:03d}-of-{num_shards:03d}"


def normalize_lane_d_shard(
    *, shard_index: int | None, num_shards: int | None
) -> tuple[int | None, int | None, str | None]:
    if shard_index is None and num_shards is None:
        return None, None, None
    if shard_index is None or num_shards is None:
        raise ValueError("shard_index and num_shards must be provided together")
    if not _is_plain_int(num_shards) or num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if not _is_plain_int(shard_index) or shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")
    return shard_index, num_shards, lane_d_shard_label(shard_index, num_shards)


def lane_d_record_selected(
    source_line_idx: int, *, shard_index: int | None, num_shards: int | None
) -> bool:
    if not _is_nonnegative_plain_int(source_line_idx):
        raise ValueError("source_line_idx must be a nonnegative integer")
    shard_index, num_shards, _ = normalize_lane_d_shard(
        shard_index=shard_index,
        num_shards=num_shards,
    )
    if shard_index is None or num_shards is None:
        return True
    return source_line_idx % num_shards == shard_index


def validate_configured_lane_d_roles(roles: Sequence[str]) -> tuple[str, ...]:
    if isinstance(roles, (str, bytes)) or not isinstance(roles, Sequence):
        raise ValueError("positions.roles must be a sequence of compact role names")

    parsed: list[str] = []
    seen: set[str] = set()
    duplicates: list[str] = []
    unknown: list[str] = []
    expected = set(LANE_D_COMPACT_ROLES)
    for index, role in enumerate(roles):
        if not isinstance(role, str) or not role:
            raise ValueError(f"positions.roles[{index}] must be a nonempty string")
        if role in seen:
            duplicates.append(role)
        if role not in expected:
            unknown.append(role)
        seen.add(role)
        parsed.append(role)

    if unknown:
        raise ValueError(f"unknown compact role(s): {', '.join(unknown)}")
    if duplicates:
        raise ValueError(f"duplicate compact role(s): {', '.join(duplicates)}")

    missing = [role for role in LANE_D_COMPACT_ROLES if role not in seen]
    if missing:
        raise ValueError(f"missing compact roles: {', '.join(missing)}")
    if tuple(parsed) != LANE_D_COMPACT_ROLES:
        raise ValueError(
            "positions.roles must exactly match LANE_D_COMPACT_ROLES order: "
            f"{', '.join(LANE_D_COMPACT_ROLES)}"
        )
    return tuple(parsed)


def load_lane_d_config(path: str | Path) -> LaneDConfig:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{config_path} must contain a YAML mapping")

    paths_raw = _required_config_mapping(payload, "paths")
    selection_raw = _optional_config_mapping(payload, "selection")
    positions_raw = _optional_config_mapping(payload, "positions")
    execution_raw = _optional_config_mapping(payload, "execution")

    paths = LaneDPaths(
        artifact_root=_required_config_path(paths_raw, "artifact_root"),
        checkpoint=_required_config_path(paths_raw, "checkpoint"),
        dataset_jsonl=_required_config_path(paths_raw, "dataset_jsonl"),
        self_rollout_root=_required_config_path(paths_raw, "self_rollout_root"),
        lane_a_rollout_root=_required_config_path(paths_raw, "lane_a_rollout_root"),
        lane_c_per_case=_required_config_path(paths_raw, "lane_c_per_case"),
        lane_c_study_config=_optional_config_path(
            paths_raw,
            "lane_c_study_config",
            default=DEFAULT_LANE_C_STUDY_CONFIG,
        ),
    )
    selection = LaneDSelectionConfig(
        max_cases=_config_nonnegative_int(
            selection_raw.get("max_cases", LaneDSelectionConfig.max_cases),
            "selection.max_cases",
        )
    )
    positions = LaneDPositionsConfig(
        roles=_parse_configured_lane_d_roles(
            positions_raw.get("roles", LANE_D_COMPACT_ROLES)
        ),
        layer_groups=_parse_lane_d_layer_groups(
            positions_raw.get("layer_groups", DEFAULT_LANE_D_LAYER_GROUPS)
        ),
    )
    execution = LaneDExecutionConfig(
        batch_size=_config_positive_int(
            execution_raw.get("batch_size", LaneDExecutionConfig.batch_size),
            "execution.batch_size",
        ),
        enable_x1_logit_lens=_config_bool(
            execution_raw.get(
                "enable_x1_logit_lens",
                LaneDExecutionConfig.enable_x1_logit_lens,
            ),
            "execution.enable_x1_logit_lens",
        ),
        x1_logit_lens_roles=_parse_lane_d_string_tuple(
            execution_raw.get(
                "x1_logit_lens_roles",
                LaneDExecutionConfig.x1_logit_lens_roles,
            ),
            "execution.x1_logit_lens_roles",
        ),
        x1_logit_lens_top_k=_config_positive_int(
            execution_raw.get(
                "x1_logit_lens_top_k",
                LaneDExecutionConfig.x1_logit_lens_top_k,
            ),
            "execution.x1_logit_lens_top_k",
        ),
        enable_coord_slot_logit_lens=_config_bool(
            execution_raw.get(
                "enable_coord_slot_logit_lens",
                LaneDExecutionConfig.enable_coord_slot_logit_lens,
            ),
            "execution.enable_coord_slot_logit_lens",
        ),
        coord_slot_logit_lens_roles=_parse_lane_d_string_tuple(
            execution_raw.get(
                "coord_slot_logit_lens_roles",
                LaneDExecutionConfig.coord_slot_logit_lens_roles,
            ),
            "execution.coord_slot_logit_lens_roles",
        ),
        coord_slot_logit_lens_top_k=_config_positive_int(
            execution_raw.get(
                "coord_slot_logit_lens_top_k",
                LaneDExecutionConfig.coord_slot_logit_lens_top_k,
            ),
            "execution.coord_slot_logit_lens_top_k",
        ),
    )
    return LaneDConfig(
        config_path=config_path,
        paths=paths,
        selection=selection,
        positions=positions,
        execution=execution,
    )


def build_lane_d_dry_run_plan(
    config: LaneDConfig,
    stages: Sequence[str],
    shard_index: int | None = None,
    num_shards: int | None = None,
    merge_shards: bool = False,
) -> dict[str, Any]:
    normalized_stages = _normalize_lane_d_stages(stages)
    selected_shards = [
        _lane_d_dry_run_shard_entry(config.paths.artifact_root, item)
        for item in _lane_d_selected_shards(
            shard_index=shard_index,
            num_shards=num_shards,
        )
    ]
    return {
        "config_path": str(config.config_path),
        "artifact_root": str(config.paths.artifact_root),
        "stages": list(normalized_stages),
        "dry_run": True,
        "merge_shards": bool(merge_shards),
        "selected_shards": selected_shards,
    }


def lane_d_config_expected_merge_metadata(config: LaneDConfig) -> dict[str, Any]:
    """Return reproducibility metadata that merged shards must match."""

    return {
        "config_path": str(config.config_path),
        "config_sha256": _sha256_path(config.config_path),
        "checkpoint": str(config.paths.checkpoint),
        "artifact_root": str(config.paths.artifact_root),
        "layer_groups": _jsonable(config.positions.layer_groups),
        "batch_size": int(config.execution.batch_size),
    }


def validate_lane_d_position_inventory(rows: Sequence[Mapping[str, Any]]) -> None:
    roles_by_context: dict[
        tuple[int, str, str, int, str], dict[str, Mapping[str, Any]]
    ] = {}
    for row in rows:
        case_id = _required_nonempty_string(row, "case_id")
        source_line_idx = _required_nonnegative_int(
            row,
            "source_line_idx",
            case_id=case_id,
        )
        prefix_mode = _required_nonempty_string(row, "prefix_mode", case_id=case_id)
        prefix_depth = _required_nonnegative_int(row, "prefix_depth", case_id=case_id)
        _required_json_scalar(row, "prefix_quality", case_id=case_id)
        render_source = _required_nonempty_string(
            row,
            "render_source",
            case_id=case_id,
        )
        if render_source not in LANE_D_RENDER_SOURCES:
            raise ValueError(
                "render_source must be one of "
                f"{', '.join(LANE_D_RENDER_SOURCES)} for {case_id}"
            )

        role = _required_nonempty_string(row, "role", case_id=case_id)
        if role not in LANE_D_COMPACT_ROLES:
            raise ValueError(f"unknown compact role for {case_id}: {role}")
        context_key = (
            source_line_idx,
            case_id,
            prefix_mode,
            prefix_depth,
            render_source,
        )
        context_roles = roles_by_context.setdefault(context_key, {})
        if role in context_roles:
            raise ValueError(f"duplicate compact role for {case_id}: {role}")

        absolute = _required_nonnegative_int(
            row,
            "absolute_token_index",
            case_id=case_id,
            role=role,
        )
        prediction = _required_value(
            row,
            "prediction_token_index",
            case_id=case_id,
            role=role,
        )
        if prediction is not None and not _is_nonnegative_plain_int(prediction):
            raise ValueError(
                "prediction_token_index must be null or a nonnegative integer "
                f"for {case_id}: {role}"
            )
        if role in LANE_D_GENERATED_TOKEN_ROLES and prediction != absolute - 1:
            raise ValueError(
                "prediction_token_index must equal absolute_token_index - 1 "
                f"for generated role {case_id}: {role}"
            )
        prefix_state_kind = _required_nonempty_string(
            row,
            "prefix_state_kind",
            case_id=case_id,
            role=role,
        )
        if prefix_state_kind not in LANE_D_PREFIX_STATE_KINDS:
            raise ValueError(
                "prefix_state_kind must be one of "
                f"{', '.join(LANE_D_PREFIX_STATE_KINDS)} for {case_id}: {role}"
            )
        if role == "final_generated_prefix_state":
            expected_prefix_state_kind: str | None = None
            if prefix_depth == 0:
                expected_prefix_state_kind = "empty_prefix_prompt_end"
            elif prefix_mode == "teacher_forced":
                expected_prefix_state_kind = "teacher_forced_prefix_boundary"

            if expected_prefix_state_kind is not None:
                if prefix_state_kind != expected_prefix_state_kind:
                    raise ValueError(
                        "prefix_state_kind must be "
                        f"{expected_prefix_state_kind} for {case_id}: {role}"
                    )
            elif prefix_state_kind not in {"generated_prefix_boundary", "partial_row"}:
                raise ValueError(
                    "prefix_state_kind must identify a generated prefix boundary "
                    f"for {case_id}: {role}"
                )
        elif prefix_state_kind != "not_applicable":
            raise ValueError(
                "prefix_state_kind must be not_applicable for ordinary role "
                f"{case_id}: {role}"
            )

        relative = _required_value(
            row,
            "assistant_relative_token_index",
            case_id=case_id,
            role=role,
        )
        if role == "prompt_end":
            if relative is not None:
                raise ValueError("prompt_end assistant_relative_token_index must be null")
        elif (
            role == "final_generated_prefix_state"
            and prefix_state_kind == "empty_prefix_prompt_end"
        ):
            if relative is not None:
                raise ValueError(
                    "final_generated_prefix_state assistant_relative_token_index "
                    "must be null for empty_prefix_prompt_end"
                )
        elif not _is_nonnegative_plain_int(relative):
            raise ValueError(
                "assistant_relative_token_index must be a nonnegative integer "
                f"for {case_id}: {role}"
            )
        _required_nonnegative_int(
            row,
            "assistant_start_token_index",
            case_id=case_id,
            role=role,
        )
        _required_nonempty_string(row, "token_text", case_id=case_id, role=role)
        shard_index = _required_nonnegative_int(
            row,
            "shard_index",
            case_id=case_id,
            role=role,
        )
        num_shards = _required_positive_int(
            row,
            "num_shards",
            case_id=case_id,
            role=role,
        )
        shard_label = _required_nonempty_string(
            row,
            "shard_label",
            case_id=case_id,
            role=role,
        )
        expected_shard_label = lane_d_shard_label(shard_index, num_shards)
        if shard_label != expected_shard_label:
            raise ValueError(
                f"shard_label must be {expected_shard_label} for {case_id}: {role}"
            )
        separator_kind = _required_nonempty_string(
            row,
            "separator_kind",
            case_id=case_id,
            role=role,
        )
        if separator_kind not in LANE_D_SEPARATOR_KINDS:
            raise ValueError(
                "separator_kind must be one of "
                f"{', '.join(LANE_D_SEPARATOR_KINDS)} for {case_id}: {role}"
            )
        if (
            render_source == "strict_compact_full"
            and prefix_mode == "teacher_forced"
            and separator_kind != "none_marker_delimited"
        ):
            raise ValueError(
                "separator_kind must be none_marker_delimited for strict "
                f"teacher_forced compact-full rows for {case_id}: {role}"
            )

        context_roles[role] = row

    for source_line_idx, case_id, prefix_mode, prefix_depth, render_source in roles_by_context:
        context_roles = roles_by_context[
            (source_line_idx, case_id, prefix_mode, prefix_depth, render_source)
        ]
        missing = [role for role in LANE_D_COMPACT_ROLES if role not in context_roles]
        if missing:
            raise ValueError(
                f"case {case_id} missing compact roles for "
                f"source_line_idx={source_line_idx}, prefix_mode={prefix_mode}, "
                f"prefix_depth={prefix_depth}, render_source={render_source}: "
                f"{', '.join(missing)}"
            )


def build_lane_d_position_inventory_for_prepared_example(
    example: Any,
    selected_case: Mapping[str, Any],
    tokenizer: Any,
    *,
    shard_index: int,
    num_shards: int,
    shard_label: str,
) -> list[dict[str, Any]]:
    """Build Lane-D compact-role positions from one Lane-C forced continuation."""

    normalized_shard_index, normalized_num_shards, expected_shard_label = (
        normalize_lane_d_shard(shard_index=shard_index, num_shards=num_shards)
    )
    if (
        normalized_shard_index is None
        or normalized_num_shards is None
        or expected_shard_label is None
    ):
        raise ValueError("shard_index and num_shards must be provided")
    if shard_label != expected_shard_label:
        raise ValueError(f"shard_label must be {expected_shard_label}")

    case_id = _required_nonempty_string(selected_case, "case_id")
    example_case_id = _lane_d_example_case_id(example)
    if example_case_id != case_id:
        raise ValueError(
            f"Lane C example case_id {example_case_id!r} does not match {case_id!r}"
        )
    source_line_idx = _required_nonnegative_int(
        selected_case,
        "source_line_idx",
        case_id=case_id,
    )
    prefix_mode = _required_nonempty_string(
        selected_case,
        "prefix_mode",
        case_id=case_id,
    )
    prefix_depth = _required_nonnegative_int(
        selected_case,
        "prefix_depth",
        case_id=case_id,
    )
    prefix_quality = selected_case.get(
        "prefix_quality",
        getattr(example, "prefix_quality", None),
    )

    assistant_text = str(getattr(example, "assistant_text"))
    assistant_ids = _lane_d_token_ids(tokenizer, assistant_text)
    if not assistant_ids:
        raise ValueError(f"Lane C example assistant_text is empty for {case_id}")
    full_input_ids = tuple(int(item) for item in getattr(example, "full_input_ids"))
    assistant_start = _lane_d_find_subsequence(full_input_ids, assistant_ids)
    if assistant_start is None:
        raise ValueError(f"assistant token subsequence not found for {case_id}")
    duplicate_assistant_start = _lane_d_find_subsequence_after(
        full_input_ids,
        assistant_ids,
        assistant_start + 1,
    )
    if duplicate_assistant_start is not None:
        raise ValueError(f"assistant token subsequence is not unique for {case_id}")
    if assistant_start <= 0:
        raise ValueError(f"assistant_start must leave a prompt token for {case_id}")

    target_row = assistant_text.splitlines()[-1]
    target_row_ids = _lane_d_token_ids(tokenizer, target_row)
    if not target_row_ids:
        raise ValueError(f"Lane C target row is empty for {case_id}")
    row_start_relative = _lane_d_find_last_subsequence(assistant_ids, target_row_ids)
    if row_start_relative is None:
        raise ValueError(f"target row token subsequence not found for {case_id}")
    row_start_absolute = assistant_start + row_start_relative
    prompt_end_absolute = assistant_start - 1

    box_start_ids = _lane_d_token_ids(tokenizer, LANE_D_BOX_START_TOKEN)
    if not box_start_ids:
        raise ValueError(f"{LANE_D_BOX_START_TOKEN} tokenized to no ids for {case_id}")
    box_start_in_row = _lane_d_find_subsequence(target_row_ids, box_start_ids)
    if box_start_in_row is None:
        raise ValueError(f"{LANE_D_BOX_START_TOKEN} not found in target row for {case_id}")
    box_start_relative = row_start_relative + box_start_in_row
    box_marker_end_relative = box_start_relative + len(box_start_ids) - 1

    coord_tokens = _LANE_D_COORD_TOKEN_RE.findall(target_row)
    if len(coord_tokens) != 4:
        raise ValueError(f"target row must contain exactly four coord tokens for {case_id}")
    coord_relatives: list[int] = []
    search_start = 0
    for coord_token in coord_tokens:
        coord_ids = _lane_d_token_ids(tokenizer, coord_token)
        coord_start = _lane_d_find_subsequence_after(
            target_row_ids,
            coord_ids,
            search_start,
        )
        if coord_start is None:
            raise ValueError(f"{coord_token} token subsequence not found for {case_id}")
        coord_relatives.append(row_start_relative + coord_start)
        search_start = coord_start + max(1, len(coord_ids))
    coord_absolutes = tuple(assistant_start + item for item in coord_relatives)
    lane_c_coord_positions = getattr(example, "assistant_coord_positions", None)
    if lane_c_coord_positions is not None:
        expected_coord_positions = tuple(int(item) for item in lane_c_coord_positions)
        if expected_coord_positions != coord_absolutes:
            raise ValueError(
                "Lane D coord positions do not match Lane C prepared example "
                f"for {case_id}"
            )

    row_end_relative = row_start_relative + len(target_row_ids)
    if row_end_relative < len(assistant_ids):
        row_end_or_separator_absolute = assistant_start + row_end_relative
    else:
        row_end_or_separator_absolute = (
            assistant_start + row_start_relative + len(target_row_ids) - 1
        )

    role_absolute: dict[str, int] = {
        "prompt_end": prompt_end_absolute,
        "row_start": row_start_absolute,
        "desc_end": assistant_start + box_start_relative,
        # box_start is the state after the complete box marker, i.e. the same
        # prediction boundary as x1. This keeps multi-token box markers honest.
        "box_start": coord_absolutes[0],
        "pre_x1": coord_absolutes[0],
        "post_x1": coord_absolutes[1],
        "post_y1": coord_absolutes[2],
        "row_end_or_separator": row_end_or_separator_absolute,
    }
    separator_kind = "newline" if "\n" in assistant_text else "none_marker_delimited"
    row_end_source_kind = (
        "next_token_after_target_row"
        if row_end_relative < len(assistant_ids)
        else "last_target_row_token"
    )

    if prefix_depth == 0:
        final_absolute = prompt_end_absolute
        final_relative: int | None = None
        final_prefix_state_kind = "empty_prefix_prompt_end"
    else:
        if row_start_relative <= 0:
            raise ValueError(
                f"prefix_depth={prefix_depth} has no boundary before target row for {case_id}"
            )
        final_absolute = row_start_absolute - 1
        final_relative = final_absolute - assistant_start
        final_prefix_state_kind = (
            "teacher_forced_prefix_boundary"
            if prefix_mode == "teacher_forced"
            else "generated_prefix_boundary"
        )
    role_absolute["final_generated_prefix_state"] = final_absolute

    rows: list[dict[str, Any]] = []
    for role in LANE_D_COMPACT_ROLES:
        absolute = role_absolute[role]
        prediction = absolute - 1 if role in LANE_D_GENERATED_TOKEN_ROLES else None
        if role == "prompt_end":
            assistant_relative: int | None = None
        elif role == "final_generated_prefix_state":
            assistant_relative = final_relative
        else:
            assistant_relative = absolute - assistant_start
        prefix_state_kind = (
            final_prefix_state_kind
            if role == "final_generated_prefix_state"
            else "not_applicable"
        )
        rows.append(
            {
                "case_id": case_id,
                "source_line_idx": source_line_idx,
                "prefix_mode": prefix_mode,
                "prefix_depth": prefix_depth,
                "prefix_quality": prefix_quality,
                "render_source": "lane_c_forced_continuation",
                "role": role,
                "absolute_token_index": absolute,
                "prediction_token_index": prediction,
                "assistant_relative_token_index": assistant_relative,
                "assistant_start_token_index": assistant_start,
                "prefix_state_kind": prefix_state_kind,
                "shard_index": normalized_shard_index,
                "num_shards": normalized_num_shards,
                "shard_label": shard_label,
                "separator_kind": separator_kind,
                "row_end_source_kind": row_end_source_kind,
                "token_text": _lane_d_decode_single_token(
                    tokenizer,
                    full_input_ids,
                    absolute,
                ),
            }
        )
    validate_lane_d_position_inventory(rows)
    return rows


def filter_lane_d_examples_for_selected_cases(
    examples: Sequence[Any],
    selected_cases: Sequence[Mapping[str, Any]],
) -> list[tuple[dict[str, Any], Any]]:
    """Return Lane-C examples aligned to materialized Lane-D selected cases."""

    selected_by_case_id: list[tuple[str, dict[str, Any]]] = []
    seen_selected: set[str] = set()
    for selected_case in selected_cases:
        case_id = _required_nonempty_string(selected_case, "case_id")
        if case_id in seen_selected:
            raise ValueError(f"duplicate selected case_id: {case_id}")
        seen_selected.add(case_id)
        selected_by_case_id.append((case_id, dict(selected_case)))

    examples_by_case_id: dict[str, Any] = {}
    duplicate_examples: set[str] = set()
    for example in examples:
        case_id = _lane_d_example_case_id(example)
        if case_id in examples_by_case_id:
            duplicate_examples.add(case_id)
        examples_by_case_id[case_id] = example
    if duplicate_examples:
        raise ValueError(
            "duplicate Lane C generated example case_id(s): "
            f"{', '.join(sorted(duplicate_examples))}"
        )

    missing = [
        case_id for case_id, _ in selected_by_case_id if case_id not in examples_by_case_id
    ]
    if missing:
        raise ValueError(
            "selected case_id(s) missing from Lane C generated examples: "
            f"{', '.join(missing)}"
        )
    return [
        (selected_case, examples_by_case_id[case_id])
        for case_id, selected_case in selected_by_case_id
    ]


def _lane_d_target_ledger(
    selected_case: Mapping[str, Any],
    example: Any,
) -> dict[str, Any]:
    metadata = getattr(example, "lane_c_metadata", None)
    metadata = metadata if isinstance(metadata, Mapping) else {}
    coord_bins = _lane_d_target_coord_bins(example)
    gt_bins = metadata.get("gt_bins_by_index")
    remaining = metadata.get("remaining_gt_indices")
    object_count = len(gt_bins) if isinstance(gt_bins, Mapping) else selected_case.get("object_count")
    remaining_count = (
        len(remaining)
        if isinstance(remaining, Sequence) and not isinstance(remaining, (str, bytes))
        else selected_case.get("remaining_count")
    )
    return {
        "object_count": _required_nonnegative_int(
            {"object_count": object_count},
            "object_count",
            case_id=str(selected_case.get("case_id")),
        ),
        "remaining_count": _required_nonnegative_int(
            {"remaining_count": remaining_count},
            "remaining_count",
            case_id=str(selected_case.get("case_id")),
        ),
        "intended_target_gt_idx": _required_nonnegative_int(
            selected_case,
            "intended_target_gt_idx",
            case_id=str(selected_case.get("case_id")),
        ),
        "prefix_quality": _required_value(
            selected_case,
            "prefix_quality",
            case_id=str(selected_case.get("case_id")),
        ),
        "x1_top_peak_attribution": _required_nonempty_string(
            selected_case,
            "x1_top_peak_attribution",
            case_id=str(selected_case.get("case_id")),
        ),
        "x1_target_rank": _required_nonnegative_int(
            selected_case,
            "x1_target_rank",
            case_id=str(selected_case.get("case_id")),
        ),
        "target_coord_bins_xyxy": list(coord_bins) if coord_bins is not None else None,
        "target_x1_bin": coord_bins[0] if coord_bins is not None else None,
        "target_y1_bin": coord_bins[1] if coord_bins is not None else None,
        "target_x2_bin": coord_bins[2] if coord_bins is not None else None,
        "target_y2_bin": coord_bins[3] if coord_bins is not None else None,
    }


def select_lane_d_cases(
    lane_c_per_case: str | Path,
    *,
    max_cases: int | None,
    shard_index: int | None,
    num_shards: int | None,
) -> list[dict[str, Any]]:
    normalize_lane_d_shard(shard_index=shard_index, num_shards=num_shards)
    if max_cases is not None and (not _is_plain_int(max_cases) or max_cases < 0):
        raise ValueError("max_cases must be null or a nonnegative integer")

    ranked: list[tuple[tuple[int, int, str, int, str], dict[str, Any]]] = []
    for input_ordinal, row in enumerate(_read_jsonl(Path(lane_c_per_case))):
        source_line_idx = _required_lane_c_nonnegative_int(
            row,
            "source_line_idx",
            input_ordinal,
        )
        if not lane_d_record_selected(
            source_line_idx,
            shard_index=shard_index,
            num_shards=num_shards,
        ):
            continue

        x1 = row.get("x1")
        if not isinstance(x1, Mapping):
            raise ValueError(f"per_case row {input_ordinal + 1} missing mapping x1")
        x1_top_peak_attribution = _required_lane_c_nonempty_string(
            x1,
            "top_peak_attribution",
            input_ordinal,
        )
        x1_target_rank = _required_lane_c_nonnegative_int(
            x1,
            "target_rank",
            input_ordinal,
        )
        high_priority = _is_x1_non_target_or_low_rank(
            top_peak_attribution=x1_top_peak_attribution,
            target_rank=x1_target_rank,
        )
        case_id = _required_lane_c_nonempty_string(row, "case_id", input_ordinal)
        prefix_mode = _required_lane_c_nonempty_string(row, "prefix_mode", input_ordinal)
        prefix_depth = _required_lane_c_nonnegative_int(row, "prefix_depth", input_ordinal)
        case = {
            "case_id": case_id,
            "source_line_idx": source_line_idx,
            "prefix_mode": prefix_mode,
            "prefix_depth": prefix_depth,
            "prefix_quality": row.get("prefix_quality"),
            "intended_target_gt_idx": row.get("intended_target_gt_idx"),
            "target_desc": row.get("target_desc"),
            "x1_top_peak_attribution": x1_top_peak_attribution,
            "x1_target_rank": x1_target_rank,
            "selection_reason": (
                "x1_non_target_or_low_rank" if high_priority else "lane_c_case"
            ),
        }

        priority = 0 if high_priority else 1
        sort_key = (
            priority,
            source_line_idx,
            case["prefix_mode"],
            case["prefix_depth"],
            case["case_id"],
        )
        ranked.append((sort_key, case))

    ranked.sort(key=lambda item: item[0])
    cases = [case for _, case in ranked]
    if max_cases is not None:
        return cases[:max_cases]
    return cases


def materialize_lane_d_select_cases_shard(
    config: LaneDConfig,
    shard_index: int,
    num_shards: int,
) -> dict[str, Any]:
    """Write CPU-only Lane-D select_cases shard artifacts."""

    normalized_shard_index, normalized_num_shards, shard_label = normalize_lane_d_shard(
        shard_index=shard_index,
        num_shards=num_shards,
    )
    if (
        normalized_shard_index is None
        or normalized_num_shards is None
        or shard_label is None
    ):
        raise ValueError("shard_index and num_shards must be provided")

    selected_cases = []
    for row in select_lane_d_cases(
        config.paths.lane_c_per_case,
        max_cases=config.selection.max_cases,
        shard_index=normalized_shard_index,
        num_shards=normalized_num_shards,
    ):
        selected_row = dict(row)
        selected_row.update(
            {
                "shard_index": normalized_shard_index,
                "num_shards": normalized_num_shards,
                "shard_label": shard_label,
            }
        )
        selected_cases.append(selected_row)

    row_counts = {
        "selected_cases": len(selected_cases),
        "position_inventory": 0,
        "probe_rows": 0,
        "patch_rows": 0,
    }
    selected_cases_sha256 = _sha256_bytes(_canonical_jsonl_bytes(selected_cases))
    summary: dict[str, Any] = {
        "stage": "hidden_state_probe",
        "shard_index": normalized_shard_index,
        "num_shards": normalized_num_shards,
        "shard_label": shard_label,
        "row_counts": row_counts,
        "stages_completed": ["select_cases"],
        "selected_cases_sha256": selected_cases_sha256,
        "runtime_kind": "cpu_select_cases",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_device": None,
        "base_seed": None,
        "case_seed_policy": "source_line_idx_mod_num_shards",
    }
    summary.update(lane_d_config_expected_merge_metadata(config))

    shard_dir = config.paths.artifact_root / "shards" / shard_label
    _write_canonical_jsonl(
        shard_dir / LANE_D_MERGE_JSONL_FILES["selected_cases"],
        selected_cases,
    )
    for domain in ("position_inventory", "probe_rows", "patch_rows"):
        _write_canonical_jsonl(shard_dir / LANE_D_MERGE_JSONL_FILES[domain], [])
    _write_json_object(shard_dir / LANE_D_SHARD_SUMMARY_FILE, summary)
    return summary


def materialize_lane_d_hidden_states_shard(
    config: LaneDConfig,
    shard_index: int,
    num_shards: int,
) -> dict[str, Any]:
    """Write one GPU-capable Lane-D hidden-state scalar shard."""

    normalized_shard_index, normalized_num_shards, shard_label = normalize_lane_d_shard(
        shard_index=shard_index,
        num_shards=num_shards,
    )
    if (
        normalized_shard_index is None
        or normalized_num_shards is None
        or shard_label is None
    ):
        raise ValueError("shard_index and num_shards must be provided")

    shard_dir = config.paths.artifact_root / "shards" / shard_label
    selected_cases_path = shard_dir / LANE_D_MERGE_JSONL_FILES["selected_cases"]
    if not selected_cases_path.exists():
        raise FileNotFoundError(
            "Lane D hidden_states requires materialized selected_cases.jsonl: "
            f"{selected_cases_path}"
        )
    selected_cases = _read_jsonl(selected_cases_path)
    if not selected_cases:
        raise ValueError(f"selected_cases.jsonl is empty: {selected_cases_path}")
    _validate_lane_d_shard_rows(
        selected_cases,
        domain="selected_cases",
        shard_label=shard_label,
        shard_index=normalized_shard_index,
        expected_shards=normalized_num_shards,
    )

    # Heavy Lane-C/model imports stay inside the GPU-capable stage.
    from src.analysis.hard_ce_coord_logit_locality import (
        _model_device,
        load_model_handle,
        load_study_config,
        prepare_lane_c_x1_basin_examples,
    )

    lane_c_config = load_study_config(config.paths.lane_c_study_config)
    model_handle = load_model_handle(lane_c_config)
    lane_c_examples, lane_c_prepare_summary = prepare_lane_c_x1_basin_examples(
        lane_c_config,
        model_handle=model_handle,
        limit=None,
        shard_index=normalized_shard_index,
        num_shards=normalized_num_shards,
    )
    selected_example_pairs = filter_lane_d_examples_for_selected_cases(
        lane_c_examples,
        selected_cases,
    )

    position_inventory: list[dict[str, Any]] = []
    for selected_case, example in selected_example_pairs:
        position_inventory.extend(
            build_lane_d_position_inventory_for_prepared_example(
                example,
                selected_case,
                model_handle.tokenizer,
                shard_index=normalized_shard_index,
                num_shards=normalized_num_shards,
                shard_label=shard_label,
            )
        )
    validate_lane_d_position_inventory(position_inventory)

    probe_rows = _collect_lane_d_hidden_scalar_probe_rows(
        config,
        selected_example_pairs=selected_example_pairs,
        position_inventory=position_inventory,
        model_handle=model_handle,
    )
    patch_rows: list[dict[str, Any]] = []
    selected_cases_sha256 = _sha256_bytes(_canonical_jsonl_bytes(selected_cases))

    row_counts = {
        "selected_cases": len(selected_cases),
        "position_inventory": len(position_inventory),
        "probe_rows": len(probe_rows),
        "patch_rows": len(patch_rows),
    }
    stages_completed = _lane_d_updated_stages_completed(
        shard_dir / LANE_D_SHARD_SUMMARY_FILE,
        "hidden_states",
    )
    summary: dict[str, Any] = {
        "stage": "hidden_state_probe",
        "shard_index": normalized_shard_index,
        "num_shards": normalized_num_shards,
        "shard_label": shard_label,
        "row_counts": row_counts,
        "stages_completed": stages_completed,
        "selected_cases_sha256": selected_cases_sha256,
        "lane_c_study_config": str(config.paths.lane_c_study_config),
        "lane_c_prepare_summary": _jsonable(lane_c_prepare_summary),
        "runtime_kind": "hidden_states_model_forward",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_device": str(_model_device(model_handle.model)),
        "base_seed": None,
        "case_seed_policy": "source_line_idx_mod_num_shards",
    }
    summary.update(lane_d_config_expected_merge_metadata(config))

    _write_canonical_jsonl(
        shard_dir / LANE_D_MERGE_JSONL_FILES["position_inventory"],
        position_inventory,
    )
    _write_canonical_jsonl(
        shard_dir / LANE_D_MERGE_JSONL_FILES["probe_rows"],
        probe_rows,
    )
    _write_canonical_jsonl(
        shard_dir / LANE_D_MERGE_JSONL_FILES["patch_rows"],
        patch_rows,
    )
    _write_json_object(shard_dir / LANE_D_SHARD_SUMMARY_FILE, summary)
    return summary


def write_lane_d_shards_manifest(
    config: LaneDConfig,
    num_shards: int,
) -> dict[str, Any]:
    """Write the CPU-readable Lane-D root shard manifest."""

    if not _is_plain_int(num_shards) or num_shards <= 0:
        raise ValueError("num_shards must be positive")

    shard_count = int(num_shards)
    manifest: dict[str, Any] = {
        "stage": "hidden_state_probe",
        "num_shards": shard_count,
        "expected_shards": shard_count,
        "shard_labels": [
            lane_d_shard_label(index, shard_count) for index in range(shard_count)
        ],
    }
    manifest.update(lane_d_config_expected_merge_metadata(config))
    _write_json_object(config.paths.artifact_root / "shards_manifest.json", manifest)
    return manifest


def merge_lane_d_shards(
    root: Path,
    expected_shards: int,
    expected_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge CPU-readable Lane-D shard artifacts with strict stale-shard gates."""

    if not _is_plain_int(expected_shards) or expected_shards <= 0:
        raise ValueError("expected_shards must be a positive integer")

    artifact_root = Path(root)
    shard_count = int(expected_shards)
    expected_labels = [
        lane_d_shard_label(index, shard_count) for index in range(shard_count)
    ]
    expected_label_set = set(expected_labels)

    manifest_path = artifact_root / "shards_manifest.json"
    manifest = _read_json_object(manifest_path, "Lane D shards manifest")
    _validate_lane_d_shards_manifest(
        manifest,
        expected_shards=shard_count,
        expected_labels=expected_labels,
    )
    shared_metadata = _required_lane_d_shared_metadata(
        manifest,
        "Lane D shards_manifest.json",
    )
    if expected_metadata is not None:
        _validate_lane_d_expected_metadata(
            shared_metadata,
            expected_metadata=expected_metadata,
        )

    shards_dir = artifact_root / "shards"
    if not shards_dir.exists():
        raise FileNotFoundError(f"Lane D shards directory not found: {shards_dir}")
    if not shards_dir.is_dir():
        raise NotADirectoryError(
            f"Lane D shards path is not a directory: {shards_dir}"
        )

    found_dirs = sorted(path for path in shards_dir.iterdir() if path.is_dir())
    malformed: list[str] = []
    found_labels: list[str] = []
    for path in found_dirs:
        match = _LANE_D_SHARD_DIR_PATTERN.fullmatch(path.name)
        if match is None or int(match.group(2)) != shard_count:
            malformed.append(path.name)
            continue
        found_labels.append(path.name)
    if malformed:
        raise ValueError(f"malformed Lane D shard dirs: {malformed}")

    found_label_set = set(found_labels)
    unexpected = sorted(found_label_set - expected_label_set)
    if unexpected:
        raise ValueError(f"unexpected Lane D shard dirs: {unexpected}")
    missing = sorted(expected_label_set - found_label_set)
    if missing:
        raise ValueError(f"missing Lane D shard dirs: {missing}")

    rows_by_domain: dict[str, list[dict[str, Any]]] = {
        domain: [] for domain in LANE_D_MERGE_DOMAINS
    }
    source_summaries: list[dict[str, Any]] = []
    source_summary_sha256_by_shard: dict[str, str] = {}

    for shard_index, shard_label in enumerate(expected_labels):
        shard_dir = shards_dir / shard_label
        required_paths = [
            shard_dir / filename for filename in LANE_D_MERGE_JSONL_FILES.values()
        ] + [shard_dir / LANE_D_SHARD_SUMMARY_FILE]
        missing_files = [str(path) for path in required_paths if not path.exists()]
        if missing_files:
            raise FileNotFoundError(
                f"Lane D shard {shard_label} missing files: {missing_files}"
            )

        shard_rows_by_domain: dict[str, list[dict[str, Any]]] = {}
        for domain, filename in LANE_D_MERGE_JSONL_FILES.items():
            rows = _read_jsonl(shard_dir / filename)
            _validate_lane_d_shard_rows(
                rows,
                domain=domain,
                shard_label=shard_label,
                shard_index=shard_index,
                expected_shards=shard_count,
            )
            shard_rows_by_domain[domain] = rows
            rows_by_domain[domain].extend(rows)

        summary_path = shard_dir / LANE_D_SHARD_SUMMARY_FILE
        shard_summary = _read_json_object(
            summary_path,
            f"Lane D shard {shard_label} summary",
        )
        shard_row_counts = {
            domain: len(rows) for domain, rows in shard_rows_by_domain.items()
        }
        shard_sha256_by_domain = {
            domain: _sha256_bytes(_canonical_jsonl_bytes(rows))
            for domain, rows in shard_rows_by_domain.items()
        }
        _validate_lane_d_shard_summary(
            shard_summary,
            shard_label=shard_label,
            shard_index=shard_index,
            expected_shards=shard_count,
            actual_row_counts=shard_row_counts,
            actual_sha256_by_domain=shard_sha256_by_domain,
            shared_metadata=shared_metadata,
        )
        source_summary_sha256_by_shard[shard_label] = _sha256_bytes(
            _canonical_json_bytes(shard_summary)
        )
        source_summaries.append(
            {
                "shard_label": shard_label,
                "summary_path": str(summary_path),
                "summary": _jsonable(shard_summary),
            }
        )

    for domain, rows in rows_by_domain.items():
        _reject_duplicate_lane_d_row_keys(domain, rows)
    validate_lane_d_position_inventory(rows_by_domain["position_inventory"])

    jsonl_bytes_by_domain = {
        domain: _canonical_jsonl_bytes(rows_by_domain[domain])
        for domain in LANE_D_MERGE_DOMAINS
    }
    sha256_by_domain = {
        domain: _sha256_bytes(content)
        for domain, content in jsonl_bytes_by_domain.items()
    }
    selected_cases_sha256 = sha256_by_domain["selected_cases"]
    row_counts = {
        domain: len(rows_by_domain[domain]) for domain in LANE_D_MERGE_DOMAINS
    }
    output_paths = {
        domain: str(artifact_root / filename)
        for domain, filename in LANE_D_MERGE_JSONL_FILES.items()
    }
    output_paths["summary"] = str(artifact_root / "summary.json")
    output_paths["merge_summary"] = str(artifact_root / "merge_summary.json")

    merge_summary: dict[str, Any] = {
        "stage": "hidden_state_probe",
        "artifact_root": str(artifact_root),
        "expected_shards": shard_count,
        "merged_shards": expected_labels,
        "row_counts": row_counts,
        "source_manifest_path": str(manifest_path),
        "source_manifest_sha256": _sha256_bytes(_canonical_json_bytes(manifest)),
        "source_summaries": source_summaries,
        "source_summary_sha256_by_shard": source_summary_sha256_by_shard,
        "output_paths": output_paths,
        "duplicate_key_domains": {
            domain: list(fields)
            for domain, fields in LANE_D_DUPLICATE_KEY_DOMAINS.items()
        },
        "compact_roles": list(LANE_D_COMPACT_ROLES),
        "sha256_by_domain": sha256_by_domain,
        "selected_cases_sha256": selected_cases_sha256,
    }
    merge_summary.update(shared_metadata)

    artifact_root.mkdir(parents=True, exist_ok=True)
    outputs_to_write = {
        artifact_root / LANE_D_MERGE_JSONL_FILES[domain]: content
        for domain, content in jsonl_bytes_by_domain.items()
    }
    summary_bytes = _pretty_json_bytes(merge_summary)
    outputs_to_write[artifact_root / "summary.json"] = summary_bytes
    outputs_to_write[artifact_root / "merge_summary.json"] = summary_bytes
    _write_lane_d_merge_outputs(outputs_to_write)
    return merge_summary


def _collect_lane_d_hidden_scalar_probe_rows(
    config: LaneDConfig,
    *,
    selected_example_pairs: Sequence[tuple[Mapping[str, Any], Any]],
    position_inventory: Sequence[Mapping[str, Any]],
    model_handle: Any,
) -> list[dict[str, Any]]:
    # Heavy tensor/model helper imports stay local to hidden-state execution.
    import torch

    from src.analysis.hard_ce_coord_logit_locality import (
        _batch_pad_offset,
        _load_image,
        _model_device,
    )

    inventory_by_case_id: dict[str, list[Mapping[str, Any]]] = {}
    for row in position_inventory:
        case_id = _required_nonempty_string(row, "case_id")
        inventory_by_case_id.setdefault(case_id, []).append(row)
    selected_by_case_id = {
        _required_nonempty_string(selected_case, "case_id"): selected_case
        for selected_case, _ in selected_example_pairs
    }
    example_by_case_id = {
        _lane_d_example_case_id(example): example
        for _, example in selected_example_pairs
    }

    rows: list[dict[str, Any]] = []
    batch_size = max(1, int(config.execution.batch_size))
    examples = [example for _, example in selected_example_pairs]
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
            key: value.to(_model_device(model_handle.model))
            if isinstance(value, torch.Tensor)
            else value
            for key, value in model_inputs.items()
        }
        with torch.inference_mode():
            outputs = model_handle.model(
                **model_inputs,
                use_cache=False,
                output_hidden_states=True,
            )
        hidden_states = getattr(outputs, "hidden_states", None)
        if not isinstance(hidden_states, tuple) or len(hidden_states) <= 1:
            raise RuntimeError("model forward did not return decoder hidden_states")
        input_ids = model_inputs.get("input_ids")
        if not isinstance(input_ids, torch.Tensor):
            raise RuntimeError("processor output missing input_ids")

        resolved_layers = _resolve_lane_d_layer_groups(
            config.positions.layer_groups,
            decoder_layer_count=len(hidden_states) - 1,
        )
        for batch_idx, example in enumerate(batch):
            case_id = _lane_d_example_case_id(example)
            pad_offset = _batch_pad_offset(
                input_ids=input_ids,
                batch_idx=batch_idx,
                expected_ids=example.full_input_ids,
            )
            for inventory_row in inventory_by_case_id.get(case_id, []):
                selected_case = selected_by_case_id[case_id]
                example_for_case = example_by_case_id[case_id]
                target_ledger = _lane_d_target_ledger(selected_case, example_for_case)
                hidden_token_index = _lane_d_hidden_token_index(inventory_row)
                tensor_token_index = int(pad_offset) + hidden_token_index
                for (
                    layer_group,
                    configured_layer,
                    model_layer,
                    hidden_state_tuple_index,
                ) in resolved_layers:
                    hidden_tensor = hidden_states[hidden_state_tuple_index]
                    hidden_vec = hidden_tensor[batch_idx, tensor_token_index].detach().float()
                    scalar_summary = _lane_d_hidden_scalar_summary(hidden_vec)
                    logit_lens_summary = _lane_d_x1_logit_lens_summary(
                        config,
                        model_handle=model_handle,
                        hidden_vec=hidden_vec,
                        role=str(inventory_row["role"]),
                        target_x1_bin=target_ledger.get("target_x1_bin"),
                    )
                    coord_slot_logit_lens_summary = _lane_d_coord_slot_logit_lens_summary(
                        config,
                        model_handle=model_handle,
                        hidden_vec=hidden_vec,
                        role=str(inventory_row["role"]),
                        target_ledger=target_ledger,
                    )
                    rows.append(
                        {
                            "source_line_idx": inventory_row["source_line_idx"],
                            "case_id": case_id,
                            "prefix_mode": inventory_row["prefix_mode"],
                            "prefix_depth": inventory_row["prefix_depth"],
                            "prefix_quality": inventory_row["prefix_quality"],
                            "task": "hidden_scalar",
                            "target_label_source": "lane_c_target",
                            "role": inventory_row["role"],
                            "slot": _lane_d_slot_for_role(str(inventory_row["role"])),
                            "layer_group": layer_group,
                            "layer": model_layer,
                            "configured_layer": configured_layer,
                            "model_layer": model_layer,
                            "hidden_state_tuple_index": hidden_state_tuple_index,
                            "prefix_condition": getattr(
                                example_for_case,
                                "prefix_condition",
                                inventory_row["prefix_mode"],
                            ),
                            "render_source": inventory_row["render_source"],
                            "separator_kind": inventory_row["separator_kind"],
                            "prefix_state_kind": inventory_row["prefix_state_kind"],
                            "absolute_token_index": inventory_row["absolute_token_index"],
                            "prediction_token_index": inventory_row[
                                "prediction_token_index"
                            ],
                            "hidden_token_index": hidden_token_index,
                            "tensor_token_index": tensor_token_index,
                            "assistant_relative_token_index": inventory_row[
                                "assistant_relative_token_index"
                            ],
                            "assistant_start_token_index": inventory_row[
                                "assistant_start_token_index"
                            ],
                            "token_text": inventory_row["token_text"],
                            "shard_index": inventory_row["shard_index"],
                            "num_shards": inventory_row["num_shards"],
                            "shard_label": inventory_row["shard_label"],
                            **target_ledger,
                            **scalar_summary,
                            **logit_lens_summary,
                            **coord_slot_logit_lens_summary,
                        }
                    )
    return rows


def _resolve_lane_d_layer_groups(
    layer_groups: Mapping[str, Sequence[int]],
    *,
    decoder_layer_count: int,
) -> list[tuple[str, int, int, int]]:
    if decoder_layer_count <= 0:
        raise RuntimeError("hidden_states must contain at least one decoder layer")
    resolved: list[tuple[str, int, int, int]] = []
    for layer_group, configured_layers in layer_groups.items():
        for configured_layer in configured_layers:
            raw_layer = int(configured_layer)
            model_layer = (
                decoder_layer_count + raw_layer if raw_layer < 0 else raw_layer
            )
            if model_layer < 0 or model_layer >= decoder_layer_count:
                raise ValueError(
                    f"positions.layer_groups.{layer_group} layer {raw_layer} "
                    f"resolves outside 0..{decoder_layer_count - 1}"
                )
            resolved.append(
                (
                    str(layer_group),
                    raw_layer,
                    int(model_layer),
                    int(model_layer) + 1,
                )
            )
    return resolved


def _lane_d_hidden_scalar_summary(hidden_vec: Any) -> dict[str, Any]:
    import torch

    finite = bool(torch.isfinite(hidden_vec).all().detach().cpu().item())
    numel = int(hidden_vec.numel())
    if not finite or numel <= 0:
        return {
            "hidden_finite": finite,
            "hidden_numel": numel,
            "hidden_norm": None,
            "hidden_rms": None,
            "hidden_mean": None,
            "hidden_std": None,
        }
    norm = float(torch.linalg.vector_norm(hidden_vec).detach().cpu().item())
    rms = float(torch.sqrt(torch.mean(hidden_vec * hidden_vec)).detach().cpu().item())
    mean = float(torch.mean(hidden_vec).detach().cpu().item())
    std = float(torch.std(hidden_vec, unbiased=False).detach().cpu().item())
    return {
        "hidden_finite": True,
        "hidden_numel": numel,
        "hidden_norm": _json_finite_float(norm),
        "hidden_rms": _json_finite_float(rms),
        "hidden_mean": _json_finite_float(mean),
        "hidden_std": _json_finite_float(std),
    }


def _lane_d_target_coord_bins(example: Any) -> tuple[int, int, int, int] | None:
    assistant_text = getattr(example, "assistant_text", None)
    if not isinstance(assistant_text, str) or not assistant_text:
        return None
    target_row = assistant_text.splitlines()[-1]
    values = tuple(int(match.group(1)) for match in _LANE_D_COORD_VALUE_RE.finditer(target_row))
    if len(values) != 4:
        return None
    if any(value < 0 or value > 999 for value in values):
        return None
    return values  # type: ignore[return-value]


def _lane_d_x1_logit_lens_summary(
    config: LaneDConfig,
    *,
    model_handle: Any,
    hidden_vec: Any,
    role: str,
    target_x1_bin: Any,
) -> dict[str, Any]:
    if not config.execution.enable_x1_logit_lens:
        return {}
    if role not in set(config.execution.x1_logit_lens_roles):
        return {}
    base_unavailable = {
        "x1_logit_lens_available": False,
        "x1_logit_lens_role_enabled": True,
        "x1_logit_lens_rank": None,
        "x1_logit_lens_top1_bin": None,
        "x1_logit_lens_target_logit": None,
        "x1_logit_lens_top1_logit": None,
        "x1_logit_lens_target_minus_top1": None,
        "x1_logit_lens_top_bins": [],
    }
    if not isinstance(target_x1_bin, int) or target_x1_bin < 0 or target_x1_bin > 999:
        return {**base_unavailable, "x1_logit_lens_unavailable_reason": "missing_target_x1_bin"}
    coord_token_ids = _lane_d_coord_token_ids_for_logit_lens(model_handle)
    if coord_token_ids is None:
        return {**base_unavailable, "x1_logit_lens_unavailable_reason": "missing_coord_token_ids"}
    logits = _lane_d_hidden_lm_head_logits(model_handle, hidden_vec)
    if logits is None:
        return {**base_unavailable, "x1_logit_lens_unavailable_reason": "missing_lm_head"}

    import torch

    coord_index = torch.tensor(
        [int(token_id) for token_id in coord_token_ids],
        dtype=torch.long,
        device=logits.device,
    )
    coord_logits = logits.index_select(0, coord_index).detach().float()
    target_logit = float(coord_logits[int(target_x1_bin)].cpu().item())
    greater_count = int((coord_logits > coord_logits[int(target_x1_bin)]).sum().cpu().item())
    top_k = min(int(config.execution.x1_logit_lens_top_k), int(coord_logits.numel()))
    top_values, top_indices = torch.topk(coord_logits, k=top_k)
    top1_bin = int(top_indices[0].cpu().item())
    top1_logit = float(top_values[0].cpu().item())
    top_bins = [
        {
            "bin": int(bin_idx.cpu().item()),
            "logit": _json_finite_float(float(value.cpu().item())),
            "distance": int(abs(int(bin_idx.cpu().item()) - int(target_x1_bin))),
        }
        for value, bin_idx in zip(top_values, top_indices, strict=True)
    ]
    return {
        "x1_logit_lens_available": True,
        "x1_logit_lens_role_enabled": True,
        "x1_logit_lens_rank": greater_count + 1,
        "x1_logit_lens_top1_bin": top1_bin,
        "x1_logit_lens_target_logit": _json_finite_float(target_logit),
        "x1_logit_lens_top1_logit": _json_finite_float(top1_logit),
        "x1_logit_lens_target_minus_top1": _json_finite_float(target_logit - top1_logit),
        "x1_logit_lens_top_bins": top_bins,
    }


def _lane_d_coord_slot_logit_lens_summary(
    config: LaneDConfig,
    *,
    model_handle: Any,
    hidden_vec: Any,
    role: str,
    target_ledger: Mapping[str, Any],
) -> dict[str, Any]:
    if not config.execution.enable_coord_slot_logit_lens:
        return {}
    if role not in set(config.execution.coord_slot_logit_lens_roles):
        return {}
    target_slot = _lane_d_slot_for_role(role)
    base_unavailable = {
        "coord_slot_logit_lens_available": False,
        "coord_slot_logit_lens_role_enabled": True,
        "coord_slot_logit_lens_target_slot": target_slot,
        "coord_slot_logit_lens_target_bin": None,
        "coord_slot_logit_lens_rank": None,
        "coord_slot_logit_lens_top1_bin": None,
        "coord_slot_logit_lens_target_logit": None,
        "coord_slot_logit_lens_top1_logit": None,
        "coord_slot_logit_lens_target_minus_top1": None,
        "coord_slot_logit_lens_top_bins": [],
    }
    target_bin_key = f"target_{target_slot}_bin"
    target_bin = target_ledger.get(target_bin_key)
    if not isinstance(target_bin, int) or target_bin < 0 or target_bin > 999:
        return {
            **base_unavailable,
            "coord_slot_logit_lens_unavailable_reason": f"missing_{target_bin_key}",
        }
    summary = _lane_d_coord_logit_lens_summary(
        config,
        model_handle=model_handle,
        hidden_vec=hidden_vec,
        target_bin=target_bin,
        top_k=config.execution.coord_slot_logit_lens_top_k,
    )
    if summary is None:
        return {
            **base_unavailable,
            "coord_slot_logit_lens_target_bin": target_bin,
            "coord_slot_logit_lens_unavailable_reason": "missing_coord_token_ids_or_lm_head",
        }
    return {
        "coord_slot_logit_lens_available": True,
        "coord_slot_logit_lens_role_enabled": True,
        "coord_slot_logit_lens_target_slot": target_slot,
        "coord_slot_logit_lens_target_bin": target_bin,
        "coord_slot_logit_lens_rank": summary["rank"],
        "coord_slot_logit_lens_top1_bin": summary["top1_bin"],
        "coord_slot_logit_lens_target_logit": summary["target_logit"],
        "coord_slot_logit_lens_top1_logit": summary["top1_logit"],
        "coord_slot_logit_lens_target_minus_top1": summary["target_minus_top1"],
        "coord_slot_logit_lens_top_bins": summary["top_bins"],
    }


def _lane_d_coord_logit_lens_summary(
    config: LaneDConfig,
    *,
    model_handle: Any,
    hidden_vec: Any,
    target_bin: int,
    top_k: int,
) -> dict[str, Any] | None:
    coord_token_ids = _lane_d_coord_token_ids_for_logit_lens(model_handle)
    if coord_token_ids is None:
        return None
    logits = _lane_d_hidden_lm_head_logits(model_handle, hidden_vec)
    if logits is None:
        return None

    import torch

    coord_index = torch.tensor(
        [int(token_id) for token_id in coord_token_ids],
        dtype=torch.long,
        device=logits.device,
    )
    coord_logits = logits.index_select(0, coord_index).detach().float()
    target_logit = float(coord_logits[int(target_bin)].cpu().item())
    greater_count = int((coord_logits > coord_logits[int(target_bin)]).sum().cpu().item())
    top_k = min(int(top_k), int(coord_logits.numel()))
    top_values, top_indices = torch.topk(coord_logits, k=top_k)
    top1_bin = int(top_indices[0].cpu().item())
    top1_logit = float(top_values[0].cpu().item())
    return {
        "rank": greater_count + 1,
        "top1_bin": top1_bin,
        "target_logit": _json_finite_float(target_logit),
        "top1_logit": _json_finite_float(top1_logit),
        "target_minus_top1": _json_finite_float(target_logit - top1_logit),
        "top_bins": [
            {
                "bin": int(bin_idx.cpu().item()),
                "logit": _json_finite_float(float(value.cpu().item())),
                "distance": int(abs(int(bin_idx.cpu().item()) - int(target_bin))),
            }
            for value, bin_idx in zip(top_values, top_indices, strict=True)
        ],
    }


def _lane_d_coord_token_ids_for_logit_lens(model_handle: Any) -> tuple[int, ...] | None:
    raw = getattr(model_handle, "coord_token_ids", None)
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        coord_ids = tuple(int(token_id) for token_id in raw)
        if len(coord_ids) == 1000:
            return coord_ids
    tokenizer = getattr(model_handle, "tokenizer", None)
    if tokenizer is None:
        return None
    try:
        from src.analysis.hard_ce_coord_logit_locality import resolve_coord_token_ids

        return tuple(int(token_id) for token_id in resolve_coord_token_ids(tokenizer).coord_token_ids)
    except Exception:
        return None


def _lane_d_hidden_lm_head_logits(model_handle: Any, hidden_vec: Any) -> Any | None:
    import torch

    model = getattr(model_handle, "model", None)
    if model is None:
        return None
    language_model, lm_head = _lane_d_language_norm_and_lm_head(model)
    if lm_head is None:
        get_output_embeddings = getattr(model, "get_output_embeddings", None)
        lm_head = get_output_embeddings() if callable(get_output_embeddings) else None
    if not callable(lm_head):
        return None
    try:
        norm = getattr(language_model, "norm", None) if language_model is not None else None
        head_input = hidden_vec
        if callable(norm):
            head_input = norm(hidden_vec.unsqueeze(0)).squeeze(0)
        weight = getattr(lm_head, "weight", None)
        if isinstance(weight, torch.Tensor):
            head_input = head_input.to(device=weight.device, dtype=weight.dtype)
        logits = lm_head(head_input)
    except Exception:
        return None
    if not isinstance(logits, torch.Tensor):
        return None
    logits = logits.squeeze()
    if logits.ndim != 1:
        return None
    return logits


def _lane_d_language_norm_and_lm_head(model: Any) -> tuple[Any | None, Any | None]:
    visited: set[int] = set()
    stack = [model]
    attr_paths = (
        "model",
        "base_model",
        "module",
        "language_model",
    )
    while stack:
        candidate = stack.pop()
        if candidate is None or id(candidate) in visited:
            continue
        visited.add(id(candidate))
        language_model = getattr(candidate, "language_model", None)
        lm_head = getattr(candidate, "lm_head", None)
        if language_model is not None and callable(lm_head):
            return language_model, lm_head
        if callable(lm_head):
            return None, lm_head
        for attr in attr_paths:
            child = getattr(candidate, attr, None)
            if child is not None and id(child) not in visited:
                stack.append(child)
    return None, None


def _lane_d_hidden_token_index(row: Mapping[str, Any]) -> int:
    prediction = row.get("prediction_token_index")
    if prediction is not None:
        return _required_nonnegative_int(row, "prediction_token_index")
    return _required_nonnegative_int(row, "absolute_token_index")


def _lane_d_updated_stages_completed(summary_path: Path, new_stage: str) -> list[str]:
    stages: list[str] = []
    if summary_path.exists():
        existing = _read_json_object(summary_path, "Lane D shard summary")
        raw_stages = existing.get("stages_completed")
        if isinstance(raw_stages, Sequence) and not isinstance(raw_stages, (str, bytes)):
            stages.extend(str(stage) for stage in raw_stages if isinstance(stage, str))
    if "select_cases" not in stages:
        stages.insert(0, "select_cases")
    if new_stage not in stages:
        stages.append(new_stage)
    return stages


def _lane_d_example_case_id(example: Any) -> str:
    metadata = getattr(example, "lane_c_metadata", None)
    if isinstance(metadata, Mapping):
        case_id = metadata.get("case_id")
        if isinstance(case_id, str) and case_id:
            return case_id
    pairing_id = getattr(example, "pairing_id", None)
    if isinstance(pairing_id, str) and pairing_id:
        return pairing_id
    raise ValueError("Lane C example missing nonempty case_id")


def _lane_d_token_ids(tokenizer: Any, text: str) -> tuple[int, ...]:
    token_ids = tokenizer.encode(str(text), add_special_tokens=False)
    if not isinstance(token_ids, Sequence):
        raise ValueError("tokenizer.encode must return a sequence")
    return tuple(int(item) for item in token_ids)


def _lane_d_decode_single_token(
    tokenizer: Any,
    full_input_ids: Sequence[int],
    absolute_index: int,
) -> str:
    if absolute_index < 0 or absolute_index >= len(full_input_ids):
        raise ValueError(f"absolute_token_index out of range: {absolute_index}")
    token_id = int(full_input_ids[absolute_index])
    decode = getattr(tokenizer, "decode", None)
    if callable(decode):
        try:
            return str(decode([token_id]))
        except Exception:
            pass
    return str(token_id)


def _lane_d_find_subsequence(
    haystack: Sequence[int],
    needle: Sequence[int],
) -> int | None:
    return _lane_d_find_subsequence_after(haystack, needle, 0)


def _lane_d_find_last_subsequence(
    haystack: Sequence[int],
    needle: Sequence[int],
) -> int | None:
    if not needle:
        return len(haystack)
    last = len(haystack) - len(needle)
    for start in range(last, -1, -1):
        if tuple(haystack[start : start + len(needle)]) == tuple(needle):
            return int(start)
    return None


def _lane_d_find_subsequence_after(
    haystack: Sequence[int],
    needle: Sequence[int],
    start_index: int,
) -> int | None:
    if not needle:
        return int(start_index)
    last = len(haystack) - len(needle)
    for start in range(max(0, int(start_index)), max(0, last + 1)):
        if tuple(haystack[start : start + len(needle)]) == tuple(needle):
            return int(start)
    return None


def _lane_d_slot_for_role(role: str) -> str:
    return _LANE_D_ROLE_SLOTS.get(role, "none")


def _json_finite_float(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_idx, line in enumerate(handle):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_idx + 1} must contain a JSON object")
            rows.append(payload)
    return rows


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _write_json_object(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_bytes(path, _pretty_json_bytes(payload))


def _write_canonical_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _atomic_write_bytes(path, _canonical_jsonl_bytes(rows))


def _write_lane_d_merge_outputs(outputs: Mapping[Path, bytes]) -> None:
    if not outputs:
        return
    output_paths = list(outputs)
    root = output_paths[0].parent
    with tempfile.TemporaryDirectory(prefix=".lane_d_merge_", dir=root) as temp_name:
        staging_dir = Path(temp_name)
        staged_paths: list[tuple[Path, Path]] = []
        for output_path, content in outputs.items():
            staged_path = staging_dir / output_path.name
            _atomic_write_bytes(staged_path, content)
            staged_paths.append((staged_path, output_path))
        published: list[tuple[Path, Path | None]] = []
        try:
            for staged_path, output_path in staged_paths:
                backup_path = None
                if output_path.exists():
                    backup_path = staging_dir / f"{output_path.name}.bak"
                    output_path.replace(backup_path)
                try:
                    _publish_lane_d_staged_output(staged_path, output_path)
                except Exception:
                    if output_path.exists():
                        output_path.unlink()
                    if backup_path is not None and backup_path.exists():
                        backup_path.replace(output_path)
                    raise
                published.append((output_path, backup_path))
        except Exception:
            for output_path, backup_path in reversed(published):
                if output_path.exists():
                    output_path.unlink()
                if backup_path is not None and backup_path.exists():
                    backup_path.replace(output_path)
            raise


def _publish_lane_d_staged_output(staged_path: Path, output_path: Path) -> None:
    staged_path.replace(output_path)


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    temp_path = Path(handle.name)
    try:
        handle.write(content)
        handle.close()
        temp_path.replace(path)
    finally:
        if not handle.file.closed:
            handle.close()
        if temp_path.exists():
            temp_path.unlink()


def _canonical_jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    content = "".join(
        json.dumps(
            _jsonable(row),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for row in rows
    )
    return content.encode("utf-8")


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            _jsonable(payload),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _pretty_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(_jsonable(payload), ensure_ascii=True, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_path(path: Path) -> str:
    return _sha256_bytes(Path(path).read_bytes())


def _validate_lane_d_shards_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_shards: int,
    expected_labels: Sequence[str],
) -> None:
    if "num_shards" not in manifest and "expected_shards" not in manifest:
        raise ValueError(
            "Lane D shards_manifest.json must include num_shards or expected_shards"
        )
    for count_key in ("num_shards", "expected_shards"):
        if count_key not in manifest:
            continue
        value = manifest[count_key]
        if not _is_plain_int(value) or int(value) != expected_shards:
            raise ValueError(
                f"Lane D shards_manifest.json {count_key} must equal "
                f"{expected_shards}; got {value!r}"
            )

    listed_labels = _lane_d_manifest_shard_labels(manifest)
    if listed_labels is None:
        raise ValueError(
            "Lane D shards_manifest.json must include shard labels via "
            "shard_labels, merged_shards, or shards"
        )
    if list(listed_labels) != list(expected_labels):
        raise ValueError(
            "Lane D shards_manifest.json shard labels must equal "
            f"{list(expected_labels)}; got {list(listed_labels)}"
        )


def _lane_d_manifest_shard_labels(manifest: Mapping[str, Any]) -> list[str] | None:
    for key in ("shard_labels", "merged_shards"):
        if key in manifest:
            return _parse_lane_d_manifest_label_list(manifest[key], key)

    if "shards" not in manifest:
        return None
    raw_shards = manifest["shards"]
    if isinstance(raw_shards, (str, bytes)) or not isinstance(raw_shards, Sequence):
        raise ValueError("Lane D shards_manifest.json shards must be a sequence")

    labels: list[str] = []
    for index, item in enumerate(raw_shards):
        label: Any | None
        if isinstance(item, str):
            label = item
        elif isinstance(item, Mapping):
            label = item.get("shard_label", item.get("label"))
        else:
            raise ValueError(
                "Lane D shards_manifest.json shards entries must be strings or mappings"
            )
        if label is None:
            raise ValueError(
                "Lane D shards_manifest.json shards "
                f"entry {index} missing shard label"
            )
        if not isinstance(label, str) or not label:
            raise ValueError(
                "Lane D shards_manifest.json shards "
                f"entry {index} has invalid shard label"
            )
        labels.append(label)
    return labels


def _parse_lane_d_manifest_label_list(value: Any, key: str) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"Lane D shards_manifest.json {key} must be a sequence")
    labels: list[str] = []
    for index, label in enumerate(value):
        if not isinstance(label, str) or not label:
            raise ValueError(
                f"Lane D shards_manifest.json {key}[{index}] must be a nonempty string"
            )
        labels.append(label)
    return labels


def _required_lane_d_shared_metadata(
    source: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key in LANE_D_SHARED_REPRODUCIBILITY_FIELDS:
        if key not in source:
            raise ValueError(f"{label} missing required {key}")
        value = source[key]
        if key in {"config_path", "config_sha256", "checkpoint", "artifact_root"}:
            if not isinstance(value, str) or not value:
                raise ValueError(f"{label} {key} must be a nonempty string")
        elif key == "layer_groups":
            if not isinstance(value, Mapping) or not value:
                raise ValueError(f"{label} layer_groups must be a nonempty mapping")
        elif key == "batch_size":
            if not _is_plain_int(value) or int(value) <= 0:
                raise ValueError(f"{label} batch_size must be a positive integer")
        metadata[key] = _jsonable(value)
    return metadata


def _validate_lane_d_expected_metadata(
    manifest_metadata: Mapping[str, Any],
    *,
    expected_metadata: Mapping[str, Any],
) -> None:
    expected = _required_lane_d_shared_metadata(
        expected_metadata,
        "expected Lane D merge metadata",
    )
    for key, expected_value in expected.items():
        if manifest_metadata[key] != expected_value:
            raise ValueError(
                "Lane D shards_manifest.json "
                f"{key} must match loaded config metadata"
            )


def _validate_lane_d_shard_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    domain: str,
    shard_label: str,
    shard_index: int,
    expected_shards: int,
) -> None:
    for row_index, row in enumerate(rows):
        _validate_lane_d_domain_row_schema(row, domain=domain, row_index=row_index)
        source_line_idx = _lane_d_required_source_line_idx(
            row,
            domain=domain,
            row_index=row_index,
        )
        if source_line_idx % expected_shards != shard_index:
            raise ValueError(
                f"Lane D {domain} row {row_index + 1} source_line_idx "
                f"{source_line_idx} is not owned by {shard_label}"
            )
        _validate_lane_d_optional_or_required_shard_field(
            row,
            domain=domain,
            row_index=row_index,
            field_name="shard_index",
            expected_value=shard_index,
            required=True,
            shard_label=shard_label,
        )
        _validate_lane_d_optional_or_required_shard_field(
            row,
            domain=domain,
            row_index=row_index,
            field_name="num_shards",
            expected_value=expected_shards,
            required=True,
            shard_label=shard_label,
        )
        _validate_lane_d_optional_or_required_shard_field(
            row,
            domain=domain,
            row_index=row_index,
            field_name="shard_label",
            expected_value=shard_label,
            required=True,
            shard_label=shard_label,
        )


def _validate_lane_d_domain_row_schema(
    row: Mapping[str, Any],
    *,
    domain: str,
    row_index: int,
) -> None:
    if domain == "selected_cases":
        _required_nonempty_string(row, "case_id")
        _required_nonempty_string(row, "prefix_mode")
        _required_nonnegative_int(row, "prefix_depth")
        _required_json_scalar(row, "prefix_quality")
        _required_nonnegative_int(row, "intended_target_gt_idx")
        _required_nonempty_string(row, "x1_top_peak_attribution")
        _required_nonnegative_int(row, "x1_target_rank")
        return
    if domain == "probe_rows":
        _required_nonempty_string(row, "case_id")
        _required_nonempty_string(row, "prefix_mode")
        _required_nonnegative_int(row, "prefix_depth")
        _required_json_scalar(row, "prefix_quality")
        _required_nonempty_string(row, "task")
        _required_nonempty_string(row, "role")
        _required_nonempty_string(row, "layer_group")
        _required_nonnegative_int(row, "layer")
        _required_nonnegative_int(row, "model_layer")
        _required_nonempty_string(row, "slot")
        _required_nonempty_string(row, "target_label_source")
        _required_nonempty_string(row, "prefix_condition")
        _required_nonnegative_int(row, "object_count")
        _required_nonnegative_int(row, "remaining_count")
        _required_nonnegative_int(row, "intended_target_gt_idx")
        _required_nonempty_string(row, "x1_top_peak_attribution")
        _required_nonnegative_int(row, "x1_target_rank")
        return
    if domain == "patch_rows":
        _required_nonempty_string(row, "case_id")
        _required_nonempty_string(row, "prefix_mode")
        _required_nonnegative_int(row, "prefix_depth")
        _required_nonempty_string(row, "patch_policy")
        _required_nonempty_string(row, "donor_policy")
        _required_value(row, "donor_case_id")
        _required_nonempty_string(row, "role")
        _required_nonempty_string(row, "layer_group")
        _required_nonnegative_int(row, "model_layer")
        _required_nonempty_string(row, "slot")
        return
    if domain != "position_inventory":
        raise ValueError(f"unknown Lane D merge domain: {domain}")


def _lane_d_required_source_line_idx(
    row: Mapping[str, Any],
    *,
    domain: str,
    row_index: int,
) -> int:
    value = row.get("source_line_idx")
    if not _is_nonnegative_plain_int(value):
        raise ValueError(
            f"Lane D {domain} row {row_index + 1} source_line_idx "
            "must be a nonnegative integer"
        )
    return int(value)


def _validate_lane_d_optional_or_required_shard_field(
    row: Mapping[str, Any],
    *,
    domain: str,
    row_index: int,
    field_name: str,
    expected_value: int | str,
    required: bool,
    shard_label: str,
) -> None:
    if field_name not in row:
        if required:
            raise ValueError(
                f"Lane D {domain} row {row_index + 1} missing {field_name} "
                f"for {shard_label}"
            )
        return
    value = row[field_name]
    if isinstance(expected_value, int):
        valid = _is_plain_int(value) and int(value) == expected_value
    else:
        valid = isinstance(value, str) and value == expected_value
    if not valid:
        raise ValueError(
            f"Lane D {domain} row {row_index + 1} {field_name} "
            f"must match {shard_label}; got {value!r}"
        )


def _validate_lane_d_shard_summary(
    summary: Mapping[str, Any],
    *,
    shard_label: str,
    shard_index: int,
    expected_shards: int,
    actual_row_counts: Mapping[str, int],
    actual_sha256_by_domain: Mapping[str, str],
    shared_metadata: Mapping[str, Any],
) -> None:
    summary_label = f"Lane D shard {shard_label} summary"
    stage = summary.get("stage")
    if stage != "hidden_state_probe":
        raise ValueError(
            f"{summary_label} stage must be hidden_state_probe; got {stage!r}"
        )
    _require_summary_int(
        summary,
        "shard_index",
        shard_index,
        summary_label,
    )
    _require_summary_int(
        summary,
        "num_shards",
        expected_shards,
        summary_label,
    )
    actual_label = summary.get("shard_label")
    if actual_label != shard_label:
        raise ValueError(
            f"{summary_label} shard_label must be {shard_label}; got {actual_label!r}"
        )

    row_counts = summary.get("row_counts")
    if not isinstance(row_counts, Mapping):
        raise ValueError(f"{summary_label} row_counts must be a mapping")
    for domain in LANE_D_MERGE_DOMAINS:
        value = row_counts.get(domain)
        expected_count = actual_row_counts[domain]
        if not _is_plain_int(value) or int(value) != expected_count:
            raise ValueError(
                f"{summary_label} row_counts.{domain} must equal "
                f"{expected_count}; got {value!r}"
            )

    summary_metadata = _required_lane_d_shared_metadata(summary, summary_label)
    for key, expected_value in shared_metadata.items():
        if summary_metadata[key] != expected_value:
            raise ValueError(
                f"{summary_label} {key} must match shards_manifest.json"
            )
    selected_hash = summary.get("selected_cases_sha256")
    if not isinstance(selected_hash, str) or not selected_hash:
        raise ValueError(f"{summary_label} selected_cases_sha256 must be recorded")
    if selected_hash != actual_sha256_by_domain["selected_cases"]:
        raise ValueError(f"{summary_label} selected_cases_sha256 must match selected_cases.jsonl")
    for key in ("runtime_kind", "cuda_visible_devices", "torch_device", "base_seed", "case_seed_policy"):
        if key not in summary:
            raise ValueError(f"{summary_label} missing required {key}")
    runtime_kind = summary.get("runtime_kind")
    if not isinstance(runtime_kind, str) or not runtime_kind:
        raise ValueError(f"{summary_label} runtime_kind must be a nonempty string")
    case_seed_policy = summary.get("case_seed_policy")
    if not isinstance(case_seed_policy, str) or not case_seed_policy:
        raise ValueError(f"{summary_label} case_seed_policy must be a nonempty string")


def _require_summary_int(
    summary: Mapping[str, Any],
    key: str,
    expected_value: int,
    summary_label: str,
) -> None:
    value = summary.get(key)
    if not _is_plain_int(value) or int(value) != expected_value:
        raise ValueError(
            f"{summary_label} {key} must equal {expected_value}; got {value!r}"
        )


def _reject_duplicate_lane_d_row_keys(
    domain: str,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    fields = LANE_D_DUPLICATE_KEY_DOMAINS[domain]
    seen: dict[tuple[Any, ...], int] = {}
    for row_index, row in enumerate(rows):
        key = _lane_d_row_key(domain, row, fields, row_index)
        if key in seen:
            raise ValueError(
                "duplicate Lane D row key "
                f"for {domain}: {key!r} at merged row {row_index + 1}; "
                f"first seen at merged row {seen[key] + 1}"
            )
        seen[key] = row_index


def _lane_d_row_key(
    domain: str,
    row: Mapping[str, Any],
    fields: Sequence[str],
    row_index: int,
) -> tuple[Any, ...]:
    values: list[Any] = []
    for field_name in fields:
        if field_name not in row:
            raise ValueError(
                f"Lane D {domain} row {row_index + 1} missing duplicate key field "
                f"{field_name}"
            )
        value = row[field_name]
        _validate_lane_d_duplicate_key_value(domain, row_index, field_name, value)
        values.append(value)
    return tuple(values)


def _validate_lane_d_duplicate_key_value(
    domain: str,
    row_index: int,
    field_name: str,
    value: Any,
) -> None:
    context = f"Lane D {domain} row {row_index + 1} duplicate key field {field_name}"
    if field_name in _LANE_D_DUPLICATE_KEY_NONNEGATIVE_INT_FIELDS:
        if not _is_nonnegative_plain_int(value):
            raise ValueError(f"{context} must be a nonnegative integer")
        return
    if field_name in _LANE_D_DUPLICATE_KEY_INT_FIELDS:
        if not _is_plain_int(value):
            raise ValueError(f"{context} must be an integer")
        return
    if field_name == "donor_case_id":
        if value is None:
            return
        if not isinstance(value, str) or not value:
            raise ValueError(f"{context} must be a nonempty string or null")
        return
    if field_name in _LANE_D_DUPLICATE_KEY_STRING_FIELDS:
        if not isinstance(value, str) or not value:
            raise ValueError(f"{context} must be a nonempty string")
        return
    if value is None or not isinstance(value, (str, int, float, bool)):
        raise ValueError(f"{context} must be a JSON scalar or null")


def _lane_d_merge_metadata(
    sources: Sequence[Mapping[str, Any]],
    *,
    keys: Mapping[str, Sequence[Sequence[str]]],
) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for output_key, paths in keys.items():
        value = _first_nested_value(sources, paths)
        if value is not None:
            metadata[output_key] = _jsonable(value)
    return metadata


def _first_nested_value(
    sources: Sequence[Mapping[str, Any]],
    paths: Sequence[Sequence[str]],
) -> Any:
    for source in sources:
        for path in paths:
            value = _nested_value(source, path)
            if value is not None:
                return value
    return None


def _nested_value(source: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: Any = source
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (str, bytes)):
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return value
    if isinstance(value, Sequence):
        return [_jsonable(item) for item in value]
    return value


def _is_x1_non_target_or_low_rank(
    *, top_peak_attribution: Any, target_rank: Any
) -> bool:
    return top_peak_attribution != "target_gt_object" or (
        _is_plain_int(target_rank) and target_rank > 32
    )


def _required_config_mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return value


def _optional_config_mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return value


def _required_config_path(parent: Mapping[str, Any], key: str) -> Path:
    value = parent.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"paths.{key} must be a nonempty string path")
    return Path(value).expanduser()


def _optional_config_path(
    parent: Mapping[str, Any],
    key: str,
    *,
    default: Path,
) -> Path:
    value = parent.get(key)
    if value is None:
        return Path(default).expanduser()
    if not isinstance(value, str) or not value:
        raise ValueError(f"paths.{key} must be a nonempty string path")
    return Path(value).expanduser()


def _parse_configured_lane_d_roles(value: Any) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("positions.roles must be a sequence of compact role names")
    return validate_configured_lane_d_roles(tuple(value))


def _parse_lane_d_layer_groups(value: Any) -> Mapping[str, tuple[int, ...]]:
    if not isinstance(value, Mapping):
        raise ValueError("positions.layer_groups must be a mapping")
    if not value:
        raise ValueError("positions.layer_groups must include at least one layer group")
    parsed: dict[str, tuple[int, ...]] = {}
    for group_name, raw_layers in value.items():
        if not isinstance(group_name, str) or not group_name:
            raise ValueError("positions.layer_groups keys must be nonempty strings")
        if isinstance(raw_layers, (str, bytes)) or not isinstance(raw_layers, Sequence):
            raise ValueError(f"positions.layer_groups.{group_name} must be a sequence")
        if not raw_layers:
            raise ValueError(
                f"positions.layer_groups.{group_name} must include at least one layer"
            )
        layers: list[int] = []
        for index, layer in enumerate(raw_layers):
            layers.append(
                _config_plain_int(
                    layer,
                    f"positions.layer_groups.{group_name}[{index}]",
                )
            )
        parsed[group_name] = tuple(layers)
    return parsed


def _parse_lane_d_string_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{field_name} must be a sequence of strings")
    parsed = tuple(str(item) for item in value)
    if not parsed:
        raise ValueError(f"{field_name} must not be empty")
    return parsed


def _normalize_lane_d_stages(stages: Sequence[str]) -> tuple[str, ...]:
    if isinstance(stages, (str, bytes)) or not isinstance(stages, Sequence):
        raise ValueError("stages must be a sequence of stage names")
    parsed: list[str] = []
    allowed = set(LANE_D_STAGES)
    for raw_stage in stages:
        stage = str(raw_stage).strip()
        if not stage:
            continue
        if stage not in allowed:
            raise ValueError(f"unknown Lane D stage {stage!r}")
        parsed.append(stage)
    if not parsed:
        raise ValueError("stages must include at least one Lane D stage")
    return tuple(parsed)


def _lane_d_selected_shards(
    *, shard_index: int | None, num_shards: int | None
) -> list[tuple[int | None, int | None, str]]:
    if num_shards is None:
        if shard_index is not None:
            raise ValueError("num_shards must be provided when shard_index is provided")
        return [(None, None, "unsharded")]

    if not _is_plain_int(num_shards) or num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if shard_index is None:
        return [
            (index, num_shards, lane_d_shard_label(index, num_shards))
            for index in range(num_shards)
        ]
    if not _is_plain_int(shard_index) or shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")
    return [(shard_index, num_shards, lane_d_shard_label(shard_index, num_shards))]


def _lane_d_dry_run_shard_entry(
    artifact_root: Path,
    shard: tuple[int | None, int | None, str],
) -> dict[str, Any]:
    shard_index, num_shards, shard_label = shard
    shard_dir = (
        artifact_root
        if shard_index is None
        else artifact_root / "shards" / shard_label
    )
    return {
        "shard_index": shard_index,
        "num_shards": num_shards,
        "shard_label": shard_label,
        "shard_dir": str(shard_dir),
        "selected_cases": str(shard_dir / "selected_cases.jsonl"),
        "position_inventory": str(shard_dir / "position_inventory.jsonl"),
        "probe_rows": str(shard_dir / "probe_rows.jsonl"),
        "patch_rows": str(shard_dir / "patch_rows.jsonl"),
        "summary": str(shard_dir / "summary.json"),
    }


def _config_nonnegative_int(value: Any, key: str) -> int:
    if not _is_plain_int(value) or value < 0:
        raise ValueError(f"{key} must be a nonnegative integer")
    return int(value)


def _config_positive_int(value: Any, key: str) -> int:
    if not _is_plain_int(value) or value <= 0:
        raise ValueError(f"{key} must be a positive integer")
    return int(value)


def _config_bool(value: Any, key: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return bool(value)


def _config_plain_int(value: Any, key: str) -> int:
    if not _is_plain_int(value):
        raise ValueError(f"{key} must be an integer")
    return int(value)


def _required_value(
    row: Mapping[str, Any],
    key: str,
    *,
    case_id: str | None = None,
    role: str | None = None,
) -> Any:
    if key not in row:
        raise ValueError(f"position inventory row missing {key}{_context(case_id, role)}")
    return row[key]


def _required_nonempty_string(
    row: Mapping[str, Any],
    key: str,
    *,
    case_id: str | None = None,
    role: str | None = None,
) -> str:
    value = _required_value(row, key, case_id=case_id, role=role)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a nonempty string{_context(case_id, role)}")
    return value


def _required_nonnegative_int(
    row: Mapping[str, Any],
    key: str,
    *,
    case_id: str | None = None,
    role: str | None = None,
) -> int:
    value = _required_value(row, key, case_id=case_id, role=role)
    if not _is_nonnegative_plain_int(value):
        raise ValueError(f"{key} must be a nonnegative integer{_context(case_id, role)}")
    return int(value)


def _required_positive_int(
    row: Mapping[str, Any],
    key: str,
    *,
    case_id: str | None = None,
    role: str | None = None,
) -> int:
    value = _required_value(row, key, case_id=case_id, role=role)
    if not _is_plain_int(value) or value <= 0:
        raise ValueError(f"{key} must be a positive integer{_context(case_id, role)}")
    return int(value)


def _required_json_scalar(
    row: Mapping[str, Any],
    key: str,
    *,
    case_id: str | None = None,
    role: str | None = None,
) -> Any:
    value = _required_value(row, key, case_id=case_id, role=role)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError(f"{key} must be null or a JSON scalar{_context(case_id, role)}")


def _context(case_id: str | None, role: str | None) -> str:
    if case_id and role:
        return f" for {case_id}: {role}"
    if case_id:
        return f" for {case_id}"
    return ""


def _required_lane_c_nonempty_string(
    row: Mapping[str, Any],
    key: str,
    input_ordinal: int,
) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"per_case row {input_ordinal + 1} missing nonempty string {key}")
    return value


def _required_lane_c_nonnegative_int(
    row: Mapping[str, Any],
    key: str,
    input_ordinal: int,
) -> int:
    value = row.get(key)
    if not _is_nonnegative_plain_int(value):
        raise ValueError(f"per_case row {input_ordinal + 1} missing nonnegative integer {key}")
    return int(value)


def _is_nonnegative_plain_int(value: Any) -> bool:
    return _is_plain_int(value) and value >= 0


def _is_plain_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)
