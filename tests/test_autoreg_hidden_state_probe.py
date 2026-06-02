from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

import src.analysis.autoreg_hidden_state_probe as lane_d_probe
from src.analysis.autoreg_hidden_state_probe import (
    LANE_D_COMPACT_ROLES,
    LANE_D_PREFIX_STATE_KINDS,
    LANE_D_RENDER_SOURCES,
    LANE_D_SEPARATOR_KINDS,
    build_lane_d_dry_run_plan,
    lane_d_config_expected_merge_metadata,
    lane_d_record_selected,
    lane_d_shard_label,
    load_lane_d_config,
    materialize_lane_d_select_cases_shard,
    merge_lane_d_shards,
    normalize_lane_d_shard,
    select_lane_d_cases,
    validate_lane_d_position_inventory,
    write_lane_d_shards_manifest,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _mutate_json(path: Path, **updates: object) -> None:
    payload = _read_json(path)
    payload.update(updates)
    _write_json(path, payload)


def _write_lane_d_config(
    path: Path,
    *,
    roles: tuple[str, ...] = LANE_D_COMPACT_ROLES,
    layer_groups_yaml: str | None = None,
    artifact_root: Path | None = None,
    checkpoint: Path | str | None = None,
    lane_c_study_config: Path | str | None = None,
    max_cases: object = 512,
    batch_size: object = 1,
) -> None:
    roles_yaml = "\n".join(f"    - {role}" for role in roles)
    layer_groups = layer_groups_yaml or "    early: [0, 1]\n    last: [-4, -3]"
    artifact_root = path.parent / "artifact_root" if artifact_root is None else artifact_root
    checkpoint = path.parent / "checkpoint" if checkpoint is None else checkpoint
    lane_c_study_config = (
        path.parent / "lane_c.yaml"
        if lane_c_study_config is None
        else lane_c_study_config
    )
    path.write_text(
        f"""
paths:
  artifact_root: {artifact_root}
  checkpoint: {checkpoint}
  dataset_jsonl: {path.parent / "dataset.jsonl"}
  self_rollout_root: {path.parent / "self_rollout"}
  lane_a_rollout_root: {path.parent / "lane_a"}
  lane_c_per_case: {path.parent / "per_case.jsonl"}
  lane_c_study_config: {lane_c_study_config}
selection:
  max_cases: {max_cases}
positions:
  roles:
{roles_yaml}
  layer_groups:
{layer_groups}
execution:
  batch_size: {batch_size}
""".lstrip(),
        encoding="utf-8",
    )


def _write_lane_d_manifest(
    root: Path,
    num_shards: int,
    *,
    config_path: str = "configs/analysis/autoreg_hidden_state_probe/test.yaml",
    config_sha256: str = "config-sha256-test",
    checkpoint: str = "checkpoint-test",
    artifact_root: str | None = None,
    layer_groups: dict[str, list[int]] | None = None,
    batch_size: int = 1,
) -> None:
    layer_groups = {"early": [0], "last": [-1]} if layer_groups is None else layer_groups
    artifact_root = str(root) if artifact_root is None else artifact_root
    _write_json(
        root / "shards_manifest.json",
        {
            "stage": "hidden_state_probe",
            "num_shards": num_shards,
            "expected_shards": num_shards,
            "shard_labels": [
                lane_d_shard_label(index, num_shards)
                for index in range(num_shards)
            ],
            "config_path": config_path,
            "config_sha256": config_sha256,
            "checkpoint": checkpoint,
            "artifact_root": artifact_root,
            "layer_groups": layer_groups,
            "batch_size": batch_size,
        },
    )


def _lane_c_case(
    case_id: str,
    *,
    source_line_idx: int | bool | str,
    prefix_mode: str | bool = "self_prefix",
    prefix_depth: int | bool | str = 0,
    prefix_quality: object = "gt_prefix",
    top_peak_attribution: object = "target_gt_object",
    target_rank: object = 1,
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "source_line_idx": source_line_idx,
        "prefix_mode": prefix_mode,
        "prefix_depth": prefix_depth,
        "prefix_quality": prefix_quality,
        "intended_target_gt_idx": 0,
        "target_desc": case_id,
        "x1": {
            "top_peak_attribution": top_peak_attribution,
            "target_rank": target_rank,
        },
    }


def _inventory_rows_for_roles(
    *roles: str,
    case_id: str = "case-1",
    source_line_idx: int = 0,
    prefix_mode: str = "self_prefix",
    prefix_depth: int = 2,
    shard_index: int = 0,
    num_shards: int = 8,
    render_source: str = "generated_prefix_replay",
    separator_kind: str = "none_marker_delimited",
) -> list[dict[str, object]]:
    boundary_roles = {
        "prompt_end",
        "row_start",
        "row_end_or_separator",
        "final_generated_prefix_state",
    }

    def prefix_state_kind_for(role: str) -> str:
        if role != "final_generated_prefix_state":
            return "not_applicable"
        if prefix_depth == 0:
            return "empty_prefix_prompt_end"
        if prefix_mode == "teacher_forced":
            return "teacher_forced_prefix_boundary"
        return "generated_prefix_boundary"

    return [
        {
            "case_id": case_id,
            "source_line_idx": source_line_idx,
            "prefix_mode": prefix_mode,
            "prefix_depth": prefix_depth,
            "prefix_quality": "fp_prefix",
            "render_source": render_source,
            "role": role,
            "absolute_token_index": index + 10,
            "prediction_token_index": None if role in boundary_roles else index + 9,
            "assistant_relative_token_index": (
                None
                if role == "prompt_end"
                or prefix_state_kind_for(role) == "empty_prefix_prompt_end"
                else index
            ),
            "assistant_start_token_index": 11,
            "prefix_state_kind": prefix_state_kind_for(role),
            "shard_index": shard_index,
            "num_shards": num_shards,
            "shard_label": lane_d_shard_label(shard_index, num_shards),
            "separator_kind": separator_kind,
            "token_text": f"tok-{index}",
        }
        for index, role in enumerate(roles)
    ]


def _lane_d_probe_row(
    shard_index: int,
    num_shards: int,
    *,
    source_line_idx: int | None = None,
    case_id: str | None = None,
) -> dict[str, object]:
    source = shard_index if source_line_idx is None else source_line_idx
    case = f"case-{source}" if case_id is None else case_id
    return {
        "source_line_idx": source,
        "case_id": case,
        "prefix_mode": "self_prefix",
        "prefix_depth": 2,
        "prefix_quality": "fp_prefix",
        "task": "hidden_scalar",
        "role": "pre_x1",
        "layer_group": "early",
        "layer": 0,
        "model_layer": 0,
        "slot": "x1",
        "target_label_source": "lane_c_target",
        "prefix_condition": "self_prefix",
        "object_count": 5,
        "remaining_count": 3,
        "intended_target_gt_idx": 0,
        "x1_top_peak_attribution": "target_gt_object",
        "x1_target_rank": 1,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "shard_label": lane_d_shard_label(shard_index, num_shards),
        "target_margin": 0.1 + shard_index,
    }


def _lane_d_patch_row(
    shard_index: int,
    num_shards: int,
    *,
    source_line_idx: int | None = None,
    case_id: str | None = None,
) -> dict[str, object]:
    source = shard_index if source_line_idx is None else source_line_idx
    case = f"case-{source}" if case_id is None else case_id
    return {
        "source_line_idx": source,
        "case_id": case,
        "prefix_mode": "self_prefix",
        "prefix_depth": 2,
        "patch_policy": "self_noop",
        "donor_policy": "self",
        "donor_case_id": case,
        "role": "pre_x1",
        "layer_group": "early",
        "model_layer": 0,
        "slot": "x1",
        "shard_index": shard_index,
        "num_shards": num_shards,
        "shard_label": lane_d_shard_label(shard_index, num_shards),
        "target_margin_delta": 0.0,
    }


def _write_lane_d_shard(
    root: Path,
    shard_index: int,
    num_shards: int,
    *,
    config_path: str = "configs/analysis/autoreg_hidden_state_probe/test.yaml",
    config_sha256: str = "config-sha256-test",
    checkpoint: str = "checkpoint-test",
    artifact_root: str | None = None,
    layer_groups: dict[str, list[int]] | None = None,
    batch_size: int = 1,
    duplicate_domain: str | None = None,
    invalid_inventory: bool = False,
    missing_file: str | None = None,
) -> None:
    layer_groups = {"early": [0], "last": [-1]} if layer_groups is None else layer_groups
    artifact_root = str(root) if artifact_root is None else artifact_root
    label = lane_d_shard_label(shard_index, num_shards)
    shard_dir = root / "shards" / label
    source_line_idx = shard_index
    case_id = f"case-{shard_index}"

    selected_cases = [
        {
            "source_line_idx": source_line_idx,
            "case_id": case_id,
            "prefix_mode": "self_prefix",
            "prefix_depth": 2,
            "prefix_quality": "fp_prefix",
            "intended_target_gt_idx": 0,
            "target_desc": f"object-{shard_index}",
            "x1_top_peak_attribution": "target_gt_object",
            "x1_target_rank": 1,
            "shard_index": shard_index,
            "num_shards": num_shards,
            "shard_label": label,
        }
    ]
    if shard_index == 0 and duplicate_domain == "selected_cases":
        selected_cases.append(dict(selected_cases[0]))
    inventory_source_line_idx = shard_index
    inventory_case_id = f"case-{shard_index}"
    position_inventory = _inventory_rows_for_roles(
        *LANE_D_COMPACT_ROLES,
        case_id=inventory_case_id,
        source_line_idx=inventory_source_line_idx,
        shard_index=shard_index,
        num_shards=num_shards,
    )
    if invalid_inventory:
        position_inventory = position_inventory[:-1]
    if shard_index == 0 and duplicate_domain == "position_inventory":
        position_inventory.append(dict(position_inventory[0]))

    probe_source_line_idx = shard_index
    probe_case_id = f"case-{shard_index}"
    probe_rows = [
        _lane_d_probe_row(
            shard_index,
            num_shards,
            source_line_idx=probe_source_line_idx,
            case_id=probe_case_id,
        )
    ]
    if shard_index == 0 and duplicate_domain == "probe_rows":
        probe_rows.append(dict(probe_rows[0]))

    patch_source_line_idx = shard_index
    patch_case_id = f"case-{shard_index}"
    patch_rows = [
        _lane_d_patch_row(
            shard_index,
            num_shards,
            source_line_idx=patch_source_line_idx,
            case_id=patch_case_id,
        )
    ]
    if shard_index == 0 and duplicate_domain == "patch_rows":
        patch_rows.append(dict(patch_rows[0]))
    selected_cases_sha256 = lane_d_probe._sha256_bytes(
        lane_d_probe._canonical_jsonl_bytes(selected_cases)
    )

    files = {
        "selected_cases.jsonl": lambda: _write_jsonl(shard_dir / "selected_cases.jsonl", selected_cases),
        "position_inventory.jsonl": lambda: _write_jsonl(shard_dir / "position_inventory.jsonl", position_inventory),
        "probe_rows.jsonl": lambda: _write_jsonl(shard_dir / "probe_rows.jsonl", probe_rows),
        "patch_rows.jsonl": lambda: _write_jsonl(shard_dir / "patch_rows.jsonl", patch_rows),
        "summary.json": lambda: _write_json(
            shard_dir / "summary.json",
            {
                "stage": "hidden_state_probe",
                "shard_index": shard_index,
                "num_shards": num_shards,
                "shard_label": label,
                "checkpoint": checkpoint,
                "config_path": config_path,
                "config_sha256": config_sha256,
                "artifact_root": artifact_root,
                "layer_groups": layer_groups,
                "batch_size": batch_size,
                "selected_cases_sha256": selected_cases_sha256,
                "runtime_kind": "test_fixture",
                "cuda_visible_devices": None,
                "torch_device": None,
                "base_seed": None,
                "case_seed_policy": "source_line_idx_mod_num_shards",
                "row_counts": {
                    "selected_cases": len(selected_cases),
                    "position_inventory": len(position_inventory),
                    "probe_rows": len(probe_rows),
                    "patch_rows": len(patch_rows),
                },
            },
        ),
    }
    for name, writer in files.items():
        if name != missing_file:
            writer()


def _write_lane_d_shards(
    root: Path,
    *,
    num_shards: int = 2,
    config_path: str = "configs/analysis/autoreg_hidden_state_probe/test.yaml",
    config_sha256: str = "config-sha256-test",
    checkpoint: str = "checkpoint-test",
    artifact_root: str | None = None,
    layer_groups: dict[str, list[int]] | None = None,
    batch_size: int = 1,
    duplicate_domain: str | None = None,
    invalid_inventory: bool = False,
    missing_file: str | None = None,
) -> None:
    layer_groups = {"early": [0], "last": [-1]} if layer_groups is None else layer_groups
    artifact_root = str(root) if artifact_root is None else artifact_root
    _write_lane_d_manifest(
        root,
        num_shards,
        config_path=config_path,
        config_sha256=config_sha256,
        checkpoint=checkpoint,
        artifact_root=artifact_root,
        layer_groups=layer_groups,
        batch_size=batch_size,
    )
    for shard_index in range(num_shards):
        _write_lane_d_shard(
            root,
            shard_index,
            num_shards,
            config_path=config_path,
            config_sha256=config_sha256,
            checkpoint=checkpoint,
            artifact_root=artifact_root,
            layer_groups=layer_groups,
            batch_size=batch_size,
            duplicate_domain=duplicate_domain,
            invalid_inventory=invalid_inventory and shard_index == 1,
            missing_file=missing_file if shard_index == 1 else None,
    )


def _lane_d_shard_file(root: Path, shard_index: int, num_shards: int, filename: str) -> Path:
    return root / "shards" / lane_d_shard_label(shard_index, num_shards) / filename


def _mutate_first_jsonl_row(path: Path, **updates: object) -> None:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows[0].update(updates)
    _write_jsonl(path, rows)


def _delete_first_jsonl_row_key(path: Path, key: str) -> None:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    del rows[0][key]
    _write_jsonl(path, rows)


def _root_merge_outputs(root: Path) -> list[Path]:
    return [
        root / "selected_cases.jsonl",
        root / "position_inventory.jsonl",
        root / "probe_rows.jsonl",
        root / "patch_rows.jsonl",
        root / "summary.json",
        root / "merge_summary.json",
    ]


class _FakeLaneDTokenizer:
    def __init__(self) -> None:
        self._ids_by_token: dict[str, int] = {}
        self._tokens_by_id: dict[int, str] = {}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [self._id(token) for token in self._tokens(str(text))]

    def decode(self, token_ids: list[int] | tuple[int, ...]) -> str:
        return "".join(self._tokens_by_id[int(token_id)] for token_id in token_ids)

    def _id(self, token: str) -> int:
        if token not in self._ids_by_token:
            token_id = 1000 + len(self._ids_by_token)
            self._ids_by_token[token] = token_id
            self._tokens_by_id[token_id] = token
        return self._ids_by_token[token]

    @staticmethod
    def _tokens(text: str) -> list[str]:
        tokens: list[str] = []
        index = 0
        while index < len(text):
            if text.startswith("<|", index):
                end = text.find("|>", index)
                if end != -1:
                    tokens.append(text[index : end + 2])
                    index = end + 2
                    continue
            tokens.append(text[index])
            index += 1
        return tokens


class _SplitBoxStartTokenizer(_FakeLaneDTokenizer):
    @staticmethod
    def _tokens(text: str) -> list[str]:
        tokens: list[str] = []
        index = 0
        box_token = "<|box_start|>"
        while index < len(text):
            if text.startswith(box_token, index):
                tokens.extend(("<|box_", "start|>"))
                index += len(box_token)
                continue
            if text.startswith("<|", index):
                end = text.find("|>", index)
                if end != -1:
                    tokens.append(text[index : end + 2])
                    index = end + 2
                    continue
            tokens.append(text[index])
            index += 1
        return tokens


class _FakeLaneDProcessor:
    def __init__(self, ids_by_text: dict[str, tuple[int, ...]]) -> None:
        self.ids_by_text = ids_by_text

    def __call__(self, *, text: list[str], images: list[object], **_: object) -> dict[str, object]:
        del images
        import torch

        rows = [self.ids_by_text[item] for item in text]
        padded_len = max(len(row) for row in rows)
        input_ids = [
            [0] * (padded_len - len(row)) + list(row)
            for row in rows
        ]
        return {"input_ids": torch.tensor(input_ids, dtype=torch.long)}


class _FakeLaneDModel:
    def __call__(
        self,
        *,
        input_ids: object,
        use_cache: bool,
        output_hidden_states: bool,
        **_: object,
    ) -> SimpleNamespace:
        del use_cache, output_hidden_states
        import torch

        batch_size, seq_len = input_ids.shape
        hidden_states = []
        for tuple_index in range(3):
            tensor = torch.zeros((batch_size, seq_len, 1), dtype=torch.float32)
            for batch_idx in range(batch_size):
                for token_idx in range(seq_len):
                    tensor[batch_idx, token_idx, 0] = (
                        tuple_index * 1000 + batch_idx * 100 + token_idx
                    )
            hidden_states.append(tensor)
        return SimpleNamespace(hidden_states=tuple(hidden_states))


def _compact_row(desc: str, coords: tuple[int, int, int, int]) -> str:
    coord_text = "".join(f"<|coord_{value}|>" for value in coords)
    return f"<|object_ref_start|>{desc}<|box_start|>{coord_text}"


def _fake_prepared_lane_c_example(
    tokenizer: _FakeLaneDTokenizer,
    *,
    assistant_text: str,
    case_id: str,
    prefix_mode: str,
    prefix_depth: int,
    source_line_idx: int = 0,
) -> SimpleNamespace:
    assistant_ids = tuple(tokenizer.encode(assistant_text, add_special_tokens=False))
    return SimpleNamespace(
        row_index=source_line_idx,
        assistant_text=assistant_text,
        full_text=f"full-text:{case_id}",
        full_input_ids=(11, 12, *assistant_ids, 13),
        image_path=Path(f"/tmp/{case_id}.jpg"),
        prefix_condition=prefix_mode,
        prefix_depth=prefix_depth,
        prefix_quality="gt_prefix",
        pairing_id=case_id,
        lane_c_metadata={
            "case_id": case_id,
            "source_line_idx": source_line_idx,
            "prefix_mode": prefix_mode,
            "prefix_depth": prefix_depth,
            "prefix_quality": "gt_prefix",
            "target_desc": "target",
            "gt_bins_by_index": {0: {}, 1: {}},
            "remaining_gt_indices": [0],
        },
    )


def test_lane_d_record_selected_keeps_images_together() -> None:
    assert lane_d_shard_label(0, 8) == "shard_000-of-008"
    assert lane_d_record_selected(0, shard_index=0, num_shards=2)
    assert lane_d_record_selected(2, shard_index=0, num_shards=2)
    assert not lane_d_record_selected(1, shard_index=0, num_shards=2)


def test_lane_d_record_selected_rejects_invalid_source_line_idx() -> None:
    for source_line_idx in (-1, True, "2"):
        with pytest.raises(ValueError, match="source_line_idx"):
            lane_d_record_selected(source_line_idx, shard_index=0, num_shards=2)  # type: ignore[arg-type]


def test_normalize_lane_d_shard_rejects_invalid_args() -> None:
    with pytest.raises(ValueError, match="shard_index"):
        normalize_lane_d_shard(shard_index=2, num_shards=2)
    with pytest.raises(ValueError, match="num_shards"):
        normalize_lane_d_shard(shard_index=0, num_shards=0)


def test_validate_lane_d_position_inventory_rejects_json_roles() -> None:
    rows = _inventory_rows_for_roles(*LANE_D_COMPACT_ROLES)
    validate_lane_d_position_inventory(rows)
    rows.append(
        {
            **rows[-1],
            "role": "bbox_open_bracket",
            "absolute_token_index": 99,
            "assistant_relative_token_index": 88,
        }
    )
    with pytest.raises(ValueError, match="unknown compact role"):
        validate_lane_d_position_inventory(rows)


def test_validate_lane_d_position_inventory_rejects_duplicate_missing_and_negative() -> None:
    with pytest.raises(ValueError, match="missing compact roles"):
        validate_lane_d_position_inventory(_inventory_rows_for_roles("prompt_end"))
    rows = _inventory_rows_for_roles(*LANE_D_COMPACT_ROLES)
    with pytest.raises(ValueError, match="duplicate compact role"):
        validate_lane_d_position_inventory(rows + [dict(rows[0])])
    bad = [dict(row) for row in rows]
    bad[1]["absolute_token_index"] = -1
    with pytest.raises(ValueError, match="nonnegative"):
        validate_lane_d_position_inventory(bad)
    missing_prediction = [dict(row) for row in rows]
    del missing_prediction[2]["prediction_token_index"]
    with pytest.raises(ValueError, match="prediction_token_index"):
        validate_lane_d_position_inventory(missing_prediction)


def test_validate_lane_d_position_inventory_enforces_generated_role_prediction_index() -> None:
    rows = _inventory_rows_for_roles(*LANE_D_COMPACT_ROLES)
    validate_lane_d_position_inventory(rows)

    bad = [dict(row) for row in rows]
    for row in bad:
        if row["role"] == "pre_x1":
            row["prediction_token_index"] = row["absolute_token_index"]
            break
    with pytest.raises(ValueError, match="prediction_token_index"):
        validate_lane_d_position_inventory(bad)


def test_validate_lane_d_position_inventory_rejects_missing_or_invalid_separator_kind() -> None:
    assert LANE_D_SEPARATOR_KINDS == ("none_marker_delimited", "newline")

    rows = _inventory_rows_for_roles(*LANE_D_COMPACT_ROLES)
    missing_separator = [dict(row) for row in rows]
    del missing_separator[0]["separator_kind"]
    with pytest.raises(ValueError, match="separator_kind"):
        validate_lane_d_position_inventory(missing_separator)

    invalid_separator = [dict(row) for row in rows]
    invalid_separator[1]["separator_kind"] = "json_comma"
    with pytest.raises(ValueError, match="separator_kind"):
        validate_lane_d_position_inventory(invalid_separator)


def test_validate_lane_d_position_inventory_rejects_missing_or_invalid_render_source() -> None:
    assert LANE_D_RENDER_SOURCES == (
        "strict_compact_full",
        "lane_c_forced_continuation",
        "generated_prefix_replay",
    )

    rows = _inventory_rows_for_roles(*LANE_D_COMPACT_ROLES)
    missing_render_source = [dict(row) for row in rows]
    del missing_render_source[0]["render_source"]
    with pytest.raises(ValueError, match="render_source"):
        validate_lane_d_position_inventory(missing_render_source)

    invalid_render_source = [dict(row) for row in rows]
    invalid_render_source[1]["render_source"] = "json_oriented_full"
    with pytest.raises(ValueError, match="render_source"):
        validate_lane_d_position_inventory(invalid_render_source)


def test_validate_lane_d_position_inventory_source_aware_teacher_forced_separator() -> None:
    strict_rows = _inventory_rows_for_roles(
        *LANE_D_COMPACT_ROLES,
        prefix_mode="teacher_forced",
        render_source="strict_compact_full",
        separator_kind="newline",
    )

    with pytest.raises(ValueError, match="separator_kind|teacher_forced"):
        validate_lane_d_position_inventory(strict_rows)

    lane_c_rows = _inventory_rows_for_roles(
        *LANE_D_COMPACT_ROLES,
        prefix_mode="teacher_forced",
        render_source="lane_c_forced_continuation",
        separator_kind="newline",
    )
    validate_lane_d_position_inventory(lane_c_rows)

    generated_replay_rows = _inventory_rows_for_roles(
        *LANE_D_COMPACT_ROLES,
        prefix_mode="teacher_forced",
        render_source="generated_prefix_replay",
        separator_kind="newline",
    )
    validate_lane_d_position_inventory(generated_replay_rows)


def test_validate_lane_d_position_inventory_allows_same_role_across_render_sources() -> None:
    strict_rows = _inventory_rows_for_roles(
        *LANE_D_COMPACT_ROLES,
        render_source="strict_compact_full",
    )
    replay_rows = _inventory_rows_for_roles(
        *LANE_D_COMPACT_ROLES,
        render_source="generated_prefix_replay",
    )

    validate_lane_d_position_inventory(strict_rows + replay_rows)


def test_validate_lane_d_position_inventory_validates_prefix_state_kind() -> None:
    assert LANE_D_PREFIX_STATE_KINDS == (
        "not_applicable",
        "empty_prefix_prompt_end",
        "teacher_forced_prefix_boundary",
        "generated_prefix_boundary",
        "partial_row",
    )

    rows = _inventory_rows_for_roles(
        *LANE_D_COMPACT_ROLES,
        prefix_mode="teacher_forced",
        render_source="strict_compact_full",
    )
    validate_lane_d_position_inventory(rows)
    final_state = next(
        row for row in rows if row["role"] == "final_generated_prefix_state"
    )
    assert final_state["prefix_state_kind"] == "teacher_forced_prefix_boundary"

    missing_prefix_state = [dict(row) for row in rows]
    del missing_prefix_state[0]["prefix_state_kind"]
    with pytest.raises(ValueError, match="prefix_state_kind"):
        validate_lane_d_position_inventory(missing_prefix_state)

    invalid_prefix_state = [dict(row) for row in rows]
    invalid_prefix_state[-1]["prefix_state_kind"] = "implicit_boundary"
    with pytest.raises(ValueError, match="prefix_state_kind"):
        validate_lane_d_position_inventory(invalid_prefix_state)

    invalid_nullable_relative = [dict(row) for row in rows]
    invalid_nullable_relative[-1]["assistant_relative_token_index"] = None
    with pytest.raises(ValueError, match="assistant_relative_token_index"):
        validate_lane_d_position_inventory(invalid_nullable_relative)


def test_validate_lane_d_position_inventory_allows_empty_prefix_final_state() -> None:
    rows = _inventory_rows_for_roles(
        *LANE_D_COMPACT_ROLES,
        prefix_depth=0,
        render_source="generated_prefix_replay",
    )
    final_state = next(
        row for row in rows if row["role"] == "final_generated_prefix_state"
    )
    assert final_state["prefix_state_kind"] == "empty_prefix_prompt_end"
    assert final_state["assistant_relative_token_index"] is None
    validate_lane_d_position_inventory(rows)


def test_build_lane_d_position_inventory_for_lane_c_teacher_forced_prefix() -> None:
    tokenizer = _FakeLaneDTokenizer()
    case_id = "row0:teacher_forced:depth1:gt1"
    assistant_text = "\n".join(
        [
            _compact_row("prefix", (1, 2, 3, 4)),
            _compact_row("target", (5, 6, 7, 8)),
        ]
    )
    example = _fake_prepared_lane_c_example(
        tokenizer,
        assistant_text=assistant_text,
        case_id=case_id,
        prefix_mode="teacher_forced",
        prefix_depth=1,
    )
    selected_case = {
        "case_id": case_id,
        "source_line_idx": 0,
        "prefix_mode": "teacher_forced",
        "prefix_depth": 1,
        "prefix_quality": "gt_prefix",
    }

    rows = lane_d_probe.build_lane_d_position_inventory_for_prepared_example(
        example,
        selected_case,
        tokenizer,
        shard_index=0,
        num_shards=2,
        shard_label=lane_d_shard_label(0, 2),
    )

    assert [row["role"] for row in rows] == list(LANE_D_COMPACT_ROLES)
    assert {row["render_source"] for row in rows} == {"lane_c_forced_continuation"}
    assert {row["separator_kind"] for row in rows} == {"newline"}
    assert all(not str(row["role"]).startswith("bbox_") for row in rows)
    assert all(row["prefix_state_kind"] == "not_applicable" for row in rows[:-1])
    assert rows[-1]["role"] == "final_generated_prefix_state"
    assert rows[-1]["prefix_state_kind"] == "teacher_forced_prefix_boundary"
    assert rows[-1]["assistant_relative_token_index"] is not None
    validate_lane_d_position_inventory(rows)


def test_build_lane_d_position_inventory_for_empty_prefix_final_state() -> None:
    tokenizer = _FakeLaneDTokenizer()
    case_id = "row0:self_prefix:depth0:gt0"
    example = _fake_prepared_lane_c_example(
        tokenizer,
        assistant_text=_compact_row("target", (5, 6, 7, 8)),
        case_id=case_id,
        prefix_mode="self_prefix",
        prefix_depth=0,
    )
    selected_case = {
        "case_id": case_id,
        "source_line_idx": 0,
        "prefix_mode": "self_prefix",
        "prefix_depth": 0,
        "prefix_quality": "gt_prefix",
    }

    rows = lane_d_probe.build_lane_d_position_inventory_for_prepared_example(
        example,
        selected_case,
        tokenizer,
        shard_index=0,
        num_shards=2,
        shard_label=lane_d_shard_label(0, 2),
    )

    prompt_end = next(row for row in rows if row["role"] == "prompt_end")
    final_state = next(
        row for row in rows if row["role"] == "final_generated_prefix_state"
    )
    assert final_state["prefix_state_kind"] == "empty_prefix_prompt_end"
    assert final_state["absolute_token_index"] == prompt_end["absolute_token_index"]
    assert final_state["assistant_relative_token_index"] is None
    validate_lane_d_position_inventory(rows)


def test_build_lane_d_position_inventory_marks_actual_separator_kind() -> None:
    tokenizer = _FakeLaneDTokenizer()
    target_only_case = "row0:self_prefix:depth0:gt0"
    target_only = _fake_prepared_lane_c_example(
        tokenizer,
        assistant_text=_compact_row("target", (5, 6, 7, 8)),
        case_id=target_only_case,
        prefix_mode="self_prefix",
        prefix_depth=0,
    )
    target_only_rows = lane_d_probe.build_lane_d_position_inventory_for_prepared_example(
        target_only,
        {
            "case_id": target_only_case,
            "source_line_idx": 0,
            "prefix_mode": "self_prefix",
            "prefix_depth": 0,
            "prefix_quality": "empty_prefix",
        },
        tokenizer,
        shard_index=0,
        num_shards=2,
        shard_label=lane_d_shard_label(0, 2),
    )
    assert {row["separator_kind"] for row in target_only_rows} == {
        "none_marker_delimited"
    }
    assert {row["row_end_source_kind"] for row in target_only_rows} == {
        "last_target_row_token"
    }

    prefixed_case = "row0:teacher_forced:depth1:gt1"
    prefixed = _fake_prepared_lane_c_example(
        tokenizer,
        assistant_text="\n".join(
            [
                _compact_row("prefix", (1, 2, 3, 4)),
                _compact_row("target", (5, 6, 7, 8)),
            ]
        ),
        case_id=prefixed_case,
        prefix_mode="teacher_forced",
        prefix_depth=1,
    )
    prefixed_rows = lane_d_probe.build_lane_d_position_inventory_for_prepared_example(
        prefixed,
        {
            "case_id": prefixed_case,
            "source_line_idx": 0,
            "prefix_mode": "teacher_forced",
            "prefix_depth": 1,
            "prefix_quality": "gt_prefix",
        },
        tokenizer,
        shard_index=0,
        num_shards=2,
        shard_label=lane_d_shard_label(0, 2),
    )
    assert {row["separator_kind"] for row in prefixed_rows} == {"newline"}


def test_build_lane_d_position_inventory_uses_box_start_as_post_marker_state() -> None:
    tokenizer = _SplitBoxStartTokenizer()
    case_id = "row0:self_prefix:depth0:gt0"
    example = _fake_prepared_lane_c_example(
        tokenizer,
        assistant_text=_compact_row("target", (5, 6, 7, 8)),
        case_id=case_id,
        prefix_mode="self_prefix",
        prefix_depth=0,
    )

    rows = lane_d_probe.build_lane_d_position_inventory_for_prepared_example(
        example,
        {
            "case_id": case_id,
            "source_line_idx": 0,
            "prefix_mode": "self_prefix",
            "prefix_depth": 0,
            "prefix_quality": "empty_prefix",
        },
        tokenizer,
        shard_index=0,
        num_shards=2,
        shard_label=lane_d_shard_label(0, 2),
    )

    by_role = {row["role"]: row for row in rows}
    assert by_role["desc_end"]["token_text"] == "<|box_"
    assert by_role["box_start"]["token_text"] == "<|coord_5|>"
    assert by_role["pre_x1"]["token_text"] == "<|coord_5|>"
    assert (
        by_role["box_start"]["absolute_token_index"]
        == by_role["pre_x1"]["absolute_token_index"]
    )
    assert (
        by_role["box_start"]["prediction_token_index"]
        == by_role["pre_x1"]["prediction_token_index"]
    )
    assert (
        by_role["desc_end"]["prediction_token_index"]
        < by_role["box_start"]["prediction_token_index"]
    )


def test_build_lane_d_position_inventory_rejects_duplicate_assistant_subsequence() -> None:
    tokenizer = _FakeLaneDTokenizer()
    case_id = "row0:self_prefix:depth0:gt0"
    assistant_text = _compact_row("target", (5, 5, 5, 5))
    assistant_ids = tuple(tokenizer.encode(assistant_text, add_special_tokens=False))
    example = _fake_prepared_lane_c_example(
        tokenizer,
        assistant_text=assistant_text,
        case_id=case_id,
        prefix_mode="self_prefix",
        prefix_depth=0,
    )
    example.full_input_ids = (11, 12, *assistant_ids, 13, *assistant_ids, 14)

    with pytest.raises(ValueError, match="not unique"):
        lane_d_probe.build_lane_d_position_inventory_for_prepared_example(
            example,
            {
                "case_id": case_id,
                "source_line_idx": 0,
                "prefix_mode": "self_prefix",
                "prefix_depth": 0,
                "prefix_quality": "empty_prefix",
            },
            tokenizer,
            shard_index=0,
            num_shards=2,
            shard_label=lane_d_shard_label(0, 2),
        )


def test_validate_lane_d_position_inventory_rejects_missing_or_invalid_metadata() -> None:
    rows = _inventory_rows_for_roles(*LANE_D_COMPACT_ROLES)

    missing_shard_label = [dict(row) for row in rows]
    del missing_shard_label[0]["shard_label"]
    with pytest.raises(ValueError, match="shard_label"):
        validate_lane_d_position_inventory(missing_shard_label)

    missing_relative = [dict(row) for row in rows]
    del missing_relative[1]["assistant_relative_token_index"]
    with pytest.raises(ValueError, match="assistant_relative_token_index"):
        validate_lane_d_position_inventory(missing_relative)

    negative_assistant_start = [dict(row) for row in rows]
    negative_assistant_start[2]["assistant_start_token_index"] = -1
    with pytest.raises(ValueError, match="assistant_start_token_index"):
        validate_lane_d_position_inventory(negative_assistant_start)

    mismatched_shard_label = [dict(row) for row in rows]
    mismatched_shard_label[3]["shard_label"] = "shard_999-of-008"
    with pytest.raises(ValueError, match="shard_label"):
        validate_lane_d_position_inventory(mismatched_shard_label)


def test_collect_lane_d_hidden_scalar_probe_rows_uses_hidden_indices_and_padding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch
    import src.analysis.hard_ce_coord_logit_locality as lane_c_module

    config_path = tmp_path / "lane_d.yaml"
    _write_lane_d_config(
        config_path,
        layer_groups_yaml="    first: [0]\n    final: [-1]",
        batch_size=2,
    )
    config = load_lane_d_config(config_path)
    examples = [
        SimpleNamespace(
            full_text="case-0-text",
            full_input_ids=(10, 11, 12, 13),
            image_path=tmp_path / "case-0.jpg",
            prefix_condition="self_prefix",
            pairing_id="case-0",
            lane_c_metadata={
                "case_id": "case-0",
                "gt_bins_by_index": {0: {}, 1: {}},
                "remaining_gt_indices": [1],
            },
        ),
        SimpleNamespace(
            full_text="case-1-text",
            full_input_ids=(20, 21, 22, 23, 24, 25),
            image_path=tmp_path / "case-1.jpg",
            prefix_condition="teacher_forced",
            pairing_id="case-1",
            lane_c_metadata={
                "case_id": "case-1",
                "gt_bins_by_index": {0: {}, 1: {}, 2: {}},
                "remaining_gt_indices": [1, 2],
            },
        ),
    ]
    selected_cases = [
        {
            "case_id": "case-0",
            "source_line_idx": 0,
            "prefix_mode": "self_prefix",
            "prefix_depth": 0,
            "prefix_quality": "empty_prefix",
            "intended_target_gt_idx": 0,
            "x1_top_peak_attribution": "target_gt_object",
            "x1_target_rank": 1,
        },
        {
            "case_id": "case-1",
            "source_line_idx": 1,
            "prefix_mode": "teacher_forced",
            "prefix_depth": 1,
            "prefix_quality": "gt_prefix",
            "intended_target_gt_idx": 2,
            "x1_top_peak_attribution": "same_desc_competitor_gt_object",
            "x1_target_rank": 7,
        },
    ]

    def inventory_row(
        *,
        selected_case: dict[str, object],
        role: str,
        absolute: int,
        prediction: int | None,
    ) -> dict[str, object]:
        return {
            "source_line_idx": selected_case["source_line_idx"],
            "case_id": selected_case["case_id"],
            "prefix_mode": selected_case["prefix_mode"],
            "prefix_depth": selected_case["prefix_depth"],
            "prefix_quality": selected_case["prefix_quality"],
            "render_source": "lane_c_forced_continuation",
            "role": role,
            "absolute_token_index": absolute,
            "prediction_token_index": prediction,
            "assistant_relative_token_index": absolute,
            "assistant_start_token_index": 0,
            "prefix_state_kind": "not_applicable",
            "shard_index": selected_case["source_line_idx"],
            "num_shards": 2,
            "shard_label": lane_d_shard_label(int(selected_case["source_line_idx"]), 2),
            "separator_kind": "none_marker_delimited",
            "token_text": role,
        }

    position_inventory = [
        inventory_row(
            selected_case=selected_cases[0],
            role="row_start",
            absolute=1,
            prediction=None,
        ),
        inventory_row(
            selected_case=selected_cases[0],
            role="pre_x1",
            absolute=3,
            prediction=2,
        ),
        inventory_row(
            selected_case=selected_cases[1],
            role="row_start",
            absolute=1,
            prediction=None,
        ),
        inventory_row(
            selected_case=selected_cases[1],
            role="pre_x1",
            absolute=4,
            prediction=3,
        ),
    ]
    model_handle = SimpleNamespace(
        processor=_FakeLaneDProcessor(
            {example.full_text: example.full_input_ids for example in examples}
        ),
        model=_FakeLaneDModel(),
    )
    monkeypatch.setattr(lane_c_module, "_load_image", lambda path: object())
    monkeypatch.setattr(lane_c_module, "_model_device", lambda model: torch.device("cpu"))

    rows = lane_d_probe._collect_lane_d_hidden_scalar_probe_rows(
        config,
        selected_example_pairs=list(zip(selected_cases, examples)),
        position_inventory=position_inventory,
        model_handle=model_handle,
    )

    assert len(rows) == 8
    by_key = {
        (
            row["case_id"],
            row["role"],
            row["layer_group"],
        ): row
        for row in rows
    }
    assert by_key[("case-0", "row_start", "first")]["hidden_token_index"] == 1
    assert by_key[("case-0", "row_start", "first")]["tensor_token_index"] == 3
    assert by_key[("case-0", "pre_x1", "first")]["hidden_token_index"] == 2
    assert by_key[("case-0", "pre_x1", "first")]["tensor_token_index"] == 4
    assert by_key[("case-1", "pre_x1", "first")]["tensor_token_index"] == 3
    assert by_key[("case-0", "pre_x1", "first")]["hidden_state_tuple_index"] == 1
    assert by_key[("case-0", "pre_x1", "final")]["hidden_state_tuple_index"] == 2
    assert by_key[("case-0", "pre_x1", "first")]["hidden_norm"] == 1004.0
    assert by_key[("case-1", "pre_x1", "final")]["hidden_norm"] == 2103.0
    assert by_key[("case-1", "pre_x1", "final")]["layer"] == 1
    assert by_key[("case-1", "pre_x1", "final")]["prefix_condition"] == "teacher_forced"
    assert by_key[("case-1", "pre_x1", "final")]["object_count"] == 3
    assert by_key[("case-1", "pre_x1", "final")]["remaining_count"] == 2
    assert by_key[("case-1", "pre_x1", "final")]["x1_target_rank"] == 7


def test_validate_lane_d_position_inventory_rejects_mixed_context_under_same_case_id() -> None:
    rows = _inventory_rows_for_roles(*LANE_D_COMPACT_ROLES)
    rows[3]["prefix_mode"] = "teacher_forced"

    with pytest.raises(ValueError, match="case-1.*missing compact roles|missing compact roles.*case-1"):
        validate_lane_d_position_inventory(rows)


def test_select_lane_d_cases_prefers_x1_failure_cohorts_and_preserves_case_fields(
    tmp_path: Path,
) -> None:
    path = tmp_path / "per_case.jsonl"
    _write_jsonl(
        path,
        [
            {
                "case_id": "row0:self_prefix:depth2:gt19",
                "source_line_idx": 0,
                "prefix_mode": "self_prefix",
                "prefix_depth": 2,
                "prefix_quality": "fp_prefix",
                "intended_target_gt_idx": 19,
                "target_desc": "vase",
                "x1": {
                    "top_peak_attribution": "same_desc_competitor_gt_object",
                    "target_rank": 350,
                },
            },
            {
                "case_id": "row1:teacher_forced:depth0:gt0",
                "source_line_idx": 1,
                "prefix_mode": "teacher_forced",
                "prefix_depth": 0,
                "prefix_quality": "gt_prefix",
                "intended_target_gt_idx": 0,
                "target_desc": "person",
                "x1": {"top_peak_attribution": "target_gt_object", "target_rank": 1},
            },
        ],
    )
    rows = select_lane_d_cases(path, max_cases=1, shard_index=None, num_shards=None)
    assert rows == [
        {
            "case_id": "row0:self_prefix:depth2:gt19",
            "source_line_idx": 0,
            "prefix_mode": "self_prefix",
            "prefix_depth": 2,
            "prefix_quality": "fp_prefix",
            "intended_target_gt_idx": 19,
            "target_desc": "vase",
            "x1_top_peak_attribution": "same_desc_competitor_gt_object",
            "x1_target_rank": 350,
            "selection_reason": "x1_non_target_or_low_rank",
        }
    ]


def test_select_lane_d_cases_applies_source_line_sharding_and_deterministic_priority_sort(
    tmp_path: Path,
) -> None:
    path = tmp_path / "per_case.jsonl"
    _write_jsonl(
        path,
        [
            {
                "case_id": "row4:z_mode:depth1:gt0",
                "source_line_idx": 4,
                "prefix_mode": "z_mode",
                "prefix_depth": 1,
                "prefix_quality": "gt_prefix",
                "intended_target_gt_idx": 0,
                "target_desc": "ordinary-z",
                "x1": {"top_peak_attribution": "target_gt_object", "target_rank": 1},
            },
            {
                "case_id": "row6:teacher_forced:depth0:gt2",
                "source_line_idx": 6,
                "prefix_mode": "teacher_forced",
                "prefix_depth": 0,
                "prefix_quality": "gt_prefix",
                "intended_target_gt_idx": 2,
                "target_desc": "rank-failure",
                "x1": {"top_peak_attribution": "target_gt_object", "target_rank": 99},
            },
            {
                "case_id": "row3:self_prefix:depth0:gt1",
                "source_line_idx": 3,
                "prefix_mode": "self_prefix",
                "prefix_depth": 0,
                "prefix_quality": "fp_prefix",
                "intended_target_gt_idx": 1,
                "target_desc": "wrong-shard",
                "x1": {
                    "top_peak_attribution": "same_desc_competitor_gt_object",
                    "target_rank": 500,
                },
            },
            {
                "case_id": "row2:self_prefix:depth2:gt5",
                "source_line_idx": 2,
                "prefix_mode": "self_prefix",
                "prefix_depth": 2,
                "prefix_quality": "fp_prefix",
                "intended_target_gt_idx": 5,
                "target_desc": "non-target",
                "x1": {
                    "top_peak_attribution": "same_desc_competitor_gt_object",
                    "target_rank": 12,
                },
            },
            {
                "case_id": "row4:a_mode:depth0:gt3",
                "source_line_idx": 4,
                "prefix_mode": "a_mode",
                "prefix_depth": 0,
                "prefix_quality": "gt_prefix",
                "intended_target_gt_idx": 3,
                "target_desc": "ordinary-a",
                "x1": {"top_peak_attribution": "target_gt_object", "target_rank": 1},
            },
        ],
    )

    rows = select_lane_d_cases(path, max_cases=None, shard_index=0, num_shards=2)

    assert [row["case_id"] for row in rows] == [
        "row2:self_prefix:depth2:gt5",
        "row6:teacher_forced:depth0:gt2",
        "row4:a_mode:depth0:gt3",
        "row4:z_mode:depth1:gt0",
    ]
    assert {row["source_line_idx"] % 2 for row in rows} == {0}
    assert rows[1]["x1_top_peak_attribution"] == "target_gt_object"
    assert rows[1]["x1_target_rank"] == 99
    assert rows[1]["selection_reason"] == "x1_non_target_or_low_rank"
    assert rows[2]["selection_reason"] == "lane_c_case"
    assert rows[3]["selection_reason"] == "lane_c_case"


def test_select_lane_d_cases_rejects_malformed_lane_c_schema(tmp_path: Path) -> None:
    cases = [
        ([{**_lane_c_case("negative-source", source_line_idx=-1)}], "source_line_idx"),
        ([{**_lane_c_case("bool-source", source_line_idx=True)}], "source_line_idx"),
        ([{**_lane_c_case("missing-prefix-mode", source_line_idx=0), "prefix_mode": ""}], "prefix_mode"),
        ([{**_lane_c_case("bool-prefix-mode", source_line_idx=0), "prefix_mode": True}], "prefix_mode"),
        ([{**_lane_c_case("bool-prefix-depth", source_line_idx=0), "prefix_depth": True}], "prefix_depth"),
        ([{**_lane_c_case("negative-prefix-depth", source_line_idx=0, prefix_depth=-1)}], "prefix_depth"),
        ([{**_lane_c_case("missing-x1", source_line_idx=0), "x1": None}], "x1"),
        (
            [
                {
                    **_lane_c_case("missing-top-peak", source_line_idx=0),
                    "x1": {"target_rank": 1},
                }
            ],
            "top_peak_attribution",
        ),
        (
            [
                {
                    **_lane_c_case("bool-rank", source_line_idx=0),
                    "x1": {"top_peak_attribution": "target_gt_object", "target_rank": True},
                }
            ],
            "target_rank",
        ),
    ]

    for index, (rows, expected_error) in enumerate(cases):
        path = tmp_path / f"bad_{index}.jsonl"
        _write_jsonl(path, rows)
        with pytest.raises(ValueError, match=expected_error):
            select_lane_d_cases(path, max_cases=None, shard_index=None, num_shards=None)


def test_filter_lane_d_examples_for_selected_cases_rejects_missing_case_id() -> None:
    tokenizer = _FakeLaneDTokenizer()
    example = _fake_prepared_lane_c_example(
        tokenizer,
        assistant_text=_compact_row("target", (5, 6, 7, 8)),
        case_id="case-present",
        prefix_mode="self_prefix",
        prefix_depth=0,
    )
    selected_cases = [
        {"case_id": "case-present", "source_line_idx": 0},
        {"case_id": "case-missing", "source_line_idx": 2},
    ]

    with pytest.raises(ValueError, match="case-missing"):
        lane_d_probe.filter_lane_d_examples_for_selected_cases(
            [example],
            selected_cases,
        )


def test_load_lane_d_config_parses_paths_roles_and_layer_groups(tmp_path: Path) -> None:
    config_path = tmp_path / "lane_d.yaml"
    _write_lane_d_config(config_path)

    config = load_lane_d_config(config_path)

    assert config.config_path == config_path
    assert config.paths.artifact_root == tmp_path / "artifact_root"
    assert config.paths.checkpoint == tmp_path / "checkpoint"
    assert config.paths.dataset_jsonl == tmp_path / "dataset.jsonl"
    assert config.paths.self_rollout_root == tmp_path / "self_rollout"
    assert config.paths.lane_a_rollout_root == tmp_path / "lane_a"
    assert config.paths.lane_c_per_case == tmp_path / "per_case.jsonl"
    assert config.paths.lane_c_study_config == tmp_path / "lane_c.yaml"
    assert config.selection.max_cases == 512
    assert config.positions.roles == LANE_D_COMPACT_ROLES
    assert config.positions.layer_groups == {"early": (0, 1), "last": (-4, -3)}
    assert config.execution.batch_size == 1


def test_default_lane_d_config_points_to_lane_c_study_config() -> None:
    config = load_lane_d_config(
        "configs/analysis/autoreg_hidden_state_probe/ckpt3664_lane_d_val200.yaml"
    )

    assert config.paths.lane_c_study_config == Path(
        "/data/CoordExp/configs/analysis/hard_ce_coord_logit_locality/"
        "ckpt3664_lane_c_val200.yaml"
    )


@pytest.mark.parametrize(
    ("roles", "expected_error"),
    [
        (("not_a_lane_d_role", *LANE_D_COMPACT_ROLES[1:]), "unknown compact role"),
        ((*LANE_D_COMPACT_ROLES, "prompt_end"), "duplicate compact role"),
        (LANE_D_COMPACT_ROLES[:-1], "missing compact roles"),
    ],
)
def test_load_lane_d_config_rejects_unknown_duplicate_and_missing_roles(
    tmp_path: Path,
    roles: tuple[str, ...],
    expected_error: str,
) -> None:
    config_path = tmp_path / "lane_d.yaml"
    _write_lane_d_config(config_path, roles=roles)

    with pytest.raises(ValueError, match=expected_error):
        load_lane_d_config(config_path)


@pytest.mark.parametrize(
    ("layer_groups_yaml", "expected_error"),
    [
        ("    {}\n", "at least one layer group"),
        ("    empty: []\n", "must include at least one layer"),
    ],
)
def test_load_lane_d_config_rejects_empty_layer_groups(
    tmp_path: Path,
    layer_groups_yaml: str,
    expected_error: str,
) -> None:
    config_path = tmp_path / "lane_d.yaml"
    _write_lane_d_config(config_path, layer_groups_yaml=layer_groups_yaml)

    with pytest.raises(ValueError, match=expected_error):
        load_lane_d_config(config_path)


def test_build_lane_d_dry_run_plan_lists_all_shards_and_expected_outputs(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "lane_d.yaml"
    _write_lane_d_config(config_path)
    config = load_lane_d_config(config_path)

    plan = build_lane_d_dry_run_plan(
        config,
        stages=("select_cases", "position_inventory"),
        shard_index=None,
        num_shards=8,
    )

    shards = plan["selected_shards"]
    shard_dirs = [item["shard_dir"] for item in shards]
    shard_labels = [item["shard_label"] for item in shards]

    assert json.loads(json.dumps(plan)) == plan
    assert plan["config_path"] == str(config_path)
    assert plan["artifact_root"] == str(tmp_path / "artifact_root")
    assert plan["stages"] == ["select_cases", "position_inventory"]
    assert plan["dry_run"] is True
    assert plan["merge_shards"] is False
    assert len(shards) == 8
    assert shard_labels == [lane_d_shard_label(index, 8) for index in range(8)]
    assert len(set(shard_dirs)) == 8
    assert all(label in shard_dir for label, shard_dir in zip(shard_labels, shard_dirs))
    for item in shards:
        shard_dir = Path(item["shard_dir"])
        assert item["selected_cases"] == str(shard_dir / "selected_cases.jsonl")
        assert item["position_inventory"] == str(shard_dir / "position_inventory.jsonl")
        assert item["probe_rows"] == str(shard_dir / "probe_rows.jsonl")
        assert item["patch_rows"] == str(shard_dir / "patch_rows.jsonl")
        assert item["summary"] == str(shard_dir / "summary.json")


def test_materialize_lane_d_select_cases_shard_writes_manifest_and_placeholders(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "lane_d.yaml"
    _write_lane_d_config(config_path, max_cases=10)
    _write_jsonl(
        tmp_path / "per_case.jsonl",
        [
            _lane_c_case("case-0", source_line_idx=0),
            _lane_c_case("case-1", source_line_idx=1),
            _lane_c_case(
                "case-2",
                source_line_idx=2,
                top_peak_attribution="same_desc_competitor_gt_object",
                target_rank=12,
            ),
            _lane_c_case("case-3", source_line_idx=3),
            _lane_c_case("case-4", source_line_idx=4),
        ],
    )
    config = load_lane_d_config(config_path)
    shard_label = lane_d_shard_label(0, 2)

    manifest = write_lane_d_shards_manifest(config, num_shards=2)
    summary = materialize_lane_d_select_cases_shard(
        config,
        shard_index=0,
        num_shards=2,
    )

    root = tmp_path / "artifact_root"
    shard_dir = root / "shards" / shard_label
    selected_rows = _read_jsonl(shard_dir / "selected_cases.jsonl")
    saved_summary = _read_json(shard_dir / "summary.json")

    assert _read_json(root / "shards_manifest.json") == manifest
    assert saved_summary == summary
    assert manifest["stage"] == "hidden_state_probe"
    assert manifest["num_shards"] == 2
    assert manifest["expected_shards"] == 2
    assert manifest["shard_labels"] == [
        lane_d_shard_label(0, 2),
        lane_d_shard_label(1, 2),
    ]
    assert [
        row["source_line_idx"] for row in selected_rows
    ] == [2, 0, 4]
    assert {row["source_line_idx"] % 2 for row in selected_rows} == {0}
    assert {
        (row["shard_index"], row["num_shards"], row["shard_label"])
        for row in selected_rows
    } == {(0, 2, shard_label)}
    for filename in (
        "position_inventory.jsonl",
        "probe_rows.jsonl",
        "patch_rows.jsonl",
    ):
        assert (shard_dir / filename).exists()
        assert _read_jsonl(shard_dir / filename) == []
    assert summary["stage"] == "hidden_state_probe"
    assert summary["shard_index"] == 0
    assert summary["num_shards"] == 2
    assert summary["shard_label"] == shard_label
    assert summary["stages_completed"] == ["select_cases"]
    assert summary["row_counts"] == {
        "selected_cases": 3,
        "position_inventory": 0,
        "probe_rows": 0,
        "patch_rows": 0,
    }
    assert {
        key: summary[key]
        for key in lane_d_config_expected_merge_metadata(config)
    } == lane_d_config_expected_merge_metadata(config)


def test_materialized_lane_d_select_case_shards_merge_with_placeholders(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "lane_d.yaml"
    _write_lane_d_config(config_path, max_cases=10)
    _write_jsonl(
        tmp_path / "per_case.jsonl",
        [
            _lane_c_case("case-0", source_line_idx=0),
            _lane_c_case("case-1", source_line_idx=1),
            _lane_c_case("case-2", source_line_idx=2),
            _lane_c_case(
                "case-3",
                source_line_idx=3,
                top_peak_attribution="same_desc_competitor_gt_object",
                target_rank=7,
            ),
        ],
    )
    config = load_lane_d_config(config_path)

    write_lane_d_shards_manifest(config, num_shards=2)
    for shard_index in range(2):
        materialize_lane_d_select_cases_shard(
            config,
            shard_index=shard_index,
            num_shards=2,
        )

    summary = merge_lane_d_shards(
        config.paths.artifact_root,
        expected_shards=2,
        expected_metadata=lane_d_config_expected_merge_metadata(config),
    )

    assert summary["row_counts"] == {
        "selected_cases": 4,
        "position_inventory": 0,
        "probe_rows": 0,
        "patch_rows": 0,
    }
    assert _read_jsonl(config.paths.artifact_root / "position_inventory.jsonl") == []
    assert _read_jsonl(config.paths.artifact_root / "probe_rows.jsonl") == []
    assert _read_jsonl(config.paths.artifact_root / "patch_rows.jsonl") == []


def test_merge_lane_d_shards_writes_root_jsonls_and_summaries(tmp_path: Path) -> None:
    root = tmp_path / "hidden_state_probe"
    _write_lane_d_shards(root, num_shards=2)

    summary = merge_lane_d_shards(root, expected_shards=2)

    assert summary["stage"] == "hidden_state_probe"
    assert summary["expected_shards"] == 2
    assert summary["merged_shards"] == [
        "shard_000-of-002",
        "shard_001-of-002",
    ]
    assert summary["row_counts"] == {
        "selected_cases": 2,
        "position_inventory": 2 * len(LANE_D_COMPACT_ROLES),
        "probe_rows": 2,
        "patch_rows": 2,
    }
    for name, count in summary["row_counts"].items():
        path = root / f"{name}.jsonl"
        assert path.exists()
        assert len(path.read_text(encoding="utf-8").splitlines()) == count
        assert summary["output_paths"][name] == str(path)
    assert json.loads((root / "merge_summary.json").read_text(encoding="utf-8")) == summary
    assert json.loads((root / "summary.json").read_text(encoding="utf-8")) == summary
    assert summary["selected_cases_sha256"]
    assert len(summary["source_summaries"]) == 2


def test_merge_lane_d_shards_rejects_missing_manifest(tmp_path: Path) -> None:
    root = tmp_path / "hidden_state_probe"
    _write_lane_d_shards(root, num_shards=2)
    (root / "shards_manifest.json").unlink()

    with pytest.raises(FileNotFoundError, match="shards_manifest.json"):
        merge_lane_d_shards(root, expected_shards=2)


@pytest.mark.parametrize(
    ("manifest_update", "expected_error"),
    [
        ({}, "num_shards|expected_shards"),
        ({"num_shards": 3, "expected_shards": 3}, "num_shards"),
        (
            {
                "num_shards": 2,
                "expected_shards": 2,
                "config_path": "configs/analysis/autoreg_hidden_state_probe/test.yaml",
                "config_sha256": "config-sha256-test",
                "checkpoint": "checkpoint-test",
                "artifact_root": "__ROOT__",
                "layer_groups": {"early": [0], "last": [-1]},
                "batch_size": 1,
            },
            "shard labels",
        ),
        (
            {
                "num_shards": 2,
                "expected_shards": 2,
                "shard_labels": ["shard_001-of-002", "shard_000-of-002"],
                "config_path": "configs/analysis/autoreg_hidden_state_probe/test.yaml",
                "config_sha256": "config-sha256-test",
                "checkpoint": "checkpoint-test",
                "artifact_root": "__ROOT__",
                "layer_groups": {"early": [0], "last": [-1]},
                "batch_size": 1,
            },
            "shard labels",
        ),
    ],
)
def test_merge_lane_d_shards_requires_strict_manifest_contract(
    tmp_path: Path,
    manifest_update: dict[str, object],
    expected_error: str,
) -> None:
    root = tmp_path / "hidden_state_probe"
    _write_lane_d_shards(root, num_shards=2)
    manifest_update = {
        key: (str(root) if value == "__ROOT__" else value)
        for key, value in manifest_update.items()
    }
    _write_json(root / "shards_manifest.json", manifest_update)

    with pytest.raises(ValueError, match=expected_error):
        merge_lane_d_shards(root, expected_shards=2)


def test_merge_lane_d_shards_rejects_missing_unexpected_and_malformed_dirs(
    tmp_path: Path,
) -> None:
    missing_root = tmp_path / "missing"
    _write_lane_d_manifest(missing_root, 2)
    _write_lane_d_shard(missing_root, 0, 2)
    with pytest.raises(ValueError, match="missing Lane D shard dirs"):
        merge_lane_d_shards(missing_root, expected_shards=2)

    unexpected_root = tmp_path / "unexpected"
    _write_lane_d_shards(unexpected_root, num_shards=2)
    (unexpected_root / "shards" / "shard_002-of-002").mkdir()
    with pytest.raises(ValueError, match="unexpected Lane D shard dirs"):
        merge_lane_d_shards(unexpected_root, expected_shards=2)

    malformed_root = tmp_path / "malformed"
    _write_lane_d_shards(malformed_root, num_shards=2)
    (malformed_root / "shards" / "lane_c_shard_000-of-002").mkdir()
    with pytest.raises(ValueError, match="malformed Lane D shard dirs"):
        merge_lane_d_shards(malformed_root, expected_shards=2)


@pytest.mark.parametrize("missing_file", ["probe_rows.jsonl", "summary.json"])
def test_merge_lane_d_shards_rejects_missing_per_shard_files(
    tmp_path: Path,
    missing_file: str,
) -> None:
    root = tmp_path / "hidden_state_probe"
    _write_lane_d_shards(root, num_shards=2, missing_file=missing_file)

    with pytest.raises(FileNotFoundError, match="missing files"):
        merge_lane_d_shards(root, expected_shards=2)


@pytest.mark.parametrize(
    ("summary_update", "expected_error"),
    [
        ({"shard_label": "shard_999-of-002"}, "shard_label"),
        ({"num_shards": 3}, "num_shards"),
        ({"shard_index": 99}, "shard_index"),
        ({"stage": "other_stage"}, "stage"),
        (
            {
                "row_counts": {
                    "selected_cases": 99,
                    "position_inventory": len(LANE_D_COMPACT_ROLES),
                    "probe_rows": 1,
                    "patch_rows": 1,
                }
            },
            "row_counts.*selected_cases",
        ),
        (
            {"config_path": "configs/analysis/autoreg_hidden_state_probe/other.yaml"},
            "config_path",
        ),
        ({"checkpoint": "checkpoint-other"}, "checkpoint"),
        ({"layer_groups": {"middle": [12]}}, "layer_groups"),
    ],
)
def test_merge_lane_d_shards_validates_per_shard_summary_contract(
    tmp_path: Path,
    summary_update: dict[str, object],
    expected_error: str,
) -> None:
    root = tmp_path / "hidden_state_probe"
    _write_lane_d_shards(root, num_shards=2)
    _mutate_json(
        _lane_d_shard_file(root, 1, 2, "summary.json"),
        **summary_update,
    )

    with pytest.raises(ValueError, match=expected_error):
        merge_lane_d_shards(root, expected_shards=2)


def test_merge_lane_d_shards_rejects_expected_metadata_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "hidden_state_probe"
    _write_lane_d_shards(root, num_shards=2)

    with pytest.raises(ValueError, match="config_path"):
        merge_lane_d_shards(
            root,
            expected_shards=2,
            expected_metadata={
                "config_path": "configs/analysis/autoreg_hidden_state_probe/other.yaml",
                "config_sha256": "config-sha256-test",
                "checkpoint": "checkpoint-test",
                "artifact_root": str(root),
                "layer_groups": {"early": [0], "last": [-1]},
                "batch_size": 1,
            },
        )


@pytest.mark.parametrize(
    "duplicate_domain",
    ["selected_cases", "position_inventory", "probe_rows", "patch_rows"],
)
def test_merge_lane_d_shards_rejects_duplicate_row_keys_for_all_domains(
    tmp_path: Path,
    duplicate_domain: str,
) -> None:
    root = tmp_path / "hidden_state_probe"
    _write_lane_d_shards(root, num_shards=2, duplicate_domain=duplicate_domain)

    with pytest.raises(
        ValueError,
        match=f"duplicate Lane D row key.*{duplicate_domain}",
    ):
        merge_lane_d_shards(root, expected_shards=2)


def test_merge_lane_d_shards_validates_merged_position_inventory(tmp_path: Path) -> None:
    root = tmp_path / "hidden_state_probe"
    _write_lane_d_shards(root, num_shards=2, invalid_inventory=True)

    with pytest.raises(ValueError, match="missing compact roles"):
        merge_lane_d_shards(root, expected_shards=2)


@pytest.mark.parametrize(
    ("filename", "updates", "deleted_key", "expected_error"),
    [
        ("selected_cases.jsonl", {"source_line_idx": True}, None, "source_line_idx"),
        ("selected_cases.jsonl", {"case_id": ""}, None, "case_id"),
        ("selected_cases.jsonl", {}, "case_id", "missing case_id"),
        ("probe_rows.jsonl", {"prefix_depth": None}, None, "prefix_depth"),
        ("probe_rows.jsonl", {"model_layer": 0.5}, None, "model_layer"),
        ("probe_rows.jsonl", {"task": ""}, None, "task"),
        (
            "probe_rows.jsonl",
            {},
            "layer_group",
            "missing layer_group",
        ),
        ("patch_rows.jsonl", {"donor_policy": False}, None, "donor_policy"),
        ("patch_rows.jsonl", {"donor_case_id": ""}, None, "donor_case_id"),
        ("patch_rows.jsonl", {}, "slot", "missing slot"),
    ],
)
def test_merge_lane_d_shards_rejects_invalid_typed_duplicate_key_fields(
    tmp_path: Path,
    filename: str,
    updates: dict[str, object],
    deleted_key: str | None,
    expected_error: str,
) -> None:
    root = tmp_path / "hidden_state_probe"
    _write_lane_d_shards(root, num_shards=2)
    target_path = _lane_d_shard_file(root, 1, 2, filename)
    if deleted_key is not None:
        _delete_first_jsonl_row_key(target_path, deleted_key)
    else:
        _mutate_first_jsonl_row(target_path, **updates)

    with pytest.raises(ValueError, match=expected_error):
        merge_lane_d_shards(root, expected_shards=2)


def test_merge_lane_d_shards_allows_null_donor_case_id_only(tmp_path: Path) -> None:
    root = tmp_path / "hidden_state_probe"
    _write_lane_d_shards(root, num_shards=2)
    _mutate_first_jsonl_row(
        _lane_d_shard_file(root, 1, 2, "patch_rows.jsonl"),
        donor_case_id=None,
    )

    summary = merge_lane_d_shards(root, expected_shards=2)

    assert summary["row_counts"]["patch_rows"] == 2


def test_merge_lane_d_shards_rejects_non_inventory_row_in_wrong_shard(
    tmp_path: Path,
) -> None:
    root = tmp_path / "hidden_state_probe"
    _write_lane_d_shards(root, num_shards=2)
    _mutate_first_jsonl_row(
        _lane_d_shard_file(root, 1, 2, "probe_rows.jsonl"),
        source_line_idx=2,
    )

    with pytest.raises(ValueError, match="source_line_idx.*shard_001-of-002"):
        merge_lane_d_shards(root, expected_shards=2)


def test_merge_lane_d_shards_rejects_inventory_shard_field_mismatch(
    tmp_path: Path,
) -> None:
    root = tmp_path / "hidden_state_probe"
    _write_lane_d_shards(root, num_shards=2)
    _mutate_first_jsonl_row(
        _lane_d_shard_file(root, 1, 2, "position_inventory.jsonl"),
        shard_index=0,
        shard_label="shard_000-of-002",
    )

    with pytest.raises(ValueError, match="shard_index.*shard_001-of-002"):
        merge_lane_d_shards(root, expected_shards=2)


def test_merge_lane_d_summary_has_required_contract_fields(tmp_path: Path) -> None:
    root = tmp_path / "hidden_state_probe"
    _write_lane_d_shards(root, num_shards=2)

    summary = merge_lane_d_shards(root, expected_shards=2)

    assert set(summary) >= {
        "stage",
        "expected_shards",
        "merged_shards",
        "row_counts",
        "source_summaries",
        "output_paths",
        "duplicate_key_domains",
        "compact_roles",
        "layer_groups",
        "config_path",
        "config_sha256",
        "checkpoint",
        "artifact_root",
        "batch_size",
        "selected_cases_sha256",
        "sha256_by_domain",
        "source_manifest_sha256",
        "source_summary_sha256_by_shard",
    }
    assert summary["duplicate_key_domains"]["probe_rows"] == [
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
    ]
    assert summary["duplicate_key_domains"]["position_inventory"] == [
        "source_line_idx",
        "case_id",
        "prefix_mode",
        "prefix_depth",
        "render_source",
        "role",
    ]
    assert summary["compact_roles"] == list(LANE_D_COMPACT_ROLES)
    assert summary["layer_groups"] == {"early": [0], "last": [-1]}
    assert (
        summary["config_path"]
        == "configs/analysis/autoreg_hidden_state_probe/test.yaml"
    )
    assert summary["checkpoint"] == "checkpoint-test"
    assert set(summary["sha256_by_domain"]) == {
        "selected_cases",
        "position_inventory",
        "probe_rows",
        "patch_rows",
    }
    assert (
        summary["selected_cases_sha256"]
        == summary["sha256_by_domain"]["selected_cases"]
    )
    assert summary["source_manifest_sha256"]
    assert set(summary["source_summary_sha256_by_shard"]) == {
        "shard_000-of-002",
        "shard_001-of-002",
    }


def test_merge_lane_d_hashes_are_domain_specific(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_lane_d_shards(left, num_shards=2)
    _write_lane_d_shards(right, num_shards=2)
    _mutate_first_jsonl_row(
        _lane_d_shard_file(right, 1, 2, "probe_rows.jsonl"),
        target_margin=999.0,
    )

    left_summary = merge_lane_d_shards(left, expected_shards=2)
    right_summary = merge_lane_d_shards(right, expected_shards=2)

    assert (
        left_summary["sha256_by_domain"]["selected_cases"]
        == right_summary["sha256_by_domain"]["selected_cases"]
    )
    assert (
        left_summary["sha256_by_domain"]["probe_rows"]
        != right_summary["sha256_by_domain"]["probe_rows"]
    )
    assert (
        left_summary["selected_cases_sha256"]
        == right_summary["selected_cases_sha256"]
    )


def test_merge_lane_d_write_failure_leaves_no_final_root_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "hidden_state_probe"
    _write_lane_d_shards(root, num_shards=2)

    def raise_on_write(path: Path, data: bytes) -> None:
        raise OSError(f"injected write failure for {path.name}")

    monkeypatch.setattr(lane_d_probe, "_atomic_write_bytes", raise_on_write)

    with pytest.raises(OSError, match="injected write failure"):
        merge_lane_d_shards(root, expected_shards=2)

    assert not any(path.exists() for path in _root_merge_outputs(root))


def test_merge_lane_d_publish_failure_rolls_back_final_root_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "hidden_state_probe"
    _write_lane_d_shards(root, num_shards=2)
    real_publish = lane_d_probe._publish_lane_d_staged_output
    publish_count = 0

    def fail_after_first_publish(staged_path: Path, output_path: Path) -> None:
        nonlocal publish_count
        publish_count += 1
        if publish_count > 1:
            raise OSError(f"injected publish failure for {output_path.name}")
        real_publish(staged_path, output_path)

    monkeypatch.setattr(
        lane_d_probe,
        "_publish_lane_d_staged_output",
        fail_after_first_publish,
    )

    with pytest.raises(OSError, match="injected publish failure"):
        merge_lane_d_shards(root, expected_shards=2)

    assert publish_count == 2
    assert not any(path.exists() for path in _root_merge_outputs(root))


def test_cli_merge_shards_runs_cpu_merge_and_prints_json(tmp_path: Path) -> None:
    config_path = tmp_path / "lane_d.yaml"
    root = tmp_path / "artifact_root"
    checkpoint = tmp_path / "checkpoint"
    layer_groups = {"early": [0, 1], "last": [-4, -3]}
    _write_lane_d_config(
        config_path,
        artifact_root=root,
        checkpoint=checkpoint,
    )
    _write_lane_d_shards(
        root,
        num_shards=2,
        config_path=str(config_path),
        config_sha256=lane_d_probe._sha256_path(config_path),
        checkpoint=str(checkpoint),
        artifact_root=str(root),
        layer_groups=layer_groups,
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/analysis/run_autoreg_hidden_state_probe.py",
            "--config",
            str(config_path),
            "--stages",
            "merge",
            "--merge-shards",
            "--num-shards",
            "2",
        ],
        check=False,
        cwd="/data/CoordExp",
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    summary = json.loads(result.stdout)
    assert summary["stage"] == "hidden_state_probe"
    assert summary["row_counts"]["probe_rows"] == 2
    assert (root / "merge_summary.json").exists()
    assert "Traceback" not in result.stderr
    assert "not implemented" not in result.stderr


def test_cli_merge_shards_rejects_config_metadata_mismatch_without_traceback(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "lane_d.yaml"
    root = tmp_path / "artifact_root"
    _write_lane_d_config(
        config_path,
        artifact_root=root,
        checkpoint=tmp_path / "different-checkpoint",
    )
    _write_lane_d_shards(root, num_shards=2)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/analysis/run_autoreg_hidden_state_probe.py",
            "--config",
            str(config_path),
            "--stages",
            "merge",
            "--merge-shards",
            "--num-shards",
            "2",
        ],
        check=False,
        cwd="/data/CoordExp",
        capture_output=True,
        text=True,
    )
    combined_output = result.stdout + result.stderr

    assert result.returncode != 0
    assert "checkpoint" in combined_output or "config_path" in combined_output
    assert "Traceback" not in combined_output


def test_cli_merge_shards_requires_num_shards_without_traceback(tmp_path: Path) -> None:
    config_path = tmp_path / "lane_d.yaml"
    _write_lane_d_config(config_path)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/analysis/run_autoreg_hidden_state_probe.py",
            "--config",
            str(config_path),
            "--stages",
            "merge",
            "--merge-shards",
        ],
        check=False,
        cwd="/data/CoordExp",
        capture_output=True,
        text=True,
    )
    combined_output = result.stdout + result.stderr

    assert result.returncode != 0
    assert "--merge-shards requires --num-shards" in combined_output
    assert "Traceback" not in combined_output


def test_cli_materializes_select_cases_shard_without_traceback(tmp_path: Path) -> None:
    config_path = tmp_path / "lane_d.yaml"
    _write_lane_d_config(config_path, max_cases=10)
    _write_jsonl(
        tmp_path / "per_case.jsonl",
        [
            _lane_c_case("case-0", source_line_idx=0),
            _lane_c_case("case-1", source_line_idx=1),
            _lane_c_case("case-2", source_line_idx=2),
        ],
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/analysis/run_autoreg_hidden_state_probe.py",
            "--config",
            str(config_path),
            "--stages",
            "select_cases",
            "--shard-index",
            "0",
            "--num-shards",
            "2",
        ],
        check=False,
        cwd="/data/CoordExp",
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["manifest"]["expected_shards"] == 2
    assert payload["shard_summary"]["shard_label"] == lane_d_shard_label(0, 2)
    assert payload["shard_summary"]["row_counts"]["selected_cases"] == 2
    assert (tmp_path / "artifact_root" / "shards_manifest.json").exists()
    assert (
        tmp_path
        / "artifact_root"
        / "shards"
        / lane_d_shard_label(0, 2)
        / "selected_cases.jsonl"
    ).exists()
    assert "Traceback" not in result.stderr


def test_cli_empty_stages_exits_without_traceback_before_config_load(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/analysis/run_autoreg_hidden_state_probe.py",
            "--config",
            str(tmp_path / "missing.yaml"),
            "--stages",
            "",
            "--dry-run",
        ],
        check=False,
        cwd="/data/CoordExp",
        capture_output=True,
        text=True,
    )
    combined_output = result.stdout + result.stderr

    assert result.returncode != 0
    assert "stages" in combined_output
    assert "Traceback" not in combined_output
    assert "No such file" not in combined_output


def test_cli_invalid_shard_args_exit_without_traceback(tmp_path: Path) -> None:
    config_path = tmp_path / "lane_d.yaml"
    _write_lane_d_config(config_path)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/analysis/run_autoreg_hidden_state_probe.py",
            "--config",
            str(config_path),
            "--stages",
            "select_cases",
            "--dry-run",
            "--shard-index",
            "8",
            "--num-shards",
            "8",
        ],
        check=False,
        cwd="/data/CoordExp",
        capture_output=True,
        text=True,
    )
    combined_output = result.stdout + result.stderr

    assert result.returncode != 0
    assert "shard_index" in combined_output
    assert "Traceback" not in combined_output


@pytest.mark.parametrize(
    "extra_args",
    (
        (),
        ("--num-shards", "2"),
        ("--shard-index", "0"),
    ),
)
def test_cli_non_dry_run_select_cases_requires_complete_shard_args(
    tmp_path: Path,
    extra_args: tuple[str, ...],
) -> None:
    config_path = tmp_path / "lane_d.yaml"
    _write_lane_d_config(config_path)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/analysis/run_autoreg_hidden_state_probe.py",
            "--config",
            str(config_path),
            "--stages",
            "select_cases",
            *extra_args,
        ],
        check=False,
        cwd="/data/CoordExp",
        capture_output=True,
        text=True,
    )
    combined_output = result.stdout + result.stderr

    assert result.returncode != 0
    assert "--shard-index and --num-shards" in combined_output
    assert "Traceback" not in combined_output


def test_cli_hidden_states_without_selected_cases_fails_cleanly(tmp_path: Path) -> None:
    config_path = tmp_path / "lane_d.yaml"
    _write_lane_d_config(config_path)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/analysis/run_autoreg_hidden_state_probe.py",
            "--config",
            str(config_path),
            "--stages",
            "hidden_states",
            "--shard-index",
            "0",
            "--num-shards",
            "2",
        ],
        check=False,
        cwd="/data/CoordExp",
        capture_output=True,
        text=True,
    )
    combined_output = result.stdout + result.stderr

    assert result.returncode != 0
    assert "selected_cases.jsonl" in combined_output
    assert "not implemented" not in combined_output
    assert "Traceback" not in combined_output


def test_materialize_hidden_states_shard_uses_sharded_lane_c_prepare_and_writes_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch
    import src.analysis.hard_ce_coord_logit_locality as lane_c_module

    config_path = tmp_path / "lane_d.yaml"
    root = tmp_path / "artifact_root"
    _write_lane_d_config(
        config_path,
        artifact_root=root,
        layer_groups_yaml="    first: [0]",
    )
    config = load_lane_d_config(config_path)
    tokenizer = _FakeLaneDTokenizer()
    case_id = "row1:self_prefix:depth0:gt0"
    example = _fake_prepared_lane_c_example(
        tokenizer,
        assistant_text=_compact_row("target", (5, 6, 7, 8)),
        case_id=case_id,
        prefix_mode="self_prefix",
        prefix_depth=0,
        source_line_idx=1,
    )
    selected_case = {
        "case_id": case_id,
        "source_line_idx": 1,
        "prefix_mode": "self_prefix",
        "prefix_depth": 0,
        "prefix_quality": "empty_prefix",
        "intended_target_gt_idx": 0,
        "target_desc": "target",
        "x1_top_peak_attribution": "target_gt_object",
        "x1_target_rank": 1,
        "selection_reason": "lane_c_case",
        "shard_index": 1,
        "num_shards": 2,
        "shard_label": lane_d_shard_label(1, 2),
    }
    shard_dir = root / "shards" / lane_d_shard_label(1, 2)
    _write_jsonl(shard_dir / "selected_cases.jsonl", [selected_case])
    prepare_calls: list[dict[str, object]] = []

    def fake_prepare(
        lane_c_config: object,
        *,
        model_handle: object,
        limit: object,
        shard_index: object,
        num_shards: object,
    ) -> tuple[list[object], dict[str, object]]:
        del lane_c_config, model_handle, limit
        prepare_calls.append({"shard_index": shard_index, "num_shards": num_shards})
        return [example], {"selected_record_count": 1, "planned_example_count": 1}

    model_handle = SimpleNamespace(
        tokenizer=tokenizer,
        processor=_FakeLaneDProcessor({example.full_text: example.full_input_ids}),
        model=_FakeLaneDModel(),
    )
    monkeypatch.setattr(lane_c_module, "load_study_config", lambda path: object())
    monkeypatch.setattr(lane_c_module, "load_model_handle", lambda cfg: model_handle)
    monkeypatch.setattr(lane_c_module, "prepare_lane_c_x1_basin_examples", fake_prepare)
    monkeypatch.setattr(lane_c_module, "_load_image", lambda path: object())
    monkeypatch.setattr(lane_c_module, "_model_device", lambda model: torch.device("cpu"))

    summary = lane_d_probe.materialize_lane_d_hidden_states_shard(
        config,
        shard_index=1,
        num_shards=2,
    )

    assert prepare_calls == [{"shard_index": 1, "num_shards": 2}]
    assert summary["row_counts"] == {
        "selected_cases": 1,
        "position_inventory": len(LANE_D_COMPACT_ROLES),
        "probe_rows": len(LANE_D_COMPACT_ROLES),
        "patch_rows": 0,
    }
    assert summary["runtime_kind"] == "hidden_states_model_forward"
    assert summary["batch_size"] == 1
    assert summary["config_sha256"] == lane_d_probe._sha256_path(config_path)
    assert summary["artifact_root"] == str(root)
    position_rows = _read_jsonl(shard_dir / "position_inventory.jsonl")
    probe_rows = _read_jsonl(shard_dir / "probe_rows.jsonl")
    assert len(position_rows) == len(LANE_D_COMPACT_ROLES)
    assert len(probe_rows) == len(LANE_D_COMPACT_ROLES)
    first_probe = probe_rows[0]
    assert first_probe["source_line_idx"] == 1
    assert first_probe["shard_index"] == 1
    assert first_probe["num_shards"] == 2
    assert first_probe["shard_label"] == lane_d_shard_label(1, 2)
    assert first_probe["prefix_condition"] == "self_prefix"
    assert first_probe["layer"] == 0
    assert first_probe["model_layer"] == 0
    assert first_probe["object_count"] == 2
    assert first_probe["remaining_count"] == 1
    assert first_probe["intended_target_gt_idx"] == 0
    assert first_probe["x1_top_peak_attribution"] == "target_gt_object"
    assert first_probe["x1_target_rank"] == 1
    assert _read_jsonl(shard_dir / "patch_rows.jsonl") == []


def test_autoreg_hidden_state_probe_import_has_no_package_side_effect_logs() -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = "/data/CoordExp"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import src.analysis.autoreg_hidden_state_probe as m; print(m.__name__)",
        ],
        check=False,
        cwd="/data/CoordExp",
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert result.stdout == "src.analysis.autoreg_hidden_state_probe\n"
    assert "[INFO:swift]" not in result.stderr
    assert "Successfully registered" not in result.stderr
