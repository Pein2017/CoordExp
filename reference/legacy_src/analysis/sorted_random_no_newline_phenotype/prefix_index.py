from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.analysis.prefix_state_transition_tomography.prefix_rendering import (
    render_teacher_prefix,
)

from . import PHASE_ID, PROJECT_ID, RUN_ID, SCHEMA_VERSION
from .config import A32Config


EVIDENCE_SCOPE = "a3_2_prefix4096_hardbiased_len12000_canonical_sorted"
PREFIX_SOURCE_POLICY = "canonical_sorted_teacher_prefix_readout"
PREFIX_ORDER_POLICY_ID = "canonical_sorted_yx_teacher_v1"
READOUT_PROMPT_ORDERING = "sorted"
TEMPLATE_CONTRACT = {"row_separator": "none"}

_WS_RE = re.compile(r"\s+")


def canonical_desc(value: str) -> str:
    return _WS_RE.sub(" ", value.strip().lower())


def canonical_sorted_objects(
    objects: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        dict(obj)
        for _, obj in sorted(
            enumerate(objects),
            key=lambda indexed: (
                int(indexed[1]["bbox_xyxy"][1]),
                int(indexed[1]["bbox_xyxy"][0]),
                _original_index(indexed[1], indexed[0]),
            ),
        )
    ]


def build_prefix_state_index(
    config: A32Config,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    light_rows, parse_stats = _collect_lightweight_rows(config)
    sampled_light_rows, underfill_reasons = _sample_prefix_states(
        light_rows,
        max_prefix_states=config.sampling.max_prefix_states,
        seed=config.sampling.seed,
        easy_sanity_max_fraction=config.sampling.easy_sanity_max_fraction,
    )
    _assert_unique_prefix_state_ids(sampled_light_rows)

    rows, selected_light_rows = _attach_sampling_fields(
        light_rows,
        sampled_light_rows,
        num_shards=config.sampling.num_shards,
    )
    sampled_rows = _materialize_sampled_rows(config, selected_light_rows)
    sample_manifest = _build_sample_manifest(
        rows=rows,
        sampled_rows=sampled_rows,
        config=config,
        underfill_reasons=underfill_reasons,
        parse_stats=parse_stats,
    )
    summary = _build_summary(
        rows=rows,
        sampled_rows=sampled_rows,
        config=config,
        sample_manifest=sample_manifest,
    )
    return rows, sampled_rows, summary, sample_manifest


def _collect_lightweight_rows(
    config: A32Config,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    parse_stats = _empty_parse_stats()
    rows: list[dict[str, Any]] = []
    for split, path in (("train", config.train_jsonl), ("val", config.val_jsonl)):
        for source_line_idx, record in _iter_jsonl_records(path):
            objects, record_stats = _parse_objects(record)
            _merge_parse_stats(parse_stats, record_stats)
            rows.extend(
                _lightweight_rows_for_record(
                    config=config,
                    record=record,
                    split=split,
                    source_dataset_jsonl=path,
                    source_line_idx=source_line_idx,
                    objects=objects,
                )
            )
    return rows, parse_stats


def _lightweight_rows_for_record(
    *,
    config: A32Config,
    record: Mapping[str, Any],
    split: str,
    source_dataset_jsonl: Path,
    source_line_idx: int,
    objects: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not objects:
        return []

    canonical_objects = canonical_sorted_objects(objects)
    canonical_sorted_gt_indices = [int(obj["gt_idx"]) for obj in canonical_objects]
    desc_counter = Counter(str(obj["desc"]) for obj in objects)
    object_count = len(objects)
    desc_count = len(desc_counter)
    max_same_desc_count = max(desc_counter.values(), default=0)
    candidate_descs = sorted(desc_counter)

    rows: list[dict[str, Any]] = []
    for prefix_len in range(object_count):
        emitted_gt_indices = canonical_sorted_gt_indices[:prefix_len]
        residual_gt_indices = canonical_sorted_gt_indices[prefix_len:]
        row = {
            "project_id": PROJECT_ID,
            "phase_id": PHASE_ID,
            "schema_version": SCHEMA_VERSION,
            "run_id": RUN_ID,
            "evidence_scope": EVIDENCE_SCOPE,
            "prefix_source_policy": PREFIX_SOURCE_POLICY,
            "prefix_order_policy_id": PREFIX_ORDER_POLICY_ID,
            "readout_prompt_ordering": READOUT_PROMPT_ORDERING,
            "template_contract": {
                "row_separator": config.template_contract.row_separator,
            },
            "split": split,
            "source_dataset_jsonl": str(source_dataset_jsonl),
            "source_line_idx": source_line_idx,
            "image_id": record.get("image_id"),
            "image_path": _extract_image_ref(record),
            "prefix_state_id": "",
            "prefix_len": prefix_len,
            "prefix_depth_bucket": _prefix_depth_bucket(prefix_len, object_count),
            "canonical_sorted_gt_indices": list(canonical_sorted_gt_indices),
            "emitted_gt_indices": list(emitted_gt_indices),
            "residual_gt_indices": list(residual_gt_indices),
            "candidate_descs": list(candidate_descs),
            "object_count": object_count,
            "desc_count": desc_count,
            "max_same_desc_count": max_same_desc_count,
            "same_desc_multi_instance": max_same_desc_count >= 2,
            "same_desc_count_bucket": _same_desc_count_bucket(max_same_desc_count),
            "object_count_bucket": _object_count_bucket(object_count),
            "desc_count_bucket": _desc_count_bucket(desc_count),
            "hardness": _hardness(
                object_count=object_count,
                desc_count=desc_count,
                max_same_desc_count=max_same_desc_count,
            ),
            "selected_for_probe": False,
            "planned_shard_id": None,
            "shard_id": None,
        }
        row["prefix_state_id"] = _prefix_state_id(row)
        rows.append(row)
    return rows


def _parse_objects(
    record: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    stats = _empty_parse_stats()
    raw_objects = record.get("objects")
    if not isinstance(raw_objects, list):
        return [], stats

    objects: list[dict[str, Any]] = []
    for gt_idx, raw_object in enumerate(raw_objects):
        if not isinstance(raw_object, Mapping):
            stats["skipped_object_count"] += 1
            stats["non_mapping_object_count"] += 1
            continue

        desc = _object_desc(raw_object)
        if not desc:
            stats["skipped_object_count"] += 1
            stats["missing_desc_object_count"] += 1
            continue

        bbox = _parse_bbox_2d(raw_object.get("bbox_2d"))
        if bbox is None:
            stats["skipped_object_count"] += 1
            stats["malformed_bbox_object_count"] += 1
            continue

        objects.append(
            {
                "gt_idx": gt_idx,
                "original_index": gt_idx,
                "desc": desc,
                "bbox_xyxy": bbox,
            }
        )
    return objects, stats


def _sample_prefix_states(
    rows: Sequence[dict[str, Any]],
    *,
    max_prefix_states: int,
    seed: int,
    easy_sanity_max_fraction: float,
) -> tuple[list[dict[str, Any]], list[str]]:
    if max_prefix_states <= 0 or not rows:
        reason = (
            "max_prefix_states_lte_zero"
            if max_prefix_states <= 0
            else "no_rows_available"
        )
        return [], [reason]

    rng = random.Random(seed)
    hard_rows = [row for row in rows if row["hardness"] == "headline_hard"]
    easy_rows = [row for row in rows if row["hardness"] == "easy_sanity"]
    hard_sample = _stratified_sample(hard_rows, max_prefix_states, rng)
    easy_budget = _easy_budget(
        hard_count=len(hard_sample),
        remaining=max(0, max_prefix_states - len(hard_sample)),
        available_easy=len(easy_rows),
        easy_sanity_max_fraction=easy_sanity_max_fraction,
    )
    easy_sample = _stratified_sample(easy_rows, easy_budget, rng)
    sampled = hard_sample + easy_sample
    rng.shuffle(sampled)

    underfill_reasons: list[str] = []
    if not hard_rows:
        underfill_reasons.append("no_headline_hard_rows_available")
    if len(rows) < max_prefix_states:
        underfill_reasons.append("available_rows_below_max_prefix_states")
    if len(sampled) < min(len(rows), max_prefix_states):
        underfill_reasons.append("hard_bias_or_easy_sanity_cap_underfilled_sample")
    return sampled, underfill_reasons


def _attach_sampling_fields(
    rows: Sequence[dict[str, Any]],
    sampled_rows: Sequence[dict[str, Any]],
    *,
    num_shards: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected_order = {
        str(row["prefix_state_id"]): index for index, row in enumerate(sampled_rows)
    }
    enriched_rows: list[dict[str, Any]] = []
    selected_by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        enriched = dict(row)
        selected_index = selected_order.get(str(row["prefix_state_id"]))
        if selected_index is not None:
            shard_id = selected_index % num_shards
            enriched["selected_for_probe"] = True
            enriched["planned_shard_id"] = shard_id
            enriched["shard_id"] = shard_id
            selected_by_id[str(enriched["prefix_state_id"])] = enriched
        enriched_rows.append(enriched)

    selected_light_rows = [
        selected_by_id[str(row["prefix_state_id"])] for row in sampled_rows
    ]
    return enriched_rows, selected_light_rows


def _materialize_sampled_rows(
    config: A32Config,
    selected_light_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not selected_light_rows:
        return []

    selected_by_locator: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in selected_light_rows:
        selected_by_locator[(str(row["split"]), int(row["source_line_idx"]))].append(row)

    materialized_by_id: dict[str, dict[str, Any]] = {}
    for split, path in (("train", config.train_jsonl), ("val", config.val_jsonl)):
        needed_line_indices = {
            line_idx
            for selected_split, line_idx in selected_by_locator
            if selected_split == split
        }
        if not needed_line_indices:
            continue
        for source_line_idx, record in _iter_jsonl_records(path):
            if source_line_idx not in needed_line_indices:
                continue
            objects, _ = _parse_objects(record)
            canonical_objects = canonical_sorted_objects(objects)
            for light_row in selected_by_locator[(split, source_line_idx)]:
                heavy_row = _materialize_sampled_row(light_row, objects, canonical_objects)
                materialized_by_id[str(heavy_row["prefix_state_id"])] = heavy_row

    sampled_rows: list[dict[str, Any]] = []
    for light_row in selected_light_rows:
        prefix_state_id = str(light_row["prefix_state_id"])
        heavy_row = materialized_by_id.get(prefix_state_id)
        if heavy_row is None:
            raise ValueError(f"selected prefix row could not be materialized: {prefix_state_id}")
        sampled_rows.append(heavy_row)
    return sampled_rows


def _materialize_sampled_row(
    light_row: Mapping[str, Any],
    objects: Sequence[Mapping[str, Any]],
    canonical_objects: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    emitted_indices = set(int(idx) for idx in light_row["emitted_gt_indices"])
    residual_indices = set(int(idx) for idx in light_row["residual_gt_indices"])
    emitted_objects = [
        obj for obj in canonical_objects if int(obj["gt_idx"]) in emitted_indices
    ]
    residual_objects = [
        obj for obj in canonical_objects if int(obj["gt_idx"]) in residual_indices
    ]

    rendered_prefix = render_teacher_prefix(emitted_objects)
    if "\n" in rendered_prefix:
        raise ValueError(f"rendered prefix contains newline: {light_row['prefix_state_id']}")

    heavy_row = dict(light_row)
    heavy_row["gt_objects"] = [_public_object(obj) for obj in objects]
    heavy_row["candidate_descs_with_roles"] = _candidate_descs_with_roles(
        emitted_objects,
        residual_objects,
    )
    heavy_row["rendered_prefix_sha256"] = hashlib.sha256(
        rendered_prefix.encode("utf-8")
    ).hexdigest()
    heavy_row["rendered_prefix_char_len"] = len(rendered_prefix)
    return heavy_row


def _build_sample_manifest(
    *,
    rows: Sequence[Mapping[str, Any]],
    sampled_rows: Sequence[Mapping[str, Any]],
    config: A32Config,
    underfill_reasons: Sequence[str],
    parse_stats: Mapping[str, int],
) -> dict[str, Any]:
    easy_count = sum(row["hardness"] == "easy_sanity" for row in sampled_rows)
    return {
        "project_id": PROJECT_ID,
        "phase_id": PHASE_ID,
        "schema_version": SCHEMA_VERSION,
        "run_id": RUN_ID,
        "evidence_scope": EVIDENCE_SCOPE,
        "prefix_order_policy_id": PREFIX_ORDER_POLICY_ID,
        "sampling_seed": config.sampling.seed,
        "max_prefix_states": config.sampling.max_prefix_states,
        "num_shards": config.sampling.num_shards,
        "available_rows": len(rows),
        "selected_rows": len(sampled_rows),
        "available_by_split": _count_by(rows, "split"),
        "selected_by_split": _count_by(sampled_rows, "split"),
        "available_by_same_desc_count_bucket": _count_by(
            rows,
            "same_desc_count_bucket",
        ),
        "selected_by_same_desc_count_bucket": _count_by(
            sampled_rows,
            "same_desc_count_bucket",
        ),
        "available_by_object_count_bucket": _count_by(rows, "object_count_bucket"),
        "selected_by_object_count_bucket": _count_by(
            sampled_rows,
            "object_count_bucket",
        ),
        "available_by_desc_count_bucket": _count_by(rows, "desc_count_bucket"),
        "selected_by_desc_count_bucket": _count_by(sampled_rows, "desc_count_bucket"),
        "easy_sanity_selected_count": easy_count,
        "easy_sanity_selected_fraction": (
            0.0 if not sampled_rows else easy_count / len(sampled_rows)
        ),
        "easy_sanity_max_fraction": config.sampling.easy_sanity_max_fraction,
        "underfill_reasons": list(dict.fromkeys(underfill_reasons)),
        "skipped_object_count": int(parse_stats["skipped_object_count"]),
        "non_mapping_object_count": int(parse_stats["non_mapping_object_count"]),
        "malformed_bbox_object_count": int(parse_stats["malformed_bbox_object_count"]),
        "missing_desc_object_count": int(parse_stats["missing_desc_object_count"]),
    }


def _build_summary(
    *,
    rows: Sequence[Mapping[str, Any]],
    sampled_rows: Sequence[Mapping[str, Any]],
    config: A32Config,
    sample_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    failed_launch_gates = _failed_launch_gates(
        sampled_rows,
        easy_sanity_max_fraction=config.sampling.easy_sanity_max_fraction,
    )
    return {
        "project_id": PROJECT_ID,
        "phase_id": PHASE_ID,
        "schema_version": SCHEMA_VERSION,
        "run_id": RUN_ID,
        "launch_eligible": not failed_launch_gates,
        "failed_launch_gates": failed_launch_gates,
        "total_rows": len(rows),
        "sampled_rows": len(sampled_rows),
        "num_shards": config.sampling.num_shards,
        "evidence_scope": EVIDENCE_SCOPE,
        "template_contract": {
            "row_separator": config.template_contract.row_separator,
        },
        "prefix_source_policy": PREFIX_SOURCE_POLICY,
        "prefix_order_policy_id": PREFIX_ORDER_POLICY_ID,
        "readout_prompt_ordering": READOUT_PROMPT_ORDERING,
        "sample_manifest": {
            "easy_sanity_selected_fraction": sample_manifest[
                "easy_sanity_selected_fraction"
            ],
            "underfill_reasons": list(sample_manifest["underfill_reasons"]),
        },
    }


def _failed_launch_gates(
    sampled_rows: Sequence[Mapping[str, Any]],
    *,
    easy_sanity_max_fraction: float,
) -> list[str]:
    failures: list[str] = []
    if not sampled_rows:
        failures.append("no_sampled_rows")
    sampled_splits = {str(row["split"]) for row in sampled_rows}
    for split in ("train", "val"):
        if split not in sampled_splits:
            failures.append(f"missing_{split}_coverage")
    easy_count = sum(row["hardness"] == "easy_sanity" for row in sampled_rows)
    easy_fraction = 0.0 if not sampled_rows else easy_count / len(sampled_rows)
    if easy_fraction > easy_sanity_max_fraction:
        failures.append("easy_sanity_fraction_gt20pct")
    return failures


def _iter_jsonl_records(path: Path) -> Iterable[tuple[int, Mapping[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            if not isinstance(record, Mapping):
                raise ValueError(f"{path}:{line_index + 1} JSONL row must be an object")
            yield line_index, record


def _object_desc(raw_object: Mapping[str, Any]) -> str | None:
    for key in ("desc", "category_name"):
        value = raw_object.get(key)
        if isinstance(value, str):
            desc = canonical_desc(value)
            if desc:
                return desc
    return None


def _parse_bbox_2d(raw_bbox: Any) -> list[int] | None:
    if not isinstance(raw_bbox, Sequence) or isinstance(raw_bbox, str):
        return None
    if len(raw_bbox) != 4:
        return None

    values: list[int] = []
    for raw_token in raw_bbox:
        if not isinstance(raw_token, str):
            return None
        if not raw_token.startswith("<|coord_") or not raw_token.endswith("|>"):
            return None
        try:
            value = int(raw_token.removeprefix("<|coord_").removesuffix("|>"))
        except ValueError:
            return None
        if value < 0 or value > 999:
            return None
        values.append(value)
    return values


def _candidate_descs_with_roles(
    emitted_objects: Sequence[Mapping[str, Any]],
    residual_objects: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    emitted_by_desc = _indices_by_desc(emitted_objects)
    residual_by_desc = _indices_by_desc(residual_objects)
    candidates: list[dict[str, Any]] = []
    for desc in sorted(set(emitted_by_desc) | set(residual_by_desc)):
        emitted_indices = emitted_by_desc.get(desc, [])
        residual_indices = residual_by_desc.get(desc, [])
        roles: list[str] = []
        if emitted_indices and residual_indices:
            roles.extend(["emitted_same_desc", "residual_same_desc"])
        elif emitted_indices:
            roles.append("emitted_other_desc")
        elif residual_indices:
            roles.append("residual_other_desc")
        candidates.append(
            {
                "desc": desc,
                "roles": roles,
                "gt_indices": sorted(emitted_indices + residual_indices),
                "emitted_gt_indices": emitted_indices,
                "residual_gt_indices": residual_indices,
            }
        )
    return candidates


def _indices_by_desc(
    objects: Sequence[Mapping[str, Any]],
) -> dict[str, list[int]]:
    indices: dict[str, list[int]] = defaultdict(list)
    for obj in objects:
        indices[str(obj["desc"])].append(int(obj["gt_idx"]))
    return {desc: sorted(values) for desc, values in indices.items()}


def _extract_image_ref(record: Mapping[str, Any]) -> str | None:
    file_name = record.get("file_name")
    if isinstance(file_name, str) and file_name:
        return file_name

    images = record.get("images")
    if not isinstance(images, list):
        return None
    for image in images:
        if isinstance(image, str) and image:
            return image
        if isinstance(image, Mapping):
            nested_file_name = image.get("file_name")
            if isinstance(nested_file_name, str) and nested_file_name:
                return nested_file_name
    return None


def _public_object(obj: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "gt_idx": int(obj["gt_idx"]),
        "desc": str(obj["desc"]),
        "bbox_xyxy": [int(value) for value in obj["bbox_xyxy"]],
    }


def _original_index(obj: Mapping[str, Any], fallback: int) -> int:
    value = obj.get("original_index", obj.get("gt_idx", fallback))
    return int(value)


def _prefix_state_id(row: Mapping[str, Any]) -> str:
    payload = {
        "split": row["split"],
        "source_line_idx": row["source_line_idx"],
        "image_id": row["image_id"],
        "canonical_sorted_gt_indices": row["canonical_sorted_gt_indices"],
        "emitted_gt_indices": row["emitted_gt_indices"],
        "residual_gt_indices": row["residual_gt_indices"],
    }
    digest = hashlib.sha1(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return f"a32-prefix-{digest}"


def _assert_unique_prefix_state_ids(rows: Sequence[Mapping[str, Any]]) -> None:
    prefix_state_ids = [str(row["prefix_state_id"]) for row in rows]
    if len(prefix_state_ids) != len(set(prefix_state_ids)):
        duplicates = sorted(
            prefix_state_id
            for prefix_state_id, count in Counter(prefix_state_ids).items()
            if count > 1
        )
        raise AssertionError(
            "sampled prefix_state_id values must be unique: "
            + ", ".join(duplicates)
        )


def _hardness(
    *,
    object_count: int,
    desc_count: int,
    max_same_desc_count: int,
) -> str:
    if object_count >= 6 and desc_count >= 2 and max_same_desc_count >= 2:
        return "headline_hard"
    return "easy_sanity"


def _easy_budget(
    *,
    hard_count: int,
    remaining: int,
    available_easy: int,
    easy_sanity_max_fraction: float,
) -> int:
    if hard_count <= 0 or remaining <= 0 or available_easy <= 0:
        return 0
    fraction = max(0.0, min(1.0, easy_sanity_max_fraction))
    if fraction >= 1.0:
        return min(available_easy, remaining)
    max_easy_for_fraction = int(hard_count * fraction / max(1e-12, 1.0 - fraction))
    return min(available_easy, remaining, max_easy_for_fraction)


def _stratified_sample(
    rows: Sequence[dict[str, Any]],
    limit: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    if limit <= 0 or not rows:
        return []
    buckets: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[
            (
                str(row["split"]),
                str(row["same_desc_count_bucket"]),
                str(row["object_count_bucket"]),
                str(row["desc_count_bucket"]),
            )
        ].append(row)
    for bucket_rows in buckets.values():
        bucket_rows.sort(key=lambda row: str(row["prefix_state_id"]))
        rng.shuffle(bucket_rows)

    selected: list[dict[str, Any]] = []
    keys = sorted(buckets)
    cursor = 0
    while keys and len(selected) < limit:
        key = keys[cursor % len(keys)]
        bucket = buckets[key]
        selected.append(bucket.pop())
        if not bucket:
            keys.remove(key)
            if keys:
                cursor %= len(keys)
        else:
            cursor += 1
    return selected


def _count_by(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    counter = Counter(str(row[key]) for row in rows)
    return {value: counter[value] for value in sorted(counter)}


def _empty_parse_stats() -> dict[str, int]:
    return {
        "skipped_object_count": 0,
        "non_mapping_object_count": 0,
        "malformed_bbox_object_count": 0,
        "missing_desc_object_count": 0,
    }


def _merge_parse_stats(target: dict[str, int], source: Mapping[str, int]) -> None:
    for key, value in source.items():
        target[key] = target.get(key, 0) + int(value)


def _same_desc_count_bucket(count: int) -> str:
    if count <= 1:
        return "same_desc_count1"
    if count == 2:
        return "same_desc_count2"
    if count == 3:
        return "same_desc_count3"
    return "same_desc_count4_plus"


def _object_count_bucket(count: int) -> str:
    if count <= 2:
        return "obj_1_2"
    if count <= 5:
        return "obj_3_5"
    if count <= 10:
        return "obj_6_10"
    if count <= 20:
        return "obj_11_20"
    return "obj_21_plus"


def _desc_count_bucket(count: int) -> str:
    if count <= 1:
        return "desc_count1"
    if count == 2:
        return "desc_count2"
    if count == 3:
        return "desc_count3"
    return "desc_count4_plus"


def _prefix_depth_bucket(prefix_len: int, object_count: int) -> str:
    if prefix_len == 0:
        return "empty"
    if prefix_len == 1:
        return "shallow_1"
    if prefix_len >= max(0, object_count - 1):
        return "late_one_left"
    return "mid_prefix"


__all__ = [
    "EVIDENCE_SCOPE",
    "PREFIX_ORDER_POLICY_ID",
    "PREFIX_SOURCE_POLICY",
    "READOUT_PROMPT_ORDERING",
    "TEMPLATE_CONTRACT",
    "build_prefix_state_index",
    "canonical_desc",
    "canonical_sorted_objects",
]
