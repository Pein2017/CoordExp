from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import PHASE_ID, PROJECT_ID, SCHEMA_VERSION


_WS_RE = re.compile(r"\s+")


def canonical_desc(value: str) -> str:
    return _WS_RE.sub(" ", value.strip().lower())


def build_prefix_state_index(
    *,
    train_jsonl: Path,
    val_jsonl: Path,
    run_id: str,
    max_prefix_states: int = 4096,
    num_shards: int = 8,
    seed: int = 3664,
    easy_sanity_max_fraction: float = 0.20,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    indexed: list[dict[str, Any]] = []
    for split, path in (("train", Path(train_jsonl)), ("val", Path(val_jsonl))):
        for source_line_idx, record in _iter_jsonl(path):
            indexed.extend(
                _build_record_prefix_states(
                    record=record,
                    split=split,
                    source_dataset_jsonl=path,
                    source_line_idx=source_line_idx,
                    run_id=run_id,
                )
            )
    sampled, sampling_adjustments = _sample_prefix_states(
        indexed,
        max_prefix_states=max_prefix_states,
        seed=seed,
        easy_sanity_max_fraction=easy_sanity_max_fraction,
    )
    sampled_ids = {row["prefix_state_id"] for row in sampled}
    rows = [
        {
            **row,
            "selected_for_probe": row["prefix_state_id"] in sampled_ids,
            "planned_shard_id": (
                _sample_index(sampled, row["prefix_state_id"]) % num_shards
                if row["prefix_state_id"] in sampled_ids
                else None
            ),
            "shard_id": (
                _sample_index(sampled, row["prefix_state_id"]) % num_shards
                if row["prefix_state_id"] in sampled_ids
                else None
            ),
        }
        for row in indexed
    ]
    sampled_rows = [
        {
            **row,
            "selected_for_probe": True,
            "planned_shard_id": idx % num_shards,
            "shard_id": idx % num_shards,
        }
        for idx, row in enumerate(sampled)
    ]
    summary = _summarize(
        indexed_rows=rows,
        sampled_rows=sampled_rows,
        sampling_adjustments=sampling_adjustments,
        easy_sanity_max_fraction=easy_sanity_max_fraction,
    )
    return rows, sampled_rows, summary


def _iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            record = json.loads(line)
            if isinstance(record, dict):
                yield idx, record


def _build_record_prefix_states(
    *,
    record: Mapping[str, Any],
    split: str,
    source_dataset_jsonl: Path,
    source_line_idx: int,
    run_id: str,
) -> list[dict[str, Any]]:
    objects = _objects(record)
    if not objects:
        return []
    by_desc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for obj in objects:
        by_desc[obj["desc"]].append(obj)
    descs = sorted(by_desc)
    rows: list[dict[str, Any]] = []
    for desc in descs:
        members = by_desc[desc]
        rows.extend(
            _same_desc_states(
                record=record,
                split=split,
                path=source_dataset_jsonl,
                source_line_idx=source_line_idx,
                run_id=run_id,
                desc=desc,
                members=members,
                all_objects=objects,
                descs=descs,
            )
        )
    if len(descs) >= 2:
        rows.extend(
            _different_desc_states(
                record=record,
                split=split,
                path=source_dataset_jsonl,
                source_line_idx=source_line_idx,
                run_id=run_id,
                all_objects=objects,
                by_desc=by_desc,
                descs=descs,
            )
        )
    return rows


def _objects(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_objects = record.get("objects") or []
    objects: list[dict[str, Any]] = []
    if not isinstance(raw_objects, list):
        return objects
    for gt_idx, obj in enumerate(raw_objects):
        if not isinstance(obj, Mapping):
            continue
        desc = canonical_desc(str(obj.get("desc") or ""))
        bbox = _bbox_values(tuple(obj.get("bbox_2d") or ()))
        if not desc or bbox is None:
            continue
        objects.append(
            {
                "gt_idx": gt_idx,
                "desc": desc,
                "bbox_xyxy": bbox,
                "area": max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1]),
            }
        )
    return objects


def _bbox_values(tokens: tuple[Any, ...]) -> list[int] | None:
    if len(tokens) != 4:
        return None
    values: list[int] = []
    for token in tokens:
        text = str(token)
        if not text.startswith("<|coord_") or not text.endswith("|>"):
            return None
        try:
            values.append(int(text.removeprefix("<|coord_").removesuffix("|>")))
        except ValueError:
            return None
    return values


def _same_desc_states(
    *,
    record: Mapping[str, Any],
    split: str,
    path: Path,
    source_line_idx: int,
    run_id: str,
    desc: str,
    members: Sequence[dict[str, Any]],
    all_objects: Sequence[dict[str, Any]],
    descs: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ordered = sorted(members, key=lambda obj: (obj["bbox_xyxy"][0], obj["bbox_xyxy"][1], obj["gt_idx"]))
    prefixes = [
        ("empty", "same_desc_prefix_k", []),
        ("shallow_1", "same_desc_prefix_k", ordered[:1]),
        ("mid_half", "same_desc_prefix_k", ordered[: max(1, len(ordered) // 2)]),
        ("late_one_left", "same_desc_prefix_k", ordered[:-1]),
    ]
    for depth, condition, emitted in prefixes:
        residual = [obj for obj in ordered if obj["gt_idx"] not in {item["gt_idx"] for item in emitted}]
        if not residual:
            continue
        rows.append(
            _base_row(
                record=record,
                split=split,
                path=path,
                source_line_idx=source_line_idx,
                run_id=run_id,
                transition_type="same_desc_transition",
                prefix_condition=condition,
                prefix_depth=depth,
                prefix_order_policy_id="same_desc_x1_order",
                emitted=emitted,
                residual=residual,
                all_objects=all_objects,
                descs=descs,
                target_desc=desc,
                hard_competitor_desc=_competitor_desc(desc, descs, emitted),
            )
        )
    return rows


def _different_desc_states(
    *,
    record: Mapping[str, Any],
    split: str,
    path: Path,
    source_line_idx: int,
    run_id: str,
    all_objects: Sequence[dict[str, Any]],
    by_desc: Mapping[str, Sequence[dict[str, Any]]],
    descs: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source_desc = max(descs, key=lambda desc: (len(by_desc[desc]), desc))
    target_descs = [desc for desc in descs if desc != source_desc]
    if not target_descs:
        return rows
    target_desc = max(target_descs, key=lambda desc: (len(by_desc[desc]), desc))
    spatial = sorted(all_objects, key=lambda obj: (obj["bbox_xyxy"][0], obj["bbox_xyxy"][1], obj["gt_idx"]))
    salient = sorted(all_objects, key=lambda obj: (-obj["area"], obj["gt_idx"]))
    class_block = list(by_desc[source_desc])
    candidates = [
        ("shallow_1", "different_desc_prefix_k", "spatial_prefix", spatial[:1]),
        ("mid_half", "different_desc_prefix_k", "spatial_prefix", spatial[: max(1, len(spatial) // 2)]),
        ("mid_half", "different_desc_prefix_k", "size_salience_prefix", salient[: max(1, len(salient) // 2)]),
        ("class_block_done", "class_block_prefix", "class_block_prefix", class_block),
        ("late_one_left", "different_desc_prefix_k", "original_order_prefix", list(all_objects[:-1])),
    ]
    for depth, condition, order_policy, emitted in candidates:
        emitted_ids = {obj["gt_idx"] for obj in emitted}
        residual = [obj for obj in all_objects if obj["gt_idx"] not in emitted_ids]
        target_residual = [obj for obj in residual if obj["desc"] == target_desc]
        if not target_residual:
            continue
        rows.append(
            _base_row(
                record=record,
                split=split,
                path=path,
                source_line_idx=source_line_idx,
                run_id=run_id,
                transition_type="different_desc_transition",
                prefix_condition=condition,
                prefix_depth=depth,
                prefix_order_policy_id=order_policy,
                emitted=emitted,
                residual=residual,
                all_objects=all_objects,
                descs=descs,
                target_desc=target_desc,
                hard_competitor_desc=source_desc,
            )
        )
    return rows


def _base_row(
    *,
    record: Mapping[str, Any],
    split: str,
    path: Path,
    source_line_idx: int,
    run_id: str,
    transition_type: str,
    prefix_condition: str,
    prefix_depth: str,
    prefix_order_policy_id: str,
    emitted: Sequence[dict[str, Any]],
    residual: Sequence[dict[str, Any]],
    all_objects: Sequence[dict[str, Any]],
    descs: Sequence[str],
    target_desc: str,
    hard_competitor_desc: str | None,
) -> dict[str, Any]:
    emitted_descs = sorted({obj["desc"] for obj in emitted})
    residual_descs = sorted({obj["desc"] for obj in residual})
    residual_target = [obj for obj in residual if obj["desc"] == target_desc]
    desc_count = len(descs)
    object_count = len(all_objects)
    row = {
        "schema_version": SCHEMA_VERSION,
        "project_id": PROJECT_ID,
        "phase_id": PHASE_ID,
        "run_id": run_id,
        "checkpoint_id": "paired_ckpt3664",
        "checkpoint_role": "paired_index",
        "split": split,
        "source_dataset_jsonl": str(path),
        "source_line_idx": source_line_idx,
        "image_id": record.get("image_id"),
        "image_path": record.get("file_name") or (record.get("images") or [""])[0],
        "transition_type": transition_type,
        "prefix_condition": prefix_condition,
        "prefix_depth": prefix_depth,
        "prefix_order_policy_id": prefix_order_policy_id,
        "emitted_gt_indices": [obj["gt_idx"] for obj in emitted],
        "residual_gt_indices": [obj["gt_idx"] for obj in residual],
        "emitted_descs": emitted_descs,
        "residual_descs": residual_descs,
        "all_descs": list(descs),
        "probe_descs": [desc for desc in (target_desc, hard_competitor_desc) if desc],
        "probe_desc": target_desc,
        "probe_desc_role": "target_residual_desc",
        "readout_type": "prefix_state_index",
        "shard_id": None,
        "target_residual_desc": target_desc,
        "hard_competitor_desc": hard_competitor_desc,
        "desc_count_bucket": _desc_count_bucket(desc_count),
        "object_count_bucket": _object_count_bucket(object_count),
        "residual_target_count": len(residual_target),
        "residual_object_count": len(residual),
        "emitted_object_count": len(emitted),
        "hardness": _hardness(
            transition_type=transition_type,
            residual_target_count=len(residual_target),
            desc_count=desc_count,
            object_count=object_count,
            prefix_depth=prefix_depth,
        ),
        "gt_objects": [
            {
                "gt_idx": obj["gt_idx"],
                "desc": obj["desc"],
                "bbox_xyxy": obj["bbox_xyxy"],
            }
            for obj in all_objects
        ],
    }
    row["prefix_state_id"] = _prefix_state_id(row)
    return row


def _hardness(
    *,
    transition_type: str,
    residual_target_count: int,
    desc_count: int,
    object_count: int,
    prefix_depth: str,
) -> str:
    if transition_type == "same_desc_transition":
        return "headline_hard" if residual_target_count >= 2 else "easy_sanity"
    if desc_count >= 3 and object_count >= 6 and prefix_depth != "late_one_left":
        return "headline_hard"
    return "easy_sanity"


def _prefix_state_id(row: Mapping[str, Any]) -> str:
    payload = "|".join(
        str(row[key])
        for key in (
            "split",
            "source_line_idx",
            "image_id",
            "transition_type",
            "prefix_condition",
            "prefix_depth",
            "prefix_order_policy_id",
            "target_residual_desc",
            "emitted_gt_indices",
            "residual_gt_indices",
        )
    )
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    return f"pst-{digest}"


def _sample_prefix_states(
    rows: Sequence[dict[str, Any]],
    *,
    max_prefix_states: int,
    seed: int,
    easy_sanity_max_fraction: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if max_prefix_states <= 0:
        return [], [{"reason": "max_prefix_states_lte_zero"}]
    rng = random.Random(seed)
    hard = [row for row in rows if row["hardness"] == "headline_hard"]
    easy = [row for row in rows if row["hardness"] == "easy_sanity"]
    hard_sample = _stratified_sample(hard, max_prefix_states, rng)
    easy_fraction_budget = (
        0
        if not hard_sample
        else int(len(hard_sample) * easy_sanity_max_fraction / max(1e-12, 1.0 - easy_sanity_max_fraction))
    )
    easy_budget = min(
        len(easy),
        max(0, easy_fraction_budget),
        max(0, max_prefix_states - len(hard_sample)),
    )
    easy_sample = _stratified_sample(easy, easy_budget, rng)
    sampled = hard_sample + easy_sample
    rng.shuffle(sampled)
    adjustments: list[dict[str, Any]] = []
    if len(hard_sample) < min(len(hard), max_prefix_states):
        adjustments.append(
            {
                "reason": "hard_rows_truncated_to_budget",
                "available": len(hard),
                "selected": len(hard_sample),
            }
        )
    if len(sampled) < min(len(rows), max_prefix_states):
        adjustments.append(
            {
                "reason": "sample_underfilled_after_easy_fraction_cap",
                "available": len(rows),
                "selected": len(sampled),
            }
        )
    return sampled, adjustments


def _stratified_sample(
    rows: Sequence[dict[str, Any]],
    limit: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    if limit <= 0 or not rows:
        return []
    buckets: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[
            (
                row["split"],
                row["transition_type"],
                row["prefix_depth"],
                row["desc_count_bucket"],
                row["object_count_bucket"],
            )
        ].append(row)
    for bucket_rows in buckets.values():
        bucket_rows.sort(key=lambda row: row["prefix_state_id"])
        rng.shuffle(bucket_rows)
    selected: list[dict[str, Any]] = []
    keys = sorted(buckets)
    cursor = 0
    while len(selected) < limit and keys:
        key = keys[cursor % len(keys)]
        bucket = buckets[key]
        if bucket:
            selected.append(bucket.pop())
        if not bucket:
            keys.remove(key)
            if not keys:
                break
            cursor %= len(keys)
        else:
            cursor += 1
    return selected


def _summarize(
    *,
    indexed_rows: Sequence[dict[str, Any]],
    sampled_rows: Sequence[dict[str, Any]],
    sampling_adjustments: Sequence[Mapping[str, Any]],
    easy_sanity_max_fraction: float,
) -> dict[str, Any]:
    failed = _launch_gate_failures(sampled_rows, easy_sanity_max_fraction)
    return {
        "schema_version": SCHEMA_VERSION,
        "project_id": PROJECT_ID,
        "phase_id": PHASE_ID,
        "launch_eligible": not failed,
        "failed_launch_gates": failed,
        "row_counts": {
            "indexed_prefix_states": len(indexed_rows),
            "sampled_prefix_states": len(sampled_rows),
            "easy_sanity_rows": sum(row["hardness"] == "easy_sanity" for row in sampled_rows),
            "headline_hard_rows": sum(row["hardness"] == "headline_hard" for row in sampled_rows),
        },
        "by_split_transition_type": _counter_dict(
            (row["split"], row["transition_type"]) for row in sampled_rows
        ),
        "by_prefix_depth": _counter_dict((row["split"], row["prefix_depth"]) for row in sampled_rows),
        "by_hardness": dict(Counter(row["hardness"] for row in sampled_rows)),
        "mixed_desc_hard_share": _mixed_desc_hard_share(sampled_rows),
        "sampling_adjustments": list(sampling_adjustments),
    }


def _launch_gate_failures(
    rows: Sequence[Mapping[str, Any]],
    easy_sanity_max_fraction: float,
) -> list[str]:
    failures: list[str] = []
    row_set = {(row["split"], row["transition_type"]) for row in rows}
    for split in ("train", "val"):
        for transition_type in ("same_desc_transition", "different_desc_transition"):
            if (split, transition_type) not in row_set:
                failures.append(f"missing_{split}_{transition_type}")
    for split in ("train", "val"):
        depths = {
            str(row["prefix_depth"])
            for row in rows
            if row["split"] == split
            and row["prefix_depth"] in {"shallow_1", "mid_half", "late_one_left", "class_block_done"}
        }
        if len(depths) < 3:
            failures.append(f"insufficient_{split}_prefix_depth_coverage")
    bad_same = [
        row
        for row in rows
        if row["hardness"] == "headline_hard"
        and row["transition_type"] == "same_desc_transition"
        and int(row["residual_target_count"]) < 2
    ]
    if bad_same:
        failures.append("same_desc_headline_residual_count_lt2")
    easy_count = sum(row["hardness"] == "easy_sanity" for row in rows)
    if rows and easy_count / len(rows) > easy_sanity_max_fraction:
        failures.append("easy_sanity_fraction_gt20pct")
    mixed_hard = [
        row
        for row in rows
        if row["transition_type"] == "different_desc_transition"
        and row["hardness"] == "headline_hard"
    ]
    if not mixed_hard:
        failures.append("missing_mixed_desc_hard_rows")
    return failures


def _mixed_desc_hard_share(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    mixed = [row for row in rows if row["transition_type"] == "different_desc_transition"]
    hard = [row for row in mixed if row["hardness"] == "headline_hard"]
    return {
        "mixed_desc_rows": len(mixed),
        "mixed_desc_headline_hard_rows": len(hard),
        "share": 0.0 if not mixed else len(hard) / len(mixed),
    }


def _counter_dict(keys: Iterable[tuple[Any, ...]]) -> dict[str, int]:
    return {"|".join(str(part) for part in key): count for key, count in Counter(keys).items()}


def _sample_index(rows: Sequence[Mapping[str, Any]], prefix_state_id: str) -> int:
    for idx, row in enumerate(rows):
        if row["prefix_state_id"] == prefix_state_id:
            return idx
    return 0


def _competitor_desc(desc: str, descs: Sequence[str], emitted: Sequence[Mapping[str, Any]]) -> str | None:
    emitted_descs = [str(obj["desc"]) for obj in emitted if obj.get("desc") != desc]
    if emitted_descs:
        return sorted(emitted_descs)[0]
    for candidate in descs:
        if candidate != desc:
            return candidate
    return None


def _desc_count_bucket(count: int) -> str:
    if count <= 1:
        return "desc_count1"
    if count == 2:
        return "desc_count2"
    if count == 3:
        return "desc_count3"
    return "desc_count4_plus"


def _object_count_bucket(count: int) -> str:
    if count <= 5:
        return "obj_1_5"
    if count <= 10:
        return "obj_6_10"
    if count <= 20:
        return "obj_11_20"
    return "obj_21_plus"
