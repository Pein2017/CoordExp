from __future__ import annotations

import hashlib
import random
from typing import Any, Mapping, Sequence


def build_probe_plan(
    case_rows: Sequence[Mapping[str, Any]],
    *,
    num_shards: int,
    max_cases: int | None = None,
    seed: int = 3664,
    sampling_policy_id: str = "stratified_round_robin_v1",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    unique: dict[str, Mapping[str, Any]] = {}
    for row in case_rows:
        unique.setdefault(str(row["case_id"]), row)
    case_ids = sorted(unique)
    rng = random.Random(seed)
    rng.shuffle(case_ids)
    sampled_ids = set(case_ids if max_cases is None else case_ids[:max_cases])
    policy_sha = hashlib.sha256(f"{sampling_policy_id}|{seed}|{num_shards}|{max_cases}".encode()).hexdigest()
    plan: list[dict[str, Any]] = []
    sampled_index = 0
    for case_id in sorted(unique):
        row = unique[case_id]
        sampled = case_id in sampled_ids
        shard = sampled_index % num_shards if sampled else None
        if sampled:
            sampled_index += 1
        plan.append(
            {
                "probe_plan_row_id": f"pp-{len(plan):06d}",
                "case_id": case_id,
                "case_index_row_id": row["case_index_row_id"],
                "probe_sampled": sampled,
                "sampling_policy_id": sampling_policy_id,
                "sampling_policy_sha256": policy_sha,
                "sampling_seed": seed,
                "sampling_weight": 1.0,
                "strata_key": _strata_key(row),
                "planned_shard_id": shard,
                "planned_gpu_id": shard,
                "planned_stage_set": [
                    "x1_candidate_field",
                    "residual_row_scoring",
                    "basin_attraction",
                    "attention_components",
                ],
                "planned_status": "planned" if sampled else "skipped",
                "planned_skip_reason": None if sampled else "not_sampled",
            }
        )
    summary = {
        "case_index_total_cases": len(unique),
        "gpu_probe_planned_cases": len(sampled_ids),
        "sampling_policy_id": sampling_policy_id,
        "sampling_policy_sha256": policy_sha,
        "num_shards": num_shards,
    }
    return plan, summary


def _strata_key(row: Mapping[str, Any]) -> str:
    parts = [
        row.get("split"),
        row.get("pool_role"),
        row.get("same_desc_count_bucket"),
        row.get("desc_text_canonical") or row.get("desc_text"),
        row.get("fn_rescue_overlay_membership"),
        row.get("object_size_bucket"),
        row.get("overlap_bucket"),
    ]
    return "|".join(str(part) for part in parts)
