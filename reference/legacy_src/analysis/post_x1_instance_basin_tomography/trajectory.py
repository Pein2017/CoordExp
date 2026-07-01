from __future__ import annotations

from typing import Any, Mapping, Sequence


SLOT_ORDER = ("y1", "x2", "y2")
LEGACY_BUCKET_ALIASES = {
    "target_instance": "target",
    "same_desc_competitor": "competitor_same_desc",
    "background_or_outlier": "background",
    "ambiguous_tied": "tied",
}


def classify_trajectory(slot_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ordered = _ordered_slot_rows(slot_rows)
    boundary_extreme_slots = [
        str(row["slot"])
        for row in ordered
        if bool(row.get("boundary_extreme_flag")) or _bucket(row) == "boundary_extreme"
    ]
    slot_winners = {
        str(row["slot"]): _winner_gt_idx(row)
        for row in ordered
        if _winner_gt_idx(row) is not None
    }
    base = {
        "slot_count": len(ordered),
        "slot_order": [str(row["slot"]) for row in ordered],
        "slot_winners": slot_winners,
        "boundary_extreme_slots": boundary_extreme_slots,
        "basin_stay_rate": _target_rate(ordered),
    }
    if any(row.get("coord_mass_low_flag") or _bucket(row) == "invalid_low_coord_mass" for row in ordered):
        return {
            **base,
            "trajectory_taxonomy": "invalid_low_coord_mass",
            "trajectory_bucket": "invalid_low_coord_mass",
            "switch_slot": None,
            "switched_to_instance_id": None,
        }
    if any(_bucket(row) == "tied" or row.get("low_margin_flag") for row in ordered):
        return {
            **base,
            "trajectory_taxonomy": "tied",
            "trajectory_bucket": "tied",
            "switch_slot": None,
            "switched_to_instance_id": None,
        }
    if any(_bucket(row) == "ambiguous" for row in ordered):
        return {
            **base,
            "trajectory_taxonomy": "ambiguous",
            "trajectory_bucket": "ambiguous",
            "switch_slot": None,
            "switched_to_instance_id": None,
        }
    buckets = [_bucket(row) for row in ordered]
    if buckets and all(bucket == "target" for bucket in buckets):
        return {
            **base,
            "trajectory_taxonomy": "target",
            "trajectory_bucket": "stay_target_all_slots",
            "switch_slot": None,
            "switched_to_instance_id": None,
        }
    if buckets and buckets[0] == "competitor_same_desc":
        return {
            **base,
            "trajectory_taxonomy": "competitor_same_desc",
            "trajectory_bucket": "early_switch",
            "switch_slot": str(ordered[0]["slot"]),
            "switched_to_instance_id": ordered[0].get("winner_instance_id"),
        }
    if "competitor_same_desc" in buckets:
        first = _first_row_with_bucket(ordered, "competitor_same_desc")
        return {
            **base,
            "trajectory_taxonomy": "competitor_same_desc",
            "trajectory_bucket": "partial_target_then_switch",
            "switch_slot": str(first["slot"]),
            "switched_to_instance_id": first.get("winner_instance_id"),
        }
    non_target = [row for row in ordered if _bucket(row) != "target"]
    non_target_taxa = {_bucket(row) for row in non_target}
    if len(non_target_taxa) > 1:
        first = non_target[0]
        return {
            **base,
            "trajectory_taxonomy": "mixed",
            "trajectory_bucket": "mixed_drift",
            "switch_slot": str(first["slot"]),
            "switched_to_instance_id": first.get("winner_instance_id"),
        }
    if len(non_target_taxa) == 1:
        taxonomy = non_target_taxa.pop()
        first = non_target[0]
        return {
            **base,
            "trajectory_taxonomy": taxonomy,
            "trajectory_bucket": f"{taxonomy}_drift",
            "switch_slot": str(first["slot"]),
            "switched_to_instance_id": first.get("winner_instance_id"),
        }
    return {
        **base,
        "trajectory_taxonomy": "ambiguous",
        "trajectory_bucket": "ambiguous",
        "switch_slot": None,
        "switched_to_instance_id": None,
    }


def build_attraction_matrix_rows(trajectory_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trajectory in trajectory_rows:
        forced_anchor = int(trajectory["forced_anchor_gt_idx"])
        winners = trajectory.get("slot_winners") or {}
        row: dict[str, Any] = {
            "case_id": trajectory.get("case_id"),
            "checkpoint_role": trajectory.get("checkpoint_role"),
            "desc": trajectory.get("desc"),
            "forced_anchor_gt_idx": forced_anchor,
            "trajectory_bucket": trajectory.get("trajectory_bucket"),
        }
        for slot in SLOT_ORDER:
            winner = winners.get(slot)
            row[f"winner_{slot}_gt_idx"] = None if winner is None else int(winner)
            row[f"diagonal_{slot}"] = winner is not None and int(winner) == forced_anchor
        rows.append(row)
    return rows


def _ordered_slot_rows(slot_rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    by_slot = {str(row.get("slot")): row for row in slot_rows}
    ordered = [by_slot[slot] for slot in SLOT_ORDER if slot in by_slot]
    extras = [row for row in slot_rows if str(row.get("slot")) not in SLOT_ORDER]
    return [*ordered, *extras]


def _bucket(row: Mapping[str, Any]) -> str:
    raw = str(row.get("winner_bucket", row.get("slot_taxonomy", "ambiguous")))
    return LEGACY_BUCKET_ALIASES.get(raw, raw)


def _target_rate(slot_rows: Sequence[Mapping[str, Any]]) -> float:
    if not slot_rows:
        return 0.0
    return sum(1 for row in slot_rows if _bucket(row) == "target") / len(slot_rows)


def _first_row_with_bucket(slot_rows: Sequence[Mapping[str, Any]], bucket: str) -> Mapping[str, Any]:
    for row in slot_rows:
        if _bucket(row) == bucket:
            return row
    raise ValueError(f"missing bucket: {bucket}")


def _winner_gt_idx(row: Mapping[str, Any]) -> int | None:
    winner = row.get("winner_instance_id")
    if winner is None or winner == "target":
        target = row.get("target_gt_idx")
        return None if target is None else int(target)
    return int(winner)
