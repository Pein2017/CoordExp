from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence


MATCH_POLICY_ID = "same_desc_greedy_iou_0.50_v1"
RANDOM_CHECKPOINT_ROLE = "fullobj_random_pure_ce_ckpt3668"
SORTED_CHECKPOINT_ROLE = "fullobj_sorted_pure_ce_ckpt3668"
RANDOM_FN_FIELD = f"is_fn_{RANDOM_CHECKPOINT_ROLE}"
SORTED_FN_FIELD = f"is_fn_{SORTED_CHECKPOINT_ROLE}"
RANDOM_MATCH_FIELD = "random_match_pred_idx"
SORTED_MATCH_FIELD = "sorted_match_pred_idx"
EXPECTED_CHECKPOINT_ROLES = (RANDOM_CHECKPOINT_ROLE, SORTED_CHECKPOINT_ROLE)
REQUIRED_CONTEXT_FIELDS = (
    "split",
    "image_id",
    "source_line_idx",
    "image_path",
    "width",
    "height",
    "coord_mode",
    "bbox_surface",
    "data_root",
    "jsonl_sha256",
    "checkpoint_fingerprint",
    "decode_policy",
    "template_contract",
    "invalid_pred_count",
)
REQUIRED_LEDGER_META_FIELDS = (
    "match_policy_id",
    "pred_rows_ordered",
    "accepted_matches",
)

_WS_RE = re.compile(r"\s+")


def match_fn_cases_for_image(
    gt_rows: Sequence[Mapping[str, Any]],
    pred_rows: Sequence[Mapping[str, Any]],
    *,
    split: str | None = None,
    image_id: Any | None = None,
    match_iou_threshold: float = 0.5,
    near_miss_iou_threshold: float = 0.3,
    duplicate_iou_threshold: float = 0.95,
    match_policy_id: str = MATCH_POLICY_ID,
) -> dict[str, Any]:
    """Match GT to predictions under the A3.2 same-desc FN policy.

    The main FN decision is intentionally narrow: only same-desc one-to-one
    greedy IoU matches at or above ``match_iou_threshold`` count as TP. The
    side labels are computed independently and never promote a GT out of FN.
    """

    gts = [_normalize_gt_row(row, index, split=split, image_id=image_id) for index, row in enumerate(gt_rows)]
    preds = [_normalize_pred_row(row, index) for index, row in enumerate(pred_rows)]
    pair_rows = _pair_rows(gts, preds)

    accepted_matches = _accepted_same_desc_matches(
        pair_rows,
        match_iou_threshold=match_iou_threshold,
        match_policy_id=match_policy_id,
    )
    accepted_by_gt = {
        int(match["gt_idx"]): match for match in accepted_matches
    }
    accepted_pair_keys = {
        (int(match["gt_idx"]), int(match["pred_idx"])) for match in accepted_matches
    }
    pred_to_accepted_gt = {
        int(match["pred_idx"]): int(match["gt_idx"]) for match in accepted_matches
    }

    gt_match_rows: list[dict[str, Any]] = []
    for gt in sorted(gts, key=lambda row: int(row["gt_idx"])):
        gt_idx = int(gt["gt_idx"])
        same_desc_pairs = [
            pair
            for pair in pair_rows
            if int(pair["gt_idx"]) == gt_idx and bool(pair["same_desc"])
        ]
        wrong_desc_pairs = [
            pair
            for pair in pair_rows
            if int(pair["gt_idx"]) == gt_idx and not bool(pair["same_desc"])
        ]
        accepted = accepted_by_gt.get(gt_idx)
        duplicate_pair = _first_duplicate_pair(
            same_desc_pairs,
            accepted_pair_keys=accepted_pair_keys,
            duplicate_iou_threshold=duplicate_iou_threshold,
        )
        near_miss = any(
            near_miss_iou_threshold <= float(pair["iou"]) < match_iou_threshold
            for pair in same_desc_pairs
        )
        wrong_desc_overlap = any(
            float(pair["iou"]) >= match_iou_threshold for pair in wrong_desc_pairs
        )
        best_same_desc_iou = max(
            (float(pair["iou"]) for pair in same_desc_pairs),
            default=None,
        )
        gt_match_rows.append(
            _json_safe(
                {
                    "gt_object_key": gt["gt_object_key"],
                    "split": gt.get("split"),
                    "image_id": gt.get("image_id"),
                    "gt_idx": gt_idx,
                    "gt_desc": gt["desc"],
                    "gt_bbox": gt["bbox_xyxy"],
                    "gt_sorted_rank": gt.get("gt_sorted_rank"),
                    "same_desc_gt_count": gt.get("same_desc_gt_count"),
                    "object_count": gt.get("object_count"),
                    "is_fn": accepted is None,
                    "match_pred_idx": (
                        None if accepted is None else int(accepted["pred_idx"])
                    ),
                    "match_iou": None if accepted is None else float(accepted["iou"]),
                    "match_policy_id": match_policy_id,
                    "match_candidates_same_desc": _candidate_rows(same_desc_pairs),
                    "best_same_desc_iou": best_same_desc_iou,
                    "near_miss": near_miss,
                    "wrong_desc_overlap": wrong_desc_overlap,
                    "same_desc_duplicate": duplicate_pair is not None,
                    "duplicate_source": _duplicate_source(
                        duplicate_pair,
                        pred_to_accepted_gt=pred_to_accepted_gt,
                    ),
                },
                "gt_match_row",
            )
        )

    return _json_safe(
        {
            "match_policy_id": match_policy_id,
            "split": split,
            "image_id": image_id,
            "pred_rows_ordered": sorted(
                preds,
                key=lambda row: int(row["pred_idx"]),
            ),
            "accepted_matches": accepted_matches,
            "gt_match_rows": gt_match_rows,
        },
        "match_ledger",
    )


def build_fn_case_universe(
    gt_rows: Sequence[Mapping[str, Any]],
    *,
    random_match_ledger: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    sorted_match_ledger: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    match_ledgers_by_role: Mapping[
        str,
        Mapping[str, Any] | Sequence[Mapping[str, Any]],
    ]
    | None = None,
    split: str,
    image_id: Any,
    sampled_gt_object_keys: set[str] | Sequence[str] | None = None,
    sampling_reasons: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Build denominator-safe per-GT rows carrying checkpoint FN statuses."""

    ledgers_by_role = _normalize_universe_ledgers(
        random_match_ledger=random_match_ledger,
        sorted_match_ledger=sorted_match_ledger,
        match_ledgers_by_role=match_ledgers_by_role,
    )
    rows_by_role = {
        role: _ledger_rows_by_key(ledger) for role, ledger in ledgers_by_role.items()
    }
    sampled_keys = None if sampled_gt_object_keys is None else {str(key) for key in sampled_gt_object_keys}
    reasons = {str(key): str(value) for key, value in (sampling_reasons or {}).items()}
    gt_key_rows = _gt_keys_by_key(gt_rows, split=split, image_id=image_id)
    gt_keys = set(gt_key_rows)
    for role, rows_by_key in rows_by_role.items():
        _validate_ledger_key_set(f"match_ledgers_by_role[{role}]", rows_by_key, gt_keys)

    rows: list[dict[str, Any]] = []
    for gt_object_key in sorted(gt_key_rows):
        ledger_rows = {
            role: _required_ledger_row(rows_by_key, gt_object_key)
            for role, rows_by_key in rows_by_role.items()
        }
        fn_by_role = {
            role: bool(row["is_fn"]) for role, row in ledger_rows.items()
        }
        sampled_for_probe = (
            any(fn_by_role.values())
            if sampled_keys is None
            else gt_object_key in sampled_keys
        )
        payload: dict[str, Any] = {
            "gt_object_key": gt_object_key,
            "fn_membership": _fn_membership_by_role(fn_by_role),
            "sampled_for_probe": sampled_for_probe,
            "sampling_reason": _sampling_reason(
                gt_object_key,
                sampled_for_probe=sampled_for_probe,
                sampling_reasons=reasons,
            ),
        }
        for role, row in ledger_rows.items():
            payload[_fn_field_for_role(role)] = fn_by_role[role]
            payload[_match_field_for_role(role)] = row.get("match_pred_idx")
        rows.append(
            _json_safe(
                payload,
                "fn_case_universe_row",
            )
        )
    return rows


def build_replayable_fn_cases(
    universe_rows: Sequence[Mapping[str, Any]],
    *,
    match_ledgers_by_role: Mapping[str, Mapping[str, Any] | Sequence[Mapping[str, Any]]],
    contexts_by_role: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Expand sampled FN universe rows into deterministic replayable case rows."""

    expected_roles = tuple(sorted(str(role) for role in match_ledgers_by_role))
    _validate_role_mapping("contexts_by_role", contexts_by_role, expected_roles)
    _validate_role_mapping(
        "match_ledgers_by_role",
        match_ledgers_by_role,
        expected_roles,
    )
    _validate_replay_match_ledgers_by_role(
        match_ledgers_by_role,
        expected_roles=expected_roles,
    )
    _validate_contexts_by_role(contexts_by_role, expected_roles=expected_roles)
    _validate_sampled_universe_role_fields(
        universe_rows,
        expected_roles=expected_roles,
    )

    ledger_rows_by_role = {
        str(role): _ledger_rows_by_key(ledger)
        for role, ledger in match_ledgers_by_role.items()
        if str(role) in expected_roles
    }
    ledger_meta_by_role = {
        str(role): _ledger_meta(ledger)
        for role, ledger in match_ledgers_by_role.items()
        if str(role) in expected_roles
    }
    _validate_ledger_meta_by_role(
        ledger_meta_by_role,
        expected_roles=expected_roles,
    )

    rows: list[dict[str, Any]] = []
    for universe_row in sorted(
        universe_rows,
        key=lambda row: str(row["gt_object_key"]),
    ):
        if not bool(universe_row.get("sampled_for_probe")):
            continue
        gt_object_key = str(universe_row["gt_object_key"])
        for role in expected_roles:
            fn_field = _fn_field_for_role(role)
            if not bool(universe_row[fn_field]):
                continue
            context = contexts_by_role[role]
            gt_match_row = _required_ledger_row(
                ledger_rows_by_role[role],
                gt_object_key,
            )
            ledger_meta = ledger_meta_by_role[role]
            case = {
                "gt_object_key": gt_object_key,
                "checkpoint_role": role,
                "split": _required_mapping_value(context, "split", f"context[{role}]"),
                "image_id": _required_mapping_value(
                    context,
                    "image_id",
                    f"context[{role}]",
                ),
                "source_line_idx": _required_mapping_value(
                    context,
                    "source_line_idx",
                    f"context[{role}]",
                ),
                "image_path": _required_mapping_value(
                    context,
                    "image_path",
                    f"context[{role}]",
                ),
                "width": _required_mapping_value(context, "width", f"context[{role}]"),
                "height": _required_mapping_value(
                    context,
                    "height",
                    f"context[{role}]",
                ),
                "coord_mode": _required_mapping_value(
                    context,
                    "coord_mode",
                    f"context[{role}]",
                ),
                "bbox_surface": _required_mapping_value(
                    context,
                    "bbox_surface",
                    f"context[{role}]",
                ),
                "data_root": _required_mapping_value(
                    context,
                    "data_root",
                    f"context[{role}]",
                ),
                "jsonl_sha256": _required_mapping_value(
                    context,
                    "jsonl_sha256",
                    f"context[{role}]",
                ),
                "checkpoint_fingerprint": _required_mapping_value(
                    context,
                    "checkpoint_fingerprint",
                    f"context[{role}]",
                ),
                "decode_policy": _required_mapping_value(
                    context,
                    "decode_policy",
                    f"context[{role}]",
                ),
                "template_contract": _required_mapping_value(
                    context,
                    "template_contract",
                    f"context[{role}]",
                ),
                "fn_gt_idx": gt_match_row["gt_idx"],
                "fn_desc": gt_match_row["gt_desc"],
                "fn_bbox": gt_match_row["gt_bbox"],
                "gt_sorted_rank": gt_match_row.get("gt_sorted_rank"),
                "same_desc_gt_count": gt_match_row.get("same_desc_gt_count"),
                "object_count": gt_match_row.get("object_count"),
                "pred_rows_ordered": ledger_meta["pred_rows_ordered"],
                "match_policy_id": gt_match_row.get(
                    "match_policy_id",
                    ledger_meta["match_policy_id"],
                ),
                "match_candidates_same_desc": gt_match_row.get(
                    "match_candidates_same_desc",
                    [],
                ),
                "accepted_matches": ledger_meta["accepted_matches"],
                "best_same_desc_iou": gt_match_row.get("best_same_desc_iou"),
                "near_miss": bool(gt_match_row.get("near_miss", False)),
                "wrong_desc_overlap": bool(
                    gt_match_row.get("wrong_desc_overlap", False)
                ),
                "same_desc_duplicate": bool(
                    gt_match_row.get("same_desc_duplicate", False)
                ),
                "duplicate_source": gt_match_row.get("duplicate_source"),
                "invalid_pred_count": int(
                    _required_mapping_value(
                        context,
                        "invalid_pred_count",
                        f"context[{role}]",
                    )
                ),
                "sample_stratum": universe_row["fn_membership"],
            }
            case["fn_case_id"] = _fn_case_id(
                checkpoint_role=role,
                gt_object_key=gt_object_key,
                jsonl_sha256=case.get("jsonl_sha256"),
                checkpoint_fingerprint=case.get("checkpoint_fingerprint"),
                match_policy_id=case.get("match_policy_id"),
            )
            rows.append(_json_safe(case, "fn_case_row"))
    return sorted(
        rows,
        key=lambda row: (str(row["gt_object_key"]), str(row["checkpoint_role"])),
    )


def _accepted_same_desc_matches(
    pair_rows: Sequence[Mapping[str, Any]],
    *,
    match_iou_threshold: float,
    match_policy_id: str,
) -> list[dict[str, Any]]:
    candidates = [
        pair
        for pair in pair_rows
        if bool(pair["same_desc"]) and float(pair["iou"]) >= match_iou_threshold
    ]
    candidates = sorted(
        candidates,
        key=lambda pair: (
            -float(pair["iou"]),
            int(pair["pred_idx"]),
            int(pair["gt_idx"]),
        ),
    )

    matched_preds: set[int] = set()
    matched_gts: set[int] = set()
    matches: list[dict[str, Any]] = []
    for pair in candidates:
        pred_idx = int(pair["pred_idx"])
        gt_idx = int(pair["gt_idx"])
        if pred_idx in matched_preds or gt_idx in matched_gts:
            continue
        matched_preds.add(pred_idx)
        matched_gts.add(gt_idx)
        matches.append(
            _json_safe(
                {
                    "gt_idx": gt_idx,
                    "pred_idx": pred_idx,
                    "iou": float(pair["iou"]),
                    "gt_object_key": pair["gt_object_key"],
                    "match_policy_id": match_policy_id,
                },
                "accepted_match",
            )
        )
    return matches


def _pair_rows(
    gts: Sequence[Mapping[str, Any]],
    preds: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for gt in gts:
        for pred in preds:
            iou = _bbox_iou_xyxy(gt["bbox_xyxy"], pred["bbox_xyxy"])
            rows.append(
                {
                    "gt_idx": gt["gt_idx"],
                    "pred_idx": pred["pred_idx"],
                    "gt_object_key": gt["gt_object_key"],
                    "same_desc": gt["canonical_desc"] == pred["canonical_desc"],
                    "iou": iou,
                }
            )
    return rows


def _candidate_rows(
    same_desc_pairs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        _json_safe(
            {
                "pred_idx": int(pair["pred_idx"]),
                "iou": float(pair["iou"]),
            },
            "same_desc_candidate",
        )
        for pair in sorted(
            same_desc_pairs,
            key=lambda pair: (
                -float(pair["iou"]),
                int(pair["pred_idx"]),
                int(pair["gt_idx"]),
            ),
        )
    ]


def _first_duplicate_pair(
    same_desc_pairs: Sequence[Mapping[str, Any]],
    *,
    accepted_pair_keys: set[tuple[int, int]],
    duplicate_iou_threshold: float,
) -> Mapping[str, Any] | None:
    duplicate_pairs = [
        pair
        for pair in same_desc_pairs
        if float(pair["iou"]) > duplicate_iou_threshold
        and (int(pair["gt_idx"]), int(pair["pred_idx"])) not in accepted_pair_keys
    ]
    if not duplicate_pairs:
        return None
    return sorted(
        duplicate_pairs,
        key=lambda pair: (
            -float(pair["iou"]),
            int(pair["pred_idx"]),
            int(pair["gt_idx"]),
        ),
    )[0]


def _duplicate_source(
    duplicate_pair: Mapping[str, Any] | None,
    *,
    pred_to_accepted_gt: Mapping[int, int],
) -> str | None:
    if duplicate_pair is None:
        return None
    pred_idx = int(duplicate_pair["pred_idx"])
    source_gt_idx = pred_to_accepted_gt.get(pred_idx, int(duplicate_pair["gt_idx"]))
    return f"same_desc_iou_gt_{source_gt_idx}_pred_{pred_idx}"


def _normalize_gt_row(
    row: Mapping[str, Any],
    fallback_idx: int,
    *,
    split: str | None,
    image_id: Any | None,
) -> dict[str, Any]:
    gt_idx = _row_int(row, "gt_idx", fallback_idx)
    row_split = row.get("split", split)
    row_image_id = row.get("image_id", image_id)
    desc = _desc_from_row(row)
    return _json_safe(
        {
            "gt_object_key": _gt_object_key(row_split, row_image_id, gt_idx, row),
            "split": row_split,
            "image_id": row_image_id,
            "gt_idx": gt_idx,
            "desc": desc,
            "canonical_desc": _canonical_desc(desc),
            "bbox_xyxy": _bbox_from_row(row),
            "gt_sorted_rank": row.get("gt_sorted_rank"),
            "same_desc_gt_count": row.get("same_desc_gt_count"),
            "object_count": row.get("object_count"),
        },
        "gt_row",
    )


def _normalize_pred_row(row: Mapping[str, Any], fallback_idx: int) -> dict[str, Any]:
    pred_idx = _row_int(row, "pred_idx", fallback_idx)
    desc = _desc_from_row(row)
    return _json_safe(
        {
            "pred_idx": pred_idx,
            "desc": desc,
            "canonical_desc": _canonical_desc(desc),
            "bbox_xyxy": _bbox_from_row(row),
        },
        "pred_row",
    )


def _desc_from_row(row: Mapping[str, Any]) -> str:
    for key in ("desc", "description", "label", "category"):
        value = row.get(key)
        if value is not None:
            return str(value)
    return ""


def _bbox_from_row(row: Mapping[str, Any]) -> list[int | float]:
    for key in ("bbox_xyxy", "bbox", "fn_bbox", "bbox_2d", "points"):
        value = row.get(key)
        if value is not None:
            return _coerce_bbox(value, f"{key}")
    raise ValueError(f"row missing bbox field: {row!r}")


def _coerce_bbox(value: Any, path: str) -> list[int | float]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise ValueError(f"{path} must be a sequence of four numbers")
    if len(value) != 4:
        raise ValueError(f"{path} must contain four numbers, got {len(value)}")
    return [_clean_number(item, f"{path}[{index}]") for index, item in enumerate(value)]


def _clean_number(value: Any, path: str) -> int | float:
    if isinstance(value, bool):
        raise ValueError(f"{path} must be numeric, got bool")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} must be numeric, got {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{path} must be finite")
    if number.is_integer():
        return int(number)
    return number


def _bbox_iou_xyxy(a: Sequence[int | float], b: Sequence[int | float]) -> float:
    ax1, ay1, ax2, ay2 = [float(value) for value in a]
    bx1, by1, bx2, by2 = [float(value) for value in b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0.0 else 0.0


def _ledger_rows_by_key(
    ledger: Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    rows = _ledger_rows(ledger)
    rows_by_key: dict[str, Mapping[str, Any]] = {}
    duplicate_keys: list[str] = []
    for row in rows:
        if "gt_object_key" not in row:
            raise ValueError("match ledger row missing gt_object_key")
        key = str(row["gt_object_key"])
        if key in rows_by_key:
            duplicate_keys.append(key)
            continue
        rows_by_key[key] = row
    if duplicate_keys:
        raise ValueError(
            "duplicate match ledger gt_object_key values: "
            + ", ".join(sorted(set(duplicate_keys)))
        )
    return rows_by_key


def _ledger_rows(
    ledger: Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> Sequence[Mapping[str, Any]]:
    if isinstance(ledger, Mapping):
        raw_rows = ledger.get("gt_match_rows")
        if not isinstance(raw_rows, Sequence):
            raise ValueError("match ledger must contain gt_match_rows")
        return raw_rows
    return ledger


def _ledger_meta(
    ledger: Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if isinstance(ledger, Mapping):
        return {
            "match_policy_id": ledger.get("match_policy_id"),
            "pred_rows_ordered": ledger.get("pred_rows_ordered"),
            "accepted_matches": ledger.get("accepted_matches"),
        }
    return {
        "match_policy_id": MATCH_POLICY_ID,
        "pred_rows_ordered": [],
        "accepted_matches": [],
    }


def _gt_keys_by_key(
    gt_rows: Sequence[Mapping[str, Any]],
    *,
    split: str,
    image_id: Any,
) -> dict[str, Mapping[str, Any]]:
    rows_by_key: dict[str, Mapping[str, Any]] = {}
    duplicate_keys: list[str] = []
    for index, gt_row in enumerate(gt_rows):
        gt_idx = _row_int(gt_row, "gt_idx", index)
        key = _gt_object_key(split, image_id, gt_idx, gt_row)
        if key in rows_by_key:
            duplicate_keys.append(key)
            continue
        rows_by_key[key] = gt_row
    if duplicate_keys:
        raise ValueError(
            "duplicate GT gt_object_key values: "
            + ", ".join(sorted(set(duplicate_keys)))
        )
    return rows_by_key


def _validate_ledger_key_set(
    label: str,
    rows_by_key: Mapping[str, Mapping[str, Any]],
    expected_keys: set[str],
) -> None:
    actual_keys = set(rows_by_key)
    missing = sorted(expected_keys - actual_keys)
    extra = sorted(actual_keys - expected_keys)
    problems: list[str] = []
    if missing:
        problems.append("missing keys: " + ", ".join(missing))
    if extra:
        problems.append("extra keys: " + ", ".join(extra))
    if problems:
        raise ValueError(f"{label} key mismatch; " + "; ".join(problems))


def _validate_role_mapping(
    label: str,
    mapping: Mapping[str, Any],
    expected_roles: Sequence[str],
) -> None:
    actual_roles = {str(role) for role in mapping}
    expected = set(expected_roles)
    missing = sorted(expected - actual_roles)
    extra = sorted(actual_roles - expected)
    problems: list[str] = []
    if missing:
        problems.append("missing expected roles: " + ", ".join(missing))
    if extra:
        problems.append("unexpected roles: " + ", ".join(extra))
    if problems:
        raise ValueError(f"{label} role mismatch; " + "; ".join(problems))


def _validate_contexts_by_role(
    contexts_by_role: Mapping[str, Mapping[str, Any]],
    *,
    expected_roles: Sequence[str],
) -> None:
    for role in expected_roles:
        context = contexts_by_role[role]
        for field in REQUIRED_CONTEXT_FIELDS:
            _required_mapping_value(context, field, f"context[{role}]")


def _validate_sampled_universe_role_fields(
    universe_rows: Sequence[Mapping[str, Any]],
    *,
    expected_roles: Sequence[str],
) -> None:
    for row in universe_rows:
        if not bool(row.get("sampled_for_probe")):
            continue
        gt_object_key = str(row.get("gt_object_key", "<missing>"))
        for role in expected_roles:
            field = _fn_field_for_role(role)
            if field not in row:
                raise ValueError(
                    f"sampled universe row {gt_object_key} missing {field}"
                )
            if not isinstance(row[field], bool):
                raise ValueError(
                    f"sampled universe row {gt_object_key} field {field} must be bool"
                )


def _validate_replay_match_ledgers_by_role(
    match_ledgers_by_role: Mapping[str, Mapping[str, Any] | Sequence[Mapping[str, Any]]],
    *,
    expected_roles: Sequence[str],
) -> None:
    for role in expected_roles:
        ledger = match_ledgers_by_role[role]
        if not isinstance(ledger, Mapping):
            raise ValueError(
                f"match_ledgers_by_role[{role}] must be a mapping replay ledger"
            )
        for field in (*REQUIRED_LEDGER_META_FIELDS, "gt_match_rows"):
            _required_mapping_value(
                ledger,
                field,
                f"match_ledgers_by_role[{role}]",
            )


def _validate_ledger_meta_by_role(
    ledger_meta_by_role: Mapping[str, Mapping[str, Any]],
    *,
    expected_roles: Sequence[str],
) -> None:
    for role in expected_roles:
        meta = ledger_meta_by_role[role]
        for field in REQUIRED_LEDGER_META_FIELDS:
            _required_mapping_value(meta, field, f"match_ledger[{role}]")


def _required_mapping_value(
    mapping: Mapping[str, Any],
    field: str,
    path: str,
) -> Any:
    if field not in mapping:
        raise ValueError(f"{path} missing required field: {field}")
    value = mapping[field]
    if value is None:
        raise ValueError(f"{path}.{field} must not be None")
    return value


def _required_ledger_row(
    rows_by_key: Mapping[str, Mapping[str, Any]],
    gt_object_key: str,
) -> Mapping[str, Any]:
    row = rows_by_key.get(gt_object_key)
    if row is None:
        raise ValueError(f"missing match ledger row for GT object: {gt_object_key}")
    return row


def _gt_object_key(
    split: Any | None,
    image_id: Any | None,
    gt_idx: int,
    row: Mapping[str, Any] | None = None,
) -> str:
    if row is not None and row.get("gt_object_key") is not None:
        return str(row["gt_object_key"])
    if split is None or image_id is None:
        return f"unknown:{image_id}:{gt_idx}"
    return f"{split}:{image_id}:{gt_idx}"


def _fn_membership(random_is_fn: bool, sorted_is_fn: bool) -> str:
    if random_is_fn and sorted_is_fn:
        return "shared_fn"
    if random_is_fn:
        return "random_only_fn"
    if sorted_is_fn:
        return "sorted_only_fn"
    return "not_fn"


def _fn_membership_by_role(fn_by_role: Mapping[str, bool]) -> str:
    if set(fn_by_role) == set(EXPECTED_CHECKPOINT_ROLES):
        return _fn_membership(
            bool(fn_by_role[RANDOM_CHECKPOINT_ROLE]),
            bool(fn_by_role[SORTED_CHECKPOINT_ROLE]),
        )
    fn_roles = sorted(role for role, is_fn in fn_by_role.items() if bool(is_fn))
    if not fn_roles:
        return "not_fn"
    return "fn:" + ",".join(fn_roles)


def _sampling_reason(
    gt_object_key: str,
    *,
    sampled_for_probe: bool,
    sampling_reasons: Mapping[str, str],
) -> str:
    if gt_object_key in sampling_reasons:
        return str(sampling_reasons[gt_object_key])
    if sampled_for_probe:
        return "selected_for_probe"
    return "not_sampled"


def _fn_field_for_role(role: str) -> str:
    return f"is_fn_{role}"


def _match_field_for_role(role: str) -> str:
    if role == RANDOM_CHECKPOINT_ROLE:
        return RANDOM_MATCH_FIELD
    if role == SORTED_CHECKPOINT_ROLE:
        return SORTED_MATCH_FIELD
    return f"match_pred_idx_{role}"


def _normalize_universe_ledgers(
    *,
    random_match_ledger: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
    sorted_match_ledger: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
    match_ledgers_by_role: Mapping[
        str,
        Mapping[str, Any] | Sequence[Mapping[str, Any]],
    ]
    | None,
) -> dict[str, Mapping[str, Any] | Sequence[Mapping[str, Any]]]:
    if match_ledgers_by_role is not None:
        if not match_ledgers_by_role:
            raise ValueError("match_ledgers_by_role must not be empty")
        return {str(role): ledger for role, ledger in match_ledgers_by_role.items()}
    if random_match_ledger is None or sorted_match_ledger is None:
        raise ValueError(
            "build_fn_case_universe requires either match_ledgers_by_role "
            "or both random_match_ledger and sorted_match_ledger"
        )
    return {
        RANDOM_CHECKPOINT_ROLE: random_match_ledger,
        SORTED_CHECKPOINT_ROLE: sorted_match_ledger,
    }


def _context_or_match(
    context: Mapping[str, Any],
    gt_match_row: Mapping[str, Any],
    key: str,
) -> Any:
    value = context.get(key)
    return gt_match_row.get(key) if value is None else value


def _fn_case_id(
    *,
    checkpoint_role: str,
    gt_object_key: str,
    jsonl_sha256: Any,
    checkpoint_fingerprint: Any,
    match_policy_id: Any,
) -> str:
    payload = {
        "checkpoint_role": checkpoint_role,
        "gt_object_key": gt_object_key,
        "jsonl_sha256": jsonl_sha256,
        "checkpoint_fingerprint": checkpoint_fingerprint,
        "match_policy_id": match_policy_id,
    }
    digest = hashlib.sha256(
        json.dumps(_json_safe(payload, "fn_case_id"), sort_keys=True).encode("utf-8")
    ).hexdigest()
    return f"fn_{digest[:24]}"


def _row_int(row: Mapping[str, Any], key: str, fallback: int) -> int:
    value = row.get(key, fallback)
    if isinstance(value, bool):
        raise ValueError(f"{key} must be an integer, got bool")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be an integer, got {value!r}") from exc


def _canonical_desc(value: str) -> str:
    return _WS_RE.sub(" ", str(value).strip().lower())


def _json_safe(value: Any, path: str) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item, f"{path}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, tuple | list):
        return [_json_safe(item, f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must be finite")
        return value
    if isinstance(value, str | int):
        return value
    return str(value)


__all__ = [
    "MATCH_POLICY_ID",
    "RANDOM_CHECKPOINT_ROLE",
    "SORTED_CHECKPOINT_ROLE",
    "build_fn_case_universe",
    "build_replayable_fn_cases",
    "match_fn_cases_for_image",
]
