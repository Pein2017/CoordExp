#!/usr/bin/env python3
"""Run the human-refined greedy set-completion condition grid.

This is intentionally an experiment-local runner.  The pure schedule, stop,
logit-processor, and matching helpers can be imported without importing
``torch`` or ``transformers``.  The command-line path loads one HF/SDPA
session per process and runs one image at a time with the exact integer token
ids of each ground-truth prefix.

The broad grid is not a benchmark and it does not replace native greedy
evidence.  Rows supplied as a prefix are context only; coverage is computed
from the free suffix with a global one-to-one owner assignment.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys
from typing import Any


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


SCHEMA_VERSION = "human_refined_greedy_set_completion_conditions.v1"
MAX_NEW_TOKENS = 4096
REPETITION_PENALTY = 1.0
DEFAULT_IMAGE_IDS = (
    "1584",
    "2685",
    "4134",
    "5001",
    "6040",
    "7511",
    "10707",
    "13348",
    "13923",
    "14038",
    "14439",
    "16228",
)
DEFAULT_RANDOM_SEEDS = (11, 12)
DEFAULT_REMAINING_DEPTHS = ("N", "16", "8", "4", "2", "1")
DEFAULT_CONDITIONS = ("native", "first_only", "full")
DEFAULT_BUDGET_MODES = ("strict", "relaxed")
DEFAULT_ORDERS = ("geometry", "reverse", "category", "random")
DEFAULT_INFER_CONFIG = Path(
    "configs/coordexp_swift/infer/research/"
    "qwen3_vl_2b_positive_path_imitation_source_human_refined_12_hf.yaml"
)

GEOMETRY_ORDER = "geometry"
REVERSE_ORDER = "reverse"
CATEGORY_ORDER = "category"
SOURCE_ORDER = "source"
RANDOM_ORDER = "random"
SAME_REMAINING_PREFIX = "same_remaining"
CONDITIONS = frozenset(DEFAULT_CONDITIONS)
BUDGET_MODES = frozenset(DEFAULT_BUDGET_MODES)

# Qwen3-VL's coordinate wrapper ids are stable for the step-4887 tokenizer.
# The structural counter still receives the ids explicitly at each call so a
# future tokenizer mismatch cannot silently credit a bare ``box_end`` token.
OBJECT_REF_START_TOKEN_ID = 151646
OBJECT_REF_END_TOKEN_ID = 151647
BOX_START_TOKEN_ID = 151648
BOX_END_TOKEN_ID = 151649
COORDINATE_TOKEN_MIN_ID = 151670
COORDINATE_TOKEN_MAX_ID = 152669

def normalize_condition(value: str) -> str:
    condition = str(value).strip().lower().replace("-", "_")
    aliases = {
        "first": "first_only",
        "first_terminal_only": "first_only",
        "suppress_first": "first_only",
        "all": "full",
        "full_suppression": "full",
    }
    condition = aliases.get(condition, condition)
    if condition not in CONDITIONS:
        raise ValueError(f"unsupported terminal condition {value!r}")
    return condition


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _normalize_category(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().replace("_", " ").split())


def _bbox(value: Any) -> tuple[float, float, float, float]:
    if isinstance(value, Mapping):
        value = value.get("bbox", value.get("bbox_xyxy", value.get("bbox_2d")))
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        raise ValueError(f"bbox must contain four values: {value!r}")
    converted: list[float] = []
    for item in value:
        if isinstance(item, str):
            match = re.search(r"coord_(\d+)", item)
            if match is None:
                raise ValueError(f"unsupported coordinate value: {item!r}")
            converted.append(float(match.group(1)))
        else:
            converted.append(float(item))
    result = tuple(converted)
    if any(not math.isfinite(item) for item in result) or result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError(f"bbox must be finite xyxy with positive area: {value!r}")
    return result  # type: ignore[return-value]


def _row_id(row: Mapping[str, Any]) -> str:
    for key in ("owner_id", "object_id", "coco_ann_id", "id"):
        if row.get(key) is not None:
            return str(row[key])
    raise ValueError("ground-truth row requires owner_id/object_id/coco_ann_id")


def _row_category(row: Mapping[str, Any]) -> str:
    category = _normalize_category(
        row.get("category", row.get("category_name", row.get("desc", row.get("description"))))
    )
    if not category:
        raise ValueError(f"ground-truth row {_row_id(row)!r} lacks a category")
    return category


def _geometry_key(row: Mapping[str, Any]) -> tuple[float, float, float, float, str]:
    box = _bbox(row.get("bbox", row.get("bbox_xyxy", row.get("bbox_2d"))))
    return (box[1], box[0], box[3], box[2], _row_id(row))


def _stable_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source_index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise ValueError("ground-truth rows must be mappings")
        row = dict(raw)
        owner_id = _row_id(row)
        if owner_id in seen:
            raise ValueError(f"duplicate owner id {owner_id!r}")
        seen.add(owner_id)
        row["owner_id"] = owner_id
        row["category"] = _row_category(row)
        row["bbox"] = list(_bbox(row.get("bbox", row.get("bbox_xyxy", row.get("bbox_2d")))))
        row["source_index"] = source_index
        normalized.append(row)
    if not normalized:
        raise ValueError("at least one ground-truth row is required")
    return normalized


def stable_geometry_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return rows in deterministic top-to-bottom then left-to-right order."""

    return sorted(_stable_rows(rows), key=_geometry_key)


def _seeded_shuffle(rows: Sequence[Mapping[str, Any]], seed: int, image_id: str | None = None) -> list[dict[str, Any]]:
    result = list(rows)
    # A text seed keeps independent image shards reproducible and avoids Python
    # hash randomisation.  Do not use the process-global random generator.
    seed_text = f"human-refined-greedy:{seed}:{image_id or ''}"
    stable_seed = int.from_bytes(hashlib.sha256(seed_text.encode("utf-8")).digest()[:8], "big")
    random.Random(stable_seed).shuffle(result)
    return result


def build_policy_schedules(
    rows: Sequence[Mapping[str, Any]],
    *,
    random_seeds: Iterable[int] = DEFAULT_RANDOM_SEEDS,
    image_id: str | None = None,
    orders: Iterable[str] | None = None,
) -> dict[str, list[str]]:
    """Build policy schedules with schedule-specific suffixes.

    ``geometry`` is the canonical top-left order.  ``source`` preserves the
    supplied row order for callers that need to attest that it is already
    geometry sorted.  Random schedules are named ``random:<seed>``.
    """

    normalized = _stable_rows(rows)
    geometry = sorted(normalized, key=_geometry_key)
    requested = tuple(orders) if orders is not None else (*DEFAULT_ORDERS, SOURCE_ORDER)
    result: dict[str, list[dict[str, Any]]] = {}
    for order in requested:
        name = str(order).strip().lower()
        if name in {GEOMETRY_ORDER, "geometry_sorted", "geo_sorted"}:
            result[GEOMETRY_ORDER] = geometry
        elif name in {SOURCE_ORDER, "source_order"}:
            result[SOURCE_ORDER] = list(normalized)
        elif name in {REVERSE_ORDER, "reverse_geometry"}:
            result[REVERSE_ORDER] = list(reversed(geometry))
        elif name in {CATEGORY_ORDER, "category_grouped", "category_group"}:
            result[CATEGORY_ORDER] = sorted(geometry, key=lambda row: (row["category"], *_geometry_key(row)))
        elif name in {RANDOM_ORDER, "random_seeded", "random"}:
            for seed in random_seeds:
                seed_int = int(seed)
                result[f"{RANDOM_ORDER}:{seed_int}"] = _seeded_shuffle(geometry, seed_int, image_id)
        elif name.startswith("random:"):
            seed_int = int(name.split(":", 1)[1])
            result[f"{RANDOM_ORDER}:{seed_int}"] = _seeded_shuffle(geometry, seed_int, image_id)
        else:
            raise ValueError(f"unsupported order {order!r}")
    return {name: [str(row["owner_id"]) for row in schedule] for name, schedule in result.items()}


def build_same_remaining_set_schedules(
    rows: Sequence[Mapping[str, Any]],
    remaining_depth: int,
    *,
    random_seeds: Iterable[int] = DEFAULT_RANDOM_SEEDS,
    image_id: str | None = None,
    orders: Iterable[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build controls that keep the geometry suffix owner set identical.

    The forced complement is reordered according to each policy.  The final
    forced row (the row immediately before the frozen geometry suffix) is kept
    fixed whenever at least two rows are forced.
    """

    normalized = _stable_rows(rows)
    geometry_rows = sorted(normalized, key=_geometry_key)
    total = len(geometry_rows)
    depth = int(remaining_depth)
    if depth < 1 or depth > total:
        raise ValueError(f"remaining depth must lie in [1,{total}], got {depth}")
    geometry_ids = [str(row["owner_id"]) for row in geometry_rows]
    remaining_ids = geometry_ids[-depth:]
    forced_ids = geometry_ids[:-depth]
    final_forced = forced_ids[-1] if len(forced_ids) >= 2 else None
    policy = build_policy_schedules(
        normalized,
        random_seeds=random_seeds,
        image_id=image_id,
        orders=orders,
    )
    result: dict[str, dict[str, Any]] = {}
    for schedule_name, policy_ids in policy.items():
        # Filter policy order to the forced complement.  The suffix is never
        # taken from this policy order: it is always the geometry suffix.
        earlier = [owner_id for owner_id in policy_ids if owner_id in set(forced_ids)]
        if final_forced is not None:
            earlier = [owner_id for owner_id in earlier if owner_id != final_forced]
            forced_order = [*earlier, final_forced]
        else:
            forced_order = earlier
        if set(forced_order) != set(forced_ids) or len(forced_order) != len(forced_ids):
            raise AssertionError("same-remaining-set forced complement changed")
        full_order = [*forced_order, *remaining_ids]
        result[f"{SAME_REMAINING_PREFIX}:{schedule_name}"] = {
            "schedule_name": schedule_name,
            "order": full_order,
            "forced_owner_ids": forced_order,
            "remaining_owner_ids": list(remaining_ids),
            "remaining_depth": depth,
            "same_remaining_owner_set": True,
            "final_forced_owner_id": final_forced,
        }
    return result


def build_remaining_depths(total_objects: int, requested: Iterable[int | str] | str | None = None) -> tuple[int, ...]:
    """Resolve ``N,16,8,4,2,1`` and skip depths unavailable for one image."""

    total = int(total_objects)
    if total <= 0:
        raise ValueError("total_objects must be positive")
    raw = DEFAULT_REMAINING_DEPTHS if requested is None else requested
    pieces = [piece.strip() for piece in raw.split(",") if piece.strip()] if isinstance(raw, str) else list(raw)
    result: list[int] = []
    for piece in pieces:
        if isinstance(piece, str) and piece.strip().upper() == "N":
            depth = total
        else:
            try:
                depth = int(piece)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid remaining depth {piece!r}") from exc
        if depth < 1 or depth > total:
            continue
        if depth not in result:
            result.append(depth)
    if not result:
        raise ValueError(f"no requested remaining depth is valid for N={total}")
    return tuple(result)


def complete_row_budget(remaining_objects: int, mode: str) -> int:
    """Return strict ``m`` or relaxed ``m+max(4,ceil(.25m))`` budget."""

    remaining = int(remaining_objects)
    if remaining <= 0:
        raise ValueError("remaining_objects must be positive")
    mode = str(mode).strip().lower()
    if mode == "strict":
        return remaining
    if mode == "relaxed":
        return remaining + max(4, math.ceil(0.25 * remaining))
    raise ValueError(f"unsupported budget mode {mode!r}")


def render_ground_truth_row(row: Mapping[str, Any]) -> str:
    """Render one exact compact row without adding separators or terminal text."""

    description = str(row.get("description", row.get("desc", row.get("category", "")))).strip()
    if not description:
        raise ValueError(f"owner {_row_id(row)!r} has an empty description")
    box = _bbox(row.get("bbox", row.get("bbox_xyxy", row.get("bbox_2d"))))
    coords = "".join(f"<|coord_{int(value)}|>" for value in box)
    return (
        f"<|object_ref_start|>{description}<|object_ref_end|>"
        f"<|box_start|>{coords}<|box_end|>"
    )


def tokenize_ground_truth_rows(rows: Sequence[Mapping[str, Any]], tokenizer: Any) -> dict[str, list[int]]:
    """Tokenize each GT row once and retain only integer ids for appending."""

    result: dict[str, list[int]] = {}
    for raw in _stable_rows(rows):
        row = dict(raw)
        row_text = str(row.get("row_text") or render_ground_truth_row(row))
        encoded = tokenizer(row_text, add_special_tokens=False)
        ids = encoded.get("input_ids") if isinstance(encoded, Mapping) else None
        if isinstance(ids, list) and len(ids) == 1 and isinstance(ids[0], list):
            ids = ids[0]
        if not isinstance(ids, Sequence) or isinstance(ids, (str, bytes)) or not ids:
            raise ValueError(f"tokenizer returned no row ids for owner {row['owner_id']!r}")
        result[str(row["owner_id"])] = [int(value) for value in ids]
    return result


def complete_row_token_summary(
    token_ids: Sequence[int],
    *,
    object_ref_start_token_id: int = OBJECT_REF_START_TOKEN_ID,
    object_ref_end_token_id: int = OBJECT_REF_END_TOKEN_ID,
    box_start_token_id: int = BOX_START_TOKEN_ID,
    box_end_token_id: int = BOX_END_TOKEN_ID,
    coordinate_token_min_id: int = COORDINATE_TOKEN_MIN_ID,
    coordinate_token_max_id: int = COORDINATE_TOKEN_MAX_ID,
) -> tuple[int, int | None]:
    """Count only exact compact rows in token ids and return the last end.

    The second value is the inclusive token index of the last structurally
    complete row, or ``None``.  A bare ``box_end`` or a malformed tail is not
    a row and therefore cannot create a row boundary or consume budget.
    """

    ids = [int(value) for value in token_ids]
    count = 0
    last_end: int | None = None
    index = 0
    while index < len(ids):
        if ids[index] != int(object_ref_start_token_id):
            index += 1
            continue
        cursor = index + 1
        description_length = 0
        # A description must contain at least one ordinary token.  Structural
        # wrapper/coordinate tokens inside the description invalidate the
        # candidate rather than being mistaken for a later row.
        while cursor < len(ids) and ids[cursor] != int(object_ref_end_token_id):
            token_id = ids[cursor]
            if token_id in {
                int(object_ref_start_token_id),
                int(box_start_token_id),
                int(box_end_token_id),
            } or int(coordinate_token_min_id) <= token_id <= int(coordinate_token_max_id):
                description_length = 0
                break
            description_length += 1
            cursor += 1
        if description_length < 1 or cursor >= len(ids) or ids[cursor] != int(object_ref_end_token_id):
            index += 1
            continue
        cursor += 1
        if cursor >= len(ids) or ids[cursor] != int(box_start_token_id):
            index += 1
            continue
        cursor += 1
        if cursor + 4 >= len(ids):
            index += 1
            continue
        coordinates = ids[cursor : cursor + 4]
        if any(
            not (int(coordinate_token_min_id) <= token_id <= int(coordinate_token_max_id))
            for token_id in coordinates
        ):
            index += 1
            continue
        row_end = cursor + 4
        if ids[row_end] != int(box_end_token_id):
            index += 1
            continue
        count += 1
        last_end = row_end
        index = row_end + 1
    return count, last_end


def count_complete_rows_from_token_ids(token_ids: Sequence[int], **kwargs: Any) -> int:
    """Shared token-structural complete-row counter used by all stop paths."""

    return complete_row_token_summary(token_ids, **kwargs)[0]


def _boundary_from_ids(suffix_ids: Sequence[int], **kwargs: Any) -> bool:
    if not suffix_ids:
        return True
    _count, last_end = complete_row_token_summary(suffix_ids, **kwargs)
    return last_end is not None and last_end == len(suffix_ids) - 1


class RowBoundaryTerminalSuppressor:
    """Suppress ``im_end`` only at a row boundary and retain scalar evidence.

    The processor deliberately stores no vocabulary-sized tensors.  At each
    possible override it records FP32 terminal, selected, and best-nonterminal
    scalar scores plus the raw-top1 decision before masking.
    """

    def __init__(
        self,
        *,
        prompt_width: int,
        im_end_token_id: int,
        box_end_token_id: int | None,
        condition: str,
        row_budget: int,
        object_ref_start_token_id: int = OBJECT_REF_START_TOKEN_ID,
        object_ref_end_token_id: int = OBJECT_REF_END_TOKEN_ID,
        box_start_token_id: int = BOX_START_TOKEN_ID,
        coordinate_token_min_id: int = COORDINATE_TOKEN_MIN_ID,
        coordinate_token_max_id: int = COORDINATE_TOKEN_MAX_ID,
    ) -> None:
        condition = normalize_condition(condition)
        if prompt_width < 0 or row_budget < 1:
            raise ValueError("prompt_width must be non-negative and row_budget positive")
        self.prompt_width = int(prompt_width)
        self.im_end_token_id = int(im_end_token_id)
        self.box_end_token_id = None if box_end_token_id is None else int(box_end_token_id)
        self.condition = condition
        self.row_budget = int(row_budget)
        self._counter_kwargs = {
            "object_ref_start_token_id": int(object_ref_start_token_id),
            "object_ref_end_token_id": int(object_ref_end_token_id),
            "box_start_token_id": int(box_start_token_id),
            "box_end_token_id": int(self.box_end_token_id if self.box_end_token_id is not None else BOX_END_TOKEN_ID),
            "coordinate_token_min_id": int(coordinate_token_min_id),
            "coordinate_token_max_id": int(coordinate_token_max_id),
        }
        self.override_receipts: list[dict[str, Any]] = []
        self.first_override_used = False

    def _rows(self, suffix_ids: Sequence[int]) -> int:
        return count_complete_rows_from_token_ids(suffix_ids, **self._counter_kwargs)

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        import torch

        ids = input_ids[0, self.prompt_width:].tolist()
        rows = self._rows(ids)
        boundary = _boundary_from_ids(ids, **self._counter_kwargs)
        allow = self.condition != "native" and boundary and rows < self.row_budget
        if self.condition == "first_only" and self.first_override_used:
            allow = False
        if not allow:
            return scores

        raw = scores[0].detach().to(dtype=torch.float32)
        terminal_score = float(raw[self.im_end_token_id].item())
        raw_top_id = int(torch.argmax(raw).item())
        raw_top_score = float(raw[raw_top_id].item())
        nonterminal = raw.clone()
        nonterminal[self.im_end_token_id] = -torch.inf
        best_nonterminal_id = int(torch.argmax(nonterminal).item())
        best_nonterminal_score = float(nonterminal[best_nonterminal_id].item())
        raw_terminal_top1 = raw_top_id == self.im_end_token_id
        # First-only means "the first actual native terminal choice".  If
        # im_end is not raw top-1, masking it cannot change the generation and
        # is not counted as an override.
        if self.condition == "first_only" and not raw_terminal_top1:
            return scores
        selected_id = best_nonterminal_id if raw_terminal_top1 else raw_top_id
        selected_score = float(raw[selected_id].item())
        processed = scores.clone()
        processed[0, self.im_end_token_id] = -torch.inf
        self.override_receipts.append(
            {
                "generation_step": len(ids),
                "generated_token_count_before_selection": len(ids),
                "complete_row_count_before_selection": rows,
                "row_boundary": True,
                "condition": self.condition,
                "override_applied": True,
                "raw_terminal_top1": raw_terminal_top1,
                "actual_raw_top1_terminal_override": raw_terminal_top1,
                "raw_top1_token_id": raw_top_id,
                "raw_top1_score_fp32": raw_top_score,
                "raw_im_end_score_fp32": terminal_score,
                "raw_selected_token_id": selected_id,
                "raw_selected_score_fp32": selected_score,
                "raw_best_nonterminal_token_id": best_nonterminal_id,
                "raw_best_nonterminal_score_fp32": best_nonterminal_score,
                "raw_terminal_minus_selected_margin_fp32": terminal_score - selected_score,
            }
        )
        if self.condition == "first_only":
            self.first_override_used = True
        return processed


class RowBudgetStoppingCriteria:
    """Transformers-compatible stop hook for the complete-row budget."""

    def __init__(
        self,
        *,
        prompt_width: int,
        row_budget: int,
        box_end_token_id: int | None,
        object_ref_start_token_id: int = OBJECT_REF_START_TOKEN_ID,
        object_ref_end_token_id: int = OBJECT_REF_END_TOKEN_ID,
        box_start_token_id: int = BOX_START_TOKEN_ID,
        coordinate_token_min_id: int = COORDINATE_TOKEN_MIN_ID,
        coordinate_token_max_id: int = COORDINATE_TOKEN_MAX_ID,
    ) -> None:
        self.prompt_width = int(prompt_width)
        self.row_budget = int(row_budget)
        self.box_end_token_id = None if box_end_token_id is None else int(box_end_token_id)
        self._counter_kwargs = {
            "object_ref_start_token_id": int(object_ref_start_token_id),
            "object_ref_end_token_id": int(object_ref_end_token_id),
            "box_start_token_id": int(box_start_token_id),
            "box_end_token_id": int(self.box_end_token_id if self.box_end_token_id is not None else BOX_END_TOKEN_ID),
            "coordinate_token_min_id": int(coordinate_token_min_id),
            "coordinate_token_max_id": int(coordinate_token_max_id),
        }
        self.complete_row_count = 0
        self.stop_reason: str | None = None

    def __call__(self, input_ids: Any, scores: Any = None, **_: Any) -> Any:
        import torch

        suffix = input_ids[0, self.prompt_width:].tolist()
        self.complete_row_count = count_complete_rows_from_token_ids(suffix, **self._counter_kwargs)
        should_stop = self.complete_row_count >= self.row_budget
        if should_stop:
            self.stop_reason = "row_budget"
        return torch.tensor([should_stop], dtype=torch.bool, device=input_ids.device)


def classify_termination(
    *,
    generated_token_ids: Sequence[int],
    complete_row_count: int,
    row_budget: int,
    im_end_token_id: int | None,
    max_new_tokens: int = MAX_NEW_TOKENS,
    stopping_reason: str | None = None,
) -> str:
    """Classify the first conclusive termination cause.

    ``token_limit`` is explicitly invalid evidence for completion claims.
    """

    if stopping_reason in {"row_budget", "native_im_end", "token_limit", "error"}:
        return str(stopping_reason)
    if int(complete_row_count) >= int(row_budget):
        return "row_budget"
    if im_end_token_id is not None and generated_token_ids and int(generated_token_ids[-1]) == int(im_end_token_id):
        return "native_im_end"
    if len(generated_token_ids) >= int(max_new_tokens):
        return "token_limit"
    return "unknown"


def termination_is_valid(termination: str) -> bool:
    return str(termination) not in {"token_limit", "error", "unknown"}


def _iou(left: Sequence[float], right: Sequence[float]) -> float:
    x1, y1 = max(float(left[0]), float(right[0])), max(float(left[1]), float(right[1]))
    x2, y2 = min(float(left[2]), float(right[2])), min(float(left[3]), float(right[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection <= 0:
        return 0.0
    area_left = max(0.0, float(left[2]) - float(left[0])) * max(0.0, float(left[3]) - float(left[1]))
    area_right = max(0.0, float(right[2]) - float(right[0])) * max(0.0, float(right[3]) - float(right[1]))
    denominator = area_left + area_right - intersection
    return intersection / denominator if denominator > 0 else 0.0


def _assignment(predictions: Sequence[Mapping[str, Any]], owners: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    try:
        from scripts.research.analyze_individual_trajectory_union_support import (
            _min_cost_max_cardinality_assignment,
        )

        return _min_cost_max_cardinality_assignment(predictions, owners)
    except ImportError:
        # Small deterministic cardinality-first fallback.  It is intentionally
        # local: the current exact assignment helper remains authoritative when
        # importable.
        candidates = []
        for p_index, prediction in enumerate(predictions):
            for o_index, owner in enumerate(owners):
                if prediction["category"] != owner["category"]:
                    continue
                overlap = _iou(prediction["bbox"], owner["bbox"])
                if overlap >= 0.5:
                    candidates.append((-overlap, p_index, o_index))
        used_predictions: set[int] = set()
        used_owners: set[int] = set()
        result = []
        for _negative_iou, p_index, o_index in sorted(candidates):
            if p_index in used_predictions or o_index in used_owners:
                continue
            used_predictions.add(p_index)
            used_owners.add(o_index)
            prediction, owner = predictions[p_index], owners[o_index]
            result.append(
                {
                    "prediction_id": str(prediction["prediction_id"]),
                    "generated_row_index": int(prediction["generated_row_index"]),
                    "owner_id": str(owner["owner_id"]),
                    "owner_category": str(owner["category"]),
                    "owner_bbox": tuple(owner["bbox"]),
                    "category": str(prediction["category"]),
                    "intersection_over_union": _iou(prediction["bbox"], owner["bbox"]),
                }
            )
        return result


def match_predictions_one_to_one(
    predictions: Sequence[Mapping[str, Any]], owners: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Match rows globally and keep ambiguity/duplicates in review queues."""

    normalized_predictions = []
    for index, raw in enumerate(predictions):
        prediction = dict(raw)
        prediction["prediction_id"] = str(prediction.get("prediction_id", prediction.get("object_span_id", f"prediction-{index}")))
        prediction["generated_row_index"] = int(prediction.get("generated_row_index", prediction.get("generated_order", index)))
        prediction["category"] = _normalize_category(prediction.get("category", prediction.get("description")))
        prediction["bbox"] = list(_bbox(prediction["bbox"]))
        normalized_predictions.append(prediction)
    normalized_owners = []
    for raw in owners:
        owner = dict(raw)
        owner["owner_id"] = _row_id(owner)
        owner["category"] = _row_category(owner)
        owner["bbox"] = list(_bbox(owner["bbox"]))
        normalized_owners.append(owner)

    matches = _assignment(normalized_predictions, normalized_owners)
    matched_by_prediction = {str(item["prediction_id"]): item for item in matches}
    candidate_by_prediction: dict[str, list[tuple[str, float]]] = {}
    for prediction in normalized_predictions:
        candidates = [
            (str(owner["owner_id"]), _iou(prediction["bbox"], owner["bbox"]))
            for owner in normalized_owners
            if owner["category"] == prediction["category"] and _iou(prediction["bbox"], owner["bbox"]) >= 0.5
        ]
        candidate_by_prediction[prediction["prediction_id"]] = sorted(candidates, key=lambda pair: (-pair[1], pair[0]))

    committed: list[dict[str, Any]] = []
    duplicate_candidates: list[dict[str, Any]] = []
    unresolved_rows: list[dict[str, Any]] = []
    ambiguous_rows: list[dict[str, Any]] = []
    for prediction in normalized_predictions:
        prediction_id = prediction["prediction_id"]
        candidates = candidate_by_prediction[prediction_id]
        match = matched_by_prediction.get(prediction_id)
        if len(candidates) > 1:
            ambiguous_rows.append({**prediction, "candidate_owner_ids": [item[0] for item in candidates], "candidate_owner_ious": {item[0]: item[1] for item in candidates}})
            continue
        if match is not None:
            committed.append(match)
            continue
        if candidates and candidates[0][0] in {str(item["owner_id"]) for item in committed}:
            duplicate_candidates.append({**prediction, "candidate_owner_ids": [item[0] for item in candidates]})
        else:
            unresolved_rows.append({**prediction, "candidate_owner_ids": [item[0] for item in candidates]})

    matched_owner_ids = sorted({str(item["owner_id"]) for item in committed})
    return {
        "matches": sorted(committed, key=lambda item: (int(item["generated_row_index"]), str(item["prediction_id"]))),
        "matched_owner_ids": matched_owner_ids,
        "matched_prediction_ids": sorted(str(item["prediction_id"]) for item in committed),
        "duplicate_candidates": duplicate_candidates,
        "ambiguous_rows": ambiguous_rows,
        "unresolved_rows": unresolved_rows,
        "review_needed_rows": [*duplicate_candidates, *ambiguous_rows, *unresolved_rows],
        "conservative_lower_bound_coverage": len(matched_owner_ids),
    }


def partition_owner_matches(
    matching: Mapping[str, Any],
    *,
    remaining_owner_ids: Sequence[str],
    prefix_owner_ids: Sequence[str],
) -> dict[str, Any]:
    """Partition global committed matches into free and forced-context owners."""

    matched = {str(value) for value in matching.get("matched_owner_ids", ())}
    remaining = [str(value) for value in remaining_owner_ids]
    prefix = [str(value) for value in prefix_owner_ids]
    remaining_discovered = [owner_id for owner_id in remaining if owner_id in matched]
    forced_repeated = [owner_id for owner_id in prefix if owner_id in matched]
    lower_bound = len(remaining_discovered)
    denominator = len(remaining)
    return {
        "remaining_owner_ids_discovered": remaining_discovered,
        "forced_context_owner_ids_repeated": forced_repeated,
        "remaining_conservative_lower_bound_coverage": lower_bound,
        "remaining_coverage_fraction": (lower_bound / denominator if denominator else 0.0),
        "completion_flag": denominator > 0 and lower_bound == denominator,
    }


def _parse_generated_text(
    text: str, *, image_width: int, image_height: int, row_id: str
) -> dict[str, Any]:
    from src.inference.parsing import parse_compact_object_box_closed

    parsed = parse_compact_object_box_closed(
        text,
        row_id=row_id,
        row_index=0,
        image_width=int(image_width),
        image_height=int(image_height),
    )
    result = []
    for index, prediction in enumerate(parsed.predictions):
        pixel_bbox = list(_bbox(prediction["bbox"]))
        # The parser already retains the authoritative coord bins.  Matching
        # must use those bins directly; converting pixel boxes back to bins
        # introduces a needless rounding/scale path.
        coord_bins = [int(value) for value in prediction.get("coord_bins", ())]
        if len(coord_bins) != 4:
            raise ValueError("accepted parser prediction lacks four coord_bins")
        normalized_bbox = list(coord_bins)
        result.append(
            {
                **dict(prediction),
                "bbox_pixel": pixel_bbox,
                "bbox": normalized_bbox,
                "bbox_format": "xyxy_norm1000",
                "prediction_id": str(prediction.get("object_span_id", f"{row_id}:prediction-{index}")),
                "generated_row_index": int(prediction.get("generated_order", index)),
                "category": _normalize_category(prediction.get("description")),
            }
        )
    parser_artifact = parsed.to_artifact_dict()
    parser_artifact["predictions"] = [dict(item) for item in parser_artifact.get("predictions", [])]
    parser_artifact["dropped_predictions"] = [dict(item) for item in parser_artifact.get("dropped_predictions", [])]
    return {
        "parse_status": parsed.parse_status,
        "predictions": result,
        "dropped_predictions": parser_artifact["dropped_predictions"],
        "dropped_prediction_count": len(parser_artifact["dropped_predictions"]),
        "parser_artifact": parser_artifact,
    }


def _token_id(tokenizer: Any, token: str) -> int:
    value = tokenizer.convert_tokens_to_ids(token)
    if value is None:
        raise ValueError(f"tokenizer has no id for {token}")
    return int(value)


def _processor_config(config: Any) -> Any:
    from src.config.models import ProcessorConfig

    return ProcessorConfig(
        do_resize=config.model.processor.do_resize,
        max_raw_pixels=1_000_000_000,
        max_merged_visual_tokens=1_000_000,
    )


def _template_config(config: Any) -> Any:
    from src.config.models import TemplateConfig, TemplatePromptConfig

    return TemplateConfig(
        object_field_order=config.template.object_field_order,
        object_ordering=config.template.object_ordering,
        assistant_format=config.template.assistant_format,
        prompt=TemplatePromptConfig(system=config.template.prompt.system, user=config.template.prompt.user),
    )


def _select_example(examples: Sequence[Any], image_id: str) -> Any:
    matches = []
    for example in examples:
        source = example.metadata.get("source", {}) if isinstance(getattr(example, "metadata", None), Mapping) else {}
        physical = source.get("image_id") if isinstance(source, Mapping) else None
        if str(example.example_id) == str(image_id) or str(physical) == str(image_id):
            matches.append(example)
    if len(matches) != 1:
        raise ValueError(f"image_id {image_id!r} resolved to {len(matches)} examples")
    return matches[0]


def _build_request(config: Any, frontend: Any, example: Any) -> tuple[Any, Any, Mapping[str, Any]]:
    from src.inference.backend import DecodeRequest, GenerationPolicy
    from src.inference.image_plan import plan_image_batch
    from src.inference.prompt import build_prompt_record

    plan = plan_image_batch([example], components=frontend.qwen, processor_config=_processor_config(config), row_indices=[0]).rows[0]
    record = build_prompt_record(example, _template_config(config), processor=frontend.qwen.processor, row_index=0, merged_visual_tokens=plan.merged_visual_tokens)
    request = DecodeRequest(
        request_id=f"human-refined-completion:{example.example_id}",
        chat_text=record.chat_text,
        input_prompt_token_ids=tuple(record.input_prompt_token_ids),
        expected_executed_prompt_token_ids=tuple(record.expected_executed_prompt_token_ids),
        image_path=plan.image_path,
        declared_image_width=plan.declared_width,
        declared_image_height=plan.declared_height,
        decoded_image_width=plan.decoded_width,
        decoded_image_height=plan.decoded_height,
        image_sha256=plan.image_content_sha256,
        expected_image_grid_thw=tuple(plan.expected_image_grid_thw),
        logical_transform_id=plan.logical_transform_id,
        generation_policy=GenerationPolicy(max_new_tokens=1),
    )
    return request, plan, {
        "prompt_token_ids": list(record.expected_executed_prompt_token_ids),
        "prompt_token_ids_sha256": _sha256_json(record.expected_executed_prompt_token_ids),
        "chat_text_sha256": hashlib.sha256(record.chat_text.encode("utf-8")).hexdigest(),
        "image_path": str(plan.image_path),
        "image_sha256": plan.image_content_sha256,
        "width": int(plan.decoded_width),
        "height": int(plan.decoded_height),
    }


def _append_prefix(native_inputs: Mapping[str, Any], prefix_token_ids: Sequence[int]) -> tuple[dict[str, Any], int]:
    import torch

    input_ids = native_inputs["input_ids"]
    prefix = torch.tensor([list(map(int, prefix_token_ids))], dtype=input_ids.dtype, device=input_ids.device)
    model_inputs = dict(native_inputs)
    model_inputs["input_ids"] = torch.cat((input_ids, prefix), dim=1)
    if "attention_mask" in model_inputs:
        model_inputs["attention_mask"] = torch.cat((model_inputs["attention_mask"], torch.ones_like(prefix)), dim=1)
    return model_inputs, int(model_inputs["input_ids"].shape[1])


def _generation_kwargs(
    *,
    model_inputs: Mapping[str, Any],
    session: Any,
    prompt_width: int,
    processor: RowBoundaryTerminalSuppressor,
    stopping: RowBudgetStoppingCriteria,
) -> dict[str, Any]:
    """Build bounded generate kwargs without retaining vocabulary tensors.

    Scalar FP32 stop evidence is captured by ``processor`` before masking.
    ``output_scores``/``output_logits`` are intentionally disabled here: on a
    4,096-token run retaining every vocabulary tensor can consume gigabytes.
    The artifact records this semantic explicitly.
    """

    try:
        from transformers import LogitsProcessorList, StoppingCriteriaList
    except ImportError as exc:  # pragma: no cover - runtime-only dependency
        raise RuntimeError("transformers is required for the HF runner") from exc
    return {
        **dict(model_inputs),
        "max_new_tokens": MAX_NEW_TOKENS,
        "repetition_penalty": REPETITION_PENALTY,
        "do_sample": False,
        "eos_token_id": session._im_end_token_id(),
        "pad_token_id": session._pad_token_id(),
        "return_dict_in_generate": True,
        "output_scores": False,
        "logits_processor": LogitsProcessorList([processor]),
        "stopping_criteria": StoppingCriteriaList([stopping]),
        "_completion_prompt_width": int(prompt_width),
    }


def generation_call_key(
    prefix_token_ids: Sequence[int], condition: str, row_budget: int
) -> tuple[tuple[int, ...], str, int]:
    """Exact deduplication key for one image-local model.generate call."""

    return (tuple(int(value) for value in prefix_token_ids), normalize_condition(condition), int(row_budget))


def generation_call_id(key: tuple[Sequence[int], str, int]) -> str:
    """Stable short receipt id for an exact generation call key."""

    prefix, condition, row_budget = key
    return "completion-call-" + _sha256_json(
        {
            "prefix_token_ids": [int(value) for value in prefix],
            "condition": str(condition),
            "row_budget": int(row_budget),
        }
    )[:20]


def _generate_shared_outcome(
    *,
    session: Any,
    native_inputs: Mapping[str, Any],
    tokenizer: Any,
    plan: Any,
    owner_rows: Sequence[Mapping[str, Any]],
    prefix_token_ids: Sequence[int],
    condition: str,
    row_budget: int,
    image_id: str,
    shared_call_id: str,
) -> dict[str, Any]:
    """Execute and parse one exact generation call, independent of schedule."""

    import torch
    from src.templates.renderer import BOX_END_TOKEN

    model_inputs, prompt_width = _append_prefix(native_inputs, prefix_token_ids)
    box_end_token_id = _token_id(tokenizer, BOX_END_TOKEN)
    processor = RowBoundaryTerminalSuppressor(
        prompt_width=prompt_width,
        im_end_token_id=session._im_end_token_id(),
        box_end_token_id=box_end_token_id,
        condition=condition,
        row_budget=row_budget,
    )
    stopping = RowBudgetStoppingCriteria(
        prompt_width=prompt_width,
        row_budget=row_budget,
        box_end_token_id=box_end_token_id,
    )
    kwargs = _generation_kwargs(
        model_inputs=model_inputs,
        session=session,
        prompt_width=prompt_width,
        processor=processor,
        stopping=stopping,
    )
    kwargs.pop("_completion_prompt_width", None)
    with torch.inference_mode():
        output = session._model.generate(**kwargs)
    sequences = getattr(output, "sequences", None)
    if sequences is None:
        raise RuntimeError("HF generate returned no sequences")
    generated_ids = [int(value) for value in sequences[0, prompt_width:].tolist()]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
    complete_rows = count_complete_rows_from_token_ids(generated_ids, box_end_token_id=box_end_token_id)
    parser_evidence = _parse_generated_text(
        generated_text,
        image_width=int(plan.decoded_width),
        image_height=int(plan.decoded_height),
        row_id=f"{image_id}:shared:{shared_call_id}",
    )
    # The parser's generated_order counts malformed object candidates too.
    # Generation is already bounded by structurally complete rows, so filtering
    # accepted predictions by that raw ordinal can discard a valid row that
    # follows a malformed candidate.
    parsed_predictions = list(parser_evidence["predictions"])
    matching = match_predictions_one_to_one(parsed_predictions, owner_rows)
    termination = classify_termination(
        generated_token_ids=generated_ids,
        complete_row_count=complete_rows,
        row_budget=row_budget,
        im_end_token_id=session._im_end_token_id(),
        max_new_tokens=MAX_NEW_TOKENS,
        stopping_reason=stopping.stop_reason,
    )
    row_termination_cause = (
        "row_budget"
        if complete_rows >= row_budget
        else "native_im_end"
        if generated_ids and int(generated_ids[-1]) == int(session._im_end_token_id())
        else "incomplete"
    )
    token_termination_cause = (
        "native_im_end"
        if generated_ids and int(generated_ids[-1]) == int(session._im_end_token_id())
        else "max_new_tokens"
        if len(generated_ids) >= MAX_NEW_TOKENS
        else "row_budget_stop"
        if stopping.stop_reason == "row_budget"
        else "unknown"
    )
    return {
        "status": "valid" if termination_is_valid(termination) else "invalid_evidence",
        "condition": normalize_condition(condition),
        "row_budget": int(row_budget),
        "shared_call_id": shared_call_id,
        "raw_generated_token_ids": generated_ids,
        "raw_generated_token_ids_sha256": _sha256_json(generated_ids),
        "raw_generated_text": generated_text,
        "complete_row_count": int(complete_rows),
        "termination": termination,
        "row_termination_cause": row_termination_cause,
        "token_termination_cause": token_termination_cause,
        "token_limit_invalid_evidence": termination == "token_limit",
        "suppression_receipts": list(processor.override_receipts),
        "actual_raw_top1_terminal_override_count": sum(
            bool(item.get("actual_raw_top1_terminal_override")) for item in processor.override_receipts
        ),
        "logit_capture_semantics": "FP32 scalar pre-suppression terminal/selected margins captured by stateful processor; full vocabulary tensors not retained",
        "parsed_predictions": parsed_predictions,
        "parser_evidence": parser_evidence,
        "parse_status": parser_evidence["parse_status"],
        "dropped_predictions": parser_evidence["dropped_predictions"],
        "dropped_prediction_count": int(parser_evidence["dropped_prediction_count"]),
        "owner_matching": matching,
    }


def _run_one_arm(
    *,
    session: Any,
    native_inputs: Mapping[str, Any],
    tokenizer: Any,
    plan: Any,
    row_map: Mapping[str, Mapping[str, Any]],
    owner_rows: Sequence[Mapping[str, Any]],
    prefix_owner_ids: Sequence[str],
    condition: str,
    budget_mode: str,
    image_id: str,
    schedule_metadata: Mapping[str, Any],
    shared_outcome: Mapping[str, Any] | None = None,
    shared_call_id: str | None = None,
) -> dict[str, Any]:
    remaining = int(schedule_metadata["remaining_depth"])
    row_budget = complete_row_budget(remaining, budget_mode)
    prefix_token_ids = [token_id for owner_id in prefix_owner_ids for token_id in row_map[owner_id]["row_token_ids"]]
    key = generation_call_key(prefix_token_ids, condition, row_budget)
    call_id = shared_call_id or generation_call_id(key)
    if shared_outcome is None:
        shared_outcome = _generate_shared_outcome(
            session=session,
            native_inputs=native_inputs,
            tokenizer=tokenizer,
            plan=plan,
            owner_rows=owner_rows,
            prefix_token_ids=prefix_token_ids,
            condition=condition,
            row_budget=row_budget,
            image_id=image_id,
            shared_call_id=call_id,
        )
    # Copy nested evidence before attaching schedule-local partition fields.
    from copy import deepcopy

    outcome = deepcopy(dict(shared_outcome))
    termination = str(outcome["termination"])
    matching = dict(outcome["owner_matching"])
    partition = partition_owner_matches(
        matching,
        remaining_owner_ids=schedule_metadata["remaining_owner_ids"],
        prefix_owner_ids=prefix_owner_ids,
    )
    partition["completion_flag"] = bool(partition["completion_flag"] and termination_is_valid(termination))
    matching = {**matching, **partition}
    return {
        **outcome,
        "condition": condition,
        "budget_mode": budget_mode,
        "row_budget": row_budget,
        "remaining_depth": remaining,
        "prefix_owner_ids": list(prefix_owner_ids),
        "prefix_token_ids": prefix_token_ids,
        "prefix_token_ids_sha256": _sha256_json(prefix_token_ids),
        "shared_call_id": call_id,
        "shared_call_key": {
            "prefix_token_ids_sha256": _sha256_json(prefix_token_ids),
            "condition": normalize_condition(condition),
            "row_budget": row_budget,
        },
        "schedule": dict(schedule_metadata),
        "owner_matching": matching,
        **partition,
    }


def _coerce_csv(value: str | Sequence[str] | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    values = [value] if isinstance(value, str) else list(value)
    result = tuple(part.strip() for raw in values for part in str(raw).split(",") if part.strip())
    return result or None


def run_grid(
    *,
    image_ids: Sequence[str] = DEFAULT_IMAGE_IDS,
    orders: Sequence[str] = DEFAULT_ORDERS,
    random_seeds: Sequence[int] = DEFAULT_RANDOM_SEEDS,
    remaining_depths: Sequence[int | str] | str | None = None,
    conditions: Sequence[str] = DEFAULT_CONDITIONS,
    budget_modes: Sequence[str] = DEFAULT_BUDGET_MODES,
    device: str = "cuda:0",
    output: Path,
    infer_config: Path = DEFAULT_INFER_CONFIG,
    force: bool = False,
) -> Path:
    """Execute the broad grid with one physical-batch-one session."""

    import torch
    from dataclasses import replace

    from src.config.fingerprint import sha256_json
    from src.config.inference import load_infer_config
    from src.data import load_raw_examples
    from src.inference.backend import open_backend_session
    from src.inference.runtime import assemble_frontend
    output = output.expanduser().resolve()
    if output.exists() and not force:
        raise ValueError(f"refusing to overwrite {output}; pass --force")
    selected_ids = tuple(str(item) for item in image_ids)
    selected_conditions = tuple(normalize_condition(str(item)) for item in conditions)
    selected_budgets = tuple(str(item).lower() for item in budget_modes)
    if any(item not in CONDITIONS for item in selected_conditions):
        raise ValueError(f"unsupported condition in {selected_conditions}")
    if any(item not in BUDGET_MODES for item in selected_budgets):
        raise ValueError(f"unsupported budget mode in {selected_budgets}")

    resolved = load_infer_config(infer_config.expanduser().resolve(strict=True))
    config = resolved.config
    raw_examples = list(load_raw_examples(config.data.input_jsonl))
    selected_examples = [_select_example(raw_examples, image_id) for image_id in selected_ids]
    # The frontend is processor-only.  The executable launch is explicitly
    # fp32, SDPA, and batch one for this unit even when the source config used a
    # wider/bfloat16 generation default.
    frontend = assemble_frontend(config, generation_config_fingerprint=sha256_json({"max_new_tokens": MAX_NEW_TOKENS, "repetition_penalty": REPETITION_PENALTY}))
    launch = replace(frontend.launch, model_dtype="fp32", batch_size=1)
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.set_device(torch.device(device))
    output.parent.mkdir(parents=True, exist_ok=True)
    images_artifacts: list[dict[str, Any]] = []

    with open_backend_session(launch) as session:
        for image_id, example in zip(selected_ids, selected_examples):
            request, plan, prompt_meta = _build_request(config, frontend, example)
            native_inputs, executed_ids, observed_grids, media_sha = session._materialize_native_inputs((request,))
            if tuple(executed_ids[0]) != request.expected_executed_prompt_token_ids:
                raise RuntimeError(f"base image prompt token parity failed for image {image_id}")
            rows = []
            for obj in example.objects:
                source = obj.metadata.get("source", {}) if isinstance(obj.metadata, Mapping) else {}
                rows.append(
                    {
                        "owner_id": str(source.get("coco_ann_id", obj.object_id)) if isinstance(source, Mapping) else str(obj.object_id),
                        "description": obj.description,
                        "category": obj.description,
                        "bbox": list(obj.bbox),
                    }
                )
            token_ids = tokenize_ground_truth_rows(rows, session._tokenizer)
            row_map = {str(row["owner_id"]): {**row, "row_token_ids": token_ids[str(row["owner_id"])]} for row in rows}
            policy = build_policy_schedules(rows, random_seeds=random_seeds, image_id=str(image_id), orders=orders)
            depths = build_remaining_depths(len(rows), remaining_depths)
            arms: list[dict[str, Any]] = []
            shared_outcomes: dict[tuple[tuple[int, ...], str, int], dict[str, Any]] = {}
            shared_call_receipts: dict[str, dict[str, Any]] = {}

            def append_arm(
                *,
                prefix_owner_ids: Sequence[str],
                condition: str,
                budget_mode: str,
                schedule_metadata: Mapping[str, Any],
            ) -> None:
                row_budget = complete_row_budget(int(schedule_metadata["remaining_depth"]), budget_mode)
                prefix_token_ids = tuple(
                    token_id
                    for owner_id in prefix_owner_ids
                    for token_id in row_map[str(owner_id)]["row_token_ids"]
                )
                key = generation_call_key(prefix_token_ids, condition, row_budget)
                call_id = generation_call_id(key)
                shared = shared_outcomes.get(key)
                if shared is None:
                    shared = _generate_shared_outcome(
                        session=session,
                        native_inputs=native_inputs,
                        tokenizer=session._tokenizer,
                        plan=plan,
                        owner_rows=rows,
                        prefix_token_ids=prefix_token_ids,
                        condition=condition,
                        row_budget=row_budget,
                        image_id=str(image_id),
                        shared_call_id=call_id,
                    )
                    shared_outcomes[key] = shared
                    shared_call_receipts[call_id] = {
                        "shared_call_id": call_id,
                        "prefix_token_ids_sha256": _sha256_json(prefix_token_ids),
                        "condition": normalize_condition(condition),
                        "row_budget": row_budget,
                    }
                arms.append(
                    _run_one_arm(
                        session=session,
                        native_inputs=native_inputs,
                        tokenizer=session._tokenizer,
                        plan=plan,
                        row_map=row_map,
                        owner_rows=rows,
                        prefix_owner_ids=prefix_owner_ids,
                        condition=condition,
                        budget_mode=budget_mode,
                        image_id=str(image_id),
                        schedule_metadata=schedule_metadata,
                        shared_outcome=shared,
                        shared_call_id=call_id,
                    )
                )

            for depth in depths:
                same = build_same_remaining_set_schedules(rows, depth, random_seeds=random_seeds, image_id=str(image_id), orders=orders)
                for schedule_name, schedule_ids in policy.items():
                    remaining_ids = schedule_ids[-depth:]
                    forced_ids = schedule_ids[:-depth]
                    schedule_metadata = {
                        "schedule_name": schedule_name,
                        "order": list(schedule_ids),
                        "forced_owner_ids": forced_ids,
                        "remaining_owner_ids": remaining_ids,
                        "remaining_depth": depth,
                        "same_remaining_owner_set": False,
                    }
                    for condition in selected_conditions:
                        for budget_mode in selected_budgets:
                            append_arm(prefix_owner_ids=forced_ids, condition=condition, budget_mode=budget_mode, schedule_metadata=schedule_metadata)
                for _control_name, control in same.items():
                    for condition in selected_conditions:
                        for budget_mode in selected_budgets:
                            append_arm(prefix_owner_ids=control["forced_owner_ids"], condition=condition, budget_mode=budget_mode, schedule_metadata=control)
            images_artifacts.append(
                {
                    "image_id": str(image_id),
                    "example_id": str(example.example_id),
                    "image": {"path": str(plan.image_path), "sha256": plan.image_content_sha256, "width": int(plan.decoded_width), "height": int(plan.decoded_height)},
                    "base_prompt": {**prompt_meta, "executed_prompt_token_ids_sha256": _sha256_json(executed_ids[0])},
                    "observed_image_grid_thw": None if observed_grids[0] is None else list(observed_grids[0]),
                    "executed_media_sha256": media_sha[0],
                    "owner_rows": rows,
                    "shared_generation_call_count": len(shared_call_receipts),
                    "shared_generation_calls": list(shared_call_receipts.values()),
                    "arms": arms,
                }
            )
        model_receipt = session.receipt.to_artifact_dict()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "human_refined_greedy_set_completion_conditions",
        "protocol_unit_id": "2026-07-22-human-refined-greedy-set-completion-conditions",
        "config": {
            "infer_config": str(infer_config.expanduser().resolve()),
            "resolved_fingerprint": resolved.fingerprint,
            "device": str(device),
            "batch_size": 1,
            "model_dtype": "fp32",
            "attn_implementation": "sdpa",
            "max_new_tokens": MAX_NEW_TOKENS,
            "repetition_penalty": REPETITION_PENALTY,
            "image_ids": list(selected_ids),
            "orders": list(orders),
            "random_seeds": [int(seed) for seed in random_seeds],
            "remaining_depths": list(remaining_depths) if remaining_depths is not None and not isinstance(remaining_depths, str) else remaining_depths,
            "conditions": list(selected_conditions),
            "budget_modes": list(selected_budgets),
        },
        "model_session_receipt": model_receipt,
        "images": images_artifacts,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-ids", default=",".join(DEFAULT_IMAGE_IDS))
    parser.add_argument("--orders", default=",".join(DEFAULT_ORDERS))
    parser.add_argument("--random-seeds", default=",".join(str(value) for value in DEFAULT_RANDOM_SEEDS))
    parser.add_argument("--remaining-depths", default=",".join(DEFAULT_REMAINING_DEPTHS))
    parser.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS))
    parser.add_argument("--budget-modes", default=",".join(DEFAULT_BUDGET_MODES))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--infer-config", type=Path, default=DEFAULT_INFER_CONFIG)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.image_ids = _coerce_csv(args.image_ids) or DEFAULT_IMAGE_IDS
    args.orders = _coerce_csv(args.orders) or DEFAULT_ORDERS
    args.random_seeds = tuple(int(value) for value in (_coerce_csv(args.random_seeds) or ()))
    args.remaining_depths = _coerce_csv(args.remaining_depths)
    args.conditions = _coerce_csv(args.conditions) or DEFAULT_CONDITIONS
    args.budget_modes = _coerce_csv(args.budget_modes) or DEFAULT_BUDGET_MODES
    return args


def main() -> int:
    args = _parse_args()
    run_grid(
        image_ids=args.image_ids,
        orders=args.orders,
        random_seeds=args.random_seeds,
        remaining_depths=args.remaining_depths,
        conditions=args.conditions,
        budget_modes=args.budget_modes,
        device=args.device,
        output=args.output.expanduser().resolve(),
        infer_config=args.infer_config,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
