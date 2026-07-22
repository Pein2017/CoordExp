#!/usr/bin/env python3
"""Run a bounded, manifest-driven one-row completion causal micro-panel.

The runner is deliberately experiment-local.  It reuses the loaded Hugging
Face session, prompt construction, compact-row parser, matching, and exact
row-append helpers already exercised by the human-refined completion runners.
No model backend is implemented here: the executable path only assembles one
FP32/SDPA/batch-one session and calls the existing private HF generation seams.

Rows supplied by a case manifest are context.  Only rows emitted after the
branch row are treated as released discovery.  The forced branch is retained
as a separate artifact so a later reader cannot accidentally credit the
intervention itself.

The branch kinds are explicit intervention boundaries: ``description_gt_owner``
forces the trusted description through ``object_ref_end``;
``first_coordinate_gt_owner``, ``first_two_coordinates_gt_owner``, and
``first_three_coordinates_gt_owner`` additionally force ``box_start`` and the
first one, two, or three trusted coordinate tokens.  Each partial branch then
releases the remaining tokens of that row and the bounded successor-row
horizon through the existing forced-partial generation helper.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import copy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


from scripts.research.run_human_refined_greedy_set_completion_conditions import (  # noqa: E402
    DEFAULT_INFER_CONFIG,
    BOX_END_TOKEN_ID,
    BOX_START_TOKEN_ID,
    COORDINATE_TOKEN_MAX_ID,
    COORDINATE_TOKEN_MIN_ID,
    MAX_NEW_TOKENS,
    OBJECT_REF_END_TOKEN_ID,
    OBJECT_REF_START_TOKEN_ID,
    _parse_generated_text,
    _sha256_json,
    match_predictions_one_to_one,
    tokenize_ground_truth_rows,
)
from scripts.research.run_local_branch_causal_value import (  # noqa: E402
    _generate_after_forced_partial_row,
    _generate_row,
    append_row_if_complete,
)


SCHEMA_VERSION = "human_refined_completion_causal_micro_panel.v1"
MANIFEST_SCHEMA_VERSION = SCHEMA_VERSION
REPETITION_PENALTY = 1.0
MAX_RELEASED_ROW_BUDGET = 8
PARTIAL_GT_BRANCH_COORDINATE_COUNTS = {
    "description_gt_owner": 0,
    "first_coordinate_gt_owner": 1,
    "first_two_coordinates_gt_owner": 2,
    "first_three_coordinates_gt_owner": 3,
}
BRANCH_KIND_OPERATIONAL_MEANINGS = {
    "complete_gt_owner": (
        "Force the trusted owner's complete ground-truth row before releasing "
        "the bounded successor-row horizon."
    ),
    "description_gt_owner": (
        "Force the trusted owner's description through object_ref_end, then "
        "release its geometry and the bounded successor-row horizon."
    ),
    "first_coordinate_gt_owner": (
        "Force the trusted owner's description, box_start, and first coordinate "
        "x1, then release the remaining row tokens and successor rows."
    ),
    "first_two_coordinates_gt_owner": (
        "Force the trusted owner's description, box_start, and first two "
        "coordinates x1,y1, then release the remaining row tokens and successor rows."
    ),
    "first_three_coordinates_gt_owner": (
        "Force the trusted owner's description, box_start, and first three "
        "coordinates x1,y1,x2, then release the remaining row tokens and successor rows."
    ),
    "covered_gt_owner": (
        "Repeat a trusted owner already present in the parent prefix as a "
        "negative-control branch before releasing successor rows."
    ),
}
BRANCH_KINDS = frozenset(
    {
        "complete_gt_owner",
        "covered_gt_owner",
        *PARTIAL_GT_BRANCH_COORDINATE_COUNTS,
    }
)
_WRAPPER_TOKENS = {
    "object_ref_start": "<|object_ref_start|>",
    "object_ref_end": "<|object_ref_end|>",
}


def _ids(value: Any, *, label: str) -> list[int]:
    """Normalize a token-id sequence without accepting booleans or strings."""

    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be a token-id sequence")
    result: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ValueError(f"{label} must contain non-negative integer token ids")
        result.append(int(item))
    return result


def _owner_id(value: Any, *, label: str = "owner_id") -> str:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{label} must be a non-empty string")
    result = str(value).strip()
    if not result:
        raise ValueError(f"{label} must be a non-empty string")
    return result


def _case_id(value: Any) -> str:
    return _owner_id(value, label="case_id")


def validate_case_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize the minimal v1 panel manifest.

    Validation intentionally does not inspect the model or image data.  Owner
    existence is checked immediately after each image's trusted GT rows are
    loaded, before any model call.  Keeping this function pure makes malformed
    manifests cheap to reject in review and tests.
    """

    if not isinstance(manifest, Mapping):
        raise ValueError("case manifest must be a JSON object")
    schema = manifest.get("schema_version", manifest.get("version"))
    if schema not in {SCHEMA_VERSION, 1, "1", "v1"}:
        raise ValueError(
            f"case manifest schema_version must be {SCHEMA_VERSION!r} (got {schema!r})"
        )
    raw_cases = manifest.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("case manifest requires a non-empty cases list")

    cases: list[dict[str, Any]] = []
    seen_case_ids: set[str] = set()
    for case_index, raw_case in enumerate(raw_cases):
        if not isinstance(raw_case, Mapping):
            raise ValueError(f"case {case_index} must be an object")
        case = dict(raw_case)
        case_id = _case_id(case.get("case_id"))
        if case_id in seen_case_ids:
            raise ValueError(f"duplicate case_id {case_id!r}")
        seen_case_ids.add(case_id)
        image_id = _owner_id(case.get("image_id"), label=f"case {case_id} image_id")

        raw_parent = case.get("parent_prefix_owner_ids")
        if isinstance(raw_parent, (str, bytes)) or not isinstance(raw_parent, list):
            raise ValueError(
                f"case {case_id} parent_prefix_owner_ids must be a list"
            )
        parent_ids = [_owner_id(value, label="parent_prefix_owner_ids item") for value in raw_parent]
        if len(parent_ids) != len(set(parent_ids)):
            raise ValueError(f"case {case_id} parent_prefix_owner_ids contain duplicates")

        budget = case.get("released_row_budget")
        if isinstance(budget, bool) or not isinstance(budget, int):
            raise ValueError(f"case {case_id} released_row_budget must be an integer")
        if not 1 <= int(budget) <= MAX_RELEASED_ROW_BUDGET:
            raise ValueError(
                f"case {case_id} released_row_budget must lie in [1,{MAX_RELEASED_ROW_BUDGET}]"
            )

        raw_arms = case.get("branch_arms")
        if not isinstance(raw_arms, list) or not raw_arms:
            raise ValueError(f"case {case_id} requires a non-empty branch_arms list")
        arms: list[dict[str, Any]] = []
        seen_arm_keys: set[tuple[str, str]] = set()
        for arm_index, raw_arm in enumerate(raw_arms):
            if not isinstance(raw_arm, Mapping):
                raise ValueError(f"case {case_id} arm {arm_index} must be an object")
            arm = dict(raw_arm)
            kind = str(arm.get("kind", "")).strip().lower()
            if kind not in BRANCH_KINDS:
                raise ValueError(
                    f"case {case_id} arm {arm_index} has unsupported kind {kind!r}"
                )
            owner = _owner_id(
                arm.get("owner_id"), label=f"case {case_id} arm {arm_index} owner_id"
            )
            key = (kind, owner)
            if key in seen_arm_keys:
                raise ValueError(
                    f"case {case_id} has duplicate branch arm {kind}:{owner}"
                )
            seen_arm_keys.add(key)
            arm["kind"] = kind
            arm["owner_id"] = owner
            arm["operational_meaning"] = str(
                arm.get("operational_meaning")
                or BRANCH_KIND_OPERATIONAL_MEANINGS[kind]
            ).strip()
            if not arm["operational_meaning"]:
                raise ValueError(
                    f"case {case_id} arm {arm_index} operational_meaning must be non-empty"
                )
            arms.append(arm)

        case["case_id"] = case_id
        case["image_id"] = image_id
        case["parent_prefix_owner_ids"] = parent_ids
        case["released_row_budget"] = int(budget)
        case["branch_arms"] = arms
        cases.append(case)

    normalized = dict(manifest)
    normalized["schema_version"] = SCHEMA_VERSION
    normalized["cases"] = cases
    return normalized


def load_case_manifest(path: Path) -> dict[str, Any]:
    """Load, parse, and validate one manifest before runtime imports."""

    path = path.expanduser().resolve(strict=True)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in case manifest {path}: {exc}") from exc
    return validate_case_manifest(value)


def _encoded_input_ids(encoded: Any, *, label: str) -> list[int]:
    if isinstance(encoded, Mapping):
        value = encoded.get("input_ids")
    else:
        value = getattr(encoded, "input_ids", None)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) == 1 and isinstance(value[0], Sequence) and not isinstance(value[0], (str, bytes)):
            value = value[0]
    return _ids(value, label=label)


def _tokenizer_wrapper_id(tokenizer: Any, token: str, fallback: int) -> int:
    converter = getattr(tokenizer, "convert_tokens_to_ids", None)
    if callable(converter):
        value = converter(token)
        if value is not None:
            return int(value)
    # The fallback is only for the current Qwen3-VL tokenizer.  A loaded HF
    # session still verifies the actual tokenizer ids before generation.
    return int(fallback)


def _description_from_owner_row(owner_row: Mapping[str, Any] | str) -> str:
    if isinstance(owner_row, Mapping):
        description = str(
            owner_row.get(
                "description", owner_row.get("desc", owner_row.get("category", ""))
            )
        ).strip()
        owner_label = _owner_id(owner_row.get("owner_id"), label="owner_id")
    else:
        description = str(owner_row).strip()
        owner_label = "description"
    if not description:
        raise ValueError(f"owner {owner_label!r} has an empty description")
    return description


def build_description_prefix_token_ids(
    owner_row: Mapping[str, Any] | str, tokenizer: Any
) -> list[int]:
    """Construct ``object_ref_start + GT description ids + object_ref_end``.

    The description is tokenized exactly as supplied by the trusted GT row;
    no decoded text from a prior rollout is reused.  Wrapper ids come from the
    active tokenizer whenever available, with the current Qwen constants as a
    compatibility fallback for pure helper callers.
    """

    description = _description_from_owner_row(owner_row)
    try:
        encoded = tokenizer(description, add_special_tokens=False)
    except TypeError:
        encoder = getattr(tokenizer, "encode", None)
        if not callable(encoder):
            raise
        try:
            encoded = encoder(description, add_special_tokens=False)
        except TypeError:
            encoded = encoder(description)
    description_ids = _encoded_input_ids(encoded, label="description token ids")
    if not description_ids:
        raise ValueError("ground-truth description tokenization returned no ids")
    start = _tokenizer_wrapper_id(
        tokenizer, _WRAPPER_TOKENS["object_ref_start"], OBJECT_REF_START_TOKEN_ID
    )
    end = _tokenizer_wrapper_id(
        tokenizer, _WRAPPER_TOKENS["object_ref_end"], OBJECT_REF_END_TOKEN_ID
    )
    return [start, *description_ids, end]


def _validated_complete_row_layout(
    token_ids: Sequence[int], *, owner_id: str = "owner"
) -> dict[str, Any]:
    """Return exact structural offsets for one canonical complete GT row.

    The row ids have already been produced by the active tokenizer.  This
    validator deliberately works only on those ids: decoding to text and
    searching for ``<|box_start|>`` would permit token-boundary drift and could
    select a marker from a malformed or concatenated row.
    """

    from scripts.research.run_human_refined_greedy_set_completion_conditions import (
        complete_row_token_summary,
    )

    ids = _ids(token_ids, label=f"complete row token ids for {owner_id}")
    count, last_end = complete_row_token_summary(
        ids,
        object_ref_start_token_id=OBJECT_REF_START_TOKEN_ID,
        object_ref_end_token_id=OBJECT_REF_END_TOKEN_ID,
        box_start_token_id=BOX_START_TOKEN_ID,
        box_end_token_id=BOX_END_TOKEN_ID,
        coordinate_token_min_id=COORDINATE_TOKEN_MIN_ID,
        coordinate_token_max_id=COORDINATE_TOKEN_MAX_ID,
    )
    if count != 1 or last_end != len(ids) - 1:
        raise ValueError(
            f"owner {owner_id!r} has invalid complete-row token ids "
            f"(count={count}, last_end={last_end}, length={len(ids)})"
        )

    if not ids or ids[0] != OBJECT_REF_START_TOKEN_ID:
        raise ValueError(
            f"owner {owner_id!r} complete row must start with object_ref_start"
        )
    object_end_indices = [
        index for index, value in enumerate(ids) if value == OBJECT_REF_END_TOKEN_ID
    ]
    if len(object_end_indices) != 1:
        raise ValueError(
            f"owner {owner_id!r} complete row must contain one object_ref_end"
        )
    object_end = object_end_indices[0]
    if object_end <= 1:
        raise ValueError(
            f"owner {owner_id!r} complete row must contain a description token"
        )
    description_ids = ids[1:object_end]
    structural_ids = {
        OBJECT_REF_START_TOKEN_ID,
        OBJECT_REF_END_TOKEN_ID,
        BOX_START_TOKEN_ID,
        BOX_END_TOKEN_ID,
    }
    if any(
        value in structural_ids
        or COORDINATE_TOKEN_MIN_ID <= value <= COORDINATE_TOKEN_MAX_ID
        for value in description_ids
    ):
        raise ValueError(
            f"owner {owner_id!r} complete row has structural tokens in description"
        )

    box_start_indices = [
        index for index, value in enumerate(ids) if value == BOX_START_TOKEN_ID
    ]
    if len(box_start_indices) != 1:
        raise ValueError(
            f"owner {owner_id!r} complete row must contain one box_start"
        )
    box_start = box_start_indices[0]
    if box_start != object_end + 1:
        raise ValueError(
            f"owner {owner_id!r} complete row must place box_start after object_ref_end"
        )

    box_end_indices = [
        index for index, value in enumerate(ids) if value == BOX_END_TOKEN_ID
    ]
    if len(box_end_indices) != 1 or box_end_indices[0] != len(ids) - 1:
        raise ValueError(
            f"owner {owner_id!r} complete row must end with one box_end"
        )
    box_end = box_end_indices[0]
    coordinate_ids = ids[box_start + 1 : box_end]
    if len(coordinate_ids) != 4 or any(
        not COORDINATE_TOKEN_MIN_ID <= value <= COORDINATE_TOKEN_MAX_ID
        for value in coordinate_ids
    ):
        raise ValueError(
            f"owner {owner_id!r} complete row must contain four coordinate tokens"
        )
    return {
        "token_ids": ids,
        "object_ref_start": 0,
        "object_ref_end": object_end,
        "box_start": box_start,
        "coordinate_start": box_start + 1,
        "coordinate_end": box_end,
        "coordinate_ids": coordinate_ids,
        "box_end": box_end,
    }


def validate_complete_row_token_ids(
    token_ids: Sequence[int], *, owner_id: str = "owner"
) -> list[int]:
    """Fail fast unless ids contain exactly one canonical complete compact row."""

    return list(_validated_complete_row_layout(token_ids, owner_id=owner_id)["token_ids"])


def build_partial_gt_owner_prefix_token_ids(
    token_ids: Sequence[int],
    *,
    kind: str,
    owner_id: str = "owner",
) -> list[int]:
    """Slice a trusted complete row at an explicit description/geometry boundary.

    ``token_ids`` must be the exact ids returned by
    :func:`tokenize_ground_truth_rows`, already validated as one complete row.
    For coordinate branches the returned prefix includes ``object_ref_start``,
    the complete description, ``box_start``, and exactly the first one, two, or
    three coordinate ids.  The helper never decodes text or reconstructs a row
    from numeric box values.
    """

    normalized_kind = str(kind).strip().lower()
    if normalized_kind not in PARTIAL_GT_BRANCH_COORDINATE_COUNTS:
        raise ValueError(f"unsupported partial GT branch kind {kind!r}")
    layout = _validated_complete_row_layout(token_ids, owner_id=owner_id)
    coordinate_count = PARTIAL_GT_BRANCH_COORDINATE_COUNTS[normalized_kind]
    ids = layout["token_ids"]
    box_start = int(layout["box_start"])
    if coordinate_count == 0:
        end = box_start
    else:
        end = box_start + 1 + coordinate_count
    prefix = list(ids[:end])
    expected_minimum = (
        box_start
        if coordinate_count == 0
        else int(layout["coordinate_start"]) + coordinate_count
    )
    if len(prefix) != expected_minimum:
        raise ValueError(
            f"owner {owner_id!r} {normalized_kind} prefix has unexpected length"
        )
    if coordinate_count and prefix[box_start] != BOX_START_TOKEN_ID:
        raise ValueError(
            f"owner {owner_id!r} {normalized_kind} prefix lost box_start"
        )
    if coordinate_count and any(
        not COORDINATE_TOKEN_MIN_ID <= value <= COORDINATE_TOKEN_MAX_ID
        for value in prefix[box_start + 1 :]
    ):
        raise ValueError(
            f"owner {owner_id!r} {normalized_kind} prefix has malformed coordinates"
        )
    return prefix


def _as_owner_set(values: Sequence[Any] | None) -> set[str]:
    if values is None:
        return set()
    return {_owner_id(value) for value in values}


def partition_branch_suffix_owner_matches(
    branch_owner_ids: Sequence[Any],
    suffix_owner_ids: Sequence[Any],
    parent_owner_ids: Sequence[Any],
    remaining_owner_ids: Sequence[Any] | None = None,
    declared_forced_owner_ids: Sequence[Any] = (),
) -> dict[str, Any]:
    """Separate branch credit from released suffix discovery.

    ``remaining_owner_ids`` should be all trusted owners not supplied in the
    parent prefix.  If omitted, it is derived from observed branch/suffix
    owners, which is useful for a pure bookkeeping check but is not sufficient
    for a scientific completion claim.
    """

    branch = _as_owner_set(branch_owner_ids)
    suffix = _as_owner_set(suffix_owner_ids)
    parent = _as_owner_set(parent_owner_ids)
    declared_forced = _as_owner_set(declared_forced_owner_ids)
    if remaining_owner_ids is None:
        remaining = (branch | suffix) - parent
    else:
        remaining = _as_owner_set(remaining_owner_ids)
    # A declared forced owner is context even when the forced-description row
    # fails the conservative geometry match.  Excluding only the branch's
    # matched owners would let a later repeat of the intervention target pose
    # as downstream discovery.
    released_only = suffix - parent - branch - declared_forced
    total_new = (branch | (suffix - declared_forced)) - parent
    branch_remaining = branch & remaining
    forced_repeats = (branch | suffix) & parent
    released_forced_repeats = suffix & parent
    completion = bool(remaining) and total_new >= remaining
    released_completion = bool(remaining) and released_only >= remaining
    return {
        "branch_owner_credits": sorted(branch),
        "released_suffix_owner_ids": sorted(suffix),
        "declared_forced_owner_ids": sorted(declared_forced),
        "released_declared_forced_owner_revisits": sorted(suffix & declared_forced),
        "released_only_remaining_owner_ids": sorted(released_only & remaining),
        "branch_remaining_owner_ids": sorted(branch_remaining),
        "total_new_remaining_owner_ids_including_branch": sorted(total_new & remaining),
        "forced_context_repeats": sorted(forced_repeats),
        "released_forced_context_repeats": sorted(released_forced_repeats),
        "remaining_owner_ids": sorted(remaining),
        "completion_against_remaining_owner_set": completion,
        "released_only_completion_against_remaining_owner_set": released_completion,
    }


# Short aliases make the pure bookkeeping contract easy to discover without
# introducing a second implementation surface.
partition_branch_suffix_owner_ids = partition_branch_suffix_owner_matches


def compare_raw_noop_parity(
    natural_suffix_rows: Sequence[Mapping[str, Any]],
    noop_suffix_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Require byte-identical raw token ids for every released successor row."""

    natural = [_ids(row.get("raw_generated_token_ids", []), label="natural suffix raw ids") for row in natural_suffix_rows]
    noop = [_ids(row.get("raw_generated_token_ids", []), label="no-op suffix raw ids") for row in noop_suffix_rows]
    checks: list[dict[str, Any]] = []
    for index in range(max(len(natural), len(noop))):
        left = natural[index] if index < len(natural) else None
        right = noop[index] if index < len(noop) else None
        checks.append(
            {
                "row_index": index,
                "passed": left == right,
                "natural_raw_token_ids": left,
                "noop_raw_token_ids": right,
            }
        )
    passed = natural == noop
    return {
        "passed": passed,
        "raw_token_ids_equal": passed,
        "natural_suffix_row_count": len(natural),
        "noop_suffix_row_count": len(noop),
        "row_checks": checks,
        "first_failure_row_index": next(
            (int(item["row_index"]) for item in checks if not item["passed"]), None
        ),
    }


compare_noop_parity = compare_raw_noop_parity


def causal_admissibility_from_noop(noop: Mapping[str, Any]) -> tuple[bool, str | None]:
    """Accept a causal case only after an exact native-row no-op replay."""

    status = str(noop.get("status", "missing"))
    if status == "valid":
        return True, None
    if status == "inadmissible":
        return False, "native_complete_row_noop_raw_token_parity_failed"
    return False, f"native_complete_row_noop_{status}"


def _owner_rows_from_example(example: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for obj in example.objects:
        metadata = obj.metadata if isinstance(getattr(obj, "metadata", None), Mapping) else {}
        source = metadata.get("source", {}) if isinstance(metadata, Mapping) else {}
        owner_id = (
            source.get("coco_ann_id", obj.object_id)
            if isinstance(source, Mapping)
            else obj.object_id
        )
        rows.append(
            {
                "owner_id": str(owner_id),
                "description": str(obj.description),
                "category": str(obj.description),
                "bbox": list(obj.bbox),
            }
        )
    if not rows:
        raise ValueError(f"image {example.example_id!r} has no trusted owner rows")
    return rows


def _raw_reparse_and_match(
    row: Mapping[str, Any],
    *,
    tokenizer: Any,
    image_width: int,
    image_height: int,
    owner_rows: Sequence[Mapping[str, Any]],
    row_id: str,
) -> dict[str, Any]:
    """Reparse raw ids through the current parser, then globally match bins."""

    result = copy.deepcopy(dict(row))
    raw_ids = _ids(result.get("raw_generated_token_ids", []), label="raw_generated_token_ids")
    raw_text = tokenizer.decode(raw_ids, skip_special_tokens=False)
    parsed = _parse_generated_text(
        raw_text,
        image_width=int(image_width),
        image_height=int(image_height),
        row_id=row_id,
    )
    matching = match_predictions_one_to_one(parsed["predictions"], owner_rows)
    stop = result.get("row_stop") if isinstance(result.get("row_stop"), Mapping) else {}
    reached_token_limit = len(raw_ids) >= MAX_NEW_TOKENS
    token_limit_invalid = bool(
        reached_token_limit and stop.get("stop_reason") != "complete_row"
    )
    # Keep both the helper's original parser receipt and the authoritative
    # raw-id reparse.  Consumers should use ``parse_evidence`` below.
    if "parse_evidence" in result:
        result["generation_helper_parse_evidence"] = result["parse_evidence"]
    result.update(
        {
            "raw_generated_token_ids": raw_ids,
            "raw_generated_token_ids_sha256": _sha256_json(raw_ids),
            "raw_generated_text": raw_text,
            "raw_reparse": parsed,
            "parse_evidence": parsed["parser_artifact"],
            "parser_receipt": parsed["parser_artifact"],
            "parsed_predictions": parsed["predictions"],
            "dropped_predictions": parsed["dropped_predictions"],
            "dropped_prediction_count": int(parsed["dropped_prediction_count"]),
            "parse_status": parsed["parse_status"],
            "owner_matching": matching,
            "malformed_evidence": {
                "parse_status": parsed["parse_status"],
                "dropped_prediction_count": int(parsed["dropped_prediction_count"]),
                "dropped_predictions": parsed["dropped_predictions"],
                "row_stop": result.get("row_stop"),
            },
            "token_limit_evidence": {
                "max_new_tokens": MAX_NEW_TOKENS,
                "generated_token_count": len(raw_ids),
                "reached_token_limit": reached_token_limit,
                "invalid_evidence": token_limit_invalid,
            },
            "token_limit_invalid_evidence": token_limit_invalid,
        }
    )
    return result


def _matched_owner_ids(row: Mapping[str, Any]) -> list[str]:
    matching = row.get("owner_matching")
    if isinstance(matching, Mapping):
        values = matching.get("matched_owner_ids", ())
    else:
        values = row.get("strict_matched_owner_ids", ())
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        return []
    return sorted({_owner_id(value) for value in values})


def _row_is_complete(row: Mapping[str, Any]) -> bool:
    stop = row.get("row_stop") if isinstance(row.get("row_stop"), Mapping) else {}
    if row.get("status") != "success" or stop.get("stop_reason") != "complete_row":
        return False
    from scripts.research.run_human_refined_greedy_set_completion_conditions import (
        complete_row_token_summary,
    )

    ids = _ids(row.get("raw_generated_token_ids", []), label="row raw ids")
    count, end = complete_row_token_summary(ids)
    return count == 1 and end == len(ids) - 1


def _forced_complete_row(
    token_ids: Sequence[int],
    *,
    owner_id: str,
    kind: str,
    tokenizer: Any,
    image_width: int,
    image_height: int,
    owner_rows: Sequence[Mapping[str, Any]],
    row_id: str,
) -> dict[str, Any]:
    ids = validate_complete_row_token_ids(token_ids, owner_id=owner_id)
    row = {
        "mode": "forced_complete_gt",
        "status": "success",
        "row_stop": {"stop_reason": "complete_row", "row_text": None},
        "raw_generated_token_ids": ids,
        "forced_row_token_ids": list(ids),
        "forced_row_token_ids_sha256": _sha256_json(ids),
        "released_tail_token_ids": [],
        "branch_kind": kind,
        "forced_owner_id": owner_id,
    }
    return _raw_reparse_and_match(
        row,
        tokenizer=tokenizer,
        image_width=image_width,
        image_height=image_height,
        owner_rows=owner_rows,
        row_id=row_id,
    )


def _annotate_generated_row(
    row: Mapping[str, Any],
    *,
    tokenizer: Any,
    image_width: int,
    image_height: int,
    owner_rows: Sequence[Mapping[str, Any]],
    row_id: str,
) -> dict[str, Any]:
    return _raw_reparse_and_match(
        row,
        tokenizer=tokenizer,
        image_width=image_width,
        image_height=image_height,
        owner_rows=owner_rows,
        row_id=row_id,
    )


def _generate_successor_rows(
    *,
    session: Any,
    native_inputs: Mapping[str, Any],
    tokenizer: Any,
    prefix_token_ids: Sequence[int],
    released_row_budget: int,
    image_width: int,
    image_height: int,
    owner_rows: Sequence[Mapping[str, Any]],
    image_id: str,
    row_index_start: int,
) -> tuple[list[dict[str, Any]], list[int]]:
    """Generate up to the bounded successor budget by exact-token appends."""

    current = _ids(prefix_token_ids, label="successor prefix token ids")
    rows: list[dict[str, Any]] = []
    for offset in range(int(released_row_budget)):
        row_index = int(row_index_start) + offset
        generated = _generate_row(
            session=session,
            native_inputs=native_inputs,
            prefix_token_ids=current,
            tokenizer=tokenizer,
            image_width=int(image_width),
            image_height=int(image_height),
            mode="greedy",
            seed=None,
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=REPETITION_PENALTY,
            max_new_tokens=MAX_NEW_TOKENS,
            malformed_limit=2,
            row_index=row_index,
        )
        generated = _annotate_generated_row(
            generated,
            tokenizer=tokenizer,
            image_width=image_width,
            image_height=image_height,
            owner_rows=owner_rows,
            row_id=f"{image_id}:released:{row_index}",
        )
        generated["row_index"] = row_index
        generated["input_prefix_token_ids"] = list(current)
        current, append_receipt = append_row_if_complete(current, generated)
        generated["append_receipt"] = append_receipt
        generated["accepted_complete_row"] = bool(append_receipt.get("appended"))
        rows.append(generated)
        if not generated["accepted_complete_row"]:
            break
    return rows, current


def _branch_and_suffix_partition(
    branch_row: Mapping[str, Any],
    suffix_rows: Sequence[Mapping[str, Any]],
    *,
    owner_rows: Sequence[Mapping[str, Any]],
    parent_owner_ids: Sequence[str],
    remaining_owner_ids: Sequence[str],
    declared_forced_owner_ids: Sequence[str] = (),
) -> dict[str, Any]:
    branch_predictions = list(branch_row.get("parsed_predictions", ()))
    suffix_predictions = [
        prediction
        for row in suffix_rows
        for prediction in row.get("parsed_predictions", ())
    ]
    global_matching = match_predictions_one_to_one(
        [*branch_predictions, *suffix_predictions], owner_rows
    )
    branch_prediction_ids = {
        str(prediction.get("prediction_id")) for prediction in branch_predictions
    }
    suffix_prediction_ids = {
        str(prediction.get("prediction_id")) for prediction in suffix_predictions
    }
    branch_ids = sorted(
        {
            str(match["owner_id"])
            for match in global_matching["matches"]
            if str(match["prediction_id"]) in branch_prediction_ids
        }
    )
    suffix_ids = sorted(
        {
            str(match["owner_id"])
            for match in global_matching["matches"]
            if str(match["prediction_id"]) in suffix_prediction_ids
        }
    )
    result = partition_branch_suffix_owner_matches(
        branch_ids,
        suffix_ids,
        parent_owner_ids,
        remaining_owner_ids,
        declared_forced_owner_ids,
    )
    result["global_branch_suffix_owner_matching"] = global_matching
    return result


def _arm_result(
    *,
    arm: Mapping[str, Any],
    branch_row: Mapping[str, Any],
    suffix_rows: Sequence[Mapping[str, Any]],
    owner_rows: Sequence[Mapping[str, Any]],
    parent_owner_ids: Sequence[str],
    remaining_owner_ids: Sequence[str],
    branch_prefix_token_ids: Sequence[int],
    branch_released_tail_token_ids: Sequence[int],
) -> dict[str, Any]:
    branch_copy = copy.deepcopy(dict(branch_row))
    suffix_copy = copy.deepcopy(list(suffix_rows))
    partition = _branch_and_suffix_partition(
        branch_copy,
        suffix_copy,
        owner_rows=owner_rows,
        parent_owner_ids=parent_owner_ids,
        remaining_owner_ids=remaining_owner_ids,
        declared_forced_owner_ids=[str(arm["owner_id"])],
    )
    return {
        "kind": str(arm["kind"]),
        "owner_id": str(arm["owner_id"]),
        "operational_meaning": str(
            arm.get("operational_meaning")
            or BRANCH_KIND_OPERATIONAL_MEANINGS[str(arm["kind"])]
        ),
        "forced_prefix_coordinate_count": (
            PARTIAL_GT_BRANCH_COORDINATE_COUNTS.get(str(arm["kind"]))
        ),
        "branch_row": branch_copy,
        "released_suffix": suffix_copy,
        "branch_owner_credits": partition["branch_owner_credits"],
        "released_only_remaining_owner_ids": partition["released_only_remaining_owner_ids"],
        "total_new_remaining_owner_ids_including_branch": partition[
            "total_new_remaining_owner_ids_including_branch"
        ],
        "forced_context_repeats": partition["forced_context_repeats"],
        "completion_against_remaining_owner_set": partition[
            "completion_against_remaining_owner_set"
        ],
        "owner_partition": partition,
        "branch_prefix_token_ids": list(map(int, branch_prefix_token_ids)),
        "branch_prefix_token_ids_sha256": _sha256_json(branch_prefix_token_ids),
        "branch_released_tail_token_ids": list(map(int, branch_released_tail_token_ids)),
        "branch_released_tail_token_ids_sha256": _sha256_json(branch_released_tail_token_ids),
        # A branch row is always context, regardless of whether it was exact
        # GT, description-forced, or native.  This explicit field is a guard
        # against accidentally treating it as released discovery.
        "forced_branch_excluded_from_released_discovery": True,
    }


def _session_launch_fp32_sdpa_batch1(frontend: Any) -> Any:
    from dataclasses import replace

    launch = frontend.launch
    if str(getattr(launch, "backend", "")) != "hf":
        raise ValueError("causal micro-panel requires the Hugging Face backend")
    backend_options = dict(getattr(launch, "backend_options", {}) or {})
    hf_options = dict(backend_options.get("hf", {}) or {})
    hf_options["attn_implementation"] = "sdpa"
    backend_options["hf"] = hf_options
    return replace(
        launch,
        model_dtype="fp32",
        batch_size=1,
        backend_options=backend_options,
    )


def _run_one_case(
    *,
    case: Mapping[str, Any],
    config: Any,
    frontend: Any,
    session: Any,
    raw_examples: Sequence[Any],
    image_id: str,
) -> dict[str, Any]:
    """Execute one validated case inside an already-open HF session."""

    from scripts.research.run_human_refined_greedy_set_completion_conditions import (
        _build_request,
        _select_example,
    )
    from scripts.research.run_local_branch_causal_value import _single_native_inputs

    example = _select_example(raw_examples, image_id)
    request, plan, prompt_meta = _build_request(config, frontend, example)
    native_inputs, executed_ids, observed_grids, media_sha = session._materialize_native_inputs((request,))
    if tuple(map(int, executed_ids[0])) != tuple(request.expected_executed_prompt_token_ids):
        raise ValueError(f"base image prompt token parity failed for case {case['case_id']}")
    native_inputs = _single_native_inputs(native_inputs)
    owner_rows = _owner_rows_from_example(example)
    owner_map = {str(row["owner_id"]): row for row in owner_rows}
    if len(owner_map) != len(owner_rows):
        raise ValueError(f"image {image_id} has duplicate trusted owner ids")
    all_owner_ids = list(owner_map)
    parent_ids = [str(value) for value in case["parent_prefix_owner_ids"]]
    unknown_parent = sorted(set(parent_ids) - set(owner_map))
    if unknown_parent:
        raise ValueError(
            f"case {case['case_id']} names unknown parent owner ids: {unknown_parent}"
        )
    remaining_ids = [owner_id for owner_id in all_owner_ids if owner_id not in set(parent_ids)]
    if not remaining_ids:
        raise ValueError(f"case {case['case_id']} has no remaining trusted owners")
    row_token_map = tokenize_ground_truth_rows(owner_rows, session._tokenizer)
    for owner_id, token_ids in row_token_map.items():
        row_token_map[owner_id] = validate_complete_row_token_ids(token_ids, owner_id=owner_id)
    # Validate every intervention owner before the first native generation.
    # Unknown branch ids must never turn into a partially executed case.
    parent_set = set(parent_ids)
    for arm in case["branch_arms"]:
        kind = str(arm["kind"])
        owner_id = str(arm["owner_id"])
        if owner_id not in owner_map:
            raise ValueError(
                f"case {case['case_id']} arm {kind} names unknown owner id {owner_id!r}"
            )
        if kind in {"complete_gt_owner", *PARTIAL_GT_BRANCH_COORDINATE_COUNTS} and owner_id in parent_set:
            raise ValueError(
                f"case {case['case_id']} arm {kind}:{owner_id} is already in parent prefix"
            )
        if kind == "covered_gt_owner" and owner_id not in parent_set:
            raise ValueError(
                f"case {case['case_id']} covered_gt_owner:{owner_id} must be in parent prefix"
            )
    parent_prefix_token_ids = [
        token_id
        for owner_id in parent_ids
        for token_id in row_token_map[owner_id]
    ]

    # The natural branch and its released successor horizon are the common
    # reference for all arms.
    native_branch = _generate_row(
        session=session,
        native_inputs=native_inputs,
        prefix_token_ids=parent_prefix_token_ids,
        tokenizer=session._tokenizer,
        image_width=int(plan.decoded_width),
        image_height=int(plan.decoded_height),
        mode="greedy",
        seed=None,
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=REPETITION_PENALTY,
        max_new_tokens=MAX_NEW_TOKENS,
        malformed_limit=2,
        row_index=len(parent_ids),
    )
    native_branch = _annotate_generated_row(
        native_branch,
        tokenizer=session._tokenizer,
        image_width=int(plan.decoded_width),
        image_height=int(plan.decoded_height),
        owner_rows=owner_rows,
        row_id=f"{image_id}:natural:branch",
    )
    native_branch["row_index"] = len(parent_ids)
    native_branch["input_prefix_token_ids"] = list(parent_prefix_token_ids)
    native_branch_prefix, native_branch_append = append_row_if_complete(
        parent_prefix_token_ids, native_branch
    )
    native_branch["append_receipt"] = native_branch_append
    native_branch["accepted_complete_row"] = bool(native_branch_append.get("appended"))
    native_branch_complete = _row_is_complete(native_branch)
    if native_branch_complete:
        natural_suffix, _natural_final_prefix = _generate_successor_rows(
            session=session,
            native_inputs=native_inputs,
            tokenizer=session._tokenizer,
            prefix_token_ids=native_branch_prefix,
            released_row_budget=int(case["released_row_budget"]),
            image_width=int(plan.decoded_width),
            image_height=int(plan.decoded_height),
            owner_rows=owner_rows,
            image_id=image_id,
            row_index_start=len(parent_ids) + 1,
        )
    else:
        natural_suffix = []
    natural_partition = _branch_and_suffix_partition(
        native_branch,
        natural_suffix,
        owner_rows=owner_rows,
        parent_owner_ids=parent_ids,
        remaining_owner_ids=remaining_ids,
    )
    natural = {
        "branch_row": native_branch,
        "released_suffix": natural_suffix,
        "owner_partition": natural_partition,
        "branch_structurally_complete": native_branch_complete,
    }

    # The native complete-row no-op uses the exact raw branch ids and the same
    # successor budget.  A mismatch invalidates causal interpretation of this
    # case but is preserved in the artifact rather than hidden by an exception.
    if native_branch_complete:
        noop_branch = _forced_complete_row(
            native_branch["raw_generated_token_ids"],
            owner_id="native",
            kind="native_complete_row_noop",
            tokenizer=session._tokenizer,
            image_width=int(plan.decoded_width),
            image_height=int(plan.decoded_height),
            owner_rows=owner_rows,
            row_id=f"{image_id}:native_complete_row_noop:branch",
        )
        noop_branch_prefix, noop_append = append_row_if_complete(
            parent_prefix_token_ids, noop_branch
        )
        noop_branch["append_receipt"] = noop_append
        noop_branch["accepted_complete_row"] = bool(noop_append.get("appended"))
        noop_suffix, _noop_final_prefix = _generate_successor_rows(
            session=session,
            native_inputs=native_inputs,
            tokenizer=session._tokenizer,
            prefix_token_ids=noop_branch_prefix,
            released_row_budget=int(case["released_row_budget"]),
            image_width=int(plan.decoded_width),
            image_height=int(plan.decoded_height),
            owner_rows=owner_rows,
            image_id=image_id,
            row_index_start=len(parent_ids) + 1,
        )
        noop_parity = compare_raw_noop_parity(natural_suffix, noop_suffix)
        noop = {
            "status": "valid" if noop_parity["passed"] else "inadmissible",
            "branch_row": noop_branch,
            "released_suffix": noop_suffix,
            "owner_partition": _branch_and_suffix_partition(
                noop_branch,
                noop_suffix,
                owner_rows=owner_rows,
                parent_owner_ids=parent_ids,
                remaining_owner_ids=remaining_ids,
            ),
            "raw_noop_parity": noop_parity,
        }
    else:
        noop = {
            "status": "not_applicable",
            "reason": "native_branch_not_structurally_complete",
            "raw_noop_parity": None,
        }

    arms: list[dict[str, Any]] = []
    for arm in case["branch_arms"]:
        kind = str(arm["kind"])
        owner_id = str(arm["owner_id"])
        if kind in PARTIAL_GT_BRANCH_COORDINATE_COUNTS:
            # The exact GT row ids were validated above.  Slice those ids at
            # the structural boundary instead of decoding/re-tokenizing the
            # description or inferring coordinate positions from text.
            forced_prefix = build_partial_gt_owner_prefix_token_ids(
                row_token_map[owner_id],
                kind=kind,
                owner_id=owner_id,
            )
            forced_branch = _generate_after_forced_partial_row(
                session=session,
                native_inputs=native_inputs,
                parent_prefix_token_ids=parent_prefix_token_ids,
                forced_row_prefix_token_ids=forced_prefix,
                tokenizer=session._tokenizer,
                image_width=int(plan.decoded_width),
                image_height=int(plan.decoded_height),
                repetition_penalty=REPETITION_PENALTY,
                max_new_tokens=MAX_NEW_TOKENS,
                malformed_limit=2,
                row_index=len(parent_ids),
            )
            branch_released_tail = _ids(
                forced_branch.get("released_tail_token_ids", []),
                label=f"{kind} released tail token ids",
            )
            branch_prefix = forced_prefix
            branch_row = _annotate_generated_row(
                forced_branch,
                tokenizer=session._tokenizer,
                image_width=int(plan.decoded_width),
                image_height=int(plan.decoded_height),
                owner_rows=owner_rows,
                row_id=f"{image_id}:{kind}:{owner_id}:branch",
            )
            branch_row["forced_row_prefix_token_ids"] = list(forced_prefix)
            branch_row["forced_owner_id"] = owner_id
            branch_row["branch_kind"] = kind
        else:
            forced_ids = row_token_map[owner_id]
            branch_prefix = forced_ids
            branch_released_tail = []
            branch_row = _forced_complete_row(
                forced_ids,
                owner_id=owner_id,
                kind=kind,
                tokenizer=session._tokenizer,
                image_width=int(plan.decoded_width),
                image_height=int(plan.decoded_height),
                owner_rows=owner_rows,
                row_id=f"{image_id}:{kind}:{owner_id}:branch",
            )
        branch_row["row_index"] = len(parent_ids)
        branch_row["input_prefix_token_ids"] = list(parent_prefix_token_ids)
        branch_row_prefix, branch_append = append_row_if_complete(
            parent_prefix_token_ids, branch_row
        )
        branch_row["append_receipt"] = branch_append
        branch_row["accepted_complete_row"] = bool(branch_append.get("appended"))
        branch_complete = _row_is_complete(branch_row)
        if branch_complete:
            suffix_rows, _final_prefix = _generate_successor_rows(
                session=session,
                native_inputs=native_inputs,
                tokenizer=session._tokenizer,
                prefix_token_ids=branch_row_prefix,
                released_row_budget=int(case["released_row_budget"]),
                image_width=int(plan.decoded_width),
                image_height=int(plan.decoded_height),
                owner_rows=owner_rows,
                image_id=image_id,
                row_index_start=len(parent_ids) + 1,
            )
        else:
            suffix_rows = []
        arm_result = _arm_result(
            arm=arm,
            branch_row=branch_row,
            suffix_rows=suffix_rows,
            owner_rows=owner_rows,
            parent_owner_ids=parent_ids,
            remaining_owner_ids=remaining_ids,
            branch_prefix_token_ids=branch_prefix,
            branch_released_tail_token_ids=branch_released_tail,
        )
        arm_result["branch_structurally_complete"] = branch_complete
        arms.append(arm_result)

    case_admissible, inadmissibility_reason = causal_admissibility_from_noop(noop)
    return {
        "case_id": str(case["case_id"]),
        "image_id": image_id,
        "parent_prefix_owner_ids": parent_ids,
        "remaining_owner_ids": remaining_ids,
        "released_row_budget": int(case["released_row_budget"]),
        "parent_prefix_token_ids": parent_prefix_token_ids,
        "parent_prefix_token_ids_sha256": _sha256_json(parent_prefix_token_ids),
        "base_prompt": {
            **dict(prompt_meta),
            "executed_prompt_token_ids_sha256": _sha256_json(executed_ids[0]),
        },
        "image": {
            "path": str(plan.image_path),
            "sha256": plan.image_content_sha256,
            "width": int(plan.decoded_width),
            "height": int(plan.decoded_height),
        },
        "observed_image_grid_thw": None if observed_grids[0] is None else list(observed_grids[0]),
        "executed_media_sha256": media_sha[0],
        "trusted_owner_rows": owner_rows,
        "natural_baseline": natural,
        "native_complete_row_noop": noop,
        "branch_arms": arms,
        "case_admissible": case_admissible,
        "inadmissibility_reason": inadmissibility_reason,
    }


def run_panel(
    *,
    case_manifest: Path,
    output: Path,
    device: str = "cuda:0",
    infer_config: Path = DEFAULT_INFER_CONFIG,
    force: bool = False,
) -> Path:
    """Run all manifest cases and write one immutable JSON artifact."""

    case_manifest = case_manifest.expanduser().resolve(strict=True)
    manifest = load_case_manifest(case_manifest)
    output = output.expanduser().resolve()
    if output.exists() and not force:
        raise ValueError(f"refusing to overwrite {output}; pass --force")

    # Runtime imports stay below pure validation so malformed manifests never
    # trigger model or CUDA initialization.
    import torch
    from src.config.fingerprint import sha256_json
    from src.config.inference import load_infer_config
    from src.data import load_raw_examples
    from src.inference.backend import open_backend_session
    from src.inference.runtime import assemble_frontend

    infer_config = infer_config.expanduser().resolve(strict=True)
    resolved = load_infer_config(infer_config)
    config = resolved.config
    raw_examples = list(load_raw_examples(config.data.input_jsonl))
    generation_fingerprint = sha256_json(
        {
            "max_new_tokens": MAX_NEW_TOKENS,
            "repetition_penalty": REPETITION_PENALTY,
            "do_sample": False,
            "batch_size": 1,
            "model_dtype": "fp32",
            "attn_implementation": "sdpa",
        }
    )
    frontend = assemble_frontend(config, generation_config_fingerprint=generation_fingerprint)
    launch = _session_launch_fp32_sdpa_batch1(frontend)
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.set_device(torch.device(device))

    results: list[dict[str, Any]] = []
    with open_backend_session(launch) as session:
        for case in manifest["cases"]:
            results.append(
                _run_one_case(
                    case=case,
                    config=config,
                    frontend=frontend,
                    session=session,
                    raw_examples=raw_examples,
                    image_id=str(case["image_id"]),
                )
            )
        model_receipt = session.receipt.to_artifact_dict()

    payload = {
        "schema_version": SCHEMA_VERSION,
        "manifest": str(case_manifest),
        "manifest_sha256": hashlib.sha256(case_manifest.read_bytes()).hexdigest(),
        "config": {
            "infer_config": str(infer_config),
            "resolved_fingerprint": resolved.fingerprint,
            "device": str(device),
            "backend": "hf",
            "model_dtype": "fp32",
            "attn_implementation": "sdpa",
            "batch_size": 1,
            "repetition_penalty": REPETITION_PENALTY,
            "max_new_tokens_per_row": MAX_NEW_TOKENS,
        },
        "model_session_receipt": model_receipt,
        "cases": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-manifest", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--infer-config", type=Path, default=DEFAULT_INFER_CONFIG)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    run_panel(
        case_manifest=args.case_manifest,
        device=args.device,
        output=args.output,
        infer_config=args.infer_config,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
