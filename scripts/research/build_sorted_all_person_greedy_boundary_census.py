#!/usr/bin/env python3
"""Build the CPU-only sealed planner for the native-greedy boundary census.

Prospective, data-driven successor to the six-context ``2026-08-03-sorted-
all-person-owner-relative-route-landscape`` plan (``plan-v2``).  Instead of
one hand-picked ``self-due``/``skip-post`` context per owner, this planner
builds *one* exact context for every complete-row boundary ``t = 0..T`` of
the native sorted rp1.00 greedy image-``7511`` rollout (``greedy.json``),
where ``T`` is that rollout's own observed complete-row count (currently
``10``, giving ``11`` boundaries).  Every one of the 41 confirmed ``person``
owners is then mapped onto exactly one of those boundaries -- deterministically,
never by score -- via the canonical Task-0 prediction-row ledger's own
strict-match verdict when one exists, and otherwise via the first spatial
overtake of the owner's ``(y1, x1)`` key by the rollout's own emitted row
keys.  The 369-row primary candidate bank is reused byte-identically from
``plan-v2``; this planner never regenerates or rescoring-selects candidates.

Like its predecessor this module never imports a model, tokenizer, or torch.
The output directory is create-or-identical: validation completes before any
byte is written, then the whole directory is atomically published.  ``plan-v2``
is read-only lineage here and is never modified or overwritten.

Sealed two-layer prefix (P0 fix)
--------------------------------
``plan-v2``'s own ``contexts.jsonl`` ends ``full_prefix_token_ids`` right at
the prompt (``root``) or at a prior row's own ``box_end`` (every other
context) -- but ``score_complete_box_candidate``'s own contract
(``score_sorted_owner_basin_landscape.py``) is that the reforward prefill's
root logits already *are* ``x1``'s distribution, i.e. the literal reforward
prefix must end at ``box_start``, not at ``box_end`` or the bare prompt.
Feeding a candidate's coordinate tokens straight onto a prefix that stops one
row-open short silently scores "what comes after a finished row" instead of
"what coordinate opens this forced box" -- a launch-blocking scoring-contract
break.  Every boundary context here therefore seals two separate, digested
layers: ``observed_self_prefix_token_ids`` (prompt plus every native complete
row through this boundary, exactly as generated, no forced or injected
tokens) and a fixed ``query_suffix_token_ids`` (``[object_ref_start,
person-token(s), object_ref_end, box_start]``, derived from -- and
cross-checked for uniformity across -- the rollout's own observed ``person``
rows, never hardcoded).  ``full_prefix_token_ids`` is their concatenation and
is the one array a scorer should open its reforward backend on; this planner
fails fast (before writing any byte) if that array does not end at
``box_start``.

Compatibility with ``score_sorted_all_person_route_landscape.load_plan()``
--------------------------------------------------------------------------
This planner deliberately reuses that scorer's own ``unit_id`` and
``schema_version`` literals (imported, not re-typed, from
``build_sorted_all_person_route_landscape``) plus the same per-row
``context_id`` / ``candidate_id`` / ``request_id`` / ``request_kind`` join
shape, so a boundary context or a Cartesian scoring request is structurally
indistinguishable from the six-context plan's own rows.  It adds one new
``plan_strategy`` receipt field so the two lineages stay distinguishable
without a schema fork.

Every token-distinct complete native greedy ``person`` row box (nine rows in
image ``7511`` -- row ``0`` is ``kite`` and is excluded) is bound as a
score-independent realized-box sidecar across all eleven boundaries: prior
calibration on this same checkpoint/image showed a native/sampled realized
``gt:7511:2`` box beating the size-aware nine-point bank by up to ``+5.3``
raw logprob, so finite-bank undercoverage is load-bearing evidence here, not
an optional diagnostic.  Sidecars bind row ID, raw generated tokens, pixel
box, the canonical (joint, whole-rollout) strict-match verdict, and an
independent per-box IoU assignment against all 41 owners (the same
isolated-candidate matcher semantics ``plan-v2`` uses for its own candidates
and sidecars) -- i.e. both a "strict" and a "loose" owner relation.  They
never enter primary ranks; primary and sidecar scoring-request counts are
always reported separately.

Full drop-in compatibility with the six-context scorer is nonetheless
infeasible without editing it (out of this module's owned scope):
``score_sorted_all_person_route_landscape.PLAN_FILE_NAMES`` hardcodes
``sampling-seeds.jsonl`` as a *required* file -- ``load_plan()`` calls
``Path.is_file()`` on it unconditionally, before it ever reads
``schema_version`` semantics -- and its own ``_read_jsonl`` additionally
rejects an empty file.  This planner has no prospective low-temperature-
sampling need for a boundary census (there is no frozen sampling policy
scoped to it), so it does not fabricate placeholder content for that one
file.  Concretely: ``load_plan()`` invoked against this planner's output
directory fails at ``plan directory is missing declared output file
'sampling-seeds.jsonl'``.  This is the exact, reported seam; see
``receipt.json``'s ``compatibility`` block.  Separately, downstream analysis
(``analyze_sorted_all_person_route_landscape.py``) independently hardcodes
the six-context ``root``/``self-due-*``/``skip-post-*`` registry and is not a
target of this planner's compatibility claim.

Note the field name ``full_prefix_token_ids`` is shared with ``plan-v2`` but
its *content contract* is deliberately stricter here (see "Sealed two-layer
prefix" above): this planner's array ends at ``box_start`` (the corrected,
launch-safe reforward boundary); ``plan-v2``'s own array of the same name
does not.  A scorer reading either plan by field name alone gets the correct
boundary from this planner and the pre-fix boundary from ``plan-v2``.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research.build_sorted_all_person_route_landscape import (  # noqa: E402
    SCHEMA_VERSION as PLAN_V2_SCHEMA_VERSION,
    UNIT_ID as PLAN_V2_UNIT_ID,
    canonical_json_bytes,
    sha256_file,
    sha256_json,
)

CENSUS_SCHEMA_VERSION = "sorted-all-person-greedy-boundary-census-plan.v1"
PLAN_STRATEGY = "native_greedy_boundary_census"

ROOT = Path("/data/CoordExp")
OUTPUTS = ROOT / "outputs/research/qwen3-vl-dense-enumeration"
PLAN_V2_DIR = (
    OUTPUTS / "2026-08-03-sorted-all-person-owner-relative-route-landscape/plan-v2"
)
GREEDY_PATH = (
    OUTPUTS
    / "2026-07-29-three-checkpoint-human-refined12-max3084/sorted/greedy/greedy.json"
)
TASK0_ROOT = (
    OUTPUTS / "2026-08-01-sorted-owner-basin-task0-v2-global-ambiguity-neutral-final"
)
PREDICTION_LEDGER_PATH = TASK0_ROOT / "prediction-row-ledger.jsonl"

IMAGE_ID = "7511"
CANONICAL_GREEDY_ROW_ID_PREFIX = "pred:sorted:greedy:0:7511:"
EXPECTED_OWNER_IDS = tuple(f"gt:7511:{index}" for index in range(2, 43))
EXPECTED_OWNER_COUNT = 41
EXPECTED_CANDIDATE_COUNT = 369
KNOWN_STRICT_MATCH_STATUSES = frozenset(
    {
        "matched",
        "unmatched",
        "ambiguous_neutral",
        "not_evaluable_invalid_geometry_or_description",
        "not_evaluable_parser_dropped",
    }
)


class BoundaryCensusContractError(ValueError):
    """Raised when a frozen input or planner invariant is not satisfied."""


@dataclass(frozen=True)
class SourcePaths:
    """All immutable inputs used by the CPU boundary-census planner."""

    plan_v2_dir: Path = PLAN_V2_DIR
    greedy: Path = GREEDY_PATH
    prediction_ledger: Path = PREDICTION_LEDGER_PATH


DEFAULT_SOURCES = SourcePaths()


# ---------------------------------------------------------------------------
# Generic JSON/JSONL loading helpers (self-contained; small and duplicated
# deliberately rather than importing another module's private internals).
# ---------------------------------------------------------------------------


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise BoundaryCensusContractError(f"{label} is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise BoundaryCensusContractError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(raw, Mapping):
        raise BoundaryCensusContractError(f"{label} must be a JSON object")
    return raw


def _read_jsonl_bytes(content: bytes, label: str) -> list[dict[str, Any]]:
    lines = content.decode("utf-8").splitlines()
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line:
            raise BoundaryCensusContractError(f"{label} line {line_number} is blank")
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BoundaryCensusContractError(
                f"{label} line {line_number} is not JSON"
            ) from exc
        if not isinstance(raw, Mapping):
            raise BoundaryCensusContractError(
                f"{label} line {line_number} must be an object"
            )
        rows.append(dict(raw))
    if not rows:
        raise BoundaryCensusContractError(f"{label} must contain at least one row")
    return rows


def _read_jsonl_file(path: Path, label: str) -> tuple[bytes, list[dict[str, Any]]]:
    try:
        content = path.read_bytes()
    except FileNotFoundError as exc:
        raise BoundaryCensusContractError(f"{label} is missing: {path}") from exc
    return content, _read_jsonl_bytes(content, label)


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise BoundaryCensusContractError(f"{label} must be a non-empty trimmed string")
    return value


def _int_list(value: Any, label: str) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise BoundaryCensusContractError(f"{label} must be an integer list")
    result: list[int] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise BoundaryCensusContractError(
                f"{label}[{index}] must be a non-negative integer"
            )
        result.append(item)
    return result


def _pixel_box(value: Any, label: str, *, width: int, height: int) -> tuple[int, int, int, int]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 4
    ):
        raise BoundaryCensusContractError(f"{label} must be a four-value pixel xyxy box")
    result: list[int] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise BoundaryCensusContractError(f"{label}[{index}] must be numeric")
        numeric = float(item)
        if not numeric.is_integer():
            raise BoundaryCensusContractError(f"{label}[{index}] must be an integer pixel value")
        result.append(int(numeric))
    x1, y1, x2, y2 = result
    if not 0 <= x1 < x2 <= width or not 0 <= y1 < y2 <= height:
        raise BoundaryCensusContractError(f"{label} is outside the frozen image canvas")
    return x1, y1, x2, y2


# ---------------------------------------------------------------------------
# plan-v2 lineage: exact reuse and bind (owners, candidates, runtime identity)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlanV2Bundle:
    plan_dir: Path
    receipt: Mapping[str, Any]
    receipt_bytes: bytes
    owner_ledger_bytes: bytes
    owner_rows: list[dict[str, Any]]
    primary_candidates_bytes: bytes
    candidate_rows: list[dict[str, Any]]
    frozen_runtime_identity: Mapping[str, Any]


def _load_plan_v2(plan_v2_dir: Path) -> PlanV2Bundle:
    receipt_path = plan_v2_dir / "receipt.json"
    receipt = _read_json(receipt_path, "plan-v2 receipt.json")
    if receipt.get("schema_version") != PLAN_V2_SCHEMA_VERSION:
        raise BoundaryCensusContractError(
            "plan-v2 receipt.schema_version does not match the expected six-context plan schema"
        )
    if receipt.get("unit_id") != PLAN_V2_UNIT_ID:
        raise BoundaryCensusContractError(
            "plan-v2 receipt.unit_id does not match the expected owner-relative-route-landscape unit"
        )
    reconstructed = sha256_json(
        {key: value for key, value in receipt.items() if key != "receipt_content_sha256"}
    )
    if reconstructed != receipt.get("receipt_content_sha256"):
        raise BoundaryCensusContractError(
            "plan-v2 receipt.json does not reconstruct its own receipt_content_sha256; "
            "stale or tampered lineage plan"
        )
    output_digests = receipt.get("output_file_digests")
    if not isinstance(output_digests, Mapping):
        raise BoundaryCensusContractError("plan-v2 receipt lacks output_file_digests")

    def _bind(name: str) -> bytes:
        path = plan_v2_dir / name
        if not path.is_file():
            raise BoundaryCensusContractError(f"plan-v2 is missing declared output file {name!r}")
        expected = output_digests.get(name)
        content = path.read_bytes()
        actual = hashlib.sha256(content).hexdigest()
        if not isinstance(expected, str) or actual != expected:
            raise BoundaryCensusContractError(
                f"plan-v2 file {name!r} does not match its own sealed receipt digest; "
                "tampered or stale lineage plan"
            )
        return content

    owner_ledger_bytes = _bind("owner-ledger.jsonl")
    primary_candidates_bytes = _bind("primary-candidates.jsonl")

    owner_rows = _read_jsonl_bytes(owner_ledger_bytes, "plan-v2 owner-ledger.jsonl")
    candidate_rows = _read_jsonl_bytes(
        primary_candidates_bytes, "plan-v2 primary-candidates.jsonl"
    )

    owner_ids = tuple(str(row.get("gt_owner_id")) for row in owner_rows)
    if owner_ids != EXPECTED_OWNER_IDS:
        raise BoundaryCensusContractError(
            "plan-v2 owner ledger does not contain exactly gt:7511:2 through gt:7511:42"
        )
    if len(candidate_rows) != EXPECTED_CANDIDATE_COUNT:
        raise BoundaryCensusContractError(
            "plan-v2 primary candidate bank is not exactly 369 rows"
        )
    candidate_ids = {str(row.get("candidate_id")) for row in candidate_rows}
    if len(candidate_ids) != EXPECTED_CANDIDATE_COUNT:
        raise BoundaryCensusContractError("plan-v2 primary candidate IDs are not unique")
    candidate_generation = receipt.get("candidate_generation")
    if not isinstance(candidate_generation, Mapping):
        raise BoundaryCensusContractError("plan-v2 receipt lacks candidate_generation")
    expected_universe_sha256 = candidate_generation.get("primary_candidate_universe_sha256")
    if sha256_json(list(candidate_rows)) != expected_universe_sha256:
        raise BoundaryCensusContractError(
            "plan-v2 primary candidate bank content does not reconstruct its own sealed digest"
        )
    for owner_id in EXPECTED_OWNER_IDS:
        owner_candidate_rows = [
            row for row in candidate_rows if row.get("generator_gt_owner_id") == owner_id
        ]
        if len(owner_candidate_rows) != 9:
            raise BoundaryCensusContractError(
                f"plan-v2 candidate bank owner {owner_id} does not have nine reused candidates"
            )

    frozen_runtime_identity = receipt.get("frozen_runtime_identity")
    if not isinstance(frozen_runtime_identity, Mapping):
        raise BoundaryCensusContractError("plan-v2 receipt lacks frozen_runtime_identity")

    return PlanV2Bundle(
        plan_dir=plan_v2_dir,
        receipt=receipt,
        receipt_bytes=receipt_path.read_bytes(),
        owner_ledger_bytes=owner_ledger_bytes,
        owner_rows=owner_rows,
        primary_candidates_bytes=primary_candidates_bytes,
        candidate_rows=candidate_rows,
        frozen_runtime_identity=frozen_runtime_identity,
    )


# ---------------------------------------------------------------------------
# Native greedy rollout: load, split into complete rows, and cross-bind
# against the canonical Task-0 prediction-row ledger.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GreedyRollout:
    prompt_token_ids: list[int]
    prompt_token_ids_sha256: str
    generated_token_ids: list[int]
    row_token_spans: list[list[int]]
    predictions: list[Mapping[str, Any]]
    total_complete_row_count: int
    decode_mode: str
    repetition_penalty: float
    seed: int
    stop_reason: str
    wrapper_token_ids: Mapping[str, int]
    person_token_ids: list[int]


def _split_generated_rows(
    tokens: Sequence[int],
    *,
    object_ref_start: int,
    object_ref_end: int,
    box_start: int,
    box_end: int,
    coord_token_start: int,
    coord_token_end: int,
) -> list[list[int]]:
    starts = [index for index, token in enumerate(tokens) if token == object_ref_start]
    if not starts or starts[0] != 0:
        raise BoundaryCensusContractError(
            "greedy rollout generated tokens do not begin with an object row"
        )
    ends = [*starts[1:], len(tokens)]
    rows = [list(tokens[start:end]) for start, end in zip(starts, ends, strict=True)]
    for index, row in enumerate(rows):
        if len(row) < 8 or row[0] != object_ref_start:
            raise BoundaryCensusContractError(f"greedy row {index} has an invalid object start")
        try:
            object_end = row.index(object_ref_end)
        except ValueError as exc:
            raise BoundaryCensusContractError(
                f"greedy row {index} lacks object_ref_end"
            ) from exc
        if object_end <= 1 or row[object_end + 1 : object_end + 2] != [box_start]:
            raise BoundaryCensusContractError(
                f"greedy row {index} has an invalid object/box boundary"
            )
        coordinate = row[object_end + 2 : object_end + 6]
        if len(coordinate) != 4 or any(
            token < coord_token_start or token > coord_token_end for token in coordinate
        ):
            raise BoundaryCensusContractError(
                f"greedy row {index} lacks four coordinate tokens"
            )
        if row[object_end + 6 :] != [box_end]:
            raise BoundaryCensusContractError(
                f"greedy row {index} violates the closed one-box grammar"
            )
    return rows


def _row_coordinates(row: Sequence[int], object_ref_end: int) -> list[int]:
    object_end = row.index(object_ref_end)
    return list(row[object_end + 2 : object_end + 6])


def _row_description_token_ids(
    row: Sequence[int], *, object_ref_start: int, object_ref_end: int
) -> list[int]:
    if row[0] != object_ref_start:
        raise BoundaryCensusContractError("row does not open with object_ref_start")
    object_end = row.index(object_ref_end)
    return list(row[1:object_end])


def _derive_person_token_ids(
    predictions: Sequence[Mapping[str, Any]],
    row_token_spans: Sequence[Sequence[int]],
    *,
    object_ref_start: int,
    object_ref_end: int,
) -> list[int]:
    """The canonical ``person`` description token span, cross-checked for
    uniformity across every observed ``person`` row rather than hardcoded.
    """

    observed: set[tuple[int, ...]] = set()
    for index, prediction in enumerate(predictions):
        if prediction.get("description") != "person":
            continue
        tokens = tuple(
            _row_description_token_ids(
                row_token_spans[index],
                object_ref_start=object_ref_start,
                object_ref_end=object_ref_end,
            )
        )
        observed.add(tokens)
    if not observed:
        raise BoundaryCensusContractError(
            "greedy rollout has no observed person row to derive the canonical description tokens from"
        )
    if len(observed) != 1:
        raise BoundaryCensusContractError(
            "greedy rollout's observed person rows do not share one canonical description token span"
        )
    return list(next(iter(observed)))


def _load_greedy_rollout(path: Path, plan_v2: PlanV2Bundle) -> GreedyRollout:
    runtime = plan_v2.frozen_runtime_identity
    width = int(runtime["image_width"])
    height = int(runtime["image_height"])
    wrapper_token_ids = runtime.get("wrapper_token_ids")
    coordinate_token_ids = runtime.get("coordinate_token_ids")
    if not isinstance(wrapper_token_ids, Mapping) or not isinstance(coordinate_token_ids, Mapping):
        raise BoundaryCensusContractError(
            "plan-v2 frozen_runtime_identity lacks wrapper/coordinate token identity"
        )
    object_ref_start = int(wrapper_token_ids["object_ref_start"])
    object_ref_end = int(wrapper_token_ids["object_ref_end"])
    box_start = int(wrapper_token_ids["box_start"])
    box_end = int(wrapper_token_ids["box_end"])
    coord_token_start = int(coordinate_token_ids["start"])
    coord_token_end = int(coordinate_token_ids["end"])

    doc = _read_json(path, "greedy rollout")
    rollouts = doc.get("rollouts")
    if not isinstance(rollouts, Sequence) or isinstance(rollouts, (str, bytes)):
        raise BoundaryCensusContractError("greedy rollout file has no rollout list")
    matching = [
        row
        for row in rollouts
        if isinstance(row, Mapping) and str(row.get("image_id")) == IMAGE_ID
    ]
    if len(matching) != 1:
        raise BoundaryCensusContractError(
            "greedy rollout file must contain exactly one image-7511 rollout"
        )
    rollout = matching[0]
    if rollout.get("decode_mode") != "greedy":
        raise BoundaryCensusContractError("greedy rollout is not a greedy-decode rollout")

    config = doc.get("config")
    if not isinstance(config, Mapping) or config.get("repetition_penalty") != 1.0:
        raise BoundaryCensusContractError(
            "greedy rollout is not the frozen native rp1.00 policy stratum"
        )

    prompt = _int_list(rollout.get("prompt_token_ids"), "greedy prompt_token_ids")
    prompt_sha256 = sha256_json(prompt)
    if (
        prompt_sha256 != runtime.get("prompt_token_ids_sha256")
        or rollout.get("prompt_token_ids_sha256") != runtime.get("prompt_token_ids_sha256")
    ):
        raise BoundaryCensusContractError(
            "greedy rollout prompt tokens do not match plan-v2's frozen prompt identity"
        )
    if rollout.get("executed_media_sha256") != runtime.get("executed_media_sha256"):
        raise BoundaryCensusContractError(
            "greedy rollout executed-media digest does not match plan-v2's frozen identity"
        )

    metadata = doc.get("prompt_metadata")
    if not isinstance(metadata, Mapping):
        raise BoundaryCensusContractError("greedy rollout lacks prompt metadata")
    image_metadata = metadata.get("coco2017_val_000000007511")
    if not isinstance(image_metadata, Mapping):
        raise BoundaryCensusContractError("greedy rollout lacks image-7511 prompt metadata")
    if (
        image_metadata.get("width") != width
        or image_metadata.get("height") != height
        or image_metadata.get("image_sha256") != runtime.get("image_sha256")
    ):
        raise BoundaryCensusContractError(
            "greedy rollout image dimensions or digest differ from plan-v2's frozen identity"
        )

    model_identity = doc.get("model_identity")
    if not isinstance(model_identity, Mapping):
        raise BoundaryCensusContractError("greedy rollout lacks model_identity")
    tokenizer_identity = model_identity.get("tokenizer_identity")
    if not isinstance(tokenizer_identity, Mapping):
        raise BoundaryCensusContractError("greedy rollout lacks tokenizer_identity")
    observed_wrapper = tokenizer_identity.get("wrapper_token_ids")
    if not isinstance(observed_wrapper, Mapping) or (
        observed_wrapper.get("<|object_ref_start|>") != object_ref_start
        or observed_wrapper.get("<|object_ref_end|>") != object_ref_end
        or observed_wrapper.get("<|box_start|>") != box_start
        or observed_wrapper.get("<|box_end|>") != box_end
    ):
        raise BoundaryCensusContractError(
            "greedy rollout tokenizer wrapper token IDs disagree with plan-v2's frozen identity"
        )
    if (
        tokenizer_identity.get("coord_token_id_min") != coord_token_start
        or tokenizer_identity.get("coord_token_id_max") != coord_token_end
    ):
        raise BoundaryCensusContractError(
            "greedy rollout coordinate token bounds disagree with plan-v2's frozen identity"
        )

    generated = _int_list(rollout.get("generated_token_ids"), "greedy generated_token_ids")
    predictions_object = rollout.get("predictions")
    if not isinstance(predictions_object, Mapping):
        raise BoundaryCensusContractError("greedy rollout lacks parsed predictions")
    if predictions_object.get("parse_status") != "accepted":
        raise BoundaryCensusContractError("greedy rollout parse_status is not accepted")
    if predictions_object.get("dropped_prediction_count") != 0:
        raise BoundaryCensusContractError(
            "greedy rollout has dropped/incomplete predictions; boundary census requires a "
            "fully complete rollout"
        )
    predictions = predictions_object.get("predictions")
    if not isinstance(predictions, Sequence) or isinstance(predictions, (str, bytes)):
        raise BoundaryCensusContractError("greedy rollout parsed predictions is invalid")
    predictions = [item for item in predictions if isinstance(item, Mapping)]
    if len(predictions) != predictions_object.get("valid_prediction_count"):
        raise BoundaryCensusContractError(
            "greedy rollout parsed prediction count does not match its own valid_prediction_count"
        )

    rows = _split_generated_rows(
        generated,
        object_ref_start=object_ref_start,
        object_ref_end=object_ref_end,
        box_start=box_start,
        box_end=box_end,
        coord_token_start=coord_token_start,
        coord_token_end=coord_token_end,
    )
    if len(rows) != len(predictions):
        raise BoundaryCensusContractError(
            "greedy rollout structural token-row count does not match its parsed prediction count"
        )

    for index, prediction in enumerate(predictions):
        if prediction.get("generated_order") != index:
            raise BoundaryCensusContractError(
                f"greedy rollout parsed row {index} is out of emission order"
            )
        if prediction.get("bbox_format") != "xyxy":
            raise BoundaryCensusContractError(f"greedy rollout parsed row {index} bbox_format is not xyxy")
        bbox = _pixel_box(prediction.get("bbox"), f"greedy parsed row {index} bbox", width=width, height=height)
        coordinate = _row_coordinates(rows[index], object_ref_end)
        parsed_bins = _int_list(prediction.get("coord_bins"), f"greedy parsed row {index} coord_bins")
        if coordinate != [coord_token_start + value for value in parsed_bins]:
            raise BoundaryCensusContractError(
                f"greedy rollout row {index} parsed coordinates disagree with its own token IDs"
            )
        if list(bbox) != list(prediction.get("bbox")):
            raise BoundaryCensusContractError(
                f"greedy rollout row {index} bbox is not a valid pixel box on the frozen canvas"
            )

    person_token_ids = _derive_person_token_ids(
        predictions, rows, object_ref_start=object_ref_start, object_ref_end=object_ref_end
    )

    return GreedyRollout(
        prompt_token_ids=prompt,
        prompt_token_ids_sha256=prompt_sha256,
        generated_token_ids=generated,
        row_token_spans=rows,
        predictions=predictions,
        total_complete_row_count=len(rows),
        decode_mode=str(rollout.get("decode_mode")),
        repetition_penalty=float(config.get("repetition_penalty")),
        seed=int(rollout.get("seed")),
        stop_reason=str(rollout.get("stop_reason")),
        wrapper_token_ids={
            "object_ref_start": object_ref_start,
            "object_ref_end": object_ref_end,
            "box_start": box_start,
            "box_end": box_end,
        },
        person_token_ids=person_token_ids,
    )


@dataclass(frozen=True)
class CanonicalLedgerRow:
    original_row_index: int
    strict_match_status: str
    strict_match_gt_owner_id: str | None
    raw_span_sha256: str
    execution_receipt_content_sha256: str | None


def _load_canonical_greedy_ledger_rows(
    path: Path, greedy: GreedyRollout, plan_v2: PlanV2Bundle
) -> dict[int, CanonicalLedgerRow]:
    _, rows = _read_jsonl_file(path, "canonical prediction-row ledger")
    selected: dict[int, CanonicalLedgerRow] = {}
    for row in rows:
        pred_row_id = row.get("pred_row_id")
        if not isinstance(pred_row_id, str) or not pred_row_id.startswith(
            CANONICAL_GREEDY_ROW_ID_PREFIX
        ):
            continue
        if row.get("row_kind") != "complete_prediction":
            raise BoundaryCensusContractError(
                "canonical greedy prediction ledger row is not a complete_prediction"
            )
        status = row.get("strict_match_status")
        if status not in KNOWN_STRICT_MATCH_STATUSES:
            raise BoundaryCensusContractError(
                f"canonical greedy prediction ledger row has an unrecognized strict_match_status: {status!r}"
            )
        try:
            index = int(pred_row_id.rsplit(":", 1)[1])
        except ValueError as exc:
            raise BoundaryCensusContractError(
                "canonical greedy prediction ledger row has an invalid row suffix"
            ) from exc
        if index in selected:
            raise BoundaryCensusContractError(
                "canonical greedy prediction ledger has duplicate row IDs"
            )
        owner_id = row.get("strict_match_gt_owner_id")
        if status == "matched" and not isinstance(owner_id, str):
            raise BoundaryCensusContractError(
                f"canonical greedy prediction ledger row {index} is matched without an owner id"
            )
        selected[index] = CanonicalLedgerRow(
            original_row_index=index,
            strict_match_status=str(status),
            strict_match_gt_owner_id=owner_id if status == "matched" else None,
            raw_span_sha256=str(row.get("raw_span_sha256")),
            execution_receipt_content_sha256=row.get("execution_receipt_content_sha256"),
        )

    total = greedy.total_complete_row_count
    if set(selected) != set(range(total)):
        raise BoundaryCensusContractError(
            "canonical greedy prediction ledger does not cover exactly the rollout's complete rows"
        )

    for index, prediction in enumerate(greedy.predictions):
        ledger_row = selected[index]
        if ledger_row.raw_span_sha256 != prediction.get("raw_span_sha256"):
            raise BoundaryCensusContractError(
                f"canonical greedy prediction ledger row {index} raw-span digest disagrees "
                "with the rollout's own parsed span"
            )

    execution_receipts = {row.execution_receipt_content_sha256 for row in selected.values()}
    if len(execution_receipts) != 1:
        raise BoundaryCensusContractError(
            "canonical greedy prediction ledger rows do not share one uniform execution receipt"
        )
    owner_execution_receipts = {
        row.get("source_owner_ledger_execution_receipt_content_sha256")
        for row in plan_v2.owner_rows
    }
    if owner_execution_receipts != execution_receipts:
        raise BoundaryCensusContractError(
            "canonical greedy prediction ledger execution receipt does not match plan-v2's "
            "bound owner-ledger execution receipt"
        )

    return selected


# ---------------------------------------------------------------------------
# Owner <-> boundary role assignment (pure, independently testable)
# ---------------------------------------------------------------------------


def owner_sort_key(bbox_pixel_xyxy: Sequence[int]) -> tuple[int, int]:
    x1, y1, _x2, _y2 = bbox_pixel_xyxy
    return (int(y1), int(x1))


def row_sort_key(bbox_xyxy: Sequence[int]) -> tuple[int, int]:
    x1, y1, _x2, _y2 = bbox_xyxy
    return (int(y1), int(x1))


def compute_owner_boundary_roles(
    *,
    owners: Sequence[Mapping[str, Any]],
    matched_owner_to_row: Mapping[str, int],
    row_keys: Sequence[tuple[int, int]],
    total_complete_row_count: int,
) -> list[dict[str, Any]]:
    """Deterministically map every owner onto exactly one boundary pair.

    Never chooses by score.  Case A (``strict_matched_emit``) is a direct
    lookup into the canonical matcher's own verdict.  Case B
    (``spatial_overtake``) scans rows in literal emission order (never
    assuming the row-key sequence is monotone) for the first row whose own
    ``(y1, x1)`` key is strictly after the owner's key.
    """

    total = total_complete_row_count
    roles: list[dict[str, Any]] = []
    for owner in owners:
        owner_id = _string(owner.get("gt_owner_id"), "owner id")
        owner_key = owner_sort_key(owner["bbox_pixel_xyxy"])
        if owner_id in matched_owner_to_row:
            matched_row = matched_owner_to_row[owner_id]
            roles.append(
                {
                    "gt_owner_id": owner_id,
                    "owner_sort_key_y1_x1": list(owner_key),
                    "mapping_kind": "strict_matched_emit",
                    "matched_row_index": matched_row,
                    "matched_row_key_y1_x1": list(row_keys[matched_row]),
                    "pre_boundary_index": matched_row,
                    "post_boundary_index": matched_row + 1,
                    "no_observed_overtake_before_stop": False,
                    "post_overtake_reversion_observed": None,
                }
            )
            continue

        found: int | None = None
        for row_index in range(total):
            if row_keys[row_index] > owner_key:
                found = row_index
                break
        if found is None:
            roles.append(
                {
                    "gt_owner_id": owner_id,
                    "owner_sort_key_y1_x1": list(owner_key),
                    "mapping_kind": "spatial_overtake",
                    "matched_row_index": None,
                    "matched_row_key_y1_x1": None,
                    "pre_boundary_index": total,
                    "post_boundary_index": total,
                    "no_observed_overtake_before_stop": True,
                    "post_overtake_reversion_observed": None,
                }
            )
        else:
            reversion = any(
                row_keys[later] <= owner_key for later in range(found + 1, total)
            )
            roles.append(
                {
                    "gt_owner_id": owner_id,
                    "owner_sort_key_y1_x1": list(owner_key),
                    "mapping_kind": "spatial_overtake",
                    "matched_row_index": found,
                    "matched_row_key_y1_x1": list(row_keys[found]),
                    "pre_boundary_index": found,
                    "post_boundary_index": found + 1,
                    "no_observed_overtake_before_stop": False,
                    "post_overtake_reversion_observed": reversion,
                }
            )
    if len({role["gt_owner_id"] for role in roles}) != len(roles):
        raise BoundaryCensusContractError("owner-boundary role map has duplicate owner IDs")
    return roles


def build_matched_owner_to_row(
    ledger_rows: Mapping[int, CanonicalLedgerRow], owner_id_set: Sequence[str]
) -> dict[str, int]:
    """The canonical ledger's joint strict-match verdict, restricted to owners
    in scope, one row per owner -- fails closed if a row claims an owner
    another row already claimed (a global bipartite match should never do
    this; this is a defense-in-depth cross-check).
    """

    owner_ids = frozenset(owner_id_set)
    matched_owner_to_row: dict[str, int] = {}
    for index, ledger_row in ledger_rows.items():
        if ledger_row.strict_match_status != "matched":
            continue
        owner_id = ledger_row.strict_match_gt_owner_id
        if owner_id not in owner_ids:
            continue
        if owner_id in matched_owner_to_row:
            raise BoundaryCensusContractError(
                f"owner {owner_id} is claimed by more than one greedy row"
            )
        matched_owner_to_row[owner_id] = index
    return matched_owner_to_row


def row_key_monotonicity(row_keys: Sequence[tuple[int, int]]) -> dict[str, Any]:
    backtracking_pairs = [
        {
            "from_index": index,
            "to_index": index + 1,
            "from_key_y1_x1": list(row_keys[index]),
            "to_key_y1_x1": list(row_keys[index + 1]),
        }
        for index in range(len(row_keys) - 1)
        if row_keys[index] > row_keys[index + 1]
    ]
    return {
        "is_monotonic_nondecreasing": not backtracking_pairs,
        "backtracking_pairs": backtracking_pairs,
    }


# ---------------------------------------------------------------------------
# Boundary contexts (one per t = 0..T), owner membership, scoring requests
# ---------------------------------------------------------------------------


def _context_id(boundary_index: int) -> str:
    return f"boundary-{boundary_index:02d}"


def assert_full_prefix_ends_at_box_start(
    full_prefix_token_ids: Sequence[int], *, box_start: int, boundary_index: int
) -> None:
    """Fail fast: the literal reforward prefix must end at ``box_start``.

    ``score_complete_box_candidate``'s own contract is that its prefill root
    logits already are ``x1``'s distribution; that is only true when the
    reforward prefix ends immediately at the forced box opener, never one
    row-open short (at a prior ``box_end``) or at the bare prompt.
    """

    if not full_prefix_token_ids or full_prefix_token_ids[-1] != box_start:
        raise BoundaryCensusContractError(
            f"boundary {boundary_index} full_prefix_token_ids does not end at box_start; "
            "the reforward prefill would not yield x1's distribution"
        )


def build_boundary_contexts(
    greedy: GreedyRollout, ledger_rows: Mapping[int, CanonicalLedgerRow]
) -> list[dict[str, Any]]:
    """Seal two layers per boundary, never one.

    ``observed_self_prefix_token_ids`` is the literal, unmodified prompt plus
    every native complete row through this boundary -- exactly what was
    actually generated, no forced or injected tokens.  ``query_suffix_token_ids``
    is the fixed ``[object_ref_start, person-token(s), object_ref_end,
    box_start]`` span that forces the canonical description and box opener
    the unit's decision score is defined against (unit.md: "raw pre-penalty
    complete-box log likelihood after forcing only the canonical description
    person and the registered box opener").  ``full_prefix_token_ids`` is
    their concatenation and is the literal reforward input a scorer must
    open a backend on: ``score_complete_box_candidate``'s own contract is
    that its root/prefill logits already are ``x1``'s distribution, i.e. the
    reforward prefix must end at ``box_start``, not at a prior row's
    ``box_end`` or at the bare prompt.  Failing to seal this second layer
    silently asks the model "what comes after a finished row" instead of
    "what coordinate opens this forced box" -- the exact P0 this planner
    fails fast against below.
    """

    total = greedy.total_complete_row_count
    prompt = greedy.prompt_token_ids
    wrapper = greedy.wrapper_token_ids
    query_suffix = [
        wrapper["object_ref_start"],
        *greedy.person_token_ids,
        wrapper["object_ref_end"],
        wrapper["box_start"],
    ]
    query_suffix_sha256 = sha256_json(query_suffix)
    contexts: list[dict[str, Any]] = []
    for boundary_index in range(total + 1):
        row_indices = list(range(boundary_index))
        self_prefix_rows = []
        for row_index in row_indices:
            tokens = greedy.row_token_spans[row_index]
            prediction = greedy.predictions[row_index]
            ledger_row = ledger_rows[row_index]
            self_prefix_rows.append(
                {
                    "row_index": row_index,
                    "generated_token_ids": list(tokens),
                    "generated_token_ids_sha256": sha256_json(list(tokens)),
                    "raw_span_sha256": ledger_row.raw_span_sha256,
                    "raw_span_text": prediction.get("raw_span_text"),
                    "description": prediction.get("description"),
                    "strict_match_status": ledger_row.strict_match_status,
                    "strict_match_gt_owner_id": ledger_row.strict_match_gt_owner_id,
                }
            )
        generated_prefix = [token for row in self_prefix_rows for token in row["generated_token_ids"]]
        observed_self_prefix = [*prompt, *generated_prefix]
        full_prefix = [*observed_self_prefix, *query_suffix]
        assert_full_prefix_ends_at_box_start(
            full_prefix, box_start=wrapper["box_start"], boundary_index=boundary_index
        )
        is_terminal = boundary_index == total
        contexts.append(
            {
                "schema_version": CENSUS_SCHEMA_VERSION,
                "row_kind": "admitted_boundary_context",
                "context_id": _context_id(boundary_index),
                "boundary_index": boundary_index,
                "total_complete_row_count": total,
                "image_id": IMAGE_ID,
                "source_seed": greedy.seed,
                "decode_mode": greedy.decode_mode,
                "repetition_penalty": greedy.repetition_penalty,
                "self_prefix_row_indices": row_indices,
                "self_prefix_rows": self_prefix_rows,
                "prompt_token_ids": prompt,
                "prompt_token_ids_sha256": greedy.prompt_token_ids_sha256,
                "generated_prefix_token_ids": generated_prefix,
                "generated_prefix_token_ids_sha256": sha256_json(generated_prefix),
                "observed_self_prefix_token_ids": observed_self_prefix,
                "observed_self_prefix_token_ids_sha256": sha256_json(observed_self_prefix),
                "query_suffix_token_ids": query_suffix,
                "query_suffix_token_ids_sha256": query_suffix_sha256,
                "full_prefix_token_ids": full_prefix,
                "full_prefix_token_ids_sha256": sha256_json(full_prefix),
                "candidate_universe_join": "primary-candidates.jsonl:candidate_id",
                "teacher_forced_chosen_token_parity": {
                    "status": "required_not_cpu_verifiable",
                    "admission_requirement": "must_pass_before_model_scoring",
                    "claimed_pass": False,
                },
                "terminal_status": {
                    "is_terminal": is_terminal,
                    "natural_stop": is_terminal and greedy.stop_reason == "im_end",
                    "stop_reason": greedy.stop_reason if is_terminal else None,
                },
            }
        )
    if len(contexts) != total + 1:
        raise BoundaryCensusContractError("boundary context registry has the wrong length")
    if len({row["context_id"] for row in contexts}) != len(contexts):
        raise BoundaryCensusContractError("boundary context IDs are not unique")
    return contexts


def attach_owner_membership(
    contexts: Sequence[Mapping[str, Any]], roles: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    membership: dict[int, dict[str, list[str]]] = {
        context["boundary_index"]: {
            "pre_emit_owner_ids": [],
            "post_emit_owner_ids": [],
            "pre_overtake_owner_ids": [],
            "post_overtake_owner_ids": [],
        }
        for context in contexts
    }
    for role in roles:
        owner_id = role["gt_owner_id"]
        pre_key = "pre_emit_owner_ids" if role["mapping_kind"] == "strict_matched_emit" else "pre_overtake_owner_ids"
        post_key = "post_emit_owner_ids" if role["mapping_kind"] == "strict_matched_emit" else "post_overtake_owner_ids"
        membership[role["pre_boundary_index"]][pre_key].append(owner_id)
        membership[role["post_boundary_index"]][post_key].append(owner_id)
    updated = []
    for context in contexts:
        entry = dict(context)
        entry["owner_membership"] = {
            key: sorted(value) for key, value in membership[context["boundary_index"]].items()
        }
        updated.append(entry)
    return updated


MATCHER_EPSILON = 1e-9


def _iou(left: Sequence[int], right: Sequence[int]) -> float:
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, float(left[2]) - float(left[0])) * max(0.0, float(left[3]) - float(left[1]))
    right_area = max(0.0, float(right[2]) - float(right[0])) * max(0.0, float(right[3]) - float(right[1]))
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def independent_owner_assignment(
    pixel_box: Sequence[int], owners: Sequence[Mapping[str, Any]], *, iou_threshold: float
) -> dict[str, Any]:
    """Isolated one-box-vs-all-41-owners assignment (plan-v2's own semantics).

    Distinct from the canonical prediction-row ledger's joint, whole-rollout
    bipartite match: this never constrains one owner to at most one row, so
    it is the "loose" bound machinery a realized-box sidecar needs alongside
    the ledger's own "strict" (canonical) verdict.
    """

    eligible: list[tuple[str, float]] = []
    all_receipts: list[dict[str, Any]] = []
    for owner in owners:
        owner_id = str(owner["gt_owner_id"])
        overlap = _iou(pixel_box, owner["bbox_pixel_xyxy"])
        all_receipts.append({"gt_owner_id": owner_id, "intersection_over_union": overlap})
        if overlap + MATCHER_EPSILON >= iou_threshold:
            eligible.append((owner_id, overlap))
    all_receipts.sort(key=lambda item: (-item["intersection_over_union"], item["gt_owner_id"]))
    if not eligible:
        return {
            "strict_assignment_status": "unmatched",
            "strict_assignment_gt_owner_id": None,
            "ambiguity_owner_ids": [],
            "lower_bound_owner_ids": [],
            "upper_bound_owner_ids": [],
            "eligible_owner_iou_receipts": all_receipts,
        }
    maximum = max(value for _, value in eligible)
    optimal = sorted(owner_id for owner_id, value in eligible if abs(value - maximum) <= MATCHER_EPSILON)
    if len(optimal) != 1:
        return {
            "strict_assignment_status": "ambiguous_neutral",
            "strict_assignment_gt_owner_id": None,
            "ambiguity_owner_ids": optimal,
            "lower_bound_owner_ids": [],
            "upper_bound_owner_ids": optimal,
            "eligible_owner_iou_receipts": all_receipts,
        }
    return {
        "strict_assignment_status": "matched",
        "strict_assignment_gt_owner_id": optimal[0],
        "ambiguity_owner_ids": [],
        "lower_bound_owner_ids": optimal,
        "upper_bound_owner_ids": optimal,
        "eligible_owner_iou_receipts": all_receipts,
    }


def build_sidecars(
    greedy: GreedyRollout,
    ledger_rows: Mapping[int, CanonicalLedgerRow],
    owners: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    *,
    object_ref_end: int,
    iou_threshold: float,
) -> list[dict[str, Any]]:
    """Every token-distinct complete native greedy ``person`` row (kite excluded).

    Binds row ID, raw tokens, and pixel box; reports both the canonical
    (joint, whole-rollout) strict-match verdict and an independent per-box
    IoU assignment against all 41 owners.  Rows that collapse to the same
    coordinate tokens as an earlier row (token-identical, not merely
    "duplicate-like") are folded into one sidecar with every contributing
    row index preserved.  Never enters primary ranks.
    """

    candidate_tokens: dict[tuple[int, ...], list[str]] = {}
    for candidate in candidates:
        candidate_tokens.setdefault(tuple(candidate["coord_token_ids"]), []).append(
            str(candidate["candidate_id"])
        )

    by_tokens: dict[tuple[int, ...], dict[str, Any]] = {}
    order: list[tuple[int, ...]] = []
    for row_index, prediction in enumerate(greedy.predictions):
        if prediction.get("description") != "person":
            continue
        raw_tokens = greedy.row_token_spans[row_index]
        coord_token_ids = tuple(_row_coordinates(raw_tokens, object_ref_end))
        ledger_row = ledger_rows[row_index]
        if coord_token_ids in by_tokens:
            by_tokens[coord_token_ids]["source_row_indices"].append(row_index)
            by_tokens[coord_token_ids]["source_raw_span_sha256s"].append(ledger_row.raw_span_sha256)
            continue
        order.append(coord_token_ids)
        by_tokens[coord_token_ids] = {
            "coord_token_ids": coord_token_ids,
            "first_row_index": row_index,
            "first_raw_tokens": raw_tokens,
            "first_prediction": prediction,
            "first_ledger_row": ledger_row,
            "source_row_indices": [row_index],
            "source_raw_span_sha256s": [ledger_row.raw_span_sha256],
        }

    rows: list[dict[str, Any]] = []
    for coord_token_ids in order:
        entry = by_tokens[coord_token_ids]
        row_index = entry["first_row_index"]
        prediction = entry["first_prediction"]
        ledger_row = entry["first_ledger_row"]
        raw_tokens = entry["first_raw_tokens"]
        bbox = tuple(_int_list(prediction.get("bbox"), f"sidecar row {row_index} bbox"))
        assignment = independent_owner_assignment(bbox, owners, iou_threshold=iou_threshold)
        members = candidate_tokens.get(coord_token_ids, [])
        rows.append(
            {
                "schema_version": CENSUS_SCHEMA_VERSION,
                "row_kind": "realized_box_sidecar",
                "sidecar_id": f"sidecar:sorted:greedy:0:7511:row-{row_index}",
                "source_kind": "native_greedy_person_row",
                "source_pred_row_id": f"{CANONICAL_GREEDY_ROW_ID_PREFIX}{row_index}",
                "source_row_indices": list(entry["source_row_indices"]),
                "source_raw_span_sha256s": list(entry["source_raw_span_sha256s"]),
                "description": prediction.get("description"),
                "coord_token_ids": list(coord_token_ids),
                "coord_token_ids_sha256": sha256_json(list(coord_token_ids)),
                "raw_generated_token_ids": list(raw_tokens),
                "raw_generated_token_ids_sha256": sha256_json(list(raw_tokens)),
                "raw_span_sha256": ledger_row.raw_span_sha256,
                "bbox_pixel_xyxy": list(bbox),
                "canonical_strict_match_status": ledger_row.strict_match_status,
                "canonical_strict_match_gt_owner_id": ledger_row.strict_match_gt_owner_id,
                "bank_member_candidate_ids": list(members),
                "requires_new_score_row": not bool(members),
                "excluded_from_primary_ranks": True,
                **assignment,
            }
        )
    if len({row["sidecar_id"] for row in rows}) != len(rows):
        raise BoundaryCensusContractError("sidecar IDs are not unique")
    return rows


def materialize_owner_boundary_map(roles: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for role in roles:
        rows.append(
            {
                "schema_version": CENSUS_SCHEMA_VERSION,
                "row_kind": "owner_boundary_mapping",
                "gt_owner_id": role["gt_owner_id"],
                "owner_sort_key_y1_x1": role["owner_sort_key_y1_x1"],
                "mapping_kind": role["mapping_kind"],
                "matched_row_index": role["matched_row_index"],
                "matched_row_key_y1_x1": role["matched_row_key_y1_x1"],
                "pre_boundary_context_id": _context_id(role["pre_boundary_index"]),
                "post_boundary_context_id": _context_id(role["post_boundary_index"]),
                "pre_boundary_index": role["pre_boundary_index"],
                "post_boundary_index": role["post_boundary_index"],
                "no_observed_overtake_before_stop": role["no_observed_overtake_before_stop"],
                "post_overtake_reversion_observed": role["post_overtake_reversion_observed"],
            }
        )
    return rows


def build_scoring_requests(
    contexts: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    sidecars: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Cartesian join of every boundary context against the reused 369-row bank,
    plus one sidecar join per boundary for every sidecar that is not already a
    token-identical member of the primary bank.

    Structural dedup: a boundary appears once in ``contexts`` regardless of
    how many owners map onto it, so it is joined against the candidate (and
    sidecar) bank exactly once per boundary here too.  Primary and sidecar
    request counts are always distinguishable via ``request_kind``.
    """

    requests: list[dict[str, Any]] = []
    for context in contexts:
        context_id = str(context["context_id"])
        for candidate in candidates:
            requests.append(
                {
                    "schema_version": CENSUS_SCHEMA_VERSION,
                    "row_kind": "scoring_request",
                    "request_id": f"score:{context_id}:{candidate['candidate_id']}",
                    "request_kind": "primary",
                    "context_id": context_id,
                    "candidate_id": candidate["candidate_id"],
                    "candidate_source": "primary-candidates.jsonl",
                    "raw_complete_box_likelihood": True,
                    "repetition_penalty": 1.0,
                }
            )
        for sidecar in sidecars:
            if not sidecar["requires_new_score_row"]:
                continue
            requests.append(
                {
                    "schema_version": CENSUS_SCHEMA_VERSION,
                    "row_kind": "scoring_request",
                    "request_id": f"score:{context_id}:{sidecar['sidecar_id']}",
                    "request_kind": "sidecar",
                    "context_id": context_id,
                    "sidecar_id": sidecar["sidecar_id"],
                    "candidate_source": "sidecars.jsonl",
                    "raw_complete_box_likelihood": True,
                    "repetition_penalty": 1.0,
                }
            )
    if len({row["request_id"] for row in requests}) != len(requests):
        raise BoundaryCensusContractError("scoring request IDs are not unique")
    new_row_sidecar_count = sum(1 for sidecar in sidecars if sidecar["requires_new_score_row"])
    expected = len(contexts) * (len(candidates) + new_row_sidecar_count)
    if len(requests) != expected:
        raise BoundaryCensusContractError("scoring request count does not match the Cartesian join")
    return requests


# ---------------------------------------------------------------------------
# Output materialization: create-or-identical, immutable receipt
# ---------------------------------------------------------------------------


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) + b"\n" for row in rows)


def _receipt_bytes(receipt: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes(receipt) + b"\n"


def _receipt_digest(receipt: Mapping[str, Any]) -> str:
    return sha256_json(
        {key: value for key, value in receipt.items() if key != "receipt_content_sha256"}
    )


def _materialize_output_bytes(
    *,
    contexts: Sequence[Mapping[str, Any]],
    owner_boundary_map: Sequence[Mapping[str, Any]],
    sidecars: Sequence[Mapping[str, Any]],
    requests: Sequence[Mapping[str, Any]],
    plan_v2: PlanV2Bundle,
    greedy: GreedyRollout,
    ledger_rows: Mapping[int, CanonicalLedgerRow],
    roles: Sequence[Mapping[str, Any]],
    monotonicity: Mapping[str, Any],
    sources: SourcePaths,
) -> dict[str, bytes]:
    files = {
        "owner-ledger.jsonl": plan_v2.owner_ledger_bytes,
        "primary-candidates.jsonl": plan_v2.primary_candidates_bytes,
        "contexts.jsonl": _jsonl_bytes(contexts),
        "owner-boundary-map.jsonl": _jsonl_bytes(owner_boundary_map),
        "sidecars.jsonl": _jsonl_bytes(sidecars),
        "scoring-requests.jsonl": _jsonl_bytes(requests),
    }
    request_counts: dict[str, int] = {}
    for request in requests:
        kind = str(request["request_kind"])
        request_counts[kind] = request_counts.get(kind, 0) + 1
    code_path = Path(__file__).resolve()
    strict_matched_emit_count = sum(
        1 for role in roles if role["mapping_kind"] == "strict_matched_emit"
    )
    spatial_overtake_count = sum(
        1 for role in roles if role["mapping_kind"] == "spatial_overtake"
    )
    no_overtake_count = sum(
        1 for role in roles if role["no_observed_overtake_before_stop"]
    )
    execution_receipts = {row.execution_receipt_content_sha256 for row in ledger_rows.values()}
    receipt: dict[str, Any] = {
        "schema_version": PLAN_V2_SCHEMA_VERSION,
        "unit_id": PLAN_V2_UNIT_ID,
        "plan_strategy": PLAN_STRATEGY,
        "boundary_census_schema_version": CENSUS_SCHEMA_VERSION,
        "execution_surface": "deterministic_cpu_planner_no_model_no_tokenizer_no_gpu",
        "claim_scope": (
            "descriptive_discovery_only: owner x observed-boundary candidate landscape; "
            "no causal or visual-absence claim."
        ),
        "lineage": {
            "plan_v2_dir": str(plan_v2.plan_dir),
            "plan_v2_receipt_content_sha256": plan_v2.receipt.get("receipt_content_sha256"),
            "plan_v2_output_file_digests": {
                "owner-ledger.jsonl": hashlib.sha256(plan_v2.owner_ledger_bytes).hexdigest(),
                "primary-candidates.jsonl": hashlib.sha256(
                    plan_v2.primary_candidates_bytes
                ).hexdigest(),
            },
            "note": (
                "plan-v2 is read-only lineage; this directory is a separate, "
                "non-overwriting sibling artifact."
            ),
        },
        "source_paths": {
            "plan_v2_dir": str(sources.plan_v2_dir),
            "greedy": str(sources.greedy),
            "prediction_ledger": str(sources.prediction_ledger),
        },
        "source_digests": {
            "greedy_rollout": sha256_file(sources.greedy),
            "prediction_ledger": sha256_file(sources.prediction_ledger),
        },
        "code": {"path": str(code_path), "sha256": sha256_file(code_path)},
        "greedy_rollout_binding": {
            "canonical_prediction_row_id_prefix": CANONICAL_GREEDY_ROW_ID_PREFIX,
            "total_complete_row_count": greedy.total_complete_row_count,
            "decode_mode": greedy.decode_mode,
            "repetition_penalty": greedy.repetition_penalty,
            "seed": greedy.seed,
            "stop_reason": greedy.stop_reason,
            "execution_receipt_content_sha256": next(iter(execution_receipts)),
            "person_token_ids": list(greedy.person_token_ids),
            "query_suffix_token_ids": list(contexts[0]["query_suffix_token_ids"]),
            "query_suffix_token_ids_sha256": contexts[0]["query_suffix_token_ids_sha256"],
        },
        "owner_role_summary": {
            "strict_matched_emit_count": strict_matched_emit_count,
            "spatial_overtake_count": spatial_overtake_count,
            "no_observed_overtake_before_stop_count": no_overtake_count,
        },
        "row_key_monotonicity": monotonicity,
        "counts": {
            "owners": EXPECTED_OWNER_COUNT,
            "primary_candidates": EXPECTED_CANDIDATE_COUNT,
            "contexts": len(contexts),
            "owner_boundary_map_rows": len(owner_boundary_map),
            "sidecars": len(sidecars),
            "scoring_requests": len(requests),
            "scoring_requests_by_kind": request_counts,
        },
        "compatibility": {
            "target_scorer": "scripts/research/score_sorted_all_person_route_landscape.py",
            "target_scorer_load_plan_status": "incompatible_missing_required_files",
            "missing_required_files": ["sampling-seeds.jsonl"],
            "reason": (
                "score_sorted_all_person_route_landscape.PLAN_FILE_NAMES hardcodes "
                "'sampling-seeds.jsonl' as a required file and load_plan() calls "
                "Path.is_file() on it unconditionally, before any schema_version/unit_id "
                "semantics are read. This planner has no prospective low-temperature-sampling "
                "need for a boundary census and does not fabricate placeholder content for that "
                "one file. sidecars.jsonl IS produced here (every token-distinct complete "
                "native greedy person row, kite excluded, bound with canonical strict-match "
                "plus an independent per-box IoU assignment) and its row_kind/sidecar_id/"
                "excluded_from_primary_ranks/requires_new_score_row shape matches plan-v2's own "
                "sidecar convention. Row-level shapes (context_id/candidate_id/request_id/"
                "request_kind joins, full_prefix_token_ids(_sha256), candidate_universe_join) "
                "are otherwise structurally aligned with the six-context plan the scorer "
                "expects, EXCEPT that this planner's full_prefix_token_ids ends at box_start "
                "(observed_self_prefix_token_ids + a fixed query_suffix_token_ids forcing "
                "object_ref_start/person/object_ref_end/box_start), matching "
                "score_complete_box_candidate's own contract that its prefill root logits are "
                "already x1's distribution; plan-v2's own full_prefix_token_ids ends one row-open "
                "short of that (at a prior box_end or the bare prompt) and does not. Separately, "
                "downstream analyze_sorted_all_person_route_landscape.py independently hardcodes "
                "the six-context root/self-due-*/skip-post-* registry and is not a target of this "
                "planner's compatibility claim."
            ),
        },
        "output_file_digests": {
            name: hashlib.sha256(content).hexdigest() for name, content in files.items()
        },
    }
    receipt["receipt_content_sha256"] = _receipt_digest(receipt)
    files["receipt.json"] = _receipt_bytes(receipt)
    return files


def _commit_create_or_identical(output_dir: Path, files: Mapping[str, bytes]) -> str:
    expected_names = set(files)
    if output_dir.exists():
        if not output_dir.is_dir():
            raise BoundaryCensusContractError(
                f"output path exists but is not a directory: {output_dir}"
            )
        existing_names = {path.name for path in output_dir.iterdir() if path.is_file()}
        if existing_names != expected_names:
            raise BoundaryCensusContractError(
                "output directory already exists with a foreign or partial file set"
            )
        for name, expected in files.items():
            if (output_dir / name).read_bytes() != expected:
                raise BoundaryCensusContractError(
                    "output directory already exists with non-identical content"
                )
        return "identical_existing_output"

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        for name, content in files.items():
            (temp_dir / name).write_bytes(content)
        os.replace(temp_dir, output_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return "created"


def build_sorted_all_person_greedy_boundary_census(
    output_dir: str | Path, *, sources: SourcePaths = DEFAULT_SOURCES
) -> dict[str, Any]:
    """Validate and publish the complete deterministic CPU boundary census."""

    plan_v2 = _load_plan_v2(sources.plan_v2_dir)
    greedy = _load_greedy_rollout(sources.greedy, plan_v2)
    ledger_rows = _load_canonical_greedy_ledger_rows(sources.prediction_ledger, greedy, plan_v2)

    owner_id_set = {str(row["gt_owner_id"]) for row in plan_v2.owner_rows}
    matched_owner_to_row = build_matched_owner_to_row(ledger_rows, owner_id_set)

    row_keys = [row_sort_key(prediction["bbox"]) for prediction in greedy.predictions]
    roles = compute_owner_boundary_roles(
        owners=plan_v2.owner_rows,
        matched_owner_to_row=matched_owner_to_row,
        row_keys=row_keys,
        total_complete_row_count=greedy.total_complete_row_count,
    )
    monotonicity = row_key_monotonicity(row_keys)

    contexts = build_boundary_contexts(greedy, ledger_rows)
    contexts = attach_owner_membership(contexts, roles)
    owner_boundary_map = materialize_owner_boundary_map(roles)
    if len(owner_boundary_map) != EXPECTED_OWNER_COUNT:
        raise BoundaryCensusContractError("owner-boundary map does not cover exactly 41 owners")

    iou_threshold = float(plan_v2.receipt["strict_matcher"]["iou_threshold"])
    object_ref_end = int(plan_v2.frozen_runtime_identity["wrapper_token_ids"]["object_ref_end"])
    sidecars = build_sidecars(
        greedy,
        ledger_rows,
        plan_v2.owner_rows,
        plan_v2.candidate_rows,
        object_ref_end=object_ref_end,
        iou_threshold=iou_threshold,
    )
    requests = build_scoring_requests(contexts, plan_v2.candidate_rows, sidecars)

    files = _materialize_output_bytes(
        contexts=contexts,
        owner_boundary_map=owner_boundary_map,
        sidecars=sidecars,
        requests=requests,
        plan_v2=plan_v2,
        greedy=greedy,
        ledger_rows=ledger_rows,
        roles=roles,
        monotonicity=monotonicity,
        sources=sources,
    )
    status = _commit_create_or_identical(Path(output_dir), files)
    return {
        "status": status,
        "output_dir": str(Path(output_dir)),
        "counts": {
            "owners": EXPECTED_OWNER_COUNT,
            "primary_candidates": EXPECTED_CANDIDATE_COUNT,
            "contexts": len(contexts),
            "owner_boundary_map_rows": len(owner_boundary_map),
            "sidecars": len(sidecars),
            "scoring_requests": len(requests),
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--plan-v2-dir", type=Path, default=PLAN_V2_DIR)
    parser.add_argument("--greedy-path", type=Path, default=GREEDY_PATH)
    parser.add_argument("--prediction-ledger-path", type=Path, default=PREDICTION_LEDGER_PATH)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = build_sorted_all_person_greedy_boundary_census(
        args.output_dir,
        sources=SourcePaths(
            plan_v2_dir=args.plan_v2_dir,
            greedy=args.greedy_path,
            prediction_ledger=args.prediction_ledger_path,
        ),
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
