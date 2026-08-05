#!/usr/bin/env python3
"""Freeze the CPU-only owner/role registry for the sorted false-negative
mechanism decomposition unit.

This is a metadata-only builder.  It reads the immutable Task-0 owner and
prediction-row ledgers, the predecessor cohort assignments, and the stored
greedy/sampled rollout artifacts, and validates and emits:

  * the exact twenty-four target/control owner registry declared by
    ``research/investigations/qwen3-vl-dense-enumeration/experiments/
    2026-08-02-sorted-false-negative-mechanism-decomposition/unit.md``;
  * the additional bound owners that serve only as physical controls or as
    the first-skip missed-owner/native-successor pair;
  * the representative image-``7511`` smoke role manifest: root/due-turn
    contexts, any explicitly supplied and reviewed collision-pair triples
    (the default collision-pair set is empty; see
    ``DEFAULT_SMOKE_COLLISION_PAIRS``), the first-skip pre/post pair, and a
    validated non-collision null-pair envelope.

No model, tokenizer, or GPU is touched.  Self-prefix token identity is
reconstructed only from the stored ``prompt_token_ids``/``generated_token_ids``
arrays in the rollout artifacts, split on the ``<|object_ref_start|>`` token
id, which is exact and requires no re-tokenization.  Every prefix role stores
its literal token-id list; the recorded ``token_ids_sha256`` is always
``sha256_json`` of that exact stored list (never of a nested digest payload),
so any consumer can independently recompute and verify it.

Every logical role is a separate registry row even when two roles share an
identical prefix-token digest; execution-level tensor deduplication is a
downstream scoring concern and is never collapsed here. In particular, when
two supplied collision pairs cut the same trajectory at the same row, their
``P`` baselines are still two distinct role rows even though both share one
execution tensor.

The registry is two layers, and the two are never merged:

  * ``mechanism_cohort`` is the exact 24-owner decision-bearing mechanism
    cohort; only these owner IDs may ever contribute to a mechanism
    disposition or a prevalence denominator.
  * ``context_control_registry`` is an auxiliary set of owners referenced
    only to construct or validate context for the cohort above: the
    first-skip missed-owner/native-successor pair and the predeclared
    physical-foil/null-pair owners. Nothing in this layer widens the
    24-owner cohort or its prevalence denominator.

Collision semantics: a collision-pair spec is arbitrary-role/versioned, not a
hardcoded image-7511 target -- each pair names its own *tested* owner ``C``
via ``target_gt_owner_id`` (there is no default). ``G`` (the covering-row
insertion) and ``F`` (the provenance-matched physical foil insertion) are
*inserted* physical-owner rows -- they are never the tested owner and never
appear as ``role["gt_owner_id"]``; they surface only as
``role["inserted_gt_owner_id"]`` on the covering/foil roles. The same
target/inserted split applies to every non-collision null pair: a null pair
must also name its own ``target_gt_owner_id`` explicitly, and every one of
its roles (baseline, covering, foil, and the target's later natural release)
keeps ``role["gt_owner_id"] == target_gt_owner_id``.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "sorted-fn-mechanism-registry.v2"
UNIT_ID = "2026-08-02-sorted-false-negative-mechanism-decomposition"
OBJECT_REF_START_TOKEN_ID = 151646

# -- Frozen twenty-four target/control owner registry (unit.md stratified
# -- exploratory cohort table). Order matches the source table.
TARGET_REGISTRY: tuple[dict[str, str], ...] = (
    {"gt_owner_id": "gt:7511:6", "image_id": "7511", "stratum": "image_7511_tiny_crowded_persons", "declared_role": "no_free_target", "expected_cohort": "no_free_spatial_support"},
    {"gt_owner_id": "gt:7511:11", "image_id": "7511", "stratum": "image_7511_tiny_crowded_persons", "declared_role": "no_free_target", "expected_cohort": "no_free_spatial_support"},
    {"gt_owner_id": "gt:7511:13", "image_id": "7511", "stratum": "image_7511_tiny_crowded_persons", "declared_role": "no_free_target", "expected_cohort": "no_free_spatial_support"},
    {"gt_owner_id": "gt:7511:41", "image_id": "7511", "stratum": "image_7511_tiny_crowded_persons", "declared_role": "no_free_target", "expected_cohort": "no_free_spatial_support"},
    {"gt_owner_id": "gt:7511:22", "image_id": "7511", "stratum": "image_7511_tiny_crowded_persons", "declared_role": "shared_control_strict_positive", "expected_cohort": "greedy_strict_present"},
    {"gt_owner_id": "gt:7511:17", "image_id": "7511", "stratum": "image_7511_tiny_crowded_persons", "declared_role": "shared_control_strict_rescue", "expected_cohort": "strict_rescued"},
    {"gt_owner_id": "gt:7511:26", "image_id": "7511", "stratum": "image_7511_tiny_crowded_persons", "declared_role": "shared_control_loose_only", "expected_cohort": "loose_only_b1"},
    {"gt_owner_id": "gt:1584:11", "image_id": "1584", "stratum": "image_1584_mid_size_persons", "declared_role": "no_free_target", "expected_cohort": "no_free_spatial_support"},
    {"gt_owner_id": "gt:1584:9", "image_id": "1584", "stratum": "image_1584_mid_size_persons", "declared_role": "no_free_target", "expected_cohort": "no_free_spatial_support"},
    {"gt_owner_id": "gt:1584:3", "image_id": "1584", "stratum": "image_1584_mid_size_persons", "declared_role": "strict_positive_control", "expected_cohort": "greedy_strict_present"},
    {"gt_owner_id": "gt:1584:8", "image_id": "1584", "stratum": "image_1584_mid_size_persons", "declared_role": "strict_rescue_control", "expected_cohort": "strict_rescued"},
    {"gt_owner_id": "gt:16228:33", "image_id": "16228", "stratum": "image_16228_person_extent_and_collision", "declared_role": "loose_or_collision_candidate", "expected_cohort": "loose_only_b1"},
    {"gt_owner_id": "gt:16228:30", "image_id": "16228", "stratum": "image_16228_person_extent_and_collision", "declared_role": "loose_or_collision_candidate", "expected_cohort": "loose_only_b1"},
    {"gt_owner_id": "gt:16228:32", "image_id": "16228", "stratum": "image_16228_person_extent_and_collision", "declared_role": "strict_positive_control", "expected_cohort": "greedy_strict_present"},
    {"gt_owner_id": "gt:16228:41", "image_id": "16228", "stratum": "image_16228_person_extent_and_collision", "declared_role": "strict_rescue_and_proposed_physical_foil", "expected_cohort": "strict_rescued"},
    {"gt_owner_id": "gt:2685:28", "image_id": "2685", "stratum": "image_2685_semantic_drift_and_collision", "declared_role": "no_free_candidate", "expected_cohort": "no_free_spatial_support"},
    {"gt_owner_id": "gt:2685:9", "image_id": "2685", "stratum": "image_2685_semantic_drift_and_collision", "declared_role": "no_free_candidate", "expected_cohort": "no_free_spatial_support"},
    {"gt_owner_id": "gt:2685:15", "image_id": "2685", "stratum": "image_2685_semantic_drift_and_collision", "declared_role": "loose_only_collision_candidate", "expected_cohort": "loose_only_b1"},
    {"gt_owner_id": "gt:2685:17", "image_id": "2685", "stratum": "image_2685_semantic_drift_and_collision", "declared_role": "strict_positive_control", "expected_cohort": "greedy_strict_present"},
    {"gt_owner_id": "gt:14038:42", "image_id": "14038", "stratum": "image_14038_annotation_neutral_books", "declared_role": "annotation_neutral", "expected_cohort": "strict_ambiguity_neutral"},
    {"gt_owner_id": "gt:14038:41", "image_id": "14038", "stratum": "image_14038_annotation_neutral_books", "declared_role": "annotation_neutral", "expected_cohort": "strict_ambiguity_neutral"},
    {"gt_owner_id": "gt:13923:0", "image_id": "13923", "stratum": "image_13923_wall_bowls", "declared_role": "wall_bowl_case_study", "expected_cohort": "no_free_spatial_support"},
    {"gt_owner_id": "gt:13923:3", "image_id": "13923", "stratum": "image_13923_wall_bowls", "declared_role": "wall_bowl_case_study", "expected_cohort": "no_free_spatial_support"},
    {"gt_owner_id": "gt:13923:4", "image_id": "13923", "stratum": "image_13923_wall_bowls", "declared_role": "wall_bowl_case_study", "expected_cohort": "no_free_spatial_support"},
)

# -- Owners bound outside the 24-target set: predeclared physical controls
# -- (never mechanism targets) and the first-skip missed-owner/successor pair.
BOUND_NON_TARGETS: tuple[dict[str, str], ...] = (
    {"gt_owner_id": "gt:7511:24", "image_id": "7511", "declared_role": "physical_control", "expected_cohort": "strict_rescued"},
    {"gt_owner_id": "gt:7511:29", "image_id": "7511", "declared_role": "physical_control", "expected_cohort": "strict_rescued"},
    {"gt_owner_id": "gt:16228:35", "image_id": "16228", "declared_role": "physical_control", "expected_cohort": "loose_only_b1"},
    {"gt_owner_id": "gt:16228:17", "image_id": "16228", "declared_role": "physical_control", "expected_cohort": "loose_only_b1"},
    {"gt_owner_id": "gt:2685:22", "image_id": "2685", "declared_role": "physical_control", "expected_cohort": "strict_rescued"},
    {"gt_owner_id": "gt:2685:18", "image_id": "2685", "declared_role": "physical_control", "expected_cohort": "loose_only_b1"},
    {"gt_owner_id": "gt:2685:26", "image_id": "2685", "declared_role": "physical_control", "expected_cohort": "no_free_spatial_support"},
    {"gt_owner_id": "gt:7511:1", "image_id": "7511", "declared_role": "first_skip_missed_owner", "expected_cohort": "no_free_spatial_support"},
    {"gt_owner_id": "gt:7511:2", "image_id": "7511", "declared_role": "first_skip_native_successor", "expected_cohort": "greedy_strict_present"},
)

# Cohorts whose only registered rows are sampled or absent; per unit.md such
# an owner may serve only as a sampled/oracle physical-foil contrast, never
# against a natural greedy covering row.
_SAMPLED_OR_ORACLE_ONLY_COHORTS = frozenset({"strict_rescued", "no_free_spatial_support"})

SMOKE_IMAGE_ID = "7511"
SMOKE_ROOT_DUE_TURN_TARGETS: tuple[str, ...] = ("gt:7511:22",)
SMOKE_ROOT_ONLY_TARGETS: tuple[str, ...] = ("gt:7511:26",)
SMOKE_ROOT_DUE_TURN_RESCUE_TARGETS: tuple[str, ...] = ("gt:7511:17",)

# A collision-pair spec is arbitrary-role/versioned, exactly like a null-pair
# spec: it names its own target_gt_owner_id, image_id, review_status, a
# baseline (decode_mode/seed/cut_before_pred_row_id), and covering/foil
# (pred_row_id/gt_owner_id) -- never a single hardcoded image-7511 target.
# See ``_collision_pair_roles`` for the exact schema and validation contract.
#
# The default collision-pair set is empty; no candidate proposed in earlier
# unit.md drafts is frozen here as a reviewed default (freezing one is a
# separate adjudication decision this builder does not make):
#
#   * seed-21011 (target gt:7511:26): its foil gt:7511:29 has a 69 px^2
#     positive intersection with the target, violating the foil's required
#     exact-zero-overlap contract -- still invalid.
#   * seed-21003 (target gt:7511:26): its covering owner gt:7511:22 is the
#     strict-matched owner of pred:sorted:greedy:0:7511:3, P's natural
#     successor row. Novelty is judged relative to literal P only (see
#     ``_assert_not_in_natural_pre_pass``): matching the natural successor is
#     the natural continuation itself, not a duplicate, so this pair is
#     *not* excluded on novelty grounds. It is still absent from the default
#     set only because no default has been affirmatively reviewed and
#     frozen, not because it is invalid.
#
# With no default pair, the registry proceeds with the collision contrast
# left unresolved rather than fabricating a replacement. A caller may still
# explicitly supply one or more reviewed pairs through the
# ``collision_pairs``/``--collision-pairs-plan`` override, where the same
# fail-fast geometry, description, provenance, and natural-pre-pass
# validation applies to each.
DEFAULT_SMOKE_COLLISION_PAIRS: tuple[dict[str, Any], ...] = ()

SMOKE_FIRST_SKIP = {
    "missed_gt_owner_id": "gt:7511:1",
    "successor_gt_owner_id": "gt:7511:2",
    "decode_mode": "greedy",
    "seed": 0,
}


class RegistryError(ValueError):
    """Raised before output when the proposed registry is not admissible."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RegistryError(f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise RegistryError(f"{label} must be an array")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise RegistryError(f"{label} must be a non-empty trimmed string")
    return value


def _resolved_file(path: str | Path, label: str) -> Path:
    try:
        resolved = Path(path).expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise RegistryError(f"{label} does not exist") from exc
    if not resolved.is_file():
        raise RegistryError(f"{label} must be a regular file")
    return resolved


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RegistryError(
                    f"{label} line {line_number} is not valid JSON"
                ) from exc
            rows.append(_mapping(record, f"{label} line {line_number}"))
    return rows


def _read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RegistryError(f"{label} is not valid JSON") from exc


# --------------------------------------------------------------------------
# Ledger loading
# --------------------------------------------------------------------------


def load_owner_index(path: Path) -> dict[str, dict[str, Any]]:
    rows = _read_jsonl(path, "owner ledger")
    index: dict[str, dict[str, Any]] = {}
    for row in rows:
        owner_id = _string(row.get("gt_owner_id"), "owner ledger row.gt_owner_id")
        if owner_id in index:
            raise RegistryError(f"owner ledger has duplicate gt_owner_id {owner_id!r}")
        index[owner_id] = row
    return index


def load_prediction_index(path: Path) -> dict[str, dict[str, Any]]:
    rows = _read_jsonl(path, "prediction row ledger")
    index: dict[str, dict[str, Any]] = {}
    for row in rows:
        pred_row_id = _string(row.get("pred_row_id"), "prediction row.pred_row_id")
        if pred_row_id in index:
            raise RegistryError(
                f"prediction row ledger has duplicate pred_row_id {pred_row_id!r}"
            )
        index[pred_row_id] = row
    return index


def load_cohort_index(path: Path) -> dict[str, dict[str, Any]]:
    rows = _read_jsonl(path, "cohort assignments")
    index: dict[str, dict[str, Any]] = {}
    for row in rows:
        owner_id = _string(row.get("gt_owner_id"), "cohort row.gt_owner_id")
        if owner_id in index:
            raise RegistryError(
                f"cohort assignments has duplicate gt_owner_id {owner_id!r}"
            )
        index[owner_id] = row
    return index


# --------------------------------------------------------------------------
# Frozen owner geometry: description, non-overlap, and scan-order checks
# --------------------------------------------------------------------------


def _owner_or_fail(
    owner_index: Mapping[str, dict[str, Any]], gt_owner_id: str, *, context: str
) -> dict[str, Any]:
    owner = owner_index.get(gt_owner_id)
    if owner is None:
        raise RegistryError(
            f"{context}: owner ledger is missing {gt_owner_id!r}; cannot verify "
            "frozen geometry, refusing to infer"
        )
    return owner


def _owner_bbox(owner: Mapping[str, Any], *, label: str) -> tuple[float, float, float, float]:
    bbox = _sequence(owner.get("bbox_xyxy"), f"{label}.bbox_xyxy")
    if len(bbox) != 4:
        raise RegistryError(f"{label}.bbox_xyxy must have exactly four coordinates")
    x1, y1, x2, y2 = (float(value) for value in bbox)
    return (x1, y1, x2, y2)


def _bbox_intersection_area(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    return iw * ih


def _assert_same_description(
    owner_index: Mapping[str, dict[str, Any]],
    *,
    target_gt_owner_id: str,
    other_gt_owner_id: str,
    other_role: str,
    context: str,
) -> None:
    target = _owner_or_fail(owner_index, target_gt_owner_id, context=context)
    other = _owner_or_fail(owner_index, other_gt_owner_id, context=context)
    if target.get("normalized_description") != other.get("normalized_description"):
        raise RegistryError(
            f"{context}: target {target_gt_owner_id!r} and {other_role} "
            f"{other_gt_owner_id!r} do not share a normalized_description"
        )


def _assert_non_overlapping(
    owner_index: Mapping[str, dict[str, Any]],
    *,
    target_gt_owner_id: str,
    foil_gt_owner_id: str,
    context: str,
) -> float:
    target = _owner_or_fail(owner_index, target_gt_owner_id, context=context)
    foil = _owner_or_fail(owner_index, foil_gt_owner_id, context=context)
    target_bbox = _owner_bbox(target, label=f"{context} target {target_gt_owner_id}")
    foil_bbox = _owner_bbox(foil, label=f"{context} foil {foil_gt_owner_id}")
    intersection = _bbox_intersection_area(target_bbox, foil_bbox)
    if intersection > 0.0:
        raise RegistryError(
            f"{context}: foil {foil_gt_owner_id!r} spatially overlaps target "
            f"{target_gt_owner_id!r} (intersection area {intersection}); a physical "
            "foil must be non-overlapping under frozen owner geometry"
        )
    return intersection


def _assert_positive_overlap(
    owner_index: Mapping[str, dict[str, Any]],
    *,
    target_gt_owner_id: str,
    covering_gt_owner_id: str,
    context: str,
) -> float:
    """The covering-owner contract requires G to spatially overlap C by some
    positive, explicitly recorded amount under frozen GT geometry. Any
    positive intersection is acceptable for the current exploratory
    contract; a non-overlapping G is rejected rather than silently admitted."""

    target = _owner_or_fail(owner_index, target_gt_owner_id, context=context)
    covering = _owner_or_fail(owner_index, covering_gt_owner_id, context=context)
    target_bbox = _owner_bbox(target, label=f"{context} target {target_gt_owner_id}")
    covering_bbox = _owner_bbox(covering, label=f"{context} covering {covering_gt_owner_id}")
    intersection = _bbox_intersection_area(target_bbox, covering_bbox)
    if intersection <= 0.0:
        raise RegistryError(
            f"{context}: covering owner {covering_gt_owner_id!r} does not spatially "
            f"overlap target {target_gt_owner_id!r} (intersection area {intersection}); "
            "the covering-owner contract requires a positive geometric overlap with C"
        )
    return intersection


def _assert_distinct_owners(*, context: str, **owners: str) -> None:
    values = list(owners.values())
    if len(set(values)) != len(values):
        raise RegistryError(
            f"{context}: owners must be pairwise distinct physical owners, got {owners}"
        )


def _scan_key(owner: Mapping[str, Any], *, label: str) -> tuple[float, float, int]:
    x1, y1, _x2, _y2 = _owner_bbox(owner, label=label)
    original_annotation_index = owner.get("original_annotation_index")
    if not isinstance(original_annotation_index, int) or isinstance(
        original_annotation_index, bool
    ):
        raise RegistryError(f"{label}.original_annotation_index must be an integer")
    return (y1, x1, original_annotation_index)


# --------------------------------------------------------------------------
# Rollout loading: exact self-prefix token reconstruction with no tokenizer
# --------------------------------------------------------------------------


def _split_row_chunks(generated_token_ids: Sequence[int]) -> list[list[int]]:
    boundaries = [
        index
        for index, token_id in enumerate(generated_token_ids)
        if token_id == OBJECT_REF_START_TOKEN_ID
    ]
    boundaries.append(len(generated_token_ids))
    return [
        list(generated_token_ids[boundaries[i] : boundaries[i + 1]])
        for i in range(len(boundaries) - 1)
    ]


def load_rollout_trajectories(paths: Sequence[Path]) -> dict[tuple[str, str, int], dict[str, Any]]:
    """Return {(image_id, decode_mode, seed): trajectory} from rollout files."""

    trajectories: dict[tuple[str, str, int], dict[str, Any]] = {}
    for path in paths:
        document = _mapping(_read_json(path, f"rollout artifact {path}"), f"rollout artifact {path}")
        artifact_sha256 = sha256_file(path)
        rollouts = _sequence(document.get("rollouts"), f"{path}.rollouts")
        for rollout in rollouts:
            rollout = _mapping(rollout, f"{path} rollout entry")
            image_id = str(rollout.get("image_id"))
            decode_mode = _string(rollout.get("decode_mode"), f"{path} rollout.decode_mode")
            seed = rollout.get("seed")
            if not isinstance(seed, int) or isinstance(seed, bool):
                raise RegistryError(f"{path} rollout for image {image_id} has non-integer seed")
            stop_reason = rollout.get("stop_reason")
            if stop_reason != "im_end":
                raise RegistryError(
                    "forced-continuation rows do not admit owners or prefixes: "
                    f"{path} rollout for image {image_id} seed {seed} has "
                    f"stop_reason={stop_reason!r}"
                )
            prompt_token_ids = [
                int(token)
                for token in _sequence(
                    rollout.get("prompt_token_ids"), f"{path} rollout.prompt_token_ids"
                )
            ]
            generated_token_ids = [
                int(token)
                for token in _sequence(
                    rollout.get("generated_token_ids"),
                    f"{path} rollout.generated_token_ids",
                )
            ]
            predictions_block = _mapping(
                rollout.get("predictions"), f"{path} rollout.predictions"
            )
            valid_rows = _sequence(
                predictions_block.get("predictions"),
                f"{path} rollout.predictions.predictions",
            )
            dropped_rows = _sequence(
                predictions_block.get("dropped_predictions", []),
                f"{path} rollout.predictions.dropped_predictions",
            )
            # The prediction-row ledger's original_row_index (and this builder's
            # pred_row_id numbering) spans the full object_ref_start-delimited
            # token stream, including parser-dropped rows: a dropped row still
            # consumes one full row-shaped token span. Both valid and dropped
            # rows carry a generated_order position in that shared stream.
            ordered_rows: dict[int, Mapping[str, Any]] = {}
            for row in valid_rows:
                row = _mapping(row, f"{path} prediction row")
                order = row.get("generated_order")
                if not isinstance(order, int) or isinstance(order, bool):
                    raise RegistryError(f"{path} prediction row.generated_order must be an integer")
                if order in ordered_rows:
                    raise RegistryError(
                        f"{path} rollout for image {image_id} seed {seed} has duplicate "
                        f"generated_order {order}"
                    )
                ordered_rows[order] = row
            for row in dropped_rows:
                row = _mapping(row, f"{path} dropped prediction row")
                order = row.get("generated_order")
                if not isinstance(order, int) or isinstance(order, bool):
                    raise RegistryError(
                        f"{path} dropped prediction row.generated_order must be an integer"
                    )
                if order in ordered_rows:
                    raise RegistryError(
                        f"{path} rollout for image {image_id} seed {seed} has duplicate "
                        f"generated_order {order}"
                    )
                ordered_rows[order] = row
            row_chunks = _split_row_chunks(generated_token_ids)
            if sorted(ordered_rows) != list(range(len(row_chunks))):
                raise RegistryError(
                    f"{path} rollout for image {image_id} seed {seed} has "
                    f"{len(row_chunks)} object_ref_start-delimited token chunks but its "
                    f"valid-plus-dropped prediction rows do not cover generated_order "
                    f"0..{len(row_chunks) - 1} exactly; row boundaries are ambiguous"
                )
            raw_span_sha256_by_row = [
                _string(ordered_rows[index].get("raw_span_sha256"), f"{path} prediction row.raw_span_sha256")
                for index in range(len(row_chunks))
            ]
            # The row's own predicted bbox (not the registered GT owner
            # bbox): collision is induced by the literal inserted prediction
            # row, whose extent may spill into or out of the target's GT
            # box even when its strict-matched physical owner would not.
            # Not every row is guaranteed a valid bbox (e.g. a
            # parser-dropped row may carry a degenerate one); capture
            # whatever is present and validate only where a row is actually
            # used as a covering/foil insertion.
            row_bbox_by_row: list[tuple[float, float, float, float] | None] = []
            for index in range(len(row_chunks)):
                raw_bbox = ordered_rows[index].get("bbox")
                if (
                    isinstance(raw_bbox, Sequence)
                    and not isinstance(raw_bbox, (str, bytes))
                    and len(raw_bbox) == 4
                ):
                    row_bbox_by_row.append(tuple(float(value) for value in raw_bbox))
                else:
                    row_bbox_by_row.append(None)
            key = (image_id, decode_mode, seed)
            if key in trajectories:
                raise RegistryError(
                    f"rollout artifacts declare duplicate trajectory {key!r}"
                )
            trajectories[key] = {
                "image_id": image_id,
                "decode_mode": decode_mode,
                "seed": seed,
                "prompt_token_ids": prompt_token_ids,
                "row_chunks": row_chunks,
                "raw_span_sha256_by_row": raw_span_sha256_by_row,
                "row_bbox_by_row": row_bbox_by_row,
                "pred_row_ids": [
                    f"pred:sorted:{decode_mode}:{seed}:{image_id}:{index}"
                    for index in range(len(row_chunks))
                ],
                "source_artifact_path": str(path),
                "source_artifact_sha256": artifact_sha256,
            }
    return trajectories


def _trajectory_for(
    trajectories: Mapping[tuple[str, str, int], dict[str, Any]],
    *,
    image_id: str,
    decode_mode: str,
    seed: int,
) -> dict[str, Any]:
    key = (image_id, decode_mode, seed)
    trajectory = trajectories.get(key)
    if trajectory is None:
        raise RegistryError(
            f"no stored rollout trajectory for image {image_id} decode_mode "
            f"{decode_mode!r} seed {seed}"
        )
    return trajectory


def _prefix_before_row_index(trajectory: Mapping[str, Any], cut_row_index: int) -> dict[str, Any]:
    row_chunks = trajectory["row_chunks"]
    if not 0 <= cut_row_index <= len(row_chunks):
        raise RegistryError(
            f"cut row index {cut_row_index} is out of range for trajectory with "
            f"{len(row_chunks)} rows"
        )
    token_ids = list(trajectory["prompt_token_ids"])
    for chunk in row_chunks[:cut_row_index]:
        token_ids.extend(chunk)
    return {
        "token_ids": token_ids,
        "token_ids_sha256": sha256_json(token_ids),
        "token_count": len(token_ids),
        "prefix_pred_row_ids": list(trajectory["pred_row_ids"][:cut_row_index]),
    }


def _row_index_for_pred_row_id(trajectory: Mapping[str, Any], pred_row_id: str) -> int:
    try:
        return trajectory["pred_row_ids"].index(pred_row_id)
    except ValueError as exc:
        raise RegistryError(
            f"pred_row_id {pred_row_id!r} is not present in trajectory "
            f"{trajectory['image_id']}/{trajectory['decode_mode']}/{trajectory['seed']}"
        ) from exc


def _inserted_row_bbox(
    trajectory: Mapping[str, Any], pred_row_id: str, *, context: str
) -> tuple[float, float, float, float]:
    row_index = _row_index_for_pred_row_id(trajectory, pred_row_id)
    bbox = trajectory["row_bbox_by_row"][row_index]
    if bbox is None:
        raise RegistryError(
            f"{context}: inserted row {pred_row_id!r} has no valid predicted bbox; "
            "cannot verify the actual inserted-row/target intersection, refusing to infer"
        )
    return bbox


def _assert_inserted_row_positive_overlap(
    trajectory: Mapping[str, Any],
    *,
    target_bbox: tuple[float, float, float, float],
    target_gt_owner_id: str,
    covering_pred_row_id: str,
    context: str,
) -> float:
    """Strict owner match alone is insufficient: the *literal inserted
    prediction row* for G may extend beyond (or fall short of) its
    strict-matched owner's registered GT box. Require the row's own
    predicted bbox to positively intersect the target's GT bbox."""

    covering_bbox = _inserted_row_bbox(trajectory, covering_pred_row_id, context=context)
    intersection = _bbox_intersection_area(target_bbox, covering_bbox)
    if intersection <= 0.0:
        raise RegistryError(
            f"{context}: inserted covering row {covering_pred_row_id!r} predicted bbox "
            f"does not spatially overlap target {target_gt_owner_id!r} "
            f"(intersection area {intersection}); the covering-row contract requires a "
            "positive geometric overlap with C's actual predicted extent, not merely its "
            "strict-matched owner"
        )
    return intersection


def _assert_inserted_row_non_overlapping(
    trajectory: Mapping[str, Any],
    *,
    target_bbox: tuple[float, float, float, float],
    target_gt_owner_id: str,
    foil_pred_row_id: str,
    context: str,
) -> float:
    foil_bbox = _inserted_row_bbox(trajectory, foil_pred_row_id, context=context)
    intersection = _bbox_intersection_area(target_bbox, foil_bbox)
    if intersection > 0.0:
        raise RegistryError(
            f"{context}: inserted foil row {foil_pred_row_id!r} predicted bbox spatially "
            f"overlaps target {target_gt_owner_id!r} (intersection area {intersection}); "
            "the foil-row contract requires exact zero overlap with C's actual predicted "
            "extent, not merely its strict-matched owner"
        )
    return intersection


def _append_row_prefix(
    baseline: Mapping[str, Any],
    trajectory: Mapping[str, Any],
    appended_pred_row_id: str,
) -> dict[str, Any]:
    row_index = _row_index_for_pred_row_id(trajectory, appended_pred_row_id)
    # The appended row token span is the row's own chunk from its own trajectory.
    appended_tokens = trajectory["row_chunks"][row_index]
    combined_token_ids = [*baseline["token_ids"], *appended_tokens]
    combined_pred_row_ids = [*baseline["prefix_pred_row_ids"], appended_pred_row_id]
    return {
        "token_ids": combined_token_ids,
        "token_ids_sha256": sha256_json(combined_token_ids),
        "token_count": len(combined_token_ids),
        "prefix_pred_row_ids": combined_pred_row_ids,
        "appended_pred_row_id": appended_pred_row_id,
        "appended_row_token_count": len(appended_tokens),
        "appended_row_raw_span_sha256": trajectory["raw_span_sha256_by_row"][row_index],
    }


# --------------------------------------------------------------------------
# Owner/role validation
# --------------------------------------------------------------------------


def _validate_owner_binding(
    entry: Mapping[str, str],
    *,
    owner_index: Mapping[str, dict[str, Any]],
    cohort_index: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    gt_owner_id = entry["gt_owner_id"]
    owner = owner_index.get(gt_owner_id)
    if owner is None:
        raise RegistryError(f"owner ledger is missing {gt_owner_id!r}")
    if str(owner.get("image_id")) != entry["image_id"]:
        raise RegistryError(
            f"{gt_owner_id} owner ledger image_id disagrees with registry table"
        )
    cohort_row = cohort_index.get(gt_owner_id)
    if cohort_row is None:
        raise RegistryError(f"cohort assignments is missing {gt_owner_id!r}")
    observed_cohort = cohort_row.get("cohort")
    if observed_cohort != entry["expected_cohort"]:
        raise RegistryError(
            f"{gt_owner_id} cohort is {observed_cohort!r}, expected "
            f"{entry['expected_cohort']!r} for declared_role {entry['declared_role']!r}"
        )
    foil_contrast = (
        "sampled_or_oracle_only"
        if observed_cohort in _SAMPLED_OR_ORACLE_ONLY_COHORTS
        else "natural_greedy_allowed"
    )
    return {
        **entry,
        "diagnostic_owner_id": f"diagnostic:{gt_owner_id}",
        "normalized_description": owner.get("normalized_description"),
        "bbox_xyxy": owner.get("bbox_xyxy"),
        "cohort": observed_cohort,
        "foil_contrast_allowed": foil_contrast,
        "supporting_pred_row_ids": sorted(
            _sequence(
                _mapping(cohort_row.get("foreign_keys"), f"{gt_owner_id} foreign_keys").get(
                    "supporting_pred_row_ids", []
                ),
                f"{gt_owner_id} supporting_pred_row_ids",
            )
        ),
    }


def build_target_registry(
    *, owner_index: Mapping[str, dict[str, Any]], cohort_index: Mapping[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    seen: set[str] = set()
    validated: list[dict[str, Any]] = []
    for entry in TARGET_REGISTRY:
        if entry["gt_owner_id"] in seen:
            raise RegistryError(f"duplicate target owner {entry['gt_owner_id']!r}")
        seen.add(entry["gt_owner_id"])
        validated.append(
            _validate_owner_binding(entry, owner_index=owner_index, cohort_index=cohort_index)
        )
    if len(validated) != 24:
        raise RegistryError(
            f"target registry must contain exactly 24 owners, got {len(validated)}"
        )
    return validated


def build_bound_non_targets(
    *, owner_index: Mapping[str, dict[str, Any]], cohort_index: Mapping[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    seen: set[str] = set()
    validated: list[dict[str, Any]] = []
    for entry in BOUND_NON_TARGETS:
        if entry["gt_owner_id"] in seen:
            raise RegistryError(f"duplicate bound non-target owner {entry['gt_owner_id']!r}")
        seen.add(entry["gt_owner_id"])
        validated.append(
            _validate_owner_binding(entry, owner_index=owner_index, cohort_index=cohort_index)
        )
    target_ids = {entry["gt_owner_id"] for entry in TARGET_REGISTRY}
    overlap = seen & target_ids
    if overlap:
        raise RegistryError(
            f"bound non-target owners overlap the 24-owner target registry: {sorted(overlap)}"
        )
    return validated


# --------------------------------------------------------------------------
# Smoke role manifest
# --------------------------------------------------------------------------


def _root_context_role(
    *, gt_owner_id: str, trajectory: Mapping[str, Any]
) -> dict[str, Any]:
    prefix = _prefix_before_row_index(trajectory, 0)
    return {
        "role_id": f"root:{gt_owner_id}",
        "role_kind": "root_context",
        "gt_owner_id": gt_owner_id,
        "trajectory": {
            "image_id": trajectory["image_id"],
            "decode_mode": trajectory["decode_mode"],
            "seed": trajectory["seed"],
        },
        "prefix": prefix,
        "provenance": {
            "source_artifact_path": trajectory["source_artifact_path"],
            "source_artifact_sha256": trajectory["source_artifact_sha256"],
        },
    }


def _due_turn_context_role(
    *,
    gt_owner_id: str,
    trajectory: Mapping[str, Any],
    prediction_index: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    matched_pred_row_id = None
    for pred_row_id in trajectory["pred_row_ids"]:
        prediction_row = prediction_index.get(pred_row_id)
        if prediction_row is None:
            continue
        if prediction_row.get("strict_match_gt_owner_id") == gt_owner_id:
            matched_pred_row_id = pred_row_id
            break
    if matched_pred_row_id is None:
        raise RegistryError(
            f"due-turn context for {gt_owner_id} requires a strict match in "
            f"trajectory {trajectory['image_id']}/{trajectory['decode_mode']}/"
            f"{trajectory['seed']}, but none was found"
        )
    cut_row_index = _row_index_for_pred_row_id(trajectory, matched_pred_row_id)
    prefix = _prefix_before_row_index(trajectory, cut_row_index)
    return {
        "role_id": f"due_turn:{gt_owner_id}",
        "role_kind": "due_turn_context",
        "gt_owner_id": gt_owner_id,
        "trajectory": {
            "image_id": trajectory["image_id"],
            "decode_mode": trajectory["decode_mode"],
            "seed": trajectory["seed"],
        },
        "reference_pred_row_id": matched_pred_row_id,
        "prefix": prefix,
        "provenance": {
            "source_artifact_path": trajectory["source_artifact_path"],
            "source_artifact_sha256": trajectory["source_artifact_sha256"],
        },
    }


def _assert_unique_strict_match(
    trajectory: Mapping[str, Any],
    prediction_index: Mapping[str, dict[str, Any]],
    *,
    gt_owner_id: str,
    expected_pred_row_id: str,
    role_label: str,
    context: str,
) -> int:
    """A sealed behavioral witness requires each of G/F/T to be exact and
    nonambiguous: exactly one row in the trajectory may strict-match the
    declared owner, and it must be the declared row. Returns that row's
    index."""

    matches = [
        pred_row_id
        for pred_row_id in trajectory["pred_row_ids"]
        if (row := prediction_index.get(pred_row_id)) is not None
        and row.get("strict_match_gt_owner_id") == gt_owner_id
    ]
    if matches != [expected_pred_row_id]:
        raise RegistryError(
            f"{context}: {role_label} owner {gt_owner_id!r} strict-matches "
            f"{matches!r} within the trajectory, not exactly the declared "
            f"{expected_pred_row_id!r}; an ambiguous binding is rejected"
        )
    return _row_index_for_pred_row_id(trajectory, expected_pred_row_id)


def _assert_not_in_natural_pre_pass(
    baseline_trajectory: Mapping[str, Any],
    prediction_index: Mapping[str, dict[str, Any]],
    *,
    cut_row_index: int,
    inserted_gt_owner_id: str,
    inserted_role: str,
    context: str,
) -> None:
    """Novelty is judged relative to literal P only: G/F must be a genuine
    counterfactual insertion, meaning neither may already be the
    strict-matched owner of any row *strictly before* the cut boundary (the
    rows that literally constitute P). The row exactly at the cut boundary
    is deliberately excluded from this scan: an inserted owner equal to that
    row's natural successor owner is the natural continuation itself, not a
    duplicate -- see ``_natural_successor_fact``, whose recorded witness
    facts make that case auditable rather than silently admitted."""

    pred_row_ids = baseline_trajectory["pred_row_ids"]
    for row_index in range(min(cut_row_index, len(pred_row_ids))):
        pred_row_id = pred_row_ids[row_index]
        row = prediction_index.get(pred_row_id)
        if row is None:
            continue
        if row.get("strict_match_gt_owner_id") == inserted_gt_owner_id:
            raise RegistryError(
                f"{context}: inserted {inserted_role} owner {inserted_gt_owner_id!r} is "
                f"already strict-matched by {pred_row_id!r} strictly inside P "
                "(before the cut boundary); it is not a genuine counterfactual insertion"
            )


def _natural_successor_fact(
    baseline_trajectory: Mapping[str, Any],
    prediction_index: Mapping[str, dict[str, Any]],
    *,
    cut_row_index: int,
) -> dict[str, Any]:
    """The row exactly at the cut boundary (P's natural successor, if any)
    and its strict-matched owner, recorded so a covering role's binding to
    that natural successor -- owner and/or exact row -- is explicit and
    auditable rather than inferred."""

    pred_row_ids = baseline_trajectory["pred_row_ids"]
    if cut_row_index >= len(pred_row_ids):
        return {"natural_successor_pred_row_id": None, "natural_successor_gt_owner_id": None}
    successor_pred_row_id = pred_row_ids[cut_row_index]
    successor_row = prediction_index.get(successor_pred_row_id)
    successor_owner = successor_row.get("strict_match_gt_owner_id") if successor_row else None
    return {
        "natural_successor_pred_row_id": successor_pred_row_id,
        "natural_successor_gt_owner_id": successor_owner,
    }


_REQUIRED_COLLISION_PAIR_REVIEW_STATUS = "reviewed"


def _collision_pair_roles(
    spec: Mapping[str, Any],
    *,
    trajectories: Mapping[tuple[str, str, int], dict[str, Any]],
    prediction_index: Mapping[str, dict[str, Any]],
    owner_index: Mapping[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Validate and build the P/P+G/P+F roles for one explicitly reviewed
    collision-pair spec. The spec shape mirrors a null-pair spec exactly
    (arbitrary target/image/baseline, not a hardcoded image-7511 target) and
    is validated against the same novelty, description, GT-and-row overlap,
    foil, and provenance contracts."""

    pair_id = _string(spec.get("pair_id"), "collision pair.pair_id")
    context = f"collision pair {pair_id}"
    image_id = _string(spec.get("image_id"), f"{context}.image_id")
    review_status = spec.get("review_status")
    if review_status != _REQUIRED_COLLISION_PAIR_REVIEW_STATUS:
        raise RegistryError(
            f"{context}.review_status must be explicitly "
            f"{_REQUIRED_COLLISION_PAIR_REVIEW_STATUS!r}, got {review_status!r}; a "
            "collision pair without an explicit reviewed-control status is rejected "
            "rather than assumed"
        )
    target_gt_owner_id = _string(spec.get("target_gt_owner_id"), f"{context}.target_gt_owner_id")
    baseline_spec = _mapping(spec.get("baseline"), f"{context}.baseline")
    covering_spec = _mapping(spec.get("covering"), f"{context}.covering")
    foil_spec = _mapping(spec.get("foil"), f"{context}.foil")
    covering_declared_owner_id = _string(
        covering_spec.get("gt_owner_id"), f"{context}.covering.gt_owner_id"
    )
    foil_declared_owner_id = _string(foil_spec.get("gt_owner_id"), f"{context}.foil.gt_owner_id")

    _assert_distinct_owners(
        context=context,
        target=target_gt_owner_id,
        inserted_covering=covering_declared_owner_id,
        inserted_foil=foil_declared_owner_id,
    )
    _assert_same_description(
        owner_index,
        target_gt_owner_id=target_gt_owner_id,
        other_gt_owner_id=covering_declared_owner_id,
        other_role="inserted covering owner",
        context=context,
    )
    _assert_same_description(
        owner_index,
        target_gt_owner_id=target_gt_owner_id,
        other_gt_owner_id=foil_declared_owner_id,
        other_role="inserted foil owner",
        context=context,
    )
    target_covering_owner_intersection_area = _assert_positive_overlap(
        owner_index,
        target_gt_owner_id=target_gt_owner_id,
        covering_gt_owner_id=covering_declared_owner_id,
        context=context,
    )
    target_foil_owner_intersection_area = _assert_non_overlapping(
        owner_index,
        target_gt_owner_id=target_gt_owner_id,
        foil_gt_owner_id=foil_declared_owner_id,
        context=context,
    )
    target_bbox = _owner_bbox(
        _owner_or_fail(owner_index, target_gt_owner_id, context=context),
        label=f"{context} target {target_gt_owner_id}",
    )

    baseline_trajectory = _trajectory_for(
        trajectories,
        image_id=image_id,
        decode_mode=_string(baseline_spec.get("decode_mode"), f"{context}.baseline.decode_mode"),
        seed=int(baseline_spec["seed"]),
    )
    cut_pred_row_id = _string(
        baseline_spec.get("cut_before_pred_row_id"),
        f"{context}.baseline.cut_before_pred_row_id",
    )
    baseline_cut_index = _row_index_for_pred_row_id(baseline_trajectory, cut_pred_row_id)
    _assert_not_in_natural_pre_pass(
        baseline_trajectory,
        prediction_index,
        cut_row_index=baseline_cut_index,
        inserted_gt_owner_id=covering_declared_owner_id,
        inserted_role="covering",
        context=context,
    )
    _assert_not_in_natural_pre_pass(
        baseline_trajectory,
        prediction_index,
        cut_row_index=baseline_cut_index,
        inserted_gt_owner_id=foil_declared_owner_id,
        inserted_role="foil",
        context=context,
    )
    successor_fact = _natural_successor_fact(
        baseline_trajectory, prediction_index, cut_row_index=baseline_cut_index
    )
    baseline_prefix = _prefix_before_row_index(baseline_trajectory, baseline_cut_index)
    baseline_role = {
        "role_id": f"collision:{pair_id}:P",
        "role_kind": "collision_baseline",
        "gt_owner_id": target_gt_owner_id,
        "trajectory": {
            "image_id": baseline_trajectory["image_id"],
            "decode_mode": baseline_trajectory["decode_mode"],
            "seed": baseline_trajectory["seed"],
        },
        "cut_before_pred_row_id": cut_pred_row_id,
        "prefix": baseline_prefix,
        "provenance": {
            "source_artifact_path": baseline_trajectory["source_artifact_path"],
            "source_artifact_sha256": baseline_trajectory["source_artifact_sha256"],
        },
    }

    covering_pred_row_id = _string(covering_spec.get("pred_row_id"), f"{context}.covering.pred_row_id")
    foil_pred_row_id = _string(foil_spec.get("pred_row_id"), f"{context}.foil.pred_row_id")
    covering_row = prediction_index.get(covering_pred_row_id)
    foil_row = prediction_index.get(foil_pred_row_id)
    if covering_row is None:
        raise RegistryError(f"{context}: prediction row ledger is missing covering row {covering_pred_row_id!r}")
    if foil_row is None:
        raise RegistryError(f"{context}: prediction row ledger is missing foil row {foil_pred_row_id!r}")
    if covering_row.get("strict_match_gt_owner_id") != covering_declared_owner_id:
        raise RegistryError(
            f"{context}: covering row {covering_pred_row_id!r} does not strict-match "
            f"declared owner {covering_declared_owner_id!r}"
        )
    if foil_row.get("strict_match_gt_owner_id") != foil_declared_owner_id:
        raise RegistryError(
            f"{context}: foil row {foil_pred_row_id!r} does not strict-match declared "
            f"owner {foil_declared_owner_id!r}"
        )
    if covering_row.get("normalized_description") != foil_row.get("normalized_description"):
        raise RegistryError(
            f"{context}: covering/foil rows do not share a normalized_description; "
            "a natural covering row cannot be paired with a GT-clean or "
            "mismatched-description foil"
        )
    if covering_row.get("decode_mode") != foil_row.get("decode_mode"):
        raise RegistryError(f"{context}: covering/foil rows are not provenance-matched: decode_mode differs")
    if covering_row.get("seed") != foil_row.get("seed"):
        raise RegistryError(f"{context}: covering/foil rows are not provenance-matched: seed differs")

    covering_trajectory = _trajectory_for(
        trajectories,
        image_id=image_id,
        decode_mode=_string(covering_row.get("decode_mode"), f"{context} covering.decode_mode"),
        seed=int(covering_row["seed"]),
    )
    foil_trajectory = _trajectory_for(
        trajectories,
        image_id=image_id,
        decode_mode=_string(foil_row.get("decode_mode"), f"{context} foil.decode_mode"),
        seed=int(foil_row["seed"]),
    )

    # Strict owner match alone is insufficient: the literal inserted
    # prediction row's extent may spill into or out of the target's GT box.
    # Validate the actual predicted bbox, not just the strict-matched owner.
    inserted_covering_row_intersection_area = _assert_inserted_row_positive_overlap(
        covering_trajectory,
        target_bbox=target_bbox,
        target_gt_owner_id=target_gt_owner_id,
        covering_pred_row_id=covering_pred_row_id,
        context=context,
    )
    inserted_foil_row_intersection_area = _assert_inserted_row_non_overlapping(
        foil_trajectory,
        target_bbox=target_bbox,
        target_gt_owner_id=target_gt_owner_id,
        foil_pred_row_id=foil_pred_row_id,
        context=context,
    )

    covering_prefix = _append_row_prefix(baseline_prefix, covering_trajectory, covering_pred_row_id)
    foil_prefix = _append_row_prefix(baseline_prefix, foil_trajectory, foil_pred_row_id)
    covering_provenance = {
        "source_artifact_path": covering_trajectory["source_artifact_path"],
        "source_artifact_sha256": covering_trajectory["source_artifact_sha256"],
    }
    foil_provenance = {
        "source_artifact_path": foil_trajectory["source_artifact_path"],
        "source_artifact_sha256": foil_trajectory["source_artifact_sha256"],
    }
    return [
        baseline_role,
        {
            "role_id": f"collision:{pair_id}:P+G",
            "role_kind": "collision_covering",
            "gt_owner_id": target_gt_owner_id,
            "inserted_gt_owner_id": covering_declared_owner_id,
            "pred_row_id": covering_pred_row_id,
            "raw_span_sha256": covering_prefix["appended_row_raw_span_sha256"],
            "target_covering_owner_intersection_area": target_covering_owner_intersection_area,
            "target_covering_row_intersection_area": inserted_covering_row_intersection_area,
            "natural_successor_pred_row_id": successor_fact["natural_successor_pred_row_id"],
            "natural_successor_gt_owner_id": successor_fact["natural_successor_gt_owner_id"],
            "covering_owner_equals_natural_successor_owner": (
                covering_declared_owner_id == successor_fact["natural_successor_gt_owner_id"]
            ),
            "covering_pred_row_id_equals_natural_successor_pred_row_id": (
                covering_pred_row_id == successor_fact["natural_successor_pred_row_id"]
            ),
            "trajectory": {
                "image_id": covering_trajectory["image_id"],
                "decode_mode": covering_trajectory["decode_mode"],
                "seed": covering_trajectory["seed"],
            },
            "prefix": covering_prefix,
            "provenance": covering_provenance,
        },
        {
            "role_id": f"collision:{pair_id}:P+F",
            "role_kind": "collision_foil",
            "gt_owner_id": target_gt_owner_id,
            "inserted_gt_owner_id": foil_declared_owner_id,
            "pred_row_id": foil_pred_row_id,
            "raw_span_sha256": foil_prefix["appended_row_raw_span_sha256"],
            "target_foil_owner_intersection_area": target_foil_owner_intersection_area,
            "target_foil_row_intersection_area": inserted_foil_row_intersection_area,
            "trajectory": {
                "image_id": foil_trajectory["image_id"],
                "decode_mode": foil_trajectory["decode_mode"],
                "seed": foil_trajectory["seed"],
            },
            "prefix": foil_prefix,
            "provenance": foil_provenance,
        },
    ]


def _first_skip_roles(
    *,
    trajectories: Mapping[tuple[str, str, int], dict[str, Any]],
    prediction_index: Mapping[str, dict[str, Any]],
    owner_index: Mapping[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    missed_gt_owner_id = SMOKE_FIRST_SKIP["missed_gt_owner_id"]
    successor_gt_owner_id = SMOKE_FIRST_SKIP["successor_gt_owner_id"]
    context = "first-skip pair"

    trajectory = _trajectory_for(
        trajectories,
        image_id=SMOKE_IMAGE_ID,
        decode_mode=SMOKE_FIRST_SKIP["decode_mode"],
        seed=SMOKE_FIRST_SKIP["seed"],
    )

    missed_owner = _owner_or_fail(owner_index, missed_gt_owner_id, context=context)
    successor_owner = _owner_or_fail(owner_index, successor_gt_owner_id, context=context)
    missed_scan_key = _scan_key(missed_owner, label=f"{context} missed owner {missed_gt_owner_id}")
    successor_scan_key = _scan_key(
        successor_owner, label=f"{context} successor owner {successor_gt_owner_id}"
    )
    if not missed_scan_key < successor_scan_key:
        raise RegistryError(
            f"{context}: {missed_gt_owner_id!r} does not precede {successor_gt_owner_id!r} "
            f"under the frozen (y1, x1, original_annotation_index) scan order "
            f"({missed_scan_key} is not < {successor_scan_key})"
        )

    successor_pred_row_id = None
    for pred_row_id in trajectory["pred_row_ids"]:
        row = prediction_index.get(pred_row_id)
        if row is not None and row.get("strict_match_gt_owner_id") == successor_gt_owner_id:
            successor_pred_row_id = pred_row_id
            break
    if successor_pred_row_id is None:
        raise RegistryError(
            f"{context}: no unambiguous native successor row for {successor_gt_owner_id!r}"
        )
    successor_row_index = _row_index_for_pred_row_id(trajectory, successor_pred_row_id)

    # Every valid mapped new-owner row from the start of the trajectory
    # through (and including) the successor row must be contamination-free:
    # no owner is strict-matched twice, the missed owner never appears, and
    # every owner's frozen scan key stays at or before the successor's --
    # otherwise the "clean" first-skip/immediate-successor claim is not
    # provable from immutable data and this must fail rather than assert
    # cleanliness.
    seen_owners: dict[str, int] = {}
    for row_index in range(successor_row_index + 1):
        pred_row_id = trajectory["pred_row_ids"][row_index]
        row = prediction_index.get(pred_row_id)
        if row is None:
            raise RegistryError(
                f"{context}: prediction row ledger is missing {pred_row_id!r}, required "
                "to verify a contamination-free transition prefix"
            )
        owner_id = row.get("strict_match_gt_owner_id")
        if owner_id is None:
            continue
        if owner_id == missed_gt_owner_id:
            raise RegistryError(
                f"{context}: {missed_gt_owner_id!r} unexpectedly has a natural strict "
                "match; it is not a clean first-skip case"
            )
        if owner_id in seen_owners:
            raise RegistryError(
                f"{context}: owner {owner_id!r} is strict-matched by both "
                f"{seen_owners[owner_id]!r} and {pred_row_id!r} before/at the "
                "successor row; the transition prefix is contaminated by a duplicate"
            )
        seen_owners[owner_id] = pred_row_id
        candidate_owner = _owner_or_fail(owner_index, owner_id, context=context)
        candidate_scan_key = _scan_key(candidate_owner, label=f"{context} row owner {owner_id}")
        if not candidate_scan_key <= successor_scan_key:
            raise RegistryError(
                f"{context}: owner {owner_id!r} covered at {pred_row_id!r} has scan key "
                f"{candidate_scan_key} after the successor's {successor_scan_key}; the "
                "sorted transition is not provably clean"
            )

    pre_prefix = _prefix_before_row_index(trajectory, successor_row_index)
    post_prefix = _prefix_before_row_index(trajectory, successor_row_index + 1)
    trajectory_summary = {
        "image_id": trajectory["image_id"],
        "decode_mode": trajectory["decode_mode"],
        "seed": trajectory["seed"],
    }
    provenance = {
        "source_artifact_path": trajectory["source_artifact_path"],
        "source_artifact_sha256": trajectory["source_artifact_sha256"],
    }
    return [
        {
            "role_id": "first_skip:P_pre",
            "role_kind": "first_skip_pre",
            "gt_owner_id": missed_gt_owner_id,
            "trajectory": trajectory_summary,
            "prefix": pre_prefix,
            "provenance": provenance,
        },
        {
            "role_id": "first_skip:P_post",
            "role_kind": "first_skip_post",
            "gt_owner_id": missed_gt_owner_id,
            "successor_gt_owner_id": successor_gt_owner_id,
            "successor_pred_row_id": successor_pred_row_id,
            "trajectory": trajectory_summary,
            "prefix": post_prefix,
            "provenance": provenance,
        },
    ]


# --------------------------------------------------------------------------
# Non-collision null-pair envelope
# --------------------------------------------------------------------------

_REQUIRED_NULL_PAIR_REVIEW_STATUS = "reviewed"


def _validate_null_pair(
    spec: Mapping[str, Any],
    *,
    trajectories: Mapping[tuple[str, str, int], dict[str, Any]],
    prediction_index: Mapping[str, dict[str, Any]],
    owner_index: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    """Validate one null-pair spec as a sealed behavioral witness, all drawn
    from a single natural trajectory (one decode_mode/seed/policy):

      * ``P`` is the exact prefix immediately before ``G`` -- not an
        independently declared cut point;
      * ``G`` is the row *at* that cut and strict-matches the covering owner
        (so ``P+G`` is exact witnessed history: the real natural
        continuation, not a counterfactual splice);
      * the target ``T`` is strict-matched at a later row in the same
        trajectory (``target_release_pred_row_id``) -- a null lacking this
        later release is rejected;
      * ``F`` is a distinct strict-matched same-description row from the
        same trajectory, novel in ``P`` (its row index must be after the
        cut, exactly like ``G``'s), with zero GT-and-actual-row overlap to
        ``T``;
      * ``G``, ``F``, and ``T`` are each required to be the *unique*
        strict-matched row for their declared owner in this trajectory --
        an ambiguous binding is rejected.
    """

    pair_id = _string(spec.get("pair_id"), "null pair.pair_id")
    context = f"null pair {pair_id}"
    image_id = _string(spec.get("image_id"), f"{context}.image_id")
    is_mechanical = bool(spec.get("is_mechanical", False))
    review_status = spec.get("review_status")
    if review_status != _REQUIRED_NULL_PAIR_REVIEW_STATUS:
        raise RegistryError(
            f"{context}.review_status must be explicitly "
            f"{_REQUIRED_NULL_PAIR_REVIEW_STATUS!r}, got {review_status!r}; a null pair "
            "without an explicit reviewed-control status is rejected rather than assumed"
        )
    target_gt_owner_id = _string(
        spec.get("target_gt_owner_id"), f"{context}.target_gt_owner_id"
    )
    baseline_spec = _mapping(spec.get("baseline"), f"{context}.baseline")
    covering_spec = _mapping(spec.get("covering"), f"{context}.covering")
    foil_spec = _mapping(spec.get("foil"), f"{context}.foil")
    covering_declared_owner_id = _string(
        covering_spec.get("gt_owner_id"), f"{context}.covering.gt_owner_id"
    )
    foil_declared_owner_id = _string(foil_spec.get("gt_owner_id"), f"{context}.foil.gt_owner_id")
    covering_pred_row_id = _string(covering_spec.get("pred_row_id"), f"{context}.covering.pred_row_id")
    foil_pred_row_id = _string(foil_spec.get("pred_row_id"), f"{context}.foil.pred_row_id")
    target_release_pred_row_id = _string(
        spec.get("target_release_pred_row_id"), f"{context}.target_release_pred_row_id"
    )

    _assert_distinct_owners(
        context=context,
        target=target_gt_owner_id,
        covering=covering_declared_owner_id,
        foil=foil_declared_owner_id,
    )
    _assert_same_description(
        owner_index,
        target_gt_owner_id=target_gt_owner_id,
        other_gt_owner_id=covering_declared_owner_id,
        other_role="covering owner",
        context=context,
    )
    _assert_same_description(
        owner_index,
        target_gt_owner_id=target_gt_owner_id,
        other_gt_owner_id=foil_declared_owner_id,
        other_role="foil owner",
        context=context,
    )
    target_covering_owner_intersection_area = _assert_positive_overlap(
        owner_index,
        target_gt_owner_id=target_gt_owner_id,
        covering_gt_owner_id=covering_declared_owner_id,
        context=context,
    )
    target_foil_owner_intersection_area = _assert_non_overlapping(
        owner_index,
        target_gt_owner_id=target_gt_owner_id,
        foil_gt_owner_id=foil_declared_owner_id,
        context=context,
    )
    target_bbox = _owner_bbox(
        _owner_or_fail(owner_index, target_gt_owner_id, context=context),
        label=f"{context} target {target_gt_owner_id}",
    )

    baseline_trajectory = _trajectory_for(
        trajectories,
        image_id=image_id,
        decode_mode=_string(baseline_spec.get("decode_mode"), f"{context}.baseline.decode_mode"),
        seed=int(baseline_spec["seed"]),
    )

    # G, F, and T must each be the unique strict match for their declared
    # owner within this one trajectory -- a sealed, nonambiguous witness.
    cut_row_index = _assert_unique_strict_match(
        baseline_trajectory,
        prediction_index,
        gt_owner_id=covering_declared_owner_id,
        expected_pred_row_id=covering_pred_row_id,
        role_label="covering",
        context=context,
    )
    foil_row_index = _assert_unique_strict_match(
        baseline_trajectory,
        prediction_index,
        gt_owner_id=foil_declared_owner_id,
        expected_pred_row_id=foil_pred_row_id,
        role_label="foil",
        context=context,
    )
    target_release_row_index = _assert_unique_strict_match(
        baseline_trajectory,
        prediction_index,
        gt_owner_id=target_gt_owner_id,
        expected_pred_row_id=target_release_pred_row_id,
        role_label="target",
        context=context,
    )
    if not foil_row_index > cut_row_index:
        raise RegistryError(
            f"{context}: foil row {foil_pred_row_id!r} (index {foil_row_index}) is not "
            f"novel in P; it must occur after G's cut position (index {cut_row_index})"
        )
    if not target_release_row_index > cut_row_index:
        raise RegistryError(
            f"{context}: target_release_pred_row_id {target_release_pred_row_id!r} "
            f"(index {target_release_row_index}) is not later than G's cut position "
            f"(index {cut_row_index}); a null lacking a later T release is rejected"
        )

    # P is the exact prefix immediately before G; G is the row at that cut,
    # so G is *always* its own natural successor here -- recorded explicitly
    # for symmetry/auditability with the collision-pair covering role.
    successor_fact = _natural_successor_fact(
        baseline_trajectory, prediction_index, cut_row_index=cut_row_index
    )
    baseline_prefix = _prefix_before_row_index(baseline_trajectory, cut_row_index)

    covering_row = prediction_index[covering_pred_row_id]
    foil_row = prediction_index[foil_pred_row_id]
    if covering_row.get("normalized_description") != foil_row.get("normalized_description"):
        raise RegistryError(
            f"{context}: covering/foil rows do not share a normalized_description"
        )

    # Strict owner match alone is insufficient: validate the literal foil
    # prediction row's own bbox against the target's GT bbox too. (G's row
    # bbox is not separately checked against T here: G is the real natural
    # row, already validated at the owner level, and the row-level
    # covering/target contract is about the *inserted* row in a
    # counterfactual splice, which G -- being witnessed history -- is not.)
    inserted_foil_row_intersection_area = _assert_inserted_row_non_overlapping(
        baseline_trajectory,
        target_bbox=target_bbox,
        target_gt_owner_id=target_gt_owner_id,
        foil_pred_row_id=foil_pred_row_id,
        context=context,
    )
    inserted_covering_row_intersection_area = _assert_inserted_row_positive_overlap(
        baseline_trajectory,
        target_bbox=target_bbox,
        target_gt_owner_id=target_gt_owner_id,
        covering_pred_row_id=covering_pred_row_id,
        context=context,
    )

    covering_prefix = _append_row_prefix(baseline_prefix, baseline_trajectory, covering_pred_row_id)
    foil_prefix = _append_row_prefix(baseline_prefix, baseline_trajectory, foil_pred_row_id)
    target_release_prefix = _prefix_before_row_index(baseline_trajectory, target_release_row_index + 1)

    provenance = {
        "source_artifact_path": baseline_trajectory["source_artifact_path"],
        "source_artifact_sha256": baseline_trajectory["source_artifact_sha256"],
    }
    trajectory_summary = {
        "image_id": baseline_trajectory["image_id"],
        "decode_mode": baseline_trajectory["decode_mode"],
        "seed": baseline_trajectory["seed"],
    }

    return {
        "pair_id": pair_id,
        "image_id": image_id,
        "is_mechanical": is_mechanical,
        "review_status": review_status,
        "target_gt_owner_id": target_gt_owner_id,
        "target_release_pred_row_id": target_release_pred_row_id,
        "roles": [
            {
                "role_id": f"null:{pair_id}:P_null",
                "role_kind": "mechanical_null_baseline" if is_mechanical else "non_collision_null_baseline",
                "gt_owner_id": target_gt_owner_id,
                "trajectory": trajectory_summary,
                "cut_before_pred_row_id": covering_pred_row_id,
                "prefix": baseline_prefix,
                "provenance": provenance,
            },
            {
                "role_id": f"null:{pair_id}:P_null+G_null",
                "role_kind": "mechanical_null_covering" if is_mechanical else "non_collision_null_covering",
                "gt_owner_id": target_gt_owner_id,
                "inserted_gt_owner_id": covering_declared_owner_id,
                "pred_row_id": covering_pred_row_id,
                "raw_span_sha256": covering_prefix["appended_row_raw_span_sha256"],
                "target_covering_owner_intersection_area": target_covering_owner_intersection_area,
                "target_covering_row_intersection_area": inserted_covering_row_intersection_area,
                "witnessed_history": True,
                "natural_successor_pred_row_id": successor_fact["natural_successor_pred_row_id"],
                "natural_successor_gt_owner_id": successor_fact["natural_successor_gt_owner_id"],
                "covering_owner_equals_natural_successor_owner": (
                    covering_declared_owner_id == successor_fact["natural_successor_gt_owner_id"]
                ),
                "covering_pred_row_id_equals_natural_successor_pred_row_id": (
                    covering_pred_row_id == successor_fact["natural_successor_pred_row_id"]
                ),
                "trajectory": trajectory_summary,
                "prefix": covering_prefix,
                "provenance": provenance,
            },
            {
                "role_id": f"null:{pair_id}:P_null+F_null",
                "role_kind": "mechanical_null_foil" if is_mechanical else "non_collision_null_foil",
                "gt_owner_id": target_gt_owner_id,
                "inserted_gt_owner_id": foil_declared_owner_id,
                "pred_row_id": foil_pred_row_id,
                "raw_span_sha256": foil_prefix["appended_row_raw_span_sha256"],
                "target_foil_owner_intersection_area": target_foil_owner_intersection_area,
                "target_foil_row_intersection_area": inserted_foil_row_intersection_area,
                "witnessed_history": False,
                "trajectory": trajectory_summary,
                "prefix": foil_prefix,
                "provenance": provenance,
            },
            {
                "role_id": f"null:{pair_id}:P_release",
                "role_kind": "mechanical_null_target_release" if is_mechanical else "non_collision_null_target_release",
                "gt_owner_id": target_gt_owner_id,
                "pred_row_id": target_release_pred_row_id,
                "trajectory": trajectory_summary,
                "prefix": target_release_prefix,
                "provenance": provenance,
            },
        ],
    }


def build_null_pair_envelope(
    null_pair_specs: Sequence[Mapping[str, Any]],
    *,
    trajectories: Mapping[tuple[str, str, int], dict[str, Any]],
    prediction_index: Mapping[str, dict[str, Any]],
    owner_index: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    validated_pairs = [
        _validate_null_pair(
            spec, trajectories=trajectories, prediction_index=prediction_index, owner_index=owner_index
        )
        for spec in null_pair_specs
    ]
    pair_ids = [pair["pair_id"] for pair in validated_pairs]
    if len(pair_ids) != len(set(pair_ids)):
        raise RegistryError("null pair envelope has duplicate pair_id values")
    mechanical_count = sum(1 for pair in validated_pairs if pair["is_mechanical"])
    calibrating_count = len(validated_pairs) - mechanical_count
    distinct_images = {pair["image_id"] for pair in validated_pairs}
    reasons: list[str] = []
    if len(validated_pairs) < 3:
        reasons.append(
            f"only {len(validated_pairs)} non-collision pairs registered; at least 3 required"
        )
    if len(distinct_images) < 2:
        reasons.append(
            f"non-collision pairs span only {len(distinct_images)} image(s)/strata; "
            "at least 2 required"
        )
    if mechanical_count < 1:
        reasons.append("no pair is flagged as the mechanical null check")
    if calibrating_count < 2:
        reasons.append(
            f"only {calibrating_count} calibrating (non-mechanical) pair(s) registered; "
            "at least 2 required because the mechanical pair cannot calibrate the envelope"
        )
    status = "sufficient" if not reasons else "insufficient"
    return {
        "status": status,
        "reasons": reasons,
        "pair_count": len(validated_pairs),
        "mechanical_pair_count": mechanical_count,
        "calibrating_pair_count": calibrating_count,
        "distinct_image_count": len(distinct_images),
        "pairs": validated_pairs,
    }


# --------------------------------------------------------------------------
# Top-level build
# --------------------------------------------------------------------------


def build_sorted_fn_mechanism_registry(
    *,
    owner_ledger: str | Path,
    prediction_row_ledger: str | Path,
    cohort_assignments: str | Path,
    rollouts: Sequence[str | Path],
    due_turn_trajectories: Mapping[str, tuple[str, int]] | None = None,
    collision_pairs: Sequence[Mapping[str, Any]] | None = None,
    null_pairs: Sequence[Mapping[str, Any]] | None = None,
    output: str | Path,
) -> dict[str, Any]:
    owner_ledger_path = _resolved_file(owner_ledger, "owner ledger")
    prediction_row_ledger_path = _resolved_file(
        prediction_row_ledger, "prediction row ledger"
    )
    cohort_assignments_path = _resolved_file(cohort_assignments, "cohort assignments")
    rollout_paths = [_resolved_file(path, "rollout artifact") for path in rollouts]
    due_turn_trajectories = dict(due_turn_trajectories or {})
    collision_pair_specs = (
        list(DEFAULT_SMOKE_COLLISION_PAIRS) if collision_pairs is None else list(collision_pairs)
    )
    null_pair_specs = list(null_pairs or [])

    owner_index = load_owner_index(owner_ledger_path)
    prediction_index = load_prediction_index(prediction_row_ledger_path)
    cohort_index = load_cohort_index(cohort_assignments_path)
    trajectories = load_rollout_trajectories(rollout_paths)

    targets = build_target_registry(owner_index=owner_index, cohort_index=cohort_index)
    bound_non_targets = build_bound_non_targets(
        owner_index=owner_index, cohort_index=cohort_index
    )

    smoke_roles: list[dict[str, Any]] = []

    def _trajectory_for_due_turn(gt_owner_id: str, default: tuple[str, int]) -> dict[str, Any]:
        decode_mode, seed = due_turn_trajectories.get(gt_owner_id, default)
        return _trajectory_for(
            trajectories, image_id=SMOKE_IMAGE_ID, decode_mode=decode_mode, seed=seed
        )

    for gt_owner_id in SMOKE_ROOT_DUE_TURN_TARGETS:
        greedy_trajectory = _trajectory_for(
            trajectories, image_id=SMOKE_IMAGE_ID, decode_mode="greedy", seed=0
        )
        smoke_roles.append(_root_context_role(gt_owner_id=gt_owner_id, trajectory=greedy_trajectory))
        smoke_roles.append(
            _due_turn_context_role(
                gt_owner_id=gt_owner_id,
                trajectory=_trajectory_for_due_turn(gt_owner_id, ("greedy", 0)),
                prediction_index=prediction_index,
            )
        )

    for gt_owner_id in SMOKE_ROOT_ONLY_TARGETS:
        greedy_trajectory = _trajectory_for(
            trajectories, image_id=SMOKE_IMAGE_ID, decode_mode="greedy", seed=0
        )
        smoke_roles.append(_root_context_role(gt_owner_id=gt_owner_id, trajectory=greedy_trajectory))

    for gt_owner_id in SMOKE_ROOT_DUE_TURN_RESCUE_TARGETS:
        greedy_trajectory = _trajectory_for(
            trajectories, image_id=SMOKE_IMAGE_ID, decode_mode="greedy", seed=0
        )
        smoke_roles.append(_root_context_role(gt_owner_id=gt_owner_id, trajectory=greedy_trajectory))
        if gt_owner_id not in due_turn_trajectories:
            raise RegistryError(
                f"{gt_owner_id} has no natural greedy strict match; its due-turn "
                "context requires an explicit --due-turn-trajectory declaration"
            )
        smoke_roles.append(
            _due_turn_context_role(
                gt_owner_id=gt_owner_id,
                trajectory=_trajectory_for_due_turn(gt_owner_id, ("greedy", 0)),
                prediction_index=prediction_index,
            )
        )

    for pair in collision_pair_specs:
        smoke_roles.extend(
            _collision_pair_roles(
                pair,
                trajectories=trajectories,
                prediction_index=prediction_index,
                owner_index=owner_index,
            )
        )

    smoke_roles.extend(
        _first_skip_roles(
            trajectories=trajectories, prediction_index=prediction_index, owner_index=owner_index
        )
    )

    null_envelope = build_null_pair_envelope(
        null_pair_specs,
        trajectories=trajectories,
        prediction_index=prediction_index,
        owner_index=owner_index,
    )

    role_ids = [role["role_id"] for role in smoke_roles]
    for pair in null_envelope["pairs"]:
        role_ids.extend(role["role_id"] for role in pair["roles"])
    if len(role_ids) != len(set(role_ids)):
        raise RegistryError("smoke role manifest has duplicate role_id values")

    content: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "sources": {
            "owner_ledger": {
                "path": str(owner_ledger_path),
                "sha256": sha256_file(owner_ledger_path),
            },
            "prediction_row_ledger": {
                "path": str(prediction_row_ledger_path),
                "sha256": sha256_file(prediction_row_ledger_path),
            },
            "cohort_assignments": {
                "path": str(cohort_assignments_path),
                "sha256": sha256_file(cohort_assignments_path),
            },
            "rollout_artifacts": [
                {"path": str(path), "sha256": sha256_file(path)} for path in rollout_paths
            ],
        },
        "mechanism_cohort": {
            "description": (
                "the exact 24-owner decision-bearing mechanism cohort; this is the "
                "only set that may contribute to a mechanism disposition or a "
                "prevalence denominator"
            ),
            "owner_count": len(targets),
            "prevalence_denominator_owner_ids": sorted(
                target["gt_owner_id"] for target in targets
            ),
            "targets": targets,
        },
        "context_control_registry": {
            "description": (
                "auxiliary owners referenced only to construct or validate context "
                "for the mechanism cohort (first-skip source/successor, physical "
                "foils, null-pair owners); never part of the 24-owner cohort or its "
                "prevalence denominator"
            ),
            "owner_count": len(bound_non_targets),
            "bound_non_targets": bound_non_targets,
        },
        "smoke": {
            "image_id": SMOKE_IMAGE_ID,
            "collision_target_gt_owner_ids": sorted(
                {pair["target_gt_owner_id"] for pair in collision_pair_specs}
            ),
            "roles": smoke_roles,
            "null_pair_envelope": null_envelope,
        },
    }
    document = {**content, "registry_digest": sha256_json(content)}

    output_path = Path(output).expanduser().resolve(strict=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_json_bytes(document) + b"\n"
    try:
        with output_path.open("xb") as handle:
            handle.write(encoded)
    except FileExistsError:
        if not output_path.is_file() or output_path.read_bytes() != encoded:
            raise RegistryError(
                "output already exists with different content; refusing to overwrite"
            ) from None
    return document


def _parse_due_turn_trajectory(value: str) -> tuple[str, tuple[str, int]]:
    try:
        owner, trajectory = value.split("=", 1)
        decode_mode, seed = trajectory.split(":", 1)
        return owner, (decode_mode, int(seed))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected OWNER=DECODE_MODE:SEED, e.g. gt:7511:17=sampled:21010"
        ) from exc


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner-ledger", required=True)
    parser.add_argument("--prediction-row-ledger", required=True)
    parser.add_argument("--cohort-assignments", required=True)
    parser.add_argument("--rollout", action="append", required=True, dest="rollouts")
    parser.add_argument(
        "--due-turn-trajectory",
        action="append",
        default=[],
        type=_parse_due_turn_trajectory,
        dest="due_turn_trajectories",
    )
    parser.add_argument(
        "--collision-pairs-plan",
        default=None,
        help=(
            "path to a JSON array overriding the default collision-pair set "
            "(empty by default); fail-fast geometry, description, provenance, "
            "and natural-pre-pass validation applies to any pair supplied this way"
        ),
    )
    parser.add_argument(
        "--null-pairs-plan",
        default=None,
        help="path to a JSON array of proposed non-collision null-pair specs",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    collision_pairs: list[Mapping[str, Any]] | None = None
    if args.collision_pairs_plan is not None:
        plan_path = _resolved_file(args.collision_pairs_plan, "collision pairs plan")
        collision_pairs = list(
            _sequence(_read_json(plan_path, "collision pairs plan"), "collision pairs plan")
        )
    null_pairs: list[Mapping[str, Any]] = []
    if args.null_pairs_plan is not None:
        plan_path = _resolved_file(args.null_pairs_plan, "null pairs plan")
        null_pairs = list(_sequence(_read_json(plan_path, "null pairs plan"), "null pairs plan"))
    document = build_sorted_fn_mechanism_registry(
        owner_ledger=args.owner_ledger,
        prediction_row_ledger=args.prediction_row_ledger,
        cohort_assignments=args.cohort_assignments,
        rollouts=args.rollouts,
        due_turn_trajectories=dict(args.due_turn_trajectories),
        collision_pairs=collision_pairs,
        null_pairs=null_pairs,
        output=args.output,
    )
    print(
        json.dumps(
            {
                "output": str(Path(args.output).expanduser().resolve(strict=False)),
                "registry_digest": document["registry_digest"],
                "target_count": document["mechanism_cohort"]["owner_count"],
                "bound_non_target_count": document["context_control_registry"]["owner_count"],
                "smoke_role_count": len(document["smoke"]["roles"]),
                "null_pair_envelope_status": document["smoke"]["null_pair_envelope"]["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
