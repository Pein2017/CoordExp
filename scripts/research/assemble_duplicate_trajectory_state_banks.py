#!/usr/bin/env python3
"""Assemble reviewed duplicate-trajectory StateBank drafts.

This is intentionally an experiment-local data factory.  It never discovers a
physical owner from overlap, decoded text, or a model prediction: the reviewed
owner ledger is the sole authority for admissibility.  The factory consumes
complete rows from one exact rollout, preserves every integer token slice, and
publishes the observed and duplicate-deleted trajectories separately.

The input contract is deliberately small and strict:

* each rollout trajectory supplies exact prompt and generated token IDs plus
  immutable generation settings;
* every complete generated row has one reviewed ledger row;
* ``accepted`` and ``recovery`` rows are usable only when all four review
  dimensions are ``trusted``;
* a ``duplicate`` row requires a reviewed ``burst_id`` and a preceding usable
  occurrence of the same image-local physical owner;
* ``unmatched``, ``uncertain``, and ``neutral`` rows never receive a gradient
  and stop automatic cleaned-suffix admission.

The public :func:`build_duplicate_trajectory_bank_drafts` result is JSON-safe
and is also the stable hand-off between this experiment-local assembler and
the typed StateBank core.  It contains five matched arms:

``source_preservation_only``
    The provided Source rows only.
``recovery_positive_only``
    Exact replay-prefix recovery positives without a duplicate negative.
``local_duplicate_rejection_and_recovery``
    Local complete-row positive/duplicate pairs at each observed duplicate
    state.
``duplicate_cleaned_imitation_only``
    Complete-row positives under prefixes created solely by deleting the
    reviewed duplicate burst.
``combined_duplicate_rejection_and_cleaned_imitation``
    Both mechanisms with one total unit of burst credit split equally between
    the two families.

The final StateBank adapter is deliberately narrow.  It maps these drafts to
the typed duplicate-event symbols in ``src.rollout_calibration`` only when a
caller supplies a valid source checkpoint and immutable Source-row StateBank
pairs.  The draft records preserve the exact information that adapter needs
without inventing source provenance: candidate-generation prefix hashes,
replay-prefix hashes, retained and removed integer row indices, and fixed
effective credit.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.inference.backend import token_ids_sha256  # noqa: E402


SCHEMA_VERSION = "duplicate_trajectory_state_bank_draft.v1"
ALLOCATION_RECEIPT_SCHEMA_VERSION = "duplicate_trajectory_allocation_receipt.v1"

OBJECT_REF_START = 151646
OBJECT_REF_END = 151647
BOX_START = 151648
BOX_END = 151649
COORDINATE_TOKEN_START = 151670
COORDINATE_TOKEN_END = 152670

TRUSTED = "trusted"
_TRUST_DIMENSIONS = (
    "entity_review_status",
    "category_review_status",
    "geometry_review_status",
    "binding_review_status",
)
_USABLE_ROLES = frozenset({"accepted", "recovery"})
_NEUTRAL_ROLES = frozenset({"neutral", "unmatched", "uncertain"})
_LEDGER_ROLES = _USABLE_ROLES | _NEUTRAL_ROLES | frozenset({"duplicate"})

SOURCE_ONLY = "source_preservation_only"
POSITIVE_ONLY = "recovery_positive_only"
LOCAL = "local_duplicate_rejection_and_recovery"
CLEANED = "duplicate_cleaned_imitation_only"
COMBINED = "combined_duplicate_rejection_and_cleaned_imitation"
BANK_NAMES = (SOURCE_ONLY, POSITIVE_ONLY, LOCAL, CLEANED, COMBINED)


class AssemblyError(ValueError):
    """Raised when immutable rollout or review evidence is insufficient."""


@dataclass(frozen=True)
class ExactRow:
    """One complete row and its exact observed prefix."""

    row_index: int
    start_token_offset: int
    end_token_offset: int
    token_ids: tuple[int, ...]
    prefix_token_ids: tuple[int, ...]

    @property
    def token_ids_sha256(self) -> str:
        return token_ids_sha256(self.token_ids)

    @property
    def prefix_token_ids_sha256(self) -> str:
        return token_ids_sha256(self.prefix_token_ids)

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "row_index": self.row_index,
            "start_token_offset": self.start_token_offset,
            "end_token_offset": self.end_token_offset,
            "token_ids": list(self.token_ids),
            "token_ids_sha256": self.token_ids_sha256,
            "prefix_token_ids": list(self.prefix_token_ids),
            "prefix_token_ids_sha256": self.prefix_token_ids_sha256,
        }


@dataclass(frozen=True)
class ReviewedRow:
    """One ledger declaration; no scientific label is inferred here."""

    trajectory_id: str
    row_index: int
    physical_owner_id: str | None
    review_role: str
    usable_owner_row: bool
    burst_id: str | None
    category: str | None
    review_provenance: Mapping[str, Any]

    @property
    def trusted_usable(self) -> bool:
        return self.review_role in _USABLE_ROLES and self.usable_owner_row


@dataclass(frozen=True)
class ReviewedBurst:
    """A reviewed run of duplicate rows and its first recovery row."""

    trajectory_id: str
    image_id: str
    burst_id: str
    owner_id: str
    first_owner_row_index: int
    duplicate_row_indices: tuple[int, ...]
    recovery_row_index: int
    cleaned_suffix_row_indices: tuple[int, ...]

    @property
    def key(self) -> tuple[str, str]:
        return self.trajectory_id, self.burst_id


@dataclass(frozen=True)
class ParsedTrajectory:
    trajectory_id: str
    image_id: str
    split: str
    image: Mapping[str, Any]
    physical_entities: tuple[Mapping[str, Any], ...]
    executed_prompt_token_ids: tuple[int, ...]
    image_pad_interval: tuple[int, int]
    generation_provenance: Mapping[str, Any]
    generated_token_ids: tuple[int, ...]
    rows: tuple[ExactRow, ...]

    @property
    def prompt_token_ids_sha256(self) -> str:
        return token_ids_sha256(self.executed_prompt_token_ids)

    @property
    def generated_token_ids_sha256(self) -> str:
        return token_ids_sha256(self.generated_token_ids)


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AssemblyError(f"{field} must be an object")
    return value


def _sequence(value: Any, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise AssemblyError(f"{field} must be an array")
    return value


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AssemblyError(f"{field} must be a non-empty string")
    return value.strip()


def _string_allow_empty(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise AssemblyError(f"{field} must be a string")
    return value


def _optional_string(value: Any, field: str) -> str | None:
    return None if value is None else _string(value, field)


def _int(value: Any, field: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AssemblyError(f"{field} must be an integer")
    if minimum is not None and value < minimum:
        raise AssemblyError(f"{field} must be >= {minimum}")
    return value


def _bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise AssemblyError(f"{field} must be a boolean")
    return value


def _token_ids(value: Any, field: str, *, allow_empty: bool = False) -> tuple[int, ...]:
    values = _sequence(value, field)
    result = tuple(
        _int(item, f"{field}[{index}]", minimum=0) for index, item in enumerate(values)
    )
    if not result and not allow_empty:
        raise AssemblyError(f"{field} must not be empty")
    return result


def _sha256(value: Any, field: str) -> str:
    text = _string(value, field)
    if len(text) != 64:
        raise AssemblyError(f"{field} must be a SHA-256 digest")
    try:
        int(text, 16)
    except ValueError as exc:
        raise AssemblyError(f"{field} must be a SHA-256 digest") from exc
    return text.lower()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _clone(value: Any) -> Any:
    """Return an immutable-JSON-style deep copy without custom object sharing."""

    return json.loads(_canonical(value))


def _image_id(value: Any, field: str) -> str:
    if isinstance(value, bool):
        raise AssemblyError(f"{field} must be an image ID")
    try:
        return str(int(str(value)))
    except ValueError as exc:
        raise AssemblyError(f"{field} must be an image ID") from exc


def _validate_complete_row(token_ids: Sequence[int], *, field: str) -> None:
    values = tuple(token_ids)
    if len(values) < 9 or values[0] != OBJECT_REF_START or values[-1] != BOX_END:
        raise AssemblyError(f"{field} is not a complete canonical object row")
    if any(
        values.count(marker) != 1
        for marker in (OBJECT_REF_START, OBJECT_REF_END, BOX_START, BOX_END)
    ):
        raise AssemblyError(f"{field} has duplicate or crossing row markers")
    try:
        description_end = values.index(OBJECT_REF_END)
        box_start = values.index(BOX_START)
    except ValueError as exc:  # guarded by the count check, retained for clarity
        raise AssemblyError(f"{field} lacks canonical row markers") from exc
    if not 0 < description_end < box_start < len(values) - 5:
        raise AssemblyError(f"{field} has an invalid description or box boundary")
    coordinates = values[box_start + 1 : -1]
    if len(coordinates) != 4 or any(
        token < COORDINATE_TOKEN_START or token >= COORDINATE_TOKEN_END
        for token in coordinates
    ):
        raise AssemblyError(f"{field} must contain exactly four coordinate token IDs")


def exact_complete_rows(generated_token_ids: Sequence[int]) -> tuple[ExactRow, ...]:
    """Split exact rollout IDs into complete rows without decoding text.

    A non-row token, a partial final row, or a cross-row marker is rejected.
    This guards the only operation that creates a cleaned counterfactual: row
    deletion remains an integer-slice operation rather than a retokenization.
    """

    values = _token_ids(generated_token_ids, "generated_token_ids")
    rows: list[ExactRow] = []
    start = 0
    for offset, token in enumerate(values):
        if token != BOX_END:
            continue
        row = values[start : offset + 1]
        _validate_complete_row(row, field=f"generated_token_ids row {len(rows)}")
        rows.append(
            ExactRow(
                row_index=len(rows),
                start_token_offset=start,
                end_token_offset=offset + 1,
                token_ids=row,
                prefix_token_ids=values[:start],
            )
        )
        start = offset + 1
    if not rows:
        raise AssemblyError("generated_token_ids contains no complete object rows")
    if start != len(values):
        raise AssemblyError(
            "generated_token_ids ends with a partial or non-row token span"
        )
    return tuple(rows)


def exact_row_slices(
    generated_token_ids: Sequence[int], row_index: int
) -> tuple[list[int], list[int]]:
    """Return the exact observed prefix and complete row for ``row_index``."""

    index = _int(row_index, "row_index", minimum=0)
    rows = exact_complete_rows(generated_token_ids)
    if index >= len(rows):
        raise AssemblyError(f"row_index {index} exceeds complete row count {len(rows)}")
    row = rows[index]
    return list(row.prefix_token_ids), list(row.token_ids)


def exact_row_site_types(row_token_ids: Sequence[int]) -> list[dict[str, Any]]:
    """Classify an exact complete row without decoding or retokenizing it."""

    values = _token_ids(row_token_ids, "row_token_ids")
    _validate_complete_row(values, field="row_token_ids")
    description_end = values.index(OBJECT_REF_END)
    box_start = values.index(BOX_START)
    sites: list[dict[str, Any]] = []
    for offset, _token in enumerate(values):
        if offset in {0, description_end, box_start, len(values) - 1}:
            token_type = "schema"
        elif box_start < offset < len(values) - 1:
            token_type = "coordinate"
        else:
            token_type = "desc_text"
        sites.append(
            {"candidate_token_offset": offset, "intended_token_type": token_type}
        )
    return sites


def _parse_generation_provenance(
    value: Any, *, prompt_hash: str, field: str
) -> dict[str, Any]:
    """Validate immutable settings shared by each row of a rollout trajectory."""

    source = _mapping(value, field)
    expected = {
        "mode",
        "seed",
        "temperature",
        "top_p",
        "repetition_penalty",
        "checkpoint_id",
        "prompt_token_ids_sha256",
    }
    unknown = set(source) - expected
    missing = expected - set(source)
    if missing or unknown:
        raise AssemblyError(
            f"{field} has missing={sorted(missing)} unknown={sorted(unknown)} fields"
        )
    mode = _string(source["mode"], f"{field}.mode")
    if mode not in {"greedy", "sampled"}:
        raise AssemblyError(f"{field}.mode must be greedy or sampled")
    seed = _int(source["seed"], f"{field}.seed", minimum=0)
    number_fields = ("temperature", "top_p", "repetition_penalty")
    parsed_numbers: dict[str, float] = {}
    for name in number_fields:
        raw = source[name]
        if (
            isinstance(raw, bool)
            or not isinstance(raw, (int, float))
            or not math.isfinite(float(raw))
        ):
            raise AssemblyError(f"{field}.{name} must be finite")
        parsed_numbers[name] = float(raw)
    if parsed_numbers["top_p"] <= 0.0 or parsed_numbers["repetition_penalty"] <= 0.0:
        raise AssemblyError(f"{field} top_p and repetition_penalty must be positive")
    if mode == "greedy" and parsed_numbers["temperature"] != 0.0:
        raise AssemblyError(f"{field} greedy provenance requires temperature zero")
    if mode == "sampled" and parsed_numbers["temperature"] <= 0.0:
        raise AssemblyError(f"{field} sampled provenance requires positive temperature")
    observed_prompt_hash = _sha256(
        source["prompt_token_ids_sha256"], f"{field}.prompt_token_ids_sha256"
    )
    if observed_prompt_hash != prompt_hash:
        raise AssemblyError(
            f"{field}.prompt_token_ids_sha256 disagrees with exact prompt IDs"
        )
    return {
        "mode": mode,
        "seed": seed,
        **parsed_numbers,
        "checkpoint_id": _sha256(source["checkpoint_id"], f"{field}.checkpoint_id"),
        "prompt_token_ids_sha256": observed_prompt_hash,
    }


def _parse_trajectory(value: Mapping[str, Any], *, index: int) -> ParsedTrajectory:
    field = f"rollout_trajectories[{index}]"
    trajectory_id = _string(value.get("trajectory_id"), f"{field}.trajectory_id")
    image_id = _image_id(value.get("image_id"), f"{field}.image_id")
    split = _string(value.get("split", "train"), f"{field}.split")
    if split not in {"train", "eval"}:
        raise AssemblyError(f"{field}.split must be train or eval")
    prompt = _token_ids(
        value.get("executed_prompt_token_ids"), f"{field}.executed_prompt_token_ids"
    )
    prompt_hash = _sha256(
        value.get("executed_prompt_token_ids_sha256"),
        f"{field}.executed_prompt_token_ids_sha256",
    )
    if prompt_hash != token_ids_sha256(prompt):
        raise AssemblyError(f"{field} prompt token hash does not match exact IDs")
    generated = _token_ids(
        value.get("generated_token_ids"), f"{field}.generated_token_ids"
    )
    generated_hash = _sha256(
        value.get("generated_token_ids_sha256"), f"{field}.generated_token_ids_sha256"
    )
    if generated_hash != token_ids_sha256(generated):
        raise AssemblyError(f"{field} generated token hash does not match exact IDs")
    pad = _sequence(value.get("image_pad_interval"), f"{field}.image_pad_interval")
    if len(pad) != 2:
        raise AssemblyError(f"{field}.image_pad_interval must have two offsets")
    start = _int(pad[0], f"{field}.image_pad_interval[0]", minimum=0)
    end = _int(pad[1], f"{field}.image_pad_interval[1]", minimum=0)
    if not start < end <= len(prompt):
        raise AssemblyError(f"{field}.image_pad_interval lies outside prompt IDs")
    image = _clone(
        _mapping(value.get("image", {"image_id": image_id}), f"{field}.image")
    )
    declared_image_id = image.get("image_id")
    if (
        declared_image_id is not None
        and _image_id(declared_image_id, f"{field}.image.image_id") != image_id
    ):
        raise AssemblyError(
            f"{field}.image.image_id disagrees with trajectory image_id"
        )
    image.setdefault("image_id", int(image_id))
    generation = _parse_generation_provenance(
        value.get("generation_provenance"),
        prompt_hash=prompt_hash,
        field=f"{field}.generation_provenance",
    )
    return ParsedTrajectory(
        trajectory_id=trajectory_id,
        image_id=image_id,
        split=split,
        image=image,
        physical_entities=tuple(
            _clone(_mapping(item, f"{field}.physical_entities[{entity_index}]"))
            for entity_index, item in enumerate(
                _sequence(
                    value.get("physical_entities", ()), f"{field}.physical_entities"
                )
            )
        ),
        executed_prompt_token_ids=prompt,
        image_pad_interval=(start, end),
        generation_provenance=generation,
        generated_token_ids=generated,
        rows=exact_complete_rows(generated),
    )


def _parse_reviewed_ledger(
    rows: Sequence[Mapping[str, Any]], *, trajectories: Mapping[str, ParsedTrajectory]
) -> dict[str, dict[int, ReviewedRow]]:
    """Parse row-level review decisions without deriving any owner semantics."""

    result: dict[str, dict[int, ReviewedRow]] = defaultdict(dict)
    for index, raw in enumerate(rows):
        field = f"reviewed_owner_ledger[{index}]"
        value = _mapping(raw, field)
        trajectory_id = _string(value.get("trajectory_id"), f"{field}.trajectory_id")
        trajectory = trajectories.get(trajectory_id)
        if trajectory is None:
            raise AssemblyError(
                f"{field} references an unknown trajectory {trajectory_id!r}"
            )
        row_index = _int(value.get("row_index"), f"{field}.row_index", minimum=0)
        if row_index >= len(trajectory.rows):
            raise AssemblyError(f"{field}.row_index exceeds exact rollout rows")
        if row_index in result[trajectory_id]:
            raise AssemblyError(
                f"{field} duplicates review for trajectory row {row_index}"
            )
        role = _string(value.get("review_role"), f"{field}.review_role")
        if role not in _LEDGER_ROLES:
            raise AssemblyError(
                f"{field}.review_role must be one of {sorted(_LEDGER_ROLES)}"
            )
        owner = _optional_string(
            value.get("physical_owner_id"), f"{field}.physical_owner_id"
        )
        usable = _bool(value.get("usable_owner_row"), f"{field}.usable_owner_row")
        burst_id = _optional_string(value.get("burst_id"), f"{field}.burst_id")
        category = _optional_string(value.get("category"), f"{field}.category")
        dimensions = {
            name: _string(value.get(name), f"{field}.{name}")
            for name in _TRUST_DIMENSIONS
        }
        provenance = _clone(
            _mapping(value.get("review_provenance"), f"{field}.review_provenance")
        )
        all_trusted = all(status == TRUSTED for status in dimensions.values())
        if role in _USABLE_ROLES:
            if (
                owner is None
                or category is None
                or not usable
                or not all_trusted
                or burst_id is not None
            ):
                raise AssemblyError(
                    f"{field} usable {role} row requires trusted owner/category, usable_owner_row=true, and no burst_id"
                )
        elif role == "duplicate":
            if (
                owner is None
                or category is None
                or usable
                or not all_trusted
                or burst_id is None
            ):
                raise AssemblyError(
                    f"{field} duplicate row requires trusted owner/category, usable_owner_row=false, and a burst_id"
                )
        else:
            # These are explicitly neutral—not weak negatives.  Keeping their
            # status values allows a receipt to describe why a suffix stopped.
            if usable or burst_id is not None:
                raise AssemblyError(
                    f"{field} {role} row must be neutral: usable_owner_row=false and burst_id=null"
                )
        result[trajectory_id][row_index] = ReviewedRow(
            trajectory_id=trajectory_id,
            row_index=row_index,
            physical_owner_id=owner,
            review_role=role,
            usable_owner_row=usable,
            burst_id=burst_id,
            category=category,
            review_provenance=provenance,
        )
    for trajectory_id, trajectory in trajectories.items():
        missing = sorted(
            set(range(len(trajectory.rows))) - set(result.get(trajectory_id, {}))
        )
        if missing:
            raise AssemblyError(
                f"reviewed owner ledger lacks exact decisions for {trajectory_id} rows {missing}"
            )
    return {key: dict(value) for key, value in result.items()}


def _trusted_suffix_indices(
    *, ledger: Mapping[int, ReviewedRow], start_after: int, duplicate_rows: set[int]
) -> tuple[int, ...]:
    """Retain a conservative cleaned suffix and stop before its first doubt."""

    retained: list[int] = []
    for index in range(start_after + 1, len(ledger)):
        reviewed = ledger[index]
        if index in duplicate_rows:
            # A second confirmed duplicate is intentionally not silently
            # removed by this burst-local rewrite; it needs its own reviewed
            # trajectory event.
            break
        if not reviewed.trusted_usable:
            break
        retained.append(index)
    return tuple(retained)


def _reviewed_bursts(
    *,
    trajectories: Mapping[str, ParsedTrajectory],
    ledger_by_trajectory: Mapping[str, Mapping[int, ReviewedRow]],
) -> tuple[list[ReviewedBurst], list[dict[str, Any]]]:
    """Return only fully reviewed bursts; all other rows stay receipt-visible neutral."""

    admitted: list[ReviewedBurst] = []
    exclusions: list[dict[str, Any]] = []
    for trajectory_id in sorted(trajectories):
        trajectory = trajectories[trajectory_id]
        ledger = ledger_by_trajectory[trajectory_id]
        by_burst: dict[str, list[ReviewedRow]] = defaultdict(list)
        for reviewed in ledger.values():
            if reviewed.review_role == "duplicate":
                assert reviewed.burst_id is not None
                by_burst[reviewed.burst_id].append(reviewed)
            elif reviewed.review_role in _NEUTRAL_ROLES:
                exclusions.append(
                    {
                        "trajectory_id": trajectory_id,
                        "image_id": trajectory.image_id,
                        "row_index": reviewed.row_index,
                        "reason": f"neutral_{reviewed.review_role}",
                        "physical_owner_id": reviewed.physical_owner_id,
                    }
                )
        for burst_id, rows in sorted(by_burst.items()):
            rows.sort(key=lambda item: item.row_index)
            indices = tuple(item.row_index for item in rows)
            owner_ids = {item.physical_owner_id for item in rows}
            categories = {item.category for item in rows}
            if len(owner_ids) != 1 or len(categories) != 1:
                raise AssemblyError(
                    f"trajectory {trajectory_id} burst {burst_id} must have one reviewed owner and category"
                )
            if indices != tuple(range(indices[0], indices[0] + len(indices))):
                raise AssemblyError(
                    f"trajectory {trajectory_id} burst {burst_id} duplicate rows must be contiguous"
                )
            owner_id = next(iter(owner_ids))
            assert owner_id is not None
            prior_usable = [
                row
                for row in ledger.values()
                if row.row_index < indices[0]
                and row.trusted_usable
                and row.physical_owner_id == owner_id
            ]
            if not prior_usable:
                exclusions.append(
                    {
                        "trajectory_id": trajectory_id,
                        "image_id": trajectory.image_id,
                        "burst_id": burst_id,
                        "row_indices": list(indices),
                        "reason": "duplicate_owner_has_no_usable_first_occurrence",
                        "physical_owner_id": owner_id,
                    }
                )
                continue
            # Retain the *first* usable occurrence even if a later duplicate
            # has better geometry.  The ledger must explicitly create a new
            # reviewed trajectory if replacement is scientifically intended.
            first_owner = min(prior_usable, key=lambda item: item.row_index)
            recovery: ReviewedRow | None = None
            for row_index in range(indices[-1] + 1, len(trajectory.rows)):
                candidate = ledger[row_index]
                if candidate.review_role in _NEUTRAL_ROLES:
                    break
                if candidate.review_role == "duplicate":
                    break
                if candidate.review_role == "recovery":
                    recovery = candidate
                    break
                # An ordinary accepted row after a burst is context, not a
                # claimed recovery.  Require an explicit human-reviewed role.
                if candidate.review_role == "accepted":
                    break
            if recovery is None or recovery.physical_owner_id is None:
                exclusions.append(
                    {
                        "trajectory_id": trajectory_id,
                        "image_id": trajectory.image_id,
                        "burst_id": burst_id,
                        "row_indices": list(indices),
                        "reason": "no_trusted_explicit_recovery_after_burst",
                        "physical_owner_id": owner_id,
                    }
                )
                continue
            if recovery.physical_owner_id == owner_id:
                exclusions.append(
                    {
                        "trajectory_id": trajectory_id,
                        "image_id": trajectory.image_id,
                        "burst_id": burst_id,
                        "row_indices": list(indices),
                        "reason": "recovery_owner_is_covered_duplicate_owner",
                        "physical_owner_id": owner_id,
                    }
                )
                continue
            suffix = _trusted_suffix_indices(
                ledger=ledger,
                start_after=indices[-1],
                duplicate_rows=set(indices),
            )
            if not suffix or suffix[0] != recovery.row_index:
                raise AssemblyError(
                    f"trajectory {trajectory_id} burst {burst_id} recovery is not the first trusted cleaned suffix row"
                )
            admitted.append(
                ReviewedBurst(
                    trajectory_id=trajectory_id,
                    image_id=trajectory.image_id,
                    burst_id=burst_id,
                    owner_id=owner_id,
                    first_owner_row_index=first_owner.row_index,
                    duplicate_row_indices=indices,
                    recovery_row_index=recovery.row_index,
                    cleaned_suffix_row_indices=suffix,
                )
            )
    return admitted, exclusions


def burst_credit_units(duplicate_row_indices: Sequence[int]) -> dict[int, float]:
    """Allocate exactly one raw credit unit to an observed duplicate burst."""

    indices = tuple(
        _int(value, f"duplicate_row_indices[{index}]", minimum=0)
        for index, value in enumerate(duplicate_row_indices)
    )
    if not indices:
        raise AssemblyError("duplicate burst must contain at least one row")
    if len(set(indices)) != len(indices) or indices != tuple(sorted(indices)):
        raise AssemblyError("duplicate burst row indices must be unique and sorted")
    if len(indices) == 1:
        return {indices[0]: 1.0}
    result = {indices[0]: 0.5}
    later_credit = 0.5 / float(len(indices) - 1)
    result.update({index: later_credit for index in indices[1:]})
    if not math.isclose(math.fsum(result.values()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError("burst credit construction lost conservation")
    return result


def _generation_for_row(trajectory: ParsedTrajectory, row: ExactRow) -> dict[str, Any]:
    """Attach an exact observed prefix hash without rewriting provenance."""

    return {
        **_clone(trajectory.generation_provenance),
        "prefix_token_ids_sha256": row.prefix_token_ids_sha256,
        "source_row_index": row.row_index,
    }


def _candidate(
    *, role: str, trajectory: ParsedTrajectory, row: ExactRow, reviewed: ReviewedRow
) -> dict[str, Any]:
    if reviewed.physical_owner_id is None:
        raise AssemblyError("gradient-bearing candidate lacks reviewed physical owner")
    return {
        "candidate_role": role,
        "source_row_index": row.row_index,
        "physical_owner_id": reviewed.physical_owner_id,
        "category": reviewed.category,
        "token_ids": list(row.token_ids),
        "token_ids_sha256": row.token_ids_sha256,
        "selected_sites": exact_row_site_types(row.token_ids),
        "generation_provenance": _generation_for_row(trajectory, row),
        "generation_prefix_token_ids_sha256": row.prefix_token_ids_sha256,
    }


def _state_bank_review_provenance(reviewed: ReviewedRow) -> dict[str, str]:
    """Project the reviewed row's provenance into the strict StateBank shape."""

    value = _mapping(
        reviewed.review_provenance,
        f"reviewed_owner_ledger[{reviewed.trajectory_id}:{reviewed.row_index}].review_provenance",
    )
    return {
        "source": _string(value.get("source"), "review_provenance.source"),
        "reviewer": _string(value.get("reviewer"), "review_provenance.reviewer"),
        "confidence": _string(value.get("confidence"), "review_provenance.confidence"),
        "comment": _string_allow_empty(
            value.get("comment", ""), "review_provenance.comment"
        ),
    }


def _state_bank_context(
    *,
    trajectory: ParsedTrajectory,
    ledger: Mapping[int, ReviewedRow],
    replay_prefix_row_indices: Sequence[int],
) -> dict[str, Any]:
    """Freeze the context needed to map one draft event into StateBank rows."""

    proofs: list[dict[str, Any]] = []
    for prefix_index, original_row_index in enumerate(replay_prefix_row_indices):
        reviewed = ledger[original_row_index]
        if reviewed.trusted_usable:
            if reviewed.physical_owner_id is None:
                raise AssemblyError(
                    "trusted replay-prefix owner row lacks its physical owner; "
                    f"trajectory={trajectory.trajectory_id}, row={original_row_index}"
                )
            proofs.append(
                {
                    "prefix_object_row_index": prefix_index,
                    "owner_id": reviewed.physical_owner_id,
                    "review_provenance": _state_bank_review_provenance(reviewed),
                }
            )
    prefix_object_row_count = len(replay_prefix_row_indices)
    if prefix_object_row_count == 0:
        coverage_status = "empty"
    elif len(proofs) == prefix_object_row_count:
        coverage_status = "resolved"
    elif proofs:
        coverage_status = "partially_resolved"
    else:
        raise AssemblyError(
            "duplicate StateBank replay prefix has no trusted owner proof; "
            f"trajectory={trajectory.trajectory_id}"
        )
    return {
        "image": _clone(trajectory.image),
        "physical_entities": _clone(trajectory.physical_entities),
        "executed_prompt_token_ids": list(trajectory.executed_prompt_token_ids),
        "executed_prompt_token_ids_sha256": trajectory.prompt_token_ids_sha256,
        "image_pad_interval": list(trajectory.image_pad_interval),
        "prefix_object_row_count": prefix_object_row_count,
        "prefix_coverage_status": coverage_status,
        "prefix_covered_owner_proofs": proofs,
    }


def _event_evidence(
    *,
    burst: ReviewedBurst,
    replay_prefix: Sequence[int],
    replay_context: str,
    burst_credit: float,
    candidate_generation_prefix_hashes: Mapping[str, str],
) -> dict[str, Any]:
    if replay_context not in {
        "exact_self_prefix_transplant",
        "counterfactual_rewritten",
    }:
        raise AssemblyError(f"unsupported replay context {replay_context!r}")
    return {
        "trajectory_id": burst.trajectory_id,
        "burst_id": burst.burst_id,
        "replay_context_kind": replay_context,
        "replay_prefix_token_ids_sha256": token_ids_sha256(replay_prefix),
        "candidate_generation_prefix_hashes": dict(
            sorted(candidate_generation_prefix_hashes.items())
        ),
        "retained_first_owner_row_index": burst.first_owner_row_index,
        "duplicate_row_indices": list(burst.duplicate_row_indices),
        "removed_row_indices": (
            list(burst.duplicate_row_indices)
            if replay_context == "counterfactual_rewritten"
            else []
        ),
        "recovery_row_index": burst.recovery_row_index,
        "burst_credit": float(burst_credit),
    }


def _event_id(
    *,
    family: str,
    burst: ReviewedBurst,
    replay_row_index: int,
    candidate_row_index: int,
) -> str:
    return (
        f"duplicate-trajectory-{family}-trajectory-{burst.trajectory_id}"
        f"-burst-{burst.burst_id}-replay-row-{replay_row_index}-candidate-row-{candidate_row_index}"
    )


def _local_events_for_burst(
    *,
    trajectory: ParsedTrajectory,
    ledger: Mapping[int, ReviewedRow],
    burst: ReviewedBurst,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    credits = burst_credit_units(burst.duplicate_row_indices)
    recovery_row = trajectory.rows[burst.recovery_row_index]
    recovery = ledger[burst.recovery_row_index]
    local_events: list[dict[str, Any]] = []
    positive_events: list[dict[str, Any]] = []
    for duplicate_index in burst.duplicate_row_indices:
        duplicate_row = trajectory.rows[duplicate_index]
        duplicate = ledger[duplicate_index]
        replay_prefix = list(duplicate_row.prefix_token_ids)
        positive_candidate = _candidate(
            role="positive_recovery",
            trajectory=trajectory,
            row=recovery_row,
            reviewed=recovery,
        )
        duplicate_candidate = _candidate(
            role="covered_owner_duplicate",
            trajectory=trajectory,
            row=duplicate_row,
            reviewed=duplicate,
        )
        raw_credit = credits[duplicate_index]
        replay_prefix_row_indices = list(range(duplicate_index))
        local_evidence = _event_evidence(
            burst=burst,
            replay_prefix=replay_prefix,
            replay_context="exact_self_prefix_transplant",
            burst_credit=raw_credit,
            candidate_generation_prefix_hashes={
                "positive_recovery": positive_candidate[
                    "generation_prefix_token_ids_sha256"
                ],
                "covered_owner_duplicate": duplicate_candidate[
                    "generation_prefix_token_ids_sha256"
                ],
            },
        )
        positive_evidence = _event_evidence(
            burst=burst,
            replay_prefix=replay_prefix,
            replay_context="exact_self_prefix_transplant",
            burst_credit=raw_credit,
            candidate_generation_prefix_hashes={
                "positive_recovery": positive_candidate[
                    "generation_prefix_token_ids_sha256"
                ],
            },
        )
        common = {
            "image_id": trajectory.image_id,
            "split": trajectory.split,
            "trajectory_id": trajectory.trajectory_id,
            "burst_id": burst.burst_id,
            "replay_prefix_token_ids": replay_prefix,
            "replay_prefix_token_ids_sha256": token_ids_sha256(replay_prefix),
            "raw_burst_credit": raw_credit,
            "source_trajectory_generated_token_ids_sha256": trajectory.generated_token_ids_sha256,
            "replay_prefix_row_indices": replay_prefix_row_indices,
            "retained_first_owner_row_index": burst.first_owner_row_index,
            "state_bank_context": _state_bank_context(
                trajectory=trajectory,
                ledger=ledger,
                replay_prefix_row_indices=replay_prefix_row_indices,
            ),
        }
        local_events.append(
            {
                "event_id": _event_id(
                    family="local",
                    burst=burst,
                    replay_row_index=duplicate_index,
                    candidate_row_index=burst.recovery_row_index,
                ),
                "event_family": "local_duplicate_rejection",
                **common,
                "duplicate_trajectory_evidence": local_evidence,
                "positive_candidate": positive_candidate,
                "duplicate_candidate": duplicate_candidate,
            }
        )
        positive_events.append(
            {
                "event_id": _event_id(
                    family="positive",
                    burst=burst,
                    replay_row_index=duplicate_index,
                    candidate_row_index=burst.recovery_row_index,
                ),
                "event_family": "recovery_positive_only",
                **common,
                "duplicate_trajectory_evidence": positive_evidence,
                "positive_candidate": _clone(positive_candidate),
            }
        )
    return local_events, positive_events


def _cleaned_prefix_for_row(
    *, trajectory: ParsedTrajectory, removed_rows: set[int], row_index: int
) -> tuple[list[int], list[int]]:
    retained_indices = [
        index for index in range(row_index) if index not in removed_rows
    ]
    prefix: list[int] = []
    for index in retained_indices:
        prefix.extend(trajectory.rows[index].token_ids)
    return retained_indices, prefix


def _cleaned_events_for_burst(
    *,
    trajectory: ParsedTrajectory,
    ledger: Mapping[int, ReviewedRow],
    burst: ReviewedBurst,
) -> list[dict[str, Any]]:
    suffix = burst.cleaned_suffix_row_indices
    if not suffix:
        return []
    removed = set(burst.duplicate_row_indices)
    raw_credit = 1.0 / float(len(suffix))
    events: list[dict[str, Any]] = []
    for row_index in suffix:
        row = trajectory.rows[row_index]
        reviewed = ledger[row_index]
        retained_indices, replay_prefix = _cleaned_prefix_for_row(
            trajectory=trajectory, removed_rows=removed, row_index=row_index
        )
        positive = _candidate(
            role="cleaned_positive", trajectory=trajectory, row=row, reviewed=reviewed
        )
        evidence = _event_evidence(
            burst=burst,
            replay_prefix=replay_prefix,
            replay_context="counterfactual_rewritten",
            burst_credit=raw_credit,
            candidate_generation_prefix_hashes={
                "cleaned_positive": positive["generation_prefix_token_ids_sha256"]
            },
        )
        events.append(
            {
                "event_id": _event_id(
                    family="cleaned",
                    burst=burst,
                    replay_row_index=row_index,
                    candidate_row_index=row_index,
                ),
                "event_family": "duplicate_cleaned_imitation",
                "image_id": trajectory.image_id,
                "split": trajectory.split,
                "trajectory_id": trajectory.trajectory_id,
                "burst_id": burst.burst_id,
                "replay_prefix_token_ids": replay_prefix,
                "replay_prefix_token_ids_sha256": token_ids_sha256(replay_prefix),
                "duplicate_trajectory_evidence": evidence,
                "raw_burst_credit": raw_credit,
                "source_trajectory_generated_token_ids_sha256": trajectory.generated_token_ids_sha256,
                "replay_prefix_row_indices": retained_indices,
                "retained_first_owner_row_index": burst.first_owner_row_index,
                "state_bank_context": _state_bank_context(
                    trajectory=trajectory,
                    ledger=ledger,
                    replay_prefix_row_indices=retained_indices,
                ),
                "positive_candidate": positive,
            }
        )
    if not math.isclose(
        math.fsum(event["raw_burst_credit"] for event in events),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise AssertionError("cleaned burst credits lost conservation")
    return events


def _normalise_mechanism_events(
    events: Sequence[Mapping[str, Any]], *, family_scales: Mapping[str, float]
) -> list[dict[str, Any]]:
    """Apply fixed family shares, then equalise aggregate credit by image."""

    normalized: list[dict[str, Any]] = []
    burst_sums: dict[tuple[str, str], float] = defaultdict(float)
    for item in events:
        event = _clone(item)
        family = _string(event["event_family"], "event.event_family")
        scale = family_scales.get(family)
        if scale is None or not math.isfinite(scale) or scale <= 0.0:
            raise AssemblyError(f"no positive family scale for {family!r}")
        credit = float(event["raw_burst_credit"]) * scale
        event["family_credit_scale"] = float(scale)
        event["mechanism_credit_before_image_balance"] = credit
        key = (str(event["trajectory_id"]), str(event["burst_id"]))
        burst_sums[key] += credit
        normalized.append(event)
    if any(
        not math.isclose(value, 1.0, rel_tol=0.0, abs_tol=1e-12)
        for value in burst_sums.values()
    ):
        raise AssemblyError(
            "combined family allocation must preserve one total mechanism credit per burst"
        )
    burst_counts = Counter(
        str(item["image_id"])
        for key in burst_sums
        for item in [
            next(
                event
                for event in normalized
                if (str(event["trajectory_id"]), str(event["burst_id"])) == key
            )
        ]
    )
    # A burst belongs to exactly one image.  The above mapping intentionally
    # uses the event itself rather than a derived image relation.
    for event in normalized:
        image = str(event["image_id"])
        burst_count = burst_counts[image]
        if burst_count <= 0:
            raise AssertionError("image burst count disappeared during normalization")
        event["image_burst_count"] = burst_count
        event["image_balanced_event_weight"] = float(
            event["mechanism_credit_before_image_balance"]
        ) / float(burst_count)
    image_totals: dict[str, float] = defaultdict(float)
    for event in normalized:
        image_totals[str(event["image_id"])] += float(
            event["image_balanced_event_weight"]
        )
    if any(
        not math.isclose(value, 1.0, rel_tol=0.0, abs_tol=1e-12)
        for value in image_totals.values()
    ):
        raise AssertionError("image balancing failed to give equal image total credit")
    return normalized


def _validate_source_events(
    source_events: Sequence[Mapping[str, Any]], *, bursts: Sequence[ReviewedBurst]
) -> dict[tuple[str, str], dict[str, Any]]:
    """Require one immutable Source row per admitted burst/window."""

    expected = {(burst.trajectory_id, burst.burst_id): burst for burst in bursts}
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for index, raw in enumerate(source_events):
        field = f"source_preservation_events[{index}]"
        value = _clone(_mapping(raw, field))
        key = (
            _string(value.get("trajectory_id"), f"{field}.trajectory_id"),
            _string(value.get("burst_id"), f"{field}.burst_id"),
        )
        burst = expected.get(key)
        if burst is None:
            raise AssemblyError(
                f"{field} does not belong to an admitted duplicate burst"
            )
        if key in result:
            raise AssemblyError(f"{field} duplicates Source preservation for {key}")
        if _image_id(value.get("image_id"), f"{field}.image_id") != burst.image_id:
            raise AssemblyError(f"{field}.image_id disagrees with its reviewed burst")
        _string(value.get("event_id"), f"{field}.event_id")
        value["event_family"] = "source_preservation"
        value["source_burst_key"] = {"trajectory_id": key[0], "burst_id": key[1]}
        result[key] = value
    missing = sorted(expected.keys() - result.keys())
    if missing:
        raise AssemblyError(
            "every admitted duplicate burst requires exactly one matched Source preservation event; "
            f"missing={missing}"
        )
    return result


def _source_weighted_events(
    source_by_burst: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    def source_balance_image_id(event: Mapping[str, Any]) -> str:
        return _image_id(
            event.get("immutable_source_image_id", event["image_id"]),
            "source preservation immutable_source_image_id",
        )

    counts = Counter(
        source_balance_image_id(event) for event in source_by_burst.values()
    )
    result: list[dict[str, Any]] = []
    for key, raw in sorted(source_by_burst.items()):
        event = _clone(raw)
        balance_image_id = source_balance_image_id(event)
        event["source_balance_image_id"] = balance_image_id
        event["image_source_event_count"] = counts[balance_image_id]
        event["image_balanced_event_weight"] = 1.0 / float(
            event["image_source_event_count"]
        )
        result.append(event)
    return result


def family_stratified_windows(
    *,
    bursts: Sequence[ReviewedBurst],
    events: Sequence[Mapping[str, Any]],
    enabled_families: Sequence[str],
) -> list[dict[str, Any]]:
    """Create one deterministic window per burst with every enabled family.

    Each event belongs to exactly one burst.  The windows therefore neither
    clone event IDs nor multiply their fixed credit.  A downstream distributed
    planner can partition a window across ranks, but the global receipt already
    proves every enabled family has a positive denominator before forward.
    """

    families = tuple(
        sorted({_string(value, "enabled_families") for value in enabled_families})
    )
    if not families:
        raise AssemblyError("family-stratified windows require at least one family")
    events_by_key_family: dict[tuple[str, str], dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for event in events:
        family = _string(event["event_family"], "event.event_family")
        if family not in families:
            continue
        key = (
            _string(event["trajectory_id"], "event.trajectory_id"),
            _string(event["burst_id"], "event.burst_id"),
        )
        events_by_key_family[key][family].append(
            _string(event["event_id"], "event.event_id")
        )
    windows: list[dict[str, Any]] = []
    for sequence, burst in enumerate(
        sorted(
            bursts,
            key=lambda item: (
                int(item.image_id),
                item.trajectory_id,
                item.duplicate_row_indices[0],
                item.burst_id,
            ),
        )
    ):
        by_family = events_by_key_family.get(burst.key, {})
        missing = [family for family in families if not by_family.get(family)]
        if missing:
            raise AssemblyError(
                f"planned window for {burst.key} lacks enabled family/families {missing}"
            )
        family_event_ids = {family: sorted(by_family[family]) for family in families}
        windows.append(
            {
                "window_id": f"duplicate-burst-window-{sequence:05d}",
                "trajectory_id": burst.trajectory_id,
                "burst_id": burst.burst_id,
                "image_id": burst.image_id,
                "global_family_event_ids": family_event_ids,
                "global_family_denominators": {
                    family: len(ids) for family, ids in family_event_ids.items()
                },
            }
        )
    return windows


def _arm_receipt(
    *,
    name: str,
    mechanism_events: Sequence[Mapping[str, Any]],
    source_events: Sequence[Mapping[str, Any]],
    windows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    mechanism_by_burst: dict[str, float] = defaultdict(float)
    mechanism_by_image: dict[str, float] = defaultdict(float)
    family_counts = Counter()
    for event in mechanism_events:
        key = f"{event['trajectory_id']}:{event['burst_id']}"
        mechanism_by_burst[key] += float(event["mechanism_credit_before_image_balance"])
        mechanism_by_image[str(event["image_id"])] += float(
            event["image_balanced_event_weight"]
        )
        family_counts[str(event["event_family"])] += 1
    source_digest = hashlib.sha256(_canonical(list(source_events))).hexdigest()
    return {
        "bank_name": name,
        "mechanism_event_count": len(mechanism_events),
        "source_preservation_event_count": len(source_events),
        "event_family_counts": dict(sorted(family_counts.items())),
        "mechanism_credit_by_burst": dict(sorted(mechanism_by_burst.items())),
        "mechanism_credit_by_image_after_balance": dict(
            sorted(mechanism_by_image.items())
        ),
        "source_event_payload_sha256": source_digest,
        "optimizer_window_count": len(windows),
        "windows_complete": all(
            all(value > 0 for value in window["global_family_denominators"].values())
            for window in windows
        ),
    }


def _trajectory_receipt(
    *, trajectory: ParsedTrajectory, bursts: Sequence[ReviewedBurst]
) -> dict[str, Any]:
    removed_by_burst = {
        burst.burst_id: set(burst.duplicate_row_indices)
        for burst in bursts
        if burst.trajectory_id == trajectory.trajectory_id
    }
    raw_rows = [row.to_artifact_dict() for row in trajectory.rows]
    cleaned: dict[str, Any] = {}
    for burst_id, removed in sorted(removed_by_burst.items()):
        retained_rows = [row for row in trajectory.rows if row.row_index not in removed]
        cleaned_ids = [token for row in retained_rows for token in row.token_ids]
        cleaned[burst_id] = {
            "removed_row_indices": sorted(removed),
            "retained_row_indices": [row.row_index for row in retained_rows],
            "generated_token_ids": cleaned_ids,
            "generated_token_ids_sha256": token_ids_sha256(cleaned_ids),
        }
    return {
        "trajectory_id": trajectory.trajectory_id,
        "image_id": trajectory.image_id,
        "observed_generated_token_ids_sha256": trajectory.generated_token_ids_sha256,
        "observed_rows": raw_rows,
        "cleaned_trajectories": cleaned,
    }


def build_duplicate_trajectory_bank_drafts(
    *,
    rollout_trajectories: Sequence[Mapping[str, Any]],
    reviewed_owner_ledger: Sequence[Mapping[str, Any]],
    source_preservation_events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build immutable, matched duplicate-treatment bank drafts.

    The return value contains no inferred owner labels and no modified source
    prefix hash.  It is suitable for JSON writing and for a later narrow
    adapter to ``DuplicateTrajectoryEvidence`` in the StateBank core.
    """

    parsed: dict[str, ParsedTrajectory] = {}
    for index, raw in enumerate(rollout_trajectories):
        trajectory = _parse_trajectory(
            _mapping(raw, f"rollout_trajectories[{index}]"), index=index
        )
        if trajectory.trajectory_id in parsed:
            raise AssemblyError(f"duplicate trajectory_id {trajectory.trajectory_id!r}")
        parsed[trajectory.trajectory_id] = trajectory
    if not parsed:
        raise AssemblyError("rollout_trajectories must not be empty")
    ledger = _parse_reviewed_ledger(
        [
            _mapping(value, f"reviewed_owner_ledger[{index}]")
            for index, value in enumerate(reviewed_owner_ledger)
        ],
        trajectories=parsed,
    )
    bursts, exclusions = _reviewed_bursts(
        trajectories=parsed, ledger_by_trajectory=ledger
    )
    if not bursts:
        raise AssemblyError(
            "no fully reviewed duplicate burst with an explicit recovery was admitted"
        )
    source_by_burst = _validate_source_events(source_preservation_events, bursts=bursts)
    source_events = _source_weighted_events(source_by_burst)

    raw_local: list[dict[str, Any]] = []
    raw_positive: list[dict[str, Any]] = []
    raw_cleaned: list[dict[str, Any]] = []
    for burst in bursts:
        trajectory = parsed[burst.trajectory_id]
        local, positive = _local_events_for_burst(
            trajectory=trajectory, ledger=ledger[burst.trajectory_id], burst=burst
        )
        cleaned = _cleaned_events_for_burst(
            trajectory=trajectory, ledger=ledger[burst.trajectory_id], burst=burst
        )
        if not local or not positive or not cleaned:
            raise AssemblyError(
                f"admitted burst {burst.key} did not yield every required mechanism family"
            )
        raw_local.extend(local)
        raw_positive.extend(positive)
        raw_cleaned.extend(cleaned)

    local_events = _normalise_mechanism_events(
        raw_local, family_scales={"local_duplicate_rejection": 1.0}
    )
    positive_events = _normalise_mechanism_events(
        raw_positive, family_scales={"recovery_positive_only": 1.0}
    )
    cleaned_events = _normalise_mechanism_events(
        raw_cleaned, family_scales={"duplicate_cleaned_imitation": 1.0}
    )
    combined_events = _normalise_mechanism_events(
        [*raw_local, *raw_cleaned],
        family_scales={
            "local_duplicate_rejection": 0.5,
            "duplicate_cleaned_imitation": 0.5,
        },
    )

    arm_events = {
        SOURCE_ONLY: [],
        POSITIVE_ONLY: positive_events,
        LOCAL: local_events,
        CLEANED: cleaned_events,
        COMBINED: combined_events,
    }
    enabled_families = {
        SOURCE_ONLY: ("source_preservation",),
        POSITIVE_ONLY: ("source_preservation", "recovery_positive_only"),
        LOCAL: ("source_preservation", "local_duplicate_rejection"),
        CLEANED: ("source_preservation", "duplicate_cleaned_imitation"),
        COMBINED: (
            "source_preservation",
            "local_duplicate_rejection",
            "duplicate_cleaned_imitation",
        ),
    }
    banks: dict[str, Any] = {}
    for name in BANK_NAMES:
        mechanisms = arm_events[name]
        all_events = [*_clone(source_events), *_clone(mechanisms)]
        windows = family_stratified_windows(
            bursts=bursts,
            events=all_events,
            enabled_families=enabled_families[name],
        )
        banks[name] = {
            "schema_version": SCHEMA_VERSION,
            "bank_name": name,
            "events": all_events,
            "optimizer_windows": windows,
            "allocation_receipt": _arm_receipt(
                name=name,
                mechanism_events=mechanisms,
                source_events=source_events,
                windows=windows,
            ),
        }
    source_digest = banks[SOURCE_ONLY]["allocation_receipt"][
        "source_event_payload_sha256"
    ]
    if any(
        bank["allocation_receipt"]["source_event_payload_sha256"] != source_digest
        for bank in banks.values()
    ):
        raise AssertionError("Source preservation payload drifted across matched arms")
    window_count = len(bursts)
    if any(len(bank["optimizer_windows"]) != window_count for bank in banks.values()):
        raise AssertionError("matched arms disagree on optimizer window count")
    allocation_receipt = {
        "schema_version": ALLOCATION_RECEIPT_SCHEMA_VERSION,
        "bank_names": list(BANK_NAMES),
        "admitted_burst_count": len(bursts),
        "admitted_image_count": len({burst.image_id for burst in bursts}),
        "admitted_bursts": [
            {
                "trajectory_id": burst.trajectory_id,
                "image_id": burst.image_id,
                "burst_id": burst.burst_id,
                "owner_id": burst.owner_id,
                "retained_first_owner_row_index": burst.first_owner_row_index,
                "duplicate_row_indices": list(burst.duplicate_row_indices),
                "recovery_row_index": burst.recovery_row_index,
                "cleaned_suffix_row_indices": list(burst.cleaned_suffix_row_indices),
            }
            for burst in sorted(
                bursts,
                key=lambda item: (
                    int(item.image_id),
                    item.trajectory_id,
                    item.burst_id,
                ),
            )
        ],
        "exclusions": sorted(
            exclusions,
            key=lambda item: (
                int(str(item["image_id"])),
                str(item["trajectory_id"]),
                int(item.get("row_index", -1)),
                str(item.get("burst_id", "")),
                str(item["reason"]),
            ),
        ),
        "source_preservation_payload_sha256": source_digest,
        "matched_optimizer_window_count": window_count,
        "arms": {
            name: _clone(banks[name]["allocation_receipt"]) for name in BANK_NAMES
        },
        "trajectories": [
            _trajectory_receipt(trajectory=trajectory, bursts=bursts)
            for trajectory in sorted(
                parsed.values(),
                key=lambda item: (int(item.image_id), item.trajectory_id),
            )
        ],
    }
    return {"banks": banks, "allocation_receipt": allocation_receipt}


def _state_bank_generation_provenance(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Drop draft-only row metadata while retaining the true generation hash."""

    source = _mapping(
        candidate.get("generation_provenance"), "candidate.generation_provenance"
    )
    keys = {
        "mode",
        "seed",
        "temperature",
        "top_p",
        "repetition_penalty",
        "checkpoint_id",
        "prompt_token_ids_sha256",
        "prefix_token_ids_sha256",
    }
    missing = keys - set(source)
    if missing:
        raise AssemblyError(
            f"candidate generation provenance is missing {sorted(missing)}"
        )
    return {key: _clone(source[key]) for key in sorted(keys)}


def _state_bank_candidate_pair(
    *, event_id: str, candidate: Mapping[str, Any], ordinal: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Map one reviewed exact row into the current narrow StateBank join API."""

    candidate_role = _string(
        candidate.get("candidate_role"), "candidate.candidate_role"
    )
    is_duplicate = candidate_role == "covered_owner_duplicate"
    candidate_id = f"{event_id}-candidate-{ordinal:02d}"
    token_ids = _token_ids(candidate.get("token_ids"), "candidate.token_ids")
    owner_id = _string(
        candidate.get("physical_owner_id"), "candidate.physical_owner_id"
    )
    selected_sites = _sequence(
        candidate.get("selected_sites"), "candidate.selected_sites"
    )
    source_row_index = _int(
        candidate.get("source_row_index"), "candidate.source_row_index", minimum=0
    )
    rollout = {
        "candidate_id": candidate_id,
        "token_ids": list(token_ids),
        "token_ids_sha256": token_ids_sha256(token_ids),
        "generation_provenance": _state_bank_generation_provenance(candidate),
        "evidence_text": "reviewed exact integer rollout row",
    }
    review = {
        "candidate_id": candidate_id,
        "role": "harmful" if is_duplicate else "positive",
        "harmful_kind": "duplicate" if is_duplicate else None,
        "physical_owner_id": owner_id,
        "coverage_status": "covered" if is_duplicate else "uncovered",
        "entity_review_status": "trusted",
        "geometry_review_status": "trusted",
        "entity_eligible": True,
        "geometry_eligible": False,
        "owner_resolution_interval": [0, len(token_ids)],
        "coordinate_decision": None,
        "selected_sites": _clone(selected_sites),
    }
    # The caller needs the stable ID and original row to construct typed
    # evidence, but neither is an extra StateBank candidate field.
    review["_draft_candidate_role"] = candidate_role
    review["_draft_source_row_index"] = source_row_index
    return rollout, review


def _state_bank_rows_for_mechanism_event(
    event: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Convert one self-contained draft mechanism event into rollout/review rows."""

    event_id = _string(event.get("event_id"), "event.event_id")
    context = _mapping(
        event.get("state_bank_context"), f"{event_id}.state_bank_context"
    )
    physical_entities = _sequence(
        context.get("physical_entities"),
        f"{event_id}.state_bank_context.physical_entities",
    )
    if not physical_entities:
        raise AssemblyError(
            f"{event_id} cannot materialize a StateBank without trajectory.physical_entities"
        )
    family = _string(event.get("event_family"), f"{event_id}.event_family")
    family_flags = {
        "recovery_positive_imitation_eligible": family == "recovery_positive_only",
        "local_duplicate_rejection_eligible": family == "local_duplicate_rejection",
        "duplicate_cleaned_imitation_eligible": family == "duplicate_cleaned_imitation",
    }
    if sum(bool(value) for value in family_flags.values()) != 1:
        raise AssemblyError(
            f"{event_id} has an unsupported draft mechanism family {family!r}"
        )
    candidates: list[Mapping[str, Any]] = []
    positive = event.get("positive_candidate")
    if positive is not None:
        candidates.append(_mapping(positive, f"{event_id}.positive_candidate"))
    duplicate = event.get("duplicate_candidate")
    if duplicate is not None:
        candidates.append(_mapping(duplicate, f"{event_id}.duplicate_candidate"))
    if not candidates:
        raise AssemblyError(f"{event_id} lacks mechanism candidates")
    rollout_candidates: list[dict[str, Any]] = []
    review_candidates: list[dict[str, Any]] = []
    candidate_ids_by_role: dict[str, str] = {}
    candidate_rows_by_id: dict[str, int] = {}
    candidate_prefix_hashes_by_id: dict[str, str] = {}
    for ordinal, candidate in enumerate(candidates):
        rollout_candidate, review_candidate = _state_bank_candidate_pair(
            event_id=event_id, candidate=candidate, ordinal=ordinal
        )
        candidate_id = str(rollout_candidate["candidate_id"])
        role = str(review_candidate.pop("_draft_candidate_role"))
        row_index = int(review_candidate.pop("_draft_source_row_index"))
        if role in candidate_ids_by_role:
            raise AssemblyError(
                f"{event_id} has duplicate draft candidate role {role!r}"
            )
        candidate_ids_by_role[role] = candidate_id
        candidate_rows_by_id[candidate_id] = row_index
        candidate_prefix_hashes_by_id[candidate_id] = str(
            rollout_candidate["generation_provenance"]["prefix_token_ids_sha256"]
        )
        rollout_candidates.append(rollout_candidate)
        review_candidates.append(review_candidate)
    evidence_draft = _mapping(
        event.get("duplicate_trajectory_evidence"),
        f"{event_id}.duplicate_trajectory_evidence",
    )
    declared_by_role = _mapping(
        evidence_draft.get("candidate_generation_prefix_hashes"),
        f"{event_id}.duplicate_trajectory_evidence.candidate_generation_prefix_hashes",
    )
    expected_roles = set(candidate_ids_by_role)
    if set(declared_by_role) != expected_roles:
        raise AssemblyError(
            f"{event_id} typed evidence candidate roles disagree with exact candidates: "
            f"evidence={sorted(declared_by_role)}, candidates={sorted(expected_roles)}"
        )
    for role, candidate_id in candidate_ids_by_role.items():
        declared = _sha256(
            declared_by_role[role],
            f"{event_id}.evidence.candidate_generation_prefix_hashes.{role}",
        )
        if declared != candidate_prefix_hashes_by_id[candidate_id]:
            raise AssemblyError(
                f"{event_id} evidence would rewrite generation provenance for {role}"
            )
    typed_evidence = {
        "trajectory_id": _string(
            evidence_draft.get("trajectory_id"), f"{event_id}.evidence.trajectory_id"
        ),
        "burst_id": _string(
            evidence_draft.get("burst_id"), f"{event_id}.evidence.burst_id"
        ),
        "replay_context_kind": _string(
            evidence_draft.get("replay_context_kind"),
            f"{event_id}.evidence.replay_context_kind",
        ),
        "candidate_generation_prefix_token_ids_sha256": candidate_prefix_hashes_by_id,
        "candidate_trajectory_row_indices": candidate_rows_by_id,
        "replay_prefix_token_ids_sha256": _sha256(
            evidence_draft.get("replay_prefix_token_ids_sha256"),
            f"{event_id}.evidence.replay_prefix_token_ids_sha256",
        ),
        "retained_first_owner_row_index": _int(
            evidence_draft.get("retained_first_owner_row_index"),
            f"{event_id}.evidence.retained_first_owner_row_index",
            minimum=0,
        ),
        "duplicate_row_indices": _clone(
            _sequence(
                evidence_draft.get("duplicate_row_indices"),
                f"{event_id}.evidence.duplicate_row_indices",
            )
        ),
        "removed_row_indices": _clone(
            _sequence(
                evidence_draft.get("removed_row_indices"),
                f"{event_id}.evidence.removed_row_indices",
            )
        ),
        "recovery_row_index": _int(
            evidence_draft.get("recovery_row_index"),
            f"{event_id}.evidence.recovery_row_index",
            minimum=0,
        ),
        # This is the *actual* atomic allocation.  The draft's raw credit is
        # retained separately for audit, while the core validates effective
        # one-unit burst totals (including the combined half/half split).
        "burst_credit": float(event["mechanism_credit_before_image_balance"]),
    }
    replay_prefix = _token_ids(
        event.get("replay_prefix_token_ids"),
        f"{event_id}.replay_prefix_token_ids",
        allow_empty=True,
    )
    replay_hash = token_ids_sha256(replay_prefix)
    if replay_hash != typed_evidence["replay_prefix_token_ids_sha256"]:
        raise AssemblyError(
            f"{event_id} exact replay prefix disagrees with typed evidence"
        )
    prefix_count = _int(
        context.get("prefix_object_row_count"),
        f"{event_id}.state_bank_context.prefix_object_row_count",
        minimum=0,
    )
    return (
        {
            "event_id": event_id,
            "image": _clone(
                _mapping(context.get("image"), f"{event_id}.state_bank_context.image")
            ),
            "split": _string(event.get("split"), f"{event_id}.split"),
            "split_group_id": f"image:{_image_id(event.get('image_id'), f'{event_id}.image_id')}",
            "executed_prompt_token_ids": _clone(
                _sequence(
                    context.get("executed_prompt_token_ids"),
                    f"{event_id}.state_bank_context.executed_prompt_token_ids",
                )
            ),
            "executed_prompt_token_ids_sha256": _sha256(
                context.get("executed_prompt_token_ids_sha256"),
                f"{event_id}.state_bank_context.executed_prompt_token_ids_sha256",
            ),
            "image_pad_interval": _clone(
                _sequence(
                    context.get("image_pad_interval"),
                    f"{event_id}.state_bank_context.image_pad_interval",
                )
            ),
            "prefix_token_ids": list(replay_prefix),
            "prefix_token_ids_sha256": replay_hash,
            "candidates": rollout_candidates,
        },
        {
            "event_id": event_id,
            "admission_status": "accepted",
            "rejection_reason": None,
            "physical_entities": _clone(physical_entities),
            "prefix_object_row_count": prefix_count,
            "prefix_coverage_status": _string(
                context.get("prefix_coverage_status"),
                f"{event_id}.state_bank_context.prefix_coverage_status",
            ),
            "prefix_covered_owner_proofs": _clone(
                _sequence(
                    context.get("prefix_covered_owner_proofs"),
                    f"{event_id}.state_bank_context.prefix_covered_owner_proofs",
                )
            ),
            "entity_transition_eligible": False,
            "coordinate_boundary_eligible": False,
            "duplicate_trajectory_evidence": typed_evidence,
            **family_flags,
            "image_balanced_event_weight": float(event["image_balanced_event_weight"]),
            "candidates": review_candidates,
            "review_provenance": {
                "schema_version": SCHEMA_VERSION,
                "policy": "reviewed-physical-owner-duplicate-trajectory",
                "event_family": family,
                "trajectory_id": typed_evidence["trajectory_id"],
                "burst_id": typed_evidence["burst_id"],
                "raw_burst_credit": float(event["raw_burst_credit"]),
                "effective_burst_credit": typed_evidence["burst_credit"],
            },
        },
    )


def state_bank_rows_from_duplicate_draft_bank(
    bank: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return strict StateBank assembler inputs for one matched draft arm.

    Source preservation events are intentionally opaque to the duplicate data
    factory except for their burst binding.  To materialize an arm, each source
    wrapper must include immutable ``state_bank_rollout`` and
    ``state_bank_review`` mappings created by the existing Source assembler.
    """

    events = _sequence(bank.get("events"), "bank.events")
    rollouts: list[dict[str, Any]] = []
    reviews: list[dict[str, Any]] = []
    seen_event_ids: set[str] = set()
    for raw in events:
        event = _mapping(raw, "bank.events[]")
        family = _string(event.get("event_family"), "bank.events[].event_family")
        if family == "source_preservation":
            rollout = _clone(
                _mapping(
                    event.get("state_bank_rollout"), "source event.state_bank_rollout"
                )
            )
            review = _clone(
                _mapping(
                    event.get("state_bank_review"), "source event.state_bank_review"
                )
            )
            # The canonical source review remains immutable in the wrapper;
            # its training dose is derived from the matched duplicate-bank
            # source grouping and must be applied to the emitted row.
            review["image_balanced_event_weight"] = float(
                event["image_balanced_event_weight"]
            )
        else:
            rollout, review = _state_bank_rows_for_mechanism_event(event)
        event_id = _string(rollout.get("event_id"), "state-bank rollout.event_id")
        if event_id in seen_event_ids:
            raise AssemblyError(
                f"materialized bank contains duplicate event_id {event_id!r}"
            )
        if review.get("event_id") != event_id:
            raise AssemblyError(
                f"materialized review event_id differs for {event_id!r}"
            )
        seen_event_ids.add(event_id)
        rollouts.append(rollout)
        reviews.append(review)
    if not rollouts:
        raise AssemblyError("materialized StateBank must contain at least one event")
    return rollouts, reviews


def materialize_duplicate_trajectory_state_banks(
    *,
    output_dir: str | Path,
    assembly: Mapping[str, Any],
    source_checkpoint: Mapping[str, Any],
    prompt_identity_sha256: str,
    source_artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Write and validate all five immutable StateBanks from a draft assembly."""

    # Keep imports inside the final adapter: the draft constructor remains
    # runnable during a core-schema implementation slice and never imports
    # trainer/loss code merely to slice integer rows.
    from src.rollout_calibration import assemble_state_bank, load_state_bank

    root = Path(output_dir).expanduser().resolve()
    if root.exists():
        raise AssemblyError(f"materialized output directory already exists: {root}")
    banks = _mapping(assembly.get("banks"), "assembly.banks")
    root.mkdir(parents=True)
    arm_receipts: dict[str, Any] = {}
    for name in BANK_NAMES:
        bank = _mapping(banks.get(name), f"assembly.banks[{name}]")
        rollouts, reviews = state_bank_rows_from_duplicate_draft_bank(bank)
        arm_root = root / name
        _write_jsonl(arm_root / "pre-state-bank" / "rollout_rows.jsonl", rollouts)
        _write_jsonl(arm_root / "pre-state-bank" / "review_rows.jsonl", reviews)
        manifest = assemble_state_bank(
            output_dir=arm_root / "state-bank",
            rollout_rows=rollouts,
            review_rows=reviews,
            source_checkpoint=source_checkpoint,
            prompt_identity_sha256=prompt_identity_sha256,
            source_artifacts=source_artifacts,
        )
        loaded = load_state_bank(
            arm_root / "state-bank" / "manifest.json",
            expected_source_checkpoint=source_checkpoint,
            expected_prompt_identity_sha256=prompt_identity_sha256,
        )
        arm_receipts[name] = {
            "state_bank_manifest": manifest.to_artifact_dict(),
            "state_bank_validation_receipt": loaded.validation_receipt.to_artifact_dict(),
            "rollout_row_count": len(rollouts),
            "review_row_count": len(reviews),
        }
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "materialized",
        "draft_allocation_receipt": _clone(
            _mapping(assembly.get("allocation_receipt"), "assembly.allocation_receipt")
        ),
        "arms": arm_receipts,
    }
    _write_json(root / "materialization-receipt.json", receipt)
    return receipt


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise AssemblyError(f"output already exists: {path}")
    path.write_bytes(_canonical(value) + b"\n")


def _write_jsonl(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise AssemblyError(f"output already exists: {path}")
    with path.open("wb") as handle:
        for value in values:
            handle.write(_canonical(dict(value)) + b"\n")


def write_duplicate_trajectory_bank_drafts(
    output_dir: str | Path, assembly: Mapping[str, Any]
) -> None:
    """Write immutable per-arm draft JSONL and one cross-arm allocation receipt."""

    root = Path(output_dir).expanduser().resolve()
    if root.exists():
        raise AssemblyError(f"output directory already exists: {root}")
    banks = _mapping(assembly.get("banks"), "assembly.banks")
    receipt = _mapping(
        assembly.get("allocation_receipt"), "assembly.allocation_receipt"
    )
    if set(banks) != set(BANK_NAMES):
        raise AssemblyError("assembly must contain exactly the five matched bank names")
    root.mkdir(parents=True)
    for name in BANK_NAMES:
        bank = _mapping(banks[name], f"assembly.banks[{name}]")
        arm_root = root / name
        _write_jsonl(
            arm_root / "pre-state-bank" / "events.jsonl",
            _sequence(bank.get("events"), f"{name}.events"),
        )
        _write_json(arm_root / "optimizer-windows.json", bank.get("optimizer_windows"))
        _write_json(
            arm_root / "allocation-receipt.json", bank.get("allocation_receipt")
        )
    _write_json(root / "allocation-receipt.json", receipt)


def _load_json(path: str | Path) -> Any:
    resolved = Path(path).expanduser().resolve(strict=True)
    try:
        return json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AssemblyError(f"invalid JSON: {resolved}: {exc}") from exc


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    resolved = Path(path).expanduser().resolve(strict=True)
    rows: list[dict[str, Any]] = []
    with resolved.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AssemblyError(
                    f"invalid JSONL at {resolved}:{line_number}: {exc}"
                ) from exc
            rows.append(dict(_mapping(value, f"{resolved}:{line_number}")))
    return rows


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rollout-trajectories",
        type=Path,
        required=True,
        help="JSON array of exact rollout trajectories",
    )
    parser.add_argument(
        "--reviewed-owner-ledger",
        type=Path,
        required=True,
        help="JSONL reviewed row ledger",
    )
    parser.add_argument(
        "--source-preservation-events",
        type=Path,
        required=True,
        help="JSONL one matched Source event per admitted burst",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--materialize-state-banks",
        action="store_true",
        help="also write and validate typed StateBanks under OUTPUT_DIR/state-banks",
    )
    parser.add_argument(
        "--source-checkpoint-json",
        type=Path,
        help="strict CheckpointIdentity JSON object; required with --materialize-state-banks",
    )
    parser.add_argument(
        "--prompt-identity-sha256",
        help="strict prompt identity digest; required with --materialize-state-banks",
    )
    parser.add_argument(
        "--source-artifacts-json",
        type=Path,
        help="JSON array of immutable source artifact id/SHA records; required with --materialize-state-banks",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    trajectories = _sequence(
        _load_json(args.rollout_trajectories), "rollout trajectory JSON"
    )
    assembly = build_duplicate_trajectory_bank_drafts(
        rollout_trajectories=[
            _mapping(value, f"rollout_trajectories[{index}]")
            for index, value in enumerate(trajectories)
        ],
        reviewed_owner_ledger=_load_jsonl(args.reviewed_owner_ledger),
        source_preservation_events=_load_jsonl(args.source_preservation_events),
    )
    write_duplicate_trajectory_bank_drafts(args.output_dir, assembly)
    receipt = _mapping(assembly["allocation_receipt"], "allocation_receipt")
    status = "assembled_drafts"
    if args.materialize_state_banks:
        if (
            args.source_checkpoint_json is None
            or args.prompt_identity_sha256 is None
            or args.source_artifacts_json is None
        ):
            raise SystemExit(
                "--materialize-state-banks requires --source-checkpoint-json, "
                "--prompt-identity-sha256, and --source-artifacts-json"
            )
        source_checkpoint = _mapping(
            _load_json(args.source_checkpoint_json), "source checkpoint JSON"
        )
        source_artifacts = _sequence(
            _load_json(args.source_artifacts_json), "source artifacts JSON"
        )
        materialize_duplicate_trajectory_state_banks(
            output_dir=args.output_dir.expanduser().resolve() / "state-banks",
            assembly=assembly,
            source_checkpoint=source_checkpoint,
            prompt_identity_sha256=_sha256(
                args.prompt_identity_sha256, "prompt_identity_sha256"
            ),
            source_artifacts=[
                _mapping(value, f"source_artifacts[{index}]")
                for index, value in enumerate(source_artifacts)
            ],
        )
        status = "assembled_and_materialized"
    print(
        json.dumps(
            {
                "status": status,
                "output_dir": str(args.output_dir.expanduser().resolve()),
                "burst_count": receipt["admitted_burst_count"],
                "image_count": receipt["admitted_image_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
