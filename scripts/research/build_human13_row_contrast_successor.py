#!/usr/bin/env python3
"""Build the immutable Human-13 row-contrast successor sidecar.

The sealed K-union manifest remains authoritative for owners and native rows.
This module derives complete-row negative states and optional, exactly aligned
prior-output states without loading a model or changing the historical
manifest.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Literal, Mapping, Sequence


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.research.analyze_human13_k_union import (
    _match_prefix,
    _ordered_predictions,
)
from scripts.research.build_human13_k_union_manifest import (
    Human13KUnionManifest,
    ImageRecord,
    load_manifest,
)
from scripts.research.run_greedy_prefix_forced_owner_path import (
    BOX_END,
    BOX_START,
    OBJECT_REF_END,
    OBJECT_REF_START,
)


SCHEMA_VERSION = "human13_row_contrast_successor.v1"
COORD_TOKEN_START = 151670
COORD_TOKEN_END_EXCLUSIVE = 152670
_HEX = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class PriorOutputSpec:
    arm_id: str
    milestone: int
    outputs_path: Path
    receipt_path: Path


@dataclass(frozen=True)
class SuccessorRow:
    row_id: str
    trajectory_id: str
    owner_id: str | None
    category: str
    token_ids: tuple[int, ...]
    description_offsets: tuple[int, ...]
    coordinate_offsets: tuple[int, ...]


@dataclass(frozen=True)
class OwnerCandidateGroup:
    owner_id: str
    category: str
    rows: tuple[SuccessorRow, ...]


@dataclass(frozen=True)
class DuplicateContrastEvent:
    event_id: str
    source_kind: Literal["manifest", "prior_output"]
    source_id: str
    image_id: int
    trajectory_id: str
    retained_row_id: str
    prefix_token_ids: tuple[int, ...]
    duplicate_row: SuccessorRow
    target_owner_ids: tuple[str, ...]
    covered_owner_ids: tuple[str, ...]
    uncovered_owner_ids: tuple[str, ...]
    candidate_groups: tuple[OwnerCandidateGroup, ...]
    contrast_branch: Literal["same_description", "cross_description", "fallback"]


@dataclass(frozen=True)
class PositiveRow:
    owner_id: str
    stratum: Literal["G", "H"]
    row: SuccessorRow


@dataclass(frozen=True)
class PriorSourceReceipt:
    arm_id: str
    milestone: int
    outputs_path: str
    outputs_sha256: str
    receipt_path: str
    receipt_sha256: str
    event_count: int


@dataclass(frozen=True)
class SourceExclusion:
    source_id: str
    code: str
    message: str


@dataclass(frozen=True)
class Human13RowContrastLedger:
    schema_version: str
    manifest_path: str
    manifest_sha256: str
    panel_sha256: str
    tokenizer_sha256: str
    coordinate_token_start: int
    coordinate_token_end_exclusive: int
    positive_rows: tuple[PositiveRow, ...]
    g_watch_rows: tuple[PositiveRow, ...]
    events: tuple[DuplicateContrastEvent, ...]
    prior_sources: tuple[PriorSourceReceipt, ...]
    exclusions: tuple[SourceExclusion, ...]


class PriorOutputAlignmentError(ValueError):
    """Raised when an optional prior output cannot be admitted exactly."""


def build_successor_ledger(
    manifest: Human13KUnionManifest,
    *,
    manifest_path: str | Path,
    prior_outputs: Sequence[PriorOutputSpec] = (),
) -> Human13RowContrastLedger:
    manifest_target = Path(manifest_path).resolve(strict=True)
    manifest_sha256 = _verify_adjacent_digest(manifest_target, "manifest")
    if manifest.binding.panel.owner_count != 392 or not manifest.full_panel:
        raise ValueError("successor requires the sealed full Human-13 manifest")

    positive_rows: list[PositiveRow] = []
    g_watch_rows: list[PositiveRow] = []
    manifest_events: list[DuplicateContrastEvent] = []
    image_context: dict[int, _ImageContext] = {}
    for image in manifest.images:
        context = _image_context(image)
        image_context[image.image_id] = context
        for owner_id in image.g_owner_ids:
            row = context.preferred_rows[owner_id]
            positive = PositiveRow(owner_id=owner_id, stratum="G", row=row)
            positive_rows.append(positive)
            g_watch_rows.append(positive)
        for owner_id in image.h_owner_ids:
            positive_rows.append(
                PositiveRow(
                    owner_id=owner_id,
                    stratum="H",
                    row=context.preferred_rows[owner_id],
                )
            )
        manifest_events.extend(_manifest_events(image, context))

    prior_events: list[DuplicateContrastEvent] = []
    sources: list[PriorSourceReceipt] = []
    exclusions: list[SourceExclusion] = []
    for spec in prior_outputs:
        source_id = f"{spec.arm_id}@{spec.milestone}"
        try:
            events, receipt = _prior_output_events(
                manifest,
                image_context,
                spec,
                manifest_sha256=manifest_sha256,
            )
        except (OSError, ValueError, TypeError, KeyError) as exc:
            exclusions.append(
                SourceExclusion(
                    source_id=source_id,
                    code="prior_output.alignment_failed",
                    message=str(exc),
                )
            )
            continue
        prior_events.extend(events)
        sources.append(receipt)

    events_by_identity: dict[
        tuple[int, tuple[int, ...], tuple[int, ...]], DuplicateContrastEvent
    ] = {}
    for event in (*manifest_events, *prior_events):
        key = (
            event.image_id,
            event.prefix_token_ids,
            event.duplicate_row.token_ids,
        )
        events_by_identity.setdefault(key, event)

    ledger = Human13RowContrastLedger(
        schema_version=SCHEMA_VERSION,
        manifest_path=str(manifest_target),
        manifest_sha256=manifest_sha256,
        panel_sha256=manifest.binding.panel.panel_sha256,
        tokenizer_sha256=manifest.binding.surface.tokenizer_sha256,
        coordinate_token_start=COORD_TOKEN_START,
        coordinate_token_end_exclusive=COORD_TOKEN_END_EXCLUSIVE,
        positive_rows=tuple(
            sorted(
                positive_rows, key=lambda item: (item.row.trajectory_id, item.owner_id)
            )
        ),
        g_watch_rows=tuple(
            sorted(
                g_watch_rows, key=lambda item: (item.row.trajectory_id, item.owner_id)
            )
        ),
        events=tuple(
            sorted(
                events_by_identity.values(),
                key=lambda item: (item.image_id, item.source_kind, item.event_id),
            )
        ),
        prior_sources=tuple(sorted(sources, key=lambda item: item.milestone)),
        exclusions=tuple(sorted(exclusions, key=lambda item: item.source_id)),
    )
    validate_ledger(ledger)
    return ledger


@dataclass(frozen=True)
class _ImageContext:
    row_by_id: Mapping[str, SuccessorRow]
    owner_by_row: Mapping[str, str]
    aliases_by_owner: Mapping[str, tuple[SuccessorRow, ...]]
    preferred_rows: Mapping[str, SuccessorRow]
    category_by_owner: Mapping[str, str]


def _image_context(image: ImageRecord) -> _ImageContext:
    owner_by_row = {
        row_id: owner.owner_id
        for owner in image.owners
        for row_id in (*owner.source_row_ids, *owner.sampled_row_ids)
    }
    category_by_owner = {owner.owner_id: owner.category for owner in image.owners}
    row_by_id: dict[str, SuccessorRow] = {}
    for trajectory in image.trajectories:
        retained = set(trajectory.retained_row_ids)
        matched = set(trajectory.matched_row_ids)
        for row in trajectory.rows:
            owner_id = owner_by_row.get(row.row_id)
            record = _row_record(
                row_id=row.row_id,
                trajectory_id=trajectory.trajectory_id,
                owner_id=owner_id,
                category=row.category,
                token_ids=trajectory.raw_token_ids[row.token_start : row.token_end],
            )
            row_by_id[row.row_id] = record
            if owner_id is not None and (
                row.row_id not in retained or row.row_id not in matched
            ):
                raise ValueError("owner-positive row is not retained and matched")

    aliases: dict[str, tuple[SuccessorRow, ...]] = {}
    preferred: dict[str, SuccessorRow] = {}
    selected_by_owner = {row.owner_id: row.row_id for row in image.selected_rows}
    for owner in image.owners:
        ids = tuple(dict.fromkeys((*owner.source_row_ids, *owner.sampled_row_ids)))
        rows = tuple(
            sorted(
                {row_by_id[row_id] for row_id in ids if row_id in row_by_id},
                key=lambda item: (item.trajectory_id, item.row_id),
            )
        )
        if owner.stratum in {"G", "H"} and not rows:
            raise ValueError(f"target owner {owner.owner_id} has no native row alias")
        aliases[owner.owner_id] = rows
        if owner.stratum == "G":
            source_rows = [row_by_id[row_id] for row_id in owner.source_row_ids]
            if not source_rows:
                raise ValueError(f"G owner {owner.owner_id} has no Source row")
            preferred[owner.owner_id] = sorted(
                source_rows, key=lambda item: (item.trajectory_id, item.row_id)
            )[0]
        elif owner.stratum == "H":
            selected_id = selected_by_owner.get(owner.owner_id)
            if selected_id not in row_by_id:
                raise ValueError(f"H owner {owner.owner_id} lacks its selected row")
            preferred[owner.owner_id] = row_by_id[selected_id]
    return _ImageContext(
        row_by_id=row_by_id,
        owner_by_row=owner_by_row,
        aliases_by_owner=aliases,
        preferred_rows=preferred,
        category_by_owner=category_by_owner,
    )


def _row_record(
    *,
    row_id: str,
    trajectory_id: str,
    owner_id: str | None,
    category: str,
    token_ids: Iterable[int],
) -> SuccessorRow:
    tokens = tuple(int(token) for token in token_ids)
    if not tokens or tokens[0] != OBJECT_REF_START or tokens[-1] != BOX_END:
        raise ValueError(f"row {row_id} is not one complete canonical row")
    try:
        object_end = tokens.index(OBJECT_REF_END)
        box_start = tokens.index(BOX_START, object_end + 1)
    except ValueError as exc:
        raise ValueError(f"row {row_id} lacks canonical row markers") from exc
    coords = tuple(
        index
        for index, token in enumerate(tokens)
        if COORD_TOKEN_START <= token < COORD_TOKEN_END_EXCLUSIVE
    )
    if len(coords) != 4 or coords != tuple(range(box_start + 1, box_start + 5)):
        raise ValueError(f"row {row_id} does not contain four aligned coordinates")
    if box_start + 5 >= len(tokens) or tokens[box_start + 5] != BOX_END:
        raise ValueError(f"row {row_id} has tokens after its coordinate body")
    description = tuple(range(1, object_end))
    if not description:
        raise ValueError(f"row {row_id} has no owner-distinguishing description")
    return SuccessorRow(
        row_id=row_id,
        trajectory_id=trajectory_id,
        owner_id=owner_id,
        category=str(category),
        token_ids=tokens,
        description_offsets=description,
        coordinate_offsets=coords,
    )


def _manifest_events(
    image: ImageRecord, context: _ImageContext
) -> tuple[DuplicateContrastEvent, ...]:
    trajectories = {item.trajectory_id: item for item in image.trajectories}
    target = tuple(sorted((*image.g_owner_ids, *image.h_owner_ids)))
    events: list[DuplicateContrastEvent] = []
    for event in image.duplicate_events:
        trajectory = trajectories[event.trajectory_id]
        row = next(
            item for item in trajectory.rows if item.row_id == event.duplicate_row_id
        )
        duplicate = context.row_by_id[row.row_id]
        covered = _covered_manifest_owners(
            trajectory, row.token_start, context.owner_by_row, target
        )
        events.append(
            _event(
                event_id=event.event_id,
                source_kind="manifest",
                source_id="sealed_manifest",
                image=image,
                context=context,
                trajectory_id=trajectory.trajectory_id,
                retained_row_id=event.retained_row_id,
                prefix_token_ids=trajectory.raw_token_ids[: row.token_start],
                duplicate=duplicate,
                covered=covered,
            )
        )
    return tuple(events)


def _covered_manifest_owners(
    trajectory: Any,
    row_start: int,
    owner_by_row: Mapping[str, str],
    target_owner_ids: Sequence[str],
) -> tuple[str, ...]:
    retained = set(trajectory.retained_row_ids)
    matched = set(trajectory.matched_row_ids)
    target = set(target_owner_ids)
    return tuple(
        sorted(
            {
                owner_by_row[row.row_id]
                for row in trajectory.rows
                if row.token_end <= row_start
                and row.row_id in retained
                and row.row_id in matched
                and row.row_id in owner_by_row
                and owner_by_row[row.row_id] in target
            }
        )
    )


def _event(
    *,
    event_id: str,
    source_kind: Literal["manifest", "prior_output"],
    source_id: str,
    image: ImageRecord,
    context: _ImageContext,
    trajectory_id: str,
    retained_row_id: str,
    prefix_token_ids: Iterable[int],
    duplicate: SuccessorRow,
    covered: Sequence[str],
) -> DuplicateContrastEvent:
    target = tuple(sorted((*image.g_owner_ids, *image.h_owner_ids)))
    covered_ids = tuple(sorted(set(covered) & set(target)))
    uncovered = tuple(sorted(set(target) - set(covered_ids)))
    groups = tuple(
        OwnerCandidateGroup(
            owner_id=owner_id,
            category=context.category_by_owner[owner_id],
            rows=context.aliases_by_owner[owner_id],
        )
        for owner_id in uncovered
        if context.aliases_by_owner.get(owner_id)
    )
    normalized_duplicate = _normalize_category(duplicate.category)
    if any(
        _normalize_category(group.category) == normalized_duplicate for group in groups
    ):
        branch: Literal["same_description", "cross_description", "fallback"] = (
            "same_description"
        )
    elif groups:
        branch = "cross_description"
    else:
        branch = "fallback"
    return DuplicateContrastEvent(
        event_id=event_id,
        source_kind=source_kind,
        source_id=source_id,
        image_id=image.image_id,
        trajectory_id=trajectory_id,
        retained_row_id=retained_row_id,
        prefix_token_ids=tuple(int(token) for token in prefix_token_ids),
        duplicate_row=duplicate,
        target_owner_ids=target,
        covered_owner_ids=covered_ids,
        uncovered_owner_ids=uncovered,
        candidate_groups=groups,
        contrast_branch=branch,
    )


def _prior_output_events(
    manifest: Human13KUnionManifest,
    contexts: Mapping[int, _ImageContext],
    spec: PriorOutputSpec,
    *,
    manifest_sha256: str,
) -> tuple[tuple[DuplicateContrastEvent, ...], PriorSourceReceipt]:
    if spec.arm_id != "A4" or spec.milestone not in {1, 2}:
        raise PriorOutputAlignmentError(
            "only authoritative A4 milestones 1/2 are admitted"
        )
    outputs_path = Path(spec.outputs_path).resolve(strict=True)
    receipt_path = Path(spec.receipt_path).resolve(strict=True)
    outputs_sha = _sha256_file(outputs_path)
    receipt_sha = _sha256_file(receipt_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("arm_id") != spec.arm_id
        or receipt.get("manifest_sha256") != manifest_sha256
    ):
        raise PriorOutputAlignmentError("prior receipt arm/manifest identity mismatch")
    declared = [
        item
        for item in receipt.get("outputs", ())
        if int(item.get("milestone", -1)) == spec.milestone
    ]
    if len(declared) != 1 or declared[0].get("sha256") != outputs_sha:
        raise PriorOutputAlignmentError(
            "prior output digest is not sealed by its receipt"
        )

    rows = [
        json.loads(line)
        for line in outputs_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    image_by_id = {item.image_id: item for item in manifest.images}
    if len(rows) != len(image_by_id) or {
        int(item.get("image_id", -1)) for item in rows
    } != set(image_by_id):
        raise PriorOutputAlignmentError(
            "prior output does not cover the exact panel once"
        )
    events: list[DuplicateContrastEvent] = []
    for output in rows:
        image_id = int(output["image_id"])
        image = image_by_id[image_id]
        _validate_prior_row(
            output,
            image=image,
            manifest=manifest,
            manifest_sha256=manifest_sha256,
            spec=spec,
        )
        predictions = _ordered_predictions(output)
        token_ids = tuple(int(token) for token in output["generated_token_ids"])
        spans = _generated_row_spans(token_ids)
        if any(int(item["generated_order"]) >= len(spans) for item in predictions):
            raise PriorOutputAlignmentError(
                "prediction order escapes generated row spans"
            )
        natural = _match_prefix(
            image,
            predictions,
            duplicate_iou_threshold=manifest.binding.matcher.duplicate_iou_threshold,
            owner_iou_threshold=manifest.binding.matcher.owner_iou_threshold,
        )
        by_order = {int(item["generated_order"]): item for item in predictions}
        trajectory_id = str(output["provenance"]["trajectory_id"])
        for duplicate_info in natural["duplicate_rows"]:
            order = int(duplicate_info["generated_order"])
            retained_order = int(duplicate_info["retained_generated_order"])
            start, end = spans[order]
            prediction = by_order[order]
            duplicate = _row_record(
                row_id=f"prior:{spec.arm_id}:{spec.milestone}:{image_id}:row:{order:03d}",
                trajectory_id=trajectory_id,
                owner_id=None,
                category=str(
                    prediction.get("description", prediction.get("category", ""))
                ),
                token_ids=token_ids[start:end],
            )
            prefix_predictions = [
                item for item in predictions if int(item["generated_order"]) < order
            ]
            covered_result = _match_prefix(
                image,
                prefix_predictions,
                duplicate_iou_threshold=manifest.binding.matcher.duplicate_iou_threshold,
                owner_iou_threshold=manifest.binding.matcher.owner_iou_threshold,
            )
            events.append(
                _event(
                    event_id=f"prior:{spec.arm_id}:{spec.milestone}:{image_id}:dup:{order:03d}",
                    source_kind="prior_output",
                    source_id=f"{spec.arm_id}@{spec.milestone}",
                    image=image,
                    context=contexts[image_id],
                    trajectory_id=trajectory_id,
                    retained_row_id=(
                        f"prior:{spec.arm_id}:{spec.milestone}:{image_id}:row:{retained_order:03d}"
                    ),
                    prefix_token_ids=token_ids[:start],
                    duplicate=duplicate,
                    covered=tuple(covered_result["owner_matches"]),
                )
            )
    return (
        tuple(events),
        PriorSourceReceipt(
            arm_id=spec.arm_id,
            milestone=spec.milestone,
            outputs_path=str(outputs_path),
            outputs_sha256=outputs_sha,
            receipt_path=str(receipt_path),
            receipt_sha256=receipt_sha,
            event_count=len(events),
        ),
    )


def _validate_prior_row(
    output: Mapping[str, Any],
    *,
    image: ImageRecord,
    manifest: Human13KUnionManifest,
    manifest_sha256: str,
    spec: PriorOutputSpec,
) -> None:
    if (
        output.get("arm_id") != spec.arm_id
        or int(output.get("milestone", -1)) != spec.milestone
    ):
        raise PriorOutputAlignmentError("prior output arm/milestone mismatch")
    if (
        output.get("decode_mode") != "original_prompt_clean_greedy"
        or float(output.get("repetition_penalty", -1.0)) != 1.0
    ):
        raise PriorOutputAlignmentError(
            "prior output is not the exact clean-greedy surface"
        )
    provenance = output.get("provenance")
    if not isinstance(provenance, Mapping):
        raise PriorOutputAlignmentError("prior output lacks provenance")
    checks = {
        "manifest_sha256": manifest_sha256,
        "tokenizer_sha256": manifest.binding.surface.tokenizer_sha256,
        "prompt_policy_fingerprint": manifest.binding.surface.prompt_policy_fingerprint,
        "panel_sha256": manifest.binding.panel.panel_sha256,
        "image_sha256": image.image_sha256,
        "arm_id": spec.arm_id,
        "milestone": spec.milestone,
        "physical_batch_size": 1,
        "backend": "hf",
    }
    for field, expected in checks.items():
        if provenance.get(field) != expected:
            raise PriorOutputAlignmentError(f"prior output provenance {field} mismatch")
    token_ids = output.get("generated_token_ids")
    if (
        not isinstance(token_ids, list)
        or not token_ids
        or any(
            isinstance(item, bool) or not isinstance(item, int) for item in token_ids
        )
    ):
        raise PriorOutputAlignmentError("prior output token IDs are invalid")


def _generated_row_spans(token_ids: Sequence[int]) -> tuple[tuple[int, int], ...]:
    spans: list[tuple[int, int]] = []
    cursor = 0
    while cursor < len(token_ids) and token_ids[cursor] == OBJECT_REF_START:
        try:
            end = token_ids.index(BOX_END, cursor + 1) + 1
        except ValueError as exc:
            raise PriorOutputAlignmentError("generated row has no BOX_END") from exc
        _row_record(
            row_id=f"aligned:{len(spans)}",
            trajectory_id="alignment",
            owner_id=None,
            category="alignment",
            token_ids=token_ids[cursor:end],
        )
        spans.append((cursor, end))
        cursor = end
    trailing = tuple(token_ids[cursor:])
    if len(trailing) > 1 or (trailing and trailing[0] == OBJECT_REF_START):
        raise PriorOutputAlignmentError(
            "generated tokens contain an unaligned trailing span"
        )
    return tuple(spans)


def _normalize_category(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def validate_ledger(value: Human13RowContrastLedger) -> None:
    if value.schema_version != SCHEMA_VERSION:
        raise ValueError("successor ledger schema is not supported")
    for digest in (value.manifest_sha256, value.panel_sha256, value.tokenizer_sha256):
        if len(digest) != 64 or set(digest) - _HEX:
            raise ValueError("successor ledger digest must be lowercase SHA-256")
    if (value.coordinate_token_start, value.coordinate_token_end_exclusive) != (
        COORD_TOKEN_START,
        COORD_TOKEN_END_EXCLUSIVE,
    ):
        raise ValueError("coordinate token identity changed")
    identities: set[tuple[int, tuple[int, ...], tuple[int, ...]]] = set()
    for event in value.events:
        if set(event.covered_owner_ids) & set(event.uncovered_owner_ids) or set(
            event.covered_owner_ids
        ) | set(event.uncovered_owner_ids) != set(event.target_owner_ids):
            raise ValueError("duplicate event owner partition is invalid")
        if tuple(group.owner_id for group in event.candidate_groups) != tuple(
            sorted(group.owner_id for group in event.candidate_groups)
        ):
            raise ValueError("candidate owner groups are not canonical")
        key = (event.image_id, event.prefix_token_ids, event.duplicate_row.token_ids)
        if key in identities:
            raise ValueError("duplicate event identity is repeated")
        identities.add(key)
        _validate_row(event.duplicate_row)
        for group in event.candidate_groups:
            if not group.rows or any(
                row.owner_id != group.owner_id for row in group.rows
            ):
                raise ValueError("candidate group aliases do not bind one owner")
            for row in group.rows:
                _validate_row(row)
    for positive in (*value.positive_rows, *value.g_watch_rows):
        if positive.row.owner_id != positive.owner_id:
            raise ValueError("positive row owner identity mismatch")
        _validate_row(positive.row)
    if any(item.stratum != "G" for item in value.g_watch_rows):
        raise ValueError("G watch rows must contain only G owners")


def _validate_row(row: SuccessorRow) -> None:
    if len(row.coordinate_offsets) != 4 or any(
        index < 0 or index >= len(row.token_ids) for index in row.coordinate_offsets
    ):
        raise ValueError("row coordinate offsets are invalid")
    if any(
        not COORD_TOKEN_START <= row.token_ids[index] < COORD_TOKEN_END_EXCLUSIVE
        for index in row.coordinate_offsets
    ):
        raise ValueError("row coordinate token identity is invalid")


def canonical_write(value: Human13RowContrastLedger, path: str | Path) -> str:
    validate_ledger(value)
    target = Path(path)
    digest_path = Path(f"{target}.sha256")
    if target.exists() or digest_path.exists():
        raise FileExistsError(f"refusing to overwrite successor ledger: {target}")
    payload = _canonical_bytes(value)
    digest = hashlib.sha256(payload).hexdigest()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    digest_path.write_text(f"{digest}  {target.name}\n", encoding="ascii")
    return digest


def load_ledger(path: str | Path) -> Human13RowContrastLedger:
    target = Path(path)
    raw = target.read_bytes()
    digest = _verify_adjacent_digest(target, "successor ledger")
    document = json.loads(raw)
    value = _ledger_from_dict(document)
    if raw != _canonical_bytes(value):
        raise ValueError("successor ledger is not canonically serialized")
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("successor ledger digest mismatch")
    validate_ledger(value)
    manifest_target = Path(value.manifest_path)
    if (
        manifest_target.is_file()
        and _sha256_file(manifest_target) != value.manifest_sha256
    ):
        raise ValueError("successor ledger manifest identity no longer matches")
    return value


def _ledger_from_dict(value: Mapping[str, Any]) -> Human13RowContrastLedger:
    def row(item: Mapping[str, Any]) -> SuccessorRow:
        return SuccessorRow(
            row_id=str(item["row_id"]),
            trajectory_id=str(item["trajectory_id"]),
            owner_id=item["owner_id"],
            category=str(item["category"]),
            token_ids=tuple(int(x) for x in item["token_ids"]),
            description_offsets=tuple(int(x) for x in item["description_offsets"]),
            coordinate_offsets=tuple(int(x) for x in item["coordinate_offsets"]),
        )

    def positive(item: Mapping[str, Any]) -> PositiveRow:
        return PositiveRow(
            owner_id=str(item["owner_id"]),
            stratum=item["stratum"],
            row=row(item["row"]),
        )

    def event(item: Mapping[str, Any]) -> DuplicateContrastEvent:
        return DuplicateContrastEvent(
            event_id=str(item["event_id"]),
            source_kind=item["source_kind"],
            source_id=str(item["source_id"]),
            image_id=int(item["image_id"]),
            trajectory_id=str(item["trajectory_id"]),
            retained_row_id=str(item["retained_row_id"]),
            prefix_token_ids=tuple(int(x) for x in item["prefix_token_ids"]),
            duplicate_row=row(item["duplicate_row"]),
            target_owner_ids=tuple(str(x) for x in item["target_owner_ids"]),
            covered_owner_ids=tuple(str(x) for x in item["covered_owner_ids"]),
            uncovered_owner_ids=tuple(str(x) for x in item["uncovered_owner_ids"]),
            candidate_groups=tuple(
                OwnerCandidateGroup(
                    owner_id=str(group["owner_id"]),
                    category=str(group["category"]),
                    rows=tuple(row(alias) for alias in group["rows"]),
                )
                for group in item["candidate_groups"]
            ),
            contrast_branch=item["contrast_branch"],
        )

    return Human13RowContrastLedger(
        schema_version=str(value["schema_version"]),
        manifest_path=str(value["manifest_path"]),
        manifest_sha256=str(value["manifest_sha256"]),
        panel_sha256=str(value["panel_sha256"]),
        tokenizer_sha256=str(value["tokenizer_sha256"]),
        coordinate_token_start=int(value["coordinate_token_start"]),
        coordinate_token_end_exclusive=int(value["coordinate_token_end_exclusive"]),
        positive_rows=tuple(positive(item) for item in value["positive_rows"]),
        g_watch_rows=tuple(positive(item) for item in value["g_watch_rows"]),
        events=tuple(event(item) for item in value["events"]),
        prior_sources=tuple(
            PriorSourceReceipt(**item) for item in value["prior_sources"]
        ),
        exclusions=tuple(SourceExclusion(**item) for item in value["exclusions"]),
    )


def _canonical_bytes(value: Human13RowContrastLedger) -> bytes:
    return (
        json.dumps(
            asdict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _verify_adjacent_digest(path: Path, label: str) -> str:
    digest = _sha256_file(path)
    receipt_path = Path(f"{path}.sha256")
    if not receipt_path.is_file():
        raise ValueError(f"{label} digest is missing")
    if receipt_path.read_text(encoding="ascii") != f"{digest}  {path.name}\n":
        raise ValueError(f"{label} digest does not match canonical bytes")
    return digest


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--prior-output",
        action="append",
        default=[],
        metavar="MILESTONE:OUTPUTS_JSONL:RECEIPT_JSON",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    specs: list[PriorOutputSpec] = []
    for raw in args.prior_output:
        milestone, outputs, receipt = raw.split(":", 2)
        specs.append(
            PriorOutputSpec(
                arm_id="A4",
                milestone=int(milestone),
                outputs_path=Path(outputs),
                receipt_path=Path(receipt),
            )
        )
    manifest = load_manifest(args.manifest)
    ledger = build_successor_ledger(
        manifest, manifest_path=args.manifest, prior_outputs=tuple(specs)
    )
    digest = canonical_write(ledger, args.output)
    print(
        json.dumps(
            {
                "schema_version": ledger.schema_version,
                "manifest_sha256": ledger.manifest_sha256,
                "event_count": len(ledger.events),
                "prior_source_count": len(ledger.prior_sources),
                "exclusion_count": len(ledger.exclusions),
                "output": str(args.output),
                "sha256": digest,
                "model_actions": 0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "COORD_TOKEN_END_EXCLUSIVE",
    "COORD_TOKEN_START",
    "DuplicateContrastEvent",
    "Human13RowContrastLedger",
    "OwnerCandidateGroup",
    "PositiveRow",
    "PriorOutputSpec",
    "SuccessorRow",
    "build_successor_ledger",
    "canonical_write",
    "load_ledger",
    "validate_ledger",
]
