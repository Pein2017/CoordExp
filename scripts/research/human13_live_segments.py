"""CPU-only materialization of exact Human-13 causal segments.

This module consumes an already loaded canonical manifest and processor-only
encoded skeletons.  It never tokenizes, loads a model, or owns prompt identity.
"""

from __future__ import annotations

from dataclasses import dataclass, is_dataclass, replace
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from scripts.research.run_human13_k_union_overfit import (
    Human13EncodedRowBinding,
    LogicalPanelSegment,
)


@dataclass(frozen=True)
class MaterializedSegments:
    segments: tuple[LogicalPanelSegment, ...]
    a4_segments: tuple[LogicalPanelSegment, ...]
    a4_aggregate_lengths: tuple[tuple[int, int], ...]

    def preflight(self, global_max_length: int = 12_000) -> None:
        for image_id, total in self.a4_aggregate_lengths:
            if total > global_max_length:
                raise ValueError(
                    f"A4 image {image_id} aggregate length {total} exceeds 12,000"
                )
        for segment in (*self.segments, *self.a4_segments):
            if segment.encoded_length > global_max_length:
                raise ValueError(
                    f"segment {segment.segment_id} length {segment.encoded_length} exceeds 12,000"
                )


def materialize_segments(
    manifest: Any,
    skeletons: Mapping[int, Any],
    *,
    prompt_token_counts: Mapping[int, int] | None = None,
    global_max_length: int = 12_000,
) -> MaterializedSegments:
    """Build exact token concatenations and row bindings from manifest spans.

    ``skeletons`` are processor-only objects with ``input_ids`` and
    ``owner_row_tokens`` (full-GT body rows).  Stored manifest row token IDs are
    copied directly; no suffix is re-tokenized.
    """
    if not skeletons:
        raise ValueError("Human-13 materialization requires skeletons")
    if prompt_token_counts is None:
        prompt_token_counts = {
            int(image_id): int(getattr(skeleton, "prompt_token_count"))
            for image_id, skeleton in skeletons.items()
        }
    segments: list[LogicalPanelSegment] = []
    a4_segments: list[LogicalPanelSegment] = []
    a4_totals: list[tuple[int, int]] = []
    for image in manifest.images:
        image_id = int(image.image_id)
        if image_id not in skeletons:
            raise ValueError(f"missing processor-only skeleton for image {image_id}")
        skeleton = skeletons[image_id]
        prompt_count = int(prompt_token_counts[image_id])
        prompt = tuple(int(x) for x in skeleton.input_ids[:prompt_count])
        if len(prompt) != prompt_count:
            raise ValueError(f"invalid prompt token count for image {image_id}")

        selected = {row.row_id: row for row in image.selected_rows}
        h_rows = tuple(row for row in image.selected_rows if row.owner_id in {o.owner_id for o in image.owners if o.stratum == "H"})

        def add(role: str, suffix: Sequence[int], bindings: Sequence[Human13EncodedRowBinding], label: str) -> LogicalPanelSegment:
            ids = prompt + tuple(int(x) for x in suffix)
            segment = _segment(skeleton, image_id, role, f"human13:{image_id}:{label}", ids, bindings)
            return segment

        clean = tuple(image.source.prefix.clean_token_ids)
        all_h = tuple(token for row in h_rows for token in row.token_ids)
        all_h_bindings = _bindings("h", h_rows, prompt_count + len(clean), unit_by_row={row.row_id: row.owner_id for row in h_rows})
        segments.append(add("a1_full_h", clean + all_h, all_h_bindings, "a1"))
        segments.append(add("a8_full_h", clean + all_h, all_h_bindings, "a8-prime"))
        for row in h_rows:
            segments.append(add("h1_independent", clean + tuple(row.token_ids), _bindings("h", (row,), prompt_count + len(clean), unit_by_row={row.row_id: row.owner_id}), f"h1:{row.row_id}"))

        # A6 uses the exact donor trajectory prefix before the selected row.
        for row in h_rows:
            donor = next((t for t in image.trajectories if t.trajectory_id == row.trajectory_id), None)
            if donor is None:
                raise ValueError(f"missing A6 donor trajectory {row.trajectory_id}")
            source_row = next((r for r in donor.rows if r.row_id == row.row_id), None)
            donor_prefix = tuple(donor.raw_token_ids[: source_row.token_start]) if source_row is not None else tuple(donor.prefix.clean_token_ids)
            segments.append(add("a6_donor_h1", donor_prefix + tuple(row.token_ids), _bindings("h", (row,), prompt_count + len(donor_prefix), unit_by_row={row.row_id: row.owner_id}), f"a6:{row.row_id}"))

        # Source replay keeps the clean source prefix and only matched rows.
        source_rows = tuple(r for r in image.source.rows if r.row_id in set(image.replay_row_ids))
        owner_by_row = {row_id: owner.owner_id for owner in image.owners for row_id in getattr(owner, "source_row_ids", ())}
        segments.append(add("source_replay", clean, _replay_bindings(image.source, source_rows, owner_by_row), "source-replay"))

        # Full-GT body-only rows are supplied by the processor-only skeleton.
        owner_tokens = getattr(skeleton, "owner_row_tokens", {})
        gt_suffix = tuple(token for owner in image.owners for token in owner_tokens.get(owner.owner_id, ()))
        gt_bindings = _full_gt_bindings(image.owners, owner_tokens, prompt_count)
        segments.append(add("full_gt", gt_suffix, gt_bindings, "full-gt"))

        # Duplicate events retain the exact original decision prefix and target.
        for event in image.duplicate_events:
            dup_ids = tuple(event.decision_prefix_token_ids) + (int(event.target_token_id),)
            segments.append(add("duplicate_event", dup_ids, (_duplicate_binding(event, prompt_count),), event.event_id))

        # A4 candidates remain separate segments, with an atomic aggregate gate.
        total = 0
        candidate_rows = dict(selected)
        for trajectory in image.trajectories:
            for row in trajectory.rows:
                if row.row_id in candidate_rows:
                    continue
                candidate_rows[row.row_id] = SimpleNamespace(
                    row_id=row.row_id,
                    token_ids=tuple(trajectory.raw_token_ids[row.token_start:row.token_end]),
                )
        for row_id in image.candidate_row_ids:
            row = candidate_rows.get(row_id)
            if row is None:
                continue
            owner_id = next((owner.owner_id for owner in image.owners if row_id in getattr(owner, "sampled_row_ids", ())), row_id)
            candidate = add("a4_union", clean + tuple(row.token_ids), _bindings("h", (row,), prompt_count + len(clean), unit_by_row={row.row_id: owner_id}), f"a4:{row.row_id}")
            a4_segments.append(candidate)
            segments.append(candidate)
            total += candidate.encoded_length
        a4_totals.append((image_id, total))

    result = MaterializedSegments(tuple(segments), tuple(a4_segments), tuple(a4_totals))
    result.preflight(global_max_length)
    return result


def _bindings(family: str, rows: Sequence[Any], offset: int, *, unit_by_row: Mapping[str, str] | None = None) -> tuple[Human13EncodedRowBinding, ...]:
    out = []
    cursor = offset
    for row in rows:
        length = len(row.token_ids) if hasattr(row, "token_ids") else int(row.token_end - row.token_start)
        if length <= 0:
            continue
        unit_id = (unit_by_row or {}).get(row.row_id, row.row_id)
        mask = tuple(row.target_token_mask) if hasattr(row, "target_token_mask") else (True,) * length
        out.append(Human13EncodedRowBinding(family, unit_id, row.row_id, cursor, cursor + length, mask))
        cursor += length
    return tuple(out)


def _replay_bindings(source: Any, rows: Sequence[Any], owner_by_row: Mapping[str, str]) -> tuple[Human13EncodedRowBinding, ...]:
    removed = {i for row in source.rows if row.row_id in set(source.duplicate_row_ids) for i in range(row.token_start, row.token_end)}
    raw_to_clean = {}
    clean_index = 0
    for raw_index in range(len(source.raw_token_ids[: source.terminal_token_index or len(source.raw_token_ids)])):
        if raw_index not in removed:
            raw_to_clean[raw_index] = clean_index
            clean_index += 1
    out = []
    replay_mask = tuple(source.replay_token_mask)
    for row in rows:
        indexes = [raw_to_clean[i] for i in range(row.token_start, row.token_end) if i in raw_to_clean]
        if not indexes:
            continue
        start, end = min(indexes), max(indexes) + 1
        # Map the raw mask to the clean span, retaining malformed/unmatched positions masked.
        clean_mask_values = [False] * (end - start)
        for raw in range(row.token_start, row.token_end):
            clean = raw_to_clean.get(raw)
            if clean is not None and raw < len(replay_mask):
                clean_mask_values[clean - start] = bool(replay_mask[raw])
        clean_mask = tuple(clean_mask_values)
        if not any(clean_mask):
            continue
        out.append(Human13EncodedRowBinding("replay", owner_by_row.get(row.row_id, row.row_id), row.row_id, start, end, clean_mask))
    return tuple(out)


def _duplicate_binding(event: Any, prompt_count: int) -> Human13EncodedRowBinding:
    length = len(event.decision_prefix_token_ids) + 1
    start = prompt_count
    return Human13EncodedRowBinding("duplicate", event.event_id, event.duplicate_row_id, start, start + length, (False,) * (length - 1) + (True,))


def _full_gt_bindings(owners: Sequence[Any], owner_tokens: Mapping[str, Sequence[int]], prompt_count: int) -> tuple[Human13EncodedRowBinding, ...]:
    cursor = prompt_count
    out = []
    for owner in owners:
        tokens = tuple(owner_tokens.get(owner.owner_id, ()))
        if not tokens:
            raise ValueError(f"full_gt owner {owner.owner_id} has no encoded body row")
        out.append(Human13EncodedRowBinding("full_gt", owner.owner_id, owner.owner_id, cursor, cursor + len(tokens), (True,) * len(tokens)))
        cursor += len(tokens)
    return tuple(out)


def _segment(skeleton: Any, image_id: int, role: str, segment_id: str, input_ids: tuple[int, ...], bindings: Sequence[Human13EncodedRowBinding]) -> LogicalPanelSegment:
    encoded = _clone(skeleton, segment_id, input_ids, tuple(bindings))
    return LogicalPanelSegment(segment_id, image_id, role, encoded)


def _clone(skeleton: Any, segment_id: str, input_ids: tuple[int, ...], bindings: tuple[Human13EncodedRowBinding, ...]) -> Any:
    if is_dataclass(skeleton):
        encoded = replace(skeleton, example_id=segment_id, input_ids=input_ids)
        object.__setattr__(encoded, "human13_row_bindings", bindings)
        return encoded
    values = dict(vars(skeleton))
    values.update(example_id=segment_id, input_ids=input_ids, human13_row_bindings=bindings)
    return SimpleNamespace(**values)
