"""Small shared CPU semantics for the recurrence-spatial producer and reducer.

The helper keeps serialization, coordinate-role mapping, and the frozen
pairwise recurrence predicate in one place.  It does not load a model or own
any scheduling/runtime state.
"""

from __future__ import annotations

from typing import Any

from probes.training_set_completion.numerical_feedback.select import same as _accepted_same


EOS = 151645
OBJ_START = 151646
OBJ_END = 151647
BOX_START = 151648
BOX_END = 151649
COORD_BASE = 151670
COORD_LIMIT = COORD_BASE + 1000
NEAR_EPS = 8


def inverse_bin(
    value: int,
    *,
    coordinate_index: int,
    source_width: int,
    source_height: int,
    canvas_width: int,
    canvas_height: int,
    tx: int,
    ty: int = 0,
) -> int:
    """Inverse-map one canvas bin using its x/y coordinate role."""

    is_x = coordinate_index in (0, 2)
    source_size = source_width if is_x else source_height
    canvas_size = canvas_width if is_x else canvas_height
    offset = tx if is_x else ty
    canvas_pixel = value * (canvas_size - 1) / 999.0
    source_pixel = canvas_pixel - offset
    return int(round(source_pixel * 999.0 / (source_size - 1)))


def runs(rows: list[dict[str, Any]], *, near: bool) -> list[dict[str, Any]]:
    """Return witnesses using the accepted consecutive-triple primitive.

    ``numerical_feedback.select.same`` is the frozen description/coordinate
    comparison.  The selector applies it to the three pairs in each adjacent
    triple; this adapter keeps the same primitive while retaining malformed,
    invalid, and out-of-source rows for reporting.  Invalid geometry is not a
    reason to remove a complete serialized row from recurrence admission.
    """
    eps = NEAR_EPS if near else 0
    complete_indices = [index for index, row in enumerate(rows) if row.get("complete", False)]
    accepted_rows = [
        {**rows[index], "values": list(rows[index]["coord_bins_source"])}
        for index in complete_indices
    ]
    result: list[dict[str, Any]] = []
    for start in range(len(accepted_rows) - 2):
        triple = accepted_rows[start : start + 3]
        if not all(
            _accepted_same(triple[left], triple[right], eps)
            for left, right in ((0, 1), (0, 2), (1, 2))
        ):
            continue
        first = rows[complete_indices[start]]
        result.append(
            {
                "start_row": int(first["row_index"]),
                "length": 3,
                "row_indices": [
                    int(rows[complete_indices[start + offset]]["row_index"])
                    for offset in range(3)
                ],
                "description": first.get("description"),
                "description_tokens": list(first.get("description_tokens", [])),
                "coord_bins_source": list(first["coord_bins_source"]),
                "kind": "near" if near else "exact",
            }
        )
    return result


def _complete_segment(segment: list[int]) -> tuple[list[int], list[int]] | None:
    """Return description and four coordinates for one complete serialized row."""

    try:
        description_end = segment.index(OBJ_END, 1)
    except ValueError:
        return None
    if OBJ_START in segment[1:description_end]:
        return None
    coordinate_start = description_end + 2
    coordinate_end = coordinate_start + 4
    if coordinate_end >= len(segment) or segment[description_end + 1] != BOX_START:
        return None
    coordinates = segment[coordinate_start:coordinate_end]
    if len(coordinates) != 4 or not all(COORD_BASE <= token < COORD_LIMIT for token in coordinates):
        return None
    if segment[coordinate_end] != BOX_END:
        return None
    description_tokens = segment[1:description_end]
    return description_tokens, [token - COORD_BASE for token in coordinates]


def complete_box_count(token_ids: list[int]) -> int:
    """Count serialized complete rows, ignoring stray terminator tokens."""

    starts = [index for index, token in enumerate(token_ids) if token == OBJ_START]
    count = 0
    for row_index, start in enumerate(starts):
        stop = starts[row_index + 1] if row_index + 1 < len(starts) else len(token_ids)
        count += _complete_segment(token_ids[start:stop]) is not None
    return count


def parse_rows(
    token_ids: list[int],
    tokenizer: Any,
    *,
    cell: dict[str, Any],
    geometry: dict[str, Any],
) -> dict[str, Any]:
    """Parse complete rows while retaining invalid and out-of-source boxes."""

    starts = [index for index, token in enumerate(token_ids) if token == OBJ_START]
    rows: list[dict[str, Any]] = []
    for row_index, start in enumerate(starts):
        stop = starts[row_index + 1] if row_index + 1 < len(starts) else len(token_ids)
        segment = token_ids[start:stop]
        decoded = _complete_segment(segment)
        if decoded is None:
            rows.append(
                {
                    "row_index": row_index,
                    "status": "malformed",
                    "complete": False,
                    "token_count": len(segment),
                    "raw_token_ids": segment,
                }
            )
            continue
        description_tokens, mapped = decoded
        description = tokenizer.decode(
            description_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        source_bins = [
            inverse_bin(
                value,
                coordinate_index=coordinate_index,
                source_width=int(geometry["source_width"]),
                source_height=int(geometry["source_height"]),
                canvas_width=int(geometry["canvas_width"]),
                canvas_height=int(geometry["canvas_height"]),
                tx=int(cell.get("visual_offset_px", 0)),
                ty=int(cell.get("visual_offset_y_px", 0)),
            )
            for coordinate_index, value in enumerate(mapped)
        ]
        valid = mapped[0] < mapped[2] and mapped[1] < mapped[3]
        source_in_bounds = all(0 <= value <= 999 for value in source_bins)
        source_ordered = source_bins[0] < source_bins[2] and source_bins[1] < source_bins[3]
        rows.append(
            {
                "row_index": row_index,
                "status": "valid" if valid else "invalid",
                "complete": True,
                "description": description,
                "description_tokens": description_tokens,
                "coord_bins_canvas": mapped,
                "coord_bins_source": source_bins,
                "source_in_bounds": source_in_bounds,
                "canvas_border": any(value in (0, 999) for value in mapped),
                "source_geometry_valid": valid and source_in_bounds and source_ordered,
                "token_count": len(segment),
                "raw_token_ids": segment,
            }
        )
    exact_runs = runs(rows, near=False)
    near_runs = runs(rows, near=True)
    complete_rows = sum(row.get("complete", False) for row in rows)
    return {
        "rows": rows,
        "complete_rows": complete_rows,
        "valid_rows": sum(row["status"] == "valid" for row in rows),
        "invalid_rows": sum(row["status"] == "invalid" for row in rows),
        "malformed_rows": sum(row["status"] == "malformed" for row in rows),
        "exact_runs": exact_runs,
        "near_runs": near_runs,
        "failure_predicate": bool(exact_runs or near_runs),
    }
