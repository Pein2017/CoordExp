"""CPU reducer for the bounded conditional numerical-repeat probe.

The reducer intentionally consumes raw sampled token IDs.  It does not infer
physical identity and it does not turn a non-event into a coordinate by
retrying or grammar projection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA = "recurrence_conditional_mass.draws.v1"
SUMMARY_SCHEMA = "recurrence_conditional_mass.reduction.v1"


def _as_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer") from exc
    if result != value and not isinstance(value, str):
        raise ValueError(f"{field} must be an integer")
    return result


def _bins_from_token_ids(
    token_ids: Sequence[int],
    *,
    coordinate_token_ids: Sequence[int] | None,
    coordinate_token_id_start: int | None,
    coordinate_token_id_end_exclusive: int | None,
) -> tuple[int, ...] | None:
    """Decode exactly four coordinate IDs, preserving non-contiguous registries."""

    if coordinate_token_ids is not None:
        lookup = {int(token): index for index, token in enumerate(coordinate_token_ids)}
        values = tuple(lookup.get(int(token), -1) for token in token_ids)
        if len(values) == 4 and all(value >= 0 for value in values):
            return values
        return None
    if coordinate_token_id_start is None or coordinate_token_id_end_exclusive is None:
        raise ValueError("state lacks an explicit coordinate-token registry")
    lo = int(coordinate_token_id_start)
    hi = int(coordinate_token_id_end_exclusive)
    if hi <= lo:
        raise ValueError("coordinate-token interval is empty")
    if len(token_ids) != 4 or any(not lo <= int(token) < hi for token in token_ids):
        return None
    return tuple(int(token) - lo for token in token_ids)


def _normalise_union(value: Any, *, field: str) -> frozenset[tuple[int, int, int, int]]:
    if value is None:
        return frozenset()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must be a sequence of four-bin boxes")
    result: set[tuple[int, int, int, int]] = set()
    for index, item in enumerate(value):
        if not isinstance(item, Sequence) or isinstance(item, (str, bytes)) or len(item) != 4:
            raise ValueError(f"{field}[{index}] must contain four bins")
        box = tuple(_as_int(v, field=f"{field}[{index}]") for v in item)
        if any(v < 0 or v > 999 for v in box):
            raise ValueError(f"{field}[{index}] contains a bin outside [0,999]")
        result.add(box)  # overlapping historical members count once
    return frozenset(result)


def _legal(box: Sequence[int]) -> bool:
    if len(box) != 4 or any(not isinstance(v, int) or not 0 <= v <= 999 for v in box):
        return False
    x1, y1, x2, y2 = box
    return x1 < x2 and y1 < y2


def _wilson(successes: int, trials: int, z: float = 1.959963984540054) -> list[float]:
    if trials <= 0:
        return [float("nan"), float("nan")]
    p = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (p + z * z / (2.0 * trials)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / denominator
    return [max(0.0, centre - radius), min(1.0, centre + radius)]


def _event_for_draw(
    draw: Mapping[str, Any],
    *,
    box_end_token_id: int,
    im_end_token_id: int | None,
    coordinate_token_ids: Sequence[int] | None,
    coordinate_token_id_start: int | None,
    coordinate_token_id_end_exclusive: int | None,
    repeat_union: frozenset[tuple[int, int, int, int]],
    literal_repeat_union: frozenset[tuple[int, int, int, int]],
    horizon: int,
    repeat_tolerance_bins: int,
) -> dict[str, Any]:
    raw = draw.get("token_ids", draw.get("sampled_token_ids"))
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("draw.token_ids must be a sequence")
    tokens = tuple(_as_int(value, field="draw.token_ids") for value in raw)
    if len(tokens) > horizon:
        raise ValueError("draw emitted more tokens than the frozen horizon")

    stop_reason = str(draw.get("stop_reason", "length"))
    expected_box_tokens = 4
    # The stored token sequence is always the emitted prefix, including EOS if
    # native generation reached it.  An EOS before the row is complete is a
    # format outcome; no token is silently substituted.
    if im_end_token_id is not None and im_end_token_id in tokens:
        eos_index = tokens.index(im_end_token_id)
        if eos_index < len(tokens) - 1:
            raise ValueError("draw contains content after its first EOS")
        if eos_index < expected_box_tokens:
            return {
                "event": False,
                "invalid_geometry_near_repeat": False,
                "exact_invalid_recurrence": False,
                "geometry_valid": False,
                "outcome": "early_eos",
                "token_ids": list(tokens),
                "stop_reason": stop_reason,
            }

    if len(tokens) < expected_box_tokens + 1:
        reason = "early_eos" if im_end_token_id is not None and im_end_token_id in tokens else "short_or_length"
        return {
            "event": False,
            "invalid_geometry_near_repeat": False,
            "exact_invalid_recurrence": False,
            "geometry_valid": False,
            "outcome": reason,
            "token_ids": list(tokens),
            "stop_reason": stop_reason,
        }

    coord_tokens = tokens[:4]
    terminator = tokens[4]
    boxes = _bins_from_token_ids(
        coord_tokens,
        coordinate_token_ids=coordinate_token_ids,
        coordinate_token_id_start=coordinate_token_id_start,
        coordinate_token_id_end_exclusive=coordinate_token_id_end_exclusive,
    )
    if boxes is None or terminator != box_end_token_id:
        return {
            "event": False,
            "invalid_geometry_near_repeat": False,
            "exact_invalid_recurrence": False,
            "geometry_valid": False,
            "outcome": "grammar_escape",
            "token_ids": list(tokens),
            "stop_reason": stop_reason,
        }
    legal = _legal(boxes)
    repeat_matches = [
        index
        for index, historical in enumerate(sorted(repeat_union))
        if max(abs(a - b) for a, b in zip(boxes, historical)) <= repeat_tolerance_bins
    ]
    literal_matches = [
        index
        for index, historical in enumerate(sorted(literal_repeat_union))
        if max(abs(a - b) for a, b in zip(boxes, historical)) <= repeat_tolerance_bins
    ]
    invalid_near_repeat = bool(literal_matches)
    exact_invalid = (not legal) and tuple(boxes) in literal_repeat_union
    repeated = legal and bool(repeat_matches)
    if repeated:
        outcome = "legal_repeat"
    elif invalid_near_repeat and not legal:
        outcome = "invalid_geometry_near_repeat"
    elif not legal:
        outcome = "invalid_extent"
    else:
        outcome = "legal_nonrepeat"
    return {
        "event": repeated,
        "invalid_geometry_near_repeat": bool(invalid_near_repeat and not legal),
        "exact_invalid_recurrence": bool(exact_invalid),
        "geometry_valid": legal,
        "outcome": outcome,
        "box_bins": list(boxes),
        "repeat_tolerance_bins": repeat_tolerance_bins,
        "repeat_union_matches": repeat_matches,
        "literal_repeat_union_matches": literal_matches,
        "token_ids": list(tokens),
        "stop_reason": stop_reason,
    }


def reduce_state(state: Mapping[str, Any]) -> dict[str, Any]:
    if str(state.get("schema")) not in {SCHEMA, "recurrence_conditional_mass.state.v1"}:
        raise ValueError("unsupported state schema")
    state_id = str(state.get("state_id"))
    source_example_id = state.get("source_example_id")
    if (
        not isinstance(source_example_id, str)
        or not source_example_id
        or source_example_id.strip().isdigit()
    ):
        raise ValueError(f"{state_id}: split-aware source_example_id is required")
    draws = state.get("draws")
    if not isinstance(draws, Sequence) or isinstance(draws, (str, bytes)):
        raise ValueError(f"{state_id}: draws must be a sequence")
    horizon = _as_int(state.get("horizon"), field=f"{state_id}.horizon")
    if horizon <= 0:
        raise ValueError(f"{state_id}: horizon must be positive")
    if len(draws) != 256:
        raise ValueError(f"{state_id}: expected exactly 256 draws, got {len(draws)}")
    registry = state.get("coordinate_registry", {})
    if not isinstance(registry, Mapping):
        raise ValueError(f"{state_id}: coordinate_registry must be an object")
    explicit = registry.get("coordinate_token_ids")
    if explicit is not None:
        explicit = tuple(_as_int(v, field=f"{state_id}.coordinate_token_ids") for v in explicit)
        if len(explicit) != 1000 or len(set(explicit)) != 1000:
            raise ValueError(f"{state_id}: coordinate registry must contain 1000 unique IDs")
    repeat_union = _normalise_union(state.get("repeat_union_bins"), field=f"{state_id}.repeat_union_bins")
    literal_union = _normalise_union(
        state.get("literal_repeat_union_bins", state.get("repeat_union_bins")),
        field=f"{state_id}.literal_repeat_union_bins",
    )
    tolerance = _as_int(state.get("repeat_tolerance_bins", 8), field=f"{state_id}.repeat_tolerance_bins")
    if tolerance < 0:
        raise ValueError(f"{state_id}: repeat_tolerance_bins must be nonnegative")
    box_end = _as_int(state.get("box_end_token_id"), field=f"{state_id}.box_end_token_id")
    im_end_raw = state.get("im_end_token_id")
    im_end = None if im_end_raw is None else _as_int(im_end_raw, field=f"{state_id}.im_end_token_id")
    outcomes: list[dict[str, Any]] = []
    for draw in draws:
        if not isinstance(draw, Mapping):
            raise ValueError(f"{state_id}: draw must be an object")
        outcomes.append(
            _event_for_draw(
                draw,
                box_end_token_id=box_end,
                im_end_token_id=im_end,
                coordinate_token_ids=explicit,
                coordinate_token_id_start=(
                    None if registry.get("coordinate_token_id_start") is None else int(registry["coordinate_token_id_start"])
                ),
                coordinate_token_id_end_exclusive=(
                    None if registry.get("coordinate_token_id_end_exclusive") is None else int(registry["coordinate_token_id_end_exclusive"])
                ),
                repeat_union=repeat_union,
                literal_repeat_union=literal_union,
                horizon=horizon,
                repeat_tolerance_bins=tolerance,
            )
        )
    counts: dict[str, int] = defaultdict(int)
    for outcome in outcomes:
        counts[str(outcome["outcome"])] += 1
    successes = counts["legal_repeat"]
    invalid_near = counts["invalid_geometry_near_repeat"]
    return {
        "state_id": state_id,
        "image_id": state.get("image_id"),
        "source_example_id": source_example_id,
        "model": state.get("model"),
        "source_policy": state.get("source_policy"),
        "state_type": state.get("state_type"),
        "description": state.get("description"),
        "draw_count": len(outcomes),
        "success_count": successes,
        "q_hat": successes / len(outcomes),
        "q_interval_95": _wilson(successes, len(outcomes)),
        "invalid_geometry_near_repeat_count": invalid_near,
        "exact_invalid_recurrence_count": sum(
            int(bool(outcome["exact_invalid_recurrence"])) for outcome in outcomes
        ),
        "outcome_counts": dict(sorted(counts.items())),
        "repeat_union_size": len(repeat_union),
        "literal_repeat_union_size": len(literal_union),
        "repeat_tolerance_bins": tolerance,
        "horizon": horizon,
        "draws": outcomes,
    }


def reduce_file(path: Path) -> dict[str, Any]:
    states: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise ValueError(f"line {line_number}: state must be an object")
        states.append(reduce_state(value))
    if not states:
        raise ValueError("draw file is empty")
    # Image summaries must use the bound split-aware identity.  Numeric COCO
    # IDs can recur across splits and are not an adequate denominator key.
    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for state in states:
        by_image[str(state["source_example_id"])].append(state)
    image_rows = []
    for source_example_id, image_states in sorted(by_image.items()):
        image_rows.append(
            {
                "source_example_id": source_example_id,
                "image_id": image_states[0].get("image_id"),
                "state_count": len(image_states),
                "q_hat_mean_across_states": sum(float(s["q_hat"]) for s in image_states) / len(image_states),
                "success_count": sum(int(s["success_count"]) for s in image_states),
                "draw_count": sum(int(s["draw_count"]) for s in image_states),
            }
        )
    return {
        "schema": SUMMARY_SCHEMA,
        "source": str(path.resolve()),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "state_count": len(states),
        "image_count": len(image_rows),
        "draw_count": sum(int(s["draw_count"]) for s in states),
        "success_count": sum(int(s["success_count"]) for s in states),
        "state_results": states,
        "image_summary": image_rows,
    }


def self_test() -> None:
    # Full-vocabulary sampling can escape coordinates; a constrained sampler
    # is a separate control and must not silently define the primary event.
    state = {
        "schema": SCHEMA,
        "state_id": "cpu-known-event",
        "image_id": 1,
        "source_example_id": "toy_split_000000000001",
        "model": "toy",
        "horizon": 5,
        "box_end_token_id": 12005,
        "im_end_token_id": 9999,
        "coordinate_registry": {"coordinate_token_id_start": 9000, "coordinate_token_id_end_exclusive": 10000},
        "repeat_union_bins": [[1, 2, 3, 4]],
        "literal_repeat_union_bins": [[1, 2, 3, 4], [8, 8, 7, 9]],
        "draws": [
            {"token_ids": [9001, 9002, 9003, 9004, 12005], "stop_reason": "length"}
            for _ in range(256)
        ],
    }
    result = reduce_state(state)
    assert result["success_count"] == 256
    assert result["outcome_counts"]["legal_repeat"] == 256
    assert result["outcome_counts"].get("invalid_geometry_near_repeat", 0) == 0
    state["draws"][0] = {"token_ids": [9999], "stop_reason": "im_end"}
    state["draws"][1] = {"token_ids": [42, 9002, 9003, 9004, 12005], "stop_reason": "length"}
    state["draws"][2] = {"token_ids": [9008, 9008, 9007, 9009, 12005], "stop_reason": "length"}
    result = reduce_state(state)
    assert result["success_count"] == 253
    assert result["outcome_counts"]["early_eos"] == 1
    assert result["outcome_counts"]["grammar_escape"] == 1
    assert result["invalid_geometry_near_repeat_count"] == 1
    assert result["exact_invalid_recurrence_count"] == 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        print("self_test: passed")
        return 0
    if args.draws is None or args.output is None:
        parser.error("--draws and --output are required unless --self-test is used")
    result = reduce_file(args.draws.expanduser().resolve(strict=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: result[k] for k in ("state_count", "image_count", "draw_count", "success_count")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
