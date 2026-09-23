"""CPU readback of four frozen natural recurrence trajectories."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from probes.training_set_completion.recurrence_census.prepare import _rows_from_tokens
from src.artifacts.source_provenance import preserve_source


OUT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-recurrence-transition-readback/dynamics")
ROLES = ("x1", "y1", "x2", "y2")


def verified(binding):
    path = Path(binding["path"]).resolve(strict=True)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != binding["sha256"] or path.stat().st_size != binding["size_bytes"]:
        raise ValueError(f"source binding drift: {path}")
    return path


def aligned(raw, trace, batch, image_id):
    sample = raw["rows"][batch]
    if int(sample["image_id"]) != image_id:
        raise ValueError("raw batch/image mismatch")
    tokens = sample["token_ids"]
    steps = trace["steps"]
    if len(tokens) != len(steps):
        raise ValueError("raw/trace token count mismatch")
    fields = ("chosen", "raw_winners", "raw_runnerups", "raw_top2", "chosen_raw_logits", "logsumexp")
    for j, (token, step) in enumerate(zip(tokens, steps, strict=True)):
        if step["offset"] != j or any(len(step[key]) <= batch for key in fields) or step["chosen"][batch] != token:
            raise ValueError(f"raw/trace alignment mismatch at step {j}")
    return sample, steps


def repeated_margin(step, batch, repeated_token):
    winner, runner = step["raw_winners"][batch], step["raw_runnerups"][batch]
    high, second = step["raw_top2"][batch]
    if repeated_token == winner:
        return "exact", high - second
    if repeated_token == runner:
        return "exact", second - high
    return "upper_bound", second - high


def role_at(row, token_position):
    offset = token_position - (row["end"] - 5)
    return ROLES[offset] if 0 <= offset < 4 else "other"


def first_fork(reference, successor, tokens):
    a, b = tokens[reference["start"]:reference["end"]], tokens[successor["start"]:successor["end"]]
    for offset, (old, new) in enumerate(zip(a, b)):
        if old != new:
            return {"offset": offset, "role": role_at(successor, successor["start"] + offset), "old_token": old, "new_token": new}
    if len(a) != len(b):
        return {"offset": min(len(a), len(b)), "role": "other", "old_token": None, "new_token": None}
    return None


def windows(points):
    """Fixed repeated winner and fixed runner-up, within one role and episode."""
    out, current = [], []
    for p in points:
        key = (p["repeated_token"], p["winner"], p["runnerup"])
        if p["winner"] != p["repeated_token"] or (current and (p["row_index"] != current[-1]["row_index"] + 1 or key != (current[-1]["repeated_token"], current[-1]["winner"], current[-1]["runnerup"]))):
            if current:
                out.append(current)
            current = []
        if p["winner"] == p["repeated_token"]:
            current.append(p)
    if current:
        out.append(current)
    return out


def slope_turns(values, deadband=1e-4):
    signs = [1 if d > deadband else -1 if d < -deadband else 0 for d in (b - a for a, b in zip(values, values[1:]))]
    nonzero = [x for x in signs if x]
    return sum(a != b for a, b in zip(nonzero, nonzero[1:])), signs


def four_point_witness(window):
    best = None
    for points in zip(window, window[1:], window[2:], window[3:]):
        values = [point["margin_vs_winner_or_runnerup"] for point in points]
        differences = [b - a for a, b in zip(values, values[1:])]
        if differences[0] * differences[1] < 0 and differences[1] * differences[2] < 0:
            strength = min(map(abs, differences))
            if best is None or strength > best["min_abs_step"]:
                best = {"rows": [point["row_index"] for point in points], "margins": values, "min_abs_step": strength}
    return best


def analyze(item):
    for name in ("image", "raw", "trace", "runtime_receipt"):
        verified(item[name])
    raw = json.loads(Path(item["raw"]["path"]).read_text())
    trace = json.loads(Path(item["trace"]["path"]).read_text())
    sample, steps = aligned(raw, trace, item["batch_index"], item["image_id"])
    if item["source"] == "new":
        if sample.get("source_row") != item["source_row"] or sample.get("split") != item["image_key"].split(":")[0]:
            raise ValueError("source row/split mismatch")
    elif sample.get("row_id") != item["source_row"]:
        raise ValueError("mature source row mismatch")
    tokens, batch = sample["token_ids"], item["batch_index"]
    rows = _rows_from_tokens(tokens)
    runs = []
    for row in rows:
        sequence = tokens[row["start"]:row["end"]]
        if runs and sequence == runs[-1]["tokens"] and row["start"] == rows[runs[-1]["last_row"]]["end"]:
            runs[-1]["last_row"] = row["row_index"]
            runs[-1]["length"] += 1
        else:
            runs.append({"start_row": row["row_index"], "last_row": row["row_index"], "length": 1, "tokens": sequence})
    longest = max(runs, key=lambda r: r["length"])
    if {"start_row": longest["start_row"], "length": longest["length"]} != item["longest_exact_run"]:
        raise ValueError("selection/census exact run mismatch")
    exact_runs, points, fixed = [], [], []
    for run in runs:
        if run["length"] < 2:
            continue
        start, end = run["start_row"], run["last_row"]
        next_row = rows[end + 1] if end + 1 < len(rows) else None
        if next_row and next_row["start"] == rows[end]["end"]:
            termination = "different_complete_row"
        elif next_row:
            termination = "incomplete_gap"
        else:
            termination = "eos" if sample["stop"] == "im_end" else "cap" if sample["stop"] == "length" else sample["stop"]
        episode = {"start_row": start, "last_row": end, "length": run["length"], "valid_geometry": all(rows[j]["valid"] for j in range(start, end + 1)), "termination": termination,
                   "tail_tokens_after_run": tokens[rows[end]["end"]:next_row["start"]] if next_row else tokens[rows[end]["end"]:],
                   "next_row": None if next_row is None else {"row_index": next_row["row_index"], "tokens": tokens[next_row["start"]:next_row["end"]], "values": next_row["values"], "first_fork": first_fork(rows[end], next_row, tokens)}}
        if next_row and episode["next_row"]["first_fork"]:
            fork = episode["next_row"]["first_fork"]
            pos = next_row["start"] + fork["offset"]
            kind, margin = repeated_margin(steps[pos], batch, fork["old_token"])
            fork.update({"token_position": pos, "old_token_margin_kind": kind, "old_token_margin_vs_winner_or_runnerup": margin,
                         "winner": steps[pos]["raw_winners"][batch], "runnerup": steps[pos]["raw_runnerups"][batch]})
        exact_runs.append(episode)
        for role_offset, role in enumerate(ROLES):
            role_points = []
            for j in range(start, end + 1):
                row = rows[j]
                pos = row["end"] - 5 + role_offset
                step = steps[pos]
                repeated = rows[start]["values"][role_offset] + 151670
                kind, margin = repeated_margin(step, batch, repeated)
                p = {"image_key": item["image_key"], "run_start": start, "run_length": run["length"], "row_index": j, "role": role, "token_position": pos,
                     "repeated_token": repeated, "chosen_token": tokens[pos], "winner": step["raw_winners"][batch], "runnerup": step["raw_runnerups"][batch],
                     "winner_logit": step["raw_top2"][batch][0], "runnerup_logit": step["raw_top2"][batch][1], "chosen_logit": step["chosen_raw_logits"][batch], "logsumexp": step["logsumexp"][batch],
                     "margin_kind": kind, "margin_vs_winner_or_runnerup": margin}
                role_points.append(p)
                points.append(p)
            for window in windows(role_points):
                turns, signs = slope_turns([p["margin_vs_winner_or_runnerup"] for p in window])
                fixed.append({"run_start": start, "role": role, "start_row": window[0]["row_index"], "last_row": window[-1]["row_index"], "length": len(window),
                              "winner": window[0]["winner"], "runnerup": window[0]["runnerup"], "slope_turns_deadband_1e_4": turns,
                              "slope_turns_deadband_1e_3": slope_turns([p["margin_vs_winner_or_runnerup"] for p in window], 1e-3)[0],
                              "slope_turns_deadband_1e_2": slope_turns([p["margin_vs_winner_or_runnerup"] for p in window], 1e-2)[0],
                              "four_point_witness": four_point_witness(window), "slope_signs": signs,
                              "margins": [p["margin_vs_winner_or_runnerup"] for p in window]})
    row_records = []
    for row in rows:
        row_records.append({**row, "tokens": tokens[row["start"]:row["end"]], "logprob": sum(steps[j]["chosen_raw_logits"][batch] - steps[j]["logsumexp"][batch] for j in range(row["start"], row["end"]))})
    grouped = defaultdict(list)
    for point in points:
        grouped[(point["run_start"], point["role"])].append(point)
    role_trends = []
    for (run_start, role), series in grouped.items():
        margins = [point["margin_vs_winner_or_runnerup"] for point in series if point["margin_kind"] == "exact"]
        role_trends.append({"run_start": run_start, "role": role, "rows": len(series),
                            "exact_margin_count": len(margins), "bound_count": len(series) - len(margins),
                            "first_margin": margins[0] if margins else None, "last_margin": margins[-1] if margins else None,
                            "min_margin": min(margins) if margins else None, "max_margin": max(margins) if margins else None,
                            "winner_switches": sum(a["winner"] != b["winner"] for a, b in zip(series, series[1:])),
                            "rival_switches": sum(a["runnerup"] != b["runnerup"] for a, b in zip(series, series[1:]))})
    return {"item": item, "stop": sample["stop"], "token_count": len(tokens), "complete_rows": row_records, "incomplete_tail_tokens": tokens[rows[-1]["end"]:] if rows else tokens,
            "exact_runs": exact_runs, "role_trends": role_trends, "fixed_pair_windows": fixed, "points": points}


def selfcheck():
    row = [151646, 42, 151647, 151648, 151680, 151690, 151700, 151710, 151649]
    assert _rows_from_tokens(row)[0]["end"] == len(row)
    sample = {"rows": [{"image_id": 1, "token_ids": [9]}]}
    trace = {"steps": [{"offset": 0, "chosen": [8], "raw_winners": [8], "raw_runnerups": [7], "raw_top2": [[2, 1]], "chosen_raw_logits": [2], "logsumexp": [2]}]}
    try:
        aligned(sample, trace, 0, 1)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupt token alignment accepted")
    trace["steps"][0]["chosen"][0] = 9
    try:
        aligned(sample, trace, 0, 2)
    except ValueError:
        pass
    else:
        raise AssertionError("wrong batch image accepted")
    step = {"raw_winners": [9], "raw_runnerups": [8], "raw_top2": [[2.0, 1.5]]}
    assert repeated_margin(step, 0, 8) == ("exact", -0.5)
    assert repeated_margin(step, 0, 7) == ("upper_bound", -0.5)
    assert repeated_margin(step, 0, 9) == ("exact", 0.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    selfcheck()
    if args.selfcheck:
        print("selfcheck ok")
        return
    manifest = json.loads((OUT / "selection.json").read_text())
    for binding in manifest["inputs"].values():
        verified(binding)
    result = [analyze(item) for item in manifest["selected"]]
    with (OUT / "episodes.json").open("w") as file:
        json.dump({"schema": "recurrence_transition_readback.episodes.v1", "selection": str(OUT / "selection.json"), "images": [{k: v for k, v in image.items() if k != "points"} for image in result]}, file, indent=2)
        file.write("\n")
    points = [p for image in result for p in image["points"]]
    with (OUT / "role_margins.tsv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(points[0]), delimiter="\t")
        writer.writeheader(); writer.writerows(points)
    digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
    capture = preserve_source(Path(__file__), run_root=OUT, relative_name=f"recurrence_transition_readback-{digest}.py")
    print(json.dumps({"images": len(result), "exact_runs": sum(len(r["exact_runs"]) for r in result), "role_points": len(points), "source_capture": str(capture), "output": str(OUT)}))


if __name__ == "__main__":
    main()
