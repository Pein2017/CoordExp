#!/usr/bin/env python3
"""Small CPU consumer for the Stage C sampling reduction.

This file only projects the reduction.  It does not read tensors or run a
model, and it deliberately leaves geometric/parser descriptors overlapping.
"""
import argparse
import json
import math
from pathlib import Path


FROZEN_PER_TEMPERATURE = 8
FROZEN_TOTAL = 24
ACCEPTED_A = (151646, 22592, 151647, 151648, 152358, 152222, 152436, 152291, 151649)
ALTERNATE_A = (151646, 22592, 151647, 151648, 151670, 152451, 151766, 152573, 151649)


def _ids(value):
    """Return stable string IDs from an optional owner-ID collection."""
    if not value:
        return []
    return sorted({str(x) for x in value})


def _metric_ids(metric):
    matches = (metric or {}).get("matches") or {}
    for key in ("covered_owner_ids", "owner_ids", "known_owner_ids"):
        if key in matches:
            return _ids(matches[key])
    return []


def _burden(metric):
    return (metric or {}).get("burden") or {}


def _event_counts(metric):
    """Preserve parser events without turning them into exclusive classes."""
    b = _burden(metric)
    drops = (metric or {}).get("drops") or []
    geometry = sum(1 for d in drops if d.get("drop_reason") == "geometry_invalid")
    malformed_drops = sum(
        1 for d in drops
        if d.get("drop_reason") in {"malformed", "malformed_object_span", "semantic_malformed"}
    )
    semantic = int((metric or {}).get("semantic_malformed") or 0)
    return {
        "geometry_invalid": geometry or int(b.get("invalid") or 0),
        "semantic_malformed": semantic or malformed_drops or int(b.get("malformed") or 0),
        "drop_reasons": dict(b.get("drop_reasons") or {}),
        "complete_rows": int(b.get("complete_rows") or len((metric or {}).get("complete_rows") or [])),
        "stop": (metric or {}).get("stop"),
        "eos": int(b.get("eos") or 0),
        "cap": int(b.get("cap") or 0),
    }


def _ended(metric, token_count=None):
    stop = (metric or {}).get("stop")
    burden = _burden(metric)
    n = token_count if token_count is not None else (metric or {}).get("token_count")
    return {
        "stop": stop,
        "eos": bool(stop in {"eos", "im_end"} or burden.get("eos")),
        "cap": bool(stop in {"cap", "length"} or burden.get("cap") or n == 3084),
    }


def _rows(tokens, base_offset):
    tokens = list(tokens or [])
    out = []
    for i in range(0, len(tokens) - 8, 9):
        row = tuple(tokens[i:i + 9])
        out.append({"offset": base_offset + i, "token_ids": list(row)})
    return out


def _runs(rows, exclude_a=True):
    runs = []
    i = 0
    while i < len(rows):
        j = i + 1
        while j < len(rows) and rows[j]["token_ids"] == rows[i]["token_ids"]:
            j += 1
        if j - i >= 3 and (not exclude_a or tuple(rows[i]["token_ids"]) != ACCEPTED_A):
            runs.append({"start_offset": rows[i]["offset"], "length": j - i,
                         "token_ids": rows[i]["token_ids"]})
        i = j
    return runs


def _exact_post_release(cell):
    tokens = cell.get("post_release_token_ids")
    release = cell.get("release")
    if not isinstance(tokens, list) or not isinstance(release, int):
        return {"available": False, "accepted_a_returns": [],
                "alternate_exact_runs_ge3_excluding_a": [], "row_count": 0}
    rows = _rows(tokens, release)
    a_returns = [r for r in rows if tuple(r["token_ids"]) == ACCEPTED_A]
    return {
        "available": True,
        "row_count": len(rows),
        "accepted_a_returns": a_returns,
        "alternate_exact_runs_ge3_excluding_a": _runs(rows, exclude_a=True),
        "accepted_a_template": list(ACCEPTED_A),
        "alternate_a_template": list(ALTERNATE_A),
    }


def _trajectory(key, cell, exclusions):
    entry = cell.get("entry") or {}
    pulse = cell.get("intervention") or {}
    post = cell.get("after_release") or {}
    full = cell.get("full") or {}
    accounting = cell.get("per_trajectory_known_accounting") or {}
    supplied = set(_ids(accounting.get("supplied_owner_ids")))
    supplied.update(_ids(cell.get("crossing_supplied_owner_ids")))
    supplied.update(_ids((cell.get("known_accounting") or {}).get("excluded_union")))
    pulse_ids, post_ids, full_ids = map(_metric_ids, (pulse, post, full))
    exclusion = sorted(supplied)
    excl = set(exclusions)
    gains = _ids(accounting.get("gained_vs_native_future"))
    gains = sorted(set(gains) - excl)
    new_ids = sorted(set(_ids(accounting.get("autonomous_new_relative_prefix"))) - excl)
    exact = _exact_post_release(cell)
    return {
        "cell": key,
        "condition": entry.get("condition"),
        "image_id": entry.get("image_id"),
        "temperature": entry.get("temperature"),
        "seed": entry.get("seed"),
        "entry": {k: entry.get(k) for k in ("stage", "condition", "mode", "temperature", "seed")},
        "pulse": {"known_ids": pulse_ids, "token_count": pulse.get("token_count"),
                  "complete_rows": pulse.get("complete_rows") or [],
                  "events": _event_counts(pulse), "ended": _ended(pulse)},
        "post_release": {"known_ids": post_ids, "token_count": post.get("token_count"),
                          "complete_rows": post.get("complete_rows") or [],
                          "events": _event_counts(post), "ended": _ended(post)},
        "full": {"known_ids": full_ids, "token_count": full.get("token_count"),
                 "complete_rows": full.get("complete_rows") or [],
                 "events": _event_counts(full), "ended": _ended(full)},
        "symmetric_exclusion": exclusion,
        "known_accounting": accounting,
        "known_gain_candidates": {
            "gained_vs_native_future_after_exclusion": gains,
            "autonomous_new_relative_prefix_after_exclusion": new_ids,
            "has_candidate": bool(gains or new_ids),
        },
        "winner_changes": cell.get("winner_changes", 0),
        "first_changed_token": cell.get("first_changed_token"),
        "exact_post_release": exact,
        "repetition_descriptors": {
            "pulse": {"literal_rows": pulse.get("exact_runs") or [],
                      "strict_valid_rows": pulse.get("post_release_recurrence_against_all_history")},
            "post_release": {"literal_rows": post.get("exact_runs") or [],
                              "strict_valid_rows": cell.get("post_release_recurrence_against_all_history")},
            "full": {"literal_rows": full.get("exact_runs") or [],
                     "strict_valid_rows": full.get("post_release_recurrence_against_all_history")},
        },
    }


def _wilson(successes, trials, z=1.959963984540054):
    if trials <= 0:
        return {"successes": 0, "trials": 0, "rate": None, "wilson95": None}
    p = successes / trials
    den = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / den
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / den
    return {"successes": successes, "trials": trials, "rate": p,
            "wilson95": [max(0.0, centre - half), min(1.0, centre + half)]}


def summarize(doc):
    cells = doc.get("cells") or {}
    if isinstance(cells, list):
        cells = {str(c.get("id", i)): c for i, c in enumerate(cells)}
    selected = [(k, v) for k, v in cells.items()
                if str(k).startswith("C/") and (v.get("entry") or {}).get("stage", "C") == "C"]
    selected.sort(key=lambda kv: (float((kv[1].get("entry") or {}).get("temperature", 0)),
                                  int((kv[1].get("entry") or {}).get("seed", 0)), kv[0]))
    exclusion = set()
    for _, c in selected:
        a = c.get("per_trajectory_known_accounting") or {}
        exclusion.update(_ids(a.get("supplied_owner_ids")))
        exclusion.update(_ids(c.get("crossing_supplied_owner_ids")))
        exclusion.update(_ids((c.get("known_accounting") or {}).get("excluded_union")))
    trajectories = [_trajectory(k, c, exclusion) for k, c in selected]

    def flag(t, name):
        if name == "accepted_a_return": return bool(t["exact_post_release"]["accepted_a_returns"])
        if name == "alternate_exact_run_ge3_excluding_a": return bool(t["exact_post_release"]["alternate_exact_runs_ge3_excluding_a"])
        if name == "known_gain_candidate": return t["known_gain_candidates"]["has_candidate"]
        if name == "winner_change": return bool(t["winner_changes"])
        if name == "eos": return t["full"]["ended"]["eos"]
        if name == "cap": return t["full"]["ended"]["cap"]
        raise KeyError(name)

    rates = {}
    for temp in sorted({t["temperature"] for t in trajectories}):
        group = [t for t in trajectories if t["temperature"] == temp]
        rates[str(temp)] = {
            "observed_trajectories": len(group),
            "frozen_trajectories": FROZEN_PER_TEMPERATURE,
            "partial": len(group) < FROZEN_PER_TEMPERATURE,
            "descriptors": {name: _wilson(sum(flag(t, name) for t in group), len(group))
                            for name in ("accepted_a_return", "alternate_exact_run_ge3_excluding_a",
                                         "known_gain_candidate", "winner_change", "eos", "cap")},
        }
    return {
        "schema": "repetition_history_runtime.sampling_summary.v1",
        "status": "partial" if len(trajectories) < FROZEN_TOTAL else "complete",
        "selection": {"stage": "C", "selected_cells": len(trajectories),
                       "frozen_cells": FROZEN_TOTAL, "partial": len(trajectories) < FROZEN_TOTAL},
        "temperatures": rates,
        "symmetric_exclusion": sorted(exclusion),
        "templates": {"accepted_a": list(ACCEPTED_A), "alternate_a": list(ALTERNATE_A),
                      "comparison": "token-exact; alternate runs exclude accepted A tokens"},
        "claim_boundary": "Known-ID and parser descriptors only; no physical claim from IoU.",
        "trajectories": trajectories,
    }


def _selfcheck():
    assert _wilson(0, 0)["wilson95"] is None
    w = _wilson(1, 2)["wilson95"]
    assert 0.0 <= w[0] <= 0.5 <= w[1] <= 1.0
    rows = _rows(list(ACCEPTED_A) * 3 + list(ALTERNATE_A) * 3, 0)
    assert len(_runs(rows, exclude_a=True)) == 1
    assert not _runs(_rows(list(ACCEPTED_A) * 3, 0), exclude_a=True)
    print(json.dumps({"status": "passed", "checks": ["wilson_endpoints", "alternate_template_distinction"]}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        _selfcheck()
        return
    if args.input is None or args.output is None:
        ap.error("--input and --output are required unless --selfcheck is used")
    result = summarize(json.loads(args.input.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "trajectories": result["selection"]["selected_cells"],
                      "output": str(args.output)}))


if __name__ == "__main__":
    main()
