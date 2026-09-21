"""Freeze Lane D paired states from the shared recurrence panel.

CPU-only preparation. This entry never loads a model or launches a job. A
missing or underspecified panel fails closed instead of inventing a cohort.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_PANEL = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-19-recurrence-distribution-census/shared-panel.json"
)
DEFAULT_SOURCES = DEFAULT_PANEL.with_name("shared-sources.json")
DEFAULT_OUTPUT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-19-coordinate-input-continuity"
)
COORD_BASE = 151670
ROLES = ("x1", "y1", "x2", "y2")


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def binding(path: Path) -> dict[str, Any]:
    p = path.resolve()
    b = p.read_bytes()
    return {"path": str(p), "sha256": hashlib.sha256(b).hexdigest(), "size_bytes": len(b)}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _panel_boundaries(panel: dict[str, Any]) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for key in ("boundaries", "existing_boundaries"):
        raw = panel.get(key)
        if isinstance(raw, list):
            values.extend(x for x in raw if isinstance(x, dict))
    raw_new = panel.get("new_boundaries")
    if isinstance(raw_new, list):
        values.extend(x for x in raw_new if isinstance(x, dict))
    cohort = panel.get("new_cohort")
    if isinstance(cohort, dict) and isinstance(cohort.get("boundaries"), list):
        values.extend(x for x in cohort["boundaries"] if isinstance(x, dict))
    by_id: dict[str, dict[str, Any]] = {}
    for value in values:
        ident = str(value.get("id", value.get("boundary_id", "")))
        if ident:
            by_id.setdefault(ident, value)
    return list(by_id.values())


def _classify(boundary: dict[str, Any]) -> str | None:
    kind = str(boundary.get("kind", "")).lower()
    stratum = str(boundary.get("selection_stratum", "")).lower()
    role = str(boundary.get("role", "")).lower()
    if kind == "failure" or "failure" in stratum or "failure" in role:
        return "failure"
    if kind in {"healthy", "proxy", "nonrecurrent_proxy"}:
        return "proxy"
    if "proxy" in stratum or "healthy" in stratum or "nonrecurrent" in stratum:
        return "proxy"
    return None


def _scene_key(boundary: dict[str, Any]) -> tuple[str, str, str]:
    record = boundary.get("input_record")
    if isinstance(record, dict):
        image = record.get("image_id", boundary.get("image_id", ""))
        file_name = record.get("file_name", "")
    else:
        image = boundary.get("image_id", "")
        file_name = boundary.get("file_name", "")
    return (str(boundary.get("model", "")), str(image), str(file_name))


def _sort_key(boundary: dict[str, Any]) -> tuple[str, str, str, str, str]:
    scene = _scene_key(boundary)
    return (*scene, str(boundary.get("group", "")), str(boundary.get("id", boundary.get("boundary_id", ""))))


def _source_row(boundary: dict[str, Any]) -> dict[str, Any]:
    row = boundary.get("source_row")
    if not isinstance(row, dict):
        raise ValueError("boundary has no source_row")
    offsets = row.get("coordinate_offsets")
    values = row.get("values")
    if not isinstance(offsets, list) or len(offsets) != 4:
        raise ValueError("source_row.coordinate_offsets must have four entries")
    if not isinstance(values, list) or len(values) != 4:
        raise ValueError("source_row.values must have four entries")
    return row


def _feasible_site(boundary: dict[str, Any]) -> dict[str, Any]:
    row = _source_row(boundary)
    values = [int(x) for x in row["values"]]
    offsets = [int(x) for x in row["coordinate_offsets"]]
    tokens = [int(x) for x in boundary.get("native_tokens", [])]
    if not tokens:
        raise ValueError("boundary has no native_tokens")
    for role_index in reversed(range(4)):
        offset = offsets[role_index]
        value = values[role_index]
        if offset < 0 or offset >= len(tokens) or tokens[offset] != COORD_BASE + value:
            raise ValueError(f"source coordinate token mismatch at offset {offset}")
        feasible: list[int] = []
        direction_flags: dict[str, dict[str, Any]] = {}
        for delta in (-1, 1):
            changed = values.copy()
            changed[role_index] += delta
            if not 0 <= changed[role_index] <= 999:
                continue
            order_valid = bool(changed[0] < changed[2] and changed[1] < changed[3])
            feasible.append(delta)
            direction_flags[str(delta)] = {
                "delta": delta,
                "values": changed,
                "in_vocabulary": True,
                "geometry_valid": order_valid,
                "order_valid": order_valid,
            }
        if feasible:
            return {
                "role": ROLES[role_index],
                "role_index": role_index,
                "offset": offset,
                "value": value,
                "feasible_deltas": feasible,
                "direction_flags": direction_flags,
                "original_values": values,
                "geometry_valid_original": bool(values[0] < values[2] and values[1] < values[3]),
                "order_valid_original": bool(values[0] < values[2] and values[1] < values[3]),
            }
    raise ValueError("no feasible +/-1 history coordinate")


def _target_slots(boundary: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    slots = boundary.get("target_slots")
    if not isinstance(slots, list):
        raise ValueError("boundary has no target_slots")
    parsed = [x for x in slots if isinstance(x, dict) and x.get("role") == "x1"]
    immediate = [x for x in parsed if int(x.get("row_delay", -1)) == 0]
    later = [x for x in parsed if int(x.get("row_delay", -1)) > 0]
    if not immediate or not later:
        raise ValueError("boundary lacks immediate and later x1 target slots")
    immediate = min(immediate, key=lambda x: (int(x["offset"]), int(x.get("row_index", 0))))
    later = min(later, key=lambda x: (int(x.get("row_delay", 0)), int(x["offset"])))
    if int(later["offset"]) <= int(immediate["offset"]):
        raise ValueError("later target slot is not after immediate target slot")
    return immediate, later


def _candidate(boundary: dict[str, Any], stratum: str) -> dict[str, Any]:
    for field in ("raw_path", "trace_path", "receipt_path", "batch_index", "native_token_hash"):
        if field not in boundary:
            raise ValueError(f"boundary missing source binding field: {field}")
    site = _feasible_site(boundary)
    immediate, later = _target_slots(boundary)
    tokens = [int(x) for x in boundary["native_tokens"]]
    if int(site["offset"]) >= int(immediate["offset"]):
        raise ValueError("replacement site is not before immediate target")
    if int(later["offset"]) >= len(tokens):
        raise ValueError("later target is outside native token sequence")
    return {"boundary": boundary, "stratum": stratum, "site": site, "scores": {"immediate": immediate, "later": later}}


def _balanced(candidates: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in sorted(candidates, key=lambda x: _sort_key(x["boundary"])):
        by_model[str(candidate["boundary"].get("model", ""))].append(candidate)
    models = sorted(by_model)
    if not models:
        raise ValueError("no candidates")
    quotas = {model: limit // len(models) for model in models}
    for model in models[: limit - sum(quotas.values())]:
        quotas[model] += 1
    chosen: list[dict[str, Any]] = []
    used: dict[str, int] = defaultdict(int)
    scenes: set[tuple[str, str, str]] = set()
    while len(chosen) < limit:
        progress = False
        for model in models:
            if used[model] >= quotas[model]:
                continue
            candidate = next((x for x in by_model[model] if _scene_key(x["boundary"]) not in scenes), None)
            if candidate is None:
                continue
            by_model[model].remove(candidate)
            chosen.append(candidate)
            used[model] += 1
            scenes.add(_scene_key(candidate["boundary"]))
            progress = True
            if len(chosen) == limit:
                break
        if not progress:
            break
    remaining = sorted((x for group in by_model.values() for x in group), key=lambda x: _sort_key(x["boundary"]))
    for candidate in remaining:
        if len(chosen) == limit:
            break
        if _scene_key(candidate["boundary"]) in scenes:
            continue
        chosen.append(candidate)
        scenes.add(_scene_key(candidate["boundary"]))
    if len(chosen) != limit:
        raise ValueError(f"only {len(chosen)} distinct scenes available; need {limit}")
    return chosen


def _state(candidate: dict[str, Any], index: int) -> dict[str, Any]:
    boundary = candidate["boundary"]
    site = candidate["site"]
    immediate = candidate["scores"]["immediate"]
    later = candidate["scores"]["later"]
    tokens = [int(x) for x in boundary["native_tokens"]]
    variants = [
        {
            "name": "native",
            "delta": 0,
            "token_id": COORD_BASE + int(site["value"]),
            "in_vocabulary": True,
            "geometry_valid": bool(site["geometry_valid_original"]),
            "order_valid": bool(site["order_valid_original"]),
            "values": list(site["original_values"]),
        }
    ]
    for delta in site["feasible_deltas"]:
        flags = site["direction_flags"][str(delta)]
        variants.append(
            {
                "name": f"delta{delta:+d}",
                "delta": int(delta),
                "token_id": COORD_BASE + int(site["value"]) + int(delta),
                "in_vocabulary": bool(flags["in_vocabulary"]),
                "geometry_valid": bool(flags["geometry_valid"]),
                "order_valid": bool(flags["order_valid"]),
                "values": list(flags["values"]),
            }
        )
    ident = boundary.get("id", boundary.get("boundary_id"))
    return {
        "id": f"d-{index:02d}-{candidate['stratum']}-{ident}",
        "stratum": candidate["stratum"],
        "boundary_id": str(ident),
        "model": boundary.get("model"),
        "group": boundary.get("group"),
        "image_id": boundary.get("image_id"),
        "scene_key": list(_scene_key(boundary)),
        "source": {
            "native_tokens_sha256": digest(tokens),
            "source_row": boundary["source_row"],
            "native_prefix_end": int(immediate["offset"]),
            "native_later_prefix_end": int(later["offset"]),
            "fixed_suffix": {
                "start": int(site["offset"]) + 1,
                "end_exclusive": int(later["offset"]),
                "token_sha256": digest(tokens[int(site["offset"]) + 1 : int(later["offset"])]),
            },
        },
        "site": site,
        "score_sites": {
            "immediate": {**immediate, "logit_index": int(immediate["offset"]) - 1},
            "later": {**later, "logit_index": int(later["offset"]) - 1},
        },
        "variants": variants,
        "runtime": {
            "companions": "exact native group, unchanged",
            "modes": ["native_no_hook", "observational_hook"],
            "generation": "none",
            "full_prefix_horizon": int(later["offset"]),
            "capture_both_score_sites_in_one_replay": True,
        },
    }


def selfcheck() -> None:
    """Falsify the old geometry gate with an invalid source-row fixture."""
    tokens = [151646, 8987, 151647, 151648, 151674, 151675, 151675, 151675, 151649]
    boundary = {
        "id": "selfcheck-invalid",
        "model": "tied",
        "group": "selfcheck",
        "image_id": 1,
        "source_row": {
            "coordinate_offsets": [4, 5, 6, 7],
            "values": [4, 5, 5, 5],
        },
        "native_tokens": tokens,
    }
    site = _feasible_site(boundary)
    assert site["role"] == "y2"
    assert site["feasible_deltas"] == [-1, 1]
    assert site["geometry_valid_original"] is False
    assert site["direction_flags"]["-1"]["geometry_valid"] is False
    assert site["direction_flags"]["1"]["geometry_valid"] is True


def prepare(panel_path: Path, sources_path: Path, output: Path) -> dict[str, Any]:
    if not panel_path.is_file() or not sources_path.is_file():
        raise FileNotFoundError(f"shared panel/sources not ready: {panel_path} {sources_path}")
    panel = json.loads(panel_path.read_text())
    sources = json.loads(sources_path.read_text())
    if panel.get("status") not in {"frozen_rule_prelaunch", "final_frozen", "frozen", "candidate", "complete"}:
        raise ValueError(f"panel is not frozen: {panel.get('status')}")
    candidates: dict[str, list[dict[str, Any]]] = {"failure": [], "proxy": []}
    excluded: list[dict[str, Any]] = []
    for boundary in _panel_boundaries(panel):
        stratum = _classify(boundary)
        if stratum is None:
            excluded.append({"id": boundary.get("id", boundary.get("boundary_id")), "reason": "unknown_stratum"})
            continue
        try:
            candidates[stratum].append(_candidate(boundary, stratum))
        except (KeyError, TypeError, ValueError) as exc:
            excluded.append({"id": boundary.get("id", boundary.get("boundary_id")), "reason": str(exc)})
    selected = _balanced(candidates["failure"], 8) + _balanced(candidates["proxy"], 8)
    states = [_state(candidate, i) for i, candidate in enumerate(selected)]
    role_coverage = dict(sorted(Counter(str(state["site"]["role"]) for state in states).items()))
    plan = {
        "schema": "coordinate_continuity.execution_plan.v1",
        "status": "frozen_before_model_calls",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "panel": binding(panel_path),
        "shared_sources": binding(sources_path),
        "selection_rule": "8 failure + 8 proxy; at most one boundary per model/image scene; balanced tied/untied quotas; stable panel order",
        "site_rule": "latest source-row coordinate with at least one +/-1 in [0,999]; geometry/order are reported flags, never feasibility gates",
        "score_rule": "first x1 row_delay=0 target and first positive row_delay x1 target; logit index is target offset - 1",
        "suffix_rule": "native token span after replaced site through later target prefix, unchanged for every variant",
        "role_coverage": role_coverage,
        "role_scope": "report observed site roles; no extra role sweep or all-role generalization",
        "states": states,
        "excluded": excluded,
        "bounds": {"states": 16, "variants_per_state_max": 3, "modes_per_variant": 2, "max_full_prefix_batch_forwards": 96, "max_native_batch_forwards": 512, "gpu": 7, "gpu_hours": 2, "tensor_bytes": 4294967296},
        "runtime_schema": {
            "capture": "runtime/<state_id>/<variant>/<mode>/capture.pt",
            "release": "runtime/<state_id>/<variant>/release.json",
            "receipt": "runtime/<state_id>/<variant>/receipt.json",
            "required_keys": ["input_ids", "input_delta", "layer_input_delta", "layer_residual_delta", "head_input_delta", "logits", "coordinate_logits", "top2", "native_winner", "native_margin", "hook_parity"],
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "selection.json", {"schema": "coordinate_continuity.selection.v1", "plan": plan})
    write_json(output / "execution-plan.json", plan)
    return plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        selfcheck()
        print(json.dumps({"status": "selfcheck_passed"}, indent=2))
        return
    plan = prepare(args.panel, args.sources, args.output_root)
    print(json.dumps({"status": plan["status"], "states": len(plan["states"]), "excluded": len(plan["excluded"])}, indent=2))


if __name__ == "__main__":
    main()
