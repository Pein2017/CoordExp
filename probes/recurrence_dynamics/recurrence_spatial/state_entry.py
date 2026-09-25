"""CPU-only binding of the final shared spatial panel to native source states.

This module is deliberately an entry contract, not a scheduler.  A final
shared panel must expose a list under ``states`` or ``boundaries``.  Each
state points at the saved natural raw output and contains the selected source
row.  The resolver checks image identity, the complete source row, and the
serialized prefix before any producer is allowed to construct a transformed
cell.

The old ``existing_boundaries`` field is accepted only with
``allow_prelaunch=True`` for a CPU compatibility check.  It cannot silently
stand in for the final shared mechanism panel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration"
)
DEFAULT_PANEL = ROOT / "2026-09-19-recurrence-distribution-census" / "shared-panel.json"
DEFAULT_OUT = ROOT / "2026-09-19-recurrence-spatial-source"
OBJ_START = 151646
COORD_BASE = 151670
COORD_LIMIT = COORD_BASE + 1000


def digest_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def binding(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "size_bytes": path.stat().st_size}


def _path_binding(value: Any, *, label: str) -> Path:
    if isinstance(value, dict):
        value = value.get("path")
    if not value:
        raise ValueError(f"{label} has no path")
    path = Path(str(value)).expanduser().resolve(strict=True)
    return path


def _nested_path(value: Any, *keys: str) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _source_path(state: dict[str, Any], *, key: str) -> Path:
    candidates = [
        state.get(key),
        _nested_path(state, "source", key),
        _nested_path(state, "bindings", key),
    ]
    if key == "raw_path":
        candidates.extend(
            [
                _nested_path(state, "source", "mature_raw"),
                _nested_path(state, "source", "raw"),
                _nested_path(state, "bindings", "raw"),
            ]
        )
    for candidate in candidates:
        if candidate:
            return _path_binding(candidate, label=key)
    raise ValueError(f"state {state.get('id', '<unknown>')} has no explicit {key}")


def _panel_candidates(panel: dict[str, Any], state: dict[str, Any], override: Path | None) -> list[Path]:
    """Collect declared panel references, including the shared-sources index."""

    candidates: list[Path] = []
    seen: set[Path] = set()

    def add(value: Any) -> None:
        if not value:
            return
        try:
            path = _path_binding(value, label="mature_panel")
        except (TypeError, ValueError, OSError):
            return
        if path not in seen:
            seen.add(path)
            candidates.append(path)

    if override is not None:
        add(override)
    for value in (
        state.get("mature_panel_path"),
        _nested_path(state, "source", "mature_panel"),
        _nested_path(state, "source", "panel"),
        _nested_path(panel, "source", "mature_panel"),
        _nested_path(panel, "sources", "mature_panel"),
    ):
        add(value)
    sources = panel.get("sources")
    if isinstance(sources, list):
        for item in sources:
            if isinstance(item, dict):
                add(item.get("path"))

    # The final panel points to shared-sources.json, whose mature, feedback,
    # and new-cohort panel references are the actual group owners.
    for source in list(candidates):
        if source.name != "shared-sources.json":
            continue
        try:
            index = json.loads(source.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for key in ("mature_panel", "feedback_panel", "panel", "new_panel"):
            add(index.get(key))
    return [path for path in candidates if path.name == "panel.json"]


def _panel_path(panel: dict[str, Any], state: dict[str, Any], override: Path | None) -> Path:
    wanted = str(state.get("group", state.get("group_key", "")))
    for candidate in _panel_candidates(panel, state, override):
        try:
            value = json.loads(candidate.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        groups = value.get("groups")
        if isinstance(groups, list) and any(str(group.get("key")) == wanted for group in groups):
            return candidate
    raise ValueError(f"no declared mature panel contains group {wanted!r}")


def _state_list(panel: dict[str, Any], *, allow_prelaunch: bool) -> tuple[str, list[dict[str, Any]]]:
    for key in ("states", "boundaries", "mechanism_states", "all_boundaries"):
        value = panel.get(key)
        if isinstance(value, list):
            return key, value
    if allow_prelaunch and isinstance(panel.get("existing_boundaries"), list):
        return "existing_boundaries", panel["existing_boundaries"]
    raise ValueError(
        "shared panel must contain final states/boundaries; "
        "existing_boundaries requires explicit allow_prelaunch"
    )


def iter_shared_states(panel: dict[str, Any], *, allow_prelaunch: bool = False) -> Iterable[dict[str, Any]]:
    """Yield state records from the explicitly declared panel collection."""

    key, states = _state_list(panel, allow_prelaunch=allow_prelaunch)
    for index, state in enumerate(states):
        if not isinstance(state, dict):
            raise ValueError(f"{key}[{index}] is not an object")
        current = dict(state)
        current.setdefault("selection_collection", key)
        yield current


def _case_image_path(group: dict[str, Any], case: dict[str, Any]) -> Path:
    candidate = case.get("image_path")
    if candidate:
        return Path(str(candidate)).expanduser().resolve(strict=True)
    record = case.get("input_record", {})
    images = record.get("images") or []
    if not images:
        raise ValueError("target case has neither image_path nor input_record.images")
    path = Path(str(images[0])).expanduser()
    if not path.is_absolute():
        path = Path(str(group["input_jsonl"])).resolve().parent / path
    return path.resolve(strict=True)


def _compact_case(group: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    record = case.get("input_record", {})
    image = _case_image_path(group, case)
    return {
        "row_id": case.get("row_id"),
        "row_index": case.get("row_index"),
        "image_id": int(record.get("image_id")),
        "file_name": record.get("file_name"),
        "image": binding(image),
    }


def _find_group(panel: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    groups = panel.get("groups")
    if not isinstance(groups, list):
        raise ValueError("mature panel has no groups list")
    wanted = str(state.get("group", state.get("group_key", "")))
    matches = [group for group in groups if str(group.get("key")) == wanted]
    if len(matches) != 1:
        raise ValueError(f"expected one mature-panel group {wanted!r}, got {len(matches)}")
    return matches[0]


def _find_case(group: dict[str, Any], state: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cases = group.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"group {group.get('key')} has no cases")
    image_id = int(state["image_id"])
    matches = [case for case in cases if int(case.get("input_record", {}).get("image_id", -1)) == image_id]
    if len(matches) != 1:
        raise ValueError(f"expected one image {image_id} in group {group.get('key')}, got {len(matches)}")
    case = matches[0]
    if "batch_index" in state:
        # Selection records use the local position in the saved group batch;
        # the mature panel's row_index is often the global source row.
        local_index = cases.index(case)
        if local_index != int(state["batch_index"]):
            raise ValueError(
                f"batch-index drift for {state.get('id')}: group position {local_index} != state {state['batch_index']}"
            )
    return case, cases


def _raw_row(raw: dict[str, Any], state: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    rows = raw.get("rows")
    if not isinstance(rows, list):
        raise ValueError("saved raw source has no rows list")
    image_id = int(state["image_id"])
    matches = [row for row in rows if int(row.get("image_id", -1)) == image_id]
    if len(matches) != 1:
        raise ValueError(f"expected one raw row for image {image_id}, got {len(matches)}")
    row = matches[0]
    expected_row_id = case.get("row_id")
    if expected_row_id is not None and row.get("row_id") not in (None, expected_row_id):
        raise ValueError(f"raw row identity drift: {row.get('row_id')} != {expected_row_id}")
    return row


def _validate_state_shape(state: dict[str, Any]) -> None:
    missing = [key for key in ("id", "model", "image_id", "source_row") if key not in state]
    if missing:
        raise ValueError(f"state missing required fields: {missing}")
    if "group" not in state and "group_key" not in state:
        raise ValueError("state missing required group/group_key")
    if state["model"] not in {"tied", "untied"}:
        raise ValueError(f"unsupported model {state['model']!r}")
    policy = state.get("policy")
    condition = str(state.get("condition", ""))
    if policy not in (None, "original") and not condition.endswith("-original"):
        raise ValueError(f"spatial lane accepts original policy only, got policy={policy!r}")


def resolve_state(
    shared_panel: dict[str, Any],
    state: dict[str, Any],
    *,
    panel_path: Path | None = None,
) -> dict[str, Any]:
    """Resolve and verify one state without loading a model or image tensor."""

    _validate_state_shape(state)
    mature_path = _panel_path(shared_panel, state, panel_path)
    mature_panel = json.loads(mature_path.read_text())
    group = _find_group(mature_panel, state)
    case, cases = _find_case(group, state)
    image_path = _case_image_path(group, case)
    raw_path = _source_path(state, key="raw_path")
    raw = json.loads(raw_path.read_text())
    row = _raw_row(raw, state, case)
    tokens = [int(token) for token in row.get("token_ids", [])]
    if not tokens:
        raise ValueError(f"raw row for {state['id']} has no token_ids")
    native_hash = digest_json(tokens)
    if state.get("native_token_hash") and state["native_token_hash"] != native_hash:
        raise ValueError(f"native token hash drift for {state['id']}")
    selected = dict(state["source_row"])
    source_index = int(selected["index"])
    starts = [index for index, token in enumerate(tokens) if token == OBJ_START]
    if source_index < 0 or source_index >= len(starts):
        raise ValueError(f"source row index out of range for {state['id']}")
    source_start = starts[source_index]
    source_end = starts[source_index + 1] if source_index + 1 < len(starts) else len(tokens)
    if source_start != int(selected["start"]) or source_end != int(selected["end"]):
        raise ValueError(f"source row boundary drift for {state['id']}")
    prefix = tokens[:source_end]
    prefix_hash = digest_json(prefix)
    expected_prefix_hash = state.get("prefix_hash") or state.get("source_prefix_hash")
    if expected_prefix_hash and expected_prefix_hash != prefix_hash:
        raise ValueError(f"prefix hash drift for {state['id']}")
    offsets = [int(offset) for offset in selected.get("coordinate_offsets", [])]
    values = [int(value) for value in selected.get("values", [])]
    if len(offsets) != 4 or len(values) != 4:
        raise ValueError(f"source row coordinates are incomplete for {state['id']}")
    actual_values = [tokens[offset] - COORD_BASE for offset in offsets]
    if actual_values != values:
        raise ValueError(f"source row coordinate drift for {state['id']}: {actual_values} != {values}")
    if any(token < COORD_BASE or token >= COORD_LIMIT for token in (tokens[offset] for offset in offsets)):
        raise ValueError(f"source row coordinate token drift for {state['id']}")
    target = _compact_case(group, case)
    companions = [_compact_case(group, item) for item in cases if item is not case]
    source_condition = state.get("condition") or f"{state['model']}-original"
    route = {
        "builder": "src.inference.bound_requests.build_bound_native_requests",
        "qualified_call": "build_bound_native_requests(q, config, [target_case])[0][0]",
        "policy": "original",
        "target_only": True,
        "requested_case_count": 1,
        "target": target,
        "source_group_case_count": len(cases),
        "source_group_companions": companions,
        "companion_use": "provenance_only; companions are not passed to the target-only spatial rerun",
        "source_runtime_difference": (
            "saved natural prefix is bound to the original group raw output; "
            "the qualified transformed-cell request rebuilds the target image/prefix "
            "through the target-only native route"
        ),
    }
    return {
        "id": str(state["id"]),
        "model": str(state["model"]),
        "condition": source_condition,
        "policy": "original",
        "selection_collection": state.get("selection_collection"),
        "kind": state.get("kind"),
        "selection_stratum": state.get("selection_stratum", state.get("episode_stratum")),
        "state_metadata": state.get("metadata", {}),
        "source": {
            "mature_panel": binding(mature_path),
            "raw": binding(raw_path),
            "group": str(group["key"]),
            "image_id": int(state["image_id"]),
            "batch_index": state.get("batch_index"),
            "split": state.get("split") or target.get("split"),
            "image": binding(image_path),
            "case": target,
            "raw_row_id": row.get("row_id"),
        },
        "prefix": {
            "source_row_index": source_index,
            "source_row_start": source_start,
            "source_row_end": source_end,
            "source_token_count": len(prefix),
            "source_sha256": prefix_hash,
            "native_token_count": len(tokens),
            "native_token_sha256": native_hash,
            "source_row": selected,
            "source_valid": bool(selected.get("valid")),
            "next_row_index": source_index + 1,
        },
        "native_route": route,
    }


def build_readiness(
    panel_path: Path,
    *,
    out_path: Path | None = None,
    mature_panel_path: Path | None = None,
    allow_prelaunch: bool = False,
) -> dict[str, Any]:
    """Build a CPU receipt; no model, processor, or image tensor is loaded."""

    panel_path = panel_path.resolve()
    receipt: dict[str, Any] = {
        "schema": "recurrence_spatial.shared_panel_entry.v1",
        "unit_id": "2026-09-19-recurrence-spatial-source",
        "panel": {"path": str(panel_path), "exists": panel_path.exists()},
        "allow_prelaunch": allow_prelaunch,
        "model_calls": 0,
        "gpu_seconds": 0.0,
        "states": [],
        "errors": [],
    }
    if not panel_path.exists():
        receipt.update(
            status="waiting_final_shared_panel",
            blocked_until="final shared-panel.json is published by Lane A",
            final_panel_required=True,
        )
    else:
        receipt["panel"] = binding(panel_path)
        panel = json.loads(panel_path.read_text())
        try:
            collection, raw_states = _state_list(panel, allow_prelaunch=allow_prelaunch)
            receipt["state_collection"] = collection
            receipt["declared_state_count"] = len(raw_states)
            for raw_state in iter_shared_states(panel, allow_prelaunch=allow_prelaunch):
                try:
                    receipt["states"].append(resolve_state(panel, raw_state, panel_path=mature_panel_path))
                except Exception as exc:  # retain every unresolved state for parent diagnosis
                    receipt["errors"].append({"id": raw_state.get("id"), "error": repr(exc)})
            receipt["resolved_state_count"] = len(receipt["states"])
            receipt["status"] = "ready" if not receipt["errors"] else "blocked_state_bindings"
            receipt["final_panel_required"] = collection == "existing_boundaries"
        except Exception as exc:
            receipt["status"] = "blocked_panel_schema"
            receipt["errors"].append({"id": None, "error": repr(exc)})
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT / "generic-entry-readiness.json")
    parser.add_argument("--mature-panel", type=Path)
    parser.add_argument("--allow-prelaunch", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build_readiness(args.panel, out_path=args.out, mature_panel_path=args.mature_panel, allow_prelaunch=args.allow_prelaunch), indent=2))
