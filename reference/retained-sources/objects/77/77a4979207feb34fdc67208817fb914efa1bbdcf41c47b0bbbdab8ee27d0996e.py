"""Exact full-vocabulary scorer for the frozen Lane A spatial gate."""
from __future__ import annotations

import argparse
import copy
import datetime as _datetime
import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Mapping

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch

from probes.training_set_completion.coordinate_continuity.runtime import _source
from probes.training_set_completion.native_row_choice.runtime import (
    _make_histories,
    _row_role,
    _score_candidate,
    _trace_compare,
)
from probes.training_set_completion.numerical_feedback.select import rows as parsed_rows
from src.qwen.input_identity import input_identity as _input_identity
from src.artifacts.source_provenance import preserve_source
from probes.training_set_completion.untied_shared import load_model


PREDECESSOR_ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-21-spatial-progress-gate"
)
ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-21-spatial-progress-recovery"
)
ADMISSION = PREDECESSOR_ROOT / "selection" / "shared-admission.json"
BUDGET = PREDECESSOR_ROOT / "selection" / "budget-estimate.json"
WALL_LIMIT = PREDECESSOR_ROOT / "selection" / "wall-limit-amendment.json"
DISPATCH = ROOT / "dispatch.json"
ADMISSION_SHA256 = "8863a4eb6d00ed9cb27cf8dd41e32429e7081e3bf363907b23bf4a7a0ba1d7d8"
BUDGET_SHA256 = "ad027198065151367a9778171f7c109ab718e4ccf428658f4d4352899dc6401a"
WALL_LIMIT_SHA256 = "3e1fd8135882c3c144c877e3a47e4c9977eb8f651345514a571422637ed0b836"
ATOL = 2e-4
EOS, ROW_OPEN, REF_END, BOX_START, COORD_BASE, ROW_END = 151645, 151646, 151647, 151648, 151670, 151649


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _binding(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return {"path": str(path), "sha256": hasher.hexdigest(), "size_bytes": path.stat().st_size}


def _write_new(path: Path, value: Any) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return _binding(path)


def _copy_new(source: Path, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as src, destination.open("xb") as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)
        dst.flush()
        os.fsync(dst.fileno())
    return _binding(destination)


def _require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _finite(values: torch.Tensor) -> list[float]:
    result = [float(item) for item in values.detach().float().cpu().tolist()]
    _require(all(math.isfinite(item) for item in result), "nonfinite full-vocabulary payload")
    return result


def _logsumexp(values: list[float]) -> float | None:
    if not values:
        return None
    tensor = torch.tensor(values, dtype=torch.float64)
    return float(torch.logsumexp(tensor, dim=0).item())


def _find_boundary(panel: dict[str, Any], boundary_id: str) -> dict[str, Any]:
    matches = [item for item in panel.get("all_boundaries", []) if item.get("id") == boundary_id]
    if len(matches) != 1:
        matches = [item for item in panel.get("boundaries", []) if item.get("id") == boundary_id]
    _require(len(matches) == 1, f"source boundary is not unique: {boundary_id}")
    return matches[0]


def _source_record(admission: dict[str, Any], boundary_id: str) -> dict[str, Any]:
    matches = [item for item in admission["source_pool"] if item.get("source_boundary_id") == boundary_id]
    _require(len(matches) == 1, f"source record is not unique: {boundary_id}")
    return matches[0]


def _cells(admission: dict[str, Any]) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for boundary in admission["lane_a"]["boundaries"]:
        for condition in boundary["unique_execution_conditions"]:
            item = copy.deepcopy(boundary)
            item["condition"] = condition
            item["id"] = f"{boundary['id']}--{condition}"
            cells.append(item)
    _require(len(cells) == 8, f"Lane A unique-cell count changed: {len(cells)}")
    return cells


def _load_admission() -> tuple[dict[str, Any], dict[str, Any]]:
    _require(ADMISSION.is_file(), f"missing frozen admission: {ADMISSION}")
    observed = _binding(ADMISSION)
    _require(observed["sha256"] == ADMISSION_SHA256, "frozen admission hash changed")
    admission = json.loads(ADMISSION.read_text())
    _require(admission.get("status") == "frozen_before_intervention_scores", "admission is not frozen")
    _require(admission.get("schema") == "spatial_progress_visual_binding.shared_admission.v1", "admission schema changed")
    _require(admission.get("lane_a", {}).get("counts", {}).get("unique_scientific_cells") == 8, "Lane A cell count changed")
    _require(len(admission.get("lane_a", {}).get("boundaries", [])) == 2, "Lane A boundary count changed")
    budget = json.loads(BUDGET.read_text())
    budget_binding = _binding(BUDGET)
    _require(budget_binding["sha256"] == BUDGET_SHA256, "frozen budget estimate hash changed")
    _require(budget.get("status") == "frozen_before_lane_model_entry", "budget estimate is not frozen")
    wall = json.loads(WALL_LIMIT.read_text())
    wall_binding = _binding(WALL_LIMIT)
    _require(wall_binding["sha256"] == WALL_LIMIT_SHA256, "wall-limit amendment hash changed")
    _require(wall.get("status") == "frozen_before_model_entry", "wall-limit amendment is not frozen")
    _require(int(wall.get("hard_package_wall_seconds", 0)) == 7200, "package wall limit changed")
    return admission, {
        "admission": observed, "budget": budget_binding, "budget_value": budget,
        "wall_limit": wall_binding, "wall_value": wall,
    }


def _candidate_rows(boundary: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    result: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()
    for set_name in ("A", "N"):
        for candidate in boundary["candidate_sets"].get(set_name, []):
            ident = str(candidate["id"])
            _require(ident not in seen, f"duplicate candidate ID: {ident}")
            seen.add(ident)
            tokens = [int(token) for token in candidate["tokens"]]
            _require(candidate.get("token_sha256") == _digest(tokens), f"candidate token binding changed: {ident}")
            parsed = parsed_rows(tokens)
            _require(len(parsed) == 1 and bool(parsed[0]["valid"]), f"candidate geometry changed: {ident}")
            result.append((set_name, candidate))
    return result


def _actual_row(native: list[int], row_index: int) -> dict[str, Any]:
    matches = [row for row in parsed_rows(native) if int(row["index"]) == int(row_index)]
    _require(len(matches) == 1, f"actual source row is not unique: {row_index}")
    return matches[0]


def _actual_tokens(native: list[int], actual: Mapping[str, Any]) -> list[int]:
    """Recover the serialized row from parser offsets at the caller boundary."""
    start = int(actual["start"])
    end = int(actual["end"])
    _require(0 <= start < end <= len(native), "actual source row offsets are invalid")
    tokens = [int(token) for token in native[start:end]]
    parsed = parsed_rows(tokens)
    _require(len(parsed) == 1 and bool(parsed[0]["valid"]), "actual source row slice is not complete")
    return tokens


def _condition_prefix(boundary: dict[str, Any], native: list[int], condition: str) -> tuple[list[int], dict[str, Any]]:
    matches = [item for item in boundary["conditions"] if item["id"] == condition]
    _require(len(matches) == 1, f"condition is not frozen: {condition}")
    spec = copy.deepcopy(matches[0])
    prefix_end = int(boundary["prefix_end"])
    prefix = native[:prefix_end]
    original = list(prefix)
    if condition == "native":
        _require(spec.get("history_tokens") == "exact_native_prefix", "native condition is not exact")
        check = {"passed": prefix == original, "absolute_offset": None, "old_token_id": None, "new_token_id": None}
    else:
        offset = int(spec["absolute_offset"])
        old_token = int(spec["old_token_id"])
        new_token = int(spec["new_token_id"])
        _require(0 <= offset < prefix_end, f"condition offset outside prefix: {condition}")
        _require(int(native[offset]) == old_token, f"condition old token drifted: {condition}")
        prefix[offset] = new_token
        check = {
            "passed": prefix[:offset] == original[:offset] and prefix[offset] == new_token and prefix[offset + 1 :] == original[offset + 1 :],
            "absolute_offset": offset,
            "old_token_id": old_token,
            "new_token_id": new_token,
        }
    spec["prefix_sha256"] = _digest(prefix)
    spec["native_prefix_sha256"] = _digest(original)
    spec["changed_prefix_check"] = check
    return prefix, spec


def _row_roles(tokens: list[int]) -> list[str]:
    parsed = parsed_rows(tokens)
    _require(len(parsed) == 1 and bool(parsed[0]["valid"]), "row is not a complete valid row")
    return [_row_role(tokens, offset) for offset in range(len(tokens))]


def _row_record(
    *,
    candidate: dict[str, Any],
    scored: dict[str, Any],
    condition: str,
    prefix: list[int],
    common_description: list[int],
) -> dict[str, Any]:
    tokens = [int(token) for token in candidate["tokens"]]
    logprobs = [float(value) for value in scored["token_logprobs"]]
    positions = _jsonable(scored["positions"])
    _require(len(tokens) == len(logprobs) == len(positions), f"row score lengths differ: {candidate['id']}")
    parsed = parsed_rows(tokens)[0]
    offsets = [int(item) for item in parsed["coordinate_offsets"]]
    xy1 = [offsets[0], offsets[1]]
    return {
        "candidate_id": str(candidate["id"]),
        "set": None,
        "owner": candidate["owner"],
        "source": candidate["source"],
        "construction_rule": candidate["construction_rule"],
        "values": list(candidate["values"]),
        "token_ids": tokens,
        "token_sha256": candidate["token_sha256"],
        "token_roles": _row_roles(tokens),
        "positions": positions,
        "token_logprobs": logprobs,
        "row_sum_logprob": float(sum(logprobs)),
        "logprob_includes": "complete_row_entry_through_terminator",
        "vocabulary_size": int(scored["vocabulary_size"]),
        "condition": condition,
        "conditional_xy1": {
            "path": "x1_then_y1",
            "token_offsets": xy1,
            "token_ids": [tokens[offset] for offset in xy1],
            "token_roles": ["x1", "y1"],
            "positions": [positions[offset] for offset in xy1],
            "token_logprobs": [logprobs[offset] for offset in xy1],
            "logprob_sum": float(sum(logprobs[offset] for offset in xy1)),
            "conditioning": {
                "prefix_sha256": _digest(prefix),
                "common_description_tokens": list(common_description),
                "recompute_y1_under_candidate_x1": True,
            },
        },
    }


def _validate_row(row: Mapping[str, Any]) -> None:
    tokens = row.get("token_ids")
    logprobs = row.get("token_logprobs")
    positions = row.get("positions")
    _require(isinstance(tokens, list) and all(type(token) is int for token in tokens), "row tokens are invalid")
    _require(isinstance(logprobs, list) and len(logprobs) == len(tokens), "row logprob length changed")
    _require(isinstance(positions, list) and len(positions) == len(tokens), "row position length changed")
    _require(len(tokens) >= 7 and tokens[0] == ROW_OPEN and tokens[-1] == ROW_END, "row opener or terminator omitted")
    parsed = parsed_rows(tokens)
    _require(len(parsed) == 1 and bool(parsed[0]["valid"]), "row is not one complete serialized row")
    _require(all(math.isfinite(float(value)) for value in logprobs), "row has nonfinite logprob")
    _require(abs(sum(float(value) for value in logprobs) - float(row["row_sum_logprob"])) <= 1e-5, "row probability accounting changed")
    xy1 = row.get("conditional_xy1")
    _require(isinstance(xy1, dict) and xy1.get("path") == "x1_then_y1", "x1/y1 conditional path missing")
    _require(len(xy1.get("token_ids", [])) == 2 and len(xy1.get("token_logprobs", [])) == 2, "x1/y1 conditional path malformed")
    _require(abs(sum(float(value) for value in xy1["token_logprobs"]) - float(xy1["logprob_sum"])) <= 1e-5, "x1/y1 probability accounting changed")


def _validate_scores(result: Mapping[str, Any], expected_ids: set[str]) -> None:
    rows = result.get("rows")
    _require(isinstance(rows, dict) and set(rows) == expected_ids, "candidate row coverage changed")
    for row in rows.values():
        _validate_row(row)
    sets = result.get("candidate_sets")
    _require(isinstance(sets, dict), "candidate sets missing")
    seen: set[str] = set()
    for name in ("A", "N"):
        group = sets.get(name)
        _require(isinstance(group, dict), f"candidate set missing: {name}")
        ids = [str(item) for item in group.get("row_ids", [])]
        _require(all(item in rows and item not in seen for item in ids), f"candidate set coverage changed: {name}")
        seen.update(ids)
        values = [float(rows[item]["row_sum_logprob"]) for item in ids]
        expected = _logsumexp(values)
        observed = group.get("row_logsumexp")
        _require((expected is None) == (observed is None), f"empty-set mass changed: {name}")
        if expected is not None:
            _require(abs(expected - float(observed)) <= 1e-5, f"finite-set mass changed: {name}")
    _require(seen == expected_ids, "candidate set union changed")


def _falsification(result: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    actual = next(iter(result["rows"].values()))
    for mutation in ("dropped_entry", "dropped_terminator"):
        corrupted = copy.deepcopy(actual)
        if mutation == "dropped_entry":
            for key in ("token_ids", "token_logprobs", "positions", "token_roles"):
                corrupted[key] = corrupted[key][1:]
        else:
            for key in ("token_ids", "token_logprobs", "positions", "token_roles"):
                corrupted[key] = corrupted[key][:-1]
        corrupted["row_sum_logprob"] = sum(float(value) for value in corrupted["token_logprobs"])
        try:
            _validate_row(corrupted)
        except (AssertionError, KeyError, TypeError, ValueError) as error:
            checks.append({"mutation": mutation, "rejected": True, "error": str(error)})
        else:
            checks.append({"mutation": mutation, "rejected": False, "error": None})
    return {
        "schema": "spatial_progress_gate.probability_accounting_falsification.v1",
        "checks": checks,
        "passed": all(item["rejected"] for item in checks),
    }


def _trace_parity(*, model: Any, batch: Any, raw: list[dict[str, Any]], target: int, native: list[int], prefix_end: int, trace: dict[str, Any], pad: int, device: torch.device) -> dict[str, Any]:
    actual = _actual_row(native, next(row["index"] for row in parsed_rows(native) if int(row["start"]) == prefix_end))
    actual_tokens = _actual_tokens(native, actual)
    prefix = native[:prefix_end]
    scored = _score_candidate(
        model=model, batch=batch, raw=raw, target=target, prefix=prefix,
        tokens=actual_tokens, pad=pad, device=device,
    )
    checks = [_trace_compare(
        logits=scored["boundary_logits"], trace=trace, batch_index=target,
        absolute_offset=prefix_end, token_id=int(native[prefix_end]), role="native_row_boundary",
    )]
    for offset, token_id in enumerate(actual_tokens):
        checks.append(_trace_compare(
            logits=scored["action_logits"][offset], trace=trace, batch_index=target,
            absolute_offset=prefix_end + offset, token_id=int(token_id), role=_row_role(actual_tokens, offset),
        ))
    return {
        "passed": all(item["passed"] for item in checks),
        "entry_included": checks[1]["role"] == "entry",
        "terminator_included": checks[-1]["role"] == "terminator",
        "max_full_vocab_logprob_abs_error": max(float(item["logprob_abs_error"]) for item in checks),
        "max_top2_abs_error": max(float(item["top2_max_abs_error"]) for item in checks),
        "tokens": checks,
    }


def _resolve_output(args: argparse.Namespace, cell: dict[str, Any]) -> Path:
    """Choose a recovery attempt directory without consulting predecessor output."""
    if args.output is not None:
        directory = args.output.resolve()
        _require(ROOT in directory.parents, f"output must be under recovery root: {directory}")
        _require(not directory.exists(), f"output already exists: {directory}")
        return directory
    if args.mode == "qualify":
        directory = ROOT / ("qualification-repair-01" if args.repair else "qualification-01")
        if args.repair:
            initial = ROOT / "qualification-01" / "receipt.json"
            _require(initial.is_file(), "qualification repair requires recovery qualification receipt")
            _require(json.loads(initial.read_text()).get("status") != "candidate_complete", "qualification already passed")
            _require(not directory.exists(), "qualification repair already exists")
        else:
            _require(not directory.exists(), "qualification campaign already exists")
        return directory
    suffix = f"--retry-{int(args.retry)}" if int(args.retry) else ""
    directory = ROOT / "cells" / f"{cell['id']}{suffix}"
    _require(not directory.exists(), f"cell attempt already exists: {directory}")
    return directory


class Runtime:
    def __init__(self, args: argparse.Namespace, cell: dict[str, Any] | None = None) -> None:
        self.args = args
        self.started = time.monotonic()
        self.admission, self.bindings = _load_admission()
        if args.mode == "qualify":
            matches = [item for item in _cells(self.admission) if item["id"] == "tied-14038-failure-before-row8--x1_before_N786"]
            _require(len(matches) == 1, "qualification cell is not frozen")
            self.cell = matches[0]
        else:
            _require(cell is not None, "cell mode requires a frozen cell")
            self.cell = cell

        # Resolve an explicit recovery output before checking campaign defaults.
        # This keeps immutable predecessor directories out of the recovery gate.
        directory = _resolve_output(args, self.cell)
        self.output = directory
        self.output.mkdir(parents=True, exist_ok=False)
        self.device = torch.device(args.device)
        self.counters: dict[str, Any] = {
            "model_forwards": 0, "vision_forwards": 0, "candidate_rows": 0,
            "retained_tensor_bytes": 0, "max_model_forwards": int(args.max_forwards),
            "max_seconds": float(args.max_seconds),
        }
        self.handles: list[Any] = []
        self.model: Any = None
        self.q: Any = None
        self.identity: Any = None
        self.shared_panel_path = Path(self.admission["panel"]["path"])
        self.shared_panel = json.loads(self.shared_panel_path.read_text())
        self.boundary = _find_boundary(self.shared_panel, self.cell["source_boundary_id"])
        self.source_panel_path = Path(_source_record(self.admission, self.cell["source_boundary_id"])["source_panel"]["path"])
        self.panel = json.loads(self.source_panel_path.read_text())
        self.source_record = _source_record(self.admission, self.cell["source_boundary_id"])
        source_raw = json.loads(Path(self.source_record["raw"]["path"]).read_text())["rows"]
        target_index = int(self.source_record["batch_index"])
        _require(0 <= target_index < len(source_raw), "source target index is outside raw rows")
        self.boundary["native_tokens"] = [int(token) for token in source_raw[target_index]["token_ids"]]
        self.boundary["native_token_hash"] = self.source_record["native_token_sha256"]
        self.boundary["raw_path"] = self.source_record["raw"]["path"]
        self.boundary["trace_path"] = self.source_record["trace"]["path"]
        self.boundary["receipt_path"] = self.source_record["receipt"]["path"]
        self._prepare_preentry()

    def _wall_start(self) -> dict[str, Any]:
        marker = ROOT / "coordination" / "first-model-launch.json"
        marker.parent.mkdir(parents=True, exist_ok=True)
        _require(DISPATCH.is_file(), f"recovery dispatch receipt is missing: {DISPATCH}")
        dispatch_binding = _binding(DISPATCH)
        dispatch = json.loads(DISPATCH.read_text())
        dispatch_start = str(dispatch.get("wall_start_utc", ""))
        dispatch_deadline = str(dispatch.get("wall_deadline_utc", ""))
        _require(dispatch_start and dispatch_deadline, "dispatch wall bounds are missing")
        dispatch_start_unix = _datetime.datetime.fromisoformat(dispatch_start.replace("Z", "+00:00")).timestamp()
        dispatch_deadline_unix = _datetime.datetime.fromisoformat(dispatch_deadline.replace("Z", "+00:00")).timestamp()
        _require(dispatch_deadline_unix > dispatch_start_unix, "dispatch wall bounds are invalid")
        _require(int(dispatch.get("wall_limit_seconds", 0)) == 7200, "dispatch wall limit changed")
        if not marker.exists():
            value = {
                "schema": "spatial_progress_gate.first_model_launch.v1",
                "status": "armed",
                "dispatch": dispatch_binding,
                "dispatch_start_utc": dispatch_start,
                "dispatch_deadline_utc": dispatch_deadline,
                "started_unix": dispatch_start_unix,
                "deadline_unix": dispatch_deadline_unix,
                "observed_preentry_unix": time.time(),
                "wall_limit_seconds": 7200,
                "amendment": self.bindings["wall_limit"],
            }
            try:
                _write_new(marker, value)
            except FileExistsError:
                pass
        value = json.loads(marker.read_text())
        _require(value.get("status") == "armed", "first-launch marker is not armed")
        _require(float(value.get("started_unix", 0)) > 0, "first-launch timestamp is missing")
        _require(value.get("dispatch") == dispatch_binding, "dispatch receipt changed")
        _require(int(value.get("wall_limit_seconds", 0)) == 7200, "first-launch wall limit changed")
        _require(float(value.get("deadline_unix", 0)) > float(value["started_unix"]), "first-launch deadline is invalid")
        self.wall_marker = _binding(marker)
        self.wall_started_unix = float(value["started_unix"])
        self.wall_deadline_unix = float(value["deadline_unix"])
        return {"value": value, "binding": self.wall_marker}

    def _check_wall(self) -> None:
        remaining = self.wall_deadline_unix - time.time()
        _require(remaining > 0, "shared package two-hour wall limit exhausted")
        self.counters["wall_remaining_seconds"] = remaining

    def _prepare_preentry(self) -> None:
        source_dir = self.output / "source"
        source_dir.mkdir(exist_ok=False)
        originals: dict[str, Path] = {
            "producer": Path(__file__),
            "native_input_identity": Path(_input_identity.__code__.co_filename),
            "source_provenance": Path(preserve_source.__code__.co_filename),
            "admission": ADMISSION,
            "budget": BUDGET,
            "wall_limit": WALL_LIMIT,
            "dispatch": DISPATCH,
            "profile": Path(self.admission["profile"]["path"]),
            "shared_panel": self.shared_panel_path,
            "panel": self.source_panel_path,
            "raw": Path(self.source_record["raw"]["path"]),
            "trace": Path(self.source_record["trace"]["path"]),
            "receipt": Path(self.source_record["receipt"]["path"]),
            "image": Path(self.source_record["image"]["path"]),
        }
        expected = {
            "shared_panel": self.admission["panel"], "panel": self.source_record["source_panel"], "raw": self.source_record["raw"],
            "trace": self.source_record["trace"], "receipt": self.source_record["receipt"],
            "image": self.source_record["image"],
        }
        snapshots: dict[str, Any] = {}
        for name, source in originals.items():
            _require(source.is_file(), f"source is missing: {source}")
            observed = _binding(source)
            if name in expected:
                _require(observed == expected[name], f"bound source changed: {name}")
            if name == "admission":
                _require(observed == self.bindings["admission"], "admission binding changed")
            if name == "budget":
                _require(observed == self.bindings["budget"], "budget binding changed")
            if name == "wall_limit":
                _require(observed == self.bindings["wall_limit"], "wall-limit binding changed")
            filename = f"{name}{source.suffix}" if source.suffix else name
            if source.suffix.lower() in {".py", ".md", ".sh"}:
                destination = preserve_source(source, run_root=self.output,
                                              relative_name=Path("source") / filename)
                captured = _binding(destination)
            else:
                destination = source_dir / filename
                captured = _copy_new(source, destination)
            snapshots[name] = {"original": observed, "snapshot": captured}
        wall_start = self._wall_start()
        snapshot = _write_new(self.output / "source-snapshot.json", {
            "schema": "spatial_progress_gate.source_snapshot.v1",
            "status": "frozen_before_model_load",
            "cell_id": self.cell["id"], "source_boundary_id": self.cell["source_boundary_id"],
            "boundary": {"model": self.cell["model"], "image_id": self.cell["image_id"], "prefix_end": self.cell["prefix_end"]},
            "wall_start": wall_start,
            "artifacts": snapshots,
        })
        self.source_snapshot = snapshot
        self.launch = _write_new(self.output / "launch.json", {
            "schema": "spatial_progress_gate.launch.v1",
            "status": "frozen_before_model_load",
            "mode": self.args.mode, "repair": bool(self.args.repair), "retry": int(self.args.retry),
            "cell_id": self.cell["id"], "boundary_id": self.cell["source_boundary_id"],
            "condition": self.cell["condition"], "model": self.cell["model"], "image_id": self.cell["image_id"],
            "device": str(self.device), "pid": os.getpid(), "argv": list(sys.argv),
            "frozen_inputs": {
                "admission": self.bindings["admission"], "budget_estimate": self.bindings["budget"],
                "wall_limit": self.bindings["wall_limit"],
                "dispatch": _binding(DISPATCH),
            },
            "budget_estimate_value": self.bindings["budget_value"],
            "wall_limit_value": self.bindings["wall_value"],
            "wall_start": wall_start, "source_snapshot": snapshot,
            "producer": _binding(Path(__file__)),
            "loader": _binding(Path(__file__).resolve().parents[1] / "untied_shared.py"),
            "source_loader": _binding(Path(__file__).resolve().parents[1] / "coordinate_continuity" / "runtime.py"),
            "candidate_digest": _digest(self.cell["candidate_sets"]),
            "policy": {"original_full_vocabulary": True, "candidate_renormalization": False, "beam_search": False, "model_change": False},
            "run_budget": {"model_forwards": int(self.args.max_forwards), "max_seconds": float(self.args.max_seconds), "package_wall_seconds": 7200},
        })

    def _install_counters(self) -> None:
        def model_forward(_module: Any, _inputs: tuple[Any, ...]) -> None:
            self.counters["model_forwards"] += 1
            _require(self.counters["model_forwards"] <= self.counters["max_model_forwards"], "model-forward budget exhausted")
            _require(time.monotonic() - self.started <= self.counters["max_seconds"], "wall-time budget exhausted")
            self._check_wall()
        self.handles.append(self.model.register_forward_pre_hook(model_forward))
        visual = getattr(getattr(self.model, "model", None), "visual", None)
        if visual is not None:
            self.handles.append(visual.register_forward_pre_hook(lambda *_: self.counters.__setitem__("vision_forwards", self.counters["vision_forwards"] + 1)))

    def _load_context(self) -> tuple[Any, list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
        source_panel = json.loads(self.source_panel_path.read_text())
        batch, raw, trace, _group, planning = _source(self.boundary, self.cell["model"], source_panel, self.q, self.device)
        target = int(self.boundary["batch_index"])
        native = [int(token) for token in self.boundary["native_tokens"]]
        _require(_digest(native) == self.source_record["native_token_sha256"], "native source token hash changed")
        return batch, raw, trace, {"planning": _jsonable(planning), "target": target, "native": native}

    def _score_condition(self, *, batch: Any, raw: list[dict[str, Any]], trace: dict[str, Any], context: dict[str, Any], condition: str, include_parity: bool) -> dict[str, Any]:
        native = context["native"]
        prefix, condition_spec = _condition_prefix(self.cell, native, condition)
        candidate_sets: dict[str, list[str]] = {"A": [], "N": []}
        rows: dict[str, dict[str, Any]] = {}
        expected_ids: set[str] = set()
        for set_name, candidate in _candidate_rows(self.cell):
            scored = _score_candidate(
                model=self.model, batch=batch, raw=raw, target=context["target"], prefix=prefix,
                tokens=[int(token) for token in candidate["tokens"]], pad=int(self.q.tokenizer.pad_token_id), device=self.device,
            )
            self.counters["candidate_rows"] += 1
            row = _row_record(candidate=candidate, scored=scored, condition=condition, prefix=prefix, common_description=self.cell["candidate_sets"].get("common_description_tokens", []))
            row["set"] = set_name
            rows[str(candidate["id"])] = row
            candidate_sets[set_name].append(str(candidate["id"]))
            expected_ids.add(str(candidate["id"]))
        groups: dict[str, Any] = {}
        for set_name, ids in candidate_sets.items():
            values = [float(rows[item]["row_sum_logprob"]) for item in ids]
            groups[set_name] = {"row_ids": ids, "row_count": len(ids), "row_logsumexp": _logsumexp(values)}
        result: dict[str, Any] = {
            "schema": "spatial_progress_gate.condition_scores.v1", "status": "candidate_complete",
            "cell_id": self.cell["id"], "boundary_id": self.cell["source_boundary_id"],
            "condition": condition, "model": self.cell["model"], "image_id": self.cell["image_id"],
            "prefix_end": int(self.cell["prefix_end"]), "history_prefix": {
                "token_ids": prefix, "token_sha256": _digest(prefix), "condition": condition_spec,
            },
            "candidate_sets": groups, "rows": rows,
            "probability_accounting": {
                "distribution": "original full vocabulary", "log_softmax": "full vocabulary at every row token",
                "includes_entry": True, "includes_terminator": True, "candidate_renormalization": False,
                "conditional_xy1": "x1 token then y1 token under the candidate's actual x1",
            },
        }
        _validate_scores(result, expected_ids)
        result["probability_accounting_falsification"] = _falsification(result)
        _require(result["probability_accounting_falsification"]["passed"], "probability accounting falsification failed")
        if include_parity:
            parity = _trace_parity(
                model=self.model, batch=batch, raw=raw, target=context["target"], native=native,
                prefix_end=int(self.cell["prefix_end"]), trace=trace,
                pad=int(self.q.tokenizer.pad_token_id), device=self.device,
            )
            result["native_trace_parity"] = parity
            _require(parity["passed"] and parity["max_full_vocab_logprob_abs_error"] <= ATOL, "native source trace parity failed")
        else:
            result["native_trace_parity"] = {"status": "not_required_for_changed_history"}
        result["native_alias"] = {"condition_id": "x1_after_N786", "alias_of": "native", "executed_once": condition == "native"}
        result["source"] = {
            "panel": self.source_record["source_panel"], "raw": self.source_record["raw"],
            "trace": self.source_record["trace"], "receipt": self.source_record["receipt"], "image": self.source_record["image"],
        }
        return result

    def _run_loaded(self) -> dict[str, Any]:
        batch, raw, trace, context = self._load_context()
        if self.args.mode == "qualify":
            native = self._score_condition(batch=batch, raw=raw, trace=trace, context=context, condition="native", include_parity=True)
            changed = self._score_condition(batch=batch, raw=raw, trace=trace, context=context, condition="x1_before_N786", include_parity=False)
            result: dict[str, Any] = {
                "schema": "spatial_progress_gate.qualification.v1", "status": "candidate_complete",
                "passed": True, "campaign": "qualification-repair-01" if self.args.repair else "qualification-01",
                "cell_id": self.cell["id"], "boundary_id": self.cell["source_boundary_id"], "model": self.cell["model"],
                "source_input_identity": _input_identity(batch), "conditions": {"native": native, "x1_before_N786": changed},
                "checks": {
                    "native_trace_parity": native["native_trace_parity"],
                    "changed_prefix_offsets": changed["history_prefix"]["condition"]["changed_prefix_check"],
                    "candidate_identity": changed["candidate_sets"],
                    "probability_accounting": changed["probability_accounting_falsification"],
                },
            }
            result["passed"] = bool(
                native["native_trace_parity"]["passed"]
                and native["native_trace_parity"]["max_full_vocab_logprob_abs_error"] <= ATOL
                and changed["history_prefix"]["condition"]["changed_prefix_check"]["passed"]
                and changed["probability_accounting_falsification"]["passed"]
            )
            result["status"] = "candidate_complete" if result["passed"] else "technical_HOLD"
            return result
        return self._score_condition(batch=batch, raw=raw, trace=trace, context=context, condition=self.cell["condition"], include_parity=self.cell["condition"] == "native")

    def _artifact_bytes(self) -> int:
        return sum(path.stat().st_size for path in self.output.rglob("*") if path.is_file())

    def run(self) -> dict[str, Any]:
        status = "candidate_complete"
        result: dict[str, Any] | None = None
        error: str | None = None
        try:
            self._check_wall()
            self.q, self.identity = load_model(self.cell["model"], self.device)
            self.model = self.q.model.eval()
            self._install_counters()
            initial_versions = {name: parameter._version for name, parameter in self.model.named_parameters()}
            result = self._run_loaded()
            if result.get("status") != "candidate_complete" or (self.args.mode == "qualify" and not result.get("passed")):
                status = "technical_HOLD"
            output_name = "result.json" if self.args.mode == "qualify" else "scores.json"
            _write_new(self.output / output_name, result)
            versions = {name: parameter._version for name, parameter in self.model.named_parameters()}
            _require(versions == initial_versions, "parameter mutation detected")
        except BaseException as exc:
            status = "technical_invalid"
            error = repr(exc)
        finally:
            for handle in self.handles:
                handle.remove()
        self.counters["elapsed_seconds"] = time.monotonic() - self.started
        self.counters["gpu_seconds"] = self.counters["elapsed_seconds"] if self.device.type == "cuda" else 0.0
        self.counters["artifact_bytes_excluding_receipt"] = self._artifact_bytes()
        receipt: dict[str, Any] = {
            "schema": "spatial_progress_gate.receipt.v1", "status": status, "mode": self.args.mode,
            "campaign": "qualification-repair-01" if self.args.mode == "qualify" and self.args.repair else ("qualification-01" if self.args.mode == "qualify" else "scientific"),
            "cell_id": self.cell["id"], "boundary_id": self.cell["source_boundary_id"], "condition": self.cell["condition"],
            "model": self.cell["model"], "device": str(self.device), "launch": self.launch,
            "source_snapshot": self.source_snapshot, "identity": _jsonable(self.identity), "counters": self.counters,
            "optional_release": "not_run",
        }
        if result is not None:
            receipt["result"] = _binding(self.output / ("result.json" if self.args.mode == "qualify" else "scores.json"))
        if error is not None:
            receipt["error"] = error
        receipt_binding = _write_new(self.output / "receipt.json", receipt)
        receipt["receipt_binding"] = receipt_binding
        return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("qualify", "cell"), required=True)
    parser.add_argument("--cell-id")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repair", action="store_true")
    parser.add_argument("--retry", type=int, default=0)
    parser.add_argument("--max-forwards", type=int, default=128)
    parser.add_argument("--max-seconds", type=float, default=3600.0)
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        from probes.training_set_completion.spatial_progress_gate.selfcheck import run_selfcheck
        run_selfcheck()
        return
    if args.mode == "qualify" and args.repair is False and args.retry:
        parser.error("--retry is valid only for cell mode")
    if args.mode == "cell" and args.repair:
        parser.error("--repair is valid only for qualification")
    admission, _ = _load_admission()
    cells = _cells(admission)
    if args.mode == "cell":
        if not args.cell_id:
            parser.error("--cell-id is required in cell mode")
        matches = [item for item in cells if item["id"] == args.cell_id]
        if len(matches) != 1:
            parser.error(f"unknown cell ID: {args.cell_id}")
        selected = matches[0]
    else:
        selected = None
    runtime = Runtime(args, selected)
    receipt = runtime.run()
    print(json.dumps({"status": receipt["status"], "cell_id": receipt["cell_id"], "receipt": str(runtime.output / "receipt.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
