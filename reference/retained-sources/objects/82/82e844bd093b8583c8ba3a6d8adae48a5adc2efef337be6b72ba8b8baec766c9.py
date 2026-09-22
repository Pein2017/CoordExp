"""History-cut crossover continuation runtime.

This is the small delta from the accepted readout-component runtime.  It
reopens a frozen source group, substitutes one exact target history prefix,
and delegates native batching, generation, readout tracing, stopping, and
artifact persistence to the qualified component runtime.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch

from probes.training_set_completion.numerical_feedback.select import rows as parsed_rows
from probes.training_set_completion.readout_component import runtime as component


EOS = component.EOS
RELEASE_ROWS = component.RELEASE_ROWS
RELEASE_TOKENS = component.RELEASE_TOKENS
FUTURE_POLICIES = ("original", "full")
PLAN_STATUS = "frozen_before_gpu"


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def token_hash(tokens: list[int]) -> str:
    return canonical_hash(tokens)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _binding(path: Path) -> dict[str, Any]:
    return component._binding(path)


def _sha_binding(path_value: Any, label: str) -> dict[str, Any]:
    if not isinstance(path_value, dict) or not path_value.get("path"):
        raise ValueError(f"{label} has no path binding")
    path = Path(path_value["path"])
    if not path.exists():
        raise FileNotFoundError(path)
    actual = _binding(path)
    for key in ("path", "sha256"):
        if path_value.get(key) is not None and actual[key] != path_value[key]:
            raise ValueError(f"{label} binding changed: {path}")
    return actual


def _shift_row(row: dict[str, Any], offset: int) -> dict[str, Any]:
    shifted = copy.deepcopy(row)
    shifted["start"] = int(row["start"]) + offset
    shifted["end"] = int(row["end"]) + offset
    shifted["coordinate_offsets"] = [int(x) + offset for x in row["coordinate_offsets"]]
    return shifted


def _history_prefix_metadata(history: dict[str, Any]) -> dict[str, Any]:
    boundary = history["boundary"]
    source_end = int(boundary["source_row"]["end"])
    prefix = list(history["prefix_tokens"])
    added = list(history["added_tokens"])
    return {
        "schema": "history_readout.history_binding.v1",
        "history_id": history["id"],
        "source_boundary_id": boundary["id"],
        "original_boundary_id": boundary["id"],
        "history_policy": history["history_policy"],
        "future_policy": None,
        "cut_rows": int(history["cut"]),
        "source_prefix_token_count": source_end,
        "history_prefix_token_count": len(prefix),
        "history_prefix_tokens": prefix,
        "history_prefix_sha256": history["prefix_sha256"],
        "added_token_count": len(added),
        "added_tokens": added,
        "added_sha256": history["added_sha256"],
        "supplied_rows": copy.deepcopy(history["supplied_rows"]),
        "saved_overlap_token_count": len(history["saved_overlap_tokens"]),
        "saved_overlap_tokens": list(history["saved_overlap_tokens"]),
        "saved_overlap_sha256": token_hash(list(history["saved_overlap_tokens"])),
        "source_release": copy.deepcopy(history["source"]),
        "source_release_receipt": copy.deepcopy(history["source_receipt"]),
        "companion_semantics": {
            "batch_membership": "same native source-order group",
            "prefix_width": len(prefix),
            "target": "replace target row with exact frozen history prefix",
            "short_companions": "EOS then pad at the matching prefix width",
            "companion_outputs": "uninterpretable and not in estimand",
        },
    }


def validate_plan_cpu(plan_path: Path, panel_path: Path) -> dict[str, Any]:
    """Validate all frozen history/source bindings without loading a model."""
    plan = json.loads(plan_path.read_text())
    panel = json.loads(panel_path.read_text())
    if plan.get("status") != PLAN_STATUS:
        raise ValueError(f"history plan is not frozen: {plan.get('status')!r}")
    histories = plan.get("histories") or []
    cells = plan.get("cells") or []
    unsupported = plan.get("unsupported") or []
    if unsupported:
        raise ValueError(f"plan contains unsupported histories: {len(unsupported)}")
    if len(histories) != 44 or len(cells) != 88:
        raise ValueError(f"expected 44 histories/88 cells, got {len(histories)}/{len(cells)}")
    if len({h["id"] for h in histories}) != len(histories):
        raise ValueError("history IDs are not unique")
    if len({c["id"] for c in cells}) != len(cells):
        raise ValueError("cell IDs are not unique")
    groups = {group["key"]: group for group in panel["groups"]}
    seen_cell_pairs: set[tuple[str, str]] = set()
    source_release_count = 0
    companion_count = 0
    for history in histories:
        boundary = history["boundary"]
        source_end = int(boundary["source_row"]["end"])
        native = list(boundary["native_tokens"])
        prefix = list(history["prefix_tokens"])
        added = list(history["added_tokens"])
        if len(prefix) != source_end + len(added):
            raise ValueError(f"history prefix length mismatch: {history['id']}")
        if prefix[:source_end] != native[:source_end] or prefix[source_end:] != added:
            raise ValueError(f"history prefix token identity mismatch: {history['id']}")
        if token_hash(prefix) != history["prefix_sha256"]:
            raise ValueError(f"history prefix hash mismatch: {history['id']}")
        if token_hash(added) != history["added_sha256"]:
            raise ValueError(f"history added hash mismatch: {history['id']}")
        supplied = parsed_rows(added)
        if supplied != history["supplied_rows"]:
            raise ValueError(f"supplied row parse mismatch: {history['id']}")
        if len(supplied) != int(history["cut"]):
            raise ValueError(f"history cut mismatch: {history['id']}")
        if EOS in added:
            raise ValueError(f"EOS occurs before frozen history cut: {history['id']}")

        release_path = Path(history["source"]["path"])
        receipt_path = Path(history["source_receipt"]["path"])
        _sha_binding(history["source"], f"source release {history['id']}")
        _sha_binding(history["source_receipt"], f"source receipt {history['id']}")
        release = json.loads(release_path.read_text())
        receipt = json.loads(receipt_path.read_text())
        expected = {
            "status": "candidate_complete",
            "boundary_id": boundary["id"],
            "group": boundary["group"],
            "model": boundary["model"],
            "policy": history["history_policy"],
        }
        for key, value in expected.items():
            if release.get(key) != value:
                raise ValueError(f"source release {history['id']} has {key}={release.get(key)!r}")
        if receipt.get("status") != "candidate_complete" or receipt.get("job_id") != release["job_id"]:
            raise ValueError(f"source receipt is not the accepted cell: {history['id']}")
        old_target = list(release["target"]["token_ids"])
        if old_target[: len(added)] != added:
            raise ValueError(f"source release does not contain frozen history: {history['id']}")
        if list(history["saved_overlap_tokens"]) != old_target[len(added) :]:
            raise ValueError(f"saved overlap mismatch: {history['id']}")
        source_release_count += 1

        for key in ("raw_path", "trace_path", "receipt_path"):
            path = Path(boundary[key])
            if not path.exists():
                raise FileNotFoundError(path)
        raw = json.loads(Path(boundary["raw_path"]).read_text())["rows"]
        group = groups.get(boundary["group"])
        if group is None:
            raise ValueError(f"source group missing from panel: {boundary['group']}")
        if len(raw) != len(group["cases"]):
            raise ValueError(f"companion count changed: {history['id']}")
        target_index = int(boundary["batch_index"])
        if list(raw[target_index]["token_ids"]) != native:
            raise ValueError(f"source native token identity changed: {history['id']}")
        if token_hash(native) != boundary["native_token_hash"]:
            raise ValueError(f"source native hash mismatch: {history['id']}")
        # Check the exact width rule for every source-order companion.  The
        # runtime adds EOS and pad only for rows shorter than this width.
        width = len(prefix)
        for row in raw:
            tokens = list(row["token_ids"][:width])
            if len(tokens) < width:
                if EOS not in tokens:
                    tokens.append(EOS)
                else:
                    tokens = tokens[: tokens.index(EOS) + 1]
                tokens.extend([None] * (width - len(tokens)))
            if len(tokens) != width:
                raise ValueError(f"companion width reconstruction failed: {history['id']}")
        companion_count += len(raw)

    by_id = {h["id"]: h for h in histories}
    for cell in cells:
        history = by_id.get(cell.get("history_id"))
        if history is None:
            raise ValueError(f"cell references missing history: {cell.get('id')}")
        if cell.get("model") != history["boundary"]["model"]:
            raise ValueError(f"cell/history model mismatch: {cell['id']}")
        if cell.get("policy") not in FUTURE_POLICIES:
            raise ValueError(f"unsupported future policy: {cell['id']}")
        pair = (history["id"], cell["policy"])
        if pair in seen_cell_pairs:
            raise ValueError(f"duplicate history/future cell: {pair}")
        seen_cell_pairs.add(pair)
    if len(seen_cell_pairs) != 88:
        raise ValueError("history/future policy coverage is incomplete")
    return {
        "schema": "history_readout.plan_validation.v1",
        "status": "pass",
        "plan": _binding(plan_path),
        "panel": _binding(panel_path),
        "histories": len(histories),
        "cells": len(cells),
        "source_releases": source_release_count,
        "companion_rows_checked": companion_count,
        "future_policies": list(FUTURE_POLICIES),
        "eos_id": EOS,
        "notes": [
            "target history prefixes are exact source-prefix plus saved rows",
            "source-order companions are width-matched with EOS then pad",
            "no model or GPU was loaded",
        ],
    }


class Runtime(component.Runtime):
    """Component runtime with exact history-prefix substitution."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.history_plan_path = args.plan
        self.history_plan = json.loads(args.plan.read_text())
        self.histories = {item["id"]: item for item in self.history_plan.get("histories", [])}
        if self.history_plan.get("status") != PLAN_STATUS:
            raise ValueError("history execution plan must be frozen before GPU execution")
        super().__init__(args)
        # The inherited component code writes the equivalent receipt schema;
        # bind this producer and expose the history delta in the shard receipt.
        self.ledger["schema"] = "history_readout.runtime_receipt.v1"
        self.ledger["producer"] = _binding(Path(__file__))
        self.ledger["history_plan"] = _binding(self.history_plan_path)
        self.ledger["history_ids"] = [job["history_id"] for job in self.jobs]
        self.persist()

    def _normalise_jobs(self, plan: dict[str, Any]) -> list[dict[str, Any]]:
        planned = list(plan.get("cells") or plan.get("jobs") or [])
        requested = set(self.args.cell_ids or [])
        if requested:
            planned = [item for item in planned if item.get("id") in requested]
            present = {item.get("id") for item in planned}
            if present != requested:
                raise ValueError(f"requested cell IDs are absent from plan: {sorted(requested - present)}")
        if not planned:
            raise ValueError("history execution plan has no selected cells")
        jobs: list[dict[str, Any]] = []
        seen: set[str] = set()
        for raw_cell in planned:
            cell = dict(raw_cell)
            cell_id = str(cell.get("id", ""))
            history_id = str(cell.get("history_id", ""))
            history = self.histories.get(history_id)
            if not cell_id or cell_id in seen:
                raise ValueError("history cell IDs must be nonempty and unique")
            if history is None:
                raise ValueError(f"cell {cell_id} references no frozen history")
            future = str(cell.get("policy", ""))
            boundary_original = copy.deepcopy(history["boundary"])
            if future not in FUTURE_POLICIES:
                raise ValueError(f"unsupported future policy: {future}")
            if cell.get("model") != boundary_original["model"]:
                raise ValueError(f"cell/history model mismatch: {cell_id}")
            prefix = list(history["prefix_tokens"])
            source_end = int(boundary_original["source_row"]["end"])
            if prefix[:source_end] != list(boundary_original["native_tokens"])[:source_end]:
                raise ValueError(f"history/source prefix mismatch: {history_id}")
            # Treat the supplied final row as the current native row.  The
            # inherited runtime only needs its exclusive end; the shifted row
            # keeps row/coordinate metadata honest in the release artifact.
            supplied_rows = history["supplied_rows"]
            if not supplied_rows:
                raise ValueError(f"history has no supplied rows: {history_id}")
            boundary = boundary_original
            boundary["_original_boundary"] = boundary_original
            boundary["_history_id"] = history_id
            boundary["_history_policy"] = history["history_policy"]
            boundary["_history_cut"] = int(history["cut"])
            boundary["_history_prefix"] = prefix
            boundary["_history_added"] = list(history["added_tokens"])
            boundary["native_tokens"] = prefix
            boundary["native_token_hash"] = history["prefix_sha256"]
            boundary["prefix_hash"] = history["prefix_sha256"]
            boundary["source_row"] = _shift_row(supplied_rows[-1], source_end)
            boundary["source_row"]["index"] = int(history["cut"]) - 1
            boundary["previous_row"] = (
                _shift_row(supplied_rows[-2], source_end) if len(supplied_rows) > 1 else boundary_original["source_row"]
            )
            boundary["target_slots"] = [
                _shift_row(row, source_end) for row in supplied_rows
            ]
            boundary["fixed_suffix_end"] = len(prefix)
            job = {
                "id": cell_id,
                "boundary": boundary,
                "operator": future,
                "policy": future,
                "qualification": bool(cell.get("qualification", False)),
                "history_id": history_id,
                "history": history,
                "model": cell.get("model"),
                "gpu": cell.get("gpu"),
            }
            jobs.append(job)
            seen.add(cell_id)
        return jobs

    def _source(self, boundary: dict[str, Any]):
        original = boundary["_original_boundary"]
        group, raw, trace, receipt, batch = super()._source(original)
        history = self.histories[boundary["_history_id"]]
        target_index = int(original["batch_index"])
        prefix = list(history["prefix_tokens"])
        source_end = int(original["source_row"]["end"])
        if prefix[:source_end] != list(original["native_tokens"])[:source_end]:
            raise ValueError("history source prefix changed before native replay")
        if prefix[source_end:] != list(history["added_tokens"]):
            raise ValueError("history added tokens changed before native replay")
        replay_raw = [dict(row) for row in raw]
        replay_raw[target_index] = dict(replay_raw[target_index])
        replay_raw[target_index]["token_ids"] = prefix
        # The inherited full_prefix builder cuts every companion at the same
        # width and supplies EOS then pad to shorter rows.  Only the target row
        # is replaced; image/mask/group identity stays from the native batch.
        return group, replay_raw, trace, receipt, batch

    def _history_meta(self, job: dict[str, Any]) -> dict[str, Any]:
        meta = _history_prefix_metadata(job["history"])
        meta["future_policy"] = job["operator"]
        meta["source_paths"] = {
            key: _binding(Path(job["boundary"][f"{key}_path"]))
            for key in ("raw", "trace", "receipt")
        }
        meta["replay_boundary"] = {
            "batch_index": int(job["boundary"]["batch_index"]),
            "source_end_exclusive": len(job["boundary"]["native_tokens"]),
            "model": job["boundary"]["model"],
            "group": job["boundary"]["group"],
            "image_id": int(job["boundary"]["image_id"]),
        }
        return meta

    def _run_job(self, job: dict[str, Any]) -> dict[str, Any]:
        payload = super()._run_job(job)
        job_dir = self.output / component._safe_name(str(job["id"]))
        result_path = job_dir / "release.json"
        receipt_path = job_dir / "receipt.json"
        meta = self._history_meta(job)
        payload = json.loads(result_path.read_text())
        payload["history"] = meta
        payload["history_source"] = {
            "original_boundary": copy.deepcopy(job["history"]["boundary"]),
            "history_prefix": list(job["history"]["prefix_tokens"]),
            "future_policy": job["operator"],
        }
        write_json(result_path, payload)
        receipt = json.loads(receipt_path.read_text())
        receipt["schema"] = "history_readout.cell_receipt.v1"
        receipt["producer"] = _binding(Path(__file__))
        receipt["history"] = meta
        receipt["release"] = _binding(result_path)
        write_json(receipt_path, receipt)
        for item in reversed(self.ledger["job_results"]):
            if item.get("job_id") == job["id"]:
                item["release"] = _binding(result_path)
                item["history_id"] = job["history_id"]
                item["history_policy"] = job["history"]["history_policy"]
                item["future_policy"] = job["operator"]
                break
        self.persist()
        return payload


def selfcheck() -> None:
    generator = torch.Generator().manual_seed(7)
    z = torch.randn(3, 1000, generator=generator, dtype=torch.float64)
    h = torch.randn(3, 8, generator=generator, dtype=torch.float64)
    w = torch.randn(1000, 8, generator=generator, dtype=torch.float64)
    alpha = w.norm(dim=1).median() / w.norm(dim=1)
    b = h @ w.mean(dim=0)
    full = z * alpha
    shared = z + (alpha - 1)[None, :] * b[:, None]
    centered = z + (alpha - 1)[None, :] * (z - b[:, None])
    assert torch.allclose(full - z, (shared - z) + (centered - z), atol=1e-12, rtol=0)
    assert parsed_rows([151646, 9, 151647, 151648, 151670, 151671, 151672, 151673, 151649])
    print("PASS history runtime formulas, decomposition, and complete-row parser")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["run", "validate-plan"], nargs="?")
    parser.add_argument("--panel", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cell-id", action="append", dest="cell_ids")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-forwards", type=int, default=100000)
    parser.add_argument("--max-seconds", type=float, default=8 * 3600)
    parser.add_argument("--max-bytes", type=int, default=16 * 1024**3)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--validation-output", type=Path)
    args = parser.parse_args()
    if args.self_check:
        selfcheck()
        return
    if args.mode == "validate-plan":
        if args.panel is None or args.plan is None:
            parser.error("validate-plan requires --panel and --plan")
        result = validate_plan_cpu(args.plan, args.panel)
        if args.validation_output:
            write_json(args.validation_output, result)
        print(json.dumps(result, sort_keys=True))
        return
    if args.mode != "run" or args.panel is None or args.plan is None or args.output is None:
        parser.error("run requires --panel, --plan, and --output")
    runtime = Runtime.__new__(Runtime)
    try:
        runtime.__init__(args)
        runtime.run()
    except BaseException as exc:
        if hasattr(runtime, "ledger"):
            runtime.close(exc)
        raise
    else:
        runtime.close()


if __name__ == "__main__":
    main()
