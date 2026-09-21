"""Fresh multimodal scoring for the frozen visual-instance-binding cells."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
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
    _row_role,
    _score_candidate,
    _trace_compare,
)
from probes.training_set_completion.readout_norm_fresh import _input_identity
from probes.training_set_completion.untied_shared import load_model
from src.inference.bound_requests import build_bound_native_requests
from src.qwen.native import prepare_native_inputs

from .compositor import binding, clean_copy, compose


ROOT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-21-visual-instance-binding")
ADMISSION = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-21-spatial-progress-gate/selection/shared-admission.json")
ADMISSION_BINDING = ROOT / "selection" / "shared-admission-binding.json"
BUDGET = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-21-spatial-progress-gate/selection/budget-estimate.json")
WALL_LIMIT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-21-spatial-progress-gate/selection/wall-limit-amendment.json")
WALL_START = ROOT / "wall-start.json"
PANEL = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-distribution-census/shared-panel.json")
ATOL = 2e-4
ROW_OPEN, ROW_END = 151646, 151649


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_once(path: Path, value: Any) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return binding(path)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_admission() -> dict[str, Any]:
    require(ADMISSION.is_file(), f"missing shared admission: {ADMISSION}")
    require(ADMISSION_BINDING.is_file(), f"missing admission binding: {ADMISSION_BINDING}")
    receipt = json.loads(ADMISSION_BINDING.read_text())
    require(receipt.get("status") == "frozen", "visual admission binding is not frozen")
    require(receipt.get("shared_admission") == binding(ADMISSION), "shared admission hash changed")
    admission = json.loads(ADMISSION.read_text())
    require(admission.get("status") == "frozen_before_intervention_scores", "shared admission is not frozen")
    require(len(admission.get("lane_b", {}).get("states", [])) == 4, "Lane B state count changed")
    return admission


def _find_boundary(panel: dict[str, Any], ident: str) -> dict[str, Any]:
    all_boundaries: list[dict[str, Any]] = []
    for key in ("boundaries", "existing_boundaries", "new_boundaries", "all_boundaries"):
        values = panel.get(key, [])
        if isinstance(values, list):
            all_boundaries.extend(item for item in values if isinstance(item, dict))
    by_id = {str(item.get("id", item.get("boundary_id"))): item for item in all_boundaries if item.get("id", item.get("boundary_id")) is not None}
    matches = [by_id[ident]] if ident in by_id else []
    require(len(matches) == 1, f"source boundary is not unique: {ident}")
    return matches[0]


def _source_record(admission: dict[str, Any], boundary_id: str) -> dict[str, Any]:
    rows = [item for item in admission["source_pool"] if item.get("source_boundary_id") == boundary_id]
    require(len(rows) == 1, f"source record is not unique: {boundary_id}")
    record = rows[0]
    for key in ("source_panel", "raw", "trace", "receipt", "image"):
        require(binding(Path(record[key]["path"])) == record[key], f"source binding changed: {boundary_id}/{key}")
    return record


def _state(admission: dict[str, Any], state_id: str) -> dict[str, Any]:
    rows = [item for item in admission["lane_b"]["states"] if item.get("id") == state_id]
    require(len(rows) == 1, f"state is not unique: {state_id}")
    state = copy.deepcopy(rows[0])
    require(state.get("status", "ready") == "ready", f"state is not ready: {state_id}")
    for set_name in ("A", "N"):
        candidates = state["candidate_sets"].get(set_name, [])
        require(isinstance(candidates, list) and candidates, f"empty frozen candidate set: {state_id}/{set_name}")
        for candidate in candidates:
            require(isinstance(candidate.get("tokens"), list), f"candidate tokens missing: {candidate.get('id')}")
            require(candidate["tokens"][0] == ROW_OPEN and candidate["tokens"][-1] == ROW_END, f"incomplete candidate row: {candidate.get('id')}")
    return state


def _row_tokens(native: list[int], prefix_end: int) -> list[int]:
    tail = native[prefix_end:]
    require(tail and tail[0] == ROW_OPEN, "native actual row is not at prefix boundary")
    end = next((index for index, token in enumerate(tail[1:], 1) if token == ROW_OPEN), len(tail))
    row = tail[:end]
    require(row[-1] == ROW_END, "native actual row is incomplete")
    return row


def _candidate_sets(state: dict[str, Any], native: list[int]) -> tuple[list[dict[str, Any]], dict[str, str], str, list[int]]:
    rows: list[dict[str, Any]] = []
    sets: dict[str, str] = {}
    for set_name in ("A", "N"):
        for candidate in state["candidate_sets"][set_name]:
            item = copy.deepcopy(candidate)
            ident = str(item["id"])
            require(ident not in sets, f"candidate ID duplicated: {ident}")
            rows.append(item)
            sets[ident] = set_name
    actual_id = str(state["actual_row"]["candidate_id"])
    by_id = {str(item["id"]): item for item in rows}
    actual = by_id.get(actual_id)
    if actual is None:
        actual = {"id": actual_id, "owner": state["actual_row"].get("owner"), "tokens": _row_tokens(native, int(state["prefix_end"])), "source": "native_actual_row", "construction_rule": "source native row at frozen boundary"}
        rows.append(actual)
        sets[actual_id] = "actual"
    require(list(native[int(state["prefix_end"]):int(state["prefix_end"]) + len(actual["tokens"])]) == [int(token) for token in actual["tokens"]], f"actual row does not bind native source: {state['id']}")
    return rows, sets, actual_id, [int(token) for token in actual["tokens"]]


def _condition_record(score: dict[str, Any]) -> dict[str, Any]:
    token_logprobs = score["token_logprobs"]
    require(len(token_logprobs) >= 6, "candidate row lacks x1/y1 path")
    score["x1_logprob"] = float(token_logprobs[4])
    score["y1_given_x1_logprob"] = float(token_logprobs[5])
    score["x1_y1_conditional_logprob"] = float(token_logprobs[4] + token_logprobs[5])
    return score


class Runtime:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.started = time.monotonic()
        self.admission = _load_admission()
        require(BUDGET.is_file(), f"budget estimate is missing: {BUDGET}")
        self.state = _state(self.admission, args.state_id)
        self.panel_path = PANEL
        require(binding(self.panel_path) == self.admission["panel"], "shared panel binding changed")
        self.panel = json.loads(self.panel_path.read_text())
        self.boundary = _find_boundary(self.panel, self.state["source_boundary_id"])
        self.source_record = _source_record(self.admission, self.state["source_boundary_id"])
        self.source_panel_path = Path(self.source_record["source_panel"]["path"])
        self.source_panel = json.loads(self.source_panel_path.read_text())
        self.output = args.output.resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        self.counters: dict[str, Any] = {"model_forwards": 0, "vision_forwards": 0, "candidate_rows": 0, "retained_tensor_bytes": 0, "max_model_forwards": int(args.max_forwards), "max_seconds": float(args.max_seconds)}
        require(WALL_LIMIT.is_file(), f"wall-limit amendment is missing: {WALL_LIMIT}")
        self.wall_start = self._wall_start()
        self._snapshot_sources()
        self.images = self._prepare_images()
        self.device = torch.device(args.device)
        self.model: Any = None
        self.handles: list[Any] = []

    def _snapshot_sources(self) -> None:
        sources = [Path(__file__), Path(_score_candidate.__code__.co_filename), Path(_source.__code__.co_filename), Path(load_model.__code__.co_filename), Path(build_bound_native_requests.__code__.co_filename), Path(prepare_native_inputs.__code__.co_filename), ADMISSION, ADMISSION_BINDING, BUDGET, WALL_LIMIT, self.panel_path, self.source_panel_path, Path(self.source_record["raw"]["path"]), Path(self.source_record["trace"]["path"]), Path(self.source_record["receipt"]["path"]), Path(self.source_record["image"]["path"])]
        snap_dir = self.output / "source-snapshot"
        snap_dir.mkdir()
        entries = []
        for index, source in enumerate(dict.fromkeys(path.resolve() for path in sources)):
            require(source.is_file(), f"source snapshot missing: {source}")
            target = snap_dir / f"{index:02d}-{source.name}"
            shutil.copyfile(source, target)
            entries.append({"source": binding(source), "snapshot": binding(target)})
        self.snapshot = write_once(self.output / "source-snapshot.json", {"schema": "visual_instance_binding.source_snapshot.v1", "status": "frozen_before_model_entry", "entries": entries})
        self.launch = write_once(self.output / "launch.json", {"schema": "visual_instance_binding.launch.v1", "status": "frozen_before_model_entry", "campaign": self.args.campaign, "mode": self.args.mode, "state_id": self.state["id"], "model": self.state["model"], "device": self.args.device, "producer": binding(Path(__file__)), "admission": binding(ADMISSION), "admission_binding": binding(ADMISSION_BINDING), "budget_estimate": binding(BUDGET), "wall_limit_amendment": binding(WALL_LIMIT), "wall_start": self.wall_start, "panel": binding(self.panel_path), "source_panel": binding(self.source_panel_path), "source_snapshot": self.snapshot, "policy": {"fresh_visual_encoding": True, "fresh_text_prefix": True, "full_vocabulary": True, "includes_entry": True, "includes_terminator": True, "candidate_renormalization": False}})

    def _wall_start(self) -> dict[str, Any]:
        WALL_START.parent.mkdir(parents=True, exist_ok=True)
        if WALL_START.exists():
            value = json.loads(WALL_START.read_text())
            require(value.get("wall_limit_seconds") == 7200, "wall start limit changed")
            require(value.get("wall_limit_amendment") == binding(WALL_LIMIT), "wall start amendment changed")
            return {"binding": binding(WALL_START), **value}
        value = {"schema": "visual_instance_binding.wall_start.v1", "wall_limit_seconds": 7200, "wall_limit_amendment": binding(WALL_LIMIT), "started_epoch": time.time(), "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        receipt = write_once(WALL_START, value)
        return {"binding": receipt, **value}

    def _prepare_images(self) -> dict[str, dict[str, Any]]:
        source = Path(self.source_record["image"]["path"])
        image_root = self.output / "images"
        result = {"clean": clean_copy(source, image_root / "clean")}
        for name in ("A", "N", "unrelated"):
            result[f"ablate_{name}"] = compose(source, self.state["masks"][name], image_root / f"ablate_{name}", name=f"ablate_{name}")
        return result

    def _install_counters(self) -> None:
        def model_forward(_module: Any, _inputs: tuple[Any, ...]) -> None:
            self.counters["model_forwards"] += 1
            if self.counters["model_forwards"] > self.counters["max_model_forwards"]:
                raise RuntimeError("model-forward budget exhausted")
            if time.monotonic() - self.started > self.counters["max_seconds"]:
                raise RuntimeError("wall-time budget exhausted")
            if time.time() - float(self.wall_start["started_epoch"]) >= float(self.wall_start["wall_limit_seconds"]):
                raise RuntimeError("package wall-time ceiling reached")

        self.handles.append(self.model.register_forward_pre_hook(model_forward))
        visual = getattr(getattr(self.model, "model", None), "visual", None)
        if visual is not None:
            self.handles.append(visual.register_forward_pre_hook(lambda *_: self.counters.__setitem__("vision_forwards", self.counters["vision_forwards"] + 1)))

    def _variant_batch(self, q: Any, group: Any, condition: str) -> Any:
        cases = copy.deepcopy(group["cases"])
        target = int(self.boundary["batch_index"])
        image_path = Path(self.images[condition]["image"]["path"])
        target_case = cases[target]
        plan = copy.deepcopy(target_case["image_plan"])
        plan["image_content_sha256"] = self.images[condition]["image"]["sha256"]
        target_case["image_path"] = str(image_path)
        target_case["image_plan"] = plan
        config = dict(self.source_panel["configs"][self.state["model"]])
        config["data"] = {"input_jsonl": group["input_jsonl"]}
        requests, _ = build_bound_native_requests(q, config, cases)
        return prepare_native_inputs(q.processor, requests, device=self.device, record_media_identity=True)

    def _score_batch(self, batch: Any, raw: list[dict[str, Any]], candidates: list[dict[str, Any]], sets: dict[str, str], actual_id: str, prefix: list[int], target: int, condition: str) -> dict[str, Any]:
        rows: dict[str, dict[str, Any]] = {}
        tensors: dict[str, torch.Tensor] = {}
        for candidate in candidates:
            ident = str(candidate["id"])
            scored = _score_candidate(model=self.model, batch=batch, raw=raw, target=target, prefix=prefix, tokens=[int(token) for token in candidate["tokens"]], pad=int(self.q.tokenizer.pad_token_id), device=self.device)
            self.counters["candidate_rows"] += 1
            tensors[ident] = scored["action_logits"].detach().float().cpu()
            boundary_logits = scored.pop("boundary_logits")
            scored.pop("action_logits")
            scored.update({"candidate_id": ident, "owner": candidate.get("owner"), "set": sets.get(ident, "actual"), "source": candidate.get("source"), "construction_rule": candidate.get("construction_rule"), "condition": condition})
            rows[ident] = _condition_record(scored)
        groups: dict[str, Any] = {}
        for set_name in ("A", "N"):
            ids = [str(candidate["id"]) for candidate in candidates if sets.get(str(candidate["id"])) == set_name]
            values = [float(rows[ident]["row_sum_logprob"]) for ident in ids]
            groups[set_name] = {"row_ids": ids, "row_count": len(ids), "row_logsumexp": None if not values else float(torch.logsumexp(torch.tensor(values, dtype=torch.float64), 0).item())}
        actual = rows[actual_id]
        return {"schema": "visual_instance_binding.condition_scores.v1", "status": "scored", "condition": condition, "rows": rows, "candidate_sets": groups, "actual_row_id": actual_id, "actual_row_sum_logprob": actual["row_sum_logprob"], "input_identity": _input_identity(batch), "tensors": tensors}

    def _run(self) -> dict[str, Any]:
        self.q, identity = load_model(self.state["model"], self.device)
        self.model = self.q.model.eval()
        self._install_counters()
        batch, raw, trace, group, planning = _source(self.boundary, self.state["model"], self.source_panel, self.q, self.device)
        target = int(self.boundary["batch_index"])
        native = [int(token) for token in self.boundary["native_tokens"]]
        prefix_end = int(self.state["prefix_end"])
        prefix = native[:prefix_end]
        candidates, sets, actual_id, actual_tokens = _candidate_sets(self.state, native)
        native_score = self._score_batch(batch, raw, candidates, sets, actual_id, prefix, target, "native_original")
        clean_batch = self._variant_batch(self.q, group, "clean")
        clean_score = self._score_batch(clean_batch, raw, candidates, sets, actual_id, prefix, target, "clean")
        parity_rows = []
        for ident in native_score["tensors"]:
            delta = (native_score["tensors"][ident] - clean_score["tensors"][ident]).abs()
            parity_rows.append({"candidate_id": ident, "max_abs_logit_error": float(delta.max().item()), "passed": bool(float(delta.max().item()) <= ATOL)})
        native_actual = native_score["tensors"][actual_id]
        trace_parity = []
        for offset, token_id in enumerate(actual_tokens):
            trace_parity.append(_trace_compare(logits=native_actual[offset], trace=trace, batch_index=target, absolute_offset=prefix_end + offset, token_id=token_id, role=_row_role(actual_tokens, offset)))
        conditions = {"clean": clean_score}
        for condition in ("ablate_A", "ablate_N", "ablate_unrelated"):
            conditions[condition] = self._score_batch(self._variant_batch(self.q, group, condition), raw, candidates, sets, actual_id, prefix, target, condition)
        for value in conditions.values():
            value.pop("tensors")
        response: dict[str, dict[str, float]] = {}
        for region, condition in (("A", "ablate_A"), ("N", "ablate_N"), ("unrelated", "ablate_unrelated")):
            response[region] = {}
            for set_name in ("A", "N"):
                for ident in conditions["clean"]["candidate_sets"][set_name]["row_ids"]:
                    response[region][ident] = float(conditions[condition]["rows"][ident]["row_sum_logprob"] - conditions["clean"]["rows"][ident]["row_sum_logprob"])
        max_condition_difference = 0.0
        condition_differences = {}
        for condition in ("ablate_A", "ablate_N", "ablate_unrelated"):
            score = conditions[condition]
            deltas = [float(score["rows"][ident]["row_sum_logprob"] - clean_score["rows"][ident]["row_sum_logprob"]) for ident in score["rows"]]
            max_condition_difference = max(max_condition_difference, max(abs(value) for value in deltas))
            condition_differences[condition] = {"actual_row_logprob_delta": deltas[list(score["rows"]).index(actual_id)], "max_candidate_row_abs_delta": max(abs(value) for value in deltas)}
        result = {"schema": "visual_instance_binding.state_scores.v1", "status": "candidate_complete", "mode": self.args.mode, "campaign": self.args.campaign, "state_id": self.state["id"], "model": self.state["model"], "image_id": self.state["image_id"], "source_boundary_id": self.state["source_boundary_id"], "prefix_end": prefix_end, "candidate_sets": clean_score["candidate_sets"], "actual_row_id": actual_id, "actual_row_tokens": actual_tokens, "conditions": conditions, "response_matrix": response, "condition_differences": condition_differences, "source_native_replay_parity": {"tokens": trace_parity, "passed": all(item["passed"] for item in trace_parity), "entry_included": bool(trace_parity and trace_parity[0]["role"] == "entry"), "terminator_included": bool(trace_parity and trace_parity[-1]["role"] == "terminator")}, "clean_compositor_full_logit_parity": {"rows": parity_rows, "max_abs_logit_error": max(item["max_abs_logit_error"] for item in parity_rows), "passed": all(item["passed"] for item in parity_rows)}, "actual_condition_difference": {"max_abs_row_logprob_delta": max_condition_difference, "passed": max_condition_difference > 0.0}, "compositor": self.images, "identity": identity, "source_planning": planning, "source_trace": self.source_record["trace"], "frozen_masks": self.state["masks"]}
        scores_binding = write_once(self.output / "scores.json", result)
        result["scores"] = scores_binding
        result["status"] = "candidate_complete" if result["source_native_replay_parity"]["passed"] and result["clean_compositor_full_logit_parity"]["passed"] and result["actual_condition_difference"]["passed"] else "technical_invalid"
        self.counters["elapsed_seconds"] = time.monotonic() - self.started
        self.counters["gpu_seconds"] = self.counters["elapsed_seconds"] if self.device.type == "cuda" else 0.0
        result["counters"] = self.counters
        return result

    def run(self) -> dict[str, Any]:
        try:
            result = self._run()
        except BaseException as error:
            self.counters["elapsed_seconds"] = time.monotonic() - self.started
            self.counters["gpu_seconds"] = self.counters["elapsed_seconds"] if self.device.type == "cuda" else 0.0
            receipt = {"schema": "visual_instance_binding.receipt.v1", "status": "technical_invalid", "mode": self.args.mode, "campaign": self.args.campaign, "state_id": self.state["id"], "model": self.state["model"], "device": self.args.device, "launch": self.launch, "source_snapshot": self.snapshot, "counters": self.counters, "error": repr(error)}
            write_once(self.output / "receipt.json", receipt)
            raise
        finally:
            for handle in self.handles:
                handle.remove()
        receipt = {"schema": "visual_instance_binding.receipt.v1", "status": result["status"], "mode": self.args.mode, "campaign": self.args.campaign, "state_id": self.state["id"], "model": self.state["model"], "device": self.args.device, "launch": self.launch, "source_snapshot": self.snapshot, "scores": result["scores"], "counters": result["counters"], "parity": {"native_replay": result["source_native_replay_parity"], "clean_compositor": result["clean_compositor_full_logit_parity"], "actual_condition_difference": result["actual_condition_difference"]}}
        write_once(self.output / "receipt.json", receipt)
        return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("qualify", "state"), required=True)
    parser.add_argument("--state-id", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--campaign", default="qualification-01")
    parser.add_argument("--max-forwards", type=int, default=512)
    parser.add_argument("--max-seconds", type=float, default=3600.0)
    args = parser.parse_args()
    receipt = Runtime(args).run()
    print(json.dumps({"status": receipt["status"], "state_id": receipt["state_id"], "receipt": str(args.output / "receipt.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
