"""Score frozen complete rows under the unchanged native original policy.

The entrypoint intentionally does one exact replay per frozen candidate row.
It keeps full-vocabulary log probabilities, while retaining full logits only at
the native row boundary and the first common candidate fork.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch

from probes.training_set_completion.coordinate_continuity.runtime import _source
from probes.training_set_completion.numerical_feedback.select import rows as parsed_rows
from src.qwen.input_identity import input_identity as _input_identity
from probes.training_set_completion.untied_shared import load_model
from src.qwen.native import exact_history_inputs


ROOT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-21-native-row-choice")
ADMISSION = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-21-history-source-sign/selection/shared-admission.json"
)
ADMISSION_BINDING = ROOT / "selection" / "shared-admission-binding.json"
EOS = 151645
ROW_OPEN = 151646
REF_END = 151647
BOX_START = 151648
COORD_BASE = 151670
COORD_COUNT = 1000
ROW_END = 151649
ATOL = 2e-4
QUALIFY_STATE = "tied-14038-first-revisit"


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _binding(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "sha256": _file_digest(path), "size_bytes": path.stat().st_size}


def _write_new(path: Path, value: Any) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return _binding(path)


def _save_pt_new(path: Path, value: Any) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    return _binding(path)


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


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_admission() -> tuple[dict[str, Any], dict[str, Any]]:
    _require(ADMISSION.is_file(), f"frozen admission is missing: {ADMISSION}")
    _require(ADMISSION_BINDING.is_file(), f"admission binding receipt is missing: {ADMISSION_BINDING}")
    admission_binding = json.loads(ADMISSION_BINDING.read_text())
    _require(admission_binding.get("status") == "frozen", "admission binding is not frozen")
    _require(admission_binding.get("shared_admission", {}).get("path") == str(ADMISSION), "admission binding path drifted")
    observed = _binding(ADMISSION)
    _require(admission_binding.get("shared_admission") == observed, "frozen admission binding changed")
    admission = json.loads(ADMISSION.read_text())
    _require(admission.get("status") == "frozen_before_model_comparisons", "shared admission is not frozen")
    lane_b = admission.get("lane_b")
    _require(isinstance(lane_b, dict), "shared admission has no Lane B")
    _require(int(lane_b.get("counts", {}).get("ready", -1)) == len(lane_b.get("states", [])), "Lane B ready count drifted")
    return admission, admission_binding


def _state(admission: dict[str, Any], state_id: str) -> dict[str, Any]:
    states = [item for item in admission["lane_b"]["states"] if item.get("id") == state_id]
    _require(len(states) == 1, f"state ID is not unique in frozen admission: {state_id}")
    state = copy.deepcopy(states[0])
    _require(state.get("status") == "ready", f"state is not ready: {state_id}")
    _require(state.get("model") in ("tied", "untied"), f"unsupported state model: {state_id}")
    candidate_sets = state.get("candidate_sets")
    _require(isinstance(candidate_sets, dict), f"state has no candidate sets: {state_id}")
    rows_by_id: dict[str, dict[str, Any]] = {}
    for set_name in ("A", "C", "N"):
        candidates = candidate_sets.get(set_name, [])
        _require(isinstance(candidates, list) and len(candidates) <= 4, f"candidate set cap changed: {state_id}/{set_name}")
        for candidate in candidates:
            ident = str(candidate.get("id"))
            _require(ident and ident not in rows_by_id, f"candidate ID is not unique: {state_id}/{ident}")
            tokens = candidate.get("tokens")
            _require(isinstance(tokens, list) and all(type(token) is int for token in tokens), f"candidate tokens are invalid: {state_id}/{ident}")
            _require(candidate.get("token_sha256") == _digest(tokens), f"candidate token binding changed: {state_id}/{ident}")
            parsed = parsed_rows(tokens)
            _require(len(parsed) == 1 and parsed[0]["start"] == 0 and parsed[0]["end"] == len(tokens), f"candidate is not one complete row: {state_id}/{ident}")
            _require(bool(parsed[0]["valid"]), f"candidate geometry is invalid: {state_id}/{ident}")
            rows_by_id[ident] = candidate
    _require(len(rows_by_id) <= 13, f"finite candidate cap changed: {state_id}")
    actual_id = str(candidate_sets.get("actual_greedy_id"))
    _require(actual_id in rows_by_id, f"actual greedy row is absent: {state_id}")
    return state


def _find_boundary(panel: dict[str, Any], boundary_id: str) -> dict[str, Any]:
    matches = [item for item in panel.get("all_boundaries", []) if item.get("id") == boundary_id]
    _require(len(matches) == 1, f"source boundary is not unique: {boundary_id}")
    return matches[0]


def _source_record(admission: dict[str, Any], boundary_id: str) -> dict[str, Any]:
    matches = [item for item in admission["source_pool"] if item.get("source_boundary_id") == boundary_id]
    _require(len(matches) == 1, f"source-pool record is not unique: {boundary_id}")
    return matches[0]


def _verify_source_bindings(record: dict[str, Any]) -> None:
    for key in ("source_panel", "raw", "trace", "receipt", "image"):
        expected = record.get(key)
        _require(isinstance(expected, dict) and isinstance(expected.get("path"), str), f"source binding missing: {key}")
        _require(_binding(Path(expected["path"])) == expected, f"source binding changed: {key}")


def _common_fork(candidates: list[dict[str, Any]]) -> int | None:
    if not candidates:
        return None
    width = min(len(item["tokens"]) for item in candidates)
    for offset in range(width):
        if len({int(item["tokens"][offset]) for item in candidates}) > 1:
            return offset
    if len({len(item["tokens"]) for item in candidates}) > 1:
        return width
    return None


def _row_role(tokens: list[int], offset: int) -> str:
    if offset == 0:
        return "entry"
    if tokens[offset] == REF_END:
        return "description_end"
    if tokens[offset] == BOX_START:
        return "box_start"
    if tokens[offset] == ROW_END:
        return "terminator"
    if COORD_BASE <= tokens[offset] < COORD_BASE + COORD_COUNT:
        return ("x1", "y1", "x2", "y2")[max(0, min(3, offset - 4))]
    return "description"


def _trace_top2(step: dict[str, Any], batch_index: int) -> tuple[list[int], list[float]]:
    values = step["raw_top2"][batch_index]
    if values and isinstance(values[0], (list, tuple)):
        token_ids = [int(item[0]) for item in values]
        logits = [float(item[1]) for item in values]
    else:
        token_ids = [int(step["raw_winners"][batch_index]), int(step["raw_runnerups"][batch_index])]
        logits = [float(item) for item in values]
    _require(len(token_ids) == 2 and len(logits) == 2, "native source trace top2 is not length two")
    return token_ids, logits


def _trace_compare(
    *,
    logits: torch.Tensor,
    trace: dict[str, Any],
    batch_index: int,
    absolute_offset: int,
    token_id: int,
    role: str,
    atol: float = ATOL,
) -> dict[str, Any]:
    steps = trace.get("steps", [])
    _require(0 <= absolute_offset < len(steps), f"source trace lacks offset {absolute_offset}")
    step = steps[absolute_offset]
    chosen = int(step["chosen"][batch_index])
    chosen_raw = float(step["chosen_raw_logits"][batch_index])
    saved_lse = float(step["logsumexp"][batch_index])
    current = logits.detach().float()
    current_lse = float(torch.logsumexp(current, dim=-1).item())
    current_chosen = float(current[int(token_id)].item())
    current_logprob = float(torch.log_softmax(current, dim=-1)[int(token_id)].item())
    source_logprob = chosen_raw - saved_lse
    values, indices = torch.topk(current, 2)
    current_top_ids = [int(item) for item in indices.tolist()]
    current_top_logits = [float(item) for item in values.tolist()]
    source_top_ids, source_top_logits = _trace_top2(step, batch_index)
    logprob_error = abs(current_logprob - source_logprob)
    chosen_logit_error = abs(current_chosen - chosen_raw)
    lse_error = abs(current_lse - saved_lse)
    top2_error = max(abs(a - b) for a, b in zip(current_top_logits, source_top_logits, strict=True))
    return {
        "absolute_offset": absolute_offset,
        "role": role,
        "token_id": int(token_id),
        "source_chosen_token_id": chosen,
        "source_top2_token_ids": source_top_ids,
        "current_top2_token_ids": current_top_ids,
        "source_chosen_raw_logit": chosen_raw,
        "source_logsumexp": saved_lse,
        "source_logprob": source_logprob,
        "current_chosen_logit": current_chosen,
        "current_logsumexp": current_lse,
        "current_logprob": current_logprob,
        "chosen_logit_abs_error": chosen_logit_error,
        "logsumexp_abs_error": lse_error,
        "logprob_abs_error": logprob_error,
        "top2_max_abs_error": top2_error,
        "winner_match": current_top_ids[0] == source_top_ids[0],
        "runnerup_match": current_top_ids[1] == source_top_ids[1],
        "chosen_token_match": chosen == int(token_id),
        "passed": bool(
            chosen == int(token_id)
            and current_top_ids == source_top_ids
            and max(chosen_logit_error, lse_error, logprob_error, top2_error) <= atol
        ),
        "tolerance": atol,
    }


def _make_histories(batch: Any, raw: list[dict[str, Any]], target: int, actions: list[int], pad: int) -> list[list[int]]:
    histories: list[list[int]] = []
    for index, prompt in enumerate(batch.prompt_token_ids):
        if index == target:
            suffix = list(actions)
        else:
            suffix = [int(token) for token in raw[index]["token_ids"][: len(actions)]]
        suffix.extend([pad] * (len(actions) - len(suffix)))
        histories.append(list(prompt) + suffix)
    return histories


def _score_candidate(
    *,
    model: Any,
    batch: Any,
    raw: list[dict[str, Any]],
    target: int,
    prefix: list[int],
    tokens: list[int],
    pad: int,
    device: torch.device,
) -> dict[str, Any]:
    actions = prefix + list(tokens)
    histories = _make_histories(batch, raw, target, actions, pad)
    inputs = exact_history_inputs(
        model,
        batch.inputs,
        histories,
        pad_token_id=pad,
        logits_to_keep=len(tokens) + 1,
    )
    with torch.inference_mode():
        output = model(**inputs)
    logits = output.logits[target].detach().float()
    _require(logits.ndim == 2 and logits.shape[0] == len(tokens) + 1, "compact exact replay shape changed")
    action_logits = logits[:-1]
    logprobs = torch.log_softmax(action_logits, dim=-1)
    selected = logprobs[torch.arange(len(tokens), device=logprobs.device), torch.tensor(tokens, device=logprobs.device)]
    input_width = int(inputs["input_ids"].shape[1])
    position_start = input_width - len(tokens) - 1
    positions = inputs["position_ids"][..., target, position_start : input_width - 1].detach().cpu().transpose(0, 1).tolist()
    _require(len(positions) == len(tokens), "compact replay position count changed")
    return {
        "token_ids": list(tokens),
        "token_logprobs": [float(value) for value in selected.detach().cpu().tolist()],
        "row_sum_logprob": float(selected.sum().item()),
        "positions": positions,
        "positions_sha256": _digest(positions),
        "vocabulary_size": int(action_logits.shape[-1]),
        "boundary_logits": logits[0].detach().cpu(),
        "action_logits": action_logits.detach().cpu(),
        "input_ids_sha256": _digest(inputs["input_ids"][target].detach().cpu().tolist()),
        "prefix_sha256": _digest(prefix),
        "device": str(device),
    }


def _validate_row(row: Mapping[str, Any]) -> None:
    tokens = row.get("token_ids")
    logprobs = row.get("token_logprobs")
    positions = row.get("positions")
    _require(isinstance(tokens, list) and all(type(token) is int for token in tokens), "row token IDs are invalid")
    _require(isinstance(logprobs, list) and len(logprobs) == len(tokens), "row token logprob length does not match tokens")
    _require(isinstance(positions, list) and len(positions) == len(tokens), "row position length does not match tokens")
    _require(len(tokens) >= 7 and tokens[0] == ROW_OPEN and tokens[-1] == ROW_END, "row opener or terminator was dropped")
    parsed = parsed_rows(tokens)
    _require(len(parsed) == 1 and bool(parsed[0]["valid"]), "row is not a valid complete serialized row")
    expected = sum(float(item) for item in logprobs)
    _require(abs(expected - float(row.get("row_sum_logprob"))) <= 1e-5, "row sum does not match token logprobs")


def _validate_state_result(result: Mapping[str, Any]) -> None:
    rows_by_id = result.get("rows")
    _require(isinstance(rows_by_id, dict) and rows_by_id, "state result has no rows")
    actual_id = str(result.get("actual_greedy_id"))
    _require(actual_id in rows_by_id, "state result has no actual greedy row")
    candidate_sets = result.get("candidate_sets")
    _require(isinstance(candidate_sets, dict), "state result has no candidate sets")
    seen: set[str] = set()
    for set_name in ("A", "C", "N"):
        group = candidate_sets.get(set_name, {})
        _require(isinstance(group, dict), f"candidate set is malformed: {set_name}")
        ids = group.get("row_ids", [])
        _require(isinstance(ids, list), f"candidate set IDs are malformed: {set_name}")
        for ident in ids:
            _require(str(ident) in rows_by_id and str(ident) not in seen, f"candidate set row is missing or duplicated: {set_name}/{ident}")
            seen.add(str(ident))
        if ids:
            _require(math.isfinite(float(group.get("row_logsumexp"))), f"candidate set logsumexp is missing: {set_name}")
        else:
            _require(group.get("row_logsumexp") is None, f"empty candidate set must have null logsumexp: {set_name}")
    _require(seen == set(rows_by_id), "candidate set coverage changed")
    for row in rows_by_id.values():
        _validate_row(row)


def _reduce_state(result: Mapping[str, Any]) -> dict[str, Any]:
    _validate_state_result(result)
    rows_by_id = result["rows"]
    reduced: dict[str, Any] = {}
    for set_name, group in result["candidate_sets"].items():
        values = [float(rows_by_id[str(ident)]["row_sum_logprob"]) for ident in group["row_ids"]]
        reduced[set_name] = {
            "row_count": len(values),
            "row_logsumexp": None if not values else float(torch.logsumexp(torch.tensor(values, dtype=torch.float64), dim=0).item()),
        }
    return reduced


def _falsification(result: dict[str, Any]) -> dict[str, Any]:
    _validate_state_result(result)
    checks: list[dict[str, Any]] = []
    actual = str(result["actual_greedy_id"])
    for name, reducer in (("dropped_entry", "validator"), ("dropped_terminator", "reducer")):
        corrupted = copy.deepcopy(result)
        row = corrupted["rows"][actual]
        if name == "dropped_entry":
            row["token_ids"] = row["token_ids"][1:]
            row["token_logprobs"] = row["token_logprobs"][1:]
            row["positions"] = row["positions"][1:]
        else:
            row["token_ids"] = row["token_ids"][:-1]
            row["token_logprobs"] = row["token_logprobs"][:-1]
            row["positions"] = row["positions"][:-1]
        row["row_sum_logprob"] = sum(float(value) for value in row["token_logprobs"])
        try:
            if reducer == "validator":
                _validate_state_result(corrupted)
            else:
                _reduce_state(corrupted)
        except (AssertionError, KeyError, TypeError, ValueError) as error:
            checks.append({"mutation": name, "checked_by": reducer, "rejected": True, "error": str(error)})
        else:
            checks.append({"mutation": name, "checked_by": reducer, "rejected": False, "error": None})
    return {"schema": "native_row_choice.probability_accounting_falsification.v1", "baseline_valid": True, "checks": checks, "passed": all(item["rejected"] for item in checks)}


class Runtime:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.started = time.monotonic()
        self.admission, self.admission_binding = _load_admission()
        self.state = _state(self.admission, args.state_id)
        self.panel_path = Path(self.admission["panel"]["path"])
        _require(_binding(self.panel_path) == self.admission["panel"], "shared panel binding changed")
        self.panel = json.loads(self.panel_path.read_text())
        self.boundary = _find_boundary(self.panel, self.state["source_boundary_id"])
        self.source_record = _source_record(self.admission, self.state["source_boundary_id"])
        _verify_source_bindings(self.source_record)
        self.output = self._output_path(args)
        self.output.mkdir(parents=True, exist_ok=False)
        self.counters = {
            "model_forwards": 0,
            "vision_forwards": 0,
            "candidate_rows": 0,
            "retained_tensor_bytes": 0,
            "max_model_forwards": int(args.max_forwards),
            "max_seconds": float(args.max_seconds),
        }
        self._write_launch()
        self.device = torch.device(args.device)
        self.q: Any = None
        self.model: Any = None
        self.handles: list[Any] = []

    def _output_path(self, args: argparse.Namespace) -> Path:
        if args.output is not None:
            output = args.output.resolve()
        elif args.mode == "qualify":
            output = ROOT / ("qualification-repair-1" if args.repair else "qualification") / args.state_id
        else:
            output = ROOT / "states" / args.state_id
        if args.mode == "qualify":
            _require(args.state_id == QUALIFY_STATE, "qualification is frozen to tied-14038-first-revisit")
            initial = ROOT / "qualification" / args.state_id / "receipt.json"
            repair = ROOT / "qualification-repair-1" / args.state_id / "receipt.json"
            if args.repair:
                _require(initial.is_file(), "bounded repair requires the original qualification receipt")
                previous = json.loads(initial.read_text())
                _require(previous.get("status") != "candidate_complete", "qualification already passed; rerun is forbidden")
                _require(not repair.exists(), "only one bounded qualification repair is allowed")
            else:
                _require(not initial.exists(), "qualification already has a terminal receipt; rerun is forbidden")
        elif args.state_id == QUALIFY_STATE:
            for path in (ROOT / "qualification" / args.state_id / "receipt.json", ROOT / "qualification-repair-1" / args.state_id / "receipt.json"):
                _require(not path.exists(), "tied-14038 production state was already qualified; state rerun is forbidden")
        return output

    def _write_launch(self) -> None:
        state_sources = {
            key: self.source_record[key]
            for key in ("source_panel", "raw", "trace", "receipt", "image")
        }
        binding_receipt = _write_new(
            self.output / "binding-receipt.json",
            {
                "schema": "native_row_choice.binding_receipt.v1",
                "status": "frozen_before_model_load",
                "shared_admission": _binding(ADMISSION),
                "admission_binding_receipt": _binding(ADMISSION_BINDING),
                "shared_panel": _binding(self.panel_path),
                "state_id": self.state["id"],
                "source_boundary_id": self.state["source_boundary_id"],
                "source_artifacts": state_sources,
                "candidate_digest": _digest(self.state["candidate_sets"]),
            },
        )
        _write_new(
            self.output / "launch.json",
            {
                "schema": "native_row_choice.launch.v1",
                "status": "frozen_before_model_load",
                "mode": self.args.mode,
                "state_id": self.state["id"],
                "model": self.state["model"],
                "device": self.args.device,
                "binding_receipt": binding_receipt,
                "producer": _binding(Path(__file__)),
                "loader": _binding(Path(__file__).resolve().parents[1] / "untied_shared.py"),
                "source_loader": _binding(Path(__file__).resolve().parents[1] / "coordinate_continuity" / "runtime.py"),
                "candidate_policy": {
                    "original_distribution": True,
                    "full_vocabulary_log_softmax": True,
                    "candidate_renormalization": False,
                    "beam_search": False,
                    "model_change": False,
                },
                "budget": {
                    "model_forwards": int(self.args.max_forwards),
                    "max_seconds": float(self.args.max_seconds),
                },
            },
        )

    def _install_counters(self) -> None:
        def model_forward(_module: Any, _inputs: tuple[Any, ...]) -> None:
            self.counters["model_forwards"] += 1
            if self.counters["model_forwards"] > self.counters["max_model_forwards"]:
                raise RuntimeError("model-forward budget exhausted")
            if time.monotonic() - self.started > self.counters["max_seconds"]:
                raise RuntimeError("wall-time budget exhausted")

        self.handles.append(self.model.register_forward_pre_hook(model_forward))
        visual = getattr(getattr(self.model, "model", None), "visual", None)
        if visual is not None:
            def vision_forward(_module: Any, _inputs: tuple[Any, ...]) -> None:
                self.counters["vision_forwards"] += 1
            self.handles.append(visual.register_forward_pre_hook(vision_forward))

    def _run(self) -> dict[str, Any]:
        self.q, identity = load_model(self.state["model"], self.device)
        self.model = self.q.model.eval()
        self._install_counters()
        source_panel = json.loads(Path(self.source_record["source_panel"]["path"]).read_text())
        batch, raw, trace, _group, planning = _source(
            self.boundary,
            self.state["model"],
            source_panel,
            self.q,
            self.device,
        )
        target = int(self.boundary["batch_index"])
        prefix_end = int(self.state["prefix_end"])
        native = [int(token) for token in self.boundary["native_tokens"]]
        prefix = native[:prefix_end]
        rows_by_id: dict[str, dict[str, Any]] = {}
        raw_by_id: dict[str, dict[str, Any]] = {}
        ordered_candidates: list[dict[str, Any]] = []
        set_by_id: dict[str, str] = {}
        for set_name in ("A", "C", "N"):
            for candidate in self.state["candidate_sets"].get(set_name, []):
                ident = str(candidate["id"])
                if ident in raw_by_id:
                    continue
                raw_by_id[ident] = candidate
                set_by_id[ident] = set_name
                ordered_candidates.append(candidate)
        fork_offset = _common_fork(ordered_candidates)
        boundary_logits: torch.Tensor | None = None
        fork_logits: dict[str, torch.Tensor] = {}
        actual_parity: list[dict[str, Any]] = []
        boundary_parity: dict[str, Any] | None = None
        actual_id = str(self.state["candidate_sets"]["actual_greedy_id"])
        actual_tokens = [int(token) for token in raw_by_id[actual_id]["tokens"]]
        _require(native[prefix_end : prefix_end + len(actual_tokens)] == actual_tokens, "actual greedy row does not bind the native source row")
        for candidate in ordered_candidates:
            ident = str(candidate["id"])
            scored = _score_candidate(
                model=self.model,
                batch=batch,
                raw=raw,
                target=target,
                prefix=prefix,
                tokens=[int(token) for token in candidate["tokens"]],
                pad=int(self.q.tokenizer.pad_token_id),
                device=self.device,
            )
            self.counters["candidate_rows"] += 1
            if boundary_logits is None:
                boundary_logits = scored["boundary_logits"]
                boundary_parity = _trace_compare(
                    logits=boundary_logits,
                    trace=trace,
                    batch_index=target,
                    absolute_offset=prefix_end,
                    token_id=int(native[prefix_end]),
                    role="native_row_boundary",
                )
            if fork_offset is not None:
                fork_logits[ident] = scored["action_logits"][fork_offset].clone()
            if ident == actual_id:
                for offset, token_id in enumerate(actual_tokens):
                    actual_parity.append(
                        _trace_compare(
                            logits=scored["action_logits"][offset],
                            trace=trace,
                            batch_index=target,
                            absolute_offset=prefix_end + offset,
                            token_id=token_id,
                            role=_row_role(actual_tokens, offset),
                        )
                    )
            scored.pop("boundary_logits")
            action_logits = scored.pop("action_logits")
            scored.update(
                {
                    "set": set_by_id[ident],
                    "candidate_id": ident,
                    "owner": candidate["owner"],
                    "source": candidate["source"],
                    "construction_rule": candidate["construction_rule"],
                }
            )
            rows_by_id[ident] = scored
        _require(boundary_logits is not None and boundary_parity is not None, "no candidate row was scored")
        fork_path = _save_pt_new(
            self.output / "first-common-fork-logits.pt",
            {
                "schema": "native_row_choice.first_common_fork_logits.v1",
                "state_id": self.state["id"],
                "relative_offset": fork_offset,
                "absolute_offset": None if fork_offset is None else prefix_end + fork_offset,
                "common_prefix_tokens": [] if fork_offset is None else ordered_candidates[0]["tokens"][:fork_offset],
                "candidate_token_ids": {str(item["id"]): int(item["tokens"][fork_offset]) for item in ordered_candidates} if fork_offset is not None else {},
                "logits": fork_logits,
            },
        )
        boundary_path = _save_pt_new(
            self.output / "native-row-boundary-logits.pt",
            {
                "schema": "native_row_choice.native_row_boundary_logits.v1",
                "state_id": self.state["id"],
                "absolute_offset": prefix_end,
                "native_next_token_id": int(native[prefix_end]),
                "logits": boundary_logits,
            },
        )
        self.counters["retained_tensor_bytes"] = fork_path["size_bytes"] + boundary_path["size_bytes"]
        candidate_sets: dict[str, Any] = {}
        for set_name in ("A", "C", "N"):
            ids = [str(item["id"]) for item in self.state["candidate_sets"].get(set_name, [])]
            values = [float(rows_by_id[ident]["row_sum_logprob"]) for ident in ids]
            candidate_sets[set_name] = {
                "row_ids": ids,
                "row_count": len(ids),
                "row_logsumexp": None if not values else float(torch.logsumexp(torch.tensor(values, dtype=torch.float64), dim=0).item()),
                "secondary_only": True,
            }
        result: dict[str, Any] = {
            "schema": "native_row_choice.state_scores.v1",
            "status": "candidate_complete",
            "mode": self.args.mode,
            "state_id": self.state["id"],
            "model": self.state["model"],
            "source_boundary_id": self.state["source_boundary_id"],
            "image_id": self.state["image_id"],
            "prefix_end": prefix_end,
            "common_description_tokens": self.state["candidate_sets"]["common_description_tokens"],
            "actual_greedy_id": actual_id,
            "candidate_sets": candidate_sets,
            "rows": rows_by_id,
            "native_row_boundary": {
                "absolute_offset": prefix_end,
                "next_token_id": int(native[prefix_end]),
                "trace_parity": boundary_parity,
            },
            "first_common_fork": {
                "relative_offset": fork_offset,
                "absolute_offset": None if fork_offset is None else prefix_end + fork_offset,
                "full_vocab_logits": fork_path,
            },
            "actual_row_trace_parity": {
                "token_count": len(actual_parity),
                "entry_included": bool(actual_parity and actual_parity[0]["role"] == "entry"),
                "terminator_included": bool(actual_parity and actual_parity[-1]["role"] == "terminator"),
                "passed": all(item["passed"] for item in actual_parity),
                "tokens": actual_parity,
            },
            "probability_accounting": {
                "distribution": "native original full vocabulary",
                "log_softmax": "full vocabulary at every serialized row token",
                "includes_entry": True,
                "includes_terminator": True,
                "candidate_renormalization": False,
                "beam_search": False,
                "candidate_set_logsumexp_secondary_only": True,
            },
            "identity": _jsonable(identity),
            "input_identity": _input_identity(batch),
            "source_planning": _jsonable(planning),
            "source_trace": {
                "path": self.source_record["trace"],
                "sha256": self.source_record["trace"]["sha256"],
            },
        }
        _validate_state_result(result)
        falsification = _falsification(result)
        falsification_binding = _write_new(self.output / "probability-accounting-falsification.json", falsification)
        result["probability_accounting_falsification"] = falsification_binding
        result_binding = _write_new(self.output / "scores.json", result)
        self.counters["elapsed_seconds"] = time.monotonic() - self.started
        self.counters["gpu_seconds"] = self.counters["elapsed_seconds"] if self.device.type == "cuda" else 0.0
        receipt = {
            "schema": "native_row_choice.receipt.v1",
            "status": "candidate_complete" if result["actual_row_trace_parity"]["passed"] and falsification["passed"] else "technical_invalid",
            "mode": self.args.mode,
            "state_id": self.state["id"],
            "model": self.state["model"],
            "device": str(self.device),
            "launch": _binding(self.output / "launch.json"),
            "binding_receipt": _binding(self.output / "binding-receipt.json"),
            "scores": result_binding,
            "falsification": falsification_binding,
            "counters": self.counters,
            "parameters_unchanged": True,
            "optional_continuation": "not_run",
        }
        return receipt

    def run(self) -> dict[str, Any]:
        receipt: dict[str, Any]
        try:
            receipt = self._run()
        except BaseException as error:
            self.counters["elapsed_seconds"] = time.monotonic() - self.started
            self.counters["gpu_seconds"] = self.counters["elapsed_seconds"] if getattr(self, "device", torch.device("cpu")).type == "cuda" else 0.0
            receipt = {
                "schema": "native_row_choice.receipt.v1",
                "status": "technical_invalid",
                "mode": self.args.mode,
                "state_id": self.state["id"],
                "model": self.state["model"],
                "device": self.args.device,
                "launch": _binding(self.output / "launch.json"),
                "binding_receipt": _binding(self.output / "binding-receipt.json"),
                "error": repr(error),
                "counters": self.counters,
            }
            _write_new(self.output / "receipt.json", receipt)
            raise
        finally:
            for handle in self.handles:
                handle.remove()
        _write_new(self.output / "receipt.json", receipt)
        return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("qualify", "state"), required=True)
    parser.add_argument("--state-id", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repair", action="store_true")
    parser.add_argument("--max-forwards", type=int, default=64)
    parser.add_argument("--max-seconds", type=float, default=3600.0)
    args = parser.parse_args()
    if args.repair and args.mode != "qualify":
        parser.error("--repair is valid only with --mode qualify")
    runtime = Runtime(args)
    receipt = runtime.run()
    print(json.dumps({"status": receipt["status"], "state_id": receipt["state_id"], "receipt": str(runtime.output / "receipt.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
