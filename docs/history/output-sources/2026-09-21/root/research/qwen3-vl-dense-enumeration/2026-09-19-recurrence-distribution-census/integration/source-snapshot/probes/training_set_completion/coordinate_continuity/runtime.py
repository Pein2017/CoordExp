"""Bounded fixed-history input-coordinate continuity replay.

The plan and shared panel are CPU-frozen before this entry is allowed to load
the model. Each replay is a full-prefix teacher-forced call: no token is
sampled and only the target history coordinate may differ.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch

from probes.training_set_completion.numerical_feedback.runtime import full_prefix
from probes.training_set_completion.readout_norm_fresh import _binding, _input_identity
from probes.training_set_completion.untied_shared import load_model
from src.config.inference import InferConfig
from src.data.examples import raw_example_from_jsonl_row
from src.inference.bound_requests import build_bound_native_requests
from src.inference.inputs import plan_examples
from src.qwen.native import prepare_native_inputs


COORD_BASE = 151670
COORD_COUNT = 1000
EOS = 151645
PARITY_ATOL = 2e-4


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def tensor_hash(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def token_hash(tokens: list[int]) -> str:
    return digest(tokens)


def _panel_boundaries(panel: dict[str, Any]) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for key in ("boundaries", "existing_boundaries"):
        raw = panel.get(key)
        if isinstance(raw, list):
            values.extend(item for item in raw if isinstance(item, dict))
    raw_new = panel.get("new_boundaries")
    if isinstance(raw_new, list):
        values.extend(item for item in raw_new if isinstance(item, dict))
    by_id: dict[str, dict[str, Any]] = {}
    for item in values:
        ident = str(item.get("id", item.get("boundary_id", "")))
        if ident:
            by_id.setdefault(ident, item)
    return list(by_id.values())


def _find_boundary(panel: dict[str, Any], ident: str) -> dict[str, Any]:
    matches = [x for x in _panel_boundaries(panel) if str(x.get("id", x.get("boundary_id"))) == ident]
    if len(matches) != 1:
        raise ValueError(f"boundary is not unique: {ident}")
    return matches[0]


def _group(panel: dict[str, Any], key: str) -> dict[str, Any]:
    groups = panel.get("groups", [])
    matches = [x for x in groups if isinstance(x, dict) and x.get("key") == key]
    if len(matches) != 1:
        raise ValueError(f"source runtime group is not unique: {key}")
    return matches[0]


def _source_panel(boundary: dict[str, Any], explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    raw = Path(boundary["raw_path"]).resolve()
    candidate = raw.parents[3] / "panel.json"
    if not candidate.is_file():
        raise FileNotFoundError(f"cannot derive source panel from {raw}")
    return candidate


def _source(boundary: dict[str, Any], model: str, source_panel: dict[str, Any], q: Any, device: torch.device) -> tuple[Any, list[dict[str, Any]], dict[str, Any], Any, dict[str, Any]]:
    group = _group(source_panel, str(boundary["group"]))
    raw_path = Path(boundary["raw_path"])
    trace_path = Path(boundary["trace_path"])
    receipt_path = Path(boundary["receipt_path"])
    for path in (raw_path, trace_path, receipt_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    raw = json.loads(raw_path.read_text())["rows"]
    trace = json.loads(trace_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("status") != "candidate_complete":
        raise ValueError("native source receipt is not complete")
    if receipt.get("condition") != f"{model}-original":
        raise ValueError("native source condition differs from declared model")
    if receipt.get("group") != boundary["group"]:
        raise ValueError("native source group differs")
    if receipt.get("raw") != _binding(raw_path) or receipt.get("trace") != _binding(trace_path):
        raise ValueError("native source receipt bindings differ")
    config = dict(source_panel["configs"][model])
    config["data"] = dict(input_jsonl=group["input_jsonl"])
    cases = group["cases"]
    if any("image_plan" not in case for case in cases):
        # New census groups use the same source-owned planning seam as the
        # accepted natural producer.  The panel intentionally stores the raw
        # cases, while the native source receipt binds the resulting input
        # identity.  Replan only this mechanical format seam and require the
        # saved receipt identity below.
        config_model = InferConfig.model_validate(source_panel["configs"][model])
        raw_examples = [
            raw_example_from_jsonl_row(
                case["input_record"],
                jsonl_path=Path(group["input_jsonl"]),
                row_number=int(case["row_index"]) + 1,
                raw_line=json.dumps(case["input_record"]),
            )
            for case in cases
        ]
        planned = plan_examples(
            raw_examples,
            config=config_model,
            components=q,
            row_indices=[int(case["row_index"]) for case in cases],
        )
        for case, item in zip(cases, planned, strict=True):
            case["image_path"] = item.image.image_path
            case["image_plan"] = item.image.to_artifact_dict()
        requests = [item.request for item in planned]
        planning = {
            "replanned_image_plan": True,
            "case_indices": [int(case["row_index"]) for case in cases],
            "image_plan_bindings": [digest(case["image_plan"]) for case in cases],
            "producer_path": str((Path(__file__).resolve().parents[1] / "recurrence_census" / "natural.py").resolve()),
            "producer_sha256": hashlib.sha256((Path(__file__).resolve().parents[1] / "recurrence_census" / "natural.py").read_bytes()).hexdigest(),
        }
    else:
        requests, _ = build_bound_native_requests(q, config, cases)
        planning = {"replanned_image_plan": False}
    batch = prepare_native_inputs(q.processor, requests, device=device, record_media_identity=True)
    if receipt.get("input_identity") != _input_identity(batch):
        raise ValueError("native prompt/media identity changed")
    index = int(boundary["batch_index"])
    if len(raw) != len(group["cases"]) or int(raw[index]["image_id"]) != int(boundary["image_id"]):
        raise ValueError("native companion or image identity changed")
    native = [int(x) for x in raw[index]["token_ids"]]
    if native != [int(x) for x in boundary["native_tokens"]]:
        raise ValueError("native token identity changed")
    if token_hash(native) != str(boundary["native_token_hash"]):
        raise ValueError("native token hash changed")
    if not isinstance(trace.get("steps"), list):
        raise ValueError("native trace has no steps")
    planning["receipt_input_identity"] = receipt.get("input_identity")
    planning["replayed_input_identity"] = _input_identity(batch)
    return batch, raw, trace, group, planning


def _trace_expectation(trace: dict[str, Any], batch_index: int, offset: int) -> dict[str, Any]:
    steps = trace["steps"]
    if not 0 <= int(offset) < len(steps):
        raise ValueError(f"source trace lacks offset {offset}")
    step = steps[int(offset)]
    values = step["raw_top2"][batch_index]
    if len(values) != 2:
        raise ValueError("source trace raw_top2 must contain two values")
    # The accepted native producer stores raw_top2 as values only and keeps
    # the two token identities in raw_winners/raw_runnerups.  Keep accepting
    # the older nested (token, value) representation for source compatibility.
    if isinstance(values[0], (list, tuple)):
        top2 = [[int(x[0]), float(x[1])] for x in values]
    else:
        top2 = [
            [int(step["raw_winners"][batch_index]), float(values[0])],
            [int(step["raw_runnerups"][batch_index]), float(values[1])],
        ]
    return {
        "winner": int(step["raw_winners"][batch_index]),
        "top2": top2,
    }


class Runtime:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.started = time.monotonic()
        self.output = args.output_root
        self.output.mkdir(parents=True, exist_ok=True)
        self.plan = json.loads(args.plan.read_text())
        self.panel = json.loads(args.panel.read_text())
        if self.plan.get("status") != "frozen_before_model_calls":
            raise ValueError("execution plan is not frozen before model calls")
        if self.plan.get("panel", {}).get("sha256") != _binding(args.panel).get("sha256"):
            raise ValueError("shared panel binding differs from execution plan")
        self.planned_states = [state for state in self.plan["states"] if state["model"] == args.model]
        self.states = list(self.planned_states)
        if args.state_ids:
            wanted = set(args.state_ids)
            self.states = [state for state in self.states if state["id"] in wanted]
            if {state["id"] for state in self.states} != wanted:
                raise ValueError("requested state ID is absent from the plan")
        if not self.states:
            raise ValueError("no states for declared model")
        self.device = torch.device(args.device)
        self.forward_count = 0
        self.handles: list[Any] = []
        self.source_panel_paths = {
            state["boundary_id"]: _source_panel(
                _find_boundary(self.panel, state["boundary_id"]), args.source_panel
            )
            for state in self.planned_states
        }
        self.source_panels = {
            str(path): json.loads(path.read_text())
            for path in set(self.source_panel_paths.values())
        }
        self._write_snapshot()
        self.q, self.identity = load_model(args.model, self.device)
        self.model = self.q.model.eval()
        self.head = self.model.get_output_embeddings()
        self.input_embeddings = self.model.get_input_embeddings()
        self.layers = list(self.model.model.language_model.layers)
        self.norm = self.model.model.language_model.norm
        self._install_counter()
        self.receipt_path = self.output / "runtime" / f"{args.model}-shard-receipt.json"
        self.receipt = {
            "schema": "coordinate_continuity.shard_receipt.v1",
            "status": "running",
            "pid": os.getpid(),
            "model": args.model,
            "device": str(self.device),
            "plan": _binding(args.plan),
            "panel": _binding(args.panel),
            "source_panels": {
                str(path): _binding(path) for path in sorted(set(self.source_panel_paths.values()))
            },
            "producer": _binding(Path(__file__)),
            "states": [state["id"] for state in self.states],
            "model_forwards": 0,
            "started_monotonic": self.started,
        }
        self.persist()

    def _write_snapshot(self) -> None:
        sources = []
        for state in self.planned_states:
            boundary = _find_boundary(self.panel, state["boundary_id"])
            sources.append(
                {
                    "state_id": state["id"],
                    "boundary_id": state["boundary_id"],
                    "raw": _binding(Path(boundary["raw_path"])),
                    "trace": _binding(Path(boundary["trace_path"])),
                    "receipt": _binding(Path(boundary["receipt_path"])),
                    "source_panel": _binding(self.source_panel_paths[state["boundary_id"]]),
                }
            )
        path = self.output / "runtime" / f"producer-snapshot-{self.args.model}.json"
        value = {
            "schema": "coordinate_continuity.producer_snapshot.v1",
            "status": "frozen_before_model_replay",
            "producer": _binding(Path(__file__)),
            "plan": _binding(self.args.plan),
            "shared_panel": _binding(self.args.panel),
            "source_panels": {
                str(path): _binding(path) for path in sorted(set(self.source_panel_paths.values()))
            },
            "model": self.args.model,
            "device": str(self.device),
            "states": [state["id"] for state in self.planned_states],
            "source_artifacts": sources,
        }
        if path.exists():
            if json.loads(path.read_text()) != value:
                raise ValueError(f"producer snapshot already exists with different bindings: {path}")
            return
        write_json(path, value)

    def persist(self, status: str | None = None, error: str | None = None) -> None:
        if not hasattr(self, "receipt"):
            return
        self.receipt["model_forwards"] = self.forward_count
        self.receipt["elapsed_seconds"] = time.monotonic() - self.started
        if status is not None:
            self.receipt["status"] = status
        if error is not None:
            self.receipt["error"] = error
        write_json(self.receipt_path, self.receipt)

    def _install_counter(self) -> None:
        def count(_module: Any, _inputs: tuple[Any, ...]) -> None:
            self.forward_count += 1
            if self.forward_count > self.args.max_forwards:
                raise RuntimeError("native batch-forward budget exhausted")
            if time.monotonic() - self.started > self.args.max_seconds:
                raise RuntimeError("wall-time budget exhausted")

        self.handles.append(self.model.register_forward_pre_hook(count))

    def _variant_input(self, batch: Any, raw: list[dict[str, Any]], state: dict[str, Any], variant: dict[str, Any]) -> tuple[dict[str, Any], list[int]]:
        boundary = _find_boundary(self.panel, state["boundary_id"])
        target_index = int(boundary["batch_index"])
        site = state["site"]
        horizon = int(state["runtime"]["full_prefix_horizon"])
        old = COORD_BASE + int(site["value"])
        new = int(variant["token_id"])
        mutation = None if int(variant["delta"]) == 0 else (target_index, int(site["offset"]), old, new)
        inputs = full_prefix(
            batch,
            raw,
            horizon,
            int(self.q.tokenizer.pad_token_id),
            self.device,
            mutation=mutation,
        )
        prompt_width = int(batch.inputs["input_ids"].shape[1])
        positions = [
            prompt_width + int(state["score_sites"][name]["logit_index"])
            for name in ("immediate", "later")
        ]
        inputs["logits_to_keep"] = torch.tensor(positions, device=self.device, dtype=torch.long)
        inputs["use_cache"] = False
        inputs["return_dict"] = True
        return inputs, positions

    def _run_forward(self, batch: Any, raw: list[dict[str, Any]], state: dict[str, Any], variant: dict[str, Any], instrument: bool) -> dict[str, Any]:
        boundary = _find_boundary(self.panel, state["boundary_id"])
        target_index = int(boundary["batch_index"])
        inputs, positions = self._variant_input(batch, raw, state, variant)
        layer_inputs: dict[str, torch.Tensor] = {}
        layer_outputs: dict[str, torch.Tensor] = {}
        norm_inputs: list[torch.Tensor] = []
        head_inputs: list[torch.Tensor] = []
        handles: list[Any] = []
        if instrument:
            def layer_hook(index: int):
                def hook(_module: Any, values: tuple[Any, ...], output: Any) -> None:
                    value = values[0]
                    result = output[0] if isinstance(output, (tuple, list)) else output
                    if not isinstance(value, torch.Tensor) or not isinstance(result, torch.Tensor):
                        raise TypeError("decoder layer hook did not receive tensors")
                    layer_inputs[str(index)] = value[target_index, positions].detach().cpu().clone()
                    layer_outputs[str(index)] = result[target_index, positions].detach().cpu().clone()
                return hook

            for index, layer in enumerate(self.layers):
                handles.append(layer.register_forward_hook(layer_hook(index)))

            def norm_hook(_module: Any, values: tuple[Any, ...]) -> None:
                value = values[0]
                if not isinstance(value, torch.Tensor):
                    raise TypeError("final norm hook did not receive tensor")
                norm_inputs.append(value[target_index, positions].detach().cpu().clone())

            def head_hook(_module: Any, values: tuple[Any, ...]) -> None:
                value = values[0]
                if not isinstance(value, torch.Tensor):
                    raise TypeError("head hook did not receive tensor")
                head_inputs.append(value[target_index].detach().cpu().clone())

            handles.append(self.norm.register_forward_pre_hook(norm_hook))
            handles.append(self.head.register_forward_pre_hook(head_hook))
        try:
            with torch.no_grad():
                output = self.model(**inputs)
        finally:
            for handle in handles:
                handle.remove()
        logits = output.logits[target_index].detach().cpu()
        if logits.shape[0] != 2:
            raise ValueError("logits_to_keep did not return both score sites")
        input_ids = inputs["input_ids"][target_index].detach().cpu()
        coord = logits[:, COORD_BASE : COORD_BASE + COORD_COUNT]
        top = torch.topk(logits, 2, dim=1)
        logprob = torch.log_softmax(logits.double(), dim=1)
        family = torch.exp(torch.logsumexp(coord.double(), dim=1) - torch.logsumexp(logits.double(), dim=1))
        return {
            "input_ids": input_ids,
            "input_ids_hash": tensor_hash(input_ids),
            "logits": logits,
            "coordinate_logits": coord,
            "top2": [
                {"token_ids": [int(x) for x in ids], "logits": [float(x) for x in values]}
                for ids, values in zip(top.indices, top.values, strict=True)
            ],
            "winner": [int(x) for x in top.indices[:, 0]],
            "margin": [float(x) for x in (top.values[:, 0] - top.values[:, 1])],
            "eos_probability": [float(x) for x in torch.exp(logprob[:, EOS])],
            "coordinate_family_probability": [float(x) for x in family],
            "positions": positions,
            "layer_inputs": layer_inputs,
            "layer_outputs": layer_outputs,
            "layer_residuals": {
                key: layer_outputs[key] - layer_inputs[key] for key in layer_inputs
            },
            "norm_inputs": norm_inputs[0] if norm_inputs else None,
            "head_inputs": head_inputs[0] if head_inputs else None,
        }

    def _check_source_parity(self, state: dict[str, Any], raw: list[dict[str, Any]], trace: dict[str, Any], result: dict[str, Any]) -> None:
        boundary = _find_boundary(self.panel, state["boundary_id"])
        index = int(boundary["batch_index"])
        for column, name in enumerate(("immediate", "later")):
            offset = int(state["score_sites"][name]["offset"])
            expected = _trace_expectation(trace, index, offset)
            if result["winner"][column] != expected["winner"]:
                raise RuntimeError(f"native/full-prefix winner mismatch at {name}")
            for actual, wanted in zip(result["top2"][column]["logits"], [x[1] for x in expected["top2"]], strict=True):
                if abs(actual - wanted) > PARITY_ATOL:
                    raise RuntimeError(f"native/full-prefix top2 mismatch at {name}: {actual} vs {wanted}")

    def run_state(self, state: dict[str, Any]) -> None:
        boundary = _find_boundary(self.panel, state["boundary_id"])
        source_panel_path = self.source_panel_paths[state["boundary_id"]]
        source_panel = self.source_panels[str(source_panel_path)]
        batch, raw, trace, _group, planning = _source(boundary, self.args.model, source_panel, self.q, self.device)
        if planning.get("replanned_image_plan"):
            correction_root = self.output / "runtime" / "source-plan-correction"
            correction_root.mkdir(parents=True, exist_ok=True)
            correction_path = correction_root / f"{self.args.model}-{state['boundary_id']}.json"
            correction = {
                "schema": "coordinate_continuity.source_plan_correction.v1",
                "status": "candidate_complete",
                "model": self.args.model,
                "boundary_id": state["boundary_id"],
                "group": _group["key"],
                "source_panel": _binding(source_panel_path),
                "raw": _binding(Path(boundary["raw_path"])),
                "trace": _binding(Path(boundary["trace_path"])),
                "receipt": _binding(Path(boundary["receipt_path"])),
                **planning,
                "identity_equal": planning["receipt_input_identity"] == planning["replayed_input_identity"],
            }
            if correction_path.exists():
                if json.loads(correction_path.read_text()) != correction:
                    raise ValueError(f"source plan correction differs: {correction_path}")
            else:
                write_json(correction_path, correction)
        target_index = int(boundary["batch_index"])
        site = state["site"]
        native_old = COORD_BASE + int(site["value"])
        state_root = self.output / "runtime" / state["id"]
        state_root.mkdir(parents=True, exist_ok=False)
        for variant in state["variants"]:
            variant_root = state_root / variant["name"]
            variant_root.mkdir()
            emb = self.input_embeddings(
                torch.tensor([native_old, int(variant["token_id"])], device=self.device)
            ).detach().cpu()
            input_delta = emb[1] - emb[0]
            results: dict[str, dict[str, Any]] = {}
            for mode in ("native_no_hook", "observational_hook"):
                result = self._run_forward(batch, raw, state, variant, mode == "observational_hook")
                if variant["name"] == "native" and mode == "native_no_hook":
                    self._check_source_parity(state, raw, trace, result)
                result["input_delta"] = input_delta
                result["variant"] = variant
                result["mode"] = mode
                result["source_target_index"] = target_index
                results[mode] = result
                mode_root = variant_root / mode
                mode_root.mkdir()
                torch.save(result, mode_root / "capture.pt")
            no_hook = results["native_no_hook"]
            hook = results["observational_hook"]
            parity = {
                "logits_max_abs": float((no_hook["logits"] - hook["logits"]).abs().max()),
                "coordinate_logits_max_abs": float((no_hook["coordinate_logits"] - hook["coordinate_logits"]).abs().max()),
                "winner_equal": no_hook["winner"] == hook["winner"],
            }
            if parity["logits_max_abs"] > PARITY_ATOL or not parity["winner_equal"]:
                raise RuntimeError(f"observational hook parity failed for {state['id']} {variant['name']}")
            write_json(
                variant_root / "receipt.json",
                {
                    "schema": "coordinate_continuity.variant_receipt.v1",
                    "status": "candidate_complete",
                    "state_id": state["id"],
                    "variant": variant,
                    "modes": {
                        mode: {
                            "path": str((variant_root / mode / "capture.pt").resolve()),
                            "sha256": hashlib.sha256((variant_root / mode / "capture.pt").read_bytes()).hexdigest(),
                        }
                        for mode in results
                    },
                    "hook_parity": parity,
                    "source": {"boundary_id": state["boundary_id"], "native_token_hash": token_hash([int(x) for x in boundary["native_tokens"]]), "site": site},
                },
            )
        write_json(state_root / "release.json", {"schema": "coordinate_continuity.state_release.v1", "status": "candidate_complete", "state_id": state["id"], "variants": [variant["name"] for variant in state["variants"]]})

    def run(self) -> None:
        try:
            for state in self.states:
                self.run_state(state)
            self.persist("candidate_complete")
        except BaseException as exc:
            self.persist("technical_invalid", repr(exc))
            raise
        finally:
            for handle in self.handles:
                handle.remove()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", choices=("tied", "untied"), required=True)
    parser.add_argument("--source-panel", type=Path)
    parser.add_argument("--state-id", dest="state_ids", action="append", default=[])
    parser.add_argument("--device", default="cuda:7")
    parser.add_argument("--max-forwards", type=int, default=512)
    parser.add_argument("--max-seconds", type=float, default=7200)
    args = parser.parse_args()
    runtime = Runtime(args)
    runtime.run()
    print(json.dumps({"status": "candidate_complete", "model": args.model, "states": len(runtime.states), "model_forwards": runtime.forward_count}, indent=2))


if __name__ == "__main__":
    main()
