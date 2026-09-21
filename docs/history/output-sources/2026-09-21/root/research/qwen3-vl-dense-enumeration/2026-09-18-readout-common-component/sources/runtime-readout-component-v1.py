"""Fixed-panel readout-component continuation runtime.

The parent worker owns the 25-boundary/100-job plan and reduction.  This entry
only reopens those native histories, applies one declared readout operator, and
persists target-only continuation evidence.  It deliberately reuses the
qualified native prefix builder and row parser from the accepted predecessor.
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
from transformers import (
    GenerationConfig,
    LogitsProcessor,
    LogitsProcessorList,
    StoppingCriteria,
    StoppingCriteriaList,
)

from probes.training_set_completion.numerical_feedback.runtime import full_prefix
from probes.training_set_completion.numerical_feedback.select import rows as parsed_rows
from probes.training_set_completion.readout_norm_fresh import _binding, _input_identity
from probes.training_set_completion.untied_shared import load_model
from src.inference.bound_requests import build_bound_native_requests
from src.qwen.native import prepare_native_inputs


EOS = 151645
COORD = 151670
RELEASE_ROWS = 32
RELEASE_TOKENS = 512
OPERATORS = ("original", "full", "shared", "centered")


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def token_hash(tokens: list[int]) -> str:
    return hashlib.sha256(
        json.dumps(tokens, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def tensor_hash(value: torch.Tensor) -> str:
    data = value.detach().cpu().contiguous()
    raw = data.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _finite_list(value: torch.Tensor) -> list[float]:
    values = value.detach().float().cpu().tolist()
    if not all(torch.isfinite(torch.tensor(item)) for item in values):
        raise ValueError("nonfinite readout trace")
    return [float(item) for item in values]


def _safe_name(value: str) -> str:
    if not value or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for char in value):
        raise ValueError(f"unsafe output job id: {value!r}")
    return value


def _trim_target(tokens: list[int]) -> tuple[list[int], str]:
    if EOS in tokens:
        return tokens[: tokens.index(EOS) + 1], "eos"
    complete = len(parsed_rows(tokens))
    if complete >= RELEASE_ROWS:
        return tokens, "rows"
    if len(tokens) >= RELEASE_TOKENS:
        return tokens[:RELEASE_TOKENS], "cap"
    return tokens, "unknown"


def _token_family(token: int) -> str:
    return "coordinate" if COORD <= int(token) < COORD + 1000 else "other"


def _source_path(boundary: dict[str, Any], key: str) -> Path:
    value = boundary.get(f"{key}_path")
    if not value:
        raise KeyError(f"boundary is missing {key}_path")
    return Path(value)


class Runtime:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.plan_path = args.plan
        self.output = args.output
        self.output.mkdir(parents=True, exist_ok=False)
        self.started = time.monotonic()
        self.panel = json.loads(args.panel.read_text())
        self.plan = json.loads(self.plan_path.read_text())
        self.jobs = self._normalise_jobs(self.plan)
        if not self.jobs:
            raise ValueError("readout plan has no jobs")
        models = {job["boundary"]["model"] for job in self.jobs}
        declared_model = self.plan.get("model")
        if declared_model is not None:
            models.add(declared_model)
        if len(models) != 1 or next(iter(models)) not in {"tied", "untied"}:
            raise ValueError("one runtime shard must contain exactly one model")
        self.model_name = next(iter(models))
        if any(job["boundary"]["model"] != self.model_name for job in self.jobs):
            raise ValueError("job model differs from shard model")
        if len(self.jobs) > 100:
            raise ValueError("readout unit allows at most 100 target continuations")
        for job in self.jobs:
            if job["operator"] not in OPERATORS:
                raise ValueError(f"unsupported readout operator: {job['operator']}")

        self.ledger: dict[str, Any] = {
            "schema": "readout_common_component.runtime_receipt.v1",
            "status": "running",
            "pid": os.getpid(),
            "model": self.model_name,
            "plan": _binding(self.plan_path),
            "panel": _binding(args.panel),
            "producer": _binding(Path(__file__)),
            "jobs_planned": [job["id"] for job in self.jobs],
            "operators": list(OPERATORS),
            "model_forwards": 0,
            "vision_forwards": 0,
            "tensor_bytes": 0,
            "artifacts": [],
            "job_results": [],
        }
        self.handles: list[Any] = []
        self.group_cache: dict[str, tuple[dict[str, Any], Any]] = {}
        self.last: dict[str, torch.Tensor] = {}
        self._saved_weights = False
        self.q, self.ledger["identity"] = load_model(self.model_name, args.device)
        self.model = self.q.model
        self.head = self.model.get_output_embeddings()
        selected = self.head.selected_token_ids
        if selected[4:].tolist() != list(range(COORD, COORD + 1000)):
            raise ValueError("unexpected coordinate token IDs")
        self.ids = selected[4:]
        self.U = (
            self.head.base.weight[selected] + self.head.shared_embed_delta
        ).detach()[4:]
        norms = self.U.double().norm(dim=1)
        if bool((norms == 0).any()):
            raise ValueError("effective coordinate output row has zero norm")
        self.alpha = (norms.median() / norms).to(device=args.device)
        self.mu = self.U.double().mean(dim=0).to(device=args.device)
        self.versions = {
            name: parameter._version for name, parameter in self.model.named_parameters()
        }

        def count(_module: Any, _inputs: tuple[Any, ...]) -> None:
            self.ledger["model_forwards"] += 1
            if self.ledger["model_forwards"] > args.max_forwards:
                raise RuntimeError("model-forward budget exhausted")
            if time.monotonic() - self.started > args.max_seconds:
                raise RuntimeError("allocated wall-time budget exhausted")

        def vision(_module: Any, _inputs: tuple[Any, ...]) -> None:
            self.ledger["vision_forwards"] += 1

        def hidden(_module: Any, inputs: tuple[Any, ...]) -> None:
            value = inputs[0]
            if not isinstance(value, torch.Tensor):
                raise TypeError("lm-head input hook did not receive a tensor")
            self.last["head_input"] = value[:, -1, :].detach()

        def logits(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
            value = output[0] if isinstance(output, tuple) else output
            if not isinstance(value, torch.Tensor):
                raise TypeError("lm-head output hook did not receive a tensor")
            self.last["logits"] = value[:, -1, :].detach()

        self.handles = [
            self.model.register_forward_pre_hook(count),
            self.model.model.visual.register_forward_pre_hook(vision),
            self.head.register_forward_pre_hook(hidden),
            self.head.register_forward_hook(logits),
        ]
        self._save_weights()
        self.persist()

    def _normalise_jobs(self, plan: dict[str, Any]) -> list[dict[str, Any]]:
        if plan.get("schema") not in {
            None,
            "readout_common_component.job_plan.v1",
        }:
            raise ValueError("unsupported readout job-plan schema")
        selection_spec = plan.get("selection")
        if isinstance(selection_spec, str):
            selection_path = Path(selection_spec)
            selection_binding = next(
                (
                    item
                    for item in plan.get("bindings", [])
                    if item.get("path") == str(selection_path)
                ),
                None,
            )
        elif isinstance(selection_spec, dict):
            selection_path = Path(selection_spec["path"])
            selection_binding = selection_spec
        else:
            selection_path = None
            selection_binding = None
        selection = None
        if selection_path:
            actual = _binding(selection_path)
            if selection_binding is not None and any(
                actual[key] != selection_binding[key]
                for key in ("path", "sha256")
                if key in selection_binding
            ):
                raise ValueError("selection binding changed")
            selection = json.loads(selection_path.read_text())
        jobs: list[dict[str, Any]] = []
        seen: set[str] = set()
        planned_jobs = plan.get("jobs") or plan.get("cells") or []
        requested_ids = set(self.args.cell_ids or [])
        if requested_ids:
            planned_jobs = [item for item in planned_jobs if item.get("id") in requested_ids]
            if len(planned_jobs) != len(requested_ids):
                present = {item.get("id") for item in planned_jobs}
                raise ValueError(
                    f"requested cell IDs are absent from plan: {sorted(requested_ids - present)}"
                )
        for raw_job in planned_jobs:
            job = dict(raw_job)
            job_id = str(job.get("id", ""))
            if not job_id or job_id in seen:
                raise ValueError("job IDs must be nonempty and unique")
            seen.add(job_id)
            boundary = job.get("boundary")
            if boundary is None:
                boundary_id = job.get("boundary_id")
                if selection is None or not boundary_id:
                    raise ValueError(f"job {job_id} has no boundary")
                matches = [
                    item for item in selection.get("boundaries", []) if item["id"] == boundary_id
                ]
                if len(matches) != 1:
                    raise ValueError(f"job {job_id} boundary is not unique")
                boundary = matches[0]
            job["boundary"] = boundary
            job["operator"] = str(
                job.get("operator", job.get("policy", job.get("arm", "")))
            )
            jobs.append(job)
        return jobs

    def _save_weights(self) -> None:
        if self._saved_weights:
            return
        path = self.output / "effective-readout.pt"
        torch.save(
            {
                "coordinate_ids": self.ids.detach().cpu(),
                "output_rows": self.U.detach().cpu(),
                "factors": self.alpha.detach().cpu(),
                "mu": self.mu.detach().cpu(),
                "formula": {
                    "full": "alpha*z",
                    "shared": "z+(alpha-1)*b",
                    "centered": "z+(alpha-1)*(z-b)",
                    "b": "mu dot h",
                },
            },
            path,
        )
        self._record_artifact(path)
        self.ledger["readout_weights"] = _binding(path)
        self._saved_weights = True

    def _record_artifact(self, path: Path) -> None:
        size = path.stat().st_size
        self.ledger["tensor_bytes"] += size
        if self.ledger["tensor_bytes"] > self.args.max_bytes:
            raise RuntimeError("tensor payload budget exhausted")
        self.ledger["artifacts"].append(_binding(path))

    def persist(self) -> None:
        self.ledger["gpu_seconds"] = time.monotonic() - self.started
        write(self.output / "shard-receipt.json", self.ledger)

    def _config(self, group: dict[str, Any]) -> dict[str, Any]:
        config = dict(self.panel["configs"][self.model_name])
        config["data"] = dict(input_jsonl=group["input_jsonl"])
        return config

    def _group(self, key: str) -> dict[str, Any]:
        return next(group for group in self.panel["groups"] if group["key"] == key)

    def _prepare_group(self, group: dict[str, Any]) -> Any:
        cached = self.group_cache.get(group["key"])
        if cached is not None:
            return cached[1]
        requests, _ = build_bound_native_requests(
            self.q, self._config(group), group["cases"]
        )
        batch = prepare_native_inputs(
            self.q.processor,
            requests,
            device=self.args.device,
            record_media_identity=True,
        )
        self.group_cache[group["key"]] = (group, batch)
        return batch

    def _source(
        self, boundary: dict[str, Any]
    ) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any], Any]:
        group = self._group(boundary["group"])
        raw_path = _source_path(boundary, "raw")
        trace_path = _source_path(boundary, "trace")
        receipt_path = _source_path(boundary, "receipt")
        for key, path in (("raw", raw_path), ("trace", trace_path), ("receipt", receipt_path)):
            expected = boundary.get(f"{key}_binding")
            if not path.exists():
                raise FileNotFoundError(path)
            if expected is not None and _binding(path) != expected:
                raise ValueError(f"source {key} binding changed")
        raw = json.loads(raw_path.read_text())["rows"]
        trace = json.loads(trace_path.read_text())
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("status") != "candidate_complete":
            raise ValueError("source natural receipt is not complete")
        if receipt.get("condition") != f"{self.model_name}-original":
            raise ValueError("source natural condition changed")
        if receipt.get("group") != boundary["group"]:
            raise ValueError("source natural group changed")
        batch = self._prepare_group(group)
        if receipt.get("input_identity") != _input_identity(batch):
            raise ValueError("source native input identity changed")
        if receipt.get("raw") != _binding(raw_path) or receipt.get("trace") != _binding(trace_path):
            raise ValueError("source receipt artifact binding changed")
        if len(raw) != len(group["cases"]):
            raise ValueError("source companion count changed")
        index = int(boundary["batch_index"])
        if int(raw[index]["image_id"]) != int(boundary["image_id"]):
            raise ValueError("source batch index/image identity changed")
        native = list(raw[index]["token_ids"])
        if native != list(boundary["native_tokens"]):
            raise ValueError("source native token identity changed")
        if token_hash(native) != boundary["native_token_hash"]:
            raise ValueError("source native token hash changed")
        source_end = int(boundary["source_row"]["end"])
        if not 0 < source_end <= len(native):
            raise ValueError("source prefix is outside native history")
        return group, raw, trace, receipt, batch

    def _operator_scores(
        self, scores: torch.Tensor, head_input: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        z = scores[:, self.ids].double()
        b = head_input.double() @ self.mu
        delta = self.alpha - 1
        full = z * self.alpha
        shared = z + delta[None, :] * b[:, None]
        centered = z + delta[None, :] * (z - b[:, None])
        result: dict[str, torch.Tensor] = {"original": scores}
        for name, coordinate in (
            ("full", full),
            ("shared", shared),
            ("centered", centered),
        ):
            transformed = scores.clone()
            transformed[:, self.ids] = coordinate.to(scores.dtype)
            result[name] = transformed
        return result, b, z

    def _trace_operator(
        self,
        name: str,
        score: torch.Tensor,
        coordinate: torch.Tensor,
        chosen: int,
    ) -> dict[str, Any]:
        logprob = torch.log_softmax(score.double(), dim=0)
        top = torch.topk(score, 2)
        family = torch.exp(
            torch.logsumexp(coordinate.double(), dim=0)
            - torch.logsumexp(score.double(), dim=0)
        )
        return {
            "top2": [
                {"token_id": int(token), "logit": float(value)}
                for token, value in zip(top.indices.tolist(), top.values.tolist(), strict=True)
            ],
            "margin": float((top.values[0] - top.values[1]).item()),
            "eos_probability": float(torch.exp(logprob[EOS]).item()),
            "coordinate_family_probability": float(family.item()),
            "chosen_token": int(chosen),
            "winner_family": _token_family(int(top.indices[0].item())),
        }

    def _qualification_step(
        self,
        scores: torch.Tensor,
        transformed: dict[str, torch.Tensor],
        z: torch.Tensor,
        b: torch.Tensor,
    ) -> dict[str, Any]:
        noncoord = torch.ones(scores.shape[-1], dtype=torch.bool, device=scores.device)
        noncoord[self.ids] = False
        full_fp64 = z * self.alpha
        shared_fp64 = z + (self.alpha - 1)[None, :] * b[:, None]
        centered_fp64 = z + (self.alpha - 1)[None, :] * (z - b[:, None])
        full_reference = full_fp64.to(scores.dtype)
        formula_error = float(
            (transformed["full"][:, self.ids] - full_reference).abs().max().item()
        )
        increments = {
            "full": full_fp64 - z,
            "shared": shared_fp64 - z,
            "centered": centered_fp64 - z,
        }
        reconstruction = float(
            (increments["shared"] + increments["centered"] - increments["full"]).abs().max().item()
        )
        unchanged = {
            name: bool(torch.equal(transformed[name][:, noncoord], scores[:, noncoord]))
            for name in OPERATORS
        }
        identity = bool(torch.equal(transformed["original"], scores))
        sensitivity = float((increments["full"].abs().max()).item())
        return {
            "identity_operator_bitwise": identity,
            "full_formula_max_abs_error": formula_error,
            "shared_plus_centered_increment_max_abs_error": reconstruction,
            "noncoordinate_bitwise_unchanged": unchanged,
            "max_full_coordinate_sensitivity": sensitivity,
            "b_max_abs": float(b.abs().max().item()),
        }

    def _run_job(self, job: dict[str, Any]) -> dict[str, Any]:
        forwards_before = int(self.ledger["model_forwards"])
        vision_before = int(self.ledger["vision_forwards"])
        seconds_before = time.monotonic() - self.started
        boundary = job["boundary"]
        operator = job["operator"]
        group, raw, _trace, receipt, batch = self._source(boundary)
        target_index = int(boundary["batch_index"])
        native = list(boundary["native_tokens"])
        source_end = int(boundary["source_row"]["end"])
        pad = int(self.q.tokenizer.pad_token_id)
        prefix = native[:source_end]
        if pad in prefix:
            raise ValueError("target source prefix contains padding")
        inputs = full_prefix(batch, raw, source_end, pad, self.args.device)
        width = int(inputs["input_ids"].shape[1])
        if int(inputs["attention_mask"][target_index].sum().item()) != width:
            raise ValueError("target native history was padded")
        trace: list[dict[str, Any]] = []
        head_inputs: list[torch.Tensor] = []
        raw_coordinates: list[torch.Tensor] = []
        shadow_coordinates: list[torch.Tensor] = []
        qualification: dict[str, Any] | None = None
        runtime = self

        class Policy(LogitsProcessor):
            def __call__(_self, tokens: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
                nonlocal qualification
                head = runtime.last.get("head_input")
                if head is None or head.ndim != 2 or scores.ndim != 2:
                    raise RuntimeError("missing aligned lm-head input at generation step")
                transformed, b, z = runtime._operator_scores(scores, head)
                used = transformed[operator]
                target_head = head[target_index].detach()
                target_raw = scores[target_index, runtime.ids].detach()
                target_shadows = torch.stack(
                    [transformed[name][target_index, runtime.ids].detach() for name in OPERATORS]
                )
                choices = used.argmax(dim=-1)
                offset = int(tokens.shape[1] - width)
                raw_winner = int(scores[target_index].argmax().item())
                raw_winner_family = _token_family(raw_winner)
                operator_trace = {
                    name: runtime._trace_operator(
                        name,
                        transformed[name][target_index],
                        target_shadows[index],
                        int(choices[target_index].item())
                        if name == operator
                        else int(transformed[name][target_index].argmax().item()),
                    )
                    for index, name in enumerate(OPERATORS)
                }
                for name, item in operator_trace.items():
                    winner = int(item["top2"][0]["token_id"])
                    item["flip_class"] = (
                        None
                        if winner == raw_winner
                        else "coordinate_to_coordinate"
                        if raw_winner_family == "coordinate" and _token_family(winner) == "coordinate"
                        else "coordinate_family_switch"
                        if raw_winner_family != _token_family(winner)
                        else "other_token_switch"
                    )
                target_trace = {
                    "offset": offset,
                    "input_token_id": int(tokens[target_index, -1].item()),
                    "raw_winner_token": raw_winner,
                    "raw_winner_family": raw_winner_family,
                    "raw_top2": runtime._trace_operator(
                        "original",
                        scores[target_index],
                        z[target_index],
                        raw_winner,
                    ),
                    "operators": operator_trace,
                    "applied_operator": operator,
                    "chosen_token": int(choices[target_index].item()),
                    "operator_dtype": str(scores.dtype),
                }
                trace.append(target_trace)
                head_inputs.append(target_head.cpu())
                raw_coordinates.append(target_raw.cpu())
                shadow_coordinates.append(target_shadows.cpu())
                if job.get("qualification") and qualification is None:
                    qualification = runtime._qualification_step(scores, transformed, z, b)
                return used

        stop_state: dict[str, Any] = {"complete_rows": 0, "eos": False}

        class TargetStop(StoppingCriteria):
            def __call__(_self, ids: torch.Tensor, _scores: Any, **_kwargs: Any) -> bool:
                target = ids[target_index, width:].tolist()
                stop_state["complete_rows"] = len(parsed_rows(target))
                stop_state["eos"] = EOS in target
                return bool(
                    stop_state["eos"] or stop_state["complete_rows"] >= RELEASE_ROWS
                )

        result = self.generate(inputs, processors=[Policy()], stopping=[TargetStop()])
        target_tokens, stop = _trim_target(result[target_index, width:].tolist())
        if len(target_tokens) > RELEASE_TOKENS:
            raise RuntimeError("target continuation exceeded token cap")
        complete = len(parsed_rows(target_tokens))
        if complete > RELEASE_ROWS:
            raise RuntimeError("target continuation exceeded row cap")
        for index, step in enumerate(trace):
            if index >= len(target_tokens):
                raise RuntimeError("trace has more steps than emitted target tokens")
            step["emitted_token"] = int(target_tokens[index])
            step["choice_matches_emitted"] = step["chosen_token"] == step["emitted_token"]
        if any(not step["choice_matches_emitted"] for step in trace):
            raise RuntimeError("operator choice and emitted token diverged")
        if not trace and not target_tokens:
            raise RuntimeError("empty target continuation produced no trace")

        job_dir = self.output / _safe_name(str(job["id"]))
        job_dir.mkdir(parents=False, exist_ok=False)
        tensors = {
            "schema": "readout_common_component.trace_tensors.v1",
            "job_id": job["id"],
            "operator": operator,
            "head_inputs": torch.stack(head_inputs) if head_inputs else torch.empty((0, self.U.shape[1])),
            "raw_coordinate_logits": torch.stack(raw_coordinates) if raw_coordinates else torch.empty((0, 1000)),
            "operator_coordinate_logits": torch.stack(shadow_coordinates) if shadow_coordinates else torch.empty((0, 4, 1000)),
            "target_tokens": torch.tensor(target_tokens, dtype=torch.long),
        }
        tensor_path = job_dir / "trajectory.pt"
        torch.save(tensors, tensor_path)
        self._record_artifact(tensor_path)
        persisted = torch.load(tensor_path, map_location="cpu", weights_only=False)
        persistence = {
            "path": _binding(tensor_path),
            "readback_schema": persisted.get("schema"),
            "head_inputs_sha256": tensor_hash(persisted["head_inputs"]),
            "raw_coordinate_logits_sha256": tensor_hash(persisted["raw_coordinate_logits"]),
            "operator_coordinate_logits_sha256": tensor_hash(persisted["operator_coordinate_logits"]),
            "target_tokens_sha256": tensor_hash(persisted["target_tokens"]),
            "readback_passed": bool(
                persisted["target_tokens"].tolist() == target_tokens
                and persisted["operator"] == operator
            ),
        }
        if not persistence["readback_passed"]:
            raise RuntimeError("trace tensor persistence/readback mismatch")
        payload = {
            "schema": "readout_common_component.job_result.v1",
            "status": "candidate_complete",
            "job_id": job["id"],
            "operator": operator,
            "policy": operator,
            "model": self.model_name,
            "group": boundary["group"],
            "boundary_id": boundary["id"],
            "image_id": int(boundary["image_id"]),
            "batch_index": target_index,
            "source": {
                "source_end_exclusive": source_end,
                "native_token_hash": token_hash(native),
                "prefix_token_hash": token_hash(prefix),
                "target_prefix_unpadded": True,
                "source_row": boundary["source_row"],
                "target_slots": boundary.get("target_slots", []),
            },
            "target": {
                "token_ids": target_tokens,
                "token_hash": token_hash(target_tokens),
                "text": self.q.tokenizer.decode(
                    target_tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False
                ),
                "stop": {
                    "reason": stop,
                    "complete_rows": complete,
                    "token_count": len(target_tokens),
                    "row_cap": RELEASE_ROWS,
                    "token_cap": RELEASE_TOKENS,
                    "parser": "select.rows",
                },
            },
            "trace": {
                "steps": trace,
                "step_count": len(trace),
                "first_divergence_source": next(
                    (index for index, step in enumerate(trace) if step["emitted_token"] != step["operators"]["original"]["chosen_token"]),
                    None,
                ),
            },
            "persistence": persistence,
            "qualification": qualification,
            "source_receipt": _binding(_source_path(boundary, "receipt")),
        }
        result_path = job_dir / "release.json"
        write(result_path, payload)
        receipt_path = job_dir / "receipt.json"
        cell_receipt = {
            "schema": "readout_common_component.cell_receipt.v1",
            "status": payload["status"],
            "job_id": job["id"],
            "boundary_id": boundary["id"],
            "policy": operator,
            "model": self.model_name,
            "group": boundary["group"],
            "producer": _binding(Path(__file__)),
            "panel": _binding(self.args.panel),
            "plan": _binding(self.plan_path),
            "source_receipt": payload["source_receipt"],
            "release": _binding(result_path),
            "trajectory": persistence["path"],
            "model_forwards": int(self.ledger["model_forwards"]) - forwards_before,
            "vision_forwards": int(self.ledger["vision_forwards"]) - vision_before,
            "tensor_bytes": persistence["path"]["size_bytes"],
            "gpu_seconds": (time.monotonic() - self.started) - seconds_before,
            "stop": payload["target"]["stop"],
            "qualification": qualification,
        }
        write(receipt_path, cell_receipt)
        self.ledger["artifacts"].append(_binding(receipt_path))
        self.ledger["job_results"].append(
            {
                "job_id": job["id"],
                "operator": operator,
                "boundary_id": boundary["id"],
                "status": payload["status"],
                "release": _binding(result_path),
                "trace": persistence["path"],
                "model_forwards_after_job": self.ledger["model_forwards"],
                "stop": payload["target"]["stop"],
            }
        )
        self.persist()
        return payload

    def generate(
        self,
        inputs: dict[str, Any],
        processors: list[Any],
        stopping: list[Any],
    ) -> torch.Tensor:
        config = GenerationConfig(
            max_new_tokens=RELEASE_TOKENS,
            do_sample=False,
            repetition_penalty=1,
            eos_token_id=EOS,
            pad_token_id=self.q.tokenizer.pad_token_id,
        )
        with torch.inference_mode():
            return self.model.generate(
                **inputs,
                generation_config=config,
                use_model_defaults=False,
                logits_processor=LogitsProcessorList(processors),
                stopping_criteria=StoppingCriteriaList(stopping),
            )

    def run(self) -> None:
        for job in self.jobs:
            self._run_job(job)
        self.ledger["status"] = "candidate_complete"
        self.ledger["jobs_completed"] = len(self.ledger["job_results"])
        self.persist()

    def close(self, error: BaseException | None = None) -> None:
        for handle in self.handles:
            handle.remove()
        if hasattr(self, "model") and hasattr(self, "versions"):
            current = {
                name: parameter._version for name, parameter in self.model.named_parameters()
            }
            if current != self.versions:
                error = error or RuntimeError("parameter mutation detected")
        if error is not None:
            self.ledger["status"] = "partial_candidate"
            self.ledger["error"] = repr(error)
        self.ledger["live_jobs"] = []
        self.ledger["ended_pid"] = os.getpid()
        self.persist()


def selfcheck() -> None:
    generator = torch.Generator().manual_seed(7)
    z = torch.randn(2, 1000, generator=generator, dtype=torch.float32)
    h = torch.randn(2, 8, generator=generator, dtype=torch.float32)
    w = torch.randn(1000, 8, generator=generator, dtype=torch.float32)
    norms = w.double().norm(dim=1)
    alpha = norms.median() / norms
    mu = w.double().mean(dim=0)
    b = h.double() @ mu
    full = z.double() * alpha
    shared = z.double() + (alpha - 1)[None, :] * b[:, None]
    centered = z.double() + (alpha - 1)[None, :] * (z.double() - b[:, None])
    assert torch.equal((full - z.double()), (shared - z.double()) + (centered - z.double())) is False or torch.allclose(
        full - z.double(), (shared - z.double()) + (centered - z.double()), atol=1e-12, rtol=0
    )
    scores = torch.randn(2, 2000, generator=generator)
    ids = torch.arange(1000) + 1000
    transformed = scores.clone()
    transformed[:, ids] = full.to(scores.dtype)
    mask = torch.ones(2000, dtype=torch.bool)
    mask[ids] = False
    assert torch.equal(scores[:, mask], transformed[:, mask])
    assert parsed_rows([151646, 9, 151647, 151648, COORD, COORD + 1, COORD + 2, COORD + 3, 151649] * 3) == [
        {
            "index": 0,
            "start": 0,
            "end": 9,
            "description_tokens": [9],
            "coordinate_offsets": [4, 5, 6, 7],
            "values": [0, 1, 2, 3],
            "valid": True,
        },
        {
            "index": 1,
            "start": 9,
            "end": 18,
            "description_tokens": [9],
            "coordinate_offsets": [13, 14, 15, 16],
            "values": [0, 1, 2, 3],
            "valid": True,
        },
        {
            "index": 2,
            "start": 18,
            "end": 27,
            "description_tokens": [9],
            "coordinate_offsets": [22, 23, 24, 25],
            "values": [0, 1, 2, 3],
            "valid": True,
        },
    ]
    print("PASS operator decomposition, noncoordinate identity, and complete-row parser")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["run"], nargs="?")
    parser.add_argument("--panel", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cell-id", action="append", dest="cell_ids")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-forwards", type=int, default=100000)
    parser.add_argument("--max-seconds", type=float, default=8 * 3600)
    parser.add_argument("--max-bytes", type=int, default=16 * 1024**3)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        selfcheck()
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
