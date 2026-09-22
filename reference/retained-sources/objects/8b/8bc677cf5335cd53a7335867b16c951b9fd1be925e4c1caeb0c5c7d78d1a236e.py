"""Direction controls for the qualified readout-component continuation runtime.

This module keeps native input construction, prefix replay, generation, and
row stopping in ``readout_component.runtime``.  It only changes the coordinate
logit operator and records all eleven shadows at each target step.
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
from transformers import LogitsProcessor, StoppingCriteria

from probes.training_set_completion.readout_component import runtime as component
from probes.training_set_completion.numerical_feedback.select import rows as parsed_rows


EOS = component.EOS
COORD = component.COORD
RELEASE_ROWS = component.RELEASE_ROWS
RELEASE_TOKENS = component.RELEASE_TOKENS
POLICIES = (
    "original",
    "full",
    "reflected",
    "sign19",
    "sign20",
    "sign21",
    "sign22",
    "sign23",
    "sign24",
    "sign25",
    "sign26",
)
SIGN_POLICIES = POLICIES[3:]


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _safe_name(value: str) -> str:
    return component._safe_name(value)


def _binding(path: Path) -> dict[str, Any]:
    return component._binding(path)


def _token_hash(tokens: list[int]) -> str:
    return component.token_hash(tokens)


def _tensor_hash(value: torch.Tensor) -> str:
    return component.tensor_hash(value)


def _trim_target(tokens: list[int]) -> tuple[list[int], str]:
    return component._trim_target(tokens)


def _token_family(token: int) -> str:
    return component._token_family(token)


def _hash_signs(signs: list[int]) -> str:
    return hashlib.sha256(
        json.dumps(signs, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _selection_from_plan(plan: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    spec = plan.get("selection")
    if isinstance(spec, str):
        path = Path(spec)
        binding = next(
            (item for item in plan.get("bindings", []) if item.get("path") == str(path)),
            None,
        )
    elif isinstance(spec, dict):
        path = Path(spec["path"])
        binding = spec
    else:
        raise ValueError("direction plan has no selection binding")
    actual = _binding(path)
    if binding is not None and any(
        actual[key] != binding[key] for key in ("path", "sha256") if key in binding
    ):
        raise ValueError("selection binding changed")
    return path, json.loads(path.read_text())


def _sign_manifest(plan: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    spec = plan.get("sign_vectors")
    if not isinstance(spec, dict) or "path" not in spec:
        raise ValueError("direction plan has no sign-vector binding")
    path = Path(spec["path"])
    actual = _binding(path)
    if actual["sha256"] != spec.get("sha256"):
        raise ValueError("sign-vector file binding changed")
    manifest = json.loads(path.read_text())
    if manifest.get("coordinate_ids") != list(range(COORD, COORD + 1000)):
        raise ValueError("sign-vector coordinate IDs changed")
    vectors = {item.get("policy"): item for item in manifest.get("vectors", [])}
    if set(vectors) != set(SIGN_POLICIES):
        raise ValueError("sign-vector policy set changed")
    for policy in SIGN_POLICIES:
        item = vectors[policy]
        signs = item.get("signs")
        if (
            not isinstance(signs, list)
            or len(signs) != 1000
            or any(value not in (-1, 1) for value in signs)
            or signs.count(1) != 500
            or signs.count(-1) != 500
            or _hash_signs(signs) != item.get("sha256")
        ):
            raise ValueError(f"invalid frozen sign vector: {policy}")
    return path, manifest


def _validate_plan(plan_path: Path, panel_path: Path) -> dict[str, Any]:
    """Validate frozen bindings without importing or loading a model."""
    plan = json.loads(plan_path.read_text())
    if plan.get("status") != "frozen_before_gpu":
        raise ValueError("direction plan is not frozen_before_gpu")
    if plan.get("policies") != list(POLICIES):
        raise ValueError("direction policy order changed")
    _sign_path, manifest = _sign_manifest(plan)
    selection_path, selection = _selection_from_plan(plan)
    boundaries = selection.get("boundaries", [])
    if len(boundaries) != 11:
        raise ValueError(f"expected 11 boundaries, found {len(boundaries)}")
    boundary_ids = [str(item["id"]) for item in boundaries]
    if len(set(boundary_ids)) != 11 or any(item.get("kind") != "failure" for item in boundaries):
        raise ValueError("selection must contain eleven unique failure boundaries")
    cells = plan.get("cells", [])
    if len(cells) != 121:
        raise ValueError(f"expected 121 cells, found {len(cells)}")
    seen: set[str] = set()
    coverage: dict[str, set[str]] = {boundary_id: set() for boundary_id in boundary_ids}
    controls = 0
    qualification = 0
    boundary_by_id = {item["id"]: item for item in boundaries}
    for cell in cells:
        cell_id = str(cell.get("id", ""))
        if not cell_id or cell_id in seen:
            raise ValueError("cell IDs must be unique and nonempty")
        seen.add(cell_id)
        boundary_id = cell.get("boundary_id")
        policy = cell.get("policy")
        if boundary_id not in boundary_by_id or policy not in POLICIES:
            raise ValueError(f"cell has invalid boundary or policy: {cell_id}")
        if cell.get("model") != boundary_by_id[boundary_id].get("model"):
            raise ValueError(f"cell model differs from boundary: {cell_id}")
        coverage[boundary_id].add(policy)
        if cell.get("qualification"):
            qualification += 1
        if policy in {"original", "full"}:
            source = cell.get("control_source")
            if not isinstance(source, dict):
                raise ValueError(f"missing saved control source: {cell_id}")
            source_path = Path(source["path"])
            if _binding(source_path)["sha256"] != source.get("sha256"):
                raise ValueError(f"control source binding changed: {cell_id}")
            controls += 1
    if any(values != set(POLICIES) for values in coverage.values()):
        raise ValueError("each boundary must have every direction policy exactly once")
    if qualification != 11 or controls != 22:
        raise ValueError(f"unexpected qualification/control counts: {qualification}/{controls}")
    panel_binding = next(
        (item for item in plan.get("bindings", []) if item.get("path") == str(panel_path)),
        None,
    )
    if panel_binding is not None and _binding(panel_path)["sha256"] != panel_binding.get("sha256"):
        raise ValueError("panel binding changed")
    return {
        "schema": "readout_direction.plan_validation.v1",
        "status": "candidate_cpu_validated",
        "plan": _binding(plan_path),
        "panel": _binding(panel_path),
        "selection": _binding(selection_path),
        "sign_vectors": _binding(_sign_path),
        "boundaries": len(boundaries),
        "cells": len(cells),
        "policies": list(POLICIES),
        "qualification_cells": qualification,
        "saved_controls": controls,
        "sign_vector_hashes": {
            policy: next(item["sha256"] for item in manifest["vectors"] if item["policy"] == policy)
            for policy in SIGN_POLICIES
        },
    }


class Runtime(component.Runtime):
    """Qualified native runtime with an explicit fixed direction operator."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.direction_plan = json.loads(args.plan.read_text())
        _validate_plan(args.plan, args.panel)
        self.direction_sign_manifest_path, self.direction_sign_manifest = _sign_manifest(
            self.direction_plan
        )
        self.direction_policies = tuple(self.direction_plan["policies"])
        super().__init__(args)
        self.direction_signs = {
            item["policy"]: torch.tensor(
                item["signs"], dtype=torch.float64, device=args.device
            )
            for item in self.direction_sign_manifest["vectors"]
        }
        self.ledger.update(
            {
                "schema": "readout_direction.runtime_receipt.v1",
                "operators": list(self.direction_policies),
                "direction_formula": {
                    "full": "z*alpha in FP64 then cast once",
                    "reflected": "z-(alpha-1)*z in FP64 then cast once",
                    "sign": "z+s*(alpha-1)*z in FP64 then cast once",
                    "noncoordinate": "scores copied bitwise unchanged",
                },
                "sign_vectors": _binding(self.direction_sign_manifest_path),
                "sign_vector_hashes": {
                    item["policy"]: item["sha256"]
                    for item in self.direction_sign_manifest["vectors"]
                },
                "coordinate_ids": [COORD, COORD + 999],
                "alpha_min": float(self.alpha.min().item()),
                "alpha_max": float(self.alpha.max().item()),
                "reflected_scale_min": float((2 - self.alpha).min().item()),
            }
        )
        self.persist()

    def _normalise_jobs(self, plan: dict[str, Any]) -> list[dict[str, Any]]:
        """Resolve direction cells while satisfying the base loader contract."""
        selection_path, selection = _selection_from_plan(plan)
        planned = plan.get("jobs") or plan.get("cells") or []
        requested = set(self.args.cell_ids or [])
        if requested:
            planned = [item for item in planned if item.get("id") in requested]
            present = {item.get("id") for item in planned}
            if present != requested:
                raise ValueError(f"requested cell IDs are absent: {sorted(requested - present)}")
        boundaries = {item["id"]: item for item in selection["boundaries"]}
        jobs: list[dict[str, Any]] = []
        seen: set[str] = set()
        for raw in planned:
            job = dict(raw)
            job_id = str(job.get("id", ""))
            policy = str(job.get("policy", job.get("operator", "")))
            boundary_id = str(job.get("boundary_id", ""))
            if not job_id or job_id in seen:
                raise ValueError("direction cell IDs must be unique and nonempty")
            if policy not in self.direction_policies:
                raise ValueError(f"unsupported direction policy: {policy}")
            boundary = job.get("boundary") or boundaries.get(boundary_id)
            if boundary is None:
                raise ValueError(f"cell has no selected boundary: {job_id}")
            if boundary.get("id") != boundary_id:
                raise ValueError(f"cell boundary mismatch: {job_id}")
            if job.get("model") != boundary.get("model"):
                raise ValueError(f"cell model mismatch: {job_id}")
            job["boundary"] = boundary
            job["direction_policy"] = policy
            # The qualified base __init__ validates only its four old names;
            # _run_job below consumes direction_policy explicitly.
            job["operator"] = "original"
            seen.add(job_id)
            jobs.append(job)
        if not jobs:
            raise ValueError("direction plan has no selected cells")
        return jobs

    def _direction_scores(
        self, scores: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Return all shadows; only coordinate columns are ever assigned."""
        z = scores[:, self.ids].double()
        delta = (self.alpha - 1)[None, :] * z
        full = z * self.alpha[None, :]
        coordinates: dict[str, torch.Tensor] = {
            "full": full,
            "reflected": z - delta,
        }
        for policy in SIGN_POLICIES:
            coordinates[policy] = z + self.direction_signs[policy][None, :] * delta
        transformed: dict[str, torch.Tensor] = {"original": scores}
        for policy in POLICIES[1:]:
            value = scores.clone()
            value[:, self.ids] = coordinates[policy].to(scores.dtype)
            transformed[policy] = value
        return transformed, z, delta

    def _qualification_direction(
        self,
        scores: torch.Tensor,
        transformed: dict[str, torch.Tensor],
        z: torch.Tensor,
        delta: torch.Tensor,
    ) -> dict[str, Any]:
        noncoord = torch.ones(scores.shape[-1], dtype=torch.bool, device=scores.device)
        noncoord[self.ids] = False
        full_expected = (z * self.alpha).to(scores.dtype)
        formula_error = float(
            (transformed["full"][:, self.ids] - full_expected).abs().max().item()
        )
        unchanged = {
            policy: bool(torch.equal(transformed[policy][:, noncoord], scores[:, noncoord]))
            for policy in POLICIES
        }
        # Check the FP64 formulas before the required single cast.  The saved
        # shadows are the cast values, so measuring them here would conflate
        # direction equality with ordinary FP32 rounding.
        increments = {
            "full": z * self.alpha[None, :] - z,
            "reflected": -delta,
            **{
                policy: self.direction_signs[policy][None, :] * delta
                for policy in SIGN_POLICIES
            },
        }
        magnitude_error = max(
            float((increments[policy].abs() - delta.abs()).abs().max().item())
            for policy in POLICIES[1:]
        )
        return {
            "identity_operator_bitwise": bool(torch.equal(transformed["original"], scores)),
            "full_formula_max_abs_error": formula_error,
            "direction_magnitude_max_abs_error_fp64": magnitude_error,
            "noncoordinate_bitwise_unchanged": unchanged,
            "sign_balance": {
                policy: {
                    "positive": int((self.direction_signs[policy] == 1).sum().item()),
                    "negative": int((self.direction_signs[policy] == -1).sum().item()),
                }
                for policy in SIGN_POLICIES
            },
            "max_full_coordinate_sensitivity": float((z * (self.alpha - 1)).abs().max().item()),
        }

    def _run_job(self, job: dict[str, Any]) -> dict[str, Any]:
        forwards_before = int(self.ledger["model_forwards"])
        vision_before = int(self.ledger["vision_forwards"])
        seconds_before = time.monotonic() - self.started
        boundary = job["boundary"]
        policy = job["direction_policy"]
        group, raw, _source_trace, _source_receipt, batch = self._source(boundary)
        target_index = int(boundary["batch_index"])
        native = list(boundary["native_tokens"])
        source_end = int(boundary["source_row"]["end"])
        pad = int(self.q.tokenizer.pad_token_id)
        prefix = native[:source_end]
        if len(prefix) != source_end or pad in prefix:
            raise ValueError("target source prefix contains padding")
        inputs = component.full_prefix(batch, raw, source_end, pad, self.args.device)
        suffix = inputs["input_ids"][target_index, -source_end:]
        expected = torch.tensor(prefix, device=suffix.device, dtype=suffix.dtype)
        if not torch.equal(suffix, expected) or pad in suffix.tolist():
            raise ValueError("target native history suffix was padded or changed")
        width = int(inputs["input_ids"].shape[1])
        trace: list[dict[str, Any]] = []
        head_inputs: list[torch.Tensor] = []
        raw_coordinates: list[torch.Tensor] = []
        shadow_coordinates: list[torch.Tensor] = []
        qualification: dict[str, Any] | None = None
        runtime = self

        class Direction(LogitsProcessor):
            def __call__(_self, tokens: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
                nonlocal qualification
                head = runtime.last.get("head_input")
                if head is None or head.ndim != 2 or scores.ndim != 2:
                    raise RuntimeError("missing aligned lm-head input at generation step")
                transformed, z, delta = runtime._direction_scores(scores)
                used = transformed[policy]
                target_head = head[target_index].detach()
                target_raw = scores[target_index, runtime.ids].detach()
                target_shadows = torch.stack(
                    [transformed[name][target_index, runtime.ids].detach() for name in POLICIES]
                )
                choices = used.argmax(dim=-1)
                offset = int(tokens.shape[1] - width)
                raw_winner = int(scores[target_index].argmax().item())
                raw_family = _token_family(raw_winner)
                operator_trace: dict[str, dict[str, Any]] = {}
                for index, name in enumerate(POLICIES):
                    selected = (
                        int(choices[target_index].item())
                        if name == policy
                        else int(transformed[name][target_index].argmax().item())
                    )
                    item = component.Runtime._trace_operator(
                        runtime,
                        name,
                        transformed[name][target_index],
                        target_shadows[index],
                        selected,
                    )
                    winner = int(item["top2"][0]["token_id"])
                    item["flip_class"] = (
                        None
                        if winner == raw_winner
                        else "coordinate_to_coordinate"
                        if raw_family == "coordinate" and _token_family(winner) == "coordinate"
                        else "coordinate_family_switch"
                        if raw_family != _token_family(winner)
                        else "other_token_switch"
                    )
                    operator_trace[name] = item
                target_trace = {
                    "offset": offset,
                    "input_token_id": int(tokens[target_index, -1].item()),
                    "raw_winner_token": raw_winner,
                    "raw_winner_family": raw_family,
                    "raw_top2": component.Runtime._trace_operator(
                        runtime,
                        "original",
                        scores[target_index],
                        z[target_index],
                        raw_winner,
                    ),
                    "operators": operator_trace,
                    "applied_operator": policy,
                    "chosen_token": int(choices[target_index].item()),
                    "operator_dtype": str(scores.dtype),
                    "coordinate_delta_abs_max": float(delta[target_index].abs().max().item()),
                }
                trace.append(target_trace)
                head_inputs.append(target_head.cpu())
                raw_coordinates.append(target_raw.cpu())
                shadow_coordinates.append(target_shadows.cpu())
                if job.get("qualification") and qualification is None:
                    qualification = runtime._qualification_direction(scores, transformed, z, delta)
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

        result = self.generate(inputs, processors=[Direction()], stopping=[TargetStop()])
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
            raise RuntimeError("direction choice and emitted token diverged")
        if not trace and not target_tokens:
            raise RuntimeError("empty target continuation produced no trace")

        job_dir = self.output / _safe_name(str(job["id"]))
        job_dir.mkdir(parents=False, exist_ok=False)
        shadows = (
            torch.stack(shadow_coordinates)
            if shadow_coordinates
            else torch.empty((0, len(POLICIES), 1000))
        )
        full_index = POLICIES.index("full")
        policy_index = POLICIES.index(policy)
        tensors = {
            "schema": "readout_direction.trace_tensors.v1",
            "job_id": job["id"],
            "policy": policy,
            "policies": list(POLICIES),
            "coordinate_ids": self.ids.detach().cpu(),
            "head_inputs": (
                torch.stack(head_inputs)
                if head_inputs
                else torch.empty((0, self.U.shape[1]))
            ),
            "raw_coordinate_logits": (
                torch.stack(raw_coordinates)
                if raw_coordinates
                else torch.empty((0, 1000))
            ),
            "full_coordinate_logits": shadows[:, full_index],
            "operator_coordinate_logits": shadows,
            "applied_coordinate_logits": shadows[:, policy_index],
            "target_tokens": torch.tensor(target_tokens, dtype=torch.long),
        }
        tensor_path = job_dir / "trajectory.pt"
        torch.save(tensors, tensor_path)
        self._record_artifact(tensor_path)
        persisted = torch.load(tensor_path, map_location="cpu", weights_only=False)
        persistence = {
            "path": _binding(tensor_path),
            "readback_schema": persisted.get("schema"),
            "head_inputs_sha256": _tensor_hash(persisted["head_inputs"]),
            "raw_coordinate_logits_sha256": _tensor_hash(persisted["raw_coordinate_logits"]),
            "operator_coordinate_logits_sha256": _tensor_hash(
                persisted["operator_coordinate_logits"]
            ),
            "applied_coordinate_logits_sha256": _tensor_hash(
                persisted["applied_coordinate_logits"]
            ),
            "target_tokens_sha256": _tensor_hash(persisted["target_tokens"]),
            "readback_passed": bool(
                persisted["target_tokens"].tolist() == target_tokens
                and persisted["policy"] == policy
                and persisted["policies"] == list(POLICIES)
            ),
        }
        if not persistence["readback_passed"]:
            raise RuntimeError("direction trace tensor persistence/readback mismatch")
        payload = {
            "schema": "readout_direction.job_result.v1",
            "status": "candidate_complete",
            "job_id": job["id"],
            "operator": policy,
            "policy": policy,
            "model": self.model_name,
            "group": boundary["group"],
            "boundary_id": boundary["id"],
            "image_id": int(boundary["image_id"]),
            "batch_index": target_index,
            "direction": {
                "formula": self.ledger["direction_formula"],
                "coordinate_ids": [COORD, COORD + 999],
                "sign_vector": (
                    None
                    if policy not in SIGN_POLICIES
                    else self.ledger["sign_vector_hashes"][policy]
                ),
            },
            "source": {
                "source_end_exclusive": source_end,
                "native_token_hash": _token_hash(native),
                "prefix_token_hash": _token_hash(prefix),
                "target_prefix_unpadded": True,
                "source_row": boundary["source_row"],
                "target_slots": boundary.get("target_slots", []),
            },
            "target": {
                "token_ids": target_tokens,
                "token_hash": _token_hash(target_tokens),
                "text": self.q.tokenizer.decode(
                    target_tokens,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
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
                    (
                        index
                        for index, step in enumerate(trace)
                        if step["emitted_token"]
                        != step["operators"]["original"]["chosen_token"]
                    ),
                    None,
                ),
            },
            "persistence": persistence,
            "qualification": qualification,
            "source_receipt": _binding(component._source_path(boundary, "receipt")),
        }
        result_path = job_dir / "release.json"
        write(result_path, payload)
        receipt_path = job_dir / "receipt.json"
        cell_receipt = {
            "schema": "readout_direction.cell_receipt.v1",
            "status": payload["status"],
            "job_id": job["id"],
            "boundary_id": boundary["id"],
            "policy": policy,
            "model": self.model_name,
            "group": boundary["group"],
            "producer": _binding(Path(__file__)),
            "panel": _binding(self.args.panel),
            "plan": _binding(self.plan_path),
            "sign_vectors": _binding(self.direction_sign_manifest_path),
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
                "operator": policy,
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


def selfcheck() -> None:
    generator = torch.Generator().manual_seed(19)
    z = torch.randn(3, 1000, generator=generator, dtype=torch.float32).double()
    alpha = torch.rand(1000, generator=generator, dtype=torch.float64) * 0.1 + 0.95
    delta = (alpha - 1)[None, :] * z
    full = z * alpha[None, :]
    reflected = z - delta
    signs = {
        policy: torch.where(
            torch.arange(1000) % 2 == 0,
            torch.ones(1000, dtype=torch.float64),
            -torch.ones(1000, dtype=torch.float64),
        )
        for policy in SIGN_POLICIES
    }
    coordinates = {"full": full, "reflected": reflected}
    coordinates.update({policy: z + signs[policy][None, :] * delta for policy in SIGN_POLICIES})
    assert torch.equal(full, z * alpha[None, :])
    for value in coordinates.values():
        assert torch.allclose((value - z).abs(), delta.abs(), atol=1e-12, rtol=0)
    scores = torch.randn(3, 2500, generator=generator, dtype=torch.float32)
    ids = torch.arange(1000) + 1000
    transformed = scores.clone()
    transformed[:, ids] = coordinates["full"].float()
    mask = torch.ones(scores.shape[-1], dtype=torch.bool)
    mask[ids] = False
    assert torch.equal(scores[:, mask], transformed[:, mask])
    corrupted = transformed.clone()
    corrupted[0, ids[0]] += 0.01
    assert not torch.equal(corrupted, transformed)
    fixture = [151646, 9, 151647, 151648, COORD, COORD + 1, COORD + 2, COORD + 3, 151649] * 3
    assert len(parsed_rows(fixture)) == 3
    print("PASS direction formulas, exact noncoordinate identity, corruption sensitivity, and row parser")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["run", "validate-plan"], nargs="?")
    parser.add_argument("--panel", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validation-output", type=Path)
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
    if args.mode == "validate-plan":
        if args.plan is None or args.panel is None:
            parser.error("validate-plan requires --panel and --plan")
        report = _validate_plan(args.plan, args.panel)
        destination = args.validation_output or args.plan.parent / "runtime" / "plan-validation.json"
        write(destination, report)
        print(json.dumps({"status": report["status"], "cells": report["cells"]}))
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
