"""One-coordinate mediation continuations on the frozen native panel.

This probe reuses the qualified component runtime for input construction and
native generation.  Its only intervention is a one-step target-row score
carrier at continuation offset five.  The raw model scores, the declared
readout operator winner, and the carrier winner are recorded separately so an
emitted forced coordinate cannot be mistaken for an operator prediction.
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

from probes.training_set_completion.numerical_feedback.select import rows as parsed_rows
from probes.training_set_completion.readout_component import runtime as component


EOS = component.EOS
COORD = component.COORD
RELEASE_ROWS = component.RELEASE_ROWS
RELEASE_TOKENS = component.RELEASE_TOKENS
FORCED_OFFSET = 5
EXPECTED_PRELUDE = (151646, 15007, 332, 151647, 151648)
POLICIES = (
    "original",
    "sign19",
    "sign20",
    "sign21",
    "sign22",
    "sign23",
    "sign24",
    "sign25",
    "sign26",
)
SIGN_POLICIES = POLICIES[1:]


class NeedsLead(RuntimeError):
    """A frozen input or semantic witness needs the owning lead's ruling."""


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _binding(path: Path) -> dict[str, Any]:
    return component._binding(path)


def _token_hash(tokens: list[int]) -> str:
    return component.token_hash(tokens)


def _tensor_hash(value: torch.Tensor) -> str:
    return component.tensor_hash(value)


def _safe_name(value: str) -> str:
    return component._safe_name(value)


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
        raise ValueError("first-coordinate plan has no selection binding")
    actual = _binding(path)
    if binding is not None and any(
        actual[key] != binding[key] for key in ("path", "sha256") if key in binding
    ):
        raise ValueError("selection binding changed")
    return path, json.loads(path.read_text())


def _sign_manifest(plan: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    spec = plan.get("sign_vectors")
    if not isinstance(spec, dict) or "path" not in spec:
        raise ValueError("first-coordinate plan has no sign-vector binding")
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


def _check_source_release(
    path: Path,
    *,
    boundary_id: str,
    model: str,
    policy: str,
    native_choice: int,
) -> dict[str, Any]:
    binding = _binding(path)
    release = json.loads(path.read_text())
    if release.get("status") != "candidate_complete":
        raise ValueError(f"saved source is not complete: {path}")
    if (
        release.get("boundary_id") != boundary_id
        or release.get("model") != model
        or release.get("policy") != policy
    ):
        raise ValueError(f"saved source identity changed: {path}")
    tokens = release.get("target", {}).get("token_ids")
    if not isinstance(tokens, list) or len(tokens) <= FORCED_OFFSET:
        raise ValueError(f"saved source has no target continuation: {path}")
    if tuple(tokens[: len(EXPECTED_PRELUDE)]) != EXPECTED_PRELUDE:
        raise ValueError(f"saved source prelude changed: {path}")
    expected = COORD + native_choice
    if int(tokens[FORCED_OFFSET]) != expected:
        raise ValueError(f"saved source native choice changed: {path}")
    return {"binding": binding, "target_token_hash": _token_hash([int(v) for v in tokens])}


def _validate_plan(plan_path: Path, panel_path: Path) -> dict[str, Any]:
    """Validate the frozen mediation plan and all immutable source bindings."""
    plan = json.loads(plan_path.read_text())
    if plan.get("status") != "frozen_before_gpu":
        raise ValueError("first-coordinate plan is not frozen_before_gpu")
    if plan.get("policies") != list(POLICIES):
        raise ValueError("first-coordinate policy order changed")
    if tuple(plan.get("expected_prelude", [])) != EXPECTED_PRELUDE:
        raise ValueError("expected prelude changed")
    bounds = plan.get("bounds", {})
    if int(bounds.get("target_continuations", -1)) != 36:
        raise ValueError("first-coordinate target count changed")
    if int(bounds.get("release_tokens", -1)) != RELEASE_TOKENS:
        raise ValueError("first-coordinate token cap changed")
    if int(bounds.get("release_rows", -1)) != RELEASE_ROWS:
        raise ValueError("first-coordinate row cap changed")
    sign_path, manifest = _sign_manifest(plan)
    selection_path, selection = _selection_from_plan(plan)
    boundaries = selection.get("boundaries", [])
    if len(boundaries) != 2:
        raise ValueError(f"expected two boundaries, found {len(boundaries)}")
    boundary_by_id = {str(item.get("id")): item for item in boundaries}
    expected_boundary_ids = {"untied-417044-failure", "tied-417044-failure"}
    if set(boundary_by_id) != expected_boundary_ids:
        raise ValueError("first-coordinate boundary set changed")
    for boundary_id, boundary in boundary_by_id.items():
        if boundary.get("kind") != "failure":
            raise ValueError(f"boundary is not a failure: {boundary_id}")
        source_end = int(boundary["source_row"]["end"])
        native = [int(value) for value in boundary["native_tokens"]]
        if not 0 < source_end <= len(native):
            raise ValueError(f"invalid source end: {boundary_id}")
        if native[source_end : source_end + len(EXPECTED_PRELUDE)] != list(EXPECTED_PRELUDE):
            raise ValueError(f"native target prelude changed: {boundary_id}")
        slots = boundary.get("target_slots", [])
        if not slots or int(slots[0]["offset"]) != source_end + FORCED_OFFSET:
            raise ValueError(f"first target x1 slot changed: {boundary_id}")
        for key in ("raw_path", "trace_path", "receipt_path"):
            if not Path(boundary[key]).exists():
                raise FileNotFoundError(boundary[key])
    source_binding = selection.get("source")
    if not isinstance(source_binding, dict):
        raise ValueError("selection has no predecessor binding")
    if _binding(Path(source_binding["path"]))["sha256"] != source_binding.get("sha256"):
        raise ValueError("predecessor selection binding changed")

    panel_binding = next(
        (item for item in plan.get("bindings", []) if item.get("path") == str(panel_path)),
        None,
    )
    if panel_binding is not None and _binding(panel_path)["sha256"] != panel_binding.get("sha256"):
        raise ValueError("panel binding changed")

    cells = plan.get("cells", [])
    if len(cells) != 36:
        raise ValueError(f"expected 36 cells, found {len(cells)}")
    seen: set[str] = set()
    coverage: set[tuple[str, str, int]] = set()
    qualification: set[str] = set()
    control_count = 0
    source_count = 0
    for cell in cells:
        cell_id = str(cell.get("id", ""))
        if not cell_id or cell_id in seen:
            raise ValueError("cell IDs must be unique and nonempty")
        seen.add(cell_id)
        boundary_id = str(cell.get("boundary_id", ""))
        policy = str(cell.get("policy", ""))
        choice = cell.get("forced_choice")
        if boundary_id not in boundary_by_id or policy not in POLICIES:
            raise ValueError(f"invalid cell identity: {cell_id}")
        if cell.get("model") != boundary_by_id[boundary_id].get("model"):
            raise ValueError(f"cell model differs from boundary: {cell_id}")
        if choice not in (0, 1) or int(cell.get("forced_offset", -1)) != FORCED_OFFSET:
            raise ValueError(f"invalid forced coordinate declaration: {cell_id}")
        if int(cell.get("forced_token", -1)) != COORD + int(choice):
            raise ValueError(f"forced token does not match choice: {cell_id}")
        coverage.add((boundary_id, policy, int(choice)))
        source = cell.get("source_policy")
        if not isinstance(source, dict) or "path" not in source or "sha256" not in source:
            raise ValueError(f"missing source policy binding: {cell_id}")
        source_path = Path(source["path"])
        if _binding(source_path)["sha256"] != source["sha256"]:
            raise ValueError(f"source policy binding changed: {cell_id}")
        _check_source_release(
            source_path,
            boundary_id=boundary_id,
            model=str(cell["model"]),
            policy=policy,
            native_choice=int(cell["native_choice"]),
        )
        source_count += 1
        if cell.get("concordant"):
            control = cell.get("control_source")
            if not isinstance(control, dict):
                raise ValueError(f"missing concordant control source: {cell_id}")
            control_path = Path(control["path"])
            if _binding(control_path)["sha256"] != control.get("sha256"):
                raise ValueError(f"control source binding changed: {cell_id}")
            _check_source_release(
                control_path,
                boundary_id=boundary_id,
                model=str(cell["model"]),
                policy=policy,
                native_choice=int(cell["native_choice"]),
            )
            control_count += 1
        elif cell.get("control_source") is not None:
            raise ValueError(f"nonconcordant cell has a control source: {cell_id}")
        if cell.get("qualification"):
            qualification.add(cell_id)

    expected_coverage = {
        (boundary_id, policy, choice)
        for boundary_id in expected_boundary_ids
        for policy in POLICIES
        for choice in (0, 1)
    }
    if coverage != expected_coverage:
        raise ValueError("cell coverage is not exactly two choices for every policy")
    expected_qualification = {
        "untied-417044-failure--original--x1-0",
        "untied-417044-failure--original--x1-1",
        "untied-417044-failure--sign20--x1-0",
        "untied-417044-failure--sign20--x1-1",
    }
    if qualification != expected_qualification:
        raise ValueError("qualification cell set changed")
    if source_count != 36 or control_count != 18:
        raise ValueError(f"unexpected source/control counts: {source_count}/{control_count}")
    return {
        "schema": "first_coordinate.plan_validation.v1",
        "status": "candidate_cpu_validated",
        "plan": _binding(plan_path),
        "panel": _binding(panel_path),
        "selection": _binding(selection_path),
        "sign_vectors": _binding(sign_path),
        "boundaries": sorted(boundary_by_id),
        "cells": len(cells),
        "policies": list(POLICIES),
        "qualification_cells": sorted(qualification),
        "saved_controls": control_count,
        "source_policy_bindings": source_count,
        "expected_prelude": list(EXPECTED_PRELUDE),
        "sign_vector_hashes": {
            policy: next(item["sha256"] for item in manifest["vectors"] if item["policy"] == policy)
            for policy in SIGN_POLICIES
        },
    }


class Runtime(component.Runtime):
    """Native continuation runtime with one explicit target score carrier."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.mediation_plan = json.loads(args.plan.read_text())
        _validate_plan(args.plan, args.panel)
        self.mediation_sign_manifest_path, self.mediation_sign_manifest = _sign_manifest(
            self.mediation_plan
        )
        self.mediation_policies = tuple(self.mediation_plan["policies"])
        super().__init__(args)
        self.mediation_signs = {
            item["policy"]: torch.tensor(item["signs"], dtype=torch.float64, device=args.device)
            for item in self.mediation_sign_manifest["vectors"]
        }
        self.ledger.update(
            {
                "schema": "first_coordinate_mediation.runtime_receipt.v1",
                "operators": list(self.mediation_policies),
                "mediation": {
                    "formula": "z+s*(alpha-1)*z in FP64 then cast once",
                    "original": "raw model scores copied without coordinate assignment",
                    "noncoordinate": "scores copied bitwise unchanged",
                    "forced_offset": FORCED_OFFSET,
                    "override_seam": "target_only_score_carrier",
                    "carrier_is_not_operator": True,
                },
                "sign_vectors": _binding(self.mediation_sign_manifest_path),
                "sign_vector_hashes": {
                    item["policy"]: item["sha256"]
                    for item in self.mediation_sign_manifest["vectors"]
                },
                "coordinate_ids": [COORD, COORD + 999],
                "expected_prelude": list(EXPECTED_PRELUDE),
                "alpha_min": float(self.alpha.min().item()),
                "alpha_max": float(self.alpha.max().item()),
            }
        )
        self.persist()

    def _normalise_jobs(self, plan: dict[str, Any]) -> list[dict[str, Any]]:
        _selection_path, selection = _selection_from_plan(plan)
        planned = plan.get("cells") or plan.get("jobs") or []
        requested = set(self.args.cell_ids or [])
        if requested:
            planned = [item for item in planned if item.get("id") in requested]
            present = {item.get("id") for item in planned}
            if present != requested:
                raise ValueError(f"requested cell IDs are absent: {sorted(requested - present)}")
        boundaries = {str(item["id"]): item for item in selection["boundaries"]}
        jobs: list[dict[str, Any]] = []
        seen: set[str] = set()
        for raw in planned:
            job = dict(raw)
            job_id = str(job.get("id", ""))
            boundary_id = str(job.get("boundary_id", ""))
            policy = str(job.get("policy", ""))
            if not job_id or job_id in seen:
                raise ValueError("first-coordinate cell IDs must be unique and nonempty")
            if policy not in POLICIES:
                raise ValueError(f"unsupported first-coordinate policy: {policy}")
            boundary = job.get("boundary") or boundaries.get(boundary_id)
            if boundary is None or str(boundary.get("id")) != boundary_id:
                raise ValueError(f"cell has no matching boundary: {job_id}")
            if job.get("model") != boundary.get("model"):
                raise ValueError(f"cell model mismatch: {job_id}")
            if int(job.get("forced_offset", -1)) != FORCED_OFFSET:
                raise ValueError(f"cell forced offset mismatch: {job_id}")
            if int(job.get("forced_token", -1)) != COORD + int(job["forced_choice"]):
                raise ValueError(f"cell forced token mismatch: {job_id}")
            job["boundary"] = boundary
            job["mediation_policy"] = policy
            # The base constructor only accepts its original four operator names.
            job["operator"] = "original"
            seen.add(job_id)
            jobs.append(job)
        if not jobs:
            raise ValueError("first-coordinate plan has no selected cells")
        return jobs

    def _direction_scores(
        self, scores: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Compute the nine declared shadows from raw coordinate logits."""
        z = scores[:, self.ids].double()
        delta = (self.alpha - 1)[None, :] * z
        transformed: dict[str, torch.Tensor] = {"original": scores}
        for policy in SIGN_POLICIES:
            coordinates = z + self.mediation_signs[policy][None, :] * delta
            value = scores.clone()
            value[:, self.ids] = coordinates.to(scores.dtype)
            transformed[policy] = value
        return transformed, z, delta

    def _qualification_mediation(
        self,
        scores: torch.Tensor,
        transformed: dict[str, torch.Tensor],
        z: torch.Tensor,
        delta: torch.Tensor,
    ) -> dict[str, Any]:
        noncoord = torch.ones(scores.shape[-1], dtype=torch.bool, device=scores.device)
        noncoord[self.ids] = False
        unchanged = {
            policy: bool(torch.equal(transformed[policy][:, noncoord], scores[:, noncoord]))
            for policy in POLICIES
        }
        magnitude_error = max(
            float(
                (
                    (z + self.mediation_signs[policy][None, :] * delta - z).abs()
                    - delta.abs()
                ).abs().max().item()
            )
            for policy in SIGN_POLICIES
        )
        return {
            "identity_operator_bitwise": bool(torch.equal(transformed["original"], scores)),
            "noncoordinate_bitwise_unchanged": unchanged,
            "direction_magnitude_max_abs_error_fp64": magnitude_error,
            "sign_balance": {
                policy: {
                    "positive": int((self.mediation_signs[policy] == 1).sum().item()),
                    "negative": int((self.mediation_signs[policy] == -1).sum().item()),
                }
                for policy in SIGN_POLICIES
            },
            "max_coordinate_sensitivity": float(delta.abs().max().item()),
        }

    def _run_job(self, job: dict[str, Any]) -> dict[str, Any]:
        forwards_before = int(self.ledger["model_forwards"])
        vision_before = int(self.ledger["vision_forwards"])
        seconds_before = time.monotonic() - self.started
        boundary = job["boundary"]
        policy = job["mediation_policy"]
        forced_choice = int(job["forced_choice"])
        forced_offset = int(job["forced_offset"])
        forced_token = int(job["forced_token"])
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
        applied_coordinates: list[torch.Tensor] = []
        carrier_coordinates: list[torch.Tensor] = []
        override_flags: list[bool] = []
        qualification: dict[str, Any] | None = None
        runtime = self

        class Mediation(LogitsProcessor):
            def __call__(_self, tokens: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
                nonlocal qualification
                head = runtime.last.get("head_input")
                if head is None or head.ndim != 2 or scores.ndim != 2:
                    raise RuntimeError("missing aligned lm-head input at generation step")
                transformed, z, delta = runtime._direction_scores(scores)
                used = transformed[policy]
                choices = {
                    name: int(transformed[name][target_index].argmax().item())
                    for name in POLICIES
                }
                offset = int(tokens.shape[1] - width)
                raw_winner = int(scores[target_index].argmax().item())
                raw_family = _token_family(raw_winner)
                target_raw = scores[target_index, runtime.ids].detach()
                target_shadows = torch.stack(
                    [transformed[name][target_index, runtime.ids].detach() for name in POLICIES]
                )
                operator_trace: dict[str, dict[str, Any]] = {}
                for index, name in enumerate(POLICIES):
                    item = component.Runtime._trace_operator(
                        runtime,
                        name,
                        transformed[name][target_index],
                        target_shadows[index],
                        choices[name],
                    )
                    winner = int(item["top2"][0]["token_id"])
                    item["operator_winner_token"] = choices[name]
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

                carrier = used.clone()
                override_applied = offset == forced_offset
                if override_applied:
                    carrier[target_index] = torch.full_like(
                        carrier[target_index], torch.finfo(scores.dtype).min
                    )
                    carrier[target_index, forced_token] = 0.0
                carrier_winner = int(carrier[target_index].argmax().item())
                if override_applied and carrier_winner != forced_token:
                    raise RuntimeError("forced score carrier did not produce the forced token")
                carrier_coordinate = carrier[target_index, runtime.ids].detach()
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
                    "operator_winner_token": choices[policy],
                    "carrier_winner_token": carrier_winner,
                    "carrier_override_applied": override_applied,
                    "override_offset": forced_offset if override_applied else None,
                    "forced_choice": forced_choice,
                    "forced_token": forced_token,
                    "operator_dtype": str(scores.dtype),
                    "coordinate_delta_abs_max": float(delta[target_index].abs().max().item()),
                    "carrier_top2": component.Runtime._trace_operator(
                        runtime,
                        "carrier",
                        carrier[target_index],
                        carrier_coordinate,
                        carrier_winner,
                    ),
                }
                trace.append(target_trace)
                head_inputs.append(head[target_index].detach().cpu())
                raw_coordinates.append(target_raw.cpu())
                shadow_coordinates.append(target_shadows.cpu())
                applied_coordinates.append(target_shadows[POLICIES.index(policy)].cpu())
                carrier_coordinates.append(carrier_coordinate.cpu())
                override_flags.append(override_applied)
                if job.get("qualification") and qualification is None:
                    qualification = runtime._qualification_mediation(scores, transformed, z, delta)
                return carrier

        stop_state: dict[str, Any] = {"complete_rows": 0, "eos": False}

        class TargetStop(StoppingCriteria):
            def __call__(_self, ids: torch.Tensor, _scores: Any, **_kwargs: Any) -> bool:
                target = ids[target_index, width:].tolist()
                stop_state["complete_rows"] = len(parsed_rows(target))
                stop_state["eos"] = EOS in target
                return bool(
                    stop_state["eos"] or stop_state["complete_rows"] >= RELEASE_ROWS
                )

        result = self.generate(inputs, processors=[Mediation()], stopping=[TargetStop()])
        target_tokens, stop = _trim_target(result[target_index, width:].tolist())
        if len(target_tokens) > RELEASE_TOKENS:
            raise RuntimeError("target continuation exceeded token cap")
        complete = len(parsed_rows(target_tokens))
        if complete > RELEASE_ROWS:
            raise RuntimeError("target continuation exceeded row cap")
        if tuple(target_tokens[: len(EXPECTED_PRELUDE)]) != EXPECTED_PRELUDE:
            raise NeedsLead("target prelude changed before the forced coordinate")
        if len(target_tokens) <= forced_offset or target_tokens[forced_offset] != forced_token:
            raise NeedsLead("forced coordinate was not consumed at the declared offset")
        if len(trace) != len(target_tokens):
            raise RuntimeError("trace step count differs from emitted target tokens")
        for index, step in enumerate(trace):
            if step["offset"] != index:
                raise RuntimeError("generation trace offsets are not contiguous")
            step["emitted_token"] = int(target_tokens[index])
            step["choice_matches_emitted"] = (
                step["carrier_winner_token"] == step["emitted_token"]
            )
        if any(not step["choice_matches_emitted"] for step in trace):
            raise RuntimeError("carrier winner and emitted token diverged")
        override_steps = [step for step in trace if step["carrier_override_applied"]]
        if len(override_steps) != 1 or override_steps[0]["offset"] != forced_offset:
            raise RuntimeError("target-only override count or offset changed")
        for step in trace:
            if step["offset"] != forced_offset and step["carrier_override_applied"]:
                raise RuntimeError("override persisted beyond offset five")
        if len(trace) <= forced_offset + 1:
            raise NeedsLead("forced token has no following native-cache consumption witness")
        if trace[forced_offset + 1]["input_token_id"] != forced_token:
            raise NeedsLead("next-step input did not consume the forced token")
        trace[forced_offset]["next_step_input_token_id"] = trace[forced_offset + 1]["input_token_id"]
        for index, step in enumerate(trace):
            if index != forced_offset:
                step["next_step_input_token_id"] = (
                    trace[index + 1]["input_token_id"] if index + 1 < len(trace) else None
                )
        if not trace:
            raise RuntimeError("empty target continuation produced no trace")

        job_dir = self.output / _safe_name(str(job["id"]))
        job_dir.mkdir(parents=False, exist_ok=False)
        shadows = torch.stack(shadow_coordinates)
        tensors = {
            "schema": "first_coordinate_mediation.trace_tensors.v1",
            "job_id": job["id"],
            "policy": policy,
            "policies": list(POLICIES),
            "coordinate_ids": self.ids.detach().cpu(),
            "forced_choice": forced_choice,
            "forced_offset": forced_offset,
            "forced_token": forced_token,
            "head_inputs": torch.stack(head_inputs),
            "raw_coordinate_logits": torch.stack(raw_coordinates),
            "operator_coordinate_logits": shadows,
            "applied_coordinate_logits": torch.stack(applied_coordinates),
            "carrier_coordinate_logits": torch.stack(carrier_coordinates),
            "override_flags": torch.tensor(override_flags, dtype=torch.bool),
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
            "carrier_coordinate_logits_sha256": _tensor_hash(
                persisted["carrier_coordinate_logits"]
            ),
            "override_flags_sha256": _tensor_hash(persisted["override_flags"]),
            "target_tokens_sha256": _tensor_hash(persisted["target_tokens"]),
            "readback_passed": bool(
                persisted["target_tokens"].tolist() == target_tokens
                and persisted["policy"] == policy
                and persisted["policies"] == list(POLICIES)
                and int(persisted["forced_token"]) == forced_token
            ),
        }
        if not persistence["readback_passed"]:
            raise RuntimeError("first-coordinate tensor persistence/readback mismatch")

        source_policy = job["source_policy"]
        control_source = job.get("control_source")
        payload = {
            "schema": "first_coordinate_mediation.job_result.v1",
            "status": "candidate_complete",
            "job_id": job["id"],
            "operator": policy,
            "policy": policy,
            "model": self.model_name,
            "group": boundary["group"],
            "boundary_id": boundary["id"],
            "image_id": int(boundary["image_id"]),
            "batch_index": target_index,
            "mediation": {
                "forced_choice": forced_choice,
                "forced_offset": forced_offset,
                "forced_token": forced_token,
                "native_choice": int(job["native_choice"]),
                "override_seam": "target_only_score_carrier",
                "raw_scores_untouched": True,
                "operator_winner_preserved": True,
                "carrier_is_emitted_choice": True,
                "no_subsequent_override": True,
                "expected_prelude": list(EXPECTED_PRELUDE),
                "sign_vector": (
                    None
                    if policy == "original"
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
                "source_policy": source_policy,
                "control_source": control_source,
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
                "forced_step": trace[forced_offset],
                "first_divergence_source": next(
                    (
                        index
                        for index, step in enumerate(trace)
                        if step["emitted_token"]
                        != step["operators"]["original"]["operator_winner_token"]
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
            "schema": "first_coordinate_mediation.cell_receipt.v1",
            "status": payload["status"],
            "job_id": job["id"],
            "boundary_id": boundary["id"],
            "policy": policy,
            "model": self.model_name,
            "group": boundary["group"],
            "forced_choice": forced_choice,
            "forced_offset": forced_offset,
            "forced_token": forced_token,
            "native_choice": int(job["native_choice"]),
            "concordant": bool(job.get("concordant", False)),
            "qualification": bool(job.get("qualification", False)),
            "producer": _binding(Path(__file__)),
            "panel": _binding(self.args.panel),
            "plan": _binding(self.plan_path),
            "sign_vectors": _binding(self.mediation_sign_manifest_path),
            "source_policy": source_policy,
            "control_source": control_source,
            "source_receipt": payload["source_receipt"],
            "release": _binding(result_path),
            "trajectory": persistence["path"],
            "model_forwards": int(self.ledger["model_forwards"]) - forwards_before,
            "vision_forwards": int(self.ledger["vision_forwards"]) - vision_before,
            "tensor_bytes": persistence["path"]["size_bytes"],
            "gpu_seconds": (time.monotonic() - self.started) - seconds_before,
            "stop": payload["target"]["stop"],
            "forced_consumption_witness": {
                "offset": forced_offset,
                "next_step_input_token_id": trace[forced_offset]["next_step_input_token_id"],
            },
            "qualification_evidence": qualification,
        }
        write(receipt_path, cell_receipt)
        self.ledger["artifacts"].append(_binding(result_path))
        self.ledger["artifacts"].append(_binding(receipt_path))
        self.ledger["job_results"].append(
            {
                "job_id": job["id"],
                "operator": policy,
                "boundary_id": boundary["id"],
                "status": payload["status"],
                "release": _binding(result_path),
                "trace": persistence["path"],
                "forced_choice": forced_choice,
                "forced_token": forced_token,
                "model_forwards_after_job": self.ledger["model_forwards"],
                "stop": payload["target"]["stop"],
            }
        )
        self.persist()
        return payload

    def close(self, error: BaseException | None = None) -> None:
        super().close(error)
        if isinstance(error, NeedsLead):
            self.ledger["status"] = "needs_lead"
            self.ledger["needs_lead"] = True
            self.persist()


def _carrier_selfcheck() -> None:
    generator = torch.Generator().manual_seed(29)
    scores = torch.randn(2, COORD + 1000, generator=generator, dtype=torch.float32)
    ids = torch.arange(1000) + COORD
    operator = scores.clone()
    operator[:, ids] = torch.randn(2, 1000, generator=generator, dtype=torch.float32)
    raw = scores.clone()
    target = 1
    forced = COORD + 1
    carrier = operator.clone()
    carrier[target] = torch.finfo(scores.dtype).min
    carrier[target, forced] = 0.0
    assert torch.equal(scores, raw)
    assert torch.equal(operator[0], carrier[0])
    assert int(carrier[target].argmax()) == forced
    assert not torch.equal(operator[target], carrier[target])
    assert torch.equal(scores[:, :COORD], raw[:, :COORD])
    assert torch.equal(scores[:, COORD + 1000 :], raw[:, COORD + 1000 :])
    input_after = torch.tensor([151648, int(carrier[target].argmax())])
    assert int(input_after[-1]) == forced
    assert len(parsed_rows(list(EXPECTED_PRELUDE) + [forced, COORD + 2, COORD + 3, COORD + 4, 151649])) == 1
    print("PASS FP64-shadow seam, target-only carrier, forced-token consumption, and row parser")


def selfcheck() -> None:
    generator = torch.Generator().manual_seed(31)
    z = torch.randn(3, 1000, generator=generator, dtype=torch.float32).double()
    alpha = torch.rand(1000, generator=generator, dtype=torch.float64) * 0.1 + 0.95
    delta = (alpha - 1)[None, :] * z
    signs = {
        policy: torch.where(
            torch.arange(1000) % 2 == 0,
            torch.ones(1000, dtype=torch.float64),
            -torch.ones(1000, dtype=torch.float64),
        )
        for policy in SIGN_POLICIES
    }
    for policy in SIGN_POLICIES:
        value = z + signs[policy][None, :] * delta
        assert torch.allclose((value - z).abs(), delta.abs(), atol=1e-12, rtol=0)
    scores = torch.randn(3, COORD + 1000, generator=generator, dtype=torch.float32)
    transformed = scores.clone()
    transformed[:, COORD:] = (z + delta).to(scores.dtype)
    noncoord = torch.ones(scores.shape[-1], dtype=torch.bool)
    noncoord[COORD:] = False
    assert torch.equal(scores[:, noncoord], transformed[:, noncoord])
    _carrier_selfcheck()
    print("PASS direction formulas, exact noncoordinate identity, and mediation carrier checks")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["run", "validate-plan"], nargs="?")
    parser.add_argument("--panel", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validation-output", type=Path)
    parser.add_argument("--cell-id", action="append", dest="cell_ids")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-forwards", type=int, default=50000)
    parser.add_argument("--max-seconds", type=float, default=4 * 3600)
    parser.add_argument("--max-bytes", type=int, default=8 * 1024**3)
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
        print(json.dumps({"status": report["status"], "cells": report["cells"], "controls": report["saved_controls"]}))
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
