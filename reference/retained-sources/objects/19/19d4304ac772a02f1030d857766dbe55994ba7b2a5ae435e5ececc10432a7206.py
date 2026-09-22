"""Static coordinate-support ablations for the frozen sign23 readout.

The native batch construction, source replay, generation, row stopping, and
artifact persistence come from the qualified component runtime.  This probe
only changes which coordinate columns receive the already frozen sign23
increment.  It never forces a token or edits an input/cache state.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
from transformers import LogitsProcessor, StoppingCriteria

from probes.training_set_completion.first_coordinate import runtime as base


component = base.component
EOS = component.EOS
COORD = component.COORD
RELEASE_ROWS = component.RELEASE_ROWS
RELEASE_TOKENS = component.RELEASE_TOKENS
POLICIES = ("original", "sign23", "only0", "only1", "pair01", "except01")
SIGN_POLICY = "sign23"
EXPECTED_SUPPORTS = {
    "original": [],
    "sign23": list(range(1000)),
    "only0": [0],
    "only1": [1],
    "pair01": [0, 1],
    "except01": list(range(2, 1000)),
}
ROLE_BY_ROW_OFFSET = {
    0: "row_start",
    1: "description",
    2: "description",
    3: "row_separator",
    4: "box_start",
    5: "x1",
    6: "y1",
    7: "x2",
    8: "y2",
    9: "box_end",
}


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


def _source_release(path: Path, boundary_id: str, model: str, policy: str) -> dict[str, Any]:
    release = json.loads(path.read_text())
    if release.get("status") != "candidate_complete":
        raise ValueError(f"saved control is not complete: {path}")
    if (
        release.get("boundary_id") != boundary_id
        or release.get("model") != model
        or release.get("policy") != policy
    ):
        raise ValueError(f"saved control identity changed: {path}")
    tokens = release.get("target", {}).get("token_ids")
    if not isinstance(tokens, list) or not tokens:
        raise ValueError(f"saved control has no target tokens: {path}")
    return {"binding": _binding(path), "target_tokens": [int(token) for token in tokens]}


def _sign23_manifest(plan: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    spec = plan.get("sign_vectors")
    if not isinstance(spec, dict) or "path" not in spec:
        raise ValueError("coordinate-pair plan has no sign-vector binding")
    path = Path(spec["path"])
    if _binding(path)["sha256"] != spec.get("sha256"):
        raise ValueError("sign-vector binding changed")
    manifest = json.loads(path.read_text())
    if manifest.get("coordinate_ids") != list(range(COORD, COORD + 1000)):
        raise ValueError("sign-vector coordinate IDs changed")
    vectors = {item.get("policy"): item for item in manifest.get("vectors", [])}
    item = vectors.get(SIGN_POLICY)
    if item is None:
        raise ValueError("frozen sign23 vector is missing")
    signs = item.get("signs")
    if (
        not isinstance(signs, list)
        or len(signs) != 1000
        or any(value not in (-1, 1) for value in signs)
        or signs.count(1) != 500
        or signs.count(-1) != 500
        or base._hash_signs(signs) != item.get("sha256")
    ):
        raise ValueError("invalid frozen sign23 vector")
    return path, manifest


def _selection_from_plan(plan: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    return base._selection_from_plan(plan)


def _validate_plan(plan_path: Path, panel_path: Path) -> dict[str, Any]:
    """Validate the full 66-cell plan without loading a model."""
    plan = json.loads(plan_path.read_text())
    if plan.get("status") != "frozen_before_gpu":
        raise ValueError("coordinate-pair plan is not frozen_before_gpu")
    if plan.get("policies") != list(POLICIES):
        raise ValueError("coordinate-pair policy order changed")
    if plan.get("supports") != EXPECTED_SUPPORTS:
        raise ValueError("coordinate-support declarations changed")
    bounds = plan.get("bounds", {})
    if int(bounds.get("target_continuations", -1)) != 66:
        raise ValueError("coordinate-pair target count changed")
    if int(bounds.get("release_tokens", -1)) != RELEASE_TOKENS:
        raise ValueError("coordinate-pair token cap changed")
    if int(bounds.get("release_rows", -1)) != RELEASE_ROWS:
        raise ValueError("coordinate-pair row cap changed")
    sign_path, manifest = _sign23_manifest(plan)
    selection_path, selection = _selection_from_plan(plan)
    boundaries = selection.get("boundaries", [])
    if len(boundaries) != 11:
        raise ValueError(f"expected 11 boundaries, found {len(boundaries)}")
    boundary_by_id = {str(item.get("id")): item for item in boundaries}
    if len(boundary_by_id) != 11 or any(item.get("kind") != "failure" for item in boundaries):
        raise ValueError("selection must contain eleven unique failure boundaries")
    for boundary in boundaries:
        for key in ("raw_path", "trace_path", "receipt_path"):
            if not Path(boundary[key]).exists():
                raise FileNotFoundError(boundary[key])
    panel_binding = next(
        (item for item in plan.get("bindings", []) if item.get("path") == str(panel_path)),
        None,
    )
    if panel_binding is not None and _binding(panel_path)["sha256"] != panel_binding.get("sha256"):
        raise ValueError("panel binding changed")
    cells = plan.get("cells", [])
    if len(cells) != 66:
        raise ValueError(f"expected 66 cells, found {len(cells)}")
    seen: set[str] = set()
    coverage: set[tuple[str, str]] = set()
    qualifications: set[str] = set()
    controls = 0
    for cell in cells:
        cell_id = str(cell.get("id", ""))
        boundary_id = str(cell.get("boundary_id", ""))
        policy = str(cell.get("policy", ""))
        if not cell_id or cell_id in seen:
            raise ValueError("cell IDs must be unique and nonempty")
        if boundary_id not in boundary_by_id or policy not in POLICIES:
            raise ValueError(f"invalid cell identity: {cell_id}")
        if cell.get("model") != boundary_by_id[boundary_id].get("model"):
            raise ValueError(f"cell model differs from boundary: {cell_id}")
        seen.add(cell_id)
        coverage.add((boundary_id, policy))
        control = cell.get("control_source")
        if policy in {"original", SIGN_POLICY}:
            if not isinstance(control, dict) or "path" not in control or "sha256" not in control:
                raise ValueError(f"missing saved control source: {cell_id}")
            control_path = Path(control["path"])
            if _binding(control_path)["sha256"] != control["sha256"]:
                raise ValueError(f"control binding changed: {cell_id}")
            _source_release(control_path, boundary_id, str(cell["model"]), policy)
            controls += 1
        elif control is not None:
            raise ValueError(f"unsupported control source on ablation cell: {cell_id}")
        if cell.get("qualification"):
            qualifications.add(cell_id)
    expected_coverage = {
        (boundary_id, policy) for boundary_id in boundary_by_id for policy in POLICIES
    }
    if coverage != expected_coverage:
        raise ValueError("each boundary must have exactly six policies")
    expected_qualification = {
        f"untied-417044-failure--{policy}" for policy in POLICIES
    }
    if qualifications != expected_qualification:
        raise ValueError("qualification cell set changed")
    if controls != 22:
        raise ValueError(f"expected 22 saved original/sign23 controls, found {controls}")
    return {
        "schema": "coordinate_pair.plan_validation.v1",
        "status": "candidate_cpu_validated",
        "plan": _binding(plan_path),
        "panel": _binding(panel_path),
        "selection": _binding(selection_path),
        "sign_vectors": _binding(sign_path),
        "boundaries": len(boundaries),
        "cells": len(cells),
        "policies": list(POLICIES),
        "supports": {policy: list(indices) for policy, indices in EXPECTED_SUPPORTS.items()},
        "qualification_cells": sorted(qualifications),
        "saved_controls": controls,
        "sign23_hash": next(item["sha256"] for item in manifest["vectors"] if item["policy"] == SIGN_POLICY),
    }


class Runtime(base.Runtime):
    """Qualified native runtime with six static coordinate-support operators."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.pair_plan = json.loads(args.plan.read_text())
        _validate_plan(args.plan, args.panel)
        self.pair_sign_manifest_path, self.pair_sign_manifest = _sign23_manifest(self.pair_plan)
        self.pair_policies = tuple(self.pair_plan["policies"])
        component.Runtime.__init__(self, args)
        vector = next(item for item in self.pair_sign_manifest["vectors"] if item["policy"] == SIGN_POLICY)
        self.sign23 = torch.tensor(vector["signs"], dtype=torch.float64, device=args.device)
        self.supports = {
            policy: tuple(int(index) for index in self.pair_plan["supports"][policy])
            for policy in POLICIES
        }
        self.ledger.update(
            {
                "schema": "coordinate_pair.runtime_receipt.v1",
                "operators": list(self.pair_policies),
                "support_formula": "z + mask*d; d=sign23*(alpha-1)*z in FP64 then cast once",
                "support_indices": {policy: list(indices) for policy, indices in self.supports.items()},
                "sign_vectors": _binding(self.pair_sign_manifest_path),
                "sign23_hash": vector["sha256"],
                "coordinate_ids": [COORD, COORD + 999],
                "noncoordinate_bitwise_unchanged": True,
                "all_rows_free": True,
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
                raise ValueError("coordinate-pair cell IDs must be unique and nonempty")
            if policy not in POLICIES:
                raise ValueError(f"unsupported coordinate-pair policy: {policy}")
            boundary = job.get("boundary") or boundaries.get(boundary_id)
            if boundary is None or str(boundary.get("id")) != boundary_id:
                raise ValueError(f"cell has no matching boundary: {job_id}")
            if job.get("model") != boundary.get("model"):
                raise ValueError(f"cell model mismatch: {job_id}")
            job["boundary"] = boundary
            job["pair_policy"] = policy
            job["operator"] = "original"
            seen.add(job_id)
            jobs.append(job)
        if not jobs:
            raise ValueError("coordinate-pair plan has no selected cells")
        return jobs

    def _pair_scores(
        self, scores: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        z = scores[:, self.ids].double()
        delta = self.sign23[None, :] * (self.alpha - 1)[None, :] * z
        transformed: dict[str, torch.Tensor] = {"original": scores}
        for policy in POLICIES[1:]:
            mask = torch.zeros(1000, dtype=torch.bool, device=scores.device)
            mask[list(self.supports[policy])] = True
            coordinate = z + torch.where(mask[None, :], delta, torch.zeros_like(delta))
            value = scores.clone()
            value[:, self.ids] = coordinate.to(scores.dtype)
            transformed[policy] = value
        return transformed, z, delta

    def _qualification_pair(
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
        coords = {policy: transformed[policy][:, self.ids].double() for policy in POLICIES}
        mask = {
            policy: torch.tensor(self.supports[policy], dtype=torch.long, device=scores.device)
            for policy in POLICIES
        }
        support_errors = {
            policy: float(
                (coords[policy][:, mask[policy]] - (z[:, mask[policy]] + delta[:, mask[policy]])).abs().max().item()
            )
            if len(self.supports[policy]) else float((coords[policy] - z).abs().max().item())
            for policy in POLICIES
        }
        reconstruction = float(
            (
                coords["only0"]
                + coords["only1"]
                - z
                - coords["pair01"]
            ).abs().max().item()
        )
        full_reconstruction = float(
            (coords["pair01"] + coords["except01"] - z - coords["sign23"]).abs().max().item()
        )
        return {
            "identity_operator_bitwise": bool(torch.equal(transformed["original"], scores)),
            "noncoordinate_bitwise_unchanged": unchanged,
            "support_formula_max_abs_error_fp64": support_errors,
            "singleton_sum_to_pair_max_abs_error_fp64": reconstruction,
            "pair_plus_except_reconstructs_sign23_max_abs_error_fp64": full_reconstruction,
            "max_sign23_coordinate_sensitivity": float(delta.abs().max().item()),
        }

    @staticmethod
    def _position(
        boundary: dict[str, Any],
        source_end: int,
        offset: int,
        generated_prefix: list[int],
    ) -> dict[str, Any]:
        absolute = source_end + offset
        rows = base.parsed_rows(generated_prefix)
        row_index = len(rows)
        row_offset: int | None = None
        role = "unknown"
        for row in rows:
            if int(row["start"]) <= offset < int(row["end"]):
                row_index = int(row["index"])
                row_offset = offset - int(row["start"])
                desc_end = int(row["start"]) + 1 + len(row["description_tokens"])
                if row_offset == 0:
                    role = "row_start"
                elif row_offset < desc_end - int(row["start"]):
                    role = "description"
                elif row_offset == desc_end - int(row["start"]):
                    role = "row_separator"
                elif row_offset == desc_end - int(row["start"]) + 1:
                    role = "box_start"
                elif row_offset == int(row["coordinate_offsets"][0]) - int(row["start"]):
                    role = "x1"
                elif row_offset == int(row["coordinate_offsets"][1]) - int(row["start"]):
                    role = "y1"
                elif row_offset == int(row["coordinate_offsets"][2]) - int(row["start"]):
                    role = "x2"
                elif row_offset == int(row["coordinate_offsets"][3]) - int(row["start"]):
                    role = "y2"
                elif row_offset == int(row["end"]) - int(row["start"]) - 1:
                    role = "box_end"
                break
        if row_offset is None:
            row_start = int(rows[-1]["end"]) if rows else 0
            row_index = len(rows)
            partial = generated_prefix[row_start:]
            row_offset = offset - row_start
            if rows:
                description_len = len(rows[-1]["description_tokens"])
            else:
                slots = boundary.get("target_slots", [])
                first_x1 = next(
                    (
                        int(slot["offset"]) - source_end
                        for slot in slots
                        if slot.get("row_delay") == 0 and slot.get("role") == "x1"
                    ),
                    None,
                )
                description_len = max(0, first_x1 - 3) if first_x1 is not None else None
            if description_len is not None:
                if row_offset == 0:
                    role = "row_start"
                elif row_offset <= description_len:
                    role = "description"
                elif row_offset == description_len + 1:
                    role = "row_separator"
                elif row_offset == description_len + 2:
                    role = "box_start"
                elif row_offset == description_len + 3:
                    role = "x1"
                elif row_offset == description_len + 4:
                    role = "y1"
                elif row_offset == description_len + 5:
                    role = "x2"
                elif row_offset == description_len + 6:
                    role = "y2"
                elif row_offset == description_len + 7:
                    role = "box_end"
            if role == "unknown" and row_offset:
                try:
                    separator = partial.index(151647)
                except ValueError:
                    separator = None
                if separator is None or row_offset < separator:
                    role = "description"
                elif row_offset == separator:
                    role = "row_separator"
                elif row_offset == separator + 1:
                    role = "box_start"
                elif row_offset == separator + 2:
                    role = "x1"
                elif row_offset == separator + 3:
                    role = "y1"
                elif row_offset == separator + 4:
                    role = "x2"
                elif row_offset == separator + 5:
                    role = "y2"
                elif row_offset == separator + 6:
                    role = "box_end"
        return {
            "continuation_offset": offset,
            "absolute_native_offset": absolute,
            "row_index": row_index,
            "row_offset": row_offset,
            "role": role,
            "target_slots": boundary.get("target_slots", []),
        }

    def _run_job(self, job: dict[str, Any]) -> dict[str, Any]:
        forwards_before = int(self.ledger["model_forwards"])
        vision_before = int(self.ledger["vision_forwards"])
        seconds_before = time.monotonic() - self.started
        boundary = job["boundary"]
        policy = job["pair_policy"]
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
        qualification: dict[str, Any] | None = None
        runtime = self

        class PairPolicy(LogitsProcessor):
            def __call__(_self, tokens: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
                nonlocal qualification
                head = runtime.last.get("head_input")
                if head is None or head.ndim != 2 or scores.ndim != 2:
                    raise RuntimeError("missing aligned lm-head input at generation step")
                transformed, z, delta = runtime._pair_scores(scores)
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
                target_trace = {
                    "offset": offset,
                    "input_token_id": int(tokens[target_index, -1].item()),
                    "position": runtime._position(
                        boundary,
                        source_end,
                        offset,
                        tokens[target_index, width:].tolist(),
                    ),
                    "raw_winner_token": raw_winner,
                    "raw_winner_family": raw_family,
                    "raw_top2": component.Runtime._trace_operator(
                        runtime, "original", scores[target_index], z[target_index], raw_winner
                    ),
                    "operators": operator_trace,
                    "applied_operator": policy,
                    "operator_winner_token": choices[policy],
                    "chosen_token": choices[policy],
                    "emission_seam": "native_greedy_argmax",
                    "operator_dtype": str(scores.dtype),
                    "coordinate_delta_abs_max": float(delta[target_index].abs().max().item()),
                }
                trace.append(target_trace)
                head_inputs.append(head[target_index].detach().cpu())
                raw_coordinates.append(target_raw.cpu())
                shadow_coordinates.append(target_shadows.cpu())
                applied_coordinates.append(target_shadows[POLICIES.index(policy)].cpu())
                if job.get("qualification") and qualification is None:
                    qualification = runtime._qualification_pair(scores, transformed, z, delta)
                return used

        stop_state: dict[str, Any] = {"complete_rows": 0, "eos": False}

        class TargetStop(StoppingCriteria):
            def __call__(_self, ids: torch.Tensor, _scores: Any, **_kwargs: Any) -> bool:
                target = ids[target_index, width:].tolist()
                stop_state["complete_rows"] = len(base.parsed_rows(target))
                stop_state["eos"] = EOS in target
                return bool(stop_state["eos"] or stop_state["complete_rows"] >= RELEASE_ROWS)

        result = self.generate(inputs, processors=[PairPolicy()], stopping=[TargetStop()])
        target_tokens, stop = _trim_target(result[target_index, width:].tolist())
        if len(target_tokens) > RELEASE_TOKENS:
            raise RuntimeError("target continuation exceeded token cap")
        complete = len(base.parsed_rows(target_tokens))
        if complete > RELEASE_ROWS:
            raise RuntimeError("target continuation exceeded row cap")
        if len(trace) != len(target_tokens):
            raise RuntimeError("trace step count differs from emitted target tokens")
        for index, step in enumerate(trace):
            if step["offset"] != index:
                raise RuntimeError("generation trace offsets are not contiguous")
            step["emitted_token"] = int(target_tokens[index])
            step["choice_matches_emitted"] = (
                step["operator_winner_token"] == step["emitted_token"]
            )
        if any(not step["choice_matches_emitted"] for step in trace):
            raise RuntimeError("operator choice and emitted token diverged")
        if not trace and not target_tokens:
            raise RuntimeError("empty target continuation produced no trace")

        control = job.get("control_source")
        control_equal: bool | None = None
        control_tokens: list[int] | None = None
        if control is not None:
            control_path = Path(control["path"])
            if _binding(control_path)["sha256"] != control["sha256"]:
                raise ValueError("control source binding changed during run")
            source = _source_release(control_path, boundary["id"], self.model_name, policy)
            control_tokens = source["target_tokens"]
            control_equal = target_tokens == control_tokens
            if not control_equal:
                raise base.NeedsLead(f"saved {policy} control diverged: {job['id']}")

        job_dir = self.output / _safe_name(str(job["id"]))
        job_dir.mkdir(parents=False, exist_ok=False)
        shadows = torch.stack(shadow_coordinates)
        tensors = {
            "schema": "coordinate_pair.trace_tensors.v1",
            "job_id": job["id"],
            "policy": policy,
            "policies": list(POLICIES),
            "support_indices": {name: list(self.supports[name]) for name in POLICIES},
            "coordinate_ids": self.ids.detach().cpu(),
            "head_inputs": torch.stack(head_inputs),
            "raw_coordinate_logits": torch.stack(raw_coordinates),
            "operator_coordinate_logits": shadows,
            "applied_coordinate_logits": torch.stack(applied_coordinates),
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
            "operator_coordinate_logits_sha256": _tensor_hash(persisted["operator_coordinate_logits"]),
            "applied_coordinate_logits_sha256": _tensor_hash(persisted["applied_coordinate_logits"]),
            "target_tokens_sha256": _tensor_hash(persisted["target_tokens"]),
            "readback_passed": bool(
                persisted["target_tokens"].tolist() == target_tokens
                and persisted["policy"] == policy
                and persisted["policies"] == list(POLICIES)
            ),
        }
        if not persistence["readback_passed"]:
            raise RuntimeError("coordinate-pair tensor persistence/readback mismatch")
        payload = {
            "schema": "coordinate_pair.job_result.v1",
            "status": "candidate_complete",
            "job_id": job["id"],
            "operator": policy,
            "policy": policy,
            "model": self.model_name,
            "group": boundary["group"],
            "boundary_id": boundary["id"],
            "image_id": int(boundary["image_id"]),
            "batch_index": target_index,
            "support": {
                "sign_policy": SIGN_POLICY,
                "indices": list(self.supports[policy]),
                "formula": "z+mask*(sign23*(alpha-1)*z) in FP64 then cast once",
                "sign_vector": self.ledger["sign23_hash"],
            },
            "source": {
                "source_end_exclusive": source_end,
                "native_token_hash": _token_hash(native),
                "prefix_token_hash": _token_hash(prefix),
                "target_prefix_unpadded": True,
                "source_row": boundary["source_row"],
                "target_slots": boundary.get("target_slots", []),
                "control_source": control,
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
                "control_equal": control_equal,
                "control_token_hash": None if control_tokens is None else _token_hash(control_tokens),
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
            "schema": "coordinate_pair.cell_receipt.v1",
            "status": payload["status"],
            "job_id": job["id"],
            "boundary_id": boundary["id"],
            "policy": policy,
            "model": self.model_name,
            "group": boundary["group"],
            "qualification": bool(job.get("qualification", False)),
            "control_source": control,
            "control_equal": control_equal,
            "producer": _binding(Path(__file__)),
            "panel": _binding(self.args.panel),
            "plan": _binding(self.plan_path),
            "sign_vectors": _binding(self.pair_sign_manifest_path),
            "source_receipt": payload["source_receipt"],
            "release": _binding(result_path),
            "trajectory": persistence["path"],
            "model_forwards": int(self.ledger["model_forwards"]) - forwards_before,
            "vision_forwards": int(self.ledger["vision_forwards"]) - vision_before,
            "tensor_bytes": persistence["path"]["size_bytes"],
            "gpu_seconds": (time.monotonic() - self.started) - seconds_before,
            "stop": payload["target"]["stop"],
            "support_indices": list(self.supports[policy]),
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
                "control_equal": control_equal,
                "model_forwards_after_job": self.ledger["model_forwards"],
                "stop": payload["target"]["stop"],
            }
        )
        self.persist()
        return payload


def selfcheck() -> None:
    generator = torch.Generator().manual_seed(43)
    z = torch.randn(3, 1000, generator=generator, dtype=torch.float64)
    delta = torch.randn(3, 1000, generator=generator, dtype=torch.float64)
    values = {"original": z}
    for policy, indices in EXPECTED_SUPPORTS.items():
        if policy == "original":
            continue
        mask = torch.zeros(1000, dtype=torch.bool)
        mask[indices] = True
        values[policy] = z + torch.where(mask[None, :], delta, torch.zeros_like(delta))
    assert torch.equal(values["original"][:, 2:], z[:, 2:])
    assert torch.equal(values["only0"][:, 1:], z[:, 1:])
    assert torch.equal(values["only1"][:, 0], z[:, 0])
    assert torch.allclose(values["only0"] + values["only1"] - z, values["pair01"], atol=1e-12, rtol=0)
    assert torch.allclose(values["pair01"] + values["except01"] - z, values["sign23"], atol=1e-12, rtol=0)
    wrong_support = values["only0"].clone()
    wrong_support[:, 1] = wrong_support[:, 1] + delta[:, 1]
    assert not torch.equal(wrong_support, values["only0"])
    scores = torch.randn(2, COORD + 1000, generator=generator, dtype=torch.float32)
    changed = scores.clone(); changed[:, COORD:] = values["pair01"][:2].to(scores.dtype)
    noncoord = torch.ones(scores.shape[-1], dtype=torch.bool); noncoord[COORD:] = False
    assert torch.equal(scores[:, noncoord], changed[:, noncoord])
    fixture = [151646, 9, 151647, 151648, COORD, COORD + 1, COORD + 2, COORD + 3, 151649] * 3
    assert len(base.parsed_rows(fixture)) == 3
    print("PASS six support formulas, reconstruction, noncoordinate identity, and row parser")


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
    parser.add_argument("--max-bytes", type=int, default=12 * 1024**3)
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
